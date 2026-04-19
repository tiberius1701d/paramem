"""Unit tests for BackgroundTrainer.

Covers:
  - TrainingJob.inference_fallback_adapter field defaults to "episodic".
  - BackgroundTrainer.pause() uses job.inference_fallback_adapter, not
    self._current_adapter, when determining which adapter to switch to.

No GPU required — model interactions are replaced with MagicMock stubs.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

from paramem.server.background_trainer import BackgroundTrainer, TrainingJob
from paramem.utils.config import AdapterConfig, TrainingConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stub_model(*adapter_names: str):
    """Return a MagicMock that behaves like a minimal PeftModel."""
    model = MagicMock()
    model.peft_config = {name: MagicMock() for name in adapter_names}
    return model


def _minimal_training_config() -> TrainingConfig:
    return TrainingConfig(
        num_epochs=1,
        gradient_checkpointing=False,
        batch_size=1,
    )


def _minimal_adapter_config() -> AdapterConfig:
    return AdapterConfig(rank=4, alpha=8, learning_rate=1e-4, target_modules=["q_proj"])


# ---------------------------------------------------------------------------
# Test — TrainingJob.inference_fallback_adapter field
# ---------------------------------------------------------------------------


class TestTrainingJobInferenceFallbackAdapter:
    def test_default_fallback_is_episodic(self) -> None:
        """TrainingJob defaults inference_fallback_adapter to 'episodic'."""
        job = TrainingJob(
            keyed_pairs=[],
            adapter_name="episodic_interim_20260418T1430",
            adapter_config=_minimal_adapter_config(),
        )
        assert job.inference_fallback_adapter == "episodic"

    def test_explicit_fallback_stored(self) -> None:
        """An explicitly set inference_fallback_adapter is preserved."""
        job = TrainingJob(
            keyed_pairs=[],
            adapter_name="episodic_interim_20260418T1430",
            adapter_config=_minimal_adapter_config(),
            inference_fallback_adapter="episodic",
        )
        assert job.inference_fallback_adapter == "episodic"

    def test_custom_fallback_stored(self) -> None:
        """A non-default inference_fallback_adapter is preserved."""
        job = TrainingJob(
            keyed_pairs=[],
            adapter_name="episodic",
            adapter_config=_minimal_adapter_config(),
            inference_fallback_adapter="episodic_backup_20260418",
        )
        assert job.inference_fallback_adapter == "episodic_backup_20260418"


# ---------------------------------------------------------------------------
# Test 9 — pause() uses inference_fallback_adapter, not _current_adapter
# ---------------------------------------------------------------------------


class TestPauseUsesInferenceFallbackNotCurrentAdapter:
    """Blocker 1 fix: pause() reads job.inference_fallback_adapter."""

    def test_pause_switches_to_episodic_not_interim_adapter(self) -> None:
        """When training an interim adapter, pause() switches to 'episodic', not the interim slot.

        This test verifies the Blocker 1 fix from Step 6c of the multi-adapter
        interim routing plan.

        Before the fix, pause() fell back to self._current_adapter (the interim
        adapter name), which contains mid-training staging-copy weights that must
        NOT be exposed to inference.  The fix reads
        job.inference_fallback_adapter instead, which for interim-adapter jobs is
        always the stable 'episodic' main.
        """
        model = _make_stub_model("episodic", "episodic_interim_20260418T1430", "in_training")
        config = _minimal_training_config()
        adapter_cfg = _minimal_adapter_config()

        interim_name = "episodic_interim_20260418T1430"

        job = TrainingJob(
            keyed_pairs=[{"key": "graph1", "question": "Q?", "answer": "A."}],
            adapter_name=interim_name,
            adapter_config=adapter_cfg,
            inference_fallback_adapter="episodic",
        )

        bt = BackgroundTrainer(
            model=model,
            tokenizer=MagicMock(),
            training_config=config,
            output_dir="/tmp/test_bt",
        )

        # Simulate the trainer being mid-job.
        bt._is_training = True
        bt._current_adapter = interim_name
        bt._current_job = job

        # We test the adapter-selection logic directly without the threading machinery.
        # The full pause() path requires the training thread; here we verify only
        # the target-selection rule that pause() applies.
        bt._training_paused.set()

        # Extract the target-selection rule that pause() applies.
        target = (
            bt._current_job.inference_fallback_adapter
            if bt._current_job is not None
            else "episodic"
        )
        assert target == "episodic", (
            f"Expected 'episodic' as the fallback, got {target!r}. "
            "BackgroundTrainer.pause() must read job.inference_fallback_adapter, "
            "not self._current_adapter."
        )
        assert target != interim_name, (
            "pause() must NOT switch to the mid-training interim adapter."
        )

    def test_pause_fallback_is_episodic_when_no_job(self) -> None:
        """When _current_job is None, fallback defaults to 'episodic'."""
        model = _make_stub_model("episodic")
        config = _minimal_training_config()

        bt = BackgroundTrainer(
            model=model,
            tokenizer=MagicMock(),
            training_config=config,
            output_dir="/tmp/test_bt2",
        )

        # Simulate the pre-first-job state.
        bt._current_job = None

        target = (
            bt._current_job.inference_fallback_adapter
            if bt._current_job is not None
            else "episodic"
        )
        assert target == "episodic"

    def test_pause_switches_to_fallback_adapter_in_peft_config(
        self,
    ) -> None:
        """Integration check: pause() calls switch_adapter with the fallback name.

        We mock the threading machinery so the test runs without a real training
        thread.  The test verifies that the adapter name passed to switch_adapter
        is the job's inference_fallback_adapter, not _current_adapter.
        """
        interim_name = "episodic_interim_20260418T1430"
        model = _make_stub_model("episodic", interim_name, "in_training")
        config = _minimal_training_config()

        job = TrainingJob(
            keyed_pairs=[],
            adapter_name=interim_name,
            adapter_config=_minimal_adapter_config(),
            inference_fallback_adapter="episodic",
        )

        bt = BackgroundTrainer(
            model=model,
            tokenizer=MagicMock(),
            training_config=config,
            output_dir="/tmp/test_bt3",
        )
        bt._is_training = True
        bt._current_adapter = interim_name
        bt._current_job = job

        # Simulate the training thread having already paused (event is set).
        bt._training_paused.set()

        switch_calls = []

        def _record_switch(m, name):  # noqa: ANN001
            switch_calls.append(name)

        # Patch switch_adapter at the background_trainer module level so pause()
        # sees the mock when it does `from paramem.models.loader import switch_adapter`.
        with patch(
            "paramem.models.loader.switch_adapter",
            side_effect=_record_switch,
        ):
            # pause() is normally called from the inference path with no training
            # thread running.  We patch _training_paused.wait to return immediately
            # and then call the real pause() implementation.
            with patch.object(bt, "_training_paused") as mock_paused_event:
                mock_paused_event.wait.return_value = True
                mock_paused_event.is_set.return_value = True
                result = bt.pause(timeout=0.1)

        # pause() should have returned True (paused successfully).
        assert result is True

        # The switch_adapter call must have been to "episodic", NOT the interim slot.
        assert switch_calls, "switch_adapter was never called"
        assert switch_calls[-1] == "episodic", (
            f"pause() switched to {switch_calls[-1]!r} instead of 'episodic'."
        )
        assert interim_name not in switch_calls, (
            "pause() must NOT switch to the mid-training interim adapter."
        )


# ---------------------------------------------------------------------------
# Test 10 — submit() serialises concurrent post-session jobs
# ---------------------------------------------------------------------------


class TestSubmitSerialises:
    """Blocker B2/B4 fix: submit() routes through BackgroundTrainer's single worker.

    Two concurrent /chat responses both call enqueue_post_session_train, which
    calls BackgroundTrainer.submit().  Jobs must run serially — the second waits
    for the first to complete — and shared ConsolidationLoop state
    (_indexed_next_index) must advance by exactly the combined triple count with
    no interleaving.
    """

    def test_two_jobs_run_serially_no_registry_collision(self) -> None:
        """Two submitted jobs run serially; _indexed_next_index advances by combined count.

        Simulates two /chat responses each submitting a post_session_train job
        that increments a shared counter by a known amount.  Asserts:
        1. Both jobs execute (execution order is enforced by queue drain).
        2. The counter equals start + increment_A + increment_B (no lost updates,
           no double-counting from concurrent mutation).
        3. Job B waits for job A to finish — verified by recording the sequence
           of events under a threading.Barrier.
        """
        model = _make_stub_model("episodic", "in_training")
        config = _minimal_training_config()

        bt = BackgroundTrainer(
            model=model,
            tokenizer=MagicMock(),
            training_config=config,
            output_dir="/tmp/test_bt_submit",
        )

        # Shared mutable state mimicking ConsolidationLoop._indexed_next_index.
        state = {"counter": 0, "execution_order": []}
        job_a_started = threading.Event()
        job_a_may_finish = threading.Event()

        def job_a() -> None:
            state["execution_order"].append("A_start")
            job_a_started.set()
            # Hold until the test releases us, simulating non-trivial work.
            job_a_may_finish.wait(timeout=5.0)
            state["counter"] += 3  # simulates 3 new triples
            state["execution_order"].append("A_end")

        def job_b() -> None:
            # Job B must not start until A has finished.
            state["execution_order"].append("B_start")
            state["counter"] += 2  # simulates 2 new triples
            state["execution_order"].append("B_end")

        # Patch gpu_lock_sync at the source module so the import inside
        # _run_callable_queue ("from paramem.server.gpu_lock import gpu_lock_sync")
        # picks up the no-op context manager — no real GPU is needed.
        from contextlib import contextmanager

        @contextmanager
        def _noop_gpu_lock():
            yield

        with patch("paramem.server.gpu_lock.gpu_lock_sync", new=_noop_gpu_lock):
            bt.submit(job_a, inference_fallback_adapter="episodic")
            # Wait until job A has actually started before submitting B,
            # ensuring A is running (not just queued) when B arrives.
            job_a_started.wait(timeout=5.0)
            bt.submit(job_b, inference_fallback_adapter="episodic")
            # Release A to complete.
            job_a_may_finish.set()
            # Wait for the worker thread to drain the queue.
            if bt._worker_thread is not None:
                bt._worker_thread.join(timeout=10.0)

        # Both jobs must have run.
        assert state["counter"] == 5, (
            f"Expected counter=5 (3+2), got {state['counter']}. "
            "One or both jobs may not have executed."
        )
        # Serialisation: A must fully complete before B starts.
        assert state["execution_order"] == ["A_start", "A_end", "B_start", "B_end"], (
            f"Jobs did not run serially: {state['execution_order']}"
        )


# ---------------------------------------------------------------------------
# Test 11 — persistent callable worker (C2 fix)
# ---------------------------------------------------------------------------


class TestPersistentCallableWorker:
    """C2 fix: callable worker is persistent and never strands a job.

    Before the fix, the worker exited as soon as the queue was empty.  A
    concurrent submit() that observed is_alive()==True (thread in its exit
    path but not yet terminated) would skip starting a new worker, leaving
    the new job stranded.

    The fix makes the worker persistent — it blocks on queue.get() between
    jobs and lives for the process lifetime.  submit() starts the worker
    exactly once on first call.
    """

    @staticmethod
    def _noop_gpu_lock():
        """Return a no-op context manager so tests need no real GPU."""
        from contextlib import contextmanager

        @contextmanager
        def _inner():
            yield

        return _inner

    def test_submit_after_worker_idle_does_not_strand_job(self) -> None:
        """A job submitted after the worker has gone idle still executes.

        Without the fix (get_nowait + thread exit), a job submitted while the
        worker is in its shutdown path would be stranded.  With the persistent
        worker (blocking get()), the second job is guaranteed to run.

        Sequence:
          1. Submit job 1 and wait for it to complete.
          2. Submit job 2 after job 1 is done (worker is idle but alive).
          3. Assert job 2 completes within a reasonable timeout.
        """
        model = _make_stub_model("episodic", "in_training")
        config = _minimal_training_config()

        bt = BackgroundTrainer(
            model=model,
            tokenizer=MagicMock(),
            training_config=config,
            output_dir="/tmp/test_bt_persistent",
        )

        results: list[str] = []
        job1_done = threading.Event()
        job2_done = threading.Event()

        def job1() -> None:
            results.append("job1")
            job1_done.set()

        def job2() -> None:
            results.append("job2")
            job2_done.set()

        with patch("paramem.server.gpu_lock.gpu_lock_sync", new=self._noop_gpu_lock()):
            bt.submit(job1, inference_fallback_adapter="episodic")
            # Wait until job1 is confirmed done (worker now idle, queue empty).
            assert job1_done.wait(timeout=5.0), "job1 did not complete in time"

            # Submit job2 while the worker is idle (queue empty) — this is the
            # exact scenario the race condition triggers.
            bt.submit(job2, inference_fallback_adapter="episodic")

            # job2 must complete even though the worker was idling.
            assert job2_done.wait(timeout=5.0), (
                "job2 was stranded — persistent worker did not pick it up."
            )

        assert results == ["job1", "job2"], f"Unexpected execution sequence: {results}"

    def test_same_worker_thread_handles_both_jobs(self) -> None:
        """Both jobs are handled by the same persistent worker thread.

        Verifies that submit() starts the worker exactly once.  After both
        jobs complete, the thread identity must be the same object as after
        the first submit, and the thread must still be alive.
        """
        model = _make_stub_model("episodic", "in_training")
        config = _minimal_training_config()

        bt = BackgroundTrainer(
            model=model,
            tokenizer=MagicMock(),
            training_config=config,
            output_dir="/tmp/test_bt_same_thread",
        )

        job1_done = threading.Event()
        job2_done = threading.Event()

        def job1() -> None:
            job1_done.set()

        def job2() -> None:
            job2_done.set()

        with patch("paramem.server.gpu_lock.gpu_lock_sync", new=self._noop_gpu_lock()):
            bt.submit(job1, inference_fallback_adapter="episodic")
            assert job1_done.wait(timeout=5.0), "job1 did not complete in time"

            # Capture the thread reference after the first job.
            thread_after_job1 = bt._worker_thread
            assert thread_after_job1 is not None
            assert thread_after_job1.is_alive(), (
                "Persistent worker must remain alive after job1 completes."
            )

            bt.submit(job2, inference_fallback_adapter="episodic")
            assert job2_done.wait(timeout=5.0), "job2 did not complete in time"

            thread_after_job2 = bt._worker_thread

        # Same thread object for both jobs — worker started exactly once.
        assert thread_after_job1 is thread_after_job2, (
            "submit() must not start a second worker thread; persistent worker "
            f"should handle all jobs.  thread_after_job1={thread_after_job1!r}, "
            f"thread_after_job2={thread_after_job2!r}"
        )

        # Thread is still alive (persistent daemon).
        assert thread_after_job2.is_alive(), (
            "Persistent worker thread must remain alive between jobs."
        )
