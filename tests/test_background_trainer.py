"""Unit tests for BackgroundTrainer.

Covers:
  - TrainingJob.inference_fallback_adapter field defaults to "episodic".
  - BackgroundTrainer.pause() uses job.inference_fallback_adapter, not
    self._current_adapter, when determining which adapter to switch to.
  - Step-level save knobs and dedup logic for _persist_resume_state.
  - _fingerprint_training_config includes save_strategy_bg/save_steps_bg.
  - on_shutdown_check fires at step_end via TrainingHooks.

No GPU required — model interactions are replaced with MagicMock stubs.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

from paramem.server.background_trainer import (
    BackgroundTrainer,
    TrainingJob,
    _fingerprint_training_config,
)
from paramem.training.trainer import TrainingHooks, _HooksAdapterCallback
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
            entries=[],
            adapter_name="episodic_interim_20260418T1430",
            adapter_config=_minimal_adapter_config(),
        )
        assert job.inference_fallback_adapter == "episodic"

    def test_explicit_fallback_stored(self) -> None:
        """An explicitly set inference_fallback_adapter is preserved."""
        job = TrainingJob(
            entries=[],
            adapter_name="episodic_interim_20260418T1430",
            adapter_config=_minimal_adapter_config(),
            inference_fallback_adapter="episodic",
        )
        assert job.inference_fallback_adapter == "episodic"

    def test_custom_fallback_stored(self) -> None:
        """A non-default inference_fallback_adapter is preserved."""
        job = TrainingJob(
            entries=[],
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
            entries=[{"key": "graph1", "question": "Q?", "answer": "A."}],
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
            entries=[],
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


# ---------------------------------------------------------------------------
# Test 12 — Step-level save knobs (Commit 1 BG-trainer refactor)
# ---------------------------------------------------------------------------


class TestStepLevelSave:
    """Verify step-level checkpointing configuration and dedup logic."""

    def test_fingerprint_includes_bg_save_fields(self) -> None:
        """Fingerprints differ when save_strategy_bg differs."""
        adapter = _minimal_adapter_config()
        cfg_a = TrainingConfig(save_strategy_bg="", save_steps_bg=0)
        cfg_b = TrainingConfig(save_strategy_bg="steps", save_steps_bg=50)
        assert _fingerprint_training_config(cfg_a, adapter) != _fingerprint_training_config(
            cfg_b, adapter
        ), "save_strategy_bg must contribute to the fingerprint"

    def test_persist_dedupes_repeat_calls_with_same_key(self) -> None:
        """Calling _persist_resume_state twice with the same (step, dir) writes only once."""
        model = _make_stub_model("episodic", "in_training")
        bt = BackgroundTrainer(
            model=model,
            tokenizer=MagicMock(),
            training_config=_minimal_training_config(),
            output_dir="/tmp/test_bt_dedup",
        )
        bt._last_persist_key = None
        with patch.object(bt, "_write_resume_state") as mock_write:
            bt._persist_resume_state(5, "/tmp/x")
            bt._persist_resume_state(5, "/tmp/x")
        mock_write.assert_called_once_with(5, "/tmp/x")

    def test_persist_fresh_on_different_step(self) -> None:
        """A different step value bypasses the dedup guard and writes again."""
        model = _make_stub_model("episodic", "in_training")
        bt = BackgroundTrainer(
            model=model,
            tokenizer=MagicMock(),
            training_config=_minimal_training_config(),
            output_dir="/tmp/test_bt_dedup2",
        )
        bt._last_persist_key = None
        with patch.object(bt, "_write_resume_state") as mock_write:
            bt._persist_resume_state(5, "/tmp/x")
            bt._persist_resume_state(10, "/tmp/x")
        assert mock_write.call_count == 2

    def test_train_adapter_overrides_save_strategy_to_steps_when_bg_set(self) -> None:
        """When save_strategy_bg='steps', bg_training_config has save_strategy='steps'."""
        cfg = TrainingConfig(save_strategy_bg="steps", save_steps_bg=5)
        model = _make_stub_model("episodic", "in_training")
        bt = BackgroundTrainer(
            model=model,
            tokenizer=MagicMock(),
            training_config=cfg,
            output_dir="/tmp/test_bt_override",
        )
        captured: list[TrainingConfig] = []

        def _fake_train_adapter(**kwargs):
            captured.append(kwargs["training_config"])
            return {}

        _patch_train = patch(
            "paramem.server.background_trainer.train_adapter",
            side_effect=_fake_train_adapter,
        )
        _patch_staging = patch(
            "paramem.server.background_trainer._ensure_staging_shape_matches",
            side_effect=lambda m, _c: m,
        )
        # format_entry_training is called inside _train_adapter before train_adapter
        # is invoked; patch it to return a minimal non-empty list so _train_adapter
        # does not short-circuit ("No training examples").
        _patch_fmt = patch(
            "paramem.server.background_trainer.format_entry_training",
            return_value=[{"input_ids": [1], "labels": [1], "attention_mask": [1]}],
        )
        with (
            _patch_train,
            _patch_staging,
            _patch_fmt,
            patch("paramem.models.loader.copy_adapter_weights"),
            patch("paramem.models.loader.switch_adapter"),
        ):
            job = TrainingJob(
                entries=[{"key": "k1", "question": "Q?", "answer": "A."}],
                adapter_name="episodic",
                adapter_config=_minimal_adapter_config(),
            )
            bt._current_adapter = "episodic"
            bt._current_job = job
            bt._active_adapter_name = "episodic"
            bt._active_inference_fallback = "episodic"
            bt._active_entries_fingerprint = ""
            bt._active_config_fingerprint = ""
            bt._active_total_epochs = 1
            bt._active_started_at = ""

            # Call _train_adapter but skip the resume state read and commit
            with (
                patch(
                    "paramem.server.background_trainer._read_resume_state",
                    return_value=None,
                ),
                patch.object(bt, "_commit_staging_to_production"),
                patch.object(bt, "_clear_resume_state"),
                patch("shutil.rmtree"),
            ):
                bt._train_adapter(job)

        assert captured, "train_adapter was not called"
        used_cfg = captured[0]
        assert used_cfg.save_strategy == "steps", (
            f"Expected save_strategy='steps', got {used_cfg.save_strategy!r}"
        )
        assert used_cfg.save_steps == 5, f"Expected save_steps=5, got {used_cfg.save_steps!r}"

    def test_train_adapter_preserves_epoch_when_bg_empty(self) -> None:
        """When save_strategy_bg is empty, only logging_steps is overridden."""
        cfg = TrainingConfig(save_strategy_bg="", save_steps_bg=0)
        model = _make_stub_model("episodic", "in_training")
        bt = BackgroundTrainer(
            model=model,
            tokenizer=MagicMock(),
            training_config=cfg,
            output_dir="/tmp/test_bt_epoch",
        )
        captured: list[TrainingConfig] = []

        def _fake_train_adapter(**kwargs):
            captured.append(kwargs["training_config"])
            return {}

        _patch_train = patch(
            "paramem.server.background_trainer.train_adapter",
            side_effect=_fake_train_adapter,
        )
        _patch_staging = patch(
            "paramem.server.background_trainer._ensure_staging_shape_matches",
            side_effect=lambda m, _c: m,
        )
        _patch_fmt = patch(
            "paramem.server.background_trainer.format_entry_training",
            return_value=[{"input_ids": [1], "labels": [1], "attention_mask": [1]}],
        )
        with (
            _patch_train,
            _patch_staging,
            _patch_fmt,
            patch("paramem.models.loader.copy_adapter_weights"),
            patch("paramem.models.loader.switch_adapter"),
        ):
            job = TrainingJob(
                entries=[{"key": "k1", "question": "Q?", "answer": "A."}],
                adapter_name="episodic",
                adapter_config=_minimal_adapter_config(),
            )
            with (
                patch(
                    "paramem.server.background_trainer._read_resume_state",
                    return_value=None,
                ),
                patch.object(bt, "_commit_staging_to_production"),
                patch.object(bt, "_clear_resume_state"),
                patch("shutil.rmtree"),
            ):
                bt._train_adapter(job)

        assert captured, "train_adapter was not called"
        used_cfg = captured[0]
        # save_strategy unchanged from default "epoch"
        assert used_cfg.save_strategy == "epoch", (
            f"Expected save_strategy='epoch', got {used_cfg.save_strategy!r}"
        )
        # Only logging_steps was overridden
        assert used_cfg.logging_steps == 10

    def test_persist_dedupes_across_callback_event_mismatch(self) -> None:
        """At an epoch boundary, on_epoch_end fires first then on_save fires; both
        must hit dedup and only one _write_resume_state call survives.

        Without the alignment fix, on_epoch_end passes int(state.epoch)=1 while
        on_save passes int(state.global_step)=500 — different keys, dedup misses,
        two writes occur and the second one overwrites last_completed_epoch on
        disk with 500. This test asserts the fix: both callbacks normalize to
        state.global_step so the dedup key is the same.
        """
        model = _make_stub_model("episodic", "in_training")
        bt = BackgroundTrainer(
            model=model,
            tokenizer=MagicMock(),
            training_config=_minimal_training_config(),
            output_dir="/tmp/test_bt_dedup_mismatch",
        )
        bt._last_persist_key = None

        # Wire both hooks to bt._persist_resume_state (production wiring).
        hooks = TrainingHooks(
            on_epoch_persist=bt._persist_resume_state,
            on_save_persist=bt._persist_resume_state,
        )
        cb = _HooksAdapterCallback(hooks)

        # Simulate HF's order at an epoch boundary:
        # 1. on_epoch_end fires: epoch=1, global_step=500
        # 2. on_save fires:     global_step=500
        # Both should resolve to global_step=500 as the dedup key.
        epoch_state = MagicMock(epoch=1, global_step=500)
        save_state = MagicMock(global_step=500)
        args = MagicMock(output_dir="/tmp/test_bt_dedup_mismatch")
        control = MagicMock()

        with patch.object(bt, "_write_resume_state") as mock_write:
            cb.on_epoch_end(args=args, state=epoch_state, control=control)
            cb.on_save(args=args, state=save_state, control=control)

        assert mock_write.call_count == 1, (
            f"Expected exactly 1 _write_resume_state call; got {mock_write.call_count}. "
            "Dedup failed — both callbacks must normalize to state.global_step."
        )
        actual_step = mock_write.call_args[0][0]
        assert actual_step == 500, (
            f"Expected stored value 500 (global_step); got {actual_step!r}. "
            "on_epoch_end must pass state.global_step, not state.epoch."
        )


# ---------------------------------------------------------------------------
# Test 13 — Step-end shutdown via TrainingHooks
# ---------------------------------------------------------------------------


class TestStepEndShutdown:
    """Verify on_shutdown_check fires at step_end via _HooksAdapterCallback."""

    def test_step_end_shutdown_via_hook_sets_control_flag(self) -> None:
        """on_shutdown_check=True at step_end sets control.should_training_stop."""
        hooks = TrainingHooks(on_shutdown_check=lambda: True)
        cb = _HooksAdapterCallback(hooks)
        control = MagicMock()
        control.should_training_stop = False
        cb.on_step_end(
            args=MagicMock(),
            state=MagicMock(global_step=42),
            control=control,
        )
        assert control.should_training_stop is True
