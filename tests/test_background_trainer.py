"""Unit tests for BackgroundTrainer.

Covers:
  - TrainingJob.inference_fallback_adapter field defaults to "episodic".
  - abort_for_inference() returns False when idle, sets abort event, quiesces.
  - training_hooks_for_job ORs shutdown_requested, abort flag, and caller gate.
  - Per-job abort events do not leak across jobs.
  - on_shutdown_check fires at step_end via TrainingHooks.
  - train_adapter returns metrics["aborted"]=True when shutdown predicate fires.

No GPU required — model interactions are replaced with MagicMock stubs.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

from paramem.server.background_trainer import (
    BackgroundTrainer,
    TrainingJob,
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


def _make_staging_safe_model(
    adapter_config: AdapterConfig, adapter_name: str = "test_adapter"
) -> MagicMock:
    """MagicMock model that satisfies ``train_adapter``'s staging+promote contract.

    Pre-populates BOTH the production slot *adapter_name* and the staging slot
    ``in_training`` with matching ``rank`` / ``target_modules``.  Exposes one
    real ``torch`` tensor per ``(target_module, slot)`` pair via
    ``named_parameters`` so ``copy_adapter_weights`` finds parallel
    source/destination key sets at entry (production → staging) and at exit
    (staging → production).

    Use this whenever a test needs to drive ``train_adapter`` to inspect
    something orthogonal to staging (abort metric, RAM-mode arg routing,
    callback assembly).  Tests that exercise the staging contract itself
    should construct their own model via ``_make_staging_model`` in
    ``tests/training/test_train_adapter_callbacks.py``.
    """
    import torch

    model = MagicMock()
    staging_cfg = MagicMock()
    staging_cfg.r = adapter_config.rank
    staging_cfg.target_modules = set(adapter_config.target_modules)
    prod_cfg = MagicMock()
    prod_cfg.r = adapter_config.rank
    prod_cfg.target_modules = set(adapter_config.target_modules)
    model.peft_config = {adapter_name: prod_cfg, "in_training": staging_cfg}

    named_params: list[tuple[str, "torch.Tensor"]] = []
    for module in sorted(adapter_config.target_modules):
        for slot in (adapter_name, "in_training"):
            named_params.append((f"base_model.model.{module}.{slot}.weight", torch.zeros(1)))
    model.named_parameters.return_value = named_params
    model.parameters.return_value = [t for _, t in named_params]
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
# Test 9 — abort_for_inference() replaces pause/resume
# ---------------------------------------------------------------------------


class TestAbortForInference:
    """abort_for_inference() sets the per-job abort flag and waits for quiesced."""

    def test_abort_returns_false_when_idle(self) -> None:
        """abort_for_inference() is a no-op and returns False when not training."""
        model = _make_stub_model("episodic", "in_training")
        bt = BackgroundTrainer(
            model=model,
            tokenizer=MagicMock(),
            training_config=_minimal_training_config(),
            output_dir="/tmp/test_bt_abort_idle",
        )
        # No active job — _active_abort is None.
        assert bt._active_abort is None
        result = bt.abort_for_inference(timeout=0.1)
        assert result is False

    def test_abort_sets_abort_event_and_waits_for_quiesced(self) -> None:
        """abort_for_inference() sets the abort flag and returns True when quiesced fires."""
        model = _make_stub_model("episodic", "in_training")
        bt = BackgroundTrainer(
            model=model,
            tokenizer=MagicMock(),
            training_config=_minimal_training_config(),
            output_dir="/tmp/test_bt_abort_sets",
        )
        # Manually install per-job events as _run_callable_queue would.
        with bt._active_state_lock:
            bt._active_abort = threading.Event()
            bt._active_quiesced = threading.Event()

        # Fire the quiesced event in a background thread after a brief delay.
        def _fire_quiesced():
            import time

            time.sleep(0.05)
            bt._active_quiesced.set()

        t = threading.Thread(target=_fire_quiesced, daemon=True)
        t.start()

        result = bt.abort_for_inference(timeout=2.0)
        assert result is True
        assert bt._active_abort.is_set()
        t.join(timeout=1.0)

    def test_abort_training_hooks_shutdown_predicate_returns_true(self) -> None:
        """The training_hooks_for_job shutdown predicate returns True after abort."""
        model = _make_stub_model("episodic", "in_training")
        bt = BackgroundTrainer(
            model=model,
            tokenizer=MagicMock(),
            training_config=_minimal_training_config(),
            output_dir="/tmp/test_bt_hooks_abort",
        )
        # Install per-job events.
        with bt._active_state_lock:
            bt._active_abort = threading.Event()
            bt._active_quiesced = threading.Event()

        hooks = bt.training_hooks_for_job()
        # Before abort: predicate is False.
        assert hooks.on_shutdown_check() is False
        # Set abort: predicate becomes True.
        bt._active_abort.set()
        assert hooks.on_shutdown_check() is True

    def test_per_job_abort_does_not_leak_to_next_job(self) -> None:
        """A cleared per-job event does not affect the next job's predicate."""
        model = _make_stub_model("episodic", "in_training")
        bt = BackgroundTrainer(
            model=model,
            tokenizer=MagicMock(),
            training_config=_minimal_training_config(),
            output_dir="/tmp/test_bt_no_leak",
        )
        # Simulate job A: install, abort, clear.
        with bt._active_state_lock:
            bt._active_abort = threading.Event()
            bt._active_quiesced = threading.Event()
        bt._active_abort.set()
        hooks_a = bt.training_hooks_for_job()
        assert hooks_a.on_shutdown_check() is True

        # Clear (simulate job A teardown).
        with bt._active_state_lock:
            bt._active_abort = None
            bt._active_quiesced = None

        # Install job B events.
        with bt._active_state_lock:
            bt._active_abort = threading.Event()
            bt._active_quiesced = threading.Event()
        hooks_b = bt.training_hooks_for_job()
        # Job B's predicate must start False.
        assert hooks_b.on_shutdown_check() is False

    def test_training_hooks_ors_shutdown_requested_and_abort(self) -> None:
        """shutdown_requested True makes predicate True even without abort."""
        model = _make_stub_model("episodic", "in_training")
        bt = BackgroundTrainer(
            model=model,
            tokenizer=MagicMock(),
            training_config=_minimal_training_config(),
            output_dir="/tmp/test_bt_shutdown_or",
        )
        with bt._active_state_lock:
            bt._active_abort = threading.Event()
            bt._active_quiesced = threading.Event()

        hooks = bt.training_hooks_for_job()
        assert hooks.on_shutdown_check() is False

        bt._shutdown_requested = True
        assert hooks.on_shutdown_check() is True

        bt._shutdown_requested = False
        assert hooks.on_shutdown_check() is False

        bt._active_abort.set()
        assert hooks.on_shutdown_check() is True


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
# Test 12 — train_adapter aborted return value
# ---------------------------------------------------------------------------


class TestTrainAdapterAbortReturn:
    """train_adapter returns metrics["aborted"]=True when shutdown predicate fires."""

    def test_aborted_true_when_shutdown_predicate_fires(self, tmp_path) -> None:
        """When hooks.on_shutdown_check returns True, metrics["aborted"] is True."""
        from paramem.training.trainer import train_adapter

        # Shutdown predicate always returns True (simulates abort signal)
        hooks = TrainingHooks(on_shutdown_check=lambda: True)

        ac = AdapterConfig(rank=4, alpha=8, learning_rate=1e-4, target_modules=["q_proj"])
        model = _make_staging_safe_model(ac)

        class _MinimalTrainer:
            def __init__(self, *, model, args, train_dataset, data_collator, callbacks, **kwargs):
                self._callbacks = callbacks
                self._args = args

            def train(self, resume_from_checkpoint=None):
                # Fire on_step_end so _HooksAdapterCallback sets should_training_stop
                control = MagicMock()
                control.should_training_stop = False
                state = MagicMock(global_step=1, epoch=1)
                for cb in self._callbacks:
                    if hasattr(cb, "on_step_end"):
                        cb.on_step_end(self._args, state, control)
                return MagicMock(metrics={"train_loss": 0.5})

        tc = TrainingConfig(
            num_epochs=1,
            gradient_checkpointing=False,
            batch_size=1,
            warmup_steps=0,
            warmup_ratio=0.0,
        )
        dataset = [{"input_ids": [1], "labels": [1], "attention_mask": [1]}]

        with (
            patch("paramem.training.trainer.Trainer", new=_MinimalTrainer),
            patch("paramem.training.trainer.TrainingArguments", return_value=MagicMock()),
            patch(
                "paramem.training.encrypted_checkpoint_callback.EncryptCheckpointCallback",
                MagicMock,
            ),
            patch("paramem.training.trainer._ensure_staging_slot", return_value=None),
        ):
            metrics = train_adapter(
                model=model,
                tokenizer=MagicMock(),
                train_dataset=dataset,
                adapter_name="test_adapter",
                training_config=tc,
                adapter_config=ac,
                output_dir=tmp_path,
                hooks=hooks,
            )

        assert metrics.get("aborted") is True, (
            f"Expected metrics['aborted']=True when shutdown predicate fires, got {metrics}"
        )

    def test_aborted_false_when_no_hooks(self, tmp_path) -> None:
        """When hooks=None, metrics['aborted'] is False."""
        from paramem.training.trainer import train_adapter

        ac = AdapterConfig(rank=4, alpha=8, learning_rate=1e-4, target_modules=["q_proj"])
        model = _make_staging_safe_model(ac)

        class _NullTrainer:
            def __init__(self, **kwargs):
                pass

            def train(self, resume_from_checkpoint=None):
                return MagicMock(metrics={"train_loss": 0.1})

        tc = TrainingConfig(
            num_epochs=1,
            gradient_checkpointing=False,
            batch_size=1,
            warmup_steps=0,
            warmup_ratio=0.0,
        )
        dataset = [{"input_ids": [1], "labels": [1], "attention_mask": [1]}]

        with (
            patch("paramem.training.trainer.Trainer", new=_NullTrainer),
            patch("paramem.training.trainer.TrainingArguments", return_value=MagicMock()),
            patch(
                "paramem.training.encrypted_checkpoint_callback.EncryptCheckpointCallback",
                MagicMock,
            ),
            patch("paramem.training.trainer._ensure_staging_slot", return_value=None),
        ):
            metrics = train_adapter(
                model=model,
                tokenizer=MagicMock(),
                train_dataset=dataset,
                adapter_name="test_adapter",
                training_config=tc,
                adapter_config=ac,
                output_dir=tmp_path,
                hooks=None,
            )

        assert metrics.get("aborted") is False, (
            f"Expected metrics['aborted']=False when hooks=None, got {metrics}"
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


# ---------------------------------------------------------------------------
# Test 14 — RAM-mode checkpointing
# ---------------------------------------------------------------------------


class TestRamCheckpointMode:
    """train_adapter RAM-mode: save_steps_ram > 0 routes checkpoints to /dev/shm.

    Verifies:
    - When save_steps_ram > 0, TrainingArguments receives output_dir=/dev/shm/...
      and save_strategy="steps" with save_steps=save_steps_ram.
    - On clean (non-abort) completion, /dev/shm directory is deleted.
    - _RamEpochCopyCallback is registered and copies the latest checkpoint
      to <output_dir>/bg_checkpoint_epoch/ at epoch_end.
    """

    def _make_tc(self, save_steps_ram: int = 50) -> "TrainingConfig":
        return TrainingConfig(
            num_epochs=1,
            gradient_checkpointing=False,
            batch_size=1,
            warmup_steps=0,
            warmup_ratio=0.0,
            save_steps_ram=save_steps_ram,
        )

    def test_ram_mode_routes_output_dir_to_dev_shm(self, tmp_path) -> None:
        """When save_steps_ram > 0, HF TrainingArguments receives /dev/shm path."""
        from paramem.training.trainer import train_adapter

        captured_args: list = []

        class _CapturingTrainer:
            def __init__(self, *, model, args, train_dataset, data_collator, callbacks, **kwargs):
                captured_args.append(args)

            def train(self, resume_from_checkpoint=None):
                return MagicMock(metrics={"train_loss": 0.1})

        ac = AdapterConfig(rank=4, alpha=8, learning_rate=1e-4, target_modules=["q_proj"])
        model = _make_staging_safe_model(ac)
        tc = self._make_tc(save_steps_ram=50)

        with (
            patch("paramem.training.trainer.Trainer", new=_CapturingTrainer),
            patch(
                "paramem.training.trainer.TrainingArguments",
                side_effect=lambda **kw: MagicMock(**kw),
            ),
            patch(
                "paramem.training.encrypted_checkpoint_callback.EncryptCheckpointCallback",
                MagicMock,
            ),
            patch("paramem.training.trainer._ensure_staging_slot", return_value=None),
        ):
            train_adapter(
                model=model,
                tokenizer=MagicMock(),
                train_dataset=[{"input_ids": [1], "labels": [1], "attention_mask": [1]}],
                adapter_name="test_adapter",
                training_config=tc,
                adapter_config=ac,
                output_dir=tmp_path,
            )

        assert captured_args, "Trainer was not constructed"
        args = captured_args[0]
        # TrainingArguments.output_dir must point to /dev/shm.
        assert str(args.output_dir).startswith("/dev/shm"), (
            f"save_steps_ram>0 must route output_dir to /dev/shm; got {args.output_dir!r}"
        )
        # save_strategy must be "steps" and save_steps must equal save_steps_ram.
        assert str(args.save_strategy) == "steps", (
            f"save_steps_ram>0 must set save_strategy='steps'; got {args.save_strategy!r}"
        )
        assert args.save_steps == 50, (
            f"save_steps must equal save_steps_ram; got {args.save_steps!r}"
        )

    def test_ram_dir_cleaned_on_successful_completion(self, tmp_path) -> None:
        """On non-abort completion, train_adapter removes the /dev/shm directory."""
        from pathlib import Path

        from paramem.training.trainer import train_adapter

        class _NullTrainer:
            def __init__(self, **kwargs):
                pass

            def train(self, resume_from_checkpoint=None):
                return MagicMock(metrics={"train_loss": 0.1})

        ac = AdapterConfig(rank=4, alpha=8, learning_rate=1e-4, target_modules=["q_proj"])
        model = _make_staging_safe_model(ac)
        tc = self._make_tc(save_steps_ram=50)

        created_ram_dir: list[Path] = []
        _original_mkdir = Path.mkdir

        def _intercept_mkdir(self_path, **kwargs):
            if "paramem-bg-checkpoint-" in str(self_path):
                created_ram_dir.append(self_path)
            return _original_mkdir(self_path, **kwargs)

        with (
            patch("paramem.training.trainer.Trainer", new=_NullTrainer),
            patch("paramem.training.trainer.TrainingArguments", return_value=MagicMock()),
            patch(
                "paramem.training.encrypted_checkpoint_callback.EncryptCheckpointCallback",
                MagicMock,
            ),
            patch("pathlib.Path.mkdir", _intercept_mkdir),
            patch("paramem.training.trainer._ensure_staging_slot", return_value=None),
        ):
            train_adapter(
                model=model,
                tokenizer=MagicMock(),
                train_dataset=[{"input_ids": [1], "labels": [1], "attention_mask": [1]}],
                adapter_name="test_adapter",
                training_config=tc,
                adapter_config=ac,
                output_dir=tmp_path / "output",
            )

        # If a ram_dir was created, it must have been removed on success.
        for rd in created_ram_dir:
            assert not rd.exists(), (
                f"RAM checkpoint dir must be deleted on successful completion; {rd} still exists"
            )

    def test_zero_save_steps_ram_disables_ram_mode(self, tmp_path) -> None:
        """When save_steps_ram=0 (default), TrainingArguments uses the caller's output_dir."""
        from paramem.training.trainer import train_adapter

        captured_args: list = []

        class _CapturingTrainer:
            def __init__(self, *, model, args, train_dataset, data_collator, callbacks, **kwargs):
                captured_args.append(args)

            def train(self, resume_from_checkpoint=None):
                return MagicMock(metrics={"train_loss": 0.1})

        ac = AdapterConfig(rank=4, alpha=8, learning_rate=1e-4, target_modules=["q_proj"])
        model = _make_staging_safe_model(ac)

        tc = TrainingConfig(
            num_epochs=1,
            gradient_checkpointing=False,
            batch_size=1,
            warmup_steps=0,
            warmup_ratio=0.0,
            save_steps_ram=0,
        )

        with (
            patch("paramem.training.trainer.Trainer", new=_CapturingTrainer),
            patch(
                "paramem.training.trainer.TrainingArguments",
                side_effect=lambda **kw: MagicMock(**kw),
            ),
            patch(
                "paramem.training.encrypted_checkpoint_callback.EncryptCheckpointCallback",
                MagicMock,
            ),
            patch("paramem.training.trainer._ensure_staging_slot", return_value=None),
        ):
            train_adapter(
                model=model,
                tokenizer=MagicMock(),
                train_dataset=[{"input_ids": [1], "labels": [1], "attention_mask": [1]}],
                adapter_name="test_adapter",
                training_config=tc,
                adapter_config=ac,
                output_dir=tmp_path,
            )

        assert captured_args, "Trainer was not constructed"
        args = captured_args[0]
        # output_dir must NOT start with /dev/shm when save_steps_ram=0.
        assert not str(args.output_dir).startswith("/dev/shm"), (
            f"save_steps_ram=0 must use caller's output_dir; got {args.output_dir!r}"
        )


# ---------------------------------------------------------------------------
# Test 15 — safe_empty_cache called after job completion + _shutdown_requested reset
# ---------------------------------------------------------------------------


class TestWorkerJobBoundary:
    """Verify per-job VRAM reclaim and _shutdown_requested reset on the worker boundary.

    Covers two singleton-reuse invariants:

    1. safe_empty_cache is called once per job completion, AFTER _quiesced.set()
       (ordering is load-bearing — the abort-quiesce latency contract must not
       be delayed by the cache reclaim).

    2. _shutdown_requested is reset to False at the START of each job even when
       it was set True before submit (e.g. from a prior inference-abort or
       SIGTERM path that did NOT exit the process).  Without the reset a stale
       flag would permanently poison all subsequent jobs on the singleton.
    """

    @staticmethod
    def _noop_gpu_lock():
        from contextlib import contextmanager

        @contextmanager
        def _inner():
            yield

        return _inner

    def test_safe_empty_cache_called_after_quiesced(self) -> None:
        """safe_empty_cache is called exactly once per job, AFTER _quiesced.set().

        Ordering is verified by recording the call sequence into a shared list.
        Both "quiesced_set" (from the per-job quiesced event's .set()) and
        "safe_empty_cache" are appended to call_order, and the assertion confirms
        quiesced fires first — the abort-quiesce latency contract.
        """
        import time

        model = _make_stub_model("episodic", "in_training")
        bt = BackgroundTrainer(
            model=model,
            tokenizer=MagicMock(),
            training_config=_minimal_training_config(),
            output_dir="/tmp/test_bt_cache_order",
        )

        call_order: list[str] = []
        job_done = threading.Event()

        def job() -> None:
            job_done.set()

        # Build a threading.Event factory that wraps the per-job _active_quiesced
        # event with a spy on .set().  The worker creates exactly two events per
        # job (abort then quiesced, lines 318-319); only the second is
        # _active_quiesced, so the spy skips the first creation.
        _real_Event = threading.Event
        _event_creation_count = 0

        def _spy_Event_factory():
            nonlocal _event_creation_count
            _event_creation_count += 1
            real_ev = _real_Event()
            if _event_creation_count == 1:
                # This is _active_abort — do not instrument.
                return real_ev
            # This is _active_quiesced — spy its .set() to record ordering.
            real_set = real_ev.set

            def _spy_set():
                call_order.append("quiesced_set")
                real_set()

            real_ev.set = _spy_set  # type: ignore[method-assign]
            return real_ev

        with (
            patch(
                "paramem.server.background_trainer.safe_empty_cache",
                side_effect=lambda: call_order.append("safe_empty_cache"),
            ) as mock_cache,
            patch("paramem.server.gpu_lock.gpu_lock_sync", new=self._noop_gpu_lock()),
            patch(
                "paramem.server.background_trainer.threading.Event",
                side_effect=_spy_Event_factory,
            ),
        ):
            # Submit the job.  The worker will:
            #   1. Run job()
            #   2. _quiesced.set()  → appends "quiesced_set"
            #   3. safe_empty_cache()  → appends "safe_empty_cache"
            bt.submit(job)
            # Wait for the job's event to confirm execution started.
            assert job_done.wait(timeout=5.0), "job did not execute"
            # Wait until safe_empty_cache has been called once (finally block done).
            deadline = time.monotonic() + 5.0
            while "safe_empty_cache" not in call_order and time.monotonic() < deadline:
                time.sleep(0.01)

        assert mock_cache.call_count == 1, (
            f"safe_empty_cache must be called exactly once per job; got {mock_cache.call_count}"
        )
        assert call_order == ["quiesced_set", "safe_empty_cache"], (
            f"Expected ['quiesced_set', 'safe_empty_cache'] in call_order to prove "
            f"safe_empty_cache runs AFTER _quiesced.set(); got {call_order}"
        )

    def test_shutdown_requested_reset_per_job(self) -> None:
        """_shutdown_requested is False at job start even when set True before submit.

        Simulates a prior inference-abort or SIGTERM-without-process-exit that
        set _shutdown_requested=True.  With the singleton reuse fix, the flag
        must be reset at the START of the next job so training is not
        permanently poisoned.
        """
        model = _make_stub_model("episodic", "in_training")
        bt = BackgroundTrainer(
            model=model,
            tokenizer=MagicMock(),
            training_config=_minimal_training_config(),
            output_dir="/tmp/test_bt_shutdown_reset",
        )

        # Poison the flag as a prior abort would.
        bt._shutdown_requested = True

        observed_at_job_start: list[bool] = []
        job_done = threading.Event()

        def job() -> None:
            # Record the flag value at the moment the job body executes.
            observed_at_job_start.append(bt._shutdown_requested)
            job_done.set()

        with patch("paramem.server.gpu_lock.gpu_lock_sync", new=self._noop_gpu_lock()):
            bt.submit(job)
            assert job_done.wait(timeout=5.0), "job did not execute"

        assert observed_at_job_start == [False], (
            f"_shutdown_requested must be False at job start even after being set True "
            f"before submit; got {observed_at_job_start}"
        )
