"""Background training manager for continuous adapter learning.

Trains adapters during idle time, pausing for inference requests.
Saves full training state at each epoch boundary for crash recovery.

The model stays loaded — switching between training and inference is
just flag flips (model.eval/train, gradient checkpointing), no reload.
"""

import hashlib
import json
import logging
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from transformers import (
    Trainer,
    TrainerCallback,
    TrainingArguments,
    default_data_collator,
)

from paramem.training.indexed_memory import format_indexed_training
from paramem.utils.config import AdapterConfig, TrainingConfig

logger = logging.getLogger(__name__)

# Sentinel placed on _job_queue to signal the persistent callable worker to exit.
_WORKER_STOP = object()

# Name of the atomic resume state file written inside output_dir/in_training/.
_RESUME_STATE_FILE = "resume_state.json"


def is_thermal_policy_active(
    mode: str,
    start: str,
    end: str,
    now: datetime | None = None,
) -> bool:
    """Pure predicate: is the thermal throttle active under this quiet-hours policy?

    Shared between ``BackgroundTrainer._should_throttle_now`` and the ``/status``
    endpoint so the latter can report policy state without a live trainer.

    See ``ConsolidationScheduleConfig`` for mode semantics. Invalid windows in
    ``auto`` mode fall back to ``True`` (prefer-silence default).
    """
    if mode == "always_off":
        return False
    if mode == "always_on":
        return True
    # mode == "auto"
    try:
        sh, sm = (int(x) for x in start.split(":"))
        eh, em = (int(x) for x in end.split(":"))
    except Exception:
        return True
    t = (now or datetime.now()).time()
    cur = t.hour * 60 + t.minute
    s = sh * 60 + sm
    e = eh * 60 + em
    if s == e:
        return True
    if s < e:
        return s <= cur < e
    return cur >= s or cur < e


def _fingerprint_keyed_pairs(keyed_pairs: list[dict]) -> str:
    """Return a SHA-256 hex digest of the canonical serialisation of keyed_pairs.

    Order-sensitive: two lists with the same items in different orders produce
    different fingerprints.  This is intentional — the QA generator output
    order encodes the key insertion sequence, so a genuinely identical job
    produces identical pairs in the same order.

    Args:
        keyed_pairs: Training pairs as returned by the QA generator.

    Returns:
        Lowercase hex SHA-256 string.
    """
    serialised = json.dumps(keyed_pairs, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialised.encode()).hexdigest()


def _fingerprint_training_config(
    training_config: TrainingConfig, adapter_config: AdapterConfig
) -> str:
    """Return a SHA-256 hex digest of the training-relevant config fields.

    Covers the fields that, if changed, would invalidate a checkpoint from a
    prior training run (epoch count, batch size, LR, LoRA rank/alpha).

    Args:
        training_config: Server training config.
        adapter_config: Per-adapter LoRA config.

    Returns:
        Lowercase hex SHA-256 string.
    """
    target_modules = adapter_config.target_modules
    relevant = {
        "num_epochs": training_config.num_epochs,
        "batch_size": training_config.batch_size,
        "gradient_accumulation_steps": training_config.gradient_accumulation_steps,
        "lr_scheduler_type": training_config.lr_scheduler_type,
        "weight_decay": training_config.weight_decay,
        "warmup_steps": training_config.warmup_steps,
        "warmup_ratio": training_config.warmup_ratio,
        "rank": adapter_config.rank,
        "alpha": adapter_config.alpha,
        "learning_rate": adapter_config.learning_rate,
        "target_modules": sorted(target_modules) if target_modules is not None else [],
    }
    serialised = json.dumps(relevant, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(serialised.encode()).hexdigest()


def _write_resume_state_atomic(state_path: Path, state: dict) -> None:
    """Write resume state to disk atomically using a temp-file + rename.

    Args:
        state_path: Absolute path to the target JSON file.
        state: Dict to serialise.

    Raises:
        OSError: If the write or rename fails.
    """
    tmp_path = state_path.with_suffix(f".tmp.{os.getpid()}")
    try:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        tmp_path.replace(state_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _read_resume_state(state_path: Path) -> dict | None:
    """Load resume state from disk.

    Returns None if the file does not exist or cannot be parsed.

    Args:
        state_path: Path to the resume_state.json file.

    Returns:
        Parsed dict, or None.
    """
    if not state_path.exists():
        return None
    try:
        with open(state_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        logger.warning("Could not parse resume state at %s — discarding", state_path)
        return None


def _gpu_temp() -> int | None:
    """Read current GPU temperature via nvidia-smi. Returns None on failure."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return int(result.stdout.strip())
    except Exception:
        pass
    return None


@dataclass
class TrainingJob:
    """A single adapter training job.

    Attributes:
        keyed_pairs: Pre-tokenized training pairs for this adapter.
        adapter_name: Name of the production adapter being trained (e.g.
            ``"episodic"`` or ``"episodic_interim_20260418T1430"``).
        adapter_config: LoRA config for the adapter.
        inference_fallback_adapter: Adapter to activate when training is paused
            for inference.  Must be a **committed** adapter slot with stable
            weights — never the mid-training ``in_training`` staging slot.

            For interim-adapter jobs (this Step 6): ``"episodic"`` (the stable
            main).  For main consolidation refresh jobs (Step 7): the backed-up
            prior main adapter name.

            Defaults to ``"episodic"`` so existing callers that create
            ``TrainingJob`` without this field continue to work correctly.
    """

    keyed_pairs: list[dict]
    adapter_name: str
    adapter_config: AdapterConfig
    inference_fallback_adapter: str = "episodic"


class _PauseForInferenceCallback(TrainerCallback):
    """Yields to inference between training steps.

    Releases the GPU lock while paused so STT/TTS/inference can proceed,
    then re-acquires before resuming training. Uses a bounded wait to
    prevent permanent stalls if the inference caller crashes.
    """

    _INFERENCE_TIMEOUT = 120.0  # max seconds to wait for inference to finish

    def __init__(self, trainer_ref: "BackgroundTrainer"):
        self._trainer = trainer_ref

    def on_step_end(self, args, state, control, **kwargs):
        # Yield to inference if requested
        if self._trainer._inference_requested.is_set():
            from paramem.models.loader import switch_adapter
            from paramem.server.gpu_lock import acquire_gpu, release_gpu

            logger.debug("Training paused for inference at step %d", state.global_step)
            release_gpu()
            self._trainer._training_paused.set()
            signalled = self._trainer._inference_done.wait(timeout=self._INFERENCE_TIMEOUT)
            if not signalled:
                logger.warning(
                    "Inference did not signal done within %.0fs — resuming training",
                    self._INFERENCE_TIMEOUT,
                )
            self._trainer._inference_done.clear()
            acquire_gpu()
            # Restore staging adapter as active — inference may have switched
            # to a production adapter. Training must resume on in_training.
            switch_adapter(self._trainer.model, "in_training")
            self._trainer._training_paused.clear()
            logger.debug("Training resumed at step %d (adapter=in_training)", state.global_step)

        # Thermal throttle — check every N steps
        self._trainer._thermal_throttle(state.global_step)

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = int(state.epoch)
        self._trainer._last_completed_epoch = epoch
        # Persist resume state so a crash during a later epoch can restart here.
        self._trainer._write_resume_state(epoch, args.output_dir)
        # Also check thermal throttle at epoch boundaries
        self._trainer._thermal_throttle(state.global_step)
        if self._trainer._shutdown_requested:
            logger.info("Shutdown requested — stopping after epoch %d", epoch)
            control.should_training_stop = True


class BackgroundTrainer:
    """Manages background training that yields to inference.

    Supports a queue of training jobs (e.g., episodic then procedural).
    Each job trains one adapter. Jobs run sequentially in a single thread.

    Usage:
        bt = BackgroundTrainer(model, tokenizer, config)
        bt.start_jobs([job1, job2], on_complete=save_callback)

        # When inference is needed:
        bt.pause()       # waits for current step to finish
        # ... run inference ...
        bt.resume()      # lets training continue

        bt.stop()        # graceful stop at next epoch boundary
    """

    def __init__(
        self,
        model,
        tokenizer,
        training_config: TrainingConfig,
        output_dir: str | Path = "data/ha/adapters",
        temp_limit: int = 0,
        temp_check_interval: int = 5,
        quiet_hours_mode: str = "always_on",
        quiet_hours_start: str = "22:00",
        quiet_hours_end: str = "07:00",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.training_config = training_config
        self.output_dir = Path(output_dir)
        self._temp_limit = temp_limit  # 0 = disabled
        self._temp_check_interval = temp_check_interval
        self._quiet_hours_mode = quiet_hours_mode
        self._quiet_hours_start = quiet_hours_start
        self._quiet_hours_end = quiet_hours_end

        self._inference_requested = threading.Event()
        self._inference_done = threading.Event()
        self._training_paused = threading.Event()
        # Serializes pause/resume cycles. A second pause() waits until the
        # first has fully completed (callback re-acquired lock, restored
        # staging, cleared _training_paused) to prevent the re-entrancy race
        # where two threads manipulate model state concurrently.
        self._pause_lock = threading.Lock()
        # State guard for _pause_active flag. Ensures exactly-once transition
        # from True→False across concurrent resume() calls.
        self._pause_state_lock = threading.Lock()
        self._pause_active = False
        self._shutdown_requested = False
        self._training_thread: threading.Thread | None = None
        self._is_training = False
        self._last_completed_epoch = 0
        self._current_adapter = ""
        self._current_job: TrainingJob | None = None
        self._on_error: Callable[[], None] | None = None
        # Fingerprints for the active training job, set at the start of
        # _train_adapter and read by the epoch callback to write resume state.
        self._active_keyed_pairs_fingerprint: str = ""
        self._active_config_fingerprint: str = ""
        self._active_total_epochs: int = 0
        self._active_adapter_name: str = ""
        self._active_inference_fallback: str = ""
        self._active_started_at: str = ""

        # Callable-job queue for submit().  Holds (callable, fallback_adapter)
        # pairs or the _WORKER_STOP sentinel.  Drained by _run_callable_queue
        # on a single persistent daemon worker thread that lives for the
        # process lifetime.  A persistent worker eliminates the race where a
        # concurrent submit() call sees is_alive()==True on a thread that has
        # already decided to exit (queue empty) but not yet terminated.
        self._job_queue: queue.SimpleQueue = queue.SimpleQueue()
        # Guards one-time worker-thread creation.  After the first submit() the
        # worker is alive for the process lifetime; subsequent submit() calls
        # skip the creation path entirely.
        self._worker_lock = threading.Lock()
        self._worker_thread: threading.Thread | None = None

    @property
    def is_training(self) -> bool:
        return self._is_training

    @property
    def current_adapter_name(self) -> str:
        """Name of the adapter currently being trained (empty if idle)."""
        return self._current_adapter

    def start_jobs(
        self,
        jobs: list[TrainingJob],
        on_complete: Callable[[], None] | None = None,
        on_error: Callable[[], None] | None = None,
    ):
        """Start training a queue of adapter jobs.

        Non-blocking — launches a thread and returns immediately.
        on_complete is called after all jobs finish successfully.
        on_error is called if training raises an exception.
        """
        if self._is_training:
            logger.warning("Training already in progress")
            return

        if not jobs:
            logger.info("No training jobs to run")
            if on_complete:
                on_complete()
            return

        self._shutdown_requested = False
        self._is_training = True
        self._on_error = on_error

        self._training_thread = threading.Thread(
            target=self._run_jobs,
            args=(jobs, on_complete),
            daemon=True,
            name="bg-trainer",
        )
        self._training_thread.start()
        logger.info("Background training started: %d jobs", len(jobs))

    def pause(self, timeout: float = 5.0) -> bool:
        """Pause training for inference. Blocks until current step finishes.

        Switches the active adapter to the production slot (last-known-good
        weights, NOT the mid-training in_training slot) so inference returns
        correct answers. Also switches model to eval mode and disables
        gradient checkpointing.

        Serialized via _pause_lock so concurrent callers queue instead of
        racing on model state.
        """
        if not self._is_training:
            return True

        self._pause_lock.acquire()
        success = False
        try:
            # Double-check after acquiring: training may have ended while we waited
            if not self._is_training:
                return True

            self._inference_requested.set()
            paused = self._training_paused.wait(timeout=timeout)
            if not paused:
                logger.warning("Training did not pause within %.1fs", timeout)
                # The callback may have seen _inference_requested before we
                # timed out and is now waiting on _inference_done. Clear the
                # request AND signal done so it doesn't block the full 120s
                # timeout. Also consume any _training_paused set by a racing
                # callback so the next pause() starts from a clean state.
                self._inference_requested.clear()
                self._inference_done.set()
                self._training_paused.clear()
                return False

            # Switch to the committed production adapter so inference queries the
            # user's established knowledge, not mid-training weights.
            #
            # Use the job's explicit inference_fallback_adapter rather than
            # _current_adapter: for interim-adapter jobs the current adapter is
            # the episodic_interim_* slot, which contains staging-copy weights
            # that must NOT be exposed to inference during a pause.  The fallback
            # is always "episodic" (main) for Step-6 jobs and the prior committed
            # main for Step-7 refresh jobs.
            #
            # Guard: _current_job is set by _train_adapter; if pause() races
            # before the first job starts, fall back to "episodic".
            from paramem.models.loader import switch_adapter

            target = (
                self._current_job.inference_fallback_adapter
                if self._current_job is not None
                else "episodic"
            )
            if target in self.model.peft_config:
                switch_adapter(self.model, target)
                logger.debug("Paused: switched to production adapter '%s'", target)
            else:
                logger.warning(
                    "Paused but no production adapter available to switch to (target=%s)", target
                )

            # Switch model to inference mode
            self.model.eval()
            if self.training_config.gradient_checkpointing:
                self.model.gradient_checkpointing_disable()

            # Lock is kept held — resume() releases it. Matched pair.
            with self._pause_state_lock:
                self._pause_active = True
            success = True
            return True
        finally:
            if not success:
                # pause() did not fully succeed — release the lock so other
                # callers can proceed. resume() will be a no-op.
                self._pause_lock.release()

    def resume(self):
        """Resume training after inference.

        Restores model to training mode with gradient checkpointing.
        No-op if training is not active or no pause is active.

        Must be paired with a successful pause() — releases _pause_lock.
        Safe to call multiple times concurrently: only one call releases
        the lock (compare-and-swap on _pause_active under _pause_state_lock).
        """
        # Atomic compare-and-swap: only the caller that transitions
        # _pause_active from True→False proceeds to release the lock.
        with self._pause_state_lock:
            if not self._pause_active:
                return
            self._pause_active = False
        try:
            if self._is_training:
                # Restore training mode
                self.model.train()
                if self.training_config.gradient_checkpointing:
                    self.model.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False}
                    )
                self._inference_requested.clear()
                self._inference_done.set()
        finally:
            self._pause_lock.release()

    def stop(self, timeout: float = 60.0) -> int:
        """Stop training gracefully. Returns last completed epoch."""
        if not self._is_training:
            return self._last_completed_epoch

        self._shutdown_requested = True
        # Unblock if paused waiting for inference
        self._inference_done.set()
        if self._training_thread is not None:
            self._training_thread.join(timeout=timeout)
            if self._training_thread.is_alive():
                logger.warning("Training thread did not stop within %.1fs", timeout)

        self._is_training = False
        return self._last_completed_epoch

    def submit(
        self,
        fn: Callable[[], None],
        *,
        inference_fallback_adapter: str = "episodic",
    ) -> None:
        """Enqueue a callable job to run serially in the background worker.

        Non-blocking — returns immediately after queuing.  Jobs from concurrent
        ``/chat`` turns are serialised: the second job waits in the queue until
        the first completes.  This eliminates concurrent-turn races on shared
        ``ConsolidationLoop`` state (``_indexed_next_index``).

        The GPU lock (``gpu_lock_sync``) is held for the duration of each job so
        that concurrent STT/TTS inference requests are correctly serialised via
        ``pause()`` / ``resume()``.

        ``inference_fallback_adapter`` is stored as a sentinel
        :class:`TrainingJob` on ``self._current_job`` so that
        :meth:`pause` switches to the correct committed adapter
        (``"episodic"`` for post-session jobs) rather than whatever adapter was
        last active during training.

        The callable worker thread is started once on the first
        :meth:`submit` call and lives for the process lifetime (persistent
        daemon).  This eliminates the race where a concurrent caller sees
        ``is_alive() == True`` on a thread that has already decided to exit
        (queue empty) but has not yet terminated, causing a new job to be
        silently stranded in the queue.

        Args:
            fn: Zero-argument callable to execute under the GPU lock.
            inference_fallback_adapter: Adapter to activate if ``pause()``
                is called while ``fn`` is executing.  Must name a committed
                adapter slot with stable weights.  Defaults to ``"episodic"``.
        """
        self._job_queue.put((fn, inference_fallback_adapter))
        with self._worker_lock:
            if self._worker_thread is None:
                self._worker_thread = threading.Thread(
                    target=self._run_callable_queue,
                    daemon=True,
                    name="bg-trainer-callable",
                )
                self._worker_thread.start()

    def _run_callable_queue(self) -> None:
        """Drain the callable-job queue serially under the GPU lock.

        Started once by the first :meth:`submit` call and runs for the
        process lifetime (persistent daemon thread).  Blocks on
        ``_job_queue.get()`` between jobs so no jobs are ever stranded.

        The previous implementation used ``get_nowait()`` and exited when the
        queue was empty.  That introduced a race: a concurrent ``submit()``
        could see ``is_alive() == True`` (thread not yet terminated) and skip
        starting a new worker, leaving the new job in the queue forever.
        The persistent-worker design eliminates the race entirely — there is
        always exactly one worker thread alive after the first submit.

        Each job runs under ``gpu_lock_sync()`` to prevent concurrent GPU
        access from consolidation training or inference.  A sentinel
        :class:`TrainingJob` with ``inference_fallback_adapter`` set is
        installed on ``self._current_job`` so that :meth:`pause` can switch
        to the correct committed adapter when yielding to inference.

        The loop exits only when the :data:`_WORKER_STOP` sentinel is
        dequeued, which is sent by :meth:`_stop_callable_worker` during
        process shutdown.
        """
        from paramem.server.gpu_lock import gpu_lock_sync

        while True:
            item = self._job_queue.get()  # blocks until a job (or sentinel) arrives
            if item is _WORKER_STOP:
                break

            fn, fallback_adapter = item

            # Install a sentinel so pause() reads the correct fallback adapter.
            sentinel = TrainingJob(
                keyed_pairs=[],
                adapter_name="_callable_",
                adapter_config=AdapterConfig(),
                inference_fallback_adapter=fallback_adapter,
            )
            self._current_job = sentinel
            self._is_training = True
            try:
                with gpu_lock_sync():
                    fn()
            except Exception:
                logger.exception("submit() job failed")
            finally:
                self._current_job = None
                self._is_training = False

    def _set_is_training(self, value: bool) -> None:
        """Override the ``_is_training`` flag from inside a submitted callable.

        Used by ``consolidate_interim_adapters`` to mark non-training phases
        (between per-tier ``_train_adapter`` calls; the entire finalize block)
        so ``pause()`` short-circuits and inference waits on the GPU lock alone
        instead of timing out on the never-firing
        ``_PauseForInferenceCallback``.

        No locking is needed: the writer (this method, called from inside the
        callable that holds the GPU lock) and the reader (``pause()``'s
        ``if not self._is_training: return True`` guard) perform simple boolean
        operations that are atomic under the GIL, and the callable is the sole
        writer while the GPU lock is held.

        Args:
            value: ``True`` to re-arm training mode before a ``_train_adapter``
                call; ``False`` to mark a non-training gap or the finalize block.
        """
        self._is_training = value

    def _stop_callable_worker(self, timeout: float = 5.0) -> None:
        """Signal the callable worker to exit and wait for it to terminate.

        Sends :data:`_WORKER_STOP` through the queue so the persistent daemon
        thread exits cleanly.  Safe to call even if no worker has ever been
        started.

        Args:
            timeout: Seconds to wait for the worker thread to join.  If the
                thread is still alive after the timeout a warning is logged and
                the method returns — the daemon will be killed on process exit.
        """
        with self._worker_lock:
            if self._worker_thread is None:
                return
            self._job_queue.put(_WORKER_STOP)
        self._worker_thread.join(timeout=timeout)
        if self._worker_thread.is_alive():
            logger.warning(
                "Callable worker did not exit within %.1fs; will be killed on process exit",
                timeout,
            )

    def _resume_state_path(self) -> Path:
        """Return the path to the resume_state.json file for the current output_dir.

        Returns:
            Absolute Path to output_dir/in_training/resume_state.json.
        """
        return self.output_dir / "in_training" / _RESUME_STATE_FILE

    def _write_resume_state(self, last_completed_epoch: int, checkpoint_output_dir: str) -> None:
        """Write the resume state file atomically after an epoch completes.

        Finds the highest-numbered checkpoint directory inside
        checkpoint_output_dir and records its path so the next restart can
        pass it directly to trainer.train(resume_from_checkpoint=...).

        Silently logs and returns on any error — a missing state file means
        a fresh restart, which is always safe.

        Args:
            last_completed_epoch: The epoch number just completed (1-based).
            checkpoint_output_dir: The HF Trainer output_dir containing
                checkpoint-<step> subdirectories.
        """
        checkpoint_dir = Path(checkpoint_output_dir)
        checkpoints = sorted(
            checkpoint_dir.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[1]) if p.name.split("-")[1].isdigit() else -1,
        )
        latest_checkpoint = str(checkpoints[-1]) if checkpoints else ""

        state = {
            "adapter_name": self._active_adapter_name,
            "inference_fallback_adapter": self._active_inference_fallback,
            "training_config_fingerprint": self._active_config_fingerprint,
            "keyed_pairs_fingerprint": self._active_keyed_pairs_fingerprint,
            "total_epochs": self._active_total_epochs,
            "last_completed_epoch": last_completed_epoch,
            "checkpoint_path": latest_checkpoint,
            "started_at": self._active_started_at,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            _write_resume_state_atomic(self._resume_state_path(), state)
            logger.debug(
                "Resume state written: adapter=%s epoch=%d/%d checkpoint=%s",
                self._active_adapter_name,
                last_completed_epoch,
                self._active_total_epochs,
                latest_checkpoint,
            )
        except Exception:
            logger.exception("Failed to write resume state — crash recovery disabled for this run")

    def _clear_resume_state(self) -> None:
        """Remove the resume state file after a successful training run.

        Safe to call even if the file does not exist.
        """
        try:
            self._resume_state_path().unlink(missing_ok=True)
        except Exception:
            logger.exception("Failed to remove resume state file")

    def _should_throttle_now(self, now: datetime | None = None) -> bool:
        """Whether the thermal throttle is active right now under the quiet-hours policy.

        Thin wrapper around ``is_thermal_policy_active`` that reads the fields
        the trainer was constructed with. See that helper for mode semantics.
        """
        return is_thermal_policy_active(
            self._quiet_hours_mode,
            self._quiet_hours_start,
            self._quiet_hours_end,
            now,
        )

    def _thermal_throttle(self, global_step: int):
        """Pause training if GPU temperature exceeds the configured limit.

        Keeps the machine quiet during background training — fans stay off
        when GPU temp stays below the ramp-up threshold. Training pauses
        until the GPU cools back down, then resumes automatically.

        Controlled by ``training_temp_limit`` (0 = disabled) and gated by
        the quiet-hours policy: outside quiet hours the throttle is a no-op
        even at hot temperatures, letting training run unthrottled when
        nobody is listening to fan noise.
        """
        if self._temp_limit <= 0:
            return
        if global_step % self._temp_check_interval != 0:
            return
        if not self._should_throttle_now():
            return

        temp = _gpu_temp()
        if temp is None or temp <= self._temp_limit:
            return

        from paramem.server.gpu_lock import acquire_gpu, release_gpu

        logger.info(
            "Thermal throttle: GPU at %d°C (limit %d°C) — releasing GPU at step %d",
            temp,
            self._temp_limit,
            global_step,
        )
        release_gpu()

        while temp is not None and temp > self._temp_limit:
            if self._shutdown_requested:
                acquire_gpu()
                return
            # Quiet-hours window may end mid-wait (e.g. 07:00 arrives) — in
            # that case we no longer care about fan noise and resume even if
            # still hot.
            if not self._should_throttle_now():
                logger.info(
                    "Thermal throttle: quiet-hours window ended at %d°C — resuming",
                    temp,
                )
                acquire_gpu()
                return
            time.sleep(5)
            temp = _gpu_temp()

        acquire_gpu()
        logger.info("Thermal throttle: GPU cooled to %d°C — resuming training", temp)

    def _run_jobs(
        self,
        jobs: list[TrainingJob],
        on_complete: Callable[[], None] | None,
    ):
        """Run training jobs sequentially in background thread.

        Holds the GPU lock during training to prevent concurrent CUDA access.
        Releases the lock between jobs for cooldown and to let STT/TTS through.
        """
        from paramem.server.gpu_lock import gpu_lock_sync

        try:
            all_complete = True
            for i, job in enumerate(jobs):
                if self._shutdown_requested:
                    logger.info("Shutdown — skipping remaining jobs")
                    all_complete = False
                    break

                # Cooldown between jobs (not before the first one)
                if i > 0:
                    self._cooldown_between_jobs()

                logger.info(
                    "Training job %d/%d: %s (%d keys)",
                    i + 1,
                    len(jobs),
                    job.adapter_name,
                    len(job.keyed_pairs),
                )
                with gpu_lock_sync():
                    self._train_adapter(job)

            if all_complete and on_complete:
                on_complete()

        except Exception:
            logger.exception("Background training failed")
            if self._on_error:
                self._on_error()
        finally:
            self._is_training = False

    def _cooldown_between_jobs(self):
        """Wait for GPU to cool between training jobs."""
        threshold = self._temp_limit if self._temp_limit > 0 else 52
        temp = _gpu_temp()
        if temp is None:
            logger.warning("Could not check GPU temperature — skipping cooldown")
            return
        if temp <= threshold:
            logger.info("GPU at %d°C — no cooldown needed", temp)
            return
        logger.info("GPU at %d°C — waiting for cooldown (target ≤%d°C)", temp, threshold)
        for _ in range(30):
            if self._shutdown_requested:
                break
            time.sleep(10)
            temp = _gpu_temp()
            if temp is not None and temp <= threshold:
                logger.info("GPU cooled to %d°C — resuming", temp)
                return
        logger.warning("Cooldown timeout — proceeding at %s°C", temp)

    def _train_adapter(self, job: TrainingJob):
        """Train a single adapter via the in_training staging slot.

        Flow (fresh start):
            1. Compute fingerprints and check for a valid resume state.
            2. Copy production adapter weights into in_training as the
               starting point (skipped when resuming from a checkpoint).
            3. Activate in_training (training modifies staging only).
            4. Train N epochs, writing resume_state.json after each epoch.
            5. On success: copy in_training → production, atomic save,
               then remove resume_state.json and bg_checkpoint/.
            6. On exception: leave resume state and checkpoint in place
               so the next restart can pick up from the last epoch.

        If training fails or the server crashes, the production adapter on
        disk is untouched — it still holds the last committed cycle's weights.
        """
        import shutil

        from paramem.models.loader import copy_adapter_weights, switch_adapter

        self._current_adapter = job.adapter_name
        self._current_job = job
        self._last_completed_epoch = 0

        if "in_training" not in self.model.peft_config:
            raise RuntimeError(
                "in_training staging adapter not found — was ConsolidationLoop initialized?"
            )

        # Compute fingerprints for this job and store them on self so the
        # epoch callback can write the state file without re-computing.
        kp_fingerprint = _fingerprint_keyed_pairs(job.keyed_pairs)
        cfg_fingerprint = _fingerprint_training_config(self.training_config, job.adapter_config)
        self._active_keyed_pairs_fingerprint = kp_fingerprint
        self._active_config_fingerprint = cfg_fingerprint
        self._active_total_epochs = self.training_config.num_epochs
        self._active_adapter_name = job.adapter_name
        self._active_inference_fallback = job.inference_fallback_adapter
        self._active_started_at = datetime.now(timezone.utc).isoformat()

        # Check for a valid resume state from a previous interrupted run.
        resume_checkpoint: str | None = None
        resume_state = _read_resume_state(self._resume_state_path())
        if resume_state is not None:
            state_kp_fp = resume_state.get("keyed_pairs_fingerprint", "")
            state_cfg_fp = resume_state.get("training_config_fingerprint", "")
            state_adapter = resume_state.get("adapter_name", "")
            state_checkpoint = resume_state.get("checkpoint_path", "")

            fingerprints_match = (
                state_kp_fp == kp_fingerprint
                and state_cfg_fp == cfg_fingerprint
                and state_adapter == job.adapter_name
            )
            checkpoint_valid = bool(state_checkpoint) and Path(state_checkpoint).is_dir()

            if fingerprints_match and checkpoint_valid:
                resume_checkpoint = state_checkpoint
                epoch_resumed = resume_state.get("last_completed_epoch", 0)
                self._last_completed_epoch = epoch_resumed
                logger.info(
                    "Resuming %s from epoch %d/%d (checkpoint=%s)",
                    job.adapter_name,
                    epoch_resumed,
                    self.training_config.num_epochs,
                    state_checkpoint,
                )
            else:
                # Stale state from a different job — wipe it and any old checkpoints.
                checkpoint_dir_stale = self.output_dir / "in_training" / "bg_checkpoint"
                if checkpoint_dir_stale.exists():
                    shutil.rmtree(checkpoint_dir_stale)
                self._clear_resume_state()
                if not fingerprints_match:
                    logger.info(
                        "Resume state fingerprint mismatch for %s — starting fresh",
                        job.adapter_name,
                    )
                else:
                    logger.info(
                        "Resume state checkpoint missing for %s — starting fresh",
                        job.adapter_name,
                    )

        if resume_checkpoint is None:
            # Fresh start: stage production weights into in_training.
            copy_adapter_weights(self.model, src=job.adapter_name, dst="in_training")

        switch_adapter(self.model, "in_training")
        self.model.train()

        examples = format_indexed_training(job.keyed_pairs, self.tokenizer, max_length=1024)
        dataset = _SimpleDataset(examples)

        if not examples:
            logger.info("No training examples for %s, skipping", job.adapter_name)
            return

        # Staging checkpoints go under in_training/ to avoid polluting production
        checkpoint_dir = self.output_dir / "in_training" / "bg_checkpoint"

        training_args = TrainingArguments(
            output_dir=str(checkpoint_dir),
            num_train_epochs=self.training_config.num_epochs,
            per_device_train_batch_size=self.training_config.batch_size,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            learning_rate=job.adapter_config.learning_rate,
            warmup_steps=self.training_config.warmup_steps
            if self.training_config.warmup_steps > 0
            else 0,
            warmup_ratio=self.training_config.warmup_ratio
            if self.training_config.warmup_steps == 0
            else 0.0,
            lr_scheduler_type=self.training_config.lr_scheduler_type,
            weight_decay=self.training_config.weight_decay,
            max_grad_norm=self.training_config.max_grad_norm,
            gradient_checkpointing=self.training_config.gradient_checkpointing,
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=2,
            report_to="none",
            run_name=f"bg-{job.adapter_name}",
            seed=self.training_config.seed,
            bf16=True,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
        )

        if self.training_config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=default_data_collator,
            callbacks=[_PauseForInferenceCallback(self)],
        )

        logger.info(
            "Training %s: %d examples, %d epochs%s",
            job.adapter_name,
            len(examples),
            self.training_config.num_epochs,
            f" (resume from epoch {self._last_completed_epoch})" if resume_checkpoint else "",
        )

        try:
            trainer.train(resume_from_checkpoint=resume_checkpoint)
        except Exception:
            # On training failure: leave resume state and checkpoint intact
            # so the next restart picks up from the last completed epoch.
            # Restore sane model state so inference isn't broken.
            logger.exception("Training failed for adapter %s — restoring state", job.adapter_name)
            try:
                switch_adapter(self.model, job.adapter_name)
                self.model.eval()
                if self.training_config.gradient_checkpointing:
                    self.model.gradient_checkpointing_disable()
            except Exception:
                logger.exception("Failed to restore model state after training error")
            raise

        # Commit to production first, then clean up resume artefacts.
        self._commit_staging_to_production(job.adapter_name)

        # Remove resume state and stale checkpoints only after the production
        # save succeeds.  If _commit_staging_to_production raises, the state
        # and checkpoint survive for the next restart.
        self._clear_resume_state()
        checkpoint_dir_done = self.output_dir / "in_training" / "bg_checkpoint"
        if checkpoint_dir_done.exists():
            shutil.rmtree(checkpoint_dir_done)

        self._last_completed_epoch = self.training_config.num_epochs
        logger.info("Training complete and committed: %s", job.adapter_name)

    def _commit_staging_to_production(self, production_name: str) -> None:
        """Atomically promote in_training weights to a production adapter.

        Order matters: weights must be copied BEFORE the atomic save so that
        save_pretrained serializes the just-committed production slot, not
        the pre-commit state.  A manifest is built and embedded in the slot
        so the startup validator can match it against the live registry hash.

        The BG trainer never mutates the registry (§2.4.3), so the manifest's
        ``registry_sha256`` is derived by reading the current on-disk registry
        bytes directly (no ``save_bytes``/``save_from_bytes`` split needed here).
        """
        from paramem.adapters.manifest import build_manifest_for
        from paramem.models.loader import atomic_save_adapter, copy_adapter_weights

        copy_adapter_weights(self.model, src="in_training", dst=production_name)

        registry_path = self.output_dir / "indexed_key_registry.json"
        kp_path = self.output_dir / production_name / "keyed_pairs.json"
        manifest = None
        try:
            manifest = build_manifest_for(
                self.model,
                self.tokenizer,
                production_name,
                registry_path=registry_path if registry_path.exists() else None,
                keyed_pairs_path=kp_path if kp_path.exists() else None,
            )
        except Exception:
            logger.warning(
                "_commit_staging_to_production: manifest build failed for %s — saving without",
                production_name,
            )

        target_dir = self.output_dir / production_name
        atomic_save_adapter(self.model, target_dir, production_name, manifest=manifest)


class _SimpleDataset:
    """Minimal dataset wrapper for pre-tokenized examples."""

    def __init__(self, examples: list[dict]):
        self._examples = examples

    def __len__(self):
        return len(self._examples)

    def __getitem__(self, idx):
        return self._examples[idx]
