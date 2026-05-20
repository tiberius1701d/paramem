"""Background training manager for continuous adapter learning.

Trains adapters during idle time, pausing for inference requests.
Saves full training state at each epoch boundary for crash recovery.

The model stays loaded — switching between training and inference is
just flag flips (model.eval/train, gradient checkpointing), no reload.
"""

import hashlib
import json
import logging
import queue
import threading
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from paramem.memory.entry import format_entry_training

# Re-exported for ``paramem/server/app.py`` (/status endpoint) which imports
# ``is_thermal_policy_active`` from this module by qualified name.  The
# canonical home is now ``paramem.training.thermal_throttle``; the shim keeps
# the existing import path working.
from paramem.training.thermal_throttle import (
    ThermalPolicy,
    _gpu_temp,
)
from paramem.training.thermal_throttle import (
    is_thermal_policy_active as is_thermal_policy_active,
)
from paramem.training.trainer import TrainingHooks, train_adapter
from paramem.utils.config import AdapterConfig, TrainingConfig

logger = logging.getLogger(__name__)

# Sentinel placed on _job_queue to signal the persistent callable worker to exit.
_WORKER_STOP = object()

# Name of the atomic resume state file written inside output_dir/in_training/.
_RESUME_STATE_FILE = "resume_state.json"


def _ensure_staging_shape_matches(model, target_config: AdapterConfig):
    """Rebuild the ``in_training`` staging slot when its LoRA shape differs from *target_config*.

    The slot is templated on ``episodic_config`` at startup
    (``ConsolidationLoop._ensure_adapters``). Production tiers don't all
    share the same shape — procedural targets ``attn+mlp`` while
    episodic/semantic target ``attn``-only. A training job for procedural
    needs the staging slot rebuilt to procedural's shape before
    ``copy_adapter_weights`` can succeed; subsequent jobs that match the
    current shape skip the rebuild (idempotent).

    Args:
        model: Live PeftModel with at least the ``in_training`` adapter.
        target_config: The job's adapter config — staging shape must match.

    Returns:
        Possibly-rebound PeftModel (``create_adapter`` may rebind on the
        re-add path; in practice the same instance is returned).
    """
    from paramem.models.loader import create_adapter

    current = model.peft_config["in_training"]
    current_modules = tuple(sorted(current.target_modules))
    target_modules = tuple(sorted(target_config.target_modules))
    if current_modules == target_modules and current.r == target_config.rank:
        return model
    logger.info(
        "Rebuilding in_training staging slot: target_modules %s → %s",
        list(current_modules),
        list(target_modules),
    )
    model.delete_adapter("in_training")
    return create_adapter(model, target_config, "in_training")


def _fingerprint_entries(entries: list[dict]) -> str:
    """Return a SHA-256 hex digest of the canonical serialisation of the training entries.

    Order-sensitive: two lists with the same items in different orders produce
    different fingerprints. This is intentional — the extraction output order
    encodes the key insertion sequence, so a genuinely identical job produces
    identical entries in the same order.

    Args:
        entries: Training entries (each dict carries key/subject/predicate/object).

    Returns:
        Lowercase hex SHA-256 string.
    """
    serialised = json.dumps(entries, sort_keys=True, ensure_ascii=False)
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
    """Write resume state to disk atomically (envelope-encrypted when a
    master key is set).  Routes through ``write_infra_bytes`` which handles
    the temp-file + rename sequence.

    Args:
        state_path: Absolute path to the target JSON file.
        state: Dict to serialise.

    Raises:
        OSError: If the write or rename fails.
    """
    from paramem.backup.encryption import write_infra_bytes

    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(state, indent=2, ensure_ascii=False).encode("utf-8")
    write_infra_bytes(state_path, payload)


def _read_resume_state(state_path: Path) -> dict | None:
    """Load resume state from disk — transparently decrypts age-wrapped
    content when the daily identity is loaded.

    Returns None if the file does not exist or cannot be parsed.

    Args:
        state_path: Path to the resume_state.json file.

    Returns:
        Parsed dict, or None.
    """
    from paramem.backup.encryption import read_maybe_encrypted

    if not state_path.exists():
        return None
    try:
        return json.loads(read_maybe_encrypted(state_path).decode("utf-8"))
    except Exception:
        logger.warning("Could not parse resume state at %s — discarding", state_path)
        return None


@dataclass
class TrainingJob:
    """A single adapter training job.

    Attributes:
        entries: Training entries for this adapter (each dict carries
            ``key/subject/predicate/object``).
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

    entries: list[dict]
    adapter_name: str
    adapter_config: AdapterConfig
    inference_fallback_adapter: str = "episodic"


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

    # Max seconds the training thread waits for an inference caller to signal
    # done before resuming.  Bound prevents a permanent stall when the caller
    # crashes between ``pause()`` and ``resume()``.
    _INFERENCE_TIMEOUT = 120.0

    def __init__(
        self,
        model,
        tokenizer,
        training_config: TrainingConfig,
        output_dir: str | Path = "data/ha/adapters",
        thermal_policy: ThermalPolicy | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.training_config = training_config
        self.output_dir = Path(output_dir)
        # ``thermal_policy`` is the single source of truth for the thermal
        # throttle.  ``None`` (the default for callers without a positive
        # ``training_temp_limit``) means no throttle install at the
        # ``train_adapter`` call site below — experiments and tests that
        # don't override the default get fast unthrottled runs by
        # construction.  Live-server only when a non-zero limit is set.
        self._thermal_policy = thermal_policy

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
        self._active_entries_fingerprint: str = ""
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

    def submit_and_wait(
        self,
        fn: Callable[[], None],
        *,
        inference_fallback_adapter: str = "episodic",
    ) -> None:
        """Submit *fn* to the BG worker and block until it completes.

        Equivalent to :meth:`submit` followed by a :class:`threading.Event`
        wait.  This is the single canonical pattern for callers that need
        synchronous completion semantics — it replaces the inline Event-wait
        boilerplate that previously appeared in both
        :func:`paramem.server.app._await_bg_cycle` and
        :meth:`paramem.training.consolidation.ConsolidationLoop.train_adapters`.

        Re-raises any exception thrown inside *fn* on the calling thread.

        Args:
            fn: Zero-argument callable to execute under the GPU lock.
            inference_fallback_adapter: Adapter to activate if ``pause()``
                is called while *fn* is executing.  Defaults to
                ``"episodic"``.

        Raises:
            Exception: Re-raises the first exception raised inside *fn*.
        """
        done_event = threading.Event()
        exc_holder: list[BaseException] = []

        def _wrapped() -> None:
            try:
                fn()
            except BaseException as exc:  # noqa: BLE001
                exc_holder.append(exc)
            finally:
                done_event.set()

        self.submit(_wrapped, inference_fallback_adapter=inference_fallback_adapter)
        done_event.wait()
        if exc_holder:
            raise exc_holder[0]

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
                entries=[],
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
        instead of timing out on the never-firing inference-yield hook
        installed by ``train_adapter`` via ``TrainingHooks``.

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

    def close(self, timeout: float = 5.0) -> None:
        """Stop the ephemeral callable-worker thread and release its resources.

        Intended for callers that construct a short-lived :class:`BackgroundTrainer`
        for a single synchronous job (e.g. the ``train_adapters`` helper in
        :class:`paramem.training.consolidation.ConsolidationLoop`).  The
        persistent callable-worker thread started by the first :meth:`submit`
        call is stopped by enqueueing the :data:`_WORKER_STOP` sentinel and
        joining the thread.

        Idempotent: calling ``close()`` on an instance that has already been
        closed (or that never started a worker) is a no-op.

        Production callers that keep a :class:`BackgroundTrainer` alive across
        many sessions (``paramem/server/app.py``) should **not** call this
        method — they intentionally keep the worker running for the process
        lifetime.

        Args:
            timeout: Seconds to wait for the worker thread to join.  If the
                thread is still alive after the timeout a warning is logged and
                the method returns — the daemon will be killed on process exit.
        """
        self._stop_callable_worker(timeout=timeout)

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
            "entries_fingerprint": self._active_entries_fingerprint,
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

    def _yield_to_inference(self, global_step: int) -> None:
        """Pause training while an inference request is being served.

        Wired into ``train_adapter`` via ``TrainingHooks.on_step_yield`` —
        invoked at every step boundary, BEFORE the thermal throttle runs at
        the same step (callback ordering locked in
        ``paramem/training/trainer.py``).  When an inference request arrives,
        releases the GPU lock so STT/TTS/inference can proceed, waits a
        bounded period for the caller to signal done, re-acquires the lock,
        and restores the ``in_training`` adapter (inference may have
        switched away).
        """
        if not self._inference_requested.is_set():
            return

        from paramem.models.loader import switch_adapter
        from paramem.server.gpu_lock import acquire_gpu, release_gpu

        logger.debug("Training paused for inference at step %d", global_step)
        release_gpu()
        self._training_paused.set()
        signalled = self._inference_done.wait(timeout=self._INFERENCE_TIMEOUT)
        if not signalled:
            logger.warning(
                "Inference did not signal done within %.0fs — resuming training",
                self._INFERENCE_TIMEOUT,
            )
        self._inference_done.clear()
        acquire_gpu()
        # Restore staging adapter as active — inference may have switched
        # to a production adapter. Training must resume on in_training.
        switch_adapter(self.model, "in_training")
        self._training_paused.clear()
        logger.debug("Training resumed at step %d (adapter=in_training)", global_step)

    def _persist_resume_state(self, epoch: int, output_dir: str) -> None:
        """Write the resume_state.json marker for crash-safe recovery.

        Wired into ``train_adapter`` via ``TrainingHooks.on_epoch_persist``.
        Updates the in-memory ``_last_completed_epoch`` book-keeping the
        ``/status`` endpoint exposes, then writes the on-disk marker.
        """
        self._last_completed_epoch = epoch
        self._write_resume_state(epoch, output_dir)

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
                    len(job.entries),
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
        threshold = self._thermal_policy.temp_limit if self._thermal_policy is not None else 52
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

        # Lazy-rebuild: if this job's tier has a different LoRA shape than the
        # current staging slot (e.g. procedural's attn+mlp vs episodic's
        # attn-only), drop and recreate the slot. Idempotent for same-shape
        # jobs (the common case).
        self.model = _ensure_staging_shape_matches(self.model, job.adapter_config)

        # Compute fingerprints for this job and store them on self so the
        # epoch callback can write the state file without re-computing.
        kp_fingerprint = _fingerprint_entries(job.entries)
        cfg_fingerprint = _fingerprint_training_config(self.training_config, job.adapter_config)
        self._active_entries_fingerprint = kp_fingerprint
        self._active_config_fingerprint = cfg_fingerprint
        self._active_total_epochs = self.training_config.num_epochs
        self._active_adapter_name = job.adapter_name
        self._active_inference_fallback = job.inference_fallback_adapter
        self._active_started_at = datetime.now(timezone.utc).isoformat()

        # Check for a valid resume state from a previous interrupted run.
        resume_checkpoint: str | None = None
        resume_state = _read_resume_state(self._resume_state_path())
        if resume_state is not None:
            state_kp_fp = resume_state.get("entries_fingerprint", "")
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

        examples = format_entry_training(job.entries, self.tokenizer, max_length=1024)
        dataset = _SimpleDataset(examples)

        if not examples:
            logger.info("No training examples for %s, skipping", job.adapter_name)
            return

        # Staging checkpoints go under in_training/ to avoid polluting production
        checkpoint_dir = self.output_dir / "in_training" / "bg_checkpoint"

        # Delegate the inner training step to the unified ``train_adapter``.
        # BG-specific concerns (inference yielding, resume-state persistence,
        # graceful shutdown) flow through ``TrainingHooks``.  Thermal throttle
        # installs only when ``self._thermal_policy is not None`` (= live
        # server with positive ``training_temp_limit``).  /dev/shm
        # encrypted-checkpoint materialization happens inside ``train_adapter``;
        # no duplicate /dev/shm handling here.
        #
        # ``logging_steps=10`` preserves the BG-side log volume that predates
        # the unification — the canonical default on ``TrainingConfig`` is 1
        # (matches the historical ``train_adapter`` hardcode).
        bg_training_config = replace(self.training_config, logging_steps=10)
        hooks = TrainingHooks(
            on_step_yield=self._yield_to_inference,
            on_epoch_persist=self._persist_resume_state,
            on_shutdown_check=lambda: self._shutdown_requested,
        )

        logger.info(
            "Training %s: %d examples, %d epochs%s",
            job.adapter_name,
            len(examples),
            self.training_config.num_epochs,
            f" (resume from epoch {self._last_completed_epoch})" if resume_checkpoint else "",
        )

        try:
            train_adapter(
                model=self.model,
                tokenizer=self.tokenizer,
                train_dataset=dataset,
                adapter_name="in_training",
                training_config=bg_training_config,
                adapter_config=job.adapter_config,
                wandb_config=None,
                output_dir=checkpoint_dir,
                run_name=f"bg-{job.adapter_name}",
                resume_from_checkpoint=resume_checkpoint,
                thermal_policy=self._thermal_policy,
                hooks=hooks,
            )
        except Exception:
            # On training failure: leave resume state and checkpoint intact
            # so the next restart picks up from the last completed epoch.
            # Restore sane model state so inference isn't broken — this is
            # BG-specific (production adapter must be re-activated; staging
            # leaves the model in ``in_training``).
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

        # Per-tier registry path (new layout): <adapter_dir>/<tier>/indexed_key_registry.json.
        _tier_dir = self.output_dir / production_name
        registry_path = _tier_dir / "indexed_key_registry.json"
        # Legacy global path fallback for upgrade compat.
        if not registry_path.exists():
            _legacy = self.output_dir / "indexed_key_registry.json"
            if _legacy.exists():
                registry_path = _legacy
        # Derive key_count from the on-disk registry if available; the BG
        # trainer reads the count from the registry (knowledge lives in the
        # adapter weights).
        _key_count: int | None = None
        if registry_path.exists():
            try:
                import json as _json

                _reg = _json.loads(registry_path.read_bytes())
                # New per-tier KeyRegistry schema: {active_keys: [...], ...}
                # Use active_keys list length; fall back to dict length for legacy registries.
                if isinstance(_reg, dict) and "active_keys" in _reg:
                    _key_count = len(_reg["active_keys"])
                elif isinstance(_reg, dict):
                    _key_count = len(_reg)
            except Exception:
                pass
        manifest = None
        try:
            manifest = build_manifest_for(
                self.model,
                self.tokenizer,
                production_name,
                registry_path=registry_path if registry_path.exists() else None,
                key_count=_key_count,
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
