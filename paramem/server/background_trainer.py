"""Background training manager for continuous adapter learning.

Trains adapters during idle time, aborting cleanly for inference requests.
Saves full training state at each epoch boundary for crash recovery.

The model stays loaded — switching between training and inference is
just flag flips (model.eval/train, gradient checkpointing), no reload.

Abort-then-resume contract:
    When /chat arrives during BG training, ``abort_for_inference()`` sets
    the per-job abort flag.  The training hooks shutdown predicate sees it
    on the next ``on_step_end``, sets ``control.should_training_stop=True``,
    and HF Trainer exits after the current step.  ``_train_adapter`` detects
    the abort and skips the commit + resume-state clear so
    ``resume_state.json`` and ``bg_checkpoint/`` survive.  The next job
    submission with matching fingerprints resumes from the checkpoint.
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

# TrainingHooks is re-exported so consolidation.py and other callers can
# import it from this module (their existing import path).
__all__ = [
    "BackgroundTrainer",
    "TrainingJob",
    "TrainingHooks",
]

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
        # BG-specific save knobs: changing these changes the checkpoint cadence
        # and therefore invalidates an in-flight checkpoint from the old settings.
        "save_strategy_bg": training_config.save_strategy_bg,
        "save_steps_bg": training_config.save_steps_bg,
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
    """Manages background training that aborts for inference requests.

    Supports a queue of training jobs (e.g., episodic then procedural).
    Each job trains one adapter. Jobs run sequentially in a single thread.

    Usage:
        bt = BackgroundTrainer(model, tokenizer, config)
        bt.start_jobs([job1, job2], on_complete=save_callback)

        # When inference is needed:
        aborted = bt.abort_for_inference(timeout=30.0)
        # Training aborted at the next step boundary.
        # resume_state.json + bg_checkpoint/ survive for the next submission.

        bt.stop()        # graceful stop at next epoch boundary
    """

    def __init__(
        self,
        model,
        tokenizer,
        training_config: TrainingConfig,
        output_dir: str | Path = "data/ha/adapters",
        thermal_policy: ThermalPolicy | None = None,
        preload_cache: bool = False,
    ):
        # BASE-MODEL HOLDER (BackgroundTrainer): released via
        # _state["background_trainer"]=None + _stop_callable_worker() in
        # _release_base_model_in_process.
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
        # When preload_cache=True (InferenceConfig default), recall is served
        # from MemoryStore pre-loaded at boot.  In the abort branch, switching
        # the adapter back to the production slot is unnecessary because model
        # generation always uses model.disable_adapter() at inference sites.
        # Skip the adapter swap to save a PEFT call; eval +
        # gradient_checkpointing_disable still run regardless.
        self._preload_cache = preload_cache

        # Per-job abort state.  Recreated at the start of each job
        # (_train_adapter or _run_callable_queue); cleared in their finally
        # blocks.  The training_hooks_for_job factory captures these by closure.
        self._active_abort: threading.Event | None = None
        self._active_quiesced: threading.Event | None = None
        self._active_state_lock = threading.Lock()

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
        # Dedup key for _persist_resume_state: prevents double-writes when
        # on_save and on_epoch_end both fire at the same step (epoch boundary
        # under save_strategy="steps"). Reset to None at _train_adapter start.
        self._last_persist_key: tuple | None = None

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

    def abort_for_inference(self, timeout: float = 30.0) -> bool:
        """Signal the active training job to stop at the next step boundary.

        Sets the per-job abort flag.  The TrainingHooks shutdown predicate
        installed via training_hooks_for_job sees it on the next on_step_end,
        sets control.should_training_stop=True.  HF Trainer exits cleanly
        after the current step.  _train_adapter detects the abort and skips
        the commit + state-clear (resume_state.json + bg_checkpoint survive).
        The GPU lock is released as _run_callable_queue / _run_jobs exits
        its gpu_lock_sync block; _active_quiesced is set OUTSIDE the lock
        so the caller's subsequent ``async with gpu_lock()`` succeeds without
        racing on the BG thread still holding the lock.

        Returns True when a job was active and quiesced within ``timeout``.
        Returns False when no job was active (no-op) OR the wait timed out.

        Args:
            timeout: Seconds to wait for the training job to finish its
                current step and release the GPU lock.
        """
        with self._active_state_lock:
            abort = self._active_abort
            quiesced = self._active_quiesced
        if abort is None or quiesced is None:
            return False
        abort.set()
        return quiesced.wait(timeout=timeout)

    def stop(self, timeout: float = 60.0) -> int:
        """Stop training gracefully. Returns last completed epoch.

        Sets _shutdown_requested which is ORed into the training hooks
        shutdown predicate via training_hooks_for_job — so both the
        start_jobs and submit paths honour it.

        Args:
            timeout: Seconds to wait for the training thread to exit.
        """
        if not self._is_training:
            return self._last_completed_epoch

        self._shutdown_requested = True
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
        ``abort_for_inference()``.

        ``inference_fallback_adapter`` is stored as a sentinel
        :class:`TrainingJob` on ``self._current_job`` for bookkeeping.
        The abort branch reads it only when ``preload_cache=False``; when
        ``preload_cache=True`` (the default) the adapter swap is skipped
        because recall is served from MemoryStore.

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
        installed on ``self._current_job`` for bookkeeping.

        Per-job abort events are installed BEFORE the gpu_lock_sync block and
        cleared AFTER (in ``finally``).  ``_active_quiesced`` is set OUTSIDE
        ``gpu_lock_sync`` so ``abort_for_inference``'s caller can acquire the
        lock immediately after the wait returns — no lock contention.

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

            # Install a sentinel so the abort branch reads the correct fallback.
            sentinel = TrainingJob(
                entries=[],
                adapter_name="_callable_",
                adapter_config=AdapterConfig(),
                inference_fallback_adapter=fallback_adapter,
            )

            # Install per-job abort events BEFORE acquiring the GPU lock so
            # abort_for_inference() can signal even before fn() starts.
            with self._active_state_lock:
                self._active_abort = threading.Event()
                self._active_quiesced = threading.Event()
            self._current_job = sentinel
            self._is_training = True
            try:
                with gpu_lock_sync():
                    fn()
            except Exception:
                logger.exception("submit() job failed")
            finally:
                # Quiesce signal fires AFTER gpu_lock_sync exits so the caller's
                # next ``async with gpu_lock()`` succeeds without lock contention.
                _quiesced = self._active_quiesced
                with self._active_state_lock:
                    self._active_abort = None
                    self._active_quiesced = None
                self._current_job = None
                self._is_training = False
                if _quiesced is not None:
                    _quiesced.set()

    def _set_is_training(self, value: bool) -> None:
        """Override the ``_is_training`` flag from inside a submitted callable.

        Used by ``consolidate_interim_adapters`` to mark non-training phases
        (between per-tier ``_train_adapter`` calls; the entire finalize block)
        so ``abort_for_inference()`` short-circuits and ``/chat`` waits on the
        GPU lock alone — there's no in-flight HF training step to abort at
        those moments, so a no-op return is the correct contract.

        No locking is needed: the writer (this method, called from inside the
        callable that holds the GPU lock) and the reader
        (``abort_for_inference()``'s ``if not self._is_training: return False``
        gate via ``/chat``'s ``bg_trainer.is_training`` check at
        ``app.py:2516``) perform simple boolean operations that are atomic
        under the GIL, and the callable is the sole writer while the lock is
        held.

        Args:
            value: ``True`` to re-arm training mode before a ``_train_adapter``
                call; ``False`` to mark a non-training gap or the finalize block.
        """
        self._is_training = value

    def training_hooks_for_job(
        self,
        *,
        base_shutdown_predicate: Callable[[], bool] | None = None,
        on_step_yield: Callable[[int], None] | None = None,
        on_epoch_persist: Callable[[int, str], None] | None = None,
        on_save_persist: Callable[[int, str], None] | None = None,
    ) -> TrainingHooks:
        """Construct TrainingHooks whose shutdown predicate ORs all signals.

        The returned predicate returns True when ANY of these is true:
        - ``_shutdown_requested`` (set by ``stop()``)
        - ``base_shutdown_predicate()`` (caller's own gate, e.g. consolidation
          ``shutdown_requested``)
        - the per-job abort event (set by ``abort_for_inference()``)

        Single canonical site for wiring all three signals.  Every
        consolidation site should use this via
        ``ConsolidationLoop._build_training_hooks()`` rather than constructing
        ``TrainingHooks`` directly.

        Args:
            base_shutdown_predicate: Additional shutdown gate.  When ``None``
                (the default), only ``_shutdown_requested`` and the abort flag
                are checked.
            on_step_yield: Passed through to ``TrainingHooks`` unchanged.
                The inference-yield hook is no longer needed by
                ``BackgroundTrainer`` (abort replaced it) but consolidation
                callers may still supply one.
            on_epoch_persist: Passed through to ``TrainingHooks`` unchanged.
            on_save_persist: Passed through to ``TrainingHooks`` unchanged.

        Returns:
            A ``TrainingHooks`` instance with the composed shutdown predicate.
        """
        # Capture the current abort event by value so the closure holds the
        # right event even after _active_abort is replaced for a subsequent job.
        abort_ref: dict[str, threading.Event | None] = {"event": self._active_abort}

        def _shutdown_or_abort() -> bool:
            if self._shutdown_requested:
                return True
            if base_shutdown_predicate is not None and base_shutdown_predicate():
                return True
            evt = abort_ref["event"]
            return evt is not None and evt.is_set()

        return TrainingHooks(
            on_step_yield=on_step_yield,
            on_epoch_persist=on_epoch_persist,
            on_save_persist=on_save_persist,
            on_shutdown_check=_shutdown_or_abort,
        )

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

    def _persist_resume_state(self, epoch_or_step: int, output_dir: str) -> None:
        """Write the resume_state.json marker for crash-safe recovery.

        Wired into ``train_adapter`` via both ``TrainingHooks.on_epoch_persist``
        (epoch boundaries) and ``TrainingHooks.on_save_persist`` (HF Trainer
        checkpoint saves, which fire on epoch boundaries too when
        ``save_strategy="epoch"``).

        Deduplicates identical ``(epoch_or_step, output_dir)`` pairs so that
        concurrent epoch-boundary fires from ``on_epoch_end`` and ``on_save``
        produce only one disk write.

        Both callbacks pass ``state.global_step`` as the dedup key — the
        ``on_epoch_end`` path was aligned to ``global_step`` (not ``state.epoch``)
        so that both events agree on the key value at every epoch boundary.
        The stored value in ``resume_state.json["last_completed_epoch"]``
        is therefore ``state.global_step`` under both ``save_strategy="epoch"``
        and ``save_strategy="steps"``. Downstream readers (``app.py`` and
        ``consolidation.py``) use this for log-display only — no arithmetic — so
        the key name is kept unchanged for backward compatibility with existing
        on-disk checkpoints.
        """
        key = (epoch_or_step, output_dir)
        if self._last_persist_key == key:
            return
        self._last_persist_key = key
        self._last_completed_epoch = epoch_or_step
        self._write_resume_state(epoch_or_step, output_dir)

    def _run_jobs(
        self,
        jobs: list[TrainingJob],
        on_complete: Callable[[], None] | None,
    ):
        """Run training jobs sequentially in background thread.

        Holds the GPU lock during training to prevent concurrent CUDA access.
        Releases the lock between jobs for cooldown and to let STT/TTS through.
        Per-job abort events are installed inside ``_train_adapter`` for the
        start_jobs path (the submit path installs them in _run_callable_queue).
        _active_quiesced is set in this method's outer finally OUTSIDE the
        gpu_lock_sync block for the same ordering reason as _run_callable_queue.
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
                # _train_adapter installed and cleared per-job events; after the
                # gpu_lock_sync block exits, _active_quiesced was already fired
                # inside _train_adapter's finally (see that method).

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
            1. Install per-job abort events (idempotent if already installed by
               _run_callable_queue).
            2. Compute fingerprints and check for a valid resume state.
            3. Copy production adapter weights into in_training as the
               starting point (skipped when resuming from a checkpoint).
            4. Activate in_training (training modifies staging only).
            5. Train N epochs, writing resume_state.json after each epoch.
            6. On abort: restore inference-mode state; skip commit + state-clear
               so resume_state.json + bg_checkpoint/ survive for the next job.
            7. On success: copy in_training → production, atomic save,
               then remove resume_state.json and bg_checkpoint/.
            8. On exception: leave resume state and checkpoint in place
               so the next restart can pick up from the last epoch.

        If training fails or the server crashes, the production adapter on
        disk is untouched — it still holds the last committed cycle's weights.
        """
        import shutil

        from paramem.models.loader import copy_adapter_weights, switch_adapter

        self._current_adapter = job.adapter_name
        self._current_job = job
        self._last_completed_epoch = 0
        # Reset dedup cache so stale keys from a prior job do not suppress the
        # first persist of this job.
        self._last_persist_key = None

        # Install per-job abort events when called from the start_jobs path.
        # The _run_callable_queue path installs them before acquiring the GPU
        # lock; this branch handles the start_jobs path where _train_adapter
        # is called directly inside gpu_lock_sync.
        with self._active_state_lock:
            if self._active_abort is None:
                self._active_abort = threading.Event()
                self._active_quiesced = threading.Event()
                _installed_here = True
            else:
                _installed_here = False

        try:
            self._train_adapter_body(job, switch_adapter, copy_adapter_weights, shutil)
        finally:
            if _installed_here:
                # Quiesce signal fires AFTER gpu_lock_sync exits (caller's finally
                # runs before the lock is released) so abort_for_inference's caller
                # can acquire the GPU lock without racing on the still-held lock.
                _quiesced = self._active_quiesced
                with self._active_state_lock:
                    self._active_abort = None
                    self._active_quiesced = None
                if _quiesced is not None:
                    _quiesced.set()

    def _train_adapter_body(self, job: TrainingJob, switch_adapter, copy_adapter_weights, shutil):
        """Inner body of _train_adapter, separated for clarity.

        All abort and quiesced event management lives in _train_adapter.
        This method handles the training lifecycle: fingerprint, resume-state,
        staging setup, train, abort-or-commit.
        """
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
        # BG-specific concerns (abort-for-inference, resume-state persistence,
        # graceful shutdown) flow through ``TrainingHooks`` via
        # training_hooks_for_job.  Thermal throttle installs only when
        # ``self._thermal_policy is not None`` (= live server with positive
        # ``training_temp_limit``).  /dev/shm encrypted-checkpoint
        # materialization happens inside ``train_adapter``; no duplicate
        # /dev/shm handling here.
        #
        # ``logging_steps=10`` preserves the BG-side log volume that predates
        # the unification — the canonical default on ``TrainingConfig`` is 1
        # (matches the historical ``train_adapter`` hardcode).
        #
        # When ``save_strategy_bg`` is non-empty it overrides the default
        # "epoch" strategy. ``save_steps`` is clamped to ≥ 1 so HF
        # ``TrainingArguments`` never sees 0 (undefined at save_strategy="steps").
        overrides: dict = {"logging_steps": 10}
        if self.training_config.save_strategy_bg:
            overrides["save_strategy"] = self.training_config.save_strategy_bg
            if self.training_config.save_strategy_bg == "steps":
                overrides["save_steps"] = max(1, self.training_config.save_steps_bg)
        bg_training_config = replace(self.training_config, **overrides)
        hooks = self.training_hooks_for_job(
            on_epoch_persist=self._persist_resume_state,
            on_save_persist=self._persist_resume_state,
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

        # Check for abort AFTER train_adapter returns normally.  If the shutdown
        # predicate fired (abort or _shutdown_requested), HF Trainer exits the
        # training loop cleanly and train_adapter returns without raising.
        # Detect this by checking the abort event directly.
        with self._active_state_lock:
            _abort_evt = self._active_abort
        if _abort_evt is not None and _abort_evt.is_set():
            logger.info(
                "Training of %s aborted for inference at step boundary — "
                "resume state preserved for next submission",
                job.adapter_name,
            )
            # Restore inference-mode state exactly like the exception path.
            # CLAUDE.md mandates gradient_checkpointing_disable before generate().
            try:
                # When preload_cache=True, recall is served from MemoryStore and
                # generation uses model.disable_adapter() at the inference sites.
                # Skipping the adapter swap saves a PEFT call in the common case.
                if not self._preload_cache:
                    switch_adapter(self.model, job.adapter_name)
                self.model.eval()
                if self.training_config.gradient_checkpointing:
                    self.model.gradient_checkpointing_disable()
            except Exception:
                logger.exception("Failed to restore model state after abort")
            return  # leave resume_state.json + bg_checkpoint/ intact

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
