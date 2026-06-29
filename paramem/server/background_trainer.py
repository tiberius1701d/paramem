"""Background training manager for continuous adapter learning.

Trains adapters during idle time, aborting cleanly for inference requests.

The model stays loaded — switching between training and inference is
just flag flips (model.eval/train, gradient checkpointing), no reload.

Abort-then-skip contract:
    When /chat arrives during BG training, ``abort_for_inference()`` sets
    the per-job abort flag.  The training hooks shutdown predicate sees it
    on the next ``on_step_end``, sets ``control.should_training_stop=True``,
    and HF Trainer exits after the current step.  ``train_adapter`` returns
    ``metrics["aborted"] = True``; the caller skips its post-train commit /
    registry mutations.  The production adapter on disk is untouched.
"""

import logging
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from paramem.server.vram_guard import safe_empty_cache
from paramem.training.thermal_throttle import (
    ThermalPolicy,
)
from paramem.training.trainer import TrainingHooks
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

            For interim-adapter jobs: ``"episodic"`` (the stable main).
            For main consolidation refresh jobs: the backed-up prior main
            adapter name.

            Defaults to ``"episodic"`` so existing callers that create
            ``TrainingJob`` without this field continue to work correctly.
    """

    entries: list[dict]
    adapter_name: str
    adapter_config: AdapterConfig
    inference_fallback_adapter: str = "episodic"


class BackgroundTrainer:
    """Manages background training that aborts for inference requests.

    Production training goes through ``submit()`` / ``submit_and_wait()``,
    which enqueue callables to a single persistent worker thread.  The
    ``start_jobs`` / ``_run_jobs`` pattern was dead code (no production
    callers) and has been removed.

    Usage:
        bt = BackgroundTrainer(model, tokenizer, config)
        bt.submit(lambda: run_consolidation_cycle(...))

        # When inference is needed:
        aborted = bt.abort_for_inference(timeout=30.0)
        # Training aborted at the next step boundary.
        # The caller checks metrics["aborted"] and skips its post-train commit.
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
        # (_run_callable_queue); cleared in their finally blocks.
        # The training_hooks_for_job factory captures these by closure.
        self._active_abort: threading.Event | None = None
        self._active_quiesced: threading.Event | None = None
        self._active_state_lock = threading.Lock()

        self._shutdown_requested = False
        self._is_training = False
        self._current_job: TrainingJob | None = None

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

    def abort_requested(self) -> bool:
        """Return True when the per-job abort event has been set.

        Called by the post-fold re-probe (``_build_store_contents``) to yield
        the GPU to a waiting ``/chat`` between adapter groups.  A partial probe
        is safe: registry and bookkeeping are always complete; the partial
        entries self-heal via on-miss probing on the next query.

        Returns False when no job is active (no-op case) or the abort flag has
        not been set.
        """
        with self._active_state_lock:
            abort = self._active_abort
        return abort is not None and abort.is_set()

    def abort_for_inference(self, timeout: float = 30.0) -> bool:
        """Signal the active training job to stop at the next step boundary.

        Sets the per-job abort flag.  The TrainingHooks shutdown predicate
        installed via training_hooks_for_job sees it on the next on_step_end,
        sets control.should_training_stop=True.  HF Trainer exits cleanly
        after the current step.  train_adapter returns metrics["aborted"]=True
        so the caller can skip its post-train commit / registry mutations.

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
            # Reset per-job: _shutdown_requested may have been set True by a
            # prior inference-abort or SIGTERM-on-a-different-job path.  For
            # the singleton reuse case that flag means "abort THIS job" — reset
            # it here so a stale flag from a previous job does not poison the
            # current one.  (The one terminal path — SIGTERM — exits the process
            # immediately, so this reset never runs in that case.)
            self._shutdown_requested = False
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
                # Reclaim transient VRAM reserved by the completed job (optimizer
                # state, gradient buffers, activation cache, cuBLAS workspaces).
                # Called AFTER _quiesced.set() so the abort-quiesce latency
                # contract is not affected.  safe_empty_cache releases only
                # unused reserved segments — live tensors (base model, tokenizer,
                # co-resident adapters held by Python refs) remain untouched.
                safe_empty_cache()

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
        - ``_shutdown_requested`` (set by direct attribute write or the former
          ``stop()`` callers, now replaced with direct flag assignment)
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

    def release(self) -> None:
        """Stop the worker and drop base-model references so the model can be freed.

        Called by :func:`paramem.server.app._release_base_model_in_process`.
        Breaks the worker cycle (via :meth:`_stop_callable_worker`, which now
        nulls ``_worker_thread`` after joining) and then nulls
        ``model``/``tokenizer``/``_current_job`` so no live attribute on this
        object retains a reference to the base model.

        Idempotent: safe to call on an instance whose worker was never started
        or that has already been stopped.
        """
        self._stop_callable_worker()
        self.model = None
        self.tokenizer = None
        self._current_job = None

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

        After joining, nulls ``self._worker_thread`` so the
        ``bt ↔ Thread._target (bound method)`` cycle dissolves on stop.
        CPython only deletes ``Thread._target`` after ``run()`` exits, and
        never clears our handle — without explicit nulling ``bt`` (and its
        ``bt.model`` = base model) remains reachable through the joined
        thread handle even after the queue is drained.  Setting the handle
        to ``None`` here is safe because :meth:`submit` lazily re-creates
        the worker when ``self._worker_thread is None``.

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
        # Null our handle so the bt ↔ Thread._target cycle dissolves on stop.
        self._worker_thread = None
