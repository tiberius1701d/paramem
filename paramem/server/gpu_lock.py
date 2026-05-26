"""Unified GPU lock for all CUDA access — inference, training, and STT/TTS.

Uses a threading.Lock so it protects across both asyncio coroutines (via
run_in_executor) and background threads (BackgroundTrainer, consolidation).

The asyncio wrapper (`async with gpu_lock`) acquires the thread lock in an
executor to avoid blocking the event loop.
"""

import asyncio
import logging
import threading
from contextlib import asynccontextmanager, contextmanager

logger = logging.getLogger(__name__)

_gpu_thread_lock = threading.Lock()


@asynccontextmanager
async def gpu_lock():
    """Async context manager — acquires the GPU lock without blocking the event loop."""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _gpu_thread_lock.acquire)
    try:
        yield
    finally:
        _gpu_thread_lock.release()


@contextmanager
def gpu_lock_sync(timeout: float = -1):
    """Synchronous context manager for background threads.

    Args:
        timeout: Seconds to wait. -1 = block forever (default).
    """
    acquired = _gpu_thread_lock.acquire(timeout=timeout)
    if not acquired:
        raise TimeoutError("Could not acquire GPU lock within timeout")
    try:
        yield
    finally:
        _gpu_thread_lock.release()


def acquire_gpu():
    """Acquire the GPU lock (blocking). For asymmetric acquire/release patterns."""
    _gpu_thread_lock.acquire()


def release_gpu():
    """Release the GPU lock. Must be paired with acquire_gpu()."""
    _gpu_thread_lock.release()


@contextmanager
def gpu_lock_released():
    """Yield the PA inference reservation for the body; reacquire on exit.

    For sites that already hold _gpu_thread_lock (e.g. a long-running
    training job) and need to hand the GPU to PA inference for a bounded
    handoff. The reacquire runs in `finally`, so the reservation is
    restored on every exit path (normal, return, exception).

    Caller invariant — MUST be enforced by usage discipline, not by this
    primitive: the calling thread already holds _gpu_thread_lock.
    `threading.Lock` is NOT owner-tracking:
    - If no thread holds the lock when release_gpu() runs, Python raises
      `RuntimeError: release unlocked lock`.
    - If a different thread holds the lock, release_gpu() silently steals
      ownership — the original holder will observe an inconsistent state
      at its own release. This is not detectable here.

    Use only from code paths that statically hold the reservation. After the
    BG-trainer abort refactor this primitive has no production callers; kept
    as a symmetric counterpart to ``gpu_lock_sync`` for future use.
    """
    release_gpu()
    try:
        yield
    finally:
        acquire_gpu()
