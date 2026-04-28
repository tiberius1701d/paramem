"""Process-side VRAM safety net.

Two integration points, one component:

1. :func:`apply_process_cap` — call once at server startup, before any
   GPU allocation. Sets ``torch.cuda.set_per_process_memory_fraction``
   so that allocator pressure that would otherwise push past the device
   ceiling becomes a Python ``torch.cuda.OutOfMemoryError`` instead of a
   dxgkrnl driver fault (which on WSL2 takes the VM down with it).

2. :func:`session_guard` — context manager that wraps a single
   consolidation extract call. On exit it always runs
   ``torch.cuda.empty_cache()`` so fragmentation does not accumulate
   across a multi-session cycle. On
   ``torch.cuda.OutOfMemoryError`` it logs the offending session id,
   clears the cache, and re-raises as :class:`VramExhausted` so the
   cycle handler aborts rather than silently continuing on indeterminate
   state.

The component is intentionally a no-op when CUDA is unavailable so test
suites and CPU-only environments are unaffected.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

import torch

logger = logging.getLogger(__name__)


DEFAULT_PROCESS_FRACTION = 0.85


class VramExhausted(RuntimeError):
    """Raised when an extract session triggers ``torch.cuda.OutOfMemoryError``.

    The cycle handler must catch (or let propagate) and abort the cycle —
    do NOT continue, the failed session leaves indeterminate model state.
    """


def apply_process_cap(
    fraction: float = DEFAULT_PROCESS_FRACTION,
    device: int = 0,
) -> None:
    """Set the per-process VRAM cap on the given CUDA device.

    Reserves ``(1 - fraction)`` of device memory as a bulkhead between
    PyTorch's allocator and the host GPU driver. Must be called before
    base-model load so the cap is in effect for the topology validator
    and every subsequent allocation.

    No-op when CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        return
    torch.cuda.set_per_process_memory_fraction(fraction, device=device)
    logger.info("VRAM guard: per-process cap set to %.2f on device %d", fraction, device)


@contextmanager
def session_guard(session_id: str) -> Iterator[None]:
    """Wrap a single extract session with VRAM hygiene + OOM containment.

    On exit (success or failure) calls ``torch.cuda.empty_cache()`` so
    inter-session fragmentation does not accumulate. On
    ``torch.cuda.OutOfMemoryError`` logs the session id and re-raises as
    :class:`VramExhausted`.

    No-op when CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        yield
        return
    try:
        yield
    except torch.cuda.OutOfMemoryError as exc:
        logger.error("VRAM guard: session %s exhausted device memory: %s", session_id, exc)
        _safe_empty_cache()
        raise VramExhausted(f"session {session_id} exhausted VRAM") from exc
    finally:
        _safe_empty_cache()


def _safe_empty_cache() -> None:
    """Call ``torch.cuda.empty_cache`` and swallow any error.

    The cache clear is best-effort hygiene. A failure here must not mask
    a more important upstream exception or block clean shutdown.
    """
    try:
        torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001
        logger.warning("VRAM guard: empty_cache failed: %s", exc)
