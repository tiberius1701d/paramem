"""Process-side VRAM safety net.

Two integration points, one component:

1. :func:`apply_process_cap` — call once at server startup, before any
   GPU allocation. Sets ``torch.cuda.set_per_process_memory_fraction``
   so that allocator pressure that would otherwise push past the device
   ceiling becomes a Python ``torch.cuda.OutOfMemoryError`` instead of a
   dxgkrnl driver fault (which on WSL2 takes the VM down with it).

2. :func:`vram_scope` — context manager that wraps a single VRAM-
   allocating phase of a consolidation cycle (an extract session, the
   training step). On exit it always runs ``torch.cuda.empty_cache()``
   so fragmentation does not accumulate across phases. On
   ``torch.cuda.OutOfMemoryError`` it logs the phase label, clears the
   cache, and re-raises as :class:`VramExhausted` so the cycle handler
   aborts rather than silently continuing on indeterminate state.

The component is intentionally a no-op when CUDA is unavailable so test
suites and CPU-only environments are unaffected.
"""

from __future__ import annotations

import gc
import logging
from contextlib import contextmanager
from typing import Iterator

import torch

logger = logging.getLogger(__name__)


DEFAULT_PROCESS_FRACTION = 0.85

# Minimum free device memory required to begin a new chunk's extraction
# pipeline. Sized for an 8 GiB device with the 4-bit Mistral 7B base
# resident (~4 GB) plus the extraction chain's working set: a single
# 8K-token plausibility KV cache is ~1 GiB on Mistral 7B fp16 weights,
# QA-gen prefill adds another few hundred MiB. 1.5 GiB free at chunk
# entry leaves ~500 MiB margin after those two; below that we abort the
# chunk early instead of failing mid-generate. Operator can override via
# the ``min_free_bytes`` argument to :func:`assert_free_vram`.
_DEFAULT_MIN_FREE_VRAM_BYTES = 1_500 * 1024 * 1024  # 1.5 GiB


class VramExhausted(RuntimeError):
    """Raised when a guarded phase triggers ``torch.cuda.OutOfMemoryError``.

    The cycle handler must catch (or let propagate) and abort the cycle —
    do NOT continue, the failed phase leaves indeterminate model state.
    The first arg of the exception is the phase label (e.g. the
    extract session id, or ``"training"``) so operators can identify
    where the cycle died.
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


# RuntimeError messages that indicate the WSL2 dxgkrnl could not satisfy
# an allocation. These manifest as bare RuntimeError, not the clean
# ``torch.cuda.OutOfMemoryError`` PyTorch raises when its own allocator
# refuses. Treated as the same VRAM-overalloc class so the cycle handler
# sees a single failure mode and ``last_consolidation_error`` populates.
# Markers must be specific enough to not catch unrelated RuntimeErrors —
# a clean OOM still surfaces via the dedicated OutOfMemoryError branch
# below.
_CUDA_DRIVER_FAULT_MARKERS = (
    "device not ready",
    "CUDA driver error",
)


def _is_cuda_driver_fault(exc: BaseException) -> bool:
    """True if *exc* is a CUDA driver fault we treat as VRAM exhaustion."""
    if not isinstance(exc, RuntimeError):
        return False
    msg = str(exc)
    return any(marker in msg for marker in _CUDA_DRIVER_FAULT_MARKERS)


@contextmanager
def vram_scope(label: str) -> Iterator[None]:
    """Wrap a VRAM-allocating phase with hygiene + OOM containment.

    *label* identifies the phase for logging and for the
    :class:`VramExhausted` payload (typical values: an extract session
    id, ``"training"``).

    On exit (success or failure) calls ``torch.cuda.empty_cache()`` so
    fragmentation does not accumulate across phases. On
    ``torch.cuda.OutOfMemoryError`` — or a bare ``RuntimeError`` whose
    message matches a CUDA-driver-fault marker (WSL2 reports
    over-allocation as ``"device not ready"`` rather than a clean Python
    OOM) — logs *label* and re-raises as :class:`VramExhausted`.

    No-op when CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        yield
        return
    try:
        yield
    except torch.cuda.OutOfMemoryError as exc:
        logger.error("VRAM guard: phase %s exhausted device memory: %s", label, exc)
        safe_empty_cache()
        raise VramExhausted(label) from exc
    except RuntimeError as exc:
        if _is_cuda_driver_fault(exc):
            logger.error(
                "VRAM guard: phase %s hit CUDA driver fault (treating as VRAM exhausted): %s",
                label,
                exc,
            )
            safe_empty_cache()
            raise VramExhausted(label) from exc
        raise
    finally:
        safe_empty_cache()


def assert_free_vram(
    label: str,
    min_free_bytes: int = _DEFAULT_MIN_FREE_VRAM_BYTES,
) -> None:
    """Raise :class:`VramExhausted` if free device memory is below threshold.

    Pre-chunk watchdog: called at the entry of each extraction phase to
    short-circuit cycles that would otherwise hit a mid-generate driver
    fault. Cheap (single ``mem_get_info`` syscall) and always called
    before allocating work — the alternative is failing several seconds
    into a generate after a 6 K-token prefill.

    The free-bytes value comes from ``torch.cuda.mem_get_info()`` which
    returns ``(free, total)``. PyTorch's per-process cap (set by
    :func:`apply_process_cap`) does NOT influence this number — it
    reports physical device free, polluted by every CUDA consumer in the
    process (STT, TTS, base model, prior allocator pool). That pollution
    is exactly what we want to detect: a 1.5 GiB threshold means "there
    is not enough headroom for the next phase regardless of who is
    holding it". Caller is responsible for any prior eviction step.

    No-op when CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        return
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
    except Exception as exc:  # noqa: BLE001
        # mem_get_info can fault on an unhealthy driver state (the same
        # state we're trying to detect). Treat as exhausted.
        logger.error(
            "VRAM guard: mem_get_info failed at %s — treating as exhausted: %s",
            label,
            exc,
        )
        raise VramExhausted(label) from exc
    if free_bytes < min_free_bytes:
        logger.error(
            "VRAM guard: %s aborted — free %d bytes < threshold %d bytes (total %d)",
            label,
            free_bytes,
            min_free_bytes,
            total_bytes,
        )
        raise VramExhausted(label)


def safe_empty_cache() -> None:
    """Release reclaimable device memory across all known allocators.

    Three release steps in order, each addressing a distinct holder of
    GPU memory that the others can't reach:

    1. ``gc.collect()`` — drops Python references to tensors so the
       caching allocator's segments can become "unused". A Python-
       referenced ``past_key_values`` tuple still pinned to a local
       variable in a caller stack frame is in-use and not reclaimable
       without this step.
    2. ``torch._C._cuda_clearCublasWorkspaces()`` — releases cuBLAS
       workspaces, which are allocated **outside** PyTorch's caching
       allocator (held by libcublas itself). Each unique GEMM shape can
       trigger a fresh workspace allocation; across many ``generate``
       calls in an extraction cycle, these accumulate to hundreds of
       MiB. ``empty_cache`` cannot touch them. Verified empirically: a
       simulate-mode cycle on a frozen 4-bit base leaves ~280 MiB held
       in cuBLAS workspaces that this call returns to the device.
       Private API but stable across PyTorch 2.x.
    3. ``torch.cuda.empty_cache()`` — returns unused PyTorch allocator
       segments to the driver (the inactive split fragments).

    Public because callers outside :func:`vram_scope` need the same
    primitive — e.g. before a VRAM-headroom check decides whether to
    reload a previously-evicted engine, every reclaimable allocator
    must be flushed so the check sees true free device memory.
    Internally :func:`vram_scope` invokes this on exit.

    Best-effort hygiene — a failure here must not mask a more important
    upstream exception or block clean shutdown.
    """
    try:
        if torch.cuda.is_available():
            # Force pending kernels to complete so the allocator sees the
            # post-kernel state. Without sync, mem_get_info can report
            # stale numbers when called immediately after a generate.
            torch.cuda.synchronize()
        gc.collect()
        if hasattr(torch._C, "_cuda_clearCublasWorkspaces"):
            torch._C._cuda_clearCublasWorkspaces()
        torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001
        logger.warning("VRAM guard: empty_cache failed: %s", exc)
