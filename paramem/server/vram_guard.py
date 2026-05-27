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
import time
from contextlib import contextmanager
from typing import Iterator, MutableMapping

import torch

logger = logging.getLogger(__name__)


DEFAULT_PROCESS_FRACTION = 0.85


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


def check_vram_headroom(
    label: str,
    headroom_bytes: int,
    state: MutableMapping[str, object] | None = None,
) -> None:
    """Warn (do NOT abort) when free VRAM has dropped below the booked headroom.

    Drift detector. Reads ``torch.cuda.mem_get_info()[0]`` and compares
    against ``headroom_bytes`` — the operator-configured per-phase peak
    (``vram.vram_cache_headroom_gib``). The boot post-load gate guaranteed
    this much was free immediately after load; if it's gone now, something
    (STT/TTS swap, allocator fragmentation, an orphan process) consumed the
    buffer we reserved for KV cache + activations.

    Action on low headroom:
      - Log a WARNING with the label, current free, and the configured floor.
      - When ``state`` is provided, populate ``state["vram_low_headroom_warning"]``
        for :func:`paramem.server.attention._collect_vram_low_headroom_items`
        to surface in ``/status.attention``.

    Phases proceed regardless. Experiments showed that running below the
    KV-cache buffer reliably OOMs, but the cleanup is :func:`vram_scope`'s
    job; this function's role is operator visibility, not enforcement.

    No-op when CUDA is unavailable. ``mem_get_info`` faults are logged and
    swallowed — an unhealthy driver state is for :func:`vram_scope` to surface.
    """
    if not torch.cuda.is_available():
        return
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
    except Exception as exc:  # noqa: BLE001
        logger.warning("VRAM headroom check: mem_get_info failed at %s: %s", label, exc)
        return
    if free_bytes >= headroom_bytes:
        return
    logger.warning(
        "VRAM headroom low at %s: free %.2f GiB < configured headroom %.2f GiB "
        "(total %.2f GiB). KV-cache / activation buffer is being consumed; "
        "extraction may OOM. Reduce max_interim_count, voice GPU residency, "
        "or raise vram.vram_cache_headroom_gib.",
        label,
        free_bytes / 2**30,
        headroom_bytes / 2**30,
        total_bytes / 2**30,
    )
    if state is not None:
        state["vram_low_headroom_warning"] = {
            "label": label,
            "free_gib": free_bytes / 2**30,
            "headroom_gib": headroom_bytes / 2**30,
            "total_gib": total_bytes / 2**30,
            "observed_at": time.time(),
        }


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
    # Each release step in its own guard: a failure on one (e.g. a
    # broken-driver synchronize) must not skip the others. The Python-
    # side gc.collect runs even when CUDA is wedged, and the allocator
    # release calls are best-effort hygiene either way.
    if torch.cuda.is_available():
        try:
            # Force pending kernels to complete so the allocator sees the
            # post-kernel state. Without sync, mem_get_info can report
            # stale numbers when called immediately after a generate.
            torch.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001
            logger.warning("VRAM guard: synchronize failed: %s", exc)
    try:
        gc.collect()
    except Exception as exc:  # noqa: BLE001
        logger.warning("VRAM guard: gc.collect failed: %s", exc)
    try:
        if hasattr(torch._C, "_cuda_clearCublasWorkspaces"):
            torch._C._cuda_clearCublasWorkspaces()
    except Exception as exc:  # noqa: BLE001
        logger.warning("VRAM guard: clearCublasWorkspaces failed: %s", exc)
    try:
        torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001
        logger.warning("VRAM guard: empty_cache failed: %s", exc)


@contextmanager
def vram_measure(label: str) -> Iterator[MutableMapping[str, int]]:
    """Capture free-VRAM delta around a load operation.

    Yields a mutable dict; populated on exit with:
        free_before  — bytes free per mem_get_info() before the loader ran
        free_after   — bytes free after the loader ran (+ a sync)
        delta        — free_before - free_after (component VRAM cost)
        total        — device total bytes

    Logs delta at INFO with the label.

    On torch.cuda.OutOfMemoryError or CUDA driver fault (matching the
    existing _CUDA_DRIVER_FAULT_MARKERS), calls safe_empty_cache and
    re-raises as VramExhausted(label). On success, does NOT call
    safe_empty_cache (the caller may want the loaded tensors live).

    No-op when CUDA is unavailable: yields {free_before: 0, free_after: 0,
    delta: 0, total: 0}.
    """
    result: dict[str, int] = {"free_before": 0, "free_after": 0, "delta": 0, "total": 0}
    if not torch.cuda.is_available():
        yield result
        return
    try:
        free_before, total = torch.cuda.mem_get_info()
        result["free_before"] = free_before
        result["total"] = total
        yield result
    except torch.cuda.OutOfMemoryError as exc:
        logger.error("vram_measure: %s exhausted device memory: %s", label, exc)
        safe_empty_cache()
        raise VramExhausted(label) from exc
    except RuntimeError as exc:
        if _is_cuda_driver_fault(exc):
            logger.error(
                "vram_measure: %s hit CUDA driver fault (treating as VRAM exhausted): %s",
                label,
                exc,
            )
            safe_empty_cache()
            raise VramExhausted(label) from exc
        raise
    else:
        try:
            torch.cuda.synchronize()
        except Exception:  # noqa: BLE001
            pass
        free_after, _ = torch.cuda.mem_get_info()
        result["free_after"] = free_after
        delta = free_before - free_after
        result["delta"] = delta
        logger.info(
            "vram_measure[%s]: used %.0f MiB (free %d → %d MiB)",
            label,
            delta / (1024 * 1024),
            free_before >> 20,
            free_after >> 20,
        )
