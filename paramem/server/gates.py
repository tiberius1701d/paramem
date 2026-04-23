"""Trial-consolidation sanity gates (Slice 4, spec §Sanity suite L368–412).

Pure-Python module — no GPU-framework imports at top level.  All model /
tokenizer / registry objects are injected via keyword arguments typed as Any
so that ``import paramem.server.gates`` never pulls in torch, peft, or
transformers.

Gates run inside the server process, reusing the already-loaded base model
(8 GB VRAM constraint — no second model load).  They are called from
``_run_trial_consolidation`` in ``app.py`` after the trial
``run_consolidation`` call returns.

Public entry point: :func:`evaluate_gates`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Literal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GATE_4_SAMPLE_SIZE: int = 20
GATE_4_MIN_REGISTRY_SIZE: int = 20
GATE_4_THRESHOLD: float = 0.90  # ≥ 18/20 on the deciding sample
_TRIAL_PROBE_ADAPTER_NAME: str = "trial_probe"

# Bounded retry for the trial-probe mount on WSL2 transient CUDA failures
# (the trial coroutine fires gate evaluation immediately after training, so
# the driver may still be processing the last batch when load_adapter races
# to copy weights).
#
# WSL2 NOTE: when the first mount attempt fails with "device not ready",
# subsequent attempts hit "INTERNAL ASSERT FAILED in CUDACachingAllocator"
# because the failed allocation corrupts PyTorch's allocator bookkeeping.
# Retries cannot recover from that — so the strategy is to PREVENT the
# first attempt from failing via a sufficient wall-clock settle period
# (synchronize() drains queued kernels but does NOT wait long enough for
# the WSL2 driver state to fully recover after a heavy training pass).
_MOUNT_RETRY_COUNT: int = 3
_MOUNT_RETRY_BACKOFF_SECONDS: float = 1.0
# 10s settle covers a ~7s training-tail recovery window observed on this
# host (Mistral 7B + RTX 5070 + WSL2). Shorter waits (3s) reproducibly
# left the driver mid-recovery; the first load_adapter then surfaced
# "device not ready" → CUDA allocator corruption → retry-impossible.
_MOUNT_INITIAL_SETTLE_SECONDS: float = 10.0
# Errors that indicate corrupted PyTorch CUDA state — retrying is futile.
_CUDA_TERMINAL_MARKERS = ("INTERNAL ASSERT FAILED", "CUDACachingAllocator")

# Training-phase exception markers (WARNING W2 — logged loudly so
# mis-categorisation is immediately visible in the journal).
_TRAINING_MARKERS = (
    "train_loss",
    "nan",
    "OutOfMemoryError",
    "CUDA out of memory",
    "adapter_model",
    "save_pretrained",
    "TrainingArguments",
    "safetensors",
)


# ---------------------------------------------------------------------------
# GateResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GateResult:
    """Result of a single sanity gate.

    Attributes
    ----------
    gate:
        Ordinal index 1–4.
    name:
        Human-readable gate name: ``"extraction"``, ``"training"``,
        ``"adapter_reload"``, or ``"live_registry_recall"``.
    status:
        ``"pass"``, ``"fail"``, or ``"skipped"``.
    reason:
        Short human-readable explanation; ``None`` when status is ``"pass"``.
    metrics:
        Optional structured metrics dict (non-empty on gate 4 runs).
    """

    gate: int
    name: str
    status: Literal["pass", "fail", "skipped"]
    reason: str | None
    metrics: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict suitable for JSON embedding.

        Returns
        -------
        dict[str, Any]
            All fields present; ``reason`` and ``metrics`` may be ``None``.
        """
        return {
            "gate": self.gate,
            "name": self.name,
            "status": self.status,
            "reason": self.reason,
            "metrics": self.metrics,
        }


# ---------------------------------------------------------------------------
# Phase-aware exception categorisation (WARNING W2)
# ---------------------------------------------------------------------------


def _is_training_marker(exc: BaseException) -> bool:
    """Return True when the exception originates from the training phase.

    WARNING W2 — logs at WARNING level so mis-categorisation is visible
    in the journal and does not silently route a training exception to the
    extraction gate.

    Parameters
    ----------
    exc:
        Exception captured from ``run_consolidation``.

    Returns
    -------
    bool
        ``True`` when at least one training marker is present in the
        exception type name or its string representation.
    """
    exc_type = type(exc).__name__
    exc_str = str(exc)
    hit = any(m in exc_type or m in exc_str for m in _TRAINING_MARKERS)
    logger.warning(
        "phase-categorizer: exc_type=%s, match=%s, msg=%s",
        exc_type,
        hit,
        exc_str[:200],
    )
    return hit


# ---------------------------------------------------------------------------
# Deterministic registry sampling (spec L382)
# ---------------------------------------------------------------------------


def _sample_registry_keys(registry_content: bytes, *, seed_suffix: bytes = b"") -> list[str]:
    """Sample up to :data:`GATE_4_SAMPLE_SIZE` keys deterministically.

    The seed is derived from the first 16 hex chars of
    ``SHA-256(registry_content + seed_suffix)``, which is a 64-bit integer.
    Same registry content and same suffix always produce the same list
    (WARNING W1 — the 64-bit seed space is documented; callers that need
    independence between runs must supply a distinct ``seed_suffix``).

    Parameters
    ----------
    registry_content:
        Raw bytes of the registry JSON file.
    seed_suffix:
        Bytes appended before hashing to produce an independent sample.
        Use ``b"|retry"`` for the re-roll (spec L383).

    Returns
    -------
    list[str]
        Sorted population list sub-sampled deterministically.
    """
    payload = registry_content + seed_suffix
    seed_hex = hashlib.sha256(payload).hexdigest()[:16]
    seed_int = int(seed_hex, 16)
    rng = random.Random(seed_int)
    parsed = json.loads(registry_content)
    all_keys = sorted(parsed.keys())  # stable
    n = min(GATE_4_SAMPLE_SIZE, len(all_keys))
    return rng.sample(all_keys, n)


# ---------------------------------------------------------------------------
# Adapter mount helpers
# ---------------------------------------------------------------------------


def _resolve_adapter_mount_path(trial_adapter_dir: Path) -> Path:
    """Resolve the directory PEFT ``load_adapter`` should be pointed at.

    Trial training writes a Slice-3a per-adapter manifest layout::

        trial_adapter/<kind>/<YYYYMMDD-HHMMSS>/adapter_model.safetensors

    PEFT's ``load_adapter`` reads ``adapter_model.safetensors`` directly from
    the path it is given; it does NOT walk subdirectories. Passing
    ``trial_adapter`` (or even ``trial_adapter/<kind>``) fails with
    "adapter model file not found".

    Resolution order (per ``_ADAPTER_KIND_SUBDIRS``):

    1. ``trial_adapter/<kind>/<newest-slot>/`` containing the safetensors —
       Slice 3a per-adapter slot layout.
    2. ``trial_adapter/<kind>/`` directly containing the safetensors —
       legacy flat per-kind layout.
    3. ``trial_adapter/`` — legacy/simulated single-adapter top-level layout.
    """
    for kind in _ADAPTER_KIND_SUBDIRS:
        kind_dir = trial_adapter_dir / kind
        if not kind_dir.is_dir():
            continue
        # 1. Per-adapter slot layout: pick the newest non-hidden timestamped slot
        #    that contains adapter_model.safetensors.
        slots = [
            entry
            for entry in kind_dir.iterdir()
            if entry.is_dir()
            and not entry.name.startswith(".")
            and (entry / "adapter_model.safetensors").exists()
        ]
        if slots:
            return max(slots, key=lambda p: p.stat().st_mtime)
        # 2. Legacy flat per-kind layout.
        if (kind_dir / "adapter_model.safetensors").exists():
            return kind_dir
    # 3. Legacy/simulated top-level layout.
    return trial_adapter_dir


def _ensure_trial_probe_mounted(model: Any, trial_adapter_dir: Path, mount_state: dict) -> None:
    """Make a trial-trained adapter active so gates 3/4 can probe it.

    The trial coroutine just trained the adapters on this exact ``model``
    object — episodic/semantic/procedural are already in ``model.peft_config``
    with the trial weights. We only need to switch the active adapter to
    one of them; we do NOT need to add another one via ``load_adapter``.

    Adding a 5th adapter via ``load_adapter`` immediately after a heavy
    training pass deadlocks the WSL2 CUDA driver — the first attempt fails
    with ``cudaErrorNotReady`` and corrupts PyTorch's IPC handle table
    (``CUDACachingAllocator.cpp:419 INTERNAL ASSERT FAILED``), at which
    point only a server restart can recover. Switching to an in-memory
    adapter avoids the failure mode entirely while still verifying that
    training produced a usable adapter.

    Fallback: when the in-memory adapter is unavailable (e.g. lifespan
    crash recovery rebuilt the model fresh), fall back to ``load_adapter``
    from disk with the bounded-retry helper.

    Parameters
    ----------
    model:
        Loaded PeftModel (already in-memory on the server).
    trial_adapter_dir:
        Root directory of the trial adapter output. Per-kind subdirs
        (``episodic/``, ``semantic/``, ``procedural/``) live below this.
    mount_state:
        Shared dict used to communicate mount results between helpers.
        Modified in-place: sets ``"pre_active_adapter"``, ``"mounted"``,
        ``"mounted_via"`` ("set" or "load"), and ``"mounted_name"``.
    """
    raw_active = getattr(model, "active_adapter", None)
    if isinstance(raw_active, list):
        mount_state["pre_active_adapter"] = list(raw_active)
    elif raw_active is not None:
        mount_state["pre_active_adapter"] = [raw_active]
    else:
        mount_state["pre_active_adapter"] = []

    # CLAUDE.md: gradient_checkpointing must be disabled before model.generate()
    # — otherwise HF Transformers silently disables KV cache and produces
    # garbage output. The trial training pass that just finished left
    # checkpointing enabled; gate 3/4 will now generate(), so disable it
    # here and let _unmount_trial_probe restore the prior state.
    mount_state["pre_checkpointing"] = bool(getattr(model, "is_gradient_checkpointing", False))
    if mount_state["pre_checkpointing"]:
        try:
            model.gradient_checkpointing_disable()
        except Exception as exc:  # noqa: BLE001
            logger.warning("gradient_checkpointing_disable failed: %s", exc)
    mount_state["pre_training_mode"] = bool(getattr(model, "training", False))
    try:
        model.eval()
    except Exception as exc:  # noqa: BLE001
        logger.warning("model.eval() failed: %s", exc)

    # Diagnostic: surface the model's current adapter state at mount time
    # so we can debug "no adapter loaded" / wrong-wrapper cases on real GPUs.
    peft_config = getattr(model, "peft_config", None)
    pc_keys = list(peft_config.keys()) if isinstance(peft_config, dict) else None
    logger.info(
        "trial probe mount: model.type=%s peft_config=%s active=%s",
        type(model).__name__,
        pc_keys,
        getattr(model, "active_adapter", None),
    )

    # Preferred path: the trial just trained the adapter on this model;
    # switch to it instead of loading a 5th adapter from disk.
    in_memory_kind = _find_trained_kind_in_memory(model, trial_adapter_dir)
    if in_memory_kind is not None:
        model.set_adapter(in_memory_kind)
        mount_state["mounted"] = True
        mount_state["mounted_via"] = "set"
        mount_state["mounted_name"] = in_memory_kind
        logger.info("Activated in-memory trial adapter '%s'", in_memory_kind)
        return

    # Fallback: lifespan recovery, no in-memory adapter — load from disk.
    mount_path = _resolve_adapter_mount_path(trial_adapter_dir)
    _settle_cuda_and_load_adapter(model, mount_path)
    mount_state["mounted"] = True
    mount_state["mounted_via"] = "load"
    mount_state["mounted_name"] = _TRIAL_PROBE_ADAPTER_NAME
    logger.info("Mounted trial probe adapter from disk at %s", mount_path)


def _find_trained_kind_in_memory(model: Any, trial_adapter_dir: Path) -> str | None:
    """Return the kind name whose keyed_pairs.json exists AND whose adapter
    is in ``model.peft_config`` — the kind that gate 3 will probe.

    Gate 3 reads the first key from ``keyed_pairs.json`` and probes it;
    that probe MUST run against the adapter that was actually trained on
    those keys. Picking a different kind (e.g. activating ``episodic``
    when keyed_pairs lives in ``procedural/``) causes ``parse_failure``
    because the probed adapter has never seen the key.

    Strategy:
    1. Use ``_find_keyed_pairs`` to locate the keyed_pairs.json file the
       gate will probe — its parent directory name is the kind.
    2. Verify that kind is in ``model.peft_config``.
    3. Return that kind. ``None`` if either step fails — caller falls back
       to loading from disk.
    """
    peft_config = getattr(model, "peft_config", None)
    if peft_config is None:
        return None
    keyed_pairs_path = _find_keyed_pairs(trial_adapter_dir)
    if keyed_pairs_path is None:
        return None
    kind = keyed_pairs_path.parent.name
    if kind in _ADAPTER_KIND_SUBDIRS and kind in peft_config:
        return kind
    return None


def _settle_cuda_and_load_adapter(model: Any, mount_path: Path) -> None:
    """Drain CUDA + load + set the trial probe with bounded retries.

    The trial coroutine fires gate evaluation immediately after the training
    executor returns. On WSL2, the CUDA driver intermittently surfaces
    ``device not ready`` because gradient/optimizer kernels from the last
    training batch are still in flight when PEFT races to copy adapter
    weights to GPU. ``torch.cuda.synchronize()`` blocks until those finish;
    a small empty_cache + brief sleep + retry covers the residual cases
    where synchronize() returns but the next allocator call still races.
    """
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError:
        torch = None  # type: ignore[assignment]

    # Initial settle: WSL2 driver needs wall-clock time after a heavy
    # training pass before it can safely service a fresh load_adapter.
    # synchronize() alone returns too quickly to cover the gap. If the
    # first attempt fails, retries are unlikely to recover (failed CUDA
    # ops corrupt PyTorch's allocator bookkeeping) — so the strategy is
    # to PREVENT first-attempt failure.
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            logger.warning("pre-mount CUDA settle failed: %s", exc)
    time.sleep(_MOUNT_INITIAL_SETTLE_SECONDS)

    last_exc: BaseException | None = None
    for attempt in range(_MOUNT_RETRY_COUNT):
        # PEFT registers the adapter name in ``peft_config`` BEFORE moving
        # weights to GPU, so a CUDA failure during load_adapter leaves the
        # name half-registered. Subsequent attempts then fail with
        # "Adapter with name X already exists". Clean up before each try.
        peft_config = getattr(model, "peft_config", None)
        if peft_config is not None and _TRIAL_PROBE_ADAPTER_NAME in peft_config:
            try:
                model.delete_adapter(_TRIAL_PROBE_ADAPTER_NAME)
            except Exception as cleanup_exc:  # noqa: BLE001
                logger.warning("trial probe pre-attempt cleanup failed: %s", cleanup_exc)
        try:
            if torch is not None and torch.cuda.is_available():
                torch.cuda.synchronize()
            model.load_adapter(str(mount_path), adapter_name=_TRIAL_PROBE_ADAPTER_NAME)
            model.set_adapter(_TRIAL_PROBE_ADAPTER_NAME)
            return
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning(
                "trial probe mount attempt %d/%d failed: %s",
                attempt + 1,
                _MOUNT_RETRY_COUNT,
                exc,
            )
            # If PyTorch's allocator is corrupted, retries cannot recover.
            if any(marker in str(exc) for marker in _CUDA_TERMINAL_MARKERS):
                logger.error(
                    "trial probe mount: CUDA allocator corruption detected, "
                    "aborting retries — server restart required to recover"
                )
                break
            if torch is not None and torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                except Exception:  # noqa: BLE001
                    pass
            time.sleep(_MOUNT_RETRY_BACKOFF_SECONDS * (attempt + 1))
    assert last_exc is not None
    raise last_exc


def _unmount_trial_probe(model: Any, mount_state: dict) -> None:
    """Remove the trial probe adapter from the model.

    WARNING W3 — if ``"trial_probe"`` is the sole adapter loaded (edge
    case: server started with no live adapters), deleting it would leave
    PeftModel with no active config and a subsequent ``create_adapter``
    would crash with ``KeyError``.  In that case we skip the delete and
    log a WARN, leaving the sole adapter in place.

    Always restores the previously-active adapter when it is safe to do so.

    Parameters
    ----------
    model:
        PeftModel with ``"trial_probe"`` currently loaded.
    mount_state:
        Dict written by :func:`_ensure_trial_probe_mounted`; must contain
        ``"pre_active_adapter"`` (list of adapter names active before mount).
    """
    if not mount_state.get("mounted"):
        return

    mounted_via = mount_state.get("mounted_via", "load")
    mounted_name = mount_state.get("mounted_name", _TRIAL_PROBE_ADAPTER_NAME)

    # Restore gradient_checkpointing + training mode that the mount disabled.
    # Done before adapter cleanup so even if delete fails the trainer can
    # resume normally on the next cycle.
    if mount_state.get("pre_checkpointing"):
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("gradient_checkpointing_enable failed: %s", exc)
    if mount_state.get("pre_training_mode"):
        try:
            model.train()
        except Exception as exc:  # noqa: BLE001
            logger.warning("model.train() failed: %s", exc)

    try:
        # When mounted via set_adapter on an in-memory trial adapter, the
        # adapter is part of the trial state — DON'T delete it. Just restore
        # the pre-mount active adapter (if there was one).
        if mounted_via == "set":
            pre = mount_state.get("pre_active_adapter", [])
            if pre:
                model.set_adapter(pre[0])
            return

        # Loaded path: delete the adapter we added.
        peft_config = getattr(model, "peft_config", {})
        loaded_adapters = list(peft_config.keys())

        if len(loaded_adapters) <= 1:
            # WARNING W3 — sole adapter; skip delete to avoid broken PeftModel.
            logger.warning(
                "gates: skipping delete_adapter('%s') — it is the sole loaded adapter; "
                "leaving in place to preserve PeftModel integrity (CLAUDE.md rule).",
                mounted_name,
            )
            return

        model.delete_adapter(mounted_name)

        pre = mount_state.get("pre_active_adapter", [])
        if pre:
            model.set_adapter(pre[0])

    except Exception as exc:  # noqa: BLE001
        logger.error("gates: error during unmount of trial probe adapter: %s", exc)
    finally:
        mount_state["mounted"] = False


# ---------------------------------------------------------------------------
# Individual gates
# ---------------------------------------------------------------------------


def _gate_1_extraction(
    *,
    session_buffer_empty: bool,
    summary: dict | None,
    exc: BaseException | None,
) -> GateResult:
    """Gate 1 — trial extraction ran to completion.

    PASS when no exception occurred and extraction finished.
    SKIPPED when the session buffer was empty (nothing to extract).
    FAIL when an extraction-categorised exception was raised.

    Parameters
    ----------
    session_buffer_empty:
        True when the pending session queue was empty before the trial run.
    summary:
        Return value from ``run_consolidation``, or ``None`` when the call
        was skipped.
    exc:
        Exception captured from ``run_consolidation``, or ``None``.

    Returns
    -------
    GateResult
        Gate 1 result.
    """
    if session_buffer_empty:
        return GateResult(
            gate=1,
            name="extraction",
            status="skipped",
            reason="no_new_sessions",
            metrics=None,
        )

    if exc is not None:
        if _is_training_marker(exc):
            # Training-phase exception — extraction itself completed.
            return GateResult(
                gate=1,
                name="extraction",
                status="pass",
                reason=None,
                metrics=None,
            )
        return GateResult(
            gate=1,
            name="extraction",
            status="fail",
            reason=f"extraction exception: {type(exc).__name__}: {exc}",
            metrics=None,
        )

    return GateResult(gate=1, name="extraction", status="pass", reason=None, metrics=None)


def _gate_2_training(
    *,
    session_buffer_empty: bool,
    summary: dict | None,
    exc: BaseException | None,
    trial_adapter_dir: Path,
) -> GateResult:
    """Gate 2 — trial adapter trained successfully.

    PASS when ``summary["status"]`` is ``"complete"`` or ``"simulated"``
    AND the adapter directory contains at least one file.
    SKIPPED when the buffer was empty, or when ``status`` is
    ``"no_facts"`` / ``"no_pending"`` / ``"disabled"`` (no training
    was attempted).
    FAIL when a training-categorised exception was raised, OR when
    ``status=="complete"`` but adapter files are absent.

    Note: ``"no_facts"`` is SKIPPED (not PASS) because no training was
    attempted — the adapter is not touched in that path (REQUIRED FIX 2).

    Parameters
    ----------
    session_buffer_empty:
        True when the pending session queue was empty before the trial run.
    summary:
        Return value from ``run_consolidation``, or ``None``.
    exc:
        Exception captured from ``run_consolidation``, or ``None``.
    trial_adapter_dir:
        Directory where the trial adapter should have been written.

    Returns
    -------
    GateResult
        Gate 2 result.
    """
    _SKIP_STATUSES = {"no_facts", "no_pending", "disabled"}
    _PASS_STATUSES = {"complete", "simulated"}

    if session_buffer_empty:
        return GateResult(
            gate=2,
            name="training",
            status="skipped",
            reason="no_new_sessions",
            metrics=None,
        )

    if exc is not None:
        if _is_training_marker(exc):
            return GateResult(
                gate=2,
                name="training",
                status="fail",
                reason=f"training exception: {type(exc).__name__}: {exc}",
                metrics=None,
            )
        # Extraction exception — training was never reached.
        return GateResult(
            gate=2,
            name="training",
            status="skipped",
            reason="not_reached (extraction failed)",
            metrics=None,
        )

    status = (summary or {}).get("status", "")

    if status in _SKIP_STATUSES:
        return GateResult(
            gate=2,
            name="training",
            status="skipped",
            reason=f"no_training_attempted ({status})",
            metrics=None,
        )

    if status in _PASS_STATUSES:
        # Verify adapter files are present on disk.
        if not trial_adapter_dir.exists() or not any(trial_adapter_dir.iterdir()):
            return GateResult(
                gate=2,
                name="training",
                status="fail",
                reason=(
                    f"status=='{status}' but adapter directory is empty or missing: "
                    f"{trial_adapter_dir}"
                ),
                metrics=None,
            )
        return GateResult(gate=2, name="training", status="pass", reason=None, metrics=None)

    # Unknown / unexpected status — treat as fail so unusual consolidation
    # outcomes do not silently pass.
    return GateResult(
        gate=2,
        name="training",
        status="fail",
        reason=f"unexpected consolidation status: '{status}'",
        metrics=None,
    )


_ADAPTER_KIND_SUBDIRS = ("episodic", "semantic", "procedural")


def _find_keyed_pairs(trial_adapter_dir: Path) -> Path | None:
    """Locate ``keyed_pairs.json`` inside *trial_adapter_dir*.

    Real trial training writes per-kind subdirectories
    (``episodic/``, ``semantic/``, ``procedural/``), each containing its own
    ``keyed_pairs.json``.  This helper checks for the file in the following
    order:

    1. ``<trial_adapter_dir>/episodic/keyed_pairs.json`` (primary PA adapter,
       Decision 21).
    2. ``<trial_adapter_dir>/semantic/keyed_pairs.json``.
    3. ``<trial_adapter_dir>/procedural/keyed_pairs.json``.
    4. Top-level ``<trial_adapter_dir>/keyed_pairs.json`` (legacy / simulated
       — fallback only when no per-kind subdir is present).

    Fix 6 (2026-04-23): per-kind subdirs are checked BEFORE the top-level
    fallback.  A stale top-level file from an older production run (where
    episodic was written at the top level) no longer shadows fresher per-kind
    files written by a trial run.

    Returns the first path that exists, or ``None`` when no ``keyed_pairs.json``
    is found anywhere (B2-residual fix — real layout is per-kind subdirs).

    Parameters
    ----------
    trial_adapter_dir:
        Root directory of the trial adapter output.

    Returns
    -------
    Path | None
        Absolute path to ``keyed_pairs.json``, or ``None`` when absent.
    """
    # 1–3. Per-kind subdirectories in preference order (episodic is primary).
    for kind in _ADAPTER_KIND_SUBDIRS:
        candidate = trial_adapter_dir / kind / "keyed_pairs.json"
        if candidate.exists():
            return candidate

    # 4. Top-level fallback (legacy / simulate-mode output).
    top_level = trial_adapter_dir / "keyed_pairs.json"
    if top_level.exists():
        return top_level

    return None


def _gate_3_reload_smoke(
    *,
    session_buffer_empty: bool,
    summary: dict | None,
    model: Any,
    tokenizer: Any,
    trial_adapter_dir: Path,
    mount_state: dict,
) -> GateResult:
    """Gate 3 — trial adapter reload smoke test.

    Mounts the trial adapter, probes the first key from ``keyed_pairs.json``,
    and parses the result.  Uses in-server loader, not the experiment harness.

    SKIPPED when the session buffer was empty or when ``summary["status"]``
    is ``"no_facts"`` (no adapter was written, nothing to probe).  Also
    SKIPPED when no kind-specific adapter was trained (no ``keyed_pairs.json``
    in any expected location — e.g. ``no_facts`` extraction path).
    FAIL on mount raise, inference raise, parse failure, or unparseable
    ``keyed_pairs.json``.

    ``keyed_pairs.json`` search order (B2-residual fix):

    1. ``<trial_adapter_dir>/keyed_pairs.json`` — top-level (legacy / simulate).
    2. ``<trial_adapter_dir>/episodic/keyed_pairs.json`` — primary PA adapter.
    3. ``<trial_adapter_dir>/semantic/keyed_pairs.json``.
    4. ``<trial_adapter_dir>/procedural/keyed_pairs.json``.

    Any one of the three kind subdirs is sufficient to verify the adapter
    was loaded — the gate's purpose is ADAPTER LOAD verification, not
    enforcement of a specific kind.

    Parameters
    ----------
    session_buffer_empty:
        True when the pending session queue was empty before the trial run.
    summary:
        Return value from ``run_consolidation``, or ``None``.
    model:
        PeftModel already loaded in memory.
    tokenizer:
        Tokenizer matching the loaded model.
    trial_adapter_dir:
        Directory containing the trial adapter.
    mount_state:
        Shared mount-state dict passed to mount/unmount helpers.

    Returns
    -------
    GateResult
        Gate 3 result.
    """
    if session_buffer_empty:
        return GateResult(
            gate=3,
            name="adapter_reload",
            status="skipped",
            reason="no_new_sessions",
            metrics=None,
        )

    status = (summary or {}).get("status", "")
    if status == "no_facts":
        return GateResult(
            gate=3,
            name="adapter_reload",
            status="skipped",
            reason="no_facts — no adapter written",
            metrics=None,
        )

    # Locate keyed_pairs.json — check top-level then per-kind subdirs.
    keyed_pairs_path = _find_keyed_pairs(trial_adapter_dir)
    if keyed_pairs_path is None:
        return GateResult(
            gate=3,
            name="adapter_reload",
            status="skipped",
            reason=(
                "no kind-specific adapter trained — keyed_pairs.json absent "
                f"at {trial_adapter_dir} and all kind subdirs "
                f"({', '.join(_ADAPTER_KIND_SUBDIRS)})"
            ),
            metrics=None,
        )

    try:
        with open(keyed_pairs_path) as f:
            keyed_pairs = json.load(f)
        if not keyed_pairs:
            return GateResult(
                gate=3,
                name="adapter_reload",
                status="fail",
                reason="keyed_pairs.json is empty — no key to probe",
                metrics=None,
            )
        first_key = (
            keyed_pairs[0].get("key") if isinstance(keyed_pairs[0], dict) else keyed_pairs[0]
        )
    except Exception as exc:  # noqa: BLE001
        return GateResult(
            gate=3,
            name="adapter_reload",
            status="fail",
            reason=f"failed to read keyed_pairs.json: {exc}",
            metrics=None,
        )

    try:
        _ensure_trial_probe_mounted(model, trial_adapter_dir, mount_state)
    except Exception as exc:  # noqa: BLE001
        return GateResult(
            gate=3,
            name="adapter_reload",
            status="fail",
            reason=f"adapter mount failed: {exc}",
            metrics=None,
        )

    try:
        from paramem.training.indexed_memory import parse_recalled_pair, probe_key

        result = probe_key(model, tokenizer, first_key)
        if result is None or "failure_reason" in result:
            reason = (result or {}).get("failure_reason", "probe_key returned None")
            return GateResult(
                gate=3,
                name="adapter_reload",
                status="fail",
                reason=f"probe failed for key '{first_key}': {reason}",
                metrics=None,
            )

        # Verify the recalled pair can be parsed (belt-and-suspenders check).
        raw_output = result.get("raw_output", "")
        if parse_recalled_pair(raw_output) is None and "question" not in result:
            return GateResult(
                gate=3,
                name="adapter_reload",
                status="fail",
                reason=f"parse_recalled_pair returned None for key '{first_key}'",
                metrics=None,
            )

        return GateResult(gate=3, name="adapter_reload", status="pass", reason=None, metrics=None)

    except Exception as exc:  # noqa: BLE001
        return GateResult(
            gate=3,
            name="adapter_reload",
            status="fail",
            reason=f"inference error during reload smoke: {exc}",
            metrics=None,
        )


def _gate_4_recall_check(
    *,
    model: Any,
    tokenizer: Any,
    trial_adapter_dir: Path,
    live_registry_path: Path,
    mount_state: dict,
) -> GateResult:
    """Gate 4 — live-registry cross-adapter recall check.

    Samples :data:`GATE_4_SAMPLE_SIZE` keys from the current live registry
    deterministically (spec L382) and probes the trial adapter's recall.
    Threshold: ≥ 90% (≥ 18/20).

    One re-roll is permitted on first-sample failure: re-sample with
    ``seed_suffix=b"|retry"`` and require both samples to fail.  First-fail /
    second-pass → PASS with a cluster-variance warning.

    SKIPPED when the live registry has fewer than
    :data:`GATE_4_MIN_REGISTRY_SIZE` keys, or when ``trial_adapter_dir`` has
    no files (NO_NEW_SESSIONS — no trial adapter exists).

    GUARDRAIL G1 — the ``"sampled_keys"`` field in ``metrics`` is the deciding
    sample list.  Slice 5 uses the same list for the comparison report so the
    same 20 keys appear in both the hard-gate result and the report.

    Parameters
    ----------
    model:
        PeftModel already loaded in memory.
    tokenizer:
        Tokenizer matching the loaded model.
    trial_adapter_dir:
        Directory containing the trial adapter files.
    live_registry_path:
        Path to the current live ``registry.json``.
    mount_state:
        Shared mount-state dict passed to mount/unmount helpers.

    Returns
    -------
    GateResult
        Gate 4 result with full metrics dict.
    """
    from paramem.training.indexed_memory import probe_key, verify_confidence

    # --- Precondition: live registry must exist and have enough keys ---
    # SKIP on missing file (legitimate fresh-install OR pre-Slice-3a layout
    # without key_metadata.json). The CRITICAL #1 fix (2026-04-23) isolates
    # trial registry writes to state/trial_registry/, so trial-induced
    # corruption of the live file is no longer a concern.
    if not live_registry_path.exists():
        return GateResult(
            gate=4,
            name="live_registry_recall",
            status="skipped",
            reason=(
                f"live registry file not found: {live_registry_path} "
                "— treating as <20 keys (fresh install or pre-Slice-3a layout)"
            ),
            metrics=None,
        )

    registry_content = live_registry_path.read_bytes()
    try:
        registry_parsed = json.loads(registry_content)
    except Exception as exc:  # noqa: BLE001
        return GateResult(
            gate=4,
            name="live_registry_recall",
            status="fail",
            reason=f"failed to parse live registry: {exc}",
            metrics=None,
        )

    n_keys = len(registry_parsed)
    if n_keys < GATE_4_MIN_REGISTRY_SIZE:
        return GateResult(
            gate=4,
            name="live_registry_recall",
            status="skipped",
            reason=f"live registry has only {n_keys} keys (< {GATE_4_MIN_REGISTRY_SIZE})",
            metrics=None,
        )

    # --- Precondition: trial adapter must exist (NO_NEW_SESSIONS guard) ---
    if not trial_adapter_dir.exists() or not any(trial_adapter_dir.iterdir()):
        return GateResult(
            gate=4,
            name="live_registry_recall",
            status="skipped",
            reason="trial_adapter_dir is empty — no trial adapter (no_new_sessions case)",
            metrics=None,
        )

    # --- Ensure trial probe is mounted ---
    if not mount_state.get("mounted"):
        try:
            _ensure_trial_probe_mounted(model, trial_adapter_dir, mount_state)
        except Exception as exc:  # noqa: BLE001
            return GateResult(
                gate=4,
                name="live_registry_recall",
                status="fail",
                reason=f"mount failed: {exc}",
                metrics=None,
            )

    def _run_sample(suffix: bytes = b"") -> tuple[list[str], int, str]:
        """Return (sampled_keys, pass_count, seed_hex)."""
        seed_hex = hashlib.sha256(registry_content + suffix).hexdigest()[:16]
        keys = _sample_registry_keys(registry_content, seed_suffix=suffix)
        passed = 0
        for k in keys:
            result = probe_key(model, tokenizer, k, registry=registry_parsed)
            if result is not None and "failure_reason" not in result:
                conf = verify_confidence(result, registry_parsed)
                if conf >= GATE_4_THRESHOLD:
                    passed += 1
        return keys, passed, seed_hex

    # --- First sample ---
    first_keys, first_recalled, first_seed = _run_sample(b"")
    threshold_count = int(GATE_4_THRESHOLD * GATE_4_SAMPLE_SIZE)  # 18

    if first_recalled >= threshold_count:
        metrics: dict[str, Any] = {
            "recalled": first_recalled,
            "sampled": len(first_keys),
            "sampled_keys": first_keys,  # GUARDRAIL G1
            "seed": first_seed,
            "retried": False,
            "warnings": [],
            "first_sample_recalled": None,
            "first_sample_seed": None,
            "first_sample_keys": None,
        }
        return GateResult(
            gate=4,
            name="live_registry_recall",
            status="pass",
            reason=None,
            metrics=metrics,
        )

    # --- First sample failed → re-roll once ---
    retry_keys, retry_recalled, retry_seed = _run_sample(b"|retry")

    if retry_recalled >= threshold_count:
        # First-fail / second-pass → PASS with cluster-variance warning.
        metrics = {
            "recalled": retry_recalled,
            "sampled": len(retry_keys),
            "sampled_keys": retry_keys,  # GUARDRAIL G1 — deciding sample
            "seed": retry_seed,
            "retried": True,
            "warnings": ["cluster variance — first sample below threshold, retry passed"],
            "first_sample_recalled": first_recalled,
            "first_sample_seed": first_seed,
            "first_sample_keys": first_keys,
        }
        return GateResult(
            gate=4,
            name="live_registry_recall",
            status="pass",
            reason=None,
            metrics=metrics,
        )

    # Both samples failed.
    metrics = {
        "recalled": retry_recalled,
        "sampled": len(retry_keys),
        "sampled_keys": retry_keys,  # GUARDRAIL G1 — deciding sample
        "seed": retry_seed,
        "retried": True,
        "warnings": [],
        "first_sample_recalled": first_recalled,
        "first_sample_seed": first_seed,
        "first_sample_keys": first_keys,
    }
    return GateResult(
        gate=4,
        name="live_registry_recall",
        status="fail",
        reason=(
            f"recall below threshold on both samples: "
            f"first={first_recalled}/{len(first_keys)}, "
            f"retry={retry_recalled}/{len(retry_keys)} "
            f"(threshold={threshold_count}/{GATE_4_SAMPLE_SIZE})"
        ),
        metrics=metrics,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def evaluate_gates(
    *,
    model: Any,
    tokenizer: Any,
    trial_adapter_dir: Path,
    live_registry_path: Path,
    session_buffer_empty: bool,
    consolidation_summary: dict | None,
    consolidation_exception: BaseException | None,
) -> list[GateResult]:
    """Evaluate all four sanity gates for a trial consolidation run.

    Runs gates 1–4 in order.  Each gate is independent: a skip on gate 2
    does NOT skip gate 3 or 4.  Gate 4 always runs unless the live registry
    has fewer than :data:`GATE_4_MIN_REGISTRY_SIZE` keys or the trial
    adapter directory is empty (NO_NEW_SESSIONS case).

    The trial probe adapter is mounted once (lazily, on first gate that
    needs it) and unmounted after gate 4 regardless of outcome, ensuring
    no adapter state leaks out of this function.

    Parameters
    ----------
    model:
        PeftModel already loaded on the server.
    tokenizer:
        Tokenizer matching the loaded model.
    trial_adapter_dir:
        Path to the trial adapter output directory.  Passed directly from
        ``_state["migration"]["trial"]["trial_adapter_dir"]`` (REQUIRED FIX 1
        — not resolved via ``find_live_slot``).
    live_registry_path:
        Path to the current live ``registry.json`` (from pre-trial
        ``config.registry_path``).
    session_buffer_empty:
        True when the pending session queue was empty before the trial run.
    consolidation_summary:
        Return value from ``run_consolidation``, or ``None`` when the call
        was skipped (buffer empty) or raised.
    consolidation_exception:
        Exception captured from ``run_consolidation``, or ``None``.

    Returns
    -------
    list[GateResult]
        Exactly four results, one per gate, in order.
    """
    mount_state: dict[str, Any] = {"mounted": False, "pre_active_adapter": []}

    try:
        g1 = _gate_1_extraction(
            session_buffer_empty=session_buffer_empty,
            summary=consolidation_summary,
            exc=consolidation_exception,
        )
        g2 = _gate_2_training(
            session_buffer_empty=session_buffer_empty,
            summary=consolidation_summary,
            exc=consolidation_exception,
            trial_adapter_dir=trial_adapter_dir,
        )
        g3 = _gate_3_reload_smoke(
            session_buffer_empty=session_buffer_empty,
            summary=consolidation_summary,
            model=model,
            tokenizer=tokenizer,
            trial_adapter_dir=trial_adapter_dir,
            mount_state=mount_state,
        )
        g4 = _gate_4_recall_check(
            model=model,
            tokenizer=tokenizer,
            trial_adapter_dir=trial_adapter_dir,
            live_registry_path=live_registry_path,
            mount_state=mount_state,
        )
    finally:
        # Acceptance criterion D — trial_probe MUST NOT remain in
        # model.active_adapters after evaluate_gates returns on any path.
        _unmount_trial_probe(model, mount_state)

    return [g1, g2, g3, g4]


# ---------------------------------------------------------------------------
# TrialLogCapture — context manager for capturing WARN/ERROR/CRITICAL logs
# ---------------------------------------------------------------------------


class TrialLogCapture:
    """Context manager that captures WARNING/ERROR/CRITICAL records emitted
    while inside the ``with`` block.

    Attaches to the root logger so records from any submodule
    (consolidation, training, gates, extraction) are captured regardless
    of their logger name (all submodule loggers propagate to root by
    default).

    Counts records at or above the configured ``level`` and tracks
    distinct exception class names.  Class names are extracted from
    ``record.exc_info[0].__name__`` when the record carries exception
    information, or via best-effort parsing of ``record.exc_text`` when
    not.

    Thread safety: :meth:`logging.Logger.addHandler` and
    :meth:`logging.Logger.removeHandler` are thread-safe (wrapped by the
    logging lock internally).  The handler stores counts in local
    attributes updated only by ``emit``; reading ``metrics`` after
    ``__exit__`` is safe.

    Usage
    -----
    >>> with TrialLogCapture() as cap:
    ...     run_trial_things()
    >>> cap.metrics
    {'trial_log_errors': 2, 'distinct_classes': ['ValueError', 'KeyError']}

    Attributes
    ----------
    metrics:
        ``{"trial_log_errors": int, "distinct_classes": list[str]}`` —
        available after the context exits.
    """

    def __init__(self, *, level: int = logging.WARNING) -> None:
        """Initialise the capture context.

        Parameters
        ----------
        level:
            Minimum log level to count.  Defaults to ``logging.WARNING``.
            Records below this level are not counted.
        """
        self._level = level
        self._handler: logging.Handler | None = None
        self._count = 0
        self._distinct: list[str] = []  # insertion-order preserved
        self._distinct_set: set[str] = set()
        # Loggers whose propagate flag we temporarily enabled in __enter__,
        # keyed by logger name so __exit__ can restore them.
        self._forced_propagate: list[str] = []

    def __enter__(self) -> "TrialLogCapture":
        """Attach the capture handler to the root logger.

        Records from child loggers reach the root handler only when
        ``logger.propagate`` is ``True``.  Some environments (e.g. ROS 2's
        ``launch.logging`` package) install a custom ``loggerClass`` that sets
        ``propagate = False`` on every new logger.  To remain portable, this
        method temporarily re-enables propagation on all *existing* loggers
        that currently have it suppressed and records which loggers it changed
        so ``__exit__`` can restore the original state.

        Returns
        -------
        TrialLogCapture
            ``self`` for use in ``with ... as cap:`` syntax.
        """
        cap = self  # closure capture for the inner handler class

        class _CaptureHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:  # noqa: PLR0912
                cap._count += 1
                cls_name: str | None = None

                # Prefer exc_info — direct exception class name.
                if record.exc_info and record.exc_info[0] is not None:
                    cls_name = record.exc_info[0].__name__
                elif record.exc_text:
                    # Best-effort: the last line of a formatted traceback is
                    # usually ``ExcClass: message``.  The first line is the
                    # ``Traceback (most recent call last):`` header.
                    last_line = record.exc_text.splitlines()[-1] if record.exc_text else ""
                    if ":" in last_line:
                        cls_name = last_line.split(":", 1)[0].strip()

                if cls_name and cls_name not in cap._distinct_set:
                    cap._distinct_set.add(cls_name)
                    cap._distinct.append(cls_name)

        h = _CaptureHandler(level=self._level)
        logging.getLogger().addHandler(h)
        self._handler = h

        # Temporarily force propagate=True on all existing loggers that have
        # it disabled so their records reach the root handler above.
        manager = logging.Logger.manager
        for name, ref in list(manager.loggerDict.items()):
            # loggerDict values may be Logger instances or PlaceHolder objects.
            if isinstance(ref, logging.Logger) and not ref.propagate:
                ref.propagate = True
                self._forced_propagate.append(name)

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Remove the capture handler from the root logger and restore state.

        Re-applies ``propagate = False`` to any logger that was temporarily
        enabled in :meth:`__enter__`.  Does NOT swallow any exception raised
        inside the ``with`` block — the exception propagates normally after
        the handler is removed.

        Parameters
        ----------
        exc_type:
            Exception class, or ``None`` when no exception occurred.
        exc_val:
            Exception instance, or ``None``.
        exc_tb:
            Traceback, or ``None``.
        """
        if self._handler is not None:
            logging.getLogger().removeHandler(self._handler)
            self._handler = None

        # Restore propagate=False on any loggers we temporarily changed.
        manager = logging.Logger.manager
        for name in self._forced_propagate:
            ref = manager.loggerDict.get(name)
            if isinstance(ref, logging.Logger):
                ref.propagate = False
        self._forced_propagate.clear()

        # Return None (falsy) — exception is not swallowed.
        return None

    @property
    def metrics(self) -> dict[str, Any]:
        """Return the capture metrics.

        Safe to call at any time; outside the ``with`` block returns the
        final counts.

        Returns
        -------
        dict[str, Any]
            ``{"trial_log_errors": int, "distinct_classes": list[str]}``.
        """
        return {
            "trial_log_errors": self._count,
            "distinct_classes": list(self._distinct),
        }
