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


def _ensure_trial_probe_mounted(model: Any, trial_adapter_dir: Path, mount_state: dict) -> None:
    """Load the trial adapter under name ``"trial_probe"``.

    Records the active adapter(s) before loading so
    :func:`_unmount_trial_probe` can restore them.

    Parameters
    ----------
    model:
        Loaded PeftModel (already in-memory on the server).
    trial_adapter_dir:
        Directory containing the trial adapter files
        (``adapter_model.safetensors`` + ``adapter_config.json``).
    mount_state:
        Shared dict used to communicate mount results between helpers.
        Modified in-place: sets ``"pre_active_adapter"`` and ``"mounted"``.

    Raises
    ------
    Exception
        Re-raised from ``model.load_adapter`` on any failure.
    """
    # Capture the currently active adapter name(s) before mounting.
    raw_active = getattr(model, "active_adapter", None)
    if isinstance(raw_active, list):
        mount_state["pre_active_adapter"] = list(raw_active)
    elif raw_active is not None:
        mount_state["pre_active_adapter"] = [raw_active]
    else:
        mount_state["pre_active_adapter"] = []

    model.load_adapter(str(trial_adapter_dir), adapter_name=_TRIAL_PROBE_ADAPTER_NAME)
    model.set_adapter(_TRIAL_PROBE_ADAPTER_NAME)
    mount_state["mounted"] = True
    logger.info("Mounted trial probe adapter from %s", trial_adapter_dir)


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

    try:
        # Determine all currently-loaded adapter names.
        # PEFT exposes them via peft_config keys on a PeftModel.
        peft_config = getattr(model, "peft_config", {})
        loaded_adapters = list(peft_config.keys())

        if len(loaded_adapters) <= 1:
            # WARNING W3 — sole adapter; skip delete to avoid broken PeftModel.
            logger.warning(
                "gates: skipping delete_adapter('%s') — it is the sole loaded adapter; "
                "leaving in place to preserve PeftModel integrity (CLAUDE.md rule).",
                _TRIAL_PROBE_ADAPTER_NAME,
            )
            mount_state["mounted"] = False
            return

        model.delete_adapter(_TRIAL_PROBE_ADAPTER_NAME)

        # Restore previous active adapter.
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
    is ``"no_facts"`` (no adapter was written, nothing to probe).
    FAIL on mount raise, inference raise, parse failure, or missing
    ``keyed_pairs.json``.

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

    # Locate keyed_pairs.json inside the trial adapter directory.
    keyed_pairs_path = trial_adapter_dir / "keyed_pairs.json"
    if not keyed_pairs_path.exists():
        return GateResult(
            gate=3,
            name="adapter_reload",
            status="fail",
            reason=f"keyed_pairs.json not found at {keyed_pairs_path}",
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
    if not live_registry_path.exists():
        return GateResult(
            gate=4,
            name="live_registry_recall",
            status="fail",
            reason=f"live registry file not found: {live_registry_path}",
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
