"""STAGING state machine and preview helpers for the ParaMem migration subsystem.

This module implements the server-side STAGING state (Slice 3b.1).  It covers
the in-memory stash of a candidate ``server.yaml``, diff computation, tier
classification, and shape-change detection.  **No files are written** — disk
writes, atomic swap, trial markers, and TRIAL state all ship in Slice 3b.2.

Design notes
------------
- ``MigrationStashState`` mirrors ``ConfigDriftState`` in ``drift.py`` as a
  TypedDict so the slot on ``_state["migration"]`` is self-describing.
- ``validate_candidate_path`` enforces that the candidate lives on the same
  filesystem as the live config (same ``st_dev``).  On WSL2, drvfs paths
  under ``/mnt/c/…`` share a single ``st_dev`` regardless of Windows drive
  letter, so this check is **best-effort**: it catches the most common mistake
  (passing a path on a different Linux mount) but cannot prevent a user from
  specifying a drvfs path that happens to share ``st_dev``.  The real safety
  net is Slice 3b.2's atomic ``rename()`` — if the paths are on different
  filesystems, ``rename()`` will raise ``OSError`` (EXDEV) at that point.
- Env-var template strings (e.g. ``${PARAMEM_DAILY_PASSPHRASE}``) are
  preserved verbatim in diffs — we use ``yaml.safe_load`` on raw bytes,
  NOT ``load_server_config``, so no env substitution occurs
  (see drift.py:13–15).
- All timestamps are ISO-8601 UTC via ``datetime.now(timezone.utc).isoformat()``.
"""

from __future__ import annotations

import difflib
import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Literal, TypedDict

import yaml

from paramem.adapters.manifest import (
    ManifestError,
    find_live_slot,
    read_manifest,
)
from paramem.config.classification import Tier, classify, walk_dict_leaves

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

MigrationStateLiteral = Literal["LIVE", "STAGING", "TRIAL"]

# ---------------------------------------------------------------------------
# TypedDicts for structured data
# ---------------------------------------------------------------------------


class TrialSlotPaths(TypedDict):
    """Absolute paths to the three pre-migration backup slot directories.

    Written into the trial marker and surfaced via ``/migration/status`` for
    the 3b.3 rollback path and Slice 6 restore-on-rollback.
    """

    config: str
    graph: str
    registry: str


class TrialStash(TypedDict):
    """In-memory mirror of ``TrialMarker`` stored on ``_state["migration"]["trial"]``.

    Populated by ``/migration/confirm`` (step 5) and by crash recovery
    (RESUME_TRIAL case).  Sentinel ``None`` in LIVE/STAGING.

    The ``gates`` sub-dict is initially ``{"status": "pending"}`` when the
    trial consolidation is running and updated to ``{"status": "no_new_sessions",
    "completed_at": <iso>}`` or ``{"status": "trial_exception", "exception": ...}``
    on completion.  Slice 4 will replace this with real gate evaluation.
    """

    started_at: str
    pre_trial_config_sha256: str
    candidate_config_sha256: str
    backup_paths: TrialSlotPaths
    trial_adapter_dir: str
    trial_graph_dir: str
    gates: dict


class TierDiffRow(TypedDict):
    """One row in the tier-classified field change list.

    Attributes
    ----------
    dotted_path:
        Dotted yaml path (e.g. ``"adapters.episodic.rank"``).
    old_value:
        Value in the live config, or ``None`` when the field is new.
    new_value:
        Value in the candidate config, or ``None`` when the field is removed.
    tier:
        String name of the impact tier (``"destructive"`` / ``"pipeline_altering"``
        / ``"operational"``).
    """

    dotted_path: str
    old_value: Any
    new_value: Any
    tier: str


class ShapeChange(TypedDict):
    """One field-level shape delta for a single adapter.

    Attributes
    ----------
    adapter:
        Adapter name (e.g. ``"episodic"``).
    field:
        LoRA field name (``"rank"``, ``"alpha"``, ``"target_modules"``, ``"dropout"``).
    old_value:
        Value currently in the on-disk ``meta.json``, or ``None`` when unknown.
    new_value:
        Value requested by the candidate config.
    consequence:
        Human-readable consequence string per the spec §L257–271.
    """

    adapter: str
    field: str
    old_value: Any
    new_value: Any
    consequence: str


class MigrationStashState(TypedDict):
    """Shape of the migration stash stored on ``_state["migration"]``.

    Mirrors ``ConfigDriftState`` from ``paramem.server.drift`` in TypedDict
    convention.  All fields are always present; sentinel values are used
    when no candidate is staged (``state="LIVE"``).

    Attributes
    ----------
    state:
        ``"LIVE"`` when no candidate is staged; ``"STAGING"`` when a
        candidate has been validated and stashed.
    candidate_path:
        Absolute path string of the staged candidate, or ``""`` in LIVE.
    candidate_hash:
        Full hex SHA-256 of the candidate file bytes, or ``""`` in LIVE.
    candidate_bytes:
        Raw bytes of the candidate file; ``b""`` in LIVE.  Kept in RAM so
        the diff can be re-rendered without a second disk read (``/migration/diff``).
    candidate_text:
        UTF-8 decoded text of the candidate; ``""`` in LIVE.
    parsed_candidate:
        Parsed YAML dict from ``yaml.safe_load``; ``{}`` in LIVE.
    staged_at:
        ISO-8601 UTC timestamp when STAGING was entered, or ``""`` in LIVE.
    simulate_mode_override:
        ``True`` when ``consolidation.mode == "simulate"`` in the candidate.
    shape_changes:
        List of ``ShapeChange`` rows; ``[]`` when none detected or in LIVE.
    tier_diff:
        List of ``TierDiffRow`` rows; ``[]`` in LIVE.
    unified_diff:
        Unified diff string; ``""`` in LIVE.
    """

    state: MigrationStateLiteral
    candidate_path: str
    candidate_hash: str
    candidate_bytes: bytes
    candidate_text: str
    parsed_candidate: dict
    staged_at: str
    simulate_mode_override: bool
    shape_changes: list[ShapeChange]
    tier_diff: list[TierDiffRow]
    unified_diff: str
    trial: "TrialStash | None"
    recovery_required: list[str]


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------


def initial_migration_state() -> MigrationStashState:
    """Return a LIVE shell for ``_state["migration"]``.

    Parallel to ``initial_drift_state()`` in ``paramem.server.drift``.  Called
    once from the lifespan bootstrap so the ``/migration/status`` endpoint can
    return immediately before any preview is requested.

    Returns
    -------
    MigrationStashState
        All fields set to their LIVE / empty sentinels.  ``state="LIVE"``,
        all string fields ``""``, bytes ``b""``, dict ``{}``, lists ``[]``,
        and ``simulate_mode_override=False``.
    """
    return MigrationStashState(
        state="LIVE",
        candidate_path="",
        candidate_hash="",
        candidate_bytes=b"",
        candidate_text="",
        parsed_candidate={},
        staged_at="",
        simulate_mode_override=False,
        shape_changes=[],
        tier_diff=[],
        unified_diff="",
        trial=None,
        recovery_required=[],
    )


# ---------------------------------------------------------------------------
# Path validation
# ---------------------------------------------------------------------------


def validate_candidate_path(path: str, live_config_path: Path) -> Path:
    """Validate and return the candidate path as a ``Path`` object.

    Enforces the rules from spec §L187 and §L282:

    1. Must be absolute (not a relative path).
    2. Must exist on the filesystem.
    3. Must be a regular file (not a directory, symlink-to-dir, etc.).
    4. Must be readable by the current process (``os.access(p, os.R_OK)``).
    5. Must be on the same filesystem as ``live_config_path`` (same
       ``st_dev``).

    **WSL2 note:** drvfs reports a single ``st_dev`` for all paths under
    ``/mnt/c/…`` regardless of Windows drive letter, so rule 5 is
    **best-effort** on WSL2.  The real safety net is Slice 3b.2's atomic
    ``os.rename()`` — ``EXDEV`` will fire at that point if the paths are on
    different filesystems.

    Parameters
    ----------
    path:
        Candidate path string supplied by the CLI caller.
    live_config_path:
        Absolute ``Path`` to the live ``configs/server.yaml``.

    Returns
    -------
    Path
        Validated absolute ``Path`` for the candidate file.

    Raises
    ------
    ValueError
        With a human-readable message for each rejection reason.  The
        HTTP layer maps ``ValueError`` to ``400 candidate_path_invalid``.
    """
    p = Path(path)

    if not p.is_absolute():
        raise ValueError(
            f"candidate_path must be absolute; got {path!r}. "
            "Pass an absolute path (e.g. /home/user/configs/server-new.yaml)."
        )

    if not p.exists():
        raise ValueError(f"candidate_path does not exist: {path!r}")

    if not p.is_file():
        raise ValueError(
            f"candidate_path is not a regular file: {path!r}. "
            "Must point to a plain file, not a directory or special file."
        )

    if not os.access(p, os.R_OK):
        raise ValueError(f"candidate_path is not readable: {path!r}")

    # Same-filesystem check — best-effort on WSL2 drvfs (see module docstring).
    try:
        candidate_dev = p.stat().st_dev
        live_dev = live_config_path.parent.stat().st_dev
    except OSError:
        # If stat fails on either path, let the caller deal with it at
        # backup/rename time.  Don't block preview on stat unavailability.
        pass
    else:
        if candidate_dev != live_dev:
            raise ValueError(
                f"candidate_path {path!r} is on a different filesystem than "
                f"the live config directory {live_config_path.parent!s}. "
                "The candidate must reside on the same filesystem as "
                "configs/server.yaml for the atomic rename in Slice 3b.2 to work."
            )

    return p


# ---------------------------------------------------------------------------
# Diff helpers
# ---------------------------------------------------------------------------


def compute_unified_diff(
    live_text: str,
    candidate_text: str,
    live_label: str = "server.yaml (live)",
    candidate_label: str = "server.yaml (candidate)",
) -> str:
    """Return a unified diff string comparing *live_text* to *candidate_text*.

    Wraps ``difflib.unified_diff`` with sensible defaults.  Output lines are
    joined with newlines; the trailing newline is stripped.

    Parameters
    ----------
    live_text:
        Content of the currently-active ``server.yaml``.
    candidate_text:
        Content of the candidate ``server.yaml``.
    live_label:
        Label used as the ``---`` header line.
    candidate_label:
        Label used as the ``+++`` header line.

    Returns
    -------
    str
        Unified diff string, or ``""`` when the contents are identical.
    """
    live_lines = live_text.splitlines(keepends=True)
    candidate_lines = candidate_text.splitlines(keepends=True)
    diff_lines = list(
        difflib.unified_diff(
            live_lines,
            candidate_lines,
            fromfile=live_label,
            tofile=candidate_label,
        )
    )
    return "".join(diff_lines).rstrip("\n")


# ---------------------------------------------------------------------------
# Leaf-walking helper (delegates to paramem.config.classification.walk_dict_leaves)
# ---------------------------------------------------------------------------


def _walk_leaves(node: object, prefix: str) -> list[tuple[str, Any]]:
    """Walk a nested dict and return a sorted ``(dotted_path, value)`` list.

    Thin wrapper around :func:`paramem.config.classification.walk_dict_leaves`
    that preserves the existing sorted-list return type expected by
    :func:`compute_tier_diff`.  The shared helper encapsulates the traversal
    rules for dynamic containers (``adapters``, ``sota_providers``, ``voices``).

    Parameters
    ----------
    node:
        The current yaml node (dict, list, or scalar).
    prefix:
        Accumulated dotted-path prefix (``""`` at the root call).

    Returns
    -------
    list[tuple[str, Any]]
        Sorted list of ``(dotted_path, value)`` pairs.
    """
    return sorted(walk_dict_leaves(node, prefix=prefix))


# ---------------------------------------------------------------------------
# Tier diff
# ---------------------------------------------------------------------------


def compute_tier_diff(live_yaml: dict, candidate_yaml: dict) -> list[TierDiffRow]:
    """Return a list of ``TierDiffRow`` for every differing leaf value.

    Walks both YAMLs and emits one row per dotted path whose value differs
    between *live_yaml* and *candidate_yaml*.  New paths (present only in
    candidate) and removed paths (present only in live) are included.

    Leaf values are compared with ``!=``.  List values (e.g.
    ``target_modules``) compare the lists directly.

    Parameters
    ----------
    live_yaml:
        Parsed dict of the live ``server.yaml``.
    candidate_yaml:
        Parsed dict of the candidate ``server.yaml``.

    Returns
    -------
    list[TierDiffRow]
        Rows sorted by tier (destructive first, then pipeline_altering, then
        operational) then by dotted_path within each tier.
    """
    live_leaves = dict(_walk_leaves(live_yaml, ""))
    cand_leaves = dict(_walk_leaves(candidate_yaml, ""))

    all_paths = sorted(set(live_leaves) | set(cand_leaves))

    rows: list[TierDiffRow] = []
    for dotted_path in all_paths:
        old_val = live_leaves.get(dotted_path)
        new_val = cand_leaves.get(dotted_path)
        if old_val == new_val:
            continue
        tier = classify(dotted_path)
        rows.append(
            TierDiffRow(
                dotted_path=dotted_path,
                old_value=old_val,
                new_value=new_val,
                tier=tier.value,
            )
        )

    # Sort: destructive → pipeline_altering → operational, then by path.
    _tier_order = {
        Tier.DESTRUCTIVE.value: 0,
        Tier.PIPELINE_ALTERING.value: 1,
        Tier.OPERATIONAL.value: 2,
    }
    rows.sort(key=lambda r: (_tier_order.get(r["tier"], 99), r["dotted_path"]))
    return rows


# ---------------------------------------------------------------------------
# Shape-change detection
# ---------------------------------------------------------------------------

# Canned consequence strings per field (spec §L257–271).
_SHAPE_CONSEQUENCE: dict[str, str] = {
    "rank": (
        "current adapter weights (trained at the old rank) will be discarded "
        "on migrate-accept. Prior recall is unrecoverable from weights. "
        "Registry entries remain; the new-shape adapter will retrain "
        "from the full key set on the next consolidation."
    ),
    "alpha": (
        "effective-rank scaling changes; retrain overwrites old weights. "
        "Same blast radius as a rank change."
    ),
    "target_modules": "same consequence.",
    "dropout": "same consequence.",
}


def compute_shape_changes(
    candidate_yaml: dict,
    adapter_dir: Path,
    live_registry_sha256: str,
) -> list[ShapeChange]:
    """Return shape-change rows for every enabled adapter that has a meta.json.

    For each adapter name whose ``adapters.<name>.enabled`` is ``True`` in
    *candidate_yaml*:

    1. Call ``find_live_slot(adapter_dir / name, live_registry_sha256)`` to
       locate the current on-disk slot.
    2. If no slot is found, skip silently (adapter not yet trained).
    3. If a slot exists but ``read_manifest`` raises, log WARN and skip (no
       row emitted).
    4. Compare ``manifest.lora.{rank, alpha, dropout, target_modules}``
       against the candidate config.  Emit one ``ShapeChange`` per differing
       field.

    Parameters
    ----------
    candidate_yaml:
        Parsed YAML dict from the candidate ``server.yaml``.
    adapter_dir:
        Filesystem path to the adapter directory
        (``config.adapter_dir`` / ``data/ha/adapters/``).
    live_registry_sha256:
        SHA-256 of the live ``key_metadata.json`` file, or ``""`` for a
        fresh install.

    Returns
    -------
    list[ShapeChange]
        All detected shape changes, ordered by adapter name then field name.
    """
    adapters_cfg = candidate_yaml.get("adapters", {})
    if not isinstance(adapters_cfg, dict):
        return []

    changes: list[ShapeChange] = []

    for adapter_name, adapter_vals in sorted(adapters_cfg.items()):
        if not isinstance(adapter_vals, dict):
            continue
        if not adapter_vals.get("enabled", False):
            continue

        kind_dir = adapter_dir / adapter_name
        slot = find_live_slot(kind_dir, live_registry_sha256)
        if slot is None:
            # Not yet trained — skip silently.
            continue

        try:
            manifest = read_manifest(slot)
        except ManifestError as exc:
            logger.warning(
                "compute_shape_changes: skipping adapter %r — cannot read manifest "
                "from slot %s: %s",
                adapter_name,
                slot,
                exc,
            )
            continue

        # Compare each shape field.
        for field in ("rank", "alpha", "dropout", "target_modules"):
            cand_val = adapter_vals.get(field)
            if cand_val is None:
                continue  # field not present in candidate — no change to surface
            manifest_val = getattr(manifest.lora, field)
            # Normalise target_modules: manifest stores tuple, yaml stores list.
            if field == "target_modules":
                cand_val_norm = tuple(sorted(cand_val)) if isinstance(cand_val, list) else cand_val
                manifest_val_norm = (
                    tuple(sorted(manifest_val)) if isinstance(manifest_val, tuple) else manifest_val
                )
                if cand_val_norm == manifest_val_norm:
                    continue
                display_old: Any = list(manifest_val)
                display_new: Any = list(cand_val)
            else:
                if cand_val == manifest_val:
                    continue
                display_old = manifest_val
                display_new = cand_val

            # Build the consequence string with the adapter name and before/after.
            consequence_base = _SHAPE_CONSEQUENCE.get(field, "same consequence.")
            if field == "rank":
                consequence = (
                    f"current adapter weights (trained at rank {display_old}) will be discarded "
                    "on migrate-accept. Prior recall is unrecoverable from weights. "
                    "Registry entries remain; the new-shape adapter will retrain "
                    "from the full key set on the next consolidation."
                )
            elif field == "alpha":
                consequence = (
                    f"effective-rank scaling changes (alpha {display_old} → {display_new}); "
                    "retrain overwrites old weights. Same blast radius as a rank change."
                )
            elif field == "target_modules":
                _old_mods = (
                    list(manifest_val) if isinstance(manifest_val, tuple) else list(manifest_val)
                )
                _new_mods = list(cand_val) if isinstance(cand_val, list) else list(cand_val)
                consequence = (
                    f"target_modules {{{','.join(str(x) for x in _old_mods)}}}"
                    f" → +{{{','.join(str(x) for x in _new_mods)}}}"
                    " — same consequence."
                )
            else:
                consequence = consequence_base

            changes.append(
                ShapeChange(
                    adapter=adapter_name,
                    field=field,
                    old_value=display_old,
                    new_value=display_new,
                    consequence=consequence,
                )
            )

    return changes


# ---------------------------------------------------------------------------
# Simulate-mode detection
# ---------------------------------------------------------------------------


def detect_simulate_mode(candidate_yaml: dict) -> bool:
    """Return ``True`` when the candidate sets ``consolidation.mode: simulate``.

    Parameters
    ----------
    candidate_yaml:
        Parsed YAML dict from the candidate ``server.yaml``.

    Returns
    -------
    bool
        ``True`` only when ``candidate_yaml["consolidation"]["mode"] == "simulate"``.
    """
    return candidate_yaml.get("consolidation", {}).get("mode") == "simulate"


# ---------------------------------------------------------------------------
# Preview response renderer
# ---------------------------------------------------------------------------


def render_preview_response(
    stash: MigrationStashState,
    *,
    pre_flight_fail: str | None = None,
) -> dict:
    """Return the ``PreviewResponse`` payload dict from a stash.

    Single source of truth for what ``/migration/preview`` and
    ``/migration/diff`` return.  Always includes ``pre_flight_fail`` — wired
    but dormant in Slice 3b.1 (disk-pressure pre-flight ships in Slice 3b.2).

    Parameters
    ----------
    stash:
        The current ``MigrationStashState`` (must be in ``"STAGING"`` state
        for a full response; callers are responsible for gating).
    pre_flight_fail:
        ``None`` in Slice 3b.1; Slice 3b.2 passes ``"disk_pressure"`` or
        similar when the pre-flight check fires.  Always included in the
        returned dict so downstream consumers can always check the field.

    Returns
    -------
    dict
        JSON-serialisable dict matching the ``PreviewResponse`` Pydantic schema.
    """
    return {
        "state": stash["state"],
        "candidate_path": stash["candidate_path"],
        "candidate_hash": stash["candidate_hash"],
        "staged_at": stash["staged_at"],
        "simulate_mode_override": stash["simulate_mode_override"],
        "unified_diff": stash["unified_diff"],
        "tier_diff": list(stash["tier_diff"]),
        "shape_changes": list(stash["shape_changes"]),
        "pre_flight_fail": pre_flight_fail,
    }


# ---------------------------------------------------------------------------
# Candidate parsing helper (internal, no file writes)
# ---------------------------------------------------------------------------


def _parse_candidate(candidate_bytes: bytes) -> dict:
    """Parse *candidate_bytes* as YAML and return the root dict.

    Uses ``yaml.safe_load`` on raw bytes so env-var template strings
    (``${PARAMEM_DAILY_PASSPHRASE}``) appear verbatim, matching
    drift-detection semantics (see module docstring / drift.py:13–15).

    Parameters
    ----------
    candidate_bytes:
        Raw bytes of the candidate ``server.yaml``.

    Returns
    -------
    dict
        Root mapping from the YAML document.

    Raises
    ------
    ValueError
        When ``yaml.safe_load`` raises or the document root is not a dict.
        The HTTP layer maps this to ``400 candidate_unparseable``.
    """
    try:
        data = yaml.safe_load(candidate_bytes)
    except yaml.YAMLError as exc:
        raise ValueError(f"candidate_path is not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"candidate_path YAML root is not a mapping (got {type(data).__name__})")
    return data


def _sha256_bytes(data: bytes) -> str:
    """Return lowercase hex SHA-256 of *data*."""
    return hashlib.sha256(data).hexdigest()
