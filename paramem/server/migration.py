"""STAGING state machine, candidate validation, and config promotion for migrations.

Covers the in-memory stash of a candidate ``server.yaml``, diff computation,
tier classification, shape-change detection, candidate construction/validation,
and the atomic promotion of a candidate over the live config.  **The preview
endpoint writes no files** — the pre-migration backup slot, the trial marker,
and the atomic swap are all driven from ``/migration/confirm``, which reaches
disk exclusively through :func:`backup_live_config` and :func:`promote_config`.

Design notes
------------
- ``promote_config`` is the **only** route by which the live ``server.yaml``
  changes.  It re-reads the candidate bytes from disk, re-checks them against the
  hash the operator previewed, constructs the candidate **as if it already sat at
  the live config path** (:func:`validate_candidate`), and only then renames.  A
  candidate that cannot boot therefore never becomes the live config.
- ``validate_candidate`` calls ``build_server_config`` — the construction stage
  boot itself runs — so validation cannot drift from boot.  It discards the
  resulting ``ServerConfig`` object at every call site: the config that goes live
  is the one boot (or ``_refresh_config_from_disk_into_state``) loads from disk.
- ``MigrationStashState`` mirrors ``ConfigDriftState`` in ``drift.py`` as a
  TypedDict so the slot on ``_state["migration"]`` is self-describing.
- ``validate_candidate_path`` enforces that the candidate lives on the same
  filesystem as the live config (same ``st_dev``).  On WSL2, drvfs paths
  under ``/mnt/c/…`` share a single ``st_dev`` regardless of Windows drive
  letter, so this check is **best-effort**: it catches the most common mistake
  (passing a path on a different Linux mount) but cannot prevent a user from
  specifying a drvfs path that happens to share ``st_dev``.  The real safety
  net is the atomic ``rename()`` in ``/migration/confirm`` — if the paths are on different
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
from paramem.backup.backup import write as backup_write
from paramem.backup.types import ArtifactKind
from paramem.config.classification import Tier, classify, walk_dict_leaves
from paramem.server.config import ServerConfig, build_server_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

MigrationStateLiteral = Literal["LIVE", "STAGING", "TRIAL"]

# ---------------------------------------------------------------------------
# TypedDicts for structured data
# ---------------------------------------------------------------------------


class TrialSlotPaths(TypedDict):
    """Absolute path to the pre-migration config backup slot directory.

    Config is the only required pre-migration artifact: the migration's sole
    live mutation is the atomic config swap, so rollback (and crash recovery)
    only ever restore the config.  Written into the trial marker and surfaced
    via ``/migration/status`` for the rollback path.
    """

    config: str


class TrialStash(TypedDict):
    """In-memory mirror of ``TrialMarker`` stored on ``_state["migration"]["trial"]``.

    Populated by ``/migration/confirm`` (step 5) and by crash recovery
    (RESUME_TRIAL case).  Sentinel ``None`` in LIVE/STAGING.

    The ``gates`` sub-dict is initially ``{"status": "pending"}`` when the
    trial consolidation is running and updated to ``{"status": "no_new_sessions",
    "completed_at": <iso>}`` or ``{"status": "trial_exception", "exception": ...}``
    on completion.  The gate evaluation layer updates this via ``_update_trial_gates``.
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
        Human-readable description of what the shape change means for trained weights and recall.
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
    parsed_live:
        Parsed YAML dict of the **live** config (``yaml.safe_load`` on the live
        ``server.yaml`` bytes at the time the candidate was staged); ``{}`` in
        LIVE or when the live config is absent.  Used by
        ``render_preview_response`` to derive the ``base_change`` block without
        re-reading the live file.
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
    parsed_live: dict


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
        parsed_live={},
    )


# ---------------------------------------------------------------------------
# Path validation
# ---------------------------------------------------------------------------


def validate_candidate_path(path: str, live_config_path: Path) -> Path:
    """Validate and return the candidate path as a ``Path`` object.

    Enforces the following rules:

    1. Must be absolute (not a relative path).
    2. Must exist on the filesystem.
    3. Must be a regular file (not a directory, symlink-to-dir, etc.).
    4. Must be readable by the current process (``os.access(p, os.R_OK)``).
    5. Must be on the same filesystem as ``live_config_path`` (same
       ``st_dev``).

    **WSL2 note:** drvfs reports a single ``st_dev`` for all paths under
    ``/mnt/c/…`` regardless of Windows drive letter, so rule 5 is
    **best-effort** on WSL2.  The real safety net is the atomic
    ``os.rename()`` in ``/migration/confirm`` — ``EXDEV`` will fire at that
    point if the paths are on different filesystems.

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
                "configs/server.yaml for the atomic rename in /migration/confirm to work."
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

# Canned consequence strings per field — describes the blast radius of each shape change.
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
# Mode-switch block helper
# ---------------------------------------------------------------------------


def _build_mode_switch_block(old_mode: str, new_mode: str) -> dict:
    """Build the ``mode_switch`` transparency block for a pure-mode-change confirm.

    Returned in the ``ConfirmResponse`` and ``PreviewResponse`` so the CLI and
    any API consumer understand that this migration was applied directly (no
    trial/accept/rollback) through the active-store rebuild mechanism.

    Parameters
    ----------
    old_mode:
        The ``consolidation.mode`` value being replaced (e.g. ``"simulate"``).
    new_mode:
        The ``consolidation.mode`` value in the candidate config (e.g. ``"train"``).

    Returns
    -------
    dict
        JSON-serialisable dict with ``from``, ``to``, ``direction``,
        ``applies_via``, and ``semantics`` keys.
    """
    direction = f"{old_mode}_to_{new_mode}"
    return {
        "from": old_mode,
        "to": new_mode,
        "direction": direction,
        "applies_via": "active_store_migration",
        "semantics": (
            "per-tier rebuild with a 1.0 recall gate and source-mode "
            "fallback until all tiers pass; no trial/accept/rollback"
        ),
    }


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
    ``/migration/diff`` return.  Always includes ``pre_flight_fail``.

    The ``base_change`` key is populated when the ``model:`` field differs
    between the live and candidate configs.  It is ``None`` otherwise.

    Parameters
    ----------
    stash:
        The current ``MigrationStashState`` (must be in ``"STAGING"`` state
        for a full response; callers are responsible for gating).
    pre_flight_fail:
        ``None`` when no pre-flight check fires; ``"disk_pressure"`` or
        similar when a pre-flight check rejects the preview.  Always included
        in the returned dict so downstream consumers can always check the field.

    Returns
    -------
    dict
        JSON-serialisable dict matching the ``PreviewResponse`` Pydantic schema.
    """
    tier_diff = list(stash["tier_diff"])
    # Derive mode_switch block when this is a pure mode-only change so the
    # CLI and API consumers know the confirm path (LIVE, no trial).
    mode_switch: dict | None = None
    if len(tier_diff) == 1 and tier_diff[0]["dotted_path"] == "consolidation.mode":
        mode_switch = _build_mode_switch_block(
            tier_diff[0]["old_value"],
            tier_diff[0]["new_value"],
        )
    # Derive base_change block when the candidate changes the base model.
    live_yaml = stash.get("parsed_live", {}) or {}
    candidate_yaml = stash.get("parsed_candidate", {}) or {}
    base_change: dict | None = compute_base_change(live_yaml, candidate_yaml)
    return {
        "state": stash["state"],
        "candidate_path": stash["candidate_path"],
        "candidate_hash": stash["candidate_hash"],
        "staged_at": stash["staged_at"],
        "simulate_mode_override": stash["simulate_mode_override"],
        "unified_diff": stash["unified_diff"],
        "tier_diff": tier_diff,
        "shape_changes": list(stash["shape_changes"]),
        "pre_flight_fail": pre_flight_fail,
        "mode_switch": mode_switch,
        "base_change": base_change,
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


# ---------------------------------------------------------------------------
# Candidate validation + atomic promotion
# ---------------------------------------------------------------------------


class CandidateConfigInvalid(ValueError):
    """The candidate ``server.yaml`` parses but cannot be constructed into a config.

    Raised by :func:`validate_candidate`.  The HTTP layer maps it to
    ``400 candidate_invalid_config`` (preview) / ``409 candidate_invalid_config``
    (confirm).  The message is the original construction error verbatim.
    """


class CandidateChanged(ValueError):
    """The candidate file on disk no longer hashes to what the operator previewed.

    Raised by :func:`promote_config` before any mutation.  The HTTP layer maps it
    to ``409 candidate_changed``: what was previewed is what gets promoted, or
    nothing gets promoted.
    """


def validate_candidate(candidate_bytes: bytes, live_config_path: Path) -> ServerConfig:
    """Construct the candidate config as if it already sat at the live config path.

    Runs the same construction stage boot runs (``build_server_config``), so a
    candidate that passes here is a candidate the server can boot from.  The
    ``live_config_path`` anchor is load-bearing: ``paths.*`` resolve against the
    project root of the YAML's directory, so validating with the staging path
    would build a *different* config than the one that goes live.

    Every caller **discards** the returned ``ServerConfig``.  It is never stashed,
    serialised, or written: it carries interpolated secrets, whereas the stash and
    the diffs keep ``${VAR}`` templates verbatim.

    Boundary error handling: operator-supplied YAML can fail construction with
    ``ValueError`` (validation guards), ``FatalConfigError`` (adapter guard), or
    ``TypeError`` (unknown key reaching ``**kwargs``; an uninterpolated ``${VAR}``
    left in a typed non-str field).  All are re-raised as
    :class:`CandidateConfigInvalid` so the HTTP layer answers 4xx, not 500.

    Parameters
    ----------
    candidate_bytes:
        Raw bytes of the candidate ``server.yaml``.
    live_config_path:
        Absolute path of the **live** ``configs/server.yaml`` — the location the
        candidate will occupy after promotion.

    Returns
    -------
    ServerConfig
        The constructed candidate config.  Discarded by all callers.

    Raises
    ------
    CandidateConfigInvalid
        The candidate is unparseable or cannot be constructed/validated.
    """
    try:
        parsed = _parse_candidate(candidate_bytes)
        return build_server_config(parsed, source_path=live_config_path)
    except Exception as exc:  # noqa: BLE001 — boundary: operator-supplied file
        raise CandidateConfigInvalid(str(exc)) from exc


def promote_config(
    candidate_path: Path,
    live_config_path: Path,
    *,
    expected_sha256: str,
) -> None:
    """Atomically promote *candidate_path* over *live_config_path*.

    The **only** route by which the live config file ever changes.  Ordering is
    load-bearing — every check that can reject runs before the first mutation:

    1. Read the candidate bytes **from disk** (authoritative; the stash is a
       mirror, not the source of truth).
    2. Reject when the bytes no longer match *expected_sha256* — what the operator
       previewed is what gets promoted.
    3. Construct + validate the candidate at the live path (:func:`validate_candidate`).
       The constructed ``ServerConfig`` is discarded — it carries interpolated
       secrets.  Callers re-load the live config from disk after promotion.
    4. ``os.rename`` — atomic swap.
    5. ``fsync`` the parent directory for rename durability.

    A failure at 1–3 leaves the filesystem untouched, so the caller can reject with
    a 4xx and keep the server bootable.

    Parameters
    ----------
    candidate_path:
        Path to the staged candidate file.
    live_config_path:
        Path to the live ``configs/server.yaml``.
    expected_sha256:
        Hex SHA-256 the candidate had when it was staged (the stash's
        ``candidate_hash``).  Required and non-empty: every production caller
        stages a candidate via ``/migration/preview``, which always computes this
        hash, so there is no legitimate call with nothing to compare against.

    Raises
    ------
    ValueError
        *expected_sha256* is empty.
    CandidateChanged
        The on-disk candidate no longer matches *expected_sha256*.
    CandidateConfigInvalid
        The candidate cannot be constructed as the live config.
    OSError
        The rename failed (e.g. EXDEV across filesystems).
    """
    if not expected_sha256:
        raise ValueError(
            "promote_config requires a non-empty expected_sha256 — every caller "
            "stages a candidate via /migration/preview first, which always "
            "computes one. An empty hash would promote without checking that the "
            "file on disk is still the one the operator previewed."
        )

    candidate_path = Path(candidate_path)
    live_config_path = Path(live_config_path)

    candidate_bytes = candidate_path.read_bytes()

    actual_sha256 = _sha256_bytes(candidate_bytes)
    if actual_sha256 != expected_sha256:
        raise CandidateChanged(
            f"candidate file {candidate_path!s} changed after it was staged "
            f"(staged sha256 {expected_sha256}, on-disk sha256 {actual_sha256}). "
            "Re-run POST /migration/preview so the diff you approve is the diff "
            "that is applied."
        )

    validate_candidate(candidate_bytes, live_config_path)

    os.rename(candidate_path, live_config_path)
    dir_fd = os.open(str(live_config_path.parent), os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    except OSError:
        # Durability best-effort: some filesystems reject directory fsync.
        pass
    finally:
        os.close(dir_fd)


def backup_live_config(live_config_path: Path, backups_root: Path) -> tuple[str, Path]:
    """Write the ``pre_migration`` backup slot for the live config.

    Config is the only required pre-migration artifact: a migration's sole live
    mutation is the atomic config swap, so rollback (and crash recovery) only ever
    restore the config.  Callers invoke this **before** :func:`promote_config`, so
    every path that renames the live config leaves a restore point behind.

    Parameters
    ----------
    live_config_path:
        Path to the live ``configs/server.yaml``.  A missing file yields an empty
        pre-hash and an empty artifact (fresh install).
    backups_root:
        Backups root directory; the slot is written under ``<backups_root>/config``.

    Returns
    -------
    tuple[str, Path]
        ``(pre_hash, slot)`` — hex SHA-256 of the live config bytes (``""`` when
        the file does not exist) and the slot directory that was written.

    Raises
    ------
    OSError
        The backup write failed.  The caller must map this to
        ``500 backup_write_failed``; no other mutation has happened yet.
    """
    live_config_path = Path(live_config_path)
    exists = live_config_path.exists()
    config_bytes = live_config_path.read_bytes() if exists else b""
    pre_hash = _sha256_bytes(config_bytes) if exists else ""

    slot = backup_write(
        ArtifactKind.CONFIG,
        config_bytes,
        meta_fields={"tier": "pre_migration", "pre_trial_hash": pre_hash},
        base_dir=Path(backups_root) / "config",
    )
    return pre_hash, slot


# ---------------------------------------------------------------------------
# Base-model-change detection
# ---------------------------------------------------------------------------


def compute_base_change(live_yaml: dict, candidate_yaml: dict) -> "dict | None":
    """Return a base-change descriptor when ``model:`` differs between the YAMLs.

    Returns ``None`` when the ``model:`` value is the same in both configs
    (no base model change) or when both sides are absent.

    Parameters
    ----------
    live_yaml:
        Parsed dict of the live ``server.yaml`` (``yaml.safe_load`` output).
    candidate_yaml:
        Parsed dict of the candidate ``server.yaml``.

    Returns
    -------
    dict | None
        When a model change is detected, a dict with keys:

        ``old_model``
            ``MODEL_REGISTRY`` alias currently in the live config
            (e.g. ``"mistral"``).
        ``new_model``
            ``MODEL_REGISTRY`` alias in the candidate config
            (e.g. ``"qwen3-4b"``).
        ``consequence``
            Operator-facing description of what the migration entails.
    """
    old_model = live_yaml.get("model", "")
    new_model = candidate_yaml.get("model", "")
    if old_model == new_model:
        return None
    consequence = (
        f"Base model changes from '{old_model}' to '{new_model}'. "
        "Every adapter (episodic/semantic/procedural) will be re-derived from the current "
        "weights onto the new base via a capture-then-relearn migration. "
        "Phase A (capture): all keyed facts are reconstructed from the live model into "
        "encrypted per-tier graph.json files, then the current adapter weights are deleted. "
        "A pre-migration bundle backup is taken before Phase A so rollback can restore the "
        "prior base model and weights. "
        "Phase B (relearn): the new base model is loaded in-process (server is briefly "
        "cloud-only during reload), then all tiers are retrained and gated on 100% recall. "
        "No server restart is required; rollback is available at any point via "
        "POST /migration/rollback."
    )
    return {
        "old_model": old_model,
        "new_model": new_model,
        "consequence": consequence,
    }
