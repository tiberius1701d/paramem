"""Operator-attention block for the ParaMem server (Slice 5a).

Pure-Python module — no torch, peft, or transformers imports at top level.
Defines :class:`AttentionItem` (frozen dataclass) plus one populator function
per alert category.  :func:`collect_attention_items` walks all active
populators in display order and concatenates results.

The module is intentionally isolated so tests can call populators directly
without spinning a FastAPI server or loading a model.

Display order (spec §L504–513):

    Migration → Consolidation → Sweeper → Backup* → Config drift →
    Key rotation* → Encryption* → Adapter fingerprint → Pre-flight*

    (* stub returns [] in Slice 5a — Slices 6/7 fill them.)
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AttentionItem dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AttentionItem:
    """One row in the operator-attention block.

    Forward-compat: ``kind`` is a free-form ``str`` (NOT ``Literal``).
    Slices 6/7 will introduce kinds like ``"backup_failed"``,
    ``"backup_stale"``, ``"key_rotation"``, ``"encryption_degraded"``,
    ``"pre_flight_fail"`` without any schema bump here.

    Attributes
    ----------
    kind:
        Stable identifier for the alert class.  See spec L434–451 for
        the catalog of kinds that Slices 5a/6/7 populate.
    level:
        ``"action_required"`` (yellow), ``"failed"`` (red), or
        ``"info"`` (cyan).  Used by the pstatus renderer to choose the
        row color and the banner color (red overrides yellow).
    summary:
        Human-readable one-line description shown by pstatus.  ≤ ~80
        chars.  Spec L422–427 examples.
    action_hint:
        Optional follow-up CLI hint.  Rendered on a continuation line
        with a leading "→ " when non-None.
    age_seconds:
        Optional age in seconds since the alert condition arose.  None
        when not applicable (e.g. config drift detected by polling does
        not carry an age).
    """

    kind: str  # NOT Literal — Slices 6/7 extend without schema bump
    level: str  # "action_required" | "failed" | "info"
    summary: str
    action_hint: str | None
    age_seconds: int | None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict suitable for JSON embedding.

        Returns
        -------
        dict[str, Any]
            All five fields present; ``action_hint`` and ``age_seconds``
            may be ``None``.
        """
        return asdict(self)


# ---------------------------------------------------------------------------
# Age helper
# ---------------------------------------------------------------------------


def _age_seconds_from_iso(iso_str: str) -> int | None:
    """Compute elapsed seconds from an ISO-8601 timestamp to now (UTC).

    Handles both timezone-aware and naive timestamps (naive treated as UTC).
    Returns ``None`` when the input is empty, ``None``, or unparseable.

    Parameters
    ----------
    iso_str:
        ISO-8601 string, e.g. ``"2026-04-21T00:10:31Z"`` or
        ``"2026-04-21T00:10:31+00:00"`` or ``"2026-04-21T00:10:31"``.

    Returns
    -------
    int | None
        Non-negative elapsed seconds, or ``None`` on parse failure.
    """
    if not iso_str:
        return None
    try:
        ts = datetime.fromisoformat(iso_str)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return max(0, int((datetime.now(timezone.utc) - ts).total_seconds()))
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Populators — 5 active in Slice 5a, 4 stubs for Slices 6/7
# ---------------------------------------------------------------------------


def _collect_migration_items(state: dict) -> list[AttentionItem]:
    """Emit migration-lifecycle attention items.

    Branch table (first matching branch wins):

    1. ``recovery_required`` non-empty (any state): one item per string,
       kind ``"migration_recovery_required"``, level ``"info"``.
    2. state == ``"LIVE"``: nothing further (after step 1's items).
    3. state == ``"STAGING"`` AND ``shape_changes`` non-empty: one item,
       kind ``"migration_shape_change_pending"``, level ``"info"``.
    4. state == ``"STAGING"`` AND no shape changes: nothing.
    5. state == ``"TRIAL"`` AND ``gates.status in {"pending", None}``:
       one item, kind ``"migration_trial_running"``, level ``"info"``.
    6. state == ``"TRIAL"`` AND ``gates.status in {"pass", "no_new_sessions"}``:
       one item, kind ``"migration_trial_pass"``, level ``"action_required"``.
    7. state == ``"TRIAL"`` AND ``gates.status in {"fail", "trial_exception"}``:
       one item, kind ``"migration_trial_failed"``, level ``"failed"``.

    Parameters
    ----------
    state:
        Server ``_state`` dict.  Read-only.

    Returns
    -------
    list[AttentionItem]
        Zero or more items in the order described above.
    """
    migration = state.get("migration") or {}
    mig_state = migration.get("state", "LIVE")
    recovery_required = migration.get("recovery_required") or []
    shape_changes = migration.get("shape_changes") or []
    trial = migration.get("trial") or {}
    gates = trial.get("gates") or {}
    gates_status = gates.get("status")

    items: list[AttentionItem] = []

    # Step 1 — recovery banners (any state)
    for msg in recovery_required:
        items.append(
            AttentionItem(
                kind="migration_recovery_required",
                level="info",
                summary=str(msg),
                action_hint=None,
                age_seconds=None,
            )
        )

    if mig_state == "LIVE":
        return items

    if mig_state == "STAGING":
        if shape_changes:
            # Summarise shape changes as a brief field list.
            field_summary = ", ".join(
                f"{sc['adapter']}.{sc['field']}" if isinstance(sc, dict) else str(sc)
                for sc in shape_changes[:3]
            )
            if len(shape_changes) > 3:
                field_summary += f" +{len(shape_changes) - 3} more"
            items.append(
                AttentionItem(
                    kind="migration_shape_change_pending",
                    level="info",
                    summary=f"STAGING — SHAPE CHANGE pending ({field_summary})",
                    action_hint="paramem migrate-confirm or migrate-cancel",
                    age_seconds=_age_seconds_from_iso(migration.get("staged_at", "")),
                )
            )
        # No shape changes in STAGING → nothing
        return items

    if mig_state == "TRIAL":
        trial_age = _age_seconds_from_iso(trial.get("started_at", ""))

        if gates_status in {"pending", None}:
            items.append(
                AttentionItem(
                    kind="migration_trial_running",
                    level="info",
                    summary="TRIAL consolidating...",
                    action_hint=None,
                    age_seconds=trial_age,
                )
            )
        elif gates_status in {"pass", "no_new_sessions"}:
            items.append(
                AttentionItem(
                    kind="migration_trial_pass",
                    level="action_required",
                    summary=f"TRIAL active (gates {gates_status.upper()})",
                    action_hint="paramem migrate-accept or paramem migrate-rollback",
                    age_seconds=trial_age,
                )
            )
        elif gates_status in {"fail", "trial_exception"}:
            # Short reason: first failing gate reason, or exception message.
            short_reason = gates.get("exception") or ""
            if not short_reason:
                for detail in gates.get("details") or []:
                    if isinstance(detail, dict) and detail.get("status") == "fail":
                        short_reason = detail.get("reason") or "gate failed"
                        break
            if not short_reason:
                short_reason = gates_status
            items.append(
                AttentionItem(
                    kind="migration_trial_failed",
                    level="failed",
                    summary=f"TRIAL FAILED — {short_reason}",
                    action_hint="paramem migrate-rollback",
                    age_seconds=trial_age,
                )
            )

    return items


def _collect_consolidation_items(state: dict) -> list[AttentionItem]:
    """Emit a consolidation-blocked item when a trial is in progress.

    The plan-reviewer fix (Fix 1) changes the emit condition: instead of
    checking ``_state["consolidating"] == True``, we check whether the
    migration state is ``"TRIAL"`` and gates are ``"pending"`` or ``None``.
    Rationale: ``_run_trial_consolidation`` runs in an executor and does NOT
    set ``_state["consolidating"] = True``; that flag is only set by the
    regular ``/consolidate`` path.

    Parameters
    ----------
    state:
        Server ``_state`` dict.  Read-only.

    Returns
    -------
    list[AttentionItem]
        Zero or one item.
    """
    migration = state.get("migration") or {}
    migration_state = migration.get("state")
    trial = migration.get("trial") or {}
    gates = trial.get("gates") or {}
    gates_status = gates.get("status")  # None when trial just kicked off

    if migration_state == "TRIAL" and gates_status in {"pending", None}:
        session_buffer = state.get("session_buffer")
        queued: str | int
        if session_buffer is not None:
            queued = getattr(session_buffer, "pending_count", "unknown")
        else:
            queued = "unknown"
        return [
            AttentionItem(
                kind="consolidation_blocked",
                level="info",
                summary=f"Consol: BLOCKED — trial in progress (transcripts queued: {queued})",
                action_hint=None,
                age_seconds=None,
            )
        ]
    return []


def _collect_sweeper_items(state: dict) -> list[AttentionItem]:
    """Emit a sweeper-held item when a trial is active with pending transcripts.

    Parameters
    ----------
    state:
        Server ``_state`` dict.  Read-only.

    Returns
    -------
    list[AttentionItem]
        Zero or one item.
    """
    migration = state.get("migration") or {}
    mig_state = migration.get("state")
    trial = migration.get("trial") or {}

    if mig_state != "TRIAL":
        return []

    session_buffer = state.get("session_buffer")
    pending_count = getattr(session_buffer, "pending_count", 0) if session_buffer else 0
    if pending_count > 0:
        return [
            AttentionItem(
                kind="sweeper_held",
                level="info",
                summary=f"HELD — {pending_count} transcripts pending deletion (trial active)",
                action_hint=None,
                age_seconds=_age_seconds_from_iso(trial.get("started_at", "")),
            )
        ]
    return []


def _collect_config_drift_items(state: dict) -> list[AttentionItem]:
    """Emit a config-drift item when server.yaml content changed since boot.

    Parameters
    ----------
    state:
        Server ``_state`` dict.  Read-only.

    Returns
    -------
    list[AttentionItem]
        Zero or one item.
    """
    drift = state.get("config_drift") or {}
    if drift.get("detected") is True:
        # Use last_checked_at as the best available timestamp for the drift age;
        # the actual change moment is unknown (content-hash only, per spec L502).
        age = _age_seconds_from_iso(drift.get("last_checked_at", ""))
        return [
            AttentionItem(
                kind="config_drift",
                level="action_required",
                summary="server.yaml content changed since server start",
                action_hint="reload or migrate to apply",
                age_seconds=age,
            )
        ]
    return []


def _collect_adapter_fingerprint_items(state: dict) -> list[AttentionItem]:
    """Emit adapter-fingerprint-mismatch items from the startup validator.

    Reads ``state["adapter_manifest_status"]`` (populated by
    ``_mount_adapters_from_slots`` at startup).  Schema per row:
    ``{status, reason, field, severity, slot_path, checked_at}``.

    Primary adapter (``"episodic"``) with severity ``"red"`` → level
    ``"failed"``.  Secondary adapters (``"semantic"``,
    ``"procedural"``) with severity ``"yellow"`` → level ``"info"``.

    Multiple mismatch rows → multiple items.  Primary first, then
    alphabetical by name.

    Parameters
    ----------
    state:
        Server ``_state`` dict.  Read-only.

    Returns
    -------
    list[AttentionItem]
        Zero or more items.
    """
    manifest_status: dict = state.get("adapter_manifest_status") or {}
    primary_items: list[AttentionItem] = []
    secondary_items: list[AttentionItem] = []

    _MISMATCH_STATUSES = {"mismatch", "manifest_missing", "migrated_unverified"}

    for name, row in sorted(manifest_status.items()):
        if not isinstance(row, dict):
            continue
        row_status = row.get("status", "")
        if row_status not in _MISMATCH_STATUSES:
            continue
        severity = row.get("severity", "yellow")
        reason = row.get("reason") or row_status
        age = _age_seconds_from_iso(row.get("checked_at", ""))

        if severity == "red":
            primary_items.append(
                AttentionItem(
                    kind="adapter_fingerprint_mismatch_primary",
                    level="failed",
                    summary=(f"FINGERPRINT MISMATCH ({name}) — {reason} — PA routing DISABLED"),
                    action_hint="paramem migrate-accept or paramem migrate-rollback",
                    age_seconds=age,
                )
            )
        else:
            secondary_items.append(
                AttentionItem(
                    kind="adapter_fingerprint_mismatch_secondary",
                    level="info",
                    summary=(f"FINGERPRINT MISMATCH ({name}) — {reason} — adapter unmounted"),
                    action_hint=None,
                    age_seconds=age,
                )
            )

    # Primary items first (already sorted by name via sorted()), then secondary.
    return primary_items + secondary_items


# ---------------------------------------------------------------------------
# Stubs — Slices 6/7 fill these
# ---------------------------------------------------------------------------


def _collect_backup_items(state: dict, config) -> list[AttentionItem]:
    """Emit backup-failure / stale / disk-pressure items.

    Emit order: FAILED → DISK PRESSURE → STALE.  Multiple alerts can fire
    simultaneously (all 3 possible at the same time).

    Conditions
    ----------
    FAILED:
        ``last_failure_at`` is newer than ``last_success_at`` (or
        ``last_success_at`` is None).
    DISK PRESSURE:
        Current backup-store usage ≥ 80% of global cap.  Level ``info``
        at 80–99%; level ``failed`` at ≥ 100%.
    STALE:
        ``last_success_at`` older than ``2 × schedule_interval`` AND
        schedule is not ``"off"`` / ``""`` AND ``last_success_at`` is
        not None.

    Silently returns ``[]`` when ``config`` is ``None`` (unit-test shim),
    when backup modules are unavailable, or when reading state raises
    ``BackupStateSchemaError`` (treated as "never run" so only non-state
    alerts still evaluate).

    Parameters
    ----------
    state:
        Server ``_state`` dict.  Read-only.
    config:
        Loaded ``ServerConfig``.  ``None`` → return ``[]`` immediately.

    Returns
    -------
    list[AttentionItem]
        Zero or more items in FAILED → DISK PRESSURE → STALE order.
    """
    if config is None:
        return []

    try:
        from paramem.backup import retention as _retention
        from paramem.backup import state as _bk_state
        from paramem.backup.timer import _backup_timer_interval_seconds
    except ImportError:
        return []

    items: list[AttentionItem] = []

    backups_cfg = config.security.backups
    state_dir = (config.paths.data / "state").resolve()
    backups_root = (config.paths.data / "backups").resolve()

    # -- Read persisted backup state (None when no run has ever happened). --
    try:
        record = _bk_state.read_backup_state(state_dir)
    except _bk_state.BackupStateSchemaError:
        # Malformed state: treated as "never run".  DISK PRESSURE still evaluates
        # (it reads the filesystem directly), but FAILED and STALE are skipped.
        record = None

    raw_schedule = backups_cfg.schedule or ""
    schedule_str = str(raw_schedule).strip().lower()

    # -- Alert 1: Backup FAILED (level=failed) --
    # Trigger: last_failure_at is newer than last_success_at (or last_success_at is None).
    if record and record.last_failure_at:
        last_succ = record.last_success_at or ""
        if record.last_failure_at > last_succ:  # ISO strings compare correctly as UTC
            reason = record.last_failure_reason or "(unknown)"
            items.append(
                AttentionItem(
                    kind="backup_failed",
                    level="failed",
                    summary=(f"Backup: FAILED {record.last_failure_at[:19]} — {reason}"),
                    action_hint="Inspect the backup runner logs.",
                    age_seconds=_age_seconds_from_iso(record.last_failure_at),
                )
            )

    # -- Alert 2: Backup DISK PRESSURE (level=info at 80–99%, failed at ≥100%) --
    try:
        usage = _retention.compute_disk_usage(backups_root, backups_cfg)
    except Exception:
        usage = None  # silently skip DISK PRESSURE alert on scan failure

    if usage is not None:
        disk_used = usage.total_bytes
        disk_cap = usage.cap_bytes
        # B4 fix (2026-04-22 E2E baseline): when cap_bytes==0, compute_disk_usage
        # returns pct_of_cap=0.0 and the old guard "usage.cap_bytes > 0" skipped
        # the alert entirely.  cap=0 + any non-zero usage means the store is over
        # capacity (infinite percent), so emit at level "failed".
        if disk_cap == 0:
            pct = float("inf") if disk_used > 0 else 0.0
        else:
            pct = disk_used / disk_cap
        if pct >= 0.80:
            lvl = "failed" if pct >= 1.0 else "info"
            cap_gb = backups_cfg.max_total_disk_gb
            items.append(
                AttentionItem(
                    kind="backup_disk_pressure",
                    level=lvl,
                    summary=(
                        f"Backup: DISK {int(min(pct, 999.0) * 100)}%"
                        f" of {cap_gb} GB cap — prune required"
                    ),
                    action_hint="Run paramem backup-prune.",
                    age_seconds=None,
                )
            )

    # -- Alert 3: Backup STALE (level=info) --
    # Skipped when schedule="off" (or empty/disabled) or when no successful run yet.
    if record and record.last_success_at and schedule_str not in ("", "off", "disabled", "none"):
        interval = _backup_timer_interval_seconds(schedule_str)
        if interval and interval > 0:
            age = _age_seconds_from_iso(record.last_success_at)
            if age is not None and age > 2 * interval:
                # Human-readable age.
                if age >= 86400:
                    age_human = f"{age // 86400}d"
                elif age >= 3600:
                    age_human = f"{age // 3600}h"
                else:
                    age_human = f"{age // 60}m"
                items.append(
                    AttentionItem(
                        kind="backup_stale",
                        level="info",
                        summary=(
                            f"Backup: STALE — last success {age_human} ago "
                            f"(schedule: {raw_schedule})"
                        ),
                        action_hint=(
                            "Run paramem backup-create or wait for the next scheduled cycle."
                        ),
                        age_seconds=age,
                    )
                )

    return items


def _collect_key_rotation_items(state: dict) -> list[AttentionItem]:
    """Emit key-rotation items when master-key fingerprint changed.

    Stub — not yet populated.  Signature is intentionally ``(state)``-only;
    when this populator is wired, extend to ``(state, config)`` rather than
    unifying with ``_collect_backup_items`` prematurely.
    """
    return []


def _collect_encryption_items(state: dict) -> list[AttentionItem]:
    """Emit one ``encryption_off`` item when posture is SECURITY: OFF.

    Reads :data:`_state['encryption']` — set once at lifespan entry from
    the posture computed by :mod:`paramem.server.security_posture`. Never
    re-probes the environment, so a key rotated into place mid-session
    does not silently suppress this alert until the next server restart
    (which is also the point at which the posture itself would flip).

    Silently returns ``[]`` when the field is missing (pre-lifespan call
    during tests) or when posture is SECURITY: ON.
    """
    posture = state.get("encryption")
    if posture != "off":
        return []
    return [
        AttentionItem(
            kind="encryption_off",
            level="action_required",
            summary=("SECURITY: OFF — all infrastructure metadata is plaintext on disk"),
            action_hint=(
                "paramem generate-key    # mint daily + recovery, then set "
                "PARAMEM_DAILY_PASSPHRASE and restart the server"
            ),
            age_seconds=None,
        )
    ]


def _collect_pre_flight_items(state: dict, config) -> list[AttentionItem]:
    """Emit migration pre-flight fail items (disk pressure, etc.).

    Re-computes the disk-pressure check at every ``/status`` poll using the
    cached ``compute_disk_usage`` (5s TTL — no filesystem re-walk per poll).

    Suppressed during STAGING / TRIAL — those states have already passed
    pre-flight at ``/preview`` time, or are mid-trial where pre-flight is
    irrelevant.

    Parameters
    ----------
    state:
        Server ``_state`` dict.  Read-only.
    config:
        Loaded ``ServerConfig``.  ``None`` → return ``[]`` immediately.

    Returns
    -------
    list[AttentionItem]
        Zero or one item when disk pressure would block a migration preview.
    """
    if config is None:
        return []

    migration = state.get("migration") or {}
    mig_state = migration.get("state", "LIVE")
    if mig_state in ("STAGING", "TRIAL"):
        return []  # suppression rule — pre-flight is only relevant in LIVE state

    # Re-compute the disk-pressure check.  loop may be None (cloud-only mode).
    try:
        from paramem.backup.preflight import compute_pre_flight_check
    except ImportError:
        return []

    loop = state.get("consolidation_loop")
    backups_root = (config.paths.data / "backups").resolve()

    live_config_path_raw = state.get("config_path")
    live_config_path = (
        Path(live_config_path_raw) if live_config_path_raw else Path("configs/server.yaml")
    )

    try:
        registry_path = config.paths.key_metadata
    except (AttributeError, TypeError):
        registry_path = None

    try:
        pf = compute_pre_flight_check(
            server_config=config,
            loop=loop,
            backups_root=backups_root,
            live_config_path=live_config_path,
            registry_path=registry_path,
        )
    except Exception:
        return []  # scan failure — silent; do not crash /status

    if pf.fail_code == "disk_pressure":
        used_gb = pf.disk_used_bytes / (1024**3)
        cap_gb = pf.disk_cap_bytes / (1024**3)
        return [
            AttentionItem(
                kind="migration_pre_flight_fail",
                level="info",
                summary=(
                    f"Migration: PRE-FLIGHT FAIL — disk pressure "
                    f"(used {used_gb:.1f} of {cap_gb:.1f} GB cap)"
                ),
                action_hint=(
                    "Run paramem backup-prune to free space, or raise "
                    "security.backups.max_total_disk_gb."
                ),
                age_seconds=None,
            )
        ]
    return []


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def collect_attention_items(
    state: dict,
    config: object | None,
) -> list[AttentionItem]:
    """Walk active populators in display order and concatenate results.

    Display order matches the Attention block layout in spec L504–513:

        Migration → Consolidation → Sweeper → Backup* → Config drift →
        Key rotation* → Encryption* → Adapter fingerprint → Pre-flight*

        (* stub returns [] in Slice 5a — Slices 6/7 fill.)

    Parameters
    ----------
    state:
        The server's ``_state`` dict.  Reads only — never mutated.
    config:
        Loaded ``ServerConfig`` for populators that need config-side
        toggles (e.g. encryption mode).  Older populators do not use
        this; pass ``None`` when convenient.

    Returns
    -------
    list[AttentionItem]
        Ordered list of active attention items.  Empty when the server
        is in a clean state (no action required from the operator).
    """
    items: list[AttentionItem] = []
    items.extend(_collect_migration_items(state))
    items.extend(_collect_consolidation_items(state))
    items.extend(_collect_sweeper_items(state))
    # NOTE: _collect_backup_items and _collect_pre_flight_items take
    # (state, config); _collect_key_rotation_items and _collect_encryption_items
    # are stubs that keep a (state)-only signature until they are wired — do NOT
    # unify the signatures here before they are populated.
    items.extend(_collect_backup_items(state, config))
    items.extend(_collect_config_drift_items(state))
    items.extend(_collect_key_rotation_items(state))  # stub
    items.extend(_collect_encryption_items(state))  # stub
    items.extend(_collect_adapter_fingerprint_items(state))
    items.extend(_collect_pre_flight_items(state, config))
    return items
