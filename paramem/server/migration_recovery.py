"""Crash recovery for the ParaMem migration subsystem.

Implements the 5-case decision matrix from spec §L289–295.  Called once at
lifespan startup BEFORE drift detection is initialised so that ``_state["migration"]``
reflects any partially-completed ``/migration/confirm`` before any other
endpoint can be reached.

Decision matrix (abridged — see spec for the authoritative table)
-----------------------------------------------------------------

| # | Condition                                                  | Action                      |
|---|------------------------------------------------------------|------------------------------|
| 1 | trial.json; live_hash != marker.pre_trial_hash             | RESUME_TRIAL                 |
| 2 | trial.json; live_hash == marker.pre_trial_hash             | STEP3_CRASH_CLEANUP          |
| 3 | No marker; pre_migration backup; hash matches; young       | ORPHAN_SWEEP                 |
| 4 | No marker; pre_migration backup; hash mismatch             | AMBIGUOUS_REQUIRES_OPERATOR  |
| 5 | No marker; no pre_migration backup                         | NORMAL_LIVE                  |

Edge cases
----------
- Unparseable marker → treat as AMBIGUOUS (do NOT delete; log ERROR).
- Multiple orphan backups with the same pre_trial_hash → delete all matches.
- max_age_hours absent → 24h fallback.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path

from paramem.backup.enumerate import enumerate_backups
from paramem.backup.types import ArtifactKind
from paramem.server.trial_state import (
    TrialMarker,
    TrialMarkerSchemaError,
    clear_trial_marker,
    read_trial_marker,
)

logger = logging.getLogger(__name__)


class RecoveryAction(str, Enum):
    """Outcome of the crash-recovery inspection."""

    NORMAL_LIVE = "normal_live"
    RESUME_TRIAL = "resume_trial"
    STEP3_CRASH_CLEANUP = "step3_crash_cleanup"
    ORPHAN_SWEEP = "orphan_sweep"
    AMBIGUOUS_REQUIRES_OPERATOR = "ambiguous_requires_operator"


@dataclass(frozen=True)
class MigrationRecoveryResult:
    """Result of a single ``recover_migration_state`` call.

    Attributes
    ----------
    action:
        Which recovery branch was taken.
    trial_marker:
        The parsed ``TrialMarker`` when ``action == RESUME_TRIAL``, else ``None``.
    pre_trial_backup_slot:
        Path to the relevant pre-migration backup slot for ORPHAN_SWEEP,
        AMBIGUOUS, or RESUME_TRIAL cases.  ``None`` for NORMAL_LIVE and
        STEP3_CRASH_CLEANUP.
    recovery_required:
        Human-readable rows populated for AMBIGUOUS_REQUIRES_OPERATOR and
        unparseable-marker cases.  Surfaced via ``/migration/status``.
    log_lines:
        ``(level, message)`` pairs for the caller to emit via the logging
        framework once ``_state`` is fully initialised.  The caller logs
        these; this module does NOT call ``logger.*`` during the return path
        to keep I/O side-effects explicit.
    """

    action: RecoveryAction
    trial_marker: TrialMarker | None
    pre_trial_backup_slot: Path | None
    recovery_required: list[str]
    log_lines: list[tuple[str, str]] = field(default_factory=list)


def _sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def recover_migration_state(
    state_dir: Path,
    live_config_path: Path,
    backups_root: Path,
    max_age_hours: int = 24,
) -> MigrationRecoveryResult:
    """Walk the 5-case decision matrix at lifespan startup.

    This function reads (never writes except for STEP3_CRASH_CLEANUP marker
    deletion and ORPHAN_SWEEP slot deletion) the relevant paths and returns
    a ``MigrationRecoveryResult`` that the lifespan uses to seed
    ``_state["migration"]``.

    Parameters
    ----------
    state_dir:
        Directory containing ``trial.json`` (e.g. ``data/ha/state/``).
    live_config_path:
        Path to the currently-active ``configs/server.yaml``.
    backups_root:
        Root of the backup store (e.g. ``data/ha/backups/``).
    max_age_hours:
        Look-back window for the orphan-sweep case.  Pre-migration backup
        slots older than this are not swept (left for operator inspection).
        Default 24h matches ``security.backups.orphan_sweep_max_age_hours``.

    Returns
    -------
    MigrationRecoveryResult
        The recovery outcome; caller seeds ``_state["migration"]`` from it.
    """
    log_lines: list[tuple[str, str]] = []

    # --- Step 1: Read live config hash ---
    live_hash: str = ""
    if live_config_path.exists():
        try:
            live_hash = _sha256_file(live_config_path)
        except OSError as exc:
            log_lines.append(("ERROR", f"migration recovery: could not hash live config: {exc}"))

    # --- Step 2: Try to read the trial marker ---
    marker: TrialMarker | None = None
    marker_parse_error: str | None = None
    try:
        marker = read_trial_marker(state_dir)
    except TrialMarkerSchemaError as exc:
        marker_parse_error = str(exc)

    # --- Case: unparseable marker → AMBIGUOUS ---
    if marker_parse_error is not None:
        msg = f"Migration: RECOVERY REQUIRED — trial.json schema/parse error: {marker_parse_error}"
        log_lines.append(("ERROR", f"migration recovery: {msg}"))
        return MigrationRecoveryResult(
            action=RecoveryAction.AMBIGUOUS_REQUIRES_OPERATOR,
            trial_marker=None,
            pre_trial_backup_slot=None,
            recovery_required=[msg],
            log_lines=log_lines,
        )

    # --- Cases 1 & 2: marker exists ---
    if marker is not None:
        if live_hash and live_hash != marker.pre_trial_config_sha256:
            # Case 1: swap completed — live config changed.  Resume TRIAL.
            msg = f"migration: resumed TRIAL state from marker started_at={marker.started_at}"
            log_lines.append(("INFO", msg))
            return MigrationRecoveryResult(
                action=RecoveryAction.RESUME_TRIAL,
                trial_marker=marker,
                pre_trial_backup_slot=None,
                recovery_required=[],
                log_lines=log_lines,
            )
        else:
            # Case 2: hashes equal (or live_hash unknown) — step-3 crash.
            # Marker written but rename didn't complete; delete marker, keep backups.
            try:
                clear_trial_marker(state_dir)
                log_lines.append(
                    (
                        "WARNING",
                        "migration: detected step-3 crash (marker without rename); "
                        "cleared marker, kept pre-migration backups in place",
                    )
                )
            except OSError as exc:
                log_lines.append(
                    (
                        "ERROR",
                        f"migration: step-3 cleanup failed — could not delete marker: {exc}",
                    )
                )
            return MigrationRecoveryResult(
                action=RecoveryAction.STEP3_CRASH_CLEANUP,
                trial_marker=None,
                pre_trial_backup_slot=None,
                recovery_required=[],
                log_lines=log_lines,
            )

    # --- No marker: check for orphan pre-migration backups ---
    config_records = enumerate_backups(backups_root, kind=ArtifactKind.CONFIG)
    pre_migration_records = [r for r in config_records if r.meta.tier == "pre_migration"]

    if not pre_migration_records:
        # Case 5: normal live startup.
        return MigrationRecoveryResult(
            action=RecoveryAction.NORMAL_LIVE,
            trial_marker=None,
            pre_trial_backup_slot=None,
            recovery_required=[],
            log_lines=log_lines,
        )

    # Find the most recent pre_migration record.
    newest = pre_migration_records[0]  # already newest-first

    if newest.pre_trial_hash is None:
        # Old-format backup without pre_trial_hash — cannot determine intent; AMBIGUOUS.
        msg = (
            f"Migration: RECOVERY REQUIRED — pre-migration backup {newest.slot_dir.name} "
            "has no pre_trial_hash field (old backup format); cannot determine recovery intent"
        )
        log_lines.append(("ERROR", f"migration recovery: {msg}"))
        return MigrationRecoveryResult(
            action=RecoveryAction.AMBIGUOUS_REQUIRES_OPERATOR,
            trial_marker=None,
            pre_trial_backup_slot=newest.slot_dir,
            recovery_required=[msg],
            log_lines=log_lines,
        )

    if live_hash and newest.pre_trial_hash == live_hash:
        # Hash matches live config — this backup is from a step-2 crash (backup written,
        # but marker never written and rename never happened).
        now_utc = datetime.now(tz=timezone.utc)
        age_limit = now_utc - timedelta(hours=max_age_hours)

        # Collect ALL matching orphan backups (could be multiple failed attempts).
        matching_records = [r for r in pre_migration_records if r.pre_trial_hash == live_hash]

        # Split into within-window and outside-window.
        in_window = [r for r in matching_records if r.created_at >= age_limit]
        # outside_window slots are left in place.

        if in_window:
            # Case 3: ORPHAN_SWEEP — delete all in-window matches.
            swept_slots: list[Path] = []
            for rec in in_window:
                try:
                    shutil.rmtree(rec.slot_dir)
                    swept_slots.append(rec.slot_dir)
                except OSError as exc:
                    log_lines.append(
                        (
                            "ERROR",
                            f"migration recovery: orphan sweep failed for {rec.slot_dir}: {exc}",
                        )
                    )

            # Also sweep matching graph + registry slots from the same timestamps.
            for rec in in_window:
                ts = rec.timestamp
                for kind in (ArtifactKind.GRAPH, ArtifactKind.REGISTRY):
                    sibling_dir = backups_root / kind.value / ts
                    if sibling_dir.exists():
                        try:
                            shutil.rmtree(sibling_dir)
                        except OSError as exc:
                            log_lines.append(
                                (
                                    "WARNING",
                                    f"migration recovery: could not sweep sibling "
                                    f"{sibling_dir}: {exc}",
                                )
                            )

            for slot in swept_slots:
                log_lines.append(
                    (
                        "WARNING",
                        f"migration: orphan-swept pre-migration backup {slot.name} "
                        "(step-2 crash, hash matches live config)",
                    )
                )

            return MigrationRecoveryResult(
                action=RecoveryAction.ORPHAN_SWEEP,
                trial_marker=None,
                pre_trial_backup_slot=None,
                recovery_required=[],
                log_lines=log_lines,
            )
        else:
            # All matching orphans are outside the window — normal live startup.
            return MigrationRecoveryResult(
                action=RecoveryAction.NORMAL_LIVE,
                trial_marker=None,
                pre_trial_backup_slot=None,
                recovery_required=[],
                log_lines=log_lines,
            )
    else:
        # Case 4: hash mismatch — ambiguous; leave backup in place, alert operator.
        live_h_short = live_hash[:8] if live_hash else "unknown"
        backup_h_short = (newest.pre_trial_hash or "")[:8]
        msg = (
            f"Migration: RECOVERY REQUIRED — pre-migration backup {newest.slot_dir.name} "
            f"has hash {backup_h_short} but live config hash is {live_h_short}. "
            "Something outside the migration path may have written to server.yaml. "
            "Investigate before proceeding."
        )
        log_lines.append(("ERROR", f"migration recovery: {msg}"))
        return MigrationRecoveryResult(
            action=RecoveryAction.AMBIGUOUS_REQUIRES_OPERATOR,
            trial_marker=None,
            pre_trial_backup_slot=newest.slot_dir,
            recovery_required=[msg],
            log_lines=log_lines,
        )
