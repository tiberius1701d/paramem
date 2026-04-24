"""Scheduled backup runner for the ParaMem backup subsystem.

Orchestrates the full backup pipeline:
  1. Schedule guard (``schedule: "off"`` → no-op).
  2. Disk-pressure write gate (rule 1).
  3. Per-artifact write loop.
  4. Post-write pruning (best-effort).
  5. Returns a ``ScheduledBackupResult`` dataclass.

The runner hard-codes ``tier="daily"`` in Slice 6a.  Weekly/monthly/yearly
tier emission is future work (Slice 6b); the schema accepts those tier names
for retention budgets.

No torch, peft, or transformers imports at module level.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from paramem.server.config import ServerConfig

logger = logging.getLogger(__name__)

STATE_FILE_NAME: str = "backup.json"


# ---------------------------------------------------------------------------
# ScheduledBackupResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScheduledBackupResult:
    """One scheduled-backup invocation outcome.

    Persisted to ``data/ha/state/backup.json`` after every run (success OR
    failure) so ``/status`` can render the latest state without polling the
    runner process.

    Attributes
    ----------
    started_at:
        ISO-8601 UTC timestamp when the run started.
    completed_at:
        ISO-8601 UTC timestamp when the run completed.
    success:
        ``True`` when at least one artifact was written and pruning did not
        raise.  ``False`` on disk-pressure refusal or write error.
    tier:
        Backup tier tag — always ``"daily"`` in Slice 6a.
    label:
        Optional operator-supplied annotation.
    written_slots:
        Mapping of artifact name → absolute slot path string for artifacts
        written successfully this run.
    skipped_artifacts:
        ``(artifact_name, reason)`` pairs for artifacts that were not written.
    error:
        ``repr`` of the first exception encountered, or ``None`` on success.
    prune_result_summary:
        Summary dict of the ``PruneResult`` from this run, or ``None`` when
        pruning did not run (disk pressure, write failure, or schedule off).
    """

    started_at: str
    completed_at: str
    success: bool
    tier: str
    label: str | None
    written_slots: dict[str, str]
    skipped_artifacts: list[tuple[str, str]]
    error: str | None
    prune_result_summary: dict | None


# ---------------------------------------------------------------------------
# run_scheduled_backup
# ---------------------------------------------------------------------------


def run_scheduled_backup(
    *,
    server_config: "ServerConfig",
    loop,  # ConsolidationLoop | None — avoid import cycle
    state_dir: Path,
    backups_root: Path,
    live_config_path: Path,
    tier: Literal["daily", "weekly", "monthly", "yearly"] = "daily",
    label: str | None = None,
    now: datetime | None = None,
) -> ScheduledBackupResult:
    """Drive the full backup pipeline for one scheduled invocation.

    Steps
    -----
    1. **Schedule guard** — when ``server_config.security.backups.schedule``
       is ``"off"`` (or empty), return a no-op success immediately.  The
       caller (runner CLI) skips the state-file write so ``/status`` keeps
       reflecting the previous run.

    1b. **keep=0 short-circuit** — when the target tier's ``keep == 0``,
        return a no-op success immediately with all artifacts in
        ``skipped_artifacts``.  Avoids writing + immediately pruning; the
        ``prune()`` call (step 4) handles removal of any existing slots.

    2. **Disk-pressure write gate** (rule 1) — compute current disk usage.
       When ``pct_of_cap >= 1.0``, refuse the write.  Return
       ``success=False`` with ``error="disk_pressure: ..."`` and all
       artifacts listed in ``skipped_artifacts``.

    3. **Per-artifact write loop** — for each artifact in
       ``server_config.security.backups.artifacts``:

       - ``"config"`` → ``live_config_path.read_bytes()``.  Skip with reason
         ``"config file missing"`` when the file does not exist.
       - ``"graph"``   → ``loop.merger.save_bytes()``.  Skip with reason
         ``"consolidation loop unavailable (cloud-only mode?)"`` when
         ``loop is None`` or ``loop`` has no ``merger`` attribute.
       - ``"registry"`` → ``server_config.paths.key_metadata.read_bytes()``.
         Skip with reason ``"registry empty (no keys yet)"`` when the file
         does not exist.  Write even when the file is empty (0 bytes).

       On any write exception: record the exception in ``error``, mark
       remaining artifacts as ``"aborted after prior failure"``, and return
       ``success=False`` (skip pruning).

    4. **Pruning** — only when at least one artifact was written.  Wrap in
       ``try/except``: prune failure logs ERROR but ``success`` stays
       ``True``.

    5. Returns ``ScheduledBackupResult``.

    Parameters
    ----------
    server_config:
        ``ServerConfig`` providing backups config, paths, and encryption
        settings.
    loop:
        ``ConsolidationLoop`` instance for graph access.  ``None`` when the
        server is in cloud-only mode.
    state_dir:
        Directory containing ``trial.json`` for immunity detection and the
        lock file for ``update_backup_state``.
    backups_root:
        Root of the backup store (``data/ha/backups/``).
    live_config_path:
        Path to the live ``server.yaml`` to back up.
    tier:
        Backup tier tag.  Always ``"daily"`` in Slice 6a.
    label:
        Optional operator-supplied annotation written into each slot's
        sidecar.
    now:
        UTC datetime for logging and slot naming.  Defaults to
        ``datetime.now(timezone.utc)``.

    Returns
    -------
    ScheduledBackupResult
    """
    from paramem.backup.backup import write as backup_write
    from paramem.backup.retention import compute_disk_usage, prune
    from paramem.backup.types import ArtifactKind

    if now is None:
        now = datetime.now(timezone.utc)

    started_at = now.isoformat()

    def _completed_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    backups_cfg = server_config.security.backups
    artifacts_cfg = backups_cfg.artifacts

    # Step 1: Schedule guard.
    schedule_str = (backups_cfg.schedule or "").strip().lower()
    if schedule_str in ("", "off", "disabled", "none"):
        return ScheduledBackupResult(
            started_at=started_at,
            completed_at=_completed_now(),
            success=True,
            tier=tier,
            label=label,
            written_slots={},
            skipped_artifacts=[],
            error=None,
            prune_result_summary=None,
        )

    # Step 1b: keep=0 short-circuit — tier emission is disabled; no writes needed.
    tier_cfg = getattr(backups_cfg.retention, tier, None)
    if tier_cfg is not None and tier_cfg.keep == 0:
        reason = f"tier keep=0 — emission disabled for tier '{tier}'"
        logger.info("run_scheduled_backup: %s", reason)
        skipped = [(a, reason) for a in artifacts_cfg]
        return ScheduledBackupResult(
            started_at=started_at,
            completed_at=_completed_now(),
            success=True,
            tier=tier,
            label=label,
            written_slots={},
            skipped_artifacts=skipped,
            error=None,
            prune_result_summary=None,
        )

    # Step 2: Disk-pressure write gate (rule 1).
    backups_root = Path(backups_root)
    disk_usage = compute_disk_usage(backups_root, backups_cfg, bypass_cache=True)
    if disk_usage.pct_of_cap >= 1.0:
        used_gb = disk_usage.total_bytes / (1024**3)
        cap_gb = backups_cfg.max_total_disk_gb
        err = f"disk_pressure: max_total_disk_gb reached (used {used_gb:.2f} of {cap_gb} GB)"
        logger.error("run_scheduled_backup: %s", err)
        return ScheduledBackupResult(
            started_at=started_at,
            completed_at=_completed_now(),
            success=False,
            tier=tier,
            label=label,
            written_slots={},
            skipped_artifacts=[(a, "disk_pressure") for a in artifacts_cfg],
            error=err,
            prune_result_summary=None,
        )

    # Step 3: Per-artifact write loop.
    written_slots: dict[str, str] = {}
    skipped_artifacts: list[tuple[str, str]] = []
    first_error: str | None = None

    for artifact_name in artifacts_cfg:
        if first_error is not None:
            skipped_artifacts.append((artifact_name, "aborted after prior failure"))
            continue

        artifact_bytes: bytes | None = None
        skip_reason: str | None = None

        if artifact_name == "config":
            config_path = Path(live_config_path)
            if not config_path.exists():
                skip_reason = "config file missing"
            else:
                try:
                    artifact_bytes = config_path.read_bytes()
                except OSError as exc:
                    first_error = repr(exc)
                    skipped_artifacts.append((artifact_name, f"read error: {exc}"))
                    continue

        elif artifact_name == "graph":
            if loop is None or not hasattr(loop, "merger"):
                skip_reason = "consolidation loop unavailable (cloud-only mode?)"
            else:
                try:
                    artifact_bytes = loop.merger.save_bytes()
                except Exception as exc:
                    first_error = repr(exc)
                    skipped_artifacts.append((artifact_name, f"graph save error: {exc}"))
                    continue

        elif artifact_name == "registry":
            registry_path = server_config.paths.key_metadata
            if not Path(registry_path).exists():
                skip_reason = "registry empty (no keys yet)"
            else:
                try:
                    artifact_bytes = Path(registry_path).read_bytes()
                    # Write even when 0 bytes — operator may want to capture empty state.
                except OSError as exc:
                    first_error = repr(exc)
                    skipped_artifacts.append((artifact_name, f"read error: {exc}"))
                    continue
        else:
            skip_reason = f"unknown artifact kind: {artifact_name!r}"

        if skip_reason is not None:
            logger.info("run_scheduled_backup: skipping %s — %s", artifact_name, skip_reason)
            skipped_artifacts.append((artifact_name, skip_reason))
            continue

        if artifact_bytes is None:
            # Should not happen, but guard defensively.
            skipped_artifacts.append((artifact_name, "no bytes produced"))
            continue

        # Write the artifact.
        try:
            kind = ArtifactKind[artifact_name.upper()]
            slot_dir = backup_write(
                kind,
                artifact_bytes,
                meta_fields={"tier": tier, "label": label},
                base_dir=backups_root / artifact_name,
            )
            written_slots[artifact_name] = str(slot_dir)
            logger.info("run_scheduled_backup: wrote %s slot %s", artifact_name, slot_dir)
        except Exception as exc:
            first_error = repr(exc)
            logger.error("run_scheduled_backup: failed to write %s: %s", artifact_name, exc)
            continue

    # Step 4: Pruning (only when at least one artifact was written).
    prune_result_summary: dict | None = None
    if written_slots and first_error is None:
        try:
            pr = prune(
                backups_root=backups_root,
                state_dir=state_dir,
                config=backups_cfg,
                dry_run=False,
            )
            prune_result_summary = {
                "deleted": len(pr.deleted),
                "preserved_immune": len(pr.preserved_immune),
                "preserved_pre_migration_window": len(pr.preserved_pre_migration_window),
                "disk_used_bytes": pr.disk_usage_after.total_bytes,
                "disk_cap_bytes": pr.disk_usage_after.cap_bytes,
                "invalid_slots": len(pr.invalid_slots),
            }
        except Exception as exc:
            logger.error("run_scheduled_backup: pruning failed (backup still on disk): %s", exc)
            # success stays True — the backup is on disk; pruning is best-effort.

    success = first_error is None
    return ScheduledBackupResult(
        started_at=started_at,
        completed_at=_completed_now(),
        success=success,
        tier=tier,
        label=label,
        written_slots=written_slots,
        skipped_artifacts=skipped_artifacts,
        error=first_error,
        prune_result_summary=prune_result_summary,
    )
