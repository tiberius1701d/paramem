"""Scheduled backup runner for the ParaMem backup subsystem.

Orchestrates the full backup pipeline:
  1. Schedule guard (``schedule: "off"`` → no-op).
  2. Per-artifact write loop.  The disk-pressure cap (rule 1) is enforced by
     ``backup.write`` / ``backup.write_bundle`` themselves, so a refusal can
     land mid-loop — see the loop's ``disk_pressure_error`` handling.
  3. Post-write pruning (best-effort).
  4. Returns a ``ScheduledBackupResult`` dataclass.

The tier is a parameter (default ``"daily"``) selected by the caller —
the standalone CLI runner, or ``/backup/create`` (which forwards the
request's tier; the scheduled timer delegates with ``tier="daily"``).
Weekly/monthly/yearly tier emission is future work; the schema accepts
those tier names for retention budgets.

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
        ``False`` when a non-cap write error or a disk-cap refusal occurred
        this run, ``True`` otherwise.  A cap refusal can land mid-loop —
        artifacts already written before it stay in ``written_slots``, and
        pruning still runs (it is what relieves the pressure).
    tier:
        Backup tier tag — always ``"daily"`` in production.
    label:
        Optional operator-supplied annotation.
    written_slots:
        Mapping of artifact name → absolute slot path string for artifacts
        written successfully this run.  Never zeroed by a later refusal in
        the same run.
    skipped_artifacts:
        ``(artifact_name, reason)`` pairs for artifacts that were not written.
    error:
        ``repr`` of the first non-cap write exception, or the disk-cap
        refusal message when one occurred, or ``None`` on success.
    prune_result_summary:
        Summary dict of the ``PruneResult`` from this run, or ``None`` when
        pruning did not run — a non-cap write error occurred, or the run
        short-circuited before the write loop (schedule off, ``keep=0``).
        Pruning DOES run after a disk-cap refusal, even when it left
        ``written_slots`` empty: pruning is what relieves the pressure the
        refusal detected, and the persistent over-cap case (every configured
        artifact refused, nothing written this run) is exactly when running
        it matters most.
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
        ``prune()`` call (step 3) handles removal of any existing slots.

    2. **Per-artifact write loop** — for each artifact in
       ``server_config.security.backups.artifacts``:

       - ``"snapshot_bundle"`` → call ``write_bundle(...)`` to produce a
         single self-contained bundle slot under ``backups_root/snapshot/``
         containing the full recovery set (config, registry, adapter weights,
         speaker profiles).  A ``BackupError`` from ``write_bundle`` (e.g.
         episodic adapter missing) surfaces as ``success=False`` with the
         error in ``ScheduledBackupResult.error``.  The bundle requires the
         server context (``PARAMEM_DAILY_PASSPHRASE`` for registry decryption
         and per-tier hash resolution); the standalone runner records a
         degraded/skipped state when the server is unreachable — see
         ``__main__``.
       - ``"config"`` → ``live_config_path.read_bytes()``.  Skip with reason
         ``"config file missing"`` when the file does not exist.
       - ``"graph"``   → ``loop.merger.save_bytes()``.  Skip with an
         accurate "graph unavailable" reason when ``loop is None`` or
         ``loop`` has no ``merger`` attribute.  The graph is in-memory only
         (RAM-only), so the standalone runner cannot capture it — the systemd
         timer delegates to the running server (which holds the loop) when
         reachable; see ``__main__``.
       - ``"registry"`` → ``server_config.paths.key_metadata.read_bytes()``.
         Skip with reason ``"registry empty (no keys yet)"`` when the file
         does not exist.  Write even when the file is empty (0 bytes).

       Each artifact's write door (``backup.write`` / ``backup.write_bundle``)
       enforces the global disk cap (rule 1) itself.  A ``DiskCapExceeded``
       refusal does NOT abort the loop early: it is recorded and every
       remaining artifact (this one included) is marked ``"disk_pressure"`` in
       ``skipped_artifacts``, while any artifact already written earlier in
       this run stays in ``written_slots`` — a cap refusal is not a reason to
       misreport slots that are genuinely on disk.  Any other write exception
       still aborts the remaining loop: record it in ``error`` and mark the
       rest ``"aborted after prior failure"``.

       Note: ``security.backups.artifacts`` still accepts the deprecated
       ``["config", "graph", "registry"]`` list for backward compatibility.
       New installations should use ``["snapshot_bundle"]``.

    3. **Pruning** — runs when ``first_error is None`` AND (at least one
       artifact was written this run OR a disk-cap refusal occurred this
       run).  Gated on ``first_error`` (a non-cap write failure) but NOT on
       a cap refusal: pruning is exactly the operation that relieves the
       pressure a refusal just detected, so it still runs after one — even
       when the refusal left ``written_slots`` empty (the persistent
       over-cap case, and the only case for a single-artifact
       ``snapshot_bundle`` config).  Wrap in ``try/except``: prune failure
       logs ERROR but does not change ``success``.

    4. Returns ``ScheduledBackupResult``.

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
        Backup tier tag.  Always ``"daily"`` in production.
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
    from paramem.backup.backup import write_bundle as backup_write_bundle
    from paramem.backup.retention import prune
    from paramem.backup.types import ArtifactKind, BackupError, DiskCapExceeded

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

    backups_root = Path(backups_root)

    # Step 2: Per-artifact write loop.  The disk cap (rule 1) is enforced by
    # backup.write / backup.write_bundle themselves — see the DiskCapExceeded
    # handling below.
    written_slots: dict[str, str] = {}
    skipped_artifacts: list[tuple[str, str]] = []
    first_error: str | None = None
    disk_pressure_error: str | None = None  # kept apart from first_error

    for artifact_name in artifacts_cfg:
        if disk_pressure_error is not None:
            # Cap already reached this run: every remaining artifact is
            # refused for that reason, not "aborted after prior failure".
            skipped_artifacts.append((artifact_name, "disk_pressure"))
            continue
        if first_error is not None:
            skipped_artifacts.append((artifact_name, "aborted after prior failure"))
            continue

        artifact_bytes: bytes | None = None
        skip_reason: str | None = None

        if artifact_name == "snapshot_bundle":
            # Self-contained bundle: one write_bundle() call captures the full
            # recovery set (config + registry + adapter weights + speaker
            # profiles).  The server context provides PARAMEM_DAILY_PASSPHRASE
            # for registry decryption and per-tier hash resolution.
            adapter_scope = getattr(backups_cfg, "adapter_scope", "live")

            # Build the adapter_dirs mapping for enabled tiers.
            adapter_dirs: dict[str, Path] = {}
            adapters_cfg = getattr(server_config, "adapters", None)
            if adapters_cfg is not None:
                for _tier_name in ("episodic", "semantic", "procedural"):
                    _tier_cfg = getattr(adapters_cfg, _tier_name, None)
                    if _tier_cfg is not None and getattr(_tier_cfg, "enabled", False):
                        _tier_dir = getattr(server_config, "adapter_dir", None)
                        if _tier_dir is not None:
                            adapter_dirs[_tier_name] = Path(_tier_dir) / _tier_name

            # Resolve key_metadata (global registry) and speaker_profiles paths.
            registry_path = Path(server_config.paths.key_metadata)
            data_dir = Path(server_config.paths.data)
            speaker_profiles_path = data_dir / "speaker_profiles.json"

            try:
                bundle_slot = backup_write_bundle(
                    config_path=Path(live_config_path),
                    registry_path=registry_path,
                    adapter_dirs=adapter_dirs,
                    backups_root=backups_root,
                    backups_cfg=backups_cfg,
                    meta_fields={"tier": tier, "label": label},
                    adapter_scope=adapter_scope,
                    speaker_profiles_path=speaker_profiles_path
                    if speaker_profiles_path.exists()
                    else None,
                )
                written_slots["snapshot_bundle"] = str(bundle_slot)
                logger.info(
                    "run_scheduled_backup: wrote snapshot_bundle slot %s",
                    bundle_slot,
                )
            except DiskCapExceeded as exc:
                disk_pressure_error = str(exc)
                logger.error("run_scheduled_backup: %s", disk_pressure_error)
                skipped_artifacts.append((artifact_name, "disk_pressure"))
            except BackupError as exc:
                first_error = repr(exc)
                logger.error("run_scheduled_backup: write_bundle failed: %s", exc)
                skipped_artifacts.append((artifact_name, f"write_bundle error: {exc}"))
            except Exception as exc:
                first_error = repr(exc)
                logger.error("run_scheduled_backup: write_bundle raised unexpectedly: %s", exc)
                skipped_artifacts.append((artifact_name, f"write_bundle error: {exc}"))
            continue

        elif artifact_name == "config":
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
                skip_reason = (
                    "graph unavailable — requires the live consolidation loop "
                    "(server down or cloud-only); the in-memory graph cannot be "
                    "captured by the standalone runner"
                )
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
                backups_root=backups_root,
                backups_cfg=backups_cfg,
            )
            written_slots[artifact_name] = str(slot_dir)
            logger.info("run_scheduled_backup: wrote %s slot %s", artifact_name, slot_dir)
        except DiskCapExceeded as exc:
            disk_pressure_error = str(exc)
            logger.error("run_scheduled_backup: %s", disk_pressure_error)
            skipped_artifacts.append((artifact_name, "disk_pressure"))
        except Exception as exc:
            first_error = repr(exc)
            logger.error("run_scheduled_backup: failed to write %s: %s", artifact_name, exc)
            continue

    # Step 3: Pruning — gated on first_error, NOT on the cap refusal.  A
    # refusal is precisely when pruning is most useful: slots that did land
    # are on disk and the retention rules are what free the space for the
    # next run.
    prune_result_summary: dict | None = None
    if first_error is None and (written_slots or disk_pressure_error is not None):
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
                "preserved_migration_window": len(pr.preserved_migration_window),
                "disk_used_bytes": pr.disk_usage_after.total_bytes,
                "disk_cap_bytes": pr.disk_usage_after.cap_bytes,
                "invalid_slots": len(pr.invalid_slots),
            }
        except Exception as exc:
            logger.error("run_scheduled_backup: pruning failed (backup still on disk): %s", exc)
            # success stays True — the backup is on disk; pruning is best-effort.

    # The two flags are mutually exclusive by construction (whichever is set
    # first short-circuits the other's branch at loop entry), so this
    # selection is unambiguous.
    success = first_error is None and disk_pressure_error is None
    return ScheduledBackupResult(
        started_at=started_at,
        completed_at=_completed_now(),
        success=success,
        tier=tier,
        label=label,
        written_slots=written_slots,
        skipped_artifacts=skipped_artifacts,
        error=first_error if first_error is not None else disk_pressure_error,
        prune_result_summary=prune_result_summary,
    )
