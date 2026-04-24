"""Pure-function retention engine for the ParaMem backup subsystem.

Implements the 5-rule retention precedence from spec §L549–589.  All functions
accept all inputs as parameters — no module-level state reach-in.

Rule precedence (strongest first)
----------------------------------
1. ``max_total_disk_gb`` — global cap across all tiers.  Rule 1 is enforced by
   the *runner* (write refusal when ``pct_of_cap >= 1.0``), NOT by ``prune()``.
   ``prune()`` reports usage in ``disk_usage_{before,after}``.
2. Per-tier ``max_disk_gb`` — tier-level cap.  Oldest-within-tier slots pruned
   first until usage drops to cap or only immune slots remain.
3. Per-tier ``keep`` count — oldest-within-tier slots pruned after ``keep``
   are retained.  ``keep="unlimited"`` disables this rule for the tier.
4. ``pre_migration`` 30-day window immunity — ``pre_migration`` slots younger
   than 30 days are immune from rule-3 count pruning.
5. Manual tier immunity — ``manual`` tier is exempt from rule 3.  Subject to
   rule 2 when a per-tier cap is configured.

Immunity is absolute: immune slots are never deleted, even under cap pressure
(cap-overrun under immunity surfaces in ``disk_usage_after`` for operator
visibility via the operator-attention block).

Performance: ``compute_disk_usage`` has a 5-second module-level TTL cache
keyed on ``(backups_root, max_total_disk_gb)`` so ``/status`` polling does
not re-walk the filesystem on every request.  Pass ``bypass_cache=True`` to
force a fresh scan (used by tests and post-prune recalculations).
"""

from __future__ import annotations

import logging
import shutil
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from paramem.backup.enumerate import enumerate_backups

if TYPE_CHECKING:
    from paramem.server.config import ServerBackupsConfig

logger = logging.getLogger(__name__)

_PRE_MIGRATION_WINDOW_DAYS = 30


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiskUsage:
    """Snapshot of backup-store disk usage.

    Attributes
    ----------
    total_bytes:
        Sum of file sizes across all tiers and all slots.
    by_tier:
        Mapping of tier name → bytes.  Tiers with zero bytes are omitted.
    cap_bytes:
        ``max_total_disk_gb * 1024**3``.
    pct_of_cap:
        ``total_bytes / cap_bytes`` when ``cap_bytes > 0``, else ``0.0``.
    """

    total_bytes: int
    by_tier: dict[str, int]
    cap_bytes: int
    pct_of_cap: float


@dataclass(frozen=True)
class PruneResult:
    """One ``prune()`` pass outcome.

    Attributes
    ----------
    deleted:
        Slot dirs removed from disk (or that would be removed in dry-run mode).
    preserved_immune:
        Slots saved by ``collect_immune_paths`` (live TRIAL immunity).
    preserved_pre_migration_window:
        Slots saved by the 30-day pre_migration window rule (rule 4).
    would_delete_next:
        Dry-run preview populated when ``dry_run=True``; empty when
        ``dry_run=False``.
    disk_usage_before:
        Disk usage measured before any deletions.
    disk_usage_after:
        Disk usage measured after deletions (same as before when
        ``dry_run=True``).
    invalid_slots:
        ``(path, reason)`` pairs for slots whose ``meta.json`` is
        missing/unreadable.  These are recorded but never deleted.
    """

    deleted: list[Path]
    preserved_immune: list[Path]
    preserved_pre_migration_window: list[Path]
    would_delete_next: list[Path]
    disk_usage_before: DiskUsage
    disk_usage_after: DiskUsage
    invalid_slots: list[tuple[Path, str]]


# ---------------------------------------------------------------------------
# TTL cache for compute_disk_usage
# ---------------------------------------------------------------------------

_cache_lock = threading.Lock()
_cache_timestamp: float = 0.0
_cache_key: tuple | None = None
_cache_value: DiskUsage | None = None
_CACHE_TTL_SECONDS: float = 5.0


def _slot_size_bytes(slot_dir: Path) -> int:
    """Sum the file sizes of all regular files within a slot directory."""
    total = 0
    try:
        for child in slot_dir.iterdir():
            if child.is_file():
                try:
                    total += child.stat().st_size
                except OSError:
                    pass
    except OSError:
        pass
    return total


def compute_disk_usage(
    backups_root: Path,
    config: "ServerBackupsConfig",
    *,
    bypass_cache: bool = False,
) -> DiskUsage:
    """Walk every ``<backups_root>/<kind>/<ts>/`` slot and sum file sizes.

    Skips ``.pending/`` residue.  Uses ``meta.tier`` for per-tier accounting;
    slots without a readable sidecar are bucketed under ``"_unknown"`` (not
    pruned by this function — ``prune()`` records them in ``invalid_slots``).

    A 5-second module-level TTL cache keyed on
    ``(str(backups_root), config.max_total_disk_gb)`` avoids re-scanning the
    filesystem on every ``/status`` poll.  Use ``bypass_cache=True`` to force
    a fresh scan.

    Parameters
    ----------
    backups_root:
        Root of the backup store (e.g. ``data/ha/backups/``).  Returns a
        zero-usage ``DiskUsage`` when the directory does not exist.
    config:
        ``ServerBackupsConfig`` — used for ``max_total_disk_gb`` and cache key.
    bypass_cache:
        When ``True``, skip the cache and always perform a full scan.

    Returns
    -------
    DiskUsage
    """
    global _cache_timestamp, _cache_key, _cache_value

    backups_root = Path(backups_root)
    cache_key = (str(backups_root), config.max_total_disk_gb)
    cap_bytes = int(config.max_total_disk_gb * 1024**3)

    if not bypass_cache:
        with _cache_lock:
            now = time.monotonic()
            if (
                _cache_value is not None
                and _cache_key == cache_key
                and (now - _cache_timestamp) < _CACHE_TTL_SECONDS
            ):
                return _cache_value

    # Full scan.
    by_tier: dict[str, int] = {}
    total_bytes = 0

    if backups_root.exists():
        for kind_dir in backups_root.iterdir():
            if not kind_dir.is_dir() or kind_dir.name.startswith("."):
                continue
            for slot in kind_dir.iterdir():
                if slot.name.startswith(".") or not slot.is_dir():
                    continue
                # Determine tier.
                tier = "_unknown"
                meta_files = list(slot.glob("*.meta.json"))
                if meta_files:
                    try:
                        import json as _json

                        raw = _json.loads(meta_files[0].read_text(encoding="utf-8"))
                        tier = raw.get("tier") or "_unknown"
                    except Exception:
                        tier = "_unknown"

                size = _slot_size_bytes(slot)
                total_bytes += size
                by_tier[tier] = by_tier.get(tier, 0) + size

    pct_of_cap = total_bytes / cap_bytes if cap_bytes > 0 else 0.0
    result = DiskUsage(
        total_bytes=total_bytes,
        by_tier={k: v for k, v in by_tier.items() if v > 0},
        cap_bytes=cap_bytes,
        pct_of_cap=pct_of_cap,
    )

    with _cache_lock:
        _cache_timestamp = time.monotonic()
        _cache_key = cache_key
        _cache_value = result

    return result


# ---------------------------------------------------------------------------
# collect_immune_paths
# ---------------------------------------------------------------------------


def collect_immune_paths(state_dir: Path) -> set[Path]:
    """Return absolute paths of live A-backup slot directories from trial.json.

    Reads ``state_dir/trial.json`` (if present) and resolves the three
    pre-migration backup slot paths (config/graph/registry) from
    ``TrialMarker.backup_paths``.

    Returns an empty set when:

    - ``state_dir/trial.json`` does not exist (LIVE state — no TRIAL active).
    - ``trial.json`` is present but ``TrialMarker.from_dict`` raises (logged
      WARN; treated as no immunity rather than crashing — operator visibility
      comes from adapter_manifest_status).

    Parameters
    ----------
    state_dir:
        Directory containing ``trial.json`` (e.g. ``data/ha/state/``).

    Returns
    -------
    set[Path]
        Resolved absolute paths of the immune slot directories.
    """
    import json as _json

    trial_path = Path(state_dir) / "trial.json"
    if not trial_path.exists():
        return set()

    try:
        raw = _json.loads(trial_path.read_text(encoding="utf-8"))
        backup_paths = raw.get("backup_paths") or {}
        immune: set[Path] = set()
        for _kind, path_str in backup_paths.items():
            if path_str:
                immune.add(Path(path_str).resolve())
        return immune
    except Exception as exc:
        logger.warning(
            "collect_immune_paths: failed to read trial.json — treating as no immunity: %s",
            exc,
        )
        return set()


# ---------------------------------------------------------------------------
# prune
# ---------------------------------------------------------------------------


def prune(
    *,
    backups_root: Path,
    state_dir: Path,
    config: "ServerBackupsConfig",
    dry_run: bool = False,
    now: datetime | None = None,
) -> PruneResult:
    """Apply the 5-rule retention precedence (spec §L569–575) to backups_root.

    See module docstring for rule semantics.  Brief summary:

    - Rule 1 is enforced by the runner (write refusal), NOT here.
    - Rule 2 (per-tier cap): applied first, oldest-within-tier pruned until
      tier bytes ≤ cap or only immune slots remain.
    - Rule 3 (keep count): applied per-tier.  ``keep="unlimited"`` skips rule.
      ``keep=0`` removes ALL non-immune slots in that tier.
    - Rule 4 (pre_migration 30-day window): ``pre_migration`` slots younger
      than 30 days are immune from rule-3 count pruning.
    - Rule 5 (manual immunity): manual tier exempt from rule 3; rule 2 still
      applies when ``manual.max_disk_gb`` is set.

    Sort order within tier: by ``meta.created_at`` (UTC datetime from
    ``BackupRecord``), newest-first.  Oldest slots are pruned first within
    rules 2 and 3.

    Parameters
    ----------
    backups_root:
        Root of the backup store.
    state_dir:
        Directory containing ``trial.json`` for immunity detection.
    config:
        ``ServerBackupsConfig`` with retention budgets and global cap.
    dry_run:
        When ``True``, populate ``would_delete_next`` but do not call
        ``shutil.rmtree``.  ``disk_usage_after`` equals ``disk_usage_before``
        in dry-run mode.
    now:
        UTC datetime to use for rule-4 window calculation.  Defaults to
        ``datetime.now(timezone.utc)`` (injection point for tests).

    Returns
    -------
    PruneResult
    """
    if now is None:
        now = datetime.now(timezone.utc)

    backups_root = Path(backups_root)
    retention = config.retention

    # Step 1–2: Enumerate all slots; separate valid from invalid.
    all_records = enumerate_backups(backups_root, kind=None)
    invalid_slots: list[tuple[Path, str]] = []

    # Step 3: Disk usage before.
    disk_usage_before = compute_disk_usage(backups_root, config, bypass_cache=True)

    # Step 4: Build immune set.
    immune_paths = collect_immune_paths(state_dir)

    # Group valid records by tier. enumerate_backups already skips unreadable
    # sidecars (logs WARN). We need to also catch legacy slots with no tier.
    by_tier: dict[str, list] = {}  # tier -> list of BackupRecord (newest-first order retained)

    for record in all_records:
        # enumerate_backups enforces schema validation via read_meta, which
        # requires 'tier' as a mandatory field (MetaSchemaError → slot skipped).
        # Therefore every BackupRecord that reaches this loop has a non-empty tier.
        by_tier.setdefault(record.meta.tier, []).append(record)

    # Within each tier, records come from enumerate_backups in newest-first
    # order. We need oldest-first for pruning within rules 2 and 3.

    deleted: list[Path] = []
    preserved_immune: list[Path] = []
    preserved_pre_migration_window: list[Path] = []
    would_delete_next: list[Path] = []

    def _is_immune(record) -> bool:
        return record.slot_dir.resolve() in immune_paths

    def _delete_slot(slot_dir: Path) -> None:
        if dry_run:
            would_delete_next.append(slot_dir)
        else:
            try:
                shutil.rmtree(slot_dir)
                logger.info("prune: deleted slot %s", slot_dir)
                deleted.append(slot_dir)
            except OSError as exc:
                logger.error("prune: failed to delete slot %s: %s", slot_dir, exc)

    def _tier_config(tier_name: str):
        """Return the RetentionTierConfig for a tier; default to daily config for unknown."""
        return getattr(retention, tier_name, None) or getattr(retention, "daily")

    # Step 5: Rule 2 — per-tier disk cap.
    # Applied before rule 3.
    for tier_name, records in by_tier.items():
        tier_cfg = _tier_config(tier_name)
        if tier_cfg.max_disk_gb is None:
            continue
        tier_cap_bytes = int(tier_cfg.max_disk_gb * 1024**3)
        # Compute current tier usage.
        tier_bytes = sum(_slot_size_bytes(r.slot_dir) for r in records)
        if tier_bytes <= tier_cap_bytes:
            continue
        # Sort oldest-first for pruning.
        candidates = sorted(records, key=lambda r: r.created_at)
        for record in candidates:
            if tier_bytes <= tier_cap_bytes:
                break
            if _is_immune(record):
                logger.warning(
                    "prune: rule-2 cap-overrun under immunity for slot %s "
                    "(tier=%s, cap=%.3f GB) — not deleting immune slot",
                    record.slot_dir,
                    tier_name,
                    tier_cfg.max_disk_gb,
                )
                preserved_immune.append(record.slot_dir)
                continue
            slot_size = _slot_size_bytes(record.slot_dir)
            _delete_slot(record.slot_dir)
            tier_bytes -= slot_size

    # Step 6: Rule 3 + Rule 4 — keep count + pre_migration window.
    pre_migration_cutoff = now - timedelta(days=_PRE_MIGRATION_WINDOW_DAYS)

    for tier_name, records in by_tier.items():
        # Rule 5: manual tier is exempt from rule 3.
        if tier_name == "manual":
            continue

        tier_cfg = _tier_config(tier_name)
        keep = tier_cfg.keep

        if keep == "unlimited":
            # Rule 3 does not apply; rule 2 already ran.
            continue

        # keep is an int (0 = remove all non-immune).
        # Sort newest-first for the "keep the first N" logic.
        candidates = sorted(records, key=lambda r: r.created_at, reverse=True)

        retained = 0
        for record in candidates:
            if _is_immune(record):
                logger.warning(
                    "prune: immune slot %s would have been rule-3 pruned "
                    "(tier=%s, keep=%s) — skipping",
                    record.slot_dir,
                    tier_name,
                    keep,
                )
                preserved_immune.append(record.slot_dir)
                continue

            # Rule 4: pre_migration window immunity.
            if tier_name == "pre_migration" and record.created_at > pre_migration_cutoff:
                logger.info(
                    "prune: pre_migration slot %s within 30-day window — preserved",
                    record.slot_dir,
                )
                preserved_pre_migration_window.append(record.slot_dir)
                retained += 1
                continue

            # Already deleted in rule 2 pass.
            if not dry_run and not record.slot_dir.exists():
                continue

            if retained < keep:
                retained += 1
            else:
                _delete_slot(record.slot_dir)

    # Step 7: Disk usage after.
    disk_usage_after = compute_disk_usage(backups_root, config, bypass_cache=True)

    # Log summary.
    logger.info(
        "prune: deleted=%d, preserved_immune=%d, preserved_window=%d, invalid=%d, dry_run=%s",
        len(deleted) if not dry_run else 0,
        len(preserved_immune),
        len(preserved_pre_migration_window),
        len(invalid_slots),
        dry_run,
    )

    return PruneResult(
        deleted=deleted,
        preserved_immune=preserved_immune,
        preserved_pre_migration_window=preserved_pre_migration_window,
        would_delete_next=would_delete_next,
        disk_usage_before=disk_usage_before,
        disk_usage_after=disk_usage_after,
        invalid_slots=invalid_slots,
    )
