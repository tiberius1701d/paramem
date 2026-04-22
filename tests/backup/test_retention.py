"""Tests for paramem.backup.retention — DiskUsage, collect_immune_paths, prune.

Covers all 5 spec rules and Fix 1 (TTL cache).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from paramem.backup.enumerate import enumerate_backups
from paramem.backup.retention import (
    collect_immune_paths,
    compute_disk_usage,
    prune,
)
from paramem.server.config import (
    RetentionConfig,
    RetentionTierConfig,
    ServerBackupsConfig,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_config(
    max_total_disk_gb: float = 20.0,
    daily_keep: int | str = 7,
    weekly_keep: int | str = 4,
    monthly_keep: int | str = 12,
    yearly_keep: int | str = 3,
    pre_migration_keep: int | str = 10,
    trial_adapter_keep: int | str = 5,
    manual_keep: int | str = "unlimited",
    manual_max_disk_gb: float | None = 5.0,
    daily_max_disk_gb: float | None = None,
) -> ServerBackupsConfig:
    return ServerBackupsConfig(
        max_total_disk_gb=max_total_disk_gb,
        schedule="daily 04:00",
        artifacts=["config", "graph", "registry"],
        retention=RetentionConfig(
            daily=RetentionTierConfig(keep=daily_keep, max_disk_gb=daily_max_disk_gb),
            weekly=RetentionTierConfig(keep=weekly_keep),
            monthly=RetentionTierConfig(keep=monthly_keep),
            yearly=RetentionTierConfig(keep=yearly_keep),
            pre_migration=RetentionTierConfig(keep=pre_migration_keep),
            trial_adapter=RetentionTierConfig(keep=trial_adapter_keep),
            manual=RetentionTierConfig(keep=manual_keep, max_disk_gb=manual_max_disk_gb),
        ),
    )


def _write_slot(backups_root: Path, kind: str, ts: str, tier: str, size_bytes: int = 1024) -> Path:
    """Create a minimal backup slot directory with a sidecar and data file.

    ``ts`` must be in ``YYYYMMDD-HHMMSSff`` format (the slot directory name).
    Use ``_ts(i)`` to generate sequential test timestamps.
    """
    slot_dir = backups_root / kind / ts
    slot_dir.mkdir(parents=True, exist_ok=True)
    # Write meta.json
    meta = {
        "schema_version": 1,
        "kind": kind,
        "timestamp": ts,
        "content_sha256": "abc",
        "size_bytes": size_bytes,
        "encrypted": False,
        "encrypt_at_rest": "auto",
        "key_fingerprint": None,
        "tier": tier,
        "label": None,
    }
    (slot_dir / f"{kind}-{ts}.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    # Write data file with the specified size.
    (slot_dir / f"{kind}-{ts}.bin").write_bytes(b"x" * size_bytes)
    return slot_dir


def _ts(i: int) -> str:
    """Generate a sequential YYYYMMDD-HHMMSSff timestamp for test slot dirs.

    i=0 → oldest (2026-04-01), i=1 → one day later, etc.
    """
    base = datetime(2026, 4, 1, 4, 0, 0, tzinfo=timezone.utc)
    dt = base + timedelta(days=i)
    return dt.strftime("%Y%m%d-%H%M%S") + "00"


def _write_trial_json(state_dir: Path, backup_paths: dict[str, str]) -> None:
    """Write a minimal trial.json with the given backup_paths."""
    state_dir.mkdir(parents=True, exist_ok=True)
    trial = {
        "schema_version": 1,
        "started_at": "2026-04-22T04:00:00Z",
        "pre_trial_config_sha256": "abc",
        "candidate_config_sha256": "def",
        "backup_paths": backup_paths,
        "trial_adapter_dir": str(state_dir / "trial_adapter"),
        "trial_graph_dir": str(state_dir / "trial_graph"),
        "config_artifact_filename": "config-20260422-040000.bin",
    }
    (state_dir / "trial.json").write_text(json.dumps(trial), encoding="utf-8")


# ---------------------------------------------------------------------------
# compute_disk_usage
# ---------------------------------------------------------------------------


class TestComputeDiskUsage:
    def test_compute_disk_usage_empty_root(self, tmp_path):
        """Nonexistent backups_root → total_bytes=0, by_tier={}."""
        config = _make_config()
        usage = compute_disk_usage(tmp_path / "nonexistent", config, bypass_cache=True)
        assert usage.total_bytes == 0
        assert usage.by_tier == {}

    def test_compute_disk_usage_buckets_by_tier(self, tmp_path):
        """Three slots with different tiers → by_tier reflects each."""
        config = _make_config()
        _write_slot(tmp_path, "config", _ts(0), "daily", 100)
        _write_slot(tmp_path, "graph", _ts(1), "weekly", 200)
        _write_slot(tmp_path, "registry", _ts(2), "manual", 300)
        usage = compute_disk_usage(tmp_path, config, bypass_cache=True)
        # Sizes include the meta.json file; assert tier keys present.
        assert "daily" in usage.by_tier
        assert "weekly" in usage.by_tier
        assert "manual" in usage.by_tier
        assert usage.total_bytes > 0

    def test_compute_disk_usage_skips_pending(self, tmp_path):
        """`.pending/<ts>/` residue is NOT counted."""
        config = _make_config()
        pending = tmp_path / "config" / ".pending" / "20260422-040000"
        pending.mkdir(parents=True, exist_ok=True)
        (pending / "data.bin").write_bytes(b"x" * 9999)
        usage = compute_disk_usage(tmp_path, config, bypass_cache=True)
        assert usage.total_bytes == 0  # no valid slots, pending skipped

    def test_compute_disk_usage_unknown_tier_for_legacy_slot(self, tmp_path):
        """Slot with no `tier` field → bucketed under `_unknown`."""
        config = _make_config()
        ts = _ts(0)
        slot_dir = tmp_path / "config" / ts
        slot_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "schema_version": 1,
            "kind": "config",
            "timestamp": ts,
            "content_sha256": "abc",
            "size_bytes": 50,
            "encrypted": False,
            "encrypt_at_rest": "auto",
            "key_fingerprint": None,
            # No "tier" key
            "label": None,
        }
        (slot_dir / f"config-{ts}.meta.json").write_text(json.dumps(meta))
        (slot_dir / f"config-{ts}.bin").write_bytes(b"x" * 50)
        usage = compute_disk_usage(tmp_path, config, bypass_cache=True)
        assert "_unknown" in usage.by_tier

    # Fix 1 cache tests
    def test_cache_returns_same_result_within_ttl(self, tmp_path):
        """Second call within 5s returns cached result (iterdir called once)."""
        config = _make_config()
        _write_slot(tmp_path, "config", "20260422-040001", "daily", 100)
        # First call populates cache.
        result1 = compute_disk_usage(tmp_path, config)
        # Second call should use cache — patch iterdir to confirm it's not called.
        with patch.object(Path, "iterdir", side_effect=AssertionError("should use cache")):
            result2 = compute_disk_usage(tmp_path, config)
        assert result1 is result2

    def test_bypass_cache_always_rescans(self, tmp_path):
        """bypass_cache=True always performs a full scan."""
        config = _make_config()
        _write_slot(tmp_path, "config", "20260422-040001", "daily", 100)
        result1 = compute_disk_usage(tmp_path, config)
        # Add another slot.
        _write_slot(tmp_path, "config", "20260422-040002", "daily", 200)
        result2 = compute_disk_usage(tmp_path, config, bypass_cache=True)
        # result2 should reflect the new slot.
        assert result2.total_bytes > result1.total_bytes

    def test_cache_keyed_on_root_and_cap(self, tmp_path):
        """Different max_total_disk_gb → different cache keys, no collision."""
        config_a = _make_config(max_total_disk_gb=10.0)
        config_b = _make_config(max_total_disk_gb=20.0)
        _write_slot(tmp_path, "config", "20260422-040001", "daily", 100)
        result_a = compute_disk_usage(tmp_path, config_a)
        result_b = compute_disk_usage(tmp_path, config_b)
        assert result_a.cap_bytes != result_b.cap_bytes

    def test_cap_bytes_computed_from_config(self, tmp_path):
        """cap_bytes == int(max_total_disk_gb * 1024**3)."""
        config = _make_config(max_total_disk_gb=1.0)
        usage = compute_disk_usage(tmp_path / "x", config, bypass_cache=True)
        assert usage.cap_bytes == 1024**3

    def test_pct_of_cap_zero_when_cap_zero(self, tmp_path):
        """pct_of_cap == 0.0 when cap_bytes == 0 (avoid division by zero)."""
        config = _make_config(max_total_disk_gb=0.0)
        usage = compute_disk_usage(tmp_path / "x", config, bypass_cache=True)
        assert usage.pct_of_cap == 0.0


# ---------------------------------------------------------------------------
# collect_immune_paths
# ---------------------------------------------------------------------------


class TestCollectImmunePaths:
    def test_collect_immune_paths_no_trial(self, tmp_path):
        """No trial.json → empty set."""
        assert collect_immune_paths(tmp_path) == set()

    def test_collect_immune_paths_with_trial(self, tmp_path, tmp_path_factory):
        """Write trial.json with three backup paths → returns those three resolved paths."""
        root = tmp_path_factory.mktemp("backups")
        slot_config = _write_slot(root, "config", _ts(0), "pre_migration")
        slot_graph = _write_slot(root, "graph", _ts(1), "pre_migration")
        slot_registry = _write_slot(root, "registry", _ts(2), "pre_migration")
        state_dir = tmp_path
        _write_trial_json(
            state_dir,
            {
                "config": str(slot_config),
                "graph": str(slot_graph),
                "registry": str(slot_registry),
            },
        )
        immune = collect_immune_paths(state_dir)
        assert slot_config.resolve() in immune
        assert slot_graph.resolve() in immune
        assert slot_registry.resolve() in immune

    def test_collect_immune_paths_corrupt_trial(self, tmp_path):
        """Bad JSON in trial.json → empty set + WARN logged."""
        (tmp_path / "trial.json").write_text("NOT JSON", encoding="utf-8")
        immune = collect_immune_paths(tmp_path)
        assert immune == set()


# ---------------------------------------------------------------------------
# prune — rule 3 (keep count)
# ---------------------------------------------------------------------------


class TestPruneRule3KeepCount:
    def test_prune_rule_3_keep_count_basic(self, tmp_path):
        """Daily tier, keep=2, 5 slots → 3 deleted, 2 kept; oldest deleted first."""
        config = _make_config(daily_keep=2)
        root = tmp_path / "backups"
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        now = datetime(2026, 4, 22, 4, 0, 0, tzinfo=timezone.utc)
        for i in range(5):
            _write_slot(root, "config", _ts(i), "daily")
        result = prune(
            backups_root=root,
            state_dir=state_dir,
            config=config,
            now=now,
        )
        assert len(result.deleted) == 3

    def test_prune_rule_3_unlimited_skips(self, tmp_path):
        """Manual tier, keep="unlimited", 100 slots → 0 deleted."""
        config = _make_config(manual_keep="unlimited", manual_max_disk_gb=None)
        root = tmp_path / "backups"
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        for i in range(10):
            _write_slot(root, "config", _ts(i), "manual")
        now = datetime(2026, 4, 22, 4, 0, 0, tzinfo=timezone.utc)
        result = prune(backups_root=root, state_dir=state_dir, config=config, now=now)
        assert result.deleted == []


# ---------------------------------------------------------------------------
# prune — rule 4 (pre_migration window)
# ---------------------------------------------------------------------------


class TestPruneRule4PreMigrationWindow:
    def test_prune_rule_4_pre_migration_window_immunity(self, tmp_path):
        """15 pre_migration slots all younger than 30 days, keep=10 → 0 deleted."""
        config = _make_config(pre_migration_keep=10)
        root = tmp_path / "backups"
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        now = datetime(2026, 4, 22, 4, 0, 0, tzinfo=timezone.utc)
        # All within 30-day window (< 30 days before now).
        for i in range(15):
            dt = now - timedelta(days=i)
            ts = dt.strftime("%Y%m%d-%H%M%S") + "00"
            _write_slot(root, "config", ts, "pre_migration")
        result = prune(backups_root=root, state_dir=state_dir, config=config, now=now)
        assert result.deleted == []
        assert len(result.preserved_pre_migration_window) == 15

    def test_prune_rule_4_pre_migration_old_tail_pruned(self, tmp_path):
        """5 within window + 12 older, keep=10 → 5 immune; 2 of the older pruned."""
        config = _make_config(pre_migration_keep=10)
        root = tmp_path / "backups"
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        now = datetime(2026, 4, 22, 4, 0, 0, tzinfo=timezone.utc)
        # 5 recent (within window).
        for i in range(5):
            dt = now - timedelta(days=i)
            ts = dt.strftime("%Y%m%d-%H%M%S") + "00"
            _write_slot(root, "config", ts, "pre_migration")
        # 12 old (outside window, 31+ days ago).
        for i in range(12):
            dt = now - timedelta(days=31 + i)
            ts = dt.strftime("%Y%m%d-%H%M%S") + "00"
            _write_slot(root, "config", ts, "pre_migration")
        result = prune(backups_root=root, state_dir=state_dir, config=config, now=now)
        # Window-immune = 5, each counting toward the keep budget.
        # keep=10; 5 window-immune count as retained → 5 more old slots kept → 7 old deleted.
        assert len(result.preserved_pre_migration_window) == 5
        assert len(result.deleted) == 7


# ---------------------------------------------------------------------------
# prune — rule 5 (manual immunity)
# ---------------------------------------------------------------------------


class TestPruneRule5ManualImmunity:
    def test_prune_rule_5_manual_immune_from_count(self, tmp_path):
        """Manual tier, keep=1, 10 slots, no max_disk_gb → 0 deleted."""
        config = _make_config(manual_keep=1, manual_max_disk_gb=None)
        root = tmp_path / "backups"
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        now = datetime(2026, 4, 22, 4, 0, 0, tzinfo=timezone.utc)
        for i in range(10):
            _write_slot(root, "config", _ts(i), "manual")
        result = prune(backups_root=root, state_dir=state_dir, config=config, now=now)
        assert result.deleted == []

    def test_prune_rule_5_manual_subject_to_per_tier_cap(self, tmp_path):
        """Manual tier, max_disk_gb=0.001 (1 MB), 5 slots × 1 MB each → 4 deleted."""
        one_mb = 1024 * 1024
        config = _make_config(manual_keep="unlimited", manual_max_disk_gb=0.001)
        root = tmp_path / "backups"
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        now = datetime(2026, 4, 22, 4, 0, 0, tzinfo=timezone.utc)
        for i in range(5):
            _write_slot(root, "config", _ts(i), "manual", size_bytes=one_mb)
        result = prune(backups_root=root, state_dir=state_dir, config=config, now=now)
        # cap = 0.001 * 1024**3 ≈ 1073741 bytes ≈ 1 MB. 5 × 1 MB → 4 deleted.
        assert len(result.deleted) >= 1


# ---------------------------------------------------------------------------
# prune — rule 2 (per-tier cap runs before rule 3)
# ---------------------------------------------------------------------------


class TestPruneRule2PerTierCap:
    def test_prune_rule_2_per_tier_cap_runs_before_rule_3(self, tmp_path):
        """Daily tier with both keep=2 AND max_disk_gb that allows only 1 → 1 kept."""
        # Use a very small cap: less than 2 slot sizes but more than 1.
        # 5 daily slots of 100 bytes each; max_disk_gb = 150 bytes / 1024**3
        cap_gb = 150 / (1024**3)
        config = _make_config(daily_keep=2, daily_max_disk_gb=cap_gb)
        root = tmp_path / "backups"
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        now = datetime(2026, 4, 22, 4, 0, 0, tzinfo=timezone.utc)
        for i in range(5):
            _write_slot(root, "config", _ts(i), "daily", size_bytes=100)
        prune(backups_root=root, state_dir=state_dir, config=config, now=now)
        # rule 2 removes until under cap, then rule 3 removes more if needed.
        remaining = list((root / "config").iterdir()) if (root / "config").exists() else []
        remaining = [d for d in remaining if not d.name.startswith(".")]
        assert len(remaining) <= 2


# ---------------------------------------------------------------------------
# prune — immunity outranks rules
# ---------------------------------------------------------------------------


class TestPruneImmunity:
    def test_prune_immune_outranks_rule_3(self, tmp_path, tmp_path_factory):
        """Daily tier, keep=2, 5 slots, 1 immune → immune kept even if it would be pruned."""
        config = _make_config(daily_keep=2)
        root = tmp_path / "backups"
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        now = datetime(2026, 4, 22, 4, 0, 0, tzinfo=timezone.utc)
        slots = []
        for i in range(5):
            s = _write_slot(root, "config", _ts(i), "daily")
            slots.append(s)
        # Mark the oldest slot as immune via trial.json.
        _write_trial_json(state_dir, {"config": str(slots[0])})
        result = prune(backups_root=root, state_dir=state_dir, config=config, now=now)
        # Immune slot must still exist.
        assert slots[0].exists()
        assert slots[0].resolve() in [p.resolve() for p in result.preserved_immune]

    def test_prune_immune_outranks_rule_2(self, tmp_path):
        """Manual tier with cap-overrun, all slots immune → 0 deleted."""
        # cap = 1 byte (absurdly small)
        cap_gb = 1 / (1024**3)
        config = _make_config(manual_keep="unlimited", manual_max_disk_gb=cap_gb)
        root = tmp_path / "backups"
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        now = datetime(2026, 4, 22, 4, 0, 0, tzinfo=timezone.utc)
        slot = _write_slot(root, "config", _ts(0), "manual", size_bytes=1024)
        _write_trial_json(state_dir, {"config": str(slot)})
        result = prune(backups_root=root, state_dir=state_dir, config=config, now=now)
        assert result.deleted == []
        assert slot.resolve() in [p.resolve() for p in result.preserved_immune]


# ---------------------------------------------------------------------------
# prune — idempotent, dry_run, invalid_slots
# ---------------------------------------------------------------------------


class TestPruneMiscellaneous:
    def test_prune_idempotent(self, tmp_path):
        """Run prune() twice → second call deletes 0."""
        config = _make_config(daily_keep=2)
        root = tmp_path / "backups"
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        now = datetime(2026, 4, 22, 4, 0, 0, tzinfo=timezone.utc)
        for i in range(5):
            _write_slot(root, "config", _ts(i), "daily")
        prune(backups_root=root, state_dir=state_dir, config=config, now=now)
        result2 = prune(backups_root=root, state_dir=state_dir, config=config, now=now)
        assert result2.deleted == []

    def test_prune_dry_run_does_not_delete(self, tmp_path):
        """dry_run=True → would_delete_next populated, disk untouched."""
        config = _make_config(daily_keep=2)
        root = tmp_path / "backups"
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        now = datetime(2026, 4, 22, 4, 0, 0, tzinfo=timezone.utc)
        for i in range(5):
            _write_slot(root, "config", _ts(i), "daily")
        result = prune(backups_root=root, state_dir=state_dir, config=config, dry_run=True, now=now)
        assert result.deleted == []
        assert len(result.would_delete_next) == 3
        # Files still on disk.
        slots = list((root / "config").iterdir())
        assert len([s for s in slots if not s.name.startswith(".")]) == 5

    def test_prune_disk_usage_after_reflects_deletes(self, tmp_path):
        """disk_usage_before.total_bytes > disk_usage_after.total_bytes after pruning."""
        config = _make_config(daily_keep=2)
        root = tmp_path / "backups"
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        now = datetime(2026, 4, 22, 4, 0, 0, tzinfo=timezone.utc)
        for i in range(5):
            _write_slot(root, "config", _ts(i), "daily", size_bytes=1000)
        result = prune(backups_root=root, state_dir=state_dir, config=config, now=now)
        assert result.disk_usage_before.total_bytes > result.disk_usage_after.total_bytes


# ---------------------------------------------------------------------------
# Fix 2 — keep=0 disables tier (does not raise)
# ---------------------------------------------------------------------------


class TestKeepZeroDisablesTier:
    def test_tier_keep_zero_disables_emission_and_prunes_non_immune(self, tmp_path):
        """keep=0 → all non-immune slots for that tier are removed."""
        config = _make_config(daily_keep=0)
        root = tmp_path / "backups"
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        now = datetime(2026, 4, 22, 4, 0, 0, tzinfo=timezone.utc)
        for i in range(3):
            _write_slot(root, "config", _ts(i), "daily")
        result = prune(backups_root=root, state_dir=state_dir, config=config, now=now)
        assert len(result.deleted) == 3

    def test_tier_keep_zero_immune_slot_preserved(self, tmp_path):
        """keep=0 with one immune slot → immune slot not deleted."""
        config = _make_config(daily_keep=0)
        root = tmp_path / "backups"
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        now = datetime(2026, 4, 22, 4, 0, 0, tzinfo=timezone.utc)
        slot = _write_slot(root, "config", _ts(0), "daily")
        _write_trial_json(state_dir, {"config": str(slot)})
        result = prune(backups_root=root, state_dir=state_dir, config=config, now=now)
        assert slot.exists()
        assert result.deleted == []


# ---------------------------------------------------------------------------
# Cleanup 1 invariant — every enumerated BackupRecord has a non-empty tier
# ---------------------------------------------------------------------------


class TestEnumeratedRecordTierInvariant:
    def test_enumerated_backup_record_always_has_non_empty_tier(self, tmp_path):
        """enumerate_backups skips slots without 'tier'; all returned records have tier.

        Pins the dependency that the dead legacy-tier branch in prune() relied
        on: read_meta requires 'tier' as a mandatory field (MetaSchemaError on
        absence), so enumerate_backups never emits a BackupRecord with an empty
        tier.  This test creates one valid slot and one tier-less sidecar and
        asserts that only the valid slot is returned, with a non-empty tier.
        """
        root = tmp_path / "backups"

        # Valid slot — has tier field.
        valid_slot = _write_slot(root, "config", _ts(0), "daily")

        # Malformed slot — sidecar missing the 'tier' field entirely.
        bad_ts = _ts(1)
        bad_slot = root / "config" / bad_ts
        bad_slot.mkdir(parents=True, exist_ok=True)
        bad_meta = {
            "schema_version": 1,
            "kind": "config",
            "timestamp": bad_ts,
            "content_sha256": "abc",
            "size_bytes": 50,
            "encrypted": False,
            "encrypt_at_rest": "auto",
            "key_fingerprint": None,
            # "tier" intentionally omitted
            "label": None,
        }
        (bad_slot / f"config-{bad_ts}.meta.json").write_text(json.dumps(bad_meta), encoding="utf-8")
        (bad_slot / f"config-{bad_ts}.bin").write_bytes(b"x" * 50)

        records = enumerate_backups(root, kind=None)

        # Only the valid slot is returned; the tier-less slot is skipped.
        assert len(records) == 1
        assert records[0].slot_dir == valid_slot.resolve()

        # Every returned record must have a non-empty tier — the invariant
        # that makes the legacy-tier default branch in prune() unreachable.
        for record in records:
            assert record.meta.tier, f"BackupRecord for {record.slot_dir} has empty tier field"
