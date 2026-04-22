"""Unit tests for _collect_backup_items populator (Slice 6b).

Tests cover:
- Empty state file / no state → []
- FAILED alert emission
- STALE alert emission (and skip when schedule="off")
- DISK PRESSURE (info at 80-99%, failed at >=100%)
- All three fire simultaneously with correct ordering
- Malformed state (BackupStateSchemaError) → no crash
- config=None → []
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from paramem.server.attention import _collect_backup_items
from paramem.server.config import (
    PathsConfig,
    SecurityConfig,
    ServerBackupsConfig,
    ServerConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(
    tmp_path: Path,
    schedule: str = "every 1h",
    max_total_disk_gb: float = 20.0,
) -> ServerConfig:
    """Build a minimal ServerConfig with configurable schedule."""
    config = ServerConfig.__new__(ServerConfig)
    config.paths = PathsConfig(
        data=tmp_path / "ha",
        sessions=tmp_path / "ha" / "sessions",
        debug=tmp_path / "ha" / "debug",
    )
    config.paths.data.mkdir(parents=True, exist_ok=True)
    config.security = SecurityConfig(
        backups=ServerBackupsConfig(
            schedule=schedule,
            artifacts=["config", "graph", "registry"],
            max_total_disk_gb=max_total_disk_gb,
        )
    )
    return config


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _ago(seconds: int) -> str:
    return _iso(datetime.now(timezone.utc) - timedelta(seconds=seconds))


# ---------------------------------------------------------------------------
# Test 12 — empty state file → []
# ---------------------------------------------------------------------------


class TestBackupItemsEmptyWhenNoStateFile:
    def test_backup_items_empty_when_no_state_file(self, tmp_path: Path) -> None:
        """read_backup_state returns None and usage < 80% → []."""
        config = _make_config(tmp_path)
        state = {}

        items = _collect_backup_items(state, config)
        assert items == []


# ---------------------------------------------------------------------------
# Test 13 — FAILED alert emitted
# ---------------------------------------------------------------------------


class TestBackupItemsEmitsFailed:
    def test_backup_items_emits_failed(self, tmp_path: Path) -> None:
        """last_failure_at newer than last_success_at → backup_failed item."""
        config = _make_config(tmp_path)
        state_dir = (config.paths.data / "state").resolve()
        state_dir.mkdir(parents=True, exist_ok=True)

        from paramem.backup.state import BackupStateRecord, write_backup_state

        now = datetime.now(timezone.utc)
        record = BackupStateRecord(
            schema_version=1,
            last_run=None,
            last_success_at=_iso(now - timedelta(hours=2)),
            last_failure_at=_iso(now - timedelta(hours=1)),
            last_failure_reason="disk full",
        )
        write_backup_state(state_dir, record)

        items = _collect_backup_items({}, config)
        assert len(items) >= 1
        failed_items = [i for i in items if i.kind == "backup_failed"]
        assert len(failed_items) == 1
        assert failed_items[0].level == "failed"
        assert "FAILED" in failed_items[0].summary
        assert "disk full" in failed_items[0].summary
        assert failed_items[0].action_hint is not None
        assert "logs" in failed_items[0].action_hint.lower()


# ---------------------------------------------------------------------------
# Test 14 — STALE alert emitted
# ---------------------------------------------------------------------------


class TestBackupItemsEmitsStale:
    def test_backup_items_emits_stale(self, tmp_path: Path) -> None:
        """last_success_at older than 2× interval → backup_stale item."""
        config = _make_config(tmp_path, schedule="every 1h")
        state_dir = (config.paths.data / "state").resolve()
        state_dir.mkdir(parents=True, exist_ok=True)

        from paramem.backup.state import BackupStateRecord, write_backup_state

        # 3 hours ago → 2× interval (2h) exceeded
        record = BackupStateRecord(
            schema_version=1,
            last_run=None,
            last_success_at=_ago(3 * 3600),
            last_failure_at=None,
            last_failure_reason=None,
        )
        write_backup_state(state_dir, record)

        items = _collect_backup_items({}, config)
        stale_items = [i for i in items if i.kind == "backup_stale"]
        assert len(stale_items) == 1
        assert stale_items[0].level == "info"
        assert "STALE" in stale_items[0].summary
        assert "backup-create" in stale_items[0].action_hint.lower()


# ---------------------------------------------------------------------------
# Test 15 — STALE skipped when schedule="off"
# ---------------------------------------------------------------------------


class TestBackupItemsSkipsStaleWhenScheduleOff:
    def test_backup_items_skips_stale_when_schedule_off(self, tmp_path: Path) -> None:
        """schedule='off' → no STALE item even when age is huge."""
        config = _make_config(tmp_path, schedule="off")
        state_dir = (config.paths.data / "state").resolve()
        state_dir.mkdir(parents=True, exist_ok=True)

        from paramem.backup.state import BackupStateRecord, write_backup_state

        record = BackupStateRecord(
            schema_version=1,
            last_run=None,
            last_success_at=_ago(365 * 86400),  # 1 year ago
            last_failure_at=None,
            last_failure_reason=None,
        )
        write_backup_state(state_dir, record)

        items = _collect_backup_items({}, config)
        stale_items = [i for i in items if i.kind == "backup_stale"]
        assert stale_items == []


# ---------------------------------------------------------------------------
# Test 16 — DISK PRESSURE info at 80–99%
# ---------------------------------------------------------------------------


class TestBackupItemsEmitsDiskPressureInfoAt80To99:
    def test_backup_items_emits_disk_pressure_info_at_85_pct(self, tmp_path: Path) -> None:
        """pct=0.85 → backup_disk_pressure item with level=info."""
        cap_gb = 0.001  # 1 MB cap
        config = _make_config(tmp_path, max_total_disk_gb=cap_gb)

        backups_root = config.paths.data / "backups"
        backups_root.mkdir(parents=True, exist_ok=True)

        # Seed 850 KB (85% of 1 MB).
        slot = backups_root / "config" / "20260421-040000"
        slot.mkdir(parents=True)
        (slot / "config.bin").write_bytes(b"x" * int(cap_gb * 1024**3 * 0.85))

        items = _collect_backup_items({}, config)
        pressure_items = [i for i in items if i.kind == "backup_disk_pressure"]
        assert len(pressure_items) == 1
        assert pressure_items[0].level == "info"
        assert "DISK" in pressure_items[0].summary
        assert "backup-prune" in pressure_items[0].action_hint.lower()


# ---------------------------------------------------------------------------
# Test 17 — DISK PRESSURE failed at >=100%
# ---------------------------------------------------------------------------


class TestBackupItemsEmitsDiskPressureFailedAt100:
    def test_backup_items_emits_disk_pressure_failed_at_100_pct(self, tmp_path: Path) -> None:
        """pct>=1.0 → backup_disk_pressure item with level=failed."""
        cap_gb = 0.001  # 1 MB cap
        config = _make_config(tmp_path, max_total_disk_gb=cap_gb)

        backups_root = config.paths.data / "backups"
        backups_root.mkdir(parents=True, exist_ok=True)

        # Seed 1.2 MB (120% of 1 MB cap).
        slot = backups_root / "config" / "20260421-040000"
        slot.mkdir(parents=True)
        (slot / "config.bin").write_bytes(b"x" * int(cap_gb * 1024**3 * 1.2))

        items = _collect_backup_items({}, config)
        pressure_items = [i for i in items if i.kind == "backup_disk_pressure"]
        assert len(pressure_items) == 1
        assert pressure_items[0].level == "failed"


# ---------------------------------------------------------------------------
# Test 18 — all three fire simultaneously, order = failed → disk → stale
# ---------------------------------------------------------------------------


class TestBackupItemsAllThreeFireSimultaneously:
    def test_all_three_fire_in_correct_order(self, tmp_path: Path) -> None:
        """Seed conditions for all 3 → order = [failed, disk_pressure, stale]."""
        cap_gb = 0.001
        config = _make_config(tmp_path, schedule="every 1h", max_total_disk_gb=cap_gb)
        state_dir = (config.paths.data / "state").resolve()
        state_dir.mkdir(parents=True, exist_ok=True)

        from paramem.backup.state import BackupStateRecord, write_backup_state

        now = datetime.now(timezone.utc)
        # FAILED condition: failure newer than success, stale too.
        record = BackupStateRecord(
            schema_version=1,
            last_run=None,
            last_success_at=_ago(4 * 3600),  # 4h ago → stale (>2h)
            last_failure_at=_iso(now - timedelta(hours=1)),
            last_failure_reason="disk full",
        )
        write_backup_state(state_dir, record)

        # DISK PRESSURE condition: 90% full.
        backups_root = config.paths.data / "backups"
        backups_root.mkdir(parents=True, exist_ok=True)
        slot = backups_root / "config" / "20260421-040000"
        slot.mkdir(parents=True)
        (slot / "config.bin").write_bytes(b"x" * int(cap_gb * 1024**3 * 0.90))

        items = _collect_backup_items({}, config)
        kinds = [i.kind for i in items]
        assert "backup_failed" in kinds
        assert "backup_disk_pressure" in kinds
        assert "backup_stale" in kinds

        # Ordering check.
        idx_failed = kinds.index("backup_failed")
        idx_disk = kinds.index("backup_disk_pressure")
        idx_stale = kinds.index("backup_stale")
        assert idx_failed < idx_disk < idx_stale, f"Wrong order: {kinds}"


# ---------------------------------------------------------------------------
# Test 19 — BackupStateSchemaError → no crash
# ---------------------------------------------------------------------------


class TestBackupItemsMalformedStateDoesNotCrash:
    def test_malformed_state_does_not_crash(self, tmp_path: Path) -> None:
        """BackupStateSchemaError raised → returns [] or only disk-pressure items."""
        config = _make_config(tmp_path)

        # Write a corrupt state file.
        state_dir = (config.paths.data / "state").resolve()
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / "backup.json").write_text("not valid json {{{", encoding="utf-8")

        # Should not raise.
        items = _collect_backup_items({}, config)
        assert isinstance(items, list)
        # No traceback; only disk-pressure items may appear (no failed/stale).
        bad_kinds = {i.kind for i in items}
        assert "backup_failed" not in bad_kinds
        assert "backup_stale" not in bad_kinds


# ---------------------------------------------------------------------------
# Test 20 — config=None → []
# ---------------------------------------------------------------------------


class TestBackupItemsConfigNoneReturnsEmpty:
    def test_config_none_returns_empty(self) -> None:
        """config=None (unit-test shim) → []."""
        items = _collect_backup_items({}, None)
        assert items == []
