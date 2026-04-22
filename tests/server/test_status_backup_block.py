"""Tests for the /status.backup block shape and populator logic.

Tests the BackupBlock model and the populate logic in isolation (pure Python,
no HTTP, no server startup).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from paramem.backup.state import (
    BACKUP_STATE_SCHEMA_VERSION,
    BackupStateRecord,
    write_backup_state,
)
from paramem.server.app import BackupBlock
from paramem.server.config import (
    PathsConfig,
    RetentionConfig,
    RetentionTierConfig,
    SecurityConfig,
    ServerBackupsConfig,
    ServerConfig,
)

# ---------------------------------------------------------------------------
# BackupBlock model shape
# ---------------------------------------------------------------------------


class TestBackupBlockShape:
    def test_default_backup_block(self):
        """BackupBlock() with no args → all defaults valid."""
        block = BackupBlock()
        assert block.schedule == ""
        assert block.last_success_at is None
        assert block.last_failure_at is None
        assert block.last_failure_reason is None
        assert block.next_scheduled_at is None
        assert block.stale is False
        assert block.disk_used_bytes == 0
        assert block.disk_cap_bytes == 0

    def test_backup_block_serialises_to_dict(self):
        """BackupBlock.model_dump() produces the expected keys."""
        block = BackupBlock(
            schedule="daily 04:00",
            last_success_at="2026-04-22T04:00:42Z",
            stale=False,
            disk_used_bytes=1234,
            disk_cap_bytes=21474836480,
        )
        d = block.model_dump()
        assert d["schedule"] == "daily 04:00"
        assert d["last_success_at"] == "2026-04-22T04:00:42Z"
        assert d["disk_used_bytes"] == 1234


# ---------------------------------------------------------------------------
# Populator helpers (unit-level — no server required)
# ---------------------------------------------------------------------------


def _make_server_config(tmp_path: Path, schedule: str = "daily 04:00") -> ServerConfig:
    config = ServerConfig.__new__(ServerConfig)
    config.paths = PathsConfig(
        data=tmp_path / "ha",
        sessions=tmp_path / "ha" / "sessions",
        debug=tmp_path / "ha" / "debug",
    )
    config.security = SecurityConfig(
        backups=ServerBackupsConfig(
            schedule=schedule,
            artifacts=["config", "graph", "registry"],
            max_total_disk_gb=20.0,
            retention=RetentionConfig(
                daily=RetentionTierConfig(keep=7),
                manual=RetentionTierConfig(keep="unlimited", max_disk_gb=5.0),
            ),
        )
    )
    return config


def _populate_backup_block(
    config: ServerConfig,
    state_dir: Path,
    backups_root: Path,
) -> BackupBlock:
    """Inline the populator logic from app.py for testing."""
    import time

    from paramem.backup import retention as _backup_retention
    from paramem.backup import state as _backup_state
    from paramem.backup.timer import _backup_timer_interval_seconds

    _backup_record = None
    try:
        _backup_record = _backup_state.read_backup_state(state_dir)
    except Exception:
        pass

    try:
        _disk_usage = _backup_retention.compute_disk_usage(
            backups_root, config.security.backups, bypass_cache=True
        )
        _disk_used_bytes = _disk_usage.total_bytes
        _disk_cap_bytes = _disk_usage.cap_bytes
    except Exception:
        _disk_used_bytes = 0
        _disk_cap_bytes = int(config.security.backups.max_total_disk_gb * 1024**3)

    _backup_timer_state: dict = {"installed": False}
    _next_scheduled_at = None
    _next_us = _backup_timer_state.get("next_elapse_us") or ""
    if str(_next_us).isdigit():
        _next_epoch = int(_next_us) / 1_000_000
        if _next_epoch - time.time() < 3.15e9:
            _next_scheduled_at = datetime.fromtimestamp(_next_epoch, tz=timezone.utc).isoformat()

    _schedule_str = (config.security.backups.schedule or "").strip().lower()
    _stale = False
    if (
        _backup_record
        and _backup_record.last_success_at
        and _schedule_str not in ("", "off", "disabled", "none")
    ):
        _interval_s = _backup_timer_interval_seconds(_schedule_str)
        if _interval_s and _interval_s > 0:
            try:
                _last_ok = datetime.fromisoformat(_backup_record.last_success_at)
                if _last_ok.tzinfo is None:
                    _last_ok = _last_ok.replace(tzinfo=timezone.utc)
                _age = (datetime.now(timezone.utc) - _last_ok).total_seconds()
                _stale = _age > 2 * _interval_s
            except Exception:
                pass

    return BackupBlock(
        schedule=config.security.backups.schedule,
        last_success_at=_backup_record.last_success_at if _backup_record else None,
        last_failure_at=_backup_record.last_failure_at if _backup_record else None,
        last_failure_reason=_backup_record.last_failure_reason if _backup_record else None,
        next_scheduled_at=_next_scheduled_at,
        stale=_stale,
        disk_used_bytes=_disk_used_bytes,
        disk_cap_bytes=_disk_cap_bytes,
    )


class TestStatusBackupNeverRun:
    def test_status_backup_never_run(self, tmp_path):
        """Fresh server (no state/backup.json) → last_success_at=None, stale=False."""
        config = _make_server_config(tmp_path)
        config.paths.data.mkdir(parents=True, exist_ok=True)
        state_dir = config.paths.data / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        backups_root = config.paths.data / "backups"
        backups_root.mkdir(parents=True, exist_ok=True)

        block = _populate_backup_block(config, state_dir, backups_root)
        assert block.last_success_at is None
        assert block.stale is False
        assert block.disk_used_bytes == 0
        assert block.disk_cap_bytes == int(20.0 * 1024**3)


class TestStatusBackupAfterSuccess:
    def test_status_backup_after_success(self, tmp_path):
        """Pre-populate state/backup.json with success → fields populated."""
        config = _make_server_config(tmp_path)
        config.paths.data.mkdir(parents=True, exist_ok=True)
        state_dir = config.paths.data / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        backups_root = config.paths.data / "backups"
        backups_root.mkdir(parents=True, exist_ok=True)

        record = BackupStateRecord(
            schema_version=BACKUP_STATE_SCHEMA_VERSION,
            last_run=None,
            last_success_at="2026-04-22T04:00:42Z",
            last_failure_at=None,
            last_failure_reason=None,
        )
        write_backup_state(state_dir, record)

        block = _populate_backup_block(config, state_dir, backups_root)
        assert block.last_success_at == "2026-04-22T04:00:42Z"
        assert block.last_failure_at is None


class TestStatusBackupAfterFailure:
    def test_status_backup_after_failure(self, tmp_path):
        """Pre-populate with failure → last_failure_at populated."""
        config = _make_server_config(tmp_path)
        config.paths.data.mkdir(parents=True, exist_ok=True)
        state_dir = config.paths.data / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        backups_root = config.paths.data / "backups"
        backups_root.mkdir(parents=True, exist_ok=True)

        record = BackupStateRecord(
            schema_version=BACKUP_STATE_SCHEMA_VERSION,
            last_run=None,
            last_success_at=None,
            last_failure_at="2026-04-22T04:00:00Z",
            last_failure_reason="disk_pressure: ...",
        )
        write_backup_state(state_dir, record)

        block = _populate_backup_block(config, state_dir, backups_root)
        assert block.last_failure_at == "2026-04-22T04:00:00Z"
        assert block.last_failure_reason == "disk_pressure: ..."
        assert block.last_success_at is None


class TestStatusBackupStaleDetection:
    def test_status_backup_stale_detection(self, tmp_path):
        """last_success_at older than 2× cadence (daily) → stale=True."""
        config = _make_server_config(tmp_path, schedule="daily 04:00")
        config.paths.data.mkdir(parents=True, exist_ok=True)
        state_dir = config.paths.data / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        backups_root = config.paths.data / "backups"
        backups_root.mkdir(parents=True, exist_ok=True)

        # 3 days ago → 3 × 86400 > 2 × 86400 → stale.
        stale_ts = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
        record = BackupStateRecord(
            schema_version=BACKUP_STATE_SCHEMA_VERSION,
            last_run=None,
            last_success_at=stale_ts,
            last_failure_at=None,
            last_failure_reason=None,
        )
        write_backup_state(state_dir, record)

        block = _populate_backup_block(config, state_dir, backups_root)
        assert block.stale is True

    def test_status_backup_stale_false_when_off(self, tmp_path):
        """schedule='off' → stale always False."""
        config = _make_server_config(tmp_path, schedule="off")
        config.paths.data.mkdir(parents=True, exist_ok=True)
        state_dir = config.paths.data / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        backups_root = config.paths.data / "backups"
        backups_root.mkdir(parents=True, exist_ok=True)

        stale_ts = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        record = BackupStateRecord(
            schema_version=BACKUP_STATE_SCHEMA_VERSION,
            last_run=None,
            last_success_at=stale_ts,
            last_failure_at=None,
            last_failure_reason=None,
        )
        write_backup_state(state_dir, record)

        block = _populate_backup_block(config, state_dir, backups_root)
        assert block.stale is False


class TestStatusBackupDiskUsage:
    def test_status_backup_disk_usage_populated(self, tmp_path):
        """Write some slots → disk_used_bytes > 0."""
        from tests.backup.test_retention import _ts, _write_slot

        config = _make_server_config(tmp_path)
        config.paths.data.mkdir(parents=True, exist_ok=True)
        state_dir = config.paths.data / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        backups_root = config.paths.data / "backups"
        backups_root.mkdir(parents=True, exist_ok=True)

        _write_slot(backups_root, "config", _ts(0), "daily", size_bytes=1000)

        block = _populate_backup_block(config, state_dir, backups_root)
        assert block.disk_used_bytes > 0
        assert block.disk_cap_bytes == int(20.0 * 1024**3)

    def test_status_backup_compute_disk_usage_failure_safe(self, tmp_path):
        """Mock compute_disk_usage to raise → block returns 0/cap, no crash."""
        config = _make_server_config(tmp_path)
        config.paths.data.mkdir(parents=True, exist_ok=True)
        state_dir = config.paths.data / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        backups_root = config.paths.data / "backups"
        backups_root.mkdir(parents=True, exist_ok=True)

        _backup_record = None

        with patch(
            "paramem.backup.retention.compute_disk_usage",
            side_effect=RuntimeError("disk scan failed"),
        ):
            try:
                from paramem.backup import retention as _backup_retention

                _disk_usage = _backup_retention.compute_disk_usage(
                    backups_root, config.security.backups, bypass_cache=True
                )
                _disk_used_bytes = _disk_usage.total_bytes
                _disk_cap_bytes = _disk_usage.cap_bytes
            except Exception:
                _disk_used_bytes = 0
                _disk_cap_bytes = int(config.security.backups.max_total_disk_gb * 1024**3)

        block = BackupBlock(
            schedule=config.security.backups.schedule,
            disk_used_bytes=_disk_used_bytes,
            disk_cap_bytes=_disk_cap_bytes,
        )
        assert block.disk_used_bytes == 0
        assert block.disk_cap_bytes == int(20.0 * 1024**3)
