"""Tests for paramem.backup.state — BackupStateRecord I/O + concurrency."""

from __future__ import annotations

import json
import threading

import pytest

from paramem.backup.state import (
    BACKUP_STATE_SCHEMA_VERSION,
    BackupStateRecord,
    BackupStateSchemaError,
    read_backup_state,
    update_backup_state,
    write_backup_state,
)


def _make_record(**kwargs) -> BackupStateRecord:
    """Return a minimal valid BackupStateRecord."""
    defaults = dict(
        schema_version=BACKUP_STATE_SCHEMA_VERSION,
        last_run={"started_at": "2026-04-22T04:00:00Z", "success": True},
        last_success_at="2026-04-22T04:00:42Z",
        last_failure_at=None,
        last_failure_reason=None,
    )
    defaults.update(kwargs)
    return BackupStateRecord(**defaults)


def _make_result(
    success: bool, completed_at: str = "2026-04-22T04:00:42Z", error: str | None = None
):
    """Minimal ScheduledBackupResult-like object for update_backup_state."""
    from paramem.backup.runner import ScheduledBackupResult

    return ScheduledBackupResult(
        started_at="2026-04-22T04:00:00Z",
        completed_at=completed_at,
        success=success,
        tier="daily",
        label=None,
        written_slots={},
        skipped_artifacts=[],
        error=error,
        prune_result_summary=None,
    )


# ---------------------------------------------------------------------------
# Roundtrip
# ---------------------------------------------------------------------------


class TestWriteThenReadRoundtrip:
    def test_write_then_read_roundtrip(self, tmp_path):
        """Write a BackupStateRecord and read it back — all fields preserved."""
        record = _make_record(
            last_failure_at="2026-04-21T04:00:00Z",
            last_failure_reason="disk_pressure: ...",
        )
        write_backup_state(tmp_path, record)
        loaded = read_backup_state(tmp_path)
        assert loaded is not None
        assert loaded.schema_version == BACKUP_STATE_SCHEMA_VERSION
        assert loaded.last_success_at == record.last_success_at
        assert loaded.last_failure_at == record.last_failure_at
        assert loaded.last_failure_reason == record.last_failure_reason

    def test_read_returns_none_when_absent(self, tmp_path):
        """Empty state_dir → None (LIVE / never run)."""
        result = read_backup_state(tmp_path)
        assert result is None

    def test_read_raises_on_bad_json(self, tmp_path):
        """Corrupt file → BackupStateSchemaError."""
        state_file = tmp_path / "backup.json"
        state_file.write_text("NOT JSON {{{{", encoding="utf-8")
        with pytest.raises(BackupStateSchemaError, match="not valid JSON"):
            read_backup_state(tmp_path)

    def test_read_raises_on_schema_version_mismatch(self, tmp_path):
        """schema_version: 99 → BackupStateSchemaError."""
        state_file = tmp_path / "backup.json"
        state_file.write_text(
            json.dumps(
                {
                    "schema_version": 99,
                    "last_run": None,
                    "last_success_at": None,
                    "last_failure_at": None,
                    "last_failure_reason": None,
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(BackupStateSchemaError, match="schema_version"):
            read_backup_state(tmp_path)


# ---------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------


class TestAtomicWrite:
    def test_atomic_write_no_partial_file(self, tmp_path):
        """After successful write, no .pending residue exists."""
        record = _make_record()
        write_backup_state(tmp_path, record)
        pending_dir = tmp_path / ".pending"
        pending_file = pending_dir / "backup.json"
        assert not pending_file.exists(), "pending file should be removed after atomic rename"
        assert (tmp_path / "backup.json").exists()

    def test_file_content_is_valid_json(self, tmp_path):
        """Written file is valid JSON with schema_version."""
        record = _make_record()
        path = write_backup_state(tmp_path, record)
        raw = json.loads(path.read_text())
        assert raw["schema_version"] == BACKUP_STATE_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# update_backup_state promotion rules
# ---------------------------------------------------------------------------


class TestUpdateBackupState:
    def test_update_promotes_success_at(self, tmp_path):
        """First success populates last_success_at."""
        result = _make_result(success=True, completed_at="2026-04-22T04:00:42Z")
        record = update_backup_state(tmp_path, result)
        assert record.last_success_at == "2026-04-22T04:00:42Z"
        assert record.last_failure_at is None

    def test_subsequent_failure_leaves_success_at_unchanged(self, tmp_path):
        """Success then failure — last_success_at is not cleared."""
        success_result = _make_result(success=True, completed_at="2026-04-22T04:00:00Z")
        update_backup_state(tmp_path, success_result)
        fail_result = _make_result(
            success=False,
            completed_at="2026-04-23T04:00:00Z",
            error="disk_pressure",
        )
        record = update_backup_state(tmp_path, fail_result)
        assert record.last_success_at == "2026-04-22T04:00:00Z"  # preserved
        assert record.last_failure_at == "2026-04-23T04:00:00Z"

    def test_update_promotes_failure_at_and_reason(self, tmp_path):
        """Failure run populates both last_failure_at and last_failure_reason."""
        result = _make_result(
            success=False,
            completed_at="2026-04-22T04:00:00Z",
            error="disk_pressure: ...",
        )
        record = update_backup_state(tmp_path, result)
        assert record.last_failure_at == "2026-04-22T04:00:00Z"
        assert record.last_failure_reason == "disk_pressure: ..."
        assert record.last_success_at is None

    def test_update_does_not_clear_failure_on_success(self, tmp_path):
        """Success after failure keeps last_failure_at/reason for operator visibility."""
        fail_result = _make_result(
            success=False,
            completed_at="2026-04-21T04:00:00Z",
            error="write error",
        )
        update_backup_state(tmp_path, fail_result)
        success_result = _make_result(success=True, completed_at="2026-04-22T04:00:00Z")
        record = update_backup_state(tmp_path, success_result)
        assert record.last_success_at == "2026-04-22T04:00:00Z"
        # Failure fields preserved for operator visibility.
        assert record.last_failure_at == "2026-04-21T04:00:00Z"
        assert record.last_failure_reason == "write error"


# ---------------------------------------------------------------------------
# Fix 3 — fcntl concurrent-write serialisation
# ---------------------------------------------------------------------------


class TestUpdateBackupStateConcurrency:
    def test_update_backup_state_serializes_concurrent_writes(self, tmp_path):
        """Two threads calling update_backup_state with different keys → both survive."""
        # We exploit the fact that update_backup_state reads the current record
        # and then writes a new one. With proper locking, both threads' payloads
        # must be reflected in the final state (as last_run alternating) but
        # crucially neither call may raise or corrupt the file.
        errors = []

        def _write(success: bool, ts: str) -> None:
            try:
                update_backup_state(tmp_path, _make_result(success=success, completed_at=ts))
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=_write, args=(True, "2026-04-22T04:00:00Z"))
        t2 = threading.Thread(target=_write, args=(False, "2026-04-22T05:00:00Z"))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors, f"Concurrent write raised: {errors}"
        # File must be readable and valid.
        record = read_backup_state(tmp_path)
        assert record is not None
        assert record.schema_version == BACKUP_STATE_SCHEMA_VERSION
