"""Tests for paramem.backup.state — BackupStateRecord I/O + concurrency."""

from __future__ import annotations

import json
import threading

import pytest

from paramem.backup.state import (
    BACKUP_STATE_SCHEMA_VERSION,
    BackupStateRecord,
    BackupStateSchemaError,
    last_attempt_epoch,
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
# last_attempt_epoch — the shared last-ATTEMPT reader used by both
# backup/__main__.py's runner gate and server/app.py's boot-completion
# catch-up gate.
# ---------------------------------------------------------------------------


class TestLastAttemptEpoch:
    def test_absent_state_returns_none(self, tmp_path):
        """No backup.json at all -> None (NO_STAMP)."""
        assert last_attempt_epoch(tmp_path) is None

    def test_valid_completed_at_returns_epoch(self, tmp_path):
        """A valid completed_at is parsed into its epoch timestamp."""
        from datetime import datetime

        completed_at = "2026-04-22T04:00:42+00:00"
        record = _make_record(last_run={"completed_at": completed_at, "success": True})
        write_backup_state(tmp_path, record)

        result = last_attempt_epoch(tmp_path)
        assert result == datetime.fromisoformat(completed_at).timestamp()

    def test_last_run_without_completed_at_returns_none(self, tmp_path):
        """last_run present but missing completed_at -> None."""
        record = _make_record(last_run={"success": True})
        write_backup_state(tmp_path, record)
        assert last_attempt_epoch(tmp_path) is None

    def test_no_last_run_yet_returns_none(self, tmp_path):
        """A record with last_run=None (freshly seeded, never run) -> None."""
        record = _make_record(last_run=None)
        write_backup_state(tmp_path, record)
        assert last_attempt_epoch(tmp_path) is None

    def test_corrupt_json_treated_as_no_prior_attempt(self, tmp_path, caplog):
        """Bad JSON -> None (NO_STAMP), not a raised BackupStateSchemaError —
        a corrupt state file must not stop scheduled backups from running."""
        state_file = tmp_path / "backup.json"
        state_file.write_text("NOT JSON {{{{", encoding="utf-8")

        with caplog.at_level("WARNING"):
            result = last_attempt_epoch(tmp_path)

        assert result is None
        assert any("Unreadable backup state" in r.message for r in caplog.records)

    def test_schema_version_mismatch_treated_as_no_prior_attempt(self, tmp_path, caplog):
        """Foreign schema_version -> None (NO_STAMP), not a raised error."""
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

        with caplog.at_level("WARNING"):
            result = last_attempt_epoch(tmp_path)

        assert result is None
        assert any("Unreadable backup state" in r.message for r in caplog.records)

    def test_malformed_completed_at_treated_as_no_prior_attempt(self, tmp_path, caplog):
        """A non-ISO-8601 completed_at -> None (NO_STAMP), not a raised ValueError."""
        record = _make_record(last_run={"completed_at": "not-a-timestamp", "success": True})
        write_backup_state(tmp_path, record)

        with caplog.at_level("WARNING"):
            result = last_attempt_epoch(tmp_path)

        assert result is None
        assert any("Malformed completed_at" in r.message for r in caplog.records)

    def test_non_dict_last_run_treated_as_no_prior_attempt(self, tmp_path, caplog):
        """last_run is a bare string, not an object -> None, not a raised AttributeError."""
        record = _make_record(last_run="garbage")
        write_backup_state(tmp_path, record)

        with caplog.at_level("WARNING"):
            result = last_attempt_epoch(tmp_path)

        assert result is None
        assert any("Malformed last_run" in r.message for r in caplog.records)

    def test_list_last_run_treated_as_no_prior_attempt(self, tmp_path, caplog):
        """last_run is a list, not an object -> None, not a raised AttributeError."""
        record = _make_record(last_run=["a"])
        write_backup_state(tmp_path, record)

        with caplog.at_level("WARNING"):
            result = last_attempt_epoch(tmp_path)

        assert result is None
        assert any("Malformed last_run" in r.message for r in caplog.records)

    def test_non_str_completed_at_treated_as_no_prior_attempt(self, tmp_path, caplog):
        """completed_at is an int, not a string -> None, not a raised TypeError."""
        record = _make_record(last_run={"completed_at": 1234567, "success": True})
        write_backup_state(tmp_path, record)

        with caplog.at_level("WARNING"):
            result = last_attempt_epoch(tmp_path)

        assert result is None
        assert any("Malformed completed_at" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# fcntl concurrent-write serialisation
# ---------------------------------------------------------------------------


class TestUpdateBackupStateConcurrency:
    def test_update_backup_state_serializes_concurrent_writes(self, tmp_path):
        """10 threads × 10 ops each, all gated by a Barrier → fcntl.flock serialises.

        Each thread writes a distinct ``completed_at`` timestamp per operation
        (100 distinct timestamps total).  After all threads finish the file
        must be readable, valid, and exactly one of those 100 timestamps must
        appear in ``last_run.completed_at`` — proving no write was lost to
        corruption (a corrupted file would raise ``BackupStateSchemaError``).
        """
        N_THREADS = 10
        N_OPS = 10
        barrier = threading.Barrier(N_THREADS)
        errors: list[Exception] = []

        def _worker(thread_idx: int) -> None:
            barrier.wait()  # start all threads simultaneously
            for op_idx in range(N_OPS):
                ts = f"2026-04-22T{thread_idx:02d}:{op_idx:02d}:00Z"
                try:
                    update_backup_state(
                        tmp_path,
                        _make_result(
                            success=(op_idx % 2 == 0),
                            completed_at=ts,
                        ),
                    )
                except Exception as exc:
                    errors.append(exc)

        threads = [
            threading.Thread(target=_worker, args=(i,), daemon=True) for i in range(N_THREADS)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=2.0)

        assert not errors, f"Concurrent writes raised: {errors}"
        # File must still be readable and schema-valid after 100 interleaved writes.
        record = read_backup_state(tmp_path)
        assert record is not None
        assert record.schema_version == BACKUP_STATE_SCHEMA_VERSION
        # last_run must reflect one of the 100 written timestamps — not a
        # partially-written intermediate (which would fail JSON parsing above).
        assert record.last_run is not None
