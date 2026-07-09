"""Unit tests for paramem.server.schedule_state — the durable last-attempt
stamp backing the consolidation scheduler's suspend/power-off catch-up gate.
"""

from __future__ import annotations

from pathlib import Path

from paramem.server.schedule_state import (
    SCHEDULE_STATE_FILENAME,
    read_last_scheduled_run,
    write_last_scheduled_run,
)


class TestScheduleStateRoundtrip:
    def test_write_then_read_roundtrips(self, tmp_path: Path):
        write_last_scheduled_run(tmp_path, 1_700_000_000.0)
        assert read_last_scheduled_run(tmp_path) == 1_700_000_000.0

    def test_absent_file_returns_none(self, tmp_path: Path):
        assert read_last_scheduled_run(tmp_path) is None

    def test_overwrite_replaces_previous_value(self, tmp_path: Path):
        write_last_scheduled_run(tmp_path, 100.0)
        write_last_scheduled_run(tmp_path, 200.0)
        assert read_last_scheduled_run(tmp_path) == 200.0

    def test_no_pending_file_left_after_write(self, tmp_path: Path):
        write_last_scheduled_run(tmp_path, 1_700_000_000.0)
        assert (tmp_path / SCHEDULE_STATE_FILENAME).exists()
        assert not (tmp_path / ".pending" / SCHEDULE_STATE_FILENAME).exists()

    def test_creates_state_dir_if_absent(self, tmp_path: Path):
        state_dir = tmp_path / "state"
        assert not state_dir.exists()
        write_last_scheduled_run(state_dir, 1_700_000_000.0)
        assert read_last_scheduled_run(state_dir) == 1_700_000_000.0
