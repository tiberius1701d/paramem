"""Unit tests for paramem.server.schedule_state — the two-mark schedule stamp
file: absence, the round trip, and the refusal shapes an on-disk file this
build cannot interpret raises rather than folding into "never stamped".
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from paramem.server.schedule_state import (
    SCHEDULE_STATE_FILENAME,
    SCHEDULE_STATE_SCHEMA_VERSION,
    ScheduleMarks,
    ScheduleStateUnreadable,
    ScheduleStateVersionUnsupported,
    read_marks,
    write_marks,
)


def _state_path(state_dir: Path) -> Path:
    return state_dir / SCHEDULE_STATE_FILENAME


def _write_raw(state_dir: Path, payload) -> Path:
    state_dir.mkdir(parents=True, exist_ok=True)
    path = _state_path(state_dir)
    path.write_text(json.dumps(payload) if not isinstance(payload, str) else payload)
    return path


def _replace_cadence_mark(state_dir: Path, epoch: float) -> None:
    """Change one mark the only way a caller can: read the current pair
    through read_marks, replace one field, write the whole pair back through
    the single writer, write_marks."""
    current = read_marks(state_dir)
    write_marks(
        state_dir,
        ScheduleMarks(
            last_cadence_mark_epoch=epoch,
            last_full_start_epoch=current.last_full_start_epoch,
        ),
    )


def _replace_full_fold_start(state_dir: Path, epoch: float) -> None:
    current = read_marks(state_dir)
    write_marks(
        state_dir,
        ScheduleMarks(
            last_cadence_mark_epoch=current.last_cadence_mark_epoch,
            last_full_start_epoch=epoch,
        ),
    )


# ---------------------------------------------------------------------------
# Absence
# ---------------------------------------------------------------------------


class TestReadMarksAbsentFile:
    def test_no_file_reads_both_marks_none(self, tmp_path: Path) -> None:
        marks = read_marks(tmp_path)
        assert marks == ScheduleMarks(last_cadence_mark_epoch=None, last_full_start_epoch=None)


# ---------------------------------------------------------------------------
# Round trip and read-modify-write via the one writer, write_marks
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_write_marks_then_read_marks_round_trips(self, tmp_path: Path) -> None:
        marks = ScheduleMarks(last_cadence_mark_epoch=100.0, last_full_start_epoch=200.0)
        write_marks(tmp_path, marks)
        assert read_marks(tmp_path) == marks

    def test_write_marks_round_trips_a_null_full_start(self, tmp_path: Path) -> None:
        """A full fold that has never started is a legal null mark."""
        marks = ScheduleMarks(last_cadence_mark_epoch=100.0, last_full_start_epoch=None)
        write_marks(tmp_path, marks)
        assert read_marks(tmp_path) == marks

    def test_write_marks_writes_the_pair_in_one_call(self, tmp_path: Path) -> None:
        write_marks(tmp_path, ScheduleMarks(last_cadence_mark_epoch=1.0, last_full_start_epoch=2.0))
        raw = json.loads(_state_path(tmp_path).read_text())
        assert raw["last_cadence_mark_epoch"] == 1.0
        assert raw["last_full_start_epoch"] == 2.0
        assert raw["schema_version"] == SCHEDULE_STATE_SCHEMA_VERSION


class TestReadModifyWritePreservesTheUntouchedMark:
    """write_marks is the one writer -- a caller that changes a single mark
    reads the pair with read_marks, replaces one field, and writes the whole
    pair back. The untouched field must survive that round trip."""

    def test_replacing_the_cadence_mark_preserves_the_existing_full_start(
        self, tmp_path: Path
    ) -> None:
        write_marks(tmp_path, ScheduleMarks(last_cadence_mark_epoch=1.0, last_full_start_epoch=2.0))
        _replace_cadence_mark(tmp_path, 5.0)
        assert read_marks(tmp_path) == ScheduleMarks(
            last_cadence_mark_epoch=5.0, last_full_start_epoch=2.0
        )

    def test_replacing_the_full_start_preserves_the_existing_cadence_mark(
        self, tmp_path: Path
    ) -> None:
        write_marks(tmp_path, ScheduleMarks(last_cadence_mark_epoch=1.0, last_full_start_epoch=2.0))
        _replace_full_fold_start(tmp_path, 9.0)
        assert read_marks(tmp_path) == ScheduleMarks(
            last_cadence_mark_epoch=1.0, last_full_start_epoch=9.0
        )

    def test_replacing_the_cadence_mark_on_an_absent_file_leaves_full_start_none(
        self, tmp_path: Path
    ) -> None:
        _replace_cadence_mark(tmp_path, 5.0)
        assert read_marks(tmp_path) == ScheduleMarks(
            last_cadence_mark_epoch=5.0, last_full_start_epoch=None
        )

    def test_replacing_the_full_start_on_an_absent_file_leaves_cadence_mark_none(
        self, tmp_path: Path
    ) -> None:
        _replace_full_fold_start(tmp_path, 9.0)
        assert read_marks(tmp_path) == ScheduleMarks(
            last_cadence_mark_epoch=None, last_full_start_epoch=9.0
        )


# ---------------------------------------------------------------------------
# ScheduleStateUnreadable — cause="undecodable"
# ---------------------------------------------------------------------------


class TestUndecodable:
    def test_not_json_raises_undecodable(self, tmp_path: Path) -> None:
        _write_raw(tmp_path, "not json at all {{{")
        with pytest.raises(ScheduleStateUnreadable) as exc_info:
            read_marks(tmp_path)
        assert exc_info.value.cause == "undecodable"

    def test_empty_file_raises_undecodable(self, tmp_path: Path) -> None:
        _write_raw(tmp_path, "")
        with pytest.raises(ScheduleStateUnreadable) as exc_info:
            read_marks(tmp_path)
        assert exc_info.value.cause == "undecodable"

    def test_non_utf8_bytes_raise_undecodable(self, tmp_path: Path) -> None:
        tmp_path.mkdir(parents=True, exist_ok=True)
        _state_path(tmp_path).write_bytes(b"\xff\xfe\x00\x01not utf8")
        with pytest.raises(ScheduleStateUnreadable) as exc_info:
            read_marks(tmp_path)
        assert exc_info.value.cause == "undecodable"


# ---------------------------------------------------------------------------
# ScheduleStateUnreadable — cause="unopenable"
# ---------------------------------------------------------------------------


class TestUnopenable:
    def test_a_directory_at_the_state_path_raises_unopenable(self, tmp_path: Path) -> None:
        tmp_path.mkdir(parents=True, exist_ok=True)
        _state_path(tmp_path).mkdir()
        with pytest.raises(ScheduleStateUnreadable) as exc_info:
            read_marks(tmp_path)
        assert exc_info.value.cause == "unopenable"

    @pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses file permission checks")
    def test_unreadable_permissions_raise_unopenable(self, tmp_path: Path) -> None:
        path = _write_raw(tmp_path, {"schema_version": SCHEDULE_STATE_SCHEMA_VERSION})
        path.chmod(0o000)
        try:
            with pytest.raises(ScheduleStateUnreadable) as exc_info:
                read_marks(tmp_path)
            assert exc_info.value.cause == "unopenable"
        finally:
            path.chmod(0o644)


# ---------------------------------------------------------------------------
# ScheduleStateUnreadable — cause="not_marks"
# ---------------------------------------------------------------------------


class TestNotMarks:
    def test_non_object_top_level_raises_not_marks(self, tmp_path: Path) -> None:
        _write_raw(tmp_path, [1, 2, 3])
        with pytest.raises(ScheduleStateUnreadable) as exc_info:
            read_marks(tmp_path)
        assert exc_info.value.cause == "not_marks"

    def test_bare_json_null_raises_not_marks(self, tmp_path: Path) -> None:
        _write_raw(tmp_path, "null")
        with pytest.raises(ScheduleStateUnreadable) as exc_info:
            read_marks(tmp_path)
        assert exc_info.value.cause == "not_marks"

    def test_missing_schema_version_raises_not_marks(self, tmp_path: Path) -> None:
        _write_raw(
            tmp_path,
            {"last_cadence_mark_epoch": 1.0, "last_full_start_epoch": None},
        )
        with pytest.raises(ScheduleStateUnreadable) as exc_info:
            read_marks(tmp_path)
        assert exc_info.value.cause == "not_marks"

    def test_missing_a_mark_field_raises_not_marks(self, tmp_path: Path) -> None:
        _write_raw(
            tmp_path,
            {
                "schema_version": SCHEDULE_STATE_SCHEMA_VERSION,
                "last_cadence_mark_epoch": 1.0,
                # last_full_start_epoch missing entirely
            },
        )
        with pytest.raises(ScheduleStateUnreadable) as exc_info:
            read_marks(tmp_path)
        assert exc_info.value.cause == "not_marks"


# ---------------------------------------------------------------------------
# ScheduleStateUnreadable — cause="bad_mark"
# ---------------------------------------------------------------------------


class TestBadMark:
    def test_string_mark_raises_bad_mark_naming_the_key(self, tmp_path: Path) -> None:
        _write_raw(
            tmp_path,
            {
                "schema_version": SCHEDULE_STATE_SCHEMA_VERSION,
                "last_cadence_mark_epoch": "not a number",
                "last_full_start_epoch": None,
            },
        )
        with pytest.raises(ScheduleStateUnreadable) as exc_info:
            read_marks(tmp_path)
        assert exc_info.value.cause == "bad_mark"
        assert "last_cadence_mark_epoch" in str(exc_info.value)

    def test_boolean_mark_raises_bad_mark_naming_the_key(self, tmp_path: Path) -> None:
        _write_raw(
            tmp_path,
            {
                "schema_version": SCHEDULE_STATE_SCHEMA_VERSION,
                "last_cadence_mark_epoch": 1.0,
                "last_full_start_epoch": True,
            },
        )
        with pytest.raises(ScheduleStateUnreadable) as exc_info:
            read_marks(tmp_path)
        assert exc_info.value.cause == "bad_mark"
        assert "last_full_start_epoch" in str(exc_info.value)

    def test_object_mark_raises_bad_mark_naming_the_key(self, tmp_path: Path) -> None:
        _write_raw(
            tmp_path,
            {
                "schema_version": SCHEDULE_STATE_SCHEMA_VERSION,
                "last_cadence_mark_epoch": {"nested": 1},
                "last_full_start_epoch": None,
            },
        )
        with pytest.raises(ScheduleStateUnreadable) as exc_info:
            read_marks(tmp_path)
        assert exc_info.value.cause == "bad_mark"
        assert "last_cadence_mark_epoch" in str(exc_info.value)


# ---------------------------------------------------------------------------
# ScheduleStateVersionUnsupported
# ---------------------------------------------------------------------------


class TestVersionUnsupported:
    @pytest.mark.parametrize("version", [1, 3, "2"])
    def test_unsupported_version_raises(self, tmp_path: Path, version) -> None:
        _write_raw(
            tmp_path,
            {
                "schema_version": version,
                "last_cadence_mark_epoch": 1.0,
                "last_full_start_epoch": None,
            },
        )
        with pytest.raises(ScheduleStateVersionUnsupported) as exc_info:
            read_marks(tmp_path)
        assert exc_info.value.version == version

    def test_the_current_version_is_not_refused(self, tmp_path: Path) -> None:
        _write_raw(
            tmp_path,
            {
                "schema_version": SCHEDULE_STATE_SCHEMA_VERSION,
                "last_cadence_mark_epoch": 1.0,
                "last_full_start_epoch": None,
            },
        )
        assert read_marks(tmp_path) == ScheduleMarks(
            last_cadence_mark_epoch=1.0, last_full_start_epoch=None
        )


# ---------------------------------------------------------------------------
# Both writers refuse over every failing shape, leaving the file untouched.
# ---------------------------------------------------------------------------


class TestReadModifyWriteNeverReachesWriteOverAnUninterpretableFile:
    """read_marks refuses over every failing shape, so a read-replace-write
    sequence -- a caller changes one mark by reading the pair, replacing one
    field, and writing the pair -- never reaches the write. The file is left
    byte-identical."""

    @pytest.mark.parametrize(
        ("build_bad_payload",),
        [
            (lambda: "not json {{{",),
            (lambda: [1, 2, 3],),
            (
                lambda: {
                    "schema_version": 1,
                    "last_cadence_mark_epoch": 1.0,
                    "last_full_start_epoch": None,
                },
            ),
        ],
    )
    def test_attempted_cadence_mark_replace_leaves_the_file_untouched(
        self, tmp_path: Path, build_bad_payload
    ) -> None:
        path = _write_raw(tmp_path, build_bad_payload())
        before = path.read_bytes()
        with pytest.raises((ScheduleStateUnreadable, ScheduleStateVersionUnsupported)):
            _replace_cadence_mark(tmp_path, 123.0)
        assert path.read_bytes() == before

    @pytest.mark.parametrize(
        ("build_bad_payload",),
        [
            (lambda: "not json {{{",),
            (lambda: [1, 2, 3],),
            (
                lambda: {
                    "schema_version": 1,
                    "last_cadence_mark_epoch": 1.0,
                    "last_full_start_epoch": None,
                },
            ),
        ],
    )
    def test_attempted_full_start_replace_leaves_the_file_untouched(
        self, tmp_path: Path, build_bad_payload
    ) -> None:
        path = _write_raw(tmp_path, build_bad_payload())
        before = path.read_bytes()
        with pytest.raises((ScheduleStateUnreadable, ScheduleStateVersionUnsupported)):
            _replace_full_fold_start(tmp_path, 456.0)
        assert path.read_bytes() == before
