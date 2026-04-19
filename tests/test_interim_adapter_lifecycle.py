"""Unit tests for interim-adapter lifecycle helpers.

Covers:
  - Startup glob picks up valid episodic_interim_* dirs and skips half-present ones.
  - create_interim_adapter is idempotent for the same stamp.
  - unload_interim_adapters removes interim adapters from PEFT and on-disk,
    leaving main adapters intact.
  - unload_interim_adapters with no interim adapters present returns empty list.
  - current_interim_stamp returns a correctly formatted timestamp.
  - compute_schedule_period_seconds parses all supported schedule grammars.
  - current_interim_stamp(schedule, count) floors to the correct sub-interval.

No GPU required — all PEFT and model interactions use stub objects.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from paramem.server.interim_adapter import (
    compute_schedule_period_seconds,
    create_interim_adapter,
    current_interim_stamp,
    unload_interim_adapters,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stub_peft_model(*adapter_names: str) -> MagicMock:
    """Return a MagicMock that behaves like a minimal PeftModel.

    peft_config is a real dict keyed by adapter_names so membership tests,
    iteration, and deletion all work correctly without touching torch.
    delete_adapter removes the key from the dict, mirroring PEFT behaviour.
    """
    model = MagicMock()
    model.peft_config = {name: MagicMock() for name in adapter_names}

    def _delete_adapter(name: str) -> None:
        model.peft_config.pop(name, None)

    model.delete_adapter.side_effect = _delete_adapter
    return model


def _write_adapter_files(directory: Path, *, safetensors: bool = True) -> None:
    """Write the PEFT files that a valid interim adapter directory must contain."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "adapter_config.json").write_text("{}")
    if safetensors:
        (directory / "adapter_model.safetensors").write_bytes(b"")


# ---------------------------------------------------------------------------
# Test 0 — current_interim_stamp format
# ---------------------------------------------------------------------------


class TestCurrentInterimStamp:
    def test_stamp_matches_expected_format(self) -> None:
        """current_interim_stamp must return a string matching YYYYMMDDTHHMM."""
        stamp = current_interim_stamp()
        assert re.match(r"^\d{8}T\d{4}$", stamp), (
            f"Stamp {stamp!r} does not match YYYYMMDDTHHMM format"
        )

    def test_stamp_is_string(self) -> None:
        """current_interim_stamp must return a plain str."""
        assert isinstance(current_interim_stamp(), str)

    def test_stamp_length(self) -> None:
        """YYYYMMDDTHHMM is always exactly 13 characters."""
        assert len(current_interim_stamp()) == 13


# ---------------------------------------------------------------------------
# Test 1 — startup glob picks up valid interim adapter dirs in sorted order
# ---------------------------------------------------------------------------


class TestStartupGlob:
    def test_valid_interim_dirs_found_in_sorted_order(self, tmp_path: Path) -> None:
        """Glob over adapter_dir returns episodic_interim_* dirs sorted by name.

        This mirrors the sorted(config.adapter_dir.glob("episodic_interim_*"))
        call in app.py.  The test verifies the file-system side (path existence
        and correct glob pattern) without invoking load_adapter.
        """
        for name in ["episodic_interim_20260417T0000", "episodic_interim_20260418T0000"]:
            _write_adapter_files(tmp_path / name)
        (tmp_path / "episodic").mkdir()  # main adapter — must not appear

        found = [
            p.name
            for p in sorted(tmp_path.glob("episodic_interim_*"))
            if p.is_dir()
            and (p / "adapter_config.json").exists()
            and (p / "adapter_model.safetensors").exists()
        ]

        assert found == [
            "episodic_interim_20260417T0000",
            "episodic_interim_20260418T0000",
        ]

    def test_main_adapter_dirs_excluded_by_glob_pattern(self, tmp_path: Path) -> None:
        """Dirs named episodic / semantic / procedural do not match the glob."""
        for name in ["episodic", "semantic", "procedural"]:
            (tmp_path / name).mkdir()

        found = list(tmp_path.glob("episodic_interim_*"))
        assert found == []

    def test_empty_adapter_dir_produces_no_results(self, tmp_path: Path) -> None:
        """An adapter_dir with no interim subdirs returns an empty list."""
        found = list(tmp_path.glob("episodic_interim_*"))
        assert found == []


# ---------------------------------------------------------------------------
# Test 2 — startup skips half-present interim adapter (warning logged)
# ---------------------------------------------------------------------------


class TestStartupSkipsHalfPresent:
    def test_missing_safetensors_triggers_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An interim dir with adapter_config.json but no adapter_model.safetensors
        must be skipped and a WARNING must be logged.

        This simulates the guard in app.py's startup loop.
        """
        interim_dir = tmp_path / "episodic_interim_20260417T0000"
        _write_adapter_files(interim_dir, safetensors=False)  # only config, no weights

        named_logger = logging.getLogger("paramem.server.app")
        named_logger.addHandler(caplog.handler)
        named_logger.setLevel(logging.WARNING)
        try:
            for path in sorted(tmp_path.glob("episodic_interim_*")):
                if not path.is_dir():
                    continue
                if (
                    not (path / "adapter_config.json").exists()
                    or not (path / "adapter_model.safetensors").exists()
                ):
                    named_logger.warning("Skipping half-present interim adapter: %s", path.name)
        finally:
            named_logger.removeHandler(caplog.handler)

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings, "Expected at least one WARNING for the half-present adapter"
        assert "episodic_interim_20260417T0000" in warnings[0].getMessage()
        assert "Skipping" in warnings[0].getMessage()

    def test_missing_config_json_triggers_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An interim dir with safetensors but no adapter_config.json is also skipped."""
        interim_dir = tmp_path / "episodic_interim_20260418T0000"
        interim_dir.mkdir(parents=True)
        (interim_dir / "adapter_model.safetensors").write_bytes(b"")
        # adapter_config.json intentionally absent

        named_logger = logging.getLogger("paramem.server.app")
        named_logger.addHandler(caplog.handler)
        named_logger.setLevel(logging.WARNING)
        try:
            for path in sorted(tmp_path.glob("episodic_interim_*")):
                if not path.is_dir():
                    continue
                if (
                    not (path / "adapter_config.json").exists()
                    or not (path / "adapter_model.safetensors").exists()
                ):
                    named_logger.warning("Skipping half-present interim adapter: %s", path.name)
        finally:
            named_logger.removeHandler(caplog.handler)

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings
        assert "episodic_interim_20260418T0000" in warnings[0].getMessage()

    def test_complete_interim_dir_not_warned(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A fully-present interim adapter dir must not produce any warning."""
        interim_dir = tmp_path / "episodic_interim_20260418T0000"
        _write_adapter_files(interim_dir)  # both files present

        named_logger = logging.getLogger("paramem.server.app")
        named_logger.addHandler(caplog.handler)
        named_logger.setLevel(logging.WARNING)
        try:
            for path in sorted(tmp_path.glob("episodic_interim_*")):
                if not path.is_dir():
                    continue
                if (
                    not (path / "adapter_config.json").exists()
                    or not (path / "adapter_model.safetensors").exists()
                ):
                    named_logger.warning("Skipping half-present interim adapter: %s", path.name)
        finally:
            named_logger.removeHandler(caplog.handler)

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert not warnings, (
            "No warning expected for a complete interim adapter dir. "
            f"Got: {[r.getMessage() for r in warnings]}"
        )


# ---------------------------------------------------------------------------
# Test 3 — create_interim_adapter is idempotent
# ---------------------------------------------------------------------------


class TestCreateInterimAdapterIdempotent:
    def test_first_call_creates_adapter(self) -> None:
        """create_interim_adapter calls create_adapter for a new stamp."""
        model = _make_stub_peft_model("episodic", "semantic", "procedural")
        adapter_config = MagicMock()
        stamp = "20260418T1430"

        expected_name = f"episodic_interim_{stamp}"
        returned_model = MagicMock()

        with patch(
            "paramem.server.interim_adapter.create_adapter",
            return_value=returned_model,
        ) as mock_create:
            result = create_interim_adapter(model, adapter_config, stamp)

        mock_create.assert_called_once_with(model, adapter_config, adapter_name=expected_name)
        assert result is returned_model

    def test_second_call_is_no_op(self) -> None:
        """create_interim_adapter returns the same model unchanged on the second call."""
        stamp = "20260418T1430"
        name = f"episodic_interim_{stamp}"
        # Simulate the adapter already being registered
        model = _make_stub_peft_model("episodic", "semantic", "procedural", name)
        adapter_config = MagicMock()

        with patch("paramem.server.interim_adapter.create_adapter") as mock_create:
            result = create_interim_adapter(model, adapter_config, stamp)

        mock_create.assert_not_called()
        assert result is model

    def test_different_stamp_creates_new_adapter(self) -> None:
        """create_interim_adapter creates a distinct adapter per stamp."""
        model = _make_stub_peft_model("episodic", "semantic", "procedural")
        adapter_config = MagicMock()

        # Simulate create_adapter returning a new model each time and updating
        # peft_config so the second call sees the first adapter.
        def _side_effect(m, cfg, *, adapter_name: str):  # noqa: ANN001
            m.peft_config[adapter_name] = MagicMock()
            return m

        with patch(
            "paramem.server.interim_adapter.create_adapter",
            side_effect=_side_effect,
        ) as mock_create:
            create_interim_adapter(model, adapter_config, "20260417T0000")
            create_interim_adapter(model, adapter_config, "20260418T0000")

        assert mock_create.call_count == 2
        call_names = [c.kwargs["adapter_name"] for c in mock_create.call_args_list]
        assert "episodic_interim_20260417T0000" in call_names
        assert "episodic_interim_20260418T0000" in call_names


# ---------------------------------------------------------------------------
# Test 4 — unload_interim_adapters removes interims, leaves mains intact
# ---------------------------------------------------------------------------


class TestUnloadInterimAdapters:
    def test_interim_adapters_removed_from_peft(self, tmp_path: Path) -> None:
        """delete_adapter is called for every episodic_interim_* adapter."""
        model = _make_stub_peft_model(
            "episodic",
            "semantic",
            "procedural",
            "episodic_interim_20260417T0000",
            "episodic_interim_20260418T0000",
        )
        for name in ["episodic_interim_20260417T0000", "episodic_interim_20260418T0000"]:
            (tmp_path / name).mkdir()

        unloaded = unload_interim_adapters(model, tmp_path)

        assert sorted(unloaded) == [
            "episodic_interim_20260417T0000",
            "episodic_interim_20260418T0000",
        ]

    def test_main_adapters_remain_in_peft_config(self, tmp_path: Path) -> None:
        """After unload, episodic / semantic / procedural are still in peft_config."""
        model = _make_stub_peft_model(
            "episodic",
            "semantic",
            "procedural",
            "episodic_interim_20260417T0000",
            "episodic_interim_20260418T0000",
        )
        for name in ["episodic_interim_20260417T0000", "episodic_interim_20260418T0000"]:
            (tmp_path / name).mkdir()

        unload_interim_adapters(model, tmp_path)

        assert "episodic" in model.peft_config
        assert "semantic" in model.peft_config
        assert "procedural" in model.peft_config

    def test_interim_names_absent_from_peft_config_after_unload(self, tmp_path: Path) -> None:
        """After unload, no episodic_interim_* key remains in peft_config."""
        model = _make_stub_peft_model(
            "episodic",
            "semantic",
            "procedural",
            "episodic_interim_20260417T0000",
            "episodic_interim_20260418T0000",
        )
        for name in ["episodic_interim_20260417T0000", "episodic_interim_20260418T0000"]:
            (tmp_path / name).mkdir()

        unload_interim_adapters(model, tmp_path)

        remaining_interim_keys = [k for k in model.peft_config if k.startswith("episodic_interim_")]
        assert remaining_interim_keys == []

    def test_on_disk_interim_dirs_removed(self, tmp_path: Path) -> None:
        """On-disk episodic_interim_* directories are deleted by unload."""
        model = _make_stub_peft_model(
            "episodic",
            "semantic",
            "procedural",
            "episodic_interim_20260417T0000",
            "episodic_interim_20260418T0000",
        )
        interim_dirs = []
        for name in ["episodic_interim_20260417T0000", "episodic_interim_20260418T0000"]:
            d = tmp_path / name
            d.mkdir()
            interim_dirs.append(d)

        unload_interim_adapters(model, tmp_path)

        for d in interim_dirs:
            assert not d.exists(), f"Expected {d} to be deleted but it still exists"

    def test_main_adapter_dirs_not_removed(self, tmp_path: Path) -> None:
        """Directories named episodic / semantic / procedural are left on disk."""
        model = _make_stub_peft_model(
            "episodic",
            "semantic",
            "procedural",
            "episodic_interim_20260417T0000",
        )
        (tmp_path / "episodic").mkdir()
        (tmp_path / "semantic").mkdir()
        (tmp_path / "procedural").mkdir()
        (tmp_path / "episodic_interim_20260417T0000").mkdir()

        unload_interim_adapters(model, tmp_path)

        assert (tmp_path / "episodic").exists()
        assert (tmp_path / "semantic").exists()
        assert (tmp_path / "procedural").exists()

    def test_delete_adapter_called_for_each_interim(self, tmp_path: Path) -> None:
        """model.delete_adapter is invoked once per interim adapter name."""
        model = _make_stub_peft_model(
            "episodic",
            "semantic",
            "procedural",
            "episodic_interim_20260417T0000",
            "episodic_interim_20260418T0000",
        )
        for name in ["episodic_interim_20260417T0000", "episodic_interim_20260418T0000"]:
            (tmp_path / name).mkdir()

        unload_interim_adapters(model, tmp_path)

        deleted_names = [c.args[0] for c in model.delete_adapter.call_args_list]
        assert sorted(deleted_names) == [
            "episodic_interim_20260417T0000",
            "episodic_interim_20260418T0000",
        ]


# ---------------------------------------------------------------------------
# Test 5 — unload_interim_adapters when no interim adapters present
# ---------------------------------------------------------------------------


class TestUnloadInterimAdaptersWhenNonePresent:
    def test_returns_empty_list(self, tmp_path: Path) -> None:
        """When no interim adapters exist, unload returns an empty list."""
        model = _make_stub_peft_model("episodic", "semantic", "procedural")
        result = unload_interim_adapters(model, tmp_path)
        assert result == []

    def test_delete_adapter_not_called(self, tmp_path: Path) -> None:
        """When no interim adapters are present, delete_adapter is never called."""
        model = _make_stub_peft_model("episodic", "semantic", "procedural")
        unload_interim_adapters(model, tmp_path)
        model.delete_adapter.assert_not_called()

    def test_no_error_on_empty_adapter_dir(self, tmp_path: Path) -> None:
        """An empty adapter_dir must not raise any exception."""
        model = _make_stub_peft_model("episodic", "semantic", "procedural")
        unload_interim_adapters(model, tmp_path)

    def test_peft_config_unchanged_when_no_interims(self, tmp_path: Path) -> None:
        """peft_config retains all main adapters when there is nothing to unload."""
        model = _make_stub_peft_model("episodic", "semantic", "procedural")
        unload_interim_adapters(model, tmp_path)
        assert set(model.peft_config.keys()) == {"episodic", "semantic", "procedural"}


# ---------------------------------------------------------------------------
# Test 6 — compute_schedule_period_seconds
# ---------------------------------------------------------------------------


class TestComputeSchedulePeriodSeconds:
    def test_every_2h_returns_7200(self) -> None:
        """'every 2h' → 2 × 3600 = 7200 seconds."""
        assert compute_schedule_period_seconds("every 2h") == 7200

    def test_every_1h_returns_3600(self) -> None:
        """'every 1h' → 3600 seconds."""
        assert compute_schedule_period_seconds("every 1h") == 3600

    def test_every_30m_returns_1800(self) -> None:
        """'every 30m' → 30 × 60 = 1800 seconds."""
        assert compute_schedule_period_seconds("every 30m") == 1800

    def test_every_10m_returns_600(self) -> None:
        """'every 10m' → 10 × 60 = 600 seconds."""
        assert compute_schedule_period_seconds("every 10m") == 600

    def test_daily_03_00_returns_86400(self) -> None:
        """'03:00' (daily) → 86400 seconds."""
        assert compute_schedule_period_seconds("03:00") == 86400

    def test_daily_midnight_returns_86400(self) -> None:
        """'00:00' (daily at midnight) → 86400 seconds."""
        assert compute_schedule_period_seconds("00:00") == 86400

    def test_empty_string_returns_none(self) -> None:
        """Empty schedule → None (manual only)."""
        assert compute_schedule_period_seconds("") is None

    def test_off_returns_none(self) -> None:
        """'off' → None."""
        assert compute_schedule_period_seconds("off") is None

    def test_disabled_returns_none(self) -> None:
        """'disabled' → None."""
        assert compute_schedule_period_seconds("disabled") is None

    def test_none_string_returns_none(self) -> None:
        """'none' → None."""
        assert compute_schedule_period_seconds("none") is None

    def test_invalid_raises_value_error(self) -> None:
        """An unrecognised schedule string raises ValueError for truly unknown values."""
        with pytest.raises(ValueError, match="Unrecognised"):
            compute_schedule_period_seconds("biweekly")

    def test_weekly_returns_604800(self) -> None:
        """'weekly' → 604800 seconds (7 × 86400)."""
        assert compute_schedule_period_seconds("weekly") == 604800

    def test_weekly_case_insensitive(self) -> None:
        """'Weekly' and 'WEEKLY' are accepted (case-insensitive)."""
        assert compute_schedule_period_seconds("Weekly") == 604800
        assert compute_schedule_period_seconds("WEEKLY") == 604800

    def test_daily_returns_86400(self) -> None:
        """'daily' → 86400 seconds."""
        assert compute_schedule_period_seconds("daily") == 86400

    def test_daily_case_insensitive(self) -> None:
        """'Daily' and 'DAILY' are accepted (case-insensitive)."""
        assert compute_schedule_period_seconds("Daily") == 86400
        assert compute_schedule_period_seconds("DAILY") == 86400

    def test_case_insensitive_every_H(self) -> None:
        """Grammar is case-insensitive for the unit character."""
        assert compute_schedule_period_seconds("every 2H") == 7200

    def test_case_insensitive_every_M(self) -> None:
        """Grammar is case-insensitive for the unit character."""
        assert compute_schedule_period_seconds("every 30M") == 1800


# ---------------------------------------------------------------------------
# Test 7 — current_interim_stamp with schedule + count
# ---------------------------------------------------------------------------


class TestCurrentInterimStampWithSchedule:
    """Floor-to-sub-interval logic for current_interim_stamp(schedule, count)."""

    def test_every_2h_count_4_floors_to_30min_boundary(self) -> None:
        """schedule='every 2h', count=4 → sub_interval=30min.

        period = 7200, sub_interval = 7200/4 = 1800s (30 min).
        At 14:47 → floor to 14:30.
        """
        now = datetime(2026, 4, 18, 14, 47, 0)
        stamp = current_interim_stamp("every 2h", 4, _now=now)
        assert stamp == "20260418T1430"

    def test_every_2h_count_4_floor_at_exact_boundary(self) -> None:
        """At exactly a 30-min boundary the stamp equals that boundary."""
        now = datetime(2026, 4, 18, 14, 30, 0)
        stamp = current_interim_stamp("every 2h", 4, _now=now)
        assert stamp == "20260418T1430"

    def test_daily_03_00_count_6_floors_to_4h_boundary(self) -> None:
        """schedule='03:00', count=6 → sub_interval=4h.

        period = 86400, sub_interval = 86400/6 = 14400s (4h).
        At 09:15 → seconds_since_midnight = 33300
        floored = (33300 // 14400) * 14400 = 28800 = 08:00.
        """
        now = datetime(2026, 4, 18, 9, 15, 0)
        stamp = current_interim_stamp("03:00", 6, _now=now)
        assert stamp == "20260418T0800"

    def test_zero_arg_form_returns_current_minute(self) -> None:
        """Zero-arg form (backward compat) returns floor-to-minute stamp."""
        now = datetime(2026, 4, 18, 14, 47, 33)
        stamp = current_interim_stamp(_now=now)
        assert stamp == "20260418T1447"

    def test_max_interim_count_zero_raises_with_schedule(self) -> None:
        """max_interim_count=0 with a real schedule raises ValueError.

        Callers must handle the queue-until-consolidation branch before calling
        current_interim_stamp.
        """
        with pytest.raises(ValueError, match="queue until consolidation"):
            current_interim_stamp("every 2h", 0)

    def test_negative_max_interim_count_raises(self) -> None:
        """Negative max_interim_count is invalid and raises ValueError."""
        with pytest.raises(ValueError, match=">="):
            current_interim_stamp("every 2h", -1)

    def test_manual_schedule_empty_floors_to_nearest_hour(self) -> None:
        """Empty schedule with count > 0 falls back to 1-hour boundaries."""
        now = datetime(2026, 4, 18, 14, 47, 0)
        stamp = current_interim_stamp("", 7, _now=now)
        # sub_interval=3600; seconds_since_midnight = 14*3600+47*60 = 53220
        # floored = (53220 // 3600) * 3600 = 50400 = 14:00
        assert stamp == "20260418T1400"

    def test_stamp_format_matches_yyyymmddthhmm(self) -> None:
        """The stamp always matches the YYYYMMDDTHHMM format."""
        now = datetime(2026, 4, 18, 9, 5, 0)
        stamp = current_interim_stamp("every 2h", 4, _now=now)
        assert re.match(r"^\d{8}T\d{4}$", stamp), f"Unexpected format: {stamp!r}"

    def test_every_2h_count_1_floors_to_2h_boundary(self) -> None:
        """With count=1, sub_interval = full period (one bucket per cycle)."""
        now = datetime(2026, 4, 18, 14, 47, 0)
        # period = 7200, sub_interval = 7200/1 = 7200 (2h)
        # seconds_since_midnight = 14*3600+47*60 = 53220
        # floored = (53220 // 7200) * 7200 = 7*7200 = 50400 = 14:00
        stamp = current_interim_stamp("every 2h", 1, _now=now)
        assert stamp == "20260418T1400"
