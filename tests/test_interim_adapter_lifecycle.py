"""Unit tests for interim-adapter lifecycle helpers.

Covers:
  - Startup glob picks up valid episodic_interim_* dirs and skips half-present ones.
  - create_interim_adapter is idempotent for the same stamp.
  - unload_interim_adapters removes interim adapters from PEFT and on-disk,
    leaving main adapters intact.
  - unload_interim_adapters with no interim adapters present returns empty list.
  - current_interim_stamp returns a correctly formatted timestamp.
  - compute_schedule_period_seconds parses all supported schedule grammars.
  - current_interim_stamp(refresh_cadence) floors to the correct cadence boundary.
    refresh_cadence IS the sub-interval directly — no division by max_interim_count.

No GPU required — all PEFT and model interactions use stub objects.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from paramem.memory.interim_adapter import (
    create_interim_adapter,
    current_interim_stamp,
    unload_interim_adapters,
)
from paramem.models.loader import main_tier_backup_scope
from paramem.server.schedule_grammar import compute_schedule_period_seconds

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
        stamp = current_interim_stamp("every 1h")
        assert re.match(r"^\d{8}T\d{4}$", stamp), (
            f"Stamp {stamp!r} does not match YYYYMMDDTHHMM format"
        )

    def test_stamp_is_string(self) -> None:
        """current_interim_stamp must return a plain str."""
        assert isinstance(current_interim_stamp("every 1h"), str)

    def test_stamp_length(self) -> None:
        """YYYYMMDDTHHMM is always exactly 13 characters."""
        assert len(current_interim_stamp("every 1h")) == 13


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
            "paramem.memory.interim_adapter.create_adapter",
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

        with patch("paramem.memory.interim_adapter.create_adapter") as mock_create:
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
            "paramem.memory.interim_adapter.create_adapter",
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
        """On-disk interim dirs (under episodic/) are deleted by unload."""
        model = _make_stub_peft_model(
            "episodic",
            "semantic",
            "procedural",
            "episodic_interim_20260417T0000",
            "episodic_interim_20260418T0000",
        )
        # 2026-05-14 hierarchy: interim dirs live under episodic/interim_<stamp>/.
        interim_dirs = []
        episodic_root = tmp_path / "episodic"
        episodic_root.mkdir()
        for stamp in ("20260417T0000", "20260418T0000"):
            d = episodic_root / f"interim_{stamp}"
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
# Test 7 — current_interim_stamp with refresh_cadence
# ---------------------------------------------------------------------------


class TestCurrentInterimStampWithCadence:
    """Floor-to-cadence logic for current_interim_stamp(refresh_cadence).

    refresh_cadence IS the sub-interval directly — no division by
    max_interim_count. Full consolidation period is derived elsewhere
    (ConsolidationScheduleConfig.consolidation_period_string).
    """

    def test_every_30m_floors_to_30min_boundary(self) -> None:
        """refresh_cadence='every 30m' → boundary every 30 minutes from midnight.

        At 14:47 → seconds_since_midnight = 14*3600+47*60 = 53220.
        floored = (53220 // 1800) * 1800 = 29*1800 = 52200 = 14:30.
        """
        now = datetime(2026, 4, 18, 14, 47, 0)
        stamp = current_interim_stamp("every 30m", _now=now)
        assert stamp == "20260418T1430"

    def test_every_30m_floor_at_exact_boundary(self) -> None:
        """At exactly a 30-min boundary the stamp equals that boundary."""
        now = datetime(2026, 4, 18, 14, 30, 0)
        stamp = current_interim_stamp("every 30m", _now=now)
        assert stamp == "20260418T1430"

    def test_every_4h_floors_to_4h_boundary(self) -> None:
        """refresh_cadence='every 4h' → boundary every 4h from midnight.

        At 09:15 → seconds_since_midnight = 33300.
        floored = (33300 // 14400) * 14400 = 28800 = 08:00.
        """
        now = datetime(2026, 4, 18, 9, 15, 0)
        stamp = current_interim_stamp("every 4h", _now=now)
        assert stamp == "20260418T0800"

    def test_daily_hhmm_floors_to_day_boundary(self) -> None:
        """refresh_cadence='03:00' (daily) → 86400s boundary from midnight.

        At 09:15 → seconds_since_midnight = 33300.
        floored = (33300 // 86400) * 86400 = 0 = 00:00.
        """
        now = datetime(2026, 4, 18, 9, 15, 0)
        stamp = current_interim_stamp("03:00", _now=now)
        assert stamp == "20260418T0000"

    def test_off_variant_cadence_floors_to_nearest_hour(self) -> None:
        """Explicit off-variant ("off"/"disabled"/"none") falls back to 1-h boundaries.

        Callers that want to skip stamping entirely should take the
        queue-branch earlier rather than rely on this fallback.
        """
        now = datetime(2026, 4, 18, 14, 47, 0)
        for cadence in ("off", "disabled", "none"):
            stamp = current_interim_stamp(cadence, _now=now)
            # sub_interval=3600; seconds_since_midnight = 53220
            # floored = (53220 // 3600) * 3600 = 50400 = 14:00
            assert stamp == "20260418T1400", (
                f"off-variant {cadence!r} should floor to 14:00, got {stamp}"
            )

    def test_stamp_format_matches_yyyymmddthhmm(self) -> None:
        """The stamp always matches the YYYYMMDDTHHMM format."""
        now = datetime(2026, 4, 18, 9, 5, 0)
        stamp = current_interim_stamp("every 30m", _now=now)
        assert re.match(r"^\d{8}T\d{4}$", stamp), f"Unexpected format: {stamp!r}"

    def test_every_2h_floors_to_2h_boundary(self) -> None:
        """refresh_cadence='every 2h' → boundary every 2h from midnight.

        At 14:47 → seconds_since_midnight = 53220.
        floored = (53220 // 7200) * 7200 = 7*7200 = 50400 = 14:00.
        """
        now = datetime(2026, 4, 18, 14, 47, 0)
        stamp = current_interim_stamp("every 2h", _now=now)
        assert stamp == "20260418T1400"


# ---------------------------------------------------------------------------
# Test 8 — main_tier_backup_scope (paramem.models.loader)
# ---------------------------------------------------------------------------


def _configure_backup_mock(
    model: MagicMock, *adapter_names: str, weights: dict | None = None
) -> None:
    """Configure a MagicMock as a minimal PeftModel-like stub in place.

    ``peft_config``/``weights`` are plain dicts so the patched
    create_adapter/copy_adapter_weights fakes below can mutate them
    directly.  ``delete_adapter`` raises if asked to delete the currently
    active adapter, mirroring PEFT's real hazard — the CM must always
    switch active off a backup before deleting it.
    """
    model.peft_config = {name: MagicMock() for name in adapter_names}
    model.weights = dict(weights) if weights is not None else dict.fromkeys(adapter_names, "orig")
    model.active_adapter = adapter_names[0] if adapter_names else None
    model.set_adapter_calls: list[str] = []
    model.delete_adapter_calls: list[str] = []

    def _set_adapter(name: str) -> None:
        if name not in model.peft_config:
            raise KeyError(name)
        model.active_adapter = name
        model.set_adapter_calls.append(name)

    def _delete_adapter(name: str) -> None:
        if name == model.active_adapter:
            raise RuntimeError(f"cannot delete active adapter {name!r} without switching first")
        del model.peft_config[name]
        model.weights.pop(name, None)
        model.delete_adapter_calls.append(name)

    model.set_adapter.side_effect = _set_adapter
    model.delete_adapter.side_effect = _delete_adapter


def _make_backup_model(*adapter_names: str, weights: dict | None = None) -> MagicMock:
    """Build a MagicMock configured as a PeftModel-like stub.

    ``__class__`` is reassigned to PeftModel so ``isinstance(model,
    PeftModel)`` passes — the same pattern used across this test suite
    (e.g. ``test_procedural.py``).  Mock's own permissive attribute
    machinery (not PeftModel's ``peft_config`` property descriptor) still
    handles every read/write, since ``type(model)`` stays ``MagicMock``;
    only ``isinstance`` is fooled.
    """
    from peft import PeftModel

    model = MagicMock()
    _configure_backup_mock(model, *adapter_names, weights=weights)
    model.__class__ = PeftModel  # satisfies main_tier_backup_scope's runtime contract check
    return model


def _fake_create_adapter(model, config, name):  # noqa: ANN001
    """Fake paramem.models.loader.create_adapter — mints a zero-init slot."""
    model.peft_config[name] = config
    model.weights[name] = "zero-init"
    model.set_adapter(name)
    return model


def _fake_copy_adapter_weights(model, src, dst):  # noqa: ANN001
    """Fake paramem.models.loader.copy_adapter_weights — copies the weight marker."""
    model.weights[dst] = model.weights[src]


class TestMainTierBackupScope:
    """Unit tests for main_tier_backup_scope (paramem.models.loader).

    Fake PeftModel (dict-like peft_config, stub delete_adapter/set_adapter),
    monkeypatched create_adapter/copy_adapter_weights — no GPU.  Mirrors the
    production caller: consolidation.py's _run_fold main_tiers branch.
    """

    def _configs(self) -> dict:
        return {"episodic": MagicMock(), "semantic": MagicMock(), "procedural": MagicMock()}

    def test_cleanup_on_success(self) -> None:
        """A clean exit leaves no *_backup adapter resident."""
        model = _make_backup_model("episodic", "semantic", "procedural")
        with (
            patch("paramem.models.loader.create_adapter", side_effect=_fake_create_adapter),
            patch(
                "paramem.models.loader.copy_adapter_weights",
                side_effect=_fake_copy_adapter_weights,
            ),
        ):
            with main_tier_backup_scope(model, self._configs()) as scope:
                assert "episodic_backup" in scope.model.peft_config
                assert "semantic_backup" in scope.model.peft_config
                assert "procedural_backup" in scope.model.peft_config

        assert [k for k in model.peft_config if k.endswith("_backup")] == []

    def test_cleanup_on_aborted_during_consolidation(self) -> None:
        """AbortedDuringConsolidation propagates unchanged; backups are freed."""
        from paramem.training.consolidation import AbortedDuringConsolidation

        model = _make_backup_model("episodic", "semantic", "procedural")
        with (
            patch("paramem.models.loader.create_adapter", side_effect=_fake_create_adapter),
            patch(
                "paramem.models.loader.copy_adapter_weights",
                side_effect=_fake_copy_adapter_weights,
            ),
        ):
            with pytest.raises(AbortedDuringConsolidation):
                with main_tier_backup_scope(model, self._configs()):
                    raise AbortedDuringConsolidation("training aborted on tier 'episodic'")

        assert [k for k in model.peft_config if k.endswith("_backup")] == []

    def test_finally_teardown_failure_does_not_mask_in_flight_exception(self) -> None:
        """A delete_adapter failure during the finally teardown must not
        replace the in-flight exception — AbortedDuringConsolidation still
        propagates unchanged (not the teardown RuntimeError), and every
        OTHER backup is still cleaned up despite one delete failing."""
        from paramem.training.consolidation import AbortedDuringConsolidation

        model = _make_backup_model("episodic", "semantic", "procedural")
        real_delete_fn = model.delete_adapter.side_effect

        def _delete(name):
            if name == "semantic_backup":
                raise RuntimeError("delete_adapter exploded for semantic_backup")
            real_delete_fn(name)

        model.delete_adapter = _delete

        with (
            patch("paramem.models.loader.create_adapter", side_effect=_fake_create_adapter),
            patch(
                "paramem.models.loader.copy_adapter_weights",
                side_effect=_fake_copy_adapter_weights,
            ),
        ):
            with pytest.raises(AbortedDuringConsolidation):
                with main_tier_backup_scope(model, self._configs()):
                    raise AbortedDuringConsolidation("training aborted on tier 'episodic'")

        # episodic_backup and procedural_backup deleted fine; semantic_backup's
        # delete failure is logged and swallowed, never raised — a leaked
        # backup is preferable to a misrouted abort.
        assert "episodic_backup" not in model.peft_config
        assert "procedural_backup" not in model.peft_config
        assert "semantic_backup" in model.peft_config

    def test_cleanup_on_generic_runtime_error(self) -> None:
        """A generic RuntimeError also propagates unchanged; backups are freed."""
        model = _make_backup_model("episodic", "semantic", "procedural")
        with (
            patch("paramem.models.loader.create_adapter", side_effect=_fake_create_adapter),
            patch(
                "paramem.models.loader.copy_adapter_weights",
                side_effect=_fake_copy_adapter_weights,
            ),
        ):
            with pytest.raises(RuntimeError, match="boom"):
                with main_tier_backup_scope(model, self._configs()):
                    raise RuntimeError("boom")

        assert [k for k in model.peft_config if k.endswith("_backup")] == []

    def test_restore_fires_before_delete_for_both_exception_kinds(self) -> None:
        """Restore (backup→tier copy) happens for every snapshotted tier BEFORE
        the backup-delete loop, for both AbortedDuringConsolidation and a
        generic exception."""
        from paramem.training.consolidation import AbortedDuringConsolidation

        for exc_cls in (AbortedDuringConsolidation, RuntimeError):
            model = _make_backup_model("episodic", "semantic", "procedural")
            call_order: list[tuple] = []

            def _create(m, config, name, _order=call_order):  # noqa: ANN001
                m.peft_config[name] = config
                m.weights[name] = "zero-init"
                m.set_adapter(name)
                return m

            def _copy(m, src, dst, _order=call_order):  # noqa: ANN001
                _order.append(("copy", src, dst))
                m.weights[dst] = m.weights[src]

            real_delete = model.delete_adapter

            def _delete(name, _order=call_order, _real=real_delete):  # noqa: ANN001
                _order.append(("delete", name))
                _real(name)

            model.delete_adapter = _delete

            with (
                patch("paramem.models.loader.create_adapter", side_effect=_create),
                patch("paramem.models.loader.copy_adapter_weights", side_effect=_copy),
            ):
                with pytest.raises(exc_cls):
                    with main_tier_backup_scope(model, self._configs()):
                        raise exc_cls("boom")

            restore_calls = [c for c in call_order if c[0] == "copy" and c[1].endswith("_backup")]
            delete_calls = [c for c in call_order if c[0] == "delete"]
            assert restore_calls, f"no restore fired for {exc_cls}"
            assert delete_calls, f"no delete fired for {exc_cls}"
            last_restore_index = max(call_order.index(c) for c in restore_calls)
            first_delete_index = min(call_order.index(c) for c in delete_calls)
            assert last_restore_index < first_delete_index, (
                f"restore must complete before any backup delete for {exc_cls}: {call_order}"
            )

    def test_no_restore_on_success(self) -> None:
        """A clean exit performs no backup→tier restore copy."""
        model = _make_backup_model("episodic", "semantic", "procedural")
        copy_calls: list[tuple] = []

        def _copy(m, src, dst):  # noqa: ANN001
            copy_calls.append((src, dst))
            m.weights[dst] = m.weights[src]

        with (
            patch("paramem.models.loader.create_adapter", side_effect=_fake_create_adapter),
            patch("paramem.models.loader.copy_adapter_weights", side_effect=_copy),
        ):
            with main_tier_backup_scope(model, self._configs()):
                pass

        restore_calls = [c for c in copy_calls if c[0].endswith("_backup")]
        assert restore_calls == []

    def test_stale_backup_discarded_and_resnapshotted(self) -> None:
        """A leaked pre-existing backup is deleted and re-created from the
        CURRENT tier weights on entry — never left stale to clobber a later
        restore."""
        model = _make_backup_model(
            "episodic",
            "semantic",
            "procedural",
            "episodic_backup",
            weights={
                "episodic": "current",
                "semantic": "current",
                "procedural": "current",
                "episodic_backup": "STALE",
            },
        )
        model.active_adapter = "episodic"

        with (
            patch("paramem.models.loader.create_adapter", side_effect=_fake_create_adapter),
            patch(
                "paramem.models.loader.copy_adapter_weights",
                side_effect=_fake_copy_adapter_weights,
            ),
        ):
            with main_tier_backup_scope(model, self._configs()) as scope:
                assert scope.model.weights["episodic_backup"] == "current", (
                    "stale backup must be discarded and re-snapshotted from the live tier"
                )

    def test_never_deletes_active_adapter_and_never_empties_peft_config(self) -> None:
        """Active is always switched to a main tier before any backup delete;
        peft_config never drops below the 3 main tiers."""
        model = _make_backup_model("episodic", "semantic", "procedural")
        min_size: list[int] = []
        real_delete = model.delete_adapter

        def _delete(name, _real=real_delete):  # noqa: ANN001
            _real(name)  # raises if name is still active — the invariant under test
            min_size.append(len(model.peft_config))

        model.delete_adapter = _delete

        with (
            patch("paramem.models.loader.create_adapter", side_effect=_fake_create_adapter),
            patch(
                "paramem.models.loader.copy_adapter_weights",
                side_effect=_fake_copy_adapter_weights,
            ),
        ):
            with main_tier_backup_scope(model, self._configs()) as scope:
                # Leave active on a backup right before exit — the CM's finally
                # must move it back to a main tier before deleting anything.
                scope.model.set_adapter("procedural_backup")

        assert min_size, "expected at least one backup delete"
        assert all(size >= 3 for size in min_size), (
            "peft_config must never drop below the 3 main tiers"
        )
        assert model.active_adapter in ("episodic", "semantic", "procedural")

    def test_partial_enter_double_fault_preserves_original_exception(self) -> None:
        """tier-2's enter copy_adapter_weights raises: tier-1's backup is
        cleaned up, tier-2's half-populated backup is cleaned up (never
        restored — it was never snapshotted), the with body never runs, and
        the ORIGINAL exception type propagates unchanged."""
        model = _make_backup_model("episodic", "semantic", "procedural")
        restore_calls: list[tuple] = []

        def _copy(m, src, dst):  # noqa: ANN001
            restore_calls.append((src, dst))
            if dst == "semantic_backup":
                raise RuntimeError("copy failed for semantic_backup")
            m.weights[dst] = m.weights.get(src, "orig")

        body_ran = False

        with (
            patch("paramem.models.loader.create_adapter", side_effect=_fake_create_adapter),
            patch("paramem.models.loader.copy_adapter_weights", side_effect=_copy),
        ):
            with pytest.raises(RuntimeError, match="copy failed for semantic_backup"):
                with main_tier_backup_scope(model, self._configs()):
                    body_ran = True

        assert not body_ran, "the with body must not run when __enter__ itself fails"
        # tier-2 (semantic) was never snapshotted — no restore copy targeting
        # "semantic" (dst) with src="semantic_backup" may have fired.
        assert ("semantic_backup", "semantic") not in restore_calls
        # Both the tier-1 backup and the half-populated tier-2 backup are freed.
        assert [k for k in model.peft_config if k.endswith("_backup")] == []

    def test_create_adapter_return_value_captured(self) -> None:
        """scope.model reflects whatever create_adapter returns, even a new object.

        ``replacement`` is a plain (non-PeftModel-classed) MagicMock — the CM
        never re-checks isinstance after entry, only reads/writes attributes
        on whatever ``create_adapter`` hands back.
        """
        model = _make_backup_model("episodic")
        replacement = MagicMock()
        _configure_backup_mock(replacement, "episodic")
        replacement.peft_config = model.peft_config
        replacement.weights = model.weights

        def _create_new_object(m, config, name):  # noqa: ANN001
            m.peft_config[name] = config
            m.weights[name] = "zero-init"
            replacement.set_adapter(name)
            return replacement

        with (
            patch("paramem.models.loader.create_adapter", side_effect=_create_new_object),
            patch(
                "paramem.models.loader.copy_adapter_weights",
                side_effect=_fake_copy_adapter_weights,
            ),
        ):
            with main_tier_backup_scope(model, self._configs(), tiers=("episodic",)) as scope:
                assert scope.model is replacement
                assert scope.model is not model
