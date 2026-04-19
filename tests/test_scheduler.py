"""Unit tests for the consolidation timer parser + reconciler.

Schedule parsing lives in `paramem.server.systemd_timer.parse_schedule` since
the server delegates actual scheduling to a systemd user timer.
"""

import subprocess
from unittest.mock import patch

import pytest

from paramem.server import systemd_timer
from paramem.server.systemd_timer import TimerSpec, parse_schedule


class TestParseSchedule:
    def test_daily_hhmm(self):
        assert parse_schedule("02:00") == TimerSpec(kind="daily", on_calendar="*-*-* 02:00:00")
        assert parse_schedule("23:59") == TimerSpec(kind="daily", on_calendar="*-*-* 23:59:00")
        assert parse_schedule("0:00") == TimerSpec(kind="daily", on_calendar="*-*-* 00:00:00")

    def test_interval_hours(self):
        assert parse_schedule("every 2h") == TimerSpec(
            kind="interval", on_boot_sec="2h", on_unit_active_sec="2h"
        )
        assert parse_schedule("every 24h") == TimerSpec(
            kind="interval", on_boot_sec="24h", on_unit_active_sec="24h"
        )

    def test_interval_minutes(self):
        assert parse_schedule("every 30m") == TimerSpec(
            kind="interval", on_boot_sec="30min", on_unit_active_sec="30min"
        )

    def test_interval_whitespace_and_case(self):
        assert parse_schedule("EVERY 2H") == TimerSpec(
            kind="interval", on_boot_sec="2h", on_unit_active_sec="2h"
        )
        assert parse_schedule("  every 2h  ") == TimerSpec(
            kind="interval", on_boot_sec="2h", on_unit_active_sec="2h"
        )

    @pytest.mark.parametrize("off", ["", "   ", "off", "OFF", "disabled", "none"])
    def test_off_values(self, off):
        assert parse_schedule(off) == TimerSpec(kind="off")

    @pytest.mark.parametrize("bad", ["bogus", "2h", "every 0h", "25:00", "12:60", "every"])
    def test_invalid_returns_none(self, bad):
        assert parse_schedule(bad) is None

    def test_weekly_returns_interval_168h(self):
        """'weekly' → interval TimerSpec with 168h period (604800 s)."""
        spec = parse_schedule("weekly")
        assert spec == TimerSpec(kind="interval", on_boot_sec="168h", on_unit_active_sec="168h")

    def test_weekly_case_insensitive(self):
        """'Weekly' and 'WEEKLY' are accepted."""
        assert parse_schedule("Weekly") == TimerSpec(
            kind="interval", on_boot_sec="168h", on_unit_active_sec="168h"
        )
        assert parse_schedule("WEEKLY") == TimerSpec(
            kind="interval", on_boot_sec="168h", on_unit_active_sec="168h"
        )

    def test_daily_returns_daily_at_03_00(self):
        """'daily' → daily OnCalendar at 03:00 (privacy-friendly off-peak hour)."""
        spec = parse_schedule("daily")
        assert spec == TimerSpec(kind="daily", on_calendar="*-*-* 03:00:00")

    def test_daily_case_insensitive(self):
        """'Daily' and 'DAILY' are accepted."""
        assert parse_schedule("Daily") == TimerSpec(kind="daily", on_calendar="*-*-* 03:00:00")
        assert parse_schedule("DAILY") == TimerSpec(kind="daily", on_calendar="*-*-* 03:00:00")


class TestReconcile:
    """Reconcile never raises; systemctl calls are mocked."""

    def _mock_run(self, *args, **kwargs):
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    def test_invalid_schedule_falls_back_to_off(self, tmp_path, monkeypatch):
        monkeypatch.setattr(systemd_timer, "UNIT_DIR", tmp_path)
        monkeypatch.setattr(systemd_timer, "TIMER_PATH", tmp_path / "paramem-consolidate.timer")
        monkeypatch.setattr(systemd_timer, "SERVICE_PATH", tmp_path / "paramem-consolidate.service")
        with patch.object(systemd_timer, "_run_systemctl", self._mock_run):
            msg = systemd_timer.reconcile("bogus")
        assert "disabled" in msg

    def test_interval_writes_units(self, tmp_path, monkeypatch):
        monkeypatch.setattr(systemd_timer, "UNIT_DIR", tmp_path)
        monkeypatch.setattr(systemd_timer, "TIMER_PATH", tmp_path / "paramem-consolidate.timer")
        monkeypatch.setattr(systemd_timer, "SERVICE_PATH", tmp_path / "paramem-consolidate.service")
        with patch.object(systemd_timer, "_run_systemctl", self._mock_run):
            msg = systemd_timer.reconcile("every 2h")
        assert "every 2h" in msg
        timer = (tmp_path / "paramem-consolidate.timer").read_text()
        assert "OnUnitActiveSec=2h" in timer
        svc = (tmp_path / "paramem-consolidate.service").read_text()
        assert "/scheduled-tick" in svc

    def test_daily_writes_oncalendar(self, tmp_path, monkeypatch):
        monkeypatch.setattr(systemd_timer, "UNIT_DIR", tmp_path)
        monkeypatch.setattr(systemd_timer, "TIMER_PATH", tmp_path / "paramem-consolidate.timer")
        monkeypatch.setattr(systemd_timer, "SERVICE_PATH", tmp_path / "paramem-consolidate.service")
        with patch.object(systemd_timer, "_run_systemctl", self._mock_run):
            systemd_timer.reconcile("03:30")
        timer = (tmp_path / "paramem-consolidate.timer").read_text()
        assert "OnCalendar=*-*-* 03:30:00" in timer
        assert "Persistent=true" in timer

    def test_off_removes_units(self, tmp_path, monkeypatch):
        monkeypatch.setattr(systemd_timer, "UNIT_DIR", tmp_path)
        monkeypatch.setattr(systemd_timer, "TIMER_PATH", tmp_path / "paramem-consolidate.timer")
        monkeypatch.setattr(systemd_timer, "SERVICE_PATH", tmp_path / "paramem-consolidate.service")
        (tmp_path / "paramem-consolidate.timer").write_text("stub")
        (tmp_path / "paramem-consolidate.service").write_text("stub")
        with patch.object(systemd_timer, "_run_systemctl", self._mock_run):
            msg = systemd_timer.reconcile("off")
        assert "disabled" in msg
        assert not (tmp_path / "paramem-consolidate.timer").exists()
