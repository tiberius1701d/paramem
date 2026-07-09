"""Unit tests for the consolidation timer parser + reconciler.

Schedule parsing lives in `paramem.server.systemd_timer.parse_schedule` since
the server delegates actual scheduling to a systemd user timer.
"""

import subprocess
from unittest.mock import patch

import pytest

from paramem.server import systemd_timer
from paramem.server.systemd_timer import (
    TimerSpec,
    _hours_to_calendar,
    _minutes_to_calendar,
    parse_schedule,
)


class TestParseSchedule:
    def test_daily_hhmm(self):
        assert parse_schedule("02:00") == TimerSpec(kind="daily", on_calendar="*-*-* 02:00:00")
        assert parse_schedule("23:59") == TimerSpec(kind="daily", on_calendar="*-*-* 23:59:00")
        assert parse_schedule("0:00") == TimerSpec(kind="daily", on_calendar="*-*-* 00:00:00")

    def test_interval_hours_calendar(self):
        assert parse_schedule("every 2h") == TimerSpec(
            kind="calendar", on_calendar="*-*-* 00,02,04,06,08,10,12,14,16,18,20,22:00:00"
        )
        assert parse_schedule("every 24h") == TimerSpec(
            kind="calendar", on_calendar="*-*-* 00:00:00"
        )

    def test_interval_hours_odd_becomes_heartbeat_calendar(self):
        """Non-divisor hour cadences no longer stay monotonic — they render at
        the gcd(count, 24) heartbeat grid (suspend/power-off catch-up gate;
        see systemd_timer module docstring). gcd(5, 24) == gcd(7, 24) == 1,
        so both land on the hourly grid.
        """
        assert parse_schedule("every 5h") == TimerSpec(
            kind="calendar", on_calendar=_hours_to_calendar(1)
        )
        assert parse_schedule("every 7h") == TimerSpec(
            kind="calendar", on_calendar=_hours_to_calendar(1)
        )

    def test_interval_minutes_calendar(self):
        assert parse_schedule("every 30m") == TimerSpec(kind="calendar", on_calendar="*:00,30:00")

    def test_interval_minutes_odd_becomes_heartbeat_calendar(self):
        """gcd(13, 60) == 1 → every-13m renders at the per-minute heartbeat grid."""
        assert parse_schedule("every 13m") == TimerSpec(
            kind="calendar", on_calendar=_minutes_to_calendar(1)
        )

    def test_interval_whitespace_and_case(self):
        assert parse_schedule("EVERY 2H") == TimerSpec(
            kind="calendar", on_calendar="*-*-* 00,02,04,06,08,10,12,14,16,18,20,22:00:00"
        )
        assert parse_schedule("  every 2h  ") == TimerSpec(
            kind="calendar", on_calendar="*-*-* 00,02,04,06,08,10,12,14,16,18,20,22:00:00"
        )

    @pytest.mark.parametrize("off", ["", "   ", "off", "OFF", "disabled", "none"])
    def test_off_values(self, off):
        assert parse_schedule(off) == TimerSpec(kind="off")

    @pytest.mark.parametrize("bad", ["bogus", "every 0h", "25:00", "12:60", "every"])
    def test_invalid_returns_none(self, bad):
        assert parse_schedule(bad) is None

    def test_bare_shorthand_calendar(self):
        """Bare ``Nh``/``Nm`` shorthand converts to calendar when cadence is divisible."""
        assert parse_schedule("2h") == TimerSpec(
            kind="calendar", on_calendar="*-*-* 00,02,04,06,08,10,12,14,16,18,20,22:00:00"
        )
        assert parse_schedule("30m") == TimerSpec(kind="calendar", on_calendar="*:00,30:00")

    def test_bare_shorthand_odd_becomes_heartbeat_calendar(self):
        """Bare ``Nh``/``Nm`` shorthand with a non-divisor cadence renders at
        the gcd(count, modulus) heartbeat grid instead of staying monotonic.
        gcd(48, 24) == 24 → daily grid; gcd(90, 60) == 30 → half-hour grid.
        """
        assert parse_schedule("48H") == TimerSpec(
            kind="calendar", on_calendar=_hours_to_calendar(24)
        )
        assert parse_schedule("90M") == TimerSpec(
            kind="calendar", on_calendar=_minutes_to_calendar(30)
        )

    def test_weekly_returns_calendar(self):
        """'weekly' → calendar TimerSpec firing Monday 00:00 with Persistent=true."""
        spec = parse_schedule("weekly")
        assert spec == TimerSpec(kind="calendar", on_calendar="Mon *-*-* 00:00:00")

    def test_weekly_case_insensitive(self):
        """'Weekly' and 'WEEKLY' are accepted."""
        assert parse_schedule("Weekly") == TimerSpec(
            kind="calendar", on_calendar="Mon *-*-* 00:00:00"
        )
        assert parse_schedule("WEEKLY") == TimerSpec(
            kind="calendar", on_calendar="Mon *-*-* 00:00:00"
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

    def test_calendar_interval_writes_units(self, tmp_path, monkeypatch):
        monkeypatch.setattr(systemd_timer, "UNIT_DIR", tmp_path)
        monkeypatch.setattr(systemd_timer, "TIMER_PATH", tmp_path / "paramem-consolidate.timer")
        monkeypatch.setattr(systemd_timer, "SERVICE_PATH", tmp_path / "paramem-consolidate.service")
        with patch.object(systemd_timer, "_run_systemctl", self._mock_run):
            msg = systemd_timer.reconcile("every 2h")
        assert "with catch-up" in msg
        timer = (tmp_path / "paramem-consolidate.timer").read_text()
        assert "OnCalendar=" in timer
        assert "Persistent=true" in timer
        assert "OnUnitActiveSec" not in timer
        svc = (tmp_path / "paramem-consolidate.service").read_text()
        assert "/scheduled-tick" in svc

    def test_heartbeat_interval_writes_units(self, tmp_path, monkeypatch):
        """A non-divisor cadence ('every 5h') still renders OnCalendar +
        Persistent=true — it is a heartbeat grid, not a monotonic timer.
        """
        monkeypatch.setattr(systemd_timer, "UNIT_DIR", tmp_path)
        monkeypatch.setattr(systemd_timer, "TIMER_PATH", tmp_path / "paramem-consolidate.timer")
        monkeypatch.setattr(systemd_timer, "SERVICE_PATH", tmp_path / "paramem-consolidate.service")
        with patch.object(systemd_timer, "_run_systemctl", self._mock_run):
            msg = systemd_timer.reconcile("every 5h")
        assert "with catch-up" in msg
        timer = (tmp_path / "paramem-consolidate.timer").read_text()
        assert "OnCalendar=" in timer
        assert "Persistent=true" in timer
        assert "OnBootSec" not in timer
        assert "OnUnitActiveSec" not in timer

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
