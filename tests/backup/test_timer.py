"""Tests for paramem.backup.timer — schedule reuse, reconcile, interval_seconds."""

from __future__ import annotations

from unittest.mock import MagicMock

from paramem.backup import timer as backup_timer
from paramem.backup.timer import (
    _backup_timer_interval_seconds,
    reconcile,
    render_service_unit,
    render_timer_unit,
)
from paramem.server import systemd_timer as server_systemd_timer
from paramem.server.systemd_timer import TimerSpec

# ---------------------------------------------------------------------------
# parse_schedule reuse (Fix 4 verification)
# ---------------------------------------------------------------------------


class TestParseScheduleReuse:
    def test_reuses_parse_schedule(self):
        """parse_schedule imported from timer is the same object as server's."""
        assert backup_timer.parse_schedule is server_systemd_timer.parse_schedule


# ---------------------------------------------------------------------------
# render_service_unit — parameterised tier (Fix 4)
# ---------------------------------------------------------------------------


class TestRenderServiceUnit:
    def test_render_service_unit_contains_tier_daily(self):
        """render_service_unit defaults to --tier daily."""
        content = render_service_unit("/usr/bin/python", "/opt/paramem")
        assert "--tier daily" in content

    def test_render_service_unit_tier_parameterised(self):
        """render_service_unit(tier='weekly') emits --tier weekly."""
        content = render_service_unit("/usr/bin/python", "/opt/paramem", tier="weekly")
        assert "--tier weekly" in content

    def test_render_service_unit_contains_python_path(self):
        content = render_service_unit("/my/python", "/my/project")
        assert "/my/python" in content
        assert "/my/project" in content


# ---------------------------------------------------------------------------
# render_timer_unit
# ---------------------------------------------------------------------------


class TestRenderTimerUnit:
    def test_render_timer_unit_calendar_daily(self):
        spec = TimerSpec(kind="daily", on_calendar="*-*-* 04:00:00")
        content = render_timer_unit(spec)
        assert "OnCalendar=*-*-* 04:00:00" in content
        assert "Persistent=true" in content

    def test_render_timer_unit_interval(self):
        spec = TimerSpec(kind="interval", on_boot_sec="1h", on_unit_active_sec="1h")
        content = render_timer_unit(spec)
        assert "OnBootSec=1h" in content
        assert "OnUnitActiveSec=1h" in content
        assert "Persistent" not in content


# ---------------------------------------------------------------------------
# reconcile — "off" removes units
# ---------------------------------------------------------------------------


class TestReconcileOff:
    def test_reconcile_off_removes_units(self, tmp_path, monkeypatch):
        """reconcile('off') with units present → stop/disable/unlink/daemon-reload."""
        # Patch the unit paths to use tmp_path.
        fake_timer = tmp_path / "paramem-backup.timer"
        fake_service = tmp_path / "paramem-backup.service"
        fake_timer.write_text("[Timer]")
        fake_service.write_text("[Service]")

        monkeypatch.setattr(backup_timer, "TIMER_PATH", fake_timer)
        monkeypatch.setattr(backup_timer, "SERVICE_PATH", fake_service)

        calls_made = []

        def fake_systemctl(*args):
            calls_made.append(args)
            return MagicMock(returncode=0, stderr="")

        monkeypatch.setattr(backup_timer, "_run_systemctl", fake_systemctl)

        result = reconcile(
            "off",
            python_path="/usr/bin/python",
            project_root="/opt/paramem",
        )
        assert "disabled" in result
        assert ("stop", "paramem-backup.timer") in calls_made
        assert ("disable", "paramem-backup.timer") in calls_made
        assert ("daemon-reload",) in calls_made

    def test_reconcile_off_no_op_when_no_units(self, tmp_path, monkeypatch):
        """reconcile('off') with no units present → no systemctl calls."""
        monkeypatch.setattr(backup_timer, "TIMER_PATH", tmp_path / "nonexistent.timer")
        monkeypatch.setattr(backup_timer, "SERVICE_PATH", tmp_path / "nonexistent.service")

        calls_made = []
        monkeypatch.setattr(
            backup_timer,
            "_run_systemctl",
            lambda *a: calls_made.append(a) or MagicMock(returncode=0),
        )

        reconcile("off", python_path="/usr/bin/python", project_root="/opt/paramem")
        assert calls_made == []


# ---------------------------------------------------------------------------
# reconcile — writes unit files
# ---------------------------------------------------------------------------


class TestReconcileWritesUnits:
    def _patch_reconcile(self, monkeypatch, tmp_path):
        timer_path = tmp_path / "paramem-backup.timer"
        service_path = tmp_path / "paramem-backup.service"
        monkeypatch.setattr(backup_timer, "TIMER_PATH", timer_path)
        monkeypatch.setattr(backup_timer, "SERVICE_PATH", service_path)
        monkeypatch.setattr(backup_timer, "UNIT_DIR", tmp_path)
        monkeypatch.setattr(
            backup_timer, "_run_systemctl", lambda *a: MagicMock(returncode=0, stderr="")
        )
        return timer_path, service_path

    def test_reconcile_writes_unit_files(self, tmp_path, monkeypatch):
        """reconcile('daily 04:00') → SERVICE_PATH + TIMER_PATH written."""
        timer_path, service_path = self._patch_reconcile(monkeypatch, tmp_path)
        reconcile("daily 04:00", python_path="/usr/bin/python", project_root="/opt/paramem")
        assert service_path.exists()
        assert timer_path.exists()

    def test_reconcile_calendar_for_daily(self, tmp_path, monkeypatch):
        """reconcile('daily 04:00') → timer contains OnCalendar=*-*-* 04:00:00."""
        timer_path, _ = self._patch_reconcile(monkeypatch, tmp_path)
        reconcile("daily 04:00", python_path="/usr/bin/python", project_root="/opt/paramem")
        content = timer_path.read_text()
        assert "OnCalendar=*-*-* 04:00:00" in content
        assert "Persistent=true" in content

    def test_reconcile_no_op_when_already_current(self, tmp_path, monkeypatch):
        """Two consecutive reconciles → second does NOT call daemon-reload."""
        timer_path, service_path = self._patch_reconcile(monkeypatch, tmp_path)
        daemon_reloads = []

        def fake_systemctl(*args):
            if args == ("daemon-reload",):
                daemon_reloads.append(1)
            return MagicMock(returncode=0, stderr="")

        monkeypatch.setattr(backup_timer, "_run_systemctl", fake_systemctl)
        reconcile("daily 04:00", python_path="/usr/bin/python", project_root="/opt/paramem")
        count_after_first = len(daemon_reloads)
        reconcile("daily 04:00", python_path="/usr/bin/python", project_root="/opt/paramem")
        # Second call must not trigger daemon-reload (unit unchanged).
        assert len(daemon_reloads) == count_after_first


# ---------------------------------------------------------------------------
# _backup_timer_interval_seconds
# ---------------------------------------------------------------------------


class TestIntervalSeconds:
    def test_interval_seconds_for_daily_time(self):
        """'daily 04:00' → 86400."""
        assert _backup_timer_interval_seconds("daily 04:00") == 86400

    def test_interval_seconds_for_hhmm(self):
        """'04:00' → 86400."""
        assert _backup_timer_interval_seconds("04:00") == 86400

    def test_interval_seconds_for_daily_bare(self):
        """'daily' → 86400."""
        assert _backup_timer_interval_seconds("daily") == 86400

    def test_interval_seconds_for_every_12h(self):
        """'every 12h' → 43200."""
        assert _backup_timer_interval_seconds("every 12h") == 43200

    def test_interval_seconds_for_every_6h(self):
        assert _backup_timer_interval_seconds("every 6h") == 6 * 3600

    def test_interval_seconds_for_weekly(self):
        assert _backup_timer_interval_seconds("weekly") == 604800

    def test_interval_seconds_for_off(self):
        """'off' → 0."""
        assert _backup_timer_interval_seconds("off") == 0

    def test_interval_seconds_for_empty(self):
        assert _backup_timer_interval_seconds("") == 0

    def test_interval_seconds_for_every_30m(self):
        assert _backup_timer_interval_seconds("every 30m") == 1800
