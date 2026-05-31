"""Tests for the interval→calendar conversion in systemd_timer.

Covers every common divisible cadence, a representative set of odd (non-divisible)
cadences that must stay monotonic, weekly, and rendered unit text assertions.
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

# Every-1h produces 24 hour marks; too long to inline cleanly.
_1H_CAL = "*-*-* 00,01,02,03,04,05,06,07,08,09,10,11,12,13,14,15,16,17,18,19,20,21,22,23:00:00"
# Every-1m produces 60 minute marks.
_1M_CAL = (
    "*:00,01,02,03,04,05,06,07,08,09,10,11,"
    "12,13,14,15,16,17,18,19,20,21,22,23,"
    "24,25,26,27,28,29,30,31,32,33,34,35,"
    "36,37,38,39,40,41,42,43,44,45,46,47,"
    "48,49,50,51,52,53,54,55,56,57,58,59:00"
)
_2M_CAL = (
    "*:00,02,04,06,08,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58:00"
)


class TestHoursToCalendar:
    @pytest.mark.parametrize(
        "n,expected",
        [
            (1, _1H_CAL),
            (2, "*-*-* 00,02,04,06,08,10,12,14,16,18,20,22:00:00"),
            (3, "*-*-* 00,03,06,09,12,15,18,21:00:00"),
            (4, "*-*-* 00,04,08,12,16,20:00:00"),
            (6, "*-*-* 00,06,12,18:00:00"),
            (8, "*-*-* 00,08,16:00:00"),
            (12, "*-*-* 00,12:00:00"),
            (24, "*-*-* 00:00:00"),
        ],
    )
    def test_divisible_cadences(self, n, expected):
        assert _hours_to_calendar(n) == expected

    @pytest.mark.parametrize("n", [5, 7, 9, 10, 11, 13, 48])
    def test_non_divisible_returns_none(self, n):
        assert _hours_to_calendar(n) is None

    def test_zero_returns_none(self):
        assert _hours_to_calendar(0) is None

    def test_negative_returns_none(self):
        assert _hours_to_calendar(-1) is None


class TestMinutesToCalendar:
    @pytest.mark.parametrize(
        "n,expected",
        [
            (1, _1M_CAL),
            (2, _2M_CAL),
            (3, "*:00,03,06,09,12,15,18,21,24,27,30,33,36,39,42,45,48,51,54,57:00"),
            (4, "*:00,04,08,12,16,20,24,28,32,36,40,44,48,52,56:00"),
            (5, "*:00,05,10,15,20,25,30,35,40,45,50,55:00"),
            (6, "*:00,06,12,18,24,30,36,42,48,54:00"),
            (10, "*:00,10,20,30,40,50:00"),
            (12, "*:00,12,24,36,48:00"),
            (15, "*:00,15,30,45:00"),
            (20, "*:00,20,40:00"),
            (30, "*:00,30:00"),
        ],
    )
    def test_divisible_cadences(self, n, expected):
        assert _minutes_to_calendar(n) == expected

    @pytest.mark.parametrize("n", [7, 9, 11, 13, 17, 19, 23, 90])
    def test_non_divisible_returns_none(self, n):
        assert _minutes_to_calendar(n) is None

    def test_zero_returns_none(self):
        assert _minutes_to_calendar(0) is None


class TestParseScheduleCalendarConversions:
    @pytest.mark.parametrize(
        "schedule,expected_calendar",
        [
            ("every 1h", _1H_CAL),
            ("every 2h", "*-*-* 00,02,04,06,08,10,12,14,16,18,20,22:00:00"),
            ("every 3h", "*-*-* 00,03,06,09,12,15,18,21:00:00"),
            ("every 4h", "*-*-* 00,04,08,12,16,20:00:00"),
            ("every 6h", "*-*-* 00,06,12,18:00:00"),
            ("every 8h", "*-*-* 00,08,16:00:00"),
            ("every 12h", "*-*-* 00,12:00:00"),
            ("every 24h", "*-*-* 00:00:00"),
        ],
    )
    def test_hour_cadences_become_calendar(self, schedule, expected_calendar):
        spec = parse_schedule(schedule)
        assert spec is not None
        assert spec.kind == "calendar"
        assert spec.on_calendar == expected_calendar
        assert spec.on_boot_sec is None
        assert spec.on_unit_active_sec is None

    @pytest.mark.parametrize(
        "schedule,expected_calendar",
        [
            ("every 5m", "*:00,05,10,15,20,25,30,35,40,45,50,55:00"),
            ("every 15m", "*:00,15,30,45:00"),
            ("every 30m", "*:00,30:00"),
        ],
    )
    def test_minute_cadences_become_calendar(self, schedule, expected_calendar):
        spec = parse_schedule(schedule)
        assert spec is not None
        assert spec.kind == "calendar"
        assert spec.on_calendar == expected_calendar

    @pytest.mark.parametrize("schedule", ["every 5h", "every 7h", "every 11h"])
    def test_odd_hour_cadences_stay_monotonic(self, schedule):
        spec = parse_schedule(schedule)
        assert spec is not None
        assert spec.kind == "interval"
        assert spec.on_boot_sec is not None
        assert spec.on_unit_active_sec is not None
        assert spec.on_calendar is None

    @pytest.mark.parametrize("schedule", ["every 7m", "every 13m", "every 17m"])
    def test_odd_minute_cadences_stay_monotonic(self, schedule):
        spec = parse_schedule(schedule)
        assert spec is not None
        assert spec.kind == "interval"
        assert spec.on_calendar is None

    def test_weekly_becomes_calendar(self):
        spec = parse_schedule("weekly")
        assert spec == TimerSpec(kind="calendar", on_calendar="Mon *-*-* 00:00:00")

    def test_weekly_case_variants(self):
        for s in ("weekly", "Weekly", "WEEKLY"):
            spec = parse_schedule(s)
            assert spec is not None
            assert spec.kind == "calendar"
            assert spec.on_calendar == "Mon *-*-* 00:00:00"


class TestRenderTimerUnitCalendar:
    def test_calendar_kind_has_oncalendar_and_persistent(self):
        spec = TimerSpec(kind="calendar", on_calendar="*-*-* 00,12:00:00")
        text = systemd_timer.render_timer_unit(spec)
        assert "OnCalendar=*-*-* 00,12:00:00" in text
        assert "Persistent=true" in text
        assert "OnBootSec" not in text
        assert "OnUnitActiveSec" not in text

    def test_daily_kind_has_oncalendar_and_persistent(self):
        spec = TimerSpec(kind="daily", on_calendar="*-*-* 03:00:00")
        text = systemd_timer.render_timer_unit(spec)
        assert "OnCalendar=*-*-* 03:00:00" in text
        assert "Persistent=true" in text

    def test_interval_kind_has_no_persistent(self):
        spec = TimerSpec(kind="interval", on_boot_sec="5h", on_unit_active_sec="5h")
        text = systemd_timer.render_timer_unit(spec)
        assert "OnBootSec=5h" in text
        assert "OnUnitActiveSec=5h" in text
        assert "Persistent=true" not in text
        assert "OnCalendar" not in text

    @pytest.mark.parametrize(
        "schedule,expected_calendar",
        [
            ("every 12h", "*-*-* 00,12:00:00"),
            ("every 6h", "*-*-* 00,06,12,18:00:00"),
            ("every 24h", "*-*-* 00:00:00"),
            ("every 15m", "*:00,15,30,45:00"),
            ("weekly", "Mon *-*-* 00:00:00"),
        ],
    )
    def test_rendered_unit_for_common_schedules(self, schedule, expected_calendar):
        spec = parse_schedule(schedule)
        assert spec is not None
        text = systemd_timer.render_timer_unit(spec)
        assert f"OnCalendar={expected_calendar}" in text
        assert "Persistent=true" in text

    def test_every_5h_rendered_unit_no_persistent(self):
        spec = parse_schedule("every 5h")
        assert spec is not None
        text = systemd_timer.render_timer_unit(spec)
        assert "OnBootSec=5h" in text
        assert "OnUnitActiveSec=5h" in text
        assert "Persistent=true" not in text


class TestRenderServiceUnitAuth:
    """Verify the generated service unit sends the Authorization header securely."""

    _PROJECT_ROOT = "/srv/paramem"

    def _render(self, endpoint: str = systemd_timer.DEFAULT_ENDPOINT) -> str:
        return systemd_timer.render_service_unit(endpoint, project_root=self._PROJECT_ROOT)

    def test_environment_file_uses_project_root(self):
        """EnvironmentFile must be derived from project_root, not hardcoded."""
        text = self._render()
        assert f"EnvironmentFile=-{self._PROJECT_ROOT}/.env" in text, (
            "EnvironmentFile must use the passed project_root, not a literal username path"
        )

    def test_no_home_literal_in_unit(self):
        """No literal /home/<user> path must appear in the rendered unit."""
        import re

        text = self._render()
        assert not re.search(r"/home/[^/]+/", text), (
            "Unit must not contain a hardcoded /home/<username>/ path"
        )

    def test_token_not_in_argv(self):
        """The PARAMEM_API_TOKEN value must not appear as a literal in ExecStart argv.

        The rendered unit must reference $PARAMEM_API_TOKEN symbolically so the
        token is expanded from the EnvironmentFile at runtime, not logged in the
        journal as part of the ExecStart command.
        """
        text = self._render()
        # The symbolic reference must be present.
        assert "$PARAMEM_API_TOKEN" in text, (
            "ExecStart must reference $PARAMEM_API_TOKEN symbolically"
        )
        # The header must be piped via stdin (-H @-) so it is not in argv.
        assert "-H @-" in text, (
            "curl must read the Authorization header from stdin (-H @-) "
            "so the token is not in process argv or the journalled ExecStart"
        )

    def test_no_literal_token_in_execstart(self):
        """ExecStart must not contain a 20+-char alphanumeric token literal after 'Bearer'."""
        import re

        text = self._render()
        # If "Bearer" appears directly in ExecStart (not via stdin pipe), it must
        # only be in the printf argument, not as a bare argv word.
        for m in re.finditer(r"Bearer\s+([A-Za-z0-9_\-]{20,})", text):
            matched = m.group(1)
            assert "$" in matched or "{" in matched, (
                f"Looks like a literal token after Bearer: {matched!r}"
            )

    def test_fail_with_body_present_and_no_bare_f_flag(self):
        """curl must use --fail-with-body (not -f or -sSf) so error bodies are logged."""
        text = self._render()
        assert "--fail-with-body" in text, (
            "curl must use --fail-with-body so HTTP error bodies surface in the journal"
        )
        # -sSf and bare -f conflict with --fail-with-body; neither must appear.
        assert "-sSf" not in text, (
            "Conflicting -f inside -sSf must not appear with --fail-with-body"
        )
        import re

        # A bare -f flag (space-separated, not part of --fail-with-body) must not appear.
        assert not re.search(r"\s-f\b", text), (
            "Bare -f flag conflicts with --fail-with-body and must be absent"
        )

    def test_execstart_is_single_logical_line(self):
        """ExecStart must be a single physical line (no backslash line-continuations)."""
        text = self._render()
        execstart_lines = [ln for ln in text.splitlines() if ln.startswith("ExecStart=")]
        assert len(execstart_lines) == 1, (
            f"Expected exactly one ExecStart= line, got {len(execstart_lines)}: {execstart_lines}"
        )
        assert not execstart_lines[0].endswith("\\"), (
            "ExecStart must not end with a backslash continuation"
        )

    def test_environment_file_in_unit(self):
        """EnvironmentFile must be present so PARAMEM_API_TOKEN is available at runtime."""
        text = self._render()
        assert "EnvironmentFile=" in text, (
            "Service unit must include EnvironmentFile to source PARAMEM_API_TOKEN"
        )


class TestReconcileProjectRoot:
    """Verify project_root is threaded through reconcile() into the service unit."""

    def _mock_run(self, *args, **kwargs):
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    def test_reconcile_passes_project_root_to_service_unit(self, tmp_path, monkeypatch):
        """The rendered service file must contain the project_root passed to reconcile()."""
        monkeypatch.setattr(systemd_timer, "UNIT_DIR", tmp_path)
        monkeypatch.setattr(systemd_timer, "TIMER_PATH", tmp_path / "paramem-consolidate.timer")
        svc_path = tmp_path / "paramem-consolidate.service"
        monkeypatch.setattr(systemd_timer, "SERVICE_PATH", svc_path)
        custom_root = "/opt/myparamem"
        with patch.object(systemd_timer, "_run_systemctl", self._mock_run):
            systemd_timer.reconcile("every 12h", project_root=custom_root)
        svc_text = svc_path.read_text()
        assert f"EnvironmentFile=-{custom_root}/.env" in svc_text, (
            "Service unit EnvironmentFile must reflect the project_root passed to reconcile()"
        )
        import re

        assert not re.search(r"/home/[^/]+/", svc_text), (
            "Rendered service unit must not contain a hardcoded /home/<user>/ path"
        )


class TestRenderServiceUnitEndpointValidation:
    """Verify that render_service_unit() rejects non-localhost endpoints."""

    def test_localhost_endpoint_accepted(self):
        """http://localhost endpoint passes validation."""
        text = systemd_timer.render_service_unit(
            "http://localhost:8420/scheduled-tick", project_root="/srv/paramem"
        )
        assert "localhost:8420" in text

    def test_127_0_0_1_endpoint_accepted(self):
        """http://127.0.0.1 endpoint passes validation."""
        text = systemd_timer.render_service_unit(
            systemd_timer.DEFAULT_ENDPOINT, project_root="/srv/paramem"
        )
        assert "127.0.0.1" in text

    def test_non_localhost_endpoint_raises_value_error(self):
        """An external endpoint raises ValueError to prevent shell-injection."""
        import pytest

        with pytest.raises(ValueError, match="http://127.0.0.1.*http://localhost"):
            systemd_timer.render_service_unit(
                "https://evil.example.com/inject", project_root="/srv/paramem"
            )

    def test_https_localhost_not_accepted(self):
        """https://127.0.0.1 (not http://) is rejected — the guard is strict."""
        import pytest

        with pytest.raises(ValueError):
            systemd_timer.render_service_unit(
                "https://127.0.0.1:8420/tick", project_root="/srv/paramem"
            )


class TestReconcileDetailString:
    def _mock_run(self, *args, **kwargs):
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    def test_calendar_detail_says_with_catchup(self, tmp_path, monkeypatch):
        monkeypatch.setattr(systemd_timer, "UNIT_DIR", tmp_path)
        monkeypatch.setattr(systemd_timer, "TIMER_PATH", tmp_path / "paramem-consolidate.timer")
        monkeypatch.setattr(systemd_timer, "SERVICE_PATH", tmp_path / "paramem-consolidate.service")
        with patch.object(systemd_timer, "_run_systemctl", self._mock_run):
            msg = systemd_timer.reconcile("every 12h")
        assert "with catch-up" in msg
        assert "no catch-up" not in msg

    def test_interval_detail_says_no_catchup(self, tmp_path, monkeypatch):
        monkeypatch.setattr(systemd_timer, "UNIT_DIR", tmp_path)
        monkeypatch.setattr(systemd_timer, "TIMER_PATH", tmp_path / "paramem-consolidate.timer")
        monkeypatch.setattr(systemd_timer, "SERVICE_PATH", tmp_path / "paramem-consolidate.service")
        with patch.object(systemd_timer, "_run_systemctl", self._mock_run):
            msg = systemd_timer.reconcile("every 5h")
        assert "no catch-up" in msg
        assert "with catch-up" not in msg

    def test_daily_detail_says_with_catchup(self, tmp_path, monkeypatch):
        monkeypatch.setattr(systemd_timer, "UNIT_DIR", tmp_path)
        monkeypatch.setattr(systemd_timer, "TIMER_PATH", tmp_path / "paramem-consolidate.timer")
        monkeypatch.setattr(systemd_timer, "SERVICE_PATH", tmp_path / "paramem-consolidate.service")
        with patch.object(systemd_timer, "_run_systemctl", self._mock_run):
            msg = systemd_timer.reconcile("03:00")
        assert "with catch-up" in msg
