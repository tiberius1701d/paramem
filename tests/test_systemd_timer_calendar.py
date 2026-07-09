"""Tests for the interval→calendar conversion in systemd_timer.

Covers every common divisible cadence, a representative set of odd
(non-divisible) cadences that now render at a coarser heartbeat grid instead
of staying monotonic (suspend/power-off catch-up gate — see systemd_timer
module docstring), weekly, and rendered unit text assertions.
"""

import subprocess
from datetime import datetime
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
        # TimerSpec no longer carries on_boot_sec/on_unit_active_sec fields at
        # all (monotonic kind deleted — see systemd_timer module docstring).
        assert not hasattr(spec, "on_boot_sec")
        assert not hasattr(spec, "on_unit_active_sec")

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
    def test_odd_hour_cadences_become_heartbeat_calendar(self, schedule):
        """Non-divisor hour cadences render at the gcd(count, 24) heartbeat
        grid instead of staying monotonic (suspend/power-off catch-up gate;
        every non-off TimerSpec is now OnCalendar-based).
        """
        spec = parse_schedule(schedule)
        assert spec is not None
        assert spec.kind == "calendar"
        assert spec.on_calendar is not None
        assert not hasattr(spec, "on_boot_sec")
        assert not hasattr(spec, "on_unit_active_sec")

    @pytest.mark.parametrize("schedule", ["every 7m", "every 13m", "every 17m"])
    def test_odd_minute_cadences_become_heartbeat_calendar(self, schedule):
        """Non-divisor minute cadences render at the gcd(count, 60) heartbeat grid."""
        spec = parse_schedule(schedule)
        assert spec is not None
        assert spec.kind == "calendar"
        assert spec.on_calendar is not None

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

    def test_heartbeat_calendar_kind_has_persistent_and_no_monotonic_fields(self):
        """A heartbeat-grid TimerSpec ('every 5h' → gcd(5,24)=1 hourly grid)
        still renders OnCalendar + Persistent=true — there is no monotonic
        "interval" TimerSpec kind left to special-case.
        """
        spec = parse_schedule("every 5h")
        assert spec is not None
        assert spec.kind == "calendar"
        text = systemd_timer.render_timer_unit(spec)
        assert "OnCalendar=" in text
        assert "Persistent=true" in text
        assert "OnBootSec" not in text
        assert "OnUnitActiveSec" not in text

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

    def test_every_5h_rendered_unit_has_persistent(self):
        """'every 5h' renders as a heartbeat OnCalendar with Persistent=true
        (catch-up gate rework) — it no longer renders as a bare monotonic
        OnBootSec/OnUnitActiveSec timer with no catch-up.
        """
        spec = parse_schedule("every 5h")
        assert spec is not None
        text = systemd_timer.render_timer_unit(spec)
        assert "OnBootSec" not in text
        assert "OnUnitActiveSec" not in text
        assert "OnCalendar=" in text
        assert "Persistent=true" in text


# ---------------------------------------------------------------------------
# heartbeat_seconds
# ---------------------------------------------------------------------------


class TestHeartbeatSeconds:
    @pytest.mark.parametrize("schedule", ["", "off", "disabled", "none"])
    def test_off_returns_none(self, schedule):
        assert systemd_timer.heartbeat_seconds(schedule) is None

    @pytest.mark.parametrize("schedule", ["daily", "weekly", "04:00", "daily 04:00"])
    def test_anchored_returns_none(self, schedule):
        assert systemd_timer.heartbeat_seconds(schedule) is None

    @pytest.mark.parametrize("schedule", ["every 12h", "every 30m"])
    def test_calendar_exact_period_returns_none(self, schedule):
        assert systemd_timer.heartbeat_seconds(schedule) is None

    def test_every_5h_returns_3600(self):
        """gcd(5, 24) == 1 → hourly grid."""
        assert systemd_timer.heartbeat_seconds("every 5h") == 3600

    def test_every_90m_returns_1800(self):
        """gcd(90, 60) == 30 → half-hour grid (48 ticks/day, not 1440)."""
        assert systemd_timer.heartbeat_seconds("every 90m") == 1800

    def test_every_48h_returns_86400(self):
        """gcd(48, 24) == 24 → daily grid (1 tick/day, not 24)."""
        assert systemd_timer.heartbeat_seconds("every 48h") == 86400

    def test_every_7m_returns_60(self):
        """gcd(7, 60) == 1 → per-minute grid."""
        assert systemd_timer.heartbeat_seconds("every 7m") == 60


# ---------------------------------------------------------------------------
# floor_to_heartbeat — drift regression
# ---------------------------------------------------------------------------


class TestFloorToHeartbeatDriftRegression:
    """The highest-value regression in the catch-up-gate change.

    Without flooring, stamping raw ``time.time()`` after a non-zero dispatch
    delay pushes every subsequent due-check one heartbeat late, silently
    inflating the effective period forever (``every 5h`` converges to a real
    6h). Flooring makes the stamp grid-aligned so ``last_run + period`` is
    exactly a grid mark, however large the per-cycle delay.
    """

    # Anchored at local midnight on a fixed, DST-transition-free date (mid-
    # January, CET/CEST transitions land in March/October) so the test is
    # deterministic regardless of the host's local timezone.
    _BASE_EPOCH = datetime(2024, 1, 15, 0, 0, 0).timestamp()

    def test_consecutive_dispatches_stay_exactly_period_apart(self):
        """Simulate 3 consecutive 'every 5h' gated cycles with a non-zero,
        varying dispatch delay between heartbeat-fire and stamp-write.
        Flooring must keep every dispatch exactly ``period`` (18000s) apart.
        """
        heartbeat_s = systemd_timer.heartbeat_seconds("every 5h")
        assert heartbeat_s == 3600
        period_s = 5 * 3600

        # Heartbeat fires exactly on the grid; each cycle's dispatch happens
        # some seconds AFTER the heartbeat fired (guards, debounce,
        # orphan-session claim, migration branch — see app.py). The delay
        # varies per cycle to prove flooring, not a fixed offset, is doing
        # the work.
        heartbeat_fire_times = [
            self._BASE_EPOCH,
            self._BASE_EPOCH + period_s,
            self._BASE_EPOCH + 2 * period_s,
        ]
        dispatch_delays = [7.0, 143.0, 1.0]  # non-zero, varying

        stamps: list[float] = []
        for fire_t, delay in zip(heartbeat_fire_times, dispatch_delays):
            dispatch_t = fire_t + delay
            stamps.append(systemd_timer.floor_to_heartbeat(dispatch_t, heartbeat_s))

        gaps = [stamps[i + 1] - stamps[i] for i in range(len(stamps) - 1)]
        assert gaps == [period_s, period_s], (
            f"Consecutive dispatch stamps must be exactly {period_s}s apart; got gaps={gaps}"
        )

    def test_unfloored_stamp_would_drift(self):
        """Sanity check that the regression is real: stamping raw dispatch
        time (no flooring) does NOT stay period-apart under the same delays.
        """
        period_s = 5 * 3600
        heartbeat_fire_times = [
            self._BASE_EPOCH,
            self._BASE_EPOCH + period_s,
            self._BASE_EPOCH + 2 * period_s,
        ]
        dispatch_delays = [7.0, 143.0, 1.0]

        raw_stamps = [
            fire_t + delay for fire_t, delay in zip(heartbeat_fire_times, dispatch_delays)
        ]
        raw_gaps = [raw_stamps[i + 1] - raw_stamps[i] for i in range(len(raw_stamps) - 1)]
        assert raw_gaps != [period_s, period_s], (
            "Expected the un-floored gaps to drift with delay — if this fails, "
            "the regression this test guards no longer reproduces"
        )

    def test_floor_is_idempotent_on_an_already_floored_stamp(self):
        heartbeat_s = 3600
        once = systemd_timer.floor_to_heartbeat(12345.0, heartbeat_s)
        twice = systemd_timer.floor_to_heartbeat(once, heartbeat_s)
        assert once == twice

    def test_non_positive_grid_raises(self):
        with pytest.raises(ValueError):
            systemd_timer.floor_to_heartbeat(100.0, 0)


# ---------------------------------------------------------------------------
# No rendered timer unit ever lacks Persistent=true — both render paths
# (consolidation timer + backup timer share render_timer_unit / TimerTarget).
# ---------------------------------------------------------------------------


class TestNoRenderedUnitLacksPersistent:
    _REPRESENTATIVE_SCHEDULES = [
        "daily",
        "weekly",
        "04:00",
        "daily 04:00",
        "every 12h",
        "every 30m",
        "every 5h",
        "every 90m",
        "every 48h",
        "every 7m",
    ]

    @pytest.mark.parametrize("schedule", _REPRESENTATIVE_SCHEDULES)
    def test_consolidation_render_path(self, schedule):
        spec = parse_schedule(schedule)
        assert spec is not None
        text = systemd_timer.render_timer_unit(spec)
        assert "OnCalendar=" in text
        assert "Persistent=true" in text
        assert "OnBootSec" not in text
        assert "OnUnitActiveSec" not in text
        assert not hasattr(spec, "on_boot_sec")

    @pytest.mark.parametrize("schedule", _REPRESENTATIVE_SCHEDULES)
    def test_backup_render_path(self, schedule):
        from paramem.backup import timer as backup_timer

        spec = backup_timer.parse_schedule(schedule)
        assert spec is not None
        text = systemd_timer.render_timer_unit(
            spec, unit_name=backup_timer.TIMER_NAME, description="ParaMem scheduled backup"
        )
        assert "OnCalendar=" in text
        assert "Persistent=true" in text
        assert "OnBootSec" not in text
        assert "OnUnitActiveSec" not in text
        assert not hasattr(spec, "on_boot_sec")


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

    def test_printf_format_survives_systemd_specifier_expansion(self):
        """The printf format must reach the shell as a literal ``%s``.

        systemd expands ``%`` specifiers in ExecStart before invoking the shell:
        ``%s`` is the user's login shell (e.g. ``/bin/bash``) and ``%%`` is a
        literal ``%``.  A bare ``printf "%s"`` therefore becomes
        ``printf "/bin/bash"`` at runtime, which emits ``/bin/bash`` as the
        Authorization header value and makes every tick fail with HTTP 401.  The
        render must escape the format as ``%%s`` so the shell receives ``%s``.
        """
        text = self._render()
        execstart = next(ln for ln in text.splitlines() if ln.startswith("ExecStart="))

        # Emulate systemd's ExecStart specifier expansion for the two specifiers
        # that matter here: ``%%`` -> ``%`` and ``%s`` -> the user's shell.
        sentinel = "\x00"
        expanded = execstart.replace("%%", sentinel).replace("%s", "/bin/bash")
        expanded = expanded.replace(sentinel, "%")

        assert 'printf "%s"' in expanded, (
            "After systemd specifier expansion the shell must receive "
            f'printf "%s"; got: {expanded!r}'
        )
        assert 'printf "/bin/bash"' not in expanded, (
            "Bare %s leaked as the user's shell into printf — escape it as %%s"
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

    def test_heartbeat_detail_says_with_catchup(self, tmp_path, monkeypatch):
        """'every 5h' (a heartbeat-grid, non-exact cadence) still reports
        'with catch-up' — the 'no catch-up' detail branch no longer exists.
        """
        monkeypatch.setattr(systemd_timer, "UNIT_DIR", tmp_path)
        monkeypatch.setattr(systemd_timer, "TIMER_PATH", tmp_path / "paramem-consolidate.timer")
        monkeypatch.setattr(systemd_timer, "SERVICE_PATH", tmp_path / "paramem-consolidate.service")
        with patch.object(systemd_timer, "_run_systemctl", self._mock_run):
            msg = systemd_timer.reconcile("every 5h")
        assert "with catch-up" in msg
        assert "no catch-up" not in msg

    def test_daily_detail_says_with_catchup(self, tmp_path, monkeypatch):
        monkeypatch.setattr(systemd_timer, "UNIT_DIR", tmp_path)
        monkeypatch.setattr(systemd_timer, "TIMER_PATH", tmp_path / "paramem-consolidate.timer")
        monkeypatch.setattr(systemd_timer, "SERVICE_PATH", tmp_path / "paramem-consolidate.service")
        with patch.object(systemd_timer, "_run_systemctl", self._mock_run):
            msg = systemd_timer.reconcile("03:00")
        assert "with catch-up" in msg


class TestCurrentTimerStateJsonParser:
    """Verify current_timer_state uses list-timers --output=json for next_elapse_us.

    systemd 255 renders NextElapseUSecRealtime as a human date string rather
    than a numeric microsecond value, even with --timestamp=unix.  The new
    implementation uses list-timers --output=json whose "next" field is always
    a plain integer.

    current_timer_state is parameterised on timer_name (default TIMER_NAME).
    Tests patch UNIT_DIR so the derived path resolves inside tmp_path.
    """

    _TIMER_NAME = systemd_timer.TIMER_NAME

    def _timer_file(self, tmp_path, name: str = "") -> object:
        """Create the timer file for ``name`` (default: TIMER_NAME) under tmp_path."""
        n = name or self._TIMER_NAME
        p = tmp_path / f"{n}.timer"
        p.touch()
        return p

    def _make_show_run(self, active_state: str = "active", last_trigger: str = "") -> object:
        """Return a mock _run_systemctl that handles 'show' and 'list-timers' calls."""
        import subprocess

        def _run(*args, **kwargs):
            args_list = list(args)
            if "show" in args_list:
                stdout = f"ActiveState={active_state}\nLastTriggerUSec={last_trigger}\n"
                return subprocess.CompletedProcess(args_list, 0, stdout=stdout, stderr="")
            if "list-timers" in args_list:
                # Default: no entry (timer not loaded by systemd).
                return subprocess.CompletedProcess(args_list, 0, stdout="[]", stderr="")
            return subprocess.CompletedProcess(args_list, 0, stdout="", stderr="")

        return _run

    def _make_list_timers_run(self, next_us: int, timer_name: str = "") -> object:
        """Return a mock _run_systemctl whose list-timers call returns next_us."""
        import json
        import subprocess

        name = timer_name or self._TIMER_NAME

        def _run(*args, **kwargs):
            args_list = list(args)
            if "show" in args_list:
                return subprocess.CompletedProcess(
                    args_list,
                    0,
                    stdout="ActiveState=active\nLastTriggerUSec=\n",
                    stderr="",
                )
            if "list-timers" in args_list:
                payload = json.dumps([{"next": next_us, "last": 0, "unit": f"{name}.timer"}])
                return subprocess.CompletedProcess(args_list, 0, stdout=payload, stderr="")
            return subprocess.CompletedProcess(args_list, 0, stdout="", stderr="")

        return _run

    def test_valid_next_us_returned_as_digit_string(self, tmp_path, monkeypatch):
        """A non-zero 'next' value must come back as a digit string of microseconds."""
        monkeypatch.setattr(systemd_timer, "UNIT_DIR", tmp_path)
        self._timer_file(tmp_path)
        _next = 1_782_856_800_000_000
        with patch.object(systemd_timer, "_run_systemctl", self._make_list_timers_run(_next)):
            state = systemd_timer.current_timer_state()
        assert state["installed"] is True
        assert state["next_elapse_us"] == str(_next)
        assert state["next_elapse_us"].isdigit()

    def test_next_zero_returns_empty_string(self, tmp_path, monkeypatch):
        """'next':0 means no scheduled elapse — must return empty string."""
        monkeypatch.setattr(systemd_timer, "UNIT_DIR", tmp_path)
        self._timer_file(tmp_path)
        with patch.object(systemd_timer, "_run_systemctl", self._make_list_timers_run(0)):
            state = systemd_timer.current_timer_state()
        assert state["next_elapse_us"] == ""

    def test_empty_json_array_returns_empty_string(self, tmp_path, monkeypatch):
        """An empty list-timers array (unit not loaded) must return empty string."""
        monkeypatch.setattr(systemd_timer, "UNIT_DIR", tmp_path)
        self._timer_file(tmp_path)
        with patch.object(systemd_timer, "_run_systemctl", self._make_show_run()):
            state = systemd_timer.current_timer_state()
        assert state["next_elapse_us"] == ""

    def test_missing_unit_in_json_returns_empty_string(self, tmp_path, monkeypatch):
        """If the matching unit is absent from the JSON, return empty string."""
        import json
        import subprocess

        monkeypatch.setattr(systemd_timer, "UNIT_DIR", tmp_path)
        self._timer_file(tmp_path)

        def _run(*args, **kwargs):
            args_list = list(args)
            if "show" in args_list:
                return subprocess.CompletedProcess(
                    args_list, 0, stdout="ActiveState=active\nLastTriggerUSec=\n", stderr=""
                )
            if "list-timers" in args_list:
                # A different unit is returned — the target unit is absent.
                payload = json.dumps([{"next": 9_000_000, "last": 0, "unit": "other.timer"}])
                return subprocess.CompletedProcess(args_list, 0, stdout=payload, stderr="")
            return subprocess.CompletedProcess(args_list, 0, stdout="", stderr="")

        with patch.object(systemd_timer, "_run_systemctl", _run):
            state = systemd_timer.current_timer_state()
        assert state["next_elapse_us"] == ""

    def test_active_and_last_trigger_from_show(self, tmp_path, monkeypatch):
        """ActiveState and LastTriggerUSec must still come from systemctl show."""
        monkeypatch.setattr(systemd_timer, "UNIT_DIR", tmp_path)
        self._timer_file(tmp_path)
        _next = 1_782_000_000_000_000

        import subprocess

        def _run(*args, **kwargs):
            args_list = list(args)
            if "show" in args_list:
                return subprocess.CompletedProcess(
                    args_list,
                    0,
                    stdout="ActiveState=active\nLastTriggerUSec=1782000000000000\n",
                    stderr="",
                )
            if "list-timers" in args_list:
                import json

                payload = json.dumps(
                    [{"next": _next, "last": 0, "unit": f"{self._TIMER_NAME}.timer"}]
                )
                return subprocess.CompletedProcess(args_list, 0, stdout=payload, stderr="")
            return subprocess.CompletedProcess(args_list, 0, stdout="", stderr="")

        with patch.object(systemd_timer, "_run_systemctl", _run):
            state = systemd_timer.current_timer_state()
        assert state["active"] is True
        assert state["last_trigger_us"] == "1782000000000000"
        assert state["next_elapse_us"] == str(_next)

    def test_timer_path_not_exists_returns_installed_false(self, tmp_path, monkeypatch):
        """Non-existent timer file must return {'installed': False}."""
        monkeypatch.setattr(systemd_timer, "UNIT_DIR", tmp_path)
        # No file created — the derived path does not exist.
        state = systemd_timer.current_timer_state()
        assert state == {"installed": False}

    def test_malformed_json_returns_empty_next(self, tmp_path, monkeypatch):
        """Malformed list-timers JSON must not crash and must leave next_elapse_us empty."""
        import subprocess

        monkeypatch.setattr(systemd_timer, "UNIT_DIR", tmp_path)
        self._timer_file(tmp_path)

        def _run(*args, **kwargs):
            args_list = list(args)
            if "show" in args_list:
                return subprocess.CompletedProcess(
                    args_list, 0, stdout="ActiveState=active\nLastTriggerUSec=\n", stderr=""
                )
            if "list-timers" in args_list:
                return subprocess.CompletedProcess(
                    args_list, 0, stdout="not valid json {{", stderr=""
                )
            return subprocess.CompletedProcess(args_list, 0, stdout="", stderr="")

        with patch.object(systemd_timer, "_run_systemctl", _run):
            state = systemd_timer.current_timer_state()
        assert state["next_elapse_us"] == ""
        assert state["installed"] is True

    def test_backup_timer_name_reads_backup_unit(self, tmp_path, monkeypatch):
        """cached_timer_state('paramem-backup') reads paramem-backup.timer, not consolidate."""
        import json
        import subprocess

        monkeypatch.setattr(systemd_timer, "UNIT_DIR", tmp_path)
        # Create the backup timer file; do NOT create the consolidation one.
        self._timer_file(tmp_path, "paramem-backup")
        _next = 1_783_000_000_000_000

        def _run(*args, **kwargs):
            args_list = list(args)
            if "show" in args_list:
                return subprocess.CompletedProcess(
                    args_list, 0, stdout="ActiveState=active\nLastTriggerUSec=\n", stderr=""
                )
            if "list-timers" in args_list:
                payload = json.dumps([{"next": _next, "last": 0, "unit": "paramem-backup.timer"}])
                return subprocess.CompletedProcess(args_list, 0, stdout=payload, stderr="")
            return subprocess.CompletedProcess(args_list, 0, stdout="", stderr="")

        # max_age_seconds=0 bypasses the module-level cache to force a fresh read.
        with patch.object(systemd_timer, "_run_systemctl", _run):
            state = systemd_timer.cached_timer_state("paramem-backup", max_age_seconds=0)
        assert state["installed"] is True
        assert state["next_elapse_us"] == str(_next)
        # Default consolidation timer (not created) must still return not-installed.
        default_state = systemd_timer.current_timer_state()
        assert default_state == {"installed": False}
