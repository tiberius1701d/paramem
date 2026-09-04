"""Unit tests for the consolidation timer parser + reconciler.

Schedule parsing lives in `paramem.server.systemd_timer.parse_schedule` since
the server delegates actual scheduling to a systemd user timer.
"""

import subprocess
from unittest.mock import patch

import pytest

from paramem.server import systemd_timer
from paramem.utils import systemctl

# ---------------------------------------------------------------------------
# TestParseSchedule — systemd_timer.parse_schedule: the string -> TimerSpec
# grammar every timer-carrying schedule string (refresh_cadence, a window
# start, security.backups.schedule) is rendered through.
# ---------------------------------------------------------------------------


class TestParseSchedule:
    def test_malformed_input_returns_none(self) -> None:
        assert systemd_timer.parse_schedule("bogus") is None
        assert systemd_timer.parse_schedule("12x") is None

    @pytest.mark.parametrize(
        "schedule", ["", "off", "disabled", "none", "OFF", "Off", "DISABLED", "None"]
    )
    def test_every_off_spelling_and_case_variant_renders_the_off_spec(self, schedule) -> None:
        spec = systemd_timer.parse_schedule(schedule)

        assert spec == systemd_timer.TimerSpec(kind="off")
        assert spec.on_calendars == ()

    def test_15m_renders_one_minute_grid_entry(self) -> None:
        spec = systemd_timer.parse_schedule("15m")

        assert spec.kind == "calendar"
        assert spec.on_calendars == ("*:00,15,30,45:00",)

    def test_every_15m_renders_the_identical_minute_grid_entry(self) -> None:
        spec = systemd_timer.parse_schedule("every 15m")

        assert spec.kind == "calendar"
        assert spec.on_calendars == ("*:00,15,30,45:00",)

    def test_daily_renders_the_anchored_daily_expression(self) -> None:
        spec = systemd_timer.parse_schedule("daily")

        assert spec.kind == "daily"
        assert spec.on_calendars == ("*-*-* 03:00:00",)

    def test_daily_hhmm_renders_a_daily_expression_at_that_time(self) -> None:
        spec = systemd_timer.parse_schedule("daily 04:00")

        assert spec.kind == "daily"
        assert spec.on_calendars == ("*-*-* 04:00:00",)

    def test_bare_hhmm_renders_a_daily_expression_at_that_time(self) -> None:
        spec = systemd_timer.parse_schedule("04:00")

        assert spec.kind == "daily"
        assert spec.on_calendars == ("*-*-* 04:00:00",)

    def test_weekly_renders_the_anchored_weekly_expression(self) -> None:
        spec = systemd_timer.parse_schedule("weekly")

        assert spec.kind == "calendar"
        assert spec.on_calendars == ("Mon *-*-* 00:00:00",)

    def test_12h_renders_the_exact_two_mark_grid(self) -> None:
        spec = systemd_timer.parse_schedule("12h")

        assert spec.kind == "calendar"
        assert spec.on_calendars == ("*-*-* 00,12:00:00",)

    def test_every_12h_renders_the_identical_exact_grid(self) -> None:
        spec = systemd_timer.parse_schedule("every 12h")

        assert spec.kind == "calendar"
        assert spec.on_calendars == ("*-*-* 00,12:00:00",)

    def test_surrounding_whitespace_is_stripped_for_off_and_interval_forms(self) -> None:
        assert systemd_timer.parse_schedule("  off  ") == systemd_timer.TimerSpec(kind="off")
        assert systemd_timer.parse_schedule(" 12h ") == systemd_timer.parse_schedule("12h")
        assert systemd_timer.parse_schedule(" every 12h ") == systemd_timer.parse_schedule("12h")


class TestReconcileEnableGating:
    """``enable --now`` (and ``daemon-reload``/``restart``) only fire when the
    rendered unit content actually changed — not on every reconcile call.

    A cadence-only edit reaches ``_reconcile_timer`` through
    ``_apply_config_live`` on every operator accept/rollback, so an
    unconditional ``enable --now`` would spam a real ``systemctl`` write
    against the live user session even when nothing changed. The returned
    state string must be truthful: "already current" only when no systemctl
    call was actually made.
    """

    def _mock_run(self, *args, **kwargs):
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    def _install(self, tmp_path, monkeypatch):
        monkeypatch.setattr(systemd_timer, "UNIT_DIR", tmp_path)
        monkeypatch.setattr(systemd_timer, "TIMER_PATH", tmp_path / "paramem-consolidate.timer")
        monkeypatch.setattr(systemd_timer, "SERVICE_PATH", tmp_path / "paramem-consolidate.service")

    def test_first_install_calls_enable_and_reports_updated(self, tmp_path, monkeypatch):
        """Fresh machine (no unit files yet): content changes → enable/restart fire,
        message says 'updated', never 'already current'."""
        self._install(tmp_path, monkeypatch)
        with patch.object(systemctl, "run", side_effect=self._mock_run) as mock_run:
            msg = systemd_timer.reconcile("every 12h")

        assert "updated" in msg
        assert "already current" not in msg
        first_args = [c.args[0] for c in mock_run.call_args_list]
        assert "daemon-reload" in first_args
        assert "enable" in first_args
        assert "restart" in first_args

    def test_unchanged_second_call_skips_all_systemctl_calls(self, tmp_path, monkeypatch):
        """Reconciling the SAME schedule twice: the second call must not touch
        systemctl at all (nothing changed), and must truthfully report
        'already current'."""
        self._install(tmp_path, monkeypatch)
        with patch.object(systemctl, "run", side_effect=self._mock_run):
            systemd_timer.reconcile("every 12h")

        with patch.object(systemctl, "run", side_effect=self._mock_run) as mock_run_second:
            msg = systemd_timer.reconcile("every 12h")

        assert "already current" in msg
        assert "updated" not in msg
        assert mock_run_second.call_args_list == [], (
            "reconcile() re-issued systemctl calls when nothing changed: "
            f"{mock_run_second.call_args_list}"
        )

    def test_cadence_change_calls_enable_and_reports_updated_not_already_current(
        self, tmp_path, monkeypatch
    ):
        """A genuine cadence change on the second call re-triggers enable/restart
        and must report 'updated', never 'already current' (message truthfulness:
        the action taken must be reflected in the returned state)."""
        self._install(tmp_path, monkeypatch)
        with patch.object(systemctl, "run", side_effect=self._mock_run):
            systemd_timer.reconcile("every 12h")

        with patch.object(systemctl, "run", side_effect=self._mock_run) as mock_run_second:
            msg = systemd_timer.reconcile("every 6h")

        assert "updated" in msg
        assert "already current" not in msg
        called_verbs = [c.args[:1] for c in mock_run_second.call_args_list]
        assert ("enable",) in called_verbs
        assert ("restart",) in called_verbs

    def test_off_to_off_still_skips_systemctl_when_no_units_installed(self, tmp_path, monkeypatch):
        """Off→off with nothing installed never calls systemctl (nothing to remove)."""
        self._install(tmp_path, monkeypatch)
        with patch.object(systemctl, "run", side_effect=self._mock_run) as mock_run:
            msg = systemd_timer.reconcile("off")

        assert "disabled" in msg
        assert mock_run.call_args_list == []
