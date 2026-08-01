"""Unit tests for paramem.server.schedule_grammar — the shared schedule-string
grammar, incl. the suspend/power-off catch-up gate helpers added alongside
the systemd_timer heartbeat rework, and the calendar-mark / scheduled-run-due
helpers that generalise dueness math to every cadence kind.

Covers:
- parse_schedule_atom: the accept/reject boundary, incl. the "daily HH:MM"
  operator idiom.
- compute_schedule_period_seconds accepting the "daily HH:MM" idiom (the
  TRAP this catch-up-gate change had to resolve — the default backup
  schedule is "daily 04:00").
- _is_due (module-internal): the shared last-attempt gate, exercised
  directly since it is the non-exact-interval branch of scheduled_run_due.
- previous_mark: the most recent calendar mark for every cadence kind.
- scheduled_run_due / scheduled_run_stamp_value: the dueness predicate and
  stamp-value pair built on top of previous_mark, covering exact and
  non-exact cadences alike.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from paramem.server.schedule_grammar import (
    DAILY_ANCHOR_HOUR,
    DAILY_ANCHOR_MINUTE,
    WEEKLY_ANCHOR_HOUR,
    WEEKLY_ANCHOR_MINUTE,
    WEEKLY_ANCHOR_WEEKDAY,
    ScheduleDueStatus,
    _is_due,
    compute_schedule_period_seconds,
    parse_schedule_atom,
    previous_mark,
    scheduled_run_due,
    scheduled_run_stamp_value,
)

# ---------------------------------------------------------------------------
# parse_schedule_atom — the accept/reject boundary
#
# Asserted directly because no other helper can serve as the oracle:
# scheduled_run_due's NO_STAMP/NOT_DUE/DUE result collapses both an off
# schedule and unparseable input to NOT_DUE, so a widening of the grammar is
# invisible through it.
# ---------------------------------------------------------------------------


class TestParseScheduleAtomBoundary:
    @pytest.mark.parametrize(
        ("schedule", "kind"),
        [
            (None, "off"),
            ("", "off"),
            ("off", "off"),
            ("Daily 4:00", "hhmm"),
            ("disabled", "off"),
            ("none", "off"),
            ("weekly", "weekly"),
            ("daily", "daily"),
            ("04:00", "hhmm"),
            ("4:00", "hhmm"),
            ("23:59", "hhmm"),
            ("00:00", "hhmm"),
            ("daily 04:00", "hhmm"),
            ("DAILY 04:00", "hhmm"),
            ("12h", "interval"),
            ("every 12h", "interval"),
            ("every 30m", "interval"),
            ("every 2H", "interval"),
            ("every 30M", "interval"),
        ],
    )
    def test_accepted_forms(self, schedule: str, kind: str) -> None:
        atom = parse_schedule_atom(schedule)
        assert atom is not None, f"{schedule!r} must parse"
        assert atom.kind == kind

    @pytest.mark.parametrize(
        "schedule",
        [
            "bogus",
            "biweekly",
            "every",
            "every 0h",  # zero interval
            "every -1h",
            "25:00",  # hour out of range
            "12:60",  # minute out of range
            "4:0",  # minute must be two digits
            "12",  # no unit
            "12d",  # unsupported unit
            "daily 25:00",  # prefix idiom does not bypass the range check
            "daily off",  # consuming the prefix must not re-open the atom list
            "daily foo",
            "every 12 h m",
        ],
    )
    def test_rejected_forms(self, schedule: str) -> None:
        assert parse_schedule_atom(schedule) is None, f"{schedule!r} must not parse"

    def test_hhmm_range_check_survives_the_daily_prefix(self) -> None:
        """The shared _HHMM_SHAPE fragment is shape-only; the range check is
        applied once, after the prefix idiom is normalised away."""
        assert parse_schedule_atom("daily 23:59").kind == "hhmm"
        assert parse_schedule_atom("daily 24:00") is None

    def test_unparseable_raises_in_compute_period(self) -> None:
        with pytest.raises(ValueError, match="Unrecognised schedule string"):
            compute_schedule_period_seconds("bogus")


# ---------------------------------------------------------------------------
# compute_schedule_period_seconds — "daily HH:MM" path (the TRAP)
# ---------------------------------------------------------------------------


class TestComputeSchedulePeriodSecondsDailyPrefix:
    def test_daily_hhmm_returns_86400(self):
        """'daily 04:00' (the server.yaml default backup schedule) → 86400.

        Before the daily-prefix strip was hoisted into parse_schedule_atom, this
        raised ValueError — only paramem.backup.timer.reconcile normalised
        the idiom, and compute_schedule_period_seconds did not.
        """
        assert compute_schedule_period_seconds("daily 04:00") == 86400

    def test_daily_hhmm_case_insensitive(self):
        assert compute_schedule_period_seconds("DAILY 04:00") == 86400


# ---------------------------------------------------------------------------
# _is_due (module-internal — the non-exact-interval branch of
# scheduled_run_due, which already resolves last_stamp=None to NO_STAMP
# before this is ever reached).
# ---------------------------------------------------------------------------


class TestIsDue:
    def test_due_when_period_elapsed(self):
        now = 1_000_000.0
        assert _is_due(now - 3600, 3600, now=now) is True

    def test_not_due_when_period_not_elapsed(self):
        now = 1_000_000.0
        assert _is_due(now - 1800, 3600, now=now) is False

    def test_due_exactly_at_period_boundary(self):
        """Exactly period_seconds elapsed → due (>=, not >)."""
        now = 1_000_000.0
        assert _is_due(now - 3600, 3600, now=now) is True

    def test_default_now_uses_wall_clock(self):
        """now=None resolves to time.time() — a last attempt far in the past is due."""
        assert _is_due(0.0, 3600) is True


# ---------------------------------------------------------------------------
# previous_mark
#
# All fixed epochs anchor on 2024-01-15 (Monday), a DST-transition-free date
# (CET/CEST transitions land in March/October) so the tests are deterministic
# regardless of the host's local timezone — same convention as
# tests/test_systemd_timer_calendar.py::TestFloorToHeartbeatDriftRegression.
# ---------------------------------------------------------------------------


def _local(*args) -> float:
    """Build a local-time epoch from datetime(...) constructor args."""
    return datetime(*args).timestamp()


class TestPreviousMarkOffAndUnparseable:
    @pytest.mark.parametrize("schedule", ["", "off", "disabled", "none"])
    def test_off_variants_return_none(self, schedule):
        assert previous_mark(schedule, _local(2024, 1, 15, 12, 0)) is None

    @pytest.mark.parametrize("schedule", ["bogus", "every 0h", "25:00"])
    def test_unparseable_returns_none(self, schedule):
        assert previous_mark(schedule, _local(2024, 1, 15, 12, 0)) is None


class TestPreviousMarkExactInterval:
    def test_12h_marks_at_midnight_and_noon(self):
        assert previous_mark("every 12h", _local(2024, 1, 15, 13, 0)) == _local(2024, 1, 15, 12, 0)
        assert previous_mark("every 12h", _local(2024, 1, 15, 5, 0)) == _local(2024, 1, 15, 0, 0)
        assert previous_mark("every 12h", _local(2024, 1, 15, 12, 0)) == _local(2024, 1, 15, 12, 0)

    def test_non_exact_interval_has_no_mark(self):
        """'every 5h' does not divide 24 — no wall-clock mark exists."""
        assert previous_mark("every 5h", _local(2024, 1, 15, 13, 0)) is None
        assert previous_mark("every 90m", _local(2024, 1, 15, 13, 0)) is None


class TestPreviousMarkDailyHHMM:
    def test_now_before_anchor_returns_yesterdays_mark(self):
        """now=02:00 with the cadence anchored at 04:00 → yesterday's 04:00."""
        mark = previous_mark("daily 04:00", _local(2024, 1, 15, 2, 0))
        assert mark == _local(2024, 1, 14, 4, 0)

    def test_now_after_anchor_returns_todays_mark(self):
        """now=05:00 with the cadence anchored at 04:00 → today's 04:00."""
        mark = previous_mark("daily 04:00", _local(2024, 1, 15, 5, 0))
        assert mark == _local(2024, 1, 15, 4, 0)

    def test_now_exactly_at_anchor_returns_todays_mark(self):
        mark = previous_mark("daily 04:00", _local(2024, 1, 15, 4, 0))
        assert mark == _local(2024, 1, 15, 4, 0)


class TestPreviousMarkBareDaily:
    def test_anchored_at_daily_anchor_constant(self):
        """Bare 'daily' anchors at DAILY_ANCHOR_HOUR:MINUTE (03:00) — the ONE
        definition systemd_timer.parse_schedule also renders from."""
        assert DAILY_ANCHOR_HOUR == 3
        assert DAILY_ANCHOR_MINUTE == 0
        mark = previous_mark("daily", _local(2024, 1, 15, 10, 0))
        assert mark == _local(2024, 1, 15, DAILY_ANCHOR_HOUR, DAILY_ANCHOR_MINUTE)

    def test_before_anchor_falls_back_to_yesterday(self):
        mark = previous_mark("daily", _local(2024, 1, 15, 1, 0))
        assert mark == _local(2024, 1, 14, DAILY_ANCHOR_HOUR, DAILY_ANCHOR_MINUTE)


class TestPreviousMarkWeekly:
    """Weekly must land on WEEKLY_ANCHOR_WEEKDAY (Monday) 00:00 — NOT degrade
    to a daily midnight floor, which is the failure mode a naive
    floor-from-local-midnight generalisation would produce.
    """

    def test_weekday_mark_is_this_weeks_monday_not_todays_midnight(self):
        """2024-01-17 is a Wednesday; the mark must be Monday 2024-01-15,
        not Wednesday's own midnight."""
        assert WEEKLY_ANCHOR_WEEKDAY == 0
        mark = previous_mark("weekly", _local(2024, 1, 17, 15, 0))
        assert mark == _local(2024, 1, 15, WEEKLY_ANCHOR_HOUR, WEEKLY_ANCHOR_MINUTE)
        assert mark != _local(2024, 1, 17, 0, 0), (
            "weekly must not degrade to a daily-midnight floor"
        )

    def test_sunday_night_rolls_back_to_previous_mondays_mark(self):
        """2024-01-14 is a Sunday; the mark must be the PRECEDING Monday
        (2024-01-08), 6 days back — not today's midnight."""
        mark = previous_mark("weekly", _local(2024, 1, 14, 23, 0))
        assert mark == _local(2024, 1, 8, 0, 0)

    def test_monday_early_morning_uses_same_days_mark(self):
        mark = previous_mark("weekly", _local(2024, 1, 15, 0, 30))
        assert mark == _local(2024, 1, 15, 0, 0)


# ---------------------------------------------------------------------------
# scheduled_run_due / scheduled_run_stamp_value
# ---------------------------------------------------------------------------


class TestScheduledRunDueNoStamp:
    @pytest.mark.parametrize("schedule", ["daily", "weekly", "every 12h", "every 5h", "off"])
    def test_absent_stamp_is_no_stamp_for_every_kind(self, schedule):
        """The caller owns the no-stamp policy (seed-and-noop vs run-now) —
        this function never guesses one, for any cadence kind."""
        assert (
            scheduled_run_due(schedule, None, now=_local(2024, 1, 15, 12, 0))
            is ScheduleDueStatus.NO_STAMP
        )


class TestScheduledRunDueExactKinds:
    def test_daily_hhmm_not_due_before_anchor(self):
        """now=02:00, stamp=yesterday's 04:00, cadence anchored 04:00 → NOT due."""
        last_stamp = _local(2024, 1, 14, 4, 0)
        status = scheduled_run_due("daily 04:00", last_stamp, now=_local(2024, 1, 15, 2, 0))
        assert status is ScheduleDueStatus.NOT_DUE

    def test_daily_hhmm_due_after_anchor(self):
        """now=05:00, stamp=yesterday's 04:00, cadence anchored 04:00 → due."""
        last_stamp = _local(2024, 1, 14, 4, 0)
        status = scheduled_run_due("daily 04:00", last_stamp, now=_local(2024, 1, 15, 5, 0))
        assert status is ScheduleDueStatus.DUE

    def test_weekly_due_after_crossing_into_a_new_week(self):
        last_stamp = _local(2024, 1, 8, 0, 0)  # previous Monday's mark
        status = scheduled_run_due("weekly", last_stamp, now=_local(2024, 1, 15, 9, 0))
        assert status is ScheduleDueStatus.DUE

    def test_weekly_not_due_within_the_same_week(self):
        last_stamp = _local(2024, 1, 15, 0, 0)  # this Monday's mark
        status = scheduled_run_due("weekly", last_stamp, now=_local(2024, 1, 17, 9, 0))
        assert status is ScheduleDueStatus.NOT_DUE

    def test_off_schedule_is_never_due(self):
        status = scheduled_run_due("off", 0.0, now=_local(2024, 1, 15, 12, 0))
        assert status is ScheduleDueStatus.NOT_DUE

    def test_unparseable_schedule_is_never_due(self):
        status = scheduled_run_due("bogus", 0.0, now=_local(2024, 1, 15, 12, 0))
        assert status is ScheduleDueStatus.NOT_DUE


class TestScheduledRunDueRedundantWakeupIdempotency:
    """A stamp written at the mark value must read NOT_DUE on a second
    evaluation inside the same mark's window — the whole point of stamping
    the mark rather than raw ``now``.
    """

    def test_exact_interval_second_wakeup_in_same_window_not_due(self):
        first_fire = _local(2024, 1, 15, 12, 5)  # 5 min after the 12:00 mark
        stamp = scheduled_run_stamp_value("every 12h", first_fire)
        assert stamp == _local(2024, 1, 15, 12, 0)

        second_fire = _local(2024, 1, 15, 12, 45)  # still before the 00:00 mark
        status = scheduled_run_due("every 12h", stamp, now=second_fire)
        assert status is ScheduleDueStatus.NOT_DUE

    def test_daily_hhmm_second_wakeup_in_same_window_not_due(self):
        first_fire = _local(2024, 1, 15, 4, 5)
        stamp = scheduled_run_stamp_value("daily 04:00", first_fire)
        assert stamp == _local(2024, 1, 15, 4, 0)

        second_fire = _local(2024, 1, 15, 23, 0)
        status = scheduled_run_due("daily 04:00", stamp, now=second_fire)
        assert status is ScheduleDueStatus.NOT_DUE

    def test_next_windows_wakeup_is_due(self):
        first_fire = _local(2024, 1, 15, 12, 5)
        stamp = scheduled_run_stamp_value("every 12h", first_fire)

        next_window_fire = _local(2024, 1, 16, 0, 5)  # past the next 00:00 mark
        status = scheduled_run_due("every 12h", stamp, now=next_window_fire)
        assert status is ScheduleDueStatus.DUE


class TestScheduledRunDueNonExactInterval:
    """Non-exact intervals keep period-based dueness on the heartbeat-floored
    stamp, exercised here directly through the grammar's own API
    (``scheduled_run_stamp_value`` / ``scheduled_run_due``).
    """

    def test_every_5h_stamp_value_floors_to_the_hourly_gcd_grid(self):
        """gcd(5, 24) == 1 -> hourly grid; the stamp floors to the current hour."""
        now = _local(2024, 1, 15, 13, 7)
        stamp = scheduled_run_stamp_value("every 5h", now)
        assert stamp == _local(2024, 1, 15, 13, 0)

    def test_due_once_period_elapses(self):
        period_s = 5 * 3600
        first_fire = _local(2024, 1, 15, 0, 0)
        stamp = scheduled_run_stamp_value("every 5h", first_fire)

        not_yet = first_fire + period_s - 1
        assert scheduled_run_due("every 5h", stamp, now=not_yet) is ScheduleDueStatus.NOT_DUE

        exactly_due = first_fire + period_s
        assert scheduled_run_due("every 5h", stamp, now=exactly_due) is ScheduleDueStatus.DUE

    def test_consecutive_dispatches_stay_exactly_period_apart(self):
        """Same drift-regression scenario as
        TestFloorToHeartbeatDriftRegression in test_systemd_timer_calendar.py,
        exercised through the new scheduled_run_stamp_value API instead of
        floor_to_heartbeat directly."""
        period_s = 5 * 3600
        base = _local(2024, 1, 15, 0, 0)
        heartbeat_fire_times = [base, base + period_s, base + 2 * period_s]
        dispatch_delays = [7.0, 143.0, 1.0]

        stamps = [
            scheduled_run_stamp_value("every 5h", fire_t + delay)
            for fire_t, delay in zip(heartbeat_fire_times, dispatch_delays)
        ]
        gaps = [stamps[i + 1] - stamps[i] for i in range(len(stamps) - 1)]
        assert gaps == [period_s, period_s]


class TestScheduledRunStampValueGrid:
    """The gcd(count, modulus) heartbeat grid each non-exact interval lands
    on, exercised through :func:`scheduled_run_stamp_value`'s flooring — see
    ``systemd_timer._period_heartbeat_calendar``, which renders the identical
    grid from the same ``schedule_grammar.non_exact_interval_grid``.
    """

    def test_every_5h_hourly_grid(self):
        """gcd(5, 24) == 1 -> hourly grid."""
        stamp = scheduled_run_stamp_value("every 5h", _local(2024, 1, 15, 13, 45))
        assert stamp == _local(2024, 1, 15, 13, 0)

    def test_every_90m_half_hour_grid(self):
        """gcd(90, 60) == 30 -> half-hour grid (48 ticks/day, not 1440)."""
        stamp = scheduled_run_stamp_value("every 90m", _local(2024, 1, 15, 13, 45))
        assert stamp == _local(2024, 1, 15, 13, 30)

    def test_every_48h_daily_grid(self):
        """gcd(48, 24) == 24 -> daily grid (1 tick/day, not 24)."""
        stamp = scheduled_run_stamp_value("every 48h", _local(2024, 1, 15, 13, 45))
        assert stamp == _local(2024, 1, 15, 0, 0)

    def test_every_7m_per_minute_grid(self):
        """gcd(7, 60) == 1 -> per-minute grid."""
        stamp = scheduled_run_stamp_value("every 7m", _local(2024, 1, 15, 13, 45, 30))
        assert stamp == _local(2024, 1, 15, 13, 45)


class TestScheduledRunStampValueRaises:
    @pytest.mark.parametrize("schedule", ["off", "", "disabled", "none", "bogus"])
    def test_off_or_unparseable_raises(self, schedule):
        with pytest.raises(ValueError):
            scheduled_run_stamp_value(schedule, _local(2024, 1, 15, 12, 0))
