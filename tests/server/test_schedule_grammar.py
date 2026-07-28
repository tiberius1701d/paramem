"""Unit tests for paramem.server.schedule_grammar — the shared schedule-string
grammar, incl. the suspend/power-off catch-up gate helpers added alongside
the systemd_timer heartbeat rework.

Covers:
- parse_schedule_atom: the accept/reject boundary, incl. the "daily HH:MM"
  operator idiom.
- compute_schedule_period_seconds accepting the "daily HH:MM" idiom (the
  TRAP this catch-up-gate change had to resolve — the default backup
  schedule is "daily 04:00").
- is_calendar_exact: anchored / exact-divisor / non-exact / off / unparseable.
- is_due: the shared last-attempt gate, including last_attempt_epoch=None.
"""

from __future__ import annotations

import pytest

from paramem.server.schedule_grammar import (
    compute_schedule_period_seconds,
    is_calendar_exact,
    is_due,
    parse_schedule_atom,
)

# ---------------------------------------------------------------------------
# parse_schedule_atom — the accept/reject boundary
#
# Asserted directly because no other helper can serve as the oracle:
# is_calendar_exact returns True both for a valid anchored atom AND for
# unparseable input, so a widening of the grammar is invisible through it.
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
# is_calendar_exact
# ---------------------------------------------------------------------------


class TestIsCalendarExact:
    @pytest.mark.parametrize("schedule", ["daily", "weekly", "04:00", "daily 04:00", "23:59"])
    def test_anchored_schedules_are_exact(self, schedule):
        assert is_calendar_exact(schedule) is True

    @pytest.mark.parametrize("schedule", ["every 12h", "every 6h", "every 24h", "12h", "2h"])
    def test_exact_divisor_hour_intervals_are_exact(self, schedule):
        assert is_calendar_exact(schedule) is True

    @pytest.mark.parametrize("schedule", ["every 30m", "every 15m", "every 5m"])
    def test_exact_divisor_minute_intervals_are_exact(self, schedule):
        assert is_calendar_exact(schedule) is True

    @pytest.mark.parametrize("schedule", ["every 5h", "every 7h", "every 48h", "every 11h"])
    def test_non_divisor_hour_intervals_are_not_exact(self, schedule):
        assert is_calendar_exact(schedule) is False

    @pytest.mark.parametrize("schedule", ["every 7m", "every 13m", "every 90m"])
    def test_non_divisor_minute_intervals_are_not_exact(self, schedule):
        assert is_calendar_exact(schedule) is False

    @pytest.mark.parametrize("schedule", ["", "off", "disabled", "none"])
    def test_off_variants_are_exact(self, schedule):
        """Off means no gate applies — vacuously exact."""
        assert is_calendar_exact(schedule) is True

    @pytest.mark.parametrize("schedule", ["bogus", "every 0h", "25:00", "every"])
    def test_unparseable_is_exact(self, schedule):
        """An invalid schedule is handled where it is parsed for real
        (reconcile / ServerConfig validation); the catch-up gate must not
        double-report the same error as a second symptom.
        """
        assert is_calendar_exact(schedule) is True


# ---------------------------------------------------------------------------
# is_due
# ---------------------------------------------------------------------------


class TestIsDue:
    def test_none_last_attempt_is_always_due(self):
        assert is_due(None, 3600) is True

    def test_due_when_period_elapsed(self):
        now = 1_000_000.0
        assert is_due(now - 3600, 3600, now=now) is True

    def test_not_due_when_period_not_elapsed(self):
        now = 1_000_000.0
        assert is_due(now - 1800, 3600, now=now) is False

    def test_due_exactly_at_period_boundary(self):
        """Exactly period_seconds elapsed → due (>=, not >)."""
        now = 1_000_000.0
        assert is_due(now - 3600, 3600, now=now) is True

    def test_default_now_uses_wall_clock(self):
        """now=None resolves to time.time() — a last attempt far in the past is due."""
        assert is_due(0.0, 3600) is True
