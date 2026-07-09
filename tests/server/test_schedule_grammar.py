"""Unit tests for paramem.server.schedule_grammar — the shared schedule-string
grammar, incl. the suspend/power-off catch-up gate helpers added alongside
the systemd_timer heartbeat rework.

Covers:
- strip_daily_prefix: "daily HH:MM" -> "HH:MM" normalisation.
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
    strip_daily_prefix,
)

# ---------------------------------------------------------------------------
# strip_daily_prefix
# ---------------------------------------------------------------------------


class TestStripDailyPrefix:
    def test_daily_hhmm_strips_to_hhmm(self):
        assert strip_daily_prefix("daily 04:00") == "04:00"

    def test_daily_hhmm_case_insensitive(self):
        assert strip_daily_prefix("DAILY 04:00") == "04:00"
        assert strip_daily_prefix("Daily 4:00") == "4:00"

    def test_bare_daily_unchanged(self):
        """Bare 'daily' is its own grammar atom, not this idiom."""
        assert strip_daily_prefix("daily") == "daily"

    def test_hhmm_unchanged(self):
        assert strip_daily_prefix("04:00") == "04:00"

    def test_interval_unchanged(self):
        assert strip_daily_prefix("every 12h") == "every 12h"

    def test_none_returns_none(self):
        assert strip_daily_prefix(None) is None

    def test_off_unchanged(self):
        assert strip_daily_prefix("off") == "off"


# ---------------------------------------------------------------------------
# compute_schedule_period_seconds — "daily HH:MM" path (the TRAP)
# ---------------------------------------------------------------------------


class TestComputeSchedulePeriodSecondsDailyPrefix:
    def test_daily_hhmm_returns_86400(self):
        """'daily 04:00' (the server.yaml default backup schedule) → 86400.

        Before strip_daily_prefix was hoisted into parse_schedule_atom, this
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
