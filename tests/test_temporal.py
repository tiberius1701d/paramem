"""Tests for the pure date/grouping primitives in ``paramem.server.temporal``.

Stdlib-only module under test — no store, model, or config required.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from paramem.server.temporal import DateWindow, build_date_by_key, last_seen_date, weekday_name


class TestWeekdayName:
    """``weekday_name`` renders the English weekday name for a
    ``datetime.date`` without going through ``strftime('%A')``, which is
    ``LC_TIME``-locale dependent."""

    @pytest.mark.parametrize(
        ("day", "expected"),
        [
            (date(2026, 8, 3), "Monday"),
            (date(2026, 8, 4), "Tuesday"),
            (date(2026, 8, 5), "Wednesday"),
            (date(2026, 8, 6), "Thursday"),
            (date(2026, 8, 7), "Friday"),
            (date(2026, 8, 8), "Saturday"),
            (date(2026, 8, 9), "Sunday"),
        ],
    )
    def test_known_dates_all_seven_weekdays(self, day, expected):
        assert weekday_name(day) == expected


class TestLastSeenDate:
    def test_aware_utc_iso_string(self):
        """Exact shape ``datetime.now(timezone.utc).isoformat()`` produces
        (``session_buffer.py:318``) — converts to the local calendar day."""
        moment = datetime(2026, 8, 6, 23, 30, 0, tzinfo=timezone.utc)
        raw = moment.isoformat()
        result = last_seen_date(raw)
        assert result == moment.astimezone().date()

    def test_naive_iso_string(self):
        result = last_seen_date("2026-08-06T10:15:00")
        assert result == date(2026, 8, 6)

    def test_empty_string(self):
        assert last_seen_date("") is None

    def test_none(self):
        assert last_seen_date(None) is None

    def test_int(self):
        assert last_seen_date(1234567890) is None

    def test_garbage_string(self):
        assert last_seen_date("not-a-date") is None

    def test_garbage_string_never_raises(self):
        for garbage in ["", "   ", "2026-13-99", "yesterday", "{}", "null", "\n"]:
            # Must not raise for any of these.
            last_seen_date(garbage)

    def test_aware_year_9999_overflow_returns_none(self):
        """An aware year-9999 timestamp with a NEGATIVE UTC offset overflows
        past ``datetime.MAXYEAR`` in the first step of ``.astimezone()``
        (``self - self.utcoffset()``, which is an addition for a negative
        offset) — before the host's own local offset is ever applied, so
        this raises ``OverflowError`` deterministically on every host
        regardless of its timezone. A ``+00:00`` input would only overflow
        on a host whose local offset is positive, which made an earlier
        version of this test host-dependent."""
        assert last_seen_date("9999-12-31T23:59:59-01:00") is None

    def test_aware_year_1_overflow_returns_none(self):
        """Symmetric case at the other end of the representable range: an
        aware year-1 timestamp can shift past ``datetime.MINYEAR``."""
        assert last_seen_date("0001-01-01T00:00:00+05:00") is None

    def test_non_str_never_raises(self):
        for value in [None, 0, 1234567890, 3.14, [], {}, object()]:
            last_seen_date(value)

    def test_date_only_string(self):
        """A bare ``YYYY-MM-DD`` (no time component) is valid ISO 8601."""
        assert last_seen_date("2026-08-06") == date(2026, 8, 6)

    def test_aware_non_utc_offset(self):
        raw = "2026-08-06T01:30:00+05:00"
        expected = (
            datetime(2026, 8, 6, 1, 30, 0, tzinfo=timezone(timedelta(hours=5))).astimezone().date()
        )
        assert last_seen_date(raw) == expected


class TestDateWindow:
    def test_contains_inclusive_start_boundary(self):
        window = DateWindow(start=date(2026, 8, 1), end=date(2026, 8, 7))
        assert window.contains(date(2026, 8, 1)) is True

    def test_contains_inclusive_end_boundary(self):
        window = DateWindow(start=date(2026, 8, 1), end=date(2026, 8, 7))
        assert window.contains(date(2026, 8, 7)) is True

    def test_contains_inside_range(self):
        window = DateWindow(start=date(2026, 8, 1), end=date(2026, 8, 7))
        assert window.contains(date(2026, 8, 4)) is True

    def test_contains_before_start(self):
        window = DateWindow(start=date(2026, 8, 1), end=date(2026, 8, 7))
        assert window.contains(date(2026, 7, 31)) is False

    def test_contains_after_end(self):
        window = DateWindow(start=date(2026, 8, 1), end=date(2026, 8, 7))
        assert window.contains(date(2026, 8, 8)) is False

    def test_single_day_window(self):
        window = DateWindow(start=date(2026, 8, 5), end=date(2026, 8, 5))
        assert window.contains(date(2026, 8, 5)) is True
        assert window.contains(date(2026, 8, 4)) is False
        assert window.contains(date(2026, 8, 6)) is False

    def test_frozen(self):
        window = DateWindow(start=date(2026, 8, 1), end=date(2026, 8, 7))
        with pytest.raises(AttributeError):
            window.start = date(2026, 8, 2)


class TestBuildDateByKey:
    def test_mixed_dated_and_undated(self):
        last_seen_by_key = {
            "graph1": "2026-08-05T10:00:00",
            "graph2": "2026-08-04T10:00:00",
            "graph3": "",  # undated: no bookkeeping timestamp
            "graph4": None,  # undated: disk-splatted null
        }
        date_by_key = build_date_by_key(last_seen_by_key)

        assert date_by_key == {
            "graph1": date(2026, 8, 5),
            "graph2": date(2026, 8, 4),
            "graph3": None,
            "graph4": None,
        }

    def test_unparseable_values_map_to_none(self):
        """Decision: unparseable maps to ``None`` — the same value a
        genuinely-absent ``last_seen`` produces. Both are equally "no
        usable date" and a caller must not be able to tell them apart."""
        last_seen_by_key = {
            "graph1": "garbage",
            "graph2": 1234567890,  # disk-splatted int, not a string
            "graph3": "2026-08-05T10:00:00",
        }
        date_by_key = build_date_by_key(last_seen_by_key)

        assert date_by_key == {
            "graph1": None,
            "graph2": None,
            "graph3": date(2026, 8, 5),
        }

    def test_multiple_keys_same_date_both_parsed(self):
        last_seen_by_key = {
            "graph1": "2026-08-05T09:00:00",
            "graph2": "2026-08-05T20:00:00",
        }
        date_by_key = build_date_by_key(last_seen_by_key)

        assert date_by_key == {"graph1": date(2026, 8, 5), "graph2": date(2026, 8, 5)}

    def test_preserves_input_key_order(self):
        """One dict comprehension, one pass — the returned mapping's key
        order matches the input's, not a date-sorted order (there is no
        sorting left to do; that was the old two-pass grouping's job)."""
        last_seen_by_key = {
            "graph_c": "2026-08-06T00:00:00",
            "graph_a": "2026-08-01T00:00:00",
            "graph_b": "2026-08-03T00:00:00",
        }
        date_by_key = build_date_by_key(last_seen_by_key)

        assert list(date_by_key.keys()) == ["graph_c", "graph_a", "graph_b"]

    def test_empty_mapping(self):
        assert build_date_by_key({}) == {}

    def test_never_raises_on_hostile_values(self):
        last_seen_by_key = {
            "graph1": None,
            "graph2": 42,
            "graph3": 3.14,
            "graph4": [],
            "graph5": {},
            "graph6": "",
            "graph7": "totally not a date",
        }
        date_by_key = build_date_by_key(last_seen_by_key)
        assert date_by_key == dict.fromkeys(last_seen_by_key, None)
