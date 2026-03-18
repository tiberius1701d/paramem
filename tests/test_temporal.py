"""Unit tests for temporal query detection and registry date filtering."""

import json
import tempfile
from datetime import date
from pathlib import Path

from paramem.server.temporal import detect_temporal_query, filter_registry_by_date


class TestDetectTemporalQuery:
    """Test temporal pattern matching and date resolution."""

    def setup_method(self):
        self.today = date(2026, 3, 18)  # Wednesday

    def test_yesterday(self):
        result = detect_temporal_query("What did we discuss yesterday?", self.today)
        assert result == (date(2026, 3, 17), date(2026, 3, 17))

    def test_today(self):
        result = detect_temporal_query("What did I mention today?", self.today)
        assert result == (date(2026, 3, 18), date(2026, 3, 18))

    def test_last_week(self):
        start, end = detect_temporal_query("What happened last week?", self.today)
        # Last week: Monday Mar 9 to Sunday Mar 15
        assert start == date(2026, 3, 9)
        assert end == date(2026, 3, 15)

    def test_this_week(self):
        start, end = detect_temporal_query("What have we discussed this week?", self.today)
        # This week: Monday Mar 16 to today
        assert start == date(2026, 3, 16)
        assert end == self.today

    def test_n_days_ago(self):
        result = detect_temporal_query("What did I say 3 days ago?", self.today)
        assert result == (date(2026, 3, 15), date(2026, 3, 15))

    def test_n_weeks_ago(self):
        start, end = detect_temporal_query("Something from 2 weeks ago", self.today)
        assert start == date(2026, 3, 4)
        assert end == date(2026, 3, 11)

    def test_on_monday(self):
        result = detect_temporal_query("What did we talk about on Monday?", self.today)
        # Most recent Monday before Wednesday = Mar 16
        assert result == (date(2026, 3, 16), date(2026, 3, 16))

    def test_last_friday(self):
        result = detect_temporal_query("What was discussed last Friday?", self.today)
        # Most recent Friday before Wednesday = Mar 13
        assert result == (date(2026, 3, 13), date(2026, 3, 13))

    def test_this_morning(self):
        result = detect_temporal_query("What did I mention this morning?", self.today)
        assert result == (self.today, self.today)

    def test_recently(self):
        start, end = detect_temporal_query("What did we discuss recently?", self.today)
        assert start == date(2026, 3, 11)
        assert end == self.today

    def test_the_other_day(self):
        start, end = detect_temporal_query("You said something the other day", self.today)
        assert start == date(2026, 3, 15)
        assert end == date(2026, 3, 17)

    def test_last_month(self):
        start, end = detect_temporal_query("What happened last month?", self.today)
        assert start == date(2026, 2, 1)
        assert end == date(2026, 2, 28)

    def test_this_month(self):
        start, end = detect_temporal_query("Anything new this month?", self.today)
        assert start == date(2026, 3, 1)
        assert end == self.today

    def test_no_temporal_reference(self):
        result = detect_temporal_query("What is my favorite restaurant?", self.today)
        assert result is None

    def test_no_temporal_in_general_knowledge(self):
        result = detect_temporal_query("Who wrote Romeo and Juliet?", self.today)
        assert result is None

    def test_case_insensitive(self):
        result = detect_temporal_query("YESTERDAY we discussed something", self.today)
        assert result is not None

    def test_day_before_yesterday(self):
        result = detect_temporal_query(
            "the day before yesterday you mentioned something", self.today
        )
        assert result == (date(2026, 3, 16), date(2026, 3, 16))

    def test_last_month_january_wraps_to_december(self):
        jan_date = date(2026, 1, 15)
        start, end = detect_temporal_query("What did we discuss last month?", jan_date)
        assert start == date(2025, 12, 1)
        assert end == date(2025, 12, 31)


class TestFilterRegistryByDate:
    """Test registry date filtering."""

    def _write_registry(self, registry: dict) -> Path:
        tmpdir = Path(tempfile.mkdtemp())
        path = tmpdir / "registry.json"
        with open(path, "w") as f:
            json.dump(registry, f)
        return path

    def test_filters_by_date_range(self):
        registry = {
            "graph0": {
                "simhash": 123,
                "created_at": "2026-03-15T10:00:00+00:00",
                "last_seen_at": "2026-03-15T10:00:00+00:00",
                "status": "active",
            },
            "graph1": {
                "simhash": 456,
                "created_at": "2026-03-17T10:00:00+00:00",
                "last_seen_at": "2026-03-17T10:00:00+00:00",
                "status": "active",
            },
            "graph2": {
                "simhash": 789,
                "created_at": "2026-03-18T10:00:00+00:00",
                "last_seen_at": "2026-03-18T10:00:00+00:00",
                "status": "active",
            },
        }
        path = self._write_registry(registry)
        keys = filter_registry_by_date(path, date(2026, 3, 17), date(2026, 3, 18))
        assert sorted(keys) == ["graph1", "graph2"]

    def test_excludes_stale_keys(self):
        registry = {
            "graph0": {
                "simhash": 123,
                "created_at": "2026-03-17T10:00:00+00:00",
                "last_seen_at": "2026-03-17T10:00:00+00:00",
                "status": "stale",
            },
        }
        path = self._write_registry(registry)
        keys = filter_registry_by_date(path, date(2026, 3, 17), date(2026, 3, 17))
        assert keys == []

    def test_empty_registry(self):
        path = self._write_registry({})
        keys = filter_registry_by_date(path, date(2026, 3, 17), date(2026, 3, 17))
        assert keys == []

    def test_missing_registry_file(self):
        keys = filter_registry_by_date(
            Path("/nonexistent/registry.json"),
            date(2026, 3, 17),
            date(2026, 3, 17),
        )
        assert keys == []

    def test_single_day_range(self):
        registry = {
            "graph0": {
                "simhash": 123,
                "created_at": "2026-03-17T10:00:00+00:00",
                "last_seen_at": "2026-03-17T10:00:00+00:00",
                "status": "active",
            },
            "graph1": {
                "simhash": 456,
                "created_at": "2026-03-18T10:00:00+00:00",
                "last_seen_at": "2026-03-18T10:00:00+00:00",
                "status": "active",
            },
        }
        path = self._write_registry(registry)
        keys = filter_registry_by_date(path, date(2026, 3, 17), date(2026, 3, 17))
        assert keys == ["graph0"]

    def test_handles_simple_simhash_format(self):
        """Old-style registry with just simhash ints should be skipped gracefully."""
        registry = {
            "graph0": 12345,
        }
        path = self._write_registry(registry)
        keys = filter_registry_by_date(path, date(2026, 3, 17), date(2026, 3, 17))
        assert keys == []
