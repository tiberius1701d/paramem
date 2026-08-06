"""Tests for the _parse_slot_timestamp private helper."""

from __future__ import annotations

from datetime import datetime, timezone

from paramem.backup.backup import _parse_slot_timestamp


class TestParseSlotTimestamp:
    """Unit tests for the _parse_slot_timestamp private helper."""

    def test_parse_slot_timestamp_valid_format(self):
        """A well-formed 17-char slot name returns a UTC datetime, not None."""
        result = _parse_slot_timestamp("20260421-04000012")
        assert result is not None
        assert isinstance(result, datetime)
        assert result.tzinfo is timezone.utc
        assert result.year == 2026
        assert result.month == 4
        assert result.day == 21
        assert result.hour == 4
        assert result.minute == 0
        assert result.second == 0

    def test_parse_slot_timestamp_rejects_18_chars(self):
        """A string of 18 chars (the old off-by-one) is rejected — returns None."""
        result = _parse_slot_timestamp("20260421-040000123")  # 18 chars
        assert result is None

    def test_parse_slot_timestamp_rejects_malformed(self):
        """Non-timestamp strings return None without raising."""
        assert _parse_slot_timestamp("") is None
        assert _parse_slot_timestamp("not-a-date") is None
        assert _parse_slot_timestamp("20260421_04000012") is None  # underscore not dash

    def test_parse_slot_timestamp_rejects_impossible_date(self):
        """Correctly-shaped 17-char name with an impossible date returns None."""
        result = _parse_slot_timestamp("20261345-04000012")  # month 13, day 45
        assert result is None
