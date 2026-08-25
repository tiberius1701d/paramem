"""``format_turn`` / ``split_marker`` round-trips."""

from __future__ import annotations

import pytest

from paramem.utils.turn_markers import format_turn, split_marker


class TestFormatSplitRoundTrip:
    @pytest.mark.parametrize("role", ["user", "assistant", "narrator"])
    def test_round_trips_for_known_and_unknown_roles(self, role: str) -> None:
        line = format_turn(role, "hello there")
        marker, text = split_marker(line)
        assert marker == f"[{role}] "
        assert text == "hello there"
        assert marker + text == line

    def test_format_turn_produces_the_documented_shape(self) -> None:
        assert format_turn("user", "hello") == "[user] hello"


class TestSplitMarkerNoMarkerLine:
    def test_line_with_no_marker_returns_empty_prefix_unchanged(self) -> None:
        marker, text = split_marker("just plain text, no brackets")
        assert marker == ""
        assert text == "just plain text, no brackets"

    def test_brackets_not_at_the_start_are_not_a_marker(self) -> None:
        line = "some text [user] mid-line"
        marker, text = split_marker(line)
        assert marker == ""
        assert text == line

    def test_empty_line_returns_empty_prefix_and_empty_text(self) -> None:
        marker, text = split_marker("")
        assert marker == ""
        assert text == ""
