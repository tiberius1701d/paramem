"""Tests for `paramem/evaluation/key_fidelity.py`."""

from __future__ import annotations

from paramem.evaluation.key_fidelity import (
    _match_verb_prefix,
    _normalize_triple,
    _parse_age_remainder,
    _split_sentences,
    _strip_temporal_prefix,
    measure_fidelity,
    parse_profile_to_triples,
)


def test_normalize_triple_lowercases_and_strips():
    t = {"subject": " Alex ", "predicate": "Lives In", "object": " Paris "}
    s, p, o = _normalize_triple(t)
    assert s == "alex"
    assert o == "paris"
    # Predicate normalization is the shared _normalize_predicate (lowercase + _)
    assert " " not in p
    assert p.islower()


def test_measure_fidelity_perfect_match():
    triples = [{"subject": "A", "predicate": "knows", "object": "B"}]
    m = measure_fidelity(triples, triples)
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0


def test_measure_fidelity_empty_reconstruction():
    m = measure_fidelity([{"subject": "A", "predicate": "knows", "object": "B"}], [])
    assert m["recall"] == 0.0


def test_measure_fidelity_partial_match():
    originals = [
        {"subject": "A", "predicate": "knows", "object": "B"},
        {"subject": "A", "predicate": "likes", "object": "C"},
    ]
    reconstructed = [
        {"subject": "A", "predicate": "knows", "object": "B"},
        {"subject": "A", "predicate": "invented", "object": "D"},
    ]
    m = measure_fidelity(originals, reconstructed)
    assert 0 < m["precision"] < 1.0
    assert 0 < m["recall"] < 1.0


def test_measure_fidelity_case_insensitive():
    originals = [{"subject": "Alex", "predicate": "KNOWS", "object": "Bob"}]
    reconstructed = [{"subject": "alex", "predicate": "knows", "object": "BOB"}]
    m = measure_fidelity(originals, reconstructed)
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0


# ---------------------------------------------------------------------------
# Profile-parsing helpers — replaced from regex to structural in this commit.
# ---------------------------------------------------------------------------


class TestSplitSentences:
    def test_terminator_with_whitespace_splits(self):
        assert _split_sentences("First. Second! Third?") == [
            "First.",
            "Second!",
            "Third?",
        ]

    def test_terminator_without_whitespace_does_not_split(self):
        assert _split_sentences("First.Second.") == ["First.Second."]

    def test_empty_input_returns_empty(self):
        assert _split_sentences("") == []

    def test_strips_blank_segments(self):
        assert _split_sentences("Hello.   World.") == ["Hello.", "World."]


class TestStripTemporalPrefix:
    def test_strips_as_of_prefix(self):
        assert (
            _strip_temporal_prefix("As of 2026, Alex lives in Kelkham") == "Alex lives in Kelkham"
        )

    def test_strips_in_prefix(self):
        assert _strip_temporal_prefix("In 2025, Alex moved") == "Alex moved"

    def test_strips_on_prefix(self):
        assert _strip_temporal_prefix("On Monday, Alex called") == "Alex called"

    def test_no_prefix_returns_unchanged(self):
        assert _strip_temporal_prefix("Alex works at the office") == "Alex works at the office"

    def test_temporal_word_without_comma_returns_unchanged(self):
        assert _strip_temporal_prefix("In Berlin Alex lives") == "In Berlin Alex lives"


class TestParseAgeRemainder:
    def test_is_digit_returns_object(self):
        assert _parse_age_remainder("is 30 years old") == "30 years old"

    def test_is_non_digit_returns_none(self):
        assert _parse_age_remainder("is a teacher") is None

    def test_does_not_start_with_is(self):
        assert _parse_age_remainder("works at home") is None

    def test_strips_trailing_period(self):
        assert _parse_age_remainder("is 30 years old.") == "30 years old"


class TestMatchVerbPrefix:
    def test_lives_in_matches(self):
        assert _match_verb_prefix("lives in Kelkham") == ("lives_in", "Kelkham")

    def test_works_as_preferred_over_works(self):
        # Longest verb prefix wins — "works as" must match before "works"
        # alone (which isn't even in the table, but the principle stands).
        assert _match_verb_prefix("works as a teacher") == ("works_as", "a teacher")

    def test_works_at_distinct_from_works_as(self):
        assert _match_verb_prefix("works at the office") == ("works_at", "the office")

    def test_collaborates_with_full_phrase(self):
        result = _match_verb_prefix("collaborates with Pat")
        assert result == ("collaborates_with", "Pat")

    def test_case_insensitive_prefix_preserves_object_casing(self):
        assert _match_verb_prefix("LIKES BEACH HOLIDAYS") == ("likes", "BEACH HOLIDAYS")

    def test_no_match_returns_none(self):
        assert _match_verb_prefix("invented quantum entanglement") is None


class TestParseProfileToTriples:
    def test_extracts_lives_in(self):
        triples = parse_profile_to_triples("Alex", "Alex lives in Kelkham.")
        assert triples == [{"subject": "Alex", "predicate": "lives_in", "object": "Kelkham"}]

    def test_extracts_age_via_is_pattern(self):
        triples = parse_profile_to_triples("Alex", "Alex is 50 years old.")
        assert triples == [{"subject": "Alex", "predicate": "has_age", "object": "50 years old"}]

    def test_skips_temporal_prefix(self):
        triples = parse_profile_to_triples("Alex", "As of 2026, Alex lives in Kelkham.")
        assert triples == [{"subject": "Alex", "predicate": "lives_in", "object": "Kelkham"}]

    def test_handles_multiple_sentences(self):
        triples = parse_profile_to_triples(
            "Alex",
            "Alex lives in Kelkham. Alex works at the office.",
        )
        assert {(t["predicate"], t["object"]) for t in triples} == {
            ("lives_in", "Kelkham"),
            ("works_at", "the office"),
        }

    def test_skips_sentence_not_starting_with_entity(self):
        triples = parse_profile_to_triples("Alex", "The weather is nice.")
        assert triples == []
