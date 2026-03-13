"""Tests for key-addressable replay."""

import pytest

from paramem.evaluation.key_fidelity import measure_all_fidelity, measure_fidelity
from paramem.training.key_registry import KeyRegistry
from paramem.training.key_replay import (
    format_keyed_training,
    format_triples,
    merge_reconstructions_with_session,
    parse_triples,
    retire_stale_keys,
)


class TestFormatTriples:
    def test_single_triple(self):
        relations = [{"subject": "Alex", "predicate": "lives_in", "object": "Heilbronn"}]
        result = format_triples(relations)
        assert result == "(Alex, lives_in, Heilbronn)"

    def test_multiple_triples(self):
        relations = [
            {"subject": "Alex", "predicate": "lives_in", "object": "Heilbronn"},
            {"subject": "Alex", "predicate": "works_at", "object": "SAP"},
        ]
        result = format_triples(relations)
        lines = result.split("\n")
        assert len(lines) == 2
        assert "(Alex, lives_in, Heilbronn)" in lines[0]

    def test_empty(self):
        assert format_triples([]) == ""


class TestFormatKeyedTraining:
    def test_produces_qa_pair(self):
        relations = [
            {"subject": "Alex", "predicate": "lives_in", "object": "Heilbronn"},
        ]
        result = format_keyed_training("session_001", relations)
        assert len(result) == 1
        assert 'key "session_001"' in result[0]["question"]
        assert '<memory key="session_001">' in result[0]["answer"]
        assert "(Alex, lives_in, Heilbronn)" in result[0]["answer"]

    def test_empty_relations(self):
        assert format_keyed_training("session_001", []) == []


class TestParseTriples:
    def test_clean_format(self):
        text = "(Alex, lives_in, Heilbronn)\n(Alex, works_at, SAP)"
        triples = parse_triples(text)
        assert len(triples) == 2
        assert triples[0] == {"subject": "Alex", "predicate": "lives_in", "object": "Heilbronn"}
        assert triples[1] == {"subject": "Alex", "predicate": "works_at", "object": "SAP"}

    def test_with_whitespace(self):
        text = "( Alex ,  lives_in , Heilbronn )"
        triples = parse_triples(text)
        assert len(triples) == 1
        assert triples[0]["subject"] == "Alex"
        assert triples[0]["object"] == "Heilbronn"

    def test_with_memory_tags(self):
        text = '<memory key="s001">\n(Alex, lives_in, Heilbronn)\n</memory>'
        triples = parse_triples(text)
        assert len(triples) == 1

    def test_normalizes_predicates(self):
        text = "(Alex, Lives In, Heilbronn)"
        triples = parse_triples(text)
        assert triples[0]["predicate"] == "lives_in"

    def test_empty_input(self):
        assert parse_triples("") == []
        assert parse_triples("no triples here") == []

    def test_noisy_output(self):
        text = "Here are the triples:\n(Alex, lives_in, Heilbronn)\nThat's all."
        triples = parse_triples(text)
        assert len(triples) == 1

    def test_rejects_incomplete_triples(self):
        text = "(Alex, lives_in)"
        triples = parse_triples(text)
        assert len(triples) == 0


class TestMergeReconstructions:
    def test_merge_with_new_session(self):
        reconstructions = {
            "session_001": [
                {"subject": "Alex", "predicate": "lives_in", "object": "Heilbronn"},
            ],
        }
        session_relations = [
            {"subject": "Alex", "predicate": "works_at", "object": "SAP"},
        ]
        merged = merge_reconstructions_with_session(
            reconstructions, session_relations, "session_002"
        )
        assert len(merged) == 2
        predicates = {r["predicate"] for r in merged}
        assert "lives_in" in predicates
        assert "works_at" in predicates

    def test_deduplicates(self):
        reconstructions = {
            "session_001": [
                {"subject": "Alex", "predicate": "lives_in", "object": "Heilbronn"},
            ],
        }
        session_relations = [
            {"subject": "Alex", "predicate": "lives_in", "object": "Heilbronn"},
        ]
        merged = merge_reconstructions_with_session(
            reconstructions, session_relations, "session_002"
        )
        assert len(merged) == 1

    def test_case_insensitive_dedup(self):
        reconstructions = {
            "s1": [{"subject": "alex", "predicate": "lives_in", "object": "heilbronn"}],
        }
        session_relations = [
            {"subject": "Alex", "predicate": "Lives_In", "object": "Heilbronn"},
        ]
        merged = merge_reconstructions_with_session(reconstructions, session_relations, "s2")
        assert len(merged) == 1

    def test_empty_reconstructions(self):
        session_relations = [
            {"subject": "Alex", "predicate": "lives_in", "object": "Heilbronn"},
        ]
        merged = merge_reconstructions_with_session({}, session_relations, "s1")
        assert len(merged) == 1

    def test_multiple_keys(self):
        reconstructions = {
            "s1": [{"subject": "Alex", "predicate": "lives_in", "object": "Heilbronn"}],
            "s2": [{"subject": "Bob", "predicate": "works_at", "object": "Google"}],
        }
        session_relations = [
            {"subject": "Clara", "predicate": "studies_at", "object": "MIT"},
        ]
        merged = merge_reconstructions_with_session(reconstructions, session_relations, "s3")
        assert len(merged) == 3


class TestRetireStaleKeys:
    def test_retires_low_fidelity(self):
        reg = KeyRegistry()
        reg.add("good_key")
        reg.add("stale_key")
        reg.update_fidelity("good_key", 0.9)
        reg.update_fidelity("good_key", 0.85)
        reg.update_fidelity("good_key", 0.88)
        reg.update_fidelity("stale_key", 0.05)
        reg.update_fidelity("stale_key", 0.03)
        reg.update_fidelity("stale_key", 0.02)

        retired = retire_stale_keys(reg, threshold=0.1, consecutive_cycles=3)
        assert "stale_key" in retired
        assert "good_key" not in retired
        assert "stale_key" not in reg
        assert "good_key" in reg

    def test_no_retirement_when_all_healthy(self):
        reg = KeyRegistry()
        reg.add("key_a")
        reg.update_fidelity("key_a", 0.9)
        reg.update_fidelity("key_a", 0.85)
        reg.update_fidelity("key_a", 0.88)
        retired = retire_stale_keys(reg)
        assert retired == []


class TestMeasureFidelity:
    def test_perfect_reconstruction(self):
        originals = [
            {"subject": "Alex", "predicate": "lives_in", "object": "Heilbronn"},
            {"subject": "Alex", "predicate": "works_at", "object": "SAP"},
        ]
        result = measure_fidelity(originals, originals)
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0

    def test_partial_reconstruction(self):
        originals = [
            {"subject": "Alex", "predicate": "lives_in", "object": "Heilbronn"},
            {"subject": "Alex", "predicate": "works_at", "object": "SAP"},
        ]
        reconstructed = [
            {"subject": "Alex", "predicate": "lives_in", "object": "Heilbronn"},
        ]
        result = measure_fidelity(originals, reconstructed)
        assert result["precision"] == 1.0
        assert result["recall"] == 0.5
        assert result["matched_count"] == 1

    def test_hallucinated_triple(self):
        originals = [
            {"subject": "Alex", "predicate": "lives_in", "object": "Heilbronn"},
        ]
        reconstructed = [
            {"subject": "Alex", "predicate": "lives_in", "object": "Heilbronn"},
            {"subject": "Alex", "predicate": "likes", "object": "Pizza"},
        ]
        result = measure_fidelity(originals, reconstructed)
        assert result["precision"] == 0.5
        assert result["recall"] == 1.0

    def test_empty_original(self):
        result = measure_fidelity([], [])
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0

    def test_empty_reconstruction(self):
        originals = [
            {"subject": "Alex", "predicate": "lives_in", "object": "Heilbronn"},
        ]
        result = measure_fidelity(originals, [])
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0

    def test_case_insensitive(self):
        originals = [
            {"subject": "Alex", "predicate": "lives_in", "object": "Heilbronn"},
        ]
        reconstructed = [
            {"subject": "alex", "predicate": "Lives In", "object": "heilbronn"},
        ]
        result = measure_fidelity(originals, reconstructed)
        assert result["recall"] == 1.0


class TestMeasureAllFidelity:
    def test_aggregate_metrics(self):
        originals = {
            "s1": [{"subject": "Alex", "predicate": "lives_in", "object": "Heilbronn"}],
            "s2": [{"subject": "Bob", "predicate": "works_at", "object": "Google"}],
        }
        reconstructed = {
            "s1": [{"subject": "Alex", "predicate": "lives_in", "object": "Heilbronn"}],
            "s2": [],  # total miss
        }
        result = measure_all_fidelity(originals, reconstructed)
        assert result["num_keys"] == 2
        assert result["per_key"]["s1"]["f1"] == 1.0
        assert result["per_key"]["s2"]["f1"] == 0.0
        assert result["mean_f1"] == pytest.approx(0.5)

    def test_empty(self):
        result = measure_all_fidelity({}, {})
        assert result["num_keys"] == 0
        assert result["mean_f1"] == 0.0
