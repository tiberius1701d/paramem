"""Tests for `paramem/evaluation/key_fidelity.py`."""

from __future__ import annotations

from paramem.evaluation.key_fidelity import _normalize_triple, measure_fidelity


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
