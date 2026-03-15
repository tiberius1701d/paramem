"""Tests for QA pair generation from graph relations.

These tests use the template fallback (no model/tokenizer provided).
LLM-based generation is validated by GPU experiments.
"""

from paramem.graph.qa_generator import generate_qa_from_relations


class TestTemplateGeneration:
    def test_known_predicate(self):
        relations = [
            {
                "subject": "Alex",
                "predicate": "lives_in",
                "object": "Heilbronn",
                "relation_type": "factual",
                "confidence": 1.0,
                "recurrence_count": 3,
            },
        ]
        qa = generate_qa_from_relations(relations)
        assert len(qa) == 1
        assert "Heilbronn" in qa[0]["answer"]
        assert "Alex" in qa[0]["question"]

    def test_known_predicate_content(self):
        relations = [
            {
                "subject": "Alex",
                "predicate": "lives_in",
                "object": "Heilbronn",
                "relation_type": "factual",
                "confidence": 1.0,
                "recurrence_count": 1,
            },
        ]
        qa = generate_qa_from_relations(relations)
        assert qa[0]["question"] == "Where does Alex live?"
        assert qa[0]["answer"] == "Alex lives in Heilbronn."

    def test_no_reverse_for_non_reversible(self):
        relations = [
            {
                "subject": "Alex",
                "predicate": "has_pet",
                "object": "Luna",
                "relation_type": "factual",
                "confidence": 1.0,
                "recurrence_count": 1,
            },
        ]
        qa = generate_qa_from_relations(relations)
        assert len(qa) == 1

    def test_multiple_relations(self):
        relations = [
            {
                "subject": "Alex",
                "predicate": "lives_in",
                "object": "Heilbronn",
                "relation_type": "factual",
                "confidence": 1.0,
                "recurrence_count": 1,
            },
            {
                "subject": "Alex",
                "predicate": "works_at",
                "object": "AutoMate",
                "relation_type": "factual",
                "confidence": 1.0,
                "recurrence_count": 1,
            },
        ]
        qa = generate_qa_from_relations(relations)
        assert len(qa) == 2  # One QA per relation

    def test_unknown_predicate_uses_fallback(self):
        relations = [
            {
                "subject": "Alex",
                "predicate": "discovered",
                "object": "a new algorithm",
                "relation_type": "factual",
                "confidence": 1.0,
                "recurrence_count": 1,
            },
        ]
        qa = generate_qa_from_relations(relations)
        assert len(qa) == 1
        assert "discovered" in qa[0]["answer"]

    def test_source_metadata_preserved(self):
        relations = [
            {
                "subject": "Alex",
                "predicate": "prefers",
                "object": "Python",
                "relation_type": "preference",
                "confidence": 0.9,
                "recurrence_count": 2,
            },
        ]
        qa = generate_qa_from_relations(relations)
        assert qa[0]["source_predicate"] == "prefers"
        assert qa[0]["source_subject"] == "Alex"
        assert qa[0]["source_object"] == "Python"

    def test_empty_relations(self):
        qa = generate_qa_from_relations([])
        assert qa == []

    def test_all_template_predicates(self):
        """Verify all mapped predicates produce valid QA pairs."""
        predicates = [
            "lives_in",
            "works_at",
            "works_as",
            "has_pet",
            "prefers",
            "studies_at",
            "speaks",
            "knows",
            "has_hobby",
            "manages",
            "has_age",
            "born_on",
            "uses",
            "likes",
            "favorite",
        ]
        for pred in predicates:
            relations = [
                {
                    "subject": "Alex",
                    "predicate": pred,
                    "object": "something",
                    "relation_type": "factual",
                    "confidence": 1.0,
                    "recurrence_count": 1,
                },
            ]
            qa = generate_qa_from_relations(relations)
            assert len(qa) == 1, f"Expected 1 QA for '{pred}', got {len(qa)}"
            assert qa[0]["question"], f"Empty question for predicate '{pred}'"
            assert qa[0]["answer"], f"Empty answer for predicate '{pred}'"
