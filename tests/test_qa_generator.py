"""Tests for QA pair generation from graph relations."""

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
        assert len(qa) == 2  # subject + reverse template
        assert "Heilbronn" in qa[0]["answer"]
        assert "Alex" in qa[0]["question"]

    def test_reverse_template(self):
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
        questions = [q["question"] for q in qa]
        assert "Where does Alex live?" in questions
        assert "Who lives in Heilbronn?" in questions

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
        assert len(qa) == 1  # has_pet has no reverse

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
        assert len(qa) == 4  # 2 subject + 2 reverse

    def test_unknown_predicate_uses_generic(self):
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
        qa = generate_qa_from_relations(relations, use_llm=False)
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
        # Predicates with reverse templates produce 2 QA pairs
        reversible = {"lives_in", "works_at", "works_as", "studies_at", "knows", "manages"}
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
            expected = 2 if pred in reversible else 1
            assert len(qa) == expected, f"Expected {expected} QA for '{pred}', got {len(qa)}"
            assert qa[0]["question"], f"Empty question for predicate '{pred}'"
            assert qa[0]["answer"], f"Empty answer for predicate '{pred}'"
