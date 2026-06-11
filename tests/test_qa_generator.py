"""Tests for QA pair generation from graph relations.

These tests use the template fallback (no model/tokenizer provided).
LLM-based generation is validated by GPU experiments.
"""

from paramem.graph.qa_generator import (
    _flatten_entity_attributes,
    filter_procedural_relations,
    generate_qa_from_graph,
    generate_qa_from_relations,
    partition_relations,
)
from paramem.graph.schema import Entity, Relation, SessionGraph


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
        assert qa[0]["predicate"] == "prefers"
        assert qa[0]["subject"] == "Alex"
        assert qa[0]["object"] == "Python"

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


class TestFilterProceduralRelations:
    def test_preference_relation_type(self):
        rels = [
            {
                "subject": "Alex",
                "predicate": "enjoys",
                "object": "jazz",
                "relation_type": "preference",
            },
        ]
        result = filter_procedural_relations(rels)
        assert len(result) == 1

    def test_factual_relation_excluded(self):
        rels = [
            {
                "subject": "Alex",
                "predicate": "lives_in",
                "object": "Millfield",
                "relation_type": "factual",
            },
        ]
        result = filter_procedural_relations(rels)
        assert len(result) == 0

    def test_predicate_whitelist_fallback(self):
        rels = [
            {
                "subject": "Alex",
                "predicate": "likes",
                "object": "coffee",
                "relation_type": "factual",
            },
        ]
        result = filter_procedural_relations(rels)
        assert len(result) == 1

    def test_mixed_relations(self):
        rels = [
            {
                "subject": "Alex",
                "predicate": "lives_in",
                "object": "Millfield",
                "relation_type": "factual",
            },
            {
                "subject": "Alex",
                "predicate": "prefers",
                "object": "jazz",
                "relation_type": "preference",
            },
            {
                "subject": "Alex",
                "predicate": "works_at",
                "object": "SAP",
                "relation_type": "factual",
            },
            {
                "subject": "Alex",
                "predicate": "drinks",
                "object": "coffee",
                "relation_type": "factual",
            },
        ]
        result = filter_procedural_relations(rels)
        assert len(result) == 2
        predicates = {r["predicate"] for r in result}
        assert predicates == {"prefers", "drinks"}

    def test_empty_input(self):
        assert filter_procedural_relations([]) == []

    def test_novel_preference_predicate(self):
        rels = [
            {
                "subject": "Alex",
                "predicate": "enjoys_cooking",
                "object": "Italian",
                "relation_type": "preference",
            },
        ]
        result = filter_procedural_relations(rels)
        assert len(result) == 1

    def test_missing_relation_type(self):
        rels = [
            {"subject": "Alex", "predicate": "likes", "object": "jazz"},
        ]
        result = filter_procedural_relations(rels)
        assert len(result) == 1


class TestPartitionRelations:
    def _sample(self):
        return [
            {
                "subject": "Alex",
                "predicate": "lives_in",
                "object": "Millfield",
                "relation_type": "factual",
            },
            {
                "subject": "Alex",
                "predicate": "prefers",
                "object": "jazz",
                "relation_type": "preference",
            },
            {
                "subject": "Alex",
                "predicate": "works_at",
                "object": "SAP",
                "relation_type": "factual",
            },
        ]

    def test_procedural_enabled_splits_preferences_out(self):
        episodic, procedural = partition_relations(self._sample(), procedural_enabled=True)
        assert {r["predicate"] for r in episodic} == {"lives_in", "works_at"}
        assert {r["predicate"] for r in procedural} == {"prefers"}

    def test_procedural_disabled_keeps_all_in_episodic(self):
        episodic, procedural = partition_relations(self._sample(), procedural_enabled=False)
        assert len(episodic) == 3
        assert procedural == []

    def test_empty_input(self):
        assert partition_relations([], procedural_enabled=True) == ([], [])
        assert partition_relations([], procedural_enabled=False) == ([], [])


class TestFlattenEntityAttributes:
    """Unit tests for the private ``_flatten_entity_attributes`` projection.

    Tests the projection in isolation; the integration with relations and
    partition routing is exercised by ``TestGenerateQaFromGraph`` below.
    All tests are CPU-only — no model or tokenizer required.
    """

    def test_empty_entity_list_returns_empty(self):
        """Empty input produces an empty output list."""
        result = _flatten_entity_attributes([])
        assert result == []

    def test_single_entity_multiple_attributes(self):
        """Each attribute on a single entity becomes one synthetic relation dict."""
        entity = Entity(
            name="Alex",
            entity_type="person",
            attributes={"email": "alex@example.com", "phone": "+49123456"},
        )
        result = _flatten_entity_attributes([entity])
        assert len(result) == 2
        predicates = {r["predicate"] for r in result}
        assert predicates == {"has_email", "has_phone"}
        for r in result:
            assert r["subject"] == "Alex"
            assert r["relation_type"] == "attribute"

    def test_predicate_form_is_has_plus_key(self):
        """Predicate is exactly 'has_<normalised_key>'."""
        entity = Entity(
            name="Sam",
            entity_type="person",
            attributes={"email": "sam@example.com"},
        )
        result = _flatten_entity_attributes([entity])
        assert result[0]["predicate"] == "has_email"
        assert result[0]["object"] == "sam@example.com"

    def test_multiple_entities_preserve_order(self):
        """Relations for entity A appear before relations for entity B."""
        entity_a = Entity(name="Alice", entity_type="person", attributes={"email": "a@a.com"})
        entity_b = Entity(name="Bob", entity_type="person", attributes={"email": "b@b.com"})
        result = _flatten_entity_attributes([entity_a, entity_b])
        assert len(result) == 2
        assert result[0]["subject"] == "Alice"
        assert result[1]["subject"] == "Bob"

    def test_exclude_pairs_skips_matching_pair(self):
        """A pair whose (subject, predicate) is in exclude_pairs is omitted."""
        entity = Entity(
            name="Alex",
            entity_type="person",
            attributes={"email": "alex@example.com", "phone": "+49123456"},
        )
        exclude = {("Alex", "has_email")}
        result = _flatten_entity_attributes([entity], exclude_pairs=exclude)
        assert len(result) == 1
        assert result[0]["predicate"] == "has_phone"

    def test_exclude_pairs_none_value_skips_nothing_extra(self):
        """When exclude_pairs is None the default is an empty set (nothing excluded)."""
        entity = Entity(
            name="Alex",
            entity_type="person",
            attributes={"email": "alex@example.com"},
        )
        result = _flatten_entity_attributes([entity], exclude_pairs=None)
        assert len(result) == 1

    def test_none_attribute_value_is_skipped(self):
        """Attributes whose value is None are silently omitted.

        Pydantic enforces dict[str, str] at validation time, so None values
        cannot be constructed via the normal constructor.  model_construct
        bypasses validation to exercise the defensive guard in
        flatten_entity_attributes — important because callers that build
        entities from raw dicts (e.g. deserialization with a lax loader) may
        inject None before Pydantic can reject it.
        """
        entity = Entity.model_construct(
            name="Alex",
            entity_type="person",
            attributes={"email": None, "phone": "+49123456"},
        )
        result = _flatten_entity_attributes([entity])
        assert len(result) == 1
        assert result[0]["predicate"] == "has_phone"

    def test_whitespace_only_attribute_value_is_skipped(self):
        """Attributes that reduce to an empty string after strip() are omitted."""
        entity = Entity(
            name="Alex",
            entity_type="person",
            attributes={"email": "   ", "phone": "+49123456"},
        )
        result = _flatten_entity_attributes([entity])
        assert len(result) == 1
        assert result[0]["predicate"] == "has_phone"

    def test_key_with_spaces_normalised_to_canonical(self):
        """Attribute keys with spaces are canonicalized (space-separated) in the predicate."""
        entity = Entity(
            name="Alex",
            entity_type="person",
            attributes={"phone number": "+49123456"},
        )
        result = _flatten_entity_attributes([entity])
        assert result[0]["predicate"] == "has_phone number"

    def test_key_with_dashes_normalised_to_canonical(self):
        """Attribute keys with dashes are canonicalized (separator-folded to spaces)."""
        entity = Entity(
            name="Alex",
            entity_type="person",
            attributes={"linked-in": "linkedin.com/in/alex"},
        )
        result = _flatten_entity_attributes([entity])
        assert result[0]["predicate"] == "has_linked in"

    def test_key_with_uppercase_lowercased(self):
        """Attribute keys are lowercased before formatting the predicate."""
        entity = Entity(
            name="Alex",
            entity_type="person",
            attributes={"Email": "alex@example.com"},
        )
        result = _flatten_entity_attributes([entity])
        assert result[0]["predicate"] == "has_email"

    def test_entity_with_no_attributes_produces_no_relations(self):
        """An entity with an empty attributes dict contributes nothing."""
        entity = Entity(name="Bob", entity_type="person", attributes={})
        result = _flatten_entity_attributes([entity])
        assert result == []

    def test_input_entities_not_mutated(self):
        """The function must not modify the input entity objects."""
        entity = Entity(
            name="Alex",
            entity_type="person",
            attributes={"email": "alex@example.com"},
        )
        original_attrs = dict(entity.attributes)
        _flatten_entity_attributes([entity])
        assert entity.attributes == original_attrs


class TestGenerateQaFromGraph:
    """Integration tests for ``generate_qa_from_graph``.

    Exercises the unified graph → keyed-QA-pair entry point: relations and
    entity attributes flow through the same pipeline (projection → partition
    → QA generation).  No model is provided, so QA generation falls through
    to the deterministic template path.
    """

    def _graph(
        self,
        *,
        relations: list[Relation] | None = None,
        entities: list[Entity] | None = None,
    ) -> SessionGraph:
        return SessionGraph(
            session_id="sg-graph-test",
            timestamp="2026-01-01T00:00:00Z",
            entities=entities or [],
            relations=relations or [],
        )

    def test_relations_only_graph_yields_only_relation_qa(self):
        """No entity attributes → no projected pairs, just relation-derived QA."""
        graph = self._graph(
            relations=[
                Relation(
                    subject="Alex",
                    predicate="lives_in",
                    object="Heilbronn",
                    relation_type="factual",
                    speaker_id="Speaker0",
                )
            ],
        )
        episodic_rels, procedural_rels = generate_qa_from_graph(graph, procedural_enabled=False)
        assert procedural_rels == []
        assert len(episodic_rels) == 1
        assert episodic_rels[0]["predicate"] == "lives_in"
        assert episodic_rels[0]["subject"] == "Alex"

    def test_attributes_alone_yield_projected_qa(self):
        """Entity with attributes but no relations still produces keyed QA."""
        graph = self._graph(
            entities=[
                Entity(
                    name="Alex",
                    entity_type="person",
                    attributes={"email": "alex@example.com"},
                )
            ],
        )
        episodic_rels, procedural_rels = generate_qa_from_graph(graph, procedural_enabled=False)
        assert procedural_rels == []
        assert len(episodic_rels) == 1
        assert episodic_rels[0]["predicate"] == "has_email"
        assert episodic_rels[0]["object"] == "alex@example.com"

    def test_relations_and_attributes_unified(self):
        """Both surfaces feed the same pipeline; output set contains both."""
        graph = self._graph(
            relations=[
                Relation(
                    subject="Alex",
                    predicate="lives_in",
                    object="Heilbronn",
                    relation_type="factual",
                    speaker_id="Speaker0",
                )
            ],
            entities=[
                Entity(
                    name="Alex",
                    entity_type="person",
                    attributes={"email": "alex@example.com"},
                )
            ],
        )
        episodic_rels, procedural_rels = generate_qa_from_graph(graph, procedural_enabled=False)
        predicates = {qa["predicate"] for qa in episodic_rels}
        assert predicates == {"lives_in", "has_email"}
        assert procedural_rels == []

    def test_existing_has_predicate_suppresses_attribute_projection(self):
        """If the extractor already emitted (subject, has_<key>) the projection skips."""
        graph = self._graph(
            relations=[
                Relation(
                    subject="Alex",
                    predicate="has_email",
                    object="alex@example.com",
                    relation_type="factual",
                    speaker_id="Speaker0",
                )
            ],
            entities=[
                Entity(
                    name="Alex",
                    entity_type="person",
                    attributes={"email": "alex@example.com"},
                )
            ],
        )
        episodic_rels, _ = generate_qa_from_graph(graph, procedural_enabled=False)
        assert len(episodic_rels) == 1  # the explicit relation only — no duplicate

    def test_preference_relation_routes_to_procedural(self):
        """relation_type='preference' goes to procedural slot when enabled."""
        graph = self._graph(
            relations=[
                Relation(
                    subject="Alex",
                    predicate="prefers",
                    object="dark_mode",
                    relation_type="preference",
                    speaker_id="Speaker0",
                ),
                Relation(
                    subject="Alex",
                    predicate="lives_in",
                    object="Heilbronn",
                    relation_type="factual",
                    speaker_id="Speaker0",
                ),
            ],
        )
        episodic_rels, procedural_rels = generate_qa_from_graph(graph, procedural_enabled=True)
        ep_preds = {qa["predicate"] for qa in episodic_rels}
        proc_preds = {r["predicate"] for r in procedural_rels}
        assert ep_preds == {"lives_in"}
        assert proc_preds == {"prefers"}

    def test_attribute_relation_type_routes_to_episodic(self):
        """Projected attributes carry relation_type='attribute', stay episodic."""
        graph = self._graph(
            entities=[
                Entity(
                    name="Alex",
                    entity_type="person",
                    attributes={"email": "alex@example.com"},
                )
            ],
        )
        episodic_rels, procedural_rels = generate_qa_from_graph(graph, procedural_enabled=True)
        assert len(episodic_rels) == 1
        assert procedural_rels == []
