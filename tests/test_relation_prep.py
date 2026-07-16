"""Tests for paramem.graph.relation_prep.

Covers the three format-neutral relation/entity preparation helpers consumed
by the indexed-key distillation path
(``ConsolidationLoop._entries_from_graph``): ``filter_procedural_relations``,
``partition_relations``, and ``_flatten_entity_attributes``.  No LLM calls —
all tests are CPU-only.
"""

from __future__ import annotations

from paramem.graph.relation_prep import (
    _PROCEDURAL_PREDICATES,
    _flatten_entity_attributes,
    filter_procedural_relations,
    partition_relations,
)
from paramem.graph.schema import Entity


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

    def test_procedural_predicates_is_frozenset(self):
        assert isinstance(_PROCEDURAL_PREDICATES, frozenset)
        assert "prefers" in _PROCEDURAL_PREDICATES


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
