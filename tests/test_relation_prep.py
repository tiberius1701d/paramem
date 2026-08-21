"""Tests for paramem.graph.relation_prep.

Covers the format-neutral relation/entity preparation helpers:
``filter_procedural_relations``, ``partition_relations``,
``attribute_value_is_empty``, ``strip_has_prefix``, ``attr_predicate``, and
``attribute_relations``.  No LLM calls — all tests are CPU-only.
"""

from __future__ import annotations

from paramem.graph.relation_prep import (
    _PROCEDURAL_PREDICATES,
    attr_predicate,
    attribute_relations,
    attribute_value_is_empty,
    filter_procedural_relations,
    partition_relations,
    strip_has_prefix,
)
from paramem.graph.schema import Entity, Relation, SessionGraph
from paramem.utils.identity import canonical


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

    def test_predicate_set_members_are_canonical(self):
        """The set literals share the comparison's surface-form contract."""
        assert all(canonical(p) == p for p in _PROCEDURAL_PREDICATES)

    def test_space_and_underscore_predicate_are_equivalent(self):
        """The multiplicity fix: extraction may emit either surface, and both
        must route to the procedural adapter."""

        def _rel(pred: str) -> dict:
            return {
                "subject": "Alex",
                "predicate": pred,
                "object": "chess",
                "relation_type": "factual",
            }

        underscore = filter_procedural_relations([_rel("has_hobby")])
        spaced = filter_procedural_relations([_rel("has hobby")])
        assert len(underscore) == 1
        assert len(spaced) == 1

    def test_cased_and_diacritic_predicate_matched(self):
        """canonical() folds case on both sides of the membership test."""
        rels = [
            {
                "subject": "Alex",
                "predicate": "Listens To",
                "object": "jazz",
                "relation_type": "factual",
            },
        ]
        assert len(filter_procedural_relations(rels)) == 1

    def test_hyphen_variant_not_matched(self):
        """``-`` is not a separator: ``has-hobby`` is a DIFFERENT predicate."""
        rels = [
            {
                "subject": "Alex",
                "predicate": "has-hobby",
                "object": "chess",
                "relation_type": "factual",
            },
        ]
        assert len(filter_procedural_relations(rels)) == 0

    def test_missing_predicate_key_does_not_raise(self):
        """A relation dict with no predicate falls through the gate cleanly."""
        assert filter_procedural_relations([{"subject": "A", "object": "B"}]) == []


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


class TestAttributeValueIsEmpty:
    def test_none_is_empty(self):
        assert attribute_value_is_empty(None) is True

    def test_whitespace_only_is_empty(self):
        assert attribute_value_is_empty("   ") is True

    def test_empty_string_is_empty(self):
        assert attribute_value_is_empty("") is True

    def test_placeholder_values_are_empty(self):
        for placeholder in ("N/A", "n/a", "None", "null", "unknown", "UNKNOWN"):
            assert attribute_value_is_empty(placeholder) is True

    def test_real_value_is_not_empty(self):
        assert attribute_value_is_empty("alex@example.com") is False

    def test_non_string_non_none_is_not_empty(self):
        assert attribute_value_is_empty(42) is False


class TestStripHasPrefix:
    def test_no_prefix_unchanged(self):
        assert strip_has_prefix("email") == "email"

    def test_single_underscore_prefix_stripped(self):
        assert strip_has_prefix("has_email") == "email"

    def test_single_space_prefix_stripped(self):
        assert strip_has_prefix("has email") == "email"

    def test_doubled_prefix_reaches_fixed_point(self):
        assert strip_has_prefix("has_has_email") == "email"

    def test_triple_prefix_reaches_fixed_point(self):
        assert strip_has_prefix("has_has_has_email") == "email"

    def test_mixed_separator_doubled_prefix(self):
        assert strip_has_prefix("has has_email") == "email"

    def test_empty_string_unchanged(self):
        assert strip_has_prefix("") == ""


class TestAttrPredicate:
    def test_basic_underscore_key(self):
        assert attr_predicate("last_name") == "has last name"

    def test_already_prefixed_key_not_doubled(self):
        assert attr_predicate("has_last_name") == "has last name"

    def test_doubled_prefix_collapses(self):
        assert attr_predicate("has_has_last_name") == "has last name"

    def test_idempotent_across_spellings(self):
        variants = ("has_last_name", "last name", "Last Name", "has has last name")
        results = {attr_predicate(v) for v in variants}
        assert results == {"has last name"}

    def test_doubled_and_bare_forms_agree(self):
        assert attr_predicate("has_has_last_name") == attr_predicate("last_name") == "has last name"


class TestAttributeRelations:
    def _graph(self, entities, relations=None):
        return SessionGraph(
            session_id="s1",
            timestamp="2026-01-01T00:00:00",
            entities=entities,
            relations=relations or [],
        )

    def test_projects_one_relation_per_nonempty_attribute(self):
        graph = self._graph(
            [
                Entity(
                    name="Alex",
                    entity_type="person",
                    attributes={"email": "a@b", "title": "N/A", "name": ""},
                )
            ]
        )
        result = attribute_relations(graph, speaker_id="speaker0")
        assert len(result) == 1
        rel = result[0]
        assert isinstance(rel, Relation)
        assert rel.subject == "Alex"
        assert rel.predicate == "has email"
        assert rel.object == "a@b"
        assert rel.relation_type == "attribute"
        assert rel.confidence == 1.0
        assert rel.speaker_id == "speaker0"

    def test_no_attributes_yields_nothing(self):
        graph = self._graph([Entity(name="Alex", entity_type="person")])
        assert attribute_relations(graph, speaker_id="speaker0") == []

    def test_skips_pair_already_present_in_relations(self):
        graph = self._graph(
            [Entity(name="Alex", entity_type="person", attributes={"email": "a@b"})],
            relations=[
                Relation(
                    subject="Alex",
                    predicate="has email",
                    object="a@b",
                    relation_type="attribute",
                    speaker_id="speaker0",
                )
            ],
        )
        assert attribute_relations(graph, speaker_id="speaker0") == []

    def test_skips_pair_already_present_under_canonical_comparison(self):
        graph = self._graph(
            [Entity(name="Alex", entity_type="person", attributes={"has_email": "a@b"})],
            relations=[
                Relation(
                    subject="alex",
                    predicate="has_email",
                    object="a@b",
                    relation_type="attribute",
                    speaker_id="speaker0",
                )
            ],
        )
        assert attribute_relations(graph, speaker_id="speaker0") == []

    def test_skips_self_loop(self):
        graph = self._graph(
            [Entity(name="Alex", entity_type="person", attributes={"nickname": "Alex"})]
        )
        assert attribute_relations(graph, speaker_id="speaker0") == []

    def test_skips_self_loop_under_canonical_comparison(self):
        graph = self._graph(
            [Entity(name="Alex Smith", entity_type="person", attributes={"nickname": "alex_smith"})]
        )
        assert attribute_relations(graph, speaker_id="speaker0") == []

    def test_does_not_mutate_input_graph(self):
        entity = Entity(name="Alex", entity_type="person", attributes={"email": "a@b"})
        graph = self._graph([entity])
        before = graph.model_copy(deep=True)
        attribute_relations(graph, speaker_id="speaker0")
        assert graph == before

    def test_multiple_entities_multiple_attributes(self):
        graph = self._graph(
            [
                Entity(
                    name="Alex",
                    entity_type="person",
                    attributes={"email": "a@b", "phone": "555-1234"},
                ),
                Entity(
                    name="Bea",
                    entity_type="person",
                    attributes={"hobby": "chess"},
                ),
            ]
        )
        result = attribute_relations(graph, speaker_id="speaker1")
        pairs = {(r.subject, r.predicate, r.object) for r in result}
        assert pairs == {
            ("Alex", "has email", "a@b"),
            ("Alex", "has phone", "555-1234"),
            ("Bea", "has hobby", "chess"),
        }
        assert all(r.speaker_id == "speaker1" for r in result)
