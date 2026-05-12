"""Smoke tests for paramem.graph.relation_prep.

Verifies that the new module exports the four names correctly and that
importing them from paramem.graph.qa_generator (the legacy public surface)
yields the identical objects — no shadow copies, no behaviour divergence.

Comprehensive behavioural tests for all four symbols live in
tests/test_qa_generator.py (TestFilterProceduralRelations,
TestPartitionRelations, TestFlattenEntityAttributes).  This file does not
duplicate those; it only guards the refactor contract.
"""

from __future__ import annotations

import paramem.graph.qa_generator as qa_mod
import paramem.graph.relation_prep as rp_mod


class TestReExportIdentity:
    """The four symbols imported via qa_generator must be the same objects as
    those defined in relation_prep — not copies."""

    def test_partition_relations_same_object(self):
        assert qa_mod.partition_relations is rp_mod.partition_relations

    def test_filter_procedural_relations_same_object(self):
        assert qa_mod.filter_procedural_relations is rp_mod.filter_procedural_relations

    def test_flatten_entity_attributes_same_object(self):
        assert qa_mod._flatten_entity_attributes is rp_mod._flatten_entity_attributes

    def test_procedural_predicates_same_object(self):
        assert qa_mod._PROCEDURAL_PREDICATES is rp_mod._PROCEDURAL_PREDICATES


class TestRelationPrepSmoke:
    """Minimal behavioural checks that the new module works in isolation."""

    def test_partition_disabled_returns_all_episodic(self):
        rels = [
            {"predicate": "lives_in", "relation_type": "factual"},
            {"predicate": "prefers", "relation_type": "preference"},
        ]
        episodic, procedural = rp_mod.partition_relations(rels, procedural_enabled=False)
        assert len(episodic) == 2
        assert procedural == []

    def test_partition_enabled_splits(self):
        rels = [
            {"predicate": "lives_in", "relation_type": "factual"},
            {"predicate": "prefers", "relation_type": "preference"},
        ]
        episodic, procedural = rp_mod.partition_relations(rels, procedural_enabled=True)
        assert len(episodic) == 1
        assert len(procedural) == 1
        assert episodic[0]["predicate"] == "lives_in"
        assert procedural[0]["predicate"] == "prefers"

    def test_filter_procedural_by_relation_type(self):
        rels = [{"predicate": "enjoys", "relation_type": "preference"}]
        assert len(rp_mod.filter_procedural_relations(rels)) == 1

    def test_filter_procedural_by_predicate_set(self):
        rels = [{"predicate": "likes", "relation_type": "factual"}]
        assert len(rp_mod.filter_procedural_relations(rels)) == 1

    def test_procedural_predicates_is_frozenset(self):
        assert isinstance(rp_mod._PROCEDURAL_PREDICATES, frozenset)
        assert "prefers" in rp_mod._PROCEDURAL_PREDICATES
