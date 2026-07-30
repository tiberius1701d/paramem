"""Unit tests for GraphMerger's ``relation_type == "attribute"`` gate.

Covers the merger-gate authority relocation (Unit 4): a relation the model
tags ``relation_type="attribute"`` folds onto the SUBJECT node's
``attributes`` dict instead of becoming an edge to a (potentially
colliding) concept node. Pure graph-state assertions — no model, no
tokenizer, no cloud.
"""

from __future__ import annotations

from paramem.graph.merger import GraphMerger, _strip_has_prefix
from paramem.graph.schema import Entity, Relation, SessionGraph


def _session(
    *relations: Relation, entities: list[Entity] | None = None, session_id="s0"
) -> SessionGraph:
    return SessionGraph(
        session_id=session_id,
        timestamp="2026-07-24T00:00:00Z",
        entities=list(entities or []),
        relations=list(relations),
    )


def _attr_relation(
    subject="speaker0",
    predicate="has_email",
    obj="alex@example.com",
    speaker_id="speaker0",
    indexed_key: str | None = None,
) -> Relation:
    return Relation(
        subject=subject,
        predicate=predicate,
        object=obj,
        relation_type="attribute",
        confidence=1.0,
        speaker_id=speaker_id,
        indexed_key=indexed_key,
    )


class TestStripHasPrefix:
    def test_strips_underscore_prefix(self):
        assert _strip_has_prefix("has_email") == "email"

    def test_strips_space_prefix(self):
        assert _strip_has_prefix("has certification") == "certification"

    def test_strips_only_one_occurrence(self):
        """A doubled prefix degrades to a single strip, never both."""
        assert _strip_has_prefix("has_has_email") == "has_email"

    def test_no_prefix_passthrough(self):
        assert _strip_has_prefix("works_at") == "works_at"

    def test_bare_has_word_not_stripped(self):
        """'has' alone (no trailing separator) is not a prefix match."""
        assert _strip_has_prefix("has") == "has"


class TestAttributeGateFoldsOntoNode:
    def test_attribute_relation_becomes_node_attribute(self):
        merger = GraphMerger()
        merger.merge(_session(_attr_relation()))
        node = merger.graph.nodes["speaker0"]
        assert node["attributes"]["email"] == "alex@example.com"

    def test_no_edge_created(self):
        merger = GraphMerger()
        merger.merge(_session(_attr_relation()))
        assert merger.graph.number_of_edges() == 0

    def test_no_object_node_created(self):
        """Only the subject node exists — the object value never mints a node."""
        merger = GraphMerger()
        merger.merge(_session(_attr_relation(obj="alex@example.com")))
        assert merger.graph.number_of_nodes() == 1
        assert "speaker0" in merger.graph
        assert "alex@example.com" not in merger.graph

    def test_value_stored_verbatim_case_preserved(self):
        """object canonicalization is mode='spaces' — case/diacritics survive."""
        merger = GraphMerger()
        merger.merge(_session(_attr_relation(obj="+1 555 123 4567")))
        node = merger.graph.nodes["speaker0"]
        assert node["attributes"]["email"] == "+1 555 123 4567"

    def test_value_blank_runs_folded_to_space(self):
        """object canonicalization is mode='spaces': underscore/blank runs
        collapse to a single space, case is preserved."""
        merger = GraphMerger()
        merger.merge(_session(_attr_relation(predicate="has_title", obj="Lead_Systems  Engineer")))
        node = merger.graph.nodes["speaker0"]
        assert node["attributes"]["title"] == "Lead Systems Engineer"

    def test_predicate_without_has_prefix_still_canonicalized(self):
        """Works_At -> attribute key 'works at' (general canonicalization path,
        not limited to has_-prefixed predicates). The OBJECT keeps its case
        (mode='spaces' does not casefold)."""
        merger = GraphMerger()
        merger.merge(_session(_attr_relation(predicate="Works_At", obj="Acme")))
        node = merger.graph.nodes["speaker0"]
        assert node["attributes"]["works at"] == "Acme"

    def test_has_prefix_stripped_once_no_has_has(self):
        merger = GraphMerger()
        merger.merge(_session(_attr_relation(predicate="has_has_email")))
        node = merger.graph.nodes["speaker0"]
        # canonical() space-folds the underscore left after a single strip.
        assert node["attributes"]["has email"] == "alex@example.com"

    def test_subject_node_display_name_seeded_from_surface(self):
        merger = GraphMerger()
        merger.merge(_session(_attr_relation(subject="Alex Morgan", obj="x@y.com")))
        node_key = "alex morgan"
        assert node_key in merger.graph
        assert merger.graph.nodes[node_key]["attributes"]["name"] == "Alex Morgan"


class TestAttributeGateNoCollision:
    def test_two_subjects_same_value_stay_separate_nodes(self):
        """The whole point of the relocation: two subjects holding the SAME
        literal value never collapse onto one shared concept node."""
        merger = GraphMerger()
        merger.merge(
            _session(
                _attr_relation(
                    subject="speaker0", predicate="has_certification", obj="SAE Level 3"
                ),
                _attr_relation(
                    subject="speaker1", predicate="has_certification", obj="SAE Level 3"
                ),
            )
        )
        assert merger.graph.number_of_nodes() == 2
        assert merger.graph.nodes["speaker0"]["attributes"]["certification"] == "SAE Level 3"
        assert merger.graph.nodes["speaker1"]["attributes"]["certification"] == "SAE Level 3"
        assert merger.graph.number_of_edges() == 0


class TestAttributeKeysBookkeeping:
    def test_indexed_key_set_populates_attribute_keys(self):
        merger = GraphMerger()
        merger.merge(_session(_attr_relation(indexed_key="graph7")))
        node = merger.graph.nodes["speaker0"]
        assert node["attribute_keys"]["email"] == "graph7"

    def test_indexed_key_none_leaves_attribute_keys_absent(self):
        merger = GraphMerger()
        merger.merge(_session(_attr_relation(indexed_key=None)))
        node = merger.graph.nodes["speaker0"]
        assert "attribute_keys" not in node or "email" not in node.get("attribute_keys", {})

    def test_attribute_keys_round_trips_through_node_link_serialization(self):
        """attribute_keys is an unknown top-level node field to
        nx.node_link_data and must survive a save/load round trip
        (cumulative-graph / backup-artifact venue)."""
        import networkx as nx

        merger = GraphMerger()
        merger.merge(_session(_attr_relation(indexed_key="graph7")))
        data = nx.node_link_data(merger.graph)
        reloaded = nx.node_link_graph(data, multigraph=True, directed=True)
        assert reloaded.nodes["speaker0"]["attribute_keys"]["email"] == "graph7"


class TestAttributeGateDoesNotReachUpsertRelation:
    def test_no_upsert_relation_side_effects(self):
        """An attribute relation must never touch the Case-1/2/3 machinery
        (collapsed/removal_ledger stay empty)."""
        merger = GraphMerger()
        merger.merge(_session(_attr_relation()))
        assert merger.collapsed == []
        assert merger.removal_ledger == {}

    def test_existing_subject_reinforcement_count_unchanged(self):
        """The subject node-ensure path mirrors the ordinary endpoint-ensure
        loop exactly: an existing node's reinforcement_count is bumped by
        entity merges (_upsert_entity), never by the relation-endpoint
        fallback path — a second attribute fact for an already-existing
        subject must not silently double-bump it."""
        merger = GraphMerger()
        merger.merge(
            _session(
                Relation(
                    subject="speaker0",
                    predicate="lives_in",
                    object="Berlin",
                    relation_type="factual",
                    speaker_id="speaker0",
                )
            )
        )
        merger.merge(_session(_attr_relation(), session_id="s1"))
        node = merger.graph.nodes["speaker0"]
        assert node["reinforcement_count"] == 1
