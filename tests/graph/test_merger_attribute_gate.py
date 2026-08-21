"""Unit tests for GraphMerger's ``relation_type == "attribute"`` gate.

Covers the merger-gate authority relocation: a relation the model tags
``relation_type="attribute"`` folds onto the SUBJECT node's ``attributes``
dict instead of becoming an edge to a (potentially colliding) concept node,
plus the provenance-bearing node-attribute record: the two merger
primitives (``reconcile_provenance``, ``attribute_fact``) and the rewritten
gate's net-new record shape, same-value reconciliation, and the
value-change-starts-a-new-lifetime rule. Pure graph-state assertions — no
model, no tokenizer, no cloud.
"""

from __future__ import annotations

from paramem.graph.merger import GraphMerger, attribute_fact, node_display, reconcile_provenance
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
    first_seen="",
    last_seen="",
    edge_source="",
) -> Relation:
    return Relation(
        subject=subject,
        predicate=predicate,
        object=obj,
        relation_type="attribute",
        confidence=1.0,
        speaker_id=speaker_id,
        indexed_key=indexed_key,
        first_seen=first_seen,
        last_seen=last_seen,
        edge_source=edge_source,
    )


class TestAttributeGateFoldsOntoNode:
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

    def test_subject_node_display_name_seeded_from_surface(self):
        merger = GraphMerger()
        merger.merge(_session(_attr_relation(subject="Alex Morgan", obj="x@y.com")))
        node_key = "alex morgan"
        assert node_key in merger.graph
        assert merger.graph.nodes[node_key]["display_name"] == "Alex Morgan"


class TestAttributeGateDisplayNameRoundTrip:
    def test_display_name_round_trips_through_node_link_serialization(self):
        """display_name is an unknown top-level node field to
        nx.node_link_data and must survive a save/load round trip — the RAM
        graph is serialised by GraphMerger.save_bytes() into the
        pre-migration backup."""
        import networkx as nx

        merger = GraphMerger()
        merger.merge(_session(_attr_relation(indexed_key="graph7")))
        data = nx.node_link_data(merger.graph)
        reloaded = nx.node_link_graph(data, multigraph=True, directed=True)
        assert reloaded.nodes["speaker0"]["display_name"] == "speaker0"


class TestAttributeKeySupersession:
    """A second attribute relation for the same (subject, attribute) pair
    displaces the incumbent indexed key.  The displaced key must be
    ledgered as a removal (reason='attribute_key_superseded', survivor_key=
    the new key) so it is not silently dropped from the fold's accounting.
    """

    def test_repeated_merge_of_the_same_key_does_not_ledger(self):
        """Re-merging the SAME indexed key onto the same attribute is a no-op
        overwrite, not a supersession — nothing to ledger."""
        merger = GraphMerger()
        merger.merge(_session(_attr_relation(indexed_key="graph7"), session_id="s0"))
        merger.merge(_session(_attr_relation(indexed_key="graph7"), session_id="s1"))
        assert merger.removal_ledger == {}

    def test_same_value_carry_forward_survives_speaker_refresh_between_merges(self):
        """A same-value `has name` carry-forward on a speaker node must still
        be read as a carry-forward (`survivor_key`) even though
        `merge_relations` synthesises a speaker Entity for every call and
        `_upsert_entity`'s speaker refresh runs before the attribute gate
        reads the incumbent value.  Before the display/fact split, the
        refresh clobbered the incumbent FACT value with the display token,
        so a same-value re-observation was misread as a different-value
        contradiction (`old_object`/`new_object`, no `survivor_key`)."""
        merger = GraphMerger()
        old_rel = _attr_relation(predicate="has_name", obj="Alex", indexed_key="graph_old")
        new_rel = _attr_relation(predicate="has_name", obj="Alex", indexed_key="graph_new")
        merger.merge_relations([old_rel], session_id="s0", log_label="name facts")
        merger.merge_relations([new_rel], session_id="s1", log_label="name facts")

        assert merger.removal_ledger["graph_old"] == {
            "reason": "attribute_key_superseded",
            "survivor_key": "graph_new",
        }


class TestAttributeGateDoesNotReachUpsertRelation:
    def test_no_upsert_relation_side_effects(self):
        """An attribute relation must never touch the Case-1/2/3 machinery
        (removal_ledger stays empty)."""
        merger = GraphMerger()
        merger.merge(_session(_attr_relation()))
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


# ---------------------------------------------------------------------------
# reconcile_provenance
# ---------------------------------------------------------------------------


class TestReconcileProvenance:
    def test_empty_target_reproduces_net_new(self):
        """An empty target degenerates to the net-new stamp: relation's
        speaker_id, and last_seen/first_seen fall back to timestamp."""
        target: dict = {}
        rel = Relation(
            subject="alex",
            predicate="has email",
            object="a@b.com",
            relation_type="attribute",
            speaker_id="speaker0",
        )
        reconcile_provenance(target, rel, "2026-01-01T00:00:00Z")
        assert target["speaker_id"] == "speaker0"
        assert target["last_seen"] == "2026-01-01T00:00:00Z"
        assert target["first_seen"] == "2026-01-01T00:00:00Z"
        assert "edge_source" not in target

    def test_widens_window_in_both_directions(self):
        target = {
            "speaker_id": "speaker0",
            "first_seen": "2026-03-01T00:00:00Z",
            "last_seen": "2026-03-01T00:00:00Z",
        }
        earlier = Relation(
            subject="a",
            predicate="p",
            object="b",
            relation_type="attribute",
            speaker_id="speaker0",
            first_seen="2026-01-01T00:00:00Z",
            last_seen="2026-01-01T00:00:00Z",
        )
        reconcile_provenance(target, earlier, "")
        assert target["first_seen"] == "2026-01-01T00:00:00Z", "must widen first_seen earlier"
        assert target["last_seen"] == "2026-03-01T00:00:00Z", "must not narrow last_seen"

        later = Relation(
            subject="a",
            predicate="p",
            object="b",
            relation_type="attribute",
            speaker_id="speaker0",
            first_seen="2026-06-01T00:00:00Z",
            last_seen="2026-06-01T00:00:00Z",
        )
        reconcile_provenance(target, later, "")
        assert target["last_seen"] == "2026-06-01T00:00:00Z", "must widen last_seen later"
        assert target["first_seen"] == "2026-01-01T00:00:00Z", "must not narrow first_seen"

    def test_speaker_id_untouched_when_already_present(self):
        target = {"speaker_id": "speaker0"}
        rel = Relation(
            subject="a",
            predicate="p",
            object="b",
            relation_type="attribute",
            speaker_id="speaker9",
        )
        reconcile_provenance(target, rel, "")
        assert target["speaker_id"] == "speaker0", "first-non-empty-wins must not overwrite"

    def test_incoming_empty_first_seen_never_wins_the_min(self):
        target = {"first_seen": "2026-01-01T00:00:00Z"}
        rel = Relation(
            subject="a",
            predicate="p",
            object="b",
            relation_type="attribute",
            speaker_id="speaker0",
            first_seen="",
        )
        # timestamp="" too, so relation.first_seen or timestamp == ""
        reconcile_provenance(target, rel, "")
        assert target["first_seen"] == "2026-01-01T00:00:00Z"

    def test_edge_source_written_only_when_relation_carries_one_and_target_has_none(self):
        target: dict = {}
        rel_no_source = Relation(
            subject="a",
            predicate="p",
            object="b",
            relation_type="attribute",
            speaker_id="speaker0",
        )
        reconcile_provenance(target, rel_no_source, "")
        assert "edge_source" not in target

        rel_with_source = Relation(
            subject="a",
            predicate="p",
            object="b",
            relation_type="attribute",
            speaker_id="speaker0",
            edge_source="graph_enrichment",
        )
        reconcile_provenance(target, rel_with_source, "")
        assert target["edge_source"] == "graph_enrichment"

        # A second, different edge_source must not overwrite the first.
        rel_second_source = Relation(
            subject="a",
            predicate="p",
            object="b",
            relation_type="attribute",
            speaker_id="speaker0",
            edge_source="other_source",
        )
        reconcile_provenance(target, rel_second_source, "")
        assert target["edge_source"] == "graph_enrichment"


# ---------------------------------------------------------------------------
# attribute_fact
# ---------------------------------------------------------------------------


class TestAttributeFact:
    def test_uses_display_surface_when_present(self):
        node_data = {"display_name": "Alex Morgan"}
        record = {
            "value": "alex@example.com",
            "speaker_id": "speaker0",
            "first_seen": "2026-01-01T00:00:00Z",
            "last_seen": "2026-02-01T00:00:00Z",
            "ik_key": "graph5",
        }
        fact = attribute_fact(node_data, "alex morgan", "email", record)
        assert fact == {
            "subject": "Alex Morgan",
            "predicate": "has email",
            "object": "alex@example.com",
            "speaker_id": "speaker0",
            "first_seen": "2026-01-01T00:00:00Z",
            "last_seen": "2026-02-01T00:00:00Z",
            "ik_key": "graph5",
        }

    def test_falls_back_to_node_key_when_no_display_name(self):
        fact = attribute_fact({}, "alex morgan", "email", {"value": "x"})
        assert fact["subject"] == "alex morgan"

    def test_ik_key_defaults_to_empty_string_for_keyless_record(self):
        fact = attribute_fact({}, "alex", "email", {"value": "x", "speaker_id": "speaker0"})
        assert fact["ik_key"] == ""

    def test_subject_matches_node_display_helper(self):
        """Both existing callers must agree with node_display's own resolution."""
        node_data = {"display_name": "Alex"}
        fact = attribute_fact(node_data, "alex", "email", {"value": "x"})
        assert fact["subject"] == node_display(node_data, "alex")

    def test_does_not_mutate_inputs(self):
        node_data = {"display_name": "Alex"}
        record = {"value": "x", "speaker_id": "speaker0"}
        node_data_before = dict(node_data)
        record_before = dict(record)
        attribute_fact(node_data, "alex", "email", record)
        assert node_data == node_data_before
        assert record == record_before


# ---------------------------------------------------------------------------
# The attribute gate's record shape and lifetime rules
# ---------------------------------------------------------------------------


class TestAttributeGateNetNewRecord:
    def test_net_new_record_carries_value_speaker_and_window(self):
        merger = GraphMerger()
        merger.merge(
            _session(
                _attr_relation(
                    speaker_id="speaker0",
                    first_seen="2026-01-01T00:00:00Z",
                    last_seen="2026-01-01T00:00:00Z",
                )
            )
        )
        record = merger.graph.nodes["speaker0"]["attributes"]["email"]
        assert record["value"] == "alex@example.com"
        assert record["speaker_id"] == "speaker0"
        assert record["first_seen"] == "2026-01-01T00:00:00Z"
        assert record["last_seen"] == "2026-01-01T00:00:00Z"
        assert "ik_key" not in record

    def test_net_new_record_carries_ik_key_when_relation_is_keyed(self):
        merger = GraphMerger()
        merger.merge(_session(_attr_relation(indexed_key="graph7")))
        record = merger.graph.nodes["speaker0"]["attributes"]["email"]
        assert record["ik_key"] == "graph7"

    def test_placeholder_value_is_skipped_and_writes_no_record_and_no_node(self):
        merger = GraphMerger()
        merger.merge(_session(_attr_relation(obj="N/A")))
        assert merger.graph.number_of_nodes() == 0
        assert merger.removal_ledger == {}


class TestAttributeGateSameValueReconciles:
    def test_re_merge_widens_window_without_changing_speaker_id(self):
        merger = GraphMerger()
        merger.merge(
            _session(
                _attr_relation(
                    speaker_id="speaker0",
                    first_seen="2026-03-01T00:00:00Z",
                    last_seen="2026-03-01T00:00:00Z",
                ),
                session_id="s0",
            )
        )
        merger.merge(
            _session(
                _attr_relation(
                    speaker_id="speaker9",  # different asserter — must not win
                    first_seen="2026-01-01T00:00:00Z",
                    last_seen="2026-06-01T00:00:00Z",
                ),
                session_id="s1",
            )
        )
        record = merger.graph.nodes["speaker0"]["attributes"]["email"]
        assert record["speaker_id"] == "speaker0", "first-non-empty-wins: original speaker kept"
        assert record["first_seen"] == "2026-01-01T00:00:00Z", "window must widen earlier"
        assert record["last_seen"] == "2026-06-01T00:00:00Z", "window must widen later"

    def test_keyed_same_value_carry_forward_ledgers_with_survivor_key(self):
        merger = GraphMerger()
        old_rel = _attr_relation(obj="Alex", predicate="has_name", indexed_key="graph_old")
        new_rel = _attr_relation(obj="Alex", predicate="has_name", indexed_key="graph_new")
        merger.merge_relations([old_rel], session_id="s0", log_label="name facts")
        merger.merge_relations([new_rel], session_id="s1", log_label="name facts")

        assert merger.removal_ledger["graph_old"] == {
            "reason": "attribute_key_superseded",
            "survivor_key": "graph_new",
        }
        record = merger.graph.nodes["speaker0"]["attributes"]["name"]
        assert record["ik_key"] == "graph_new"

    def test_repeated_merge_of_the_same_key_does_not_ledger(self):
        merger = GraphMerger()
        merger.merge(_session(_attr_relation(indexed_key="graph7"), session_id="s0"))
        merger.merge(_session(_attr_relation(indexed_key="graph7"), session_id="s1"))
        assert merger.removal_ledger == {}


class TestAttributeGateValueChangeStartsNewLifetime:
    """A different value never reconciles onto the incumbent record — it
    starts a new record lifetime with the new relation's own speaker and
    window."""

    def test_keyless_value_change_supersedes_bound_key_no_survivor(self):
        merger = GraphMerger()
        merger.merge(
            _session(
                _attr_relation(
                    obj="alex@example.com",
                    indexed_key="graph5",
                    speaker_id="speaker0",
                    first_seen="2026-01-01T00:00:00Z",
                    last_seen="2026-01-01T00:00:00Z",
                ),
                session_id="s0",
            )
        )
        merger.merge(
            _session(
                _attr_relation(
                    obj="alex-new@example.com",  # different value, no indexed_key
                    speaker_id="speaker1",
                    first_seen="2026-06-01T00:00:00Z",
                    last_seen="2026-06-01T00:00:00Z",
                ),
                session_id="s1",
            )
        )
        record = merger.graph.nodes["speaker0"]["attributes"]["email"]
        assert record["value"] == "alex-new@example.com"
        assert record["speaker_id"] == "speaker1"
        assert record["first_seen"] == "2026-06-01T00:00:00Z"
        assert record["last_seen"] == "2026-06-01T00:00:00Z"
        assert "ik_key" not in record, "new value has no key yet — the keyed walk mints one"

        assert merger.removal_ledger["graph5"] == {
            "reason": "attribute_key_superseded",
            "old_object": "alex@example.com",
            "new_object": "alex-new@example.com",
        }

    def test_keyless_value_change_with_no_bound_key_ledgers_nothing(self):
        """Inline check: speaker0 asserts a value keyless, speaker1
        re-asserts a DIFFERENT value keyless (later date) — no key was ever
        bound, so nothing is ledgered, but the record still carries
        speaker1 and speaker1's own dates as BOTH first_seen and last_seen —
        never speaker0's, never blended."""
        merger = GraphMerger()
        merger.merge(
            _session(
                _attr_relation(
                    obj="a@b.com",
                    speaker_id="speaker0",
                    first_seen="2026-01-01T00:00:00Z",
                    last_seen="2026-01-01T00:00:00Z",
                ),
                session_id="s0",
            )
        )
        merger.merge(
            _session(
                _attr_relation(
                    obj="c@d.com",
                    speaker_id="speaker1",
                    first_seen="2026-06-01T00:00:00Z",
                    last_seen="2026-06-01T00:00:00Z",
                ),
                session_id="s1",
            )
        )
        record = merger.graph.nodes["speaker0"]["attributes"]["email"]
        assert record["value"] == "c@d.com"
        assert record["speaker_id"] == "speaker1"
        assert record["first_seen"] == "2026-06-01T00:00:00Z"
        assert record["last_seen"] == "2026-06-01T00:00:00Z"
        assert "ik_key" not in record
        assert merger.removal_ledger == {}, "no key was ever bound — nothing to ledger"

    def test_keyed_value_change_keeps_two_arm_ledger_and_binds_new_key(self):
        """The keyed case (relation.indexed_key set, incumbent key
        different) keeps its current two-arm behaviour: a different value
        winning ledgers old_object/new_object with no survivor_key."""
        merger = GraphMerger()
        merger.merge(
            _session(_attr_relation(obj="alex@example.com", indexed_key="graph5"), session_id="s0")
        )
        merger.merge(
            _session(
                _attr_relation(obj="alex-new@example.com", indexed_key="graph6"), session_id="s1"
            )
        )
        record = merger.graph.nodes["speaker0"]["attributes"]["email"]
        assert record["value"] == "alex-new@example.com"
        assert record["ik_key"] == "graph6"
        assert merger.removal_ledger["graph5"] == {
            "reason": "attribute_key_superseded",
            "old_object": "alex@example.com",
            "new_object": "alex-new@example.com",
        }
