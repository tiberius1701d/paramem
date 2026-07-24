"""Unit tests for paramem.graph.relation_build.

Pure post-processing: no model, no tokenizer, no cloud, no config. Every
test here is a total function of its inputs — which is exactly why this
logic was extracted out of the middle of what used to be
``extractor._cloud_pipeline``, where reaching it required standing up the
whole anonymize → enrich → judge chain first.

Covers:
- schema-validation drop recording (kept vs. dropped, and the record shape)
- entity typing derived from a placeholder prefix, plus the pruning and
  the inverted-resolution lookup that feeds it
- the recovery gate's decision

The ``CAUSE_*`` vocabulary's own classification tests
(``cause_kind``/``EMPTY_CAUSE_KIND``) live in ``test_empty_cause.py`` now
that the vocabulary itself lives in ``paramem.graph.empty_cause``.

Literal-value ("scalar") facts are no longer routed by this module — the
model tags them ``relation_type="attribute"`` at extraction time and
``GraphMerger`` folds them onto the subject node's ``attributes`` dict
(see ``tests/graph/test_merger_attribute_gate.py``).
"""

from __future__ import annotations

from paramem.graph.empty_cause import (
    CAUSE_DEANON_JUDGE,
    CAUSE_SCHEMA_VALIDATION,
    CAUSE_UNATTRIBUTED,
)
from paramem.graph.relation_build import (
    apply_rebuild,
    build_relations,
    recovery_gate,
)
from paramem.graph.schema import Entity, Relation, SessionGraph


def _graph(entities: list[Entity] | None = None) -> SessionGraph:
    return SessionGraph(
        session_id="s0",
        timestamp="2026-07-21T00:00:00Z",
        entities=list(entities or []),
        relations=[],
    )


def _fact(subject="Alex", predicate="lives_in", obj="Millfield", **extra) -> dict:
    return {"subject": subject, "predicate": predicate, "object": obj, **extra}


class TestBuildRelations:
    def test_valid_facts_become_relations_with_speaker_provenance(self):
        graph = _graph()
        kept = build_relations(graph, [_fact(), _fact(obj="Berlin")], speaker_id="speaker3")
        assert [(r.subject, r.predicate, r.object) for r in kept] == [
            ("Alex", "lives_in", "Millfield"),
            ("Alex", "lives_in", "Berlin"),
        ]
        assert {r.speaker_id for r in kept} == {"speaker3"}
        assert "pydantic_validation_dropped" not in graph.diagnostics

    def test_invalid_relation_type_is_dropped_and_recorded(self):
        """A relation_type outside the schema's Literal set is dropped —
        and the drop is recorded, never silent."""
        graph = _graph()
        kept = build_relations(
            graph,
            [_fact(), _fact(obj="Berlin", relation_type="not_a_real_type")],
            speaker_id="speaker0",
        )
        assert len(kept) == 1
        dropped = graph.diagnostics["pydantic_validation_dropped"]
        assert len(dropped) == 1
        assert dropped[0]["subject"] == "Alex"
        assert dropped[0]["object"] == "Berlin"
        assert dropped[0]["relation_type"] == "not_a_real_type"
        assert dropped[0]["reason"]

    def test_all_facts_dropped_records_every_drop(self):
        graph = _graph()
        kept = build_relations(
            graph,
            [_fact(relation_type="bogus"), _fact(obj="Berlin", relation_type="bogus")],
            speaker_id="speaker0",
        )
        assert kept == []
        assert len(graph.diagnostics["pydantic_validation_dropped"]) == 2

    def test_graph_relations_are_not_installed_here(self):
        """Building is separate from installing — ``apply_rebuild`` owns
        the assignment, so a caller can consult the recovery gate in
        between."""
        graph = _graph()
        build_relations(graph, [_fact()], speaker_id="speaker0")
        assert graph.relations == []

    def test_defaults_fill_missing_fields(self):
        graph = _graph()
        kept = build_relations(
            graph, [{"subject": "A", "predicate": "p", "object": "B"}], speaker_id="speaker0"
        )
        assert kept[0].relation_type == "factual"
        assert kept[0].confidence == 1.0
        assert kept[0].symmetric is False


class TestApplyRebuild:
    def _relation(self, subject: str, obj: str) -> Relation:
        return Relation(
            subject=subject,
            predicate="lives_in",
            object=obj,
            relation_type="factual",
            confidence=1.0,
            speaker_id="speaker0",
        )

    def test_entity_type_derived_from_placeholder_prefix(self):
        """A name the cloud minted has no pre-existing Entity; its type
        comes from the prefix of the placeholder it resolved from — the
        prefix name IS the type name in the brace-binding protocol."""
        graph = _graph()
        apply_rebuild(
            graph,
            [self._relation("Alex", "Millfield")],
            {"Person_1": "Alex", "Paper_1": "Millfield"},
        )
        types = {e.name: e.entity_type for e in graph.entities}
        assert types["Alex"] == "person"
        assert types["Millfield"] == "paper"

    def test_unmapped_name_defaults_to_concept(self):
        graph = _graph()
        apply_rebuild(graph, [self._relation("Alex", "Springfield")], {})
        types = {e.name: e.entity_type for e in graph.entities}
        assert types == {"Alex": "concept", "Springfield": "concept"}

    def test_lookup_uses_the_inverted_resolution_map(self):
        """``resolution`` is keyed by PLACEHOLDER; the names being typed
        are real names, so a non-inverted lookup would never match and
        everything would fall back to "concept"."""
        graph = _graph()
        apply_rebuild(graph, [self._relation("Alex", "Millfield")], {"City_1": "Millfield"})
        types = {e.name: e.entity_type for e in graph.entities}
        assert types["Millfield"] != "concept"

    def test_existing_entities_keep_their_type_and_orphans_are_pruned(self):
        graph = _graph(
            [
                Entity(name="Alex", entity_type="person"),
                Entity(name="Millfield", entity_type="place"),
                Entity(name="Orphan", entity_type="concept"),
            ]
        )
        apply_rebuild(graph, [self._relation("Alex", "Millfield")], {"Paper_1": "Alex"})
        types = {e.name: e.entity_type for e in graph.entities}
        assert types == {"Alex": "person", "Millfield": "place"}

    def test_relations_are_installed(self):
        graph = _graph()
        rel = self._relation("Alex", "Millfield")
        apply_rebuild(graph, [rel], {})
        assert graph.relations == [rel]


class TestRecoveryGate:
    def _rel(self) -> Relation:
        return Relation(
            subject="Alex",
            predicate="lives_in",
            object="Millfield",
            relation_type="factual",
            confidence=1.0,
            speaker_id="speaker0",
        )

    def test_does_not_fire_when_relations_survived(self):
        graph = _graph()
        assert recovery_gate(graph, [self._rel()], 3, CAUSE_DEANON_JUDGE) is False
        assert "all_dropped_cause" not in graph.diagnostics

    def test_does_not_fire_when_there_was_nothing_to_lose(self):
        """No relations went in, so none were 'dropped' — the net exists
        to catch losses, not empty input."""
        graph = _graph()
        assert recovery_gate(graph, [], 0, None) is False
        assert "all_dropped_cause" not in graph.diagnostics

    def test_fires_and_records_a_judgment_cause(self):
        graph = _graph()
        assert recovery_gate(graph, [], 4, CAUSE_DEANON_JUDGE) is True
        assert graph.diagnostics["all_dropped_cause"] == {
            "cause": CAUSE_DEANON_JUDGE,
            "kind": "judgment",
        }

    def test_fires_and_records_a_breakage_cause(self):
        graph = _graph()
        assert recovery_gate(graph, [], 4, CAUSE_SCHEMA_VALIDATION) is True
        assert graph.diagnostics["all_dropped_cause"] == {
            "cause": CAUSE_SCHEMA_VALIDATION,
            "kind": "breakage",
        }

    def test_fires_with_no_claimed_cause(self):
        graph = _graph()
        assert recovery_gate(graph, [], 4, None) is True
        assert graph.diagnostics["all_dropped_cause"] == {
            "cause": CAUSE_UNATTRIBUTED,
            "kind": CAUSE_UNATTRIBUTED,
        }
