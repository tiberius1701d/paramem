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
- the scalar partition / projection round trip
- the recovery gate's decision, including the ``routing`` kind that
  suppresses it

The ``CAUSE_*`` vocabulary's own classification tests
(``cause_kind``/``EMPTY_CAUSE_KIND``) live in ``test_empty_cause.py`` now
that the vocabulary itself lives in ``paramem.graph.empty_cause``.
"""

from __future__ import annotations

import pytest

from paramem.graph.empty_cause import (
    CAUSE_DEANON_JUDGE,
    CAUSE_SCALAR_PARTITION,
    CAUSE_SCHEMA_VALIDATION,
    CAUSE_UNATTRIBUTED,
)
from paramem.graph.relation_build import (
    apply_rebuild,
    build_relations,
    is_scalar_value,
    partition_scalar_facts,
    project_scalar_facts_to_attributes,
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
            [],
            {"Person_1": "Alex", "Paper_1": "Millfield"},
        )
        types = {e.name: e.entity_type for e in graph.entities}
        assert types["Alex"] == "person"
        assert types["Millfield"] == "paper"

    def test_unmapped_name_defaults_to_concept(self):
        graph = _graph()
        apply_rebuild(graph, [self._relation("Alex", "Springfield")], [], {})
        types = {e.name: e.entity_type for e in graph.entities}
        assert types == {"Alex": "concept", "Springfield": "concept"}

    def test_lookup_uses_the_inverted_resolution_map(self):
        """``resolution`` is keyed by PLACEHOLDER; the names being typed
        are real names, so a non-inverted lookup would never match and
        everything would fall back to "concept"."""
        graph = _graph()
        apply_rebuild(graph, [self._relation("Alex", "Millfield")], [], {"City_1": "Millfield"})
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
        apply_rebuild(graph, [self._relation("Alex", "Millfield")], [], {"Paper_1": "Alex"})
        types = {e.name: e.entity_type for e in graph.entities}
        assert types == {"Alex": "person", "Millfield": "place"}

    def test_relations_are_installed(self):
        graph = _graph()
        rel = self._relation("Alex", "Millfield")
        apply_rebuild(graph, [rel], [], {})
        assert graph.relations == [rel]

    def test_scalar_subject_survives_pruning_and_is_projected(self):
        """A subject that appears ONLY in a scalar fact must not be pruned
        away before the projection attaches the attribute to it."""
        graph = _graph([Entity(name="Alex", entity_type="person")])
        scalar = [_fact(subject="Alex", predicate="has_email", obj="alex@example.com")]
        apply_rebuild(graph, [], scalar, {})
        alex = next(e for e in graph.entities if e.name == "Alex")
        assert alex.attributes["email"] == "alex@example.com"
        assert graph.relations == []


class TestScalarRoundTrip:
    def test_partition_then_project_round_trip(self):
        graph = _graph([Entity(name="Alex", entity_type="person")])
        facts = [
            _fact(subject="Alex", predicate="has_email", obj="alex@example.com"),
            _fact(subject="Alex", predicate="uses", obj="ROS2"),
            _fact(subject="Alex", predicate="lives_in", obj="Millfield"),
        ]
        scalar, non_scalar = partition_scalar_facts(facts, graph.entities)
        assert [f["object"] for f in scalar] == ["alex@example.com", "ROS2"]
        assert [f["object"] for f in non_scalar] == ["Millfield"]

        project_scalar_facts_to_attributes(graph, scalar)
        alex = next(e for e in graph.entities if e.name == "Alex")
        # ``has_`` is stripped so relation_prep's re-prefixing cannot
        # produce ``has_has_email``.
        assert alex.attributes == {"email": "alex@example.com", "uses": "ROS2"}

    def test_object_naming_an_existing_entity_is_never_scalar(self):
        entities = [Entity(name="R2D2", entity_type="person")]
        scalar, non_scalar = partition_scalar_facts(
            [_fact(subject="Alex", predicate="knows", obj="R2D2")], entities
        )
        assert scalar == []
        assert len(non_scalar) == 1

    def test_synthetic_facts_pass_through(self):
        scalar, non_scalar = partition_scalar_facts(
            [_fact(obj="https://example.com/x", synthetic=True)], []
        )
        assert scalar == []
        assert len(non_scalar) == 1

    def test_projection_mints_a_missing_subject_entity(self):
        graph = _graph()
        project_scalar_facts_to_attributes(
            graph, [_fact(subject="Acme", predicate="has_url", obj="acme.example/jobs")]
        )
        acme = next(e for e in graph.entities if e.name == "Acme")
        assert acme.entity_type == "concept"
        assert acme.attributes == {"url": "acme.example/jobs"}

    @pytest.mark.parametrize(
        "value",
        ["https://example.com/a", "alex@example.com", "+1 555 123 4567", "ROS2", "10.1000/xyz123"],
    )
    def test_scalar_shapes(self, value):
        assert is_scalar_value(value)

    @pytest.mark.parametrize("value", ["Millfield", "the electricity bill", "", "a place"])
    def test_non_scalar_shapes(self, value):
        assert not is_scalar_value(value)


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

    def test_does_not_fire_on_a_routing_cause(self):
        """A scalar-only session emptied the relation set on PURPOSE: the
        facts moved to the entity-attribute surface, nothing was lost, and
        the recovery path would both skip the projection and
        re-materialize the same facts as relations."""
        graph = _graph()
        assert recovery_gate(graph, [], 4, CAUSE_SCALAR_PARTITION) is False
        # Still recorded — the operator sees WHY the relation set is empty.
        assert graph.diagnostics["all_dropped_cause"] == {
            "cause": CAUSE_SCALAR_PARTITION,
            "kind": "routing",
        }

    def test_fires_with_no_claimed_cause(self):
        graph = _graph()
        assert recovery_gate(graph, [], 4, None) is True
        assert graph.diagnostics["all_dropped_cause"] == {
            "cause": CAUSE_UNATTRIBUTED,
            "kind": CAUSE_UNATTRIBUTED,
        }
