"""Tests for knowledge graph schema models."""

import pytest
from pydantic import ValidationError

from paramem.graph.schema import Entity, Relation, SessionGraph


class TestEntity:
    def test_basic_entity(self):
        entity = Entity(name="Alex", entity_type="person")
        assert entity.name == "Alex"
        assert entity.entity_type == "person"
        assert entity.attributes == {}

    def test_entity_with_attributes(self):
        entity = Entity(
            name="Alex",
            entity_type="person",
            attributes={"age": "29", "role": "engineer"},
        )
        assert entity.attributes["age"] == "29"

    def test_novel_entity_type_accepted(self):
        # entity_type is open (no Literal enforcement). Novel types from the
        # SOTA enrichment path — product, certification, program, paper —
        # pass through verbatim. The schema YAML's entity_types list is a
        # soft prior used for prompt examples, not closed-set enforcement.
        entity = Entity(name="Honda Legend", entity_type="product")
        assert entity.entity_type == "product"
        entity = Entity(name="ASIL-D", entity_type="certification")
        assert entity.entity_type == "certification"

    def test_all_entity_types(self):
        from paramem.graph.schema_config import entity_types

        for etype in entity_types():
            entity = Entity(name="test", entity_type=etype)
            assert entity.entity_type == etype

    def test_all_relation_types(self):
        from paramem.graph.schema_config import relation_types

        for rtype in relation_types():
            rel = Relation(
                subject="A",
                predicate="p",
                object="B",
                relation_type=rtype,
                speaker_id="speaker0",
            )
            assert rel.relation_type == rtype


class TestRelation:
    def test_basic_relation(self):
        rel = Relation(
            subject="Alex",
            predicate="lives_in",
            object="Heilbronn",
            relation_type="factual",
            speaker_id="speaker0",
        )
        assert rel.subject == "Alex"
        assert rel.predicate == "lives_in"
        assert rel.confidence == 1.0

    def test_relation_with_confidence(self):
        rel = Relation(
            subject="Alex",
            predicate="prefers",
            object="Python",
            relation_type="preference",
            confidence=0.8,
            speaker_id="speaker0",
        )
        assert rel.confidence == 0.8

    def test_confidence_above_one_normalized_downstream(self):
        """Schema accepts > 1.0; _normalize_extraction scales to 0-1 before use."""
        rel = Relation(
            subject="A",
            predicate="p",
            object="B",
            relation_type="factual",
            confidence=99.9,
            speaker_id="speaker0",
        )
        assert rel.confidence == 99.9

    def test_invalid_confidence_negative(self):
        with pytest.raises(ValidationError):
            Relation(
                subject="A",
                predicate="p",
                object="B",
                relation_type="factual",
                confidence=-0.5,
                speaker_id="speaker0",
            )

    def test_invalid_relation_type(self):
        with pytest.raises(ValidationError):
            Relation(
                subject="A",
                predicate="p",
                object="B",
                relation_type="unknown",
                speaker_id="speaker0",
            )

    def test_session_ids_defaults_to_empty_list(self):
        """Relation.session_ids defaults to [] when not supplied."""
        rel = Relation(
            subject="A",
            predicate="lives_in",
            object="B",
            relation_type="factual",
            speaker_id="speaker0",
        )
        assert rel.session_ids == []

    def test_session_ids_excluded_from_serialisation(self):
        """session_ids is exclude=True — never persisted to JSON / registry.

        Acceptance test: a Relation round-trip through model_dump() (which
        drives all JSON persistence in the pipeline) must NOT contain
        'session_ids', so the field never leaks into the registry,
        cumulative_graph.json, or adapter-training data.
        """
        import json

        rel = Relation(
            subject="Alex",
            predicate="lives_in",
            object="Berlin",
            relation_type="factual",
            speaker_id="speaker0",
            session_ids=["real-session-abc"],
        )
        dumped = rel.model_dump()
        assert "session_ids" not in dumped, (
            "session_ids must be excluded from model_dump() — it is a transient "
            "carry-slot and must never be persisted to disk"
        )
        json_str = json.dumps(dumped)
        assert "session_ids" not in json_str
        assert "real-session-abc" not in json_str

    def test_indexed_key_excluded_from_serialisation(self):
        """indexed_key remains excluded from model_dump() (regression guard)."""
        rel = Relation(
            subject="A",
            predicate="p",
            object="B",
            relation_type="factual",
            speaker_id="speaker0",
            indexed_key="graph42",
        )
        dumped = rel.model_dump()
        assert "indexed_key" not in dumped


class TestSessionGraph:
    def test_empty_session_graph(self):
        graph = SessionGraph(
            session_id="s001",
            timestamp="2026-03-10T10:00:00Z",
        )
        assert graph.entities == []
        assert graph.relations == []
        assert graph.summary == ""

    def test_full_session_graph(self):
        graph = SessionGraph(
            session_id="s001",
            timestamp="2026-03-10T10:00:00Z",
            entities=[Entity(name="Alex", entity_type="person")],
            relations=[
                Relation(
                    subject="Alex",
                    predicate="lives_in",
                    object="Heilbronn",
                    relation_type="factual",
                    speaker_id="speaker0",
                )
            ],
            summary="Alex lives in Heilbronn.",
        )
        assert len(graph.entities) == 1
        assert len(graph.relations) == 1
        assert graph.summary == "Alex lives in Heilbronn."

    def test_session_graph_serialization(self):
        graph = SessionGraph(
            session_id="s001",
            timestamp="2026-03-10T10:00:00Z",
            entities=[Entity(name="Alex", entity_type="person")],
            relations=[],
        )
        data = graph.model_dump()
        restored = SessionGraph.model_validate(data)
        assert restored.session_id == "s001"
        assert restored.entities[0].name == "Alex"
