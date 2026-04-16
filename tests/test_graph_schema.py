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

    def test_invalid_entity_type(self):
        with pytest.raises(ValidationError):
            Entity(name="Alex", entity_type="invalid_type")

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
            )
            assert rel.relation_type == rtype


class TestRelation:
    def test_basic_relation(self):
        rel = Relation(
            subject="Alex",
            predicate="lives_in",
            object="Heilbronn",
            relation_type="factual",
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
            )

    def test_invalid_relation_type(self):
        with pytest.raises(ValidationError):
            Relation(
                subject="A",
                predicate="p",
                object="B",
                relation_type="unknown",
            )


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
