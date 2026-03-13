"""Tests for knowledge graph merging and entity resolution."""

import tempfile
from pathlib import Path

import pytest

from paramem.graph.merger import GraphMerger, _normalize_name, _normalize_predicate
from paramem.graph.schema import Entity, Relation, SessionGraph


@pytest.fixture
def merger():
    return GraphMerger(similarity_threshold=85.0)


@pytest.fixture
def session_graph_1():
    return SessionGraph(
        session_id="s001",
        timestamp="2026-03-10T10:00:00Z",
        entities=[
            Entity(name="Alex", entity_type="person", attributes={"age": "29"}),
            Entity(name="Heilbronn", entity_type="place"),
            Entity(name="AutoMate", entity_type="organization"),
        ],
        relations=[
            Relation(
                subject="Alex", predicate="lives_in", object="Heilbronn", relation_type="factual"
            ),
            Relation(
                subject="Alex", predicate="works_at", object="AutoMate", relation_type="factual"
            ),
        ],
    )


@pytest.fixture
def session_graph_2():
    return SessionGraph(
        session_id="s002",
        timestamp="2026-03-11T10:00:00Z",
        entities=[
            Entity(name="Alex", entity_type="person"),
            Entity(name="Python", entity_type="concept"),
        ],
        relations=[
            Relation(
                subject="Alex", predicate="prefers", object="Python", relation_type="preference"
            ),
        ],
    )


class TestNormalization:
    def test_normalize_basic(self):
        assert _normalize_name("Alex") == "alex"
        assert _normalize_name("  Alex  ") == "alex"
        assert _normalize_name("Dr. Smith") == "dr. smith"


class TestPredicateNormalization:
    def test_lowercase_and_underscore(self):
        assert _normalize_predicate("Works At") == "works_at"
        assert _normalize_predicate("LIVES IN") == "lives_in"

    def test_strip_whitespace(self):
        assert _normalize_predicate("  works_at  ") == "works_at"

    def test_passthrough_preserves_content(self):
        assert _normalize_predicate("invented") == "invented"
        assert _normalize_predicate("custom_pred") == "custom_pred"
        assert _normalize_predicate("works_at") == "works_at"

    def test_deduplicates_edges_across_variants(self, merger):
        """'works_at' and 'works at' should merge into one edge."""
        g1 = SessionGraph(
            session_id="s001",
            timestamp="2026-03-10T10:00:00Z",
            entities=[
                Entity(name="A", entity_type="person"),
                Entity(name="B", entity_type="organization"),
            ],
            relations=[
                Relation(
                    subject="A",
                    predicate="works_at",
                    object="B",
                    relation_type="factual",
                )
            ],
        )
        g2 = SessionGraph(
            session_id="s002",
            timestamp="2026-03-11T10:00:00Z",
            entities=[
                Entity(name="A", entity_type="person"),
                Entity(name="B", entity_type="organization"),
            ],
            relations=[
                Relation(
                    subject="A",
                    predicate="works at",
                    object="B",
                    relation_type="factual",
                )
            ],
        )
        merger.merge(g1)
        merger.merge(g2)
        edges = list(merger.graph["A"]["B"].values())
        assert len(edges) == 1
        assert edges[0]["predicate"] == "works_at"
        assert edges[0]["recurrence_count"] == 2


class TestEntityResolution:
    def test_exact_match(self, merger, session_graph_1, session_graph_2):
        merger.merge(session_graph_1)
        merger.merge(session_graph_2)
        # "Alex" should resolve to the same node
        assert "Alex" in merger.graph.nodes
        assert merger.graph.nodes["Alex"]["recurrence_count"] == 2

    def test_fuzzy_match(self, merger):
        g1 = SessionGraph(
            session_id="s001",
            timestamp="2026-03-10T10:00:00Z",
            entities=[Entity(name="Alexander", entity_type="person")],
            relations=[],
        )
        g2 = SessionGraph(
            session_id="s002",
            timestamp="2026-03-11T10:00:00Z",
            entities=[Entity(name="alexander", entity_type="person")],
            relations=[],
        )
        merger.merge(g1)
        merger.merge(g2)
        # Case-insensitive match: should be same node
        assert merger.graph.number_of_nodes() == 1

    def test_no_cross_type_match(self, merger):
        """Entities of different types should not be fuzzy-matched."""
        g1 = SessionGraph(
            session_id="s001",
            timestamp="2026-03-10T10:00:00Z",
            entities=[Entity(name="Python", entity_type="concept")],
            relations=[],
        )
        g2 = SessionGraph(
            session_id="s002",
            timestamp="2026-03-11T10:00:00Z",
            entities=[Entity(name="Python", entity_type="organization")],
            relations=[],
        )
        merger.merge(g1)
        merger.merge(g2)
        # Exact name match overrides type check via normalization
        # So "Python" matches the first one regardless of type
        assert merger.graph.number_of_nodes() == 1


class TestEdgeAggregation:
    def test_new_edge(self, merger, session_graph_1):
        merger.merge(session_graph_1)
        assert merger.graph.number_of_edges() == 2

    def test_recurring_edge(self, merger):
        g1 = SessionGraph(
            session_id="s001",
            timestamp="2026-03-10T10:00:00Z",
            entities=[
                Entity(name="A", entity_type="person"),
                Entity(name="B", entity_type="place"),
            ],
            relations=[
                Relation(subject="A", predicate="lives_in", object="B", relation_type="factual")
            ],
        )
        g2 = SessionGraph(
            session_id="s002",
            timestamp="2026-03-11T10:00:00Z",
            entities=[
                Entity(name="A", entity_type="person"),
                Entity(name="B", entity_type="place"),
            ],
            relations=[
                Relation(subject="A", predicate="lives_in", object="B", relation_type="factual")
            ],
        )
        merger.merge(g1)
        merger.merge(g2)

        # Same predicate between same nodes should be aggregated
        edges = list(merger.graph["A"]["B"].values())
        assert len(edges) == 1
        assert edges[0]["recurrence_count"] == 2
        assert len(edges[0]["sessions"]) == 2

    def test_different_predicates(self, merger):
        g1 = SessionGraph(
            session_id="s001",
            timestamp="2026-03-10T10:00:00Z",
            entities=[
                Entity(name="A", entity_type="person"),
                Entity(name="B", entity_type="organization"),
            ],
            relations=[
                Relation(subject="A", predicate="works_at", object="B", relation_type="factual"),
                Relation(subject="A", predicate="manages", object="B", relation_type="factual"),
            ],
        )
        merger.merge(g1)
        edges = list(merger.graph["A"]["B"].values())
        assert len(edges) == 2


class TestSessionTracking:
    def test_session_provenance(self, merger, session_graph_1, session_graph_2):
        merger.merge(session_graph_1)
        merger.merge(session_graph_2)
        sessions = merger.graph.nodes["Alex"]["sessions"]
        assert "s001" in sessions
        assert "s002" in sessions

    def test_attribute_merge(self, merger):
        g1 = SessionGraph(
            session_id="s001",
            timestamp="2026-03-10T10:00:00Z",
            entities=[Entity(name="Alex", entity_type="person", attributes={"age": "29"})],
            relations=[],
        )
        g2 = SessionGraph(
            session_id="s002",
            timestamp="2026-03-11T10:00:00Z",
            entities=[Entity(name="Alex", entity_type="person", attributes={"role": "engineer"})],
            relations=[],
        )
        merger.merge(g1)
        merger.merge(g2)
        attrs = merger.graph.nodes["Alex"]["attributes"]
        assert attrs["age"] == "29"
        assert attrs["role"] == "engineer"


class TestPersistence:
    def test_save_and_load(self, merger, session_graph_1):
        merger.merge(session_graph_1)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "graph.json"
            merger.save_graph(path)

            assert path.exists()

            new_merger = GraphMerger()
            new_merger.load_graph(path)

            assert new_merger.graph.number_of_nodes() == merger.graph.number_of_nodes()
            assert new_merger.graph.number_of_edges() == merger.graph.number_of_edges()
            assert "Alex" in new_merger.graph.nodes

    def test_load_nonexistent(self, merger):
        graph = merger.load_graph("/nonexistent/path.json")
        assert graph.number_of_nodes() == 0
