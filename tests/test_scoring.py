"""Tests for promotion scoring and node classification."""

import networkx as nx
import pytest

from paramem.graph.scoring import (
    PromotionScorer,
    ScoringWeights,
    get_relations_for_nodes,
)


@pytest.fixture
def sample_graph():
    """Build a small knowledge graph with varied recurrence."""
    G = nx.MultiDiGraph()

    # Highly connected, recurring entity
    G.add_node(
        "Alex",
        entity_type="person",
        recurrence_count=10,
        sessions=["s001", "s002", "s003", "s004", "s005"],
    )
    # Moderately connected
    G.add_node(
        "Heilbronn", entity_type="place", recurrence_count=5, sessions=["s001", "s003", "s005"]
    )
    # Low recurrence
    G.add_node("Barcelona", entity_type="place", recurrence_count=1, sessions=["s010"])
    # Recently added
    G.add_node(
        "Python", entity_type="concept", recurrence_count=3, sessions=["s001", "s002", "s003"]
    )

    G.add_edge("Alex", "Heilbronn", predicate="lives_in", recurrence_count=5)
    G.add_edge("Alex", "Python", predicate="prefers", recurrence_count=3)
    G.add_edge("Alex", "Barcelona", predicate="visited", recurrence_count=1)

    return G


class TestPromotionScorer:
    def test_score_nodes_returns_all_nodes(self, sample_graph):
        scorer = PromotionScorer()
        scores = scorer.score_nodes(sample_graph, current_cycle=5)
        assert set(scores.keys()) == set(sample_graph.nodes)

    def test_scores_between_zero_and_one(self, sample_graph):
        scorer = PromotionScorer()
        scores = scorer.score_nodes(sample_graph, current_cycle=5)
        for score in scores.values():
            assert 0.0 <= score <= 1.0

    def test_high_recurrence_scores_higher(self, sample_graph):
        scorer = PromotionScorer()
        scores = scorer.score_nodes(sample_graph, current_cycle=5)
        # Alex (recurrence=10) should score higher than Barcelona (recurrence=1)
        assert scores["Alex"] > scores["Barcelona"]

    def test_empty_graph(self):
        scorer = PromotionScorer()
        scores = scorer.score_nodes(nx.MultiDiGraph(), current_cycle=1)
        assert scores == {}

    def test_custom_weights(self, sample_graph):
        weights = ScoringWeights(pagerank=0.0, degree_centrality=0.0, recurrence=1.0, recency=0.0)
        scorer = PromotionScorer(weights=weights)
        scores = scorer.score_nodes(sample_graph, current_cycle=5)
        # With only recurrence weight, Alex should score max
        assert scores["Alex"] == 1.0


class TestNodeClassification:
    def test_classify_promotes_recurring(self, sample_graph):
        scorer = PromotionScorer()
        result = scorer.classify_nodes(
            sample_graph,
            promotion_threshold=3,
            decay_window=5,
            current_cycle=15,
        )
        # Alex (recurrence=10) and Heilbronn (5) and Python (3) should promote
        assert "Alex" in result.promote
        assert "Heilbronn" in result.promote
        assert "Python" in result.promote

    def test_classify_decays_old(self, sample_graph):
        scorer = PromotionScorer()
        result = scorer.classify_nodes(
            sample_graph,
            promotion_threshold=3,
            decay_window=5,
            current_cycle=20,
        )
        # Barcelona (1 session, last seen at cycle ~1) should decay at cycle 20
        assert "Barcelona" in result.decay

    def test_classify_retains_recent(self, sample_graph):
        scorer = PromotionScorer()
        # At cycle 2, Barcelona (1 session) is still within decay window
        result = scorer.classify_nodes(
            sample_graph,
            promotion_threshold=3,
            decay_window=5,
            current_cycle=2,
        )
        assert "Barcelona" in result.retain

    def test_classification_is_exhaustive(self, sample_graph):
        scorer = PromotionScorer()
        result = scorer.classify_nodes(
            sample_graph,
            promotion_threshold=3,
            decay_window=5,
            current_cycle=5,
        )
        all_classified = set(result.promote + result.decay + result.retain)
        assert all_classified == set(sample_graph.nodes)


class TestGetRelationsForNodes:
    def test_get_relations(self, sample_graph):
        relations = get_relations_for_nodes(sample_graph, ["Alex"])
        assert len(relations) == 3  # lives_in, prefers, visited
        predicates = {r["predicate"] for r in relations}
        assert "lives_in" in predicates

    def test_get_relations_empty(self, sample_graph):
        relations = get_relations_for_nodes(sample_graph, ["nonexistent"])
        assert len(relations) == 0

    def test_get_relations_subset(self, sample_graph):
        relations = get_relations_for_nodes(sample_graph, ["Heilbronn"])
        assert len(relations) == 1
        assert relations[0]["predicate"] == "lives_in"
