"""Promotion scoring for knowledge graph nodes.

Scores entities based on graph metrics to decide which episodic
memories should be promoted to the semantic adapter or decayed.
"""

import logging
from dataclasses import dataclass, field

import networkx as nx

logger = logging.getLogger(__name__)


@dataclass
class ScoringWeights:
    """Weights for composite promotion score components."""

    pagerank: float = 0.4
    degree_centrality: float = 0.3
    recurrence: float = 0.2
    recency: float = 0.1


@dataclass
class NodeClassification:
    """Result of classifying nodes into promote/decay/retain."""

    promote: list[str] = field(default_factory=list)
    decay: list[str] = field(default_factory=list)
    retain: list[str] = field(default_factory=list)


class PromotionScorer:
    """Scores graph nodes for promotion from episodic to semantic memory."""

    def __init__(self, weights: ScoringWeights | None = None):
        self.weights = weights or ScoringWeights()

    def score_nodes(self, graph: nx.MultiDiGraph, current_cycle: int) -> dict[str, float]:
        """Compute composite promotion scores for all nodes.

        Args:
            graph: The cumulative knowledge graph.
            current_cycle: Current consolidation cycle number.

        Returns:
            Dict mapping node names to promotion scores (0.0 to 1.0).
        """
        if graph.number_of_nodes() == 0:
            return {}

        # Convert to simple DiGraph for centrality algorithms
        simple_graph = nx.DiGraph(graph)

        pagerank = nx.pagerank(simple_graph, alpha=0.85)
        degree = nx.degree_centrality(simple_graph)

        # Normalize recurrence and recency to 0-1
        max_recurrence = max(
            (graph.nodes[n].get("recurrence_count", 1) for n in graph.nodes),
            default=1,
        )

        scores = {}
        for node in graph.nodes:
            node_data = graph.nodes[node]

            pr = pagerank.get(node, 0.0)
            dc = degree.get(node, 0.0)
            recurrence = node_data.get("recurrence_count", 1) / max(max_recurrence, 1)

            # Recency: map to 0-1 where recent=higher
            sessions = node_data.get("sessions", [])
            recency = 1.0 / (1.0 + (current_cycle - len(sessions))) if sessions else 0.0

            composite = (
                self.weights.pagerank * pr
                + self.weights.degree_centrality * dc
                + self.weights.recurrence * recurrence
                + self.weights.recency * recency
            )
            scores[node] = composite

        # Normalize to 0-1
        if scores:
            max_score = max(scores.values())
            if max_score > 0:
                scores = {k: v / max_score for k, v in scores.items()}

        return scores

    def classify_nodes(
        self,
        graph: nx.MultiDiGraph,
        promotion_threshold: int,
        decay_window: int,
        current_cycle: int,
    ) -> NodeClassification:
        """Classify nodes into promote, decay, or retain.

        Args:
            graph: The cumulative knowledge graph.
            promotion_threshold: Minimum recurrence count to promote.
            decay_window: Cycles since last seen before decay.
            current_cycle: Current consolidation cycle number.

        Returns:
            NodeClassification with lists of node names.
        """
        result = NodeClassification()

        for node in graph.nodes:
            node_data = graph.nodes[node]
            recurrence = node_data.get("recurrence_count", 1)
            sessions = node_data.get("sessions", [])

            # Estimate last seen cycle from session count
            cycles_since_seen = max(0, current_cycle - len(sessions))

            if recurrence >= promotion_threshold:
                result.promote.append(node)
            elif cycles_since_seen >= decay_window:
                result.decay.append(node)
            else:
                result.retain.append(node)

        logger.info(
            "Classification at cycle %d: %d promote, %d decay, %d retain",
            current_cycle,
            len(result.promote),
            len(result.decay),
            len(result.retain),
        )
        return result


def get_relations_for_nodes(graph: nx.MultiDiGraph, node_names: list[str]) -> list[dict]:
    """Get all relations involving the given nodes.

    Returns list of dicts with subject, predicate, object, and edge attributes.
    """
    relations = []
    node_set = set(node_names)

    for u, v, data in graph.edges(data=True):
        if u in node_set or v in node_set:
            relations.append(
                {
                    "subject": u,
                    "predicate": data.get("predicate", "related_to"),
                    "object": v,
                    "relation_type": data.get("relation_type", "factual"),
                    "confidence": data.get("confidence", 1.0),
                    "recurrence_count": data.get("recurrence_count", 1),
                }
            )

    return relations
