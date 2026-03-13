"""Knowledge graph merging with entity resolution and edge aggregation."""

import json
import logging
from pathlib import Path

import networkx as nx
from rapidfuzz import fuzz

from paramem.graph.schema import Entity, Relation, SessionGraph

logger = logging.getLogger(__name__)


class GraphMerger:
    """Merges per-session graphs into a cumulative knowledge graph.

    Handles entity resolution (exact + fuzzy matching), edge aggregation
    with recurrence counting, and JSON persistence via NetworkX.
    """

    def __init__(self, similarity_threshold: float = 85.0):
        self.similarity_threshold = similarity_threshold
        self.graph = nx.MultiDiGraph()

    def merge(self, session_graph: SessionGraph) -> nx.MultiDiGraph:
        """Merge a session graph into the cumulative graph.

        Returns the updated cumulative graph.
        """
        session_id = session_graph.session_id
        timestamp = session_graph.timestamp

        # Merge entities
        entity_name_map = {}  # session entity name -> canonical name in graph
        for entity in session_graph.entities:
            canonical = self._resolve_entity(entity)
            entity_name_map[entity.name] = canonical
            self._upsert_entity(canonical, entity, session_id, timestamp)

        # Merge relations
        for relation in session_graph.relations:
            subject = entity_name_map.get(relation.subject, relation.subject)
            obj = entity_name_map.get(relation.object, relation.object)

            # Ensure both endpoints exist as nodes
            for name in (subject, obj):
                if name not in self.graph:
                    self.graph.add_node(
                        name,
                        entity_type="concept",
                        attributes={},
                        first_seen=session_id,
                        last_seen=session_id,
                        recurrence_count=1,
                        sessions=[session_id],
                    )

            self._upsert_relation(subject, obj, relation, session_id, timestamp)

        logger.info(
            "Merged session %s: graph now has %d nodes, %d edges",
            session_id,
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )
        return self.graph

    def _resolve_entity(self, entity: Entity) -> str:
        """Resolve an entity name to its canonical form in the graph.

        Uses exact normalization first, then fuzzy matching.
        """
        normalized = _normalize_name(entity.name)

        # Exact match on normalized names
        for node in self.graph.nodes:
            if _normalize_name(node) == normalized:
                return node

        # Fuzzy match
        best_match = None
        best_score = 0.0
        for node in self.graph.nodes:
            node_data = self.graph.nodes[node]
            # Only match against same entity type
            if node_data.get("entity_type") != entity.entity_type:
                continue
            score = fuzz.token_sort_ratio(normalized, _normalize_name(node))
            if score > best_score and score >= self.similarity_threshold:
                best_score = score
                best_match = node

        if best_match is not None:
            logger.debug(
                "Fuzzy matched '%s' -> '%s' (score=%.1f)",
                entity.name,
                best_match,
                best_score,
            )
            return best_match

        return entity.name

    def _upsert_entity(
        self, canonical: str, entity: Entity, session_id: str, timestamp: str
    ) -> None:
        """Insert or update an entity node."""
        if canonical in self.graph:
            node = self.graph.nodes[canonical]
            node["recurrence_count"] = node.get("recurrence_count", 0) + 1
            node["last_seen"] = session_id
            sessions = node.get("sessions", [])
            if session_id not in sessions:
                sessions.append(session_id)
            node["sessions"] = sessions
            # Merge attributes (new values overwrite old)
            existing_attrs = node.get("attributes", {})
            existing_attrs.update(entity.attributes)
            node["attributes"] = existing_attrs
        else:
            self.graph.add_node(
                canonical,
                entity_type=entity.entity_type,
                attributes=dict(entity.attributes),
                first_seen=session_id,
                last_seen=session_id,
                recurrence_count=1,
                sessions=[session_id],
            )

    def _upsert_relation(
        self,
        subject: str,
        obj: str,
        relation: Relation,
        session_id: str,
        timestamp: str,
    ) -> None:
        """Insert or update a relation edge."""
        normalized_pred = _normalize_predicate(relation.predicate)

        # Check if this predicate already exists between these nodes
        existing_key = None
        if self.graph.has_node(subject) and self.graph.has_node(obj):
            existing_key = next(
                (
                    key
                    for key, data in self.graph[subject].get(obj, {}).items()
                    if data.get("predicate") == normalized_pred
                ),
                None,
            )

        if existing_key is not None:
            edge = self.graph[subject][obj][existing_key]
            edge["recurrence_count"] = edge.get("recurrence_count", 0) + 1
            edge["last_seen"] = session_id
            edge["confidence"] = max(edge.get("confidence", 0), relation.confidence)
            sessions = edge.get("sessions", [])
            if session_id not in sessions:
                sessions.append(session_id)
            edge["sessions"] = sessions
        else:
            self.graph.add_edge(
                subject,
                obj,
                predicate=normalized_pred,
                relation_type=relation.relation_type,
                confidence=relation.confidence,
                first_seen=session_id,
                last_seen=session_id,
                recurrence_count=1,
                sessions=[session_id],
            )

    def save_graph(self, path: str | Path) -> None:
        """Save cumulative graph to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = nx.node_link_data(self.graph)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Graph saved to %s", path)

    def load_graph(self, path: str | Path) -> nx.MultiDiGraph:
        """Load cumulative graph from JSON."""
        path = Path(path)
        if not path.exists():
            logger.info("No existing graph at %s, starting fresh", path)
            return self.graph

        with open(path) as f:
            data = json.load(f)
        self.graph = nx.node_link_graph(data)
        logger.info(
            "Graph loaded from %s: %d nodes, %d edges",
            path,
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )
        return self.graph


def _normalize_name(name: str) -> str:
    """Normalize an entity name for matching."""
    return name.strip().lower()


def _normalize_predicate(predicate: str) -> str:
    """Normalize a predicate for consistent matching and storage.

    Lowercases, strips whitespace, replaces spaces with underscores.
    Semantic alignment is handled upstream by the extraction prompt,
    which provides few-shot examples of canonical predicate forms.
    """
    return predicate.strip().lower().replace(" ", "_")
