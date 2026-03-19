"""Knowledge graph merging with entity resolution and edge aggregation.

Two contradiction resolution strategies:
1. Graph-level: normalized predicate matching (lives_in == lives_in)
2. Model-level: LLM semantic reasoning (moved_to contradicts lives_in)
"""

import json
import logging
from pathlib import Path

import networkx as nx
from rapidfuzz import fuzz

from paramem.graph.schema import Entity, Relation, SessionGraph

logger = logging.getLogger(__name__)

_CONTRADICTION_PROMPT = """\
You are checking if a new fact contradicts any existing fact about a person.

Existing facts:
{existing_facts}

New fact: {subject} | {predicate} | {object}

Does the new fact contradict or update any existing fact? A contradiction means \
the new fact makes an old fact no longer true (e.g. "moved to Berlin" contradicts \
"lives in Munich").

Reply with EXACTLY one of:
- CONTRADICTS <subject> | <predicate> | <object> (the old fact it replaces)
- NO_CONTRADICTION"""


def detect_contradiction_with_model(
    subject: str,
    predicate: str,
    obj: str,
    existing_triples: list[tuple[str, str, str]],
    model,
    tokenizer,
) -> tuple[str, str, str] | None:
    """Use the model to detect if a new triple contradicts an existing one.

    Returns the contradicted (subject, predicate, object) triple, or None.
    """
    from paramem.evaluation.recall import generate_answer
    from paramem.models.loader import adapt_messages

    if not existing_triples:
        return None

    # Filter to triples about the same subject
    relevant = [(s, p, o) for s, p, o in existing_triples if s.lower() == subject.lower()]
    if not relevant:
        return None

    facts_str = "\n".join(f"- {s} | {p} | {o}" for s, p, o in relevant)
    prompt = _CONTRADICTION_PROMPT.format(
        existing_facts=facts_str,
        subject=subject,
        predicate=predicate,
        object=obj,
    )

    messages = adapt_messages(
        [
            {"role": "system", "content": "You detect contradictions between facts."},
            {"role": "user", "content": prompt},
        ],
        tokenizer,
    )
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    output = generate_answer(
        model,
        tokenizer,
        formatted,
        max_new_tokens=64,
        temperature=0.0,
    )

    output = output.strip()
    if output.startswith("CONTRADICTS"):
        # Parse "CONTRADICTS subject | predicate | object"
        parts = output[len("CONTRADICTS") :].strip().split("|")
        if len(parts) == 3:
            old_s = parts[0].strip()
            old_p = parts[1].strip()
            old_o = parts[2].strip()
            return (old_s, old_p, old_o)
        logger.warning("Could not parse contradiction response: %s", output)
    return None


class GraphMerger:
    """Merges per-session graphs into a cumulative knowledge graph.

    Handles entity resolution (exact + fuzzy matching), edge aggregation
    with recurrence counting, and JSON persistence via NetworkX.
    """

    def __init__(
        self,
        similarity_threshold: float = 85.0,
        strategy: str = "graph",
        model=None,
        tokenizer=None,
    ):
        """Initialize merger.

        Args:
            strategy: "graph" for predicate-matching only,
                "model" for graph + LLM semantic resolution.
        """
        self.similarity_threshold = similarity_threshold
        self.strategy = strategy
        self.graph = nx.MultiDiGraph()
        self.model = model
        self.tokenizer = tokenizer
        self.contradictions_resolved = []  # log of resolved contradictions

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
        """Insert or update a relation edge.

        Handles three cases:
        1. Same (subject, predicate, object) — reinforcement: bump recurrence.
        2. Same (subject, predicate) but different object — contradiction:
           remove old edge, add new one. New facts win.
        3. New (subject, predicate) — insert new edge.
        """
        normalized_pred = _normalize_predicate(relation.predicate)

        # Case 1: exact same triple already exists
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
            return

        # Case 2: same (subject, predicate) but different object — contradiction
        graph_resolved = False
        if self.graph.has_node(subject):
            for old_obj in list(self.graph.successors(subject)):
                if old_obj == obj:
                    continue
                keys_to_remove = [
                    key
                    for key, data in self.graph[subject][old_obj].items()
                    if data.get("predicate") == normalized_pred
                ]
                for key in keys_to_remove:
                    old_val = old_obj
                    self.graph.remove_edge(subject, old_obj, key=key)
                    self.contradictions_resolved.append(
                        {
                            "method": "graph",
                            "subject": subject,
                            "old_predicate": normalized_pred,
                            "old_object": old_val,
                            "new_predicate": normalized_pred,
                            "new_object": obj,
                            "session": session_id,
                        }
                    )
                    logger.info(
                        "Contradiction resolved (graph): %s | %s | %s → %s (session %s)",
                        subject,
                        normalized_pred,
                        old_val,
                        obj,
                        session_id,
                    )
                    graph_resolved = True

        # Case 2b: model-based semantic contradiction detection
        # Catches cases like moved_to vs lives_in that graph matching misses
        if (
            not graph_resolved
            and self.strategy == "model"
            and self.model is not None
            and self.graph.has_node(subject)
        ):
            existing_triples = []
            for succ in self.graph.successors(subject):
                for _, data in self.graph[subject][succ].items():
                    existing_triples.append((subject, data["predicate"], succ))

            if existing_triples:
                contradicted = detect_contradiction_with_model(
                    subject,
                    normalized_pred,
                    obj,
                    existing_triples,
                    self.model,
                    self.tokenizer,
                )
                if contradicted is not None:
                    old_s, old_p, old_o = contradicted
                    # Find and remove the contradicted edge
                    old_p_norm = _normalize_predicate(old_p)
                    for old_obj in list(self.graph.successors(subject)):
                        keys_to_remove = [
                            key
                            for key, data in self.graph[subject][old_obj].items()
                            if data.get("predicate") == old_p_norm
                        ]
                        for key in keys_to_remove:
                            self.graph.remove_edge(subject, old_obj, key=key)
                            self.contradictions_resolved.append(
                                {
                                    "method": "model",
                                    "subject": subject,
                                    "old_predicate": old_p_norm,
                                    "old_object": old_obj,
                                    "new_predicate": normalized_pred,
                                    "new_object": obj,
                                    "session": session_id,
                                }
                            )
                            logger.info(
                                "Contradiction resolved (model): "
                                "%s | %s | %s → %s | %s (session %s)",
                                subject,
                                old_p_norm,
                                old_obj,
                                normalized_pred,
                                obj,
                                session_id,
                            )

        # Case 3 (and after contradiction cleanup): add new edge
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

    def get_all_triples(self) -> list[tuple[str, str, str]]:
        """Return all (subject, predicate, object) triples from the graph."""
        triples = []
        for subject, obj, data in self.graph.edges(data=True):
            triples.append((subject, data.get("predicate", "related_to"), obj))
        return triples

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
