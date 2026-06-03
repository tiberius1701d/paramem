"""Knowledge graph merging with entity resolution and edge aggregation.

Contradiction resolution is model-only and always-on when a model is present:
- Single-valued predicate (REPLACE): the new value removes the old edge.
- Multi-valued predicate (COEXIST): both values coexist, no removal.
- Cardinality judgment is cached per predicate (one model call per unique predicate).
- When no model is present (experiments, after release): all triples coexist (no removal).
"""

import json
import logging
from pathlib import Path

import networkx as nx
from rapidfuzz import fuzz

from paramem.graph.schema import Entity, Relation, SessionGraph

logger = logging.getLogger(__name__)

_COEXISTENCE_PROMPT = """\
Can a person have more than one value for the relationship "{predicate}"?

Example: {subject} {predicate} {old_value}, and now {subject} {predicate} {new_value}.

Can both be true at the same time, or does the new value replace the old one?

Reply with EXACTLY one word:
- COEXIST (both can be true, e.g. a person can have multiple pets)
- REPLACE (only one can be true, e.g. a person has one date of birth)"""

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


def check_predicate_coexistence(
    subject: str,
    predicate: str,
    old_value: str,
    new_value: str,
    model,
    tokenizer,
) -> bool:
    """Ask the model whether two values for the same predicate can coexist.

    Returns True if both values can be true simultaneously (multi-valued),
    False if the new value replaces the old one (single-valued).
    """
    from paramem.evaluation.recall import generate_answer
    from paramem.models.loader import adapt_messages

    prompt = _COEXISTENCE_PROMPT.format(
        subject=subject,
        predicate=predicate,
        old_value=old_value,
        new_value=new_value,
    )

    messages = adapt_messages(
        [
            {"role": "system", "content": "You classify relationship cardinality."},
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
        max_new_tokens=16,
        temperature=0.0,
    )

    decision = output.strip().upper()
    if "COEXIST" in decision:
        return True
    if "REPLACE" in decision:
        return False
    # Default to coexistence (safer — don't lose data)
    logger.warning(
        "Ambiguous coexistence response for '%s': '%s', defaulting to COEXIST",
        predicate,
        output.strip(),
    )
    return True


class GraphMerger:
    """Merges per-session graphs into a cumulative knowledge graph.

    Handles entity resolution (exact + fuzzy matching), edge aggregation
    with recurrence counting, and JSON persistence via NetworkX.
    """

    def __init__(
        self,
        similarity_threshold: float = 85.0,
        model=None,
        tokenizer=None,
    ):
        """Initialize merger.

        Contradiction resolution is model-only and always-on when a model is
        present.  The cardinality of each predicate (single-valued → REPLACE,
        multi-valued → COEXIST) is determined by one model inference call and
        cached for the lifetime of this merger instance.  When ``model`` is
        ``None`` all same-(subject, predicate)/different-object pairs coexist
        (no removal) — this is the experiment path and the post-release state.

        Args:
            similarity_threshold: Minimum rapidfuzz token_sort_ratio score (0–100)
                for the fuzzy tier of entity resolution.
            model: Optional LLM for model-based contradiction resolution.
                When set, this merger is a BASE-MODEL HOLDER; call
                :meth:`release` to drop the reference.
            tokenizer: Tokenizer paired with *model*.
        """
        self.similarity_threshold = similarity_threshold
        self.graph = nx.MultiDiGraph()
        self.model = model
        self.tokenizer = tokenizer
        self.contradictions_resolved = []  # log of resolved contradictions
        # Cache: predicate → True (multi-valued/coexist) or False (single-valued/replace)
        self._predicate_cardinality: dict[str, bool] = {}

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
        """Resolve an entity to its canonical NetworkX node key.

        Speaker entities are first-class graph roots: when
        ``entity.speaker_id`` is set, the canonical key **is** the
        speaker_id (an immutable system identifier such as
        ``"Speaker0"``).  Display name lives at
        ``node_data["attributes"]["name"]``.  This is the architecture
        documented in :class:`paramem.graph.schema.Entity` —
        ``speaker_id`` is the canonical graph identity; ``name`` is a
        mutable display attribute.

        Two enrolled speakers who share a display name (e.g. both
        ``"Alex"``) keep separate graph nodes because their
        ``speaker_id`` values differ by construction.  Name changes
        across sessions for the same speaker (e.g. anonymous
        ``Speaker0`` → disclosed ``Alex``) collapse onto the same
        speaker_id node.  Third-party mentions (no ``speaker_id``)
        live in the **name namespace** which is disjoint from speaker
        IDs by construction (speaker IDs follow ``Speaker_N``).

        Resolution rules:

        * **Speaker entity** (``entity.speaker_id`` set) — canonical key
          IS the speaker_id.  No name-based matching; no fuzzy match.
          The display name is stored as a mutable attribute downstream.
        * **Non-speaker entity** (``entity.speaker_id is None``) —
          two-tier name resolution:

          1. Exact normalised name match.
          2. Fuzzy ``rapidfuzz.token_sort_ratio`` at
             ``similarity_threshold``, same ``entity_type`` only.

          Returns the existing canonical key on a hit, or
          ``entity.name`` (a new node) on a miss.
        """
        # Speaker entities: canonical key IS the speaker_id.
        if entity.speaker_id is not None:
            return entity.speaker_id

        normalized = _normalize_name(entity.name)

        # Tier 1: Exact match on normalised names (non-speaker entities).
        for node in self.graph.nodes:
            if _normalize_name(node) == normalized:
                return node

        # Tier 2: Fuzzy match (non-speaker entities).
        fuzzy_best: str | None = None
        fuzzy_score: float = 0.0
        for node in self.graph.nodes:
            node_data = self.graph.nodes[node]
            # Only match against same entity type
            if node_data.get("entity_type") != entity.entity_type:
                continue
            score = fuzz.token_sort_ratio(normalized, _normalize_name(node))
            if score > fuzzy_score and score >= self.similarity_threshold:
                fuzzy_score = score
                fuzzy_best = node

        if fuzzy_best is not None:
            logger.debug(
                "Fuzzy matched '%s' -> '%s' (score=%.1f, method=fuzzy)",
                entity.name,
                fuzzy_best,
                fuzzy_score,
            )
            return fuzzy_best

        return entity.name

    def _upsert_entity(
        self,
        canonical: str,
        entity: Entity,
        session_id: str,
        timestamp: str,
    ) -> None:
        """Insert or update an entity node.

        Speaker entities (``entity.speaker_id`` set) are keyed by
        ``speaker_id`` (see :meth:`_resolve_entity`).  The display name
        from ``entity.name`` is stored as a mutable attribute under
        ``attributes["name"]`` so consumers that need a human-readable
        label can read it without ever using the NetworkX node ID for
        display.  Anonymous→disclosed name change (``"Speaker0"`` →
        ``"Alex"``) updates ``attributes["name"]`` on the existing
        ``speaker_id``-keyed node — no separate node is created.

        Non-speaker entities are keyed by name; ``attributes["name"]``
        is not written because the node ID is already the display
        name.
        """
        is_speaker = entity.speaker_id is not None

        if canonical in self.graph:
            node = self.graph.nodes[canonical]
            node["recurrence_count"] = node.get("recurrence_count", 0) + 1
            node["last_seen"] = session_id
            sessions = node.get("sessions", [])
            if session_id not in sessions:
                sessions.append(session_id)
            node["sessions"] = sessions
            # Merge attributes — only non-empty values are stored. Empty
            # strings and "N/A"/"None"/"Unknown" placeholders that one
            # chunk's LLM emitted for missing data are dropped: they neither
            # overwrite an existing real value nor introduce a noisy
            # placeholder key into the cumulative graph (a known LLM-
            # compliance failure mode where the extractor enumerates every
            # advertised attribute key even when the source has no value).
            existing_attrs = node.get("attributes", {})
            for k, v in entity.attributes.items():
                if _attr_value_is_empty(v):
                    continue
                existing_attrs[k] = v
            # Speaker entity: refresh the display ``name`` attribute if
            # the latest session carries one.  Older sessions may have
            # used an anonymous id (e.g. ``"Speaker0"``); a later
            # disclosure (``"Alex"``) updates the stored display name.
            if is_speaker and entity.name and not _attr_value_is_empty(entity.name):
                existing_attrs["name"] = entity.name
            node["attributes"] = existing_attrs
            # Speaker_id should already be set on this node (it WAS the
            # canonical key).  Defensive: if a legacy node from before
            # this refactor lacks the field, populate it.
            if is_speaker and node.get("speaker_id") is None:
                node["speaker_id"] = entity.speaker_id
        else:
            attributes = {k: v for k, v in entity.attributes.items() if not _attr_value_is_empty(v)}
            # Speaker entity: stash the display name on the node so it
            # is recoverable without keeping the node ID in display
            # space.  Non-speaker entities skip this — their node ID IS
            # the display name.
            if is_speaker and entity.name and not _attr_value_is_empty(entity.name):
                attributes["name"] = entity.name
            node_kwargs: dict = dict(
                entity_type=entity.entity_type,
                attributes=attributes,
                first_seen=session_id,
                last_seen=session_id,
                recurrence_count=1,
                sessions=[session_id],
            )
            if is_speaker:
                node_kwargs["speaker_id"] = entity.speaker_id
            self.graph.add_node(canonical, **node_kwargs)

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
        2. Same (subject, predicate) but different object — model contradiction
           resolution (always-on when a model is present): ask whether the
           predicate is single-valued (REPLACE, old edge removed) or
           multi-valued (COEXIST, both edges kept).  Cardinality judgment is
           cached per predicate.  When no model is present, fall through to
           Case 3 (coexist-all).
        3. New (subject, predicate, object) — insert new edge.
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

        # Case 2: same (subject, predicate) but different object
        # Ask the model whether both values can coexist (multi-valued)
        # or the new value replaces the old (single-valued/contradiction).
        # Decision is cached per predicate — one inference call per unique predicate.
        graph_resolved = False
        if self.model is not None and self.graph.has_node(subject):
            for old_obj in list(self.graph.successors(subject)):
                if old_obj == obj:
                    continue
                keys_with_same_pred = [
                    key
                    for key, data in self.graph[subject][old_obj].items()
                    if data.get("predicate") == normalized_pred
                ]
                if not keys_with_same_pred:
                    continue

                # Check cardinality (cached per predicate)
                if normalized_pred not in self._predicate_cardinality:
                    can_coexist = check_predicate_coexistence(
                        subject,
                        normalized_pred,
                        old_obj,
                        obj,
                        self.model,
                        self.tokenizer,
                    )
                    self._predicate_cardinality[normalized_pred] = can_coexist
                    logger.info(
                        "Predicate cardinality: %s → %s",
                        normalized_pred,
                        "multi-valued" if can_coexist else "single-valued",
                    )

                if self._predicate_cardinality[normalized_pred]:
                    # Multi-valued: both values coexist, no contradiction
                    continue

                # Single-valued: replace old with new
                for key in keys_with_same_pred:
                    self.graph.remove_edge(subject, old_obj, key=key)
                    self.contradictions_resolved.append(
                        {
                            "method": "model_cardinality",
                            "subject": subject,
                            "old_predicate": normalized_pred,
                            "old_object": old_obj,
                            "new_predicate": normalized_pred,
                            "new_object": obj,
                            "session": session_id,
                        }
                    )
                    logger.info(
                        "Contradiction resolved (cardinality): %s | %s | %s → %s (session %s)",
                        subject,
                        normalized_pred,
                        old_obj,
                        obj,
                        session_id,
                    )
                    graph_resolved = True

        # Case 2b: model-based cross-predicate semantic contradiction detection
        # Catches cases like moved_to vs lives_in that same-predicate matching misses
        if not graph_resolved and self.model is not None and self.graph.has_node(subject):
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

    def release(self) -> None:
        """Drop the base-model reference this merger holds (BASE-MODEL HOLDER).

        Sets ``self.model`` and ``self.tokenizer`` to ``None`` so the base model
        can be freed by the Python garbage collector.  Called by
        ``ConsolidationLoop.release()`` as part of the VRAM-release path.

        Idempotent: safe to call multiple times or when no model was set.
        """
        self.model = None
        self.tokenizer = None

    def get_all_triples(self) -> list[tuple[str, str, str]]:
        """Return all (subject, predicate, object) triples from the graph."""
        triples = []
        for subject, obj, data in self.graph.edges(data=True):
            triples.append((subject, data.get("predicate", "related_to"), obj))
        return triples

    def save_bytes(self) -> bytes:
        """Return the serialized graph as bytes; mirror of save_graph(path) for in-memory consumers.

        Produces the same JSON that save_graph would write to disk, but returns
        the bytes without performing any I/O.  Used by /migration/confirm step 2
        to capture a point-in-time snapshot of the graph for the pre-migration
        backup without requiring a temporary file.

        Returns:
            UTF-8-encoded JSON bytes (node-link format, indent=2).
        """
        data = nx.node_link_data(self.graph)
        return json.dumps(data, indent=2).encode("utf-8")

    def save_graph(self, path: str | Path, *, encrypted: bool = True) -> None:
        """Save cumulative graph to JSON — atomic write, fsynced parent for
        power-loss safety.

        ``encrypted=True`` (default) routes through the infrastructure
        envelope — age under Security ON, plaintext under Security OFF.
        ``encrypted=False`` bypasses the envelope and always writes
        plaintext; used by debug-directory writers so ``debug/*`` output
        is uniformly inspectable with ``cat``/``grep`` regardless of the
        server's Security posture.
        """
        from paramem.backup.encryption import write_infra_bytes, write_plaintext_atomic

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = nx.node_link_data(self.graph)
        payload = json.dumps(data, indent=2).encode("utf-8")
        if encrypted:
            write_infra_bytes(path, payload)
        else:
            write_plaintext_atomic(path, payload)
        logger.info("Graph saved to %s", path)

    def load_graph(self, path: str | Path) -> nx.MultiDiGraph:
        """Load cumulative graph from JSON — transparently decrypts
        age-wrapped content when the daily identity is loaded."""
        from paramem.backup.encryption import read_maybe_encrypted

        path = Path(path)
        if not path.exists():
            logger.info("No existing graph at %s, starting fresh", path)
            return self.graph

        data = json.loads(read_maybe_encrypted(path).decode("utf-8"))
        self.graph = nx.node_link_graph(data)
        logger.info(
            "Graph loaded from %s: %d nodes, %d edges",
            path,
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )
        return self.graph


def _attr_value_is_empty(value) -> bool:
    """True iff an entity-attribute value carries no information.

    Treats the empty string, the literal placeholder strings ``"N/A"`` /
    ``"n/a"`` / ``"None"`` / ``"null"`` (case-insensitive), and ``None``
    as empty.  Used by the attribute-merge step so a non-empty value
    captured in one chunk is never overwritten by an LLM-emitted
    placeholder from another chunk that happened to lack the data.
    """
    if value is None:
        return True
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return True
        if stripped.lower() in ("n/a", "none", "null", "unknown"):
            return True
    return False


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
