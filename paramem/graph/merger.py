"""Knowledge graph merging with entity resolution and cardinality resolution.

Contradiction resolution:
- Same-predicate, different-object cardinality resolution: the model returns one of two
  verdicts (COEXIST / REPLACE) for each same-(subject, predicate)/different-object pair.
  Cardinality judgment is cached per predicate (one model call per unique predicate).
  Active whenever a model is present; unaffected by ``cross_predicate_contradiction``.
- COEXIST: both values are independent and multi-valued; keep both edges.
- REPLACE: new value supersedes old (single-valued predicate); old edge removed.
  At fold time (``additive=True``), REPLACE is skipped — folds are purely additive and
  never remove a registered edge.
- Cross-predicate contradiction detection (REPLACE across different predicates):
  OFF by default (``cross_predicate_contradiction=False``) because it over-removes
  legitimate multi-valued and independent facts (observed over-removing valid facts in live use).
  Enable only when the operator has verified it is safe for their knowledge domain.
- When no model is present (experiments, after release): all triples coexist (no removal).
"""

import json
import logging
from pathlib import Path

import networkx as nx
from rapidfuzz import fuzz

from paramem.graph.name_match import canonical as canonical_id
from paramem.graph.prompts import _load_prompt
from paramem.graph.schema import Entity, Relation, SessionGraph

logger = logging.getLogger(__name__)

_COEXISTENCE_PROMPT = """\
Classify the relationship between two values for the predicate "{predicate}".

--- COEXIST ---
Both values can be true simultaneously (multi-valued predicate).
POSITIVE: Alex has_pet cat, and now Alex has_pet dog → COEXIST
POSITIVE: Alex speaks German, and now Alex speaks French → COEXIST

--- REPLACE ---
The new value supersedes the old one (single-valued predicate).
POSITIVE: Alex lives_in Munich, and now Alex lives_in Berlin → REPLACE
POSITIVE: Alex date_of_birth 1990, and now Alex date_of_birth 1991 → REPLACE

--- CLASSIFY ---
Subject: {subject}
Predicate: {predicate}
Old value: {old_value}
New value: {new_value}

Reply with EXACTLY one of:
- COEXIST
- REPLACE"""

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
    prompt: str = _CONTRADICTION_PROMPT,
) -> tuple[str, str, str] | None:
    """Use the model to detect if a new triple contradicts an existing one.

    Args:
        subject: Subject of the new triple.
        predicate: Predicate of the new triple.
        obj: Object of the new triple.
        existing_triples: All triples currently in the graph.
        model: LLM to use for contradiction detection.
        tokenizer: Tokenizer paired with *model*.
        prompt: Prompt template with ``{existing_facts}``, ``{subject}``,
            ``{predicate}``, and ``{object}`` slots.  Defaults to
            ``_CONTRADICTION_PROMPT``; pass a custom string (e.g. loaded via
            ``_load_prompt``) to override without code changes.

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
    prompt = prompt.format(
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
    prompt: str = _COEXISTENCE_PROMPT,
) -> str:
    """Ask the model whether two values for the same predicate can coexist.

    Returns the verdict string:

    - ``"COEXIST"`` — values are independent; keep both edges.
    - ``"REPLACE"`` — new value supersedes old; retire old edge.

    The default on an ambiguous response is ``"COEXIST"`` (safer —
    do not lose data).

    Args:
        subject: Entity whose predicate cardinality is being judged.
        predicate: The predicate shared by both values.
        old_value: The existing object value for the predicate.
        new_value: The incoming object value for the predicate.
        model: LLM to use for the cardinality judgment.
        tokenizer: Tokenizer paired with *model*.
        prompt: Prompt template with ``{subject}``, ``{predicate}``,
            ``{old_value}``, and ``{new_value}`` slots.  Defaults to
            ``_COEXISTENCE_PROMPT``; pass a custom string (e.g. loaded via
            ``_load_prompt``) to override without code changes.
    """
    from paramem.evaluation.recall import generate_answer
    from paramem.models.loader import adapt_messages

    prompt = prompt.format(
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
        max_new_tokens=32,
        temperature=0.0,
    )

    first_line = output.strip().split("\n")[0].strip()
    upper_line = first_line.upper()

    if "COEXIST" in upper_line:
        return "COEXIST"
    if "REPLACE" in upper_line:
        return "REPLACE"
    # Default to coexistence (safer — don't lose data)
    logger.warning(
        "Ambiguous coexistence response for '%s': '%s', defaulting to COEXIST",
        predicate,
        output.strip(),
    )
    return "COEXIST"


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
        cross_predicate_contradiction: bool = False,
        prompts_dir: str | Path | None = None,
    ):
        """Initialize merger.

        Contradiction resolution operates in two independent layers:

        Same-predicate, different-object cardinality resolution: always-on when
        a model is present.  The cardinality of each predicate (single-valued →
        REPLACE, multi-valued → COEXIST) is determined by one model inference
        call and cached for the lifetime of this merger instance.  When
        ``model`` is ``None`` all same-(subject, predicate)/different-object
        pairs coexist (no removal) — this is the experiment path and the
        post-release state.

        Cross-predicate contradiction detection: OFF by default
        (``cross_predicate_contradiction=False``) because it over-removes
        legitimate multi-valued facts (multiple valid values for one relation)
        and independent facts expressed under different predicates.  Set
        ``cross_predicate_contradiction=True`` only after verifying it is safe
        for a specific knowledge domain.

        Args:
            similarity_threshold: Minimum rapidfuzz token_sort_ratio score (0–100)
                for the fuzzy tier of entity resolution.
            model: Optional LLM for model-based contradiction resolution.
                When set, this merger is a BASE-MODEL HOLDER; call
                :meth:`release` to drop the reference.
            tokenizer: Tokenizer paired with *model*.
            cross_predicate_contradiction: Enable cross-predicate contradiction
                detection via ``detect_contradiction_with_model``.  Default OFF
                — over-removes legitimate multi-valued/independent facts.
                Same-predicate cardinality resolution is unaffected by this flag.
            prompts_dir: Optional directory to load ``merger_coexistence.txt`` and
                ``merger_contradiction.txt`` from.  Falls back to
                ``configs/prompts/`` in the project root, then to the inline
                constants ``_COEXISTENCE_PROMPT`` / ``_CONTRADICTION_PROMPT``.
                Resolved once at construction so a config edit takes effect at the
                next consolidation cycle (when a new merger instance is created).
        """
        self.similarity_threshold = similarity_threshold
        self.graph = nx.MultiDiGraph()
        self.model = model
        self.tokenizer = tokenizer
        self.cross_predicate_contradiction = cross_predicate_contradiction
        self.contradictions_resolved = []  # log of resolved contradictions
        # Per-merge/fold output lists — also initialised here so _upsert_relation
        # is safe to call without a preceding merge() call (e.g. in unit tests).
        self.reinforcements: list[str] = []
        # collapsed: incoming ik_keys that were deduplicated away in a Case-1
        # duplicate-SPO collapse.  Parallel to reinforcements (which records the
        # surviving key); collapsed records the drifting key.  Used by the
        # drift-accounting site in consolidation to distinguish intended dedup
        # from genuine reconstruction loss.
        self.collapsed: list[str] = []
        # removal_ledger: records every edge REMOVAL keyed by the removed edge's
        # ik_key, with a stable reason code.  Complements collapsed/reinforcements
        # (different lifetimes — collapsed/reinforcements carry soft-stale and
        # recurrence semantics; the ledger carries removal cause for drift
        # accounting and future contradiction observability).  Reset in
        # reset_graph(), NOT in merge() — must survive the fold's
        # reset_graph→re-merge→enrich→classify span.
        # reason ∈ {"dedup", "contradiction_same_pred", "contradiction_cross_pred",
        #            "enrichment_same_as"}
        self.removal_ledger: dict[str, dict] = {}
        # Cache: predicate → True (multi-valued/coexist) or False (single-valued/replace)
        self._predicate_cardinality: dict[str, bool] = {}
        # Resolve prompts once at construction so a file edit takes effect next cycle.
        _pd = Path(prompts_dir) if prompts_dir else None
        self._coexistence_prompt = _load_prompt("merger_coexistence.txt", _COEXISTENCE_PROMPT, _pd)
        self._contradiction_prompt = _load_prompt(
            "merger_contradiction.txt", _CONTRADICTION_PROMPT, _pd
        )

    def merge(self, session_graph: SessionGraph, *, additive: bool = False) -> nx.MultiDiGraph:
        """Merge a session graph into the cumulative graph.

        Args:
            session_graph: The per-session graph to merge in.
            additive: When ``True`` (fold-only path), Case-2 same-predicate
                cardinality resolution is short-circuited — no model call, no
                edge removal.  Every reconstructed edge is inserted as a new
                edge regardless of existing same-(subject, predicate) edges.
                This makes full-consolidation folds purely additive and lossless
                with respect to registered edges.  Defaults to ``False`` so
                normal per-session live-ingest REPLACE is unaffected.

        Returns the updated cumulative graph.

        ``self.reinforcements`` is a list of surviving ``ik_key`` strings for
        each Case-1 exact-duplicate collapse that fired during this merge — i.e.
        every fold where an incoming ``Relation.indexed_key`` matched an
        existing edge with an ``ik_key`` already stamped.  The surviving key is
        the existing edge's key (the incoming duplicate drifts).  Only populated
        when ``Relation.indexed_key`` is set on the incoming relation (fold-only
        path); always empty during normal live ingest where
        ``Relation.indexed_key is None``.

        ``self.collapsed`` is the parallel list of INCOMING ``ik_key`` strings
        that were deduplicated away in each such collapse.  Where
        ``reinforcements`` records the surviving key, ``collapsed`` records the
        key whose edge was dropped.  Used by the drift-accounting site in
        ``consolidate_interim_adapters`` to distinguish intended dedup (fact
        preserved under the twin key) from genuine reconstruction loss.
        """
        self.reinforcements: list[str] = []
        self.collapsed: list[str] = []

        session_id = session_graph.session_id
        timestamp = session_graph.timestamp

        # Merge entities
        entity_name_map = {}  # session entity name -> canonical node key in graph
        for entity in session_graph.entities:
            node_key = self._resolve_entity(entity)
            entity_name_map[entity.name] = node_key
            self._upsert_entity(node_key, entity, session_id, timestamp)

        # Merge relations
        for relation in session_graph.relations:
            subj_surface = relation.subject
            obj_surface = relation.object
            subject = entity_name_map.get(subj_surface, canonical_id(subj_surface))
            obj = entity_name_map.get(obj_surface, canonical_id(obj_surface))

            # Build a display-name map for endpoints not resolved through entities.
            # Keys in entity_name_map already have _upsert_entity called for them
            # (which writes attributes["name"]).  Remaining endpoints are raw relation
            # endpoints that arrived without a corresponding Entity; they need the
            # surface form stashed so downstream display reads work.
            _endpoint_display: dict[str, str] = {}
            if subj_surface not in entity_name_map:
                _endpoint_display[subject] = subj_surface
            if obj_surface not in entity_name_map:
                _endpoint_display[obj] = obj_surface

            # Ensure both endpoints exist as nodes
            for name in (subject, obj):
                if name not in self.graph:
                    self.graph.add_node(
                        name,
                        entity_type="concept",
                        attributes={"name": _endpoint_display[name]},
                        first_seen=session_id,
                        last_seen=session_id,
                        recurrence_count=1,
                        sessions=[session_id],
                    )
                elif name in _endpoint_display:
                    # Node already exists but display name not yet set (first-seen wins).
                    node_attrs = self.graph.nodes[name].get("attributes", {})
                    if not node_attrs.get("name"):
                        node_attrs["name"] = _endpoint_display[name]
                        self.graph.nodes[name]["attributes"] = node_attrs

            self._upsert_relation(subject, obj, relation, session_id, timestamp, additive=additive)

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

          1. Exact canonical key match: ``canonical_id(entity.name)``
             against existing node keys (with node-key model A, every
             node key IS the canonical form, so a direct lookup suffices).
          2. Fuzzy ``rapidfuzz.token_sort_ratio`` at
             ``similarity_threshold``, same ``entity_type`` only.
             Comparison uses ``canonical_id`` on both sides.

          Returns the existing node key on a hit, or
          ``canonical_id(entity.name)`` (a new node key) on a miss.
        """
        # Speaker entities: canonical key IS the speaker_id.
        if entity.speaker_id is not None:
            return entity.speaker_id

        entity_canonical = canonical_id(entity.name)

        # Tier 1: Exact match on canonical node key (non-speaker entities).
        # With node-key model A the node key IS the canonical form, so a direct
        # lookup suffices; the old _normalize_name scan is replaced by key lookup.
        if entity_canonical in self.graph:
            return entity_canonical

        # Tier 2: Fuzzy match (non-speaker entities).
        fuzzy_best: str | None = None
        fuzzy_score: float = 0.0
        for node in self.graph.nodes:
            node_data = self.graph.nodes[node]
            # Only match against same entity type
            if node_data.get("entity_type") != entity.entity_type:
                continue
            score = fuzz.token_sort_ratio(entity_canonical, canonical_id(node))
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

        return entity_canonical

    def _upsert_entity(
        self,
        node_key: str,
        entity: Entity,
        session_id: str,
        timestamp: str,
    ) -> None:
        """Insert or update an entity node.

        With node-key model A, every node is keyed by its canonical form
        (``canonical_id(name)`` for non-speakers; ``speaker_id`` for
        speakers).  The human-readable display name is stored as a mutable
        attribute under ``attributes["name"]`` for ALL node types — not just
        speakers — so downstream consumers never need to use the node key for
        display.  First-seen surface wins: the ``"name"`` attribute is set on
        insertion and NOT overwritten on subsequent updates (idempotent).

        Speaker entities (``entity.speaker_id`` set) are keyed by
        ``speaker_id`` (see :meth:`_resolve_entity`).  Anonymous→disclosed
        name change (``"Speaker0"`` → ``"Alex"``) updates
        ``attributes["name"]`` on the existing ``speaker_id``-keyed node —
        no separate node is created.  For speakers the ``"name"`` attribute
        IS refreshed on update (to capture disclosure), whereas for
        non-speaker entities it is first-seen-wins only.
        """
        is_speaker = entity.speaker_id is not None

        if node_key in self.graph:
            node = self.graph.nodes[node_key]
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
            # All entities: store display name in attributes["name"].
            # For speaker entities, refresh on update (captures anonymous→
            # disclosed name change, e.g. "Speaker0" → "Alex").
            # For non-speaker entities, first-seen wins — only write when
            # the attribute is absent or empty.
            if entity.name and not _attr_value_is_empty(entity.name):
                if is_speaker:
                    existing_attrs["name"] = entity.name
                elif not existing_attrs.get("name"):
                    existing_attrs["name"] = entity.name
            node["attributes"] = existing_attrs
            # Speaker_id should already be set on this node (it WAS the
            # canonical key).  Defensive: if a legacy node from before
            # this refactor lacks the field, populate it.
            if is_speaker and node.get("speaker_id") is None:
                node["speaker_id"] = entity.speaker_id
        else:
            attributes = {k: v for k, v in entity.attributes.items() if not _attr_value_is_empty(v)}
            # Store display name for all entities.  The node key is now the
            # canonical form so the node ID is no longer the display name.
            if entity.name and not _attr_value_is_empty(entity.name):
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
            self.graph.add_node(node_key, **node_kwargs)

    def _upsert_relation(
        self,
        subject: str,
        obj: str,
        relation: Relation,
        session_id: str,
        timestamp: str,
        *,
        additive: bool = False,
    ) -> None:
        """Insert or update a relation edge.

        Handles three cases:

        1. Identical triple already exists — exact-duplicate reinforcement: bump
           recurrence.  Case-1-adopt: if the existing edge has no ``ik_key`` and
           the incoming ``relation.indexed_key`` is set, adopt the key onto the
           existing edge (fold-only provenance carry-through).
        2. Same (subject, predicate) but different object — 2-way cardinality
           resolution when a model is present and ``additive=False`` (live ingest):

           - ``COEXIST``: both values are independent; both edges kept.
           - ``REPLACE``: new value supersedes old; old edge removed.

           When ``additive=True`` (fold-only), Case-2 is short-circuited: no
           model call, no edge removal.  The incoming edge is inserted as a new
           edge (Case 3) so folds are purely additive and never remove a
           registered edge.

           Cardinality (COEXIST vs REPLACE axis) is cached per predicate.
        3. New (subject, predicate, object) — net-new edge insertion.  The
           ``ik_key`` from ``relation.indexed_key`` is stamped on the edge when
           set (fold provenance; None at normal ingest — no-op).
        """
        from paramem.memory.persistence import _IK_KEY_ATTR

        normalized_pred = canonical_id(relation.predicate)

        # --- Case 1: Exact-duplicate reinforcement ---
        # Identical (subject, norm_pred, obj) already exists — bump recurrence.
        # Case-1-adopt: if the existing edge has no ik_key and the incoming
        # relation carries one, stamp it onto the existing edge (fold-only; no-op
        # for normal ingest where relation.indexed_key is None).
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
            # Case-1-adopt: adopt incoming ik_key onto a keyless existing edge.
            if relation.indexed_key and not edge.get(_IK_KEY_ATTR):
                edge[_IK_KEY_ATTR] = relation.indexed_key
            # Case-1 reinforcement: when BOTH the existing edge AND the incoming
            # relation carry an ik_key, this is a fold-time duplicate-SPO collapse.
            # The SURVIVING key is the existing edge's ik_key (the incoming
            # relation.indexed_key drifts).  Record the survivor for the fold's
            # bump_recurrence pass.
            # NOTE: existing_key is the NetworkX integer edge id, NOT the key
            # string.  The survivor's key string is read from the edge attribute.
            elif relation.indexed_key and edge.get(_IK_KEY_ATTR):
                surviving_ik = edge.get(_IK_KEY_ATTR)
                if surviving_ik:
                    self.reinforcements.append(surviving_ik)
                # Record the incoming (drifting) key so the drift-accounting site
                # in consolidate_interim_adapters can distinguish intended dedup
                # (fact preserved under the surviving twin) from genuine loss.
                self.collapsed.append(relation.indexed_key)
                # §4.8.i — raw-surface evidence for dedup collapses.
                # The incoming raw surfaces (relation.*, only .strip()ed at
                # extraction) are recorded alongside the surviving twin's
                # first-seen surfaces (stored in node attributes["name"] and
                # edge["predicate"]).  pre_surfaces is the ground-truth record
                # of what was discarded vs. what survived; readers compare
                # incoming vs. surviving directly rather than relying on a
                # derived boolean.
                _surviving_subj_surface = (
                    self.graph.nodes.get(subject, {}).get("attributes", {}).get("name", subject)
                )
                _surviving_pred_surface = edge.get("predicate", "")
                _surviving_obj_surface = (
                    self.graph.nodes.get(obj, {}).get("attributes", {}).get("name", obj)
                )
                self.removal_ledger[relation.indexed_key] = {
                    "reason": "dedup",
                    "surviving_twin": surviving_ik,
                    "pre_surfaces": {
                        "incoming": {
                            "subject": relation.subject,
                            "predicate": relation.predicate,
                            "object": relation.object,
                        },
                        "surviving": {
                            "subject": _surviving_subj_surface,
                            "predicate": _surviving_pred_surface,
                            "object": _surviving_obj_surface,
                        },
                    },
                }
            return None

        # --- Case 2: Same-predicate, different-object cardinality resolution ---
        # At live ingest (additive=False): ask the model for a 2-way verdict
        # (COEXIST / REPLACE).  Cardinality judgment is cached per predicate.
        # At fold time (additive=True): skip entirely — folds are purely additive
        # and never remove a registered edge; fall through to new-edge insertion.
        # When no model is present, fall through to new-edge insertion (coexist-all).
        graph_resolved = False

        if not additive and self.model is not None and self.graph.has_node(subject):
            for old_obj in list(self.graph.successors(subject)):
                if old_obj == obj:
                    continue
                old_edges_with_pred = [
                    (key, data)
                    for key, data in self.graph[subject][old_obj].items()
                    if data.get("predicate") == normalized_pred
                ]
                if not old_edges_with_pred:
                    continue

                # Determine cardinality (cached per predicate).
                if normalized_pred not in self._predicate_cardinality:
                    verdict = check_predicate_coexistence(
                        subject,
                        normalized_pred,
                        old_obj,
                        obj,
                        self.model,
                        self.tokenizer,
                        self._coexistence_prompt,
                    )
                    # Cache the COEXIST/REPLACE axis (True = multi-valued).
                    if verdict == "REPLACE":
                        self._predicate_cardinality[normalized_pred] = False
                    else:
                        # COEXIST → multi-valued axis
                        self._predicate_cardinality[normalized_pred] = True
                    _card_label = (
                        "multi-valued"
                        if self._predicate_cardinality[normalized_pred]
                        else "single-valued"
                    )
                    logger.info(
                        "Predicate cardinality: %s → %s (verdict=%s)",
                        normalized_pred,
                        _card_label,
                        verdict,
                    )

                if not self._predicate_cardinality[normalized_pred]:
                    # Single-valued (REPLACE): remove old edges, insert new one below.
                    for key, _ in old_edges_with_pred:
                        _removed_ik = (
                            self.graph.get_edge_data(subject, old_obj, key=key) or {}
                        ).get(_IK_KEY_ATTR)
                        if _removed_ik:
                            self.removal_ledger[_removed_ik] = {
                                "reason": "contradiction_same_pred",
                                "old_object": old_obj,
                                "new_object": obj,
                            }
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
                # COEXIST: fall through to new-edge insertion.

        # --- Cross-predicate contradiction detection (model-based) ---
        # Catches cases like moved_to vs lives_in that same-predicate matching misses.
        # Gated by self.cross_predicate_contradiction (default False) because it
        # over-removes legitimate multi-valued and independent facts (observed in
        # live use).  Same-predicate cardinality resolution above is unaffected by this flag.
        if (
            self.cross_predicate_contradiction
            and not graph_resolved
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
                    self._contradiction_prompt,
                )
                if contradicted is not None:
                    old_s, old_p, old_o = contradicted
                    # Find and remove the contradicted edge
                    old_p_norm = canonical_id(old_p)
                    for old_obj in list(self.graph.successors(subject)):
                        keys_to_remove = [
                            key
                            for key, data in self.graph[subject][old_obj].items()
                            if data.get("predicate") == old_p_norm
                        ]
                        for key in keys_to_remove:
                            _removed_ik = (
                                self.graph.get_edge_data(subject, old_obj, key=key) or {}
                            ).get(_IK_KEY_ATTR)
                            if _removed_ik:
                                self.removal_ledger[_removed_ik] = {
                                    "reason": "contradiction_cross_pred",
                                    "old_object": old_obj,
                                    "old_predicate": old_p_norm,
                                    "new_object": obj,
                                    "new_predicate": normalized_pred,
                                }
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

        # --- Case 3: New-edge insertion ---
        # After contradiction cleanup or when no same-pred edge exists.
        # Stamp ik_key from relation.indexed_key when set (fold-only; None = no-op).
        new_eid = self.graph.add_edge(
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
        if relation.indexed_key:
            self.graph[subject][obj][new_eid][_IK_KEY_ATTR] = relation.indexed_key
        return None

    def reset_graph(self) -> None:
        """Reset the keying surface to an empty graph, clearing per-fold caches.

        Called by ``consolidate_interim_adapters`` BEFORE the Stage-2 re-merge
        so the keying surface is empty and Option-(a) provenance keying is
        unconditional: reconstructed-key edges are always net-new (Case 3) or
        intra-fold-collapsed (Case 1 among recon edges), with no dependence on
        any pre-existing edge state.

        Cleared caches:
        - ``graph`` — fresh MultiDiGraph (no prior edges/nodes)
        - ``_predicate_cardinality`` — per-predicate COEXIST/REPLACE cache
        - ``contradictions_resolved`` — log of prior resolves
        - ``reinforcements`` — prior fold's Case-1 surviving keys
        - ``collapsed`` — prior fold's Case-1 deduplicated (incoming) keys
        - ``removal_ledger`` — prior fold's reason-coded edge removal records

        Does NOT touch ``model``, ``tokenizer``, or the prompt strings — those
        are construction-time state and must survive across folds.
        """
        import networkx as nx

        self.graph = nx.MultiDiGraph()
        self._predicate_cardinality = {}
        self.contradictions_resolved = []
        self.reinforcements = []
        self.collapsed = []
        self.removal_ledger = {}

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
