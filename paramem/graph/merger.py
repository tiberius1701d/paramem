"""Knowledge graph merging with entity resolution and cardinality resolution.

Contradiction resolution:
- Same-predicate, different-object cardinality resolution: the model returns one of two
  verdicts (COEXIST / REPLACE) for each same-(subject, predicate)/different-object group.
  Cardinality judgment is cached per predicate (one model call per unique predicate).
  Active whenever a model is present and ``resolve_contradictions=True``.
- COEXIST: both values are independent and multi-valued; keep both edges.
- REPLACE (single-valued): recency selection over ``{incoming} ∪ rivals``.
  Rule (applied uniformly at ingest, interim, and fold; no positional fork):
  1. ANY candidate ``last_seen`` is empty (``""``) → COEXIST: insert incoming, remove
     nothing.  Covers legacy timestamp-less keys at fold (all ""), and mixed registries
     where one side is a dated key and the other is a legacy "".
  2. All candidates have datestamps: ``max_ls = max(candidates)``.  Strictly-older rivals
     (``last_seen < max_ls``) are retired and ledgered.  Ties at ``max_ls`` coexist.
     Incoming: if ``incoming_ls == max_ls`` → insert (fall through to Case-3);
     if ``incoming_ls < max_ls`` → incoming loses → NOT inserted, ledgered.
  At fold, the session ``timestamp`` passed to the merger is ``""`` so the fallback
  ``relation.last_seen or timestamp`` yields ``""`` for legacy relations — they hit
  rule 1 (coexist) rather than fabricating a NOW value.
- When no model is present (experiments, after release): all triples coexist (no removal).
"""

import json
import logging
from pathlib import Path

import networkx as nx
from rapidfuzz import fuzz

from paramem.graph.name_match import canonical as canonical_id
from paramem.graph.name_match import is_speaker_id
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
        prompts_dir: str | Path | None = None,
    ):
        """Initialize merger.

        Same-predicate, different-object cardinality resolution: always-on when
        a model is present.  The cardinality of each predicate (single-valued →
        REPLACE, multi-valued → COEXIST) is determined by one model inference
        call and cached for the lifetime of this merger instance.  When
        ``model`` is ``None`` all same-(subject, predicate)/different-object
        pairs coexist (no removal) — this is the experiment path and the
        post-release state.

        Args:
            similarity_threshold: Minimum rapidfuzz token_sort_ratio score (0–100)
                for the fuzzy tier of entity resolution.
            model: Optional LLM for model-based contradiction resolution.
                When set, this merger is a BASE-MODEL HOLDER; call
                :meth:`release` to drop the reference.
            tokenizer: Tokenizer paired with *model*.
            prompts_dir: Optional directory to load ``merger_coexistence.txt``
                from.  Falls back to ``configs/prompts/`` in the project root,
                then to the inline constant ``_COEXISTENCE_PROMPT``.
                Resolved once at construction so a config edit takes effect at the
                next consolidation cycle (when a new merger instance is created).
        """
        self.similarity_threshold = similarity_threshold
        self.graph = nx.MultiDiGraph()
        self.model = model
        self.tokenizer = tokenizer
        self.contradictions_resolved = []  # log of resolved contradictions
        # Per-merge/fold output lists — also initialised here so _upsert_relation
        # is safe to call without a preceding merge() call (e.g. in unit tests).
        self.reinforcements: dict[str, str] = {}
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
        # reason ∈ {"dedup", "contradiction_same_pred",
        #            "enrichment_same_as", "predicate_synonym_collapse", "semantic_dedup"}
        self.removal_ledger: dict[str, dict] = {}
        # Cache: predicate → True (multi-valued/coexist) or False (single-valued/replace)
        self._predicate_cardinality: dict[str, bool] = {}
        # Resolve prompts once at construction so a file edit takes effect next cycle.
        _pd = Path(prompts_dir) if prompts_dir else None
        self._coexistence_prompt = _load_prompt("merger_coexistence.txt", _COEXISTENCE_PROMPT, _pd)

    def merge(
        self,
        session_graph: SessionGraph,
        *,
        resolve_contradictions: bool = True,
    ) -> nx.MultiDiGraph:
        """Merge a session graph into the cumulative graph.

        Args:
            session_graph: The per-session graph to merge in.
            resolve_contradictions: When ``True`` (default), Case-2
                same-predicate/different-object cardinality resolution fires
                when a model is present — the model returns COEXIST or REPLACE.
                For REPLACE (single-valued), recency selection fires: if ANY
                candidate ``last_seen`` is ``""`` → coexist (no removal).
                Otherwise ``max_ls = max(candidates)``; strictly-older rivals
                retired; ties at ``max_ls`` coexist; incoming loses if strictly
                older than ``max_ls`` (NOT inserted).  Applied uniformly at
                ingest, interim, and fold (no positional fork).  At fold the
                session timestamp is ``""`` so legacy relations (``last_seen=""``
                ) coexist rather than fabricating a NOW recency value.
                When ``False``, Case-2 is short-circuited: no model call, no
                edge removal.

        Returns the updated cumulative graph.

        ``self.reinforcements`` is a ``dict[ik_key, last_seen]`` of surviving
        keys for each Case-1 exact-duplicate collapse that fired during this
        merge — i.e. every fold where an incoming ``Relation.indexed_key``
        matched an existing edge with an ``ik_key`` already stamped.  The
        surviving key is the existing edge's key (the incoming duplicate
        drifts); the value is the freshest (``max``) ``last_seen`` timestamp
        after the collapse.  The bump_recurrence site reads this value to
        advance bookkeeping without fabricating ``now()``.  Only populated
        when ``Relation.indexed_key`` is set on the incoming relation
        (fold-only path); always empty during normal live ingest where
        ``Relation.indexed_key is None``.

        ``self.collapsed`` is the parallel list of INCOMING ``ik_key`` strings
        that were deduplicated away in each such collapse.  Where
        ``reinforcements`` records the surviving key, ``collapsed`` records the
        key whose edge was dropped.  Used by the drift-accounting site in
        ``consolidate_interim_adapters`` to distinguish intended dedup (fact
        preserved under the twin key) from genuine reconstruction loss.
        """
        self.reinforcements: dict[str, str] = {}
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
            # Fallback resolution for endpoints not present in entity_name_map:
            # speaker endpoints (is_speaker_id) pass through verbatim — the
            # token is already lowercase (ingest safety-net + mint guarantee).
            # Routing through canonical_id would be semantically wrong (it also
            # folds separators/diacritics) even though speaker ids only contain
            # ASCII alpha+digits and the result would be identical; keep the
            # branch so the path is explicit and not merged with non-speaker ids.
            # Non-speaker endpoints resolve via canonical_id (node-key model A).
            if subj_surface in entity_name_map:
                subject = entity_name_map[subj_surface]
            elif is_speaker_id(subj_surface):
                subject = subj_surface
            else:
                subject = canonical_id(subj_surface)

            if obj_surface in entity_name_map:
                obj = entity_name_map[obj_surface]
            elif is_speaker_id(obj_surface):
                obj = obj_surface
            else:
                obj = canonical_id(obj_surface)

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
                        reinforcement_count=1,
                        sessions=[session_id],
                    )
                elif name in _endpoint_display:
                    # Node already exists but display name not yet set (first-seen wins).
                    node_attrs = self.graph.nodes[name].get("attributes", {})
                    if not node_attrs.get("name"):
                        node_attrs["name"] = _endpoint_display[name]
                        self.graph.nodes[name]["attributes"] = node_attrs

            self._upsert_relation(
                subject,
                obj,
                relation,
                session_id,
                timestamp,
                resolve_contradictions=resolve_contradictions,
            )

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
        ``entity.speaker_id`` is set, the node key IS ``entity.speaker_id``
        verbatim (already lowercase ``speaker{N}`` — guaranteed by the ingest
        safety-net in :func:`~paramem.graph.extractor._normalize_extraction`
        and by :meth:`~paramem.server.speaker.SpeakerStore._mint_anon_speaker_id`).
        No casing step is needed.  Both the entity path and the relation-endpoint
        fallback path produce the same node key, preventing casing-collision dups.
        The display name lives at ``node_data["attributes"]["name"]``; for speakers
        this is the same lowercase ``speaker{N}`` string — resolved to a human
        name at render time by the Phase-B resolver.

        Two enrolled speakers who share a display name (e.g. both ``"Alex"``)
        keep separate graph nodes because their ``speaker_id`` values differ by
        construction.  Name changes across sessions for the same speaker collapse
        onto the same node because the key is the immutable ``speaker{N}`` id.
        Third-party mentions (no ``speaker_id``) live in the name namespace which
        is disjoint from speaker ids (speaker ids follow ``speaker{N}``).

        Resolution rules:

        * **Speaker entity** (``entity.speaker_id`` set) — node key is
          ``entity.speaker_id`` verbatim (lowercase).  No name-based matching;
          no fuzzy match.  The display name is stored as a mutable attribute
          downstream.
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
        # Speaker entities: node key is entity.speaker_id verbatim (lowercase).
        # Both the entity path and the relation-endpoint fallback path produce the
        # same key because both now carry an already-lowercase speaker{N} token.
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
        (``canonical_id(name)`` for non-speakers; ``entity.speaker_id`` verbatim
        for speakers — always lowercase ``speaker{N}``).  The human-readable
        display name is stored as a mutable attribute under ``attributes["name"]``
        for ALL node types — not just speakers — so downstream consumers never need
        to use the node key for display.  First-seen surface wins: the ``"name"``
        attribute is set on insertion and NOT overwritten on subsequent updates
        (idempotent).

        Speaker entities (``entity.speaker_id`` set) are keyed by
        ``entity.speaker_id`` verbatim (lowercase ``speaker{N}``,
        e.g. ``"speaker0"`` — see :meth:`_resolve_entity`).  The node's
        ``speaker_id`` attribute carries the same lowercase id.
        ``attributes["name"]`` stores the same lowercase ``speaker{N}`` id
        and IS refreshed on update.  Display-name resolution happens at the
        fact-render boundary via ``resolve_speaker_name``, not at graph-write time.
        For non-speaker entities ``attributes["name"]`` is first-seen-wins only.
        """
        is_speaker = entity.speaker_id is not None

        if node_key in self.graph:
            node = self.graph.nodes[node_key]
            node["reinforcement_count"] = node.get("reinforcement_count", 0) + 1
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
            # For speaker entities, refresh on update (captures attribute
            # changes; entity.name is always the lowercase speaker{N} id).
            # For non-speaker entities, first-seen wins — only write when
            # the attribute is absent or empty.
            if entity.name and not _attr_value_is_empty(entity.name):
                if is_speaker:
                    existing_attrs["name"] = entity.name
                elif not existing_attrs.get("name"):
                    existing_attrs["name"] = entity.name
            node["attributes"] = existing_attrs
            # The node key equals entity.speaker_id (lowercase speaker{N}).
            # The ``speaker_id`` node attribute carries the same value.
            # Defensive: populate the attribute when it is missing (e.g. a node
            # inserted before this refactor that lacks the attribute).
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
                reinforcement_count=1,
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
        resolve_contradictions: bool = True,
    ) -> None:
        """Insert or update a relation edge.

        Handles three cases:

        1. Identical triple already exists — exact-duplicate reinforcement: bump
           recurrence.  Case-1-adopt: if the existing edge has no ``ik_key`` and
           the incoming ``relation.indexed_key`` is set, adopt the key onto the
           existing edge (fold-only provenance carry-through).
        2. Same (subject, predicate) but different object — cardinality resolution
           when a model is present and ``resolve_contradictions=True``:

           - ``COEXIST``: both values are independent; all edges kept.
           - ``REPLACE`` (single-valued): recency selection over
             ``{incoming} ∪ rivals``.  If ANY candidate ``last_seen`` is ``""``
             → COEXIST (no removal; covers legacy fold keys and mixed
             dated/legacy registries).  Otherwise all are dated: strictly-older
             rivals retired + ledgered; ties at ``max_ls`` coexist; incoming
             NOT inserted (returns ``None``) and ledgered when strictly older
             than ``max_ls``.  ``incoming_ls = relation.last_seen or timestamp``
             (``timestamp=""`` at fold/recon/simulate; ``now()`` at ingest).

           When ``resolve_contradictions=False``, Case-2 is short-circuited:
           no model call, no edge removal.

           Cardinality (COEXIST vs REPLACE axis) is cached per predicate.
        3. New (subject, predicate, object) — net-new edge insertion.  The
           ``ik_key`` from ``relation.indexed_key`` is stamped on the edge when
           set (fold provenance; None at normal ingest — no-op).
        """
        from paramem.memory.persistence import _EDGE_SOURCE_ATTR, _IK_KEY_ATTR

        normalized_pred = canonical_id(relation.predicate)

        # E-2: symmetric direction canonicalization — collapse (A,P,B) / (B,P,A) into
        # a single direction so Case-1 reinforcement deduplicates them.
        # Guard: when BOTH endpoints are speaker nodes (each has a speaker_id), keep
        # both directions distinct (each mints its own key for per-speaker recall).
        both_speakers = bool(self.graph.nodes.get(subject, {}).get("speaker_id")) and bool(
            self.graph.nodes.get(obj, {}).get("speaker_id")
        )
        if relation.symmetric and subject > obj and not both_speakers:
            subject, obj = obj, subject

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
            edge["reinforcement_count"] = edge.get("reinforcement_count", 0) + 1
            edge["last_seen"] = max(edge.get("last_seen", ""), relation.last_seen or timestamp)
            edge["confidence"] = max(edge.get("confidence", 0), relation.confidence)
            sessions = edge.get("sessions", [])
            if session_id not in sessions:
                sessions.append(session_id)
            # Union any per-relation session_ids carried on the Relation object.
            # These are the real contributing session ids stamped at the extraction
            # point; the scalar session_id param may be a synthetic sentinel for
            # fold/re-merge paths.
            for _sid in relation.session_ids:
                if _sid not in sessions:
                    sessions.append(_sid)
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
                    # Record surviving key → freshest last_seen so bump_recurrence
                    # can advance bookkeeping without fabricating now().
                    self.reinforcements[surviving_ik] = edge.get("last_seen", "")
                # Record the incoming (drifting) key so the drift-accounting site
                # in consolidate_interim_adapters can distinguish intended dedup
                # (fact preserved under the surviving twin) from genuine loss.
                self.collapsed.append(relation.indexed_key)
                # Raw-surface evidence for dedup collapses (observability hook).
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
            # A-2 + B-4: first-non-empty-wins adopts for speaker_id and edge_source.
            # Run unconditionally after the ik_key if/elif chain so they never
            # disturb the elif's adopt-vs-reinforce accounting.
            if relation.speaker_id and not edge.get("speaker_id"):
                edge["speaker_id"] = relation.speaker_id
            if relation.edge_source and not edge.get(_EDGE_SOURCE_ATTR):
                edge[_EDGE_SOURCE_ATTR] = relation.edge_source
            return None

        # --- Case 2: Same-predicate, different-object cardinality resolution ---
        # When no model is present or resolve_contradictions=False, fall through
        # to Case-3 insertion (coexist-all / short-circuit).
        # When a model is present and resolve_contradictions=True:
        #   1. Gather ALL rival edges (same subject+predicate, different object).
        #   2. Run ONE cardinality verdict (cached per predicate).
        #   3. COEXIST → fall through.
        #   4. REPLACE → recency selection across {incoming} ∪ rivals:
        #      - incoming_ls = relation.last_seen or timestamp (never empty)
        #      - rival_ls  = each rival's stored last_seen (may be "")
        #      - max_ls = lexicographic max across the full set
        #      - n_at_max >= 2 (empty or tied) → coexist; fall through, no removal
        #      - incoming uniquely freshest → retire ALL rivals, fall through
        #      - rival uniquely freshest → that rival survives; other rivals retired;
        #        incoming is NOT inserted (return None after ledgering its key)
        if resolve_contradictions and self.model is not None and self.graph.has_node(subject):
            # Gather all rival (obj, edge_key, edge_data) triples.
            rivals: list[tuple[str, int, dict]] = []
            for old_obj in list(self.graph.successors(subject)):
                if old_obj == obj:
                    continue
                for key, data in self.graph[subject][old_obj].items():
                    if data.get("predicate") == normalized_pred:
                        rivals.append((old_obj, key, data))

            if rivals:
                # Determine cardinality (cached per predicate).
                if normalized_pred not in self._predicate_cardinality:
                    verdict = check_predicate_coexistence(
                        subject,
                        normalized_pred,
                        rivals[0][0],  # first rival object as old_value
                        obj,
                        self.model,
                        self.tokenizer,
                        self._coexistence_prompt,
                    )
                    # Cache: True = multi-valued (COEXIST), False = single-valued (REPLACE).
                    if verdict == "REPLACE":
                        self._predicate_cardinality[normalized_pred] = False
                    else:
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
                    # Single-valued (REPLACE): recency selection.
                    # incoming_ls: use the relation's own last_seen when set; falls back to
                    # session timestamp ("" at fold/recon/simulate; now() at live ingest).
                    incoming_ls = relation.last_seen or timestamp
                    rival_ls_list = [data.get("last_seen", "") for _, _, data in rivals]

                    if any(ls == "" for ls in [incoming_ls, *rival_ls_list]):
                        # ANY candidate last_seen is empty (unknown recency) → COEXIST.
                        # Covers: legacy timestamp-less keys at fold (all ""), and mixed
                        # registries where a legacy "" key is a rival of a dated incoming
                        # (or vice versa).  Safe no-op: insert incoming, remove nothing.
                        pass
                    else:
                        # All candidates have datestamps; freshest wins.
                        # Strictly-older rivals (last_seen < max_ls) are retired.
                        # Ties at max_ls coexist (rivals AND incoming if at max_ls).
                        max_ls = max([incoming_ls, *rival_ls_list])
                        # winner_obj: representative "superseded by" pointer for ledger
                        # entries on retired edges.  When incoming also wins (at max_ls)
                        # use obj; when incoming loses, use the first rival at max_ls.
                        winner_obj = (
                            obj
                            if incoming_ls == max_ls
                            else next(
                                rv for rv, _, rd in rivals if rd.get("last_seen", "") == max_ls
                            )
                        )
                        for rival_obj, rival_key, rival_data in rivals:
                            if rival_data.get("last_seen", "") >= max_ls:
                                continue  # at max_ls — coexist (tied rivals kept)
                            _removed_ik = rival_data.get(_IK_KEY_ATTR)
                            if _removed_ik:
                                self.removal_ledger[_removed_ik] = {
                                    "reason": "contradiction_same_pred",
                                    "old_object": rival_obj,
                                    "new_object": winner_obj,
                                }
                            self.graph.remove_edge(subject, rival_obj, key=rival_key)
                            self.contradictions_resolved.append(
                                {
                                    "method": "model_cardinality",
                                    "subject": subject,
                                    "old_predicate": normalized_pred,
                                    "old_object": rival_obj,
                                    "new_predicate": normalized_pred,
                                    "new_object": winner_obj,
                                    "session": session_id,
                                }
                            )
                            logger.info(
                                "Contradiction resolved (recency): %s | %s | %s"
                                " → %s wins (session %s)",
                                subject,
                                normalized_pred,
                                rival_obj,
                                winner_obj,
                                session_id,
                            )
                        if incoming_ls < max_ls:
                            # Incoming loses to a rival at max_ls; skip Case-3 insertion.
                            if relation.indexed_key:
                                self.removal_ledger[relation.indexed_key] = {
                                    "reason": "contradiction_same_pred",
                                    "old_object": obj,
                                    "new_object": winner_obj,
                                }
                            self.contradictions_resolved.append(
                                {
                                    "method": "model_cardinality",
                                    "subject": subject,
                                    "old_predicate": normalized_pred,
                                    "old_object": obj,
                                    "new_predicate": normalized_pred,
                                    "new_object": winner_obj,
                                    "session": session_id,
                                }
                            )
                            logger.info(
                                "Contradiction resolved (recency): rival %s | %s | %s"
                                " wins over incoming %s (session %s)",
                                subject,
                                normalized_pred,
                                winner_obj,
                                obj,
                                session_id,
                            )
                            return None  # Skip Case-3: incoming is not inserted.
                        # incoming_ls == max_ls: incoming ties or is the unique freshest.
                        # Fall through to Case-3: incoming is inserted.
                # COEXIST: fall through to Case-3 insertion.

        # --- Case 3: New-edge insertion ---
        # After contradiction cleanup, alignment check, or when no same-pred edge exists.
        # Stamp ik_key from relation.indexed_key when set (fold-only; None = no-op).
        # Union relation.session_ids into the initial sessions list so the real
        # contributing session ids ride the edge from the first insertion.
        # The scalar session_id may be a synthetic sentinel for fold/re-merge paths.
        _initial_sessions: list[str] = [session_id]
        for _sid in relation.session_ids:
            if _sid not in _initial_sessions:
                _initial_sessions.append(_sid)
        new_eid = self.graph.add_edge(
            subject,
            obj,
            predicate=normalized_pred,
            relation_type=relation.relation_type,
            confidence=relation.confidence,
            reinforcement_count=1,
            sessions=_initial_sessions,
        )
        self.graph[subject][obj][new_eid]["last_seen"] = relation.last_seen or timestamp
        if relation.indexed_key:
            self.graph[subject][obj][new_eid][_IK_KEY_ATTR] = relation.indexed_key
        # A-1 + B-4: stamp speaker_id unconditionally (net-new edge has no prior value)
        # and stamp edge_source conditionally (only when the carry-slot is non-empty).
        self.graph[subject][obj][new_eid]["speaker_id"] = relation.speaker_id
        if relation.edge_source:
            self.graph[subject][obj][new_eid][_EDGE_SOURCE_ATTR] = relation.edge_source
        return None

    def reset_graph(self) -> None:
        """Reset the keying surface to an empty graph, clearing per-fold caches.

        Called by ``consolidate_interim_adapters`` BEFORE the reconstruction-
        and-re-merge pass so the keying surface is empty and provenance keying
        is unconditional: reconstructed-key edges are always net-new (Case 3)
        or intra-fold-collapsed (Case 1 among recon edges), with no dependence
        on any pre-existing edge state.

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
        self.reinforcements = {}
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
