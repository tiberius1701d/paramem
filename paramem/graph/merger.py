"""Knowledge graph merging with entity resolution and cardinality resolution.

Contradiction resolution:
- Same-predicate, different-object cardinality resolution: the model returns one of two
  verdicts (COEXIST / REPLACE) for each same-(subject, predicate)/different-object group.
  Cardinality judgment is cached per predicate (one model call per unique predicate).
  Active whenever a model is present and ``resolve_contradictions=True``.
- COEXIST: both values are independent and multi-valued; keep both edges.
- REPLACE (single-valued): recency selection over ``{incoming} ∪ rivals``, treating an
  empty ``last_seen`` ("") as the oldest possible timestamp (dated always beats undated).
  Rule (applied uniformly at ingest, interim, and fold; no positional fork):
  1. ALL candidates' ``last_seen`` are empty (``""``) → COEXIST: insert incoming, remove
     nothing.  Covers legacy timestamp-less keys at fold (all "") where no candidate
     carries a recency signal.
  2. At least one candidate is dated: ``max_ls = max(candidates)`` (empty "" sorts below
     every ISO-8601 string, so it can never win unless every candidate is empty).
     Strictly-older rivals (``last_seen < max_ls``, including undated ones) are retired
     and ledgered.  Ties at ``max_ls`` coexist.  Incoming: if ``incoming_ls == max_ls`` →
     insert (fall through to Case-3); if ``incoming_ls < max_ls`` → incoming loses → NOT
     inserted, ledgered.  An undated incoming fact never wins against a dated rival; a
     dated incoming fact always outranks an undated rival.
  At fold, the session ``timestamp`` passed to the merger is ``""`` so the fallback
  ``relation.last_seen or timestamp`` yields ``""`` for legacy relations — they only
  coexist when every rival is likewise undated; a dated rival supersedes them.
- When no model is present (experiments, after release): all triples coexist (no removal).
"""

import json
import logging
from pathlib import Path

import networkx as nx
from rapidfuzz import fuzz

from paramem.graph.prompts import _load_prompt
from paramem.graph.relation_prep import attr_predicate, attribute_value_is_empty, strip_has_prefix
from paramem.graph.schema import Entity, Relation, SessionGraph
from paramem.utils.identity import canonical as canonical_id
from paramem.utils.identity import is_speaker_id

logger = logging.getLogger(__name__)

#: The complete, executable vocabulary of ``GraphMerger.removal_ledger``
#: reason codes.  :meth:`GraphMerger.record_removal` — the ledger's ONE
#: writer — rejects any other reason.  Consumers that enumerate reason codes
#: (e.g. :func:`paramem.utils.artifacts.on_removal_ledger`) reference this
#: constant rather than restating the list, so the two can never drift apart.
REMOVAL_REASONS: frozenset[str] = frozenset(
    {
        "dedup",
        "contradiction_same_pred",
        "enrichment_same_as",
        "predicate_synonym_collapse",
        "attribute_key_superseded",
        "unkeyable_no_predicate",
        "duplicate_projection",
    }
)


def min_nonempty(a: str, b: str) -> str:
    """Return the lexicographically smallest of *a* and *b*, treating "" as absent.

    ISO 8601 timestamp strings sort lexicographically, so ``min`` is
    chronological. Plain ``min("", x)`` would wrongly pick ``""`` (it sorts
    lowest), which is correct for ``max``-based last_seen merging but wrong
    for first_seen: an empty string means "unknown", not "earliest possible
    time", so it must never win a first_seen ``min``. When exactly one side
    is empty, the non-empty side wins; when both are empty, the result is
    ``""``; when both are non-empty, the true chronological minimum wins.
    """
    if not a:
        return b
    if not b:
        return a
    return min(a, b)


def reconcile_provenance(target: dict, relation: "Relation", timestamp: str) -> None:
    """Apply the one relation -> target provenance rule to *target* in place.

    ``speaker_id``   ``target["speaker_id"] = target.get("speaker_id") or
                     relation.speaker_id`` — always written, so the key is
                     present after every call (first-non-empty-wins; an
                     absent key and an empty value are the same case).
    ``last_seen``    ``max(target.get("last_seen", ""), relation.last_seen or
                     timestamp)``.
    ``first_seen``   ``min_nonempty(target.get("first_seen", ""),
                     relation.first_seen or timestamp)``.
    ``edge_source``  written only when ``relation.edge_source`` is non-empty
                     AND *target* carries none yet (same first-non-empty-wins
                     rule).

    No ``existing``/net-new flag: on an empty *target* every rule degenerates
    to the net-new form (``max("", x) == x``, ``min_nonempty("", x) == x``,
    and first-non-empty-wins on a key that is not yet present is an
    unconditional stamp) — one rule reproduces both today's re-observation
    branch and today's net-new branch.

    Three callers, all reconciling a relation onto ONE target: the edge
    Case-1 update in :meth:`GraphMerger._upsert_relation` (called BEFORE the
    ``ik_key`` if/elif chain — the keyless-onto-keyed arm copies the
    already-merged ``target["last_seen"]``/``target["first_seen"]`` into
    ``adopt_reinforcements`` and needs the merged values, not the pre-merge
    ones), the edge Case-3 insert (called after ``add_edge``), and the node
    attribute record built by :meth:`GraphMerger.merge`'s
    ``relation_type == "attribute"`` gate — called BEFORE that gate's own
    ``ik_key`` if/elif chain, for the identical reason: its keyless-onto-keyed
    arm also copies the already-merged ``record["last_seen"]``/
    ``record["first_seen"]`` into ``adopt_reinforcements``.  ``confidence``
    and the ``sessions`` union stay inline in ``_upsert_relation`` —
    edge-only, no analog on the attribute record.

    Args:
        target: The edge or attribute-record dict, mutated in place.
        relation: The incoming :class:`Relation` supplying the provenance.
        timestamp: Fallback wall-clock stamp used when *relation* carries
            no ``last_seen``/``first_seen`` of its own.
    """
    from paramem.memory.persistence import _EDGE_SOURCE_ATTR

    target["last_seen"] = max(target.get("last_seen", ""), relation.last_seen or timestamp)
    target["first_seen"] = min_nonempty(
        target.get("first_seen", ""), relation.first_seen or timestamp
    )
    target["speaker_id"] = target.get("speaker_id") or relation.speaker_id
    if relation.edge_source and not target.get(_EDGE_SOURCE_ATTR):
        target[_EDGE_SOURCE_ATTR] = relation.edge_source


def check_predicate_coexistence(
    subject: str,
    predicate: str,
    model,
    tokenizer,
    prompt: str,
    system_prompt: str,
) -> str:
    """Ask the model whether the predicate is single-valued (REPLACE) or multi-valued (COEXIST).

    Classifies the *predicate* alone — no object values are sent to the model.
    The verdict is cached per predicate by the caller so the model is called at
    most once per unique predicate per merger instance.

    Returns the verdict string:

    - ``"COEXIST"`` — predicate is multi-valued; multiple objects can be held
      simultaneously (e.g. ``speaks``, ``developed``, ``has_pet``).
    - ``"REPLACE"`` — predicate is single-valued; a new value supersedes the old
      (e.g. ``lives_in``, ``current_employer``, ``date_of_birth``).

    The default on an ambiguous response is ``"COEXIST"`` (safer —
    do not lose data).

    Args:
        subject: Entity whose predicate cardinality is being judged (used for
            logging only; not injected into the prompt).
        predicate: The predicate to classify.
        model: LLM to use for the cardinality judgment.
        tokenizer: Tokenizer paired with *model*.
        prompt: Prompt template with a ``{predicate}`` slot.  Load from
            ``merger_coexistence.txt`` via ``_load_prompt(...)``
            and pass here — no inline fallback constant exists.
        system_prompt: System-role content for the classification call.  Load
            from ``merger_coexistence_system.txt`` via
            ``_load_prompt(...)`` and pass here.
    """
    from paramem.evaluation.recall import generate_answer
    from paramem.models.loader import render_chat_prompt

    prompt = prompt.format(predicate=predicate)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    formatted = render_chat_prompt(messages, tokenizer, add_generation_prompt=True)

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


def node_display(node_data: dict, node_key: str) -> str:
    """The display surface for a merged-graph node.

    Returns the node's first-seen display surface when it carries one, else
    *node_key* (already the canonical identity form). The ONE place display
    is RESOLVED for a node (the write-side first-seen guards read
    ``display_name`` directly, since they exist to decide whether to write
    it): the fold's edge walk and node-attribute walk
    (:mod:`paramem.training.consolidation`), the cloud-enrichment endpoint
    surface (:mod:`paramem.training.graph_enrich`) and this module's dedup
    ``pre_surfaces`` record all resolve display through here, so the
    "surface, else key" rule has one implementation.

    Args:
        node_data: A node's data dict from ``GraphMerger.graph.nodes``.
        node_key: The node's canonical key, used as the fallback.
    """
    return node_data.get("display_name") or node_key


def attribute_fact(node_data: dict, node_key: str, attr_key: str, record: dict) -> dict:
    """Project one node attribute record into a fact dict.

    Returns ``{subject, predicate, object, speaker_id, first_seen,
    last_seen, ik_key}`` — ``subject`` is the node's DISPLAY surface via
    :func:`node_display`, ``predicate`` is :func:`~paramem.graph.relation_prep.attr_predicate`
    of *attr_key*, ``object`` is ``record["value"]``.  ``ik_key`` is ``""``
    when *record* carries none (a keyless attribute).

    Placed beside :func:`node_display` — the module that owns node-data
    reads.  The ONE node-record -> fact projection, shared by the fold's
    keyed walk and its pending-relation capture
    (:mod:`paramem.training.consolidation`), so the two callers cannot
    disagree on the subject surface: the display surface is correct for
    both — trained content should read the human-readable surface, and a
    pending-relation ``Relation`` re-merges through :meth:`GraphMerger.merge`,
    which folds the subject through :func:`~paramem.utils.identity.canonical`
    for node identity, so display and node-key surfaces resolve to the same
    node either way.

    Args:
        node_data: The node's data dict from ``GraphMerger.graph.nodes``
            (read only to resolve the display surface via
            :func:`node_display`).
        node_key: The node's canonical key — :func:`node_display`'s
            fallback when no ``display_name`` is stored.
        attr_key: The attribute's canonical key on
            ``node_data["attributes"]``.
        record: The attribute record — ``{value, speaker_id, first_seen,
            last_seen, edge_source?, ik_key?}`` (see the record-shape note
            on :meth:`GraphMerger.merge`).

    Returns:
        A fresh dict; *node_data* and *record* are not mutated.
    """
    return {
        "subject": node_display(node_data, node_key),
        "predicate": attr_predicate(attr_key),
        "object": record.get("value", ""),
        "speaker_id": record.get("speaker_id", ""),
        "first_seen": record.get("first_seen", ""),
        "last_seen": record.get("last_seen", ""),
        "ik_key": record.get("ik_key", ""),
    }


def _set_display_name(node: dict, value: str, *, refresh: bool = False) -> None:
    """Write *value* onto *node*'s ``display_name`` field under the
    project-wide display-write rule: first-seen wins by default.

    Only writes when the node carries no ``display_name`` yet, UNLESS
    *refresh* is ``True`` — the speaker-refresh case, where the caller's
    value is always the current canonical ``speaker{N}`` token and must
    overwrite whatever is stored.  The single implementation of the
    "first-seen wins, speaker refreshes" rule spelled at every node-update
    call site that writes ``display_name``.

    Args:
        node: A node's data dict from ``GraphMerger.graph.nodes``, mutated
            in place.
        value: The display surface to write.
        refresh: When ``True``, overwrite unconditionally.  Default
            ``False`` (first-seen-wins).
    """
    if refresh or not node.get("display_name"):
        node["display_name"] = value


def _synth_speaker_entities(relations: "list[Relation]") -> "list[Entity]":
    """Synthesise :class:`Entity` objects for speaker-attributed subjects.

    For each :class:`Relation` in *relations* whose ``speaker_id`` is non-empty
    and whose ``subject == speaker_id`` (plain equality — both are lowercase
    ``speaker{N}`` under the lowercase-uniform design), emit one :class:`Entity`
    with ``entity_type="person"`` and the matching ``speaker_id``.  Deduplicates
    by ``speaker_id`` so exactly one entity is produced per unique speaker.

    Non-speaker subjects (``speaker_id == ""`` OR ``subject != speaker_id``)
    are skipped; their nodes retain no ``speaker_id`` attribute.

    Used by :meth:`GraphMerger.merge_relations` so that
    :meth:`GraphMerger._upsert_entity` stamps the node's own ``speaker_id``
    attribute onto a speaker subject's node — the identity marker
    :func:`~paramem.training.graph_enrich.enrich_graph`'s endpoint-surface
    rule reads to decide whether an enrichment relation's endpoint passes its
    canonical key (a speaker node) or its display surface (any other node).
    This node attribute is a separate concern from a relation's or a node
    attribute record's own ``speaker_id`` provenance field: the fold's keyed
    walk
    (:meth:`~paramem.training.consolidation.ConsolidationLoop._build_working_keyed_walk`)
    reads an edge's or an attribute record's own ``speaker_id`` directly and
    never falls back to this node attribute.

    Args:
        relations: The list of :class:`Relation` objects from which speaker
            entities are derived.

    Returns:
        A :class:`list` of :class:`Entity` objects, one per unique speaker
        subject found in *relations*.  May be empty when no relation carries a
        non-empty ``speaker_id`` whose subject equals the speaker ID.
    """
    _seen_speaker_ids: set[str] = set()
    entities: list[Entity] = []
    for _r in relations:
        if _r.speaker_id and _r.speaker_id not in _seen_speaker_ids:
            # Both subject and speaker_id are lowercase speaker{N} — plain ==
            # is sufficient.  Non-speaker subjects are skipped.
            if _r.subject == _r.speaker_id:
                entities.append(
                    Entity(
                        # Use _r.subject (== _r.speaker_id) as the entity name.
                        # This refreshes the node's display_name field to the
                        # lowercase speaker_id on the existing speaker node.
                        name=_r.subject,
                        entity_type="person",
                        speaker_id=_r.speaker_id,
                    )
                )
                _seen_speaker_ids.add(_r.speaker_id)
    return entities


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
                and ``merger_coexistence_system.txt`` from.  Falls back to
                ``configs/prompts/`` in the project root.  Both files are
                required; absence raises :exc:`FileNotFoundError`.  Resolved
                once at construction so a config edit takes effect at the
                next consolidation cycle (when a new merger instance is created).
        """
        self.similarity_threshold = similarity_threshold
        self.graph = nx.MultiDiGraph()
        self.model = model
        self.tokenizer = tokenizer
        self.contradictions_resolved = []  # log of resolved contradictions
        # Per-fold output lists — also initialised here so _upsert_relation is
        # safe to call without a preceding merge() call (e.g. in unit tests).
        # Reset ONLY in reset_graph() (NOT in merge()) — every accumulator
        # below accumulates across every merge()/merge_relations() call within
        # one fold (recon re-merge, extra-relations re-merge, interim
        # recital-dedup re-merge, any intervening enrichment merge) and must
        # survive intact until the fold's reinforcement-credit pass reads
        # removal_ledger/adopt_reinforcements.
        # removal_ledger: records, keyed by ik_key with a stable reason code,
        # every reason a previously-registered key is absent from the merged
        # graph this fold — an edge/attribute removal (dedup, contradiction,
        # synonym collapse, enrichment contraction, attribute-key
        # supersession, duplicate-projection collision) OR a key that was
        # never merged at all (unkeyable_no_predicate — its store entry has
        # no predicate, so it never reaches the merge surface under any
        # key).  Reset in reset_graph(), NOT in merge() — must survive the
        # fold's reset_graph→re-merge→enrich→classify span.  The ONLY writer
        # is :meth:`record_removal`; every in-module removal site calls it,
        # and the two out-of-module writers
        # (:meth:`~paramem.training.graph_tier.GraphTierRefiner.run_normalization`
        # and :func:`~paramem.training.graph_enrich.enrich_graph`, both of
        # which already hold the merger they mutate) call it through the
        # merger they were constructed/passed with.
        # reason ∈ :data:`REMOVAL_REASONS` (``record_removal`` rejects any
        # other value).
        # ``survivor_key`` is present exactly when the removed fact carries
        # forward under another indexed key — always true for "dedup",
        # "predicate_synonym_collapse", and "duplicate_projection" (the
        # fact carries forward under the already-emitted key), and true for
        # "attribute_key_superseded" ONLY when the same value carries forward
        # under the new key (a different value winning is the contradiction
        # shape — see old_object/new_object — and omits it).  It is the
        # fold's reinforcement-credit input: the survivor inherits the
        # removed keys' maturity, which would otherwise be discarded when
        # they are staled.  A contradiction (either the edge kind or the
        # different-value attribute-key kind) is a supersession (a DIFFERENT
        # fact won, see old_object/new_object), an enrichment same_as is a
        # node contraction (see keep_node), and an unkeyable-no-predicate
        # removal has no surviving fact at all — none of those three carries
        # a ``survivor_key``.
        self.removal_ledger: dict[str, dict] = {}
        # adopt_reinforcements: main-tier ik_key -> (last_seen, first_seen) recorded
        # by any merge called with credit_adopt_reinforcement=True, from either
        # Case-1 arm: the adopt branch (relation carries an indexed_key, existing
        # edge is keyless — the interim recital-dedup merge) or the
        # keyless-onto-keyed arm (relation carries no indexed_key, existing edge
        # already does — the extra_relations/pending-session merge, both scopes).
        # No edge is removed by either arm, so there is no ledger entry to carry
        # it: this is the credit pass's second input, and its own accumulator for
        # that reason.  Consumed EXACTLY ONCE per fold at the reinforcement-credit
        # site in stage_event (_apply_working_reinforcement_credit) — a second
        # refine invocation would double-credit (do not add one).
        self.adopt_reinforcements: dict[str, tuple[str, str]] = {}
        # Cache: predicate → True (multi-valued/coexist) or False (single-valued/replace)
        self._predicate_cardinality: dict[str, bool] = {}
        # Resolve prompts once at construction so a file edit takes effect next cycle.
        _pd = Path(prompts_dir) if prompts_dir else None
        self._coexistence_prompt = _load_prompt(
            "merger_coexistence.txt",
            prompts_dir=_pd,
        )
        self._coexistence_system = _load_prompt(
            "merger_coexistence_system.txt",
            prompts_dir=_pd,
        )

    def merge(
        self,
        session_graph: SessionGraph,
        *,
        resolve_contradictions: bool = True,
        credit_adopt_reinforcement: bool = False,
    ) -> nx.MultiDiGraph:
        """Merge a session graph into the cumulative graph.

        A relation whose ``relation_type == "attribute"`` never becomes an
        edge: it is diverted onto the SUBJECT node's ``attributes`` dict as
        a provenance-bearing record — ``{value, speaker_id, first_seen,
        last_seen, edge_source?, ik_key?}``, reconciled by the same
        :func:`reconcile_provenance` rule the edge path uses (see the
        ``relation.relation_type == "attribute"`` branch below).  This is
        the merger-gate authority for literal-value facts (phone, email,
        certification, job title, exact date, ...) that would otherwise mint
        a concept node colliding across subjects sharing the same value. A
        value carrying no information
        (:func:`~paramem.graph.relation_prep.attribute_value_is_empty`) is
        skipped before any record is written. A re-observation carrying the
        SAME value reconciles onto the existing record (the window widens,
        provenance is first-non-empty-wins); a DIFFERENT value starts a NEW
        record lifetime instead of reconciling — the new value's own
        speaker and window, never blended with the superseded assertion —
        and, when a key was bound to the superseded value, ledgers the
        displacement (``attribute_key_superseded``). A keyless same-value
        re-observation landing on an already-keyed record earns
        reinforcement credit under ``credit_adopt_reinforcement``, the
        attribute mirror of the edge path's keyless-onto-keyed arm — an
        attribute-typed key matures and promotes through the same credit
        surface as any other key.

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
            credit_adopt_reinforcement: When ``True``, three arms record the
                adopted key in ``self.adopt_reinforcements`` for the fold's
                reinforcement-credit pass: two Case-1 edge arms — the adopt
                branch (an incoming keyed relation adopts its
                ``indexed_key`` onto a pre-existing keyless edge) and the
                keyless-onto-keyed arm (an incoming keyless relation
                re-observes an already-keyed edge) — and the attribute
                gate's same-value keyless-onto-keyed arm (an incoming
                keyless attribute relation re-observes an already-keyed
                attribute record). Default ``False``.

        Returns the updated cumulative graph.

        A Case-1 exact-duplicate collapse (an incoming ``Relation.indexed_key``
        matching an existing edge with an ``ik_key`` already stamped — the
        fold-only path; never happens during normal live ingest where
        ``Relation.indexed_key is None``) supersedes the incoming key: the
        surviving key is the existing edge's key, and the collapse is named
        by the ``removal_ledger`` entry written in the same branch, under
        ``survivor_key`` — that ledger entry is what the fold's
        reinforcement-credit pass reads to distinguish intended dedup (fact
        preserved under the surviving key) from genuine reconstruction loss.

        No accumulator is reset at the top of this method — ``removal_ledger``
        and ``adopt_reinforcements`` are both reset ONLY in
        :meth:`reset_graph`, so they accumulate across every ``merge()`` call
        within one fold (recon re-merge, extra-relations re-merge, interim
        recital-dedup re-merge, and any intervening enrichment merge) and
        survive intact until the fold's consumers read them.  A per-call reset
        here would silently wipe a genuine Case-1 collapse recorded by an
        earlier merge in the same fold the moment any later merge in that fold
        ran — even one producing zero collapses of its own — before the
        reinforcement-credit pass in ``stage_event``
        (:meth:`~paramem.training.consolidation.ConsolidationLoop._apply_working_reinforcement_credit`)
        ever gets to read ``removal_ledger``/``adopt_reinforcements``.
        """
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
            # canonical_id is the node-key fold for EVERY endpoint, speaker or
            # not.  A speaker id is already its own canonical form, so the fold
            # is a no-op on it and there is no separate speaker branch to keep.
            if subj_surface in entity_name_map:
                subject = entity_name_map[subj_surface]
            else:
                subject = canonical_id(subj_surface)

            if obj_surface in entity_name_map:
                obj = entity_name_map[obj_surface]
            else:
                obj = canonical_id(obj_surface)

            if relation.relation_type == "attribute":
                # Literal-value relation (phone/email/date/certification/job
                # title, ...): the model has marked this as a scalar datum
                # ABOUT the subject rather than a claim relating the subject
                # to a distinct concept. Fold it onto the SUBJECT node's
                # ``attributes`` dict as a provenance-bearing record instead
                # of minting a colliding concept node keyed on the value
                # (two subjects with the same certification would otherwise
                # collapse onto one node). No object node is created, no
                # edge is inserted — this diversion runs BEFORE the
                # object-node-ensure below so an attribute relation never
                # reaches ``_upsert_relation``.
                #
                # A value carrying no information is skipped before any node
                # or record is touched — this is in ADDITION to
                # ``attribute_relations``'s own skip (relation_prep.py): a
                # model-emitted attribute relation reaches this gate
                # directly, never through that projection, and a projected
                # relation reaches ``_entries_from_graph`` WITHOUT crossing
                # this gate, so a placeholder-valued projection would
                # otherwise become a keyed entry the gate would have
                # dropped. Two doors, two guards, one rule.
                if attribute_value_is_empty(relation.object):
                    continue

                # Subject-node-ensure mirrors the endpoint-ensure loop below,
                # restricted to the subject alone.
                if subject not in self.graph:
                    self.graph.add_node(
                        subject,
                        entity_type="concept",
                        attributes={},
                        display_name=subj_surface,
                        reinforcement_count=1,
                        sessions=[session_id],
                    )
                elif subj_surface not in entity_name_map:
                    # Node already exists but display name not yet set
                    # (first-seen wins) — same rule the endpoint loop below
                    # applies.
                    _set_display_name(self.graph.nodes[subject], subj_surface)

                node = self.graph.nodes[subject]
                node_attrs = node.get("attributes", {})
                attr_key = canonical_id(strip_has_prefix(relation.predicate), mode="full")
                new_value = canonical_id(relation.object, mode="spaces")
                record = node_attrs.get(attr_key)

                if record is None:
                    # Net-new attribute: one record, provenance reconciled
                    # onto an empty target degenerates to an unconditional
                    # stamp (see reconcile_provenance's docstring).
                    record = {"value": new_value}
                    reconcile_provenance(record, relation, timestamp)
                    if relation.indexed_key:
                        record["ik_key"] = relation.indexed_key
                    node_attrs[attr_key] = record
                elif record["value"] == new_value:
                    # Same value re-observed: this is a genuine
                    # reconciliation, not a new lifetime — widen the window
                    # and adopt provenance, and only touch the key binding
                    # when the incoming relation itself carries a key.
                    reconcile_provenance(record, relation, timestamp)
                    if relation.indexed_key:
                        incumbent_key = record.get("ik_key")
                        if incumbent_key is not None and incumbent_key != relation.indexed_key:
                            # Same value carried forward under a NEW key —
                            # the documented condition for survivor_key: the
                            # displaced key's reinforcement maturity flows to
                            # the survivor via _apply_working_reinforcement_credit
                            # instead of being silently discarded.
                            self.record_removal(
                                incumbent_key,
                                reason="attribute_key_superseded",
                                survivor_key=relation.indexed_key,
                            )
                        record["ik_key"] = relation.indexed_key
                    elif record.get("ik_key") and credit_adopt_reinforcement:
                        # Keyless-onto-keyed re-sighting, attribute mirror of
                        # the edge arm below (_upsert_relation): the incoming
                        # relation carries no ik_key but the record was
                        # already keyed by an earlier merge in this fold —
                        # credit the reinforcement so an attribute-typed key
                        # matures through the same surface an edge-typed key
                        # does.  Gated on credit_adopt_reinforcement so an
                        # enrichment-time merge (which restates a fact, not
                        # re-observes it) never earns credit here.  Reads the
                        # window reconcile_provenance just merged above, not
                        # the pre-merge relation values.
                        self.adopt_reinforcements[record["ik_key"]] = (
                            record["last_seen"],
                            record["first_seen"],
                        )
                else:
                    # Different value: a NEW record lifetime, never
                    # reconciled onto the incumbent — reconciling would
                    # carry the superseded value's first_seen and its
                    # original asserter forward onto a fact neither
                    # describes (a window claiming the new value was
                    # asserted when the old one was, and, where two
                    # speakers assert different values, attribution of the
                    # new value to the speaker who asserted the old one).
                    incumbent_key = record.get("ik_key")
                    if relation.indexed_key:
                        # Keyed case: unchanged two-arm ledger shape — a
                        # different value winning is the contradiction
                        # shape (old_object/new_object, no survivor_key,
                        # unlike the same-value carry-forward arm above).
                        if incumbent_key is not None and incumbent_key != relation.indexed_key:
                            self.record_removal(
                                incumbent_key,
                                reason="attribute_key_superseded",
                                old_object=record["value"],
                                new_object=new_value,
                            )
                        node_attrs[attr_key] = {
                            "value": new_value,
                            "speaker_id": relation.speaker_id,
                            "first_seen": relation.first_seen or timestamp,
                            "last_seen": relation.last_seen or timestamp,
                            "ik_key": relation.indexed_key,
                        }
                    else:
                        # Keyless case: a re-observation with a different
                        # value strands the old key's binding — ledger it
                        # (no survivor_key: the new value has no key of its
                        # own yet) and replace the record wholesale.  A
                        # keyless incumbent (never bound to a key) has
                        # nothing to ledger.
                        if incumbent_key is not None:
                            self.record_removal(
                                incumbent_key,
                                reason="attribute_key_superseded",
                                old_object=record["value"],
                                new_object=new_value,
                            )
                        node_attrs[attr_key] = {
                            "value": new_value,
                            "speaker_id": relation.speaker_id,
                            "first_seen": relation.first_seen or timestamp,
                            "last_seen": relation.last_seen or timestamp,
                        }  # no ik_key — the keyed walk's keyless branch mints a fresh one
                node["attributes"] = node_attrs
                continue

            # Build a display-name map for endpoints not resolved through entities.
            # Keys in entity_name_map already have _upsert_entity called for them
            # (which writes display_name).  Remaining endpoints are raw relation
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
                        attributes={},
                        display_name=_endpoint_display[name],
                        reinforcement_count=1,
                        sessions=[session_id],
                    )
                elif name in _endpoint_display:
                    # Node already exists but display name not yet set (first-seen wins).
                    _set_display_name(self.graph.nodes[name], _endpoint_display[name])

            self._upsert_relation(
                subject,
                obj,
                relation,
                session_id,
                timestamp,
                resolve_contradictions=resolve_contradictions,
                credit_adopt_reinforcement=credit_adopt_reinforcement,
            )

        logger.info(
            "Merged session %s: graph now has %d nodes, %d edges",
            session_id,
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )
        return self.graph

    def record_removal(
        self,
        indexed_key: str,
        *,
        reason: str,
        survivor_key: "str | None" = None,
        **detail,
    ) -> None:
        """Record one reason *indexed_key* is absent from the merged graph.

        The ONE writer of ``removal_ledger`` across the three modules that
        record removals (this module, :mod:`paramem.training.graph_tier`, and
        :mod:`paramem.training.graph_enrich`) — no other line anywhere writes
        ``removal_ledger`` directly.  Covers both an edge/attribute removal
        (dedup, contradiction, synonym collapse, enrichment contraction,
        attribute-key supersession) and a key that was never merged at all
        (``unkeyable_no_predicate``).

        Args:
            indexed_key: The removed key's ``ik_key`` string — the ledger's
                dict key.
            reason: A stable reason code from the documented vocabulary (see
                the ``removal_ledger`` field comment in ``__init__``), e.g.
                ``"dedup"``, ``"contradiction_same_pred"``,
                ``"predicate_synonym_collapse"``, ``"enrichment_same_as"``,
                ``"attribute_key_superseded"``, ``"unkeyable_no_predicate"``,
                or ``"duplicate_projection"``.
            survivor_key: Set exactly when the removed fact carries forward
                under another indexed key — that is what the fold's
                reinforcement-credit pass
                (:meth:`~paramem.training.consolidation.ConsolidationLoop._apply_working_reinforcement_credit`)
                consumes.  Omitted from the stored entry when ``None`` (the
                default), so removal shapes that carry no survivor (a
                contradiction, an enrichment same_as contraction, an
                unkeyable-no-predicate skip) are unchanged on disk.
            **detail: Reason-specific fields stored verbatim on the entry,
                e.g. ``pre_surfaces`` for a dedup, ``old_object``/
                ``new_object`` for a contradiction, ``survivor_predicate`` for
                a synonym collapse, or ``keep_node`` for an enrichment
                contraction.

        Raises:
            ValueError: when *reason* is not one of :data:`REMOVAL_REASONS`.
        """
        if reason not in REMOVAL_REASONS:
            raise ValueError(
                f"record_removal: reason={reason!r} is not in REMOVAL_REASONS "
                f"({sorted(REMOVAL_REASONS)})"
            )
        entry: dict = {"reason": reason}
        if survivor_key is not None:
            entry["survivor_key"] = survivor_key
        entry.update(detail)
        self.removal_ledger[indexed_key] = entry

    def merge_relations(
        self,
        relations: "list[Relation]",
        *,
        session_id: str,
        log_label: str,
        timestamp: str = "",
        resolve_contradictions: bool = False,
        credit_adopt_reinforcement: bool = False,
    ) -> None:
        """Build a synthetic :class:`SessionGraph` from *relations* and merge it.

        Shared builder for turning a ``list[Relation]`` into an entitied,
        merged :class:`SessionGraph`.  Callers today: ``stage_event``'s own
        three merges (``session_id="__stage_recon__"`` for this event's
        recalled primary-tier content, ``session_id="__stage_pending__"``
        for its newly extracted material, and
        ``session_id="__stage_dedup_targets__"`` for its dedup-only
        candidate-tier content), and the graph-enrichment path
        (``session_id="__graph_enrichment__"``).  Graph enrichment separately
        constructs its own ``SessionGraph`` for a different purpose earlier
        in its pipeline; this is not the only place a ``SessionGraph`` is
        built.

        The entity list is synthesised from *relations* via
        :func:`_synth_speaker_entities`, which stamps ``speaker_id`` onto
        speaker subject nodes.  This ensures the edge walk in
        :meth:`~paramem.training.consolidation.ConsolidationLoop._build_working_keyed_walk`
        reads the correct ``speaker_id`` from each node.
        Without this, reconstructed/synthesised person nodes would be stored
        as ``entity_type="concept"`` with no ``speaker_id``, and
        graph-enrichment would root facts at unattributed nodes.

        This method does not manage the gradient-checkpointing guard around
        ``model.generate()`` calls made by cardinality resolution when
        ``resolve_contradictions=True`` — that is the caller's responsibility,
        since only the caller knows whether a model is present.

        Returns early without side effects when *relations* is empty.

        Args:
            relations: The :class:`Relation` objects to merge.  May be the
                registry-true recon set, the pending extra-relations set,
                the interim main-tier dedup set, the simulate fold's
                collected interim-slot relations, or accumulated
                graph-enrichment relations.
            session_id: Synthetic session identifier passed to
                :class:`~paramem.graph.schema.SessionGraph`.  Used only for
                logging/debugging.
            log_label: Human-readable label for the count log line, e.g.
                ``"reconstructed triples"`` or ``"extra (pending-session) relations"``.
            timestamp: The session timestamp passed to :class:`SessionGraph`.
                Default ``""`` (empty string) for all HISTORICAL callers
                (recon/registry-true/simulate-disk/enrichment), which ensures
                the merger's ``relation.last_seen or timestamp`` fallback
                yields ``""`` for legacy relations instead of fabricating a
                NOW recency value.  A legacy ``""`` relation coexists with its
                rivals only when every rival is also undated; a genuinely
                dated rival always outranks it (dated wins over undated) and
                the legacy relation is retired.  Pass ``datetime.now()`` only
                for genuinely FRESH sessions where an empty ``last_seen``
                should resolve to the current wall-clock time.  Currently all
                callers are historical and omit this parameter (receive the
                ``""`` default).
            resolve_contradictions: Forwarded to :meth:`merge`.  Default
                ``False`` (no cardinality resolution).  Set ``True`` when the
                config ``refinement_contradiction == "on"``.
            credit_adopt_reinforcement: Forwarded to :meth:`merge`.  Default
                ``False``.  ``True`` for the interim recital-dedup merge
                (credits a recited fact's Case-1-adopt onto a main-tier key)
                and for the extra-relations (pending-session) merge (credits a
                keyless pending re-observation that lands on an
                already-keyed edge).
        """
        if not relations:
            return
        entities = _synth_speaker_entities(relations)
        _session = SessionGraph(
            session_id=session_id,
            timestamp=timestamp,
            entities=entities,
            relations=relations,
        )
        self.merge(
            _session,
            resolve_contradictions=resolve_contradictions,
            credit_adopt_reinforcement=credit_adopt_reinforcement,
        )
        logger.info(
            "merge_relations: merged %d %s into cumulative graph",
            len(relations),
            log_label,
        )

    def _resolve_entity(self, entity: Entity) -> str:
        """Resolve an entity to its canonical NetworkX node key.

        Speaker entities are first-class graph roots: when
        ``entity.speaker_id`` is set, the node key IS ``entity.speaker_id``
        verbatim (already lowercase ``speaker{N}`` — guaranteed by the ingest
        safety-net in :func:`~paramem.graph.extractor._normalize_extraction`
        and by :meth:`~paramem.server.speaker.SpeakerStore._mint_anon_speaker_id`).
        No casing step is needed.  Both the entity path and the relation-endpoint
        fallback path produce the same node key, preventing casing-collision dups.
        The display name lives at ``node_data["display_name"]``; for speakers
        this is the same lowercase ``speaker{N}`` string — resolved to a human
        name only at the reply boundary, by
        :func:`~paramem.server.speaker.resolve_speaker_tokens`.

        Two enrolled speakers who share a display name (e.g. both ``"Alex"``)
        keep separate graph nodes because their ``speaker_id`` values differ by
        construction.  Name changes across sessions for the same speaker collapse
        onto the same node because the key is the immutable ``speaker{N}`` id.
        Third-party mentions (no ``speaker_id``) live in the name namespace which
        is disjoint from speaker ids (speaker ids follow ``speaker{N}``).

        Resolution rules:

        * **Speaker entity** (``entity.speaker_id`` set) — node key is
          ``entity.speaker_id`` verbatim (lowercase).  No name-based matching;
          no fuzzy match.  The display name is stored on the dedicated
          ``display_name`` node field downstream.
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
        display name is stored on the dedicated ``display_name`` node field for
        ALL node types — not just speakers — so downstream consumers never need
        to use the node key for display.  This method never writes ``attributes``:
        ``entity.attributes`` (including a model-emitted ``"name"`` key, if
        present) reaches the node's ``attributes`` dict exclusively through
        the projection into attribute-typed relations
        (:func:`~paramem.graph.relation_prep.attribute_relations`, run at
        extraction time) and :meth:`GraphMerger.merge`'s
        ``relation_type == "attribute"`` gate — never here, and never read
        for display.  First-seen surface wins for non-speakers:
        ``display_name`` is set on insertion and NOT overwritten on
        subsequent updates (idempotent).

        Speaker entities (``entity.speaker_id`` set) are keyed by
        ``entity.speaker_id`` verbatim (lowercase ``speaker{N}``,
        e.g. ``"speaker0"`` — see :meth:`_resolve_entity`).  The node's
        ``speaker_id`` attribute carries the same lowercase id.
        ``display_name`` stores the same lowercase ``speaker{N}`` id and IS
        refreshed on update.  Display-name resolution happens only at the
        reply boundary via
        :func:`~paramem.server.speaker.resolve_speaker_tokens`, not at
        graph-write time.  For non-speaker entities ``display_name`` is
        first-seen-wins only.
        """
        is_speaker = entity.speaker_id is not None

        if node_key in self.graph:
            node = self.graph.nodes[node_key]
            node["reinforcement_count"] = node.get("reinforcement_count", 0) + 1
            sessions = node.get("sessions", [])
            if session_id not in sessions:
                sessions.append(session_id)
            node["sessions"] = sessions
            # Display surface, never read from entity.attributes.  Speaker
            # entities refresh on update (entity.name is always the lowercase
            # speaker{N} id); non-speaker entities are first-seen-wins only.
            if entity.name and not attribute_value_is_empty(entity.name):
                _set_display_name(node, entity.name, refresh=is_speaker)
            # The node key equals entity.speaker_id (lowercase speaker{N}).
            # The ``speaker_id`` node attribute carries the same value.
            # Defensive: populate the attribute when it is missing (e.g. a node
            # inserted before this refactor that lacks the attribute).
            if is_speaker and node.get("speaker_id") is None:
                node["speaker_id"] = entity.speaker_id
        else:
            node_kwargs: dict = dict(
                entity_type=entity.entity_type,
                attributes={},
                reinforcement_count=1,
                sessions=[session_id],
            )
            # Display surface, never read from entity.attributes.  The node
            # key is now the canonical form so the node ID is no longer the
            # display name.
            if entity.name and not attribute_value_is_empty(entity.name):
                node_kwargs["display_name"] = entity.name
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
        credit_adopt_reinforcement: bool = False,
    ) -> None:
        """Insert or update a relation edge.

        Handles three cases:

        1. Identical triple already exists — exact-duplicate reinforcement: bump
           recurrence and reconcile provenance (``speaker_id``, ``edge_source``,
           ``first_seen``/``last_seen``) via :func:`reconcile_provenance`,
           called BEFORE the ``ik_key`` if/elif chain below — the
           keyless-onto-keyed arm reads the already-merged window into
           ``adopt_reinforcements`` and needs the merged values, not the
           pre-merge ones.  Case-1-adopt: if the existing edge has no
           ``ik_key`` and the incoming ``relation.indexed_key`` is set, adopt
           the key onto the existing edge (fold-only provenance
           carry-through).  When the existing edge already carries an
           ``ik_key`` and the incoming relation carries none, this is a
           keyless re-observation of an already-keyed fact rather than an
           adopt.  In both cases, when ``credit_adopt_reinforcement`` is
           ``True``, the surviving key is recorded in
           ``self.adopt_reinforcements`` for the fold's reinforcement-credit
           pass.
        2. Same (subject, predicate) but different object — cardinality resolution
           when a model is present and ``resolve_contradictions=True``:

           - ``COEXIST``: both values are independent; all edges kept.
           - ``REPLACE`` (single-valued): recency selection over
             ``{incoming} ∪ rivals``, treating an empty ``last_seen`` (``""``)
             as the oldest possible timestamp.  If EVERY candidate's
             ``last_seen`` is ``""`` → COEXIST (no removal; no recency signal
             anywhere).  Otherwise at least one candidate is dated:
             strictly-older rivals (including undated ones) are retired +
             ledgered; ties at ``max_ls`` coexist; incoming NOT inserted
             (returns ``None``) and ledgered when strictly older than
             ``max_ls``.  A dated candidate always outranks an undated one.
             ``incoming_ls = relation.last_seen or timestamp``
             (``timestamp=""`` at fold/recon/simulate; ``now()`` at ingest).

           When ``resolve_contradictions=False``, Case-2 is short-circuited:
           no model call, no edge removal.

           Cardinality (COEXIST vs REPLACE axis) is cached per predicate.
        3. New (subject, predicate, object) — net-new edge insertion, followed
           by :func:`reconcile_provenance` (degenerates to an unconditional
           stamp on the fresh edge — see that function's docstring).  The
           ``ik_key`` from ``relation.indexed_key`` is stamped on the edge when
           set (fold provenance; None at normal ingest — no-op).
        """
        from paramem.memory.persistence import _IK_KEY_ATTR

        normalized_pred = canonical_id(relation.predicate)

        # Symmetric-direction canonicalization — collapse (A,P,B) / (B,P,A) into
        # a single direction so Case-1 reinforcement deduplicates them.
        # Guard: when ANY speaker endpoint is involved, keep the recorded direction
        # (recall is always speaker-anchored, so demoting a speaker out of the
        # subject slot is never correct — even for a lone speaker-owned edge like
        # `speaker0 has_sibling nadia`). Uses the structural is_speaker_id token
        # check rather than the node's speaker_id attribute: a speaker endpoint
        # that arrives without a matching Entity is node-created with no
        # speaker_id attribute (the endpoint-ensure loop in merge()), so the
        # attribute lookup has a hole the token check closes.
        any_speaker = is_speaker_id(subject) or is_speaker_id(obj)
        if relation.symmetric and subject > obj and not any_speaker:
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
            # Reconcile provenance BEFORE the ik_key if/elif chain below: the
            # keyless-onto-keyed arm reads edge["last_seen"]/edge["first_seen"]
            # into adopt_reinforcements and needs the already-merged values.
            reconcile_provenance(edge, relation, timestamp)
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
                if credit_adopt_reinforcement:
                    # The edge's last_seen/first_seen are already merged
                    # (max/min_nonempty, above) over the recited pending fact
                    # and the main-tier dedup target — the exact values the
                    # credit pass wants; no now() fabrication.  The merged
                    # last_seen is also its temporal-order evidence: it exceeds
                    # the key's stored last_seen only when the recital came
                    # from a later session.
                    self.adopt_reinforcements[relation.indexed_key] = (
                        edge.get("last_seen", ""),
                        edge.get("first_seen", ""),
                    )
            # Case-1 reinforcement: when BOTH the existing edge AND the incoming
            # relation carry an ik_key, this is a fold-time duplicate-SPO collapse.
            # The SURVIVING key is the existing edge's ik_key (the incoming
            # relation.indexed_key is superseded); the ledger entry below names
            # it, and the fold's reinforcement-credit pass reads it from there.
            # NOTE: existing_key is the NetworkX integer edge id, NOT the key
            # string.  The survivor's key string is read from the edge attribute.
            elif relation.indexed_key and edge.get(_IK_KEY_ATTR):
                surviving_ik = edge.get(_IK_KEY_ATTR)
                # The removal_ledger entry written below (record_removal,
                # survivor_key=...) is what lets a consumer distinguish
                # intended dedup (fact preserved under the surviving twin)
                # from genuine loss — it is the fate authority.
                # Raw-surface evidence for dedup collapses (observability hook).
                # The incoming raw surfaces (relation.*, only .strip()ed at
                # extraction) are recorded alongside the surviving twin's
                # first-seen surfaces (stored in the node's display_name field
                # and edge["predicate"]).  pre_surfaces is the ground-truth
                # record of what was discarded vs. what survived; readers
                # compare incoming vs. surviving directly rather than relying
                # on a derived boolean.
                _surviving_subj_surface = node_display(self.graph.nodes.get(subject, {}), subject)
                _surviving_pred_surface = edge.get("predicate", "")
                _surviving_obj_surface = node_display(self.graph.nodes.get(obj, {}), obj)
                self.record_removal(
                    relation.indexed_key,
                    reason="dedup",
                    survivor_key=surviving_ik,
                    pre_surfaces={
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
                )
            # Keyless-onto-keyed re-sighting: the incoming relation carries no
            # ik_key (a pending re-observation, never stamped at capture) but
            # the existing edge already carries one — the fact was already
            # keyed by an earlier merge in this fold (e.g. the recon merge).
            # Neither adopt (Branch A needs an incoming key) nor dedup
            # (Branch B needs both sides keyed) fires, so without this arm the
            # re-observation is silently dropped and the key's reinforcement
            # count never grows.  Gated on credit_adopt_reinforcement so an
            # enrichment-time merge (which restates a fact, not re-observes
            # it) never earns credit here.  ``GraphMerger.merge``'s attribute
            # gate carries the identical arm for attribute-typed records
            # (its same-value branch) — this is not the only such arm, edges
            # and attributes share the mechanism.
            elif not relation.indexed_key and edge.get(_IK_KEY_ATTR) and credit_adopt_reinforcement:
                self.adopt_reinforcements[edge[_IK_KEY_ATTR]] = (
                    edge.get("last_seen", ""),
                    edge.get("first_seen", ""),
                )
            return None

        # --- Case 2: Same-predicate, different-object cardinality resolution ---
        # When no model is present or resolve_contradictions=False, fall through
        # to Case-3 insertion (coexist-all / short-circuit).
        # When a model is present and resolve_contradictions=True:
        #   1. Gather ALL rival edges (same subject+predicate, different object).
        #   2. Run ONE cardinality verdict (cached per predicate).
        #   3. COEXIST → fall through.
        #   4. REPLACE → recency selection across {incoming} ∪ rivals, "" sorts as the
        #      oldest possible timestamp (dated always beats undated):
        #      - incoming_ls = relation.last_seen or timestamp (may be "" at fold)
        #      - rival_ls  = each rival's stored last_seen (may be "")
        #      - max_ls = lexicographic max across the full set ("" only wins when
        #        every candidate in the set is "")
        #      - every candidate "" → coexist; fall through, no removal
        #      - incoming uniquely freshest (dated, beats any undated rival) → retire
        #        ALL rivals, fall through
        #      - rival uniquely freshest (dated, beats an undated incoming) → that
        #        rival survives; other rivals retired; incoming is NOT inserted
        #        (return None after ledgering its key)
        #      - ties at max_ls (including an all-"" set) coexist
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
                        self.model,
                        self.tokenizer,
                        self._coexistence_prompt,
                        self._coexistence_system,
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

                    if incoming_ls == "" and all(ls == "" for ls in rival_ls_list):
                        # EVERY candidate last_seen is empty (no recency signal anywhere)
                        # → COEXIST.  Covers legacy timestamp-less keys at fold (all "").
                        # Safe no-op: insert incoming, remove nothing.
                        pass
                    else:
                        # At least one candidate is dated; freshest wins.  An empty
                        # last_seen sorts as the oldest possible timestamp (Python
                        # string comparison: "" < any non-empty ISO-8601 string), so a
                        # dated candidate always outranks an undated one — a dated
                        # incoming fact supersedes an undated rival, and an undated
                        # incoming fact never wins against a dated rival.
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
                                self.record_removal(
                                    _removed_ik,
                                    reason="contradiction_same_pred",
                                    old_object=rival_obj,
                                    new_object=winner_obj,
                                )
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
                                self.record_removal(
                                    relation.indexed_key,
                                    reason="contradiction_same_pred",
                                    old_object=obj,
                                    new_object=winner_obj,
                                )
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
        edge = self.graph[subject][obj][new_eid]
        # On a fresh edge every reconcile_provenance rule degenerates to an
        # unconditional stamp from *relation* (see that function's docstring).
        reconcile_provenance(edge, relation, timestamp)
        if relation.indexed_key:
            edge[_IK_KEY_ATTR] = relation.indexed_key
        return None

    def reset_graph(self) -> None:
        """Reset the keying surface to an empty graph, clearing per-fold caches.

        Called by ``stage_event`` BEFORE its own re-merge sequence so the
        keying surface is empty and provenance keying is unconditional:
        reconstructed-key edges are always net-new (Case 3) or
        intra-fold-collapsed (Case 1 among recon edges), with no dependence
        on any pre-existing edge state.

        Cleared caches:
        - ``graph`` — fresh MultiDiGraph (no prior edges/nodes)
        - ``_predicate_cardinality`` — per-predicate COEXIST/REPLACE cache
        - ``contradictions_resolved`` — log of prior resolves
        - ``removal_ledger`` — prior fold's reason-coded key-absence records
        - ``adopt_reinforcements`` — prior fold's dedup-adopt credited main keys

        Does NOT touch ``model``, ``tokenizer``, or the prompt strings — those
        are construction-time state and must survive across folds.
        """
        import networkx as nx

        self.graph = nx.MultiDiGraph()
        self._predicate_cardinality = {}
        self.contradictions_resolved = []
        self.removal_ledger = {}
        self.adopt_reinforcements = {}

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
        """Return the serialized graph as bytes for in-memory consumers.

        Produces the same node-link JSON that
        :func:`~paramem.memory.persistence.save_memory_to_disk` writes, but
        returns the bytes without performing any I/O.  Used by
        /migration/confirm to capture a point-in-time snapshot of the graph for
        the pre-migration backup without requiring a temporary file.

        Returns:
            UTF-8-encoded JSON bytes (node-link format, indent=2).
        """
        data = nx.node_link_data(self.graph)
        return json.dumps(data, indent=2).encode("utf-8")

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
