"""Post-merge, cloud-cloud graph enrichment pass.

Owns the cross-transcript graph enrichment step that runs after a fold's
graph has been materialized: chunking the cumulative graph into N-hop
ego-graphs around high-recurrence entities, anonymizing each chunk
through the same local-model contract session-tier extraction uses,
sending it to a configured cloud provider for second-order relation and
``same_as`` coreference discovery, and merging the result back into the
graph via :meth:`~paramem.graph.merger.GraphMerger.merge_relations`.

Boundary: this module mutates exactly the ``GraphMerger`` it is handed
and reaches nothing else — no consolidation loop, no memory store, no
cycle counter. The caller owns scheduling, gating, and any
post-enrichment bookkeeping (debug snapshots, reinforcement credit).
"""

import logging
import math
from typing import TYPE_CHECKING, Callable

import networkx as nx

from paramem.cloud.admission import evaluate_cloud_egress
from paramem.cloud.anonymize import anonymize
from paramem.config.taxonomy import (
    fallback_relation_type,
    relation_types,
)
from paramem.graph.extractor import request_graph_enrichment
from paramem.graph.merger import GraphMerger, min_nonempty
from paramem.graph.prompts import _load_prompt
from paramem.graph.schema import Relation, SessionGraph
from paramem.memory.persistence import _IK_KEY_ATTR
from paramem.utils.identity import canonical, is_speaker_id
from paramem.utils.vram_guard import VramExhausted, is_fatal_cuda_fault

if TYPE_CHECKING:
    from paramem.graph.extraction_pipeline import ExtractionConfig

logger = logging.getLogger(__name__)

# Frozen set of valid relation types drawn from the single source of truth in
# paramem.config.taxonomy so that the stage-2 clamp stays in sync with the
# Pydantic Relation schema (_RelationType = Literal[relation_types()]).
_VALID_RTYPES: frozenset[str] = frozenset(relation_types())
_FALLBACK_RTYPE: str = fallback_relation_type()


def resolve_to_node_key(
    name: str,
    in_graph: "Callable[[str], bool]",
    coref_map: "dict[str, str] | None" = None,
) -> str:
    """Resolve a surface name to the actual node key used in the graph.

    Collapses the two formerly-duplicated nested resolvers
    (``_resolve_node_key`` / ``_resolve_name``) into one module-level
    function so the resolution logic lives in exactly one place.

    Resolution order:

    1. **Membership shortcut** — if ``in_graph(name)`` is ``True``, the name
       IS already a valid node key; return it unchanged.  This handles ordinary
       non-speaker node keys (node-key model A: the key IS the canonical form)
       without an extra ``canonical()`` call.  Note: with casefolded speaker
       keys, verbatim speaker ids (``"Speaker0"``) are NOT in
       the graph (the key is ``"speaker0"``), so they fall through to step 2.
    2. **Canonical fallback** — ``canonical(name)`` (casefolds, diacritic-folds,
       separator-normalizes).  This resolves speaker ids to their casefolded node
       keys (``"Speaker0"`` → ``"speaker0"``) and ordinary display-surface names
       to their canonical keys.
    3. **Coref-chain follow** (optional) — if ``coref_map`` is provided, follow
       the drop→keep chain on the resolved key.  Cycle-guarded via a ``seen``
       set so a malformed coref loop does not block.

    The stale rationale "verbatim-first because speaker nodes are keyed
    VERBATIM" no longer applies: speaker node keys are now casefolded, so the
    membership shortcut is only useful for ordinary node keys.

    Args:
        name: Surface name or node key to resolve.
        in_graph: Callable that returns ``True`` when its argument is a live
            node key in the graph (typically ``graph.__contains__`` or
            ``lambda n: n in graph``).
        coref_map: Optional mapping from dropped node key to kept node key,
            built during the same_as contraction pass.  When provided, the
            resolved key is followed through the chain (cycle-guarded).

    Returns:
        The resolved node key as a string.  May not be present in the graph
        if neither the input nor its canonical form is a live node.
    """
    # Step 1: membership shortcut (node already keyed canonically).
    if in_graph(name):
        return name
    # Step 2: canonical fallback (casefolds speaker ids, normalizes surfaces).
    resolved = canonical(name)
    if coref_map is None:
        return resolved
    # Step 3: follow the drop→keep coref chain (cycle-guarded).
    seen: set[str] = set()
    while resolved in coref_map and resolved not in seen:
        seen.add(resolved)
        resolved = coref_map[resolved]
    return resolved


_SAME_AS_HONORIFICS = {
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "professor",
    "sir",
    "madam",
    "mister",
}


def _strip_honorifics(name: str) -> list[str]:
    """Return lowercased tokens of ``name`` with trailing-dot honorifics removed."""
    toks = []
    for raw in name.lower().split():
        t = raw.rstrip(".,")
        if t and t not in _SAME_AS_HONORIFICS:
            toks.append(t)
    return toks


def _safe_to_merge_surface(a: str, b: str) -> bool:
    """Heuristic gate: is it safe to merge two surface forms as the same entity?

    Two-stage check:

    1. Token-subset after honorific strip. "Mr. Yang" → {"yang"} is a
       subset of "Yang Ming" → {"yang", "ming"}. Safe.
    2. Single-token diff + Jaro-Winkler on the distinct tokens only.
       "Catherine Holmes" / "Katherine Holmes" share "holmes"; JW on
       "catherine" vs "katherine" ≈ 0.95 → accept. "Zhang Min" /
       "Wang Min" share "min"; JW on "zhang" vs "wang" ≈ 0.50 →
       reject. Multi-token symmetric difference always rejects.

    Returns ``False`` on empty or all-honorific inputs.
    """
    a_toks = _strip_honorifics(a)
    b_toks = _strip_honorifics(b)
    if not a_toks or not b_toks:
        return False
    a_set = set(a_toks)
    b_set = set(b_toks)
    if a_set <= b_set or b_set <= a_set:
        return True
    only_a = a_set - b_set
    only_b = b_set - a_set
    if len(only_a) != 1 or len(only_b) != 1:
        return False
    from rapidfuzz.distance import JaroWinkler

    jw = JaroWinkler.normalized_similarity(next(iter(only_a)), next(iter(only_b)))
    return jw >= 0.85


def serialize_subgraph_triples(subgraph) -> list[dict]:
    """Serialize a NetworkX subgraph into a list of triple dicts.

    Iterates ``subgraph.edges(data=True)`` and produces one dict per edge with
    keys ``subject``, ``predicate``, ``object``, ``relation_type``, and
    ``speaker_id``.  The ``predicate`` field is taken directly from the edge
    ``"predicate"`` attribute; ``relation_type`` defaults to ``"factual"`` when
    absent; ``speaker_id`` defaults to ``""`` when absent.

    The ``speaker_id`` field allows the cloud enrichment prompt to identify speaker
    endpoints and apply the speaker↔speaker exception (emit BOTH directions of a
    symmetric relation when both endpoints are speakers).

    Args:
        subgraph: A NetworkX (Multi)DiGraph subgraph view or instance.

    Returns:
        List of ``{"subject": str, "predicate": str, "object": str,
        "relation_type": str, "speaker_id": str}`` dicts, one per directed edge.
    """
    triples = []
    for src, tgt, data in subgraph.edges(data=True):
        triples.append(
            {
                "subject": str(src),
                "predicate": str(data.get("predicate", "")),
                "object": str(tgt),
                "relation_type": str(data.get("relation_type", "factual")),
                "speaker_id": str(data.get("speaker_id", "")),
            }
        )
    return triples


def enrich_graph(
    merger: "GraphMerger",
    *,
    model,
    tokenizer,
    extraction_config_provider: "Callable[[], ExtractionConfig]",
    neighborhood_hops: int,
    max_entities_per_pass: int,
    gc_disable: "Callable[[], None] | None" = None,
    gc_enable: "Callable[[], None] | None" = None,
) -> dict:
    """Post-merge graph-level cloud enrichment pass (Task #10).

    Runs at full consolidation over the cumulative ``merger.graph`` to
    capture cross-transcript second-order relations that per-transcript
    enrichment cannot see.  Folds in coreference resolution via
    ``same_as`` pairs emitted by the cloud response.

    Every cloud call this function makes runs through the SAME
    anonymize -> cloud -> de-anonymize contract as session-tier
    extraction (``paramem.graph.flows.SESSION_EXTRACT``: the
    ``anonymize``/``enrich`` stages anonymize and enrich, the
    ``deanonymize`` stage substitutes back), via the shared
    primitives in :mod:`paramem.cloud.placeholders`. The
    cumulative fold graph carries no entity types of its own (registry
    SPO triples have none; the merger's fallback for an untyped
    relation endpoint is ``entity_type="concept"``), so this function
    runs :func:`~paramem.cloud.anonymize.anonymize_transcript`
    (the SAME local-model anonymizer session-tier extraction uses, via
    :func:`~paramem.cloud.anonymize.anonymize`) over each chunk's triples
    first. That call is where ``ext_cfg.scrub`` (sourced from
    ``sanitization.scrub``) is honoured — it is the SOLE scope authority
    for THIS tier (no second detector): it decides which real names the
    chunk's ``chunk_mapping`` even contains, and its placeholders are
    threaded straight through — never re-derived or re-minted. That
    mapping is what feeds
    :func:`~paramem.graph.extractor.request_graph_enrichment`, which
    applies no scope gate of its own — it substitutes every entry it
    is handed (see its docstring for the contract and the accepted
    person-level ``same_as`` loss under the default scope, which only
    ever hands it person names). ``serialize_subgraph_triples`` stays a
    plain, un-anonymized serializer — anonymization happens on its
    output, not inside it. A local-anonymization parse failure fails
    the chunk closed (skips the cloud call for that chunk) rather than
    risk sending it unmasked — mirroring the session flow's ``anonymize``
    stage (:func:`~paramem.graph.stage_anonymize._stage_anonymize`)'s own
    fallback-to-local-only behaviour on the same failure mode.

    Forward-path privacy: the local anonymizer's mapping keys are
    real-name surfaces the LLM produced independently of the fold
    graph's own (canonical, lowercase, separator-folded) node keys —
    a re-cased, separator-varied, or diacritic-varied key (e.g. the
    LLM emits ``"Yang Ming"`` for fold-graph node ``"yang ming"``)
    would silently fail to substitute under a raw string comparison,
    leaking the real name into the cloud payload. This function
    reconciles that mismatch itself, per chunk, before building
    ``chunk_mapping`` (identity reconciliation, not
    classification): every ``_llm_mapping`` key is run through
    :func:`~paramem.utils.identity.canonical` and matched against
    the (also-canonicalized) node keys of ``chunk_nodes``; on a match
    the entry is re-keyed onto the actual node-key surface with the
    MODEL's own placeholder preserved verbatim, and on no match (or an
    ambiguous multiple-node match) it is dropped and counted in
    ``mapping_rekey_dropped``. The shared ``_substitute_whole_words``
    primitive (:mod:`paramem.cloud.placeholders`) keeps EXACT matching
    everywhere else — canonical folding there would let a mapped
    person name silently consume a lowercase common-noun homograph in
    free transcript text, and would defeat the fail-closed
    residual-token drop on the deanonymize side (see that module's
    docstring). A local ``_llm_mapping`` that comes back completely
    EMPTY is the anonymizer's own legitimate verdict that nothing in
    the chunk is in scope against ``scrub`` — egress PROCEEDS on
    that verdict, the same way
    ``mapping == {}`` proceeds at the session tier
    (:func:`~paramem.graph.flows.anonymize_turn`).
    Separately, when ``_llm_mapping`` DID name something but every
    entry is dropped by this reconciliation, while the chunk has
    real (non-speaker) node names, that residual is a
    classification/identity-match failure, not a scope verdict — so
    that chunk's cloud call is skipped (fail-closed) and counted in
    the returned ``privacy_skipped_chunks``. This residual (an
    entity the local model named but reconciliation could not match
    to a node) is owner-accepted and not otherwise engineered around
    — see ``benchmarking.md``.

    The function mutates ``merger.graph`` in place: first applying
    ``same_as`` node contractions, then inserting new edges tagged with
    the provenance attribute ``edge_source="graph_enrichment"`` (stored
    under :data:`paramem.memory.persistence._EDGE_SOURCE_ATTR`, not the
    NetworkX-reserved ``"source"`` field, so the tag survives persist).

    Each enrichment edge is a second-order fact derived from its chunk's
    source edges, so it inherits their assertion window: ``last_seen`` is
    the max (most recent) and ``first_seen`` the earliest non-empty
    (:func:`~paramem.graph.merger.min_nonempty`) across the chunk
    subgraph's edges, computed before same_as contraction mutates the
    graph.  This mirrors ``ConsolidationLoop._build_registry_true_relations``
    stamping ``last_seen``/``first_seen`` from bookkeeping.

    Early-return conditions (all return ``skipped=True``):
    - No local model available (``model is None``).
    - Graph has fewer than 10 nodes (floor — too little signal).
    - ``extraction_enrichment_provider`` is empty (no cloud provider configured).
    - Provider env-var is absent (API key not set).

    Chunking strategy:
    Entities are ranked by ``reinforcement_count`` descending.  For each
    focal entity an N-hop ego-graph is built (``radius=neighborhood_hops``).
    Chunks are deduplicated by node frozenset so overlapping ego-graphs do
    not re-send the same payload.  The number of chunks is capped at
    ``ceil(total_nodes / max_entities_per_pass)`` to prevent O(N) cloud
    calls on large graphs.

    Args:
        merger: The :class:`~paramem.graph.merger.GraphMerger` whose
            ``.graph`` is mutated in place and whose ``.removal_ledger``
            receives enrichment-collapsed ``same_as`` keys.
        model: Local model used for the anonymize/de-anonymize contract.
            ``None`` short-circuits the whole pass (``skip_reason="no_model"``).
        tokenizer: Tokenizer paired with ``model``.
        extraction_config_provider: Zero-arg callable returning the
            :class:`~paramem.graph.extraction_pipeline.ExtractionConfig` whose
            ``enrichment_provider``, ``enrichment_provider_model``,
            ``enrichment_provider_endpoint``, ``scrub``, and
            ``anonymize_token_envelope`` fields drive the cloud provider
            selection and cloud-egress contract. Deferred, not a resolved
            value: the caller's config typically lives behind a base-model
            holder that is released on the cloud-only path, and every skip
            condition below (``no_model``, ``floor``) must stay reachable
            without touching it. Called exactly once, past both guards.
        neighborhood_hops: Ego-graph radius used to build each chunk.
        max_entities_per_pass: Per-chunk node cap, also used to derive the
            chunk count ceiling.
        gc_disable: Optional zero-arg callable invoked before each
            ``model.generate()``-touching call (``anonymize``) to
            disable gradient checkpointing (HF silently disables the KV
            cache when checkpointing is active). Defaults to a no-op.
        gc_enable: Optional zero-arg callable invoked after each such call
            to restore gradient checkpointing. Defaults to a no-op.

    Returns:
        Diagnostics dict with keys:
            - ``chunks`` (int): number of cloud calls made.
            - ``new_edges`` (int): edges added to the graph.
            - ``same_as_merges`` (int): node contractions applied.
            - ``privacy_skipped_chunks`` (int): chunks skipped because
              the local anonymizer NAMED entities but NONE survived
              reconciliation for a chunk that had real (non-speaker)
              node names — see above. A local mapping that came back
              EMPTY outright is not counted here; it proceeds.
            - ``mapping_rekey_dropped`` (int): local-anonymizer mapping
              entries dropped because they named nothing (or an
              ambiguous multiple) in their chunk once reconciled
              against the chunk's actual node keys via ``canonical()``
              — see the ``anonymize(identity_domain=...)`` call below.
            - ``dropped_relations`` (int): relations
              :func:`~paramem.graph.extractor.request_graph_enrichment`
              individually dropped post-substitution (predicate-invariant
              plus residual-placeholder drops in
              :func:`~paramem.cloud.placeholders._apply_bindings`) —
              summed across every chunk's cloud call.  Replaces the
              retired ``totality_rejected_chunks`` (2026-07-22
              cloud-admission redesign): a cloud response naming an
              orphan/unresolvable token used to reject the WHOLE chunk
              delta; it now sheds only the offending relation(s), counted
              here. Distinct from ``privacy_skipped_chunks`` (which fires
              before any cloud call is made).
            - ``anonymize_slices`` (int): total local
              :func:`~paramem.cloud.anonymize.anonymize_transcript`
              (``generate()``) calls made across all chunks
              (``sum(payload.slices)``) — NOT a count of
              :func:`~paramem.cloud.anonymize.anonymize` invocations
              (exactly one of those is made per chunk that reaches the
              anonymize stage at all, opt-out included). A single
              ``anonymize()`` call may now cost more than one underlying
              local ``generate()`` call, since fact lists are packed into
              token-envelope-bounded slices (see
              :func:`~paramem.cloud.anonymize._slice_facts_to_envelope`);
              ``payload.slices`` counts those. Accumulated for every chunk
              whose payload ends up ``"ok"`` or ``"failed"`` (either way at
              least one ``generate()`` call was attempted); an
              ``"opted_out"`` chunk contributes 0 — ``opted_out_contract``
              sets ``slices=0`` because no local call is made on that path
              at all.
            - ``privacy_skipped_slices`` (int): sum of
              ``payload.slices_failed`` across all chunks — local calls
              whose slice was dropped fail-closed (parse or guard
              failure), so that slice's facts never reached the cloud.
              Distinct from ``privacy_skipped_chunks``, which counts WHOLE
              chunks skipped before any cloud call for that chunk (every
              slice failed); ``privacy_skipped_slices`` is the finer
              per-slice granularity and can be nonzero even when a
              chunk's OTHER slices succeeded and their facts egressed —
              that partial-withholding case also logs a per-chunk
              WARNING (``0 < slices_failed < slices``).
            - ``skipped`` (bool): ``True`` when enrichment was bypassed.
            - ``skip_reason`` (str | None): reason token when skipped —
              ``"no_model"``, ``"floor"``, or ``"cloud_egress_blocked"``
              (the shared cloud-admission verdict said no; the individual
              unmet terms are in the accompanying warning, not the token).
              A VRAM abort after N successful chunks is NOT a skip — the
              pass ran and made progress — so ``skipped`` stays ``False``
              and ``skip_reason`` stays ``None`` in that case; see
              ``aborted_reason`` below.
            - ``aborted_reason`` (str | None): ``None`` normally; ``"vram"``
              when a :class:`~paramem.utils.vram_guard.VramExhausted`
              stopped the chunk loop early.  The pass keeps whatever it
              already merged from completed chunks (degrade granularity is
              the CHUNK: ``anonymize()`` returns nothing until the whole
              call completes, so a fault mid-chunk discards only that
              chunk's own work) and returns normally rather than raising —
              the caller (:meth:`~paramem.training.consolidation.
              ConsolidationLoop._refine_consolidation_graph`) records an
              incident and the fold trains on the merged-but-unenriched
              graph.  Enrichment self-heals next cycle: the pass runs over
              the cumulative graph every fold.
    """
    _noop = lambda: None  # noqa: E731
    _gc_disable = gc_disable or _noop
    _gc_enable = gc_enable or _noop

    _empty = {
        "chunks": 0,
        "new_edges": 0,
        "same_as_merges": 0,
        "privacy_skipped_chunks": 0,
        "mapping_rekey_dropped": 0,
        "dropped_relations": 0,
        "aborted_reason": None,
        "anonymize_slices": 0,
        "privacy_skipped_slices": 0,
    }

    if model is None:
        logger.info("graph_enrichment: no local model — skipping")
        return {**_empty, "skipped": True, "skip_reason": "no_model"}

    graph = merger.graph
    node_count = graph.number_of_nodes()

    if node_count < 10:
        logger.info(
            "graph_enrichment: graph too small (%d nodes < 10 floor) — skipping",
            node_count,
        )
        return {**_empty, "skipped": True, "skip_reason": "floor"}

    # Graph-tier cloud enrichment shares the operator-configured provider
    # with session-tier extraction (anonymize → noise-filter → plausibility
    # chain).  Reading from the extraction config keeps both tiers
    # pointing at the same model + endpoint without an extra knob.
    # Resolved HERE, past both guards above: the provider read must never be
    # hoisted above the ``no_model`` skip (see the parameter's docstring).
    ext_cfg = extraction_config_provider()
    verdict = evaluate_cloud_egress(
        cloud_enabled=ext_cfg.cloud_enabled,
        provider=ext_cfg.enrichment_provider,
        model=ext_cfg.enrichment_provider_model,
        endpoint=ext_cfg.enrichment_provider_endpoint or None,
    )
    if not verdict.permitted:
        logger.warning("graph_enrichment: cloud egress refused — %s", "; ".join(verdict.gaps))
        return {**_empty, "skipped": True, "skip_reason": "cloud_egress_blocked"}

    provider = verdict.provider
    api_key = verdict.api_key
    filter_model = verdict.model
    endpoint = verdict.endpoint
    # Same cloud-egress scrub categories as session-tier extraction
    # (``sanitization.scrub`` -> ``ExtractionConfig.scrub`` at
    # bootstrap) — feeds the per-chunk ``anonymize_transcript`` call
    # below, which is the prompt-side scope authority at this tier (see
    # the function docstring).
    scrub = ext_cfg.scrub
    # Anonymization prompt templates — loaded ONCE per enrichment pass
    # (identical for every chunk), not per chunk, so a calibration
    # override / provenance-recording chokepoint (:func:`_load_prompt`)
    # is still honoured without re-reading the file per chunk.
    # ``anonymization_facts.txt`` (not the session tier's
    # ``anonymization.txt``): this tier has no transcript at all
    # (``transcript=""`` below), so the facts-only variant — same core
    # contract, no transcript-rewrite half, output contract
    # ``{"mapping": {...}}`` only — is the correct template; the mapping-validity
    # rule is satisfied via the transcript-empty leg.
    anon_prompt = _load_prompt("anonymization_facts.txt", required=True)
    anon_system = _load_prompt("anonymization_system.txt", required=True)
    max_entities = max(1, max_entities_per_pass)
    hops = max(1, neighborhood_hops)

    # Rank nodes by reinforcement descending.
    nodes_by_recurrence = sorted(
        graph.nodes(data=True),
        key=lambda nd: nd[1].get("reinforcement_count", 0),
        reverse=True,
    )

    # Build deduplicated chunks from N-hop ego-graphs.
    undirected = graph.to_undirected(as_view=True)
    seen_chunks: set[frozenset] = set()
    chunks: list[list[str]] = []
    chunk_cap = max(1, math.ceil(node_count / max_entities))

    for focal, _ in nodes_by_recurrence:
        if len(chunks) >= chunk_cap:
            break
        if focal not in undirected:
            continue
        ego = nx.ego_graph(undirected, focal, radius=hops)
        nodes = list(ego.nodes)
        if len(nodes) > max_entities:
            # Trim: keep focal + top-(cap-1) neighbours by degree.
            neighbours = sorted(
                (n for n in nodes if n != focal),
                key=lambda n: undirected.degree(n),
                reverse=True,
            )
            nodes = [focal] + neighbours[: max_entities - 1]
        key = frozenset(nodes)
        if key in seen_chunks:
            continue
        seen_chunks.add(key)
        chunks.append(nodes)

    total_merges = 0
    calls_made = 0
    privacy_skipped_chunks = 0
    mapping_rekey_dropped = 0
    dropped_relations = 0
    # Slice-level counters (fact-boundary slicing) — see the docstring's
    # Returns section. Accumulated per chunk from payload.slices /
    # payload.slices_failed, whether that chunk's payload ends up "ok" or
    # "failed".
    anonymize_slices = 0
    privacy_skipped_slices = 0
    # None normally; set to "vram" when a VramExhausted stops the chunk loop
    # early (see the except block below and the docstring's Returns section).
    aborted_reason: str | None = None
    # Accumulates ik_keys from edges dropped by successful same_as contractions.
    # Keys are written to merger.removal_ledger after the loop completes
    # so the classifier can distinguish intended enrichment-driven removals from
    # genuine reconstruction failures.
    _collapsed_ik: dict[str, str] = {}  # ik_key → keep node
    # Accumulate Relation objects across all chunks; the edge-count delta is
    # computed after merger.merge_relations so merger deduplication is counted.
    enrichment_relations: list[Relation] = []
    _edges_before = graph.number_of_edges()

    for chunk_idx, chunk_nodes in enumerate(chunks):
        try:
            chunk_subgraph = graph.subgraph(chunk_nodes)
            # Enrichment edges are second-order facts derived from this
            # chunk's source edges, so they inherit the chunk's assertion
            # window rather than landing untimed.  Computed from the
            # subgraph view BEFORE any same_as contraction below mutates
            # ``graph`` (contraction would change what the view sees).
            _chunk_last_seen = ""
            _chunk_first_seen = ""
            for _u, _v, _edata in chunk_subgraph.edges(data=True):
                _chunk_last_seen = max(_chunk_last_seen, _edata.get("last_seen") or "")
                _chunk_first_seen = min_nonempty(_chunk_first_seen, _edata.get("first_seen") or "")
            triples = serialize_subgraph_triples(chunk_subgraph)
            # The fold graph carries no entity types of its own:
            # registry SPO triples have none, and the merger's fallback
            # for a relation endpoint that isn't already a known Entity
            # is entity_type="concept" (GraphMerger._merge_relations).
            # Node attributes are therefore NOT a usable scope source at
            # this tier — reading them here previously masked ONLY the
            # speaker (the sole node synthesized with entity_type=
            # "person") and sent every other real name to the cloud
            # verbatim.
            #
            # The local model is the SOLE scope authority instead: run
            # THE one anonymize chain (:func:`~paramem.cloud.anonymize.
            # anonymize`) directly over this chunk's ``triples`` — they
            # are already a valid ``facts: list[dict]`` (subject/
            # predicate/object/relation_type/speaker_id), so no
            # ``Relation``/``SessionGraph`` round trip is needed to reach
            # this cloud-package call (interface narrowing, 2026-07-21:
            # ``anonymize`` takes facts directly, never a graph carrier).
            # ``transcript=""`` — there is no transcript at this tier.
            # ``identity_domain=chunk_nodes`` drives (A)'s
            # identity-reconciliation step (5) — the local model's
            # mapping is keyed by whatever real-name surface it
            # independently produced (e.g. "Yang Ming"); the fold
            # graph's own node keys are already canonicalized (e.g.
            # "yang ming") — a raw comparison between the two silently
            # misses every re-cased/separator-varied/diacritic-varied
            # key, so a ``chunk_mapping`` entry keyed on "Yang Ming"
            # would never match the "yang ming" text
            # ``_substitute_whole_words`` compares against inside
            # ``triples``' subject/object fields below. (A)'s
            # domain-scoped fail-closed guard (step 6) is derived from
            # ``facts`` — the SLICE's own facts at evaluation time (fact
            # boundaries slice ``triples`` into token-envelope-bounded
            # pieces), NOT
            # necessarily the whole ``triples`` list, and NOT the same as
            # what this chunk ultimately sends to cloud: under partial
            # withholding ``payload.facts`` is ``triples`` MINUS every
            # fail-closed slice's facts (see ``AnonymizedContract``'s
            # ``facts`` field docstring) — never from ``identity_domain``
            # (see that function's docstring for why the two domains must
            # stay distinct). The anonymization prompt instructs the model
            # to leave speaker{N} ids verbatim (never map them), so a
            # speaker anchor never becomes a ``chunk_mapping`` entry here
            # — it is already anonymous and reaches the cloud payload bare
            # by design (ONE-lowercase-speaker{N} invariant), with no
            # mint/restore round trip needed.
            #
            # ``_chunk_session_graph`` carries no relations of its own —
            # it exists ONLY as the diagnostics sink
            # ``request_graph_enrichment`` writes the binding-collision
            # findings to below; it is NOT how this chunk's facts reach
            # that call — ``request_graph_enrichment`` reads
            # ``payload.facts`` (the anonymize contract's egress-cleared
            # subset of ``triples``), never ``triples`` directly (see that
            # function's ``anon_triples = insert_placeholders(payload.facts,
            # payload.forward)``, ``paramem/graph/extractor.py``).
            _chunk_session_graph = SessionGraph(session_id="__graph_enrichment__", timestamp="")
            # anonymize calls model.generate() internally (CLAUDE.md:
            # gradient checkpointing must be disabled around ANY
            # model.generate() site — HF silently disables the KV cache
            # when checkpointing is active).  The gradient-checkpointing
            # pair stays HERE, at the call site — it is a trainer
            # concern, not a cloud-egress concern.
            #
            # Chunk-identifying context for the per-call telemetry logged
            # inside anonymize_transcript (paramem.cloud.anonymize): that
            # call's "anonymize_transcript prompt: chars=... tokens=..."
            # line has no notion of chunk identity, so this line — logged
            # immediately before it fires — is what lets a fold's log
            # stream be read as a per-chunk sequence rather than one
            # anonymous "anonymize" entry per call.
            logger.info(
                "graph_enrichment: anonymize chunk %d/%d nodes=%d triples=%d",
                chunk_idx + 1,
                len(chunks),
                len(chunk_nodes),
                len(triples),
            )
            _gc_disable()
            try:
                payload = anonymize(
                    triples,
                    model,
                    tokenizer,
                    transcript="",
                    scrub=scrub,
                    identity_domain=chunk_nodes,
                    token_envelope=ext_cfg.anonymize_token_envelope,
                    user_prompt_template=anon_prompt,
                    system_prompt=anon_system,
                )
            finally:
                _gc_enable()
            # Slice-level counters — accumulated for every chunk that
            # reaches this point, whether the payload ends up "ok" or
            # "failed" (payload.slices / payload.slices_failed are
            # meaningful in both cases: a "failed" payload means every
            # slice failed, so slices_failed == slices).
            anonymize_slices += payload.slices
            privacy_skipped_slices += payload.slices_failed
            if payload.status == "failed":
                # Two distinct fail-closed causes collapse to the SAME
                # status; ``payload.failure`` names which one fired —
                # see ``AnonymizedContract``'s docstring.  NOT
                # ``payload.rekey_dropped``: that count stays ``0`` in
                # BOTH the parse-failure case AND the guard case where
                # the model's mapping was dropped entirely by shape
                # validation before the reconciliation loop that
                # increments it ever ran — a count can't discriminate
                # a cause it may legitimately be zero for either way.
                if payload.failure == "guard":
                    # The domain-scoped fail-closed guard fired: the
                    # local anonymizer named entities but none survived
                    # reconciliation onto this chunk's actual node keys
                    # (or none survived the model's own placeholder-shape
                    # validation) — a classification/identity-match
                    # failure, not a scope verdict.
                    privacy_skipped_chunks += 1
                    mapping_rekey_dropped += payload.rekey_dropped
                    logger.warning(
                        "graph_enrichment: local anonymizer named entities but none "
                        "survived reconciliation for a %d-triple chunk — skipping "
                        "cloud call (fail-closed)",
                        len(triples),
                    )
                else:
                    # Mirrors the session tier's fail-closed behaviour:
                    # the ``anonymize`` stage falls back to LOCAL
                    # plausibility on anonymization parse failure rather
                    # than ever sending unmasked content to the cloud.
                    # This chunk
                    # has no trustworthy type source at all when the
                    # local call fails to parse — skip the cloud call
                    # for this chunk entirely rather than sending it
                    # unmasked.
                    logger.warning(
                        "graph_enrichment: local anonymization failed for chunk — "
                        "skipping cloud call (fail-closed)"
                    )
                continue
            if payload.slices_failed:
                # status != "failed" here, so this is necessarily PARTIAL
                # withholding (0 < slices_failed < slices): at least one
                # slice of this chunk dropped fail-closed while the rest
                # survived and will egress below — the operator-visible
                # signal the whole-chunk-only privacy_skipped_chunks
                # counter cannot surface on its own.
                logger.warning(
                    "graph_enrichment: %d/%d slice(s) dropped fail-closed for this "
                    "chunk (partial withholding) — %d slice(s) still egress",
                    payload.slices_failed,
                    payload.slices,
                    payload.slices - payload.slices_failed,
                )
            mapping_rekey_dropped += payload.rekey_dropped
            if payload.rekey_dropped:
                logger.warning(
                    "graph_enrichment: dropped %d local-anonymizer mapping "
                    "entry(ies) naming nothing in this chunk (post-canonical "
                    "reconciliation)",
                    payload.rekey_dropped,
                )
            result = request_graph_enrichment(
                payload,
                _chunk_session_graph,
                api_key,
                provider,
                filter_model,
                endpoint,
            )
            calls_made += 1
            if result is None:
                logger.warning("graph_enrichment: Cloud call returned None for chunk")
                continue
            new_rels, same_as_pairs, _raw, dropped_relation_count = result
            # The count of relations this chunk's cloud response had
            # individually dropped post-substitution — arrives as a
            # returned value, not as a mutation on
            # ``_chunk_session_graph.diagnostics`` read back before the
            # throwaway graph is discarded.
            dropped_relations += dropped_relation_count
        except VramExhausted as exc:
            # Fold-level degrade, not a chunk-level skip: a VRAM
            # exhaustion here means the driver is under pressure, and
            # continuing the loop would very likely re-fault on the next
            # chunk and burn minutes re-faulting — so this stops the WHOLE
            # pass rather than trying the next chunk (``break``, not
            # ``continue``).  Chunks that already completed keep their
            # enrichment relations: they were already appended to
            # ``enrichment_relations`` and are merged by the post-loop
            # ``merger.merge_relations`` call below, which still runs after
            # the ``break``.  The faulted chunk's own work is lost — no
            # slice-level salvage (``anonymize()`` returns nothing until the
            # whole call completes).  Enrichment self-heals next cycle: this
            # pass runs over the cumulative graph every fold, so the skipped
            # chunk's relations are simply re-discovered later.  The caller
            # (``ConsolidationLoop._refine_consolidation_graph``) records an
            # operator-visible incident from ``aborted_reason`` and the fold
            # proceeds to training on the merged-but-unenriched graph.
            logger.warning(
                "graph_enrichment: VRAM exhausted on chunk %d/%d — stopping "
                "the enrichment pass, keeping %d already-merged chunk(s): %s",
                chunk_idx + 1,
                len(chunks),
                chunk_idx,
                exc,
            )
            aborted_reason = "vram"
            break
        except RuntimeError as exc:
            # Sticky, process-fatal CUDA context faults (vram_guard.
            # is_fatal_cuda_fault's contract: recovery is os._exit + process
            # restart, NEVER an in-process release) must never be swallowed
            # here — continuing to the next chunk would run every subsequent
            # GPU call against a poisoned context.
            if is_fatal_cuda_fault(exc):
                raise
            # Narrow by design otherwise.  The CUDA "device not ready"
            # driver-fault class is converted to ``VramExhausted`` by
            # ``vram_scope`` before it ever reaches this branch (see the
            # ``except VramExhausted`` above) — this branch is UNREACHABLE
            # for that class in production.  It exists for a genuinely
            # different, non-driver-fault ``RuntimeError`` surfacing from
            # the local ``generate()`` inside ``anonymize_transcript``.  The
            # cloud leg cannot raise — ``_cloud_call`` and the response
            # parse both return ``None`` on failure — so a broad ``except
            # Exception`` here could only ever swallow a programming error
            # (e.g. a KeyError from a malformed prompt template), silently
            # disabling graph enrichment forever.  Those must kill the fold.
            # Widen by NAME if a legitimate runtime condition surfaces;
            # never back to ``Exception``.
            logger.warning(
                "graph_enrichment: runtime error during chunk — %s: %s",
                type(exc).__name__,
                exc,
            )
            continue

        # Apply same_as contractions FIRST so subsequent edge inserts
        # reference canonical nodes. Gate on:
        #   1. Both endpoints exist in the live graph.
        #   2. Surface-form safety gate (token-subset + Jaro-Winkler).
        #
        # There is deliberately NO cross-chunk proposal memo here.  Chunks are
        # overlapping ego-graphs, so the same entity pair is genuinely proposed
        # repeatedly, but each proposal carries its OWN surface strings and the
        # gate below is a function of those surfaces (it tokenizes on
        # whitespace, so "Yang-Ming" and "Yang Ming" are NOT interchangeable
        # inputs).  Any memo keyed on node identity — or on any casefolded
        # surface form — is strictly coarser than the gate, so it would let the
        # first-visited chunk's verdict silently decide the pair for every later
        # chunk, making the outcome depend on focal-entity iteration order.
        # Re-running the gate is a cheap pure call; the repeat-proposal case is
        # already absorbed by rule 1, since a successful contraction removes the
        # dropped node from the graph.
        coref_map: dict[str, str] = {}

        # _in_graph closure passed to resolve_to_node_key.
        _in_graph = graph.__contains__

        for pair in same_as_pairs:
            keep, drop = pair[0], pair[1]
            if keep == drop:
                continue
            # Guard: skip same_as pairs where BOTH surface strings are
            # speaker ids.  Speaker identity is authoritative (voice/enrollment);
            # it must never be coalesced by a surface-similarity heuristic.
            # Two speaker-id surfaces are either the SAME speaker (already
            # unified by canonical node-keying, so no merge needed) or
            # DIFFERENT speakers (must never merge — Jaro-Winkler treats the
            # distinguishing digit as a typo and would incorrectly merge
            # speaker0/speaker1).  Skip unconditionally: the pair is always
            # either redundant or catastrophically wrong.
            # Note: the ``keep_canon == drop_canon`` post-resolution check
            # handles the casing-only case (Speaker0/speaker0), but does NOT
            # catch distinct speaker ids (speaker0 ≠ speaker1) — this guard
            # is load-bearing for the distinct-speaker scenario.
            if is_speaker_id(keep) and is_speaker_id(drop):
                logger.debug(
                    "graph_enrichment: same_as skip — both surfaces are speaker ids %r / %r",
                    keep,
                    drop,
                )
                continue
            # Resolve to actual node keys via resolve_to_node_key
            # (membership shortcut then canonical).  Keep _safe_to_merge_surface
            # on the ORIGINAL SURFACE strings (fuzzy layer-2 check; done
            # before resolution).
            keep_canon = resolve_to_node_key(keep, _in_graph)
            drop_canon = resolve_to_node_key(drop, _in_graph)
            if keep_canon == drop_canon:
                continue
            if keep_canon not in graph or drop_canon not in graph:
                logger.debug(
                    "graph_enrichment: same_as skip — keep=%r drop=%r not both in graph",
                    keep_canon,
                    drop_canon,
                )
                continue
            if not _safe_to_merge_surface(keep, drop):
                logger.info(
                    "graph_enrichment: same_as rejected by surface gate — %r / %r",
                    keep,
                    drop,
                )
                continue
            # Collect ik_keys from edges in both directions that will become
            # self-loops (and be dropped by self_loops=False) on success.
            # Use inner-dict .values() iteration — MultiDiGraph get_edge_data
            # returns {edge_id: data_dict}; do NOT treat the outer dict as data.
            _pending: dict[str, str] = {}
            for _u, _v in [(keep_canon, drop_canon), (drop_canon, keep_canon)]:
                for _edata in (graph.get_edge_data(_u, _v) or {}).values():
                    _ik = _edata.get(_IK_KEY_ATTR)
                    if _ik:
                        _pending[_ik] = keep_canon
            try:
                nx.contracted_nodes(graph, keep_canon, drop_canon, self_loops=False, copy=False)
                total_merges += 1
                # Success — absorb pending keys into the accumulator.
                _collapsed_ik.update(_pending)
                coref_map[drop_canon] = keep_canon
                logger.debug("graph_enrichment: contracted %r → %r", drop_canon, keep_canon)
            except Exception as exc:
                logger.warning(
                    "graph_enrichment: same_as contraction failed %r → %r: %s",
                    drop_canon,
                    keep_canon,
                    exc,
                )

        # Build Relation objects from cloud-emitted new_rels for this chunk.
        # Endpoint surface rule: speaker endpoints pass their canonical
        # key (the speaker_id), non-speaker endpoints pass the display surface.
        for rel in new_rels:
            if not isinstance(rel, dict):
                continue
            # Remap endpoints through this chunk's coref map so edges
            # referencing a to-be-dropped node still land on the canonical.
            # resolve_to_node_key(membership shortcut → canonical → coref chain).
            subj_canon = resolve_to_node_key(rel.get("subject", ""), _in_graph, coref_map)
            raw_pred = rel.get("predicate", "")
            obj_canon = resolve_to_node_key(rel.get("object", ""), _in_graph, coref_map)
            rtype = rel.get("relation_type", _FALLBACK_RTYPE)
            if rtype not in _VALID_RTYPES:
                rtype = _FALLBACK_RTYPE
            if not (subj_canon and raw_pred and obj_canon):
                continue

            # Choose endpoint surface string per endpoint.
            # Speaker endpoint (node carries speaker_id attribute): pass the node
            # key (lowercase canonical speaker{N} id) so
            # paramem.graph.merger._synth_speaker_entities can emit the
            # correct Entity from the canonical key.
            # Non-speaker endpoint: pass the display surface from node attributes.
            def _endpoint_str(canon: str) -> str:
                _n = graph.nodes.get(canon, {})
                if _n.get("speaker_id"):
                    return canon
                return _n.get("attributes", {}).get("name", canon)

            subj_endpoint = _endpoint_str(subj_canon)
            obj_endpoint = _endpoint_str(obj_canon)

            if not (subj_endpoint and obj_endpoint and subj_endpoint != obj_endpoint):
                continue

            try:
                confidence = float(rel.get("confidence", 0.8))
            except (TypeError, ValueError):
                confidence = 0.8
            # Safety net for the prompt-level 0.7 rule: discard low-confidence
            # enriched edges even if the model ignored its own instruction.
            if confidence < 0.7:
                continue

            # Derive speaker_id from the subject node's speaker_id attribute.
            _subj_sid = graph.nodes.get(subj_canon, {}).get("speaker_id", "")

            enrichment_relations.append(
                Relation(
                    subject=subj_endpoint,
                    predicate=raw_pred,
                    object=obj_endpoint,
                    relation_type=rtype,  # type: ignore[arg-type]
                    confidence=confidence,
                    speaker_id=_subj_sid,
                    symmetric=bool(rel.get("symmetric")),
                    edge_source="graph_enrichment",
                    last_seen=_chunk_last_seen,
                    first_seen=_chunk_first_seen,
                )
            )

    # Route all accumulated enrichment relations through the merger so they
    # receive full Case-1/Case-3 treatment (dedup, edge-source stamp, speaker_id).
    if enrichment_relations:
        merger.merge_relations(
            enrichment_relations,
            session_id="__graph_enrichment__",
            log_label="enrichment relations",
            resolve_contradictions=False,
        )

    # Edge-count delta (the merger may absorb some via Case-1).
    total_new = max(0, graph.number_of_edges() - _edges_before)

    logger.info(
        "graph_enrichment: provider=%s chunks=%d new_edges=%d same_as_merges=%d "
        "privacy_skipped_chunks=%d mapping_rekey_dropped=%d dropped_relations=%d "
        "anonymize_slices=%d privacy_skipped_slices=%d aborted_reason=%s",
        provider,
        calls_made,
        total_new,
        total_merges,
        privacy_skipped_chunks,
        mapping_rekey_dropped,
        dropped_relations,
        anonymize_slices,
        privacy_skipped_slices,
        aborted_reason,
    )
    # Write enrichment-collapsed ik_keys to the merger's removal ledger so the
    # drift classifier can route them to drift_intended_removal rather than
    # drift_genuine_loss.  Only keys from SUCCESSFUL contractions are written
    # (failures were discarded from _pending before _collapsed_ik was updated).
    # No ``survivor_key``: a same_as contraction merges NODES, and the edges
    # recorded here are the ones that became self-loops and were dropped —
    # the fact does not carry forward under another indexed key, so there is
    # no maturity to inherit.  ``keep_node`` is the surviving NODE.
    for _ik, _keep in _collapsed_ik.items():
        merger.removal_ledger[_ik] = {
            "reason": "enrichment_same_as",
            "keep_node": _keep,
        }
    return {
        "chunks": calls_made,
        "new_edges": total_new,
        "same_as_merges": total_merges,
        "privacy_skipped_chunks": privacy_skipped_chunks,
        "mapping_rekey_dropped": mapping_rekey_dropped,
        "dropped_relations": dropped_relations,
        "skipped": False,
        "skip_reason": None,
        "aborted_reason": aborted_reason,
        "anonymize_slices": anonymize_slices,
        "privacy_skipped_slices": privacy_skipped_slices,
    }
