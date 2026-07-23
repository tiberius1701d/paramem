"""``SESSION_EXTRACT`` — the declarative flow topology for session-tier
extraction, its ``extract_graph`` entry point, and the conversation-egress
composite ``anonymize_turn``.

This module owns the flow TOPOLOGY (``SESSION_EXTRACT``, the stage bodies
that don't have a dedicated module of their own, and ``extract_graph``
itself) — the layer above ``paramem.graph.extractor`` (primitives),
``paramem.graph.stage_anonymize`` and ``paramem.graph.stage_enrich`` (the
two stage modules carved out because they compose SHARED components or
carry substantial stage-local logic of their own). Import layering is
one-directional and acyclic: ``extractor.py`` (primitives only) is
imported by ``stage_anonymize.py``/``stage_enrich.py`` (peers, neither
imports the other), which are in turn imported by this module. Nothing
downstream of this module ever needs to be imported back into
``extractor.py``, ``stage_anonymize.py`` or ``stage_enrich.py``.

Each stage body below is a VERBATIM lift of the corresponding block that
used to live inline in ``extractor.extract_graph`` (``local_extract``,
``second_order_extract``) or in the tail of ``extractor._cloud_pipeline``
(``deanonymize``, ``rebuild``) — same primitives, same arguments, same
``phase_trace`` scopes. ``run_flow`` does not open phases itself; every
``phase_trace`` call below is exactly the one the imperative version made.

``anonymize_turn`` lives here rather than in ``extractor.py`` for the
same reason as the rest of this module: per
``paramem/cloud/admission.py``'s placement principle ("a primitive every
tier needs, owned by none of them" — why that module is a stdlib-only
leaf), :func:`~paramem.cloud.admission.evaluate_cloud_egress` is the
primitive, :func:`~paramem.cloud.anonymize.anonymize` is the shared
component operating on a plain fact list, and ``anonymize_turn`` is a
COMPOSITE of ``extract_graph`` (this module) and ``anonymize`` — so it
belongs at the flow layer, not with the primitives. It is the
conversation-egress composite; its callers are
``paramem/server/inference.py`` and
``scripts/dev/calibrate_cloud_anonymizer.py``.
"""

from __future__ import annotations

import dataclasses
import logging
from datetime import datetime, timezone
from pathlib import Path

from paramem.cloud.admission import evaluate_cloud_egress
from paramem.cloud.anonymize import (
    _DEFAULT_ANONYMIZER_MAX_TOKENS,
    AnonymizedContract,
    anonymize,
    opted_out_contract,
)
from paramem.cloud.deanonymize import deanonymize_facts
from paramem.graph.empty_cause import (
    CAUSE_DEANON_JUDGE,
    CAUSE_DEANON_SUBSTITUTION,
    CAUSE_SCALAR_PARTITION,
    CAUSE_SCHEMA_VALIDATION,
)
from paramem.graph.extractor import (
    _DEFAULT_FILTER_MAX_TOKENS,
    _DEFAULT_FILTER_TEMPERATURE,
    DEFAULT_SYSTEM_PROMPT_FILENAME,
    DEFAULT_USER_PROMPT_FILENAME,
    _fallback_plausibility_on_raw,
    _record_binding_diagnostics,
    _run_local_extraction,
    _vram_snapshot,
    judge_plausibility,
)
from paramem.graph.flow import StageContext, StageSpec, StageState, run_flow
from paramem.graph.phase_trace import chain_seed, chain_stopped, extraction_trace, phase_trace
from paramem.graph.prompts import _load_prompt
from paramem.graph.relation_build import (
    apply_rebuild,
    build_relations,
    partition_scalar_facts,
    recovery_gate,
)
from paramem.graph.schema import SessionGraph, facts_from_relations
from paramem.graph.stage_anonymize import _stage_anonymize
from paramem.graph.stage_enrich import _stage_enrich
from paramem.models.loader import base_model_inference
from paramem.server.session_buffer import SessionBuffer
from paramem.utils.identity import is_speaker_id

logger = logging.getLogger(__name__)


# Filename of the second-order extraction pass — it extracts facts ABOUT
# the named entities local_extract (first-order) surfaced, recovering a
# named relative's own attribute (location, job, trait) when Mistral 7B
# collapses a single-relative clause ("my brother Nadeem lives in Porto")
# into ONE relation instead of two. Reuses DEFAULT_SYSTEM_PROMPT_FILENAME —
# no second-order-specific system prompt.
DEFAULT_SECOND_ORDER_USER_PROMPT_FILENAME = "extraction_second_order.txt"


def _stage_local_extract(ctx: StageContext, state: StageState) -> StageState:
    """``local_extract`` stage body — always runs.

    Local-model extraction step. Raw output is the canonical isolation
    point for the extraction prompt (calibration/debugging diffs prompt
    variants by comparing this raw_output, before any downstream phase
    has had a chance to mutate the result).
    """
    graph = _run_local_extraction(
        ctx.model,
        ctx.tokenizer,
        ctx.transcript,
        ctx.session_id,
        ctx.speaker_id,
        ctx.temperature,
        ctx.max_tokens,
        ctx.prompts_dir,
        ctx.speaker_name,
        ctx.system_prompt_filename,
        ctx.user_prompt_filename,
        ctx.model_alias,
        ctx.seed,
        ctx.timestamp,
        ctx.source_type,
        phase_name="local_extract",
    )
    return StageState(graph=graph)


def _has_named_non_speaker_person(graph: SessionGraph) -> bool:
    """Gate for the ``second_order_extract`` phase: does the pass-1 graph
    contain a named (proper-name) person entity other than a speaker id?

    ``local_extract`` is speaker-centric, so a clause naming a non-speaker
    person by relationship ("my brother Nadeem lives in Porto") tends to
    keep only the speaker->person edge and drop that person's OWN fact
    (measured on Mistral 7B). A surviving named non-speaker person entity
    is exactly the set of people this failure mode can hit, and exactly
    what ``second_order_extract`` re-extracts facts about — so it is the
    gate: no such entity means nothing to recover, and the caller skips
    the phase entirely (no LLM call, no phase_trace record).
    """
    return any(e.entity_type == "person" and not is_speaker_id(e.name) for e in graph.entities)


def _stage_second_order_extract(ctx: StageContext, state: StageState) -> StageState:
    """``second_order_extract`` stage body — gated by
    :func:`_has_named_non_speaker_person` (the flow's ``applies_when``).

    Extracts facts ABOUT the named entities ``local_extract`` (first-order)
    surfaced, recovering a named relative's own attribute (location, job,
    trait) when local_extract collapsed a single-relative clause ("my
    brother Nadeem lives in Porto") into ONE relation instead of two
    (measured Mistral 7B failure mode).
    """
    graph = state.graph
    second_order_graph = _run_local_extraction(
        ctx.model,
        ctx.tokenizer,
        ctx.transcript,
        ctx.session_id,
        ctx.speaker_id,
        ctx.temperature,
        ctx.max_tokens,
        ctx.prompts_dir,
        ctx.speaker_name,
        ctx.system_prompt_filename,
        DEFAULT_SECOND_ORDER_USER_PROMPT_FILENAME,
        ctx.model_alias,
        ctx.seed,
        ctx.timestamp,
        ctx.source_type,
        phase_name="second_order_extract",
    )
    # Plain union: the second-order pass contributes recovered facts
    # (recall) — it is not a dedup boundary. Predicate-surface drift for a
    # fact both passes capture (e.g. "picked_up" vs "picks_up") is a
    # PRE-EXISTING pipeline phenomenon (the same drift already happens
    # across sessions) and is handled downstream exactly as cross-session
    # drift is: triple-identity dedup at GraphMerger._upsert_relation Case 1
    # (paramem/graph/merger.py:580) and, when enabled,
    # refinement_normalization's (subject, object)-grouped predicate-synonym
    # fold. A redundant near-dup key is benign — not a wrong answer, at
    # worst a redundant indexed key — so it is deliberately NOT special-cased
    # here; filtering on (subject, object) identity would also destroy
    # genuinely distinct same-(s,o) facts (e.g. born_in + lives_in).
    graph.relations.extend(second_order_graph.relations)
    graph.entities.extend(second_order_graph.entities)
    return StageState(graph=graph)


def _session_egress_permitted(ctx: StageContext) -> bool:
    """``anonymize``/``enrich``'s cloud-admission gate, as a flow predicate.

    Routes the session-tier question through the one shared component
    (:func:`~paramem.cloud.admission.evaluate_cloud_egress`) rather
    than restating its terms as a boolean expression in the stage spec.
    Logs every unmet term in a single line when the answer is no — a flow
    ``enabled_when`` skip is otherwise silent, and the operator whose
    ``cloud_enabled`` is on but whose key is unset needs to be told which
    term failed.

    Args:
        ctx: The run-constant flow context.

    Returns:
        ``True`` when a cloud enrichment call may be placed for this run.
    """
    verdict = evaluate_cloud_egress(
        cloud_enabled=ctx.cloud_enabled,
        provider=ctx.enrichment_provider,
        model=ctx.enrichment_provider_model,
        endpoint=ctx.enrichment_provider_endpoint,
    )
    if not verdict.permitted:
        logger.info("Skipping cloud enrichment — %s", "; ".join(verdict.gaps))
    return verdict.permitted


def _stage_deanonymize(ctx: StageContext, state: StageState) -> StageState:
    """``deanonymize`` stage body — placeholders back to real names.

    Three things, in this order, because the order is load-bearing:

    1. The ``deanon`` phase: pure dict substitution restoring real names
       from placeholders (no LLM call, so ``raw_output`` stays ``None``).
       Facts still carrying an unresolved placeholder, and facts with a
       placeholder glued into the predicate, are dropped and recorded.
    2. The scalar partition. It is OWNED by ``rebuild``
       (``paramem.graph.relation_build``) but INVOKED here, before the
       judge below, and that placement is deliberate: scalars are URLs,
       emails, DOIs and version-tagged tool names, and the judge's drop
       rule R6 targets a "dot-separated or URI-shaped namespaced token"
       (``configs/prompts/cloud_plausibility.txt``). Partitioning after
       the judge would expose exactly the values the partition protects.
       When the partition absorbs EVERY surviving fact (a scalar-only
       session) it records ``CAUSE_SCALAR_PARTITION`` — the ``routing``
       kind, which tells ``rebuild``'s recovery gate the resulting empty
       relation set is legitimate rather than a loss.
    3. The deanon-stage plausibility judge (local model, real names),
       which receives the ORIGINAL real-name transcript — it runs
       locally on de-anonymized facts, so there is no reason to hand it
       the anonymized text.

    The mid-stage ``chain_stopped()`` check after the ``deanon`` phase is
    the one the imperative version made: a calibration caller stopping at
    ``deanon`` must not get the partition or the judge.
    """
    graph = state.graph
    scope = state.scope
    facts = state.facts
    empty_cause = state.empty_cause

    # THE ONE call to ``deanonymize_facts`` for this response (the
    # ``cloud_enrich`` phase's own call was retired along with the
    # whole-delta totality gate — see
    # ``paramem.graph.extractor._apply_enrichment_delta``, which now
    # decides per-triple resolvability before this stage ever runs).  This
    # call is where the actual substitution happens: whatever survived to
    # this point (post per-triple accept/drop/revert AND post anon-stage
    # plausibility, if it ran) has its placeholders resolved to real
    # names, then the fail-closed residual sweep in ``_apply_bindings``
    # sheds any fact still carrying an unresolved token.
    with phase_trace("deanon") as t:
        deanon_input_count = len(facts)
        deanon_result = deanonymize_facts(scope, facts)
        _record_binding_diagnostics(graph, deanon_result)
        deanon_facts = deanon_result.facts
        predicate_dropped = deanon_result.predicate_dropped
        residual_dropped = deanon_result.residual_dropped
        # predicate_dropped: facts cloud returned with a placeholder glued
        # into the predicate field (_apply_bindings' step 1, pre-
        # substitution).  residual_dropped: facts still carrying an
        # unresolved placeholder after substitution (step 3).  The two
        # categories are returned already partitioned — no caller-side
        # recomputation.
        if predicate_dropped:
            graph.diagnostics["predicate_placeholder_dropped_facts"] = (
                graph.diagnostics.get("predicate_placeholder_dropped_facts", []) + predicate_dropped
            )
            graph.diagnostics["predicate_placeholder_dropped"] = graph.diagnostics.get(
                "predicate_placeholder_dropped", 0
            ) + len(predicate_dropped)
        if residual_dropped:
            graph.diagnostics["residual_dropped_facts"] = residual_dropped
        dropped_facts = predicate_dropped + residual_dropped
        if dropped_facts:
            logger.warning(
                "Dropped %d fact(s) post-substitution (%d predicate-invariant, "
                "%d residual placeholder sweep — composite string with an "
                "unresolved placeholder; a bare missing-binding orphan on an "
                "add/modify is already dropped/reverted upstream by "
                "_apply_enrichment_delta before reaching here).",
                len(dropped_facts),
                len(predicate_dropped),
                len(residual_dropped),
            )
        deanon_dropped = deanon_input_count - len(deanon_facts)
        if deanon_dropped:
            logger.info(
                "De-anon: %d → %d facts (%d dropped)",
                deanon_input_count,
                len(deanon_facts),
                deanon_dropped,
            )
        if deanon_input_count and not deanon_facts:
            empty_cause = CAUSE_DEANON_SUBSTITUTION
        t.set_parsed(
            {
                "input_count": deanon_input_count,
                "output_count": len(deanon_facts),
                "dropped_count": deanon_dropped,
                "dropped_facts": dropped_facts,
            }
        )
    if chain_stopped():
        # Calibration short-circuit: deanon recorded.  graph.relations
        # remains the local-extract output; deanonymized facts list is
        # in phases[deanon].parsed.
        return dataclasses.replace(state, graph=graph, facts=deanon_facts, empty_cause=empty_cause)

    # Route scalar-valued objects (URLs, emails, phone numbers, DOIs,
    # version-tagged tool names like "ROS2") off the relation surface and
    # onto Entity.attributes of the subject.  Scalars are verbatim
    # identifiers that flow through to plausibility and downstream filters
    # without modification.  Routing them to attributes mirrors the
    # email/phone/linkedin path the local extractor already populates and
    # which ``relation_prep._flatten_entity_attributes`` mints into keyed
    # pairs for indexed-key distillation.  The projection itself is applied
    # by the ``rebuild`` stage, after its entity rebuild, so the subject
    # entity survives pruning.
    scalar_facts, deanon_facts = partition_scalar_facts(deanon_facts, graph.entities)
    if scalar_facts:
        graph.diagnostics["scalar_facts_projected"] = len(scalar_facts)
        if not deanon_facts and empty_cause is None:
            # Scalar-only session: every surviving fact moved to the
            # attribute surface. ``empty_cause is None`` is what makes
            # this unambiguous — the substitution's own emptying is
            # recorded above and runs BEFORE the partition (so it would
            # leave nothing to partition), and the deanon judge below is
            # skipped on an empty fact set. A cause already recorded
            # means the emptying was mixed, and the recovery net keeps
            # its normal behaviour.
            empty_cause = CAUSE_SCALAR_PARTITION

    if state.cloud_raw:
        graph.diagnostics["cloud_raw_response"] = state.cloud_raw
    if state.updated_anon_transcript:
        graph.diagnostics["cloud_updated_transcript"] = state.updated_anon_transcript

    # Plausibility on de-anonymized data (local judge, stage="deanon").
    # Runs when plausibility_judge != "off" AND plausibility_stage == "deanon"
    # AND model/tokenizer are available (guard against tests that pass None).
    # "auto" resolves to the local model.
    if (
        ctx.plausibility_stage == "deanon"
        and ctx.plausibility_judge != "off"
        and deanon_facts
        and ctx.model is not None
        and ctx.tokenizer is not None
    ):
        with phase_trace("deanon_plausibility") as t:
            _vram_snapshot(f"before_plausibility_deanon session={graph.session_id}")
            filtered_deanon, plaus_raw = judge_plausibility(
                deanon_facts,
                ctx.transcript,  # original real-name transcript — intentional, see docstring
                ctx.model,
                ctx.tokenizer,
                max_tokens=ctx.plausibility_max_tokens,
                temperature=_DEFAULT_FILTER_TEMPERATURE,
                seed=ctx.seed,
                prompts_dir=ctx.prompts_dir,
            )
            t.set_raw(plaus_raw)
            if filtered_deanon is not None:
                pre_deanon = len(deanon_facts)
                deanon_facts = filtered_deanon
                dropped_deanon = pre_deanon - len(deanon_facts)
                graph.diagnostics["plausibility"] = "deanon"
                # Own key: the three plausibility writers (anon judge,
                # this one, and the raw-fallback judge) used to share
                # ``plausibility_dropped`` with three different
                # semantics, making its final value order-dependent.
                graph.diagnostics["plausibility_dropped_deanon"] = dropped_deanon
                graph.diagnostics["plausibility_judge_actual"] = (
                    ctx.plausibility_judge if ctx.plausibility_judge != "auto" else "local"
                )
                if pre_deanon and not deanon_facts:
                    empty_cause = CAUSE_DEANON_JUDGE
                t.set_parsed(
                    {
                        "judge": (
                            ctx.plausibility_judge if ctx.plausibility_judge != "auto" else "local"
                        ),
                        "input_count": pre_deanon,
                        "kept_count": len(deanon_facts),
                        "dropped_count": dropped_deanon,
                    }
                )
                logger.info(
                    "Deanon-stage plausibility (local): %d → %d facts (%d dropped)",
                    pre_deanon,
                    len(deanon_facts),
                    dropped_deanon,
                )
            else:
                t.set_outcome("failed", reason="plausibility parse returned None")
                t.set_parsed(
                    {
                        "judge": (
                            ctx.plausibility_judge if ctx.plausibility_judge != "auto" else "local"
                        ),
                        "input_count": len(deanon_facts),
                        "kept_count": len(deanon_facts),
                        "dropped_count": 0,
                    }
                )
                logger.warning("Deanon-stage plausibility call failed — keeping deanon facts")

    return dataclasses.replace(
        state,
        graph=graph,
        facts=deanon_facts,
        scalar_facts=scalar_facts,
        empty_cause=empty_cause,
    )


def _stage_rebuild(ctx: StageContext, state: StageState) -> StageState:
    """``rebuild`` stage body — facts back to a ``SessionGraph``.

    Schema-validates the surviving fact dicts into ``Relation`` objects
    (recording every drop), consults the all-dropped recovery gate, and —
    when the gate does not fire — installs the relations together with
    their entity surface and the scalar-attribute projection. The pure
    half lives in :mod:`paramem.graph.relation_build`; what stays here is
    the recovery ACTION, which needs the model and tokenizer that module
    deliberately never sees.

    A scalar-only session reaches here with no relations and a
    ``routing`` cause: the gate declines, so ``apply_rebuild`` runs and
    the scalar projection lands on the entity surface. Suppressing the
    gate is what makes that possible — the recovery path returns before
    ``apply_rebuild``, and ``apply_rebuild`` is the only caller of the
    projection.
    """
    graph = state.graph
    kept_relations = build_relations(graph, state.facts, speaker_id=ctx.speaker_id)
    empty_cause = state.empty_cause
    if state.facts and not kept_relations:
        empty_cause = CAUSE_SCHEMA_VALIDATION

    # All-dropped safety net — if every relation was dropped and the
    # original extraction had facts, fall back to raw plausibility so the
    # session does not yield zero facts due to anonymizer inconsistency.
    if recovery_gate(graph, kept_relations, state.original_relation_count, empty_cause):
        return dataclasses.replace(
            state,
            graph=_fallback_plausibility_on_raw(
                graph,
                ctx.transcript,
                ctx.model,
                ctx.tokenizer,
                "all_dropped",
                speaker_name=ctx.speaker_name,
                speaker_id=ctx.speaker_id,
                max_tokens=ctx.max_tokens,
                plausibility_max_tokens=ctx.plausibility_max_tokens,
                seed=ctx.seed,
            ),
            empty_cause=empty_cause,
        )

    apply_rebuild(graph, kept_relations, state.scalar_facts, state.scope.resolution)

    added = len(kept_relations) - state.original_relation_count
    logger.info(
        "cloud enrichment: %d → %d relations (%+d)",
        state.original_relation_count,
        len(kept_relations),
        added,
    )
    return dataclasses.replace(state, graph=graph, empty_cause=empty_cause)


SESSION_EXTRACT: list[StageSpec] = [
    StageSpec(
        stage="local_extract",
        trace_names=("local_extract",),
        run=_stage_local_extract,
        requires=frozenset(),
        produces=frozenset({"graph"}),
        terminal_when=lambda s: not s.graph.relations,
    ),
    StageSpec(
        stage="second_order_extract",
        trace_names=("second_order_extract",),
        run=_stage_second_order_extract,
        requires=frozenset({"graph"}),
        produces=frozenset({"graph"}),
        applies_when=lambda s: _has_named_non_speaker_person(s.graph),
    ),
    StageSpec(
        stage="anonymize",
        trace_names=("anonymize",),
        run=_stage_anonymize,
        requires=frozenset({"graph"}),
        produces=frozenset({"graph", "payload", "original_relation_count"}),
        enabled_when=lambda c: bool(c.validate) and _session_egress_permitted(c),
        applies_when=lambda s: bool(s.graph.relations),
        # The fail-closed divert (anonymization parse failure) returns
        # ``payload=None`` — that's this stage's own terminal condition:
        # ``enrich`` must not run over a payload that was never produced.
        terminal_when=lambda s: s.payload is None,
    ),
    StageSpec(
        stage="enrich",
        # entity_correction / cloud_enrich / anon_plausibility — which of
        # the latter two fire depends on config (anon_plausibility only
        # when plausibility_stage == "anon").
        trace_names=("entity_correction", "cloud_enrich", "anon_plausibility"),
        run=_stage_enrich,
        requires=frozenset({"graph", "payload", "original_relation_count"}),
        produces=frozenset(
            {
                "graph",
                "facts",
                "scope",
                "cloud_raw",
                "updated_anon_transcript",
                "empty_cause",
            }
        ),
        enabled_when=lambda c: bool(c.validate) and _session_egress_permitted(c),
        applies_when=lambda s: s.payload is not None,
        # Every exit that does NOT reach the hand-over point leaves
        # ``facts`` empty: the unsupported-provider and missing-config
        # bails, and the "nothing survived the anon-stage judge" exit
        # (which clears the graph itself before returning). These used
        # to be a plain ``return graph`` from the middle of one long
        # function; here they stop the walk so the tail siblings never
        # run on a state that was never produced.
        terminal_when=lambda s: not s.facts,
    ),
    StageSpec(
        stage="deanonymize",
        # Two phases: the substitution itself, then the local judge at the
        # end of the same span (config-gated on ``plausibility_stage``).
        trace_names=("deanon", "deanon_plausibility"),
        run=_stage_deanonymize,
        requires=frozenset({"graph", "facts", "scope", "cloud_raw", "updated_anon_transcript"}),
        produces=frozenset({"graph", "facts", "scalar_facts", "empty_cause"}),
        # A round-trip scope exists only once ``enrich`` reached its
        # hand-over. Without one there was no cloud egress at all — the
        # anonymize/enrich stages were disabled by config, or skipped
        # because the graph had no relations — and there is nothing to
        # de-anonymize. ``enrich``'s own ``terminal_when`` covers the
        # other case (it RAN but produced no surviving facts);
        # ``applies_when`` cannot, because a SKIPPED stage never stops
        # the walk.
        applies_when=lambda s: s.scope is not None,
    ),
    StageSpec(
        stage="rebuild",
        # Pure post-processing plus the recovery action — no LLM phase of
        # its own. The fallback's plausibility call runs outside any
        # phase_trace scope, as it did before this runner existed.
        trace_names=(),
        run=_stage_rebuild,
        requires=frozenset(
            {"graph", "facts", "scalar_facts", "scope", "original_relation_count", "empty_cause"}
        ),
        produces=frozenset({"graph", "empty_cause"}),
        # Same gate as ``deanonymize``, and deliberately NOT ``bool(facts)``:
        # an empty fact set here is precisely the all-dropped case the
        # recovery gate inside this stage exists to catch.
        applies_when=lambda s: s.scope is not None,
    ),
]


def extract_graph(
    model,
    tokenizer,
    transcript: str,
    session_id: str,
    speaker_id: str,
    temperature: float = 0.0,
    max_tokens: int = _DEFAULT_FILTER_MAX_TOKENS,
    plausibility_max_tokens: int = _DEFAULT_FILTER_MAX_TOKENS,
    prompts_dir: str | Path | None = None,
    validate: bool = True,
    enrichment_provider: str = "",
    enrichment_provider_model: str = "claude-sonnet-4-6",
    enrichment_provider_endpoint: str | None = None,
    cloud_enabled: bool = False,
    speaker_name: str | None = None,
    plausibility_judge: str = "auto",
    plausibility_stage: str = "deanon",
    plausibility_model: str = "claude-sonnet-4-6",
    plausibility_endpoint: str | None = None,
    *,
    scrub: set[str] | frozenset[str],
    correction_entity_types: set[str] | frozenset[str] | None = None,
    system_prompt_filename: str = DEFAULT_SYSTEM_PROMPT_FILENAME,
    user_prompt_filename: str = DEFAULT_USER_PROMPT_FILENAME,
    model_alias: str | None = None,
    seed: int | None = None,
    timestamp: str | None = None,
    source_type: str = "transcript",
) -> SessionGraph:
    """Extract a knowledge graph from a session transcript.

    Multi-pass pipeline:
    1. Extract candidate triples from transcript
    2. Second-order extraction: ``local_extract`` is speaker-centric, so a
       clause naming a non-speaker person by relationship ("my brother
       Nadeem lives in Porto") tends to keep only the speaker->person edge
       and drop that person's OWN fact (measured on Mistral 7B). This pass
       re-extracts facts ABOUT each named non-speaker person the first
       pass surfaced and unions them in; predicate drift or redundant
       re-emits are left to the existing merger dedup and normalization.
    3. Anonymize -> enrich (cloud enrichment + anon-stage plausibility,
       configurable)
    4. De-anonymize (substitute real names back, scalar partition,
       deanon-stage plausibility)
    5. Rebuild (schema-validate relations, all-dropped recovery gate,
       entity surface + scalar-attribute projection)

    All filters fail gracefully — extraction result is preserved on any failure.

    A calibration caller may wrap this call in
    :func:`paramem.graph.phase_trace.stop_at` to return immediately after
    a named phase completes with a non-``"failed"`` outcome — saves
    compute when the operator only needs to inspect the early phases of
    the trace.  See :func:`~paramem.graph.phase_trace.stop_at`'s
    docstring for the mechanism; this function checks
    :func:`~paramem.graph.phase_trace.chain_stopped` after each phase
    block and returns early when it is set.  No scope open (the
    production default) means the full pipeline always runs.

    Args:
        temperature: Sampling temperature for extraction (default 0.0 for determinism).
        max_tokens: Max output tokens for extraction (default 2048).
        prompts_dir: Optional override for prompt config directory.
        validate: Run the anonymize/enrich pass (default True).
        enrichment_provider: Cloud provider for enrichment ("" = disabled).
        plausibility_judge: Plausibility filter judge ("auto"=local, "off"=disabled,
            or a provider name from
            :data:`~paramem.cloud.admission.PROVIDER_KEY_ENV` — e.g.
            "anthropic" — for cloud judging at anon stage).
        plausibility_stage: When to run plausibility ("deanon"=after de-anon,
            "anon"=on anonymized data with cloud judge).
        plausibility_model: Model id the cloud judge runs. Ignored when
            ``plausibility_judge`` is "auto" or "off".
        plausibility_endpoint: Endpoint override for a self-hosted
            OpenAI-compatible judge. ``None`` accepts the provider's
            default; ignored for native-SDK providers.
        scrub: PII-vocabulary hints (``SanitizationConfig.scrub``) forwarded
            to the anonymize stage's local anonymizer call — the prompt is
            the sole scope authority (see :func:`~paramem.cloud.anonymize.anonymize_transcript`).
            Required — no implicit default; an empty ``set``/``frozenset``
            is the operator opt-out.
        correction_entity_types: Scope-and-enable knob for the local
            entity-surface correction stage (see
            :func:`paramem.graph.entity_correction.correct_entity_surfaces`).
            Entity-type members (``place``/``organization``/``concept``)
            gate which surfaces are corrected; ``"attributes"`` additionally
            enables correcting ``graph.entities[*].attributes`` values.
            ``None`` or empty disables the stage entirely — there is no
            implicit default scope; production always threads the
            configured value.
        speaker_id: Speaker store ID (e.g. ``"speaker0"``). Stamped onto every
            ``Relation`` produced by this extraction pass as provenance.
            Required — callers must always supply the session's speaker ID.
        system_prompt_filename: Filename of the system prompt within the prompts
            directory.  Defaults to :data:`DEFAULT_SYSTEM_PROMPT_FILENAME`
            (``"extraction_system.txt"``).  Used for every source type;
            document chunks land in the ``{transcript}`` slot of the
            same prompt.
        user_prompt_filename: Filename of the user-turn prompt template.
            Defaults to :data:`DEFAULT_USER_PROMPT_FILENAME`
            (``"extraction.txt"``).  Same prompt for every source type.
        model_alias: Model alias (e.g. ``"qwen3-4b"``).  Enables per-file
            prompt resolution — see :func:`load_extraction_prompts`.  The
            ``cloud_*`` prompts (cloud enricher) and ``anonymization.txt``
            are model-independent by design and are NOT affected by this
            parameter.
        seed: Optional RNG seed forwarded to every :func:`generate_answer`
            call within the pipeline (extraction, anonymization,
            plausibility).  At the default ``temperature=0.0`` (greedy
            decoding) this is a strict no-op.  Default ``None`` preserves
            production behaviour unchanged.
        timestamp: Session-start assertion time (ISO 8601), typically the
            session's ``started_at``.  Stamped onto the returned
            ``SessionGraph.timestamp`` so ``last_seen`` on newly-merged
            edges reflects when the facts were asserted, not when
            extraction ran.  ``None`` (default) falls back to ``now()``.
        source_type: ``"transcript"`` (default) or ``"document"``. Forwarded
            to :func:`_parse_extraction` / :func:`_stamp_speaker_entity` as
            the Guard B gate for the document-only exact-full-name rewrite
            of third-person speaker mentions onto ``speaker_id``.
    """
    # Open the extraction trace.  phase_trace() calls reachable from
    # any helper in this scope append to the same trace via contextvar;
    # the trace survives every `graph = ...` rebinding inside.  At the
    # end (any return path), trace.attach_to(graph) materialises the
    # records on the final graph's diagnostics.  A calibration caller may
    # additionally wrap this whole call in `with stop_at(phase):` (see
    # that function's docstring) to request an early return.
    #
    # The control flow (local_extract -> second_order_extract -> anonymize
    # -> enrich -> deanonymize -> rebuild) is expressed declaratively as
    # SESSION_EXTRACT and walked by paramem.graph.flow.run_flow — see that
    # module and the _stage_* functions above (and in
    # paramem.graph.stage_anonymize / paramem.graph.stage_enrich) for the
    # per-phase bodies. This function's job is: build the run-constant StageContext
    # once, seed the initial StageState, walk the flow, and keep the
    # extraction_trace lifecycle (open/attach) exactly as before.
    with extraction_trace() as trace:
        # A calibration caller entering the chain past ``local_extract``
        # supplies the graph that stage would have produced (see
        # paramem.graph.phase_trace.start_at); with no such request open —
        # every production call — the seed is the empty graph the local
        # extractor rebinds wholesale.
        seed = chain_seed()
        state = StageState(
            graph=SessionGraph(
                session_id=session_id,
                timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
            )
            if seed is None
            else seed
        )
        try:
            ctx = StageContext(
                model=model,
                tokenizer=tokenizer,
                transcript=transcript,
                session_id=session_id,
                speaker_id=speaker_id,
                speaker_name=speaker_name,
                temperature=temperature,
                max_tokens=max_tokens,
                plausibility_max_tokens=plausibility_max_tokens,
                prompts_dir=prompts_dir,
                system_prompt_filename=system_prompt_filename,
                user_prompt_filename=user_prompt_filename,
                model_alias=model_alias,
                seed=seed,
                timestamp=timestamp,
                source_type=source_type,
                validate=validate,
                cloud_enabled=cloud_enabled,
                enrichment_provider=enrichment_provider,
                enrichment_provider_model=enrichment_provider_model,
                enrichment_provider_endpoint=enrichment_provider_endpoint,
                plausibility_judge=plausibility_judge,
                plausibility_stage=plausibility_stage,
                plausibility_model=plausibility_model,
                plausibility_endpoint=plausibility_endpoint,
                scrub=scrub,
                correction_entity_types=correction_entity_types,
            )
            state = run_flow(SESSION_EXTRACT, ctx, state)
            return state.graph
        finally:
            # Materialise the trace on whatever graph we're about to
            # return — covers every return path including early returns
            # on parse failure and empty-relations short-circuit.
            trace.attach_to(state.graph)


def anonymize_turn(
    transcript: str,
    model,
    tokenizer,
    *,
    speaker_id: str | None = None,
    speaker_name: str | None = None,
    prompts_dir: str | Path | None = None,
    scrub: set[str] | frozenset[str],
) -> AnonymizedContract:
    """Local extract + local anonymize for cloud egress.

    Composition over existing primitives — same anonymization chain the
    session flow's ``anonymize`` stage (``paramem.graph.stage_anonymize``)
    runs every consolidation cycle, minus the ``enrich`` stage's cloud
    enrichment call:

    0. **Turn-marking (model-facing only).** ``transcript`` — a bare,
       unmarked chat sentence at this call site — is rendered through
       :meth:`~paramem.server.session_buffer.SessionBuffer._format_turns`
       (single source of truth for the ``[user] <text>`` /
       ``[assistant] <text>`` marker surface every extraction/
       anonymization few-shot is calibrated on) into
       ``model_facing_transcript``.  Only the two LLM calls below see the
       turn-marked copy — the marker exists solely to keep the model
       in-distribution while it authors the anonymized transcript. A bare
       sentence was observed to glue a possessive into a single
       anonymization token (``"Pat's dog"`` minted as one placeholder
       instead of splitting ``"Pat"`` + ``"dog"``) because it is
       off-distribution from every few-shot example.
    1. ``extract_graph(validate=False)`` — local extraction only,
       produces a SessionGraph whose relations anchor the anonymizer
       (rendered to facts via :func:`~paramem.graph.schema.facts_from_relations`
       — see step 2).
    2. :func:`~paramem.cloud.anonymize.anonymize` — THE one anonymize
       chain, shared with every other cloud-egress path (identical to
       what the session flow's ``anonymize`` stage
       (:func:`~paramem.graph.stage_anonymize._stage_anonymize`) and
       :func:`~paramem.training.graph_enrich.enrich_graph` call).
       ``identity_domain`` is not passed — this path has no closed
       node-key domain to reconcile against (free-text chat, not a fold
       graph).

    This helper anonymizes a TRANSCRIPT for cloud egress — it never
    builds or returns facts (``graph.relations`` is discarded after
    anchoring the anonymizer — see step 1 — so a caller deriving the
    anonymized fact array from this contract via
    :func:`~paramem.cloud.placeholders.insert_placeholders` would always
    get ``[]``; nothing does, since this path's only output is the
    anonymized transcript).

    ``scrub`` is the operator's PII-vocabulary hint list
    (``SanitizationConfig.scrub``), rendered verbatim into the
    anonymizer prompt's ``{scrub_categories}`` slot — the prompt is the
    SOLE scope authority; there is no code-side entity-type gate.
    Required — an omitted ``scrub`` would silently anonymize against a
    hidden default on a security-critical egress path.  An EMPTY
    ``scrub`` is the meaningful operator opt-out: it short-circuits
    before any LLM call (before even the local extraction pass — no
    compute is wasted anchoring an anonymizer that will never run), and
    the helper returns ``status="opted_out"`` with ``anon_transcript``
    sourced from the passed-in ``transcript`` verbatim, never a model
    artifact.

    ``speaker_id`` is the resolved speaker store ID, threaded to
    :func:`extract_graph` (which requires it) and stamped on the
    ephemeral graph's relations as provenance.  That graph exists only
    to anchor anonymization and is discarded immediately, so the value
    is never persisted.  Text-only ``/chat`` requests with no enrolled
    speaker pass ``None``; the helper falls back to the
    ``"cloud_egress"`` sentinel rather than failing extraction.

    ``AnonymizedContract.status``:

    * ``"ok"`` — anonymization ran; ``anon_transcript`` is the MODEL's
      own rewrite (the single synthetic turn marker from step 0 stripped
      back off, restoring the bare-text contract this helper's callers
      expect).  ``forward``/``reverse`` may still be empty — a
      legitimate "ran, found nothing in scope" verdict, not a failure;
      egress PROCEEDS.
    * ``"opted_out"`` — operator opted out (``scrub`` empty).
    * ``"failed"`` — block.  Every other early exit lands here:
      empty/whitespace-only input, local extraction raising, zero
      relations extracted (``not graph.relations``), the anonymizer
      raising, an anonymizer parse failure, or the model's rewritten
      transcript coming back empty after the marker strip.  Callers must
      NEVER fall back to the original real-name transcript on this
      status.

    The companion :func:`~paramem.cloud.deanonymize.deanonymize_text`
    is the caller's exit gate for the cloud's response text.
    """
    # ``failure`` left unset (``None``) on this sentinel: it belongs to
    # ``anonymize``'s own "parse"/"guard" vocabulary (see
    # :class:`~paramem.cloud.anonymize.AnonymizedContract`), and none of
    # this helper's own early exits below (extraction exception, zero
    # relations, anonymizer exception) are that call's parse/guard
    # distinction — they're this helper's own failure modes, and no
    # caller here branches on cause.
    _failed = AnonymizedContract(
        status="failed",
        forward={},
        reverse={},
        anon_transcript="",
        declared=frozenset(),
        norm_stats={"inverted": 0, "dropped": 0},
        rekey_dropped=0,
        raw="",
        failure=None,
    )
    if not transcript or not transcript.strip():
        return _failed

    # Empty scrub = operator opt-out.  Skip the entire LLM-driven
    # anonymization path (and the local extraction pass that would only
    # exist to anchor it) — the caller forwards the transcript verbatim.
    # Distinguished from the "failed" block status by non-empty text.
    # Shares the ONE opt-out constructor with ``anonymize``'s own
    # ``scrub``-empty branch (see :func:`~paramem.cloud.anonymize.
    # opted_out_contract`'s docstring for why this call site cannot just
    # call ``anonymize`` itself and let IT branch).
    if not scrub:
        return opted_out_contract(transcript)

    # The local_extract and anonymization few-shots (configs/prompts/
    # extraction.txt, configs/prompts/anonymization.txt) are calibrated
    # exclusively on the ``[user] <text>`` / ``[assistant] <text>``
    # turn-marked surface produced by ``SessionBuffer._format_turns`` (the
    # single source of truth for that rendering — see its docstring,
    # which already names this call site as the intended second caller).
    # A bare, unmarked sentence puts the model off-distribution from
    # every example it was tuned on and has been observed to glue a
    # possessive into a single anonymization token (e.g. "Pat's dog" ->
    # one placeholder instead of "Pat" + "dog" separately).
    turn_marked_lines, _ = SessionBuffer._format_turns([{"role": "user", "text": transcript}])
    model_facing_transcript = "\n".join(turn_marked_lines)
    # The single synthetic turn always renders as ``"<marker> " + transcript``
    # (see ``_format_turns``) — derive the marker prefix mechanically rather
    # than hardcoding the literal, so a future marker-format change (single
    # source of truth: ``_format_turns``) cannot silently desync this strip.
    _turn_marker_prefix = model_facing_transcript[: len(model_facing_transcript) - len(transcript)]

    # Both LLM calls below (extraction + anonymization) are structured
    # extraction and must run on the base weights, never the training-active
    # adapter.  One shared scope disables the adapter and keeps the KV cache
    # live for both generates, restoring the model's entry state on exit.
    with base_model_inference(model):
        try:
            graph = extract_graph(
                model,
                tokenizer,
                model_facing_transcript,
                session_id="cloud_egress",
                # Ephemeral graph: extracted only to anchor anonymization, then
                # discarded — the stamped provenance is never persisted.  Use the
                # resolved speaker_id when the caller has one (text-only /chat
                # requests may not), falling back to the "cloud_egress" sentinel
                # that already names the session.
                speaker_id=speaker_id or "cloud_egress",
                speaker_name=speaker_name,
                # No model_alias: cloud-egress extraction is deliberately
                # model-independent.  This graph exists only to anchor PII
                # anonymization (entity spans), not to build the knowledge graph,
                # and entity/attribute extraction is reliable across models — the
                # per-model prompt overrides target relation decomposition, which
                # is not load-bearing here.  So the shared base prompt is used
                # regardless of the configured model.
                prompts_dir=prompts_dir,
                validate=False,
                enrichment_provider="",
                scrub=scrub,
            )
        except Exception:
            logger.exception("Cloud egress: local extraction failed; treating as block")
            return _failed

        if not graph.relations:
            return _failed

        try:
            anon_prompt = _load_prompt("anonymization.txt", prompts_dir=prompts_dir, required=True)
            anon_system = _load_prompt("anonymization_system.txt", required=True)
            payload = anonymize(
                facts_from_relations(graph.relations),
                model,
                tokenizer,
                transcript=model_facing_transcript,
                scrub=scrub,
                speaker_name=speaker_name,
                max_tokens=_DEFAULT_ANONYMIZER_MAX_TOKENS,
                user_prompt_template=anon_prompt,
                system_prompt=anon_system,
            )
        except Exception:
            logger.exception("Cloud egress: anonymization raised; treating as block")
            return _failed

    if payload.status == "failed":
        return payload

    # The returned transcript is the MODEL's own rewrite — never
    # mechanically rebuilt from the mapping.  Strip the single synthetic
    # turn marker (step 0) back off so the bare-text contract this
    # helper's callers rely on is preserved; best-effort (the model is
    # instructed to preserve everything outside the substitutions
    # verbatim, including the marker, but a strip that finds nothing to
    # strip is a no-op).
    anon_transcript = payload.anon_transcript
    if _turn_marker_prefix and anon_transcript.startswith(_turn_marker_prefix):
        anon_transcript = anon_transcript[len(_turn_marker_prefix) :]

    if not anon_transcript:
        return _failed

    return dataclasses.replace(payload, anon_transcript=anon_transcript)
