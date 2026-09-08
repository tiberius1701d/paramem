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

Each stage body below composes the graph-extraction primitives in
``paramem.graph.extractor`` directly — ``local_extract`` and
``second_order_extract`` call the local-extraction primitive;
``deanonymize`` and ``rebuild`` call the deanonymization/rebuild
primitives — same primitives, same arguments, same ``phase_trace``
scopes those primitives declare on their own. ``run_flow`` does not open
phases itself; every ``phase_trace`` call below belongs to the stage
body that makes it.

``anonymize_turn`` lives here rather than in ``extractor.py`` for the
same reason as the rest of this module: per
``paramem/cloud/admission.py``'s placement principle ("a primitive every
tier needs, owned by none of them" — why that module is a stdlib-only
leaf), :func:`~paramem.cloud.admission.evaluate_cloud_egress` is the
primitive and :func:`~paramem.cloud.anonymize.anonymize` is the shared
component every cloud-egress path composes through — so
``anonymize_turn``, the conversation-egress composition of that shared
component, belongs at the flow layer rather than with the primitives.
Its callers are ``paramem/server/egress.py`` and
``scripts/dev/anonymizer_gate.py``.
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from paramem.cloud.admission import evaluate_cloud_egress
from paramem.cloud.anonymize import (
    _DEFAULT_ANONYMIZER_TOKEN_ENVELOPE,
    AnonymizedContract,
    anonymize,
)
from paramem.cloud.deanonymize import deanonymize_facts
from paramem.config.taxonomy import ScrubCategory
from paramem.graph.anonymizer_prompts import load_anonymizer_prompts
from paramem.graph.empty_cause import (
    CAUSE_DEANON_JUDGE,
    CAUSE_DEANON_SUBSTITUTION,
    CAUSE_SCHEMA_VALIDATION,
)
from paramem.graph.extractor import (
    _DEFAULT_FILTER_MAX_TOKENS,
    _DEFAULT_FILTER_TEMPERATURE,
    DEFAULT_SYSTEM_PROMPT_FILENAME,
    DEFAULT_USER_PROMPT_FILENAME,
    PLAUSIBILITY_FAILED,
    PLAUSIBILITY_OFF,
    PLAUSIBILITY_SKIPPED,
    _fallback_plausibility_on_raw,
    _record_binding_diagnostics,
    _run_local_extraction,
    _vram_snapshot,
    judge_plausibility,
    record_plausibility_state,
    record_plausibility_verdict,
)
from paramem.graph.flow import StageContext, StageSpec, StageState, run_flow
from paramem.graph.phase_trace import chain_seed, chain_stopped, extraction_trace, phase_trace
from paramem.graph.relation_build import (
    apply_rebuild,
    build_relations,
    recovery_gate,
)
from paramem.graph.schema import SessionGraph
from paramem.graph.stage_anonymize import _stage_anonymize
from paramem.graph.stage_enrich import _stage_enrich
from paramem.models.loader import base_model_inference
from paramem.utils.identity import canonical, is_speaker_id
from paramem.utils.turn_markers import format_turn

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


def _named_non_speaker_people(graph: SessionGraph) -> list[str]:
    """Named (proper-name) person entities in *graph* other than a speaker id.

    Single source of truth for the ``second_order_extract`` target set:
    :func:`_has_named_non_speaker_person` (the flow gate) asks the
    membership question, and :func:`_stage_second_order_extract` threads
    this SAME list into the prompt's ``{named_people}`` slot and enforces
    it post-parse. Deriving the set once here — instead of computing it
    for the gate and then asking the LLM to re-derive it from raw prose
    (which includes the addressee by construction) — is what closes the
    second-order phase's double-derivation defect.
    """
    return [
        e.name for e in graph.entities if e.entity_type == "person" and not is_speaker_id(e.name)
    ]


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
    return bool(_named_non_speaker_people(graph))


def _stage_second_order_extract(ctx: StageContext, state: StageState) -> StageState:
    """``second_order_extract`` stage body — gated by
    :func:`_has_named_non_speaker_person` (the flow's ``applies_when``).

    Extracts facts ABOUT the named entities ``local_extract`` (first-order)
    surfaced, recovering a named relative's own attribute (location, job,
    trait) when local_extract collapsed a single-relative clause ("my
    brother Nadeem lives in Porto") into ONE relation instead of two
    (measured Mistral 7B failure mode).

    The closed target set — :func:`_named_non_speaker_people`, the exact
    entities the gate matched — is threaded into
    ``extraction_second_order.txt``'s ``{named_people}`` slot (via
    ``extra_slots``) rather than asking the LLM to re-derive it from raw
    prose, which includes the addressee by construction. Threading the set
    narrows what the model reaches for but does not guarantee it, so every
    relation the second-order pass emits is enforced against it
    (:func:`_enforce_second_order_targets`, run as the
    :func:`~paramem.graph.extractor._run_local_extraction` ``postprocess``
    hook — INSIDE that primitive's ``phase_trace`` scope, so the
    ``second_order_extract`` trace record's ``parsed`` summary reflects
    the post-enforcement graph, not the raw parse):

    * Subject already in the closed set (STRICT
      :func:`~paramem.utils.identity.canonical` equality — no fuzzy
      matching) — kept unmodified. A genuine namesake is unaffected: it
      arrives as its own pass-1 entity and is therefore already in the
      set.
    * Subject canonically equal to ``ctx.speaker_name`` (the display name)
      but NOT in the closed set — REMAPPED to ``ctx.speaker_id`` and kept,
      per ``configs/prompts/speaker_directive.txt``'s own declared
      semantics (its ``EXTRACTION-DIRECTIVE`` section: "when the assistant
      names its addressee ... that name refers to this same speaker").
      ``ctx.speaker_name`` is the run-constant field
      :func:`extract_graph` seeds onto :class:`~paramem.graph.flow.
      StageContext` from its own ``speaker_name`` parameter
      (``paramem.graph.flow.StageContext.speaker_name``); ``None`` (no
      display name resolved, e.g. an anonymous speaker) disables this
      branch entirely — every relation then goes through the plain
      in-set/drop split below.
    * Anything else — dropped, the deterministic complement.

    Both counts are recorded on ``graph.diagnostics`` (this module's
    existing drop-site naming style — see ``predicate_placeholder_dropped``
    in :func:`_stage_deanonymize`): ``second_order_subject_remapped`` and
    ``second_order_subject_dropped``.

    The entity surface follows the same enforcement: an entity from the
    second-order pass survives only if its canonical name is in the closed
    set OR it is an endpoint (subject or object) of a KEPT, POST-REMAP
    relation — so the remapped subject's own former display-name entity
    does not survive via its own relation (that relation's subject is now
    ``ctx.speaker_id``, not the display name) or get minted as a fresh
    node downstream (every session entity mints/updates a graph node —
    ``GraphMerger.merge``, ``paramem/graph/merger.py:384-388``).
    """
    graph = state.graph
    named_people = _named_non_speaker_people(graph)
    allowed = {canonical(name) for name in named_people}
    speaker_name_canonical = canonical(ctx.speaker_name) if ctx.speaker_name else None
    enforcement_counts = {"remapped": 0, "dropped": 0}

    def _enforce_second_order_targets(second_order_graph: SessionGraph) -> SessionGraph:
        """Restrict ``second_order_graph`` to the closed named-people
        target set, remapping the addressee's display name onto
        ``ctx.speaker_id`` rather than dropping it (see the enclosing
        function's docstring). Mutates and returns ``second_order_graph``
        in place; run as ``_run_local_extraction``'s ``postprocess`` hook
        so the phase trace records the enforced result.
        """
        kept_relations = []
        for rel in second_order_graph.relations:
            subject_canonical = canonical(rel.subject)
            if subject_canonical in allowed:
                kept_relations.append(rel)
            elif speaker_name_canonical is not None and subject_canonical == speaker_name_canonical:
                rel.subject = ctx.speaker_id
                kept_relations.append(rel)
                enforcement_counts["remapped"] += 1
            else:
                enforcement_counts["dropped"] += 1
        second_order_graph.relations = kept_relations
        kept_endpoints = {canonical(r.subject) for r in kept_relations} | {
            canonical(r.object) for r in kept_relations
        }
        second_order_graph.entities = [
            ent
            for ent in second_order_graph.entities
            if canonical(ent.name) in allowed or canonical(ent.name) in kept_endpoints
        ]
        return second_order_graph

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
        extra_slots={"named_people": ", ".join(named_people)},
        postprocess=_enforce_second_order_targets,
    )
    remapped = enforcement_counts["remapped"]
    dropped = enforcement_counts["dropped"]
    if remapped or dropped:
        logger.debug(
            "second_order_extract: remapped %d relation(s) with subject == the "
            "speaker's display name onto %s; dropped %d relation(s) with subject "
            "outside the closed named-people set",
            remapped,
            ctx.speaker_id,
            dropped,
        )
    if remapped:
        graph.diagnostics["second_order_subject_remapped"] = (
            graph.diagnostics.get("second_order_subject_remapped", 0) + remapped
        )
    if dropped:
        graph.diagnostics["second_order_subject_dropped"] = (
            graph.diagnostics.get("second_order_subject_dropped", 0) + dropped
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

    Two things, in this order, because the order is load-bearing:

    1. The ``deanon`` phase: pure dict substitution restoring real names
       from placeholders (no LLM call, so ``raw_output`` stays ``None``).
       Facts still carrying an unresolved placeholder, and facts with a
       placeholder glued into the predicate, are dropped and recorded.
    2. The deanon-stage plausibility judge (local model, real names),
       which receives the ORIGINAL real-name transcript — it runs
       locally on de-anonymized facts, so there is no reason to hand it
       the anonymized text. Every reachable path through the gate records
       its state onto ``graph.diagnostics["plausibility_state_deanon"]``
       (the ``paramem.graph.extractor`` PLAUSIBILITY_* vocabulary), and
       ``predicate_placeholder_dropped``/``residual_dropped`` (step 1's
       two loss counters) are written unconditionally.

    The mid-stage ``chain_stopped()`` check after the ``deanon`` phase
    exists so a calibration caller stopping at ``deanon`` does not get
    the judge.
    """
    graph = state.graph
    scope = state.scope
    facts = state.facts
    empty_cause = state.empty_cause

    # THE ONE call to ``deanonymize_facts`` for this response —
    # ``paramem.graph.extractor._apply_enrichment_delta`` decides
    # per-triple resolvability before this stage ever runs, so there is no
    # earlier ``deanonymize_facts`` call to duplicate this one.  This
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
        # Loss counters are written unconditionally (0 when nothing was
        # lost) so absence means exactly one thing — the site was not
        # reached; the fact-dict payload lists beside them stay
        # conditional (bulky). ``predicate_placeholder_dropped``'s
        # accumulate-onto-prior-value pattern is preserved verbatim.
        graph.diagnostics["predicate_placeholder_dropped"] = graph.diagnostics.get(
            "predicate_placeholder_dropped", 0
        ) + len(predicate_dropped)
        if predicate_dropped:
            graph.diagnostics["predicate_placeholder_dropped_facts"] = (
                graph.diagnostics.get("predicate_placeholder_dropped_facts", []) + predicate_dropped
            )
        graph.diagnostics["residual_dropped"] = len(residual_dropped)
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

    if state.cloud_raw:
        graph.diagnostics["cloud_raw_response"] = state.cloud_raw
    if state.updated_anon_transcript:
        graph.diagnostics["cloud_updated_transcript"] = state.updated_anon_transcript

    # Plausibility on de-anonymized data (local judge, stage="deanon").
    # Runs when plausibility_judge != "off" AND plausibility_stage == "deanon"
    # AND model/tokenizer are available (guard against tests that pass None).
    # "auto" resolves to the local model. Every branch records its judge
    # state (paramem.graph.extractor's PLAUSIBILITY_* vocabulary) so the
    # absence of plausibility_dropped_deanon never has to be interpreted.
    judge_label = ctx.plausibility_judge if ctx.plausibility_judge != "auto" else "local"
    if ctx.plausibility_judge == "off":
        record_plausibility_state(graph, "deanon", PLAUSIBILITY_OFF)
    elif ctx.plausibility_stage != "deanon":
        record_plausibility_state(graph, "deanon", PLAUSIBILITY_SKIPPED, reason="stage_not_deanon")
    elif not deanon_facts:
        record_plausibility_state(graph, "deanon", PLAUSIBILITY_SKIPPED, reason="no_facts")
    elif ctx.model is None or ctx.tokenizer is None:
        record_plausibility_state(graph, "deanon", PLAUSIBILITY_SKIPPED, reason="no_model")
    else:
        with phase_trace("deanon_plausibility") as t:
            _vram_snapshot(f"before_plausibility_deanon session={graph.session_id}")
            verdict, plaus_raw = judge_plausibility(
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
            if verdict is not None:
                pre_deanon = len(deanon_facts)
                deanon_facts = verdict.kept
                dropped_deanon = len(verdict.dropped)
                graph.diagnostics["plausibility"] = "deanon"
                record_plausibility_verdict(graph, "deanon", verdict, judge=judge_label)
                if pre_deanon and not deanon_facts:
                    empty_cause = CAUSE_DEANON_JUDGE
                t.set_parsed(
                    {
                        "judge": judge_label,
                        "input_count": pre_deanon,
                        "kept_count": len(deanon_facts),
                        "dropped_count": dropped_deanon,
                        "dropped_facts": verdict.dropped,
                        "out_of_range": verdict.out_of_range,
                        "unattributed": verdict.unattributed,
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
                record_plausibility_state(
                    graph, "deanon", PLAUSIBILITY_FAILED, reason="parse_failed"
                )
                t.set_parsed(
                    {
                        "judge": judge_label,
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
        empty_cause=empty_cause,
    )


def _stage_rebuild(ctx: StageContext, state: StageState) -> StageState:
    """``rebuild`` stage body — facts back to a ``SessionGraph``.

    Schema-validates the surviving fact dicts into ``Relation`` objects
    (recording every drop), consults the all-dropped recovery gate, and —
    when the gate does not fire — installs the relations together with
    their entity surface. The pure half lives in
    :mod:`paramem.graph.relation_build`; what stays here is the recovery
    ACTION, which needs the model and tokenizer that module deliberately
    never sees.
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
                speaker_id=ctx.speaker_id,
                max_tokens=ctx.max_tokens,
                plausibility_max_tokens=ctx.plausibility_max_tokens,
                seed=ctx.seed,
            ),
            empty_cause=empty_cause,
        )

    apply_rebuild(graph, kept_relations, state.scope.resolution)

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
        # (which clears the graph itself before returning). Each one
        # stops the walk here so the tail siblings never run on a state
        # that was never produced.
        terminal_when=lambda s: not s.facts,
    ),
    StageSpec(
        stage="deanonymize",
        # Two phases: the substitution itself, then the local judge at the
        # end of the same span (config-gated on ``plausibility_stage``).
        trace_names=("deanon", "deanon_plausibility"),
        run=_stage_deanonymize,
        requires=frozenset({"graph", "facts", "scope", "cloud_raw", "updated_anon_transcript"}),
        produces=frozenset({"graph", "facts", "empty_cause"}),
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
        # phase_trace scope.
        trace_names=(),
        run=_stage_rebuild,
        requires=frozenset({"graph", "facts", "scope", "original_relation_count", "empty_cause"}),
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
    scrub_categories: Sequence[ScrubCategory],
    correction_entity_types: set[str] | frozenset[str] | None = None,
    system_prompt_filename: str = DEFAULT_SYSTEM_PROMPT_FILENAME,
    user_prompt_filename: str = DEFAULT_USER_PROMPT_FILENAME,
    model_alias: str | None = None,
    seed: int | None = None,
    timestamp: str | None = None,
    source_type: str = "transcript",
    anonymize_token_envelope: int = _DEFAULT_ANONYMIZER_TOKEN_ENVELOPE,
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
       entity surface filtering)

    The entity scalar-attribute projection (an entity's ``attributes``
    dict rendered into attribute-typed relations) does NOT run inside this
    flow — it runs once, after this whole chain returns, in
    :meth:`~paramem.graph.extraction_pipeline.ExtractionPipeline._run_extractor`,
    outside the ``base_model_inference`` scope. That placement is what lets
    a calibration ``dispatch_chain`` call
    (:func:`~paramem.server.calibrate.dispatch_chain`) see the same
    projected relations the fold merges, including under ``stop_at``.

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
        scrub_categories: Resolved scrub categories
            (``SanitizationConfig.scrub_categories``) forwarded to the
            anonymize stage's local anonymizer call, whose activated
            categories are the sole scope authority (see
            :func:`~paramem.cloud.anonymize.anonymize`). Required — no
            implicit default; an empty tuple is the operator opt-out.
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
            of third-person speaker mentions onto ``speaker_id``, and to
            :func:`build_document_context` to select the
            ``{document_context}`` rendering (non-empty only for
            ``"document"`` with a known speaker; empty otherwise).
        anonymize_token_envelope: Total (prompt + output) token budget the
            ``anonymize`` stage's local ``generate()`` call(s) may occupy —
            forwarded to :func:`~paramem.cloud.anonymize.anonymize` as
            ``token_envelope``. Defaults to
            :data:`~paramem.cloud.anonymize._DEFAULT_ANONYMIZER_TOKEN_ENVELOPE`
            (session tier / calibration); production consolidation threads
            ``consolidation.extraction_anonymize_token_envelope`` through
            :meth:`~paramem.graph.extraction_pipeline.ExtractionPipeline.kwargs`.
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
    # extraction_trace lifecycle (open/attach) around the whole walk.
    with extraction_trace() as trace:
        # A calibration caller entering the chain past ``local_extract``
        # supplies, via :func:`~paramem.graph.phase_trace.start_at`, the graph
        # that stage would have produced; it seeds the initial ``StageState``.
        # With no such request open — every production call — ``chain_seed()``
        # is ``None`` and the chain starts from a fresh empty graph.  This is
        # the injected GRAPH; keep it DISTINCT from the sampling ``seed`` (an
        # int forwarded verbatim to every ``generate_answer`` via ``ctx.seed``)
        # — sharing one name would feed a ``SessionGraph`` into ``int(seed)``
        # the moment a caller injected a graph.
        injected_graph = chain_seed()
        state = StageState(
            graph=SessionGraph(
                session_id=session_id,
                timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
            )
            if injected_graph is None
            else injected_graph
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
                scrub_categories=scrub_categories,
                correction_entity_types=correction_entity_types,
                anonymize_token_envelope=anonymize_token_envelope,
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
    history: Sequence[dict] = (),
    speaker_id: str | None = None,
    speaker_name: str | None = None,
    prompts_dir: str | Path | None = None,
    categories: Sequence[ScrubCategory],
    token_envelope: int,
) -> AnonymizedContract:
    """Local anonymize for cloud egress - a thin composition over the one
    shared anonymize chain (:func:`~paramem.cloud.anonymize.anonymize`),
    the same chain the session flow's ``anonymize`` stage
    (:func:`~paramem.graph.stage_anonymize._stage_anonymize`) and
    :func:`~paramem.training.graph_enrich.enrich_graph` call.

    ``transcript`` (the current turn's bare text) and every turn of
    ``history`` are rendered through
    :func:`~paramem.utils.turn_markers.format_turn` into the
    ``[<role>] <text>`` surface — structurally required because
    :func:`~paramem.cloud.anonymize.assemble_payload` splits it into the
    transcript's anchor range and the marker-bearing anchor evidence the
    ANCHOR call is shown — then handed to
    :func:`~paramem.cloud.anonymize.anonymize` as its ``transcript`` and
    ``history`` arguments respectively - the
    same SCAN call that names values in the current turn also names them
    in the drop-gated history, and both surfaces build ONE forward table,
    so a value named only in an earlier turn is placeholdered exactly like
    one named in the current turn. ``history`` is already the drop-gated,
    ``{role, text}``-shaped turn list the caller resolved (``()`` for a
    text-only ``/chat`` request with no prior turns).

    This path passes no facts - ``facts=[]`` - the anonymize chain's
    documented shape for "transcript but no facts" (chat egress). No
    local extraction runs here: the SCAN call's kept person values anchor
    the anonymizer's self-introduction question directly against the
    payload.

    ``model`` / ``tokenizer`` may be ``None`` - a cloud-only deferral
    (base model not resident), never a configuration. ``None`` fails the
    whole call closed (``status="failed"``, ``failure="model_unavailable"``)
    - the SCAN call needs a resident model and cannot be deferred the way
    the ANCHOR call can. The adapter-off scope below is entered only when
    a model is present, since it is a property of the resident PEFT model.

    ``categories`` is the resolved scrub-category tuple
    (``SanitizationConfig.scrub_categories``) - the operator's ``scrub``
    selection decides which SCAN keyword's values are kept; there is no
    model-side judgement of scope. Required - an omitted value would
    silently anonymize against a hidden default on a security-critical
    egress path. An empty tuple is the operator opt-out, handled entirely
    inside :func:`~paramem.cloud.anonymize.anonymize`'s own
    ``categories``-empty door - this helper has no door of its own to
    duplicate it.

    ``token_envelope`` is the total (prompt + output) token budget the
    ``anonymize`` call below may occupy - forwarded verbatim as its own
    ``token_envelope`` argument. Required keyword-only, no module
    default: this is a VRAM-safety-critical parameter by the same
    standard as ``categories`` above, and an unbudgeted anonymize call is
    a defect to trace, not a value to fall back on silently. Production's
    only caller, :meth:`~paramem.server.egress.OutboundText.contract`
    (read by :func:`~paramem.server.egress.answer_via_cloud`; the HA door,
    :func:`~paramem.server.egress.answer_via_ha`, sends the turn verbatim
    and never builds a contract), sources it from
    ``config.consolidation.extraction_anonymize_token_envelope``
    - the one operator envelope value that also sizes session-tier
    extraction and graph-tier enrichment (:data:`_DEFAULT_ANONYMIZER_TOKEN_ENVELOPE`
    remains the module default for those other paths' own signatures; it
    is not read here).

    ``speaker_id`` is the resolved speaker store ID. It is NOT threaded
    to a local extraction call (there is none) - its only use here is the
    anonymizer's own speaker-anchor slot, which may carry a value only
    when it satisfies :func:`~paramem.utils.identity.is_speaker_id` -
    every other case (no speaker, or an unrecognised/non-token-shaped id)
    renders anchor-less. Anonymous-enrolled speakers KEEP the anchor
    (deliberate): a well-shaped ``speaker_id`` is sufficient regardless
    of whether ``speaker_name`` resolves to a display name - their
    session facts and cloud payloads stay in token space exactly like a
    named speaker's; what the reply boundary later renders for that
    token is a separate concern from this gate. See
    :func:`~paramem.cloud.anonymize.anonymize`'s ``speaker_id`` docstring
    for the render-time contract this gate feeds (fold-onto-token
    anonymization).

    ``AnonymizedContract.status``:

    * ``"ok"`` - anonymization ran. ``forward``/``reverse`` may still be
      empty - a legitimate "ran, found nothing in scope" verdict, not a
      failure; egress PROCEEDS. The caller derives the anonymized
      current turn the same way it derives every history turn -
      substituting ``payload.forward`` over the bare turn text - this
      helper does not rewrite or invert anything on the contract it
      returns.
    * ``"opted_out"`` - operator opted out (``categories`` empty).
    * ``"failed"`` - block, with ``failure`` naming the cause
      (``"guard"``, ``"model_unavailable"`` or ``"scan_failed"``) - see
      :class:`~paramem.cloud.anonymize.AnonymizedContract`. Callers must
      NEVER fall back to the original real-name transcript on this status.

    An empty or whitespace-only ``transcript`` is a caller precondition,
    not a status this function can return: :exc:`ValueError` is raised
    immediately, before :func:`~paramem.cloud.anonymize.anonymize` is
    ever reached - every caller of this function already checks for
    non-empty text before escalating.

    A raise out of :func:`~paramem.cloud.anonymize.anonymize` is a
    defect and propagates unchanged - this helper installs no handler
    around the call. Callers reach it on their own existing error path
    rather than have it laundered into a "block" verdict indistinguishable
    from a privacy decision.

    The companion :func:`~paramem.cloud.deanonymize.deanonymize_text`
    is the caller's exit gate for the cloud's response text.

    Raises:
        ValueError: *transcript* is empty or whitespace-only.
    """
    if not transcript or not transcript.strip():
        raise ValueError("anonymize_turn: transcript must be non-empty")

    anon_prompts = load_anonymizer_prompts(prompts_dir=prompts_dir)
    # The anonymizer's speaker-anchor slot may only carry a value that
    # satisfies is_speaker_id - every other case (no speaker, or an
    # unrecognised/non-token-shaped id) renders anchor-less.  See this
    # function's docstring for why anonymous-enrolled speakers keep it.
    anchor_speaker_id = speaker_id if is_speaker_id(speaker_id) else None
    history_lines = [format_turn(turn["role"], turn["text"]) for turn in history]
    model_facing_transcript = format_turn("user", transcript)

    # The ANCHOR call inside ``anonymize`` is structured extraction and
    # must run on the base weights, never the training-active adapter -
    # but only when a model is actually resident: on a cloud-only
    # deferral (model=None) there is no adapter to disable and no KV
    # cache to keep live, and base_model_inference raises on a non-PeftModel,
    # so the scope must not be entered at all. When a model IS present,
    # this scope disables the adapter and keeps the KV cache live for
    # that generate, restoring the model's entry state on exit.
    scope = base_model_inference(model) if model is not None else contextlib.nullcontext()
    with scope:
        payload = anonymize(
            [],
            model,
            tokenizer,
            transcript=model_facing_transcript,
            history=history_lines,
            categories=categories,
            speaker_name=speaker_name,
            speaker_id=anchor_speaker_id,
            token_envelope=token_envelope,
            prompts=anon_prompts,
        )

    return payload
