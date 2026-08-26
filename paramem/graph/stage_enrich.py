"""``enrich`` flow stage — entity-surface correction, cloud enrichment, and
the anon-stage plausibility judge.

The sibling of ``paramem.graph.stage_anonymize``'s ``anonymize`` stage.
Where ``anonymize`` projects the shared
:func:`~paramem.cloud.anonymize.anonymize` chain, this module is
session-tier-specific: it drives the cloud enrichment round trip
(coreference resolution, compound splitting, safe reification) and the two
judges either side of it.
"""

from __future__ import annotations

import dataclasses
import logging

from paramem.cloud.admission import PROVIDER_KEY_ENV, evaluate_cloud_egress
from paramem.cloud.deanonymize import CloudScope
from paramem.cloud.placeholders import insert_placeholders
from paramem.graph.empty_cause import CAUSE_ANON_JUDGE, CAUSE_CLOUD_EMPTY
from paramem.graph.entity_correction import correct_entity_surfaces
from paramem.graph.extractor import (
    _DEFAULT_FILTER_TEMPERATURE,
    PLAUSIBILITY_FAILED,
    PLAUSIBILITY_OFF,
    PLAUSIBILITY_SKIPPED,
    ExtractionFailed,
    _apply_enrichment_delta,
    _cloud_facing_payload,
    _wait_for_gpu_ready,
    record_plausibility_state,
    record_plausibility_verdict,
    request_enrichment,
    request_plausibility,
)
from paramem.graph.flow import StageContext, StageState
from paramem.graph.phase_trace import chain_stopped, phase_trace

logger = logging.getLogger(__name__)


def _stage_enrich(ctx: StageContext, state: StageState) -> StageState:
    """``enrich`` stage body — entity-surface correction, cloud enrichment,
    anon-stage plausibility.

    Each sub-phase (``entity_correction``, ``cloud_enrich``,
    ``anon_plausibility``) records its own block via ``phase_trace``, and
    the stage still reads ``chain_stopped()`` after each one (a
    calibration caller's ``stop_at`` scope) to short-circuit mid-stage —
    unlike ``anonymize``'s stage-boundary check (owned by
    ``run_flow``), these are MID-stage checks between sub-phases the
    runner cannot see, so they stay in the body.

    Produces the seam the ``deanonymize`` and ``rebuild`` siblings
    consume: the surviving anonymized fact set, the cloud round-trip
    scope, the raw cloud response, and the cloud's updated anonymized
    transcript. Every early exit inside this stage hands back a state
    with an EMPTY ``facts``, which is exactly the spec's
    ``terminal_when`` — so the siblings do not run.

    Gated by ``ctx.validate`` plus
    :func:`~paramem.graph.flows._session_egress_permitted` (the shared
    cloud-admission verdict — the flow's ``enabled_when``, same term as
    ``anonymize``'s) and ``state.payload is not None`` (the flow's
    ``applies_when`` — nothing to enrich when ``anonymize`` diverted).

    Credential resolution: this stage (not ``anonymize``) is where
    ``ctx.cloud_enabled`` / ``ctx.enrichment_provider`` / ``ctx.enrichment_provider_model``
    / ``ctx.enrichment_provider_endpoint`` resolve to an ``api_key``/``endpoint``
    pair via :func:`~paramem.cloud.admission.evaluate_cloud_egress`
    — the same verdict ``enabled_when`` already computed to admit this
    stage, so no re-gate-and-bail is needed here: ``_stage_enrich`` only
    runs via ``run_flow``, which has already gated it.

    Every reachable path through the anon-stage judge gate records its
    state onto ``graph.diagnostics["plausibility_state_anon"]`` (the
    ``paramem.graph.extractor`` PLAUSIBILITY_* vocabulary), and a
    ``cloud_enrich`` run also writes ``declared_unobserved_tokens`` — the
    anonymizer's declared tokens cloud was never shown at all.
    """
    graph = state.graph
    payload = state.payload
    original_count = state.original_relation_count
    # Which site emptied the working fact set, if any. Recorded onto the
    # returned state for the recovery gate's diagnostics; nothing branches
    # on it. See paramem.graph.empty_cause for the vocabulary.
    empty_cause: str | None = None

    verdict = evaluate_cloud_egress(
        cloud_enabled=ctx.cloud_enabled,
        provider=ctx.enrichment_provider,
        model=ctx.enrichment_provider_model,
        endpoint=ctx.enrichment_provider_endpoint,
    )
    api_key = verdict.api_key
    endpoint = verdict.endpoint

    # ``anon_transcript`` is the chain's output regardless of which
    # ``anonymize`` branch ran — THE speaker-value guard in
    # ``build_forward_table`` (paramem/cloud/placeholders.py) applies on
    # every path by construction: ``payload.reverse`` is
    # :func:`~paramem.cloud.anonymize.anonymize`'s own inversion of
    # ``build_forward_table``'s ``forward`` (via
    # :func:`~paramem.cloud.placeholders.invert_forward_mapping`), after
    # dropping any entry whose value is speaker-id-shaped.
    anon_transcript = payload.anon_transcript

    # Phase — entity_correction.  Local model classifies+corrects misspelled
    # real place/organization/concept surfaces at two loci: the reverse map
    # VALUES ONLY (not keys — the forward map used to anonymize the
    # transcript below, and every downstream identity check keyed on
    # placeholders, are unaffected) and, when the "attributes" knob member
    # is set, graph.entities[*].attributes values (e.g. current_location).
    # correct_entity_surfaces() is read-only over its inputs — it returns
    # accepted corrections as data, applied below onto a NEW reverse dict
    # and graph.entities (the same graph object this function returns, so
    # the applied change reaches keyed-entry assembly/indexed-key
    # distillation downstream). Correction is independent of cloud
    # enrichment and safely precedes the fact-construction block below
    # (placeholder keys are untouched). See paramem.graph.entity_correction
    # for the full contract.
    with phase_trace("entity_correction") as t:
        correction_result = correct_entity_surfaces(
            payload.reverse,
            graph.entities,
            ctx.model,
            ctx.tokenizer,
            correction_entity_types=ctx.correction_entity_types,
            prompts_dir=ctx.prompts_dir,
            model_alias=ctx.model_alias,
            seed=ctx.seed,
        )
        applied = correction_result["applied"]
        verdicts = correction_result["verdicts"]
        # correct_entity_surfaces is read-only over its inputs — it returns
        # accepted corrections as data.  ``payload`` is a frozen dataclass:
        # rebinding it to a NEW dict (never mutating ``payload.reverse`` in
        # place) is what makes "frozen" actually hold — mutating the same
        # dict object in place through an alias would corrupt the
        # "immutable" contract's interior after ``declared``
        # (frozenset(reverse.keys())) has already been snapshotted from
        # it.  Corrected VALUES still reach the cloud call below (the
        # rebound ``payload`` is what ``CloudScope.response`` reads) and
        # ``graph.entities`` (feeds keyed-entry assembly).
        if applied:
            corrected_reverse = dict(payload.reverse)
            for entry in applied:
                if entry["locus"] == "placeholder":
                    corrected_reverse[entry["placeholder"]] = entry["after"]
                elif entry["locus"] == "attribute":
                    graph.entities[entry["entity_index"]].attributes[entry["key"]] = entry["after"]
            payload = dataclasses.replace(payload, reverse=corrected_reverse)
            graph.diagnostics["entity_corrections"] = applied
        graph.diagnostics["entity_correction_verdicts"] = verdicts
        t.set_parsed(
            {"applied_count": len(applied), "rejected_count": len(verdicts) - len(applied)}
        )
    if chain_stopped():
        # Calibration short-circuit: entity_correction completed; downstream
        # phases (cloud_enrich, anon_plausibility, deanon, …) are skipped.  graph.relations
        # remains the local-extract output; the correction result lives in
        # graph.diagnostics["entity_corrections"] (applied only) /
        # graph.diagnostics["entity_correction_verdicts"] (every evaluated
        # target) / phases[entity_correction].
        return dataclasses.replace(state, graph=graph, original_relation_count=original_count)

    # The fact array — rendered via
    # :func:`~paramem.cloud.placeholders.insert_placeholders`
    # (subject/object substituted through ``payload.forward``; every other
    # field copied verbatim) over ``payload.facts`` — the (real-name,
    # un-substituted) fact subset :func:`~paramem.cloud.anonymize.anonymize`
    # already cleared for egress: the full input facts verbatim on this
    # one call's success, or empty on its own fail-closed terminal (see
    # ``AnonymizedContract``'s docstring) — never a partial subset.  Facts
    # are never
    # taken from the model's response — the model's job is the TRANSCRIPT
    # (``anon_transcript``, already built above); the fact array is always
    # deterministic.  A fact can therefore never be lost, reworded, or
    # dropped by the anonymizer, and a placeholder cannot be glued into a
    # predicate at this stage — a predicate shaped like
    # ``language_proficiency_Language_3`` cannot occur here.  It can
    # still occur in cloud's *returned* facts, which is why the
    # deanon-stage predicate invariant (:func:`_apply_bindings`) stays.
    # An orphan placeholder in a fact is likewise impossible: every
    # placeholder a fact can carry comes from this same forward map.
    # Correct to reuse here even though ``entity_correction`` ran between
    # this payload's construction and this line — correction mutates only
    # the REVERSE map's values, never the forward map (or ``payload.facts``)
    # ``anon_facts`` is built from.  With an empty ``mapping`` (opt-out),
    # substitution is a no-op and facts egress verbatim.  Nothing between
    # the ``anonymize`` and ``enrich`` stages mutates ``graph.relations``,
    # so ``payload.facts`` (captured at anonymize time) stays byte-parity
    # with a fresh ``facts_from_relations(graph.relations)`` render for
    # the session tier's one-payload, one-scan, one-build ``status ==
    # "ok"`` case.
    anon_facts = insert_placeholders(payload.facts, payload.forward)

    # Phase — cloud_enrich.  Cloud (Anthropic by default) runs the
    # enrichment prompt; emits enriched facts + new_entity_bindings +
    # updated_anon_transcript.
    #
    # ``observed`` (computed inside ``CloudScope.response`` below) =
    # every placeholder token that occurs as a whole word in what cloud
    # was shown — the rendered facts_json (subject/predicate/object) and
    # the anonymized transcript.  A token glued inside a longer identifier
    # (e.g. a placeholder embedded in a predicate) is visible to cloud but
    # not observed.  This is CORE's legality domain for this cloud cycle:
    # only tokens cloud actually saw as a whole word may be treated as
    # legitimately bound.
    with phase_trace("cloud_enrich") as t:
        # :func:`_cloud_facing_payload` is the SAME render
        # :func:`request_enrichment` uses for its prompt, so the two cannot
        # drift.
        _facts_text, _transcript_text = _cloud_facing_payload(anon_facts, anon_transcript)
        # Send anon facts and transcript to cloud as the SCRIPT built them
        # (the anonymizer LLM returns the mapping and its own anonymized
        # transcript, but it never produces facts). The cloud prompt's convention
        # is "anonymizer placeholders are bare; only new entities
        # introduced by cloud use braced form (`{Prefix_N}`)". Cloud also
        # returns explicit bindings for any braced placeholders it
        # minted, so de-anonymization is pure dict substitution
        # downstream — no transcript diff, no LLM call, no regex
        # post-processing.
        delta, _cloud_raw, _cloud_info = request_enrichment(
            anon_facts,
            api_key,
            ctx.enrichment_provider,
            ctx.enrichment_provider_model,
            anon_transcript,
            endpoint=endpoint,
            max_tokens=ctx.max_tokens,
            prompts_dir=ctx.prompts_dir,
            speaker_id=ctx.speaker_id,
        )
        t.set_raw(_cloud_raw or "")
        if _cloud_info:
            graph.diagnostics["cloud_call_info"] = _cloud_info
            t.add("cloud_call_info", _cloud_info)
        if delta is None:
            # ``request_enrichment`` has already retried the call
            # ``_ENRICHMENT_MAX_ATTEMPTS`` times.  ``parse_path`` splits the two
            # failure modes it distinguishes (see its docstring):
            parse_path = (_cloud_info or {}).get("parse_path", "failed")
            attempts = (_cloud_info or {}).get("attempts", 0)
            if parse_path == "no_response":
                # OUTAGE — the provider was unreachable on every attempt
                # (``_cloud_call`` collapses API/network/SDK errors to
                # ``None``).  Abort so the batch's sessions stay PENDING and
                # re-run next cycle once cloud recovers: the session graph is
                # not merged here, so re-extraction enriches cleanly with no
                # dedup loss.  This is the abort machinery's real purpose —
                # a shape hiccup (below) is NOT routed here.
                t.set_outcome("failed", reason="cloud provider unreachable (no response)")
                raise ExtractionFailed(
                    "cloud_enrich",
                    "cloud enrichment provider unreachable after retries",
                )
            # HICCUP — the provider answered but never in the delta-envelope
            # shape.  Fail OPEN: a rare per-input deviation is a performance
            # degradation (this session goes un-enriched), NOT a reason to fail
            # the whole run.  Same policy as the graph-tier enrichment path
            # (``graph_enrich``) and both plausibility judges: keep the
            # pre-enrichment facts, warn loudly, and record the degradation in
            # diagnostics so the server layer can surface it on ``pstatus``.
            # Only the outage case above may raise: routing a shape hiccup to
            # the fatal path would abort a whole run over one malformed reply.
            # ``enriched_anon`` is the local facts unchanged;
            # ``updated_anon_transcript`` stays the anonymizer's transcript —
            # no cloud rewrite happened.
            logger.warning(
                "cloud enrichment unparseable after %d attempt(s) — keeping "
                "%d pre-enrichment fact(s), no enrichment applied this session "
                "(degraded, not fatal)",
                attempts,
                len(anon_facts),
            )
            enriched_anon = anon_facts
            updated_anon_transcript = anon_transcript
            # No cloud delta ⇒ no cloud-minted bindings.  Build the same
            # round-trip scope the ``deanonymize`` sibling consumes, but with
            # ``cloud_bindings=None`` (the no-op form the conversation path
            # uses) so de-anonymization resolves only the anonymizer's own
            # placeholders — there are no new ones to bind.
            scope = CloudScope.response(
                payload, cloud_bindings=None, sent=(_facts_text, _transcript_text)
            )
            graph.diagnostics["cloud_enrichment_degraded"] = {
                "parse_path": (_cloud_info or {}).get("parse_path", "failed"),
                "attempts": attempts,
                "response_chars": (_cloud_info or {}).get("response_chars", 0),
                "kept_facts": len(anon_facts),
            }
            t.set_parsed(
                {
                    "input_count": len(anon_facts),
                    "output_count": len(anon_facts),
                    "new_bindings_count": 0,
                    "new_bindings": {},
                    "updated_anon_transcript_len": len(updated_anon_transcript or ""),
                    "degraded": True,
                }
            )
            t.set_outcome("degraded", reason="cloud response unparseable after retries")
        else:
            # The ONE anonymize/deanonymize round-trip scope for this
            # response.  ``CloudScope.response`` computes ``observed`` from
            # the DECLARED vocabulary and the rendered payload (never a shape
            # scrape — see its docstring) and prunes any binding whose own
            # value carries an unresolvable placeholder.
            scope = CloudScope.response(
                payload, cloud_bindings=delta.bindings, sent=(_facts_text, _transcript_text)
            )
            # Apply the delta — per-triple accept/drop/revert against
            # ``scope.resolution``, never a whole-delta rejection: an
            # unresolvable ``add`` is dropped, an
            # unresolvable ``modify`` is reverted to its original fact, and
            # ``drop`` is honored unconditionally. See
            # ``_apply_enrichment_delta``'s own docstring for the full
            # per-action contract and the ``report`` keys below.
            enriched_anon, updated_anon_transcript, report = _apply_enrichment_delta(
                anon_facts, delta, scope, anon_transcript
            )
            graph.diagnostics["cloud_enrichment_report"] = report
            if report["rejected_adds"] or report["reverted_modifies"]:
                logger.warning(
                    "cloud enrichment: %d add(s) dropped, %d modify(ies) reverted "
                    "(unresolvable token(s): %s)%s.",
                    report["rejected_adds"],
                    report["reverted_modifies"],
                    report["rejected_tokens"][:5],
                    " — co-occurred with a non-empty drop set"
                    if report["drop_with_rejection"]
                    else "",
                )
            t.set_parsed(
                {
                    "input_count": len(anon_facts),
                    "output_count": len(enriched_anon),
                    "new_bindings_count": len(delta.bindings or {}),
                    "new_bindings": dict(delta.bindings) if delta.bindings else {},
                    "updated_anon_transcript_len": len(updated_anon_transcript or ""),
                    "observed_count": len(scope.observed),
                    "mapped_count": len(scope.resolution),
                    **report,
                }
            )
            if not enriched_anon:
                logger.info("cloud enrichment removed all relations")
                empty_cause = CAUSE_CLOUD_EMPTY
        # Legality-domain view, both branches above ("degraded" and the
        # normal path both bind `scope`): which of the anonymizer's own
        # CORE declared tokens were never shown to cloud at all. Read from
        # `payload.declared` (the CORE vocabulary), never `scope.declared`
        # (the union with cloud's own minted binding keys, which would
        # report every cloud mint as "unobserved" by construction). One
        # write here, not at either `CloudScope.response` call site above —
        # those would be the same write twice.
        graph.diagnostics["declared_unobserved_tokens"] = sorted(payload.declared - scope.observed)
    if chain_stopped():
        # Calibration short-circuit: Cloud enrichment block recorded,
        # downstream (anon_plausibility, deanon, deanon_plausibility) skipped.
        # graph.relations stays at the local-extract output; enrichment result
        # is in phases[cloud_enrich].
        return dataclasses.replace(state, graph=graph, original_relation_count=original_count)

    # Step 3a: Plausibility on anonymized data (cloud judge, stage="anon").
    # Only runs when: explicit cloud provider, plausibility_stage=="anon",
    # and enriched_anon is non-empty. Every branch records its judge state
    # (paramem.graph.extractor's PLAUSIBILITY_* vocabulary) so the absence
    # of plausibility_dropped_anon never has to be interpreted.
    # Guard: use `plausibility_judge in PROVIDER_KEY_ENV` (NOT != "off") —
    # "auto" is not a provider and would crash PROVIDER_KEY_ENV.get("auto").
    if ctx.plausibility_judge == "off":
        record_plausibility_state(graph, "anon", PLAUSIBILITY_OFF)
    elif ctx.plausibility_stage != "anon":
        record_plausibility_state(graph, "anon", PLAUSIBILITY_SKIPPED, reason="stage_not_anon")
    elif ctx.plausibility_judge not in PROVIDER_KEY_ENV:
        record_plausibility_state(
            graph, "anon", PLAUSIBILITY_SKIPPED, reason="judge_not_a_provider"
        )
    elif not enriched_anon:
        record_plausibility_state(graph, "anon", PLAUSIBILITY_SKIPPED, reason="no_facts")
    else:
        with phase_trace("anon_plausibility") as t:
            judge_verdict = evaluate_cloud_egress(
                cloud_enabled=True,
                provider=ctx.plausibility_judge,
                model=ctx.plausibility_model,
                endpoint=ctx.plausibility_endpoint,
            )
            if not judge_verdict.permitted:
                reason = "; ".join(judge_verdict.gaps)
                t.set_outcome("skipped", reason=reason)
                record_plausibility_state(
                    graph, "anon", PLAUSIBILITY_SKIPPED, reason="not_permitted"
                )
                logger.warning(
                    "Anon-stage plausibility (%s) skipped — %s", ctx.plausibility_judge, reason
                )
            else:
                verdict, plaus_raw = request_plausibility(
                    enriched_anon,
                    judge_verdict.api_key,
                    provider=judge_verdict.provider,
                    filter_model=judge_verdict.model,
                    anon_transcript=updated_anon_transcript or anon_transcript,
                    endpoint=judge_verdict.endpoint,
                    max_tokens=ctx.max_tokens,
                    temperature=_DEFAULT_FILTER_TEMPERATURE,
                    prompts_dir=ctx.prompts_dir,
                )
                # Cloud round-trip can take 30–90s during which the WSL2 GPU
                # goes idle and the next local CUDA op fails with
                # "device not ready". Wake + settle before the deanon-stage
                # local plausibility filter that follows below.
                _wait_for_gpu_ready()
                t.set_raw(plaus_raw or "")
                if verdict is not None:
                    pre_plaus = len(enriched_anon)
                    enriched_anon = verdict.kept
                    dropped_plaus = len(verdict.dropped)
                    graph.diagnostics["plausibility"] = "anon"
                    record_plausibility_verdict(
                        graph, "anon", verdict, judge=ctx.plausibility_judge
                    )
                    if not enriched_anon:
                        empty_cause = CAUSE_ANON_JUDGE
                    if plaus_raw:
                        graph.diagnostics["cloud_plausibility_raw_response"] = plaus_raw
                    t.set_parsed(
                        {
                            "judge": ctx.plausibility_judge,
                            "input_count": pre_plaus,
                            "kept_count": len(enriched_anon),
                            "dropped_count": dropped_plaus,
                            "dropped_facts": verdict.dropped,
                            "out_of_range": verdict.out_of_range,
                            "unattributed": verdict.unattributed,
                        }
                    )
                    logger.info(
                        "Anon-stage plausibility (%s): %d → %d facts (%d dropped)",
                        ctx.plausibility_judge,
                        pre_plaus,
                        len(enriched_anon),
                        dropped_plaus,
                    )
                else:
                    t.set_outcome("failed", reason="plausibility call returned None")
                    record_plausibility_state(
                        graph, "anon", PLAUSIBILITY_FAILED, reason="parse_failed"
                    )
                    t.set_parsed(
                        {
                            "judge": ctx.plausibility_judge,
                            "input_count": len(enriched_anon),
                            "kept_count": len(enriched_anon),
                            "dropped_count": 0,
                        }
                    )
                    logger.warning("Anon-stage plausibility call failed — keeping enriched facts")
        if chain_stopped():
            # Calibration short-circuit after the optional anon-stage judge.
            return dataclasses.replace(state, graph=graph, original_relation_count=original_count)

    # Empty-check guard: if enriched_anon is empty after the anon-stage
    # judge (or was already empty), clear the graph and hand back a state
    # whose empty ``facts`` is this stage's ``terminal_when`` — the
    # deanonymize/rebuild siblings must not run.
    if not enriched_anon:
        logger.info("No facts remain after anon-stage plausibility — returning empty graph")
        graph.relations = []
        graph.entities = []
        return dataclasses.replace(
            state,
            graph=graph,
            original_relation_count=original_count,
            empty_cause=empty_cause,
        )

    # Hand-over to the ``deanonymize`` sibling: de-anonymization via
    # state-machine substitution, the deanon-stage judge and the
    # relation/entity rebuild each live in their own stage.  ``scope``
    # is the ONE anonymize/de-anonymize round-trip scope for this
    # response — the substitution and the entity-type rebuild are both
    # keyed on it.
    return dataclasses.replace(
        state,
        graph=graph,
        facts=enriched_anon,
        scope=scope,
        cloud_raw=_cloud_raw,
        updated_anon_transcript=updated_anon_transcript,
        original_relation_count=original_count,
        empty_cause=empty_cause,
    )
