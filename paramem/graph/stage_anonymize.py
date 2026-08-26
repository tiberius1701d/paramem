"""``anonymize`` flow stage — the session flow's projection of the shared
anonymize component.

:func:`~paramem.cloud.anonymize.anonymize` is a SHARED component with
three consumers across three features: session-tier extraction (this
module's ``anonymize`` stage), graph-tier enrichment
(``paramem.training.graph_enrich``), and conversation egress
(``paramem.server.egress`` via
:func:`~paramem.graph.flows.anonymize_turn`). This module is the SESSION
FLOW's own projection of that shared call — a stage body over
:func:`~paramem.cloud.anonymize.anonymize`, not a second implementation of
the anonymize chain.

Because this module is the home of the anonymize STAGE CANDIDATE (as
opposed to a bare function call), it is also where a future
``conversation`` flow's own anonymize stage would land, rather than being
lifted back out into a fourth location: the stage-shaped wrapper over
:func:`~paramem.cloud.anonymize.anonymize` belongs here regardless of
which flow calls it.
"""

from __future__ import annotations

import logging

from paramem.cloud.anonymize import anonymize, opted_out_contract
from paramem.graph.anonymizer_prompts import load_anonymizer_prompts
from paramem.graph.extractor import _fallback_plausibility_on_raw, _vram_snapshot
from paramem.graph.flow import StageContext, StageState
from paramem.graph.phase_trace import phase_trace
from paramem.graph.schema import facts_from_relations

logger = logging.getLogger(__name__)


def _stage_anonymize(ctx: StageContext, state: StageState) -> StageState:
    """``anonymize`` stage body — local anonymization of the session graph
    for cloud egress, via :func:`~paramem.cloud.anonymize.anonymize` (THE
    one anonymize chain, shared with every other cloud-egress path).

    Gated by ``ctx.validate`` plus
    :func:`~paramem.graph.flows._session_egress_permitted` (the shared
    cloud-admission verdict over ``cloud_enabled`` / provider / model / key
    / endpoint — the flow's ``enabled_when``) and ``state.graph.relations``
    being non-empty (the flow's ``applies_when``).

    Two branches:

    1. ``ctx.scrub_categories`` empty — operator opt-out: no tagger call,
       no anonymizer call, no phase trace — including that a
       ``stop_at("anonymize")`` request does NOT short-circuit here, since
       there is nothing to stop after: the "anonymize" phase never fires.
       The transcript egresses verbatim,
       sourced from the passed-in transcript — never a model artifact.
       The ``enrich`` stage derives the (empty-mapping, identity)
       anonymized fact array from the returned ``payload``.
    2. Non-empty ``scrub_categories`` — the span tagger's configured
       labels are the SOLE scope authority: it tags real values against
       those labels, and code-side substitution
       (:func:`~paramem.cloud.placeholders._substitute_whole_words`)
       produces both the real_name -> placeholder mapping AND the
       rewritten transcript with those values placeholdered. The one
       remaining local model call is the ANCHOR self-introduction
       question. The ``anonymize`` phase trace captures the raw tagger +
       ANCHOR record, plus ``status``/``failure``/``model_calls``/
       ``call_tokens`` (the SAME per-call telemetry
       :func:`~paramem.server.calibrate.dispatch_anonymize_facts`
       surfaces for the graph tier — one carrier, both calibration doors,
       and every production debug snapshot that serializes
       ``graph.diagnostics`` for free), so calibration can diagnose a
       failed run's specific cause from the same record.

    Fail-closed divert: ``payload.status == "failed"`` (``failure`` is
    ``"guard"`` or ``"tagger"`` — see
    :attr:`~paramem.cloud.anonymize.AnonymizedContract.failure` —
    recorded on this stage's own phase-trace record, never just a
    generic "failed") falls back to local plausibility on the
    LOCAL-EXTRACT facts (never to the original real-name transcript over
    the cloud) and returns ``payload=None`` — the stage's
    ``terminal_when`` — so ``enrich`` does not run on a payload that was
    never produced.

    On the ``ctx.scrub_categories``-non-empty branch (whether the call
    ends up ``"ok"`` or ``"failed"``), this stage also writes
    ``scan_dropped``/``inert_dropped``/``rekey_dropped``/
    ``scan_dropped_entries`` (the scan step's, the inert-key prune's, and
    the identity-reconciliation step's own per-entry drop counts and
    attribution — see
    :attr:`~paramem.cloud.anonymize.AnonymizedContract.scan_dropped` /
    :attr:`~paramem.cloud.anonymize.AnonymizedContract.inert_dropped` /
    :attr:`~paramem.cloud.anonymize.AnonymizedContract.rekey_dropped`,
    named for what each actually is — the diagnostic entries carry
    ``category``/``side``/``reason`` only, never the dropped surface's
    real text, which stays in-memory on the contract).

    This stage body deliberately does NOT check ``chain_stopped()``
    itself: :func:`~paramem.graph.flow.run_flow` already checks
    ``chain_stopped()`` after every stage that runs (``flow.py``), so the
    stage boundary performs the early return. A ``stop_at("anonymize")``
    caller still gets back ``graph.relations`` at the local-extract output
    with the anonymize result in ``phases["anonymize"].parsed``.
    """
    graph = state.graph
    original_count = len(graph.relations)

    _vram_snapshot(f"cloud_pipeline_entry session={graph.session_id}")
    if not ctx.scrub_categories:
        # Operator opt-out: no tagger call, no anonymizer call, no phase
        # trace, no prompt load — the ONE opt-out constructor, called
        # directly rather than routing through ``anonymize()`` purely to
        # reach it. A ``stop_at("anonymize")`` request does NOT
        # short-circuit here, since there is nothing to stop after: the
        # "anonymize" phase never fires, so ``chain_stopped()`` can never
        # become true from it on this branch.  The transcript egresses
        # verbatim, sourced from the passed-in transcript — never a
        # model artifact.  Facts follow via the ``enrich`` stage's
        # (empty-mapping, identity) ``insert_placeholders`` call over
        # this payload.
        payload = opted_out_contract(ctx.transcript, facts=facts_from_relations(graph.relations))
        graph.diagnostics["anonymize"] = "opted_out"
    else:
        # Anonymization step — THE one anonymize chain (A), shared with
        # every other cloud-egress path.  The span tagger's configured
        # labels are the SOLE scope authority: it tags real values in
        # scope, and code-side substitution rewrites both the forward
        # table and the transcript.  The one remaining local model call
        # is the ANCHOR self-introduction question.  Phase trace captures
        # the raw record so calibration can diagnose the anonymizer in
        # isolation.
        with phase_trace("anonymize") as t:
            anon_prompts = load_anonymizer_prompts(prompts_dir=ctx.prompts_dir)
            payload = anonymize(
                facts_from_relations(graph.relations),
                ctx.model,
                ctx.tokenizer,
                transcript=ctx.transcript,
                categories=ctx.scrub_categories,
                # Session-tier egress feeds the graph, never the reply
                # boundary — ctx.speaker_id is required/non-empty
                # (StageContext) and threaded unconditionally, unlike
                # chat egress's reply-boundary-gated anchor (see
                # paramem.graph.flows.anonymize_turn).
                speaker_name=ctx.speaker_name,
                speaker_id=ctx.speaker_id,
                token_envelope=ctx.anonymize_token_envelope,
                seed=ctx.seed,
                prompts=anon_prompts,
            )
            t.set_raw(payload.raw)
            t.set_parsed(
                {
                    "mapping": dict(payload.forward),
                    "mapping_size": len(payload.forward),
                    "status": payload.status,
                    "failure": payload.failure,
                    "anonymized_transcript_len": len(payload.anon_transcript or ""),
                    "tagger_windows": payload.tagger_windows,
                    "model_calls": payload.model_calls,
                    "scan_dropped": payload.scan_dropped,
                    "call_tokens": list(payload.call_tokens),
                }
            )
            if payload.status == "failed":
                t.set_outcome(
                    "failed",
                    reason=f"anonymization failed: {payload.failure}",
                )
            elif not graph.relations:
                t.set_outcome("no_input", reason="graph has 0 relations")
        _vram_snapshot(f"after_anonymize session={graph.session_id}")
        # ``payload.scan_dropped``/``inert_dropped``/``rekey_dropped``/
        # ``scan_dropped_entries`` are LIVE signals from the scan,
        # inert-key-prune, and identity-reconciliation steps (see
        # paramem.cloud.anonymize.anonymize) — written unconditionally,
        # and BEFORE the fail-closed return below, so a guard-failure
        # terminal (the scan ran; pruning ran; the reconciliation guard
        # fired afterward, on the pruned table) still carries the real
        # accumulated counts instead of any key being silently absent; a
        # tagger-failure terminal (the scan never ran) carries them at
        # their 0/empty default instead, since there is nothing to
        # report.  Diagnostics carry counts and categories only — never a
        # dropped entry's real text (that stays in-memory on ``payload``
        # itself), matching the keys-and-counts-only rule this module's
        # diagnostics already follow.
        graph.diagnostics["scan_dropped"] = payload.scan_dropped
        graph.diagnostics["inert_dropped"] = payload.inert_dropped
        graph.diagnostics["rekey_dropped"] = payload.rekey_dropped
        if payload.scan_dropped_entries:
            graph.diagnostics["scan_dropped_entries"] = [
                {"category": e["category"], "side": e["side"], "reason": e["reason"]}
                for e in payload.scan_dropped_entries
            ]
        if payload.status == "failed":
            # Fail-closed: the identity-reconciliation guard fired
            # (payload.failure == "guard"), or the span tagger was
            # unavailable (payload.failure == "tagger").  Never fall back
            # to raw plausibility on the ORIGINAL real-name transcript —
            # fall back to local plausibility on the LOCAL-EXTRACT facts
            # instead (no cloud egress at all).
            logger.warning("Anonymization failed — falling back to raw plausibility")
            graph.diagnostics["anonymize"] = "failed"
            return StageState(
                graph=_fallback_plausibility_on_raw(
                    graph,
                    ctx.transcript,
                    ctx.model,
                    ctx.tokenizer,
                    "anon_failed",
                    speaker_id=ctx.speaker_id,
                    max_tokens=ctx.max_tokens,
                    plausibility_max_tokens=ctx.plausibility_max_tokens,
                    seed=ctx.seed,
                ),
                payload=None,
                original_relation_count=original_count,
            )

        # CORE-map diagnostic.  ``anonymize.parsed.mapping`` above is ``payload.forward``
        # (surface -> placeholder, or the speaker id for the speaker's own surfaces),
        # as returned by ``anonymize()``.  ``payload.reverse`` is NOT ``forward``'s
        # inverse: it is the placeholder side of the table, with the speaker-folded
        # surfaces excluded and one entry per distinct placeholder — the first
        # forward key wins on a many-to-one collision (``paramem/cloud/anonymize.py``'s
        # ``invert_forward_mapping`` call).  This block records ``payload.reverse``'s KEYS
        # (placeholders, non-identifying) and COUNT only, never the real names its
        # values hold, since ``graph_snapshot.json`` debug dumps serialize
        # ``graph.diagnostics`` wholesale.
        graph.diagnostics["core_placeholders"] = {
            "keys": sorted(payload.reverse.keys()),
            "count": len(payload.reverse),
        }
        graph.diagnostics["anonymize"] = "ok"

    return StageState(graph=graph, payload=payload, original_relation_count=original_count)
