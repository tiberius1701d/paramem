"""``anonymize`` flow stage — the session flow's projection of the shared
anonymize component.

:func:`~paramem.cloud.anonymize.anonymize` is a SHARED component with
three consumers across three features: session-tier extraction (this
module's ``anonymize`` stage), graph-tier enrichment
(``paramem.training.graph_enrich``), and conversation egress
(``paramem.server.inference`` via
:func:`~paramem.graph.flows.anonymize_turn`). This module is the SESSION
FLOW's own projection of that shared call — the stage body that used to
be the front half of ``extractor._cloud_pipeline`` — not a second
implementation of the anonymize chain.

Because this module is the home of the anonymize STAGE CANDIDATE (as
opposed to a bare function call), it is also where a future
``conversation`` flow's own anonymize stage would land, rather than being
lifted back out into a fourth location: the stage-shaped wrapper over
:func:`~paramem.cloud.anonymize.anonymize` belongs here regardless of
which flow calls it.
"""

from __future__ import annotations

import logging

from paramem.cloud.anonymize import anonymize
from paramem.graph.extractor import _fallback_plausibility_on_raw, _vram_snapshot
from paramem.graph.flow import StageContext, StageState
from paramem.graph.phase_trace import phase_trace
from paramem.graph.prompts import _load_prompt
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

    1. ``ctx.scrub`` empty — operator opt-out: no anonymizer call, no
       phase trace (mirrors the pre-carve structure exactly — including
       that a ``stop_at("anonymize")`` request does NOT short-circuit
       here, since there is nothing to stop after: the "anonymize" phase
       never fires). The transcript egresses verbatim, sourced from the
       passed-in transcript — never a model artifact. The ``enrich``
       stage derives the (empty-mapping, identity) anonymized fact array
       from the returned ``payload``.
    2. Non-empty ``scrub`` — the local model is the SOLE scope authority:
       it classifies real values against ``scrub`` and returns BOTH the
       real_name -> placeholder mapping AND its own rewrite of the
       transcript with those values placeholdered. The ``anonymize``
       phase trace captures the raw model JSON so calibration can diff
       prompt variants on the anonymizer in isolation.

    Fail-closed divert: a parse failure or a missing/empty
    ``anonymized_transcript`` (``payload.status == "failed"``) falls back
    to local plausibility on the LOCAL-EXTRACT facts (never to the
    original real-name transcript over the cloud) and returns
    ``payload=None`` — the stage's ``terminal_when`` — so ``enrich`` does
    not run on a payload that was never produced.

    This stage body deliberately does NOT check ``chain_stopped()``
    itself, though an earlier version of this code did and returned early
    on a satisfied ``stop_at("anonymize")`` request. That check is gone
    from the body — :func:`~paramem.graph.flow.run_flow` already checks
    ``chain_stopped()`` after every stage that runs (``flow.py``), so the
    stage boundary now performs exactly the same early return the in-body
    check used to. A ``stop_at("anonymize")`` caller still gets back
    ``graph.relations`` at the local-extract output with the anonymize
    result in ``phases["anonymize"].parsed`` — verified by
    ``tests/test_extraction_pipeline.py``.
    """
    graph = state.graph
    original_count = len(graph.relations)

    _vram_snapshot(f"cloud_pipeline_entry session={graph.session_id}")
    if not ctx.scrub:
        # Operator opt-out: no anonymizer call, no phase trace (mirrors
        # the pre-unification structure exactly — including that a
        # ``stop_at("anonymize")`` request does NOT short-circuit here,
        # since there is nothing to stop after: the "anonymize" phase
        # never fires, so ``chain_stopped()`` can never become true from
        # it on this branch).  The transcript egresses verbatim, sourced
        # from the passed-in transcript — never a model artifact.  Facts
        # follow via the ``enrich`` stage's (empty-mapping, identity)
        # ``insert_placeholders`` call over this payload.  No prompt load
        # on this branch either — ``anonymize`` returns before touching
        # ``user_prompt_template``/``system_prompt``, so passing ``""``
        # for both here is safe.
        payload = anonymize(
            facts_from_relations(graph.relations),
            ctx.model,
            ctx.tokenizer,
            transcript=ctx.transcript,
            scrub=ctx.scrub,
            speaker_name=ctx.speaker_name,
            user_prompt_template="",
            system_prompt="",
        )
        graph.diagnostics["anonymize"] = "opted_out"
    else:
        # Anonymization step — THE one anonymize chain (A), shared with
        # every other cloud-egress path.  The local model is the SOLE
        # scope authority: it classifies real values against ``scrub``
        # and returns BOTH the real_name -> placeholder mapping AND its
        # own rewrite of the transcript with those values placeholdered.
        # Phase trace captures the raw model JSON so calibration can diff
        # prompt variants on the anonymizer in isolation.
        with phase_trace("anonymize") as t:
            anon_prompt = _load_prompt(
                "anonymization.txt", prompts_dir=ctx.prompts_dir, required=True
            )
            anon_system = _load_prompt(
                "anonymization_system.txt", prompts_dir=ctx.prompts_dir, required=True
            )
            payload = anonymize(
                facts_from_relations(graph.relations),
                ctx.model,
                ctx.tokenizer,
                transcript=ctx.transcript,
                scrub=ctx.scrub,
                speaker_name=ctx.speaker_name,
                max_tokens=ctx.max_tokens,
                seed=ctx.seed,
                user_prompt_template=anon_prompt,
                system_prompt=anon_system,
            )
            t.set_raw(payload.raw)
            t.set_parsed(
                {
                    "mapping": dict(payload.forward),
                    "mapping_size": len(payload.forward),
                    "parse_ok": payload.status != "failed",
                    "anonymized_transcript_len": len(payload.anon_transcript or ""),
                }
            )
            if payload.status == "failed":
                t.set_outcome(
                    "failed",
                    reason="anonymization parse failed or missing/empty anonymized_transcript",
                )
            elif not graph.relations:
                t.set_outcome("no_input", reason="graph has 0 relations")
        _vram_snapshot(f"after_anonymize session={graph.session_id}")
        if payload.status == "failed":
            # Fail-closed: parse failure OR a missing/empty
            # anonymized_transcript.  Never fall back to raw plausibility
            # on the ORIGINAL real-name transcript — fall back to local
            # plausibility on the LOCAL-EXTRACT facts instead (no cloud
            # egress at all).
            logger.warning("Anonymization failed — falling back to raw plausibility")
            graph.diagnostics["anonymize"] = "failed"
            return StageState(
                graph=_fallback_plausibility_on_raw(
                    graph,
                    ctx.transcript,
                    ctx.model,
                    ctx.tokenizer,
                    "anon_failed",
                    speaker_name=ctx.speaker_name,
                    speaker_id=ctx.speaker_id,
                    max_tokens=ctx.max_tokens,
                    plausibility_max_tokens=ctx.plausibility_max_tokens,
                    seed=ctx.seed,
                ),
                payload=None,
                original_relation_count=original_count,
            )
        # ``payload.norm_stats`` is the LIVE signal — reaches
        # ``mapping_ambiguous_dropped`` unconditionally now (this is the
        # only normalize call in the chain; see paramem.cloud.anonymize).
        if payload.norm_stats["dropped"]:
            graph.diagnostics["mapping_ambiguous_dropped"] = payload.norm_stats["dropped"]

        # CORE-map diagnostic.  The CORE map is never otherwise
        # persisted — ``anonymize.parsed.mapping`` above is the LLM HINT map,
        # recorded before ``anonymize``'s table build runs, not CORE.
        # Keys and COUNTS only — never the real names:
        # ``graph_snapshot.json`` (debug dumps under ``data/ha/debug/``)
        # serializes ``graph.diagnostics`` wholesale, and the placeholder
        # keyset is non-identifying while the values are the PII.
        graph.diagnostics["core_placeholders"] = {
            "keys": sorted(payload.reverse.keys()),
            "count": len(payload.reverse),
        }
        graph.diagnostics["anonymize"] = "ok"

    return StageState(graph=graph, payload=payload, original_relation_count=original_count)
