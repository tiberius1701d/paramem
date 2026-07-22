"""Test-support: drive SESSION_EXTRACT's cloud arc over a seeded graph.

Why this exists
---------------

``anonymize``, ``enrich``, ``deanonymize`` and ``rebuild`` are four
sibling stages of ``paramem.graph.flows.SESSION_EXTRACT``. Tests that
want to observe the arc end-to-end (relations out, diagnostics recorded)
must therefore walk the flow, not call one function — ``anonymize``/
``enrich`` alone stop at the hand-over.

This module contains no pipeline logic: it is the ARRANGE step those tests
need — build the run-constant :class:`~paramem.graph.flow.StageContext`,
seed a :class:`~paramem.graph.flow.StageState` with a caller-built graph,
and hand both to the real :func:`~paramem.graph.flow.run_flow` with the
real specs. Same construction ``extract_graph`` performs, minus the local
extraction stages that would otherwise need a model to produce the graph.
Parameter names are ``StageContext``'s own — there is no renaming layer.
"""

from __future__ import annotations

from pathlib import Path

from paramem.graph.extractor import _DEFAULT_FILTER_MAX_TOKENS, EnrichmentDelta
from paramem.graph.flow import StageContext, StageSpec, StageState, run_flow
from paramem.graph.flows import SESSION_EXTRACT
from paramem.graph.schema import SessionGraph

#: The stages downstream of local extraction: the ``anonymize``/``enrich``
#: pair and their two siblings. Sliced off the production list by stage
#: name so a reordering or a new stage cannot silently desync this
#: harness from the flow it is supposed to drive.
_ARC_STAGE_NAMES = ("anonymize", "enrich", "deanonymize", "rebuild")


def cloud_arc_specs() -> list[StageSpec]:
    """The ``SESSION_EXTRACT`` specs this harness walks, in flow order."""
    specs = [s for s in SESSION_EXTRACT if s.stage in _ARC_STAGE_NAMES]
    assert [s.stage for s in specs] == list(_ARC_STAGE_NAMES), (
        f"SESSION_EXTRACT no longer contains {_ARC_STAGE_NAMES} in order: "
        f"{[s.stage for s in SESSION_EXTRACT]}"
    )
    return specs


def run_cloud_stages(
    graph: SessionGraph,
    transcript: str,
    model,
    tokenizer,
    *,
    speaker_id: str,
    scrub: set[str] | frozenset[str],
    correction_entity_types: set[str] | frozenset[str] | None = None,
    plausibility_judge: str = "auto",
    plausibility_stage: str = "deanon",
    plausibility_model: str = "claude-sonnet-4-6",
    plausibility_endpoint: str | None = None,
    speaker_name: str | None = None,
    prompts_dir: str | Path | None = None,
    model_alias: str | None = None,
    enrichment_provider: str = "anthropic",
    enrichment_provider_model: str = "claude-sonnet-4-6",
    enrichment_provider_endpoint: str | None = None,
    max_tokens: int = _DEFAULT_FILTER_MAX_TOKENS,
    plausibility_max_tokens: int = _DEFAULT_FILTER_MAX_TOKENS,
    seed: int | None = None,
) -> SessionGraph:
    """Walk the cloud arc over ``graph`` and return the resulting graph.

    ``validate``/``cloud_enabled`` are pinned on so the ``anonymize``/
    ``enrich`` stages' shared ``enabled_when`` admits the arc — every
    caller of this harness is by definition testing the arc.

    Args:
        graph: Seed graph, standing in for local extraction's output.
        transcript: Real-name transcript (the deanon-stage judge's input).
        model / tokenizer: Local model handles, or ``None`` to exercise
            the guards that skip local LLM calls.
        speaker_id: Provenance stamped onto every rebuilt relation.
        scrub: PII-vocabulary hints; empty is the operator opt-out.
        Remaining arguments map 1:1 onto ``StageContext`` fields.

    Returns:
        ``state.graph`` after the walk — whatever stage last ran.
    """
    ctx = StageContext(
        model=model,
        tokenizer=tokenizer,
        transcript=transcript,
        session_id=graph.session_id,
        speaker_id=speaker_id,
        speaker_name=speaker_name,
        temperature=0.0,
        max_tokens=max_tokens,
        plausibility_max_tokens=plausibility_max_tokens,
        prompts_dir=prompts_dir,
        system_prompt_filename="extraction_system.txt",
        user_prompt_filename="extraction.txt",
        model_alias=model_alias,
        seed=seed,
        timestamp=graph.timestamp,
        source_type="transcript",
        validate=True,
        cloud_enabled=True,
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
    return run_flow(cloud_arc_specs(), ctx, StageState(graph=graph)).graph


def enrichment_side_effect(
    facts: list[dict],
    *,
    bindings: dict[str, str] | None = None,
    raw: str | None = None,
    info: dict | None = None,
):
    """Build a ``request_enrichment`` ``side_effect`` reproducing the
    RETIRED 5-tuple mock contract's final fact list.

    Before the cloud-admission redesign (2026-07-22), tests patched
    ``request_enrichment`` directly with a ``(facts, updated_transcript,
    bindings, raw, info)`` 5-tuple that fully overrode the stage's
    enriched output — ``request_enrichment`` no longer applies a delta at
    all (:func:`~paramem.graph.extractor._apply_enrichment_delta`, called
    by the ``enrich`` stage itself, does that now), so a bare
    ``return_value`` can no longer express "this is the final fact list"
    directly.

    ``facts`` here is that OLD mock's first tuple element — the desired
    FINAL surviving fact list. This helper reconstructs it as an
    :class:`~paramem.graph.extractor.EnrichmentDelta`, computed at CALL
    TIME (needs the real ``anon_facts`` the pipeline built, which a bare
    ``return_value`` cannot see):

    * If ``anon_facts`` is a PREFIX of ``facts`` (the common shape — cloud
      left the local facts untouched and only appended), the prefix is
      treated as unmodified (KEEP-by-default; ``drop=set()``) and
      whatever ``facts`` adds beyond that prefix becomes ``add``.
    * Otherwise (``facts`` does not extend ``anon_facts`` — the old mock
      fixture replaced the fact set wholesale), every input index is
      ``drop``ped and every entry of ``facts`` becomes an ``add``.

    Tests exercising ``modify`` (a partial field update on an existing
    index) or a ``drop`` NOT paired with a full replace build an
    :class:`~paramem.graph.extractor.EnrichmentDelta` directly instead —
    this helper only reproduces the two shapes every pre-redesign fixture
    in this test suite actually used.
    """

    def _fake(anon_facts, *_args, **_kwargs):
        n = len(anon_facts)
        if facts[:n] == anon_facts:
            add = list(facts[n:])
            drop: set[int] = set()
        else:
            add = list(facts)
            drop = set(range(n))
        delta = EnrichmentDelta(add=add, modify=[], drop=drop, bindings=dict(bindings or {}))
        return delta, raw, info or {}

    return _fake
