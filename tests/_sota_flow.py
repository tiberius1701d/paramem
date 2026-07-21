"""Test-support: drive SESSION_EXTRACT's cloud arc over a seeded graph.

Why this exists
---------------

The ``sota_pipeline`` composite, ``deanonymize`` and ``rebuild`` are three
sibling stages of ``paramem.graph.extractor.SESSION_EXTRACT``. Tests that
want to observe the arc end-to-end (relations out, diagnostics recorded)
must therefore walk the flow, not call one function — the composite alone
stops at the hand-over.

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

from paramem.graph.cloud_egress import _DEFAULT_FILTER_MAX_TOKENS
from paramem.graph.extractor import SESSION_EXTRACT
from paramem.graph.flow import StageContext, StageSpec, StageState, run_flow
from paramem.graph.schema import SessionGraph

#: The stages downstream of local extraction: the ``sota_pipeline``
#: composite and its two siblings. Sliced off the production list by
#: stage name so a reordering or a new stage cannot silently desync this
#: harness from the flow it is supposed to drive.
_ARC_STAGE_NAMES = ("sota_pipeline", "deanonymize", "rebuild")


def sota_arc_specs() -> list[StageSpec]:
    """The ``SESSION_EXTRACT`` specs this harness walks, in flow order."""
    specs = [s for s in SESSION_EXTRACT if s.stage in _ARC_STAGE_NAMES]
    assert [s.stage for s in specs] == list(_ARC_STAGE_NAMES), (
        f"SESSION_EXTRACT no longer contains {_ARC_STAGE_NAMES} in order: "
        f"{[s.stage for s in SESSION_EXTRACT]}"
    )
    return specs


def run_sota_stages(
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
    speaker_name: str | None = None,
    prompts_dir: str | Path | None = None,
    model_alias: str | None = None,
    noise_filter: str = "anthropic",
    noise_filter_model: str = "claude-sonnet-4-6",
    noise_filter_endpoint: str | None = None,
    max_tokens: int = _DEFAULT_FILTER_MAX_TOKENS,
    plausibility_max_tokens: int = _DEFAULT_FILTER_MAX_TOKENS,
    seed: int | None = None,
) -> SessionGraph:
    """Walk the cloud arc over ``graph`` and return the resulting graph.

    ``validate``/``sota_enabled`` are pinned on so the composite's
    ``enabled_when`` admits the arc — every caller of this harness is by
    definition testing the arc.

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
        sota_enabled=True,
        noise_filter=noise_filter,
        noise_filter_model=noise_filter_model,
        noise_filter_endpoint=noise_filter_endpoint,
        plausibility_judge=plausibility_judge,
        plausibility_stage=plausibility_stage,
        scrub=scrub,
        correction_entity_types=correction_entity_types,
    )
    return run_flow(sota_arc_specs(), ctx, StageState(graph=graph)).graph
