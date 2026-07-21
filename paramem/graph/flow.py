"""Generic linear stage-flow runner.

Boundary
--------

This module is flow TOPOLOGY, not extraction: it defines the shapes
(``StageContext``, ``StageState``, ``StageSpec``) and the single walk
function (:func:`run_flow`) that any linear, gated pipeline of stages
can be expressed against. It must NOT import any extraction primitive
(``_run_local_extraction``, ``_sota_pipeline``, ``_validate_with_ha_context``,
etc.) — that would collapse the topology/primitive boundary this module
exists to keep, and would make the runner untestable without a model.
The one exception is :func:`~paramem.graph.phase_trace.chain_stopped`,
which is a control-flow signal (a contextvar read), not an extraction
primitive.

The first (and, at present, only) consumer is
``paramem.graph.extractor.SESSION_EXTRACT`` — the four-phase
``local_extract`` -> ``second_order_extract`` -> ``ha_validation`` ->
``sota_pipeline`` flow that used to be an imperative if-cascade inside
``extract_graph``. See that module for the concrete stage specs; this
module never references them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from paramem.graph.phase_trace import chain_stopped

if TYPE_CHECKING:
    from paramem.graph.schema import SessionGraph


@dataclass(frozen=True)
class StageContext:
    """Every parameter of a flow run that is constant across its stages.

    Mirrors ``extract_graph``'s parameter list verbatim (see
    ``paramem.graph.extractor.extract_graph``) — one field per
    parameter, constructed once at the top of that function and passed
    to every stage's ``run`` callable. Adding a flow parameter means
    adding a field here; there is no other place a stage can read
    run-constant configuration from.
    """

    model: object
    tokenizer: object
    transcript: str
    session_id: str
    speaker_id: str
    speaker_name: str | None
    temperature: float
    max_tokens: int
    plausibility_max_tokens: int
    prompts_dir: str | Path | None
    system_prompt_filename: str
    user_prompt_filename: str
    model_alias: str | None
    seed: int | None
    timestamp: str | None
    source_type: str
    ha_context: dict | None
    ha_validation: bool
    validate: bool
    sota_enabled: bool
    noise_filter: str
    noise_filter_model: str
    noise_filter_endpoint: str | None
    plausibility_judge: str
    plausibility_stage: str
    scrub: set[str] | frozenset[str]
    correction_entity_types: set[str] | frozenset[str] | None


@dataclass(frozen=True)
class StageState:
    """The mutable-across-stages part of a flow run.

    Holds only ``graph`` — the one value ``extract_graph``'s stages
    thread and rebind today. Grow this shell only when a concrete stage
    needs a new field to consume or produce; no speculative fields.
    """

    graph: "SessionGraph"


@dataclass(frozen=True)
class StageSpec:
    """One stage in a linear flow.

    Attributes:
        stage: Concept name for the stage (e.g. ``"local_extract"``).
        trace_name: Informational label naming the ``phase_trace`` name
            the stage's body opens. Not used by :func:`run_flow` to open
            a phase itself — each stage body still opens its own
            ``phase_trace`` scope (or none, for a gate with no LLM
            call) exactly as it did before this runner existed.
        run: Callable invoked when the stage is enabled and applies.
            Receives ``(ctx, state)`` and returns the next ``StageState``.
        enabled_when: Predicate over ``ctx`` gating whether the stage
            runs at all (an operator/config toggle — e.g. ``ha_validation``
            being off). ``None`` means always enabled. A disabled stage
            is skipped with no call to ``run`` and therefore opens no
            phase trace record — mirrors an ``if <config flag>:`` gate
            that was never entered.
        applies_when: Predicate over ``state`` gating whether the stage
            runs given the graph built so far (a data-dependent gate —
            e.g. ``second_order_extract``'s named-non-speaker-person
            check). ``None`` means always applies. Same no-call,
            no-record skip semantics as ``enabled_when``.
        terminal_when: Predicate over the POST-run ``state`` that stops
            the walk after this stage completes (e.g. ``local_extract``'s
            empty-relations short-circuit). ``None`` means never
            terminal.
        requires: ``StageState`` field names this stage's ``run`` reads.
            Required (no default) — the per-stage input contract. A stage
            with no state dependency (its input lives entirely on
            ``StageContext``) declares ``frozenset()``.
        produces: ``StageState`` field names this stage's ``run`` writes.
            Required (no default) — the per-stage output contract. Must be
            non-empty: a stage that produces nothing has no observable
            effect on the state a later stage or the caller can read.

    ``requires``/``produces`` are metadata only — :func:`run_flow` does not
    read them. They are the declared I/O contract a well-formedness check
    (see ``tests/graph/test_flow.py``) verifies a flow's stage ORDER
    against: a stage requiring a field before any prior stage (or the
    initial state) produces it is a modeling bug the check catches
    mechanically rather than at runtime.
    """

    stage: str
    trace_name: str | None
    run: Callable[[StageContext, StageState], StageState]
    requires: frozenset[str]
    produces: frozenset[str]
    enabled_when: Callable[[StageContext], bool] | None = None
    applies_when: Callable[[StageState], bool] | None = None
    terminal_when: Callable[[StageState], bool] | None = None


def run_flow(flow: list[StageSpec], ctx: StageContext, state: StageState) -> StageState:
    """Walk ``flow`` in order, mutating ``state`` through each stage.

    For each :class:`StageSpec` in ``flow``:

    1. If ``enabled_when`` is set and false, skip the stage entirely —
       no call to ``run``, no phase record, ``state`` unchanged.
    2. Else if ``applies_when`` is set and false, skip likewise.
    3. Else call ``state = spec.run(ctx, state)``.

    After a stage actually RUNS (step 3), the walk checks
    :func:`~paramem.graph.phase_trace.chain_stopped` — if a calibration
    caller's ``stop_at`` request has been satisfied, the walk returns
    ``state`` immediately, before any later stage is considered. If the
    chain has not stopped, ``terminal_when(state)`` (when set) is
    checked next; a true result stops the walk the same way.

    A stage's ``run`` is never wrapped in a ``try/except`` here —
    an exception it raises propagates unchanged to ``run_flow``'s
    caller. Skipped stages therefore cannot fail, and a running stage's
    failure is never swallowed or converted into a skip.

    Args:
        flow: Ordered list of stage specs to walk.
        ctx: Run-constant parameters shared by every stage.
        state: Initial state (before any stage has run).

    Returns:
        The state after the last stage that ran, or after an early stop
        (chain-stopped or terminal).
    """
    for spec in flow:
        if spec.enabled_when is not None and not spec.enabled_when(ctx):
            continue
        if spec.applies_when is not None and not spec.applies_when(state):
            continue
        state = spec.run(ctx, state)
        if chain_stopped():
            return state
        if spec.terminal_when is not None and spec.terminal_when(state):
            return state
    return state
