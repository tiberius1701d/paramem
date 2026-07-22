"""Generic linear stage-flow runner.

Boundary
--------

This module is flow TOPOLOGY, not extraction: it defines the shapes
(``StageContext``, ``StageState``, ``StageSpec``) and the single walk
function (:func:`run_flow`) that any linear, gated pipeline of stages
can be expressed against. It must NOT import any extraction primitive
(``_run_local_extraction``, ``_cloud_pipeline``, etc.) — that would
collapse the topology/primitive boundary this module
exists to keep, and would make the runner untestable without a model.
The one exception is :func:`~paramem.graph.phase_trace.chain_stopped`,
which is a control-flow signal (a contextvar read), not an extraction
primitive.

The first (and, at present, only) consumer is
``paramem.graph.flows.SESSION_EXTRACT`` — the ``local_extract`` ->
``second_order_extract`` -> ``anonymize`` -> ``enrich`` -> ``deanonymize``
-> ``rebuild`` flow that used to be an imperative if-cascade inside
``extract_graph``. Stage bodies are split across
``paramem.graph.stage_anonymize``, ``paramem.graph.stage_enrich`` and
``paramem.graph.flows`` (which also owns ``SESSION_EXTRACT`` itself and
``extract_graph``); see those modules for the concrete stage specs — this
module never references them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from paramem.graph.phase_trace import chain_stopped

if TYPE_CHECKING:
    # Type-only, and data SHAPES rather than extraction primitives — the
    # boundary this module keeps is "no primitive may be imported here",
    # not "no domain type may be named in an annotation".
    from paramem.cloud.anonymize import AnonymizedContract
    from paramem.cloud.deanonymize import CloudScope
    from paramem.graph.schema import SessionGraph


@dataclass(frozen=True)
class StageContext:
    """Every parameter of a flow run that is constant across its stages.

    Mirrors ``extract_graph``'s parameter list verbatim (see
    ``paramem.graph.flows.extract_graph``) — one field per
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
    validate: bool
    cloud_enabled: bool
    enrichment_provider: str
    enrichment_provider_model: str
    enrichment_provider_endpoint: str | None
    plausibility_judge: str
    plausibility_stage: str
    plausibility_model: str
    plausibility_endpoint: str | None
    scrub: set[str] | frozenset[str]
    correction_entity_types: set[str] | frozenset[str] | None


@dataclass(frozen=True)
class StageState:
    """The mutable-across-stages part of a flow run.

    ``graph`` is the only field the INITIAL state carries, and the only
    one without a default: every other field is the OUTPUT of some stage
    and is meaningless before that stage has run. That split is what the
    well-formedness check in ``tests/graph/test_flow.py`` keys on — it
    seeds "available" with the defaultless fields and then unions each
    spec's ``produces`` in order, so a stage ordered ahead of its
    producer fails mechanically.

    Grow this shell only when a concrete stage needs a new field to
    consume or produce; no speculative fields.

    Attributes:
        graph: The session graph under construction. Threaded and rebound
            by every stage.
        payload: The local anonymizer's result for this session
            (:class:`~paramem.cloud.anonymize.AnonymizedContract`),
            produced by the ``anonymize`` stage and consumed by
            ``enrich`` (which derives ``reverse``/``anon_transcript``/the
            anonymized fact array from it). ``None`` before ``anonymize``
            has run, and ``anonymize``'s own ``terminal_when`` — set back
            to ``None`` on the fail-closed divert (anonymization parse
            failure) so the walk stops before ``enrich`` runs on a
            payload that was never produced.
        facts: The working fact set — dicts, not ``Relation`` objects.
            Anonymized (placeholder-bearing) as the ``enrich`` stage hands
            them over; de-anonymized after the ``deanonymize`` stage
            substitutes real names back in. Empty means "no facts
            survived", which is how ``enrich``'s ``terminal_when`` stops
            the walk before its tail siblings.
        scalar_facts: Facts routed off the relation surface because their
            object is a verbatim identifier (URL, email, DOI,
            version-tagged tool name). Partitioned inside the
            ``deanonymize`` span — deliberately BEFORE the deanon
            plausibility judge, whose drop rule targets URI-shaped
            tokens — and projected onto entity attributes by ``rebuild``.
        scope: The one anonymize/de-anonymize round-trip scope for the
            cloud response — the resolution map every substitution and
            the entity-type rebuild are keyed on.
        cloud_raw: The cloud enricher's raw response string, recorded to
            diagnostics by ``deanonymize``.
        updated_anon_transcript: The cloud's rewritten anonymized
            transcript, or ``None`` when it returned none (or its delta
            was rejected).
        original_relation_count: Relation count captured before the cloud
            pipeline ran — the ``rebuild`` recovery gate's second term.
        empty_cause: Which site emptied ``facts``, from the ``CAUSE_*``
            vocabulary in ``paramem.graph.empty_cause``, or ``None``
            while facts survive. Diagnostic for the ``judgment`` and
            ``breakage`` kinds; load-bearing for the ``routing`` kind,
            which tells ``rebuild``'s recovery gate that the empty
            relation set is legitimate (the facts moved to the entity-
            attribute surface) and no recovery is due.
    """

    graph: "SessionGraph"
    payload: "AnonymizedContract | None" = None
    facts: list[dict] = field(default_factory=list)
    scalar_facts: list[dict] = field(default_factory=list)
    scope: "CloudScope | None" = None
    cloud_raw: str | None = None
    updated_anon_transcript: str | None = None
    original_relation_count: int = 0
    empty_cause: str | None = None


@dataclass(frozen=True)
class StageSpec:
    """One stage in a linear flow.

    Attributes:
        stage: Concept name for the stage (e.g. ``"local_extract"``).
        trace_names: Every ``phase_trace`` phase the stage's body opens,
            in the order it opens them — transitively, so a stage whose
            body delegates to a composite primitive declares the phases
            that primitive opens too. ``()`` for a stage that opens none
            (a pure gate or a pure data transform). This is a
            DECLARATION, not a label: :func:`run_flow` never opens a
            phase itself (each stage body opens its own scopes exactly
            as it did before this runner existed), so the field's only
            job is to be checkable — a well-formedness check
            (``tests/graph/test_flow.py``) asserts every declared name is
            a :data:`~paramem.graph.phase_trace.PHASE_NAMES` member, and
            a reader can see which phases a run can produce without
            entering the stage bodies. A stage that opens two phases
            declares two: one name per phase, never a stand-in for the
            others.
        run: Callable invoked when the stage is enabled and applies.
            Receives ``(ctx, state)`` and returns the next ``StageState``.
        enabled_when: Predicate over ``ctx`` gating whether the stage
            runs at all (an operator/config toggle — e.g.
            ``cloud_pipeline``'s cloud-admission gate refusing egress).
            ``None`` means always enabled. A disabled stage
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
    trace_names: tuple[str, ...]
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
