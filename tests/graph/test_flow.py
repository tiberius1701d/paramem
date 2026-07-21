"""Unit tests for paramem.graph.flow — the generic linear stage-flow runner.

Pure topology tests: no model, no extraction primitive. ``StageState.graph``
is stubbed with a plain counter object (a stand-in for a SessionGraph) since
``run_flow`` never inspects its shape — only the stages' own predicates do.

Covers:
- a disabled ``enabled_when`` skips the stage (no call, state unchanged)
- a false ``applies_when`` skips the stage likewise
- a true ``terminal_when`` after a stage runs stops the walk before later
  stages run
- a stage raising propagates unchanged (no swallow)
- SESSION_EXTRACT's declared ``requires``/``produces`` contract is
  well-formed (order-consistent with the actual StageState)
- SESSION_EXTRACT's declared ``trace_names`` are real PHASE_NAMES members
"""

from __future__ import annotations

import dataclasses

import pytest

from paramem.graph.flow import StageContext, StageSpec, StageState, run_flow
from paramem.graph.phase_trace import extraction_trace, phase_trace, stop_at

# requires/produces are required (no default) StageSpec fields, but the
# gating tests below don't exercise the I/O contract itself (run_flow never
# reads these fields — see TestSessionExtractWellFormed for the check that
# does). A single inert non-empty placeholder keeps every StageSpec() call
# below valid without each test having to reason about the contract.
_INERT_REQUIRES = frozenset()
_INERT_PRODUCES = frozenset({"graph"})


class _Counter:
    """Minimal stand-in for a SessionGraph: a mutable call log."""

    def __init__(self) -> None:
        self.ran: list[str] = []


def _make_ctx(**overrides) -> StageContext:
    """Build a minimal StageContext; every field has an inert default so
    individual tests only need to override what they care about."""
    fields = dict(
        model=None,
        tokenizer=None,
        transcript="",
        session_id="s0",
        speaker_id="speaker0",
        speaker_name=None,
        temperature=0.0,
        max_tokens=1,
        plausibility_max_tokens=1,
        prompts_dir=None,
        system_prompt_filename="system.txt",
        user_prompt_filename="user.txt",
        model_alias=None,
        seed=None,
        timestamp=None,
        source_type="transcript",
        validate=False,
        sota_enabled=False,
        noise_filter="",
        noise_filter_model="",
        noise_filter_endpoint=None,
        plausibility_judge="off",
        plausibility_stage="deanon",
        plausibility_model="",
        plausibility_endpoint=None,
        scrub=frozenset(),
        correction_entity_types=None,
    )
    fields.update(overrides)
    return StageContext(**fields)


def _run_stage(name: str):
    def _inner(ctx: StageContext, state: StageState) -> StageState:
        state.graph.ran.append(name)
        return state

    return _inner


class TestRunFlowGating:
    def test_disabled_enabled_when_skips_stage(self):
        """A stage whose ``enabled_when`` is false is never called and
        does not advance state."""
        flow = [
            StageSpec(
                stage="never_runs",
                trace_names=(),
                run=_run_stage("never_runs"),
                requires=_INERT_REQUIRES,
                produces=_INERT_PRODUCES,
                enabled_when=lambda c: False,
            ),
            StageSpec(
                stage="always_runs",
                trace_names=(),
                run=_run_stage("always_runs"),
                requires=_INERT_REQUIRES,
                produces=_INERT_PRODUCES,
            ),
        ]
        state = run_flow(flow, _make_ctx(), StageState(graph=_Counter()))
        assert state.graph.ran == ["always_runs"]

    def test_false_applies_when_skips_stage(self):
        """A stage whose ``applies_when`` is false given current state is
        never called and does not advance state."""
        flow = [
            StageSpec(
                stage="never_applies",
                trace_names=(),
                run=_run_stage("never_applies"),
                requires=_INERT_REQUIRES,
                produces=_INERT_PRODUCES,
                applies_when=lambda s: False,
            ),
            StageSpec(
                stage="always_runs",
                trace_names=(),
                run=_run_stage("always_runs"),
                requires=_INERT_REQUIRES,
                produces=_INERT_PRODUCES,
            ),
        ]
        state = run_flow(flow, _make_ctx(), StageState(graph=_Counter()))
        assert state.graph.ran == ["always_runs"]

    def test_terminal_when_stops_walk_before_later_stages(self):
        """A stage whose ``terminal_when`` is true after it runs stops the
        walk — a later stage in the flow never runs."""
        flow = [
            StageSpec(
                stage="first",
                trace_names=(),
                run=_run_stage("first"),
                requires=_INERT_REQUIRES,
                produces=_INERT_PRODUCES,
                terminal_when=lambda s: True,
            ),
            StageSpec(
                stage="second",
                trace_names=(),
                run=_run_stage("second"),
                requires=_INERT_REQUIRES,
                produces=_INERT_PRODUCES,
            ),
        ]
        state = run_flow(flow, _make_ctx(), StageState(graph=_Counter()))
        assert state.graph.ran == ["first"]

    def test_no_stop_at_scope_runs_every_stage(self):
        """With no active stop_at() request (production default, or an
        explicit stop_at(None)), chain_stopped() never fires and every
        stage in the flow runs."""
        flow = [
            StageSpec(
                stage="first",
                trace_names=(),
                run=_run_stage("first"),
                requires=_INERT_REQUIRES,
                produces=_INERT_PRODUCES,
            ),
            StageSpec(
                stage="second",
                trace_names=(),
                run=_run_stage("second"),
                requires=_INERT_REQUIRES,
                produces=_INERT_PRODUCES,
            ),
        ]
        with stop_at(None):
            state = run_flow(flow, _make_ctx(), StageState(graph=_Counter()))
        assert state.graph.ran == ["first", "second"]

    def test_chain_stopped_stops_walk_before_later_stages(self):
        """When a calibration stop_at request is satisfied by a stage's
        own phase_trace record, run_flow returns immediately after that
        stage — a later stage in the flow never runs. Mirrors
        extract_graph's chain_stopped() early returns."""

        def _first(ctx: StageContext, state: StageState) -> StageState:
            with phase_trace("local_extract"):
                state.graph.ran.append("first")
            return state

        flow = [
            StageSpec(
                stage="first",
                trace_names=("local_extract",),
                run=_first,
                requires=_INERT_REQUIRES,
                produces=_INERT_PRODUCES,
            ),
            StageSpec(
                stage="second",
                trace_names=(),
                run=_run_stage("second"),
                requires=_INERT_REQUIRES,
                produces=_INERT_PRODUCES,
            ),
        ]
        with extraction_trace():
            with stop_at("local_extract"):
                state = run_flow(flow, _make_ctx(), StageState(graph=_Counter()))
        assert state.graph.ran == ["first"]

    def test_stage_exception_propagates_unswallowed(self):
        """A stage raising propagates to run_flow's caller — no blanket
        try/except swallows it, and no later stage runs."""

        def _boom(ctx: StageContext, state: StageState) -> StageState:
            state.graph.ran.append("boom")
            raise ValueError("stage failed")

        flow = [
            StageSpec(
                stage="boom",
                trace_names=(),
                run=_boom,
                requires=_INERT_REQUIRES,
                produces=_INERT_PRODUCES,
            ),
            StageSpec(
                stage="never_runs",
                trace_names=(),
                run=_run_stage("never_runs"),
                requires=_INERT_REQUIRES,
                produces=_INERT_PRODUCES,
            ),
        ]
        state_holder = StageState(graph=_Counter())
        with pytest.raises(ValueError, match="stage failed"):
            run_flow(flow, _make_ctx(), state_holder)
        assert state_holder.graph.ran == ["boom"]


class TestSessionExtractWellFormed:
    """SESSION_EXTRACT's declared ``requires``/``produces`` I/O contract is
    well-formed against its own stage ORDER: for each spec, in order,
    ``spec.requires`` must already be satisfiable — either an initial
    ``StageState`` field or something a PRIOR spec's ``produces`` supplied
    — and every spec must ``produces`` something (a stage with no
    observable output is a modeling error).

    The INITIAL fields are the ones ``StageState`` declares without a
    default — ``extract_graph`` constructs the seed state from exactly
    those, and everything else is some stage's output, meaningless before
    that stage has run. Seeding ``available`` with every declared field
    instead would make the check vacuous: any ``requires`` would be
    satisfied by the dataclass definition alone, whatever the order.
    """

    @staticmethod
    def _initial_fields() -> set[str]:
        """``StageState`` fields the seed state supplies (no default)."""
        return {
            f.name
            for f in dataclasses.fields(StageState)
            if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
        }

    def test_requires_satisfied_by_prior_produces_in_order(self):
        # Deferred import: SESSION_EXTRACT lives in paramem.graph.extractor,
        # which (unlike this module) does pull in extraction primitives —
        # kept out of this file's top-level imports so the module docstring's
        # "no extraction primitive" claim stays true for every OTHER test
        # in this file.
        from paramem.graph.extractor import SESSION_EXTRACT

        available = self._initial_fields()
        for spec in SESSION_EXTRACT:
            missing = spec.requires - available
            assert not missing, (
                f"{spec.stage!r} requires {sorted(missing)} not yet available "
                "from StageState's initial fields or any prior stage's produces"
            )
            assert spec.produces, f"{spec.stage!r} must produce something"
            available |= spec.produces

    def test_declared_names_are_real_stage_state_fields(self):
        """Every ``requires``/``produces`` name must be an actual
        ``StageState`` field — a typo would otherwise sail through the
        order check as an unsatisfiable requirement or a phantom
        producer."""
        from paramem.graph.extractor import SESSION_EXTRACT

        known = {f.name for f in dataclasses.fields(StageState)}
        for spec in SESSION_EXTRACT:
            unknown = (spec.requires | spec.produces) - known
            assert not unknown, f"{spec.stage!r} names non-existent StageState field(s): {unknown}"

    def test_declared_trace_names_are_real_phase_names(self):
        """``trace_names`` declares the phases a stage's body opens, so
        every declared name must be a PHASE_NAMES member — the same
        whitelist ``phase_trace`` itself enforces at runtime. A stage
        declaring a name ``phase_trace`` would reject is a declaration
        that could never come true."""
        from paramem.graph.extractor import SESSION_EXTRACT
        from paramem.graph.phase_trace import PHASE_NAMES

        for spec in SESSION_EXTRACT:
            unknown = set(spec.trace_names) - set(PHASE_NAMES)
            assert not unknown, f"{spec.stage!r} declares non-PHASE_NAMES trace(s): {unknown}"

    def test_every_stage_declares_the_phases_it_opens(self):
        """The declaration is honest for the stages whose bodies open
        more than one phase: ``deanonymize`` opens BOTH the substitution
        phase and the local judge, and ``sota_pipeline`` opens the four
        its composite primitive runs. A single-name field could not say
        either."""
        from paramem.graph.extractor import SESSION_EXTRACT

        declared = {spec.stage: spec.trace_names for spec in SESSION_EXTRACT}
        assert declared["deanonymize"] == ("deanon", "deanon_plausibility")
        assert declared["sota_pipeline"] == (
            "anonymize",
            "entity_correction",
            "sota_enrich",
            "anon_plausibility",
        )
        assert declared["rebuild"] == ()

    def test_out_of_order_stage_fails_the_check(self):
        """The check is load-bearing: a stage requiring a field no
        predecessor produces must fail it."""
        consumer = StageSpec(
            stage="consumer",
            trace_names=(),
            run=_run_stage("consumer"),
            requires=frozenset({"facts"}),
            produces=frozenset({"graph"}),
        )
        producer = StageSpec(
            stage="producer",
            trace_names=(),
            run=_run_stage("producer"),
            requires=frozenset(),
            produces=frozenset({"facts"}),
        )
        available = self._initial_fields()
        # consumer BEFORE producer — unsatisfiable.
        assert consumer.requires - available
        # producer first — satisfiable.
        available |= producer.produces
        assert not (consumer.requires - available)
