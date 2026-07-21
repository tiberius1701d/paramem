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
        ha_context=None,
        ha_validation=False,
        validate=False,
        sota_enabled=False,
        noise_filter="",
        noise_filter_model="",
        noise_filter_endpoint=None,
        plausibility_judge="off",
        plausibility_stage="deanon",
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
                trace_name="never_runs",
                run=_run_stage("never_runs"),
                requires=_INERT_REQUIRES,
                produces=_INERT_PRODUCES,
                enabled_when=lambda c: False,
            ),
            StageSpec(
                stage="always_runs",
                trace_name="always_runs",
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
                trace_name="never_applies",
                run=_run_stage("never_applies"),
                requires=_INERT_REQUIRES,
                produces=_INERT_PRODUCES,
                applies_when=lambda s: False,
            ),
            StageSpec(
                stage="always_runs",
                trace_name="always_runs",
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
                trace_name="first",
                run=_run_stage("first"),
                requires=_INERT_REQUIRES,
                produces=_INERT_PRODUCES,
                terminal_when=lambda s: True,
            ),
            StageSpec(
                stage="second",
                trace_name="second",
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
                trace_name="first",
                run=_run_stage("first"),
                requires=_INERT_REQUIRES,
                produces=_INERT_PRODUCES,
            ),
            StageSpec(
                stage="second",
                trace_name="second",
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
                trace_name="local_extract",
                run=_first,
                requires=_INERT_REQUIRES,
                produces=_INERT_PRODUCES,
            ),
            StageSpec(
                stage="second",
                trace_name="second",
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
                trace_name="boom",
                run=_boom,
                requires=_INERT_REQUIRES,
                produces=_INERT_PRODUCES,
            ),
            StageSpec(
                stage="never_runs",
                trace_name="never_runs",
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

    Currently ``StageState`` holds only ``graph``, so every stage in
    SESSION_EXTRACT either requires nothing (``local_extract``, whose input
    is on ``StageContext``) or requires ``graph`` (already produced by
    ``local_extract``) — the check is trivially satisfied. It becomes
    load-bearing the moment ``StageState`` grows a second field and some
    stage is ordered ahead of the stage that is supposed to produce it —
    exactly the class of bug this test exists to catch mechanically
    instead of at runtime.
    """

    def test_requires_satisfied_by_prior_produces_in_order(self):
        # Deferred import: SESSION_EXTRACT lives in paramem.graph.extractor,
        # which (unlike this module) does pull in extraction primitives —
        # kept out of this file's top-level imports so the module docstring's
        # "no extraction primitive" claim stays true for every OTHER test
        # in this file.
        from paramem.graph.extractor import SESSION_EXTRACT

        available = {f.name for f in dataclasses.fields(StageState)}
        for spec in SESSION_EXTRACT:
            missing = spec.requires - available
            assert not missing, (
                f"{spec.stage!r} requires {sorted(missing)} not yet available "
                "from StageState's initial fields or any prior stage's produces"
            )
            assert spec.produces, f"{spec.stage!r} must produce something"
            available |= spec.produces
