"""Unit tests for paramem.graph.phase_trace.

Covers:
- record contract: name, outcome, raw_output, parsed, wall_clock_seconds
- failure path: exception inside the scope writes outcome="failed" with reason
  before propagating
- whitelist enforcement: unknown phase name raises ValueError
- contract enforcement: phase_trace outside extraction_trace raises RuntimeError
- nesting: re-entry of extraction_trace raises RuntimeError
- attach_to materialises records on graph.diagnostics["phases"]
- get_phases reads back what attach_to wrote
- ordering: phases are appended in the order they fired
- backward compat: graph.diagnostics flat keys are not disturbed
- decoupling: trace survives graph rebinding inside the scope
- prompt-recording contract: record_prompt appends to the active phase
  scope, no-ops outside one, and round-trips through to_dict/get_phases/records
- stop_at/chain_stopped: nesting, reset on exit, None is a no-op, invalid
  name raises, a failed outcome does not stop the chain
"""

from __future__ import annotations

import hashlib

import pytest

from paramem.graph.phase_trace import (
    PHASE_NAMES,
    PhaseRecord,
    chain_stopped,
    extraction_trace,
    get_phases,
    phase_trace,
    record_prompt,
    stop_at,
)
from paramem.graph.schema import SessionGraph


def _empty_graph() -> SessionGraph:
    return SessionGraph(session_id="test", timestamp="2026-05-05T00:00:00Z")


class TestPhaseTraceContract:
    def test_records_basic_phase(self):
        graph = _empty_graph()
        with extraction_trace() as trace:
            with phase_trace("local_extract") as t:
                t.set_raw("model emitted text")
                t.set_parsed({"entity_count": 3, "relation_count": 5})
            trace.attach_to(graph)
        phases = get_phases(graph)
        assert len(phases) == 1
        assert phases[0].name == "local_extract"
        assert phases[0].outcome == "ok"
        assert phases[0].raw_output == "model emitted text"
        assert phases[0].parsed == {"entity_count": 3, "relation_count": 5}
        assert phases[0].wall_clock_seconds >= 0.0

    def test_outcome_skipped_with_reason(self):
        graph = _empty_graph()
        with extraction_trace() as trace:
            with phase_trace("sota_enrich") as t:
                t.set_outcome("skipped", reason="no API key")
            trace.attach_to(graph)
        phases = get_phases(graph)
        assert phases[0].outcome == "skipped"
        assert phases[0].reason == "no API key"

    def test_extra_fields_recorded(self):
        graph = _empty_graph()
        with extraction_trace() as trace:
            with phase_trace("anonymize") as t:
                t.add("mapping_ambiguous_dropped", 2)
                t.add("verifier_strict", True)
            trace.attach_to(graph)
        phases = get_phases(graph)
        assert phases[0].extra == {
            "mapping_ambiguous_dropped": 2,
            "verifier_strict": True,
        }


class TestPhaseTraceFailure:
    def test_exception_records_failed_outcome_then_propagates(self):
        graph = _empty_graph()
        with extraction_trace() as trace:
            with pytest.raises(RuntimeError, match="boom"):
                with phase_trace("anonymize") as t:
                    t.set_raw("partial output")
                    raise RuntimeError("boom")
            trace.attach_to(graph)
        phases = get_phases(graph)
        assert len(phases) == 1
        assert phases[0].name == "anonymize"
        assert phases[0].outcome == "failed"
        assert phases[0].reason == "RuntimeError: boom"
        # Partial state captured before the exception is preserved.
        assert phases[0].raw_output == "partial output"


class TestPhaseTraceContractEnforcement:
    def test_phase_trace_outside_extraction_trace_raises(self):
        """Silent record loss is unacceptable — calling phase_trace
        outside an active extraction_trace must fail loudly.

        The conftest autouse fixture establishes a trace for every test;
        we explicitly clear it here to simulate the production miss-case
        (phase_trace called outside ``extract_graph``)."""
        from paramem.graph.phase_trace import _ACTIVE_TRACE

        token = _ACTIVE_TRACE.set(None)
        try:
            with pytest.raises(RuntimeError, match="outside an active extraction_trace"):
                with phase_trace("local_extract"):
                    pass
        finally:
            _ACTIVE_TRACE.reset(token)

    def test_extraction_trace_nesting_is_noop(self):
        """Inner ``extraction_trace`` reuses the active trace — phase
        records from the inner scope land on the outer trace, not a
        detached one.  This makes the API safe to use from test fixtures
        that auto-wrap every test."""
        graph = _empty_graph()
        with extraction_trace() as outer:
            with extraction_trace() as inner:
                assert inner is outer
                with phase_trace("local_extract") as t:
                    t.set_parsed({"emitted_inside": "inner"})
            outer.attach_to(graph)
        phases = get_phases(graph)
        assert len(phases) == 1
        assert phases[0].parsed == {"emitted_inside": "inner"}

    def test_unknown_phase_name_raises(self):
        with extraction_trace():
            with pytest.raises(ValueError, match="Unknown phase name"):
                with phase_trace("made_up_phase"):
                    pass

    def test_all_documented_phase_names_accepted(self):
        graph = _empty_graph()
        with extraction_trace() as trace:
            for name in PHASE_NAMES:
                with phase_trace(name):
                    pass
            trace.attach_to(graph)
        phases = get_phases(graph)
        assert [p.name for p in phases] == list(PHASE_NAMES)


class TestPhaseTraceOrdering:
    def test_phases_appear_in_firing_order(self):
        graph = _empty_graph()
        with extraction_trace() as trace:
            for name in ("local_extract", "anonymize", "sota_enrich", "deanon"):
                with phase_trace(name):
                    pass
            trace.attach_to(graph)
        phases = get_phases(graph)
        assert [p.name for p in phases] == [
            "local_extract",
            "anonymize",
            "sota_enrich",
            "deanon",
        ]


class TestPhaseRecordSerialization:
    def test_to_dict_drops_empty_extra(self):
        r = PhaseRecord(name="local_extract")
        d = r.to_dict()
        assert "extra" not in d
        assert d["name"] == "local_extract"
        assert d["outcome"] == "ok"

    def test_to_dict_keeps_non_empty_extra(self):
        r = PhaseRecord(name="anonymize", extra={"k": "v"})
        d = r.to_dict()
        assert d["extra"] == {"k": "v"}

    def test_to_dict_omits_prompts_when_empty(self):
        r = PhaseRecord(name="local_extract")
        d = r.to_dict()
        assert "prompts" not in d

    def test_to_dict_keeps_prompts_when_present(self):
        entry = {"path": None, "sha": "abc123", "template": "hi"}
        r = PhaseRecord(name="local_extract", prompts=[entry])
        d = r.to_dict()
        assert d["prompts"] == [entry]


class TestRecordPrompt:
    """record_prompt appends the loader's own resolution to the active
    phase scope. See paramem/graph/prompts.py::_load_prompt for the
    caller and tests/test_prompts_contract.py::TestLoadPromptPhaseTraceRecording
    for the end-to-end regression pin against the real prompt loader."""

    def test_record_prompt_appends_to_active_scope(self):
        graph = _empty_graph()
        with extraction_trace() as trace:
            with phase_trace("local_extract"):
                record_prompt(path="/some/path.txt", content="hello world")
            trace.attach_to(graph)
        phases = get_phases(graph)
        assert phases[0].prompts == [
            {
                "path": "/some/path.txt",
                "sha": hashlib.sha256(b"hello world").hexdigest()[:12],
                "template": "hello world",
            }
        ]

    def test_multiple_prompts_in_one_phase_accumulate_in_order(self):
        graph = _empty_graph()
        with extraction_trace() as trace:
            with phase_trace("local_extract"):
                record_prompt(path="/sys.txt", content="sys")
                record_prompt(path="/user.txt", content="usr")
            trace.attach_to(graph)
        phases = get_phases(graph)
        assert [p["path"] for p in phases[0].prompts] == ["/sys.txt", "/user.txt"]

    def test_record_prompt_outside_active_scope_is_noop(self):
        """No active phase_trace scope: record_prompt must no-op, never
        raise. Two real production call paths run the prompt loader with
        no phase scope open — ``anonymize_with_local_model`` reached from
        ``extract_and_anonymize_for_cloud`` (the live chat egress,
        extractor.py:1108) and from ``paramem.training.consolidation``
        (consolidation.py:2739) — so this must never become a raise the
        way phase_trace's own missing-scope case is."""
        from paramem.graph.phase_trace import _ACTIVE_SCOPE

        assert _ACTIVE_SCOPE.get() is None
        record_prompt(path="/x.txt", content="anything")  # must not raise

    def test_record_prompt_scoped_to_current_phase_only(self):
        """A record_prompt call after a phase_trace block has exited must
        not retroactively attach to the already-closed phase."""
        graph = _empty_graph()
        with extraction_trace() as trace:
            with phase_trace("local_extract"):
                record_prompt(path="/a.txt", content="a")
            # Scope has exited — this call is outside any active phase.
            record_prompt(path="/b.txt", content="b")
            trace.attach_to(graph)
        phases = get_phases(graph)
        assert len(phases) == 1
        assert phases[0].prompts == [
            {
                "path": "/a.txt",
                "sha": hashlib.sha256(b"a").hexdigest()[:12],
                "template": "a",
            }
        ]

    def test_get_phases_and_records_roundtrip_prompts(self):
        graph = _empty_graph()
        with extraction_trace() as trace:
            with phase_trace("anonymize"):
                record_prompt(path="/p.txt", content="tmpl")
            trace.attach_to(graph)
            in_process = trace.records
        expected = [
            {
                "path": "/p.txt",
                "sha": hashlib.sha256(b"tmpl").hexdigest()[:12],
                "template": "tmpl",
            }
        ]
        assert in_process[0].prompts == expected
        phases = get_phases(graph)
        assert phases[0].prompts == expected


class TestActiveScopeResetDiscipline:
    """Pins ``_ACTIVE_SCOPE``'s reset discipline in ``phase_trace``'s
    ``finally: _ACTIVE_SCOPE.reset(token)``.

    A future edit that swapped ``.reset(token)`` for ``.set(None)`` would
    still make ``record_prompt`` no-op cleanly at the top level, so
    nothing here would raise — but on NESTED ``phase_trace`` exit it
    would clobber the OUTER scope with ``None`` instead of restoring it,
    silently dropping any prompt the outer phase records afterward (or,
    with a different bug shape, cross-attributing it to the wrong
    phase). Nothing else in this test file would notice; these tests
    exist specifically to catch that class of regression.
    """

    def test_exception_leaves_active_scope_restored_not_leaked(self):
        from paramem.graph.phase_trace import _ACTIVE_SCOPE

        assert _ACTIVE_SCOPE.get() is None
        with extraction_trace():
            with pytest.raises(RuntimeError, match="boom"):
                with phase_trace("anonymize"):
                    assert _ACTIVE_SCOPE.get() is not None
                    raise RuntimeError("boom")
            # The scope opened by the failed phase_trace must not leak
            # past its own `with` block.
            assert _ACTIVE_SCOPE.get() is None
        assert _ACTIVE_SCOPE.get() is None

    def test_nested_phase_trace_restores_outer_scope_not_none(self):
        """Inner ``phase_trace`` exit must restore the OUTER scope object
        — not ``None`` — so a ``record_prompt`` call made after the inner
        block, but still inside the outer block, lands on the OUTER
        phase's record rather than being silently dropped."""
        graph = _empty_graph()
        with extraction_trace() as trace:
            with phase_trace("local_extract"):
                record_prompt(path="/outer-before.txt", content="outer-before")
                with phase_trace("anonymize"):
                    record_prompt(path="/inner.txt", content="inner")
                # Inner scope has exited — this must land back on the
                # OUTER ("local_extract") phase's scope, not be dropped.
                record_prompt(path="/outer-after.txt", content="outer-after")
            trace.attach_to(graph)

        phases = get_phases(graph)
        assert [p.name for p in phases] == ["anonymize", "local_extract"]
        inner = next(p for p in phases if p.name == "anonymize")
        outer = next(p for p in phases if p.name == "local_extract")
        assert [pr["path"] for pr in inner.prompts] == ["/inner.txt"]
        assert [pr["path"] for pr in outer.prompts] == [
            "/outer-before.txt",
            "/outer-after.txt",
        ]

    def test_nested_phase_trace_inner_raise_still_restores_outer_scope(self):
        """An exception inside the INNER ``phase_trace`` must still
        restore the OUTER scope on the way out (not ``None``) — the
        outer block can keep recording after the inner phase fails."""
        graph = _empty_graph()
        with extraction_trace() as trace:
            with phase_trace("local_extract"):
                record_prompt(path="/outer-before.txt", content="outer-before")
                with pytest.raises(RuntimeError, match="boom"):
                    with phase_trace("anonymize"):
                        record_prompt(path="/inner.txt", content="inner")
                        raise RuntimeError("boom")
                record_prompt(path="/outer-after.txt", content="outer-after")
            trace.attach_to(graph)

        phases = get_phases(graph)
        inner = next(p for p in phases if p.name == "anonymize")
        outer = next(p for p in phases if p.name == "local_extract")
        assert inner.outcome == "failed"
        assert [pr["path"] for pr in inner.prompts] == ["/inner.txt"]
        assert [pr["path"] for pr in outer.prompts] == [
            "/outer-before.txt",
            "/outer-after.txt",
        ]


class TestBackwardCompat:
    def test_phases_list_does_not_disturb_existing_flat_keys(self):
        """Existing diagnostics like ``anonymize`` / ``plausibility`` /
        ``residual_dropped_facts`` must coexist with the new ``phases``
        list — same dict, additive only."""
        graph = _empty_graph()
        graph.diagnostics["anonymize"] = "ok"
        graph.diagnostics["plausibility_dropped"] = 3
        with extraction_trace() as trace:
            with phase_trace("anonymize"):
                pass
            trace.attach_to(graph)
        # Flat keys preserved.
        assert graph.diagnostics["anonymize"] == "ok"
        assert graph.diagnostics["plausibility_dropped"] == 3
        # Phase list present.
        assert isinstance(graph.diagnostics["phases"], list)
        assert len(graph.diagnostics["phases"]) == 1


class TestGraphRebindingDecoupling:
    def test_trace_survives_graph_rebinding(self):
        """The architectural reason for contextvar-backed trace: helpers
        rebind ``graph = ...`` (parse, HA validation, etc.) — the trace
        must NOT live on a particular SessionGraph."""
        with extraction_trace() as trace:
            graph_a = _empty_graph()
            with phase_trace("local_extract") as t:
                t.set_parsed({"step": "a"})
            # Simulate _parse_extraction returning a NEW SessionGraph.
            graph_b = _empty_graph()
            with phase_trace("anonymize") as t:
                t.set_parsed({"step": "b"})
            # Final graph is the one we attach to — trace records survive
            # even though graph_a is now orphaned.
            trace.attach_to(graph_b)
        phases = get_phases(graph_b)
        assert [p.name for p in phases] == ["local_extract", "anonymize"]
        assert phases[0].parsed == {"step": "a"}
        assert phases[1].parsed == {"step": "b"}
        # graph_a never received an attach; remains pristine.
        assert get_phases(graph_a) == []


class TestExtractGraphStopPhase:
    """Integration: wrapping ``extract_graph(...)`` in ``with
    stop_at(NAME):`` returns immediately after the named phase records,
    and the phases beyond it do not fire.

    Mocks ``_generate_extraction`` so no real model is required.  The
    contract under test is the wiring: the pipeline checks
    ``chain_stopped()`` after each phase block and short-circuits.
    """

    def test_stop_phase_local_extract_skips_everything_after(self, monkeypatch):
        from unittest.mock import MagicMock

        from paramem.graph.extractor import extract_graph

        # Mistral output: minimal valid JSON the parser accepts.
        monkeypatch.setattr(
            "paramem.graph.extractor._generate_extraction",
            lambda *a, **kw: (
                '{"entities": [{"name": "Alex", "entity_type": "person"}], '
                '"relations": [{"subject": "Alex", "predicate": "lives_in", '
                '"object": "Berlin", "relation_type": "factual", "confidence": 1.0}], '
                '"summary": "Alex lives in Berlin."}'
            ),
        )

        # Tokenizer / model stubs — extraction prompt formatting paths.
        with stop_at("local_extract"):
            graph = extract_graph(
                model=MagicMock(),
                tokenizer=MagicMock(),
                transcript="Alex lives in Berlin.",
                session_id="test-stop-phase",
                speaker_id="speaker0",
                speaker_name="Alex",
                validate=False,  # don't try the SOTA pipeline
                ha_validation=False,
                scrub={"person name"},
            )

        phase_names = [p["name"] for p in graph.diagnostics.get("phases", [])]
        assert phase_names == ["local_extract"]

    def test_stop_phase_invalid_name_raises(self):
        with pytest.raises(ValueError, match="not a valid phase name"):
            with stop_at("anonymise"):  # British spelling — should fail
                pass

    def test_stop_phase_local_extract_returns_populated_graph_not_empty_stub(self, monkeypatch):
        """Regression pin (one of two named in the flag-vs-exception design
        review): an exception/sentinel-based stop would raise from INSIDE
        ``_run_local_extraction``'s own ``phase_trace`` scope, before its
        ``return graph`` (after the ``with phase_trace(...)`` block)
        executes — leaving ``extract_graph``'s ``graph`` bound to the EMPTY
        stub it was initialised to before ``_run_local_extraction`` was
        even called (``extractor.py``: the stub binds at trace-open, is
        rebound by the ``_run_local_extraction`` call).  The flag design
        checks ``chain_stopped()`` only AFTER that assignment lands, so the
        returned graph must carry the real parsed entities/relations, not
        the stub."""
        from unittest.mock import MagicMock

        from paramem.graph.extractor import extract_graph

        monkeypatch.setattr(
            "paramem.graph.extractor._generate_extraction",
            lambda *a, **kw: (
                '{"entities": [{"name": "Alex", "entity_type": "person"}], '
                '"relations": [{"subject": "Alex", "predicate": "lives_in", '
                '"object": "Berlin", "relation_type": "factual", "confidence": 1.0}], '
                '"summary": "Alex lives in Berlin."}'
            ),
        )

        with stop_at("local_extract"):
            graph = extract_graph(
                model=MagicMock(),
                tokenizer=MagicMock(),
                transcript="Alex lives in Berlin.",
                session_id="test-stop-phase-populated",
                speaker_id="speaker0",
                speaker_name="Alex",
                validate=False,
                ha_validation=False,
                scrub={"person name"},
            )

        assert graph.relations, "must be the parsed graph, not the empty stub"
        assert graph.relations[0].subject == "Alex"
        assert graph.relations[0].object == "Berlin"

    def test_stop_phase_second_order_extract_retains_union(self, monkeypatch):
        """Regression pin (the second of the two): an exception/sentinel-
        based stop would raise from inside the second-order pass's own
        nested ``phase_trace`` scope, skipping the
        ``graph.relations.extend(second_order_graph.relations)`` /
        ``.entities.extend(...)`` calls that sit BETWEEN the second-order
        ``_run_local_extraction`` call and the stop check.  The flag
        design's check runs strictly after those two ``.extend()`` calls,
        so the pass-1 kinship edge and the recovered second-order fact
        must BOTH survive in the returned graph."""
        from unittest.mock import MagicMock

        from paramem.graph.extractor import extract_graph

        def _pass1(*a, **kw):
            return (
                '{"entities": [{"name": "speaker0", "entity_type": "person"}, '
                '{"name": "Nadeem", "entity_type": "person"}], '
                '"relations": [{"subject": "speaker0", "predicate": "has_brother", '
                '"object": "Nadeem", "relation_type": "social", "confidence": 1.0}]}'
            )

        def _second_order(*a, **kw):
            return (
                '{"entities": [{"name": "Nadeem", "entity_type": "person"}], '
                '"relations": [{"subject": "Nadeem", "predicate": "lives_in", '
                '"object": "Porto", "relation_type": "factual", "confidence": 1.0}]}'
            )

        def _fake_generate(*args, **kwargs):
            if kwargs.get("user_prompt_filename") == "extraction_second_order.txt":
                return _second_order()
            return _pass1()

        monkeypatch.setattr("paramem.graph.extractor._generate_extraction", _fake_generate)

        with stop_at("second_order_extract"):
            graph = extract_graph(
                model=MagicMock(),
                tokenizer=MagicMock(),
                transcript="My brother Nadeem lives in Porto.",
                session_id="test-stop-phase-union",
                speaker_id="speaker0",
                validate=False,
                ha_validation=False,
                scrub={"person name"},
            )

        phase_names = [p["name"] for p in graph.diagnostics.get("phases", [])]
        assert phase_names == ["local_extract", "second_order_extract"]
        assert any(r.subject == "speaker0" and r.object == "Nadeem" for r in graph.relations), (
            "pass-1 kinship edge must survive the union"
        )
        assert any(r.subject == "Nadeem" and r.object == "Porto" for r in graph.relations), (
            "second-order recovered fact must survive the union"
        )


class TestGetPhases:
    def test_empty_graph_returns_empty_list(self):
        graph = _empty_graph()
        assert get_phases(graph) == []

    def test_handles_malformed_phases_gracefully(self):
        """If something else writes garbage to diagnostics['phases'],
        get_phases should skip non-dict entries rather than crashing."""
        graph = _empty_graph()
        graph.diagnostics["phases"] = [
            {"name": "local_extract", "outcome": "ok"},
            "not a dict",
            {"name": "anonymize", "outcome": "skipped"},
        ]
        phases = get_phases(graph)
        assert [p.name for p in phases] == ["local_extract", "anonymize"]


class TestStopAt:
    """Unit tests for :func:`stop_at`/:func:`chain_stopped` — the
    low-level chain-stop mechanics, independent of ``extract_graph``.
    Exactly mirrors :func:`paramem.graph.prompts.prompt_overrides`:
    opened by the caller, never threaded as a parameter."""

    def test_default_chain_stopped_is_false(self):
        assert chain_stopped() is False

    def test_none_phase_is_a_no_op(self):
        with stop_at(None):
            with phase_trace("local_extract"):
                pass
            assert chain_stopped() is False

    def test_invalid_phase_name_raises(self):
        with pytest.raises(ValueError, match="not a valid phase name"):
            with stop_at("anonymise"):  # British spelling — should fail
                pass

    def test_matching_phase_sets_chain_stopped_true(self):
        with stop_at("anonymize"):
            with phase_trace("local_extract"):
                pass
            assert chain_stopped() is False
            with phase_trace("anonymize"):
                pass
            assert chain_stopped() is True

    def test_non_matching_phase_leaves_chain_stopped_false(self):
        with stop_at("deanon"):
            with phase_trace("local_extract"):
                pass
            with phase_trace("anonymize"):
                pass
            assert chain_stopped() is False

    def test_no_stop_at_scope_never_sets_chain_stopped(self):
        with extraction_trace():
            for name in PHASE_NAMES:
                with phase_trace(name):
                    pass
            assert chain_stopped() is False

    def test_soft_failed_outcome_does_not_set_chain_stopped(self):
        """A phase that finishes normally (no exception) but calls
        ``t.set_outcome("failed", ...)`` — e.g. a SOTA call returning
        ``None`` — must NOT be mistaken for the requested stop point."""
        with stop_at("anon_plausibility"):
            with phase_trace("anon_plausibility") as t:
                t.set_outcome("failed", reason="plausibility call returned None")
            assert chain_stopped() is False

    def test_raised_exception_does_not_set_chain_stopped(self):
        """A phase whose body raises never reaches the success branch at
        all, so it can never set the stopped flag — the ``except`` branch
        is a structurally separate code path."""
        with stop_at("anonymize"):
            with pytest.raises(RuntimeError, match="boom"):
                with phase_trace("anonymize"):
                    raise RuntimeError("boom")
            assert chain_stopped() is False

    def test_chain_stopped_stays_true_once_set(self):
        """Once set, the stopped flag is not cleared by subsequent phases
        (there is no path back to ``False`` within a single scope)."""
        with stop_at("local_extract"):
            with phase_trace("local_extract"):
                pass
            assert chain_stopped() is True
            with phase_trace("anonymize"):
                pass
            assert chain_stopped() is True

    def test_nested_stop_at_replaces_outer_value_and_restores_on_exit(self):
        """Mirrors ``prompt_overrides``: an inner ``stop_at`` call fully
        replaces the outer requested phase for its duration (no merge),
        and both the requested phase and the stopped flag are restored to
        the outer scope's state on exit."""
        with stop_at("local_extract"):
            with phase_trace("local_extract"):
                pass
            assert chain_stopped() is True
            with stop_at("anonymize"):
                # Inner scope gets its own fresh stopped=False, even
                # though the outer scope had already stopped.
                assert chain_stopped() is False
                with phase_trace("anonymize"):
                    pass
                assert chain_stopped() is True
            # Outer scope's own already-satisfied request is restored.
            assert chain_stopped() is True

    def test_reset_on_exit_leaves_no_active_request(self):
        with stop_at("local_extract"):
            with phase_trace("local_extract"):
                pass
            assert chain_stopped() is True
        assert chain_stopped() is False
