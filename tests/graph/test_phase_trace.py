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
"""

from __future__ import annotations

import pytest

from paramem.graph.phase_trace import (
    PHASE_NAMES,
    PhaseRecord,
    extraction_trace,
    get_phases,
    phase_trace,
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
        rebind ``graph = ...`` (parse, STT correction, HA validation,
        etc.) — the trace must NOT live on a particular SessionGraph."""
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
    """Integration: ``extract_graph(..., stop_phase=NAME)`` returns
    immediately after the named phase records, and the phases beyond it
    do not fire.

    Mocks ``_generate_extraction`` so no real model is required.  The
    contract under test is the wiring: pipeline checks ``stop_phase``
    after each phase block and short-circuits.
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
        graph = extract_graph(
            model=MagicMock(),
            tokenizer=MagicMock(),
            transcript="Alex lives in Berlin.",
            session_id="test-stop-phase",
            speaker_id="Speaker0",
            speaker_name="Alex",
            validate=False,  # don't try the SOTA pipeline
            stt_correction=False,
            ha_validation=False,
            stop_phase="local_extract",
        )

        phase_names = [p["name"] for p in graph.diagnostics.get("phases", [])]
        assert phase_names == ["local_extract"]

    def test_stop_phase_invalid_name_raises(self):
        from unittest.mock import MagicMock

        import pytest

        from paramem.graph.extractor import extract_graph

        with pytest.raises(ValueError, match="not a valid phase name"):
            extract_graph(
                model=MagicMock(),
                tokenizer=MagicMock(),
                transcript="x",
                session_id="t",
                speaker_id="Speaker0",
                stop_phase="anonymise",  # British spelling — should fail
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
