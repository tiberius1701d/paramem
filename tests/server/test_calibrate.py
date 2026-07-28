"""Tests for the calibration endpoint (paramem/server/calibrate.py).

Coverage:
- 404 when calibrate_endpoint_enabled is False (production-safe default)
- 503 when consolidation cycle is in flight (concurrency guard)
- 503 when local model is not loaded (cloud-only mode)
- 400 when prompt file is missing (no embedded fallback)
- 400 when a ``transcript`` field is not turn-marked (``[user]``/
  ``[assistant]`` — the production surface every prompt is calibrated
  on; see ``_require_turn_marked_transcript``)

The tests use a MagicMock-based fake state — they do NOT load the real
Mistral model.  End-to-end live testing of the actual extraction logic is
left to the existing integration suite + manual calibration runs.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from paramem.graph.extraction_pipeline import ExtractionPipeline
from paramem.graph.phase_trace import (
    _STOP_AT,
    PHASE_NAMES,
    chain_seed,
    chain_start,
    phase_trace,
)
from paramem.graph.prompts import _load_prompt
from paramem.graph.schema import SessionGraph
from paramem.server.calibrate import (
    _CHAIN,
    CalibrateAnonymizeFactsRequest,
    CalibrateChainRequest,
    CalibrateNameRequest,
    CalibrateNormalizeRequest,
    CalibrateParams,
    _effective_params,
    _preflight,
    _production_turn_markers,
    _require_turn_marked_transcript,
    _run_calibration,
    calibrate_anonymize_facts,
    calibrate_chain,
    calibrate_name,
    calibrate_normalize,
)
from paramem.server.config import CloudConfig, PathsConfig, SanitizationConfig


def _empty_graph() -> SessionGraph:
    """The shape every mocked pipeline entry returns."""
    return SessionGraph(session_id="calib", timestamp="2026-01-01T00:00:00Z")


def _state_disabled() -> dict:
    """Server state where calibrate is OFF (production default)."""
    consolidation_cfg = SimpleNamespace(calibrate_endpoint_enabled=False)
    config = SimpleNamespace(
        consolidation=consolidation_cfg,
        sanitization=SanitizationConfig(),
        cloud=CloudConfig(),
        paths=PathsConfig(),
    )
    return {
        "config": config,
        "consolidating": False,
        "model": MagicMock(),
        "tokenizer": MagicMock(),
    }


def _state_enabled() -> dict:
    consolidation_cfg = SimpleNamespace(calibrate_endpoint_enabled=True)
    # SanitizationConfig() carries the production default ``scrub`` list
    # (SanitizationConfig._DEFAULT_SCRUB) — the chain's anonymize step
    # sources ``scrub`` from the server config, the same policy knob every
    # production call site reads; it is never a request field.
    # ``cloud`` is the ONE cloud master switch (CloudConfig, ship default OFF);
    # the enrich tests flip ``config.cloud.enabled`` per case.
    config = SimpleNamespace(
        consolidation=consolidation_cfg,
        sanitization=SanitizationConfig(),
        cloud=CloudConfig(),
        paths=PathsConfig(),
    )
    return {
        "config": config,
        "consolidating": False,
        "model": MagicMock(),
        "tokenizer": MagicMock(),
        "consolidation_loop": MagicMock(),
        "model_id": "test-model",
    }


def _chain_side_effect(phase: str, extra=None):
    """A pipeline stand-in that behaves like a real chain run.

    A real run opens the phase it is stopped at; the calibration substrate
    refuses (400) when the declared step left no record. A double that
    skips the phase would therefore be testing the refusal, not the wiring
    it is aimed at.
    """

    def _run(*_args, **_kwargs):
        with phase_trace(phase):
            pass
        if extra is not None:
            extra(*_args, **_kwargs)
        return _empty_graph()

    return _run


def _stub_refiner() -> MagicMock:
    """A GraphTierRefiner stand-in returning the pass's diagnostics shape.

    The real pass needs a loaded model; what these tests pin is that the
    endpoint routes through the loop's construction site and reports what
    the pass returned, not how the pass itself clusters.
    """
    refiner = MagicMock()

    def _run_normalization():
        with phase_trace("normalize"):
            pass
        return {
            "groups_examined": 1,
            "groups_collapsed": 0,
            "edges_retired": 0,
            "chunks": 1,
            "skipped": False,
            "skip_reason": None,
        }

    refiner.run_normalization.side_effect = _run_normalization
    return refiner


def _peft_model_mock() -> MagicMock:
    """MagicMock that passes ``isinstance(model, PeftModel)``.

    ``disable_adapter`` returns a working context manager so the
    ``base_model_inference`` guard can enter and exit it, and the
    gradient-checkpointing toggles are mocked no-ops.  Used by the
    base-weights guard tests to prove the calibrate handlers disable the
    active adapter around the model call.
    """
    from peft import PeftModel

    model = MagicMock(spec=PeftModel)
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=None)
    cm.__exit__ = MagicMock(return_value=False)
    model.disable_adapter = MagicMock(return_value=cm)
    model.gradient_checkpointing_disable = MagicMock()
    model.gradient_checkpointing_enable = MagicMock()
    return model


class TestPreflight:
    def test_404_when_flag_disabled(self):
        with pytest.raises(HTTPException) as exc:
            _preflight(_state_disabled())
        assert exc.value.status_code == 404
        assert "disabled" in exc.value.detail.lower()

    def test_503_when_consolidating(self):
        state = _state_enabled()
        state["consolidating"] = True
        with pytest.raises(HTTPException) as exc:
            _preflight(state)
        assert exc.value.status_code == 503
        assert "consolidation" in exc.value.detail.lower()
        assert exc.value.headers.get("Retry-After") == "60"

    def test_503_when_model_missing(self):
        state = _state_enabled()
        state["model"] = None
        with pytest.raises(HTTPException) as exc:
            _preflight(state)
        assert exc.value.status_code == 503
        assert "local model" in exc.value.detail.lower()

    def test_passes_when_enabled_idle_loaded(self):
        # Should not raise.
        _preflight(_state_enabled())


class TestRequireTurnMarkedTranscript:
    """Unit coverage for the shared turn-marking gate.

    Every ``/calibrate/*`` endpoint that accepts a ``transcript`` field
    routes through this check (see ``TestChainGuards`` below for the
    per-use-case 400s).
    """

    def test_markers_derived_from_format_turns(self):
        """Markers come from SessionBuffer._format_turns, not a hardcoded list.

        Mutation guard: hardcode ``("[user]", "[assistant]")`` in
        ``_production_turn_markers`` instead of calling
        ``SessionBuffer._format_turns`` -> this test still passes (the
        values happen to match today), but the point of this assertion is
        cross-checked against the renderer directly rather than a literal,
        so a future role-vocabulary or marker-shape change in
        ``_format_turns`` is caught here instead of silently diverging.
        """
        from paramem.server.session_buffer import SessionBuffer

        expected = []
        for role in ("user", "assistant"):
            lines, _ = SessionBuffer._format_turns([{"role": role, "text": "x"}])
            marker, _sep, _rest = lines[0].partition(" ")
            expected.append(marker)
        assert _production_turn_markers() == tuple(expected)

    def test_unmarked_transcript_raises_400(self):
        with pytest.raises(HTTPException) as exc:
            _require_turn_marked_transcript("Should I follow up with Alex tomorrow?")
        assert exc.value.status_code == 400
        assert "turn-marked" in exc.value.detail.lower()

    def test_user_marked_transcript_passes(self):
        # Should not raise.
        _require_turn_marked_transcript("[user] Should I follow up with Alex tomorrow?")

    def test_assistant_marked_transcript_passes(self):
        # Should not raise.
        _require_turn_marked_transcript("[assistant] Sure, I can help with that.")

    def test_empty_transcript_raises_400(self):
        with pytest.raises(HTTPException) as exc:
            _require_turn_marked_transcript("")
        assert exc.value.status_code == 400


class TestChainDeclarations:
    """The declaration table is the contract every chain endpoint runs on.

    A typo'd phase name or a non-existent pipeline entry would otherwise
    surface only at request time, on the operator's machine, after the
    model is loaded.
    """

    def test_every_declared_phase_is_a_real_phase_name(self):
        for use_case, decl in _CHAIN.items():
            assert decl.start in PHASE_NAMES, use_case
            assert decl.stop is None or decl.stop in PHASE_NAMES, use_case

    def test_every_declared_entry_is_a_real_pipeline_method(self):
        for use_case, decl in _CHAIN.items():
            assert callable(getattr(ExtractionPipeline, decl.entry, None)), use_case

    def test_injects_vocabulary_is_closed(self):
        for use_case, decl in _CHAIN.items():
            assert decl.injects in {"transcript", "graph"}, use_case

    def test_graph_injecting_use_cases_enter_past_local_extract(self):
        """A use case that injects a graph must enter the chain AFTER the
        step that would have produced it — otherwise the seed is discarded
        by the local extractor's own rebind."""
        for use_case, decl in _CHAIN.items():
            if decl.injects == "graph":
                assert decl.start != "local_extract", use_case

    def test_only_the_open_stop_use_case_leaves_stop_unset(self):
        """Exactly one use case exists to inspect an operator-chosen point
        of a transcript-fed run; every other endpoint fixes its own stop."""
        open_stop = [u for u, d in _CHAIN.items() if d.stop is None]
        assert open_stop == ["extract"]


@pytest.mark.parametrize("use_case", sorted(_CHAIN))
class TestChainGuards:
    """Every unusable input is refused before any inference runs — the
    same guard for every use case, since they share one handler."""

    def _req(self, use_case: str, **overrides) -> CalibrateChainRequest:
        fields: dict = {
            "transcript": "[user] Should I follow up with Alex tomorrow?",
            "speaker_id": "speaker0",
        }
        if _CHAIN[use_case].injects == "graph":
            fields["graph"] = {"session_id": "calib", "timestamp": "2026-01-01T00:00:00Z"}
        fields.update(overrides)
        return CalibrateChainRequest(**fields)

    def test_disabled_404(self, use_case):
        with pytest.raises(HTTPException) as exc:
            calibrate_chain(_state_disabled(), use_case, self._req(use_case))
        assert exc.value.status_code == 404

    def test_consolidating_503(self, use_case):
        state = _state_enabled()
        state["consolidating"] = True
        with pytest.raises(HTTPException) as exc:
            calibrate_chain(state, use_case, self._req(use_case))
        assert exc.value.status_code == 503

    def test_unmarked_transcript_400(self, use_case):
        state = _state_enabled()
        req = self._req(use_case, transcript="Should I follow up with Alex tomorrow?")
        with pytest.raises(HTTPException) as exc:
            calibrate_chain(state, use_case, req)
        assert exc.value.status_code == 400
        assert "turn-marked" in exc.value.detail.lower()
        assert not state["consolidation_loop"].extraction.run.called

    def test_empty_speaker_id_400(self, use_case):
        state = _state_enabled()
        with pytest.raises(HTTPException) as exc:
            calibrate_chain(state, use_case, self._req(use_case, speaker_id=""))
        assert exc.value.status_code == 400
        assert "speaker_id" in exc.value.detail

    def test_missing_prompt_variant_400(self, use_case, tmp_path):
        """A named variant that does not exist in the calibration prompt
        directory is refused — calibration never silently falls back to the
        shipped prompt of the same name."""
        state = _state_enabled()
        state["config"].paths = PathsConfig(calibration=tmp_path)
        req = self._req(use_case, prompt_variants={"extraction.txt": "typo_variant.txt"})
        with pytest.raises(HTTPException) as exc:
            calibrate_chain(state, use_case, req)
        assert exc.value.status_code == 400
        assert "variant not found" in exc.value.detail.lower()
        assert not state["consolidation_loop"].extraction.run.called


class TestChainSeedGuards:
    """Guards specific to the use cases that inject a graph."""

    _GRAPH_USE_CASES = sorted(u for u, d in _CHAIN.items() if d.injects == "graph")

    @pytest.mark.parametrize("use_case", _GRAPH_USE_CASES)
    def test_missing_graph_400(self, use_case):
        state = _state_enabled()
        req = CalibrateChainRequest(transcript="[user] hi there", speaker_id="speaker0")
        with pytest.raises(HTTPException) as exc:
            calibrate_chain(state, use_case, req)
        assert exc.value.status_code == 400
        assert _CHAIN[use_case].start in exc.value.detail
        assert not state["consolidation_loop"].extraction.run.called

    @pytest.mark.parametrize("use_case", _GRAPH_USE_CASES)
    def test_invalid_graph_400(self, use_case):
        state = _state_enabled()
        req = CalibrateChainRequest(
            transcript="[user] hi there",
            speaker_id="speaker0",
            graph={"not_a": "session graph"},
        )
        with pytest.raises(HTTPException) as exc:
            calibrate_chain(state, use_case, req)
        assert exc.value.status_code == 400
        assert "SessionGraph" in exc.value.detail

    def test_transcript_use_cases_need_no_graph(self):
        for use_case, decl in _CHAIN.items():
            if decl.injects != "transcript":
                continue
            state = _state_enabled()
            state["consolidation_loop"].extraction.run.side_effect = _chain_side_effect(
                decl.stop or decl.start
            )
            state["consolidation_loop"].extraction.run_procedural.side_effect = _chain_side_effect(
                decl.stop or decl.start
            )
            result = calibrate_chain(
                state,
                use_case,
                CalibrateChainRequest(transcript="[user] hi there", speaker_id="speaker0"),
            )
            assert result["stage"] == use_case


class TestChainDispatch:
    """What the handler hands the chain: the declared entry, the declared
    start (with the validated seed), and the declared stop."""

    def _capturing_state(self, captured: dict) -> dict:
        """State whose pipeline records the chain-entry request it sees."""
        state = _state_enabled()

        def _record(*_args, **_kwargs):
            captured["start"] = chain_start()
            captured["seed"] = chain_seed()
            # No public reader for the requested stop: production reads only
            # whether it FIRED (chain_stopped), which a mocked run never
            # does. The request itself is what this asserts.
            captured["stop"] = _STOP_AT.get()
            captured["kwargs"] = _kwargs
            with phase_trace(captured["stop"] or captured["start"]):
                pass
            return _empty_graph()

        state["consolidation_loop"].extraction.run.side_effect = _record
        state["consolidation_loop"].extraction.run_procedural.side_effect = _record
        return state

    @pytest.mark.parametrize("use_case", sorted(_CHAIN))
    def test_declared_entry_is_the_one_called(self, use_case):
        state = self._capturing_state({})
        fields: dict = {"transcript": "[user] hi there", "speaker_id": "speaker0"}
        if _CHAIN[use_case].injects == "graph":
            fields["graph"] = {"session_id": "calib", "timestamp": "2026-01-01T00:00:00Z"}
        calibrate_chain(state, use_case, CalibrateChainRequest(**fields))
        entry = getattr(state["consolidation_loop"].extraction, _CHAIN[use_case].entry)
        assert entry.called

    @pytest.mark.parametrize("use_case", sorted(_CHAIN))
    def test_declared_start_and_stop_are_opened(self, use_case):
        captured: dict = {}
        state = self._capturing_state(captured)
        decl = _CHAIN[use_case]
        fields: dict = {"transcript": "[user] hi there", "speaker_id": "speaker0"}
        if decl.injects == "graph":
            fields["graph"] = {"session_id": "calib", "timestamp": "2026-01-01T00:00:00Z"}
        calibrate_chain(state, use_case, CalibrateChainRequest(**fields))
        assert captured["start"] == decl.start
        assert captured["stop"] == decl.stop

    def test_graph_seed_reaches_the_chain_validated(self):
        """The seed the chain receives is the parsed SessionGraph, not the
        raw request dict — the chain seeds StageState.graph with it."""
        captured: dict = {}
        state = self._capturing_state(captured)
        req = CalibrateChainRequest(
            transcript="[user] hi there",
            speaker_id="speaker0",
            graph={"session_id": "seeded", "timestamp": "2026-01-01T00:00:00Z"},
        )
        calibrate_chain(state, "anonymize", req)
        assert isinstance(captured["seed"], SessionGraph)
        assert captured["seed"].session_id == "seeded"

    def test_transcript_use_case_seeds_nothing(self):
        captured: dict = {}
        state = self._capturing_state(captured)
        calibrate_chain(
            state,
            "extract",
            CalibrateChainRequest(transcript="[user] hi there", speaker_id="speaker0"),
        )
        assert captured["seed"] is None

    def test_operator_stop_honoured_only_where_the_declaration_leaves_it_open(self):
        captured: dict = {}
        state = self._capturing_state(captured)
        req = CalibrateChainRequest(
            transcript="[user] hi there",
            speaker_id="speaker0",
            stop_phase="second_order_extract",
        )
        calibrate_chain(state, "extract", req)
        assert captured["stop"] == "second_order_extract"

    def test_operator_stop_ignored_where_the_endpoint_fixes_one(self):
        captured: dict = {}
        state = self._capturing_state(captured)
        req = CalibrateChainRequest(
            transcript="[user] hi there",
            speaker_id="speaker0",
            graph={"session_id": "calib", "timestamp": "2026-01-01T00:00:00Z"},
            stop_phase="second_order_extract",
        )
        calibrate_chain(state, "plausibility", req)
        assert captured["stop"] == "deanon_plausibility"

    def test_run_opens_a_calibration_scope_under_the_calibration_root(self, tmp_path):
        """The run's artifacts are directed at a per-run directory under
        paths.calibration — so a production hook firing DURING the run (the
        graph tier's normalization pass writes through the same writer)
        lands there too, independently of the production debug switch."""
        from paramem.utils.artifacts import _CALIBRATION_ROOT

        state = _state_enabled()
        state["config"].paths = PathsConfig(calibration=tmp_path)
        seen: dict = {}

        def _capture(*_args, **_kwargs):
            seen["root"] = _CALIBRATION_ROOT.get()

        state["consolidation_loop"].extraction.run.side_effect = _chain_side_effect(
            "local_extract", _capture
        )
        calibrate_chain(
            state,
            "extract",
            CalibrateChainRequest(transcript="[user] hello there", speaker_id="speaker0"),
        )

        assert seen["root"] is not None
        assert seen["root"].parent == tmp_path / "artifacts"
        assert seen["root"].name.startswith("extract_")
        # Scope closed on the way out — it never leaks past the run.
        assert _CALIBRATION_ROOT.get() is None

    def test_result_is_emitted_through_the_one_writer(self):
        """The response goes to the shared artifact hook — the single surface
        for pipeline artifacts — not to a second writer owned by this module.

        Mutation: reintroduce a private write in calibrate.py -> the hook is
        never called and this fails.
        """
        state = _state_enabled()
        state["consolidation_loop"].extraction.run.side_effect = _chain_side_effect("local_extract")
        req = CalibrateChainRequest(transcript="[user] hello there", speaker_id="speaker0")

        with patch("paramem.server.calibrate.on_calibration_result") as hook:
            result = calibrate_chain(state, "extract", req)

        assert hook.call_count == 1
        assert hook.call_args.args[0] == result

    def test_dispatch_surfaces_focus_step_raw_output(self):
        """The inspected step's ``raw_output`` is surfaced at the response top
        level, read off the ``PhaseRecord`` object (not a dict).

        Regression: the dispatch built ``record`` from ``r.to_dict()`` then read
        ``record.raw_output`` — an ``AttributeError`` on a dict, masked because
        the enrichment fail-closed raise short-circuited the walk before this
        line ever ran on a completed extract.  The other dispatch mocks return a
        graph with no attached phases (``get_phases`` empty → ``record is None``
        → ``""``), so only a graph carrying a real focus-step record exercises
        it.
        """
        state = _state_enabled()

        def _record(*_a, **_k):
            # (1) open the phase on the OUTER trace so the substrate's
            # declared-step-reached check passes; (2) attach a phase record with
            # a raw_output to the returned graph for the dispatch's focus lookup.
            with phase_trace("local_extract") as t:
                t.set_raw("RAW_FROM_LOCAL_EXTRACT")
            g = _empty_graph()
            g.diagnostics["phases"] = [
                {"name": "local_extract", "raw_output": "RAW_FROM_LOCAL_EXTRACT"}
            ]
            return g

        state["consolidation_loop"].extraction.run.side_effect = _record
        result = calibrate_chain(
            state,
            "extract",
            CalibrateChainRequest(transcript="[user] hi there", speaker_id="speaker0"),
        )
        assert result["raw_output"] == "RAW_FROM_LOCAL_EXTRACT"


class TestChainSessionSnapshot:
    """A transcript-extracting run persists the per-session graph snapshot into
    its own run dir — the same ``sessions/<id>/graph_snapshot.json`` artifact
    ``ConsolidationLoop.extract_session`` writes in production, so a calibration
    tree can be diffed file-for-file against a production cycle. Mid-chain
    endpoints inject a graph and run a sub-step, not a session extraction, and
    write no session snapshot."""

    def _paths_state(self, tmp_path, phase: str) -> dict:
        state = _state_enabled()
        state["config"].paths = PathsConfig(calibration=tmp_path)
        pipeline = state["consolidation_loop"].extraction
        pipeline.run.side_effect = _chain_side_effect(phase)
        pipeline.run_procedural.side_effect = _chain_side_effect(phase)
        return state

    def test_extract_writes_the_session_snapshot(self, tmp_path):
        state = self._paths_state(tmp_path, "local_extract")
        calibrate_chain(
            state,
            "extract",
            CalibrateChainRequest(transcript="[user] hi there", speaker_id="speaker0"),
        )
        snaps = list((tmp_path / "artifacts").glob("extract_*/sessions/calib/graph_snapshot.json"))
        assert len(snaps) == 1

    def test_procedural_writes_the_procedural_snapshot(self, tmp_path):
        state = self._paths_state(tmp_path, "procedural_extract")
        calibrate_chain(
            state,
            "procedural",
            CalibrateChainRequest(transcript="[user] hi there", speaker_id="speaker0"),
        )
        snaps = list(
            (tmp_path / "artifacts").glob(
                "procedural_*/sessions/calib/procedural_graph_snapshot.json"
            )
        )
        assert len(snaps) == 1

    def test_graph_injecting_use_case_writes_no_session_snapshot(self, tmp_path):
        state = self._paths_state(tmp_path, "anonymize")
        req = CalibrateChainRequest(
            transcript="[user] hi there",
            speaker_id="speaker0",
            graph={"session_id": "calib", "timestamp": "2026-01-01T00:00:00Z"},
        )
        calibrate_chain(state, "anonymize", req)
        assert not list((tmp_path / "artifacts").glob("**/*graph_snapshot.json"))


class TestDeclaredStepUnreached:
    """A calibration promises ONE step's output. When the configured chain
    cannot reach that step, the operator must get a refusal naming the gap
    — never a 200 whose provenance is silently empty.

    Mutation: drop the check in ``_run_calibration`` -> these fail, and a
    cloud-disabled server answers /calibrate/enrich with an empty envelope
    exactly as the pre-unification standalone endpoints never did (they
    raised 400 from their own guard).
    """

    def _state_with_chain(self, *, phases_that_run: list[str]) -> dict:
        state = _state_enabled()

        def _run(*_args, **_kwargs):
            for name in phases_that_run:
                with phase_trace(name):
                    pass
            return _empty_graph()

        state["consolidation_loop"].extraction.run.side_effect = _run
        return state

    def _graph_req(self) -> CalibrateChainRequest:
        return CalibrateChainRequest(
            transcript="[user] hi there",
            speaker_id="speaker0",
            graph={"session_id": "calib", "timestamp": "2026-01-01T00:00:00Z"},
        )

    @pytest.mark.parametrize("use_case", ["enrich", "plausibility"])
    def test_cloud_gated_step_never_reached_is_a_400(self, use_case):
        """With cloud egress refused the anonymize/enrich stages are skipped,
        so the declared stop never records. The endpoint says so."""
        state = self._state_with_chain(phases_that_run=["local_extract"])
        with pytest.raises(HTTPException) as exc:
            calibrate_chain(state, use_case, self._graph_req())
        assert exc.value.status_code == 400
        assert _CHAIN[use_case].stop in exc.value.detail

    def test_detail_names_the_steps_that_did_run(self):
        state = self._state_with_chain(phases_that_run=["local_extract", "anonymize"])
        with pytest.raises(HTTPException) as exc:
            calibrate_chain(state, "enrich", self._graph_req())
        assert "local_extract" in exc.value.detail
        assert "anonymize" in exc.value.detail

    def test_detail_reports_the_cloud_verdict_from_the_pipeline_config(self):
        """The verdict is read from the ExtractionConfig the chain itself
        runs on, through the shared admission component — not re-derived."""
        from paramem.graph.extraction_pipeline import ExtractionConfig

        state = self._state_with_chain(phases_that_run=["local_extract"])
        state["consolidation_loop"].extraction.config = ExtractionConfig(
            cloud_enabled=False, enrichment_provider="anthropic", scrub=frozenset()
        )
        with pytest.raises(HTTPException) as exc:
            calibrate_chain(state, "enrich", self._graph_req())
        assert "Cloud egress is refused" in exc.value.detail
        assert "cloud.enabled is off" in exc.value.detail

    def test_reached_step_returns_normally(self):
        state = self._state_with_chain(phases_that_run=["local_extract", "anonymize"])
        result = calibrate_chain(state, "anonymize", self._graph_req())
        assert result["stage"] == "anonymize"


class TestChainProductionParity:
    """The properties that make a calibration run 1:1 with production."""

    def test_plausibility_inspects_the_de_anonymized_judge(self):
        """Regression guard. The standalone endpoint judged whatever fact
        list the client posted, and the client posted ANONYMIZED facts —
        while production feeds this judge DE-ANONYMIZED ones. Running the
        real chain and stopping at ``deanon_plausibility`` is what closes
        that gap; a declaration pointing anywhere else reopens it."""
        assert _CHAIN["plausibility"].stop == "deanon_plausibility"

    def test_no_use_case_reaches_a_step_primitive(self):
        """The handler must route through ExtractionPipeline only. A
        dispatch calling anonymize()/judge_plausibility()/
        request_enrichment() directly is the standalone shape this
        unification removed."""
        source = inspect.getsource(calibrate_chain)
        for primitive in ("anonymize(", "judge_plausibility(", "request_enrichment("):
            assert primitive not in source

    def test_prompt_variant_content_reaches_the_loader(self, tmp_path):
        """An operator variant is injected through prompt_overrides, so the
        chain's own ``_load_prompt`` call returns the variant's CONTENT —
        the same mechanism for every step, no per-endpoint filename knob."""
        (tmp_path / "prompts").mkdir()
        (tmp_path / "prompts" / "my_extraction.txt").write_text("VARIANT BODY {transcript}")

        state = _state_enabled()
        state["config"].paths = PathsConfig(calibration=tmp_path)
        seen: dict = {}

        def _capture(*_args, **_kwargs):
            seen["loaded"] = _load_prompt("extraction.txt", required=True)

        state["consolidation_loop"].extraction.run.side_effect = _chain_side_effect(
            "local_extract", _capture
        )
        req = CalibrateChainRequest(
            transcript="[user] hi there",
            speaker_id="speaker0",
            prompt_variants={"extraction.txt": "my_extraction.txt"},
        )
        calibrate_chain(state, "extract", req)
        assert seen["loaded"] == "VARIANT BODY {transcript}"


class TestCalibrateNormalize:
    """Tests for CalibrateNormalizeRequest validation and calibrate_normalize."""

    def test_disabled_404(self):
        req = CalibrateNormalizeRequest(relations=[])
        with pytest.raises(HTTPException) as exc:
            calibrate_normalize(_state_disabled(), req)
        assert exc.value.status_code == 404

    def test_consolidating_503(self):
        state = _state_enabled()
        state["consolidating"] = True
        req = CalibrateNormalizeRequest(relations=[])
        with pytest.raises(HTTPException) as exc:
            calibrate_normalize(state, req)
        assert exc.value.status_code == 503

    def test_neither_relations_nor_snapshot_400(self, tmp_path):
        """Providing neither relations nor snapshot_path raises 400."""
        state = _state_enabled()
        req = CalibrateNormalizeRequest(
            relations=None,
            snapshot_path=None,
        )
        with pytest.raises(HTTPException) as exc:
            calibrate_normalize(state, req)
        assert exc.value.status_code == 400
        assert "exactly one" in exc.value.detail.lower()

    def test_both_relations_and_snapshot_400(self, tmp_path):
        """Providing both relations and snapshot_path raises 400."""
        state = _state_enabled()
        req = CalibrateNormalizeRequest(
            relations=[{"subject": "A", "predicate": "p", "object": "B"}],
            snapshot_path="/some/path.json",
        )
        with pytest.raises(HTTPException) as exc:
            calibrate_normalize(state, req)
        assert exc.value.status_code == 400
        assert "exactly one" in exc.value.detail.lower()

    def test_snapshot_node_link_flattening(self, tmp_path):
        """Snapshot node-link edges are flattened into the relation set the
        production pass is seeded with — edges with no predicate are skipped."""

        snap = {
            "nodes": [{"id": "Alex"}, {"id": "Acme"}],
            "links": [
                {"source": "Alex", "target": "Acme", "predicate": "works_for"},
                {"source": "Alex", "target": "Acme", "predicate": "employed_by"},
                {"source": "Alex", "target": "Acme"},  # missing predicate — skip
            ],
        }
        snap_path = tmp_path / "graph_merged_snapshot.json"
        snap_path.write_text(json.dumps(snap), encoding="utf-8")

        state = _state_enabled()
        refiner = _stub_refiner()
        state["consolidation_loop"].build_tier_refiner.return_value = refiner

        result = calibrate_normalize(
            state,
            CalibrateNormalizeRequest(snapshot_path=str(snap_path)),
        )

        assert result["stage"] == "normalize"
        assert result["parsed"]["input_count"] == 2
        assert isinstance(result["raw_output"], str)

    def test_runs_the_production_tier_pass(self):
        """The endpoint reaches the production pass through the loop's own
        construction site, seeded with a throwaway merger holding the
        injected relations — never a second normalization implementation.

        Mutation: have the handler call ``normalize_predicates`` directly
        again -> ``build_tier_refiner`` is never called and this fails.
        """
        state = _state_enabled()
        refiner = _stub_refiner()
        state["consolidation_loop"].build_tier_refiner.return_value = refiner

        result = calibrate_normalize(
            state,
            CalibrateNormalizeRequest(
                relations=[
                    {"subject": "Alex", "predicate": "works_for", "object": "Acme"},
                    {"subject": "Alex", "predicate": "employed_by", "object": "Acme"},
                ]
            ),
        )

        builder = state["consolidation_loop"].build_tier_refiner
        assert builder.call_count == 1
        seeded = builder.call_args.args[0]
        # The merger handed to the production builder carries the injected
        # relations, and it is NOT the loop's live merger. GraphMerger folds
        # predicate identity through canonical() (mode="full") at insertion,
        # which collapses "_" to a space, so the seeded triples carry the
        # space-form predicate surface, not the raw underscore input.
        assert seeded is not state["consolidation_loop"].merger
        assert {t[1] for t in seeded.get_all_triples()} == {"works for", "employed by"}
        assert refiner.run_normalization.call_count == 1
        # The pass's own diagnostics are reported verbatim.
        assert result["parsed"]["groups_examined"] == 1
        assert result["parsed"]["input_count"] == 2

    def test_no_reimplementation_of_the_survivor_rule(self):
        """The handler must not re-derive which predicate survives — that
        rule lives in the tier pass (highest reinforcement_count) and a
        second copy is exactly the drift this unification removed."""
        import ast

        tree = ast.parse(inspect.getsource(calibrate_normalize).lstrip())
        # Prose describing the production rule is fine; executing it is not,
        # so strip docstrings and comments by reading identifiers only.
        names = {
            node.id if isinstance(node, ast.Name) else node.attr
            for node in ast.walk(tree)
            if isinstance(node, (ast.Name, ast.Attribute))
        }
        assert "normalize_predicates" not in names
        assert "reinforcement_count" not in names


def _anonymized_contract(**overrides):
    """A canned ``AnonymizedContract`` for mocking
    ``paramem.cloud.anonymize.anonymize`` — the shared primitive
    ``calibrate_anonymize_facts`` and
    ``paramem.training.graph_enrich.enrich_graph`` both call.
    """
    from paramem.cloud.anonymize import AnonymizedContract

    fields = dict(
        status="ok",
        forward={"Alex": "Person_1"},
        reverse={"Person_1": "Alex"},
        anon_transcript="",
        declared=frozenset({"Person_1"}),
        norm_stats={"inverted": 0, "dropped": 0},
        rekey_dropped=0,
        raw='{"mapping": {"Alex": "Person_1"}}',
        failure=None,
        facts=[],
        slices=1,
        slices_failed=0,
    )
    fields.update(overrides)
    return AnonymizedContract(**fields)


def _state_with_extraction_config(*, scrub=None, anonymize_token_envelope=8192):
    """``_state_enabled()`` plus a real-shaped ``loop.extraction.config``
    — ``calibrate_anonymize_facts`` reads ``scrub``/``anonymize_token_envelope``
    from there, the same object ``graph_enrich.enrich_graph`` reads in
    production, never from the request.
    """
    state = _state_enabled()
    ext_cfg = SimpleNamespace(
        scrub=scrub if scrub is not None else {"person name"},
        anonymize_token_envelope=anonymize_token_envelope,
    )
    state["consolidation_loop"].extraction.config = ext_cfg
    state["consolidation_loop"].model = state["model"]
    state["consolidation_loop"].tokenizer = state["tokenizer"]
    return state


class TestCalibrateAnonymizeFacts:
    """Tests for CalibrateAnonymizeFactsRequest validation and
    calibrate_anonymize_facts — the graph-tier facts-only anonymize
    calibration stage (open nit n3)."""

    def test_disabled_404(self):
        req = CalibrateAnonymizeFactsRequest(facts=[])
        with pytest.raises(HTTPException) as exc:
            calibrate_anonymize_facts(_state_disabled(), req)
        assert exc.value.status_code == 404

    def test_consolidating_503(self):
        state = _state_with_extraction_config()
        state["consolidating"] = True
        req = CalibrateAnonymizeFactsRequest(facts=[])
        with pytest.raises(HTTPException) as exc:
            calibrate_anonymize_facts(state, req)
        assert exc.value.status_code == 503

    def test_neither_facts_nor_snapshot_400(self):
        state = _state_with_extraction_config()
        req = CalibrateAnonymizeFactsRequest(facts=None, snapshot_path=None)
        with pytest.raises(HTTPException) as exc:
            calibrate_anonymize_facts(state, req)
        assert exc.value.status_code == 400
        assert "exactly one" in exc.value.detail.lower()

    def test_both_facts_and_snapshot_400(self):
        state = _state_with_extraction_config()
        req = CalibrateAnonymizeFactsRequest(
            facts=[{"subject": "Alex", "predicate": "p", "object": "Acme"}],
            snapshot_path="/some/path.json",
        )
        with pytest.raises(HTTPException) as exc:
            calibrate_anonymize_facts(state, req)
        assert exc.value.status_code == 400
        assert "exactly one" in exc.value.detail.lower()

    def test_empty_facts_list_400(self):
        state = _state_with_extraction_config()
        req = CalibrateAnonymizeFactsRequest(facts=[])
        with pytest.raises(HTTPException) as exc:
            calibrate_anonymize_facts(state, req)
        assert exc.value.status_code == 400
        assert "no facts" in exc.value.detail.lower()

    def test_snapshot_node_link_flattening(self, tmp_path):
        """Snapshot node-link edges are flattened into the fact list the
        anonymize call is seeded with — reuses the SAME reader
        calibrate_normalize uses (_relations_from_snapshot); edges with
        no predicate are skipped."""
        snap = {
            "nodes": [{"id": "Alex"}, {"id": "Acme"}],
            "links": [
                {"source": "Alex", "target": "Acme", "predicate": "works_for"},
                {"source": "Alex", "target": "Acme"},  # missing predicate — skip
            ],
        }
        snap_path = tmp_path / "graph_merged_snapshot.json"
        snap_path.write_text(json.dumps(snap), encoding="utf-8")

        state = _state_with_extraction_config()
        with patch(
            "paramem.cloud.anonymize.anonymize",
            return_value=_anonymized_contract(),
        ) as mocked:
            result = calibrate_anonymize_facts(
                state, CalibrateAnonymizeFactsRequest(snapshot_path=str(snap_path))
            )

        assert result["stage"] == "anonymize_facts"
        facts_arg = mocked.call_args.args[0]
        assert len(facts_arg) == 1
        assert facts_arg[0]["predicate"] == "works_for"

    def test_runs_the_shared_anonymize_primitive_with_graph_tier_shape(self):
        """The handler calls the SAME ``anonymize()`` primitive
        ``graph_enrich.enrich_graph`` calls per chunk, with the graph-tier
        call shape: ``transcript=""``, ``identity_domain`` derived from
        the facts' own subject/object endpoints, ``scrub``/
        ``token_envelope`` from ``loop.extraction.config`` — never a
        request override.

        Mutation: have the handler re-implement anonymization instead of
        calling the shared primitive -> ``mocked`` is never called and
        this fails.
        """
        state = _state_with_extraction_config(
            scrub={"person name", "email address"}, anonymize_token_envelope=4321
        )
        facts = [
            {"subject": "Alex", "predicate": "works_at", "object": "Acme"},
            {"subject": "Riley", "predicate": "knows", "object": "Alex"},
        ]
        with patch(
            "paramem.cloud.anonymize.anonymize",
            return_value=_anonymized_contract(),
        ) as mocked:
            result = calibrate_anonymize_facts(state, CalibrateAnonymizeFactsRequest(facts=facts))

        assert mocked.call_count == 1
        call = mocked.call_args
        assert call.args[0] == facts
        assert call.kwargs["transcript"] == ""
        assert call.kwargs["scrub"] == {"person name", "email address"}
        assert call.kwargs["token_envelope"] == 4321
        assert set(call.kwargs["identity_domain"]) == {"Alex", "Acme", "Riley"}
        assert result["stage"] == "anonymize_facts"
        assert result["parsed"]["mapping"] == {"Alex": "Person_1"}

    def test_no_reimplementation_of_the_anonymize_chain(self):
        """The handler must not re-derive mapping/reconciliation logic —
        that lives in paramem.cloud.anonymize.anonymize, and a second
        copy here is exactly the drift the shared primitive exists to
        prevent."""
        import ast

        tree = ast.parse(inspect.getsource(calibrate_anonymize_facts).lstrip())
        names = {
            node.id if isinstance(node, ast.Name) else node.attr
            for node in ast.walk(tree)
            if isinstance(node, (ast.Name, ast.Attribute))
        }
        assert "anonymize" in names  # calls the shared primitive
        assert "anonymize_transcript" not in names  # never bypasses it
        assert "_normalize_anonymization_mapping" not in names
        assert "_build_anonymization_mapping" not in names

    def test_anonymize_phase_recorded_in_response(self):
        """``anonymize()`` opens no phase-trace scope itself (paramem.cloud
        must not import paramem.graph); the handler opens
        phase_trace("anonymize") around the call so prompt provenance and
        the phases list are populated, the same way the session-tier
        anonymize stage body already does."""
        state = _state_with_extraction_config()
        with patch(
            "paramem.cloud.anonymize.anonymize",
            return_value=_anonymized_contract(),
        ):
            result = calibrate_anonymize_facts(
                state,
                CalibrateAnonymizeFactsRequest(
                    facts=[{"subject": "Alex", "predicate": "works_at", "object": "Acme"}]
                ),
            )

        assert any(p["name"] == "anonymize" for p in result["phases"])

    def test_prompt_variant_override_reaches_the_anonymize_facts_template(self, tmp_path):
        """A ``prompt_variants`` override for ``anonymization_facts.txt``
        is honoured — the same mechanism every other stage uses, proving
        this stage's few-shots are tunable (the gap this stage closes)."""
        from paramem.server.config import PathsConfig

        variant_dir = tmp_path / "prompts"
        variant_dir.mkdir()
        (variant_dir / "calib_anonymization_facts.txt").write_text(
            "SENTINEL {scrub_categories} {facts_json}", encoding="utf-8"
        )

        state = _state_with_extraction_config()
        state["config"].paths = PathsConfig(calibration=tmp_path)

        captured_templates: list[str] = []

        def _capture(*args, **kwargs):
            captured_templates.append(kwargs.get("user_prompt_template", ""))
            return _anonymized_contract()

        with patch("paramem.cloud.anonymize.anonymize", side_effect=_capture):
            calibrate_anonymize_facts(
                state,
                CalibrateAnonymizeFactsRequest(
                    facts=[{"subject": "Alex", "predicate": "works_at", "object": "Acme"}],
                    prompt_variants={"anonymization_facts.txt": "calib_anonymization_facts.txt"},
                ),
            )

        assert captured_templates
        assert captured_templates[0].startswith("SENTINEL")


class TestEffectiveParamsSeed:
    """Verify seed threads through _effective_params for all three local stages."""

    def test_extract_supports_seed(self):
        params = CalibrateParams(seed=42)
        result = _effective_params(params, supports_seed=True)
        assert result["seed"] == 42

    def test_anonymize_supports_seed(self):
        params = CalibrateParams(seed=42)
        result = _effective_params(params, supports_seed=True)
        assert result["seed"] == 42

    def test_plausibility_supports_seed(self):
        params = CalibrateParams(seed=42)
        result = _effective_params(params, supports_seed=True)
        assert result["seed"] == 42

    def test_seed_none_when_supports_seed_false(self):
        """Cloud stages report seed=null even when caller sends a seed."""
        params = CalibrateParams(seed=99)
        result = _effective_params(params, supports_seed=False)
        assert result["seed"] is None

    def test_seed_none_when_not_set(self):
        params = CalibrateParams()
        result = _effective_params(params, supports_seed=True)
        assert result["seed"] is None


class TestRunCalibrationResponseEnvelope:
    """Verify _run_calibration's inlined response assembly emits the
        expected uniform envelope.

    These tests exercise the envelope through a trivial ``guard``/
        ``dispatch`` pair so the assembly logic is pinned independent of any
        one handler's real behavior. The stand-in ``dispatch`` opens its own
        phase, exactly as the production paths the real handlers call do.
    """

    _REQUIRED_KEYS = {
        "stage",
        "prompts",
        "raw_output",
        "parsed",
        "n_input_tokens",
        "n_output_tokens",
        "wall_clock_seconds",
        "model",
        "params_effective",
        "vram_before",
        "vram_after",
        "artifact_dir",
    }

    def _run(
        self,
        *,
        stage: str = "anonymize",
        raw_output="some output",
        parsed: dict | None = None,
        params: CalibrateParams | None = None,
        supports_seed: bool = True,
        state: dict | None = None,
    ) -> dict:
        state = state if state is not None else _state_enabled()
        params = params if params is not None else CalibrateParams()
        parsed = parsed if parsed is not None else {}

        def guard() -> None:
            return None

        def dispatch() -> tuple:
            # A production path opens its own phase onto the trace
            # _run_calibration holds open; the substrate never synthesises
            # one. This stand-in does the same.
            with phase_trace("anonymize"):
                return raw_output, parsed

        return _run_calibration(
            stage=stage,
            guard=guard,
            dispatch=dispatch,
            input_prompt_phase="anonymize",
            state=state,
            params=params,
            supports_seed=supports_seed,
        )

    def test_all_required_keys_present(self):
        """All 11 uniform keys appear in the output."""
        result = self._run(raw_output="some output", parsed={"mapping": {}})
        assert self._REQUIRED_KEYS.issubset(result.keys())

    def test_stage_field_matches_argument(self):
        result = self._run(stage="plausibility")
        assert result["stage"] == "plausibility"

    def test_wall_clock_seconds_present_and_nonnegative(self):
        result = self._run()
        assert result["wall_clock_seconds"] >= 0

    def test_vram_keys_present(self):
        result = self._run()
        assert "vram_before" in result
        assert "vram_after" in result

    def test_n_output_tokens_minus1_for_empty_raw_output(self):
        """Empty raw_output → n_output_tokens == -1."""
        result = self._run(raw_output="")
        assert result["n_output_tokens"] == -1

    def test_n_tokens_minus1_when_no_tokenizer(self):
        """No tokenizer in state → both fields stay -1 (U1's explicit
        decision: the ``if tokenizer`` / ``if tokenizer and count_str``
        sentinel guards at calibrate.py:668-670 are unchanged).  The real
        endpoint path 503s on a missing tokenizer before reaching this
        guard (``_preflight`` requires one) — patched to a no-op here to
        isolate the guard's own behavior rather than re-test ``_preflight``."""
        state = _state_enabled()
        state["tokenizer"] = None
        with patch("paramem.server.calibrate._preflight"):
            result = self._run(state=state)
        assert result["n_input_tokens"] == -1
        assert result["n_output_tokens"] == -1

    def test_n_input_tokens_uses_fallback_estimate_when_tokenizer_raises(self):
        """A tokenizer that raises on call no longer collapses to -1 (U1):
        ``estimate_tokens`` falls back to a conservative words-based
        estimate instead — the raising-tokenizer case item 47 names as the
        one thing this migration actually changes."""
        from paramem.graph.phase_trace import record_prompt

        state = _state_enabled()
        state["tokenizer"].side_effect = RuntimeError("tokenizer boom")

        def dispatch() -> tuple:
            with phase_trace("anonymize"):
                record_prompt(
                    path="anonymization.txt",
                    content="one two three four five six seven eight",
                )
            return "some raw output text", {"mapping": {}}

        result = _run_calibration(
            stage="anonymize",
            guard=lambda: None,
            dispatch=dispatch,
            input_prompt_phase="anonymize",
            state=state,
            params=CalibrateParams(),
            supports_seed=True,
        )
        assert result["n_input_tokens"] != -1
        assert result["n_input_tokens"] > 0

    def test_phases_present_from_trace(self):
        """``phases`` is populated from the phase-trace records the
        ``entry="standalone"`` scope opens around ``dispatch`` — no
        caller passes it in directly."""
        result = self._run()
        assert "phases" in result
        assert any(p["name"] == "anonymize" for p in result["phases"])

    def test_supports_seed_false_nulls_seed(self):
        """supports_seed=False causes seed=null in params_effective."""
        result = self._run(params=CalibrateParams(seed=99), supports_seed=False)
        assert result["params_effective"]["seed"] is None


class TestExtractionPipelineKwargsSeed:
    """Verify ExtractionPipeline.kwargs passes seed through."""

    def test_seed_forwarded(self):
        from unittest.mock import MagicMock

        from paramem.graph.extraction_pipeline import ExtractionConfig

        pipeline = ExtractionPipeline(
            MagicMock(), MagicMock(), config=ExtractionConfig(scrub=set())
        )
        kwargs = pipeline.kwargs(seed=7, speaker_id="speaker0")
        assert kwargs["seed"] == 7

    def test_seed_none_by_default(self):
        from unittest.mock import MagicMock

        from paramem.graph.extraction_pipeline import ExtractionConfig

        pipeline = ExtractionPipeline(
            MagicMock(), MagicMock(), config=ExtractionConfig(scrub=set())
        )
        kwargs = pipeline.kwargs(speaker_id="speaker0")
        assert kwargs["seed"] is None


class TestCalibrateName:
    """Tests for the /calibrate/name stage."""

    def test_disabled_404(self):
        """calibrate_name returns 404 when calibrate_endpoint_enabled is False."""
        req = CalibrateNameRequest(
            turns=[{"role": "user", "text": "Hi, I'm Alex."}],
        )
        with pytest.raises(HTTPException) as exc:
            calibrate_name(_state_disabled(), req)
        assert exc.value.status_code == 404

    def test_consolidating_503(self):
        """calibrate_name returns 503 when a consolidation cycle is in progress."""
        state = _state_enabled()
        state["consolidating"] = True
        req = CalibrateNameRequest(turns=[{"role": "user", "text": "I'm Taylor."}])
        with pytest.raises(HTTPException) as exc:
            calibrate_name(state, req)
        assert exc.value.status_code == 503

    def test_model_missing_503(self):
        """calibrate_name returns 503 in cloud-only / defer-model mode."""
        state = _state_enabled()
        state["model"] = None
        req = CalibrateNameRequest(turns=[{"role": "user", "text": "I'm Jordan."}])
        with pytest.raises(HTTPException) as exc:
            calibrate_name(state, req)
        assert exc.value.status_code == 503

    def test_missing_prompt_variant_raises_400(self, tmp_path):
        """A named variant absent from the calibration prompt directory is
        refused before any model call — never a silent fall-back to the
        shipped prompt of the same name."""
        state = _state_enabled()
        state["config"].paths = PathsConfig(calibration=tmp_path)
        req = CalibrateNameRequest(
            turns=[{"role": "user", "text": "Hi, I'm Riley."}],
            prompt_variants={"name_extraction.txt": "absent_variant.txt"},
        )
        with pytest.raises(HTTPException) as exc:
            calibrate_name(state, req)
        assert exc.value.status_code == 400
        assert "variant not found" in exc.value.detail.lower()

    def test_returns_uniform_shape(self, tmp_path):
        """calibrate_name returns the uniform calibration response shape."""
        import unittest.mock as _mock

        state = _state_enabled()
        variants = tmp_path / "prompts"
        variants.mkdir()
        (variants / "my_name_extraction.txt").write_text(
            "Extract name from:\n{transcript}\nAnswer:"
        )
        state["config"].paths = PathsConfig(calibration=tmp_path)

        req = CalibrateNameRequest(
            turns=[{"role": "user", "text": "Hi, I'm Alex."}],
            prompt_variants={"name_extraction.txt": "my_name_extraction.txt"},
        )

        # generate_answer is a lazy local import inside extract_name_via_llm;
        # patch at the definition site so it's intercepted wherever imported.
        with _mock.patch(
            "paramem.evaluation.recall.generate_answer",
            return_value="Alex",
        ):
            result = calibrate_name(state, req)

        # Uniform shape keys.
        assert result["stage"] == "name"
        assert "prompts" in result
        assert "raw_output" in result
        assert "parsed" in result
        assert "params_effective" in result
        assert "wall_clock_seconds" in result
        assert "n_input_tokens" in result
        assert "n_output_tokens" in result
        # parsed carries the extracted name.
        assert "name" in result["parsed"]
        # Two prompt entries (system + user).
        assert len(result["prompts"]) == 2

        # Reported provenance is sourced from the real phase-trace record
        # (populated by `_load_prompt`'s own `record_prompt` call inside
        # `extract_name_via_llm`), never a hand-built literal: the user
        # prompt was overridden by a variant and the system prompt was not,
        # so the two entries must report exactly that split, each with the
        # sha of the content actually used.
        from paramem.graph.prompts import _load_prompt

        by_path = {p["path"]: p for p in result["prompts"]}
        variant = (variants / "my_name_extraction.txt").read_text()
        shipped_sys = _load_prompt("name_extraction_system.txt", required=True)
        assert (
            by_path["<override:name_extraction.txt>"]["sha"]
            == (hashlib.sha256(variant.encode("utf-8")).hexdigest()[:12])
        )
        sys_entry = next(p for p in result["prompts"] if p["path"].endswith("_system.txt"))
        assert sys_entry["sha"] == hashlib.sha256(shipped_sys.encode("utf-8")).hexdigest()[:12]
        assert result["parsed"]["name"] == "Alex"

    def test_filename_override_used_at_execution(self, tmp_path):
        """The variant is what the model actually sees, not a display-only
        value: what the provenance block reports and what reaches the model
        are the same resolution, because both come from the one loader."""
        import unittest.mock as _mock

        state = _state_enabled()
        variants = tmp_path / "prompts"
        variants.mkdir()
        (variants / "my_custom_system.txt").write_text("custom system")
        (variants / "my_custom_user.txt").write_text("custom user {transcript}")
        state["config"].paths = PathsConfig(calibration=tmp_path)

        req = CalibrateNameRequest(
            turns=[{"role": "user", "text": "I'm Alex."}],
            prompt_variants={
                "name_extraction.txt": "my_custom_user.txt",
                "name_extraction_system.txt": "my_custom_system.txt",
            },
        )

        captured_messages: list[list[dict]] = []

        def _capture_template(messages, **kw):
            captured_messages.append(messages)
            return "formatted-prompt"

        state["tokenizer"].apply_chat_template.side_effect = _capture_template

        with _mock.patch(
            "paramem.evaluation.recall.generate_answer",
            return_value="Alex",
        ):
            result = calibrate_name(state, req)

        # The provenance block must report the variants as overrides —
        # _load_prompt records "<override:NAME>" for a prompt_overrides hit,
        # so what is reported is what the model was actually given.
        paths_reported = {p["path"] for p in result["prompts"]}
        assert "<override:name_extraction_system.txt>" in paths_reported
        assert "<override:name_extraction.txt>" in paths_reported

        # The model must have received the OVERRIDE content, not the default.
        assert captured_messages, "apply_chat_template was never called"
        msgs = captured_messages[0]
        sys_content = next(m["content"] for m in msgs if m["role"] == "system")
        user_content = next(m["content"] for m in msgs if m["role"] == "user")
        assert sys_content == "custom system", (
            f"Model received default system prompt instead of override: {sys_content!r}"
        )
        assert "custom user" in user_content, (
            f"Model received default user prompt instead of override: {user_content!r}"
        )


class TestExtractNameViaLlmUserTurnFilter:
    """Unit tests for extract_name_via_llm's user-turn filtering.

    These tests mock generate_answer and verify that:
    1. Assistant turns are excluded from the transcript passed to the model
       when user_turns_only=True (fixes the salutation leak).
    2. Post-filters still apply (NONE sentinel, length, word-count).
    3. When user_turns_only=False, assistant turns ARE included.
    """

    def _call(self, turns, *, user_turns_only=True, model_output="NONE"):
        from unittest.mock import MagicMock, patch

        from paramem.graph.name_extraction import extract_name_via_llm

        model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "formatted-prompt"

        # generate_answer and _load_prompt are imported inside extract_name_via_llm
        # at call time; patch at their definition sites so the interception lands.
        def _fake_generate(m, t, prompt, **kw):
            return model_output

        with (
            patch("paramem.evaluation.recall.generate_answer", side_effect=_fake_generate),
            patch(
                "paramem.graph.prompts._load_prompt",
                side_effect=lambda filename, *args, **kw: "{transcript}",
            ),
        ):
            name, _raw = extract_name_via_llm(
                turns, model, tokenizer, user_turns_only=user_turns_only
            )

        # Reconstruct what was passed to apply_chat_template as messages.
        # We inspect tokenizer.apply_chat_template call args for the messages list.
        call_args = tokenizer.apply_chat_template.call_args
        messages = call_args[0][0]  # first positional arg
        # The user message content is the rendered template = transcript portion.
        user_msg = next(m["content"] for m in messages if m["role"] == "user")
        return name, user_msg

    def test_assistant_turns_excluded_by_default(self):
        """When user_turns_only=True, assistant turns must not appear in the transcript."""
        turns = [
            {"role": "user", "text": "Hi there."},
            {"role": "assistant", "text": "Good evening, user."},
            {"role": "user", "text": "My name is Alex."},
        ]
        _, user_msg = self._call(turns, model_output="Alex")
        assert "assistant" not in user_msg, (
            "Assistant turn must be excluded from the transcript when user_turns_only=True."
        )
        assert "Good evening" not in user_msg

    def test_user_turns_present_in_transcript(self):
        """User turns must appear in the transcript."""
        turns = [
            {"role": "user", "text": "My name is Riley."},
        ]
        _, user_msg = self._call(turns, model_output="Riley")
        assert "My name is Riley" in user_msg

    def test_assistant_included_when_flag_false(self):
        """When user_turns_only=False, assistant turns ARE included."""
        turns = [
            {"role": "user", "text": "Hi."},
            {"role": "assistant", "text": "Hello user."},
        ]
        _, user_msg = self._call(turns, user_turns_only=False, model_output="NONE")
        assert "assistant" in user_msg or "Hello user" in user_msg

    def test_none_sentinel_returns_none(self):
        """Model returning NONE sentinel → None."""
        turns = [{"role": "user", "text": "Play music."}]
        result, _ = self._call(turns, model_output="NONE")
        assert result is None

    def test_empty_output_returns_none(self):
        """Empty model output → None."""
        turns = [{"role": "user", "text": "Play music."}]
        result, _ = self._call(turns, model_output="")
        assert result is None

    def test_too_long_output_returns_none(self):
        """Output longer than 30 chars → None (post-filter)."""
        turns = [{"role": "user", "text": "x"}]
        result, _ = self._call(turns, model_output="A" * 31)
        assert result is None

    def test_too_many_words_returns_none(self):
        """Output with >3 words → None (post-filter)."""
        turns = [{"role": "user", "text": "x"}]
        result, _ = self._call(turns, model_output="one two three four")
        assert result is None

    def test_valid_name_returned(self):
        """Single-word name under 30 chars is returned as-is."""
        turns = [{"role": "user", "text": "Hi, I'm Morgan."}]
        result, _ = self._call(turns, model_output="Morgan")
        assert result == "Morgan"

    def test_valid_two_word_name_returned(self):
        """Two-word name is within post-filter bounds and returned."""
        turns = [{"role": "user", "text": "I'm Casey Rivera."}]
        result, _ = self._call(turns, model_output="Casey Rivera")
        assert result == "Casey Rivera"

    def test_raw_output_populated_even_when_filtered(self):
        """raw_output is the verbatim model string, populated even when name is None."""
        from unittest.mock import MagicMock, patch

        from paramem.graph.name_extraction import extract_name_via_llm

        model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "formatted-prompt"

        with (
            patch(
                "paramem.evaluation.recall.generate_answer",
                return_value="too many words here to pass filter",
            ),
            patch(
                "paramem.graph.prompts._load_prompt",
                side_effect=lambda filename, *args, **kw: "{transcript}",
            ),
        ):
            name, raw_output = extract_name_via_llm(
                [{"role": "user", "text": "x"}], model, tokenizer
            )

        assert name is None
        # raw_output carries the stripped model string before the word-count filter.
        assert "too many words" in raw_output

    def test_required_raises_file_not_found(self, tmp_path):
        """_load_prompt(required=True) raises FileNotFoundError when file is absent everywhere.

        We test this via the prompts module directly (with a nonexistent filename) rather than
        via extract_name_via_llm, because the production prompt files exist in _DEFAULT_PROMPT_DIR
        and would be found as fallback even when prompts_dir has no files.
        """
        from paramem.graph.prompts import _load_prompt

        with pytest.raises(FileNotFoundError) as exc_info:
            _load_prompt("no_such_prompt_xyz.txt", prompts_dir=tmp_path, required=True)
        assert "no_such_prompt_xyz.txt" in str(exc_info.value)
        assert "Searched" in str(exc_info.value)
