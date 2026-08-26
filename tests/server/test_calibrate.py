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
from pathlib import Path
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
from paramem.graph.prompts import _load_prompt, prompt_overrides
from paramem.graph.schema import SessionGraph
from paramem.server import calibrate
from paramem.server.calibrate import (
    _CHAIN,
    CalibrateChainRequest,
    CalibrateNameRequest,
    CalibrateNormalizeRequest,
    CalibrateParams,
    CalibrateRespondRequest,
    _effective_params,
    preflight,
)
from paramem.server.chat_result import ChatResult
from paramem.server.config import CloudConfig, PathsConfig, SanitizationConfig
from paramem.server.inference import handle_chat
from paramem.utils.artifacts import calibration_run


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
        "memory_store": MagicMock(),
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
        model_config=SimpleNamespace(model_id="mistralai/Mistral-7B-Instruct-v0.3"),
    )
    return {
        "config": config,
        "consolidating": False,
        "model": _peft_model_mock(),
        "tokenizer": MagicMock(),
        "memory_store": MagicMock(),
        "consolidation_loop": MagicMock(),
    }


# ---------------------------------------------------------------------------
# Test-only composition helper — drives the real production pieces
# (``preflight`` -> ``build_spec`` -> ``run_stage``) in the boundary's own
# order, so behaviour is asserted against real code, never a synthetic
# double.  Confined to this test module.  ``build_spec`` is itself
# production code (``paramem/server/calibrate.py``), shared with the route
# handlers in ``app.py`` — there is exactly one declaration of each stage's
# shape (route path, input-prompt phase, seed support, params source:
# ``calibrate._CHAIN`` / ``calibrate._STANDALONE``).  The ``with
# calibration_run(...), prompt_overrides(...):`` pair below is a deliberate,
# small, test-only mirror of the two lines
# :func:`~paramem.server.app._run_calibration_sync` opens around this same
# ``run_stage`` call for a real dispatch — that function is the one
# production owner of the scope; this helper exists only so a test can
# drive ``run_stage`` without the FastAPI/executor/GPU-lock machinery
# around it.
# ---------------------------------------------------------------------------

_DEFAULT_TEST_ARTIFACT_ROOT = Path("/tmp/test-calibrate")


def _run_stage(state: dict, stage: str, req, *, artifact_dir: Path | None = None) -> dict:
    """Run one declared calibration stage exactly as a route would, minus
    the FastAPI/executor/GPU-lock machinery: ``preflight`` -> ``build_spec``
    -> ``run_stage``, inside the same ``calibration_run``/``prompt_overrides``
    scope the real dispatch opens.  ``stage`` is a key of ``calibrate._CHAIN``
    (the five chain use cases) or ``calibrate._STANDALONE`` (``normalize``,
    ``anonymize_facts``, ``name``, ``respond``)."""
    preflight(state)
    run_dir = artifact_dir if artifact_dir is not None else _DEFAULT_TEST_ARTIFACT_ROOT / stage
    spec = calibrate.build_spec(stage, state, req, run_id="test-run", artifact_dir=run_dir)
    with calibration_run(run_dir), prompt_overrides(spec.overrides):
        return calibrate.run_stage(spec, state)


def _run_chain(
    state: dict, use_case: str, req: CalibrateChainRequest, *, artifact_dir: Path | None = None
) -> dict:
    return _run_stage(state, use_case, req, artifact_dir=artifact_dir)


def _run_normalize(
    state: dict, req: CalibrateNormalizeRequest, *, artifact_dir: Path | None = None
) -> dict:
    return _run_stage(state, "normalize", req, artifact_dir=artifact_dir)


def _run_name(state: dict, req: CalibrateNameRequest, *, artifact_dir: Path | None = None) -> dict:
    return _run_stage(state, "name", req, artifact_dir=artifact_dir)


def _run_respond(
    state: dict, req: CalibrateRespondRequest, *, artifact_dir: Path | None = None
) -> dict:
    return _run_stage(state, "respond", req, artifact_dir=artifact_dir)


def _state_respond() -> dict:
    """``_state_enabled()`` plus the collaborators ``calibrate_respond``
    needs: a speaker store, session buffer, router, and memory store (each
    a permissive ``MagicMock``), and a ``text_lang_detection`` config
    namespace with detection disabled — ``resolve_text_language`` short-
    circuits on ``cfg.enabled`` before touching a fastText model, so no
    model file is required for these tests.
    """
    state = _state_enabled()
    store = MagicMock()
    store.get_name.return_value = "Alex"
    store.resolve_speaker_name.return_value = "Alex"
    state["speaker_store"] = store
    buffer = MagicMock()
    buffer.get_conversation_turns.return_value = []
    state["session_buffer"] = buffer
    state["router"] = MagicMock()
    state["memory_store"] = MagicMock()
    state["config"].text_lang_detection = SimpleNamespace(
        enabled=False, model_path="", confidence_threshold=0.5
    )
    return state


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
            preflight(_state_disabled())
        assert exc.value.status_code == 404
        assert "disabled" in exc.value.detail.lower()

    def test_503_when_model_missing(self):
        """No model handle -> 503, independently of ``_state["mode"]``.

        A running consolidation cycle is the arbitrator's own guard,
        answering 200 ``deferred_already_running`` — see
        ``TestConsolidationDispatchGuards`` in test_consolidate_dispatch.py.
        """
        state = _state_enabled()
        state["model"] = None
        with pytest.raises(HTTPException) as exc:
            preflight(state)
        assert exc.value.status_code == 503
        assert "local model" in exc.value.detail.lower()

    def test_503_when_tokenizer_missing(self):
        state = _state_enabled()
        state["tokenizer"] = None
        with pytest.raises(HTTPException) as exc:
            preflight(state)
        assert exc.value.status_code == 503
        assert "local model" in exc.value.detail.lower()

    def test_503_when_memory_store_missing(self):
        """No memory store handle -> 503 ``store_unavailable`` — the arm
        that keeps a calibration dispatch from being the first construction
        of the process-lifetime loop singleton with no store override."""
        state = _state_enabled()
        state["memory_store"] = None
        with pytest.raises(HTTPException) as exc:
            preflight(state)
        assert exc.value.status_code == 503
        assert "memory store" in exc.value.detail.lower()

    def test_passes_when_enabled_idle_loaded(self):
        # Should not raise.
        preflight(_state_enabled())


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
            _run_chain(_state_disabled(), use_case, self._req(use_case))
        assert exc.value.status_code == 404

    def test_empty_speaker_id_400(self, use_case):
        state = _state_enabled()
        with pytest.raises(HTTPException) as exc:
            _run_chain(state, use_case, self._req(use_case, speaker_id=""))
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
            _run_chain(state, use_case, req)
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
            _run_chain(state, use_case, req)
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
            _run_chain(state, use_case, req)
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
            result = _run_chain(
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
        _run_chain(state, use_case, CalibrateChainRequest(**fields))
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
        _run_chain(state, use_case, CalibrateChainRequest(**fields))
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
        _run_chain(state, "anonymize", req)
        assert isinstance(captured["seed"], SessionGraph)
        assert captured["seed"].session_id == "seeded"

    def test_transcript_use_case_seeds_nothing(self):
        captured: dict = {}
        state = self._capturing_state(captured)
        _run_chain(
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
        _run_chain(state, "extract", req)
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
        _run_chain(state, "plausibility", req)
        assert captured["stop"] == "deanon_plausibility"

    def test_invalid_stop_phase_400s_before_any_pipeline_call(self):
        """An unknown ``stop_phase`` posted to the one open-stop use case
        (``extract``) is rejected at guard time — before the extraction
        pipeline is ever invoked, not as a downstream ValueError surfaced
        as a 500."""
        state = self._capturing_state({})
        req = CalibrateChainRequest(
            transcript="[user] hi there",
            speaker_id="speaker0",
            stop_phase="not_a_real_phase",
        )
        with pytest.raises(HTTPException) as exc:
            _run_chain(state, "extract", req)
        assert exc.value.status_code == 400
        assert "not_a_real_phase" in exc.value.detail
        assert not state["consolidation_loop"].extraction.run.called

    def test_valid_but_non_firing_stop_phase_passes_guard(self):
        """A real member of PHASE_NAMES that this chain run never opens
        clears guard-time validation (it is a valid name, just not
        applicable) — the run still completes (post-200 there is no
        response left to fail), naming the gap in ``unreached_step``,
        proven here only by the pipeline having been called at all."""
        state = self._capturing_state({})
        state["consolidation_loop"].extraction.run.side_effect = _chain_side_effect("local_extract")
        req = CalibrateChainRequest(
            transcript="[user] hi there",
            speaker_id="speaker0",
            stop_phase="name_extract",
        )
        result = _run_chain(state, "extract", req)
        assert result["unreached_step"] is not None
        assert state["consolidation_loop"].extraction.run.called

    # Artifact-scope opening (``calibration_run``) and the response.json
    # write (``on_calibration_result``) are the executor envelope's job
    # (``app._run_calibration_sync``), not calibrate.py's — covered there.

    def test_dispatch_surfaces_focus_step_raw_output(self):
        """The inspected step's ``raw_output`` is surfaced at the response top
        level, read off the ``PhaseRecord`` object (not a dict).

        The dispatch must read ``record.raw_output`` directly off the
        ``PhaseRecord`` object.  The other dispatch mocks return a graph
        with no attached phases (``get_phases`` empty → ``record is None``
        → ``""``), so only a graph carrying a real focus-step record
        exercises it.
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
        result = _run_chain(
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
        _run_chain(
            state,
            "extract",
            CalibrateChainRequest(transcript="[user] hi there", speaker_id="speaker0"),
            artifact_dir=tmp_path / "artifacts" / "extract_1",
        )
        snaps = list((tmp_path / "artifacts").glob("extract_*/sessions/calib/graph_snapshot.json"))
        assert len(snaps) == 1

    def test_procedural_writes_the_procedural_snapshot(self, tmp_path):
        state = self._paths_state(tmp_path, "procedural_extract")
        _run_chain(
            state,
            "procedural",
            CalibrateChainRequest(transcript="[user] hi there", speaker_id="speaker0"),
            artifact_dir=tmp_path / "artifacts" / "procedural_1",
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
        _run_chain(state, "anonymize", req, artifact_dir=tmp_path / "artifacts" / "anonymize_1")
        assert not list((tmp_path / "artifacts").glob("**/*graph_snapshot.json"))


class TestDeclaredStepUnreached:
    """A calibration promises ONE step's output. When the configured chain
    cannot reach that step, the run still completes — post-200 there is no
    response left to fail — and the gap is reported as DATA:
    ``response.json``'s ``unreached_step`` field
    (``{declared_phase, phases_ran, detail}``), never an HTTP error.

    Mutation: drop the check in ``run_stage`` -> these fail, and a
    cloud-disabled server's ``/calibrate/enrich`` run reports
    ``unreached_step: None`` with silently empty provenance instead of
    naming the gap.
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
    def test_cloud_gated_step_never_reached_is_data_not_an_error(self, use_case):
        """With cloud egress refused the anonymize/enrich stages are skipped,
        so the declared stop never records. The run still completes, 200,
        and names the gap in ``unreached_step``."""
        state = self._state_with_chain(phases_that_run=["local_extract"])
        result = _run_chain(state, use_case, self._graph_req())
        assert result["unreached_step"] is not None
        assert result["unreached_step"]["declared_phase"] == _CHAIN[use_case].stop
        assert _CHAIN[use_case].stop in result["unreached_step"]["detail"]

    def test_detail_names_the_steps_that_did_run(self):
        state = self._state_with_chain(phases_that_run=["local_extract", "anonymize"])
        result = _run_chain(state, "enrich", self._graph_req())
        assert result["unreached_step"]["phases_ran"] == ["local_extract", "anonymize"]
        assert "local_extract" in result["unreached_step"]["detail"]
        assert "anonymize" in result["unreached_step"]["detail"]

    def test_detail_reports_the_cloud_verdict_from_the_pipeline_config(self):
        """The verdict is read from the ExtractionConfig the chain itself
        runs on, through the shared admission component — not re-derived."""
        from paramem.graph.extraction_pipeline import ExtractionConfig

        state = self._state_with_chain(phases_that_run=["local_extract"])
        state["consolidation_loop"].extraction.config = ExtractionConfig(
            cloud_enabled=False, enrichment_provider="anthropic", scrub_categories=()
        )
        result = _run_chain(state, "enrich", self._graph_req())
        detail = result["unreached_step"]["detail"]
        assert "Cloud egress is refused" in detail
        assert "cloud.enabled is off" in detail

    def test_reached_step_returns_normally(self):
        state = self._state_with_chain(phases_that_run=["local_extract", "anonymize"])
        result = _run_chain(state, "anonymize", self._graph_req())
        assert result["stage"] == "anonymize"
        assert result["unreached_step"] is None


class TestChainProductionParity:
    """The properties that make a calibration run 1:1 with production."""

    def test_plausibility_inspects_the_de_anonymized_judge(self):
        """The plausibility judge must inspect DE-ANONYMIZED facts, never
        the anonymized ones a client would post directly. Running the real
        chain and stopping at ``deanon_plausibility`` guarantees this; a
        declaration pointing anywhere else would feed the judge anonymized
        facts."""
        assert _CHAIN["plausibility"].stop == "deanon_plausibility"

    def test_no_use_case_reaches_a_step_primitive(self):
        """The handler must route through ExtractionPipeline only; it must
        never call anonymize()/judge_plausibility()/request_enrichment()
        directly."""
        source = inspect.getsource(calibrate.dispatch_chain)
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
            seen["loaded"] = _load_prompt("extraction.txt")

        state["consolidation_loop"].extraction.run.side_effect = _chain_side_effect(
            "local_extract", _capture
        )
        req = CalibrateChainRequest(
            transcript="[user] hi there",
            speaker_id="speaker0",
            prompt_variants={"extraction.txt": "my_extraction.txt"},
        )
        _run_chain(state, "extract", req)
        assert seen["loaded"] == "VARIANT BODY {transcript}"


class TestRelationsFromSnapshot:
    """Direct unit coverage for the shared snapshot-to-relations reader."""

    def test_omits_speaker_id_key_when_edge_carries_none(self, tmp_path):
        """An edge with no ``speaker_id`` (or an empty one) yields a dict
        with no ``speaker_id`` key at all — not ``""`` — so a consumer's
        ``.get("speaker_id", <placeholder>)`` default actually fires."""
        snap = {
            "nodes": [{"id": "Alex"}, {"id": "Acme"}, {"id": "Bo"}],
            "links": [
                {"source": "Alex", "target": "Acme", "predicate": "works_for"},
                {
                    "source": "Alex",
                    "target": "Bo",
                    "predicate": "knows",
                    "speaker_id": "",
                },
            ],
        }
        snap_path = tmp_path / "graph_merged_snapshot.json"
        snap_path.write_text(json.dumps(snap), encoding="utf-8")

        relations = calibrate._relations_from_snapshot(str(snap_path))

        assert len(relations) == 2
        assert "speaker_id" not in relations[0]
        assert "speaker_id" not in relations[1]

    def test_emits_speaker_id_when_edge_carries_one(self, tmp_path):
        """An edge that does carry a ``speaker_id`` still emits it verbatim."""
        snap = {
            "nodes": [{"id": "Alex"}, {"id": "Acme"}],
            "links": [
                {
                    "source": "Alex",
                    "target": "Acme",
                    "predicate": "employed_by",
                    "speaker_id": "speaker3",
                },
            ],
        }
        snap_path = tmp_path / "graph_merged_snapshot.json"
        snap_path.write_text(json.dumps(snap), encoding="utf-8")

        relations = calibrate._relations_from_snapshot(str(snap_path))

        assert relations[0]["speaker_id"] == "speaker3"


class TestCalibrateNormalize:
    """Tests for CalibrateNormalizeRequest validation and calibrate_normalize."""

    def test_disabled_404(self):
        req = CalibrateNormalizeRequest(relations=[])
        with pytest.raises(HTTPException) as exc:
            _run_normalize(_state_disabled(), req)
        assert exc.value.status_code == 404

    def test_neither_relations_nor_snapshot_400(self, tmp_path):
        """Providing neither relations nor snapshot_path raises 400."""
        state = _state_enabled()
        req = CalibrateNormalizeRequest(
            relations=None,
            snapshot_path=None,
        )
        with pytest.raises(HTTPException) as exc:
            _run_normalize(state, req)
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
            _run_normalize(state, req)
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

        result = _run_normalize(
            state,
            CalibrateNormalizeRequest(snapshot_path=str(snap_path)),
        )

        assert result["stage"] == "normalize"
        assert result["parsed"]["input_count"] == 2
        assert isinstance(result["raw_output"], str)

    def test_snapshot_edge_with_no_speaker_gets_structural_placeholder(self, tmp_path):
        """A snapshot edge carrying no ``speaker_id`` reaches the merger's
        ``Relation`` construction through the dispatcher's structural
        ``"speaker0"`` default — not an empty string baked in by the
        reader. Mutation: reverting ``_relations_from_snapshot`` to always
        emit ``speaker_id`` (even as ``""``) makes this fail, since ``""``
        is present and skips the ``.get(..., "speaker0")`` default."""
        snap = {
            "nodes": [{"id": "Alex"}, {"id": "Acme"}],
            "links": [
                {"source": "Alex", "target": "Acme", "predicate": "works_for"},
            ],
        }
        snap_path = tmp_path / "graph_merged_snapshot.json"
        snap_path.write_text(json.dumps(snap), encoding="utf-8")

        state = _state_enabled()
        refiner = _stub_refiner()
        state["consolidation_loop"].build_tier_refiner.return_value = refiner

        with patch(
            "paramem.graph.merger.GraphMerger.merge_relations",
            return_value=None,
        ) as mocked:
            _run_normalize(state, CalibrateNormalizeRequest(snapshot_path=str(snap_path)))

        relations_arg = mocked.call_args.args[0]
        assert len(relations_arg) == 1
        assert relations_arg[0].speaker_id == "speaker0"

    def test_inline_relation_with_empty_speaker_gets_structural_placeholder(self):
        """An inline relation carrying an explicit empty ``speaker_id`` must
        still resolve to the dispatcher's structural ``"speaker0"``
        default — the same placeholder the snapshot branch gets — since
        normalization consults attribution nowhere."""
        state = _state_enabled()
        refiner = _stub_refiner()
        state["consolidation_loop"].build_tier_refiner.return_value = refiner

        req = CalibrateNormalizeRequest(
            relations=[
                {
                    "subject": "Alex",
                    "predicate": "works_for",
                    "object": "Acme",
                    "speaker_id": "",
                }
            ],
        )

        with patch(
            "paramem.graph.merger.GraphMerger.merge_relations",
            return_value=None,
        ) as mocked:
            _run_normalize(state, req)

        relations_arg = mocked.call_args.args[0]
        assert len(relations_arg) == 1
        assert relations_arg[0].speaker_id == "speaker0"

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

        result = _run_normalize(
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
        rule lives in the tier pass (highest reinforcement_count); a second
        copy would drift from it."""
        import ast

        tree = ast.parse(inspect.getsource(calibrate.dispatch_normalize).lstrip())
        # Prose describing the production rule is fine; executing it is not,
        # so strip docstrings and comments by reading identifiers only.
        names = {
            node.id if isinstance(node, ast.Name) else node.attr
            for node in ast.walk(tree)
            if isinstance(node, (ast.Name, ast.Attribute))
        }
        assert "normalize_predicates" not in names
        assert "reinforcement_count" not in names


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


class TestExtractionPipelineKwargsSeed:
    """Verify ExtractionPipeline.kwargs passes seed through."""

    def test_seed_forwarded(self):
        from unittest.mock import MagicMock

        from paramem.graph.extraction_pipeline import ExtractionConfig

        pipeline = ExtractionPipeline(
            MagicMock(), MagicMock(), config=ExtractionConfig(scrub_categories=())
        )
        kwargs = pipeline.kwargs(seed=7, speaker_id="speaker0")
        assert kwargs["seed"] == 7

    def test_seed_none_by_default(self):
        from unittest.mock import MagicMock

        from paramem.graph.extraction_pipeline import ExtractionConfig

        pipeline = ExtractionPipeline(
            MagicMock(), MagicMock(), config=ExtractionConfig(scrub_categories=())
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
            _run_name(_state_disabled(), req)
        assert exc.value.status_code == 404

    def test_model_missing_503(self):
        """calibrate_name returns 503 in cloud-only / defer-model mode."""
        state = _state_enabled()
        state["model"] = None
        req = CalibrateNameRequest(turns=[{"role": "user", "text": "I'm Jordan."}])
        with pytest.raises(HTTPException) as exc:
            _run_name(state, req)
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
            _run_name(state, req)
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
            result = _run_name(state, req)

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
        shipped_sys = _load_prompt("name_extraction_system.txt")
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
            # Echo every message's content into the rendered text (like a
            # real chat template would) — supports_system_role's own probe
            # call (inside adapt_messages, invoked by render_chat_prompt)
            # checks for its marker string in the rendered output, and a
            # fixed return value would make that probe always report "no
            # system-role support", folding system content into user and
            # breaking this test's separate system/user assertions below.
            return "".join(m["content"] for m in messages)

        state["tokenizer"].apply_chat_template.side_effect = _capture_template

        with _mock.patch(
            "paramem.evaluation.recall.generate_answer",
            return_value="Alex",
        ):
            result = _run_name(state, req)

        # The provenance block must report the variants as overrides —
        # _load_prompt records "<override:NAME>" for a prompt_overrides hit,
        # so what is reported is what the model was actually given.
        paths_reported = {p["path"] for p in result["prompts"]}
        assert "<override:name_extraction_system.txt>" in paths_reported
        assert "<override:name_extraction.txt>" in paths_reported

        # The model must have received the OVERRIDE content, not the default.
        # captured_messages[0] is supports_system_role's own probe call
        # (render_chat_prompt applies adapt_messages, which checks system-role
        # support via a throwaway apply_chat_template call before the real
        # render) — the actual rendered messages are always the LAST call.
        assert captured_messages, "apply_chat_template was never called"
        msgs = captured_messages[-1]
        sys_content = next(m["content"] for m in msgs if m["role"] == "system")
        user_content = next(m["content"] for m in msgs if m["role"] == "user")
        assert sys_content == "custom system", (
            f"Model received default system prompt instead of override: {sys_content!r}"
        )
        assert "custom user" in user_content, (
            f"Model received default user prompt instead of override: {user_content!r}"
        )


def _respond_dispatch_double(*, text: str = "a reply", escalated: bool = False, diagnostics=None):
    """A ``handle_chat`` stand-in for ``/calibrate/respond`` tests.

    Opens the real ``"serve_turn"`` phase around a stamp of *text*/
    *diagnostics* rather than skipping it — see ``_chain_side_effect``'s
    docstring above for why a dispatch double must fire the real phase: a
    double that skips it would only exercise the calibration substrate's
    "declared step never ran" refusal, not the wiring this test aims at.
    """
    diagnostics = diagnostics if diagnostics is not None else {"exit_via": "personal_probe"}

    def _call(**_kwargs):
        with phase_trace("serve_turn") as phase:
            phase.set_raw(text)
            phase.set_parsed(dict(diagnostics))
        return ChatResult(text=text, escalated=escalated, diagnostics=dict(diagnostics))

    return _call


class TestCalibrateRespond:
    """Tests for the ``/calibrate/respond`` use case."""

    def test_disabled_404(self):
        req = CalibrateRespondRequest(text="Hello", speaker_id="speaker0")
        with pytest.raises(HTTPException) as exc:
            _run_respond(_state_disabled(), req)
        assert exc.value.status_code == 404

    def test_model_missing_503(self):
        state = _state_respond()
        state["model"] = None
        req = CalibrateRespondRequest(text="Hello", speaker_id="speaker0")
        with pytest.raises(HTTPException) as exc:
            _run_respond(state, req)
        assert exc.value.status_code == 503

    def test_unknown_speaker_400_names_the_id(self):
        """An unenrolled speaker is a 400, never a 404 — the driver script
        (``scripts/dev/calibrate_prompts.py``) turns any 404 into an
        actionable ``calibrate_endpoint_enabled`` operator hint, which would
        mislead on an unenrolled speaker rather than an unenabled endpoint."""
        state = _state_respond()
        state["speaker_store"].get_name.return_value = None
        req = CalibrateRespondRequest(text="Hello", speaker_id="speaker99")
        with pytest.raises(HTTPException) as exc:
            _run_respond(state, req)
        assert exc.value.status_code == 400
        assert "speaker99" in exc.value.detail

    def test_empty_text_400(self):
        """A blank body is rejected at the request-schema boundary — the
        same ``NonBlankText`` rule ``/chat`` applies — before construction
        even succeeds, so this dispatch never sees an empty transcript."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="empty or whitespace-only"):
            CalibrateRespondRequest(text="", speaker_id="speaker0")

    @pytest.mark.parametrize(
        "component", ["speaker_store", "session_buffer", "router", "memory_store"]
    )
    def test_missing_component_503(self, component):
        state = _state_respond()
        state[component] = None
        req = CalibrateRespondRequest(text="Hello", speaker_id="speaker0")
        with pytest.raises(HTTPException) as exc:
            _run_respond(state, req)
        assert exc.value.status_code == 503

    def test_missing_prompt_variant_400_before_dispatch(self, tmp_path):
        state = _state_respond()
        state["config"].paths = PathsConfig(calibration=tmp_path)
        req = CalibrateRespondRequest(
            text="Hello",
            speaker_id="speaker0",
            prompt_variants={"serving_system.txt": "absent_variant.txt"},
        )
        with patch(
            "paramem.server.inference.handle_chat",
            side_effect=AssertionError("dispatch must not run when the guard rejects"),
        ):
            with pytest.raises(HTTPException) as exc:
                _run_respond(state, req)
        assert exc.value.status_code == 400
        assert "variant not found" in exc.value.detail.lower()

    def test_parameter_coverage_matches_handle_chat_signature(self):
        """Fails the moment ``handle_chat`` grows a parameter this dispatch
        does not pass through. ``handle_chat`` currently takes 14
        parameters; this test protects only THIS call site
        (``calibrate_respond``'s own kwarg set) — the other two production
        call sites (``POST /chat``, ``POST /debug/probe``) each need their
        own equivalent coverage against the same signature drift."""
        expected = set(inspect.signature(handle_chat).parameters)
        captured: dict = {}

        def _capture(**kwargs):
            captured.update(kwargs)
            with phase_trace("serve_turn"):
                pass
            return ChatResult(text="ok", escalated=False, diagnostics={})

        state = _state_respond()
        req = CalibrateRespondRequest(text="Hello", speaker_id="speaker0")
        with patch("paramem.server.inference.handle_chat", side_effect=_capture):
            _run_respond(state, req)

        assert set(captured) == expected

    def test_uniform_envelope(self):
        state = _state_respond()
        req = CalibrateRespondRequest(text="Hello", speaker_id="speaker0")
        diagnostics = {"exit_via": "personal_probe", "intent": "PERSONAL"}
        with patch(
            "paramem.server.inference.handle_chat",
            side_effect=_respond_dispatch_double(
                text="a reply", escalated=True, diagnostics=diagnostics
            ),
        ):
            result = _run_respond(state, req)

        # Exact field-set pin, not a subset check: run_stage's own Returns
        # contract (see its docstring) names every key it emits, including
        # run_id and unreached_step — a field this envelope silently
        # started or stopped emitting must fail this test.
        expected_keys = {
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
            "phases",
            "artifact_dir",
            "run_id",
            "unreached_step",
        }
        assert set(result.keys()) == expected_keys
        assert result["stage"] == "respond"
        assert result["raw_output"] == "a reply"
        assert result["parsed"]["escalated"] is True
        # No resolved_text companion: it would write a household member's
        # real display name into an on-disk artifact and duplicate
        # raw_output — see dispatch_respond's docstring.
        assert "resolved_text" not in result["parsed"]
        assert result["parsed"]["exit_via"] == "personal_probe"
        assert result["parsed"]["intent"] == "PERSONAL"
        # No prompt_variants were supplied on this request, so there is
        # nothing to have failed to exercise.
        assert result["parsed"]["variants_unexercised"] == []

    def test_no_params_field_and_params_effective_all_null(self):
        """Pins against a later re-introduction of an echo-only ``params``
        field: the request model exposes none, and ``params_effective`` is
        all-``null`` because the ``respond`` stage's declaration
        (``calibrate._STANDALONE["respond"]``) always builds the spec with
        a bare ``CalibrateParams()`` and ``supports_seed=False``."""
        req = CalibrateRespondRequest(
            text="Hello", speaker_id="speaker0", **{"params": {"temperature": 0.9}}
        )
        assert not hasattr(req, "params")

        state = _state_respond()
        with patch(
            "paramem.server.inference.handle_chat",
            side_effect=_respond_dispatch_double(),
        ):
            result = _run_respond(state, req)

        assert all(v is None for v in result["params_effective"].values())

    def test_prompt_variant_provenance_and_content_reach_dispatch(self, tmp_path):
        """The overridden ``serving_system.txt`` is reported in provenance
        AND is the exact content the dispatch double's own prompt load
        returns — one derivation (``_load_prompt`` under the active
        override), two readers, per ``TestCalibrateName.
        test_filename_override_used_at_execution``."""
        state = _state_respond()
        variants = tmp_path / "prompts"
        variants.mkdir()
        variant_content = "custom serving system prompt"
        (variants / "my_serving_system.txt").write_text(variant_content)
        state["config"].paths = PathsConfig(calibration=tmp_path)

        req = CalibrateRespondRequest(
            text="Hello",
            speaker_id="speaker0",
            prompt_variants={"serving_system.txt": "my_serving_system.txt"},
        )

        def _loads_serving_system(**_kwargs):
            from paramem.server.prompts import serving_system_prompt

            with phase_trace("serve_turn") as phase:
                content = serving_system_prompt()
                phase.set_raw(content)
                phase.set_parsed({"exit_via": "test"})
            return ChatResult(text=content, escalated=False, diagnostics={"exit_via": "test"})

        with patch(
            "paramem.server.inference.handle_chat",
            side_effect=_loads_serving_system,
        ):
            result = _run_respond(state, req)

        paths_reported = {p["path"] for p in result["prompts"]}
        assert "<override:serving_system.txt>" in paths_reported
        assert result["raw_output"] == variant_content
        # The override's production basename shows up as <override:...> in
        # provenance above, so nothing is unexercised.
        assert result["parsed"]["variants_unexercised"] == []

    def test_variants_unexercised_lists_basename_never_loaded(self, tmp_path):
        """A branch that never touches a given production prompt (e.g. an
        HA-answered or non-cloud, non-temporal turn skipping
        ``recall_selection.txt``) reports the gap in
        ``variants_unexercised`` rather than 400ing — a multi-variant sweep
        must not fail because one leg didn't fire on this utterance."""
        state = _state_respond()
        variants = tmp_path / "prompts"
        variants.mkdir()
        (variants / "my_recall_selection.txt").write_text("unused variant")
        state["config"].paths = PathsConfig(calibration=tmp_path)

        req = CalibrateRespondRequest(
            text="Hello",
            speaker_id="speaker0",
            prompt_variants={"recall_selection.txt": "my_recall_selection.txt"},
        )

        with patch(
            "paramem.server.inference.handle_chat",
            side_effect=_respond_dispatch_double(),
        ):
            result = _run_respond(state, req)

        assert result["parsed"]["variants_unexercised"] == ["recall_selection.txt"]

    def test_serve_turn_phase_lands_in_envelope_through_real_handle_chat(self):
        """No dispatch double: exercises the trace no-op nesting the doubles
        in the tests above bypass.  ``handle_chat`` opens its OWN
        ``extraction_trace()``/``phase_trace("serve_turn")`` scope around
        its whole dispatch (see ``paramem.server.inference.handle_chat``);
        ``run_stage`` also opens ``extraction_trace()`` around
        ``spec.dispatch()``.  Re-entry into an already-active
        ``extraction_trace()`` is documented as a no-op (the inner scope
        lands on the SAME trace) — this proves that nesting actually works
        end to end, not just in the phase_trace unit tests, by calling the
        real ``handle_chat`` and reading the ``serve_turn`` record back out
        of the calibration envelope.

        Only ``_base_model_answer`` — the actual model-touching leaf — is
        patched, mirroring ``tests/server/test_chat_diagnostics.py``.  The
        router is a permissive stand-in returning a GENERAL, no-steps
        ``RoutingPlan`` so the turn falls through HA (no ``ha_client``) and
        cloud (no ``cloud_agent``) to the base model; ``abstention.enabled``
        is set False so the abstention gate short-circuits without needing
        a real ``sentence_type`` config namespace — the minimal config
        additions this dispatch path actually touches, not a full
        ``ServerConfig()``.
        """
        from paramem.server.router import Intent, RoutingPlan

        state = _state_respond()
        state["config"].personal_referent = None
        state["config"].abstention = SimpleNamespace(enabled=False)
        state["router"] = MagicMock()
        state["router"].route.return_value = RoutingPlan(strategy="direct", intent=Intent.GENERAL)
        req = CalibrateRespondRequest(text="hello there", speaker_id="speaker0")

        with patch(
            "paramem.server.inference._base_model_answer",
            return_value=ChatResult(text="base reply", escalated=False, diagnostics={}),
        ):
            result = _run_respond(state, req)

        serve_records = [p for p in result["phases"] if p.get("name") == "serve_turn"]
        assert len(serve_records) == 1
        assert serve_records[0]["raw_output"] == "base reply"
        assert result["raw_output"] == "base reply"
        assert result["parsed"]["exit_via"] == "base_model"


class TestExtractNameViaLlmUserTurnFilter:
    """Unit tests for extract_name_via_llm's user-turn filtering.

    These tests mock generate_answer and verify that:
    1. Assistant turns are excluded from the transcript passed to the model
       when user_turns_only=True (an assistant's own salutation must not be
       mistaken for a user's self-introduction).
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
        """_load_prompt raises FileNotFoundError when file is absent everywhere.

        We test this via the prompts module directly (with a nonexistent filename) rather than
        via extract_name_via_llm, because the production prompt files exist in _DEFAULT_PROMPT_DIR
        and would be found as fallback even when prompts_dir has no files.
        """
        from paramem.graph.prompts import _load_prompt

        with pytest.raises(FileNotFoundError) as exc_info:
            _load_prompt("no_such_prompt_xyz.txt", prompts_dir=tmp_path)
        assert "no_such_prompt_xyz.txt" in str(exc_info.value)
        assert "Searched" in str(exc_info.value)
