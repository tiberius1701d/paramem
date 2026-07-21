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
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from paramem.graph.extraction_pipeline import ExtractionPipeline
from paramem.server.calibrate import (
    CalibrateAnonymizeRequest,
    CalibrateEnrichRequest,
    CalibrateExtractRequest,
    CalibrateNameRequest,
    CalibrateNormalizeRequest,
    CalibrateParams,
    CalibratePlausibilityRequest,
    CalibrateProceduralRequest,
    _effective_params,
    _preflight,
    _production_turn_markers,
    _require_turn_marked_transcript,
    _run_calibration,
    calibrate_anonymize,
    calibrate_enrich,
    calibrate_extract,
    calibrate_name,
    calibrate_normalize,
    calibrate_plausibility,
    calibrate_procedural,
)
from paramem.server.config import SanitizationConfig


def _state_disabled() -> dict:
    """Server state where calibrate is OFF (production default)."""
    consolidation_cfg = SimpleNamespace(calibrate_endpoint_enabled=False)
    config = SimpleNamespace(consolidation=consolidation_cfg, sanitization=SanitizationConfig())
    return {
        "config": config,
        "consolidating": False,
        "model": MagicMock(),
        "tokenizer": MagicMock(),
    }


def _state_enabled() -> dict:
    consolidation_cfg = SimpleNamespace(calibrate_endpoint_enabled=True)
    # SanitizationConfig() carries the production default ``scrub`` list
    # (SanitizationConfig._DEFAULT_SCRUB) — calibrate_anonymize sources
    # ``scrub`` from here, the same policy knob every production call
    # site reads (see calibrate_anonymize's docstring).
    config = SimpleNamespace(consolidation=consolidation_cfg, sanitization=SanitizationConfig())
    return {
        "config": config,
        "consolidating": False,
        "model": MagicMock(),
        "tokenizer": MagicMock(),
        "consolidation_loop": MagicMock(),
        "model_id": "test-model",
    }


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
    routes through this check (see ``TestCalibrateExtract`` /
    ``TestCalibrateProcedural`` / ``TestCalibrateAnonymize`` /
    ``TestCalibratePlausibility`` below for the per-endpoint 400s).
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


class TestCalibrateExtract:
    def test_disabled_404(self):
        req = CalibrateExtractRequest(
            transcript="[user] x",
            speaker_id="speaker0",
            source_type="document",
        )
        with pytest.raises(HTTPException) as exc:
            calibrate_extract(_state_disabled(), req)
        assert exc.value.status_code == 404

    def test_consolidating_503(self):
        state = _state_enabled()
        state["consolidating"] = True
        req = CalibrateExtractRequest(
            transcript="[user] x",
            speaker_id="speaker0",
            source_type="document",
        )
        with pytest.raises(HTTPException) as exc:
            calibrate_extract(state, req)
        assert exc.value.status_code == 503

    def test_debug_writer_invoked_with_response_and_session_id(self):
        """calibrate_extract routes its assembled response through the debug hook.

        Regression guard for the call site at calibrate.py's
        ``loop._debug_writer.on_calibrate_extract(response, session_id=...)``
        line: DebugSnapshotWriter.on_calibrate_extract's own file-write
        behaviour is covered by
        tests/test_consolidation.py::TestCalibrateExtractDebugSnapshot, but
        nothing previously asserted that calibrate_extract actually calls it.
        """
        from paramem.graph.schema import SessionGraph

        state = _state_enabled()
        graph = SessionGraph(session_id="calib", timestamp="2026-01-01T00:00:00Z")
        state["consolidation_loop"].extraction.run.return_value = graph
        req = CalibrateExtractRequest(
            transcript="[user] hello there",
            speaker_id="speaker0",
            source_type="transcript",
        )

        result = calibrate_extract(state, req)

        writer = state["consolidation_loop"]._debug_writer
        assert writer.on_calibrate_extract.call_count == 1
        call = writer.on_calibrate_extract.call_args
        assert call.args[0] == result
        assert call.kwargs["session_id"] == req.session_id

    def test_unmarked_transcript_raises_400(self):
        """A bare, unmarked transcript is rejected before the pipeline runs.

        Mutation: remove the ``_require_turn_marked_transcript`` call from
        ``calibrate_extract`` -> this test fails (no 400 raised, or the
        pipeline mock gets invoked on unmarked input).
        """
        state = _state_enabled()
        req = CalibrateExtractRequest(
            transcript="Should I follow up with Alex tomorrow?",
            speaker_id="speaker0",
            source_type="transcript",
        )
        with pytest.raises(HTTPException) as exc:
            calibrate_extract(state, req)
        assert exc.value.status_code == 400
        assert "turn-marked" in exc.value.detail.lower()
        assert not state["consolidation_loop"].extraction.run.called

    def test_marked_transcript_reaches_pipeline(self):
        """A turn-marked transcript clears the gate and reaches the pipeline.

        Mutation: make the gate reject marked transcripts too (e.g. require
        an exact-match on a different marker set) -> this test fails
        because ``loop.extraction.run`` is never reached.
        """
        from paramem.graph.schema import SessionGraph

        state = _state_enabled()
        graph = SessionGraph(session_id="calib", timestamp="2026-01-01T00:00:00Z")
        state["consolidation_loop"].extraction.run.return_value = graph
        req = CalibrateExtractRequest(
            transcript="[user] Should I follow up with Alex tomorrow?",
            speaker_id="speaker0",
            source_type="transcript",
        )
        result = calibrate_extract(state, req)
        assert state["consolidation_loop"].extraction.run.called
        assert result["stage"] == "extract"

    def test_prompt_override_reaches_model(self, tmp_path):
        """A prompts_dir + filename override actually drives the model call,
        with a REAL ``ExtractionPipeline`` (not a mocked ``.run``).

        Regression guard for the ``_run_calibration(entry="pipeline")``
        conversion: provenance must come from the ``local_extract`` phase
        record the pipeline itself opens (via ``_load_prompt``'s
        ``record_prompt`` call) — never a hand-built ``prompts=[...]``
        literal. Mirrors
        ``TestCalibratePlausibility::test_prompt_override_reaches_model``.
        ``stop_phase="local_extract"`` keeps the run cheap (no SOTA/second-
        order passes).
        """
        import unittest.mock as _mock

        from paramem.graph.extraction_pipeline import ExtractionConfig
        from paramem.graph.prompts import _load_prompt

        state = _state_enabled()
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        sentinel = "SENTINEL-EXTRACT-OVERRIDE"
        (prompts_dir / "my_extract.txt").write_text(
            f"{sentinel}\n{{speaker_context}}\ntranscript: {{transcript}}"
        )
        (prompts_dir / "my_extract_system.txt").write_text("SENTINEL-EXTRACT-SYSTEM")

        pipeline = ExtractionPipeline(
            state["model"], state["tokenizer"], config=ExtractionConfig(scrub=set())
        )
        loop = MagicMock()
        loop.extraction = pipeline
        loop.model = state["model"]
        state["consolidation_loop"] = loop

        captured: list[list[dict]] = []

        def _capture_template(messages, **kw):
            captured.append(messages)
            return "formatted-prompt"

        state["tokenizer"].apply_chat_template.side_effect = _capture_template

        req = CalibrateExtractRequest(
            transcript="[user] hello there",
            speaker_id="speaker0",
            source_type="transcript",
            prompts_dir=str(prompts_dir),
            extraction_prompt_filename="my_extract.txt",
            extraction_system_prompt_filename="my_extract_system.txt",
            stop_phase="local_extract",
        )

        with (
            _mock.patch("paramem.graph.extractor.adapt_messages", side_effect=lambda m, t: m),
            _mock.patch(
                "paramem.graph.extractor.generate_answer",
                return_value='{"entities": [], "relations": []}',
            ),
        ):
            result = calibrate_extract(state, req)

        assert result["stage"] == "extract"
        assert captured, "apply_chat_template was never called"
        user_content = next(m["content"] for m in captured[0] if m["role"] == "user")
        assert sentinel in user_content, (
            f"Model received default prompt instead of override: {user_content!r}"
        )

        # Reported provenance comes from the real phase-trace record
        # (populated by _load_prompt's own record_prompt call inside
        # _generate_extraction), not a hand-built _read_prompt literal.
        expected = _load_prompt("my_extract.txt", prompts_dir=prompts_dir, required=True)
        expected_sha = hashlib.sha256(expected.encode("utf-8")).hexdigest()[:12]
        shas = {p["sha"] for p in result["prompts"]}
        templates = {p["template"] for p in result["prompts"]}
        assert expected_sha in shas
        assert expected in templates


class TestCalibrateProceduralRequest:
    """Schema-level tests for CalibrateProceduralRequest (no GPU)."""

    def test_defaults(self):
        req = CalibrateProceduralRequest(transcript="[user] x", speaker_id="speaker0")
        assert req.source_type == "transcript"
        assert req.session_id == "calib"
        assert req.speaker_name is None
        assert req.prompts_dir is None
        assert req.procedural_prompt_filename is None
        assert req.procedural_system_prompt_filename is None

    def test_source_type_pattern_rejects_bad_value(self):
        with pytest.raises(ValidationError):
            CalibrateProceduralRequest(
                transcript="[user] x", speaker_id="speaker0", source_type="voice"
            )


class TestCalibrateProcedural:
    def test_disabled_404(self):
        req = CalibrateProceduralRequest(transcript="[user] x", speaker_id="speaker0")
        with pytest.raises(HTTPException) as exc:
            calibrate_procedural(_state_disabled(), req)
        assert exc.value.status_code == 404

    def test_consolidating_503(self):
        state = _state_enabled()
        state["consolidating"] = True
        req = CalibrateProceduralRequest(transcript="[user] x", speaker_id="speaker0")
        with pytest.raises(HTTPException) as exc:
            calibrate_procedural(state, req)
        assert exc.value.status_code == 503

    def test_empty_speaker_id_400(self):
        state = _state_enabled()
        req = CalibrateProceduralRequest(transcript="[user] x", speaker_id="")
        with pytest.raises(HTTPException) as exc:
            calibrate_procedural(state, req)
        assert exc.value.status_code == 400
        assert "speaker_id" in exc.value.detail.lower()

    def test_unmarked_transcript_raises_400(self):
        """Mutation: remove the gate call from ``calibrate_procedural`` ->
        this test fails (no 400, or the pipeline mock gets invoked)."""
        state = _state_enabled()
        req = CalibrateProceduralRequest(
            transcript="My sister lives in Frankfurt.", speaker_id="speaker0"
        )
        with pytest.raises(HTTPException) as exc:
            calibrate_procedural(state, req)
        assert exc.value.status_code == 400
        assert "turn-marked" in exc.value.detail.lower()
        assert not state["consolidation_loop"].extraction.run_procedural.called

    def test_marked_transcript_reaches_pipeline(self):
        """Mutation: make the gate reject marked transcripts too -> this
        test fails because ``loop.extraction.run_procedural`` is never
        reached."""
        from paramem.graph.schema import SessionGraph

        state = _state_enabled()
        graph = SessionGraph(session_id="calib", timestamp="2026-01-01T00:00:00Z")
        state["consolidation_loop"].extraction.run_procedural.return_value = graph
        req = CalibrateProceduralRequest(
            transcript="[user] My sister lives in Frankfurt.", speaker_id="speaker0"
        )
        result = calibrate_procedural(state, req)
        assert state["consolidation_loop"].extraction.run_procedural.called
        assert result["stage"] == "procedural"

    def test_prompt_override_reaches_model(self, tmp_path):
        """A prompts_dir + filename override actually drives the model call,
        with a REAL ``ExtractionPipeline`` (not a mocked ``.run_procedural``).

        Regression guard for the ``_run_calibration(entry="pipeline")``
        conversion: procedural extraction now runs through the shared
        ``_run_local_extraction`` primitive inside ``extract_procedural_graph``,
        which self-traces the ``procedural_extract`` phase — proven here by
        asserting ``result["phases"]`` carries that record.  Provenance
        comes from the real phase-trace record (``_load_prompt``'s own
        ``record_prompt`` call), never a hand-built ``prompts=[...]``
        literal.
        """
        import unittest.mock as _mock

        from paramem.graph.extraction_pipeline import ExtractionConfig
        from paramem.graph.prompts import _load_prompt

        state = _state_enabled()
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        sentinel = "SENTINEL-PROCEDURAL-OVERRIDE"
        (prompts_dir / "my_procedural.txt").write_text(
            f"{sentinel}\n{{speaker_context}}\ntranscript: {{transcript}}"
        )
        (prompts_dir / "my_procedural_system.txt").write_text("SENTINEL-PROCEDURAL-SYSTEM")

        pipeline = ExtractionPipeline(
            state["model"], state["tokenizer"], config=ExtractionConfig(scrub=set())
        )
        loop = MagicMock()
        loop.extraction = pipeline
        loop.model = state["model"]
        state["consolidation_loop"] = loop

        captured: list[list[dict]] = []

        def _capture_template(messages, **kw):
            captured.append(messages)
            return "formatted-prompt"

        state["tokenizer"].apply_chat_template.side_effect = _capture_template

        req = CalibrateProceduralRequest(
            transcript="[user] hello there",
            speaker_id="speaker0",
            source_type="transcript",
            prompts_dir=str(prompts_dir),
            procedural_prompt_filename="my_procedural.txt",
            procedural_system_prompt_filename="my_procedural_system.txt",
        )

        with (
            _mock.patch("paramem.graph.extractor.adapt_messages", side_effect=lambda m, t: m),
            _mock.patch(
                "paramem.graph.extractor.generate_answer",
                return_value='{"entities": [], "relations": []}',
            ),
        ):
            result = calibrate_procedural(state, req)

        assert result["stage"] == "procedural"
        assert captured, "apply_chat_template was never called"
        user_content = next(m["content"] for m in captured[0] if m["role"] == "user")
        assert sentinel in user_content, (
            f"Model received default prompt instead of override: {user_content!r}"
        )

        expected = _load_prompt("my_procedural.txt", prompts_dir=prompts_dir, required=True)
        expected_sha = hashlib.sha256(expected.encode("utf-8")).hexdigest()[:12]
        shas = {p["sha"] for p in result["prompts"]}
        templates = {p["template"] for p in result["prompts"]}
        assert result["prompts"], "procedural stage must surface prompt provenance"
        assert expected_sha in shas
        assert expected in templates

        # Proves self-tracing: extract_procedural_graph opens its own
        # extraction_trace() and records a "procedural_extract" phase that
        # nest-no-ops onto _run_calibration's outer trace.
        phase_names = {p.get("name") for p in result["phases"]}
        assert "procedural_extract" in phase_names


class TestCalibrateAnonymize:
    def test_invalid_graph_400(self, tmp_path):
        state = _state_enabled()
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "anonymization.txt").write_text("x")
        req = CalibrateAnonymizeRequest(
            graph={"not": "a valid sessiongraph"},
            transcript="[user] x",
            prompts_dir=str(prompts_dir),
        )
        with pytest.raises(HTTPException) as exc:
            calibrate_anonymize(state, req)
        assert exc.value.status_code == 400

    def test_disabled_404(self):
        req = CalibrateAnonymizeRequest(
            graph={"session_id": "x", "timestamp": "x", "entities": [], "relations": []},
            transcript="[user] x",
        )
        with pytest.raises(HTTPException) as exc:
            calibrate_anonymize(_state_disabled(), req)
        assert exc.value.status_code == 404

    def test_prompt_override_reaches_model(self, tmp_path):
        """A prompts_dir + filename override actually drives the model call.

        Regression guard: anonymize_with_local_model previously ignored
        prompts_dir and always loaded configs/prompts/anonymization.txt, so
        the override was cosmetic (displayed but not executed).  This asserts
        the SENTINEL override template is what apply_chat_template receives.
        """
        import unittest.mock as _mock

        state = _state_enabled()
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        sentinel = "SENTINEL-ANON-OVERRIDE"
        (prompts_dir / "my_anon.txt").write_text(
            f"{sentinel}\nfacts: {{facts_json}}\ntranscript: {{transcript}}"
        )

        req = CalibrateAnonymizeRequest(
            graph={"session_id": "x", "timestamp": "x", "entities": [], "relations": []},
            transcript="[user] hello there",
            prompts_dir=str(prompts_dir),
            anonymization_prompt_filename="my_anon.txt",
        )

        captured: list[list[dict]] = []

        def _capture_template(messages, **kw):
            captured.append(messages)
            return "formatted-prompt"

        state["tokenizer"].apply_chat_template.side_effect = _capture_template

        valid_json = '{"mapping": {}, "anonymized_transcript": "[user] hello there"}'
        with (
            _mock.patch("paramem.graph.cloud_egress.adapt_messages", side_effect=lambda m, t: m),
            _mock.patch("paramem.graph.cloud_egress.generate_answer", return_value=valid_json),
        ):
            result = calibrate_anonymize(state, req)

        assert result["stage"] == "anonymize"
        assert captured, "apply_chat_template was never called"
        user_content = next(m["content"] for m in captured[0] if m["role"] == "user")
        assert sentinel in user_content, (
            f"Model received default prompt instead of override: {user_content!r}"
        )

    def test_runs_on_base_weights(self):
        """calibrate_anonymize disables the active adapter around the model call.

        Mirrors production, where anonymization runs inside extract_graph's
        disable_adapter scope.  The direct calibrate call must do the same via
        base_model_inference so it does not read the training-active adapter.
        """
        import unittest.mock as _mock

        state = _state_enabled()
        state["model"] = _peft_model_mock()
        req = CalibrateAnonymizeRequest(
            graph={"session_id": "x", "timestamp": "x", "entities": [], "relations": []},
            transcript="[user] hello",
        )

        with _mock.patch(
            "paramem.graph.cloud_egress.anonymize_with_local_model",
            return_value=({}, "[user] hello", "raw"),
        ) as helper:
            result = calibrate_anonymize(state, req)

        assert result["stage"] == "anonymize"
        assert helper.called
        assert state["model"].disable_adapter.called, (
            "calibrate_anonymize must run the helper under base_model_inference"
        )

    def test_calibrate_anonymize_returns_mapping_and_transcript(self):
        """The ``parsed`` payload carries the model's two artifacts —
        ``forward`` AND ``anonymized_transcript`` — plus a ``status``
        (current 3-tuple ``anonymize_with_local_model`` contract; facts
        are still never part of the anonymizer's response).

        Mutation: drop ``anonymized_transcript`` or ``status`` from
        ``parsed`` -> this test fails.
        """
        import unittest.mock as _mock

        state = _state_enabled()
        req = CalibrateAnonymizeRequest(
            graph={"session_id": "x", "timestamp": "x", "entities": [], "relations": []},
            transcript="[user] hello",
        )

        with _mock.patch(
            "paramem.graph.cloud_egress.anonymize_with_local_model",
            return_value=({"Alex": "Person_1"}, "[user] Person_1 said hi", "raw"),
        ):
            result = calibrate_anonymize(state, req)

        assert result["parsed"] == {
            "status": "ok",
            "anonymized_transcript": "[user] Person_1 said hi",
            "forward": {"Alex": "Person_1"},
            "reverse": {"Person_1": "Alex"},
            "anon_facts": [],
            "norm_stats": {"inverted": 0, "dropped": 0},
        }

    def test_fail_closed_when_mapping_is_none(self):
        """Fail-closed: a parse failure (``mapping is None``) is
        reported as ``status: "failed"`` with an empty ``forward`` table
        and an empty ``anonymized_transcript`` — never a fallback to the
        real-name transcript.
        """
        import unittest.mock as _mock

        state = _state_enabled()
        req = CalibrateAnonymizeRequest(
            graph={"session_id": "x", "timestamp": "x", "entities": [], "relations": []},
            transcript="[user] My friend Alex moved to Berlin.",
        )

        with _mock.patch(
            "paramem.graph.cloud_egress.anonymize_with_local_model",
            return_value=(None, "", "raw"),
        ):
            result = calibrate_anonymize(state, req)

        assert result["parsed"] == {
            "status": "failed",
            "anonymized_transcript": "",
            "forward": {},
            "reverse": {},
            "anon_facts": [],
            "norm_stats": {"inverted": 0, "dropped": 0},
        }

    def test_empty_scrub_opts_out_without_model_call(self):
        """An operator-configured empty ``scrub`` (operator opt-out) short-circuits
        before any model call — the passed-in transcript egresses verbatim.

        Mutation: call ``anonymize_with_local_model`` unconditionally ->
        this test fails (``helper.called`` is True).
        """
        import unittest.mock as _mock

        state = _state_enabled()
        state["config"].sanitization.scrub = []
        req = CalibrateAnonymizeRequest(
            graph={"session_id": "x", "timestamp": "x", "entities": [], "relations": []},
            transcript="[user] My friend Alex moved to Berlin.",
        )

        with _mock.patch(
            "paramem.graph.cloud_egress.anonymize_with_local_model",
        ) as helper:
            result = calibrate_anonymize(state, req)

        assert not helper.called
        assert result["parsed"] == {
            "status": "opted_out",
            "anonymized_transcript": "[user] My friend Alex moved to Berlin.",
            "forward": {},
            "reverse": {},
            "anon_facts": [],
            "norm_stats": {"inverted": 0, "dropped": 0},
        }

    def test_scrub_sourced_from_config(self):
        """``scrub`` reaches ``anonymize_with_local_model`` from
        ``state["config"].sanitization.scrub`` — the same policy knob
        every production call site reads — not a request field.

        Mutation: hardcode a different scrub set or drop the ``scrub=``
        kwarg -> this test fails.
        """
        import unittest.mock as _mock

        state = _state_enabled()
        state["config"].sanitization.scrub = ["person name", "phone number"]
        req = CalibrateAnonymizeRequest(
            graph={"session_id": "x", "timestamp": "x", "entities": [], "relations": []},
            transcript="[user] hello",
        )

        with _mock.patch(
            "paramem.graph.cloud_egress.anonymize_with_local_model",
            return_value=({}, "[user] hello", "raw"),
        ) as helper:
            calibrate_anonymize(state, req)

        assert helper.call_args.kwargs["scrub"] == {"person name", "phone number"}

    def test_speaker_name_reaches_speaker_seeding(self):
        """``req.speaker_name`` reaches
        ``anonymize_for_cloud``'s speaker-name seeding — the same
        runtime-known speaker name every production caller threads
        through.  A transcript the model itself never named (empty
        mapping) still gets the speaker minted into ``forward``/``reverse``
        via seeding — proving ``speaker_name`` actually reached the
        builder, not merely accepted and dropped.

        Mutation: drop the ``speaker_name=req.speaker_name`` kwarg from
        the ``anonymize_for_cloud`` call -> ``forward`` comes back empty
        -> this test fails.
        """
        import unittest.mock as _mock

        state = _state_enabled()
        req = CalibrateAnonymizeRequest(
            graph={"session_id": "x", "timestamp": "x", "entities": [], "relations": []},
            transcript="[user] hello",
            speaker_name="Alex",
        )

        with _mock.patch(
            "paramem.graph.cloud_egress.anonymize_with_local_model",
            return_value=({}, "[user] hello", "raw"),
        ):
            result = calibrate_anonymize(state, req)

        assert result["parsed"]["forward"] == {"Alex": "Person_1"}
        assert result["parsed"]["reverse"] == {"Person_1": "Alex"}

    def test_unmarked_transcript_raises_400(self):
        """Mutation: remove the gate call from ``calibrate_anonymize`` ->
        this test fails (no 400, or ``anonymize_with_local_model`` runs)."""
        import unittest.mock as _mock

        state = _state_enabled()
        req = CalibrateAnonymizeRequest(
            graph={"session_id": "x", "timestamp": "x", "entities": [], "relations": []},
            transcript="My friend Alex moved to Berlin.",
        )
        with (
            _mock.patch("paramem.graph.cloud_egress.anonymize_with_local_model") as helper,
            pytest.raises(HTTPException) as exc,
        ):
            calibrate_anonymize(state, req)
        assert exc.value.status_code == 400
        assert "turn-marked" in exc.value.detail.lower()
        assert not helper.called

    def test_marked_transcript_reaches_model(self):
        """Mutation: make the gate reject marked transcripts too -> this
        test fails because ``anonymize_with_local_model`` is never called."""
        import unittest.mock as _mock

        state = _state_enabled()
        req = CalibrateAnonymizeRequest(
            graph={"session_id": "x", "timestamp": "x", "entities": [], "relations": []},
            transcript="[user] My friend Alex moved to Berlin.",
        )
        with _mock.patch(
            "paramem.graph.cloud_egress.anonymize_with_local_model",
            return_value=({}, "[user] My friend Person_1 moved to City_1.", "raw"),
        ) as helper:
            result = calibrate_anonymize(state, req)
        assert helper.called
        assert result["stage"] == "anonymize"


class TestCalibratePlausibility:
    def test_disabled_404(self):
        req = CalibratePlausibilityRequest(facts=[], transcript="[user] x")
        with pytest.raises(HTTPException) as exc:
            calibrate_plausibility(_state_disabled(), req)
        assert exc.value.status_code == 404

    def test_prompt_override_reaches_model(self, tmp_path):
        """A prompts_dir + filename override actually drives the model call.

        Regression guard: local_plausibility_filter previously ignored
        prompts_dir and always loaded configs/prompts/sota_plausibility.txt,
        so the override was cosmetic.  This asserts the SENTINEL override
        template is what apply_chat_template receives.
        """
        import unittest.mock as _mock

        state = _state_enabled()
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        sentinel = "SENTINEL-PLAUS-OVERRIDE"
        (prompts_dir / "my_plaus.txt").write_text(
            f"{sentinel}\nfacts: {{facts_json}}\ntranscript: {{transcript}}"
        )

        req = CalibratePlausibilityRequest(
            facts=[],
            transcript="[user] hello there",
            prompts_dir=str(prompts_dir),
            plausibility_prompt_filename="my_plaus.txt",
        )

        captured: list[list[dict]] = []

        def _capture_template(messages, **kw):
            captured.append(messages)
            return "formatted-prompt"

        state["tokenizer"].apply_chat_template.side_effect = _capture_template

        with (
            _mock.patch("paramem.graph.extractor.adapt_messages", side_effect=lambda m, t: m),
            _mock.patch("paramem.graph.extractor.generate_answer", return_value="{}"),
        ):
            result = calibrate_plausibility(state, req)

        assert result["stage"] == "plausibility"
        assert captured, "apply_chat_template was never called"
        user_content = next(m["content"] for m in captured[0] if m["role"] == "user")
        assert sentinel in user_content, (
            f"Model received default prompt instead of override: {user_content!r}"
        )

        # Reported provenance is sourced from the real
        # phase-trace record (populated by `_load_prompt`'s own
        # `record_prompt` call inside `local_plausibility_filter`), not a
        # hand-built `_read_prompt` literal — its sha must match a live
        # `_load_prompt` computation of the same file.
        from paramem.graph.prompts import _load_prompt

        expected = _load_prompt("my_plaus.txt", prompts_dir=prompts_dir, required=True)
        expected_sha = hashlib.sha256(expected.encode("utf-8")).hexdigest()[:12]
        # ``local_plausibility_filter`` now also loads its SYSTEM prompt
        # (``sota_plausibility_system.txt``) at call time — no longer a
        # module-import-time constant unreachable by provenance — so this
        # phase records TWO prompts: the (overridden) user template, then
        # the system prompt.  ``req`` carries no system-prompt override
        # here, so the second entry resolves the shipped default.
        expected_system = _load_prompt("sota_plausibility_system.txt", required=True)
        expected_system_sha = hashlib.sha256(expected_system.encode("utf-8")).hexdigest()[:12]
        assert len(result["prompts"]) == 2
        assert result["prompts"][0]["sha"] == expected_sha
        assert result["prompts"][0]["template"] == expected
        assert result["prompts"][1]["sha"] == expected_system_sha
        assert result["prompts"][1]["template"] == expected_system

    def test_runs_on_base_weights(self):
        """calibrate_plausibility disables the active adapter around the model call.

        Mirrors production, where the plausibility filter runs inside
        extract_graph's disable_adapter scope.
        """
        import unittest.mock as _mock

        state = _state_enabled()
        state["model"] = _peft_model_mock()
        req = CalibratePlausibilityRequest(facts=[], transcript="[user] hello")

        with _mock.patch(
            "paramem.graph.extractor.local_plausibility_filter",
            return_value=([], "raw"),
        ) as helper:
            result = calibrate_plausibility(state, req)

        assert result["stage"] == "plausibility"
        assert helper.called
        assert state["model"].disable_adapter.called, (
            "calibrate_plausibility must run the helper under base_model_inference"
        )

    def test_unmarked_transcript_raises_400(self):
        """Mutation: remove the gate call from ``calibrate_plausibility`` ->
        this test fails (no 400, or ``local_plausibility_filter`` runs)."""
        import unittest.mock as _mock

        state = _state_enabled()
        req = CalibratePlausibilityRequest(facts=[], transcript="Should I call the vet?")
        with (
            _mock.patch("paramem.graph.extractor.local_plausibility_filter") as helper,
            pytest.raises(HTTPException) as exc,
        ):
            calibrate_plausibility(state, req)
        assert exc.value.status_code == 400
        assert "turn-marked" in exc.value.detail.lower()
        assert not helper.called

    def test_marked_transcript_reaches_model(self):
        """Mutation: make the gate reject marked transcripts too -> this
        test fails because ``local_plausibility_filter`` is never called."""
        import unittest.mock as _mock

        state = _state_enabled()
        req = CalibratePlausibilityRequest(facts=[], transcript="[user] Should I call the vet?")
        with _mock.patch(
            "paramem.graph.extractor.local_plausibility_filter",
            return_value=([], "raw"),
        ) as helper:
            result = calibrate_plausibility(state, req)
        assert helper.called
        assert result["stage"] == "plausibility"


class TestCalibrateEnrich:
    """Tests for CalibrateEnrichRequest validation and calibrate_enrich.

    Mirrors ``TestCalibratePlausibility`` — ``calibrate_enrich`` is the
    same ``_run_calibration(entry="standalone")`` shape, wrapping
    ``_filter_with_sota`` (the production ``sota_enrich`` stage) instead
    of ``local_plausibility_filter``.  The SOTA provider config
    (``extraction_noise_filter`` / ``_model`` / ``_endpoint``) is not on
    the shared ``_state_enabled()`` fixture's ``consolidation``
    ``SimpleNamespace`` by default, so each test sets the attributes it
    needs directly (``SimpleNamespace`` accepts arbitrary attributes).
    """

    def test_disabled_404(self):
        req = CalibrateEnrichRequest(facts=[], transcript="[user] x")
        with pytest.raises(HTTPException) as exc:
            calibrate_enrich(_state_disabled(), req)
        assert exc.value.status_code == 404

    def test_unmarked_transcript_raises_400(self):
        """Mutation: remove the gate call from ``calibrate_enrich`` ->
        this test fails (no 400, or ``_filter_with_sota`` runs)."""
        import unittest.mock as _mock

        state = _state_enabled()
        state["config"].consolidation.sota_enabled = True
        state["config"].consolidation.extraction_noise_filter = "anthropic"
        state["config"].consolidation.extraction_noise_filter_model = "claude-sonnet-4-6"
        state["config"].consolidation.extraction_noise_filter_endpoint = ""
        req = CalibrateEnrichRequest(facts=[], transcript="Should I call the vet?")
        with (
            _mock.patch("paramem.graph.extractor._filter_with_sota") as helper,
            pytest.raises(HTTPException) as exc,
        ):
            calibrate_enrich(state, req)
        assert exc.value.status_code == 400
        assert "turn-marked" in exc.value.detail.lower()
        assert not helper.called

    def test_provider_not_configured_400(self):
        """Empty ``extraction_noise_filter`` (production default) -> 400,
        no cloud call."""
        import unittest.mock as _mock

        state = _state_enabled()
        state["config"].consolidation.sota_enabled = True
        state["config"].consolidation.extraction_noise_filter = ""
        state["config"].consolidation.extraction_noise_filter_model = "claude-sonnet-4-6"
        state["config"].consolidation.extraction_noise_filter_endpoint = ""
        req = CalibrateEnrichRequest(facts=[], transcript="[user] x")
        with (
            _mock.patch("paramem.graph.extractor._filter_with_sota") as helper,
            pytest.raises(HTTPException) as exc,
        ):
            calibrate_enrich(state, req)
        assert exc.value.status_code == 400
        assert "no cloud provider configured" in exc.value.detail.lower()
        assert not helper.called

    def test_master_switch_off_400(self, monkeypatch: pytest.MonkeyPatch):
        """``sota_enabled: false`` -> 400, no cloud call.

        This endpoint used to egress with the master switch OFF: it checked
        the provider and the key but never ``sota_enabled``. It now asks the
        same cloud-admission component every other egress site asks, and —
        being operator-triggered — fails loudly rather than skipping.

        Mutation: drop the ``sota_enabled`` term from the verdict -> this
        test sees a successful call instead of a 400.
        """
        import unittest.mock as _mock

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        state = _state_enabled()
        state["config"].consolidation.sota_enabled = False
        state["config"].consolidation.extraction_noise_filter = "anthropic"
        state["config"].consolidation.extraction_noise_filter_model = "claude-sonnet-4-6"
        state["config"].consolidation.extraction_noise_filter_endpoint = ""
        req = CalibrateEnrichRequest(facts=[], transcript="[user] x")
        with (
            _mock.patch("paramem.graph.extractor._filter_with_sota") as helper,
            pytest.raises(HTTPException) as exc,
        ):
            calibrate_enrich(state, req)
        assert exc.value.status_code == 400
        assert "sota_enabled is off" in exc.value.detail.lower()
        assert not helper.called

    def test_missing_api_key_400(self, monkeypatch: pytest.MonkeyPatch):
        """Provider configured but its key env var is unset -> 400, no cloud call."""
        import unittest.mock as _mock

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        state = _state_enabled()
        state["config"].consolidation.sota_enabled = True
        state["config"].consolidation.extraction_noise_filter = "anthropic"
        state["config"].consolidation.extraction_noise_filter_model = "claude-sonnet-4-6"
        state["config"].consolidation.extraction_noise_filter_endpoint = ""
        req = CalibrateEnrichRequest(facts=[], transcript="[user] x")
        with (
            _mock.patch("paramem.graph.extractor._filter_with_sota") as helper,
            pytest.raises(HTTPException) as exc,
        ):
            calibrate_enrich(state, req)
        assert exc.value.status_code == 400
        assert "anthropic_api_key env var is unset" in exc.value.detail.lower()
        assert not helper.called

    def test_success(self, monkeypatch: pytest.MonkeyPatch):
        """Provider configured + key present -> ``_filter_with_sota`` runs;
        its returned facts surface at ``result["parsed"]["facts"]``."""
        import unittest.mock as _mock

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        state = _state_enabled()
        state["config"].consolidation.sota_enabled = True
        state["config"].consolidation.extraction_noise_filter = "anthropic"
        state["config"].consolidation.extraction_noise_filter_model = "claude-sonnet-4-6"
        state["config"].consolidation.extraction_noise_filter_endpoint = ""
        req = CalibrateEnrichRequest(
            facts=[{"subject": "speaker0", "predicate": "likes", "object": "coffee"}],
            transcript="[user] I like coffee.",
        )
        enriched_facts = [{"subject": "speaker0", "predicate": "likes", "object": "coffee"}]
        with _mock.patch(
            "paramem.graph.extractor._filter_with_sota",
            return_value=(enriched_facts, "anon transcript", {}, "raw", {}),
        ) as helper:
            result = calibrate_enrich(state, req)
        assert helper.called
        assert result["stage"] == "enrich"
        assert result["parsed"]["facts"] == enriched_facts

    def test_prompt_override_reaches_model(self, tmp_path, monkeypatch: pytest.MonkeyPatch):
        """A prompts_dir + filename override actually drives the cloud call.

        Mirrors ``TestCalibratePlausibility::test_prompt_override_reaches_model``.
        Patches the leaf cloud call (``_filter_anthropic``), NOT
        ``_filter_with_sota`` itself, so the real ``_load_prompt`` call
        inside ``_filter_with_sota`` records onto the ``sota_enrich``
        phase trace — provenance must come from that real record, not a
        hand-built ``prompts=[...]`` literal.
        """
        import unittest.mock as _mock

        from paramem.graph.prompts import _load_prompt

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        state = _state_enabled()
        state["config"].consolidation.sota_enabled = True
        state["config"].consolidation.extraction_noise_filter = "anthropic"
        state["config"].consolidation.extraction_noise_filter_model = "claude-sonnet-4-6"
        state["config"].consolidation.extraction_noise_filter_endpoint = ""

        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        sentinel = "SENTINEL-ENRICH-OVERRIDE"
        (prompts_dir / "my_enrich.txt").write_text(
            f"{sentinel}\nfacts: {{facts_json}}\ntranscript: {{transcript}}"
        )

        req = CalibrateEnrichRequest(
            facts=[],
            transcript="[user] hello there",
            prompts_dir=str(prompts_dir),
            enrichment_prompt_filename="my_enrich.txt",
        )

        with _mock.patch(
            "paramem.graph.extractor._filter_anthropic",
            return_value='{"add": [], "modify": [], "drop": [], "bindings": {}}',
        ):
            result = calibrate_enrich(state, req)

        assert result["stage"] == "enrich"

        # Reported provenance is sourced from the real phase-trace record
        # (populated by `_load_prompt`'s own `record_prompt` call inside
        # `_filter_with_sota`), not a hand-built `prompts=[...]` literal.
        expected = _load_prompt("my_enrich.txt", prompts_dir=prompts_dir, required=True)
        expected_sha = hashlib.sha256(expected.encode("utf-8")).hexdigest()[:12]
        shas = {p["sha"] for p in result["prompts"]}
        templates = {p["template"] for p in result["prompts"]}
        assert expected_sha in shas
        assert expected in templates


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
        """Snapshot node-link edges are correctly flattened to relation dicts."""
        import json

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

        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "predicate_normalization.txt").write_text("filter {predicates_json}")

        state = _state_enabled()
        # One candidate group (Alex/Acme with works_for + employed_by) → one model call.
        import json as _json
        import unittest.mock as _mock

        filter_raw = _json.dumps({"clusters": [["works_for", "employed_by"]]})
        state["model"].gradient_checkpointing_disable = MagicMock()

        with _mock.patch(
            "paramem.graph.extractor.generate_answer",
            side_effect=[filter_raw],
        ):
            result = calibrate_normalize(
                state,
                CalibrateNormalizeRequest(
                    snapshot_path=str(snap_path),
                    prompts_dir=str(prompts_dir),
                ),
            )

        parsed = result["parsed"]
        # Two grounded links (third has no predicate and is skipped).
        assert parsed["input_count"] == 2
        assert result["stage"] == "normalize"
        # raw_output is a string (single-stage shape).
        assert isinstance(result["raw_output"], str)

        # Reported provenance is sourced from the real
        # phase-trace record (populated by `_load_prompt`'s own
        # `record_prompt` call inside `normalize_predicates`), not a
        # hand-built `_read_prompt` literal — its sha must match a live
        # `_load_prompt` computation of the same file.
        from paramem.graph.prompts import _load_prompt

        expected = _load_prompt(
            "predicate_normalization.txt", prompts_dir=prompts_dir, required=True
        )
        expected_sha = hashlib.sha256(expected.encode("utf-8")).hexdigest()[:12]
        filter_entry = next(p for p in result["prompts"] if not p["path"].endswith("_system.txt"))
        assert filter_entry["sha"] == expected_sha
        assert filter_entry["template"] == expected

    def test_n_output_tokens_positive_for_nonempty_filter_output(self, tmp_path):
        """normalize n_output_tokens counts raw output tokens (not -1).

        Two relations on the SAME (subject, object) pair with DIFFERENT
        predicates are required: normalize_predicates only calls
        generate_answer when it finds at least one candidate group (≥2 distinct
        predicates on the same s/o pair).  A single relation produces no
        candidate group → early return with empty raw_outputs → n_output_tokens=-1.
        """
        import json
        import unittest.mock as _mock

        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "predicate_normalization.txt").write_text("filter {predicates_json}")

        state = _state_enabled()
        # Tokenizer mock: _count_tokens calls tokenizer(text)["input_ids"]; the
        # dedup function also calls tokenizer.apply_chat_template — keep both.
        tok = MagicMock()
        tok.side_effect = lambda text, **kw: {"input_ids": text.split()}
        tok.apply_chat_template.return_value = "formatted-prompt"
        state["tokenizer"] = tok

        # model.gradient_checkpointing_disable() is called inside dedup primitive.
        state["model"].gradient_checkpointing_disable = MagicMock()

        filter_raw = json.dumps({"clusters": [["works_for", "employed_by"]]})

        with _mock.patch(
            "paramem.graph.extractor.generate_answer",
            side_effect=[filter_raw],
        ):
            result = calibrate_normalize(
                state,
                CalibrateNormalizeRequest(
                    # Two relations on identical (A, B) pair with distinct predicates
                    # → forms one candidate group → generate_answer is called →
                    # raw_outputs is non-empty → n_output_tokens > 0.
                    relations=[
                        {"subject": "A", "predicate": "works_for", "object": "B"},
                        {"subject": "A", "predicate": "employed_by", "object": "B"},
                    ],
                    prompts_dir=str(prompts_dir),
                ),
            )

        # n_output_tokens must reflect the model output, not be -1.
        assert result["n_output_tokens"] != -1, (
            "n_output_tokens must be a positive token count from raw model output, not -1. "
            "Verify the relations list forms a candidate group so generate_answer is called."
        )
        assert result["n_output_tokens"] > 0


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
        """SOTA stages report seed=null even when caller sends a seed."""
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

    ``_build_calibrate_response`` (the former single-caller indirection)
    has been folded directly into :func:`_run_calibration`; these tests
    exercise the envelope through a trivial ``guard``/``dispatch`` pair
    that returns a fixed ``(raw_output, parsed)`` so the assembly logic
    is pinned independent of any one handler's real behavior.  This
    proves the fold is behavior-preserving: the 11-key tail (plus
    ``phases``) matches what the six handlers previously assembled via
    the now-deleted ``_build_calibrate_response``.
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
            return raw_output, parsed

        return _run_calibration(
            stage=stage,
            entry="standalone",
            phase="anonymize",
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

    def test_missing_prompt_file_raises_400(self, tmp_path):
        """calibrate_name raises 400 when the prompt file is absent."""
        state = _state_enabled()
        # Use a prompts dir that has no name_extraction files.
        prompts_dir = tmp_path / "empty_prompts"
        prompts_dir.mkdir()
        req = CalibrateNameRequest(
            turns=[{"role": "user", "text": "Hi, I'm Riley."}],
            prompts_dir=str(prompts_dir),
        )
        with pytest.raises(HTTPException) as exc:
            calibrate_name(state, req)
        assert exc.value.status_code == 400
        assert "prompt file not found" in exc.value.detail.lower()

    def test_returns_uniform_shape(self, tmp_path):
        """calibrate_name returns the uniform calibration response shape."""
        import unittest.mock as _mock

        state = _state_enabled()
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "name_extraction_system.txt").write_text("system prompt")
        (prompts_dir / "name_extraction.txt").write_text(
            "Extract name from:\n{transcript}\nAnswer:"
        )

        req = CalibrateNameRequest(
            turns=[{"role": "user", "text": "Hi, I'm Alex."}],
            prompts_dir=str(prompts_dir),
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

        # Reported provenance is sourced from the real
        # phase-trace record (populated by `_load_prompt`'s own
        # `record_prompt` call inside `extract_name_via_llm`), not a
        # hand-built `_read_prompt` literal — its sha must match a live
        # `_load_prompt` computation of each file.
        from paramem.graph.prompts import _load_prompt

        expected_sys = _load_prompt(
            "name_extraction_system.txt", prompts_dir=prompts_dir, required=True
        )
        expected_user = _load_prompt("name_extraction.txt", prompts_dir=prompts_dir, required=True)
        expected_sys_sha = hashlib.sha256(expected_sys.encode("utf-8")).hexdigest()[:12]
        expected_user_sha = hashlib.sha256(expected_user.encode("utf-8")).hexdigest()[:12]
        sys_entry = next(p for p in result["prompts"] if p["path"].endswith("_system.txt"))
        user_entry = next(p for p in result["prompts"] if not p["path"].endswith("_system.txt"))
        assert sys_entry["sha"] == expected_sys_sha
        assert user_entry["sha"] == expected_user_sha
        assert result["parsed"]["name"] == "Alex"

    def test_filename_override_used_at_execution(self, tmp_path):
        """Filename override is what the model actually sees, not a display-only value.

        Regression guard for BLOCKING-2: before the fix, _read_prompt used the
        override for the provenance block while extract_name_via_llm hardcoded
        the default filenames — so a calibration A/B run would report file X's
        sha/content but execute file Y.
        """
        import unittest.mock as _mock

        state = _state_enabled()
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()

        # Write BOTH default and override files with distinct content.
        (prompts_dir / "name_extraction_system.txt").write_text("default system")
        (prompts_dir / "name_extraction.txt").write_text("default user {transcript}")
        (prompts_dir / "my_custom_system.txt").write_text("custom system")
        (prompts_dir / "my_custom_user.txt").write_text("custom user {transcript}")

        req = CalibrateNameRequest(
            turns=[{"role": "user", "text": "I'm Alex."}],
            prompts_dir=str(prompts_dir),
            name_prompt_filename="my_custom_user.txt",
            name_system_prompt_filename="my_custom_system.txt",
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

        # The provenance block must report the override file.
        paths_reported = {p["path"] for p in result["prompts"]}
        assert any("my_custom_system.txt" in p for p in paths_reported), (
            "provenance block must reference my_custom_system.txt"
        )
        assert any("my_custom_user.txt" in p for p in paths_reported), (
            "provenance block must reference my_custom_user.txt"
        )

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
