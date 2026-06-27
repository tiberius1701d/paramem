"""Tests for the calibration endpoint (paramem/server/calibrate.py).

Coverage:
- 404 when calibrate_endpoint_enabled is False (production-safe default)
- 503 when consolidation cycle is in flight (concurrency guard)
- 503 when local model is not loaded (cloud-only mode)
- 400 when prompt file is missing (no embedded fallback)

The tests use a MagicMock-based fake state — they do NOT load the real
Mistral model.  End-to-end live testing of the actual extraction logic is
left to the existing integration suite + manual calibration runs.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from paramem.graph.extraction_pipeline import ExtractionPipeline
from paramem.server.calibrate import (
    CalibrateAnonymizeRequest,
    CalibrateExtractRequest,
    CalibrateNameRequest,
    CalibrateNormalizeRequest,
    CalibrateParams,
    CalibratePlausibilityRequest,
    _build_calibrate_response,
    _effective_params,
    _Measurement,
    _preflight,
    _read_prompt,
    calibrate_anonymize,
    calibrate_extract,
    calibrate_name,
    calibrate_normalize,
    calibrate_plausibility,
)


def _state_disabled() -> dict:
    """Server state where calibrate is OFF (production default)."""
    consolidation_cfg = SimpleNamespace(calibrate_endpoint_enabled=False)
    config = SimpleNamespace(consolidation=consolidation_cfg)
    return {
        "config": config,
        "consolidating": False,
        "model": MagicMock(),
        "tokenizer": MagicMock(),
    }


def _state_enabled() -> dict:
    consolidation_cfg = SimpleNamespace(calibrate_endpoint_enabled=True)
    config = SimpleNamespace(consolidation=consolidation_cfg)
    return {
        "config": config,
        "consolidating": False,
        "model": MagicMock(),
        "tokenizer": MagicMock(),
        "consolidation_loop": MagicMock(),
        "model_id": "test-model",
    }


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


class TestReadPrompt:
    def test_missing_file_raises_400(self, tmp_path):
        with pytest.raises(HTTPException) as exc:
            _read_prompt(tmp_path, "does_not_exist.txt")
        assert exc.value.status_code == 400
        assert "prompt file not found" in exc.value.detail.lower()

    def test_returns_path_sha_content(self, tmp_path):
        f = tmp_path / "p.txt"
        f.write_text("hello prompt", encoding="utf-8")
        path, sha, content = _read_prompt(tmp_path, "p.txt")
        assert path == str(f)
        assert content == "hello prompt"
        assert len(sha) == 12 and all(c in "0123456789abcdef" for c in sha)

    def test_default_prompts_dir_when_none(self):
        # When prompts_dir is None, the default is configs/prompts/.
        # We exercise the path-construction branch only.
        with pytest.raises(HTTPException):
            _read_prompt(None, "does_not_exist_12345.txt")


class TestCalibrateExtract:
    def test_disabled_404(self):
        req = CalibrateExtractRequest(
            transcript="x",
            speaker_id="Speaker0",
            source_type="document",
        )
        with pytest.raises(HTTPException) as exc:
            calibrate_extract(_state_disabled(), req)
        assert exc.value.status_code == 404

    def test_consolidating_503(self):
        state = _state_enabled()
        state["consolidating"] = True
        req = CalibrateExtractRequest(
            transcript="x",
            speaker_id="Speaker0",
            source_type="document",
        )
        with pytest.raises(HTTPException) as exc:
            calibrate_extract(state, req)
        assert exc.value.status_code == 503


class TestCalibrateAnonymize:
    def test_invalid_graph_400(self, tmp_path):
        state = _state_enabled()
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "anonymization.txt").write_text("x")
        req = CalibrateAnonymizeRequest(
            graph={"not": "a valid sessiongraph"},
            transcript="x",
            prompts_dir=str(prompts_dir),
        )
        with pytest.raises(HTTPException) as exc:
            calibrate_anonymize(state, req)
        assert exc.value.status_code == 400

    def test_disabled_404(self):
        req = CalibrateAnonymizeRequest(
            graph={"session_id": "x", "timestamp": "x", "entities": [], "relations": []},
            transcript="x",
        )
        with pytest.raises(HTTPException) as exc:
            calibrate_anonymize(_state_disabled(), req)
        assert exc.value.status_code == 404


class TestCalibratePlausibility:
    def test_disabled_404(self):
        req = CalibratePlausibilityRequest(facts=[], transcript="x")
        with pytest.raises(HTTPException) as exc:
            calibrate_plausibility(_state_disabled(), req)
        assert exc.value.status_code == 404


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
        (prompts_dir / "graph_dedup_filter.txt").write_text("filter {predicates_json}")

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

    def test_n_output_tokens_positive_for_nonempty_filter_output(self, tmp_path):
        """normalize n_output_tokens counts raw output tokens (not -1).

        Two relations on the SAME (subject, object) pair with DIFFERENT
        predicates are required: dedup_synonym_predicates only calls
        generate_answer when it finds at least one candidate group (≥2 distinct
        predicates on the same s/o pair).  A single relation produces no
        candidate group → early return with empty raw_outputs → n_output_tokens=-1.
        """
        import json
        import unittest.mock as _mock

        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "graph_dedup_filter.txt").write_text("filter {predicates_json}")

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


class TestBuildCalibrateResponse:
    """Verify _build_calibrate_response emits the expected uniform envelope.

    This proves the Seam-B factoring is behavior-preserving: the 9-key tail
    matches what the five handlers previously assembled inline.
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

    def _make_measurement(self, elapsed: float = 0.5) -> _Measurement:
        m = _Measurement()
        m.vram_before = {"alloc_mib": 100.0}
        m.vram_after = {"alloc_mib": 110.0}
        m.elapsed = elapsed
        return m

    def test_all_required_keys_present(self):
        """All 11 uniform keys appear in the output."""
        state = _state_enabled()
        result = _build_calibrate_response(
            stage="anonymize",
            prompts=[{"role": "user", "path": "p.txt", "sha": "abc", "content": "hello"}],
            raw_output="some output",
            parsed={"anonymized_facts": []},
            input_prompt_text="hello",
            measurement=self._make_measurement(),
            params=CalibrateParams(),
            state=state,
        )
        assert self._required_keys_present(result)

    def _required_keys_present(self, result: dict) -> bool:
        return self._REQUIRED_KEYS.issubset(result.keys())

    def test_stage_field_matches_argument(self):
        state = _state_enabled()
        result = _build_calibrate_response(
            stage="plausibility",
            prompts=[],
            raw_output="x",
            parsed={},
            input_prompt_text="x",
            measurement=self._make_measurement(),
            params=CalibrateParams(),
            state=state,
        )
        assert result["stage"] == "plausibility"

    def test_elapsed_propagated(self):
        state = _state_enabled()
        result = _build_calibrate_response(
            stage="anonymize",
            prompts=[],
            raw_output="x",
            parsed={},
            input_prompt_text="x",
            measurement=self._make_measurement(elapsed=1.23),
            params=CalibrateParams(),
            state=state,
        )
        assert result["wall_clock_seconds"] == pytest.approx(1.23)

    def test_vram_blocks_propagated(self):
        state = _state_enabled()
        m = self._make_measurement()
        result = _build_calibrate_response(
            stage="anonymize",
            prompts=[],
            raw_output="",
            parsed={},
            input_prompt_text="",
            measurement=m,
            params=CalibrateParams(),
            state=state,
        )
        assert result["vram_before"] == m.vram_before
        assert result["vram_after"] == m.vram_after

    def test_n_output_tokens_minus1_for_empty_raw_output(self):
        """Empty raw_output → n_output_tokens == -1."""
        state = _state_enabled()
        result = _build_calibrate_response(
            stage="anonymize",
            prompts=[],
            raw_output="",
            parsed={},
            input_prompt_text="",
            measurement=self._make_measurement(),
            params=CalibrateParams(),
            state=state,
        )
        assert result["n_output_tokens"] == -1

    def test_extra_kwargs_merged(self):
        """Extra kwargs (e.g. phases=) are merged into the response."""
        state = _state_enabled()
        result = _build_calibrate_response(
            stage="extract",
            prompts=[],
            raw_output="x",
            parsed={},
            input_prompt_text="x",
            measurement=self._make_measurement(),
            params=CalibrateParams(),
            state=state,
            phases=[{"name": "local_extract", "raw_output": "x"}],
        )
        assert "phases" in result
        assert result["phases"][0]["name"] == "local_extract"

    def test_supports_seed_false_nulls_seed(self):
        """supports_seed=False causes seed=null in params_effective."""
        state = _state_enabled()
        result = _build_calibrate_response(
            stage="anonymize",
            prompts=[],
            raw_output="",
            parsed={},
            input_prompt_text="",
            measurement=self._make_measurement(),
            params=CalibrateParams(seed=99),
            state=state,
            supports_seed=False,
        )
        assert result["params_effective"]["seed"] is None


class TestExtractionPipelineKwargsSeed:
    """Verify ExtractionPipeline.kwargs passes seed through."""

    def test_seed_forwarded(self):
        from unittest.mock import MagicMock

        pipeline = ExtractionPipeline(MagicMock(), MagicMock())
        kwargs = pipeline.kwargs(seed=7, speaker_id="Speaker0")
        assert kwargs["seed"] == 7

    def test_seed_none_by_default(self):
        from unittest.mock import MagicMock

        pipeline = ExtractionPipeline(MagicMock(), MagicMock())
        kwargs = pipeline.kwargs(speaker_id="Speaker0")
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
        roles = {p["role"] for p in result["prompts"]}
        assert roles == {"system", "user"}

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
