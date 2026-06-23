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
    CalibrateNormalizeRequest,
    CalibrateParams,
    CalibratePlausibilityRequest,
    _effective_params,
    _preflight,
    _read_prompt,
    calibrate_anonymize,
    calibrate_extract,
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
        # Write minimal prompt files so _read_prompt doesn't raise first.
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "graph_dedup_filter.txt").write_text("filter {facts_json}")
        (prompts_dir / "graph_dedup_merge.txt").write_text("merge {clusters_json}")
        req = CalibrateNormalizeRequest(
            relations=None,
            snapshot_path=None,
            prompts_dir=str(prompts_dir),
        )
        with pytest.raises(HTTPException) as exc:
            calibrate_normalize(state, req)
        assert exc.value.status_code == 400
        assert "exactly one" in exc.value.detail.lower()

    def test_both_relations_and_snapshot_400(self, tmp_path):
        """Providing both relations and snapshot_path raises 400."""
        state = _state_enabled()
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "graph_dedup_filter.txt").write_text("filter {facts_json}")
        (prompts_dir / "graph_dedup_merge.txt").write_text("merge {clusters_json}")
        req = CalibrateNormalizeRequest(
            relations=[{"subject": "A", "predicate": "p", "object": "B"}],
            snapshot_path="/some/path.json",
            prompts_dir=str(prompts_dir),
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
        (prompts_dir / "graph_dedup_filter.txt").write_text("filter {facts_json}")
        (prompts_dir / "graph_dedup_merge.txt").write_text("merge {clusters_json}")

        state = _state_enabled()
        # Stub the model to return empty JSON so the primitive completes.
        import json as _json

        filter_raw = _json.dumps({"groups": []})
        merge_raw = _json.dumps({"merges": []})
        state["model"].generate = MagicMock()
        # generate_answer is called via tokenizer + model; mock at the module level.
        import unittest.mock as _mock

        with _mock.patch(
            "paramem.graph.extractor.generate_answer",
            side_effect=[filter_raw, merge_raw],
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
