"""Tests for the POST /debug/recall endpoint.

Covers:
- 403 when ``config.debug=False`` (gating contract).
- 503 in cloud-only mode (no local model to probe).
- 503 when model/tokenizer aren't loaded yet.
- 400 with available adapter list when ``adapter`` is unknown.
- Happy path: adapter switch is honoured, output parsed, prior adapter restored.
- ``adapter="none"`` path: ``model.disable_adapter()`` context is used.
- gradient_checkpointing_disable is called before generation (KV-cache invariant).
- BG trainer abort is invoked when the trainer is mid-run.

Tests use FastAPI TestClient with monkeypatched ``_state``; no live server, no GPU.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

import paramem.server.app as app_module


def _make_config(tmp_path: Path, debug: bool = True) -> MagicMock:
    cfg = MagicMock()
    cfg.debug = debug
    cfg.paths.data = tmp_path / "data"
    cfg.consolidation.abort_quiesce_timeout_s = 1.0
    return cfg


def _make_peft_model(active: str = "episodic", adapters=("episodic", "procedural")) -> MagicMock:
    """Build a MagicMock that passes isinstance(model, PeftModel)."""
    from peft import PeftModel

    model = MagicMock(spec=PeftModel)
    model.peft_config = {name: MagicMock() for name in adapters}
    model.active_adapter = active
    # disable_adapter must work as a context manager.
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=None)
    cm.__exit__ = MagicMock(return_value=False)
    model.disable_adapter = MagicMock(return_value=cm)
    model.gradient_checkpointing_disable = MagicMock()
    return model


def _make_state(
    tmp_path: Path,
    *,
    debug: bool = True,
    mode: str = "local",
    model_loaded: bool = True,
    active_adapter: str = "episodic",
    adapters=("episodic", "procedural"),
) -> dict:
    state: dict = {
        "config": _make_config(tmp_path, debug=debug),
        "mode": mode,
        "tokenizer": MagicMock() if model_loaded else None,
        "background_trainer": None,
    }
    state["model"] = (
        _make_peft_model(active=active_adapter, adapters=adapters) if model_loaded else None
    )
    return state


def _make_client(monkeypatch, state: dict) -> TestClient:
    monkeypatch.setattr(app_module, "_state", state)
    return TestClient(app_module.app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Gating
# ---------------------------------------------------------------------------


class TestDebugRecallGating:
    def test_debug_false_returns_403(self, tmp_path, monkeypatch):
        state = _make_state(tmp_path, debug=False)
        client = _make_client(monkeypatch, state)
        resp = client.post(
            "/debug/recall",
            json={"text": "hi", "adapter": "episodic"},
        )
        assert resp.status_code == 403
        assert resp.json()["status"] == "forbidden_not_debug"

    def test_cloud_only_returns_503(self, tmp_path, monkeypatch):
        state = _make_state(tmp_path, mode="cloud-only")
        client = _make_client(monkeypatch, state)
        resp = client.post(
            "/debug/recall",
            json={"text": "hi", "adapter": "episodic"},
        )
        assert resp.status_code == 503
        body = resp.json()
        assert body["status"] == "not_ready"
        assert "cloud-only" in body["detail"]

    def test_no_model_loaded_returns_503(self, tmp_path, monkeypatch):
        state = _make_state(tmp_path, model_loaded=False)
        client = _make_client(monkeypatch, state)
        resp = client.post(
            "/debug/recall",
            json={"text": "hi", "adapter": "episodic"},
        )
        assert resp.status_code == 503
        assert resp.json()["status"] == "not_ready"


# ---------------------------------------------------------------------------
# Adapter validation
# ---------------------------------------------------------------------------


class TestAdapterValidation:
    def test_unknown_adapter_returns_400_with_available_list(self, tmp_path, monkeypatch):
        state = _make_state(tmp_path, adapters=("episodic", "semantic"))
        client = _make_client(monkeypatch, state)
        resp = client.post(
            "/debug/recall",
            json={"text": "hi", "adapter": "interim_does_not_exist"},
        )
        assert resp.status_code == 400
        body = resp.json()
        assert body["status"] == "unknown_adapter"
        assert body["requested"] == "interim_does_not_exist"
        assert set(body["available"]) == {"episodic", "semantic", "none"}

    def test_none_adapter_is_accepted(self, tmp_path, monkeypatch):
        state = _make_state(tmp_path)
        client = _make_client(monkeypatch, state)
        with patch(
            "paramem.evaluation.recall.generate_answer",
            return_value="ok",
        ):
            resp = client.post(
                "/debug/recall",
                json={"text": "hi", "adapter": "none"},
            )
        assert resp.status_code == 200, resp.text
        assert resp.json()["adapter_active"] == "disabled"
        # disable_adapter() context manager must have been entered.
        assert state["model"].disable_adapter.called


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_returns_raw_text_and_parsed_entry(self, tmp_path, monkeypatch):
        state = _make_state(tmp_path)
        client = _make_client(monkeypatch, state)
        raw = '{"key": "graph1", "subject": "Tobias", "predicate": "lives_in", "object": "Berlin"}'
        with patch("paramem.evaluation.recall.generate_answer", return_value=raw):
            resp = client.post(
                "/debug/recall",
                json={"text": "Tell me about Tobias", "adapter": "episodic"},
            )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["text"] == raw
        assert body["adapter_active"] == "episodic"
        assert body["parsed_entry"] == {
            "key": "graph1",
            "subject": "Tobias",
            "predicate": "lives_in",
            "object": "Berlin",
        }
        assert body["latency_ms"] >= 0
        assert set(body["adapter_available"]) >= {"episodic", "procedural", "none"}

    def test_unparseable_output_yields_null_parsed_entry(self, tmp_path, monkeypatch):
        state = _make_state(tmp_path)
        client = _make_client(monkeypatch, state)
        with patch(
            "paramem.evaluation.recall.generate_answer",
            return_value="I don't know that.",
        ):
            resp = client.post(
                "/debug/recall",
                json={"text": "Q", "adapter": "episodic"},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["text"] == "I don't know that."
        assert body["parsed_entry"] is None


# ---------------------------------------------------------------------------
# Side-effect invariants
# ---------------------------------------------------------------------------


class TestSideEffectInvariants:
    def test_prior_adapter_is_restored(self, tmp_path, monkeypatch):
        state = _make_state(tmp_path, active_adapter="procedural")
        client = _make_client(monkeypatch, state)
        with (
            patch("paramem.evaluation.recall.generate_answer", return_value="ok"),
            patch("paramem.models.loader.switch_adapter") as mock_switch,
        ):
            resp = client.post(
                "/debug/recall",
                json={"text": "Q", "adapter": "episodic"},
            )
        assert resp.status_code == 200, resp.text
        # First call: switch into requested adapter; last call: restore prior.
        calls = mock_switch.call_args_list
        assert len(calls) >= 2
        assert calls[0].args[1] == "episodic"
        assert calls[-1].args[1] == "procedural"

    def test_gradient_checkpointing_disabled_before_generate(self, tmp_path, monkeypatch):
        state = _make_state(tmp_path)
        client = _make_client(monkeypatch, state)
        with patch("paramem.evaluation.recall.generate_answer", return_value="x"):
            client.post("/debug/recall", json={"text": "Q", "adapter": "episodic"})
        # Defensive disable mirrors handle_chat (inference.py:305).
        assert state["model"].gradient_checkpointing_disable.called

    def test_bg_trainer_abort_invoked_when_training(self, tmp_path, monkeypatch):
        state = _make_state(tmp_path)
        bg = MagicMock()
        bg.is_training = True
        bg.abort_for_inference = MagicMock(return_value=True)
        state["background_trainer"] = bg
        client = _make_client(monkeypatch, state)
        with patch("paramem.evaluation.recall.generate_answer", return_value="x"):
            client.post("/debug/recall", json={"text": "Q", "adapter": "episodic"})
        bg.abort_for_inference.assert_called_once()
