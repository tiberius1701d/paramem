"""Cloud-egress funnel + degraded-serving contracts at the app layer.

Split out of ``tests/test_cloud_agent.py`` (which owns provider adapters and
``answer_via_cloud``'s policy matrix) because these are endpoint- and
state-shaped: they pin WHICH funnel a request reaches and WHETHER the cloud
leg is open, not what the funnel does once entered.

Covered:

* ``POST /chat`` with ``route="cloud"`` — forced routing selects the
  PROVIDER; it does not buy a policy bypass.  Both local mode and
  cloud-only mode route through ``answer_via_cloud`` — the sole
  cloud-egress funnel; cloud-only passes ``model``/``tokenizer=None`` so
  the funnel selects its cannot-anonymize (verbatim) branch instead of the
  ``cloud_mode`` policy.  This endpoint path had no test coverage at all
  before.
* ``cloud.allow_degraded_serving`` — the cloud leg is gated only when the
  server is cloud-only for an INVOLUNTARY reason.
* The degradation notice fires exactly once per conversation.

CPU-only: no model, no GPU, no network.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import paramem.server.app as app_module
from paramem.server.inference import ChatResult
from paramem.server.session_buffer import SessionBuffer


def _make_config() -> MagicMock:
    cfg = MagicMock()
    cfg.debug = False
    cfg.cloud.enabled = True
    cfg.cloud.allow_degraded_serving = False
    cfg.consolidation.abort_quiesce_timeout_s = 5.0
    return cfg


def _make_state(tmp_path, *, mode: str = "local", cloud_only_reason=None) -> dict:
    """A minimal ``_state`` for the chat paths under test."""
    state = dict(app_module._state)
    state.update(
        {
            "config": _make_config(),
            "mode": mode,
            "cloud_only_reason": cloud_only_reason,
            "session_buffer": SessionBuffer(tmp_path / "sessions", tmp_path / "state", debug=False),
            "speaker_store": None,
            "ha_client": None,
            "cloud_agent": MagicMock(),
            "cloud_providers": {},
            "model": MagicMock() if mode == "local" else None,
            "tokenizer": MagicMock() if mode == "local" else None,
            "background_trainer": None,
            "degraded_notice_conversations": set(),
        }
    )
    return state


# ---------------------------------------------------------------------------
# POST /chat with route="cloud"
# ---------------------------------------------------------------------------


def _post_chat(client, **body):
    token = getattr(app_module, "_api_token", "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return client.post("/chat", json=body, headers=headers)


class TestForcedCloudRouting:
    def test_local_mode_goes_through_the_funnel(self, tmp_path, monkeypatch):
        """``route="cloud"`` in local mode reaches ``answer_via_cloud`` with
        a live model/tokenizer — selecting the ``cloud_mode`` policy branch,
        not the cannot-anonymize branch.
        """
        monkeypatch.setattr(app_module, "_state", _make_state(tmp_path, mode="local"))

        with (
            patch.object(app_module, "_resolve_speaker", return_value=("speaker0", "Alex")),
            patch.object(
                app_module,
                "answer_via_cloud",
                return_value=ChatResult(text="cloud answer", escalated=True),
            ) as mock_funnel,
        ):
            resp = _post_chat(
                TestClient(app_module.app),
                text="What's the population of Berlin?",
                route="cloud",
            )

        assert resp.status_code == 200
        assert resp.json()["text"] == "cloud answer"
        mock_funnel.assert_called_once()
        assert mock_funnel.call_args.args[0] == "What's the population of Berlin?"
        assert mock_funnel.call_args.kwargs["model"] is not None
        assert mock_funnel.call_args.kwargs["tokenizer"] is not None

    def test_local_mode_forwards_the_personal_verdict(self, tmp_path, monkeypatch):
        """A personal turn on the forced route carries ``is_personal=True``.

        The funnel — not this branch — decides what to do with it, per
        ``cloud_mode``.  What matters here is that the verdict is computed
        and passed instead of being skipped.
        """
        monkeypatch.setattr(app_module, "_state", _make_state(tmp_path, mode="local"))

        with (
            patch.object(app_module, "_resolve_speaker", return_value=("speaker0", "Alex")),
            patch.object(app_module, "answer_via_cloud", return_value=None) as mock_funnel,
        ):
            resp = _post_chat(
                TestClient(app_module.app),
                text="Where do I live?",
                route="cloud",
            )

        assert resp.status_code == 200
        assert "unavailable" in resp.json()["text"]
        assert mock_funnel.call_args.kwargs["is_personal"] is True

    def test_local_mode_non_personal_verdict_is_false(self, tmp_path, monkeypatch):
        monkeypatch.setattr(app_module, "_state", _make_state(tmp_path, mode="local"))

        with (
            patch.object(app_module, "_resolve_speaker", return_value=("speaker0", "Alex")),
            patch.object(
                app_module,
                "answer_via_cloud",
                return_value=ChatResult(text="ok", escalated=True),
            ) as mock_funnel,
        ):
            _post_chat(
                TestClient(app_module.app),
                text="What is the boiling point of water?",
                route="cloud",
            )

        assert mock_funnel.call_args.kwargs["is_personal"] is False

    def test_cloud_only_mode_goes_through_the_same_funnel(self, tmp_path, monkeypatch):
        """Cloud-only also reaches ``answer_via_cloud`` — with no local model
        (``model``/``tokenizer=None``) so the funnel selects its
        cannot-anonymize (verbatim) branch instead of the ``cloud_mode``
        policy.  There is no separate bypass primitive on this path anymore.
        """
        monkeypatch.setattr(app_module, "_state", _make_state(tmp_path, mode="cloud-only"))

        with (
            patch.object(app_module, "_resolve_speaker", return_value=("speaker0", "Alex")),
            patch.object(
                app_module,
                "answer_via_cloud",
                return_value=ChatResult(text="cloud answer", escalated=True),
            ) as mock_funnel,
        ):
            resp = _post_chat(
                TestClient(app_module.app),
                text="What's the population of Berlin?",
                route="cloud",
            )

        assert resp.json()["text"] == "cloud answer"
        mock_funnel.assert_called_once()
        assert mock_funnel.call_args.kwargs["model"] is None
        assert mock_funnel.call_args.kwargs["tokenizer"] is None
        assert mock_funnel.call_args.kwargs["cloud_permitted"] is True

    def test_unavailable_provider_reports_the_route(self, tmp_path, monkeypatch):
        state = _make_state(tmp_path, mode="local")
        state["cloud_agent"] = None
        monkeypatch.setattr(app_module, "_state", state)

        with patch.object(app_module, "_resolve_speaker", return_value=("speaker0", "Alex")):
            resp = _post_chat(TestClient(app_module.app), text="hi", route="cloud")

        assert resp.json()["text"] == "Route 'cloud' unavailable."
        assert resp.json()["escalated"] is False


# ---------------------------------------------------------------------------
# Degraded serving
# ---------------------------------------------------------------------------


def _turn(conversation_id: str = "c1"):
    """Drive one cloud-only turn synchronously (no pytest-asyncio in-project)."""
    return asyncio.run(
        app_module._run_chat_turn(
            text="What's the population of Berlin?",
            conversation_id=conversation_id,
            speaker_id="speaker0",
            speaker="Alex",
            display_speaker="Alex",
            history=None,
            speaker_embedding=None,
            language="en",
            greeting_prefix=None,
        )
    )


@pytest.mark.parametrize(
    "reason",
    ["gpu_conflict", "insufficient_vram", "reload_failed", "apply_failed"],
)
def test_involuntary_reasons_close_the_cloud_leg(tmp_path, monkeypatch, reason):
    """The local model is gone against the operator's wishes → cloud gated."""
    monkeypatch.setattr(
        app_module,
        "_state",
        _make_state(tmp_path, mode="cloud-only", cloud_only_reason=reason),
    )

    with patch.object(
        app_module, "_cloud_only_route", return_value=ChatResult(text="x")
    ) as mock_route:
        _turn()

    assert mock_route.call_args.kwargs["cloud_permitted"] is False


@pytest.mark.parametrize("reason", ["explicit", "released", "training", "live_reload", None])
def test_deliberate_and_transient_reasons_proceed(tmp_path, monkeypatch, reason):
    """Deliberate cloud-only and transient internal states are not degraded."""
    monkeypatch.setattr(
        app_module,
        "_state",
        _make_state(tmp_path, mode="cloud-only", cloud_only_reason=reason),
    )

    with patch.object(
        app_module, "_cloud_only_route", return_value=ChatResult(text="x")
    ) as mock_route:
        _turn()

    assert mock_route.call_args.kwargs["cloud_permitted"] is True


def test_operator_opt_in_reopens_the_cloud_leg(tmp_path, monkeypatch):
    state = _make_state(tmp_path, mode="cloud-only", cloud_only_reason="gpu_conflict")
    state["config"].cloud.allow_degraded_serving = True
    monkeypatch.setattr(app_module, "_state", state)

    with patch.object(
        app_module, "_cloud_only_route", return_value=ChatResult(text="x")
    ) as mock_route:
        _turn()

    assert mock_route.call_args.kwargs["cloud_permitted"] is True


# ---------------------------------------------------------------------------
# Degradation notice — once per conversation
# ---------------------------------------------------------------------------


def test_notice_fires_once_per_conversation(tmp_path, monkeypatch):
    state = _make_state(tmp_path, mode="cloud-only", cloud_only_reason="gpu_conflict")
    state["config"].cloud.allow_degraded_serving = True
    monkeypatch.setattr(app_module, "_state", state)

    with patch.object(app_module, "_cloud_only_route", return_value=ChatResult(text="answer")):
        _, first = _turn("conv-a")
        _, second = _turn("conv-a")
        _, other = _turn("conv-b")

    assert first == f"{app_module._DEGRADED_SERVING_NOTICE}answer"
    assert second == "answer"
    # A different conversation gets its own single announcement.
    assert other == f"{app_module._DEGRADED_SERVING_NOTICE}answer"


def test_notice_is_never_written_to_the_session_buffer(tmp_path, monkeypatch):
    """App-layer prefix only — a training transcript must never carry it."""
    state = _make_state(tmp_path, mode="cloud-only", cloud_only_reason="gpu_conflict")
    state["config"].cloud.allow_degraded_serving = True
    monkeypatch.setattr(app_module, "_state", state)

    with patch.object(app_module, "_cloud_only_route", return_value=ChatResult(text="answer")):
        result, spoken = _turn("conv-buf")

    assert app_module._DEGRADED_SERVING_NOTICE in spoken
    assert result.text == "answer"
    turns = state["session_buffer"].get_session_turns("conv-buf")
    assert all(app_module._DEGRADED_SERVING_NOTICE not in t["text"] for t in turns)


def test_no_notice_when_the_cloud_leg_is_closed(tmp_path, monkeypatch):
    """Gated: the person is not talking to a cloud model, so say nothing."""
    monkeypatch.setattr(
        app_module,
        "_state",
        _make_state(tmp_path, mode="cloud-only", cloud_only_reason="gpu_conflict"),
    )

    with patch.object(app_module, "_cloud_only_route", return_value=ChatResult(text="answer")):
        _, spoken = _turn("conv-c")

    assert spoken == "answer"


def test_no_notice_for_deliberate_cloud_only(tmp_path, monkeypatch):
    monkeypatch.setattr(
        app_module,
        "_state",
        _make_state(tmp_path, mode="cloud-only", cloud_only_reason="explicit"),
    )

    with patch.object(app_module, "_cloud_only_route", return_value=ChatResult(text="answer")):
        _, spoken = _turn("conv-d")

    assert spoken == "answer"
