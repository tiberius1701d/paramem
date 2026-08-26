"""Degraded-serving and relay-fork contracts at the app layer.

These pin WHETHER the cloud leg is open under degraded serving and WHICH
fork (``_relay_route`` vs ``handle_chat``) a turn reaches — endpoint- and
state-shaped, not what a leg does once entered (that lives with the leg's
own policy tests).

Covered:

* Persist-before-resolve on the cloud-only leg: the persisted assistant
  turn keeps the raw ``speaker{N}`` token; only the returned spoken text
  has it resolved to a display name.
* ``cloud.allow_degraded_serving`` — the cloud leg is gated only when the
  server is cloud-only for an INVOLUNTARY reason.
* The degradation notice fires exactly once per conversation.
* LOCAL-mode relay fork: ``ServingPath.for_speaker(speaker_id)`` — not
  server mode alone — decides whether a turn reaches ``_relay_route`` or
  ``handle_chat``, and the relay leg's live model/tokenizer threading.

CPU-only: no model, no GPU, no network.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from peft import PeftModel

import paramem.server.app as app_module
from paramem.server.chat_result import ChatResult
from paramem.server.session_buffer import SessionBuffer


def _peft_model_mock() -> MagicMock:
    """``MagicMock(spec=PeftModel)`` -- passes the ``isinstance(model,
    PeftModel)`` precondition ``base_model_inference`` enforces
    wherever a local-mode state's reasoning generate reaches it.
    ``is_gradient_checkpointing`` defaults False and
    ``gradient_checkpointing_disable``/``_enable`` are pre-set -- dynamic
    ``__getattr__``-delegated attributes a real (wrapped) PeftModel
    exposes that ``spec`` cannot see via ``dir(PeftModel)`` (mirrors
    ``tests/server/test_gates.py::_make_mock_model``)."""
    model = MagicMock(spec=PeftModel)
    model.is_gradient_checkpointing = False
    model.gradient_checkpointing_disable = MagicMock()
    model.gradient_checkpointing_enable = MagicMock()
    return model


def _make_config() -> MagicMock:
    cfg = MagicMock()
    cfg.debug = False
    cfg.cloud.enabled = True
    cfg.cloud.allow_degraded_serving = False
    cfg.consolidation.abort_quiesce_timeout_s = 5.0
    # Shaped like the shipped deployment config (configs/server.yaml).
    cfg.sanitization.cloud_mode = "anonymize"
    cfg.sanitization.scrub = {"person name"}
    return cfg


def _make_state(tmp_path, *, mode: str = "local", cloud_only_reason=None) -> dict:
    """A minimal ``_state`` for the chat paths under test."""
    state = dict(app_module._state)
    state.update(
        {
            "config": _make_config(),
            "mode": mode,
            "cloud_only_reason": cloud_only_reason,
            "session_buffer": SessionBuffer(tmp_path / "sessions", debug=False),
            "speaker_store": None,
            "ha_client": None,
            "cloud_agent": MagicMock(),
            "cloud_providers": {},
            "model": _peft_model_mock() if mode == "local" else None,
            "tokenizer": MagicMock() if mode == "local" else None,
            "background_trainer": None,
            "relay_notice_conversations": set(),
        }
    )
    return state


# ---------------------------------------------------------------------------
# Persist-before-resolve — cloud-only leg
# ---------------------------------------------------------------------------


def test_persist_before_resolve_cloud_only_leg(tmp_path, monkeypatch):
    """Cloud-only leg: the persisted assistant turn keeps the raw
    ``speaker{N}`` token; only the returned ``spoken_text`` has it resolved
    to a display name.  Mirrors ``test_persist_before_resolve_reply_boundary``
    (local-mode leg, ``tests/server/test_voice_endpoint.py``) at the
    cloud-only call site (``app.py::_run_chat_turn``'s cloud-only branch,
    the ``resolve_speaker_tokens(cloud_text, speaker_store)`` call)."""
    state = _make_state(tmp_path, mode="cloud-only", cloud_only_reason=None)
    store = MagicMock()
    store.resolve_speaker_name.side_effect = lambda sid: {"speaker0": "Alex"}.get(sid)
    state["speaker_store"] = store
    monkeypatch.setattr(app_module, "_state", state)

    with patch.object(
        app_module,
        "_relay_route",
        return_value=ChatResult(text="speaker0 asked about the weather."),
    ):
        result, spoken = _turn("conv-persist")

    # Reply boundary: the display name, not the token, reaches the caller.
    assert spoken == "Alex asked about the weather."
    assert result.text == "speaker0 asked about the weather."

    # Persisted turn: token-space, unresolved — the buffer never sees the name.
    turns = state["session_buffer"].get_conversation_turns("conv-persist")
    assistant_turns = [t for t in turns if t["role"] == "assistant"]
    assert len(assistant_turns) == 1
    assert assistant_turns[0]["text"] == "speaker0 asked about the weather."


# ---------------------------------------------------------------------------
# Cloud door — absent agent writes no record
# ---------------------------------------------------------------------------


def test_cloud_agent_none_writes_neither_cloud_key():
    """``answer_via_cloud`` returns ``None`` immediately when ``cloud_agent``
    is absent, before any diagnostics write — there was no cloud leg to
    refuse, so neither ``cloud_egress`` nor ``cloud_refusal`` is stamped."""
    from paramem.server.egress import OutboundText, answer_via_cloud

    diagnostics: dict = {}
    outbound = OutboundText(
        "What's the population of Berlin?",
        _make_config(),
        diagnostics=diagnostics,
    )

    result = answer_via_cloud(outbound, None)

    assert result is None
    assert "cloud_egress" not in diagnostics
    assert "cloud_refusal" not in diagnostics


# ---------------------------------------------------------------------------
# Degraded serving
# ---------------------------------------------------------------------------


def _turn(
    conversation_id: str = "c1", *, speaker_id: str | None = "speaker0", text: str | None = None
):
    """Drive one turn through ``_run_chat_turn`` synchronously (no
    pytest-asyncio in-project).  ``speaker_id=None`` drives the RELAY fork
    (``ServingPath.for_speaker`` derives it internally from this value)."""
    return asyncio.run(
        app_module._run_chat_turn(
            text=text or "What's the population of Berlin?",
            conversation_id=conversation_id,
            speaker_id=speaker_id,
            speaker="Alex" if speaker_id else None,
            speaker_embedding=None,
            language="en",
            greeting_prefix=None,
        )
    )


@pytest.mark.parametrize(
    "reason",
    ["gpu_conflict", "insufficient_vram", "reload_failed", "apply_failed", "config_refused"],
)
def test_involuntary_reasons_close_the_cloud_leg(tmp_path, monkeypatch, reason):
    """The local model is gone against the operator's wishes → cloud gated."""
    monkeypatch.setattr(
        app_module,
        "_state",
        _make_state(tmp_path, mode="cloud-only", cloud_only_reason=reason),
    )

    with patch.object(app_module, "_relay_route", return_value=ChatResult(text="x")) as mock_route:
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

    with patch.object(app_module, "_relay_route", return_value=ChatResult(text="x")) as mock_route:
        _turn()

    assert mock_route.call_args.kwargs["cloud_permitted"] is True


def test_operator_opt_in_reopens_the_cloud_leg(tmp_path, monkeypatch):
    state = _make_state(tmp_path, mode="cloud-only", cloud_only_reason="gpu_conflict")
    state["config"].cloud.allow_degraded_serving = True
    monkeypatch.setattr(app_module, "_state", state)

    with patch.object(app_module, "_relay_route", return_value=ChatResult(text="x")) as mock_route:
        _turn()

    assert mock_route.call_args.kwargs["cloud_permitted"] is True


# ---------------------------------------------------------------------------
# Degradation notice — once per conversation
# ---------------------------------------------------------------------------


def test_notice_fires_once_per_conversation(tmp_path, monkeypatch):
    state = _make_state(tmp_path, mode="cloud-only", cloud_only_reason="gpu_conflict")
    state["config"].cloud.allow_degraded_serving = True
    monkeypatch.setattr(app_module, "_state", state)

    with patch.object(app_module, "_relay_route", return_value=ChatResult(text="answer")):
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

    with patch.object(app_module, "_relay_route", return_value=ChatResult(text="answer")):
        result, spoken = _turn("conv-buf")

    assert app_module._DEGRADED_SERVING_NOTICE in spoken
    assert result.text == "answer"
    turns = state["session_buffer"].get_conversation_turns("conv-buf")
    # Non-vacuous: pin that both turns (user + assistant) were actually
    # persisted, so the "not in" check below cannot pass on an empty list.
    assert len(turns) == 2
    assert all(app_module._DEGRADED_SERVING_NOTICE not in t["text"] for t in turns)


def test_no_notice_when_the_cloud_leg_is_closed(tmp_path, monkeypatch):
    """Gated: the person is not talking to a cloud model, so say nothing."""
    monkeypatch.setattr(
        app_module,
        "_state",
        _make_state(tmp_path, mode="cloud-only", cloud_only_reason="gpu_conflict"),
    )

    with patch.object(app_module, "_relay_route", return_value=ChatResult(text="answer")):
        _, spoken = _turn("conv-c")

    assert spoken == "answer"


def test_no_notice_for_deliberate_cloud_only(tmp_path, monkeypatch):
    monkeypatch.setattr(
        app_module,
        "_state",
        _make_state(tmp_path, mode="cloud-only", cloud_only_reason="explicit"),
    )

    with patch.object(app_module, "_relay_route", return_value=ChatResult(text="answer")):
        _, spoken = _turn("conv-d")

    assert spoken == "answer"


# ---------------------------------------------------------------------------
# LOCAL-mode relay fork — ServingPath derived from speaker_id, not server mode
# ---------------------------------------------------------------------------


class TestLocalModeRelayFork:
    """``_run_chat_turn`` forks on ``ServingPath.for_speaker(speaker_id)``,
    computed internally — NOT on server mode alone.  A speakerless turn on
    an otherwise-healthy LOCAL server must still reach the relay path, even
    though a mode-only check (``_state["mode"] == "cloud-only"``) would pass
    every cloud-only-mode test elsewhere in this file while missing this
    exact case.
    """

    def test_local_mode_speakerless_calls_relay_route_not_handle_chat(self, tmp_path, monkeypatch):
        monkeypatch.setattr(app_module, "_state", _make_state(tmp_path, mode="local"))

        with (
            patch.object(
                app_module, "_relay_route", return_value=ChatResult(text="relay answer")
            ) as mock_relay,
            patch.object(app_module, "handle_chat") as mock_handle_chat,
        ):
            _, spoken = _turn("conv-relay-local", speaker_id=None)

        mock_relay.assert_called_once()
        mock_handle_chat.assert_not_called()
        assert spoken.endswith("relay answer")

    def test_local_mode_resolved_speaker_calls_handle_chat_not_relay_route(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(app_module, "_state", _make_state(tmp_path, mode="local"))

        with (
            patch.object(app_module, "_relay_route") as mock_relay,
            patch.object(
                app_module, "handle_chat", return_value=ChatResult(text="handled")
            ) as mock_handle_chat,
        ):
            _turn("conv-personal-local", speaker_id="speaker0")

        mock_handle_chat.assert_called_once()
        mock_relay.assert_not_called()

    def test_local_mode_relay_turn_gets_empty_history(self, tmp_path, monkeypatch):
        """No history egress for a speakerless request, even on an
        otherwise-healthy LOCAL server."""
        state = _make_state(tmp_path, mode="local")
        monkeypatch.setattr(app_module, "_state", state)
        state["session_buffer"].append("conv-hist-local", "user", "earlier turn")
        state["session_buffer"].append("conv-hist-local", "assistant", "earlier reply")

        with patch.object(
            app_module, "_relay_route", return_value=ChatResult(text="x")
        ) as mock_relay:
            _turn("conv-hist-local", speaker_id=None)

        assert mock_relay.call_args.kwargs["history"] == []

    def test_cloud_only_personal_turn_still_passes_full_history(self, tmp_path, monkeypatch):
        """Contrast case: a RESOLVED speaker on a server-wide cloud-only
        server (condition 1, not condition 2) still gets full history —
        only ``identity_absent`` (no speaker at all) drops it."""
        state = _make_state(tmp_path, mode="cloud-only", cloud_only_reason=None)
        monkeypatch.setattr(app_module, "_state", state)
        state["session_buffer"].append(
            "conv-hist-cloud", "user", "earlier turn", speaker_id="speaker0", speaker="Alex"
        )
        state["session_buffer"].append(
            "conv-hist-cloud",
            "assistant",
            "earlier reply",
            speaker_id="speaker0",
            speaker="Alex",
        )

        with patch.object(
            app_module, "_relay_route", return_value=ChatResult(text="x")
        ) as mock_relay:
            _turn("conv-hist-cloud", speaker_id="speaker0")

        history = mock_relay.call_args.kwargs["history"]
        assert len(history) == 2


# ---------------------------------------------------------------------------
# Relay leg passes live model/tokenizer in LOCAL mode
# ---------------------------------------------------------------------------


class TestRelayLegLocalModelThreading:
    """LOCAL-mode relay dispatch (a speakerless turn on an otherwise-healthy
    server) must pass the LIVE model/tokenizer into ``_relay_route`` so its
    cloud leg can sanitize via the local anonymizer instead of skipping it —
    a personal declarative must not egress verbatim.  A server-wide
    cloud-only turn keeps ``model``/``tokenizer=None`` (no local model
    exists there)."""

    def test_local_mode_relay_receives_live_model_and_tokenizer(self, tmp_path, monkeypatch):
        state = _make_state(tmp_path, mode="local")
        monkeypatch.setattr(app_module, "_state", state)

        with patch.object(
            app_module, "_relay_route", return_value=ChatResult(text="x")
        ) as mock_relay:
            _turn("conv-model-local", speaker_id=None)

        assert mock_relay.call_args.kwargs["model"] is state["model"]
        assert mock_relay.call_args.kwargs["tokenizer"] is state["tokenizer"]

    def test_cloud_only_relay_receives_none_model_and_tokenizer(self, tmp_path, monkeypatch):
        state = _make_state(tmp_path, mode="cloud-only", cloud_only_reason=None)
        monkeypatch.setattr(app_module, "_state", state)

        with patch.object(
            app_module, "_relay_route", return_value=ChatResult(text="x")
        ) as mock_relay:
            _turn("conv-model-cloud", speaker_id="speaker0")

        assert mock_relay.call_args.kwargs["model"] is None
        assert mock_relay.call_args.kwargs["tokenizer"] is None
