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
from peft import PeftModel

import paramem.server.app as app_module
from paramem.server.inference import ChatResult
from paramem.server.session_buffer import SessionBuffer


def _peft_model_mock() -> MagicMock:
    """``MagicMock(spec=PeftModel)`` -- passes the ``isinstance(model,
    PeftModel)`` precondition ``base_model_inference`` now enforces
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
    # Shaped like the shipped deployment config (configs/server.yaml), so the
    # tests below that drive the REAL funnel select the anonymize policy
    # instead of falling into ``answer_via_cloud``'s unknown-value guard
    # (which maps a bare MagicMock to the safest mode, "block").
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
# POST /chat with route="cloud"
# ---------------------------------------------------------------------------


def _post_chat(client, **body):
    # Auth is OFF: _make_state's dict carries no "user_token_store" key, so
    # BearerTokenMiddleware's getter returns None and no header is needed.
    return client.post("/chat", json=body, headers={})


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

    def test_forced_cloud_route_holds_gpu_lock(self, tmp_path, monkeypatch):
        """The local-mode forced-cloud dispatch reaches the live model (the
        anonymizer's extract_graph/anonymize_turn calls generate() under
        base_model_inference), so it needs the same GPU discipline as the
        routed local path: abort in-flight background training, then hold
        the shared GPU thread lock for the duration of the dispatch."""
        from paramem.server.gpu_lock import _gpu_thread_lock

        monkeypatch.setattr(app_module, "_state", _make_state(tmp_path, mode="local"))

        call_order = []

        def fake_abort():
            call_order.append("abort")

        def fake_funnel(*args, **kwargs):
            call_order.append(("dispatch", _gpu_thread_lock.locked()))
            return ChatResult(text="cloud answer", escalated=True)

        with (
            patch.object(app_module, "_resolve_speaker", return_value=("speaker0", "Alex")),
            patch.object(
                app_module, "_abort_background_training_for_inference", side_effect=fake_abort
            ) as mock_abort,
            patch.object(app_module, "answer_via_cloud", side_effect=fake_funnel),
        ):
            resp = _post_chat(
                TestClient(app_module.app),
                text="What's the population of Berlin?",
                route="cloud",
            )

        assert resp.status_code == 200
        mock_abort.assert_called_once()
        assert call_order == ["abort", ("dispatch", True)], (
            "the bg-training abort must run before the dispatch, and the "
            "dispatch must run with the GPU thread lock held"
        )
        assert not _gpu_thread_lock.locked(), "the lock must be released after the dispatch"

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

    def test_forced_cloud_route_resolves_speaker_tokens_before_returning(
        self, tmp_path, monkeypatch
    ):
        """The shared forced-routing exit (``if result and result.text``)
        resolves any ``speaker{N}`` token in the funnel's answer before it
        reaches the caller — the same reply-boundary contract as every
        other exit."""
        state = _make_state(tmp_path, mode="local")
        store = MagicMock()
        store.resolve_speaker_name.side_effect = lambda sid: {"speaker1": "Bob"}.get(sid)
        state["speaker_store"] = store
        monkeypatch.setattr(app_module, "_state", state)

        with (
            patch.object(app_module, "_resolve_speaker", return_value=("speaker0", "Alex")),
            patch.object(
                app_module,
                "answer_via_cloud",
                return_value=ChatResult(text="speaker1 asked that too.", escalated=True),
            ),
        ):
            resp = _post_chat(TestClient(app_module.app), text="hi", route="cloud")

        assert resp.json()["text"] == "Bob asked that too."

    def test_forced_route_relay_speaker_gets_empty_history(self, tmp_path, monkeypatch):
        """RELAY on the forced route (no speaker resolved at all) does not
        buy a history-egress bypass — ``_forced_history`` is ``[]`` exactly
        like the normal ``/chat`` fork, even though forced routing selects
        the PROVIDER directly."""
        state = _make_state(tmp_path, mode="local")
        monkeypatch.setattr(app_module, "_state", state)
        state["session_buffer"].append("conv-forced-relay", "user", "earlier turn")
        state["session_buffer"].append("conv-forced-relay", "assistant", "earlier reply")

        with (
            patch.object(app_module, "_resolve_speaker", return_value=(None, None)),
            patch.object(
                app_module,
                "answer_via_cloud",
                return_value=ChatResult(text="cloud answer", escalated=True),
            ) as mock_funnel,
        ):
            _post_chat(
                TestClient(app_module.app),
                text="What's the population of Berlin?",
                route="cloud",
                conversation_id="conv-forced-relay",
            )

        assert mock_funnel.call_args.kwargs["history"] == []

    def test_forced_ha_route_resolves_speaker_tokens_before_returning(self, tmp_path, monkeypatch):
        """Same shared exit, reached via the ``route="ha"`` branch instead
        of ``route="cloud"`` — both funnel into the one
        ``resolve_speaker_tokens`` call at the bottom of the forced-routing
        block."""
        state = _make_state(tmp_path, mode="local")
        store = MagicMock()
        store.resolve_speaker_name.side_effect = lambda sid: {"speaker1": "Bob"}.get(sid)
        state["speaker_store"] = store
        ha_client = MagicMock()
        ha_client.conversation_process.return_value = "speaker1 asked that too."
        state["ha_client"] = ha_client
        monkeypatch.setattr(app_module, "_state", state)

        with patch.object(app_module, "_resolve_speaker", return_value=("speaker0", "Alex")):
            resp = _post_chat(TestClient(app_module.app), text="hi", route="ha")

        assert resp.json()["text"] == "Bob asked that too."


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
    an otherwise-healthy LOCAL server must still reach the relay path.  A
    regression that reverts the fork to a mode-only check
    (``_state["mode"] == "cloud-only"``) would pass every pre-existing
    cloud-only-mode test in this file while silently breaking this exact
    case — these tests exist to catch that regression.
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
# Relay leg passes live model/tokenizer in LOCAL mode (owner-ruled fix)
# ---------------------------------------------------------------------------


class TestRelayLegLocalModelThreading:
    """LOCAL-mode relay dispatch (a speakerless turn on an otherwise-healthy
    server) must pass the LIVE model/tokenizer into ``_relay_route`` so its
    cloud leg can sanitize via the local anonymizer instead of skipping it —
    the owner-ruled fix for personal declaratives egressing verbatim.  A
    server-wide cloud-only turn keeps ``model``/``tokenizer=None`` (no local
    model exists there)."""

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
