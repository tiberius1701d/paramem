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
from paramem.cloud.providers.base import CloudResponse
from paramem.graph.schema import Relation, SessionGraph
from paramem.server.inference import ChatResult
from paramem.server.session_buffer import SessionBuffer


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
            "session_buffer": SessionBuffer(tmp_path / "sessions", tmp_path / "state", debug=False),
            "speaker_store": None,
            "ha_client": None,
            "cloud_agent": MagicMock(),
            "cloud_providers": {},
            "model": MagicMock() if mode == "local" else None,
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

    def test_relation_free_turn_still_reaches_the_provider(self, tmp_path, monkeypatch):
        """A turn the local extractor finds no relations in is NOT a block.

        Drives the REAL funnel (``answer_via_cloud`` -> ``anonymize_turn``
        -> ``anonymize``) with only the two local LLM calls stubbed: the
        extraction pass returns an empty graph (the ordinary outcome for a
        non-personal question) and the anonymizer returns the legitimate
        "nothing in scope" verdict (empty mapping, no rewrite).  The
        provider must be called with the turn text.

        Mutation: reinstate a relation-count gate anywhere in the chain ->
        the funnel returns ``None`` -> the endpoint answers
        ``Route 'cloud' unavailable.`` -> this test fails.
        """
        state = _make_state(tmp_path, mode="local")
        state["model"].is_gradient_checkpointing = False
        agent = state["cloud_agent"]
        agent.call.return_value = CloudResponse(text="Paris.")
        monkeypatch.setattr(app_module, "_state", state)

        empty_graph = SessionGraph(session_id="cloud_egress", timestamp="2026-08-03T00:00:00Z")

        with (
            patch.object(app_module, "_resolve_speaker", return_value=("speaker0", "Alex")),
            patch("paramem.graph.flows.extract_graph", return_value=empty_graph),
            patch(
                "paramem.cloud.anonymize.anonymize_transcript",
                return_value=({}, "", "raw"),
            ),
        ):
            resp = _post_chat(
                TestClient(app_module.app),
                text="What is the capital of France?",
                route="cloud",
            )

        assert resp.status_code == 200
        assert resp.json()["text"] == "Paris."
        assert resp.json()["escalated"] is True
        agent.call.assert_called_once()
        assert agent.call.call_args.kwargs["query"] == "What is the capital of France?"

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
    ["gpu_conflict", "insufficient_vram", "reload_failed", "apply_failed"],
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


# ---------------------------------------------------------------------------
# Relay egress surfaces — what each leg is actually shown
# ---------------------------------------------------------------------------


class TestRelayLegEgressSurfaces:
    """The two relay legs carry DIFFERENT payloads, by contract.

    * **HA** — cleartext, per ``architecture.md`` AD-21 ("The HA agent must
      be local ... ParaMem sends it cleartext") and identical to the
      identified-speaker path's own HA leg
      (``paramem.server.inference._escalate_to_ha_agent``, called with the
      raw turn text).  The relay is not looser than the pre-existing
      posture — it is the same posture.
    * **CLOUD** — locally anonymized, per ``sanitization.cloud_mode``, with
      no speaker to anchor on (``speaker_id=None``).  The raw name must
      never appear in the provider payload.

    Both are asserted in ONE turn so a change that collapses the two legs
    onto a single payload cannot pass.
    """

    def test_ha_gets_cleartext_and_cloud_gets_the_anonymized_turn(self, tmp_path, monkeypatch):
        raw = "My name is Greta Feldmann and I take heart medication every morning."
        anon = "My name is Person_1 and I take heart medication every morning."

        state = _make_state(tmp_path, mode="local")
        state["model"].is_gradient_checkpointing = False
        ha_client = MagicMock()
        # HA declines this turn, so the chain continues to the cloud leg and
        # BOTH payload surfaces are observable in one pass.
        ha_client.conversation_process.return_value = None
        state["ha_client"] = ha_client
        agent = state["cloud_agent"]
        agent.call.return_value = CloudResponse(text="Noted, Person_1.")
        monkeypatch.setattr(app_module, "_state", state)

        graph = SessionGraph(
            session_id="cloud_egress",
            timestamp="2026-08-03T00:00:00Z",
            relations=[
                Relation(
                    subject="Greta Feldmann",
                    predicate="takes",
                    object="heart medication",
                    relation_type="factual",
                    confidence=1.0,
                    speaker_id="cloud_egress",
                )
            ],
        )

        with (
            patch("paramem.graph.flows.extract_graph", return_value=graph),
            patch(
                "paramem.cloud.anonymize.anonymize_transcript",
                return_value=({"Greta Feldmann": "Person_1"}, f"[user] {anon}", "raw"),
            ),
        ):
            result, _ = _turn("conv-relay-egress", speaker_id=None, text=raw)

        # HA leg — cleartext, verbatim.
        ha_client.conversation_process.assert_called_once()
        assert ha_client.conversation_process.call_args.args[0] == raw

        # Cloud leg — anonymized, and the real name never leaves.
        agent.call.assert_called_once()
        sent = agent.call.call_args.kwargs["query"]
        assert sent == anon
        assert "Greta Feldmann" not in sent
        assert agent.call.call_args.kwargs["history"] == []

        # The reply is de-anonymized locally before it reaches the caller.
        assert result.text == "Noted, Greta Feldmann."

    def test_cloud_leg_anchors_on_no_speaker(self, tmp_path, monkeypatch):
        """A relay turn has no resolved speaker, so the anonymizer's
        speaker-anchor slot must stay empty — the ``"cloud_egress"``
        extraction sentinel is not a speaker id and must never be
        forwarded as one."""
        state = _make_state(tmp_path, mode="local")
        state["model"].is_gradient_checkpointing = False
        state["ha_client"] = None
        state["cloud_agent"].call.return_value = CloudResponse(text="ok")
        monkeypatch.setattr(app_module, "_state", state)

        graph = SessionGraph(
            session_id="cloud_egress",
            timestamp="2026-08-03T00:00:00Z",
            relations=[
                Relation(
                    subject="Greta Feldmann",
                    predicate="takes",
                    object="heart medication",
                    relation_type="factual",
                    confidence=1.0,
                    speaker_id="cloud_egress",
                )
            ],
        )

        with (
            patch("paramem.graph.flows.extract_graph", return_value=graph),
            patch(
                "paramem.cloud.anonymize.anonymize_transcript",
                return_value=({}, "", "raw"),
            ) as mock_anonymizer,
        ):
            _turn(
                "conv-relay-anchor",
                speaker_id=None,
                text="My name is Greta Feldmann.",
            )

        assert mock_anonymizer.call_args.kwargs["speaker_id"] is None
