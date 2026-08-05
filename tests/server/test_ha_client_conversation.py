"""Unit tests for ``HAClient.conversation_process``'s WebSocket transport.

Fakes the ``websockets.sync.client`` transport (no real HA connection).
Covers the structured-error gate: an HA conversation-agent envelope with
``response_type == "error"`` is split by ``data.code``. The routine
semantic outcomes ``no_intent_match`` and ``no_valid_targets`` pass
through like a success (HA's own speech is the correct user-facing
answer); every other code — including a missing or unrecognized one, the
agent-side-failure class — is classified as a failure (return ``None``)
rather than passed through as successful speech.
"""

import json as json_mod
import logging

from paramem.server.tools.ha_client import HAClient


class _FakeWS:
    """Scripted WebSocket connection: replays a fixed message sequence.

    Used as both the context manager and the connection object —
    ``websockets.sync.client.connect(...)`` returns it directly, and
    ``with ... as ws:`` enters it via ``__enter__``.
    """

    def __init__(self, messages):
        self._messages = [json_mod.dumps(m) for m in messages]
        self.sent: list[dict] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def recv(self, timeout=None):
        return self._messages.pop(0)

    def send(self, payload):
        self.sent.append(json_mod.loads(payload))


def _handshake(*result_messages):
    """Auth handshake followed by the caller-supplied result message(s)."""
    return [
        {"type": "auth_required"},
        {"type": "auth_ok"},
        *result_messages,
    ]


def _envelope(speech="", response_type=None, data=None):
    """Build a ``conversation.process`` success envelope.

    Mirrors HA's ``result.response.response`` nesting, where the inner
    ``response`` carries ``response_type``, ``data`` and ``speech``.
    """
    intent_response: dict = {"speech": {"plain": {"speech": speech}}}
    if response_type is not None:
        intent_response["response_type"] = response_type
    if data is not None:
        intent_response["data"] = data
    return {
        "success": True,
        "result": {"response": {"response": intent_response}},
    }


def _make_client() -> HAClient:
    return HAClient(url="http://ha.local:8123", token="test-token")


def _patch_connect(monkeypatch, ws: _FakeWS):
    monkeypatch.setattr("websockets.sync.client.connect", lambda *a, **kw: ws)


class TestConversationProcessErrorGate:
    def test_error_response_type_returns_none(self, monkeypatch, caplog):
        ws = _FakeWS(
            _handshake(
                _envelope(
                    speech="Sorry, I had a problem talking to OpenAI: Error code: 400 - ...",
                    response_type="error",
                    data={"code": "tool_use_failed"},
                )
            )
        )
        _patch_connect(monkeypatch, ws)
        client = _make_client()

        with caplog.at_level(logging.WARNING, logger="paramem.server.tools.ha_client"):
            result = client.conversation_process("what time is it", agent_id="conversation.groq")

        assert result is None
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "tool_use_failed" in warnings[0].getMessage()

    def test_error_code_without_response_type_passes_through(self, monkeypatch):
        """``data.code`` only classifies the response when ``response_type
        == "error"`` (HA only populates ``data.code`` on error-typed
        responses in practice) — a stray code on a non-error envelope must
        not fail the call. This is the disjunction the gate used to trigger
        on before the ``response_type``-first split."""
        ws = _FakeWS(
            _handshake(
                _envelope(
                    speech="Turned on the kitchen light.",
                    response_type="action_done",
                    data={"code": "failed_to_handle"},
                )
            )
        )
        _patch_connect(monkeypatch, ws)
        client = _make_client()

        result = client.conversation_process("turn on the lights", agent_id="conversation.groq")

        assert result == "Turned on the kitchen light."

    def test_no_intent_match_returns_speech_without_warning(self, monkeypatch, caplog):
        """``no_intent_match`` is a routine semantic outcome (HA understood
        the request and found no matching intent) — HA's own speech is the
        correct user-facing answer, and no WARNING should fire for it."""
        ws = _FakeWS(
            _handshake(
                _envelope(
                    speech="Sorry, I didn't understand that.",
                    response_type="error",
                    data={"code": "no_intent_match"},
                )
            )
        )
        _patch_connect(monkeypatch, ws)
        client = _make_client()

        with caplog.at_level(logging.WARNING, logger="paramem.server.tools.ha_client"):
            result = client.conversation_process("gibberish query", agent_id="conversation.groq")

        assert result == "Sorry, I didn't understand that."
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings == []

    def test_no_valid_targets_returns_speech_without_warning(self, monkeypatch, caplog):
        """``no_valid_targets`` is a routine semantic outcome (HA understood
        the request but found no matching device/area) — same treatment as
        ``no_intent_match``."""
        ws = _FakeWS(
            _handshake(
                _envelope(
                    speech="I couldn't find that device.",
                    response_type="error",
                    data={"code": "no_valid_targets"},
                )
            )
        )
        _patch_connect(monkeypatch, ws)
        client = _make_client()

        with caplog.at_level(logging.WARNING, logger="paramem.server.tools.ha_client"):
            result = client.conversation_process(
                "turn on the nonexistent light", agent_id="conversation.groq"
            )

        assert result == "I couldn't find that device."
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings == []

    def test_error_without_code_returns_none_and_warns(self, monkeypatch, caplog):
        """An error-typed envelope with no ``data.code`` at all (not merely
        an unrecognized one) is still classified as a failure — the
        motivating incident's class, where the agent blew up without
        reporting a specific code."""
        ws = _FakeWS(
            _handshake(
                _envelope(
                    speech="Sorry, something went wrong.",
                    response_type="error",
                )
            )
        )
        _patch_connect(monkeypatch, ws)
        client = _make_client()

        with caplog.at_level(logging.WARNING, logger="paramem.server.tools.ha_client"):
            result = client.conversation_process("do a thing", agent_id="conversation.groq")

        assert result is None
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1

    def test_response_type_absent_passes_through_as_success(self, monkeypatch):
        """An envelope with no ``response_type`` key at all (not merely a
        non-error value) is not error-typed and passes through like any
        success."""
        ws = _FakeWS(_handshake(_envelope(speech="It's 42 degrees.")))
        _patch_connect(monkeypatch, ws)
        client = _make_client()

        result = client.conversation_process("what's the weather", agent_id="conversation.groq")

        assert result == "It's 42 degrees."

    def test_error_warning_body_truncated_at_cap(self, monkeypatch, caplog):
        """The WARNING log truncates the serialized envelope at 500 chars —
        a long agent-side error body must not blow up the log line."""
        long_message = "x" * 2000
        ws = _FakeWS(
            _handshake(
                _envelope(
                    speech=long_message,
                    response_type="error",
                    data={"code": "unknown"},
                )
            )
        )
        _patch_connect(monkeypatch, ws)
        client = _make_client()

        with caplog.at_level(logging.WARNING, logger="paramem.server.tools.ha_client"):
            result = client.conversation_process("do a thing", agent_id="conversation.groq")

        assert result is None
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        # The envelope-body portion of the log line is capped at 500 chars;
        # the full 2000-char speech must not appear verbatim.
        assert long_message not in warnings[0].getMessage()

    def test_success_path_emits_no_warning(self, monkeypatch, caplog):
        ws = _FakeWS(_handshake(_envelope(speech="It's 42 degrees.", response_type="query_answer")))
        _patch_connect(monkeypatch, ws)
        client = _make_client()

        with caplog.at_level(logging.WARNING, logger="paramem.server.tools.ha_client"):
            result = client.conversation_process("what's the weather", agent_id="conversation.groq")

        assert result == "It's 42 degrees."
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings == []

    def test_query_answer_with_speech_passes_through(self, monkeypatch):
        ws = _FakeWS(_handshake(_envelope(speech="It's 42 degrees.", response_type="query_answer")))
        _patch_connect(monkeypatch, ws)
        client = _make_client()

        result = client.conversation_process("what's the weather", agent_id="conversation.groq")

        assert result == "It's 42 degrees."

    def test_action_done_with_targets_data_does_not_over_trigger(self, monkeypatch):
        ws = _FakeWS(
            _handshake(
                _envelope(
                    speech="Turned on the kitchen light.",
                    response_type="action_done",
                    data={"targets": [], "success": [{"id": "light.kitchen"}], "failed": []},
                )
            )
        )
        _patch_connect(monkeypatch, ws)
        client = _make_client()

        result = client.conversation_process(
            "turn on the kitchen light", agent_id="conversation.groq"
        )

        assert result == "Turned on the kitchen light."

    def test_empty_speech_non_error_type_returns_none(self, monkeypatch):
        ws = _FakeWS(_handshake(_envelope(speech="", response_type="action_done")))
        _patch_connect(monkeypatch, ws)
        client = _make_client()

        result = client.conversation_process("do a thing", agent_id="conversation.groq")

        assert result is None

    def test_service_failure_returns_none_and_logs_error(self, monkeypatch, caplog):
        ws = _FakeWS(_handshake({"success": False, "error": {"message": "agent unavailable"}}))
        _patch_connect(monkeypatch, ws)
        client = _make_client()

        with caplog.at_level(logging.ERROR, logger="paramem.server.tools.ha_client"):
            result = client.conversation_process("anything", agent_id="conversation.groq")

        assert result is None
        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 1

    def test_null_result_returns_none_without_error(self, monkeypatch, caplog):
        """HA can send an explicit ``"result": null`` on an otherwise
        successful envelope. The outermost traversal level must be
        null-hardened like the inner ones (``response``, ``speech``), so
        this returns ``None`` via the graceful classification path — not
        via the generic ``except Exception`` handler, which would log an
        ERROR-level "HA conversation.process error"."""
        ws = _FakeWS(_handshake({"success": True, "result": None}))
        _patch_connect(monkeypatch, ws)
        client = _make_client()

        with caplog.at_level(logging.ERROR, logger="paramem.server.tools.ha_client"):
            result = client.conversation_process("anything", agent_id="conversation.groq")

        assert result is None
        error_logs = [
            r
            for r in caplog.records
            if r.levelno == logging.ERROR and "HA conversation.process error" in r.getMessage()
        ]
        assert error_logs == []

    def test_empty_agent_id_returns_none_without_connecting(self, monkeypatch):
        """Records connection attempts instead of raising inside the
        connect() call — an exception raised there is caught by
        ``conversation_process``'s own broad ``except Exception`` and
        would make this test pass even if the empty-agent-id guard were
        removed. Asserting on the recorded-attempts list, outside that
        try/except, actually detects a bypassed guard."""
        connect_attempts: list[tuple] = []

        def _record_connect(*a, **kw):
            connect_attempts.append((a, kw))
            raise AssertionError("conversation_process must not connect with an empty agent_id")

        monkeypatch.setattr("websockets.sync.client.connect", _record_connect)
        client = _make_client()

        result = client.conversation_process("anything", agent_id="")

        assert result is None
        assert connect_attempts == []
