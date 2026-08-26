"""Unit tests for cloud agent adapters (no API calls — mocked)."""

import logging
from unittest.mock import MagicMock, patch

import pytest

from paramem.cloud.admission import PROVIDER_KEY_ENV
from paramem.cloud.providers.base import CloudAgent, CloudAgentConfig, CloudResponse, ToolCall
from paramem.cloud.providers.openai_compat import OpenAICompatAgent
from paramem.cloud.providers.registry import get_cloud_agent


class TestCloudResponse:
    def test_no_tool_calls(self):
        resp = CloudResponse(text="Hello")
        assert not resp.requires_tool_execution
        assert resp.text == "Hello"

    def test_with_tool_calls(self):
        resp = CloudResponse(tool_calls=[ToolCall(id="1", name="get_weather", arguments={})])
        assert resp.requires_tool_execution


class TestOpenAICompatAdapter:
    def _make_config(self, **kwargs):
        defaults = {
            "provider": "openai",
            "model": "gpt-4o",
            "api_key": "sk-test",
            "endpoint": "https://api.openai.com/v1/chat/completions",
        }
        defaults.update(kwargs)
        return CloudAgentConfig(**defaults)

    def test_format_tools(self):
        agent = OpenAICompatAgent(self._make_config())
        standard_tools = [
            {
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            }
        ]
        formatted = agent.format_tools(standard_tools)
        assert len(formatted) == 1
        assert formatted[0]["type"] == "function"
        assert formatted[0]["function"]["name"] == "get_weather"
        assert "location" in formatted[0]["function"]["parameters"]["properties"]

    def test_parse_text_response(self):
        agent = OpenAICompatAgent(self._make_config())
        data = {
            "choices": [
                {
                    "message": {"content": "The weather is sunny."},
                    "finish_reason": "stop",
                }
            ]
        }
        resp = agent._parse_response(data)
        assert resp.text == "The weather is sunny."
        assert not resp.requires_tool_execution
        assert resp.finish_reason == "stop"

    def test_parse_tool_call_response(self):
        agent = OpenAICompatAgent(self._make_config())
        data = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_abc123",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "Berlin"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
        resp = agent._parse_response(data)
        assert resp.requires_tool_execution
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "get_weather"
        assert resp.tool_calls[0].arguments == {"location": "Berlin"}
        assert resp.tool_calls[0].id == "call_abc123"

    def test_parse_malformed_arguments(self):
        agent = OpenAICompatAgent(self._make_config())
        data = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "test",
                                    "arguments": "not valid json{",
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }
        resp = agent._parse_response(data)
        assert resp.tool_calls[0].arguments == {}

    @patch("paramem.cloud.providers.openai_compat.httpx.Client")
    def test_call_success(self, mock_client_cls):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {"content": "42 degrees"},
                    "finish_reason": "stop",
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response
        mock_client_cls.return_value = mock_client

        agent = OpenAICompatAgent(self._make_config())
        resp = agent.call("What's the temperature?")
        assert resp.text == "42 degrees"
        assert not resp.requires_tool_execution

    @patch("paramem.cloud.providers.openai_compat.httpx.Client")
    def test_call_timeout_returns_error(self, mock_client_cls):
        import httpx

        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.side_effect = httpx.TimeoutException("timeout")
        mock_client_cls.return_value = mock_client

        agent = OpenAICompatAgent(self._make_config())
        resp = agent.call("test")
        assert "couldn't reach" in resp.text

    @patch("paramem.cloud.providers.openai_compat.httpx.Client")
    def test_call_http_error_logs_response_body(self, mock_client_cls, caplog):
        """The provider HTTP error body (where a 400 like Groq's
        tool_choice/tool_use_failed rejection explains itself) must reach
        the log, not just the status code."""
        import httpx

        error_body = (
            '{"error": {"message": "Tool choice is none, but model called '
            'a tool", "code": "tool_use_failed"}}'
        )
        response = httpx.Response(
            400,
            text=error_body,
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        )
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = response
        mock_client_cls.return_value = mock_client

        agent = OpenAICompatAgent(self._make_config())
        with caplog.at_level(logging.ERROR, logger="paramem.cloud.providers.openai_compat"):
            resp = agent.call("What's the weather?")

        assert "couldn't get an answer" in resp.text
        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 1
        assert "400" in errors[0].getMessage()
        assert "tool_use_failed" in errors[0].getMessage()

    def test_default_endpoint_openai(self):
        agent = OpenAICompatAgent(self._make_config(provider="openai", endpoint=""))
        assert "openai.com" in agent._endpoint

    def test_default_endpoint_groq(self):
        agent = OpenAICompatAgent(self._make_config(provider="groq", endpoint=""))
        assert "groq.com" in agent._endpoint

    def test_custom_endpoint_overrides_default(self):
        agent = OpenAICompatAgent(
            self._make_config(endpoint="http://localhost:11434/v1/chat/completions")
        )
        assert "localhost" in agent._endpoint


class TestAnthropicAdapter:
    def _make_config(self, **kwargs):
        defaults = {"provider": "anthropic", "model": "claude-sonnet", "api_key": "sk-test"}
        defaults.update(kwargs)
        return CloudAgentConfig(**defaults)

    def test_call_api_status_error_logs_response_body(self, caplog):
        """``APIStatusError.message`` already carries "Error code: {status} -
        {body}" (verified against the installed anthropic SDK); the log line
        must surface it, not just the bare status code."""
        anthropic = pytest.importorskip("anthropic")
        import httpx

        from paramem.cloud.providers.anthropic_adapter import AnthropicAgent

        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        response = httpx.Response(400, request=request)
        body = {
            "error": {
                "message": "Tool choice is none, but model called a tool",
                "code": "tool_use_failed",
            }
        }
        error = anthropic.APIStatusError(f"Error code: 400 - {body}", response=response, body=body)

        agent = AnthropicAgent(self._make_config())
        agent._client.messages.create = MagicMock(side_effect=error)

        with caplog.at_level(logging.ERROR, logger="paramem.cloud.providers.anthropic_adapter"):
            resp = agent.call("What's the weather?")

        assert "couldn't get an answer" in resp.text
        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 1
        assert "400" in errors[0].getMessage()
        assert "tool_use_failed" in errors[0].getMessage()


class TestRegistry:
    """``get_cloud_agent`` admits solely via ``evaluate_cloud_egress``.

    The registry has no predicate of its own: the master switch
    (``cloud.enabled``, passed as ``cloud_enabled=``), the provider, the
    model and the provider's API-key ENV VAR are all checked in one place.
    ``CloudAgentConfig.api_key`` is a YAML surface only — the built agent
    carries the env-resolved key, so the credential that authenticates a
    call is the one admission checked.
    """

    @pytest.fixture(autouse=True)
    def _clear_provider_keys(self, monkeypatch):
        """No developer's real key may make a "missing key" case pass."""
        for env_name in PROVIDER_KEY_ENV.values():
            monkeypatch.delenv(env_name, raising=False)

    def test_master_switch_off_returns_none(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
        config = CloudAgentConfig(provider="openai", model="gpt-4o")
        assert get_cloud_agent(config, cloud_enabled=False) is None

    def test_openai_with_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
        config = CloudAgentConfig(provider="openai", model="gpt-4o")
        agent = get_cloud_agent(config, cloud_enabled=True)
        assert isinstance(agent, OpenAICompatAgent)
        assert agent.config.api_key == "sk-env"

    def test_groq_with_key(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk-env")
        config = CloudAgentConfig(provider="groq", model="llama-4-scout")
        agent = get_cloud_agent(config, cloud_enabled=True)
        assert isinstance(agent, OpenAICompatAgent)

    def test_missing_key_returns_none(self):
        config = CloudAgentConfig(provider="openai", model="gpt-4o")
        assert get_cloud_agent(config, cloud_enabled=True) is None

    def test_no_model_returns_none(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
        config = CloudAgentConfig(provider="openai", model="")
        assert get_cloud_agent(config, cloud_enabled=True) is None

    def test_endpoint_without_key_returns_none(self, monkeypatch):
        """A configured endpoint alone must not admit: with no key set the
        agent must not be built (it would otherwise POST
        ``Authorization: Bearer `` empty). Admission requires the key."""
        config = CloudAgentConfig(
            provider="groq",
            model="llama-3.3-70b",
            endpoint="https://api.groq.com/openai/v1/chat/completions",
        )
        assert get_cloud_agent(config, cloud_enabled=True) is None
        monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
        assert isinstance(get_cloud_agent(config, cloud_enabled=True), OpenAICompatAgent)

    def test_yaml_api_key_alone_is_not_admission(self, monkeypatch):
        """A literal key in YAML with the env var unset does NOT admit — the
        env var named in PROVIDER_KEY_ENV is the one key source."""
        config = CloudAgentConfig(provider="openai", model="gpt-4o", api_key="sk-yaml-only")
        assert get_cloud_agent(config, cloud_enabled=True) is None

    def test_unknown_provider_returns_none(self):
        config = CloudAgentConfig(provider="unknown_ai", model="test", api_key="key")
        assert get_cloud_agent(config, cloud_enabled=True) is None

    def test_anthropic_agent_created(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env")
        mock_anthropic = MagicMock()
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            # Force reimport so the adapter picks up the mock
            import sys

            mod_name = "paramem.cloud.providers.anthropic_adapter"
            sys.modules.pop(mod_name, None)

            config = CloudAgentConfig(provider="anthropic", model="claude-sonnet")
            agent = get_cloud_agent(config, cloud_enabled=True)
            assert agent is not None

            from paramem.cloud.providers.anthropic_adapter import AnthropicAgent

            assert isinstance(agent, AnthropicAgent)
            assert agent.config.api_key == "sk-env"


class TestCloudModePolicy:
    """Architecture #3: ``sanitization.cloud_mode`` selects the egress policy.

    These tests pin the dispatch in ``answer_via_cloud`` against each
    (cloud_mode, is_personal) combination, calling the door directly over
    an :class:`~paramem.server.egress.OutboundText` built for the test —
    the door's own unit-level contract, independent of routing.
    """

    def _make_cloud_agent(self):
        agent = MagicMock(spec=CloudAgent)
        agent.call.return_value = CloudResponse(text="<placeholder> answer")
        return agent

    def _config(self, cloud_mode: str):
        config = MagicMock()
        config.sanitization.cloud_mode = cloud_mode
        return config

    def _outbound(self, *, config, is_personal, history=None):
        from paramem.server.egress import OutboundText

        return OutboundText(
            "What's the population of Berlin?",
            config,
            diagnostics={},
            history=history,
            is_personal=is_personal,
        )

    # ---- block mode ----

    def test_block_mode_personal_query_blocks_cloud(self):
        from paramem.server.egress import answer_via_cloud

        cloud_agent = self._make_cloud_agent()
        outbound = self._outbound(config=self._config("block"), is_personal=True)
        result = answer_via_cloud(outbound, cloud_agent)
        cloud_agent.call.assert_not_called()
        assert result is None

    def test_block_mode_general_query_sends_verbatim(self):
        from paramem.server.egress import answer_via_cloud

        cloud_agent = self._make_cloud_agent()
        cloud_agent.call.return_value = CloudResponse(text="Berlin has 3.7M people.")
        outbound = self._outbound(config=self._config("block"), is_personal=False)
        answer_via_cloud(outbound, cloud_agent)
        cloud_agent.call.assert_called_once()
        # block mode + non-PERSONAL: text passed through unmodified.
        sent = cloud_agent.call.call_args.kwargs["query"]
        assert sent == "What's the population of Berlin?"

    def test_block_mode_general_query_calls_history_sanitizer(self):
        """``_sanitize_history`` (``paramem.server.egress._sanitize_history``)
        is content-only — it takes no ``speaker_id``.  The ``cloud_mode=
        block`` + non-PERSONAL branch of ``answer_via_cloud`` calls it with
        the history, with no speaker_id kwarg threaded through."""
        from paramem.server.egress import answer_via_cloud

        cloud_agent = self._make_cloud_agent()
        cloud_agent.call.return_value = CloudResponse(text="Berlin has 3.7M people.")
        outbound = self._outbound(
            config=self._config("block"),
            is_personal=False,
            history=[{"role": "user", "text": "hi"}],
        )
        with patch("paramem.server.egress._sanitize_history", return_value=[]) as mock_sanitize:
            answer_via_cloud(outbound, cloud_agent)
        mock_sanitize.assert_called_once()
        assert mock_sanitize.call_args.kwargs == {}
        assert mock_sanitize.call_args.args[0] == [{"role": "user", "text": "hi"}]


class TestCloudPermittedFalseRefusesEvenOnADeferral:
    """``answer_via_cloud`` still refuses on ``cloud_permitted=False`` when
    ``model``/``tokenizer`` are absent (a cloud-only deferral) — the
    ``not_permitted`` refusal is decided before ``cloud_mode``/
    ``is_personal`` are ever consulted, on every residency state."""

    def test_blocked_when_not_permitted(self):
        from paramem.server.egress import OutboundText, answer_via_cloud

        cloud_agent = MagicMock(spec=CloudAgent)
        config = MagicMock()

        outbound = OutboundText(
            "What's the population of Berlin?",
            config,
            diagnostics={},
            model=None,
            tokenizer=None,
        )
        result = answer_via_cloud(outbound, cloud_agent, cloud_permitted=False)

        assert result is None
        cloud_agent.call.assert_not_called()


class TestCloudOnlyRouteSpeakerId:
    """``_relay_route``'s ``speaker_id``/``text`` parameters must reach
    ``answer_via_cloud`` — the sole cloud-egress door, called here with
    ``model``/``tokenizer`` left ``None`` (a cloud-only deferral) — via the
    :class:`~paramem.server.egress.OutboundText` it builds.  The door still
    applies the full ``cloud_mode`` policy in that state; residency only
    decides whether the chain's self-introduction anchor can run.  History
    is still drop-gated inside the door via ``_sanitize_history``.
    """

    def test_speaker_id_reaches_the_funnel(self):
        from paramem.server.app import _relay_route
        from paramem.server.chat_result import ChatResult

        config = MagicMock()
        cloud_agent = MagicMock()

        with patch(
            "paramem.server.app.answer_via_cloud",
            return_value=ChatResult(text="Berlin has 3.7M people."),
        ) as mock_funnel:
            _relay_route(
                text="What's the population of Berlin?",
                history=[{"role": "user", "text": "hi"}],
                config=config,
                cloud_permitted=True,
                ha_client=None,
                cloud_agent=cloud_agent,
                speaker_id="spk-test",
            )

        mock_funnel.assert_called_once()
        outbound = mock_funnel.call_args.args[0]
        assert outbound.speaker_id == "spk-test"
        assert outbound.model is None
        assert outbound.tokenizer is None


class TestRelayRoutePersonalVerdictAndDisplayName:
    """The personal verdict is computed for EVERY turn that can reach
    the cloud leg, not only under ``identity_absent`` — and the resolved
    display name is threaded to the funnel exactly as the forced route
    already supplies it: one scrub surface per speaker, whichever door
    the turn came through.
    """

    def _config(self):
        config = MagicMock()
        config.personal_referent = None
        return config

    def test_a_resolved_speaker_personal_turn_passes_a_computed_true_verdict(self):
        from paramem.server.app import _relay_route
        from paramem.server.chat_result import ChatResult

        cloud_agent = MagicMock()

        with (
            patch("paramem.server.app.is_self_referential", return_value=True),
            patch(
                "paramem.server.app.answer_via_cloud", return_value=ChatResult(text="ok")
            ) as mock_funnel,
        ):
            _relay_route(
                text="Where do I live?",
                history=None,
                config=self._config(),
                cloud_permitted=True,
                ha_client=None,
                cloud_agent=cloud_agent,
                speaker="Alex",
                speaker_id="speaker0",
                identity_absent=False,
            )

        mock_funnel.assert_called_once()
        assert mock_funnel.call_args.args[0].is_personal is True

    def test_a_resolved_speaker_non_personal_turn_passes_false(self):
        from paramem.server.app import _relay_route
        from paramem.server.chat_result import ChatResult

        cloud_agent = MagicMock()

        with (
            patch("paramem.server.app.is_self_referential", return_value=False),
            patch(
                "paramem.server.app.answer_via_cloud", return_value=ChatResult(text="ok")
            ) as mock_funnel,
        ):
            _relay_route(
                text="What's the weather like?",
                history=None,
                config=self._config(),
                cloud_permitted=True,
                ha_client=None,
                cloud_agent=cloud_agent,
                speaker="Alex",
                speaker_id="speaker0",
                identity_absent=False,
            )

        assert mock_funnel.call_args.args[0].is_personal is False

    def test_the_resolved_display_name_reaches_the_funnel(self):
        from paramem.server.app import _relay_route
        from paramem.server.chat_result import ChatResult

        cloud_agent = MagicMock()

        with (
            patch("paramem.server.app.is_self_referential", return_value=False),
            patch(
                "paramem.server.app.answer_via_cloud", return_value=ChatResult(text="ok")
            ) as mock_funnel,
        ):
            _relay_route(
                text="hi",
                history=None,
                config=self._config(),
                cloud_permitted=True,
                ha_client=None,
                cloud_agent=cloud_agent,
                speaker="Alex",
                speaker_id="speaker0",
                identity_absent=False,
            )

        assert mock_funnel.call_args.args[0].speaker == "Alex"


class TestNoIdentityShortCircuitDoesNotWiden:
    """The personal verdict is computed for every turn, but the
    no-identity short-circuit's CALL stays conjuncted with
    ``identity_absent`` — a resolved-speaker personal turn must reach the
    normal HA/cloud dispatch, never the canned no-identity response.
    """

    def test_resolved_speaker_personal_interrogative_reaches_the_cloud_leg(self):
        from paramem.server.app import _relay_route
        from paramem.server.chat_result import ChatResult

        config = MagicMock()
        config.personal_referent = None
        cloud_agent = MagicMock()

        with (
            patch("paramem.server.app.is_self_referential", return_value=True),
            patch(
                "paramem.server.inference._is_personal_interrogative", return_value=True
            ) as mock_interrogative,
            patch(
                "paramem.server.app.answer_via_cloud",
                return_value=ChatResult(text="cloud answer"),
            ) as mock_funnel,
        ):
            result = _relay_route(
                text="Where do I live?",
                history=None,
                config=config,
                cloud_permitted=True,
                ha_client=None,
                cloud_agent=cloud_agent,
                speaker="Alex",
                speaker_id="speaker0",
                identity_absent=False,
            )

        mock_interrogative.assert_not_called()
        mock_funnel.assert_called_once()
        assert result.text == "cloud answer"

    def test_identity_absent_personal_interrogative_still_short_circuits(self):
        from paramem.server.app import _relay_route

        config = MagicMock()
        config.personal_referent = None
        config.abstention.load_no_identity_response.return_value = "I don't know who you are."
        cloud_agent = MagicMock()

        with (
            patch("paramem.server.app.is_self_referential", return_value=True),
            patch("paramem.server.inference._is_personal_interrogative", return_value=True),
            patch("paramem.server.app.answer_via_cloud") as mock_funnel,
        ):
            result = _relay_route(
                text="Where do I live?",
                history=None,
                config=config,
                cloud_permitted=True,
                ha_client=None,
                cloud_agent=cloud_agent,
                speaker=None,
                speaker_id=None,
                identity_absent=True,
            )

        mock_funnel.assert_not_called()
        assert result.text == "I don't know who you are."


class TestRelayRouteDiagnosticsCarrier:
    """Every result ``_relay_route`` returns carries the turn's threaded
    diagnostics dict — empty when the cloud leg was never reached (HA
    answered, or the no-identity short-circuit fired), and populated with
    the funnel's own record when it was."""

    def test_no_identity_short_circuit_result_carries_no_cloud_keys(self):
        from paramem.server.app import _relay_route

        config = MagicMock()
        config.personal_referent = None
        config.abstention.load_no_identity_response.return_value = "I don't know who you are."

        with (
            patch("paramem.server.app.is_self_referential", return_value=True),
            patch("paramem.server.inference._is_personal_interrogative", return_value=True),
        ):
            result = _relay_route(
                text="Where do I live?",
                history=None,
                config=config,
                cloud_permitted=True,
                ha_client=None,
                cloud_agent=MagicMock(),
                identity_absent=True,
            )

        assert result.diagnostics == {}

    def test_a_cloud_refusal_carries_the_cause_on_the_canned_limited_mode_result(self):
        from paramem.server.app import _relay_route

        config = MagicMock()
        config.personal_referent = None

        def fake_funnel(outbound, cloud_agent, **kwargs):
            outbound.diagnostics["cloud_refusal"] = "personal_blocked"
            return None

        with patch("paramem.server.app.answer_via_cloud", side_effect=fake_funnel):
            result = _relay_route(
                text="Where do I live?",
                history=None,
                config=config,
                cloud_permitted=True,
                ha_client=None,
                cloud_agent=MagicMock(),
            )

        assert result.diagnostics == {"cloud_refusal": "personal_blocked"}

    def test_a_cloud_answer_carries_the_egress_record(self):
        from paramem.server.app import _relay_route
        from paramem.server.chat_result import ChatResult

        config = MagicMock()
        config.personal_referent = None

        def fake_funnel(outbound, cloud_agent, **kwargs):
            outbound.diagnostics["cloud_egress"] = "scrubbed"
            return ChatResult(text="cloud answer", diagnostics={})

        with patch("paramem.server.app.answer_via_cloud", side_effect=fake_funnel):
            result = _relay_route(
                text="hi",
                history=None,
                config=config,
                cloud_permitted=True,
                ha_client=None,
                cloud_agent=MagicMock(),
            )

        assert result.text == "cloud answer"
        assert result.diagnostics == {"cloud_egress": "scrubbed"}


class TestDegradedServingGate:
    """``cloud.allow_degraded_serving`` closes the CLOUD leg only.

    The HA leg carries no ParaMem-held knowledge and runs on the user's own
    network, so it stays open in every degraded state. The HA door itself
    (its scrub, its own refusal causes) is exercised by dedicated egress
    tests; here it is stubbed to isolate ``_relay_route``'s HA-then-cloud
    dispatch from the door's own tagger dependency.
    """

    def _run(self, *, cloud_permitted, ha_answers):
        from paramem.server.app import _relay_route
        from paramem.server.chat_result import ChatResult

        ha_client = MagicMock()
        ha_client.conversation_process.return_value = "HA answer" if ha_answers else None

        def fake_ha_door(outbound, client, *, ha_graph=None):
            if client is None:
                return None
            reply = client.conversation_process(outbound.text, agent_id=outbound.config.ha_agent_id)
            return None if reply is None else ChatResult(text=reply, escalated=True)

        # Real (non-Mock) ha_agent_id: the HA leg's disposal in this test is
        # stated explicitly rather than inherited from a MagicMock's truthy
        # default.
        config = MagicMock()
        config.ha_agent_id = "conversation.test_agent"

        with (
            patch("paramem.server.app.answer_via_ha", side_effect=fake_ha_door),
            patch(
                "paramem.server.app.answer_via_cloud",
                return_value=ChatResult(text="cloud answer"),
            ) as mock_funnel,
        ):
            result = _relay_route(
                text="What's the population of Berlin?",
                history=None,
                config=config,
                cloud_permitted=cloud_permitted,
                ha_client=ha_client,
                cloud_agent=MagicMock(),
            )
        return result, mock_funnel, ha_client

    def test_gated_keeps_ha_leg(self):
        result, mock_funnel, ha_client = self._run(cloud_permitted=False, ha_answers=True)
        ha_client.conversation_process.assert_called_once()
        mock_funnel.assert_not_called()
        assert result.text == "HA answer"

    def test_permitted_opens_cloud_leg(self):
        result, mock_funnel, _ = self._run(cloud_permitted=True, ha_answers=False)
        mock_funnel.assert_called_once()
        assert mock_funnel.call_args.kwargs["cloud_permitted"] is True
        assert result.text == "cloud answer"


class TestCloudPermittedStillThreadedToTheFunnelOnDecline:
    """``_relay_route`` always threads ``cloud_permitted`` to
    :func:`~paramem.server.egress.answer_via_cloud`, which decides the
    ``not_permitted`` refusal itself before ever reading ``cloud_mode`` or
    building a scrub contract.  When HA declines to answer, the turn still
    reaches the REAL cloud door (not mocked here) and the ``not_permitted``
    cause lands in the turn's diagnostics."""

    def test_cloud_permitted_false_and_ha_declining_reaches_the_funnel(self):
        from paramem.server.app import _relay_route
        from paramem.server.chat_result import ChatResult

        ha_client = MagicMock()
        ha_client.conversation_process.return_value = None
        cloud_agent = MagicMock()
        # Real (non-Mock) ha_agent_id: the HA leg's disposal in this test is
        # stated explicitly rather than inherited from a MagicMock's truthy
        # default.
        config = MagicMock()
        config.ha_agent_id = "conversation.test_agent"

        def fake_ha_door(outbound, client, *, ha_graph=None):
            if client is None:
                return None
            reply = client.conversation_process(outbound.text, agent_id=outbound.config.ha_agent_id)
            return None if reply is None else ChatResult(text=reply, escalated=True)

        with patch("paramem.server.app.answer_via_ha", side_effect=fake_ha_door):
            result = _relay_route(
                text="What's the population of Berlin?",
                history=None,
                config=config,
                cloud_permitted=False,
                ha_client=ha_client,
                cloud_agent=cloud_agent,
                model=None,
                tokenizer=None,
            )

        ha_client.conversation_process.assert_called_once()
        cloud_agent.call.assert_not_called()
        assert result.diagnostics == {"cloud_refusal": "not_permitted"}
