"""Unit tests for cloud agent adapters (no API calls — mocked)."""

from unittest.mock import MagicMock, patch

from paramem.server.cloud.base import CloudAgent, CloudResponse, ToolCall
from paramem.server.cloud.openai_compat import OpenAICompatAgent
from paramem.server.cloud.registry import get_cloud_agent
from paramem.server.config import CloudAgentConfig


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
            "enabled": True,
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

    @patch("paramem.server.cloud.openai_compat.httpx.Client")
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

    @patch("paramem.server.cloud.openai_compat.httpx.Client")
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


class TestRegistry:
    def test_disabled_returns_none(self):
        config = CloudAgentConfig(enabled=False)
        assert get_cloud_agent(config) is None

    def test_openai_with_key(self):
        config = CloudAgentConfig(
            enabled=True, provider="openai", model="gpt-4o", api_key="sk-test"
        )
        agent = get_cloud_agent(config)
        assert isinstance(agent, OpenAICompatAgent)

    def test_groq_with_key(self):
        config = CloudAgentConfig(
            enabled=True, provider="groq", model="llama-4-scout", api_key="gsk-test"
        )
        agent = get_cloud_agent(config)
        assert isinstance(agent, OpenAICompatAgent)

    def test_missing_key_returns_none(self):
        config = CloudAgentConfig(enabled=True, provider="openai", model="gpt-4o", api_key="")
        assert get_cloud_agent(config) is None

    def test_unknown_provider_returns_none(self):
        config = CloudAgentConfig(enabled=True, provider="unknown_ai", model="test", api_key="key")
        assert get_cloud_agent(config) is None

    def test_anthropic_agent_created(self):
        mock_anthropic = MagicMock()
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            # Force reimport so the adapter picks up the mock
            import sys

            mod_name = "paramem.server.cloud.anthropic_adapter"
            sys.modules.pop(mod_name, None)

            config = CloudAgentConfig(
                enabled=True, provider="anthropic", model="claude-sonnet", api_key="key"
            )
            agent = get_cloud_agent(config)
            assert agent is not None

            from paramem.server.cloud.anthropic_adapter import AnthropicAgent

            assert isinstance(agent, AnthropicAgent)


class TestPrivacyRouting:
    """Integration tests verifying personal queries never reach the cloud agent.

    These test the routing logic in handle_chat: queries containing known
    graph entities must be handled locally, never forwarded to cloud.
    """

    def _make_mock_cloud_agent(self):
        """Create a mock cloud agent that tracks whether call() was invoked."""
        agent = MagicMock(spec=CloudAgent)
        agent.call.return_value = CloudResponse(text="cloud answer")
        agent.is_available.return_value = True
        return agent

    def _make_mock_router(self, known_entities=None):
        """Create a mock router with configurable entity matching.

        Mirrors production classify_intent: PA hit → PERSONAL, no hit → UNKNOWN
        (residual would normally be GENERAL with config; UNKNOWN is the
        no-config default and produces identical dispatch behaviour to
        GENERAL).
        """
        from paramem.server.router import Intent, RoutingPlan, RoutingStep

        router = MagicMock()
        known = {e.lower() for e in (known_entities or [])}

        def route(text, speaker=None, speaker_id=None):
            text_lower = text.lower()
            matched = [e for e in known if e in text_lower]
            if speaker and speaker.lower() in known:
                matched.append(speaker.lower())
            if matched:
                return RoutingPlan(
                    steps=[
                        RoutingStep(
                            adapter_name="episodic",
                            keys_to_probe=["graph1"],
                        )
                    ],
                    strategy="entity",
                    matched_entities=matched,
                    match_source="pa",
                    intent=Intent.PERSONAL,
                )
            return RoutingPlan(strategy="direct", match_source="none", intent=Intent.UNKNOWN)

        router.route = route
        return router

    def _make_ha_only_router(self, imperative=False):
        """Create a mock router returning an HA-only match (no PA steps).

        HA hit without PA → COMMAND under classify_intent's state-first dispatch.
        """
        from paramem.server.router import Intent, RoutingPlan

        router = MagicMock()

        def route(text, speaker=None, speaker_id=None):
            return RoutingPlan(
                steps=[],
                strategy="entity",
                matched_entities=["lights"],
                match_source="ha",
                imperative=imperative,
                ha_domains=["light"],
                intent=Intent.COMMAND,
            )

        router.route = route
        return router

    def _make_both_match_router(self):
        """Create a router returning a 'both' match (PA steps + HA entity).

        PA hit + HA hit → PERSONAL (PA wins under classify_intent), so the
        privacy invariant applies: SOTA must never be reached for queries
        from this router.
        """
        from paramem.server.router import Intent, RoutingPlan, RoutingStep

        router = MagicMock()

        def route(text, speaker=None, speaker_id=None):
            return RoutingPlan(
                steps=[RoutingStep(adapter_name="episodic", keys_to_probe=["graph1"])],
                strategy="entity",
                matched_entities=["lights", "alex"],
                match_source="both",
                imperative=True,
                ha_domains=["light"],
                intent=Intent.PERSONAL,
            )

        router.route = route
        return router

    def test_personal_query_never_reaches_cloud(self):
        """Query mentioning a known entity must NOT call cloud agent."""
        from paramem.server.inference import handle_chat

        cloud_agent = self._make_mock_cloud_agent()
        router = self._make_mock_router(known_entities=["Jordan", "Berlin"])

        # Mock model and tokenizer — _probe_and_reason will be called
        # but we mock probe_key to return a fact
        model = MagicMock()
        model.gradient_checkpointing_disable = MagicMock()
        model.peft_config = {"episodic": MagicMock()}
        tokenizer = MagicMock()

        config = MagicMock()
        config.registry_path = MagicMock()
        config.registry_path.exists.return_value = False
        config.voice.load_prompt.return_value = "You are a helper."
        with (
            patch(
                "paramem.training.indexed_memory.probe_key",
                return_value={"answer": "Jordan lives in Berlin"},
            ),
            patch(
                "paramem.models.loader.switch_adapter",
            ),
            patch(
                "paramem.server.inference.generate_answer",
                return_value="Jordan lives in Berlin.",
            ),
            patch(
                "paramem.server.inference.detect_escalation",
                return_value=(False, ""),
            ),
            patch(
                "paramem.server.inference.adapt_messages",
                side_effect=lambda msgs, tok: msgs,
            ),
            patch.object(
                tokenizer,
                "apply_chat_template",
                return_value="prompt",
            ),
        ):
            result = handle_chat(
                text="Where does Jordan live?",
                conversation_id="test",
                speaker=None,
                history=None,
                model=model,
                tokenizer=tokenizer,
                config=config,
                router=router,
            )

        # Cloud agent must NOT have been called
        cloud_agent.call.assert_not_called()
        assert "Berlin" in result.text

    def test_non_personal_query_goes_to_cloud(self):
        """Query with no entity match → HA first (None) → SOTA fallback."""
        from paramem.server.inference import handle_chat

        cloud_agent = self._make_mock_cloud_agent()
        router = self._make_mock_router(known_entities=["Jordan"])

        model = MagicMock()
        model.gradient_checkpointing_disable = MagicMock()
        tokenizer = MagicMock()

        config = MagicMock()
        config.voice.load_prompt.return_value = "You are a helper."
        config.sanitization.mode = "off"

        # Mock HA client that returns None (simulates HA unavailable)
        ha_client = MagicMock()
        ha_client.conversation_process.return_value = None

        result = handle_chat(
            text="What is the weather today?",
            conversation_id="test",
            speaker=None,
            history=None,
            model=model,
            tokenizer=tokenizer,
            config=config,
            router=router,
            ha_client=ha_client,
            sota_agent=cloud_agent,
        )

        # HA was attempted first and returned None
        ha_client.conversation_process.assert_called_once()
        # SOTA agent called as fallback
        cloud_agent.call.assert_called_once()
        assert result.escalated is True
        assert result.text == "cloud answer"

    def test_no_cloud_falls_back_to_local(self):
        """Without cloud agent, non-personal query uses local model."""
        from paramem.server.inference import handle_chat

        router = self._make_mock_router(known_entities=["Jordan"])

        model = MagicMock()
        model.gradient_checkpointing_disable = MagicMock()
        tokenizer = MagicMock()

        config = MagicMock()
        config.registry_path = MagicMock()
        config.voice.load_prompt.return_value = "You are a helper."

        with (
            patch(
                "paramem.server.inference.generate_answer",
                return_value="I'm not sure about that.",
            ),
            patch(
                "paramem.server.inference.detect_escalation",
                return_value=(False, ""),
            ),
            patch(
                "paramem.server.inference.adapt_messages",
                side_effect=lambda msgs, tok: msgs,
            ),
            patch.object(
                tokenizer,
                "apply_chat_template",
                return_value="prompt",
            ),
        ):
            result = handle_chat(
                text="What is the weather today?",
                conversation_id="test",
                speaker=None,
                history=None,
                model=model,
                tokenizer=tokenizer,
                config=config,
                router=router,
            )

        assert result.escalated is False

    def test_imperative_ha_command_sota_fallback(self):
        """Path 2a (match_source=ha): HA fails → SOTA fallback."""
        from paramem.server.inference import handle_chat

        cloud_agent = self._make_mock_cloud_agent()
        router = self._make_ha_only_router(imperative=True)

        model = MagicMock()
        model.gradient_checkpointing_disable = MagicMock()
        tokenizer = MagicMock()

        config = MagicMock()
        config.sanitization.mode = "off"
        config.voice.load_prompt.return_value = "You are a helper."

        ha_client = MagicMock()
        ha_client.conversation_process.return_value = None  # HA fails

        result = handle_chat(
            text="Turn on the lights",
            conversation_id="test",
            speaker=None,
            history=None,
            model=model,
            tokenizer=tokenizer,
            config=config,
            router=router,
            ha_client=ha_client,
            sota_agent=cloud_agent,
        )

        ha_client.conversation_process.assert_called_once()
        cloud_agent.call.assert_called_once()
        assert result.escalated is True
        assert result.text == "cloud answer"

    def test_ha_nonimperative_match_sota_fallback(self):
        """Path 2c (non-imperative HA match): HA fails → SOTA fallback."""
        from paramem.server.inference import handle_chat

        cloud_agent = self._make_mock_cloud_agent()
        router = self._make_ha_only_router(imperative=False)

        model = MagicMock()
        model.gradient_checkpointing_disable = MagicMock()
        tokenizer = MagicMock()

        config = MagicMock()
        config.sanitization.mode = "off"
        config.voice.load_prompt.return_value = "You are a helper."

        ha_client = MagicMock()
        ha_client.conversation_process.return_value = None  # HA fails

        result = handle_chat(
            text="Is the light on?",
            conversation_id="test",
            speaker=None,
            history=None,
            model=model,
            tokenizer=tokenizer,
            config=config,
            router=router,
            ha_client=ha_client,
            sota_agent=cloud_agent,
        )

        ha_client.conversation_process.assert_called_once()
        cloud_agent.call.assert_called_once()
        assert result.escalated is True
        assert result.text == "cloud answer"

    def test_personal_both_match_uses_pa_probe_no_pre_flight_ha(self):
        """match_source=both + intent=PERSONAL: PA probe runs directly.

        Under intent-keyed dispatch, intent=PERSONAL routes straight to the
        local PA probe.  The pre-flight HA call from the legacy cascade is
        gone — HA is reachable only via [ESCALATE] from the local model.
        SOTA stays blocked by the privacy invariant.
        """
        from paramem.server.inference import handle_chat

        cloud_agent = self._make_mock_cloud_agent()
        router = self._make_both_match_router()

        model = MagicMock()
        model.gradient_checkpointing_disable = MagicMock()
        model.peft_config = {"episodic": MagicMock()}
        tokenizer = MagicMock()

        config = MagicMock()
        config.registry_path = MagicMock()
        config.registry_path.exists.return_value = False
        config.sanitization.mode = "off"
        config.voice.load_prompt.return_value = "You are a helper."

        ha_client = MagicMock()
        ha_client.conversation_process.return_value = None  # HA would fail if called

        with (
            patch(
                "paramem.training.indexed_memory.probe_key",
                return_value={"answer": "Alex prefers dim lights"},
            ),
            patch("paramem.models.loader.switch_adapter"),
            patch(
                "paramem.server.inference.generate_answer",
                return_value="Noted: Alex prefers dim lights.",
            ),
            patch("paramem.server.inference.detect_escalation", return_value=(False, "")),
            patch("paramem.server.inference.adapt_messages", side_effect=lambda msgs, tok: msgs),
            patch.object(tokenizer, "apply_chat_template", return_value="prompt"),
        ):
            result = handle_chat(
                text="Turn on the lights for Alex",
                conversation_id="test",
                speaker=None,
                history=None,
                model=model,
                tokenizer=tokenizer,
                config=config,
                router=router,
                ha_client=ha_client,
                sota_agent=cloud_agent,
            )

        # HA was NOT pre-flighted (intent=PERSONAL → PA probe direct).
        ha_client.conversation_process.assert_not_called()
        # SOTA blocked by privacy invariant regardless of HA outcome.
        cloud_agent.call.assert_not_called()
        assert result.escalated is False

    def test_personal_intent_blocks_sota_via_escalate(self):
        """Privacy invariant: PERSONAL + local [ESCALATE] + HA failure → no SOTA.

        The local model emits [ESCALATE] (a real production path when the
        local answer is unsure).  HA is reachable as a tool fallback but
        returns None.  Without the privacy invariant the next step would be
        SOTA — the invariant must block that for personal-class queries.
        """
        from paramem.server.inference import handle_chat

        cloud_agent = self._make_mock_cloud_agent()
        router = self._make_mock_router(known_entities=["Jordan"])

        model = MagicMock()
        model.gradient_checkpointing_disable = MagicMock()
        model.peft_config = {"episodic": MagicMock()}
        tokenizer = MagicMock()

        config = MagicMock()
        config.registry_path = MagicMock()
        config.registry_path.exists.return_value = False
        config.sanitization.mode = "off"
        config.voice.load_prompt.return_value = "You are a helper."

        ha_client = MagicMock()
        ha_client.conversation_process.return_value = None  # HA fallback fails

        with (
            patch(
                "paramem.training.indexed_memory.probe_key",
                return_value={"answer": "Jordan lives somewhere"},
            ),
            patch("paramem.models.loader.switch_adapter"),
            patch(
                "paramem.server.inference.generate_answer",
                return_value="I'm not sure. [ESCALATE] Where does Jordan live?",
            ),
            # Local model decides to escalate.
            patch(
                "paramem.server.inference.detect_escalation",
                return_value=(True, "Where does Jordan live?"),
            ),
            patch("paramem.server.inference.adapt_messages", side_effect=lambda msgs, tok: msgs),
            patch.object(tokenizer, "apply_chat_template", return_value="prompt"),
        ):
            result = handle_chat(
                text="Where does Jordan live?",
                conversation_id="test",
                speaker=None,
                history=None,
                model=model,
                tokenizer=tokenizer,
                config=config,
                router=router,
                ha_client=ha_client,
                sota_agent=cloud_agent,
            )

        # HA was tried as a tool fallback (allowed for PERSONAL).
        ha_client.conversation_process.assert_called_once()
        # SOTA blocked by the privacy invariant — this is the new guarantee.
        cloud_agent.call.assert_not_called()
        # Pre-[ESCALATE] portion of local response is returned when both
        # HA and SOTA are unavailable.
        assert "I'm not sure" in result.text
