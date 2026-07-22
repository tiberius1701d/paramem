"""Unit tests for cloud agent adapters (no API calls — mocked)."""

from unittest.mock import MagicMock, patch

import pytest

from paramem.cloud.admission import PROVIDER_KEY_ENV
from paramem.cloud.providers.base import CloudAgent, CloudAgentConfig, CloudResponse, ToolCall
from paramem.cloud.providers.openai_compat import OpenAICompatAgent
from paramem.cloud.providers.registry import get_cloud_agent
from paramem.memory.store import MemoryStore as _MS


def _stub_grouped_recall(fact_text: str):
    """Build a probe_keys_grouped_by_adapter side_effect that returns the same
    canned fact for every queried key. Used by tests that want to short-circuit
    recall (skip real generate) and assert on the routing layer alone."""

    def _stub(model, tokenizer, keys_by_adapter, *args, **kwargs):
        return {
            k: {
                "key": k,
                "subject": "x",
                "predicate": "p",
                "object": "y",
                "confidence": 1.0,
                "fact_text": fact_text,
            }
            for keys in keys_by_adapter.values()
            for k in keys
        }

    return _stub


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
        """The bug the old ``is_available`` shipped: a configured endpoint
        satisfied it with no key at all, so the agent was built and POSTed
        ``Authorization: Bearer `` (empty). Admission requires the key."""
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


class TestPrivacyRouting:
    """Integration tests verifying personal queries never reach the cloud agent.

    These test the routing logic in handle_chat: queries containing known
    graph entities must be handled locally, never forwarded to cloud.
    """

    def _make_mock_cloud_agent(self):
        """Create a mock cloud agent that tracks whether call() was invoked."""
        agent = MagicMock(spec=CloudAgent)
        agent.call.return_value = CloudResponse(text="cloud answer")
        return agent

    def _make_mock_router(self, known_entities=None):
        """Create a mock router that emits a PERSONAL plan when *known_entities*
        appear in the query (or the speaker name), and a GENERAL plan
        otherwise.

        Production routing no longer derives PERSONAL from entity matches —
        intent is encoder-driven.  This mock continues to use entity matching
        as a convenient way to simulate "encoder says PERSONAL" for the
        privacy-invariant tests below, without standing up the encoder.  The
        no-match branch uses ``Intent.GENERAL`` (a genuinely non-personal
        classification) rather than ``Intent.UNKNOWN`` — since the fail-
        closed gate in ``handle_chat`` now treats UNKNOWN as personal (see
        ``test_unknown_intent_never_reaches_cloud``), UNKNOWN is no longer a
        valid stand-in for "non-personal" here.
        """
        from paramem.server.router import Intent, RoutingPlan, RoutingStep

        router = MagicMock()
        known = {e.lower() for e in (known_entities or [])}

        def route(text, speaker_id=None):
            text_lower = text.lower()
            matched = [e for e in known if e in text_lower]
            if matched:
                return RoutingPlan(
                    steps=[
                        RoutingStep(
                            adapter_name="episodic",
                            keys_to_probe=["graph1"],
                        )
                    ],
                    strategy="targeted_probe",
                    intent=Intent.PERSONAL,
                )
            return RoutingPlan(strategy="direct", intent=Intent.GENERAL)

        router.route = route
        return router

    def _make_unknown_router(self):
        """Create a mock router returning ``Intent.UNKNOWN`` with no steps.

        Simulates the "intent could not be established" case (no
        ``IntentConfig``, classifier unavailable, below-margin confidence):
        ``RoutingPlan.intent`` carries the raw ``UNKNOWN`` value and
        ``plan.steps`` is empty, exercising the defensive fallthrough
        branch in ``handle_chat`` rather than the direct personal-probe
        branch.
        """
        from paramem.server.router import Intent, RoutingPlan

        router = MagicMock()
        router.route = lambda text, speaker_id=None: RoutingPlan(
            strategy="direct", intent=Intent.UNKNOWN
        )
        return router

    def _make_ha_only_router(self):
        """Create a mock router returning an HA-only match (no PA steps).

        Production classify_intent: ``has_ha_match=True`` → :attr:`Intent.COMMAND`.
        """
        from paramem.server.router import Intent, RoutingPlan

        router = MagicMock()

        def route(text, speaker_id=None):
            return RoutingPlan(
                steps=[],
                strategy="direct",
                ha_domains=["light"],
                intent=Intent.COMMAND,
            )

        router.route = route
        return router

    def _make_both_match_router(self):
        """Create a router returning a PERSONAL plan with HA domains attached.

        Production rule under the new state-signal model: PA enrollment does
        NOT short-circuit intent; the encoder decides.  When the encoder says
        PERSONAL even though HA also matched, the privacy invariant applies:
        Cloud must never be reached for this query.
        """
        from paramem.server.router import Intent, RoutingPlan, RoutingStep

        router = MagicMock()

        def route(text, speaker_id=None):
            return RoutingPlan(
                steps=[RoutingStep(adapter_name="episodic", keys_to_probe=["graph1"])],
                strategy="targeted_probe",
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
        # cooldown_gate_threshold_c <= 0 disables the wait_for_cooldown inference gate.
        config.vram.cooldown_gate_threshold_c = 0
        with (
            patch(
                "paramem.memory.probe.probe_keys_grouped_by_adapter",
                side_effect=_stub_grouped_recall("Jordan lives in Berlin"),
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
                speaker_id="spk-test",
                history=None,
                model=model,
                tokenizer=tokenizer,
                config=config,
                router=router,
                memory_store=_MS(replay_enabled=False),
            )

        # Cloud agent must NOT have been called
        cloud_agent.call.assert_not_called()
        assert "Berlin" in result.text

    def test_non_personal_query_goes_to_cloud(self):
        """Query with no entity match → HA first (None) → cloud fallback."""
        from paramem.server.inference import handle_chat

        cloud_agent = self._make_mock_cloud_agent()
        router = self._make_mock_router(known_entities=["Jordan"])

        model = MagicMock()
        model.gradient_checkpointing_disable = MagicMock()
        tokenizer = MagicMock()

        config = MagicMock()
        config.voice.load_prompt.return_value = "You are a helper."

        # Mock HA client that returns None (simulates HA unavailable)
        ha_client = MagicMock()
        ha_client.conversation_process.return_value = None

        result = handle_chat(
            text="What is the weather today?",
            conversation_id="test",
            speaker=None,
            speaker_id="spk-test",
            history=None,
            model=model,
            tokenizer=tokenizer,
            config=config,
            router=router,
            ha_client=ha_client,
            cloud_agent=cloud_agent,
            memory_store=_MS(replay_enabled=False),
        )

        # HA was attempted first and returned None
        ha_client.conversation_process.assert_called_once()
        # cloud agent called as fallback
        cloud_agent.call.assert_called_once()
        assert result.escalated is True
        assert result.text == "cloud answer"

    def test_unknown_intent_never_reaches_cloud(self):
        """Regression: Intent.UNKNOWN must fail closed to personal.

        Without an ``IntentConfig`` (or below the confidence margin),
        ``classify_intent`` returns ``Intent.UNKNOWN``.  The privacy gate in
        ``handle_chat`` must treat that identically to ``PERSONAL`` — an
        unclassifiable query is never escalated to the external cloud
        provider, even though the routing plan carries no probe steps and
        falls through the same HA-first/cloud-fallback branch a GENERAL
        query would use.
        """
        from paramem.server.inference import handle_chat

        cloud_agent = self._make_mock_cloud_agent()
        router = self._make_unknown_router()

        model = MagicMock()
        model.gradient_checkpointing_disable = MagicMock()
        tokenizer = MagicMock()

        config = MagicMock()
        config.voice.load_prompt.return_value = "You are a helper."

        # Mock HA client that returns None (simulates HA unavailable) so the
        # cloud fallback is the next stop — same shape as the GENERAL case
        # in test_non_personal_query_goes_to_cloud.
        ha_client = MagicMock()
        ha_client.conversation_process.return_value = None

        # HA fails and cloud is blocked (is_personal), so handle_chat falls
        # through to the local base-model answer as the last resort — mock
        # its generation the same way test_no_cloud_falls_back_to_local
        # does so the assertions below exercise the routing/privacy gate,
        # not real (unmocked) model.generate() output.
        with (
            patch(
                "paramem.server.inference.generate_answer",
                return_value="I'm not sure about that.",
            ),
            patch(
                "paramem.server.inference.detect_escalation",
                return_value=(False, ""),
            ),
        ):
            result = handle_chat(
                text="What is the weather today?",
                conversation_id="test",
                speaker=None,
                speaker_id="spk-test",
                history=None,
                model=model,
                tokenizer=tokenizer,
                config=config,
                router=router,
                ha_client=ha_client,
                cloud_agent=cloud_agent,
                memory_store=_MS(replay_enabled=False),
            )

        # HA (local) is still reachable as a tool fallback for UNKNOWN...
        ha_client.conversation_process.assert_called_once()
        # ...but cloud (external cloud) must NOT have been called.
        cloud_agent.call.assert_not_called()
        assert result.escalated is False

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
                speaker_id="spk-test",
                history=None,
                model=model,
                tokenizer=tokenizer,
                config=config,
                router=router,
                memory_store=_MS(replay_enabled=False),
            )

        assert result.escalated is False

    def test_imperative_ha_command_cloud_fallback(self):
        """COMMAND intent + HA fails → cloud fallback (imperative shape)."""
        from paramem.server.inference import handle_chat

        cloud_agent = self._make_mock_cloud_agent()
        router = self._make_ha_only_router()

        model = MagicMock()
        model.gradient_checkpointing_disable = MagicMock()
        tokenizer = MagicMock()

        config = MagicMock()
        config.voice.load_prompt.return_value = "You are a helper."

        ha_client = MagicMock()
        ha_client.conversation_process.return_value = None  # HA fails

        result = handle_chat(
            text="Turn on the lights",
            conversation_id="test",
            speaker=None,
            speaker_id="spk-test",
            history=None,
            model=model,
            tokenizer=tokenizer,
            config=config,
            router=router,
            ha_client=ha_client,
            cloud_agent=cloud_agent,
            memory_store=_MS(replay_enabled=False),
        )

        ha_client.conversation_process.assert_called_once()
        cloud_agent.call.assert_called_once()
        assert result.escalated is True
        assert result.text == "cloud answer"

    def test_ha_nonimperative_match_cloud_fallback(self):
        """COMMAND intent + HA fails → cloud fallback (interrogative shape)."""
        from paramem.server.inference import handle_chat

        cloud_agent = self._make_mock_cloud_agent()
        router = self._make_ha_only_router()

        model = MagicMock()
        model.gradient_checkpointing_disable = MagicMock()
        tokenizer = MagicMock()

        config = MagicMock()
        config.voice.load_prompt.return_value = "You are a helper."

        ha_client = MagicMock()
        ha_client.conversation_process.return_value = None  # HA fails

        result = handle_chat(
            text="Is the light on?",
            conversation_id="test",
            speaker=None,
            speaker_id="spk-test",
            history=None,
            model=model,
            tokenizer=tokenizer,
            config=config,
            router=router,
            ha_client=ha_client,
            cloud_agent=cloud_agent,
            memory_store=_MS(replay_enabled=False),
        )

        ha_client.conversation_process.assert_called_once()
        cloud_agent.call.assert_called_once()
        assert result.escalated is True
        assert result.text == "cloud answer"

    def test_personal_both_match_uses_pa_probe_no_pre_flight_ha(self):
        """PA + HA overlap → intent=PERSONAL → PA probe runs directly.

        Under intent-keyed dispatch, intent=PERSONAL routes straight to the
        local PA probe.  The pre-flight HA call from the legacy cascade is
        gone — HA is reachable only via [ESCALATE] from the local model.
        Cloud stays blocked by the privacy invariant.
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
        config.voice.load_prompt.return_value = "You are a helper."
        # cooldown_gate_threshold_c <= 0 disables the wait_for_cooldown inference gate.
        config.vram.cooldown_gate_threshold_c = 0

        ha_client = MagicMock()
        ha_client.conversation_process.return_value = None  # HA would fail if called

        with (
            patch(
                "paramem.memory.probe.probe_keys_grouped_by_adapter",
                side_effect=_stub_grouped_recall("Alex prefers dim lights"),
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
                speaker_id="spk-test",
                history=None,
                model=model,
                tokenizer=tokenizer,
                config=config,
                router=router,
                ha_client=ha_client,
                cloud_agent=cloud_agent,
                memory_store=_MS(replay_enabled=False),
            )

        # HA was NOT pre-flighted (intent=PERSONAL → PA probe direct).
        ha_client.conversation_process.assert_not_called()
        # cloud blocked by privacy invariant regardless of HA outcome.
        cloud_agent.call.assert_not_called()
        assert result.escalated is False

    def test_personal_intent_blocks_cloud_via_escalate(self):
        """Privacy invariant: PERSONAL + local [ESCALATE] + HA failure → no cloud.

        The local model emits [ESCALATE] (a real production path when the
        local answer is unsure).  HA is reachable as a tool fallback but
        returns None.  Without the privacy invariant the next step would be
        Cloud — the invariant must block that for personal-class queries.
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
        config.voice.load_prompt.return_value = "You are a helper."
        # cooldown_gate_threshold_c <= 0 disables the wait_for_cooldown inference gate.
        config.vram.cooldown_gate_threshold_c = 0

        ha_client = MagicMock()
        ha_client.conversation_process.return_value = None  # HA fallback fails

        with (
            patch(
                "paramem.memory.probe.probe_keys_grouped_by_adapter",
                side_effect=_stub_grouped_recall("Jordan lives somewhere"),
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
                speaker_id="spk-test",
                history=None,
                model=model,
                tokenizer=tokenizer,
                config=config,
                router=router,
                ha_client=ha_client,
                cloud_agent=cloud_agent,
                memory_store=_MS(replay_enabled=False),
            )

        # HA was tried as a tool fallback (allowed for PERSONAL).
        ha_client.conversation_process.assert_called_once()
        # cloud blocked by the privacy invariant — this is the new guarantee.
        cloud_agent.call.assert_not_called()
        # Pre-[ESCALATE] portion of local response is returned when both
        # HA and cloud are unavailable.
        assert "I'm not sure" in result.text


class TestForwardedQueryVerdict:
    """The model-authored forwarded query carries its own personal verdict.

    ``detect_escalation`` returns everything after ``[ESCALATE]``.  On the
    personal path the local model has already recalled facts from
    parametric memory, so that string can name entities the user never
    typed.  ``_maybe_escalate`` therefore re-runs
    ``check_personal_content`` on the forwarded query and gates BOTH
    external hops with the result — the HA hop included, because
    ``ha_agent_id`` is operator-configurable and routinely points at a
    cloud-backed agent.
    """

    RESPONSE = "I'm not sure where to look. [ESCALATE] Find ceramics shops in Munich for Maria."
    CONTROL_RESPONSE = "I'm not sure. [ESCALATE] What is the capital of France?"

    def _config(self):
        config = MagicMock()
        config.sanitization.cloud_mode = "block"
        # No encoder in unit tests — check_personal_content falls back to the
        # English token-set arm; the known-entity arm is unaffected.
        config.personal_referent = None
        return config

    def _run(self, response, *, is_personal=False):
        from paramem.server.inference import _maybe_escalate

        cloud_agent = MagicMock(spec=CloudAgent)
        cloud_agent.call.return_value = CloudResponse(text="cloud answer")
        ha_client = MagicMock()
        ha_client.conversation_process.return_value = "HA answer"

        result = _maybe_escalate(
            response,
            self._config(),
            cloud_agent=cloud_agent,
            ha_client=ha_client,
            speaker_id="spk-test",
            is_personal=is_personal,
            model=MagicMock(),
            tokenizer=MagicMock(),
            known_entities={"Maria"},
        )
        return result, ha_client, cloud_agent

    def test_personal_forwarded_query_suppresses_ha_hop(self):
        _result, ha_client, _cloud_agent = self._run(self.RESPONSE)
        ha_client.conversation_process.assert_not_called()

    def test_personal_forwarded_query_blocks_cloud_hop(self):
        """Turn verdict is False — only the forwarded-query verdict blocks."""
        _result, _ha_client, cloud_agent = self._run(self.RESPONSE, is_personal=False)
        cloud_agent.call.assert_not_called()

    def test_both_hops_suppressed_returns_pre_escalation_text(self):
        result, _ha_client, _cloud_agent = self._run(self.RESPONSE)
        assert result.text == "I'm not sure where to look."
        assert result.escalated is False

    def test_non_personal_forwarded_query_still_reaches_ha(self):
        result, ha_client, cloud_agent = self._run(self.CONTROL_RESPONSE)
        ha_client.conversation_process.assert_called_once()
        assert ha_client.conversation_process.call_args.args[0] == (
            "What is the capital of France?"
        )
        cloud_agent.call.assert_not_called()  # HA answered, no fallback needed
        assert result.text == "HA answer"
        assert result.escalated is True


class TestCloudModePolicy:
    """Architecture #3: ``sanitization.cloud_mode`` selects the egress policy.

    These tests pin the dispatch in ``answer_via_cloud`` against
    each (cloud_mode, is_personal) combination.  They mock the anonymizer
    surface (``anonymize_outbound``, ``deanonymize_inbound``) so the
    policy logic is exercised without invoking the local LLM.
    """

    def _make_cloud_agent(self):
        agent = MagicMock(spec=CloudAgent)
        agent.call.return_value = CloudResponse(text="<placeholder> answer")
        return agent

    def _config(self, cloud_mode: str):
        config = MagicMock()
        config.sanitization.cloud_mode = cloud_mode
        config.voice.load_prompt.return_value = "You are a helper."
        return config

    def _personal_router(self):
        from paramem.server.router import Intent, RoutingPlan

        router = MagicMock()
        router.route = lambda text, speaker=None, speaker_id=None: RoutingPlan(
            strategy="direct", intent=Intent.PERSONAL
        )
        router._speaker_key_index = {}
        return router

    def _general_router(self):
        from paramem.server.router import Intent, RoutingPlan

        router = MagicMock()
        router.route = lambda text, speaker=None, speaker_id=None: RoutingPlan(
            strategy="direct", intent=Intent.GENERAL
        )
        router._speaker_key_index = {}
        return router

    def _run(self, *, router, config, cloud_agent, ha_client=None, history=None):
        """Drive handle_chat with HA missing/None so cloud is the next stop.

        Patches ``_base_model_answer`` so tests that get blocked at cloud
        (PERSONAL under block / both, leak-guard tripped under anonymize)
        still terminate cleanly without invoking the base-model path.
        """
        from paramem.server.inference import ChatResult, handle_chat

        if ha_client is None:
            ha_client = MagicMock()
            ha_client.conversation_process.return_value = None

        model = MagicMock()
        model.gradient_checkpointing_disable = MagicMock()
        tokenizer = MagicMock()

        with patch(
            "paramem.server.inference._base_model_answer",
            return_value=ChatResult(text="<base-fallback>"),
        ):
            return handle_chat(
                text="What's the population of Berlin?",
                conversation_id="cloud-mode-test",
                speaker=None,
                speaker_id="spk-test",
                history=history,
                model=model,
                tokenizer=tokenizer,
                config=config,
                router=router,
                ha_client=ha_client,
                cloud_agent=cloud_agent,
                memory_store=_MS(replay_enabled=False),
            )

    # ---- block mode ----

    def test_block_mode_personal_query_blocks_cloud(self):
        cloud_agent = self._make_cloud_agent()
        result = self._run(
            router=self._personal_router(),
            config=self._config("block"),
            cloud_agent=cloud_agent,
        )
        cloud_agent.call.assert_not_called()
        assert result.escalated is False

    def test_block_mode_general_query_sends_verbatim(self):
        cloud_agent = self._make_cloud_agent()
        cloud_agent.call.return_value = CloudResponse(text="Berlin has 3.7M people.")
        self._run(
            router=self._general_router(),
            config=self._config("block"),
            cloud_agent=cloud_agent,
        )
        cloud_agent.call.assert_called_once()
        # block mode + non-PERSONAL: text passed through unmodified.
        sent = cloud_agent.call.call_args.kwargs["query"]
        assert sent == "What's the population of Berlin?"

    def test_block_mode_general_query_threads_speaker_id_to_history_sanitizer(self):
        """The ``cloud_mode=block`` + non-PERSONAL branch of
        ``answer_via_cloud`` must pass ``speaker_id`` through to
        ``_sanitize_history`` — without it the first-person detector is dead
        on this channel, unlike the other three ``_sanitize_history`` call
        sites (the anonymizing branch here, and both ``app.py`` sites).
        """
        cloud_agent = self._make_cloud_agent()
        cloud_agent.call.return_value = CloudResponse(text="Berlin has 3.7M people.")
        with patch("paramem.server.inference._sanitize_history", return_value=[]) as mock_sanitize:
            self._run(
                router=self._general_router(),
                config=self._config("block"),
                cloud_agent=cloud_agent,
                history=[{"role": "user", "text": "hi"}],
            )
        mock_sanitize.assert_called_once()
        assert mock_sanitize.call_args.kwargs["speaker_id"] == "spk-test"

    # ---- anonymize mode ----

    @staticmethod
    def _payload(*, status: str, anon_transcript: str = "", forward=None, reverse=None):
        from paramem.cloud.anonymize import AnonymizedContract

        reverse = reverse or {}
        return AnonymizedContract(
            status=status,
            forward=forward or {},
            reverse=reverse,
            anon_transcript=anon_transcript,
            declared=frozenset(reverse.keys()),
            norm_stats={"inverted": 0, "dropped": 0},
            rekey_dropped=0,
            raw="",
        )

    def test_anonymize_mode_round_trips_via_anonymizer(self):
        cloud_agent = self._make_cloud_agent()
        cloud_agent.call.return_value = CloudResponse(text="Person_1 is a useful placeholder here.")
        payload = self._payload(
            status="ok",
            anon_transcript="Person_1 query",
            forward={"Alex": "Person_1"},
            reverse={"Person_1": "Alex"},
        )
        with (
            patch(
                "paramem.graph.flows.anonymize_turn",
                return_value=payload,
            ) as mock_anon,
            patch(
                "paramem.cloud.deanonymize.deanonymize_text",
                return_value="Alex is a useful placeholder here.",
            ) as mock_deanon,
        ):
            result = self._run(
                router=self._personal_router(),
                config=self._config("anonymize"),
                cloud_agent=cloud_agent,
            )

        mock_anon.assert_called_once()
        mock_deanon.assert_called_once()
        cloud_agent.call.assert_called_once()
        sent = cloud_agent.call.call_args.kwargs["query"]
        assert sent == "Person_1 query"
        assert result.text == "Alex is a useful placeholder here."

    def test_anonymize_mode_leak_guard_blocks_query(self):
        """A ``status="failed"`` payload (leak guard or model failure) =
        per-query block, cloud never called."""
        cloud_agent = self._make_cloud_agent()
        with patch(
            "paramem.graph.flows.anonymize_turn",
            return_value=self._payload(status="failed"),
        ):
            self._run(
                router=self._personal_router(),
                config=self._config("anonymize"),
                cloud_agent=cloud_agent,
            )
        cloud_agent.call.assert_not_called()

    # ---- both mode ----

    def test_both_mode_personal_blocked_non_personal_anonymized(self):
        # PERSONAL blocked.
        cloud_agent_p = self._make_cloud_agent()
        self._run(
            router=self._personal_router(),
            config=self._config("both"),
            cloud_agent=cloud_agent_p,
        )
        cloud_agent_p.call.assert_not_called()

        # non-PERSONAL anonymized + sent.
        cloud_agent_g = self._make_cloud_agent()
        cloud_agent_g.call.return_value = CloudResponse(text="<answer>")
        payload = self._payload(
            status="ok",
            anon_transcript="anon q",
            forward={"Berlin": "City_1"},
            reverse={"City_1": "Berlin"},
        )
        with (
            patch(
                "paramem.graph.flows.anonymize_turn",
                return_value=payload,
            ),
            patch(
                "paramem.cloud.deanonymize.deanonymize_text",
                return_value="<answer>",
            ),
        ):
            self._run(
                router=self._general_router(),
                config=self._config("both"),
                cloud_agent=cloud_agent_g,
            )
        cloud_agent_g.call.assert_called_once()
        sent = cloud_agent_g.call.call_args.kwargs["query"]
        assert sent == "anon q"

    # ---- dual-scope closure (history alongside the anonymized transcript) ----

    def test_history_names_are_anonymized_under_anonymize_mode(self):
        """Under ``cloud_mode="anonymize"``, no history turn reaches
        ``cloud_agent.call`` carrying a name present in ``payload.forward``.

        The history channel has no policy knob of its own: it is always
        drop-gated, and under an anonymizing ``cloud_mode`` the survivors are
        additionally substituted through the forward map.
        """
        cloud_agent = self._make_cloud_agent()
        cloud_agent.call.return_value = CloudResponse(text="<answer>")
        payload = self._payload(
            status="ok",
            anon_transcript="Person_1 query",
            forward={"Alex": "Person_1"},
            reverse={"Person_1": "Alex"},
        )
        # No first-person markers and "Alex" is not a known_entity in this
        # test's setup, so the drop-gate lets the turn through unchanged —
        # proving any absence of "Alex" downstream is the forward-map
        # substitution, not the drop gate.
        history = [{"role": "user", "text": "Did Alex call?"}]
        with (
            patch(
                "paramem.graph.flows.anonymize_turn",
                return_value=payload,
            ),
            patch(
                "paramem.cloud.deanonymize.deanonymize_text",
                return_value="<answer>",
            ),
        ):
            self._run(
                router=self._personal_router(),
                config=self._config("anonymize"),
                cloud_agent=cloud_agent,
                history=history,
            )

        cloud_agent.call.assert_called_once()
        sent_history = cloud_agent.call.call_args.kwargs["history"]
        assert sent_history, "expected the history turn to survive the drop gate"
        for turn in sent_history:
            assert "Alex" not in turn["text"], (
                f"History turn carried the real name verbatim: {turn!r}"
            )
        assert any("Person_1" in turn["text"] for turn in sent_history), (
            f"Expected the forward-map substitution to apply to history: {sent_history!r}"
        )


class TestCloudOnlyRouteSpeakerId:
    """``_cloud_only_route``'s ``speaker_id`` parameter
    must reach ``_sanitize_history`` — consistent with the other
    ``_sanitize_history`` call sites (both branches of
    ``inference.answer_via_cloud`` and the forced-routing
    path in ``app.py``).
    """

    def test_speaker_id_reaches_history_sanitizer(self):
        from paramem.server.app import _cloud_only_route
        from paramem.server.inference import ChatResult

        config = MagicMock()
        cloud_agent = MagicMock()

        with (
            patch("paramem.server.app._sanitize_history", return_value=[]) as mock_sanitize,
            patch(
                "paramem.server.app._escalate_to_cloud",
                return_value=ChatResult(text="Berlin has 3.7M people."),
            ),
        ):
            _cloud_only_route(
                text="What's the population of Berlin?",
                speaker="Alex",
                history=[{"role": "user", "text": "hi"}],
                config=config,
                cloud_permitted=True,
                ha_client=None,
                cloud_agent=cloud_agent,
                speaker_id="spk-test",
            )

        mock_sanitize.assert_called_once()
        assert mock_sanitize.call_args.kwargs["speaker_id"] == "spk-test"

    def test_current_turn_reaches_cloud_verbatim(self):
        """Cloud-only has no local model, so there is nothing to anonymize.

        The old ``sanitize_for_cloud`` call here was a second policy gate on
        a path that holds no ParaMem knowledge; it is deleted.  The turn
        itself must arrive at ``_escalate_to_cloud`` unmodified.
        """
        from paramem.server.app import _cloud_only_route
        from paramem.server.inference import ChatResult

        config = MagicMock()

        with (
            patch("paramem.server.app._sanitize_history", return_value=[]),
            patch(
                "paramem.server.app._escalate_to_cloud",
                return_value=ChatResult(text="answer"),
            ) as mock_escalate,
        ):
            _cloud_only_route(
                text="Where does Alex live?",
                speaker="Alex",
                history=None,
                config=config,
                cloud_permitted=True,
                ha_client=None,
                cloud_agent=MagicMock(),
                known_entities={"Alex"},
            )

        assert mock_escalate.call_args.args[0] == "Where does Alex live?"


class TestDegradedServingGate:
    """``cloud.allow_degraded_serving`` closes the CLOUD leg only.

    The HA leg carries no ParaMem-held knowledge and runs on the user's own
    network, so it stays open in every degraded state.
    """

    def _run(self, *, cloud_permitted, ha_answers):
        from paramem.server.app import _cloud_only_route
        from paramem.server.inference import ChatResult

        ha_client = MagicMock()
        ha_client.conversation_process.return_value = "HA answer" if ha_answers else None

        with (
            patch("paramem.server.app._sanitize_history", return_value=[]),
            patch(
                "paramem.server.app._escalate_to_cloud",
                return_value=ChatResult(text="cloud answer"),
            ) as mock_escalate,
        ):
            result = _cloud_only_route(
                text="What's the population of Berlin?",
                speaker="Alex",
                history=None,
                config=MagicMock(),
                cloud_permitted=cloud_permitted,
                ha_client=ha_client,
                cloud_agent=MagicMock(),
            )
        return result, mock_escalate, ha_client

    def test_gated_closes_cloud_leg(self):
        result, mock_escalate, _ = self._run(cloud_permitted=False, ha_answers=False)
        mock_escalate.assert_not_called()
        assert "limited mode" in result.text

    def test_gated_keeps_ha_leg(self):
        result, mock_escalate, ha_client = self._run(cloud_permitted=False, ha_answers=True)
        ha_client.conversation_process.assert_called_once()
        mock_escalate.assert_not_called()
        assert result.text == "HA answer"

    def test_permitted_opens_cloud_leg(self):
        result, mock_escalate, _ = self._run(cloud_permitted=True, ha_answers=False)
        mock_escalate.assert_called_once()
        assert result.text == "cloud answer"
