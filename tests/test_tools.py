"""Unit tests for tool execution infrastructure (no HA or API calls)."""

from unittest.mock import MagicMock, patch

from paramem.server.cloud.base import CloudResponse, ToolCall
from paramem.server.config import load_server_config
from paramem.server.tools.executor import execute_tool_loop
from paramem.server.tools.ha_client import HAClient
from paramem.server.tools.registry import ToolRegistry


class TestToolRegistry:
    def test_empty_by_default(self):
        registry = ToolRegistry()
        assert registry.tools == []

    def test_default_deny_no_allowlist(self):
        """Auto-discovery with no allowlist loads nothing."""
        registry = ToolRegistry()
        ha_services = [
            {
                "domain": "light",
                "services": {
                    "turn_on": {"description": "Turn on a light", "fields": {}},
                },
            }
        ]
        registry.load_from_ha(ha_services, allowlist=None)
        assert registry.tools == []

    def test_default_deny_empty_allowlist(self):
        registry = ToolRegistry()
        ha_services = [
            {
                "domain": "light",
                "services": {
                    "turn_on": {"description": "Turn on a light", "fields": {}},
                },
            }
        ]
        registry.load_from_ha(ha_services, allowlist=[])
        assert registry.tools == []

    def test_allowlist_filters_services(self):
        registry = ToolRegistry()
        ha_services = [
            {
                "domain": "light",
                "services": {
                    "turn_on": {"description": "Turn on", "fields": {}},
                    "turn_off": {"description": "Turn off", "fields": {}},
                },
            },
            {
                "domain": "switch",
                "services": {
                    "toggle": {"description": "Toggle", "fields": {}},
                },
            },
        ]
        registry.load_from_ha(ha_services, allowlist=["light.*"])
        assert len(registry.tools) == 2
        names = {t["name"] for t in registry.tools}
        assert "light.turn_on" in names
        assert "light.turn_off" in names
        assert "switch.toggle" not in names

    def test_sensitive_domains_blocked(self):
        registry = ToolRegistry()
        ha_services = [
            {
                "domain": "alarm_control_panel",
                "services": {
                    "arm_away": {"description": "Arm alarm", "fields": {}},
                },
            },
        ]
        registry.load_from_ha(ha_services, allowlist=["alarm_control_panel.*"])
        assert registry.tools == []

    def test_sensitive_override(self):
        registry = ToolRegistry()
        ha_services = [
            {
                "domain": "lock",
                "services": {
                    "lock": {"description": "Lock door", "fields": {}},
                },
            },
        ]
        registry.load_from_ha(
            ha_services,
            allowlist=["lock.*"],
            sensitive_override=True,
        )
        assert len(registry.tools) == 1

    def test_ha_tools_marked_as_proxied(self):
        registry = ToolRegistry()
        ha_services = [
            {
                "domain": "light",
                "services": {
                    "turn_on": {"description": "Turn on", "fields": {}},
                },
            },
        ]
        registry.load_from_ha(ha_services, allowlist=["light.*"])
        assert registry.is_ha_proxied("light.turn_on")

    def test_yaml_override(self, tmp_path):
        registry = ToolRegistry()
        ha_services = [
            {
                "domain": "script",
                "services": {
                    "get_weather": {"description": "Old desc", "fields": {}},
                },
            },
        ]
        registry.load_from_ha(ha_services, allowlist=["script.*"])

        yaml_file = tmp_path / "tools.yaml"
        yaml_file.write_text(
            "tools:\n"
            "  - name: script.get_weather\n"
            "    description: Get weather for home\n"
            "    execution: ha\n"
            "  - name: tavily_search\n"
            "    description: Web search\n"
            "    execution: cloud_native\n"
            "    parameters:\n"
            "      type: object\n"
            "      properties:\n"
            "        query:\n"
            "          type: string\n"
            "      required:\n"
            "        - query\n"
        )
        registry.load_from_yaml(yaml_file)

        # Override preserved
        assert registry.get_tool("script.get_weather")["description"] == "Get weather for home"
        assert registry.is_ha_proxied("script.get_weather")

        # Cloud-native tool loaded
        assert "tavily_search" in registry.cloud_native_tools
        assert not registry.is_ha_proxied("tavily_search")

    def test_field_types_from_selector(self):
        registry = ToolRegistry()
        ha_services = [
            {
                "domain": "light",
                "services": {
                    "turn_on": {
                        "description": "Turn on",
                        "fields": {
                            "brightness": {
                                "description": "Brightness level",
                                "selector": {"number": {}},
                            },
                            "entity_id": {
                                "description": "Target",
                                "required": True,
                            },
                        },
                    },
                },
            },
        ]
        registry.load_from_ha(ha_services, allowlist=["light.*"])
        tool = registry.get_tool("light.turn_on")
        params = tool["parameters"]
        assert params["properties"]["brightness"]["type"] == "number"
        assert "entity_id" in params["required"]


class TestHAClient:
    @patch("paramem.server.tools.ha_client.httpx.Client")
    def test_health_check_success(self, mock_client_cls):
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"message": "API running."}
        mock_resp.raise_for_status = MagicMock()
        mock_client.get.return_value = mock_resp
        mock_client.is_closed = False
        mock_client_cls.return_value = mock_client

        client = HAClient("http://localhost:8123", "token")
        client._client = mock_client
        result = client.health_check()
        assert result == {"message": "API running."}

    @patch("paramem.server.tools.ha_client.httpx.Client")
    def test_call_service_with_return_response(self, mock_client_cls):
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"content": "search results"}
        mock_resp.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_resp
        mock_client.is_closed = False
        mock_client_cls.return_value = mock_client

        client = HAClient("http://localhost:8123", "token")
        client._client = mock_client
        result = client.call_service(
            "script",
            "tavily_search",
            data={"variables": {"query": "test"}},
            return_response=True,
        )
        assert result == {"content": "search results"}
        call_args = mock_client.post.call_args
        assert call_args[1]["json"]["return_response"] is True


class TestExecutor:
    def test_no_tool_calls_returns_text(self):
        """If cloud returns text immediately, no loop needed."""
        cloud_agent = MagicMock()
        response = CloudResponse(text="The answer is 42.")
        result = execute_tool_loop(
            initial_response=response,
            cloud_agent=cloud_agent,
            query="test",
            system_prompt="",
            ha_client=None,
            registry=ToolRegistry(),
        )
        assert result == "The answer is 42."
        cloud_agent.call.assert_not_called()

    def test_single_tool_round(self):
        """Cloud calls one tool, gets result, returns text."""
        registry = ToolRegistry()
        registry._tools["light.turn_on"] = {
            "name": "light.turn_on",
            "description": "Turn on",
            "parameters": {},
        }
        registry._ha_proxied.add("light.turn_on")

        ha_client = MagicMock(spec=HAClient)
        ha_client.call_service.return_value = [{"state": "on"}]

        cloud_agent = MagicMock()
        # After receiving tool result, cloud returns final text
        cloud_agent.call.return_value = CloudResponse(text="Light is now on.")

        initial = CloudResponse(
            tool_calls=[
                ToolCall(
                    id="call_1",
                    name="light.turn_on",
                    arguments={"entity_id": "light.living_room"},
                )
            ]
        )

        result = execute_tool_loop(
            initial_response=initial,
            cloud_agent=cloud_agent,
            query="turn on the lights",
            system_prompt="",
            ha_client=ha_client,
            registry=registry,
        )
        assert result == "Light is now on."
        ha_client.call_service.assert_called_once()
        cloud_agent.call.assert_called_once()

    def test_max_rounds_exhausted(self):
        """Loop stops after max rounds even if model keeps requesting tools."""
        registry = ToolRegistry()
        registry._tools["test.tool"] = {
            "name": "test.tool",
            "description": "Test",
            "parameters": {},
        }
        registry._ha_proxied.add("test.tool")

        ha_client = MagicMock(spec=HAClient)
        ha_client.call_service.return_value = {"result": "ok"}

        cloud_agent = MagicMock()
        # Cloud always requests another tool call
        cloud_agent.call.return_value = CloudResponse(
            tool_calls=[ToolCall(id="call_n", name="test.tool", arguments={})]
        )

        initial = CloudResponse(tool_calls=[ToolCall(id="call_0", name="test.tool", arguments={})])

        result = execute_tool_loop(
            initial_response=initial,
            cloud_agent=cloud_agent,
            query="test",
            system_prompt="",
            ha_client=ha_client,
            registry=registry,
            max_rounds=3,
        )
        assert "too many steps" in result
        assert cloud_agent.call.call_count == 3

    def test_unknown_tool_returns_error(self):
        """Tool not in registry returns error string to cloud model."""
        cloud_agent = MagicMock()
        cloud_agent.call.return_value = CloudResponse(text="Tool failed.")

        initial = CloudResponse(
            tool_calls=[ToolCall(id="call_1", name="nonexistent.tool", arguments={})]
        )

        result = execute_tool_loop(
            initial_response=initial,
            cloud_agent=cloud_agent,
            query="test",
            system_prompt="",
            ha_client=None,
            registry=ToolRegistry(),
        )
        # Cloud model should receive error and produce a text response
        assert result == "Tool failed."


class TestExtendedToolParsing:
    """Tests for Extended OpenAI Conversation format parsing."""

    def test_parse_template_tool(self, tmp_path):
        yaml_file = tmp_path / "tools.yaml"
        yaml_file.write_text(
            "- spec:\n"
            "    name: get_time\n"
            "    description: Get current time\n"
            "    parameters:\n"
            "      type: object\n"
            "      properties: {}\n"
            "  function:\n"
            "    type: template\n"
            "    value_template: '{{ now() }}'\n"
        )
        registry = ToolRegistry()
        registry.load_from_yaml(yaml_file)
        assert len(registry.tools) == 1
        assert registry.tools[0]["name"] == "get_time"
        execution = registry.get_execution_info("get_time")
        assert execution is not None
        assert execution["type"] == "template"
        assert "now()" in execution["value_template"]

    def test_parse_script_tool(self, tmp_path):
        yaml_file = tmp_path / "tools.yaml"
        yaml_file.write_text(
            "- spec:\n"
            "    name: search\n"
            "    description: Search the web\n"
            "    parameters:\n"
            "      type: object\n"
            "      properties:\n"
            "        query:\n"
            "          type: string\n"
            "      required: [query]\n"
            "  function:\n"
            "    type: script\n"
            "    sequence:\n"
            "      - action: script.do_search\n"
            "        data:\n"
            "          query: '{{ query }}'\n"
            "        response_variable: result\n"
        )
        registry = ToolRegistry()
        registry.load_from_yaml(yaml_file)
        execution = registry.get_execution_info("search")
        assert execution["type"] == "script"
        assert len(execution["sequence"]) == 1
        assert execution["sequence"][0]["action"] == "script.do_search"

    def test_parse_native_tool(self, tmp_path):
        yaml_file = tmp_path / "tools.yaml"
        yaml_file.write_text(
            "- spec:\n"
            "    name: execute_services\n"
            "    description: Execute HA services\n"
            "    parameters:\n"
            "      type: object\n"
            "      properties: {}\n"
            "  function:\n"
            "    type: native\n"
            "    name: execute_service\n"
        )
        registry = ToolRegistry()
        registry.load_from_yaml(yaml_file)
        execution = registry.get_execution_info("execute_services")
        assert execution["type"] == "native"
        assert execution["native_name"] == "execute_service"

    def test_extended_tools_override_auto_discovered(self, tmp_path):
        """Extended tools override auto-discovered services with same name."""
        registry = ToolRegistry()
        ha_services = [
            {
                "domain": "script",
                "services": {"do_search": {"description": "bare", "fields": {}}},
            }
        ]
        registry.load_from_ha(ha_services, allowlist=["script.*"])
        assert registry.tools[0]["description"] == "bare"

        yaml_file = tmp_path / "tools.yaml"
        yaml_file.write_text(
            "- spec:\n"
            "    name: script.do_search\n"
            "    description: Rich search tool\n"
            "    parameters:\n"
            "      type: object\n"
            "      properties: {}\n"
            "  function:\n"
            "    type: script\n"
            "    sequence: []\n"
        )
        registry.load_from_yaml(yaml_file)
        tool = registry.get_tool("script.do_search")
        assert tool["description"] == "Rich search tool"

    def test_execution_info_not_in_tool_defs(self, tmp_path):
        """Execution metadata should not leak into tool defs sent to LLM."""
        yaml_file = tmp_path / "tools.yaml"
        yaml_file.write_text(
            "- spec:\n"
            "    name: test_tool\n"
            "    description: Test\n"
            "    parameters:\n"
            "      type: object\n"
            "      properties: {}\n"
            "  function:\n"
            "    type: template\n"
            "    value_template: hello\n"
        )
        registry = ToolRegistry()
        registry.load_from_yaml(yaml_file)
        for tool in registry.tools:
            assert "_execution" not in tool
            assert "value_template" not in tool

    def test_legacy_format_still_works(self, tmp_path):
        """Legacy tools.yaml format with {tools: [...]} key."""
        yaml_file = tmp_path / "tools.yaml"
        yaml_file.write_text(
            "tools:\n  - name: custom.tool\n    description: A custom tool\n    execution: ha\n"
        )
        registry = ToolRegistry()
        registry.load_from_yaml(yaml_file)
        assert len(registry.tools) == 1
        assert registry.is_ha_proxied("custom.tool")
        assert registry.get_execution_info("custom.tool") is None


class TestConversationProcess:
    """Tests for HA conversation.process WebSocket call (mocked)."""

    def test_conversation_process_returns_speech(self):
        client = HAClient("http://ha.local:8123", "test-token")
        ws_mock = MagicMock()
        ws_mock.recv.side_effect = [
            '{"type": "auth_required"}',
            '{"type": "auth_ok"}',
            '{"success": true, "result": {"response": {"response": '
            '{"speech": {"plain": {"speech": "It is 3pm."}}}}}}',
        ]
        connect_mock = MagicMock()
        connect_mock.__enter__ = MagicMock(return_value=ws_mock)
        connect_mock.__exit__ = MagicMock(return_value=False)

        with patch("websockets.sync.client.connect", return_value=connect_mock):
            result = client.conversation_process("What time is it?")
        assert result == "It is 3pm."

    def test_conversation_process_returns_none_on_failure(self):
        client = HAClient("http://ha.local:8123", "test-token")
        ws_mock = MagicMock()
        ws_mock.recv.side_effect = [
            '{"type": "auth_required"}',
            '{"type": "auth_ok"}',
            '{"success": false, "error": {"message": "agent not found"}}',
        ]
        connect_mock = MagicMock()
        connect_mock.__enter__ = MagicMock(return_value=ws_mock)
        connect_mock.__exit__ = MagicMock(return_value=False)

        with patch("websockets.sync.client.connect", return_value=connect_mock):
            result = client.conversation_process("test")
        assert result is None

    def test_conversation_process_returns_none_on_connection_error(self):
        client = HAClient("http://ha.local:8123", "test-token")
        with patch(
            "websockets.sync.client.connect",
            side_effect=ConnectionRefusedError,
        ):
            result = client.conversation_process("test")
        assert result is None


class TestExecuteScriptWs:
    """Tests for execute_script WebSocket call (mocked)."""

    def test_execute_script_returns_response(self):
        client = HAClient("http://ha.local:8123", "test-token")
        ws_mock = MagicMock()
        ws_mock.recv.side_effect = [
            '{"type": "auth_required"}',
            '{"type": "auth_ok"}',
            '{"success": true, "result": {"response": {"result": "Tokyo, 12C"}}}',
        ]
        connect_mock = MagicMock()
        connect_mock.__enter__ = MagicMock(return_value=ws_mock)
        connect_mock.__exit__ = MagicMock(return_value=False)

        with patch("websockets.sync.client.connect", return_value=connect_mock):
            result = client.execute_script_ws(
                [{"action": "script.weather", "data": {"location": "Tokyo"}}]
            )
        assert result == {"result": "Tokyo, 12C"}

    def test_execute_script_returns_none_on_failure(self):
        client = HAClient("http://ha.local:8123", "test-token")
        ws_mock = MagicMock()
        ws_mock.recv.side_effect = [
            '{"type": "auth_required"}',
            '{"type": "auth_ok"}',
            '{"success": false, "error": {"message": "invalid sequence"}}',
        ]
        connect_mock = MagicMock()
        connect_mock.__enter__ = MagicMock(return_value=ws_mock)
        connect_mock.__exit__ = MagicMock(return_value=False)

        with patch("websockets.sync.client.connect", return_value=connect_mock):
            result = client.execute_script_ws([{"action": "bad.action"}])
        assert result is None


class TestToolsConfig:
    def test_tools_config_from_yaml(self, tmp_path):
        config_file = tmp_path / "server.yaml"
        config_file.write_text(
            "tools:\n"
            "  ha:\n"
            "    url: http://ha.local:8123\n"
            "    token: ${HA_TOKEN}\n"
            "    auto_discover: true\n"
            "    allowlist:\n"
            "      - light.*\n"
            "      - script.*\n"
            "  max_tool_rounds: 3\n"
            "  tool_timeout_seconds: 2.0\n"
            "  total_timeout_seconds: 6.0\n"
        )
        config = load_server_config(config_file)
        assert config.tools.ha.url == "http://ha.local:8123"
        assert config.tools.ha.auto_discover is True
        assert config.tools.ha.allowlist == ["light.*", "script.*"]
        assert config.tools.max_tool_rounds == 3
        assert config.tools.tool_timeout_seconds == 2.0
        assert config.tools.total_timeout_seconds == 6.0
