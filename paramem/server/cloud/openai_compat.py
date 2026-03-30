"""OpenAI-compatible cloud agent adapter.

Works with: OpenAI, Groq, Mistral API, local ollama, and any provider
that implements the OpenAI chat completions API format.
"""

import json
import logging

import httpx

from paramem.server.cloud.base import CloudAgent, CloudResponse, ToolCall
from paramem.server.config import GeneralAgentConfig

logger = logging.getLogger(__name__)

# Default endpoints per provider
_DEFAULT_ENDPOINTS = {
    "openai": "https://api.openai.com/v1/chat/completions",
    "groq": "https://api.groq.com/openai/v1/chat/completions",
}

# Providers known to be OpenAI-compatible
COMPATIBLE_PROVIDERS = {"openai", "groq", "mistral", "ollama"}


class OpenAICompatAgent(CloudAgent):
    """Adapter for OpenAI-compatible chat completions API."""

    def __init__(self, config: GeneralAgentConfig):
        super().__init__(config)
        self._endpoint = config.endpoint or _DEFAULT_ENDPOINTS.get(config.provider, "")
        if not self._endpoint:
            logger.warning(
                "No endpoint for provider '%s'. Set endpoint in config.", config.provider
            )

    def call(
        self,
        query: str,
        system_prompt: str = "",
        tool_results: list[dict] | None = None,
        tools: list[dict] | None = None,
        history: list[dict] | None = None,
    ) -> CloudResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Include conversation history before the current query
        if history:
            for turn in history:
                role = turn.get("role", "user")
                text = turn.get("text", "")
                if role in ("user", "assistant") and text:
                    messages.append({"role": role, "content": text})

        messages.append({"role": "user", "content": query})

        # Insert tool results: user → assistant(tool_calls) → tool → ...
        # OpenAI requires this ordering after the initial user message.
        if tool_results:
            for result in tool_results:
                messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": result["tool_call_id"],
                                "type": "function",
                                "function": {
                                    "name": result["name"],
                                    "arguments": json.dumps(result.get("arguments", {})),
                                },
                            }
                        ],
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": result["tool_call_id"],
                        "content": result["result"],
                    }
                )

        is_search_model = "search" in self.config.model

        payload: dict = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": 1024,
        }

        # Search models reject temperature; regular models use it
        if not is_search_model:
            payload["temperature"] = 0.7

        # Enable web search for OpenAI search models (gpt-4o-search-preview, etc.)
        if self.config.provider == "openai" and is_search_model:
            payload["web_search_options"] = {}

        if tools:
            payload["tools"] = self.format_tools(tools)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }

        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(self._endpoint, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
        except httpx.TimeoutException:
            logger.error("Cloud agent timed out")
            return CloudResponse(text="I couldn't reach the cloud service in time.")
        except httpx.HTTPStatusError as e:
            logger.error("Cloud agent HTTP error: %s", e.response.status_code)
            return CloudResponse(text="I couldn't get an answer from the cloud service.")
        except httpx.RequestError as e:
            logger.error("Cloud agent connection error: %s", e)
            return CloudResponse(text="I couldn't connect to the cloud service.")

        return self._parse_response(data)

    def format_tools(self, tools: list[dict]) -> list[dict]:
        """Convert standard tool definitions to OpenAI format.

        Standard format: {name, description, parameters}
        OpenAI format: {type: "function", function: {name, description, parameters}}
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
                },
            }
            for tool in tools
        ]

    def _parse_response(self, data: dict) -> CloudResponse:
        """Parse OpenAI-format response into CloudResponse."""
        choices = data.get("choices", [])
        if not choices:
            return CloudResponse(text="", finish_reason="empty_choices")
        choice = choices[0]
        message = choice.get("message", {})
        finish_reason = choice.get("finish_reason", "")

        tool_calls = []
        if message.get("tool_calls"):
            for tc in message["tool_calls"]:
                func = tc.get("function", {})
                try:
                    arguments = json.loads(func.get("arguments", "{}"))
                except json.JSONDecodeError:
                    arguments = {}
                tool_calls.append(
                    ToolCall(
                        id=tc.get("id", ""),
                        name=func.get("name", ""),
                        arguments=arguments,
                    )
                )

        return CloudResponse(
            text=message.get("content", "") or "",
            tool_calls=tool_calls,
            finish_reason=finish_reason,
        )
