"""Anthropic cloud agent adapter using the official SDK."""

import logging

import anthropic

from paramem.server.cloud.base import CloudAgent, CloudResponse
from paramem.server.config import GeneralAgentConfig

logger = logging.getLogger(__name__)


class AnthropicAgent(CloudAgent):
    """Adapter for Anthropic's Messages API via the official SDK."""

    def __init__(self, config: GeneralAgentConfig):
        super().__init__(config)
        self._client = anthropic.Anthropic(api_key=config.api_key, timeout=30.0)

    def call(
        self,
        query: str,
        system_prompt: str = "",
        tool_results: list[dict] | None = None,
        tools: list[dict] | None = None,
        history: list[dict] | None = None,
    ) -> CloudResponse:
        messages = []

        # Include conversation history before the current query
        if history:
            for turn in history:
                role = turn.get("role", "user")
                text = turn.get("text", "")
                if role in ("user", "assistant") and text:
                    messages.append({"role": role, "content": text})

        messages.append({"role": "user", "content": query})

        kwargs: dict = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": 1024,
        }

        # Use caller-supplied tools if provided, otherwise enable web search
        if tools:
            kwargs["tools"] = self.format_tools(tools)
        else:
            kwargs["tools"] = [{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}]

        if system_prompt:
            kwargs["system"] = system_prompt

        try:
            response = self._client.messages.create(**kwargs)
        except anthropic.APITimeoutError:
            logger.error("Anthropic agent timed out")
            return CloudResponse(text="I couldn't reach the cloud service in time.")
        except anthropic.APIStatusError as e:
            logger.error("Anthropic agent API error: %s", e.status_code)
            return CloudResponse(text="I couldn't get an answer from the cloud service.")
        except anthropic.APIConnectionError as e:
            logger.error("Anthropic agent connection error: %s", e)
            return CloudResponse(text="I couldn't connect to the cloud service.")

        # Extract text from content blocks
        text_parts = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)

        return CloudResponse(
            text=" ".join(text_parts),
            finish_reason=response.stop_reason or "",
        )

    def format_tools(self, tools: list[dict]) -> list[dict]:
        """Convert standard tool definitions to Anthropic format.

        Not used for SOTA reasoning path (no tools), but required by ABC.
        """
        return [
            {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "input_schema": tool.get("parameters", {"type": "object", "properties": {}}),
            }
            for tool in tools
        ]
