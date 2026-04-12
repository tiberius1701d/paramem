"""Abstract base class for cloud LLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from paramem.server.config import CloudAgentConfig


@dataclass
class ToolCall:
    """A tool call requested by the cloud model."""

    id: str
    name: str
    arguments: dict


@dataclass
class CloudResponse:
    """Response from a cloud agent call."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = ""

    @property
    def requires_tool_execution(self) -> bool:
        return len(self.tool_calls) > 0


class CloudAgent(ABC):
    """Abstract interface for cloud LLM providers.

    Each provider adapter implements this interface. The server calls
    `call()` and gets back a `CloudResponse` — either final text or
    tool calls that need execution.
    """

    def __init__(self, config: CloudAgentConfig):
        self.config = config

    @abstractmethod
    def call(
        self,
        query: str,
        system_prompt: str = "",
        tool_results: list[dict] | None = None,
        tools: list[dict] | None = None,
        history: list[dict] | None = None,
    ) -> CloudResponse:
        """Send a query to the cloud model.

        Args:
            query: The user query.
            system_prompt: Optional system prompt.
            tool_results: Results from previous tool calls in the agentic loop.
            tools: Tool definitions in the standard internal format.
            history: Optional conversation history as list of
                {"role": "user"|"assistant", "text": "..."} dicts.

        Returns:
            CloudResponse with either final text or tool calls.
        """

    @abstractmethod
    def format_tools(self, tools: list[dict]) -> list[dict]:
        """Convert standard tool definitions to provider-specific format."""

    def is_available(self) -> bool:
        """Check if the provider is configured and reachable.

        Requires either an API key or a custom endpoint (for keyless
        providers like ollama).
        """
        return bool(self.config.enabled and (self.config.api_key or self.config.endpoint))
