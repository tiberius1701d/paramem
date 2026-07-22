"""Abstract base class for cloud LLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class CloudAgentConfig:
    """Which cloud provider to call, and with what credentials.

    Carries no on-off switch of its own.  Whether this agent may be
    instantiated at all is decided once, by
    :func:`paramem.cloud.admission.evaluate_cloud_egress`, from the
    ``cloud.enabled`` master switch plus this config's provider/model/
    endpoint — see :func:`paramem.cloud.providers.registry.get_cloud_agent`.

    ``api_key`` is a YAML surface (``${ENV}``-interpolated at load time) and
    is NOT the admission key source: ``get_cloud_agent`` overwrites it with
    the env-resolved key from the verdict, so the key that authenticates a
    request and the key that admitted it are always the same string.
    """

    provider: str = "openai"  # openai, anthropic, google, groq
    model: str = ""
    api_key: str = field(default="", repr=False)
    endpoint: str = ""  # optional custom endpoint override for this provider
    timeout_seconds: float = 90.0  # request timeout per call to this provider's API


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
