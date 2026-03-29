"""Google Gemini cloud agent adapter using the official google-genai SDK."""

import logging

from google import genai

from paramem.server.cloud.base import CloudAgent, CloudResponse
from paramem.server.config import GeneralAgentConfig

logger = logging.getLogger(__name__)


class GoogleAgent(CloudAgent):
    """Adapter for Google's Gemini API via the official SDK."""

    def __init__(self, config: GeneralAgentConfig):
        super().__init__(config)
        self._client = genai.Client(api_key=config.api_key)

    def call(
        self,
        query: str,
        system_prompt: str = "",
        tool_results: list[dict] | None = None,
        tools: list[dict] | None = None,
        history: list[dict] | None = None,
    ) -> CloudResponse:
        contents = []

        # Include conversation history before the current query
        if history:
            for turn in history:
                role = turn.get("role", "user")
                text = turn.get("text", "")
                if role in ("user", "assistant") and text:
                    # Gemini uses "user" and "model" roles
                    gemini_role = "model" if role == "assistant" else "user"
                    contents.append(
                        genai.types.Content(
                            role=gemini_role,
                            parts=[genai.types.Part(text=text)],
                        )
                    )

        contents.append(query)

        config = genai.types.GenerateContentConfig(
            max_output_tokens=1024,
        )
        if system_prompt:
            config.system_instruction = system_prompt

        try:
            response = self._client.models.generate_content(
                model=self.config.model,
                contents=contents,
                config=config,
            )
        except Exception as e:
            logger.error("Google agent error: %s", e)
            return CloudResponse(text="I couldn't get an answer from the cloud service.")

        text = response.text or ""
        finish_reason = ""
        if response.candidates:
            fr = response.candidates[0].finish_reason
            finish_reason = fr.name if fr else ""

        return CloudResponse(text=text, finish_reason=finish_reason)

    def format_tools(self, tools: list[dict]) -> list[dict]:
        """Convert standard tool definitions to Google format.

        Not used for SOTA reasoning path (no tools), but required by ABC.
        """
        return [
            {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
            }
            for tool in tools
        ]
