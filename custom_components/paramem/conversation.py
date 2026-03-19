"""ParaMem conversation entity — forwards turns to the ParaMem server."""

import logging

import aiohttp
from homeassistant.components.conversation import (
    AssistantContent,
    ChatLog,
    ConversationEntity,
    ConversationInput,
    ConversationResult,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import intent
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import DEFAULT_TIMEOUT, DOMAIN

logger = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the ParaMem conversation entity."""
    data = hass.data[DOMAIN][config_entry.entry_id]
    async_add_entities(
        [
            ParaMemConversationEntity(
                config_entry=config_entry,
                server_url=data["server_url"],
                timeout=data.get("timeout", DEFAULT_TIMEOUT),
            )
        ]
    )


class ParaMemConversationEntity(ConversationEntity):
    """Conversation entity that forwards to the ParaMem server."""

    _attr_has_entity_name = True
    _attr_name = "ParaMem"

    def __init__(
        self,
        config_entry: ConfigEntry,
        server_url: str,
        timeout: int,
    ) -> None:
        self._server_url = server_url
        self._timeout = timeout
        self._attr_unique_id = config_entry.entry_id

    @property
    def supported_languages(self) -> list[str]:
        return ["en"]

    async def _async_handle_message(
        self,
        user_input: ConversationInput,
        chat_log: ChatLog,
    ) -> ConversationResult:
        """Forward the user message to the ParaMem server."""
        history = _extract_history(chat_log)

        payload = {
            "text": user_input.text,
            "conversation_id": user_input.conversation_id or "default",
            "history": history,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self._server_url}/chat",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self._timeout),
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    response_text = data.get("text", "")
        except aiohttp.ClientError as err:
            logger.error("ParaMem server error: %s", err)
            response_text = "Sorry, I couldn't reach my memory server."
        except TimeoutError:
            logger.error(
                "ParaMem server timed out after %ds", self._timeout
            )
            response_text = (
                "Sorry, my memory server took too long to respond."
            )

        # Record assistant response in chat log
        chat_log.async_add_assistant_content_without_tools(
            AssistantContent(
                agent_id=user_input.agent_id,
                content=response_text,
            )
        )

        # Build HA IntentResponse
        response = intent.IntentResponse(
            language=user_input.language
        )
        response.async_set_speech(response_text)

        return ConversationResult(
            response=response,
            conversation_id=chat_log.conversation_id,
        )


def _extract_history(chat_log: ChatLog) -> list[dict]:
    """Extract conversation history from the HA ChatLog.

    ChatLog entries are typed objects (UserContent, AssistantContent).
    We infer role from the type name and extract the content field.
    """
    history = []
    for entry in chat_log.content:
        content = getattr(entry, "content", None)
        if not content:
            continue
        # Infer role from class name: UserContent → user, AssistantContent → assistant
        type_name = type(entry).__name__.lower()
        if "user" in type_name:
            role = "user"
        elif "assistant" in type_name:
            role = "assistant"
        else:
            continue
        history.append({"role": role, "text": str(content)})
    return history
