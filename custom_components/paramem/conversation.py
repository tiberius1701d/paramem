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

from .const import DEFAULT_FALLBACK_AGENT, DEFAULT_TIMEOUT, DOMAIN

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

        # Pass HA user display name as speaker identity
        speaker = None
        if user_input.context and user_input.context.user_id:
            user = await self.hass.auth.async_get_user(user_input.context.user_id)
            if user:
                speaker = user.name

        payload = {
            "text": user_input.text,
            "conversation_id": user_input.conversation_id or "default",
            "speaker": speaker,
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
                    follow_up = data.get("follow_up")
        except (aiohttp.ClientError, TimeoutError) as err:
            logger.warning("ParaMem server unavailable (%s), falling back to HA agent", err)
            response_text = await self._fallback_to_ha(user_input)
            follow_up = None

        # Record assistant response in chat log (without follow-up prompt)
        chat_log.async_add_assistant_content_without_tools(
            AssistantContent(
                agent_id=user_input.agent_id,
                content=response_text,
            )
        )

        # Append follow-up after recording — spoken by TTS but not in history
        if follow_up:
            response_text = f"{response_text} ... {follow_up}"

        # Build HA IntentResponse
        response = intent.IntentResponse(language=user_input.language)
        response.async_set_speech(response_text)

        return ConversationResult(
            response=response,
            conversation_id=chat_log.conversation_id,
        )

    async def _fallback_to_ha(self, user_input: ConversationInput) -> str:
        """Fall back to HA's default conversation agent when ParaMem is unavailable."""
        try:
            result = await self.hass.services.async_call(
                "conversation",
                "process",
                {"text": user_input.text, "agent_id": DEFAULT_FALLBACK_AGENT},
                blocking=True,
                return_response=True,
            )
            if result and "response" in result:
                resp = result["response"]
                # HA conversation responses have speech in response.speech.plain.speech
                # but may also be directly in response.speech as a string
                speech = resp.get("speech", {})
                if isinstance(speech, str):
                    return speech
                plain = speech.get("plain", {})
                if isinstance(plain, str):
                    return plain
                text = plain.get("speech", "")
                if text:
                    return text
            logger.warning("HA fallback returned unexpected format: %s", result)
        except Exception:
            logger.exception("HA conversation fallback also failed")
        return "I'm having trouble connecting to my services right now."


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
