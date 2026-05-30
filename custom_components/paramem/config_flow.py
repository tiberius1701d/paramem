"""Config flow for ParaMem integration."""

from typing import Any

import aiohttp
import voluptuous as vol
from homeassistant import config_entries

from .const import CONF_API_TOKEN, DEFAULT_SERVER_URL, DEFAULT_TIMEOUT, DOMAIN


class ParaMemConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for ParaMem."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.ConfigFlowResult:
        """Handle the initial configuration step."""
        errors: dict[str, str] = {}

        if user_input is not None:
            server_url = user_input["server_url"].rstrip("/")
            user_input["server_url"] = server_url

            if await self._test_connection(
                server_url,
                user_input.get("timeout", DEFAULT_TIMEOUT),
                user_input.get(CONF_API_TOKEN, ""),
            ):
                return self.async_create_entry(
                    title=f"ParaMem ({server_url})",
                    data=user_input,
                )
            errors["base"] = "cannot_connect"

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema(
                {
                    vol.Required("server_url", default=DEFAULT_SERVER_URL): str,
                    vol.Optional("timeout", default=DEFAULT_TIMEOUT): vol.All(
                        vol.Coerce(int), vol.Range(min=1, max=120)
                    ),
                    vol.Optional(CONF_API_TOKEN, default=""): str,
                }
            ),
            errors=errors,
        )

    async def _test_connection(
        self, server_url: str, timeout: int, api_token: str = ""
    ) -> bool:
        """Test that the ParaMem server is reachable.

        Sends ``Authorization: Bearer <token>`` when *api_token* is non-empty so
        the connection test works against an auth-enabled ParaMem server.
        """
        headers = {"Authorization": f"Bearer {api_token}"} if api_token else {}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{server_url}/status",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as resp:
                    return resp.status == 200
        except (aiohttp.ClientError, TimeoutError):
            return False
