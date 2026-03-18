"""Config flow for ParaMem integration."""

from typing import Any

import aiohttp
import voluptuous as vol
from homeassistant import config_entries

from .const import DEFAULT_SERVER_URL, DEFAULT_TIMEOUT, DOMAIN


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

            if await self._test_connection(server_url, user_input.get("timeout", DEFAULT_TIMEOUT)):
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
                }
            ),
            errors=errors,
        )

    async def _test_connection(self, server_url: str, timeout: int) -> bool:
        """Test that the ParaMem server is reachable."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{server_url}/status",
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as resp:
                    return resp.status == 200
        except (aiohttp.ClientError, TimeoutError):
            return False
