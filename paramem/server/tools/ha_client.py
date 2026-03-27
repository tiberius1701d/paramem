"""Home Assistant REST and WebSocket client for tool execution.

Uses connection pooling via httpx for low-latency repeated calls
during agentic loops. Supports service calls with return_response
(HA 2024.1+), state queries, and entity name resolution.

WebSocket is used for script-type tools that need response_variable
capture — the REST API cannot capture script return values unless
the script declares response_variables: at the entity level.
"""

import json as json_mod
import logging

import httpx
from rapidfuzz import fuzz

logger = logging.getLogger(__name__)

# Minimum fuzzy match score for entity name resolution
ENTITY_MATCH_THRESHOLD = 70


class HAClient:
    """Persistent HTTP client for Home Assistant REST API."""

    def __init__(self, url: str, token: str, timeout: float = 3.0):
        self._base_url = url.rstrip("/")
        self._token = token
        self._timeout = timeout
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        self._client = httpx.Client(
            base_url=self._base_url,
            headers=self._headers,
            timeout=self._timeout,
        )
        # friendly_name_lower → entity_id
        self._entity_map: dict[str, str] = {}
        # WebSocket URL derived from REST URL
        ws_scheme = "wss" if self._base_url.startswith("https") else "ws"
        rest_base = self._base_url.split("://", 1)[1]
        self._ws_url = f"{ws_scheme}://{rest_base}/api/websocket"

    def _get_client(self) -> httpx.Client:
        """Return pooled client, recreating if closed."""
        if self._client.is_closed:
            self._client = httpx.Client(
                base_url=self._base_url,
                headers=self._headers,
                timeout=self._timeout,
            )
        return self._client

    def health_check(self) -> dict | None:
        """Check HA connectivity and version. Returns API status or None."""
        try:
            resp = self._get_client().get("/api/")
            resp.raise_for_status()
            return resp.json()
        except (httpx.HTTPError, httpx.RequestError) as e:
            logger.warning("HA health check failed: %s", e)
            return None

    def call_service(
        self,
        domain: str,
        service: str,
        data: dict | None = None,
        return_response: bool = False,
    ) -> dict | list | None:
        """Call an HA service.

        Args:
            domain: Service domain (e.g., "light", "script").
            service: Service name (e.g., "turn_on", "music_control_ma").
            data: Service data payload.
            return_response: If True, request script response_variable
                (HA 2024.1+).

        Returns:
            Service response (state changes or script return value),
            or None on failure.
        """
        payload = dict(data) if data else {}
        if return_response:
            payload["return_response"] = True

        url = f"/api/services/{domain}/{service}"
        try:
            resp = self._get_client().post(url, json=payload)
            resp.raise_for_status()
            return resp.json()
        except httpx.TimeoutException:
            logger.error("HA service call timed out: %s/%s", domain, service)
            return None
        except httpx.HTTPStatusError as e:
            logger.error(
                "HA service call failed: %s/%s → %s",
                domain,
                service,
                e.response.status_code,
            )
            return None
        except httpx.RequestError as e:
            logger.error("HA connection error: %s", e)
            return None

    def load_entity_map(self) -> int:
        """Fetch all HA entity states and build friendly_name → entity_id map.

        Call once at startup. Returns the number of entities indexed.
        """
        try:
            resp = self._get_client().get("/api/states")
            resp.raise_for_status()
            states = resp.json()
        except (httpx.HTTPError, httpx.RequestError) as e:
            logger.error("Failed to load HA entity states: %s", e)
            return 0

        self._entity_map.clear()
        for state in states:
            entity_id = state.get("entity_id", "")
            friendly = state.get("attributes", {}).get("friendly_name", "")
            if entity_id and friendly:
                self._entity_map[friendly.lower()] = entity_id
        logger.info("Entity map: %d entities indexed", len(self._entity_map))
        return len(self._entity_map)

    def resolve_entity_name(self, value: str) -> str:
        """Resolve a friendly name or partial name to an HA entity_id.

        Returns the entity_id if a match is found, otherwise the original value.
        Already-valid entity IDs (containing a dot) are returned as-is.
        """
        if not value or not self._entity_map:
            return value

        # Already an entity_id
        if "." in value and value.split(".")[0].isalpha():
            return value

        value_lower = value.lower()

        # Exact match
        if value_lower in self._entity_map:
            entity_id = self._entity_map[value_lower]
            logger.info("Entity resolved (exact): '%s' → %s", value, entity_id)
            return entity_id

        # Fuzzy match — find the best scoring friendly name
        best_score = 0
        best_entity_id = None
        for friendly_lower, entity_id in self._entity_map.items():
            score = fuzz.ratio(value_lower, friendly_lower)
            if score > best_score:
                best_score = score
                best_entity_id = entity_id
            # Also check if the input is a substring of the friendly name
            if value_lower in friendly_lower:
                sub_score = max(score, ENTITY_MATCH_THRESHOLD + 5)
                if sub_score > best_score:
                    best_score = sub_score
                    best_entity_id = entity_id

        if best_score >= ENTITY_MATCH_THRESHOLD and best_entity_id:
            logger.info(
                "Entity resolved (fuzzy %.0f%%): '%s' → %s",
                best_score,
                value,
                best_entity_id,
            )
            return best_entity_id

        return value

    def resolve_arguments(self, arguments: dict) -> dict:
        """Resolve any entity-like argument values to HA entity IDs.

        Scans all string values in the arguments dict and attempts
        resolution. Non-matching values are left unchanged.
        """
        if not self._entity_map:
            return arguments

        resolved = {}
        for key, val in arguments.items():
            if isinstance(val, str):
                resolved[key] = self.resolve_entity_name(val)
            else:
                resolved[key] = val
        return resolved

    def render_template(self, template: str) -> str | None:
        """Render a Jinja2 template against HA state via /api/template."""
        try:
            resp = self._get_client().post("/api/template", json={"template": template})
            resp.raise_for_status()
            return resp.text
        except httpx.TimeoutException:
            logger.error("HA template render timed out")
            return None
        except httpx.HTTPStatusError as e:
            logger.error("HA template render failed: %s", e.response.status_code)
            return None
        except httpx.RequestError as e:
            logger.error("HA template connection error: %s", e)
            return None

    def execute_script_ws(
        self,
        sequence: list[dict],
        timeout: float | None = None,
    ) -> dict | None:
        """Execute a raw automation sequence via HA WebSocket API.

        Runs the sequence with full response_variable capture — the
        same mechanism HA's internal automation engine uses. No script
        entity needed; the sequence runs directly.

        Returns the response dict, or None on failure.
        """
        import websockets.sync.client as ws_sync

        ws_timeout = timeout or self._timeout

        message = {
            "id": 1,
            "type": "execute_script",
            "sequence": sequence,
        }

        try:
            with ws_sync.connect(self._ws_url, close_timeout=2, open_timeout=ws_timeout) as ws:
                msg = json_mod.loads(ws.recv(timeout=ws_timeout))
                if msg.get("type") != "auth_required":
                    logger.error("WS: unexpected message: %s", msg.get("type"))
                    return None

                ws.send(json_mod.dumps({"type": "auth", "access_token": self._token}))
                msg = json_mod.loads(ws.recv(timeout=ws_timeout))
                if msg.get("type") != "auth_ok":
                    logger.error("WS: auth failed: %s", msg.get("message", ""))
                    return None

                ws.send(json_mod.dumps(message))
                msg = json_mod.loads(ws.recv(timeout=ws_timeout))

                if msg.get("success"):
                    return msg.get("result", {}).get("response")
                else:
                    error = msg.get("error", {})
                    logger.error(
                        "WS execute_script failed: %s",
                        error.get("message", "unknown"),
                    )
                    return None

        except Exception as e:
            logger.error("WS execute_script error: %s", e)
            return None

    def conversation_process(
        self,
        text: str,
        agent_id: str = "conversation.groq",
        timeout: float | None = None,
    ) -> str | None:
        """Forward a query to an HA conversation agent.

        Calls conversation.process via WebSocket, which runs the full
        agent pipeline (system prompt, tools, entity resolution) inside
        HA. Returns the speech response text, or None on failure.
        """
        import websockets.sync.client as ws_sync

        ws_timeout = timeout or max(self._timeout, 30.0)

        try:
            with ws_sync.connect(self._ws_url, close_timeout=2, open_timeout=ws_timeout) as ws:
                msg = json_mod.loads(ws.recv(timeout=ws_timeout))
                if msg.get("type") != "auth_required":
                    return None

                ws.send(json_mod.dumps({"type": "auth", "access_token": self._token}))
                msg = json_mod.loads(ws.recv(timeout=ws_timeout))
                if msg.get("type") != "auth_ok":
                    return None

                ws.send(
                    json_mod.dumps(
                        {
                            "id": 1,
                            "type": "call_service",
                            "domain": "conversation",
                            "service": "process",
                            "service_data": {
                                "agent_id": agent_id,
                                "text": text,
                            },
                            "return_response": True,
                        }
                    )
                )
                msg = json_mod.loads(ws.recv(timeout=ws_timeout))

                if msg.get("success"):
                    response = msg.get("result", {}).get("response", {})
                    speech = (
                        response.get("response", {})
                        .get("speech", {})
                        .get("plain", {})
                        .get("speech", "")
                    )
                    return speech if speech else None
                else:
                    error = msg.get("error", {})
                    logger.error(
                        "HA conversation.process failed: %s",
                        error.get("message", "unknown"),
                    )
                    return None

        except Exception as e:
            logger.error("HA conversation.process error: %s", e)
            return None

    def get_services(self) -> dict | None:
        """Get all available HA services (for auto-discovery)."""
        try:
            resp = self._get_client().get("/api/services")
            resp.raise_for_status()
            return resp.json()
        except (httpx.HTTPError, httpx.RequestError) as e:
            logger.error("HA services query failed: %s", e)
            return None

    def close(self):
        """Close the pooled connection."""
        if self._client and not self._client.is_closed:
            self._client.close()
