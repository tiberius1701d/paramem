"""Bearer-token authentication middleware for the ParaMem REST server.

Opt-in via the ``PARAMEM_API_TOKEN`` environment variable.

Behavior:
- Token **unset** → middleware is a no-op. A single WARN is emitted at startup
  ("auth disabled"). Preserves current behavior for existing deployments.
- Token **set** → every REST request must carry ``Authorization: Bearer <token>``.
  Missing or mismatching tokens return HTTP 401 with a JSON error.

Constant-time comparison avoids timing-based token discovery. Token is read
once at middleware construction; to rotate, restart the server with the new
value.

Implemented as pure ASGI middleware — Starlette's ``BaseHTTPMiddleware``
wraps every request/response in a task-group and memory-object-stream which
can deadlock on certain FastAPI request patterns (observed: second POST
/migration/preview after /migration/cancel hangs indefinitely even in
no-op mode). Pure ASGI is the Starlette-recommended pattern for auth.

Wyoming :10300 / :10301 sockets are NOT covered here — the Wyoming protocol
has no native auth concept. Those endpoints are protected at the network
layer by the Windows Firewall rule scoping (``PARAMEM_NAS_IP``).
"""

from __future__ import annotations

import hmac
import json
import logging
import os

logger = logging.getLogger(__name__)

TOKEN_ENV_VAR = "PARAMEM_API_TOKEN"
AUTH_HEADER = "authorization"
BEARER_PREFIX = "Bearer "


class BearerTokenMiddleware:
    """Pure ASGI middleware requiring ``Authorization: Bearer <token>``.

    When the configured token is empty, the middleware is a pass-through.
    """

    def __init__(self, app, token: str) -> None:
        self.app = app
        self._token = token
        self._enabled = bool(token)

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http" or not self._enabled:
            await self.app(scope, receive, send)
            return

        header_value = b""
        for name, value in scope.get("headers", []):
            if name == AUTH_HEADER.encode():
                header_value = value
                break

        header = header_value.decode("latin-1")
        if not header.startswith(BEARER_PREFIX):
            await _send_unauthorized(send, "missing or malformed Authorization header")
            return

        presented = header[len(BEARER_PREFIX) :].strip()
        if not hmac.compare_digest(presented, self._token):
            await _send_unauthorized(send, "invalid bearer token")
            return

        await self.app(scope, receive, send)


async def _send_unauthorized(send, detail: str) -> None:
    body = json.dumps({"error": "unauthorized", "detail": detail}).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": 401,
            "headers": [
                (b"content-type", b"application/json"),
                (b"www-authenticate", b'Bearer realm="paramem"'),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


def load_token_from_env() -> str:
    """Return the configured token, or an empty string if unset."""
    return os.environ.get(TOKEN_ENV_VAR, "").strip()


def log_startup_posture(token: str) -> None:
    """Emit a single startup line describing the auth posture.

    Call once during app lifespan-startup. Matches the Security-ON/OFF
    log convention used by the encryption-key switch.
    """
    if token:
        logger.info("AUTH: ON (%s set) — all REST endpoints require bearer token", TOKEN_ENV_VAR)
    else:
        logger.warning(
            "AUTH: OFF (%s unset) — all REST endpoints are reachable without credentials. "
            "Set %s to enable. See SECURITY.md for the authentication model.",
            TOKEN_ENV_VAR,
            TOKEN_ENV_VAR,
        )
