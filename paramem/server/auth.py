"""Bearer-token authentication middleware for the ParaMem REST server.

Opt-in via the ``PARAMEM_API_TOKEN`` environment variable (shared legacy token)
and/or a populated :class:`~paramem.server.user_tokens.UserTokenStore` (per-user
tokens, wired by the app lifespan).

Behavior:
- Token **unset** and no user-token store wired → middleware is a no-op.  A
  single WARN is emitted at startup ("auth disabled").  Preserves current
  behavior for existing deployments.
- Token **set** or a user-token store is wired → every REST request must carry
  a valid token in ``Authorization: Bearer <token>`` or the configured cookie.
  Missing or invalid tokens return HTTP 401 with a JSON error.  A wired store
  with zero active tokens stays **fail-closed** (all requests 401) so that
  revoking the last token does not silently re-open every endpoint.

Two-mode (Security OFF/ON) model:

- **OFF** — ``_enabled`` (static shared-token check) is ``False`` AND no
  user-token store is wired (getter is ``None`` or returns ``None``).
- **ON-shared** — ``PARAMEM_API_TOKEN`` is set.  All requests validated via
  constant-time comparison.  No ``speaker_id`` attached (legacy = unattributed).
- **ON-per-user** — a :class:`~paramem.server.user_tokens.UserTokenStore` is
  wired via ``user_token_getter`` and has active tokens.  Authorized requests
  have ``scope["state"]["speaker_id"]`` set to the matched speaker.
- **ON-both** — both shared token and per-user store active.  Shared token
  checked first; per-user store is the fallback.

Token extraction order (per request):
1. ``Authorization: Bearer <token>`` header.
2. Cookie named by ``cookie_name`` (default ``"paramem_token"``).

Path exemption:
- Paths in ``exempt_paths`` or starting with any prefix in ``exempt_prefixes``
  pass through without token checks (e.g. PWA shell, manifest).

Constant-time comparison (``hmac.compare_digest``) applies to the legacy shared
token.  Per-user tokens use a dict lookup keyed by ``sha256(token)`` — no
iteration required.

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
import http.cookies
import json
import logging
import os
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from paramem.server.user_tokens import UserTokenStore

logger = logging.getLogger(__name__)

TOKEN_ENV_VAR = "PARAMEM_API_TOKEN"
AUTH_HEADER = "authorization"
BEARER_PREFIX = "Bearer "

_DEFAULT_COOKIE_NAME = "paramem_token"
_DEFAULT_EXEMPT_PATHS = ("/", "/manifest.json")


class BearerTokenMiddleware:
    """Pure ASGI middleware requiring a valid bearer token on protected paths.

    Accepts tokens via ``Authorization: Bearer <token>`` header or a cookie
    named by *cookie_name*.  Header takes precedence over cookie.

    Two authentication paths:
    - **Shared (legacy)**: constant-time comparison against *token*.
    - **Per-user**: dict lookup on ``sha256(presented)`` against the
      :class:`~paramem.server.user_tokens.UserTokenStore` returned by
      *user_token_getter*.  Authorized requests carry
      ``scope["state"]["speaker_id"]``.

    When *token* is empty and *user_token_getter* is ``None`` (or returns
    ``None``), the middleware is a **pass-through** (Security OFF).  A wired
    store with zero active tokens is **fail-closed** (401) so that revoking
    the last token does not silently open every endpoint.

    The ``enabled`` property reflects the *static* shared-token part only.
    Dynamic per-user enablement is computed per request from the store.

    *cookie_name_getter* is an optional zero-arg callable that returns the
    effective cookie name at request time.  When provided and it returns a
    truthy string, that string is used; otherwise the static *cookie_name*
    default applies.  This allows the cookie name to be driven from a config
    that loads after the middleware is constructed (e.g. FastAPI lifespan).
    """

    def __init__(
        self,
        app,
        token: str,
        user_token_getter: Callable[[], "UserTokenStore | None"] | None = None,
        cookie_name: str = _DEFAULT_COOKIE_NAME,
        cookie_name_getter: Callable[[], str | None] | None = None,
        exempt_paths: tuple[str, ...] = _DEFAULT_EXEMPT_PATHS,
        exempt_prefixes: tuple[str, ...] = (),
    ) -> None:
        # Note: exempt_prefixes entries should end with "/" (e.g. "/app/") to
        # avoid "/app" inadvertently matching "/application/secret".
        self.app = app
        self._token = token
        self._user_token_getter = user_token_getter
        self._cookie_name = cookie_name
        self._cookie_name_getter = cookie_name_getter
        self._exempt_paths = set(exempt_paths)
        self._exempt_prefixes = tuple(exempt_prefixes)

    @property
    def enabled(self) -> bool:
        """``True`` when the static shared token is configured.

        Note: dynamic per-user enablement is computed per request from the
        :class:`~paramem.server.user_tokens.UserTokenStore`.  This property
        does not reflect per-user state.
        """
        return bool(self._token)

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Path exemptions — pass through before any token check.
        path: str = scope.get("path", "")
        if path in self._exempt_paths or any(
            path.startswith(prefix) for prefix in self._exempt_prefixes
        ):
            await self.app(scope, receive, send)
            return

        # Dynamic enablement: ON when shared token set OR a user-token store is
        # wired.  A wired store with zero active tokens stays fail-closed (401)
        # so that revoking the last token does not silently open every endpoint.
        # has_active_tokens() is used only for the startup posture log.
        store: UserTokenStore | None = (
            self._user_token_getter() if self._user_token_getter else None
        )
        enabled = bool(self._token) or (store is not None)
        if not enabled:
            # Security OFF — the whole server is open by design (no credential
            # configured).  Stamp admin scope so the ``require_admin`` gate
            # stays a pure ``scope == "admin"`` check and OFF-mode requests
            # reach every endpoint as before.
            scope.setdefault("state", {})["scope"] = "admin"
            await self.app(scope, receive, send)
            return

        # Resolve the effective cookie name: prefer the getter's live value
        # (allows config-driven override after middleware construction), fall
        # back to the static default.
        effective_cookie_name = (
            (self._cookie_name_getter() or self._cookie_name)
            if self._cookie_name_getter
            else self._cookie_name
        )
        presented = _extract_token(scope, effective_cookie_name)
        if presented is None:
            await _send_unauthorized(send, "missing or malformed Authorization header")
            return

        # Shared (legacy) token — constant-time comparison.
        if self._token and hmac.compare_digest(presented, self._token):
            # Shared token always has admin scope.  No speaker_id is attached
            # (legacy unattributed semantics).
            scope.setdefault("state", {})["scope"] = "admin"
            await self.app(scope, receive, send)
            return

        # Per-user token — sha256 dict lookup (no iteration).
        if store is not None:
            record = store.resolve(presented)
            if record is not None:
                _authenticated, sid, scope_val = record
                scope.setdefault("state", {})["scope"] = scope_val
                if sid is not None:
                    scope["state"]["speaker_id"] = sid
                await self.app(scope, receive, send)
                return

        await _send_unauthorized(send, "invalid bearer token")


def _extract_token(scope: dict, cookie_name: str) -> str | None:
    """Extract the bearer token from the request headers.

    Checks ``Authorization: Bearer <token>`` first; falls back to the named
    cookie.  Returns ``None`` when neither source yields a usable token.

    Parameters
    ----------
    scope:
        ASGI HTTP scope dict.
    cookie_name:
        Name of the cookie carrying the bearer token.

    Returns
    -------
    str | None
        The raw token string, or ``None`` if absent.
    """
    raw_auth = b""
    raw_cookie = b""

    for name, value in scope.get("headers", []):
        if name == b"authorization":
            raw_auth = value
        elif name == b"cookie":
            raw_cookie = value

    # Authorization header takes precedence.
    header = raw_auth.decode("latin-1")
    if header.startswith(BEARER_PREFIX):
        return header[len(BEARER_PREFIX) :].strip()

    # Cookie fallback.  Bound the input before handing it to SimpleCookie to
    # prevent a quadratic-complexity DoS via an oversized Cookie header.
    _COOKIE_HEADER_MAX = 8192
    if raw_cookie and len(raw_cookie) <= _COOKIE_HEADER_MAX:
        try:
            jar = http.cookies.SimpleCookie()
            jar.load(raw_cookie.decode("latin-1"))
            morsel = jar.get(cookie_name)
            if morsel is not None:
                # Normalise whitespace-only cookie values to None so they
                # are treated as absent (fail-closed) rather than flowing
                # to the token comparison as an all-spaces string.
                return (morsel.value or "").strip() or None
        except http.cookies.CookieError:
            pass

    return None


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


def log_startup_posture(token: str, n_user_tokens: int = 0, per_user_active: bool = False) -> None:
    """Emit a single startup line describing the auth posture.

    Four states, keyed on *store presence* (matching the middleware enablement
    rule ``enabled = bool(shared_token) or (store is not None)``):

    - **OFF** — no shared token and per-user store is not wired.
    - **ON-shared** — shared token set, per-user store not wired.
    - **ON-per-user** — per-user store is wired (fail-closed), no shared token.
    - **ON-both** — shared token set *and* per-user store is wired.

    Call once during app lifespan-startup after the
    :class:`~paramem.server.user_tokens.UserTokenStore` is assigned.  The
    *n_user_tokens* count is informational only — a wired store with zero active
    tokens stays **fail-closed** (401) until a token is minted; the posture is
    still ON-per-user.

    Backward-compatible: callers that pass only *token* (and omit both
    *n_user_tokens* and *per_user_active*) retain the prior two-state OFF /
    ON-shared behavior.

    Parameters
    ----------
    token:
        The static shared token string (empty when not configured).
    n_user_tokens:
        Number of active (non-revoked) per-user tokens currently in the store.
        Used only as informational text in the log message.
    per_user_active:
        ``True`` when a :class:`~paramem.server.user_tokens.UserTokenStore` is
        wired into the middleware (i.e. the store object is not ``None``).
        Drives the ON-per-user / ON-both branches regardless of token count so
        that a wired-but-empty store logs ON-per-user (fail-closed), not OFF.
    """
    has_shared = bool(token)

    if has_shared and per_user_active:
        logger.info(
            "AUTH: ON-both (%s set + %d active per-user token(s)) — "
            "all REST endpoints require a bearer token",
            TOKEN_ENV_VAR,
            n_user_tokens,
        )
    elif has_shared:
        logger.info(
            "AUTH: ON-shared (%s set) — all REST endpoints require bearer token",
            TOKEN_ENV_VAR,
        )
    elif per_user_active:
        if n_user_tokens == 0:
            logger.info(
                "AUTH: ON-per-user (store wired, 0 active per-user tokens — "
                "fail-closed, mint one to grant access) — "
                "all REST endpoints require a bearer token",
            )
        else:
            logger.info(
                "AUTH: ON-per-user (%d active per-user token(s)) — "
                "all REST endpoints require a bearer token",
                n_user_tokens,
            )
    else:
        logger.warning(
            "AUTH: OFF (%s unset, no per-user token store wired) — "
            "all REST endpoints are reachable without credentials. "
            "Set %s to enable. See SECURITY.md for the authentication model.",
            TOKEN_ENV_VAR,
            TOKEN_ENV_VAR,
        )
