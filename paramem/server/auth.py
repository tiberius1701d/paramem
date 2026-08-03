"""Bearer-token authentication middleware for the ParaMem REST server.

Opt-in via a populated :class:`~paramem.server.user_tokens.UserTokenStore`
(per-user tokens, wired by the app lifespan — see
``paramem.server.app._build_user_token_store``).  There is no separate
shared-token credential: every accepted token is a ``UserTokenStore`` entry,
attributed to a speaker or not.  ``PARAMEM_API_TOKEN`` survives only as the
CARRIER env var infra consumers (the systemd scheduling timer, the HA custom
component) read to source their own ``Authorization`` header value — the
value itself must be a token minted into the store (typically an
unattributed admin token, ``mint-user-token --unattributed --scope admin
--force-admin``).  This module does not use that env var as a credential —
its ONE read of it, in :func:`log_startup_posture`, is a fail-open migration
guard: a pre-migration deployment that set it as the old shared-token
credential and never minted a per-user token would otherwise land OPEN
silently on upgrade, with the operator's old token ignored and no signal
that anything changed.

Behavior:
- No user-token store wired (getter is ``None`` or returns ``None``) → no
  token check is performed (any request passes through), but the request is
  stamped with the non-admin ``"chat"`` scope, so administrative endpoints
  (gated by ``require_admin``) stay locked until a store is configured —
  fail-closed admin.  A single WARN is emitted at startup ("auth disabled").
- A user-token store is wired → every REST request must carry a valid token
  in ``Authorization: Bearer <token>`` or the configured cookie.  Missing or
  invalid tokens return HTTP 401 with a JSON error.  A wired store with zero
  active tokens stays **fail-closed** (all requests 401) so that revoking the
  last token does not silently re-open every endpoint.

Two-mode (Security OFF/ON) model:

- **OFF** — no user-token store is wired.  Chat/voice endpoints (unguarded by
  any scope check) remain reachable; admin endpoints 403 via
  ``require_admin`` until a store is configured (i.e. until the first
  ``mint-user-token``).
- **ON** — a :class:`~paramem.server.user_tokens.UserTokenStore` is wired via
  ``user_token_getter``.  Every accepted token carries a capability scope
  (``chat`` / ``admin``) and, for an attributed token, a ``speaker_id`` —
  authorized requests have ``scope["state"]["speaker_id"]`` set to the
  matched speaker; an unattributed token leaves it unset.

Token extraction order (per request):
1. ``Authorization: Bearer <token>`` header.
2. Cookie named by ``cookie_name`` (default ``"paramem_token"``).

Path exemption:
- Paths in ``exempt_paths`` or starting with any prefix in ``exempt_prefixes``
  pass through without token checks (e.g. PWA shell, manifest).

Per-user tokens use a dict lookup keyed by ``sha256(token)`` — no iteration,
no timing exposure from a linear scan.

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

import http.cookies
import json
import logging
import os
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from paramem.server.user_tokens import UserTokenStore

logger = logging.getLogger(__name__)

AUTH_HEADER = "authorization"
BEARER_PREFIX = "Bearer "

_DEFAULT_COOKIE_NAME = "paramem_token"
_DEFAULT_EXEMPT_PATHS = ("/", "/manifest.json")


class BearerTokenMiddleware:
    """Pure ASGI middleware requiring a valid bearer token on protected paths.

    Accepts tokens via ``Authorization: Bearer <token>`` header or a cookie
    named by *cookie_name*.  Header takes precedence over cookie.

    Single authentication path — **per-user**: dict lookup on
    ``sha256(presented)`` against the
    :class:`~paramem.server.user_tokens.UserTokenStore` returned by
    *user_token_getter*.  Authorized requests carry
    ``scope["state"]["scope"]`` (``"chat"`` / ``"admin"``) and, for an
    attributed token, ``scope["state"]["speaker_id"]``.

    When *user_token_getter* is ``None`` (or returns ``None``), no token is
    required (Security OFF), but the request is stamped with the non-admin
    ``"chat"`` scope rather than ``"admin"`` — fail-closed admin: chat/voice
    endpoints stay reachable, administrative endpoints 403 via
    ``require_admin`` until a store is configured.  A wired store with zero
    active tokens is **fail-closed** (401) so that revoking the last token
    does not silently open every endpoint.

    *cookie_name_getter* is an optional zero-arg callable that returns the
    effective cookie name at request time.  When provided and it returns a
    truthy string, that string is used; otherwise the static *cookie_name*
    default applies.  This allows the cookie name to be driven from a config
    that loads after the middleware is constructed (e.g. FastAPI lifespan).
    """

    def __init__(
        self,
        app,
        user_token_getter: Callable[[], "UserTokenStore | None"] | None = None,
        cookie_name: str = _DEFAULT_COOKIE_NAME,
        cookie_name_getter: Callable[[], str | None] | None = None,
        exempt_paths: tuple[str, ...] = _DEFAULT_EXEMPT_PATHS,
        exempt_prefixes: tuple[str, ...] = (),
    ) -> None:
        # Note: exempt_prefixes entries should end with "/" (e.g. "/app/") to
        # avoid "/app" inadvertently matching "/application/secret".
        self.app = app
        self._user_token_getter = user_token_getter
        self._cookie_name = cookie_name
        self._cookie_name_getter = cookie_name_getter
        self._exempt_paths = set(exempt_paths)
        self._exempt_prefixes = tuple(exempt_prefixes)

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

        # Dynamic enablement: ON iff a user-token store is wired.  A wired
        # store with zero active tokens stays fail-closed (401) so that
        # revoking the last token does not silently open every endpoint.
        store: UserTokenStore | None = (
            self._user_token_getter() if self._user_token_getter else None
        )
        if store is None:
            # Security OFF — fail-closed admin: the server is usable (chat/voice
            # endpoints reachable, no scope gate on them) but administrative
            # endpoints stay locked until a store is configured.  Stamp the
            # same non-admin sentinel used for a per-user "chat" token (see
            # ``user_tokens._DEFAULT_SCOPE``) so ``require_admin`` 403s exactly
            # as it does for a real chat-scope token — no separate sentinel.
            scope.setdefault("state", {})["scope"] = "chat"
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

        # Per-user token — sha256 dict lookup (no iteration).
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


def log_startup_posture(n_user_tokens: int = 0, per_user_active: bool = False) -> None:
    """Emit a single startup line describing the auth posture.

    Two states, keyed on *store presence* — the same condition the
    middleware's ``__call__`` checks per-request (``store is None`` → OFF):

    - **OFF** — no per-user store is wired.
    - **ON** — a per-user store is wired.  Sub-variant on whether it has any
      active tokens yet (fail-closed either way).

    Call once during app lifespan-startup after the
    :class:`~paramem.server.user_tokens.UserTokenStore` is assigned.  The
    *n_user_tokens* count is informational only — a wired store with zero
    active tokens stays **fail-closed** (401) until a token is minted; the
    posture is still ON.

    Fail-open migration guard (this module's ONE read of
    ``PARAMEM_API_TOKEN`` — see the module docstring): when the OFF branch
    fires AND the env var is set, an additional LOUD warning fires.  A
    deployment that ran under the old shared-token model has the env var
    set, PWA off, and never minted a per-user token — on upgrade it lands
    OPEN (the OFF branch above already logs that), with the operator's old
    token now silently inert.  The env var's value is never read as a
    credential here, only checked for presence.

    Parameters
    ----------
    n_user_tokens:
        Number of active (non-revoked) per-user tokens currently in the store.
        Used only as informational text in the log message.
    per_user_active:
        ``True`` when a :class:`~paramem.server.user_tokens.UserTokenStore` is
        wired into the middleware (i.e. the store object is not ``None``).
        Drives the ON branch regardless of token count so that a
        wired-but-empty store logs ON (fail-closed), not OFF.
    """
    if per_user_active:
        if n_user_tokens == 0:
            logger.info(
                "AUTH: ON (store wired, 0 active token(s) — "
                "fail-closed, mint one to grant access) — "
                "all REST endpoints require a bearer token",
            )
        else:
            logger.info(
                "AUTH: ON (%d active token(s)) — all REST endpoints require a bearer token",
                n_user_tokens,
            )
    else:
        logger.warning(
            "AUTH: OFF (no per-user token store wired) — "
            "all REST endpoints are reachable without credentials. "
            "Run `paramem mint-user-token` to enable. "
            "See SECURITY.md for the authentication model.",
        )
        if os.environ.get("PARAMEM_API_TOKEN"):
            logger.warning(
                "PARAMEM_API_TOKEN is set but is NO LONGER a credential — "
                "the server does not check it. This deployment appears to be "
                "upgrading from the old shared-token model: the token above "
                "is silently ignored and every REST endpoint is open (see the "
                "AUTH: OFF warning above). Run `paramem mint-user-token` to "
                "mint a real per-user token, then set PARAMEM_API_TOKEN to "
                "that value to protect this server again. See SECURITY.md "
                "for the migration.",
            )
