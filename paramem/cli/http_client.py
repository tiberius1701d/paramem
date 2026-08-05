"""Minimal httpx wrapper shared by all CLI subcommand modules.

Each call opens a fresh ``httpx.Client`` context — the CLI is stateless, so
connection pooling across subcommands buys nothing.

Exceptions are normalized to two public types so callers can handle them
without importing httpx directly:

- :exc:`ServerUnavailable` — the route itself is absent (older server); the
  404 body carries no structured ``detail`` object (FastAPI's own
  route-missing body is the string ``"Not Found"``, not a dict).
- :exc:`ServerUnreachable` — TCP refused / DNS failure / read timeout.

A 404 whose body DOES carry a structured ``detail`` object is an endpoint
that exists and is reporting its own well-formed "not found" outcome (e.g.
``/migration/accept`` with no trial active) — that raises
:exc:`ServerHTTPError` like any other HTTP error, so the command-level
handler can inspect ``detail["error"]`` and respond accordingly instead of
being told the route doesn't exist.

:func:`parse_error_detail` is the one shared parser for a
:class:`ServerHTTPError`'s JSON body — every subcommand that needs to branch
on the server's ``detail.error`` discriminator calls it rather than keeping a
local copy; it is also what distinguishes the two 404 shapes above.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx

from paramem.utils.paths import find_project_root


def _repo_env_path(start: Path) -> Path | None:
    """Walk from *start* toward the filesystem root looking for a ``pyproject.toml``.

    Returns the path to the ``.env`` file beside it when found, regardless of
    whether that file actually exists (existence is checked by the caller).
    Returns ``None`` when no ``pyproject.toml`` anchor is found.

    Factored out of :func:`resolve_token` so tests can exercise the walk
    against a temp tree by passing an arbitrary *start* path without
    monkeypatching module-level ``__file__``.
    """
    root = find_project_root(start)
    return root / ".env" if root is not None else None


def resolve_token(*, allow_files: bool | None = None) -> str | None:
    """Resolve the CLI bearer token for server authentication.

    This is a CLIENT-side resolution order for the CLI's own outbound
    ``Authorization`` header — not a mirror of any server-side token
    resolution.  The server has no such thing: ``PARAMEM_API_TOKEN`` is no
    longer a credential the server validates at all (every credential lives
    in a :class:`~paramem.server.user_tokens.UserTokenStore`; see
    ``paramem/server/auth.py``'s module docstring).  The value this function
    returns must itself be a token minted via ``paramem mint-user-token`` —
    these three locations are just where an operator may have PUT that
    minted value, layered as fallbacks so the CLI keeps working across
    ``.env``/secret-file/env-var placement:

    1. Ambient environment variable ``PARAMEM_API_TOKEN``.
    2. Per-secret file ``~/.config/paramem/secrets/PARAMEM_API_TOKEN``
       (populated by ``paramem/server/secret_store.py`` on the server side;
       the ``.env`` line may be deleted once this file exists).
    3. Repo ``.env`` file beside ``pyproject.toml`` (walk up from this file).

    Returns ``None`` when no token is present — auth-OFF servers keep working
    because the ``Authorization`` header is omitted when the token is absent.

    Parameters
    ----------
    allow_files:
        Controls whether the on-disk fallbacks (secret file + repo ``.env``)
        are consulted.  Defaults to reading the ``PARAMEM_CLI_NO_TOKEN_FILES``
        environment flag, which tests set to ``"1"`` for hermetic runs so the
        real repo ``.env`` and ``~/.config`` are never touched.  Pass
        ``allow_files`` explicitly in unit tests instead of monkeypatching
        module globals.

    Notes
    -----
    The token is never logged — callers receive it as a plain return value and
    the ``Authorization`` header is assembled inside :func:`get_json` /
    :func:`post_json` without any logging.
    """
    tok = os.environ.get("PARAMEM_API_TOKEN", "").strip()
    if tok:
        return tok
    if allow_files is None:
        allow_files = os.environ.get("PARAMEM_CLI_NO_TOKEN_FILES") != "1"
    if not allow_files:
        return None
    # Per-secret file (server migration target; .env line may be deleted).
    secret = Path.home() / ".config" / "paramem" / "secrets" / "PARAMEM_API_TOKEN"
    if secret.is_file():
        v = secret.read_text(encoding="utf-8").strip().strip('"').strip("'")
        if v:
            return v
    # Repo .env fallback: walk up to the directory holding pyproject.toml.
    env_path = _repo_env_path(Path(__file__))
    if env_path is not None and env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("PARAMEM_API_TOKEN="):
                v = line.split("=", 1)[1].strip().strip('"').strip("'")
                return v or None
    return None


class ServerUnavailable(Exception):
    """Server returned 404 with no structured ``detail`` body — the route itself is absent.

    The feature has not been implemented in the server version currently
    running.  The caller should print a version-alignment message and exit 1.
    A 404 that DOES carry a structured ``detail`` object is an existing
    endpoint reporting its own "not found" outcome and raises
    :exc:`ServerHTTPError` instead — see this module's docstring.
    """


class ServerUnreachable(Exception):
    """TCP connection refused, DNS resolution failure, or read timeout.

    The server is not running or is not reachable at the given URL.  The
    caller should print a troubleshooting hint and exit 2.
    """


class ServerHTTPError(Exception):
    """Server responded with a non-2xx status other than 404.

    Typically a 5xx from the server or a 4xx other than 404 (e.g. 400
    validation failure).  The caller should surface the status code and body
    to the operator and exit 1.
    """

    def __init__(self, status_code: int, url: str, body: str) -> None:
        self.status_code = status_code
        self.url = url
        self.body = body
        super().__init__(f"HTTP {status_code} from {url}")


def parse_error_detail(body: str) -> dict:
    """Return the ``detail`` object from a FastAPI JSON error body, or ``{}``.

    Best-effort: never raises.  *body* is typically :attr:`ServerHTTPError.body`.
    A non-JSON body, a JSON body without a ``detail`` key, or a ``detail``
    that is not itself an object all return ``{}`` rather than raising.

    Parameters
    ----------
    body:
        Raw response body string from the server.

    Returns
    -------
    dict
        The parsed ``detail`` dict, or ``{}`` when it is absent or malformed.
    """
    try:
        parsed = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        return {}
    if isinstance(parsed, dict) and isinstance(parsed.get("detail"), dict):
        return parsed["detail"]
    return {}


def get_json(url: str, *, timeout: float = 5.0, token: str | None = None) -> dict:
    """Perform a GET request and return the parsed JSON body.

    Parameters
    ----------
    url:
        Absolute URL to request.
    timeout:
        Request timeout in seconds.
    token:
        Bearer token for the ``Authorization`` header.  When ``None``
        (the default), :func:`resolve_token` is called to auto-resolve from
        the environment, secret file, or repo ``.env``.  Pass an explicit
        string to override; pass ``""`` (empty string) to force no header.

    Returns
    -------
    dict
        Parsed JSON response body.

    Raises
    ------
    ServerUnavailable
        HTTP 404 with no structured ``detail`` body — the route itself is
        absent (older server).
    ServerUnreachable
        If the TCP connection fails or times out.
    ServerHTTPError
        Any non-2xx response other than a route-missing 404 — including a
        404 that DOES carry a structured ``detail`` (an existing endpoint
        reporting its own "not found" outcome; see this module's docstring).
    """
    if token is None:
        token = resolve_token()
    try:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        with httpx.Client(timeout=timeout) as client:
            response = client.get(url, headers=headers)
    except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError) as exc:
        raise ServerUnreachable(str(exc)) from exc

    if response.status_code == 404 and not parse_error_detail(response.text):
        raise ServerUnavailable(f"404 from {url}")
    if response.status_code >= 400:
        raise ServerHTTPError(response.status_code, url, response.text)
    return response.json()


def post_json(
    url: str,
    body: dict | None = None,
    *,
    timeout: float = 5.0,
    token: str | None = None,
) -> dict:
    """Perform a POST request with an optional JSON body and return the parsed response.

    Parameters
    ----------
    url:
        Absolute URL to request.
    body:
        Optional dict serialized as the JSON request body.  When ``None``,
        the request is sent with no body.
    timeout:
        Request timeout in seconds.
    token:
        Bearer token for the ``Authorization`` header.  When ``None``
        (the default), :func:`resolve_token` is called to auto-resolve from
        the environment, secret file, or repo ``.env``.  Pass an explicit
        string to override; pass ``""`` (empty string) to force no header.

    Returns
    -------
    dict
        Parsed JSON response body.

    Raises
    ------
    ServerUnavailable
        HTTP 404 with no structured ``detail`` body — the route itself is
        absent (older server).
    ServerUnreachable
        If the TCP connection fails or times out.
    ServerHTTPError
        Any non-2xx response other than a route-missing 404 — including a
        404 that DOES carry a structured ``detail`` (an existing endpoint
        reporting its own "not found" outcome; see this module's docstring).
    """
    if token is None:
        token = resolve_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(url, json=body, headers=headers)
    except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError) as exc:
        raise ServerUnreachable(str(exc)) from exc

    if response.status_code == 404 and not parse_error_detail(response.text):
        raise ServerUnavailable(f"404 from {url}")
    if response.status_code >= 400:
        raise ServerHTTPError(response.status_code, url, response.text)
    return response.json()
