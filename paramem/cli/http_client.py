"""Minimal httpx wrapper shared by all CLI subcommand modules.

Each call opens a fresh ``httpx.Client`` context — the CLI is stateless, so
connection pooling across subcommands buys nothing.

Exceptions are normalized to two public types so callers can handle them
without importing httpx directly:

- :exc:`ServerUnavailable` — server returned 404; feature not in this version.
- :exc:`ServerUnreachable` — TCP refused / DNS failure / read timeout.
"""

from __future__ import annotations

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

    Resolution order mirrors the server's own load order
    (``paramem/server/secret_store.py`` + ``app.py``):

    1. Ambient environment variable ``PARAMEM_API_TOKEN``.
    2. Per-secret file ``~/.config/paramem/secrets/PARAMEM_API_TOKEN``
       (the server's migration target; the ``.env`` line may be deleted).
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
    """Server returned 404 for the requested endpoint.

    The feature has not been implemented in the server version currently
    running.  The caller should print a version-alignment message and exit 1.
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
        If the server responds with HTTP 404.
    ServerUnreachable
        If the TCP connection fails or times out.
    ServerHTTPError
        For any non-2xx, non-404 response (e.g. 5xx or 400).
    """
    if token is None:
        token = resolve_token()
    try:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        with httpx.Client(timeout=timeout) as client:
            response = client.get(url, headers=headers)
    except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError) as exc:
        raise ServerUnreachable(str(exc)) from exc

    if response.status_code == 404:
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
        If the server responds with HTTP 404.
    ServerUnreachable
        If the TCP connection fails or times out.
    ServerHTTPError
        For any non-2xx, non-404 response (e.g. 5xx or 400).
    """
    if token is None:
        token = resolve_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(url, json=body, headers=headers)
    except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError) as exc:
        raise ServerUnreachable(str(exc)) from exc

    if response.status_code == 404:
        raise ServerUnavailable(f"404 from {url}")
    if response.status_code >= 400:
        raise ServerHTTPError(response.status_code, url, response.text)
    return response.json()
