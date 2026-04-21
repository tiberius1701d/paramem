"""Minimal httpx wrapper shared by all CLI subcommand modules.

Each call opens a fresh ``httpx.Client`` context — the CLI is stateless, so
connection pooling across subcommands buys nothing.

Exceptions are normalized to two public types so callers can handle them
without importing httpx directly:

- :exc:`ServerUnavailable` — server returned 404; feature not in this version.
- :exc:`ServerUnreachable` — TCP refused / DNS failure / read timeout.
"""

from __future__ import annotations

import httpx


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


def get_json(url: str, *, timeout: float = 5.0) -> dict:
    """Perform a GET request and return the parsed JSON body.

    Parameters
    ----------
    url:
        Absolute URL to request.
    timeout:
        Request timeout in seconds.

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
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(url)
    except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError) as exc:
        raise ServerUnreachable(str(exc)) from exc

    if response.status_code == 404:
        raise ServerUnavailable(f"404 from {url}")
    if response.status_code >= 400:
        raise ServerHTTPError(response.status_code, url, response.text)
    return response.json()


def post_json(url: str, body: dict | None = None, *, timeout: float = 5.0) -> dict:
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
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(url, json=body)
    except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError) as exc:
        raise ServerUnreachable(str(exc)) from exc

    if response.status_code == 404:
        raise ServerUnavailable(f"404 from {url}")
    if response.status_code >= 400:
        raise ServerHTTPError(response.status_code, url, response.text)
    return response.json()
