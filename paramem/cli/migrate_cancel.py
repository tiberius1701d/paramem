"""Handler for ``paramem migrate-cancel``.

POSTs to ``/migration/cancel``.  Clears the in-memory STAGING stash and
returns the server to LIVE state.  Valid only when a candidate is staged;
returns 409 ``not_staging`` otherwise.

This module is intentionally minimal — the cancel logic is entirely
server-side.  The CLI is a thin stateless HTTP client (spec §L187).
"""

from __future__ import annotations

import argparse
import json
import sys

from paramem.cli import http_client


def _parse_409_body(body: str) -> dict:
    """Attempt to parse a 409 response body as JSON.

    Returns the parsed dict, or an empty dict when the body is not valid JSON.

    Parameters
    ----------
    body:
        Raw response body string from the server.

    Returns
    -------
    dict
        Parsed JSON object, or ``{}`` on parse failure.
    """
    try:
        return json.loads(body)
    except (json.JSONDecodeError, ValueError):
        return {}


def run(args: argparse.Namespace) -> int:
    """Execute the ``migrate-cancel`` subcommand.

    POSTs to ``/migration/cancel``, prints the response, and exits 0 on
    success.  On HTTP error or connection failure, prints a message to
    stderr and exits non-zero.

    Special handling: a 409 ``not_staging`` response means no candidate is
    currently staged — the operator's intent is already satisfied, so the
    command prints a friendly message and exits 0 (idempotent cancel).
    Other 409 codes are reported generically and exit 1.

    Parameters
    ----------
    args:
        Parsed namespace from the ``migrate-cancel`` subparser.

    Returns
    -------
    int
        0 on success or 409 ``not_staging`` (intent already satisfied),
        1 on other HTTP errors / 404, 2 on connection failure.
    """
    url = f"{args.server_url}/migration/cancel"
    try:
        result = http_client.post_json(url)
    except http_client.ServerUnavailable:
        print(
            f"paramem migrate-cancel: the server at {args.server_url} returned 404 for\n"
            "/migration/cancel.\n"
            "Available migration endpoints: /migration/preview, /cancel, /status, /diff.\n"
            "Check `paramem --version` and server version are aligned.",
            file=sys.stderr,
        )
        return 1
    except http_client.ServerUnreachable:
        print(
            f"paramem migrate-cancel: server unreachable at {args.server_url}.\n"
            "Is paramem-server running? `systemctl --user status paramem-server`.",
            file=sys.stderr,
        )
        return 2
    except http_client.ServerHTTPError as exc:
        if exc.status_code == 409:
            parsed = _parse_409_body(exc.body)
            detail = parsed.get("detail", {})
            if isinstance(detail, dict) and detail.get("error") == "not_staging":
                print(
                    "paramem migrate-cancel: no candidate is currently staged; nothing to cancel."
                )
                return 0
        # All other 409 codes and non-409 HTTP errors.
        print(
            f"paramem migrate-cancel: server returned HTTP {exc.status_code} from {exc.url}.\n"
            f"{exc.body.strip() or '(empty response body)'}",
            file=sys.stderr,
        )
        return 1

    if getattr(args, "json", False):
        print(json.dumps(result, indent=2))
    else:
        for key, value in result.items():
            print(f"{key}: {value}")
    return 0
