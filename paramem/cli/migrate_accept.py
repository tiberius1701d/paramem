"""Handler for ``paramem migrate-accept``.

POSTs to ``/migration/accept``.  Promotes the trial config B to live, archives
the trial adapter, and returns the server to LIVE state.  Valid only when a
trial is active and gates have finished with an accept-eligible status.

This module mirrors the ``migrate_cancel.py`` shape exactly — thin stateless
HTTP client, server owns all state (spec §L187).
"""

from __future__ import annotations

import argparse
import json
import sys

from paramem.cli import http_client


def _parse_detail(body: str) -> dict:
    """Attempt to parse an HTTP error response body as JSON.

    Returns the ``detail`` sub-dict if present, else an empty dict.  Never
    raises — parsing failures are best-effort.

    Parameters
    ----------
    body:
        Raw response body string from the server.

    Returns
    -------
    dict
        Parsed ``detail`` dict, or ``{}`` on parse failure.
    """
    try:
        parsed = json.loads(body)
        if isinstance(parsed, dict) and isinstance(parsed.get("detail"), dict):
            return parsed["detail"]
        return {}
    except (json.JSONDecodeError, ValueError):
        return {}


def run(args: argparse.Namespace) -> int:
    """Execute the ``migrate-accept`` subcommand.

    POSTs to ``/migration/accept``, prints the response, and exits 0 on
    success.  On HTTP error or connection failure, prints a message to stderr
    and exits non-zero.

    Special handling:

    - 409 ``not_trial``: no trial active — intent already satisfied; exit 0
      (idempotent/informational, matches ``migrate-cancel`` / ``not_staging``
      precedent).
    - 409 ``gates_failed``: gates failed — only rollback is valid; exit 1.
    - 409 ``gates_not_finished``: gates still running; exit 1.
    - ``ServerUnreachable``: server not reachable; exit 2.

    Parameters
    ----------
    args:
        Parsed namespace from the ``migrate-accept`` subparser.

    Returns
    -------
    int
        0 on success or 409 ``not_trial`` (intent already satisfied),
        1 on other HTTP errors / 404 / gates-blocked conditions,
        2 on connection failure.
    """
    url = f"{args.server_url}/migration/accept"
    try:
        result = http_client.post_json(url)
    except http_client.ServerUnavailable:
        print(
            f"paramem migrate-accept: the server at {args.server_url} does not implement"
            " /migration/accept.\n"
            "Check `paramem --version` and server version are aligned. "
            "migrate-accept and migrate-rollback are separate subcommands for non-interactive use.",
            file=sys.stderr,
        )
        return 1
    except http_client.ServerUnreachable:
        print(
            f"paramem migrate-accept: server unreachable at {args.server_url}.\n"
            "Is paramem-server running? `systemctl --user status paramem-server`.",
            file=sys.stderr,
        )
        return 2
    except http_client.ServerHTTPError as exc:
        if exc.status_code == 409:
            detail = _parse_detail(exc.body)
            error = detail.get("error", "")
            if error == "not_trial":
                print("paramem migrate-accept: no trial active; nothing to accept.")
                return 0
            if error == "gates_failed":
                print(
                    "paramem migrate-accept: gates failed — only migrate-rollback is valid.",
                    file=sys.stderr,
                )
                return 1
            if error == "gates_not_finished":
                print(
                    "paramem migrate-accept: gates still running; wait or run migrate-status.",
                    file=sys.stderr,
                )
                return 1
        print(
            f"paramem migrate-accept: server returned HTTP {exc.status_code} from {exc.url}.\n"
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
