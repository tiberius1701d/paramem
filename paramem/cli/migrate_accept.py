"""Handler for ``paramem migrate-accept``.

POSTs to ``/migration/accept``.  Promotes the trial config B to live, archives
the trial adapter, and returns the server to LIVE state.  Valid only when a
trial is active and gates have finished with an accept-eligible status.

This module mirrors the ``migrate_cancel.py`` shape exactly — thin stateless
HTTP client, server owns all state.
"""

from __future__ import annotations

import argparse
import json
import sys

from paramem.cli import http_client
from paramem.cli.migrate import _render_apply_result


def run(args: argparse.Namespace) -> int:
    """Execute the ``migrate-accept`` subcommand.

    POSTs to ``/migration/accept``, prints the response, and exits 0 on
    success.  On HTTP error or connection failure, prints a message to stderr
    and exits non-zero.

    Special handling:

    - 409 ``not_trial`` / 404 ``not_found``: no trial active — intent already
      satisfied; exit 0 (idempotent/informational, matches ``migrate-cancel``
      / ``not_staging`` precedent).  Both codes mean the same thing: the
      server rejects with 404 ``not_found`` when ``state == "LIVE"`` and 409
      ``not_trial`` when ``state == "STAGING"``.
    - 409 ``gates_failed``: gates failed — only rollback is valid; exit 1.
    - 409 ``gates_not_finished``: gates still running; exit 1.
    - ``ServerUnreachable``: server not reachable; exit 2.
    - A plain 404 with no structured ``detail`` (route absent on an older
      server) raises ``ServerUnavailable`` instead of ``ServerHTTPError`` —
      handled separately below.

    Parameters
    ----------
    args:
        Parsed namespace from the ``migrate-accept`` subparser.

    Returns
    -------
    int
        0 on success or not_trial/not_found (intent already satisfied),
        1 on other HTTP errors / route-absent 404 / gates-blocked conditions,
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
        if exc.status_code in (404, 409):
            detail = http_client.parse_error_detail(exc.body)
            error = detail.get("error", "")
            if error in ("not_trial", "not_found"):
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
        print("Migration accepted.")
        _render_apply_result(result, args.server_url)
    return 0
