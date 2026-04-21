"""Handler for ``paramem migrate-status``.

Queries ``/migration/status``.  In Slice 2 the server does not implement this
endpoint yet; the handler degrades gracefully on 404.
"""

from __future__ import annotations

import argparse
import json
import sys

from paramem.cli import http_client


def run(args: argparse.Namespace) -> int:
    """Execute the ``migrate-status`` subcommand.

    GETs ``/migration/status`` and renders the response as ``key: value`` lines
    (or raw JSON with ``--json``).  Degrades gracefully on 404.

    Parameters
    ----------
    args:
        Parsed namespace from the ``migrate-status`` subparser.

    Returns
    -------
    int
        0 on success, 1 on HTTP error / 404, 2 on connection failure.
    """
    url = f"{args.server_url}/migration/status"
    try:
        result = http_client.get_json(url)
    except http_client.ServerUnavailable:
        print(
            f"paramem migrate-status: the server at {args.server_url} does not\n"
            f"implement /migration/status yet (ships in Slice 3). Check\n"
            "`paramem --version` and server version are aligned.",
            file=sys.stderr,
        )
        return 1
    except http_client.ServerUnreachable:
        print(
            f"paramem migrate-status: server unreachable at {args.server_url}.\n"
            "Is paramem-server running? `systemctl --user status paramem-server`.",
            file=sys.stderr,
        )
        return 2
    except http_client.ServerHTTPError as exc:
        print(
            f"paramem migrate-status: server returned HTTP {exc.status_code} from {exc.url}.\n"
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
