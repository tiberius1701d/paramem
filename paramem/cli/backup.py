"""Handler for ``paramem backup list``.

Queries ``/backup/list``.  In Slice 2 the server does not implement this
endpoint yet; the handler degrades gracefully on 404.
"""

from __future__ import annotations

import argparse
import json
import sys

from paramem.cli import http_client


def run(args: argparse.Namespace) -> int:
    """Execute the ``backup list`` subcommand.

    GETs ``/backup/list`` and renders a row per backup slot.  On 404 prints the
    Slice 6 version-alignment message.  The success render path is tested with a
    mocked response so Slice 6 does not have to re-plumb it.

    Parameters
    ----------
    args:
        Parsed namespace from the ``backup list`` subparser.

    Returns
    -------
    int
        0 on success, 1 on HTTP error / 404, 2 on connection failure.
    """
    url = f"{args.server_url}/backup/list"
    try:
        result = http_client.get_json(url)
    except http_client.ServerUnavailable:
        print(
            f"paramem backup list: the server at {args.server_url} does not\n"
            f"implement /backup/list yet (ships in Slice 6). Check\n"
            "`paramem --version` and server version are aligned.",
            file=sys.stderr,
        )
        return 1
    except http_client.ServerUnreachable:
        print(
            f"paramem backup list: server unreachable at {args.server_url}.\n"
            "Is paramem-server running? `systemctl --user status paramem-server`.",
            file=sys.stderr,
        )
        return 2
    except http_client.ServerHTTPError as exc:
        print(
            f"paramem backup list: server returned HTTP {exc.status_code} from {exc.url}.\n"
            f"{exc.body.strip() or '(empty response body)'}",
            file=sys.stderr,
        )
        return 1

    if getattr(args, "json", False):
        print(json.dumps(result, indent=2))
    else:
        slots = result.get("slots", [])
        if not slots:
            print("No backup slots found.")
        else:
            for slot in slots:
                kind = slot.get("kind", "?")
                slot_id = slot.get("slot_id", "?")
                created_at = slot.get("created_at", "?")
                encrypted = slot.get("encrypted", False)
                print(f"{kind}  {slot_id}  {created_at}  {'encrypted' if encrypted else 'plain'}")
    return 0
