"""Handler for ``paramem migrate <path>``.

Sends a preview request to ``/migration/preview``.  In Slice 2 the server does
not implement this endpoint yet; the handler degrades gracefully on 404.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from paramem.cli import http_client


def run(args: argparse.Namespace) -> int:
    """Execute the ``migrate`` subcommand.

    Validates that ``args.path`` is an absolute path, then POSTs to
    ``/migration/preview``.  On :exc:`~paramem.cli.http_client.ServerUnavailable`
    (HTTP 404) prints a version-alignment message and returns 1.  On
    :exc:`~paramem.cli.http_client.ServerUnreachable` prints a
    troubleshooting hint and returns 2.

    Parameters
    ----------
    args:
        Parsed namespace from the ``migrate`` subparser.

    Returns
    -------
    int
        0 on success, 1 on HTTP error / 404, 2 on connection failure.
    """
    candidate = Path(args.path)
    if not candidate.is_absolute():
        print(
            f"paramem migrate: path must be absolute, got {args.path!r}.\n"
            "Example: paramem migrate /home/user/configs/server-new.yaml",
            file=sys.stderr,
        )
        return 1

    url = f"{args.server_url}/migration/preview"
    try:
        result = http_client.post_json(url, {"candidate_path": str(candidate)})
    except http_client.ServerUnavailable:
        print(
            f"paramem migrate: the server at {args.server_url} does not\n"
            f"implement /migration/preview yet (ships in Slice 3). Check\n"
            "`paramem --version` and server version are aligned.",
            file=sys.stderr,
        )
        return 1
    except http_client.ServerUnreachable:
        print(
            f"paramem migrate: server unreachable at {args.server_url}.\n"
            "Is paramem-server running? `systemctl --user status paramem-server`.",
            file=sys.stderr,
        )
        return 2
    except http_client.ServerHTTPError as exc:
        print(
            f"paramem migrate: server returned HTTP {exc.status_code} from {exc.url}.\n"
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
