"""Handler for ``paramem backup-list``.

GETs ``/backup/list`` and renders one row per backup slot.  Supports
optional ``--kind`` filtering and ``--json`` raw-output mode.
"""

from __future__ import annotations

import argparse
import json
import sys

from paramem.cli import http_client


def run(args: argparse.Namespace) -> int:
    """Execute the ``backup-list`` subcommand.

    Performs ``GET /backup/list`` (with optional ``?kind=<kind>`` query
    parameter) and renders a formatted table of backup slots.

    Parameters
    ----------
    args:
        Parsed namespace from the ``backup-list`` subparser.  Expected
        attributes: ``server_url`` (str), ``kind`` (str | None),
        ``json`` (bool).

    Returns
    -------
    int
        0 on success, 1 on HTTP error / 400, 2 on server unreachable.
    """
    kind = getattr(args, "kind", None)
    if kind:
        url = f"{args.server_url}/backup/list?kind={kind}"
    else:
        url = f"{args.server_url}/backup/list"

    try:
        result = http_client.get_json(url)
    except http_client.ServerUnavailable:
        print(
            f"paramem backup-list: the server at {args.server_url} does not\n"
            "implement /backup/list yet. Check `paramem --version` and server\n"
            "version are aligned.",
            file=sys.stderr,
        )
        return 1
    except http_client.ServerUnreachable:
        print(
            f"paramem backup-list: server unreachable at {args.server_url}.\n"
            "Is paramem-server running? `systemctl --user status paramem-server`.",
            file=sys.stderr,
        )
        return 2
    except http_client.ServerHTTPError as exc:
        print(
            f"paramem backup-list: server returned HTTP {exc.status_code} from {exc.url}.\n"
            f"{exc.body.strip() or '(empty response body)'}",
            file=sys.stderr,
        )
        return 1

    if getattr(args, "json", False):
        print(json.dumps(result, indent=2))
        return 0

    # Non-JSON render.
    disk_used = result.get("disk_used_bytes", 0)
    disk_cap = result.get("disk_cap_bytes", 0)
    used_gb = disk_used / (1024**3)
    cap_gb = disk_cap / (1024**3)
    print(f"disk: {used_gb:.2f} GB / {cap_gb:.2f} GB")
    print()

    items = result.get("items", [])
    if not items:
        print("No backup slots found.")
        return 0

    # Header row.
    hdr = f"{'backup_id':<22}  {'kind':<10}  {'tier':<14}  {'timestamp':<20}  {'size':>8}  label"
    print(hdr)
    print("-" * 90)
    for item in items:
        backup_id = item.get("backup_id", "?")
        item_kind = item.get("kind", "?")
        tier = item.get("tier", "?")
        timestamp = item.get("timestamp", "?")[:19]  # truncate to seconds
        size_bytes = item.get("size_bytes", 0)
        # Human-readable size.
        if size_bytes >= 1024 * 1024:
            size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
        elif size_bytes >= 1024:
            size_str = f"{size_bytes / 1024:.1f} KB"
        else:
            size_str = f"{size_bytes} B"
        label = item.get("label") or "—"
        print(
            f"{backup_id:<22}  {item_kind:<10}  {tier:<14}  {timestamp:<20}  {size_str:>8}  {label}"
        )

    return 0
