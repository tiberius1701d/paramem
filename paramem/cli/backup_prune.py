"""Handler for ``paramem backup-prune``.

POSTs to ``/backup/prune`` and renders the retention-policy outcome —
deleted slots, preserved-immune slots, pre-migration window slots, and
disk-usage deltas.
"""

from __future__ import annotations

import argparse
import json
import sys

from paramem.cli import http_client


def run(args: argparse.Namespace) -> int:
    """Execute the ``backup-prune`` subcommand.

    POSTs ``{"dry_run": <bool>}`` to ``/backup/prune`` and renders the
    pruning outcome with before/after disk usage.

    Parameters
    ----------
    args:
        Parsed namespace from the ``backup-prune`` subparser.  Expected
        attributes: ``server_url`` (str), ``dry_run`` (bool),
        ``json`` (bool).

    Returns
    -------
    int
        0 on success (200).  1 on HTTP error.  2 on server unreachable.
    """
    dry_run = getattr(args, "dry_run", False)
    body = {"dry_run": dry_run}

    url = f"{args.server_url}/backup/prune"
    try:
        result = http_client.post_json(url, body)
    except http_client.ServerUnavailable:
        print(
            f"paramem backup-prune: the server at {args.server_url} does not\n"
            "implement /backup/prune yet. Check `paramem --version` and server\n"
            "version are aligned.",
            file=sys.stderr,
        )
        return 1
    except http_client.ServerUnreachable:
        print(
            f"paramem backup-prune: server unreachable at {args.server_url}.\n"
            "Is paramem-server running? `systemctl --user status paramem-server`.",
            file=sys.stderr,
        )
        return 2
    except http_client.ServerHTTPError as exc:
        print(
            f"paramem backup-prune: server returned HTTP {exc.status_code} from {exc.url}.\n"
            f"{exc.body.strip() or '(empty response body)'}",
            file=sys.stderr,
        )
        return 1

    if getattr(args, "json", False):
        print(json.dumps(result, indent=2))
        return 0

    # Non-JSON render.
    result_dry_run = result.get("dry_run", dry_run)
    deleted = result.get("deleted", [])
    preserved_immune = result.get("preserved_immune", [])
    preserved_window = result.get("preserved_pre_migration_window", [])
    would_delete_next = result.get("would_delete_next", [])
    usage_before = result.get("disk_usage_before", {})
    usage_after = result.get("disk_usage_after", {})

    def _fmt_usage(usage: dict) -> str:
        total = usage.get("total_bytes", 0)
        cap = usage.get("cap_bytes", 0)
        pct = usage.get("pct_of_cap", 0.0)
        total_gb = total / (1024**3)
        cap_gb = cap / (1024**3)
        return f"{total_gb:.2f} GB / {cap_gb:.2f} GB ({pct * 100:.1f}%)"

    print(f"Backup prune (dry-run: {'true' if result_dry_run else 'false'}).")
    print(f"  before: {_fmt_usage(usage_before)}")
    print(f"  after:  {_fmt_usage(usage_after)}")

    if result_dry_run:
        print(f"  would delete next:  {len(would_delete_next)} slots")
        for path in would_delete_next:
            print(f"    {path}")
    else:
        print(f"  deleted:            {len(deleted)} slots")
        for path in deleted:
            print(f"    {path}")

    if preserved_immune:
        print(f"  preserved (immune): {len(preserved_immune)} slots")
        for path in preserved_immune:
            print(f"    {path}")

    if preserved_window:
        print(f"  preserved (pre-migration window): {len(preserved_window)} slots")
        for path in preserved_window:
            print(f"    {path}")

    return 0
