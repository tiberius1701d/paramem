"""Handler for ``paramem backup-create``.

POSTs to ``/backup/create`` and renders the outcome — written slots and
any skipped artifacts with their reasons.
"""

from __future__ import annotations

import argparse
import json
import sys

from paramem.cli import http_client

_DEFAULT_KINDS = ["config", "graph", "registry"]


def run(args: argparse.Namespace) -> int:
    """Execute the ``backup-create`` subcommand.

    Parses ``--kinds`` (comma-separated) and ``--label``, POSTs to
    ``/backup/create``, and renders the result.

    Parameters
    ----------
    args:
        Parsed namespace from the ``backup-create`` subparser.  Expected
        attributes: ``server_url`` (str), ``kinds`` (str), ``label``
        (str | None), ``json`` (bool).

    Returns
    -------
    int
        0 on success (including partial success with skips).
        1 on 400 kind_invalid, HTTP error, or ``success=False`` from the server.
        2 on server unreachable.
    """
    # Parse --kinds (comma-separated; empty list → default).
    kinds_raw = getattr(args, "kinds", "")
    if kinds_raw:
        kinds = [k.strip() for k in kinds_raw.split(",") if k.strip()]
    else:
        kinds = []
    if not kinds:
        kinds = _DEFAULT_KINDS

    label = getattr(args, "label", None)
    body = {"kinds": kinds, "label": label}

    url = f"{args.server_url}/backup/create"
    try:
        result = http_client.post_json(url, body)
    except http_client.ServerUnavailable:
        print(
            f"paramem backup-create: the server at {args.server_url} does not\n"
            "implement /backup/create yet. Check `paramem --version` and server\n"
            "version are aligned.",
            file=sys.stderr,
        )
        return 1
    except http_client.ServerUnreachable:
        print(
            f"paramem backup-create: server unreachable at {args.server_url}.\n"
            "Is paramem-server running? `systemctl --user status paramem-server`.",
            file=sys.stderr,
        )
        return 2
    except http_client.ServerHTTPError as exc:
        print(
            f"paramem backup-create: server returned HTTP {exc.status_code} from {exc.url}.\n"
            f"{exc.body.strip() or '(empty response body)'}",
            file=sys.stderr,
        )
        return 1

    if getattr(args, "json", False):
        print(json.dumps(result, indent=2))
        return 0

    # Non-JSON render.
    success = result.get("success", False)
    tier = result.get("tier", "manual")
    result_label = result.get("label")
    label_str = f", label={result_label!r}" if result_label else ""
    written_slots = result.get("written_slots", {})
    skipped_artifacts = result.get("skipped_artifacts", [])
    error = result.get("error")

    if not success:
        err_msg = error or "(unknown error)"
        print(f"paramem backup-create: backup failed — {err_msg}", file=sys.stderr)
        return 1

    print(f"Backup created (tier={tier}{label_str}).")
    for kind_name, slot_path in written_slots.items():
        print(f"  {kind_name:<10} → {slot_path}")
    for skip_item in skipped_artifacts:
        # skipped_artifacts is a list of {"kind": ..., "reason": ...} dicts.
        if isinstance(skip_item, dict):
            skip_kind = skip_item.get("kind", "?")
            skip_reason = skip_item.get("reason", "?")
        elif isinstance(skip_item, (list, tuple)) and len(skip_item) >= 2:
            skip_kind, skip_reason = skip_item[0], skip_item[1]
        else:
            skip_kind = str(skip_item)
            skip_reason = "unknown"
        print(f"  {skip_kind:<10} skipped: {skip_reason}")

    return 0
