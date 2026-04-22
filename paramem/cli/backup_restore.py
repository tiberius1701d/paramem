"""Handler for ``paramem backup-restore``.

POSTs to ``/backup/restore`` with a ``backup_id`` and renders the outcome.
Handles 409 (STAGING/TRIAL/consolidating) and 400 (wrong kind) with
operator-actionable messages.
"""

from __future__ import annotations

import argparse
import json
import sys

from paramem.cli import http_client


def _parse_error_body(body: str) -> dict:
    """Attempt to parse a JSON error response body.

    Returns the parsed dict, or an empty dict on parse failure.

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
    """Execute the ``backup-restore`` subcommand.

    POSTs ``{"backup_id": <backup_id>}`` to ``/backup/restore`` and
    renders the outcome.  Provides operator-actionable messages for 409
    (active TRIAL/STAGING or consolidation) and 400 (wrong artifact kind).

    Parameters
    ----------
    args:
        Parsed namespace from the ``backup-restore`` subparser.  Expected
        attributes: ``server_url`` (str), ``backup_id`` (str),
        ``json`` (bool).

    Returns
    -------
    int
        0 on success (200).  1 on 4xx / HTTP error.  2 on unreachable.
    """
    backup_id = args.backup_id
    body = {"backup_id": backup_id}

    url = f"{args.server_url}/backup/restore"
    try:
        result = http_client.post_json(url, body)
    except http_client.ServerUnavailable:
        print(
            f"paramem backup-restore: the server at {args.server_url} does not\n"
            "implement /backup/restore yet. Check `paramem --version` and server\n"
            "version are aligned.",
            file=sys.stderr,
        )
        return 1
    except http_client.ServerUnreachable:
        print(
            f"paramem backup-restore: server unreachable at {args.server_url}.\n"
            "Is paramem-server running? `systemctl --user status paramem-server`.",
            file=sys.stderr,
        )
        return 2
    except http_client.ServerHTTPError as exc:
        if exc.status_code == 409:
            parsed = _parse_error_body(exc.body)
            detail = parsed.get("detail", {})
            error_code = detail.get("error", "") if isinstance(detail, dict) else ""
            if error_code == "trial_active":
                print(
                    "Cannot restore during TRIAL. Run 'paramem migrate-accept' or "
                    "'paramem migrate-rollback' first.",
                    file=sys.stderr,
                )
            elif error_code == "staging_active":
                print(
                    "Cannot restore during STAGING. Run 'paramem migrate-cancel' first.",
                    file=sys.stderr,
                )
            elif error_code == "consolidating":
                print(
                    "Consolidation running; wait for completion before restoring.",
                    file=sys.stderr,
                )
            else:
                print(
                    f"paramem backup-restore: server returned HTTP 409 from {exc.url}.\n"
                    f"{exc.body.strip() or '(empty response body)'}",
                    file=sys.stderr,
                )
            return 1
        if exc.status_code == 400:
            parsed = _parse_error_body(exc.body)
            detail = parsed.get("detail", {})
            if isinstance(detail, dict):
                message = detail.get("message", exc.body.strip())
            else:
                message = exc.body.strip()
            print(f"paramem backup-restore: {message}", file=sys.stderr)
            return 1
        print(
            f"paramem backup-restore: server returned HTTP {exc.status_code} from {exc.url}.\n"
            f"{exc.body.strip() or '(empty response body)'}",
            file=sys.stderr,
        )
        return 1

    if getattr(args, "json", False):
        print(json.dumps(result, indent=2))
        return 0

    # Non-JSON render.
    restored = result.get("restored", {})
    backed_up = result.get("backed_up_pre_restore", {})
    restart_required = result.get("restart_required", True)
    restart_hint = result.get("restart_hint", "systemctl --user restart paramem-server")

    print(f"Restored config from backup {backup_id}.")
    for kind_name, live_path in restored.items():
        print(f"  live path:            {live_path}")
    for kind_name, safety_path in backed_up.items():
        print(f"  safety backup:        {safety_path}")
    print(f"  restart_required:     {'true' if restart_required else 'false'}")
    print()
    print(f"Run '{restart_hint}' to load the restored config.")

    return 0
