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

    POSTs ``{"backup_id": <backup_id>, "restore_config": <bool>}`` to
    ``/backup/restore`` and renders the outcome.  ``restore_config`` is a
    ``store_true`` flag and is always sent explicitly (never omitted) so
    the restore semantics never depend on a server-side default.  Provides
    operator-actionable messages for 409 (active TRIAL/STAGING or
    consolidation) and 400 (wrong artifact kind).

    Parameters
    ----------
    args:
        Parsed namespace from the ``backup-restore`` subparser.  Expected
        attributes: ``server_url`` (str), ``backup_id`` (str),
        ``restore_config`` (bool), ``json`` (bool).

    Returns
    -------
    int
        0 on success (200).  1 on 4xx / HTTP error.  2 on unreachable.
    """
    backup_id = args.backup_id
    body: dict = {
        "backup_id": backup_id,
        "restore_config": getattr(args, "restore_config", False),
    }

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

    # Non-JSON render.  Bundle-aware: renders only the fields the server sent
    # (guarded with .get) since a plain config-kind restore and a
    # snapshot_bundle restore populate different subsets of
    # BackupRestoreResponse.
    restored = result.get("restored", {})
    restored_adapters = result.get("restored_adapters", [])
    pruned_orphans = result.get("pruned_orphans", [])
    backed_up = result.get("backed_up_pre_restore", {})
    restart_required = result.get("restart_required", True)
    restart_hint = result.get("restart_hint", "systemctl --user restart paramem-server")

    print(f"Restored backup {backup_id}.")
    for live_path in restored.values():
        print(f"  live path:            {live_path}")
    for adapter_name in restored_adapters:
        print(f"  adapter restored:     {adapter_name}")
    for orphan in pruned_orphans:
        if isinstance(orphan, dict):
            print(
                f"  orphan pruned:        {orphan.get('name', '?')} "
                f"(kind={orphan.get('kind', '?')}, "
                f"active_keys={orphan.get('active_keys', '?')})"
            )
        else:
            print(f"  orphan pruned:        {orphan}")
    for safety_path in backed_up.values():
        print(f"  safety backup:        {safety_path}")
    print(f"  restart_required:     {'true' if restart_required else 'false'}")
    print()
    print(f"Run '{restart_hint}' to load the restored config.")

    return 0
