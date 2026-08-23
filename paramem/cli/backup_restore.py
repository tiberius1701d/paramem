"""Handler for ``paramem backup-restore``.

POSTs to ``/backup/restore`` with a ``backup_id`` and renders the outcome.
Handles 409 (STAGING/TRIAL/consolidating/training/base-swap/pending-event)
and 400 (wrong kind) with operator-actionable messages.
"""

from __future__ import annotations

import argparse
import json
import sys

from paramem.cli import http_client


def run(args: argparse.Namespace) -> int:
    """Execute the ``backup-restore`` subcommand.

    POSTs ``{"backup_id": <backup_id>, "restore_config": <bool>}`` to
    ``/backup/restore`` and renders the outcome.  ``restore_config`` is a
    ``store_true`` flag and is always sent explicitly (never omitted) so
    the restore semantics never depend on a server-side default.  Provides
    operator-actionable messages for 409 (active TRIAL/STAGING/
    consolidation/background-training/base-swap/pending-event) and 400
    (wrong artifact kind).

    On 200, the non-JSON render distinguishes three outcomes from
    ``BackupRestoreResponse``'s own fields plus what this call already knows
    client-side: ``serving=True`` reports the server is live on the restored
    artifacts, no restart needed; a config-kind restore (recognised by
    ``backed_up_pre_restore`` keyed ``"config"``) or any restore where
    ``--restore-config`` was passed advises an operator restart to converge
    (both leave their existing restart posture deliberately unchanged,
    since the base model may have changed); a ``snapshot_bundle`` restore
    left non-serving with a ``quarantine_cause`` reports the memory store is
    offline, naming the cause and the two ways out — retire the affected
    keys via ``POST /debug/erase-keys``, or restore a healthy backup.

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
            detail = http_client.parse_error_detail(exc.body)
            error_code = detail.get("error", "")
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
            elif error_code == "training_active":
                print(
                    "Background training is active; wait for completion before restoring.",
                    file=sys.stderr,
                )
            elif error_code == "base_swap_active":
                print(
                    "A base-swap migration is actively running; wait for it to complete "
                    "(or fail) before restoring.",
                    file=sys.stderr,
                )
            elif error_code == "consolidation_pending":
                print(
                    "A pending consolidation event is being resumed; wait before "
                    "restoring, or run 'paramem consolidate'/'paramem reconsolidate' first.",
                    file=sys.stderr,
                )
            else:
                print(f"paramem backup-restore: server returned {exc}", file=sys.stderr)
            return 1
        if exc.status_code == 400:
            detail = http_client.parse_error_detail(exc.body)
            message = detail.get("message", exc.body.strip()) if detail else exc.body.strip()
            print(f"paramem backup-restore: {message}", file=sys.stderr)
            return 1
        print(f"paramem backup-restore: server returned {exc}", file=sys.stderr)
        return 1

    if getattr(args, "json", False):
        print(json.dumps(result, indent=2))
        return 0

    # Non-JSON render.  Bundle-aware: renders only the fields the server sent
    # (guarded with .get) since a plain config-kind restore and a
    # snapshot_bundle restore populate different subsets of
    # BackupRestoreResponse.  Advice below is derived from BackupRestoreResponse's
    # actual fields plus what this call already knows client-side (whether
    # ``--restore-config`` was passed, and whether the response shape is a
    # config-kind restore -- ``backed_up_pre_restore`` keyed ``"config"``
    # rather than ``"bundle"``) -- never from fields the server does not send.
    restored = result.get("restored", {})
    restored_adapters = result.get("restored_adapters", [])
    pruned_orphans = result.get("pruned_orphans", [])
    backed_up = result.get("backed_up_pre_restore", {})
    serving = result.get("serving", False)
    quarantine_cause = result.get("quarantine_cause")
    is_config_kind_restore = "config" in backed_up
    restore_config_requested = getattr(args, "restore_config", False)

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
    print()

    if serving:
        print("Server is serving the restored artifacts — no restart needed.")
    elif is_config_kind_restore or restore_config_requested:
        # A config-kind restore, or a snapshot_bundle restore that also
        # restored config, deliberately leaves the store on its existing
        # restart posture — the base model may have changed, so a same-base
        # lift is never attempted. An operator restart is what converges it.
        print("Restore complete on disk. Restart the server to converge onto the restored config.")
    elif quarantine_cause:
        cause_msg = quarantine_cause.get("message", "unknown cause")
        print(
            f"The memory store is offline ({cause_msg}). Retire the affected keys "
            "via POST /debug/erase-keys, or restore a healthy backup."
        )
    else:
        print("Restore did not report a serving state — check server logs.")

    return 0
