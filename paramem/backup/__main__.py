"""CLI entry point for the ParaMem scheduled backup runner.

Invoked by the ``paramem-backup.service`` systemd unit as:

    python -m paramem.backup --tier daily

Bundle backups require the running server: the server holds
``PARAMEM_DAILY_PASSPHRASE`` (needed to decrypt per-tier registries for hash
resolution) and has the live adapters context.  This runner therefore
**delegates to the server when it is reachable**: it POSTs
``/backup/create`` with the configured artifacts and the requested tier, and
the server produces the bundle, prunes, and persists ``backup.json`` itself.

When the server is unreachable, a standalone process cannot produce a valid
bundle (no passphrase to decrypt registries, no live adapter context).
Instead of producing a degraded partial backup, the standalone path records
a **skipped/degraded state** via ``update_backup_state`` so ``/status``
reflects the gap, and exits 0 (best-effort, non-fatal — the next scheduled
run retries when the server is back up).

Sequence:
1. Load ``ServerConfig`` from ``--config`` (default ``configs/server.yaml``).
2. ``schedule="off"`` → no-op (state file left unchanged).
3. Delegate to the running server via ``POST /backup/create`` when reachable.
4. Otherwise record a degraded/skipped state and exit 0.
5. Exit 0 on success or skipped, 1 on hard failure (HTTP error, bad config).

No GPU, no torch, no model loading.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# Generous timeout: a server-side backup writes up to three artifacts (with
# encryption) and runs the retention sweep.  Graph serialization from the live
# loop is fast; disk I/O + prune dominates.
_DELEGATE_TIMEOUT_SECONDS: float = 30.0


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="ParaMem scheduled backup runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="configs/server.yaml",
        help="Path to server.yaml (default: configs/server.yaml)",
    )
    parser.add_argument(
        "--tier",
        default="daily",
        choices=["daily", "weekly", "monthly", "yearly"],
        help="Backup tier (default: daily)",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Optional operator-supplied annotation for this backup slot",
    )
    return parser.parse_args(argv)


def _run_standalone(server_config, args, state_dir, backups_root, config_path) -> int:
    """Standalone path — used only when the server is unreachable.

    Bundle backups require the running server (passphrase for registry
    decryption, live adapter context).  When the server is unreachable, this
    path does NOT attempt a partial local bundle — a degraded partial backup
    is worse than no backup (it would silently omit adapters and produce an
    incomplete recovery set that looks valid).

    Instead, records a **degraded/skipped** ``ScheduledBackupResult`` via
    ``update_backup_state`` so ``/status`` shows the gap, then exits 0
    (best-effort, non-fatal — the next scheduled run retries when the server
    is back up).
    """
    from datetime import datetime, timezone

    from paramem.backup.runner import ScheduledBackupResult
    from paramem.backup.state import update_backup_state

    now_iso = datetime.now(timezone.utc).isoformat()
    _err = "server unavailable — bundle backup requires the running server"
    artifacts_cfg = list(getattr(getattr(server_config.security, "backups", None), "artifacts", []))
    result = ScheduledBackupResult(
        started_at=now_iso,
        completed_at=now_iso,
        success=False,
        tier=args.tier,
        label=args.label,
        written_slots={},
        skipped_artifacts=[(a, _err) for a in artifacts_cfg],
        error=_err,
        prune_result_summary=None,
    )

    try:
        update_backup_state(state_dir, result)
        logger.info("Backup state written to %s/%s", state_dir, "backup.json")
    except Exception as exc:
        logger.error("Failed to write backup state: %s", exc)
        # Non-fatal — the skipped state is advisory, not blocking.

    logger.warning(
        "Server unreachable — bundle backup skipped for tier=%s. "
        "The next scheduled run will retry when the server is available.",
        args.tier,
    )
    return 0


def main(argv=None) -> int:
    """Entry point.  Returns exit code (0 = success, 1 = failure)."""
    args = _parse_args(argv)

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        return 1

    try:
        from paramem.server.config import load_server_config

        server_config = load_server_config(config_path)
    except Exception as exc:
        logger.error("Failed to load server config: %s", exc)
        return 1

    state_dir = (server_config.paths.data / "state").resolve()
    backups_root = (server_config.paths.data / "backups").resolve()

    # Schedule guard — matches run_scheduled_backup's no-op so the state file
    # keeps reflecting the previous run when scheduling is disabled.
    schedule_str = (server_config.security.backups.schedule or "").strip().lower()
    if schedule_str in ("", "off", "disabled", "none"):
        logger.info("Scheduled backups disabled (schedule=%r); nothing to do.", schedule_str)
        return 0

    # Delegate to the running server when reachable: the graph exists only in
    # its in-memory consolidation loop, so a server-mediated backup is the only
    # way to capture it.  The server prunes and persists backup.json itself.
    server_url = f"http://127.0.0.1:{server_config.server.port}"
    artifacts = list(server_config.security.backups.artifacts)
    try:
        from paramem.cli import http_client

        resp = http_client.post_json(
            f"{server_url}/backup/create",
            {"kinds": artifacts, "tier": args.tier, "label": args.label},
            timeout=_DELEGATE_TIMEOUT_SECONDS,
        )
    except http_client.ServerUnreachable as exc:
        logger.warning(
            "Server unreachable at %s (%s); falling back to standalone backup "
            "(config + registry; graph requires the live server).",
            server_url,
            exc,
        )
        return _run_standalone(server_config, args, state_dir, backups_root, config_path)
    except http_client.ServerUnavailable as exc:
        logger.warning(
            "Server at %s does not implement /backup/create (%s); falling back "
            "to standalone backup.",
            server_url,
            exc,
        )
        return _run_standalone(server_config, args, state_dir, backups_root, config_path)
    except http_client.ServerHTTPError as exc:
        logger.error(
            "Server backup request failed: HTTP %s from %s",
            exc.status_code,
            exc.url,
        )
        return 1

    # Server captured the backup (graph included when its loop is live) and
    # updated backup.json itself — do not write state here.
    success = resp.get("success", False)
    written = resp.get("written_slots") or {}
    skipped = [s.get("kind") for s in (resp.get("skipped_artifacts") or []) if isinstance(s, dict)]
    if success:
        logger.info(
            "Backup delegated to server: written=%s skipped=%s tier=%s",
            list(written.keys()),
            skipped,
            resp.get("tier", args.tier),
        )
        return 0
    logger.error("Server backup reported failure: %s", resp.get("error"))
    return 1


if __name__ == "__main__":
    sys.exit(main())
