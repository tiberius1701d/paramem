"""CLI entry point for the ParaMem scheduled backup runner.

Invoked by the ``paramem-backup.service`` systemd unit as:

    python -m paramem.backup --tier daily

Also callable directly for manual backups (``--tier`` is the only knob).

Sequence:
1. Load ``ServerConfig`` from ``--config`` (default ``configs/server.yaml``).
2. Call ``run_scheduled_backup(tier=<tier>)``.
3. Call ``update_backup_state(state_dir, result)`` to persist the outcome —
   UNLESS ``schedule="off"`` (no-op run), in which case the state file is
   left unchanged so ``/status`` continues to reflect the previous run.
4. Exit 0 on success, 1 on failure.

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

    # No loop in the CLI context — graph backups are skipped when loop=None.
    from paramem.backup.runner import run_scheduled_backup
    from paramem.backup.state import update_backup_state

    try:
        result = run_scheduled_backup(
            server_config=server_config,
            loop=None,
            state_dir=state_dir,
            backups_root=backups_root,
            live_config_path=config_path.resolve(),
            tier=args.tier,
            label=args.label,
        )
    except Exception as exc:
        logger.error("Backup runner raised unexpectedly: %s", exc, exc_info=True)
        return 1

    # Persist state unless this was a no-op (schedule=off).
    schedule_str = (server_config.security.backups.schedule or "").strip().lower()
    is_noop = schedule_str in ("", "off", "disabled", "none")
    if not is_noop:
        try:
            update_backup_state(state_dir, result)
            logger.info("Backup state written to %s/%s", state_dir, "backup.json")
        except Exception as exc:
            logger.error("Failed to write backup state: %s", exc)
            # Continue — the backup is on disk even if state write failed.

    if result.success:
        written = list(result.written_slots.keys())
        skipped = [a for a, _ in result.skipped_artifacts]
        logger.info(
            "Backup completed: written=%s skipped=%s tier=%s",
            written,
            skipped,
            result.tier,
        )
        return 0
    else:
        logger.error("Backup failed: %s", result.error)
        return 1


if __name__ == "__main__":
    sys.exit(main())
