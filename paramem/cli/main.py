"""Top-level argparse dispatcher for the ``paramem`` console script.

Each subcommand group lives in a sibling module so later slices can drop
``migrate_accept.py``, ``migrate_rollback.py``, ``migrate_cancel.py``, and
``backup.py`` (for ``list/create/restore/prune``) without re-architecting the
dispatcher.

Default server URL is ``http://127.0.0.1:8420`` — the standard local port.
Override with ``--server-url`` on any subcommand.

Exit codes:
    0  success
    1  HTTP error or 404 (server does not implement the endpoint)
    2  server unreachable (connection refused / timeout)
    3  argparse error (handled by argparse itself)
"""

from __future__ import annotations

import argparse
import sys

from paramem.cli import (
    backup,
    migrate,
    migrate_accept,
    migrate_cancel,
    migrate_rollback,
    migrate_status,
)

CLI_VERSION = "0.1.0"


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the root argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Fully configured parser with all subcommands wired.
    """
    parser = argparse.ArgumentParser(
        prog="paramem",
        description="ParaMem server management CLI.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"paramem {CLI_VERSION}",
    )
    parser.add_argument(
        "--server-url",
        default="http://127.0.0.1:8420",
        metavar="URL",
        help="Base URL of the running ParaMem server (default: http://127.0.0.1:8420).",
    )

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = False

    # --- migrate ---
    p_migrate = subparsers.add_parser(
        "migrate",
        help="Preview a candidate server.yaml migration.",
        description=("POST /migration/preview with a candidate config path. Ships in Slice 3."),
    )
    p_migrate.add_argument(
        "path",
        metavar="PATH",
        help="Absolute path to the candidate server.yaml.",
    )
    p_migrate.add_argument(
        "--json",
        action="store_true",
        help="Emit raw JSON response instead of formatted output.",
    )

    # --- migrate-status ---
    p_ms = subparsers.add_parser(
        "migrate-status",
        help="Show the current migration trial status.",
        description=("GET /migration/status. Ships in Slice 3."),
    )
    p_ms.add_argument(
        "--json",
        action="store_true",
        help="Emit raw JSON response instead of formatted output.",
    )

    # --- migrate-cancel ---
    p_mc = subparsers.add_parser(
        "migrate-cancel",
        help="Cancel the current staged migration candidate.",
        description=("POST /migration/cancel. Ships in Slice 3b.1."),
    )
    p_mc.add_argument(
        "--json",
        action="store_true",
        help="Emit raw JSON response instead of formatted output.",
    )

    # --- migrate-accept ---
    p_ma = subparsers.add_parser(
        "migrate-accept",
        help="Accept the active trial migration and promote config B to live.",
        description=(
            "POST /migration/accept. "
            "Promotes the trial config to live, archives the trial adapter, "
            "and returns the server to LIVE state. Ships in Slice 3b.3."
        ),
    )
    p_ma.add_argument(
        "--json",
        action="store_true",
        help="Emit raw JSON response instead of formatted output.",
    )

    # --- migrate-rollback ---
    p_mr = subparsers.add_parser(
        "migrate-rollback",
        help="Rollback the active trial migration and restore config A.",
        description=(
            "POST /migration/rollback. "
            "Restores config A from backup, archives the trial adapter, "
            "and returns the server to LIVE state. Ships in Slice 3b.3."
        ),
    )
    p_mr.add_argument(
        "--json",
        action="store_true",
        help="Emit raw JSON response instead of formatted output.",
    )

    # --- backup ---
    p_backup = subparsers.add_parser(
        "backup",
        help="Backup management subcommands.",
        description="Manage ParaMem artifact backups.",
    )
    backup_sub = p_backup.add_subparsers(dest="backup_command", metavar="SUBCOMMAND")
    backup_sub.required = True

    p_backup_list = backup_sub.add_parser(
        "list",
        help="List available backup slots.",
        description=("GET /backup/list. Ships in Slice 6."),
    )
    p_backup_list.add_argument(
        "--json",
        action="store_true",
        help="Emit raw JSON response instead of formatted output.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point referenced by ``[project.scripts]``.

    Parses *argv* (defaults to ``sys.argv[1:]``) and dispatches to the
    appropriate subcommand handler.

    Parameters
    ----------
    argv:
        Argument list.  ``None`` means use ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit code (0 success, 1 HTTP error, 2 unreachable, 3 arg error).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Propagate the top-level --server-url into the namespace so subcommand
    # handlers can read it directly from args.server_url.
    # (argparse puts it there already — this comment documents intent.)

    if args.command is None:
        parser.print_help(sys.stderr)
        return 1

    if args.command == "migrate":
        return migrate.run(args)

    if args.command == "migrate-status":
        return migrate_status.run(args)

    if args.command == "migrate-cancel":
        return migrate_cancel.run(args)

    if args.command == "migrate-accept":
        return migrate_accept.run(args)

    if args.command == "migrate-rollback":
        return migrate_rollback.run(args)

    if args.command == "backup":
        if args.backup_command == "list":
            return backup.run(args)

    # Unreachable after subparsers.required = True, but keeps mypy happy.
    parser.print_help(sys.stderr)
    return 1
