"""``paramem revoke-user-token`` — revoke per-user bearer tokens.

Revokes one or more bearer tokens from the
:class:`paramem.server.user_tokens.UserTokenStore` by speaker identity or
device label.  Supports listing current tokens so the operator can identify
which entries to revoke before acting.

Revocation modes
----------------
- ``--speaker <speaker_id>`` — revoke **all** non-revoked tokens for that
  speaker (calls :meth:`~paramem.server.user_tokens.UserTokenStore.revoke_speaker`).
- ``--label <label>`` — revoke all non-revoked tokens carrying that exact
  device label (calls :meth:`~paramem.server.user_tokens.UserTokenStore.revoke_label`).
- ``--list`` — print the current token table (``speaker_id``, ``label``,
  ``created``, ``revoked``) without modifying anything; exit 0.

Security mode behaviour
-----------------------
The token store follows the deployment-wide AUTO encryption mode:

- **Security OFF** (no daily key loaded): store is read and written in
  plaintext.  No passphrase required.
- **Security ON** (daily key loaded): store is age-encrypted.
  ``PARAMEM_DAILY_PASSPHRASE`` must be set in the environment (same
  requirement as all other infra CLIs — ``mint-user-token``,
  ``encrypt-infra``, ``rotate-daily``, etc.).

Take-effect note
----------------
The CLI writes the revocation to ``user_tokens.json`` on disk.  A running
server loads the token store **once at startup** and holds it in memory; the
in-process store is not reloaded from disk between requests.  A server
restart is required for the revocation to take effect on the live server.

Exit codes:
    0  success (or ``--list`` completed)
    1  error (config not found, store write failed, no match, other I/O)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Attach ``paramem revoke-user-token`` to the top-level dispatcher.

    Parameters
    ----------
    subparsers:
        The ``_SubParsersAction`` returned by ``parser.add_subparsers()``.
    """
    p = subparsers.add_parser(
        "revoke-user-token",
        help="Revoke per-user bearer tokens by speaker or label.",
        description=(
            "Revoke bearer tokens from the UserTokenStore. "
            "Use --speaker to revoke all tokens for a speaker, "
            "--label to revoke tokens by device label, "
            "or --list to print the current token table without modifying it. "
            "A running server must be restarted for the revocation to take effect — "
            "the in-memory token store is loaded once at startup and is not refreshed "
            "between requests. "
            "When PARAMEM_DAILY_PASSPHRASE is set and a daily key is loaded, "
            "the store is read and written as an age envelope (Security ON). "
            "Without a daily key the store is read and written in plaintext (Security OFF)."
        ),
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--speaker",
        metavar="SPEAKER_ID",
        help="Revoke all non-revoked tokens for this speaker (e.g. Speaker0).",
    )
    mode.add_argument(
        "--label",
        metavar="LABEL",
        help="Revoke all non-revoked tokens carrying this exact device label.",
    )
    mode.add_argument(
        "--list",
        action="store_true",
        dest="list_tokens",
        help="Print the current token table (speaker_id, label, created, revoked) and exit.",
    )
    p.add_argument(
        "--config",
        default="configs/server.yaml",
        metavar="PATH",
        help="Server config used to resolve paths.data (default: configs/server.yaml).",
    )
    p.add_argument(
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt and revoke immediately.",
    )


def _resolve_data_dir(args: argparse.Namespace) -> Path | None:
    """Resolve the data directory from the config file.

    Mirrors the pattern in :func:`paramem.cli.mint_user_token._resolve_data_dir`.

    Parameters
    ----------
    args:
        Parsed argument namespace from :func:`add_parser`.

    Returns
    -------
    Path | None
        The resolved data directory, or ``None`` if the config file was not
        found (an error message is printed to stderr before returning).
    """
    from paramem.server.config import load_server_config

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        print(f"ERROR: config file not found: {config_path}", file=sys.stderr)
        return None
    cfg = load_server_config(str(config_path))
    return cfg.paths.data


def run(args: argparse.Namespace) -> int:
    """Execute the ``revoke-user-token`` subcommand.

    Resolves the data directory from the server config, then either lists
    current tokens (``--list``), revokes all tokens for a speaker
    (``--speaker``), or revokes tokens by label (``--label``).

    For ``--speaker`` and ``--label``, a confirmation prompt is shown unless
    ``--yes`` is passed.

    After a successful revocation the operator is reminded that the running
    server must be restarted for the change to take effect.

    Parameters
    ----------
    args:
        Parsed argument namespace produced by :func:`add_parser`.

    Returns
    -------
    int
        0 on success or ``--list``; 1 on error (config not found, no match,
        I/O error, store write failure).
    """
    from paramem.server.user_tokens import UserTokenStore

    data_dir = _resolve_data_dir(args)
    if data_dir is None:
        return 1

    store_path = Path(data_dir) / "user_tokens.json"
    store = UserTokenStore(store_path)

    # --list: print table and exit without modifying anything.
    if args.list_tokens:
        entries = store.list()
        if not entries:
            print("No tokens in store.")
            return 0
        print(f"{'SPEAKER_ID':<20}  {'LABEL':<20}  {'CREATED':<26}  {'REVOKED'}")
        print("-" * 78)
        for e in entries:
            print(f"{e['speaker_id']:<20}  {e['label']:<20}  {e['created']:<26}  {e['revoked']}")
        return 0

    # --speaker: revoke all tokens for the named speaker.
    if args.speaker is not None:
        # Preview what will be revoked.
        entries = store.list()
        targets = [e for e in entries if e["speaker_id"] == args.speaker and not e["revoked"]]
        if not targets:
            print(
                f"No active tokens found for speaker '{args.speaker}'.",
                file=sys.stderr,
            )
            return 1

        print(f"Will revoke {len(targets)} token(s) for speaker '{args.speaker}':")
        for e in targets:
            print(f"  label={e['label']!r}  created={e['created']}")

        if not args.yes:
            try:
                answer = input("Confirm revocation? [y/N] ").strip().lower()
            except EOFError:
                answer = ""
            if answer not in ("y", "yes"):
                print("Aborted.")
                return 1

        count = store.revoke_speaker(args.speaker)
        print(f"Revoked {count} token(s) for speaker '{args.speaker}'.")
        print("NOTE: restart the server for this revocation to take effect on the live service.")
        return 0

    # --label: revoke all tokens carrying the named label.
    if args.label is not None:
        entries = store.list()
        targets = [e for e in entries if e["label"] == args.label and not e["revoked"]]
        if not targets:
            print(
                f"No active tokens found with label '{args.label}'.",
                file=sys.stderr,
            )
            return 1

        print(f"Will revoke {len(targets)} token(s) with label '{args.label}':")
        for e in targets:
            print(f"  speaker_id={e['speaker_id']!r}  created={e['created']}")

        if not args.yes:
            try:
                answer = input("Confirm revocation? [y/N] ").strip().lower()
            except EOFError:
                answer = ""
            if answer not in ("y", "yes"):
                print("Aborted.")
                return 1

        count = store.revoke_label(args.label)
        print(f"Revoked {count} token(s) with label '{args.label}'.")
        print("NOTE: restart the server for this revocation to take effect on the live service.")
        return 0

    # Unreachable: argparse requires one of the mutually exclusive options.
    print("ERROR: one of --speaker, --label, or --list is required.", file=sys.stderr)
    return 1
