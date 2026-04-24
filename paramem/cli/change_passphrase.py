"""``paramem change-passphrase`` — rewrap the daily identity with a new passphrase.

The daily X25519 identity itself is unchanged; only the passphrase that
wraps ``~/.config/paramem/daily_key.age`` is replaced. All existing age
envelopes remain decryptable by the same identity — this is a single-file
rewrap, not a rotation. Use :program:`paramem rotate-daily` if you want a
fresh identity.

Secret handling mirrors :program:`generate-key`:

- **Old passphrase** priority: ``--old-passphrase-file PATH`` →
  ``PARAMEM_DAILY_PASSPHRASE`` env → interactive ``getpass``. The env-var
  path is the typical operator flow — "the current passphrase is already
  loaded for the running server; reuse it."
- **New passphrase** priority: ``--new-passphrase-file PATH`` → interactive
  ``getpass`` with confirmation (typed twice).

Neither passphrase is accepted as an inline CLI flag; they would leak into
shell history.

Crash-safety: the rewrap goes through :func:`write_daily_key_file` (the
primitive that powers :program:`generate-key`), which performs
``O_CREAT|O_EXCL`` + fsync + atomic rename. A crash between the load and the
rewrap leaves the old ``daily_key.age`` intact; a crash after the rename
leaves the new ``daily_key.age`` intact. Never a partial file.

Refusal cases (all with operator-actionable messages):

- ``daily_key.age`` missing — run :program:`paramem generate-key` first.
- Old passphrase does not unwrap the identity.
- Old == new — refuses the no-op rather than silently succeeding.
- Either passphrase empty.
"""

from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

import pyrage

from paramem.backup.key_store import (
    DAILY_KEY_PATH_DEFAULT,
    DAILY_PASSPHRASE_ENV_VAR,
    _clear_daily_identity_cache,
    daily_passphrase_env_value,
    load_daily_identity,
    wrap_daily_identity,
    write_daily_key_file,
)


def _read_first_line(path: Path, label: str) -> str | None:
    """Read the first line of *path*, return ``None`` on any error.

    *label* identifies the file in error output ("old passphrase file",
    "new passphrase file") for operator clarity.
    """
    try:
        content = Path(path).expanduser().read_text(encoding="utf-8")
    except OSError as exc:
        print(f"ERROR: could not read {label}: {exc}", file=sys.stderr)
        return None
    pw = content.splitlines()[0] if content else ""
    if not pw:
        print(f"ERROR: {label} is empty", file=sys.stderr)
        return None
    return pw


def _resolve_old_passphrase(args: argparse.Namespace) -> str | None:
    if args.old_passphrase_file is not None:
        return _read_first_line(args.old_passphrase_file, "old passphrase file")
    env_pw = daily_passphrase_env_value()
    if env_pw:
        return env_pw
    if not sys.stdin.isatty():
        print(
            f"ERROR: no old passphrase supplied. Set {DAILY_PASSPHRASE_ENV_VAR}, "
            "pass --old-passphrase-file, or run in an interactive terminal.",
            file=sys.stderr,
        )
        return None
    pw = getpass.getpass("Current daily passphrase: ")
    if not pw:
        print("ERROR: old passphrase must be non-empty", file=sys.stderr)
        return None
    return pw


def _resolve_new_passphrase(args: argparse.Namespace) -> str | None:
    if args.new_passphrase_file is not None:
        return _read_first_line(args.new_passphrase_file, "new passphrase file")
    if not sys.stdin.isatty():
        print(
            "ERROR: no new passphrase supplied. Pass --new-passphrase-file or "
            "run in an interactive terminal to be prompted.",
            file=sys.stderr,
        )
        return None
    pw1 = getpass.getpass("New daily passphrase: ")
    if not pw1:
        print("ERROR: new passphrase must be non-empty", file=sys.stderr)
        return None
    pw2 = getpass.getpass("Confirm new passphrase: ")
    if pw1 != pw2:
        print("ERROR: new passphrases did not match", file=sys.stderr)
        return None
    return pw1


def run(args: argparse.Namespace) -> int:
    daily_path = Path(args.daily_key_path).expanduser()
    if not daily_path.exists():
        print(
            f"ERROR: daily key file not found at {daily_path}. "
            "Run `paramem generate-key` first to mint the daily + recovery "
            "identity pair.",
            file=sys.stderr,
        )
        return 1

    old = _resolve_old_passphrase(args)
    if old is None:
        return 1

    # Unwrap with OLD. Wrong passphrase → pyrage.DecryptError → actionable
    # refuse. The file is unmodified at this point.
    try:
        identity = load_daily_identity(daily_path, passphrase=old)
    except pyrage.DecryptError:
        print(
            f"ERROR: old passphrase does not unwrap {daily_path}. "
            f"Verify ${DAILY_PASSPHRASE_ENV_VAR} or --old-passphrase-file "
            "matches the passphrase used by the current deployment.",
            file=sys.stderr,
        )
        return 1

    new = _resolve_new_passphrase(args)
    if new is None:
        return 1

    if old == new:
        print(
            "ERROR: old and new passphrases are identical — refusing no-op "
            "rewrap. Pick a different new passphrase.",
            file=sys.stderr,
        )
        return 1

    # Rewrap: atomic write via the Slice C primitive.
    write_daily_key_file(wrap_daily_identity(identity, new), daily_path)
    _clear_daily_identity_cache()

    print(
        f"Passphrase changed for {daily_path}.\n"
        f"Update {DAILY_PASSPHRASE_ENV_VAR} in your .env / systemd drop-in to "
        "the NEW value and restart the server. The previous passphrase no "
        "longer unlocks this file."
    )
    return 0


def add_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "change-passphrase",
        help="Rewrap the daily identity with a new passphrase (identity unchanged).",
        description=(
            "Replace the passphrase that wraps ~/.config/paramem/daily_key.age. "
            "The X25519 identity itself is unchanged — existing age envelopes "
            "stay decryptable by the same identity, so no re-encrypt of the "
            "data store is required. Use `paramem rotate-daily` if you want a "
            "fresh identity instead."
        ),
    )
    p.add_argument(
        "--daily-key-path",
        type=Path,
        default=DAILY_KEY_PATH_DEFAULT,
        help=f"Daily-key file path (default: {DAILY_KEY_PATH_DEFAULT}).",
    )
    p.add_argument(
        "--old-passphrase-file",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Read the old (current) passphrase from the first line of this "
            f"file. Overrides ${DAILY_PASSPHRASE_ENV_VAR}."
        ),
    )
    p.add_argument(
        "--new-passphrase-file",
        type=Path,
        default=None,
        metavar="PATH",
        help="Read the new passphrase from the first line of this file.",
    )
