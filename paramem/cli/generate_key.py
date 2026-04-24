"""``paramem generate-key`` — mint the two-identity key pair for age encryption.

Mints a fresh ``daily.identity`` + ``recovery.identity`` pair, wraps the daily
secret with an operator-supplied passphrase, and writes:

- ``~/.config/paramem/daily_key.age`` (mode ``0o600``) — passphrase-wrapped
  daily secret.
- ``~/.config/paramem/recovery.pub`` (mode ``0o644``) — bech32 ``age1…``
  public recipient for the recovery identity.

The recovery *secret* (``AGE-SECRET-KEY-1…``) is printed once to stderr with
a BitLocker-style warning and is never persisted on this device. The operator
must confirm (interactively or via ``--yes``) that they have saved the recovery
key before the daily/recovery files are written — this guarantees no
half-initialised state remains if the operator aborts.

Passphrase source priority (highest first):

1. ``--passphrase-file PATH`` — reads the first line.
2. ``$PARAMEM_DAILY_PASSPHRASE`` — environment variable.
3. Interactive ``getpass`` prompt (with confirmation).

Non-interactive CI flows should combine ``--yes`` with ``--passphrase-file`` or
the env var. Passphrases are deliberately not accepted on the command line —
they would leak into shell history.

Refusing to overwrite existing key files (unless ``--force`` is passed) is a
deliberate safety: overwriting invalidates all existing encrypted data without
recourse.
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from pyrage import x25519

from paramem.backup.key_store import (
    DAILY_KEY_PATH_DEFAULT,
    DAILY_PASSPHRASE_ENV_VAR,
    RECOVERY_PUB_PATH_DEFAULT,
    daily_passphrase_env_value,
    mint_daily_identity,
    wrap_daily_identity,
    write_daily_key_file,
    write_recovery_pub_file,
)

CONFIRM_PHRASE = "I have saved the recovery key"


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Attach ``paramem generate-key`` to the top-level dispatcher."""
    p = subparsers.add_parser(
        "generate-key",
        help="Mint the two-identity (daily + recovery) key pair for age encryption.",
        description=(
            "Mints a fresh daily identity and recovery identity. Writes the "
            "passphrase-wrapped daily secret and the recovery public key to "
            "~/.config/paramem. Prints the recovery secret once to stderr "
            "with a BitLocker-style warning — the operator is responsible "
            "for saving it offline."
        ),
    )
    p.add_argument(
        "--daily-key-path",
        type=Path,
        default=DAILY_KEY_PATH_DEFAULT,
        help=f"Daily-key file path (default: {DAILY_KEY_PATH_DEFAULT}).",
    )
    p.add_argument(
        "--recovery-pub-path",
        type=Path,
        default=RECOVERY_PUB_PATH_DEFAULT,
        help=f"Recovery public-key file path (default: {RECOVERY_PUB_PATH_DEFAULT}).",
    )
    p.add_argument(
        "--passphrase-file",
        type=Path,
        default=None,
        help=(
            "Read the daily passphrase from the first line of this file. "
            "Overrides $PARAMEM_DAILY_PASSPHRASE. The file is not modified."
        ),
    )
    p.add_argument(
        "--force",
        action="store_true",
        help=(
            "Overwrite existing key files. Doing so invalidates all existing "
            "encrypted data — there is no recovery."
        ),
    )
    p.add_argument(
        "--yes",
        action="store_true",
        help=(
            "Skip the interactive confirmation that the recovery key has been "
            "saved. Intended for CI / scripted provisioning — the operator "
            "remains responsible for capturing the stderr output."
        ),
    )


@dataclass(frozen=True)
class _ResolvedArgs:
    daily_key_path: Path
    recovery_pub_path: Path
    passphrase_file: Path | None
    force: bool
    yes: bool


def _resolve_args(args: argparse.Namespace) -> _ResolvedArgs:
    return _ResolvedArgs(
        daily_key_path=Path(args.daily_key_path).expanduser(),
        recovery_pub_path=Path(args.recovery_pub_path).expanduser(),
        passphrase_file=Path(args.passphrase_file).expanduser() if args.passphrase_file else None,
        force=bool(args.force),
        yes=bool(args.yes),
    )


def _resolve_passphrase(resolved: _ResolvedArgs) -> str | None:
    """Resolve the daily passphrase from file → env → interactive prompt.

    Returns ``None`` on failure (empty passphrase, file read error, confirmation
    mismatch). The caller prints an actionable message and returns non-zero.
    The interactive branch calls ``getpass.getpass`` through the module
    attribute so tests can monkeypatch it — avoid capturing the function in a
    default argument, which would freeze the reference at import time.
    """
    if resolved.passphrase_file is not None:
        try:
            content = resolved.passphrase_file.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"ERROR: could not read passphrase file: {exc}", file=sys.stderr)
            return None
        pw = content.splitlines()[0] if content else ""
        if not pw:
            print("ERROR: passphrase file is empty", file=sys.stderr)
            return None
        return pw

    env_pw = daily_passphrase_env_value()
    if env_pw:
        return env_pw

    if not sys.stdin.isatty():
        print(
            f"ERROR: no passphrase supplied. Set {DAILY_PASSPHRASE_ENV_VAR}, "
            "pass --passphrase-file, or run in an interactive terminal.",
            file=sys.stderr,
        )
        return None

    pw1 = getpass.getpass("Daily-key passphrase: ")
    if not pw1:
        print("ERROR: passphrase must be non-empty", file=sys.stderr)
        return None
    pw2 = getpass.getpass("Confirm passphrase    : ")
    if pw1 != pw2:
        print("ERROR: passphrases did not match", file=sys.stderr)
        return None
    return pw1


def _refuse_existing(resolved: _ResolvedArgs) -> int | None:
    """Return an exit code to abort on existing files, or None when safe to proceed."""
    if resolved.force:
        return None
    existing = [p for p in (resolved.daily_key_path, resolved.recovery_pub_path) if p.exists()]
    if not existing:
        return None
    print("ERROR: key files already exist:", file=sys.stderr)
    for p in existing:
        print(f"  {p}", file=sys.stderr)
    print(
        "       Pass --force to overwrite. Overwriting invalidates all "
        "existing encrypted data — there is no recovery.",
        file=sys.stderr,
    )
    return 1


def _format_bech32_groups(s: str, group: int = 5, per_line: int = 4) -> str:
    """Render *s* as space-separated groups of *group* chars, *per_line* per line."""
    groups = [s[i : i + group] for i in range(0, len(s), group)]
    lines = [" ".join(groups[i : i + per_line]) for i in range(0, len(groups), per_line)]
    return "\n".join(lines)


def _print_recovery_banner(recovery: x25519.Identity) -> None:
    secret = str(recovery)
    # The 16-char prefix ``AGE-SECRET-KEY-1`` is kept verbatim; grouping applies
    # to the payload for easier hand-copy (bech32 has a built-in BCH checksum
    # so 5-char grouping plus the full line beneath it both round-trip).
    prefix, payload = secret[:16], secret[16:]
    grouped_payload = _format_bech32_groups(payload)

    lines = [
        "",
        "# " + "-" * 70,
        "# RECOVERY KEY — WRITE THIS DOWN NOW",
        "# " + "-" * 70,
        "# This is the ONLY copy of your recovery key. It is NEVER stored on",
        "# this device. Without it, losing your daily passphrase means losing",
        "# all your encrypted data. We cannot help you.",
        "#",
        "# Store it offline: printed paper, metal seed plate, or a password",
        "# manager note held separately from your daily passphrase.",
        "#",
        f"#   {secret}",
        "#",
        "# For hand-copy:",
        f"#   {prefix}",
    ]
    for ln in grouped_payload.splitlines():
        lines.append(f"#     {ln}")
    lines.append("# " + "-" * 70)
    print("\n".join(lines), file=sys.stderr)


def _confirm_saved(resolved: _ResolvedArgs) -> bool:
    if resolved.yes:
        return True
    if not sys.stdin.isatty():
        print(
            "ERROR: refusing to proceed non-interactively without --yes. "
            "Confirmation that the recovery key has been saved is required "
            "before the daily / recovery files are written.",
            file=sys.stderr,
        )
        return False
    print(
        f'\nType "{CONFIRM_PHRASE}" exactly to continue (or Ctrl-C to abort):',
        file=sys.stderr,
    )
    try:
        answer = input().strip()
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.", file=sys.stderr)
        return False
    if answer != CONFIRM_PHRASE:
        print("ERROR: confirmation phrase did not match — aborting.", file=sys.stderr)
        return False
    return True


def run(args: argparse.Namespace) -> int:
    resolved = _resolve_args(args)

    abort = _refuse_existing(resolved)
    if abort is not None:
        return abort

    passphrase = _resolve_passphrase(resolved)
    if passphrase is None:
        return 1

    daily = mint_daily_identity()
    recovery = x25519.Identity.generate()

    print("# " + "-" * 70, file=sys.stderr)
    print("# ParaMem two-identity key generation", file=sys.stderr)
    print("# " + "-" * 70, file=sys.stderr)
    print(f"# Daily key file : {resolved.daily_key_path}", file=sys.stderr)
    print(f"# Recovery pub   : {resolved.recovery_pub_path}", file=sys.stderr)
    print("# " + "-" * 70, file=sys.stderr)

    _print_recovery_banner(recovery)

    if not _confirm_saved(resolved):
        # Nothing has been written yet — identities live only in this process.
        return 1

    wrapped = wrap_daily_identity(daily, passphrase)
    write_daily_key_file(wrapped, resolved.daily_key_path)
    write_recovery_pub_file(recovery.to_public(), resolved.recovery_pub_path)

    print("", file=sys.stderr)
    print("# " + "-" * 70, file=sys.stderr)
    print(
        f"# Daily key wrapped and written to {resolved.daily_key_path} (0600).",
        file=sys.stderr,
    )
    print(
        f"# Recovery public key written to   {resolved.recovery_pub_path} (0644).",
        file=sys.stderr,
    )
    print("# " + "-" * 70, file=sys.stderr)

    # Operator guidance for the next step — emit only when the env var is not
    # already set, to avoid nagging operators who wired this up via a drop-in.
    if not os.environ.get(DAILY_PASSPHRASE_ENV_VAR):
        print(
            f"# Next: set {DAILY_PASSPHRASE_ENV_VAR} in your server environment,",
            file=sys.stderr,
        )
        print(
            "#       then run `paramem migrate-to-age` to flip existing data.",
            file=sys.stderr,
        )
        print("# " + "-" * 70, file=sys.stderr)

    return 0
