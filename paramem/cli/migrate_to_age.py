"""``paramem migrate-to-age`` — flip the infrastructure store from PMEM1 to age.

Walks the paths returned by :func:`paramem.backup.encryption.infra_paths`,
decrypts each PMEM1 file using the Fernet master key, re-encrypts it into an
age multi-recipient envelope ``[daily, recovery]``, and atomically renames
the new envelope into place. Already-age files are skipped, plaintext files
are refused (they indicate a mismatched startup mode).

Preconditions (all three must hold):

- ``PARAMEM_MASTER_KEY`` is set so the PMEM1 source files are decryptable.
- ``PARAMEM_DAILY_PASSPHRASE`` is set and ``~/.config/paramem/daily_key.age``
  exists so the age daily identity is loadable.
- ``~/.config/paramem/recovery.pub`` exists so every output envelope can list
  both recipients. Loss of the recovery recipient at this step would
  permanently strip the recovery safety net from the data at rest — refuse
  rather than silently degrade.

Atomicity is per-file (``<path>.tmp`` → fsync → rename → fsync parent), so
a crash mid-sweep leaves every file in exactly one of two valid states:
PMEM1 (not yet migrated) or age (already migrated). Re-running is safe and
idempotent.

After the command reports success the operator can, one release later,
remove ``PARAMEM_MASTER_KEY`` from their environment. Until then the Fernet
key is retained as the rollback safety net so any lingering PMEM1 file (a
backup restore, a forgotten directory) stays decryptable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from paramem.backup import key_store as _ks
from paramem.backup.age_envelope import age_encrypt_bytes, is_age_envelope
from paramem.backup.encryption import (
    MASTER_KEY_ENV_VAR,
    _atomic_write_bytes,
    infra_paths,
    is_pmem1_envelope,
    master_key_loaded,
    read_maybe_encrypted,
)
from paramem.backup.key_store import DAILY_PASSPHRASE_ENV_VAR


def _resolve_data_dir(args: argparse.Namespace) -> Path | None:
    if args.data_dir:
        return Path(args.data_dir).expanduser().resolve()
    from paramem.server.config import load_server_config

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        print(f"ERROR: config file not found: {config_path}", file=sys.stderr)
        return None
    cfg = load_server_config(str(config_path))
    return cfg.paths.data


def _check_preconditions() -> list[str]:
    """Return a list of operator-actionable missing-precondition messages.

    Reads ``DAILY_KEY_PATH_DEFAULT`` and ``RECOVERY_PUB_PATH_DEFAULT`` via
    module-attribute lookup so tests can monkeypatch the paths without having
    them frozen into this function's defaults at import time.
    """
    errors: list[str] = []
    if not master_key_loaded():
        errors.append(
            f"{MASTER_KEY_ENV_VAR} must be set so PMEM1 source files can be "
            "decrypted during the migration."
        )
    if not _ks.daily_identity_loadable(_ks.DAILY_KEY_PATH_DEFAULT):
        errors.append(
            f"Daily identity is not loadable — set {DAILY_PASSPHRASE_ENV_VAR} "
            f"and ensure {_ks.DAILY_KEY_PATH_DEFAULT} exists. "
            "Run `paramem generate-key` if neither is in place."
        )
    if not _ks.recovery_pub_available(_ks.RECOVERY_PUB_PATH_DEFAULT):
        errors.append(
            f"Recovery public recipient is missing — expected at "
            f"{_ks.RECOVERY_PUB_PATH_DEFAULT}. Run `paramem generate-key` to "
            "mint a new daily+recovery pair, or supply --allow-daily-only "
            "to migrate without the recovery safety net (STRONGLY DISCOURAGED)."
        )
    return errors


def run(args: argparse.Namespace) -> int:
    errors = _check_preconditions()
    if errors:
        # --allow-daily-only escapes only the recovery-pub check.
        recovery_error_prefix = "Recovery public recipient is missing"
        if args.allow_daily_only:
            errors = [e for e in errors if not e.startswith(recovery_error_prefix)]
        for msg in errors:
            print(f"ERROR: {msg}", file=sys.stderr)
        if errors:
            return 1

    data_dir = _resolve_data_dir(args)
    if data_dir is None:
        return 1
    if not data_dir.exists():
        print(f"ERROR: data directory does not exist: {data_dir}", file=sys.stderr)
        return 1

    # Build the recipient list once; skip the load when the operator opted
    # out of recovery via --allow-daily-only AND no recovery.pub is present.
    daily_identity = _ks.load_daily_identity_cached(_ks.DAILY_KEY_PATH_DEFAULT)
    recipients = [daily_identity.to_public()]
    if _ks.recovery_pub_available(_ks.RECOVERY_PUB_PATH_DEFAULT):
        recipients.append(_ks.load_recovery_recipient(_ks.RECOVERY_PUB_PATH_DEFAULT))
    elif args.allow_daily_only:
        print(
            "WARN: --allow-daily-only — writing age envelopes without the "
            "recovery recipient. Losing the daily passphrase will make this "
            "data permanently unrecoverable.",
            file=sys.stderr,
        )

    converted: list[Path] = []
    already_age: list[Path] = []
    plaintext: list[Path] = []
    missing: list[Path] = []
    failed: list[tuple[Path, str]] = []

    for path in infra_paths(data_dir):
        if not path.exists() or not path.is_file():
            missing.append(path)
            continue
        if is_age_envelope(path):
            already_age.append(path)
            continue
        if not is_pmem1_envelope(path):
            plaintext.append(path)
            continue

        if args.dry_run:
            converted.append(path)
            if args.verbose:
                print(f"[dry-run] would migrate: {path}")
            continue

        try:
            source_plaintext = read_maybe_encrypted(path)
            _atomic_write_bytes(path, age_encrypt_bytes(source_plaintext, recipients))
        except Exception as exc:  # noqa: BLE001 — per-file recovery
            failed.append((path, f"{type(exc).__name__}: {exc}"))
            print(f"ERROR: migration of {path} failed: {exc}", file=sys.stderr)
            if not args.continue_on_error:
                return 2
            continue
        converted.append(path)
        if args.verbose:
            print(f"migrated: {path}")

    # Plaintext infra files are a fatal mismatch — refuse the migration so the
    # operator reconciles via `paramem encrypt-infra` first. Report but exit
    # non-zero once all encrypted files have been processed.
    for path in plaintext:
        print(
            f"ERROR: {path} is plaintext; run `paramem encrypt-infra` first "
            "to reconcile before migrating.",
            file=sys.stderr,
        )

    summary = (
        f"\nSummary: {len(converted)} migrated"
        f"{' (dry-run)' if args.dry_run else ''}, "
        f"{len(already_age)} already age, "
        f"{len(missing)} not present (skipped), "
        f"{len(plaintext)} plaintext (refused), "
        f"{len(failed)} failed."
    )
    print(summary)

    if plaintext:
        return 1
    if failed:
        return 2
    return 0


def add_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "migrate-to-age",
        help="Flip infrastructure files from PMEM1 (Fernet) to age multi-recipient envelopes.",
        description=(
            "Walks the data directory and re-encrypts each PMEM1 file as an "
            "age multi-recipient envelope listing the loaded daily identity "
            "and the recovery public recipient. Idempotent: already-age "
            "files are skipped, plaintext files are refused. Per-file "
            "atomic rename; safe to re-run after a crash."
        ),
    )
    p.add_argument(
        "--data-dir",
        default=None,
        metavar="PATH",
        help=(
            "Override the data directory. When unset, the value is read from "
            "the server config's paths.data."
        ),
    )
    p.add_argument(
        "--config",
        default="configs/server.yaml",
        metavar="PATH",
        help="Server config used to resolve paths.data (default: configs/server.yaml).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be migrated without modifying any file.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Log each file that is migrated or would be migrated.",
    )
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help=(
            "Continue migrating after a per-file failure instead of aborting. "
            "The exit code still reflects failure when any file was skipped."
        ),
    )
    p.add_argument(
        "--allow-daily-only",
        action="store_true",
        help=(
            "Run the migration without the recovery recipient on the envelope. "
            "STRONGLY DISCOURAGED — losing the daily passphrase after this "
            "makes the data permanently unrecoverable."
        ),
    )
