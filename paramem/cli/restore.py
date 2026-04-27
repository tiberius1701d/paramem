"""``paramem restore --recovery-key-file <path>`` — hardware-replacement restore.

Recovers access to an age-encrypted store using the recovery secret printed
at ``paramem generate-key`` time. The intended scenario is hardware loss:
the operator has the encrypted data (from a backup archive restored onto
fresh hardware) plus the recovery bech32 on paper, but has lost the old
daily identity and its passphrase.

Distinct from ``paramem backup-restore`` — that command restores a backup
archive over REST; this command re-keys an already-on-disk age store.

Operation:

1. Load the recovery identity from the bech32 supplied via
   ``--recovery-key-file`` (first line of file) or stdin.
2. Sanity-check by trying to decrypt the first age file under ``data_dir``.
   A wrong bech32 fails immediately, before anything on disk changes.
3. Mint a fresh daily X25519 identity, wrap it with a new operator-supplied
   passphrase, and write ``~/.config/paramem/daily_key.age`` (mode ``0600``).
4. Write ``~/.config/paramem/recovery.pub`` with the recovery public
   recipient derived from the bech32. The recovery identity is reused — it
   is the thing that authorised the restore.
5. Walk every age envelope under ``data_dir`` and re-encrypt to
   ``[daily_new, recovery]``. Per-file atomic rename + rotation manifest
   (shared with ``rotate-daily`` / ``rotate-recovery``) makes the sweep
   crash-safe.

Preconditions:

- No existing ``daily_key.age`` or ``recovery.pub`` in the config dir
  (or ``--force`` to overwrite them — destroys any lingering access via
  the pre-existing daily identity).
- No plaintext files under ``data_dir`` — plaintext alongside age would
  be a broken mode.

Crash-safety:

- ``daily_key.age`` and ``recovery.pub`` are written BEFORE the file walk.
  On resume, the manifest's ``new_daily_pub`` is cross-checked against the
  on-disk ``daily_key.age``; a mismatch (e.g. the operator re-ran with a
  different passphrase / mint) fails loudly.
- Decrypt always goes through the recovery identity — works for files
  that are still under ``[old_daily_lost, recovery]`` AND for files that
  were rotated in a prior run to ``[new_daily, recovery]``.
"""

from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

import pyrage

from paramem.backup import key_store as _ks
from paramem.backup import rotation as _rot
from paramem.backup.age_envelope import (
    age_decrypt_bytes,
    identity_from_bech32,
    is_age_envelope,
)
from paramem.backup.encryption import infra_paths
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


def _resolve_simulate_dir(args: argparse.Namespace) -> Path | None:
    """Resolve the simulate-mode peer-storage root.

    Returns ``None`` when neither the explicit override nor the config
    yields a path — callers treat that as "no simulate store to consider".
    """
    if getattr(args, "simulate_dir", None):
        return Path(args.simulate_dir).expanduser().resolve()
    from paramem.server.config import load_server_config

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        return None
    cfg = load_server_config(str(config_path))
    return cfg.paths.simulate


def _resolve_recovery_bech32(args: argparse.Namespace) -> str | None:
    """Read the recovery bech32 from a file (priority) or interactive stdin.

    CLI-positional secrets are deliberately NOT supported — they would leak
    into shell history.
    """
    if args.recovery_key_file is not None:
        try:
            content = Path(args.recovery_key_file).expanduser().read_text(encoding="utf-8")
        except OSError as exc:
            print(f"ERROR: could not read recovery-key file: {exc}", file=sys.stderr)
            return None
        bech32 = content.splitlines()[0].strip() if content else ""
        if not bech32:
            print("ERROR: recovery-key file is empty", file=sys.stderr)
            return None
        return bech32

    if not sys.stdin.isatty():
        print(
            "ERROR: no recovery key supplied. Pass --recovery-key-file or run "
            "in an interactive terminal to paste the bech32 at the prompt.",
            file=sys.stderr,
        )
        return None
    bech32 = getpass.getpass("Recovery key (AGE-SECRET-KEY-1…): ").strip()
    if not bech32:
        print("ERROR: recovery key must be non-empty", file=sys.stderr)
        return None
    return bech32


def _resolve_passphrase(args: argparse.Namespace) -> str | None:
    """Resolve the NEW daily passphrase; same precedence as generate-key."""
    if args.passphrase_file is not None:
        try:
            content = Path(args.passphrase_file).expanduser().read_text(encoding="utf-8")
        except OSError as exc:
            print(f"ERROR: could not read passphrase file: {exc}", file=sys.stderr)
            return None
        pw = content.splitlines()[0] if content else ""
        if not pw:
            print("ERROR: passphrase file is empty", file=sys.stderr)
            return None
        return pw

    env_pw = _ks.daily_passphrase_env_value()
    if env_pw:
        return env_pw

    if not sys.stdin.isatty():
        print(
            f"ERROR: no passphrase supplied. Set {DAILY_PASSPHRASE_ENV_VAR}, "
            "pass --passphrase-file, or run in an interactive terminal.",
            file=sys.stderr,
        )
        return None
    pw1 = getpass.getpass("New daily-key passphrase: ")
    if not pw1:
        print("ERROR: passphrase must be non-empty", file=sys.stderr)
        return None
    pw2 = getpass.getpass("Confirm passphrase       : ")
    if pw1 != pw2:
        print("ERROR: passphrases did not match", file=sys.stderr)
        return None
    return pw1


def _check_preconditions(args: argparse.Namespace, data_dir: Path) -> list[str]:
    """Return a list of operator-actionable precondition errors."""
    errors: list[str] = []
    daily_path = _ks.DAILY_KEY_PATH_DEFAULT
    recovery_pub_path = _ks.RECOVERY_PUB_PATH_DEFAULT

    if not args.force and not _resume_in_progress(daily_path):
        for existing in (daily_path, recovery_pub_path):
            if existing.exists():
                errors.append(
                    f"{existing} already exists — refuse to overwrite without "
                    "--force. On hardware-replacement the key files should be "
                    "absent; if they were restored from a backup, move them "
                    "aside before running this command."
                )

    simulate_dir = _resolve_simulate_dir(args)
    if data_dir.exists():
        plaintext_like = [
            p
            for p in infra_paths(data_dir, simulate_dir=simulate_dir)
            if p.exists() and not is_age_envelope(p)
        ]
        if plaintext_like:
            errors.append(
                f"{len(plaintext_like)} plaintext file(s) on disk "
                f"(e.g. {plaintext_like[0]}). Plaintext alongside age is a "
                "mismatch state; reconcile the data directory before restore."
            )

    return errors


def _resume_in_progress(daily_path: Path) -> bool:
    """Return True when a prior restore attempt left its artefacts in place."""
    manifest = _rot.read_manifest(_rot.manifest_path_default())
    return manifest is not None and manifest.operation == "restore" and daily_path.exists()


def run(args: argparse.Namespace) -> int:
    data_dir = _resolve_data_dir(args)
    if data_dir is None:
        return 1
    if not data_dir.exists():
        print(f"ERROR: data directory does not exist: {data_dir}", file=sys.stderr)
        return 1

    errors = _check_preconditions(args, data_dir)
    if errors:
        for msg in errors:
            print(f"ERROR: {msg}", file=sys.stderr)
        return 1

    recovery_bech32 = _resolve_recovery_bech32(args)
    if recovery_bech32 is None:
        return 1
    try:
        recovery_identity = identity_from_bech32(recovery_bech32)
    except (ValueError, Exception) as exc:  # noqa: BLE001
        print(
            f"ERROR: supplied recovery key is not a valid age secret: {exc}",
            file=sys.stderr,
        )
        return 1
    recovery_recipient = recovery_identity.to_public()

    # Sanity-check the recovery bech32 against at least one age envelope so
    # an operator typo fails NOW, before any on-disk mutation.
    simulate_dir = _resolve_simulate_dir(args)
    age_files = [
        p
        for p in infra_paths(data_dir, simulate_dir=simulate_dir)
        if p.exists() and is_age_envelope(p)
    ]
    if age_files:
        try:
            age_decrypt_bytes(age_files[0].read_bytes(), [recovery_identity])
        except pyrage.DecryptError:
            print(
                f"ERROR: recovery key does not decrypt {age_files[0]}. The "
                "supplied bech32 does not match any recipient on the "
                "envelopes. Verify the key against the paper copy before "
                "retrying.",
                file=sys.stderr,
            )
            return 1

    if args.dry_run:
        print(
            f"[dry-run] would mint a new daily, persist recovery.pub="
            f"{recovery_recipient}, and re-encrypt {len(age_files)} age file(s)."
        )
        if args.verbose:
            for p in age_files:
                print(f"[dry-run] pending: {p}")
        return 0

    daily_path = _ks.DAILY_KEY_PATH_DEFAULT
    recovery_pub_path = _ks.RECOVERY_PUB_PATH_DEFAULT
    manifest_path = _rot.manifest_path_default()

    # Check for resume state BEFORE resolving the passphrase so an operator
    # who wants to abort can do so without typing their passphrase again.
    existing_manifest = _rot.read_manifest(manifest_path)
    if existing_manifest is not None:
        if existing_manifest.operation != "restore":
            print(
                f"ERROR: existing manifest at {manifest_path} belongs to a "
                f"{existing_manifest.operation!r} run — refuse to mix operations. "
                "Resolve or delete the manifest before retrying.",
                file=sys.stderr,
            )
            return 1
        if not daily_path.exists():
            print(
                f"ERROR: restore manifest exists but {daily_path} is missing. "
                "The prior run crashed in an unrecoverable state; delete the "
                "manifest and start fresh.",
                file=sys.stderr,
            )
            return 1
        if recovery_pub_path.exists():
            from paramem.backup.age_envelope import recipient_from_bech32

            on_disk_recipient = recipient_from_bech32(recovery_pub_path.read_text("utf-8").strip())
            if str(on_disk_recipient) != str(recovery_recipient):
                print(
                    f"ERROR: on-disk recovery.pub ({on_disk_recipient}) does "
                    f"not match the supplied recovery bech32 ({recovery_recipient}). "
                    "Refuse to mix recovery identities mid-restore.",
                    file=sys.stderr,
                )
                return 1

    passphrase = _resolve_passphrase(args)
    if passphrase is None:
        return 1

    # Write daily + recovery.pub BEFORE the walk so a crash after any file
    # has been re-encrypted can still resume via the same daily.
    if existing_manifest is None:
        new_daily = _ks.mint_daily_identity()
        _ks.write_daily_key_file(_ks.wrap_daily_identity(new_daily, passphrase), daily_path)
        _ks.write_recovery_pub_file(recovery_recipient, recovery_pub_path)
        files = list(age_files)
        manifest = _rot.RotationManifest.fresh(
            operation="restore",
            files=files,
            new_daily_pub=str(new_daily.to_public()),
        )
        _rot.write_manifest_atomic(manifest_path, manifest)
    else:
        # Resume: load the daily that the prior run persisted.
        new_daily = _ks.load_daily_identity(daily_path, passphrase=passphrase)
        if str(new_daily.to_public()) != existing_manifest.new_daily_pub:
            print(
                f"ERROR: loaded daily public ({new_daily.to_public()}) does "
                f"not match manifest new_daily_pub ({existing_manifest.new_daily_pub}). "
                "The key file may have been replaced; abort and investigate.",
                file=sys.stderr,
            )
            return 1
        manifest = existing_manifest

    new_recipients = [new_daily.to_public(), recovery_recipient]

    while manifest.files_pending:
        path_str = manifest.files_pending[0]
        path = Path(path_str)
        if not path.exists():
            if args.verbose:
                print(f"skipped (missing): {path}")
            manifest.files_pending = manifest.files_pending[1:]
            manifest.files_done.append(path_str)
            _rot.write_manifest_atomic(manifest_path, manifest)
            continue
        try:
            _rot.rotate_file_to_recipients(
                path,
                decrypt_identities=[recovery_identity],
                new_recipients=new_recipients,
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"ERROR: failed to re-encrypt {path}: {type(exc).__name__}: {exc}. "
                "Manifest preserved for resume.",
                file=sys.stderr,
            )
            return 2
        manifest.files_pending = manifest.files_pending[1:]
        manifest.files_done.append(path_str)
        _rot.write_manifest_atomic(manifest_path, manifest)
        if args.verbose:
            print(f"restored: {path}")

    _rot.delete_manifest(manifest_path)
    _ks._clear_daily_identity_cache()

    print(
        f"\nSummary: {len(manifest.files_done)} re-encrypted, "
        f"new daily recipient: {new_daily.to_public()}, "
        f"recovery recipient preserved: {recovery_recipient}."
    )
    return 0


def add_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "restore",
        help="Restore access to an age store using the recovery bech32 (hardware-replacement).",
        description=(
            "Hardware-replacement flow. Given the recovery bech32 secret "
            "(printed once by `paramem generate-key`) and a new passphrase, "
            "mints a fresh daily identity, re-creates daily_key.age + "
            "recovery.pub, and re-encrypts every age-format infrastructure "
            "file under the data directory to [daily_new, recovery]. "
            "Distinct from `paramem backup-restore` which restores a backup "
            "archive over REST — this command re-keys an on-disk age store."
        ),
    )
    p.add_argument(
        "--data-dir",
        default=None,
        metavar="PATH",
        help="Override the data directory. Defaults to the server config's paths.data.",
    )
    p.add_argument(
        "--simulate-dir",
        default=None,
        metavar="PATH",
        help=(
            "Override the simulate-mode peer-storage directory. Defaults to "
            "the server config's paths.simulate. Pass when --data-dir is "
            "overridden too; restore re-encrypts simulate keyed_pairs alongside "
            "the train-mode store."
        ),
    )
    p.add_argument(
        "--config",
        default="configs/server.yaml",
        metavar="PATH",
        help="Server config used to resolve paths.data (default: configs/server.yaml).",
    )
    p.add_argument(
        "--recovery-key-file",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Read the recovery bech32 from the first line of this file. "
            "Accepted as a file (not an inline flag) so the secret does not "
            "leak into shell history. Omit to enter it interactively."
        ),
    )
    p.add_argument(
        "--passphrase-file",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Read the new daily passphrase from the first line of this file. "
            f"Overrides ${DAILY_PASSPHRASE_ENV_VAR}."
        ),
    )
    p.add_argument(
        "--force",
        action="store_true",
        help=(
            "Overwrite existing daily_key.age / recovery.pub if they are "
            "present. Use when the config dir was partially restored from "
            "a backup that included stale key files."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the recovery key + count age files without mutating disk.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Log each re-encrypted path.",
    )
