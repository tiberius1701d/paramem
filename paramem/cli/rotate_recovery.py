"""``paramem rotate-recovery`` — replace the recovery identity across all age files.

Mints a fresh recovery X25519 identity, re-encrypts every age infrastructure
file to ``[daily, recovery_new]``, and atomically swaps the new recovery
public key into place. The daily identity is unchanged, so during the
rotation the DAILY is used as the decrypt key for every file (daily is a
recipient on both old and new envelopes).

Like ``paramem generate-key``, this command prints the recovery secret
bech32 ONCE to stderr with a BitLocker-style warning and requires operator
confirmation before any file on disk is modified. ``--yes`` skips the
confirmation for scripted flows — the operator remains responsible for
capturing the stderr output.

Crash-safety:

- The NEW recovery public recipient is persisted to ``recovery.pub.pending``
  before any file is touched.
- A manifest at ``rotation.manifest.json`` records the new recovery pub +
  the pending file list. After each successful per-file re-encrypt the
  manifest is atomically rewritten.
- Finalization atomically renames ``recovery.pub.pending`` -> ``recovery.pub``
  and deletes the manifest. The operator is responsible for destroying the
  OLD recovery secret (paper copy, etc.); the device never stored it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pyrage import x25519

from paramem.backup import key_store as _ks
from paramem.backup import rotation as _rot
from paramem.backup.age_envelope import is_age_envelope
from paramem.backup.encryption import infra_paths
from paramem.backup.key_store import DAILY_PASSPHRASE_ENV_VAR
from paramem.cli.generate_key import CONFIRM_PHRASE, _format_bech32_groups


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


def _check_preconditions(data_dir: Path) -> list[str]:
    errors: list[str] = []
    if not _ks.daily_identity_loadable(_ks.DAILY_KEY_PATH_DEFAULT):
        errors.append(
            f"Daily identity is not loadable — set {DAILY_PASSPHRASE_ENV_VAR} and "
            f"ensure {_ks.DAILY_KEY_PATH_DEFAULT} exists."
        )
    if not _ks.recovery_pub_available(_ks.RECOVERY_PUB_PATH_DEFAULT):
        errors.append(
            f"Recovery public recipient is missing — expected at "
            f"{_ks.RECOVERY_PUB_PATH_DEFAULT}. A first-time setup should use "
            "`paramem generate-key`, not rotate-recovery."
        )
    return errors


def _print_recovery_banner(new_recovery: x25519.Identity, pending_path: Path) -> None:
    secret = str(new_recovery)
    prefix, payload = secret[:16], secret[16:]
    grouped = _format_bech32_groups(payload)
    lines = [
        "",
        "# " + "-" * 70,
        "# NEW RECOVERY KEY — WRITE THIS DOWN NOW",
        "# " + "-" * 70,
        "# This is the ONLY copy of your new recovery key. It is NEVER stored",
        "# on this device. Without it, losing your daily passphrase means",
        "# losing all your encrypted data. We cannot help you.",
        "#",
        "# Destroy the PREVIOUS recovery copy only AFTER you confirm the",
        f"# pending rotation finalizes successfully ({pending_path}).",
        "#",
        f"#   {secret}",
        "#",
        "# For hand-copy:",
        f"#   {prefix}",
    ]
    for ln in grouped.splitlines():
        lines.append(f"#     {ln}")
    lines.append("# " + "-" * 70)
    print("\n".join(lines), file=sys.stderr)


def _confirm_saved(yes: bool) -> bool:
    if yes:
        return True
    if not sys.stdin.isatty():
        print(
            "ERROR: refusing to proceed non-interactively without --yes. "
            "Confirmation that the new recovery key has been saved is required "
            "before any file on disk is modified.",
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
    data_dir = _resolve_data_dir(args)
    if data_dir is None:
        return 1
    if not data_dir.exists():
        print(f"ERROR: data directory does not exist: {data_dir}", file=sys.stderr)
        return 1

    errors = _check_preconditions(data_dir)
    if errors:
        for msg in errors:
            print(f"ERROR: {msg}", file=sys.stderr)
        return 1

    passphrase = _ks.daily_passphrase_env_value()
    if not passphrase:
        print(
            f"ERROR: {DAILY_PASSPHRASE_ENV_VAR} is not set — recovery rotation "
            "needs the daily identity to decrypt every file before re-keying.",
            file=sys.stderr,
        )
        return 1

    recovery_pub_path = _ks.RECOVERY_PUB_PATH_DEFAULT
    pending_path = recovery_pub_path.with_suffix(recovery_pub_path.suffix + ".pending")
    manifest_path = _rot.manifest_path_default()

    # On resume (pending file present), we cannot recover the new recovery
    # SECRET — it was print-once. Refuse with a clear instruction.
    existing_manifest = _rot.read_manifest(manifest_path)
    if existing_manifest is not None:
        if existing_manifest.operation != "rotate-recovery":
            print(
                f"ERROR: existing manifest at {manifest_path} belongs to a "
                f"{existing_manifest.operation!r} run — refuse to mix rotations.",
                file=sys.stderr,
            )
            return 1
        print(
            "ERROR: a prior rotate-recovery run left state behind. Because the "
            "new recovery secret is print-once and not stored on disk, this "
            "rotation cannot be resumed — it must be aborted and restarted.\n"
            f"       Delete {pending_path} and {manifest_path}, confirm no data "
            "file ended up encrypted to the abandoned recovery key "
            "(each file must still decrypt with the current daily), and rerun "
            "this command.",
            file=sys.stderr,
        )
        return 1

    daily_identity = _ks.load_daily_identity(_ks.DAILY_KEY_PATH_DEFAULT, passphrase=passphrase)
    new_recovery = x25519.Identity.generate()

    # Print the new secret first so the operator sees it before ANY on-disk
    # mutation happens. If they abort, nothing is left behind.
    _print_recovery_banner(new_recovery, pending_path)
    if not _confirm_saved(args.yes):
        return 1

    # Persist the new recovery pub to the pending path BEFORE walking files.
    _ks.write_recovery_pub_file(new_recovery.to_public(), pending_path)

    files = [p for p in infra_paths(data_dir) if p.exists() and is_age_envelope(p)]
    manifest = _rot.RotationManifest.fresh(
        operation="rotate-recovery",
        files=files,
        new_recovery_pub=str(new_recovery.to_public()),
    )
    _rot.write_manifest_atomic(manifest_path, manifest)

    new_recipients = [daily_identity.to_public(), new_recovery.to_public()]

    if args.dry_run:
        print(
            f"[dry-run] would rotate {len(manifest.files_pending)} age file(s) "
            f"from old recovery to new recovery ({new_recovery.to_public()})."
        )
        if args.verbose:
            for p in manifest.files_pending:
                print(f"[dry-run] pending: {p}")
        # Roll back the pending artefacts — dry-run must not leave state.
        _rot.delete_manifest(manifest_path)
        try:
            pending_path.unlink()
        except FileNotFoundError:
            pass
        return 0

    while manifest.files_pending:
        path_str = manifest.files_pending[0]
        path = Path(path_str)
        try:
            _rot.rotate_file_to_recipients(
                path,
                decrypt_identities=[daily_identity],
                new_recipients=new_recipients,
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"ERROR: failed to rotate {path}: {type(exc).__name__}: {exc}. "
                "Pending state preserved for manual recovery — see the resume "
                "instructions printed when a manifest already exists.",
                file=sys.stderr,
            )
            return 2
        manifest.files_pending = manifest.files_pending[1:]
        manifest.files_done.append(path_str)
        _rot.write_manifest_atomic(manifest_path, manifest)
        if args.verbose:
            print(f"rotated: {path}")

    _rot.finalise_pending_rename(pending_path, recovery_pub_path)
    _rot.delete_manifest(manifest_path)

    print(
        f"\nSummary: {len(manifest.files_done)} rotated, "
        f"new recovery recipient: {new_recovery.to_public()}.\n"
        "Destroy any PREVIOUS paper copy of the recovery key now — it no "
        "longer decrypts anything on this device."
    )
    return 0


def add_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "rotate-recovery",
        help="Rotate the recovery identity; re-encrypt every age file to daily + the new recovery.",
        description=(
            "Mints a fresh recovery X25519 identity, prints the new secret "
            "once to stderr with a BitLocker-style warning, and re-encrypts "
            "every age-format infrastructure file under the data directory "
            "to [daily, new_recovery]. The daily identity is unchanged. "
            "Crash-safe via rotation.manifest.json and a recovery.pub."
            "pending intermediate."
        ),
    )
    p.add_argument(
        "--data-dir",
        default=None,
        metavar="PATH",
        help="Override the data directory. Defaults to the server config's paths.data.",
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
        help="Mint + print the new recovery bech32, then exit without touching disk.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Log each rotated path.",
    )
    p.add_argument(
        "--yes",
        action="store_true",
        help=(
            "Skip the interactive confirmation prompt. Intended for CI / "
            "scripted provisioning — the operator remains responsible for "
            "capturing the stderr output."
        ),
    )
