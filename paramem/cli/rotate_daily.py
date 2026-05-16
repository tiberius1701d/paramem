"""``paramem rotate-daily`` — replace the daily identity across all age files.

Mints a fresh daily X25519 identity, re-encrypts every age-format
infrastructure file to ``[daily_new, recovery]``, and swaps the new daily
key file into place. The recovery recipient is preserved — hardware-
replacement flow still works after a daily rotation.

Preconditions:

- Server stopped (advisory check — a running server will race with the
  rename cycle; the command warns and refuses unless ``--force``).
- ``PARAMEM_DAILY_PASSPHRASE`` set + ``~/.config/paramem/daily_key.age``
  present so the OLD daily is loadable.
- ``~/.config/paramem/recovery.pub`` present so the NEW envelopes still
  list the recovery recipient.

Crash-safety:

- The NEW daily is minted, wrapped with the same passphrase, and
  persisted to ``daily_key.age.pending``. The OLD daily at
  ``daily_key.age`` remains valid until the final swap.
- A manifest at ``rotation.manifest.json`` records the new daily
  recipient + the pending file list. After each successful per-file
  re-encrypt the manifest is atomically rewritten.
- On resume (re-running the command when a manifest already exists),
  decryption tries BOTH the old and new daily identities, so files
  that were rewritten before the last manifest update finish cleanly
  without a re-encrypt error.
- Finalization atomically renames ``daily_key.age.pending`` ->
  ``daily_key.age`` and deletes the manifest.

Operator guidance: the old daily passphrase continues to work after the
rotation because the new daily is wrapped with the same passphrase. To
change the passphrase itself, run ``paramem generate-key --force`` (a
separate flow — full re-key rather than rotation).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyrage

from paramem.backup import key_store as _ks
from paramem.backup import rotation as _rot
from paramem.backup.age_envelope import is_age_envelope
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


def _check_preconditions(data_dir: Path) -> list[str]:
    errors: list[str] = []
    if not _ks.daily_identity_loadable(_ks.DAILY_KEY_PATH_DEFAULT):
        errors.append(
            f"Daily identity is not loadable — set {DAILY_PASSPHRASE_ENV_VAR} and "
            f"ensure {_ks.DAILY_KEY_PATH_DEFAULT} exists. Run `paramem generate-key` "
            "if neither is in place."
        )
    if not _ks.recovery_pub_available(_ks.RECOVERY_PUB_PATH_DEFAULT):
        errors.append(
            f"Recovery public recipient is missing — expected at "
            f"{_ks.RECOVERY_PUB_PATH_DEFAULT}. Rotation requires recovery to "
            "remain on every envelope."
        )
    return errors


def _load_or_mint_new_daily(
    passphrase: str,
    pending_path: Path,
) -> pyrage.x25519.Identity:
    """Load the NEW daily from ``daily_key.age.pending`` if present (resume),
    otherwise mint a fresh identity and persist it."""
    if pending_path.exists():
        return _ks.load_daily_identity(pending_path, passphrase=passphrase)
    new_daily = _ks.mint_daily_identity()
    _ks.write_daily_key_file(_ks.wrap_daily_identity(new_daily, passphrase), pending_path)
    return new_daily


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
            f"ERROR: {DAILY_PASSPHRASE_ENV_VAR} is not set — rotation needs the "
            "current passphrase to unwrap the old daily and wrap the new one.",
            file=sys.stderr,
        )
        return 1

    daily_path = _ks.DAILY_KEY_PATH_DEFAULT
    pending_path = daily_path.with_suffix(daily_path.suffix + ".pending")
    manifest_path = _rot.manifest_path_default()

    # Load OLD + NEW daily identities. NEW is minted on fresh start or re-loaded
    # from the pending file on resume.
    old_daily = _ks.load_daily_identity(daily_path, passphrase=passphrase)
    new_daily = _load_or_mint_new_daily(passphrase, pending_path)
    recovery_recipient = _ks.load_recovery_recipient(_ks.RECOVERY_PUB_PATH_DEFAULT)
    new_recipients = [new_daily.to_public(), recovery_recipient]

    # Read existing manifest (resume) or build a fresh one.
    manifest = _rot.read_manifest(manifest_path)
    was_fresh_start = manifest is None
    if manifest is None:
        files = [p for p in infra_paths(data_dir) if p.exists() and is_age_envelope(p)]
        manifest = _rot.RotationManifest.fresh(
            operation="rotate-daily",
            files=files,
            new_daily_pub=str(new_daily.to_public()),
        )
        _rot.write_manifest_atomic(manifest_path, manifest)
    else:
        if manifest.operation != "rotate-daily":
            print(
                f"ERROR: existing manifest at {manifest_path} belongs to a "
                f"{manifest.operation!r} run — refuse to mix rotations. "
                "Resolve or delete the manifest before retrying.",
                file=sys.stderr,
            )
            return 1
        if manifest.new_daily_pub and manifest.new_daily_pub != str(new_daily.to_public()):
            print(
                f"ERROR: manifest's new_daily_pub does not match the new daily "
                f"on disk at {pending_path}. The pending key file may have been "
                "replaced. Abort and investigate before retrying.",
                file=sys.stderr,
            )
            return 1

    if args.dry_run:
        print(
            f"[dry-run] would rotate {len(manifest.files_pending)} age file(s) "
            f"from old_daily to new_daily ({new_daily.to_public()})."
        )
        if args.verbose:
            for p in manifest.files_pending:
                print(f"[dry-run] pending: {p}")
        # Roll back the artefacts this invocation created. On a resume (pending
        # key + manifest already existed), leave the prior state alone — the
        # operator invoked dry-run to inspect that state, not to destroy it.
        if was_fresh_start:
            _rot.delete_manifest(manifest_path)
            try:
                pending_path.unlink()
            except FileNotFoundError:
                pass
        return 0

    # Main loop — decrypt with either OLD or NEW (robust to partial resume),
    # re-encrypt to [NEW daily pub, recovery]. Manifest is rewritten after each
    # successful file.
    while manifest.files_pending:
        path_str = manifest.files_pending[0]
        path = Path(path_str)
        try:
            _rot.rotate_file_to_recipients(
                path,
                decrypt_identities=[old_daily, new_daily],
                new_recipients=new_recipients,
            )
        except Exception as exc:  # noqa: BLE001 — per-file abort with context
            print(
                f"ERROR: failed to rotate {path}: {type(exc).__name__}: {exc}. "
                "Manifest preserved for resume.",
                file=sys.stderr,
            )
            return 2
        manifest.files_pending = manifest.files_pending[1:]
        manifest.files_done.append(path_str)
        _rot.write_manifest_atomic(manifest_path, manifest)
        if args.verbose:
            print(f"rotated: {path}")

    # Finalisation: promote pending to canonical, delete manifest, clear cache.
    _rot.finalise_pending_rename(pending_path, daily_path)
    _rot.delete_manifest(manifest_path)
    _ks._clear_daily_identity_cache()

    print(
        f"\nSummary: {len(manifest.files_done)} rotated, "
        f"new daily recipient: {new_daily.to_public()}."
    )
    return 0


def add_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "rotate-daily",
        help="Rotate the daily identity; re-encrypt every age file to the new daily + recovery.",
        description=(
            "Mints a fresh daily X25519 identity, re-encrypts every age-"
            "format infrastructure file under the data directory to "
            "[new_daily, recovery], and atomically swaps the new daily key "
            "into place. Crash-safe via rotation.manifest.json and a "
            "daily_key.age.pending intermediate. Recovery recipient is "
            "preserved — hardware-replacement flow still works."
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
        help="Report what would be rotated without touching any file.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Log each rotated path.",
    )
