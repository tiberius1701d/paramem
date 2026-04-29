"""``paramem encrypt-infra`` — migrate plaintext infrastructure files to age envelopes.

Walks the same set of infrastructure paths that the startup mode-consistency
gate scans (:func:`paramem.backup.encryption.infra_paths`), identifies any
files that are not already age envelopes, and re-encrypts them in-place using
:func:`paramem.backup.encryption.envelope_encrypt_bytes` with the currently
loaded daily identity.

Typical use-case: encryption was enabled (or ``PARAMEM_DAILY_PASSPHRASE`` was
set) after some files were written plaintext.  The startup gate then refuses
with a "mixed state" or "plaintext present" error.  Run this tool once to
migrate the store, then restart the server.

Exit codes:
    0  success (including dry-run and pure no-op when nothing needs migration).
    1  daily identity not loadable, or a non-recoverable I/O error.
    2  partial success — ``--continue-on-error`` was given and at least one
       file failed to encrypt.

Progress lines go to stdout.  Errors go to stderr.  A summary line is emitted
at exit on both stdout and stderr paths.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from paramem.backup.age_envelope import is_age_envelope
from paramem.backup.encryption import _atomic_write_bytes, envelope_encrypt_bytes, infra_paths
from paramem.backup.key_store import DAILY_KEY_PATH_DEFAULT, DAILY_PASSPHRASE_ENV_VAR


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    """Attach ``paramem encrypt-infra`` to the top-level dispatcher."""
    p = subparsers.add_parser(
        "encrypt-infra",
        help="Migrate plaintext infrastructure files to age envelopes.",
        description=(
            "Walks the infrastructure paths that the startup gate scans and "
            "re-encrypts any plaintext files in-place using the currently "
            "loaded daily identity. Idempotent: already-encrypted files are "
            "skipped silently (--verbose to log them). Refuses to run when "
            "the daily identity is not loadable. Typical use-case: encryption "
            "was enabled after some files were written plaintext; run this "
            "tool once, then restart the server."
        ),
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/server.yaml"),
        metavar="PATH",
        help="Server config file (default: configs/server.yaml).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help=(
            "List what would be encrypted without writing anything. "
            "Exits 0; does not require the daily identity to be loaded "
            "(reports whether migration is needed)."
        ),
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Log every file — both skipped (already encrypted) and written.",
    )
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        dest="continue_on_error",
        help=(
            "Log errors per file and keep going (default: stop on first "
            "error). When at least one file fails, exits 2 instead of 0."
        ),
    )


def _load_config(config_path: Path):
    """Load server config; return a minimal object with paths.data and paths.simulate.

    Falls back to defaults when the config file is missing so the CLI is
    usable in very early provisioning contexts.

    Returns
    -------
    object
        Object with ``.paths.data`` and ``.paths.simulate`` attributes.
    """
    try:
        from paramem.server.config import load_server_config

        return load_server_config(config_path)
    except Exception as exc:  # noqa: BLE001
        print(
            f"WARNING: could not load config from {config_path}: {exc}. "
            "Falling back to defaults (data=data/ha, simulate=data/ha/simulate).",
            file=sys.stderr,
        )

        # Return a minimal stand-in.
        class _FallbackPaths:
            data = Path("data/ha")
            simulate = Path("data/ha/simulate")

        class _FallbackCfg:
            paths = _FallbackPaths()

        return _FallbackCfg()


def run(args: argparse.Namespace) -> int:
    """Execute the ``encrypt-infra`` subcommand.

    Parameters
    ----------
    args:
        Parsed argument namespace produced by :func:`add_parser`.

    Returns
    -------
    int
        Exit code — 0 success, 1 fatal error, 2 partial success.
    """
    from paramem.backup.key_store import daily_identity_loadable

    dry_run: bool = bool(args.dry_run)
    verbose: bool = bool(args.verbose)
    continue_on_error: bool = bool(args.continue_on_error)

    # Gate: the daily identity must be loadable (unless --dry-run).
    # In dry-run mode we only enumerate; no encryption happens, so no key
    # is needed — but we still report whether a key would be required.
    if not dry_run and not daily_identity_loadable(DAILY_KEY_PATH_DEFAULT):
        print(
            f"ERROR: the daily identity is not loadable.\n"
            f"  {DAILY_PASSPHRASE_ENV_VAR} must be set and "
            f"~/.config/paramem/daily_key.age must exist.\n"
            f"\n"
            f"  To create a key pair:  paramem generate-key\n"
            f"  Then export the env var before running encrypt-infra.\n"
            f"\n"
            f"  See SECURITY.md for a full reset procedure.",
            file=sys.stderr,
        )
        return 1

    # Resolve paths from config.
    cfg = _load_config(Path(args.config))
    data_dir = Path(cfg.paths.data)
    simulate_dir = Path(cfg.paths.simulate) if cfg.paths.simulate else None

    # Gather candidate paths (existence-filtered below).
    candidates = infra_paths(data_dir, simulate_dir=simulate_dir)

    n_encrypted = 0
    n_skipped = 0
    n_failed = 0

    for path in candidates:
        if not path.exists() or not path.is_file():
            if verbose:
                print(f"  skip (missing): {path}")
            continue

        if is_age_envelope(path):
            n_skipped += 1
            if verbose:
                print(f"  skip (already encrypted): {path}")
            continue

        # File is plaintext — would need encryption.
        if dry_run:
            print(f"  would encrypt: {path}")
            n_encrypted += 1  # count as "to encrypt" in dry-run
            continue

        # Live encryption path.
        try:
            plaintext = path.read_bytes()
            ciphertext = envelope_encrypt_bytes(plaintext)
            _atomic_write_bytes(path, ciphertext)
            n_encrypted += 1
            if verbose:
                print(f"  encrypted: {path}")
        except Exception as exc:  # noqa: BLE001
            n_failed += 1
            print(f"ERROR: failed to encrypt {path}: {exc}", file=sys.stderr)
            if not continue_on_error:
                _print_summary(n_encrypted, n_skipped, n_failed, dry_run=dry_run)
                return 1

    _print_summary(n_encrypted, n_skipped, n_failed, dry_run=dry_run)

    if n_failed > 0:
        return 2
    return 0


def _print_summary(
    n_encrypted: int,
    n_skipped: int,
    n_failed: int,
    *,
    dry_run: bool,
) -> None:
    """Emit a one-line summary to stdout.

    Parameters
    ----------
    n_encrypted:
        Files encrypted (or, in dry-run, files that would be encrypted).
    n_skipped:
        Files already age-encrypted (skipped).
    n_failed:
        Files that raised an error during encryption.
    dry_run:
        When True, adjust wording to "would encrypt".
    """
    verb = "would encrypt" if dry_run else "encrypted"
    parts = [f"{verb} {n_encrypted} file(s)", f"skipped {n_skipped} (already encrypted)"]
    if n_failed:
        parts.append(f"failed {n_failed}")
    print("Summary: " + ", ".join(parts) + ".")
