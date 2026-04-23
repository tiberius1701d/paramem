"""``paramem encrypt-infra`` — convert plaintext infra files to ciphertext.

Enumerates the paths returned by :func:`paramem.backup.encryption.infra_paths`,
reads each existing plaintext file, and re-writes it through
``write_infra_bytes`` so the on-disk body becomes the PMEM1 envelope.

Requires ``PARAMEM_MASTER_KEY`` to be set.  Files already carrying the
PMEM1 magic are left untouched (re-running the command is idempotent).

Typical use: a fresh deployment that previously ran without a key, or a
deployment that just flipped Security ON.  Run before restarting the
server so the startup mode-consistency check sees all-ciphertext state.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from paramem.backup.encryption import (
    MASTER_KEY_ENV_VAR,
    infra_paths,
    is_pmem1_envelope,
    master_key_loaded,
    write_infra_bytes,
)


def _resolve_data_dir(args: argparse.Namespace) -> Path | None:
    if args.data_dir:
        return Path(args.data_dir).expanduser().resolve()
    # Fall back to loading the server config.
    from paramem.server.config import load_server_config

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        print(f"ERROR: config file not found: {config_path}", file=sys.stderr)
        return None
    cfg = load_server_config(str(config_path))
    return cfg.paths.data


def run(args: argparse.Namespace) -> int:
    if not master_key_loaded():
        print(
            f"ERROR: {MASTER_KEY_ENV_VAR} must be set to encrypt infrastructure files.\n"
            "       Add it to your .env or export it before running this command.",
            file=sys.stderr,
        )
        return 1

    data_dir = _resolve_data_dir(args)
    if data_dir is None:
        return 1
    if not data_dir.exists():
        print(f"ERROR: data directory does not exist: {data_dir}", file=sys.stderr)
        return 1

    converted: list[Path] = []
    already: list[Path] = []
    missing: list[Path] = []

    for path in infra_paths(data_dir):
        if not path.exists() or not path.is_file():
            missing.append(path)
            continue
        if is_pmem1_envelope(path):
            already.append(path)
            continue
        plaintext = path.read_bytes()
        write_infra_bytes(path, plaintext)
        converted.append(path)
        print(f"encrypted: {path}")

    print(
        f"\nSummary: {len(converted)} converted, "
        f"{len(already)} already encrypted, "
        f"{len(missing)} not present (skipped)."
    )
    return 0


def add_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "encrypt-infra",
        help="Convert plaintext infrastructure files to PMEM1-encrypted on-disk form.",
        description=(
            "Reads each existing plaintext infrastructure file under the data "
            "directory and re-writes it through the PMEM1 envelope.  Requires "
            "PARAMEM_MASTER_KEY.  Idempotent (already-encrypted files are left "
            "untouched)."
        ),
    )
    p.add_argument(
        "--data-dir",
        default=None,
        metavar="PATH",
        help=(
            "Override the data directory.  When unset, the value is read from "
            "the server config's paths.data."
        ),
    )
    p.add_argument(
        "--config",
        default="configs/server.yaml",
        metavar="PATH",
        help="Server config used to resolve paths.data (default: configs/server.yaml).",
    )
