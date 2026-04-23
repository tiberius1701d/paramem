"""``paramem decrypt-infra --i-accept-plaintext`` — convert ciphertext
infra files back to plaintext.

Reverse of ``encrypt-infra``.  Reads each existing PMEM1-wrapped infra file,
decrypts, and re-writes as naked plaintext.  Requires ``PARAMEM_MASTER_KEY``
to be set (to unlock the ciphertext) AND the explicit
``--i-accept-plaintext`` flag to acknowledge the operator is choosing to
leave personal facts on disk in readable form.

After running, the operator should also remove ``PARAMEM_MASTER_KEY`` from
the environment so subsequent writes do not re-encrypt.  The startup
mode-consistency check will refuse to start a server with the key still
set after a decrypt pass — that refuse message is the reminder.
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
    read_maybe_encrypted,
    write_plaintext_atomic,
)


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


def run(args: argparse.Namespace) -> int:
    if not args.i_accept_plaintext:
        print(
            "ERROR: --i-accept-plaintext is required.  This command converts "
            "personal facts from encrypted to readable on-disk form.",
            file=sys.stderr,
        )
        return 1

    if not master_key_loaded():
        print(
            f"ERROR: {MASTER_KEY_ENV_VAR} must be set to decrypt.  Restore the "
            "key that was used to encrypt the files, or delete the files if "
            "they are unrecoverable.",
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
        if not is_pmem1_envelope(path):
            already.append(path)
            continue
        plaintext = read_maybe_encrypted(path)
        write_plaintext_atomic(path, plaintext)
        converted.append(path)
        print(f"decrypted: {path}")

    print(
        f"\nSummary: {len(converted)} converted, "
        f"{len(already)} already plaintext, "
        f"{len(missing)} not present (skipped)."
    )
    print(
        f"\nNOTE: remove {MASTER_KEY_ENV_VAR} from your environment before "
        "restarting the server, otherwise subsequent writes will re-encrypt."
    )
    return 0


def add_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "decrypt-infra",
        help="Convert PMEM1-encrypted infrastructure files back to plaintext.",
        description=(
            "Reverses encrypt-infra.  Requires PARAMEM_MASTER_KEY to unlock "
            "the ciphertext plus the explicit --i-accept-plaintext flag.  "
            "Idempotent (already-plaintext files are left untouched)."
        ),
    )
    p.add_argument(
        "--i-accept-plaintext",
        action="store_true",
        dest="i_accept_plaintext",
        help=(
            "Required acknowledgement that the operator is choosing to leave "
            "personal facts on disk in readable form."
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
