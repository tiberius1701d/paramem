"""``paramem dump PATH`` — print the decrypted contents of an infra file.

Reads *PATH* through ``read_maybe_encrypted`` and writes the resulting
plaintext bytes to stdout verbatim.  Files that are already plaintext are
passed through unchanged; PMEM1-wrapped files are decrypted on the fly
(requires ``PARAMEM_MASTER_KEY``).

Typical use: debugging a specific artifact without running a full server
or writing a one-off script.  Use with care — the output is the decrypted
content and may include personal facts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from paramem.backup.encryption import read_maybe_encrypted


def run(args: argparse.Namespace) -> int:
    path = Path(args.path).expanduser().resolve()
    if not path.exists():
        print(f"ERROR: {path} does not exist.", file=sys.stderr)
        return 1
    if not path.is_file():
        print(f"ERROR: {path} is not a regular file.", file=sys.stderr)
        return 1
    try:
        plaintext = read_maybe_encrypted(path)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 — Fernet errors surface here
        print(f"ERROR: could not decrypt {path}: {exc}", file=sys.stderr)
        return 1
    sys.stdout.buffer.write(plaintext)
    return 0


def add_parser(subparsers) -> None:
    p = subparsers.add_parser(
        "dump",
        help="Print the decrypted contents of an infrastructure file to stdout.",
        description=(
            "Reads the file through read_maybe_encrypted.  PMEM1-wrapped files "
            "are decrypted on the fly (requires PARAMEM_MASTER_KEY); plaintext "
            "files are passed through."
        ),
    )
    p.add_argument(
        "path",
        metavar="PATH",
        help="Path to the infrastructure file to dump.",
    )
