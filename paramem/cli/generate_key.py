"""``paramem generate-key`` — mint a new master-key value.

Prints a freshly generated Fernet key to stdout along with a BitLocker-style
warning and instructions for activation.  The operator is expected to store
the key in ``.env`` as ``PARAMEM_MASTER_KEY=<value>`` AND keep an offline
copy (printed paper, metal seed plate, password-manager secure note) —
losing both is equivalent to losing the encrypted data.
"""

from __future__ import annotations

import argparse
import sys

from cryptography.fernet import Fernet


def run(args: argparse.Namespace) -> int:  # noqa: ARG001 — signature contract
    key = Fernet.generate_key().decode()

    banner = """\
# -----------------------------------------------------------------------
# ParaMem master key
# -----------------------------------------------------------------------
# WARNING: This is the ONLY copy of your encryption key until you save it.
# If you lose this key and lose access to your daily-unlock hardware, your
# encrypted data is unrecoverable.  We cannot help you.
#
# To activate:
#   1. Append the line below to your .env file.
#   2. Store a second copy OFFLINE (printed paper, metal seed plate,
#      password-manager secure note).
#   3. Restart the server.  Startup log should read
#      "SECURITY: ON (PARAMEM_MASTER_KEY set)".
# -----------------------------------------------------------------------"""
    print(banner, file=sys.stderr)
    print(f"PARAMEM_MASTER_KEY={key}")
    return 0
