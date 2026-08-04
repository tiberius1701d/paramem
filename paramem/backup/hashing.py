"""Content-hash helpers for the backup subsystem.

All hashing is SHA-256 with no canonicalization: whitespace changes and
key-order changes in YAML/JSON source data are visible as hash changes. This
matches the ``config_rev`` semantics used in pstatus. ``plaintext_sha256`` is
the one exception to "raw bytes": it hashes the DECRYPTED content of an
age-encrypted file (differing from the on-disk ciphertext bytes) so the
digest survives re-encryption; see its docstring for the rationale.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def content_sha256_bytes(data: bytes) -> str:
    """Return the hex SHA-256 digest of *data*.

    Parameters
    ----------
    data:
        Raw bytes to hash.  No canonicalization is applied — byte-identical
        inputs produce byte-identical digests.

    Returns
    -------
    str
        Lowercase hexadecimal SHA-256 digest (64 characters).
    """
    return hashlib.sha256(data).hexdigest()


def plaintext_sha256(path: Path) -> str:
    """Return the hex SHA-256 digest of *path*'s PLAINTEXT content.

    Unwraps the age envelope via
    :func:`~paramem.backup.encryption.read_maybe_encrypted` when *path* is
    encrypted, and hashes the original bytes; hashes the raw bytes directly
    otherwise. Hashing the plaintext (rather than the on-disk ciphertext) is
    load-bearing: age re-encrypts with a fresh content key on every write, so
    a ciphertext-based hash would change on every re-encrypt and break
    drift/live-slot detection that compares hashes across writes.

    THE single decrypt-then-hash primitive — every caller that needs a
    plaintext content hash of a possibly-encrypted artifact (adapter registry
    slot matching, backup manifest cross-checks) composes this rather than
    re-implementing the decrypt-then-sha256 sequence.

    Parameters
    ----------
    path:
        Filesystem path to an existing file (plaintext or age-encrypted).

    Returns
    -------
    str
        Lowercase hexadecimal SHA-256 digest (64 characters) of the
        plaintext bytes.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    RuntimeError
        If *path* is an age envelope but the daily identity is not loaded
        (see :func:`read_maybe_encrypted`).
    pyrage.DecryptError
        If *path* is an age envelope that cannot be decrypted.

    None of these are caught here — callers that want to degrade to an
    empty hash on decrypt failure must catch locally at their own boundary
    (e.g. a boot-path caller that treats an undecryptable registry as
    absent); this primitive never swallows errors.
    """
    from paramem.backup.encryption import read_maybe_encrypted

    return hashlib.sha256(read_maybe_encrypted(path)).hexdigest()


def content_sha256_path(path: Path) -> str:
    """Return the hex SHA-256 digest of the file at *path*.

    Streams the file in 64 KiB chunks to avoid loading large artifacts
    entirely into memory.  The digest is identical to
    ``content_sha256_bytes(path.read_bytes())``.

    Parameters
    ----------
    path:
        Filesystem path to an existing, readable file.

    Returns
    -------
    str
        Lowercase hexadecimal SHA-256 digest (64 characters).

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    OSError
        If the file cannot be read.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
