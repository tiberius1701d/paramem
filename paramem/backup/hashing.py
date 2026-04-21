"""Content-hash helpers for the backup subsystem.

All hashing is raw-bytes SHA-256 with no canonicalization (Resolved Decision
29 / spec §Slice 1 primitives): whitespace changes and key-order changes in
YAML/JSON source data are visible as hash changes.  This matches the
``config_rev`` semantics used in pstatus.

The ``fingerprint_key_bytes`` helper produces the 64-bit key fingerprint
stored in every ``ArtifactMeta.key_fingerprint`` field for key-rotation
detection.
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


def fingerprint_key_bytes(fernet_key_bytes: bytes) -> str:
    """Return the first 16 hex characters of ``sha256(fernet_key_bytes)``.

    This 64-bit fingerprint is stored in ``ArtifactMeta.key_fingerprint`` so
    operators can detect key rotation without decrypting any artifact.  It is
    *not* the full SHA-256 digest — truncation is intentional: a full 256-bit
    hash of the key bytes would provide partial key material to an attacker
    who can read sidecar files.

    Parameters
    ----------
    fernet_key_bytes:
        Raw bytes of the Fernet key (typically 44 bytes of base64url-encoded
        32-byte key material).

    Returns
    -------
    str
        16-character lowercase hex string.
    """
    return hashlib.sha256(fernet_key_bytes).hexdigest()[:16]
