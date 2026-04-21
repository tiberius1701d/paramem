"""Fernet encryption wrapper for the backup subsystem.

Provides a thin layer over ``cryptography.fernet.Fernet`` with:

- Lazy cipher construction (built on first use, cached in-module state).
- Per-artifact policy resolution (``auto`` / ``always`` / ``never``).
- Startup feasibility assertion (``always`` with no key → fatal error).
- ``_clear_cipher_cache()`` — a **supported operational call** for key
  rotation (Slice 7).  It is *not* test-only; operator code may call it
  after rotating ``PARAMEM_SNAPSHOT_KEY`` in the environment.

Key source
----------
The Fernet key is read from ``os.environ["PARAMEM_SNAPSHOT_KEY"]`` on first
use.  If the variable is absent, ``encrypt_bytes`` / ``decrypt_bytes`` raise
``RuntimeError``.  Import is always safe — the key is never read at import
time.

Per-artifact encryption policy
--------------------------------
``SecurityBackupsConfig`` is defined locally here; Slice 2 may promote it to
``paramem.server.config.SecurityConfig``.  Callers pass a config object with:

- ``encrypt_at_rest``                  — global fallback policy (``EncryptAtRest``).
- ``per_kind``                         — optional dict mapping ``ArtifactKind``
                                         values to per-kind ``EncryptAtRest``
                                         policies; falls back to global when absent.

NIT 3 (Slice 1 v3 plan): ``encrypt_bytes`` / ``decrypt_bytes`` accept single-file
inputs only.  BG-trainer resume directory (Slice 7) iterates per-file; this
module is unaware of directory structure.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from cryptography.fernet import Fernet

from paramem.backup.types import ArtifactKind, EncryptAtRest, FatalConfigError

# ---------------------------------------------------------------------------
# SecurityBackupsConfig — defined locally; Slice 2 may promote to server.config
# ---------------------------------------------------------------------------


@dataclass
class SecurityBackupsConfig:
    """Encryption policy configuration for the backup subsystem.

    Attributes
    ----------
    encrypt_at_rest : EncryptAtRest
        Global fallback policy applied to all artifact kinds unless overridden
        by ``per_kind``.
    per_kind : dict[ArtifactKind, EncryptAtRest]
        Optional per-kind overrides.  Keys are ``ArtifactKind`` enum members;
        values are ``EncryptAtRest`` policies.
    """

    encrypt_at_rest: EncryptAtRest = EncryptAtRest.AUTO
    per_kind: dict[ArtifactKind, EncryptAtRest] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Module-level cipher cache
# ---------------------------------------------------------------------------

_cipher: Fernet | None = None


def _get_cipher() -> Fernet:
    """Return the module-level cached Fernet cipher, building it on first call.

    Reads ``PARAMEM_SNAPSHOT_KEY`` from the environment on first call; raises
    ``RuntimeError`` if the variable is absent.

    Returns
    -------
    Fernet
        The cached cipher instance.

    Raises
    ------
    RuntimeError
        If ``PARAMEM_SNAPSHOT_KEY`` is not set in the environment.
    FatalConfigError
        If ``PARAMEM_SNAPSHOT_KEY`` is set but is not a valid Fernet key
        (wrong length, invalid base64, etc.).  The raw key value is never
        included in the exception message.
    """
    global _cipher
    if _cipher is None:
        key = os.environ.get("PARAMEM_SNAPSHOT_KEY")
        if not key:
            raise RuntimeError("PARAMEM_SNAPSHOT_KEY is not set — encryption is unavailable")
        try:
            _cipher = Fernet(key.encode())
        except ValueError as exc:
            raise FatalConfigError(
                "PARAMEM_SNAPSHOT_KEY is not a valid Fernet key — "
                "ensure the value is a URL-safe base64-encoded 32-byte key"
            ) from exc
    return _cipher


def _clear_cipher_cache() -> None:
    """Invalidate the module-level cipher cache.

    **Supported operational call** — Slice 7's key-rotation handler calls this
    after updating ``PARAMEM_SNAPSHOT_KEY`` in the environment so the next
    ``encrypt_bytes`` / ``decrypt_bytes`` call builds a fresh cipher with the
    new key.

    This function never raises; it is safe to call even when the cache is
    already empty.
    """
    global _cipher
    _cipher = None


# ---------------------------------------------------------------------------
# Policy helpers
# ---------------------------------------------------------------------------


def resolve_policy(kind: ArtifactKind, config: SecurityBackupsConfig) -> EncryptAtRest:
    """Return the effective ``EncryptAtRest`` policy for *kind*.

    Checks ``config.per_kind`` first; falls back to ``config.encrypt_at_rest``.

    Parameters
    ----------
    kind:
        The artifact kind to resolve the policy for.
    config:
        Backup security configuration.

    Returns
    -------
    EncryptAtRest
        The effective policy for *kind*.
    """
    return config.per_kind.get(kind, config.encrypt_at_rest)


def should_encrypt(policy: EncryptAtRest, key_loaded: bool) -> bool:
    """Return ``True`` when an artifact should be encrypted under *policy*.

    Parameters
    ----------
    policy:
        The resolved ``EncryptAtRest`` policy for this artifact.
    key_loaded:
        Whether ``PARAMEM_SNAPSHOT_KEY`` is available in the environment.

    Returns
    -------
    bool
        ``True`` → write ciphertext.  ``False`` → write plaintext.

    Note
    ----
    For ``ALWAYS`` this returns ``True`` regardless of ``key_loaded``.
    ``assert_encryption_feasible`` is the appropriate guard to call at startup
    to catch the misconfiguration before any artifact is written.
    """
    if policy is EncryptAtRest.ALWAYS:
        return True
    if policy is EncryptAtRest.NEVER:
        return False
    # AUTO: follow key presence
    return key_loaded


def assert_encryption_feasible(config: SecurityBackupsConfig, key_loaded: bool) -> None:
    """Assert that all ``always`` policies have a key available.

    Called at server startup.  Any ``always`` policy (global or per-kind)
    without a key is a fatal configuration error (spec §Security invariants).

    Parameters
    ----------
    config:
        Backup security configuration.
    key_loaded:
        Whether ``PARAMEM_SNAPSHOT_KEY`` is available in the environment.

    Raises
    ------
    FatalConfigError
        If ``encrypt_at_rest: always`` is configured but no key is loaded.
    """
    if key_loaded:
        return  # all policies are satisfiable when a key is present

    # Check global policy
    if config.encrypt_at_rest is EncryptAtRest.ALWAYS:
        raise FatalConfigError(
            "encrypt_at_rest=always but PARAMEM_SNAPSHOT_KEY is not set — "
            "server startup refused (spec §Security invariants)"
        )

    # Check per-kind overrides
    for kind, policy in config.per_kind.items():
        if policy is EncryptAtRest.ALWAYS:
            raise FatalConfigError(
                f"encrypt_at_rest=always for {kind.value!r} but "
                "PARAMEM_SNAPSHOT_KEY is not set — "
                "server startup refused (spec §Security invariants)"
            )


# ---------------------------------------------------------------------------
# Encrypt / decrypt (single-file inputs — NIT 3)
# ---------------------------------------------------------------------------


def encrypt_bytes(plaintext: bytes) -> bytes:
    """Encrypt *plaintext* using the cached Fernet cipher.

    Operates on single-file inputs only; directory-level encryption (BG
    trainer resume, Slice 7) iterates per-file and calls this function once
    per file.

    Parameters
    ----------
    plaintext:
        Raw artifact bytes to encrypt.

    Returns
    -------
    bytes
        Fernet ciphertext (includes token, HMAC, IV — can be passed directly
        to ``decrypt_bytes``).

    Raises
    ------
    RuntimeError
        If ``PARAMEM_SNAPSHOT_KEY`` is not set.
    """
    return _get_cipher().encrypt(plaintext)


def decrypt_bytes(ciphertext: bytes) -> bytes:
    """Decrypt *ciphertext* using the cached Fernet cipher.

    Parameters
    ----------
    ciphertext:
        Fernet token produced by ``encrypt_bytes``.

    Returns
    -------
    bytes
        Original plaintext bytes.

    Raises
    ------
    RuntimeError
        If ``PARAMEM_SNAPSHOT_KEY`` is not set.
    cryptography.fernet.InvalidToken
        If the ciphertext is corrupt, tampered with, or was produced with a
        different key.
    """
    return _get_cipher().decrypt(ciphertext)
