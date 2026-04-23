"""age-format envelope helpers for ParaMem infrastructure data (``pyrage`` backend).

Provides a narrow wrapper over ``pyrage`` that:

- Encrypts and decrypts bytes or file streams using native X25519 identities.
- Accepts multi-recipient envelopes so a single ciphertext can be decrypted by
  any one of the configured identities (``[daily, recovery]`` in the live
  deployment).
- Refuses plugin-typed recipients / identities at every public entry point,
  closing the CVE-2024-56327 class of attacks at the ParaMem boundary rather
  than trusting pyrage's defaults.
- Uses ``encrypt_io`` / ``decrypt_io`` streaming for file paths so memory
  stays bounded on the multi-hundred-MB HF trainer shards and queue file.

The age binary envelope starts with the literal header
``age-encryption.org/v1\\n`` (:data:`AGE_MAGIC`). The old PMEM1 Fernet envelope
is handled by :mod:`paramem.backup.encryption`; the two formats are mutually
exclusive on-disk and can be distinguished by their magic bytes.

This module is the Phase-1 primitives layer for the two-identity key model in
``docs/plan_security_hardening.md``. Higher-level concerns (daily-key file,
passphrase wrapping, key-lifecycle CLI, rotation, migration from PMEM1) land
in later slices and build on top of these primitives.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from pathlib import Path

import pyrage
from pyrage import x25519

logger = logging.getLogger(__name__)

# age v1 binary envelope magic prefix. The header is a text line.
AGE_MAGIC: bytes = b"age-encryption.org/v1\n"


class PluginRecipientRejected(ValueError):
    """Raised when a caller supplies a plugin-typed recipient or identity.

    ParaMem accepts only native X25519 recipients/identities. Plugin recipients
    (``age1<plugin>1…``) are refused at the API boundary to eliminate the
    CVE-2024-56327 class of attacks (arbitrary binary execution via hostile
    plugin recipient names).
    """


def _require_x25519_recipients(recipients: Iterable) -> list[x25519.Recipient]:
    """Return *recipients* as a list, raising if any is non-X25519 or empty."""
    materialised = list(recipients)
    if not materialised:
        raise ValueError("at least one recipient is required")
    for r in materialised:
        if not isinstance(r, x25519.Recipient):
            raise PluginRecipientRejected(
                f"only x25519 recipients are accepted; got {type(r).__name__}"
            )
    return materialised


def _require_x25519_identities(identities: Iterable) -> list[x25519.Identity]:
    """Return *identities* as a list, raising if any is non-X25519 or empty."""
    materialised = list(identities)
    if not materialised:
        raise ValueError("at least one identity is required")
    for i in materialised:
        if not isinstance(i, x25519.Identity):
            raise PluginRecipientRejected(
                f"only x25519 identities are accepted; got {type(i).__name__}"
            )
    return materialised


def identity_from_bech32(s: str) -> x25519.Identity:
    """Parse an ``AGE-SECRET-KEY-1…`` bech32 string to an :class:`x25519.Identity`.

    Wraps ``pyrage.x25519.Identity.from_str`` with a prefix check and
    predictable error type. Plugin-format strings fail the underlying bech32
    decode and raise :class:`pyrage.RecipientError` — we surface them as
    :class:`PluginRecipientRejected` for a single catch site upstream.
    """
    if not isinstance(s, str):
        raise TypeError(f"identity string expected, got {type(s).__name__}")
    if not s.upper().startswith("AGE-SECRET-KEY-1"):
        raise ValueError("not an age secret key (expected AGE-SECRET-KEY-1…)")
    try:
        return x25519.Identity.from_str(s)
    except pyrage.IdentityError as exc:
        raise PluginRecipientRejected(f"not a native X25519 identity: {exc}") from exc


def recipient_from_bech32(s: str) -> x25519.Recipient:
    """Parse an ``age1…`` bech32 string to an :class:`x25519.Recipient`.

    Plugin-format strings are refused — this is the ParaMem-level enforcement
    that keeps CVE-2024-56327's attack surface off the decrypt path.
    """
    if not isinstance(s, str):
        raise TypeError(f"recipient string expected, got {type(s).__name__}")
    if not s.startswith("age1"):
        raise ValueError("not an age recipient (expected age1…)")
    try:
        return x25519.Recipient.from_str(s)
    except pyrage.RecipientError as exc:
        raise PluginRecipientRejected(f"not a native X25519 recipient: {exc}") from exc


def is_age_envelope(path: Path) -> bool:
    """Return True iff *path* opens cleanly and starts with :data:`AGE_MAGIC`."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(len(AGE_MAGIC))
    except OSError:
        return False
    return head == AGE_MAGIC


def age_encrypt_bytes(plaintext: bytes, recipients: Iterable) -> bytes:
    """Encrypt *plaintext* into an age multi-recipient envelope.

    Any supplied identity can decrypt the result. Only X25519 recipients are
    accepted — plugin recipients raise :class:`PluginRecipientRejected`.
    """
    recips = _require_x25519_recipients(recipients)
    return pyrage.encrypt(plaintext, recips)


def age_decrypt_bytes(ciphertext: bytes, identities: Iterable) -> bytes:
    """Decrypt *ciphertext* using any of the supplied identities.

    Raises :class:`pyrage.DecryptError` on tampered ciphertext or when none
    of the supplied identities match an envelope recipient.
    """
    idents = _require_x25519_identities(identities)
    return pyrage.decrypt(ciphertext, idents)


def age_encrypt_file(src: Path, dst: Path, recipients: Iterable) -> None:
    """Encrypt *src* → *dst* via streaming ``encrypt_io``.

    Writes to ``<dst>.tmp`` then atomic-renames to *dst* so a crash mid-write
    never leaves a partial envelope at the destination path. The parent
    directory is fsync'd after rename for power-loss durability.
    """
    recips = _require_x25519_recipients(recipients)
    src_path = Path(src)
    dst_path = Path(dst)
    tmp = dst_path.with_suffix(dst_path.suffix + ".tmp")

    try:
        with open(src_path, "rb") as reader, open(tmp, "wb") as writer:
            pyrage.encrypt_io(reader, writer, recips)
            writer.flush()
            os.fsync(writer.fileno())
        os.rename(tmp, dst_path)
    except BaseException:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise

    _fsync_parent(dst_path)


def age_decrypt_file(src: Path, dst: Path, identities: Iterable) -> None:
    """Decrypt *src* → *dst* via streaming ``decrypt_io``.

    Same atomic-rename semantics as :func:`age_encrypt_file`: a partial
    plaintext write never appears at the destination path.
    """
    idents = _require_x25519_identities(identities)
    src_path = Path(src)
    dst_path = Path(dst)
    tmp = dst_path.with_suffix(dst_path.suffix + ".tmp")

    try:
        with open(src_path, "rb") as reader, open(tmp, "wb") as writer:
            pyrage.decrypt_io(reader, writer, idents)
            writer.flush()
            os.fsync(writer.fileno())
        os.rename(tmp, dst_path)
    except BaseException:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise

    _fsync_parent(dst_path)


def _fsync_parent(path: Path) -> None:
    """Best-effort fsync of *path*'s parent dir; swallows OSError.

    Mirrors the pattern in :func:`paramem.backup.encryption._atomic_write_bytes`
    so the power-loss durability story is identical across both envelope
    formats.
    """
    try:
        dir_fd = os.open(str(path.parent), os.O_RDONLY)
    except OSError as exc:
        logger.warning("age_envelope: could not open parent for fsync: %s", exc)
        return
    try:
        os.fsync(dir_fd)
    except OSError as exc:
        logger.warning("age_envelope: parent dir fsync failed: %s", exc)
    finally:
        os.close(dir_fd)
