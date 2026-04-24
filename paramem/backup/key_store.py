"""Daily-key file primitives for the WP2b two-identity key model.

The daily identity is an age X25519 secret key that sits on disk as a
passphrase-wrapped envelope::

    ~/.config/paramem/daily_key.age    # mode 0600, parent dir 0700

Wrapping is performed via :func:`pyrage.passphrase.encrypt`. The outer
envelope carries the standard age v1 magic (:data:`paramem.backup.age_envelope.AGE_MAGIC`),
so ``is_age_envelope`` recognises it alongside recipient-keyed files.

This slice ships primitives only — no production call site flips the
Fernet/PMEM1 store over to age yet. Slice D is the operator-visible flip.

Trust model
-----------
The passphrase lives in :data:`DAILY_PASSPHRASE_ENV_VAR`
(``PARAMEM_DAILY_PASSPHRASE``), mirroring the pre-existing
``PARAMEM_MASTER_KEY`` handling in :mod:`paramem.backup.encryption`. The
research-spike findings in ``memory/project_wp2b_research_findings.md``
rule out OS key stores (no libsecret on WSL2, DPAPI unavailable pre-login)
for Phase 1. This module is the neutral seam a future ``KeyProtector``
backend can slot in without touching call sites.

Atomic-write semantics
----------------------
:func:`write_daily_key_file` creates the parent directory with mode ``0o700``
when absent, writes to ``<path>.tmp`` with ``O_CREAT | O_EXCL`` at mode
``0o600``, fsyncs the file, atomically renames, and fsyncs the parent
directory. This matches :func:`paramem.backup.encryption._atomic_write_bytes`
so the crash-safety story is uniform across the two envelope formats.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from pyrage import passphrase as _pyrage_passphrase
from pyrage import x25519

from paramem.backup.age_envelope import identity_from_bech32

logger = logging.getLogger(__name__)

# Env var carrying the passphrase that unlocks the daily-key file.
DAILY_PASSPHRASE_ENV_VAR: str = "PARAMEM_DAILY_PASSPHRASE"

# Default on-disk location for the wrapped daily identity.
DAILY_KEY_PATH_DEFAULT: Path = Path("~/.config/paramem/daily_key.age").expanduser()


def daily_passphrase_env_value() -> str | None:
    """Return the daily passphrase from the environment, or ``None`` when unset."""
    value = os.environ.get(DAILY_PASSPHRASE_ENV_VAR)
    return value if value else None


def mint_daily_identity() -> x25519.Identity:
    """Mint a fresh native X25519 identity via ``pyrage.x25519.Identity.generate``.

    The returned object carries both halves of the keypair; ``.to_public()``
    yields the recipient used when encrypting to this identity.
    """
    return x25519.Identity.generate()


def wrap_daily_identity(identity: x25519.Identity, passphrase: str) -> bytes:
    """Passphrase-wrap *identity* into an age envelope.

    The bech32 secret-key string (``AGE-SECRET-KEY-1…``) is encoded as UTF-8
    and handed to :func:`pyrage.passphrase.encrypt`. The returned bytes carry
    the age v1 magic and decrypt only with the exact passphrase supplied.

    Raises
    ------
    TypeError
        If *identity* is not an :class:`x25519.Identity`.
    ValueError
        If *passphrase* is empty — guards against silent-empty-env footguns.
    """
    if not isinstance(identity, x25519.Identity):
        raise TypeError(f"x25519 Identity expected, got {type(identity).__name__}")
    if not passphrase:
        raise ValueError("passphrase must be a non-empty string")
    secret_bech32 = str(identity).encode("utf-8")
    return _pyrage_passphrase.encrypt(secret_bech32, passphrase)


def write_daily_key_file(wrapped: bytes, path: Path = DAILY_KEY_PATH_DEFAULT) -> None:
    """Atomically write *wrapped* daily-key envelope to *path* with mode 0600.

    Creates the parent directory with mode ``0o700`` when missing (``exist_ok``
    does not rewrite the mode of an existing directory). Writes to ``<path>.tmp``
    via ``O_CREAT | O_EXCL`` so a fresh inode always carries the intended
    mode, fsyncs the payload, ``os.rename``'s to *path* for atomicity, and
    fsyncs the parent directory for power-loss durability.

    Any stale ``<path>.tmp`` from a prior crash is removed before the exclusive
    create so the mode guarantee is not sabotaged by a leftover file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.unlink()
    except FileNotFoundError:
        pass

    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(wrapped)
            fh.flush()
            os.fsync(fh.fileno())
        os.rename(tmp, path)
    except BaseException:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise

    try:
        dir_fd = os.open(str(path.parent), os.O_RDONLY)
    except OSError as exc:
        logger.warning("write_daily_key_file: could not open parent for fsync: %s", exc)
        return
    try:
        os.fsync(dir_fd)
    except OSError as exc:
        logger.warning("write_daily_key_file: parent dir fsync failed: %s", exc)
    finally:
        os.close(dir_fd)


def load_daily_identity(
    path: Path = DAILY_KEY_PATH_DEFAULT,
    passphrase: str | None = None,
) -> x25519.Identity:
    """Load and unlock the daily identity wrapped at *path*.

    When *passphrase* is ``None`` the value is read from
    :data:`DAILY_PASSPHRASE_ENV_VAR`. An unset / empty env var raises
    :class:`RuntimeError` with the actionable env-var name, matching
    :func:`paramem.backup.encryption._get_cipher`'s error style.

    Raises
    ------
    FileNotFoundError
        When *path* does not exist — surfaced verbatim for operator clarity.
    RuntimeError
        When no passphrase is supplied and the env var is unset.
    pyrage.DecryptError
        On wrong passphrase or tampered envelope.
    """
    path = Path(path)
    if passphrase is None:
        passphrase = daily_passphrase_env_value()
    if not passphrase:
        raise RuntimeError(f"{DAILY_PASSPHRASE_ENV_VAR} is not set — daily key cannot be unlocked")

    wrapped = path.read_bytes()
    secret_bech32 = _pyrage_passphrase.decrypt(wrapped, passphrase).decode("utf-8")
    return identity_from_bech32(secret_bech32)


__all__ = [
    "DAILY_KEY_PATH_DEFAULT",
    "DAILY_PASSPHRASE_ENV_VAR",
    "daily_passphrase_env_value",
    "load_daily_identity",
    "mint_daily_identity",
    "wrap_daily_identity",
    "write_daily_key_file",
]
