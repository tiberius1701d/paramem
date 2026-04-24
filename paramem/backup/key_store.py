"""Daily-key file primitives for the WP2b two-identity key model.

The daily identity is an age X25519 secret key that sits on disk as a
passphrase-wrapped envelope::

    ~/.config/paramem/daily_key.age    # mode 0600, parent dir 0700

Wrapping is performed via :func:`pyrage.passphrase.encrypt`. The outer
envelope carries the standard age v1 magic (:data:`paramem.backup.age_envelope.AGE_MAGIC`),
so ``is_age_envelope`` recognises it alongside recipient-keyed files.

Trust model
-----------
The passphrase lives in :data:`DAILY_PASSPHRASE_ENV_VAR`
(``PARAMEM_DAILY_PASSPHRASE``), loaded from the operator's shell env, a
``.env`` file, or a per-secret file under ``~/.config/paramem/secrets/``.
OS key stores (libsecret, DPAPI) are out of scope for this phase — no
libsecret on WSL2 and DPAPI is unavailable pre-login. This module is the
neutral seam a future ``KeyProtector`` backend can slot in without
touching call sites.

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

from paramem.backup.age_envelope import identity_from_bech32, recipient_from_bech32

logger = logging.getLogger(__name__)

# Env var carrying the passphrase that unlocks the daily-key file.
DAILY_PASSPHRASE_ENV_VAR: str = "PARAMEM_DAILY_PASSPHRASE"

# Default on-disk location for the wrapped daily identity.
DAILY_KEY_PATH_DEFAULT: Path = Path("~/.config/paramem/daily_key.age").expanduser()

# Default on-disk location for the recovery identity's public key.
# The recovery *secret* (``AGE-SECRET-KEY-1…``) is print-once-never-on-disk;
# only the *public recipient* (``age1…``) is persisted so the server can include
# it in every multi-recipient envelope.
RECOVERY_PUB_PATH_DEFAULT: Path = Path("~/.config/paramem/recovery.pub").expanduser()


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


def _atomic_write_with_mode(path: Path, body: bytes, mode: int) -> None:
    """Atomically write *body* to *path* with file mode *mode*.

    Creates the parent directory with mode ``0o700`` when missing (``exist_ok``
    does not rewrite the mode of an existing directory). Writes to ``<path>.tmp``
    via ``O_CREAT | O_EXCL`` so a fresh inode always carries *mode* exactly,
    fsyncs the payload, ``os.rename``'s to *path* for atomicity, and fsyncs
    the parent directory for power-loss durability.

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

    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(body)
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
        logger.warning("_atomic_write_with_mode: could not open parent for fsync: %s", exc)
        return
    try:
        os.fsync(dir_fd)
    except OSError as exc:
        logger.warning("_atomic_write_with_mode: parent dir fsync failed: %s", exc)
    finally:
        os.close(dir_fd)


def write_daily_key_file(wrapped: bytes, path: Path = DAILY_KEY_PATH_DEFAULT) -> None:
    """Atomically write *wrapped* daily-key envelope to *path* with mode 0600.

    Thin wrapper over :func:`_atomic_write_with_mode` that pins the mode to the
    secret-material default.
    """
    _atomic_write_with_mode(Path(path), wrapped, 0o600)


def write_recovery_pub_file(
    recipient: x25519.Recipient,
    path: Path = RECOVERY_PUB_PATH_DEFAULT,
) -> None:
    """Atomically write the recovery identity's public recipient to *path*.

    The on-disk body is the bech32 ``age1…`` string plus a trailing newline,
    at mode ``0o644`` (the ``~/.ssh/*.pub`` convention — public keys are not
    secret). The enclosing directory is created at ``0o700`` when missing;
    existing-directory modes are preserved.

    Raises
    ------
    TypeError
        If *recipient* is not an :class:`x25519.Recipient`.
    """
    if not isinstance(recipient, x25519.Recipient):
        raise TypeError(f"x25519 Recipient expected, got {type(recipient).__name__}")
    body = (str(recipient) + "\n").encode("utf-8")
    _atomic_write_with_mode(Path(path), body, 0o644)


def load_recovery_recipient(path: Path = RECOVERY_PUB_PATH_DEFAULT) -> x25519.Recipient:
    """Load the recovery recipient from *path* and validate it is native X25519.

    The file is expected to contain a single bech32 ``age1…`` line; leading and
    trailing whitespace is stripped. Plugin-typed recipients are refused at the
    :func:`paramem.backup.age_envelope.recipient_from_bech32` boundary.

    Raises
    ------
    FileNotFoundError
        When *path* does not exist.
    paramem.backup.age_envelope.PluginRecipientRejected
        On a plugin-typed or otherwise malformed recipient.
    ValueError
        When the file is empty or has no recognisable content.
    """
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"recovery pub file at {path} is empty")
    return recipient_from_bech32(text)


_daily_identity_cache: x25519.Identity | None = None


def _clear_daily_identity_cache() -> None:
    """Invalidate the module-level daily-identity cache.

    **Supported operational call** — the daily-key rotation handler calls
    this after replacing the on-disk envelope or changing the passphrase
    env var so the next read builds a fresh identity. Never raises; safe
    to call when the cache is already empty.
    """
    global _daily_identity_cache
    _daily_identity_cache = None


def load_daily_identity_cached(
    path: Path = DAILY_KEY_PATH_DEFAULT,
    passphrase: str | None = None,
) -> x25519.Identity:
    """Return the cached daily identity, loading on first call.

    Mirrors :func:`paramem.backup.encryption._get_cipher`'s lazy-cache shape
    so the universal read path can unwrap age envelopes without paying the
    scrypt cost on every decrypt. Subsequent calls re-use the unlocked
    identity until :func:`_clear_daily_identity_cache` is invoked.

    Raises the same exceptions as :func:`load_daily_identity` on first load;
    a failed load does not poison the cache (next call retries).
    """
    global _daily_identity_cache
    if _daily_identity_cache is None:
        _daily_identity_cache = load_daily_identity(path=path, passphrase=passphrase)
    return _daily_identity_cache


def daily_identity_loadable(
    daily_key_path: Path = DAILY_KEY_PATH_DEFAULT,
) -> bool:
    """Return True when the daily identity *can* be loaded without attempting it.

    Probes the two preconditions — the wrapped-key file exists and the
    passphrase env var is set — without running the scrypt unwrap. Used by
    startup mode-consistency checks to decide posture without paying the
    KDF cost. A stale passphrase or tampered envelope still surfaces on the
    first actual read.
    """
    if not daily_passphrase_env_value():
        return False
    return Path(daily_key_path).is_file()


def recovery_pub_available(path: Path = RECOVERY_PUB_PATH_DEFAULT) -> bool:
    """Return True when the recovery public-key file exists and is readable.

    Used by the startup log line to decide whether new writes will be
    multi-recipient (``[daily, recovery]``) or daily-only.
    """
    p = Path(path)
    return p.is_file() and os.access(p, os.R_OK)


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
    "RECOVERY_PUB_PATH_DEFAULT",
    "daily_identity_loadable",
    "daily_passphrase_env_value",
    "load_daily_identity",
    "load_daily_identity_cached",
    "load_recovery_recipient",
    "mint_daily_identity",
    "recovery_pub_available",
    "wrap_daily_identity",
    "write_daily_key_file",
    "write_recovery_pub_file",
]
