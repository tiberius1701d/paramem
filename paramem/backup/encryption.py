"""Encryption wrapper for the backup subsystem + infrastructure store.

Mediates access to two on-disk envelope formats:

- **PMEM1** — the legacy Fernet envelope (``PMEM1\\n`` magic). Built from
  ``PARAMEM_MASTER_KEY`` via a cached :class:`cryptography.fernet.Fernet`.
  Retained for one release as a rollback safety net during the age
  migration.
- **age v1** — the two-identity envelope (``age-encryption.org/v1\\n``
  magic). Decrypted with the cached daily identity from
  :mod:`paramem.backup.key_store`. Primitives live in
  :mod:`paramem.backup.age_envelope`.

Services provided:

- Lazy cipher construction (built on first use, cached in-module state)
  for the Fernet path; ``_clear_cipher_cache()`` is a supported operator
  call after rotating ``PARAMEM_MASTER_KEY``.
- Uniform AUTO semantics: :func:`envelope_encrypt_bytes` encrypts when a
  key is loaded and returns plaintext otherwise; no per-artifact policy
  knob exists.  Operators opt into a fail-loud posture via the single
  uniform ``security.require_encryption`` flag enforced at server startup
  by :func:`paramem.server.security_posture.assert_startup_posture`.
- Mode-mismatch startup refuse (:func:`assert_mode_consistency`):
  classifies infrastructure files as PMEM1 / age / plaintext and refuses
  any combination that would be silently unreadable or that mixes
  plaintext with an encryption format. A mixed PMEM1 + age store with
  both keys loaded is permitted as a transitional state during the age
  migration and logged at WARN.
- The universal read path :func:`read_maybe_encrypted` dispatches by
  envelope magic so callers never branch on key-loaded state — a
  PMEM1-era file and an age-era file sitting in the same directory both
  unwrap transparently.

``encrypt_bytes`` / ``decrypt_bytes`` operate on single-file inputs only;
directory-shaped artifacts iterate per-file externally. The writer
:func:`write_infra_bytes` routes to age when the daily identity is
loadable (multi-recipient when ``recovery.pub`` is also present,
single-recipient when not) and falls back to PMEM1 when only the Fernet
master key is loaded. The ``paramem migrate-to-age`` CLI walks the
infrastructure paths and rewrites any lingering PMEM1 envelopes as age
multi-recipient envelopes; Fernet support is retained as a rollback
safety net for one release and then removed.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from cryptography.fernet import Fernet

from paramem.backup.age_envelope import (
    AGE_MAGIC,
    age_decrypt_bytes,
    age_encrypt_bytes,
    is_age_envelope,
)
from paramem.backup.types import FatalConfigError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Env var name
# ---------------------------------------------------------------------------

MASTER_KEY_ENV_VAR: str = "PARAMEM_MASTER_KEY"

# Envelope magic for infrastructure files written via write_infra_bytes.
PMEM1_MAGIC: bytes = b"PMEM1\n"


# ---------------------------------------------------------------------------
# Module-level cipher cache
# ---------------------------------------------------------------------------

_cipher: Fernet | None = None


def master_key_env_value() -> str | None:
    """Return the master key from the environment, or ``None`` when unset."""
    value = os.environ.get(MASTER_KEY_ENV_VAR)
    return value if value else None


def master_key_loaded() -> bool:
    """Return ``True`` when a master key is present in the environment."""
    return master_key_env_value() is not None


def _get_cipher() -> Fernet:
    """Return the module-level cached Fernet cipher, building it on first call.

    Reads ``PARAMEM_MASTER_KEY`` from the environment on first call;
    raises ``RuntimeError`` if it is not set.

    Returns
    -------
    Fernet
        The cached cipher instance.

    Raises
    ------
    RuntimeError
        If ``PARAMEM_MASTER_KEY`` is not set in the environment.
    FatalConfigError
        If the master key is set but is not a valid Fernet key (wrong length,
        invalid base64, etc.).  The raw key value is never included in the
        exception message.
    """
    global _cipher
    if _cipher is None:
        key = master_key_env_value()
        if not key:
            raise RuntimeError(f"{MASTER_KEY_ENV_VAR} is not set — encryption is unavailable")
        try:
            _cipher = Fernet(key.encode())
        except ValueError as exc:
            raise FatalConfigError(
                f"{MASTER_KEY_ENV_VAR} is not a valid Fernet key — "
                "ensure the value is a URL-safe base64-encoded 32-byte key"
            ) from exc
    return _cipher


def _clear_cipher_cache() -> None:
    """Invalidate the module-level cipher cache.

    **Supported operational call** — the key-rotation handler calls this
    after updating ``PARAMEM_MASTER_KEY`` in the environment so the next
    ``encrypt_bytes`` / ``decrypt_bytes`` call builds a fresh cipher with
    the new key.

    This function never raises; it is safe to call even when the cache is
    already empty.
    """
    global _cipher
    _cipher = None


def current_key_fingerprint() -> str | None:
    """Return the 16-hex-char fingerprint of the current master key, or None.

    Stable across processes: identical key bytes yield an identical fingerprint.
    Reads the env every call — never cached — so a ``_clear_cipher_cache()``
    rotation is reflected immediately.

    Used by:
    - ``backup.write`` to populate ``ArtifactMeta.key_fingerprint``.
    - ``/backup/restore`` to refuse on fingerprint mismatch unless
      ``force_rotate_key`` is set.
    - The key-rotation attention populator (when wired).

    Returns
    -------
    str | None
        16-character lowercase hex string, or ``None`` when no key is set.
    """
    key = master_key_env_value()
    if not key:
        return None
    return hashlib.sha256(key.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Encrypt / decrypt (single-file inputs — NIT 3)
# ---------------------------------------------------------------------------


def encrypt_bytes(plaintext: bytes) -> bytes:
    """Encrypt *plaintext* using the cached Fernet cipher.

    Operates on single-file inputs only; directory-level encryption (e.g.
    the BG-trainer resume directory) iterates per-file and calls this
    function once per file.

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
        If no master key is set.
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
        If no master key is set.
    cryptography.fernet.InvalidToken
        If the ciphertext is corrupt, tampered with, or was produced with a
        different key.
    """
    return _get_cipher().decrypt(ciphertext)


# ---------------------------------------------------------------------------
# Infrastructure envelope (PMEM1) — generic on-disk read/write helpers
# ---------------------------------------------------------------------------


def read_maybe_encrypted(path: Path) -> bytes:
    """Return plaintext bytes from *path*, dispatching by envelope magic.

    The universal read path for any infrastructure file that may have been
    written via :func:`write_infra_bytes`. Three on-disk shapes are handled
    transparently so callers never branch on key-loaded state:

    - **age v1** envelope (``age-encryption.org/v1\\n`` magic) — decrypted
      with the cached daily identity loaded from
      :func:`paramem.backup.key_store.load_daily_identity_cached`. The
      identity is unwrapped once on first read via the scrypt KDF and
      cached module-side; rotation handlers call
      :func:`paramem.backup.key_store._clear_daily_identity_cache`.
    - **PMEM1** envelope (``PMEM1\\n`` magic) — Fernet ciphertext, decrypted
      with the cached master key built from ``PARAMEM_MASTER_KEY``.
      Retained for one release as a rollback safety net while the age
      migration is in flight.
    - No recognised magic — returned verbatim (plaintext pass-through).

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    RuntimeError
        If the file carries an encryption magic but the corresponding key /
        identity is not loaded — actionable message names the env var the
        operator needs to set.
    cryptography.fernet.InvalidToken
        PMEM1 ciphertext is corrupt, tampered with, or was produced with a
        different key.
    pyrage.DecryptError
        age ciphertext is corrupt, tampered with, or cannot be decrypted by
        the loaded daily identity (neither daily nor recovery recipient
        match).
    """
    raw = Path(path).read_bytes()
    if raw.startswith(AGE_MAGIC):
        # Late-lookup the key_store module so tests (and a future operator
        # override) can monkeypatch DAILY_KEY_PATH_DEFAULT without having the
        # value frozen into this function's defaults at import time.
        from paramem.backup import key_store as _ks

        try:
            identity = _ks.load_daily_identity_cached(_ks.DAILY_KEY_PATH_DEFAULT)
        except RuntimeError as exc:
            raise RuntimeError(
                f"{path} is an age envelope but the daily identity is not loaded: "
                f"{exc}. Set {_ks.DAILY_PASSPHRASE_ENV_VAR} and ensure "
                f"{_ks.DAILY_KEY_PATH_DEFAULT} exists."
            ) from exc
        return age_decrypt_bytes(raw, [identity])
    if raw.startswith(PMEM1_MAGIC):
        return decrypt_bytes(raw[len(PMEM1_MAGIC) :])
    return raw


def _atomic_write_bytes(path: Path, body: bytes) -> None:
    """Shared atomic-write core: ``<path>.tmp`` → fsync → rename → fsync parent.

    On any failure before the rename completes, the temp file is removed so
    no partial content is left on disk.  Callers that need to make an
    encryption decision layer it on top of this helper.
    """
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    try:
        with open(tmp_path, "wb") as fh:
            fh.write(body)
            fh.flush()
            os.fsync(fh.fileno())
        os.rename(tmp_path, path)
    except BaseException:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise

    # fsync the parent directory for rename durability (power-loss safety).
    try:
        dir_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        except OSError as exc:
            logger.warning("_atomic_write_bytes: parent dir fsync failed: %s", exc)
        finally:
            os.close(dir_fd)
    except OSError as exc:
        logger.warning("_atomic_write_bytes: could not open parent for fsync: %s", exc)


def envelope_encrypt_bytes(plaintext: bytes) -> bytes:
    """Return encrypted envelope bytes based on the loaded-key posture.

    Priority (highest first):

    1. **age multi-recipient** ``[daily, recovery]`` — when the daily
       identity is loadable AND ``recovery.pub`` is on disk.
    2. **age single-recipient** ``[daily]`` — daily loadable but
       ``recovery.pub`` missing. Degraded mode; the startup log warns.
    3. **PMEM1** (legacy Fernet envelope, magic + Fernet token) — only the
       Fernet master key is loaded.
    4. **Plaintext** — no key material loaded; caller should gate on this
       case explicitly if they require encryption.

    Unlike :func:`encrypt_bytes` (which produces a bare Fernet token),
    this always returns a magic-prefixed envelope when encryption happens.
    Used by both :func:`write_infra_bytes` (writes to disk) and the backup
    subsystem (holds the bytes in hand for sidecar construction).
    """
    from paramem.backup import key_store as _ks

    if _ks.daily_identity_loadable(_ks.DAILY_KEY_PATH_DEFAULT):
        try:
            daily = _ks.load_daily_identity_cached(_ks.DAILY_KEY_PATH_DEFAULT)
        except Exception:  # noqa: BLE001 — graceful fallback on any unwrap failure
            daily = None
        if daily is not None:
            recipients = [daily.to_public()]
            if _ks.recovery_pub_available(_ks.RECOVERY_PUB_PATH_DEFAULT):
                recipients.append(_ks.load_recovery_recipient(_ks.RECOVERY_PUB_PATH_DEFAULT))
            return age_encrypt_bytes(plaintext, recipients)
    if master_key_loaded():
        return PMEM1_MAGIC + encrypt_bytes(plaintext)
    return plaintext


def envelope_decrypt_bytes(raw: bytes) -> bytes:
    """Return plaintext from envelope-wrapped bytes, dispatching by magic.

    Recognises:

    - age v1 envelope (``age-encryption.org/v1\\n``) → decrypt via the cached
      daily identity.
    - PMEM1 envelope (``PMEM1\\n``) → strip magic + Fernet decrypt.
    - **Bare Fernet token** (no magic, base64 body) → Fernet decrypt. This
      is the legacy shape produced by the backup subsystem before the age
      flip; kept so pre-envelope backups still restore.

    Unlike :func:`read_maybe_encrypted`, this does NOT treat non-magic bytes
    as plaintext — the caller is expected to know the bytes are encrypted
    (e.g. via a sidecar ``meta.encrypted`` field). Pass-through plaintext is
    the :func:`read_maybe_encrypted` behaviour.
    """
    if raw.startswith(AGE_MAGIC):
        from paramem.backup import key_store as _ks

        try:
            identity = _ks.load_daily_identity_cached(_ks.DAILY_KEY_PATH_DEFAULT)
        except RuntimeError as exc:
            raise RuntimeError(
                f"age envelope encountered but the daily identity is not loaded: {exc}. "
                f"Set {_ks.DAILY_PASSPHRASE_ENV_VAR} and ensure "
                f"{_ks.DAILY_KEY_PATH_DEFAULT} exists."
            ) from exc
        return age_decrypt_bytes(raw, [identity])
    if raw.startswith(PMEM1_MAGIC):
        return decrypt_bytes(raw[len(PMEM1_MAGIC) :])
    # Legacy bare Fernet token (pre-envelope backup format).
    return decrypt_bytes(raw)


def write_infra_bytes(path: Path, plaintext: bytes) -> None:
    """Atomically write *plaintext* to *path*, encrypting when a key is loaded.

    Delegates format selection to :func:`envelope_encrypt_bytes`. See its
    docstring for the four-tier priority (age multi / age single / PMEM1 /
    plaintext). Callers do not need to branch on key state; the universal
    reader :func:`read_maybe_encrypted` unwraps any of the three formats.

    Parameters
    ----------
    path:
        Destination path.  Parent directory must exist.
    plaintext:
        Raw content to write.

    Raises
    ------
    OSError
        On any filesystem error.
    """
    _atomic_write_bytes(Path(path), envelope_encrypt_bytes(plaintext))


def write_plaintext_atomic(path: Path, plaintext: bytes) -> None:
    """Atomically write *plaintext* to *path*, never encrypting.

    Used by ``paramem decrypt-infra`` to convert a ciphertext file to
    plaintext regardless of whether a master key is currently loaded.
    Normal infrastructure writers should use ``write_infra_bytes`` instead —
    the naked plaintext variant is a migration-tool-only escape hatch.
    """
    _atomic_write_bytes(Path(path), plaintext)


def is_pmem1_envelope(path: Path) -> bool:
    """Return ``True`` when *path* is prefixed with the PMEM1 magic.

    Cheap probe used by ``assert_mode_consistency`` — reads only the first
    few bytes of the file.
    """
    try:
        with open(path, "rb") as fh:
            head = fh.read(len(PMEM1_MAGIC))
    except OSError:
        return False
    return head == PMEM1_MAGIC


# ---------------------------------------------------------------------------
# 4-case mode-mismatch startup refuse (SECURITY.md §4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModeProbe:
    """Result of scanning the data directory for encryption-state evidence.

    Attributes
    ----------
    encrypted_paths:
        Files carrying the PMEM1 envelope magic. Legacy Fernet ciphertext.
    age_paths:
        Files carrying the age v1 envelope magic. Decryptable with the
        loaded daily identity.
    plaintext_paths:
        Files that carry neither encryption magic but fall under the §4
        encrypted-infrastructure list.  These are the targets of ``paramem
        encrypt-infra`` migration.
    """

    encrypted_paths: list[Path] = field(default_factory=list)
    age_paths: list[Path] = field(default_factory=list)
    plaintext_paths: list[Path] = field(default_factory=list)


def infra_paths(data_dir: Path) -> list[Path]:
    """Return the list of infrastructure files subject to envelope encryption.

    Single source of truth for:
    - ``_probe_data_dir`` — the startup mode-consistency scan.
    - ``paramem encrypt-infra`` / ``paramem decrypt-infra`` — the migration
      commands that convert between plaintext and ciphertext in bulk.

    Paths that do not currently exist on disk are still returned — the
    caller filters as needed.  This keeps the "what counts as infra
    metadata" definition in one place regardless of whether the operator
    has populated every path yet.

    Excluded (plaintext-by-design per SECURITY.md §4 carve-out):
    - ``state/trial.json`` and ``state/backup.json`` — control-plane only.
    - Backup ``*.meta.json`` sidecars — operator visibility on wrong-key
      restore trumps the marginal info hiding.

    Parameters
    ----------
    data_dir:
        Root of the ParaMem data directory (typically
        ``configs/server.yaml``'s ``paths.data``).

    Returns
    -------
    list[Path]
        Ordered list of candidate paths.  Neither filtered by existence
        nor classified by on-disk state.
    """
    data_dir = Path(data_dir)
    paths: list[Path] = [
        data_dir / "graph.json",
        data_dir / "registry.json",
        data_dir / "indexed_key_registry.json",
        data_dir / "registry" / "key_metadata.json",
        data_dir / "speaker_profiles.json",
        data_dir / "adapters" / "post_session_queue.json",
        data_dir / "adapters" / "keyed_pairs.json",
        data_dir / "adapters" / "episodic" / "keyed_pairs.json",
        data_dir / "adapters" / "semantic" / "keyed_pairs.json",
        data_dir / "adapters" / "procedural" / "keyed_pairs.json",
    ]
    # BG-trainer resume states live under per-job in_training directories.
    adapters_root = data_dir / "adapters"
    if adapters_root.exists():
        for resume in adapters_root.rglob("in_training/resume_state.json"):
            paths.append(resume)
    return paths


def _probe_data_dir(data_dir: Path) -> ModeProbe:
    """Scan *data_dir* for infrastructure files and classify each as
    encrypted / plaintext.

    Uses ``infra_paths`` as the authoritative candidate set.  Missing files
    do NOT contribute to either classification — only files that actually
    exist on disk steer the mode verdict.

    Parameters
    ----------
    data_dir:
        Root of the ParaMem data directory (typically
        ``configs/server.yaml``'s ``paths.data``).

    Returns
    -------
    ModeProbe
    """
    probe = ModeProbe()
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return probe

    for path in infra_paths(data_dir):
        if not path.exists() or not path.is_file():
            continue
        if is_pmem1_envelope(path):
            probe.encrypted_paths.append(path)
        elif is_age_envelope(path):
            probe.age_paths.append(path)
        else:
            probe.plaintext_paths.append(path)

    return probe


def assert_mode_consistency(
    data_dir: Path,
    key_loaded: bool,
    *,
    daily_identity_loadable: bool = False,
) -> None:
    """Refuse startup on mode mismatch; permit transitional age/PMEM1 states.

    Extends the original Fernet-only four-case matrix to cover the age
    envelope format introduced by the two-identity key rollout. The
    on-disk classifier now distinguishes three file shapes — PMEM1,
    age, plaintext — and the refusal rules below preserve the spirit of
    the original: Security-ON cannot cohabit with plaintext, and an
    encryption magic the server cannot decrypt is a fatal mismatch.

    Acceptable combinations:

    - Empty store                               → OK regardless of keys.
    - Plaintext only, no keys loaded            → OK (Security OFF).
    - PMEM1 only, Fernet key loaded             → OK.
    - age only, daily identity loadable         → OK.
    - Mixed PMEM1 + age, both keys available    → OK, logged as
      *transitional* while the age migration is in flight.

    Refusal cases:

    - Plaintext present while any key is loaded → operator must migrate
      (``paramem encrypt-infra`` for the Fernet path; the future
      ``paramem migrate-to-age`` for age).
    - PMEM1 present without the Fernet key      → restore the key or
      convert plaintext with ``--i-accept-plaintext``.
    - age present without the daily identity    → set
      ``PARAMEM_DAILY_PASSPHRASE`` and ensure the daily key file exists.
    - Mixed age + plaintext, or PMEM1 + plaintext → refuse (same as the
      legacy matrix).

    Parameters
    ----------
    data_dir:
        Root of the data directory.
    key_loaded:
        Whether the Fernet master key (``PARAMEM_MASTER_KEY``) is available.
    daily_identity_loadable:
        Whether the daily age identity is loadable (passphrase env var set
        + daily key file exists). Does not force an unwrap; a stale
        passphrase surfaces on first actual read.

    Raises
    ------
    FatalConfigError
        On any refusal case above, with an operator-actionable message.
    """
    from paramem.backup.key_store import DAILY_PASSPHRASE_ENV_VAR

    probe = _probe_data_dir(Path(data_dir))

    has_pmem1 = bool(probe.encrypted_paths)
    has_age = bool(probe.age_paths)
    has_pt = bool(probe.plaintext_paths)

    # Any mixing of plaintext with an encrypted format is a fatal mismatch.
    if has_pt and (has_pmem1 or has_age):
        sample_enc = (probe.encrypted_paths or probe.age_paths)[0]
        sample_pt = probe.plaintext_paths[0]
        raise FatalConfigError(
            "Mixed encryption state on disk: "
            f"{sample_enc} is encrypted but {sample_pt} is plaintext "
            f"({len(probe.encrypted_paths)} PMEM1, {len(probe.age_paths)} age, "
            f"{len(probe.plaintext_paths)} plaintext in total). "
            "Run `paramem encrypt-infra` (if key set) or "
            "`paramem decrypt-infra --i-accept-plaintext` (if intentional "
            "plaintext) to reconcile the store."
        )

    # Plaintext files while any key is loaded → encryption enabled but the
    # store is not migrated. Refuse before writing anything.
    if has_pt and (key_loaded or daily_identity_loadable):
        raise FatalConfigError(
            f"A key is loaded but {len(probe.plaintext_paths)} infrastructure "
            f"file(s) on disk are plaintext (e.g. {probe.plaintext_paths[0]}). "
            "Run `paramem encrypt-infra` to migrate the store before startup."
        )

    # PMEM1 files present without the Fernet key → unreadable.
    if has_pmem1 and not key_loaded:
        raise FatalConfigError(
            f"{MASTER_KEY_ENV_VAR} is not set but {len(probe.encrypted_paths)} "
            "infrastructure file(s) on disk are PMEM1-encrypted "
            f"(e.g. {probe.encrypted_paths[0]}). "
            f"Restore the key (via {MASTER_KEY_ENV_VAR} env var) or run "
            "`paramem decrypt-infra --i-accept-plaintext` to convert the "
            "store to plaintext."
        )

    # age files present without the daily identity → unreadable.
    if has_age and not daily_identity_loadable:
        raise FatalConfigError(
            f"{len(probe.age_paths)} infrastructure file(s) on disk are age-"
            f"encrypted (e.g. {probe.age_paths[0]}) but the daily identity "
            f"is not loadable. Set {DAILY_PASSPHRASE_ENV_VAR} and ensure "
            "~/.config/paramem/daily_key.age exists before startup."
        )

    # Mixed PMEM1 + age (both decryptable) is a transitional state during the
    # age migration. Log at WARN so operators see the pending work without
    # blocking startup.
    if has_pmem1 and has_age:
        logger.warning(
            "Transitional encryption state on disk: %d PMEM1 file(s) alongside "
            "%d age file(s). Run `paramem migrate-to-age` to complete the flip.",
            len(probe.encrypted_paths),
            len(probe.age_paths),
        )

    # Otherwise the store is consistent with the loaded keys — proceed.
    return
