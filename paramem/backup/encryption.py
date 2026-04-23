"""Fernet encryption wrapper for the backup subsystem + infrastructure store.

Provides a thin layer over ``cryptography.fernet.Fernet`` with:

- Lazy cipher construction (built on first use, cached in-module state).
- Per-artifact policy resolution (``auto`` / ``always`` / ``never``).
- Startup feasibility assertion (``always`` with no key → fatal error).
- 4-case mode-mismatch startup refuse (SECURITY.md §4).
- Envelope helpers (``write_infra_bytes`` / ``read_maybe_encrypted``) for any
  on-disk infrastructure metadata — graph, registry, queue, sidecars, etc.
- ``_clear_cipher_cache()`` — a **supported operational call** for key
  rotation (Slice 7).  It is *not* test-only; operator code may call it
  after rotating ``PARAMEM_MASTER_KEY`` in the environment.

Key source
----------
The Fernet key is read from ``os.environ["PARAMEM_MASTER_KEY"]`` on first
use.  The legacy name ``PARAMEM_SNAPSHOT_KEY`` is accepted as an alias for
one release and emits a single deprecation ``logger.warning`` when used.
If neither variable is set, ``encrypt_bytes`` / ``decrypt_bytes`` raise
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

Envelope format (PMEM1)
-----------------------
``write_infra_bytes`` / ``read_maybe_encrypted`` wrap the ciphertext with a
6-byte magic prefix ``b"PMEM1\\n"`` so that ``assert_mode_consistency`` can
classify an on-disk file as encrypted-or-plaintext without attempting to
decrypt it.  Plaintext JSON starts with ``{`` or ``[``; the magic is
unambiguous.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from cryptography.fernet import Fernet

from paramem.backup.types import ArtifactKind, EncryptAtRest, FatalConfigError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Env var names (rename with one-release alias)
# ---------------------------------------------------------------------------

MASTER_KEY_ENV_VAR: str = "PARAMEM_MASTER_KEY"
LEGACY_KEY_ENV_VAR: str = "PARAMEM_SNAPSHOT_KEY"

# Envelope magic for infrastructure files written via write_infra_bytes.
PMEM1_MAGIC: bytes = b"PMEM1\n"

# One-shot deprecation warning state.
_deprecation_warned: bool = False


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


def master_key_env_value() -> str | None:
    """Return the master key from env, preferring the new name.

    Reads ``PARAMEM_MASTER_KEY`` first; falls back to ``PARAMEM_SNAPSHOT_KEY``.
    Emits a one-shot deprecation warning when only the legacy name is set.
    Returns ``None`` when neither is set.
    """
    global _deprecation_warned
    value = os.environ.get(MASTER_KEY_ENV_VAR)
    if value:
        return value
    legacy = os.environ.get(LEGACY_KEY_ENV_VAR)
    if legacy:
        if not _deprecation_warned:
            logger.warning(
                "%s is deprecated — rename to %s. Alias accepted for one release.",
                LEGACY_KEY_ENV_VAR,
                MASTER_KEY_ENV_VAR,
            )
            _deprecation_warned = True
        return legacy
    return None


def master_key_loaded() -> bool:
    """Return ``True`` when a master key is present in the environment."""
    return master_key_env_value() is not None


def _get_cipher() -> Fernet:
    """Return the module-level cached Fernet cipher, building it on first call.

    Reads ``PARAMEM_MASTER_KEY`` from the environment on first call
    (or the legacy ``PARAMEM_SNAPSHOT_KEY`` alias with a deprecation warning);
    raises ``RuntimeError`` if neither is set.

    Returns
    -------
    Fernet
        The cached cipher instance.

    Raises
    ------
    RuntimeError
        If neither ``PARAMEM_MASTER_KEY`` nor the legacy
        ``PARAMEM_SNAPSHOT_KEY`` is set in the environment.
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

    **Supported operational call** — Slice 7's key-rotation handler calls this
    after updating ``PARAMEM_MASTER_KEY`` in the environment so the next
    ``encrypt_bytes`` / ``decrypt_bytes`` call builds a fresh cipher with the
    new key.

    Also resets the one-shot deprecation-warning flag so a subsequent legacy
    usage re-emits its warning.

    This function never raises; it is safe to call even when the cache is
    already empty.
    """
    global _cipher, _deprecation_warned
    _cipher = None
    _deprecation_warned = False


def current_key_fingerprint() -> str | None:
    """Return the 16-hex-char fingerprint of the current master key, or None.

    Stable across processes: identical key bytes yield an identical fingerprint.
    Reads the env every call — never cached — so a ``_clear_cipher_cache()``
    rotation is reflected immediately.

    Used by:
    - ``backup.write`` to populate ``ArtifactMeta.key_fingerprint``.
    - Slice 7's ``/backup/restore`` fingerprint-mismatch check.
    - Slice 7's attention populator for key-rotation detection.

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
        Whether a master key is available in the environment.

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
        Whether a master key is available in the environment.

    Raises
    ------
    FatalConfigError
        If ``encrypt_at_rest=always`` is configured but no key is loaded.
    """
    if key_loaded:
        return  # all policies are satisfiable when a key is present

    # Check global policy
    if config.encrypt_at_rest is EncryptAtRest.ALWAYS:
        raise FatalConfigError(
            f"encrypt_at_rest=always but {MASTER_KEY_ENV_VAR} is not set — "
            "server startup refused (spec §Security invariants)"
        )

    # Check per-kind overrides
    for kind, policy in config.per_kind.items():
        if policy is EncryptAtRest.ALWAYS:
            raise FatalConfigError(
                f"encrypt_at_rest=always for {kind.value!r} but "
                f"{MASTER_KEY_ENV_VAR} is not set — "
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
    """Return plaintext bytes from *path*, decrypting when PMEM1-wrapped.

    Sniffs the 6-byte ``PMEM1\\n`` magic prefix.  When present, the remainder
    is Fernet ciphertext; decryption requires a valid master key.  When
    absent, the file is returned verbatim (plaintext).

    This is the universal read path for any infrastructure file that may have
    been written via ``write_infra_bytes``.  Callers do not need to branch on
    key-loaded state.

    Parameters
    ----------
    path:
        Filesystem path to read.

    Returns
    -------
    bytes
        The decrypted (or verbatim) content.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    RuntimeError
        If the file carries the PMEM1 magic but no master key is loaded.
    cryptography.fernet.InvalidToken
        If the ciphertext is corrupt, tampered with, or was produced with a
        different key.
    """
    raw = Path(path).read_bytes()
    if raw.startswith(PMEM1_MAGIC):
        return decrypt_bytes(raw[len(PMEM1_MAGIC) :])
    return raw


def write_infra_bytes(path: Path, plaintext: bytes) -> None:
    """Atomically write *plaintext* to *path*, encrypting when a key is loaded.

    When a master key is present the on-disk body is the PMEM1 envelope
    (magic + Fernet ciphertext).  Otherwise the body is *plaintext* verbatim.
    The caller does not need to branch on key state.

    Write sequence: ``<path>.tmp`` → fsync → ``os.rename`` → fsync parent.

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
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    if master_key_loaded():
        body = PMEM1_MAGIC + encrypt_bytes(plaintext)
    else:
        body = plaintext

    with open(tmp_path, "wb") as fh:
        fh.write(body)
        fh.flush()
        os.fsync(fh.fileno())

    os.rename(tmp_path, path)

    # fsync the parent directory for rename durability (power-loss safety).
    try:
        dir_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        except OSError as exc:
            logger.warning("write_infra_bytes: parent dir fsync failed: %s", exc)
        finally:
            os.close(dir_fd)
    except OSError as exc:
        logger.warning("write_infra_bytes: could not open parent for fsync: %s", exc)


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
        Files carrying the PMEM1 envelope magic.
    plaintext_paths:
        Files that do NOT carry the magic but fall under the §4 encrypted-
        infrastructure list.  These are the targets of ``paramem
        encrypt-infra`` migration.
    """

    encrypted_paths: list[Path] = field(default_factory=list)
    plaintext_paths: list[Path] = field(default_factory=list)


def _probe_data_dir(data_dir: Path) -> ModeProbe:
    """Scan *data_dir* for infrastructure files and classify each as
    encrypted / plaintext.

    The set of paths inspected follows SECURITY.md §4's "encrypted
    infrastructure metadata" list — excluding the plaintext-by-design carve-
    outs ``state/trial.json`` and ``state/backup.json`` (control-plane only,
    no secrets).

    Empty / missing files do NOT contribute either classification — they are
    neutral.  The probe is conservative: only files that actually exist on
    disk steer the mode verdict.

    Sidecars under ``backups/`` (``*.meta.json``) are sampled because the
    core §4 contract includes manifest sidecars.

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
    if not data_dir.exists():
        return probe

    # Core infrastructure metadata paths.  Anchors follow current layout.
    candidates: list[Path] = []
    for name in ("graph.json", "registry.json", "indexed_key_registry.json"):
        candidates.append(data_dir / name)
    candidates.append(data_dir / "queue" / "post_session_queue.json")
    candidates.append(data_dir / "speaker_profiles.json")

    # Sidecars — any *.meta.json under backups/ (one sample per kind is enough).
    backups_root = data_dir / "backups"
    if backups_root.exists():
        for sidecar in backups_root.rglob("*.meta.json"):
            candidates.append(sidecar)

    for path in candidates:
        if not path.exists() or not path.is_file():
            continue
        if is_pmem1_envelope(path):
            probe.encrypted_paths.append(path)
        else:
            probe.plaintext_paths.append(path)

    return probe


def assert_mode_consistency(data_dir: Path, key_loaded: bool) -> None:
    """Refuse startup on mode mismatch (SECURITY.md §4, 4-case matrix).

    Cases (``key`` × ``on-disk``):

    - (set, encrypted)   → OK, proceed.
    - (set, plaintext)   → refuse; operator must run ``paramem encrypt-infra``.
    - (unset, plaintext) → OK, Security OFF mode.
    - (unset, encrypted) → refuse; operator restores the key or runs
      ``paramem decrypt-infra --i-accept-plaintext``.

    Mixed on-disk state (some encrypted, some plaintext files present) is
    also a refuse condition regardless of the key state.  Empty data
    directories are treated as consistent with either mode.

    Parameters
    ----------
    data_dir:
        Root of the data directory.
    key_loaded:
        Whether a master key is available in the environment.

    Raises
    ------
    FatalConfigError
        On any of the three refuse cases, with an operator-actionable
        message naming the CLI command to use.
    """
    probe = _probe_data_dir(Path(data_dir))

    has_enc = bool(probe.encrypted_paths)
    has_pt = bool(probe.plaintext_paths)

    # Mixed on-disk state — refuse regardless of key.
    if has_enc and has_pt:
        sample_enc = probe.encrypted_paths[0]
        sample_pt = probe.plaintext_paths[0]
        raise FatalConfigError(
            "Mixed encryption state on disk: "
            f"{sample_enc} is encrypted but {sample_pt} is plaintext "
            f"(and {len(probe.encrypted_paths) - 1} other encrypted + "
            f"{len(probe.plaintext_paths) - 1} other plaintext files). "
            "Run `paramem encrypt-infra` (if key set) or "
            "`paramem decrypt-infra --i-accept-plaintext` (if intentional "
            "plaintext) to reconcile the store."
        )

    if key_loaded and has_pt:
        raise FatalConfigError(
            f"{MASTER_KEY_ENV_VAR} is set but {len(probe.plaintext_paths)} "
            "infrastructure file(s) on disk are plaintext "
            f"(e.g. {probe.plaintext_paths[0]}). "
            "Run `paramem encrypt-infra` to migrate the store before startup."
        )

    if not key_loaded and has_enc:
        raise FatalConfigError(
            f"{MASTER_KEY_ENV_VAR} is not set but {len(probe.encrypted_paths)} "
            "infrastructure file(s) on disk are encrypted "
            f"(e.g. {probe.encrypted_paths[0]}). "
            f"Restore the key (via {MASTER_KEY_ENV_VAR} env var) or run "
            "`paramem decrypt-infra --i-accept-plaintext` to convert the "
            "store to plaintext."
        )

    # (set, encrypted) or (unset, plaintext) or empty store → OK.
    return
