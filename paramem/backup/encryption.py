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
use.  If the variable is not set, ``encrypt_bytes`` / ``decrypt_bytes``
raise ``RuntimeError``.  Import is always safe — the key is never read at
import time.

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
# Env var name
# ---------------------------------------------------------------------------

MASTER_KEY_ENV_VAR: str = "PARAMEM_MASTER_KEY"

# Envelope magic for infrastructure files written via write_infra_bytes.
PMEM1_MAGIC: bytes = b"PMEM1\n"


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


def write_infra_bytes(path: Path, plaintext: bytes) -> None:
    """Atomically write *plaintext* to *path*, encrypting when a key is loaded.

    When a master key is present the on-disk body is the PMEM1 envelope
    (magic + Fernet ciphertext).  Otherwise the body is *plaintext* verbatim.
    The caller does not need to branch on key state.

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
    if master_key_loaded():
        body = PMEM1_MAGIC + encrypt_bytes(plaintext)
    else:
        body = plaintext
    _atomic_write_bytes(Path(path), body)


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
        Files carrying the PMEM1 envelope magic.
    plaintext_paths:
        Files that do NOT carry the magic but fall under the §4 encrypted-
        infrastructure list.  These are the targets of ``paramem
        encrypt-infra`` migration.
    """

    encrypted_paths: list[Path] = field(default_factory=list)
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
