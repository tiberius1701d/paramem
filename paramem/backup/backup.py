"""High-level backup API — write, read, prune.

Public surface
--------------
- ``write()``  — write an artifact + sidecar into a new slot directory.
- ``read()``   — read an artifact from a slot, validate, decrypt if needed.
- ``prune()``  — apply a retention policy to a kind directory.
- ``sweep_orphan_pending()`` — startup sweep (re-exported from ``atomic``).

**Filesystem requirement** — *base_dir* (and all slot directories) must live
on the Linux-native ext4 filesystem, *not* on a 9P-mounted Windows path
(``/mnt/c/…``).  ``os.rename()`` on 9P-mounted paths has known atomicity
edge cases on WSL2.

Slot naming
-----------
Slots are named ``YYYYMMDD-HHMMSSff`` where ``ff`` is hundredths of a second.
Two writes within the same hundredth-of-a-second window produce a
``FileExistsError`` (via ``rename_pending_to_slot``).  Retry after the
conflicting slot is removed is safe — the collision is not a partial-write
indicator.

Content-hash rule
-----------------
``content_sha256`` in the sidecar is the SHA-256 of the raw bytes *as written
to disk*.  For encrypted artifacts this is the hash of the ciphertext; for
plaintext artifacts it is the hash of the source bytes.  Whitespace changes
and key-order changes in YAML/JSON source data are visible as hash changes.
No canonicalization is applied.
"""

from __future__ import annotations

import hashlib
import json as _json
import logging
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

from paramem.backup.age_envelope import AGE_MAGIC
from paramem.backup.atomic import rename_pending_to_slot
from paramem.backup.atomic import sweep_orphan_pending as _sweep_pending
from paramem.backup.encryption import (
    envelope_decrypt_bytes,
    envelope_encrypt_bytes,
    read_maybe_encrypted,
)
from paramem.backup.hashing import (
    content_sha256_bytes,
)
from paramem.backup.meta import read_meta, verify_fingerprint, write_meta
from paramem.backup.types import (
    BUNDLE_SCHEMA_VERSION,
    SCHEMA_VERSION,
    ArtifactKind,
    ArtifactMeta,
    BackupError,
    BundleManifest,
    BundleManifestError,
    FingerprintMismatchError,
    PruneReport,
    RestoreAbortedError,
    RestoreResult,
    RetentionPolicy,
)

# iter_interim_dirs is imported at module level (no cycle: interim_adapter
# does not import from paramem.backup.backup).
from paramem.memory.interim_adapter import iter_interim_dirs

logger = logging.getLogger(__name__)

_PENDING_DIR_NAME = ".pending"


# ---------------------------------------------------------------------------
# Public re-export of sweep_orphan_pending
# ---------------------------------------------------------------------------


def sweep_orphan_pending(base_dir: Path) -> int:
    """Remove ``.pending/`` residue in *base_dir* and return count of removed entries.

    Called at server startup before any write operations.  Each kind's backup
    directory is passed separately; callers that manage multiple kinds should
    call this once per kind directory (or iterate over subdirectories of the
    global backups root).

    Parameters
    ----------
    base_dir:
        The per-kind backup directory (e.g. ``data/ha/backups/config/``).

    Returns
    -------
    int
        Number of orphaned pending sub-directories removed.
    """
    removed = _sweep_pending(base_dir)
    return len(removed)


# ---------------------------------------------------------------------------
# Slot name helpers
# ---------------------------------------------------------------------------


def _slot_name_now() -> str:
    """Return a ``YYYYMMDD-HHMMSSff`` slot name aligned to UTC now.

    ``ff`` = hundredths of a second (microseconds // 10000).
    """
    now = datetime.now(tz=timezone.utc)
    hh = now.microsecond // 10000
    return now.strftime("%Y%m%d-%H%M%S") + f"{hh:02d}"


def _artifact_filename(kind: ArtifactKind, timestamp: str, encrypted: bool) -> str:
    """Return the artifact filename for a given kind and timestamp.

    Parameters
    ----------
    kind:
        Artifact kind (determines the filename prefix).
    timestamp:
        Slot timestamp string (``YYYYMMDD-HHMMSSff``).
    encrypted:
        When ``True``, appends ``.enc`` to signal ciphertext.
    """
    base = f"{kind.value}-{timestamp}.bin"
    return base + ".enc" if encrypted else base


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fsync_dir(path: Path) -> None:
    """fsync a directory inode for rename-durability on Linux/ext4.

    After ``os.rename(pending_slot, slot_dir)`` the rename is atomic (POSIX),
    but the new directory entry is not guaranteed to survive a power loss until
    the parent directory inode is fsynced.  This function opens the directory
    with ``O_RDONLY`` (required on Linux for ``fsync`` on a directory fd),
    fsyncs, and closes, wrapping everything in a try/finally.

    On filesystems or platforms where directory fsync is not supported (e.g.
    some network mounts or Windows-backed 9P mounts), ``EINVAL`` or similar is
    caught and logged as WARN rather than raised — the rename already succeeded
    and the durability guarantee is best-effort on those filesystems.

    Parameters
    ----------
    path:
        Directory to fsync.  If *path* does not exist, the call is a no-op.
    """
    if not path.exists():
        return
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    except OSError as exc:
        logger.warning("_fsync_dir: directory fsync failed for %s: %s", path, exc)
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------
# _promote_slot() — shared pending-dir allocation + atomic rename core
# ---------------------------------------------------------------------------


def _promote_slot(base_dir: Path) -> tuple[Path, str]:
    """Allocate a unique ``.pending/<ts>/`` directory and return it with its timestamp.

    This is the shared core used by both ``write()`` and ``write_bundle()``.
    It handles the timestamp-collision retry loop: if the candidate timestamp
    (or its pending directory) already exists, the clock is bumped by one
    hundredth of a second and a new candidate is tried.  Both the pending
    directory *and* the eventual final slot directory are checked so that a
    previously-promoted slot's timestamp is never reused.

    The caller is responsible for:

    1. Writing all files into the returned ``pending_slot`` directory.
    2. fsyncing those files and the pending directory.
    3. Calling ``rename_pending_to_slot(pending_slot, base_dir / timestamp)``
       to atomically promote the slot.
    4. fsyncing the parent directories for rename durability.

    A crash between step 1–2 and step 3 leaves a ``.pending/<ts>/`` residue
    that ``sweep_orphan_pending`` removes on startup.

    Parameters
    ----------
    base_dir:
        The per-kind or per-bundle backup directory.  Created with parents if
        absent.  The ``.pending/`` subdirectory is created inside it.

    Returns
    -------
    tuple[Path, str]
        ``(pending_slot, timestamp)`` where *pending_slot* is the freshly-
        created pending directory and *timestamp* is the ``YYYYMMDD-HHMMSSff``
        string used to name both the pending and final slot directories.

    Raises
    ------
    BackupError
        If a unique pending slot could not be allocated after 10 collision
        retries (pathological — frozen clock or extreme write rate).
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    pending_root = base_dir / _PENDING_DIR_NAME
    pending_root.mkdir(exist_ok=True)

    _MAX_COLLISION_RETRIES = 10
    _ts_now = datetime.now(tz=timezone.utc)
    for _attempt in range(_MAX_COLLISION_RETRIES):
        hh = _ts_now.microsecond // 10000
        _ts_candidate = _ts_now.strftime("%Y%m%d-%H%M%S") + f"{hh:02d}"
        # A timestamp is usable only when neither the pending nor the final
        # slot directory already exists.  The final-slot check handles the
        # case where a previous write already promoted the same timestamp.
        _candidate_pending = pending_root / _ts_candidate
        _candidate_final = base_dir / _ts_candidate
        if _candidate_final.exists():
            _ts_now = _ts_now + timedelta(microseconds=10_000)
            continue
        try:
            _candidate_pending.mkdir(exist_ok=False)
            return _candidate_pending, _ts_candidate
        except FileExistsError:
            # Bump by one hundredth of a second and retry.
            _ts_now = _ts_now + timedelta(microseconds=10_000)

    raise BackupError(
        "could not allocate unique pending slot after "
        f"{_MAX_COLLISION_RETRIES} attempts — clock skew or extreme write rate"
    )


# ---------------------------------------------------------------------------
# write()
# ---------------------------------------------------------------------------


def write(
    kind: ArtifactKind,
    source: bytes | Path,
    meta_fields: dict,
    *,
    base_dir: Path,
) -> Path:
    """Write an artifact + sidecar into a new timestamped slot directory.

    Sequence (crash-safe):
    1. Determine whether the daily age identity is loadable.
    2. Create ``.pending/<ts>/`` inside *base_dir* via ``_promote_slot``.
    3. Write artifact file (ciphertext or plaintext) inside pending.
    4. Compute content hash of the on-disk bytes.
    5. Write ``.meta.json`` sidecar inside pending.
    6. ``fsync`` the artifact file, sidecar, and the pending directory entry.
    7. ``os.rename(.pending/<ts>/, <base_dir>/<ts>/)`` — atomic promotion.

    A crash at any step leaves either no slot (steps 2–6) or a ``.pending/``
    residue (step 7 incomplete) that ``sweep_orphan_pending`` removes on
    startup.

    Parameters
    ----------
    kind:
        Which artifact type is being stored.
    source:
        Either raw bytes (in-memory) or a ``Path`` to a file to read.  File
        paths are read in full; the source file is *not* deleted.
    meta_fields:
        Caller-supplied metadata.  Required keys: ``"tier"`` (str).  Optional:
        ``"label"`` (str | None), any future kind-specific extension fields.
    base_dir:
        Per-kind backup directory (e.g. ``data/ha/backups/config/``).
        Created with parents if absent.

    Returns
    -------
    Path
        The promoted slot directory (``<base_dir>/<ts>/``).

    Raises
    ------
    BackupError
        If a unique pending slot could not be allocated after 10 collision
        retries (pathological — frozen clock or extreme write rate).
    OSError
        On any filesystem error.
    """
    # --- resolve payload bytes ---
    payload: bytes = source if isinstance(source, bytes) else Path(source).read_bytes()

    # --- AUTO semantics: encrypt when the daily identity is loadable, else
    # plaintext. envelope_encrypt_bytes produces an age envelope or
    # returns the raw plaintext. Operators who want fail-loud set
    # security.require_encryption at startup; no per-write policy knob.
    # Late-bind key_store attrs so tests can monkeypatch the default path.
    from paramem.backup import key_store as _ks

    do_encrypt = _ks.daily_identity_loadable(_ks.DAILY_KEY_PATH_DEFAULT)
    on_disk_bytes = envelope_encrypt_bytes(payload) if do_encrypt else payload

    # --- compute content hash (ciphertext when encrypted, plaintext otherwise) ---
    hash_hex = content_sha256_bytes(on_disk_bytes)

    # --- allocate pending slot (with collision retry) ---
    encrypted_flag = do_encrypt
    pending_slot, timestamp = _promote_slot(Path(base_dir))

    artifact_filename = _artifact_filename(kind, timestamp, encrypted_flag)

    artifact_path = pending_slot / artifact_filename

    # --- write artifact ---
    artifact_path.write_bytes(on_disk_bytes)

    # --- build and write sidecar ---
    meta = ArtifactMeta(
        schema_version=SCHEMA_VERSION,
        kind=kind,
        timestamp=timestamp,
        content_sha256=hash_hex,
        size_bytes=len(on_disk_bytes),
        encrypted=encrypted_flag,
        tier=meta_fields.get("tier", "manual"),
        label=meta_fields.get("label"),
        pre_trial_hash=meta_fields.get("pre_trial_hash"),
    )
    sidecar_path = write_meta(pending_slot, meta)

    # --- fsync artifact, sidecar, and directory entry ---
    with open(artifact_path, "rb") as fh:
        os.fsync(fh.fileno())
    # fsync sidecar — path returned directly by write_meta; no existence guard needed
    with open(sidecar_path, "rb") as fh:
        os.fsync(fh.fileno())
    # fsync the directory entry
    dir_fd = os.open(str(pending_slot), os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)

    # --- atomic promotion ---
    base_dir = Path(base_dir)
    slot_dir = base_dir / timestamp
    rename_pending_to_slot(pending_slot, slot_dir)

    # --- fsync parent directories for rename durability (power-loss safety) ---
    # On Linux/ext4 the rename is atomic for ordering, but the directory entry
    # is not guaranteed durable across power loss until the parent inode is
    # fsynced.  We fsync both the immediate parent (kind directory) and base_dir
    # in case base_dir/kind/ was freshly created during this write.
    for parent in (slot_dir.parent, base_dir):
        _fsync_dir(parent)

    logger.debug(
        "backup.write: %s slot written to %s (encrypted=%s, %d bytes)",
        kind.value,
        slot_dir,
        encrypted_flag,
        len(on_disk_bytes),
    )
    return slot_dir


# ---------------------------------------------------------------------------
# read()
# ---------------------------------------------------------------------------


def read(slot_dir: Path) -> tuple[bytes, ArtifactMeta]:
    """Read an artifact from *slot_dir*, validate, and decrypt if needed.

    Steps:
    1. Read and validate the sidecar (``read_meta`` — schema_version gate).
    2. Locate the artifact file in the slot directory.
    3. Read raw artifact bytes.
    4. Verify content hash (raw bytes as stored — ciphertext if encrypted).
    5. Decrypt if ``meta.encrypted``.
    6. Return ``(plaintext_bytes, meta)``.

    Parameters
    ----------
    slot_dir:
        A fully-promoted slot directory containing an artifact + ``.meta.json``.

    Returns
    -------
    tuple[bytes, ArtifactMeta]
        The decrypted plaintext bytes and the validated sidecar metadata.

    Raises
    ------
    MetaSchemaError
        If the sidecar fails schema validation.
    FingerprintMismatchError
        If the artifact bytes do not match the stored hash, or if the sidecar
        is present but the artifact file is missing (partial slot — sidecar
        without artifact).
    FileNotFoundError
        If the slot directory or sidecar file is missing.
    RuntimeError
        If the artifact is age-encrypted but the daily identity is not
        loadable (``PARAMEM_DAILY_PASSPHRASE`` unset or daily key file missing).
    """
    slot_dir = Path(slot_dir)

    # 1. Read and validate sidecar
    meta = read_meta(slot_dir)

    # 2. Locate artifact file — raise FingerprintMismatchError for partial
    #    slots (valid sidecar present but artifact file absent).
    try:
        artifact_path = _find_artifact(slot_dir, meta)
    except FileNotFoundError:
        canonical = slot_dir / _artifact_filename(meta.kind, meta.timestamp, meta.encrypted)
        raise FingerprintMismatchError(f"artifact file missing: {canonical}") from None

    # 3. Read raw bytes
    raw_bytes = artifact_path.read_bytes()

    # 4. Verify content hash against raw on-disk bytes
    verify_fingerprint(slot_dir, artifact_path)

    # 5. Decrypt if needed. envelope_decrypt_bytes expects age-wrapped bytes;
    #    meta.encrypted is authoritative about whether to decrypt at all.
    if meta.encrypted:
        plaintext = envelope_decrypt_bytes(raw_bytes)
    else:
        plaintext = raw_bytes

    return plaintext, meta


def _find_artifact(slot_dir: Path, meta: ArtifactMeta) -> Path:
    """Locate the artifact file in *slot_dir*, excluding the sidecar.

    Prefers the canonical filename (``<kind>-<timestamp>.bin[.enc]``) but
    falls back to any non-``.meta.json`` file in the slot.

    Raises ``FileNotFoundError`` if no artifact is found.
    """
    # Try canonical name first
    for suffix in (".bin.enc", ".bin"):
        candidate = slot_dir / f"{meta.kind.value}-{meta.timestamp}{suffix}"
        if candidate.exists():
            return candidate

    # Fallback: any file that is not the sidecar
    for entry in slot_dir.iterdir():
        if not entry.name.endswith(".meta.json"):
            return entry

    raise FileNotFoundError(f"No artifact file found in slot {slot_dir}")


# ---------------------------------------------------------------------------
# write_bundle()
# ---------------------------------------------------------------------------

_BUNDLE_MANIFEST_FILENAME = "bundle.meta.json"
_BUNDLE_EXCLUDED_DEFAULTS = [
    "graph (RAM-only by design; not required for recall recovery)",
    "interim/checkpoint weights (regenerable from training; excluded by adapter_scope)",
    "keyed_pairs (transient; regenerated from graph on every cycle; not on disk)",
]

# Files inside an adapter slot that are always excluded from bundles — these are
# transient training scaffolding that must not be captured.
_ADAPTER_EXCLUDED_PATTERNS = frozenset({"resume_state.json", "in_training", "bg_checkpoint"})
# Subdirectory prefixes that are transient scaffolding.
_ADAPTER_EXCLUDED_DIR_PREFIXES = ("checkpoint-", "in_training", "bg_checkpoint")


def _copy_artifact(
    src: Path,
    dst: Path,
) -> dict:
    """Copy one artifact file into the bundle, respecting the encrypt-as-is rule.

    Weight blobs (``.safetensors``) and other files that are already age
    envelopes are copied **byte-for-byte** (no double-encrypt).  Files that
    are plaintext go through the AUTO encryption path
    (``envelope_encrypt_bytes``).

    Returns a file-inventory dict entry:
    ``{"path": rel_str, "content_sha256": hex, "encrypted": bool, "size_bytes": int}``.
    The ``path`` key is set to an empty string here; callers must fill it with
    the relative path inside the bundle slot.

    Parameters
    ----------
    src:
        Source file path on disk.
    dst:
        Destination path inside the ``.pending/<ts>/`` directory.

    Returns
    -------
    dict
        File inventory entry (``path`` is empty; caller sets it).
    """
    raw_bytes = src.read_bytes()
    if raw_bytes.startswith(AGE_MAGIC):
        # Already an age envelope — copy verbatim to avoid double-encryption.
        # Weight blobs (.safetensors) and other artifacts written via the live
        # encryption path are already [daily, recovery] envelopes; re-running
        # envelope_encrypt_bytes on them would produce a nested envelope (the
        # decryption layer would unwrap the outer shell and return another
        # ciphertext rather than plaintext). Copying verbatim preserves both
        # recipients. For age-wrapped artifacts, copy verbatim to avoid double-encryption.
        on_disk_bytes = raw_bytes
        encrypted_flag = True
    else:
        # Plaintext source — e.g. the meta.json / adapter_config.json carve-outs
        # the live store keeps plaintext, or any file under Security OFF. Copy
        # VERBATIM so the bundle mirrors each file's on-disk encryption state and
        # restore is a byte-faithful round-trip. Re-encrypting a plaintext
        # carve-out here would write back an unreadable meta.json on restore
        # (find_live_slot/read_manifest and PEFT expect plaintext). Sensitive
        # artifacts (weights, registries, speaker profiles) are already age
        # envelopes on disk and are copied verbatim by the branch above, so the
        # bundle preserves the live store's exact security posture either way.
        on_disk_bytes = raw_bytes
        encrypted_flag = False

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(on_disk_bytes)

    return {
        "path": "",  # caller fills relative path
        "content_sha256": content_sha256_bytes(on_disk_bytes),
        "encrypted": encrypted_flag,
        "size_bytes": len(on_disk_bytes),
    }


def write_bundle(
    *,
    config_path: Path,
    registry_path: Path,
    adapter_dirs: dict[str, Path],
    base_dir: Path,
    meta_fields: dict,
    adapter_scope: str = "live",
    live_registry_sha256: str = "",
    speaker_profiles_path: Path | None = None,
    candidate_config_path: Path | None = None,
) -> Path:
    """Capture a self-contained recovery-set bundle into a single slot directory.

    A bundle slot contains every artifact required to restore a working ParaMem
    instance: live server config, key-metadata registry, per-adapter weights,
    per-tier indexed-key registries, per-tier SimHash registries, interim adapter
    slots (when ``adapter_scope="live"``), and speaker profiles.  The bundle
    manifest (``bundle.meta.json``) indexes all captured files with their content
    hashes.

    The capture set mirrors exactly what ``MemoryStore.load_registries_from_disk``
    mounts for recall:

    - **Per enabled MAIN tier** (``episodic`` / ``semantic`` / ``procedural``):
      live weight slot via ``find_live_slot(<tier_dir>, hash)``, then the
      tier-root ``indexed_key_registry.json`` (fingerprints are stored inside
      this file's ``"simhash"`` map; ``simhash_registry.json`` no longer exists).
    - **Per INTERIM family** (``iter_interim_dirs(<adapter_base>)``), only when
      ``adapter_scope="live"``: live inner weight slot via
      ``find_live_slot(<interim_dir>, hash)`` (the ``<ts>/`` slot;
      ``checkpoint-*/`` scaffolding is naturally excluded because those dirs
      carry no matching ``meta.registry_sha256``), then the interim-dir
      ``indexed_key_registry.json``.
    - Shared: ``key_metadata.json``, ``speaker_profiles.json``, ``server.yaml``.
    - Optional (base-swap only): ``server.yaml.candidate`` — the candidate
      (new-target) config sidecar, included when ``candidate_config_path`` is
      supplied.  Captured only to serve as an operator retry-anchor after a
      rollback; **never restored** by ``restore_bundle`` (the
      ``startswith("config/")`` filter excludes it by construction).
    - Excluded: ``graph.json`` (RAM-only), ``checkpoint-*/``, ``in_training/``,
      ``bg_checkpoint/``, ``resume_state.json`` (training scaffolding),
      ``keyed_pairs.json`` (transient; regenerated from graph).

    Per-slot hashes: each captured adapter entry records the slot's **own**
    ``meta.registry_sha256``.  Main and interim slots carry different hashes;
    a single global hash cannot address both.

    Crash safety follows the same pattern as ``write()``:

    1. All files are written into ``.pending/<ts>/``.
    2. Every file is fsynced, then the pending directory entry is fsynced.
    3. A single ``os.rename()`` atomically promotes the pending directory.
    4. Parent directories are fsynced for rename durability.

    A crash at any step either leaves no slot or a ``.pending/<ts>/`` residue
    that ``sweep_orphan_pending()`` removes on startup.

    This function does **not** emit an ``ArtifactMeta`` sidecar — only
    ``bundle.meta.json`` is written.

    Parameters
    ----------
    config_path:
        Path to the live ``server.yaml`` (or ``server.yaml.enc``).
    registry_path:
        Path to ``key_metadata.json`` (the ESSENTIAL registry that ties
        weights to indexed keys).
    adapter_dirs:
        Mapping of adapter name → adapter-kind directory (e.g.
        ``{"episodic": Path("data/ha/adapters/episodic")}``) for every
        **enabled** adapter.  The function resolves the live main slot under
        each directory using ``live_registry_sha256``.  The adapter-base dir
        (parent of the episodic dir) is derived from the episodic entry to
        discover interim families via ``iter_interim_dirs``.
    base_dir:
        Bundle-kind backup directory (e.g. ``data/ha/backups/snapshot/``).
        Created with parents if absent.
    meta_fields:
        Caller-supplied metadata dict.  Required key: ``"tier"`` (str).
        Optional: ``"label"`` (str | None).
    adapter_scope:
        ``"live"`` (default) — capture the live-serving slot for each enabled
        main tier plus every interim family.  Under ``"live"`` the primary
        ``episodic`` recall may be satisfied by an interim slot when no
        finalized main slot exists (the normal production state while the
        episodic adapter is still accumulating sessions before first
        consolidation).
        ``"main"`` — capture only finalized main slots.  Fails loud for the
        ``episodic`` tier when it has no finalized main slot, with an
        actionable message to switch to ``"live"`` or run a full consolidation.
    live_registry_sha256:
        SHA-256 hex of the current ``key_metadata.json`` bytes.  Used by
        ``find_live_slot`` to select the correct adapter weight slot.  When
        empty string (fresh-install / no registry), empty-registry slots are
        matched.
    speaker_profiles_path:
        Optional path to ``speaker_profiles.json``.  When present and the
        file exists, it is included in the bundle.  When ``None`` or the file
        does not exist, the artifact is noted as absent in the manifest.
    candidate_config_path:
        Optional path to the candidate ``server.yaml`` for a base-model swap.
        When supplied, the file is copied verbatim as ``server.yaml.candidate``
        at the **top level** of the bundle slot (NOT under ``config/``) and
        hash-indexed in the manifest.  It is never restored by ``restore_bundle``
        — the ``startswith("config/")`` filter that selects the live config
        explicitly excludes this top-level sidecar.  Its purpose is to give the
        operator a retry anchor inside the immune bundle after a rollback.  The
        internal safety-bundle call site leaves this ``None`` — it snapshots the
        live state, not a migration candidate.  If supplied, the path must exist;
        a missing file raises ``BackupError`` rather than silently omitting it.

    Returns
    -------
    Path
        The promoted bundle slot directory (``<base_dir>/<ts>/``).

    Raises
    ------
    BackupError
        If the chosen ``adapter_scope`` resolves no weight slot for the primary
        ``episodic`` recall.  For ``adapter_scope="main"`` this happens when
        episodic has only an interim slot (use ``"live"`` or run a full
        consolidation).  For ``adapter_scope="live"`` this happens when no main
        or interim slot exists at all.  Non-episodic tiers with no slot are
        recorded as absent in the manifest; they do not trigger a failure.
    BackupError
        If a unique pending slot could not be allocated after 10 collision
        retries.
    BackupError
        If ``candidate_config_path`` is supplied but does not exist.
    OSError
        On any filesystem error.
    """
    # Import find_live_slot inside the function to avoid a potential import cycle:
    # manifest.py imports from paramem.backup.encryption (not backup.py), so there
    # is no actual cycle today.  The local import is kept as a guard — the comment
    # documents the intent so future refactors don't move the import without review.
    from paramem.adapters.manifest import find_live_slot

    def _tier_registry_sha256(tier_root: Path) -> str:
        """Hash of this tier's ``indexed_key_registry.json`` (decrypted) — the
        value ``find_live_slot`` matches against each slot's
        ``meta.registry_sha256``.

        Mirrors the server mount path (``app.py::_compute_tier_registry_sha256``
        for mains and the per-interim hash at the interim mount): every tier and
        interim slot is stamped with its OWN registry hash, so a single global
        ``live_registry_sha256`` cannot resolve all tiers (a full cycle leaves
        the main registry empty while interims carry their own). Returns ``""``
        when the registry is absent (matches empty-stamped slots).
        """
        reg = tier_root / "indexed_key_registry.json"
        if not reg.exists():
            return ""
        return hashlib.sha256(read_maybe_encrypted(reg)).hexdigest()

    base_dir = Path(base_dir)
    tier = meta_fields.get("tier", "manual")
    label = meta_fields.get("label")

    # Validate a requested candidate exists BEFORE allocating the pending slot,
    # so a set-but-missing candidate fails with zero on-disk residue (no orphan
    # .pending/<ts>/). The other write_bundle raises occur after allocation by
    # the function's existing pattern; this guard avoids adding to that set.
    if candidate_config_path is not None and not candidate_config_path.exists():
        raise BackupError(
            f"write_bundle: candidate_config_path does not exist: {candidate_config_path}"
        )

    # --- allocate pending slot ---
    pending_slot, timestamp = _promote_slot(base_dir)

    created_at = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    files_inventory: list[dict] = []
    adapters_record: dict[str, dict] = {}
    base_model_info: dict = {}

    # --- capture config ---
    if config_path.exists():
        dst = pending_slot / "config" / config_path.name
        entry = _copy_artifact(config_path, dst)
        entry["path"] = f"config/{config_path.name}"
        files_inventory.append(entry)

    # --- capture registry (key_metadata.json) ---
    if registry_path.exists():
        dst = pending_slot / "registry" / registry_path.name
        entry = _copy_artifact(registry_path, dst)
        entry["path"] = f"registry/{registry_path.name}"
        files_inventory.append(entry)

    # --- capture speaker_profiles.json ---
    if speaker_profiles_path is not None and speaker_profiles_path.exists():
        dst = pending_slot / "speaker_profiles.json"
        entry = _copy_artifact(speaker_profiles_path, dst)
        entry["path"] = "speaker_profiles.json"
        files_inventory.append(entry)

    # --- capture server.yaml.candidate (base-swap only) ---
    # Top-level placement is load-bearing: restore_bundle's Step-5c filter
    # matches startswith("config/") for the live config.  This sidecar lives
    # at the bundle root so it is hash-verified (Step 2 integrity check) but
    # never restored — the operator pulls it manually after a rollback.
    if candidate_config_path is not None:
        if not candidate_config_path.exists():
            raise BackupError(
                f"write_bundle: candidate_config_path does not exist: {candidate_config_path}"
            )
        dst = pending_slot / "server.yaml.candidate"
        entry = _copy_artifact(candidate_config_path, dst)
        entry["path"] = "server.yaml.candidate"
        files_inventory.append(entry)

    # --- helper: capture one adapter slot (main or interim) ---
    # _WEIGHT_FILES lists the three durable files in every slot directory.
    # Training scaffolding (checkpoint-*/, in_training/, bg_checkpoint/,
    # resume_state.json) is never copied: we only copy the named files below,
    # so scaffolding is excluded structurally.  _ADAPTER_EXCLUDED_PATTERNS and
    # _ADAPTER_EXCLUDED_DIR_PREFIXES document the exclusion contract explicitly.
    _WEIGHT_FILES = ("adapter_model.safetensors", "adapter_config.json", "meta.json")

    def _capture_adapter_slot(
        bundle_key: str,
        slot_path: Path,
        tier_root: Path,
        dst_prefix: str,
    ) -> None:
        """Capture one adapter slot (main or interim) into the pending directory.

        Reads the slot's own ``meta.json`` to record the slot's
        ``registry_sha256`` in the bundle manifest (main and interim slots
        carry different hashes).  Captures the per-tier
        ``indexed_key_registry.json`` from *tier_root* (the parent directory
        of the slot for main tiers, or the interim-family directory for
        interim slots).  Fingerprints live inside that file's ``"simhash"``
        map; ``simhash_registry.json`` no longer exists.

        Parameters
        ----------
        bundle_key:
            Key used in ``adapters_record`` (e.g. ``"episodic"`` or
            ``"episodic_interim_20260517T1200"``).
        slot_path:
            The timestamped slot directory (e.g. ``episodic/20260517-180431/``
            or ``episodic/interim_20260517T1200/20260517-180430/``).
        tier_root:
            Directory that holds the registries at its root — the
            adapter-kind dir for main tiers (``episodic/``), or the interim
            family dir (``interim_20260517T1200/``) for interim slots.
        dst_prefix:
            Relative path prefix inside the bundle slot directory for this
            adapter's files (e.g. ``"adapters/episodic"``).
        """
        # Read this slot's own meta.json for its registry_sha256 and key_count.
        slot_meta: dict = {}
        meta_src = slot_path / "meta.json"
        if meta_src.exists():
            try:
                slot_meta = _json.loads(meta_src.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise BackupError(
                    f"write_bundle: failed to read meta.json from {meta_src}: {exc}"
                ) from exc

        nonlocal base_model_info
        if not base_model_info and slot_meta.get("base_model"):
            bm = slot_meta["base_model"]
            base_model_info = {
                "repo": bm.get("repo", ""),
                "sha": bm.get("sha", ""),
                "hash": bm.get("hash", ""),
            }

        adapter_dst_dir = pending_slot / dst_prefix
        for fname in _WEIGHT_FILES:
            src = slot_path / fname
            if not src.exists():
                continue
            dst = adapter_dst_dir / fname
            entry = _copy_artifact(src, dst)
            entry["path"] = f"{dst_prefix}/{fname}"
            files_inventory.append(entry)

        # Capture per-tier indexed_key_registry.json (mirrors
        # MemoryStore.load_registries_from_disk which reads this file at
        # <tier_root>/indexed_key_registry.json for both main and interim tiers).
        indexed_key_src = tier_root / "indexed_key_registry.json"
        indexed_key_present = False
        if indexed_key_src.exists():
            dst = adapter_dst_dir / "indexed_key_registry.json"
            entry = _copy_artifact(indexed_key_src, dst)
            entry["path"] = f"{dst_prefix}/indexed_key_registry.json"
            files_inventory.append(entry)
            indexed_key_present = True

        # simhash_registry.json has been eliminated; simhashes now live in
        # indexed_key_registry.json under the "simhash" key.  No separate
        # simhash file to capture.

        adapters_record[bundle_key] = {
            "slot_source": str(slot_path),
            # Each slot's OWN registry_sha256 — main and interim hashes differ;
            # a single global live_registry_sha256 cannot address both.
            "registry_sha256": slot_meta.get("registry_sha256", ""),
            "key_count": slot_meta.get("key_count", "unknown"),
            "indexed_key_registry_present": indexed_key_present,
            "keyed_pairs_present": False,  # transient; regenerated from graph
        }

    # --- capture main tiers ---
    for adapter_name, adapter_kind_dir in adapter_dirs.items():
        adapter_kind_dir = Path(adapter_kind_dir)
        main_slot = find_live_slot(adapter_kind_dir, _tier_registry_sha256(adapter_kind_dir))

        if main_slot is None:
            if adapter_name == "episodic":
                # Episodic is the PRIMARY recall tier.  Under adapter_scope="main"
                # an interim-only episodic is a hard error (the caller must use
                # "live" or run a full consolidation first).  Under "live" we defer
                # the failure check until after the interim pass below — an interim
                # slot may satisfy the episodic requirement.
                if adapter_scope == "main":
                    raise BackupError(
                        "write_bundle: adapter_scope='main' but episodic has no finalized "
                        f"main slot in {adapter_kind_dir}. "
                        "Use adapter_scope='live' to capture the interim slot, or run a "
                        "full consolidation first."
                    )
                # Under "live": defer — interim pass will capture episodic interims.
                logger.debug(
                    "write_bundle: no main slot for episodic in %s; "
                    "will attempt interim capture (adapter_scope='live')",
                    adapter_kind_dir,
                )
            else:
                # Non-episodic tiers with no main slot are recorded as absent;
                # they do not fail the bundle.
                logger.debug(
                    "write_bundle: no main slot for %r in %s; recording absent",
                    adapter_name,
                    adapter_kind_dir,
                )
            continue

        _capture_adapter_slot(
            bundle_key=adapter_name,
            slot_path=main_slot,
            tier_root=adapter_kind_dir,  # registries live at the tier-kind dir root
            dst_prefix=f"adapters/{adapter_name}",
        )

    # --- capture interim families (only under adapter_scope="live") ---
    # iter_interim_dirs yields (adapter_name, interim_dir) for each
    # interim_<stamp>/ family found under the episodic dir.  It uses glob
    # pattern "interim_*" so checkpoint-*/, in_training/, and bg_checkpoint/
    # siblings are never yielded (_ADAPTER_EXCLUDED_DIR_PREFIXES contract).
    # find_live_slot on the interim_dir locates the <ts>/ slot inside it;
    # checkpoint-*/ dirs within the interim family are excluded naturally
    # because they carry no meta.json with a matching registry_sha256.
    if adapter_scope == "live" and adapter_dirs:
        # Derive the adapter base dir from the first entry (all adapter_kind dirs
        # share the same parent: data/ha/adapters/).
        adapter_base_dir = next(iter(adapter_dirs.values())).parent

        for interim_name, interim_dir in iter_interim_dirs(adapter_base_dir):
            interim_slot = find_live_slot(interim_dir, _tier_registry_sha256(interim_dir))
            if interim_slot is None:
                logger.debug(
                    "write_bundle: no live slot in interim family %s "
                    "(registry hash mismatch or empty family); skipping",
                    interim_dir,
                )
                continue

            _capture_adapter_slot(
                bundle_key=interim_name,
                slot_path=interim_slot,
                tier_root=interim_dir,  # interim registries live at the interim-family dir root
                dst_prefix=f"adapters/{interim_name}",
            )

    # --- fail-loud check for episodic primary recall ---
    # Episodic must be captured (as main OR as interim) when it is in adapter_dirs.
    # A bundle without episodic is not a valid recall-recovery set.
    if "episodic" in adapter_dirs:
        episodic_keys = {
            k for k in adapters_record if k == "episodic" or k.startswith("episodic_interim_")
        }
        if not episodic_keys:
            raise BackupError(
                "write_bundle: no live slot found for the primary episodic recall "
                f"(adapter_scope={adapter_scope!r}, "
                f"live_registry_sha256={live_registry_sha256!r}). "
                "Cannot write a self-contained recovery bundle without episodic weights. "
                "If consolidation has not run yet, use adapter_scope='live' so interim "
                "slots are included."
            )

    # --- write bundle.meta.json ---
    bundle_manifest = BundleManifest(
        bundle_schema_version=BUNDLE_SCHEMA_VERSION,
        created_at=created_at,
        tier=tier,
        label=label,
        live_registry_sha256=live_registry_sha256,
        base_model=base_model_info,
        files=files_inventory,
        adapters=adapters_record,
        excluded=list(_BUNDLE_EXCLUDED_DEFAULTS),
    )
    manifest_path = pending_slot / _BUNDLE_MANIFEST_FILENAME
    manifest_path.write_text(
        _json.dumps(bundle_manifest.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # --- fsync all captured files and the pending directory ---
    for fpath in pending_slot.rglob("*"):
        if fpath.is_file():
            with open(fpath, "rb") as fh:
                os.fsync(fh.fileno())

    dir_fd = os.open(str(pending_slot), os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)

    # --- atomic promotion ---
    slot_dir = base_dir / timestamp
    rename_pending_to_slot(pending_slot, slot_dir)

    # --- fsync parent directories for rename durability ---
    for parent in (slot_dir.parent, base_dir):
        _fsync_dir(parent)

    logger.debug(
        "backup.write_bundle: bundle slot written to %s (tier=%s, adapters=%s, files=%d)",
        slot_dir,
        tier,
        list(adapters_record.keys()),
        len(files_inventory),
    )
    return slot_dir


# ---------------------------------------------------------------------------
# prune()
# ---------------------------------------------------------------------------


def prune(
    kind: ArtifactKind,
    retention_policy: RetentionPolicy,
    *,
    base_dir: Path,
    live_slot: Path | None,
) -> PruneReport:
    """Enforce retention on *base_dir* for artifact *kind*.

    Enumerates slots in ``<base_dir>/`` (timestamped directories), sorts by
    timestamp descending (newest first), and deletes slots that exceed the
    ``keep`` count in *retention_policy*.

    The ``live_slot`` is **never deleted** — it is moved to
    ``PruneReport.skipped_live`` even when the retention policy would otherwise
    remove it.

    Slots with missing or corrupt sidecars are recorded in
    ``PruneReport.invalid`` but are *not* deleted.  Operator visibility and
    remediation are the responsibility of the operator-attention layer.

    **Idempotent** — calling ``prune`` multiple times with the same policy
    produces the same retained set.

    Retention policy keys
    ----------------------------------------
    - ``keep`` — ``int`` or ``"unlimited"``.  Maximum number of slots to
      retain.  When ``"unlimited"``, no count-based pruning is performed.
    - ``immunity_days`` — ``int | None``.  Slots whose ``timestamp`` is
      younger than this many days are immune from count-based pruning.
    - ``max_disk_gb`` — ``float | None``.  Per-tier disk cap.  When set,
      oldest-first slots are deleted until total kept size is under the cap.
      Disk pressure overrides immunity (spec rule 2 > rule 4).  The
      currently-live slot is never deleted even under disk pressure.

    Parameters
    ----------
    kind:
        Artifact kind being pruned (informational; used only for logging).
    retention_policy:
        Dict with at least ``{"keep": int | "unlimited"}``.
    base_dir:
        Per-kind backup directory (e.g. ``data/ha/backups/config/``).
    live_slot:
        Caller-supplied live slot path.  Never deleted.  ``None`` means no
        live slot protection (unusual; document intent at call site).

    Returns
    -------
    PruneReport
        Record of kept, deleted, skipped_live, and invalid slots.
    """
    base_dir = Path(base_dir)
    report = PruneReport()

    if not base_dir.exists():
        return report

    keep = retention_policy.get("keep", "unlimited")
    immunity_days = retention_policy.get("immunity_days")

    # Cutoff for immunity window
    if immunity_days is not None:
        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=immunity_days)
    else:
        cutoff = None

    # Enumerate candidate slot directories (exclude .pending)
    slots: list[Path] = []
    for entry in base_dir.iterdir():
        if entry.is_dir() and not entry.name.startswith("."):
            slots.append(entry)

    # Sort newest first (timestamp strings are lexicographically ordered)
    slots.sort(key=lambda p: p.name, reverse=True)

    # Validate each slot — populate invalid, segregate the rest
    valid_slots: list[Path] = []
    for slot in slots:
        try:
            meta = read_meta(slot)
        except Exception as exc:  # noqa: BLE001
            report.invalid.append((slot, str(exc)))
            continue
        # After a valid sidecar, verify the artifact file also exists.
        # A slot with a sidecar but no artifact is a partial write that
        # survived (e.g. sweep_orphan_pending missed it).  Treat as invalid
        # so operators are alerted; do NOT delete automatically.
        artifact_filename = _artifact_filename(meta.kind, meta.timestamp, meta.encrypted)
        expected_artifact = slot / artifact_filename
        if not expected_artifact.exists():
            report.invalid.append((slot, f"artifact file missing: {expected_artifact}"))
        else:
            valid_slots.append(slot)

    # Apply retention rules to valid slots
    retained_count = 0
    for slot in valid_slots:
        is_live = live_slot is not None and slot.resolve() == Path(live_slot).resolve()

        if is_live:
            report.skipped_live.append(slot)
            report.kept.append(slot)
            continue

        # Immunity check: if slot is younger than immunity cutoff, keep it.
        # Immune slots are exempt from the tier count: the keep limit applies only
        # to the non-immune tail (slots older than the 30-day window). Do NOT
        # increment retained_count — immunity must not starve the non-immune tail.
        if cutoff is not None:
            slot_ts = _parse_slot_timestamp(slot.name)
            if slot_ts is not None and slot_ts > cutoff:
                report.kept.append(slot)
                continue

        # Count-based check
        if keep == "unlimited":
            report.kept.append(slot)
            retained_count += 1
        elif retained_count < int(keep):
            report.kept.append(slot)
            retained_count += 1
        else:
            # Prune this slot
            try:
                shutil.rmtree(slot)
                report.deleted.append(slot)
                logger.debug("prune: removed %s slot %s", kind.value, slot)
            except OSError as exc:
                logger.warning("prune: could not remove %s: %s", slot, exc)
                report.kept.append(slot)

    # --- Per-tier disk-cap enforcement (spec retention rule 2) ---
    # Applied after the keep + immunity pass.  Oldest-first slots are deleted
    # until total size of kept slots is within max_disk_gb.  Disk pressure
    # overrides immunity (spec rule 2 > rule 4) but NEVER deletes live_slot.
    max_disk_gb = retention_policy.get("max_disk_gb")
    if max_disk_gb is not None:
        max_disk_bytes = int(max_disk_gb * 1024**3)

        def _slot_size(p: Path) -> int:
            """Return total byte size of all files in slot directory *p*."""
            return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())

        # Build ordered list of kept slots oldest-first (lowest timestamp first)
        # to find deletion candidates when over cap.
        live_resolved = Path(live_slot).resolve() if live_slot is not None else None
        candidates = sorted(
            [s for s in report.kept if s.resolve() != live_resolved],
            key=lambda p: p.name,  # ascending = oldest first
        )

        total_bytes = sum(_slot_size(s) for s in report.kept)

        for slot in candidates:
            if total_bytes <= max_disk_bytes:
                break
            slot_bytes = _slot_size(slot)
            try:
                shutil.rmtree(slot)
                report.kept.remove(slot)
                report.deleted.append(slot)
                total_bytes -= slot_bytes
                # Check whether this slot had immunity (was in skipped_live is
                # already guarded above; here we log if it was immune from
                # count-based pruning by being inside the immunity window).
                if cutoff is not None:
                    slot_ts = _parse_slot_timestamp(slot.name)
                    if slot_ts is not None and slot_ts > cutoff:
                        logger.warning(
                            "prune: deleted immune slot %s under max_disk_gb pressure", slot
                        )
                logger.debug(
                    "prune: removed %s slot %s under max_disk_gb=%.4f pressure",
                    kind.value,
                    slot,
                    max_disk_gb,
                )
            except OSError as exc:
                logger.warning("prune: could not remove %s under disk pressure: %s", slot, exc)

    return report


def _parse_slot_timestamp(name: str) -> datetime | None:
    """Parse a slot directory name ``YYYYMMDD-HHMMSSff`` into a UTC datetime.

    Returns ``None`` if the name does not match the expected format (so
    malformed slots are not accidentally protected by the immunity window).
    """
    try:
        # Slot name format: YYYYMMDD-HHMMSSff (17 chars: 8 date + 1 dash + 8 time+hundredths)
        if len(name) != 17 or name[8] != "-":
            return None
        dt = datetime.strptime(name[:15], "%Y%m%d-%H%M%S").replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, IndexError):
        return None


# ---------------------------------------------------------------------------
# restore_bundle()
# ---------------------------------------------------------------------------


def _atomic_write_file(src_bytes: bytes, dst: Path, mode: int = 0o600) -> None:
    """Write *src_bytes* to *dst* atomically via a temp-sibling + rename.

    Uses ``tempfile.mkstemp`` to generate a unique temp path (avoiding
    ``FileExistsError`` from a stale ``.restore-pending`` temp left by a prior
    crash between create and rename), then opens that path with ``O_WRONLY`` at
    *mode* permissions (default 0o600) to prevent plaintext exposure under the
    default umask.  Fsyncs the file and the parent directory for rename
    durability.

    Parameters
    ----------
    src_bytes:
        Bytes to write.
    dst:
        Target destination path.  The parent directory must already exist.
    mode:
        File mode (octal) for the temp file.  Default 0o600.
    """
    import tempfile

    dst.parent.mkdir(parents=True, exist_ok=True)
    # mkstemp creates a unique temp path — no stale-temp FileExistsError.
    # The returned fd is already open O_WRONLY at the default umask; we
    # re-open with the requested mode (0o600) by closing first and using
    # os.open so the mode is applied before any write.
    tmp_fd, tmp_str = tempfile.mkstemp(
        dir=str(dst.parent),
        prefix=dst.name + ".",
        suffix=".restore-pending",
    )
    os.close(tmp_fd)
    tmp = Path(tmp_str)
    # Re-open with the requested mode so the temp file is never world-readable.
    os.chmod(str(tmp), mode)
    fd = os.open(str(tmp), os.O_WRONLY, mode)
    try:
        os.write(fd, src_bytes)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.rename(str(tmp), str(dst))
    # fsync parent directory for rename durability.
    parent_fd = os.open(str(dst.parent), os.O_RDONLY)
    try:
        os.fsync(parent_fd)
    except OSError as exc:
        logger.warning("_atomic_write_file: parent fsync failed for %s: %s", dst, exc)
    finally:
        os.close(parent_fd)


def restore_bundle(
    bundle_slot_dir: Path,
    *,
    data_dir: Path,
    config_path: Path,
    restore_config: bool = False,
) -> RestoreResult:
    """Restore a self-contained ``snapshot_bundle`` slot into *data_dir*.

    Performs a crash-safe, pre-mutation-verified restore of a bundle produced
    by :func:`write_bundle`.  The sequence is:

    1. **Read + validate** ``bundle.meta.json`` → :class:`BundleManifest`.
       Reject if missing or forward-version (``BundleManifestError``).
    2. **Verify file hashes**: each entry in ``manifest.files`` must match the
       on-disk bundle file's SHA-256.  Fail loud on mismatch
       (``FingerprintMismatchError``) — no live mutation occurs.
    3. **Decrypt-probe**: confirm the age-encrypted metadata files decrypt with
       the CURRENT daily identity.  Raises ``RuntimeError`` (no key) or
       ``pyrage.DecryptError`` (stale/wrong key) **before** any mutation.
    4. **Safety bundle**: capture the CURRENT live state via
       :func:`write_bundle` into ``data_dir/backups/snapshot/`` (tier
       ``"manual"``, label ``"pre_restore_safety_<bundle_id>"``).  Skipped
       gracefully (``safety_slot=None``) when the live store has no episodic
       slot (fresh/empty target): a ``BackupError`` from the safety write is
       caught, logged, and restore continues.  All other errors abort.
    5. **Atomic restore** (adapter slots written; registry written LAST):

       - For each adapter in ``manifest.adapters``: create a fresh slot dir
         via ``_promote_slot`` (NOT the bundle's original timestamp) under the
         tier-root resolved by ``adapter_slot_root_for_name``.  Copy
         ``adapter_model.safetensors``, ``adapter_config.json``, ``meta.json``
         AS-IS.  Write the per-tier ``indexed_key_registry.json`` to the
         tier-root (where ``MemoryStore.load_registries_from_disk`` reads them;
         simhashes now live inside this file under the ``"simhash"`` key).
       - ``speaker_profiles.json`` → ``data_dir/speaker_profiles.json``
         (atomic temp+rename).
       - ``server.yaml`` → ONLY if ``restore_config=True``: atomic temp+rename
         to ``config_path``.
       - **LAST**: ``registry/key_metadata.json`` → ``data_dir/registry/
         key_metadata.json`` (atomic temp+rename).

       Registry-last crash invariant: a crash before the registry swap leaves
       the OLD registry live.  ``find_live_slot`` resolves the OLD slots →
       graceful (no half-restored live set).  The safety bundle is the
       documented rollback target when the operator wants to undo.

    5e. **Clean-slate sweep** (after 5d, inside the step-5 try): make
        ``data_dir/adapters/`` contain EXACTLY the bundle's adapters.  Removes
        orphan main tiers (whole tier absent from bundle), orphan interim
        families, stale slot dirs inside kept tiers, stale registries in
        episodic-as-interim tiers, and legacy top-level entries.  Orphan
        adapter removals are recorded in ``RestoreResult.pruned_orphans``;
        within-tier stale-slot cleanup is logged at INFO/DEBUG.

    6. Return :class:`RestoreResult` with ``restart_required=True`` — no hot
       VRAM swap (8 GB; mounted adapters are stale until restart).

    Parameters
    ----------
    bundle_slot_dir:
        Promoted bundle slot directory (e.g. ``data/ha/backups/snapshot/<ts>/``).
        Must contain a valid ``bundle.meta.json``.
    data_dir:
        Live data directory (e.g. ``data/ha/``).  Adapter slots, registries,
        and ``speaker_profiles.json`` are restored into this tree.  Can be a
        scratch directory for tests / dry runs.
    config_path:
        Path to the live ``server.yaml``.  Only written when
        ``restore_config=True``.
    restore_config:
        When ``True``, atomically restore the bundle's ``server.yaml`` to
        ``config_path``.  Default ``False`` — leave the live config untouched.

    Returns
    -------
    RestoreResult
        On success.

    Raises
    ------
    BundleManifestError
        If ``bundle.meta.json`` is missing, unreadable, or schema-mismatched
        (forward version).  **No mutation has occurred.**
    FingerprintMismatchError
        If any file listed in the bundle manifest does not match its stored
        content hash (corrupt bundle).  **No mutation has occurred.**
    RuntimeError
        If an age-encrypted metadata file cannot be decrypted because the
        daily identity is not loaded.  **No mutation has occurred.**
    BackupError
        If the bundle is structurally incomplete (no files for an adapter
        listed in ``manifest.adapters``).
    OSError
        On any filesystem error during the restore write phase.  The safety
        bundle path is preserved; the operator should restore from it.
    """
    from paramem.memory.interim_adapter import (  # noqa: PLC0415
        INTERIM_NAME_PREFIX as _INTERIM_NAME_PREFIX,
    )
    from paramem.memory.interim_adapter import (  # noqa: PLC0415
        adapter_slot_root_for_name,
    )

    bundle_slot_dir = Path(bundle_slot_dir)
    data_dir = Path(data_dir)
    config_path = Path(config_path)

    # -------------------------------------------------------------------------
    # Step 1: Read + validate bundle.meta.json
    # -------------------------------------------------------------------------
    manifest_path = bundle_slot_dir / _BUNDLE_MANIFEST_FILENAME
    if not manifest_path.exists():
        raise BundleManifestError(
            f"restore_bundle: bundle.meta.json not found in {bundle_slot_dir}"
        )
    try:
        raw = _json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BundleManifestError(
            f"restore_bundle: cannot read bundle.meta.json from {bundle_slot_dir}: {exc}"
        ) from exc
    manifest = BundleManifest.from_dict(raw)  # raises BundleManifestError on schema mismatch

    # -------------------------------------------------------------------------
    # Step 2: Verify file hashes — BEFORE any mutation
    # -------------------------------------------------------------------------
    for entry in manifest.files:
        rel_path = entry["path"]
        expected_sha = entry["content_sha256"]
        file_path = bundle_slot_dir / rel_path
        if not file_path.exists():
            raise FingerprintMismatchError(
                f"restore_bundle: bundle file missing: {file_path}. "
                "Bundle may be corrupt or incomplete."
            )
        actual_sha = content_sha256_bytes(file_path.read_bytes())
        if actual_sha != expected_sha:
            raise FingerprintMismatchError(
                f"restore_bundle: hash mismatch for {rel_path}: "
                f"manifest={expected_sha}, disk={actual_sha}. "
                "Bundle is corrupt — restore aborted without any live mutation."
            )

    # -------------------------------------------------------------------------
    # Step 3: Decrypt-probe encrypted metadata — BEFORE any mutation
    # -------------------------------------------------------------------------
    # Probe each file marked encrypted=True.  Weight blobs (.safetensors) are
    # copied AS-IS; only metadata files (registry, config, speaker_profiles,
    # simhash, indexed_key_registry) are probed because they are the ones the
    # server needs to READ post-restore.  Raises RuntimeError (no key) or
    # pyrage.DecryptError (stale/wrong key) — both surface as actionable errors
    # before any live mutation.
    for entry in manifest.files:
        if not entry.get("encrypted", False):
            continue
        # Skip weight blobs — copied verbatim; decrypt-validity for weights is
        # deferred to restart-time mount (server loads adapter from disk).
        rel_path = entry["path"]
        if rel_path.endswith(".safetensors"):
            continue
        file_path = bundle_slot_dir / rel_path
        # read_maybe_encrypted raises RuntimeError or pyrage.DecryptError on failure.
        read_maybe_encrypted(file_path)  # decrypt-probe only; discard result

    # -------------------------------------------------------------------------
    # Step 4: Safety bundle of current live state
    # -------------------------------------------------------------------------
    # Capture the current live recovery set into a pre-restore snapshot bundle.
    # Skipped gracefully when the live store has no episodic slot (fresh/empty
    # target) — a missing-episodic BackupError from write_bundle must NOT abort
    # the restore (it means we're restoring into a clean slate).
    safety_slot: Path | None = None
    bundle_id = bundle_slot_dir.name
    safety_label = f"pre_restore_safety_{bundle_id}"

    # Derive the adapter dirs from data_dir for the safety bundle.
    adapters_base = data_dir / "adapters"
    safety_adapter_dirs: dict[str, Path] = {}
    for tier_name in ("episodic", "semantic", "procedural"):
        tier_dir = adapters_base / tier_name
        if tier_dir.is_dir():
            safety_adapter_dirs[tier_name] = tier_dir

    # Registry path in the LIVE data_dir (not the bundle's registry).
    live_registry_path = data_dir / "registry" / "key_metadata.json"
    live_config_path_for_safety = config_path
    live_speaker_profiles = data_dir / "speaker_profiles.json"

    if safety_adapter_dirs:
        try:
            safety_slot = write_bundle(
                config_path=live_config_path_for_safety,
                registry_path=live_registry_path,
                adapter_dirs=safety_adapter_dirs,
                base_dir=data_dir / "backups" / "snapshot",
                meta_fields={"tier": "manual", "label": safety_label},
                speaker_profiles_path=(
                    live_speaker_profiles if live_speaker_profiles.exists() else None
                ),
            )
            logger.info("restore_bundle: safety bundle written to %s", safety_slot)
        except BackupError as exc:
            # No episodic slot in the live store (fresh target or mid-consolidation).
            # Log + continue: the safety bundle is best-effort on a fresh target.
            logger.warning(
                "restore_bundle: safety bundle skipped (live store has no episodic slot): %s",
                exc,
            )
            safety_slot = None
    else:
        logger.debug(
            "restore_bundle: safety bundle skipped — no adapter dirs in live store (fresh target)"
        )

    # -------------------------------------------------------------------------
    # Step 5: Atomic restore — adapter slots + registries + speaker_profiles
    #         registry (key_metadata.json) written LAST (crash-safety invariant)
    # -------------------------------------------------------------------------
    #
    # Registry-last crash invariant:
    #   A crash before the registry swap leaves the OLD registry live.
    #   find_live_slot resolves the OLD slots → graceful (no half-restored live
    #   set).  The NEW adapter slots are latent + harmless (no slot meta matches
    #   the old registry hash; they will be swept or ignored on next prune).
    #   A crash AFTER the registry swap but before banner leaves the restored
    #   set fully live — the desired end state; only the banner is missing.
    #
    # Safety-slot surface: the entire step-5 write phase is wrapped so that
    # any exception logs the safety_slot path at ERROR before propagating.
    # The operator can then use the safety bundle's backup_id to recover.

    restored_adapters: list[str] = []
    restored_config = False
    pruned_orphans: list[dict] = []

    # Tracks precisely what step 5a writes so the sweep can keep only those
    # paths and remove everything else.
    #
    # restored_main_slots: tier name → new slot dir (for main-tier adapters:
    #     episodic / semantic / procedural that are finalized in the bundle).
    # restored_interim_slots: interim adapter name → (interim_family_dir, new_slot_dir).
    restored_main_slots: dict[str, Path] = {}
    restored_interim_slots: dict[str, tuple[Path, Path]] = {}

    try:
        # 5a. Adapter slots — each gets a NEW timestamped dir (not the bundle's ts)
        for adapter_name, adapter_record in manifest.adapters.items():
            adapter_bundle_prefix = f"adapters/{adapter_name}"

            # Determine which files belong to this adapter in the bundle.
            # Weight files live under adapters/<name>/; registries also under adapters/<name>/.
            adapter_files = [
                entry
                for entry in manifest.files
                if entry["path"].startswith(f"{adapter_bundle_prefix}/")
            ]
            if not adapter_files:
                logger.warning(
                    "restore_bundle: adapter %r has no files in bundle; skipping", adapter_name
                )
                continue

            # Resolve the tier-root for this adapter name.
            tier_root = adapter_slot_root_for_name(data_dir / "adapters", adapter_name)

            # Allocate a fresh slot dir under the tier-root (registry-last constraint:
            # the new slot carries the bundle's meta.json with its registry_sha256;
            # find_live_slot will resolve it as live once the registry is swapped).
            new_slot_pending, new_ts = _promote_slot(tier_root)

            # Copy the three weight files (adapter_model.safetensors, adapter_config.json,
            # meta.json) into the new slot.
            _SLOT_WEIGHT_FILES = {
                "adapter_model.safetensors",
                "adapter_config.json",
                "meta.json",
            }
            for entry in adapter_files:
                fname = Path(entry["path"]).name
                if fname not in _SLOT_WEIGHT_FILES:
                    continue
                src = bundle_slot_dir / entry["path"]
                dst = new_slot_pending / fname
                dst.write_bytes(src.read_bytes())

            # fsync slot files and the pending dir, then promote.
            for fpath in new_slot_pending.iterdir():
                if fpath.is_file():
                    with open(fpath, "rb") as fh:
                        os.fsync(fh.fileno())
            slot_fd = os.open(str(new_slot_pending), os.O_RDONLY)
            try:
                os.fsync(slot_fd)
            finally:
                os.close(slot_fd)

            new_slot_dir = tier_root / new_ts
            rename_pending_to_slot(new_slot_pending, new_slot_dir)
            _fsync_dir(tier_root)

            # Write per-tier registries to the tier-root (where MemoryStore reads them).
            # simhash_registry.json has been eliminated — simhashes now live inside
            # indexed_key_registry.json under the "simhash" key.
            _TIER_REGISTRY_FILES = {"indexed_key_registry.json"}
            for entry in adapter_files:
                fname = Path(entry["path"]).name
                if fname not in _TIER_REGISTRY_FILES:
                    continue
                src = bundle_slot_dir / entry["path"]
                src_bytes = src.read_bytes()
                dst = tier_root / fname
                _atomic_write_file(src_bytes, dst)

            # Record the new slot dir for the clean-slate sweep (part A).
            if adapter_name.startswith(_INTERIM_NAME_PREFIX):
                # Interim adapters: tier_root IS the interim family dir
                # (adapter_slot_root_for_name returns <adapters>/episodic/interim_<stamp>/).
                restored_interim_slots[adapter_name] = (tier_root, new_slot_dir)
            else:
                restored_main_slots[adapter_name] = new_slot_dir

            logger.debug(
                "restore_bundle: adapter %r restored to new slot %s", adapter_name, new_slot_dir
            )
            restored_adapters.append(adapter_name)

        # 5b. speaker_profiles.json
        speaker_file_entries = [
            entry for entry in manifest.files if entry["path"] == "speaker_profiles.json"
        ]
        if speaker_file_entries:
            src_bytes = (bundle_slot_dir / "speaker_profiles.json").read_bytes()
            _atomic_write_file(src_bytes, data_dir / "speaker_profiles.json")
            logger.debug("restore_bundle: speaker_profiles.json restored")

        # 5c. server.yaml (only when restore_config=True)
        # NOTE: a top-level "server.yaml.candidate" sidecar (present in pre_base_swap
        # bundles) is intentionally hash-verified by Step 2 above but NOT restored here.
        # It is the operator's retry anchor inside the immune bundle after a rollback.
        # The startswith("config/") filter below excludes it by construction — do not
        # "helpfully" add it to config_entries or restore it to the live config path.
        if restore_config:
            config_entries = [
                entry for entry in manifest.files if entry["path"].startswith("config/")
            ]
            if config_entries:
                # Read the bundle's config (may be encrypted; read_maybe_encrypted decrypts).
                config_src = bundle_slot_dir / config_entries[0]["path"]
                config_bytes = read_maybe_encrypted(config_src)
                _atomic_write_file(config_bytes, config_path)
                restored_config = True
                logger.debug("restore_bundle: server.yaml restored to %s", config_path)

        # 5d. LAST: registry/key_metadata.json
        # This is the crash-safety sentinel.  Writing this last ensures
        # find_live_slot resolves the restored adapter slots as live by construction.
        # A crash before this step leaves the old registry live — old slots remain
        # authoritative; the new (latent) slots are harmless.
        registry_entries = [
            entry for entry in manifest.files if entry["path"].startswith("registry/")
        ]
        if registry_entries:
            registry_src = bundle_slot_dir / registry_entries[0]["path"]
            # Byte-faithful copy — preserve the registry's on-disk encryption state
            # (key_metadata.json is in infra_paths and is age-encrypted under
            # Security ON).  Do NOT decrypt-then-write: writing it plaintext while
            # the per-tier registries / weights / speaker_profiles stay encrypted
            # produces a mixed infra state that assert_mode_consistency refuses to
            # boot, and leaks speaker_id at rest.  Decryptability was already
            # validated by the Step-3 decrypt-probe; the live reader
            # (_load_key_metadata) uses read_maybe_encrypted so an encrypted file
            # loads fine.
            registry_bytes = registry_src.read_bytes()
            dst_registry = data_dir / "registry" / "key_metadata.json"
            dst_registry.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write_file(registry_bytes, dst_registry)
            logger.debug(
                "restore_bundle: key_metadata.json written (registry-last invariant satisfied)"
            )

        # 5e. Clean-slate sweep — make adapters/ contain EXACTLY the bundle set.
        #
        # Placement: AFTER the registry-last swap (5d).  A crash BEFORE the registry
        # swap must leave the fully-old graceful state — this sweep must not have run
        # yet (preserves the registry-last invariant documented at backup.py:1466-1473).
        # After the swap, the new registry references only bundle adapters, so removing
        # stale on-disk content cannot create a registry-references-missing-slot
        # inconsistency.
        #
        # The sweep makes <data_dir>/adapters/ exactly equal to the restored set by:
        #  1. Removing whole main tiers absent from the bundle (orphan main tiers).
        #     Reported in pruned_orphans (adapter-level: "kind"="main").
        #  2. Removing orphan interim families not in the bundle.
        #     Reported in pruned_orphans (adapter-level: "kind"="interim").
        #  3. Within each kept tier: removing all children except the freshly-written
        #     slot dir (from step 5a), the tier-level indexed_key_registry.json, and
        #     restored interim family dirs.  This removes stale old slot dirs,
        #     in_training/bg_checkpoint/checkpoint-* scratch, stale registries, and
        #     stale simhash sidecars.  Logged at INFO/DEBUG — not in pruned_orphans.
        #  4. Within each kept interim family: removing all children except the new
        #     slot dir and indexed_key_registry.json.
        #  5. Removing any other top-level entry under adapters/ (legacy flat
        #     episodic_interim_* dirs at the adapters root, stray files), EXCEPT
        #     durable infra files at the adapters root (e.g. post_session_queue.json).
        #
        # Critical: never delete a slot that step 5a just wrote — every rmtree/unlink
        # is guarded against the tracked paths in restored_main_slots /
        # restored_interim_slots.
        #
        # pruned_orphans records only ADAPTER-LEVEL removals (whole orphan tiers /
        # orphan interim families absent from the bundle).  Routine within-tier stale-
        # slot hygiene is logged at INFO/DEBUG but NOT listed in pruned_orphans.
        #
        # The step-4 safety bundle captures the tier set (episodic/semantic/procedural
        # + their registries + speaker_profiles), so tier-level restore is reversible.
        # Scratch dirs and re-creatable state are not captured by the safety bundle.
        from paramem.backup.encryption import infra_paths  # noqa: PLC0415
        from paramem.training.key_registry import KeyRegistry  # noqa: PLC0415 — local import guard

        # Infra files that live directly at the adapters root — these are durable,
        # restart-replayed files that must never be removed by the sweep.
        # post_session_queue.json is the canonical example: it is NOT captured by
        # the step-4 safety bundle, so deleting it is irreversible data loss.
        # Reuse infra_paths() — single source of truth; future additions are
        # auto-protected without touching this code.
        adapters_root_infra = {p for p in infra_paths(data_dir) if p.parent == adapters_base}

        def _count_active_keys(reg_path: Path) -> int:
            """Return active key count from *reg_path*; 0 on any error.

            The except clause is intentional boundary handling: a corrupt or
            missing registry must not abort the restore.
            """
            try:
                if reg_path.exists():
                    reg = KeyRegistry.load(reg_path)
                    return len(reg.list_active())
            except Exception:  # noqa: BLE001 — count failure must not abort restore
                pass
            return 0

        # Precompute per-tier interim families that belong to the bundle so that
        # the tier-level sweep below can identify which interim_* dirs to keep.
        # Maps tier_name → set of kept interim family dirs.
        _kept_interim_family_dirs: dict[str, set[Path]] = {}
        for _iname, (_ifam, _islot) in restored_interim_slots.items():
            # _ifam is e.g. <adapters>/episodic/interim_<stamp>/
            _tier_name = _ifam.parent.name  # "episodic"
            _kept_interim_family_dirs.setdefault(_tier_name, set()).add(_ifam)

        if adapters_base.is_dir():
            for top_entry in list(adapters_base.iterdir()):
                if top_entry.name.startswith("."):
                    continue  # skip .pending and other hidden entries

                if top_entry.name in ("episodic", "semantic", "procedural"):
                    tier = top_entry.name
                    tier_dir = top_entry
                    keep_main = tier in restored_main_slots
                    interims_here = _kept_interim_family_dirs.get(tier, set())

                    if not keep_main and not interims_here:
                        # Whole tier is absent from the bundle — orphan main tier.
                        active_keys = _count_active_keys(tier_dir / "indexed_key_registry.json")
                        shutil.rmtree(tier_dir)
                        pruned_orphans.append(
                            {"name": tier, "kind": "main", "active_keys": active_keys}
                        )
                        logger.warning(
                            "restore_bundle: pruned orphan main tier %r "
                            "(kind=main, %d active keys) "
                            "— not present in bundle recovery set. "
                            "Safety bundle at %s can restore it if needed.",
                            tier,
                            active_keys,
                            safety_slot,
                        )
                        continue

                    # Tier dir is kept (has finalized main weights OR hosts bundle interims).
                    # Clean its children to exactly: kept main slot dir (if keep_main) +
                    # indexed_key_registry.json (if keep_main) + restored interim family dirs.
                    # Everything else (stale slot dirs, scratch dirs, orphan interim families,
                    # stale registries/simhash sidecars) is removed.
                    kept_main_slot = restored_main_slots.get(tier)

                    for child in list(tier_dir.iterdir()):
                        if child.name.startswith("."):
                            continue  # skip .pending

                        # A restored interim family dir is always kept.
                        if child in interims_here:
                            # Recurse one level into the interim family: keep only its
                            # new slot dir and indexed_key_registry.json; remove everything
                            # else (stale slots, scratch, stale registry).
                            _islot_for_fam = next(
                                slot
                                for (_ifam2, slot) in restored_interim_slots.values()
                                if _ifam2 == child
                            )
                            for fam_child in list(child.iterdir()):
                                if fam_child.name.startswith("."):
                                    continue
                                if fam_child == _islot_for_fam:
                                    continue  # the freshly-written slot: keep
                                if fam_child.name == "indexed_key_registry.json":
                                    continue  # freshly-written registry: keep
                                if fam_child.is_dir():
                                    logger.info(
                                        "restore_bundle: removing stale child %s "
                                        "from restored interim family %s",
                                        fam_child.name,
                                        child.name,
                                    )
                                    shutil.rmtree(fam_child)
                                else:
                                    logger.info(
                                        "restore_bundle: removing stale file %s "
                                        "from restored interim family %s",
                                        fam_child.name,
                                        child.name,
                                    )
                                    fam_child.unlink()
                            continue  # done with this interim family

                        # Orphan interim family: in the tier dir but not in the bundle.
                        if child.name.startswith("interim_") and child.is_dir():
                            # Derive the adapter name for this interim family using
                            # the current tier name — interims only exist under episodic/
                            # today, but deriving from tier makes this robust if a future
                            # tier hosts interim families.
                            _orphan_stamp = child.name[len("interim_") :]
                            _orphan_iname = f"{tier}_interim_{_orphan_stamp}"
                            active_keys = _count_active_keys(child / "indexed_key_registry.json")
                            shutil.rmtree(child)
                            pruned_orphans.append(
                                {
                                    "name": _orphan_iname,
                                    "kind": "interim",
                                    "active_keys": active_keys,
                                }
                            )
                            logger.warning(
                                "restore_bundle: pruned orphan interim %r "
                                "(kind=interim, %d active keys) "
                                "— not present in bundle recovery set. "
                                "Safety bundle at %s can restore it if needed.",
                                _orphan_iname,
                                active_keys,
                                safety_slot,
                            )
                            continue

                        # indexed_key_registry.json at the tier root is kept
                        # ONLY when this tier has a finalized main adapter in the bundle
                        # (keep_main=True).  When keep_main is False (episodic-as-interim
                        # case: bundle has episodic_interim_* but no main episodic), a
                        # stale tier-level registry must be removed — otherwise it would
                        # be mounted as a stale main-episodic at boot.
                        if child.name == "indexed_key_registry.json" and not child.is_dir():
                            if keep_main:
                                continue  # freshly-written tier registry: keep
                            # Episodic-as-interim: stale main registry — remove.
                            logger.info(
                                "restore_bundle: removing stale main-tier registry %s "
                                "(tier %r has no finalized main adapter in bundle)",
                                child,
                                tier,
                            )
                            child.unlink()
                            continue

                        # Freshly-written main slot: keep.
                        if keep_main and child == kept_main_slot:
                            continue

                        # Everything else in the tier dir is stale:
                        # old slot dirs, in_training/, bg_checkpoint/, checkpoint-* scratch,
                        # stale simhash sidecars, or any other leftover file.
                        if child.is_dir():
                            logger.info(
                                "restore_bundle: removing stale tier child %s/%s",
                                tier,
                                child.name,
                            )
                            shutil.rmtree(child)
                        else:
                            logger.info(
                                "restore_bundle: removing stale tier file %s/%s",
                                tier,
                                child.name,
                            )
                            child.unlink()

                else:
                    # Top-level entry under adapters/ that is not a recognised main tier.
                    # This covers legacy flat episodic_interim_* dirs (pre-hierarchy-refactor
                    # layout) and stray files.  Remove unconditionally — the bundle never
                    # captures these paths and they must not be mounted at boot.
                    #
                    # Exception: durable infra files at the adapters root
                    # (e.g. post_session_queue.json) are restart-replayed and are NOT
                    # captured by the step-4 safety bundle.  Skip them — deleting them
                    # would be irreversible data loss of queued sessions.
                    if top_entry in adapters_root_infra:
                        continue
                    if top_entry.is_dir():
                        logger.info(
                            "restore_bundle: removing legacy/stray top-level entry adapters/%s",
                            top_entry.name,
                        )
                        shutil.rmtree(top_entry)
                    else:
                        logger.info(
                            "restore_bundle: removing stray file adapters/%s",
                            top_entry.name,
                        )
                        top_entry.unlink()

    except Exception as _step5_exc:
        # Wrap in RestoreAbortedError so callers (e.g. the /backup/restore
        # HTTP handler) can surface the safety_slot path in the error response.
        # Also log at ERROR so the path is in the server log regardless of how
        # the caller handles the exception.
        if safety_slot is not None:
            logger.error(
                "restore_bundle: FAILED during atomic restore phase. "
                "Safety bundle captured before mutation: %s — "
                "use this backup_id to recover the live store. Cause: %s",
                safety_slot,
                _step5_exc,
            )
        raise RestoreAbortedError(
            f"restore_bundle failed during atomic restore phase: {_step5_exc}. "
            + (
                f"Safety bundle at {safety_slot} — use its backup_id to recover."
                if safety_slot is not None
                else "No safety bundle was captured (fresh target)."
            ),
            safety_slot=safety_slot,
            cause=_step5_exc,
        ) from _step5_exc

    logger.info(
        "restore_bundle: complete — adapters=%s, safety_slot=%s, restore_config=%s",
        restored_adapters,
        safety_slot,
        restored_config,
    )

    return RestoreResult(
        restored_adapters=restored_adapters,
        safety_slot=safety_slot,
        restart_required=True,
        restored_config=restored_config,
        pruned_orphans=pruned_orphans,
    )
