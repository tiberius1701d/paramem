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

Content-hash rule (Resolved Decision 29)
-----------------------------------------
``content_sha256`` in the sidecar is the SHA-256 of the raw bytes *as written
to disk*.  For encrypted artifacts this is the hash of the ciphertext; for
plaintext artifacts it is the hash of the source bytes.  Whitespace changes
and key-order changes in YAML/JSON source data are visible as hash changes.
No canonicalization is applied.
"""

from __future__ import annotations

import logging
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

from paramem.backup.atomic import rename_pending_to_slot
from paramem.backup.atomic import sweep_orphan_pending as _sweep_pending
from paramem.backup.encryption import (
    MASTER_KEY_ENV_VAR,
    SecurityBackupsConfig,
    current_key_fingerprint,
    decrypt_bytes,
    encrypt_bytes,
    master_key_loaded,
    resolve_policy,
    should_encrypt,
)
from paramem.backup.hashing import (
    content_sha256_bytes,
)
from paramem.backup.meta import read_meta, verify_fingerprint, write_meta
from paramem.backup.types import (
    SCHEMA_VERSION,
    ArtifactKind,
    ArtifactMeta,
    BackupError,
    EncryptAtRest,
    FatalConfigError,
    FingerprintMismatchError,
    PruneReport,
    RetentionPolicy,
)

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
# write()
# ---------------------------------------------------------------------------


def write(
    kind: ArtifactKind,
    source: bytes | Path,
    meta_fields: dict,
    *,
    base_dir: Path,
    security_config: SecurityBackupsConfig,
) -> Path:
    """Write an artifact + sidecar into a new timestamped slot directory.

    Sequence (crash-safe):
    1. Resolve encryption policy and determine whether to encrypt.
    2. Create ``.pending/<ts>/`` inside *base_dir*.
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
        ``"label"`` (str | None), any future kind-specific extension fields
        (ignored by Slice 1).
    base_dir:
        Per-kind backup directory (e.g. ``data/ha/backups/config/``).
        Created with parents if absent.
    security_config:
        Encryption policy configuration.

    Returns
    -------
    Path
        The promoted slot directory (``<base_dir>/<ts>/``).

    Raises
    ------
    FatalConfigError
        If ``encrypt_at_rest=always`` and no key is available.
    BackupError
        If a unique pending slot could not be allocated after 10 collision
        retries (pathological — frozen clock or extreme write rate).
    OSError
        On any filesystem error.
    """
    # --- resolve payload bytes ---
    payload: bytes = source if isinstance(source, bytes) else Path(source).read_bytes()

    # --- resolve encryption policy ---
    policy = resolve_policy(kind, security_config)
    key_loaded = master_key_loaded()

    if policy is EncryptAtRest.ALWAYS and not key_loaded:
        raise FatalConfigError(
            f"Cannot write {kind.value} artifact: encrypt_at_rest=always "
            f"but {MASTER_KEY_ENV_VAR} is not set"
        )

    do_encrypt = should_encrypt(policy, key_loaded)

    if do_encrypt:
        on_disk_bytes = encrypt_bytes(payload)
    else:
        on_disk_bytes = payload

    # --- compute content hash (of bytes as written — Resolved Decision 29) ---
    hash_hex = content_sha256_bytes(on_disk_bytes)

    # --- key fingerprint ---
    key_fp = current_key_fingerprint() if do_encrypt else None

    # --- timestamp + filenames (with collision retry) ---
    encrypted_flag = do_encrypt
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    pending_root = base_dir / _PENDING_DIR_NAME
    pending_root.mkdir(exist_ok=True)

    _MAX_COLLISION_RETRIES = 10
    timestamp: str | None = None
    pending_slot: Path | None = None
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
            timestamp = _ts_candidate
            pending_slot = _candidate_pending
            break
        except FileExistsError:
            # Bump by one hundredth of a second and retry
            _ts_now = _ts_now + timedelta(microseconds=10_000)

    if pending_slot is None or timestamp is None:
        raise BackupError(
            "could not allocate unique pending slot after "
            f"{_MAX_COLLISION_RETRIES} attempts — clock skew or extreme write rate"
        )

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
        encrypt_at_rest=policy,
        key_fingerprint=key_fp,
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
        If the artifact is encrypted but ``PARAMEM_MASTER_KEY`` is not set
        in the environment.
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

    # 5. Decrypt if needed
    if meta.encrypted:
        plaintext = decrypt_bytes(raw_bytes)
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
    remediation are Slice 5/6 responsibilities.

    **Idempotent** — calling ``prune`` multiple times with the same policy
    produces the same retained set.

    Retention policy keys (Slice 1 honours)
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
        # Immune slots are exempt from the tier count (spec §retention policy:
        # "keep: 10 applies only to the >30-day tail"). Do NOT increment
        # retained_count — immunity must not starve the non-immune tail.
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
