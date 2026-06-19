"""Backup slot enumeration for crash-recovery and /migration/confirm orphan detection.

Public surface
--------------
- ``enumerate_backups(base_dir, kind=None)`` — scan slot directories under
  ``data/ha/backups/``, skip ``.pending/`` residue, return newest-first list
  of ``BackupRecord`` instances.

Bundle-aware enumeration
------------------------
Bundle slots (``snapshot_bundle`` kind) store a ``bundle.meta.json`` top-level
manifest instead of the per-artifact ``<kind>-<ts>.meta.json`` sidecar that
``read_meta`` expects.  ``enumerate_backups`` detects the presence of
``bundle.meta.json`` *before* calling ``read_meta`` so that bundle slots are
never mis-parsed as invalid per-artifact slots.

For bundle slots, a synthetic ``ArtifactMeta`` is constructed from the bundle
manifest's fields so that the returned ``BackupRecord`` is structurally
identical to a per-artifact record and all downstream callers (listing,
prune, size reporting) handle bundle slots without modification.
"""

from __future__ import annotations

import json as _json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from paramem.backup.backup import _parse_slot_timestamp
from paramem.backup.meta import read_meta
from paramem.backup.types import (
    SCHEMA_VERSION,
    ArtifactKind,
    ArtifactMeta,
    BundleManifest,
    BundleManifestError,
    MetaSchemaError,
)

logger = logging.getLogger(__name__)

_BUNDLE_MANIFEST_FILENAME = "bundle.meta.json"


@dataclass(frozen=True)
class BackupRecord:
    """One slot enumerated from data/ha/backups/<kind>/.

    Attributes
    ----------
    slot_dir:
        Absolute path to the slot directory.
    kind:
        Artifact kind, from meta.json or bundle.meta.json.
    timestamp:
        YYYYMMDD-HHMMSSff string (slot directory name).
    created_at:
        UTC datetime parsed from ``timestamp``.
    content_sha256:
        SHA-256 of the artifact bytes as stored on disk (ciphertext when
        encrypted, plaintext otherwise).  For bundle slots this is the
        SHA-256 of the ``bundle.meta.json`` bytes (the index of the bundle).
    pre_trial_hash:
        Optional SHA-256 of the live config at /migration/confirm step 2
        time.  Present only on pre-migration backups written by 3b.2+.
        Always ``None`` for bundle slots.
    label:
        Optional operator-supplied annotation from the sidecar.
    meta:
        Full ``ArtifactMeta`` sidecar for callers that need additional fields.
        For bundle slots this is a synthetic ``ArtifactMeta`` derived from the
        ``BundleManifest`` so that callers can access ``meta.tier``.
    is_bundle:
        ``True`` when the slot is a ``snapshot_bundle`` with a
        ``bundle.meta.json`` manifest.  Callers that need bundle-specific
        fields (``live_registry_sha256``, ``adapters``, ``files``) should
        read the ``bundle.meta.json`` directly from ``slot_dir``.
    """

    slot_dir: Path
    kind: ArtifactKind
    timestamp: str
    created_at: datetime
    content_sha256: str
    pre_trial_hash: str | None
    label: str | None
    meta: ArtifactMeta
    is_bundle: bool = False


def _read_bundle_record(slot: Path) -> BackupRecord | None:
    """Attempt to read a bundle slot from *slot* and return a BackupRecord.

    Reads ``bundle.meta.json`` from *slot*, validates the schema version, and
    constructs a synthetic ``ArtifactMeta`` so the returned ``BackupRecord``
    is structurally compatible with per-artifact records for listing and prune.

    Returns ``None`` (logging WARN) if the manifest is missing, unparseable,
    or schema-mismatched.

    Parameters
    ----------
    slot:
        Slot directory containing ``bundle.meta.json``.

    Returns
    -------
    BackupRecord | None
    """
    manifest_path = slot / _BUNDLE_MANIFEST_FILENAME
    try:
        raw = _json.loads(manifest_path.read_text(encoding="utf-8"))
        bundle = BundleManifest.from_dict(raw)
    except BundleManifestError as exc:
        logger.warning(
            "enumerate_backups: skipping bundle slot %s — manifest invalid: %s",
            slot,
            exc,
        )
        return None
    except (OSError, ValueError) as exc:
        logger.warning(
            "enumerate_backups: skipping bundle slot %s — cannot read manifest: %s",
            slot,
            exc,
        )
        return None

    # Use the slot directory name as the timestamp (canonical slot naming).
    ts = slot.name
    created_at = _parse_slot_timestamp(ts)
    if created_at is None:
        logger.warning(
            "enumerate_backups: skipping bundle slot %s — cannot parse timestamp %r",
            slot,
            ts,
        )
        return None

    # Build a synthetic ArtifactMeta so downstream callers can access .meta.tier
    # without needing to be bundle-aware.  content_sha256 is the manifest hash.
    manifest_bytes = manifest_path.read_bytes()
    import hashlib as _hashlib

    manifest_sha256 = _hashlib.sha256(manifest_bytes).hexdigest()
    synthetic_meta = ArtifactMeta(
        schema_version=SCHEMA_VERSION,
        kind=ArtifactKind.SNAPSHOT_BUNDLE,
        timestamp=ts,
        content_sha256=manifest_sha256,
        size_bytes=len(manifest_bytes),
        encrypted=False,
        tier=bundle.tier,
        label=bundle.label,
    )

    return BackupRecord(
        slot_dir=slot.resolve(),
        kind=ArtifactKind.SNAPSHOT_BUNDLE,
        timestamp=ts,
        created_at=created_at.replace(tzinfo=timezone.utc),
        content_sha256=manifest_sha256,
        pre_trial_hash=None,
        label=bundle.label,
        meta=synthetic_meta,
        is_bundle=True,
    )


def enumerate_backups(
    base_dir: Path,
    kind: ArtifactKind | None = None,
) -> list[BackupRecord]:
    """Scan backup slot directories and return a newest-first list of records.

    Scans ``<base_dir>/<kind>/<ts>/`` when *kind* is given, or
    ``<base_dir>/<any_kind>/<ts>/`` when *kind* is ``None``.  Skips
    ``.pending/`` residue and slots with unparseable or missing sidecars
    (logs WARN for each skipped slot).

    Bundle-aware: slot directories containing ``bundle.meta.json`` are
    detected **before** ``read_meta`` is called, so bundle slots are never
    mis-classified as invalid per-artifact slots.

    Parameters
    ----------
    base_dir:
        Root of the backup store (e.g. ``data/ha/backups/``).  Returns an
        empty list when this directory does not exist.
    kind:
        When provided, only slots matching this kind are returned.  When
        ``None``, all kinds are included.

    Returns
    -------
    list[BackupRecord]
        Newest-first by ``timestamp`` string (lexicographic, which matches
        chronological order for the ``YYYYMMDD-HHMMSSff`` format).
    """
    base_dir = Path(base_dir)
    if not base_dir.exists():
        return []

    records: list[BackupRecord] = []

    if kind is not None and kind != ArtifactKind.SNAPSHOT_BUNDLE:
        # Scan a single kind directory for concrete per-artifact kinds whose
        # directory name matches the ArtifactKind value exactly.
        kind_dirs = [base_dir / kind.value]
    else:
        # For kind=None (all slots) and kind=SNAPSHOT_BUNDLE (bundles may live
        # under any directory name the caller chose for base_dir — bundles are
        # stored under "snapshot/", not "snapshot_bundle/"), scan
        # every immediate subdirectory and apply the kind filter on the detected
        # record kind after reading the manifest or sidecar.
        kind_dirs = [d for d in base_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]

    for kind_dir in kind_dirs:
        if not kind_dir.is_dir():
            continue
        for slot in kind_dir.iterdir():
            # Skip .pending/ and any hidden directory.
            if slot.name.startswith("."):
                continue
            if not slot.is_dir():
                continue
            # Fix 9 (2026-04-23): skip symlinks — a symlink in the backup root
            # could allow a restore to read files outside the backup directory tree.
            if slot.is_symlink():
                logger.warning("enumerate_backups: skipping symlink %s", slot)
                continue

            # Detect bundle slots BEFORE calling read_meta.
            # bundle.meta.json is the index for snapshot_bundle slots; calling
            # read_meta on a bundle slot would fail because there is no
            # per-artifact .meta.json sidecar.
            bundle_manifest_path = slot / _BUNDLE_MANIFEST_FILENAME
            if bundle_manifest_path.exists():
                record = _read_bundle_record(slot)
                if record is None:
                    continue
                # Apply kind filter.
                if kind is not None and record.kind != kind:
                    continue
                records.append(record)
                continue

            # Regular per-artifact slot path.
            try:
                meta = read_meta(slot)
            except (MetaSchemaError, FileNotFoundError, OSError) as exc:
                logger.warning(
                    "enumerate_backups: skipping slot %s — sidecar unreadable: %s",
                    slot,
                    exc,
                )
                continue

            ts = meta.timestamp
            created_at = _parse_slot_timestamp(ts)
            if created_at is None:
                logger.warning(
                    "enumerate_backups: skipping slot %s — cannot parse timestamp %r",
                    slot,
                    ts,
                )
                continue

            record = BackupRecord(
                slot_dir=slot.resolve(),
                kind=meta.kind,
                timestamp=ts,
                created_at=created_at.replace(tzinfo=timezone.utc),
                content_sha256=meta.content_sha256,
                pre_trial_hash=meta.pre_trial_hash,
                label=meta.label,
                meta=meta,
                is_bundle=False,
            )
            # When scanning all kind dirs (kind=None or kind=SNAPSHOT_BUNDLE),
            # filter per-artifact records against the requested kind.
            if kind is not None and record.kind != kind:
                continue
            records.append(record)

    # Newest-first (timestamps are lexicographically comparable).
    records.sort(key=lambda r: r.timestamp, reverse=True)
    return records
