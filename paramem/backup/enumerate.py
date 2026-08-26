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

A bundle whose ``bundle_schema_version`` does not match this build's is
still enumerated — marked :attr:`BackupRecord.incompatible` rather than
hidden — so an operator can see it in ``/backup/list``; restore of one
still refuses.
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
    BUNDLE_SCHEMA_VERSION,
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
        Optional SHA-256 of the live config captured when the pre-migration
        backup was written (see ``ArtifactMeta.pre_trial_hash``).  Present
        only on pre-migration backups; ``None`` on every other kind and
        always ``None`` for bundle slots.
    label:
        Optional operator-supplied annotation from the sidecar.
    meta:
        Full ``ArtifactMeta`` sidecar for callers that need additional fields.
        For bundle slots this is a synthetic ``ArtifactMeta`` derived from the
        ``BundleManifest`` so that callers can access ``meta.tier``.
    is_bundle:
        ``True`` when the slot is a ``snapshot_bundle`` with a
        ``bundle.meta.json`` manifest.  Callers that need bundle-specific
        fields (``adapters``, ``files``) should read the
        ``bundle.meta.json`` directly from ``slot_dir``.
    incompatible:
        ``True`` when the slot's ``bundle_schema_version`` does not match
        this build's :data:`~paramem.backup.types.BUNDLE_SCHEMA_VERSION` —
        the bundle is real and enumerated (visible in ``/backup/list``) but
        cannot be restored by this version. Always ``False`` for
        non-bundle records.
    found_bundle_schema_version:
        The ``bundle_schema_version`` actually found on disk, when
        *incompatible* is ``True``. ``None`` otherwise.
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
    incompatible: bool = False
    found_bundle_schema_version: int | None = None


def _build_bundle_record(
    slot: Path,
    *,
    tier: str,
    label: str | None,
    incompatible: bool,
    found_bundle_schema_version: int | None,
) -> BackupRecord | None:
    """Build the :class:`BackupRecord` for a bundle slot at *slot*.

    The one bundle-record builder — shared by the compatible path
    (:func:`_read_bundle_record`) and the incompatible-schema-version path
    (:func:`_read_incompatible_bundle_record`) so a future
    :class:`BackupRecord` field never has to be added in two places at once.
    The two callers differ only in the four keyword parameters this function
    takes: a compatible bundle passes the fully-validated
    :class:`~paramem.backup.types.BundleManifest`'s own ``tier``/``label``
    with ``incompatible=False``; an incompatible one passes whatever
    ``tier``/``label`` the raw (unvalidated) manifest dict happened to carry,
    with ``incompatible=True`` and the mismatched version it found.

    Parameters
    ----------
    slot:
        Slot directory containing ``bundle.meta.json``.
    tier:
        Backup tier tag to embed in the synthetic sidecar and the record.
    label:
        Optional operator-supplied annotation.
    incompatible:
        Whether this bundle's ``bundle_schema_version`` differs from this
        build's :data:`~paramem.backup.types.BUNDLE_SCHEMA_VERSION`.
    found_bundle_schema_version:
        The ``bundle_schema_version`` value found on disk when
        *incompatible* is ``True``; ``None`` otherwise.

    Returns
    -------
    BackupRecord | None
        ``None`` (logging WARN) when the slot's timestamp cannot be parsed.
    """
    ts = slot.name
    created_at = _parse_slot_timestamp(ts)
    if created_at is None:
        logger.warning(
            "enumerate_backups: skipping bundle slot %s — cannot parse timestamp %r",
            slot,
            ts,
        )
        return None

    manifest_path = slot / _BUNDLE_MANIFEST_FILENAME
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
        tier=tier,
        label=label,
    )

    return BackupRecord(
        slot_dir=slot.resolve(),
        kind=ArtifactKind.SNAPSHOT_BUNDLE,
        timestamp=ts,
        created_at=created_at.replace(tzinfo=timezone.utc),
        content_sha256=manifest_sha256,
        pre_trial_hash=None,
        label=label,
        meta=synthetic_meta,
        is_bundle=True,
        incompatible=incompatible,
        found_bundle_schema_version=found_bundle_schema_version,
    )


def _read_incompatible_bundle_record(slot: Path, raw: dict, found_version) -> BackupRecord | None:
    """Build a :class:`BackupRecord` for a bundle whose ``bundle_schema_version``
    does not match this build's :data:`~paramem.backup.types.BUNDLE_SCHEMA_VERSION`.

    The manifest cannot be trusted to validate against the current
    :meth:`BundleManifest.from_dict` schema (fields may have been renamed or
    removed since *found_version*), so this reads only what every bundle
    manifest version is expected to carry opportunistically (``tier``,
    ``label``) and delegates construction to :func:`_build_bundle_record`
    with ``incompatible=True`` — visible in enumeration, refused by restore
    (which re-validates the manifest itself and raises
    :class:`~paramem.backup.types.BundleManifestError`).

    Parameters
    ----------
    slot:
        Slot directory containing ``bundle.meta.json``.
    raw:
        The parsed manifest dict (already JSON-decoded).
    found_version:
        The ``bundle_schema_version`` value found in *raw* (may be any type
        or absent).

    Returns
    -------
    BackupRecord | None
        ``None`` (logging WARN) when the slot's timestamp cannot be parsed.
    """
    logger.warning(
        "enumerate_backups: bundle slot %s has incompatible bundle_schema_version "
        "(expected %s, got %r) — enumerated as incompatible, not restorable by "
        "this version",
        slot,
        BUNDLE_SCHEMA_VERSION,
        found_version,
    )

    tier = raw.get("tier", "") if isinstance(raw, dict) else ""
    label = raw.get("label") if isinstance(raw, dict) else None
    return _build_bundle_record(
        slot,
        tier=tier,
        label=label,
        incompatible=True,
        found_bundle_schema_version=found_version,
    )


def _read_bundle_record(slot: Path) -> BackupRecord | None:
    """Attempt to read a bundle slot from *slot* and return a BackupRecord.

    Reads ``bundle.meta.json`` from *slot*.  A ``bundle_schema_version``
    mismatch does NOT hide the slot — it is enumerated as an
    :attr:`BackupRecord.incompatible` record (see
    :func:`_read_incompatible_bundle_record`) so an operator can still see it
    in ``/backup/list``; only a genuinely unparseable manifest (missing
    required fields, corrupt JSON, unreadable file) is skipped with a
    WARNING and ``None`` returned.

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
    except (OSError, ValueError) as exc:
        logger.warning(
            "enumerate_backups: skipping bundle slot %s — cannot read manifest: %s",
            slot,
            exc,
        )
        return None

    # Only a manifest that NAMES a version (however old) is a legitimate
    # incompatible bundle rather than a corrupt/garbage file — a manifest
    # missing the field entirely falls through to BundleManifest.from_dict,
    # which raises BundleManifestError (version mismatch against None) and
    # is caught below the same as any other malformed manifest.
    if isinstance(raw, dict) and "bundle_schema_version" in raw:
        found_version = raw["bundle_schema_version"]
        if found_version != BUNDLE_SCHEMA_VERSION:
            return _read_incompatible_bundle_record(slot, raw, found_version)

    try:
        bundle = BundleManifest.from_dict(raw)
    except BundleManifestError as exc:
        logger.warning(
            "enumerate_backups: skipping bundle slot %s — manifest invalid: %s",
            slot,
            exc,
        )
        return None

    return _build_bundle_record(
        slot,
        tier=bundle.tier,
        label=bundle.label,
        incompatible=False,
        found_bundle_schema_version=None,
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
            # Skip symlinks — a symlink in the backup root could allow a restore
            # to read files outside the backup directory tree.
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
