"""Backup slot enumeration for crash-recovery and /migration/confirm orphan detection.

Public surface
--------------
- ``enumerate_backups(base_dir, kind=None)`` — scan slot directories under
  ``data/ha/backups/``, skip ``.pending/`` residue, return newest-first list
  of ``BackupRecord`` instances.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from paramem.backup.backup import _parse_slot_timestamp
from paramem.backup.meta import read_meta
from paramem.backup.types import ArtifactKind, ArtifactMeta, MetaSchemaError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BackupRecord:
    """One slot enumerated from data/ha/backups/<kind>/.

    Attributes
    ----------
    slot_dir:
        Absolute path to the slot directory.
    kind:
        Artifact kind, from meta.json.
    timestamp:
        YYYYMMDD-HHMMSSff string (slot directory name).
    created_at:
        UTC datetime parsed from ``timestamp``.
    content_sha256:
        SHA-256 of the artifact bytes as stored on disk (ciphertext when
        encrypted, plaintext otherwise).
    pre_trial_hash:
        Optional SHA-256 of the live config at /migration/confirm step 2
        time.  Present only on pre-migration backups written by 3b.2+.
    label:
        Optional operator-supplied annotation from the sidecar.
    meta:
        Full ``ArtifactMeta`` sidecar for callers that need additional fields.
    """

    slot_dir: Path
    kind: ArtifactKind
    timestamp: str
    created_at: datetime
    content_sha256: str
    pre_trial_hash: str | None
    label: str | None
    meta: ArtifactMeta


def enumerate_backups(
    base_dir: Path,
    kind: ArtifactKind | None = None,
) -> list[BackupRecord]:
    """Scan backup slot directories and return a newest-first list of records.

    Scans ``<base_dir>/<kind>/<ts>/`` when *kind* is given, or
    ``<base_dir>/<any_kind>/<ts>/`` when *kind* is ``None``.  Skips
    ``.pending/`` residue and slots with unparseable or missing sidecars
    (logs WARN for each skipped slot).

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

    if kind is not None:
        # Scan a single kind directory.
        kind_dirs = [base_dir / kind.value]
    else:
        # Scan every immediate subdirectory as a potential kind directory.
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

            records.append(
                BackupRecord(
                    slot_dir=slot.resolve(),
                    kind=meta.kind,
                    timestamp=ts,
                    created_at=created_at.replace(tzinfo=timezone.utc),
                    content_sha256=meta.content_sha256,
                    pre_trial_hash=meta.pre_trial_hash,
                    label=meta.label,
                    meta=meta,
                )
            )

    # Newest-first (timestamps are lexicographically comparable).
    records.sort(key=lambda r: r.timestamp, reverse=True)
    return records
