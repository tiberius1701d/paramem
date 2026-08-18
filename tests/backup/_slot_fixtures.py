"""Shared backup-slot fixture home for the retention/status suite.

``_ts``/``_write_slot`` mint sequential ``YYYYMMDD-HHMMSSff`` backup-slot
timestamps and write a minimal backup slot (sidecar + data file) for the
retention-policy suite.

Consumers: ``tests/backup/test_retention.py``, ``tests/backup/test_runner_e2e.py``,
``tests/server/test_status_backup_block.py``.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path


def _write_slot(backups_root: Path, kind: str, ts: str, tier: str, size_bytes: int = 1024) -> Path:
    """Create a minimal backup slot directory with a sidecar and data file.

    ``ts`` must be in ``YYYYMMDD-HHMMSSff`` format (the slot directory name).
    Use ``_ts(i)`` to generate sequential test timestamps.
    """
    slot_dir = backups_root / kind / ts
    slot_dir.mkdir(parents=True, exist_ok=True)
    # Write meta.json
    meta = {
        "schema_version": 1,
        "kind": kind,
        "timestamp": ts,
        "content_sha256": "abc",
        "size_bytes": size_bytes,
        "encrypted": False,
        "tier": tier,
        "label": None,
    }
    (slot_dir / f"{kind}-{ts}.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    # Write data file with the specified size.
    (slot_dir / f"{kind}-{ts}.bin").write_bytes(b"x" * size_bytes)
    return slot_dir


def _ts(i: int) -> str:
    """Generate a sequential YYYYMMDD-HHMMSSff timestamp for test slot dirs.

    i=0 → oldest (2026-04-01), i=1 → one day later, etc.
    """
    base = datetime(2026, 4, 1, 4, 0, 0, tzinfo=timezone.utc)
    dt = base + timedelta(days=i)
    return dt.strftime("%Y%m%d-%H%M%S") + "00"
