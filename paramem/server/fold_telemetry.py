"""Bounded machine-lifetime telemetry ring for the consolidation fold.

Splits per-fold VRAM/adapter integer metrics out of both
``paramem.training.consolidation`` (a 7000+-line orchestrator, per project
rules an oversized file is a defect to split, not to enlarge) and
``paramem.server.incidents`` (a failure-event store: dedup-by-``(type,
key)`` lifecycle, no cap, written only on threshold crossings). Telemetry has
a different shape and a different lifecycle — every fold cycle writes its own
record set regardless of outcome, so the store is a bounded ring of the last
``max_cycles`` cycles rather than a deduplicated event table.

This module exists for general hang diagnosability, not to test any specific
VRAM-cost hypothesis. The payload is integers, adapter/tier names, and
ISO-8601 timestamps only — never transcripts, facts, keys, or speaker ids.
Always written, regardless of ``debug`` (the privacy / artifact-persistence
switch that gates whether transcript-derived debug snapshots are written) —
gating telemetry on it would blind every ship-safe (``debug: false``)
deployment.

Schema
------
``{"version": 1, "cycles": [{"cycle_stamp": str, "records": [dict, ...]}]}``

Each ``record`` is tagged with a caller-supplied ``kind`` discriminator
(e.g. ``"backup_creation"``, ``"tier_train"``) and a ``recorded_at``
ISO-8601 timestamp, plus caller-supplied integer/string fields.

Ring
----
Bounded at ``max_cycles`` (default 200) cycles, oldest evicted first on
every write once the ring is full — a ring by construction, not a
separate prune job.

Written incrementally: one write after the backup-creation measurement and
one after each tier's training completes, so a mid-fold process death still
leaves the records for every phase that finished before the crash.

Concurrency
-----------
Uses :func:`paramem.server.atomic_json.flock_rmw` — the same
``fcntl.flock``-guarded read-modify-write primitive as
``paramem.server.incidents``. No new locking helper.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from paramem.server.atomic_json import flock_rmw

TELEMETRY_FILENAME: str = "fold_telemetry.json"
TELEMETRY_SCHEMA_VERSION: int = 1
TELEMETRY_LOCKFILE: str = "fold_telemetry.lock"


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _parse_store(raw: dict | None) -> list[dict]:
    """Parse the raw store dict into its list of per-cycle entries.

    Tolerant of an absent, malformed, or version-mismatched store (unlike
    ``incidents.py``'s strict schema guard): this is a diagnostics ring, not
    operator-facing control-plane state — a version bump silently starts a
    fresh ring rather than blocking the fold that is trying to write it.
    """
    if not isinstance(raw, dict) or raw.get("version") != TELEMETRY_SCHEMA_VERSION:
        return []
    cycles = raw.get("cycles", [])
    return cycles if isinstance(cycles, list) else []


def _build_store(cycles: list[dict]) -> dict:
    """Build the on-disk store dict from a list of per-cycle entries."""
    return {"version": TELEMETRY_SCHEMA_VERSION, "cycles": cycles}


def record_fold_telemetry(
    telemetry_dir: Path,
    *,
    cycle_stamp: str,
    kind: str,
    record: dict,
    max_cycles: int = 200,
) -> dict:
    """Append one telemetry record to the ring, upserting the current cycle.

    Parameters
    ----------
    telemetry_dir:
        Directory containing ``fold_telemetry.json`` (``paths.telemetry``).
        Created if absent.
    cycle_stamp:
        Identifier for the fold RUN this record belongs to — records sharing
        a ``cycle_stamp`` are appended to the same cycle entry. Callers MUST
        key on run identity (e.g. a fold-start timestamp), never on a
        content fingerprint of the training data: two distinct runs over an
        unchanged dataset (the common steady-state case) would otherwise
        collapse into one growing, unbounded cycle entry — the ring only
        bounds the number of *cycle* entries, not the records inside one. A
        content fingerprint is still useful; carry it as a field inside
        ``record`` instead (this module does not care what ``record``
        contains).
    kind:
        Discriminator for the record shape (e.g. ``"backup_creation"``,
        ``"tier_train"``, ``"interim_tier_train"``).
    record:
        Caller-supplied integer/string fields (e.g. ``free_before``,
        ``free_after``, ``peak_alloc``, ``adapter_count``, ``interim_count``,
        ``epochs``, ``tier``). Never transcripts, facts, keys, or speaker
        ids.
    max_cycles:
        Ring size. Once the number of distinct cycles exceeds this, the
        oldest cycles are evicted (a ring, not a prune job).

    Returns
    -------
    dict
        The new on-disk store dict (``{"version": 1, "cycles": [...]}]``).
    """

    def _mutate(current: dict | None) -> dict:
        cycles = _parse_store(current)
        entry = {"kind": kind, "recorded_at": _now_iso(), **record}
        new_cycles: list[dict] = []
        found = False
        for cycle in cycles:
            if cycle.get("cycle_stamp") == cycle_stamp:
                found = True
                updated = dict(cycle)
                updated["records"] = list(cycle.get("records", [])) + [entry]
                new_cycles.append(updated)
            else:
                new_cycles.append(cycle)
        if not found:
            new_cycles.append({"cycle_stamp": cycle_stamp, "records": [entry]})
        if len(new_cycles) > max_cycles:
            new_cycles = new_cycles[-max_cycles:]
        return _build_store(new_cycles)

    return flock_rmw(Path(telemetry_dir), TELEMETRY_LOCKFILE, _mutate, TELEMETRY_FILENAME)
