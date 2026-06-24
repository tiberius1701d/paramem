"""Actionable-failure event store for the ParaMem server.

Persists operator-visible failure events to ``data/state/incidents.json``
(plaintext control-plane — survives a KEYLESS restart so failures are visible
before the daily key loads).  Not registered in ``infra_paths()``; the
restore-bundle sweep only touches ``adapters/``-rooted infra.

Schema
------
``{"version": 1, "incidents": [Incident, ...]}``

Each ``Incident`` has a deterministic ``id = f"{type}:{key}"`` so that
``ack_incident(state_dir, id)`` needs no separate lookup table.

Dedup
-----
Repeated failures with the same ``(type, key)`` bump ``count`` and refresh
``last_seen`` / ``summary`` / ``detail`` / ``severity`` rather than appending
a new row.  A previously-resolved incident is REOPENED (status → active,
count bumped) if the same ``(type, key)`` recurs.

Lifecycle
---------
``status`` ∈ {``"active"``, ``"acknowledged"``, ``"resolved"``}.

- ``record_incident`` → active (or reopen if resolved).
- ``ack_incident`` → acknowledged (silences the attention row; details still
  visible).
- ``resolve_incident`` / ``resolve_incidents_by_type`` → resolved.
- AUTO-RESOLVE on the next successful op of that type (caller's
  responsibility; see ``resolve_incidents_by_type``).

Concurrency
-----------
All public write functions use ``flock_rmw`` (``fcntl.flock`` on
``incidents.lock``) to serialise concurrent callers (BG-trainer thread,
event-loop, migration coroutine).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from paramem.server.atomic_json import flock_rmw, read_json_or_none

logger = logging.getLogger(__name__)

INCIDENTS_FILENAME: str = "incidents.json"
INCIDENTS_SCHEMA_VERSION: int = 1
INCIDENTS_LOCKFILE: str = "incidents.lock"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class IncidentStoreError(Exception):
    """Base class for incident-store I/O errors."""


class IncidentStoreSchemaError(IncidentStoreError):
    """Incident store file fails schema validation (version mismatch, bad JSON)."""


# ---------------------------------------------------------------------------
# Incident dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Incident:
    """One actionable-failure event in the incident store.

    Attributes
    ----------
    id:
        Deterministic identifier ``f"{type}:{key}"``.  Stable across reads;
        used by ``ack_incident``.
    type:
        Discriminator string (e.g. ``"vram_exhausted"``,
        ``"consolidation_retry_exhausted"``).  Extensible: new failure types
        add a ``type`` without a schema bump.
    severity:
        One of ``"info"``, ``"warning"``, ``"failed"``.  Maps 1:1 to
        ``AttentionItem.level`` for the attention populator.
    first_seen:
        ISO-8601 UTC timestamp when the incident was first recorded.
        Monotonic — never updated on subsequent occurrences.
    last_seen:
        ISO-8601 UTC timestamp of the most recent occurrence.
    count:
        Number of times this incident has been recorded (including reopens).
    status:
        One of ``"active"``, ``"acknowledged"``, ``"resolved"``.
    summary:
        Human-readable one-line description (≤ ~80 chars).
    detail:
        Arbitrary JSON-serialisable dict for structured context.  Shape is
        per-``type``; callers own the schema.
    """

    id: str
    type: str
    severity: str
    first_seen: str
    last_seen: str
    count: int
    status: str
    summary: str
    detail: dict

    @classmethod
    def from_dict(cls, d: dict) -> "Incident":
        """Deserialise from a JSON dict.

        Raises ``IncidentStoreSchemaError`` on missing required fields.
        """
        if not isinstance(d, dict):
            raise IncidentStoreSchemaError("incident entry is not a JSON object")
        required = (
            "id",
            "type",
            "severity",
            "first_seen",
            "last_seen",
            "count",
            "status",
            "summary",
            "detail",
        )
        for field in required:
            if field not in d:
                raise IncidentStoreSchemaError(f"incident entry missing required field: {field}")
        return cls(
            id=d["id"],
            type=d["type"],
            severity=d["severity"],
            first_seen=d["first_seen"],
            last_seen=d["last_seen"],
            count=d["count"],
            status=d["status"],
            summary=d["summary"],
            detail=d["detail"],
        )

    def to_dict(self) -> dict:
        """Serialise to a JSON-ready dict."""
        return {
            "id": self.id,
            "type": self.type,
            "severity": self.severity,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "count": self.count,
            "status": self.status,
            "summary": self.summary,
            "detail": self.detail,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _parse_store(raw: dict | None) -> tuple[int, list[dict]]:
    """Parse the raw store dict into (version, incidents_list).

    Raises ``IncidentStoreSchemaError`` on version mismatch.
    """
    if raw is None:
        return INCIDENTS_SCHEMA_VERSION, []
    if not isinstance(raw, dict):
        raise IncidentStoreSchemaError("incidents store root is not a JSON object")
    version = raw.get("version")
    if version is None:
        raise IncidentStoreSchemaError("incidents store missing required field: version")
    if version != INCIDENTS_SCHEMA_VERSION:
        raise IncidentStoreSchemaError(
            f"incidents store version={version} does not match "
            f"current INCIDENTS_SCHEMA_VERSION={INCIDENTS_SCHEMA_VERSION}"
        )
    incidents = raw.get("incidents", [])
    if not isinstance(incidents, list):
        raise IncidentStoreSchemaError("incidents store 'incidents' field is not a list")
    return version, incidents


def _build_store(incidents: list[dict]) -> dict:
    """Build the on-disk store dict from a list of raw incident dicts."""
    return {"version": INCIDENTS_SCHEMA_VERSION, "incidents": incidents}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def record_incident(
    state_dir: Path,
    *,
    type: str,
    key: str,
    severity: str,
    summary: str,
    detail: dict,
) -> Incident:
    """Record a failure event, deduplicating by ``(type, key)``.

    If an incident with ``id = f"{type}:{key}"`` already exists:

    - ``active`` or ``acknowledged``: bump ``count`` and ``last_seen``;
      refresh ``summary``, ``detail``, and ``severity``.
    - ``resolved``: REOPEN — status → ``"active"``, ``count`` bumped,
      ``last_seen`` refreshed.

    If no matching incident exists, append a new one with ``count=1`` and
    ``status="active"``.

    Parameters
    ----------
    state_dir:
        Directory containing ``incidents.json`` (e.g. ``data/state/``).
        Created if absent.
    type:
        Failure-type discriminator string (e.g. ``"vram_exhausted"``).
    key:
        Per-type dedup key (e.g. a session id or phase name).  The incident
        ``id`` is ``f"{type}:{key}"``.
    severity:
        ``"info"``, ``"warning"``, or ``"failed"``.
    summary:
        Human-readable one-line description (≤ ~80 chars).
    detail:
        Arbitrary JSON-serialisable dict for structured context.

    Returns
    -------
    Incident
        The resulting (new or updated) incident.
    """
    incident_id = f"{type}:{key}"
    result_holder: list[Incident] = []

    def _mutate(current: dict | None) -> dict:
        _version, rows = _parse_store(current)
        now = _now_iso()
        found = False
        new_rows = []
        for row in rows:
            if row.get("id") == incident_id:
                found = True
                new_row = dict(row)
                new_row["count"] = row.get("count", 0) + 1
                new_row["last_seen"] = now
                new_row["summary"] = summary
                new_row["detail"] = detail
                new_row["severity"] = severity
                if row.get("status") == "resolved":
                    new_row["status"] = "active"
                new_rows.append(new_row)
            else:
                new_rows.append(row)
        if not found:
            new_rows.append(
                {
                    "id": incident_id,
                    "type": type,
                    "severity": severity,
                    "first_seen": now,
                    "last_seen": now,
                    "count": 1,
                    "status": "active",
                    "summary": summary,
                    "detail": detail,
                }
            )
        result_incident = next(r for r in new_rows if r["id"] == incident_id)
        result_holder.append(Incident.from_dict(result_incident))
        return _build_store(new_rows)

    flock_rmw(Path(state_dir), INCIDENTS_LOCKFILE, _mutate, INCIDENTS_FILENAME)
    return result_holder[0]


def read_incidents(state_dir: Path) -> list[Incident]:
    """Read all incidents from ``state_dir/incidents.json``.

    Parameters
    ----------
    state_dir:
        Directory containing ``incidents.json``.

    Returns
    -------
    list[Incident]
        All incidents (any status).  Empty list when the file is absent.

    Raises
    ------
    IncidentStoreSchemaError
        When the file exists but is malformed JSON or has a version mismatch.
    """
    try:
        raw = read_json_or_none(Path(state_dir), INCIDENTS_FILENAME)
    except json.JSONDecodeError as exc:
        raise IncidentStoreSchemaError(f"incidents store is not valid JSON: {exc}") from exc
    except OSError as exc:
        raise IncidentStoreSchemaError(f"cannot read incidents store: {exc}") from exc

    _version, rows = _parse_store(raw)
    return [Incident.from_dict(r) for r in rows]


def resolve_incident(state_dir: Path, type: str, key: str) -> bool:
    """Resolve the incident matching ``(type, key)``.

    Idempotent: returns ``False`` if no matching incident is found (normal
    when a success op has no prior failure).

    Parameters
    ----------
    state_dir:
        Directory containing ``incidents.json``.
    type:
        Failure-type discriminator string.
    key:
        Per-type dedup key.

    Returns
    -------
    bool
        ``True`` if a matching incident was found and resolved; ``False``
        otherwise.
    """
    incident_id = f"{type}:{key}"
    found_holder: list[bool] = [False]

    def _mutate(current: dict | None) -> dict:
        _version, rows = _parse_store(current)
        now = _now_iso()
        new_rows = []
        for row in rows:
            if row.get("id") == incident_id:
                if row.get("status") != "resolved":
                    # Actual transition: flip status and record the change.
                    found_holder[0] = True
                    new_row = dict(row)
                    new_row["status"] = "resolved"
                    new_row["last_seen"] = now
                    new_rows.append(new_row)
                else:
                    # Already resolved — idempotent no-op; return False to signal
                    # no transition occurred (matching resolve_incidents_by_type
                    # counting semantics so callers can trust the boolean).
                    new_rows.append(row)
            else:
                new_rows.append(row)
        return _build_store(new_rows)

    flock_rmw(Path(state_dir), INCIDENTS_LOCKFILE, _mutate, INCIDENTS_FILENAME)
    return found_holder[0]


def resolve_incidents_by_type(state_dir: Path, type: str) -> int:
    """Resolve every active incident of the given ``type``.

    Used by success-path auto-resolve (all incidents of a type clear when the
    op succeeds).  Idempotent: resolving already-resolved incidents is a no-op.

    Parameters
    ----------
    state_dir:
        Directory containing ``incidents.json``.
    type:
        Failure-type discriminator string.

    Returns
    -------
    int
        The number of incidents whose status changed to ``"resolved"``.
    """
    count_holder: list[int] = [0]

    def _mutate(current: dict | None) -> dict:
        _version, rows = _parse_store(current)
        now = _now_iso()
        new_rows = []
        for row in rows:
            if row.get("type") == type and row.get("status") in ("active", "acknowledged"):
                count_holder[0] += 1
                new_row = dict(row)
                new_row["status"] = "resolved"
                new_row["last_seen"] = now
                new_rows.append(new_row)
            else:
                new_rows.append(row)
        return _build_store(new_rows)

    flock_rmw(Path(state_dir), INCIDENTS_LOCKFILE, _mutate, INCIDENTS_FILENAME)
    return count_holder[0]


def ack_incident(state_dir: Path, id: str) -> bool:
    """Acknowledge the incident with the given ``id``.

    Acknowledged incidents are silenced from the attention block but remain
    visible in the store.  Idempotent; returns ``False`` when the id is not
    found.

    Parameters
    ----------
    state_dir:
        Directory containing ``incidents.json``.
    id:
        The deterministic incident id (``f"{type}:{key}"``).

    Returns
    -------
    bool
        ``True`` if a matching incident was found and acknowledged; ``False``
        otherwise.
    """
    found_holder: list[bool] = [False]

    def _mutate(current: dict | None) -> dict:
        _version, rows = _parse_store(current)
        new_rows = []
        for row in rows:
            if row.get("id") == id:
                found_holder[0] = True
                new_row = dict(row)
                new_row["status"] = "acknowledged"
                new_rows.append(new_row)
            else:
                new_rows.append(row)
        return _build_store(new_rows)

    flock_rmw(Path(state_dir), INCIDENTS_LOCKFILE, _mutate, INCIDENTS_FILENAME)
    return found_holder[0]
