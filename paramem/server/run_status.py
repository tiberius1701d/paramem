"""Universal per-op-type last-run registry for the ParaMem server.

Persists the latest run outcome for each op type to ``data/state/run_status.json``
(plaintext control-plane — survives a KEYLESS restart).  Not registered in
``infra_paths()``.

Schema
------
``{"version": 1, "last_runs": {<op_type>: RunRecord}}``

Each ``RunRecord`` holds the terminal outcome of the most recent run for that
op type.  Overwrite-latest: a new run replaces the previous record for the
same ``op_type`` — no dedup, no lifecycle, no count.

Op types
--------
Known op types: ``"consolidation"``, ``"migration"``, ``"training"``.
The ``op_type`` field is a free-form string; new types extend without a schema
bump.

Why two files (not one combined with ``incidents.json``)
---------------------------------------------------------
- Lifecycle divergence: ``run_status.json`` overwrites the latest per type;
  ``incidents.json`` appends/dedups/lifecycles.
- Write-frequency coupling: successes are common and must not churn the rare
  actionable incident list.

Concurrency
-----------
All public write functions use ``flock_rmw`` (``fcntl.flock`` on
``run_status.lock``) to serialise concurrent callers.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from paramem.server.atomic_json import flock_rmw, read_json_or_none

logger = logging.getLogger(__name__)

RUN_STATUS_FILENAME: str = "run_status.json"
RUN_STATUS_SCHEMA_VERSION: int = 1
RUN_STATUS_LOCKFILE: str = "run_status.lock"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class RunStatusError(Exception):
    """Base class for run-status I/O errors."""


class RunStatusSchemaError(RunStatusError):
    """Run-status file fails schema validation (version mismatch, bad JSON)."""


# ---------------------------------------------------------------------------
# RunRecord dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunRecord:
    """Terminal state of the most recent run for one op type.

    Attributes
    ----------
    op_type:
        Free-form op type string (e.g. ``"consolidation"``, ``"migration"``).
    outcome:
        Terminal outcome string for this op type.  Known values for
        ``"consolidation"``: ``"no_facts"``, ``"simulated"``, ``"trained"``,
        ``"accumulating"``, ``"noop"``, ``"aborted"``, ``"migration_complete"``,
        ``"migration_partial"``.  Extensible: new outcomes are additive.
    summary:
        Human-readable one-line description of the outcome (≤ ~80 chars).
    at:
        ISO-8601 UTC timestamp when this run completed.
    detail:
        Arbitrary JSON-serialisable dict for structured context.  Shape is
        per-``op_type`` and per-``outcome``; callers own the schema.
    """

    op_type: str
    outcome: str
    summary: str
    at: str
    detail: dict

    @classmethod
    def from_dict(cls, d: dict) -> "RunRecord":
        """Deserialise from a JSON dict.

        Raises ``RunStatusSchemaError`` on missing required fields.
        """
        if not isinstance(d, dict):
            raise RunStatusSchemaError("run record is not a JSON object")
        required = ("op_type", "outcome", "summary", "at", "detail")
        for field in required:
            if field not in d:
                raise RunStatusSchemaError(f"run record missing required field: {field}")
        return cls(
            op_type=d["op_type"],
            outcome=d["outcome"],
            summary=d["summary"],
            at=d["at"],
            detail=d["detail"],
        )

    def to_dict(self) -> dict:
        """Serialise to a JSON-ready dict."""
        return {
            "op_type": self.op_type,
            "outcome": self.outcome,
            "summary": self.summary,
            "at": self.at,
            "detail": self.detail,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _parse_store(raw: dict | None) -> tuple[int, dict]:
    """Parse the raw store dict into (version, last_runs).

    Raises ``RunStatusSchemaError`` on version mismatch.
    """
    if raw is None:
        return RUN_STATUS_SCHEMA_VERSION, {}
    if not isinstance(raw, dict):
        raise RunStatusSchemaError("run_status store root is not a JSON object")
    version = raw.get("version")
    if version is None:
        raise RunStatusSchemaError("run_status store missing required field: version")
    if version != RUN_STATUS_SCHEMA_VERSION:
        raise RunStatusSchemaError(
            f"run_status store version={version} does not match "
            f"current RUN_STATUS_SCHEMA_VERSION={RUN_STATUS_SCHEMA_VERSION}"
        )
    last_runs = raw.get("last_runs", {})
    if not isinstance(last_runs, dict):
        raise RunStatusSchemaError("run_status store 'last_runs' field is not a dict")
    return version, last_runs


def _build_store(last_runs: dict) -> dict:
    """Build the on-disk store dict from a last_runs mapping."""
    return {"version": RUN_STATUS_SCHEMA_VERSION, "last_runs": last_runs}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def record_last_run(
    state_dir: Path,
    *,
    op_type: str,
    outcome: str,
    summary: str,
    detail: dict,
) -> RunRecord:
    """Record the outcome of the most recent run for ``op_type``.

    Overwrites the previous record for the same ``op_type``.  No dedup,
    no lifecycle, no count.

    Parameters
    ----------
    state_dir:
        Directory containing ``run_status.json`` (e.g. ``data/state/``).
        Created if absent.
    op_type:
        Free-form op type string (e.g. ``"consolidation"``).
    outcome:
        Terminal outcome string (e.g. ``"trained"``, ``"aborted"``).
    summary:
        Human-readable one-line description (≤ ~80 chars).
    detail:
        Arbitrary JSON-serialisable dict for structured context.

    Returns
    -------
    RunRecord
        The new record as written to disk.
    """
    result_holder: list[RunRecord] = []

    def _mutate(current: dict | None) -> dict:
        _version, last_runs = _parse_store(current)
        now = _now_iso()
        record = RunRecord(
            op_type=op_type,
            outcome=outcome,
            summary=summary,
            at=now,
            detail=detail,
        )
        last_runs = dict(last_runs)
        last_runs[op_type] = record.to_dict()
        result_holder.append(record)
        return _build_store(last_runs)

    flock_rmw(Path(state_dir), RUN_STATUS_LOCKFILE, _mutate, RUN_STATUS_FILENAME)
    return result_holder[0]


def read_last_runs(state_dir: Path) -> dict[str, RunRecord]:
    """Read all last-run records from ``state_dir/run_status.json``.

    Parameters
    ----------
    state_dir:
        Directory containing ``run_status.json``.

    Returns
    -------
    dict[str, RunRecord]
        Mapping of ``op_type → RunRecord``.  Empty dict when the file is
        absent.

    Raises
    ------
    RunStatusSchemaError
        When the file exists but is malformed JSON or has a version mismatch.
    """
    try:
        raw = read_json_or_none(Path(state_dir), RUN_STATUS_FILENAME)
    except json.JSONDecodeError as exc:
        raise RunStatusSchemaError(f"run_status store is not valid JSON: {exc}") from exc
    except OSError as exc:
        raise RunStatusSchemaError(f"cannot read run_status store: {exc}") from exc

    _version, last_runs = _parse_store(raw)
    return {op_type: RunRecord.from_dict(rec) for op_type, rec in last_runs.items()}
