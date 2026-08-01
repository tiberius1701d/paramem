"""Durable last-attempt stamp for the consolidation suspend/power-off catch-up gate.

Boundary: this module owns ONLY the on-disk representation of "when did the
scheduled-tick dispatcher last actually attempt a run" — for EVERY cadence
kind, not only non-calendar-exact ones (see ``schedule_grammar.scheduled_run_due``
/ ``server/app.py::_dispatch_consolidation``). A non-exact cadence needs the
stamp because its rendered ``OnCalendar`` timer is a wakeup source only, not
the schedule itself; an exact cadence needs it too, to keep a duplicate or
manual-adjacent tick inside the same calendar mark's window from being read
as due twice.

Deliberately NOT in ``systemd_timer.py``: the consumer is the runtime
dispatcher in ``app.py`` (``scheduled_tick`` →
``_dispatch_consolidation``), not the unit-rendering module —
and ``systemd_timer.py`` already carries three responsibilities (rendering,
reconciling, timer-state reading) without taking on run-state I/O too.

Schema
------
``{"schema_version": 1, "last_scheduled_run_epoch": <float>}``

Concurrency
-----------
Written from a single place (the event-loop-bound scheduled-tick
dispatcher) — no ``flock_rmw`` read-modify-write is needed, unlike
``backup/state.py`` (which has both a CLI runner process and a server
endpoint writing concurrently).
"""

from __future__ import annotations

from pathlib import Path

from paramem.server.atomic_json import atomic_write_json, read_json_or_none

SCHEDULE_STATE_FILENAME: str = "consolidation_schedule.json"
SCHEDULE_STATE_SCHEMA_VERSION: int = 1


def read_last_scheduled_run(state_dir: Path) -> float | None:
    """Return the epoch timestamp of the last scheduled-tick attempt, or ``None``.

    ``None`` covers both "file absent" (never seeded) and a missing
    ``last_scheduled_run_epoch`` field.

    Parameters
    ----------
    state_dir:
        Directory containing ``consolidation_schedule.json``
        (``config.paths.data / "state"``).
    """
    raw = read_json_or_none(Path(state_dir), SCHEDULE_STATE_FILENAME)
    if raw is None:
        return None
    return raw.get("last_scheduled_run_epoch")


def write_last_scheduled_run(state_dir: Path, epoch: float) -> None:
    """Atomically stamp *epoch* as the last scheduled-tick attempt.

    *epoch* must already be the value produced by
    ``schedule_grammar.scheduled_run_stamp_value`` (a calendar mark for
    anchored/exact-divisor cadences, a heartbeat-floored stamp for non-exact
    intervals) — this function does not floor or otherwise transform it; it
    writes exactly what it is given.

    Parameters
    ----------
    state_dir:
        Directory that will contain ``consolidation_schedule.json``.
        Created with parents if absent.
    epoch:
        Floored epoch timestamp of this attempt.
    """
    atomic_write_json(
        Path(state_dir),
        SCHEDULE_STATE_FILENAME,
        {"schema_version": SCHEDULE_STATE_SCHEMA_VERSION, "last_scheduled_run_epoch": epoch},
    )
