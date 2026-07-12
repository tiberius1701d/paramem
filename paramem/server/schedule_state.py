"""Durable last-attempt stamp for the consolidation suspend/power-off catch-up gate.

Boundary: this module owns ONLY the on-disk representation of "when did the
scheduled-tick dispatcher last actually attempt a run" for non-calendar-exact
cadences (see ``systemd_timer`` module docstring — a heartbeat schedule needs
a durable stamp because the rendered ``OnCalendar`` timer is a wakeup source
only, not the schedule itself).

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

    *epoch* must already be floored to the heartbeat grid
    (``systemd_timer.floor_to_heartbeat``) — this function does not floor;
    it writes exactly what it is given.

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
