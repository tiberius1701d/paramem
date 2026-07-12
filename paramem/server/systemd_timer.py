"""Systemd user-timer reconciliation for the full-consolidation cycle.

``refresh_cadence`` (``consolidation.refresh_cadence``) is the only
user-facing scheduling knob and the only schedule string this module ever
sees — it is passed directly to :func:`reconcile` from the single call site in
``app.py``'s ``lifespan``.
This module does not receive the derived full-consolidation period
(``refresh_cadence × max_interim_count``); that derivation lives in
``ConsolidationScheduleConfig`` and is consumed by ``_is_full_cycle_due``,
not by the timer renderer.

THE PRINCIPLE — the systemd timer IS the schedule when the cadence is
exactly calendar-expressible; otherwise it degrades to a pure wakeup source
and a durable last-attempt stamp becomes the schedule.

``systemd``'s ``Persistent=true`` only affects ``OnCalendar=`` timers —
monotonic ``OnBootSec``/``OnUnitActiveSec`` timers run on ``CLOCK_MONOTONIC``,
which does not advance while the host is suspended, so a missed monotonic
tick is gone forever (``WakeSystem=true`` would fix that, but it requires
system-manager privileges this ``--user`` timer does not have). ``OnCalendar=``
cannot express every rolling period, though: it can only land on exact
divisors of its calendar field (24 for hours, 60 for minutes) and cannot
express a period longer than 24h.

So every cadence renders as ``OnCalendar`` + ``Persistent=true`` — there is
no more monotonic ``TimerSpec`` kind:

* Calendar-exact cadences (``daily``, ``weekly``, ``HH:MM``, and any
  ``every Nh``/``every Nm`` that divides 24/60) render at their exact
  period. The rendered timer alone is the schedule; systemd's catch-up
  fires the (single, coalesced) missed run on resume.
* Non-exact cadences (``every 5h``, ``every 90m``, ``every 48h``, ...)
  render at a coarser HEARTBEAT grid (see :func:`heartbeat_seconds`) —
  the timer is a wakeup source only. Each heartbeat, the dispatcher
  (``app.py::_dispatch_consolidation``) checks a durable
  last-attempt stamp (``schedule_state.py``) against the real cadence
  period and no-ops until it is actually due. This is the same answer the
  full fold already gives for its own due-ness (``_is_full_cycle_due``
  reads durable on-disk state and wall clock, never timer identity) —
  applied one level up, to the timer that drives the tick itself.

Accepted schedule strings (same parser as before, plus "off"):
    ""  / "off" / "disabled"  → no timer (manual /consolidate only)
    "every Nh" (24 % N == 0)  → OnCalendar=*-*-* HH,...:00:00 + Persistent=true
    "every Nh" (24 % N != 0)  → OnCalendar heartbeat (see heartbeat_seconds) + Persistent=true
    "every Nm" (60 % N == 0)  → OnCalendar=*:MM,...:00 + Persistent=true
    "every Nm" (60 % N != 0)  → OnCalendar heartbeat + Persistent=true
    "weekly"                  → OnCalendar=Mon *-*-* 00:00:00 + Persistent=true
    "HH:MM"                   → OnCalendar daily + Persistent=true
    "daily"                   → OnCalendar *-*-* 03:00:00 + Persistent=true
"""

from __future__ import annotations

import logging
import math
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from paramem.server.schedule_grammar import is_calendar_exact, parse_schedule_atom
from paramem.utils.paths import find_project_root

logger = logging.getLogger(__name__)

TIMER_NAME = "paramem-consolidate"
UNIT_DIR = Path.home() / ".config" / "systemd" / "user"
SERVICE_PATH = UNIT_DIR / f"{TIMER_NAME}.service"
TIMER_PATH = UNIT_DIR / f"{TIMER_NAME}.timer"

# Endpoint the timer curls. Matches server port in configs/server.yaml.
DEFAULT_ENDPOINT = "http://127.0.0.1:8420/scheduled-tick"


@dataclass(frozen=True)
class TimerSpec:
    """Rendered systemd timer unit parameters for a given schedule.

    kind values:
      "off"      — no timer installed.
      "calendar" — OnCalendar + Persistent=true; exact grid or heartbeat
                    grid (see module docstring), always catches up missed
                    ticks on boot/resume.
      "daily"    — OnCalendar + Persistent=true; fixed daily wall-clock time.

    There is no monotonic kind — every non-"off" timer is OnCalendar-based.
    """

    kind: str  # "off" | "calendar" | "daily"
    on_calendar: str | None = None


def _hours_to_calendar(n: int) -> str | None:
    """Return an OnCalendar expression for every-N-hours if 24 % n == 0, else None.

    The exact-grid renderer — unchanged by the catch-up-gate rework.  Also
    reused by :func:`_period_heartbeat_calendar` to render the coarser
    heartbeat grid for non-exact cadences (``gcd(count, 24)`` always divides
    24, so it is always a valid argument here).

    Examples::

        _hours_to_calendar(12) → "*-*-* 00,12:00:00"
        _hours_to_calendar(6)  → "*-*-* 00,06,12,18:00:00"
        _hours_to_calendar(24) → "*-*-* 00:00:00"
        _hours_to_calendar(5)  → None
    """
    if n <= 0 or 24 % n != 0:
        return None
    hours = [f"{h:02d}" for h in range(0, 24, n)]
    return f"*-*-* {','.join(hours)}:00:00"


def _minutes_to_calendar(n: int) -> str | None:
    """Return an OnCalendar expression for every-N-minutes if 60 % n == 0, else None.

    The exact-grid renderer — unchanged by the catch-up-gate rework.  Also
    reused by :func:`_period_heartbeat_calendar` (see :func:`_hours_to_calendar`).

    Examples::

        _minutes_to_calendar(15) → "*:00,15,30,45:00"
        _minutes_to_calendar(30) → "*:00,30:00"
        _minutes_to_calendar(13) → None
    """
    if n <= 0 or 60 % n != 0:
        return None
    minutes = [f"{m:02d}" for m in range(0, 60, n)]
    return f"*:{','.join(minutes)}:00"


def _period_heartbeat_calendar(count: int, unit: str) -> str:
    """Return the OnCalendar heartbeat expression for an interval cadence.

    Exact-grid cadences (``24 % count == 0`` for hours / ``60 % count == 0``
    for minutes) render at the exact period boundaries via
    ``_hours_to_calendar`` / ``_minutes_to_calendar`` — the timer alone is
    the schedule.

    Non-exact cadences render at the ``gcd(count, modulus)`` grid — the
    coarsest grid on which every period boundary still lands, since
    ``count`` is by construction a multiple of ``gcd(count, modulus)``.
    The timer is a pure wakeup source here; ``heartbeat_seconds`` +
    ``schedule_state.py`` decide which wakeups actually dispatch.
    """
    modulus = 24 if unit == "h" else 60
    if modulus % count == 0:
        cal = _hours_to_calendar(count) if unit == "h" else _minutes_to_calendar(count)
        assert cal is not None  # modulus % count == 0 guarantees this
        return cal
    grid = math.gcd(count, modulus)
    cal = _hours_to_calendar(grid) if unit == "h" else _minutes_to_calendar(grid)
    assert cal is not None  # gcd(count, modulus) always divides modulus
    return cal


def heartbeat_seconds(schedule: str) -> int | None:
    """Return the heartbeat grid, in seconds, for a non-calendar-exact cadence.

    ``None`` when *schedule* is calendar-exact (anchored, an exact-divisor
    interval, off, or unparseable — see
    :func:`~paramem.server.schedule_grammar.is_calendar_exact`) — meaning
    the rendered ``OnCalendar`` timer IS the schedule and no durable
    last-attempt gate is needed.

    For a non-exact interval cadence the grid is ``gcd(count, modulus)`` in
    the cadence's own unit (modulus = 24 for hours, 60 for minutes), i.e.
    the same grid :func:`_period_heartbeat_calendar` renders.
    """
    if is_calendar_exact(schedule):
        return None
    atom = parse_schedule_atom(schedule)
    modulus = 24 if atom.unit == "h" else 60
    grid = math.gcd(atom.count, modulus)
    return grid * 3600 if atom.unit == "h" else grid * 60


def floor_to_heartbeat(now: float, grid_seconds: int) -> float:
    """Floor an epoch timestamp DOWN to the current heartbeat grid boundary.

    ``grid_seconds`` is the value returned by :func:`heartbeat_seconds`.

    Grid boundaries are computed in LOCAL time from local midnight —
    matching ``interim_adapter.current_interim_stamp``'s flooring
    convention and the LOCAL time base of rendered ``OnCalendar``
    expressions. A UTC-epoch floor would only coincide with those LOCAL
    grid marks when the local UTC offset is a whole number of hours (it
    would misalign for e.g. UTC+05:30); flooring in local time is correct
    for every offset.

    Flooring (never rounding to nearest, never stamping raw wall-clock) is
    mandatory: the dispatch stamp is written strictly AFTER the heartbeat
    fires (guards, debounce, orphan-session claim, migration branch all run
    first), so an un-floored stamp lands one instant past a grid mark —
    pushing the next due-check to the FOLLOWING heartbeat and silently
    inflating the effective period every cycle (``every 5h`` converges to a
    real 6h, ``every 48h`` to 49h). A tolerance window would only narrow
    that error, not eliminate it; flooring is exact.
    """
    if grid_seconds <= 0:
        raise ValueError(f"grid_seconds must be positive, got {grid_seconds}")
    dt = datetime.fromtimestamp(now)
    midnight = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    since_midnight = int((dt - midnight).total_seconds())
    floored_since_midnight = (since_midnight // grid_seconds) * grid_seconds
    return (midnight + timedelta(seconds=floored_since_midnight)).timestamp()


def parse_schedule(schedule: str) -> TimerSpec | None:
    """Parse a schedule string into a systemd TimerSpec.

    Accepted formats:

    - ``""`` / ``"off"`` / ``"disabled"`` / ``"none"`` → disabled timer.
    - ``"weekly"`` → ``OnCalendar=Mon *-*-* 00:00:00`` + ``Persistent=true``.
    - ``"daily"`` → OnCalendar daily at 03:00 (same as ``"03:00"``).
    - ``"every Nh"`` where ``24 % N == 0`` → calendar timer at each N-hour mark.
    - ``"every Nh"`` where ``24 % N != 0`` → calendar HEARTBEAT timer (see
      :func:`heartbeat_seconds`); the dispatcher's durable stamp decides
      whether a given wakeup actually runs.
    - ``"every Nm"`` where ``60 % N == 0`` → calendar timer at each N-minute mark.
    - ``"every Nm"`` where ``60 % N != 0`` → calendar HEARTBEAT timer.
    - ``"HH:MM"`` → daily OnCalendar timer at the given time.

    Every non-off kind is ``OnCalendar`` + ``Persistent=true`` (see module
    docstring) — there is no monotonic fallback.

    Returns TimerSpec(kind="off") for an explicit off setting.
    Returns None on malformed input (caller logs + falls back to off).
    """
    atom = parse_schedule_atom(schedule)
    if atom is None:
        return None
    if atom.kind == "off":
        return TimerSpec(kind="off")
    if atom.kind == "weekly":
        return TimerSpec(kind="calendar", on_calendar="Mon *-*-* 00:00:00")
    if atom.kind == "daily":
        return TimerSpec(kind="daily", on_calendar="*-*-* 03:00:00")
    if atom.kind == "interval":
        cal = _period_heartbeat_calendar(atom.count, atom.unit)
        return TimerSpec(kind="calendar", on_calendar=cal)
    if atom.kind == "hhmm":
        return TimerSpec(
            kind="daily",
            on_calendar=f"*-*-* {atom.hh:02d}:{atom.mm:02d}:00",
        )
    return None


def render_service_unit(endpoint: str, project_root: str) -> str:
    """Render the systemd .service unit that curls ``/scheduled-tick``.

    The curl includes an ``Authorization: Bearer`` header sourced from the
    ``PARAMEM_API_TOKEN`` environment variable at service-execution time.
    The token is never written as a literal into the unit file; instead the
    service sources ``PARAMEM_API_TOKEN`` from the project ``.env`` file
    (``{project_root}/.env``).  The main ``paramem-server.service`` unit
    sources GPU environment from ``%t/paramem-gpu.env`` (rendered by
    ``gpu-guard``), not from the project ``.env``; the two units share the
    same token value but source it from different files.

    The curl command is wrapped in ``/bin/sh -c`` with ``-H @-`` so that the
    ``Authorization`` header is piped via stdin rather than appearing in
    process argv (visible via ``ps``/``/proc/<pid>/cmdline``) or in the
    systemd journal's logged ``ExecStart``.  Only the literal string
    ``$PARAMEM_API_TOKEN`` appears in the unit file; the token value is
    expanded from the ``EnvironmentFile`` at service-execution time.

    ``--fail-with-body`` (curl 7.76+) causes a non-zero exit on HTTP errors
    (e.g. 401 / 403) so a future misconfiguration surfaces as a failed
    systemd unit rather than a silent no-op.

    Parameters
    ----------
    endpoint:
        The URL to POST to (e.g. ``http://127.0.0.1:8420/scheduled-tick``).
        Must begin with ``http://127.0.0.1`` or ``http://localhost`` to prevent
        shell-metacharacter injection into the ``/bin/sh -c`` ExecStart line.
    project_root:
        Absolute path to the project root.  Used to derive the
        ``EnvironmentFile`` path so no username is hardcoded.

    Raises
    ------
    ValueError
        If *endpoint* does not start with ``http://127.0.0.1`` or
        ``http://localhost``.
    """
    _ALLOWED_ENDPOINT_PREFIXES = ("http://127.0.0.1", "http://localhost")
    if not any(endpoint.startswith(p) for p in _ALLOWED_ENDPOINT_PREFIXES):
        raise ValueError(
            f"endpoint must start with 'http://127.0.0.1' or 'http://localhost', got {endpoint!r}"
        )
    return (
        "[Unit]\n"
        "Description=ParaMem consolidation tick (curl /scheduled-tick)\n"
        "After=paramem-server.service\n"
        "\n"
        "[Service]\n"
        "Type=oneshot\n"
        f"EnvironmentFile=-{project_root}/.env\n"
        # ``%%s`` escapes systemd's ``%s`` specifier (= the user's login shell,
        # e.g. ``/bin/bash``); systemd expands ``%%s`` -> literal ``%s`` in the
        # rendered unit so ``sh``/``printf`` receives the intended format string.
        'ExecStart=/bin/sh -c \'printf "%%s" "Authorization: Bearer $PARAMEM_API_TOKEN"'
        f" | /usr/bin/curl -sS --fail-with-body -X POST --max-time 10 -H @- {endpoint}'\n"
    )


def render_timer_unit(
    spec: TimerSpec,
    *,
    unit_name: str = TIMER_NAME,
    description: str = "ParaMem consolidation scheduler",
) -> str:
    """Render the systemd .timer unit text for the given spec.

    Every non-``"off"`` kind (``"calendar"``, ``"daily"``) emits
    ``OnCalendar`` + ``Persistent=true`` so that missed ticks fire on the
    next boot/resume — there is no monotonic kind left to special-case (see
    module docstring).

    Parameterised on ``unit_name``/``description`` so the backup timer
    (``paramem.backup.timer``) renders through this one implementation
    instead of carrying its own copy.
    """
    lines = [
        "[Unit]",
        f"Description={description}",
        "",
        "[Timer]",
        f"Unit={unit_name}.service",
    ]
    if spec.kind in ("calendar", "daily"):
        lines.append(f"OnCalendar={spec.on_calendar}")
        lines.append("Persistent=true")
    lines.extend(["", "[Install]", "WantedBy=timers.target", ""])
    return "\n".join(lines)


def _run_systemctl(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["systemctl", "--user", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def _write_if_changed(path: Path, content: str) -> bool:
    """Write content only if it differs from current file. Returns True on change."""
    try:
        current = path.read_text()
    except FileNotFoundError:
        current = None
    if current == content:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return True


@dataclass(frozen=True)
class TimerTarget:
    """Identifies one systemd timer/service pair for :func:`_reconcile_timer`.

    The consolidation timer (this module) and the backup timer
    (``paramem.backup.timer``) reconcile identically except for these five
    values — ``_reconcile_timer`` is the single reconciliation core both
    build a ``TimerTarget`` for, rather than each carrying its own copy of
    the reconcile/render logic.
    """

    timer_name: str
    description: str
    service_path: Path
    timer_path: Path
    service_content: str


def _reconcile_timer(target: TimerTarget, schedule: str) -> str:
    """Shared reconciliation core for both the consolidation and backup timers.

    Parses *schedule*, renders/writes the unit pair, and enables the timer.
    Never raises on systemd errors — logs and returns a notice so the caller
    (server startup) still proceeds.

    Returns a short human-readable description of the action taken, which
    the caller logs.
    """
    spec = parse_schedule(schedule)
    if spec is None:
        logger.error(
            "Invalid %s schedule: %r — expected '', 'off', 'HH:MM', 'every Nh', "
            "or 'every Nm'. Timer will be disabled.",
            target.timer_name,
            schedule,
        )
        spec = TimerSpec(kind="off")

    if spec.kind == "off":
        changed = False
        if target.timer_path.exists():
            _run_systemctl("stop", f"{target.timer_name}.timer")
            _run_systemctl("disable", f"{target.timer_name}.timer")
            target.timer_path.unlink(missing_ok=True)
            target.service_path.unlink(missing_ok=True)
            _run_systemctl("daemon-reload")
            changed = True
        return f"{target.timer_name}: disabled" + (" (removed)" if changed else "")

    svc_changed = _write_if_changed(target.service_path, target.service_content)
    tmr_changed = _write_if_changed(
        target.timer_path,
        render_timer_unit(spec, unit_name=target.timer_name, description=target.description),
    )

    if svc_changed or tmr_changed:
        _run_systemctl("daemon-reload")

    enable = _run_systemctl("enable", "--now", f"{target.timer_name}.timer")
    if enable.returncode != 0:
        logger.warning(
            "systemctl enable --now %s.timer failed: %s",
            target.timer_name,
            enable.stderr.strip(),
        )
        return f"{target.timer_name}: enable failed ({enable.stderr.strip()[:80]})"

    # If unit already enabled, systemd won't restart it on daemon-reload —
    # force a restart so the new OnCalendar takes effect.
    if svc_changed or tmr_changed:
        _run_systemctl("restart", f"{target.timer_name}.timer")

    # Every non-off kind is OnCalendar + Persistent=true — catch-up always
    # applies, whether the grid is exact or a heartbeat (see module docstring).
    if spec.kind == "calendar":
        detail = f"calendar {spec.on_calendar} (with catch-up)"
    else:
        detail = f"daily at {spec.on_calendar} (with catch-up)"
    state = "updated" if (svc_changed or tmr_changed) else "already current"
    return f"{target.timer_name}: {state}, {detail}"


def reconcile(
    schedule: str,
    endpoint: str = DEFAULT_ENDPOINT,
    project_root: str | None = None,
) -> str:
    """Reconcile the systemd user timer with the configured schedule.

    Returns a short human-readable description of the action taken, which the
    caller logs. Never raises on systemd errors — logs and returns a notice so
    the server still starts.

    Parameters
    ----------
    schedule:
        The interim refresh cadence string (``consolidation.refresh_cadence``,
        e.g. ``"every 12h"`` / ``"12h"`` / ``"HH:MM"`` / ``"daily"``). The timer
        fires at this cadence; whether a given tick runs a full consolidation
        is decided per-tick by ``_is_full_cycle_due`` (interim accumulation plus
        an oldest-interim deadline), not by a separate derived-period timer.
    endpoint:
        URL the timer curls.  Defaults to
        ``http://127.0.0.1:8420/scheduled-tick``.
    project_root:
        Absolute path to the project root used to derive the
        ``EnvironmentFile`` path in the rendered service unit.  Defaults to
        ``None``, in which case it is resolved via
        ``find_project_root(Path(__file__))`` (nearest ancestor containing
        ``pyproject.toml``), falling back to
        ``Path(__file__).resolve().parents[2]`` when no such ancestor exists
        (e.g. an installed package under site-packages).
    """
    if project_root is None:
        _r = find_project_root(Path(__file__))
        project_root = str(_r if _r is not None else Path(__file__).resolve().parents[2])
    target = TimerTarget(
        timer_name=TIMER_NAME,
        description="ParaMem consolidation scheduler",
        service_path=SERVICE_PATH,
        timer_path=TIMER_PATH,
        service_content=render_service_unit(endpoint, project_root),
    )
    return _reconcile_timer(target, schedule)


def current_timer_state(timer_name: str = TIMER_NAME) -> dict:
    """Return current timer state for /status reporting. Empty dict on failure.

    Parameterised on ``timer_name`` so consolidation and backup timers share
    one reader.  The timer-unit path is derived as
    ``UNIT_DIR / f"{timer_name}.timer"``.

    ``next_elapse_us`` is sourced from ``systemctl list-timers --output=json``
    rather than ``systemctl show -p NextElapseUSecRealtime``.  On systemd 255
    the ``show`` property renders as a human-readable date string even with
    ``--timestamp=unix``, making integer parsing unreliable.  The JSON output
    of ``list-timers`` always carries ``"next"`` as a plain integer in
    microseconds (``0`` = no next elapse), which is the reliable source.
    """
    timer_path = UNIT_DIR / f"{timer_name}.timer"
    if not timer_path.exists():
        return {"installed": False}
    show = _run_systemctl(
        "show",
        f"{timer_name}.timer",
        "--property=ActiveState,LastTriggerUSec",
    )
    if show.returncode != 0:
        return {"installed": True, "error": show.stderr.strip()}
    props = {}
    for line in show.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            props[k] = v

    # Retrieve next elapse via list-timers --output=json.  The JSON entry
    # for the timer carries "next" as an integer in microseconds; 0 means
    # no next elapse scheduled.  An absent unit (timer not yet loaded by
    # systemd) returns an empty array — treat as no-next.
    import json as _json

    next_elapse_us = ""
    list_out = _run_systemctl(
        "list-timers",
        f"{timer_name}.timer",
        "--output=json",
    )
    if list_out.returncode == 0 and list_out.stdout.strip():
        try:
            timers = _json.loads(list_out.stdout)
            for entry in timers:
                if entry.get("unit") == f"{timer_name}.timer":
                    next_val = entry.get("next", 0)
                    if isinstance(next_val, int) and next_val > 0:
                        next_elapse_us = str(next_val)
                    break
        except (ValueError, TypeError, KeyError):
            pass  # malformed JSON or unexpected shape — leave next_elapse_us ""

    return {
        "installed": True,
        "active": props.get("ActiveState") == "active",
        "next_elapse_us": next_elapse_us,
        "last_trigger_us": props.get("LastTriggerUSec", ""),
    }


# Per-name TTL cache: keyed by timer_name so consolidation and backup states
# do not collide.
_state_cache: dict[str, dict] = {}


def cached_timer_state(timer_name: str = TIMER_NAME, max_age_seconds: float = 5.0) -> dict:
    """Short-TTL cache wrapper for :func:`current_timer_state`.

    ``/status`` is polled frequently; forking ``systemctl show`` on every
    request adds avoidable latency.  5 s is short enough that timer state
    updates are still visible in ``pstatus`` without being noticeably stale.

    Parameterised on ``timer_name`` so consolidation and backup states are
    cached independently (separate entries in ``_state_cache``).
    """
    import time as _time

    now = _time.monotonic()
    entry = _state_cache.get(timer_name, {"at": 0.0, "value": {}})
    if now - entry["at"] < max_age_seconds and entry["value"]:
        return entry["value"]
    value = current_timer_state(timer_name)
    _state_cache[timer_name] = {"at": now, "value": value}
    return value
