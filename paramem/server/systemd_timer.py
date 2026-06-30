"""Systemd user-timer reconciliation for the full-consolidation cycle.

The server derives the full-consolidation period from
``refresh_cadence × max_interim_count`` (see
``ConsolidationScheduleConfig.consolidation_period_string``) and passes that
derived period string to :func:`reconcile`. ``refresh_cadence`` is the only
user-facing scheduling knob; this module never sees it directly. On startup
the server renders the expected ``paramem-consolidate.timer`` and
``paramem-consolidate.service`` units, diffs them against what is currently
installed under ``~/.config/systemd/user/``, and reconciles with
``daemon-reload`` + ``enable --now`` (or ``stop`` + ``disable`` when the
derived period is "off"/empty).

This replaces the in-process `_consolidation_scheduler` asyncio task. The
in-process scheduler did not survive server restart — auto-reclaim restarts
the service on GPU-free, and each restart reset the next-tick timestamp to
now + interval, so interval schedules effectively never fired. A systemd
timer is wall-clock driven and survives restarts.

Accepted schedule strings (same parser as before, plus "off"):
    ""  / "off" / "disabled"  → no timer (manual /consolidate only)
    "every Nh" (24 % N == 0)  → OnCalendar=*-*-* HH,...:00:00 + Persistent=true
    "every Nh" (24 % N != 0)  → OnBootSec + OnUnitActiveSec (no catch-up)
    "every Nm" (60 % N == 0)  → OnCalendar=*:MM,...:00 + Persistent=true
    "every Nm" (60 % N != 0)  → OnBootSec + OnUnitActiveSec (no catch-up)
    "weekly"                  → OnCalendar=Mon *-*-* 00:00:00 + Persistent=true
    "HH:MM"                   → OnCalendar daily + Persistent=true
    "daily"                   → OnCalendar *-*-* 03:00:00 + Persistent=true

Interval cadences that divide evenly into 24h or 60min are converted to
OnCalendar expressions so that systemd's ``Persistent=true`` can fire missed
ticks on the next boot. Non-divisible cadences fall back to monotonic
OnBootSec + OnUnitActiveSec; missed ticks are silently dropped in that case.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

from paramem.server.schedule_grammar import parse_schedule_atom

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
      "calendar" — OnCalendar + Persistent=true; catches up missed ticks on boot.
      "daily"    — OnCalendar + Persistent=true; fixed daily wall-clock time.
      "interval" — OnBootSec + OnUnitActiveSec; missed ticks are silently dropped.
    """

    kind: str  # "off" | "interval" | "calendar" | "daily"
    on_calendar: str | None = None  # calendar / daily mode
    on_boot_sec: str | None = None  # interval mode
    on_unit_active_sec: str | None = None  # interval mode


def _hours_to_calendar(n: int) -> str | None:
    """Return an OnCalendar expression for every-N-hours if 24 % n == 0, else None.

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

    Examples::

        _minutes_to_calendar(15) → "*:00,15,30,45:00"
        _minutes_to_calendar(30) → "*:00,30:00"
        _minutes_to_calendar(13) → None
    """
    if n <= 0 or 60 % n != 0:
        return None
    minutes = [f"{m:02d}" for m in range(0, 60, n)]
    return f"*:{','.join(minutes)}:00"


def parse_schedule(schedule: str) -> TimerSpec | None:
    """Parse a schedule string into a systemd TimerSpec.

    Accepted formats:

    - ``""`` / ``"off"`` / ``"disabled"`` / ``"none"`` → disabled timer.
    - ``"weekly"`` → ``OnCalendar=Mon *-*-* 00:00:00`` + ``Persistent=true``.
    - ``"daily"`` → OnCalendar daily at 03:00 (same as ``"03:00"``).
    - ``"every Nh"`` where ``24 % N == 0`` → calendar timer at each N-hour mark
      with ``Persistent=true`` (catches up missed ticks on boot).
    - ``"every Nh"`` where ``24 % N != 0`` → monotonic interval timer; missed
      ticks during outages are silently dropped.
    - ``"every Nm"`` where ``60 % N == 0`` → calendar timer at each N-minute mark
      with ``Persistent=true``.
    - ``"every Nm"`` where ``60 % N != 0`` → monotonic interval timer.
    - ``"HH:MM"`` → daily OnCalendar timer at the given time.

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
        n = atom.count
        if atom.unit == "h":
            cal = _hours_to_calendar(n)
            if cal is not None:
                return TimerSpec(kind="calendar", on_calendar=cal)
            sec = f"{n}h"
        else:
            cal = _minutes_to_calendar(n)
            if cal is not None:
                return TimerSpec(kind="calendar", on_calendar=cal)
            sec = f"{n}min"
        return TimerSpec(kind="interval", on_boot_sec=sec, on_unit_active_sec=sec)
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
        'ExecStart=/bin/sh -c \'printf "%s" "Authorization: Bearer $PARAMEM_API_TOKEN"'
        f" | /usr/bin/curl -sS --fail-with-body -X POST --max-time 10 -H @- {endpoint}'\n"
    )


def render_timer_unit(spec: TimerSpec) -> str:
    """Render the systemd .timer unit text for the given spec.

    Both ``"calendar"`` and ``"daily"`` kinds emit ``OnCalendar`` +
    ``Persistent=true`` so that missed ticks fire on the next boot.
    ``"interval"`` kind uses monotonic ``OnBootSec`` / ``OnUnitActiveSec``
    and does NOT include ``Persistent=true`` (systemd ignores it for
    monotonic timers; missed ticks during outages are silently dropped).
    """
    lines = [
        "[Unit]",
        "Description=ParaMem consolidation scheduler",
        "",
        "[Timer]",
        f"Unit={TIMER_NAME}.service",
    ]
    if spec.kind == "interval":
        lines.append(f"OnBootSec={spec.on_boot_sec}")
        lines.append(f"OnUnitActiveSec={spec.on_unit_active_sec}")
    elif spec.kind in ("calendar", "daily"):
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


def reconcile(
    schedule: str,
    endpoint: str = DEFAULT_ENDPOINT,
    project_root: str = str(Path(__file__).resolve().parent.parent.parent),
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
        the project root inferred from this module's location so the server
        lifespan can rely on the default; the ``app.py`` call site passes it
        explicitly via ``str(Path(__file__).resolve().parent.parent.parent)``
        for clarity and testability.
    """
    spec = parse_schedule(schedule)
    if spec is None:
        logger.error(
            "Invalid consolidation refresh cadence: %r — expected '', 'off', "
            "'HH:MM', 'every Nh', or 'every Nm'. This is "
            "consolidation.refresh_cadence; check it in "
            "server.yaml. Timer will be disabled.",
            schedule,
        )
        spec = TimerSpec(kind="off")

    if spec.kind == "off":
        changed = False
        if TIMER_PATH.exists():
            _run_systemctl("stop", f"{TIMER_NAME}.timer")
            _run_systemctl("disable", f"{TIMER_NAME}.timer")
            TIMER_PATH.unlink(missing_ok=True)
            SERVICE_PATH.unlink(missing_ok=True)
            _run_systemctl("daemon-reload")
            changed = True
        return "consolidation timer: disabled" + (" (removed)" if changed else "")

    svc_changed = _write_if_changed(SERVICE_PATH, render_service_unit(endpoint, project_root))
    tmr_changed = _write_if_changed(TIMER_PATH, render_timer_unit(spec))

    if svc_changed or tmr_changed:
        _run_systemctl("daemon-reload")

    enable = _run_systemctl("enable", "--now", f"{TIMER_NAME}.timer")
    if enable.returncode != 0:
        logger.warning(
            "systemctl enable --now %s.timer failed: %s",
            TIMER_NAME,
            enable.stderr.strip(),
        )
        return f"consolidation timer: enable failed ({enable.stderr.strip()[:80]})"

    # If unit already enabled, systemd won't restart it on daemon-reload —
    # force a restart so the new OnCalendar/OnUnitActiveSec takes effect.
    if svc_changed or tmr_changed:
        _run_systemctl("restart", f"{TIMER_NAME}.timer")

    if spec.kind == "interval":
        detail = f"every {spec.on_unit_active_sec} (no catch-up)"
    elif spec.kind == "calendar":
        detail = f"calendar {spec.on_calendar} (with catch-up)"
    else:
        detail = f"daily at {spec.on_calendar} (with catch-up)"
    state = "updated" if (svc_changed or tmr_changed) else "already current"
    return f"consolidation timer: {state}, {detail}"


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
