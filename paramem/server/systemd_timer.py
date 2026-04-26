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


def render_service_unit(endpoint: str) -> str:
    return f"""[Unit]
Description=ParaMem consolidation tick (curl /scheduled-tick)
After=paramem-server.service

[Service]
Type=oneshot
ExecStart=/usr/bin/curl -sS -X POST --max-time 10 {endpoint}
"""


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


def reconcile(schedule: str, endpoint: str = DEFAULT_ENDPOINT) -> str:
    """Reconcile the systemd user timer with the configured schedule.

    Returns a short human-readable description of the action taken, which the
    caller logs. Never raises on systemd errors — logs and returns a notice so
    the server still starts.
    """
    spec = parse_schedule(schedule)
    if spec is None:
        logger.error(
            "Invalid derived consolidation period: %r — expected '', 'off', "
            "'HH:MM', 'every Nh', or 'every Nm'. This is derived from "
            "consolidation.refresh_cadence × max_interim_count; check those in "
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

    svc_changed = _write_if_changed(SERVICE_PATH, render_service_unit(endpoint))
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


def _boot_monotonic_seconds() -> float:
    """Seconds since boot (matches systemd's CLOCK_MONOTONIC)."""
    with open("/proc/uptime") as f:
        return float(f.read().split()[0])


_DURATION_UNITS = {
    "us": 1e-6,
    "ms": 1e-3,
    "s": 1.0,
    "min": 60.0,
    "h": 3600.0,
    "d": 86400.0,
    "w": 604800.0,
    "month": 2629800.0,
    "y": 31557600.0,
}


def _parse_duration_seconds(s: str) -> float | None:
    """Parse systemd duration strings like '2h 10.990051s' → seconds.

    Systemd renders NextElapseUSecMonotonic as a human-readable duration,
    not raw microseconds, so we parse it directly. Returns None on any
    parse failure; caller treats that as 'no next elapse available'.
    """
    import re as _re

    s = s.strip()
    if not s:
        return None
    total = 0.0
    found = False
    for m in _re.finditer(r"([0-9]+(?:\.[0-9]+)?)\s*([a-z]+)", s):
        value = float(m.group(1))
        unit = m.group(2)
        if unit not in _DURATION_UNITS:
            return None
        total += value * _DURATION_UNITS[unit]
        found = True
    return total if found else None


def current_timer_state() -> dict:
    """Return current timer state for /status reporting. Empty dict on failure."""
    if not TIMER_PATH.exists():
        return {"installed": False}
    show = _run_systemctl(
        "show",
        f"{TIMER_NAME}.timer",
        "--property=ActiveState,NextElapseUSecRealtime,NextElapseUSecMonotonic,LastTriggerUSec",
    )
    if show.returncode != 0:
        return {"installed": True, "error": show.stderr.strip()}
    props = {}
    for line in show.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            props[k] = v

    # Interval timers (OnBootSec/OnUnitActiveSec) populate Monotonic as a
    # human-readable duration (e.g. "2h 10.990051s"); calendar timers
    # populate Realtime as a timestamp string. Return next-elapse as a
    # realtime microsecond integer regardless of source.
    import time as _time

    next_elapse_us = props.get("NextElapseUSecRealtime", "").strip()
    if not next_elapse_us.isdigit():
        duration_s = _parse_duration_seconds(props.get("NextElapseUSecMonotonic", ""))
        if duration_s is not None:
            boot_s = _time.time() - _boot_monotonic_seconds()
            next_elapse_us = str(int((boot_s + duration_s) * 1_000_000))
        else:
            next_elapse_us = ""

    return {
        "installed": True,
        "active": props.get("ActiveState") == "active",
        "next_elapse_us": next_elapse_us,
        "last_trigger_us": props.get("LastTriggerUSec", ""),
    }


_state_cache: dict = {"at": 0.0, "value": {}}


def cached_timer_state(max_age_seconds: float = 5.0) -> dict:
    """Short-TTL cache wrapper for current_timer_state().

    /status is polled frequently; forking `systemctl show` on every request
    adds avoidable latency. 5s is short enough that timer state updates are
    still visible in pstatus without being noticeably stale.
    """
    import time as _time

    now = _time.monotonic()
    if now - _state_cache["at"] < max_age_seconds and _state_cache["value"]:
        return _state_cache["value"]
    value = current_timer_state()
    _state_cache["at"] = now
    _state_cache["value"] = value
    return value
