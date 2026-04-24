"""Systemd user-timer reconciliation for the scheduled backup runner.

Mirrors ``paramem/server/systemd_timer.py`` for the ``paramem-backup`` timer.
Reuses ``parse_schedule`` / ``TimerSpec`` from the consolidation timer module
(Decision E — same grammar, same parser).

Unit files point at ``python -m paramem.backup --tier daily`` (oneshot service).

``render_service_unit(tier)`` is parameterised on ``tier`` so weekly /
monthly / yearly timer units can be minted by passing ``tier="weekly"``
etc. — no backward-incompatible change to this function needed.

``_backup_timer_interval_seconds(schedule_str)`` maps a schedule string to
seconds so the ``/status`` populator can derive the stale threshold without
re-implementing the grammar.
"""

from __future__ import annotations

import logging
from pathlib import Path

from paramem.server.systemd_timer import (
    TimerSpec,
    _run_systemctl,  # noqa: PLC2701 – intentional private reuse
    _write_if_changed,  # noqa: PLC2701 – intentional private reuse
    parse_schedule,
)

logger = logging.getLogger(__name__)

TIMER_NAME = "paramem-backup"
UNIT_DIR = Path.home() / ".config" / "systemd" / "user"
SERVICE_PATH = UNIT_DIR / f"{TIMER_NAME}.service"
TIMER_PATH = UNIT_DIR / f"{TIMER_NAME}.timer"


# Re-export so tests can ``from paramem.backup.timer import parse_schedule``
# and verify it is the same object as the server timer's parse_schedule.
# (test_timer.py::test_reuses_parse_schedule)
__all__ = [
    "parse_schedule",
    "render_service_unit",
    "render_timer_unit",
    "reconcile",
    "cached_timer_state",
    "_backup_timer_interval_seconds",
]


def render_service_unit(python_path: str, project_root: str, tier: str = "daily") -> str:
    """Render the systemd .service unit for the backup runner.

    Parameterised on ``tier`` so weekly / monthly / yearly timer units
    can be emitted by passing ``tier="weekly"`` etc.

    Parameters
    ----------
    python_path:
        Absolute path to the Python interpreter (``sys.executable``).
    project_root:
        Absolute path to the project root (``WorkingDirectory``).
    tier:
        Backup tier — written as ``--tier <tier>`` in ``ExecStart``.
        Defaults to ``"daily"``.

    Returns
    -------
    str
        Content of the ``.service`` unit file.
    """
    return f"""[Unit]
Description=ParaMem scheduled backup (paramem.backup runner)
After=paramem-server.service

[Service]
Type=oneshot
WorkingDirectory={project_root}
ExecStart={python_path} -m paramem.backup --tier {tier}
"""


def render_timer_unit(spec: TimerSpec) -> str:
    """Render the systemd .timer unit for the backup schedule.

    Same shape as ``systemd_timer.render_timer_unit`` but with the
    backup-specific description and unit name.
    """
    lines = [
        "[Unit]",
        "Description=ParaMem scheduled backup",
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


def reconcile(
    schedule: str,
    *,
    python_path: str,
    project_root: str,
    tier: str = "daily",
) -> str:
    """Reconcile the ``paramem-backup`` systemd user timer.

    Mirrors ``paramem.server.systemd_timer.reconcile``.  Differences:

    - ``schedule`` comes from ``server_config.security.backups.schedule``
      (default ``"daily 04:00"``).
    - ``"off"`` / ``""`` disables and removes the timer units.
    - Unit files point at ``python -m paramem.backup --tier <tier>`` (not
      curl).

    Parameters
    ----------
    schedule:
        Schedule string (same grammar as ``consolidation.refresh_cadence``).
    python_path:
        Absolute path to the Python interpreter.
    project_root:
        Absolute path to the project root.
    tier:
        Backup tier written into the service unit's ``ExecStart``.  Always
        ``"daily"`` in production.

    Returns
    -------
    str
        Short human-readable description of the action taken.
    """
    # "daily HH:MM" is a natural operator idiom. Normalise to "HH:MM" so
    # parse_schedule (which expects either "daily" or "HH:MM", not both) can
    # parse it.  Keep the original string for error messages.
    import re as _re

    _schedule_normalised = schedule
    _daily_hhmm = _re.fullmatch(
        r"daily\s+(\d{1,2}:\d{2})", (schedule or "").strip(), _re.IGNORECASE
    )
    if _daily_hhmm:
        _schedule_normalised = _daily_hhmm.group(1)

    spec = parse_schedule(_schedule_normalised)
    if spec is None:
        logger.error(
            "Invalid backup schedule: %r — expected '', 'off', "
            "'HH:MM', 'daily HH:MM', 'every Nh', or 'every Nm'. Timer will be disabled.",
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
        return "backup timer: disabled" + (" (removed)" if changed else "")

    svc_content = render_service_unit(python_path, project_root, tier=tier)
    tmr_content = render_timer_unit(spec)

    svc_changed = _write_if_changed(SERVICE_PATH, svc_content)
    tmr_changed = _write_if_changed(TIMER_PATH, tmr_content)

    if svc_changed or tmr_changed:
        _run_systemctl("daemon-reload")

    enable = _run_systemctl("enable", "--now", f"{TIMER_NAME}.timer")
    if enable.returncode != 0:
        logger.warning(
            "systemctl enable --now %s.timer failed: %s",
            TIMER_NAME,
            enable.stderr.strip(),
        )
        return f"backup timer: enable failed ({enable.stderr.strip()[:80]})"

    if svc_changed or tmr_changed:
        _run_systemctl("restart", f"{TIMER_NAME}.timer")

    if spec.kind == "interval":
        detail = f"every {spec.on_unit_active_sec} (no catch-up)"
    elif spec.kind == "calendar":
        detail = f"calendar {spec.on_calendar} (with catch-up)"
    else:
        detail = f"daily at {spec.on_calendar} (with catch-up)"
    state = "updated" if (svc_changed or tmr_changed) else "already current"
    return f"backup timer: {state}, {detail}"


# ---------------------------------------------------------------------------
# Timer state cache (mirrors systemd_timer.cached_timer_state)
# ---------------------------------------------------------------------------

_state_cache: dict = {"at": 0.0, "value": {}}


def current_timer_state() -> dict:
    """Return current backup timer state.  Empty dict on failure."""
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

    import time as _time

    from paramem.server.systemd_timer import (
        _boot_monotonic_seconds,
        _parse_duration_seconds,
    )

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


def cached_timer_state(max_age_seconds: float = 5.0) -> dict:
    """Short-TTL cache wrapper for ``current_timer_state()``.

    ``/status`` is polled frequently; forking ``systemctl show`` on every
    request adds avoidable latency.  5 seconds is short enough that timer
    state updates are still visible in pstatus.

    Parameters
    ----------
    max_age_seconds:
        Cache TTL.  Defaults to 5 seconds.

    Returns
    -------
    dict
        Current timer state dict.
    """
    import time as _time

    now = _time.monotonic()
    if now - _state_cache["at"] < max_age_seconds and _state_cache["value"]:
        return _state_cache["value"]
    value = current_timer_state()
    _state_cache["at"] = now
    _state_cache["value"] = value
    return value


# ---------------------------------------------------------------------------
# Interval-seconds helper
# ---------------------------------------------------------------------------


def _backup_timer_interval_seconds(schedule_str: str) -> int:
    """Return the schedule interval in seconds for stale-detection.

    Maps a schedule string to the number of seconds between expected runs.
    Used by the ``/status`` populator to derive the 2× cadence stale
    threshold.

    Rules:

    - ``"off"`` / ``""`` / disabled → 0 (stale detection disabled).
    - ``"every Nh"`` → ``N * 3600``.
    - ``"every Nm"`` → ``N * 60``.
    - ``"HH:MM"`` (daily at time) → 86400.
    - ``"daily"`` → 86400.
    - ``"weekly"`` → 604800.
    - Invalid / unparseable → 0 (stale detection skipped).

    Parameters
    ----------
    schedule_str:
        Schedule string in the same grammar as ``consolidation.refresh_cadence``.

    Returns
    -------
    int
        Interval in seconds.  0 means "unknown / disabled".
    """
    import re

    s = (schedule_str or "").strip().lower()
    if s in ("", "off", "disabled", "none"):
        return 0

    if s in ("daily",):
        return 86400

    if s == "weekly":
        return 604800

    # HH:MM → daily.
    m = re.fullmatch(r"(\d{1,2}):(\d{2})", s)
    if m:
        return 86400

    # "daily HH:MM" — shorthand for a daily schedule at a specific time.
    m = re.fullmatch(r"daily\s+(\d{1,2}):(\d{2})", s)
    if m:
        return 86400

    # "every Nh" / "Nh".
    m = re.fullmatch(r"(?:every\s+)?(\d+)\s*h", s)
    if m:
        return int(m.group(1)) * 3600

    # "every Nm" / "Nm".
    m = re.fullmatch(r"(?:every\s+)?(\d+)\s*m(?:in)?", s)
    if m:
        return int(m.group(1)) * 60

    return 0
