"""Single source of truth for schedule-string grammar.

Consumers:

* ``systemd_timer.parse_schedule`` — translates the parsed atom into a
  ``TimerSpec`` for unit-file generation; ``systemd_timer.heartbeat_seconds``
  uses :func:`is_calendar_exact` to decide whether a cadence needs a
  heartbeat grid instead of an exact one.
* :func:`compute_schedule_period_seconds` — translates the parsed atom into
  a wall-clock period in seconds.  Used by ``server/config.py``
  (``ConsolidationScheduleConfig``), ``memory/interim_adapter.py``
  (``current_interim_stamp``), and ``server/app.py`` (``/status`` and the
  full-cycle deadline gate).
* :func:`is_calendar_exact` / :func:`is_due` — the shared suspend/power-off
  catch-up gate.  Called from ``server/app.py``'s scheduled-tick dispatcher
  and ``backup/__main__.py``'s standalone runner; both gate on a durable
  last-ATTEMPT stamp for cadences the systemd timer can only approximate.

Until the catch-up-gate change each of these carried its own near-identical
regex copy of the grammar — including a THIRD copy in
``backup/timer.py::_backup_timer_interval_seconds`` that earlier revisions of
this docstring missed; that copy has been retired in favour of
:func:`compute_schedule_period_seconds`. This module is the merger.

Accepted forms (case-insensitive on the kind keywords; HH:MM is plain
digits):

* ``""`` / ``"off"`` / ``"disabled"`` / ``"none"`` → off (no schedule)
* ``"weekly"``                                     → weekly
* ``"daily"``                                      → daily (24-hour period)
* ``"HH:MM"`` / ``"daily HH:MM"``                   → daily at HH:MM
* ``"Nh"`` / ``"Nm"`` / ``"every Nh"`` / ``"every Nm"`` → interval

Regex usage here is permitted because the grammar is fully controlled
(operators write these strings into ``server.yaml``; we document the
form). The project-wide rule about not using regex for intent
classification on free-form user text does *not* apply to a bounded
internal grammar — only one place to update if the grammar changes.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

_OFF_VALUES = frozenset({"", "off", "disabled", "none"})

# Bounded grammar — these match a finite set of accepted strings, not
# user-supplied free text. Parsed once on config load.
_INTERVAL_RE = re.compile(r"^(?:every\s+)?(\d+)\s*([hm])$", re.IGNORECASE)
_HHMM_RE = re.compile(r"^(\d{1,2}):(\d{2})$")
_DAILY_HHMM_RE = re.compile(r"^daily\s+(\d{1,2}:\d{2})$", re.IGNORECASE)


def strip_daily_prefix(schedule: str | None) -> str | None:
    """Normalise the ``"daily HH:MM"`` operator idiom to bare ``"HH:MM"``.

    ``"daily 04:00"`` -> ``"04:00"``; any other input (including bare
    ``"daily"``, its own grammar atom) is returned unchanged.

    Applied once at the top of :func:`parse_schedule_atom` so every consumer
    accepts the idiom uniformly — previously only
    ``paramem.backup.timer.reconcile`` normalised it, with its own private
    regex. Widening the grammar here is deliberate: ``compute_schedule_period_seconds``
    and the systemd-timer parser now accept the same idiom the backup
    schedule has always used.
    """
    if schedule is None:
        return None
    m = _DAILY_HHMM_RE.match(schedule.strip())
    return m.group(1) if m else schedule


@dataclass(frozen=True)
class ParsedSchedule:
    """Atoms produced by :func:`parse_schedule_atom`.

    Attributes:
        kind: one of ``"off"``, ``"weekly"``, ``"daily"``, ``"interval"``,
            ``"hhmm"``.
        count: interval multiplier (only for ``kind="interval"``).
        unit: ``"h"`` or ``"m"`` (only for ``kind="interval"``).
        hh: hour 0-23 (only for ``kind="hhmm"``).
        mm: minute 0-59 (only for ``kind="hhmm"``).
    """

    kind: str
    count: int = 0
    unit: str = ""
    hh: int = 0
    mm: int = 0


def parse_schedule_atom(schedule: str | None) -> ParsedSchedule | None:
    """Parse a schedule string into a structured atom.

    Returns ``None`` for unparseable input. Distinguishing ``None`` from
    ``ParsedSchedule(kind="off")`` matters for the two consumers:

    * Both treat the latter as "no schedule".
    * Only ``compute_schedule_period_seconds`` raises ``ValueError`` on
      ``None`` (true grammar errors); ``parse_schedule`` is more
      forgiving and returns its own ``"off"`` variant.

    Out-of-range values for ``HH:MM`` (e.g. ``"24:01"``) and zero/negative
    interval counts return ``None`` here so the caller can decide whether
    to raise or fall back.
    """
    schedule = strip_daily_prefix(schedule)
    s = (schedule or "").strip()
    lower = s.lower()
    if lower in _OFF_VALUES:
        return ParsedSchedule(kind="off")
    if lower == "weekly":
        return ParsedSchedule(kind="weekly")
    if lower == "daily":
        return ParsedSchedule(kind="daily")

    m = _INTERVAL_RE.match(s)
    if m:
        count = int(m.group(1))
        unit = m.group(2).lower()
        if count <= 0:
            return None
        return ParsedSchedule(kind="interval", count=count, unit=unit)

    m = _HHMM_RE.match(s)
    if m:
        hh = int(m.group(1))
        mm = int(m.group(2))
        if 0 <= hh < 24 and 0 <= mm < 60:
            return ParsedSchedule(kind="hhmm", hh=hh, mm=mm)

    return None


def compute_schedule_period_seconds(schedule: str) -> int | None:
    """Return the full consolidation period in seconds for a schedule string.

    Relocated from ``paramem.memory.interim_adapter`` (2026-07 catch-up-gate
    change) — this is schedule-grammar logic, not interim-adapter lifecycle
    logic, and ``interim_adapter`` importing from ``backup`` would be a
    layering violation once the backup runner shares this function too.

    - ``"weekly"``                → 604800 seconds (7 days)
    - ``"daily"``                 → 86400 seconds (1 day)
    - ``"every Nh"`` / ``"Nh"``   → N × 3600 seconds
    - ``"every Nm"`` / ``"Nm"``   → N × 60 seconds
    - ``"HH:MM"`` / ``"daily HH:MM"`` → 86400 seconds (daily)
    - ``""`` / ``"off"`` / ``"disabled"`` / ``"none"`` → ``None`` (manual only)

    Returns:
        Period in seconds, or ``None`` when no schedule is configured.

    Raises:
        ValueError: if *schedule* is not ``None`` / off / a valid period string.
    """
    atom = parse_schedule_atom(schedule)
    if atom is None:
        raise ValueError(
            f"Unrecognised schedule string: {schedule!r}. Expected '', 'off', "
            "'weekly', 'daily', 'HH:MM', 'Nh'/'Nm', or 'every Nh'/'every Nm'."
        )
    if atom.kind == "off":
        return None
    if atom.kind == "weekly":
        return 604800
    if atom.kind == "daily":
        return 86400
    if atom.kind == "interval":
        return atom.count * 3600 if atom.unit == "h" else atom.count * 60
    if atom.kind == "hhmm":
        return 86400
    raise ValueError(f"Unhandled schedule kind: {atom.kind!r}")


def is_calendar_exact(schedule: str) -> bool:
    """Return True when *schedule* is exactly expressible as a calendar grid.

    ``daily`` / ``weekly`` / ``HH:MM`` are exact by construction (anchored
    wall-clock points). An ``interval`` cadence is exact iff it divides its
    calendar modulus (24 for hours, 60 for minutes) — the same condition
    ``systemd_timer._hours_to_calendar`` / ``_minutes_to_calendar`` gate on.

    ``off`` returns True: no schedule means no catch-up gate applies
    (vacuously exact — there is nothing to catch up).

    Unparseable input (``parse_schedule_atom`` returns ``None``) also
    returns True, deliberately: an invalid schedule is already handled at
    the point it is parsed for real (``systemd_timer.reconcile`` logs and
    disables the timer at startup; ``ConsolidationScheduleConfig`` also
    rejects it outright when ``max_interim_count == 0``, the one mode where
    an unparseable cadence would otherwise accumulate pending sessions
    unboundedly). The catch-up gate must not ALSO block a tick on a schedule
    string some other layer already flagged — that would double-report the
    same error as two different symptoms.
    """
    atom = parse_schedule_atom(schedule)
    if atom is None:
        return True
    if atom.kind in ("off", "weekly", "daily", "hhmm"):
        return True
    modulus = 24 if atom.unit == "h" else 60
    return modulus % atom.count == 0


def is_due(
    last_attempt_epoch: float | None,
    period_seconds: int,
    now: float | None = None,
) -> bool:
    """Return True when at least ``period_seconds`` have elapsed since the last attempt.

    ``last_attempt_epoch=None`` (nothing recorded yet) is always due — pure
    function, no I/O; callers own reading/writing the durable stamp
    (``schedule_state.py`` for consolidation, ``backup.json`` for backups).
    Gates on the last ATTEMPT, never the last SUCCESS: a persistently
    failing run must not pass the gate on every heartbeat (that would be a
    retry storm), so callers pass an attempt timestamp that is written on
    both success and failure.
    """
    if last_attempt_epoch is None:
        return True
    if now is None:
        now = time.time()
    return (now - last_attempt_epoch) >= period_seconds
