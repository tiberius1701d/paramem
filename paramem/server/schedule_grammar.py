"""Single source of truth for schedule-string grammar.

Two consumers share this:

* ``systemd_timer.parse_schedule`` — translates the parsed atom into a
  ``TimerSpec`` for unit-file generation.
* ``interim_adapter.compute_schedule_period_seconds`` — translates the
  parsed atom into a wall-clock period in seconds.

Until this commit each carried its own near-identical regex copy of the
grammar; this module is the merger.

Accepted forms (case-insensitive on the kind keywords; HH:MM is plain
digits):

* ``""`` / ``"off"`` / ``"disabled"`` / ``"none"`` → off (no schedule)
* ``"weekly"``                                     → weekly
* ``"daily"``                                      → daily (24-hour period)
* ``"HH:MM"``                                      → daily at HH:MM
* ``"Nh"`` / ``"Nm"`` / ``"every Nh"`` / ``"every Nm"`` → interval

Regex usage here is permitted because the grammar is fully controlled
(operators write these strings into ``server.yaml``; we document the
form). The project-wide rule about not using regex for intent
classification on free-form user text does *not* apply to a bounded
internal grammar — only one place to update if the grammar changes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_OFF_VALUES = frozenset({"", "off", "disabled", "none"})

# Bounded grammar — these match a finite set of accepted strings, not
# user-supplied free text. Parsed once on config load.
_INTERVAL_RE = re.compile(r"^(?:every\s+)?(\d+)\s*([hm])$", re.IGNORECASE)
_HHMM_RE = re.compile(r"^(\d{1,2}):(\d{2})$")


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
