"""Single source of truth for schedule-string grammar.

Consumers:

* ``systemd_timer.parse_schedule`` — translates the parsed atom into a
  ``TimerSpec`` for unit-file generation, importing the anchor constants
  (:data:`DAILY_ANCHOR_HOUR`/:data:`DAILY_ANCHOR_MINUTE`,
  :data:`WEEKLY_ANCHOR_HOUR`/:data:`WEEKLY_ANCHOR_MINUTE`/
  :data:`WEEKLY_ANCHOR_LABEL`) and the non-exact-interval grid helpers
  (:func:`interval_is_exact` / :func:`non_exact_interval_grid`) from here so
  the rendered ``OnCalendar`` expression and this module's own mark/dueness
  math agree by construction — no anchor is stated twice.
* :func:`compute_schedule_period_seconds` — translates the parsed atom into
  a wall-clock period in seconds.  Used by ``server/config.py``
  (``ConsolidationScheduleConfig``), ``memory/interim_adapter.py``
  (``current_interim_stamp``), and ``server/app.py`` (``/status`` and the
  full-cycle deadline gate).
* :func:`previous_mark` / :func:`scheduled_run_due` /
  :func:`scheduled_run_stamp_value` — this module's ownership of "where are
  the calendar marks for a cadence" and "is a scheduled run due", covering
  every cadence kind (anchored daily/weekly/HH:MM, exact-divisor intervals,
  and non-exact intervals on their heartbeat-floored stamp).  ``server/app.py``'s
  arbitrator (``_dispatch_consolidation``) and ``backup/__main__.py``'s
  standalone runner (policy on an absent stamp: RUN, opposite of the
  arbitrator's seed-and-noop) are both wired onto these directly for every
  cadence kind.

This module is the single grammar and dueness-math implementation shared by
``server/systemd_timer.py`` (unit rendering), ``server/app.py`` (the
consolidation arbitrator), and ``backup/__main__.py`` (the standalone backup
runner).

Accepted forms (case-insensitive on the kind keywords; HH:MM is plain
digits):

* ``""`` / ``"off"`` / ``"disabled"`` / ``"none"`` → off (no schedule)
* ``"weekly"``                                     → weekly
* ``"daily"``                                      → daily (24-hour period)
* ``"HH:MM"`` / ``"daily HH:MM"``                   → daily at HH:MM
* ``"Nh"`` / ``"Nm"`` / ``"every Nh"`` / ``"every Nm"`` → interval

Regex usage here is one of the project's two permitted sites (the other is
``cloud/placeholders.py``); both are recorded in ``architecture.md``.  The
admissibility rule is structural, not a judgement about this module: each
shape is declared exactly ONCE as a fragment constant and composed into the
patterns that need it, so a grammar change is a single edit and no second
renderer can drift from it.  Operators write these strings into
``server.yaml``, so the input is bounded and trusted — but that is what makes
regex *safe* here, not what makes it *admissible*.

Mark / dueness math (:func:`previous_mark` and friends) is naive-local-time
throughout — never converted to UTC — to match ``OnCalendar``'s own local-time
semantics.  The one DST edge this leaves is the autumn fall-back hour, which
repeats a wall-clock local time; every ``datetime.replace(...)`` call in this
module pins ``fold=0`` (the first, pre-transition occurrence) so a mark
computed during the repeated hour is deterministic rather than ambiguous.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

_OFF_VALUES = frozenset({"", "off", "disabled", "none"})

# Anchor constants — the single definition of "where" each anchored cadence
# lands, imported by systemd_timer.parse_schedule for rendering so the
# rendered OnCalendar expression and previous_mark() below can never drift
# apart. WEEKLY_ANCHOR_LABEL is the systemd OnCalendar weekday token for
# WEEKLY_ANCHOR_WEEKDAY (Python's datetime.weekday(), 0=Monday) — the two
# must name the same day; there is only one weekly anchor so this is a
# constant pair, not a general weekday-to-label mapping.
DAILY_ANCHOR_HOUR = 3
DAILY_ANCHOR_MINUTE = 0
WEEKLY_ANCHOR_WEEKDAY = 0
WEEKLY_ANCHOR_HOUR = 0
WEEKLY_ANCHOR_MINUTE = 0
WEEKLY_ANCHOR_LABEL = "Mon"

# Bounded grammar — these match a finite set of accepted strings, not
# user-supplied free text. Parsed once on config load.  Each shape appears
# in exactly one pattern; the ``daily HH:MM`` idiom is a prefix strip in
# parse_schedule_atom, not a third pattern re-spelling the time.
_INTERVAL_RE = re.compile(r"^(?:every\s+)?(\d+)\s*([hm])$", re.IGNORECASE)
_HHMM_RE = re.compile(r"^(\d{1,2}:\d{2})$")


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


def _parse_hhmm(s: str) -> ParsedSchedule | None:
    """Parse a bare ``HH:MM`` time into an ``hhmm`` atom, else ``None``.

    THE only place a time is recognised — both the bare form and the
    ``"daily HH:MM"`` idiom land here, so the shape is matched and
    range-checked exactly once per input.  (The idiom previously had its
    own pattern, which matched the time, discarded the result, and left the
    bare-form branch to match it a second time.)
    """
    m = _HHMM_RE.match(s)
    if not m:
        return None
    hh_text, _, mm_text = m.group(1).partition(":")
    hh, mm = int(hh_text), int(mm_text)
    if 0 <= hh < 24 and 0 <= mm < 60:
        return ParsedSchedule(kind="hhmm", hh=hh, mm=mm)
    return None


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

    # ``"daily HH:MM"`` is an operator idiom for a daily schedule anchored
    # at HH:MM (the ``schedule: "daily 04:00"`` default in server.yaml).
    # Bare ``"daily"`` is its own atom, so the prefix is only consumed when
    # something follows it; and once consumed the remainder must be a time —
    # ``"daily off"`` is not a way to spell ``"off"``.
    if s[:5].lower() == "daily" and s[5:6].isspace():
        return _parse_hhmm(s[6:].lstrip())

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

    return _parse_hhmm(s)


def compute_schedule_period_seconds(schedule: str) -> int | None:
    """Return the full consolidation period in seconds for a schedule string.

    Schedule-grammar logic, not interim-adapter lifecycle logic:
    ``paramem.memory.interim_adapter`` and backup-schedule-aware code
    (``paramem.server.attention``, which reads ``security.backups.schedule``;
    ``paramem.backup.__main__``, which shares this module's dueness math)
    both depend on it, so it lives here rather than in ``interim_adapter`` —
    which never imports from ``paramem.backup``.

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


def _interval_modulus(unit: str) -> int:
    """Return the calendar modulus an interval unit divides: 24 for hours, 60 for minutes."""
    return 24 if unit == "h" else 60


def interval_is_exact(count: int, unit: str) -> bool:
    """True when an interval cadence divides its calendar modulus exactly.

    The single definition of "exact" for interval cadences — shared by
    :func:`previous_mark` and ``systemd_timer._period_heartbeat_calendar``
    (imported from there) so neither can drift from the other.
    """
    return _interval_modulus(unit) % count == 0


def non_exact_interval_grid(count: int, unit: str) -> int:
    """Coarsest divisor grid, in the cadence's own unit, for a non-exact interval.

    ``gcd(count, modulus)`` — the coarsest grid on which every period
    boundary of the cadence still lands, since *count* is by construction a
    multiple of this grid.  Shared by :func:`scheduled_run_stamp_value` (the
    heartbeat-floored stamp for non-exact intervals) and
    ``systemd_timer._period_heartbeat_calendar`` (the rendered heartbeat
    ``OnCalendar`` grid) — one gcd, two consumers, so the rendered timer and
    the server's dueness math always agree on which wakeups matter.
    """
    return math.gcd(count, _interval_modulus(unit))


def _non_exact_interval_grid_seconds(count: int, unit: str) -> int:
    """:func:`non_exact_interval_grid`, converted to seconds."""
    grid = non_exact_interval_grid(count, unit)
    return grid * 3600 if unit == "h" else grid * 60


def _is_due(
    last_attempt_epoch: float,
    period_seconds: int,
    now: float | None = None,
) -> bool:
    """Return True when at least ``period_seconds`` have elapsed since the last attempt.

    Module-internal: the only caller is :func:`scheduled_run_due`, which
    already resolves ``last_stamp is None`` to :attr:`ScheduleDueStatus.NO_STAMP`
    before reaching here, so *last_attempt_epoch* is always a real timestamp.
    Gates on the last ATTEMPT, never the last SUCCESS: a persistently failing
    run must not pass the gate on every heartbeat (that would be a retry
    storm), so callers of :func:`scheduled_run_due` pass an attempt timestamp
    that is written on both success and failure.
    """
    if now is None:
        now = time.time()
    return (now - last_attempt_epoch) >= period_seconds


# ---------------------------------------------------------------------------
# Calendar marks and dueness for ALL cadence kinds.
#
# previous_mark() is the single owner of "where does this cadence land on the
# wall clock" — anchored kinds (daily/weekly/HH:MM) and exact-divisor
# intervals all have real calendar marks; a non-exact interval does not (its
# period does not evenly tile the day), so it uses heartbeat-floored-stamp
# dueness instead (see scheduled_run_due below).
# There is deliberately no next_mark(): no current or planned caller needs
# "the next mark after now", only "the most recent mark at or before now".
# ---------------------------------------------------------------------------


def _floor_from_local_midnight(now: float, grid_seconds: int) -> float:
    """Floor an epoch timestamp DOWN to the current local-midnight-anchored grid boundary.

    The one definition of "floor to a same-day grid" — used directly for
    exact-divisor interval marks (:func:`previous_mark`) and for the
    heartbeat-floored stamp of non-exact intervals
    (:func:`scheduled_run_stamp_value`).

    Grid boundaries are computed in LOCAL time from local midnight, matching
    the LOCAL time base of rendered ``OnCalendar`` expressions — a UTC-epoch
    floor would only coincide with those LOCAL grid marks when the local UTC
    offset is a whole number of hours.

    Flooring (never rounding to nearest, never stamping raw wall-clock) is
    mandatory for the non-exact-interval caller: its stamp is written
    strictly AFTER the heartbeat fires (guards, debounce, orphan-session
    claim, migration branch all run first), so an un-floored stamp lands one
    instant past a grid mark — pushing the next due-check to the FOLLOWING
    heartbeat and silently inflating the effective period every cycle
    (``every 5h`` converges to a real 6h, ``every 48h`` to 49h). A tolerance
    window would only narrow that error, not eliminate it; flooring is exact.
    """
    if grid_seconds <= 0:
        raise ValueError(f"grid_seconds must be positive, got {grid_seconds}")
    dt = datetime.fromtimestamp(now)
    midnight = dt.replace(hour=0, minute=0, second=0, microsecond=0, fold=0)
    since_midnight = int((dt - midnight).total_seconds())
    floored_since_midnight = (since_midnight // grid_seconds) * grid_seconds
    return (midnight + timedelta(seconds=floored_since_midnight)).timestamp()


def _daily_mark(dt: datetime, hh: int, mm: int) -> float:
    """Return the most recent daily anchor at *hh*:*mm* at or before *dt*."""
    anchor_today = dt.replace(hour=hh, minute=mm, second=0, microsecond=0, fold=0)
    if dt >= anchor_today:
        return anchor_today.timestamp()
    return (anchor_today - timedelta(days=1)).timestamp()


def _weekly_mark(dt: datetime) -> float:
    """Return the most recent ``WEEKLY_ANCHOR_WEEKDAY``/``HOUR``/``MINUTE`` at or before *dt*."""
    days_since_anchor = (dt.weekday() - WEEKLY_ANCHOR_WEEKDAY) % 7
    anchor_date = dt - timedelta(days=days_since_anchor)
    anchor_dt = anchor_date.replace(
        hour=WEEKLY_ANCHOR_HOUR, minute=WEEKLY_ANCHOR_MINUTE, second=0, microsecond=0, fold=0
    )
    return anchor_dt.timestamp()


def previous_mark(schedule: str, now: float) -> float | None:
    """Return the most recent calendar mark at or before *now* for *schedule*.

    Marks exist for every anchored/exact-divisor cadence:

    * ``"hhmm"``      — daily at the parsed ``HH:MM``.
    * ``"daily"``     — daily at :data:`DAILY_ANCHOR_HOUR`:`DAILY_ANCHOR_MINUTE`.
    * ``"weekly"``    — :data:`WEEKLY_ANCHOR_LABEL` at
      :data:`WEEKLY_ANCHOR_HOUR`:`WEEKLY_ANCHOR_MINUTE`.
    * ``"interval"``  — only when ``interval_is_exact`` (24 % N == 0 for
      hours, 60 % N == 0 for minutes): the most recent N-hour/N-minute grid
      line since local midnight.

    Returns ``None`` for ``"off"``, unparseable input, and a non-exact
    interval cadence — none of these have a fixed wall-clock instant to
    report.  Non-exact intervals are not a degenerate case of this function;
    they use a different dueness strategy entirely (a heartbeat-floored
    stamp compared against a period, not a mark comparison) — see
    :func:`scheduled_run_due`.

    All computation is in naive local time (matches ``OnCalendar``
    semantics) — see the module docstring for the DST fall-back handling.
    """
    atom = parse_schedule_atom(schedule)
    if atom is None or atom.kind == "off":
        return None
    dt = datetime.fromtimestamp(now)
    if atom.kind == "hhmm":
        return _daily_mark(dt, atom.hh, atom.mm)
    if atom.kind == "daily":
        return _daily_mark(dt, DAILY_ANCHOR_HOUR, DAILY_ANCHOR_MINUTE)
    if atom.kind == "weekly":
        return _weekly_mark(dt)
    if atom.kind == "interval":
        if not interval_is_exact(atom.count, atom.unit):
            return None
        grid_seconds = atom.count * 3600 if atom.unit == "h" else atom.count * 60
        return _floor_from_local_midnight(now, grid_seconds)
    raise ValueError(f"Unhandled schedule kind: {atom.kind!r}")


class ScheduleDueStatus(str, Enum):
    """Three-way result of :func:`scheduled_run_due`.

    ``NO_STAMP`` is deliberately distinct from ``DUE``: whether an absent
    stamp means "seed it and skip this tick" (consolidation,
    ``server/app.py::_dispatch_consolidation``) or "run now" (the backup
    runner, ``backup/__main__.py`` — a first backup is cheap and welcome, so
    it must not be faked into looking already-run) is a caller policy, never
    a default this function should pick for them.
    """

    DUE = "due"
    NOT_DUE = "not_due"
    NO_STAMP = "no_stamp"


def scheduled_run_due(
    schedule: str,
    last_stamp: float | None,
    now: float | None = None,
) -> ScheduleDueStatus:
    """Return whether a scheduled run of *schedule* is due, given *last_stamp*.

    ``last_stamp`` is either a prior value returned by
    :func:`scheduled_run_stamp_value` for the same schedule (the
    consolidation cadence caller) or a raw last-attempt wall-clock timestamp
    such as ``backup.json``'s ``last_run.completed_at`` (both backup
    callers) — or ``None`` if no run has ever been stamped/attempted. Pure
    function, no I/O; callers own reading/writing the durable stamp.

    * ``last_stamp is None`` → :attr:`ScheduleDueStatus.NO_STAMP`. The caller
      decides the policy (seed-and-noop vs run-now); this function never
      guesses one.
    * ``schedule`` is ``"off"`` or unparseable → :attr:`ScheduleDueStatus.NOT_DUE`.
      There is no cadence to be due against.
    * Otherwise, for an anchored/exact-divisor cadence (:func:`previous_mark`
      returns a value): due iff that mark is strictly after *last_stamp* —
      the mark comparison absorbs an unfloored *last_stamp* without issue,
      since any value before the current mark reads as due regardless of
      alignment.
    * For a non-exact interval cadence (:func:`previous_mark` returns
      ``None``): due iff at least one period has elapsed since *last_stamp*
      (:func:`_is_due`), a plain elapsed-time comparison — it accepts either
      stamp origin, but only the :func:`scheduled_run_stamp_value`
      grid-floored stamp keeps successive dispatch times drift-free; a raw
      wall-clock *last_stamp* (as the backup callers pass) measures elapsed
      time from the exact moment of the last attempt instead.
    """
    if last_stamp is None:
        return ScheduleDueStatus.NO_STAMP
    if now is None:
        now = time.time()
    atom = parse_schedule_atom(schedule)
    if atom is None or atom.kind == "off":
        return ScheduleDueStatus.NOT_DUE
    mark = previous_mark(schedule, now)
    if mark is not None:
        return ScheduleDueStatus.DUE if mark > last_stamp else ScheduleDueStatus.NOT_DUE
    period_seconds = compute_schedule_period_seconds(schedule)
    return (
        ScheduleDueStatus.DUE
        if _is_due(last_stamp, period_seconds, now=now)
        else ScheduleDueStatus.NOT_DUE
    )


def scheduled_run_stamp_value(schedule: str, now: float) -> float:
    """Return the value to persist as the last-attempt stamp for a dispatch at *now*.

    * Anchored/exact-divisor cadences: the stamp is the mark itself
      (:func:`previous_mark`) rather than raw *now* — so a second dispatch
      inside the same mark's window is read as not-due by
      :func:`scheduled_run_due` (``mark > last_stamp`` is false when they're
      equal).
    * Non-exact interval cadences: the stamp is *now* floored to the
      coarsest calendar grid on which every period boundary still lands
      (:func:`_non_exact_interval_grid_seconds`) — preserving the
      drift-free dispatch spacing that flooring exists for (see
      :func:`_floor_from_local_midnight`).

    Raises
    ------
    ValueError
        If *schedule* is ``"off"`` or unparseable — there is no cadence to
        stamp a dispatch against. Callers dispatch only when a cadence is
        actually configured, so this signals a caller bug, not a runtime
        condition to handle.
    """
    atom = parse_schedule_atom(schedule)
    if atom is None or atom.kind == "off":
        raise ValueError(f"No scheduled-run stamp for schedule={schedule!r}")
    mark = previous_mark(schedule, now)
    if mark is not None:
        return mark
    grid_seconds = _non_exact_interval_grid_seconds(atom.count, atom.unit)
    return _floor_from_local_midnight(now, grid_seconds)
