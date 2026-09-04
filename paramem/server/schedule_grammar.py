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

Regex usage here is a declared-syntax matcher, admitted under
``ARCHITECTURE.md``, AD-22: Regex Confined to Declared Syntax, and pinned
by ``tests/test_regex_confinement.py``.  The admissibility rule is
structural, not a judgement about this module: each
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
    range-checked exactly once per input.
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
# ---------------------------------------------------------------------------


def _floor_from_local_midnight_dt(dt: datetime, grid_seconds: int) -> datetime:
    """Floor a naive local datetime DOWN to its local-midnight-anchored grid boundary.

    The one definition of "floor to a same-day grid", in naive-local-time
    space — the shared core :func:`_floor_from_local_midnight` (the epoch
    view) and :func:`_previous_mark_dt` (an exact-divisor interval mark)
    both build on.

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
    midnight = dt.replace(hour=0, minute=0, second=0, microsecond=0, fold=0)
    since_midnight = int((dt - midnight).total_seconds())
    floored_since_midnight = (since_midnight // grid_seconds) * grid_seconds
    return (midnight + timedelta(seconds=floored_since_midnight)).replace(fold=0)


def _floor_from_local_midnight(now: float, grid_seconds: int) -> float:
    """Floor an epoch timestamp DOWN to the current local-midnight-anchored grid boundary.

    Epoch-in/epoch-out view of :func:`_floor_from_local_midnight_dt` — used
    directly for the heartbeat-floored stamp of non-exact intervals
    (:func:`scheduled_run_stamp_value`).
    """
    return _floor_from_local_midnight_dt(datetime.fromtimestamp(now), grid_seconds).timestamp()


def _daily_mark_dt(dt: datetime, hh: int, mm: int) -> datetime:
    """Return the most recent daily anchor at *hh*:*mm* at or before *dt*, as naive local time."""
    anchor_today = dt.replace(hour=hh, minute=mm, second=0, microsecond=0, fold=0)
    if dt >= anchor_today:
        return anchor_today
    return anchor_today - timedelta(days=1)


def _weekly_mark_dt(dt: datetime) -> datetime:
    """Return the most recent weekly anchor at or before *dt*, as naive local time.

    Anchor is :data:`WEEKLY_ANCHOR_WEEKDAY`/:data:`WEEKLY_ANCHOR_HOUR`/
    :data:`WEEKLY_ANCHOR_MINUTE`.
    """
    days_since_anchor = (dt.weekday() - WEEKLY_ANCHOR_WEEKDAY) % 7
    anchor_date = dt - timedelta(days=days_since_anchor)
    return anchor_date.replace(
        hour=WEEKLY_ANCHOR_HOUR, minute=WEEKLY_ANCHOR_MINUTE, second=0, microsecond=0, fold=0
    )


def _previous_mark_dt(schedule: str, dt: datetime) -> datetime | None:
    """Return the most recent calendar-mark instant at or before *dt*, as naive local time.

    The one per-kind mark placement — :func:`previous_mark` (the epoch view)
    and :func:`next_mark` (which steps this same naive face forward by a
    period, never by raw epoch seconds) both build on it, so the two can
    never place a mark differently. Parses *schedule* once.

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
    """
    atom = parse_schedule_atom(schedule)
    if atom is None or atom.kind == "off":
        return None
    if atom.kind == "hhmm":
        return _daily_mark_dt(dt, atom.hh, atom.mm)
    if atom.kind == "daily":
        return _daily_mark_dt(dt, DAILY_ANCHOR_HOUR, DAILY_ANCHOR_MINUTE)
    if atom.kind == "weekly":
        return _weekly_mark_dt(dt)
    if atom.kind == "interval":
        if not interval_is_exact(atom.count, atom.unit):
            return None
        grid_seconds = atom.count * 3600 if atom.unit == "h" else atom.count * 60
        return _floor_from_local_midnight_dt(dt, grid_seconds)
    raise ValueError(f"Unhandled schedule kind: {atom.kind!r}")


def previous_mark(schedule: str, now: float) -> float | None:
    """Return the most recent calendar mark at or before *now* for *schedule*.

    Epoch-in/epoch-out view of :func:`_previous_mark_dt`, the module's one
    per-kind mark placement.

    All computation is in naive local time (matches ``OnCalendar``
    semantics) — see the module docstring for the DST fall-back handling.
    """
    mark_dt = _previous_mark_dt(schedule, datetime.fromtimestamp(now))
    return mark_dt.timestamp() if mark_dt is not None else None


def next_mark(schedule: str, now: float) -> float | None:
    """Return the next calendar mark strictly after *now* for *schedule*.

    Starts from the same naive-local mark face :func:`previous_mark` places
    (:func:`_previous_mark_dt`) and steps it forward by the cadence's period,
    one period at a time, entirely in naive-local-time arithmetic — a
    calendar ``timedelta`` added to a wall-clock face, never a raw
    epoch-seconds sum, since a local day is 23 or 25 hours long across a DST
    transition and epoch-second stepping would walk off the wall clock the
    anchored kinds and exact-divisor intervals are defined on.

    A single step can still land at or behind *now*: the naive face the
    stepping starts from always resolves an ambiguous fall-back instant to
    its first (``fold=0``) occurrence — the module's one fold convention —
    so when *now* itself falls in the repeated hour's second lap, one period
    past the mark can still be no later than *now*. The loop keeps stepping
    until the candidate's real instant is strictly after *now*, bounded by
    the number of marks a cadence can place inside its own repeating
    calendar unit (a day, or a week for ``"weekly"``) — a bound no cadence's
    genuine DST correction can exceed, so exhausting it signals a caller bug
    rather than a real schedule.

    Returns ``None`` for ``"off"``, unparseable input, and a non-exact
    interval cadence — the same cases :func:`previous_mark` returns ``None``
    for, since a non-exact interval has no wall-clock marks to be "next" from.
    """
    mark_dt = _previous_mark_dt(schedule, datetime.fromtimestamp(now))
    if mark_dt is None:
        return None
    period_seconds = compute_schedule_period_seconds(schedule)
    step = timedelta(seconds=period_seconds)
    modulus_seconds = 604800 if period_seconds == 604800 else 86400
    max_steps = modulus_seconds // period_seconds + 1
    candidate = (mark_dt + step).replace(fold=0)
    steps = 1
    while candidate.timestamp() <= now:
        if steps >= max_steps:
            raise RuntimeError(
                f"next_mark({schedule!r}, {now!r}) did not clear 'now' within "
                f"{max_steps} steps — this signals a caller bug, not a real schedule"
            )
        candidate = (candidate + step).replace(fold=0)
        steps += 1
    return candidate.timestamp()


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


# ---------------------------------------------------------------------------
# Time-of-day windows.
#
# Window and parse_window are the single definition of a daily opening in the
# codebase: quiet hours and any other HH:MM-bounded window both build one, so
# there is exactly one containment/next-opening rule to get right.
# ---------------------------------------------------------------------------


class InvalidWindow(ValueError):
    """The two bounds name no daily opening.

    Raised for a bound outside ``0..1439`` and for ``start == end``, which
    names no opening at all rather than an always-open or never-open one. The
    message names both bounds and the rule they broke.
    """


def _window_minutes_from_hhmm(field: str, value: str) -> int:
    """Parse one ``HH:MM`` window bound, raising :class:`InvalidWindow` on a bad shape.

    Delegates shape and range checking entirely to :func:`_parse_hhmm` — the
    module's one ``HH:MM`` recognizer — so a window bound and a cadence
    anchor are held to the same grammar. A non-``str`` *value* (an unset
    YAML key parsing to ``None``, an unquoted ``HH:MM`` parsing to an int)
    is refused here rather than reaching the regex, so every config call
    site inherits :class:`InvalidWindow` with no widened ``except``.
    """
    if not isinstance(value, str):
        raise InvalidWindow(f"window {field} must be HH:MM text, got {value!r}")
    atom = _parse_hhmm(value)
    if atom is None:
        raise InvalidWindow(f"window {field} must be HH:MM, got {value!r}")
    return atom.hh * 60 + atom.mm


@dataclass(frozen=True)
class Window:
    """A positive-length half-open daily time-of-day window, naive local time.

    Two bounds in minutes since local midnight; wraps past midnight when
    ``end_minute < start_minute``. ``__post_init__`` raises
    :class:`InvalidWindow`, so no ``Window`` anywhere holds an empty opening
    and every construction path — text via :func:`parse_window`, a config
    pair of ``HH:MM`` fields via :meth:`from_hhmm` — is held to the one rule.
    One meaning of the type project-wide, quiet hours included.
    """

    start_minute: int
    end_minute: int

    def __post_init__(self) -> None:
        for field, minute in (
            ("start_minute", self.start_minute),
            ("end_minute", self.end_minute),
        ):
            if not 0 <= minute < 1440:
                raise InvalidWindow(f"window {field} must be within 0..1439, got {minute}")
        if self.start_minute == self.end_minute:
            raise InvalidWindow(
                "window start and end must differ (an equal pair names no "
                f"opening): start_minute={self.start_minute}, end_minute={self.end_minute}"
            )

    @classmethod
    def from_hhmm(cls, start: str, end: str) -> "Window":
        """Build a window from two ``HH:MM`` bounds (the config-pair path, e.g. quiet hours).

        Raises :class:`InvalidWindow` naming the field and the value for a
        bound that is not ``HH:MM`` in this module's own shape (a one- or
        two-digit hour, two-digit minutes — ``"9:5"`` is refused), and for a
        pair that names no opening.
        """
        return cls(
            _window_minutes_from_hhmm("start", start),
            _window_minutes_from_hhmm("end", end),
        )

    def _start_on(self, now: datetime) -> datetime:
        """Return the opening's start-of-day instant on *now*'s calendar date.

        The one "today's start" arithmetic in the module — :meth:`current_start`
        and :meth:`next_start` both place their candidate on it rather than
        re-deriving the ``.replace(...)`` themselves.
        """
        return now.replace(
            hour=self.start_minute // 60,
            minute=self.start_minute % 60,
            second=0,
            microsecond=0,
            fold=0,
        )

    def current_start(self, now: datetime | None = None) -> datetime | None:
        """Return the start instant of the opening containing *now*, or ``None``.

        The one containment arithmetic in the module — :meth:`contains` is
        built on this. On a wrapping window inside the small hours (e.g.
        ``23:00-02:00`` at 01:00) this is yesterday's start.
        """
        now = now if now is not None else datetime.now()
        cur = now.hour * 60 + now.minute
        start_today = self._start_on(now)
        if self.start_minute < self.end_minute:
            if self.start_minute <= cur < self.end_minute:
                return start_today
            return None
        # Wrapping: the opening spans from start_minute through midnight to
        # end_minute the following calendar day.
        if cur >= self.start_minute:
            return start_today
        if cur < self.end_minute:
            return start_today - timedelta(days=1)
        return None

    def contains(self, now: datetime | None = None) -> bool:
        """Whether *now* falls inside an opening — ``current_start(now) is not None``."""
        return self.current_start(now) is not None

    def next_start(self, now: datetime | None = None) -> datetime:
        """Return the first opening strictly after *now*.

        Standing inside an opening this is the next day's, never the one
        *now* is in.
        """
        now = now if now is not None else datetime.now()
        cur = now.hour * 60 + now.minute
        start_today = self._start_on(now)
        if cur < self.start_minute:
            return start_today
        return start_today + timedelta(days=1)

    @property
    def start_hhmm(self) -> str:
        """The start bound rendered back as ``HH:MM``."""
        return f"{self.start_minute // 60:02d}:{self.start_minute % 60:02d}"


def parse_window(text: str) -> Window:
    """Build the window ``"HH:MM-HH:MM"`` names.

    Raises :class:`InvalidWindow`, naming the form and the value received,
    for a non-``str`` payload (an unset YAML key parses to ``None``), for
    text that is not two ``HH:MM`` bounds separated by ``"-"``, and for
    bounds that name no opening. Delegates to :meth:`Window.from_hhmm` so one
    grammar covers cadences and windows alike.
    """
    if not isinstance(text, str):
        raise InvalidWindow(f'window must be "HH:MM-HH:MM" text, got {text!r}')
    start, sep, end = text.partition("-")
    if not sep:
        raise InvalidWindow(f'window must be "HH:MM-HH:MM", got {text!r}')
    return Window.from_hhmm(start, end)
