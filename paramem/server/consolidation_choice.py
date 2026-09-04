"""Chooses which consolidation run a dispatch earns.

The decision (:func:`choose_consolidation_run`) is pure: every value it
reads — the pending event's head, the two schedule marks, the clock — is a
plain argument, and it touches no disk, no config object, and no server
state.

The pending-event head (:class:`PendingEvent`) that feeds the decision is
read off the stage ledger by ``_pending_event_head``
(``paramem/server/app.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from paramem.server.consolidation_action import ConsolidationAction
from paramem.server.schedule_grammar import (
    ScheduleDueStatus,
    next_mark,
    parse_schedule_atom,
    parse_window,
    scheduled_run_due,
)


class DispatchReason(str, Enum):
    """Why a consolidation dispatch is happening.

    Each member names one class of production source:

    * ``TIMER`` — the ``paramem-consolidate.timer`` systemd unit, hitting
      ``POST /scheduled-tick``. Its calendar entries cover both cadence
      marks and window starts; the decider tells the two apart from the
      clock, not from a flag the timer passes.
    * ``IDLE`` — the in-process idle watch, firing once the server has been
      idle for ``session.idle_timeout_minutes``.
    * ``BOOT`` — the boot-completion catch-up task, once at process start.
    * ``OPERATOR`` — the three consolidation doors (``POST /consolidate``,
      ``/consolidate/interim``, ``/reconsolidate``) and every
      ``/calibrate/*`` route: an operator asked directly.
    """

    TIMER = "timer"
    IDLE = "idle"
    BOOT = "boot"
    OPERATOR = "operator"


@dataclass(frozen=True)
class PendingEvent:
    """The head of a stage ledger found on disk.

    Attributes:
        kind: The pending event's own kind — ``"interim"``, ``"full"``, or
            ``"reconcile"`` — read off the ledger's ``event`` field.
            ``None`` when the ledger is present but this process cannot
            interpret it (``readable`` is ``False``).
        readable: ``True`` when the ledger parsed and its head fields could
            be read. ``False`` for a present-but-uninterpretable ledger —
            an event is still pending, just illegible right now.
        since_epoch: The extraction stage's ``completed_at``, as a Unix
            timestamp. ``None`` exactly when ``readable`` is ``False`` — a
            readable head always carries one, since ``read_ledger`` admits
            no ledger whose stages lack an extraction entry.
        cause: Why the ledger could not be interpreted, when ``readable``
            is ``False``. ``None`` otherwise, including whenever
            ``readable`` is ``True``.
    """

    kind: "str | None"
    readable: bool
    since_epoch: "float | None"
    cause: "str | None" = None


@dataclass(frozen=True)
class ConsolidationChoice:
    """What one consolidation dispatch earns.

    Attributes:
        run: The action to run, or ``None`` when nothing runs this
            dispatch.
        resume_pending: ``True`` when *run* finishes a pending event rather
            than starting a fresh one.
        status: The terminal status string when *run* is ``None`` — a
            ``"deferred_*"`` or ``"noop_*"`` value. Empty when *run* is
            set; the caller reports its own outcome status once the run
            (or resume) is actually dispatched.
        consumes_cadence_mark: Whether the caller should write the cadence
            mark this dispatch stood on.
        starts_full_fold: Whether the caller should record this instant as
            the last full-fold start. Written together with
            *consumes_cadence_mark*, in one call
            (:func:`~paramem.server.schedule_state.write_marks`), when a
            verdict carries both — a crash between two separate writes
            could otherwise strand the full-start mark stale and let a
            second full fold start inside the same window opening.
        next_opportunity_seconds: For a ``"deferred_resume_waiting"``
            *status*, the seconds until the next opportunity to resume.
            ``None`` otherwise.
        next_opportunity_reason: The :class:`DispatchReason` that owns the
            next opportunity named by *next_opportunity_seconds*. ``None``
            otherwise.
    """

    run: "ConsolidationAction | None"
    resume_pending: bool
    status: str
    consumes_cadence_mark: bool
    starts_full_fold: bool
    next_opportunity_seconds: "int | None"
    next_opportunity_reason: "DispatchReason | None"


#: The action a resumed event's ledger ``kind`` maps to. ``"full"`` and
#: ``"reconcile"`` both name a full-topology fold, so both follow the full
#: event's resume rule below (idle, unconditionally — the window never
#: gates a resume).
_RESUME_ACTION_FOR_KIND: "dict[str, ConsolidationAction]" = {
    "interim": ConsolidationAction.INTERIM,
    "full": ConsolidationAction.FULL,
    "reconcile": ConsolidationAction.RECONCILE,
}

#: Ledger kinds whose resume rule is "idle, whichever firing asks, even if
#: the window closed meanwhile" — a full-topology fold, resumed or reconcile.
_ALWAYS_RESUMABLE_KINDS = frozenset({"full", "reconcile"})

_CADENCE_REASONS = frozenset({DispatchReason.TIMER, DispatchReason.BOOT})


def _direct_run(
    action: "ConsolidationAction",
    *,
    resume_pending: bool = False,
    consumes_cadence_mark: bool = False,
    starts_full_fold: bool = False,
) -> ConsolidationChoice:
    """Build the choice for a run that dispatches now."""
    return ConsolidationChoice(
        run=action,
        resume_pending=resume_pending,
        status="",
        consumes_cadence_mark=consumes_cadence_mark,
        starts_full_fold=starts_full_fold,
        next_opportunity_seconds=None,
        next_opportunity_reason=None,
    )


def _noop(status: str, *, consumes_cadence_mark: bool = False) -> ConsolidationChoice:
    """Build the choice for a named noop — nothing to run this dispatch."""
    return ConsolidationChoice(
        run=None,
        resume_pending=False,
        status=status,
        consumes_cadence_mark=consumes_cadence_mark,
        starts_full_fold=False,
        next_opportunity_seconds=None,
        next_opportunity_reason=None,
    )


def _deferral(seconds: float, owner: DispatchReason) -> ConsolidationChoice:
    """Build a ``deferred_resume_waiting`` choice naming the next opportunity."""
    return ConsolidationChoice(
        run=None,
        resume_pending=False,
        status="deferred_resume_waiting",
        consumes_cadence_mark=False,
        starts_full_fold=False,
        next_opportunity_seconds=max(0, int(seconds)),
        next_opportunity_reason=owner,
    )


def _cadence_tick_status(
    cadence: str, last_cadence_mark: "float | None", now: float
) -> "ScheduleDueStatus | None":
    """The cadence's own dueness reading for *now*, or ``None`` for an off
    or unparseable cadence — which has no ticks to be due against at all.

    Delegates to :func:`~paramem.server.schedule_grammar.scheduled_run_due`,
    the single dueness reading for both mark-bearing cadences (anchored,
    exact-divisor intervals) and non-exact intervals (elapsed time on the
    interval's own heartbeat grid) alike — a non-exact interval (e.g.
    ``"7h"``) has no wall-clock mark for :func:`~paramem.server.schedule_grammar.previous_mark`
    to answer, but it is still due on its own schedule, and only
    ``scheduled_run_due`` reads that.
    """
    atom = parse_schedule_atom(cadence)
    if atom is None or atom.kind == "off":
        return None
    return scheduled_run_due(cadence, last_cadence_mark, now=now)


def _interim_resume_allowed(
    interim_resume: str,
    cadence: str,
    now: float,
    since_epoch: float,
    now_dt: datetime,
) -> bool:
    """Whether an idle-owned firing may resume a pending interim event now.

    *since_epoch* is always a real timestamp here, never ``None`` —
    :class:`PendingEvent`'s own invariant: a caller reaches this only after
    confirming ``pending.readable``, and a readable head always carries one.
    """
    if interim_resume == "immediate":
        return True
    if interim_resume == "tick":
        mark = next_mark(cadence, since_epoch)
        return mark is not None and mark <= now
    return parse_window(interim_resume).contains(now_dt)


def _resume_allowed(
    kind: str,
    interim_resume: str,
    cadence: str,
    now: float,
    since_epoch: float,
    now_dt: datetime,
) -> bool:
    """Whether the pending event's own resume rule is met now."""
    if kind in _ALWAYS_RESUMABLE_KINDS:
        return True
    return _interim_resume_allowed(interim_resume, cadence, now, since_epoch, now_dt)


def _resume_deferral(
    pending: PendingEvent,
    interim_resume: str,
    cadence: str,
    now: float,
    now_dt: datetime,
    seconds_until_idle: int,
) -> ConsolidationChoice:
    """The deferral a firing earns when the pending event's resume rule is not met.

    A conversation still open owns the next opportunity (``IDLE``); an idle
    server waiting on a cadence mark or a window start owns it via
    ``TIMER`` — the firing that will actually supply it.
    """
    if seconds_until_idle > 0:
        return _deferral(seconds_until_idle, DispatchReason.IDLE)
    if interim_resume == "tick":
        mark = next_mark(cadence, pending.since_epoch)
        seconds = (mark - now) if mark is not None else 0
        return _deferral(seconds, DispatchReason.TIMER)
    window = parse_window(interim_resume)
    seconds = window.next_start(now_dt).timestamp() - now
    return _deferral(seconds, DispatchReason.TIMER)


def choose_consolidation_run(
    *,
    requested: "ConsolidationAction",
    reason: "DispatchReason",
    pending: "PendingEvent | None",
    interim_resume: str,
    full_window: str,
    cadence: str,
    now: float,
    last_cadence_mark: "float | None",
    last_full_start: "float | None",
    seconds_until_idle: int,
    full_fold_deadline: "float | None",
    max_interim_count: int,
) -> ConsolidationChoice:
    """Decide which consolidation run this dispatch earns. Pure; no I/O.

    Resolution order:

    1. A non-staging *requested* action (``CALIBRATE``, ``CALIBRATE_PENDING``)
       runs directly — it stages nothing and never touches *pending*.
    2. *pending* present and unreadable → ``"deferred_event_unreadable"``.
    3. *pending* present and readable → resumed when its own resume rule is
       met: an ``OPERATOR`` dispatch resumes at once; ``TIMER``/``BOOT``/
       ``IDLE`` resume only while idle (``seconds_until_idle == 0``) and
       under the kind's own rule — a full or reconcile event resumes
       unconditionally, an interim event under ``interim_resume``
       (``"immediate"`` always, ``"tick"`` once a cadence mark has passed
       since it became pending, a window once *now* falls inside it).
       Otherwise a ``"deferred_resume_waiting"`` naming the next
       opportunity and the reason that owns it. A resume consumes the
       cadence mark only when a cadence reason (``TIMER``/``BOOT``) fired
       it on a due cadence tick (:func:`_cadence_tick_status` is
       :attr:`~paramem.server.schedule_grammar.ScheduleDueStatus.DUE`); a
       window-start or ``IDLE`` resume consumes none. *pending*, resumed or
       deferred, ends the resolution here — nothing below considers a
       pending event.
    4. A staging *requested* action other than ``AUTO`` (``FULL``,
       ``INTERIM``, ``RECONCILE``) runs directly, ignoring the window and
       the cadence marks — an operator's own door.
    5. ``AUTO`` with *reason* ``IDLE`` → ``"noop_nothing_pending"``: the
       idle firing finishes a pending event and never starts one, so it
       never reaches the schedule's own resolutions below — seeding a
       never-stamped cadence, a full fold in its window, an overdue fold
       with the window shut, an interim on a due tick — which belong to
       ``TIMER``/``BOOT`` alone; an ``AUTO`` request under any other reason
       answers ``"noop_not_due"`` here without touching the cadence or the
       window.
    6. ``AUTO`` with *reason* ``TIMER``/``BOOT``, on a cadence that actually
       schedules (not off/unparseable) and has never been stamped
       (:func:`_cadence_tick_status` is
       :attr:`~paramem.server.schedule_grammar.ScheduleDueStatus.NO_STAMP`)
       → ``"noop_scheduler_seeded"`` (consumes the cadence mark — the
       caller seeds the stamp file). An off/unparseable cadence has no
       ticks to seed against and skips straight to the full-fold-in-its-window
       resolution.
    7. ``AUTO`` with a ring (``max_interim_count > 0``): a full fold starts
       when *now* is inside `full_window`, *full_fold_deadline* falls
       before the next window opening, and no full fold has started in
       this opening — independent of the cadence, so a ring an operator
       has since turned the cadence off on still drains at the window.
    8. With a ring, when the fold is already due (*full_fold_deadline* at
       or before *now*) and *now* falls outside `full_window` — an
       overdue fold with the window shut — the dispatch noops
       (``"noop_outside_window"``) rather than minting another interim
       slot: interim capacity cannot overflow because a full fold was not
       allowed to run.
    9. On a due cadence tick (:func:`_cadence_tick_status` is ``DUE``) →
       ``INTERIM`` with a ring, ``FULL`` (and ``starts_full_fold``) without
       one — every due tick is a full-fold tick at ``max_interim_count ==
       0``. An off/unparseable cadence is never due, so neither ever fires
       from it.
    10. Otherwise ``"noop_not_due"``.

    Args:
        requested: The action the door asked for.
        reason: Why this dispatch is happening.
        pending: The pending event's head, or ``None`` when nothing is
            pending.
        interim_resume: ``"immediate"``, ``"tick"``, or an
            ``"HH:MM-HH:MM"`` window — when an interrupted interim event
            resumes.
        full_window: The ``"HH:MM-HH:MM"`` daily window a full fold may
            start in.
        cadence: The interim cadence string (``consolidation.refresh_cadence``).
        now: ``time.time()``, read once by the caller and reused for every
            comparison this decision makes.
        last_cadence_mark: The last cadence mark this process consumed, or
            ``None`` when never stamped.
        last_full_start: The instant the last full fold started, or
            ``None`` when never started.
        seconds_until_idle: Seconds remaining before the server is judged
            idle; ``0`` means idle now.
        full_fold_deadline: The epoch at which the oldest payload-bearing
            interim slot reaches ``refresh_cadence × max_interim_count`` of
            age, or ``None`` when there is no such slot, no period, or no
            ring (``max_interim_count == 0``).
        max_interim_count: The operator's ring size. ``0`` means no ring —
            the cadence itself is the full-fold schedule and neither
            *full_window* nor *full_fold_deadline* is read.

    Returns:
        The :class:`ConsolidationChoice` this dispatch earns.
    """
    now_dt = datetime.fromtimestamp(now)

    # Step 1: a non-staging action stages nothing and never touches a
    # pending ledger.
    if not requested.stages_event:
        return _direct_run(requested)

    # Steps 2-3: a pending ledger ends the resolution here, resumed or
    # deferred.
    if pending is not None:
        if not pending.readable:
            return _noop("deferred_event_unreadable")

        resumed_action = _RESUME_ACTION_FOR_KIND[pending.kind]
        if reason is DispatchReason.OPERATOR:
            resume_ok = True
        elif seconds_until_idle > 0:
            resume_ok = False
        else:
            resume_ok = _resume_allowed(
                pending.kind, interim_resume, cadence, now, pending.since_epoch, now_dt
            )

        if resume_ok:
            consumes = False
            if reason in _CADENCE_REASONS:
                consumes = (
                    _cadence_tick_status(cadence, last_cadence_mark, now) is ScheduleDueStatus.DUE
                )
            return _direct_run(resumed_action, resume_pending=True, consumes_cadence_mark=consumes)

        return _resume_deferral(pending, interim_resume, cadence, now, now_dt, seconds_until_idle)

    # Step 4: an operator staging action with nothing pending runs directly.
    if requested is not ConsolidationAction.AUTO:
        return _direct_run(requested)

    # requested is AUTO and pending is None from here — the schedule's own
    # business.
    if reason is DispatchReason.IDLE:
        return _noop("noop_nothing_pending")

    # The schedule's own resolutions -- seeding a never-stamped cadence, a
    # full fold in its window, an overdue fold with the window shut, an
    # interim on a due tick -- belong to TIMER/BOOT alone: the cadence and
    # the window are that pair's business, not an arbitrary caller's. An
    # AUTO request under another reason answers noop_not_due.
    if reason not in _CADENCE_REASONS:
        return _noop("noop_not_due")

    tick_status = _cadence_tick_status(cadence, last_cadence_mark, now)
    if tick_status is ScheduleDueStatus.NO_STAMP:
        return _noop("noop_scheduler_seeded", consumes_cadence_mark=True)
    standing = tick_status is ScheduleDueStatus.DUE

    if max_interim_count > 0:
        window = parse_window(full_window)
        current_start = window.current_start(now_dt)
        if (
            current_start is not None
            and full_fold_deadline is not None
            and full_fold_deadline < window.next_start(now_dt).timestamp()
            and (last_full_start is None or last_full_start < current_start.timestamp())
        ):
            return _direct_run(
                ConsolidationAction.FULL, consumes_cadence_mark=standing, starts_full_fold=True
            )

        if current_start is None and full_fold_deadline is not None and full_fold_deadline <= now:
            return _noop("noop_outside_window")

        if standing:
            return _direct_run(ConsolidationAction.INTERIM, consumes_cadence_mark=True)

        return _noop("noop_not_due")

    # max_interim_count == 0: the cadence itself is the full-fold schedule.
    if standing:
        return _direct_run(
            ConsolidationAction.FULL, consumes_cadence_mark=True, starts_full_fold=True
        )
    return _noop("noop_not_due")
