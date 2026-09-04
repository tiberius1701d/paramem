"""``choose_consolidation_run`` — the full decision table.

Pure function, no I/O: every case here builds its own inputs and reads the
returned :class:`~paramem.server.consolidation_choice.ConsolidationChoice`
directly. The process timezone is pinned to Europe/Berlin for every test in
this module (the decider converts ``now`` to naive local time internally),
restored after each test. Every ``now``/``since_epoch`` value is built
*inside* a test body via :func:`_at`, never at collection time, since a
module-level call would run under whatever timezone was active when pytest
collected this file rather than under the pinned one.
"""

from __future__ import annotations

import time
from datetime import datetime

import pytest

from paramem.server.consolidation_action import ConsolidationAction
from paramem.server.consolidation_choice import (
    ConsolidationChoice,
    DispatchReason,
    PendingEvent,
    choose_consolidation_run,
)
from paramem.server.schedule_grammar import next_mark, parse_window

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def berlin_timezone(monkeypatch):
    """Pin the process timezone to Europe/Berlin for this module's window and
    cadence arithmetic, restored (env and C library state alike) after."""
    monkeypatch.setenv("TZ", "Europe/Berlin")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


def _at(hour: int, minute: int, *, day: int = 6, month: int = 1, year: int = 2026) -> float:
    """A naive-local epoch at ``HH:MM`` on a fixed winter date (no DST ambiguity)."""
    return datetime(year, month, day, hour, minute).timestamp()


def _choice(**overrides) -> ConsolidationChoice:
    """``choose_consolidation_run`` with every keyword defaulted so a test
    states only what it varies. Defaults are built here, inside the function
    body, so ``_at(...)`` runs under the pinned timezone."""
    defaults = dict(
        requested=ConsolidationAction.AUTO,
        reason=DispatchReason.TIMER,
        pending=None,
        interim_resume="immediate",
        full_window="01:00-04:00",
        cadence="12h",
        now=_at(12, 0),
        last_cadence_mark=None,
        last_full_start=None,
        seconds_until_idle=0,
        full_fold_deadline=None,
        max_interim_count=7,
    )
    defaults.update(overrides)
    return choose_consolidation_run(**defaults)


def _due_mark(now: float) -> float:
    """A ``last_cadence_mark`` old enough that any real cadence's own mark is due."""
    return now - 100 * 86400


def _not_due_mark(now: float) -> float:
    """A ``last_cadence_mark`` that can never be exceeded by ``previous_mark(cadence, now)``,
    so the cadence reads NOT_DUE regardless of the cadence's own grid."""
    return now


# ---------------------------------------------------------------------------
# A pending FULL/RECONCILE event resumes unconditionally once idle; a
# pending reconcile resumes under the same rule as a pending full.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reason", [DispatchReason.TIMER, DispatchReason.IDLE, DispatchReason.BOOT])
@pytest.mark.parametrize("now_hhmm", [(2, 0), (12, 0)], ids=["inside_window", "outside_window"])
def test_pending_full_resumes_on_every_reason_at_idle_inside_and_outside_window(reason, now_hhmm):
    now = _at(*now_hhmm)
    pending = PendingEvent(kind="full", readable=True, since_epoch=_at(1, 0))
    choice = _choice(reason=reason, pending=pending, now=now, seconds_until_idle=0)
    assert choice.run is ConsolidationAction.FULL
    assert choice.resume_pending is True
    assert choice.status == ""


def test_operator_dispatch_resumes_a_pending_full_event_at_once():
    choice = _choice(
        requested=ConsolidationAction.FULL,
        reason=DispatchReason.OPERATOR,
        pending=PendingEvent(kind="full", readable=True, since_epoch=_at(1, 0)),
        now=_at(12, 0),
        seconds_until_idle=0,
    )
    assert choice.run is ConsolidationAction.FULL
    assert choice.resume_pending is True


@pytest.mark.parametrize("now_hhmm", [(2, 0), (12, 0)], ids=["inside_window", "outside_window"])
def test_pending_reconcile_resumes_like_full_regardless_of_window(now_hhmm):
    now = _at(*now_hhmm)
    pending = PendingEvent(kind="reconcile", readable=True, since_epoch=_at(1, 0))
    choice = _choice(reason=DispatchReason.TIMER, pending=pending, now=now, seconds_until_idle=0)
    assert choice.run is ConsolidationAction.RECONCILE
    assert choice.resume_pending is True


# ---------------------------------------------------------------------------
# A pending interim event under "immediate" resumes at idle on every one of
# the server's own reasons.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reason", [DispatchReason.IDLE, DispatchReason.TIMER, DispatchReason.BOOT])
def test_pending_interim_immediate_resumes_at_idle_on_every_own_reason(reason):
    now = _at(12, 0)
    pending = PendingEvent(kind="interim", readable=True, since_epoch=_at(1, 0))
    choice = _choice(
        reason=reason,
        pending=pending,
        interim_resume="immediate",
        now=now,
        seconds_until_idle=0,
        last_cadence_mark=_not_due_mark(now),
    )
    assert choice.run is ConsolidationAction.INTERIM
    assert choice.resume_pending is True


# ---------------------------------------------------------------------------
# An OPERATOR dispatch resumes at once regardless of kind, policy, idle
# state, or window.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind,action",
    [
        ("full", ConsolidationAction.FULL),
        ("interim", ConsolidationAction.INTERIM),
        ("reconcile", ConsolidationAction.RECONCILE),
    ],
)
@pytest.mark.parametrize("interim_resume", ["immediate", "tick", "01:00-04:00"])
@pytest.mark.parametrize("seconds_until_idle", [0, 500])
@pytest.mark.parametrize("now_hhmm", [(2, 0), (12, 0)], ids=["inside_window", "outside_window"])
def test_operator_dispatch_resumes_a_pending_event_at_once(
    kind, action, interim_resume, seconds_until_idle, now_hhmm
):
    now = _at(*now_hhmm)
    pending = PendingEvent(kind=kind, readable=True, since_epoch=_at(1, 0))
    choice = _choice(
        requested=action,
        reason=DispatchReason.OPERATOR,
        pending=pending,
        interim_resume=interim_resume,
        now=now,
        seconds_until_idle=seconds_until_idle,
        last_cadence_mark=_not_due_mark(now),
    )
    assert choice.run is action
    assert choice.resume_pending is True
    assert choice.status == ""


# ---------------------------------------------------------------------------
# A pending interim event under "tick" resumes once a cadence mark has
# passed since it became pending, on any of the server's own reasons, and
# defers naming the seconds to that mark otherwise.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reason", [DispatchReason.TIMER, DispatchReason.BOOT, DispatchReason.IDLE])
def test_pending_interim_tick_resumes_once_a_cadence_mark_has_passed(reason):
    cadence = "12h"
    since = _at(1, 0)
    mark = next_mark(cadence, since)
    now = mark + 1
    pending = PendingEvent(kind="interim", readable=True, since_epoch=since)
    choice = _choice(
        reason=reason,
        pending=pending,
        interim_resume="tick",
        cadence=cadence,
        now=now,
        seconds_until_idle=0,
    )
    assert choice.run is ConsolidationAction.INTERIM
    assert choice.resume_pending is True


def test_pending_interim_tick_defers_before_the_mark_naming_seconds_to_it():
    cadence = "12h"
    since = _at(1, 0)
    mark = next_mark(cadence, since)
    now = mark - 5
    pending = PendingEvent(kind="interim", readable=True, since_epoch=since)
    choice = _choice(
        reason=DispatchReason.TIMER,
        pending=pending,
        interim_resume="tick",
        cadence=cadence,
        now=now,
        seconds_until_idle=0,
    )
    assert choice.status == "deferred_resume_waiting"
    assert choice.next_opportunity_reason is DispatchReason.TIMER
    assert choice.next_opportunity_seconds == 5


# ---------------------------------------------------------------------------
# A pending interim event under a window resumes when now is inside it, and
# defers naming the seconds to the next window start otherwise.
# ---------------------------------------------------------------------------


def test_pending_interim_window_resumes_when_now_is_inside_it():
    now = _at(2, 0)  # inside 01:00-04:00
    pending = PendingEvent(kind="interim", readable=True, since_epoch=_at(1, 0))
    choice = _choice(
        reason=DispatchReason.IDLE,
        pending=pending,
        interim_resume="01:00-04:00",
        now=now,
        seconds_until_idle=0,
        last_cadence_mark=_not_due_mark(now),
    )
    assert choice.run is ConsolidationAction.INTERIM
    assert choice.resume_pending is True


def test_pending_interim_window_defers_naming_seconds_to_next_window_start():
    now = _at(12, 0)  # outside 01:00-04:00
    pending = PendingEvent(kind="interim", readable=True, since_epoch=_at(1, 0))
    choice = _choice(
        reason=DispatchReason.TIMER,
        pending=pending,
        interim_resume="01:00-04:00",
        now=now,
        seconds_until_idle=0,
        last_cadence_mark=_not_due_mark(now),
    )
    assert choice.status == "deferred_resume_waiting"
    assert choice.next_opportunity_reason is DispatchReason.TIMER
    window = parse_window("01:00-04:00")
    expected = int(window.next_start(datetime.fromtimestamp(now)).timestamp() - now)
    assert choice.next_opportunity_seconds == expected


# ---------------------------------------------------------------------------
# A scheduled firing landing mid-conversation never resumes, whatever its
# policy would otherwise allow.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reason", [DispatchReason.TIMER, DispatchReason.BOOT])
@pytest.mark.parametrize(
    "kind,interim_resume",
    [
        ("full", "immediate"),
        ("interim", "immediate"),
        ("interim", "tick"),
        ("interim", "01:00-04:00"),
    ],
)
def test_scheduled_firing_mid_conversation_never_resumes(reason, kind, interim_resume):
    now = _at(2, 0)
    pending = PendingEvent(kind=kind, readable=True, since_epoch=_at(1, 0))
    choice = _choice(
        reason=reason,
        pending=pending,
        interim_resume=interim_resume,
        now=now,
        seconds_until_idle=45,
    )
    assert choice.status == "deferred_resume_waiting"
    assert choice.next_opportunity_seconds == 45
    assert choice.next_opportunity_reason is DispatchReason.IDLE
    assert choice.run is None


def test_idle_firing_mid_conversation_defers_owned_by_idle():
    """A turn landed between the watch waking and the arbitrator reading the
    clock: the watch waits rather than exiting."""
    now = _at(2, 0)
    pending = PendingEvent(kind="interim", readable=True, since_epoch=_at(1, 0))
    choice = _choice(
        reason=DispatchReason.IDLE,
        pending=pending,
        interim_resume="immediate",
        now=now,
        seconds_until_idle=10,
    )
    assert choice.status == "deferred_resume_waiting"
    assert choice.next_opportunity_seconds == 10
    assert choice.next_opportunity_reason is DispatchReason.IDLE


# ---------------------------------------------------------------------------
# An unreadable head is a refusal on every reason.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reason",
    [DispatchReason.TIMER, DispatchReason.IDLE, DispatchReason.BOOT, DispatchReason.OPERATOR],
)
def test_unreadable_head_defers_on_every_reason(reason):
    pending = PendingEvent(kind=None, readable=False, since_epoch=None, cause="undecodable")
    choice = _choice(reason=reason, pending=pending, now=_at(12, 0))
    assert choice.status == "deferred_event_unreadable"
    assert choice.run is None


# ---------------------------------------------------------------------------
# An IDLE firing with no pending head never starts a new event, whatever the
# marks, the window and max_interim_count say.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("now_hhmm", [(2, 0), (12, 0)], ids=["inside_window", "outside_window"])
@pytest.mark.parametrize("max_interim_count", [0, 7])
@pytest.mark.parametrize("last_cadence_mark_kind", ["none", "due", "not_due"])
def test_idle_with_no_pending_head_always_noops_nothing_pending(
    now_hhmm, max_interim_count, last_cadence_mark_kind
):
    now = _at(*now_hhmm)
    last_cadence_mark = {
        "none": None,
        "due": _due_mark(now),
        "not_due": _not_due_mark(now),
    }[last_cadence_mark_kind]
    choice = _choice(
        requested=ConsolidationAction.AUTO,
        reason=DispatchReason.IDLE,
        pending=None,
        now=now,
        last_cadence_mark=last_cadence_mark,
        last_full_start=_due_mark(now) if max_interim_count else None,
        max_interim_count=max_interim_count,
        full_fold_deadline=(now - 1) if max_interim_count else None,
    )
    assert choice.status == "noop_nothing_pending"
    assert choice.run is None


# ---------------------------------------------------------------------------
# An operator staging action with no pending head runs directly, whatever
# the window says.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "action",
    [ConsolidationAction.FULL, ConsolidationAction.INTERIM, ConsolidationAction.RECONCILE],
)
@pytest.mark.parametrize("now_hhmm", [(2, 0), (12, 0)], ids=["inside_window", "outside_window"])
def test_operator_staging_action_with_no_pending_head_runs_directly(action, now_hhmm):
    now = _at(*now_hhmm)
    choice = _choice(requested=action, reason=DispatchReason.OPERATOR, pending=None, now=now)
    assert choice.run is action
    assert choice.resume_pending is False
    assert choice.status == ""


# ---------------------------------------------------------------------------
# A calibrate action skips the resume branch entirely.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "action", [ConsolidationAction.CALIBRATE, ConsolidationAction.CALIBRATE_PENDING]
)
def test_calibrate_action_skips_the_resume_branch(action):
    pending = PendingEvent(kind="interim", readable=True, since_epoch=_at(1, 0))
    choice = _choice(
        requested=action, reason=DispatchReason.OPERATOR, pending=pending, now=_at(12, 0)
    )
    assert choice.run is action
    assert choice.resume_pending is False
    assert choice.status == ""


# ---------------------------------------------------------------------------
# FULL resolution: only TIMER/BOOT, only inside the window, only when the
# deadline falls before the next opening, only once per opening.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reason", [DispatchReason.TIMER, DispatchReason.BOOT])
def test_full_resolves_on_timer_or_boot_inside_window_with_a_due_deadline(reason):
    now = _at(2, 0)  # inside 01:00-04:00
    choice = _choice(
        reason=reason,
        pending=None,
        cadence="off",
        now=now,
        full_fold_deadline=now,
        last_full_start=None,
        max_interim_count=7,
    )
    assert choice.run is ConsolidationAction.FULL
    assert choice.starts_full_fold is True
    assert choice.consumes_cadence_mark is False


def test_auto_under_operator_reason_answers_noop_not_due():
    """The schedule's own resolutions — seeding a never-stamped cadence, a
    full fold in its window, an overdue fold with the window shut, an
    interim on a due tick — belong to the timer and boot firings alone; an
    AUTO request under any other reason never touches the window or the
    cadence."""
    now = _at(2, 0)
    choice = _choice(
        requested=ConsolidationAction.AUTO,
        reason=DispatchReason.OPERATOR,
        pending=None,
        cadence="off",
        now=now,
        full_fold_deadline=now,
        max_interim_count=7,
    )
    assert choice.status == "noop_not_due"
    assert choice.run is None


def test_full_second_firing_in_the_same_opening_noops():
    now = _at(2, 0)  # inside 01:00-04:00 -> current_start is today 01:00
    window = parse_window("01:00-04:00")
    current_start = window.current_start(datetime.fromtimestamp(now)).timestamp()
    choice = _choice(
        reason=DispatchReason.TIMER,
        pending=None,
        cadence="off",
        now=now,
        full_fold_deadline=now,
        last_full_start=current_start + 60,  # already started inside this opening
        max_interim_count=7,
    )
    assert choice.run is None
    assert choice.status == "noop_not_due"


def test_full_deadline_after_the_next_opening_defers_not_due():
    now = _at(2, 0)
    window = parse_window("01:00-04:00")
    next_start = window.next_start(datetime.fromtimestamp(now)).timestamp()
    choice = _choice(
        reason=DispatchReason.TIMER,
        pending=None,
        cadence="off",
        now=now,
        full_fold_deadline=next_start + 3600,  # falls after the next opening
        last_full_start=None,
        max_interim_count=7,
    )
    assert choice.run is None
    assert choice.status == "noop_not_due"


def test_max_interim_count_zero_never_yields_interim_on_an_unconsumed_mark():
    """max_interim_count=0 resolves FULL on every unconsumed cadence mark,
    inside the window and outside it, with full_fold_deadline=None, and
    never yields INTERIM."""
    cadence = "12h"
    for now_hhmm, label in [((2, 0), "inside"), ((12, 0), "outside")]:
        now = _at(*now_hhmm)
        choice = _choice(
            reason=DispatchReason.TIMER,
            pending=None,
            cadence=cadence,
            now=now,
            last_cadence_mark=_due_mark(now),
            full_fold_deadline=None,
            max_interim_count=0,
        )
        assert choice.run is ConsolidationAction.FULL, label
        assert choice.consumes_cadence_mark is True, label
        assert choice.starts_full_fold is True, label


def test_ring_with_no_deadline_never_yields_full():
    now = _at(2, 0)  # inside window
    choice = _choice(
        reason=DispatchReason.TIMER,
        pending=None,
        cadence="12h",
        now=now,
        last_cadence_mark=_due_mark(now),
        full_fold_deadline=None,
        max_interim_count=7,
    )
    assert choice.run is ConsolidationAction.INTERIM
    assert choice.run is not ConsolidationAction.FULL


def test_cadence_tick_outside_window_with_deadline_past_noops_outside_window():
    now = _at(12, 0)  # outside 01:00-04:00
    choice = _choice(
        reason=DispatchReason.TIMER,
        pending=None,
        cadence="12h",
        now=now,
        last_cadence_mark=_due_mark(now),
        full_fold_deadline=now - 3600,
        max_interim_count=7,
    )
    assert choice.status == "noop_outside_window"
    assert choice.run is None


# ---------------------------------------------------------------------------
# A wrapping full_window resolves FULL at both ends of the small hours
# against the same opening, and reads a start already recorded there as
# inside that opening.
# ---------------------------------------------------------------------------


def test_wrapping_full_window_resolves_full_at_2330():
    now = _at(23, 30, day=6)
    choice = _choice(
        reason=DispatchReason.TIMER,
        pending=None,
        cadence="off",
        full_window="23:00-02:00",
        now=now,
        full_fold_deadline=_at(12, 0, day=6),
        last_full_start=None,
        max_interim_count=7,
    )
    assert choice.run is ConsolidationAction.FULL
    assert choice.starts_full_fold is True


def test_wrapping_full_window_resolves_full_at_0100_same_opening():
    now = _at(1, 0, day=7)
    choice = _choice(
        reason=DispatchReason.TIMER,
        pending=None,
        cadence="off",
        full_window="23:00-02:00",
        now=now,
        full_fold_deadline=_at(12, 0, day=6),
        last_full_start=None,
        max_interim_count=7,
    )
    assert choice.run is ConsolidationAction.FULL
    assert choice.starts_full_fold is True


def test_wrapping_full_window_after_start_at_2330_noops_at_0100_same_opening():
    started_at = _at(23, 30, day=6)
    now = _at(1, 0, day=7)
    choice = _choice(
        reason=DispatchReason.TIMER,
        pending=None,
        cadence="off",
        full_window="23:00-02:00",
        now=now,
        full_fold_deadline=_at(12, 0, day=6),
        last_full_start=started_at,
        max_interim_count=7,
    )
    assert choice.run is None
    assert choice.status == "noop_not_due"


# ---------------------------------------------------------------------------
# consumes_cadence_mark: True exactly for a verdict standing on an
# unconsumed cadence mark, False for a window resume and for an IDLE-earned
# resume.
# ---------------------------------------------------------------------------


def test_consumes_cadence_mark_true_for_interim_dispatched_on_a_due_tick():
    now = _at(2, 0)
    choice = _choice(
        reason=DispatchReason.TIMER,
        pending=None,
        cadence="12h",
        now=now,
        last_cadence_mark=_due_mark(now),
        full_fold_deadline=None,
        max_interim_count=7,
    )
    assert choice.run is ConsolidationAction.INTERIM
    assert choice.consumes_cadence_mark is True
    assert choice.starts_full_fold is False


def test_consumes_cadence_mark_false_for_a_window_resume():
    now = _at(2, 0)  # inside 01:00-04:00
    pending = PendingEvent(kind="interim", readable=True, since_epoch=_at(1, 0))
    choice = _choice(
        reason=DispatchReason.TIMER,
        pending=pending,
        interim_resume="01:00-04:00",
        cadence="12h",
        now=now,
        last_cadence_mark=_not_due_mark(now),
        seconds_until_idle=0,
    )
    assert choice.run is ConsolidationAction.INTERIM
    assert choice.resume_pending is True
    assert choice.consumes_cadence_mark is False


def test_consumes_cadence_mark_false_for_an_idle_earned_resume():
    now = _at(2, 0)
    pending = PendingEvent(kind="interim", readable=True, since_epoch=_at(1, 0))
    choice = _choice(
        reason=DispatchReason.IDLE,
        pending=pending,
        interim_resume="immediate",
        cadence="12h",
        now=now,
        last_cadence_mark=_due_mark(now),  # cadence IS due, but IDLE never consumes
        seconds_until_idle=0,
    )
    assert choice.run is ConsolidationAction.INTERIM
    assert choice.resume_pending is True
    assert choice.consumes_cadence_mark is False


# ---------------------------------------------------------------------------
# NO_STAMP seeds the cadence mark and noops, only on TIMER/BOOT.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reason", [DispatchReason.TIMER, DispatchReason.BOOT])
def test_no_stamp_noops_scheduler_seeded_and_consumes_the_mark(reason):
    now = _at(2, 0)
    choice = _choice(
        reason=reason,
        pending=None,
        cadence="12h",
        now=now,
        last_cadence_mark=None,
        max_interim_count=7,
    )
    assert choice.status == "noop_scheduler_seeded"
    assert choice.run is None
    assert choice.consumes_cadence_mark is True
    assert choice.starts_full_fold is False


# ---------------------------------------------------------------------------
# A cadence tick's dueness comes from schedule_grammar.scheduled_run_due: a
# non-exact "7h" cadence resolves INTERIM at 8h after the mark,
# noop_not_due at 6h, noop_scheduler_seeded with no stamp; an off cadence
# never seeds.
# ---------------------------------------------------------------------------


def test_seven_hour_cadence_resolves_interim_at_eight_hours_since_the_mark():
    now = _at(12, 0)
    choice = _choice(
        reason=DispatchReason.TIMER,
        pending=None,
        cadence="7h",
        now=now,
        last_cadence_mark=now - 8 * 3600,
        full_fold_deadline=None,
        max_interim_count=7,
    )
    assert choice.run is ConsolidationAction.INTERIM
    assert choice.consumes_cadence_mark is True


def test_seven_hour_cadence_noop_not_due_at_six_hours_since_the_mark():
    now = _at(12, 0)
    choice = _choice(
        reason=DispatchReason.TIMER,
        pending=None,
        cadence="7h",
        now=now,
        last_cadence_mark=now - 6 * 3600,
        full_fold_deadline=None,
        max_interim_count=7,
    )
    assert choice.run is None
    assert choice.status == "noop_not_due"


def test_seven_hour_cadence_noop_scheduler_seeded_on_no_stamp():
    now = _at(12, 0)
    choice = _choice(
        reason=DispatchReason.TIMER,
        pending=None,
        cadence="7h",
        now=now,
        last_cadence_mark=None,
        max_interim_count=7,
    )
    assert choice.status == "noop_scheduler_seeded"
    assert choice.consumes_cadence_mark is True


def test_off_cadence_never_seeds_even_with_no_stamp():
    now = _at(12, 0)
    choice = _choice(
        reason=DispatchReason.TIMER,
        pending=None,
        cadence="off",
        now=now,
        last_cadence_mark=None,
        full_fold_deadline=None,
        max_interim_count=7,
    )
    assert choice.status != "noop_scheduler_seeded"
    assert choice.consumes_cadence_mark is False
