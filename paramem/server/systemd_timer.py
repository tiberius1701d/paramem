"""Systemd user-timer reconciliation for the consolidate cycle.

A timer carries a set of calendar entries: the cadence's own marks
(``consolidation.refresh_cadence``, passed to :func:`reconcile` as
*schedule*) plus, when a caller supplies them, a full-fold window start and
an interim-resume window start (passed as *extra_calendars*, rendered
through :func:`window_start_calendar`). Every entry wakes the same
endpoint, ``POST /scheduled-tick``; the arbitrator tells the entries apart
by the wall clock and its own two durable marks, not by which entry fired,
and decides from those what the wakeup earns — a resumed event, a full
fold, an interim fold, or a noop.

THE PRINCIPLE — the rendered ``OnCalendar`` timer decides WHEN the process
wakes: exact-divisor cadences wake at their own period, non-exact cadences
wake at a coarser grid (see below), and a window start wakes once a day at
its own time. WHETHER a given wakeup actually starts a fold is decided one
level up, by the arbitrator, from the wall clock and its own durable marks
— never from timer identity. For an exact cadence the wakeup and the mark
it represents coincide, so the arbitrator's own check ordinarily passes
straight through; it still guards against a duplicate/manual tick landing
inside the same mark's window and against a missed exact-cadence tick after
a suspend where systemd's own coalesced catch-up fires only once for
however many marks were missed.

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
  render at a coarser HEARTBEAT grid (the ``gcd`` grid computed by
  :func:`~paramem.server.schedule_grammar.non_exact_interval_grid`, imported
  from ``schedule_grammar`` rather than recomputed here — see
  :func:`_period_heartbeat_calendar`) — the timer is a wakeup source only.
  Each heartbeat, the arbitrator checks the real cadence period against its
  own durable last-attempt mark and no-ops until it is actually due — the
  same due-ness answer the full fold already gives itself, applied one
  level up, to the timer that drives the tick.

Accepted schedule strings (same parser as before, plus "off"):
    ""  / "off" / "disabled"  → no timer (manual /consolidate only)
    "every Nh" (24 % N == 0)  → OnCalendar=*-*-* HH,...:00:00 + Persistent=true
    "every Nh" (24 % N != 0)  → OnCalendar heartbeat grid (see
                                 _period_heartbeat_calendar) + Persistent=true
    "every Nm" (60 % N == 0)  → OnCalendar=*:MM,...:00 + Persistent=true
    "every Nm" (60 % N != 0)  → OnCalendar heartbeat + Persistent=true
    "weekly"                  → OnCalendar=Mon *-*-* 00:00:00 + Persistent=true
    "HH:MM"                   → OnCalendar daily + Persistent=true
    "daily"                   → OnCalendar *-*-* 03:00:00 + Persistent=true
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from paramem.server import schedule_grammar
from paramem.server.schedule_grammar import Window, parse_schedule_atom
from paramem.utils import systemctl
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

    A timer carries a set of calendar entries: the consolidate timer wakes
    at every cadence mark and at each window start a caller supplies — one
    ``OnCalendar=`` line per distinct rendered expression, in
    first-occurrence order. ``on_calendars`` holds that set.

    kind values:
      "off"      — no timer installed; carries no entries.
      "calendar" — OnCalendar + Persistent=true; exact grid or heartbeat
                    grid (see module docstring), always catches up missed
                    ticks on boot/resume.
      "daily"    — OnCalendar + Persistent=true; fixed daily wall-clock time.

    There is no monotonic kind — every non-"off" timer is OnCalendar-based.

    ``__post_init__`` normalises ``on_calendars`` to a deduplicated tuple in
    first-occurrence order, so every consumer (``render_timer_unit``,
    ``_reconcile_timer``'s union) sees the set already reduced and never
    dedups it a second time. It also holds the one invariant a ``TimerSpec``
    must satisfy — ``kind == "off"`` exactly when there are no entries —
    raising ``ValueError`` naming both when it does not.
    """

    kind: str  # "off" | "calendar" | "daily"
    on_calendars: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        deduped = tuple(dict.fromkeys(self.on_calendars))
        object.__setattr__(self, "on_calendars", deduped)
        if (self.kind == "off") != (not deduped):
            raise ValueError(
                f"kind={self.kind!r} and on_calendars={deduped!r} disagree — "
                '"off" must carry no entries and every other kind must carry at least one'
            )


def _hours_to_calendar(n: int) -> str | None:
    """Return an OnCalendar expression for every-N-hours if 24 % n == 0, else None.

    The exact-grid renderer.  Also
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

    The exact-grid renderer.  Also
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
    The gcd itself is computed by ``schedule_grammar.non_exact_interval_grid``
    (imported rather than recomputed here) so this rendered grid and
    ``schedule_grammar.scheduled_run_stamp_value``'s heartbeat-floored stamp
    for the same cadence can never disagree. The timer is a pure wakeup
    source here; the dispatcher's durable stamp decides which wakeups
    actually dispatch.
    """
    if schedule_grammar.interval_is_exact(count, unit):
        cal = _hours_to_calendar(count) if unit == "h" else _minutes_to_calendar(count)
        assert cal is not None  # interval_is_exact(count, unit) guarantees this
        return cal
    grid = schedule_grammar.non_exact_interval_grid(count, unit)
    cal = _hours_to_calendar(grid) if unit == "h" else _minutes_to_calendar(grid)
    assert cal is not None  # gcd(count, modulus) always divides modulus
    return cal


def _hhmm_calendar(hh: int, mm: int) -> str:
    """Render an hour/minute pair as the OnCalendar expression for daily at that time.

    The one place this rendering is written; both :func:`parse_schedule`'s
    ``"HH:MM"`` cadence branch and :func:`window_start_calendar` (the
    window-start renderer offered to callers outside this module) call it,
    so the two can never disagree on what "daily at HH:MM" looks like as an
    OnCalendar expression.
    """
    return f"*-*-* {hh:02d}:{mm:02d}:00"


def window_start_calendar(window: Window) -> str:
    """Render a window's start as the OnCalendar expression for daily at that time.

    Meant for a caller outside this module — the full-fold and
    interim-resume window starts — that needs to add a wakeup to
    ``extra_calendars`` on :func:`reconcile`. Its only production source is a
    config-validated :class:`~paramem.server.schedule_grammar.Window`, so
    there is no text to re-parse here. Uses the same rendering
    :func:`parse_schedule` uses for a bare ``"HH:MM"`` cadence, so the entry
    a caller hands in is byte-identical to what the timer would already
    carry for that same time as a cadence mark, and ``_reconcile_timer``'s
    dedup-by-rendered-expression recognises it as the same entry when the
    two coincide.

    Args:
        window: The window whose start renders as a daily wakeup.

    Returns:
        The OnCalendar expression, e.g. ``"*-*-* 01:00:00"``.
    """
    return _hhmm_calendar(window.start_minute // 60, window.start_minute % 60)


def parse_schedule(schedule: str) -> TimerSpec | None:
    """Parse a schedule string into a systemd TimerSpec.

    Accepted formats:

    - ``""`` / ``"off"`` / ``"disabled"`` / ``"none"`` → disabled timer.
    - ``"weekly"`` → ``OnCalendar=Mon *-*-* 00:00:00`` + ``Persistent=true``.
    - ``"daily"`` → OnCalendar daily at 03:00 (same as ``"03:00"``).
    - ``"every Nh"`` where ``24 % N == 0`` → calendar timer at each N-hour mark.
    - ``"every Nh"`` where ``24 % N != 0`` → calendar HEARTBEAT timer (see
      :func:`_period_heartbeat_calendar`); the dispatcher's durable stamp
      decides whether a given wakeup actually runs.
    - ``"every Nm"`` where ``60 % N == 0`` → calendar timer at each N-minute mark.
    - ``"every Nm"`` where ``60 % N != 0`` → calendar HEARTBEAT timer.
    - ``"HH:MM"`` → daily OnCalendar timer at the given time.

    Every non-off kind is ``OnCalendar`` + ``Persistent=true`` (see module
    docstring) — there is no monotonic fallback. ``on_calendars`` carries
    exactly the one expression this schedule string renders to; a caller
    that needs to add further wakeups (a window start) does so via
    :func:`reconcile`'s ``extra_calendars``, not by extending the tuple
    this function returns.

    Returns TimerSpec(kind="off") for an explicit off setting.
    Returns None on malformed input (caller logs + falls back to off).
    """
    atom = parse_schedule_atom(schedule)
    if atom is None:
        return None
    if atom.kind == "off":
        return TimerSpec(kind="off")
    if atom.kind == "weekly":
        cal = (
            f"{schedule_grammar.WEEKLY_ANCHOR_LABEL} *-*-* "
            f"{schedule_grammar.WEEKLY_ANCHOR_HOUR:02d}:"
            f"{schedule_grammar.WEEKLY_ANCHOR_MINUTE:02d}:00"
        )
        return TimerSpec(kind="calendar", on_calendars=(cal,))
    if atom.kind == "daily":
        cal = _hhmm_calendar(
            schedule_grammar.DAILY_ANCHOR_HOUR, schedule_grammar.DAILY_ANCHOR_MINUTE
        )
        return TimerSpec(kind="daily", on_calendars=(cal,))
    if atom.kind == "interval":
        cal = _period_heartbeat_calendar(atom.count, atom.unit)
        return TimerSpec(kind="calendar", on_calendars=(cal,))
    if atom.kind == "hhmm":
        return TimerSpec(kind="daily", on_calendars=(_hhmm_calendar(atom.hh, atom.mm),))
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

    ``--retry 3 --retry-connrefused --retry-delay 120`` covers the observed
    startup race where ``paramem-server.service`` is ordered (``After=``)
    but not gated on readiness: systemd starts the server process and the
    tick unit in process order, not listener-ready order, and the server
    has taken up to ~4 minutes to reach ``listen()``. A firing that lands
    in that window without retry would hit connection-refused and be lost
    outright (there is no cross-firing retry — the next attempt is the
    *next* scheduled tick). With these flags curl attempts at t=0/2/4/6 minutes
    (fixed ``--retry-delay``, not the default exponential backoff) and then
    gives up until the next scheduled firing.
    ``--retry-connrefused`` (curl 7.52.0+) is required because plain
    ``--retry`` classes only timeouts, FTP 4xx, and HTTP
    408/429/500/502/503/504 as transient (``man curl`` / ``curl --help
    all``, verified curl 8.5.0) — ECONNREFUSED is not on that list without
    it. A ``200`` deferral response is not a transient error and is never
    retried; a 401/403 refusal is likewise not on the transient list (only
    408/429 among 4xx are) and, combined with ``--fail-with-body``, still
    fails the unit on the first attempt without retrying. 5xx/408/429
    responses DO retry under ``--retry`` — accepted, since those are
    legitimately transient on this server too.
    A ``refresh_cadence`` shorter than the ~6-minute retry window (the
    grammar's only floor is `count > 0`, e.g. ``"every 1m"`` parses) can let
    the next scheduled firing land while a retry sequence from the previous
    firing is still in flight. This is harmless, not raced: per
    ``man systemd.timer``, "in case the unit to activate is already active
    at the time the timer elapses it is not restarted, but simply left
    running — there is no concept of spawning new service instances in this
    case"; a redundant ``start`` job for a still-activating unit is merged
    with the pending one under systemd's default ``--job-mode=replace``
    (``man systemctl``), not queued as a second concurrent run. So the
    in-flight retry continues undisturbed and the redundant trigger is a
    no-op — no concurrent curl, no duplicate dispatch. (The dispatcher's own
    durable last-attempt stamp, see the module docstring, is a second,
    independent guard against a duplicate dispatch reaching the arbitrator.)

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
        f" | /usr/bin/curl -sS --fail-with-body -X POST --max-time 10"
        f" --retry 3 --retry-connrefused --retry-delay 120 -H @- {endpoint}'\n"
    )


def render_timer_unit(
    spec: TimerSpec,
    *,
    unit_name: str = TIMER_NAME,
    description: str = "ParaMem consolidation scheduler",
) -> str:
    """Render the systemd .timer unit text for the given spec.

    Every non-``"off"`` kind (``"calendar"``, ``"daily"``) emits one
    ``OnCalendar=`` line per entry in ``spec.on_calendars`` — already a
    deduplicated set, in first-occurrence order, by ``TimerSpec.__post_init__``
    — plus one ``Persistent=true``, so a missed tick fires on the next
    boot/resume regardless of which entry it belongs to; there is no
    monotonic kind left to special-case (see module docstring). ``systemd``
    unions multiple ``OnCalendar=`` lines on one timer.

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
        for cal in spec.on_calendars:
            lines.append(f"OnCalendar={cal}")
        lines.append("Persistent=true")
    lines.extend(["", "[Install]", "WantedBy=timers.target", ""])
    return "\n".join(lines)


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


def _reconcile_timer(
    target: TimerTarget, schedule: str, *, extra_calendars: Sequence[str] = ()
) -> str:
    """Shared reconciliation core for both the consolidation and backup timers.

    Parses *schedule*, builds the effective spec by constructing a
    ``TimerSpec`` from the cadence's own entries plus *extra_calendars* —
    ``TimerSpec.__post_init__`` does the dedup — renders/writes the unit
    pair carrying it, and — only when the rendered unit content actually
    changed — (re)enables and restarts the timer. Never raises on systemd
    errors — logs and returns a notice so the caller (server startup, or
    ``_apply_config_live`` on a live cadence change) still proceeds.

    The timer is removed only when the cadence contributes no entries and
    *extra_calendars* is empty: an off cadence carrying one or more
    *extra_calendars* still leaves a running timer that wakes at those
    entries, and only an off cadence with no extras removes the unit. The
    effective kind reported and rendered is the cadence's own kind, or
    ``"calendar"`` when the cadence itself is off and the entries are made
    up entirely of extras.

    ``enable --now`` and ``restart`` are gated on ``svc_changed or
    tmr_changed`` (the same flag that gates ``daemon-reload``), not called
    unconditionally: when nothing changed, the previously-reconciled timer
    is already enabled and running, so re-issuing ``enable --now`` on every
    call would be a needless real ``systemctl`` write against the live user
    session for no effect. Content-unchanged also covers first install
    (``_write_if_changed`` reports a change when the unit files do not yet
    exist), so a fresh machine still gets enabled.

    Returns a short human-readable description of the action taken, naming
    the entries installed, which the caller logs. The returned state
    ("updated" vs "already current") is truthful by construction: it reports
    exactly ``svc_changed or tmr_changed``, the same condition that gated
    every systemctl call in this branch, so "already current" is never
    returned after an action was actually taken.
    """
    spec = parse_schedule(schedule)
    if spec is None:
        logger.error(
            "Invalid %s schedule: %r — does not match the accepted schedule grammar "
            "(see paramem.server.schedule_grammar.parse_schedule_atom), so the cadence "
            "contributes no timer entries; see the return line for what is actually installed.",
            target.timer_name,
            schedule,
        )
        spec = TimerSpec(kind="off")

    if not spec.on_calendars and not extra_calendars:
        changed = False
        if target.timer_path.exists():
            systemctl.run("stop", f"{target.timer_name}.timer")
            systemctl.run("disable", f"{target.timer_name}.timer")
            target.timer_path.unlink(missing_ok=True)
            target.service_path.unlink(missing_ok=True)
            systemctl.run("daemon-reload")
            changed = True
        return f"{target.timer_name}: disabled" + (" (removed)" if changed else "")

    effective_kind = spec.kind if spec.kind != "off" else "calendar"
    effective_spec = TimerSpec(effective_kind, (*spec.on_calendars, *extra_calendars))

    svc_changed = _write_if_changed(target.service_path, target.service_content)
    tmr_changed = _write_if_changed(
        target.timer_path,
        render_timer_unit(
            effective_spec, unit_name=target.timer_name, description=target.description
        ),
    )
    changed = svc_changed or tmr_changed

    if changed:
        systemctl.run("daemon-reload")

        enable = systemctl.run("enable", "--now", f"{target.timer_name}.timer")
        if enable.returncode != 0:
            logger.warning(
                "systemctl enable --now %s.timer failed: %s",
                target.timer_name,
                enable.stderr.strip(),
            )
            return f"{target.timer_name}: enable failed ({enable.stderr.strip()[:80]})"

        # If unit already enabled, systemd won't restart it on daemon-reload —
        # force a restart so the new OnCalendar entries take effect.
        systemctl.run("restart", f"{target.timer_name}.timer")

    # Every non-off kind is OnCalendar + Persistent=true — catch-up always
    # applies, whether the grid is exact or a heartbeat (see module docstring).
    entries = ", ".join(effective_spec.on_calendars)
    if effective_spec.kind == "calendar":
        detail = f"calendar {entries} (with catch-up)"
    else:
        detail = f"daily at {entries} (with catch-up)"
    state = "updated" if changed else "already current"
    return f"{target.timer_name}: {state}, {detail}"


def reconcile(
    schedule: str,
    endpoint: str = DEFAULT_ENDPOINT,
    project_root: str | None = None,
    *,
    extra_calendars: Sequence[str] = (),
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
        is decided per-tick by the arbitrator (interim accumulation plus an
        oldest-interim deadline), not by a separate derived-period timer.
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
    extra_calendars:
        Further OnCalendar expressions to union with the cadence's own
        entries, deduplicated by rendered expression — the full-fold and
        interim-resume window starts, rendered via :func:`window_start_calendar`.
        Empty by default, in which case the timer carries exactly the
        cadence's own entries (or none, for an off cadence).
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
    return _reconcile_timer(target, schedule, extra_calendars=extra_calendars)


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
    show = systemctl.run(
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
    list_out = systemctl.run(
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
