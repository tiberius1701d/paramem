"""Durable schedule marks for the consolidation suspend/power-off catch-up gate.

Boundary: this module owns ONLY the on-disk representation of two marks the
scheduled-tick dispatcher keeps across restarts: the last cadence mark it has
consumed — for EVERY cadence kind, not only non-calendar-exact ones (see
``schedule_grammar.scheduled_run_due`` / ``server/app.py::_dispatch_consolidation``);
a non-exact cadence needs it because its rendered ``OnCalendar`` timer is a
wakeup source only, not the schedule itself, and an exact cadence needs it too,
to keep a duplicate or manual-adjacent tick inside the same calendar mark's
window from being read as due twice — and the last instant a full fold
started, a record the one-fold-per-window-opening rule reads to tell "already
started this opening" from "due again".

Deliberately NOT in ``systemd_timer.py``: the consumer is the runtime
dispatcher in ``app.py`` (``scheduled_tick`` → ``_dispatch_consolidation``),
not the unit-rendering module — and ``systemd_timer.py`` already carries three
responsibilities (rendering, reconciling, timer-state reading) without taking
on run-state I/O too.

The file carries both marks together, never in separate files, since a caller
reasoning about one mark commonly needs the other in the same decision (a full
fold started this window is read alongside the cadence mark that window last
consumed) and a dispatch that changes either or both writes the whole pair in
one atomic call (:func:`write_marks`) rather than a read-modify-write per
mark. A caller reads the pair with :func:`read_marks`, decides the new pair —
one field changed, both changed, or neither — and writes it once.

A present file this process cannot interpret is a refusal, never "never
stamped": a mark this build cannot vouch for, read as absent, would seed the
cadence mark and skip a cycle in silence. :func:`read_marks` raises
:class:`ScheduleStateUnreadable` for a payload that does not decode, cannot be
opened, does not decode to the marks record, or carries a mark of the wrong
type, and :class:`ScheduleStateVersionUnsupported` for a ``schema_version``
this build does not write — the same pair
:class:`~paramem.training.stage_ledger.StageLedgerUnreadable` /
:class:`~paramem.training.stage_ledger.StageLedgerVersionUnsupported` make for
the stage ledger. A present-but-uninterpretable file is never overwritten:
the read that precedes every write is what refuses it, so the write itself
never runs.

Schema
------
``{"schema_version": 2, "last_cadence_mark_epoch": <float | null>,
"last_full_start_epoch": <float | null>}``. A ``null`` mark is a legal value
(a full fold that has never started); a missing key, a non-numeric mark, and a
non-object payload (including a bare JSON ``null``) are not.

Concurrency
-----------
Written from a single place (the event-loop-bound scheduled-tick
dispatcher) — no ``flock_rmw`` read-modify-write is needed, unlike
``backup/state.py`` (which has both a CLI runner process and a server
endpoint writing concurrently).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path

from paramem.server.atomic_json import atomic_write_json

SCHEDULE_STATE_FILENAME: str = "consolidation_schedule.json"
SCHEDULE_STATE_SCHEMA_VERSION: int = 2


@dataclass(frozen=True)
class ScheduleMarks:
    """The two marks the schedule stamp file carries.

    Attributes:
        last_cadence_mark_epoch: The last cadence mark this process consumed,
            or ``None`` when no cadence mark has ever been consumed.
        last_full_start_epoch: The instant the last full fold started, or
            ``None`` when a full fold has never started.
    """

    last_cadence_mark_epoch: "float | None"
    last_full_start_epoch: "float | None"


# The two mark field names, derived from the dataclass itself rather than
# re-spelled at each call site — the presence check, the type check, and both
# writers all walk this one tuple.
_MARK_FIELDS: "tuple[str, ...]" = tuple(f.name for f in fields(ScheduleMarks))


class ScheduleStateUnreadable(RuntimeError):
    """The stamp file is present and this process cannot interpret it.

    Raised by :func:`read_marks` for each of the four ways a present file
    fails: a payload that does not decode as UTF-8 JSON
    (``cause="undecodable"``); a file that cannot be opened at all — a
    permissions failure, or a path that is a directory
    (``cause="unopenable"``); a payload that decodes but is not the marks
    record — not a JSON object, a bare JSON ``null``, missing
    ``schema_version``, or missing a mark (``cause="not_marks"``); and a mark
    present but neither ``None`` nor a real number — a quoted epoch, a bool,
    an object (``cause="bad_mark"``, the message naming the key and the
    value). Reporting "never stamped" over a record this process cannot read
    would seed the cadence mark and skip a cycle in silence, so none of the
    four folds into absence — a present file always either yields marks or
    raises.
    """

    def __init__(self, *, path: Path, cause: str, detail: "str | None" = None) -> None:
        self.path = path
        self.cause = cause
        self.detail = detail
        message = f"schedule_state: {path} is present but unreadable ({cause})"
        if detail:
            message += f": {detail}"
        super().__init__(message)


class ScheduleStateVersionUnsupported(RuntimeError):
    """The stamp file's ``schema_version`` is not the one this build writes.

    Raised rather than folded into "never stamped": marks this build cannot
    vouch for, read as absent, would seed the cadence mark and skip a cycle
    in silence.
    """

    def __init__(self, *, path: Path, version: object) -> None:
        self.path = path
        self.version = version
        super().__init__(
            f"schedule_state: {path} has unsupported schema_version {version!r} "
            f"(supported: {SCHEDULE_STATE_SCHEMA_VERSION})"
        )


def read_marks(state_dir: Path) -> ScheduleMarks:
    """Read both schedule marks.

    Args:
        state_dir: Directory containing ``consolidation_schedule.json`` — the
            data root's state directory, e.g.
            :func:`~paramem.training.stage_ledger.data_state_dir`
            ``(config.paths.data)``.

    Returns:
        ``ScheduleMarks(None, None)`` when the file is genuinely absent — a
        present file holding the JSON literal ``null`` is NOT absence and
        raises, since it carries no marks this build can vouch for.

    Raises:
        ScheduleStateUnreadable: The file is present but its payload does not
            decode, cannot be opened, does not decode to the marks record, or
            carries a mark of the wrong type. See the class's own docstring
            for the four causes.
        ScheduleStateVersionUnsupported: The file's ``schema_version`` is not
            :data:`SCHEDULE_STATE_SCHEMA_VERSION`.
    """
    path = Path(state_dir) / SCHEDULE_STATE_FILENAME
    try:
        raw_bytes = path.read_bytes()
    except FileNotFoundError:
        return ScheduleMarks(last_cadence_mark_epoch=None, last_full_start_epoch=None)
    except OSError as exc:
        # A permissions failure or a directory at this path is not a decoding
        # problem -- point the operator at the file, not its contents.
        raise ScheduleStateUnreadable(path=path, cause="unopenable") from exc

    try:
        raw = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ScheduleStateUnreadable(path=path, cause="undecodable") from exc

    if not isinstance(raw, dict) or "schema_version" not in raw:
        raise ScheduleStateUnreadable(path=path, cause="not_marks")

    version = raw["schema_version"]
    if version != SCHEDULE_STATE_SCHEMA_VERSION:
        raise ScheduleStateVersionUnsupported(path=path, version=version)

    if any(field not in raw for field in _MARK_FIELDS):
        raise ScheduleStateUnreadable(path=path, cause="not_marks")

    values: "dict[str, float | None]" = {}
    for field_name in _MARK_FIELDS:
        value = raw[field_name]
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise ScheduleStateUnreadable(
                path=path, cause="bad_mark", detail=f"{field_name}={value!r}"
            )
        values[field_name] = value

    return ScheduleMarks(**values)


def write_marks(state_dir: Path, marks: ScheduleMarks) -> None:
    """Atomically write *marks* as the current schedule state.

    The one write this module offers. A caller that means to change one
    mark or both reads the current pair with :func:`read_marks` first —
    which refuses a present file it cannot interpret rather than answering
    something to build a new pair from — then passes the decided pair here.
    Changing both marks in the same dispatch (a full fold that starts on a
    due cadence tick) costs the same one call as changing one: there is no
    read-modify-write per mark to race, so a crash between two single-mark
    writes can never strand one of them stale.

    Args:
        state_dir: Directory that will contain ``consolidation_schedule.json``.
            Created with parents if absent.
        marks: The complete pair to write.
    """
    atomic_write_json(
        Path(state_dir),
        SCHEDULE_STATE_FILENAME,
        {
            "schema_version": SCHEDULE_STATE_SCHEMA_VERSION,
            **{field_name: getattr(marks, field_name) for field_name in _MARK_FIELDS},
        },
    )
