"""Pure date/grouping primitives for temporal recall.

Stdlib-only — no ``paramem`` imports — so this module and
:mod:`paramem.server.temporal_selection` (which consumes it) stay unit
testable with no store, model, or config in the loop. ``DateWindow`` is the
range representation a date-group selection resolves to (e.g. "last week"
selects one range, not seven enumerated days).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime


@dataclass(frozen=True)
class DateWindow:
    """An inclusive calendar-day range."""

    start: date
    end: date

    def contains(self, day: date) -> bool:
        """True iff *day* falls within ``[start, end]``, both ends inclusive."""
        return self.start <= day <= self.end


def last_seen_date(last_seen: object) -> date | None:
    """Parse a bookkeeping ``last_seen`` value into a local calendar date.

    ``last_seen`` is disk-splatted, unvalidated data
    (``paramem/memory/store.py:1449``) reaching a request path with no
    exception handler around it — this is the one boundary that turns it
    into a date without ever raising, rather than error suppression.

    Non-``str`` values (``None``, an ``int``, ...) and the empty string
    return ``None``. A timezone-aware ISO string (the shape
    ``datetime.now(timezone.utc).isoformat()`` produces at
    ``paramem/server/session_buffer.py:318``) is converted to the
    process's local calendar day via ``.astimezone().date()``; a naive ISO
    string is read as already-local via ``.date()``. Any value
    ``datetime.fromisoformat`` rejects returns ``None``, as does a value
    it accepts whose local-day conversion overflows ``datetime``'s
    representable year range (an aware year-9999 or year-1 timestamp can
    shift past ``datetime.MAXYEAR``/``MINYEAR`` once ``.astimezone()``
    applies the process's local offset).

    Args:
        last_seen: The raw bookkeeping value for one key.

    Returns:
        The parsed local calendar date, or ``None`` when *last_seen* is
        not a non-empty, ISO-parseable string, or its local-day
        conversion overflows.
    """
    if not isinstance(last_seen, str) or not last_seen:
        return None
    try:
        parsed = datetime.fromisoformat(last_seen)
        if parsed.tzinfo is not None:
            return parsed.astimezone().date()
        return parsed.date()
    except (ValueError, TypeError, OverflowError):
        return None


def build_date_by_key(
    last_seen_by_key: Mapping[str, object],
) -> dict[str, date | None]:
    """Parse every key's raw ``last_seen`` bookkeeping value exactly once.

    One :func:`last_seen_date` call per key — the single parse point every
    downstream consumer (the selection inventory, the survivor filter, the
    nothing-in-period note, the per-fact date rendering) reads from, so a
    key's raw bookkeeping value is never parsed twice in one request.

    Args:
        last_seen_by_key: Mapping of key id to its raw ``last_seen``
            bookkeeping value.

    Returns:
        ``{key: parsed date or None}``. ``None`` denotes "no usable
        date" — unparseable, or genuinely absent — and both cases map to
        the same ``None`` on purpose: a caller filtering on parsed dates
        must treat them identically, never dropping one silently.
    """
    return {key: last_seen_date(raw) for key, raw in last_seen_by_key.items()}
