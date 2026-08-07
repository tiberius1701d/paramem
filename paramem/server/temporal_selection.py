"""Model-selected date-group filtering for temporal recall.

Single-purpose prompt-build/generate/parse unit: given the user's turn and
the speaker's key inventory as a ``{key: parsed date or None}`` map
(:func:`paramem.server.temporal.build_date_by_key`), ask the local model
(adapter off, temperature 0) which date groups the query needs, and hand the
caller back a :class:`DateSelection` to filter probe keys against. Kept out
of ``paramem/server/inference.py`` — already the largest server module — so
that module does not absorb another responsibility. Wired into
``_probe_and_reason`` (:mod:`paramem.server.inference`), which is its sole
production caller.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING

from paramem.models.loader import generate_adapter_off
from paramem.server.temporal import DateWindow, weekday_name

if TYPE_CHECKING:
    from paramem.server.config import ServerConfig

logger = logging.getLogger(__name__)

# Injectable seam: unit tests monkeypatch this module attribute so no test
# loads a model or touches the GPU. Production callers never override it.
_generate = generate_adapter_off


@dataclass(frozen=True)
class DateSelection:
    """The model's judgment of which recorded dates a query needs.

    ``all=True`` is both the model's own verdict for an unconstrained query
    and the fail-open shape :func:`select_date_groups` returns on any
    failure — :meth:`selects` treats the two identically, so recall scope
    is unaffected either way: probe everything. :attr:`fail_open`
    distinguishes them for a caller that wants provenance (e.g. a log line)
    without changing that behaviour.
    """

    all: bool
    ranges: tuple[DateWindow, ...]
    include_undated: bool
    fail_open: bool = False

    def selects(self, day: date | None) -> bool:
        """True iff *day* is covered by this selection.

        ``day=None`` denotes an undated key → returns
        :attr:`include_undated`. When :attr:`all` is ``True`` this always
        returns ``True``, including for undated keys, regardless of
        :attr:`include_undated`.
        """
        if self.all:
            return True
        if day is None:
            return self.include_undated
        return any(window.contains(day) for window in self.ranges)


def _render_inventory(date_by_key: Mapping[str, date | None]) -> str:
    """Render the date inventory as ``"{Weekday}, {ISO date}: {N} facts"`` lines.

    Key ids are opaque ``graphN`` names and carry no content, so only
    per-date counts are rendered — the model never sees fact text at this
    stage. Counts are tallied from *date_by_key* in one pass (a key maps
    to ``None`` when it has no usable date); date lines are sorted
    ascending, and the undated line is always last. Each date line is
    labeled with its English weekday name (:func:`paramem.server.temporal.weekday_name`,
    matching the "Today is {weekday}, {ISO}." header line) so the model
    can resolve weekday references ("am Montag") by lookup instead of by
    calendar arithmetic.
    """
    counts: Counter[date] = Counter()
    undated_count = 0
    for day in date_by_key.values():
        if day is None:
            undated_count += 1
        else:
            counts[day] += 1
    lines = [
        f"{weekday_name(day)}, {day.isoformat()}: {n} facts" for day, n in sorted(counts.items())
    ]
    lines.append(f"undated: {undated_count} facts")
    return "\n".join(lines)


def _parse_date(value: object) -> date:
    """Parse one ISO ``YYYY-MM-DD`` string; raises on any other shape."""
    if not isinstance(value, str):
        raise ValueError(f"expected an ISO date string, got {value!r}")
    return date.fromisoformat(value)


def _parse_response(raw: str) -> DateSelection:
    """Parse the model's JSON verdict into a :class:`DateSelection`.

    Progressive first-object extraction — find ``{``, try each ``}`` via
    ``json.JSONDecoder.raw_decode`` — never ``rfind("}")``, so a trailing
    sentence after the JSON object cannot corrupt or swallow the match.

    Raises ``ValueError`` on any malformed shape (no JSON object, missing
    ``ranges``, a non-ISO date, ``start > end``, a non-bool ``undated``);
    :func:`select_date_groups` is the sole caller and treats every
    exception here as fail-open to ``all``.
    """
    decoder = json.JSONDecoder()
    obj = None
    for i, ch in enumerate(raw):
        if ch != "{":
            continue
        try:
            obj, _end = decoder.raw_decode(raw[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            break
        obj = None
    if not isinstance(obj, dict):
        raise ValueError(f"no parseable JSON object in selection response: {raw!r}")

    if obj.get("all") is True:
        return DateSelection(all=True, ranges=(), include_undated=True)

    raw_ranges = obj.get("ranges")
    if not isinstance(raw_ranges, list) or not raw_ranges:
        raise ValueError(f"selection response has neither all nor ranges: {obj!r}")

    windows: list[DateWindow] = []
    for item in raw_ranges:
        if not isinstance(item, dict):
            raise ValueError(f"range entry is not an object: {item!r}")
        start = _parse_date(item.get("start"))
        end = _parse_date(item.get("end"))
        if start > end:
            raise ValueError(f"range start after end: {item!r}")
        windows.append(DateWindow(start=start, end=end))

    include_undated = obj.get("undated")
    if not isinstance(include_undated, bool):
        raise ValueError(f"selection response 'undated' is not a bool: {obj!r}")

    return DateSelection(all=False, ranges=tuple(windows), include_undated=include_undated)


def select_date_groups(
    text: str,
    date_by_key: Mapping[str, date | None],
    *,
    model,
    tokenizer,
    config: ServerConfig,
    today: date,
) -> DateSelection:
    """Ask the local model which recorded date groups *text* needs.

    Builds the selection prompt from the recall-selection section of
    ``configs/prompts/pa_voice.txt`` (``config.voice.load_recall_selection_prompt()``,
    system role) plus a rendered ``Today is {weekday}, {ISO date}.`` line,
    the date inventory (:func:`_render_inventory`), and *text* (user
    role), then runs one adapter-off, temperature-0 generate
    (:func:`generate_adapter_off`, reached through the module-level
    :data:`_generate` seam so tests can stub it without a model). The
    response is parsed by :func:`_parse_response`.

    Fails open to ``DateSelection(all=True, ..., fail_open=True)`` —
    today's unfiltered full probe — on ANY failure: the recall-selection
    section absent (file missing or marker missing), generate raising, no
    parseable JSON, a non-ISO date, ``start > end``, or a wrong type
    anywhere in the shape. Exactly one WARNING is logged on that path.
    This function never raises.

    Args:
        text: The user's current turn.
        date_by_key: ``{key: parsed date or None}`` for every key on the
            routing plan, from
            :func:`paramem.server.temporal.build_date_by_key` over
            ``store.bookkeeping_for_key`` — the single parse of each
            key's raw ``last_seen`` value; this function derives its
            per-date counts from it and never re-parses.
        model: The loaded local model (keyword-only).
        tokenizer: The model's tokenizer (keyword-only).
        config: The server config — supplies
            ``config.inference.temporal_selection_max_new_tokens`` and
            ``config.voice`` (:class:`~paramem.server.config.VoiceConfig`,
            for the recall-selection prompt section) (keyword-only).
        today: The calendar date to render as "Today" and resolve
            relative phrasing against — the caller's single ``date.today()``
            read, passed in so tests can pin it and so the selection prompt
            and the context header agree on the same value (keyword-only).

    Returns:
        The parsed :class:`DateSelection` (``fail_open=False``), or the
        fail-open ``all=True`` shape (``fail_open=True``) on any error.
    """
    try:
        template = config.voice.load_recall_selection_prompt()
        if template is None:
            raise ValueError(
                "recall-selection prompt section missing from configs/prompts/pa_voice.txt"
            )
        today_line = f"Today is {weekday_name(today)}, {today.isoformat()}."
        inventory = _render_inventory(date_by_key)
        messages = [
            {"role": "system", "content": template},
            {"role": "user", "content": f"{today_line}\n{inventory}\n\n{text}"},
        ]
        raw = _generate(
            model,
            tokenizer,
            messages,
            max_new_tokens=config.inference.temporal_selection_max_new_tokens,
            temperature=0.0,
        )
        return _parse_response(raw)
    except Exception:
        logger.warning("Date-group selection failed; falling open to a full probe", exc_info=True)
        return DateSelection(all=True, ranges=(), include_undated=True, fail_open=True)
