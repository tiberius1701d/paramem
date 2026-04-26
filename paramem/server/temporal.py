"""Temporal query detection and registry date filtering.

Detects time references in user queries ("yesterday", "last week", etc.)
and resolves them to absolute date ranges for registry-based key lookup.

Detection is structural — explicit phrase tables scanned with ``in`` /
``str.split()`` lookups, no regex. Misses are bounded by the phrase
table; adding a new phrase is one dict entry. The same approach as the
sanitizer's first-person token-set: every recognised case is a literal
in source.
"""

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


def _day_before_yesterday(today, _n=None):
    d = today - timedelta(days=2)
    return (d, d)


def _yesterday(today, _n=None):
    d = today - timedelta(days=1)
    return (d, d)


def _last_week(today, _n=None):
    start = today - timedelta(days=today.weekday() + 7)
    end = today - timedelta(days=today.weekday() + 1)
    return (start, end)


def _this_week(today, _n=None):
    return (today - timedelta(days=today.weekday()), today)


def _n_days_ago(today, n):
    d = today - timedelta(days=int(n))
    return (d, d)


def _n_weeks_ago(today, n):
    start = today - timedelta(weeks=int(n))
    end = today - timedelta(weeks=int(n) - 1)
    return (start, end)


def _first_of_last_month(today: date) -> date:
    if today.month == 1:
        return date(today.year - 1, 12, 1)
    return date(today.year, today.month - 1, 1)


def _last_of_last_month(today: date) -> date:
    return today.replace(day=1) - timedelta(days=1)


def _last_month(today, _n=None):
    return (_first_of_last_month(today), _last_of_last_month(today))


def _this_month(today, _n=None):
    return (today.replace(day=1), today)


def _last_weekday(today: date, target_weekday: int) -> tuple[date, date]:
    days_back = (today.weekday() - target_weekday) % 7
    if days_back == 0:
        days_back = 7  # "on Monday" means last Monday, not today
    d = today - timedelta(days=days_back)
    return (d, d)


def _same_day(today, _n=None):
    return (today, today)


def _recently(today, _n=None):
    return (today - timedelta(days=7), today)


def _the_other_day(today, _n=None):
    return (today - timedelta(days=3), today - timedelta(days=1))


# Multi-word phrase → resolver. Order matters: longer phrases must be
# tested before shorter ones that are a substring (e.g. "the day before
# yesterday" before "yesterday"). Iteration order is insertion order
# (Python 3.7+).
_TEMPORAL_PHRASES: dict[str, callable] = {
    # Longer phrases first — substring containment is checked in order.
    "the day before yesterday": _day_before_yesterday,
    "the other day": _the_other_day,
    "this morning": _same_day,
    "this afternoon": _same_day,
    "this evening": _same_day,
    "last week": _last_week,
    "this week": _this_week,
    "last month": _last_month,
    "this month": _this_month,
    "yesterday": _yesterday,
    "tonight": _same_day,
    "today": _same_day,
    "recently": _recently,
}

# Day name → weekday index. Matched as a standalone token (split on
# whitespace, strip punctuation) so "Monday." / "Monday?" still match.
_WEEKDAYS: dict[str, int] = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}

# Trailing punctuation stripped before token comparison.
_TOKEN_PUNCT = ".,!?;:'\"()[]{}"


def _normalize_tokens(text: str) -> list[str]:
    """Lowercase + whitespace-split + per-token punctuation strip. No regex."""
    return [tok.strip(_TOKEN_PUNCT).lower() for tok in text.split()]


def _detect_n_unit_ago(tokens: list[str], today: date) -> tuple[date, date] | None:
    """Token scan for ``<digit> day(s)/week(s) ago`` patterns."""
    for i in range(len(tokens) - 2):
        n_tok = tokens[i]
        unit = tokens[i + 1]
        ago = tokens[i + 2]
        if not n_tok.isdigit() or ago != "ago":
            continue
        if unit in ("day", "days"):
            return _n_days_ago(today, int(n_tok))
        if unit in ("week", "weeks"):
            return _n_weeks_ago(today, int(n_tok))
    return None


def _detect_weekday(tokens: list[str], today: date) -> tuple[date, date] | None:
    """Token scan for a day name (optionally preceded by ``on`` / ``last``)."""
    for tok in tokens:
        if tok in _WEEKDAYS:
            return _last_weekday(today, _WEEKDAYS[tok])
    return None


def detect_temporal_query(
    text: str, reference_date: date | None = None
) -> tuple[date, date] | None:
    """Detect temporal references in text and resolve to a date range.

    Returns (start_date, end_date) inclusive, or None if no temporal
    reference is found.

    Detection order:
    1. Multi-word phrases from ``_TEMPORAL_PHRASES`` (longest-first by
       insertion order). ``str.__contains__`` against the lowercased
       query — substring match, not pattern match.
    2. Numeric "<n> day(s)/week(s) ago" via token scan.
    3. Standalone day names via token scan.
    """
    today = reference_date or date.today()
    text_lower = text.lower()

    # 1. Phrase containment in declared order (longest first).
    for phrase, resolver in _TEMPORAL_PHRASES.items():
        if phrase in text_lower:
            return resolver(today)

    tokens = _normalize_tokens(text_lower)

    # 2. "<n> days/weeks ago"
    result = _detect_n_unit_ago(tokens, today)
    if result is not None:
        return result

    # 3. Standalone day name
    return _detect_weekday(tokens, today)


def filter_registry_by_date(
    registry_path: Path,
    start_date: date,
    end_date: date,
) -> list[str]:
    """Return keys whose last_seen_at falls within the date range.

    Both start_date and end_date are inclusive.
    """
    if not registry_path.exists():
        return []

    from paramem.backup.encryption import read_maybe_encrypted

    registry = json.loads(read_maybe_encrypted(registry_path).decode("utf-8"))

    matching_keys = []
    for key, meta in registry.items():
        if not isinstance(meta, dict):
            continue
        if meta.get("status") != "active":
            continue

        last_seen = meta.get("last_seen_at")
        if not last_seen:
            continue

        try:
            seen_date = datetime.fromisoformat(last_seen).date()
        except (ValueError, TypeError):
            continue

        if start_date <= seen_date <= end_date:
            matching_keys.append(key)

    return matching_keys
