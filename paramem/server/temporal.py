"""Temporal query detection and registry date filtering.

Detects time references in user queries ("yesterday", "last week", etc.)
and resolves them to absolute date ranges for registry-based key lookup.
"""

import json
import logging
import re
from datetime import date, datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


def _day_before_yesterday(today):
    d = today - timedelta(days=2)
    return (d, d)


def _yesterday(today):
    d = today - timedelta(days=1)
    return (d, d)


def _last_week(today):
    start = today - timedelta(days=today.weekday() + 7)
    end = today - timedelta(days=today.weekday() + 1)
    return (start, end)


def _this_week(today):
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


def _last_month(today):
    return (_first_of_last_month(today), _last_of_last_month(today))


def _this_month(today):
    return (today.replace(day=1), today)


def _last_weekday(today: date, target_weekday: int) -> tuple[date, date]:
    days_back = (today.weekday() - target_weekday) % 7
    if days_back == 0:
        days_back = 7  # "on Monday" means last Monday, not today
    d = today - timedelta(days=days_back)
    return (d, d)


def _same_day(today):
    return (today, today)


def _recently(today):
    return (today - timedelta(days=7), today)


def _the_other_day(today):
    return (today - timedelta(days=3), today - timedelta(days=1))


# Patterns ordered from most specific to least specific
_TEMPORAL_PATTERNS = [
    # Relative days — longer phrases first
    (r"\bthe day before yesterday\b", _day_before_yesterday),
    (r"\byesterday\b", _yesterday),
    (r"\btoday\b", _same_day),
    # Relative weeks
    (r"\blast week\b", _last_week),
    (r"\bthis week\b", _this_week),
    # Relative months
    (r"\blast month\b", _last_month),
    (r"\bthis month\b", _this_month),
    # N days/weeks ago
    (r"\b(\d+)\s+days?\s+ago\b", _n_days_ago),
    (r"\b(\d+)\s+weeks?\s+ago\b", _n_weeks_ago),
    # Day names (interpret as most recent occurrence)
    (r"\b(?:on\s+|last\s+)?(monday)\b", lambda t: _last_weekday(t, 0)),
    (r"\b(?:on\s+|last\s+)?(tuesday)\b", lambda t: _last_weekday(t, 1)),
    (r"\b(?:on\s+|last\s+)?(wednesday)\b", lambda t: _last_weekday(t, 2)),
    (r"\b(?:on\s+|last\s+)?(thursday)\b", lambda t: _last_weekday(t, 3)),
    (r"\b(?:on\s+|last\s+)?(friday)\b", lambda t: _last_weekday(t, 4)),
    (r"\b(?:on\s+|last\s+)?(saturday)\b", lambda t: _last_weekday(t, 5)),
    (r"\b(?:on\s+|last\s+)?(sunday)\b", lambda t: _last_weekday(t, 6)),
    # Time of day (same day)
    (r"\bthis morning\b", _same_day),
    (r"\btonight\b", _same_day),
    (r"\bthis afternoon\b", _same_day),
    (r"\bthis evening\b", _same_day),
    # Recent past
    (r"\brecently\b", _recently),
    (r"\bthe other day\b", _the_other_day),
]

# Compile patterns once
_COMPILED_PATTERNS = [(re.compile(p, re.IGNORECASE), fn) for p, fn in _TEMPORAL_PATTERNS]


def detect_temporal_query(
    text: str, reference_date: date | None = None
) -> tuple[date, date] | None:
    """Detect temporal references in text and resolve to a date range.

    Returns (start_date, end_date) inclusive, or None if no temporal
    reference is found.
    """
    today = reference_date or date.today()
    text_lower = text.lower()

    for pattern, resolver in _COMPILED_PATTERNS:
        match = pattern.search(text_lower)
        if match:
            groups = match.groups()
            if groups and groups[0].isdigit():
                result = resolver(today, groups[0])
            else:
                result = resolver(today)
            return result

    return None


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
