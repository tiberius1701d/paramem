"""Interim-adapter lifecycle helpers for multi-adapter interim routing.

This module owns two operations that must stay co-located so Step 7's
consolidate_interim_adapters can call unload without importing app.py:

  create_interim_adapter  — live creation of the current episodic_interim_* adapter
  unload_interim_adapters — post-consolidation removal of all interim adapters

It also provides timestamp and schedule helpers:

  current_interim_stamp(schedule, max_interim_count) — returns the current
      sub-interval stamp as ``YYYYMMDDTHHMM``, floored to the boundary of the
      current sub-interval.  The zero-argument form (backward compat) uses the
      current minute as the stamp.
  compute_schedule_period_seconds(schedule) — returns the full consolidation
      period in seconds for a schedule string.

Callers (wiring schedule):
  Step 6 — calls create_interim_adapter after each conversation is processed.
  Step 7 — calls unload_interim_adapters inside consolidate_interim_adapters
            as phase 3 of the atomic finalize sequence.

SOLE-ADAPTER TRAP NOTE: unload_interim_adapters is only safe to call while
the three main adapters (episodic, semantic, procedural) are still loaded.
The sole-adapter trap (delete_adapter → create_adapter on a PeftModel with
zero remaining adapters) does NOT apply here because the mains survive the
call. Do NOT call unload_interim_adapters before confirming the main adapters
are present in model.peft_config.
"""

from __future__ import annotations

import logging
import re
import shutil
from datetime import datetime, timedelta
from pathlib import Path

from peft import PeftModel

from paramem.models.loader import create_adapter
from paramem.utils.config import AdapterConfig

logger = logging.getLogger(__name__)

# Shared grammar — same as systemd_timer.parse_schedule so there is one
# source of truth for schedule string syntax.
_INTERVAL_RE = re.compile(r"^every\s+(\d+)\s*([hm])$", re.IGNORECASE)
_DAILY_RE = re.compile(r"^(\d{1,2}):(\d{2})$")
_OFF_VALUES = frozenset(("", "off", "disabled", "none"))


def compute_schedule_period_seconds(schedule: str) -> int | None:
    """Return the full consolidation period in seconds for a schedule string.

    Accepted formats (same grammar as ``systemd_timer.parse_schedule``):

    - ``"weekly"``    → 604800 seconds (7 days)
    - ``"daily"``     → 86400 seconds (1 day)
    - ``"every Nh"``  → N × 3600 seconds
    - ``"every Nm"``  → N × 60 seconds
    - ``"HH:MM"``     → 86400 seconds (daily)
    - ``""`` / ``"off"`` / ``"disabled"`` / ``"none"`` → ``None`` (manual only)

    Returns:
        Period in seconds, or ``None`` when no schedule is configured.

    Raises:
        ValueError: if *schedule* is not ``None`` / off / a valid period string.
    """
    s = (schedule or "").strip()
    if s.lower() in _OFF_VALUES:
        return None

    if s.lower() == "weekly":
        return 604800

    if s.lower() == "daily":
        return 86400

    m = _INTERVAL_RE.match(s)
    if m:
        n, unit = int(m.group(1)), m.group(2).lower()
        if n <= 0:
            raise ValueError(f"Invalid schedule period: {schedule!r}")
        return n * 3600 if unit == "h" else n * 60

    m = _DAILY_RE.match(s)
    if m:
        hh, mm = int(m.group(1)), int(m.group(2))
        if 0 <= hh < 24 and 0 <= mm < 60:
            return 86400
        raise ValueError(f"Invalid HH:MM schedule: {schedule!r}")

    raise ValueError(
        f"Unrecognised schedule string: {schedule!r}. "
        "Expected '', 'off', 'weekly', 'daily', 'HH:MM', 'every Nh', or 'every Nm'."
    )


def current_interim_stamp(
    schedule: str = "",
    max_interim_count: int = 0,
    *,
    _now: datetime | None = None,
) -> str:
    """Return the current sub-interval's ``YYYYMMDDTHHMM`` stamp.

    The stamp is floored to the nearest sub-interval boundary measured from
    midnight of the current local day.  Two calls within the same sub-interval
    return the same stamp, so a single interim adapter is reused for the entire
    interval.

    **When to call the zero-arg form.**  Pass no arguments only when you do not
    have a schedule string and just want the current minute — for example when
    generating a one-off stamp in a test.  The zero-arg form always returns the
    current minute without any sub-interval flooring.

    Args:
        schedule: Consolidation schedule string (``"every 2h"``, ``"03:00"``,
            ``""``, etc.).  ``""`` / off-variants → fall back to flooring to
            the nearest hour (sub-intervals still sensible, just coarser).
        max_interim_count: Number of sub-intervals per consolidation period.
            **Must be >= 1** when a real sub-interval calculation is desired.
            Pass ``0`` only via the zero-arg form to get the raw current minute
            stamp.  Callers must check ``max_interim_count == 0`` and handle
            the queue-until-consolidation branch *before* calling this function.

    Raises:
        ValueError: when ``max_interim_count < 0`` or when ``max_interim_count``
            is explicitly set to ``0`` alongside a non-empty schedule (the
            queue-until-consolidation branch must be handled by the caller).

    Returns:
        Timestamp string, e.g. ``"20260418T1430"`` for 2026-04-18 14:30 local.
    """
    now = _now if _now is not None else datetime.now()

    # Zero-arg backward-compatible form: return current minute.
    if not schedule and max_interim_count == 0:
        return now.strftime("%Y%m%dT%H%M")

    if max_interim_count < 0:
        raise ValueError(f"max_interim_count must be >= 0, got {max_interim_count}")
    if max_interim_count == 0:
        raise ValueError(
            "max_interim_count == 0 means 'queue until consolidation' — "
            "callers must handle this branch before calling current_interim_stamp()."
        )

    period = compute_schedule_period_seconds(schedule)

    if period is None:
        # Manual schedule: no period defined — floor to nearest hour so adapter
        # names remain sensible boundaries even without a configured schedule.
        sub_interval = 3600
    else:
        sub_interval = period // max_interim_count
        if sub_interval <= 0:
            sub_interval = 1  # guard against misconfiguration

    # Floor to the nearest sub-interval boundary measured from midnight local time.
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    seconds_since_midnight = int((now - midnight).total_seconds())
    floored_seconds = (seconds_since_midnight // sub_interval) * sub_interval

    floored_dt = midnight + timedelta(seconds=floored_seconds)
    return floored_dt.strftime("%Y%m%dT%H%M")


def create_interim_adapter(
    model: PeftModel,
    adapter_config: AdapterConfig,
    stamp: str,
) -> PeftModel:
    """Create an episodic interim adapter on the live model.

    Idempotent: if the adapter for *stamp* already exists in model.peft_config
    the model is returned unchanged.  The caller is responsible for switching
    the active adapter back to "episodic" (main) after any training on the new
    interim adapter is complete.

    Args:
        model: Live PeftModel that already has the main adapters loaded.
        adapter_config: LoRA config to use for the new adapter (should match
            the episodic_adapter_config from server config so all interim
            adapters are topology-compatible with the main episodic adapter).
        stamp: ISO 8601 basic timestamp string (``YYYYMMDDTHHMM``) used as the
            adapter-name suffix, e.g. ``"20260418T1430"`` →
            ``"episodic_interim_20260418T1430"``.

    Returns:
        Updated PeftModel (same object when the adapter already exists;
        may be re-assigned by create_adapter when adding a new adapter).
    """
    name = f"episodic_interim_{stamp}"
    if name in model.peft_config:
        logger.debug("Interim adapter already exists for %s — no-op", stamp)
        return model
    model = create_adapter(model, adapter_config, adapter_name=name)
    logger.info("Created interim adapter: %s", name)
    return model


def unload_interim_adapters(model: PeftModel, adapter_dir: Path) -> list[str]:
    """Delete every loaded episodic_interim_* adapter from PEFT and its on-disk dir.

    This is phase 3 of Step 7's atomic finalize sequence.  Call it ONLY after:
      1. Registry rewrite (adapter_id values updated to main tier names).
      2. On-disk delete of interim adapter dirs (done here for completeness).
    And BEFORE:
      3. Router.reload() — which must not see any episodic_interim_* dirs.

    Phase ordering relative to registry rewrite and Router.reload() is the
    caller's responsibility; see Step 7 of the multi-adapter interim routing plan.

    The three main adapters (episodic, semantic, procedural) remain loaded
    throughout — the sole-adapter trap does not apply.

    Args:
        model: Live PeftModel.  Must contain at least one main adapter so
            delete_adapter never removes the last adapter from the model.
        adapter_dir: Parent directory (config.adapter_dir) whose
            episodic_interim_* subdirectories are removed.

    Returns:
        Sorted list of adapter names that were unloaded (may be empty).
    """
    interim_names = sorted(n for n in model.peft_config if n.startswith("episodic_interim_"))
    for name in interim_names:
        model.delete_adapter(name)
        logger.info("Deleted interim adapter from PEFT: %s", name)

    for path in sorted(adapter_dir.glob("episodic_interim_*")):
        if path.is_dir():
            shutil.rmtree(path)
            logger.info("Removed interim adapter directory: %s", path)

    return interim_names
