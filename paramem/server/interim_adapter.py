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
import shutil
from datetime import datetime, timedelta
from pathlib import Path

from peft import PeftModel

from paramem.models.loader import create_adapter
from paramem.server.schedule_grammar import parse_schedule_atom
from paramem.utils.config import AdapterConfig

logger = logging.getLogger(__name__)


def compute_schedule_period_seconds(schedule: str) -> int | None:
    """Return the full consolidation period in seconds for a schedule string.

    Grammar is shared with ``systemd_timer.parse_schedule`` via
    :func:`paramem.server.schedule_grammar.parse_schedule_atom`.

    - ``"weekly"``                → 604800 seconds (7 days)
    - ``"daily"``                 → 86400 seconds (1 day)
    - ``"every Nh"`` / ``"Nh"``   → N × 3600 seconds
    - ``"every Nm"`` / ``"Nm"``   → N × 60 seconds
    - ``"HH:MM"``                 → 86400 seconds (daily)
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


def current_interim_stamp(
    refresh_cadence: str = "",
    *,
    _now: datetime | None = None,
) -> str:
    """Return the current refresh-interval's ``YYYYMMDDTHHMM`` stamp.

    ``refresh_cadence`` IS the sub-interval directly — no division by
    ``max_interim_count``. The stamp is floored to the nearest cadence
    boundary measured from midnight of the current local day, so two calls
    within the same cadence window return the same stamp and a single
    interim adapter is reused for the entire window.

    **When to call the zero-arg form.** Pass no arguments when you just
    want the current minute — for example to generate a one-off stamp in a
    test. The zero-arg form returns the raw current minute with no
    flooring.

    Args:
        refresh_cadence: Interim refresh cadence (``"every 12h"``,
            ``"every 30m"``, ``"daily"``, ``"HH:MM"``, etc.). An empty /
            off-variant falls back to flooring to the nearest hour so
            adapter names remain sensible boundaries even without a
            configured cadence.

    Returns:
        Timestamp string, e.g. ``"20260418T1430"`` for 2026-04-18 14:30 local.
    """
    now = _now if _now is not None else datetime.now()

    # Zero-arg form: return current minute with no flooring.
    if not refresh_cadence:
        return now.strftime("%Y%m%dT%H%M")

    sub_interval = compute_schedule_period_seconds(refresh_cadence)
    if sub_interval is None:
        # Off-variant (``"off"``/``"disabled"``/``"none"``): no cadence configured.
        # Fall back to hourly flooring so stamps stay sensible. Callers that
        # truly want to skip stamping should handle the queue-branch earlier.
        sub_interval = 3600
    if sub_interval <= 0:
        sub_interval = 1  # guard against misconfiguration

    # Floor to the nearest refresh-cadence boundary measured from midnight local time.
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    seconds_since_midnight = int((now - midnight).total_seconds())
    floored_seconds = (seconds_since_midnight // sub_interval) * sub_interval

    floored_dt = midnight + timedelta(seconds=floored_seconds)
    return floored_dt.strftime("%Y%m%dT%H%M")


def current_full_consolidation_stamp(
    consolidation_period: str = "",
    *,
    _now: datetime | None = None,
) -> str:
    """Return the current full-consolidation window's ``YYYYMMDDTHHMM`` stamp.

    Companion to :func:`current_interim_stamp`.  Identical flooring logic
    (anchored to local midnight) but applied to the FULL consolidation
    period (``refresh_cadence × max_interim_count``) instead of the interim
    cadence.  The stamp identifies which full-cycle window we are currently
    in: two calls within the same window return the same stamp, which is
    how the Phase-4 gate decides whether the current window has already
    been consolidated (compared against the latest main slot's
    ``meta.json.window_stamp``).

    Args:
        consolidation_period: Full-cycle period string from
            ``ConsolidationConfig.consolidation_period_string``.  Empty
            string disables the gate (manual-only).

    Returns:
        ``"YYYYMMDDTHHMM"`` for the floored window boundary, or empty
        string when *consolidation_period* is empty/disabled.
    """
    if not consolidation_period:
        return ""
    return current_interim_stamp(consolidation_period, _now=_now)


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
