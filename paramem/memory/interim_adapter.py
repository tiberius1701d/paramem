"""Interim-adapter lifecycle helpers for multi-adapter interim routing.

This module owns two operations that must stay co-located so the full
consolidation fold can call unload without importing app.py:

  create_interim_adapter  — live creation of the current episodic_interim_* adapter
  unload_interim_adapters — post-consolidation removal of all interim adapters

It also provides a timestamp helper:

  current_interim_stamp(refresh_cadence) — returns the current
      sub-interval stamp as ``YYYYMMDDTHHMM``, floored to the boundary of the
      current sub-interval.

Schedule-string parsing (``compute_schedule_period_seconds``) lives in
``paramem.server.schedule_grammar`` — relocated there (2026-07) so the
backup runner can share it without ``interim_adapter`` (a ``memory``-layer
module) importing from ``backup``.

Callers (wiring schedule):
  Scheduled consolidation path — calls create_interim_adapter when run_consolidation_cycle
      mints a new interim adapter slot during an interim training tick.
  Full consolidation fold (ConsolidationLoop.consolidate) — calls
      unload_interim_adapters as phase 3 of the atomic finalize sequence.

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
from collections.abc import Iterator
from datetime import datetime, timedelta
from pathlib import Path

from peft import PeftModel

from paramem.models.loader import create_adapter
from paramem.server.schedule_grammar import compute_schedule_period_seconds
from paramem.utils.config import AdapterConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# On-disk layout helpers (2026-05-14 hierarchy refactor)
# ---------------------------------------------------------------------------
#
# PEFT adapter NAME is decoupled from on-disk DIR.  The NAME stays
# ``"episodic_interim_<stamp>"`` so router patterns, inference, and
# ``startswith("episodic_interim_")`` checks remain unchanged.  The DIR is
# nested under ``<adapter_dir>/episodic/`` to mirror the conceptual hierarchy:
#
#   <adapter_dir>/
#     episodic/
#       <slot_date>/                 ← main episodic slot
#       interim_<stamp>/<slot_date>/ ← interim under episodic
#     semantic/<slot_date>/
#     procedural/<slot_date>/
#
# Use these helpers everywhere a path is built or scanned; never glob the
# legacy flat ``adapter_dir/episodic_interim_*`` pattern in new code.


INTERIM_NAME_PREFIX = "episodic_interim_"
INTERIM_DIR_PREFIX = "interim_"


def interim_stamp_from_name(name: str) -> str:
    """Extract the YYYYMMDDTHHMM stamp from an interim adapter name."""
    if not name.startswith(INTERIM_NAME_PREFIX):
        raise ValueError(
            f"Not an interim adapter name: {name!r} (expected '{INTERIM_NAME_PREFIX}<stamp>')"
        )
    return name[len(INTERIM_NAME_PREFIX) :]


def interim_dir_for_name(adapter_dir: Path, name: str) -> Path:
    """Return the on-disk directory for a PEFT interim adapter name.

    Maps ``"episodic_interim_<stamp>"`` →
    ``<adapter_dir>/episodic/interim_<stamp>/``.
    """
    return adapter_dir / "episodic" / f"{INTERIM_DIR_PREFIX}{interim_stamp_from_name(name)}"


def iter_interim_dirs(
    adapter_dir: Path,
    *,
    mode: str | None = None,
) -> Iterator[tuple[str, Path]]:
    """Yield ``(adapter_name, dir_path)`` for interim slots on disk.

    Scans ``<adapter_dir>/episodic/interim_*`` and synthesises the PEFT
    adapter name as ``"episodic_interim_<stamp>"``.

    An interim *directory* is not the same thing as an interim slot that holds
    *content*: a slot whose payload write never landed (crash between the
    directory creation and the payload flush) is an empty shell that carries
    nothing to fold.  ``mode`` selects which of the two sets the caller wants,
    and the venue→payload mapping lives here — in one place — so no caller has
    to re-implement it.

    Args:
        adapter_dir: Adapter root (``config.adapter_dir``).
        mode: Payload filter.

            * ``None`` (default) — every interim directory on disk, regardless
              of payload.  Required by the reaper, backup, registry hydration
              and every other caller that must see the whole on-disk set.
            * ``"simulate"`` — only slots carrying a ``graph.json`` (the
              simulate venue's payload).
            * ``"train"`` — only slots carrying ``adapter_model.safetensors``
              anywhere beneath the slot dir (the train venue's payload).  Keyed
              on the weights file, not ``adapter_config.json``: a config without
              weights is a broken slot, hence payload-less.

    Raises:
        ValueError: On an unknown *mode* string.  Silently degrading to the
            unfiltered set would let a typo re-open the payload-blind
            behaviour this parameter exists to close.
    """
    if mode not in (None, "simulate", "train"):
        raise ValueError(f"iter_interim_dirs: unknown mode {mode!r} (expected None/simulate/train)")

    episodic = adapter_dir / "episodic"
    if not episodic.is_dir():
        return
    for path in sorted(episodic.glob(f"{INTERIM_DIR_PREFIX}*")):
        if not path.is_dir():
            continue
        if mode == "simulate" and not (path / "graph.json").exists():
            continue
        if mode == "train" and not any(path.rglob("adapter_model.safetensors")):
            continue
        stamp = path.name[len(INTERIM_DIR_PREFIX) :]
        yield f"{INTERIM_NAME_PREFIX}{stamp}", path


def adapter_slot_root_for_name(adapter_dir: Path, name: str) -> Path:
    """Return the slot-root directory for any adapter name.

    Main tiers map directly to ``<adapter_dir>/<name>/``.  Interim adapters
    map to ``<adapter_dir>/episodic/interim_<stamp>/`` per the 2026-05-14
    hierarchy refactor.  Use this helper at every callsite that writes or
    reads an adapter slot dir by NAME so the on-disk layout follows one
    rule.
    """
    if name.startswith(INTERIM_NAME_PREFIX):
        return interim_dir_for_name(adapter_dir, name)
    return adapter_dir / name


def detect_legacy_adapter_layout(adapter_dir: Path) -> list[Path]:
    """Return any legacy top-level ``episodic_interim_<stamp>`` dirs.

    Used by the boot lifespan to refuse start until the migration script
    has been run.  Empty list = clean layout.
    """
    if not adapter_dir.is_dir():
        return []
    legacy: list[Path] = []
    for path in adapter_dir.glob(f"{INTERIM_NAME_PREFIX}*"):
        if path.is_dir():
            legacy.append(path)
    return sorted(legacy)


def current_interim_stamp(
    refresh_cadence: str,
    *,
    _now: datetime | None = None,
) -> str:
    """Return the current refresh-interval's ``YYYYMMDDTHHMM`` stamp.

    ``refresh_cadence`` IS the sub-interval directly — no division by
    ``max_interim_count``. The stamp is floored to the nearest cadence
    boundary measured from midnight of the current local day, so two calls
    within the same cadence window return the same stamp and a single
    interim adapter is reused for the entire window.

    Args:
        refresh_cadence: Interim refresh cadence (``"every 12h"``,
            ``"every 30m"``, ``"daily"``, ``"HH:MM"``, etc.).  An off-variant
            falls back to hourly flooring so adapter names remain sensible
            boundaries even without a configured cadence.

    Returns:
        Timestamp string, e.g. ``"20260418T1430"`` for 2026-04-18 14:30 local.
    """
    now = _now if _now is not None else datetime.now()

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

    This is phase 3 of the consolidation finalize sequence.  Call it ONLY after:
      1. Registry rewrite (adapter_id values updated to main tier names).
      2. On-disk delete of interim adapter dirs (done here for completeness).
    And BEFORE:
      3. Router.reload() — which must not see any episodic_interim_* dirs.

    Phase ordering relative to registry rewrite and Router.reload() is the
    caller's responsibility: registry rewrite must complete before this call,
    and Router.reload() must not be called until after this call returns.

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
    interim_names = sorted(n for n in model.peft_config if n.startswith(INTERIM_NAME_PREFIX))
    for name in interim_names:
        model.delete_adapter(name)
        logger.info("Deleted interim adapter from PEFT: %s", name)

    # UNFILTERED ON PURPOSE — never pass ``mode`` here.  The reap must see EVERY
    # interim directory, payload-bearing or not.  A payload-filtered reap would
    # leave empty/torn slot dirs behind on every fold and they would accumulate
    # forever.  ``mode`` narrows the set to slots that carry content; that is the
    # right question for the schedule gate and the fold collector, and the wrong
    # question for a reaper.
    for _name, path in iter_interim_dirs(adapter_dir):
        shutil.rmtree(path)
        logger.info("Removed interim adapter directory: %s", path)

    return interim_names
