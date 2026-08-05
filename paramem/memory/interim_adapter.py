"""Interim-adapter lifecycle helpers for multi-adapter interim routing.

This module owns two operations that must stay co-located so the full
consolidation fold can call unload without importing app.py:

  create_interim_adapter  — live creation of the current episodic_interim_* adapter
  unload_interim_adapters — post-consolidation reap of all interim slots
                            (PEFT adapters where they exist, on-disk dirs always)

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
  POST /interim/discard (paramem.server.app) — calls unload_interim_adapters
      to reap the ring without folding it into the main tiers first.

SOLE-ADAPTER TRAP NOTE: unload_interim_adapters is only safe to call while
the three main adapters (episodic, semantic, procedural) are still loaded.
The sole-adapter trap (delete_adapter → create_adapter on a PeftModel with
zero remaining adapters) does NOT apply here because the mains survive the
call. Do NOT call unload_interim_adapters before confirming the main adapters
are present in model.peft_config.

ACTIVE-ADAPTER DETERMINISM: see :func:`paramem.models.loader.detach_adapters`'s
docstring — the switch-before-delete guard unload_interim_adapters relies on
now lives there, not in this module.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import datetime, timedelta
from pathlib import Path

from peft import PeftModel

from paramem.memory.persistence import reap_tier_artifacts
from paramem.models.loader import create_adapter, detach_adapters
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

# THE single declaration of the interim stamp format.  Every mint
# (:func:`current_interim_stamp`), every validation
# (:func:`interim_stamp_from_name`) and every re-parse to a datetime
# (``app._full_cycle_deadline_dt``) composes from this one constant — the
# shape is never re-declared as a pattern or a literal length.
INTERIM_STAMP_FORMAT = "%Y%m%dT%H%M"


def interim_stamp_from_name(name: str) -> str | None:
    """Return the validated stamp from an interim adapter name, else ``None``.

    ``"episodic_interim_20260417T0000"`` -> ``"20260417T0000"``.  Returns
    ``None`` when *name* does not carry :data:`INTERIM_NAME_PREFIX`, or when
    the tail is not a real :data:`INTERIM_STAMP_FORMAT` stamp.

    THE only place an interim adapter name is parsed.  Validation is a
    round-trip through :data:`INTERIM_STAMP_FORMAT` — ``strptime`` alone
    accepts short fields (``"2026041T000"`` parses), so the re-rendered
    stamp must equal the input.  This makes the format constant the sole
    authority on the shape: there is no second length or pattern to keep
    in step, and unlike a shape-only check it also rejects stamps that are
    well-formed but not real datetimes (``"99999999T9999"``).

    Total by contract: callers that require a stamp raise on ``None``
    themselves, at the point where the failure means something specific.
    """
    if not name.startswith(INTERIM_NAME_PREFIX):
        return None
    stamp = name[len(INTERIM_NAME_PREFIX) :]
    try:
        parsed = datetime.strptime(stamp, INTERIM_STAMP_FORMAT)
    except ValueError:
        return None
    return stamp if parsed.strftime(INTERIM_STAMP_FORMAT) == stamp else None


def interim_tiers_newest_first(store) -> list[str]:
    """Interim slot names carried by *store*, newest stamp first.

    THE only enumeration of interim tiers for probe ordering AND for dedup
    scope — every caller composes this rather than repeating the
    filter-then-sort: :meth:`~paramem.server.router.QueryRouter._personal_tier_order`
    and :meth:`~paramem.server.router.QueryRouter._command_interim_tiers`
    order probes with it, and the interim consolidation fold's recital-dedup
    scan widens its candidate scan with it (main tiers alone miss a fact
    keyed in a sibling interim slot).  Name parsing is
    :func:`interim_stamp_from_name`; this function does not re-declare the
    name shape.

    A ``None`` *store* (no registry — replay-disabled) yields an empty list,
    which is what makes interim slots silently unreachable in that
    configuration.
    """
    if store is None:
        return []
    stamped = [
        (stamp, tier)
        for tier in store.tiers_with_registry()
        if (stamp := interim_stamp_from_name(tier)) is not None
    ]
    # Sort on the stamp alone: Python's stable sort then preserves the
    # store's enumeration order for any two slots sharing a stamp.
    stamped.sort(key=lambda pair: pair[0], reverse=True)
    return [tier for _, tier in stamped]


def interim_dir_for_name(adapter_dir: Path, name: str) -> Path:
    """Return the on-disk directory for a PEFT interim adapter name.

    Maps ``"episodic_interim_<stamp>"`` →
    ``<adapter_dir>/episodic/interim_<stamp>/``.

    Raises:
        ValueError: if *name* is not a well-formed interim adapter name —
            building a path from an unvalidated stamp would silently create
            a directory like ``interim_today/``.
    """
    stamp = interim_stamp_from_name(name)
    if stamp is None:
        raise ValueError(
            f"Not an interim adapter name: {name!r} "
            f"(expected '{INTERIM_NAME_PREFIX}<{INTERIM_STAMP_FORMAT}>')"
        )
    return adapter_dir / "episodic" / f"{INTERIM_DIR_PREFIX}{stamp}"


def slot_payload_kind(path: Path) -> str | None:
    """Classify the payload one adapter slot directory carries.

    ``"train"`` when *path* (or any of its own subdirectories) carries
    adapter weights (``adapter_model.safetensors`` anywhere beneath *path*),
    ``"simulate"`` when it carries only a ``graph.json``, and ``None`` when
    it carries neither. Weights win over a co-resident ``graph.json`` — a
    slot only ever legitimately holds one venue's payload, so this ordering
    exists to make an accidental mix classify sanely rather than to choose
    between two valid shapes.

    The weights scan prunes ``interim_*`` children before descending into
    them, so a MAIN tier root's signal (``<adapter_dir>/<tier>/``) is never
    satisfied by a sibling interim slot living underneath it — an episodic
    root with no weights of its own reads as payload-less even when a child
    ``interim_<stamp>/`` carries weights, because that child is a distinct
    tier. The prune is a no-op for an interim slot root
    (``<adapter_dir>/episodic/interim_<stamp>/``), which never nests another
    ``interim_*`` directory, so this one predicate applies to both shapes.

    THE single "does this slot carry content" predicate — both
    :func:`iter_interim_dirs`'s ``mode`` filter and the boot-time
    keyless-tier sweep
    (:func:`paramem.server.app._sweep_keyless_tier_artifacts`) read this,
    not a re-derived ``rglob``/``.exists()`` pair, so the venue -> payload
    mapping cannot drift between the two callers.

    Args:
        path: Slot root directory to classify.

    Returns:
        ``"train"``, ``"simulate"``, or ``None``.
    """
    import os

    for _root, dirnames, filenames in os.walk(path):
        dirnames[:] = [d for d in dirnames if not d.startswith(INTERIM_DIR_PREFIX)]
        if "adapter_model.safetensors" in filenames:
            return "train"
    if (path / "graph.json").exists():
        return "simulate"
    return None


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
    nothing to fold.  ``mode`` selects which of the two sets the caller wants;
    the venue -> payload classification itself is :func:`slot_payload_kind`,
    called from here so no caller has to re-implement it.

    Args:
        adapter_dir: Adapter root (``config.adapter_dir``).
        mode: Payload filter.

            * ``None`` (default) — every interim directory on disk, regardless
              of payload.  Required by the reaper, backup, registry hydration
              and every other caller that must see the whole on-disk set.
            * ``"simulate"`` — only slots :func:`slot_payload_kind` classifies
              ``"simulate"`` (the simulate venue's payload).
            * ``"train"`` — only slots :func:`slot_payload_kind` classifies
              ``"train"`` (the train venue's payload).

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
        if mode is not None:
            kind = slot_payload_kind(path)
            if mode == "simulate" and kind != "simulate":
                continue
            if mode == "train" and kind != "train":
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
    return floored_dt.strftime(INTERIM_STAMP_FORMAT)


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
    in: two calls within the same window return the same stamp.  Its only
    consumer is the main-slot manifest's ``window_stamp`` field, which is
    written as provenance and read back by no gate.

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
    name = f"{INTERIM_NAME_PREFIX}{stamp}"
    if name in model.peft_config:
        logger.debug("Interim adapter already exists for %s — no-op", stamp)
        return model
    model = create_adapter(model, adapter_config, adapter_name=name)
    logger.info("Created interim adapter: %s", name)
    return model


def unload_interim_adapters(model, adapter_dir: Path) -> list[str]:
    """Reap every interim slot: the PEFT adapters (when any) and the on-disk dirs.

    This is phase 3 of the consolidation finalize sequence.  It must run AFTER
    the registry rewrite that rebooks interim keys onto the main tiers, so no
    live registry still points at a slot this call removes.

    **Both fold venues call this.**  The weights venue has PEFT interim adapters
    mounted and an on-disk slot dir per adapter; the disk venue has only the
    on-disk slot dirs (``self.model`` there is a bare base model, not a
    :class:`~peft.PeftModel`, and holds no ``peft_config``).  The PEFT half is
    therefore skipped when *model* is not a ``PeftModel`` — the on-disk reap is
    unconditional and is the same reap in both venues.  Do not write a second
    reaper for the disk venue.

    The three main adapters (episodic, semantic, procedural) remain loaded
    throughout — the sole-adapter trap does not apply.

    The PEFT half delegates to :func:`paramem.models.loader.detach_adapters`,
    which switches the active adapter onto a resident survivor (episodic
    first) before deleting any interim adapter, so the post-reap active
    adapter is deterministic regardless of which adapter happened to be
    active when this was called — see that function's docstring for the
    switch-before-delete rationale.  The disk half delegates to
    :func:`paramem.memory.persistence.reap_tier_artifacts`, one call per
    interim slot directory yielded by :func:`iter_interim_dirs`.

    Args:
        model: Live model.  A :class:`~peft.PeftModel` has its interim adapters
            deleted and must contain at least one main adapter so
            ``delete_adapter`` never removes the last adapter.  Anything else
            (bare base model, ``None``) skips the PEFT half.
        adapter_dir: Parent directory (config.adapter_dir) whose
            episodic_interim_* subdirectories are removed.

    Returns:
        Sorted list of adapter names that were unloaded from PEFT (empty in the
        disk venue, where there are none).
    """
    interim_names = (
        sorted(n for n in model.peft_config if n.startswith(INTERIM_NAME_PREFIX))
        if isinstance(model, PeftModel)
        else []
    )
    detach_adapters(model, interim_names)

    # UNFILTERED ON PURPOSE — never pass ``mode`` here.  The reap must see EVERY
    # interim directory, payload-bearing or not.  A payload-filtered reap would
    # leave empty/torn slot dirs behind on every fold and they would accumulate
    # forever.  ``mode`` narrows the set to slots that carry content; that is the
    # right question for the schedule gate and the fold collector, and the wrong
    # question for a reaper.  The path is the one ``iter_interim_dirs`` yields —
    # never re-derived via ``interim_dir_for_name``, which raises on a stray dir
    # whose stamp is malformed; the whole-ring reap must still remove those.
    for _name, path in iter_interim_dirs(adapter_dir):
        removed = reap_tier_artifacts(path)
        logger.info("Removed interim adapter directory: %s (%d paths)", path, len(removed))

    return interim_names
