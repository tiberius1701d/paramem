"""Server consolidation — thin wrapper around ConsolidationLoop.

Uses the same ConsolidationLoop that powers Tests 1-8, with
indexed_key_replay=True. The graph is transient (RAM-only).
Promotion is key-level: per-key session counts persisted in
key_metadata.json (no personal data on disk).

The loop saves adapters directly to output_dir (= adapter_dir), so
the router can reload from the standard paths without any bridging.
"""

import enum
import json
import logging
from pathlib import Path

from paramem.backup.encryption import read_maybe_encrypted, write_infra_bytes
from paramem.server.config import ServerConfig
from paramem.training.consolidation import ConsolidationLoop
from paramem.training.thermal_throttle import ThermalPolicy

logger = logging.getLogger(__name__)


class SessionClass(enum.Enum):
    """3-way classification of a pending session's attribution state.

    NAMED         — speaker_id is present and not anonymous_voice.
                    Extract and train immediately.
    HOLDABLE      — speaker is anonymous (anonymous_voice enroll_method), OR
                    the session has no speaker_id but at least one user turn
                    carries a voice embedding (may be attributed later via
                    retro-claim).  Hold pending; retire only past the
                    orphan_retirement TTL.
    UNIDENTIFIABLE — no speaker_id AND no voice embedding anywhere — an
                    unauthenticated text session that can never be attributed.
                    Drop immediately; no TTL.
    """

    NAMED = "named"
    HOLDABLE = "holdable"
    UNIDENTIFIABLE = "unidentifiable"


def classify_session(
    *,
    speaker_id: str | None,
    is_anonymous: bool,
    has_voice_embedding: bool,
) -> SessionClass:
    """Classify a pending session by attribution state.

    Pure function — callers resolve ``is_anonymous`` via
    ``store.is_anonymous(speaker_id)`` before calling; pass ``False`` when
    the speaker store is unavailable.

    Parameters
    ----------
    speaker_id:
        Dominant speaker id from the session (may be ``None``).
    is_anonymous:
        ``True`` iff the speaker profile's ``enroll_method == "anonymous_voice"``.
        Callers obtain this via ``SpeakerStore.is_anonymous(speaker_id)``.
        When the store is ``None``, pass ``False``.
    has_voice_embedding:
        ``True`` iff any user-role turn in the session carries a non-``None``
        ``"embedding"`` field.  Obtained from :meth:`SessionBuffer.pending_facts`.

    Returns
    -------
    SessionClass
        NAMED if the speaker is enrolled and non-anonymous.
        HOLDABLE if anonymous or holds a voice embedding (retro-claimable).
        UNIDENTIFIABLE if no speaker_id and no embedding.
    """
    if speaker_id and not is_anonymous:
        return SessionClass.NAMED
    if is_anonymous or has_voice_embedding:
        return SessionClass.HOLDABLE
    return SessionClass.UNIDENTIFIABLE


def discard_session_sink(config: ServerConfig) -> Path:
    """Return the discard-sink directory for unattributable / retired-holdable sessions.

    Under ``config.debug_dir/discarded_sessions/``.  Distinct from
    :func:`session_retention_dir` (trained-session archive) so the trained
    archive is never polluted with dropped sessions.

    Under ``debug=False`` (privacy mode), :meth:`SessionBuffer.mark_consolidated`
    unlinks the JSONL unconditionally regardless of ``retention_dir`` — this
    sink is still passed so the code path is identical in both modes.
    """
    return config.debug_dir / "discarded_sessions"


def create_consolidation_loop(
    model,
    tokenizer,
    config: ServerConfig,
    memory_store,
    state_provider=None,
    *,
    output_dir=None,
    save_cycle_snapshots: bool | None = None,
    seed_state_from_disk: bool = True,
    keep_prior_slots: int | None = None,
) -> ConsolidationLoop:
    """Create a ConsolidationLoop configured for the server.

    Graph is transient (RAM-only). Key metadata is seeded
    from key_metadata.json to restore cycle count, promoted keys, and
    per-key bookkeeping (reinforcement_count, last_reinforced_cycle, last_seen,
    first_seen) across restarts.

    Parameters
    ----------
    state_provider:
        Optional zero-argument callable returning the server ``_state`` dict.
        Passed to ``ConsolidationLoop`` so ``run_cycle`` can call
        ``guard_trial_state`` and raise ``TrialActiveError`` when a migration
        TRIAL is active.  Experiment scripts that do not pass ``state_provider``
        are unaffected (default ``None`` → guard is a no-op).
    output_dir:
        Override ``config.adapter_dir`` as the loop's output directory.
        ``None`` (default) falls through to ``config.adapter_dir``, which
        preserves production behaviour.
    save_cycle_snapshots:
        Override ``config.debug`` as the cycle-snapshot toggle.  ``None``
        (default) falls through to ``config.debug``.
    seed_state_from_disk:
        When ``False``, skip seeding key metadata and keyed-pairs QA from
        disk.  Use for probe/experiment runs that must start from a clean
        state rather than inheriting live-system data.  Default ``True``
        preserves production behaviour.
    keep_prior_slots:
        Override ``config.consolidation.training_keep_prior_slots``.
        ``None`` (default) falls through to the config value.
    """
    _output_dir = output_dir if output_dir is not None else config.adapter_dir
    _save_cycle_snapshots = (
        save_cycle_snapshots if save_cycle_snapshots is not None else config.debug
    )

    loop = ConsolidationLoop(
        model=model,
        tokenizer=tokenizer,
        consolidation_config=config.consolidation_config,
        training_config=config.training_config,
        episodic_adapter_config=config.episodic_adapter_config,
        semantic_adapter_config=config.semantic_adapter_config,
        memory_store=memory_store,
        procedural_adapter_config=(
            config.procedural_adapter_config if config.adapters.procedural.enabled else None
        ),
        output_dir=_output_dir,
        extraction_temperature=0.0,
        extraction_max_tokens=config.consolidation.extraction_max_tokens,
        extraction_plausibility_max_tokens=config.consolidation.extraction_plausibility_max_tokens,
        save_cycle_snapshots=_save_cycle_snapshots,
        snapshot_dir=config.debug_dir if _save_cycle_snapshots else None,
        prompts_dir=config.prompts_dir,
        model_name=config.model_name,
        graph_config=config.graph_config,
        sota_enabled=config.consolidation.sota_enabled,
        graph_enrichment_neighborhood_hops=config.consolidation.graph_enrichment_neighborhood_hops,
        graph_enrichment_max_entities_per_pass=config.consolidation.graph_enrichment_max_entities_per_pass,
        # Same `cloud_scope` knob as inference-time cloud egress: the SOTA
        # enrichment cycle sends placeholders to the cloud just like the
        # cloud_anonymizer egress path, so the privacy policy must match.
        # Empty list disables NER-scope filtering entirely (consolidation
        # then uses the primitive default {person, place}).
        extraction_pii_scope=set(config.sanitization.cloud_scope),
        extraction_stt_correction=config.consolidation.extraction_stt_correction,
        extraction_ha_validation=config.consolidation.extraction_ha_validation,
        extraction_noise_filter=config.consolidation.extraction_noise_filter,
        extraction_noise_filter_model=config.consolidation.extraction_noise_filter_model,
        extraction_noise_filter_endpoint=config.consolidation.extraction_noise_filter_endpoint,
        extraction_ner_check=config.consolidation.extraction_ner_check,
        extraction_ner_model=config.consolidation.extraction_ner_model,
        extraction_plausibility_judge=config.consolidation.extraction_plausibility_judge,
        extraction_plausibility_stage=config.consolidation.extraction_plausibility_stage,
        extraction_verify_anonymization=config.consolidation.extraction_verify_anonymization,
        state_provider=state_provider,
        # Thermal fields live on ConsolidationScheduleConfig
        # (config.consolidation), NOT on ConsolidationConfig (which the loop
        # accepts as consolidation_config).  Build the policy here where both
        # configs are reachable, pass it to the loop precomputed.
        thermal_policy=ThermalPolicy.from_consolidation_config(config.consolidation),
        keep_prior_slots=(
            keep_prior_slots
            if keep_prior_slots is not None
            else config.consolidation.training_keep_prior_slots
        ),
    )

    # Wire the base-model weight-hash cache from server _state into the loop so
    # build_manifest_for memoizes the SHA-256 across consolidations within one
    # process lifetime. Without this, every cycle re-hashes the full base model
    # (~2 min for Mistral 7B). Cache is keyed by id(model); resets on restart.
    if state_provider is not None:
        state = state_provider()
        if state is not None:
            loop.fingerprint_cache = state.setdefault("base_model_hash_cache", {})

    # Wire the full-consolidation period string so _save_adapters can stamp
    # main slots with the current full-cycle window. Empty string in
    # experiment paths (state_provider=None) → window_stamp="" → Phase 4
    # gate treats those slots as unknown-window and forces a first full
    # cycle on adoption.
    loop.full_consolidation_period_string = config.consolidation.consolidation_period_string

    if seed_state_from_disk:
        # Key metadata (cycle counts, promotion bookkeeping) is loop-state
        # and still seeded here.  Entry payloads (subject/predicate/object/
        # speaker_id) live in the lifespan-owned MemoryStore — preload runs
        # at lifespan boot, not here, so the loop factory no longer touches
        # the model or reads graph.json for that purpose.
        metadata = _load_key_metadata(config.key_metadata_path)
        if metadata:
            loop.seed_key_metadata(metadata)

    return loop


def session_retention_dir(loop, config) -> Path | None:
    """Return the directory to retain consolidated session JSONL into.

    Returns ``None`` when neither retention nor debug mode is enabled —
    the SessionBuffer will unlink the JSONL after consume.  Otherwise
    returns ``loop.snapshot_dir_for(interim_stamp=...)/sessions/``
    (2026-05-14 hierarchy:
    ``paths.debug/episodic/[interim_<stamp>/]cycle_<N>/run_<run_id>/sessions/``).
    Falls back to ``config.debug_dir/cycle_<N>/sessions/`` when the loop
    cannot produce a snapshot path.
    """
    if not (config.consolidation.retain_sessions or config.debug):
        return None
    # _current_interim_stamp is never set on the loop (the attribute was removed
    # when the context-variable pattern was replaced by explicit parameter passing
    # in run_consolidation_cycle).  Pass None so snapshot_dir_for uses the
    # cycle-scoped path (paths.debug/episodic/cycle_<N>/run_<run_id>/).
    snap = None
    snap_fn = getattr(loop, "snapshot_dir_for", None)
    if callable(snap_fn):
        snap = snap_fn(interim_stamp=None)
    if snap is None:
        cycle = getattr(loop, "cycle_count", 0)
        return config.debug_dir / f"cycle_{cycle}" / "sessions"
    return snap / "sessions"


# run_consolidation was deleted when it was merged into paramem.server.app._run_extraction_phase,
# which closes over _state instead of taking model/tokenizer/config/session_buffer as args.
# Trial-migration callers: use _run_extraction_phase(loop, mark_sessions=False).
# Dev scripts that imported run_consolidation directly will need updating separately.


# --- Key-level promotion ---


# Dedup moved onto ConsolidationLoop (see paramem/training/consolidation.py).
# Re-exported as module-level aliases for existing call sites.
_dedup_episodic = ConsolidationLoop.dedup_episodic
_dedup_procedural = ConsolidationLoop.dedup_procedural


# --- Persistence ---


def _atomic_json_write(data: dict | list, path: Path, *, encrypted: bool = True) -> None:
    """Write JSON atomically.

    ``encrypted=True`` (default) routes through ``write_infra_bytes`` so
    every infrastructure JSON artifact respects the Security-ON/OFF
    contract through a single chokepoint.  ``encrypted=False`` bypasses
    the envelope and always writes plaintext; used by debug-directory
    writers so ``debug/*`` output is uniformly inspectable with ``cat``
    regardless of the server's Security posture.
    """
    from paramem.backup.encryption import write_plaintext_atomic

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2).encode("utf-8")
    if encrypted:
        write_infra_bytes(path, payload)
    else:
        write_plaintext_atomic(path, payload)


def _load_key_metadata(path: Path) -> dict | None:
    """Load key metadata from disk. Returns None if not found."""
    if not path.exists():
        logger.info("No key metadata found at %s, starting fresh", path)
        return None
    return json.loads(read_maybe_encrypted(path).decode("utf-8"))


def prune_key_metadata_orphans(config: ServerConfig) -> int:
    """Drop entries from ``key_metadata.json`` whose tier registry is gone.

    Per the wipe invariant (2026-05-14): ``key_metadata.json`` is bookkeeping
    for active keys, not a recovery source.  Called from the boot lifespan
    after :func:`_mount_adapters_from_slots` so the on-disk file never carries
    stale entries between a wipe and the next consolidation cycle.

    Reads every tier's ``indexed_key_registry.json`` (main + interim slots),
    takes the union of active keys, and rewrites ``key_metadata.json`` keeping
    only keys in that union.  ``promoted_keys`` is filtered to the same set.

    Returns the number of orphan keys removed (0 when nothing to prune).
    """
    from paramem.training.key_registry import KeyRegistry

    path = config.key_metadata_path
    if not path.exists():
        return 0

    try:
        raw = json.loads(read_maybe_encrypted(path).decode("utf-8"))
    except Exception:
        logger.exception("prune_key_metadata_orphans: could not read %s — skipping", path)
        return 0

    from paramem.memory.interim_adapter import iter_interim_dirs

    keys_in = raw.get("keys", {}) if isinstance(raw, dict) else {}
    if not keys_in:
        return 0

    # Data-loss guard (2026-06-02): prune is destructive, so it must only run when
    # the on-disk registries are AUTHORITATIVE for the active-key set.  Two
    # transient states violate that and must NOT be treated as permanent
    # orphanhood — they are exactly how an unfolded interim adapter's keys were
    # silently and permanently deleted:
    #
    #   1. An active-store migration / base-swap is in flight.  The live store is
    #      deliberately empty or mid-relocation (see _load_model_into_state's
    #      preload base-swap gate), and an interim slot may have been converted to
    #      a registry-less intermediate.  The on-disk registries do not yet
    #      describe the post-migration key set, so an absent key proves nothing.
    #
    #   2. The registry union is EMPTY while key_metadata still references keys.
    #      An empty union is indistinguishable from a transient unmounted state
    #      (e.g. an interim slot deleted earlier this session, main tier not yet
    #      refilled).  Pruning here would delete keys that were never folded into a
    #      persisted main-tier adapter.  An empty union can never PROVE orphanhood;
    #      a genuine wipe leaves the surviving tier registries non-empty.
    from paramem.server.active_store_migration import load_state as _load_migration_state

    if _load_migration_state(config.adapter_dir) is not None:
        logger.info(
            "prune_key_metadata_orphans: active-store migration in flight — "
            "skipping prune so transiently-relocated keys are not deleted (%s)",
            path,
        )
        return 0

    # The retention union includes BOTH active and stale keys so that a
    # soft-staled key's bookkeeping survives the prune.  A stale key is absent
    # from list_active() but present in list_stale(); pruning it would break the
    # stale-echo seam (bookkeeping is required to resolve speaker/relation_type).
    active: set[str] = set()
    for tier in ("episodic", "semantic", "procedural"):
        reg_path = config.adapter_dir / tier / "indexed_key_registry.json"
        if reg_path.exists():
            _loaded_reg = KeyRegistry.load(reg_path)
            active.update(_loaded_reg.list_known())
    for _name, interim_dir in iter_interim_dirs(config.adapter_dir):
        reg_path = interim_dir / "indexed_key_registry.json"
        if reg_path.exists():
            _loaded_reg = KeyRegistry.load(reg_path)
            active.update(_loaded_reg.list_known())

    if not active:
        # Empty registry union with non-empty key_metadata: cannot prove the keys
        # are permanently orphaned — refuse to prune (transient-absence guard).
        logger.warning(
            "prune_key_metadata_orphans: every tier registry is empty/absent but "
            "key_metadata.json still carries %d key(s) — refusing to prune "
            "(transient unmounted/migration state, not proof of orphanhood): %s",
            len(keys_in),
            path,
        )
        return 0

    pruned_keys = {k: v for k, v in keys_in.items() if k in active}
    pruned_promoted = sorted(k for k in raw.get("promoted_keys", []) if k in active)
    removed = len(keys_in) - len(pruned_keys)
    if removed == 0:
        return 0

    raw["keys"] = pruned_keys
    raw["promoted_keys"] = pruned_promoted
    _atomic_json_write(raw, path)
    logger.info(
        "prune_key_metadata_orphans: removed %d orphan key(s) from %s",
        removed,
        path,
    )
    return removed


def _save_key_metadata(loop: ConsolidationLoop, config: ServerConfig) -> None:
    """Save key metadata for cross-restart persistence.

    Per the wipe invariant (2026-05-14): ``key_metadata.json`` is
    bookkeeping for active keys, not a recovery source.  Saved entries
    reflect only :meth:`MemoryStore.all_active_keys` — keys absent from
    every tier registry are orphans and never persisted.

    When ``loop.trial_key_metadata_path`` is set (trial consolidation path),
    writes to that isolated path instead of the live ``config.key_metadata_path``.
    This ensures trial runs never touch the live ``data/ha/registry/key_metadata.json``
    (CRITICAL Fix 1 — trial registry isolation, 2026-04-23).
    """
    keys_payload: dict = {}
    # Persist bookkeeping for BOTH active and stale keys so that stale-echo
    # probes can resolve speaker/relation_type for a soft-staled key.
    # A key with no bookkeeping record is skipped rather than given a
    # fabricated record — it stays recordless on reload, which every
    # bookkeeping read site already tolerates via ``bookkeeping_for_key(k)
    # or {}`` / ``.get(...)``.
    all_keys = loop.store.all_known_keys()
    for key in all_keys:
        bk = loop.store.bookkeeping_for_key(key)
        if bk is None:
            continue
        keys_payload[key] = dict(bk)
    metadata = {
        "cycle_count": loop.cycle_count,
        "promoted_keys": sorted(loop.promoted_keys),
        "keys": keys_payload,
    }
    # CRITICAL Fix 1: honor loop-level override set by _build_trial_loop so the
    # trial never writes to the live registry paths.
    dest = getattr(loop, "trial_key_metadata_path", None) or config.key_metadata_path
    _atomic_json_write(metadata, dest)
