"""Server consolidation — thin wrapper around ConsolidationLoop.

Uses the same ConsolidationLoop that powers Tests 1-8, with
indexed_key_replay_enabled=True. The graph is transient (RAM-only).
Promotion is key-level: per-key session counts persisted in
key_metadata.json (no personal data on disk).

The loop saves adapters directly to output_dir (= adapter_dir), so
the router can reload from the standard paths without any bridging.
"""

import json
import logging
from pathlib import Path

from paramem.backup.encryption import read_maybe_encrypted, write_infra_bytes
from paramem.server.config import ServerConfig
from paramem.training.consolidation import ConsolidationLoop
from paramem.training.thermal_throttle import ThermalPolicy

logger = logging.getLogger(__name__)


def create_consolidation_loop(
    model,
    tokenizer,
    config: ServerConfig,
    memory_store,
    state_provider=None,
    *,
    output_dir=None,
    save_cycle_snapshots: bool | None = None,
    persist_graph: bool | None = None,
    seed_state_from_disk: bool = True,
) -> ConsolidationLoop:
    """Create a ConsolidationLoop configured for the server.

    Graph is transient (persist_graph=False). Key metadata is seeded
    from key_metadata.json to restore cycle count, promoted keys, and
    per-key session counts across server restarts.

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
    persist_graph:
        Override the hardcoded ``False`` used for production (transient
        graph, RAM-only).  ``None`` (default) resolves to ``False``.
    seed_state_from_disk:
        When ``False``, skip seeding key metadata and keyed-pairs QA from
        disk.  Use for probe/experiment runs that must start from a clean
        state rather than inheriting live-system data.  Default ``True``
        preserves production behaviour.
    """
    _output_dir = output_dir if output_dir is not None else config.adapter_dir
    _save_cycle_snapshots = (
        save_cycle_snapshots if save_cycle_snapshots is not None else config.debug
    )
    _persist_graph = persist_graph if persist_graph is not None else False

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
        graph_path=None,
        extraction_temperature=0.0,
        extraction_max_tokens=config.consolidation.extraction_max_tokens,
        extraction_plausibility_max_tokens=config.consolidation.extraction_plausibility_max_tokens,
        save_cycle_snapshots=_save_cycle_snapshots,
        snapshot_dir=config.debug_dir if _save_cycle_snapshots else None,
        persist_graph=_persist_graph,
        prompts_dir=config.prompts_dir,
        graph_config=config.graph_config,
        graph_enrichment_enabled=config.consolidation.graph_enrichment_enabled,
        graph_enrichment_neighborhood_hops=config.consolidation.graph_enrichment_neighborhood_hops,
        graph_enrichment_max_entities_per_pass=config.consolidation.graph_enrichment_max_entities_per_pass,
        graph_enrichment_interim_enabled=config.consolidation.graph_enrichment_interim_enabled,
        graph_enrichment_min_triples_floor=config.consolidation.graph_enrichment_min_triples_floor,
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


def _do_mark_consolidated(
    session_buffer,
    session_ids,
    callback,
    *,
    retention_dir: Path | None = None,
):
    """Invoke the mark-consolidated callback or fall back to the buffer method.

    When *callback* is ``None``, ``session_buffer.mark_consolidated(session_ids,
    retention_dir=retention_dir)`` is called (standard production path).  When
    *callback* is not ``None`` it is called instead, allowing the trial path to
    pass a no-op so pending sessions remain in the buffer (spec L364 —
    "transcript sweeper blocks archive+delete").

    Parameters
    ----------
    session_buffer:
        The live session buffer.
    session_ids:
        List of session IDs that finished extraction/training.
    callback:
        Optional callable accepting ``session_ids``.  ``None`` → use buffer method.
    retention_dir:
        Directory to move JSONL into when ``retain_sessions`` or ``debug`` is
        true on the buffer.  Forwarded to :meth:`SessionBuffer.mark_consolidated`.
        Ignored when *callback* is provided.
    """
    if callback is None:
        session_buffer.mark_consolidated(session_ids, retention_dir=retention_dir)
    else:
        callback(session_ids)


# run_consolidation was deleted here (D2 — voice-profile-switch commit).
# Its logic now lives in paramem.server.app._run_extraction_phase, which closes
# over _state instead of taking model/tokenizer/config/session_buffer as args.
# Trial-migration callers: use _run_extraction_phase(loop, mark_callback=lambda _: None).
# Dev scripts that imported run_consolidation directly will need updating separately.


# --- Key-level promotion ---


def _increment_key_sessions(loop: ConsolidationLoop, session_id: str) -> None:
    """Increment sessions_seen for keys whose entities appeared in this session.

    Uses the graph merger's node metadata: nodes where last_seen == session_id
    were active in this specific session (not the cumulative graph).

    A relation's ``subject`` / ``object`` carry the entity's display name
    (e.g. ``"Alex"``).  Speaker entities are keyed in the cumulative graph
    by ``speaker_id`` (e.g. ``"Speaker0"``) with the display name stored at
    ``attributes["name"]``.  The matching set therefore includes both the
    node ID AND the ``attributes["name"]`` value (when set) so
    display-name-driven relations continue to match.
    """
    # Find entities that appeared in this session
    session_entities = set()
    for node in loop.merger.graph.nodes:
        node_data = loop.merger.graph.nodes[node]
        if node_data.get("last_seen") == session_id:
            session_entities.add(node.lower())
            display_name = (node_data.get("attributes") or {}).get("name")
            if isinstance(display_name, str) and display_name:
                session_entities.add(display_name.lower())

    if not session_entities:
        return

    # Increment session count for keys referencing these entities
    for _tier, key, qa in loop.store.iter_entries():
        if key in loop.promoted_keys:
            continue
        subject = qa.get("subject", "").lower()
        obj = qa.get("object", "").lower()
        if subject in session_entities or obj in session_entities:
            loop.key_sessions[key] = loop.key_sessions.get(key, 0) + 1


# Dedup moved onto ConsolidationLoop (see paramem/training/consolidation.py).
# Re-exported as module-level aliases for existing call sites.
_dedup_episodic = ConsolidationLoop.dedup_episodic
_dedup_procedural = ConsolidationLoop.dedup_procedural


def _save_debug_artifacts(
    loop: ConsolidationLoop,
    config: ServerConfig,
    episodic_rels: list[dict],
    procedural_rels: list[dict],
) -> None:
    """Write plaintext debug artifacts under ``loop.snapshot_dir/cycle_<N>/``.

    Called in BOTH simulate and train branches when ``config.debug`` is true,
    so per-cycle debug dumps are symmetric regardless of mode.  All filenames
    carry the ``_snapshot`` postfix (locked decision #7) so every file under
    ``paths.debug`` is trivially distinguishable from production output by
    name alone.

    The output path uses the same ``run_<id>/cycle_<N>/`` hierarchy that
    :class:`ConsolidationLoop` already builds for its own per-cycle artifacts
    (graph snapshot, adapter checkpoint shadows).  ONE save routine — no
    duplicate ``debug_dir/cycle_<N>/`` legacy path (2026-05-14 collapse).

    Always plaintext (``encrypted=False``), regardless of the server's Security
    posture — debug output is inspection-first; operators must be able to read
    it with ``cat`` / ``grep`` without any decrypt step.

    Parameters
    ----------
    loop:
        Active ``ConsolidationLoop`` (provides ``merger``, ``cycle_count``,
        and ``snapshot_dir``).
    config:
        Server configuration (fallback ``debug_dir`` when no snapshot_dir
        is configured on the loop).
    episodic_rels:
        Episodic relations produced by the consolidation pipeline.
    procedural_rels:
        Procedural relation triples produced by the consolidation pipeline.
    """
    # _current_interim_stamp is never set on the loop (the attribute was removed
    # when the context-variable pattern was replaced by explicit parameter passing
    # in run_consolidation_cycle).  Pass None so snapshot_dir_for uses the
    # cycle-scoped path (paths.debug/episodic/cycle_<N>/run_<run_id>/).
    snap = None
    snap_fn = getattr(loop, "snapshot_dir_for", None)
    if callable(snap_fn):
        snap = snap_fn(interim_stamp=None)
    out_dir = snap if snap is not None else config.debug_dir / f"cycle_{loop.cycle_count}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plaintext — debug output is inspection-first, regardless of Security posture.
    loop.merger.save_graph(out_dir / "graph_snapshot.json", encrypted=False)
    _atomic_json_write(episodic_rels, out_dir / "episodic_rels_snapshot.json", encrypted=False)
    if procedural_rels:
        _atomic_json_write(
            procedural_rels, out_dir / "procedural_rels_snapshot.json", encrypted=False
        )

    logger.info(
        "Debug artifacts written to %s: %d episodic, %d procedural relations",
        out_dir,
        len(episodic_rels),
        len(procedural_rels),
    )


def _promote_mature_keys(loop: ConsolidationLoop, config: ServerConfig) -> list[str]:
    """Promote keys that reached the session count threshold.

    Moves keys from episodic to semantic SimHash registry.
    Returns list of newly promoted key IDs.
    """
    threshold = config.consolidation.promotion_threshold
    newly_promoted = []

    for key, count in loop.key_sessions.items():
        if count >= threshold and key not in loop.promoted_keys:
            if loop.store.has_simhash("episodic", key):
                # Move entry + simhash + registry atomically to semantic.
                loop.store.move(key, "semantic")
                newly_promoted.append(key)
            elif loop.store.has_simhash("semantic", key):
                logger.debug("Key %s already in semantic, marking promoted", key)
            loop.promoted_keys.add(key)

    if newly_promoted:
        logger.info("Promoted %d keys to semantic", len(newly_promoted))

    return newly_promoted


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

    from paramem.server.interim_adapter import iter_interim_dirs

    active: set[str] = set()
    for tier in ("episodic", "semantic", "procedural"):
        reg_path = config.adapter_dir / tier / "indexed_key_registry.json"
        if reg_path.exists():
            active.update(KeyRegistry.load(reg_path).list_active())
    for _name, interim_dir in iter_interim_dirs(config.adapter_dir):
        reg_path = interim_dir / "indexed_key_registry.json"
        if reg_path.exists():
            active.update(KeyRegistry.load(reg_path).list_active())

    keys_in = raw.get("keys", {}) if isinstance(raw, dict) else {}
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
    for key in loop.store.all_active_keys():
        sessions_seen = loop.key_sessions.get(key, 0)
        cache_entry = loop.store.get(key) or {}
        keys_payload[key] = {
            "sessions_seen": sessions_seen,
            "speaker_id": cache_entry.get("speaker_id", ""),
            "first_seen_cycle": cache_entry.get("first_seen_cycle", 0),
        }
    metadata = {
        "cycle_count": loop.cycle_count,
        "promoted_keys": sorted(loop.promoted_keys),
        "keys": keys_payload,
    }
    # CRITICAL Fix 1: honor loop-level override set by _build_trial_loop so the
    # trial never writes to the live registry paths.
    dest = getattr(loop, "trial_key_metadata_path", None) or config.key_metadata_path
    _atomic_json_write(metadata, dest)
