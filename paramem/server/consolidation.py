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
        # Seed key metadata from disk (survives restarts)
        metadata = _load_key_metadata(config.key_metadata_path)
        if metadata:
            loop.seed_key_metadata(metadata)

        # Seed indexed_key_qa from disk-persisted keyed_pairs.json files.  This
        # populates the loop's in-memory QA store across restarts so
        # consolidate_interim_adapters has the question/answer text it needs to
        # re-derive each active key's tier from the cumulative graph.  Without
        # the seed, a full cycle that runs before any in-process training has
        # populated indexed_key_qa would log "no QA metadata for key X" for
        # every active key and skip every per-tier rebuild — observably the
        # gate fires forever because the rebuild never advances main slots'
        # window_stamp.  Transparently decrypts age-wrapped content when the
        # daily identity is loaded.
        #
        # Seed reads from the canonical store for the active mode (locked
        # decision #2). Train mode reads from paths.adapters (where _save_adapters
        # wrote canonical bytes); simulate mode reads from paths.simulate (where
        # _save_keyed_pairs_for_router writes canonical bytes).
        store_dir = (
            config.simulate_dir if config.consolidation.mode == "simulate" else config.adapter_dir
        )

        # In quad mode use read_keyed_pairs_quad (has legacy-QA→quad back-compat
        # projection built in).  In QA mode use the standard read_keyed_pairs.
        if config.consolidation.indexed_format == "quad":
            from paramem.training.keyed_pairs_io import read_keyed_pairs_quad as _read_kp
        else:
            from paramem.training.keyed_pairs_io import read_keyed_pairs as _read_kp

        ep_kp_path = store_dir / "episodic" / "keyed_pairs.json"
        if ep_kp_path.exists():
            try:
                loop.seed_episodic_qa(_read_kp(ep_kp_path))
            except Exception:
                logger.exception(
                    "Failed to seed indexed_key_qa from episodic store %s — skipping",
                    ep_kp_path,
                )

        sem_kp_path = store_dir / "semantic" / "keyed_pairs.json"
        if sem_kp_path.exists():
            try:
                loop.seed_semantic_qa(_read_kp(sem_kp_path))
            except Exception:
                logger.exception(
                    "Failed to seed indexed_key_qa from semantic store %s — skipping",
                    sem_kp_path,
                )

        proc_kp_path = store_dir / "procedural" / "keyed_pairs.json"
        if proc_kp_path.exists():
            try:
                loop.seed_procedural_qa(_read_kp(proc_kp_path))
            except Exception:
                logger.exception(
                    "Failed to seed indexed_key_qa from procedural store %s — skipping",
                    proc_kp_path,
                )

        # Interim slots are training-only (locked decision #3) — always under
        # paths.adapters regardless of mode. The simulate store has no interim
        # concept by design.
        for interim_dir in sorted(config.adapter_dir.glob("episodic_interim_*")):
            if not interim_dir.is_dir():
                continue
            kp = interim_dir / "keyed_pairs.json"
            if not kp.exists():
                continue
            try:
                loop.seed_episodic_qa(_read_kp(kp))
            except Exception:
                logger.exception(
                    "Failed to seed indexed_key_qa from interim slot %s — skipping",
                    interim_dir.name,
                )

    # One-time startup WARN when the non-active store has stale keyed_pairs
    # from a previous mode. The startup path runs once per process so this
    # does not spam logs. Runs unconditionally — operator telemetry for
    # detecting mode-switch drift, not a seeding side-effect.
    inactive_dir = (
        config.adapter_dir if config.consolidation.mode == "simulate" else config.simulate_dir
    )
    if inactive_dir.exists():
        stale = list(inactive_dir.rglob("keyed_pairs.json"))
        if stale:
            logger.warning(
                "Found %d stale keyed_pairs.json under %s (active mode is %r). "
                "Inference reads only the active store; remove the inactive "
                "store or re-run consolidation in the matching mode to clear.",
                len(stale),
                inactive_dir,
                config.consolidation.mode,
            )

    return loop


def _do_mark_consolidated(session_buffer, session_ids, callback):
    """Invoke the mark-consolidated callback or fall back to the buffer method.

    When *callback* is ``None``, ``session_buffer.mark_consolidated(session_ids)``
    is called (standard production path).  When *callback* is not ``None`` it is
    called instead, allowing the trial path to pass a no-op so pending sessions
    remain in the buffer (spec L364 — "transcript sweeper blocks archive+delete").

    Parameters
    ----------
    session_buffer:
        The live session buffer.
    session_ids:
        List of session IDs that finished extraction/training.
    callback:
        Optional callable accepting ``session_ids``.  ``None`` → use buffer method.
    """
    if callback is None:
        session_buffer.mark_consolidated(session_ids)
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

    A QA pair's ``source_subject`` / ``source_object`` carry the entity's
    display name (e.g. ``"Alex"``).  Speaker entities are keyed in the
    cumulative graph by ``speaker_id`` (e.g. ``"Speaker0"``) with the
    display name stored at ``attributes["name"]``.  The matching set
    therefore includes both the node ID AND the ``attributes["name"]``
    value (when set) so display-name-driven QA pairs continue to match.
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
    for key, qa in loop.indexed_key_qa.items():
        if key in loop.promoted_keys:
            continue
        subject = qa.get("source_subject", "").lower()
        obj = qa.get("source_object", "").lower()
        if subject in session_entities or obj in session_entities:
            loop.key_sessions[key] = loop.key_sessions.get(key, 0) + 1


# Dedup moved onto ConsolidationLoop (see paramem/training/consolidation.py).
# Re-exported as module-level aliases for existing call sites.
_dedup_episodic = ConsolidationLoop.dedup_episodic
_dedup_procedural = ConsolidationLoop.dedup_procedural


def _save_debug_artifacts(
    loop: ConsolidationLoop,
    config: ServerConfig,
    episodic_qa: list[dict],
    procedural_rels: list[dict],
) -> None:
    """Write plaintext debug artifacts to ``config.debug_dir/cycle_<N>/``.

    Called in BOTH simulate and train branches when ``config.debug`` is true,
    so per-cycle debug dumps are symmetric regardless of mode.  All filenames
    carry the ``_snapshot`` postfix (locked decision #7) so every file under
    ``config.debug_dir`` is trivially distinguishable from production output
    by name alone.

    Always plaintext (``encrypted=False``), regardless of the server's Security
    posture — debug output is inspection-first; operators must be able to read
    it with ``cat`` / ``grep`` without any decrypt step.

    Parameters
    ----------
    loop:
        Active ``ConsolidationLoop`` (provides ``merger`` and ``cycle_count``).
    config:
        Server configuration (provides ``debug_dir``).
    episodic_qa:
        Episodic QA pairs produced by the consolidation pipeline.
    procedural_rels:
        Procedural relation triples produced by the consolidation pipeline.
    """
    out_dir = config.debug_dir / f"cycle_{loop.cycle_count}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plaintext — debug output is inspection-first, regardless of Security posture.
    loop.merger.save_graph(out_dir / "graph_snapshot.json", encrypted=False)
    _atomic_json_write(episodic_qa, out_dir / "episodic_qa_snapshot.json", encrypted=False)
    if procedural_rels:
        _atomic_json_write(
            procedural_rels, out_dir / "procedural_rels_snapshot.json", encrypted=False
        )

    logger.info(
        "Debug artifacts written to %s: %d episodic QA, %d procedural rels",
        out_dir,
        len(episodic_qa),
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
            if key in loop.episodic_simhash:
                loop.semantic_simhash[key] = loop.episodic_simhash.pop(key)
                newly_promoted.append(key)
            elif key in loop.semantic_simhash:
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


def _save_key_metadata(loop: ConsolidationLoop, config: ServerConfig) -> None:
    """Save key metadata for cross-restart persistence.

    When ``loop.trial_key_metadata_path`` is set (trial consolidation path),
    writes to that isolated path instead of the live ``config.key_metadata_path``.
    This ensures trial runs never touch the live ``data/ha/registry/key_metadata.json``
    (CRITICAL Fix 1 — trial registry isolation, 2026-04-23).
    """
    metadata = {
        "cycle_count": loop.cycle_count,
        "promoted_keys": sorted(loop.promoted_keys),
        "keys": {key: {"sessions_seen": count} for key, count in loop.key_sessions.items()},
    }
    # CRITICAL Fix 1: honor loop-level override set by _build_trial_loop so the
    # trial never writes to the live registry paths.
    dest = getattr(loop, "trial_key_metadata_path", None) or config.key_metadata_path
    _atomic_json_write(metadata, dest)


def _save_keyed_pairs_for_router(loop: ConsolidationLoop, config: ServerConfig) -> None:
    """Save keyed_pairs.json per adapter for router entity indexing.

    Mode-aware: in simulate mode the canonical store is ``paths.simulate``;
    in train mode the canonical store is ``paths.adapters``. Layout is
    identical in both stores: ``<store_dir>/<tier>/keyed_pairs.json`` for
    all three tiers.

    Note: in train mode this helper is no longer the primary writer —
    ``ConsolidationLoop._save_adapters`` writes canonical keyed_pairs
    bytes-identically as part of the I5 reorder before the manifest is
    built (paramem/training/consolidation.py:2154-2158). This helper
    runs only on the simulate path today (see run_consolidation L386-393).
    Post-canonicalization: this is the **only** simulate-mode persistence —
    the cycle_<N>/ snapshot writer (``_save_simulate_store``) is retired
    because it produced no load-bearing artifacts.
    """
    store_dir = (
        config.simulate_dir if config.consolidation.mode == "simulate" else config.adapter_dir
    )
    store_dir.mkdir(parents=True, exist_ok=True)

    _fmt = config.consolidation.indexed_format

    # Episodic keyed_pairs in the canonical episodic subdirectory.
    if loop.episodic_simhash:
        ep_dir = store_dir / "episodic"
        ep_dir.mkdir(parents=True, exist_ok=True)
        _write_keyed_pairs(
            loop.indexed_key_qa,
            loop.episodic_simhash,
            ep_dir / "keyed_pairs.json",
            indexed_format=_fmt,
        )

    if loop.semantic_simhash:
        sem_dir = store_dir / "semantic"
        sem_dir.mkdir(parents=True, exist_ok=True)
        _write_keyed_pairs(
            loop.indexed_key_qa,
            loop.semantic_simhash,
            sem_dir / "keyed_pairs.json",
            indexed_format=_fmt,
        )

    if loop.procedural_simhash:
        proc_dir = store_dir / "procedural"
        proc_dir.mkdir(parents=True, exist_ok=True)
        _write_keyed_pairs(
            loop.indexed_key_qa,
            loop.procedural_simhash,
            proc_dir / "keyed_pairs.json",
            indexed_format=_fmt,
        )


def _write_keyed_pairs(
    indexed_key_qa: dict,
    simhash_registry: dict,
    path: Path,
    *,
    indexed_format: str = "qa",
) -> None:
    """Write keyed_pairs.json for keys in the given SimHash registry.

    Delegates to :func:`paramem.training.keyed_pairs_io.write_keyed_pairs`
    (QA mode) or :func:`paramem.training.keyed_pairs_io.write_keyed_pairs_quad`
    (quad mode) so the canonical schema is enforced by construction.  Every
    entry in *indexed_key_qa* must carry the fields expected by the chosen
    writer before this function is called — a missing field raises
    ``KeyError`` at write time.

    Parameters
    ----------
    indexed_format:
        ``"quad"`` routes to ``write_keyed_pairs_quad``; any other value (incl.
        the default ``"qa"``) routes to the standard ``write_keyed_pairs``.
    """
    if indexed_format == "quad":
        from paramem.training.keyed_pairs_io import write_keyed_pairs_quad as _wkp
    else:
        from paramem.training.keyed_pairs_io import write_keyed_pairs as _wkp

    pairs = [indexed_key_qa[k] for k in simhash_registry if k in indexed_key_qa]
    _wkp(path, pairs)
