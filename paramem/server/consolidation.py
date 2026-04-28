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
import time
from pathlib import Path

from paramem.backup.encryption import read_maybe_encrypted, write_infra_bytes
from paramem.server.config import ServerConfig
from paramem.server.session_buffer import SessionBuffer
from paramem.server.vram_guard import vram_scope
from paramem.training.consolidation import ConsolidationLoop

logger = logging.getLogger(__name__)


def create_consolidation_loop(
    model,
    tokenizer,
    config: ServerConfig,
    state_provider=None,
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
    """
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
        output_dir=config.adapter_dir,
        graph_path=None,
        extraction_temperature=0.0,
        extraction_max_tokens=config.consolidation.extraction_max_tokens,
        save_cycle_snapshots=config.debug,
        snapshot_dir=config.debug_dir if config.debug else None,
        persist_graph=False,
        prompts_dir=config.prompts_dir,
        graph_config=config.graph_config,
        graph_enrichment_enabled=config.consolidation.graph_enrichment_enabled,
        graph_enrichment_neighborhood_hops=config.consolidation.graph_enrichment_neighborhood_hops,
        graph_enrichment_max_entities_per_pass=config.consolidation.graph_enrichment_max_entities_per_pass,
        graph_enrichment_interim_enabled=config.consolidation.graph_enrichment_interim_enabled,
        graph_enrichment_min_triples_floor=config.consolidation.graph_enrichment_min_triples_floor,
        state_provider=state_provider,
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

    ep_kp_path = store_dir / "episodic" / "keyed_pairs.json"
    if ep_kp_path.exists():
        loop.seed_episodic_qa(json.loads(read_maybe_encrypted(ep_kp_path).decode("utf-8")))

    sem_kp_path = store_dir / "semantic" / "keyed_pairs.json"
    if sem_kp_path.exists():
        loop.seed_semantic_qa(json.loads(read_maybe_encrypted(sem_kp_path).decode("utf-8")))

    proc_kp_path = store_dir / "procedural" / "keyed_pairs.json"
    if proc_kp_path.exists():
        loop.seed_procedural_qa(json.loads(read_maybe_encrypted(proc_kp_path).decode("utf-8")))

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
            loop.seed_episodic_qa(json.loads(read_maybe_encrypted(kp).decode("utf-8")))
        except Exception:
            logger.exception(
                "Failed to seed indexed_key_qa from interim slot %s — skipping",
                interim_dir.name,
            )

    # One-time startup WARN when the non-active store has stale keyed_pairs
    # from a previous mode. The startup path runs once per process so this
    # does not spam logs.
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


def run_consolidation(
    model,
    tokenizer,
    config: ServerConfig,
    session_buffer: SessionBuffer,
    loop: ConsolidationLoop | None = None,
    ha_context: dict | None = None,
    speaker_store=None,
    mark_consolidated_callback=None,
) -> dict:
    """Run consolidation on all pending sessions.

    Batch mode: extract all sessions first, then train once.
    - Phase 1: Extract graphs and generate QA for each session
    - Phase 2: Train all adapters once on the accumulated QA
    - Phase 3: Save artifacts and mark sessions consolidated

    Parameters
    ----------
    mark_consolidated_callback:
        Optional callable invoked with ``session_ids`` after a successful
        train cycle.  When ``None``, ``session_buffer.mark_consolidated`` is
        called directly (production behaviour).  Pass ``None`` explicitly for
        the trial consolidation path so pending sessions remain in the buffer
        and ``/migration/rollback`` can restore the full queue (spec L364).

    Returns a result dict including the loop instance for reuse.
    """
    if not config.adapters.episodic.enabled:
        logger.info("Episodic adapter is disabled in config, skipping consolidation")
        return {"status": "disabled", "sessions": 0, "loop": loop}

    start_time = time.time()

    pending = session_buffer.get_pending()
    if not pending:
        logger.info("No pending sessions to consolidate")
        return {"status": "no_pending", "sessions": 0, "loop": loop}

    logger.info("Consolidating %d pending sessions", len(pending))

    # Create loop on first use — persists across consolidation runs
    if loop is None:
        loop = create_consolidation_loop(model, tokenizer, config)

    # --- Phase 1: Extract all sessions ---
    all_episodic_qa = []
    all_procedural_rels = []
    session_ids = []
    speaker_ids = []  # track per-session speakers for key tagging
    total_relations = 0

    for session in pending:
        session_id = session["session_id"]
        transcript = session["transcript"]
        session_speaker_id = session.get("speaker_id")

        session_ids.append(session_id)

        # Skip truly-None sessions (no speaker ID assigned, i.e. text-only sessions
        # with no voice channel). Named speakers and anonymous Speaker{N} IDs both
        # flow through extraction. A None speaker_id would key procedural_sp_index
        # on (None, subject, predicate), causing unrelated text-only sessions to
        # collide and cross-retire each other's procedural keys.
        if not session_speaker_id:
            logger.debug(
                "Session %s has no speaker_id — skipping (text-only, no voice attribution)",
                session_id,
            )
            continue

        speaker_name = None
        if speaker_store is not None:
            try:
                speaker_name = speaker_store.get_name(session_speaker_id)
            except Exception as e:
                logger.warning("speaker_store.get_name(%s) failed: %s", session_speaker_id, e)
        with vram_scope(session_id):
            episodic_qa, procedural_rels = loop.extract_session(
                transcript,
                session_id,
                speaker_id=session_speaker_id,
                speaker_name=speaker_name,
                ha_context=ha_context,
                stt_correction=config.consolidation.extraction_stt_correction,
                ha_validation=config.consolidation.extraction_ha_validation,
                noise_filter=config.consolidation.extraction_noise_filter,
                noise_filter_model=config.consolidation.extraction_noise_filter_model,
                noise_filter_endpoint=config.consolidation.extraction_noise_filter_endpoint or None,
                ner_check=config.consolidation.extraction_ner_check,
                ner_model=config.consolidation.extraction_ner_model,
                plausibility_judge=config.consolidation.extraction_plausibility_judge,
                plausibility_stage=config.consolidation.extraction_plausibility_stage,
                verify_anonymization=config.consolidation.extraction_verify_anonymization,
                source_type=session.get("source_type", "transcript"),
            )

        # Increment key session counts while last_seen is still correct
        _increment_key_sessions(loop, session_id)

        # Tag QA pairs with speaker — assign_keys preserves speaker_id
        for qa in episodic_qa:
            qa["speaker_id"] = session_speaker_id
        for rel in procedural_rels:
            rel["speaker_id"] = session_speaker_id

        all_episodic_qa.extend(episodic_qa)
        all_procedural_rels.extend(procedural_rels)
        total_relations += len(episodic_qa) + len(procedural_rels)
        speaker_ids.append(session_speaker_id)

        logger.info(
            "Extracted session %s: %d episodic, %d procedural relations",
            session_id,
            len(episodic_qa),
            len(procedural_rels),
        )

        # Check shutdown flag between sessions
        if loop.shutdown_requested:
            logger.info("Shutdown requested — stopping extraction after %s", session_id)
            break

    if not all_episodic_qa and not all_procedural_rels:
        logger.info("No QA pairs extracted — skipping training")
        _do_mark_consolidated(session_buffer, session_ids, mark_consolidated_callback)
        return {
            "status": "no_facts",
            "sessions": len(session_ids),
            "loop": loop,
        }

    # --- Cross-session dedup on (subject, predicate, object) identity ---
    # Applied before the simulate/train branch so both paths see the same
    # post-dedup set. Duplicates arise when independent sessions extract the
    # same triple (e.g. "Alex listens_to Music" from two transcripts).
    pre_ep, pre_pr = len(all_episodic_qa), len(all_procedural_rels)
    all_episodic_qa = _dedup_episodic(all_episodic_qa)
    all_procedural_rels = _dedup_procedural(all_procedural_rels)
    if pre_ep != len(all_episodic_qa) or pre_pr != len(all_procedural_rels):
        logger.info(
            "Dedup: episodic %d→%d, procedural %d→%d",
            pre_ep,
            len(all_episodic_qa),
            pre_pr,
            len(all_procedural_rels),
        )

    # --- Simulate mode: peer storage backend ---
    # Same upstream pipeline as train (extraction → dedup → key assignment →
    # contradiction handling → SimHash registry); the persistence venue is
    # an encrypted JSON store under paths.simulate instead of LoRA weight
    # updates. mark_consolidated runs in both branches — sessions retire
    # when their work has been persisted, regardless of medium. Inference
    # reads from the JSON store at retrieval time (Task #7 follow-up).

    # Symmetric debug dump — runs in BOTH simulate and train branches when
    # debug is on. Plaintext, inspection-first; closes the asymmetry that
    # previously gated debug output on simulate mode only.
    #
    # Best-effort: a debug-write failure (disk full, permission, etc.) MUST
    # NOT abort consolidation. The call has its own log-and-continue envelope
    # so a debug failure does not regress train-branch error handling.
    if config.debug:
        try:
            _save_debug_artifacts(loop, config, all_episodic_qa, all_procedural_rels)
        except Exception:
            logger.exception("debug-artifact write failed; continuing consolidation")

    simulate = config.consolidation.mode == "simulate"
    if simulate:
        primary_speaker_sim = speaker_ids[-1] if speaker_ids else ""
        try:
            sim_result = loop.simulated_training(
                all_episodic_qa, all_procedural_rels, speaker_id=primary_speaker_sim
            )

            newly_promoted = _promote_mature_keys(loop, config)

            _save_keyed_pairs_for_router(loop, config)
            _save_key_metadata(loop, config)
            # _save_simulate_store retired with the canonicalization — the per-tier
            # keyed_pairs.json written by _save_keyed_pairs_for_router is the
            # only simulate-mode persistence (cycle_<N>/ snapshots dropped).
        except Exception:
            logger.exception(
                "Simulated consolidation failed — leaving %d sessions pending",
                len(session_ids),
            )
            raise

        # Simulate is peer storage — sessions retire here just like train.
        # The simulate JSON store IS the persistence venue; the merger is
        # still idempotent on (subject, predicate, object) so a re-run of
        # those sessions would be a no-op anyway.
        _do_mark_consolidated(session_buffer, session_ids, mark_consolidated_callback)
        elapsed = time.time() - start_time
        summary = {
            "status": "simulated",
            "sessions": len(session_ids),
            "total_relations": total_relations,
            "newly_promoted": len(newly_promoted),
            "episodic_qa": len(all_episodic_qa),
            "procedural_rels": len(all_procedural_rels),
            "episodic_keys": len(loop.episodic_simhash),
            "semantic_keys": len(loop.semantic_simhash),
            "procedural_keys": len(loop.procedural_simhash),
            "elapsed_seconds": round(elapsed, 1),
            "simulated": sim_result.get("simulated", True),
            "loop": loop,
        }
        logger.info("Simulation complete: %s", {k: v for k, v in summary.items() if k != "loop"})
        return summary

    # --- Phase 2: Train once ---
    logger.info(
        "Training on %d episodic + %d procedural QA pairs",
        len(all_episodic_qa),
        len(all_procedural_rels),
    )

    # Fallback speaker_id for procedural contradiction scoping.
    # Each QA pair already has its own speaker_id (tagged during extraction).
    # This fallback is only used when a relation lacks a speaker_id.
    primary_speaker = speaker_ids[-1] if speaker_ids else ""
    try:
        # Task #7 fix (train branch only): train without the internal _save_adapters call,
        # then promote mature keys so _save_adapters captures post-promotion SimHash
        # membership in the manifest's keyed_pairs_sha256.  The redundant
        # _save_keyed_pairs_for_router call below is removed because _save_adapters
        # already writes keyed_pairs.json per adapter as part of the I5 reorder.
        #
        # NOTE: the simulate branch (above, line ~242) still calls
        # _save_keyed_pairs_for_router because it uses simulated_training which does
        # NOT call _save_adapters — the simulate path's _save_keyed_pairs_for_router
        # is the only kp write in that branch and must be preserved.
        with vram_scope("training"):
            train_result = loop.train_adapters_no_save(
                all_episodic_qa, all_procedural_rels, speaker_id=primary_speaker
            )

        # Key-level promotion: promote keys that reached the threshold.
        # Must run BEFORE _save_adapters so the manifest reflects post-promotion
        # SimHash membership (Task #7 fix).
        newly_promoted = _promote_mature_keys(loop, config)

        # --- Phase 3: Save adapters (writes keyed_pairs.json per adapter internally) ---
        loop._save_adapters()
        _save_key_metadata(loop, config)
    except Exception:
        logger.exception(
            "Consolidation failed during train/save — leaving %d sessions pending",
            len(session_ids),
        )
        raise

    _do_mark_consolidated(session_buffer, session_ids, mark_consolidated_callback)

    elapsed = time.time() - start_time

    summary = {
        "status": "complete",
        "sessions": len(session_ids),
        "total_relations": total_relations,
        "newly_promoted": len(newly_promoted),
        "episodic_keys": len(loop.episodic_simhash),
        "semantic_keys": len(loop.semantic_simhash),
        "procedural_keys": len(loop.procedural_simhash),
        "train_loss": train_result.get("episodic_train_loss"),
        "elapsed_seconds": round(elapsed, 1),
        "loop": loop,
    }
    logger.info(
        "Consolidation complete: %s",
        {k: v for k, v in summary.items() if k != "loop"},
    )
    return summary


# --- Key-level promotion ---


def _increment_key_sessions(loop: ConsolidationLoop, session_id: str) -> None:
    """Increment sessions_seen for keys whose entities appeared in this session.

    Uses the graph merger's node metadata: nodes where last_seen == session_id
    were active in this specific session (not the cumulative graph).
    """
    # Find entities that appeared in this session
    session_entities = set()
    for node in loop.merger.graph.nodes:
        node_data = loop.merger.graph.nodes[node]
        if node_data.get("last_seen") == session_id:
            session_entities.add(node.lower())

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

    # Episodic keyed_pairs in the canonical episodic subdirectory.
    if loop.episodic_simhash:
        ep_dir = store_dir / "episodic"
        ep_dir.mkdir(parents=True, exist_ok=True)
        _write_keyed_pairs(
            loop.indexed_key_qa,
            loop.episodic_simhash,
            ep_dir / "keyed_pairs.json",
        )

    if loop.semantic_simhash:
        sem_dir = store_dir / "semantic"
        sem_dir.mkdir(parents=True, exist_ok=True)
        _write_keyed_pairs(
            loop.indexed_key_qa,
            loop.semantic_simhash,
            sem_dir / "keyed_pairs.json",
        )

    if loop.procedural_simhash:
        proc_dir = store_dir / "procedural"
        proc_dir.mkdir(parents=True, exist_ok=True)
        _write_keyed_pairs(
            loop.indexed_key_qa,
            loop.procedural_simhash,
            proc_dir / "keyed_pairs.json",
        )


def _write_keyed_pairs(
    indexed_key_qa: dict,
    simhash_registry: dict,
    path: Path,
) -> None:
    """Write keyed_pairs.json for keys in the given SimHash registry."""
    pairs = []
    for key in simhash_registry:
        if key in indexed_key_qa:
            qa = indexed_key_qa[key]
            entry = {
                "key": key,
                "question": qa["question"],
                "answer": qa["answer"],
            }
            for meta_key in ("source_subject", "source_object", "source_predicate", "speaker_id"):
                if meta_key in qa:
                    entry[meta_key] = qa[meta_key]
            pairs.append(entry)

    _atomic_json_write(pairs, path)
