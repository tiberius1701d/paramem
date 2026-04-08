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
import os
import time
from pathlib import Path

from paramem.server.config import ServerConfig
from paramem.server.session_buffer import SessionBuffer
from paramem.training.consolidation import ConsolidationLoop

logger = logging.getLogger(__name__)


def create_consolidation_loop(
    model,
    tokenizer,
    config: ServerConfig,
) -> ConsolidationLoop:
    """Create a ConsolidationLoop configured for the server.

    Graph is transient (persist_graph=False). Key metadata is seeded
    from key_metadata.json to restore cycle count, promoted keys, and
    per-key session counts across server restarts.
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
    )

    # Seed key metadata from disk (survives restarts)
    metadata = _load_key_metadata(config.key_metadata_path)
    if metadata:
        loop.seed_key_metadata(metadata)

    return loop


def run_consolidation(
    model,
    tokenizer,
    config: ServerConfig,
    session_buffer: SessionBuffer,
    loop: ConsolidationLoop | None = None,
) -> dict:
    """Run consolidation on all pending sessions.

    Each pending session becomes one cycle. After all cycles:
    - Update per-key session counts for promotion tracking
    - Promote keys that reach the threshold to semantic
    - Save key_metadata.json, registry, and keyed_pairs

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

    # Run one cycle per pending session, tracking per-session entities
    cycle_results = []
    session_ids = []
    for session in pending:
        session_id = session["session_id"]
        transcript = session["transcript"]
        session_speaker_id = session.get("speaker_id")

        # Skip anonymous sessions — no speaker, no key ownership
        if not session_speaker_id:
            logger.warning("Skipping session %s: no speaker_id", session_id)
            session_ids.append(session_id)  # still archive it
            continue

        # Snapshot keys before cycle to identify new ones by set diff
        keys_before = set(loop.indexed_key_qa.keys())
        session_ids.append(session_id)

        result = loop.run_cycle(transcript, session_id)
        cycle_results.append(result)

        # Tag newly created keys with speaker_id
        new_keys = set(loop.indexed_key_qa.keys()) - keys_before
        for key in new_keys:
            loop.indexed_key_qa[key]["speaker_id"] = session_speaker_id

        # Update per-key session counts using this session's entities
        _increment_key_sessions(loop, session_id)

        logger.info(
            "Cycle %d (session=%s): %d relations, loss=%.4f",
            result.cycle_index,
            session_id,
            result.relations_extracted,
            result.episodic_train_loss or 0.0,
        )

    # Key-level promotion: promote keys that reached the threshold
    newly_promoted = _promote_mature_keys(loop, config)

    # Save all server artifacts
    _save_keyed_pairs_for_router(loop, config)
    _save_registry(loop, config)
    _save_key_metadata(loop, config)

    session_buffer.mark_consolidated(session_ids)

    elapsed = time.time() - start_time
    last = cycle_results[-1] if cycle_results else None

    summary = {
        "status": "complete",
        "sessions": len(session_ids),
        "cycles": len(cycle_results),
        "total_relations": sum(r.relations_extracted for r in cycle_results),
        "newly_promoted": len(newly_promoted),
        "episodic_keys": len(loop.episodic_simhash),
        "semantic_keys": len(loop.semantic_simhash),
        "train_loss": last.episodic_train_loss if last else None,
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


def _atomic_json_write(data: dict | list, path: Path) -> None:
    """Write JSON atomically: write to .tmp, then os.replace()."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, path)


def _load_key_metadata(path: Path) -> dict | None:
    """Load key metadata from disk. Returns None if not found."""
    if not path.exists():
        logger.info("No key metadata found at %s, starting fresh", path)
        return None
    with open(path) as f:
        return json.load(f)


def _save_key_metadata(loop: ConsolidationLoop, config: ServerConfig) -> None:
    """Save key metadata for cross-restart persistence."""
    metadata = {
        "cycle_count": loop.cycle_count,
        "promoted_keys": sorted(loop.promoted_keys),
        "keys": {key: {"sessions_seen": count} for key, count in loop.key_sessions.items()},
    }
    _atomic_json_write(metadata, config.key_metadata_path)


def _save_registry(loop: ConsolidationLoop, config: ServerConfig) -> None:
    """Save combined SimHash registry (no personal data)."""
    combined = {}
    for key, simhash in loop.episodic_simhash.items():
        combined[key] = {"simhash": simhash, "adapter": "episodic"}
    for key, simhash in loop.semantic_simhash.items():
        combined[key] = {"simhash": simhash, "adapter": "semantic"}
    _atomic_json_write(combined, config.registry_path)


def _save_keyed_pairs_for_router(loop: ConsolidationLoop, config: ServerConfig) -> None:
    """Save keyed_pairs.json per adapter for router entity indexing."""
    config.adapter_dir.mkdir(parents=True, exist_ok=True)

    # Episodic keyed_pairs at the top level
    _write_keyed_pairs(
        loop.indexed_key_qa,
        loop.episodic_simhash,
        config.adapter_dir / "keyed_pairs.json",
    )

    # Semantic keyed_pairs in the semantic adapter directory
    if loop.semantic_simhash:
        sem_dir = config.adapter_dir / "semantic"
        sem_dir.mkdir(parents=True, exist_ok=True)
        _write_keyed_pairs(
            loop.indexed_key_qa,
            loop.semantic_simhash,
            sem_dir / "keyed_pairs.json",
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
