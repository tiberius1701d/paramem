"""Consolidation job — extracts knowledge from buffered sessions and retrains the adapter.

The cumulative knowledge graph is the single structural record. On each
consolidation cycle: merge new session graphs → regenerate QA from the
full graph → retrain the adapter from scratch. No external QA file is
used as input — facts live in the adapter weights and the graph.
"""

import json
import logging
import time
from pathlib import Path

from paramem.graph.extractor import extract_graph
from paramem.graph.merger import GraphMerger
from paramem.graph.qa_generator import generate_qa_from_relations
from paramem.models.loader import create_adapter, save_adapter
from paramem.server.config import ServerConfig
from paramem.server.session_buffer import SessionBuffer
from paramem.training.indexed_memory import (
    assign_keys,
    build_enriched_registry,
    format_indexed_training,
    save_registry,
)
from paramem.training.trainer import train_adapter

logger = logging.getLogger(__name__)


def run_consolidation(
    model,
    tokenizer,
    config: ServerConfig,
    session_buffer: SessionBuffer,
) -> dict:
    """Run a full consolidation cycle on all pending session transcripts.

    Pipeline:
        1. Extract graphs from pending sessions, merge into cumulative graph
        2. Regenerate QA pairs from the FULL cumulative graph
        3. Assign keys, retrain adapter from scratch
        4. Persist adapter, registry, graph

    The cumulative graph is the source of truth for what knowledge exists.
    QA pairs are always regenerated — never loaded from a cached file.
    """
    start_time = time.time()

    pending = session_buffer.get_pending()
    if not pending:
        logger.info("No pending sessions to consolidate")
        return {"status": "no_pending", "sessions": 0}

    logger.info("Consolidating %d pending sessions", len(pending))

    # Load cumulative graph
    merger = GraphMerger()
    if config.graph_path.exists():
        merger.load_graph(config.graph_path)

    # Phase 1: Extract and merge graphs from pending sessions
    session_ids = []
    new_relation_count = 0

    for session in pending:
        session_id = session["session_id"]
        transcript = session["transcript"]
        session_ids.append(session_id)

        logger.info("Extracting graph from session: %s", session_id)
        model.gradient_checkpointing_disable()

        session_graph = extract_graph(model, tokenizer, transcript, session_id, temperature=0.0)

        if not session_graph.relations:
            logger.warning("No relations extracted from session %s", session_id)
            continue

        new_relation_count += len(session_graph.relations)
        merger.merge(session_graph)
        logger.info(
            "Session %s: %d relations extracted",
            session_id,
            len(session_graph.relations),
        )

    # Phase 2: Regenerate QA from the full cumulative graph
    all_triples = merger.get_all_triples()
    if not all_triples:
        logger.warning("Cumulative graph has no triples")
        session_buffer.mark_consolidated(session_ids)
        return {"status": "no_triples", "sessions": len(session_ids)}

    logger.info("Regenerating QA from full graph: %d triples", len(all_triples))
    relations = [{"subject": s, "predicate": p, "object": o} for s, p, o in all_triples]
    all_qa = generate_qa_from_relations(relations, model, tokenizer)

    if not all_qa:
        logger.warning("QA generation produced no pairs from %d triples", len(all_triples))
        session_buffer.mark_consolidated(session_ids)
        return {"status": "no_qa_pairs", "sessions": len(session_ids)}

    logger.info("Generated %d QA pairs from %d triples", len(all_qa), len(all_triples))

    # Phase 3: Assign keys, retrain adapter from scratch
    keyed_pairs = assign_keys(all_qa)

    logger.info("Training on %d keyed pairs", len(keyed_pairs))
    training_examples = format_indexed_training(keyed_pairs, tokenizer)

    adapter_name = "episodic"
    from peft import PeftModel as _PeftModel

    if isinstance(model, _PeftModel):
        model = model.base_model.model

    model = create_adapter(model, config.adapter_config, adapter_name)

    config.adapter_dir.mkdir(parents=True, exist_ok=True)
    metrics = train_adapter(
        model,
        tokenizer,
        training_examples,
        adapter_name,
        config.training_config,
        config.adapter_config,
        output_dir=config.adapter_dir / "checkpoints",
    )

    # Phase 4: Persist adapter, registry, graph, keyed_pairs
    save_adapter(model, config.adapter_dir, adapter_name)

    kp_path = config.adapter_dir / "keyed_pairs.json"
    with open(kp_path, "w") as f:
        json.dump(
            [
                {
                    k: v
                    for k, v in kp.items()
                    if k
                    in (
                        "key",
                        "question",
                        "answer",
                        "source_predicate",
                        "source_subject",
                        "source_object",
                    )
                }
                for kp in keyed_pairs
            ],
            f,
            indent=2,
        )

    registry = build_enriched_registry(
        keyed_pairs,
        session_id=session_ids[-1] if session_ids else None,
        existing=_load_existing_registry(config.registry_path),
    )
    save_registry(registry, config.registry_path)

    merger.save_graph(config.graph_path)

    session_buffer.mark_consolidated(session_ids)

    elapsed = time.time() - start_time
    result = {
        "status": "complete",
        "sessions": len(session_ids),
        "new_relations": new_relation_count,
        "total_triples": len(all_triples),
        "total_qa_pairs": len(all_qa),
        "total_keys": len(keyed_pairs),
        "train_loss": metrics.get("train_loss", -1),
        "elapsed_seconds": round(elapsed, 1),
    }
    logger.info("Consolidation complete: %s", result)
    return result


def _load_existing_registry(registry_path: Path) -> dict | None:
    """Load existing registry if it exists."""
    if not registry_path.exists():
        return None
    with open(registry_path) as f:
        return json.load(f)
