"""F4.10: Indexed key memory integration with consolidation loop.

Validates that the indexed key mechanism (F4.9c) works within the full
consolidation pipeline: session extraction → key assignment → indexed
training → promotion → per-key recall evaluation.

Smoke test: 10 sessions, 30 epochs, max 20 keys
Expected: ~4 min model load + ~60s per cycle × 10 = ~14 min total

Usage:
    python experiments/f4_10_indexed_consolidation.py                     # full 10 sessions
    python experiments/f4_10_indexed_consolidation.py --max-cycles 3      # first 3 only
    python experiments/f4_10_indexed_consolidation.py --max-keys 10       # cap at 10 keys
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(project_root / ".env")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from paramem.models.loader import create_adapter, load_base_model, switch_adapter  # noqa: E402
from paramem.training.consolidation import ConsolidationLoop  # noqa: E402
from paramem.training.indexed_memory import probe_all_keys  # noqa: E402
from paramem.utils.config import (  # noqa: E402
    AdapterConfig,
    ConsolidationConfig,
    TrainingConfig,
    load_config,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def make_output_dir(base_name, model_id):
    """Create timestamped, model-specific output directory."""
    from datetime import datetime

    model_short = model_id.split("/")[-1].lower().replace(" ", "-")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = project_root / "outputs" / base_name / model_short / timestamp
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_args():
    parser = argparse.ArgumentParser(description="F4.10: Indexed Key Consolidation")
    parser.add_argument("--max-cycles", type=int, default=10)
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--max-keys", type=int, default=20)
    parser.add_argument("--rank", type=int, default=8)
    return parser.parse_args()


def evaluate_indexed_recall(model, tokenizer, loop, adapter_name):
    """Probe all active indexed keys for an adapter and return recall stats."""
    switch_adapter(model, adapter_name)
    model.gradient_checkpointing_disable()

    registry = loop.indexed_key_registry
    simhash = loop.episodic_simhash if adapter_name == "episodic" else loop.semantic_simhash

    # Determine which keys belong to this adapter
    if adapter_name == "episodic":
        keys = registry.list_active()
    else:
        keys = list(simhash.keys())

    if not keys:
        return {"adapter": adapter_name, "total": 0, "recalled": 0, "details": []}

    results = probe_all_keys(
        model,
        tokenizer,
        keys,
        registry=simhash,
        confidence_threshold=0.5,
    )

    details = []
    recalled = 0
    for key in keys:
        result = results[key]
        original = loop.indexed_key_qa.get(key, {})
        if result is not None:
            recalled += 1
            exact = (
                result.get("question", "").strip() == original.get("question", "").strip()
                and result.get("answer", "").strip() == original.get("answer", "").strip()
            )
            details.append(
                {
                    "key": key,
                    "status": "EXACT" if exact else "PARTIAL",
                    "confidence": result.get("confidence", 0.0),
                    "raw_output": result.get("raw_output", ""),
                    "original_q": original.get("question", ""),
                    "recalled_q": result.get("question", ""),
                    "original_a": original.get("answer", ""),
                    "recalled_a": result.get("answer", ""),
                }
            )
        else:
            details.append(
                {
                    "key": key,
                    "status": "MISS",
                    "confidence": 0.0,
                    "raw_output": "",
                    "original_q": original.get("question", ""),
                }
            )

    return {
        "adapter": adapter_name,
        "total": len(keys),
        "recalled": recalled,
        "recall_rate": recalled / len(keys) if keys else 0.0,
        "details": details,
    }


def main():
    args = parse_args()
    config = load_config()
    output_dir = make_output_dir("f4_10_indexed_consolidation", config.model.model_id)

    logger.info("Loading base model...")
    model, tokenizer = load_base_model(config.model)

    adapter_config = AdapterConfig(
        rank=args.rank,
        alpha=args.rank * 2,
        learning_rate=1e-4,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        dropout=0.0,
    )

    model = create_adapter(model, adapter_config, "episodic")
    model = create_adapter(model, adapter_config, "semantic")

    # Load sessions
    sessions_path = project_root / "data" / "synthetic" / "synthetic_sessions.json"
    with open(sessions_path) as f:
        sessions = json.load(f)
    logger.info("Loaded %d sessions", len(sessions))

    session_slice = sessions[: args.max_cycles]

    # Consolidation config with indexed key replay enabled
    consolidation_config = ConsolidationConfig(
        indexed_key_replay_enabled=True,
        max_active_keys=args.max_keys,
        key_retirement_threshold=0.1,
        key_retirement_cycles=3,
        promotion_threshold=3,
        decay_window=10,
    )

    training_config = TrainingConfig(
        batch_size=1,
        gradient_accumulation_steps=2,
        max_seq_length=1024,
        num_epochs=args.num_epochs,
        warmup_ratio=0.1,
        weight_decay=0.01,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        seed=42,
    )

    loop = ConsolidationLoop(
        model=model,
        tokenizer=tokenizer,
        consolidation_config=consolidation_config,
        training_config=training_config,
        episodic_adapter_config=adapter_config,
        semantic_adapter_config=adapter_config,
        output_dir=output_dir,
        extraction_temperature=config.graph.extraction_temperature,
    )

    # Run consolidation cycles
    cycle_results = []
    total_start = time.time()

    for session in session_slice:
        result = loop.run_cycle(
            session_transcript=session["transcript"],
            session_id=session["session_id"],
        )
        cycle_results.append(result)

        ep_keys = loop.indexed_key_registry.list_active()
        sem_keys = list(loop.semantic_simhash.keys())
        logger.info(
            "Cycle %d: %d relations, %d promoted, %d episodic keys, %d semantic keys (%.1fs)",
            result.cycle_index,
            result.relations_extracted,
            result.nodes_promoted,
            len(ep_keys),
            len(sem_keys),
            result.wall_clock_seconds,
        )

    total_time = time.time() - total_start

    # Final evaluation
    print("\n" + "=" * 72)
    print("FINAL EVALUATION — Indexed Key Recall")
    print("=" * 72)

    model.gradient_checkpointing_disable()

    ep_eval = evaluate_indexed_recall(model, tokenizer, loop, "episodic")
    sem_eval = evaluate_indexed_recall(model, tokenizer, loop, "semantic")

    print(
        f"\nEpisodic: {ep_eval['recalled']}/{ep_eval['total']} "
        f"({ep_eval.get('recall_rate', 0):.1%})"
    )
    for d in ep_eval["details"]:
        print(f"  [{d['status']}] {d['key']} conf={d['confidence']:.3f}: {d.get('original_q', '')}")

    print(
        f"\nSemantic: {sem_eval['recalled']}/{sem_eval['total']} "
        f"({sem_eval.get('recall_rate', 0):.1%})"
    )
    for d in sem_eval["details"]:
        print(f"  [{d['status']}] {d['key']} conf={d['confidence']:.3f}: {d.get('original_q', '')}")

    # Summary
    print("\n" + "=" * 72)
    print("SUMMARY")
    print(f"  Cycles:          {len(cycle_results)}")
    print(f"  Total keys:      {loop._indexed_next_index - 1}")
    print(f"  Episodic active: {ep_eval['total']}")
    print(f"  Semantic active: {sem_eval['total']}")
    print(f"  Episodic recall: {ep_eval['recalled']}/{ep_eval['total']}")
    print(f"  Semantic recall: {sem_eval['recalled']}/{sem_eval['total']}")
    print(f"  Total time:      {total_time:.0f}s ({total_time / 60:.1f} min)")
    print(f"  Mean cycle time: {total_time / len(cycle_results):.1f}s")
    print("=" * 72)

    # Save results
    results_data = {
        "experiment": "f4_10_indexed_consolidation",
        "model_id": config.model.model_id,
        "config": {
            "max_cycles": args.max_cycles,
            "num_epochs": args.num_epochs,
            "max_keys": args.max_keys,
            "rank": args.rank,
        },
        "total_time_seconds": total_time,
        "total_keys_assigned": loop._indexed_next_index - 1,
        "episodic_eval": {k: v for k, v in ep_eval.items() if k != "details"},
        "semantic_eval": {k: v for k, v in sem_eval.items() if k != "details"},
        "episodic_details": ep_eval["details"],
        "semantic_details": sem_eval["details"],
        "cycle_results": [
            {
                "cycle": r.cycle_index,
                "session_id": r.session_id,
                "entities": r.entities_extracted,
                "relations": r.relations_extracted,
                "promoted": r.nodes_promoted,
                "decayed": r.nodes_decayed,
                "episodic_loss": r.episodic_train_loss,
                "semantic_loss": r.semantic_train_loss,
                "wall_clock": r.wall_clock_seconds,
            }
            for r in cycle_results
        ],
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2, default=str)

    print(f"\nResults saved to {results_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
