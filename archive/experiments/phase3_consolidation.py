"""Phase 3 Experiment: Consolidation Loop across 55 sessions.

Runs the full consolidation pipeline:
1. Load base model + create episodic and semantic adapters
2. Process 55 synthetic sessions through the consolidation loop
3. Track promotion, decay, and retention metrics
4. Evaluate against Phase 3 spec targets

Targets:
- Promoted memory retention >80% across 50+ cycles
- Measurable episodic decay for unreinforced memories over 10 cycles
- Semantic adapter <5% drift on consolidated facts after 20 cycles
- Wall-clock <30 min per session on RTX 5070

Usage:
    python experiments/phase3_consolidation.py                     # full 55 sessions
    python experiments/phase3_consolidation.py --max-cycles 5      # first 5 only
    python experiments/phase3_consolidation.py --start-cycle 6     # resume from cycle 6
    python experiments/phase3_consolidation.py --output-dir outputs/phase3b
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Ensure project root is on path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment
from dotenv import load_dotenv  # noqa: E402

load_dotenv(project_root / ".env")

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import wandb  # noqa: E402, I001

from paramem.evaluation.consolidation_metrics import (  # noqa: E402
    compute_consolidation_metrics,
    format_phase3_summary,
)
from paramem.models.loader import create_adapter, load_base_model  # noqa: E402
from paramem.training.consolidation import ConsolidationLoop  # noqa: E402
from paramem.utils.config import load_config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 3 Consolidation Loop")
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=None,
        help="Stop after this many cycles (default: run all sessions)",
    )
    parser.add_argument(
        "--start-cycle",
        type=int,
        default=1,
        help="Start from this cycle number (1-indexed, default: 1). "
        "Sessions before this are skipped. The ConsolidationLoop's "
        "internal cycle counter is set accordingly.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/phase3b",
        help="Output directory relative to project root (default: outputs/phase3b)",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=20,
        help="Training epochs per cycle (default: 20)",
    )
    parser.add_argument(
        "--wandb-group",
        type=str,
        default="phase3b-consolidation",
        help="wandb group name (default: phase3b-consolidation)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config()
    output_dir = project_root / args.output_dir

    # Initialize wandb
    if config.wandb.enabled:
        wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            name=f"{args.wandb_group}",
            group=args.wandb_group,
            config={
                "model": config.model.model_id,
                "num_epochs": args.num_epochs,
                "start_cycle": args.start_cycle,
                "max_cycles": args.max_cycles,
                "consolidation": {
                    "promotion_threshold": config.consolidation.promotion_threshold,
                    "decay_window": config.consolidation.decay_window,
                    "episodic_new_weight": config.consolidation.episodic_new_weight,
                    "semantic_replay_weight": config.consolidation.semantic_replay_weight,
                },
                "graph": {
                    "extraction_temperature": config.graph.extraction_temperature,
                    "entity_similarity_threshold": config.graph.entity_similarity_threshold,
                },
            },
            resume="allow",
        )

    # Load model
    logger.info("Loading base model...")
    model, tokenizer = load_base_model(config.model)

    # Create both adapters
    episodic_config = config.adapters.get("episodic")
    semantic_config = config.adapters.get("semantic")

    if episodic_config is None or semantic_config is None:
        raise ValueError("Both 'episodic' and 'semantic' adapter configs required")

    model = create_adapter(model, episodic_config, "episodic")
    model = create_adapter(model, semantic_config, "semantic")

    # Load sessions
    sessions_path = project_root / "data" / "synthetic" / "synthetic_sessions.json"
    with open(sessions_path) as f:
        sessions = json.load(f)
    logger.info("Loaded %d sessions", len(sessions))

    # Build consolidation training config
    from paramem.utils.config import TrainingConfig

    consolidation_training = TrainingConfig(
        batch_size=config.training.batch_size,
        gradient_accumulation_steps=2,
        max_seq_length=config.training.max_seq_length,
        num_epochs=args.num_epochs,
        warmup_ratio=config.training.warmup_ratio,
        weight_decay=config.training.weight_decay,
        gradient_checkpointing=config.training.gradient_checkpointing,
        max_grad_norm=config.training.max_grad_norm,
        seed=config.training.seed,
    )

    # Run consolidation loop
    loop = ConsolidationLoop(
        model=model,
        tokenizer=tokenizer,
        consolidation_config=config.consolidation,
        training_config=consolidation_training,
        episodic_adapter_config=episodic_config,
        semantic_adapter_config=semantic_config,
        wandb_config=config.wandb,
        output_dir=output_dir,
        extraction_temperature=config.graph.extraction_temperature,
    )

    # If resuming, fast-forward the cycle counter
    if args.start_cycle > 1:
        loop.cycle_count = args.start_cycle - 1
        logger.info(
            "Resuming from cycle %d (skipping %d sessions)",
            args.start_cycle,
            args.start_cycle - 1,
        )

        # Load existing graph state if resuming
        if loop.graph_path.exists():
            loop.merger.load_graph(loop.graph_path)
            logger.info("Loaded existing cumulative graph")

        # Load replay pools from previous results if available
        prev_results_path = output_dir / "results.json"
        if prev_results_path.exists():
            logger.info("Found previous results, replay pools will rebuild from training")

    # Determine session range
    start_idx = args.start_cycle - 1
    if args.max_cycles is not None:
        end_idx = min(start_idx + args.max_cycles, len(sessions))
    else:
        end_idx = len(sessions)

    session_slice = sessions[start_idx:end_idx]
    logger.info(
        "Running cycles %d-%d (%d sessions)",
        start_idx + 1,
        end_idx,
        len(session_slice),
    )

    results = []
    for session in session_slice:
        result = loop.run_cycle(
            session_transcript=session["transcript"],
            session_id=session["session_id"],
        )
        results.append(result)

        # Log to wandb
        if config.wandb.enabled:
            wandb.log(
                {
                    "cycle": result.cycle_index,
                    "entities_extracted": result.entities_extracted,
                    "relations_extracted": result.relations_extracted,
                    "nodes_promoted": result.nodes_promoted,
                    "nodes_decayed": result.nodes_decayed,
                    "episodic_loss": result.episodic_train_loss,
                    "semantic_loss": result.semantic_train_loss,
                    "wall_clock_seconds": result.wall_clock_seconds,
                }
            )

        logger.info(
            "Cycle %d/%d complete (%.1fs)",
            result.cycle_index,
            len(sessions),
            result.wall_clock_seconds,
        )

    # Compute aggregate metrics
    metrics = compute_consolidation_metrics(results)

    # Placeholder retention/decay/drift (full evaluation requires recall probing)
    retention = {"mean_retention": 0.0, "min_retention": 0.0, "per_node": {}}
    decay = {"mean_decay_rate": 0.0, "per_node": {}}
    drift = {"mean_drift": 0.0, "max_drift": 0.0, "per_node": {}}

    summary = format_phase3_summary(metrics, retention, decay, drift)
    print(summary)

    # Save results (append if resuming)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json"

    # Load existing results if resuming
    existing_cycle_results = []
    if args.start_cycle > 1 and results_path.exists():
        with open(results_path) as f:
            existing_data = json.load(f)
        existing_cycle_results = existing_data.get("cycle_results", [])

    new_cycle_results = [
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
            "promoted_nodes": r.promoted_nodes,
            "decayed_nodes": r.decayed_nodes,
        }
        for r in results
    ]

    all_cycle_results = existing_cycle_results + new_cycle_results

    results_data = {
        "num_epochs": args.num_epochs,
        "total_cycles": len(all_cycle_results),
        "total_entities_extracted": metrics.total_entities_extracted,
        "total_promotions": metrics.total_promotions,
        "total_decays": metrics.total_decays,
        "mean_wall_clock_seconds": metrics.mean_wall_clock_seconds,
        "cycle_results": all_cycle_results,
    }
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)

    if config.wandb.enabled:
        wandb.finish()

    logger.info("Phase 3b experiment complete. Results saved to %s", output_dir)


if __name__ == "__main__":
    main()
