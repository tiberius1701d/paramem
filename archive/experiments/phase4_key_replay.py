"""Phase 4: Key-Addressable Replay — consolidation with adapter weights as source of truth.

Runs the consolidation loop with key_replay_enabled=True. Each session's
knowledge graph is stored in the adapter with a unique key. During replay,
the model reconstructs graphs from its own weights, merges with the new
session, and retrains on the complete result.

Compares recall against Phase 3b baseline (pool-based replay).

Usage:
    python experiments/phase4_key_replay.py
    python experiments/phase4_key_replay.py --max-cycles 10 --num-epochs 5   # smoke test
    python experiments/phase4_key_replay.py --num-epochs 20                  # match Phase 3b
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(project_root / ".env")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import wandb  # noqa: E402
from paramem.evaluation.consolidation_metrics import (  # noqa: E402
    compute_consolidation_metrics,
)
from paramem.evaluation.embedding_scorer import compute_similarity  # noqa: E402
from paramem.evaluation.key_fidelity import measure_all_fidelity  # noqa: E402
from paramem.evaluation.recall import generate_answer  # noqa: E402
from paramem.graph.qa_generator import generate_qa_from_relations  # noqa: E402
from paramem.models.loader import (  # noqa: E402
    create_adapter,
    load_base_model,
    switch_adapter,
)
from paramem.training.consolidation import ConsolidationLoop  # noqa: E402
from paramem.training.dataset import _format_inference_prompt  # noqa: E402
from paramem.training.key_replay import (  # noqa: E402
    reconstruct_all,
)
from paramem.utils.config import (  # noqa: E402
    ConsolidationConfig,
    TrainingConfig,
    load_config,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Same facts as Phase 3b recall probe
PROMOTED_FACTS = [
    {"subject": "Alex", "predicate": "lives_in", "object": "Heilbronn"},
    {"subject": "Alex", "predicate": "works_at", "object": "AutoMate"},
    {"subject": "Alex", "predicate": "has_pet", "object": "Luna the German Shepherd"},
    {"subject": "Alex", "predicate": "prefers", "object": "Python"},
    {"subject": "Alex", "predicate": "prefers", "object": "dark mode"},
    {"subject": "Alex", "predicate": "studies_at", "object": "KIT"},
    {"subject": "Alex", "predicate": "knows", "object": "Jonas"},
    {"subject": "Alex", "predicate": "prefers", "object": "black coffee"},
    {"subject": "Alex", "predicate": "has_hobby", "object": "hiking in the Black Forest"},
    {"subject": "Maria", "predicate": "manages", "object": "robotics team budget"},
]

ONESHOT_FACTS = [
    {"subject": "Alex", "predicate": "visited", "object": "Barcelona last weekend"},
    {"subject": "Alex", "predicate": "read", "object": "a paper on transformers"},
    {"subject": "Alex", "predicate": "fixed", "object": "a bug in the CI pipeline"},
    {"subject": "Alex", "predicate": "attended", "object": "a meetup on Rust"},
    {"subject": "Alex", "predicate": "bought", "object": "new running shoes"},
]


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 4: Key-Addressable Replay")
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=20,
        help="Number of sessions to process (default: 20, fast iteration)",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=20,
        help="Training epochs per cycle (default: 20, matches Phase 3b)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/phase4_key_replay",
        help="Output directory",
    )
    parser.add_argument(
        "--wandb-group",
        type=str,
        default="phase4-key-replay",
        help="wandb group name",
    )
    return parser.parse_args()


def probe_recall(model, tokenizer, qa_pairs, adapter_name=None):
    """Probe model recall on QA pairs, optionally with a specific adapter."""
    from peft import PeftModel

    model.gradient_checkpointing_disable()

    if adapter_name and isinstance(model, PeftModel):
        model.enable_adapter_layers()
        switch_adapter(model, adapter_name)
    elif isinstance(model, PeftModel):
        model.disable_adapter_layers()

    results = []
    for qa in qa_pairs:
        prompt = _format_inference_prompt(qa["question"], tokenizer)
        generated = generate_answer(model, tokenizer, prompt, temperature=0.1)
        score = compute_similarity(qa["answer"], generated)
        results.append(
            {
                "question": qa["question"],
                "expected": qa["answer"],
                "generated": generated,
                "embedding_score": score,
            }
        )

    scores = [r["embedding_score"] for r in results]
    return {
        "mean_embedding": sum(scores) / len(scores) if scores else 0.0,
        "details": results,
    }


def main():
    args = parse_args()
    config = load_config()
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize wandb
    if config.wandb.enabled:
        wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            name=args.wandb_group,
            group=args.wandb_group,
            config={
                "model": config.model.model_id,
                "num_epochs": args.num_epochs,
                "max_cycles": args.max_cycles,
                "key_replay_enabled": True,
            },
            resume="allow",
        )

    # Load model
    logger.info("Loading base model...")
    model, tokenizer = load_base_model(config.model)

    episodic_config = config.adapters.get("episodic")
    semantic_config = config.adapters.get("semantic")
    if episodic_config is None or semantic_config is None:
        raise ValueError("Both 'episodic' and 'semantic' adapter configs required")

    model = create_adapter(model, episodic_config, "episodic")
    model = create_adapter(model, semantic_config, "semantic")

    # Load sessions
    sessions_path = project_root / "data" / "sessions" / "synthetic_sessions.json"
    with open(sessions_path) as f:
        sessions = json.load(f)
    logger.info("Loaded %d sessions, using first %d", len(sessions), args.max_cycles)

    session_slice = sessions[: args.max_cycles]

    # Build configs with key replay ENABLED
    consolidation_config = ConsolidationConfig(
        promotion_threshold=config.consolidation.promotion_threshold,
        decay_window=config.consolidation.decay_window,
        procedural_detection_window=config.consolidation.procedural_detection_window,
        episodic_new_weight=config.consolidation.episodic_new_weight,
        semantic_replay_weight=config.consolidation.semantic_replay_weight,
        key_replay_enabled=True,
        max_active_keys=config.consolidation.max_active_keys,
        key_retirement_threshold=config.consolidation.key_retirement_threshold,
        key_retirement_cycles=config.consolidation.key_retirement_cycles,
    )

    training_config = TrainingConfig(
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

    # Run consolidation loop with key replay
    loop = ConsolidationLoop(
        model=model,
        tokenizer=tokenizer,
        consolidation_config=consolidation_config,
        training_config=training_config,
        episodic_adapter_config=episodic_config,
        semantic_adapter_config=semantic_config,
        wandb_config=config.wandb,
        output_dir=output_dir,
        extraction_temperature=config.graph.extraction_temperature,
    )

    results = []
    for session in session_slice:
        result = loop.run_cycle(
            session_transcript=session["transcript"],
            session_id=session["session_id"],
        )
        results.append(result)

        if config.wandb.enabled:
            log_data = {
                "cycle": result.cycle_index,
                "entities_extracted": result.entities_extracted,
                "relations_extracted": result.relations_extracted,
                "nodes_promoted": result.nodes_promoted,
                "nodes_decayed": result.nodes_decayed,
                "episodic_loss": result.episodic_train_loss,
                "semantic_loss": result.semantic_train_loss,
                "wall_clock_seconds": result.wall_clock_seconds,
                "active_keys": len(loop.key_registry),
            }
            wandb.log(log_data)

        logger.info(
            "Cycle %d/%d complete (%.1fs) — %d active keys",
            result.cycle_index,
            args.max_cycles,
            result.wall_clock_seconds,
            len(loop.key_registry),
        )

    # === FINAL RECONSTRUCTION FIDELITY ===
    logger.info("=== Final Reconstruction Fidelity ===")
    model.gradient_checkpointing_disable()
    switch_adapter(model, "episodic")

    final_reconstructions = reconstruct_all(model, tokenizer, loop.key_registry, max_keys=50)
    fidelity = measure_all_fidelity(loop.key_original_triples, final_reconstructions)

    print(f"\nReconstruction fidelity ({fidelity['num_keys']} keys):")
    print(f"  Mean precision: {fidelity['mean_precision']:.3f}")
    print(f"  Mean recall:    {fidelity['mean_recall']:.3f}")
    print(f"  Mean F1:        {fidelity['mean_f1']:.3f}")

    # === RECALL PROBE ===
    logger.info("=== Recall Probe ===")

    promoted_qa = generate_qa_from_relations(PROMOTED_FACTS)
    oneshot_qa = generate_qa_from_relations(ONESHOT_FACTS)

    # Base model (adapters disabled)
    logger.info("Probing base model...")
    base_promoted = probe_recall(model, tokenizer, promoted_qa)
    base_oneshot = probe_recall(model, tokenizer, oneshot_qa)

    # Episodic adapter
    logger.info("Probing episodic adapter...")
    epi_promoted = probe_recall(model, tokenizer, promoted_qa, "episodic")
    epi_oneshot = probe_recall(model, tokenizer, oneshot_qa, "episodic")

    # Semantic adapter
    logger.info("Probing semantic adapter...")
    sem_promoted = probe_recall(model, tokenizer, promoted_qa, "semantic")
    sem_oneshot = probe_recall(model, tokenizer, oneshot_qa, "semantic")

    # === PRINT RESULTS ===
    print("\n" + "=" * 72)
    print("Phase 4: Key-Addressable Replay — Recall Probe")
    print(
        f"Sessions: {args.max_cycles}, Epochs: {args.num_epochs}, "
        f"Active keys: {len(loop.key_registry)}"
    )
    print("=" * 72)

    fmt = "{:<25s} {:>15s} {:>15s}"
    print(fmt.format("Condition", "Promoted", "Oneshot"))
    print("-" * 58)
    print(
        fmt.format(
            "Base",
            f"{base_promoted['mean_embedding']:.1%}",
            f"{base_oneshot['mean_embedding']:.1%}",
        )
    )
    print(
        fmt.format(
            "Episodic",
            f"{epi_promoted['mean_embedding']:.1%}",
            f"{epi_oneshot['mean_embedding']:.1%}",
        )
    )
    print(
        fmt.format(
            "Semantic",
            f"{sem_promoted['mean_embedding']:.1%}",
            f"{sem_oneshot['mean_embedding']:.1%}",
        )
    )

    # Deltas
    base_p = base_promoted["mean_embedding"]
    epi_delta = epi_promoted["mean_embedding"] - base_p
    sem_delta = sem_promoted["mean_embedding"] - base_p
    print()
    print(f"Episodic delta vs base (promoted):  {epi_delta:+.1%}")
    print(f"Semantic delta vs base (promoted):  {sem_delta:+.1%}")

    # Load Phase 3b baseline for comparison
    phase3b_probe = project_root / "outputs" / "phase3b" / "recall_probe.json"
    if phase3b_probe.exists():
        with open(phase3b_probe) as f:
            baseline = json.load(f)
        print()
        print("Comparison vs Phase 3b (55 sessions, 20 epochs, pool-based replay):")
        print(f"  Phase 3b episodic promoted:  {baseline['episodic_promoted_emb']:.1%}")
        print(f"  Phase 4  episodic promoted:  {epi_promoted['mean_embedding']:.1%}")
        print(f"  Phase 3b semantic promoted:  {baseline['semantic_promoted_emb']:.1%}")
        print(f"  Phase 4  semantic promoted:  {sem_promoted['mean_embedding']:.1%}")

    # Key registry stats
    print()
    print("Key registry stats:")
    print(f"  Active keys: {len(loop.key_registry)}")
    print(f"  Original triples tracked: {len(loop.key_original_triples)}")
    if fidelity["per_key"]:
        high_fidelity = sum(1 for m in fidelity["per_key"].values() if m["f1"] >= 0.5)
        print(f"  Keys with F1 >= 0.5: {high_fidelity}/{fidelity['num_keys']}")

    print("=" * 72)

    # === SAVE RESULTS ===
    metrics = compute_consolidation_metrics(results)

    results_data = {
        "experiment": "phase4_key_replay",
        "num_sessions": args.max_cycles,
        "num_epochs": args.num_epochs,
        "key_replay_enabled": True,
        "consolidation_metrics": {
            "total_cycles": metrics.total_cycles,
            "total_entities_extracted": metrics.total_entities_extracted,
            "total_promotions": metrics.total_promotions,
            "total_decays": metrics.total_decays,
            "mean_wall_clock_seconds": metrics.mean_wall_clock_seconds,
        },
        "recall_probe": {
            "base_promoted_emb": base_promoted["mean_embedding"],
            "base_oneshot_emb": base_oneshot["mean_embedding"],
            "episodic_promoted_emb": epi_promoted["mean_embedding"],
            "episodic_oneshot_emb": epi_oneshot["mean_embedding"],
            "semantic_promoted_emb": sem_promoted["mean_embedding"],
            "semantic_oneshot_emb": sem_oneshot["mean_embedding"],
            "episodic_delta": epi_delta,
            "semantic_delta": sem_delta,
        },
        "reconstruction_fidelity": {
            "mean_precision": fidelity["mean_precision"],
            "mean_recall": fidelity["mean_recall"],
            "mean_f1": fidelity["mean_f1"],
            "num_keys": fidelity["num_keys"],
            "per_key": fidelity["per_key"],
        },
        "recall_details": {
            "base_promoted": base_promoted["details"],
            "base_oneshot": base_oneshot["details"],
            "episodic_promoted": epi_promoted["details"],
            "episodic_oneshot": epi_oneshot["details"],
            "semantic_promoted": sem_promoted["details"],
            "semantic_oneshot": sem_oneshot["details"],
        },
        "key_registry": {
            "active_keys": loop.key_registry.list_active(),
            "fidelity_history": {
                key: loop.key_registry.get_fidelity_history(key)
                for key in loop.key_registry.list_active()
            },
        },
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
            for r in results
        ],
    }

    results_path = output_dir / "key_replay_results.json"
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)
    logger.info("Results saved to %s", results_path)

    if config.wandb.enabled:
        wandb.finish()

    logger.info("Phase 4 key-replay experiment complete.")


if __name__ == "__main__":
    main()
