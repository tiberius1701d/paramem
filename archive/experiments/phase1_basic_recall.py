"""Phase 1 Experiment: Basic LoRA fine-tuning and recall evaluation.

This experiment:
1. Loads a base model with QLoRA quantization
2. Creates an episodic LoRA adapter
3. Fine-tunes on personal facts
4. Evaluates recall: adapted model vs. base model
5. Logs results to wandb

Usage:
    python experiments/phase1_basic_recall.py
    python experiments/phase1_basic_recall.py --config configs/default.yaml
    python experiments/phase1_basic_recall.py --no-wandb
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from paramem.evaluation.recall import compare_base_vs_adapted
from paramem.models.loader import (
    create_adapter,
    get_adapter_info,
    load_base_model,
)
from paramem.training.dataset import PersonalFactsDataset
from paramem.training.trainer import train_adapter
from paramem.utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("phase1")


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Basic LoRA recall")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--num-epochs", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.no_wandb:
        config.wandb.enabled = False
    if args.num_epochs is not None:
        config.training.num_epochs = args.num_epochs

    data_path = Path(config.paths.data_dir) / "synthetic" / "personal_facts.json"
    output_dir = Path(config.paths.output_dir) / "phase1"

    if config.wandb.enabled:
        import wandb

        wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity or None,
            name="phase1-basic-recall",
            config={
                "model_id": config.model.model_id,
                "quantization": config.model.quantization,
                "adapter_rank": config.adapters["episodic"].rank,
                "adapter_lr": config.adapters["episodic"].learning_rate,
                "num_epochs": config.training.num_epochs,
                "batch_size": config.training.batch_size,
                "max_seq_length": config.training.max_seq_length,
            },
        )

    # Step 1: Load base model
    logger.info("=== Step 1: Loading base model ===")
    model, tokenizer = load_base_model(config.model)

    # Step 2: Create episodic adapter
    logger.info("=== Step 2: Creating episodic adapter ===")
    adapter_config = config.adapters.get("episodic")
    if adapter_config is None:
        raise ValueError("No 'episodic' adapter defined in config")
    model = create_adapter(model, adapter_config, adapter_name="episodic")
    logger.info("Adapter info: %s", get_adapter_info(model))

    # Step 3: Prepare dataset
    logger.info("=== Step 3: Preparing dataset ===")
    dataset = PersonalFactsDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=config.training.max_seq_length,
    )
    logger.info("Dataset size: %d examples", len(dataset))

    # Step 4: Train
    logger.info("=== Step 4: Training ===")
    metrics = train_adapter(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        adapter_name="episodic",
        training_config=config.training,
        adapter_config=adapter_config,
        wandb_config=config.wandb,
        output_dir=output_dir / "adapters" / "episodic",
        run_name="phase1-basic-recall",
    )
    logger.info("Training metrics: %s", metrics)

    # Step 5: Evaluate recall
    logger.info("=== Step 5: Evaluating recall ===")
    comparison = compare_base_vs_adapted(
        model=model,
        tokenizer=tokenizer,
        data_path=str(data_path),
        adapter_name="episodic",
    )

    # Print results
    print("\n" + "=" * 60)
    print("PHASE 1 RESULTS: Base vs. Adapted Model Recall")
    print("=" * 60)
    print(f"Base model recall:    {comparison['base_recall']:.3f}")
    print(f"Adapted model recall: {comparison['adapted_recall']:.3f}")
    print(f"Delta:                +{comparison['delta']:.3f}")
    print("\nPer-category (adapted):")
    for cat, score in comparison["adapted_details"]["category_scores"].items():
        base_cat = comparison["base_details"]["category_scores"].get(cat, 0.0)
        print(f"  {cat:15s}: {score:.3f} (base: {base_cat:.3f})")
    print("=" * 60)

    # Save results
    results_path = output_dir / "results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        "base_recall": comparison["base_recall"],
        "adapted_recall": comparison["adapted_recall"],
        "delta": comparison["delta"],
        "category_scores": comparison["adapted_details"]["category_scores"],
        "fact_scores": comparison["adapted_details"]["fact_scores"],
    }
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info("Results saved to %s", results_path)

    if config.wandb.enabled:
        import wandb

        wandb.log(
            {
                "base_recall": comparison["base_recall"],
                "adapted_recall": comparison["adapted_recall"],
                "recall_delta": comparison["delta"],
                **{
                    f"category/{cat}": score
                    for cat, score in comparison["adapted_details"]["category_scores"].items()
                },
            }
        )
        wandb.finish()

    return comparison


if __name__ == "__main__":
    main()
