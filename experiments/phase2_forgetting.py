"""Phase 2 Experiment: Sequential learning and catastrophic forgetting.

This experiment:
1. Trains an episodic adapter sequentially on multiple topics
2. Measures catastrophic forgetting (baseline)
3. Compares three mitigation strategies: naive replay, EWC, generative replay
4. Reports forgetting reduction vs. baseline

Usage:
    python experiments/phase2_forgetting.py
    python experiments/phase2_forgetting.py --num-topics 4
    python experiments/phase2_forgetting.py --conditions baseline naive
    python experiments/phase2_forgetting.py --no-wandb
"""

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from paramem.evaluation.forgetting import (
    compute_forgetting_metrics,
    compute_forgetting_reduction,
    format_results_table,
)
from paramem.models.loader import create_adapter, load_base_model
from paramem.training.replay import GenerativeReplay, NaiveReplay
from paramem.training.sequential import TopicDefinition, train_sequential
from paramem.utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("phase2")

ALL_CONDITIONS = ["baseline", "naive", "ewc", "generative"]


def reset_adapter(model, adapter_config, adapter_name="episodic"):
    """Delete and recreate the adapter for a clean start."""
    try:
        model.delete_adapter(adapter_name)
    except ValueError:
        pass
    return create_adapter(model, adapter_config, adapter_name=adapter_name)


def run_condition(
    condition_name,
    model,
    tokenizer,
    topics,
    config,
    output_dir,
):
    """Run a single experimental condition."""
    logger.info("\n" + "=" * 60)
    logger.info("CONDITION: %s", condition_name.upper())
    logger.info("=" * 60)

    adapter_config = config.adapters["episodic"]
    replay_config = config.replay

    # Override training epochs to per-topic setting
    training_config = config.training
    training_config.num_epochs = replay_config.epochs_per_topic
    training_config.gradient_accumulation_steps = replay_config.gradient_accumulation_steps

    # Set up wandb for this condition
    wandb_config = config.wandb
    if wandb_config.enabled:
        import wandb

        wandb.init(
            project=wandb_config.project,
            entity=wandb_config.entity or None,
            group="phase2-forgetting",
            name=f"phase2-{condition_name}",
            config={
                "condition": condition_name,
                "num_topics": len(topics),
                "epochs_per_topic": training_config.num_epochs,
                "model_id": config.model.model_id,
                "adapter_rank": adapter_config.rank,
            },
            reinit=True,
        )

    replay_strategy = None
    replay_ratio = 0.5
    ewc_lambda = None
    ewc_fisher_samples = None

    if condition_name == "naive":
        replay_strategy = NaiveReplay()
        replay_ratio = replay_config.naive_replay_ratio
    elif condition_name == "ewc":
        ewc_lambda = replay_config.ewc_lambda
        ewc_fisher_samples = replay_config.ewc_fisher_samples
    elif condition_name == "generative":
        replay_strategy = GenerativeReplay(temperature=replay_config.generative_temperature)
        replay_ratio = replay_config.generative_replay_ratio

    data_path = Path(config.paths.data_dir) / "synthetic" / "personal_facts.json"
    condition_output = output_dir / condition_name

    results = train_sequential(
        model=model,
        tokenizer=tokenizer,
        topics=topics,
        data_path=data_path,
        training_config=training_config,
        adapter_config=adapter_config,
        adapter_name="episodic",
        replay_strategy=replay_strategy,
        replay_ratio=replay_ratio,
        ewc_lambda=ewc_lambda,
        ewc_fisher_samples=ewc_fisher_samples,
        wandb_config=wandb_config if wandb_config.enabled else None,
        output_dir=condition_output,
    )

    # Log step-by-step recall to wandb
    if wandb_config.enabled:
        import wandb

        for r in results:
            log_data = {"step": r.step_index, "topic": r.topic_name}
            for topic_name, recall in r.recall_per_topic.items():
                log_data[f"recall/{topic_name}"] = recall
            wandb.log(log_data)
        wandb.finish()

    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Catastrophic forgetting & replay")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument(
        "--num-topics",
        type=int,
        default=2,
        choices=[2, 3, 4],
        help="Number of topics for sequential training (default: 2)",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=ALL_CONDITIONS,
        choices=ALL_CONDITIONS,
        help="Which conditions to run (default: all)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.no_wandb:
        config.wandb.enabled = False

    # Build topic list from config
    all_topics = [TopicDefinition(name=t.name, fact_ids=t.fact_ids) for t in config.replay.topics]
    topics = all_topics[: args.num_topics]
    logger.info(
        "Topics: %s",
        [(t.name, len(t.fact_ids)) for t in topics],
    )

    output_dir = Path(config.paths.output_dir) / "phase2"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load base model once
    logger.info("=== Loading base model ===")
    model, tokenizer = load_base_model(config.model)

    adapter_config = config.adapters["episodic"]

    # Run each condition
    all_results = {}
    all_forgetting = {}

    for condition in args.conditions:
        # Fresh adapter for each condition
        model = reset_adapter(model, adapter_config, "episodic")

        results = run_condition(
            condition_name=condition,
            model=model,
            tokenizer=tokenizer,
            topics=topics,
            config=config,
            output_dir=output_dir,
        )

        all_results[condition] = results
        all_forgetting[condition] = compute_forgetting_metrics(results)

    # Print comparison table
    print("\n" + "=" * 60)
    print("PHASE 2 RESULTS: Catastrophic Forgetting Comparison")
    print("=" * 60)
    print(format_results_table(all_forgetting))

    # Compute forgetting reduction vs. baseline
    if "baseline" in all_forgetting:
        print("\n--- Forgetting Reduction vs. Baseline ---")
        for condition in args.conditions:
            if condition == "baseline":
                continue
            if condition not in all_forgetting:
                continue
            reduction = compute_forgetting_reduction(
                all_forgetting["baseline"], all_forgetting[condition]
            )
            target_met = reduction["overall_reduction"] >= 0.5
            status = "PASS" if target_met else "FAIL"
            print(
                f"  {condition:>12}: {reduction['overall_reduction']:.1%} reduction "
                f"(baseline={reduction['baseline_mean_forgetting']:.1%}, "
                f"strategy={reduction['strategy_mean_forgetting']:.1%}) [{status}]"
            )
    print("=" * 60)

    # Save results
    results_path = output_dir / "results.json"
    serializable = {}
    for condition, metrics in all_forgetting.items():
        serializable[condition] = {topic: asdict(m) for topic, m in metrics.items()}
    if "baseline" in all_forgetting:
        serializable["_comparisons"] = {}
        for condition in args.conditions:
            if condition == "baseline" or condition not in all_forgetting:
                continue
            serializable["_comparisons"][condition] = compute_forgetting_reduction(
                all_forgetting["baseline"], all_forgetting[condition]
            )

    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info("Results saved to %s", results_path)

    return all_forgetting


if __name__ == "__main__":
    main()
