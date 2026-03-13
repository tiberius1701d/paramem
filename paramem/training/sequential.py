"""Sequential training orchestrator for catastrophic forgetting experiments."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from paramem.evaluation.recall import evaluate_recall
from paramem.training.dataset import PersonalFactsDataset
from paramem.training.ewc import (
    compute_fisher_information,
    get_adapter_snapshot,
    train_adapter_ewc,
)
from paramem.training.replay import MixedReplayDataset, ReplayStrategy
from paramem.training.trainer import train_adapter
from paramem.utils.config import AdapterConfig, TrainingConfig, WandbConfig

logger = logging.getLogger(__name__)


@dataclass
class TopicDefinition:
    name: str
    fact_ids: list[str]


@dataclass
class SequentialResult:
    topic_name: str
    step_index: int
    training_metrics: dict
    recall_per_topic: dict[str, float] = field(default_factory=dict)


def train_sequential(
    model,
    tokenizer,
    topics: list[TopicDefinition],
    data_path: str | Path,
    training_config: TrainingConfig,
    adapter_config: AdapterConfig,
    adapter_name: str = "episodic",
    replay_strategy: Optional[ReplayStrategy] = None,
    replay_ratio: float = 0.5,
    ewc_lambda: Optional[float] = None,
    ewc_fisher_samples: Optional[int] = None,
    wandb_config: Optional[WandbConfig] = None,
    output_dir: Optional[str | Path] = None,
) -> list[SequentialResult]:
    """Train adapter sequentially on topics, measuring recall after each step.

    Args:
        model: PeftModel with the adapter already created.
        tokenizer: Tokenizer for the model.
        topics: Ordered list of topics to train on.
        data_path: Path to the facts JSON file.
        training_config: Training hyperparameters (num_epochs is per-topic).
        adapter_config: Adapter hyperparameters.
        adapter_name: Name of the adapter to train.
        replay_strategy: If set, use this strategy to mix old examples.
        replay_ratio: Fraction of replay examples in mixed dataset.
        ewc_lambda: If set, use EWC with this penalty strength.
        ewc_fisher_samples: Number of samples for Fisher computation.
        wandb_config: wandb configuration.
        output_dir: Base output directory.

    Returns:
        List of SequentialResult, one per training step.
    """
    if output_dir is None:
        output_dir = Path("outputs") / "phase2"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    all_trained_fact_ids = []

    # EWC state: accumulated Fisher and snapshots
    accumulated_fisher = {}
    accumulated_snapshot = {}

    for step_idx, topic in enumerate(topics):
        logger.info(
            "=== Step %d/%d: Training on topic '%s' (%d facts) ===",
            step_idx + 1,
            len(topics),
            topic.name,
            len(topic.fact_ids),
        )

        # Build dataset for current topic
        new_dataset = PersonalFactsDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            max_length=training_config.max_seq_length,
            fact_ids=topic.fact_ids,
        )

        train_dataset = new_dataset
        step_output_dir = output_dir / f"step_{step_idx}_{topic.name}"

        # Apply replay strategy if we have previous topics
        if step_idx > 0 and replay_strategy is not None:
            # Disable gradient checkpointing for generative replay (needs generate())
            model.gradient_checkpointing_disable()
            replay_dataset = replay_strategy.prepare(
                model=model,
                tokenizer=tokenizer,
                old_fact_ids=all_trained_fact_ids.copy(),
                data_path=data_path,
                max_length=training_config.max_seq_length,
            )
            if training_config.gradient_checkpointing:
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            train_dataset = MixedReplayDataset(
                new_dataset=new_dataset,
                replay_dataset=replay_dataset,
                replay_ratio=replay_ratio,
            )
            logger.info(
                "Mixed dataset: %d new + %d replay = %d total",
                len(new_dataset),
                len(replay_dataset),
                len(train_dataset),
            )

        # Train with or without EWC
        use_ewc = step_idx > 0 and ewc_lambda is not None and accumulated_fisher
        if use_ewc:
            metrics = train_adapter_ewc(
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                adapter_name=adapter_name,
                training_config=training_config,
                adapter_config=adapter_config,
                fisher_dict=accumulated_fisher,
                param_snapshot=accumulated_snapshot,
                ewc_lambda=ewc_lambda,
                wandb_config=wandb_config,
                output_dir=step_output_dir,
                run_name=f"phase2-ewc-step{step_idx}-{topic.name}",
            )
        else:
            metrics = train_adapter(
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                adapter_name=adapter_name,
                training_config=training_config,
                adapter_config=adapter_config,
                wandb_config=wandb_config,
                output_dir=step_output_dir,
                run_name=f"phase2-step{step_idx}-{topic.name}",
            )

        # Update EWC state after training on this topic
        if ewc_lambda is not None:
            topic_dataset = PersonalFactsDataset(
                data_path=data_path,
                tokenizer=tokenizer,
                max_length=training_config.max_seq_length,
                fact_ids=topic.fact_ids,
            )
            fisher = compute_fisher_information(
                model, topic_dataset, adapter_name, num_samples=ewc_fisher_samples
            )
            snapshot = get_adapter_snapshot(model, adapter_name)

            # Accumulate: sum Fisher matrices, keep latest snapshot per param
            for name in fisher:
                if name in accumulated_fisher:
                    accumulated_fisher[name] = accumulated_fisher[name] + fisher[name]
                else:
                    accumulated_fisher[name] = fisher[name]
            accumulated_snapshot = snapshot

        # Evaluate recall on all topics trained so far + current
        # Disable gradient checkpointing for generation — it forces use_cache=False
        # which degrades model.generate() quality
        model.gradient_checkpointing_disable()

        all_trained_fact_ids.extend(topic.fact_ids)
        recall_per_topic = {}

        for eval_topic in topics[: step_idx + 1]:
            eval_result = evaluate_recall(
                model=model,
                tokenizer=tokenizer,
                data_path=str(data_path),
                fact_ids=eval_topic.fact_ids,
                adapter_name=adapter_name,
            )
            recall_per_topic[eval_topic.name] = eval_result["mean_recall"]

        # Re-enable gradient checkpointing for next training step
        if training_config.gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        logger.info("Recall after step %d: %s", step_idx, recall_per_topic)

        results.append(
            SequentialResult(
                topic_name=topic.name,
                step_index=step_idx,
                training_metrics=metrics,
                recall_per_topic=recall_per_topic,
            )
        )

    return results
