"""LoRA fine-tuning trainer for personal memory adapters."""

import logging
from pathlib import Path
from typing import Optional

from peft import PeftModel
from transformers import (
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

from paramem.utils.config import AdapterConfig, TrainingConfig, WandbConfig

logger = logging.getLogger(__name__)


def train_adapter(
    model: PeftModel,
    tokenizer: PreTrainedTokenizer,
    train_dataset,
    adapter_name: str,
    training_config: TrainingConfig,
    adapter_config: AdapterConfig,
    wandb_config: Optional[WandbConfig] = None,
    output_dir: Optional[str | Path] = None,
    eval_dataset=None,
    run_name: Optional[str] = None,
) -> dict:
    """Train a LoRA adapter on the given dataset.

    Returns training metrics dict.
    """
    if output_dir is None:
        output_dir = Path("outputs") / "adapters" / adapter_name
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.set_adapter(adapter_name)

    report_to = "none"
    if wandb_config and wandb_config.enabled:
        report_to = "wandb"

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=training_config.num_epochs,
        per_device_train_batch_size=training_config.batch_size,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        learning_rate=adapter_config.learning_rate,
        warmup_ratio=training_config.warmup_ratio,
        weight_decay=training_config.weight_decay,
        max_grad_norm=training_config.max_grad_norm,
        gradient_checkpointing=training_config.gradient_checkpointing,
        logging_steps=1,
        save_strategy="epoch",
        report_to=report_to,
        run_name=run_name or f"paramem-{adapter_name}",
        seed=training_config.seed,
        bf16=True,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    if training_config.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )

    logger.info(
        "Starting training: adapter=%s, epochs=%d, lr=%e",
        adapter_name,
        training_config.num_epochs,
        adapter_config.learning_rate,
    )

    result = trainer.train()
    metrics = result.metrics

    model.save_pretrained(str(output_dir), selected_adapters=[adapter_name])
    tokenizer.save_pretrained(str(output_dir))

    logger.info("Training complete: %s", metrics)
    return metrics
