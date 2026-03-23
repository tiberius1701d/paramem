"""LoRA fine-tuning trainer for personal memory adapters."""

import logging
from pathlib import Path
from typing import Optional

from peft import PeftModel
from transformers import (
    PreTrainedTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    default_data_collator,
)

from paramem.utils.config import AdapterConfig, TrainingConfig, WandbConfig

logger = logging.getLogger(__name__)


class LossEarlyStoppingCallback(TrainerCallback):
    """Stop training when epoch-average loss drops below threshold.

    Accumulates per-step losses and checks the average at each epoch
    boundary (after the floor). Stops when the epoch-average loss
    stays below the threshold for `patience` consecutive epochs.
    """

    def __init__(
        self,
        loss_threshold: float = 0.01,
        epoch_floor: int = 10,
        patience: int = 2,
    ):
        self.loss_threshold = loss_threshold
        self.epoch_floor = epoch_floor
        self.patience = patience
        self._below_count = 0
        self._last_epoch = 0
        self._epoch_losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        epoch = logs.get("epoch", 0)
        loss = logs.get("loss")
        if loss is None:
            return

        current_epoch = int(epoch)

        # Accumulate losses for the current epoch
        self._epoch_losses.append(loss)

        # Check at epoch boundaries
        if current_epoch <= self._last_epoch:
            return
        self._last_epoch = current_epoch

        if epoch < self.epoch_floor:
            self._epoch_losses = []
            return

        # Compute epoch average from accumulated steps
        avg_loss = sum(self._epoch_losses) / len(self._epoch_losses)
        self._epoch_losses = []

        if avg_loss < self.loss_threshold:
            self._below_count += 1
            if self._below_count >= self.patience:
                logger.info(
                    "Early stopping: avg_loss=%.6f < %.4f for %d consecutive epochs at epoch %d",
                    avg_loss,
                    self.loss_threshold,
                    self.patience,
                    current_epoch,
                )
                control.should_training_stop = True
        else:
            self._below_count = 0


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
    callbacks_extra: Optional[list] = None,
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
        save_total_limit=2,
        report_to=report_to,
        run_name=run_name or f"paramem-{adapter_name}",
        seed=training_config.seed,
        bf16=True,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    if training_config.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    callbacks = []
    if training_config.early_stopping:
        callbacks.append(
            LossEarlyStoppingCallback(
                loss_threshold=training_config.early_stopping_threshold,
                epoch_floor=training_config.early_stopping_floor,
                patience=training_config.early_stopping_patience,
            )
        )
    if callbacks_extra:
        callbacks.extend(callbacks_extra)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
        callbacks=callbacks,
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
