"""Elastic Weight Consolidation (EWC) for catastrophic forgetting mitigation."""

import logging
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import (
    Trainer,
    TrainingArguments,
    default_data_collator,
)

from paramem.utils.config import AdapterConfig, TrainingConfig, WandbConfig

logger = logging.getLogger(__name__)


def compute_fisher_information(
    model,
    dataset,
    adapter_name: str,
    num_samples: Optional[int] = None,
) -> dict[str, Tensor]:
    """Compute diagonal Fisher Information Matrix for adapter parameters.

    Runs forward passes on the dataset and accumulates squared gradients
    of the log-likelihood for each LoRA parameter.

    Args:
        model: PeftModel with the adapter to analyze.
        dataset: Training dataset used for this topic.
        adapter_name: Name of the active adapter.
        num_samples: If set, use only this many samples. None = all.

    Returns:
        Dict mapping parameter names to Fisher diagonal tensors.
    """
    model.set_adapter(adapter_name)
    model.eval()

    fisher = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher[name] = torch.zeros_like(param)

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    n_samples = 0

    for batch in loader:
        if num_samples is not None and n_samples >= num_samples:
            break

        batch = {k: v.to(model.device) for k, v in batch.items()}
        model.zero_grad()

        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher[name] += param.grad.detach() ** 2

        n_samples += 1

    # Average over samples
    for name in fisher:
        fisher[name] /= max(n_samples, 1)

    model.train()
    logger.info(
        "Computed Fisher information: %d parameters, %d samples",
        len(fisher),
        n_samples,
    )
    return fisher


def get_adapter_snapshot(model, adapter_name: str) -> dict[str, Tensor]:
    """Snapshot current adapter parameter values (theta*)."""
    model.set_adapter(adapter_name)
    snapshot = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            snapshot[name] = param.detach().clone()
    return snapshot


class EWCTrainer(Trainer):
    """HuggingFace Trainer with EWC regularization penalty.

    Adds a penalty term: ewc_lambda * sum(F_i * (theta - theta_i*)^2)
    to the standard cross-entropy loss.
    """

    def __init__(
        self,
        *args,
        fisher_dict: dict[str, Tensor],
        param_snapshot: dict[str, Tensor],
        ewc_lambda: float = 400.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.fisher_dict = fisher_dict
        self.param_snapshot = param_snapshot
        self.ewc_lambda = ewc_lambda

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        ce_loss = outputs.loss

        ewc_penalty = torch.tensor(0.0, device=ce_loss.device)
        for name, param in model.named_parameters():
            if name in self.fisher_dict and name in self.param_snapshot:
                fisher = self.fisher_dict[name].to(param.device)
                old_param = self.param_snapshot[name].to(param.device)
                ewc_penalty = ewc_penalty + (fisher * (param - old_param) ** 2).sum()

        loss = ce_loss + self.ewc_lambda * ewc_penalty

        if return_outputs:
            return loss, outputs
        return loss


def train_adapter_ewc(
    model,
    tokenizer,
    train_dataset,
    adapter_name: str,
    training_config: TrainingConfig,
    adapter_config: AdapterConfig,
    fisher_dict: dict[str, Tensor],
    param_snapshot: dict[str, Tensor],
    ewc_lambda: float = 400.0,
    wandb_config: Optional[WandbConfig] = None,
    output_dir: Optional[str | Path] = None,
    run_name: Optional[str] = None,
) -> dict:
    """Train a LoRA adapter with EWC regularization.

    Mirrors train_adapter() but uses EWCTrainer for the penalty term.
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
        save_strategy="no",
        report_to=report_to,
        run_name=run_name or f"paramem-{adapter_name}-ewc",
        seed=training_config.seed,
        bf16=True,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    if training_config.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    trainer = EWCTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=default_data_collator,
        fisher_dict=fisher_dict,
        param_snapshot=param_snapshot,
        ewc_lambda=ewc_lambda,
    )

    logger.info(
        "Starting EWC training: adapter=%s, epochs=%d, lambda=%.1f",
        adapter_name,
        training_config.num_epochs,
        ewc_lambda,
    )

    result = trainer.train()
    metrics = result.metrics

    model.save_pretrained(str(output_dir), selected_adapters=[adapter_name])
    tokenizer.save_pretrained(str(output_dir))

    logger.info("EWC training complete: %s", metrics)
    return metrics
