"""LoRA fine-tuning trainer for personal memory adapters."""

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from peft import PeftModel
from transformers import (
    PreTrainedTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    default_data_collator,
)

from paramem.training.thermal_throttle import ThermalPolicy, ThermalThrottleCallback
from paramem.utils.config import AdapterConfig, TrainingConfig, WandbConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingHooks:
    """Caller-supplied behaviour that ``train_adapter`` cannot derive from config.

    The callable fields are converted internally into a single HF
    ``TrainerCallback`` (``_HooksAdapterCallback``) that runs BEFORE the
    thermal throttle in the registered callback list, so inference yielding
    pre-empts throttle waits within the same step.

    All fields default to ``None`` — callers pass only the intents they need:

    - ``on_step_yield(global_step)``: invoked at every step boundary, used by
      ``BackgroundTrainer`` to yield to inference requests.
    - ``on_epoch_persist(epoch, output_dir)``: invoked at every epoch end,
      used by ``BackgroundTrainer`` to write ``resume_state.json``.
    - ``on_shutdown_check()``: invoked at every epoch end; returning ``True``
      sets ``control.should_training_stop``. Replaces ad-hoc
      ``GracefulShutdownCallback`` instantiation at the call site.
    - ``on_inference_active()``: polled by ``ThermalThrottleCallback`` to
      suppress throttling while a PA conversation is in progress (latency
      protection). Defaults to ``None`` (throttle never suppressed for
      non-server callers).
    """

    on_step_yield: Optional[Callable[[int], None]] = None
    on_epoch_persist: Optional[Callable[[int, str], None]] = None
    on_shutdown_check: Optional[Callable[[], bool]] = None
    on_inference_active: Optional[Callable[[], bool]] = None


class _HooksAdapterCallback(TrainerCallback):
    """Routes ``TrainingHooks`` intents to HF callback events.

    Registered before ``ThermalThrottleCallback`` so that inference yielding
    (``on_step_yield``) runs before any potential throttle wait at the same
    step. Resume-state persistence (``on_epoch_persist``) and shutdown checks
    (``on_shutdown_check``) fire at epoch boundaries.
    """

    def __init__(self, hooks: TrainingHooks):
        self._hooks = hooks

    def on_step_end(self, args, state, control, **kwargs):
        if self._hooks.on_step_yield is not None:
            self._hooks.on_step_yield(state.global_step)

    def on_epoch_end(self, args, state, control, **kwargs):
        if self._hooks.on_epoch_persist is not None:
            self._hooks.on_epoch_persist(int(state.epoch), args.output_dir)
        if self._hooks.on_shutdown_check is not None and self._hooks.on_shutdown_check():
            logger.info(
                "Graceful shutdown requested via TrainingHooks — stopping after epoch %d",
                int(state.epoch),
            )
            control.should_training_stop = True


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


class _FixedDecayTrainer(Trainer):
    """Trainer that decays LR over a fixed step count instead of the budget.

    HF's stock schedulers compute their decay window from
    ``num_training_steps`` (= ``len(dataloader) * num_train_epochs``). When
    ``lr_decay_steps`` is set, this subclass substitutes that value so the
    LR trajectory at any given step is invariant to ``num_train_epochs``.
    Steps past ``lr_decay_steps`` sit at the scheduler's tail (LR=0 for
    ``linear``); training continues so an early-stopping callback can
    decide when to halt.
    """

    def __init__(self, *args, lr_decay_steps: Optional[int] = None, **kwargs):
        self._lr_decay_steps = lr_decay_steps
        super().__init__(*args, **kwargs)

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        if self._lr_decay_steps is not None:
            num_training_steps = self._lr_decay_steps
        return super().create_scheduler(num_training_steps, optimizer)


class GracefulShutdownCallback(TrainerCallback):
    """Stop training cleanly when a shutdown flag is set.

    Checks the flag at each epoch boundary. When set, the trainer
    finishes the current step, saves state, and exits the training loop.
    """

    def __init__(self, shutdown_flag: callable):
        """Args: shutdown_flag — callable returning True when shutdown requested."""
        self._should_stop = shutdown_flag

    def on_epoch_end(self, args, state, control, **kwargs):
        if self._should_stop():
            logger.info(
                "Graceful shutdown requested — stopping after epoch %d",
                int(state.epoch),
            )
            control.should_training_stop = True


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
    active_adapters: Optional[list[str]] = None,
    resume_from_checkpoint: Optional[str | Path] = None,
    thermal_policy: Optional[ThermalPolicy] = None,
    hooks: Optional[TrainingHooks] = None,
) -> dict:
    """Train a LoRA adapter on the given dataset.

    If active_adapters is provided, all listed adapters are set active in the
    forward pass (for chained/compositional training). Only adapter_name
    receives gradients — caller must freeze others via set_requires_grad.

    Args:
        resume_from_checkpoint: Optional path to an HF Trainer checkpoint
            directory (e.g. ``adapter/checkpoint-120``). When provided, HF
            Trainer restores optimizer state, LR schedule, and step count from
            that checkpoint so training continues without resetting Adam
            momentum. When ``None`` (default), a fresh training run starts.
            Mirrors the semantics of
            ``paramem.server.background_trainer.BackgroundTrainer._train_adapter``
            which already uses ``trainer.train(resume_from_checkpoint=...)``.
        thermal_policy: When set, installs a ``ThermalThrottleCallback`` that
            pauses training when GPU temperature exceeds the policy limit
            during the configured quiet-hours window. Default ``None`` skips
            the install — experiments and tests get fast unthrottled runs by
            construction; only callers wired to ``ConsolidationConfig`` with
            a positive ``training_temp_limit`` opt in.
        hooks: When set, wraps caller intents (inference yielding,
            resume-state persistence, shutdown predicate) into a single HF
            callback that runs before the thermal throttle so yielding
            pre-empts throttle waits.

    Returns:
        Training metrics dict (same as before; ``resume_from_checkpoint`` does
        not change the shape of the returned dict).
    """
    if output_dir is None:
        output_dir = Path("outputs") / "adapters" / adapter_name
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if active_adapters is not None:
        # Activate all adapters in forward pass, frozen by default
        model.base_model.set_adapter(active_adapters, inference_mode=True)
        # Unfreeze only the adapter being trained
        model.set_requires_grad(adapter_name, requires_grad=True)
        # Verify gradients are live — silent failure here would invalidate results
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError(
                f"No trainable parameters after activating {active_adapters} "
                f"and unfreezing '{adapter_name}'"
            )
        logger.info(
            "Compose training: %d adapters active, %d trainable params in '%s'",
            len(active_adapters),
            sum(p.numel() for p in model.parameters() if p.requires_grad),
            adapter_name,
        )
    else:
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
        warmup_steps=training_config.warmup_steps if training_config.warmup_steps > 0 else 0,
        warmup_ratio=training_config.warmup_ratio if training_config.warmup_steps == 0 else 0.0,
        lr_scheduler_type=training_config.lr_scheduler_type,
        weight_decay=training_config.weight_decay,
        max_grad_norm=training_config.max_grad_norm,
        gradient_checkpointing=training_config.gradient_checkpointing,
        logging_steps=training_config.logging_steps,
        save_strategy=training_config.save_strategy,
        save_total_limit=training_config.save_total_limit,
        report_to=report_to,
        run_name=run_name or f"paramem-{adapter_name}",
        seed=training_config.seed,
        bf16=True,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    if training_config.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    from paramem.training.encrypted_checkpoint_callback import EncryptCheckpointCallback

    # EncryptCheckpointCallback wraps every HF-written ``checkpoint-<step>/``
    # file in the age envelope on ``on_save``.  Without it, HF Trainer leaves
    # plaintext ``adapter_model.safetensors`` files inside ``args.output_dir``
    # — which the consolidation flow places under ``data/ha/adapters/`` —
    # and the next server boot's mode-consistency check (which expects
    # every infra file to be encrypted-or-plaintext consistently with the
    # rest) fires and refuses startup.  No-op when Security is OFF.  Same
    # callback :class:`BackgroundTrainer` already uses for its own HF
    # Trainer; sharing it keeps both code paths posture-consistent.
    # Callback assembly order is load-bearing (HF iterates registrations in
    # order at every event). _HooksAdapterCallback must run BEFORE
    # ThermalThrottleCallback so on_step_yield (inference yielding) pre-empts
    # throttle waits within the same step. callbacks_extra (call-bound, e.g.
    # recall probe) trail.
    callbacks: list = [EncryptCheckpointCallback()]
    if training_config.early_stopping:
        callbacks.append(
            LossEarlyStoppingCallback(
                loss_threshold=training_config.early_stopping_threshold,
                epoch_floor=training_config.early_stopping_floor,
                patience=training_config.early_stopping_patience,
            )
        )
    if hooks is not None:
        callbacks.append(_HooksAdapterCallback(hooks))
    if thermal_policy is not None:
        # When hooks expose a shutdown predicate, route it into the throttle
        # so a mid-wait shutdown breaks the loop cleanly. Otherwise the
        # throttle's default constant-False shutdown_fn applies.
        shutdown_fn = (
            hooks.on_shutdown_check
            if hooks is not None and hooks.on_shutdown_check is not None
            else (lambda: False)
        )
        inference_active_fn = (
            hooks.on_inference_active
            if hooks is not None and hooks.on_inference_active is not None
            else (lambda: False)
        )
        callbacks.append(
            ThermalThrottleCallback(
                thermal_policy,
                shutdown_fn=shutdown_fn,
                inference_active_fn=inference_active_fn,
            )
        )
    if callbacks_extra:
        callbacks.extend(callbacks_extra)

    trainer_cls = _FixedDecayTrainer if training_config.lr_decay_steps is not None else Trainer
    trainer_kwargs = {}
    if training_config.lr_decay_steps is not None:
        trainer_kwargs["lr_decay_steps"] = training_config.lr_decay_steps
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
        callbacks=callbacks,
        **trainer_kwargs,
    )

    logger.info(
        "Starting training: adapter=%s, epochs=%d, lr=%e",
        adapter_name,
        training_config.num_epochs,
        adapter_config.learning_rate,
    )

    ckpt_arg = str(resume_from_checkpoint) if resume_from_checkpoint is not None else None

    # When Security is ON, ``EncryptCheckpointCallback`` wrote every file in
    # the on-disk ``checkpoint-N/`` tree as an age envelope.  HF Trainer's
    # ``_load_from_checkpoint`` reads safetensors directly via
    # ``safe_load_file`` and crashes on the age magic with
    # ``SafetensorError: header too large``.  Mirror the symmetric pattern
    # already in ``BackgroundTrainer._train_adapter`` (background_trainer.py
    # ~L1040): materialize the checkpoint into a ``/dev/shm`` tempdir,
    # decrypting age envelopes en route, hand HF the plaintext path, then
    # remove the tempdir in ``finally``.  No-op when Security is OFF (the
    # daily age identity isn't loadable) or when no resume path was passed.
    shm_resume_dir: Path | None = None
    effective_ckpt_arg = ckpt_arg
    if ckpt_arg is not None:
        from paramem.backup import key_store as _ks

        if _ks.daily_identity_loadable(_ks.DAILY_KEY_PATH_DEFAULT):
            from paramem.backup.checkpoint_shard import materialize_checkpoint_to_shm

            shm_resume_dir = materialize_checkpoint_to_shm(Path(ckpt_arg))
            effective_ckpt_arg = str(shm_resume_dir)
            logger.info(
                "Materialized encrypted checkpoint %s → %s for HF Trainer load",
                ckpt_arg,
                shm_resume_dir,
            )

    try:
        result = trainer.train(resume_from_checkpoint=effective_ckpt_arg)
    finally:
        if shm_resume_dir is not None:
            shutil.rmtree(shm_resume_dir, ignore_errors=True)
    metrics = result.metrics

    # No final save here.  ``train_adapter`` is responsible only for training;
    # the canonical encrypted slot-dir save is the orchestrator's job
    # (``ConsolidationLoop._save_adapters`` → :func:`atomic_save_adapter` →
    # :func:`_encrypt_adapter_safetensors`).  Writing here would duplicate
    # the canonical save AND leave a plaintext ``adapter_model.safetensors``
    # inside ``data/ha/adapters/`` which trips the next boot's
    # mode-consistency check.

    logger.info("Training complete: %s", metrics)
    return metrics
