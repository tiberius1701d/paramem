"""Background training manager for continuous adapter learning.

Trains adapters during idle time, pausing for inference requests.
Saves full training state at each epoch boundary for crash recovery.

The model stays loaded — switching between training and inference is
just flag flips (model.eval/train, gradient checkpointing), no reload.
"""

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from transformers import (
    Trainer,
    TrainerCallback,
    TrainingArguments,
    default_data_collator,
)

from paramem.training.indexed_memory import format_indexed_training
from paramem.utils.config import AdapterConfig, TrainingConfig

logger = logging.getLogger(__name__)


@dataclass
class TrainingJob:
    """A single adapter training job."""

    keyed_pairs: list[dict]
    adapter_name: str
    adapter_config: AdapterConfig


class _PauseForInferenceCallback(TrainerCallback):
    """Yields to inference between training steps."""

    def __init__(self, trainer_ref: "BackgroundTrainer"):
        self._trainer = trainer_ref

    def on_step_end(self, args, state, control, **kwargs):
        if self._trainer._inference_requested.is_set():
            logger.debug("Training paused for inference at step %d", state.global_step)
            self._trainer._training_paused.set()
            self._trainer._inference_done.wait()
            self._trainer._inference_done.clear()
            self._trainer._training_paused.clear()
            logger.debug("Training resumed at step %d", state.global_step)

    def on_epoch_end(self, args, state, control, **kwargs):
        self._trainer._last_completed_epoch = int(state.epoch)
        if self._trainer._shutdown_requested:
            logger.info("Shutdown requested — stopping after epoch %d", int(state.epoch))
            control.should_training_stop = True


class BackgroundTrainer:
    """Manages background training that yields to inference.

    Supports a queue of training jobs (e.g., episodic then procedural).
    Each job trains one adapter. Jobs run sequentially in a single thread.

    Usage:
        bt = BackgroundTrainer(model, tokenizer, config)
        bt.start_jobs([job1, job2], on_complete=save_callback)

        # When inference is needed:
        bt.pause()       # waits for current step to finish
        # ... run inference ...
        bt.resume()      # lets training continue

        bt.stop()        # graceful stop at next epoch boundary
    """

    def __init__(
        self,
        model,
        tokenizer,
        training_config: TrainingConfig,
        output_dir: str | Path = "data/ha/adapters",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.training_config = training_config
        self.output_dir = Path(output_dir)

        self._inference_requested = threading.Event()
        self._inference_done = threading.Event()
        self._training_paused = threading.Event()
        self._shutdown_requested = False
        self._training_thread: threading.Thread | None = None
        self._is_training = False
        self._last_completed_epoch = 0
        self._current_adapter = ""
        self._on_error: Callable[[], None] | None = None

    @property
    def is_training(self) -> bool:
        return self._is_training

    def start_jobs(
        self,
        jobs: list[TrainingJob],
        on_complete: Callable[[], None] | None = None,
        on_error: Callable[[], None] | None = None,
    ):
        """Start training a queue of adapter jobs.

        Non-blocking — launches a thread and returns immediately.
        on_complete is called after all jobs finish successfully.
        on_error is called if training raises an exception.
        """
        if self._is_training:
            logger.warning("Training already in progress")
            return

        if not jobs:
            logger.info("No training jobs to run")
            if on_complete:
                on_complete()
            return

        self._shutdown_requested = False
        self._is_training = True
        self._on_error = on_error

        self._training_thread = threading.Thread(
            target=self._run_jobs,
            args=(jobs, on_complete),
            daemon=True,
            name="bg-trainer",
        )
        self._training_thread.start()
        logger.info("Background training started: %d jobs", len(jobs))

    def pause(self, timeout: float = 5.0) -> bool:
        """Pause training for inference. Blocks until current step finishes.

        Switches model to eval mode and disables gradient checkpointing
        so inference produces correct output.
        """
        if not self._is_training:
            return True

        self._inference_requested.set()
        paused = self._training_paused.wait(timeout=timeout)
        if not paused:
            logger.warning("Training did not pause within %.1fs", timeout)
            return False

        # Switch model to inference mode
        self.model.eval()
        if self.training_config.gradient_checkpointing:
            self.model.gradient_checkpointing_disable()
        return True

    def resume(self):
        """Resume training after inference.

        Restores model to training mode with gradient checkpointing.
        No-op if training is not active (e.g. after stop()).
        """
        if not self._is_training:
            return
        # Restore training mode
        self.model.train()
        if self.training_config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        self._inference_requested.clear()
        self._inference_done.set()

    def stop(self, timeout: float = 60.0) -> int:
        """Stop training gracefully. Returns last completed epoch."""
        if not self._is_training:
            return self._last_completed_epoch

        self._shutdown_requested = True
        # Unblock if paused waiting for inference
        self._inference_done.set()
        if self._training_thread is not None:
            self._training_thread.join(timeout=timeout)
            if self._training_thread.is_alive():
                logger.warning("Training thread did not stop within %.1fs", timeout)

        self._is_training = False
        return self._last_completed_epoch

    def _run_jobs(
        self,
        jobs: list[TrainingJob],
        on_complete: Callable[[], None] | None,
    ):
        """Run training jobs sequentially in background thread."""
        try:
            all_complete = True
            for i, job in enumerate(jobs):
                if self._shutdown_requested:
                    logger.info("Shutdown — skipping remaining jobs")
                    all_complete = False
                    break

                logger.info(
                    "Training job %d/%d: %s (%d keys)",
                    i + 1,
                    len(jobs),
                    job.adapter_name,
                    len(job.keyed_pairs),
                )
                self._train_adapter(job)

            if all_complete and on_complete:
                on_complete()

        except Exception:
            logger.exception("Background training failed")
            if self._on_error:
                self._on_error()
        finally:
            self._is_training = False

    def _train_adapter(self, job: TrainingJob):
        """Train a single adapter."""
        from paramem.models.loader import switch_adapter

        self._current_adapter = job.adapter_name
        self._last_completed_epoch = 0

        switch_adapter(self.model, job.adapter_name)
        self.model.train()

        examples = format_indexed_training(job.keyed_pairs, self.tokenizer, max_length=1024)
        dataset = _SimpleDataset(examples)

        if not examples:
            logger.info("No training examples for %s, skipping", job.adapter_name)
            return

        checkpoint_dir = self.output_dir / job.adapter_name / "bg_checkpoint"

        training_args = TrainingArguments(
            output_dir=str(checkpoint_dir),
            num_train_epochs=self.training_config.num_epochs,
            per_device_train_batch_size=self.training_config.batch_size,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            learning_rate=job.adapter_config.learning_rate,
            warmup_steps=self.training_config.warmup_steps
            if self.training_config.warmup_steps > 0
            else 0,
            warmup_ratio=self.training_config.warmup_ratio
            if self.training_config.warmup_steps == 0
            else 0.0,
            lr_scheduler_type=self.training_config.lr_scheduler_type,
            weight_decay=self.training_config.weight_decay,
            max_grad_norm=self.training_config.max_grad_norm,
            gradient_checkpointing=self.training_config.gradient_checkpointing,
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=2,
            report_to="none",
            run_name=f"bg-{job.adapter_name}",
            seed=self.training_config.seed,
            bf16=True,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
        )

        if self.training_config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=default_data_collator,
            callbacks=[_PauseForInferenceCallback(self)],
        )

        logger.info(
            "Training %s: %d examples, %d epochs",
            job.adapter_name,
            len(examples),
            self.training_config.num_epochs,
        )

        trainer.train()

        self._last_completed_epoch = self.training_config.num_epochs
        logger.info("Training complete: %s", job.adapter_name)


class _SimpleDataset:
    """Minimal dataset wrapper for pre-tokenized examples."""

    def __init__(self, examples: list[dict]):
        self._examples = examples

    def __len__(self):
        return len(self._examples)

    def __getitem__(self, idx):
        return self._examples[idx]
