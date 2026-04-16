"""Background training manager for continuous adapter learning.

Trains adapters during idle time, pausing for inference requests.
Saves full training state at each epoch boundary for crash recovery.

The model stays loaded — switching between training and inference is
just flag flips (model.eval/train, gradient checkpointing), no reload.
"""

import logging
import subprocess
import threading
import time
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


def _gpu_temp() -> int | None:
    """Read current GPU temperature via nvidia-smi. Returns None on failure."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return int(result.stdout.strip())
    except Exception:
        pass
    return None


@dataclass
class TrainingJob:
    """A single adapter training job."""

    keyed_pairs: list[dict]
    adapter_name: str
    adapter_config: AdapterConfig


class _PauseForInferenceCallback(TrainerCallback):
    """Yields to inference between training steps.

    Releases the GPU lock while paused so STT/TTS/inference can proceed,
    then re-acquires before resuming training. Uses a bounded wait to
    prevent permanent stalls if the inference caller crashes.
    """

    _INFERENCE_TIMEOUT = 120.0  # max seconds to wait for inference to finish

    def __init__(self, trainer_ref: "BackgroundTrainer"):
        self._trainer = trainer_ref

    def on_step_end(self, args, state, control, **kwargs):
        # Yield to inference if requested
        if self._trainer._inference_requested.is_set():
            from paramem.models.loader import switch_adapter
            from paramem.server.gpu_lock import acquire_gpu, release_gpu

            logger.debug("Training paused for inference at step %d", state.global_step)
            release_gpu()
            self._trainer._training_paused.set()
            signalled = self._trainer._inference_done.wait(timeout=self._INFERENCE_TIMEOUT)
            if not signalled:
                logger.warning(
                    "Inference did not signal done within %.0fs — resuming training",
                    self._INFERENCE_TIMEOUT,
                )
            self._trainer._inference_done.clear()
            acquire_gpu()
            # Restore staging adapter as active — inference may have switched
            # to a production adapter. Training must resume on in_training.
            switch_adapter(self._trainer.model, "in_training")
            self._trainer._training_paused.clear()
            logger.debug("Training resumed at step %d (adapter=in_training)", state.global_step)

        # Thermal throttle — check every N steps
        self._trainer._thermal_throttle(state.global_step)

    def on_epoch_end(self, args, state, control, **kwargs):
        self._trainer._last_completed_epoch = int(state.epoch)
        # Also check thermal throttle at epoch boundaries
        self._trainer._thermal_throttle(state.global_step)
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
        temp_limit: int = 0,
        temp_check_interval: int = 5,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.training_config = training_config
        self.output_dir = Path(output_dir)
        self._temp_limit = temp_limit  # 0 = disabled
        self._temp_check_interval = temp_check_interval

        self._inference_requested = threading.Event()
        self._inference_done = threading.Event()
        self._training_paused = threading.Event()
        # Serializes pause/resume cycles. A second pause() waits until the
        # first has fully completed (callback re-acquired lock, restored
        # staging, cleared _training_paused) to prevent the re-entrancy race
        # where two threads manipulate model state concurrently.
        self._pause_lock = threading.Lock()
        # State guard for _pause_active flag. Ensures exactly-once transition
        # from True→False across concurrent resume() calls.
        self._pause_state_lock = threading.Lock()
        self._pause_active = False
        self._shutdown_requested = False
        self._training_thread: threading.Thread | None = None
        self._is_training = False
        self._last_completed_epoch = 0
        self._current_adapter = ""
        self._on_error: Callable[[], None] | None = None

    @property
    def is_training(self) -> bool:
        return self._is_training

    @property
    def current_adapter_name(self) -> str:
        """Name of the adapter currently being trained (empty if idle)."""
        return self._current_adapter

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

        Switches the active adapter to the production slot (last-known-good
        weights, NOT the mid-training in_training slot) so inference returns
        correct answers. Also switches model to eval mode and disables
        gradient checkpointing.

        Serialized via _pause_lock so concurrent callers queue instead of
        racing on model state.
        """
        if not self._is_training:
            return True

        self._pause_lock.acquire()
        success = False
        try:
            # Double-check after acquiring: training may have ended while we waited
            if not self._is_training:
                return True

            self._inference_requested.set()
            paused = self._training_paused.wait(timeout=timeout)
            if not paused:
                logger.warning("Training did not pause within %.1fs", timeout)
                # The callback may have seen _inference_requested before we
                # timed out and is now waiting on _inference_done. Clear the
                # request AND signal done so it doesn't block the full 120s
                # timeout. Also consume any _training_paused set by a racing
                # callback so the next pause() starts from a clean state.
                self._inference_requested.clear()
                self._inference_done.set()
                self._training_paused.clear()
                return False

            # Switch to production adapter (last committed snapshot) so inference
            # queries the user's established knowledge, not mid-training weights.
            # Guard: _current_adapter is set by _train_adapter; if pause() races
            # before the first job starts, fall back to a sane production slot.
            from paramem.models.loader import switch_adapter

            target = self._current_adapter or "episodic"
            if target in self.model.peft_config:
                switch_adapter(self.model, target)
                logger.debug("Paused: switched to production adapter '%s'", target)
            else:
                logger.warning(
                    "Paused but no production adapter available to switch to (target=%s)", target
                )

            # Switch model to inference mode
            self.model.eval()
            if self.training_config.gradient_checkpointing:
                self.model.gradient_checkpointing_disable()

            # Lock is kept held — resume() releases it. Matched pair.
            with self._pause_state_lock:
                self._pause_active = True
            success = True
            return True
        finally:
            if not success:
                # pause() did not fully succeed — release the lock so other
                # callers can proceed. resume() will be a no-op.
                self._pause_lock.release()

    def resume(self):
        """Resume training after inference.

        Restores model to training mode with gradient checkpointing.
        No-op if training is not active or no pause is active.

        Must be paired with a successful pause() — releases _pause_lock.
        Safe to call multiple times concurrently: only one call releases
        the lock (compare-and-swap on _pause_active under _pause_state_lock).
        """
        # Atomic compare-and-swap: only the caller that transitions
        # _pause_active from True→False proceeds to release the lock.
        with self._pause_state_lock:
            if not self._pause_active:
                return
            self._pause_active = False
        try:
            if self._is_training:
                # Restore training mode
                self.model.train()
                if self.training_config.gradient_checkpointing:
                    self.model.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False}
                    )
                self._inference_requested.clear()
                self._inference_done.set()
        finally:
            self._pause_lock.release()

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

    def _thermal_throttle(self, global_step: int):
        """Pause training if GPU temperature exceeds the configured limit.

        Keeps the machine quiet during background training — fans stay off
        when GPU temp stays below the ramp-up threshold. Training pauses
        until the GPU cools back down, then resumes automatically.

        Controlled by training_temp_limit in server.yaml (0 = disabled).
        """
        if self._temp_limit <= 0:
            return
        if global_step % self._temp_check_interval != 0:
            return

        temp = _gpu_temp()
        if temp is None or temp <= self._temp_limit:
            return

        from paramem.server.gpu_lock import acquire_gpu, release_gpu

        logger.info(
            "Thermal throttle: GPU at %d°C (limit %d°C) — releasing GPU at step %d",
            temp,
            self._temp_limit,
            global_step,
        )
        release_gpu()

        while temp is not None and temp > self._temp_limit:
            if self._shutdown_requested:
                acquire_gpu()
                return
            time.sleep(5)
            temp = _gpu_temp()

        acquire_gpu()
        logger.info("Thermal throttle: GPU cooled to %d°C — resuming training", temp)

    def _run_jobs(
        self,
        jobs: list[TrainingJob],
        on_complete: Callable[[], None] | None,
    ):
        """Run training jobs sequentially in background thread.

        Holds the GPU lock during training to prevent concurrent CUDA access.
        Releases the lock between jobs for cooldown and to let STT/TTS through.
        """
        from paramem.server.gpu_lock import gpu_lock_sync

        try:
            all_complete = True
            for i, job in enumerate(jobs):
                if self._shutdown_requested:
                    logger.info("Shutdown — skipping remaining jobs")
                    all_complete = False
                    break

                # Cooldown between jobs (not before the first one)
                if i > 0:
                    self._cooldown_between_jobs()

                logger.info(
                    "Training job %d/%d: %s (%d keys)",
                    i + 1,
                    len(jobs),
                    job.adapter_name,
                    len(job.keyed_pairs),
                )
                with gpu_lock_sync():
                    self._train_adapter(job)

            if all_complete and on_complete:
                on_complete()

        except Exception:
            logger.exception("Background training failed")
            if self._on_error:
                self._on_error()
        finally:
            self._is_training = False

    def _cooldown_between_jobs(self):
        """Wait for GPU to cool between training jobs."""
        threshold = self._temp_limit if self._temp_limit > 0 else 52
        temp = _gpu_temp()
        if temp is None:
            logger.warning("Could not check GPU temperature — skipping cooldown")
            return
        if temp <= threshold:
            logger.info("GPU at %d°C — no cooldown needed", temp)
            return
        logger.info("GPU at %d°C — waiting for cooldown (target ≤%d°C)", temp, threshold)
        for _ in range(30):
            if self._shutdown_requested:
                break
            time.sleep(10)
            temp = _gpu_temp()
            if temp is not None and temp <= threshold:
                logger.info("GPU cooled to %d°C — resuming", temp)
                return
        logger.warning("Cooldown timeout — proceeding at %s°C", temp)

    def _train_adapter(self, job: TrainingJob):
        """Train a single adapter via the in_training staging slot.

        Flow:
            1. Copy production adapter weights into in_training.
            2. Activate in_training (training modifies staging only).
            3. Train N epochs.
            4. Copy in_training back to production slot on success.
            5. Atomic save of production adapter to disk.

        If training fails or the server crashes, the production adapter on
        disk is untouched — it still holds the last committed cycle's weights.
        """
        from paramem.models.loader import copy_adapter_weights, switch_adapter

        self._current_adapter = job.adapter_name
        self._last_completed_epoch = 0

        # Stage: copy production weights into in_training as the starting point
        if "in_training" not in self.model.peft_config:
            raise RuntimeError(
                "in_training staging adapter not found — was ConsolidationLoop initialized?"
            )
        copy_adapter_weights(self.model, src=job.adapter_name, dst="in_training")
        switch_adapter(self.model, "in_training")
        self.model.train()

        examples = format_indexed_training(job.keyed_pairs, self.tokenizer, max_length=1024)
        dataset = _SimpleDataset(examples)

        if not examples:
            logger.info("No training examples for %s, skipping", job.adapter_name)
            return

        # Staging checkpoints go under in_training/ to avoid polluting production
        checkpoint_dir = self.output_dir / "in_training" / "bg_checkpoint"

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

        try:
            trainer.train()
        except Exception:
            # On training failure: restore sane model state before the
            # exception propagates to _run_jobs. Production adapter on disk
            # is untouched (commit is skipped), but the in-memory active
            # adapter must be switched back to production so any lingering
            # inference path doesn't operate on stale in_training state.
            logger.exception("Training failed for adapter %s — restoring state", job.adapter_name)
            try:
                switch_adapter(self.model, job.adapter_name)
                self.model.eval()
                if self.training_config.gradient_checkpointing:
                    self.model.gradient_checkpointing_disable()
            except Exception:
                logger.exception("Failed to restore model state after training error")
            raise

        self._commit_staging_to_production(job.adapter_name)

        self._last_completed_epoch = self.training_config.num_epochs
        logger.info("Training complete and committed: %s", job.adapter_name)

    def _commit_staging_to_production(self, production_name: str) -> None:
        """Atomically promote in_training weights to a production adapter.

        Order matters: weights must be copied BEFORE the atomic save so that
        save_pretrained serializes the just-committed production slot, not
        the pre-commit state. Encapsulated in one method to prevent
        accidental reordering.
        """
        from paramem.models.loader import atomic_save_adapter, copy_adapter_weights

        copy_adapter_weights(self.model, src="in_training", dst=production_name)
        target_dir = self.output_dir / production_name
        atomic_save_adapter(self.model, target_dir, production_name)


class _SimpleDataset:
    """Minimal dataset wrapper for pre-tokenized examples."""

    def __init__(self, examples: list[dict]):
        self._examples = examples

    def __len__(self):
        return len(self._examples)

    def __getitem__(self, idx):
        return self._examples[idx]
