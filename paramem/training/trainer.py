"""LoRA fine-tuning trainer for personal memory adapters."""

import hashlib
import json
import logging
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
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

# ---------------------------------------------------------------------------
# Staging slot constants
# ---------------------------------------------------------------------------

#: Singleton PEFT adapter name used as a scratch training slot.  Must match
#: the name used in ``ConsolidationLoop._ensure_adapters``.
_STAGING_ADAPTER = "in_training"


@dataclass(frozen=True)
class TrainingHooks:
    """Caller-supplied behaviour that ``train_adapter`` cannot derive from config.

    The callable fields are converted internally into a single HF
    ``TrainerCallback`` (``_HooksAdapterCallback``) that runs BEFORE the
    thermal throttle in the registered callback list, so inference yielding
    pre-empts throttle waits within the same step.

    All fields default to ``None`` — callers pass only the intents they need:

    - ``on_step_yield(global_step)``: invoked at every step boundary (optional;
      consolidation callers may supply one for yielding between steps).
    - ``on_epoch_persist(epoch, output_dir)``: invoked at every epoch end.
      Used by RAM-mode to copy the latest checkpoint from /dev/shm to the
      caller's output_dir at each epoch boundary.
    - ``on_save_persist(global_step, output_dir)``: invoked whenever HF
      Trainer saves a checkpoint (``on_save`` event). Fires in addition to
      ``on_epoch_persist`` at epoch boundaries.
    - ``on_shutdown_check()``: invoked at every step end and every epoch end;
      returning ``True`` sets ``control.should_training_stop``. Step-level
      check enables sub-epoch shutdown granularity.  When constructed via
      ``BackgroundTrainer.training_hooks_for_job``, this predicate ORs the
      BG abort event, ``_shutdown_requested``, and the caller's own gate —
      so abort signals reach the throttle's shutdown_fn without a separate
      field.
    """

    on_step_yield: Optional[Callable[[int], None]] = None
    on_epoch_persist: Optional[Callable[[int, str], None]] = None
    on_save_persist: Optional[Callable[[int, str], None]] = None
    on_shutdown_check: Optional[Callable[[], bool]] = None


class _HooksAdapterCallback(TrainerCallback):
    """Routes ``TrainingHooks`` intents to HF callback events.

    Registered before ``ThermalThrottleCallback`` so that inference yielding
    (``on_step_yield``) runs before any potential throttle wait at the same
    step. The epoch hook (``on_epoch_persist``) is used by the RAM-mode
    epoch-mirror writer. Shutdown checks (``on_shutdown_check``) fire at
    both step and epoch boundaries for sub-epoch shutdown granularity.
    """

    def __init__(self, hooks: TrainingHooks):
        self._hooks = hooks

    def on_step_end(self, args, state, control, **kwargs):
        if self._hooks.on_step_yield is not None:
            self._hooks.on_step_yield(state.global_step)
        if self._hooks.on_shutdown_check is not None and self._hooks.on_shutdown_check():
            logger.info(
                "Graceful shutdown requested via TrainingHooks — stopping after step %d",
                state.global_step,
            )
            control.should_training_stop = True

    def on_save(self, args, state, control, **kwargs):
        """Fire ``on_save_persist`` whenever HF Trainer writes a checkpoint.

        Fires in addition to ``on_epoch_end`` at epoch boundaries when
        ``save_strategy="epoch"``; callers that need dedup handle it themselves.
        """
        if self._hooks.on_save_persist is not None:
            self._hooks.on_save_persist(int(state.global_step), args.output_dir)

    def on_epoch_end(self, args, state, control, **kwargs):
        if self._hooks.on_epoch_persist is not None:
            self._hooks.on_epoch_persist(int(state.global_step), args.output_dir)
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


class _RamEpochCopyCallback(TrainerCallback):
    """Copy the latest /dev/shm checkpoint to the caller's output_dir at each epoch.

    Installed by ``train_adapter`` when ``save_steps_ram > 0``.  HF Trainer
    writes its checkpoints to a ``/dev/shm`` RAM-backed tmpfs directory
    (``ram_dir``).  At every epoch end this callback finds the highest-numbered
    ``checkpoint-N`` under ``ram_dir`` and copies it to
    ``<caller_output_dir>/bg_checkpoint_epoch/checkpoint-N/``, replacing the
    previous copy.  This gives callers a durable (if one-epoch-stale) copy
    without paying encrypted-disk IO per step.
    """

    def __init__(self, ram_dir: Path, caller_output_dir: Path) -> None:
        """Args:
        ram_dir: The /dev/shm tmpfs directory where HF Trainer saves checkpoints.
        caller_output_dir: The caller's original output_dir; epoch copies land
            under ``<caller_output_dir>/bg_checkpoint_epoch/``.
        """
        self._ram_dir = ram_dir
        self._epoch_dir = caller_output_dir / "bg_checkpoint_epoch"

    def on_epoch_end(self, args, state, control, **kwargs) -> None:
        """Copy the latest RAM checkpoint to the epoch-persistent directory."""
        checkpoints = sorted(
            self._ram_dir.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[1]) if p.name.split("-")[1].isdigit() else -1,
        )
        if not checkpoints:
            return
        latest = checkpoints[-1]
        dest = self._epoch_dir / latest.name
        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(str(latest), str(dest))
        logger.debug(
            "RAM-mode: copied %s → %s",
            latest,
            dest,
        )


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


# ---------------------------------------------------------------------------
# Staging slot helpers
# ---------------------------------------------------------------------------


def _ensure_staging_slot(model: PeftModel, adapter_config: AdapterConfig) -> None:
    """Create the transient ``in_training`` PEFT adapter for this training event.

    Per the staging+promote contract, the staging slot is transient: it exists
    only while a training event is in flight, and it is deleted by the caller
    at the post-save cleanup step.  Every training event therefore enters this
    helper with the slot absent and creates a byte-fresh adapter from seeded
    LoRA initialisation.

    Pre-existing slot at entry is a lifecycle-invariant violation — it means
    the prior training event did not clean up after itself.  Raising here
    surfaces the bug loudly rather than silently inheriting potentially-stale
    weights.  This is the "no room for side effects" property of the design.

    Args:
        model: Live ``PeftModel`` to mutate.  Must NOT carry an existing
            ``in_training`` slot.
        adapter_config: Target LoRA config for the new slot.

    Raises:
        RuntimeError: when ``in_training`` already exists in ``model.peft_config``.
            Indicates a missing cleanup at the prior training event's success or
            rollback path.
    """
    from paramem.models.loader import create_adapter

    if _STAGING_ADAPTER in model.peft_config:
        raise RuntimeError(
            f"Lifecycle invariant violated: {_STAGING_ADAPTER!r} already present "
            "in model.peft_config at training entry. The prior training event "
            "did not delete the slot — check the post-save cleanup site "
            "(active_store_migration.migrate / consolidation._save_adapters) "
            "and the recall-gate-failure rollback path."
        )
    create_adapter(model, adapter_config, _STAGING_ADAPTER)


def _fingerprint_dataset(train_dataset) -> str:
    """SHA-256 fingerprint of the training dataset content.

    When ``train_dataset`` is a ``list`` or ``tuple``, each element is
    serialised as canonical JSON and hashed in order.  For other dataset
    types (e.g. HF ``Dataset``), the fingerprint is derived from
    ``str(train_dataset)`` — deterministic within a run but not across
    library version changes.  The limitation is documented: callers using
    non-list datasets may not benefit from crash-resume matching across
    process restarts.

    Returns:
        Hex-encoded SHA-256 digest string.
    """
    h = hashlib.sha256()
    if isinstance(train_dataset, (list, tuple)):
        for item in train_dataset:
            h.update(json.dumps(item, sort_keys=True, default=str).encode("utf-8"))
    else:
        h.update(str(train_dataset).encode("utf-8"))
    return h.hexdigest()


def _fingerprint_training_config(
    training_config: TrainingConfig, adapter_config: AdapterConfig
) -> str:
    """SHA-256 fingerprint of the training-relevant configuration fields.

    Covers all fields that determine the learning-rate schedule, training
    budget, and adapter architecture.  Changes to any of these fields
    invalidate a cached staging_resume.json so a fresh run starts.

    Args:
        training_config: The ``TrainingConfig`` for the current job.
        adapter_config: The ``AdapterConfig`` for the adapter being trained.

    Returns:
        Hex-encoded SHA-256 digest string.
    """
    relevant = {
        "save_strategy": training_config.save_strategy,
        "save_steps": training_config.save_steps,
        "save_steps_ram": training_config.save_steps_ram,
        "num_epochs": training_config.num_epochs,
        "batch_size": training_config.batch_size,
        "lr_scheduler_type": training_config.lr_scheduler_type,
        "weight_decay": training_config.weight_decay,
        "warmup_steps": training_config.warmup_steps,
        "warmup_ratio": training_config.warmup_ratio,
        "rank": adapter_config.rank,
        "alpha": adapter_config.alpha,
        "learning_rate": adapter_config.learning_rate,
        "target_modules": sorted(adapter_config.target_modules),
    }
    payload = json.dumps(relevant, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_staging_resume(scratch_path: Path) -> dict | None:
    """Read and deserialise ``staging_resume.json`` if it exists.

    Uses ``read_maybe_encrypted`` to handle both plaintext and age-encrypted
    files transparently (encryption posture determined at write time by
    ``write_infra_bytes``).

    Args:
        scratch_path: Path to ``staging_resume.json``.

    Returns:
        Parsed dict or ``None`` when the file is absent, unreadable, or
        malformed.
    """
    if not scratch_path.exists():
        return None
    try:
        from paramem.backup.encryption import read_maybe_encrypted

        raw = read_maybe_encrypted(scratch_path)
        return json.loads(raw.decode("utf-8"))
    except Exception:  # noqa: BLE001  # boundary: external-file read
        logger.debug("staging_resume.json unreadable — treating as absent", exc_info=True)
        return None


def _write_staging_resume(scratch_path: Path, state: dict) -> None:
    """Write ``state`` to ``staging_resume.json`` via ``write_infra_bytes``.

    The file is age-encrypted when a daily identity is loaded; plaintext
    otherwise.  ``scratch_path.parent`` must exist before this call.

    Args:
        scratch_path: Destination path for ``staging_resume.json``.
        state: Dict to serialise.  Must be JSON-serialisable.
    """
    from paramem.backup.encryption import write_infra_bytes

    payload = json.dumps(state, indent=2).encode("utf-8")
    write_infra_bytes(scratch_path, payload)


def _resolve_resume_checkpoint(
    scratch_path: Path,
    fingerprints: dict[str, str],
) -> str | None:
    """Resolve the best available checkpoint path for a crash-resume.

    Reads ``staging_resume.json`` at *scratch_path* and validates it against
    *fingerprints*.  Returns the highest-priority existing checkpoint path
    using the 3-way preference:

    1. ``ram_checkpoint_path`` under ``/dev/shm`` — present only when
       ``save_steps_ram > 0`` and the process has NOT restarted since the
       interrupted run.
    2. ``disk_checkpoint_path`` under ``<output_dir>/bg_checkpoint_epoch/``
       — the durable epoch-mirror written by ``_RamEpochCopyCallback``; survives
       WSL reboot.
    3. ``checkpoint_path`` — back-compat field for pre-existing files.

    Returns ``None`` when the file is absent, fingerprints mismatch, or no
    valid checkpoint directory is found.

    Args:
        scratch_path: Path to ``staging_resume.json``.
        fingerprints: Dict with ``"dataset"`` and ``"config"`` keys (hex SHA-256).
    """
    state = _read_staging_resume(scratch_path)
    if state is None:
        return None

    if state.get("dataset_fingerprint") != fingerprints["dataset"]:
        logger.debug("staging_resume.json dataset fingerprint mismatch — fresh start")
        return None
    if state.get("training_config_fingerprint") != fingerprints["config"]:
        logger.debug("staging_resume.json config fingerprint mismatch — fresh start")
        return None

    # 3-way preference: RAM → disk epoch mirror → legacy
    for field_name in ("ram_checkpoint_path", "disk_checkpoint_path", "checkpoint_path"):
        ckpt = state.get(field_name, "")
        if ckpt and Path(ckpt).is_dir():
            logger.info("Crash-resume: resolved checkpoint via %s → %s", field_name, ckpt)
            return ckpt

    logger.debug("staging_resume.json present but no checkpoint dir found — fresh start")
    return None


def _clean_scratch(output_dir: Path, ram_dir: Optional[Path]) -> None:
    """Remove transient training scratch directories on normal or abort completion.

    Deletes the HF Trainer checkpoint trees written during training so that
    subsequent boot integrity checks (which expect adapter slots to contain
    only ``adapter_model.safetensors`` and ``adapter_config.json``) do not
    trip on stale ``checkpoint-*`` or ``bg_checkpoint_epoch`` directories.

    On crash, scratch is intentionally preserved to enable crash-resume.

    Args:
        output_dir: The caller's ``output_dir`` passed to ``train_adapter``.
            Checkpoint debris under this directory is removed.
        ram_dir: The ``/dev/shm`` directory created by RAM-mode, or ``None``
            when RAM mode is disabled.
    """
    # Epoch-mirror directory written by _RamEpochCopyCallback.
    epoch_mirror = output_dir / "bg_checkpoint_epoch"
    if epoch_mirror.exists():
        shutil.rmtree(epoch_mirror, ignore_errors=True)
        logger.debug("Cleaned epoch-mirror checkpoint dir: %s", epoch_mirror)

    # Any checkpoint-* directories that HF Trainer may have written directly
    # to output_dir when save_steps_ram == 0.
    for ckpt_dir in output_dir.glob("checkpoint-*"):
        if ckpt_dir.is_dir():
            shutil.rmtree(ckpt_dir, ignore_errors=True)
            logger.debug("Cleaned checkpoint dir: %s", ckpt_dir)

    # RAM-mode /dev/shm directory.
    if ram_dir is not None and ram_dir.exists():
        shutil.rmtree(ram_dir, ignore_errors=True)
        logger.debug("Cleaned RAM checkpoint dir: %s", ram_dir)


class _StagingResumeCallback(TrainerCallback):
    """Update ``staging_resume.json`` at every HF Trainer checkpoint save.

    Installed by ``train_adapter`` when the staging+promote contract is active.
    Records the latest checkpoint paths (RAM and disk epoch mirror) so a
    subsequent crash-resume call can find them.

    Args:
        scratch_path: Path to ``staging_resume.json``.
        ram_dir: ``/dev/shm`` checkpoint root, or ``None`` when RAM mode is
            disabled.
        output_dir: The caller's ``output_dir`` (not the RAM dir).  Used to
            derive ``disk_checkpoint_path`` under ``bg_checkpoint_epoch/``.
        base_state: The initial state dict written at ``train_adapter`` entry;
            this callback only updates the checkpoint path fields and
            ``updated_at`` without re-writing the fingerprints.
    """

    def __init__(
        self,
        scratch_path: Path,
        ram_dir: Optional[Path],
        output_dir: Path,
        base_state: dict,
    ) -> None:
        self._scratch_path = scratch_path
        self._ram_dir = ram_dir
        self._epoch_dir = output_dir / "bg_checkpoint_epoch"
        self._base_state = base_state

    def _latest_checkpoint(self, search_root: Path) -> str:
        """Return the path of the highest-numbered checkpoint under *search_root*."""
        checkpoints = sorted(
            search_root.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[1]) if p.name.split("-")[1].isdigit() else -1,
        )
        return str(checkpoints[-1]) if checkpoints else ""

    def on_save(self, args, state, control, **kwargs) -> None:
        """Update scratch state whenever HF Trainer writes a checkpoint."""
        updated = dict(self._base_state)
        updated["updated_at"] = datetime.now(timezone.utc).isoformat()

        if self._ram_dir is not None and self._ram_dir.is_dir():
            updated["ram_checkpoint_path"] = self._latest_checkpoint(self._ram_dir)
        if self._epoch_dir.is_dir():
            updated["disk_checkpoint_path"] = self._latest_checkpoint(self._epoch_dir)

        try:
            _write_staging_resume(self._scratch_path, updated)
        except Exception:  # noqa: BLE001  # boundary: filesystem write
            logger.warning("Failed to update staging_resume.json", exc_info=True)


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
    """Train a LoRA adapter on the given dataset with staging+promote contract.

    Implements the staging+promote contract that prevents mutation of
    production adapter weights until training has successfully completed.
    The staging slot (``in_training``) is **transient** — created at
    training entry, deleted at training exit (both success and abort paths)
    — so it never carries weights from one training event into the next.

    Steps:

    1. ``_ensure_staging_slot`` creates a byte-fresh ``in_training`` PEFT
       adapter from seeded LoRA initialisation.  If the slot is already
       present at entry, the prior training event failed to clean up — raises
       ``RuntimeError`` (lifecycle invariant guard).
    2. Copies production weights into the staging slot.  Consolidation:
       production holds prior-cycle weights → staging starts there (incremental
       learning).  Migration: caller force-resets production to LoRA-zero
       before this call → staging starts at LoRA-zero (fresh start).
    3. Activates the staging slot so HF Trainer trains there exclusively.
    4. Checks for a prior crash-resume via ``staging_resume.json`` and
       resolves the best available checkpoint (RAM → disk epoch-mirror →
       legacy, in preference order).
    5. Runs HF Trainer.
    6. On normal completion: promotes staging weights into the production slot
       (``copy_adapter_weights(staging → production)``), switches the active
       adapter to production, cleans scratch state, **deletes the staging
       slot**.
       On abort: restores the production adapter without promoting, cleans
       scratch, **deletes the staging slot**.
       On exception (crash): restores the production adapter (best-effort),
       leaves scratch intact for the next crash-resume.  PEFT state dies with
       the process; the next process boot enters this function with a fresh
       ``model.peft_config`` (no staging slot present).

    Caller responsibilities (post-return):

    - Run the 1.0 recall sanity gate against the prior-model key-triple set
      (``loop._run_recall_sanity_probe`` in migration,
      ``_verify_saved_adapter_from_disk`` in consolidation).  On failure, roll
      back the production adapter to LoRA-zero — staging is already gone, no
      cleanup needed.
    - ``atomic_save_adapter(production)`` to persist the slot durably.

    If ``active_adapters`` is provided, the staging+promote path is skipped
    and the existing multi-adapter compose-training path runs instead (all
    listed adapters active; only *adapter_name* receives gradients).

    Args:
        model: ``PeftModel`` carrying at least the production adapter named by
            *adapter_name*.  Must NOT carry an existing ``in_training`` slot —
            a violation raises ``RuntimeError``.
        tokenizer: HF tokenizer passed through to ``Trainer`` (not used
            directly inside this function).
        train_dataset: Tokenised training data.  May be a ``list``, HF
            ``Dataset``, or any object with ``__len__`` / ``__getitem__``.
        adapter_name: Name of the **production** adapter tier to train (e.g.
            ``"episodic"`` or ``"episodic_interim_20260501T1200"``).  Training
            runs on the staging slot; results are promoted back here.
        training_config: Training hyper-parameters.
        adapter_config: LoRA config for the production adapter tier.  Staging
            slot shape is matched to this config.
        wandb_config: When set and ``wandb_config.enabled`` is ``True``,
            enables wandb logging.
        output_dir: Directory for HF Trainer outputs (checkpoints, logs).
            Defaults to ``outputs/adapters/<adapter_name>``.
        eval_dataset: Optional eval dataset; passed through to ``Trainer``.
        run_name: ``wandb`` run name; defaults to ``paramem-<adapter_name>``.
        callbacks_extra: Optional list of extra HF callbacks appended after
            all cross-cutting callbacks (thermal throttle, recall probe, etc.).
        active_adapters: When set, activates all listed adapters in the forward
            pass and skips the staging+promote path (compose-training mode).
            Only *adapter_name* receives gradients — caller must freeze others
            via ``set_requires_grad``.
        resume_from_checkpoint: Explicit HF Trainer checkpoint path.  When set,
            it takes precedence over any crash-resume found in
            ``staging_resume.json``.  When ``None`` (default), the function
            checks ``staging_resume.json`` for a prior interrupted run before
            starting fresh.
        thermal_policy: When set, installs a ``ThermalThrottleCallback`` that
            pauses training when GPU temperature exceeds the policy limit.
            Default ``None`` skips the install.
        hooks: Caller-supplied ``TrainingHooks`` (inference yielding, epoch
            persist, shutdown predicate).  Installed before the thermal throttle
            so yielding pre-empts throttle waits.

    Returns:
        Training metrics dict with the following keys:

        - Standard HF Trainer metrics (``train_loss``, etc.).
        - ``aborted`` (``bool``): ``True`` when the training loop was stopped
          early because ``hooks.on_shutdown_check()`` returned ``True`` (abort
          for inference or graceful shutdown).  The predicate is re-polled once
          after ``trainer.train()`` returns — the poll is race-free because the
          abort holder's event remains set until the caller clears it.
          Callers that do not pass ``hooks`` always see ``aborted=False``.
    """
    if output_dir is None:
        output_dir = Path("outputs") / "adapters" / adapter_name
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Orphan PID sweep: when RAM mode is enabled, clean up /dev/shm directories
    # left by prior processes that no longer exist (crash/kill between runs).
    # os.kill(pid, 0) raises ProcessLookupError when the PID is dead; raises
    # PermissionError when it is alive but owned by another user (leave alone).
    ram_dir: Path | None = None
    if training_config.save_steps_ram > 0:
        for _shm_dir in Path("/dev/shm").glob("paramem-bg-checkpoint-*"):
            try:
                _pid = int(_shm_dir.name.split("-")[-1])
                os.kill(_pid, 0)
                # PID is alive — leave the directory
            except ProcessLookupError:
                shutil.rmtree(_shm_dir, ignore_errors=True)
                logger.debug("RAM-mode: swept orphan /dev/shm dir %s (PID dead)", _shm_dir)
            except (ValueError, PermissionError):
                pass  # non-numeric suffix or alive foreign process — leave alone

    # ------------------------------------------------------------------
    # Staging+promote setup (skipped for compose-training via active_adapters)
    # ------------------------------------------------------------------
    _use_staging = active_adapters is None

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
        # Step 1: Ensure the staging slot exists and matches shape.
        _ensure_staging_slot(model, adapter_config)

        # Step 2: Copy production → staging (best-effort; first-time tiers
        # have no production weights yet and start from LoRA-zero).
        if adapter_name in model.peft_config:
            from paramem.models.loader import copy_adapter_weights

            copy_adapter_weights(model, src=adapter_name, dst=_STAGING_ADAPTER)
            logger.debug(
                "Staging: copied production weights %s → %s", adapter_name, _STAGING_ADAPTER
            )
        else:
            logger.info(
                "Staging: no production adapter '%s' yet — staging starts from LoRA-zero",
                adapter_name,
            )

        # Step 3: Switch active adapter to staging so HF Trainer trains there.
        model.set_adapter(_STAGING_ADAPTER)

    # ------------------------------------------------------------------
    # Crash-resume: check staging_resume.json (overridden by explicit arg)
    # ------------------------------------------------------------------
    scratch_path = output_dir / "staging_resume.json"
    _effective_resume: Optional[str] = (
        str(resume_from_checkpoint) if resume_from_checkpoint is not None else None
    )
    if _use_staging and _effective_resume is None:
        # Step 4: Check for a prior interrupted run.
        _fingerprints = {
            "dataset": _fingerprint_dataset(train_dataset),
            "config": _fingerprint_training_config(training_config, adapter_config),
        }
        _resolved = _resolve_resume_checkpoint(scratch_path, _fingerprints)
        if _resolved is not None:
            _effective_resume = _resolved
            logger.info("Crash-resume: will resume from %s", _resolved)
    elif _use_staging and _effective_resume is not None:
        # Explicit resume_from_checkpoint — compute fingerprints for marker only.
        _fingerprints = {
            "dataset": _fingerprint_dataset(train_dataset),
            "config": _fingerprint_training_config(training_config, adapter_config),
        }
    else:
        _fingerprints = None  # compose-training path; no scratch marker written

    if _use_staging and _fingerprints is not None and _effective_resume is None:
        # No prior checkpoint found; write a fresh staging_resume.json so a
        # crash during this run leaves a marker for the next invocation.
        _scratch_state: dict = {
            "adapter_name": adapter_name,
            "dataset_fingerprint": _fingerprints["dataset"],
            "training_config_fingerprint": _fingerprints["config"],
            "ram_checkpoint_path": "",
            "disk_checkpoint_path": "",
            "checkpoint_path": "",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            _write_staging_resume(scratch_path, _scratch_state)
        except Exception:  # noqa: BLE001  # boundary: filesystem write
            logger.warning(
                "Could not write staging_resume.json — crash-resume disabled", exc_info=True
            )
            _scratch_state = {}
    elif _use_staging and _fingerprints is not None and _effective_resume is not None:
        # Resume case: keep the prior scratch state (already written by a prior
        # invocation); we do NOT overwrite it with fresh timestamps here so the
        # fingerprints remain intact for a potential further crash.
        _scratch_state = _read_staging_resume(scratch_path) or {}
    else:
        _scratch_state = {}

    report_to = "none"
    if wandb_config and wandb_config.enabled:
        report_to = "wandb"

    # When RAM mode is active, route HF Trainer's checkpoints to /dev/shm and
    # override save_strategy/save_steps so the Trainer actually writes checkpoints.
    # _RamEpochCopyCallback (registered below) copies each epoch's latest
    # checkpoint back to <output_dir>/bg_checkpoint_epoch/ for durability.
    _trainer_output_dir = str(output_dir)
    _save_strategy = training_config.save_strategy
    _save_steps = (
        max(1, training_config.save_steps)
        if training_config.save_steps > 0
        # HF default; preserves prior behaviour for callers that did not set save_steps
        else 500
    )
    if training_config.save_steps_ram > 0:
        ram_dir = Path(f"/dev/shm/paramem-bg-checkpoint-{os.getpid()}")
        ram_dir.mkdir(parents=True, exist_ok=True)
        _trainer_output_dir = str(ram_dir)
        _save_strategy = "steps"
        _save_steps = training_config.save_steps_ram

    training_args = TrainingArguments(
        output_dir=_trainer_output_dir,
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
        save_strategy=_save_strategy,
        save_steps=_save_steps,
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
        # Route the hooks' shutdown predicate into the throttle's shutdown_fn
        # so an abort or shutdown signal breaks the throttle's wait loop
        # cleanly.  The abort event is ORed into on_shutdown_check by
        # BackgroundTrainer.training_hooks_for_job, so no separate abort_fn
        # field is needed on ThermalThrottleCallback.
        shutdown_fn = (
            hooks.on_shutdown_check
            if hooks is not None and hooks.on_shutdown_check is not None
            else (lambda: False)
        )
        callbacks.append(
            ThermalThrottleCallback(
                thermal_policy,
                shutdown_fn=shutdown_fn,
            )
        )
    if ram_dir is not None:
        # Copy the latest RAM checkpoint to the caller's output_dir at each epoch
        # so there is always a durable (one-epoch-stale) copy available.
        callbacks.append(_RamEpochCopyCallback(ram_dir, output_dir))
    if _use_staging and _scratch_state:
        # Install the staging resume callback AFTER _RamEpochCopyCallback so
        # the epoch-mirror copy is already written when we record its path.
        callbacks.append(
            _StagingResumeCallback(
                scratch_path=scratch_path,
                ram_dir=ram_dir,
                output_dir=output_dir,
                base_state=_scratch_state,
            )
        )
    if callbacks_extra:
        callbacks.extend(callbacks_extra)

    # Under the staging+promote contract, HF trains the transient ``in_training``
    # slot, so the recall early-stop probe must measure that slot — not the
    # caller's production adapter name, which holds un-promoted weights until the
    # post-train promote.  As the single owner of the staging lifecycle, bind
    # the probe target explicitly here; the callback never infers it.  No-op
    # for compose/direct training, where the production adapter trains in place.
    if _use_staging:
        from paramem.training.early_stop import RecallEarlyStopCallback

        for _cb in callbacks:
            if isinstance(_cb, RecallEarlyStopCallback):
                _cb.set_probe_adapter(_STAGING_ADAPTER)

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
        "Starting training: adapter=%s (staging=%s), epochs=%d, lr=%e",
        adapter_name,
        _STAGING_ADAPTER if _use_staging else "compose-mode",
        training_config.num_epochs,
        adapter_config.learning_rate,
    )

    # Step 5: Resolve the effective checkpoint for HF Trainer.
    ckpt_arg = _effective_resume  # may be None (fresh start) or a path (resume)

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

    # Step 5 (continued): Run HF Trainer.  The try/except implements the
    # 3-path post-return decision (normal → promote, abort → no-promote,
    # crash → preserve scratch).
    try:
        try:
            result = trainer.train(resume_from_checkpoint=effective_ckpt_arg)
        finally:
            if shm_resume_dir is not None:
                shutil.rmtree(shm_resume_dir, ignore_errors=True)

        metrics = dict(result.metrics)

        # Detect abort: re-poll hooks.on_shutdown_check after trainer.train()
        # returns.  HF Trainer exits its training loop cleanly when
        # control.should_training_stop was set by _HooksAdapterCallback's
        # on_shutdown_check.  The abort holder's event stays set until the caller
        # clears it (in _run_callable_queue's finally, which runs after this
        # function returns), so the post-train poll is race-free.
        aborted = bool(
            hooks is not None and hooks.on_shutdown_check is not None and hooks.on_shutdown_check()
        )
        metrics["aborted"] = aborted

        if _use_staging:
            if not aborted:
                # Step 6a: NORMAL completion — promote staging → production.
                from paramem.models.loader import copy_adapter_weights, switch_adapter

                copy_adapter_weights(model, src=_STAGING_ADAPTER, dst=adapter_name)
                switch_adapter(model, adapter_name)
                logger.info("Staging: promoted %s → %s (VRAM)", _STAGING_ADAPTER, adapter_name)
                # Clean scratch on success.
                _clean_scratch(output_dir, ram_dir)
                scratch_path.unlink(missing_ok=True)
                # Delete the staging slot now that promote is done.
                # The staging slot is transient — exists only during this training
                # event.  Crash-safety + rollback rationale no longer applies: the
                # new weights are in production (VRAM), and if save fails later, the
                # prior production state on disk is still recoverable.
                model.delete_adapter(_STAGING_ADAPTER)
                logger.info("Staging: deleted %s (lifecycle: per-training-event)", _STAGING_ADAPTER)
            else:
                # Step 6b: ABORT — restore active adapter, do NOT promote.
                from paramem.models.loader import switch_adapter

                switch_adapter(model, adapter_name)
                logger.info(
                    "Staging: aborted — production %s unchanged; cleaning scratch",
                    adapter_name,
                )
                _clean_scratch(output_dir, ram_dir)
                scratch_path.unlink(missing_ok=True)
                # Delete the staging slot on abort too.  The staging slot is
                # transient and must not survive past this training event,
                # otherwise the next event's _ensure_staging_slot will trip its
                # lifecycle-invariant guard.
                model.delete_adapter(_STAGING_ADAPTER)
                logger.info(
                    "Staging: deleted %s after abort (lifecycle: per-training-event)",
                    _STAGING_ADAPTER,
                )
        else:
            # Compose-training path: clean up RAM dir on success.
            if not aborted and ram_dir is not None and ram_dir.exists():
                shutil.rmtree(ram_dir, ignore_errors=False)

        # No final save here.  ``train_adapter`` is responsible only for
        # training; the canonical encrypted slot-dir save is the
        # orchestrator's job (``ConsolidationLoop._save_adapters`` →
        # ``atomic_save_adapter`` → ``_encrypt_adapter_safetensors``).
        logger.info("Training complete: %s", metrics)
        return metrics

    except BaseException:
        # Step 6c: CRASH path — best-effort restore of production adapter;
        # scratch state intentionally preserved for crash-resume.
        # The try/except here is boundary teardown (safe-state restore on
        # exception), not error suppression — the exception is always re-raised.
        if _use_staging:
            try:
                from paramem.models.loader import switch_adapter

                switch_adapter(model, adapter_name)
                logger.info(
                    "Staging: exception — restored active adapter to %s (best-effort)",
                    adapter_name,
                )
            except Exception:  # noqa: BLE001  # best-effort: switch may fail if model is in bad state
                logger.warning(
                    "Staging: could not restore active adapter to %s after exception",
                    adapter_name,
                    exc_info=True,
                )
        # Do NOT clean scratch — it is needed for crash-resume on next start.
        raise
