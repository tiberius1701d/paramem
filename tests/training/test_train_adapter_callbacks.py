"""Verify ``train_adapter`` callback assembly order and staging+promote contract.

Callback assembly order is locked by registration order in ``train_adapter``;
``TestCallbackOrdering`` asserts it by inspecting the constructed list
directly.

The ``TestStagingPromoteContract`` class verifies the staging contract:
- staging slot created/reshaped at entry
- production weights copied to staging at entry
- normal completion leaves staging resident and active for the caller, and
  cleans scratch — train_adapter itself never promotes or deletes the slot
- promote_staging_adapter (called by the caller) copies staging into
  production and switches the active adapter
- a second call without caller disposal trips the lifecycle guard
- abort path does NOT promote, deletes staging, and still cleans scratch
- crash path preserves scratch for crash-resume and still deletes staging
- resume resolution via the on-disk checkpoint pointer (found → absent)

No GPU required: the test patches ``ParamemTrainer``, PEFT, and encryption
helpers so staging logic runs without a real training run.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
import torch
from transformers import TrainerCallback

from paramem.training.thermal_throttle import ThermalPolicy
from paramem.training.trainer import (
    TrainingHooks,
    _HooksAdapterCallback,
    train_adapter,
)
from paramem.utils.config import AdapterConfig, TrainingConfig


class _MarkerCallback(TrainerCallback):
    """Sentinel for the call-bound ``callbacks_extra`` slot."""


def _capture_callbacks(**train_adapter_kwargs):
    """Run ``train_adapter`` with ``Trainer`` mocked; return the callbacks list.

    Builds a model that satisfies the staging contract end-to-end — both
    production and staging slots present with matching shape, and
    ``named_parameters`` returns one real tensor per ``(target_module, slot)``
    pair so ``copy_adapter_weights`` runs cleanly at entry.  Tests here cover
    callback assembly, not staging behaviour itself.
    """
    import torch

    captured = {}

    def _capture_init(*args, callbacks=None, **kwargs):
        captured["callbacks"] = callbacks
        instance = MagicMock()
        instance.train.return_value = MagicMock(metrics={"train_loss": 0.0})
        return instance

    adapter_config = train_adapter_kwargs.pop("adapter_config", AdapterConfig())
    adapter_name = "episodic"

    model = MagicMock()
    prod_cfg = MagicMock()
    prod_cfg.r = adapter_config.rank
    prod_cfg.target_modules = set(adapter_config.target_modules)
    staging_cfg = MagicMock()
    staging_cfg.r = adapter_config.rank
    staging_cfg.target_modules = set(adapter_config.target_modules)
    # peft_config pre-populated with both production + staging; the test patches
    # _ensure_staging_slot to a no-op below so the pre-existing slot does NOT
    # trip the staging lifecycle-invariant guard.  Callback-ordering tests do
    # not exercise staging slot create/delete — those are covered by
    # TestStagingPromoteContract.
    model.peft_config = {adapter_name: prod_cfg, "in_training": staging_cfg}

    named_params: list[tuple[str, "torch.Tensor"]] = []
    for module in sorted(adapter_config.target_modules):
        for slot in (adapter_name, "in_training"):
            named_params.append((f"base_model.model.{module}.{slot}.weight", torch.zeros(1)))
    model.named_parameters.return_value = named_params
    model.parameters.return_value = [t for _, t in named_params]

    # TrainingArguments validates bf16 against device support at __init__.
    # CI runs CPU-only — patch it out; the callbacks list is what's under test.
    with (
        tempfile.TemporaryDirectory() as tmpdir,
        patch("paramem.training.trainer.TrainingArguments", return_value=MagicMock()),
        patch("paramem.training.trainer.ParamemTrainer", side_effect=_capture_init),
        # Bypass the staging lifecycle guard for callback-ordering tests.
        patch("paramem.training.trainer._ensure_staging_slot", return_value=None),
    ):
        train_adapter(
            model=model,
            tokenizer=MagicMock(),
            train_dataset=MagicMock(),
            adapter_name=adapter_name,
            training_config=train_adapter_kwargs.pop("training_config", TrainingConfig()),
            adapter_config=adapter_config,
            output_dir=Path(tmpdir),
            **train_adapter_kwargs,
        )
    return captured["callbacks"]


class TestCallbackOrdering:
    def _policy(self):
        return ThermalPolicy(
            temp_limit=55,
            check_interval=1,
            quiet_hours_mode="always_on",
            quiet_hours_start="22:00",
            quiet_hours_end="07:00",
        )

    def test_baseline_minimal(self):
        # No hooks, no thermal_policy, no callbacks_extra → encryption +
        # staging-resume bookkeeping (always installed on the staging path).
        cbs = _capture_callbacks()
        types = [type(cb).__name__ for cb in cbs]
        assert types == ["EncryptCheckpointCallback", "_StagingResumeCallback"]

    def test_extra_callbacks_trail(self):
        marker = _MarkerCallback()
        hooks = TrainingHooks(on_shutdown_check=lambda: False)
        cbs = _capture_callbacks(
            hooks=hooks,
            thermal_policy=self._policy(),
            callbacks_extra=[marker],
        )
        # marker must be the last entry — call-bound callbacks (e.g. recall
        # probe) are assembled after every cross-cutting concern.
        assert cbs[-1] is marker

    def test_full_assembly_order(self):
        """With loss early stopping off, the cross-cutting callbacks register
        in one fixed order: EncryptCheckpointCallback, _HooksAdapterCallback,
        ThermalThrottleCallback, _StagingResumeCallback, then
        callbacks_extra."""
        marker = _MarkerCallback()
        hooks = TrainingHooks(on_shutdown_check=lambda: False)
        cbs = _capture_callbacks(
            hooks=hooks,
            thermal_policy=self._policy(),
            callbacks_extra=[marker],
        )
        types = [type(cb).__name__ for cb in cbs]
        assert types == [
            "EncryptCheckpointCallback",
            "_HooksAdapterCallback",
            "ThermalThrottleCallback",
            "_StagingResumeCallback",
            "_MarkerCallback",
        ]


class TestHooksAdapterCallbackBehaviour:
    def test_shutdown_check_sets_should_stop(self):
        hooks = TrainingHooks(on_shutdown_check=lambda: True)
        cb = _HooksAdapterCallback(hooks)
        control = MagicMock()
        control.should_training_stop = False
        cb.on_epoch_end(args=MagicMock(), state=MagicMock(epoch=1.0), control=control)
        assert control.should_training_stop is True

    def test_shutdown_check_false_leaves_control_alone(self):
        hooks = TrainingHooks(on_shutdown_check=lambda: False)
        cb = _HooksAdapterCallback(hooks)
        control = MagicMock()
        control.should_training_stop = False
        cb.on_epoch_end(args=MagicMock(), state=MagicMock(epoch=1.0), control=control)
        assert control.should_training_stop is False

    def test_all_intents_none_is_safe(self):
        cb = _HooksAdapterCallback(TrainingHooks())
        # All event handlers must run without raising when intents are None.
        cb.on_step_end(args=MagicMock(), state=MagicMock(global_step=1), control=MagicMock())
        cb.on_epoch_end(args=MagicMock(), state=MagicMock(epoch=1.0), control=MagicMock())
        cb.on_save(
            args=MagicMock(output_dir="/tmp/x"),
            state=MagicMock(global_step=1),
            control=MagicMock(),
        )

    def test_step_end_shutdown_check_sets_should_stop(self):
        """on_shutdown_check=True at step_end sets control.should_training_stop."""
        hooks = TrainingHooks(on_shutdown_check=lambda: True)
        cb = _HooksAdapterCallback(hooks)
        control = MagicMock()
        control.should_training_stop = False
        cb.on_step_end(
            args=MagicMock(),
            state=MagicMock(global_step=10),
            control=control,
        )
        assert control.should_training_stop is True


# ---------------------------------------------------------------------------
# Helpers shared by TestStagingPromoteContract
# ---------------------------------------------------------------------------


def _make_staging_model(
    *,
    has_staging: bool = False,
    staging_rank: int = 4,
    staging_modules: tuple[str, ...] = ("q_proj",),
    production_warm: bool = False,
) -> MagicMock:
    """Return a MagicMock PeftModel for staging+promote tests.

    ``peft_config`` starts with ``"episodic"`` (production tier).  The
    ``"in_training"`` staging slot is pre-populated only when
    ``has_staging=True``.

    ``production_warm`` selects which of the production tier's two real
    states the fixture models: ``False`` (default) is a freshly created,
    never-trained adapter — ``named_parameters`` carries no LoRA tensors,
    so ``has_prior_trained_weights`` reads ``False`` and the staging slot
    starts from LoRA-zero init.  ``True`` models a production adapter that
    HAS trained weights: ``named_parameters`` carries a non-zero
    ``lora_B.episodic`` tensor, so the adapter measures warm and
    ``train_adapter``'s entry copy (production → staging) fires.

    The mock absorbs ``set_adapter``, ``add_adapter``, ``delete_adapter``,
    ``named_parameters``, and ``parameters`` calls so staging logic runs
    without real PEFT.
    """
    model = MagicMock()

    episodic_cfg = MagicMock()
    episodic_cfg.r = 4
    episodic_cfg.target_modules = {"q_proj"}

    if has_staging:
        staging_cfg = MagicMock()
        staging_cfg.r = staging_rank
        staging_cfg.target_modules = set(staging_modules)
        model.peft_config = {"episodic": episodic_cfg, "in_training": staging_cfg}
    else:
        model.peft_config = {"episodic": episodic_cfg}

    model.set_adapter.return_value = None
    model.add_adapter.return_value = None
    model.delete_adapter.return_value = None
    if production_warm:
        warm_b = torch.nn.Parameter(torch.ones(2, 2))
        model.named_parameters.return_value = [
            ("base_model.model.layers.0.q_proj.lora_B.episodic.weight", warm_b),
        ]
    else:
        model.named_parameters.return_value = []
    model.parameters.return_value = []
    return model


def _minimal_tc(**overrides) -> TrainingConfig:
    """Return a minimal ``TrainingConfig`` for staging tests."""
    cfg = TrainingConfig(
        num_epochs=1,
        batch_size=1,
        gradient_accumulation_steps=1,
        warmup_steps=0,
        lr_scheduler_type="linear",
        weight_decay=0.01,
        max_grad_norm=1.0,
        gradient_checkpointing=False,
        seed=42,
        save_strategy="no",
        save_total_limit=1,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _minimal_ac(rank: int = 4, target_modules: tuple[str, ...] = ("q_proj",)) -> AdapterConfig:
    return AdapterConfig(
        rank=rank,
        alpha=rank * 2,
        learning_rate=1e-4,
        target_modules=list(target_modules),
    )


def _minimal_dataset() -> list[dict]:
    return [{"input_ids": [1, 2], "labels": [1, 2]}]


class _NullTrainer:
    """Minimal fake Trainer: train() returns clean metrics without aborting."""

    def __init__(self, *, model, args, train_dataset, data_collator, callbacks, **kwargs):
        self._callbacks = callbacks
        self._args = args

    def train(self, resume_from_checkpoint=None):
        result = MagicMock()
        result.metrics = {"train_loss": 0.05}
        return result


class _AbortingTrainer(_NullTrainer):
    """Fake Trainer that fires the shutdown hook during training."""

    def train(self, resume_from_checkpoint=None):
        # Fire on_shutdown_check via on_epoch_end so hooks see abort.
        control = MagicMock()
        control.should_training_stop = False
        state = MagicMock(global_step=1, epoch=1.0)
        for cb in self._callbacks:
            if hasattr(cb, "on_epoch_end"):
                cb.on_epoch_end(self._args, state, control)
        result = MagicMock()
        result.metrics = {"train_loss": 0.5}
        return result


class _RaisingTrainer(_NullTrainer):
    """Fake Trainer that raises RuntimeError mid-train (crash simulation)."""

    def train(self, resume_from_checkpoint=None):
        raise RuntimeError("simulated crash")


def _make_checkpoint_writing_trainer(out_dir: Path):
    """Return a Trainer subclass that writes checkpoint-10 during train().

    Models what HF Trainer does in practice: it writes checkpoint dirs to
    output_dir DURING training, not before.  Used by TestRetainScratchFlag
    so that the pre-training fresh-start purge never sees a stale checkpoint,
    and the post-training retain flag operates on a checkpoint written during
    the current run.

    Args:
        out_dir: The ``output_dir`` passed to ``train_adapter``.  The trainer
            writes ``out_dir / "checkpoint-10"`` during ``train()``.
    """

    class _CheckpointWritingTrainer(_NullTrainer):
        def train(self, resume_from_checkpoint=None):
            (out_dir / "checkpoint-10").mkdir(parents=True, exist_ok=True)
            return super().train(resume_from_checkpoint=resume_from_checkpoint)

    return _CheckpointWritingTrainer


def _make_checkpoint_and_save_trainer(out_dir: Path):
    """Return a Trainer subclass that writes checkpoint-10 and fires
    ``on_save`` on every registered callback during ``train()``.

    Models what HF Trainer does in practice: it writes a checkpoint dir AND
    dispatches ``on_save`` to every callback at the same point, so
    ``_StagingResumeCallback.on_save`` records the checkpoint path into
    ``staging_resume.json``. ``_CheckpointWritingTrainer`` above writes the
    checkpoint dir but never dispatches ``on_save``, so it cannot be used to
    prove a subsequent ``train_adapter`` call resumes from the checkpoint —
    this variant closes that gap for the retain-then-resume test.

    Args:
        out_dir: The ``output_dir`` passed to ``train_adapter``. The trainer
            writes ``out_dir / "checkpoint-10"`` during ``train()``.
    """

    class _CheckpointAndSaveTrainer(_NullTrainer):
        def train(self, resume_from_checkpoint=None):
            (out_dir / "checkpoint-10").mkdir(parents=True, exist_ok=True)
            state = MagicMock(global_step=10)
            control = MagicMock()
            for cb in self._callbacks:
                if hasattr(cb, "on_save"):
                    cb.on_save(self._args, state, control)
            return super().train(resume_from_checkpoint=resume_from_checkpoint)

    return _CheckpointAndSaveTrainer


def _staging_patches(tmp_path, *, trainer_cls=_NullTrainer, abort_shutdown=False):
    """Return a context-manager stack of patches for staging+promote tests.

    Patches:
    - TrainingArguments (no HF validation)
    - ParamemTrainer
    - paramem.models.loader.create_adapter (no real PEFT)
    - paramem.models.loader.copy_adapter_weights (no real tensor copy)
    - paramem.models.loader.switch_adapter (no real adapter activation)
    - paramem.backup.encryption.write_infra_bytes (plaintext write)
    - paramem.backup.encryption.read_maybe_encrypted (plaintext read)
    - paramem.backup.key_store.daily_identity_available (Security OFF)
    - EncryptCheckpointCallback (no-op)

    ``abort_shutdown=True`` replaces trainer_cls with _AbortingTrainer AND
    wires an always-True shutdown predicate so the post-train re-poll returns
    True.
    """
    from contextlib import ExitStack

    stack = ExitStack()

    if abort_shutdown:
        trainer_cls = _AbortingTrainer

    # --- TrainingArguments / Trainer ---
    stack.enter_context(
        patch("paramem.training.trainer.TrainingArguments", return_value=MagicMock())
    )
    stack.enter_context(patch("paramem.training.trainer.ParamemTrainer", new=trainer_cls))

    # --- Loader helpers (deferred import inside _ensure_staging_slot et al) ---
    # create_adapter's side effect mirrors real PEFT: it registers the new
    # adapter name in peft_config.  Without this, drop_adapter_slot's
    # presence guard (`if name in model.peft_config`) sees the staging slot
    # as never having been created and silently no-ops the delete.
    def _mock_create_adds_slot(model_arg, adapter_config, name):
        model_arg.peft_config[name] = MagicMock()
        return model_arg

    mock_create = stack.enter_context(
        patch("paramem.models.loader.create_adapter", side_effect=_mock_create_adds_slot)
    )
    mock_copy = stack.enter_context(
        patch("paramem.models.loader.copy_adapter_weights", return_value=None)
    )
    mock_switch = stack.enter_context(
        patch("paramem.models.loader.switch_adapter", return_value=None)
    )

    # --- Encryption (write as plaintext, read plaintext back) ---
    def _write_plain(path, data):
        Path(path).write_bytes(data)

    def _read_plain(path):
        return Path(path).read_bytes()

    stack.enter_context(
        patch(
            "paramem.backup.encryption.write_infra_bytes",
            side_effect=_write_plain,
        )
    )
    stack.enter_context(
        patch(
            "paramem.backup.encryption.read_maybe_encrypted",
            side_effect=_read_plain,
        )
    )
    stack.enter_context(
        patch("paramem.backup.key_store.daily_identity_available", return_value=False)
    )

    # --- EncryptCheckpointCallback ---
    stack.enter_context(
        patch(
            "paramem.training.encrypted_checkpoint_callback.EncryptCheckpointCallback",
            MagicMock,
        )
    )

    return stack, mock_create, mock_copy, mock_switch


def _track_adapter_state(model: MagicMock, mock_create: MagicMock, mock_switch: MagicMock) -> None:
    """Give a staging model PEFT's adapter bookkeeping.

    The production tier ``"episodic"`` starts active. ``set_adapter``
    activates only a resident adapter, ``delete_adapter`` removes the slot
    from ``peft_config``, and the patched ``create_adapter`` and
    ``switch_adapter`` from :func:`_staging_patches` register and activate
    adapters through the model as the real helpers do, so a test can assert
    on the model's resulting state instead of on the calls made to it.

    Args:
        model: A model from :func:`_make_staging_model`.
        mock_create: The patched ``create_adapter``.
        mock_switch: The patched ``switch_adapter``.
    """
    model.active_adapter = "episodic"

    def _set_adapter(name: str) -> None:
        if name not in model.peft_config:
            raise ValueError(f"Adapter {name} not found.")
        model.active_adapter = name

    def _delete_adapter(name: str) -> None:
        del model.peft_config[name]

    def _create_adapter(model_arg, adapter_config, name: str) -> None:
        model_arg.peft_config[name] = MagicMock()
        model_arg.set_adapter(name)

    model.set_adapter.side_effect = _set_adapter
    model.delete_adapter.side_effect = _delete_adapter
    mock_create.side_effect = _create_adapter
    mock_switch.side_effect = lambda model_arg, name: model_arg.set_adapter(name)


def _plant_resume_checkpoint(
    out_dir: Path,
    dataset: list[dict],
    training_config: TrainingConfig,
    adapter_config: AdapterConfig,
) -> Path:
    """Leave an interrupted run's crash-resume state under *out_dir*.

    Creates ``checkpoint-10`` and a ``staging_resume.json`` whose fingerprints
    match *dataset* and the two configs, so ``train_adapter`` resumes from
    that checkpoint.

    Args:
        out_dir: The ``output_dir`` the test passes to ``train_adapter``.
        dataset: The training dataset the test passes.
        training_config: The training config the test passes.
        adapter_config: The adapter config the test passes.

    Returns:
        The checkpoint directory.
    """
    from paramem.training.trainer import _fingerprint_dataset, _fingerprint_training_config

    ckpt_dir = out_dir / "checkpoint-10"
    ckpt_dir.mkdir(parents=True)
    resume_state = {
        "adapter_name": "episodic",
        "dataset_fingerprint": _fingerprint_dataset(dataset),
        "training_config_fingerprint": _fingerprint_training_config(
            training_config, adapter_config
        ),
        "disk_checkpoint_path": str(ckpt_dir),
    }
    (out_dir / "staging_resume.json").write_bytes(json.dumps(resume_state, indent=2).encode())
    return ckpt_dir


@contextmanager
def _failing_warm_start_copy(error: BaseException) -> Iterator[None]:
    """Make copying the production weights into the staging slot raise *error*."""
    with patch("paramem.models.loader.copy_adapter_weights", side_effect=error):
        yield


@contextmanager
def _failing_training_arguments(error: BaseException) -> Iterator[None]:
    """Make building ``TrainingArguments`` raise *error*."""
    with patch("paramem.training.trainer.TrainingArguments", side_effect=error):
        yield


@contextmanager
def _failing_trainer_construction(error: BaseException) -> Iterator[None]:
    """Make constructing ``ParamemTrainer`` raise *error*."""
    with patch("paramem.training.trainer.ParamemTrainer", side_effect=error):
        yield


@contextmanager
def _failing_resume_checkpoint_materialization(error: BaseException) -> Iterator[None]:
    """Report a daily identity as available and make decrypting the resume
    checkpoint raise *error*."""
    with (
        patch("paramem.backup.key_store.daily_identity_available", return_value=True),
        patch("paramem.backup.checkpoint_shard.materialize_checkpoint_to_shm", side_effect=error),
    ):
        yield


# ---------------------------------------------------------------------------
# TestStagingPromoteContract — staging slot lifecycle and promote semantics
# ---------------------------------------------------------------------------


class TestStagingPromoteContract:
    """Verify ``train_adapter``'s staging+promote contract without GPU.

    Staged invariants:
    1. ``in_training`` slot is created when absent (correct shape).
    2. A pre-existing ``in_training`` slot at entry trips the lifecycle guard
       — the prior training event's caller did not dispose of it.
    3. Production weights are copied to staging at entry.
    4. Normal completion leaves staging resident and active for the caller,
       and cleans scratch — ``train_adapter`` itself never promotes or
       deletes the slot; the caller promotes it with
       ``promote_staging_adapter`` and disposes of it with ``staged_weights``.
    5. Abort path: no promote; scratch cleaned.
    6. Crash path: scratch preserved (staging_resume.json + checkpoint).
    7. Crash-resume: staging_resume.json fingerprint match → checkpoint forwarded.
    8. Resume resolution: a recorded on-disk checkpoint pointer that still
       exists is forwarded; otherwise resume is absent.
    """

    def test_staging_slot_created_at_entry_with_shape_match(self, tmp_path):
        """When 'in_training' is absent, train_adapter creates it via create_adapter."""
        model = _make_staging_model(has_staging=False)
        stack, mock_create, mock_copy, mock_switch = _staging_patches(tmp_path)
        with stack:
            train_adapter(
                model=model,
                tokenizer=MagicMock(),
                train_dataset=_minimal_dataset(),
                adapter_name="episodic",
                training_config=_minimal_tc(),
                adapter_config=_minimal_ac(),
                output_dir=tmp_path / "adapter",
            )

        # create_adapter must have been called with the staging adapter name.
        create_calls = [str(c) for c in mock_create.call_args_list]
        assert any("in_training" in s for s in create_calls), (
            f"Expected create_adapter called with 'in_training'; "
            f"calls: {mock_create.call_args_list}"
        )

    def test_staging_slot_pre_existing_raises_lifecycle_error(self, tmp_path):
        """Pre-existing 'in_training' at entry violates the staging lifecycle — RuntimeError."""
        import pytest

        # Staging is transient (created at training entry). On abort or an
        # exception, train_adapter drops it itself before returning or
        # re-raising; on normal completion it stays resident for the
        # caller's probe -> promote -> dispose sequence. If 'in_training'
        # is present at entry, the prior event's caller skipped that
        # sequence or a disposal failed and was only logged — a real bug
        # either way.
        model = _make_staging_model(has_staging=True)
        stack, mock_create, mock_copy, mock_switch = _staging_patches(tmp_path)
        with stack, pytest.raises(RuntimeError, match="Lifecycle invariant violated"):
            train_adapter(
                model=model,
                tokenizer=MagicMock(),
                train_dataset=_minimal_dataset(),
                adapter_name="episodic",
                training_config=_minimal_tc(),
                adapter_config=_minimal_ac(),
                output_dir=tmp_path / "adapter",
            )

    def test_second_call_without_caller_disposal_raises(self, tmp_path):
        """A second train_adapter call on the same model, without the caller
        having disposed of the first call's staged weights, raises the
        lifecycle-invariant RuntimeError.

        Kills: a caller obligation that is not actually enforced (e.g. the
        guard silently rebuilding the slot instead of refusing).
        """
        import pytest

        model = _make_staging_model(has_staging=False)
        stack, mock_create, mock_copy, mock_switch = _staging_patches(tmp_path)
        with stack:
            train_adapter(
                model=model,
                tokenizer=MagicMock(),
                train_dataset=_minimal_dataset(),
                adapter_name="episodic",
                training_config=_minimal_tc(),
                adapter_config=_minimal_ac(),
                output_dir=tmp_path / "adapter_1",
            )
            # No disposal here — the caller obligation is violated deliberately.
            with pytest.raises(RuntimeError, match="Lifecycle invariant violated"):
                train_adapter(
                    model=model,
                    tokenizer=MagicMock(),
                    train_dataset=_minimal_dataset(),
                    adapter_name="episodic",
                    training_config=_minimal_tc(),
                    adapter_config=_minimal_ac(),
                    output_dir=tmp_path / "adapter_2",
                )

        # The refusal leaves the first call's staged weights resident for
        # that call's caller: the teardown covers only a slot this call made.
        assert "in_training" in model.peft_config
        assert call("in_training") not in model.delete_adapter.call_args_list
        mock_switch.assert_not_called()

    def test_staging_active_at_normal_completion(self, tmp_path):
        """On normal completion, 'in_training' is the active adapter — the caller
        probes it directly by name, with no extra switch of its own.

        Kills: leaving some other adapter active after training returns.
        """
        model = _make_staging_model(has_staging=False)
        stack, mock_create, mock_copy, mock_switch = _staging_patches(tmp_path)
        with stack:
            train_adapter(
                model=model,
                tokenizer=MagicMock(),
                train_dataset=_minimal_dataset(),
                adapter_name="episodic",
                training_config=_minimal_tc(),
                adapter_config=_minimal_ac(),
                output_dir=tmp_path / "adapter",
            )

        assert model.set_adapter.call_args_list[-1] == call("in_training"), (
            f"Expected 'in_training' to be the last-activated adapter; "
            f"calls: {model.set_adapter.call_args_list}"
        )

    def test_exception_path_still_deletes_staging_slot(self, tmp_path):
        """The crash (exception) path leaves 'in_training' absent — the same
        disposal normal-path abort gets, but here it must survive being
        routed through a re-raise.

        Kills: an exception path that stops disposing of the staged slot
        (which would permanently block the next training event).
        """
        import pytest as _pytest

        model = _make_staging_model(has_staging=False)
        stack, _, _, _ = _staging_patches(tmp_path, trainer_cls=_RaisingTrainer)
        with stack, _pytest.raises(RuntimeError, match="simulated crash"):
            train_adapter(
                model=model,
                tokenizer=MagicMock(),
                train_dataset=_minimal_dataset(),
                adapter_name="episodic",
                training_config=_minimal_tc(),
                adapter_config=_minimal_ac(),
                output_dir=tmp_path / "adapter_crash",
            )
        assert call("in_training") in model.delete_adapter.call_args_list, (
            "Exception path must still delete the staging slot (best-effort)"
        )

    @pytest.mark.parametrize(
        "failing_step",
        [
            pytest.param(_failing_warm_start_copy, id="warm_start_copy"),
            pytest.param(_failing_training_arguments, id="training_arguments"),
            pytest.param(_failing_trainer_construction, id="trainer_construction"),
            pytest.param(
                _failing_resume_checkpoint_materialization,
                id="resume_checkpoint_materialization",
            ),
        ],
    )
    def test_failure_before_training_tears_down_staging_slot(self, tmp_path, failing_step):
        """An exception raised after the staging slot exists but before
        ``trainer.train()`` runs leaves the model as an exception inside
        training does: it propagates unchanged, the staging slot is deleted,
        the production adapter is active again, and the crash-resume scratch
        survives. The next call therefore trains instead of tripping the
        lifecycle guard.

        Kills: a teardown that covers only ``trainer.train()``.
        """
        out_dir = tmp_path / "adapter"
        dataset = _minimal_dataset()
        training_config = _minimal_tc()
        adapter_config = _minimal_ac()
        ckpt_dir = _plant_resume_checkpoint(out_dir, dataset, training_config, adapter_config)

        # Trained production weights, so the warm-start copy into the slot runs.
        model = _make_staging_model(has_staging=False, production_warm=True)
        stack, mock_create, _, mock_switch = _staging_patches(tmp_path)
        _track_adapter_state(model, mock_create, mock_switch)
        injected = RuntimeError("injected failure before training")

        def _train() -> dict:
            return train_adapter(
                model=model,
                tokenizer=MagicMock(),
                train_dataset=dataset,
                adapter_name="episodic",
                training_config=training_config,
                adapter_config=adapter_config,
                output_dir=out_dir,
            )

        with stack:
            with failing_step(injected), pytest.raises(RuntimeError) as raised:
                _train()

            assert raised.value is injected, "the original exception must propagate"
            assert "in_training" not in model.peft_config, "the staging slot must be deleted"
            assert model.active_adapter == "episodic", "the production adapter must be active"
            assert (out_dir / "staging_resume.json").is_file(), "the resume marker must survive"
            assert ckpt_dir.is_dir(), "the resume checkpoint must survive"

            metrics = _train()

        assert metrics["aborted"] is False
        assert "in_training" in model.peft_config
        assert model.active_adapter == "in_training"

    def test_staging_survives_at_normal_completion(self, tmp_path):
        """On normal completion, 'in_training' is left resident — the caller owns disposal.

        Kills: putting the promote (and the staging delete that used to
        follow it) back inside the trainer.
        """
        model = _make_staging_model(has_staging=False)
        stack, mock_create, mock_copy, mock_switch = _staging_patches(tmp_path)
        with stack:
            train_adapter(
                model=model,
                tokenizer=MagicMock(),
                train_dataset=_minimal_dataset(),
                adapter_name="episodic",
                training_config=_minimal_tc(),
                adapter_config=_minimal_ac(),
                output_dir=tmp_path / "adapter",
            )

        assert "in_training" not in [c.args[0] for c in model.delete_adapter.call_args_list], (
            "in_training must NOT be deleted by train_adapter on normal completion"
        )
        assert "in_training" in model.peft_config, (
            "in_training must remain resident for the caller to probe and promote"
        )

    def test_staging_deleted_at_abort(self, tmp_path):
        """On abort, model.delete_adapter('in_training') is called."""
        model = _make_staging_model(has_staging=False)
        stack, mock_create, mock_copy, mock_switch = _staging_patches(tmp_path, abort_shutdown=True)
        hooks = TrainingHooks(on_shutdown_check=lambda: True)
        with stack:
            train_adapter(
                model=model,
                tokenizer=MagicMock(),
                train_dataset=_minimal_dataset(),
                adapter_name="episodic",
                training_config=_minimal_tc(),
                adapter_config=_minimal_ac(),
                output_dir=tmp_path / "adapter",
                hooks=hooks,
            )

        model.delete_adapter.assert_called_with("in_training")

    def test_two_sequential_calls_do_not_trip_lifecycle_guard(self, tmp_path):
        """Multi-tier consolidation safety: episodic→semantic in the same process.

        Under the caller-owns-disposal contract, the caller MUST run its
        probe/promote sequence inside ``staged_weights`` between successive
        ``train_adapter`` calls on the same model — this is the caller
        obligation the lifecycle guard (``assert_staging_absent``) enforces.
        This test disposes via ``staged_weights``/``promote_staging_adapter``
        after each call and asserts the second call does not trip the guard.

        Kills: softening the lifecycle backstop to silently rebuild instead
        of raising when a caller fails to dispose.
        """
        from paramem.training.trainer import promote_staging_adapter, staged_weights

        # The mock's peft_config is a dict; treat delete_adapter as a real mutation
        # so the second train_adapter call sees an absent in_training slot.
        model = _make_staging_model(has_staging=False)

        def _delete_from_peft_config(name):
            model.peft_config.pop(name, None)

        model.delete_adapter.side_effect = _delete_from_peft_config

        stack, mock_create, mock_copy, mock_switch = _staging_patches(tmp_path)

        with stack:
            for tier in ("episodic", "semantic"):
                train_adapter(
                    model=model,
                    tokenizer=MagicMock(),
                    train_dataset=_minimal_dataset(),
                    adapter_name=tier,
                    training_config=_minimal_tc(),
                    adapter_config=_minimal_ac(),
                    output_dir=tmp_path / f"adapter_{tier}",
                )
                # Caller obligation: probe/promote/dispose before the next
                # training event reuses the staging slot.
                with staged_weights(model, fallback_adapter=tier):
                    promote_staging_adapter(model, tier)

        # The slot must be absent at the end of the sequence.
        assert "in_training" not in model.peft_config, (
            "in_training must be deleted after every caller disposal"
        )

    def test_production_weights_copied_to_staging_at_entry(self, tmp_path):
        """A production adapter with prior trained weights (measures warm)
        gets copied into the staging slot at entry —
        copy_adapter_weights(src='episodic', dst='in_training') — and
        train_adapter itself never issues the reverse (promote) copy; that
        copy belongs to the caller's promote_staging_adapter call.

        Kills: putting the promote back inside the trainer, and dropping
        the warm-start entry copy for a trained production adapter.
        """
        model = _make_staging_model(has_staging=False, production_warm=True)
        stack, mock_create, mock_copy, mock_switch = _staging_patches(tmp_path)
        with stack:
            train_adapter(
                model=model,
                tokenizer=MagicMock(),
                train_dataset=_minimal_dataset(),
                adapter_name="episodic",
                training_config=_minimal_tc(),
                adapter_config=_minimal_ac(),
                output_dir=tmp_path / "adapter",
            )

        entry_copy = call(model, src="episodic", dst="in_training")
        promote_copy = call(model, src="in_training", dst="episodic")
        all_calls = mock_copy.call_args_list
        assert all_calls == [entry_copy], (
            f"Expected exactly the entry copy (episodic → in_training) and no "
            f"promote copy; got {all_calls}"
        )
        assert promote_copy not in all_calls

    def test_normal_completion_leaves_promote_to_caller(self, tmp_path):
        """On normal completion, promote_staging_adapter (called by the caller,
        not by train_adapter) is what copies staging into production and
        switches the active adapter.

        Kills: reintroducing an internal promote inside train_adapter, and
        kills promote_staging_adapter itself skipping either the copy or the
        switch.
        """
        from paramem.training.trainer import promote_staging_adapter

        model = _make_staging_model(has_staging=False)
        stack, mock_create, mock_copy, mock_switch = _staging_patches(tmp_path)
        with stack:
            train_adapter(
                model=model,
                tokenizer=MagicMock(),
                train_dataset=_minimal_dataset(),
                adapter_name="episodic",
                training_config=_minimal_tc(),
                adapter_config=_minimal_ac(),
                output_dir=tmp_path / "adapter",
            )
            # train_adapter itself must not have promoted.
            promote_copy = call(model, src="in_training", dst="episodic")
            assert promote_copy not in mock_copy.call_args_list

            promote_staging_adapter(model, "episodic")

        assert promote_copy in mock_copy.call_args_list, (
            f"Expected promote copy (in_training → episodic); got {mock_copy.call_args_list}"
        )
        # Active adapter must switch back to production.
        mock_switch.assert_called_with(model, "episodic")

    def test_abort_does_not_promote_and_production_unchanged(self, tmp_path):
        """When aborted, staging weights are NOT promoted; production slot unchanged."""
        model = _make_staging_model(has_staging=False)
        stack, mock_create, mock_copy, mock_switch = _staging_patches(tmp_path, abort_shutdown=True)
        hooks = TrainingHooks(on_shutdown_check=lambda: True)
        with stack:
            metrics = train_adapter(
                model=model,
                tokenizer=MagicMock(),
                train_dataset=_minimal_dataset(),
                adapter_name="episodic",
                training_config=_minimal_tc(),
                adapter_config=_minimal_ac(),
                output_dir=tmp_path / "adapter",
                hooks=hooks,
            )

        assert metrics.get("aborted") is True, f"Expected aborted=True; got {metrics}"
        # Promote copy must NOT have happened.
        promote_copy = call(model, src="in_training", dst="episodic")
        assert promote_copy not in mock_copy.call_args_list, (
            "Abort path must not promote staging weights to production"
        )
        # Active adapter must have been restored to production on abort.
        mock_switch.assert_called_with(model, "episodic")

    def test_crash_preserves_scratch_for_resume(self, tmp_path):
        """When trainer.train() raises, staging_resume.json is NOT deleted."""
        model = _make_staging_model(has_staging=False)
        out_dir = tmp_path / "adapter"
        out_dir.mkdir(parents=True, exist_ok=True)

        stack, mock_create, mock_copy, mock_switch = _staging_patches(
            tmp_path, trainer_cls=_RaisingTrainer
        )
        import pytest as _pytest

        with stack, _pytest.raises(RuntimeError, match="simulated crash"):
            train_adapter(
                model=model,
                tokenizer=MagicMock(),
                train_dataset=_minimal_dataset(),
                adapter_name="episodic",
                training_config=_minimal_tc(),
                adapter_config=_minimal_ac(),
                output_dir=out_dir,
            )

        # staging_resume.json must exist (not cleaned on crash).
        scratch = out_dir / "staging_resume.json"
        assert scratch.exists(), (
            "staging_resume.json must be preserved after crash for crash-resume"
        )

    def test_resume_reads_scratch_and_passes_resume_from_checkpoint(self, tmp_path):
        """When staging_resume.json fingerprints match and a checkpoint dir exists,
        train_adapter resolves the checkpoint and passes it to trainer.train()."""
        out_dir = tmp_path / "adapter"
        ds = _minimal_dataset()
        tc = _minimal_tc()
        ac = _minimal_ac()
        ckpt_dir = _plant_resume_checkpoint(out_dir, ds, tc, ac)

        # Capture the resume_from_checkpoint kwarg passed to trainer.train().
        captured_resume: list = []

        class _CapturingResumeTrainer(_NullTrainer):
            def train(self, resume_from_checkpoint=None):
                captured_resume.append(resume_from_checkpoint)
                result = MagicMock()
                result.metrics = {"train_loss": 0.01}
                return result

        model = _make_staging_model(has_staging=False)
        stack, mock_create, mock_copy, mock_switch = _staging_patches(
            tmp_path, trainer_cls=_CapturingResumeTrainer
        )
        with stack:
            train_adapter(
                model=model,
                tokenizer=MagicMock(),
                train_dataset=ds,
                adapter_name="episodic",
                training_config=tc,
                adapter_config=ac,
                output_dir=out_dir,
            )

        assert len(captured_resume) == 1, "train() must be called exactly once"
        assert captured_resume[0] == str(ckpt_dir), (
            f"Expected resume_from_checkpoint={ckpt_dir!r}; got {captured_resume[0]!r}"
        )


# ---------------------------------------------------------------------------
# TrainingArguments construction
# ---------------------------------------------------------------------------


class TestTrainingArgumentsConstruction:
    """``train_adapter`` builds ``TrainingArguments`` with the caller's own
    ``output_dir``, reports to no tracker, and passes no run name."""

    def test_output_dir_report_to_none_and_no_run_name(self, tmp_path):
        model = _make_staging_model(has_staging=False)
        out_dir = tmp_path / "adapter"
        stack, mock_create, mock_copy, mock_switch = _staging_patches(tmp_path)
        with (
            stack,
            patch(
                "paramem.training.trainer.TrainingArguments", return_value=MagicMock()
            ) as mock_training_args,
        ):
            train_adapter(
                model=model,
                tokenizer=MagicMock(),
                train_dataset=_minimal_dataset(),
                adapter_name="episodic",
                training_config=_minimal_tc(),
                adapter_config=_minimal_ac(),
                output_dir=out_dir,
            )

        assert mock_training_args.call_count == 1
        kwargs = mock_training_args.call_args.kwargs
        assert kwargs["output_dir"] == str(out_dir)
        assert kwargs["report_to"] == "none"
        assert "run_name" not in kwargs


# ---------------------------------------------------------------------------
# Staging slot starting-weights outcome (warm / donor / cold)
# ---------------------------------------------------------------------------


class TestStagingInitOutcome:
    """``train_adapter``'s staging-slot starting-weights decision, recorded
    as ``metrics["init"]``:

    | prior trained weights | staging starts from | ``init`` |
    |---|---|---|
    | yes | copy of *adapter_name* (warm) | ``"warm"`` |
    | no, donor_checkpoint_dir set | copy of the donor checkpoint | ``"donor"`` |
    | no, no valid donor | LoRA-zero | ``"cold"`` |
    """

    def test_prior_trained_weights_starts_warm(self, tmp_path):
        model = _make_staging_model(has_staging=False, production_warm=True)
        stack, mock_create, mock_copy, mock_switch = _staging_patches(tmp_path)
        with stack:
            metrics = train_adapter(
                model=model,
                tokenizer=MagicMock(),
                train_dataset=_minimal_dataset(),
                adapter_name="episodic",
                training_config=_minimal_tc(),
                adapter_config=_minimal_ac(),
                output_dir=tmp_path / "adapter",
            )

        assert metrics["init"] == "warm"

    def test_no_prior_weights_and_donor_given_starts_from_donor(self, tmp_path):
        model = _make_staging_model(has_staging=False, production_warm=False)
        stack, mock_create, mock_copy, mock_switch = _staging_patches(tmp_path)
        donor_dir = tmp_path / "donor"
        donor_dir.mkdir()
        with (
            stack,
            patch(
                "paramem.training.donor.load_donor_into_transient_slot", return_value=None
            ) as mock_load_donor,
        ):
            metrics = train_adapter(
                model=model,
                tokenizer=MagicMock(),
                train_dataset=_minimal_dataset(),
                adapter_name="episodic",
                training_config=_minimal_tc(),
                adapter_config=_minimal_ac(),
                output_dir=tmp_path / "adapter",
                donor_checkpoint_dir=donor_dir,
            )

        assert metrics["init"] == "donor"
        mock_load_donor.assert_called_once()

    def test_no_prior_weights_and_no_donor_starts_cold(self, tmp_path):
        model = _make_staging_model(has_staging=False, production_warm=False)
        stack, mock_create, mock_copy, mock_switch = _staging_patches(tmp_path)
        with stack:
            metrics = train_adapter(
                model=model,
                tokenizer=MagicMock(),
                train_dataset=_minimal_dataset(),
                adapter_name="episodic",
                training_config=_minimal_tc(),
                adapter_config=_minimal_ac(),
                output_dir=tmp_path / "adapter",
            )

        assert metrics["init"] == "cold"

    def test_donor_load_failure_degrades_to_cold(self, tmp_path):
        """A donor that fails to load costs only the seed — training starts
        cold rather than the fold failing."""
        model = _make_staging_model(has_staging=False, production_warm=False)
        stack, mock_create, mock_copy, mock_switch = _staging_patches(tmp_path)
        donor_dir = tmp_path / "donor"
        donor_dir.mkdir()
        with (
            stack,
            patch(
                "paramem.training.donor.load_donor_into_transient_slot",
                side_effect=OSError("donor checkpoint unreadable"),
            ),
        ):
            metrics = train_adapter(
                model=model,
                tokenizer=MagicMock(),
                train_dataset=_minimal_dataset(),
                adapter_name="episodic",
                training_config=_minimal_tc(),
                adapter_config=_minimal_ac(),
                output_dir=tmp_path / "adapter",
                donor_checkpoint_dir=donor_dir,
            )

        assert metrics["init"] == "cold"

    def test_init_recorded_on_abort_too(self, tmp_path):
        """``metrics["init"]`` is set on every non-exception return, not only
        on normal completion."""
        model = _make_staging_model(has_staging=False, production_warm=False)
        stack, mock_create, mock_copy, mock_switch = _staging_patches(tmp_path, abort_shutdown=True)
        hooks = TrainingHooks(on_shutdown_check=lambda: True)
        with stack:
            metrics = train_adapter(
                model=model,
                tokenizer=MagicMock(),
                train_dataset=_minimal_dataset(),
                adapter_name="episodic",
                training_config=_minimal_tc(),
                adapter_config=_minimal_ac(),
                output_dir=tmp_path / "adapter",
                hooks=hooks,
            )

        assert metrics.get("aborted") is True
        assert metrics["init"] == "cold"


# ---------------------------------------------------------------------------
# retain_scratch_until_external_commit flag
# ---------------------------------------------------------------------------


class TestRetainScratchFlag:
    """Verify ``retain_scratch_until_external_commit`` semantics.

    On NORMAL completion, when True:
    - ``checkpoint-N`` dir under ``output_dir`` survives.
    - ``staging_resume.json`` survives.
    - ``in_training`` stays resident and active — the flag governs on-disk
      scratch only, never the in-VRAM staging slot's caller-owned lifecycle.

    When False (default), both are cleaned.

    The abort branch honours the same flag: when True, ``checkpoint-N`` and
    ``staging_resume.json`` survive an abort exactly as they do on normal
    completion, so a subsequent ``train_adapter`` call against the same
    dataset resumes from the last epoch checkpoint instead of restarting the
    tier. When False, abort cleans scratch immediately. In both cases the
    transient ``in_training`` staging slot is deleted on abort
    unconditionally — the flag never governs the in-VRAM slot's lifecycle.
    """

    def _run_train(self, tmp_path, *, retain: bool, trainer_cls=None) -> dict:
        """Helper: run train_adapter with controlled staging patches and return metrics.

        The Trainer writes ``checkpoint-10`` during ``train()`` (not before), so
        that the fresh-start purge (which runs at the top of the fresh-start
        branch, before training) never sees it as a stale checkpoint from a prior
        crashed run.  This accurately models the production flow: HF Trainer
        creates checkpoint dirs DURING training; the retain flag governs whether
        they survive the post-training ``_clean_scratch`` call.
        """
        model = _make_staging_model(has_staging=False)
        out_dir = tmp_path / "adapter"
        out_dir.mkdir(parents=True, exist_ok=True)

        if trainer_cls is None:
            trainer_cls = _make_checkpoint_writing_trainer(out_dir)

        stack, mock_create, mock_copy, mock_switch = _staging_patches(
            tmp_path, trainer_cls=trainer_cls
        )
        with stack:
            metrics = train_adapter(
                model=model,
                tokenizer=MagicMock(),
                train_dataset=_minimal_dataset(),
                adapter_name="episodic",
                training_config=_minimal_tc(),
                adapter_config=_minimal_ac(),
                output_dir=out_dir,
                retain_scratch_until_external_commit=retain,
            )
        return metrics, out_dir, model, mock_copy, mock_switch

    def test_retain_false_default_cleans_checkpoint_and_scratch(self, tmp_path):
        """Default (retain=False): checkpoint-N and staging_resume.json are deleted on success."""
        metrics, out_dir, model, mock_copy, mock_switch = self._run_train(tmp_path, retain=False)

        assert not metrics.get("aborted"), f"Expected success; got {metrics}"
        assert not (out_dir / "checkpoint-10").exists(), (
            "checkpoint-10 must be deleted on success with retain=False (default)"
        )
        assert not (out_dir / "staging_resume.json").exists(), (
            "staging_resume.json must be deleted on success with retain=False (default)"
        )

    def test_retain_true_keeps_checkpoint_and_scratch(self, tmp_path):
        """retain=True: checkpoint-N and staging_resume.json survive a successful train_adapter."""
        metrics, out_dir, model, mock_copy, mock_switch = self._run_train(tmp_path, retain=True)

        assert not metrics.get("aborted"), f"Expected success; got {metrics}"
        assert (out_dir / "checkpoint-10").exists(), (
            "checkpoint-10 must survive when retain_scratch_until_external_commit=True"
        )
        assert (out_dir / "staging_resume.json").exists(), (
            "staging_resume.json must survive when retain_scratch_until_external_commit=True"
        )

    def test_retain_true_still_leaves_staging_resident(self, tmp_path):
        """retain=True does not change the in-VRAM staging contract: 'in_training'
        stays resident for the caller, exactly as it does with retain=False.
        """
        metrics, out_dir, model, mock_copy, mock_switch = self._run_train(tmp_path, retain=True)

        assert "in_training" not in [c.args[0] for c in model.delete_adapter.call_args_list], (
            "in_training must NOT be deleted by train_adapter regardless of retain"
        )
        assert "in_training" in model.peft_config

    def test_retain_true_survives_abort(self, tmp_path):
        """retain=True: checkpoint-N and staging_resume.json survive an
        aborted train_adapter call, while the transient in_training staging
        slot is still deleted (the flag governs on-disk scratch only, never
        the in-VRAM slot's lifecycle).
        """
        model = _make_staging_model(has_staging=False)
        out_dir = tmp_path / "adapter_abort"
        out_dir.mkdir(parents=True, exist_ok=True)

        trainer_cls = _make_checkpoint_writing_trainer(out_dir)
        stack, mock_create, mock_copy, mock_switch = _staging_patches(
            tmp_path, trainer_cls=trainer_cls
        )
        hooks = TrainingHooks(on_shutdown_check=lambda: True)
        with stack:
            metrics = train_adapter(
                model=model,
                tokenizer=MagicMock(),
                train_dataset=_minimal_dataset(),
                adapter_name="episodic",
                training_config=_minimal_tc(),
                adapter_config=_minimal_ac(),
                output_dir=out_dir,
                hooks=hooks,
                retain_scratch_until_external_commit=True,
            )

        assert metrics.get("aborted") is True
        assert (out_dir / "checkpoint-10").exists(), (
            "checkpoint-10 must survive an abort when retain_scratch_until_external_commit=True"
        )
        assert (out_dir / "staging_resume.json").exists(), (
            "staging_resume.json must survive an abort when "
            "retain_scratch_until_external_commit=True"
        )
        assert call("in_training") in model.delete_adapter.call_args_list, (
            "in_training staging slot must still be deleted on abort regardless of the retain flag"
        )

    def test_retain_false_default_cleans_checkpoint_and_scratch_on_abort(self, tmp_path):
        """Default (retain=False): checkpoint-N and staging_resume.json are
        deleted on an aborted train_adapter call, exactly as they are on a
        successful one — between success and abort, the flag decides
        whether on-disk scratch survives. (A raised exception is a third
        outcome, not covered by this test: it always keeps scratch, no
        matter what the flag is set to.)
        """
        model = _make_staging_model(has_staging=False)
        out_dir = tmp_path / "adapter_abort_default"
        out_dir.mkdir(parents=True, exist_ok=True)

        trainer_cls = _make_checkpoint_writing_trainer(out_dir)
        stack, mock_create, mock_copy, mock_switch = _staging_patches(
            tmp_path, trainer_cls=trainer_cls
        )
        hooks = TrainingHooks(on_shutdown_check=lambda: True)
        with stack:
            metrics = train_adapter(
                model=model,
                tokenizer=MagicMock(),
                train_dataset=_minimal_dataset(),
                adapter_name="episodic",
                training_config=_minimal_tc(),
                adapter_config=_minimal_ac(),
                output_dir=out_dir,
                hooks=hooks,
            )

        assert metrics.get("aborted") is True
        assert not (out_dir / "checkpoint-10").exists(), (
            "checkpoint-10 must be deleted on abort with retain=False (default)"
        )
        assert not (out_dir / "staging_resume.json").exists(), (
            "staging_resume.json must be deleted on abort with retain=False (default)"
        )
        assert call("in_training") in model.delete_adapter.call_args_list, (
            "in_training staging slot must be deleted on abort"
        )

    def test_retain_true_abort_then_resume_uses_surviving_checkpoint(self, tmp_path):
        """A second train_adapter call against the same dataset, after an
        aborted call with retain=True, resumes from the checkpoint the
        abort left behind via staging_resume.json / _resolve_resume_checkpoint.
        """
        model = _make_staging_model(has_staging=False)

        def _delete_from_peft_config(name):
            model.peft_config.pop(name, None)

        model.delete_adapter.side_effect = _delete_from_peft_config

        out_dir = tmp_path / "adapter"
        out_dir.mkdir(parents=True, exist_ok=True)
        ds = _minimal_dataset()
        tc = _minimal_tc()
        ac = _minimal_ac()

        # First call: aborts, retains scratch, and records checkpoint-10 via
        # on_save — mirroring HF Trainer's real checkpoint-write dispatch.
        abort_trainer_cls = _make_checkpoint_and_save_trainer(out_dir)
        stack1, _, _, _ = _staging_patches(tmp_path, trainer_cls=abort_trainer_cls)
        hooks = TrainingHooks(on_shutdown_check=lambda: True)
        with stack1:
            metrics = train_adapter(
                model=model,
                tokenizer=MagicMock(),
                train_dataset=ds,
                adapter_name="episodic",
                training_config=tc,
                adapter_config=ac,
                output_dir=out_dir,
                hooks=hooks,
                retain_scratch_until_external_commit=True,
            )
        assert metrics.get("aborted") is True

        # Second call: same dataset/config, no abort — must resume from the
        # checkpoint the first call left behind.
        captured_resume: list = []

        class _CapturingResumeTrainer(_NullTrainer):
            def train(self, resume_from_checkpoint=None):
                captured_resume.append(resume_from_checkpoint)
                result = MagicMock()
                result.metrics = {"train_loss": 0.01}
                return result

        stack2, _, _, _ = _staging_patches(tmp_path, trainer_cls=_CapturingResumeTrainer)
        with stack2:
            train_adapter(
                model=model,
                tokenizer=MagicMock(),
                train_dataset=ds,
                adapter_name="episodic",
                training_config=tc,
                adapter_config=ac,
                output_dir=out_dir,
            )

        assert len(captured_resume) == 1, "train() must be called exactly once"
        assert captured_resume[0] == str(out_dir / "checkpoint-10"), (
            f"Expected resume from the surviving checkpoint; got {captured_resume[0]!r}"
        )
