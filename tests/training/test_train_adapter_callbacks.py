"""Verify ``train_adapter`` callback assembly order and staging+promote contract.

Inference yielding (via ``TrainingHooks.on_step_yield``) MUST run before the
thermal throttle so a yield request pre-empts a throttle wait within the
same step. The order is locked by registration order in ``train_adapter``;
this test asserts it by inspecting the constructed list directly.

The ``TestStagingPromoteContract`` class verifies the staging+promote invariants:
- staging slot created/reshaped at entry
- production weights copied to staging at entry
- normal completion promotes staging → production and cleans scratch
- abort path does NOT promote and still cleans scratch
- crash path preserves scratch for crash-resume
- 3-way resume resolution (RAM → disk → absent)

No GPU required: the test patches ``Trainer``, PEFT, and encryption helpers so
staging+promote logic runs without a real training run.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, call, patch

from transformers import TrainerCallback

from paramem.training.thermal_throttle import ThermalPolicy, ThermalThrottleCallback
from paramem.training.trainer import (
    LossEarlyStoppingCallback,
    TrainingHooks,
    _HooksAdapterCallback,
    train_adapter,
)
from paramem.utils.config import AdapterConfig, TrainingConfig


class _MarkerCallback(TrainerCallback):
    """Sentinel for the call-bound ``callbacks_extra`` slot."""


def _capture_callbacks(**train_adapter_kwargs):
    """Run ``train_adapter`` with ``Trainer`` mocked; return the callbacks list.

    Builds a model that satisfies the staging+promote contract end-to-end —
    both production and staging slots present with matching shape, and
    ``named_parameters`` returns one real tensor per ``(target_module, slot)``
    pair so ``copy_adapter_weights`` runs cleanly at entry and at promote.
    Tests here cover callback assembly, not staging behaviour itself.
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
    # trip the per-AD-20 lifecycle-invariant guard.  Callback-ordering tests do
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
        patch("paramem.training.trainer.Trainer", side_effect=_capture_init),
        patch(
            "paramem.training.trainer._FixedDecayTrainer",
            side_effect=_capture_init,
        ),
        # Bypass the AD-20 lifecycle guard for callback-ordering tests.
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

    def test_hooks_before_throttle(self):
        # The load-bearing invariant: when both hooks and throttle install,
        # _HooksAdapterCallback must appear BEFORE ThermalThrottleCallback in
        # registration order so on_step_yield runs first at every step.
        hooks = TrainingHooks(on_step_yield=lambda step: None)
        cbs = _capture_callbacks(hooks=hooks, thermal_policy=self._policy())
        idx_hooks = next(i for i, cb in enumerate(cbs) if isinstance(cb, _HooksAdapterCallback))
        idx_throttle = next(
            i for i, cb in enumerate(cbs) if isinstance(cb, ThermalThrottleCallback)
        )
        assert idx_hooks < idx_throttle, (
            "Inference yielding (HooksAdapterCallback) must run before "
            "ThermalThrottleCallback so on_step_yield pre-empts throttle waits."
        )

    def test_loss_early_stop_when_enabled(self):
        cfg = TrainingConfig(early_stopping=True)
        cbs = _capture_callbacks(training_config=cfg)
        types = [type(cb).__name__ for cb in cbs]
        assert "LossEarlyStoppingCallback" in types
        # LossEarlyStoppingCallback is registered immediately after Encrypt.
        assert types.index("LossEarlyStoppingCallback") == 1

    def test_extra_callbacks_trail(self):
        marker = _MarkerCallback()
        hooks = TrainingHooks(on_step_yield=lambda step: None)
        cbs = _capture_callbacks(
            hooks=hooks,
            thermal_policy=self._policy(),
            callbacks_extra=[marker],
        )
        # marker must be the last entry — call-bound callbacks (e.g. recall
        # probe) are assembled after every cross-cutting concern.
        assert cbs[-1] is marker

    def test_full_assembly_order(self):
        # All slots populated → exact expected order:
        # Encrypt → LossEarlyStop → HooksAdapter → ThermalThrottle →
        # _StagingResumeCallback → marker.
        cfg = TrainingConfig(early_stopping=True)
        hooks = TrainingHooks(on_step_yield=lambda step: None)
        marker = _MarkerCallback()
        cbs = _capture_callbacks(
            training_config=cfg,
            hooks=hooks,
            thermal_policy=self._policy(),
            callbacks_extra=[marker],
        )
        types = [type(cb).__name__ for cb in cbs]
        assert types == [
            "EncryptCheckpointCallback",
            "LossEarlyStoppingCallback",
            "_HooksAdapterCallback",
            "ThermalThrottleCallback",
            "_StagingResumeCallback",
            "_MarkerCallback",
        ]


class TestHooksAdapterCallbackBehaviour:
    def test_step_yield_invoked_at_step_end(self):
        seen = []
        hooks = TrainingHooks(on_step_yield=lambda step: seen.append(step))
        cb = _HooksAdapterCallback(hooks)
        state = MagicMock(global_step=42)
        cb.on_step_end(args=MagicMock(), state=state, control=MagicMock())
        assert seen == [42]

    def test_epoch_persist_invoked_at_epoch_end(self):
        """on_epoch_persist receives (global_step, output_dir) at epoch end.

        Both on_epoch_end and on_save normalize to state.global_step so that
        the BackgroundTrainer dedup key matches across both callbacks at every
        epoch boundary.  The stored value is always the step count.
        """
        seen = []
        hooks = TrainingHooks(
            on_epoch_persist=lambda step, output_dir: seen.append((step, output_dir))
        )
        cb = _HooksAdapterCallback(hooks)
        args = MagicMock(output_dir="/tmp/test")
        state = MagicMock(epoch=3.0, global_step=1500)
        cb.on_epoch_end(args=args, state=state, control=MagicMock())
        assert seen == [(1500, "/tmp/test")]

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

    def test_on_save_invokes_persist_with_global_step(self):
        """on_save_persist receives (global_step, output_dir) from on_save."""
        seen: list = []
        hooks = TrainingHooks(on_save_persist=lambda step, d: seen.append((step, d)))
        cb = _HooksAdapterCallback(hooks)
        cb.on_save(
            args=MagicMock(output_dir="/tmp/x"),
            state=MagicMock(global_step=87),
            control=MagicMock(),
        )
        assert seen == [(87, "/tmp/x")]

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

    def test_step_end_yield_runs_before_shutdown_check(self):
        """on_step_yield fires before on_shutdown_check in on_step_end."""
        order: list[str] = []

        def _yield(step: int) -> None:
            order.append("yield")

        def _shutdown() -> bool:
            order.append("shutdown")
            return False

        hooks = TrainingHooks(on_step_yield=_yield, on_shutdown_check=_shutdown)
        cb = _HooksAdapterCallback(hooks)
        cb.on_step_end(
            args=MagicMock(),
            state=MagicMock(global_step=5),
            control=MagicMock(),
        )
        assert order == ["yield", "shutdown"], (
            f"on_step_yield must run before on_shutdown_check; got {order}"
        )


# Touch LossEarlyStoppingCallback so the import is exercised (no behavior test
# needed — its existing tests in tests/test_trainer_callbacks.py cover behaviour).
def test_loss_early_stopping_class_importable():
    assert LossEarlyStoppingCallback is not None


# ---------------------------------------------------------------------------
# Helpers shared by TestStagingPromoteContract
# ---------------------------------------------------------------------------


def _make_staging_model(
    *,
    has_staging: bool = False,
    staging_rank: int = 4,
    staging_modules: tuple[str, ...] = ("q_proj",),
) -> MagicMock:
    """Return a MagicMock PeftModel for staging+promote tests.

    ``peft_config`` starts with ``"episodic"`` (production tier).  The
    ``"in_training"`` staging slot is pre-populated only when
    ``has_staging=True``.

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
    model.named_parameters.return_value = []
    model.parameters.return_value = []
    return model


def _minimal_tc(**overrides) -> TrainingConfig:
    """Return a minimal ``TrainingConfig`` for staging tests (no RAM mode)."""
    cfg = TrainingConfig(
        num_epochs=1,
        batch_size=1,
        gradient_accumulation_steps=1,
        warmup_steps=0,
        warmup_ratio=0.0,
        lr_scheduler_type="linear",
        weight_decay=0.01,
        max_grad_norm=1.0,
        gradient_checkpointing=False,
        seed=42,
        save_strategy="no",
        save_total_limit=1,
        save_steps_ram=0,
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


def _staging_patches(tmp_path, *, trainer_cls=_NullTrainer, abort_shutdown=False):
    """Return a context-manager stack of patches for staging+promote tests.

    Patches:
    - TrainingArguments (no HF validation)
    - Trainer / _FixedDecayTrainer
    - paramem.models.loader.create_adapter (no real PEFT)
    - paramem.models.loader.copy_adapter_weights (no real tensor copy)
    - paramem.models.loader.switch_adapter (no real adapter activation)
    - paramem.backup.encryption.write_infra_bytes (plaintext write)
    - paramem.backup.encryption.read_maybe_encrypted (plaintext read)
    - paramem.backup.key_store.daily_identity_loadable (Security OFF)
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
    stack.enter_context(patch("paramem.training.trainer.Trainer", new=trainer_cls))
    stack.enter_context(patch("paramem.training.trainer._FixedDecayTrainer", new=trainer_cls))

    # --- Loader helpers (deferred import inside _ensure_staging_slot et al) ---
    mock_create = stack.enter_context(
        patch("paramem.models.loader.create_adapter", return_value=MagicMock())
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
        patch("paramem.backup.key_store.daily_identity_loadable", return_value=False)
    )

    # --- EncryptCheckpointCallback ---
    stack.enter_context(
        patch(
            "paramem.training.encrypted_checkpoint_callback.EncryptCheckpointCallback",
            MagicMock,
        )
    )

    return stack, mock_create, mock_copy, mock_switch


# ---------------------------------------------------------------------------
# TestStagingPromoteContract — staging slot lifecycle and promote semantics
# ---------------------------------------------------------------------------


class TestStagingPromoteContract:
    """Verify ``train_adapter``'s staging+promote contract without GPU.

    Staged invariants:
    1. ``in_training`` slot is created when absent (correct shape).
    2. ``in_training`` slot is deleted+recreated when target_modules change.
    3. Production weights are copied to staging at entry.
    4. Normal completion promotes staging → production and cleans scratch.
    5. Abort path: no promote; scratch cleaned.
    6. Crash path: scratch preserved (staging_resume.json + checkpoint).
    7. Crash-resume: staging_resume.json fingerprint match → checkpoint forwarded.
    8. 3-way resume preference: RAM first, then disk, then absent.
    9. Normal completion cleans staging_resume.json + bg_checkpoint_epoch.
    10. Abort completion cleans staging_resume.json + bg_checkpoint_epoch.
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
        """Pre-existing 'in_training' at entry violates AD-20 lifecycle — RuntimeError."""
        import pytest

        # Per AD-20, staging is transient: created at training entry, deleted at
        # exit (both success and abort paths). If 'in_training' is present at
        # entry, the prior training event failed to clean up — a real bug.
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

    def test_staging_deleted_at_normal_completion(self, tmp_path):
        """On normal completion, model.delete_adapter('in_training') is called."""
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

        model.delete_adapter.assert_called_with("in_training")

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

        The single load-bearing invariant of AD-20 is that the staging slot is
        DELETED on training exit so the next training event enters
        ``_ensure_staging_slot`` with a clean slate.  This test simulates the
        consolidation cycle's per-tier sequential ``train_adapter`` calls and
        asserts that the second call does not trip the lifecycle guard.
        """
        # The mock's peft_config is a dict; treat delete_adapter as a real mutation
        # so the second train_adapter call sees an absent in_training slot.
        model = _make_staging_model(has_staging=False)

        def _delete_from_peft_config(name):
            model.peft_config.pop(name, None)

        model.delete_adapter.side_effect = _delete_from_peft_config

        stack, mock_create, mock_copy, mock_switch = _staging_patches(tmp_path)

        # mock_create normally returns a fresh MagicMock; have it also add the
        # in_training key to peft_config so _ensure_staging_slot's first-time
        # path realistically updates state on each call.
        def _create_adds_slot(model_arg, cfg, name):
            model.peft_config[name] = cfg
            return model

        mock_create.side_effect = _create_adds_slot

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

        # The slot must be absent at the end of the sequence.
        assert "in_training" not in model.peft_config, (
            "in_training must be deleted after every training event"
        )

    def test_production_weights_copied_to_staging_at_entry(self, tmp_path):
        """copy_adapter_weights(src='episodic', dst='in_training') is called at entry."""
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

        # At entry: copy_adapter_weights(model, src="episodic", dst="in_training").
        # At normal completion: copy_adapter_weights(model, src="in_training", dst="episodic").
        # Both calls must be present; the entry copy must come first.
        entry_copy = call(model, src="episodic", dst="in_training")
        promote_copy = call(model, src="in_training", dst="episodic")
        all_calls = mock_copy.call_args_list
        assert entry_copy in all_calls, (
            f"Expected entry copy (episodic → in_training); got {all_calls}"
        )
        assert all_calls.index(entry_copy) < all_calls.index(promote_copy), (
            "Entry copy must precede promote copy"
        )

    def test_normal_completion_promotes_staging_to_production(self, tmp_path):
        """On normal completion, staging weights are promoted back to the production slot."""
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

        # Promote: copy_adapter_weights(model, src="in_training", dst="episodic").
        promote_copy = call(model, src="in_training", dst="episodic")
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
        out_dir.mkdir(parents=True, exist_ok=True)

        # Plant a real checkpoint directory (disk path).
        ckpt_dir = out_dir / "bg_checkpoint_epoch" / "checkpoint-10"
        ckpt_dir.mkdir(parents=True)

        # Compute the fingerprints that train_adapter will compute for the same dataset/config.
        from paramem.training.trainer import (
            _fingerprint_dataset,
            _fingerprint_training_config,
        )

        ds = _minimal_dataset()
        tc = _minimal_tc()
        ac = _minimal_ac()
        fp_ds = _fingerprint_dataset(ds)
        fp_cfg = _fingerprint_training_config(tc, ac)

        # Write a matching staging_resume.json.
        resume_state = {
            "adapter_name": "episodic",
            "dataset_fingerprint": fp_ds,
            "training_config_fingerprint": fp_cfg,
            "ram_checkpoint_path": "",
            "disk_checkpoint_path": str(ckpt_dir),
            "checkpoint_path": "",
            "started_at": "2026-05-27T00:00:00+00:00",
            "updated_at": "2026-05-27T00:00:00+00:00",
        }
        scratch = out_dir / "staging_resume.json"
        scratch.write_bytes(json.dumps(resume_state, indent=2).encode())

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

    def test_resume_3way_prefers_ram_then_disk(self, tmp_path):
        """3-way resume: RAM checkpoint is preferred over disk epoch-mirror."""
        out_dir = tmp_path / "adapter"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Create both a RAM and a disk checkpoint directory.
        ram_ckpt = tmp_path / "shm_fake" / "checkpoint-20"
        ram_ckpt.mkdir(parents=True)
        disk_ckpt = out_dir / "bg_checkpoint_epoch" / "checkpoint-10"
        disk_ckpt.mkdir(parents=True)

        from paramem.training.trainer import (
            _fingerprint_dataset,
            _fingerprint_training_config,
        )

        ds = _minimal_dataset()
        tc = _minimal_tc()
        ac = _minimal_ac()
        fp_ds = _fingerprint_dataset(ds)
        fp_cfg = _fingerprint_training_config(tc, ac)

        resume_state = {
            "adapter_name": "episodic",
            "dataset_fingerprint": fp_ds,
            "training_config_fingerprint": fp_cfg,
            "ram_checkpoint_path": str(ram_ckpt),
            "disk_checkpoint_path": str(disk_ckpt),
            "checkpoint_path": "",
            "started_at": "2026-05-27T00:00:00+00:00",
            "updated_at": "2026-05-27T00:00:00+00:00",
        }
        scratch = out_dir / "staging_resume.json"
        scratch.write_bytes(json.dumps(resume_state, indent=2).encode())

        captured_resume: list = []

        class _CapturingTrainer2(_NullTrainer):
            def train(self, resume_from_checkpoint=None):
                captured_resume.append(resume_from_checkpoint)
                result = MagicMock()
                result.metrics = {"train_loss": 0.01}
                return result

        model = _make_staging_model(has_staging=False)

        # --- Part 1: both RAM and disk exist → RAM wins ---
        stack, _, _, _ = _staging_patches(tmp_path, trainer_cls=_CapturingTrainer2)
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

        assert captured_resume[0] == str(ram_ckpt), (
            f"RAM checkpoint must be preferred; got {captured_resume[0]!r}"
        )

        # --- Part 2: remove RAM dir → disk wins ---
        import shutil

        shutil.rmtree(ram_ckpt)

        # Part 1's successful train_adapter completion ran _clean_scratch which
        # removed both the staging_resume.json AND the bg_checkpoint_epoch dir.
        # Recreate both so Part 2's _resolve_resume_checkpoint sees a valid
        # disk fallback (state present AND disk_checkpoint_path is_dir()).
        disk_ckpt.mkdir(parents=True, exist_ok=True)
        scratch.write_bytes(json.dumps(resume_state, indent=2).encode())
        captured_resume.clear()

        model2 = _make_staging_model(has_staging=False)
        stack2, _, _, _ = _staging_patches(tmp_path, trainer_cls=_CapturingTrainer2)
        with stack2:
            train_adapter(
                model=model2,
                tokenizer=MagicMock(),
                train_dataset=ds,
                adapter_name="episodic",
                training_config=tc,
                adapter_config=ac,
                output_dir=out_dir,
            )

        assert captured_resume[0] == str(disk_ckpt), (
            f"Disk checkpoint must be fallback when RAM absent; got {captured_resume[0]!r}"
        )

    def test_successful_completion_cleans_scratch(self, tmp_path):
        """After normal completion, staging_resume.json and bg_checkpoint_epoch are deleted."""
        out_dir = tmp_path / "adapter"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Pre-create scratch artifacts to verify they are cleaned.
        scratch = out_dir / "staging_resume.json"
        scratch.write_bytes(b"{}")
        epoch_mirror = out_dir / "bg_checkpoint_epoch" / "checkpoint-5"
        epoch_mirror.mkdir(parents=True)

        model = _make_staging_model(has_staging=False)
        stack, _, _, _ = _staging_patches(tmp_path, trainer_cls=_NullTrainer)
        with stack:
            train_adapter(
                model=model,
                tokenizer=MagicMock(),
                train_dataset=_minimal_dataset(),
                adapter_name="episodic",
                training_config=_minimal_tc(),
                adapter_config=_minimal_ac(),
                output_dir=out_dir,
            )

        assert not scratch.exists(), "staging_resume.json must be deleted on successful completion"
        assert not (out_dir / "bg_checkpoint_epoch").exists(), (
            "bg_checkpoint_epoch must be deleted on successful completion"
        )

    def test_abort_cleans_scratch(self, tmp_path):
        """After abort, staging_resume.json and bg_checkpoint_epoch are deleted."""
        out_dir = tmp_path / "adapter"
        out_dir.mkdir(parents=True, exist_ok=True)

        scratch = out_dir / "staging_resume.json"
        scratch.write_bytes(b"{}")
        epoch_mirror = out_dir / "bg_checkpoint_epoch" / "checkpoint-3"
        epoch_mirror.mkdir(parents=True)

        model = _make_staging_model(has_staging=False)
        stack, _, _, _ = _staging_patches(tmp_path, abort_shutdown=True)
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
        assert not scratch.exists(), "staging_resume.json must be deleted on abort"
        assert not (out_dir / "bg_checkpoint_epoch").exists(), (
            "bg_checkpoint_epoch must be deleted on abort"
        )
