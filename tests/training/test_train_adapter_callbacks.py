"""Verify ``train_adapter`` callback assembly order.

Inference yielding (via ``TrainingHooks.on_step_yield``) MUST run before the
thermal throttle so a yield request pre-empts a throttle wait within the
same step. The order is locked by registration order in ``train_adapter``;
this test asserts it by inspecting the constructed list directly.

No GPU required: the test patches ``Trainer`` so callback assembly happens
without a real training run.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

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
    """Run ``train_adapter`` with ``Trainer`` mocked; return the callbacks list."""
    captured = {}

    def _capture_init(*args, callbacks=None, **kwargs):
        captured["callbacks"] = callbacks
        instance = MagicMock()
        instance.train.return_value = MagicMock(metrics={"train_loss": 0.0})
        return instance

    with (
        patch("paramem.training.trainer.Trainer", side_effect=_capture_init),
        patch(
            "paramem.training.trainer._FixedDecayTrainer",
            side_effect=_capture_init,
        ),
    ):
        model = MagicMock()
        # set_adapter is called unconditionally; let the mock absorb it.
        train_adapter(
            model=model,
            tokenizer=MagicMock(),
            train_dataset=MagicMock(),
            adapter_name="episodic",
            training_config=train_adapter_kwargs.pop("training_config", TrainingConfig()),
            adapter_config=train_adapter_kwargs.pop("adapter_config", AdapterConfig()),
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
        # No hooks, no thermal_policy, no callbacks_extra → only the
        # always-installed encryption callback.
        cbs = _capture_callbacks()
        types = [type(cb).__name__ for cb in cbs]
        assert types == ["EncryptCheckpointCallback"]

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
        # Encrypt → LossEarlyStop → HooksAdapter → ThermalThrottle → marker.
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
        seen = []
        hooks = TrainingHooks(
            on_epoch_persist=lambda epoch, output_dir: seen.append((epoch, output_dir))
        )
        cb = _HooksAdapterCallback(hooks)
        args = MagicMock(output_dir="/tmp/test")
        state = MagicMock(epoch=3.0)
        cb.on_epoch_end(args=args, state=state, control=MagicMock())
        assert seen == [(3, "/tmp/test")]

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
        # Both event handlers must run without raising when intents are None.
        cb.on_step_end(args=MagicMock(), state=MagicMock(global_step=1), control=MagicMock())
        cb.on_epoch_end(args=MagicMock(), state=MagicMock(epoch=1.0), control=MagicMock())


# Touch LossEarlyStoppingCallback so the import is exercised (no behavior test
# needed — its existing tests in tests/test_trainer_callbacks.py cover behaviour).
def test_loss_early_stopping_class_importable():
    assert LossEarlyStoppingCallback is not None
