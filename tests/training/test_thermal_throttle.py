"""Unit tests for ``paramem.training.thermal_throttle``.

Covers the thermal-throttle module that ``train_adapter`` installs as a
callback when a non-None ``ThermalPolicy`` is supplied. No GPU required:
``_gpu_temp`` is patched; the throttle does not touch the GPU lock.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

from paramem.training.thermal_throttle import (
    ThermalPolicy,
    ThermalThrottleCallback,
    is_thermal_policy_active,
)


class TestThermalPolicyFromConsolidationConfig:
    def test_zero_limit_returns_none(self):
        cfg = _make_cfg(training_temp_limit=0)
        assert ThermalPolicy.from_consolidation_config(cfg) is None

    def test_negative_limit_returns_none(self):
        cfg = _make_cfg(training_temp_limit=-1)
        assert ThermalPolicy.from_consolidation_config(cfg) is None

    def test_positive_limit_returns_policy(self):
        cfg = _make_cfg(
            training_temp_limit=55,
            training_temp_check_interval=5,
            quiet_hours_mode="auto",
            quiet_hours_start="22:00",
            quiet_hours_end="07:00",
        )
        policy = ThermalPolicy.from_consolidation_config(cfg)
        assert policy is not None
        assert policy.temp_limit == 55
        assert policy.check_interval == 5
        assert policy.quiet_hours_mode == "auto"


class TestThermalThrottleCallbackBehaviour:
    def _make_policy(self, **overrides):
        defaults = dict(
            temp_limit=55,
            check_interval=1,
            quiet_hours_mode="always_on",
            quiet_hours_start="22:00",
            quiet_hours_end="07:00",
        )
        defaults.update(overrides)
        return ThermalPolicy(**defaults)

    def test_skips_when_temp_below_limit(self):
        policy = self._make_policy()
        cb = ThermalThrottleCallback(policy)
        with patch("paramem.training.thermal_throttle._gpu_temp", return_value=40):
            cb._maybe_throttle(global_step=10)
        # No lock operations expected — throttle does not touch the GPU lock.

    def test_skips_when_check_interval_misses(self):
        policy = self._make_policy(check_interval=5)
        cb = ThermalThrottleCallback(policy)
        with patch("paramem.training.thermal_throttle._gpu_temp") as temp:
            cb._maybe_throttle(global_step=3)  # 3 % 5 != 0
            temp.assert_not_called()

    def test_sleeps_in_place_when_hot(self):
        """Throttle sleeps in place; does NOT touch the GPU lock."""
        policy = self._make_policy()
        cb = ThermalThrottleCallback(policy)
        # First read above limit (entry), second below (loop exit).
        with (
            patch(
                "paramem.training.thermal_throttle._gpu_temp",
                side_effect=[99, 40],
            ),
            patch("paramem.training.thermal_throttle.time.sleep") as sleep_mock,
        ):
            cb._maybe_throttle(global_step=10)
        # One sleep iteration before the loop sees the cool reading.
        sleep_mock.assert_called_once_with(5)

    def test_shutdown_fn_breaks_wait_loop(self):
        policy = self._make_policy()
        # shutdown_fn returns True on the second call, breaking the wait.
        flag = {"calls": 0}

        def shutdown_fn():
            flag["calls"] += 1
            return flag["calls"] >= 2

        cb = ThermalThrottleCallback(policy, shutdown_fn=shutdown_fn)
        # _gpu_temp keeps reading hot — only shutdown_fn can break out.
        with (
            patch(
                "paramem.training.thermal_throttle._gpu_temp",
                side_effect=[99, 99, 99, 99],
            ),
            patch("paramem.training.thermal_throttle.time.sleep"),
        ):
            cb._maybe_throttle(global_step=10)
        # Verify the shutdown branch was reached — shutdown_fn called at least twice.
        assert flag["calls"] >= 2

    def test_default_shutdown_fn_is_constant_false(self):
        # When shutdown_fn is not supplied, the default lambda is False.
        policy = self._make_policy()
        cb = ThermalThrottleCallback(policy)
        assert cb._shutdown_fn() is False


class TestIsThermalPolicyActive:
    """Truth table for the pure quiet-hours predicate served by
    schedule_grammar.Window: the three modes, plus the auto mode's
    prefer-silence fallback on a window that names no opening."""

    _INSIDE = datetime(2026, 1, 15, 23, 0)  # inside 22:00-07:00
    _OUTSIDE = datetime(2026, 1, 15, 12, 0)  # outside 22:00-07:00

    def test_always_off_is_never_active(self):
        assert is_thermal_policy_active("always_off", "22:00", "07:00", self._INSIDE) is False
        assert is_thermal_policy_active("always_off", "22:00", "07:00", self._OUTSIDE) is False

    def test_always_on_is_always_active(self):
        assert is_thermal_policy_active("always_on", "22:00", "07:00", self._INSIDE) is True
        assert is_thermal_policy_active("always_on", "22:00", "07:00", self._OUTSIDE) is True

    def test_auto_is_active_inside_the_window(self):
        assert is_thermal_policy_active("auto", "22:00", "07:00", self._INSIDE) is True

    def test_auto_is_inactive_outside_the_window(self):
        assert is_thermal_policy_active("auto", "22:00", "07:00", self._OUTSIDE) is False

    def test_auto_with_a_malformed_window_prefers_silence(self):
        """A window that cannot be parsed (bad HH:MM shape) falls back to
        True -- the prefer-silence default."""
        assert is_thermal_policy_active("auto", "9:5", "07:00", self._INSIDE) is True

    def test_auto_with_equal_start_and_end_prefers_silence(self):
        """start == end names no opening -- also the prefer-silence True."""
        assert is_thermal_policy_active("auto", "22:00", "22:00", self._INSIDE) is True

    def test_auto_wrapping_window_spans_midnight(self):
        """A window that wraps past midnight (22:00-07:00) is active late at
        night and early morning, inactive during the day."""
        assert (
            is_thermal_policy_active("auto", "22:00", "07:00", datetime(2026, 1, 16, 3, 0)) is True
        )
        assert (
            is_thermal_policy_active("auto", "22:00", "07:00", datetime(2026, 1, 16, 12, 0))
            is False
        )


def _make_cfg(**overrides):
    """Build a minimal ConsolidationConfig with thermal fields."""

    class _Cfg:
        training_temp_limit = 0
        training_temp_check_interval = 5
        quiet_hours_mode = "always_on"
        quiet_hours_start = "22:00"
        quiet_hours_end = "07:00"

    cfg = _Cfg()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg
