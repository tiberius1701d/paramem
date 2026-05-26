"""Unit tests for ``paramem.training.thermal_throttle``.

Covers the new thermal-throttle module that ``train_adapter`` installs as a
callback when a non-None ``ThermalPolicy`` is supplied. No GPU required:
``_gpu_temp`` is patched; the throttle no longer touches the GPU lock.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

from paramem.training.thermal_throttle import (
    ThermalPolicy,
    ThermalThrottleCallback,
    _should_throttle_now,
    is_thermal_policy_active,
)


class TestIsThermalPolicyActive:
    def test_always_off(self):
        assert is_thermal_policy_active("always_off", "22:00", "07:00") is False

    def test_always_on(self):
        assert is_thermal_policy_active("always_on", "22:00", "07:00") is True

    def test_auto_within_window_simple(self):
        # 23:00 sits inside 22:00-07:00 (overnight wrap).
        now = datetime(2026, 5, 7, 23, 0)
        assert is_thermal_policy_active("auto", "22:00", "07:00", now) is True

    def test_auto_outside_window_simple(self):
        # 12:00 is outside 22:00-07:00.
        now = datetime(2026, 5, 7, 12, 0)
        assert is_thermal_policy_active("auto", "22:00", "07:00", now) is False

    def test_auto_within_same_day_window(self):
        # 03:00 inside 02:00-04:00 (same-day forward window).
        now = datetime(2026, 5, 7, 3, 0)
        assert is_thermal_policy_active("auto", "02:00", "04:00", now) is True

    def test_auto_invalid_window_falls_back_to_active(self):
        # Garbage strings → the prefer-silence fallback returns True.
        assert is_thermal_policy_active("auto", "bad", "input") is True


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

    def test_skips_when_window_inactive(self):
        policy = self._make_policy(quiet_hours_mode="always_off")
        cb = ThermalThrottleCallback(policy)
        with patch("paramem.training.thermal_throttle._gpu_temp") as temp_mock:
            cb._maybe_throttle(global_step=10)
        # Window inactive → _gpu_temp not even called.
        temp_mock.assert_not_called()

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

    def test_should_throttle_now_routes_through_predicate(self):
        policy = self._make_policy(quiet_hours_mode="always_on")
        assert _should_throttle_now(policy) is True
        policy_off = self._make_policy(quiet_hours_mode="always_off")
        assert _should_throttle_now(policy_off) is False


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
