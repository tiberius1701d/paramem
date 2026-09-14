"""Tests for GPU lock, device placement verification, thermal throttle, and server guards."""

import asyncio
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from paramem.server.gpu_lock import (
    gpu_lock,
    gpu_lock_is_held,
    gpu_lock_sync,
)
from paramem.utils.config import ModelConfig


class TestGpuLockSync:
    def test_acquire_release(self):
        with gpu_lock_sync():
            pass  # should not raise

    def test_mutual_exclusion(self):
        """Two threads cannot hold the lock simultaneously."""
        results = []

        def worker(label):
            with gpu_lock_sync():
                results.append(f"{label}_enter")
                time.sleep(0.05)
                results.append(f"{label}_exit")

        t1 = threading.Thread(target=worker, args=("a",))
        t2 = threading.Thread(target=worker, args=("b",))
        t1.start()
        time.sleep(0.01)  # ensure t1 acquires first
        t2.start()
        t1.join()
        t2.join()

        # t1 must fully complete before t2 enters
        assert results.index("a_exit") < results.index("b_enter")

    def test_released_on_exception(self):
        """Lock is released even if the body raises."""
        with pytest.raises(ValueError):
            with gpu_lock_sync():
                raise ValueError("test")
        # Should be acquirable again
        with gpu_lock_sync(timeout=0.1):
            pass

    def test_timeout_raises(self):
        """A bounded acquire against a lock held by another thread raises
        TimeoutError instead of blocking forever."""
        acquired = threading.Event()
        release = threading.Event()

        def holder():
            with gpu_lock_sync():
                acquired.set()
                release.wait(timeout=5)

        t = threading.Thread(target=holder)
        t.start()
        try:
            assert acquired.wait(timeout=5), "holder thread never acquired the lock"
            with pytest.raises(TimeoutError, match="Could not acquire GPU lock"):
                with gpu_lock_sync(timeout=0.05):
                    pass
        finally:
            release.set()
            t.join(timeout=5)


class TestGpuLockAsync:
    def test_async_acquire_release(self):
        async def _run():
            async with gpu_lock():
                pass

        asyncio.run(_run())

    def test_async_mutual_exclusion_with_sync(self):
        """Async and sync callers cannot hold the lock simultaneously."""
        results = []

        def sync_worker():
            with gpu_lock_sync():
                results.append("sync_enter")
                time.sleep(0.05)
                results.append("sync_exit")

        async def _run():
            t = threading.Thread(target=sync_worker)
            t.start()
            await asyncio.sleep(0.01)  # let sync worker acquire first
            async with gpu_lock():
                results.append("async_enter")
            t.join()

        asyncio.run(_run())
        assert results.index("sync_exit") < results.index("async_enter")


class TestGpuLockIsHeld:
    """gpu_lock_is_held() is a read-only probe — it never takes the lock."""

    def test_false_when_free(self):
        assert gpu_lock_is_held() is False

    def test_does_not_take_the_lock(self):
        """Calling the probe must not itself acquire the lock — a real
        acquirer must still succeed immediately afterward."""
        assert gpu_lock_is_held() is False
        # If the probe had leaked an acquire, this would time out.
        with gpu_lock_sync(timeout=0.1):
            pass

    def test_true_when_held(self):
        """Reads True while a caller holds the lock, False again once released."""
        with gpu_lock_sync():
            assert gpu_lock_is_held() is True
        assert gpu_lock_is_held() is False

    def test_reflects_held_state_from_another_thread(self):
        """The probe reflects the lock's state regardless of which thread
        holds it — a threading.Lock is not owned by the acquiring thread."""
        acquired = threading.Event()
        release = threading.Event()

        def holder():
            with gpu_lock_sync():
                acquired.set()
                release.wait(timeout=5)

        t = threading.Thread(target=holder)
        t.start()
        try:
            assert acquired.wait(timeout=5), "holder thread never acquired the lock"
            assert gpu_lock_is_held() is True
        finally:
            release.set()
            t.join(timeout=5)
        assert gpu_lock_is_held() is False


class TestDevicePlacement:
    def test_cpu_params_raise_when_no_offload(self):
        from paramem.models.loader import _verify_device_placement

        model = MagicMock()
        param_cpu = torch.zeros(10, device="cpu")
        model.parameters.return_value = [param_cpu]

        config = ModelConfig(
            model_id="test/model",
            cpu_offload=False,
        )

        with pytest.raises(RuntimeError, match="params on CPU"):
            _verify_device_placement(model, config)

    def test_cpu_offload_true_allows_cpu_params(self):
        from paramem.models.loader import _verify_device_placement

        model = MagicMock()
        param_cpu = torch.zeros(10, device="cpu")
        model.parameters.return_value = [param_cpu]

        config = ModelConfig(
            model_id="test/model",
            cpu_offload=True,
        )

        # Should not raise
        _verify_device_placement(model, config)

    @pytest.mark.gpu
    @pytest.mark.skipif(
        "not config.getoption('--gpu', default=False)",
        reason="GPU tests require --gpu flag",
    )
    def test_all_gpu_params_pass(self):
        from paramem.models.loader import _verify_device_placement

        model = MagicMock()
        param_gpu = torch.zeros(10, device="cuda")
        model.parameters.return_value = [param_gpu]

        config = ModelConfig(
            model_id="test/model",
            cpu_offload=False,
        )

        # Should not raise
        _verify_device_placement(model, config)


# The thermal-throttle body lives in ThermalThrottleCallback
# (paramem/training/thermal_throttle.py), not on BackgroundTrainer, and is
# exercised by tests/training/test_thermal_throttle.py.
# Quiet-hours predicate tests below remain — is_thermal_policy_active is still
# importable from background_trainer via a re-export shim.


class TestQuietHoursPolicy:
    """Quiet-hours gate for the thermal throttle (smartphone sleep mode).

    The ``auto`` (hours-driven) branch is the real feature and gets the bulk of
    coverage here — ``always_on`` / ``always_off`` are simple short-circuits.
    """

    def _dt(self, hh: int, mm: int = 0):
        from datetime import datetime

        return datetime(2026, 4, 20, hh, mm)

    # --- Pure predicate ---

    # --- Throttle-with-policy integration coverage ---
    # The throttle body lives in ThermalThrottleCallback, not on
    # BackgroundTrainer; coverage is in
    # tests/training/test_thermal_throttle.py::TestThermalThrottleCallbackBehaviour
    # (skips_when_window_inactive, releases_and_reacquires_when_hot,
    # shutdown_fn_breaks_wait_loop, etc.).

    # --- Config validator ---

    def test_config_rejects_unknown_mode(self):
        from paramem.server.config import ConsolidationScheduleConfig

        with pytest.raises(ValueError, match="quiet_hours_mode"):
            ConsolidationScheduleConfig(quiet_hours_mode="maybe")

    def test_config_rejects_malformed_window_in_auto(self):
        from paramem.server.config import ConsolidationScheduleConfig

        with pytest.raises(ValueError, match="quiet_hours_start"):
            ConsolidationScheduleConfig(quiet_hours_mode="auto", quiet_hours_start="2500")
        with pytest.raises(ValueError, match="quiet_hours_end"):
            ConsolidationScheduleConfig(quiet_hours_mode="auto", quiet_hours_end="7:99")

    def test_config_accepts_malformed_window_when_not_auto(self):
        """Validator only enforces HH:MM format in auto mode — other modes ignore it."""
        from paramem.server.config import ConsolidationScheduleConfig

        # Garbage strings accepted because mode won't consume them.
        ConsolidationScheduleConfig(quiet_hours_mode="always_on", quiet_hours_start="bogus")
        ConsolidationScheduleConfig(quiet_hours_mode="always_off", quiet_hours_end="25:99")


class TestGpuTemp:
    def test_gpu_temp_returns_int(self):
        from paramem.training.thermal_throttle import _gpu_temp

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="55\n")
            assert _gpu_temp() == 55

    def test_gpu_temp_returns_none_on_failure(self):
        from paramem.training.thermal_throttle import _gpu_temp

        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert _gpu_temp() is None

    def test_gpu_temp_returns_none_on_bad_rc(self):
        from paramem.training.thermal_throttle import _gpu_temp

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="")
            assert _gpu_temp() is None


class TestConsolidateGuard:
    @pytest.mark.skipif(
        not Path("configs/server.yaml").exists(),
        reason="operator-local configs/server.yaml absent (CI / fresh clone)",
    )
    def test_consolidation_config_from_yaml(self):
        from paramem.server.config import load_server_config

        config = load_server_config("configs/server.yaml")
        assert config.consolidation.training_temp_limit == 55
        assert config.consolidation.training_temp_check_interval == 5
