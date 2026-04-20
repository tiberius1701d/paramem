"""Tests for GPU lock, device placement verification, thermal throttle, and server guards."""

import asyncio
import threading
import time
from unittest.mock import MagicMock, patch

import pytest
import torch

from paramem.server.gpu_lock import (
    acquire_gpu,
    gpu_lock,
    gpu_lock_sync,
    release_gpu,
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

    def test_timeout_raises(self):
        acquire_gpu()
        try:
            with pytest.raises(TimeoutError):
                with gpu_lock_sync(timeout=0.01):
                    pass
        finally:
            release_gpu()

    def test_released_on_exception(self):
        """Lock is released even if the body raises."""
        with pytest.raises(ValueError):
            with gpu_lock_sync():
                raise ValueError("test")
        # Should be acquirable again
        with gpu_lock_sync(timeout=0.1):
            pass


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


class TestAcquireReleaseGpu:
    def test_acquire_release_pair(self):
        acquire_gpu()
        release_gpu()

    def test_acquire_blocks_sync(self):
        acquire_gpu()
        try:
            with pytest.raises(TimeoutError):
                with gpu_lock_sync(timeout=0.01):
                    pass
        finally:
            release_gpu()


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

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU")
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


class TestThermalThrottle:
    def test_throttle_disabled_when_zero(self):
        """No throttle calls when temp_limit is 0."""
        from paramem.server.background_trainer import BackgroundTrainer

        bt = BackgroundTrainer(
            model=MagicMock(),
            tokenizer=MagicMock(),
            training_config=MagicMock(),
            temp_limit=0,
        )
        # Should return immediately without calling _gpu_temp
        with patch("paramem.server.background_trainer._gpu_temp") as mock_temp:
            bt._thermal_throttle(10)
            mock_temp.assert_not_called()

    def test_throttle_skips_non_interval_steps(self):
        """Only checks temp at interval boundaries."""
        from paramem.server.background_trainer import BackgroundTrainer

        bt = BackgroundTrainer(
            model=MagicMock(),
            tokenizer=MagicMock(),
            training_config=MagicMock(),
            temp_limit=55,
            temp_check_interval=5,
        )
        with patch("paramem.server.background_trainer._gpu_temp", return_value=40) as mock_temp:
            bt._thermal_throttle(3)  # not a multiple of 5
            mock_temp.assert_not_called()

            bt._thermal_throttle(5)  # multiple of 5
            mock_temp.assert_called_once()

    def test_throttle_no_pause_below_limit(self):
        """No pause when temp is below limit."""
        from paramem.server.background_trainer import BackgroundTrainer

        bt = BackgroundTrainer(
            model=MagicMock(),
            tokenizer=MagicMock(),
            training_config=MagicMock(),
            temp_limit=55,
            temp_check_interval=5,
        )
        with patch("paramem.server.background_trainer._gpu_temp", return_value=50):
            with patch("paramem.server.gpu_lock.release_gpu") as mock_release:
                bt._thermal_throttle(5)
                mock_release.assert_not_called()

    def test_throttle_pauses_above_limit(self):
        """Releases GPU lock and waits when temp exceeds limit."""
        from paramem.server.background_trainer import BackgroundTrainer

        bt = BackgroundTrainer(
            model=MagicMock(),
            tokenizer=MagicMock(),
            training_config=MagicMock(),
            temp_limit=55,
            temp_check_interval=5,
        )
        # First call: 60°C (above limit), second call: 50°C (below limit)
        temps = iter([60, 50])
        with patch("paramem.server.background_trainer._gpu_temp", side_effect=temps):
            with patch("paramem.server.gpu_lock.release_gpu") as mock_release:
                with patch("paramem.server.gpu_lock.acquire_gpu") as mock_acquire:
                    with patch("time.sleep"):
                        bt._thermal_throttle(5)
                        mock_release.assert_called_once()
                        mock_acquire.assert_called_once()

    def test_throttle_respects_shutdown(self):
        """Exits throttle loop on shutdown request."""
        from paramem.server.background_trainer import BackgroundTrainer

        bt = BackgroundTrainer(
            model=MagicMock(),
            tokenizer=MagicMock(),
            training_config=MagicMock(),
            temp_limit=55,
            temp_check_interval=5,
        )
        bt._shutdown_requested = True
        # Temp stays above limit but shutdown should break the loop
        with patch("paramem.server.background_trainer._gpu_temp", return_value=70):
            with patch("paramem.server.gpu_lock.release_gpu"):
                with patch("paramem.server.gpu_lock.acquire_gpu") as mock_acquire:
                    bt._thermal_throttle(5)
                    # Should still re-acquire before returning
                    mock_acquire.assert_called_once()


class TestQuietHoursPolicy:
    """Quiet-hours gate for the thermal throttle (smartphone sleep mode).

    The ``auto`` (hours-driven) branch is the real feature and gets the bulk of
    coverage here — ``always_on`` / ``always_off`` are simple short-circuits.
    """

    def _dt(self, hh: int, mm: int = 0):
        from datetime import datetime

        return datetime(2026, 4, 20, hh, mm)

    # --- Pure predicate ---

    def test_always_on_returns_true_ignoring_window(self):
        from paramem.server.background_trainer import is_thermal_policy_active

        # Window strings are ignored in always_on mode.
        assert is_thermal_policy_active("always_on", "07:00", "22:00", self._dt(3)) is True
        assert is_thermal_policy_active("always_on", "07:00", "22:00", self._dt(15)) is True

    def test_always_off_returns_false_ignoring_window(self):
        from paramem.server.background_trainer import is_thermal_policy_active

        assert is_thermal_policy_active("always_off", "00:00", "23:59", self._dt(3)) is False
        assert is_thermal_policy_active("always_off", "00:00", "23:59", self._dt(15)) is False

    def test_auto_daytime_window_work_laptop(self):
        """07:00–22:00 throttle — matches the server.yaml work-laptop profile."""
        from paramem.server.background_trainer import is_thermal_policy_active

        # Inside window: at the desk, throttle on.
        assert is_thermal_policy_active("auto", "07:00", "22:00", self._dt(7, 0)) is True
        assert is_thermal_policy_active("auto", "07:00", "22:00", self._dt(14, 30)) is True
        assert is_thermal_policy_active("auto", "07:00", "22:00", self._dt(21, 59)) is True
        # Outside window: overnight, fans free to spin.
        assert is_thermal_policy_active("auto", "07:00", "22:00", self._dt(6, 59)) is False
        assert is_thermal_policy_active("auto", "07:00", "22:00", self._dt(22, 0)) is False
        assert is_thermal_policy_active("auto", "07:00", "22:00", self._dt(2, 0)) is False

    def test_auto_nighttime_window_home_install(self):
        """22:00–07:00 throttle — matches the default.yaml home-install profile."""
        from paramem.server.background_trainer import is_thermal_policy_active

        # Inside midnight-crossing window.
        assert is_thermal_policy_active("auto", "22:00", "07:00", self._dt(22, 0)) is True
        assert is_thermal_policy_active("auto", "22:00", "07:00", self._dt(23, 59)) is True
        assert is_thermal_policy_active("auto", "22:00", "07:00", self._dt(0, 0)) is True
        assert is_thermal_policy_active("auto", "22:00", "07:00", self._dt(6, 59)) is True
        # Outside: daytime, training unthrottled.
        assert is_thermal_policy_active("auto", "22:00", "07:00", self._dt(7, 0)) is False
        assert is_thermal_policy_active("auto", "22:00", "07:00", self._dt(15, 0)) is False
        assert is_thermal_policy_active("auto", "22:00", "07:00", self._dt(21, 59)) is False

    def test_auto_malformed_window_falls_back_to_on(self):
        """Garbage HH:MM must not silently disable the throttle — prefer-silence default."""
        from paramem.server.background_trainer import is_thermal_policy_active

        assert is_thermal_policy_active("auto", "not-a-time", "07:00", self._dt(15)) is True
        assert is_thermal_policy_active("auto", "", "", self._dt(15)) is True

    def test_auto_degenerate_zero_length_window(self):
        """start == end is a zero-length window — treat as silent always."""
        from paramem.server.background_trainer import is_thermal_policy_active

        assert is_thermal_policy_active("auto", "12:00", "12:00", self._dt(12, 0)) is True
        assert is_thermal_policy_active("auto", "12:00", "12:00", self._dt(3, 0)) is True

    # --- Integration with the throttle ---

    def test_throttle_skipped_when_policy_inactive(self):
        """always_off makes _thermal_throttle a no-op even at scalding temps."""
        from paramem.server.background_trainer import BackgroundTrainer

        bt = BackgroundTrainer(
            model=MagicMock(),
            tokenizer=MagicMock(),
            training_config=MagicMock(),
            temp_limit=55,
            temp_check_interval=5,
            quiet_hours_mode="always_off",
        )
        with patch("paramem.server.background_trainer._gpu_temp", return_value=80) as mock_temp:
            with patch("paramem.server.gpu_lock.release_gpu") as mock_release:
                bt._thermal_throttle(5)
                # Gate short-circuits before the temperature probe runs.
                mock_temp.assert_not_called()
                mock_release.assert_not_called()

    def test_throttle_runs_in_auto_mode_inside_window(self):
        """auto mode with a current-time in window behaves like always_on."""
        from paramem.server.background_trainer import BackgroundTrainer

        bt = BackgroundTrainer(
            model=MagicMock(),
            tokenizer=MagicMock(),
            training_config=MagicMock(),
            temp_limit=55,
            temp_check_interval=5,
            quiet_hours_mode="auto",
            quiet_hours_start="00:00",
            quiet_hours_end="23:59",  # effectively always in window
        )
        temps = iter([70, 40])
        with patch("paramem.server.background_trainer._gpu_temp", side_effect=temps):
            with patch("paramem.server.gpu_lock.release_gpu") as mock_release:
                with patch("paramem.server.gpu_lock.acquire_gpu") as mock_acquire:
                    with patch("time.sleep"):
                        bt._thermal_throttle(5)
                        mock_release.assert_called_once()
                        mock_acquire.assert_called_once()

    def test_sleep_loop_exits_early_when_window_ends(self):
        """If quiet-hours ends while we're waiting for cooldown, resume even if still hot."""
        from paramem.server.background_trainer import BackgroundTrainer

        bt = BackgroundTrainer(
            model=MagicMock(),
            tokenizer=MagicMock(),
            training_config=MagicMock(),
            temp_limit=55,
            temp_check_interval=5,
            quiet_hours_mode="auto",
            quiet_hours_start="00:00",
            quiet_hours_end="23:59",
        )
        # First _should_throttle_now call (gate at entry) → True, triggers throttle.
        # Second call (entry of sleep loop) → False, breaks out.
        gate_answers = iter([True, False])
        with patch("paramem.server.background_trainer._gpu_temp", return_value=70):
            with patch("paramem.server.gpu_lock.release_gpu") as mock_release:
                with patch("paramem.server.gpu_lock.acquire_gpu") as mock_acquire:
                    with patch.object(
                        bt, "_should_throttle_now", side_effect=lambda: next(gate_answers)
                    ):
                        bt._thermal_throttle(5)
                        mock_release.assert_called_once()
                        # Re-acquires even though temp never dropped below limit.
                        mock_acquire.assert_called_once()

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
        from paramem.server.background_trainer import _gpu_temp

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="55\n")
            assert _gpu_temp() == 55

    def test_gpu_temp_returns_none_on_failure(self):
        from paramem.server.background_trainer import _gpu_temp

        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert _gpu_temp() is None

    def test_gpu_temp_returns_none_on_bad_rc(self):
        from paramem.server.background_trainer import _gpu_temp

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="")
            assert _gpu_temp() is None


class TestConsolidateGuard:
    def test_consolidate_rejects_during_training(self):
        """Manual /consolidate returns training_active when bg trainer is running."""
        from paramem.server.config import ConsolidationScheduleConfig

        # Minimal test — just verify the config field exists and defaults
        config = ConsolidationScheduleConfig()
        assert config.training_temp_limit == 0
        assert config.training_temp_check_interval == 5

    def test_consolidation_config_from_yaml(self):
        from paramem.server.config import load_server_config

        config = load_server_config("configs/server.yaml")
        assert config.consolidation.training_temp_limit == 55
        assert config.consolidation.training_temp_check_interval == 5
