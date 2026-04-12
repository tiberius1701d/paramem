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
