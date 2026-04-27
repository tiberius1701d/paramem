"""Tests verifying that on_released() fires for the paramem consumer on
a free-GPU acquire-release cycle (Fix 3 regression guard).

Before Fix 3, on_released() was only called for consumers involved in the
GPU-deferral path (i.e. when the paramem server was actively using the GPU
and we asked it to release).  A normal training run that acquired a free
GPU would call on_acquired (stamping PARAMEM_HOLD_PID) but never on_released,
leaving PARAMEM_HOLD_PID stale in the systemd environment across runs.

This test uses a recording stub for subprocess.run that captures the
``systemctl --user unset-environment`` call.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# gpu_guard is provided by lab-tools (separate repo, not on PyPI).  CI does
# not install it; skip the whole module rather than erroring at collection.
pytest.importorskip("gpu_guard")

import gpu_guard._core as _core_module  # noqa: E402


class TestParamemHoldEnvVarClearedOnRelease:
    """Verify PARAMEM_HOLD_PID is cleared via unset-environment on normal exit."""

    def test_on_released_fires_for_free_gpu_acquire(self):
        """After a free-GPU acquire-and-release the paramem consumer's
        on_released() must be called, which issues systemctl unset-environment.
        """
        # Build a stub consumer whose subprocess.run calls we can inspect.
        run_calls: list[list] = []

        def recording_run(args, **kwargs):
            run_calls.append(list(args))
            m = MagicMock()
            m.returncode = 0
            return m

        from paramem.gpu_consumer import ParamemEnvStampAdapter

        consumer = ParamemEnvStampAdapter()

        with (
            patch.object(_core_module, "_get_gpu_pids", return_value=[]),
            patch.object(_core_module, "read_all_live", return_value=[]),
            patch.object(_core_module, "write_holder"),
            patch.object(_core_module, "remove_holder"),
            # Patch subprocess.run inside gpu_consumer so we capture its calls.
            patch("paramem.gpu_consumer.subprocess.run", side_effect=recording_run),
        ):
            from gpu_guard._core import acquire_gpu
            from gpu_guard.inhibitor import NullInhibitor
            from gpu_guard.notifier import NullNotifier

            with acquire_gpu(
                priority=0,
                name="test-free-gpu",
                consumer=consumer,
                notifier=NullNotifier(),
                inhibitor=NullInhibitor(),
            ):
                pass  # normal body

        # Verify that at least one call was "systemctl --user unset-environment"
        unset_calls = [
            args
            for args in run_calls
            if len(args) >= 3 and args[1] == "--user" and args[2] == "unset-environment"
        ]
        assert unset_calls, (
            "Expected systemctl --user unset-environment to be called in on_released, "
            f"but recorded calls were: {run_calls}"
        )
        # Verify the hold vars are included in the unset call.
        unset_args = unset_calls[0]
        assert "PARAMEM_HOLD_PID" in unset_args, f"PARAMEM_HOLD_PID missing from: {unset_args}"

    def test_on_acquired_fires_for_free_gpu_acquire(self):
        """on_acquired must also be called so PARAMEM_HOLD_PID gets stamped initially."""
        run_calls: list[list] = []

        def recording_run(args, **kwargs):
            run_calls.append(list(args))
            m = MagicMock()
            m.returncode = 0
            return m

        from paramem.gpu_consumer import ParamemEnvStampAdapter

        consumer = ParamemEnvStampAdapter()

        with (
            patch.object(_core_module, "_get_gpu_pids", return_value=[]),
            patch.object(_core_module, "read_all_live", return_value=[]),
            patch.object(_core_module, "write_holder"),
            patch.object(_core_module, "remove_holder"),
            patch("paramem.gpu_consumer.subprocess.run", side_effect=recording_run),
        ):
            from gpu_guard._core import acquire_gpu
            from gpu_guard.inhibitor import NullInhibitor
            from gpu_guard.notifier import NullNotifier

            with acquire_gpu(
                priority=0,
                name="test-free-gpu",
                consumer=consumer,
                notifier=NullNotifier(),
                inhibitor=NullInhibitor(),
            ):
                pass

        # set-environment should appear (on_acquired).
        set_calls = [
            args
            for args in run_calls
            if len(args) >= 3 and args[1] == "--user" and args[2] == "set-environment"
        ]
        assert set_calls, (
            "Expected systemctl --user set-environment to be called in on_acquired, "
            f"but recorded calls were: {run_calls}"
        )
        set_args_flat = " ".join(set_calls[0])
        assert "PARAMEM_HOLD_PID" in set_args_flat
