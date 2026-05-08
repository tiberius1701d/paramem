"""Tests for POST /gpu/acquire endpoint (D4 rename from /gpu/force-local).

Exercises the operator override that clears PARAMEM_EXTRA_ARGS=--defer-model
and, when the process is in cloud-only / defer mode, reloads the base model
in-process and switches the voice pipeline to the gpu profile.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch


def _call_gpu_acquire() -> object:
    from paramem.server.app import gpu_acquire

    return asyncio.run(gpu_acquire())


def test_acquire_in_defer_mode_reloads_and_switches_voice():
    """defer_model=True + cloud-only: reloads model in-process and switches voice to gpu.

    Assert ordering: _live_reload_base_model called, THEN _set_voice_pipeline_profile("gpu").
    """
    from paramem.server import app as app_module

    state_patch = {
        "defer_model": True,
        "mode": "cloud-only",
        "cloud_only_reason": "training",
    }

    call_order = []

    def fake_reload():
        call_order.append("reload")
        # Simulate successful reload setting mode to local.
        app_module._state["mode"] = "local"

    def fake_profile(profile):
        call_order.append(f"profile:{profile}")

    with (
        patch.dict(app_module._state, state_patch, clear=False),
        patch.object(app_module, "_clear_hold_env", return_value=True),
        patch.object(
            app_module,
            "_get_hold_state",
            return_value={"hold_active": True, "owner_pid": 1234, "owner_alive": True},
        ),
        patch.object(app_module, "_live_reload_base_model", side_effect=fake_reload),
        patch.object(app_module, "_set_voice_pipeline_profile", side_effect=fake_profile),
    ):
        result = _call_gpu_acquire()

    assert result["reloaded_live"] is True
    assert call_order == ["reload", "profile:gpu"], (
        f"Expected reload then voice switch; got {call_order}"
    )


def test_acquire_in_local_mode_is_noop():
    """When mode is already local, acquire is a no-op (idempotent).

    No reload, no voice switch; returns reloaded_live=False.
    """
    from paramem.server import app as app_module

    state_patch = {
        "defer_model": False,
        "mode": "local",
        "cloud_only_reason": None,
    }

    with (
        patch.dict(app_module._state, state_patch, clear=False),
        patch.object(app_module, "_clear_hold_env", return_value=False),
        patch.object(
            app_module,
            "_get_hold_state",
            return_value={"hold_active": False, "owner_pid": None, "owner_alive": None},
        ),
        patch.object(app_module, "_live_reload_base_model") as mock_reload,
        patch.object(app_module, "_set_voice_pipeline_profile") as mock_profile,
    ):
        result = _call_gpu_acquire()

    mock_reload.assert_not_called()
    mock_profile.assert_not_called()
    assert result["reloaded_live"] is False


def test_acquire_after_release_reloads_and_switches_voice():
    """cloud_only_reason="released" + cloud-only: acquire reloads + voice→gpu.

    Standard reclaim path: external consumer called /gpu/release, operator
    (or the consumer when done) calls /gpu/acquire to give GPU back to ParaMem.
    """
    from paramem.server import app as app_module

    state_patch = {
        "defer_model": False,
        "mode": "cloud-only",
        "cloud_only_reason": "released",
    }

    def _flip_mode_local():
        app_module._state["mode"] = "local"

    with (
        patch.dict(app_module._state, state_patch, clear=False),
        patch.object(app_module, "_clear_hold_env", return_value=False),
        patch.object(
            app_module,
            "_get_hold_state",
            return_value={"hold_active": False, "owner_pid": None, "owner_alive": None},
        ),
        patch.object(
            app_module, "_live_reload_base_model", side_effect=_flip_mode_local
        ) as mock_reload,
        patch.object(app_module, "_set_voice_pipeline_profile") as mock_profile,
    ):
        result = _call_gpu_acquire()

    mock_reload.assert_called_once()
    mock_profile.assert_called_once_with("gpu")
    assert result["reloaded_live"] is True


def test_acquire_respects_explicit_cloud_only_config():
    """cloud_only_reason="explicit" (yaml cloud_only: true): acquire is a no-op.

    Yaml-configured cloud-only represents persistent operator intent;
    leaving requires a config edit + restart, not a runtime override.
    """
    from paramem.server import app as app_module

    state_patch = {
        "defer_model": False,
        "mode": "cloud-only",
        "cloud_only_reason": "explicit",
    }

    with (
        patch.dict(app_module._state, state_patch, clear=False),
        patch.object(app_module, "_clear_hold_env", return_value=False),
        patch.object(
            app_module,
            "_get_hold_state",
            return_value={"hold_active": False, "owner_pid": None, "owner_alive": None},
        ),
        patch.object(app_module, "_live_reload_base_model") as mock_reload,
        patch.object(app_module, "_set_voice_pipeline_profile") as mock_profile,
    ):
        result = _call_gpu_acquire()

    mock_reload.assert_not_called()
    mock_profile.assert_not_called()
    assert result["reloaded_live"] is False


def test_acquire_falls_back_to_restart_on_reload_failure():
    """When _live_reload_base_model raises, _restart_service is called, voice NOT switched.

    The operator's intent (move to local) is still honoured via service restart.
    """
    from paramem.server import app as app_module

    state_patch = {
        "defer_model": True,
        "mode": "cloud-only",
        "cloud_only_reason": "training",
    }

    with (
        patch.dict(app_module._state, state_patch, clear=False),
        patch.object(app_module, "_clear_hold_env", return_value=True),
        patch.object(
            app_module,
            "_get_hold_state",
            return_value={"hold_active": True, "owner_pid": 1234, "owner_alive": True},
        ),
        patch.object(app_module, "_live_reload_base_model", side_effect=RuntimeError("CUDA OOM")),
        patch.object(app_module, "_set_voice_pipeline_profile") as mock_profile,
        patch.object(app_module, "_restart_service") as mock_restart,
    ):
        result = _call_gpu_acquire()

    mock_profile.assert_not_called()
    mock_restart.assert_called_once()
    assert result["will_restart"] is True
    assert result["reloaded_live"] is False
