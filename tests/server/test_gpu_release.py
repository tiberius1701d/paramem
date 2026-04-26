"""Tests for the POST /gpu/release endpoint.

The endpoint is the canonical local→cloud-only release path used by
external GPU consumers (gpu_guard ConfigConsumer / V1
paramem.gpu_consumer / lerobot). It replaces the old SIGUSR1 protocol,
which the V1 ``ParamemServerConsumer.request_release`` documented as
"switch to cloud-only" but ``app.py``'s SIGUSR1 handler implemented as
"save snapshot and exit" — protocol mismatch surfaced under V2 testing.

These tests exercise the endpoint function directly (no TestClient,
so we avoid the heavy app lifespan). Behavior contract:

- Idempotent on already-cloud-only.
- 503 when a consolidation cycle is in flight.
- Synchronous: by the time the call returns 200, the model is unloaded
  and ``_state["mode"]`` is ``"cloud-only"``.
- Auto-reclaim loop is started on success so the server reclaims the
  GPU once the external consumer goes away.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

from fastapi.responses import JSONResponse


def _call_gpu_release() -> object:
    from paramem.server.app import gpu_release

    return asyncio.run(gpu_release())


def test_already_cloud_only_is_idempotent_returns_released_false():
    """A server already in cloud-only mode returns 200 with released=False."""
    from paramem.server import app as app_module

    state_patch = {
        "mode": "cloud-only",
        "cloud_only_reason": "explicit",
        "consolidating": False,
        "model": None,
        "tokenizer": None,
        "reclaim_task": None,
    }
    with patch.dict(app_module._state, state_patch, clear=False):
        result = _call_gpu_release()

    assert result == {"mode": "cloud-only", "released": False, "reason": "explicit"}


def test_consolidating_returns_503_with_retry_hint():
    """Mid-consolidation: refuse with 503 so the caller may retry."""
    from paramem.server import app as app_module

    state_patch = {
        "mode": "local",
        "consolidating": True,
        "model": object(),
        "tokenizer": object(),
        "reclaim_task": None,
    }
    with patch.dict(app_module._state, state_patch, clear=False):
        result = _call_gpu_release()

    assert isinstance(result, JSONResponse)
    assert result.status_code == 503
    body = result.body.decode("utf-8")
    assert "consolidating" in body


def test_local_mode_unloads_model_and_switches_to_cloud_only():
    """Happy path: model unloaded, mode flipped, reclaim task started."""
    from paramem.server import app as app_module

    fake_model = MagicMock(name="model")
    fake_tokenizer = MagicMock(name="tokenizer")
    fake_config = MagicMock()
    fake_config.server.reclaim_interval_minutes = 5
    fake_task = MagicMock()
    fake_task.done.return_value = False

    state_patch = {
        "mode": "local",
        "cloud_only_reason": None,
        "consolidating": False,
        "model": fake_model,
        "tokenizer": fake_tokenizer,
        "reclaim_task": None,
        "config": fake_config,
    }
    with (
        patch.dict(app_module._state, state_patch, clear=False),
        patch.object(app_module, "unload_model") as mock_unload,
        patch("asyncio.create_task", return_value=fake_task) as mock_create_task,
    ):
        result = _call_gpu_release()

        # All assertions inside the with-block: patch.dict restores _state on exit.
        assert result == {"mode": "cloud-only", "released": True, "reason": "released"}
        mock_unload.assert_called_once_with(fake_model, fake_tokenizer)
        assert app_module._state["mode"] == "cloud-only"
        assert app_module._state["cloud_only_reason"] == "released"
        assert app_module._state["model"] is None
        assert app_module._state["tokenizer"] is None
        mock_create_task.assert_called_once()


def test_local_mode_with_running_reclaim_task_does_not_double_start():
    """If the reclaim task is already running, do not start a second one."""
    from paramem.server import app as app_module

    fake_running = MagicMock(name="reclaim_task")
    fake_running.done.return_value = False
    fake_config = MagicMock()
    fake_config.server.reclaim_interval_minutes = 5

    state_patch = {
        "mode": "local",
        "consolidating": False,
        "model": MagicMock(),
        "tokenizer": MagicMock(),
        "reclaim_task": fake_running,
        "config": fake_config,
    }
    with (
        patch.dict(app_module._state, state_patch, clear=False),
        patch.object(app_module, "unload_model"),
        patch("asyncio.create_task") as mock_create_task,
    ):
        _call_gpu_release()
        mock_create_task.assert_not_called()
        assert app_module._state["reclaim_task"] is fake_running


def test_local_mode_with_completed_reclaim_task_starts_new_one():
    """A done() reclaim task is replaced by a fresh one."""
    from paramem.server import app as app_module

    fake_done = MagicMock(name="completed_task")
    fake_done.done.return_value = True
    fake_new = MagicMock(name="new_task")
    fake_new.done.return_value = False
    fake_config = MagicMock()
    fake_config.server.reclaim_interval_minutes = 5

    state_patch = {
        "mode": "local",
        "consolidating": False,
        "model": MagicMock(),
        "tokenizer": MagicMock(),
        "reclaim_task": fake_done,
        "config": fake_config,
    }
    with (
        patch.dict(app_module._state, state_patch, clear=False),
        patch.object(app_module, "unload_model"),
        patch("asyncio.create_task", return_value=fake_new) as mock_create_task,
    ):
        _call_gpu_release()
        mock_create_task.assert_called_once()
        assert app_module._state["reclaim_task"] is fake_new


def test_unload_failure_does_not_block_mode_switch():
    """If unload_model raises, the mode still flips (model state cleared) and
    the response is still 200. Ensures we never end up stuck in 'local' mode
    because unload threw — caller's worst case is a logged exception, which
    is the right tradeoff vs returning 500 to the consumer."""
    from paramem.server import app as app_module

    fake_config = MagicMock()
    fake_config.server.reclaim_interval_minutes = 5

    state_patch = {
        "mode": "local",
        "consolidating": False,
        "model": MagicMock(),
        "tokenizer": MagicMock(),
        "reclaim_task": None,
        "config": fake_config,
    }
    with (
        patch.dict(app_module._state, state_patch, clear=False),
        patch.object(app_module, "unload_model", side_effect=RuntimeError("boom")),
        patch("asyncio.create_task"),
    ):
        result = _call_gpu_release()
        assert result == {"mode": "cloud-only", "released": True, "reason": "released"}
        assert app_module._state["mode"] == "cloud-only"
        assert app_module._state["model"] is None
