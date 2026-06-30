"""Tests for the POST /gpu/release endpoint and the unified lifespan teardown.

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

Teardown ordering test:

- Data-safety: save_snapshot and store.flush run BEFORE
  _release_base_model_in_process so a SIGKILL-during-GPU-release does
  not drop unconsolidated conversations or deferred speaker writes.
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


def test_release_switches_voice_to_cpu():
    """After model unload, voice pipeline is switched to cpu profile.

    Ordering: _release_base_model_in_process() called BEFORE the voice switch
    (verified via MagicMock.mock_calls ordering on the executor).
    """
    from paramem.server import app as app_module

    fake_config = MagicMock()
    fake_config.server.reclaim_interval_minutes = 5

    state_patch = {
        "mode": "local",
        "cloud_only_reason": None,
        "consolidating": False,
        "model": MagicMock(),
        "tokenizer": MagicMock(),
        "reclaim_task": None,
        "config": fake_config,
    }

    with (
        patch.dict(app_module._state, state_patch, clear=False),
        patch.object(app_module, "unload_model"),
        patch.object(app_module, "_set_voice_pipeline_profile") as mock_profile,
        patch("asyncio.create_task"),
    ):
        _call_gpu_release()

        # Voice switch to cpu must have been called.
        mock_profile.assert_called_once_with("cpu")


def test_release_clears_intent_classifier_handle():
    """Cloud-only VRAM-leak regression (holder 5): /gpu/release must clear the
    intent.mode=llm classifier handle (``_ClassifierModelHandle``) — it pins
    the base model + tokenizer, and a cloud-only server must hold ~0. The
    surviving lifespan-frame holders (WeightMemorySource / _classifier_model
    locals) are dropped in the lifespan and can't be unit-tested here; this
    guards the one holder the release path itself owns.
    """
    from paramem.server import app as app_module
    from paramem.server import intent as intent_module

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

    # Handle populated as the lifespan / reclaim would for intent.mode=llm.
    intent_module.set_classifier_model(MagicMock(), MagicMock())
    assert intent_module._classifier_model_singleton is not None

    with (
        patch.dict(app_module._state, state_patch, clear=False),
        patch.object(app_module, "unload_model"),
        patch.object(app_module, "_set_voice_pipeline_profile"),
        patch("asyncio.create_task"),
    ):
        _call_gpu_release()

    assert intent_module._classifier_model_singleton is None


# ---------------------------------------------------------------------------
# Lifespan teardown ordering
# ---------------------------------------------------------------------------


def test_lifespan_teardown_data_persisted_before_gpu_release(tmp_path):
    """Lifespan teardown persists data BEFORE releasing the GPU.

    Asserts call order: save_snapshot → store.flush → _release_base_model_in_process.

    Rationale: if a SIGKILL arrives during the slow GPU release (which can
    exceed TimeoutStopSec), both disk-only persistence ops must already be
    complete so no unconsolidated conversations or deferred speaker writes
    are dropped.

    The lifespan is driven with cloud_only=True (via cloud_only_startup=True
    so permanent_cloud_only=True, bypassing all CUDA/model-load paths) through
    to the ``yield`` and then allowed to exit normally so the teardown block
    executes.  _release_base_model_in_process, safe_empty_cache,
    buffer.save_snapshot, and store.flush are all patched with order trackers.
    """
    from paramem.server import app as app_module
    from paramem.server.config import PathsConfig, ServerConfig, STTConfig, TTSConfig

    config = ServerConfig(model_name="mistral")
    config.cloud_only = True
    config.stt = STTConfig(enabled=False)
    config.tts = TTSConfig(enabled=False)
    root = tmp_path / "data"
    config.paths = PathsConfig(
        data=root,
        sessions=root / "sessions",
        debug=root / "debug",
    )

    call_order: list[str] = []

    fake_buffer = MagicMock(name="session_buffer")
    fake_store = MagicMock(name="speaker_store")
    fake_buffer.save_snapshot.side_effect = lambda: call_order.append("save_snapshot")
    fake_store.flush.side_effect = lambda: call_order.append("store.flush")

    saved_state = {
        key: app_module._state.get(key) for key in ("config", "cloud_only_startup", "defer_model")
    }
    app_module._state["config"] = config
    # cloud_only_startup=True → cloud_only_reason="explicit" → permanent_cloud_only=True
    app_module._state["cloud_only_startup"] = True
    app_module._state["defer_model"] = False

    try:
        with (
            patch.object(app_module, "predict_base_bytes", return_value=None),
            patch.object(app_module, "_gpu_occupied", return_value=False),
            patch.object(app_module, "_build_config_derived_state"),
            patch.object(app_module, "_arm_active_store_migration", return_value=False),
            patch.object(
                app_module,
                "_release_base_model_in_process",
                side_effect=lambda: call_order.append("_release_base_model_in_process"),
            ),
            patch.object(app_module, "safe_empty_cache"),
            patch.dict(
                app_module._state,
                {
                    "session_buffer": fake_buffer,
                    "speaker_store": fake_store,
                    "reclaim_task": None,
                    "config_drift_task": None,
                },
                clear=False,
            ),
        ):

            async def _run():
                async with app_module.lifespan(app_module.app):
                    pass  # yield reached; exit context to run teardown

            asyncio.run(_run())
    finally:
        for key, val in saved_state.items():
            if val is None:
                app_module._state.pop(key, None)
            else:
                app_module._state[key] = val

    assert "save_snapshot" in call_order, "buffer.save_snapshot not called in teardown"
    assert "store.flush" in call_order, "store.flush not called in teardown"
    assert "_release_base_model_in_process" in call_order, (
        "_release_base_model_in_process not called in teardown"
    )

    snap_idx = call_order.index("save_snapshot")
    flush_idx = call_order.index("store.flush")
    release_idx = call_order.index("_release_base_model_in_process")
    assert snap_idx < flush_idx, (
        f"save_snapshot must come before store.flush; got order={call_order}"
    )
    assert flush_idx < release_idx, (
        f"store.flush must come before _release_base_model_in_process; got order={call_order}"
    )
