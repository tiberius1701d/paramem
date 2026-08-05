"""Tests for the POST /gpu/release endpoint and the unified lifespan teardown.

The endpoint is the canonical local→cloud-only release path used by
external GPU consumers (gpu_guard ConfigConsumer / V1
paramem.utils.gpu_consumer / lerobot). It replaces the old SIGUSR1 protocol,
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
import threading
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.responses import JSONResponse


def _call_gpu_release() -> object:
    from paramem.server.app import gpu_release

    return asyncio.run(gpu_release())


def test_release_refuses_during_base_swap_409():
    """An actively running base-swap migration refuses with 409
    base_swap_active — checked before the cloud-only idempotent
    short-circuit, since a swap can transiently hold mode='cloud-only'
    between its own Phase A -> Phase B reload."""
    from paramem.server import app as app_module

    state_patch = {
        "mode": "local",
        "consolidating": False,
        "model": object(),
        "tokenizer": object(),
        "migration": {"base_swap_active": True},
    }
    with (
        patch.dict(app_module._state, state_patch, clear=False),
        pytest.raises(HTTPException) as exc_info,
    ):
        _call_gpu_release()

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail["error"] == "base_swap_active"


def test_release_refuses_during_base_swap_even_when_mode_is_cloud_only():
    """The base-swap guard fires before the cloud-only idempotent
    short-circuit — a transient cloud-only window mid-swap must not slip
    through as a no-op 200."""
    from paramem.server import app as app_module

    state_patch = {
        "mode": "cloud-only",
        "cloud_only_reason": "live_reload",
        "consolidating": False,
        "model": None,
        "tokenizer": None,
        "migration": {"base_swap_active": True},
    }
    with (
        patch.dict(app_module._state, state_patch, clear=False),
        pytest.raises(HTTPException) as exc_info,
    ):
        _call_gpu_release()

    assert exc_info.value.status_code == 409
    assert exc_info.value.detail["error"] == "base_swap_active"


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
        # Also patch the coroutine function itself: even with create_task
        # mocked, `_auto_reclaim_loop(reclaim_interval)` is still evaluated
        # as create_task's argument, minting a real un-awaited coroutine
        # object that triggers a RuntimeWarning at garbage-collection time.
        # A plain MagicMock, NOT the AsyncMock patch.object would autospec by
        # default for an async target: an AsyncMock's __call__ still returns
        # a real coroutine, which (with create_task also mocked below) would
        # itself go unawaited and just relocate the RuntimeWarning rather
        # than removing it. A synchronous MagicMock returns a plain object
        # instead, so no coroutine is ever created.
        patch.object(app_module, "_auto_reclaim_loop", MagicMock()),
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
        # Also patch the coroutine function itself: even with create_task
        # mocked, `_auto_reclaim_loop(reclaim_interval)` is still evaluated
        # as create_task's argument, minting a real un-awaited coroutine
        # object that triggers a RuntimeWarning at garbage-collection time.
        # A plain MagicMock, NOT the AsyncMock patch.object would autospec by
        # default for an async target: an AsyncMock's __call__ still returns
        # a real coroutine, which (with create_task also mocked below) would
        # itself go unawaited and just relocate the RuntimeWarning rather
        # than removing it. A synchronous MagicMock returns a plain object
        # instead, so no coroutine is ever created.
        patch.object(app_module, "_auto_reclaim_loop", MagicMock()),
        patch("asyncio.create_task", return_value=fake_new) as mock_create_task,
    ):
        _call_gpu_release()
        mock_create_task.assert_called_once()
        assert app_module._state["reclaim_task"] is fake_new


def test_route_starts_reclaim_task_on_real_release():
    """The ``/gpu/release`` ROUTE handler starts the auto-reclaim loop on a
    successful release — this is route policy, asserted here against the
    real route (``_call_gpu_release`` -> ``gpu_release()``), distinct from
    ``_gpu_release_internal`` itself (see
    ``test_internal_release_never_starts_reclaim_task`` below)."""
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
        patch.object(app_module, "_set_voice_pipeline_profile"),
        # A plain MagicMock, NOT the AsyncMock patch.object would autospec by
        # default for an async target: an AsyncMock's __call__ still returns
        # a real coroutine, which (with create_task also mocked below) would
        # itself go unawaited and just relocate the RuntimeWarning rather
        # than removing it. A synchronous MagicMock returns a plain object
        # instead, so no coroutine is ever created.
        patch.object(app_module, "_auto_reclaim_loop", MagicMock()),
        patch("asyncio.create_task") as mock_create_task,
    ):
        result = _call_gpu_release()

    assert result == {"mode": "cloud-only", "released": True, "reason": "released"}
    mock_create_task.assert_called_once()


def test_internal_release_never_starts_reclaim_task():
    """``_gpu_release_internal`` — the function the base-swap
    orchestration's fresh-start reload step calls DIRECTLY (never the
    ``/gpu/release`` route) — must never start the auto-reclaim loop.
    Starting it here is route policy, owned by the ``/gpu/release`` handler
    alone (see ``_gpu_release_internal``'s docstring): a reclaim task
    started on the orchestration's release could fire and reload the OLD
    base after a deferred base-swap reload, flipping ``mode`` back to
    ``"local"`` before ``/gpu/acquire``'s deferred-Phase-B relaunch gate
    (``mode == "cloud-only"``) ever observes it. This test calls the real
    ``_gpu_release_internal`` directly, exactly as the orchestration does,
    with ``_apply_config_live`` out of scope (this function only covers the
    release half)."""
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
        patch.object(app_module, "_set_voice_pipeline_profile"),
        # A plain MagicMock, NOT the AsyncMock patch.object would autospec by
        # default for an async target: an AsyncMock's __call__ still returns
        # a real coroutine, which (with create_task also mocked below) would
        # itself go unawaited and just relocate the RuntimeWarning rather
        # than removing it. A synchronous MagicMock returns a plain object
        # instead, so no coroutine is ever created.
        patch.object(app_module, "_auto_reclaim_loop", MagicMock()),
        patch("asyncio.create_task") as mock_create_task,
    ):
        result = asyncio.run(app_module._gpu_release_internal())

    assert result == {"mode": "cloud-only", "released": True, "reason": "released"}
    mock_create_task.assert_not_called()
    assert app_module._state.get("reclaim_task") is None


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
        # Also patch the coroutine function itself: even with create_task
        # mocked, `_auto_reclaim_loop(reclaim_interval)` is still evaluated
        # as create_task's argument, minting a real un-awaited coroutine
        # object that triggers a RuntimeWarning at garbage-collection time.
        # A plain MagicMock, NOT the AsyncMock patch.object would autospec by
        # default for an async target: an AsyncMock's __call__ still returns
        # a real coroutine, which (with create_task also mocked below) would
        # itself go unawaited and just relocate the RuntimeWarning rather
        # than removing it. A synchronous MagicMock returns a plain object
        # instead, so no coroutine is ever created.
        patch.object(app_module, "_auto_reclaim_loop", MagicMock()),
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
        # Also patch the coroutine function itself: even with create_task
        # mocked, `_auto_reclaim_loop(reclaim_interval)` is still evaluated
        # as create_task's argument, minting a real un-awaited coroutine
        # object that triggers a RuntimeWarning at garbage-collection time.
        # A plain MagicMock, NOT the AsyncMock patch.object would autospec by
        # default for an async target: an AsyncMock's __call__ still returns
        # a real coroutine, which (with create_task also mocked below) would
        # itself go unawaited and just relocate the RuntimeWarning rather
        # than removing it. A synchronous MagicMock returns a plain object
        # instead, so no coroutine is ever created.
        patch.object(app_module, "_auto_reclaim_loop", MagicMock()),
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
        # Also patch the coroutine function itself: even with create_task
        # mocked, `_auto_reclaim_loop(reclaim_interval)` is still evaluated
        # as create_task's argument, minting a real un-awaited coroutine
        # object that triggers a RuntimeWarning at garbage-collection time.
        # A plain MagicMock, NOT the AsyncMock patch.object would autospec by
        # default for an async target: an AsyncMock's __call__ still returns
        # a real coroutine, which (with create_task also mocked below) would
        # itself go unawaited and just relocate the RuntimeWarning rather
        # than removing it. A synchronous MagicMock returns a plain object
        # instead, so no coroutine is ever created.
        patch.object(app_module, "_auto_reclaim_loop", MagicMock()),
    ):
        _call_gpu_release()

    assert intent_module._classifier_model_singleton is None


# ---------------------------------------------------------------------------
# GPU lock symmetry for /gpu/release
# ---------------------------------------------------------------------------


def test_release_holds_gpu_lock_across_unload():
    """The teardown dispatch runs with the shared GPU thread lock held —
    mirror of ``test_acquire_holds_gpu_lock_across_reload``. Asserts the lock
    is locked while ``_release_base_model_in_process`` runs and free once the
    request completes."""
    from paramem.server import app as app_module
    from paramem.server.gpu_lock import _gpu_thread_lock

    lock_state_during_release = []

    def fake_release():
        lock_state_during_release.append(_gpu_thread_lock.locked())

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
        patch.object(app_module, "_release_base_model_in_process", side_effect=fake_release),
        patch.object(app_module, "_set_voice_pipeline_profile"),
        patch("asyncio.create_task"),
        # Also patch the coroutine function itself: even with create_task
        # mocked, `_auto_reclaim_loop(reclaim_interval)` is still evaluated
        # as create_task's argument, minting a real un-awaited coroutine
        # object that triggers a RuntimeWarning at garbage-collection time.
        # A plain MagicMock, NOT the AsyncMock patch.object would autospec by
        # default for an async target: an AsyncMock's __call__ still returns
        # a real coroutine, which (with create_task also mocked below) would
        # itself go unawaited and just relocate the RuntimeWarning rather
        # than removing it. A synchronous MagicMock returns a plain object
        # instead, so no coroutine is ever created.
        patch.object(app_module, "_auto_reclaim_loop", MagicMock()),
    ):
        _call_gpu_release()

    assert lock_state_during_release == [True], (
        "_release_base_model_in_process must run with the GPU thread lock held"
    )
    assert not _gpu_thread_lock.locked(), "the lock must be released after the request completes"


def test_release_unloads_off_event_loop_thread():
    """``_release_base_model_in_process`` is dispatched via ``run_in_executor``
    — it must run on a worker thread, not the event-loop thread that services
    the coroutine."""
    from paramem.server import app as app_module

    release_thread_id = []

    def fake_release():
        release_thread_id.append(threading.get_ident())

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

    caller_thread_id = threading.get_ident()

    with (
        patch.dict(app_module._state, state_patch, clear=False),
        patch.object(app_module, "_release_base_model_in_process", side_effect=fake_release),
        patch.object(app_module, "_set_voice_pipeline_profile"),
        patch("asyncio.create_task"),
        # Also patch the coroutine function itself: even with create_task
        # mocked, `_auto_reclaim_loop(reclaim_interval)` is still evaluated
        # as create_task's argument, minting a real un-awaited coroutine
        # object that triggers a RuntimeWarning at garbage-collection time.
        # A plain MagicMock, NOT the AsyncMock patch.object would autospec by
        # default for an async target: an AsyncMock's __call__ still returns
        # a real coroutine, which (with create_task also mocked below) would
        # itself go unawaited and just relocate the RuntimeWarning rather
        # than removing it. A synchronous MagicMock returns a plain object
        # instead, so no coroutine is ever created.
        patch.object(app_module, "_auto_reclaim_loop", MagicMock()),
    ):
        _call_gpu_release()

    assert len(release_thread_id) == 1
    assert release_thread_id[0] != caller_thread_id, (
        "_release_base_model_in_process must run off the event-loop thread"
    )


def test_release_rechecks_consolidating_inside_lock():
    """TOCTOU guard: if ``consolidating`` flips to True after the pre-lock
    guard passes but before/while the lock is acquired, the in-lock recheck
    must still abort with the 503 shape (mirrors
    ``_apply_config_live``'s in-lock consolidating recheck).

    Simulated deterministically: a helper thread holds
    ``_gpu_thread_lock`` first; the request blocks acquiring it (via
    ``run_in_executor`` inside ``gpu_lock()``); while blocked, the test sets
    ``consolidating=True`` and releases the helper's hold, so the request's
    lock acquisition succeeds and its in-lock recheck must observe the flag.
    """
    from paramem.server import app as app_module
    from paramem.server.gpu_lock import _gpu_thread_lock

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

    helper_acquired = threading.Event()
    release_helper = threading.Event()

    def _hold_lock():
        _gpu_thread_lock.acquire()
        helper_acquired.set()
        release_helper.wait(timeout=5)
        # Flip consolidating on before releasing so the request's in-lock
        # recheck (which runs immediately after it wins the lock) observes it.
        app_module._state["consolidating"] = True
        _gpu_thread_lock.release()

    helper = threading.Thread(target=_hold_lock)
    helper.start()
    assert helper_acquired.wait(timeout=5), "helper thread failed to acquire the lock"

    async def _run_release_after_helper_signals():
        release_task = asyncio.create_task(app_module._gpu_release_internal())
        # Give the request a chance to start blocking on the lock acquire
        # before letting the helper thread release it.
        await asyncio.sleep(0.05)
        release_helper.set()
        return await release_task

    with (
        patch.dict(app_module._state, state_patch, clear=False),
        patch.object(app_module, "_release_base_model_in_process"),
        patch.object(app_module, "_set_voice_pipeline_profile"),
    ):
        result = asyncio.run(_run_release_after_helper_signals())

    helper.join(timeout=5)
    assert not _gpu_thread_lock.locked()

    assert isinstance(result, JSONResponse)
    assert result.status_code == 503
    body = result.body.decode("utf-8")
    assert "consolidating" in body


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
