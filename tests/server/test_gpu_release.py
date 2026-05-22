"""Tests for the POST /gpu/release endpoint and the _graceful_exit signal handler.

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

_graceful_exit tests (Fix 1 — boot-drain producer side):

- SIGTERM: _release_base_model_in_process + safe_empty_cache called BEFORE
  SystemExit; log line "graceful exit: GPU released" emitted.
- SIGUSR1: same release + cache flush before os._exit(1).
- Error in release: exception is caught (not re-raised), exit still proceeds.
"""

from __future__ import annotations

import asyncio
import signal
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
# Fix 1 — _graceful_exit: GPU release before process exit
# ---------------------------------------------------------------------------
# The signal handler is a closure defined inside lifespan.  We obtain a
# reference to it by running the lifespan in cloud_only mode (skips GPU paths)
# up to the signal registration point, then reading signal.getsignal().
# The lifespan is then cancelled so no server actually starts.  This lets us
# call the registered closure directly with mocks for both exit paths
# (SIGTERM / SIGUSR1) and assert the call order without actually exiting.


def _get_graceful_exit_handler(tmp_path):
    """Run the lifespan in cloud_only mode to register signal handlers, then
    return the registered SIGTERM handler (the same closure is used for both
    SIGTERM and SIGUSR1).

    cloud_only=True skips all CUDA/model-load paths so only minimal mocks
    are needed.  The lifespan is stopped with GeneratorExit (cancel on the
    asyncio task) before reaching the yield so the server never becomes ready.
    Signal state is restored on exit.
    """
    import asyncio

    from paramem.server import app as app_module
    from paramem.server.config import PathsConfig, ServerConfig

    class _Registered(Exception):
        """Raised by the signal spy after both handlers are registered."""

    from paramem.server.config import STTConfig, TTSConfig

    config = ServerConfig(model_name="mistral")
    config.cloud_only = True  # explicit cloud-only → skips CUDA checks entirely
    # Disable voice servers so Wyoming sockets are not bound during the test.
    config.stt = STTConfig(enabled=False)
    config.tts = TTSConfig(enabled=False)

    root = tmp_path / "data"
    config.paths = PathsConfig(
        data=root,
        sessions=root / "sessions",
        debug=root / "debug",
    )

    saved_state = {
        key: app_module._state.get(key) for key in ("config", "cloud_only_startup", "defer_model")
    }
    app_module._state["config"] = config
    app_module._state["cloud_only_startup"] = True  # explicit → no auto-reclaim task
    app_module._state["defer_model"] = False

    original_sigterm = signal.getsignal(signal.SIGTERM)
    original_sigusr1 = signal.getsignal(signal.SIGUSR1)

    # Short-circuit at detect_mode_switch, which is called AFTER the signal
    # handlers are registered (signal registration is at lifespan ~line 1879,
    # detect_mode_switch is called a few lines later).  _build_config_derived_state
    # is called BEFORE signal registration and cannot be used as the cut point.
    from paramem.server import active_store_migration as asm_module

    def short_circuit(*args, **kwargs):
        raise _Registered("handlers installed — stopping before migration detection")

    try:
        with (
            patch.object(app_module, "predict_base_bytes", return_value=None),
            patch.object(app_module, "_gpu_occupied", return_value=False),
            patch.object(app_module, "_build_config_derived_state"),
            patch.object(asm_module, "detect_mode_switch", short_circuit),
        ):

            async def _run():
                async with app_module.lifespan(app_module.app):
                    pass

            try:
                asyncio.run(_run())
            except _Registered:
                pass
    finally:
        for key, val in saved_state.items():
            if val is None:
                app_module._state.pop(key, None)
            else:
                app_module._state[key] = val

    handler = signal.getsignal(signal.SIGTERM)
    # Restore original handlers so test isolation is maintained.
    signal.signal(signal.SIGTERM, original_sigterm)
    signal.signal(signal.SIGUSR1, original_sigusr1)

    assert callable(handler) and handler not in (
        signal.SIG_DFL,
        signal.SIG_IGN,
    ), "SIGTERM handler was not registered during lifespan startup"
    return handler


def test_graceful_exit_sigterm_releases_gpu_before_exit(tmp_path):
    """SIGTERM path: _release_base_model_in_process then safe_empty_cache are
    called BEFORE SystemExit is raised, and in that order.

    The 'graceful exit: GPU released' log line must also be emitted.
    """
    from paramem.server import app as app_module

    handler = _get_graceful_exit_handler(tmp_path)

    call_order = []

    def fake_release():
        call_order.append("release")

    def fake_cache():
        call_order.append("cache")

    with (
        patch.object(app_module, "_release_base_model_in_process", fake_release),
        patch.object(app_module, "safe_empty_cache", fake_cache),
        patch.dict(app_module._state, {"session_buffer": None}, clear=False),
    ):
        try:
            handler(signal.SIGTERM, None)
        except SystemExit:
            pass  # expected — SIGTERM raises SystemExit(0)

    assert "release" in call_order, "_release_base_model_in_process not called on SIGTERM"
    assert "cache" in call_order, "safe_empty_cache not called on SIGTERM"
    release_idx = call_order.index("release")
    cache_idx = call_order.index("cache")
    assert release_idx < cache_idx, (
        f"_release_base_model_in_process must come before safe_empty_cache; got order={call_order}"
    )


def test_graceful_exit_sigusr1_releases_gpu_before_exit(tmp_path):
    """SIGUSR1 path: _release_base_model_in_process then safe_empty_cache are
    called BEFORE os._exit(1) is invoked.
    """
    from paramem.server import app as app_module

    handler = _get_graceful_exit_handler(tmp_path)

    call_order = []

    def fake_release():
        call_order.append("release")

    def fake_cache():
        call_order.append("cache")

    def fake_os_exit(code):
        call_order.append(f"os._exit({code})")

    with (
        patch.object(app_module, "_release_base_model_in_process", fake_release),
        patch.object(app_module, "safe_empty_cache", fake_cache),
        patch.object(app_module.os, "_exit", fake_os_exit),
        patch.dict(app_module._state, {"session_buffer": None}, clear=False),
    ):
        handler(signal.SIGUSR1, None)

    assert "release" in call_order, "_release_base_model_in_process not called on SIGUSR1"
    assert "cache" in call_order, "safe_empty_cache not called on SIGUSR1"
    assert "os._exit(1)" in call_order, "os._exit(1) not called on SIGUSR1"
    release_idx = call_order.index("release")
    cache_idx = call_order.index("cache")
    exit_idx = call_order.index("os._exit(1)")
    assert release_idx < cache_idx < exit_idx, (
        f"Expected release → cache → os._exit; got order={call_order}"
    )


def test_graceful_exit_release_error_does_not_prevent_exit(tmp_path):
    """If _release_base_model_in_process raises, the exception must be caught
    and the exit must still proceed (boundary error handling — the process is
    exiting; a teardown error must not hang it).
    """
    from paramem.server import app as app_module

    handler = _get_graceful_exit_handler(tmp_path)

    exited = []

    def raise_on_release():
        raise RuntimeError("simulated release failure")

    def fake_os_exit(code):
        exited.append(code)

    with (
        patch.object(app_module, "_release_base_model_in_process", raise_on_release),
        patch.object(app_module, "safe_empty_cache"),
        patch.object(app_module.os, "_exit", fake_os_exit),
        patch.dict(app_module._state, {"session_buffer": None}, clear=False),
    ):
        # SIGUSR1: error in release must NOT prevent os._exit(1)
        handler(signal.SIGUSR1, None)

    assert exited == [1], "os._exit(1) must still be called when release raises"
