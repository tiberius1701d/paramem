"""Tests for _auto_reclaim_loop D3 rewire: in-process reload + voice gpu switch.

Verifies:
- Hold-cleared path uses _live_reload_base_model + _set_voice_pipeline_profile("gpu").
- _restart_service is NOT called on any path.
- Transient failures populate last_reclaim_error and WARN (not ERROR); loop continues.
- Retry increments attempt_count.
- Orphan branch unchanged from before D3.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from unittest.mock import patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_hold_state(active=False, owner_pid=None, owner_alive=None):
    return {
        "hold_active": active,
        "owner_pid": owner_pid,
        "owner_alive": owner_alive,
    }


def _make_null_gpu_lock():
    """Return a context-manager factory that does nothing (no-op lock)."""

    @asynccontextmanager
    async def _lock():
        yield

    return _lock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_auto_reclaim_calls_live_reload_then_voice_gpu_on_success():
    """Hold cleared: _live_reload_base_model called, then voice switched to gpu.

    _restart_service must NOT be called.
    last_reclaim_error cleared on success.
    """
    import paramem.server.app as app_module

    state_patch = {
        "last_reclaim_error": {"at": "t", "error": "old", "attempt_count": 1},
        "mode": "cloud-only",  # precondition: the loop only runs cloud-only
    }

    call_log = []
    tick_count = 0

    async def fake_sleep(_s):
        nonlocal tick_count
        tick_count += 1
        if tick_count > 2:
            raise asyncio.CancelledError

    async def fake_run_in_executor(_executor, fn, *args):
        call_log.append(fn.__name__ if hasattr(fn, "__name__") else str(fn))

    def _fake_reload_success(*_args, **_kwargs):
        # Faithfully simulate a successful in-process reload: the real
        # _live_reload_base_model sets mode="local" on success, which the loop
        # reads (app.py:11057) to take the gpu-restore branch. Establishing it
        # here makes the test independent of any incoming _state["mode"] a prior
        # test left behind (e.g. test_gpu_release leaves "cloud-only").
        app_module._state["mode"] = "local"

    with (
        patch.dict(app_module._state, state_patch, clear=False),
        patch("paramem.server.app._gpu_has_compute_processes", return_value=False),
        patch("paramem.server.app._get_hold_state", return_value=_make_hold_state(active=False)),
        patch(
            "paramem.server.app._live_reload_base_model", side_effect=_fake_reload_success
        ) as mock_reload,
        patch("paramem.server.app._set_voice_pipeline_profile") as mock_profile,
        patch("paramem.server.app._restart_service") as mock_restart,
        patch("paramem.server.gpu_lock.gpu_lock", _make_null_gpu_lock()),
        patch("asyncio.sleep", side_effect=fake_sleep),
    ):
        # Override run_in_executor on the loop so fn is called synchronously.
        async def _run():
            event_loop = asyncio.get_event_loop()
            original_rie = event_loop.run_in_executor

            async def _fake_rie(_exc, fn, *args):
                fn(*args)

            event_loop.run_in_executor = _fake_rie
            try:
                await app_module._auto_reclaim_loop(interval_minutes=0)
            except asyncio.CancelledError:
                pass
            finally:
                event_loop.run_in_executor = original_rie

        asyncio.run(_run())

        mock_reload.assert_called_once()
        mock_profile.assert_called_once_with("gpu")
        mock_restart.assert_not_called()
        assert app_module._state.get("last_reclaim_error") is None


def test_auto_reclaim_exits_when_already_local():
    """If the GPU was reclaimed externally (mode=local) during the loop's sleep —
    operator /gpu/acquire, a config apply, or a base-swap reload — the loop must
    exit WITHOUT a redundant reclaim, avoiding a release+reload churn of an
    already-loaded model (the ~10 s spurious cloud-only window seen post-swap).
    """
    import paramem.server.app as app_module

    state_patch = {"mode": "local"}  # reclaimed externally during the sleep

    async def fake_sleep(_s):
        return  # first sleep returns; the mode==local guard should then exit

    with (
        patch.dict(app_module._state, state_patch, clear=False),
        patch("paramem.server.app._gpu_has_compute_processes", return_value=False),
        patch("paramem.server.app._get_hold_state", return_value=_make_hold_state(active=False)),
        patch("paramem.server.app._live_reload_base_model") as mock_reload,
        patch("paramem.server.gpu_lock.gpu_lock", _make_null_gpu_lock()),
        patch("asyncio.sleep", side_effect=fake_sleep),
    ):

        async def _run():
            await app_module._auto_reclaim_loop(interval_minutes=0)

        asyncio.run(_run())

        mock_reload.assert_not_called()  # no churn reclaim when already local


def test_auto_reclaim_records_error_and_continues_on_reload_failure(caplog):
    """Transient reload failure: WARN logged (not ERROR), last_reclaim_error populated.

    Loop continues and retries on the next tick.
    """
    import paramem.server.app as app_module

    state_patch = {
        "last_reclaim_error": None,
        "mode": "cloud-only",  # precondition: the loop only runs cloud-only
    }

    reload_calls = 0
    tick_count = 0

    async def fake_sleep(_s):
        nonlocal tick_count
        tick_count += 1
        if tick_count > 3:
            raise asyncio.CancelledError

    def fake_reload():
        nonlocal reload_calls
        reload_calls += 1
        if reload_calls == 1:
            raise RuntimeError("CUDA load failed")
        # Second call succeeds: simulate the real reload setting mode="local"
        # (read by the loop at app.py:11057) so the success branch — which
        # clears last_reclaim_error — is reached regardless of any incoming
        # _state["mode"] from a prior test.
        app_module._state["mode"] = "local"

    named = logging.getLogger("paramem.server.app")
    orig_propagate = named.propagate
    named.propagate = True
    caplog.set_level(logging.WARNING, logger="paramem.server.app")
    named.addHandler(caplog.handler)

    try:
        with (
            patch.dict(app_module._state, state_patch, clear=False),
            patch("paramem.server.app._gpu_has_compute_processes", return_value=False),
            patch(
                "paramem.server.app._get_hold_state", return_value=_make_hold_state(active=False)
            ),
            patch("paramem.server.app._live_reload_base_model", side_effect=fake_reload),
            patch("paramem.server.app._set_voice_pipeline_profile"),
            patch("paramem.server.app._restart_service") as mock_restart,
            patch("paramem.server.gpu_lock.gpu_lock", _make_null_gpu_lock()),
            patch("asyncio.sleep", side_effect=fake_sleep),
        ):

            async def _run():
                event_loop = asyncio.get_event_loop()

                async def _fake_rie(_exc, fn, *args):
                    fn(*args)

                event_loop.run_in_executor = _fake_rie
                try:
                    await app_module._auto_reclaim_loop(interval_minutes=0)
                except asyncio.CancelledError:
                    pass

            asyncio.run(_run())

            mock_restart.assert_not_called()
            # After failure + success: error cleared.
            assert app_module._state.get("last_reclaim_error") is None
    finally:
        named.removeHandler(caplog.handler)
        named.propagate = orig_propagate

    warn_records = [
        r for r in caplog.records if r.levelno == logging.WARNING and "reclaim" in r.message.lower()
    ]
    assert warn_records, (
        f"Expected WARN about reclaim failure; got {[r.message for r in caplog.records]}"
    )


def test_auto_reclaim_retry_increments_attempt_count():
    """Two consecutive failures: attempt_count reaches 2."""
    import paramem.server.app as app_module

    state_patch = {
        "last_reclaim_error": None,
        "mode": "cloud-only",  # precondition: the loop only runs cloud-only
    }

    reload_calls = 0
    tick_count = 0

    async def fake_sleep(_s):
        nonlocal tick_count
        tick_count += 1
        if tick_count > 4:
            raise asyncio.CancelledError

    def fake_reload_two_fail():
        nonlocal reload_calls
        reload_calls += 1
        if reload_calls <= 2:
            raise RuntimeError(f"failure #{reload_calls}")

    with (
        patch.dict(app_module._state, state_patch, clear=False),
        patch("paramem.server.app._gpu_has_compute_processes", return_value=False),
        patch("paramem.server.app._get_hold_state", return_value=_make_hold_state(active=False)),
        patch("paramem.server.app._live_reload_base_model", side_effect=fake_reload_two_fail),
        patch("paramem.server.app._set_voice_pipeline_profile"),
        patch("paramem.server.app._restart_service"),
        patch("paramem.server.gpu_lock.gpu_lock", _make_null_gpu_lock()),
        patch("asyncio.sleep", side_effect=fake_sleep),
    ):
        max_count_seen = []

        async def _run():
            event_loop = asyncio.get_event_loop()

            async def _fake_rie(_exc, fn, *args):
                fn(*args)
                err = app_module._state.get("last_reclaim_error")
                if err:
                    max_count_seen.append(err.get("attempt_count", 0))

            event_loop.run_in_executor = _fake_rie
            try:
                await app_module._auto_reclaim_loop(interval_minutes=0)
            except asyncio.CancelledError:
                pass

        asyncio.run(_run())

    assert max(max_count_seen, default=0) >= 2, (
        f"Expected attempt_count >= 2; observed {max_count_seen}"
    )


def test_auto_reclaim_defers_when_reload_stays_cloud_only():
    """Reload declined for insufficient VRAM (mode stays cloud-only): the loop
    must NOT load STT/TTS on GPU or declare success. It forces voice to cpu and
    keeps polling — that is how a cloud-only server avoids squatting VRAM.
    """
    import paramem.server.app as app_module

    state_patch = {"last_reclaim_error": None, "mode": "cloud-only"}

    profile_calls = []
    reload_calls = 0
    tick_count = 0

    async def fake_sleep(_s):
        nonlocal tick_count
        tick_count += 1
        if tick_count > 3:
            raise asyncio.CancelledError

    def fake_reload_declines():
        nonlocal reload_calls
        reload_calls += 1
        # Pre-flight declined: base model not loaded, server stays cloud-only.
        app_module._state["mode"] = "cloud-only"
        app_module._state["cloud_only_reason"] = "insufficient_vram"

    with (
        patch.dict(app_module._state, state_patch, clear=False),
        patch("paramem.server.app._gpu_has_compute_processes", return_value=False),
        patch("paramem.server.app._get_hold_state", return_value=_make_hold_state(active=False)),
        patch("paramem.server.app._live_reload_base_model", side_effect=fake_reload_declines),
        patch(
            "paramem.server.app._set_voice_pipeline_profile",
            side_effect=lambda p: profile_calls.append(p),
        ),
        patch("paramem.server.app._restart_service") as mock_restart,
        patch("paramem.server.gpu_lock.gpu_lock", _make_null_gpu_lock()),
        patch("asyncio.sleep", side_effect=fake_sleep),
    ):

        async def _run():
            event_loop = asyncio.get_event_loop()

            async def _fake_rie(_exc, fn, *args):
                fn(*args)

            event_loop.run_in_executor = _fake_rie
            try:
                await app_module._auto_reclaim_loop(interval_minutes=0)
            except asyncio.CancelledError:
                pass

        asyncio.run(_run())

    mock_restart.assert_not_called()
    assert "gpu" not in profile_calls, (
        f"voice must not go to gpu when base not loaded: {profile_calls}"
    )
    assert "cpu" in profile_calls, "voice must be forced to cpu on a declined reclaim"
    assert reload_calls >= 2, f"loop must keep polling, not exit after one decline: {reload_calls}"
    assert app_module._state.get("last_reclaim_error") is None  # deferral is not an error


def test_auto_reclaim_orphan_path_exits_loop_without_restart():
    """Orphan: hold_active=True, owner_alive=False → emit WARN, exit loop (no restart).

    Regression guard for pre-D3 orphan detection behaviour.
    """
    import paramem.server.app as app_module

    state_patch = {
        "last_reclaim_error": None,
    }
    hold = _make_hold_state(active=True, owner_pid=99999, owner_alive=False)

    tick_count = 0

    async def fake_sleep(_s):
        nonlocal tick_count
        tick_count += 1
        if tick_count > 2:
            raise asyncio.CancelledError

    with (
        patch.dict(app_module._state, state_patch, clear=False),
        patch("paramem.server.app._gpu_has_compute_processes", return_value=False),
        patch("paramem.server.app._get_hold_state", return_value=hold),
        patch("paramem.server.app._live_reload_base_model") as mock_reload,
        patch("paramem.server.app._set_voice_pipeline_profile") as mock_profile,
        patch("paramem.server.app._restart_service") as mock_restart,
        patch("asyncio.sleep", side_effect=fake_sleep),
    ):

        async def _run():
            try:
                await app_module._auto_reclaim_loop(interval_minutes=0)
            except asyncio.CancelledError:
                pass

        asyncio.run(_run())

        mock_restart.assert_not_called()
        mock_reload.assert_not_called()
        mock_profile.assert_not_called()
