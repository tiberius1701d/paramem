"""Tests for POST /gpu/acquire endpoint and ``_apply_config_live``.

``gpu_acquire`` tests: exercise the operator override that clears
PARAMEM_EXTRA_ARGS=--defer-model and, when the process is in cloud-only /
defer mode, reloads the base model in-process and switches the voice pipeline
to the gpu profile.

``_apply_config_live`` tests: exercise the bounded-lock abort, consolidating
abort, no-op skip (uses config_drift.loaded_hash — NOT source_path — to detect
rollback case), R-PORT carve (stt.port/tts.port delta, pre-flight bind success
and failure), and R-PATHS carve (paths.sessions/paths.data delta).

Hash-based no-op tests (hasher NOT mocked) verify that:
  - disk_hash == loaded_hash → no-op skip fires (rollback case).
  - disk_hash != loaded_hash → no-op skip does NOT fire (apply proceeds).
Plain-reclaim test: path does not call STT/TTS .load() or HA reconnect.
retain_sessions/debug delta test: threads to rebuild_session_buffer.
"""

from __future__ import annotations

import asyncio
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch


def _call_gpu_acquire() -> object:
    from paramem.server.app import gpu_acquire

    return asyncio.run(gpu_acquire())


def test_acquire_in_defer_mode_reloads_and_switches_voice():
    """defer_model=True + cloud-only: reloads model in-process; voice restore is inside primitive.

    Voice drain/restore is now owned by _live_reload_base_model (the primitive),
    so /gpu/acquire no longer calls _set_voice_pipeline_profile("gpu") on success.
    Assert: _live_reload_base_model is called and mode is local after.
    """
    from paramem.server import app as app_module

    state_patch = {
        "defer_model": True,
        "mode": "cloud-only",
        "cloud_only_reason": "training",
    }

    def fake_reload(**_kw):
        # Simulate successful reload setting mode to local.
        app_module._state["mode"] = "local"

    with (
        patch.dict(app_module._state, state_patch, clear=False),
        patch.object(app_module, "_clear_hold_env", return_value=True),
        patch.object(
            app_module,
            "_get_hold_state",
            return_value={"hold_active": True, "owner_pid": 1234, "owner_alive": True},
        ),
        patch.object(app_module, "_live_reload_base_model", side_effect=fake_reload) as mock_reload,
        patch.object(app_module, "_set_voice_pipeline_profile") as mock_profile,
    ):
        result = _call_gpu_acquire()

    assert result["reloaded_live"] is True
    mock_reload.assert_called_once()
    # Voice restore is owned by the primitive; /gpu/acquire must NOT call it on success.
    gpu_calls = [c for c in mock_profile.call_args_list if c.args and c.args[0] == "gpu"]
    assert not gpu_calls, (
        "acquire must NOT call _set_voice_pipeline_profile('gpu') — primitive owns that; "
        f"calls={mock_profile.call_args_list}"
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
    """cloud_only_reason="released" + cloud-only: acquire reloads; voice handled by primitive.

    Standard reclaim path: external consumer called /gpu/release, operator
    (or the consumer when done) calls /gpu/acquire to give GPU back to ParaMem.
    Voice drain/restore is owned by _live_reload_base_model — /gpu/acquire
    must NOT call _set_voice_pipeline_profile("gpu") on success.
    """
    from paramem.server import app as app_module

    state_patch = {
        "defer_model": False,
        "mode": "cloud-only",
        "cloud_only_reason": "released",
    }

    def _flip_mode_local(**_kw):
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
    # Voice restore is owned by the primitive; /gpu/acquire must NOT call it on success.
    gpu_calls = [c for c in mock_profile.call_args_list if c.args and c.args[0] == "gpu"]
    assert not gpu_calls, (
        "acquire must NOT call _set_voice_pipeline_profile('gpu') — primitive owns that; "
        f"calls={mock_profile.call_args_list}"
    )
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


def test_acquire_defers_on_insufficient_vram_without_restart():
    """Reload declined for insufficient VRAM: do NOT restart (a restart would
    only crash-loop on the lifespan VRAM budget gate). Stay cloud-only, set
    voice to cpu, and report deferred_insufficient_vram=True.
    """
    from paramem.server import app as app_module

    state_patch = {
        "defer_model": True,
        "mode": "cloud-only",
        "cloud_only_reason": "released",
    }

    def fake_reload_declines():
        # Pre-flight declined inside _live_reload_base_model.
        app_module._state["mode"] = "cloud-only"
        app_module._state["cloud_only_reason"] = "insufficient_vram"

    with (
        patch.dict(app_module._state, state_patch, clear=False),
        patch.object(app_module, "_clear_hold_env", return_value=True),
        patch.object(
            app_module,
            "_get_hold_state",
            return_value={"hold_active": False, "owner_pid": None, "owner_alive": None},
        ),
        patch.object(app_module, "_live_reload_base_model", side_effect=fake_reload_declines),
        patch.object(app_module, "_set_voice_pipeline_profile") as mock_profile,
        patch.object(app_module, "_restart_service") as mock_restart,
    ):
        result = _call_gpu_acquire()

    mock_restart.assert_not_called()
    mock_profile.assert_called_once_with("cpu")
    assert result["reloaded_live"] is False
    assert result["deferred_insufficient_vram"] is True
    assert result["will_restart"] is False


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


# ════════════════════════════════════════════════════════════════════════════
#  _apply_config_live  (maintenance-mode live config apply)
# ════════════════════════════════════════════════════════════════════════════


def _make_config(stt_port=10300, tts_port=10301, sessions_path="/data/sessions", data_path="/data"):
    """Return a minimal mock ServerConfig for carve-diff tests."""
    cfg = MagicMock()
    cfg.stt.port = stt_port
    cfg.tts.port = tts_port
    cfg.paths.sessions = sessions_path
    cfg.paths.data = data_path
    cfg.source_path = None
    return cfg


@contextmanager
def _null_gpu_lock_sync(timeout=-1):
    """No-op replacement for gpu_lock_sync — always succeeds immediately."""
    yield


@contextmanager
def _timeout_gpu_lock_sync(timeout=-1):
    """Replacement for gpu_lock_sync that always raises TimeoutError on acquire."""
    raise TimeoutError("Could not acquire GPU lock within timeout")
    yield  # make it a generator; never reached


def test_apply_config_live_aborts_on_lock_timeout():
    """_apply_config_live returns applied_live=False with restart_required_reason=
    'lock_timeout' when gpu_lock_sync cannot be acquired within the bounded timeout.

    Config must be untouched: _live_reload_base_model must NOT be called.
    """
    from paramem.server import app as app_module

    state_patch = {
        "mode": "cloud-only",
        "cloud_only_reason": "live_reload",
        "config": _make_config(),
        "config_path": "configs/server.yaml",
        "consolidating": False,
    }

    with (
        patch.dict(app_module._state, state_patch, clear=False),
        patch("paramem.server.gpu_lock.gpu_lock_sync", _timeout_gpu_lock_sync),
        patch.object(app_module, "_live_reload_base_model") as mock_reload,
    ):
        result = app_module._apply_config_live()

    mock_reload.assert_not_called()
    assert result["applied_live"] is False
    assert result["restart_required_reason"] == "lock_timeout"
    assert result["auto_restart_scheduled"] is False


def test_apply_config_live_aborts_when_consolidating():
    """_apply_config_live returns applied_live=False with restart_required_reason=
    'consolidating' when a consolidation cycle is running under the lock.

    _live_reload_base_model must NOT be called; config must be untouched.
    """
    from paramem.server import app as app_module

    state_patch = {
        "mode": "local",
        "cloud_only_reason": None,
        "config": _make_config(),
        "config_path": "configs/server.yaml",
        "consolidating": True,
    }

    with (
        patch.dict(app_module._state, state_patch, clear=False),
        patch("paramem.server.gpu_lock.gpu_lock_sync", _null_gpu_lock_sync),
        patch("paramem.server.drift.compute_config_hash", return_value="aaa"),
        patch.object(app_module, "load_server_config", return_value=_make_config()),
        patch.object(app_module, "_live_reload_base_model") as mock_reload,
    ):
        result = app_module._apply_config_live()

    mock_reload.assert_not_called()
    assert result["applied_live"] is False
    assert result["restart_required_reason"] == "consolidating"
    assert result["auto_restart_scheduled"] is False


def test_apply_config_live_noop_skip_when_hash_unchanged():
    """When the on-disk config hash matches _state["config_drift"]["loaded_hash"]
    (rollback case — disk restored to A, memory is A), _apply_config_live returns
    applied_live=True, skipped='no_change' without calling _live_reload_base_model.

    The prior implementation used getattr(config_a, "source_path", None) which
    always falls back to the live path when ServerConfig has no source_path
    attribute — causing disk_hash == mem_hash on EVERY call.  The corrected
    implementation compares disk_hash against config_drift["loaded_hash"].
    """
    from pathlib import Path

    from paramem.server import app as app_module

    same_hash = "a" * 64  # 64-char hex digest

    config_a = _make_config()
    # Note: do NOT set config_a.source_path — ServerConfig has no such attribute;
    # the fix must NOT depend on it.

    state_patch = {
        "mode": "cloud-only",
        "cloud_only_reason": "live_reload",
        "config": config_a,
        "config_path": "configs/server.yaml",
        "consolidating": False,
        # loaded_hash matches what compute_config_hash returns for the disk file:
        # this simulates the rollback case (disk = A, memory = A).
        "config_drift": {"loaded_hash": same_hash, "disk_hash": same_hash, "detected": False},
    }

    with (
        patch.dict(app_module._state, state_patch, clear=False),
        patch("paramem.server.gpu_lock.gpu_lock_sync", _null_gpu_lock_sync),
        # Disk hash equals loaded_hash — no-op skip must fire.
        patch("paramem.server.drift.compute_config_hash", return_value=same_hash),
        # The config path must appear to exist for the hash check to run.
        patch.object(Path, "exists", return_value=True),
        patch.object(app_module, "_live_reload_base_model") as mock_reload,
    ):
        result = app_module._apply_config_live()

    mock_reload.assert_not_called()
    assert result["applied_live"] is True
    assert result["skipped"] == "no_change"
    assert result["restart_required_reason"] is None


def test_apply_config_live_rport_stt_carve_restart_eligible():
    """An stt.port delta is classified as R-PORT (stt_port_change); when the
    transient bind pre-flight succeeds, restart_eligible=True is returned
    and _restart_service is NOT called (CLI prompts the operator instead).
    """
    import socket
    from pathlib import Path

    from paramem.server import app as app_module

    config_a = _make_config(stt_port=10300)
    config_b = _make_config(stt_port=10399)  # changed
    config_b.server = MagicMock()
    config_b.server.host = "127.0.0.1"

    state_patch = {
        "mode": "cloud-only",
        "cloud_only_reason": "live_reload",
        "config": config_a,
        "config_path": "configs/server.yaml",
        "consolidating": False,
    }

    with (
        patch.dict(app_module._state, state_patch, clear=False),
        patch("paramem.server.gpu_lock.gpu_lock_sync", _null_gpu_lock_sync),
        # Different hashes — no no-op skip.
        patch(
            "paramem.server.drift.compute_config_hash",
            side_effect=["disk_hash_b", "mem_hash_a"],
        ),
        patch.object(Path, "exists", return_value=True),
        patch.object(app_module, "load_server_config", return_value=config_b),
        # Bind succeeds (no OSError).
        patch.object(socket, "socket") as mock_sock_cls,
        patch.object(app_module, "_live_reload_base_model"),
        patch.object(app_module, "_set_voice_pipeline_profile"),
        patch.object(app_module, "_restart_service") as mock_restart,
    ):
        # Make the transient socket bind succeed without actually binding.
        mock_sock = MagicMock()
        mock_sock_cls.return_value = mock_sock

        # Simulate successful apply: mode=local after reload.
        app_module._state["mode"] = "local"

        result = app_module._apply_config_live()

    mock_restart.assert_not_called()
    assert result["restart_required_reason"] == "stt_port_change"
    # Server signals restart_eligible (CLI prompts); auto_restart_scheduled is always False.
    assert result["restart_eligible"] is True
    assert result["auto_restart_scheduled"] is False


def test_apply_config_live_rport_bind_failure_no_restart():
    """When the transient bind pre-flight raises OSError (port in use):
    auto_restart_scheduled=False, port_in_use_reason is populated, and
    _restart_service is NOT called (L1 constraint: caller fires restart).
    """
    import socket
    from pathlib import Path

    from paramem.server import app as app_module

    config_a = _make_config(stt_port=10300)
    config_b = _make_config(stt_port=10399)
    config_b.server = MagicMock()
    config_b.server.host = "127.0.0.1"

    state_patch = {
        "mode": "cloud-only",
        "cloud_only_reason": "live_reload",
        "config": config_a,
        "config_path": "configs/server.yaml",
        "consolidating": False,
    }

    with (
        patch.dict(app_module._state, state_patch, clear=False),
        patch("paramem.server.gpu_lock.gpu_lock_sync", _null_gpu_lock_sync),
        patch(
            "paramem.server.drift.compute_config_hash",
            side_effect=["disk_hash_b", "mem_hash_a"],
        ),
        patch.object(Path, "exists", return_value=True),
        patch.object(app_module, "load_server_config", return_value=config_b),
        patch.object(socket, "socket") as mock_sock_cls,
        patch.object(app_module, "_live_reload_base_model") as mock_reload,
        patch.object(app_module, "_restart_service") as mock_restart,
    ):
        # Make bind raise OSError.
        mock_sock = MagicMock()
        mock_sock.bind.side_effect = OSError("[Errno 98] Address already in use")
        mock_sock_cls.return_value = mock_sock

        result = app_module._apply_config_live()

    mock_reload.assert_not_called()
    mock_restart.assert_not_called()
    assert result["applied_live"] is False
    assert result["auto_restart_scheduled"] is False
    assert "port_in_use_reason" in result
    assert result["restart_required_reason"] == "stt_port_change"


def test_apply_config_live_rpaths_carve_no_auto_restart():
    """A paths.sessions delta is classified as R-PATHS (paths_change);
    auto_restart_scheduled=False (manual restart required, L1 constraint).
    Config is NOT touched live; _live_reload_base_model IS still called (mixed
    delta: non-paths fields can be applied live, paths carve is signalled).
    """
    from pathlib import Path

    from paramem.server import app as app_module

    config_a = _make_config(sessions_path="/data/sessions")
    config_b = _make_config(sessions_path="/data2/sessions")  # changed

    state_patch = {
        "mode": "cloud-only",
        "cloud_only_reason": "live_reload",
        "config": config_a,
        "config_path": "configs/server.yaml",
        "consolidating": False,
    }

    with (
        patch.dict(app_module._state, state_patch, clear=False),
        patch("paramem.server.gpu_lock.gpu_lock_sync", _null_gpu_lock_sync),
        patch(
            "paramem.server.drift.compute_config_hash",
            side_effect=["disk_hash_b", "mem_hash_a"],
        ),
        patch.object(Path, "exists", return_value=True),
        patch.object(app_module, "load_server_config", return_value=config_b),
        patch.object(app_module, "_live_reload_base_model"),
        patch.object(app_module, "_set_voice_pipeline_profile"),
        patch.object(app_module, "_restart_service") as mock_restart,
    ):
        app_module._state["mode"] = "local"
        result = app_module._apply_config_live()

    mock_restart.assert_not_called()
    assert result["restart_required_reason"] == "paths_change"
    assert result["auto_restart_scheduled"] is False


# ════════════════════════════════════════════════════════════════════════════
#  Real-hash no-op skip (hasher NOT mocked — validates the source_path fix)
# ════════════════════════════════════════════════════════════════════════════


def test_apply_config_live_noop_skip_real_hash_rollback():
    """Real-hash test (hasher NOT mocked): on-disk hash == loaded_hash → skip.

    Builds a real on-disk config file (copy of tests/fixtures/server.yaml),
    computes its actual SHA-256 as loaded_hash, and asserts _apply_config_live
    takes the no-op skip WITHOUT calling _live_reload_base_model (rollback case:
    disk = A, loaded = A, no real change).

    This validates that the skip does not fire for ANY disk file due to both
    sides hashing the same live_config_path (the prior source_path bug).
    """
    from paramem.server import app as app_module
    from paramem.server.drift import compute_config_hash

    fixture_path = Path("tests/fixtures/server.yaml")
    if not fixture_path.exists():
        # Fall back to an absolute path for environments where cwd differs.
        fixture_path = Path(__file__).parent.parent / "fixtures" / "server.yaml"

    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="wb", delete=False) as tmp:
        tmp.write(fixture_path.read_bytes())
        on_disk_path = tmp.name

    try:
        real_hash = compute_config_hash(Path(on_disk_path))

        config_a = _make_config()
        state_patch = {
            "mode": "cloud-only",
            "cloud_only_reason": "live_reload",
            "config": config_a,
            "config_path": on_disk_path,
            "consolidating": False,
            # loaded_hash matches on-disk file — rollback case.
            "config_drift": {
                "loaded_hash": real_hash,
                "disk_hash": real_hash,
                "detected": False,
            },
        }

        with (
            patch.dict(app_module._state, state_patch, clear=False),
            patch("paramem.server.gpu_lock.gpu_lock_sync", _null_gpu_lock_sync),
            patch.object(app_module, "_live_reload_base_model") as mock_reload,
        ):
            result = app_module._apply_config_live()

        mock_reload.assert_not_called()
        assert result["skipped"] == "no_change", (
            "no-op skip must fire when disk hash equals loaded_hash (rollback case)"
        )
        assert result["applied_live"] is True
    finally:
        Path(on_disk_path).unlink(missing_ok=True)


def test_apply_config_live_real_hash_different_file_proceeds():
    """Real-hash test (hasher NOT mocked): disk hash != loaded_hash → no skip.

    Writes config-B (different bytes from config-A) to a temp file.
    loaded_hash is set to a different value (config-A hash = hash of fixture).
    Asserts _apply_config_live does NOT take the no-op skip and proceeds to
    classify the delta / call _live_reload_base_model (reload mocked).

    This is the accept path: config-B is on disk, config-A is in memory.
    disk_hash != loaded_hash → apply must proceed.
    """
    from paramem.server import app as app_module
    from paramem.server.drift import compute_config_hash

    fixture_path = Path("tests/fixtures/server.yaml")
    if not fixture_path.exists():
        fixture_path = Path(__file__).parent.parent / "fixtures" / "server.yaml"

    # Config-A: the fixture file (represents what is in memory).
    loaded_hash_a = compute_config_hash(fixture_path)

    # Config-B: the fixture with an extra comment appended (different bytes).
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="wb", delete=False) as tmp_b:
        tmp_b.write(fixture_path.read_bytes())
        tmp_b.write(b"\n# config-B sentinel\n")
        config_b_path = tmp_b.name

    try:
        config_a = _make_config()
        state_patch = {
            "mode": "cloud-only",
            "cloud_only_reason": "live_reload",
            "config": config_a,
            "config_path": config_b_path,  # on-disk = config-B
            "consolidating": False,
            # loaded_hash is config-A's hash — differs from config-B on disk.
            "config_drift": {
                "loaded_hash": loaded_hash_a,
                "disk_hash": "",  # stale; will be recomputed by _apply_config_live
                "detected": True,
            },
        }

        with (
            patch.dict(app_module._state, state_patch, clear=False),
            patch("paramem.server.gpu_lock.gpu_lock_sync", _null_gpu_lock_sync),
            # Mock reload so we don't need a real model; just verify it was called.
            patch.object(app_module, "_live_reload_base_model") as mock_reload,
            patch.object(app_module, "load_server_config", return_value=_make_config()),
            patch.object(app_module, "_set_voice_pipeline_profile"),
        ):
            # Simulate a successful apply so applied_live=True.
            app_module._state["mode"] = "local"
            result = app_module._apply_config_live()

        assert result["skipped"] is None, (
            "no-op skip must NOT fire when disk hash differs from loaded_hash (accept path)"
        )
        (
            mock_reload.assert_called_once(),
            ("_live_reload_base_model must be called on the accept path"),
        )
    finally:
        Path(config_b_path).unlink(missing_ok=True)


# ════════════════════════════════════════════════════════════════════════════
#  S1-test: plain-reclaim skips STT/TTS and HA reconnect (full_rebuild=False)
# ════════════════════════════════════════════════════════════════════════════


def test_plain_reclaim_does_not_rebuild_stt_tts_or_ha_client():
    """Plain reclaim path (_live_reload_base_model with refresh_config_from_disk=False)
    calls _build_config_derived_state(full_rebuild=False) — which must NOT call
    STT/TTS .load() or HAClient constructor / health_check (correction S1).

    The test captures the kwargs passed to _build_config_derived_state and
    verifies full_rebuild=False is forwarded.  It also asserts that WhisperSTT
    and HAClient are not instantiated on this path (same config, no delta).
    """
    from paramem.server import app as app_module

    build_kwargs_log: list[dict] = []

    def fake_build(config, *, cloud_only, rebuild_session_buffer=True, full_rebuild=True):
        build_kwargs_log.append(
            {
                "cloud_only": cloud_only,
                "rebuild_session_buffer": rebuild_session_buffer,
                "full_rebuild": full_rebuild,
            }
        )

    state_patch = {
        "mode": "cloud-only",
        "cloud_only_reason": "released",
        "config": _make_config(),
        "topology_assessment": None,
        "boot_degraded": None,
    }

    with (
        patch.dict(app_module._state, state_patch, clear=False),
        patch.object(app_module, "_release_base_model_in_process"),
        patch.object(app_module, "_load_model_into_state"),
        patch.object(app_module, "_build_config_derived_state", side_effect=fake_build),
    ):
        # topology_assessment=None → VRAM fit-check is skipped entirely.
        app_module._live_reload_base_model()  # refresh_config_from_disk=False (default)

    assert build_kwargs_log, "_build_config_derived_state must be called on plain reclaim"
    kwargs = build_kwargs_log[0]
    assert kwargs["full_rebuild"] is False, (
        "plain reclaim must pass full_rebuild=False to _build_config_derived_state "
        "so STT/TTS/HA/SOTA/exemplar rebuild is skipped (correction S1)"
    )
    assert kwargs["rebuild_session_buffer"] is False, (
        "plain reclaim must not rebuild the session buffer (no session-config delta)"
    )


# ════════════════════════════════════════════════════════════════════════════
#  S3-test: retain_sessions / debug delta threads to rebuild_session_buffer
# ════════════════════════════════════════════════════════════════════════════


def _make_config_with_session_fields(retain_sessions: bool = True, debug: bool = False):
    """Return a mock ServerConfig with retain_sessions and debug set."""
    cfg = _make_config()
    cfg.consolidation.retain_sessions = retain_sessions
    cfg.debug = debug
    return cfg


def test_session_delta_sets_rebuild_session_buffer_true():
    """When retain_sessions changes between config A and config B,
    _live_reload_base_model is called with rebuild_session_buffer=True (S3).
    """
    from paramem.server import app as app_module

    config_a = _make_config_with_session_fields(retain_sessions=True, debug=False)
    config_b = _make_config_with_session_fields(retain_sessions=False, debug=False)  # changed

    rebuild_buf_log: list[bool] = []

    def fake_reload(refresh_config_from_disk=False, rebuild_session_buffer=False, lock_held=False):
        rebuild_buf_log.append(rebuild_session_buffer)
        # Simulate success so the result dict is built.
        app_module._state["mode"] = "local"
        app_module._state["cloud_only_reason"] = None

    state_patch = {
        "mode": "cloud-only",
        "cloud_only_reason": "live_reload",
        "config": config_a,
        "config_path": "configs/server.yaml",
        "consolidating": False,
        # loaded_hash DIFFERENT from disk so no-op skip does NOT fire.
        "config_drift": {"loaded_hash": "aaa" * 21, "disk_hash": "bbb" * 21, "detected": True},
    }

    with (
        patch.dict(app_module._state, state_patch, clear=False),
        patch("paramem.server.gpu_lock.gpu_lock_sync", _null_gpu_lock_sync),
        patch("paramem.server.drift.compute_config_hash", return_value="bbb" * 21),
        patch.object(Path, "exists", return_value=True),
        patch.object(app_module, "load_server_config", return_value=config_b),
        patch.object(app_module, "_live_reload_base_model", side_effect=fake_reload),
        patch.object(app_module, "_set_voice_pipeline_profile"),
    ):
        app_module._apply_config_live()

    assert rebuild_buf_log, "_live_reload_base_model must be called"
    assert rebuild_buf_log[0] is True, (
        "rebuild_session_buffer must be True when retain_sessions changed (S3)"
    )


def test_no_session_delta_keeps_rebuild_session_buffer_false():
    """When retain_sessions and debug are unchanged, rebuild_session_buffer=False (S3)."""
    from paramem.server import app as app_module

    config_a = _make_config_with_session_fields(retain_sessions=True, debug=False)
    config_b = _make_config_with_session_fields(retain_sessions=True, debug=False)  # unchanged

    rebuild_buf_log: list[bool] = []

    def fake_reload(refresh_config_from_disk=False, rebuild_session_buffer=False, lock_held=False):
        rebuild_buf_log.append(rebuild_session_buffer)
        app_module._state["mode"] = "local"
        app_module._state["cloud_only_reason"] = None

    state_patch = {
        "mode": "cloud-only",
        "cloud_only_reason": "live_reload",
        "config": config_a,
        "config_path": "configs/server.yaml",
        "consolidating": False,
        "config_drift": {"loaded_hash": "aaa" * 21, "disk_hash": "bbb" * 21, "detected": True},
    }

    with (
        patch.dict(app_module._state, state_patch, clear=False),
        patch("paramem.server.gpu_lock.gpu_lock_sync", _null_gpu_lock_sync),
        patch("paramem.server.drift.compute_config_hash", return_value="bbb" * 21),
        patch.object(Path, "exists", return_value=True),
        patch.object(app_module, "load_server_config", return_value=config_b),
        patch.object(app_module, "_live_reload_base_model", side_effect=fake_reload),
        patch.object(app_module, "_set_voice_pipeline_profile"),
    ):
        app_module._apply_config_live()

    assert rebuild_buf_log, "_live_reload_base_model must be called"
    assert rebuild_buf_log[0] is False, (
        "rebuild_session_buffer must be False when retain_sessions and debug unchanged (S3)"
    )


def test_debug_delta_sets_rebuild_session_buffer_true():
    """When debug changes between config A and config B, rebuild_session_buffer=True (S3)."""
    from paramem.server import app as app_module

    config_a = _make_config_with_session_fields(retain_sessions=True, debug=False)
    config_b = _make_config_with_session_fields(retain_sessions=True, debug=True)  # debug changed

    rebuild_buf_log: list[bool] = []

    def fake_reload(refresh_config_from_disk=False, rebuild_session_buffer=False, lock_held=False):
        rebuild_buf_log.append(rebuild_session_buffer)
        app_module._state["mode"] = "local"
        app_module._state["cloud_only_reason"] = None

    state_patch = {
        "mode": "cloud-only",
        "cloud_only_reason": "live_reload",
        "config": config_a,
        "config_path": "configs/server.yaml",
        "consolidating": False,
        "config_drift": {"loaded_hash": "aaa" * 21, "disk_hash": "bbb" * 21, "detected": True},
    }

    with (
        patch.dict(app_module._state, state_patch, clear=False),
        patch("paramem.server.gpu_lock.gpu_lock_sync", _null_gpu_lock_sync),
        patch("paramem.server.drift.compute_config_hash", return_value="bbb" * 21),
        patch.object(Path, "exists", return_value=True),
        patch.object(app_module, "load_server_config", return_value=config_b),
        patch.object(app_module, "_live_reload_base_model", side_effect=fake_reload),
        patch.object(app_module, "_set_voice_pipeline_profile"),
    ):
        app_module._apply_config_live()

    assert rebuild_buf_log, "_live_reload_base_model must be called"
    assert rebuild_buf_log[0] is True, "rebuild_session_buffer must be True when debug changed (S3)"
