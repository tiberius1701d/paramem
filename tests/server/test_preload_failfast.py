"""Tests for Tier-1 CUDA fail-fast containment.

Covers Steps 2-4 of the fail-fast plan:

- Step 2: probe re-raise vs benign-swallow in _build_store_contents.
- Step 3: _fail_fast_cuda patches os._exit; the "ready" log is NOT emitted
  on the fatal path; _release_base_model_in_process is never called.
- Step 4: crash-loop counter (_record_cuda_fatal_exit / _cuda_crashloop_exhausted);
  exhausted counter → _degrade_to_cloud_only, NOT os._exit;
  _degrade_to_cloud_only("cuda_fault_persistent") sets state correctly and
  is in permanent_cloud_only.

All tests run CPU-only — no model loading or GPU required.
"""

from __future__ import annotations

import inspect
import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import paramem.server.app as app_module
from paramem.server.app import (
    _cuda_crashloop_exhausted,
    _cuda_liveness_canary,
    _degrade_to_cloud_only,
    _fail_fast_cuda,
    _record_cuda_fatal_exit,
)
from paramem.server.vram_guard import is_fatal_cuda_fault

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(tmp_path: Path):
    """Minimal ServerConfig with state dir pointing at tmp_path."""
    from paramem.server.config import PathsConfig, ServerConfig

    config = ServerConfig()
    ha = tmp_path / "ha"
    config.paths = PathsConfig(
        data=ha,
        sessions=ha / "sessions",
        debug=ha / "debug",
    )
    (ha / "adapters").mkdir(parents=True, exist_ok=True)
    return config


def _inject_config(config, *, model=None, tokenizer=None):
    """Inject config (and optionally model/tokenizer) into _state; return a restore callable."""
    prior = {k: app_module._state.get(k) for k in ("config", "model", "tokenizer")}

    app_module._state["config"] = config
    app_module._state["model"] = model
    app_module._state["tokenizer"] = tokenizer

    def _restore():
        for k, v in prior.items():
            app_module._state[k] = v

    return _restore


# ---------------------------------------------------------------------------
# Step 2 — probe re-raise vs benign-swallow
# ---------------------------------------------------------------------------


class TestProbeReraise:
    """_build_store_contents must re-raise a fatal CUDA fault and swallow a benign one."""

    @staticmethod
    def _run_build_store_contents_with_source(config, source_mock):
        """Drive _build_store_contents with a mock _source injected at the simulate path.

        Uses mode='simulate' so the DiskMemorySource constructor path is taken,
        then replaces the constructed source with source_mock by patching the
        class.  The registry is stubbed with one active key so _source.probe()
        is reached.
        """
        from paramem.server.app import _build_store_contents

        config.consolidation.mode = "simulate"
        config.inference.preload_cache = True

        # A fake KeyRegistry that has one active key.
        fake_reg = MagicMock()
        fake_reg.list_active.return_value = ["key_001"]

        with (
            patch(
                "paramem.memory.store.MemoryStore.read_registries_from_disk",
                return_value={"episodic": fake_reg},
            ),
            patch("paramem.memory.source.DiskMemorySource", return_value=source_mock),
        ):
            return _build_store_contents(config, model=None, tokenizer=None)

    def test_fatal_probe_raises(self, tmp_path):
        """Fatal CUDA fault from probe propagates — not swallowed to boot_degraded."""
        config = _make_config(tmp_path)
        restore = _inject_config(config)
        try:
            fatal_exc = RuntimeError("CUDA error: an illegal memory access was encountered")
            assert is_fatal_cuda_fault(fatal_exc)

            source_mock = MagicMock()
            source_mock.probe.side_effect = fatal_exc

            with pytest.raises(RuntimeError, match="illegal memory access"):
                self._run_build_store_contents_with_source(config, source_mock)
        finally:
            restore()

    def test_benign_probe_failure_sets_boot_degraded(self, tmp_path):
        """Benign RuntimeError from probe → _results={}, boot_degraded set, no raise."""
        config = _make_config(tmp_path)
        restore = _inject_config(config)
        try:
            benign_exc = RuntimeError("simulated transient network error")
            assert not is_fatal_cuda_fault(benign_exc)

            source_mock = MagicMock()
            source_mock.probe.side_effect = benign_exc

            _, _, _, stats = self._run_build_store_contents_with_source(config, source_mock)
            # With _results={} and one active key, hits == 0 < total == 1
            # → boot_degraded set.
            assert stats["boot_degraded"] is not None
            assert stats["boot_degraded"]["reason"] == "preload_partial"
            assert stats["boot_degraded"]["hits"] == 0
        finally:
            restore()

    def test_fatal_probe_does_not_set_boot_degraded(self, tmp_path):
        """Fatal CUDA fault re-raises before boot_degraded can be set."""
        config = _make_config(tmp_path)
        restore = _inject_config(config)
        try:
            fatal_exc = RuntimeError("CUDA error: an illegal memory access was encountered")
            source_mock = MagicMock()
            source_mock.probe.side_effect = fatal_exc

            with pytest.raises(RuntimeError):
                self._run_build_store_contents_with_source(config, source_mock)
            # boot_degraded never set because the function raised before reaching that line
        finally:
            restore()


# ---------------------------------------------------------------------------
# Step 3 — _fail_fast_cuda / _cuda_liveness_canary
# ---------------------------------------------------------------------------


class TestFailFastCuda:
    """_fail_fast_cuda calls os._exit(1) on a fresh context, NOT _release_base_model_in_process."""

    def test_fail_fast_calls_os_exit_with_1(self, tmp_path, caplog):
        """_fail_fast_cuda → os._exit(1) when crash-loop is not exhausted.

        Also asserts that the 'ParaMem server ready' log is NOT emitted on
        the fatal fail-fast path (plan §7): the fail-fast handler must exit
        the process before the 'ready' advertisement is ever logged.
        """
        import logging

        config = _make_config(tmp_path)
        restore = _inject_config(config)
        try:
            exc = RuntimeError("illegal memory access")

            exits: list[int] = []

            def _fake_exit(code):
                exits.append(code)
                raise SystemExit(code)  # prevent actual exit in tests

            with caplog.at_level(logging.INFO):
                with patch.object(app_module.os, "_exit", side_effect=_fake_exit):
                    with pytest.raises(SystemExit):
                        _fail_fast_cuda(exc, "test_phase")

            assert exits == [1]
            ready_messages = [
                r.message for r in caplog.records if "server ready" in r.message.lower()
            ]
            assert not ready_messages, (
                "Fatal fail-fast path must NOT emit 'ParaMem server ready'; "
                f"found: {ready_messages}"
            )
        finally:
            restore()

    def test_fail_fast_never_calls_release_base_model(self, tmp_path):
        """_fail_fast_cuda MUST NOT call _release_base_model_in_process.

        safe_empty_cache→synchronize re-hits the sticky error on a poisoned
        context.  os._exit is the only safe teardown.
        """
        config = _make_config(tmp_path)
        restore = _inject_config(config)
        try:
            exc = RuntimeError("illegal memory access")
            release_called: list[bool] = []

            def _fail_if_called():
                release_called.append(True)
                raise AssertionError(
                    "_release_base_model_in_process must NOT be called by _fail_fast_cuda"
                )

            with (
                patch.object(
                    app_module,
                    "_release_base_model_in_process",
                    side_effect=_fail_if_called,
                ),
                patch.object(
                    app_module.os,
                    "_exit",
                    side_effect=lambda c: (_ for _ in ()).throw(SystemExit(c)),
                ),
            ):
                with pytest.raises(SystemExit):
                    _fail_fast_cuda(exc, "test_phase")

            assert release_called == [], "os._exit-only-recovery invariant violated"
        finally:
            restore()

    def test_liveness_canary_noop_when_no_cuda(self):
        """_cuda_liveness_canary is a no-op when CUDA is unavailable."""
        with patch("paramem.server.app.torch.cuda.is_available", return_value=False):
            with patch("paramem.server.app.torch.cuda.synchronize") as sync_mock:
                _cuda_liveness_canary()
        sync_mock.assert_not_called()

    def test_liveness_canary_noop_when_no_model(self):
        """_cuda_liveness_canary is a no-op when no model is loaded."""
        prior = app_module._state.get("model")
        try:
            app_module._state["model"] = None
            with patch("paramem.server.app.torch.cuda.is_available", return_value=True):
                with patch("paramem.server.app.torch.cuda.synchronize") as sync_mock:
                    _cuda_liveness_canary()
            sync_mock.assert_not_called()
        finally:
            app_module._state["model"] = prior

    def test_liveness_canary_calls_synchronize_when_model_loaded(self):
        """_cuda_liveness_canary calls torch.cuda.synchronize() unguarded when model present."""
        prior = app_module._state.get("model")
        try:
            app_module._state["model"] = MagicMock()
            with patch("paramem.server.app.torch.cuda.is_available", return_value=True):
                with patch("paramem.server.app.torch.cuda.synchronize") as sync_mock:
                    _cuda_liveness_canary()
            sync_mock.assert_called_once_with()
        finally:
            app_module._state["model"] = prior


# ---------------------------------------------------------------------------
# Step 4 — crash-loop counter + _degrade_to_cloud_only
# ---------------------------------------------------------------------------


class TestCrashLoopCounter:
    """_record_cuda_fatal_exit / _cuda_crashloop_exhausted behaviour."""

    def test_below_burst_not_exhausted(self, tmp_path):
        """Single record → exhausted=False (default burst is 2)."""
        config = _make_config(tmp_path)
        restore = _inject_config(config)
        try:
            _record_cuda_fatal_exit()
            assert _cuda_crashloop_exhausted() is False
        finally:
            restore()

    def test_at_burst_exhausted(self, tmp_path):
        """Two records → exhausted=True (burst=2)."""
        config = _make_config(tmp_path)
        restore = _inject_config(config)
        try:
            _record_cuda_fatal_exit()
            _record_cuda_fatal_exit()
            assert _cuda_crashloop_exhausted() is True
        finally:
            restore()

    def test_stale_entries_pruned(self, tmp_path):
        """Entries older than cuda_fault_history_window_s are pruned on read."""
        config = _make_config(tmp_path)
        restore = _inject_config(config)
        try:
            state_dir = (config.paths.data / "state").resolve()
            state_dir.mkdir(parents=True, exist_ok=True)
            history_file = state_dir / "cuda_fault_history.json"

            # Write two stale entries (well outside the window).
            stale_ts = time.time() - config.vram.cuda_fault_history_window_s - 10
            history_file.write_text(json.dumps([stale_ts, stale_ts]), encoding="utf-8")

            # Still fresh — stale entries don't count.
            assert _cuda_crashloop_exhausted() is False
        finally:
            restore()

    def test_no_history_file_not_exhausted(self, tmp_path):
        """No history file → not exhausted (first boot)."""
        config = _make_config(tmp_path)
        restore = _inject_config(config)
        try:
            assert _cuda_crashloop_exhausted() is False
        finally:
            restore()

    def test_corrupted_history_file_not_exhausted(self, tmp_path):
        """Corrupted history file → not-exhausted (disk error prefers os._exit retry)."""
        config = _make_config(tmp_path)
        restore = _inject_config(config)
        try:
            state_dir = (config.paths.data / "state").resolve()
            state_dir.mkdir(parents=True, exist_ok=True)
            (state_dir / "cuda_fault_history.json").write_text("NOT VALID JSON", encoding="utf-8")
            assert _cuda_crashloop_exhausted() is False
        finally:
            restore()

    def test_exhausted_counter_calls_degrade_not_exit(self, tmp_path):
        """When crash-loop is exhausted, _fail_fast_cuda calls _degrade_to_cloud_only."""
        config = _make_config(tmp_path)
        restore = _inject_config(config)
        try:
            # Pre-fill history to burst level.
            _record_cuda_fatal_exit()
            _record_cuda_fatal_exit()
            assert _cuda_crashloop_exhausted() is True

            exc = RuntimeError("illegal memory access")
            degrade_calls: list[str] = []
            exits: list[int] = []

            with (
                patch.object(
                    app_module,
                    "_degrade_to_cloud_only",
                    side_effect=lambda r: degrade_calls.append(r),
                ),
                patch.object(
                    app_module.os,
                    "_exit",
                    side_effect=lambda c: exits.append(c),
                ),
            ):
                _fail_fast_cuda(exc, "test_phase")

            assert degrade_calls == ["cuda_fault_persistent"]
            assert exits == [], "os._exit must NOT be called when crash-loop is exhausted"
        finally:
            restore()


class TestDegradeToCloudOnly:
    """_degrade_to_cloud_only sets model/tokenizer to None, sets reason, notifies."""

    def test_cuda_fault_persistent_sets_state(self, tmp_path):
        """_degrade_to_cloud_only('cuda_fault_persistent') updates _state correctly."""
        config = _make_config(tmp_path)
        restore = _inject_config(config, model=MagicMock(), tokenizer=MagicMock())
        prior_reason = app_module._state.get("cloud_only_reason")
        try:
            with (
                patch.object(app_module, "_release_base_model_in_process"),
                patch.object(app_module, "notify_server"),
            ):
                _degrade_to_cloud_only("cuda_fault_persistent")

            assert app_module._state["cloud_only_reason"] == "cuda_fault_persistent"
            assert app_module._state["model"] is None
            assert app_module._state["tokenizer"] is None
        finally:
            app_module._state["cloud_only_reason"] = prior_reason
            restore()

    def test_cuda_fault_persistent_in_permanent_cloud_only(self):
        """'cuda_fault_persistent' must be in the permanent_cloud_only membership set.

        If the reason is not in the set, the server would try to auto-reclaim
        the GPU, re-poisoning the context.  Verified by inspecting the lifespan
        source — a future refactor that removes the reason is caught here.
        """
        source = inspect.getsource(app_module.lifespan)
        assert "cuda_fault_persistent" in source, (
            "'cuda_fault_persistent' must appear in the lifespan permanent_cloud_only "
            "membership set so the GPU is never auto-reclaimed after a fatal CUDA fault"
        )

    def test_degrade_calls_release_and_notify(self, tmp_path):
        """_degrade_to_cloud_only calls _release_base_model_in_process and notify_server."""
        config = _make_config(tmp_path)
        restore = _inject_config(config, model=MagicMock(), tokenizer=MagicMock())
        prior_reason = app_module._state.get("cloud_only_reason")
        try:
            release_calls: list[bool] = []
            notify_calls: list = []

            with (
                patch.object(
                    app_module,
                    "_release_base_model_in_process",
                    side_effect=lambda: release_calls.append(True),
                ),
                patch.object(
                    app_module,
                    "notify_server",
                    side_effect=lambda s: notify_calls.append(s),
                ),
            ):
                _degrade_to_cloud_only("cuda_fault_persistent")

            assert len(release_calls) == 1
            assert len(notify_calls) == 1
        finally:
            app_module._state["cloud_only_reason"] = prior_reason
            restore()
