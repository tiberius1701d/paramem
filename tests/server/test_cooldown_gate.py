"""Tests for Tier-2 pre-task GPU cooldown gate.

Covers:

- wait_for_cooldown helper (hot→cool, already-cool, None-sensor,
  bounded-timeout, disabled).
- Order assertions that each GPU-burst head calls the gate BEFORE the
  first GPU op and passes the correct per-site max-wait knob:
    Boot preload (app._build_store_contents)
    Fold workers (_run_interim_training / _run_full_cycle — source
           structural assertions, since both are nested closures)

The inference path is deliberately NOT gated: STT pre-heats the GPU past
any near-idle threshold and a per-request stall breaks voice-pipeline
client timeouts (removed 2026-08-01).

All tests run CPU-only — no model loading or GPU required.
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch

import pytest

import paramem.server.app as app_module
from paramem.server.app import _build_store_contents

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(tmp_path):
    """Minimal ServerConfig with paths pointing at tmp_path."""
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
    """Inject config (and optionally model/tokenizer) into _state; return restore."""
    prior = {k: app_module._state.get(k) for k in ("config", "model", "tokenizer")}
    app_module._state["config"] = config
    app_module._state["model"] = model
    app_module._state["tokenizer"] = tokenizer

    def _restore():
        for k, v in prior.items():
            app_module._state[k] = v

    return _restore


# ---------------------------------------------------------------------------
# wait_for_cooldown helper
# ---------------------------------------------------------------------------


class TestWaitForCooldown:
    """Unit tests for wait_for_cooldown — all CPU-only via _gpu_temp patching."""

    @pytest.fixture(autouse=True)
    def _real_gate(self, monkeypatch):
        """These tests call the real ``wait_for_cooldown`` and assert its
        polling/return behaviour, so the conftest-wide
        ``PARAMEM_COOLDOWN_DISABLED=1`` (set for non-gpu test runs) must not
        short-circuit it here.
        """
        monkeypatch.delenv("PARAMEM_COOLDOWN_DISABLED", raising=False)

    def test_hot_to_cool_sequence(self):
        """Hot→cool sequence: polls until temp drops to or below threshold.

        _gpu_temp returns 60, 58, 51. Threshold 52. Expected: two poll sleeps
        (60 > 52, 58 > 52), then 51 ≤ 52 → exit. Returns 51.
        """
        from paramem.training.thermal_throttle import wait_for_cooldown

        temps = iter([60, 58, 51])
        sleep_calls: list[float] = []

        with (
            patch("paramem.training.thermal_throttle._gpu_temp", side_effect=lambda: next(temps)),
            patch("paramem.training.thermal_throttle.time.sleep", side_effect=sleep_calls.append),
        ):
            result = wait_for_cooldown(52, max_wait_s=30, poll_s=5, label="test")

        assert result == 51
        # Two sleeps: after 60 and after 58; 51 is already cool so no third sleep.
        assert sleep_calls == [5, 5]

    def test_already_cool_no_sleep(self):
        """Already cool: returns immediately without calling time.sleep."""
        from paramem.training.thermal_throttle import wait_for_cooldown

        sleep_calls: list[float] = []

        with (
            patch("paramem.training.thermal_throttle._gpu_temp", return_value=50),
            patch("paramem.training.thermal_throttle.time.sleep", side_effect=sleep_calls.append),
        ):
            result = wait_for_cooldown(52, max_wait_s=30, poll_s=5)

        assert result == 50
        assert sleep_calls == [], "sleep must NOT be called when already cool"

    def test_gpu_temp_none_instant_return(self):
        """_gpu_temp returns None → instant return without sleeping (no sensor = no block)."""
        from paramem.training.thermal_throttle import wait_for_cooldown

        sleep_calls: list[float] = []

        with (
            patch("paramem.training.thermal_throttle._gpu_temp", return_value=None),
            patch("paramem.training.thermal_throttle.time.sleep", side_effect=sleep_calls.append),
        ):
            result = wait_for_cooldown(52, max_wait_s=30, poll_s=5)

        assert result is None
        assert sleep_calls == [], "sleep must NOT be called when sensor is unavailable"

    def test_hot_forever_bounded_by_max_wait(self):
        """Hot-forever case: exits after max_wait_s with a WARNING — never loops past cap.

        Verifies the WARNING via patching logger.warning directly (pytest's caplog
        routing is environment-specific, as noted in test_bg_trainer_checkpoint_callback).
        """
        from paramem.training import thermal_throttle as _tt_mod
        from paramem.training.thermal_throttle import wait_for_cooldown

        sleep_calls: list[float] = []
        warning_messages: list[str] = []

        with (
            patch("paramem.training.thermal_throttle._gpu_temp", return_value=90),
            patch("paramem.training.thermal_throttle.time.sleep", side_effect=sleep_calls.append),
            patch.object(
                _tt_mod.logger,
                "warning",
                side_effect=lambda msg, *args: warning_messages.append(msg % args if args else msg),
            ),
        ):
            result = wait_for_cooldown(52, max_wait_s=10, poll_s=5, label="test")

        # Should have slept at most max_wait_s / poll_s = 2 times (5 + 5 = 10 >= max_wait_s).
        assert len(sleep_calls) <= 2, (
            f"Loop exceeded max_wait_s=10 / poll_s=5 cap; sleep_calls={sleep_calls}"
        )
        assert result == 90, "should return the still-hot temperature"
        assert warning_messages, "a WARNING must be logged when the cap is hit"
        assert any("proceeding" in msg for msg in warning_messages)

    def test_threshold_zero_noop(self):
        """threshold_c=0 disables the gate — _gpu_temp must not be consulted."""
        from paramem.training.thermal_throttle import wait_for_cooldown

        with patch("paramem.training.thermal_throttle._gpu_temp") as temp_mock:
            result = wait_for_cooldown(0, max_wait_s=30, poll_s=5)

        temp_mock.assert_not_called()
        assert result is None


# ---------------------------------------------------------------------------
# Boot preload order assertion (_build_store_contents)
# ---------------------------------------------------------------------------


class TestPreloadCooldownOrder:
    """wait_for_cooldown is called BEFORE _source.probe in _build_store_contents."""

    @staticmethod
    def _drive_build_store_contents(config, source_mock):
        """Drive _build_store_contents via simulate mode with one active key."""
        config.consolidation.mode = "simulate"
        config.inference.preload_cache = True

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

    def test_cooldown_called_before_probe(self, tmp_path):
        """Cooldown gate fires before _source.probe in the preload burst."""
        config = _make_config(tmp_path)
        restore = _inject_config(config)
        try:
            call_order: list[str] = []

            source_mock = MagicMock()
            source_mock.probe.side_effect = lambda *a, **kw: call_order.append("probe") or {}

            with patch(
                "paramem.server.app.wait_for_cooldown",
                side_effect=lambda *a, **kw: call_order.append("cooldown"),
            ):
                self._drive_build_store_contents(config, source_mock)

            assert "cooldown" in call_order, "wait_for_cooldown must be called during preload"
            assert "probe" in call_order, "probe must be called during preload"
            assert call_order.index("cooldown") < call_order.index("probe"), (
                f"cooldown must precede probe; got order: {call_order}"
            )
        finally:
            restore()

    def test_preload_passes_boot_max_wait(self, tmp_path):
        """_build_store_contents passes cooldown_gate_max_wait_boot_s as max_wait_s."""
        config = _make_config(tmp_path)
        config.vram.cooldown_gate_max_wait_boot_s = 42  # sentinel value
        restore = _inject_config(config)
        try:
            captured_kwargs: list[dict] = []

            source_mock = MagicMock()
            source_mock.probe.return_value = {}

            def _capture_cooldown(*args, **kwargs):
                captured_kwargs.append({"args": args, "kwargs": kwargs})

            with patch("paramem.server.app.wait_for_cooldown", side_effect=_capture_cooldown):
                self._drive_build_store_contents(config, source_mock)

            assert captured_kwargs, "wait_for_cooldown must have been called"
            # max_wait_s is the second positional arg
            assert captured_kwargs[0]["args"][1] == 42, (
                f"preload gate must pass cooldown_gate_max_wait_boot_s=42 as max_wait_s; "
                f"got args={captured_kwargs[0]['args']}"
            )
        finally:
            restore()


# ---------------------------------------------------------------------------
# Fold worker order (structural source assertion)
# ---------------------------------------------------------------------------


class TestFoldWorkerCooldownOrder:
    """Structural source check: wait_for_cooldown appears before the first GPU
    training call in the shared Stage-B cycle-lifecycle primitive
    (``_run_stage_b_cycle``, the entry point shared by the interim, full-cycle,
    and active-store-migration closures) and in the simulate ``_run`` worker
    inside ``_await_bg_cycle``.

    ``_run_stage_b_cycle`` owns the entry cooldown gate for all three Stage-B
    paths; the per-path bodies (``_run_interim_training`` / ``_run_full_cycle``
    / ``_run_migration_on_worker``) no longer have their own gate.  These
    workers are nested closures; source inspection is the only viable CPU-only
    verification without fully driving the outer endpoint functions.  The
    check mirrors the pattern in test_preload_failfast.py::
    TestDegradeToCloudOnly::test_cuda_fault_persistent_in_permanent_cloud_only.
    """

    def test_stage_b_cycle_has_cooldown_before_body_dispatch(self):
        """_run_stage_b_cycle: wait_for_cooldown appears before body(loop, bt)."""
        source = inspect.getsource(app_module._run_stage_b_cycle)
        cooldown_pos = source.find("wait_for_cooldown")
        body_pos = source.find("body(loop, bt)")
        assert cooldown_pos != -1, "wait_for_cooldown not found in _run_stage_b_cycle source"
        assert body_pos != -1, "body(loop, bt) dispatch not found in _run_stage_b_cycle source"
        assert cooldown_pos < body_pos, (
            "wait_for_cooldown must appear before the body(loop, bt) dispatch in "
            "_run_stage_b_cycle; check that the gate is at the top of the worker body"
        )

    def test_stage_b_cycle_uses_fold_max_wait(self):
        """_run_stage_b_cycle source references cooldown_gate_max_wait_fold_s."""
        source = inspect.getsource(app_module._run_stage_b_cycle)
        assert "cooldown_gate_max_wait_fold_s" in source, (
            "_run_stage_b_cycle must pass cooldown_gate_max_wait_fold_s to wait_for_cooldown"
        )

    def test_run_interim_training_has_no_own_cooldown_gate(self):
        """The interim body no longer duplicates the gate — it is owned by the primitive."""
        source = inspect.getsource(app_module._extract_and_start_training)
        assert "wait_for_cooldown" not in source, (
            "_extract_and_start_training must not call wait_for_cooldown directly — "
            "the entry cooldown gate belongs to _run_stage_b_cycle"
        )

    def test_run_full_cycle_has_no_own_cooldown_gate(self):
        """The full-cycle body no longer duplicates the gate — it is owned by the primitive."""
        source = inspect.getsource(app_module._run_full_consolidation_sync)
        assert "wait_for_cooldown" not in source, (
            "_run_full_consolidation_sync must not call wait_for_cooldown directly — "
            "the entry cooldown gate belongs to _run_stage_b_cycle"
        )

    def test_await_bg_cycle_run_has_cooldown_before_run_consolidation(self):
        """Simulate-fold _run: wait_for_cooldown appears before loop.run_consolidation_cycle.

        Searches within the _run closure body (not the outer docstring, which
        also references run_consolidation_cycle).
        """
        source = inspect.getsource(app_module._await_bg_cycle)
        # Restrict to the _run closure body — the docstring of _await_bg_cycle
        # also mentions loop.run_consolidation_cycle and would produce a false
        # ordering if we searched the full outer-function source.
        run_start = source.find("def _run()")
        assert run_start != -1, "def _run() closure not found in _await_bg_cycle source"
        run_body = source[run_start:]
        cooldown_pos = run_body.find("wait_for_cooldown")
        training_pos = run_body.find("loop.run_consolidation_cycle")
        assert cooldown_pos != -1, (
            "wait_for_cooldown not found in _await_bg_cycle._run body "
            "(expected as the first statement of the _run closure)"
        )
        assert training_pos != -1, (
            "loop.run_consolidation_cycle not found in _await_bg_cycle._run body"
        )
        assert cooldown_pos < training_pos, (
            "wait_for_cooldown must appear before loop.run_consolidation_cycle in "
            "_await_bg_cycle._run; check that the gate is at the top of the worker body"
        )

    def test_await_bg_cycle_run_uses_fold_max_wait(self):
        """Simulate-fold _run source references cooldown_gate_max_wait_fold_s."""
        source = inspect.getsource(app_module._await_bg_cycle)
        assert "cooldown_gate_max_wait_fold_s" in source, (
            "_await_bg_cycle._run must pass cooldown_gate_max_wait_fold_s to wait_for_cooldown"
        )
