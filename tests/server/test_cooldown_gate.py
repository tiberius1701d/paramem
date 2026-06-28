"""Tests for Tier-2 pre-task GPU cooldown gate.

Covers:

- Step T1: wait_for_cooldown helper (hot→cool, already-cool, None-sensor,
  bounded-timeout, disabled).
- Step T2: order assertions that each GPU-burst head calls the gate BEFORE the
  first GPU op and passes the correct per-site max-wait knob:
    T2a — boot preload (app._build_store_contents)
    T2b — fold workers (_run_interim_training / _run_full_cycle — source
           structural assertions, since both are nested closures)
    T2c — inference local-PA path (handle_chat PERSONAL branch)
    T2c neg — HA/SOTA-routed request does NOT call the inference gate

All tests run CPU-only — no model loading or GPU required.
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch

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
# Step T1 — wait_for_cooldown helper
# ---------------------------------------------------------------------------


class TestWaitForCooldown:
    """Unit tests for wait_for_cooldown — all CPU-only via _gpu_temp patching."""

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
# Step T2a — Boot preload order assertion (_build_store_contents)
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
# Step T2b — Fold worker order (structural source assertion)
# ---------------------------------------------------------------------------


class TestFoldWorkerCooldownOrder:
    """Structural source check: wait_for_cooldown appears before the first GPU
    training call in _run_interim_training, _run_full_cycle, and the simulate
    _run worker inside _await_bg_cycle.

    These workers are nested closures; source inspection is the only viable
    CPU-only verification without fully driving the outer endpoint functions.
    The check mirrors the pattern in test_preload_failfast.py::
    TestDegradeToCloudOnly::test_cuda_fault_persistent_in_permanent_cloud_only.
    """

    def test_run_interim_training_has_cooldown_before_run_consolidation(self):
        """_run_interim_training: wait_for_cooldown appears before loop.run_consolidation_cycle."""
        source = inspect.getsource(app_module._extract_and_start_training)
        cooldown_pos = source.find("wait_for_cooldown")
        training_pos = source.find("loop.run_consolidation_cycle")
        assert cooldown_pos != -1, (
            "wait_for_cooldown not found in _extract_and_start_training source "
            "(expected inside _run_interim_training closure)"
        )
        assert training_pos != -1, (
            "loop.run_consolidation_cycle not found in _extract_and_start_training source"
        )
        assert cooldown_pos < training_pos, (
            "wait_for_cooldown must appear before loop.run_consolidation_cycle in "
            "_run_interim_training; check that the gate is at the top of the worker body"
        )

    def test_run_interim_training_uses_fold_max_wait(self):
        """_run_interim_training source references cooldown_gate_max_wait_fold_s."""
        source = inspect.getsource(app_module._extract_and_start_training)
        assert "cooldown_gate_max_wait_fold_s" in source, (
            "_run_interim_training must pass cooldown_gate_max_wait_fold_s to wait_for_cooldown"
        )

    def test_run_full_cycle_has_cooldown_at_head(self):
        """_run_full_cycle source has wait_for_cooldown near the top, before _consume_pending."""
        source = inspect.getsource(app_module._run_full_consolidation_sync)
        cooldown_pos = source.find("wait_for_cooldown")
        consume_pending_pos = source.find("_consume_pending")
        assert cooldown_pos != -1, (
            "wait_for_cooldown not found in _run_full_consolidation_sync source "
            "(expected inside _run_full_cycle closure)"
        )
        assert consume_pending_pos != -1, (
            "_consume_pending not found in _run_full_consolidation_sync source"
        )
        assert cooldown_pos < consume_pending_pos, (
            "wait_for_cooldown must appear before _consume_pending in _run_full_cycle; "
            "the gate belongs at the very top of the worker body"
        )

    def test_run_full_cycle_uses_fold_max_wait(self):
        """_run_full_cycle source references cooldown_gate_max_wait_fold_s."""
        source = inspect.getsource(app_module._run_full_consolidation_sync)
        assert "cooldown_gate_max_wait_fold_s" in source, (
            "_run_full_cycle must pass cooldown_gate_max_wait_fold_s to wait_for_cooldown"
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


# ---------------------------------------------------------------------------
# Step T2c — Inference gate (handle_chat PERSONAL branch)
# ---------------------------------------------------------------------------


class TestInferenceCooldownGate:
    """Inference gate fires for PERSONAL-routed requests and is absent for HA/SOTA routes."""

    @staticmethod
    def _make_config(tmp_path):
        config = _make_config(tmp_path)
        return config

    @staticmethod
    def _make_personal_plan():
        """A RoutingPlan with PERSONAL intent and one probe step."""
        from paramem.server.router import Intent, RoutingPlan, RoutingStep

        step = RoutingStep(adapter_name="episodic", keys_to_probe=["key_001"])
        return RoutingPlan(intent=Intent.PERSONAL, steps=[step])

    @staticmethod
    def _make_general_plan():
        """A RoutingPlan with GENERAL intent and no steps."""
        from paramem.server.router import Intent, RoutingPlan

        return RoutingPlan(intent=Intent.GENERAL, steps=[])

    def test_personal_calls_cooldown_before_probe_and_reason(self, tmp_path):
        """PERSONAL-routed request: cooldown fires before _probe_and_reason."""
        from paramem.server.inference import ChatResult, handle_chat

        config = self._make_config(tmp_path)
        plan = self._make_personal_plan()
        call_order: list[str] = []

        model = MagicMock()
        memory_store = MagicMock()
        memory_store.iter_entries.return_value = []

        def fake_cooldown(*args, **kwargs):
            call_order.append("cooldown")

        def fake_probe_and_reason(*args, **kwargs):
            call_order.append("probe_and_reason")
            return ChatResult(text="answer", escalated=False)

        with (
            patch("paramem.server.inference._wait_for_cooldown", side_effect=fake_cooldown),
            patch(
                "paramem.server.inference._probe_and_reason",
                side_effect=fake_probe_and_reason,
            ),
            patch(
                "paramem.server.inference.sanitize_for_cloud",
                return_value=("sanitized query", []),
            ),
        ):
            handle_chat(
                "who am I",
                "conv_001",
                speaker="Alice",
                history=None,
                model=model,
                tokenizer=MagicMock(),
                config=config,
                router=MagicMock(route=MagicMock(return_value=plan)),
                speaker_id="speaker0",
                memory_store=memory_store,
            )

        assert "cooldown" in call_order, "_wait_for_cooldown must be called on PERSONAL path"
        assert "probe_and_reason" in call_order
        assert call_order.index("cooldown") < call_order.index("probe_and_reason"), (
            f"cooldown must precede _probe_and_reason; got: {call_order}"
        )

    def test_personal_passes_inference_max_wait(self, tmp_path):
        """PERSONAL path passes cooldown_gate_max_wait_inference_s as max_wait_s."""
        from paramem.server.inference import ChatResult, handle_chat

        config = self._make_config(tmp_path)
        config.vram.cooldown_gate_max_wait_inference_s = 17  # sentinel
        plan = self._make_personal_plan()

        captured: list[tuple] = []
        model = MagicMock()
        memory_store = MagicMock()
        memory_store.iter_entries.return_value = []

        with (
            patch(
                "paramem.server.inference._wait_for_cooldown",
                side_effect=lambda *a, **kw: captured.append(a),
            ),
            patch(
                "paramem.server.inference._probe_and_reason",
                return_value=ChatResult(text="ok", escalated=False),
            ),
            patch(
                "paramem.server.inference.sanitize_for_cloud",
                return_value=("q", []),
            ),
        ):
            handle_chat(
                "who am I",
                "conv_001",
                speaker="Alice",
                history=None,
                model=model,
                tokenizer=MagicMock(),
                config=config,
                router=MagicMock(route=MagicMock(return_value=plan)),
                speaker_id="speaker0",
                memory_store=memory_store,
            )

        assert captured, "_wait_for_cooldown must have been called"
        # max_wait_s is the second positional arg (args: threshold_c, max_wait_s, poll_s)
        assert captured[0][1] == 17, (
            f"inference gate must pass cooldown_gate_max_wait_inference_s=17; "
            f"got args={captured[0]}"
        )

    def test_ha_sota_routed_request_does_not_call_inference_gate(self, tmp_path):
        """HA/SOTA-routed (GENERAL intent) request must NOT call the inference cooldown gate."""
        from paramem.server.inference import ChatResult, handle_chat

        config = self._make_config(tmp_path)
        plan = self._make_general_plan()

        cooldown_calls: list = []
        model = MagicMock()
        memory_store = MagicMock()
        memory_store.iter_entries.return_value = []

        with (
            patch(
                "paramem.server.inference._wait_for_cooldown",
                side_effect=lambda *a, **kw: cooldown_calls.append(a),
            ),
            patch(
                "paramem.server.inference._escalate_to_ha_agent",
                return_value=ChatResult(text="HA says hi", escalated=True),
            ),
            patch(
                "paramem.server.inference.sanitize_for_cloud",
                return_value=("what time is it", []),
            ),
        ):
            handle_chat(
                "what time is it",
                "conv_002",
                speaker=None,
                history=None,
                model=model,
                tokenizer=MagicMock(),
                config=config,
                router=MagicMock(route=MagicMock(return_value=plan)),
                speaker_id=None,
                memory_store=memory_store,
            )

        assert not cooldown_calls, (
            "Inference cooldown gate must NOT fire for HA/SOTA-routed (non-PERSONAL) requests; "
            f"got {len(cooldown_calls)} call(s)"
        )
