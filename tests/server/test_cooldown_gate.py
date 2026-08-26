"""Tests for Tier-2 pre-task GPU cooldown gate.

Covers:

- wait_for_cooldown helper (hot→cool, already-cool, None-sensor,
  bounded-timeout, disabled).
- Order assertions that each GPU-burst fold-worker head calls the gate
  BEFORE the first GPU op and passes the correct per-site max-wait knob
  (source structural assertions, since the workers are nested closures).

The inference path is deliberately NOT gated: STT pre-heats the GPU past
any near-idle threshold and a per-request stall breaks voice-pipeline
client timeouts.

All tests run CPU-only — no model loading or GPU required.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from unittest.mock import patch

import pytest

import paramem.server.app as app_module


def _first_call_linenos(func, *call_names: str) -> dict[str, int]:
    """Line number of the first call to each name in ``call_names``, found by
    walking ``func``'s AST body — never its docstring or any other prose.

    A raw ``source.find(...)`` string search is fooled the moment a
    docstring happens to quote the same call the assertion is checking
    for. Walking the parsed AST, with the docstring statement excluded,
    is immune to what the prose says.
    """
    source = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(source)
    func_def = tree.body[0]
    assert isinstance(func_def, (ast.FunctionDef, ast.AsyncFunctionDef))
    body = func_def.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]  # drop the docstring statement
    found: dict[str, int] = {}
    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        if not isinstance(node, ast.Call):
            continue
        func_node = node.func
        name = func_node.id if isinstance(func_node, ast.Name) else None
        if name in call_names and name not in found:
            found[name] = node.lineno
    return found


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
    / ``_run_migration_on_worker``) do not have their own gate.  These
    workers are nested closures; source inspection is the only viable CPU-only
    verification without fully driving the outer endpoint functions.  The
    same "assert the invariant directly rather than driving the endpoint"
    idiom is used in test_preload_failfast.py::TestDegradeToCloudOnly::
    test_cuda_fault_persistent_in_permanent_cloud_only_reasons (a membership
    assertion there, source inspection here).
    """

    def test_stage_b_cycle_has_cooldown_before_body_dispatch(self):
        """_run_stage_b_cycle: wait_for_cooldown appears before body(loop, bt).

        AST-based (see ``_first_call_linenos``): a raw string search over
        the whole source can be fooled by the docstring's own narration of
        ``body(loop, bt)``.
        """
        linenos = _first_call_linenos(app_module._run_stage_b_cycle, "wait_for_cooldown", "body")
        assert "wait_for_cooldown" in linenos, "wait_for_cooldown not called in _run_stage_b_cycle"
        assert "body" in linenos, "body(loop, bt) dispatch not called in _run_stage_b_cycle"
        assert linenos["wait_for_cooldown"] < linenos["body"], (
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
        """The interim body does not duplicate the gate — it is owned by the primitive."""
        source = inspect.getsource(app_module._extract_and_start_training)
        assert "wait_for_cooldown" not in source, (
            "_extract_and_start_training must not call wait_for_cooldown directly — "
            "the entry cooldown gate belongs to _run_stage_b_cycle"
        )

    def test_run_full_cycle_has_no_own_cooldown_gate(self):
        """The full-cycle body does not duplicate the gate — it is owned by the primitive."""
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
