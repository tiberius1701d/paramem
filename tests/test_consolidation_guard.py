"""Tests for ConsolidationLoop.guard_trial_state and TrialActiveError (Slice 3b.2).

No GPU — all tests use mocked ConsolidationLoop instances.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from paramem.training.consolidation import ConsolidationLoop, TrialActiveError


def _make_loop() -> ConsolidationLoop:
    """Build a ConsolidationLoop without loading any model or GPU resources."""
    loop = ConsolidationLoop.__new__(ConsolidationLoop)
    # Provide just enough attributes for guard_trial_state and run_cycle guard.
    loop.state_provider = None
    # Minimal attributes required for run_cycle before _run_extract_graph is called.
    loop.cycle_count = 0
    return loop


# ---------------------------------------------------------------------------
# guard_trial_state behaviour
# ---------------------------------------------------------------------------


class TestGuardTrialState:
    def test_guard_trial_state_raises_when_trial_active(self):
        """state["migration"]["state"] == "TRIAL" → TrialActiveError."""
        loop = _make_loop()
        state = {"migration": {"state": "TRIAL"}}
        with pytest.raises(TrialActiveError):
            loop.guard_trial_state(state)

    def test_guard_trial_state_noop_when_live(self):
        """LIVE → no exception."""
        loop = _make_loop()
        state = {"migration": {"state": "LIVE"}}
        loop.guard_trial_state(state)  # must not raise

    def test_guard_trial_state_noop_when_staging(self):
        """STAGING → no exception."""
        loop = _make_loop()
        state = {"migration": {"state": "STAGING"}}
        loop.guard_trial_state(state)

    def test_guard_trial_state_noop_when_state_none(self):
        """state=None (experiment path) → no exception."""
        loop = _make_loop()
        loop.guard_trial_state(None)  # must not raise

    def test_guard_trial_state_noop_when_migration_missing(self):
        """state without 'migration' key → no exception."""
        loop = _make_loop()
        loop.guard_trial_state({"foo": "bar"})  # must not raise

    def test_guard_trial_state_error_message_is_human_readable(self):
        """TrialActiveError message mentions rollback."""
        loop = _make_loop()
        state = {"migration": {"state": "TRIAL"}}
        with pytest.raises(TrialActiveError, match="rollback"):
            loop.guard_trial_state(state)


# ---------------------------------------------------------------------------
# /scheduled-tick and /consolidate return 409 when TRIAL is active
# ---------------------------------------------------------------------------


class TestRunCycleGuard:
    """Fix 5: run_cycle raises TrialActiveError when state_provider returns TRIAL state."""

    def test_run_cycle_blocks_when_trial_active(self):
        """run_cycle raises TrialActiveError before extraction when TRIAL is active.

        Verifies the state_provider injection: loop.state_provider returns a
        TRIAL state dict; run_cycle must raise TrialActiveError before any
        extraction (verified by patching _run_extract_graph to AssertionError).
        """
        from unittest.mock import patch

        trial_state = {"migration": {"state": "TRIAL"}}
        loop = _make_loop()
        loop.state_provider = lambda: trial_state

        with patch.object(
            loop.__class__,
            "_run_extract_graph",
            side_effect=AssertionError("should not be called — guard must raise first"),
        ):
            with pytest.raises(TrialActiveError):
                loop.run_cycle("some transcript", "session-1")

    def test_run_cycle_proceeds_when_live(self):
        """run_cycle does NOT block when state_provider returns LIVE state.

        This test only verifies that the guard does not raise; it does NOT run
        the full extraction (which requires a real model).
        """
        from unittest.mock import patch

        live_state = {"migration": {"state": "LIVE"}}
        loop = _make_loop()
        loop.state_provider = lambda: live_state

        # Patch the cycle body so no real GPU work happens.
        with patch.object(
            loop.__class__,
            "_run_extract_graph",
            side_effect=RuntimeError("extract called — guard did not block"),
        ):
            # Guard should pass; RuntimeError from extract (not TrialActiveError).
            with pytest.raises(RuntimeError, match="extract called"):
                loop.run_cycle("some transcript", "session-1")

    def test_run_cycle_proceeds_when_no_state_provider(self):
        """run_cycle is unaffected when state_provider is None (experiment path)."""
        from unittest.mock import patch

        loop = _make_loop()
        # state_provider defaults to None for experiment scripts.
        assert loop.state_provider is None

        with patch.object(
            loop.__class__,
            "_run_extract_graph",
            side_effect=RuntimeError("extract called — guard did not block"),
        ):
            # No TrialActiveError; RuntimeError from extract is expected.
            with pytest.raises(RuntimeError, match="extract called"):
                loop.run_cycle("some transcript", "session-1")


class TestEndpointGuards:
    """Verify that the HTTP endpoints refuse new cycles during TRIAL.

    Uses TestClient without touching GPU (no model load).
    """

    def _make_state(self, migration_state: str = "TRIAL") -> dict:
        config = MagicMock()
        config.adapter_dir = MagicMock()
        return {
            "model": None,
            "config": config,
            "config_path": "",
            "consolidating": False,
            "migration": {"state": migration_state, "trial": None, "recovery_required": []},
            "server_started_at": "2026-04-22T00:00:00+00:00",
            "mode": "normal",
            "background_trainer": None,
        }

    def test_scheduled_tick_409_when_trial_active(self, monkeypatch):
        """POST /scheduled-tick returns 409 trial_active when TRIAL is active."""
        from fastapi.testclient import TestClient

        import paramem.server.app as app_module

        state = self._make_state("TRIAL")
        monkeypatch.setattr(app_module, "_state", state)
        client = TestClient(app_module.app, raise_server_exceptions=False)
        resp = client.post("/scheduled-tick")
        assert resp.status_code == 409
        assert resp.json()["detail"]["error"] == "trial_active"

    def test_consolidate_409_when_trial_active(self, monkeypatch):
        """POST /consolidate returns 409 trial_active when TRIAL is active."""
        from fastapi.testclient import TestClient

        import paramem.server.app as app_module

        state = self._make_state("TRIAL")
        monkeypatch.setattr(app_module, "_state", state)
        client = TestClient(app_module.app, raise_server_exceptions=False)
        resp = client.post("/consolidate")
        assert resp.status_code == 409
        assert resp.json()["detail"]["error"] == "trial_active"

    def test_scheduled_tick_proceeds_when_live(self, monkeypatch):
        """POST /scheduled-tick does NOT return 409 when state is LIVE."""
        from fastapi.testclient import TestClient

        import paramem.server.app as app_module

        state = self._make_state("LIVE")
        # Patch _maybe_trigger_scheduled_consolidation to avoid actual work.
        monkeypatch.setattr(app_module, "_state", state)
        monkeypatch.setattr(
            app_module,
            "_maybe_trigger_scheduled_consolidation",
            lambda: "deferred",
        )
        client = TestClient(app_module.app, raise_server_exceptions=False)
        resp = client.post("/scheduled-tick")
        # Should NOT be 409 trial_active.
        detail = resp.json().get("detail", {})
        assert resp.status_code != 409 or detail.get("error") != "trial_active"


class TestReplayLoopGuard:
    """Fix 4: the _replay_loop constructed during lifespan gets state_provider
    so that run_cycle raises TrialActiveError if a trial becomes active between
    the lifespan check and the cycle call.
    """

    def test_consolidation_loop_stores_state_provider(self):
        """ConsolidationLoop stores state_provider and uses it in run_cycle.

        Verifies the end-to-end wiring: a loop built with state_provider set to
        a callable returns TRIAL state → run_cycle raises TrialActiveError.
        This mirrors the _replay_loop path in app.py:1321 where Fix 4 adds
        state_provider=lambda: _state.
        """
        from unittest.mock import patch

        from paramem.training.consolidation import ConsolidationLoop, TrialActiveError

        trial_state = {"migration": {"state": "TRIAL"}}
        loop = ConsolidationLoop.__new__(ConsolidationLoop)
        loop.state_provider = lambda: trial_state
        loop.cycle_count = 0

        with patch.object(
            loop.__class__,
            "_run_extract_graph",
            side_effect=AssertionError("should not reach extraction"),
        ):
            with pytest.raises(TrialActiveError):
                loop.run_cycle("transcript", "session-x")

        # Verify state_provider is stored and callable.
        assert loop.state_provider is not None, "state_provider must not be None"
        assert callable(loop.state_provider), "state_provider must be callable"
        assert loop.state_provider() is trial_state, "state_provider must return the state dict"

    def test_replay_loop_run_cycle_raises_trial_active_error(self):
        """A ConsolidationLoop with state_provider raises TrialActiveError when TRIAL.

        Verifies end-to-end that a loop created with state_provider=lambda: state
        correctly blocks run_cycle once the state transitions to TRIAL.
        This is the regression guard for Fix 4.
        """
        from unittest.mock import patch

        from paramem.training.consolidation import ConsolidationLoop, TrialActiveError

        shared_state = {"migration": {"state": "TRIAL"}}

        loop = ConsolidationLoop.__new__(ConsolidationLoop)
        loop.state_provider = lambda: shared_state
        loop.cycle_count = 0

        with patch.object(
            loop.__class__,
            "_run_extract_graph",
            side_effect=AssertionError("should not reach extraction — guard must fire first"),
        ):
            with pytest.raises(TrialActiveError):
                loop.run_cycle("some transcript", "session-1")
