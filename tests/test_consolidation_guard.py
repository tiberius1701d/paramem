"""Tests for ConsolidationLoop.guard_trial_state and TrialActiveError.

No GPU — all tests use mocked ConsolidationLoop instances.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from paramem.training.consolidation import ConsolidationLoop, TrialActiveError


def _make_loop() -> ConsolidationLoop:
    """Build a ConsolidationLoop without loading any model or GPU resources."""
    loop = ConsolidationLoop.__new__(ConsolidationLoop)
    # Provide just enough attributes for guard_trial_state.
    loop.state_provider = None
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
