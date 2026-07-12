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

    @pytest.mark.parametrize(
        "path",
        ["/scheduled-tick", "/consolidate", "/consolidate/interim", "/reconsolidate"],
    )
    def test_consolidation_routes_409_when_trial_active(self, monkeypatch, path):
        """Every consolidation door returns 409 trial_active — via require_no_trial.

        One dependency, four routes: the refusal cannot drift between them.
        """
        from fastapi.testclient import TestClient

        import paramem.server.app as app_module

        state = self._make_state("TRIAL")
        monkeypatch.setattr(app_module, "_state", state)
        client = TestClient(app_module.app, raise_server_exceptions=False)
        resp = client.post(path)
        assert resp.status_code == 409
        assert resp.json()["detail"]["error"] == "trial_active"

    def test_ingest_sessions_keeps_its_own_trial_guard_behind_body_validation(
        self, monkeypatch, tmp_path
    ):
        """/ingest-sessions is deliberately NOT on ``require_no_trial``.

        A route-level dependency is resolved BEFORE body validation, so a
        malformed body would come back 409 instead of 422, and the ingest CLI's
        400/404 payloads would be pre-empted under TRIAL.  Its guard stays
        in-handler, after those checks.
        """
        from fastapi.testclient import TestClient

        import paramem.server.app as app_module

        state = self._make_state("TRIAL")
        state["speaker_store"] = None
        state["session_buffer"] = None
        monkeypatch.setattr(app_module, "_state", state)
        client = TestClient(app_module.app, raise_server_exceptions=False)

        # Malformed body under TRIAL → 422 (body validation still runs first).
        malformed = client.post("/ingest-sessions", json={"speaker_id": "speaker1"})
        assert malformed.status_code == 422

        # Empty speaker_id under TRIAL → the CLI's 400 payload, not a 409.
        empty_speaker = client.post(
            "/ingest-sessions",
            json={
                "speaker_id": "",
                "sessions": [],
                "document_filename": "d.txt",
                "document_b64": "",
            },
        )
        assert empty_speaker.status_code == 400
        assert empty_speaker.json()["rejected_no_speaker_id"] is True

        # Unknown speaker under TRIAL → the CLI's 404 payload, not a 409.
        unknown_speaker = client.post(
            "/ingest-sessions",
            json={
                "speaker_id": "speaker9",
                "sessions": [],
                "document_filename": "d.txt",
                "document_b64": "",
            },
        )
        assert unknown_speaker.status_code == 404
        assert unknown_speaker.json()["rejected_unknown_speaker"] is True

    def test_scheduled_tick_proceeds_when_live(self, monkeypatch):
        """POST /scheduled-tick does NOT return 409 when state is LIVE."""
        from fastapi.testclient import TestClient

        import paramem.server.app as app_module

        state = self._make_state("LIVE")

        def _fake_dispatch(action, *, apply_schedule_gate):
            # /scheduled-tick must apply the suspend/power-off catch-up gate.
            assert apply_schedule_gate is True
            assert action is app_module.ConsolidationAction.AUTO
            return "deferred", action

        # Patch the arbitrator to avoid actual work.
        monkeypatch.setattr(app_module, "_state", state)
        monkeypatch.setattr(app_module, "_dispatch_consolidation", _fake_dispatch)
        client = TestClient(app_module.app, raise_server_exceptions=False)
        resp = client.post("/scheduled-tick")
        # Should NOT be 409 trial_active.
        detail = resp.json().get("detail", {})
        assert resp.status_code != 409 or detail.get("error") != "trial_active"
