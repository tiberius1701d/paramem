"""Tests for POST /migration/accept (Slice 3b.3).

All tests run without GPU — the trial consolidation task is pre-seeded to a
terminal gate status.  Tests validate the 5-step accept ordering, precondition
gates, drift state refresh, rotation slot creation, and banner setting.

Fixture conventions mirror test_migration_confirm.py (_make_state, _sha256).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import paramem.server.app as app_module
from paramem.server.migration import TrialStash, initial_migration_state
from paramem.server.trial_state import (
    TRIAL_MARKER_SCHEMA_VERSION,
    TrialMarker,
    read_trial_marker,
    write_trial_marker,
)

_LIVE_YAML = b"model: mistral\ndebug: false\n"
_CAND_YAML = b"model: mistral\ndebug: true\n"


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _make_state(tmp_path: Path, gates_status: str = "pass") -> dict:
    """Build a TRIAL _state with a pre-seeded gate status.

    Parameters
    ----------
    tmp_path:
        Pytest tmp_path fixture for isolation.
    gates_status:
        Gate status string to pre-seed (e.g. ``"pass"``, ``"no_new_sessions"``,
        ``"fail"``).
    """
    live_yaml = tmp_path / "server.yaml"
    live_yaml.write_bytes(_LIVE_YAML)

    config = MagicMock()
    config.paths.data = tmp_path / "data" / "ha"
    config.paths.data.mkdir(parents=True, exist_ok=True)
    config.adapter_dir = tmp_path / "data" / "ha" / "adapters"
    config.adapter_dir.mkdir(parents=True, exist_ok=True)
    config.key_metadata_path = tmp_path / "data" / "ha" / "key_metadata.json"

    state_dir = config.paths.data / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    # Create trial_adapter and trial_graph directories to test rotation.
    trial_adapter_dir = state_dir / "trial_adapter"
    trial_adapter_dir.mkdir(exist_ok=True)
    (trial_adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")

    trial_graph_dir = state_dir / "trial_graph"
    trial_graph_dir.mkdir(exist_ok=True)
    (trial_graph_dir / "cumulative_graph.json").write_text("{}", encoding="utf-8")

    # Create a config backup slot with a real config artifact.
    backups_root = config.paths.data / "backups"
    config_slot = backups_root / "config" / "20260422-010000"
    config_slot.mkdir(parents=True, exist_ok=True)
    config_artifact = config_slot / "config-20260422-010000.bin"
    config_artifact.write_bytes(_LIVE_YAML)

    # Write a trial marker with the config artifact filename.
    marker = TrialMarker(
        schema_version=TRIAL_MARKER_SCHEMA_VERSION,
        started_at="2026-04-22T01:00:00+00:00",
        pre_trial_config_sha256=_sha256(_LIVE_YAML),
        candidate_config_sha256=_sha256(_CAND_YAML),
        backup_paths={
            "config": str(config_slot.resolve()),
            "graph": str(backups_root / "graph" / "20260422-010000"),
            "registry": str(backups_root / "registry" / "20260422-010000"),
        },
        trial_adapter_dir=str(trial_adapter_dir.resolve()),
        trial_graph_dir=str(trial_graph_dir.resolve()),
        config_artifact_filename="config-20260422-010000.bin",
    )
    write_trial_marker(state_dir, marker)

    trial_stash = TrialStash(
        started_at="2026-04-22T01:00:00+00:00",
        pre_trial_config_sha256=_sha256(_LIVE_YAML),
        candidate_config_sha256=_sha256(_CAND_YAML),
        backup_paths={
            "config": str(config_slot.resolve()),
            "graph": str(backups_root / "graph" / "20260422-010000"),
            "registry": str(backups_root / "registry" / "20260422-010000"),
        },
        trial_adapter_dir=str(trial_adapter_dir.resolve()),
        trial_graph_dir=str(trial_graph_dir.resolve()),
        gates={
            "status": gates_status,
            "completed_at": "2026-04-22T02:00:00+00:00",
        },
    )

    migration = initial_migration_state()
    migration["state"] = "TRIAL"
    migration["trial"] = trial_stash

    return {
        "model": None,
        "tokenizer": None,
        "config": config,
        "config_path": str(live_yaml),
        "consolidating": False,
        "migration": migration,
        "migration_lock": asyncio.Lock(),
        "server_started_at": "2026-04-22T00:00:00+00:00",
        "mode": "normal",
        "background_trainer": None,
        "consolidation_loop": None,
        "config_drift": {
            "detected": False,
            "loaded_hash": _sha256(_LIVE_YAML),
            "disk_hash": _sha256(_LIVE_YAML),
            "last_checked_at": "2026-04-22T00:00:00+00:00",
        },
    }


@pytest.fixture()
def state(tmp_path, monkeypatch):
    """TRIAL state monkeypatched into app_module (gates=pass)."""
    fresh = _make_state(tmp_path)
    monkeypatch.setattr(app_module, "_state", fresh)
    return fresh


@pytest.fixture()
def client(state):
    """TestClient with TRIAL state set up."""
    return TestClient(app_module.app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Happy path — pass
# ---------------------------------------------------------------------------


class TestAcceptHappyPath:
    def test_accept_pass_returns_200(self, client, state):
        """TRIAL + gates.status=pass → 200 AcceptResponse."""
        resp = client.post("/migration/accept")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["state"] == "LIVE"

    def test_accept_returns_restart_required(self, client, state):
        """AcceptResponse.restart_required is always True."""
        resp = client.post("/migration/accept")
        assert resp.json()["restart_required"] is True

    def test_accept_returns_restart_hint(self, client, state):
        """AcceptResponse.restart_hint is the systemctl command."""
        resp = client.post("/migration/accept")
        assert "systemctl" in resp.json()["restart_hint"]

    def test_accept_returns_pre_migration_backup_retained(self, client, state):
        """AcceptResponse.pre_migration_backup_retained is always True."""
        resp = client.post("/migration/accept")
        assert resp.json()["pre_migration_backup_retained"] is True

    def test_accept_clears_trial_marker(self, client, state, tmp_path):
        """Accept clears state/trial.json (step 3 before rotation — IMPROVEMENT 7)."""
        client.post("/migration/accept")
        state_dir = state["config"].paths.data / "state"
        assert read_trial_marker(state_dir) is None

    def test_accept_creates_rotation_slot(self, client, state, tmp_path):
        """Accept creates a slot directory under trial_adapters/."""
        client.post("/migration/accept")
        backups_root = state["config"].paths.data / "backups"
        trial_adapters = backups_root / "trial_adapters"
        slots = [d for d in trial_adapters.iterdir() if d.is_dir() and not d.name.startswith(".")]
        assert len(slots) == 1

    def test_accept_slot_has_meta_json(self, client, state, tmp_path):
        """Rotation slot contains meta.json with source='accept'."""
        client.post("/migration/accept")
        backups_root = state["config"].paths.data / "backups"
        trial_adapters = backups_root / "trial_adapters"
        slots = [d for d in trial_adapters.iterdir() if d.is_dir() and not d.name.startswith(".")]
        assert slots
        meta = json.loads((slots[0] / "meta.json").read_text(encoding="utf-8"))
        assert meta["source"] == "accept"
        assert meta["schema_version"] == 1
        assert meta["gates_status"] == "pass"

    def test_accept_moves_trial_adapter(self, client, state, tmp_path):
        """Trial adapter is moved into the rotation slot."""
        trial_adapter_dir = Path(state["migration"]["trial"]["trial_adapter_dir"])
        assert trial_adapter_dir.exists()
        resp = client.post("/migration/accept")
        assert resp.status_code == 200
        # Trial adapter dir should be gone after move.
        assert not trial_adapter_dir.exists()

    def test_accept_moves_trial_graph(self, client, state, tmp_path):
        """Trial graph is moved into the rotation slot."""
        trial_graph_dir = Path(state["migration"]["trial"]["trial_graph_dir"])
        assert trial_graph_dir.exists()
        resp = client.post("/migration/accept")
        assert resp.status_code == 200
        assert not trial_graph_dir.exists()

    def test_accept_resets_migration_state_to_live(self, client, state):
        """After accept, _state["migration"]["state"] == "LIVE"."""
        client.post("/migration/accept")
        assert state["migration"]["state"] == "LIVE"

    def test_accept_sets_restart_banner(self, client, state):
        """After accept, recovery_required contains the restart banner."""
        client.post("/migration/accept")
        banners = state["migration"]["recovery_required"]
        assert any("RESTART REQUIRED" in b for b in banners)

    def test_accept_refreshes_drift_state_all_fields(self, client, state, tmp_path):
        """Accept refreshes the full ConfigDriftState dict (REQUIRED FIX 3).

        All four fields (detected, loaded_hash, disk_hash, last_checked_at) must
        be present and detected=False.
        """
        resp = client.post("/migration/accept")
        assert resp.status_code == 200
        drift = state["config_drift"]
        assert isinstance(drift, dict)
        assert drift["detected"] is False
        assert "loaded_hash" in drift
        assert "disk_hash" in drift
        assert "last_checked_at" in drift
        # Both hashes should be equal (no drift after accept).
        assert drift["loaded_hash"] == drift["disk_hash"]

    def test_accept_preserves_pre_migration_backup(self, client, state, tmp_path):
        """The pre-migration config backup slot remains after accept."""
        backups_root = state["config"].paths.data / "backups"
        config_slot = backups_root / "config" / "20260422-010000"
        assert config_slot.exists()
        client.post("/migration/accept")
        assert config_slot.exists()


# ---------------------------------------------------------------------------
# Happy path — no_new_sessions
# ---------------------------------------------------------------------------


class TestAcceptNoNewSessions:
    def test_accept_no_new_sessions_returns_200(self, tmp_path, monkeypatch):
        """TRIAL + gates.status=no_new_sessions → 200 (accept-eligible)."""
        fresh = _make_state(tmp_path, gates_status="no_new_sessions")
        monkeypatch.setattr(app_module, "_state", fresh)
        client = TestClient(app_module.app, raise_server_exceptions=False)
        resp = client.post("/migration/accept")
        assert resp.status_code == 200
        assert resp.json()["state"] == "LIVE"


# ---------------------------------------------------------------------------
# Forward-compat test: accept-eligible statuses are set membership
# ---------------------------------------------------------------------------


class TestAcceptEligibleStatusesSetMembership:
    def test_accept_eligible_statuses_are_set_membership(self, tmp_path, monkeypatch):
        """pass_with_warnings (Slice 4 hypothetical) currently 409 gates_failed.

        This test documents the current accept-eligible set:
        {"pass", "no_new_sessions"}. Slice 4 will widen it to include
        "pass_with_warnings". This test will then be updated to reflect that.

        Forward-compat guardrail 1: the accept gate uses set membership, not
        exact equality, so Slice 4 only needs to add the new status to
        _ACCEPT_ELIGIBLE_STATUSES (one place, in app.py).
        """
        fresh = _make_state(tmp_path, gates_status="pass_with_warnings")
        monkeypatch.setattr(app_module, "_state", fresh)
        client = TestClient(app_module.app, raise_server_exceptions=False)
        resp = client.post("/migration/accept")
        # Currently 409 because "pass_with_warnings" is not in the eligible set.
        assert resp.status_code == 409
        assert resp.json()["detail"]["error"] == "gates_failed"


# ---------------------------------------------------------------------------
# 4xx precondition gates
# ---------------------------------------------------------------------------


class TestAccept4xx:
    def test_accept_404_when_live(self, tmp_path, monkeypatch):
        """State=LIVE → 404 not_found."""
        fresh = _make_state(tmp_path)
        fresh["migration"]["state"] = "LIVE"
        fresh["migration"]["trial"] = None
        monkeypatch.setattr(app_module, "_state", fresh)
        client = TestClient(app_module.app, raise_server_exceptions=False)
        resp = client.post("/migration/accept")
        assert resp.status_code == 404
        assert resp.json()["detail"]["error"] == "not_found"

    def test_accept_409_when_staging(self, tmp_path, monkeypatch):
        """State=STAGING → 409 not_trial."""
        fresh = _make_state(tmp_path)
        fresh["migration"]["state"] = "STAGING"
        monkeypatch.setattr(app_module, "_state", fresh)
        client = TestClient(app_module.app, raise_server_exceptions=False)
        resp = client.post("/migration/accept")
        assert resp.status_code == 409
        assert resp.json()["detail"]["error"] == "not_trial"

    def test_accept_409_gates_not_finished_pending(self, tmp_path, monkeypatch):
        """gates.status=pending → 409 gates_not_finished."""
        fresh = _make_state(tmp_path)
        fresh["migration"]["trial"]["gates"] = {"status": "pending"}
        monkeypatch.setattr(app_module, "_state", fresh)
        client = TestClient(app_module.app, raise_server_exceptions=False)
        resp = client.post("/migration/accept")
        assert resp.status_code == 409
        assert resp.json()["detail"]["error"] == "gates_not_finished"

    def test_accept_409_gates_not_finished_no_completed_at(self, tmp_path, monkeypatch):
        """gates.status=pass but no completed_at → 409 gates_not_finished."""
        fresh = _make_state(tmp_path)
        fresh["migration"]["trial"]["gates"] = {"status": "pass"}  # no completed_at
        monkeypatch.setattr(app_module, "_state", fresh)
        client = TestClient(app_module.app, raise_server_exceptions=False)
        resp = client.post("/migration/accept")
        assert resp.status_code == 409
        assert resp.json()["detail"]["error"] == "gates_not_finished"

    def test_accept_409_gates_failed(self, tmp_path, monkeypatch):
        """gates.status=fail → 409 gates_failed."""
        fresh = _make_state(tmp_path, gates_status="fail")
        monkeypatch.setattr(app_module, "_state", fresh)
        client = TestClient(app_module.app, raise_server_exceptions=False)
        resp = client.post("/migration/accept")
        assert resp.status_code == 409
        assert resp.json()["detail"]["error"] == "gates_failed"

    def test_accept_409_trial_exception(self, tmp_path, monkeypatch):
        """gates.status=trial_exception → 409 gates_failed."""
        fresh = _make_state(tmp_path, gates_status="trial_exception")
        monkeypatch.setattr(app_module, "_state", fresh)
        client = TestClient(app_module.app, raise_server_exceptions=False)
        resp = client.post("/migration/accept")
        assert resp.status_code == 409
        assert resp.json()["detail"]["error"] == "gates_failed"

    def test_accept_409_migration_in_progress(self, tmp_path, monkeypatch):
        """Lock already held → 409 migration_in_progress."""
        import asyncio

        fresh = _make_state(tmp_path)
        lock = asyncio.Lock()
        # Acquire the lock in a dedicated event loop to simulate a held lock.
        loop = asyncio.new_event_loop()
        loop.run_until_complete(lock.acquire())
        fresh["migration_lock"] = lock
        monkeypatch.setattr(app_module, "_state", fresh)
        client = TestClient(app_module.app, raise_server_exceptions=False)
        try:
            resp = client.post("/migration/accept")
            assert resp.status_code == 409
            assert resp.json()["detail"]["error"] == "migration_in_progress"
        finally:
            lock.release()
            loop.close()


# ---------------------------------------------------------------------------
# Step failure — rotation non-fatal
# ---------------------------------------------------------------------------


class TestAcceptRotationFailure:
    def test_accept_rotation_failure_returns_200_degraded(self, tmp_path, monkeypatch):
        """Step 4 adapter/graph move failure → 200 with ARCHIVE INCOMPLETE banner (non-fatal).

        The slot is created successfully (step 2).  The adapter move (step 4)
        fails.  Config + marker are already coherent (marker cleared at step 3
        BEFORE the move — IMPROVEMENT 7).  State returns LIVE; recovery_required
        contains the ARCHIVE INCOMPLETE banner.
        """
        fresh = _make_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", fresh)

        original_move = __import__("shutil").move

        def _fail_move(src, dst):
            # Fail only the adapter/graph move into the slot.
            if "adapter" in str(dst) or "graph" in str(dst):
                raise OSError("simulated move failure")
            return original_move(src, dst)

        with patch("paramem.server.app.shutil.move", _fail_move):
            client = TestClient(app_module.app, raise_server_exceptions=False)
            resp = client.post("/migration/accept")

        # Still 200 — primary action succeeded (slot exists, marker cleared).
        assert resp.status_code == 200
        assert fresh["migration"]["state"] == "LIVE"
        banners = fresh["migration"]["recovery_required"]
        # "RESTART REQUIRED" is always present after accept, so OR-ing with it
        # would make the assertion trivially pass even if ARCHIVE INCOMPLETE were
        # absent.  Assert the degraded banner specifically.
        assert any("ARCHIVE INCOMPLETE" in b for b in banners)

    def test_accept_rotation_failure_marker_already_cleared(self, tmp_path, monkeypatch):
        """After step-4 move failure, trial marker is still cleared (IMPROVEMENT 7).

        Marker-clear happens at step 3, BEFORE adapter/graph move at step 4.
        """
        fresh = _make_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", fresh)

        original_move = __import__("shutil").move

        def _fail_move(src, dst):
            if "adapter" in str(dst) or "graph" in str(dst):
                raise OSError("simulated move failure")
            return original_move(src, dst)

        with patch("paramem.server.app.shutil.move", _fail_move):
            client = TestClient(app_module.app, raise_server_exceptions=False)
            client.post("/migration/accept")

        state_dir = fresh["config"].paths.data / "state"
        assert read_trial_marker(state_dir) is None


# ---------------------------------------------------------------------------
# Drift-refresh OSError is best-effort (does not block accept)
# ---------------------------------------------------------------------------


class TestAcceptDriftRefreshOSError:
    def test_accept_succeeds_when_live_config_path_absent(self, tmp_path, monkeypatch):
        """Accept succeeds when live_config_path doesn't exist at drift-refresh time.

        Step 5 (drift refresh) is best-effort: the condition guard at
        ``app.py:3202`` uses ``live_config_path.exists()`` before calling
        ``compute_config_hash``.  When the file is absent, drift refresh is
        silently skipped; the state transition still completes.
        """
        fresh = _make_state(tmp_path)
        # Remove the live config to simulate a missing file.
        live_yaml = Path(fresh["config_path"])
        live_yaml.unlink()
        monkeypatch.setattr(app_module, "_state", fresh)

        client = TestClient(app_module.app, raise_server_exceptions=False)
        resp = client.post("/migration/accept")
        # Drift refresh is skipped, but accept must still succeed.
        assert resp.status_code == 200, resp.text
        assert fresh["migration"]["state"] == "LIVE"

    def test_accept_succeeds_when_compute_config_hash_raises_oserror(self, tmp_path, monkeypatch):
        """Accept succeeds when compute_config_hash raises OSError (best-effort refresh).

        The inner ``try/except OSError`` at ``app.py:3203`` must swallow the
        error and log a warning, not propagate it.  Absence of drift refresh
        does not block the state transition.
        """
        fresh = _make_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", fresh)

        with patch(
            "paramem.server.drift.compute_config_hash",
            side_effect=OSError("simulated hash failure"),
        ):
            client = TestClient(app_module.app, raise_server_exceptions=False)
            resp = client.post("/migration/accept")

        assert resp.status_code == 200, resp.text
        assert fresh["migration"]["state"] == "LIVE"
        # config_drift must remain a dict (not replaced with a ConfigDriftState,
        # since the OSError prevented the refresh).
        assert isinstance(fresh.get("config_drift"), dict)


# ---------------------------------------------------------------------------
# Accept does NOT call mark_consolidated
# ---------------------------------------------------------------------------


class TestAcceptDoesNotMarkConsolidated:
    def test_accept_does_not_call_mark_consolidated(self, tmp_path, monkeypatch):
        """accept does NOT call session_buffer.mark_consolidated.

        The trial consolidation path uses a no-op mark_consolidated_callback
        (line ~2838 in app.py) so that pending sessions are not marked
        consolidated during the trial run.  The accept handler itself also
        must not call mark_consolidated — doing so would permanently discard
        sessions that should feed the next consolidation cycle.
        """
        from unittest.mock import MagicMock

        fresh = _make_state(tmp_path)
        session_buffer_mock = MagicMock()
        fresh["session_buffer"] = session_buffer_mock
        monkeypatch.setattr(app_module, "_state", fresh)

        client = TestClient(app_module.app, raise_server_exceptions=False)
        resp = client.post("/migration/accept")
        assert resp.status_code == 200, resp.text
        session_buffer_mock.mark_consolidated.assert_not_called()


# ---------------------------------------------------------------------------
# Concurrent accept
# ---------------------------------------------------------------------------


class TestAcceptConcurrent:
    def test_accept_concurrent_409(self, tmp_path, monkeypatch):
        """Second accept while lock is held → 409 migration_in_progress."""
        import asyncio

        fresh = _make_state(tmp_path)
        lock = asyncio.Lock()
        loop = asyncio.new_event_loop()
        loop.run_until_complete(lock.acquire())
        fresh["migration_lock"] = lock
        monkeypatch.setattr(app_module, "_state", fresh)
        client = TestClient(app_module.app, raise_server_exceptions=False)
        try:
            resp = client.post("/migration/accept")
            assert resp.status_code == 409
        finally:
            lock.release()
            loop.close()
