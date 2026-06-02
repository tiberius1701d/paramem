"""Tests for POST /migration/accept.

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


def _default_apply_stub():
    """Default _apply_config_live stub: success (applied_live=True, no restart needed)."""

    def _impl():
        return {
            "applied_live": True,
            "restart_required_reason": None,
            "auto_restart_scheduled": False,
            "skipped": None,
            "cloud_only_reason": None,
        }

    return _impl


@pytest.fixture()
def state(tmp_path, monkeypatch):
    """TRIAL state monkeypatched into app_module (gates=pass).

    Also patches _apply_config_live to avoid a real GPU load in unit tests.
    Tests that need a specific apply result override via monkeypatch.setattr
    AFTER this fixture runs.
    """
    fresh = _make_state(tmp_path)
    monkeypatch.setattr(app_module, "_state", fresh)
    monkeypatch.setattr(app_module, "_apply_config_live", _default_apply_stub())
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

    def test_accept_returns_restart_required_false_on_live_apply(self, client, state):
        """AcceptResponse.restart_required is False when the live apply succeeds.

        With the default success stub (applied_live=True, no carve), the
        response must reflect the new live-apply contract: restart_required=False.
        """
        resp = client.post("/migration/accept")
        assert resp.json()["restart_required"] is False

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

    def test_accept_deletes_trial_graph(self, client, state, tmp_path):
        """Trial graph is deleted (not archived) after accept (transient by design).

        The trial graph is persisted only for the before/after comparison report
        during the trial window.  Once the operator accepts, it is deleted rather
        than moved into the rotation slot, consistent with the ``persist_graph=False``
        invariant for live production cycles.
        """
        trial_graph_dir = Path(state["migration"]["trial"]["trial_graph_dir"])
        assert trial_graph_dir.exists()
        resp = client.post("/migration/accept")
        assert resp.status_code == 200
        # Trial graph dir must be deleted, not moved.
        assert not trial_graph_dir.exists(), "trial graph must be deleted post-accept"

    def test_accept_archive_slot_has_no_graph_subdir(self, client, state, tmp_path):
        """Rotation slot must NOT contain a 'graph/' subdirectory after accept.

        The trial graph is no longer archived — it is deleted post-accept.
        """
        resp = client.post("/migration/accept")
        assert resp.status_code == 200
        backups_root = state["config"].paths.data / "backups"
        trial_adapters = backups_root / "trial_adapters"
        slots = [d for d in trial_adapters.iterdir() if d.is_dir() and not d.name.startswith(".")]
        assert slots, "At least one rotation slot must have been created"
        for slot in slots:
            assert not (slot / "graph").exists(), (
                f"Rotation slot {slot.name} must not contain a 'graph/' subdir "
                "(trial graph is deleted, not archived)"
            )

    def test_accept_resets_migration_state_to_live(self, client, state):
        """After accept, _state["migration"]["state"] == "LIVE"."""
        client.post("/migration/accept")
        assert state["migration"]["state"] == "LIVE"

    def test_accept_sets_applied_live_banner(self, client, state):
        """After accept with live-apply success, banner says 'applied live'.

        The default stub returns applied_live=True, so the banner
        must say 'applied live', NOT 'RESTART REQUIRED'.
        """
        client.post("/migration/accept")
        banners = state["migration"]["recovery_required"]
        assert any("applied live" in b.lower() for b in banners), (
            f"Expected 'applied live' banner, got: {banners}"
        )

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
        """ "pass_with_warnings" is not yet accept-eligible; currently returns 409.

        This test documents the current accept-eligible set:
        {"pass", "no_new_sessions"}.  When "pass_with_warnings" is added,
        this test will be updated to reflect that.

        Forward-compat guardrail: the accept gate uses set membership, not
        exact equality, so adding the new status only requires updating
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

        The drift refresh step uses ``live_config_path.exists()`` before calling
        ``compute_config_hash``.  When the file is absent, drift refresh is
        silently skipped; the state transition still completes.
        """
        fresh = _make_state(tmp_path)
        # Remove the live config to simulate a missing file.
        live_yaml = Path(fresh["config_path"])
        live_yaml.unlink()
        monkeypatch.setattr(app_module, "_state", fresh)
        monkeypatch.setattr(app_module, "_apply_config_live", _default_apply_stub())

        client = TestClient(app_module.app, raise_server_exceptions=False)
        resp = client.post("/migration/accept")
        # Drift refresh is skipped, but accept must still succeed.
        assert resp.status_code == 200, resp.text
        assert fresh["migration"]["state"] == "LIVE"

    def test_accept_succeeds_when_compute_config_hash_raises_oserror(self, tmp_path, monkeypatch):
        """Accept succeeds when compute_config_hash raises OSError (best-effort refresh).

        The inner ``try/except OSError`` at the drift-refresh step must swallow the
        error and log a warning, not propagate it.  Absence of drift refresh
        does not block the state transition.
        """
        fresh = _make_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", fresh)
        monkeypatch.setattr(app_module, "_apply_config_live", _default_apply_stub())

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


# ---------------------------------------------------------------------------
# Response contract + live-apply wiring
# ---------------------------------------------------------------------------


def _stub_apply(result: dict):
    """Return a side-effect that replaces _apply_config_live with a stub."""

    def _impl():
        return result

    return _impl


class TestAcceptResponseContract:
    """New response fields: applied_live, restart_required_reason, auto_restart_scheduled."""

    def test_response_has_applied_live_field(self, tmp_path, monkeypatch):
        """AcceptResponse includes applied_live field."""
        fresh = _make_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", fresh)
        monkeypatch.setattr(
            app_module,
            "_apply_config_live",
            _stub_apply(
                {
                    "applied_live": True,
                    "restart_required_reason": None,
                    "auto_restart_scheduled": False,
                    "skipped": None,
                    "cloud_only_reason": None,
                }
            ),
        )
        client = TestClient(app_module.app, raise_server_exceptions=False)
        resp = client.post("/migration/accept")
        assert resp.status_code == 200
        body = resp.json()
        assert "applied_live" in body

    def test_response_has_restart_required_reason_field(self, tmp_path, monkeypatch):
        """AcceptResponse includes restart_required_reason field."""
        fresh = _make_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", fresh)
        monkeypatch.setattr(
            app_module,
            "_apply_config_live",
            _stub_apply(
                {
                    "applied_live": True,
                    "restart_required_reason": None,
                    "auto_restart_scheduled": False,
                    "skipped": None,
                    "cloud_only_reason": None,
                }
            ),
        )
        client = TestClient(app_module.app, raise_server_exceptions=False)
        resp = client.post("/migration/accept")
        body = resp.json()
        assert "restart_required_reason" in body

    def test_response_has_auto_restart_scheduled_field(self, tmp_path, monkeypatch):
        """AcceptResponse includes auto_restart_scheduled field."""
        fresh = _make_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", fresh)
        monkeypatch.setattr(
            app_module,
            "_apply_config_live",
            _stub_apply(
                {
                    "applied_live": True,
                    "restart_required_reason": None,
                    "auto_restart_scheduled": False,
                    "skipped": None,
                    "cloud_only_reason": None,
                }
            ),
        )
        client = TestClient(app_module.app, raise_server_exceptions=False)
        resp = client.post("/migration/accept")
        body = resp.json()
        assert "auto_restart_scheduled" in body


class TestAcceptLiveApplySuccess:
    """Accept with _apply_config_live returning success."""

    def test_accept_live_apply_success_applied_live_true(self, tmp_path, monkeypatch):
        """Success stub → applied_live=True, restart_required=False."""
        fresh = _make_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", fresh)
        monkeypatch.setattr(
            app_module,
            "_apply_config_live",
            _stub_apply(
                {
                    "applied_live": True,
                    "restart_required_reason": None,
                    "auto_restart_scheduled": False,
                    "skipped": None,
                    "cloud_only_reason": None,
                }
            ),
        )
        client = TestClient(app_module.app, raise_server_exceptions=False)
        resp = client.post("/migration/accept")
        assert resp.status_code == 200
        body = resp.json()
        assert body["applied_live"] is True
        assert body["restart_required"] is False

    def test_accept_live_apply_success_no_restart_banner(self, tmp_path, monkeypatch):
        """Success stub → banner says 'applied live', not 'RESTART REQUIRED'."""
        fresh = _make_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", fresh)
        monkeypatch.setattr(
            app_module,
            "_apply_config_live",
            _stub_apply(
                {
                    "applied_live": True,
                    "restart_required_reason": None,
                    "auto_restart_scheduled": False,
                    "skipped": None,
                    "cloud_only_reason": None,
                }
            ),
        )
        client = TestClient(app_module.app, raise_server_exceptions=False)
        client.post("/migration/accept")
        banners = fresh["migration"]["recovery_required"]
        # 'RESTART REQUIRED' banner must NOT appear on a successful live apply.
        assert not any("RESTART REQUIRED" in b for b in banners), (
            f"Unexpected 'RESTART REQUIRED' banner on successful live apply: {banners}"
        )
        assert any("applied live" in b.lower() for b in banners), (
            f"Expected 'applied live' banner but got: {banners}"
        )

    def test_accept_live_apply_sets_maintenance_guard_before_dispatch(self, tmp_path, monkeypatch):
        """Maintenance guard (mode=cloud-only) is set BEFORE _apply_config_live runs (S-4)."""
        fresh = _make_state(tmp_path)
        fresh["mode"] = "local"
        monkeypatch.setattr(app_module, "_state", fresh)

        guard_set_before_apply = []

        def _capturing_apply():
            # Record the mode at the time the apply executes.
            guard_set_before_apply.append(fresh.get("mode"))
            return {
                "applied_live": True,
                "restart_required_reason": None,
                "auto_restart_scheduled": False,
                "skipped": None,
                "cloud_only_reason": None,
            }

        monkeypatch.setattr(app_module, "_apply_config_live", _capturing_apply)
        client = TestClient(app_module.app, raise_server_exceptions=False)
        client.post("/migration/accept")
        # The mode captured inside _apply_config_live must be "cloud-only" —
        # the handler set it synchronously before dispatching the executor.
        assert guard_set_before_apply, "apply was never called"
        assert guard_set_before_apply[0] == "cloud-only", (
            f"Expected mode='cloud-only' before apply, got {guard_set_before_apply[0]!r} (S-4)"
        )


class TestAcceptLiveApplyFailure:
    """Accept with _apply_config_live returning failure."""

    def test_accept_apply_failure_restart_required_true(self, tmp_path, monkeypatch):
        """Failure stub → applied_live=False, restart_required=True."""
        fresh = _make_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", fresh)
        monkeypatch.setattr(
            app_module,
            "_apply_config_live",
            _stub_apply(
                {
                    "applied_live": False,
                    "restart_required_reason": None,
                    "auto_restart_scheduled": False,
                    "skipped": None,
                    "cloud_only_reason": "apply_failed",
                }
            ),
        )
        client = TestClient(app_module.app, raise_server_exceptions=False)
        resp = client.post("/migration/accept")
        assert resp.status_code == 200
        body = resp.json()
        assert body["applied_live"] is False
        assert body["restart_required"] is True

    def test_accept_apply_failure_keeps_restart_banner(self, tmp_path, monkeypatch):
        """Failure stub → RESTART REQUIRED banner present."""
        fresh = _make_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", fresh)
        monkeypatch.setattr(
            app_module,
            "_apply_config_live",
            _stub_apply(
                {
                    "applied_live": False,
                    "restart_required_reason": None,
                    "auto_restart_scheduled": False,
                    "skipped": None,
                    "cloud_only_reason": "apply_failed",
                }
            ),
        )
        client = TestClient(app_module.app, raise_server_exceptions=False)
        client.post("/migration/accept")
        banners = fresh["migration"]["recovery_required"]
        assert any("RESTART REQUIRED" in b for b in banners), (
            f"Expected 'RESTART REQUIRED' banner on apply failure, got: {banners}"
        )


class TestAcceptRPortCarve:
    """R-PORT carve: stt_port_change / tts_port_change."""

    def test_r_port_carve_restart_eligible_true(self, tmp_path, monkeypatch):
        """R-PORT carve with pre-flight success → response carries restart_eligible=True."""
        fresh = _make_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", fresh)
        monkeypatch.setattr(
            app_module,
            "_apply_config_live",
            _stub_apply(
                {
                    "applied_live": False,
                    "restart_required_reason": "stt_port_change",
                    "auto_restart_scheduled": False,
                    "restart_eligible": True,
                    "skipped": None,
                    "cloud_only_reason": None,
                }
            ),
        )
        restart_calls = []
        monkeypatch.setattr(app_module, "_restart_service", lambda: restart_calls.append("called"))
        client = TestClient(app_module.app, raise_server_exceptions=False)
        resp = client.post("/migration/accept")
        assert resp.status_code == 200
        body = resp.json()
        assert body["restart_required"] is True
        # The server never auto-fires a restart; restart_eligible signals the CLI to prompt.
        assert body["auto_restart_scheduled"] is False
        assert body["restart_eligible"] is True
        assert body["restart_required_reason"] == "stt_port_change"

    def test_r_port_carve_server_does_not_fire_restart_service(self, tmp_path, monkeypatch):
        """R-PORT carve → server does NOT schedule _restart_service; restart_eligible=True."""
        fresh = _make_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", fresh)
        monkeypatch.setattr(
            app_module,
            "_apply_config_live",
            _stub_apply(
                {
                    "applied_live": False,
                    "restart_required_reason": "stt_port_change",
                    "auto_restart_scheduled": False,
                    "restart_eligible": True,
                    "skipped": None,
                    "cloud_only_reason": None,
                }
            ),
        )
        restart_calls = []
        monkeypatch.setattr(app_module, "_restart_service", lambda: restart_calls.append("called"))

        import asyncio as _asyncio

        real_get_running_loop = _asyncio.get_running_loop
        call_later_fired = []

        def _patched_get_running_loop():
            loop = real_get_running_loop()

            def _recording_call_later(delay, callback, *args):
                call_later_fired.append((delay, callback))

            loop.call_later = _recording_call_later
            return loop

        monkeypatch.setattr(_asyncio, "get_running_loop", _patched_get_running_loop)
        client = TestClient(app_module.app, raise_server_exceptions=False)
        resp = client.post("/migration/accept")
        assert resp.status_code == 200
        body = resp.json()
        assert body["restart_eligible"] is True
        # Server MUST NOT self-fire _restart_service — the CLI prompts the operator.
        assert not restart_calls, (
            f"_restart_service must NOT be called by the server for R-PORT: {restart_calls}"
        )
        assert not call_later_fired, (
            f"call_later must NOT be used to schedule a restart: {call_later_fired}"
        )

    def test_r_port_port_in_use_no_restart(self, tmp_path, monkeypatch):
        """Port-in-use pre-flight result → auto_restart_scheduled=False, no restart."""
        fresh = _make_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", fresh)
        monkeypatch.setattr(
            app_module,
            "_apply_config_live",
            _stub_apply(
                {
                    "applied_live": False,
                    "restart_required_reason": "stt_port_change",
                    "auto_restart_scheduled": False,
                    "skipped": None,
                    "cloud_only_reason": None,
                    "port_in_use_reason": "stt.port=10300 is not bindable: already in use",
                }
            ),
        )
        restart_calls = []
        monkeypatch.setattr(app_module, "_restart_service", lambda: restart_calls.append("called"))
        client = TestClient(app_module.app, raise_server_exceptions=False)
        resp = client.post("/migration/accept")
        assert resp.status_code == 200
        body = resp.json()
        assert body["auto_restart_scheduled"] is False
        # _restart_service must NOT have been called.
        assert not restart_calls, f"_restart_service was called unexpectedly: {restart_calls}"


class TestAcceptRPathsCarve:
    """R-PATHS carve: paths_change."""

    def test_r_paths_carve_manual_restart(self, tmp_path, monkeypatch):
        """R-PATHS carve → restart_required=True, auto_restart_scheduled=False."""
        fresh = _make_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", fresh)
        monkeypatch.setattr(
            app_module,
            "_apply_config_live",
            _stub_apply(
                {
                    "applied_live": False,
                    "restart_required_reason": "paths_change",
                    "auto_restart_scheduled": False,
                    "skipped": None,
                    "cloud_only_reason": None,
                }
            ),
        )
        restart_calls = []
        monkeypatch.setattr(app_module, "_restart_service", lambda: restart_calls.append("called"))
        client = TestClient(app_module.app, raise_server_exceptions=False)
        resp = client.post("/migration/accept")
        assert resp.status_code == 200
        body = resp.json()
        assert body["restart_required"] is True
        assert body["auto_restart_scheduled"] is False
        assert body["restart_required_reason"] == "paths_change"
        assert not restart_calls, (
            f"_restart_service must NOT be called for R-PATHS: {restart_calls}"
        )

    def test_r_paths_carve_data_not_migrated_banner(self, tmp_path, monkeypatch):
        """R-PATHS carve → banner warns about data not migrated."""
        fresh = _make_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", fresh)
        monkeypatch.setattr(
            app_module,
            "_apply_config_live",
            _stub_apply(
                {
                    "applied_live": False,
                    "restart_required_reason": "paths_change",
                    "auto_restart_scheduled": False,
                    "skipped": None,
                    "cloud_only_reason": None,
                }
            ),
        )
        client = TestClient(app_module.app, raise_server_exceptions=False)
        client.post("/migration/accept")
        banners = fresh["migration"]["recovery_required"]
        assert any("DATA IS NOT MIGRATED" in b for b in banners), (
            f"Expected DATA IS NOT MIGRATED warning in banners: {banners}"
        )


# ---------------------------------------------------------------------------
# Real _apply_config_live: R-PATHS short-circuit guard
# ---------------------------------------------------------------------------


def _make_apply_state(tmp_path: Path, yaml_a_bytes: bytes, yaml_b_bytes: bytes) -> dict:
    """Build a minimal _state dict for direct _apply_config_live calls.

    Writes yaml_a to a sidecar file for hashing and yaml_b as the on-disk
    config at ``config_path``.  ``config_a`` is loaded from yaml_a so the
    carve comparison uses real ``ServerConfig`` instances rather than mocks.

    Parameters
    ----------
    tmp_path:
        Pytest tmp_path fixture for isolation.
    yaml_a_bytes:
        Content of config A (currently loaded in memory).
    yaml_b_bytes:
        Content of config B (on disk at ``config_path`` — the pending migration).
    """
    from paramem.server.config import load_server_config
    from paramem.server.drift import compute_config_hash

    yaml_a_path = tmp_path / "config_a.yaml"
    yaml_a_path.write_bytes(yaml_a_bytes)
    yaml_b_path = tmp_path / "server.yaml"
    yaml_b_path.write_bytes(yaml_b_bytes)

    config_a = load_server_config(yaml_a_path)
    loaded_hash = compute_config_hash(yaml_a_path)

    return {
        "model": None,
        "tokenizer": None,
        "config": config_a,
        "config_path": str(yaml_b_path),
        "consolidating": False,
        "migration": {},
        "mode": "cloud-only",
        "background_trainer": None,
        "consolidation_loop": None,
        "cloud_only_reason": None,
        "session_buffer": None,
        "config_drift": {
            "detected": False,
            "loaded_hash": loaded_hash,
            "disk_hash": loaded_hash,
            "last_checked_at": "2026-04-22T00:00:00+00:00",
        },
    }


class TestApplyConfigLiveRPathsShortCircuit:
    """Real _apply_config_live: R-PATHS carve short-circuits before _live_reload_base_model."""

    def test_paths_data_change_short_circuits_before_reload(self, tmp_path, monkeypatch):
        """R-PATHS delta (paths.data changed) → _live_reload_base_model is NOT called.

        Exercises the real _apply_config_live with real ServerConfig objects.
        Only _live_reload_base_model is mocked to track invocations.
        """
        yaml_a = b"model: mistral\n"
        yaml_b = b"model: mistral\npaths:\n  data: /tmp/new_data_path\n"

        state = _make_apply_state(tmp_path, yaml_a, yaml_b)
        monkeypatch.setattr(app_module, "_state", state)

        reload_calls = []

        def _noop_reload(**kwargs):
            reload_calls.append(kwargs)

        monkeypatch.setattr(app_module, "_live_reload_base_model", _noop_reload)

        result = app_module._apply_config_live()

        assert not reload_calls, (
            f"_live_reload_base_model must NOT be called for R-PATHS carve; got: {reload_calls}"
        )
        assert result["applied_live"] is False
        assert result["restart_required_reason"] == "paths_change"
        assert result["auto_restart_scheduled"] is False
        assert result["restart_eligible"] is False

    def test_paths_sessions_change_short_circuits_before_reload(self, tmp_path, monkeypatch):
        """R-PATHS delta (paths.sessions changed) → _live_reload_base_model is NOT called."""
        yaml_a = b"model: mistral\n"
        yaml_b = b"model: mistral\npaths:\n  sessions: /tmp/new_sessions_path\n"

        state = _make_apply_state(tmp_path, yaml_a, yaml_b)
        monkeypatch.setattr(app_module, "_state", state)

        reload_calls = []

        def _noop_reload(**kwargs):
            reload_calls.append(kwargs)

        monkeypatch.setattr(app_module, "_live_reload_base_model", _noop_reload)

        result = app_module._apply_config_live()

        assert not reload_calls, (
            "_live_reload_base_model must NOT be called for R-PATHS (sessions) carve; "
            f"got: {reload_calls}"
        )
        assert result["applied_live"] is False
        assert result["restart_required_reason"] == "paths_change"


# ---------------------------------------------------------------------------
# Real _apply_config_live on the accept no-op-skip path (disk hash == loaded hash)
# ---------------------------------------------------------------------------


class TestApplyConfigLiveNoOpSkip:
    """Real _apply_config_live: no-op skip fires iff disk hash == loaded hash.

    Correction B1: ServerConfig has no source_path attribute.  The prior
    implementation derived mem_hash from the live file (always matching disk),
    so the skip fired on every accept call.  The fix uses
    ``_state["config_drift"]["loaded_hash"]`` (set at boot / last accept) as
    the in-memory reference.

    These tests use real hash computation (not mocked) to exercise the actual
    guard.  Only _live_reload_base_model is mocked.
    """

    def test_config_b_differs_from_a_skip_does_not_fire_reload_entered(self, tmp_path, monkeypatch):
        """config B ≠ config A → no-op skip does NOT fire; _live_reload_base_model is called."""
        yaml_a = b"model: mistral\ndebug: false\n"
        yaml_b = b"model: mistral\ndebug: true\n"

        state = _make_apply_state(tmp_path, yaml_a, yaml_b)
        monkeypatch.setattr(app_module, "_state", state)

        reload_calls = []

        def _noop_reload(**kwargs):
            reload_calls.append(kwargs)

        monkeypatch.setattr(app_module, "_live_reload_base_model", _noop_reload)

        result = app_module._apply_config_live()

        assert reload_calls, (
            "_live_reload_base_model must be called when config B ≠ config A "
            "(no-op skip must NOT fire)"
        )
        # skipped must not be "no_change".
        assert result.get("skipped") != "no_change", (
            f"No-op skip incorrectly fired when config B ≠ config A: {result}"
        )

    def test_config_a_equals_disk_skip_fires_no_reload(self, tmp_path, monkeypatch):
        """Rollback case: disk hash == loaded hash => no-op skip fires; no model reload."""
        yaml_a = b"model: mistral\ndebug: false\n"
        # yaml_b is the same content as yaml_a — simulates rollback restoring config A.
        yaml_b = yaml_a

        state = _make_apply_state(tmp_path, yaml_a, yaml_b)
        monkeypatch.setattr(app_module, "_state", state)

        reload_calls = []

        def _noop_reload(**kwargs):
            reload_calls.append(kwargs)

        monkeypatch.setattr(app_module, "_live_reload_base_model", _noop_reload)

        result = app_module._apply_config_live()

        assert not reload_calls, (
            "_live_reload_base_model must NOT be called when disk hash == loaded hash "
            f"(no-op skip must fire): reload_calls={reload_calls}"
        )
        assert result.get("skipped") == "no_change", (
            f"No-op skip did not fire when disk hash == loaded hash: {result}"
        )
        assert result["applied_live"] is True
