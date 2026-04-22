"""Tests for POST /migration/rollback (Slice 3b.3).

All tests run without GPU.  Covers the 8-step atomic ordering, precondition
gates, pre-mortem backup, A-config restore, marker clear (IMPROVEMENT 8),
and the 207 Multi-Status path when rotation fails.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
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

_LIVE_YAML = b"model: mistral\ndebug: true\n"  # This is config B (after confirm).
_A_YAML = b"model: mistral\ndebug: false\n"  # This is config A (original).


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _make_state(
    tmp_path: Path,
    gates_status: str = "fail",
    write_marker: bool = True,
    config_artifact_filename: str = "config-20260422-010000.bin",
) -> dict:
    """Build a TRIAL _state for rollback tests.

    Parameters
    ----------
    tmp_path:
        Pytest tmp_path fixture.
    gates_status:
        Gate status (can be any TRIAL status — rollback is always valid).
    write_marker:
        When ``True``, writes a TrialMarker with ``config_artifact_filename``.
    config_artifact_filename:
        Filename to use in the marker (set to "" to test missing-filename error).
    """
    # Config B is the live config during TRIAL.
    live_yaml = tmp_path / "server.yaml"
    live_yaml.write_bytes(_LIVE_YAML)

    config = MagicMock()
    config.paths.data = tmp_path / "data" / "ha"
    config.paths.data.mkdir(parents=True, exist_ok=True)
    config.adapter_dir = tmp_path / "data" / "ha" / "adapters"
    config.adapter_dir.mkdir(parents=True, exist_ok=True)

    state_dir = config.paths.data / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    # Trial adapter and graph dirs.
    trial_adapter_dir = state_dir / "trial_adapter"
    trial_adapter_dir.mkdir(exist_ok=True)
    (trial_adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")

    trial_graph_dir = state_dir / "trial_graph"
    trial_graph_dir.mkdir(exist_ok=True)

    # Create config A backup slot with the A-config artifact.
    backups_root = config.paths.data / "backups"
    config_slot = backups_root / "config" / "20260422-010000"
    config_slot.mkdir(parents=True, exist_ok=True)
    if config_artifact_filename:
        a_artifact = config_slot / config_artifact_filename
        a_artifact.write_bytes(_A_YAML)

    if write_marker:
        marker = TrialMarker(
            schema_version=TRIAL_MARKER_SCHEMA_VERSION,
            started_at="2026-04-22T01:00:00+00:00",
            pre_trial_config_sha256=_sha256(_A_YAML),
            candidate_config_sha256=_sha256(_LIVE_YAML),
            backup_paths={
                "config": str(config_slot.resolve()),
                "graph": str(backups_root / "graph" / "20260422-010000"),
                "registry": str(backups_root / "registry" / "20260422-010000"),
            },
            trial_adapter_dir=str(trial_adapter_dir.resolve()),
            trial_graph_dir=str(trial_graph_dir.resolve()),
            config_artifact_filename=config_artifact_filename,
        )
        write_trial_marker(state_dir, marker)

    trial_stash = TrialStash(
        started_at="2026-04-22T01:00:00+00:00",
        pre_trial_config_sha256=_sha256(_A_YAML),
        candidate_config_sha256=_sha256(_LIVE_YAML),
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
    """TRIAL state with gates=fail (always-valid for rollback)."""
    fresh = _make_state(tmp_path)
    monkeypatch.setattr(app_module, "_state", fresh)
    return fresh


@pytest.fixture()
def client(state):
    return TestClient(app_module.app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Happy path — rollback valid at any gate status
# ---------------------------------------------------------------------------


class TestRollbackHappyPath:
    def test_rollback_returns_200(self, client, state):
        """TRIAL + gates=fail → 200 RollbackResponse."""
        resp = client.post("/migration/rollback")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["state"] == "LIVE"
        assert body["restart_required"] is True

    def test_rollback_valid_when_gates_pending(self, tmp_path, monkeypatch):
        """Rollback valid when gates=pending (not yet finished)."""
        fresh = _make_state(tmp_path, gates_status="pending")
        fresh["migration"]["trial"]["gates"] = {"status": "pending"}
        monkeypatch.setattr(app_module, "_state", fresh)
        client = TestClient(app_module.app, raise_server_exceptions=False)
        resp = client.post("/migration/rollback")
        assert resp.status_code == 200

    def test_rollback_valid_when_gates_pass(self, tmp_path, monkeypatch):
        """Rollback valid even when gates=pass (operator may still choose rollback)."""
        fresh = _make_state(tmp_path, gates_status="pass")
        monkeypatch.setattr(app_module, "_state", fresh)
        client = TestClient(app_module.app, raise_server_exceptions=False)
        resp = client.post("/migration/rollback")
        assert resp.status_code == 200

    def test_rollback_restores_a_config(self, client, state, tmp_path):
        """After rollback, live config contains config A bytes."""
        live_yaml = Path(state["config_path"])
        assert live_yaml.read_bytes() == _LIVE_YAML  # was config B
        client.post("/migration/rollback")
        assert live_yaml.read_bytes() == _A_YAML  # restored to config A

    def test_rollback_writes_pre_mortem_backup(self, client, state, tmp_path):
        """Rollback writes a rollback_pre_mortem config backup (step 2)."""
        backups_root = state["config"].paths.data / "backups"
        client.post("/migration/rollback")
        config_dir = backups_root / "config"
        slots = [d for d in config_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
        # Should have at least the original + pre_mortem slot.
        assert len(slots) >= 2

    def test_rollback_pre_mortem_path_in_response(self, client, state):
        """RollbackResponse.rollback_pre_mortem_backup_path is an absolute path."""
        resp = client.post("/migration/rollback")
        body = resp.json()
        assert "rollback_pre_mortem_backup_path" in body
        path = body["rollback_pre_mortem_backup_path"]
        assert os.path.isabs(path)

    def test_rollback_archives_trial_adapter(self, client, state, tmp_path):
        """Trial adapter is moved into the trial_adapters archive slot (step 6)."""
        trial_adapter_dir = Path(state["migration"]["trial"]["trial_adapter_dir"])
        assert trial_adapter_dir.exists()
        client.post("/migration/rollback")
        assert not trial_adapter_dir.exists()

    def test_rollback_archive_slot_meta_source_rollback(self, client, state, tmp_path):
        """Rotation slot meta.json has source='rollback'."""
        client.post("/migration/rollback")
        backups_root = state["config"].paths.data / "backups"
        trial_adapters = backups_root / "trial_adapters"
        slots = [d for d in trial_adapters.iterdir() if d.is_dir() and not d.name.startswith(".")]
        assert slots
        meta = json.loads((slots[0] / "meta.json").read_text(encoding="utf-8"))
        assert meta["source"] == "rollback"

    def test_rollback_clears_trial_marker(self, client, state, tmp_path):
        """Rollback clears state/trial.json (IMPROVEMENT 8 — step 5 before rotation)."""
        client.post("/migration/rollback")
        state_dir = state["config"].paths.data / "state"
        assert read_trial_marker(state_dir) is None

    def test_rollback_resets_migration_state_to_live(self, client, state):
        """After rollback, migration.state == "LIVE"."""
        client.post("/migration/rollback")
        assert state["migration"]["state"] == "LIVE"

    def test_rollback_sets_restart_banner(self, client, state):
        """After rollback, recovery_required contains RESTART REQUIRED banner."""
        client.post("/migration/rollback")
        banners = state["migration"]["recovery_required"]
        assert any("RESTART REQUIRED" in b for b in banners)

    def test_rollback_does_not_refresh_drift(self, client, state):
        """Rollback does NOT refresh config_drift.

        In-memory config matches A after restore; drift stays coherent.
        Per spec: rollback handler must NOT call compute_config_hash /
        update config_drift (unlike accept which does — REQUIRED FIX 3).
        The drift loop stays coherent because rollback restores the original
        config that was hashed at startup.
        """
        original_loaded_hash = state["config_drift"]["loaded_hash"]
        client.post("/migration/rollback")
        # loaded_hash must be unchanged (drift not refreshed by rollback).
        assert state["config_drift"]["loaded_hash"] == original_loaded_hash


# ---------------------------------------------------------------------------
# 4xx precondition gates
# ---------------------------------------------------------------------------


class TestRollback4xx:
    def test_rollback_404_when_live(self, tmp_path, monkeypatch):
        """State=LIVE → 404 not_found."""
        fresh = _make_state(tmp_path)
        fresh["migration"]["state"] = "LIVE"
        fresh["migration"]["trial"] = None
        monkeypatch.setattr(app_module, "_state", fresh)
        client = TestClient(app_module.app, raise_server_exceptions=False)
        resp = client.post("/migration/rollback")
        assert resp.status_code == 404
        assert resp.json()["detail"]["error"] == "not_found"

    def test_rollback_409_when_staging(self, tmp_path, monkeypatch):
        """State=STAGING → 409 not_trial."""
        fresh = _make_state(tmp_path)
        fresh["migration"]["state"] = "STAGING"
        monkeypatch.setattr(app_module, "_state", fresh)
        client = TestClient(app_module.app, raise_server_exceptions=False)
        resp = client.post("/migration/rollback")
        assert resp.status_code == 409
        assert resp.json()["detail"]["error"] == "not_trial"

    def test_rollback_409_migration_in_progress(self, tmp_path, monkeypatch):
        """Lock already held → 409 migration_in_progress."""
        fresh = _make_state(tmp_path)
        lock = asyncio.Lock()
        loop = asyncio.new_event_loop()
        loop.run_until_complete(lock.acquire())
        fresh["migration_lock"] = lock
        monkeypatch.setattr(app_module, "_state", fresh)
        client = TestClient(app_module.app, raise_server_exceptions=False)
        try:
            resp = client.post("/migration/rollback")
            assert resp.status_code == 409
        finally:
            lock.release()
            loop.close()


# ---------------------------------------------------------------------------
# Step-2 failure: pre-mortem backup failed → 500, state=TRIAL
# ---------------------------------------------------------------------------


class TestRollbackStep2Failure:
    def test_rollback_step2_failure_returns_500(self, tmp_path, monkeypatch):
        """Step 2 (pre-mortem backup) failure → 500; state=TRIAL preserved."""
        fresh = _make_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", fresh)

        with patch("paramem.server.app.backup_write", side_effect=OSError("disk full")):
            client = TestClient(app_module.app, raise_server_exceptions=False)
            resp = client.post("/migration/rollback")

        assert resp.status_code == 500
        assert resp.json()["detail"]["error"] == "rollback_backup_failed"
        # State must remain TRIAL — no mutations before step 2.
        assert fresh["migration"]["state"] == "TRIAL"


# ---------------------------------------------------------------------------
# Step-3 failure: missing A config artifact → 500, pre-mortem deleted
# ---------------------------------------------------------------------------


class TestRollbackStep3Failure:
    def test_rollback_step3_missing_artifact_returns_500(self, tmp_path, monkeypatch):
        """Step 3: config_artifact not found → 500, pre-mortem backup deleted."""
        fresh = _make_state(
            tmp_path,
            config_artifact_filename="config-20260422-010000.bin",
        )
        # Remove the A artifact to trigger the missing-file error.
        backups_root = fresh["config"].paths.data / "backups"
        config_slot = backups_root / "config" / "20260422-010000"
        a_artifact = config_slot / "config-20260422-010000.bin"
        a_artifact.unlink()
        monkeypatch.setattr(app_module, "_state", fresh)
        client = TestClient(app_module.app, raise_server_exceptions=False)
        resp = client.post("/migration/rollback")
        assert resp.status_code == 500
        assert resp.json()["detail"]["error"] == "rollback_precondition_failed"
        # Pre-mortem backup should be cleaned up.
        config_dir = backups_root / "config"
        slots = [d for d in config_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
        # Only the original slot remains (pre_mortem was cleaned up).
        assert len(slots) == 1

    def test_rollback_step3_empty_filename_returns_500(self, tmp_path, monkeypatch):
        """config_artifact_filename="" → 500 rollback_precondition_failed."""
        fresh = _make_state(tmp_path, config_artifact_filename="")
        monkeypatch.setattr(app_module, "_state", fresh)
        client = TestClient(app_module.app, raise_server_exceptions=False)
        resp = client.post("/migration/rollback")
        assert resp.status_code == 500
        assert resp.json()["detail"]["error"] == "rollback_precondition_failed"


# ---------------------------------------------------------------------------
# Step-4 failure: config rename failed → 500, pre-mortem deleted
# ---------------------------------------------------------------------------


class TestRollbackStep4Failure:
    def test_rollback_step4_rename_failure_returns_500(self, tmp_path, monkeypatch):
        """Step 4 rename failure → 500; pre-mortem cleaned up; state=TRIAL."""
        fresh = _make_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", fresh)

        original_rename = os.rename

        def _fail_rename(src, dst):
            src_path = Path(src)
            dst_path = Path(dst)
            # Fail only the rename of the A config artifact to live path.
            if src_path.suffix == ".bin" and dst_path.name == "server.yaml":
                raise OSError("EXDEV: cross-device rename")
            return original_rename(src, dst)

        with patch("paramem.server.app.os.rename", _fail_rename):
            client = TestClient(app_module.app, raise_server_exceptions=False)
            resp = client.post("/migration/rollback")

        assert resp.status_code == 500
        assert resp.json()["detail"]["error"] == "config_restore_failed"
        # State still TRIAL — step 5 (marker clear) never ran.
        assert fresh["migration"]["state"] == "TRIAL"
        # Pre-mortem backup should have been cleaned up.
        backups_root = fresh["config"].paths.data / "backups"
        config_dir = backups_root / "config"
        slots = [d for d in config_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
        assert len(slots) == 1  # only original slot


# ---------------------------------------------------------------------------
# Step-6 failure: rotation failed → 207 Multi-Status (REQUIRED FIX 2)
# ---------------------------------------------------------------------------


class TestRollbackStep6Failure:
    def test_rollback_rotation_failure_returns_207(self, tmp_path, monkeypatch):
        """Step 6 rotation failure → 207 with archive_warning body."""
        fresh = _make_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", fresh)

        original_rename = os.rename

        def _fail_rename(src, dst):
            # Fail the trial_adapters slot rename.
            if "trial_adapters" in str(dst):
                raise OSError("simulated rotation failure")
            return original_rename(src, dst)

        with patch("paramem.server.app.os.rename", _fail_rename):
            client = TestClient(app_module.app, raise_server_exceptions=False)
            resp = client.post("/migration/rollback")

        assert resp.status_code == 207
        body = resp.json()
        assert body["state"] == "LIVE"
        assert "archive_warning" in body
        assert isinstance(body["archive_warning"], dict)
        assert "path" in body["archive_warning"]
        assert "message" in body["archive_warning"]

    def test_rollback_207_config_is_restored(self, tmp_path, monkeypatch):
        """When 207 is returned, the config A is still restored."""
        fresh = _make_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", fresh)

        original_rename = os.rename

        def _fail_rename(src, dst):
            if "trial_adapters" in str(dst):
                raise OSError("simulated rotation failure")
            return original_rename(src, dst)

        with patch("paramem.server.app.os.rename", _fail_rename):
            client = TestClient(app_module.app, raise_server_exceptions=False)
            client.post("/migration/rollback")

        live_yaml = Path(fresh["config_path"])
        assert live_yaml.read_bytes() == _A_YAML

    def test_rollback_207_marker_is_cleared(self, tmp_path, monkeypatch):
        """When 207 is returned, trial marker is still cleared (IMPROVEMENT 8)."""
        fresh = _make_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", fresh)

        original_rename = os.rename

        def _fail_rename(src, dst):
            if "trial_adapters" in str(dst):
                raise OSError("simulated rotation failure")
            return original_rename(src, dst)

        with patch("paramem.server.app.os.rename", _fail_rename):
            client = TestClient(app_module.app, raise_server_exceptions=False)
            client.post("/migration/rollback")

        state_dir = fresh["config"].paths.data / "state"
        assert read_trial_marker(state_dir) is None

    def test_rollback_207_state_is_live(self, tmp_path, monkeypatch):
        """When 207 is returned, migration.state is LIVE."""
        fresh = _make_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", fresh)

        original_rename = os.rename

        def _fail_rename(src, dst):
            if "trial_adapters" in str(dst):
                raise OSError("simulated rotation failure")
            return original_rename(src, dst)

        with patch("paramem.server.app.os.rename", _fail_rename):
            client = TestClient(app_module.app, raise_server_exceptions=False)
            client.post("/migration/rollback")

        assert fresh["migration"]["state"] == "LIVE"
