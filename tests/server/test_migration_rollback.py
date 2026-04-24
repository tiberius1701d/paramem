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
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

import paramem.server.app as app_module
from paramem.backup.backup import write as backup_write
from paramem.backup.encryption import _clear_cipher_cache
from paramem.backup.types import ArtifactKind
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
        Filename to use in the marker.  Pass ``""`` to test the empty-filename
        error path.  Pass any non-empty string to use the real backup.write()
        writer (the filename is recorded in the marker but the slot is always
        created via backup.write() so the sidecar is present and backup.read()
        succeeds).

    Notes
    -----
    The A-config backup slot is always created using ``backup.write()`` so that
    ``backup.read()`` (called in the rollback decrypt step, B6 fix) finds a
    valid sidecar.  Callers that need to test missing-artifact (step 3) should
    call ``_make_state`` and then unlink the artifact from the returned slot
    directory.  The slot directory path is stored in
    ``state["migration"]["trial"]["backup_paths"]["config"]``.
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

    backups_root = config.paths.data / "backups"

    # Create config A backup slot using the real backup writer so the sidecar
    # (*.meta.json) is present and backup.read() can decrypt/validate correctly.
    # For the empty-filename error path (config_artifact_filename=""), we still
    # create a proper slot but record "" in the marker so the precondition gate
    # fires before backup.read() is reached.
    config_slot = backup_write(
        ArtifactKind.CONFIG,
        _A_YAML,
        {"tier": "pre_migration"},
        base_dir=backups_root / "config",
    )

    # Derive the artifact filename from the real slot for the marker.
    # For empty-filename tests, override below.
    real_artifact_files = [e for e in config_slot.iterdir() if not e.name.endswith(".meta.json")]
    real_config_artifact_filename = real_artifact_files[0].name if real_artifact_files else ""

    # Honour the caller's config_artifact_filename for the marker: use "" for
    # the missing-filename error test; use the real filename otherwise.
    marker_artifact_filename = (
        config_artifact_filename
        if config_artifact_filename == ""
        else real_config_artifact_filename
    )

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
            config_artifact_filename=marker_artifact_filename,
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
        fresh = _make_state(tmp_path)
        # Remove the A artifact from the real backup slot to trigger the missing-file
        # error.  The slot path is stored in backup_paths["config"]; find the artifact
        # file (not the sidecar) and unlink it so the precondition check fires.
        config_slot = Path(fresh["migration"]["trial"]["backup_paths"]["config"])
        artifact_files = [e for e in config_slot.iterdir() if not e.name.endswith(".meta.json")]
        assert len(artifact_files) == 1, "Expected exactly one artifact in slot"
        artifact_files[0].unlink()
        monkeypatch.setattr(app_module, "_state", fresh)
        client = TestClient(app_module.app, raise_server_exceptions=False)
        resp = client.post("/migration/rollback")
        assert resp.status_code == 500
        assert resp.json()["detail"]["error"] == "rollback_precondition_failed"
        # Pre-mortem backup should be cleaned up.
        backups_root = fresh["config"].paths.data / "backups"
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
            # Fail only the atomic rename of the pending rollback temp to live path.
            if ".pending-rollback-" in src_path.name and dst_path.name == "server.yaml":
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


# ---------------------------------------------------------------------------
# B1 regression — rollback restores original A bytes (not the sidecar)
# ---------------------------------------------------------------------------


class TestRollbackRestoresOriginalBytesViaRealWriter:
    """Verify that rollback restores the exact pre-confirm config bytes when the
    A-config backup slot was created by the real backup.write() call path.

    This test would have caught the B1 bug (2026-04-22 E2E baseline):
    the sidecar ``config-<ts>.meta.json`` was returned first by iterdir on some
    filesystems, causing rollback to overwrite configs/server.yaml with the 415-byte
    sidecar JSON instead of the real config artifact.
    """

    def test_rollback_restores_a_config_bytes_via_real_writer(self, tmp_path, monkeypatch):
        """Rollback with a real-writer slot: post-rollback live config == original A bytes.

        1. Write the A-config backup using backup.write() (exercises the real sidecar
           naming convention ``config-<ts>.meta.json``).
        2. Build a TRIAL state with the slot path and the artifact filename from the
           marker (as set by the confirm handler after the B1 fix).
        3. POST /migration/rollback.
        4. Assert live_config_path.read_bytes() == _A_YAML.
        """
        # --- Write A-config into a real backup slot ---
        a_bytes = _A_YAML
        backups_root = tmp_path / "data" / "ha" / "backups"
        config_slot = backup_write(
            ArtifactKind.CONFIG,
            a_bytes,
            {"tier": "pre_migration"},
            base_dir=backups_root / "config",
        )

        # Identify the artifact filename (must end with .bin, not .meta.json).
        artifact_files = [e for e in config_slot.iterdir() if not e.name.endswith(".meta.json")]
        got_names = [e.name for e in config_slot.iterdir()]
        assert len(artifact_files) == 1, (
            f"Expected exactly 1 artifact in {config_slot}, got: {got_names}"
        )
        config_artifact_filename = artifact_files[0].name
        assert config_artifact_filename.endswith(".bin") or config_artifact_filename.endswith(
            ".bin.enc"
        ), f"Real writer artifact must end with .bin or .bin.enc, got: {config_artifact_filename!r}"

        # --- Build TRIAL state pointing at that real slot ---
        live_yaml = tmp_path / "server.yaml"
        live_yaml.write_bytes(_LIVE_YAML)  # config B (the candidate)

        config = MagicMock()
        config.paths.data = tmp_path / "data" / "ha"
        config.paths.data.mkdir(parents=True, exist_ok=True)
        config.adapter_dir = tmp_path / "data" / "ha" / "adapters"
        config.adapter_dir.mkdir(parents=True, exist_ok=True)

        state_dir = config.paths.data / "state"
        state_dir.mkdir(parents=True, exist_ok=True)

        trial_adapter_dir = state_dir / "trial_adapter"
        trial_adapter_dir.mkdir(exist_ok=True)
        (trial_adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
        trial_graph_dir = state_dir / "trial_graph"
        trial_graph_dir.mkdir(exist_ok=True)

        marker = TrialMarker(
            schema_version=TRIAL_MARKER_SCHEMA_VERSION,
            started_at="2026-04-22T01:00:00+00:00",
            pre_trial_config_sha256=_sha256(a_bytes),
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
            pre_trial_config_sha256=_sha256(a_bytes),
            candidate_config_sha256=_sha256(_LIVE_YAML),
            backup_paths={
                "config": str(config_slot.resolve()),
                "graph": str(backups_root / "graph" / "20260422-010000"),
                "registry": str(backups_root / "registry" / "20260422-010000"),
            },
            trial_adapter_dir=str(trial_adapter_dir.resolve()),
            trial_graph_dir=str(trial_graph_dir.resolve()),
            gates={"status": "fail", "completed_at": "2026-04-22T02:00:00+00:00"},
        )

        migration = initial_migration_state()
        migration["state"] = "TRIAL"
        migration["trial"] = trial_stash

        fresh = {
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
        monkeypatch.setattr(app_module, "_state", fresh)
        client = TestClient(app_module.app, raise_server_exceptions=False)

        resp = client.post("/migration/rollback")
        assert resp.status_code == 200, resp.text

        # Post-rollback: live config must contain A bytes, not sidecar JSON.
        restored = live_yaml.read_bytes()
        assert restored == a_bytes, (
            f"Rollback restored wrong content.  "
            f"Expected A config ({len(a_bytes)} bytes), got {len(restored)} bytes.  "
            f"First 100 chars: {restored[:100]!r}.  "
            "This is the B1 regression: sidecar JSON was restored instead of the config artifact."
        )


# ---------------------------------------------------------------------------
# B6 regression — rollback must decrypt encrypted A-config artifact
# (2026-04-22 re-test: B1 fix correctly picks artifact, exposing missing decrypt)
# ---------------------------------------------------------------------------


class TestRollbackDecryptsEncryptedArtifact:
    """Verify that rollback decrypts the A-config artifact when ``PARAMEM_MASTER_KEY``
    is set and the backup writer used Fernet encryption.

    Before the B6 fix, rollback did ``os.rename(artifact, live_config_path)``
    which wrote ciphertext bytes verbatim.  The server then failed to start
    because ``yaml.safe_load`` raised on binary Fernet data.

    This test exercises the FULL encrypt → backup → rollback → decrypt round-trip
    using a real Fernet key in the environment (key loaded → encryption happens).
    It asserts that the post-rollback file content is the original plaintext
    bytes, not ciphertext.
    """

    def test_rollback_decrypts_encrypted_a_config(self, tmp_path, monkeypatch, request):
        """Rollback with encrypted A-config slot: post-rollback file is plaintext.

        Steps
        ------
        1. Generate a real Fernet key and set ``PARAMEM_MASTER_KEY``.
        2. Write the A-config into a backup slot using the real ``backup.write()``
           path (key loaded → Fernet ciphertext on disk).
        3. Assert the artifact file on disk is ciphertext (not plaintext) so we
           know encryption actually happened.
        4. Build a TRIAL state with the encrypted slot.
        5. POST /migration/rollback.
        6. Assert live_config_path.read_bytes() == original plaintext (decrypt verified).
        7. Assert the artifact on disk is NOT the on-disk bytes (decryption happened).
        """
        # --- Step 1: real Fernet key in env ---
        fernet_key = Fernet.generate_key().decode()
        monkeypatch.setenv("PARAMEM_MASTER_KEY", fernet_key)
        # Clear the module-level cipher cache so our new key is picked up.
        _clear_cipher_cache()
        # Clear cache after test — monkeypatch restores the env var, but the
        # module-level cipher object retains the old key until cleared.
        request.addfinalizer(_clear_cipher_cache)

        a_bytes = _A_YAML  # original plaintext config

        # --- Step 2: write encrypted slot ---
        backups_root = tmp_path / "data" / "ha" / "backups"
        config_slot = backup_write(
            ArtifactKind.CONFIG,
            a_bytes,
            {"tier": "pre_migration"},
            base_dir=backups_root / "config",
        )

        # --- Step 3: verify the artifact is ciphertext (encryption happened) ---
        artifact_files = [e for e in config_slot.iterdir() if not e.name.endswith(".meta.json")]
        assert len(artifact_files) == 1, (
            f"Expected exactly 1 artifact in {config_slot}, "
            f"got: {[e.name for e in config_slot.iterdir()]}"
        )
        artifact_file = artifact_files[0]
        assert artifact_file.name.endswith(".bin.enc"), (
            f"Expected encrypted artifact (.bin.enc) when key is set, got: {artifact_file.name!r}"
        )
        on_disk_bytes = artifact_file.read_bytes()
        assert on_disk_bytes != a_bytes, (
            "Artifact on disk should be ciphertext (not plaintext) when PARAMEM_MASTER_KEY is set"
        )

        config_artifact_filename = artifact_file.name

        # --- Step 4: build TRIAL state with encrypted slot ---
        live_yaml = tmp_path / "server.yaml"
        live_yaml.write_bytes(_LIVE_YAML)

        config = MagicMock()
        config.paths.data = tmp_path / "data" / "ha"
        config.paths.data.mkdir(parents=True, exist_ok=True)
        config.adapter_dir = tmp_path / "data" / "ha" / "adapters"
        config.adapter_dir.mkdir(parents=True, exist_ok=True)

        state_dir = config.paths.data / "state"
        state_dir.mkdir(parents=True, exist_ok=True)

        trial_adapter_dir = state_dir / "trial_adapter"
        trial_adapter_dir.mkdir(exist_ok=True)
        (trial_adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
        trial_graph_dir = state_dir / "trial_graph"
        trial_graph_dir.mkdir(exist_ok=True)

        marker = TrialMarker(
            schema_version=TRIAL_MARKER_SCHEMA_VERSION,
            started_at="2026-04-22T01:00:00+00:00",
            pre_trial_config_sha256=_sha256(a_bytes),
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
            pre_trial_config_sha256=_sha256(a_bytes),
            candidate_config_sha256=_sha256(_LIVE_YAML),
            backup_paths={
                "config": str(config_slot.resolve()),
                "graph": str(backups_root / "graph" / "20260422-010000"),
                "registry": str(backups_root / "registry" / "20260422-010000"),
            },
            trial_adapter_dir=str(trial_adapter_dir.resolve()),
            trial_graph_dir=str(trial_graph_dir.resolve()),
            gates={"status": "fail", "completed_at": "2026-04-22T02:00:00+00:00"},
        )

        migration = initial_migration_state()
        migration["state"] = "TRIAL"
        migration["trial"] = trial_stash

        fresh = {
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
        monkeypatch.setattr(app_module, "_state", fresh)
        client = TestClient(app_module.app, raise_server_exceptions=False)

        # --- Step 5: trigger rollback ---
        resp = client.post("/migration/rollback")
        assert resp.status_code == 200, (
            f"Rollback should succeed (200), got {resp.status_code}: {resp.text}"
        )

        # --- Step 6: post-rollback file is plaintext ---
        restored = live_yaml.read_bytes()
        assert restored == a_bytes, (
            f"B6 regression: rollback wrote wrong content.  "
            f"Expected plaintext A config ({len(a_bytes)} bytes), "
            f"got {len(restored)} bytes.  "
            f"First 60 bytes: {restored[:60]!r}.  "
            "If this starts with 'gAAAAA', the decrypt step was skipped."
        )

        # --- Step 7: confirm the on-disk artifact was ciphertext (decrypt was needed) ---
        # The on-disk bytes must differ from the restored plaintext.
        assert on_disk_bytes != restored, (
            "on_disk_bytes == restored: encryption did not happen, so the test "
            "does not exercise the decrypt path.  Check PARAMEM_MASTER_KEY setup."
        )


# ---------------------------------------------------------------------------
# Fix 2 — pending restore temp file must be written at 0o600
# ---------------------------------------------------------------------------


class TestRollbackPendingRestoreFileMode:
    """Fix 2 (2026-04-23): the .pending-rollback-*.yaml temp file must be created
    at 0o600 so that plaintext config bytes are not exposed to other users during
    the rename window."""

    def test_pending_rollback_temp_file_has_mode_0o600(self, tmp_path, monkeypatch):
        """Rollback temp file is created with mode 0o600 (Fix 2 regression test).

        Intercepts os.open at the point where _build_trial_loop writes the
        pending temp and asserts the mode argument is 0o600.
        """
        captured_modes: list[int] = []
        real_os_open = os.open

        def _spy_open(path, flags, mode=0o644, **kwargs):
            # Capture the mode passed to os.open for files matching .pending-rollback
            if ".pending-rollback" in str(path):
                captured_modes.append(mode)
            return real_os_open(path, flags, mode, **kwargs)

        fresh = _make_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", fresh)
        client = TestClient(app_module.app, raise_server_exceptions=False)

        with patch("paramem.server.app.os.open", side_effect=_spy_open):
            resp = client.post("/migration/rollback")

        assert resp.status_code == 200, f"rollback failed: {resp.text}"
        assert len(captured_modes) >= 1, (
            "os.open was not called for the .pending-rollback temp file — "
            "Fix 2 may have been reverted"
        )
        for mode in captured_modes:
            assert mode == 0o600, (
                f"pending-rollback temp file created with mode {oct(mode)}, expected 0o600 "
                "(Fix 2 regression: plaintext exposed during rename window)"
            )
