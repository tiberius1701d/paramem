"""Tests for POST /migration/rollback.

All tests run without GPU.  Covers the 8-step atomic ordering, precondition
gates, pre-mortem backup, A-config restore, marker clear, and the 207
Multi-Status path when rotation fails.
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
from paramem.backup.backup import write as backup_write
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
    a_yaml_bytes: bytes = _A_YAML,
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
    ``backup.read()`` (called in the rollback decrypt step) finds a
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
        a_yaml_bytes,
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
            pre_trial_config_sha256=_sha256(a_yaml_bytes),
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
        pre_trial_config_sha256=_sha256(a_yaml_bytes),
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


def _default_apply_stub_rollback():
    """Default _apply_config_live stub for rollback: no-op skip (applied_live=True)."""

    def _impl():
        return {
            "applied_live": True,
            "restart_required_reason": None,
            "skipped": "no_change",
            "cloud_only_reason": None,
        }

    return _impl


@pytest.fixture()
def state(tmp_path, monkeypatch):
    """TRIAL state with gates=fail (always-valid for rollback).

    Also patches _apply_config_live to avoid a real GPU load in unit tests.
    """
    fresh = _make_state(tmp_path)
    monkeypatch.setattr(app_module, "_state", fresh)
    monkeypatch.setattr(app_module, "_apply_config_live", _default_apply_stub_rollback())
    return fresh


@pytest.fixture()
def client(state):
    return TestClient(app_module.app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Happy path — rollback valid at any gate status
# ---------------------------------------------------------------------------


class TestRollbackHappyPath:
    def test_rollback_returns_200(self, client, state):
        """TRIAL + gates=fail → 200 RollbackResponse.

        With the default no-op-skip stub (disk=A, memory=A), applied_live=True
        and restart_required=False (live-apply path succeeded, no restart needed).
        """
        resp = client.post("/migration/rollback")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["state"] == "LIVE"
        assert body["applied_live"] is True
        assert body["restart_required"] is False

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

    def test_rollback_deletes_trial_graph(self, client, state, tmp_path):
        """Trial graph is deleted (not archived) after rollback (transient by design).

        The trial graph is RAM-only (stored in _state, not on disk) and its
        directory is cleaned up after rollback.  Once the operator rolls back,
        the trial_graph_dir is deleted rather than moved into the rotation slot.
        """
        trial_graph_dir = Path(state["migration"]["trial"]["trial_graph_dir"])
        assert trial_graph_dir.exists(), "fixture must create trial_graph dir"
        resp = client.post("/migration/rollback")
        assert resp.status_code == 200
        assert not trial_graph_dir.exists(), "trial graph must be deleted post-rollback"

    def test_rollback_archive_slot_has_no_graph_subdir(self, client, state, tmp_path):
        """Rotation slot must NOT contain a 'graph/' subdirectory after rollback.

        The trial graph is no longer archived — it is deleted post-rollback.
        """
        resp = client.post("/migration/rollback")
        assert resp.status_code == 200
        backups_root = state["config"].paths.data / "backups"
        trial_adapters = backups_root / "trial_adapters"
        if trial_adapters.exists():
            slots = [
                d for d in trial_adapters.iterdir() if d.is_dir() and not d.name.startswith(".")
            ]
            for slot in slots:
                assert not (slot / "graph").exists(), (
                    f"Rotation slot {slot.name} must not contain a 'graph/' subdir "
                    "(trial graph is deleted, not archived)"
                )

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

    def test_rollback_sets_banner(self, client, state):
        """After rollback, recovery_required contains a rollback banner.

        With the default no-op-skip stub, the banner says 'rolled back' and
        'already active' (not 'RESTART REQUIRED' since no restart is needed).
        """
        client.post("/migration/rollback")
        banners = state["migration"]["recovery_required"]
        assert banners, "No banner was set after rollback"
        assert any("rolled back" in b.lower() or "restart" in b.lower() for b in banners), (
            f"Expected rollback-related banner, got: {banners}"
        )

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


class TestRollbackRejectsUnbootableBackup:
    """A restore is a second door onto the "unbootable config goes live" defect.

    Config A was validated against the schema in force when it was captured; new
    load-time guards (e.g. ``max_interim_count=0`` + ``mode=simulate``, which is
    only rejected in this codebase version) can reject bytes that were bootable
    when the backup was written.  Rollback must refuse rather than rename.
    """

    def test_unbootable_a_config_rejected_live_config_untouched(self, tmp_path, monkeypatch):
        unbootable_a = b"consolidation:\n  mode: simulate\n  max_interim_count: 0\n"
        fresh = _make_state(tmp_path, a_yaml_bytes=unbootable_a)
        monkeypatch.setattr(app_module, "_state", fresh)
        monkeypatch.setattr(app_module, "_apply_config_live", _default_apply_stub_rollback())
        live_yaml = Path(fresh["config_path"])
        live_bytes_before = live_yaml.read_bytes()

        client = TestClient(app_module.app, raise_server_exceptions=False)
        resp = client.post("/migration/rollback")

        assert resp.status_code == 400, resp.text
        assert resp.json()["detail"]["error"] == "backup_unbootable"
        assert live_yaml.read_bytes() == live_bytes_before, "live config was mutated on rejection"
        # State remains TRIAL — the operator can retry with a different backup or
        # inspect the store; rollback did not silently drop them into no-state.
        assert fresh["migration"]["state"] == "TRIAL"

    def test_bootable_a_config_still_restores(self, tmp_path, monkeypatch):
        """Control: a bootable A config restores exactly as before this change."""
        fresh = _make_state(tmp_path)  # default _A_YAML is bootable
        monkeypatch.setattr(app_module, "_state", fresh)
        monkeypatch.setattr(app_module, "_apply_config_live", _default_apply_stub_rollback())
        live_yaml = Path(fresh["config_path"])

        client = TestClient(app_module.app, raise_server_exceptions=False)
        resp = client.post("/migration/rollback")

        assert resp.status_code in (200, 207), resp.text
        assert live_yaml.read_bytes() == _A_YAML


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
# Sidecar-vs-artifact regression — rollback restores original A bytes (not the sidecar)
# ---------------------------------------------------------------------------


class TestRollbackRestoresOriginalBytesViaRealWriter:
    """Verify that rollback restores the exact pre-confirm config bytes when the
    A-config backup slot was created by the real backup.write() call path.

    Regression: on some filesystems ``iterdir`` returns the sidecar
    ``config-<ts>.meta.json`` before the real artifact, causing rollback to
    overwrite configs/server.yaml with the 415-byte sidecar JSON instead of
    the real config artifact.
    """

    def test_rollback_restores_a_config_bytes_via_real_writer(self, tmp_path, monkeypatch):
        """Rollback with a real-writer slot: post-rollback live config == original A bytes.

        1. Write the A-config backup using backup.write() (exercises the real sidecar
           naming convention ``config-<ts>.meta.json``).
        2. Build a TRIAL state with the slot path and the artifact filename from the
           marker (as set by the confirm handler, which explicitly records the artifact name).
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
            "Sidecar-vs-artifact regression: sidecar JSON was restored instead of "
            "the config artifact."
        )


# ---------------------------------------------------------------------------
# Encrypted-artifact regression — rollback must decrypt encrypted A-config artifact
# (artifact-selection fix correctly picks the artifact, exposing missing decrypt step)
# ---------------------------------------------------------------------------


class TestRollbackDecryptsEncryptedArtifact:
    """Verify that rollback decrypts the A-config artifact when the daily
    identity is loaded and the backup writer produced an age envelope.

    Regression: rollback used ``os.rename(artifact, live_config_path)``
    which wrote ciphertext bytes verbatim.  The server then failed to start
    because ``yaml.safe_load`` raised on binary data.

    This test exercises the FULL encrypt → backup → rollback → decrypt round-trip
    using a real daily identity (key loaded → age envelope on disk).  It asserts
    that the post-rollback file content is the original plaintext bytes, not
    ciphertext.
    """

    def test_rollback_decrypts_encrypted_a_config(self, tmp_path, monkeypatch):
        """Rollback with age-encrypted A-config slot: post-rollback file is plaintext.

        Steps
        ------
        1. Mint + wire a daily identity so backup_write produces an age envelope.
        2. Write the A-config into a backup slot.
        3. Assert the artifact file is an age envelope (not plaintext).
        4. Build a TRIAL state with the encrypted slot.
        5. POST /migration/rollback.
        6. Assert live_config_path.read_bytes() == original plaintext.
        """
        from paramem.backup.age_envelope import is_age_envelope  # noqa: PLC0415
        from paramem.backup.key_store import (  # noqa: PLC0415
            DAILY_PASSPHRASE_ENV_VAR,
            _clear_daily_identity_cache,
            mint_daily_identity,
            wrap_daily_identity,
            write_daily_key_file,
        )

        # --- Step 1: real daily identity ---
        ident = mint_daily_identity()
        key_path = tmp_path / "daily_key.age"
        write_daily_key_file(wrap_daily_identity(ident, "rollback-pw"), key_path)
        monkeypatch.setenv(DAILY_PASSPHRASE_ENV_VAR, "rollback-pw")
        monkeypatch.setattr("paramem.backup.key_store.DAILY_KEY_PATH_DEFAULT", key_path)
        _clear_daily_identity_cache()

        a_bytes = _A_YAML  # original plaintext config

        # --- Step 2: write encrypted slot ---
        backups_root = tmp_path / "data" / "ha" / "backups"
        config_slot = backup_write(
            ArtifactKind.CONFIG,
            a_bytes,
            {"tier": "pre_migration"},
            base_dir=backups_root / "config",
        )

        # --- Step 3: verify the artifact is an age envelope ---
        artifact_files = [e for e in config_slot.iterdir() if not e.name.endswith(".meta.json")]
        assert len(artifact_files) == 1, (
            f"Expected exactly 1 artifact in {config_slot}, "
            f"got: {[e.name for e in config_slot.iterdir()]}"
        )
        artifact_file = artifact_files[0]
        assert is_age_envelope(artifact_file), (
            f"Expected age-encrypted artifact when daily identity is loaded, "
            f"got: {artifact_file.name!r} with magic "
            f"{artifact_file.read_bytes()[:8]!r}"
        )
        on_disk_bytes = artifact_file.read_bytes()
        assert on_disk_bytes != a_bytes, (
            "Artifact on disk should be ciphertext (not plaintext) when daily identity is loaded"
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
            f"Encrypted-artifact regression: rollback wrote wrong content.  "
            f"Expected plaintext A config ({len(a_bytes)} bytes), "
            f"got {len(restored)} bytes.  "
            f"First 60 bytes: {restored[:60]!r}."
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


# ---------------------------------------------------------------------------
# Response contract + live-apply wiring for rollback
# ---------------------------------------------------------------------------


def _stub_apply(result: dict):
    """Return a callable that stubs _apply_config_live."""

    def _impl():
        return result

    return _impl


class TestRollbackResponseContract:
    """Response contract: applied_live and restart_required_reason fields."""

    def test_rollback_response_has_applied_live(self, tmp_path, monkeypatch):
        """RollbackResponse includes applied_live field."""
        fresh = _make_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", fresh)
        monkeypatch.setattr(
            app_module,
            "_apply_config_live",
            _stub_apply(
                {
                    "applied_live": True,
                    "restart_required_reason": None,
                    "skipped": "no_change",
                    "cloud_only_reason": None,
                }
            ),
        )
        client = TestClient(app_module.app, raise_server_exceptions=False)
        resp = client.post("/migration/rollback")
        assert resp.status_code == 200
        body = resp.json()
        assert "applied_live" in body

    def test_rollback_response_has_restart_required_reason(self, tmp_path, monkeypatch):
        """RollbackResponse includes restart_required_reason field."""
        fresh = _make_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", fresh)
        monkeypatch.setattr(
            app_module,
            "_apply_config_live",
            _stub_apply(
                {
                    "applied_live": True,
                    "restart_required_reason": None,
                    "skipped": "no_change",
                    "cloud_only_reason": None,
                }
            ),
        )
        client = TestClient(app_module.app, raise_server_exceptions=False)
        resp = client.post("/migration/rollback")
        body = resp.json()
        assert "restart_required_reason" in body


class TestRollbackNoOpSkip:
    """Rollback no-op skip (S-6): disk hash == memory hash → applied_live=True."""

    def test_rollback_noop_skip_applied_live_true(self, tmp_path, monkeypatch):
        """No-op skip → applied_live=True, restart_required=False."""
        fresh = _make_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", fresh)
        monkeypatch.setattr(
            app_module,
            "_apply_config_live",
            _stub_apply(
                {
                    "applied_live": True,
                    "restart_required_reason": None,
                    "skipped": "no_change",
                    "cloud_only_reason": None,
                }
            ),
        )
        client = TestClient(app_module.app, raise_server_exceptions=False)
        resp = client.post("/migration/rollback")
        assert resp.status_code == 200
        body = resp.json()
        assert body["applied_live"] is True
        assert body["restart_required"] is False

    def test_rollback_noop_skip_banner_no_restart_required(self, tmp_path, monkeypatch):
        """No-op skip → banner does not say 'RESTART REQUIRED'."""
        fresh = _make_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", fresh)
        monkeypatch.setattr(
            app_module,
            "_apply_config_live",
            _stub_apply(
                {
                    "applied_live": True,
                    "restart_required_reason": None,
                    "skipped": "no_change",
                    "cloud_only_reason": None,
                }
            ),
        )
        client = TestClient(app_module.app, raise_server_exceptions=False)
        client.post("/migration/rollback")
        banners = fresh["migration"]["recovery_required"]
        assert not any("RESTART REQUIRED" in b for b in banners), (
            f"Unexpected 'RESTART REQUIRED' banner on no-op skip: {banners}"
        )


class TestRollback207BodyCarriesApplyFields:
    """207 rotation-failure path must carry the new apply fields in the body (correction #3)."""

    def test_207_body_has_applied_live(self, tmp_path, monkeypatch):
        """207 body includes applied_live field."""
        fresh = _make_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", fresh)
        monkeypatch.setattr(
            app_module,
            "_apply_config_live",
            _stub_apply(
                {
                    "applied_live": True,
                    "restart_required_reason": None,
                    "skipped": "no_change",
                    "cloud_only_reason": None,
                }
            ),
        )

        original_rename = os.rename

        def _fail_rotation(src, dst):
            if "trial_adapters" in str(dst):
                raise OSError("simulated rotation failure for 207")
            return original_rename(src, dst)

        with patch("paramem.server.app.os.rename", _fail_rotation):
            client = TestClient(app_module.app, raise_server_exceptions=False)
            resp = client.post("/migration/rollback")

        assert resp.status_code == 207
        body = resp.json()
        assert "applied_live" in body

    def test_207_body_has_restart_required(self, tmp_path, monkeypatch):
        """207 body includes restart_required field."""
        fresh = _make_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", fresh)
        monkeypatch.setattr(
            app_module,
            "_apply_config_live",
            _stub_apply(
                {
                    "applied_live": True,
                    "restart_required_reason": None,
                    "skipped": "no_change",
                    "cloud_only_reason": None,
                }
            ),
        )

        original_rename = os.rename

        def _fail_rotation(src, dst):
            if "trial_adapters" in str(dst):
                raise OSError("simulated rotation failure")
            return original_rename(src, dst)

        with patch("paramem.server.app.os.rename", _fail_rotation):
            client = TestClient(app_module.app, raise_server_exceptions=False)
            resp = client.post("/migration/rollback")

        assert resp.status_code == 207
        body = resp.json()
        assert "restart_required" in body
        # No-op skip with no carve → restart_required=False.
        assert body["restart_required"] is False

    def test_207_body_has_restart_required_reason(self, tmp_path, monkeypatch):
        """207 body includes restart_required_reason field."""
        fresh = _make_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", fresh)
        monkeypatch.setattr(
            app_module,
            "_apply_config_live",
            _stub_apply(
                {
                    "applied_live": True,
                    "restart_required_reason": None,
                    "skipped": "no_change",
                    "cloud_only_reason": None,
                }
            ),
        )

        original_rename = os.rename

        def _fail_rotation(src, dst):
            if "trial_adapters" in str(dst):
                raise OSError("simulated rotation failure")
            return original_rename(src, dst)

        with patch("paramem.server.app.os.rename", _fail_rotation):
            client = TestClient(app_module.app, raise_server_exceptions=False)
            resp = client.post("/migration/rollback")

        assert resp.status_code == 207
        body = resp.json()
        assert "restart_required_reason" in body


class TestRollbackMaintenanceGuard:
    """Maintenance guard (mode=cloud-only) is set before dispatching _apply_config_live."""

    def test_rollback_sets_maintenance_guard_before_apply(self, tmp_path, monkeypatch):
        """mode=cloud-only is set before _apply_config_live runs."""
        fresh = _make_state(tmp_path)
        fresh["mode"] = "local"
        monkeypatch.setattr(app_module, "_state", fresh)

        guard_set_before_apply = []

        def _capturing_apply():
            guard_set_before_apply.append(fresh.get("mode"))
            return {
                "applied_live": True,
                "restart_required_reason": None,
                "skipped": "no_change",
                "cloud_only_reason": None,
            }

        monkeypatch.setattr(app_module, "_apply_config_live", _capturing_apply)
        client = TestClient(app_module.app, raise_server_exceptions=False)
        client.post("/migration/rollback")
        assert guard_set_before_apply, "apply was never called"
        assert guard_set_before_apply[0] == "cloud-only", (
            f"Expected mode='cloud-only' before apply, got {guard_set_before_apply[0]!r}"
        )


# ---------------------------------------------------------------------------
# Base-swap rollback: restore_bundle, split-brain fix, marker + state cleanup
# ---------------------------------------------------------------------------


def _make_base_swap_state(tmp_path: Path) -> dict:
    """Build a TRIAL _state for base-swap rollback tests.

    Places a bundle slot directory on disk and writes a base-swap TrialMarker
    pointing at it.  The marker's ``migration_kind`` is ``"base_swap"`` so the
    rollback handler takes the base-swap branch.
    """
    live_yaml = tmp_path / "server.yaml"
    live_yaml.write_bytes(_LIVE_YAML)

    config = MagicMock()
    config.paths.data = tmp_path / "data" / "ha"
    config.paths.data.mkdir(parents=True, exist_ok=True)
    config.adapter_dir = tmp_path / "data" / "ha" / "adapters"
    config.adapter_dir.mkdir(parents=True, exist_ok=True)

    state_dir = config.paths.data / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    backups_root = config.paths.data / "backups"

    # Bundle slot: a directory that exists on disk (restore_bundle is mocked in
    # tests so the slot only needs to exist, not contain real bundle files).
    bundle_slot = backups_root / "bundles" / "20260524-000000"
    bundle_slot.mkdir(parents=True, exist_ok=True)
    bundle_slot_str = str(bundle_slot.resolve())

    marker = TrialMarker(
        schema_version=TRIAL_MARKER_SCHEMA_VERSION,
        started_at="2026-05-24T00:00:00+00:00",
        pre_trial_config_sha256=_sha256(_A_YAML),
        candidate_config_sha256=_sha256(_LIVE_YAML),
        backup_paths={"bundle": bundle_slot_str},
        trial_adapter_dir=str(state_dir / "trial_adapter"),
        trial_graph_dir=str(state_dir / "trial_graph"),
        config_artifact_filename="",
        migration_kind="base_swap",
        base_swap_phase="phaseA_done",
        old_model="mistral",
        new_model="qwen3-4b",
        bundle_slot=bundle_slot_str,
    )
    write_trial_marker(state_dir, marker)

    from paramem.server.migration import TrialStash, initial_migration_state

    trial_stash = TrialStash(
        started_at="2026-05-24T00:00:00+00:00",
        pre_trial_config_sha256=_sha256(_A_YAML),
        candidate_config_sha256=_sha256(_LIVE_YAML),
        backup_paths={"bundle": bundle_slot_str},
        trial_adapter_dir=str(state_dir / "trial_adapter"),
        trial_graph_dir=str(state_dir / "trial_graph"),
        gates={"status": "pass", "completed_at": "2026-05-24T01:00:00+00:00"},
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
        "server_started_at": "2026-05-24T00:00:00+00:00",
        "mode": "normal",
        "background_trainer": None,
        "consolidation_loop": None,
        "config_drift": {
            "detected": False,
            "loaded_hash": _sha256(_LIVE_YAML),
            "disk_hash": _sha256(_LIVE_YAML),
            "last_checked_at": "2026-05-24T00:00:00+00:00",
        },
    }


class TestBaseSwapRollback:
    """Base-swap rollback: restore_bundle path, split-brain fix, cleanup."""

    def _make_apply_stub(self, state_dict: dict) -> callable:
        """Return a stub that marks mode='local' after being called."""

        def _apply():
            state_dict["mode"] = "local"
            state_dict["cloud_only_reason"] = None

        return _apply

    def test_restore_bundle_called_with_restore_config_true(self, tmp_path, monkeypatch):
        """POST /migration/rollback on base-swap TRIAL calls restore_bundle(restore_config=True).

        restore_bundle must be called exactly once, with the bundle slot from
        the marker, restore_config=True, and the live config path.
        """
        fresh = _make_base_swap_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", fresh)
        monkeypatch.setattr(app_module, "_apply_config_live", self._make_apply_stub(fresh))

        restore_calls = []

        def _fake_restore(bundle_slot_path, *, data_dir, config_path, restore_config=False):
            restore_calls.append(
                {
                    "bundle_slot_path": bundle_slot_path,
                    "restore_config": restore_config,
                    "config_path": config_path,
                }
            )

        with patch("paramem.backup.backup.restore_bundle", _fake_restore):
            client = TestClient(app_module.app, raise_server_exceptions=False)
            resp = client.post("/migration/rollback")

        assert resp.status_code == 200, resp.text
        assert len(restore_calls) == 1, f"restore_bundle must be called once; got {restore_calls}"
        assert restore_calls[0]["restore_config"] is True, (
            "Base-swap rollback must call restore_bundle(restore_config=True)"
        )
        # bundle_slot_path must be a Path, not a string.
        import pathlib

        assert isinstance(restore_calls[0]["bundle_slot_path"], pathlib.Path), (
            "restore_bundle receives a Path, not a string"
        )

    def test_loaded_hash_invalidated_before_apply(self, tmp_path, monkeypatch):
        """config_drift['loaded_hash'] is set to the sentinel BEFORE _apply_config_live runs.

        The split-brain fix: if Phase A of the base-swap already swapped the live
        config to Qwen3, the in-memory config_drift.loaded_hash equals the Qwen3
        hash.  restore_bundle writes the Mistral config to disk; loaded_hash must be
        invalidated so _apply_config_live cannot skip the reload.
        """
        fresh = _make_base_swap_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", fresh)

        loaded_hash_at_apply_time = []

        def _capturing_apply():
            # Record what loaded_hash was at apply time.
            cd = fresh.get("config_drift") or {}
            loaded_hash_at_apply_time.append(cd.get("loaded_hash"))
            fresh["mode"] = "local"
            fresh["cloud_only_reason"] = None

        monkeypatch.setattr(app_module, "_apply_config_live", _capturing_apply)

        with patch("paramem.backup.backup.restore_bundle", lambda *a, **kw: None):
            client = TestClient(app_module.app, raise_server_exceptions=False)
            resp = client.post("/migration/rollback")

        assert resp.status_code == 200, resp.text
        assert loaded_hash_at_apply_time, "_apply_config_live was never called"
        assert loaded_hash_at_apply_time[0] == "__rollback_invalidated__", (
            f"loaded_hash must be '__rollback_invalidated__' at apply time; "
            f"got {loaded_hash_at_apply_time[0]!r}"
        )

    def test_marker_cleared_after_restore(self, tmp_path, monkeypatch):
        """Trial marker is cleared after restore_bundle and apply succeed."""
        fresh = _make_base_swap_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", fresh)
        monkeypatch.setattr(app_module, "_apply_config_live", self._make_apply_stub(fresh))

        state_dir = fresh["config"].paths.data / "state"
        # Marker exists before rollback.
        assert read_trial_marker(state_dir) is not None

        with patch("paramem.backup.backup.restore_bundle", lambda *a, **kw: None):
            client = TestClient(app_module.app, raise_server_exceptions=False)
            resp = client.post("/migration/rollback")

        assert resp.status_code == 200, resp.text
        assert read_trial_marker(state_dir) is None, "Trial marker must be cleared after rollback"

    def test_migration_state_reset_to_live(self, tmp_path, monkeypatch):
        """migration.state is LIVE after base-swap rollback."""
        fresh = _make_base_swap_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", fresh)
        monkeypatch.setattr(app_module, "_apply_config_live", self._make_apply_stub(fresh))

        with patch("paramem.backup.backup.restore_bundle", lambda *a, **kw: None):
            client = TestClient(app_module.app, raise_server_exceptions=False)
            resp = client.post("/migration/rollback")

        assert resp.status_code == 200, resp.text
        assert fresh["migration"]["state"] == "LIVE", (
            f"Expected migration.state=LIVE after rollback; got {fresh['migration']['state']!r}"
        )

    def test_rollback_response_no_restart_required(self, tmp_path, monkeypatch):
        """Base-swap rollback returns restart_required=False (in-process reload)."""
        fresh = _make_base_swap_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", fresh)
        monkeypatch.setattr(app_module, "_apply_config_live", self._make_apply_stub(fresh))

        with patch("paramem.backup.backup.restore_bundle", lambda *a, **kw: None):
            client = TestClient(app_module.app, raise_server_exceptions=False)
            resp = client.post("/migration/rollback")

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["restart_required"] is False, (
            "Base-swap rollback must not require restart — in-process reload is used"
        )
        assert body["applied_live"] is True

    def test_missing_bundle_slot_returns_500(self, tmp_path, monkeypatch):
        """If the bundle slot directory is missing, rollback returns 500."""
        fresh = _make_base_swap_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", fresh)
        monkeypatch.setattr(app_module, "_apply_config_live", self._make_apply_stub(fresh))

        # Delete the bundle slot that the marker points to.
        bundle_slot_path = fresh["config"].paths.data / "backups" / "bundles" / "20260524-000000"
        import shutil

        shutil.rmtree(bundle_slot_path)

        client = TestClient(app_module.app, raise_server_exceptions=False)
        resp = client.post("/migration/rollback")

        assert resp.status_code == 500, resp.text
        body = resp.json()
        assert body["detail"]["error"] == "rollback_precondition_failed"

    def test_maintenance_guard_set_before_apply_in_base_swap(self, tmp_path, monkeypatch):
        """mode=cloud-only is set before _apply_config_live runs in base-swap rollback."""
        fresh = _make_base_swap_state(tmp_path)
        fresh["mode"] = "local"
        monkeypatch.setattr(app_module, "_state", fresh)

        guard_mode_at_apply = []

        def _capturing_apply():
            guard_mode_at_apply.append(fresh.get("mode"))
            fresh["mode"] = "local"
            fresh["cloud_only_reason"] = None

        monkeypatch.setattr(app_module, "_apply_config_live", _capturing_apply)

        with patch("paramem.backup.backup.restore_bundle", lambda *a, **kw: None):
            client = TestClient(app_module.app, raise_server_exceptions=False)
            client.post("/migration/rollback")

        assert guard_mode_at_apply, "_apply_config_live was never called"
        assert guard_mode_at_apply[0] == "cloud-only", (
            f"Expected mode='cloud-only' before apply in base-swap rollback; "
            f"got {guard_mode_at_apply[0]!r}"
        )
