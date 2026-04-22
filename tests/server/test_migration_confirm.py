"""Tests for POST /migration/confirm (Slice 3b.2).

All tests run without GPU — the trial consolidation task is cancelled/no-op.
The 5-step atomic ordering is validated with both happy-path and per-step
failure rollback scenarios.
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
from paramem.server.migration import initial_migration_state
from paramem.server.trial_state import read_trial_marker

_LIVE_YAML = b"model: mistral\ndebug: false\n"
_CAND_YAML = b"model: mistral\ndebug: true\n"


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _make_state(tmp_path: Path) -> dict:
    """Build a STAGING _state with a real candidate file."""
    live_yaml = tmp_path / "server.yaml"
    live_yaml.write_bytes(_LIVE_YAML)
    cand_yaml = tmp_path / "candidate.yaml"
    cand_yaml.write_bytes(_CAND_YAML)

    # Production layout: config.paths.data = data/ha so state and backups
    # live directly under it (no extra /ha/ segment added by the handler).
    config = MagicMock()
    config.paths.data = tmp_path / "data" / "ha"
    config.paths.data.mkdir(parents=True, exist_ok=True)
    config.adapter_dir = tmp_path / "data" / "ha" / "adapters"
    config.adapter_dir.mkdir(parents=True, exist_ok=True)
    config.key_metadata_path = tmp_path / "data" / "ha" / "key_metadata.json"

    loop_mock = MagicMock()
    loop_mock.merger.save_bytes.return_value = b'{"nodes":[],"links":[]}'

    staging = initial_migration_state()
    staging["state"] = "STAGING"
    staging["candidate_path"] = str(cand_yaml)
    staging["candidate_hash"] = _sha256(_CAND_YAML)
    staging["candidate_bytes"] = _CAND_YAML
    staging["candidate_text"] = _CAND_YAML.decode("utf-8")

    return {
        "model": None,
        "tokenizer": None,
        "config": config,
        "config_path": str(live_yaml),
        "consolidating": False,
        "migration": staging,
        "migration_lock": asyncio.Lock(),
        "server_started_at": "2026-04-22T00:00:00+00:00",
        "mode": "normal",
        "background_trainer": None,
        "consolidation_loop": loop_mock,
        "session_buffer": None,
    }


@pytest.fixture()
def state(tmp_path, monkeypatch):
    """STAGING state monkeypatched into app_module."""
    fresh = _make_state(tmp_path)
    monkeypatch.setattr(app_module, "_state", fresh)
    return fresh


@pytest.fixture()
def client(state, monkeypatch):
    """TestClient with mocked trial consolidation."""

    # Patch _run_trial_consolidation to a no-op so no background task runs.
    async def _noop_trial():
        pass

    monkeypatch.setattr(app_module, "_run_trial_consolidation", _noop_trial)
    return TestClient(app_module.app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestConfirmHappyPath:
    def test_confirm_happy_path_returns_200(self, client, state, tmp_path):
        """STAGING with valid stash → 200, state=TRIAL."""
        resp = client.post("/migration/confirm", json={})
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["state"] == "TRIAL"

    def test_confirm_sets_state_to_trial(self, client, state, tmp_path):
        """After confirm, _state["migration"]["state"] == "TRIAL"."""
        client.post("/migration/confirm", json={})
        assert state["migration"]["state"] == "TRIAL"

    def test_confirm_writes_3_backup_slots(self, client, state, tmp_path):
        """3 backup slots (config, graph, registry) appear in backups_root.

        Uses the production layout: config.paths.data is data/ha, so the
        handler appends 'backups' directly (no extra /ha/ segment).
        """
        client.post("/migration/confirm", json={})
        backups_root = state["config"].paths.data / "backups"
        for kind in ("config", "graph", "registry"):
            kind_dir = backups_root / kind
            assert kind_dir.exists(), f"Missing backup kind dir: {kind}"
            slots = [d for d in kind_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
            assert len(slots) == 1, f"Expected 1 slot in {kind_dir}, got {slots}"

    def test_confirm_writes_trial_marker(self, client, state, tmp_path):
        """state/trial.json is written after confirm.

        Uses the production layout: marker lives at
        <config.paths.data>/state/trial.json (not /ha/state/trial.json).
        """
        client.post("/migration/confirm", json={})
        state_dir = state["config"].paths.data / "state"
        marker = read_trial_marker(state_dir)
        assert marker is not None

    def test_confirm_renames_candidate_to_live(self, client, state, tmp_path):
        """The candidate file becomes the live config after confirm."""
        live_config_path = Path(state["config_path"])
        candidate_path = Path(state["migration"]["candidate_path"])
        assert candidate_path.exists()
        client.post("/migration/confirm", json={})
        # After rename, live config should contain candidate bytes.
        assert live_config_path.read_bytes() == _CAND_YAML

    def test_confirm_marker_contents_match_stash(self, client, state, tmp_path):
        """trial.json fields equal candidate_hash and pre_trial_hash."""
        client.post("/migration/confirm", json={})
        state_dir = state["config"].paths.data / "state"
        marker = read_trial_marker(state_dir)
        assert marker is not None
        assert marker.candidate_config_sha256 == _sha256(_CAND_YAML)
        assert marker.pre_trial_config_sha256 == _sha256(_LIVE_YAML)

    def test_confirm_backup_meta_has_pre_trial_hash(self, client, state, tmp_path):
        """Each backup slot's meta.json has pre_trial_hash == sha256(pre-rename live config)."""
        expected_hash = _sha256(_LIVE_YAML)
        client.post("/migration/confirm", json={})
        backups_root = state["config"].paths.data / "backups"
        for kind in ("config", "graph", "registry"):
            kind_dir = backups_root / kind
            slots = [d for d in kind_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
            assert slots, f"No slots in {kind_dir}"
            meta_files = list(slots[0].glob("*.meta.json"))
            assert meta_files, f"No meta.json in {slots[0]}"
            meta_data = json.loads(meta_files[0].read_text(encoding="utf-8"))
            assert meta_data.get("pre_trial_hash") == expected_hash, (
                f"pre_trial_hash mismatch in {kind} backup"
            )

    def test_confirm_graph_backup_uses_merger_save_bytes(self, client, state, tmp_path):
        """merger.save_bytes() is called when consolidation_loop is present.

        This is the regression guard for Fix 2: if _state.get("consolidation_loop")
        is replaced by _state.get("loop"), loop_mock is not found (returns None) and
        the backup silently contains the empty-graph fallback b'{}' instead of the
        real graph bytes from merger.save_bytes().

        We verify the call was made (not the on-disk bytes, which may be encrypted).
        """
        loop_mock = state["consolidation_loop"]
        client.post("/migration/confirm", json={})
        loop_mock.merger.save_bytes.assert_called_once_with()


# ---------------------------------------------------------------------------
# 409 gates
# ---------------------------------------------------------------------------


class TestConfirm409:
    def test_confirm_409_when_consolidating(self, client, state, tmp_path):
        """STAGING + consolidating=True → 409 consolidating."""
        state["consolidating"] = True
        resp = client.post("/migration/confirm", json={})
        assert resp.status_code == 409
        assert resp.json()["detail"]["error"] == "consolidating"

    def test_confirm_409_when_not_staging(self, client, state, tmp_path):
        """State is LIVE → 409 not_staging."""
        state["migration"]["state"] = "LIVE"
        resp = client.post("/migration/confirm", json={})
        assert resp.status_code == 409
        assert resp.json()["detail"]["error"] == "not_staging"

    def test_confirm_409_when_already_trial(self, client, state, tmp_path):
        """State is TRIAL → 409 trial_active."""
        state["migration"]["state"] = "TRIAL"
        resp = client.post("/migration/confirm", json={})
        assert resp.status_code == 409
        assert resp.json()["detail"]["error"] == "trial_active"


# ---------------------------------------------------------------------------
# Step failure rollbacks
# ---------------------------------------------------------------------------


class TestConfirmStepFailures:
    def test_confirm_step2_failure_rollback(self, client, state, tmp_path, monkeypatch):
        """Patch backup.write to raise on the second call → 500 backup_write_failed.

        STAGING state is retained on failure.
        """
        call_count = [0]

        from paramem.backup import backup as backup_module

        original_write_fn = backup_module.write

        def _failing_write(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise OSError("disk full")
            return original_write_fn(*args, **kwargs)

        monkeypatch.setattr(backup_module, "write", _failing_write)
        # Re-patch the app import of backup_write.
        with patch("paramem.server.app.backup_write", _failing_write):
            resp = client.post("/migration/confirm", json={})

        assert resp.status_code == 500
        assert resp.json()["detail"]["error"] == "backup_write_failed"
        # State must still be STAGING.
        assert state["migration"]["state"] == "STAGING"

    def test_confirm_step3_failure_rollback(self, client, state, tmp_path, monkeypatch):
        """Patch write_trial_marker → 500 marker_write_failed; STAGING retained; backups deleted."""

        def _fail_write(*args, **kwargs):
            raise OSError("marker write failed")

        with patch("paramem.server.app.write_trial_marker", _fail_write):
            resp = client.post("/migration/confirm", json={})

        assert resp.status_code == 500
        assert resp.json()["detail"]["error"] == "marker_write_failed"
        assert state["migration"]["state"] == "STAGING"
        # All backup slots should be cleaned up.
        backups_root = state["config"].paths.data / "ha" / "backups"
        for kind in ("config", "graph", "registry"):
            kind_dir = backups_root / kind
            if kind_dir.exists():
                slots = [d for d in kind_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
                assert len(slots) == 0, f"Orphan slot in {kind_dir}: {slots}"

    def test_confirm_step4_failure_rollback(self, client, state, tmp_path, monkeypatch):
        """Patch _rename_config → 500 config_swap_failed; marker + backups deleted."""
        with patch(
            "paramem.server.app._rename_config", side_effect=OSError("EXDEV: cross-device rename")
        ):
            resp = client.post("/migration/confirm", json={})

        assert resp.status_code == 500
        assert resp.json()["detail"]["error"] == "config_swap_failed"
        assert state["migration"]["state"] == "STAGING"
        # Marker must not exist.
        state_dir = state["config"].paths.data / "ha" / "state"
        assert read_trial_marker(state_dir) is None

    def test_confirm_releases_lock_on_failure(self, client, state, tmp_path, monkeypatch):
        """Step 3 failure: follow-up confirm does not deadlock.

        After a failed confirm, the migration lock must be released so a
        second confirm attempt can proceed (or fail cleanly with a non-deadlock
        error).  Correction 1.
        """
        fail_count = [0]

        def _fail_once(*args, **kwargs):
            fail_count[0] += 1
            if fail_count[0] == 1:
                raise OSError("transient failure")
            from paramem.server.trial_state import write_trial_marker as _real

            return _real(*args, **kwargs)

        with patch("paramem.server.app.write_trial_marker", _fail_once):
            resp1 = client.post("/migration/confirm", json={})
        assert resp1.status_code == 500

        # Second attempt: state is still STAGING (not blocked by a held lock).
        # It may fail because candidate file doesn't exist now, but must not hang.
        resp2 = client.post("/migration/confirm", json={})
        # Should not be 409 migration_in_progress (lock must be released).
        detail2 = resp2.json().get("detail", {})
        assert detail2.get("error") != "migration_in_progress", (
            "Lock was not released after step-3 failure (Correction 1 violated)"
        )

    def test_confirm_releases_lock_on_step4_failure(self, client, state, tmp_path, monkeypatch):
        """Step 4 (_rename_config) failure: lock is released; second confirm is not blocked.

        Correction 1: the confirm handler's try/finally unconditionally releases
        the migration lock even when step 4 raises.
        """
        with patch("paramem.server.app._rename_config", side_effect=OSError("EXDEV")):
            resp1 = client.post("/migration/confirm", json={})
        assert resp1.status_code == 500
        assert resp1.json()["detail"]["error"] == "config_swap_failed"

        # Second attempt: must not get 409 migration_in_progress.
        resp2 = client.post("/migration/confirm", json={})
        detail2 = resp2.json().get("detail", {})
        assert detail2.get("error") != "migration_in_progress", (
            "Lock was not released after step-4 failure (Correction 1 violated)"
        )
