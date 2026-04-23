"""Deterministic no-GPU E2E tests for accept + rollback lifecycle (Slice 3b.3).

Covers the full preview → confirm → [mock trial] → accept/rollback paths.
All GPU operations are patched out.  Uses the same fixture pattern as
test_migration_confirm_e2e.py.
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

import paramem.server.app as app_module
from paramem.backup.backup import write as backup_write
from paramem.backup.encryption import SecurityBackupsConfig
from paramem.backup.types import ArtifactKind
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


def _make_state(tmp_path: Path) -> dict:
    """Build a STAGING state for E2E tests."""
    live_yaml = tmp_path / "server.yaml"
    live_yaml.write_bytes(_LIVE_YAML)
    cand_yaml = tmp_path / "candidate.yaml"
    cand_yaml.write_bytes(_CAND_YAML)

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
        "config_drift": {
            "detected": False,
            "loaded_hash": _sha256(_LIVE_YAML),
            "disk_hash": _sha256(_LIVE_YAML),
            "last_checked_at": "2026-04-22T00:00:00+00:00",
        },
    }


def _seed_trial_state(state: dict, tmp_path: Path, gates_status: str = "pass") -> None:
    """Seed the in-memory state and disk marker to TRIAL with given gate status.

    Called after ``/migration/confirm`` has been POSTed in E2E tests.
    """
    config = state["config"]
    state_dir = config.paths.data / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    # Create trial adapter + graph dirs.
    trial_adapter_dir = state_dir / "trial_adapter"
    trial_adapter_dir.mkdir(exist_ok=True)
    (trial_adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    trial_graph_dir = state_dir / "trial_graph"
    trial_graph_dir.mkdir(exist_ok=True)

    # Create config A backup slot using the real backup writer so the sidecar
    # (*.meta.json) is present and backup.read() (called by rollback's decrypt
    # step) can validate and optionally decrypt the artifact.
    backups_root = config.paths.data / "backups"
    sec_cfg = SecurityBackupsConfig()  # encrypt_at_rest=AUTO (no key → plaintext)
    config_slot = backup_write(
        ArtifactKind.CONFIG,
        _LIVE_YAML,
        {"tier": "pre_migration"},
        base_dir=backups_root / "config",
        security_config=sec_cfg,
    )

    # Derive the artifact filename from the real slot.
    artifact_files = [e for e in config_slot.iterdir() if not e.name.endswith(".meta.json")]
    assert len(artifact_files) == 1, (
        f"Expected exactly 1 artifact in {config_slot}, got: "
        f"{[e.name for e in config_slot.iterdir()]}"
    )
    config_artifact_filename = artifact_files[0].name

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

    state["migration"]["state"] = "TRIAL"
    state["migration"]["trial"] = trial_stash
    state["migration"]["candidate_path"] = ""

    # Write matching disk marker.
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
        config_artifact_filename=config_artifact_filename,
    )
    write_trial_marker(state_dir, marker)


# ---------------------------------------------------------------------------
# E2E: full preview → confirm → mock-trial → accept
# ---------------------------------------------------------------------------


class TestE2EAcceptPath:
    def test_e2e_full_preview_confirm_accept(self, tmp_path, monkeypatch):
        """Full E2E: STAGING → TRIAL → accept → LIVE.

        Uses /migration/preview, /migration/confirm, seeds TRIAL, then POSTs
        /migration/accept.  Verifies the terminal state is LIVE with marker
        cleared and restart banner set.
        """
        fresh = _make_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", fresh)

        async def _noop_trial():
            pass

        monkeypatch.setattr(app_module, "_run_trial_consolidation", _noop_trial)
        client = TestClient(app_module.app, raise_server_exceptions=False)

        # 1. Preview — seeds STAGING (already set in _make_state).
        status = client.get("/migration/status").json()
        assert status["state"] == "STAGING"

        # 2. Confirm — transitions to TRIAL (via patched trial task).
        resp = client.post("/migration/confirm", json={})
        assert resp.status_code == 200
        assert fresh["migration"]["state"] == "TRIAL"

        # 3. Seed the TRIAL state with pass gates + disk artifacts.
        _seed_trial_state(fresh, tmp_path, gates_status="pass")

        # 4. Accept.
        resp = client.post("/migration/accept")
        assert resp.status_code == 200
        body = resp.json()
        assert body["state"] == "LIVE"
        assert body["restart_required"] is True

        # 5. Verify terminal state.
        assert fresh["migration"]["state"] == "LIVE"
        state_dir = fresh["config"].paths.data / "state"
        assert read_trial_marker(state_dir) is None
        banners = fresh["migration"]["recovery_required"]
        assert any("RESTART REQUIRED" in b for b in banners)

    def test_e2e_accept_status_shows_comparison_report(self, tmp_path, monkeypatch):
        """/migration/status shows comparison_report when TRIAL + pass gates."""
        fresh = _make_state(tmp_path)
        fresh["migration"]["state"] = "TRIAL"
        fresh["migration"]["trial"] = TrialStash(
            started_at="2026-04-22T01:00:00+00:00",
            pre_trial_config_sha256=_sha256(_LIVE_YAML),
            candidate_config_sha256=_sha256(_CAND_YAML),
            backup_paths={},
            trial_adapter_dir="/tmp/trial_adapter",
            trial_graph_dir="/tmp/trial_graph",
            gates={"status": "pass", "completed_at": "2026-04-22T02:00:00+00:00"},
        )
        monkeypatch.setattr(app_module, "_state", fresh)
        client = TestClient(app_module.app, raise_server_exceptions=False)
        status = client.get("/migration/status").json()
        assert status["state"] == "TRIAL"
        report = status.get("comparison_report")
        assert report is not None
        assert report["schema_version"] >= 1
        assert len(report["rows"]) >= 5
        # Forward-compat guardrail 3: operator_line verbatim.
        from paramem.server.migration_report import COMPARISON_REPORT_OPERATOR_LINE

        assert report["operator_line"] == COMPARISON_REPORT_OPERATOR_LINE


# ---------------------------------------------------------------------------
# E2E: full preview → confirm → mock-trial → rollback
# ---------------------------------------------------------------------------


class TestE2ERollbackPath:
    def test_e2e_full_preview_confirm_rollback(self, tmp_path, monkeypatch):
        """Full E2E: STAGING → TRIAL → rollback → LIVE (A restored).

        Uses /migration/preview, /migration/confirm, seeds TRIAL with fail
        gates, then POSTs /migration/rollback.  Verifies config A is restored
        and marker cleared.
        """
        fresh = _make_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", fresh)

        async def _noop_trial():
            pass

        monkeypatch.setattr(app_module, "_run_trial_consolidation", _noop_trial)
        client = TestClient(app_module.app, raise_server_exceptions=False)

        # Confirm — transitions to TRIAL.
        resp = client.post("/migration/confirm", json={})
        assert resp.status_code == 200

        # Seed TRIAL (live yaml was overwritten by confirm — write back for test).
        live_yaml = Path(fresh["config_path"])
        # After confirm, live_yaml has _CAND_YAML contents.
        _seed_trial_state(fresh, tmp_path, gates_status="fail")

        # The live config is the B config after confirm.
        live_yaml.write_bytes(_CAND_YAML)

        # Rollback.
        resp = client.post("/migration/rollback")
        assert resp.status_code == 200
        body = resp.json()
        assert body["state"] == "LIVE"
        assert body["restart_required"] is True

        # Config A is restored.
        assert live_yaml.read_bytes() == _LIVE_YAML

        # Terminal state verified.
        assert fresh["migration"]["state"] == "LIVE"
        state_dir = fresh["config"].paths.data / "state"
        assert read_trial_marker(state_dir) is None


# ---------------------------------------------------------------------------
# E2E: crash during accept recovery
# ---------------------------------------------------------------------------


class TestE2ECrashRecovery:
    def test_e2e_crash_during_accept_marker_cleared(self, tmp_path, monkeypatch):
        """After a crash mid-accept (after marker clear, before adapter move):

        - Marker is cleared → no stale marker misdirects recovery (IMPROVEMENT 7).
        - Config B remains live (accept was progressing).
        - Recovery on restart: no marker + B config → LIVE on B.

        This test simulates the crash by patching shutil.move to raise after
        the marker has been cleared.
        """
        fresh = _make_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", fresh)

        # Seed TRIAL.
        _seed_trial_state(fresh, tmp_path, gates_status="pass")

        original_move = __import__("shutil").move
        call_count = [0]

        def _fail_on_first_move(src, dst):
            call_count[0] += 1
            if call_count[0] == 1:
                raise OSError("simulated crash mid-move")
            return original_move(src, dst)

        with patch("paramem.server.app.shutil.move", _fail_on_first_move):
            client = TestClient(app_module.app, raise_server_exceptions=False)
            resp = client.post("/migration/accept")

        # Accept should still return 200 (rotation failure is non-fatal).
        assert resp.status_code == 200
        # Marker is cleared — IMPROVEMENT 7 ensures this even on move failure.
        state_dir = fresh["config"].paths.data / "state"
        assert read_trial_marker(state_dir) is None
        # State is LIVE.
        assert fresh["migration"]["state"] == "LIVE"
