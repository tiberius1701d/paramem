"""Deterministic no-GPU E2E tests for the accept + rollback lifecycle.

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
    config_slot = backup_write(
        ArtifactKind.CONFIG,
        _LIVE_YAML,
        {"tier": "pre_migration"},
        base_dir=backups_root / "config",
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
        backup_paths={"config": str(config_slot.resolve())},
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
        backup_paths={"config": str(config_slot.resolve())},
        trial_adapter_dir=str(trial_adapter_dir.resolve()),
        trial_graph_dir=str(trial_graph_dir.resolve()),
        config_artifact_filename=config_artifact_filename,
    )
    write_trial_marker(state_dir, marker)


# ---------------------------------------------------------------------------
# E2E: full preview → confirm → mock-trial → accept
# ---------------------------------------------------------------------------


def _stub_apply_success():
    """Return a stub _apply_config_live that reports success (applied_live=True)."""

    def _impl():
        return {
            "applied_live": True,
            "restart_required_reason": None,
            "auto_restart_scheduled": False,
            "skipped": None,
            "cloud_only_reason": None,
        }

    return _impl


def _stub_apply_failure():
    """Return a stub _apply_config_live that reports failure (applied_live=False)."""

    def _impl():
        return {
            "applied_live": False,
            "restart_required_reason": None,
            "auto_restart_scheduled": False,
            "skipped": None,
            "cloud_only_reason": "apply_failed",
        }

    return _impl


def _stub_apply_r_port():
    """Return a stub _apply_config_live reporting R-PORT carve (pre-flight passed)."""

    def _impl():
        return {
            "applied_live": False,
            "restart_required_reason": "stt_port_change",
            "auto_restart_scheduled": False,
            "restart_eligible": True,
            "skipped": None,
            "cloud_only_reason": None,
        }

    return _impl


def _stub_apply_r_paths():
    """Return a stub _apply_config_live reporting R-PATHS carve (manual restart)."""

    def _impl():
        return {
            "applied_live": False,
            "restart_required_reason": "paths_change",
            "auto_restart_scheduled": False,
            "skipped": None,
            "cloud_only_reason": None,
        }

    return _impl


def _stub_apply_reload_success_mutating():
    """Faithful stub: mimics a SUCCESSFUL _live_reload_base_model.

    The real function sets _state mode=local / cloud_only_reason=None before
    returning (app.py:3917). A dict-only stub leaves the guard sentinel resident
    and so cannot exercise the handler's restore discrimination for this path;
    this stub reproduces the real _state mutation.
    """

    def _impl():
        app_module._state["mode"] = "local"
        app_module._state["cloud_only_reason"] = None
        return {
            "applied_live": True,
            "restart_required_reason": None,
            "auto_restart_scheduled": False,
            "skipped": None,
            "cloud_only_reason": None,
        }

    return _impl


def _stub_apply_reload_failure_mutating():
    """Faithful stub: mimics a GENUINE reload failure.

    The real function sets a SPECIFIC cloud_only_reason (e.g. apply_failed at
    app.py:3896, insufficient_vram at :3836) — NOT the transient 'live_reload'
    guard sentinel — so the honest degraded state must survive the restore.
    """

    def _impl():
        app_module._state["mode"] = "cloud-only"
        app_module._state["cloud_only_reason"] = "apply_failed"
        return {
            "applied_live": False,
            "restart_required_reason": None,
            "auto_restart_scheduled": False,
            "skipped": None,
            "cloud_only_reason": "apply_failed",
        }

    return _impl


class TestE2EAcceptPath:
    def test_e2e_full_preview_confirm_accept(self, tmp_path, monkeypatch):
        """Full E2E: STAGING → TRIAL → accept → LIVE (with live-apply success stub).

        Uses /migration/preview, /migration/confirm, seeds TRIAL, then POSTs
        /migration/accept.  Verifies the terminal state is LIVE with marker
        cleared.  With a live-apply success stub, restart_required=False and
        the banner says 'applied live' (not 'RESTART REQUIRED').
        """
        fresh = _make_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", fresh)
        monkeypatch.setattr(app_module, "_apply_config_live", _stub_apply_success())

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
        # Live-apply success: applied_live=True, restart_required=False.
        assert body["applied_live"] is True
        assert body["restart_required"] is False

        # 5. Verify terminal state.
        assert fresh["migration"]["state"] == "LIVE"
        state_dir = fresh["config"].paths.data / "state"
        assert read_trial_marker(state_dir) is None
        banners = fresh["migration"]["recovery_required"]
        # With live-apply success, the banner says 'applied live', NOT 'RESTART REQUIRED'.
        assert not any("RESTART REQUIRED" in b for b in banners), (
            f"Unexpected 'RESTART REQUIRED' banner on live-apply success: {banners}"
        )
        assert any("applied live" in b.lower() for b in banners), (
            f"Expected 'applied live' banner, got: {banners}"
        )

    def test_e2e_accept_apply_failure_restart_required(self, tmp_path, monkeypatch):
        """E2E: accept with apply failure stub → restart_required=True, RESTART REQUIRED banner.

        Regression coverage: the old default (restart_required=True always) now
        only applies when the live apply fails.
        """
        fresh = _make_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", fresh)
        monkeypatch.setattr(app_module, "_apply_config_live", _stub_apply_failure())

        async def _noop_trial():
            pass

        monkeypatch.setattr(app_module, "_run_trial_consolidation", _noop_trial)
        client = TestClient(app_module.app, raise_server_exceptions=False)

        _seed_trial_state(fresh, tmp_path, gates_status="pass")
        resp = client.post("/migration/accept")
        assert resp.status_code == 200
        body = resp.json()
        assert body["applied_live"] is False
        assert body["restart_required"] is True

    def test_e2e_accept_genuine_reload_failure_stays_cloud_only(self, tmp_path, monkeypatch):
        """E2E: a GENUINE reload failure must NOT be unwound by the mode-restore.

        A real _live_reload_base_model failure sets a specific cloud_only_reason
        (apply_failed/insufficient_vram/...), the honest degraded state. The restore
        guard fires only on the untouched 'live_reload' sentinel, so it must leave a
        real failure's mode=cloud-only and reason intact.
        """
        fresh = _make_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", fresh)
        monkeypatch.setattr(app_module, "_apply_config_live", _stub_apply_reload_failure_mutating())

        async def _noop_trial():
            pass

        monkeypatch.setattr(app_module, "_run_trial_consolidation", _noop_trial)
        client = TestClient(app_module.app, raise_server_exceptions=False)

        _seed_trial_state(fresh, tmp_path, gates_status="pass")
        resp = client.post("/migration/accept")
        assert resp.status_code == 200

        # Genuine failure: stay cloud-only with the specific reason, NOT restored.
        assert fresh["mode"] == "cloud-only"
        assert fresh["cloud_only_reason"] == "apply_failed"

    def test_e2e_accept_successful_reload_leaves_local(self, tmp_path, monkeypatch):
        """E2E: a successful reload (mode already local) must be left untouched by restore."""
        fresh = _make_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", fresh)
        monkeypatch.setattr(app_module, "_apply_config_live", _stub_apply_reload_success_mutating())

        async def _noop_trial():
            pass

        monkeypatch.setattr(app_module, "_run_trial_consolidation", _noop_trial)
        client = TestClient(app_module.app, raise_server_exceptions=False)

        _seed_trial_state(fresh, tmp_path, gates_status="pass")
        resp = client.post("/migration/accept")
        assert resp.status_code == 200

        assert fresh["mode"] == "local"
        assert fresh["cloud_only_reason"] is None

    def test_e2e_accept_explicit_cloud_only_preserved(self, tmp_path, monkeypatch):
        """E2E: a pre-existing explicit cloud-only (yaml cloud_only: true) survives a
        non-reload apply — the restore returns the prior (cloud-only, 'explicit'); it
        must NOT force the server to local.
        """
        fresh = _make_state(tmp_path)
        fresh["mode"] = "cloud-only"
        fresh["cloud_only_reason"] = "explicit"
        monkeypatch.setattr(app_module, "_state", fresh)
        # R-PORT carve short-circuits without reload (dict-only stub, no _state mutation).
        monkeypatch.setattr(app_module, "_apply_config_live", _stub_apply_r_port())
        monkeypatch.setattr(app_module, "_restart_service", lambda: None)

        async def _noop_trial():
            pass

        monkeypatch.setattr(app_module, "_run_trial_consolidation", _noop_trial)
        client = TestClient(app_module.app, raise_server_exceptions=False)

        _seed_trial_state(fresh, tmp_path, gates_status="pass")
        resp = client.post("/migration/accept")
        assert resp.status_code == 200

        # Prior explicit cloud-only faithfully restored, NOT forced to local.
        assert fresh["mode"] == "cloud-only"
        assert fresh["cloud_only_reason"] == "explicit"

    def test_e2e_accept_r_paths_carve_restores_prior_mode(self, tmp_path, monkeypatch):
        """E2E: R-PATHS carve short-circuits without reload → prior mode restored
        (symmetric with the R-PORT carve case).
        """
        fresh = _make_state(tmp_path)
        assert fresh["mode"] == "normal"
        monkeypatch.setattr(app_module, "_state", fresh)
        monkeypatch.setattr(app_module, "_apply_config_live", _stub_apply_r_paths())
        monkeypatch.setattr(app_module, "_restart_service", lambda: None)

        async def _noop_trial():
            pass

        monkeypatch.setattr(app_module, "_run_trial_consolidation", _noop_trial)
        client = TestClient(app_module.app, raise_server_exceptions=False)

        _seed_trial_state(fresh, tmp_path, gates_status="pass")
        resp = client.post("/migration/accept")
        assert resp.status_code == 200

        assert fresh["mode"] == "normal", (
            f"R-PATHS carve left server in {fresh['mode']!r}; expected restored 'normal'"
        )
        assert fresh.get("cloud_only_reason") is None
        banners = fresh["migration"]["recovery_required"]
        assert any("RESTART REQUIRED" in b for b in banners), (
            f"Expected 'RESTART REQUIRED' banner on apply failure: {banners}"
        )

    def test_e2e_accept_r_port_carve_restart_eligible_fields(self, tmp_path, monkeypatch):
        """E2E: R-PORT carve → restart_eligible=True, auto_restart_scheduled=False, named reason."""
        fresh = _make_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", fresh)
        monkeypatch.setattr(app_module, "_apply_config_live", _stub_apply_r_port())
        monkeypatch.setattr(app_module, "_restart_service", lambda: None)

        async def _noop_trial():
            pass

        monkeypatch.setattr(app_module, "_run_trial_consolidation", _noop_trial)
        client = TestClient(app_module.app, raise_server_exceptions=False)

        _seed_trial_state(fresh, tmp_path, gates_status="pass")
        resp = client.post("/migration/accept")
        assert resp.status_code == 200
        body = resp.json()
        # Server signals restart_eligible (CLI prompts); it never fires the restart itself.
        assert body["restart_eligible"] is True
        assert body["auto_restart_scheduled"] is False
        assert body["restart_required_reason"] == "stt_port_change"
        assert body["restart_required"] is True

    def test_e2e_accept_r_paths_carve_no_auto_restart(self, tmp_path, monkeypatch):
        """E2E: R-PATHS carve → auto_restart_scheduled=False, DATA IS NOT MIGRATED banner."""
        fresh = _make_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", fresh)
        monkeypatch.setattr(app_module, "_apply_config_live", _stub_apply_r_paths())
        monkeypatch.setattr(app_module, "_restart_service", lambda: None)

        async def _noop_trial():
            pass

        monkeypatch.setattr(app_module, "_run_trial_consolidation", _noop_trial)
        client = TestClient(app_module.app, raise_server_exceptions=False)

        _seed_trial_state(fresh, tmp_path, gates_status="pass")
        resp = client.post("/migration/accept")
        assert resp.status_code == 200
        body = resp.json()
        assert body["auto_restart_scheduled"] is False
        assert body["restart_required_reason"] == "paths_change"
        banners = fresh["migration"]["recovery_required"]
        assert any("DATA IS NOT MIGRATED" in b for b in banners), (
            f"Expected DATA IS NOT MIGRATED warning in banners: {banners}"
        )

    def test_e2e_accept_carve_restores_prior_mode_not_stuck_cloud_only(self, tmp_path, monkeypatch):
        """E2E regression: an accept whose apply does NOT reload (R-PORT carve)
        restores the pre-guard mode instead of leaving the server stuck cloud-only.

        The accept handler sets mode=cloud-only as a maintenance guard, but only a
        successful _live_reload_base_model resets it to local. The carve
        short-circuit (and no-op skip) return with the guard untouched, so without
        the restore the live server is left degraded to cloud-only.
        """
        fresh = _make_state(tmp_path)
        assert fresh["mode"] == "normal"  # pre-guard serving mode
        monkeypatch.setattr(app_module, "_state", fresh)
        monkeypatch.setattr(app_module, "_apply_config_live", _stub_apply_r_port())
        monkeypatch.setattr(app_module, "_restart_service", lambda: None)

        async def _noop_trial():
            pass

        monkeypatch.setattr(app_module, "_run_trial_consolidation", _noop_trial)
        client = TestClient(app_module.app, raise_server_exceptions=False)

        _seed_trial_state(fresh, tmp_path, gates_status="pass")
        resp = client.post("/migration/accept")
        assert resp.status_code == 200

        # The carve short-circuited (no reload), so the guard must be unwound.
        assert fresh["mode"] == "normal", (
            f"accept left server in {fresh['mode']!r} after a non-reload apply"
        )
        assert fresh.get("cloud_only_reason") is None

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
        """Full E2E: STAGING → TRIAL → rollback → LIVE (A restored, no-op skip).

        Uses /migration/preview, /migration/confirm, seeds TRIAL with fail
        gates, then POSTs /migration/rollback.  Verifies config A is restored
        and marker cleared.

        With a no-op-skip stub (disk=A, memory=A), applied_live=True
        and restart_required=False (config was already at A; no reload needed).
        """
        fresh = _make_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", fresh)
        # Rollback restores disk to A, memory is A → no-op skip.
        monkeypatch.setattr(
            app_module,
            "_apply_config_live",
            lambda: {
                "applied_live": True,
                "restart_required_reason": None,
                "auto_restart_scheduled": False,
                "skipped": "no_change",
                "cloud_only_reason": None,
            },
        )

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
        # No-op skip: applied_live=True, restart_required=False (disk=A already matches memory=A).
        assert body["applied_live"] is True
        assert body["restart_required"] is False

        # Config A is restored.
        assert live_yaml.read_bytes() == _LIVE_YAML

        # Terminal state verified.
        assert fresh["migration"]["state"] == "LIVE"
        state_dir = fresh["config"].paths.data / "state"
        assert read_trial_marker(state_dir) is None

    def test_e2e_rollback_restores_prior_mode_not_stuck_cloud_only(self, tmp_path, monkeypatch):
        """E2E regression: rollback's no-op-skip apply restores the pre-guard mode
        instead of leaving the server stuck cloud-only.

        Rollback sets mode=cloud-only as a maintenance guard; the no-op skip (the
        normal rollback path — disk==memory==A) returns without resetting it, and
        step 8 did not restore it. Without the fix, every rollback degrades the
        live server to cloud-only.
        """
        fresh = _make_state(tmp_path)
        assert fresh["mode"] == "normal"
        monkeypatch.setattr(app_module, "_state", fresh)
        monkeypatch.setattr(
            app_module,
            "_apply_config_live",
            lambda: {
                "applied_live": True,
                "restart_required_reason": None,
                "auto_restart_scheduled": False,
                "skipped": "no_change",
                "cloud_only_reason": None,
            },
        )

        async def _noop_trial():
            pass

        monkeypatch.setattr(app_module, "_run_trial_consolidation", _noop_trial)
        client = TestClient(app_module.app, raise_server_exceptions=False)

        resp = client.post("/migration/confirm", json={})
        assert resp.status_code == 200
        live_yaml = Path(fresh["config_path"])
        _seed_trial_state(fresh, tmp_path, gates_status="fail")
        live_yaml.write_bytes(_CAND_YAML)

        resp = client.post("/migration/rollback")
        assert resp.status_code == 200

        # No-op-skip apply did not reload, so the guard must be unwound to the
        # pre-rollback serving mode rather than left at cloud-only.
        assert fresh["mode"] == "normal", (
            f"rollback left server in {fresh['mode']!r}; expected restored 'normal'"
        )
        assert fresh.get("cloud_only_reason") is None


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
