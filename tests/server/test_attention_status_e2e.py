"""Endpoint-level E2E tests for /status attention + migration blocks (Slice 5a).

Drives /status through various _state configurations, asserting
attention.items content and migration block at each transition.

Convention: TestClient without lifespan; _state monkeypatched per test.
See test_migration_endpoints.py for the canonical pattern.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import paramem.server.app as app_module
from paramem.server.migration import initial_migration_state

# ---------------------------------------------------------------------------
# State factory helpers
# ---------------------------------------------------------------------------


def _base_config(tmp_path: Path) -> MagicMock:
    """Return a minimal MagicMock ServerConfig."""
    cfg = MagicMock()
    cfg.model_name = "mistral"
    cfg.model_config.model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    cfg.registry_path = tmp_path / "registry.json"
    cfg.registry_path.write_text("{}")
    cfg.adapter_dir = tmp_path / "adapters"
    cfg.adapter_dir.mkdir(parents=True, exist_ok=True)
    (cfg.adapter_dir / "indexed_key_registry.json").write_text("{}")
    cfg.adapters.episodic.enabled = True
    cfg.adapters.episodic.rank = 8
    cfg.adapters.episodic.alpha = 16
    cfg.adapters.episodic.learning_rate = 2e-4
    cfg.adapters.episodic.target_modules = ["q_proj", "k_proj"]
    cfg.adapters.semantic.enabled = False
    cfg.adapters.procedural.enabled = False
    cfg.consolidation.refresh_cadence = ""
    cfg.consolidation.consolidation_period_string = ""
    cfg.consolidation.max_interim_count = 0
    cfg.consolidation.mode = "train"
    cfg.consolidation.quiet_hours_mode = "always_off"
    cfg.consolidation.quiet_hours_start = "00:00"
    cfg.consolidation.quiet_hours_end = "00:00"
    cfg.consolidation.training_temp_limit = 0
    cfg.paths.data = tmp_path / "data"
    cfg.paths.data.mkdir(parents=True, exist_ok=True)
    return cfg


def _make_state(tmp_path: Path) -> dict:
    """Build a minimal clean LIVE _state dict."""
    cfg = _base_config(tmp_path)
    buf = MagicMock()
    buf.get_summary.return_value = {
        "total": 0,
        "orphaned": 0,
        "oldest_age_seconds": None,
        "per_speaker": {},
    }
    buf.pending_count = 0

    return {
        "model": None,
        "tokenizer": None,
        "config": cfg,
        "config_path": str(tmp_path / "server.yaml"),
        "session_buffer": buf,
        "speaker_store": None,
        "router": None,
        "sota_agent": None,
        "ha_client": None,
        "consolidation_loop": None,
        "consolidating": False,
        "last_consolidation": None,
        "background_trainer": None,
        "post_session_queue": None,
        "mode": "local",
        "cloud_only_reason": None,
        "tts_manager": None,
        "stt": None,
        "speaker_embedding_backend": None,
        "unknown_speakers": {},
        "pending_enrollments": set(),
        "migration": initial_migration_state(),
        "server_started_at": "2026-04-22T08:00:00+00:00",
        "config_drift": {
            "detected": False,
            "loaded_hash": "a1b2c3d4e5f6a7b8",
            "disk_hash": "a1b2c3d4e5f6a7b8",
            "last_checked_at": "2026-04-22T08:00:00+00:00",
        },
        "adapter_manifest_status": {},
        "last_consolidation_result": None,
    }


@pytest.fixture()
def state(tmp_path, monkeypatch):
    fresh = _make_state(tmp_path)
    monkeypatch.setattr(app_module, "_state", fresh)
    return fresh


@pytest.fixture()
def client(state):
    return TestClient(app_module.app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_status(client) -> dict:
    resp = client.get("/status")
    assert resp.status_code == 200, resp.text
    return resp.json()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAttentionEmptyWhenLiveClean:
    def test_status_attention_empty_when_live_clean(self, client):
        """LIVE state, no drift, no recovery → attention.items == []."""
        body = _get_status(client)
        assert body["attention"]["items"] == []

    def test_status_migration_state_live(self, client):
        """LIVE state → migration.state == 'live'."""
        body = _get_status(client)
        assert body["migration"]["state"] == "live"

    def test_status_migration_comparison_none(self, client):
        """LIVE state → migration.comparison is None."""
        body = _get_status(client)
        assert body["migration"]["comparison"] is None

    def test_status_server_started_at_exposed(self, client, state):
        """server_started_at is exposed on /status (Fix 2)."""
        state["server_started_at"] = "2026-04-22T08:00:00+00:00"
        body = _get_status(client)
        assert body["server_started_at"] == "2026-04-22T08:00:00+00:00"


class TestAttentionAfterConfigDrift:
    def test_status_attention_after_drift(self, client, state):
        """Config drift detected → attention.items contains config_drift item."""
        state["config_drift"]["detected"] = True
        state["config_drift"]["disk_hash"] = "ffffffff"
        body = _get_status(client)
        kinds = [it["kind"] for it in body["attention"]["items"]]
        assert "config_drift" in kinds

    def test_status_attention_config_drift_action_required(self, client, state):
        """Config drift item has level=action_required."""
        state["config_drift"]["detected"] = True
        body = _get_status(client)
        item = next(it for it in body["attention"]["items"] if it["kind"] == "config_drift")
        assert item["level"] == "action_required"


class TestAttentionDuringTrial:
    def _seed_trial(self, state, gates_status: str = "pending", pending_count: int = 0):
        """Seed _state with a TRIAL migration at given gates status."""
        buf = MagicMock()
        buf.pending_count = pending_count
        buf.get_summary.return_value = {
            "total": pending_count,
            "orphaned": 0,
            "oldest_age_seconds": None,
            "per_speaker": {},
        }
        state["session_buffer"] = buf
        state["migration"] = {
            "state": "TRIAL",
            "recovery_required": [],
            "shape_changes": [],
            "staged_at": None,
            "candidate_path": None,
            "candidate_hash": None,
            "trial": {
                "started_at": "2026-04-22T00:00:00+00:00",
                "gates": {
                    "status": gates_status,
                    "details": [],
                    "completed_at": (
                        "2026-04-22T00:10:00+00:00" if gates_status != "pending" else None
                    ),
                },
                "trial_adapter_dir": None,
                "trial_graph_dir": None,
                "pre_trial_config_sha256": None,
                "candidate_config_sha256": None,
                "backup_paths": None,
            },
        }

    def test_status_attention_during_trial_pending(self, client, state):
        """TRIAL + gates=pending → items contains migration_trial_running."""
        self._seed_trial(state, "pending", pending_count=2)
        body = _get_status(client)
        kinds = [it["kind"] for it in body["attention"]["items"]]
        assert "migration_trial_running" in kinds

    def test_status_attention_sweeper_held_when_pending_sessions(self, client, state):
        """TRIAL + pending_count=2 → items contains sweeper_held."""
        self._seed_trial(state, "pending", pending_count=2)
        body = _get_status(client)
        kinds = [it["kind"] for it in body["attention"]["items"]]
        assert "sweeper_held" in kinds

    def test_status_migration_block_during_trial(self, client, state):
        """TRIAL state → migration.state == 'trial'."""
        self._seed_trial(state, "pending")
        body = _get_status(client)
        assert body["migration"]["state"] == "trial"

    def test_status_migration_comparison_none_when_pending(self, client, state):
        """TRIAL + gates pending → migration.comparison is None."""
        self._seed_trial(state, "pending")
        body = _get_status(client)
        assert body["migration"]["comparison"] is None

    def test_status_attention_after_trial_pass(self, client, state):
        """TRIAL + gates pass → items contains migration_trial_pass."""
        self._seed_trial(state, "pass")
        body = _get_status(client)
        kinds = [it["kind"] for it in body["attention"]["items"]]
        assert "migration_trial_pass" in kinds

    def test_status_migration_block_comparison_rendered(self, client, state):
        """TRIAL + gates pass → migration.comparison.rendered == True."""
        self._seed_trial(state, "pass")
        body = _get_status(client)
        assert body["migration"]["comparison"] is not None
        assert body["migration"]["comparison"]["rendered"] is True

    def test_status_trial_pass_level_action_required(self, client, state):
        """TRIAL pass item has level=action_required."""
        self._seed_trial(state, "pass")
        body = _get_status(client)
        item = next(it for it in body["attention"]["items"] if it["kind"] == "migration_trial_pass")
        assert item["level"] == "action_required"

    def test_status_trial_failed_level_failed(self, client, state):
        """TRIAL fail → migration_trial_failed item with level=failed."""
        self._seed_trial(state, "fail")
        body = _get_status(client)
        kinds = [it["kind"] for it in body["attention"]["items"]]
        assert "migration_trial_failed" in kinds
        item = next(
            it for it in body["attention"]["items"] if it["kind"] == "migration_trial_failed"
        )
        assert item["level"] == "failed"


class TestMigrationBlockConfigRev:
    def test_status_migration_block_config_rev(self, client, state):
        """config_drift.loaded_hash present → migration.config_rev is first 8 chars."""
        state["config_drift"]["loaded_hash"] = "a1b2c3d4e5f6a7b8"
        body = _get_status(client)
        assert body["migration"]["config_rev"] == "a1b2c3d4"

    def test_status_migration_config_rev_empty_when_no_drift(self, client, state):
        """No config_drift key → migration.config_rev is empty string."""
        state["config_drift"] = {}
        body = _get_status(client)
        assert body["migration"]["config_rev"] == ""


class TestAttentionAfterAcceptRecovery:
    def test_status_attention_after_accept_recovery_banner(self, client, state):
        """LIVE + recovery_required=['RESTART REQUIRED'] → migration_recovery_required item."""
        state["migration"]["recovery_required"] = ["RESTART REQUIRED — adapter reloaded"]
        body = _get_status(client)
        kinds = [it["kind"] for it in body["attention"]["items"]]
        assert "migration_recovery_required" in kinds

    def test_status_attention_recovery_level_info(self, client, state):
        """Recovery banner has level=info."""
        state["migration"]["recovery_required"] = ["banner"]
        body = _get_status(client)
        item = next(
            it for it in body["attention"]["items"] if it["kind"] == "migration_recovery_required"
        )
        assert item["level"] == "info"


class TestAdapterFingerprintInStatus:
    def test_status_fingerprint_primary_emits_in_items(self, client, state):
        """adapter_manifest_status primary mismatch → attention item with level=failed."""
        state["adapter_manifest_status"] = {
            "episodic": {
                "status": "mismatch",
                "reason": "sha mismatch",
                "field": "base_model.sha",
                "severity": "red",
                "slot_path": "/a",
                "checked_at": "",
            }
        }
        body = _get_status(client)
        kinds = [it["kind"] for it in body["attention"]["items"]]
        assert "adapter_fingerprint_mismatch_primary" in kinds
        item = next(
            it
            for it in body["attention"]["items"]
            if it["kind"] == "adapter_fingerprint_mismatch_primary"
        )
        assert item["level"] == "failed"
