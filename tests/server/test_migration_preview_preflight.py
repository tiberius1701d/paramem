"""Integration tests for /migration/preview disk-pressure pre-flight gate (Slice 6b).

Tests verify:
- Under-cap: preview succeeds → state=STAGING, pre_flight_fail=None.
- Over-cap: 200 with pre_flight_fail="disk_pressure", state="LIVE", diff still populated.
- Over-cap does not enter STAGING (second call still works → proves LIVE).
- End-to-end chain: preview pre-flight-fail → /status emits migration_pre_flight_fail item.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

import paramem.server.app as app_module
from paramem.server.config import (
    PathsConfig,
    SecurityConfig,
    ServerBackupsConfig,
    ServerConfig,
)
from paramem.server.migration import initial_migration_state

# ---------------------------------------------------------------------------
# Sample YAML content
# ---------------------------------------------------------------------------

_LIVE_YAML = (
    b"model: mistral\ndebug: false\nadapters:\n"
    b"  episodic:\n    enabled: true\n    rank: 8\n    alpha: 16\n"
)
_CAND_YAML = (
    b"model: mistral\ndebug: true\nadapters:\n"
    b"  episodic:\n    enabled: true\n    rank: 8\n    alpha: 16\n"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(tmp_path: Path, max_total_disk_gb: float = 20.0) -> ServerConfig:
    """Build a minimal real ServerConfig pointing at tmp_path."""
    config = ServerConfig.__new__(ServerConfig)
    config.paths = PathsConfig(
        data=tmp_path / "ha",
        sessions=tmp_path / "ha" / "sessions",
        debug=tmp_path / "ha" / "debug",
    )
    config.security = SecurityConfig(
        backups=ServerBackupsConfig(
            schedule="daily 04:00",
            artifacts=["config", "graph", "registry"],
            max_total_disk_gb=max_total_disk_gb,
        )
    )
    # Required by adapter_dir property.
    config.paths.data.mkdir(parents=True, exist_ok=True)
    return config


def _make_state(tmp_path: Path, config: ServerConfig | None = None) -> dict:
    """Build a minimal fresh _state dict for endpoint tests."""
    live_yaml = tmp_path / "server.yaml"
    live_yaml.write_bytes(_LIVE_YAML)

    if config is None:
        mock_config = MagicMock()
        mock_config.adapter_dir = tmp_path / "adapters"
        mock_config.adapter_dir.mkdir(parents=True, exist_ok=True)
        mock_config.paths.data = tmp_path / "data"
        mock_config.paths.data.mkdir(parents=True, exist_ok=True)
        used_config = mock_config
    else:
        used_config = config

    return {
        "model": None,
        "config": used_config,
        "config_path": str(live_yaml),
        "consolidating": False,
        "migration": initial_migration_state(),
        "server_started_at": "2026-04-22T00:00:00+00:00",
        "consolidation_loop": None,
    }


def _write_candidate(tmp_path: Path, content: bytes = _CAND_YAML) -> Path:
    p = tmp_path / "candidate.yaml"
    p.write_bytes(content)
    return p


def _seed_backups_over_cap(backups_root: Path, size_bytes: int = 200_000) -> None:
    """Seed a backup slot large enough to exceed the cap."""
    slot = backups_root / "config" / "20260421-040000"
    slot.mkdir(parents=True)
    (slot / "config-20260421-040000.bin").write_bytes(b"x" * size_bytes)


# ---------------------------------------------------------------------------
# Test 6 — under-cap → preview succeeds, state=STAGING, pre_flight_fail=None
# ---------------------------------------------------------------------------


class TestPreviewUnderCapStagesCandidate:
    def test_preview_under_cap_stages_candidate(self, tmp_path: Path, monkeypatch) -> None:
        """Under-cap: preview succeeds → state=STAGING, pre_flight_fail=None."""
        config = _make_config(tmp_path, max_total_disk_gb=20.0)
        # Ensure backups root is empty.
        backups_root = config.paths.data / "backups"
        backups_root.mkdir(parents=True, exist_ok=True)

        state = _make_state(tmp_path, config=config)
        monkeypatch.setattr(app_module, "_state", state)

        client = TestClient(app_module.app, raise_server_exceptions=False)
        cand = _write_candidate(tmp_path)
        resp = client.post("/migration/preview", json={"candidate_path": str(cand)})

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["state"] == "STAGING"
        assert body["pre_flight_fail"] is None
        assert body.get("pre_flight_disk_used_gb") is None  # not set when passing
        assert body.get("pre_flight_disk_cap_gb") is None


# ---------------------------------------------------------------------------
# Test 7 — over-cap → 200, pre_flight_fail="disk_pressure", state="LIVE"
# ---------------------------------------------------------------------------


class TestPreviewOverCapReturnsPreFlightFail:
    def test_preview_over_cap_returns_preflight_fail(self, tmp_path: Path, monkeypatch) -> None:
        """Over-cap: 200 with pre_flight_fail='disk_pressure', state still LIVE."""
        cap_gb = 0.0001  # 100 KB — very tight
        config = _make_config(tmp_path, max_total_disk_gb=cap_gb)
        backups_root = config.paths.data / "backups"
        backups_root.mkdir(parents=True, exist_ok=True)
        # Seed enough data to exceed the cap.
        _seed_backups_over_cap(backups_root, size_bytes=200_000)

        state = _make_state(tmp_path, config=config)
        monkeypatch.setattr(app_module, "_state", state)

        client = TestClient(app_module.app, raise_server_exceptions=False)
        cand = _write_candidate(tmp_path)
        resp = client.post("/migration/preview", json={"candidate_path": str(cand)})

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["pre_flight_fail"] == "disk_pressure"
        assert body["state"] == "LIVE"
        assert body["pre_flight_disk_used_gb"] > 0
        assert body["pre_flight_disk_cap_gb"] > 0
        # Diff still populated (preview was computed before the gate).
        assert body.get("unified_diff") is not None
        assert body.get("tier_diff") is not None

    def test_preview_over_cap_does_not_enter_staging_state(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Over-cap: _state['migration']['state'] stays 'LIVE', never 'STAGING'."""
        cap_gb = 0.0001
        config = _make_config(tmp_path, max_total_disk_gb=cap_gb)
        backups_root = config.paths.data / "backups"
        backups_root.mkdir(parents=True, exist_ok=True)
        _seed_backups_over_cap(backups_root, size_bytes=200_000)

        state = _make_state(tmp_path, config=config)
        monkeypatch.setattr(app_module, "_state", state)

        client = TestClient(app_module.app, raise_server_exceptions=False)
        cand = _write_candidate(tmp_path)
        client.post("/migration/preview", json={"candidate_path": str(cand)})

        # Direct inspection of live _state dict (A.7).
        assert state["migration"]["state"] == "LIVE"


# ---------------------------------------------------------------------------
# Test 8 — second /preview call after pre-flight-fail still returns LIVE (not 409)
# ---------------------------------------------------------------------------


class TestPreviewOverCapDoesNotEnterStaging:
    def test_second_preview_after_preflight_fail_works(self, tmp_path: Path, monkeypatch) -> None:
        """Second /migration/preview call after a pre-flight-fail succeeds (proves LIVE)."""
        cap_gb = 0.0001
        config = _make_config(tmp_path, max_total_disk_gb=cap_gb)
        backups_root = config.paths.data / "backups"
        backups_root.mkdir(parents=True, exist_ok=True)
        _seed_backups_over_cap(backups_root, size_bytes=200_000)

        state = _make_state(tmp_path, config=config)
        monkeypatch.setattr(app_module, "_state", state)

        client = TestClient(app_module.app, raise_server_exceptions=False)
        cand = _write_candidate(tmp_path)

        # First call.
        resp1 = client.post("/migration/preview", json={"candidate_path": str(cand)})
        assert resp1.status_code == 200
        assert resp1.json()["pre_flight_fail"] == "disk_pressure"

        # Second call — must also be 200 (not 409 already_staging).
        resp2 = client.post("/migration/preview", json={"candidate_path": str(cand)})
        assert resp2.status_code == 200, f"Second call got {resp2.status_code}: {resp2.text}"
        assert resp2.json()["pre_flight_fail"] == "disk_pressure"
        assert resp2.json()["state"] == "LIVE"


# ---------------------------------------------------------------------------
# Helpers for the end-to-end /status chain test
# ---------------------------------------------------------------------------


def _make_status_compatible_config(tmp_path: Path, max_total_disk_gb: float) -> MagicMock:
    """Build a MagicMock config that satisfies both /migration/preview and /status.

    Uses MagicMock for all attributes that /status enumerates (adapters, consolidation,
    registry_path, adapter_dir, etc.) but sets a *real* ServerBackupsConfig so
    compute_pre_flight_check activates its disk-pressure check.
    """
    cfg = MagicMock()

    # Paths — real so filesystem operations work.
    data_dir = tmp_path / "ha"
    data_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.data = data_dir

    # Registry and adapter dir — real files so /status doesn't error on open().
    registry = tmp_path / "registry.json"
    registry.write_text("{}")
    cfg.registry_path = registry

    adapter_dir = tmp_path / "adapters"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "indexed_key_registry.json").write_text("{}")
    cfg.adapter_dir = adapter_dir

    # String fields validated by StatusResponse (Pydantic rejects MagicMock for str fields).
    cfg.model_name = "mistral"
    cfg.model_config.model_id = "mistralai/Mistral-7B-Instruct-v0.3"

    # Adapter inventory fields consumed by /status.
    cfg.adapters.episodic.enabled = False
    cfg.adapters.semantic.enabled = False
    cfg.adapters.procedural.enabled = False
    cfg.consolidation.max_interim_count = 0
    cfg.consolidation.refresh_cadence = ""
    cfg.consolidation.consolidation_period_string = ""
    cfg.consolidation.mode = "train"
    cfg.consolidation.quiet_hours_mode = "always_off"
    cfg.consolidation.quiet_hours_start = "00:00"
    cfg.consolidation.quiet_hours_end = "00:00"
    cfg.consolidation.training_temp_limit = 0

    # Real security.backups so compute_pre_flight_check triggers.
    cfg.security.backups = ServerBackupsConfig(
        schedule="daily 04:00",
        artifacts=["config", "graph", "registry"],
        max_total_disk_gb=max_total_disk_gb,
    )

    return cfg


def _make_full_state(tmp_path: Path, config: MagicMock) -> dict:
    """Build a complete _state dict compatible with both /migration/preview and /status."""
    live_yaml = tmp_path / "server.yaml"
    live_yaml.write_bytes(_LIVE_YAML)

    buf = MagicMock()
    buf.pending_count = 0
    buf.get_summary.return_value = {
        "total": 0,
        "orphaned": 0,
        "oldest_age_seconds": None,
        "per_speaker": {},
    }

    return {
        "model": None,
        "tokenizer": None,
        "config": config,
        "config_path": str(live_yaml),
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
        "server_started_at": "2026-04-22T00:00:00+00:00",
        "config_drift": {
            "detected": False,
            "loaded_hash": "a1b2c3d4e5f6a7b8",
            "disk_hash": "a1b2c3d4e5f6a7b8",
            "last_checked_at": "2026-04-22T00:00:00+00:00",
        },
        "adapter_manifest_status": {},
        "last_consolidation_result": None,
    }


# ---------------------------------------------------------------------------
# Test 9 — end-to-end: preview pre-flight-fail → /status emits attention item
# ---------------------------------------------------------------------------


class TestPreFlightFailSurfacesInStatusAttentionItems:
    def test_preflight_fail_surfaces_in_status_attention_items(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """End-to-end chain: preview disk_pressure fail → /status has migration_pre_flight_fail.

        Steps (Acceptance A.6 step 5):
        1. Configure a ServerConfig whose max_total_disk_gb is tiny (100 KB).
        2. Seed a backup slot (200 KB) so current usage exceeds the cap.
        3. POST /migration/preview → assert pre_flight_fail=="disk_pressure", state=="LIVE".
        4. GET /status → assert attention.items contains kind=="migration_pre_flight_fail".
        """
        cap_gb = 0.0001  # 100 KB cap
        config = _make_status_compatible_config(tmp_path, max_total_disk_gb=cap_gb)

        # Seed backups root so current usage exceeds cap.
        backups_root = config.paths.data / "backups"
        backups_root.mkdir(parents=True, exist_ok=True)
        _seed_backups_over_cap(backups_root, size_bytes=200_000)

        state = _make_full_state(tmp_path, config)
        monkeypatch.setattr(app_module, "_state", state)

        client = TestClient(app_module.app, raise_server_exceptions=False)
        cand = _write_candidate(tmp_path)

        # Step 3: POST /migration/preview — must return disk_pressure, state stays LIVE.
        resp_preview = client.post("/migration/preview", json={"candidate_path": str(cand)})
        assert resp_preview.status_code == 200, resp_preview.text
        preview_body = resp_preview.json()
        assert preview_body["pre_flight_fail"] == "disk_pressure"
        assert preview_body["state"] == "LIVE"

        # Step 4: GET /status — attention.items must contain migration_pre_flight_fail.
        resp_status = client.get("/status")
        assert resp_status.status_code == 200, resp_status.text
        status_body = resp_status.json()
        attention_kinds = [it["kind"] for it in status_body["attention"]["items"]]
        assert "migration_pre_flight_fail" in attention_kinds, (
            f"Expected migration_pre_flight_fail in attention items; got: {attention_kinds}"
        )
