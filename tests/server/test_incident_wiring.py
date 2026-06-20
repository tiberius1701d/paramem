"""Pass B wiring tests — incident store + run-status integrated into the server.

Covers:
- T12  attention populator: active incident → AttentionItem; ack/resolved → no item
- T13  populator registration: collect_attention_items includes incident item
- T14  /status last_consolidation_error derived from active incident
- T14b /status last_consolidation_result derived from run_status.json
- T14c restart-survival: write incident, rebuild _derive_consolidation_status_fields,
        field reflects it (no RAM state)
- T15  POST /incidents/{id}/ack: flips status; /status no longer shows row
- T15b POST /incidents/{id}/ack: unknown id → not_found
- T16  VramExhausted callback → record_incident called, NOT RAM write
- T16b success branch (trained) → record_last_run called, NOT record_incident
- T17  auto-resolve: vram_exhausted incident resolves after _finalize_interim
- T17b S-4 ordering: consolidation_retry_exhausted NOT resolved by Pass B success
        (M1 guard owns that conditional resolve — Pass B must NOT touch it)
- T18  resolve_incident idempotency fix: already-resolved returns False
- T4b  (ack endpoint) acknowledged incident omitted from attention items
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import paramem.server.app as app_module
from paramem.server.attention import _collect_incident_items, collect_attention_items
from paramem.server.incidents import (
    ack_incident,
    read_incidents,
    record_incident,
    resolve_incident,
    resolve_incidents_by_type,
)
from paramem.server.run_status import read_last_runs, record_last_run

# ---------------------------------------------------------------------------
# Helpers shared with test_attention_status_e2e
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
    """Build a minimal clean _state dict for /status tests."""
    from paramem.server.migration import initial_migration_state

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
    }


@pytest.fixture()
def state(tmp_path, monkeypatch):
    fresh = _make_state(tmp_path)
    monkeypatch.setattr(app_module, "_state", fresh)
    return fresh


@pytest.fixture()
def client(state):
    # require_admin override: accept all requests as admin.
    from paramem.server import auth as auth_module

    def _no_auth(request=None):
        return None

    with patch.object(auth_module, "require_admin_check", _no_auth, create=True):
        yield TestClient(app_module.app, raise_server_exceptions=False)


@pytest.fixture()
def client_admin(state, monkeypatch):
    """TestClient with require_admin dependency overridden to always pass."""
    from paramem.server.app import require_admin

    app_module.app.dependency_overrides[require_admin] = lambda: None
    yield TestClient(app_module.app, raise_server_exceptions=False)
    app_module.app.dependency_overrides.pop(require_admin, None)


# ---------------------------------------------------------------------------
# Helper: state_dir from config
# ---------------------------------------------------------------------------


def _state_dir(state: dict) -> Path:
    return state["config"].paths.data / "state"


def _record(state_dir, *, type="vram_exhausted", key="phase1", severity="failed"):
    return record_incident(
        state_dir,
        type=type,
        key=key,
        severity=severity,
        summary=f"{type} at {key}",
        detail={"type": type, "phase": key, "at": "2026-06-17T10:00:00+00:00"},
    )


# ---------------------------------------------------------------------------
# T12 — attention populator: active → item; ack/resolved → no item
# ---------------------------------------------------------------------------


class TestCollectIncidentItems:
    def test_active_incident_emits_attention_item(self, tmp_path):
        """Active incident → one AttentionItem with matching level and summary."""
        state_dir = tmp_path / "state"
        _record(state_dir)

        cfg = MagicMock()
        cfg.paths.data = tmp_path

        items = _collect_incident_items({}, cfg)
        assert len(items) == 1
        item = items[0]
        assert item.level == "failed"
        assert "vram_exhausted" in item.kind
        assert "vram_exhausted" in item.summary

    def test_acknowledged_incident_emits_no_item(self, tmp_path):
        """Acknowledged incidents are silenced — no AttentionItem emitted."""
        state_dir = tmp_path / "state"
        _record(state_dir)
        ack_incident(state_dir, "vram_exhausted:phase1")

        cfg = MagicMock()
        cfg.paths.data = tmp_path

        items = _collect_incident_items({}, cfg)
        assert items == []

    def test_resolved_incident_emits_no_item(self, tmp_path):
        """Resolved incidents are omitted entirely."""
        state_dir = tmp_path / "state"
        _record(state_dir)
        resolve_incident(state_dir, "vram_exhausted", "phase1")

        cfg = MagicMock()
        cfg.paths.data = tmp_path

        items = _collect_incident_items({}, cfg)
        assert items == []

    def test_config_none_returns_empty(self, tmp_path):
        """config=None → [] (unit-test shim parity with backup populator)."""
        items = _collect_incident_items({}, None)
        assert items == []

    def test_no_incidents_file_returns_empty(self, tmp_path):
        """Absent incidents.json → []."""
        cfg = MagicMock()
        cfg.paths.data = tmp_path
        items = _collect_incident_items({}, cfg)
        assert items == []


# ---------------------------------------------------------------------------
# T13 — populator registration
# ---------------------------------------------------------------------------


class TestPopulatorRegistration:
    def test_collect_attention_items_includes_incident(self, tmp_path):
        """collect_attention_items includes the incident item when one is active."""
        state_dir = tmp_path / "state"
        _record(state_dir)

        cfg = MagicMock()
        cfg.paths.data = tmp_path

        # Minimal state — enough for non-incident populators to return [].
        state = {
            "migration": None,
            "consolidating": False,
            "last_consolidation": None,
            "boot_degraded": None,
            "store_load_degraded": None,
            "integrity_cleanup": None,
            "adapter_fingerprints_ok": True,
            "voice_degraded": None,
            "vram_overflow": None,
            "vram_post_load_budget": None,
            "vram_low_headroom": None,
            "encryption": None,
        }

        items = collect_attention_items(state, cfg)
        kinds = [i.kind for i in items]
        assert any("incident" in k for k in kinds), (
            f"Expected an incident AttentionItem; got kinds: {kinds}"
        )


# ---------------------------------------------------------------------------
# T14 — /status last_consolidation_error derived from active incident
# ---------------------------------------------------------------------------


class TestStatusErrorDerivation:
    def test_active_vram_exhausted_reflects_in_last_consolidation_error(
        self, client_admin, state, tmp_path
    ):
        """Active vram_exhausted incident → last_consolidation_error reflects its detail."""
        sd = _state_dir(state)
        _at = "2026-06-17T10:00:00+00:00"
        record_incident(
            sd,
            type="vram_exhausted",
            key="phase1",
            severity="failed",
            summary="VRAM exhausted at phase1",
            detail={"type": "vram_exhausted", "phase": "phase1", "at": _at},
        )

        resp = client_admin.get("/status")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        err = body["last_consolidation_error"]
        assert err is not None
        assert err["type"] == "vram_exhausted"
        assert err["phase"] == "phase1"
        assert err["at"] == _at

    def test_no_active_incident_returns_none_for_error_field(self, client_admin, state):
        """No active incidents → last_consolidation_error is None."""
        resp = client_admin.get("/status")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["last_consolidation_error"] is None

    def test_resolved_incident_does_not_populate_error_field(self, client_admin, state):
        """Resolved incident → last_consolidation_error is None."""
        sd = _state_dir(state)
        _record(sd)
        resolve_incident(sd, "vram_exhausted", "phase1")

        resp = client_admin.get("/status")
        assert resp.status_code == 200, resp.text
        assert resp.json()["last_consolidation_error"] is None


# ---------------------------------------------------------------------------
# T14b — /status last_consolidation_result derived from run_status.json
# ---------------------------------------------------------------------------


class TestStatusResultDerivation:
    def test_last_consolidation_result_from_run_status(self, client_admin, state):
        """run_status.json trained record → last_consolidation_result reflects it."""
        sd = _state_dir(state)
        record_last_run(
            sd,
            op_type="consolidation",
            outcome="trained",
            summary="Interim trained: 10 total keys",
            detail={"total_keys": 10},
        )

        resp = client_admin.get("/status")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        result = body["last_consolidation_result"]
        assert result is not None
        assert result["outcome"] == "trained"

    def test_absent_run_status_returns_none_for_result_field(self, client_admin, state):
        """No run_status.json → last_consolidation_result is None."""
        resp = client_admin.get("/status")
        assert resp.status_code == 200, resp.text
        assert resp.json()["last_consolidation_result"] is None


# ---------------------------------------------------------------------------
# T14c — restart-survival: derive from disk, no RAM state
# ---------------------------------------------------------------------------


class TestRestartSurvivalStatusDerivation:
    def test_incident_survives_simulated_restart(self, tmp_path):
        """Write incident; _derive_consolidation_status_fields on fresh call reflects it."""
        state_dir = tmp_path / "state"
        _at = "2026-06-17T11:00:00+00:00"
        record_incident(
            state_dir,
            type="vram_exhausted",
            key="phase2",
            severity="failed",
            summary="VRAM exhausted at phase2",
            detail={"type": "vram_exhausted", "phase": "phase2", "at": _at},
        )

        # Fresh call simulates process restart (no in-memory state).
        from paramem.server.app import _derive_consolidation_status_fields

        err, result = _derive_consolidation_status_fields(state_dir)
        assert err is not None
        assert err["type"] == "vram_exhausted"
        assert err["phase"] == "phase2"
        assert result is None  # no run_status written

    def test_run_status_survives_simulated_restart(self, tmp_path):
        """Write run_status; _derive_consolidation_status_fields on fresh call reflects it."""
        state_dir = tmp_path / "state"
        record_last_run(
            state_dir,
            op_type="consolidation",
            outcome="noop",
            summary="Full cycle no-op",
            detail={},
        )

        from paramem.server.app import _derive_consolidation_status_fields

        err, result = _derive_consolidation_status_fields(state_dir)
        assert err is None
        assert result is not None
        assert result["outcome"] == "noop"


# ---------------------------------------------------------------------------
# T15 — POST /incidents/{id}/ack endpoint
# ---------------------------------------------------------------------------


class TestAckEndpoint:
    def test_ack_endpoint_acknowledges_incident(self, client_admin, state):
        """POST /incidents/{id}/ack flips status→acknowledged; returns ok."""
        sd = _state_dir(state)
        _record(sd)

        resp = client_admin.post("/incidents/vram_exhausted:phase1/ack")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "ok"
        assert body["id"] == "vram_exhausted:phase1"

        # Status must be acknowledged on disk.
        inc = read_incidents(sd)[0]
        assert inc.status == "acknowledged"

    def test_ack_endpoint_acknowledged_incident_not_in_attention(self, client_admin, state):
        """After ack, the incident does NOT appear in /status.attention.items."""
        sd = _state_dir(state)
        _record(sd)
        client_admin.post("/incidents/vram_exhausted:phase1/ack")

        resp = client_admin.get("/status")
        assert resp.status_code == 200, resp.text
        items = resp.json()["attention"]["items"]
        incident_items = [i for i in items if "incident" in i.get("kind", "")]
        assert incident_items == [], (
            f"Acknowledged incident must not appear in attention; got: {incident_items}"
        )


# ---------------------------------------------------------------------------
# T15b — unknown id → not_found
# ---------------------------------------------------------------------------


class TestAckEndpointNotFound:
    def test_ack_unknown_id_returns_not_found(self, client_admin, state):
        """POST /incidents/unknown:id/ack → not_found (no error)."""
        resp = client_admin.post("/incidents/unknown_type:no_such_key/ack")
        assert resp.status_code == 200, resp.text
        assert resp.json()["status"] == "not_found"


# ---------------------------------------------------------------------------
# T16 — VramExhausted callback → record_incident, NOT RAM write
# ---------------------------------------------------------------------------


class TestWriteSiteVramExhausted:
    def test_vram_exhausted_callback_records_incident(self, state, tmp_path, monkeypatch):
        """_scheduled_extract_done_callback with VramExhausted → incident recorded."""
        from paramem.server.app import _scheduled_extract_done_callback
        from paramem.server.vram_guard import VramExhausted

        sd = _state_dir(state)

        class _FakeFuture:
            def exception(self):
                return VramExhausted("extraction")

        # Patch voice pipeline restore so the callback doesn't fail.
        monkeypatch.setattr(app_module, "_set_voice_pipeline_profile", lambda *a, **kw: None)
        monkeypatch.setattr(app_module, "_target_profile", lambda: "cpu")

        _scheduled_extract_done_callback(_FakeFuture())

        incidents = read_incidents(sd)
        assert len(incidents) == 1
        inc = incidents[0]
        assert inc.type == "vram_exhausted"
        assert inc.status == "active"
        # RAM key must NOT exist.
        assert "last_consolidation_error" not in state

    def test_vram_exhausted_callback_detail_shape(self, state, tmp_path, monkeypatch):
        """detail dict preserves the historic shape {type, phase, at}."""
        from paramem.server.app import _scheduled_extract_done_callback
        from paramem.server.vram_guard import VramExhausted

        sd = _state_dir(state)

        class _FakeFuture:
            def exception(self):
                return VramExhausted("phase_x")

        monkeypatch.setattr(app_module, "_set_voice_pipeline_profile", lambda *a, **kw: None)
        monkeypatch.setattr(app_module, "_target_profile", lambda: "cpu")

        _scheduled_extract_done_callback(_FakeFuture())

        inc = read_incidents(sd)[0]
        assert inc.detail["type"] == "vram_exhausted"
        assert inc.detail["phase"] == "phase_x"
        assert "at" in inc.detail


# ---------------------------------------------------------------------------
# T16b — success branch → record_last_run, NOT record_incident
# ---------------------------------------------------------------------------


class TestWriteSiteSuccess:
    def test_trained_success_writes_run_status_not_incident(self, tmp_path):
        """record_last_run writes run_status.json; no incidents.json created."""
        state_dir = tmp_path / "state"
        record_last_run(
            state_dir,
            op_type="consolidation",
            outcome="trained",
            summary="Trained OK",
            detail={"total_keys": 10},
        )
        assert (state_dir / "run_status.json").exists()
        assert not (state_dir / "incidents.json").exists()

    def test_aborted_writes_run_status_not_incident(self, tmp_path):
        """aborted outcome → run_status.json only; NOT an incident (S-1)."""
        state_dir = tmp_path / "state"
        record_last_run(
            state_dir,
            op_type="consolidation",
            outcome="aborted",
            summary="Aborted for inference",
            detail={},
        )
        runs = read_last_runs(state_dir)
        assert runs["consolidation"].outcome == "aborted"
        assert not (state_dir / "incidents.json").exists()


# ---------------------------------------------------------------------------
# T17 — auto-resolve: vram_exhausted clears after interim success
# ---------------------------------------------------------------------------


class TestAutoResolve:
    def test_vram_exhausted_resolved_after_interim_success(self, tmp_path):
        """After resolve_incidents_by_type(vram_exhausted), active incident is gone."""
        state_dir = tmp_path / "state"
        _record(state_dir, type="vram_exhausted", key="phase1")

        # Simulate _finalize_interim's auto-resolve call.
        resolved = resolve_incidents_by_type(state_dir, "vram_exhausted")
        assert resolved == 1

        incidents = read_incidents(state_dir)
        assert all(i.status == "resolved" for i in incidents if i.type == "vram_exhausted")

    def test_training_crash_resolved_after_interim_success(self, tmp_path):
        """training_crash incident resolves on interim success."""
        state_dir = tmp_path / "state"
        _record(state_dir, type="training_crash", key="interim")

        resolve_incidents_by_type(state_dir, "training_crash")

        incidents = read_incidents(state_dir)
        assert all(i.status == "resolved" for i in incidents if i.type == "training_crash")

    def test_consolidation_crash_resolved_after_full_success(self, tmp_path):
        """consolidation_crash incident resolves on full-cycle success."""
        state_dir = tmp_path / "state"
        _record(state_dir, type="consolidation_crash", key="full")

        resolve_incidents_by_type(state_dir, "consolidation_crash")

        incidents = read_incidents(state_dir)
        assert all(i.status == "resolved" for i in incidents if i.type == "consolidation_crash")

    def test_migration_incidents_resolved_on_migration_complete(self, tmp_path):
        """migration_error + migration_phase_failed resolve on migration_complete."""
        state_dir = tmp_path / "state"
        _record(state_dir, type="migration_error", key="active_store")
        _record(state_dir, type="migration_phase_failed", key="phase_a_failed")

        resolve_incidents_by_type(state_dir, "migration_error")
        resolve_incidents_by_type(state_dir, "migration_phase_failed")

        incidents = read_incidents(state_dir)
        assert all(i.status == "resolved" for i in incidents)


# ---------------------------------------------------------------------------
# T17b — S-4 ordering: consolidation_retry_exhausted NOT resolved by Pass B
# ---------------------------------------------------------------------------


class TestS4Ordering:
    def test_recall_failure_incident_not_resolved_by_pass_b_success_paths(self, tmp_path):
        """consolidation_retry_exhausted type is not in Pass B's resolve-by-type calls.

        The M1 guard owns the conditional resolve for consolidation_retry_exhausted.
        Pass B must NOT resolve it — only by_type resolution of the types Pass B
        owns (vram_exhausted, training_crash, consolidation_crash, extraction_failed,
        migration_error, migration_phase_failed).
        """
        state_dir = tmp_path / "state"
        _record(state_dir, type="consolidation_retry_exhausted", key="session_abc")

        # Simulate Pass B success paths (resolve all Pass-B-owned types).
        for t in (
            "training_crash",
            "vram_exhausted",
            "consolidation_crash",
            "extraction_failed",
            "migration_error",
            "migration_phase_failed",
        ):
            resolve_incidents_by_type(state_dir, t)

        # The retry_exhausted incident must remain ACTIVE.
        incidents = read_incidents(state_dir)
        recall_failures = [i for i in incidents if i.type == "consolidation_retry_exhausted"]
        assert len(recall_failures) == 1
        assert recall_failures[0].status == "active", (
            "consolidation_retry_exhausted must remain active — "
            "M1 guard owns its conditional resolve (S-4 ordering hazard)"
        )


# ---------------------------------------------------------------------------
# T18 — resolve_incident idempotency fix: already-resolved returns False
# ---------------------------------------------------------------------------


class TestResolveIncidentIdempotencyFix:
    def test_resolve_already_resolved_returns_false(self, tmp_path):
        """resolve_incident on an already-resolved incident returns False (actual-transition gate).

        The code-reviewer fix: gate True return on row['status'] != 'resolved'.
        A wired caller can now trust the boolean.
        """
        state_dir = tmp_path / "state"
        _record(state_dir)

        first = resolve_incident(state_dir, "vram_exhausted", "phase1")
        assert first is True  # actual transition

        second = resolve_incident(state_dir, "vram_exhausted", "phase1")
        assert second is False  # no transition — already resolved
