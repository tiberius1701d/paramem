"""Wiring tests — incident store + run-status integrated into the server.

Covers:
- Attention populator: active incident → AttentionItem; ack/resolved → no item
- Populator registration: collect_attention_items includes incident item
- /status last_consolidation_error derived from active incident
- /status last_consolidation_result derived from run_status.json
- Restart-survival: write incident, rebuild _derive_consolidation_status_fields,
  field reflects it (no RAM state)
- POST /incidents/{id}/ack: flips status; /status no longer shows row
- POST /incidents/{id}/ack: unknown id → not_found
- VramExhausted callback → record_incident called, NOT RAM write
- Success branch (trained) → record_last_run called, NOT record_incident
- Auto-resolve: vram_exhausted incident resolves after _finalize_interim
- consolidation_retry_exhausted is NOT resolved by this module's
  incident-wiring success paths (_finalize_interim's clean-success guard
  owns that conditional resolve)
- resolve_incident idempotency fix: already-resolved returns False
- Ack endpoint: acknowledged incident omitted from attention items
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
    cfg.security.backups.max_total_disk_gb = 20.0
    cfg.paths.key_metadata = tmp_path / "data" / "registry" / "key_metadata.json"
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
        "cloud_agent": None,
        "ha_client": None,
        "consolidation_loop": None,
        "consolidating": False,
        "last_consolidation": None,
        "background_trainer": None,
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
# Attention populator: active → item; ack/resolved → no item
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
# Populator registration
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
# /status last_consolidation_error derived from active incident
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
# /status last_consolidation_result derived from run_status.json
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
# Restart-survival: derive from disk, no RAM state
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
# POST /incidents/{id}/ack endpoint
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
# Unknown id → not_found
# ---------------------------------------------------------------------------


class TestAckEndpointNotFound:
    def test_ack_unknown_id_returns_not_found(self, client_admin, state):
        """POST /incidents/unknown:id/ack → not_found (no error)."""
        resp = client_admin.post("/incidents/unknown_type:no_such_key/ack")
        assert resp.status_code == 200, resp.text
        assert resp.json()["status"] == "not_found"


# ---------------------------------------------------------------------------
# VramExhausted callback → record_incident, NOT RAM write
# ---------------------------------------------------------------------------


class TestWriteSiteVramExhausted:
    def test_vram_exhausted_callback_records_incident(self, state, tmp_path, monkeypatch):
        """_scheduled_extract_done_callback with VramExhausted → incident recorded."""
        from paramem.server.app import _scheduled_extract_done_callback
        from paramem.utils.vram_guard import VramExhausted

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
        from paramem.utils.vram_guard import VramExhausted

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
# Success branch → record_last_run, NOT record_incident
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
        """aborted outcome → run_status.json only; NOT an incident."""
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
# Auto-resolve: vram_exhausted clears after interim success
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
# Regression coverage: an exception in the interim post-cycle bookkeeping
# region (not just the run_consolidation_cycle call itself) must still
# clear _state["consolidating"] and record a training_crash incident.  Before
# the Stage-B cycle-lifecycle primitive, only the run_consolidation_cycle
# call was try/except-wrapped; a crash anywhere in the unwrapped bookkeeping
# region below it (session-buffer retry bump, incident recording, etc.)
# left the flag stuck forever.  _run_stage_b_cycle's envelope now wraps the
# ENTIRE interim body, closing that leak structurally.
# ---------------------------------------------------------------------------


class TestInterimBookkeepingRegionCrash:
    def _drive_extract_and_start_training(self, state, tmp_path, *, bump_retry_error):
        """Run _extract_and_start_training end to end with a mocked BG trainer.

        The extraction pass and run_consolidation_cycle are mocked (no GPU);
        the crash is injected into session_buffer.bump_retry_and_release,
        which lives in the interim body's post-cycle bookkeeping region —
        outside any try/except that wraps run_consolidation_cycle alone.
        """
        cfg = state["config"]
        cfg.vram.cooldown_gate_threshold_c = 0
        cfg.vram.vram_cache_headroom_gib = 0.5
        cfg.consolidation.mode = "train"
        cfg.consolidation.refresh_cadence = ""
        cfg.consolidation.interim_overflow_slack = 0

        loop = MagicMock()
        loop.model = MagicMock(name="model")
        loop.config.indexed_key_replay = True
        loop.shutdown_requested = False
        loop.extract_session.return_value = (
            [
                {
                    "subject": "Alex",
                    "predicate": "lives_in",
                    "object": "Millfield",
                    "relation_type": "factual",
                }
            ],
            [],
        )
        # Recall-failed session forces the _count_sids branch, which calls
        # session_buffer.bump_retry_and_release — the crash injection site.
        loop.run_consolidation_cycle.return_value = {
            "mode": "trained",
            "adapter_name": "episodic_interim_20260417T0000",
            "new_keys": [],
            "recall_failed_session_ids": ["sess-1"],
            "overflow_slot": False,
        }
        state["consolidation_loop"] = loop

        sb = state["session_buffer"]
        sb.pending_facts.return_value = [
            {"session_id": "sess-1", "speaker_id": "speaker0", "has_voice_embedding": False}
        ]
        sb.get_pending.return_value = [
            {
                "session_id": "sess-1",
                "speaker_id": "speaker0",
                "transcript": "hi",
                "source_type": "transcript",
                "started_at": "2026-01-01T00:00:00Z",
            }
        ]
        sb._consolidation_retry_cap = 3
        sb._sessions = {}
        sb.bump_retry_and_release.side_effect = bump_retry_error

        mock_bt = MagicMock()
        mock_bt.submit.side_effect = lambda fn, **kw: fn()

        with (
            patch("paramem.server.app.BackgroundTrainer", return_value=mock_bt),
            patch("paramem.server.app.check_vram_headroom"),
            patch("paramem.server.app.vram_scope"),
            patch("paramem.server.app._set_voice_pipeline_profile"),
            patch("paramem.server.consolidation.session_retention_dir", return_value=None),
        ):
            app_module._extract_and_start_training()

    def test_bookkeeping_crash_clears_consolidating_flag(self, state, tmp_path):
        """An uncaught exception in the bookkeeping region still clears the flag."""
        state["consolidating"] = True
        self._drive_extract_and_start_training(
            state, tmp_path, bump_retry_error=RuntimeError("disk io error")
        )
        assert state["consolidating"] is False, (
            "_state['consolidating'] must be cleared even when the interim body "
            "raises outside the run_consolidation_cycle call itself"
        )

    def test_bookkeeping_crash_records_training_crash_incident(self, state, tmp_path):
        """The crash records a training_crash/interim incident via the shared envelope."""
        state["consolidating"] = True
        self._drive_extract_and_start_training(
            state, tmp_path, bump_retry_error=RuntimeError("disk io error")
        )
        incidents = read_incidents(_state_dir(state))
        training_crashes = [i for i in incidents if i.type == "training_crash"]
        assert len(training_crashes) == 1, (
            f"expected exactly one training_crash incident; got {incidents}"
        )
        assert training_crashes[0].id == "training_crash:interim"
        assert training_crashes[0].status == "active"


# ---------------------------------------------------------------------------
# _finalize_interim's run-status detail carries relation counts for every
# outcome — one outcome label, one detail contract, so
# scripts/dev/paramem-status.sh's "simulated" render (which reads
# detail["episodic_rels"]/detail["procedural_rels"]) is populated regardless
# of which finalizer wrote the record.
# ---------------------------------------------------------------------------


class TestFinalizeInterimDetailShape:
    def _make_loop(self, state):
        state["router"] = MagicMock()
        loop = MagicMock()
        loop.model = MagicMock()
        loop.store.replay_enabled = True
        loop.store.all_active_keys.return_value = {"k1", "k2"}
        return loop

    def test_simulated_outcome_carries_rel_counts(self, state):
        """A ``result["mode"] == "simulated"`` cycle's run record carries
        both relation counts — the exact shape
        ``scripts/dev/paramem-status.sh``'s simulated branch renders."""
        loop = self._make_loop(state)
        result = {"mode": "simulated", "adapter_name": "episodic_interim_x"}

        app_module._finalize_interim(
            loop,
            result,
            session_ids=["sess-1"],
            released_sids=[],
            episodic_rels=3,
            procedural_rels=2,
        )

        record = read_last_runs(_state_dir(state))["consolidation"]
        assert record.outcome == "simulated"
        assert record.detail["episodic_rels"] == 3
        assert record.detail["procedural_rels"] == 2

    def test_trained_outcome_carries_rel_counts_and_existing_keys_unchanged(self, state):
        """The ``trained`` outcome's existing detail keys (sessions,
        total_keys, adapter) are unchanged by adding the new counts."""
        loop = self._make_loop(state)
        result = {"mode": "trained", "adapter_name": "episodic_interim_x"}

        app_module._finalize_interim(
            loop,
            result,
            session_ids=["sess-1"],
            released_sids=[],
            episodic_rels=1,
            procedural_rels=0,
        )

        record = read_last_runs(_state_dir(state))["consolidation"]
        assert record.outcome == "trained"
        assert record.detail["episodic_rels"] == 1
        assert record.detail["procedural_rels"] == 0
        assert record.detail["sessions"] == 1
        assert record.detail["total_keys"] == 2
        assert record.detail["adapter"] == "episodic_interim_x"

    def test_rel_counts_default_to_zero_when_omitted(self, state):
        """Callers that do not pass the new kwargs still get a valid,
        zero-filled detail — no KeyError on the render side."""
        loop = self._make_loop(state)
        result = {"mode": "trained", "adapter_name": "episodic_interim_x"}

        app_module._finalize_interim(
            loop,
            result,
            session_ids=["sess-1"],
            released_sids=[],
        )

        record = read_last_runs(_state_dir(state))["consolidation"]
        assert record.detail["episodic_rels"] == 0
        assert record.detail["procedural_rels"] == 0


# ---------------------------------------------------------------------------
# consolidation_retry_exhausted NOT resolved by the incident-wiring success paths
# ---------------------------------------------------------------------------


class TestS4Ordering:
    def test_recall_failure_incident_not_resolved_by_pass_b_success_paths(self, tmp_path):
        """consolidation_retry_exhausted is not resolved by this module's
        incident-wiring resolve-by-type calls.

        _finalize_interim's clean-success guard (app._is_interim_clean_success)
        owns the conditional resolve for consolidation_retry_exhausted.
        These success paths must NOT resolve it — only by_type resolution of
        the types owned here (vram_exhausted, training_crash,
        consolidation_crash, extraction_failed, migration_error,
        migration_phase_failed).
        """
        state_dir = tmp_path / "state"
        _record(state_dir, type="consolidation_retry_exhausted", key="session_abc")

        # Simulate the incident-wiring success paths (resolve all types owned here).
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
            "_finalize_interim's clean-success guard owns its conditional resolve — "
            "Pass B must not touch it"
        )


# ---------------------------------------------------------------------------
# resolve_incident idempotency fix: already-resolved returns False
# ---------------------------------------------------------------------------


class TestResolveIncidentIdempotencyFix:
    def test_resolve_already_resolved_returns_false(self, tmp_path):
        """resolve_incident on an already-resolved incident returns False (actual-transition gate).

        Gates the True return on row['status'] != 'resolved'.
        A wired caller can now trust the boolean.
        """
        state_dir = tmp_path / "state"
        _record(state_dir)

        first = resolve_incident(state_dir, "vram_exhausted", "phase1")
        assert first is True  # actual transition

        second = resolve_incident(state_dir, "vram_exhausted", "phase1")
        assert second is False  # no transition — already resolved


# ---------------------------------------------------------------------------
# Intent classifier fail-loud: load-time incident when the embeddings
# residual (encoder/exemplars) fails to load — see
# ``app_module._report_intent_classifier_health``.
# ---------------------------------------------------------------------------


class TestIntentClassifierUnavailableIncident:
    def _cfg(self, tmp_path: Path, *, enabled: bool = True, mode: str = "embeddings") -> MagicMock:
        cfg = MagicMock()
        cfg.intent.enabled = enabled
        cfg.intent.mode = mode
        cfg.paths.data = tmp_path
        return cfg

    def test_recorded_when_encoder_missing(self, tmp_path):
        """encoder=None, mode=embeddings, enabled=True → incident recorded."""
        cfg = self._cfg(tmp_path)

        app_module._report_intent_classifier_health(cfg, None, None)

        incidents = read_incidents(tmp_path / "state")
        assert len(incidents) == 1
        inc = incidents[0]
        assert inc.type == "intent_classifier_unavailable"
        assert inc.id == "intent_classifier_unavailable:embeddings"
        assert inc.severity == "warning"
        assert inc.detail["mode"] == "embeddings"
        assert set(inc.detail["missing"]) == {"encoder", "exemplars"}

    def test_recorded_when_exemplars_missing_but_encoder_loaded(self, tmp_path):
        """encoder loaded, exemplars=None → incident recorded, missing=[\"exemplars\"]."""
        cfg = self._cfg(tmp_path)
        fake_encoder = MagicMock()

        app_module._report_intent_classifier_health(cfg, fake_encoder, None)

        incidents = read_incidents(tmp_path / "state")
        assert len(incidents) == 1
        assert incidents[0].detail["missing"] == ["exemplars"]

    def test_not_recorded_when_both_loaded(self, tmp_path):
        """Both encoder and exemplar bank present → no incident."""
        cfg = self._cfg(tmp_path)

        app_module._report_intent_classifier_health(cfg, MagicMock(), MagicMock())

        assert not (tmp_path / "state" / "incidents.json").exists()

    def test_not_recorded_when_mode_is_llm(self, tmp_path):
        """mode=llm has its own encoder-residual fallback — no incident on this path."""
        cfg = self._cfg(tmp_path, mode="llm")

        app_module._report_intent_classifier_health(cfg, None, None)

        assert not (tmp_path / "state" / "incidents.json").exists()

    def test_not_recorded_when_intent_disabled(self, tmp_path):
        """intent.enabled=False → classifier is intentionally off, not degraded."""
        cfg = self._cfg(tmp_path, enabled=False)

        app_module._report_intent_classifier_health(cfg, None, None)

        assert not (tmp_path / "state" / "incidents.json").exists()

    def test_clean_load_resolves_a_standing_incident(self, tmp_path):
        """A repaired classifier clears its own incident.

        The record site is the only place that observes the load outcome, so
        it owns both halves; without this the warning rides on GET /status
        forever after the encoder is restored.
        """
        cfg = self._cfg(tmp_path)
        app_module._report_intent_classifier_health(cfg, None, None)
        assert read_incidents(tmp_path / "state")[0].status == "active"

        app_module._report_intent_classifier_health(cfg, MagicMock(), MagicMock())

        assert read_incidents(tmp_path / "state")[0].status == "resolved"


class TestEveryRecordedTypeHasAClearSite:
    """Every incident type recorded anywhere must also be resolved somewhere.

    The lifecycle contract (``incidents`` module docstring) is that an incident
    auto-resolves on the next success of the op it describes, and it is the
    caller that has to honour it.  Nothing in the module can enforce that, so
    this scan does: a type recorded with no clear site anywhere leaves a warning
    riding on ``GET /status`` forever, however healthy the system becomes.
    """

    #: Types recorded through the crash-envelope helpers, which pass the type
    #: as a ``kind=``/``_inc_type`` variable rather than a literal.  Grep cannot
    #: see them at the record site, so they are named here; their clear sites
    #: are asserted exactly like the literal ones.
    INDIRECTLY_RECORDED = frozenset(
        {
            "training_crash",
            "consolidation_crash",
            "migration_error",
            "interim_cap_reached",
            "interim_overflow_pending",
        }
    )

    @staticmethod
    def _sources() -> str:
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[2] / "paramem"
        return "\n".join(p.read_text() for p in sorted(root.rglob("*.py")))

    def test_no_type_is_recorded_without_a_clear_site(self):
        import re

        sources = self._sources()
        recorded = set(re.findall(r'record_incident\(\s*[^)]*?type="([a-z_]+)"', sources))
        recorded |= self.INDIRECTLY_RECORDED
        assert recorded, "scan found no record_incident call sites — the pattern has drifted"

        cleared = set(re.findall(r'resolve_incidents_by_type\(\s*[^,]+,\s*"([a-z_]+)"', sources))
        cleared |= set(re.findall(r'resolve_incident\(\s*[^,]+,\s*"([a-z_]+)"', sources))
        cleared |= set(re.findall(r'resolve_incident\(\s*\n\s*[^,]+,\s*\n\s*"([a-z_]+)"', sources))

        missing = sorted(recorded - cleared)
        assert not missing, (
            "these incident types are recorded but never resolved, so they stay on "
            f"GET /status forever: {missing}"
        )


class TestSameTypeDifferentKeysStaySeparate:
    """``enrichment_degraded`` can be active for two pipeline stages at once.

    The session-tier pass (per transcript, at extraction) and the graph-tier
    pass (over the merged graph, full fold only) share one incident type and
    are told apart only by key.  Both halves of that have to hold: resolving
    one must not clear the other, and ``/status`` must let a consumer tell
    which stage is degraded without parsing prose.
    """

    @staticmethod
    def _record_both(state_dir: Path) -> None:
        record_incident(
            state_dir,
            type="enrichment_degraded",
            key="cloud_enrich",
            severity="warning",
            summary="Session-tier cloud enrichment degraded",
            detail={},
        )
        record_incident(
            state_dir,
            type="enrichment_degraded",
            key="graph_enrich_vram",
            severity="warning",
            summary="Graph-tier cloud enrichment degraded",
            detail={},
        )

    def test_resolving_one_stage_leaves_the_other_active(self, tmp_path):
        """Recovery of one pass must not silence a still-degraded other pass."""
        state_dir = tmp_path / "state"
        self._record_both(state_dir)

        resolve_incident(state_dir, "enrichment_degraded", "cloud_enrich")

        by_id = {i.id: i.status for i in read_incidents(state_dir)}
        assert by_id["enrichment_degraded:cloud_enrich"] == "resolved"
        assert by_id["enrichment_degraded:graph_enrich_vram"] == "active", (
            f"the graph-tier degradation must survive a session-tier recovery — got {by_id}"
        )

    def test_status_rows_are_distinguishable_by_incident_id(self, tmp_path):
        """Both rows carry the same ``kind``; only ``incident_id`` separates them."""
        state_dir = tmp_path / "state"
        self._record_both(state_dir)

        cfg = MagicMock()
        cfg.paths.data = tmp_path
        items = [i for i in _collect_incident_items({}, cfg) if "enrichment" in i.kind]

        assert len(items) == 2
        assert {i.kind for i in items} == {"incident_enrichment_degraded"}, (
            "kind is per-type by design, so it cannot discriminate the two stages"
        )
        assert {i.incident_id for i in items} == {
            "enrichment_degraded:cloud_enrich",
            "enrichment_degraded:graph_enrich_vram",
        }
        # The field survives serialisation into the /status payload.
        assert all("incident_id" in i.to_dict() for i in items)

    def test_non_incident_rows_carry_no_incident_id(self, tmp_path):
        """``incident_id`` is None for rows that do not come from the store."""
        from paramem.server.attention import AttentionItem

        assert (
            AttentionItem(
                kind="token_ratio_drift",
                level="warning",
                summary="drift",
                action_hint=None,
                age_seconds=None,
            ).incident_id
            is None
        )
