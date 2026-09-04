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
- resolve_incident idempotency: already-resolved returns False
- Ack endpoint: acknowledged incident omitted from attention items
- _run_stage_b_cycle's crash envelope: a raised exception's own structured
  fields (BookkeepingInvariantViolation's divergent_keys,
  ActiveKeyHydrationFailure's dropped_keys/venue) merged into the recorded
  incident detail; any other exception keeps the plain detail
"""

from __future__ import annotations

import ast
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
from tests._guard_utils import enclosing_function_name, find_function, tracked_python_files

# ---------------------------------------------------------------------------
# Helpers shared with test_attention_status_e2e
# ---------------------------------------------------------------------------


def _base_config(tmp_path: Path) -> MagicMock:
    """Return a minimal MagicMock ServerConfig."""
    cfg = MagicMock()
    cfg.model_name = "mistral"
    cfg.model_config.model_id = "mistralai/Mistral-7B-Instruct-v0.3"
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
    cfg.consolidation.interim_resume = "immediate"
    cfg.consolidation.full_window = "01:00-04:00"
    cfg.consolidation.quiet_hours_mode = "always_off"
    cfg.consolidation.quiet_hours_start = "00:00"
    cfg.consolidation.quiet_hours_end = "00:00"
    cfg.consolidation.training_temp_limit = 0
    cfg.paths.data = tmp_path / "data"
    cfg.paths.data.mkdir(parents=True, exist_ok=True)
    cfg.security.backups.max_total_disk_gb = 20.0
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

    def test_config_refused_incident_surfaces_as_an_attention_row_naming_the_refusal(
        self, tmp_path
    ):
        """A ``config_refused`` incident (the shape ``_live_reload_base_model``
        records when a ``ConfigStoreMismatch`` is caught) surfaces as one
        ``incident_config_refused`` attention row naming the refusal text —
        pinning that the generic incident collector is the surfacing
        mechanism, so a dedicated populator is never added."""
        state_dir = tmp_path / "state"
        refusal_message = (
            "adapters.episodic.enabled=false but 1 interim slot(s) still exist "
            "under adapter_dir/episodic"
        )
        record_incident(
            state_dir,
            type="config_refused",
            key="interim_ring_without_episodic",
            severity="failed",
            summary=f"Config refused on reload: {refusal_message.splitlines()[0][:160]}",
            detail={"message": refusal_message, "adapter_dir": "adapter_dir"},
        )

        cfg = MagicMock()
        cfg.paths.data = tmp_path

        items = _collect_incident_items({}, cfg)
        assert len(items) == 1
        item = items[0]
        assert item.kind == "incident_config_refused"
        assert item.level == "failed"
        assert "interim slot" in item.summary


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
            "integrity_check_failed": None,
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

        err, result, calibration_result = _derive_consolidation_status_fields(state_dir)
        assert err is not None
        assert err["type"] == "vram_exhausted"
        assert err["phase"] == "phase2"
        assert result is None  # no run_status written
        assert calibration_result is None  # no calibration run_status written

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

        err, result, calibration_result = _derive_consolidation_status_fields(state_dir)
        assert err is None
        assert result is not None
        assert result["outcome"] == "noop"
        assert calibration_result is None  # no calibration run_status written


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
        """_consolidation_run_done with VramExhausted → incident recorded."""
        from paramem.server.app import _consolidation_run_done
        from paramem.server.consolidation_action import ConsolidationAction
        from paramem.utils.vram_guard import VramExhausted

        sd = _state_dir(state)

        class _FakeFuture:
            def exception(self):
                return VramExhausted("extraction")

        # Patch voice pipeline restore so the callback doesn't fail.
        monkeypatch.setattr(app_module, "_set_voice_pipeline_profile", lambda *a, **kw: None)
        monkeypatch.setattr(app_module, "_target_profile", lambda: "cpu")

        _consolidation_run_done(ConsolidationAction.INTERIM, None, _FakeFuture())

        incidents = read_incidents(sd)
        assert len(incidents) == 1
        inc = incidents[0]
        assert inc.type == "vram_exhausted"
        assert inc.status == "active"
        # RAM key must NOT exist.
        assert "last_consolidation_error" not in state

    def test_vram_exhausted_callback_detail_shape(self, state, tmp_path, monkeypatch):
        """detail dict preserves the historic shape {type, phase, at}."""
        from paramem.server.app import _consolidation_run_done
        from paramem.server.consolidation_action import ConsolidationAction
        from paramem.utils.vram_guard import VramExhausted

        sd = _state_dir(state)

        class _FakeFuture:
            def exception(self):
                return VramExhausted("phase_x")

        monkeypatch.setattr(app_module, "_set_voice_pipeline_profile", lambda *a, **kw: None)
        monkeypatch.setattr(app_module, "_target_profile", lambda: "cpu")

        _consolidation_run_done(ConsolidationAction.INTERIM, None, _FakeFuture())

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
# resolve_incident idempotency: already-resolved returns False
# ---------------------------------------------------------------------------


class TestResolveIncidentIdempotency:
    def test_resolve_already_resolved_returns_false(self, tmp_path):
        """resolve_incident on an already-resolved incident returns False (actual-transition gate).

        Gates the True return on row['status'] != 'resolved'.
        A wired caller can trust the boolean.
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


#: Incident-lifecycle call names this scan follows the ``type`` argument of —
#: matched both as a bare ``Name`` (``record_incident(...)``) and as an
#: ``ast.Attribute`` ending in one of these names (``incidents.record_incident(...)``).
_INCIDENT_CALL_NAMES = frozenset(
    {"record_incident", "resolve_incident", "resolve_incidents_by_type"}
)

#: Record sites whose ``type`` argument is a run-time variable rather than a
#: literal, a module-level constant, or a for-target over a module-level
#: tuple — keyed ``"<repo-relative path>::<innermost enclosing function
#: name>"``. The scan cannot see the type at these two sites, so they are
#: named here instead; their recorded types are asserted to have a clear
#: site exactly like every literal one. Applies to record sites only: an
#: unresolvable resolve-side ``type`` argument is always an error, since a
#: clear site that cannot be proven to name a type could silently fail to
#: clear the very thing it claims to.
#:
#: ``_worker`` (nested inside ``_run_stage_b_cycle``, ``paramem/server/app.py``)
#: receives its type as the ``kind`` parameter, threaded in from three call
#: sites (``"training_crash"``, ``"consolidation_crash"``, ``"migration_error"``).
#: ``_run_interim_training`` (nested inside ``_extract_and_start_training``,
#: same module) receives its type as the first element of
#: ``_overflow_incident_for``'s ``(type, severity)`` return — either
#: ``"interim_overflow_pending"`` or ``"interim_cap_reached"``.
RECORDED_THROUGH_A_VARIABLE: dict[str, frozenset[str]] = {
    "paramem/server/app.py::_worker": frozenset(
        {"training_crash", "consolidation_crash", "migration_error"}
    ),
    "paramem/server/app.py::_run_interim_training": frozenset(
        {"interim_overflow_pending", "interim_cap_reached"}
    ),
}


def _module_level_bindings(tree: ast.Module) -> "tuple[dict[str, str], dict[str, tuple[str, ...]]]":
    """Return ``(str_constants, tuple_constants)`` bound at *tree*'s module scope.

    ``str_constants`` maps a name to the string literal a module-level
    ``NAME = "..."`` (``Assign``) or ``NAME: <ann> = "..."`` (``AnnAssign``)
    binds it to. ``tuple_constants`` maps a name to the tuple of string
    literals a module-level tuple-of-string-constants binds it to (covers
    ``NAME: tuple[str, ...] = (...)`` incident-type registries).
    """
    str_constants: dict[str, str] = {}
    tuple_constants: dict[str, tuple[str, ...]] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            target, value = node.target, node.value
        else:
            continue
        if not isinstance(target, ast.Name):
            continue
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            str_constants[target.id] = value.value
        elif (
            isinstance(value, ast.Tuple)
            and value.elts
            and all(
                isinstance(elt, ast.Constant) and isinstance(elt.value, str) for elt in value.elts
            )
        ):
            tuple_constants[target.id] = tuple(elt.value for elt in value.elts)
    return str_constants, tuple_constants


def _incident_callee_name(func: ast.expr) -> "str | None":
    """Return the incident-lifecycle call name *func* denotes, or ``None``.

    Matches both a bare ``Name`` (``record_incident(...)``) and an
    ``ast.Attribute`` whose final component is one of the incident call
    names (``incidents.record_incident(...)``) — an attribute-qualified
    call must resolve or become an offender the same as a bare one, never
    vanish from the scan because of how the callable was imported.
    """
    if isinstance(func, ast.Name):
        name = func.id
    elif isinstance(func, ast.Attribute):
        name = func.attr
    else:
        return None
    return name if name in _INCIDENT_CALL_NAMES else None


def _iter_incident_calls(node: ast.AST, ancestors: list):
    """Yield ``(call, ancestors)`` for every incident-lifecycle call under *node*.

    Recurses explicitly (rather than ``ast.walk``, which discards parent
    links) so a call site's enclosing ``for`` loops and function scope are
    recoverable at the point it is found — *ancestors* is the chain of AST
    nodes from the module root down to and including the yielded call's
    immediate parent.
    """
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.Call) and _incident_callee_name(child.func) is not None:
            yield child, ancestors + [node]
        yield from _iter_incident_calls(child, ancestors + [node])


def _resolve_type_arg(
    call: ast.Call,
    callee: str,
    ancestors: list,
    str_constants: dict,
    tuple_constants: dict,
) -> "set[str] | None":
    """Resolve *call*'s ``type`` argument to the set of type strings it denotes.

    ``record_incident``'s ``type`` is keyword-only, so only a ``type=``
    keyword is read for it. ``resolve_incident``/``resolve_incidents_by_type``
    take it positional-or-keyword as their second parameter — the second
    positional argument wins when present, falling back to a ``type=``
    keyword for a caller that named it explicitly. No ``type`` argument
    found at all resolves to ``None``, same as any other unresolvable shape.

    Three resolvable shapes: a string constant; a ``Name`` bound at module
    level to a string constant; or a ``Name`` bound as the target of the
    nearest enclosing ``for`` loop whose iterable is a module-level tuple of
    string constants (every element counts — the loop body cannot
    statically narrow which one runs). Anything else — an f-string, an
    attribute, a function-parameter Name, a Name assigned from a call
    result — resolves to ``None``, unresolved.
    """
    if callee == "record_incident":
        expr = next((kw.value for kw in call.keywords if kw.arg == "type"), None)
    elif len(call.args) >= 2:
        expr = call.args[1]
    else:
        expr = next((kw.value for kw in call.keywords if kw.arg == "type"), None)

    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return {expr.value}
    if isinstance(expr, ast.Name):
        for anc in reversed(ancestors):
            if (
                isinstance(anc, ast.For)
                and isinstance(anc.target, ast.Name)
                and anc.target.id == expr.id
            ):
                if isinstance(anc.iter, ast.Name) and anc.iter.id in tuple_constants:
                    return set(tuple_constants[anc.iter.id])
                return None
        if expr.id in str_constants:
            return {str_constants[expr.id]}
    return None


def _scan_sources(
    sources: "list[tuple[str, str]]",
) -> "tuple[set[str], set[str], list[tuple[str, int, str]]]":
    """Run the incident-lifecycle type scan over in-memory ``(path, source)`` pairs.

    The tree-walking half of :func:`_scan_incident_types`, split out so a
    self-test can exercise the resolver and the offender classification
    against small synthetic snippets without needing a git-tracked file on
    disk. Returns ``(recorded, cleared, offenders)`` — see
    :func:`_scan_incident_types` for what each holds.
    """
    recorded: set = set()
    cleared: set = set()
    offenders: list = []

    for rel, text in sources:
        tree = ast.parse(text)
        lines = text.splitlines()
        str_constants, tuple_constants = _module_level_bindings(tree)

        for call, ancestors in _iter_incident_calls(tree, []):
            callee = _incident_callee_name(call.func)
            resolved = _resolve_type_arg(call, callee, ancestors, str_constants, tuple_constants)
            if resolved is not None:
                (recorded if callee == "record_incident" else cleared).update(resolved)
                continue

            fn_name = enclosing_function_name(tree, call.lineno)
            allow_key = f"{rel}::{fn_name}" if fn_name else None
            if (
                callee == "record_incident"
                and allow_key is not None
                and allow_key in RECORDED_THROUGH_A_VARIABLE
            ):
                recorded.update(RECORDED_THROUGH_A_VARIABLE[allow_key])
                continue

            line = lines[call.lineno - 1] if 0 < call.lineno <= len(lines) else ""
            offenders.append((rel, call.lineno, line.strip()))

    return recorded, cleared, offenders


def _scan_incident_types(
    repo_root: Path,
) -> "tuple[set[str], set[str], list[tuple[str, int, str]]]":
    """Walk every tracked ``paramem/`` source file for the incident-lifecycle calls.

    Returns ``(recorded, cleared, offenders)``: the set of incident type
    strings ever passed to ``record_incident``, the set ever passed to
    ``resolve_incident``/``resolve_incidents_by_type``, and
    ``(path, line, source)`` triples for a call whose ``type`` argument
    could not be resolved and — for a record site — is not covered by
    :data:`RECORDED_THROUGH_A_VARIABLE`. A tracked file that fails to
    decode or parse is a repo defect and is left to raise rather than
    silently scanning fewer files than the guard claims to.
    """
    sources = [
        (py_file.relative_to(repo_root).as_posix(), py_file.read_text())
        for py_file in tracked_python_files(repo_root)
        if py_file.relative_to(repo_root).as_posix().startswith("paramem/")
    ]
    return _scan_sources(sources)


class TestEveryRecordedTypeHasAClearSite:
    """Every incident type recorded anywhere must also be resolved somewhere.

    The lifecycle contract (``incidents`` module docstring) is that an incident
    auto-resolves on the next success of the op it describes, and it is the
    caller that has to honour it.  Nothing in the module can enforce that, so
    this scan does: a type recorded with no clear site anywhere leaves a warning
    riding on ``GET /status`` forever, however healthy the system becomes.

    Walks the AST of every tracked ``paramem/`` file — see
    :func:`_scan_incident_types`.
    """

    def test_no_type_is_recorded_without_a_clear_site(self):
        """Every ``record_incident`` type resolves to a ``resolve_incident``/
        ``resolve_incidents_by_type`` type somewhere in ``paramem/``."""
        repo_root = Path(__file__).resolve().parents[2]
        recorded, cleared, offenders = _scan_incident_types(repo_root)

        assert not offenders, (
            "incident-lifecycle call site(s) with an unresolvable type argument "
            "(not a literal, a module-level constant, or a for-target over a "
            "module-level tuple) and not covered by RECORDED_THROUGH_A_VARIABLE:\n"
            + "\n".join(f"  {path}:{line} — {src}" for path, line, src in offenders)
        )

        assert recorded, "scan found no record_incident call sites — the walk has drifted"

        missing = sorted(recorded - cleared)
        assert not missing, (
            "these incident types are recorded but never resolved, so they stay on "
            f"GET /status forever: {missing}"
        )


class TestAllowlistEntriesAreLiveHits:
    """Every ``RECORDED_THROUGH_A_VARIABLE`` entry must correspond to an
    ACTUAL unresolvable ``record_incident`` type argument inside its named
    function in the current tree — the function merely existing is not
    enough: if the call site were rewritten to pass a literal type (or the
    function were renamed or removed), the entry would silently stop
    covering anything, and the types it names would drop out of the
    recorded set entirely instead of failing loud for lacking a clear site.
    """

    def test_allowlist_entries_are_live_hits(self):
        repo_root = Path(__file__).resolve().parents[2]
        for label in RECORDED_THROUGH_A_VARIABLE:
            rel, fn_name = label.split("::", 1)
            tree = ast.parse((repo_root / rel).read_text())
            assert find_function(tree, fn_name) is not None, (
                f"RECORDED_THROUGH_A_VARIABLE names {label!r} but no such function "
                "exists in that file — remove the stale entry."
            )
            str_constants, tuple_constants = _module_level_bindings(tree)
            hits = [
                call.lineno
                for call, ancestors in _iter_incident_calls(tree, [])
                if _incident_callee_name(call.func) == "record_incident"
                and enclosing_function_name(tree, call.lineno) == fn_name
                and _resolve_type_arg(
                    call, "record_incident", ancestors, str_constants, tuple_constants
                )
                is None
            ]
            assert hits, (
                f"RECORDED_THROUGH_A_VARIABLE names {label!r} but no unresolvable "
                "record_incident type argument was found inside that function — "
                "remove the stale entry, or its type is now a literal/constant and "
                "no longer needs the exemption."
            )


class TestScanSelfTest:
    """Self-test the scan against inline synthetic sources — proof the
    resolver and the offender classification behave as designed, independent
    of whatever the current tree happens to contain."""

    def test_literal_type_with_no_clear_site_is_reported_missing(self):
        recorded, cleared, offenders = _scan_sources(
            [("mod.py", 'record_incident(sd, type="lonely_type", key="k")\n')]
        )
        assert offenders == []
        assert recorded - cleared == {"lonely_type"}

    def test_module_constant_type_resolves(self):
        source = (
            '_MY_TYPE = "my_type"\n'
            'record_incident(sd, type=_MY_TYPE, key="k")\n'
            'resolve_incident(sd, _MY_TYPE, "k")\n'
        )
        recorded, cleared, offenders = _scan_sources([("mod.py", source)])
        assert offenders == []
        assert recorded == {"my_type"}
        assert cleared == {"my_type"}

    def test_for_target_over_module_tuple_resolves(self):
        source = (
            '_TYPES = ("a_type", "b_type")\n'
            "def clear_all():\n"
            "    for _t in _TYPES:\n"
            "        resolve_incidents_by_type(sd, _t)\n"
        )
        recorded, cleared, offenders = _scan_sources([("mod.py", source)])
        assert offenders == []
        assert recorded == set()
        assert cleared == {"a_type", "b_type"}

    def test_unresolvable_record_side_type_in_non_allowlisted_function_is_an_offender(self):
        source = 'def some_function(kind):\n    record_incident(sd, type=kind, key="k")\n'
        recorded, cleared, offenders = _scan_sources([("mod.py", source)])
        assert recorded == set()
        assert cleared == set()
        assert len(offenders) == 1
        assert offenders[0][0] == "mod.py"

    def test_unresolvable_resolve_side_type_is_an_offender(self):
        source = 'def some_function(kind):\n    resolve_incident(sd, kind, "k")\n'
        recorded, cleared, offenders = _scan_sources([("mod.py", source)])
        assert recorded == set()
        assert cleared == set()
        assert len(offenders) == 1

    def test_attribute_form_call_is_seen(self):
        """``incidents.record_incident(...)`` must resolve or become an
        offender — never vanish silently because the callee is an
        ``ast.Attribute`` rather than a bare ``Name``."""
        recorded, cleared, offenders = _scan_sources(
            [("mod.py", 'incidents.record_incident(sd, type="attr_type", key="k")\n')]
        )
        assert offenders == []
        assert recorded == {"attr_type"}


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


# ---------------------------------------------------------------------------
# _run_stage_b_cycle's crash envelope: a raised exception's own structured
# fields (BookkeepingInvariantViolation's divergent_keys,
# ActiveKeyHydrationFailure's dropped_keys/venue) must survive into the
# recorded incident's detail; any other exception keeps the caller-supplied
# detail unchanged.
# ---------------------------------------------------------------------------


def _drive_stage_b_cycle_crash(state, *, exc):
    """Call ``_run_stage_b_cycle`` directly with a body that raises *exc*.

    Mirrors ``TestInterimBookkeepingRegionCrash``'s synchronous-submit
    idiom: ``consolidation_loop`` and ``background_trainer`` are
    pre-seeded ``MagicMock``s so
    ``paramem.server.consolidation.get_or_create_consolidation_loop`` /
    ``_active_bg_trainer`` short-circuit to them without touching a real
    model, and ``bt.submit`` runs the worker inline rather than on a
    background thread.
    """
    state["consolidation_loop"] = MagicMock()
    mock_bt = MagicMock()
    mock_bt.submit.side_effect = lambda fn, **kw: fn()
    state["background_trainer"] = mock_bt

    def _body(loop, bt):
        raise exc

    with patch("paramem.server.app._set_voice_pipeline_profile"):
        app_module._run_stage_b_cycle(
            kind="consolidation_crash",
            incident_key="full",
            failure_summary="full consolidation crashed",
            failure_detail={"phase": "fold"},
            body=_body,
        )


class TestStageBCycleDivergentKeysIncidentDetail:
    """``_run_stage_b_cycle``'s crash envelope (``paramem/server/app.py``,
    the ``except Exception`` block inside its ``_worker`` closure) merges a
    ``BookkeepingInvariantViolation``'s ``divergent_keys`` into the incident
    detail it records, so the incident names exactly what diverged instead
    of leaving that only in the log traceback.  Any other exception type
    records the caller-supplied ``failure_detail`` verbatim."""

    def test_divergent_keys_merged_into_incident_detail(self, state):
        """A BookkeepingInvariantViolation's divergent_keys is folded into
        the incident detail alongside the caller-supplied fields."""
        from paramem.memory.store import BookkeepingInvariantViolation

        divergent = {"episodic": ["g0", "g1"]}
        exc = BookkeepingInvariantViolation(
            "pre-write parity: tier 'episodic', key(s) ['g0', 'g1']", divergent_keys=divergent
        )
        _drive_stage_b_cycle_crash(state, exc=exc)

        incidents = read_incidents(_state_dir(state))
        crashes = [i for i in incidents if i.type == "consolidation_crash"]
        assert len(crashes) == 1, (
            f"expected exactly one consolidation_crash incident; got {incidents}"
        )
        assert crashes[0].detail["divergent_keys"] == divergent
        assert crashes[0].detail["phase"] == "fold", (
            "caller-supplied detail fields must survive the merge"
        )

    def test_generic_exception_keeps_plain_detail(self, state):
        """A non-divergence exception records the failure_detail unchanged --
        no divergent_keys key is synthesized."""
        _drive_stage_b_cycle_crash(state, exc=RuntimeError("boom"))

        incidents = read_incidents(_state_dir(state))
        crashes = [i for i in incidents if i.type == "consolidation_crash"]
        assert len(crashes) == 1, (
            f"expected exactly one consolidation_crash incident; got {incidents}"
        )
        assert "divergent_keys" not in crashes[0].detail
        assert crashes[0].detail == {"phase": "fold"}

    def test_recall_gate_rejected_failed_keys_merged_into_incident_detail(self, state):
        """A RecallGateRejected's failed_keys is folded into the incident
        detail alongside adapter_name/recall_rate/threshold -- naming exactly
        which keys fell short, not just the tier and rate.

        Kills: dropping the failed_keys payload at the incident-recording site.
        """
        from paramem.training.consolidation import RecallGateRejected

        exc = RecallGateRejected(
            "tier 'episodic' reached 1/2 keys",
            adapter_name="episodic",
            recall_rate=0.5,
            threshold=1.0,
            failed_keys=("graph_bad",),
        )
        _drive_stage_b_cycle_crash(state, exc=exc)

        incidents = read_incidents(_state_dir(state))
        crashes = [i for i in incidents if i.type == "consolidation_crash"]
        assert len(crashes) == 1, (
            f"expected exactly one consolidation_crash incident; got {incidents}"
        )
        assert crashes[0].detail["adapter_name"] == "episodic"
        assert crashes[0].detail["recall_rate"] == 0.5
        assert crashes[0].detail["threshold"] == 1.0
        assert crashes[0].detail["failed_keys"] == ["graph_bad"]
        assert crashes[0].detail["phase"] == "fold", (
            "caller-supplied detail fields must survive the merge"
        )

    def test_recall_gate_rejected_empty_failed_keys_omitted_from_incident_detail(self, state):
        """A RecallGateRejected constructed with no per-key data
        (``failed_keys`` defaults to ``()``) omits the field entirely from
        the incident detail rather than publishing an empty list next to a
        failing recall_rate -- an empty ``failed_keys: []`` reads as "no
        keys failed", which is wrong when the tier plainly did fail
        (recall_rate < threshold).
        """
        from paramem.training.consolidation import RecallGateRejected

        exc = RecallGateRejected(
            "recall gate rejected adapter 'episodic'",
            adapter_name="episodic",
            recall_rate=0.5,
            threshold=1.0,
            failed_keys=(),
        )
        _drive_stage_b_cycle_crash(state, exc=exc)

        incidents = read_incidents(_state_dir(state))
        crashes = [i for i in incidents if i.type == "consolidation_crash"]
        assert len(crashes) == 1, (
            f"expected exactly one consolidation_crash incident; got {incidents}"
        )
        assert crashes[0].detail["adapter_name"] == "episodic"
        assert crashes[0].detail["recall_rate"] == 0.5
        assert "failed_keys" not in crashes[0].detail


class TestStageBCycleHydrationFailureIncidentDetail:
    """``_run_stage_b_cycle``'s crash envelope merges an
    ``ActiveKeyHydrationFailure``'s ``dropped_keys`` and ``venue`` into the
    incident detail it records, so the incident names exactly which keys
    could not be hydrated and from which venue."""

    def test_dropped_keys_and_venue_merged_into_incident_detail(self, state):
        """An ActiveKeyHydrationFailure's dropped_keys and venue are folded
        into the incident detail alongside the caller-supplied fields."""
        from paramem.training.consolidation import ActiveKeyHydrationFailure

        exc = ActiveKeyHydrationFailure(dropped_keys=["g0", "g1"], venue="train")
        _drive_stage_b_cycle_crash(state, exc=exc)

        incidents = read_incidents(_state_dir(state))
        crashes = [i for i in incidents if i.type == "consolidation_crash"]
        assert len(crashes) == 1, (
            f"expected exactly one consolidation_crash incident; got {incidents}"
        )
        assert crashes[0].detail["dropped_keys"] == ["g0", "g1"]
        assert crashes[0].detail["venue"] == "train"
        assert crashes[0].detail["phase"] == "fold", (
            "caller-supplied detail fields must survive the merge"
        )


class TestStageBCycleRecallGateIncidentDetail:
    """``_run_stage_b_cycle``'s crash envelope merges a
    ``RecallGateRejected``'s ``adapter_name``/``recall_rate``/``threshold``
    into the incident detail it records — a fold refusal reaches here
    uncaught for BOTH fold kinds, main-tiers and interim: the gate
    (``ConsolidationLoop._assert_tier_recall``) is the same all-or-nothing
    verdict for either, and each fold's own ``except RecallGateRejected``
    only rolls back its in-flight store mutations before re-raising
    unchanged onto this same crash path, so the incident names the tier
    that fell short regardless of which fold raised it."""

    def test_abort_reaches_the_incident_detail(self, state):
        """A RecallGateRejected's adapter_name/recall_rate/threshold are
        folded into the incident detail alongside the caller-supplied
        fields."""
        from paramem.training.consolidation import RecallGateRejected

        exc = RecallGateRejected(
            "tier fell short of 100% recall",
            adapter_name="episodic",
            recall_rate=0.98,
            threshold=1.0,
        )
        _drive_stage_b_cycle_crash(state, exc=exc)

        incidents = read_incidents(_state_dir(state))
        crashes = [i for i in incidents if i.type == "consolidation_crash"]
        assert len(crashes) == 1, (
            f"expected exactly one consolidation_crash incident; got {incidents}"
        )
        assert crashes[0].detail["adapter_name"] == "episodic"
        assert crashes[0].detail["recall_rate"] == 0.98
        assert crashes[0].detail["threshold"] == 1.0
        assert crashes[0].detail["phase"] == "fold", (
            "caller-supplied detail fields must survive the merge"
        )


class TestStageBCycleInterimRecallGateKeepsSessionsPending:
    """An interim cycle's ``run_consolidation_cycle`` call applies the same
    all-or-nothing gate as a main-tiers fold — a shortfall raises
    ``RecallGateRejected`` directly, with no soft ``recall_failed`` return
    value to inspect.  The interim body's own session-retirement step
    (``session_buffer.mark_consolidated``), reached only after a
    successful ``run_consolidation_cycle`` return, is therefore never
    executed, so every session the cycle was consuming stays pending
    alongside the recorded incident."""

    def test_recall_gate_rejection_records_incident_and_never_retires_sessions(self, state):
        """A RecallGateRejected from run_consolidation_cycle is recorded
        with the gate's detail and leaves session retirement unreached."""
        from paramem.training.consolidation import RecallGateRejected

        exc = RecallGateRejected(
            "_assert_tier_recall: tier 'episodic_interim_20260417T0000' "
            "reached 1/2 keys (0.500) on its own trained weights",
            adapter_name="episodic_interim_20260417T0000",
            recall_rate=0.5,
            threshold=1.0,
            failed_keys=("graph_bad",),
        )

        state["consolidation_loop"] = MagicMock()
        mock_bt = MagicMock()
        mock_bt.submit.side_effect = lambda fn, **kw: fn()
        state["background_trainer"] = mock_bt
        session_buffer = state["session_buffer"]

        def _body(loop, bt):
            # Mirrors the interim body's real shape (paramem/server/app.py's
            # _run_interim_training): session retirement runs only after
            # run_consolidation_cycle returns successfully, so a raise here
            # means the mark_consolidated call below is never reached.
            loop.run_consolidation_cycle.side_effect = exc
            loop.run_consolidation_cycle(
                [], [], speaker_id="speaker0", mode="train", run_label="tick"
            )
            session_buffer.mark_consolidated(["session-a"], retention_dir=None)
            return "trained", None

        with patch("paramem.server.app._set_voice_pipeline_profile"):
            app_module._run_stage_b_cycle(
                kind="training_crash",
                incident_key="interim",
                failure_summary="Interim training crashed — 1 session(s) still pending",
                failure_detail={"sessions": 1},
                body=_body,
            )

        incidents = read_incidents(_state_dir(state))
        crashes = [i for i in incidents if i.type == "training_crash"]
        assert len(crashes) == 1, f"expected exactly one training_crash incident; got {incidents}"
        assert crashes[0].detail["adapter_name"] == "episodic_interim_20260417T0000"
        assert crashes[0].detail["recall_rate"] == 0.5
        assert crashes[0].detail["threshold"] == 1.0
        assert crashes[0].detail["failed_keys"] == ["graph_bad"]

        session_buffer.mark_consolidated.assert_not_called()


# ---------------------------------------------------------------------------
# Calibration crash → calibration_crash incident + calibration_run outcome
# ---------------------------------------------------------------------------


class TestCalibrationCrashOutcome:
    """A non-staging (calibration) action's crash records BOTH a
    ``calibration_crash``/``vram_exhausted`` incident (as before) AND sets
    ``_state["calibration_run"]["outcome"] = "crashed"`` — otherwise the
    record :func:`_submit_calibration_run` published at dispatch keeps
    ``outcome: None`` forever, since the run's own normal-completion
    terminal (:func:`_run_calibration_sync`'s) never runs on a crash."""

    def _spec(self, *, run_id: str, route_path: str = "/calibrate/extract"):
        from types import SimpleNamespace

        return SimpleNamespace(run_id=run_id, route_path=route_path)

    def test_generic_crash_sets_outcome_crashed(self, state, monkeypatch):
        from paramem.server.app import _consolidation_run_done
        from paramem.server.consolidation_action import ConsolidationAction

        spec = self._spec(run_id="run-1")
        state["calibration_run"] = {
            "run_id": "run-1",
            "action": "calibrate",
            "route": spec.route_path,
            "artifact_dir": "/tmp/calib/run-1",
            "started_at": "2026-04-22T08:00:00+00:00",
            "outcome": None,
            "finished_at": None,
        }

        class _FakeFuture:
            def exception(self):
                return RuntimeError("boom")

        monkeypatch.setattr(app_module, "_set_voice_pipeline_profile", lambda *a, **kw: None)
        monkeypatch.setattr(app_module, "_target_profile", lambda: "cpu")

        _consolidation_run_done(ConsolidationAction.CALIBRATE, spec, _FakeFuture())

        incidents = read_incidents(_state_dir(state))
        crashes = [i for i in incidents if i.type == "calibration_crash"]
        assert len(crashes) == 1
        assert crashes[0].detail["route_path"] == spec.route_path
        assert crashes[0].detail["run_id"] == "run-1"

        record = state["calibration_run"]
        assert record["outcome"] == "crashed"
        assert record["finished_at"] is not None

    def test_vram_exhausted_crash_on_calibrate_also_sets_outcome_crashed(self, state, monkeypatch):
        """A VramExhausted crash records the vram_exhausted incident (not
        calibration_crash — the two are mutually exclusive by exception
        type) but still sets the outcome, since that bookkeeping is gated
        purely on "non-staging action with a spec", independent of which
        incident branch fired."""
        from paramem.server.app import _consolidation_run_done
        from paramem.server.consolidation_action import ConsolidationAction
        from paramem.utils.vram_guard import VramExhausted

        spec = self._spec(run_id="run-2")
        state["calibration_run"] = {
            "run_id": "run-2",
            "action": "calibrate",
            "route": spec.route_path,
            "artifact_dir": "/tmp/calib/run-2",
            "started_at": "2026-04-22T08:00:00+00:00",
            "outcome": None,
            "finished_at": None,
        }

        class _FakeFuture:
            def exception(self):
                return VramExhausted("dispatch")

        monkeypatch.setattr(app_module, "_set_voice_pipeline_profile", lambda *a, **kw: None)
        monkeypatch.setattr(app_module, "_target_profile", lambda: "cpu")

        _consolidation_run_done(ConsolidationAction.CALIBRATE, spec, _FakeFuture())

        incidents = read_incidents(_state_dir(state))
        assert [i.type for i in incidents] == ["vram_exhausted"]

        record = state["calibration_run"]
        assert record["outcome"] == "crashed"
        assert record["finished_at"] is not None

    def test_stale_run_id_does_not_clobber_a_newer_run(self, state, monkeypatch):
        """A crash callback for an OLD run must not overwrite the record if
        a NEWER run has already started and published its own record —
        matched by run_id, exactly like the normal-completion terminal in
        _run_calibration_sync."""
        from paramem.server.app import _consolidation_run_done
        from paramem.server.consolidation_action import ConsolidationAction

        spec = self._spec(run_id="stale-run")
        state["calibration_run"] = {
            "run_id": "newer-run",
            "action": "calibrate",
            "route": spec.route_path,
            "artifact_dir": "/tmp/calib/newer-run",
            "started_at": "2026-04-22T08:05:00+00:00",
            "outcome": None,
            "finished_at": None,
        }

        class _FakeFuture:
            def exception(self):
                return RuntimeError("boom")

        monkeypatch.setattr(app_module, "_set_voice_pipeline_profile", lambda *a, **kw: None)
        monkeypatch.setattr(app_module, "_target_profile", lambda: "cpu")

        _consolidation_run_done(ConsolidationAction.CALIBRATE, spec, _FakeFuture())

        record = state["calibration_run"]
        assert record["run_id"] == "newer-run"
        assert record["outcome"] is None

    def test_staging_action_crash_leaves_calibration_run_untouched(self, state, monkeypatch):
        """A staging action's crash (spec is None) keeps the pre-existing
        logged-only behaviour — no calibration_run bookkeeping at all."""
        from paramem.server.app import _consolidation_run_done
        from paramem.server.consolidation_action import ConsolidationAction

        state["calibration_run"] = None

        class _FakeFuture:
            def exception(self):
                return RuntimeError("boom")

        monkeypatch.setattr(app_module, "_set_voice_pipeline_profile", lambda *a, **kw: None)
        monkeypatch.setattr(app_module, "_target_profile", lambda: "cpu")

        _consolidation_run_done(ConsolidationAction.INTERIM, None, _FakeFuture())

        assert state["calibration_run"] is None


class TestCalibrationCrashResolvedOnCleanRun:
    """A non-staging action that completes WITHOUT an exception resolves the
    ``calibration_crash`` incident recorded for the run's OWN route — the
    same per-route key the crash branch above records under.  An incident
    recorded for a DIFFERENT route is left untouched."""

    def _spec(self, *, run_id: str, route_path: str):
        from types import SimpleNamespace

        return SimpleNamespace(run_id=run_id, route_path=route_path)

    def test_clean_completion_resolves_the_incident_for_its_own_route(self, state, monkeypatch):
        from paramem.server.app import _consolidation_run_done
        from paramem.server.consolidation_action import ConsolidationAction

        state_dir = _state_dir(state)
        record_incident(
            state_dir,
            type="calibration_crash",
            key="/calibrate/extract",
            severity="failed",
            summary="prior crash",
            detail={},
        )

        spec = self._spec(run_id="run-clean", route_path="/calibrate/extract")

        class _FakeFuture:
            def exception(self):
                return None

        _consolidation_run_done(ConsolidationAction.CALIBRATE, spec, _FakeFuture())

        incidents = read_incidents(state_dir)
        matching = [
            i
            for i in incidents
            if i.type == "calibration_crash" and i.id == "calibration_crash:/calibrate/extract"
        ]
        assert len(matching) == 1
        assert matching[0].status == "resolved"

    def test_clean_completion_leaves_a_different_routes_incident_active(self, state, monkeypatch):
        from paramem.server.app import _consolidation_run_done
        from paramem.server.consolidation_action import ConsolidationAction

        state_dir = _state_dir(state)
        record_incident(
            state_dir,
            type="calibration_crash",
            key="/calibrate/anonymize_facts",
            severity="failed",
            summary="prior crash on a different route",
            detail={},
        )

        spec = self._spec(run_id="run-clean-2", route_path="/calibrate/extract")

        class _FakeFuture:
            def exception(self):
                return None

        _consolidation_run_done(ConsolidationAction.CALIBRATE, spec, _FakeFuture())

        incidents = read_incidents(state_dir)
        other = [
            i
            for i in incidents
            if i.type == "calibration_crash"
            and i.id == "calibration_crash:/calibrate/anonymize_facts"
        ]
        assert len(other) == 1
        assert other[0].status == "active"
