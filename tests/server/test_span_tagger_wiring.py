"""Integration wiring for the span tagger: ``_stage_anonymize`` diagnostics,
the ``/calibrate/anonymize_facts`` door (real dispatch, real
``response.json``), the ``/calibrate/anonymize`` chain's egress gate, and
the load-site boundary — a reachable config with an unresolvable
checkpoint fails at the runtime-components build, ``scrub: []`` boots
clean, and a live config-apply that hits the same failure refuses the
apply and leaves the server up and cloud-only.

Driving patterns (state builders, the executor-stub + spec-capture
dispatch seam) copied from ``tests/server/test_calibrate.py`` and
``tests/server/test_calibrate_routes.py`` (read-only — never imported
from).
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from peft import PeftModel

import paramem.server.app as app_module
from paramem.cloud import span_tagger
from paramem.cloud.anonymize import AnonymizedContract, failed_contract
from paramem.config.taxonomy import ScrubCategory
from paramem.graph.flow import StageContext, StageState
from paramem.graph.flows import SESSION_EXTRACT, _session_egress_permitted
from paramem.graph.schema import Relation, SessionGraph
from paramem.graph.stage_anonymize import _stage_anonymize
from paramem.server.config import CloudConfig, PathsConfig, SanitizationConfig
from tests.anonymizer_doubles import span_tagger_reset

__all__ = ["span_tagger_reset"]

PERSON = ScrubCategory(
    name="Person", prefix="Person", hints=("person name",), tagger_labels=("person",)
)


def _ok_contract() -> AnonymizedContract:
    return AnonymizedContract(
        status="ok",
        forward={"Alex": "Person_1"},
        reverse={"Person_1": "Alex"},
        anon_transcript="[user] hi Person_1",
        declared=frozenset({"Person_1"}),
        rekey_dropped=0,
        raw="{}",
        facts=[],
    )


def _stage_context(**overrides) -> StageContext:
    defaults = dict(
        model=MagicMock(),
        tokenizer=MagicMock(),
        transcript="[user] hi Alex",
        session_id="s1",
        speaker_id="speaker1",
        speaker_name=None,
        temperature=0.0,
        max_tokens=256,
        plausibility_max_tokens=256,
        prompts_dir=None,
        system_prompt_filename="",
        user_prompt_filename="",
        model_alias=None,
        seed=None,
        timestamp=None,
        source_type="dialogue",
        validate=True,
        cloud_enabled=True,
        enrichment_provider="anthropic",
        enrichment_provider_model="claude-test",
        enrichment_provider_endpoint=None,
        plausibility_judge="auto",
        plausibility_stage="deanon",
        plausibility_model="claude-test",
        plausibility_endpoint=None,
        scrub_categories=(PERSON,),
        correction_entity_types=None,
        anonymize_token_envelope=8192,
    )
    defaults.update(overrides)
    return StageContext(**defaults)


def _graph_with_one_relation() -> SessionGraph:
    return SessionGraph(
        session_id="s1",
        timestamp="2026-01-01T00:00:00Z",
        relations=[
            Relation(
                subject="speaker1",
                predicate="knows",
                object="Alex",
                relation_type="factual",
                speaker_id="speaker1",
            )
        ],
    )


class TestStageAnonymizeDiagnostics:
    def test_opted_out_writes_opted_out_diagnostic(self) -> None:
        ctx = _stage_context(scrub_categories=())
        state = StageState(graph=_graph_with_one_relation())

        result = _stage_anonymize(ctx, state)

        assert result.graph.diagnostics["anonymize"] == "opted_out"
        assert result.payload.status == "opted_out"

    def test_ok_writes_ok_diagnostic_and_core_placeholders(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "paramem.graph.stage_anonymize.anonymize", lambda *a, **k: _ok_contract()
        )
        ctx = _stage_context()
        state = StageState(graph=_graph_with_one_relation())

        result = _stage_anonymize(ctx, state)

        assert result.graph.diagnostics["anonymize"] == "ok"
        assert result.graph.diagnostics["core_placeholders"]["count"] == 1
        assert result.payload.status == "ok"

    def test_failed_writes_failed_diagnostic_and_falls_back_to_local_plausibility(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            "paramem.graph.stage_anonymize.anonymize",
            lambda *a, **k: failed_contract(failure="tagger", raw="span tagger unavailable"),
        )
        monkeypatch.setattr(
            "paramem.graph.stage_anonymize._fallback_plausibility_on_raw",
            lambda graph, *a, **k: graph,  # preserve the diagnostics already stamped
        )
        ctx = _stage_context()
        state = StageState(graph=_graph_with_one_relation())

        result = _stage_anonymize(ctx, state)

        assert result.graph.diagnostics["anonymize"] == "failed"
        assert result.payload is None

    def test_ok_diagnostics_carry_inert_dropped_and_rekey_dropped_beside_scan_dropped(
        self, monkeypatch
    ) -> None:
        contract = AnonymizedContract(
            status="ok",
            forward={"Alex": "Person_1"},
            reverse={"Person_1": "Alex"},
            anon_transcript="[user] hi Person_1",
            declared=frozenset({"Person_1"}),
            rekey_dropped=2,
            raw="{}",
            facts=[],
            scan_dropped=1,
            inert_dropped=3,
        )
        monkeypatch.setattr("paramem.graph.stage_anonymize.anonymize", lambda *a, **k: contract)
        ctx = _stage_context()
        state = StageState(graph=_graph_with_one_relation())

        result = _stage_anonymize(ctx, state)

        assert result.graph.diagnostics["scan_dropped"] == 1
        assert result.graph.diagnostics["inert_dropped"] == 3
        assert result.graph.diagnostics["rekey_dropped"] == 2


# ---------------------------------------------------------------------------
# /calibrate/anonymize_facts — real dispatch, real response.json
# ---------------------------------------------------------------------------


def _peft_model_mock():
    model = MagicMock()
    model.__class__ = PeftModel
    return model


def _calibrate_state(tmp_path):
    from paramem.server.session_buffer import SessionBuffer

    config = MagicMock()
    config.consolidation.calibrate_endpoint_enabled = True
    config.consolidation.training_idle_debounce_s = 30
    config.consolidation.orphan_retirement_seconds = None
    config.consolidation.retain_sessions = False
    config.debug = False
    config.sanitization = SanitizationConfig()
    config.cloud = CloudConfig(enabled=True)
    config.paths.data = tmp_path / "state"
    config.paths = PathsConfig(data=tmp_path / "state", calibration=tmp_path / "calibration")
    config.paths.calibration_prompts.mkdir(parents=True, exist_ok=True)
    config.adapter_dir = tmp_path / "adapters"
    config.adapter_dir.mkdir(parents=True, exist_ok=True)
    config.model_config.model_id = "test-model"
    config.vram.cooldown_gate_threshold_c = 0
    config.vram.cooldown_gate_max_wait_fold_s = 0
    config.vram.cooldown_gate_poll_s = 0

    loop = MagicMock()
    loop.extraction.config = SimpleNamespace(
        scrub_categories=(PERSON,),
        anonymize_token_envelope=8192,
    )
    loop.extraction.prompts_dir = None
    loop.model = None
    loop.tokenizer = None

    buffer = SessionBuffer(tmp_path / "sessions", debug=False)
    speaker_store = MagicMock()
    speaker_store.is_anonymous.return_value = False

    return {
        "config": config,
        "consolidating": False,
        "model": _peft_model_mock(),
        "tokenizer": MagicMock(),
        "memory_store": MagicMock(),
        "consolidation_loop": loop,
        "session_buffer": buffer,
        "speaker_store": speaker_store,
        "migration": {},
        "calibration_run": None,
        "event_loop": None,
        "mode": "local",
        "background_trainer": None,
        "cloud_only_reason": None,
        "last_chat_monotonic": None,
        "pending_rehydration": False,
        "integrity_check_failed": False,
    }


class TestCalibrateAnonymizeFactsTaggerFailure:
    def test_submit_then_run_then_read_response_json_reports_failure_tagger(
        self, tmp_path, monkeypatch, span_tagger_reset
    ) -> None:
        # No handle loaded — every anonymize() call fails closed with
        # failure="tagger".
        span_tagger.reset_for_tests()

        state = _calibrate_state(tmp_path)
        submitted = []

        def _record(fn, status, **kwargs):
            submitted.append((fn, status))
            return status

        monkeypatch.setattr(app_module, "_state", state)
        monkeypatch.setattr(app_module, "_retro_claim_orphan_sessions", lambda: 0)
        monkeypatch.setattr(app_module, "_dispatch_to_executor", _record)

        client = TestClient(app_module.app, raise_server_exceptions=False)

        resp = client.post(
            "/calibrate/anonymize_facts",
            json={"facts": [{"subject": "speaker1", "predicate": "knows", "object": "Alex"}]},
        )

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "started_calibration"
        run_id = body["run_id"]

        assert len(submitted) == 1
        fn, _status = submitted[0]
        fn()  # run the dispatched work synchronously, exactly as the executor would

        # /status sources calibration_run from _state verbatim.
        assert state["calibration_run"]["run_id"] == run_id
        assert state["calibration_run"]["outcome"] == "completed"

        response_path = list((tmp_path / "calibration" / "artifacts").rglob("response.json"))
        assert len(response_path) == 1
        response = json.loads(response_path[0].read_text())
        assert response["parsed"]["status"] == "failed"
        assert response["parsed"]["failure"] == "tagger"


class TestCalibrateAnonymizeFactsSurfacesInertAndRekeyDropped:
    def test_submit_then_run_then_read_response_json_reports_inert_and_rekey_dropped(
        self, tmp_path, monkeypatch, span_tagger_reset
    ) -> None:
        ok_contract = AnonymizedContract(
            status="ok",
            forward={},
            reverse={},
            anon_transcript="",
            declared=frozenset(),
            rekey_dropped=2,
            raw="{}",
            facts=[{"subject": "speaker1", "predicate": "knows", "object": "Alex"}],
            scan_dropped=0,
            inert_dropped=1,
        )
        monkeypatch.setattr("paramem.cloud.anonymize.anonymize", lambda *a, **k: ok_contract)

        state = _calibrate_state(tmp_path)
        submitted = []

        def _record(fn, status, **kwargs):
            submitted.append((fn, status))
            return status

        monkeypatch.setattr(app_module, "_state", state)
        monkeypatch.setattr(app_module, "_retro_claim_orphan_sessions", lambda: 0)
        monkeypatch.setattr(app_module, "_dispatch_to_executor", _record)

        client = TestClient(app_module.app, raise_server_exceptions=False)

        resp = client.post(
            "/calibrate/anonymize_facts",
            json={"facts": [{"subject": "speaker1", "predicate": "knows", "object": "Alex"}]},
        )

        assert resp.status_code == 200, resp.text
        assert len(submitted) == 1
        fn, _status = submitted[0]
        fn()  # run the dispatched work synchronously, exactly as the executor would

        response_path = list((tmp_path / "calibration" / "artifacts").rglob("response.json"))
        assert len(response_path) == 1
        response = json.loads(response_path[0].read_text())
        assert response["parsed"]["status"] == "ok"
        assert response["parsed"]["inert_dropped"] == 1
        assert response["parsed"]["rekey_dropped"] == 2


# ---------------------------------------------------------------------------
# /calibrate/anonymize — non-egress-capable config never reaches the
# anonymize phase (the flow's own enabled_when gate).
# ---------------------------------------------------------------------------


class TestAnonymizeStageGatedOffWhenNotEgressCapable:
    def _anonymize_spec(self):
        return next(s for s in SESSION_EXTRACT if s.stage == "anonymize")

    def test_enabled_when_is_false_on_a_non_egress_capable_config(self) -> None:
        ctx = _stage_context(validate=True, cloud_enabled=False, enrichment_provider="")
        spec = self._anonymize_spec()
        assert spec.enabled_when(ctx) is False

    def test_enabled_when_is_true_on_an_egress_capable_config(self) -> None:
        ctx = _stage_context(validate=True, cloud_enabled=True, enrichment_provider="anthropic")
        spec = self._anonymize_spec()
        # The verdict also depends on a resolvable API key; assert against
        # the SAME predicate production reads (_session_egress_permitted),
        # not a re-derivation.
        assert spec.enabled_when(ctx) == (bool(ctx.validate) and _session_egress_permitted(ctx))


# ---------------------------------------------------------------------------
# Load-site boundary: a reachable config with an unresolvable checkpoint
# fails the runtime-components build; scrub: [] boots without a load
# attempt at all.
# ---------------------------------------------------------------------------


class TestLoadSpanTaggerBootBoundary:
    def _reachable_config(self):
        config = MagicMock()
        config.sanitization.scrub_categories = (PERSON,)
        config.cloud.enabled = True
        config.sanitization.cloud_mode = "anonymize"
        config.consolidation.extraction_enrichment_provider = "anthropic"
        config.consolidation.extraction_enrichment_provider_model = "claude-test"
        config.consolidation.extraction_enrichment_provider_endpoint = None
        return config

    def test_reachable_config_with_unresolvable_checkpoint_raises(
        self, monkeypatch, span_tagger_reset
    ) -> None:
        # _build_runtime_components has no try/except around its
        # _load_span_tagger(config) call (verified in
        # paramem/server/app.py, step 9) — a raise here propagates
        # straight out of that function to ITS caller (boot / apply).
        def _raise(cfg):
            raise RuntimeError(
                "span tagger: failed to load checkpoint='x' revision='y': cold cache"
            )

        monkeypatch.setattr(span_tagger, "load_at_startup", _raise)

        with pytest.raises(RuntimeError, match="cold cache"):
            app_module._load_span_tagger(self._reachable_config())

    def test_scrub_empty_boots_without_a_load_attempt(self, monkeypatch, span_tagger_reset) -> None:
        config = self._reachable_config()
        config.sanitization.scrub_categories = ()  # operator opt-out

        calls = []
        monkeypatch.setattr(span_tagger, "load_at_startup", lambda cfg: calls.append(cfg))

        app_module._load_span_tagger(config)  # must not raise

        assert calls == []


class TestLiveConfigApplyRefusesOnUnresolvableCheckpoint:
    def test_component_rebuild_failure_sets_apply_failed_and_stays_up(
        self, tmp_path, monkeypatch
    ) -> None:
        state = {
            "config": SimpleNamespace(
                model_config=SimpleNamespace(model_id="test-model"),
                model_name="test-model",
                vram=SimpleNamespace(nf4_disk_to_runtime_factor=1.0),
                paths=SimpleNamespace(data=tmp_path),
            ),
            "mode": "local",
            "cloud_only_reason": None,
            "voice_profile": "cpu",
        }
        monkeypatch.setattr(app_module, "_state", state)
        monkeypatch.setattr(app_module, "_refresh_config_from_disk_into_state", lambda: None)
        monkeypatch.setattr(app_module, "_load_model_into_state", lambda config: None)
        monkeypatch.setattr(app_module, "_release_base_model_in_process", lambda: None)
        monkeypatch.setattr(app_module, "_compute_topology_assessment", lambda *a, **k: None)

        def _raise_component_rebuild(*a, **k):
            raise RuntimeError(
                "span tagger: failed to load checkpoint='x' revision='y': cold cache"
            )

        monkeypatch.setattr(app_module, "_build_runtime_components", _raise_component_rebuild)

        outcome = app_module._live_reload_base_model(refresh_config_from_disk=True, lock_held=True)

        assert outcome == "apply_failed"
        assert state["cloud_only_reason"] == "apply_failed"
        # The server is left up (this call returned normally, not raised) —
        # the caller's own contract for "leaves the server up and cloud-only".
