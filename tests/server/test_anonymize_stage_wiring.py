"""``_stage_anonymize``'s opted-out diagnostic (``scrub_categories`` empty
— no model call), its drop-record projection (``category``/``side``/
``reason`` only — never ``text``/``word``), and ``_live_reload_base_model``'s
handling of a SCAN prompt skeleton drift refusal raised during a live
reload: the server stays up, cloud-only, with ``reload_failed``.

Driving patterns (state builders) copied from ``tests/server/test_calibrate.py``
and ``tests/server/test_calibrate_routes.py`` (read-only — never imported
from).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import paramem.graph.stage_anonymize as stage_anonymize_module
import paramem.server.app as app_module
from paramem.cloud.anonymize import AnonymizedContract
from paramem.config.taxonomy import ScrubCategory
from paramem.graph.flow import StageContext, StageState
from paramem.graph.schema import Relation, SessionGraph
from paramem.graph.stage_anonymize import _stage_anonymize


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
        scrub_categories=(),
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


class TestStageAnonymizeDropRecordProjection:
    """The ``scan_dropped_entries`` diagnostic keeps ``category``/``side``/
    ``reason`` only — never ``text`` (the real value) or ``word`` (the
    model's own keyword) — so neither reaches ``graph.diagnostics``."""

    def test_only_category_side_and_reason_reach_the_diagnostic(self, monkeypatch) -> None:
        contract = AnonymizedContract(
            status="ok",
            forward={},
            reverse={},
            anon_transcript="hi Alex",
            declared=frozenset(),
            rekey_dropped=0,
            raw="{}",
            failure=None,
            facts=[],
            model_calls=1,
            scan_dropped=1,
            scan_dropped_entries=[
                {
                    "category": "City",
                    "side": "scan",
                    "text": "a real value that must not leak",
                    "reason": "reverted",
                    "word": "a model keyword that must not leak",
                }
            ],
        )
        monkeypatch.setattr(stage_anonymize_module, "anonymize", lambda *a, **k: contract)

        ctx = _stage_context(scrub_categories=(ScrubCategory(prefix="Person"),))
        state = StageState(graph=_graph_with_one_relation())

        result = _stage_anonymize(ctx, state)

        assert result.graph.diagnostics["scan_dropped_entries"] == [
            {"category": "City", "side": "scan", "reason": "reverted"}
        ]


class TestLiveReloadRefusesOnScanSkeletonDrift:
    def test_drift_refusal_sets_reload_failed_and_stays_up(self, tmp_path, monkeypatch) -> None:
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
        monkeypatch.setattr(app_module, "_release_base_model_in_process", lambda: None)
        monkeypatch.setattr(app_module, "_compute_topology_assessment", lambda *a, **k: None)

        def _raise_drift(config):
            raise RuntimeError(
                "SCAN prompt skeleton drift: the live configs/schema.yaml anonymizer "
                "prefix table renders a SCAN skeleton over the pinned token budget"
            )

        monkeypatch.setattr(app_module, "_load_model_into_state", _raise_drift)

        outcome = app_module._live_reload_base_model(refresh_config_from_disk=True, lock_held=True)

        assert outcome == "reload_failed"
        assert state["cloud_only_reason"] == "reload_failed"
        # The server is left up (this call returned normally, not raised) —
        # the caller's own contract for "leaves the server up and cloud-only".
