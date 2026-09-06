"""``_stage_anonymize``'s opted-out diagnostic (``scrub_categories`` empty
— no model call) and its drop-record projection (``category``/``side``/
``reason`` only — never ``text``/``word``).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import paramem.graph.stage_anonymize as stage_anonymize_module
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
