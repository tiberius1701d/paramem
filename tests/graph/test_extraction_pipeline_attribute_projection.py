"""Unit tests for the entity-attribute projection wired into
``ExtractionPipeline._run_extractor``.

Both :meth:`~paramem.graph.extraction_pipeline.ExtractionPipeline.run` and
:meth:`~paramem.graph.extraction_pipeline.ExtractionPipeline.run_procedural`
must return a graph whose ``relations`` include
:func:`~paramem.graph.relation_prep.attribute_relations`'s projection of
``graph.entities[*].attributes`` — the ONE place either entry projects
entity attributes into relations. No model, no tokenizer — the extractor
function itself is monkeypatched to return a real ``SessionGraph``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from peft import PeftModel

from paramem.graph.extraction_pipeline import ExtractionConfig, ExtractionPipeline
from paramem.graph.schema import Entity, Relation, SessionGraph


def _peft_model_mock() -> MagicMock:
    """``MagicMock(spec=PeftModel)`` -- passes the ``isinstance(model,
    PeftModel)`` precondition ``base_model_inference`` (entered by
    ``ExtractionPipeline.run``/``run_procedural``) now enforces.
    ``gradient_checkpointing_disable``/``_enable`` are dynamic
    ``__getattr__``-delegated attributes a real (wrapped) PeftModel
    exposes that ``spec`` cannot see via ``dir(PeftModel)``, so they are
    pre-set explicitly (mirrors ``tests/server/test_gates.py::
    _make_mock_model``)."""
    model = MagicMock(spec=PeftModel)
    model.gradient_checkpointing_disable = MagicMock()
    model.gradient_checkpointing_enable = MagicMock()
    return model


def _pipeline(**config_overrides) -> ExtractionPipeline:
    config_overrides.setdefault("scrub", {"person name"})
    return ExtractionPipeline(
        model=_peft_model_mock(),
        tokenizer=MagicMock(),
        config=ExtractionConfig(**config_overrides),
        prompts_dir=None,
    )


def _graph_with_attribute_entity() -> SessionGraph:
    return SessionGraph(
        session_id="s001",
        timestamp="2026-01-01T00:00:00Z",
        entities=[Entity(name="Alex", entity_type="person", attributes={"email": "a@b.com"})],
        relations=[],
    )


class TestRunProjectsAttributeRelations:
    def test_run_returns_graph_with_projected_attribute_relation(self, monkeypatch):
        monkeypatch.setattr(
            "paramem.graph.extraction_pipeline.extract_graph",
            lambda *a, **kw: _graph_with_attribute_entity(),
        )
        pipeline = _pipeline()
        graph = pipeline.run("transcript", "s001", speaker_id="speaker0")

        attr_rels = [r for r in graph.relations if r.relation_type == "attribute"]
        assert len(attr_rels) == 1
        assert attr_rels[0].subject == "Alex"
        assert attr_rels[0].predicate == "has email"
        assert attr_rels[0].object == "a@b.com"
        assert attr_rels[0].speaker_id == "speaker0"

    def test_run_procedural_returns_graph_with_projected_attribute_relation(self, monkeypatch):
        monkeypatch.setattr(
            "paramem.graph.extraction_pipeline.extract_procedural_graph",
            lambda *a, **kw: _graph_with_attribute_entity(),
        )
        pipeline = _pipeline()
        graph = pipeline.run_procedural("transcript", "s001", speaker_id="speaker0")

        attr_rels = [r for r in graph.relations if r.relation_type == "attribute"]
        assert len(attr_rels) == 1
        assert attr_rels[0].speaker_id == "speaker0"

    def test_projected_relations_attributed_to_the_calling_speaker(self, monkeypatch):
        monkeypatch.setattr(
            "paramem.graph.extraction_pipeline.extract_graph",
            lambda *a, **kw: _graph_with_attribute_entity(),
        )
        pipeline = _pipeline()
        graph = pipeline.run("transcript", "s001", speaker_id="speaker9")

        attr_rels = [r for r in graph.relations if r.relation_type == "attribute"]
        assert attr_rels[0].speaker_id == "speaker9"

    def test_stop_at_truncated_chain_still_returns_projected_relations(self, monkeypatch):
        """The projection runs AFTER extract_fn returns, unconditionally --
        never gated on whether a stop_at(...) scope is open. A caller
        requesting an early return (calibration's dispatch_chain) still
        gets the projected relations for whatever entities the truncated
        chain produced."""
        from paramem.graph.phase_trace import stop_at

        monkeypatch.setattr(
            "paramem.graph.extraction_pipeline.extract_graph",
            lambda *a, **kw: _graph_with_attribute_entity(),
        )
        pipeline = _pipeline()
        with stop_at("local_extract"):
            graph = pipeline.run("transcript", "s001", speaker_id="speaker0")

        attr_rels = [r for r in graph.relations if r.relation_type == "attribute"]
        assert len(attr_rels) == 1
        assert attr_rels[0].object == "a@b.com"

    def test_no_attributes_no_projected_relations(self, monkeypatch):
        empty_graph = SessionGraph(
            session_id="s002",
            timestamp="2026-01-01T00:00:00Z",
            entities=[Entity(name="Alex", entity_type="person")],
            relations=[],
        )
        monkeypatch.setattr(
            "paramem.graph.extraction_pipeline.extract_graph",
            lambda *a, **kw: empty_graph,
        )
        pipeline = _pipeline()
        graph = pipeline.run("transcript", "s002", speaker_id="speaker0")
        assert graph.relations == []

    def test_existing_relations_are_preserved_alongside_the_projection(self, monkeypatch):
        base_relation = Relation(
            subject="speaker0",
            predicate="lives in",
            object="Berlin",
            relation_type="factual",
            speaker_id="speaker0",
        )
        graph_with_both = SessionGraph(
            session_id="s003",
            timestamp="2026-01-01T00:00:00Z",
            entities=[Entity(name="Alex", entity_type="person", attributes={"email": "a@b.com"})],
            relations=[base_relation],
        )
        monkeypatch.setattr(
            "paramem.graph.extraction_pipeline.extract_graph",
            lambda *a, **kw: graph_with_both,
        )
        pipeline = _pipeline()
        graph = pipeline.run("transcript", "s003", speaker_id="speaker0")

        assert base_relation in graph.relations
        assert any(r.relation_type == "attribute" for r in graph.relations)
        assert len(graph.relations) == 2
