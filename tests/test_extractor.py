"""Tests for knowledge graph extraction."""

import json

import pytest

from paramem.graph.extractor import (
    _extract_json_block,
    _normalize_extraction,
    _stamp_speaker_entity,
)
from paramem.graph.schema import Entity, Relation, SessionGraph


class TestExtractJsonBlock:
    def test_json_in_code_block(self):
        text = 'Some text\n```json\n{"key": "value"}\n```\nMore text'
        result = _extract_json_block(text)
        assert json.loads(result) == {"key": "value"}

    def test_json_in_plain_code_block(self):
        text = 'Some text\n```\n{"key": "value"}\n```'
        result = _extract_json_block(text)
        assert json.loads(result) == {"key": "value"}

    def test_raw_json(self):
        text = 'Here is the result: {"key": "value"} done.'
        result = _extract_json_block(text)
        assert json.loads(result) == {"key": "value"}

    def test_nested_json(self):
        text = '{"outer": {"inner": "value"}}'
        result = _extract_json_block(text)
        parsed = json.loads(result)
        assert parsed["outer"]["inner"] == "value"

    def test_no_json_raises(self):
        with pytest.raises(ValueError, match="No JSON found"):
            _extract_json_block("no json here")

    def test_unbalanced_braces_raises(self):
        # Parser uses json.raw_decode now; "{unclosed" is treated as
        # malformed/truncated JSON, raising ValueError with a "truncated"
        # message rather than the legacy "Unbalanced" wording.
        with pytest.raises(ValueError, match="(?i)truncated|malformed"):
            _extract_json_block("{unclosed")


class TestSessionGraphFromJson:
    def test_parse_extraction_output(self):
        """Simulate what the extractor would produce."""
        data = {
            "session_id": "s001",
            "timestamp": "2026-03-10T10:00:00Z",
            "entities": [
                {"name": "Alex", "entity_type": "person", "attributes": {"age": "29"}},
                {"name": "Heilbronn", "entity_type": "place"},
            ],
            "relations": [
                {
                    "subject": "Alex",
                    "predicate": "lives_in",
                    "object": "Heilbronn",
                    "relation_type": "factual",
                    "confidence": 1.0,
                    "speaker_id": "Speaker0",
                },
            ],
            "summary": "Alex lives in Heilbronn.",
        }
        graph = SessionGraph.model_validate(data)
        assert len(graph.entities) == 2
        assert len(graph.relations) == 1
        assert graph.entities[0].attributes["age"] == "29"


class TestNormalizeExtraction:
    def test_renames_entity_to_name(self):
        data = {
            "entities": [{"entity": "Alex", "type": "person"}],
            "relations": [],
        }
        result = _normalize_extraction(data)
        assert result["entities"][0]["name"] == "Alex"
        assert result["entities"][0]["entity_type"] == "person"

    def test_defaults_missing_relation_type(self):
        data = {
            "entities": [],
            "relations": [{"subject": "A", "predicate": "likes", "object": "B"}],
        }
        result = _normalize_extraction(data)
        assert result["relations"][0]["relation_type"] == "factual"

    def test_novel_entity_type_passes_through(self):
        # entity_type is open (no Literal enforcement). Model-emitted novel
        # types like "widget", "product", "certification" pass through verbatim
        # — the schema YAML's entity_types list is a soft prior for prompt
        # examples, not a closed set.
        data = {
            "entities": [{"name": "X", "type": "widget"}],
            "relations": [],
        }
        result = _normalize_extraction(data)
        assert result["entities"][0]["entity_type"] == "widget"

    def test_empty_entity_type_falls_back(self):
        # Only missing/empty values fall back to the configured default.
        data = {
            "entities": [{"name": "X", "type": ""}],
            "relations": [],
        }
        result = _normalize_extraction(data)
        assert result["entities"][0]["entity_type"] == "concept"

    def test_invalid_relation_type_defaults_to_factual(self):
        data = {
            "entities": [],
            "relations": [
                {
                    "subject": "A",
                    "predicate": "likes",
                    "object": "B",
                    "relation_type": "unknown_type",
                }
            ],
        }
        result = _normalize_extraction(data)
        assert result["relations"][0]["relation_type"] == "factual"

    def test_captures_extra_fields_as_attributes(self):
        data = {
            "entities": [{"entity": "Alex", "type": "person", "age": "29"}],
            "relations": [],
        }
        result = _normalize_extraction(data)
        assert result["entities"][0]["attributes"]["age"] == "29"

    def test_normalized_data_validates(self):
        """End-to-end: normalize then validate as SessionGraph."""
        data = {
            "session_id": "s001",
            "timestamp": "2026-03-10T10:00:00Z",
            "entities": [
                {
                    "entity": "user",
                    "type": "person",
                    "preference": "black coffee",
                }
            ],
            "relations": [
                {
                    "subject": "user",
                    "predicate": "prefers",
                    "object": "black coffee",
                    "confidence": 1.0,
                    "speaker_id": "Speaker0",
                }
            ],
        }
        normalized = _normalize_extraction(data)
        graph = SessionGraph.model_validate(normalized)
        # Entity name preserves the model's literal output — no static
        # title-casing. World-knowledge correction is the surface enrichment
        # stage's job, not a destructive normalization here.
        assert graph.entities[0].name == "user"
        assert graph.relations[0].relation_type == "factual"


class TestStampSpeakerEntity:
    """Unit tests for _stamp_speaker_entity post-processor."""

    def _make_graph(self, entities, relations=None):
        """Helper: build a minimal SessionGraph."""
        return SessionGraph(
            session_id="s001",
            timestamp="2026-01-01T00:00:00Z",
            entities=entities,
            relations=relations or [],
        )

    def test_stamps_speaker_id_on_matching_entity(self):
        """Entity whose name matches speaker_name receives speaker_id."""
        graph = self._make_graph(
            entities=[
                Entity(name="Alex", entity_type="person"),
                Entity(name="Berlin", entity_type="place"),
            ]
        )
        result = _stamp_speaker_entity(graph, speaker_name="Alex", speaker_id="Speaker0")
        alex = next(e for e in result.entities if e.name == "Alex")
        assert alex.speaker_id == "Speaker0"
        berlin = next(e for e in result.entities if e.name == "Berlin")
        assert berlin.speaker_id is None

    def test_folds_full_name_duplicate_into_first_name_canonical(self):
        """When the model emits both 'Alex' and 'Alex Morgan', fold 'Alex Morgan'
        into 'Alex', merge attributes, and rewrite relations."""
        graph = self._make_graph(
            entities=[
                Entity(name="Alex", entity_type="person"),
                Entity(
                    name="Alex Morgan",
                    entity_type="person",
                    attributes={"has_last_name": "Morgan"},
                ),
                Entity(name="Brightfield Labs", entity_type="organization"),
            ],
            relations=[
                Relation(
                    subject="Alex Morgan",
                    predicate="works_at",
                    object="Brightfield Labs",
                    relation_type="factual",
                    speaker_id="Speaker0",
                ),
                Relation(
                    subject="Alex",
                    predicate="lives_in",
                    object="Berlin",
                    relation_type="factual",
                    speaker_id="Speaker0",
                ),
            ],
        )
        result = _stamp_speaker_entity(graph, speaker_name="Alex", speaker_id="Speaker0")

        # Only one speaker entity should remain.
        names = [e.name for e in result.entities]
        assert "Alex" in names
        assert "Alex Morgan" not in names, f"Duplicate not folded; entities: {names}"

        # The canonical entity has the last name merged in and speaker_id set.
        alex = next(e for e in result.entities if e.name == "Alex")
        assert alex.speaker_id == "Speaker0"
        assert alex.attributes.get("has_last_name") == "Morgan"

        # The relation originally referencing "Alex Morgan" now references "Alex".
        works_at = next(r for r in result.relations if r.predicate == "works_at")
        assert works_at.subject == "Alex"

    def test_no_match_returns_graph_unchanged(self):
        """When speaker_name is not found among entities, return unchanged."""
        graph = self._make_graph(entities=[Entity(name="Berlin", entity_type="place")])
        result = _stamp_speaker_entity(graph, speaker_name="Alex", speaker_id="Speaker0")
        assert len(result.entities) == 1
        assert result.entities[0].speaker_id is None

    def test_empty_entities_list(self):
        """Empty entity list does not raise."""
        graph = self._make_graph(entities=[])
        result = _stamp_speaker_entity(graph, speaker_name="Alex", speaker_id="Speaker0")
        assert result.entities == []

    def test_idempotent_when_no_duplicates(self):
        """When there is no full-name duplicate, only stamping occurs — no fold."""
        graph = self._make_graph(
            entities=[
                Entity(name="Alex", entity_type="person"),
                Entity(name="Berlin", entity_type="place"),
            ],
            relations=[
                Relation(
                    subject="Alex",
                    predicate="lives_in",
                    object="Berlin",
                    relation_type="factual",
                    speaker_id="Speaker0",
                )
            ],
        )
        result = _stamp_speaker_entity(graph, speaker_name="Alex", speaker_id="Speaker0")
        assert len(result.entities) == 2
        assert len(result.relations) == 1
        alex = next(e for e in result.entities if e.name == "Alex")
        assert alex.speaker_id == "Speaker0"

    def test_case_insensitive_name_match(self):
        """Name matching is case-insensitive (speaker_name='alex', entity='Alex')."""
        graph = self._make_graph(entities=[Entity(name="Alex", entity_type="person")])
        result = _stamp_speaker_entity(graph, speaker_name="alex", speaker_id="Speaker0")
        alex = next(e for e in result.entities if e.name == "Alex")
        assert alex.speaker_id == "Speaker0"
