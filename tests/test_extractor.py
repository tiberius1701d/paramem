"""Tests for knowledge graph extraction."""

import json

import pytest

from paramem.graph.extractor import _extract_json_block, _normalize_extraction
from paramem.graph.schema import SessionGraph


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
        with pytest.raises(ValueError, match="Unbalanced"):
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

    def test_invalid_entity_type_defaults_to_concept(self):
        data = {
            "entities": [{"name": "X", "type": "widget"}],
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
                }
            ],
        }
        normalized = _normalize_extraction(data)
        graph = SessionGraph.model_validate(normalized)
        assert graph.entities[0].name == "User"
        assert graph.relations[0].relation_type == "factual"
