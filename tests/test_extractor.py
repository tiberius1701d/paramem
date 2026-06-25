"""Tests for knowledge graph extraction."""

import json

import pytest

from paramem.graph.extractor import (
    _extract_json_block,
    _normalize_extraction,
    _stamp_speaker_entity,
)
from paramem.graph.schema import Entity, SessionGraph


class TestExtractJsonBlock:
    def test_json_in_code_block(self):
        text = 'Some text\n```json\n{"entities": [], "relations": []}\n```\nMore text'
        result = _extract_json_block(text)
        assert json.loads(result) == {"entities": [], "relations": []}

    def test_json_in_plain_code_block(self):
        text = 'Some text\n```\n{"facts": []}\n```'
        result = _extract_json_block(text)
        assert json.loads(result) == {"facts": []}

    def test_raw_json(self):
        text = 'Here is the result: {"entities": [], "relations": []} done.'
        result = _extract_json_block(text)
        assert json.loads(result) == {"entities": [], "relations": []}

    def test_nested_json(self):
        text = (
            '{"entities": [{"name": "Alex", "entity_type": "person", '
            '"attributes": {}}], "relations": []}'
        )
        result = _extract_json_block(text)
        parsed = json.loads(result)
        assert parsed["entities"][0]["name"] == "Alex"

    def test_no_json_raises(self):
        with pytest.raises(ValueError, match="No JSON found"):
            _extract_json_block("no json here")

    def test_unbalanced_braces_raises(self):
        # Parser walks every `{` candidate; "{unclosed" never closes so
        # raw_decode fails for the only candidate.  Surfaces as the "no
        # parseable JSON" path with a max_tokens-truncation hint.
        with pytest.raises(ValueError, match="(?i)no parseable JSON"):
            _extract_json_block("{unclosed")

    def test_skips_brace_quoted_placeholder_in_preamble(self):
        """SOTA's preamble narration sometimes references placeholder names
        in brace notation like ``{Topic_1}`` — the parser must skip past
        those and find the real envelope further down."""
        text = (
            "I'll introduce:\n"
            "- `{Topic_1}` = Mechanical Engineering\n"
            "- `{City_1}` = Duisburg\n\n"
            "```json\n"
            '{"facts": [{"subject": "Person_1", "predicate": "studied", "object": "{Topic_1}"}]}\n'
            "```"
        )
        result = _extract_json_block(text)
        parsed = json.loads(result)
        assert "facts" in parsed
        assert parsed["facts"][0]["subject"] == "Person_1"

    def test_skips_brace_placeholder_when_no_code_fence(self):
        """Same case but without a code-fence — the parser must walk through
        ``{Topic_1}`` (raw_decode raises, skip), then ``{City_1}`` (skip),
        then find the real ``{"facts": …}`` envelope."""
        text = (
            "I'll introduce {Topic_1} for the degree field and {City_1} for "
            "the university city. Here are the enriched facts:\n"
            '{"facts": [{"subject": "Person_1", "predicate": "lives_in", "object": "Germany"}]}'
        )
        result = _extract_json_block(text)
        parsed = json.loads(result)
        assert "facts" in parsed

    def test_rejects_inner_subobject_when_outer_envelope_truncated(self):
        """Truncation discipline: a model cut at max_tokens mid-string
        emits a valid inner sub-object even though the outer envelope
        never closes.  The parser MUST reject the inner sub-object so
        the truncation surfaces as a real parse failure."""
        # Outer envelope opens but never closes; inner sub-object is fine.
        text = (
            '{"entities": [{"name": "Alex", "entity_type": "person", '
            '"attributes": {}}, {"name": "Bob"'
        )
        with pytest.raises(ValueError, match="(?i)envelope keys|no parseable JSON"):
            _extract_json_block(text)

    def test_accepts_plausibility_empty_list(self):
        """Plausibility legitimately returns ``[]`` when all facts were
        filtered.  The parser must accept lists (even empty) as valid
        envelopes."""
        result = _extract_json_block("[]")
        assert json.loads(result) == []

    def test_accepts_plausibility_nonempty_list(self):
        text = '[{"subject": "Alex", "predicate": "lives_in", "object": "Berlin"}]'
        result = _extract_json_block(text)
        parsed = json.loads(result)
        assert parsed[0]["subject"] == "Alex"


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
                    "speaker_id": "speaker0",
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
                    "speaker_id": "speaker0",
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
    """Unit tests for _stamp_speaker_entity post-processor (id-based stamping).

    Under the id-as-subject convention the model emits ``Speaker{N}`` as the
    entity name and relation subject.  The stamp function identifies entities
    by structural speaker-id form, not by display-name match.
    """

    def _make_graph(self, entities, relations=None):
        """Helper: build a minimal SessionGraph."""
        return SessionGraph(
            session_id="s001",
            timestamp="2026-01-01T00:00:00Z",
            entities=entities,
            relations=relations or [],
        )

    def test_stamps_session_speaker_entity_by_id(self):
        """Entity whose name is the session speaker id receives the authoritative lowercase id."""
        graph = self._make_graph(
            entities=[
                Entity(name="speaker0", entity_type="person"),
                Entity(name="Berlin", entity_type="place"),
            ]
        )
        result = _stamp_speaker_entity(graph, speaker_name="Tobias", speaker_id="speaker0")
        speaker_ent = next(e for e in result.entities if e.name == "speaker0")
        assert speaker_ent.speaker_id == "speaker0"
        berlin = next(e for e in result.entities if e.name == "Berlin")
        assert berlin.speaker_id is None

    def test_display_name_entity_is_not_stamped(self):
        """A display-name entity (not a speaker id) must not receive speaker_id."""
        graph = self._make_graph(
            entities=[
                Entity(name="Tobias Becker", entity_type="person"),
                Entity(name="speaker0", entity_type="person"),
            ]
        )
        result = _stamp_speaker_entity(graph, speaker_name="Tobias", speaker_id="speaker0")
        third_party = next(e for e in result.entities if e.name == "Tobias Becker")
        assert third_party.speaker_id is None
        speaker_ent = next(e for e in result.entities if e.name == "speaker0")
        assert speaker_ent.speaker_id == "speaker0"

    def test_other_speaker_id_entity_receives_own_name(self):
        """A different speaker{N} entity (not the session speaker) gets its own
        name as speaker_id, preserving separate speaker identity."""
        graph = self._make_graph(
            entities=[
                Entity(name="speaker0", entity_type="person"),
                Entity(name="speaker1", entity_type="person"),
                Entity(name="Vienna", entity_type="place"),
            ]
        )
        result = _stamp_speaker_entity(graph, speaker_name="Tobias", speaker_id="speaker0")
        s0 = next(e for e in result.entities if e.name == "speaker0")
        s1 = next(e for e in result.entities if e.name == "speaker1")
        assert s0.speaker_id == "speaker0"
        assert s1.speaker_id == "speaker1"
        vienna = next(e for e in result.entities if e.name == "Vienna")
        assert vienna.speaker_id is None

    def test_empty_entities_list_does_not_raise(self):
        """Empty entity list does not raise."""
        graph = self._make_graph(entities=[])
        result = _stamp_speaker_entity(graph, speaker_name="Tobias", speaker_id="speaker0")
        assert result.entities == []

    def test_no_speaker_id_entity_returns_graph_unchanged(self):
        """When no entity has a speaker-id name, all speaker_id fields stay None."""
        graph = self._make_graph(entities=[Entity(name="Berlin", entity_type="place")])
        result = _stamp_speaker_entity(graph, speaker_name="Tobias", speaker_id="speaker0")
        assert len(result.entities) == 1
        assert result.entities[0].speaker_id is None

    def test_wrong_digit_guard_stamps_authoritative_id(self):
        """Authoritative-id pin: when ent.name == speaker_id.lower(), stamp the
        authoritative id (guards against model emitting wrong digit)."""
        graph = self._make_graph(
            entities=[
                Entity(name="speaker0", entity_type="person"),
            ]
        )
        result = _stamp_speaker_entity(graph, speaker_name=None, speaker_id="speaker0")
        ent = result.entities[0]
        # ent.name == "speaker0" == "speaker0".lower() → authoritative pin fires.
        assert ent.speaker_id == "speaker0"

    def test_wrong_digit_entity_gets_own_name(self):
        """Entity with different digit (model wrong-digit error) gets own name, not session id."""
        graph = self._make_graph(
            entities=[
                Entity(name="speaker1", entity_type="person"),
            ]
        )
        # Session speaker is speaker0; model emitted speaker1 — wrong digit.
        result = _stamp_speaker_entity(graph, speaker_name=None, speaker_id="speaker0")
        ent = result.entities[0]
        # ent.name ("speaker1") != "speaker0".lower() → falls to else branch.
        assert ent.speaker_id == "speaker1"

    def test_speaker_name_none_still_stamps(self):
        """speaker_name=None is accepted; stamping proceeds based on speaker_id only."""
        graph = self._make_graph(
            entities=[
                Entity(name="speaker0", entity_type="person"),
            ]
        )
        result = _stamp_speaker_entity(graph, speaker_name=None, speaker_id="speaker0")
        ent = result.entities[0]
        assert ent.speaker_id == "speaker0"

    def test_idempotent_on_already_stamped_entity(self):
        """Calling stamp twice on an already-stamped entity is idempotent."""
        graph = self._make_graph(
            entities=[
                Entity(name="speaker0", entity_type="person", speaker_id="speaker0"),
                Entity(name="Berlin", entity_type="place"),
            ]
        )
        result = _stamp_speaker_entity(graph, speaker_name="Tobias", speaker_id="speaker0")
        result2 = _stamp_speaker_entity(result, speaker_name="Tobias", speaker_id="speaker0")
        speaker_ent = next(e for e in result2.entities if e.name == "speaker0")
        assert speaker_ent.speaker_id == "speaker0"
        assert len(result2.entities) == 2

    def test_other_speaker_already_lowercase_gets_own_name(self):
        """Other-speaker entity (e.g. 'speaker1') gets its own name as speaker_id.
        Under lowercase-uniform identity there is no re-casing step."""
        graph = self._make_graph(
            entities=[
                Entity(name="speaker0", entity_type="person"),
                Entity(name="speaker1", entity_type="person"),
            ]
        )
        result = _stamp_speaker_entity(graph, speaker_name="Tobias", speaker_id="speaker0")
        s0 = next(e for e in result.entities if e.name == "speaker0")
        s1 = next(e for e in result.entities if e.name == "speaker1")
        assert s0.speaker_id == "speaker0"
        # speaker1 != "speaker0".lower() → else branch: ent.speaker_id = ent.name.
        assert s1.speaker_id == "speaker1"
