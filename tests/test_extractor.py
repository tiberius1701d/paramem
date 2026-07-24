"""Tests for knowledge graph extraction."""

import json
from unittest.mock import MagicMock, patch

import pytest

from paramem.cloud.deanonymize import CloudScope
from paramem.graph.extractor import (
    _extract_json_block,
    _fallback_plausibility_on_raw,
    _normalize_extraction,
    _stamp_speaker_entity,
    extract_procedural_graph,
)
from paramem.graph.flow import StageContext, StageState
from paramem.graph.flows import _stage_rebuild, extract_graph
from paramem.graph.phase_trace import extraction_trace, get_phases, stop_at
from paramem.graph.schema import Entity, Relation, SessionGraph


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
        """Cloud's preamble narration sometimes references placeholder names
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

    def test_no_envelope_message_does_not_assert_truncation_when_complete(self):
        """A response that ends on a closing brace (i.e. is NOT truncated)
        but contains an unescaped control character inside a string value
        must not be misdiagnosed as ``max_tokens`` truncation — the outer
        ``raw_decode`` fails on the control character, only an inner
        sub-object (no envelope keys) parses, and the response text itself
        is complete."""
        text = (
            '{"mapping": {"Alice": "Person_1"}, "anonymized_transcript": '
            '"[user] hi\n[assistant] there"}'
        )
        assert text.rstrip().endswith("}")
        with pytest.raises(ValueError) as excinfo:
            _extract_json_block(text)
        message = str(excinfo.value)
        assert "envelope keys" in message
        assert "control character" in message
        assert "truncated at max_tokens" not in message

    def test_no_envelope_message_still_reports_truncation_when_incomplete(self):
        """The truncation diagnosis is still surfaced when the response
        text genuinely does not end on a closing brace/bracket: a
        complete inner sub-object (no envelope keys) parses, but the
        outer envelope never closes."""
        text = (
            '{"entities": [{"name": "Alex", "entity_type": "person", '
            '"attributes": {}}, {"name": "Bob"'
        )
        assert not text.rstrip().endswith(("}", "]"))
        with pytest.raises(ValueError) as excinfo:
            _extract_json_block(text)
        message = str(excinfo.value)
        assert "truncated at max_tokens" in message

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

    def _self_loop_relations(self, subject: str, obj: str) -> list[dict]:
        """Run one relation through _normalize_extraction and return the survivors."""
        data = {
            "entities": [],
            "relations": [{"subject": subject, "predicate": "studied_at", "object": obj}],
        }
        return _normalize_extraction(data)["relations"]

    def test_self_loop_exact_dropped(self):
        assert self._self_loop_relations("KIT", "KIT") == []

    def test_self_loop_case_variant_dropped(self):
        assert self._self_loop_relations("KIT", "kit") == []

    def test_self_loop_diacritic_variant_dropped(self):
        """canonical() folds diacritics, so these land on ONE node key."""
        assert self._self_loop_relations("José", "Jose") == []

    def test_self_loop_blank_run_variant_dropped(self):
        """Blank runs fold to ``_``: both endpoints are node key ``new_york``."""
        assert self._self_loop_relations("New York", "New_York") == []

    def test_self_loop_full_casefold_variant_dropped(self):
        """ß casefolds to ss — .lower() did not catch this."""
        assert self._self_loop_relations("Straße", "Strasse") == []

    def test_hyphen_variant_is_not_a_self_loop(self):
        """``-`` is not a blank: ``anna-maria`` and ``anna_maria`` are DISTINCT
        node keys, so the relation survives."""
        assert len(self._self_loop_relations("Anna-Maria", "Anna Maria")) == 1

    def test_distinct_endpoints_survive(self):
        assert len(self._self_loop_relations("Alex", "KIT")) == 1

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
        result = _stamp_speaker_entity(graph, speaker_id="speaker0")
        speaker_ent = next(e for e in result.entities if e.name == "speaker0")
        assert speaker_ent.speaker_id == "speaker0"
        berlin = next(e for e in result.entities if e.name == "Berlin")
        assert berlin.speaker_id is None

    def test_display_name_entity_is_not_stamped(self):
        """A display-name entity (not a speaker id) must not receive speaker_id."""
        graph = self._make_graph(
            entities=[
                Entity(name="Jordan Becker", entity_type="person"),
                Entity(name="speaker0", entity_type="person"),
            ]
        )
        result = _stamp_speaker_entity(graph, speaker_id="speaker0")
        third_party = next(e for e in result.entities if e.name == "Jordan Becker")
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
        result = _stamp_speaker_entity(graph, speaker_id="speaker0")
        s0 = next(e for e in result.entities if e.name == "speaker0")
        s1 = next(e for e in result.entities if e.name == "speaker1")
        assert s0.speaker_id == "speaker0"
        assert s1.speaker_id == "speaker1"
        vienna = next(e for e in result.entities if e.name == "Vienna")
        assert vienna.speaker_id is None

    def test_empty_entities_list_does_not_raise(self):
        """Empty entity list does not raise."""
        graph = self._make_graph(entities=[])
        result = _stamp_speaker_entity(graph, speaker_id="speaker0")
        assert result.entities == []

    def test_no_speaker_id_entity_returns_graph_unchanged(self):
        """When no entity has a speaker-id name, all speaker_id fields stay None."""
        graph = self._make_graph(entities=[Entity(name="Berlin", entity_type="place")])
        result = _stamp_speaker_entity(graph, speaker_id="speaker0")
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
        result = _stamp_speaker_entity(graph, speaker_id="speaker0")
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
        result = _stamp_speaker_entity(graph, speaker_id="speaker0")
        ent = result.entities[0]
        # ent.name ("speaker1") != "speaker0".lower() → falls to else branch.
        assert ent.speaker_id == "speaker1"

    def test_single_entity_stamped_by_speaker_id(self):
        """Stamping proceeds based on speaker_id; entity receives authoritative id."""
        graph = self._make_graph(
            entities=[
                Entity(name="speaker0", entity_type="person"),
            ]
        )
        result = _stamp_speaker_entity(graph, speaker_id="speaker0")
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
        result = _stamp_speaker_entity(graph, speaker_id="speaker0")
        result2 = _stamp_speaker_entity(result, speaker_id="speaker0")
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
        result = _stamp_speaker_entity(graph, speaker_id="speaker0")
        s0 = next(e for e in result.entities if e.name == "speaker0")
        s1 = next(e for e in result.entities if e.name == "speaker1")
        assert s0.speaker_id == "speaker0"
        # speaker1 != "speaker0".lower() → else branch: ent.speaker_id = ent.name.
        assert s1.speaker_id == "speaker1"

    # --- Document exact-full-name rewrite (Guard A + Guard B) ---

    def test_document_exact_full_name_subject_rewritten(self):
        """Third-person document, full-name entity + relation subject match
        → rewritten to speaker0 and stamped."""
        graph = self._make_graph(
            entities=[Entity(name="Alex Walker", entity_type="person")],
            relations=[
                Relation(
                    subject="Alex Walker",
                    predicate="leads",
                    object="the platform team",
                    relation_type="factual",
                    speaker_id="speaker0",
                )
            ],
        )
        result = _stamp_speaker_entity(
            graph,
            speaker_id="speaker0",
            speaker_name="Alex Walker",
            source_type="document",
        )
        assert len(result.entities) == 1
        ent = result.entities[0]
        assert ent.name == "speaker0"
        assert ent.speaker_id == "speaker0"
        assert result.relations[0].subject == "speaker0"

    def test_document_exact_full_name_object_rewritten(self):
        """Object-position self-disclosure rewritten to speaker0; subject unchanged."""
        graph = self._make_graph(
            entities=[Entity(name="the platform team", entity_type="organization")],
            relations=[
                Relation(
                    subject="the platform team",
                    predicate="led_by",
                    object="Alex Walker",
                    relation_type="factual",
                    speaker_id="speaker0",
                )
            ],
        )
        result = _stamp_speaker_entity(
            graph,
            speaker_id="speaker0",
            speaker_name="Alex Walker",
            source_type="document",
        )
        rel = result.relations[0]
        assert rel.subject == "the platform team"
        assert rel.object == "speaker0"

    def test_single_token_speaker_name_fails_closed(self):
        """Guard A: a single-token speaker_name never triggers the rewrite."""
        graph = self._make_graph(
            entities=[Entity(name="Alex", entity_type="person")],
            relations=[
                Relation(
                    subject="Alex",
                    predicate="leads",
                    object="the team",
                    relation_type="factual",
                    speaker_id="speaker0",
                )
            ],
        )
        result = _stamp_speaker_entity(
            graph,
            speaker_id="speaker0",
            speaker_name="Alex",
            source_type="document",
        )
        assert result.entities[0].name == "Alex"
        assert result.relations[0].subject == "Alex"

    def test_transcript_source_never_rewrites(self):
        """Guard B: an exact full-name match on a transcript source is left alone."""
        graph = self._make_graph(
            entities=[Entity(name="Alex Walker", entity_type="person")],
            relations=[
                Relation(
                    subject="Alex Walker",
                    predicate="leads",
                    object="the team",
                    relation_type="factual",
                    speaker_id="speaker0",
                )
            ],
        )
        result = _stamp_speaker_entity(
            graph,
            speaker_id="speaker0",
            speaker_name="Alex Walker",
            source_type="transcript",
        )
        assert result.entities[0].name == "Alex Walker"
        assert result.relations[0].subject == "Alex Walker"

    def test_first_name_only_not_rewritten(self):
        """A first-name-only entity does not match the speaker's full name."""
        graph = self._make_graph(
            entities=[Entity(name="Alex", entity_type="person")],
            relations=[
                Relation(
                    subject="Alex",
                    predicate="leads",
                    object="the team",
                    relation_type="factual",
                    speaker_id="speaker0",
                )
            ],
        )
        result = _stamp_speaker_entity(
            graph,
            speaker_id="speaker0",
            speaker_name="Alex Walker",
            source_type="document",
        )
        assert result.entities[0].name == "Alex"
        assert result.relations[0].subject == "Alex"

    def test_different_full_name_not_rewritten(self):
        """A different full name (same first name, different surname) is untouched."""
        graph = self._make_graph(
            entities=[Entity(name="Alex Müller", entity_type="person")],
            relations=[
                Relation(
                    subject="Alex Müller",
                    predicate="supervised",
                    object="the thesis",
                    relation_type="factual",
                    speaker_id="speaker0",
                )
            ],
        )
        result = _stamp_speaker_entity(
            graph,
            speaker_id="speaker0",
            speaker_name="Alex Walker",
            source_type="document",
        )
        assert result.entities[0].name == "Alex Müller"
        assert result.relations[0].subject == "Alex Müller"

    def test_superset_name_not_rewritten(self):
        """A superset name (organization sharing the speaker's name) is not collapsed."""
        graph = self._make_graph(
            entities=[Entity(name="Alex Walker Foundation", entity_type="organization")],
        )
        result = _stamp_speaker_entity(
            graph,
            speaker_id="speaker0",
            speaker_name="Alex Walker",
            source_type="document",
        )
        assert result.entities[0].name == "Alex Walker Foundation"

    def test_case_and_diacritic_insensitive_match(self):
        """canonical() folds case/diacritics/whitespace runs for the exact-match rewrite."""
        graph = self._make_graph(
            entities=[Entity(name="álex  walker", entity_type="person")],
        )
        result = _stamp_speaker_entity(
            graph,
            speaker_id="speaker0",
            speaker_name="Alex Walker",
            source_type="document",
        )
        assert result.entities[0].name == "speaker0"

    def test_self_loop_dropped_after_rewrite(self):
        """A relation whose subject AND object are both the full name is DROPPED,
        not emitted as (speaker0, pred, speaker0)."""
        graph = self._make_graph(
            entities=[Entity(name="Alex Walker", entity_type="person")],
            relations=[
                Relation(
                    subject="Alex Walker",
                    predicate="reports_to",
                    object="Alex Walker",
                    relation_type="factual",
                    speaker_id="speaker0",
                )
            ],
        )
        result = _stamp_speaker_entity(
            graph,
            speaker_id="speaker0",
            speaker_name="Alex Walker",
            source_type="document",
        )
        assert result.relations == []

    def test_speaker_name_none_is_noop(self):
        """speaker_name=None skips the rewrite entirely; existing is_speaker_id
        stamping still fires."""
        graph = self._make_graph(
            entities=[
                Entity(name="Alex Walker", entity_type="person"),
                Entity(name="speaker0", entity_type="person"),
            ],
        )
        result = _stamp_speaker_entity(
            graph,
            speaker_id="speaker0",
            speaker_name=None,
            source_type="document",
        )
        third_party = next(e for e in result.entities if e.name == "Alex Walker")
        assert third_party.speaker_id is None
        speaker_ent = next(e for e in result.entities if e.name == "speaker0")
        assert speaker_ent.speaker_id == "speaker0"


class TestTimestampPropagation:
    """extract_procedural_graph stamps SessionGraph.timestamp from the passed
    ``timestamp`` param (session-start assertion time) rather than now().

    Mocks generate_answer (the sole LLM call in the procedural pipeline) so
    the test runs on CPU with no real model — the surface under test is the
    kwarg plumbing (timestamp → SessionGraph.timestamp), not generation
    quality.
    """

    def _fake_raw_output(self) -> str:
        return json.dumps(
            {
                "entities": [{"name": "speaker0", "entity_type": "person"}],
                "relations": [
                    {
                        "subject": "speaker0",
                        "predicate": "prefers",
                        "object": "tea",
                        "relation_type": "preference",
                        "confidence": 1.0,
                    }
                ],
            }
        )

    def test_explicit_timestamp_wins_over_now(self):
        """A caller-supplied timestamp lands verbatim on SessionGraph.timestamp."""
        with patch(
            "paramem.graph.extractor.generate_answer",
            return_value=self._fake_raw_output(),
        ):
            graph = extract_procedural_graph(
                model=MagicMock(),
                tokenizer=MagicMock(),
                transcript="I like tea.",
                session_id="s001",
                speaker_id="speaker0",
                timestamp="2026-05-01T12:00:00+00:00",
            )
        assert graph.timestamp == "2026-05-01T12:00:00+00:00"

    def test_no_timestamp_falls_back_to_now(self):
        """Omitting timestamp preserves the now() fallback (unchanged behaviour)."""
        from datetime import datetime, timezone

        before = datetime.now(timezone.utc)
        with patch(
            "paramem.graph.extractor.generate_answer",
            return_value=self._fake_raw_output(),
        ):
            graph = extract_procedural_graph(
                model=MagicMock(),
                tokenizer=MagicMock(),
                transcript="I like tea.",
                session_id="s001",
                speaker_id="speaker0",
            )
        after = datetime.now(timezone.utc)
        parsed = datetime.fromisoformat(graph.timestamp)
        assert before <= parsed <= after


class TestSecondOrderExtractPhase:
    """Unit tests for the ``second_order_extract`` phase — gate, union, and
    ``stop_at`` contract.

    Mocks ``_generate_extraction`` (dispatched on the ``user_prompt_filename``
    kwarg, which differs between ``local_extract`` and
    ``second_order_extract``) so the tests run on CPU with no real model —
    the surface under test is the gate/union/``stop_at`` plumbing, not
    generation quality.
    """

    def _pass1_no_named_person(self) -> str:
        """Pass-1 output with only the speaker + a place — no named
        non-speaker person, so the gate must fail."""
        return json.dumps(
            {
                "entities": [
                    {"name": "speaker0", "entity_type": "person"},
                    {"name": "Berlin", "entity_type": "place"},
                ],
                "relations": [
                    {
                        "subject": "speaker0",
                        "predicate": "lives_in",
                        "object": "Berlin",
                        "relation_type": "factual",
                        "confidence": 1.0,
                    }
                ],
            }
        )

    def _pass1_with_named_person(self) -> str:
        """Pass-1 output collapsing 'my brother Nadeem lives in Porto' into
        only the kinship edge — the measured Mistral 7B failure mode."""
        return json.dumps(
            {
                "entities": [
                    {"name": "speaker0", "entity_type": "person"},
                    {"name": "Nadeem", "entity_type": "person"},
                ],
                "relations": [
                    {
                        "subject": "speaker0",
                        "predicate": "has_brother",
                        "object": "Nadeem",
                        "relation_type": "social",
                        "confidence": 1.0,
                    }
                ],
            }
        )

    def _second_order_output(self) -> str:
        """Second-order-pass output recovering Nadeem's own attribute."""
        return json.dumps(
            {
                "entities": [{"name": "Nadeem", "entity_type": "person"}],
                "relations": [
                    {
                        "subject": "Nadeem",
                        "predicate": "lives_in",
                        "object": "Porto",
                        "relation_type": "factual",
                        "confidence": 1.0,
                    }
                ],
            }
        )

    def _pass1_sam_picked_up_kids(self) -> str:
        """Pass-1 output already capturing Sam's fact in full — the
        measured live regression fixture (real transcript: 'Sam picked
        the kids up from school today')."""
        return json.dumps(
            {
                "entities": [{"name": "Sam", "entity_type": "person"}],
                "relations": [
                    {
                        "subject": "Sam",
                        "predicate": "picked_up",
                        "object": "kids",
                        "relation_type": "factual",
                        "confidence": 1.0,
                    }
                ],
            }
        )

    def _second_order_sam_picks_up_kids_drifted(self) -> str:
        """Second-order-pass re-emit of the SAME fact with a drifted
        predicate surface ('picks_up' vs pass-1's 'picked_up'). The plain
        union keeps both; collapsing this near-dup is
        refinement_normalization's job, not this union site's."""
        return json.dumps(
            {
                "entities": [{"name": "Sam", "entity_type": "person"}],
                "relations": [
                    {
                        "subject": "Sam",
                        "predicate": "picks_up",
                        "object": "kids",
                        "relation_type": "factual",
                        "confidence": 1.0,
                    }
                ],
            }
        )

    def _fake_generate(self, pass1_output: str, second_order_output: str):
        def _inner(*args, **kwargs):
            if kwargs.get("user_prompt_filename") == "extraction_second_order.txt":
                return second_order_output
            return pass1_output

        return _inner

    def test_gate_skips_when_no_named_non_speaker_person(self):
        """No named non-speaker person in the pass-1 graph ->
        second_order_extract never fires (no second _generate_extraction
        call, no phase record)."""
        with patch(
            "paramem.graph.extractor._generate_extraction",
            side_effect=self._fake_generate(
                self._pass1_no_named_person(), self._second_order_output()
            ),
        ) as mock_gen:
            graph = extract_graph(
                model=None,
                tokenizer=None,
                transcript="I live in Berlin.",
                session_id="s001",
                speaker_id="speaker0",
                scrub={"person name"},
            )
        assert mock_gen.call_count == 1, (
            "second_order_extract must not call _generate_extraction when its gate fails"
        )
        phase_names = [p.name for p in get_phases(graph)]
        assert "second_order_extract" not in phase_names

    def test_second_order_relations_unioned_when_named_person_present(self):
        """Plain union: the second-order pass's recovered fact (Nadeem,
        lives_in, Porto — the dropped-attribute case) survives alongside
        the pass-1 kinship edge (speaker0, has_brother, Nadeem). The
        second-order pass owns recall only; it is not a dedup boundary."""
        with patch(
            "paramem.graph.extractor._generate_extraction",
            side_effect=self._fake_generate(
                self._pass1_with_named_person(), self._second_order_output()
            ),
        ):
            graph = extract_graph(
                model=None,
                tokenizer=None,
                transcript="My brother Nadeem lives in Porto.",
                session_id="s001",
                speaker_id="speaker0",
                scrub={"person name"},
            )
        phase_names = [p.name for p in get_phases(graph)]
        assert "second_order_extract" in phase_names
        assert any(r.subject == "Nadeem" and r.object == "Porto" for r in graph.relations)
        # Pass-1 kinship edge survives alongside the second-order relation.
        assert any(r.subject == "speaker0" and r.object == "Nadeem" for r in graph.relations)

    def test_second_order_union_is_plain_recall_preserving(self):
        """Second-order re-emits a fact pass-1 already has, with a drifted
        predicate surface ('picks_up' vs pass-1's 'picked_up' for the same
        (Sam, kids) fact). The union is a plain extend — BOTH survive here;
        collapsing same-fact predicate drift is refinement_normalization's
        job (and GraphMerger exact-triple dedup for literal repeats),
        tested elsewhere, not this union site's."""
        with patch(
            "paramem.graph.extractor._generate_extraction",
            side_effect=self._fake_generate(
                self._pass1_sam_picked_up_kids(), self._second_order_sam_picks_up_kids_drifted()
            ),
        ):
            graph = extract_graph(
                model=None,
                tokenizer=None,
                transcript="Sam picked the kids up from school today.",
                session_id="s001",
                speaker_id="speaker0",
                scrub={"person name"},
            )
        phase_names = [p.name for p in get_phases(graph)]
        assert "second_order_extract" in phase_names, (
            "gate must pass: Sam is a named non-speaker person"
        )
        sam_relations = {(r.predicate, r.object) for r in graph.relations if r.subject == "Sam"}
        assert ("picked_up", "kids") in sam_relations
        assert ("picks_up", "kids") in sam_relations

    def test_stop_phase_second_order_extract_returns_after_union(self):
        """``stop_at("second_order_extract")`` returns immediately after
        the union, before any later phase fires."""
        with patch(
            "paramem.graph.extractor._generate_extraction",
            side_effect=self._fake_generate(
                self._pass1_with_named_person(), self._second_order_output()
            ),
        ):
            with stop_at("second_order_extract"):
                graph = extract_graph(
                    model=None,
                    tokenizer=None,
                    transcript="My brother Nadeem lives in Porto.",
                    session_id="s001",
                    speaker_id="speaker0",
                    scrub={"person name"},
                )
        phase_names = [p.name for p in get_phases(graph)]
        assert phase_names == ["local_extract", "second_order_extract"]
        assert any(r.subject == "Nadeem" and r.object == "Porto" for r in graph.relations)


class TestExtractGraphTimestampPropagation:
    """extract_graph stamps SessionGraph.timestamp from the passed
    ``timestamp`` param on the SUCCESS (parsed) path — i.e. through
    ``_parse_extraction`` — not just on the parse-failure fallback branch.

    Mocks ``_generate_extraction`` (the sole LLM call before parsing) with
    valid, parseable extraction JSON containing at least one entity and one
    relation, so ``_parse_extraction`` succeeds rather than falling back to
    the empty-graph exception handler. ``stop_at("local_extract")`` returns
    immediately after parsing succeeds, isolating the surface under test
    (timestamp plumbing) from the unrelated STT/HA/cloud phases.
    """

    def _fake_raw_output(self) -> str:
        return json.dumps(
            {
                "entities": [
                    {"name": "speaker0", "entity_type": "person"},
                    {"name": "tea", "entity_type": "object"},
                ],
                "relations": [
                    {
                        "subject": "speaker0",
                        "predicate": "prefers",
                        "object": "tea",
                        "relation_type": "preference",
                        "confidence": 1.0,
                    }
                ],
            }
        )

    def test_explicit_timestamp_wins_over_now_on_success_path(self):
        """A caller-supplied timestamp lands verbatim on SessionGraph.timestamp
        when extraction parses successfully (proves the success path ran)."""
        with patch(
            "paramem.graph.extractor._generate_extraction",
            return_value=self._fake_raw_output(),
        ):
            with stop_at("local_extract"):
                graph = extract_graph(
                    model=None,
                    tokenizer=None,
                    transcript="I like tea.",
                    session_id="s001",
                    speaker_id="speaker0",
                    timestamp="2026-06-28T23:21:30+00:00",
                    scrub={"person name"},
                )
        assert graph.relations, "fake output must parse successfully, not fall back"
        assert graph.timestamp == "2026-06-28T23:21:30+00:00"

    def test_no_timestamp_falls_back_to_now_on_success_path(self):
        """Omitting timestamp preserves the now() fallback on the success path."""
        from datetime import datetime, timezone

        before = datetime.now(timezone.utc)
        with patch(
            "paramem.graph.extractor._generate_extraction",
            return_value=self._fake_raw_output(),
        ):
            with stop_at("local_extract"):
                graph = extract_graph(
                    model=None,
                    tokenizer=None,
                    transcript="I like tea.",
                    session_id="s001",
                    speaker_id="speaker0",
                    scrub={"person name"},
                )
        after = datetime.now(timezone.utc)
        assert graph.relations, "fake output must parse successfully, not fall back"
        parsed = datetime.fromisoformat(graph.timestamp)
        assert before <= parsed <= after


class TestEmptyRelationsTerminal:
    """``local_extract``'s empty-relations terminal: when the pass-1 graph
    has no relations, extract_graph returns immediately — no
    second_order_extract, no cloud_pipeline — even when
    those later phases are otherwise enabled/configured to fire."""

    def _empty_output(self) -> str:
        return json.dumps({"entities": [], "relations": []})

    def test_no_relations_yields_only_local_extract_phase(self):
        with patch(
            "paramem.graph.extractor._generate_extraction",
            return_value=self._empty_output(),
        ):
            graph = extract_graph(
                model=None,
                tokenizer=None,
                transcript="Just saying hello.",
                session_id="s001",
                speaker_id="speaker0",
                scrub={"person name"},
                # Later phases are configured ON; the empty-relations
                # terminal must still short-circuit before any of them.
                validate=True,
                cloud_enabled=True,
                enrichment_provider="anthropic",
            )
        assert graph.relations == []
        phase_names = [p.name for p in get_phases(graph)]
        assert phase_names == ["local_extract"]


class TestAttributeTypedFactsSurviveTheFlow:
    """A session whose every surviving fact is literal-valued
    (``relation_type="attribute"`` — phone/email/date/certification/job
    title) is NOT routed off the relation surface at the flow level (Unit
    4): the model tags these facts at extraction time and they stay
    ``Relation`` objects all the way through ``rebuild`` — the diversion
    onto the subject node's ``attributes`` dict happens downstream, at
    ``GraphMerger.merge`` (see ``tests/graph/test_merger_attribute_gate.py``),
    not in this flow. So ``kept_relations`` is non-empty and the all-dropped
    recovery net never fires for an attribute-only session.
    """

    def _scope(self):
        return CloudScope(
            reverse={},
            cloud_bindings={},
            observed=frozenset(),
            resolution={},
            core_resolution={},
            declared=frozenset(),
        )

    def _ctx(self) -> StageContext:
        return StageContext(
            model=None,
            tokenizer=None,
            transcript="[user] My email is alex@example.com and I use ROS2.",
            session_id="s0",
            speaker_id="speaker0",
            speaker_name=None,
            temperature=0.0,
            max_tokens=64,
            plausibility_max_tokens=64,
            prompts_dir=None,
            system_prompt_filename="extraction_system.txt",
            user_prompt_filename="extraction.txt",
            model_alias=None,
            seed=None,
            timestamp=None,
            source_type="transcript",
            validate=True,
            cloud_enabled=True,
            enrichment_provider="anthropic",
            enrichment_provider_model="claude-sonnet-4-6",
            enrichment_provider_endpoint=None,
            plausibility_judge="off",
            plausibility_stage="deanon",
            plausibility_model="claude-sonnet-4-6",
            plausibility_endpoint=None,
            scrub=frozenset({"person name"}),
            correction_entity_types=None,
        )

    def _attribute_facts(self) -> list[dict]:
        return [
            {
                "subject": "Alex",
                "predicate": "has_email",
                "object": "alex@example.com",
                "relation_type": "attribute",
            },
        ]

    def _graph(self) -> SessionGraph:
        return SessionGraph(
            session_id="s0",
            timestamp="2026-07-21T00:00:00Z",
            entities=[Entity(name="Alex", entity_type="person")],
            relations=[],
        )

    def _run(self):
        ctx = self._ctx()
        graph = self._graph()
        state = StageState(
            graph=graph,
            facts=self._attribute_facts(),
            scope=self._scope(),
            original_relation_count=0,
        )
        with extraction_trace():
            return _stage_rebuild(ctx, state)

    def test_attribute_relation_kept_not_dropped(self):
        state = self._run()
        assert len(state.graph.relations) == 1
        assert state.graph.relations[0].relation_type == "attribute"
        assert state.graph.relations[0].object == "alex@example.com"

    def test_no_fallback_is_invoked(self):
        with patch("paramem.graph.flows._fallback_plausibility_on_raw") as fallback:
            self._run()
        fallback.assert_not_called()

    def test_all_dropped_cause_not_recorded(self):
        state = self._run()
        assert "all_dropped_cause" not in state.graph.diagnostics


class TestFallbackRebuildRecordsValidationDrops:
    """The all-dropped / anon-failed recovery path builds its relations
    through the SAME ``build_relations`` the ``rebuild`` stage uses, so a
    schema-validation failure there is RECORDED.

    It used to have its own construction loop with a bare
    ``except: continue`` — two standing violations at once (a second
    implementation of the same logic, and error suppression), and a
    validation failure on the recovery path was invisible.
    """

    def _graph(self) -> SessionGraph:
        return SessionGraph(
            session_id="s0",
            timestamp="2026-07-21T00:00:00Z",
            entities=[
                Entity(name="Alex", entity_type="person"),
                Entity(name="Millfield", entity_type="place"),
                Entity(name="Berlin", entity_type="place"),
            ],
            relations=[
                Relation(
                    subject="Alex",
                    predicate="lives_in",
                    object="Millfield",
                    relation_type="factual",
                    confidence=1.0,
                    speaker_id="speaker0",
                ),
                Relation(
                    subject="Alex",
                    predicate="born_in",
                    object="Berlin",
                    relation_type="factual",
                    confidence=1.0,
                    speaker_id="speaker0",
                ),
            ],
        )

    def test_out_of_schema_relation_type_is_dropped_and_recorded(self):
        """A judge that hands back a fact with a ``relation_type`` outside
        the schema's Literal set: the fact cannot become a ``Relation``,
        and the drop lands in diagnostics instead of vanishing."""
        graph = self._graph()
        judged = [
            {
                "subject": "Alex",
                "predicate": "lives_in",
                "object": "Millfield",
                "relation_type": "factual",
            },
            {
                "subject": "Alex",
                "predicate": "born_in",
                "object": "Berlin",
                "relation_type": "not_a_real_type",
            },
        ]
        with patch(
            "paramem.graph.extractor.judge_plausibility",
            return_value=(judged, "raw"),
        ):
            out = _fallback_plausibility_on_raw(
                graph,
                "[user] I live in Millfield.",
                MagicMock(),
                MagicMock(),
                "all_dropped",
                speaker_id="speaker0",
            )
        assert [(r.subject, r.object) for r in out.relations] == [("Alex", "Millfield")]
        dropped = out.diagnostics["pydantic_validation_dropped"]
        assert len(dropped) == 1
        assert dropped[0]["object"] == "Berlin"
        assert dropped[0]["relation_type"] == "not_a_real_type"
        assert dropped[0]["reason"]
        # Entities are pruned to the surviving endpoints, as before.
        assert {e.name for e in out.entities} == {"Alex", "Millfield"}
        assert out.diagnostics["fallback_path"] == "all_dropped"

    def test_no_drops_records_nothing(self):
        """The recording is drop-triggered: a clean rebuild leaves the
        diagnostics key absent rather than writing an empty list."""
        graph = self._graph()
        out = _fallback_plausibility_on_raw(
            graph,
            "[user] I live in Millfield.",
            None,
            None,
            "anon_failed",
            speaker_id="speaker7",
        )
        assert len(out.relations) == 2
        assert {r.speaker_id for r in out.relations} == {"speaker7"}
        assert "pydantic_validation_dropped" not in out.diagnostics
