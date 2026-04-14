"""Tests for the extraction pipeline — STT correction, HA validation, noise filter, JSON parsing."""

import json
from unittest.mock import MagicMock, patch

import pytest

from paramem.graph.extractor import (
    _correct_entity_names,
    _extract_json_block,
    _find_correction,
    _levenshtein,
    _validate_with_ha_context,
)
from paramem.graph.schema import Entity, Relation, SessionGraph


def _make_graph(relations, entities=None):
    """Helper to create a SessionGraph with relations."""
    if entities is None:
        names = set()
        for r in relations:
            names.add(r[0])
            names.add(r[2])
        entities = [Entity(name=n, entity_type="concept") for n in names]
    rels = [
        Relation(
            subject=r[0],
            predicate=r[1],
            object=r[2],
            relation_type=r[3] if len(r) > 3 else "factual",
            confidence=r[4] if len(r) > 4 else 1.0,
        )
        for r in relations
    ]
    return SessionGraph(
        session_id="test",
        timestamp="2026-04-09T00:00:00Z",
        entities=entities,
        relations=rels,
    )


# --- Levenshtein ---


class TestLevenshtein:
    def test_identical(self):
        assert _levenshtein("hello", "hello") == 0

    def test_one_char_diff(self):
        assert _levenshtein("hello", "hallo") == 1

    def test_insertion(self):
        assert _levenshtein("hello", "helloo") == 1

    def test_deletion(self):
        assert _levenshtein("hello", "helo") == 1

    def test_one_substitution(self):
        assert _levenshtein("dinslaker", "dinslaken") == 1

    def test_two_changes(self):
        assert _levenshtein("dinslager", "dinslaken") == 2

    def test_empty(self):
        assert _levenshtein("", "abc") == 3
        assert _levenshtein("abc", "") == 3

    def test_completely_different(self):
        assert _levenshtein("abc", "xyz") == 3


# --- STT Correction ---


class TestFindCorrection:
    def test_close_match(self):
        assert _find_correction("Frankford", {"Frankfurt", "Frankfurt"}) == "Frankfurt"

    def test_exact_match_no_correction(self):
        assert _find_correction("Frankfurt", {"Frankfurt", "Berlin"}) is None

    def test_no_match(self):
        assert _find_correction("Tokyo", {"Frankfurt", "Berlin"}) is None

    def test_too_far(self):
        assert _find_correction("Paris", {"Frankfurt"}) is None

    def test_case_insensitive(self):
        assert _find_correction("millfeld", {"Millfield"}) == "Millfield"


class TestCorrectEntityNames:
    def test_corrects_from_assistant_response(self):
        graph = _make_graph([("Alex", "parents_live_in", "Frankford")])
        transcript = "[user] My parents live in Frankford.\n[assistant] Frankfurt is about 300km."
        result = _correct_entity_names(graph, transcript)
        assert result.relations[0].object == "Frankfurt"

    def test_no_correction_when_exact(self):
        graph = _make_graph([("Alex", "lives_in", "Frankfurt")])
        transcript = "[user] I live in Frankfurt.\n[assistant] Frankfurt is nice."
        result = _correct_entity_names(graph, transcript)
        assert result.relations[0].object == "Frankfurt"

    def test_no_assistant_response(self):
        graph = _make_graph([("Alex", "lives_in", "Kelkham")])
        transcript = "[user] I live in Kelkham."
        result = _correct_entity_names(graph, transcript)
        # No assistant tokens to correct from
        assert result.relations[0].object == "Kelkham"

    def test_short_words_skipped(self):
        graph = _make_graph([("Alex", "likes", "Tea")])
        transcript = "[user] I like tea.\n[assistant] Tea is great."
        result = _correct_entity_names(graph, transcript)
        # "Tea" is < 4 chars, skipped in assistant token extraction
        assert result.relations[0].object == "Tea"


# --- HA Context Validation ---


class TestHAContextValidation:
    def test_home_location_boosts_confidence(self):
        graph = _make_graph([("Alex", "lives_in", "Millfield", "factual", 0.7)])
        ha_context = {
            "location_name": "Millfield",
            "timezone": "Europe/Berlin",
            "latitude": 50.1,
            "longitude": 8.4,
            "zones": ["Home"],
            "areas": ["Living Room", "Office"],
        }
        result = _validate_with_ha_context(graph, ha_context)
        assert result.relations[0].confidence == 1.0

    def test_partial_match_boosts(self):
        graph = _make_graph([("Alex", "lives_in", "Millfield/Greenshire", "factual", 0.7)])
        ha_context = {
            "location_name": "Millfield",
            "timezone": "",
            "latitude": 0,
            "longitude": 0,
            "zones": [],
            "areas": [],
        }
        result = _validate_with_ha_context(graph, ha_context)
        assert result.relations[0].confidence == 1.0

    def test_zone_match_boosts(self):
        graph = _make_graph([("Alex", "lives_near", "Home", "factual", 0.5)])
        ha_context = {
            "location_name": "",
            "timezone": "",
            "latitude": 0,
            "longitude": 0,
            "zones": ["Home", "Work"],
            "areas": [],
        }
        result = _validate_with_ha_context(graph, ha_context)
        assert result.relations[0].confidence >= 0.9

    def test_no_match_unchanged(self):
        graph = _make_graph([("Alex", "lives_in", "Tokyo", "factual", 0.7)])
        ha_context = {
            "location_name": "Millfield",
            "timezone": "",
            "latitude": 0,
            "longitude": 0,
            "zones": [],
            "areas": [],
        }
        result = _validate_with_ha_context(graph, ha_context)
        assert result.relations[0].confidence == 0.7

    def test_non_location_predicate_unchanged(self):
        graph = _make_graph([("Alex", "works_at", "Millfield", "factual", 0.7)])
        ha_context = {
            "location_name": "Millfield",
            "timezone": "",
            "latitude": 0,
            "longitude": 0,
            "zones": [],
            "areas": [],
        }
        result = _validate_with_ha_context(graph, ha_context)
        assert result.relations[0].confidence == 0.7

    def test_empty_context(self):
        graph = _make_graph([("Alex", "lives_in", "Millfield", "factual", 0.7)])
        ha_context = {
            "location_name": "",
            "timezone": "",
            "latitude": 0,
            "longitude": 0,
            "zones": [],
            "areas": [],
        }
        result = _validate_with_ha_context(graph, ha_context)
        assert result.relations[0].confidence == 0.7


# --- JSON Block Extraction ---


class TestExtractJsonBlock:
    def test_object(self):
        text = 'Some text {"key": "value"} more text'
        result = json.loads(_extract_json_block(text))
        assert result == {"key": "value"}

    def test_array(self):
        text = 'Some text [{"a": 1}, {"b": 2}] more text'
        result = json.loads(_extract_json_block(text))
        assert result == [{"a": 1}, {"b": 2}]

    def test_array_before_object(self):
        text = '[1, 2] {"key": "value"}'
        result = json.loads(_extract_json_block(text))
        assert result == [1, 2]

    def test_markdown_code_block(self):
        text = '```json\n{"key": "value"}\n```'
        result = json.loads(_extract_json_block(text))
        assert result == {"key": "value"}

    def test_nested_object(self):
        text = '{"outer": {"inner": 1}}'
        result = json.loads(_extract_json_block(text))
        assert result == {"outer": {"inner": 1}}

    def test_empty_array(self):
        text = "Result: []"
        result = json.loads(_extract_json_block(text))
        assert result == []

    def test_no_json_raises(self):
        with pytest.raises(ValueError):
            _extract_json_block("no json here")


# --- SOTA Noise Filter ---


class TestSOTANoiseFilter:
    def test_filter_function_exists(self):
        from paramem.graph.extractor import _filter_with_sota

        assert callable(_filter_with_sota)

    def test_filter_with_sota_no_api_key(self):
        from paramem.graph.extractor import _sota_pipeline

        graph = _make_graph(
            [("Alex", "lives_in", "Millfield")],
            entities=[
                Entity(name="Alex", entity_type="person"),
                Entity(name="Millfield", entity_type="place"),
            ],
        )
        # No ANTHROPIC_API_KEY → skips gracefully
        with patch.dict("os.environ", {}, clear=True):
            result = _sota_pipeline(graph, "transcript", None, None)
            # Should return original graph unchanged
            assert len(result.relations) == 1

    def test_anonymize_graceful_on_bad_output(self):
        from paramem.graph.extractor import _anonymize_with_local_model

        graph = _make_graph(
            [("Alex", "lives_in", "Millfield")],
            entities=[
                Entity(name="Alex", entity_type="person"),
                Entity(name="Millfield", entity_type="place"),
            ],
        )
        model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.apply_chat_template = MagicMock(return_value="formatted")
        with (
            patch("paramem.evaluation.recall.generate_answer", return_value="not json"),
            patch("paramem.models.loader.adapt_messages", return_value=[]),
        ):
            result, mapping, anon_transcript = _anonymize_with_local_model(graph, model, tokenizer)
        assert result is None
        assert anon_transcript == ""

    def test_pipeline_anonymize_failure_returns_original(self):
        """If anonymization fails, the original graph is returned unchanged."""
        from paramem.graph.extractor import _sota_pipeline

        graph = _make_graph(
            [("Alex", "lives_in", "Millfield")],
            entities=[
                Entity(name="Alex", entity_type="person"),
                Entity(name="Millfield", entity_type="place"),
            ],
        )
        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.extractor._anonymize_with_local_model",
                return_value=(None, {}, ""),
            ),
        ):
            result = _sota_pipeline(graph, "transcript", None, None)
        assert len(result.relations) == 1
        assert result.relations[0].subject == "Alex"

    def test_pipeline_enrichment_failure_keeps_anon_facts(self):
        """If enrichment fails, the anonymized facts pass through to de-anonymization."""
        from paramem.graph.extractor import _sota_pipeline

        graph = _make_graph(
            [("Alex", "lives_in", "Millfield")],
            entities=[
                Entity(name="Alex", entity_type="person"),
                Entity(name="Millfield", entity_type="place"),
            ],
        )
        anon_facts = [{"subject": "Person_1", "predicate": "lives_in", "object": "City_1"}]
        mapping = {"Alex": "Person_1", "Millfield": "City_1"}

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.extractor._anonymize_with_local_model",
                return_value=(anon_facts, mapping, ""),
            ),
            patch("paramem.graph.extractor._filter_with_sota", return_value=(None, None, None)),
        ):
            result = _sota_pipeline(graph, "transcript", None, None)

        # Enrichment failed → anon_facts used → de-anonymized to real names
        assert len(result.relations) == 1
        assert result.relations[0].subject == "Alex"
        assert result.relations[0].object == "Millfield"

    def test_pipeline_enriched_facts_get_deanonymized(self):
        """Enrichment output flows through de-anonymization to real names."""
        from paramem.graph.extractor import _sota_pipeline

        graph = _make_graph(
            [("Alex", "lives_in", "Millfield")],
            entities=[
                Entity(name="Alex", entity_type="person"),
                Entity(name="Millfield", entity_type="place"),
            ],
        )
        anon_facts = [{"subject": "Person_1", "predicate": "lives_in", "object": "City_1"}]
        enriched_anon = anon_facts + [
            {"subject": "Person_1", "predicate": "born_in", "object": "City_1"}
        ]
        mapping = {"Alex": "Person_1", "Millfield": "City_1"}

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.extractor._anonymize_with_local_model",
                return_value=(anon_facts, mapping, ""),
            ),
            patch(
                "paramem.graph.extractor._filter_with_sota",
                return_value=(enriched_anon, None, None),
            ),
        ):
            result = _sota_pipeline(graph, "transcript", None, None)

        # Both enriched relations survive and get de-anonymized
        assert len(result.relations) == 2
        predicates = {r.predicate for r in result.relations}
        assert predicates == {"lives_in", "born_in"}
        for r in result.relations:
            assert r.subject == "Alex"
            assert r.object == "Millfield"

    def test_local_plausibility_filter_round_trip(self):
        """Local plausibility filter parses the standard array response."""
        from paramem.graph.extractor import _local_plausibility_filter

        facts = [
            {"subject": "Alex", "predicate": "lives_in", "object": "Millfield"},
            {"subject": "Alex", "predicate": "has_name", "object": "Alex"},  # self-loop
        ]
        # Model output: drops the self-loop, keeps the valid fact
        kept_response = '[{"subject": "Alex", "predicate": "lives_in", "object": "Millfield"}]'
        tokenizer = MagicMock()
        tokenizer.apply_chat_template = MagicMock(return_value="formatted")
        with (
            patch("paramem.evaluation.recall.generate_answer", return_value=kept_response),
            patch("paramem.models.loader.adapt_messages", return_value=[]),
        ):
            result = _local_plausibility_filter(facts, "transcript", MagicMock(), tokenizer)
        assert result is not None
        assert len(result) == 1
        assert result[0]["predicate"] == "lives_in"

    def test_verify_anonymization_catches_leak(self):
        """Forward-path guard detects a real name leaking past anonymization."""
        from paramem.graph.extractor import verify_anonymization_completeness

        graph = _make_graph(
            [("Alex", "lives_in", "Millfield")],
            entities=[
                Entity(name="Alex", entity_type="person"),
                Entity(name="Millfield", entity_type="place"),
            ],
        )
        # Anonymizer forgot "Millfield" — mapping is incomplete ({real: placeholder})
        mapping = {"Alex": "Person_1"}
        anon_facts = [{"subject": "Person_1", "predicate": "lives_in", "object": "Millfield"}]
        anon_transcript = "Person_1 moved to Millfield last year."
        leaked = verify_anonymization_completeness(graph, mapping, anon_facts, anon_transcript)
        assert "Millfield" in leaked
        assert "Alex" not in leaked  # Alex was properly replaced

    def test_verify_anonymization_clean_mapping(self):
        """Guard returns empty list when anonymization is complete."""
        from paramem.graph.extractor import verify_anonymization_completeness

        graph = _make_graph(
            [("Alex", "lives_in", "Millfield")],
            entities=[
                Entity(name="Alex", entity_type="person"),
                Entity(name="Millfield", entity_type="place"),
            ],
        )
        # {real: placeholder} — production direction.
        mapping = {"Alex": "Person_1", "Millfield": "City_1"}
        anon_facts = [{"subject": "Person_1", "predicate": "lives_in", "object": "City_1"}]
        anon_transcript = "Person_1 moved to City_1 last year."
        leaked = verify_anonymization_completeness(graph, mapping, anon_facts, anon_transcript)
        assert leaked == []

    def test_normalize_anonymization_mapping_inverts_placeholder_keys(self):
        """Mapping with placeholder keys is inverted to {real: placeholder} canonical."""
        from paramem.graph.extractor import _normalize_anonymization_mapping

        wrong_direction = {"Person_1": "Alex", "City_1": "Millfield"}
        normalized, stats = _normalize_anonymization_mapping(wrong_direction)
        assert normalized == {"Alex": "Person_1", "Millfield": "City_1"}
        assert stats == {"inverted": 2, "dropped": 0}

    def test_normalize_anonymization_mapping_keeps_canonical(self):
        """Mapping already in {real: placeholder} canonical form passes through."""
        from paramem.graph.extractor import _normalize_anonymization_mapping

        canonical = {"Alex": "Person_1", "Millfield": "City_1"}
        normalized, stats = _normalize_anonymization_mapping(canonical)
        assert normalized == canonical
        assert stats == {"inverted": 0, "dropped": 0}

    def test_normalize_anonymization_mapping_empty(self):
        from paramem.graph.extractor import _normalize_anonymization_mapping

        normalized, stats = _normalize_anonymization_mapping({})
        assert normalized == {}
        assert stats == {"inverted": 0, "dropped": 0}

    def test_clean_ner_span_strips_dialogue_tail(self):
        """NER span cleanup removes 'Name: Response' dialogue artifacts."""
        from paramem.graph.extractor import _clean_ner_span

        assert _clean_ner_span("Li Ming: True") == "Li Ming"
        assert _clean_ner_span("Li Yu: Indeed") == "Li Yu"
        assert _clean_ner_span("Alex: Yes I agree") == "Alex"

    def test_clean_ner_span_strips_possessive(self):
        """NER span cleanup removes trailing possessive 's."""
        from paramem.graph.extractor import _clean_ner_span

        assert _clean_ner_span("Eugene Mekinen's") == "Eugene Mekinen"
        assert _clean_ner_span("Alex\u2019s") == "Alex"

    def test_clean_ner_span_keeps_clean(self):
        """Names without dialogue tails or possessives pass through unchanged."""
        from paramem.graph.extractor import _clean_ner_span

        assert _clean_ner_span("Alex") == "Alex"
        assert _clean_ner_span("Li Ming") == "Li Ming"
        assert _clean_ner_span("New York City") == "New York City"

    def test_verify_anonymization_catches_missing_mapping(self):
        """Guard catches silent de-anonymization failure (name replaced but mapping incomplete)."""
        from paramem.graph.extractor import verify_anonymization_completeness

        graph = _make_graph(
            [("Alex", "lives_in", "Millfield")],
            entities=[
                Entity(name="Alex", entity_type="person"),
                Entity(name="Millfield", entity_type="place"),
            ],
        )
        # Anonymizer replaced "Alex" with "Person_1" everywhere — but forgot to
        # include the mapping entry. Real name doesn't leak in output, but
        # de-anonymization would silently emit "Person_1" as the final subject.
        # {real: placeholder} — mapping has Millfield but is missing Alex.
        mapping = {"Millfield": "City_1"}
        anon_facts = [{"subject": "Person_1", "predicate": "lives_in", "object": "City_1"}]
        anon_transcript = "Person_1 moved to City_1."
        problems = verify_anonymization_completeness(graph, mapping, anon_facts, anon_transcript)
        assert "Alex" in problems
        assert "Millfield" not in problems  # properly mapped

    def test_verify_anonymization_case_insensitive(self):
        """Guard catches case-different leaks (Alex vs alex)."""
        from paramem.graph.extractor import verify_anonymization_completeness

        graph = _make_graph(
            [("Alex", "lives_in", "Millfield")],
            entities=[
                Entity(name="Alex", entity_type="person"),
                Entity(name="Millfield", entity_type="place"),
            ],
        )
        # {real: placeholder} — complete mapping but transcript leaks "alex" (lowercase).
        mapping = {"Alex": "Person_1", "Millfield": "City_1"}
        anon_facts = [{"subject": "Person_1", "predicate": "lives_in", "object": "City_1"}]
        anon_transcript = "the speaker alex moved to City_1."
        leaked = verify_anonymization_completeness(graph, mapping, anon_facts, anon_transcript)
        assert "Alex" in leaked

    def test_pipeline_repairs_missed_leak_then_calls_sota(self):
        """Leaked name present in transcript is auto-repaired; SOTA gets clean data."""
        from paramem.graph.extractor import _sota_pipeline

        graph = _make_graph(
            [("Alex", "lives_in", "Millfield")],
            entities=[
                Entity(name="Alex", entity_type="person"),
                Entity(name="Millfield", entity_type="place"),
            ],
        )
        anon_facts = [{"subject": "Person_1", "predicate": "lives_in", "object": "Millfield"}]
        mapping = {"Alex": "Person_1"}
        anon_transcript = "Person_1 lives in Millfield."
        filter_calls = []

        def fake_filter(facts, *args, **kwargs):
            filter_calls.append(list(facts))
            return facts, None, None

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.extractor._anonymize_with_local_model",
                return_value=(anon_facts, mapping, anon_transcript),
            ),
            patch("paramem.graph.extractor._filter_with_sota", side_effect=fake_filter),
        ):
            result = _sota_pipeline(graph, "Alex lives in Millfield.", None, None)

        assert len(filter_calls) == 1, "SOTA was not called after repair"
        # The call payload must not contain any real name.
        payload = filter_calls[0]
        assert all("Millfield" not in str(v) for f in payload for v in f.values())
        # Final graph is de-anonymized correctly.
        assert len(result.relations) == 1
        assert result.relations[0].subject == "Alex"
        assert result.relations[0].object == "Millfield"

    def test_pipeline_drops_hallucinated_leak(self):
        """A leaked name NOT in the transcript is classified hallucinated and dropped."""
        from paramem.graph.extractor import _sota_pipeline

        graph = _make_graph(
            [("Alex", "knows", "Ghost")],
            entities=[
                Entity(name="Alex", entity_type="person"),
                Entity(name="Ghost", entity_type="person"),
            ],
        )
        anon_facts = [{"subject": "Person_1", "predicate": "knows", "object": "Ghost"}]
        mapping = {"Alex": "Person_1"}
        anon_transcript = "Person_1 mentioned something."
        filter_calls = []

        def fake_filter(facts, *args, **kwargs):
            filter_calls.append(list(facts))
            return facts, None, None

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.extractor._anonymize_with_local_model",
                return_value=(anon_facts, mapping, anon_transcript),
            ),
            patch("paramem.graph.extractor._filter_with_sota", side_effect=fake_filter),
        ):
            result = _sota_pipeline(graph, "Alex mentioned something.", None, None)

        # The hallucinated triple is dropped from anon_facts; guard still flags
        # the missing mapping, so SOTA is skipped. Graph is returned unchanged —
        # callers run plausibility downstream to clean the hallucination.
        assert filter_calls == []
        assert len(result.relations) == 1
        assert result.relations[0].object == "Ghost"

    def test_pipeline_normalizes_mixed_direction_mapping_per_pair(self):
        """Mixed-direction mappings from the anonymizer are normalized per-pair.

        Real anonymizers sometimes emit a dict where some entries are
        `{real: placeholder}` and others are `{placeholder: real}`. Per-pair
        normalization independently canonicalizes each, enabling the pipeline
        to proceed (SOTA call, de-anonymization) rather than aborting.
        """
        from paramem.graph.extractor import _normalize_anonymization_mapping

        mixed = {
            "Alex": "Person_1",  # canonical
            "Person_2": "Millfield",  # inverted
        }
        out, stats = _normalize_anonymization_mapping(mixed)
        # Both pairs end up canonical: keys are real, values are placeholders.
        assert out == {"Alex": "Person_1", "Millfield": "Person_2"}
        assert stats == {"inverted": 1, "dropped": 0}


class TestSOTAEntityBindings:
    def test_extract_sota_bindings_basic(self):
        """Braced placeholders in the updated transcript yield real-span bindings."""
        from paramem.graph.extractor import _extract_sota_bindings

        old = "Person_1 organized a community fitness event at the local gym."
        new = "Person_1 organized {Event_1} at {Location_1}."
        bindings = _extract_sota_bindings(old, new)
        assert "a community fitness event" in bindings
        assert bindings["a community fitness event"] == "Event_1"
        assert bindings["the local gym"] == "Location_1"

    def test_extract_sota_bindings_no_changes(self):
        """Identical transcripts produce no bindings."""
        from paramem.graph.extractor import _extract_sota_bindings

        t = "Person_1 lives in City_1."
        assert _extract_sota_bindings(t, t) == {}

    def test_strip_placeholder_braces(self):
        """Braces are removed from subject/object; other fields untouched."""
        from paramem.graph.extractor import _strip_placeholder_braces

        facts = [
            {"subject": "{Event_1}", "predicate": "located_at", "object": "{Location_1}"},
            {"subject": "Person_1", "predicate": "organized", "object": "{Event_1}"},
        ]
        out = _strip_placeholder_braces(facts)
        assert out[0]["subject"] == "Event_1"
        assert out[0]["object"] == "Location_1"
        assert out[1]["subject"] == "Person_1"
        assert out[1]["object"] == "Event_1"

    def test_strip_placeholder_braces_inline(self):
        """Inline braced tokens inside longer strings are also stripped."""
        from paramem.graph.extractor import _strip_placeholder_braces

        facts = [
            {"subject": "Person_1", "predicate": "attended", "object": "meeting at {Event_1}"},
            {"subject": "{Person_2}'s cousin", "predicate": "visited", "object": "home"},
        ]
        out = _strip_placeholder_braces(facts)
        assert out[0]["object"] == "meeting at Event_1"
        assert out[1]["subject"] == "Person_2's cousin"

    def test_strip_residual_placeholders_catches_bare_and_composite(self):
        """Residual sweep drops facts with any placeholder-shaped token, bare or composite."""
        from paramem.graph.extractor import _strip_residual_placeholders

        facts = [
            {"subject": "Alice", "predicate": "knows", "object": "Bob"},  # clean
            {"subject": "Alice", "predicate": "supports", "object": "Person_2"},  # bare
            {"subject": "Alice", "predicate": "values", "object": "Person_2's Support"},  # embedded
            {"subject": "{Topic_1}", "predicate": "related_to", "object": "Bob"},  # braced
        ]
        kept, dropped = _strip_residual_placeholders(facts)
        assert len(dropped) == 3
        assert len(kept) == 1
        assert kept[0]["object"] == "Bob"


class TestPlausibilityTupleReturn:
    def test_plausibility_with_sota_returns_facts_and_raw(self):
        """_plausibility_filter_with_sota returns (facts, raw_response)."""
        from paramem.graph.extractor import _plausibility_filter_with_sota

        fake_raw = (
            '[{"subject":"A","predicate":"knows","object":"B",'
            '"relation_type":"social","confidence":1.0}]'
        )
        with patch("paramem.graph.extractor._sota_call", return_value=fake_raw):
            facts, raw = _plausibility_filter_with_sota(
                [{"subject": "A", "predicate": "knows", "object": "B"}],
                api_key="k",
                provider="anthropic",
                anon_transcript="A knows B.",
            )
        assert facts == [
            {
                "subject": "A",
                "predicate": "knows",
                "object": "B",
                "relation_type": "social",
                "confidence": 1.0,
            }
        ]
        assert raw == fake_raw

    def test_plausibility_with_sota_none_on_api_failure(self):
        """API failure returns (None, None) — callers must destructure both."""
        from paramem.graph.extractor import _plausibility_filter_with_sota

        with patch("paramem.graph.extractor._sota_call", return_value=None):
            facts, raw = _plausibility_filter_with_sota(
                [],
                api_key="k",
                provider="anthropic",
            )
        assert facts is None
        assert raw is None


class TestGroundingGate:
    def test_known_name_is_grounded_without_transcript_mention(self):
        from paramem.graph.extractor import _entity_is_grounded

        known = {"Alex"}
        # "Alex" not in transcript but IS in known real-names.
        assert _entity_is_grounded("Alex", "the weather is nice", known)

    def test_transcript_grounded_entity_passes(self):
        from paramem.graph.extractor import _entity_is_grounded

        t = "i drove to munich airport this morning"
        assert _entity_is_grounded("Munich Airport", t, set())
        assert _entity_is_grounded("munich_airport", t, set())

    def test_world_knowledge_inference_rejected(self):
        """SOTA infers CIA from a mention of Langley — CIA has no transcript grounding."""
        from paramem.graph.extractor import _entity_is_grounded

        t = "i'm heading to langley tomorrow"
        assert _entity_is_grounded("Langley", t, set())
        assert not _entity_is_grounded("CIA", t, set())
        assert not _entity_is_grounded("Central Intelligence Agency", t, set())

    def test_drop_ungrounded_facts_catches_inferences(self):
        from paramem.graph.extractor import _drop_ungrounded_facts

        transcript = "Alex is heading to Langley tomorrow."
        known = {"Alex"}
        facts = [
            {"subject": "Alex", "predicate": "travels_to", "object": "Langley"},  # OK
            {"subject": "Alex", "predicate": "works_at", "object": "CIA"},  # drop
            {"subject": "CIA", "predicate": "headquartered_in", "object": "Langley"},  # drop
        ]
        kept, dropped = _drop_ungrounded_facts(facts, transcript, known)
        assert len(kept) == 1
        assert kept[0]["object"] == "Langley"
        assert len(dropped) == 2

    def test_empty_transcript_with_empty_mapping_drops_all(self):
        """Degenerate inputs: no transcript, no known names → everything dropped."""
        from paramem.graph.extractor import _drop_ungrounded_facts

        facts = [{"subject": "Alex", "predicate": "knows", "object": "Bob"}]
        kept, dropped = _drop_ungrounded_facts(facts, "", set())
        assert kept == []
        assert dropped == facts

    def test_empty_transcript_with_known_mapping_keeps_known(self):
        """Known real names pass even without a transcript."""
        from paramem.graph.extractor import _drop_ungrounded_facts

        facts = [{"subject": "Alex", "predicate": "knows", "object": "Bob"}]
        kept, dropped = _drop_ungrounded_facts(facts, "", {"Alex", "Bob"})
        assert kept == facts
        assert dropped == []

    def test_short_name_not_matched_as_substring(self):
        """Short names like 'Li' must not match inside longer words (e.g. 'Libya')."""
        from paramem.graph.extractor import _entity_is_grounded

        # "Libya" contains the substring "li", but whole-word match must reject it.
        assert not _entity_is_grounded("Li", "i travelled to libya last year", set())
        # When "Li" is genuinely present as a whole word, it's accepted.
        assert _entity_is_grounded("Li", "i spoke with li yesterday", set())


class TestSpeakerContextInjection:
    def test_build_speaker_context_empty_when_unknown(self):
        from paramem.graph.extractor import build_speaker_context

        assert build_speaker_context(None) == ""
        assert build_speaker_context("") == ""

    def test_build_speaker_context_includes_name_and_directive(self):
        from paramem.graph.extractor import build_speaker_context

        out = build_speaker_context("Ye Jie")
        assert "Ye Jie" in out
        assert "'Ye Jie'" in out
        # Must forbid generic fallback strings so the model cannot emit them.
        for forbidden in ("Speaker", "User", "'I'"):
            assert forbidden in out, f"directive must mention {forbidden!r}"


# --- Background Trainer ---


class TestBackgroundTrainer:
    def test_init(self):
        from paramem.server.background_trainer import BackgroundTrainer

        bt = BackgroundTrainer(
            model=MagicMock(),
            tokenizer=MagicMock(),
            training_config=MagicMock(),
        )
        assert not bt.is_training

    def test_start_jobs_empty(self):
        from paramem.server.background_trainer import BackgroundTrainer

        bt = BackgroundTrainer(
            model=MagicMock(),
            tokenizer=MagicMock(),
            training_config=MagicMock(),
        )
        completed = []
        bt.start_jobs([], on_complete=lambda: completed.append(True))
        assert completed == [True]
        assert not bt.is_training

    def test_stop_when_not_training(self):
        from paramem.server.background_trainer import BackgroundTrainer

        bt = BackgroundTrainer(
            model=MagicMock(),
            tokenizer=MagicMock(),
            training_config=MagicMock(),
        )
        epoch = bt.stop()
        assert epoch == 0

    def test_pause_when_not_training(self):
        from paramem.server.background_trainer import BackgroundTrainer

        bt = BackgroundTrainer(
            model=MagicMock(),
            tokenizer=MagicMock(),
            training_config=MagicMock(),
        )
        assert bt.pause() is True

    def test_resume_when_not_training(self):
        from paramem.server.background_trainer import BackgroundTrainer

        bt = BackgroundTrainer(
            model=MagicMock(),
            tokenizer=MagicMock(),
            training_config=MagicMock(),
        )
        # Should not raise
        bt.resume()


# --- Consolidation: collect_semantic_keys ---


class TestCollectSemanticKeys:
    def _make_loop_stub(self):
        class Stub:
            def __init__(self):
                self.indexed_key_qa = {
                    "graph1": {"key": "graph1", "question": "Q1", "answer": "A1"},
                    "graph2": {"key": "graph2", "question": "Q2", "answer": "A2"},
                    "proc1": {"key": "proc1", "question": "QP", "answer": "AP"},
                }
                self.semantic_simhash = {"graph1": 12345}

        from paramem.training.consolidation import ConsolidationLoop

        stub = Stub()
        stub._collect_semantic_keys = ConsolidationLoop._collect_semantic_keys.__get__(stub)
        return stub

    def test_collects_semantic_keys(self):
        stub = self._make_loop_stub()
        result = stub._collect_semantic_keys()
        assert len(result) == 1
        assert result[0]["key"] == "graph1"

    def test_empty_semantic(self):
        stub = self._make_loop_stub()
        stub.semantic_simhash = {}
        assert stub._collect_semantic_keys() == []

    def test_missing_qa_skipped(self):
        stub = self._make_loop_stub()
        stub.semantic_simhash = {"graph99": 99999}
        assert stub._collect_semantic_keys() == []


# --- Simulation mode ---


class TestSimulationMode:
    def test_save_simulation_results(self, tmp_path):
        from paramem.server.consolidation import _save_simulation_results

        loop = MagicMock()
        loop.merger.save_graph = MagicMock()

        config = MagicMock()
        config.debug_dir = tmp_path

        episodic_qa = [{"question": "Q", "answer": "A"}]
        procedural_rels = [{"subject": "S", "predicate": "P", "object": "O"}]

        _save_simulation_results(episodic_qa, procedural_rels, loop, config)

        # Check files were created
        sim_dirs = list(tmp_path.glob("sim_*"))
        assert len(sim_dirs) == 1
        sim_dir = sim_dirs[0]
        assert (sim_dir / "episodic_qa.json").exists()
        assert (sim_dir / "procedural_rels.json").exists()

        with open(sim_dir / "episodic_qa.json") as f:
            saved = json.load(f)
        assert len(saved) == 1
        assert saved[0]["question"] == "Q"
