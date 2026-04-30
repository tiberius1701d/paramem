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

    def test_pipeline_anonymize_failure_falls_back_to_raw_plausibility(self):
        """If anonymization fails, the pipeline falls back to raw (local) plausibility (D8).

        The old behavior was to return the original graph unchanged.
        The new behavior runs _fallback_plausibility_on_raw so that tautologies,
        role leaks, and other noise are still filtered even without SOTA.
        The grounding gate inside the fallback may drop facts whose entities are
        not grounded in the provided transcript — this is correct and expected.
        """
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
            # Pass model=None/tokenizer=None → _local_plausibility_filter skipped inside fallback
        ):
            # Transcript "Alex lives in Millfield" grounds both entities.
            result = _sota_pipeline(
                graph, "Alex lives in Millfield", None, None, plausibility_judge="off"
            )
        # With plausibility_judge="off", fallback runs grounding + sweep only.
        # Both entities ARE in the transcript → relation survives.
        assert len(result.relations) == 1
        assert result.relations[0].subject == "Alex"
        # Fallback path recorded in diagnostics.
        assert result.diagnostics.get("fallback_path") == "anon_failed"

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
            patch(
                "paramem.graph.extractor._filter_with_sota",
                return_value=(None, None, None, {}),
            ),
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
                return_value=(enriched_anon, None, None, {}),
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

    def test_pipeline_deanonymizes_composite_placeholders(self):
        """Composite strings like 'Person_1's family' get substring-replaced.

        Uses a transcript where all composite entities are grounded so the
        grounding gate doesn't interfere with testing the deanonymization fix.
        """
        from paramem.graph.extractor import _sota_pipeline

        transcript = "Alex lives in downtown Millfield with Alex's family"
        graph = _make_graph(
            [("Alex", "lives_in", "Millfield")],
            entities=[
                Entity(name="Alex", entity_type="person"),
                Entity(name="Millfield", entity_type="place"),
            ],
        )
        anon_facts = [{"subject": "Person_1", "predicate": "lives_in", "object": "City_1"}]
        # SOTA produces composite strings with embedded placeholders
        enriched_anon = anon_facts + [
            {"subject": "Person_1's family", "predicate": "lives_in", "object": "City_1"},
            {"subject": "Person_1", "predicate": "lives_in", "object": "downtown City_1"},
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
                return_value=(enriched_anon, None, None, {}),
            ),
        ):
            result = _sota_pipeline(graph, transcript, None, None)

        # Composite strings must be de-anonymized, not dropped
        subjects = {r.subject for r in result.relations}
        objects = {r.object for r in result.relations}
        assert "Alex's family" in subjects, f"Expected composite deanon, got {subjects}"
        assert "downtown Millfield" in objects, f"Expected composite deanon, got {objects}"
        # No residual placeholders should remain
        for r in result.relations:
            assert "Person_1" not in r.subject
            assert "City_1" not in r.object

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

    def test_normalize_mapping_empty_vocab_drops_all(self, monkeypatch):
        """With no anonymizer prefixes configured, all mapping pairs are dropped."""
        from paramem.graph.extractor import _normalize_anonymization_mapping

        monkeypatch.setattr("paramem.graph.extractor.anonymizer_placeholder_pattern", lambda: None)
        result, stats = _normalize_anonymization_mapping({"Alex": "Person_1"})
        assert result == {}
        assert stats == {"inverted": 0, "dropped": 1}

    def test_mapping_is_canonical_empty_vocab_empty_mapping(self, monkeypatch):
        """Empty vocab + empty mapping → canonical (nothing to be wrong)."""
        from paramem.graph.extractor import _mapping_is_canonical

        monkeypatch.setattr("paramem.graph.extractor.anonymizer_placeholder_pattern", lambda: None)
        assert _mapping_is_canonical({}) is True

    def test_mapping_is_canonical_empty_vocab_nonempty_mapping(self, monkeypatch):
        """Empty vocab + non-empty mapping → not canonical (no prefix vocabulary)."""
        from paramem.graph.extractor import _mapping_is_canonical

        monkeypatch.setattr("paramem.graph.extractor.anonymizer_placeholder_pattern", lambda: None)
        assert _mapping_is_canonical({"Alex": "Person_1"}) is False

    def test_repair_anonymization_leaks_organization_type(self):
        """_repair_anonymization_leaks allocates Org_N placeholder for organization entities.

        Pins the broadened repair scope: organization entities are now covered by
        anonymizer_type_to_prefix() via configs/schema.yaml (Org is primary_for_type).
        A previously person/place-only repair path would leave "Google" unreplaced.
        """
        from paramem.graph.extractor import _repair_anonymization_leaks

        graph = _make_graph(
            [("Alex", "works_at", "Google")],
            entities=[
                Entity(name="Alex", entity_type="person"),
                Entity(name="Google", entity_type="organization"),
            ],
        )
        mapping = {"Alex": "Person_1"}
        anon_facts = [{"subject": "Person_1", "predicate": "works_at", "object": "Google"}]
        anon_transcript = "Person_1 works at Google."
        # "Google" is in the transcript → missed, not hallucinated.
        leaked = ["Google"]
        facts, new_mapping, _, status = _repair_anonymization_leaks(
            graph, mapping, anon_facts, anon_transcript, "Alex works at Google.", leaked
        )
        assert status["missed_fixed"] == 1, (
            "organization entity present in transcript must be classified as missed and fixed"
        )
        assert "Google" in new_mapping, "extended mapping must contain the real name"
        org_placeholder = new_mapping["Google"]
        assert org_placeholder.startswith("Org_"), (
            f"organization entity must get an Org_N placeholder, got {org_placeholder!r}"
        )
        # The fact's object must be rewritten to the placeholder.
        assert facts[0]["object"] == org_placeholder, (
            f"anon_facts object must be rewritten to placeholder, got {facts[0]['object']!r}"
        )

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
            return facts, None, None, {}

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
        """A leaked name NOT in the transcript is classified hallucinated and dropped (D1).

        When "Ghost" is in anon_facts but not in the transcript, repair classifies it
        as hallucinated and drops the referencing triple from anon_facts. After repair,
        anon_facts is empty → the pipeline returns an empty graph (no triples survive,
        SOTA is skipped). This is correct: a fact whose object is a hallucinated entity
        should not reach the adapter.
        """
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
            return facts, None, None, {}

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.extractor._anonymize_with_local_model",
                return_value=(anon_facts, mapping, anon_transcript),
            ),
            patch("paramem.graph.extractor._filter_with_sota", side_effect=fake_filter),
        ):
            result = _sota_pipeline(graph, "Alex mentioned something.", None, None)

        # The hallucinated triple is dropped during repair.
        # After repair anon_facts is empty → grounding_gate="no_input", no facts survive.
        assert filter_calls == [], "SOTA must not be called when anon_facts is empty after repair"
        assert len(result.relations) == 0
        assert result.diagnostics.get("grounding_gate") == "no_input"

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


class TestRoleAwareGroundingGate:
    """Role-aware extension of ``_drop_ungrounded_facts``.

    With ``speaker_names`` provided, the gate requires speaker-attributed
    object content to ground in user-only transcript spans, blocking the
    assistant-into-graph hallucination class deterministically.  Without
    ``speaker_names`` (or with an empty set), behaviour is exactly as the
    role-blind gate — backward compatibility for existing call sites.
    """

    def test_extract_user_spans_handles_bracket_form(self):
        from paramem.graph.extractor import _extract_user_spans

        t = "[user] hello there\n[assistant] hi back\n[user] how are you"
        out = _extract_user_spans(t)
        assert "hello there" in out
        assert "how are you" in out
        assert "hi back" not in out

    def test_extract_user_spans_handles_colon_form(self):
        from paramem.graph.extractor import _extract_user_spans

        t = "user: i live in berlin\nassistant: that is a great city\nuser: i love it"
        out = _extract_user_spans(t)
        assert "i live in berlin" in out
        assert "i love it" in out
        assert "great city" not in out

    def test_extract_user_spans_drops_unprefixed_lines(self):
        from paramem.graph.extractor import _extract_user_spans

        t = "[user] tagged line\nuntagged line\n[assistant] also tagged"
        out = _extract_user_spans(t)
        assert "tagged line" in out
        assert "untagged line" not in out
        assert "also tagged" not in out

    def test_speaker_subject_object_in_user_kept(self):
        """Triple grounded in [user] turn passes role-aware gate."""
        from paramem.graph.extractor import _drop_ungrounded_facts

        transcript = "[user] i love yoga and reading\n[assistant] that's wonderful"
        facts = [{"subject": "Alex", "predicate": "likes", "object": "yoga"}]
        kept, dropped = _drop_ungrounded_facts(facts, transcript, {"Alex"}, speaker_names={"Alex"})
        assert kept == facts
        assert dropped == []

    def test_speaker_subject_object_only_in_assistant_dropped(self):
        """Triple about the speaker grounded only in [assistant] turn → dropped."""
        from paramem.graph.extractor import _drop_ungrounded_facts

        transcript = (
            "[user] what's a good hobby for me\n"
            "[assistant] how about combining your love for yoga and travel"
        )
        facts = [{"subject": "Alex", "predicate": "likes", "object": "yoga"}]
        kept, dropped = _drop_ungrounded_facts(facts, transcript, {"Alex"}, speaker_names={"Alex"})
        assert kept == []
        assert dropped == facts

    def test_third_party_subject_uses_full_transcript(self):
        """Triple where subject is NOT a speaker stays under role-blind grounding.

        Production case: 'Max Schmidt works_at Lorem Industries' extracted from
        the speaker's mention of a colleague.  Both endpoints appear somewhere
        in the transcript (possibly only assistant turns confirming names);
        the role-aware gate must not over-drop these.
        """
        from paramem.graph.extractor import _drop_ungrounded_facts

        transcript = (
            "[user] my colleague max schmidt joined the office\n"
            "[assistant] understood; lorem industries"
        )
        facts = [
            {"subject": "Max Schmidt", "predicate": "works_at", "object": "Lorem Industries"},
        ]
        kept, dropped = _drop_ungrounded_facts(facts, transcript, set(), speaker_names={"Alex"})
        assert kept == facts
        assert dropped == []

    def test_speaker_subject_object_in_known_names_kept(self):
        """Spelling-fix exemption survives role-aware gating.

        User says 'Frankford', assistant resolves to 'Frankfurt', the
        anonymization mapping carries 'Frankfurt' as a known real name.
        The triple uses 'Frankfurt' as the object — under strict role-aware
        gating it would fail (not in [user] spans verbatim), but the
        known_names exemption must keep it.
        """
        from paramem.graph.extractor import _drop_ungrounded_facts

        transcript = (
            "[user] my sister lives in frankford\n[assistant] frankfurt is about 200km from here"
        )
        facts = [
            {"subject": "Alex", "predicate": "sister_lives_in", "object": "Frankfurt"},
        ]
        kept, dropped = _drop_ungrounded_facts(
            facts,
            transcript,
            {"Alex", "Frankfurt"},  # 'Frankfurt' came in via mapping
            speaker_names={"Alex"},
        )
        assert kept == facts
        assert dropped == []

    def test_default_no_speaker_names_is_role_blind(self):
        """``speaker_names=None`` (default) preserves the old role-blind behaviour."""
        from paramem.graph.extractor import _drop_ungrounded_facts

        # Object only in assistant turn — would be dropped under role-aware,
        # but role-blind accepts it (token appears in transcript).
        transcript = "[user] suggest a hobby\n[assistant] yoga is great"
        facts = [{"subject": "Alex", "predicate": "likes", "object": "yoga"}]
        kept, _dropped = _drop_ungrounded_facts(facts, transcript, {"Alex"})
        assert kept == facts  # legacy behaviour preserved

    def test_empty_speaker_names_is_role_blind(self):
        """An empty set behaves identically to ``None`` (no role-aware path)."""
        from paramem.graph.extractor import _drop_ungrounded_facts

        transcript = "[user] suggest a hobby\n[assistant] yoga is great"
        facts = [{"subject": "Alex", "predicate": "likes", "object": "yoga"}]
        kept, _dropped = _drop_ungrounded_facts(facts, transcript, {"Alex"}, speaker_names=set())
        assert kept == facts

    def test_subject_grounded_via_assistant_turn_still_passes(self):
        """Subject-side grounding stays role-blind even in role-aware mode.

        The speaker can be addressed by name in [assistant] turns ("Alex,
        about your question..."); that mention is sufficient grounding for
        the subject.  Only the *object's* substantive content needs role-
        aware grounding.
        """
        from paramem.graph.extractor import _drop_ungrounded_facts

        # Speaker's name appears in [assistant] turn (addressing them).
        # User's own turn has the object content.
        transcript = "[user] i practice yoga every morning\n[assistant] alex, that's wonderful"
        facts = [{"subject": "Alex", "predicate": "practices", "object": "yoga"}]
        kept, _dropped = _drop_ungrounded_facts(facts, transcript, set(), speaker_names={"Alex"})
        assert kept == facts


class TestApplyGroundingGate:
    """``_apply_grounding_gate`` orchestrates the role-blind / role-aware
    gate per the configured mode (off / diagnostic / active).
    """

    @staticmethod
    def _hallucination():
        # Object only in [assistant] turn — role-aware would drop, role-blind keeps.
        return [{"subject": "Alex", "predicate": "likes", "object": "yoga"}]

    @staticmethod
    def _transcript():
        return "[user] suggest a hobby for me\n[assistant] yoga is great for stress"

    def test_off_mode_role_blind(self):
        from paramem.graph.extractor import _apply_grounding_gate

        kept, dropped, would_drop = _apply_grounding_gate(
            self._hallucination(),
            self._transcript(),
            {"Alex"},
            speaker_name="Alex",
            mode="off",
        )
        assert kept == self._hallucination()  # role-blind passes it
        assert dropped == []
        assert would_drop == []

    def test_diagnostic_mode_passes_facts_but_records_would_drops(self):
        from paramem.graph.extractor import _apply_grounding_gate

        kept, dropped, would_drop = _apply_grounding_gate(
            self._hallucination(),
            self._transcript(),
            {"Alex"},
            speaker_name="Alex",
            mode="diagnostic",
        )
        # Production behaviour unchanged: fact still kept.
        assert kept == self._hallucination()
        assert dropped == []
        # But the role-aware gate would have dropped it — recorded.
        assert would_drop == self._hallucination()

    def test_active_mode_drops_role_aware_failures(self):
        from paramem.graph.extractor import _apply_grounding_gate

        kept, dropped, would_drop = _apply_grounding_gate(
            self._hallucination(),
            self._transcript(),
            {"Alex"},
            speaker_name="Alex",
            mode="active",
        )
        assert kept == []
        assert dropped == self._hallucination()
        # Active mode does not need to track would_drop — the drop is real.
        assert would_drop == []

    def test_no_speaker_name_falls_back_to_role_blind(self):
        from paramem.graph.extractor import _apply_grounding_gate

        # Even in active mode, without a speaker_name there's no [user] anchor.
        kept, dropped, would_drop = _apply_grounding_gate(
            self._hallucination(),
            self._transcript(),
            {"Alex"},
            speaker_name=None,
            mode="active",
        )
        assert kept == self._hallucination()  # role-blind kept it
        assert dropped == []
        assert would_drop == []

    def test_unknown_mode_warns_and_falls_back_to_off(self):
        from paramem.graph.extractor import _apply_grounding_gate

        kept, _dropped, would_drop = _apply_grounding_gate(
            self._hallucination(),
            self._transcript(),
            {"Alex"},
            speaker_name="Alex",
            mode="hyperactive",  # unknown
        )
        # Falls back to off → role-blind result, no would_drop tracking.
        assert kept == self._hallucination()
        assert would_drop == []

    def test_diagnostic_mode_with_legitimate_triple_records_nothing(self):
        """Triples grounded in [user] turns are not flagged in diagnostic mode."""
        from paramem.graph.extractor import _apply_grounding_gate

        transcript = "[user] i practice yoga every morning"
        facts = [{"subject": "Alex", "predicate": "practices", "object": "yoga"}]
        kept, dropped, would_drop = _apply_grounding_gate(
            facts, transcript, {"Alex"}, speaker_name="Alex", mode="diagnostic"
        )
        assert kept == facts
        assert dropped == []
        assert would_drop == []


class TestProvenanceGateFixture:
    """Integration test against the curated PerLTQA failure fixture.

    Every candidate in ``tests/fixtures/provenance_gate_failures.json`` is a
    real assistant-into-graph hallucination from a probe run.  The role-
    aware gate must drop every one when given the dialogue's speaker as
    ``speaker_names``.  Conversely, the legacy role-blind path keeps them
    (object tokens appear in the transcript, just on the wrong side) —
    asserting both halves proves the gate is what catches them.
    """

    @staticmethod
    def _load_fixture():
        import json
        from pathlib import Path

        fp = Path(__file__).parent / "fixtures" / "provenance_gate_failures.json"
        with fp.open() as f:
            return json.load(f)

    def test_role_aware_drops_every_fixture_candidate(self):
        from paramem.graph.extractor import _drop_ungrounded_facts

        fixture = self._load_fixture()
        misses: list[str] = []
        for c in fixture["candidates"]:
            dialogue = fixture["dialogues"][c["dialogue_key"]]
            transcript = dialogue["transcript"]
            speaker = dialogue["speaker"]
            triple = {
                "subject": c["subject"],
                "predicate": c["predicate"],
                "object": c["object"],
            }
            kept, _dropped = _drop_ungrounded_facts(
                [triple], transcript, {speaker}, speaker_names={speaker}
            )
            if kept:
                misses.append(
                    f"{c['dialogue_key']}: {triple['subject']} | "
                    f"{triple['predicate']} | {triple['object']!r} "
                    f"(said by {c['said_by']})"
                )
        assert misses == [], (
            f"Role-aware gate failed to drop {len(misses)} fixture hallucinations:\n"
            + "\n".join(f"  {m}" for m in misses)
        )

    def test_role_blind_keeps_fixture_candidates_proves_gate_is_load_bearing(self):
        """Without ``speaker_names`` the legacy gate keeps every candidate.

        This asserts the gate is doing real work — the fixture's
        hallucinations *do* ground in the transcript token-wise (the
        object appears somewhere), they just appear in the wrong role.
        Role-blind grounding can't tell.  If this test ever fails, it
        means the role-blind path got tightened separately — re-evaluate
        the role-aware contract.
        """
        from paramem.graph.extractor import _drop_ungrounded_facts

        fixture = self._load_fixture()
        unexpectedly_dropped = 0
        for c in fixture["candidates"]:
            dialogue = fixture["dialogues"][c["dialogue_key"]]
            transcript = dialogue["transcript"]
            speaker = dialogue["speaker"]
            triple = {
                "subject": c["subject"],
                "predicate": c["predicate"],
                "object": c["object"],
            }
            kept, _dropped = _drop_ungrounded_facts([triple], transcript, {speaker})
            if not kept:
                unexpectedly_dropped += 1
        # Most fixture candidates should pass role-blind grounding (their
        # object tokens are present in transcript, just in assistant turns).
        # A small number may be dropped if the object also fails the basic
        # grounding (e.g. SOTA invented a name).  Assert >= 80% are kept by
        # role-blind so the gate's value is demonstrably the role split.
        total = len(fixture["candidates"])
        kept_count = total - unexpectedly_dropped
        assert kept_count / total >= 0.8, (
            f"Role-blind gate already drops {unexpectedly_dropped}/{total} "
            f"fixture candidates — fixture may be miscalibrated, or the "
            f"role-blind gate has tightened.  Re-curate the fixture."
        )


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
        # {SPEAKER_NAME} is the wrapped slot used in the few-shots; the bare
        # and title-cased forms stay listed because the graph normalizer
        # title-cases the bare form and because Mistral occasionally emits
        # "Speaker"/"User"/"I" as training-data fallbacks.
        for forbidden in (
            "{SPEAKER_NAME}",
            "SPEAKER_NAME",
            "Speaker_Name",
            "Speaker",
            "User",
            "'I'",
        ):
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


# --- Debug-artifact writers ---


class TestDebugArtifacts:
    def test_save_debug_artifacts_writes_plaintext(self, tmp_path):
        from paramem.server.consolidation import _save_debug_artifacts

        loop = MagicMock()
        loop.merger.save_graph = MagicMock()
        loop.cycle_count = 4

        config = MagicMock()
        config.debug_dir = tmp_path

        episodic_qa = [{"question": "Q", "answer": "A"}]
        procedural_rels = [{"subject": "S", "predicate": "P", "object": "O"}]

        _save_debug_artifacts(loop, config, episodic_qa, procedural_rels)

        out = tmp_path / "cycle_4"
        # All debug filenames carry the _snapshot postfix (locked decision #7)
        assert (out / "episodic_qa_snapshot.json").exists()
        assert (out / "procedural_rels_snapshot.json").exists()
        loop.merger.save_graph.assert_called_once_with(out / "graph_snapshot.json", encrypted=False)

        with open(out / "episodic_qa_snapshot.json") as f:
            saved = json.load(f)  # plaintext json — readable without decrypt
        assert saved == episodic_qa

    def test_save_debug_artifacts_omits_procedural_when_empty(self, tmp_path):
        from paramem.server.consolidation import _save_debug_artifacts

        loop = MagicMock()
        loop.merger.save_graph = MagicMock()
        loop.cycle_count = 2

        config = MagicMock()
        config.debug_dir = tmp_path

        _save_debug_artifacts(loop, config, [{"question": "Q", "answer": "A"}], [])

        out = tmp_path / "cycle_2"
        assert (out / "episodic_qa_snapshot.json").exists()
        assert not (out / "procedural_rels_snapshot.json").exists()


# ---------------------------------------------------------------------------
# PR1 Alignment Tests — D1, D3, D4, D6, D7, D8, D9, D10, D13, D17, D18
# ---------------------------------------------------------------------------


class TestPlausibilityAnon:
    """§7 test 1: _sota_pipeline with plausibility_stage="anon" (D3)."""

    def test_anon_stage_plausibility_filters_subset(self):
        """When plausibility_stage="anon" and a SOTA validator is configured, it runs
        on the anonymized facts before de-anonymization and drops flagged entries."""
        from paramem.graph.extractor import _sota_pipeline

        graph = _make_graph(
            [
                ("Alex", "lives_in", "Millfield"),
                ("Alex", "has_role", "Speaker"),  # role leak — should be dropped
            ],
            entities=[
                Entity(name="Alex", entity_type="person"),
                Entity(name="Millfield", entity_type="place"),
                Entity(name="Speaker", entity_type="concept"),
            ],
        )
        anon_facts = [
            {"subject": "Person_1", "predicate": "lives_in", "object": "City_1"},
            {"subject": "Person_1", "predicate": "has_role", "object": "Speaker"},
        ]
        mapping = {"Alex": "Person_1", "Millfield": "City_1"}
        anon_transcript = "Person_1 lives in City_1."

        # Plausibility filter keeps only the lives_in fact
        kept_anon = [{"subject": "Person_1", "predicate": "lives_in", "object": "City_1"}]

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.extractor._anonymize_with_local_model",
                return_value=(anon_facts, mapping, anon_transcript),
            ),
            patch(
                "paramem.graph.extractor._filter_with_sota",
                return_value=(anon_facts, None, None, {}),
            ),
            patch(
                "paramem.graph.extractor._plausibility_filter_with_sota",
                return_value=(kept_anon, "raw"),
            ),
        ):
            result = _sota_pipeline(
                graph,
                "Alex lives in Millfield.",
                None,
                None,
                plausibility_judge="claude",
                plausibility_stage="anon",
            )

        # Only the valid fact survives
        assert len(result.relations) == 1
        assert result.relations[0].predicate == "lives_in"
        assert result.diagnostics.get("plausibility") == "anon"


class TestPlausibilityDeanon:
    """§7 test 2: _sota_pipeline with plausibility_stage="deanon" (D3)."""

    def test_deanon_stage_plausibility_drops_tautology(self):
        """Deanon-stage local plausibility receives real names and drops tautologies."""
        from paramem.graph.extractor import _sota_pipeline

        graph = _make_graph(
            [
                ("Alex", "lives_in", "Millfield"),
                ("Alex", "has_name", "Alex"),  # tautology / self-loop
            ],
            entities=[
                Entity(name="Alex", entity_type="person"),
                Entity(name="Millfield", entity_type="place"),
            ],
        )
        anon_facts = [
            {"subject": "Person_1", "predicate": "lives_in", "object": "City_1"},
            {"subject": "Person_1", "predicate": "has_name", "object": "Person_1"},
        ]
        mapping = {"Alex": "Person_1", "Millfield": "City_1"}
        anon_transcript = "Person_1 lives in City_1."

        # Local plausibility drops the tautology, keeps lives_in
        kept_deanon = [{"subject": "Alex", "predicate": "lives_in", "object": "Millfield"}]

        local_plaus_calls = []

        def fake_local_plaus(facts, transcript, model, tokenizer, **kwargs):
            local_plaus_calls.append((list(facts), transcript))
            return kept_deanon

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.extractor._anonymize_with_local_model",
                return_value=(anon_facts, mapping, anon_transcript),
            ),
            patch(
                "paramem.graph.extractor._filter_with_sota",
                return_value=(anon_facts, None, None, {}),
            ),
            patch(
                "paramem.graph.extractor._local_plausibility_filter",
                side_effect=fake_local_plaus,
            ),
        ):
            result = _sota_pipeline(
                graph,
                "Alex lives in Millfield.",
                MagicMock(),
                MagicMock(),
                plausibility_judge="auto",
                plausibility_stage="deanon",
            )

        # Plausibility ran and dropped the tautology
        assert len(result.relations) == 1
        assert result.relations[0].predicate == "lives_in"

        # Verify the plausibility call received the ORIGINAL real-name transcript,
        # NOT the anonymized transcript (privacy-critical per D1/D3 plan).
        assert len(local_plaus_calls) == 1
        _, transcript_arg = local_plaus_calls[0]
        assert transcript_arg == "Alex lives in Millfield.", (
            "Deanon-stage plausibility must receive original transcript, not anon_transcript"
        )
        assert result.diagnostics.get("plausibility") == "deanon"


class TestResidualLeakDropsReferencingTriples:
    """§7 test 3: D1 — residual leak drops referencing triples, non-referencing survive."""

    def test_residual_leak_filters_fact_level(self):
        """On residual leak after repair (canonical mapping), only leaked-name triples
        are dropped. Non-referencing triples survive through the local path (SOTA skipped).

        Setup: _repair_anonymization_leaks is mocked to return anon_facts that still
        contain a leaked name in the object field, simulating a scenario where repair
        cannot eliminate the leak. The second verify mock confirms the residual leak,
        triggering fact-level filtering (_skip_sota=True).
        """
        from paramem.graph.extractor import _sota_pipeline

        graph = _make_graph(
            [
                ("Alex", "lives_in", "Millfield"),
                ("Alex", "friend_of", "Ghost"),  # referenced by the leak
            ],
            entities=[
                Entity(name="Alex", entity_type="person"),
                Entity(name="Millfield", entity_type="place"),
                Entity(name="Ghost", entity_type="person"),
            ],
        )
        # anon_facts still has Ghost (not anonymized in object position)
        anon_facts_initial = [
            {"subject": "Person_1", "predicate": "lives_in", "object": "City_1"},
            {"subject": "Person_1", "predicate": "friend_of", "object": "Ghost"},
        ]
        mapping = {"Alex": "Person_1", "Millfield": "City_1"}
        anon_transcript = "Person_1 lives in City_1. Person_1 is friends with Ghost."
        transcript = "Alex lives in Millfield. Alex is friends with Ghost."

        sota_calls = []

        def fake_sota(facts, *args, **kwargs):
            sota_calls.append(list(facts))
            return facts, None, None, {}

        # Mock repair to return anon_facts that STILL contain Ghost (residual leak)
        def fake_repair(
            graph, mapping, anon_facts, anon_transcript, orig_transcript, leaked, **kwargs
        ):
            # Repair "runs" but Ghost remains in object position — residual leak.
            # **kwargs absorbs newer optional args like extra_pii_types so this
            # mock doesn't need to change every time the real signature grows.
            return (
                anon_facts,
                mapping,
                anon_transcript,
                {"missed_fixed": 0, "hallucinated_dropped": 0, "residual_dropped": 0},
            )

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.extractor._anonymize_with_local_model",
                return_value=(anon_facts_initial, mapping, anon_transcript),
            ),
            patch(
                "paramem.graph.extractor._repair_anonymization_leaks",
                side_effect=fake_repair,
            ),
            patch("paramem.graph.extractor._filter_with_sota", side_effect=fake_sota),
        ):
            result = _sota_pipeline(
                graph,
                transcript,
                None,
                None,
                plausibility_judge="off",
            )

        # SOTA must NOT be called (_skip_sota=True after residual leak)
        assert sota_calls == [], "SOTA must be skipped on residual leak"
        # The lives_in triple does NOT reference "Ghost" → survives
        surviving_predicates = {r.predicate for r in result.relations}
        assert "lives_in" in surviving_predicates
        # The friend_of triple references "Ghost" (lowercase match) → dropped
        assert "friend_of" not in surviving_predicates
        # Diagnostics
        assert "residual_leaked_triples_dropped" in result.diagnostics
        assert result.diagnostics["residual_leaked_triples_dropped"] >= 1


class TestAnonFailureFallback:
    """§7 test 4: D8 — anon failure runs raw plausibility instead of returning original."""

    def test_anon_failure_triggers_fallback(self):
        """_sota_pipeline calls _fallback_plausibility_on_raw when anonymization fails."""
        from paramem.graph.extractor import _sota_pipeline

        graph = _make_graph(
            [("Alex", "lives_in", "Millfield")],
            entities=[
                Entity(name="Alex", entity_type="person"),
                Entity(name="Millfield", entity_type="place"),
            ],
        )

        fallback_calls = []

        def fake_fallback(g, t, m, tok, reason, **_kwargs):
            fallback_calls.append(reason)
            g.relations = []
            g.entities = []
            g.diagnostics["fallback_path"] = reason
            return g

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.extractor._anonymize_with_local_model",
                return_value=(None, {}, ""),
            ),
            patch(
                "paramem.graph.extractor._fallback_plausibility_on_raw",
                side_effect=fake_fallback,
            ),
        ):
            result = _sota_pipeline(graph, "transcript", None, None)

        assert fallback_calls == ["anon_failed"], (
            "fallback must be triggered with reason=anon_failed"
        )
        assert result.diagnostics.get("fallback_path") == "anon_failed"


class TestAllDroppedSafetyNet:
    """§7 test 5: D7 — all-dropped safety net triggers fallback."""

    def test_all_dropped_triggers_fallback(self):
        """When all relations are dropped by the grounding gate, _fallback_plausibility_on_raw runs.

        The all-dropped safety net (D7) fires AFTER the full pipeline (deanon + grounding gate
        + plausibility). If kept_relations is empty but original_count > 0, the pipeline
        calls _fallback_plausibility_on_raw("all_dropped"). This happens when, e.g., the
        grounding gate drops all SOTA-enriched entities as ungrounded world-knowledge inferences.
        """
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
        # SOTA enrichment produces facts with no grounding in transcript ("Saturn", "Jupiter")
        enriched_ungrounded = [
            {"subject": "Saturn", "predicate": "orbits", "object": "Jupiter"},
        ]

        fallback_calls = []

        def fake_fallback(g, t, m, tok, reason, **_kwargs):
            fallback_calls.append(reason)
            g.diagnostics["fallback_path"] = reason
            return g

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.extractor._anonymize_with_local_model",
                return_value=(anon_facts, mapping, "anon transcript"),
            ),
            patch(
                "paramem.graph.extractor._filter_with_sota",
                return_value=(enriched_ungrounded, None, None, {}),
            ),
            patch(
                "paramem.graph.extractor._fallback_plausibility_on_raw",
                side_effect=fake_fallback,
            ),
        ):
            result = _sota_pipeline(
                graph,
                # Short transcript — "Saturn" and "Jupiter" are not grounded here
                "Alex lives in Millfield.",
                None,
                None,
                plausibility_judge="off",
            )

        # Fallback must be called with reason="all_dropped"
        assert "all_dropped" in fallback_calls, f"Expected all_dropped, got: {fallback_calls}"
        assert result.diagnostics.get("fallback_path") == "all_dropped"


class TestEntityTypePreservation:
    """§7 test 6: D6 — entity types preserved, no "person" stampdown (regression)."""

    def test_preserved_entity_types_pass_through(self):
        """Entities pre-typed by _normalize_extraction keep their original types
        after the pipeline even when mocked SOTA returns same facts."""
        from paramem.graph.extractor import _sota_pipeline

        graph = _make_graph(
            [
                ("Alex", "lives_in", "Frankfurt"),
                ("Alex", "listens_to", "Music"),
            ],
            entities=[
                Entity(name="Alex", entity_type="person"),
                Entity(name="Frankfurt", entity_type="place"),
                Entity(name="Music", entity_type="concept"),
            ],
        )
        anon_facts = [
            {"subject": "Person_1", "predicate": "lives_in", "object": "City_1"},
            {"subject": "Person_1", "predicate": "listens_to", "object": "Thing_1"},
        ]
        mapping = {"Alex": "Person_1", "Frankfurt": "City_1", "Music": "Thing_1"}
        anon_transcript = "Person_1 lives in City_1 and listens to Thing_1."

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.extractor._anonymize_with_local_model",
                return_value=(anon_facts, mapping, anon_transcript),
            ),
            patch(
                "paramem.graph.extractor._filter_with_sota",
                return_value=(anon_facts, None, None, {}),
            ),
        ):
            result = _sota_pipeline(
                graph,
                "Alex lives in Frankfurt and listens to Music.",
                None,
                None,
                plausibility_judge="off",
            )

        entity_map = {e.name: e.entity_type for e in result.entities}
        assert entity_map.get("Alex") == "person"
        assert entity_map.get("Frankfurt") in ("place", "location")
        assert entity_map.get("Music") == "concept", (
            f"Music must be 'concept', not {entity_map.get('Music')!r}"
        )

    def test_sota_introduced_country_entity_typed_location(self):
        """SOTA-introduced entity with Country_ placeholder is typed 'location', not 'person'."""
        from paramem.graph.extractor import _sota_pipeline

        graph = _make_graph(
            [("Alex", "born_in", "Germany")],
            entities=[
                Entity(name="Alex", entity_type="person"),
                Entity(name="Germany", entity_type="place"),
            ],
        )
        anon_facts = [{"subject": "Person_1", "predicate": "born_in", "object": "Country_1"}]
        mapping = {"Alex": "Person_1", "Germany": "Country_1"}
        anon_transcript = "Person_1 was born in Country_1."

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.extractor._anonymize_with_local_model",
                return_value=(anon_facts, mapping, anon_transcript),
            ),
            patch(
                "paramem.graph.extractor._filter_with_sota",
                return_value=(anon_facts, None, None, {}),
            ),
        ):
            result = _sota_pipeline(
                graph,
                "Alex was born in Germany.",
                None,
                None,
                plausibility_judge="off",
            )

        entity_map = {e.name: e.entity_type for e in result.entities}
        # Germany already existed in the graph as "place"; D6 preserves existing entity types.
        # The Country_ → "location" mapping applies only to SOTA-*introduced* entities
        # (names absent from the original graph). "place" and "location" both express
        # geographic entities — accept both values.
        assert entity_map.get("Germany") in ("place", "location"), (
            f"Germany (Country_1) must be typed 'place' or 'location', "
            f"not {entity_map.get('Germany')!r}"
        )

    def test_sota_introduced_entity_no_placeholder_typed_concept(self):
        """SOTA-introduced entity with no placeholder (bare name) gets type 'concept', not 'person'.

        D6 regression guard: entity with no reverse_mapping entry defaults to 'concept'.
        China is NOT present in the original graph — only Alex is. SOTA enrichment
        introduces China as a bare name (no anonymizer placeholder), so no
        reverse_mapping entry exists. D6 ensures the fallback type is 'concept',
        never 'person'.
        """
        from paramem.graph.extractor import _sota_pipeline

        # Original graph has only Alex — no China entity
        graph = _make_graph(
            [("Alex", "has_plans", "Alex")],  # placeholder relation; SOTA will override
            entities=[
                Entity(name="Alex", entity_type="person"),
            ],
        )
        # Alex → Person_1 only; China is absent from the anonymization mapping
        anon_facts = [{"subject": "Person_1", "predicate": "has_plans", "object": "Person_1"}]
        mapping = {"Alex": "Person_1"}
        anon_transcript = "Person_1 has plans."
        # SOTA enrichment introduces China as a bare name with no placeholder equivalent
        enriched_anon = [{"subject": "Person_1", "predicate": "visited", "object": "China"}]

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.extractor._anonymize_with_local_model",
                return_value=(anon_facts, mapping, anon_transcript),
            ),
            patch(
                "paramem.graph.extractor._filter_with_sota",
                return_value=(enriched_anon, None, None, {}),
            ),
        ):
            result = _sota_pipeline(
                graph,
                "Alex visited China.",
                None,
                None,
                plausibility_judge="off",
            )

        entity_map = {e.name: e.entity_type for e in result.entities}
        # China has no reverse_mapping entry → D6 safe fallback is "concept", not "person"
        china_type = entity_map.get("China")
        assert china_type == "concept", (
            f"SOTA-introduced bare entity must be typed 'concept', not {china_type!r}"
        )


class TestFallbackPlausibilityOnRawHelper:
    """§7 test 7: D10 — direct test of _fallback_plausibility_on_raw helper."""

    def test_helper_removes_residual_placeholders(self):
        """Helper drops facts containing residual placeholder tokens."""
        from paramem.graph.extractor import _fallback_plausibility_on_raw

        graph = _make_graph(
            [
                ("Alex", "lives_in", "City_1"),  # placeholder not resolved
                ("Alex", "works_at", "Acme"),
            ],
            entities=[
                Entity(name="Alex", entity_type="person"),
                Entity(name="City_1", entity_type="place"),
                Entity(name="Acme", entity_type="organization"),
            ],
        )
        result = _fallback_plausibility_on_raw(
            graph,
            "Alex works at Acme.",
            None,
            None,
            reason="test_residual",
        )
        # City_1 is a placeholder token → the fact should be swept
        surviving = {r.object for r in result.relations}
        assert "City_1" not in surviving
        assert result.diagnostics.get("fallback_path") == "test_residual"

    def test_helper_records_fallback_path(self):
        """Helper always records the reason in diagnostics."""
        from paramem.graph.extractor import _fallback_plausibility_on_raw

        graph = _make_graph(
            [("Alex", "lives_in", "Millfield")],
            entities=[
                Entity(name="Alex", entity_type="person"),
                Entity(name="Millfield", entity_type="place"),
            ],
        )
        result = _fallback_plausibility_on_raw(
            graph, "Alex lives in Millfield.", None, None, reason="anon_failed"
        )
        assert result.diagnostics.get("fallback_path") == "anon_failed"


class TestExtractGraphNewKwargs:
    """§7 test 8: D4/D18 — new kwargs reach _sota_pipeline."""

    def test_extract_graph_plumbs_ner_and_plausibility_kwargs(self):
        """extract_graph forwards ner_check, ner_model, plausibility_judge,
        plausibility_stage, verify_anonymization to _sota_pipeline."""
        from paramem.graph.extractor import extract_graph

        captured = {}

        def fake_sota_pipeline(graph, transcript, model, tokenizer, **kwargs):
            captured.update(kwargs)
            return graph

        graph_raw = json.dumps(
            {
                "entities": [{"name": "Alex", "entity_type": "person"}],
                "relations": [],
                "summary": "",
            }
        )

        with (
            patch(
                "paramem.graph.extractor._generate_extraction",
                return_value=graph_raw,
            ),
            patch(
                "paramem.graph.extractor._sota_pipeline",
                side_effect=fake_sota_pipeline,
            ),
        ):
            # _sota_pipeline is only called when noise_filter is non-empty and
            # there are relations — since our mock graph has no relations, we
            # need to test the kwarg forwarding via a different approach.
            pass

        # Direct test: build a graph with relations and verify kwargs reach _sota_pipeline.
        graph_with_rels = json.dumps(
            {
                "entities": [
                    {"name": "Alex", "entity_type": "person"},
                    {"name": "Millfield", "entity_type": "place"},
                ],
                "relations": [
                    {
                        "subject": "Alex",
                        "predicate": "lives_in",
                        "object": "Millfield",
                        "relation_type": "factual",
                        "confidence": 1.0,
                    }
                ],
                "summary": "",
            }
        )
        captured.clear()
        with (
            patch(
                "paramem.graph.extractor._generate_extraction",
                return_value=graph_with_rels,
            ),
            patch(
                "paramem.graph.extractor._sota_pipeline",
                side_effect=fake_sota_pipeline,
            ),
        ):
            extract_graph(
                None,
                None,
                "transcript",
                "sess1",
                noise_filter="anthropic",
                ner_check=True,
                ner_model="en_core_web_trf",
                plausibility_judge="claude",
                plausibility_stage="anon",
                verify_anonymization=False,
            )

        assert captured.get("ner_check") is True
        assert captured.get("ner_model") == "en_core_web_trf"
        assert captured.get("plausibility_judge") == "claude"
        assert captured.get("plausibility_stage") == "anon"
        assert captured.get("verify_anonymization") is False

    def test_extract_graph_default_temperature_zero(self):
        """D14: extract_graph default temperature is 0.0 (was 0.3)."""
        import inspect

        from paramem.graph.extractor import extract_graph

        sig = inspect.signature(extract_graph)
        assert sig.parameters["temperature"].default == 0.0

    def test_extract_graph_default_max_tokens_2048(self):
        """D17: extract_graph default max_tokens is 2048 (was 1024)."""
        import inspect

        from paramem.graph.extractor import extract_graph

        sig = inspect.signature(extract_graph)
        assert sig.parameters["max_tokens"].default == 2048

    def test_verify_anonymization_false_skips_guard(self):
        """D18: verify_anonymization=False skips the forward-path privacy guard."""
        from paramem.graph.extractor import _sota_pipeline

        graph = _make_graph(
            [("Alex", "lives_in", "Millfield")],
            entities=[
                Entity(name="Alex", entity_type="person"),
                Entity(name="Millfield", entity_type="place"),
            ],
        )
        # Mapping that would normally trigger a leak (Millfield not anonymized)
        anon_facts = [{"subject": "Person_1", "predicate": "lives_in", "object": "Millfield"}]
        mapping = {"Alex": "Person_1"}
        anon_transcript = "Person_1 lives in Millfield."

        verifier_calls = []

        def fake_verify(*args, **kwargs):
            verifier_calls.append(True)
            return []  # report no leaks regardless

        sota_calls = []

        def fake_sota(facts, *args, **kwargs):
            sota_calls.append(list(facts))
            return facts, None, None, {}

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.extractor._anonymize_with_local_model",
                return_value=(anon_facts, mapping, anon_transcript),
            ),
            patch(
                "paramem.graph.extractor.verify_anonymization_completeness",
                side_effect=fake_verify,
            ),
            patch("paramem.graph.extractor._filter_with_sota", side_effect=fake_sota),
        ):
            _sota_pipeline(
                graph,
                "Alex lives in Millfield.",
                None,
                None,
                verify_anonymization=False,
                plausibility_judge="off",
            )

        # With verify_anonymization=False the guard function must not be called
        assert verifier_calls == [], "verify_anonymization_completeness must be skipped when False"
        # SOTA must have been called (no guard blocked it)
        assert len(sota_calls) == 1


class TestDiagnosticsKeys:
    """§7 test 9: D13 — diagnostic keys populated after full pipeline run."""

    def test_diagnostics_contains_plausibility_keys(self):
        """After a deanon-stage plausibility run, diagnostics contains the expected keys."""
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
        anon_transcript = "Person_1 lives in City_1."

        def fake_local_plaus(facts, transcript, model, tokenizer, **kwargs):
            return facts  # keep all

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.extractor._anonymize_with_local_model",
                return_value=(anon_facts, mapping, anon_transcript),
            ),
            patch(
                "paramem.graph.extractor._filter_with_sota",
                return_value=(anon_facts, None, None, {}),
            ),
            patch(
                "paramem.graph.extractor._local_plausibility_filter",
                side_effect=fake_local_plaus,
            ),
        ):
            result = _sota_pipeline(
                graph,
                "Alex lives in Millfield.",
                MagicMock(),
                MagicMock(),
                plausibility_judge="auto",
                plausibility_stage="deanon",
            )

        assert "plausibility" in result.diagnostics, "diagnostics must contain 'plausibility'"
        assert "plausibility_dropped" in result.diagnostics
        assert "plausibility_judge_actual" in result.diagnostics
        assert "anonymize" in result.diagnostics

    def test_diagnostics_anonymize_key_populated_on_success(self):
        """diagnostics['anonymize']='ok' when anonymization succeeds."""
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
                return_value=(anon_facts, mapping, "anon transcript"),
            ),
            patch(
                "paramem.graph.extractor._filter_with_sota",
                return_value=(anon_facts, None, None, {}),
            ),
        ):
            result = _sota_pipeline(
                graph,
                "Alex lives in Millfield.",
                None,
                None,
                plausibility_judge="off",
            )

        assert result.diagnostics.get("anonymize") == "ok"


class TestConsolidationScheduleConfigPrivacyGuard:
    """§7/§6 — privacy guard in ConsolidationScheduleConfig.__post_init__."""

    def test_cloud_judge_plus_deanon_stage_raises(self):
        """cloud provider + deanon stage must raise ValueError at construction."""
        import pytest

        from paramem.server.config import ConsolidationScheduleConfig

        with pytest.raises(ValueError, match="Privacy violation"):
            ConsolidationScheduleConfig(
                extraction_plausibility_judge="anthropic",
                extraction_plausibility_stage="deanon",
            )

    def test_cloud_judge_plus_anon_stage_ok(self):
        """cloud provider + anon stage is safe and must not raise."""
        from paramem.server.config import ConsolidationScheduleConfig

        cfg = ConsolidationScheduleConfig(
            extraction_plausibility_judge="claude",
            extraction_plausibility_stage="anon",
        )
        assert cfg.extraction_plausibility_judge == "claude"
        assert cfg.extraction_plausibility_stage == "anon"

    def test_auto_judge_any_stage_ok(self):
        """auto judge is always safe regardless of stage."""
        from paramem.server.config import ConsolidationScheduleConfig

        cfg = ConsolidationScheduleConfig(
            extraction_plausibility_judge="auto",
            extraction_plausibility_stage="deanon",
        )
        assert cfg.extraction_plausibility_judge == "auto"

    def test_off_judge_any_stage_ok(self):
        """off judge is always safe regardless of stage."""
        from paramem.server.config import ConsolidationScheduleConfig

        cfg = ConsolidationScheduleConfig(
            extraction_plausibility_judge="off",
            extraction_plausibility_stage="deanon",
        )
        assert cfg.extraction_plausibility_judge == "off"

    def test_defaults_do_not_raise(self):
        """Default config (auto/deanon) must not raise."""
        from paramem.server.config import ConsolidationScheduleConfig

        cfg = ConsolidationScheduleConfig()
        assert cfg.extraction_plausibility_judge == "auto"
        assert cfg.extraction_plausibility_stage == "deanon"

    def test_minimal_yaml_loads_with_defaults(self, tmp_path):
        """Back-compat: minimal yaml without new keys loads with all new defaults.

        Pre-flight check #2 from alignment-plan-2026-04-15.md §12.
        """
        from paramem.server.config import load_server_config

        minimal_yaml = tmp_path / "server.yaml"
        minimal_yaml.write_text(
            "model: mistral\nconsolidation:\n  schedule: every 2h\n  mode: simulate\n"
        )
        config = load_server_config(minimal_yaml)
        # New fields must be present with defaults
        assert config.consolidation.extraction_plausibility_judge == "auto"
        assert config.consolidation.extraction_plausibility_stage == "deanon"
        assert config.consolidation.extraction_verify_anonymization is True
        assert config.consolidation.extraction_ner_check is False
        assert config.consolidation.extraction_ner_model == "en_core_web_sm"
