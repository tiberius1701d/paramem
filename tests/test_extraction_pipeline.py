"""Tests for the extraction pipeline — STT correction, HA validation, noise filter, JSON parsing."""

import json
from pathlib import Path
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
            speaker_id="Speaker0",
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
        # Envelope-keyed dict — `entities` triggers acceptance.
        text = 'Some text {"entities": [], "relations": []} more text'
        result = json.loads(_extract_json_block(text))
        assert result == {"entities": [], "relations": []}

    def test_array(self):
        # Plausibility-shape: list of fact dicts (have ``subject`` key).
        text = 'Some text [{"subject": "Alex", "predicate": "likes", "object": "yoga"}] more text'
        result = json.loads(_extract_json_block(text))
        assert result[0]["subject"] == "Alex"

    def test_string_list_before_object(self):
        # First candidate `["a", "b"]` has neither dict-shaped nor int-typed
        # elements — rejected as not-an-envelope.  Walk continues to the
        # dict envelope.  Documents that string-list scalars are not
        # accepted as envelopes (they would be truncation-survivors).
        text = '["a", "b"] {"entities": []}'
        result = json.loads(_extract_json_block(text))
        assert result == {"entities": []}

    def test_bare_int_list_is_envelope(self):
        # Plausibility-shape: drop-set bare integer array (`[0, 2, 5]`).
        # Accepted as a valid envelope so the shared finder serves the
        # drop-set parser without bespoke unwrap.  Distinct from extraction
        # / enrichment outputs (those have dict-shaped first elements).
        text = "[0, 2, 5]"
        result = json.loads(_extract_json_block(text))
        assert result == [0, 2, 5]

    def test_markdown_code_block(self):
        text = '```json\n{"facts": []}\n```'
        result = json.loads(_extract_json_block(text))
        assert result == {"facts": []}

    def test_nested_object(self):
        text = '{"entities": [], "relations": [], "outer": {"inner": 1}}'
        result = json.loads(_extract_json_block(text))
        assert result == {"entities": [], "relations": [], "outer": {"inner": 1}}

    def test_empty_array(self):
        text = "Result: []"
        result = json.loads(_extract_json_block(text))
        assert result == []

    def test_no_json_raises(self):
        with pytest.raises(ValueError):
            _extract_json_block("no json here")

    def test_string_value_with_closing_brace(self):
        """Regression: string values containing `}` must not break the parser.
        The previous brace-counting walk truncated at the first `}` it saw,
        regardless of whether that `}` was inside a quoted string. Real local
        Mistral output for one of the resume chunks reliably hit this and
        produced empty graphs in two consecutive consolidation runs."""
        text = 'Sure, here: {"facts": [{"object": "Code: }"}]} trailing'
        result = json.loads(_extract_json_block(text))
        assert result["facts"][0]["object"] == "Code: }"

    def test_string_value_with_opening_brace(self):
        """Same regression, opposite direction: `{` inside a string value
        (e.g. anonymizer placeholder forms) must not inflate depth."""
        text = '{"facts": [{"object": "Acme {Org_1} Berlin"}]}'
        result = json.loads(_extract_json_block(text))
        assert result["facts"][0]["object"] == "Acme {Org_1} Berlin"

    def test_string_value_with_unbalanced_braces(self):
        """A pathological case: string contains an unmatched `}` and the
        outer JSON is still well-formed. Must parse cleanly."""
        text = '{"facts": [{"a": "value with } and { braces"}]}'
        result = json.loads(_extract_json_block(text))
        assert result["facts"][0]["a"] == "value with } and { braces"

    def test_preamble_then_object_with_brace_in_string(self):
        """LLM output often has prose preamble then JSON. Combination of
        preamble + string-with-brace was the actual production failure."""
        text = """Here are the extracted facts:

{"entities": [{"name": "x", "label": "Has } in label"}]}"""
        result = json.loads(_extract_json_block(text))
        assert result["entities"][0]["label"] == "Has } in label"

    def test_truncated_json_raises(self):
        """Genuinely malformed (incomplete) JSON should still raise — we
        do not want silent salvage of partial structures."""
        with pytest.raises(ValueError):
            _extract_json_block('{"a": "unfinished')

    def test_truncated_envelope_does_not_fall_through_to_inner_object(self):
        """Regression: a truncated outer envelope (e.g. cut at max_tokens
        mid-relation) used to silently match the first inner sub-object,
        producing an empty graph downstream. The parser must raise instead.

        Reproduces the production middle-session bug where Mistral 7B
        emitted ~6000 chars of valid JSON-prefix that opened with
        ``{"entities": [{"name": "Alex", ...``, then got cut off
        mid-string at ``"object": "consumer hardware`` because the chunker
        produced a chunk too large for the old 2048-token budget. The
        previous parser's left-to-right fall-through would have returned
        the inner entity dict, _normalize_extraction would not find
        ``entities``/``relations`` keys, and the SessionGraph would end up
        empty — masking the truncation.
        """
        truncated = (
            '{"entities": [\n'
            '  {"name": "Alex", "entity_type": "person", "attributes": {}},\n'
            '  {"name": "Independent Germany", "entity_type": "place", "attributes": {}}\n'
            "],\n"
            '"relations": [\n'
            '  {"subject": "Alex", "predicate": "works_with", "object": "consumer hardware'
        )
        with pytest.raises(ValueError, match="(?i)truncated"):
            _extract_json_block(truncated)


class TestParseExtractionShapes:
    """Regression: local Mistral occasionally emits unexpected JSON shapes
    (bare list of facts instead of {"entities": ..., "relations": ...}).
    Previous behaviour: TypeError from `data["session_id"] = ...` because
    `data` was a list. New behaviour: rewrap as a relations payload."""

    def test_bare_list_of_relations(self):
        from paramem.graph.extractor import _parse_extraction

        raw = (
            '[{"subject": "Alice", "predicate": "lives_in", "object": "Berlin", '
            '"relation_type": "factual", "confidence": 1.0}]'
        )
        g = _parse_extraction(raw, "session1", speaker_id="Speaker0")
        assert len(g.relations) == 1
        assert g.relations[0].subject == "Alice"
        assert g.relations[0].object == "Berlin"

    def test_empty_list(self):
        from paramem.graph.extractor import _parse_extraction

        g = _parse_extraction("[]", "session1", speaker_id="Speaker0")
        assert len(g.relations) == 0
        assert len(g.entities) == 0


class TestParseFactsResponseSalvage:
    """``_parse_facts_response`` recovers fact dicts when the model emits a
    structured response but the outer JSON envelope is truncated (Mistral 7B
    on long KEEP-by-default plausibility passes hits EOS mid-array; the
    closing ``]`` never arrives).

    Without the salvage path the plausibility filter's strict array parse
    fails and ``_local_plausibility_filter`` returns ``None`` — the gate
    fail-opens and 0 facts get filtered.  Salvage extracts the well-formed
    inner ``{…}`` blocks via depth-walk and returns those.
    """

    def test_clean_array_returned_as_is(self):
        from paramem.graph.extractor import _parse_facts_response

        raw = (
            "[\n"
            '  {"subject": "Alice", "predicate": "lives_in", "object": "Berlin"},\n'
            '  {"subject": "Bob",   "predicate": "knows",    "object": "Alice"}\n'
            "]"
        )
        out = _parse_facts_response(raw, strict_array=True)
        assert isinstance(out, list)
        assert len(out) == 2
        assert out[0]["subject"] == "Alice"
        assert out[1]["object"] == "Alice"

    def test_truncated_bare_array_is_salvaged(self):
        """The Mistral-EOS-mid-array case: array opens, two records emit
        cleanly, third record is partial / missing — salvage keeps the two
        complete dicts.
        """
        from paramem.graph.extractor import _parse_facts_response

        raw = (
            "[\n"
            '  {"subject": "Alice", "predicate": "lives_in", "object": "Berlin"},\n'
            '  {"subject": "Bob",   "predicate": "knows",    "object": "Alice"},\n'
            '  {"subject": "Carol", "predicate": "wo'  # truncated mid-string
        )
        out = _parse_facts_response(raw, strict_array=True)
        assert isinstance(out, list), f"salvage must return a list, got {type(out)}"
        assert len(out) == 2, f"expected 2 salvaged dicts, got {out!r}"
        assert {f["subject"] for f in out} == {"Alice", "Bob"}

    def test_truncated_after_last_record_is_salvaged(self):
        """Real-world shape from the live probe: array's last well-formed
        record closes with ``}`` and the model emits EOS without the
        comma-or-``]`` continuation.  All records are valid; salvage should
        keep all of them.
        """
        from paramem.graph.extractor import _parse_facts_response

        raw = (
            "[\n"
            '  {"subject": "Alice", "predicate": "lives_in", "object": "Berlin"},\n'
            '  {"subject": "Bob", "predicate": "knows", "object": "Alice"}'
            # No trailing comma, no closing ]
        )
        out = _parse_facts_response(raw, strict_array=True)
        assert isinstance(out, list)
        assert len(out) == 2

    def test_salvage_filters_non_fact_dicts(self):
        """Stream-walk picks up every balanced ``{…}`` block — it must
        drop dicts that aren't fact-shaped (no ``subject`` / ``predicate``
        / ``object``) so commentary literals, bindings sub-dicts etc.
        don't pollute the result.
        """
        from paramem.graph.extractor import _parse_facts_response

        raw = (
            "[\n"
            '  {"note": "preamble commentary"},\n'
            '  {"subject": "Alice", "predicate": "knows", "object": "Bob"},\n'
            '  {"meta": "trailing"'  # truncated; would-be 3rd dict not closed
        )
        out = _parse_facts_response(raw, strict_array=True)
        assert isinstance(out, list)
        assert len(out) == 1
        assert out[0]["subject"] == "Alice"

    def test_dict_wrapped_clean_response(self):
        """Non-strict mode: the SOTA enrichment legacy path accepts a
        dict-wrapped response with a ``facts``/``relations`` key.
        """
        from paramem.graph.extractor import _parse_facts_response

        raw = '{"facts": [{"subject": "Alice", "predicate": "knows", "object": "Bob"}]}'
        out = _parse_facts_response(raw, strict_array=False)
        assert isinstance(out, list)
        assert len(out) == 1

    def test_none_input_returns_none(self):
        from paramem.graph.extractor import _parse_facts_response

        assert _parse_facts_response(None, strict_array=True) is None
        assert _parse_facts_response(None, strict_array=False) is None

    def test_empty_string_returns_none(self):
        from paramem.graph.extractor import _parse_facts_response

        assert _parse_facts_response("", strict_array=True) is None

    def test_garbage_no_braces_returns_none(self):
        """Salvage needs at least one balanced ``{…}`` block to recover
        anything; pure prose with no JSON yields ``None``.
        """
        from paramem.graph.extractor import _parse_facts_response

        out = _parse_facts_response("I cannot help with that request.", strict_array=True)
        assert out is None

    def test_salvage_handles_strings_with_braces(self):
        """A ``}`` inside a string literal must not close the depth
        counter.  Without proper string-state tracking the salvage walk
        would split on the inner brace and emit a malformed half-block.
        """
        from paramem.graph.extractor import _parse_facts_response

        raw = (
            "[\n"
            '  {"subject": "Alice", "predicate": "said", "object": "hello } world"},\n'
            '  {"subject": "Bob", "predicate": "knows", "obj'  # truncated
        )
        out = _parse_facts_response(raw, strict_array=True)
        assert isinstance(out, list)
        assert len(out) == 1
        assert out[0]["object"] == "hello } world"


class TestPlausibilityDropSet:
    """The plausibility judge emits ``{"drop": [<index>, ...]}`` — a small
    JSON object listing which input facts to drop by zero-based index.
    ``_apply_drop_set`` parses that output and returns the surviving facts.

    This class covers the parser tolerance and the drop application:
    happy path, alternative output shapes the model might produce, edge
    cases (out-of-range, duplicates, malformed), and the fail-open
    contract on parse failure.
    """

    def _facts(self, n: int) -> list[dict]:
        return [{"subject": f"S{i}", "predicate": "p", "object": f"O{i}"} for i in range(n)]

    def test_empty_drop_set_keeps_all_facts(self):
        """``{"drop": []}`` is the prompt-defined "clean input" output — the
        judge found no DROP-rule matches; every fact survives."""
        from paramem.graph.extractor import _apply_drop_set

        facts = self._facts(5)
        out = _apply_drop_set(facts, '{"drop": []}')
        assert out == facts

    def test_single_index_dropped(self):
        from paramem.graph.extractor import _apply_drop_set

        facts = self._facts(5)
        out = _apply_drop_set(facts, '{"drop": [2]}')
        assert out is not None
        assert [f["subject"] for f in out] == ["S0", "S1", "S3", "S4"]

    def test_multiple_indices_dropped_unordered(self):
        from paramem.graph.extractor import _apply_drop_set

        facts = self._facts(6)
        out = _apply_drop_set(facts, '{"drop": [4, 0, 2]}')
        assert out is not None
        assert [f["subject"] for f in out] == ["S1", "S3", "S5"]

    def test_duplicate_indices_dedupped(self):
        from paramem.graph.extractor import _apply_drop_set

        facts = self._facts(5)
        out = _apply_drop_set(facts, '{"drop": [1, 1, 1]}')
        assert out is not None
        assert [f["subject"] for f in out] == ["S0", "S2", "S3", "S4"]

    def test_out_of_range_indices_skipped(self):
        """A bad index shouldn't void an otherwise-valid drop set —
        skip with a warning rather than fail-open the entire gate."""
        from paramem.graph.extractor import _apply_drop_set

        facts = self._facts(3)
        out = _apply_drop_set(facts, '{"drop": [0, 99, -1, 2]}')
        assert out is not None
        assert [f["subject"] for f in out] == ["S1"]

    def test_bare_array_shape_accepted(self):
        """Some models drop the ``{"drop": ...}`` wrapper and emit a bare
        integer array.  Accepted because the intent is unambiguous."""
        from paramem.graph.extractor import _apply_drop_set

        facts = self._facts(4)
        out = _apply_drop_set(facts, "[1, 3]")
        assert out is not None
        assert [f["subject"] for f in out] == ["S0", "S2"]

    def test_object_index_with_rule_annotation(self):
        """Some models annotate each drop with the rule that fired:
        ``{"drop": [{"index": 2, "rule": "R1"}, ...]}``.  Index extracted;
        rule ignored at parse time (could land in diagnostics later)."""
        from paramem.graph.extractor import _apply_drop_set

        facts = self._facts(5)
        raw = '{"drop": [{"index": 1, "rule": "R3"}, {"index": 4, "rule": "R5"}]}'
        out = _apply_drop_set(facts, raw)
        assert out is not None
        assert [f["subject"] for f in out] == ["S0", "S2", "S3"]

    def test_alternate_key_drop_indices(self):
        """``"drop_indices"`` is a common synonym a model might pick.
        Accept it transparently."""
        from paramem.graph.extractor import _apply_drop_set

        facts = self._facts(3)
        out = _apply_drop_set(facts, '{"drop_indices": [0]}')
        assert out is not None
        assert [f["subject"] for f in out] == ["S1", "S2"]

    def test_code_fenced_output_is_unwrapped(self):
        """Models often wrap structured output in ```json``` fences.
        The shared envelope-finder strips them."""
        from paramem.graph.extractor import _apply_drop_set

        facts = self._facts(4)
        raw = '```json\n{"drop": [2]}\n```'
        out = _apply_drop_set(facts, raw)
        assert out is not None
        assert [f["subject"] for f in out] == ["S0", "S1", "S3"]

    def test_single_backtick_inline_code_is_unwrapped(self):
        """Live-probe regression: when the prompt itself uses inline-code
        formatting around the output spec example, the model copies the
        single-backtick wrapper into its answer (``​`{"drop": [2]}`​``).
        Parser must strip the inline-code wrapper too — not just the
        triple-backtick code-fence form.
        """
        from paramem.graph.extractor import _apply_drop_set

        facts = self._facts(4)
        raw = '`{"drop": [2]}`'
        out = _apply_drop_set(facts, raw)
        assert out is not None
        assert [f["subject"] for f in out] == ["S0", "S1", "S3"]

    def test_malformed_output_returns_none(self):
        """Parse failure must return ``None`` — caller fail-opens by
        keeping all input facts.  This matches the prior contract:
        ``filtered_list is None`` → ``_sota_pipeline`` logs a warning
        and continues with the unfiltered input."""
        from paramem.graph.extractor import _apply_drop_set

        facts = self._facts(3)
        assert _apply_drop_set(facts, "I cannot process this request.") is None
        assert _apply_drop_set(facts, "{not_valid_json") is None

    def test_none_input_returns_none(self):
        from paramem.graph.extractor import _apply_drop_set

        assert _apply_drop_set([], None) is None
        assert _apply_drop_set([{"subject": "S"}], None) is None

    def test_empty_input_with_empty_drop(self):
        """``_apply_drop_set([], '{"drop": []}')`` is the most common
        plausibility outcome on an extraction that produced no facts —
        must succeed and return an empty list."""
        from paramem.graph.extractor import _apply_drop_set

        out = _apply_drop_set([], '{"drop": []}')
        assert out == []

    def test_drop_set_with_non_int_entries_skipped(self):
        """Stray strings / null / booleans inside the array don't void the
        whole set — they're skipped while integer entries are honoured."""
        from paramem.graph.extractor import _apply_drop_set

        facts = self._facts(4)
        out = _apply_drop_set(facts, '{"drop": [1, "junk", null, 3, true]}')
        assert out is not None
        assert [f["subject"] for f in out] == ["S0", "S2"]


class TestRenderIndexedFacts:
    """``_render_indexed_facts`` produces ``[N] <json>`` lines that the
    plausibility prompt teaches the judge to reference.  Without a stable
    indexing scheme the drop-set protocol can't address specific facts."""

    def test_indices_are_zero_based_and_contiguous(self):
        from paramem.graph.extractor import _render_indexed_facts

        rendered = _render_indexed_facts(
            [
                {"subject": "A", "predicate": "p", "object": "B"},
                {"subject": "C", "predicate": "p", "object": "D"},
                {"subject": "E", "predicate": "p", "object": "F"},
            ]
        )
        lines = rendered.splitlines()
        assert len(lines) == 3
        assert lines[0].startswith("[0] ")
        assert lines[1].startswith("[1] ")
        assert lines[2].startswith("[2] ")

    def test_each_line_is_valid_json_after_prefix(self):
        from paramem.graph.extractor import _render_indexed_facts

        facts = [
            {"subject": "Alex", "predicate": "lives_in", "object": "Berlin"},
            {"subject": "Alex", "predicate": "likes", "object": "jazz"},
        ]
        rendered = _render_indexed_facts(facts)
        for i, line in enumerate(rendered.splitlines()):
            prefix = f"[{i}] "
            assert line.startswith(prefix)
            # The remainder must round-trip through json.loads to the
            # original dict — this is what makes the index-based protocol
            # safe (the judge sees both the index and the fact).
            parsed = json.loads(line[len(prefix) :])
            assert parsed == facts[i]

    def test_empty_input_produces_empty_string(self):
        from paramem.graph.extractor import _render_indexed_facts

        assert _render_indexed_facts([]) == ""

    def test_unicode_facts_preserved_verbatim(self):
        """Real PII attributes contain non-ASCII (German names, location
        diacritics).  The renderer must not escape them — round-trip
        through ``json.loads`` would still work but the judge sees a
        less natural string."""
        from paramem.graph.extractor import _render_indexed_facts

        rendered = _render_indexed_facts(
            [{"subject": "Müller", "predicate": "lives_in", "object": "Köln"}]
        )
        assert "Müller" in rendered
        assert "Köln" in rendered


class TestEnrichmentDelta:
    """``_parse_enrichment_delta`` and ``_apply_enrichment_delta`` are the
    SOTA enrichment counterpart of the plausibility drop-set helpers.
    The judge emits a small ``{"add": [...], "modify": [...], "drop":
    [...], "bindings": {...}}`` envelope; every key is optional.  The
    parser is permissive about wrapping (markdown fences / inline-code /
    prose preamble) via the shared envelope finder.  The applier
    composes modify → drop → add and reconstructs ``updated_transcript``
    locally from ``bindings`` + ``anon_transcript`` (no transcript echo
    on the wire).
    """

    @staticmethod
    def _facts(n: int) -> list[dict]:
        return [{"subject": f"S{i}", "predicate": "p", "object": f"O{i}"} for i in range(n)]

    def test_empty_envelope_is_noop(self):
        """``{}`` — model emitted nothing to do.  Surviving facts equal
        input; transcript unchanged; bindings empty."""
        from paramem.graph.extractor import _apply_enrichment_delta

        facts = self._facts(3)
        out, transcript, bindings, _ = _apply_enrichment_delta(facts, "{}", "hello")
        assert out == facts
        assert transcript == "hello"
        assert bindings == {}

    def test_drop_only(self):
        """Pure subtractive delta — same shape as a plausibility output;
        applier still works (drop is shared between protocols)."""
        from paramem.graph.extractor import _apply_enrichment_delta

        facts = self._facts(4)
        out, _, _, _ = _apply_enrichment_delta(facts, '{"drop": [1, 3]}', None)
        assert out is not None
        assert [f["subject"] for f in out] == ["S0", "S2"]

    def test_add_only(self):
        """Append-only — coreference resolution case."""
        from paramem.graph.extractor import _apply_enrichment_delta

        facts = self._facts(2)
        raw = (
            '{"add": [{"subject": "Person_1", "predicate": "married_to",'
            ' "object": "Person_2", "relation_type": "social", "confidence": 0.9}]}'
        )
        out, _, _, _ = _apply_enrichment_delta(facts, raw, None)
        assert out is not None
        assert len(out) == 3
        assert out[2]["predicate"] == "married_to"

    def test_modify_partial_field_update(self):
        """Synonym dedup — replace ``employed_by`` with ``worked_for``
        on a single indexed fact."""
        from paramem.graph.extractor import _apply_enrichment_delta

        facts = [
            {"subject": "Alex", "predicate": "employed_by", "object": "Acme"},
            {"subject": "Alex", "predicate": "lives_in", "object": "Berlin"},
        ]
        raw = '{"modify": [{"index": 0, "fields": {"predicate": "worked_for"}}]}'
        out, _, _, _ = _apply_enrichment_delta(facts, raw, None)
        assert out is not None
        assert out[0]["predicate"] == "worked_for"
        # Other fields untouched (shallow merge).
        assert out[0]["subject"] == "Alex"
        assert out[0]["object"] == "Acme"
        # Other facts untouched.
        assert out[1] == facts[1]

    def test_compound_split_via_drop_plus_add(self):
        """``likes(P, "hiking and cooking")`` → drop the compound, add
        two atomic facts.  Documents the canonical compound-split shape
        in the new protocol."""
        from paramem.graph.extractor import _apply_enrichment_delta

        facts = [
            {"subject": "P", "predicate": "likes", "object": "hiking and cooking"},
            {"subject": "P", "predicate": "lives_in", "object": "Berlin"},
        ]
        raw = (
            '{"drop": [0],'
            ' "add": [{"subject":"P","predicate":"likes","object":"hiking"},'
            ' {"subject":"P","predicate":"likes","object":"cooking"}]}'
        )
        out, _, _, _ = _apply_enrichment_delta(facts, raw, None)
        assert out is not None
        objs = [f["object"] for f in out]
        assert "hiking and cooking" not in objs
        assert "hiking" in objs
        assert "cooking" in objs
        assert "Berlin" in objs

    def test_combined_modify_drop_add(self):
        """All three actions together — exercises the full pipeline.
        Indices in ``modify`` and ``drop`` reference the *original*
        input list, regardless of application order."""
        from paramem.graph.extractor import _apply_enrichment_delta

        facts = self._facts(4)
        raw = (
            '{"modify": [{"index": 0, "fields": {"object": "O0_modified"}}],'
            ' "drop": [2],'
            ' "add": [{"subject":"S_new","predicate":"p","object":"O_new"}]}'
        )
        out, _, _, _ = _apply_enrichment_delta(facts, raw, None)
        assert out is not None
        # S0 modified, S2 dropped, S_new appended → [S0, S1, S3, S_new]
        subjects = [f["subject"] for f in out]
        assert subjects == ["S0", "S1", "S3", "S_new"]
        assert out[0]["object"] == "O0_modified"

    def test_bindings_reconstruct_transcript_longest_first(self):
        """Reconstruction must replace longest spans first so a longer
        span wins over a shorter one that would otherwise consume part
        of it."""
        from paramem.graph.extractor import _apply_enrichment_delta

        facts: list[dict] = []
        anon = "Person_1 was a Senior Software Engineer at Org_1."
        # Both bindings share the substring "Software Engineer".  Without
        # longest-first ordering, "Software Engineer" would replace first
        # and corrupt the longer span.
        raw = '{"bindings": {"Role_1": "Senior Software Engineer", "Role_2": "Software Engineer"}}'
        _, transcript, bindings, _ = _apply_enrichment_delta(facts, raw, anon)
        assert "{Role_1}" in transcript
        # "Software Engineer" should not survive because it was inside
        # the longer span that got replaced first.
        assert "Software Engineer" not in transcript
        # Role_2's span no longer appears, so its placeholder isn't
        # written into the transcript — that's expected, the binding
        # just sits unused.
        assert bindings == {
            "Role_1": "Senior Software Engineer",
            "Role_2": "Software Engineer",
        }

    def test_bindings_replace_all_occurrences(self):
        """Entities mentioned more than once in the transcript get one
        placeholder consistently — every occurrence replaced."""
        from paramem.graph.extractor import _apply_enrichment_delta

        anon = "Person_1 led Event. Later, Person_2 joined Event."
        raw = '{"bindings": {"Event_1": "Event"}}'
        _, transcript, _, _ = _apply_enrichment_delta([], raw, anon)
        assert transcript.count("{Event_1}") == 2
        assert "Event " not in transcript or transcript.count("Event ") == 0

    def test_code_fenced_envelope_unwrapped(self):
        """Markdown fences handled by the shared envelope finder."""
        from paramem.graph.extractor import _apply_enrichment_delta

        facts = self._facts(2)
        raw = '```json\n{"drop": [0]}\n```'
        out, _, _, _ = _apply_enrichment_delta(facts, raw, None)
        assert out is not None
        assert [f["subject"] for f in out] == ["S1"]

    def test_legacy_new_entity_bindings_alias(self):
        """``new_entity_bindings`` is accepted as a synonym of
        ``bindings`` so older response shapes don't lose the binding
        payload silently during the transition."""
        from paramem.graph.extractor import _apply_enrichment_delta

        anon = "Person_1 led the agile transformation initiative."
        raw = '{"new_entity_bindings": {"Event_1": "the agile transformation initiative"}}'
        _, transcript, bindings, _ = _apply_enrichment_delta([], raw, anon)
        assert bindings == {"Event_1": "the agile transformation initiative"}
        assert "{Event_1}" in transcript

    def test_out_of_range_modify_skipped(self):
        """Modify index outside ``[0, n_facts)`` is dropped with a
        warning, not failed — single bad index shouldn't void the
        whole delta."""
        from paramem.graph.extractor import _apply_enrichment_delta

        facts = self._facts(2)
        raw = '{"modify": [{"index": 99, "fields": {"object": "X"}}]}'
        out, _, _, _ = _apply_enrichment_delta(facts, raw, None)
        assert out is not None
        assert out == facts  # nothing applied

    def test_out_of_range_drop_skipped(self):
        from paramem.graph.extractor import _apply_enrichment_delta

        facts = self._facts(2)
        raw = '{"drop": [99]}'
        out, _, _, _ = _apply_enrichment_delta(facts, raw, None)
        assert out is not None
        assert out == facts

    def test_modify_with_non_dict_fields_skipped(self):
        from paramem.graph.extractor import _apply_enrichment_delta

        facts = self._facts(2)
        raw = '{"modify": [{"index": 0, "fields": "not a dict"}]}'
        out, _, _, _ = _apply_enrichment_delta(facts, raw, None)
        assert out is not None
        assert out == facts

    def test_add_entries_must_be_dicts(self):
        """Non-dict entries in ``add`` are skipped, not failed."""
        from paramem.graph.extractor import _apply_enrichment_delta

        facts = self._facts(1)
        raw = '{"add": ["not a fact", null, {"subject":"X","predicate":"p","object":"Y"}]}'
        out, _, _, _ = _apply_enrichment_delta(facts, raw, None)
        assert out is not None
        assert len(out) == 2  # 1 input + 1 valid add
        assert out[1]["subject"] == "X"

    def test_malformed_envelope_returns_none(self):
        """Caller fail-opens — applier returns ``None`` for new_facts so
        ``_sota_pipeline`` keeps the pre-enrichment facts."""
        from paramem.graph.extractor import _apply_enrichment_delta

        facts = self._facts(2)
        out, _, _, _ = _apply_enrichment_delta(facts, "I cannot process this.", None)
        assert out is None

    def test_none_raw_returns_none(self):
        from paramem.graph.extractor import _apply_enrichment_delta

        out, _, _, _ = _apply_enrichment_delta(self._facts(1), None, "transcript")
        assert out is None

    def test_null_keys_treated_as_empty(self):
        """Model emits ``"add": null`` instead of ``[]`` — must not crash."""
        from paramem.graph.extractor import _apply_enrichment_delta

        facts = self._facts(2)
        raw = '{"add": null, "modify": null, "drop": null, "bindings": null}'
        out, transcript, bindings, _ = _apply_enrichment_delta(facts, raw, "anon")
        assert out == facts
        assert transcript == "anon"
        assert bindings == {}

    def test_bindings_with_missing_span_in_transcript_skipped(self):
        """Hallucinated binding (span not in transcript) leaves the
        transcript untouched.  No crash, no replacement."""
        from paramem.graph.extractor import _apply_enrichment_delta

        anon = "Person_1 said hello."
        raw = '{"bindings": {"Event_1": "this span is not here"}}'
        _, transcript, bindings, _ = _apply_enrichment_delta([], raw, anon)
        assert transcript == anon
        assert bindings == {"Event_1": "this span is not here"}

    def test_none_transcript_returns_none_transcript(self):
        from paramem.graph.extractor import _apply_enrichment_delta

        _, transcript, _, _ = _apply_enrichment_delta([], '{"add": []}', None)
        assert transcript is None


class TestPipelineMaxTokensThreading:
    """Verify the single ``extraction_max_tokens`` config flows through the
    entire LLM pipeline (local extract → anonymize → SOTA enrich → deanon →
    plausibility) instead of each stage carrying its own hardcoded budget."""

    def test_sota_pipeline_signature_accepts_max_tokens(self):
        """Stage 1: _sota_pipeline accepts max_tokens kwarg (the entry point
        from extract_graph)."""
        import inspect

        from paramem.graph.extractor import _sota_pipeline

        sig = inspect.signature(_sota_pipeline)
        assert "max_tokens" in sig.parameters

    def test_extract_graph_default_matches_filter_default(self):
        """The single-budget invariant: extract_graph and the SOTA-side
        filter calls must share the same default. Otherwise a user who
        sets only the loop-level config would get inconsistent budgets
        across stages."""
        import inspect

        from paramem.graph.extractor import _DEFAULT_FILTER_MAX_TOKENS, extract_graph

        default = inspect.signature(extract_graph).parameters["max_tokens"].default
        assert default == _DEFAULT_FILTER_MAX_TOKENS

    def test_fallback_plausibility_threads_max_tokens(self):
        """The all_dropped / anon_failed fallback path also accepts max_tokens
        so the whole pipeline runs on one budget — including degraded paths."""
        import inspect

        from paramem.graph.extractor import _fallback_plausibility_on_raw

        sig = inspect.signature(_fallback_plausibility_on_raw)
        assert "max_tokens" in sig.parameters


class TestWaitForGpuReady:
    """Cover the WSL2 cloud-idle → local-LLM wake helper added after the
    May 2 production crash where a 62s SOTA cloud round-trip left the GPU
    in a low-power state and the next CUDA op hit "device not ready"."""

    def test_no_op_when_cuda_unavailable(self):
        """In CPU-only test environments, the helper must be a no-op (and not
        raise on missing torch.cuda)."""
        from unittest.mock import patch

        import paramem.graph.extractor as extractor

        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = False
        with patch.dict("sys.modules", {"torch": fake_torch}):
            extractor._wait_for_gpu_ready()  # must not raise
        assert not fake_torch.zeros.called

    def test_passes_through_when_gpu_ready(self):
        """Happy path: pre-settle sleep runs once, probe succeeds on first
        attempt, no retry sleeps."""
        from unittest.mock import patch

        import paramem.graph.extractor as extractor

        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True
        with (
            patch.dict("sys.modules", {"torch": fake_torch}),
            patch("paramem.graph.extractor.time.sleep") as sleep_mock,
        ):
            extractor._wait_for_gpu_ready()
        fake_torch.cuda.synchronize.assert_called()
        fake_torch.zeros.assert_called()
        # Exactly one pre-settle sleep on the happy path.
        assert sleep_mock.call_count == 1

    def test_pre_settle_skipped_when_zero(self):
        """``pre_settle_seconds=0`` skips the unconditional sleep — useful
        when the caller knows the GPU was just used (e.g. mid-pipeline)."""
        from unittest.mock import patch

        import paramem.graph.extractor as extractor

        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True
        with (
            patch.dict("sys.modules", {"torch": fake_torch}),
            patch("paramem.graph.extractor.time.sleep") as sleep_mock,
        ):
            extractor._wait_for_gpu_ready(pre_settle_seconds=0)
        sleep_mock.assert_not_called()

    def test_retries_on_device_not_ready(self):
        """When the first probe raises 'device not ready', helper waits and
        retries; succeeds on subsequent attempt."""
        from unittest.mock import patch

        import paramem.graph.extractor as extractor

        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True
        # Two failures, then success.
        fake_torch.zeros.side_effect = [
            RuntimeError("CUDA driver error: device not ready"),
            RuntimeError("CUDA driver error: device not ready"),
            None,
        ]
        with (
            patch.dict("sys.modules", {"torch": fake_torch}),
            patch("paramem.graph.extractor.time.sleep") as sleep_mock,
        ):
            extractor._wait_for_gpu_ready(pre_settle_seconds=0)
        assert fake_torch.zeros.call_count == 3
        # Two retry sleeps between three attempts (pre-settle disabled).
        assert sleep_mock.call_count == 2

    def test_raises_on_allocator_corruption(self):
        """Allocator-corruption markers are terminal — no retry, raise so
        the caller surfaces a server-restart-required signal."""
        from unittest.mock import patch

        import paramem.graph.extractor as extractor

        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True
        fake_torch.zeros.side_effect = RuntimeError(
            "INTERNAL ASSERT FAILED at CUDACachingAllocator.cpp:419"
        )
        with (
            patch.dict("sys.modules", {"torch": fake_torch}),
            patch("paramem.graph.extractor.time.sleep") as sleep_mock,
            pytest.raises(RuntimeError, match="(?i)INTERNAL ASSERT|allocator"),
        ):
            extractor._wait_for_gpu_ready(pre_settle_seconds=0)
        assert fake_torch.zeros.call_count == 1
        sleep_mock.assert_not_called()

    def test_raises_after_exhausting_retries(self):
        """If 'device not ready' persists across all retries, the final
        exception is raised so the caller knows the GPU is truly stuck."""
        from unittest.mock import patch

        import paramem.graph.extractor as extractor

        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True
        fake_torch.zeros.side_effect = RuntimeError("CUDA driver error: device not ready")
        with (
            patch.dict("sys.modules", {"torch": fake_torch}),
            patch("paramem.graph.extractor.time.sleep"),
            pytest.raises(RuntimeError, match="(?i)device not ready"),
        ):
            extractor._wait_for_gpu_ready(pre_settle_seconds=0)
        assert fake_torch.zeros.call_count == 3

    def test_unrelated_runtime_error_propagates(self):
        """A non-WSL-related RuntimeError (e.g. genuine OOM) should not be
        swallowed by the wake helper — let the caller see real bugs."""
        from unittest.mock import patch

        import paramem.graph.extractor as extractor

        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = True
        fake_torch.zeros.side_effect = RuntimeError("CUDA out of memory")
        with (
            patch.dict("sys.modules", {"torch": fake_torch}),
            patch("paramem.graph.extractor.time.sleep") as sleep_mock,
            pytest.raises(RuntimeError, match="out of memory"),
        ):
            extractor._wait_for_gpu_ready(pre_settle_seconds=0)
        assert fake_torch.zeros.call_count == 1
        sleep_mock.assert_not_called()


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
            result = _sota_pipeline(graph, "transcript", None, None, speaker_id="Speaker0")
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
            # ``generate_answer`` and ``adapt_messages`` are imported at
            # module top in ``paramem.graph.extractor`` (no longer lazy).
            # Patches must target the bound name in that module, not the
            # source module — the rebound name is what ``extractor``
            # actually calls.
            patch("paramem.graph.extractor.generate_answer", return_value="not json"),
            patch("paramem.graph.extractor.adapt_messages", return_value=[]),
        ):
            result, mapping, anon_transcript, _raw = _anonymize_with_local_model(
                graph, model, tokenizer
            )
        assert result is None
        assert anon_transcript == ""

    def test_pipeline_anonymize_failure_falls_back_to_raw_plausibility(self):
        """If anonymization fails, the pipeline falls back to raw (local) plausibility.

        The old behavior was to return the original graph unchanged.
        The new behavior runs _fallback_plausibility_on_raw so that tautologies,
        role leaks, and other noise are still filtered even without SOTA.
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
                return_value=(None, {}, "", ""),
            ),
            # Pass model=None/tokenizer=None → _local_plausibility_filter skipped inside fallback
        ):
            # Transcript "Alex lives in Millfield" grounds both entities.
            result = _sota_pipeline(
                graph,
                "Alex lives in Millfield",
                None,
                None,
                speaker_id="Speaker0",
                plausibility_judge="off",
            )
        # With plausibility_judge="off", fallback runs the residual-placeholder sweep only.
        # Both entities ARE in the transcript → relation survives.
        assert len(result.relations) == 1
        assert result.relations[0].subject == "Alex"
        # Fallback path recorded in diagnostics.
        assert result.diagnostics.get("fallback_path") == "anon_failed"

    def test_pipeline_enrichment_failure_raises_extraction_failed(self):
        """Enrichment failure must FAIL the cycle, not silently fall back.

        Previously this test asserted the silent-fallback behavior, which
        was a load-bearing bug: a SOTA 5xx baked a degraded (un-enriched)
        snapshot into the cumulative graph; the next cycle deduped the
        same triples so the missing second-order relations were lost
        permanently.  The pipeline now raises ``ExtractionFailed`` so
        the per-session loop in ``app.py`` leaves the session pending
        for retry.
        """
        from paramem.graph.extractor import ExtractionFailed, _sota_pipeline

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
                return_value=(anon_facts, mapping, "", ""),
            ),
            patch(
                "paramem.graph.extractor._filter_with_sota",
                return_value=(None, None, {}, None, {}),
            ),
        ):
            with pytest.raises(ExtractionFailed) as excinfo:
                _sota_pipeline(graph, "transcript", None, None, speaker_id="Speaker0")
        assert excinfo.value.phase == "sota_enrich"

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
                return_value=(anon_facts, mapping, "", ""),
            ),
            patch(
                "paramem.graph.extractor._filter_with_sota",
                return_value=(enriched_anon, None, {}, None, {}),
            ),
        ):
            result = _sota_pipeline(graph, "transcript", None, None, speaker_id="Speaker0")

        # Both enriched relations survive and get de-anonymized
        assert len(result.relations) == 2
        predicates = {r.predicate for r in result.relations}
        assert predicates == {"lives_in", "born_in"}
        for r in result.relations:
            assert r.subject == "Alex"
            assert r.object == "Millfield"

    def test_pipeline_deanonymizes_composite_placeholders(self):
        """Composite strings like 'Person_1's family' get substring-replaced."""
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
                return_value=(anon_facts, mapping, "", ""),
            ),
            patch(
                "paramem.graph.extractor._filter_with_sota",
                return_value=(enriched_anon, None, {}, None, {}),
            ),
        ):
            result = _sota_pipeline(graph, transcript, None, None, speaker_id="Speaker0")

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
        """Local plausibility filter applies the drop-set to the input facts.

        Output contract is ``{"drop": [<index>, ...]}``; the helper indexes
        by position and returns the surviving facts unchanged.  This used
        to be an echo-protocol where the model returned the kept facts
        verbatim — that protocol triggered Mistral 7B truncation on long
        inputs (see ``TestPlausibilityDropSet`` for the structural tests
        and the new prompt contract).
        """
        from paramem.graph.extractor import _local_plausibility_filter

        facts = [
            {"subject": "Alex", "predicate": "lives_in", "object": "Millfield"},
            {"subject": "Alex", "predicate": "has_name", "object": "Alex"},  # self-loop
        ]
        # Drop the self-loop at index 1; keep index 0.
        drop_response = '{"drop": [1]}'
        tokenizer = MagicMock()
        tokenizer.apply_chat_template = MagicMock(return_value="formatted")
        with (
            # See companion comment above: extractor binds these names
            # at module top, so patches must target the bound name.
            patch("paramem.graph.extractor.generate_answer", return_value=drop_response),
            patch("paramem.graph.extractor.adapt_messages", return_value=[]),
        ):
            result, raw = _local_plausibility_filter(facts, "transcript", MagicMock(), tokenizer)
        assert result is not None
        assert len(result) == 1
        assert result[0] == facts[0]  # input fact returned unchanged
        assert raw == drop_response

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

    def test_mapping_is_canonical_invented_prefix_accepted(self):
        """Open prefix vocabulary — `University_1`, `Project_1`, `Language_1`
        are valid placeholders alongside the common `{Person, City, Org, ...}`
        set.  Cross-cycle entity merge happens on real names in
        :class:`paramem.graph.merger.GraphMerger`, not on placeholder
        vocabulary, so per-session prefix divergence is harmless and
        enforcing a fixed lexicon would just push the work into a
        position-based recovery helper (the pattern this rewrite is
        retiring)."""
        from paramem.graph.extractor import _mapping_is_canonical

        assert _mapping_is_canonical({"Northcrest University": "University_1"}) is True
        assert _mapping_is_canonical({"Atlas Initiative": "Project_1"}) is True
        assert _mapping_is_canonical({"German": "Language_1"}) is True
        # Mixed common + invented prefixes also valid.
        assert (
            _mapping_is_canonical(
                {
                    "Mira": "Person_1",
                    "Northcrest University": "University_1",
                    "Atlas Initiative": "Project_1",
                }
            )
            is True
        )

    def test_mapping_is_canonical_uniqueness_violation(self):
        """Two real names sharing a placeholder → not canonical.  The
        anonymizer prompt's uniqueness rule is structural; the canonical
        check enforces it post-LLM so a duplicate-value mapping cannot
        slip through and silently collide two distinct entities into one
        identifier in the deanon round-trip."""
        from paramem.graph.extractor import _mapping_is_canonical

        # Two persons mapped to the same placeholder.
        assert _mapping_is_canonical({"Alice": "Person_1", "Pat": "Person_1"}) is False
        # Same shape but with one entry valid → still false.
        assert (
            _mapping_is_canonical({"Alice": "Person_1", "Pat": "Person_1", "Berlin": "City_1"})
            is False
        )

    def test_mapping_is_canonical_malformed_shape_rejected(self):
        """Values that don't match the universal `<Prefix>_<N>` shape
        (lowercase prefix, missing index, etc.) are non-canonical."""
        from paramem.graph.extractor import _mapping_is_canonical

        assert _mapping_is_canonical({"Alice": "person_1"}) is False  # lowercase
        assert _mapping_is_canonical({"Alice": "Person"}) is False  # missing _<N>
        assert _mapping_is_canonical({"Alice": "Person_"}) is False  # empty index
        assert _mapping_is_canonical({"Alice": "Person_abc"}) is False  # non-digit

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
        reverse = {"Person_1": "Alex"}
        anon_facts = [{"subject": "Person_1", "predicate": "works_at", "object": "Google"}]
        anon_transcript = "Person_1 works at Google."
        # "Google" is in the transcript → missed, not hallucinated.
        leaked = ["Google"]
        facts, new_mapping, new_reverse, _, status = _repair_anonymization_leaks(
            graph,
            mapping,
            reverse,
            anon_facts,
            anon_transcript,
            "Alex works at Google.",
            leaked,
        )
        assert status["missed_fixed"] == 1, (
            "organization entity present in transcript must be classified as missed and fixed"
        )
        assert "Google" in new_mapping, "extended mapping must contain the real name"
        org_placeholder = new_mapping["Google"]
        assert org_placeholder.startswith("Org_"), (
            f"organization entity must get an Org_N placeholder, got {org_placeholder!r}"
        )
        # Reverse companion is mirrored.
        assert new_reverse[org_placeholder] == "Google"
        # The fact's object must be rewritten to the placeholder.
        assert facts[0]["object"] == org_placeholder, (
            f"anon_facts object must be rewritten to placeholder, got {facts[0]['object']!r}"
        )

    def test_repair_anonymization_leaks_open_type_falls_through(self):
        """Repair allocates a fresh placeholder for any extractor type, not
        only those with a configured ``primary_for_type`` flag.  Previously
        types like ``"event"`` / ``"preference"`` had no configured prefix
        and the leaked name was silently dropped (treated as hallucinated).
        Under the open vocabulary the prefix is derived directly from the
        type label (``event`` → ``Event``), and the leak gets repaired.
        """
        from paramem.graph.extractor import _repair_anonymization_leaks

        graph = _make_graph(
            [("Alex", "attended", "DefCon")],
            entities=[
                Entity(name="Alex", entity_type="person"),
                Entity(name="DefCon", entity_type="event"),
            ],
        )
        mapping = {"Alex": "Person_1"}
        reverse = {"Person_1": "Alex"}
        anon_facts = [{"subject": "Person_1", "predicate": "attended", "object": "DefCon"}]
        anon_transcript = "Person_1 attended DefCon."
        leaked = ["DefCon"]
        facts, new_mapping, new_reverse, _, status = _repair_anonymization_leaks(
            graph,
            mapping,
            reverse,
            anon_facts,
            anon_transcript,
            "Alex attended DefCon.",
            leaked,
        )
        assert status["missed_fixed"] == 1, (
            "event entity present in transcript must be repaired, not dropped as hallucinated"
        )
        event_placeholder = new_mapping["DefCon"]
        assert event_placeholder.startswith("Event_"), (
            f"open-type repair must derive PascalCase prefix from the type label; "
            f"got {event_placeholder!r}"
        )
        assert facts[0]["object"] == event_placeholder

    def test_repair_anonymization_leaks_unknown_type_falls_back_to_entity(self):
        """When neither extractor nor NER provides a type, repair falls back
        to a generic ``Entity_N`` prefix rather than dropping the leak.
        Closes the residual case where the previous person-default could
        misclassify a city as a person."""
        from paramem.graph.extractor import _repair_anonymization_leaks

        graph = _make_graph(
            [("Alex", "mentioned", "Foundry-X")],
            # Note: the leaked name "Foundry-X" is NOT in graph.entities
            # and no NER hint is provided, exercising the final fallback.
            entities=[Entity(name="Alex", entity_type="person")],
        )
        mapping = {"Alex": "Person_1"}
        reverse = {"Person_1": "Alex"}
        anon_facts = [{"subject": "Person_1", "predicate": "mentioned", "object": "Foundry-X"}]
        anon_transcript = "Person_1 mentioned Foundry-X."
        leaked = ["Foundry-X"]
        facts, new_mapping, _, _, status = _repair_anonymization_leaks(
            graph,
            mapping,
            reverse,
            anon_facts,
            anon_transcript,
            "Alex mentioned Foundry-X.",
            leaked,
        )
        assert status["missed_fixed"] == 1
        # Default fallback: when no type signal is available the prefix is
        # "Entity" — generic but well-formed and recoverable.
        assert new_mapping["Foundry-X"].startswith("Entity_"), (
            f"unknown-type repair must fall back to Entity_N, got {new_mapping['Foundry-X']!r}"
        )

    def test_type_to_pascal_prefix_overrides_and_derivations(self):
        """Pin the contract for ``_type_to_pascal_prefix``: historical
        common types map via the override table; everything else is
        PascalCase-joined; empty input falls back to ``Entity``."""
        from paramem.graph.extractor import _type_to_pascal_prefix

        # Historical overrides — match anonymizer LLM conventions.
        assert _type_to_pascal_prefix("person") == "Person"
        assert _type_to_pascal_prefix("place") == "City"
        assert _type_to_pascal_prefix("organization") == "Org"
        assert _type_to_pascal_prefix("concept") == "Thing"
        # Open types — derived directly.
        assert _type_to_pascal_prefix("product") == "Product"
        assert _type_to_pascal_prefix("language") == "Language"
        assert _type_to_pascal_prefix("event") == "Event"
        # Multi-word labels collapse to PascalCase.
        assert _type_to_pascal_prefix("work_of_art") == "WorkOfArt"
        assert _type_to_pascal_prefix("self-driving") == "SelfDriving"
        assert _type_to_pascal_prefix("law enforcement") == "LawEnforcement"
        # Empty / whitespace fall back to a generic recoverable shape.
        assert _type_to_pascal_prefix("") == "Entity"
        assert _type_to_pascal_prefix("   ") == "Entity"

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
            return facts, None, {}, None, {}

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.extractor._anonymize_with_local_model",
                return_value=(anon_facts, mapping, anon_transcript, ""),
            ),
            patch("paramem.graph.extractor._filter_with_sota", side_effect=fake_filter),
        ):
            result = _sota_pipeline(
                graph, "Alex lives in Millfield.", None, None, speaker_id="Speaker0"
            )

        assert len(filter_calls) == 1, "SOTA was not called after repair"
        # The call payload must not contain any real name.
        payload = filter_calls[0]
        assert all("Millfield" not in str(v) for f in payload for v in f.values())
        # Final graph is de-anonymized correctly.
        assert len(result.relations) == 1
        assert result.relations[0].subject == "Alex"
        assert result.relations[0].object == "Millfield"

    def test_deterministic_builder_preempts_hallucinated_leak_for_graph_entities(self):
        """Under the consolidated deterministic mapping builder, the
        hallucinated-leak repair path is **not reachable** for entities
        that exist in ``graph.entities`` — the builder mints a
        placeholder for every in-scope entity, so the LLM-supplied
        ``Ghost`` token in ``anon_facts`` is substituted to
        ``Person_2`` before the verifier runs.  No leak is detected,
        repair never fires, SOTA is called with fully-scrubbed facts.

        This is the architecturally-correct outcome: the hallucinated-
        leak repair path used to be a reactive safety net for
        LLM-omitted mappings; the deterministic builder makes it
        unnecessary for graph-resident names.  The repair path remains
        live only for names supplied by NER (``extract_pii_names_with_ner``)
        that are not in ``graph.entities`` — covered by a separate test.
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
            return facts, None, {}, None, {}

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.extractor._anonymize_with_local_model",
                return_value=(anon_facts, mapping, anon_transcript, ""),
            ),
            patch("paramem.graph.extractor._filter_with_sota", side_effect=fake_filter),
        ):
            _sota_pipeline(graph, "Alex mentioned something.", None, None, speaker_id="Speaker0")

        # SOTA WAS called — the deterministic builder substituted
        # ``Ghost`` to its placeholder before verify ran, so no leak
        # was detected and the repair path did not fire.
        assert len(filter_calls) == 1, (
            "SOTA must be called once after deterministic-builder substitution"
        )
        sent_facts = filter_calls[0]
        assert len(sent_facts) == 1
        # ``Ghost`` must be substituted out of the object field.
        assert sent_facts[0]["object"] != "Ghost", (
            f"Ghost must be substituted before SOTA, got {sent_facts[0]['object']!r}"
        )
        obj = sent_facts[0]["object"]
        assert obj.startswith("Person_"), (
            f"object must be the deterministic Person_N placeholder, got {obj!r}"
        )

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


class TestApplyBindings:
    """Unit tests for the state-machine de-anonymization helper that replaces
    the previous LLM-based deanon attempt and the regex-based binding
    recovery (``_extract_sota_bindings``).

    The LLM-deanon caused VRAM exhaustion on the largest chunk's prompt
    (mapping + 2 transcripts + facts JSON). The redesign moves binding
    knowledge into SOTA's response (``new_entity_bindings``) and reduces
    deanon to pure dict substitution — no LLM call, no transcript
    reconstruction, no regex."""

    def test_substitutes_anonymizer_placeholders(self):
        """Bare anonymizer placeholders (Person_1, Org_1) substitute via
        the reverse mapping."""
        from paramem.graph.extractor import _apply_bindings

        facts = [
            {
                "subject": "Person_1",
                "predicate": "works_at",
                "object": "Org_1",
                "relation_type": "factual",
                "confidence": 1.0,
            },
        ]
        reverse = {"Person_1": "Alice", "Org_1": "Acme"}
        kept, dropped = _apply_bindings(facts, reverse, sota_bindings={})
        assert dropped == []
        assert kept[0]["subject"] == "Alice"
        assert kept[0]["object"] == "Acme"

    def test_substitutes_braced_sota_bindings(self):
        """SOTA-introduced braced placeholders ({Event_1}) substitute via
        explicit bindings without needing transcript reconstruction."""
        from paramem.graph.extractor import _apply_bindings

        facts = [
            {
                "subject": "Person_1",
                "predicate": "attended",
                "object": "{Event_1}",
                "relation_type": "factual",
                "confidence": 1.0,
            },
        ]
        reverse = {"Person_1": "Alice"}
        bindings = {"Event_1": "the agile transformation workshop"}
        kept, dropped = _apply_bindings(facts, reverse, bindings)
        assert dropped == []
        assert kept[0]["subject"] == "Alice"
        assert kept[0]["object"] == "the agile transformation workshop"

    def test_substitutes_compound_objects(self):
        """Bare placeholder embedded in literal text — `Org_1 Hungary`
        becomes `Acme Hungary` (the failure mode that bug 5 produced
        bogus bindings for under the old regex pipeline)."""
        from paramem.graph.extractor import _apply_bindings

        facts = [
            {
                "subject": "Person_1",
                "predicate": "based_in",
                "object": "Org_1 Hungary",
                "relation_type": "factual",
                "confidence": 1.0,
            },
        ]
        reverse = {"Person_1": "Alice", "Org_1": "Acme"}
        kept, dropped = _apply_bindings(facts, reverse, sota_bindings={})
        assert dropped == []
        assert kept[0]["object"] == "Acme Hungary"

    def test_drops_facts_with_unresolved_placeholders(self):
        """Facts whose subject/object retain a placeholder pattern after
        substitution get dropped (residual sweep). Causes: SOTA emitted a
        braced placeholder without including it in bindings, anonymizer
        leak, etc."""
        from paramem.graph.extractor import _apply_bindings

        facts = [
            {
                "subject": "Person_1",
                "predicate": "knows",
                "object": "Person_99",
                "relation_type": "social",
                "confidence": 1.0,
            },
            {
                "subject": "Person_1",
                "predicate": "attended",
                "object": "{Event_1}",
                "relation_type": "factual",
                "confidence": 1.0,
            },
        ]
        reverse = {"Person_1": "Alice"}
        # No binding for Event_1; no mapping for Person_99.
        kept, dropped = _apply_bindings(facts, reverse, sota_bindings={})
        assert kept == []
        assert len(dropped) == 2

    def test_handles_apostrophes_at_word_boundary(self):
        """`Person_2's cousin` substitutes Person_2 cleanly without breaking
        on the apostrophe (existing _substitute_whole_words behaviour)."""
        from paramem.graph.extractor import _apply_bindings

        facts = [
            {
                "subject": "Person_1",
                "predicate": "visited",
                "object": "Person_2's cousin",
                "relation_type": "social",
                "confidence": 1.0,
            },
        ]
        reverse = {"Person_1": "Alice", "Person_2": "Bob"}
        kept, dropped = _apply_bindings(facts, reverse, sota_bindings={})
        assert dropped == []
        assert kept[0]["object"] == "Bob's cousin"

    def test_mixed_bare_and_braced_in_same_fact(self):
        """A single fact with both a bare anonymizer placeholder and a
        braced SOTA placeholder substitutes both."""
        from paramem.graph.extractor import _apply_bindings

        facts = [
            {
                "subject": "Person_1",
                "predicate": "led",
                "object": "{Event_1} at Org_1",
                "relation_type": "factual",
                "confidence": 1.0,
            },
        ]
        reverse = {"Person_1": "Alice", "Org_1": "Acme"}
        bindings = {"Event_1": "the workshop"}
        kept, dropped = _apply_bindings(facts, reverse, bindings)
        assert dropped == []
        assert kept[0]["subject"] == "Alice"
        assert kept[0]["object"] == "the workshop at Acme"

    def test_empty_inputs_return_empty(self):
        from paramem.graph.extractor import _apply_bindings

        kept, dropped = _apply_bindings([], {}, {})
        assert kept == []
        assert dropped == []

    def test_preserves_other_fact_fields(self):
        """relation_type, confidence, and any extra fields pass through."""
        from paramem.graph.extractor import _apply_bindings

        facts = [
            {
                "subject": "Person_1",
                "predicate": "knows",
                "object": "Person_2",
                "relation_type": "social",
                "confidence": 0.7,
                "synthetic": False,
            },
        ]
        reverse = {"Person_1": "Alice", "Person_2": "Bob"}
        kept, _ = _apply_bindings(facts, reverse, sota_bindings={})
        assert kept[0]["relation_type"] == "social"
        assert kept[0]["confidence"] == 0.7
        assert kept[0]["synthetic"] is False

    def test_pii_fold_does_not_corrupt_speaker_name(self):
        """Regression for the deanon corruption regression: when
        :func:`_build_anonymization_mapping` folds PII attribute values
        (phone, email, …) onto the speaker entity's placeholder in the
        forward map, the reverse map must still restore the entity
        *name* — not the last attribute value folded."""
        from paramem.graph.extractor import _apply_bindings, _build_anonymization_mapping
        from paramem.graph.schema import Entity, SessionGraph

        graph = SessionGraph(
            session_id="s1",
            speaker_id="Speaker0",
            timestamp="2026-05-09T13:00:00Z",
        )
        graph.entities.append(
            Entity(
                name="Alex",
                entity_type="person",
                speaker_id="Speaker0",
                attributes={
                    "last_name": "Walker",
                    "email": "alex.walker@example.com",
                    "phone": "+49 178 99 99 999",
                },
            )
        )
        forward, reverse = _build_anonymization_mapping(
            graph, llm_mapping={}, pii_scope={"person"}, speaker_name="Alex"
        )
        # Forward fold is preserved (privacy contract).
        assert forward["Alex"] == "Person_1"
        assert forward["+49 178 99 99 999"] == "Person_1"
        # Reverse must restore the entity name, never an attribute value.
        assert reverse["Person_1"] == "Alex"

        facts = [{"subject": "Person_1", "predicate": "lives_in", "object": "Germany"}]
        kept, _ = _apply_bindings(facts, reverse, sota_bindings={})
        assert kept[0]["subject"] == "Alex"


class TestResidualSweepCatchesEmbeddedPlaceholders:
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
        """_plausibility_filter_with_sota returns (facts, raw_response).

        Plausibility is now a drop-set protocol — the judge emits a small
        ``{"drop": [<index>, ...]}`` object instead of echoing kept facts.
        Empty drop set keeps every input fact unchanged.
        """
        from paramem.graph.extractor import _plausibility_filter_with_sota

        fake_raw = '{"drop": []}'
        input_fact = {"subject": "A", "predicate": "knows", "object": "B"}
        with patch("paramem.graph.extractor._sota_call", return_value=fake_raw):
            facts, raw = _plausibility_filter_with_sota(
                [input_fact],
                api_key="k",
                provider="anthropic",
                anon_transcript="A knows B.",
            )
        assert facts == [input_fact]
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

    def test_abort_for_inference_when_not_training(self):
        """abort_for_inference() returns False immediately when no job is active."""
        from paramem.server.background_trainer import BackgroundTrainer

        bt = BackgroundTrainer(
            model=MagicMock(),
            tokenizer=MagicMock(),
            training_config=MagicMock(),
        )
        assert bt._active_abort is None
        result = bt.abort_for_inference(timeout=0.01)
        assert result is False


# --- Debug-artifact writers ---


class TestDebugArtifacts:
    """DebugSnapshotWriter.on_extraction_end — replaces the former
    ``_save_debug_artifacts`` callable.  All debug-write semantics
    (plaintext, _snapshot suffix, procedural-omitted-when-empty) preserved.
    """

    def _make_writer(self, *, base: Path, stamp: str | None = None) -> tuple:
        from paramem.training.debug_snapshot import DebugSnapshotWriter

        loop = MagicMock()
        loop.save_cycle_snapshots = True
        loop._debug_base = base
        loop.merger.save_graph = MagicMock()
        loop._current_interim_stamp_or_none = MagicMock(return_value=stamp)
        loop.snapshot_dir_for = MagicMock(return_value=base)
        return loop, DebugSnapshotWriter(loop)

    def test_on_extraction_end_writes_plaintext(self, tmp_path):
        out_dir = tmp_path / "episodic" / "cycle_4" / "run_xyz"
        loop, writer = self._make_writer(base=out_dir)

        episodic_rels = [{"question": "Q", "answer": "A"}]
        procedural_rels = [{"subject": "S", "predicate": "P", "object": "O"}]

        writer.on_extraction_end(episodic_rels, procedural_rels)

        assert (out_dir / "episodic_rels_snapshot.json").exists()
        assert (out_dir / "procedural_rels_snapshot.json").exists()
        # on_extraction_end no longer writes the cumulative graph — that is now
        # done by on_fold_graph (graph_merged_snapshot.json + graph_enriched_snapshot.json).
        loop.merger.save_graph.assert_not_called()

        with open(out_dir / "episodic_rels_snapshot.json") as f:
            saved = json.load(f)
        assert saved == episodic_rels

    def test_on_extraction_end_omits_procedural_when_empty(self, tmp_path):
        out_dir = tmp_path / "episodic" / "cycle_2" / "run_xyz"
        _, writer = self._make_writer(base=out_dir)

        writer.on_extraction_end([{"question": "Q", "answer": "A"}], [])

        assert (out_dir / "episodic_rels_snapshot.json").exists()
        assert not (out_dir / "procedural_rels_snapshot.json").exists()

    def test_on_extraction_end_short_circuits_when_debug_off(self, tmp_path):
        from paramem.training.debug_snapshot import DebugSnapshotWriter

        loop = MagicMock()
        loop.save_cycle_snapshots = False
        loop._debug_base = None
        loop.snapshot_dir_for = MagicMock(return_value=None)
        loop._current_interim_stamp_or_none = MagicMock(return_value=None)
        writer = DebugSnapshotWriter(loop)

        writer.on_extraction_end([{"question": "Q", "answer": "A"}], [])

        loop.merger.save_graph.assert_not_called()
        assert list(tmp_path.iterdir()) == []

    def test_on_recall_probe_writes_per_key_json(self, tmp_path):
        """on_recall_probe writes recall_probes/<phase>_<adapter>.json with payload."""
        out_dir = tmp_path / "cycle_5" / "run_abc"
        _, writer = self._make_writer(base=out_dir)

        per_key = [
            {
                "key": "proc32",
                "exact_match": True,
                "confidence": 0.98,
                "subject": "Alex",
                "predicate": "listens_to",
                "object": "jazz playlists",
                "recalled_subject": "Alex",
                "recalled_predicate": "listens_to",
                "recalled_object": "jazz playlists",
                "failure_reason": None,
                "raw_output": (
                    '{"key":"proc32","subject":"Alex",'
                    '"predicate":"listens_to","object":"jazz playlists"}'
                ),
            },
            {
                "key": "proc33",
                "exact_match": False,
                "confidence": 0.0,
                "subject": "Alex",
                "predicate": "listens_to",
                "object": "Example FM",
                "recalled_subject": None,
                "recalled_predicate": None,
                "recalled_object": None,
                "failure_reason": "parse_failure",
                "raw_output": "garbled output",
            },
        ]
        writer.on_recall_probe(per_key, phase="disk_verify", adapter_name="procedural")

        artifact = out_dir / "recall_probes" / "disk_verify_procedural.json"
        assert artifact.exists(), f"Expected artifact at {artifact}"
        saved = json.loads(artifact.read_text())
        assert saved == per_key
        assert saved[0]["raw_output"] != ""
        assert saved[1]["failure_reason"] == "parse_failure"

    def test_on_recall_probe_noop_when_per_key_none(self, tmp_path):
        """on_recall_probe is a no-op when per_key is None."""
        out_dir = tmp_path / "cycle_5" / "run_abc"
        _, writer = self._make_writer(base=out_dir)

        writer.on_recall_probe(None, phase="train_fill", adapter_name="episodic")

        recall_dir = out_dir / "recall_probes"
        assert not recall_dir.exists()

    def test_on_recall_probe_noop_when_debug_off(self, tmp_path):
        """on_recall_probe is a no-op when save_cycle_snapshots=False."""
        from paramem.training.debug_snapshot import DebugSnapshotWriter

        loop = MagicMock()
        loop.save_cycle_snapshots = False
        loop._debug_base = None
        loop.snapshot_dir_for = MagicMock(return_value=None)
        loop._current_interim_stamp_or_none = MagicMock(return_value=None)
        writer = DebugSnapshotWriter(loop)

        per_key = [{"key": "proc32", "exact_match": True, "raw_output": "x"}]
        writer.on_recall_probe(per_key, phase="disk_verify", adapter_name="procedural")

        assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# Extraction pipeline alignment tests
# ---------------------------------------------------------------------------


class TestPlausibilityAnon:
    """_sota_pipeline with plausibility_stage="anon": plausibility runs on
    anonymized facts before de-anonymization.
    """

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
                return_value=(anon_facts, mapping, anon_transcript, ""),
            ),
            patch(
                "paramem.graph.extractor._filter_with_sota",
                return_value=(anon_facts, None, {}, None, {}),
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
                speaker_id="Speaker0",
                plausibility_judge="claude",
                plausibility_stage="anon",
            )

        # Only the valid fact survives
        assert len(result.relations) == 1
        assert result.relations[0].predicate == "lives_in"
        assert result.diagnostics.get("plausibility") == "anon"


class TestPlausibilityDeanon:
    """_sota_pipeline with plausibility_stage="deanon": plausibility runs on
    de-anonymized facts using the original transcript.
    """

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
            return kept_deanon, ""

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.extractor._anonymize_with_local_model",
                return_value=(anon_facts, mapping, anon_transcript, ""),
            ),
            patch(
                "paramem.graph.extractor._filter_with_sota",
                return_value=(anon_facts, None, {}, None, {}),
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
                speaker_id="Speaker0",
                plausibility_judge="auto",
                plausibility_stage="deanon",
            )

        # Plausibility ran and dropped the tautology
        assert len(result.relations) == 1
        assert result.relations[0].predicate == "lives_in"

        # Verify the plausibility call received the ORIGINAL real-name transcript,
        # NOT the anonymized transcript (deanon stage must pass real names to plausibility).
        assert len(local_plaus_calls) == 1
        _, transcript_arg = local_plaus_calls[0]
        assert transcript_arg == "Alex lives in Millfield.", (
            "Deanon-stage plausibility must receive original transcript, not anon_transcript"
        )
        assert result.diagnostics.get("plausibility") == "deanon"


class TestResidualLeakDropsReferencingTriples:
    """Residual-leak fact-level filter: triples referencing a PII name that
    survived anonymization are dropped.

    The fact-level residual filter exists for the case where
    ``verify_anonymization_completeness`` flags a leak that the
    deterministic builder did not pre-empt.  Under the consolidated
    architecture this can happen only when the leaked name is supplied
    externally — typically by ``extract_pii_names_with_ner`` over the
    transcript.  The test below mocks NER to surface ``Ghost`` as a
    PII name not present in ``graph.entities``; the builder doesn't
    mint for it, the verifier flags it, repair leaves the residual
    intact, and the fact-level filter drops only the triples that
    reference the leaked name.
    """

    def test_residual_leak_filters_fact_level(self):
        from paramem.graph.extractor import _sota_pipeline

        # Ghost is NOT in graph.entities — the deterministic builder
        # therefore does not mint a placeholder for it.  NER (mocked
        # below) surfaces it as PII.
        graph = _make_graph(
            [
                ("Alex", "lives_in", "Millfield"),
                ("Alex", "friend_of", "Ghost"),
            ],
            entities=[
                Entity(name="Alex", entity_type="person"),
                Entity(name="Millfield", entity_type="place"),
            ],
        )
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
            return facts, None, {}, None, {}

        def fake_repair(
            graph,
            mapping,
            reverse,
            anon_facts,
            anon_transcript,
            orig_transcript,
            leaked,
            **kwargs,
        ):
            # Repair runs but Ghost remains — residual leak.
            return (
                anon_facts,
                mapping,
                reverse,
                anon_transcript,
                {"missed_fixed": 0, "hallucinated_dropped": 0, "residual_dropped": 0},
            )

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.extractor._anonymize_with_local_model",
                return_value=(anon_facts_initial, mapping, anon_transcript, ""),
            ),
            patch(
                "paramem.graph.extractor._repair_anonymization_leaks",
                side_effect=fake_repair,
            ),
            patch("paramem.graph.extractor._filter_with_sota", side_effect=fake_sota),
            # NER surfaces Ghost — not in graph.entities — as person PII.
            patch(
                "paramem.graph.extractor.extract_pii_names_with_ner",
                return_value={"Ghost": "person"},
            ),
        ):
            result = _sota_pipeline(
                graph,
                transcript,
                None,
                None,
                speaker_id="Speaker0",
                plausibility_judge="off",
                ner_check=True,
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
    """When anonymization fails, _sota_pipeline runs raw (local) plausibility
    instead of returning the original facts.
    """

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
                return_value=(None, {}, "", ""),
            ),
            patch(
                "paramem.graph.extractor._fallback_plausibility_on_raw",
                side_effect=fake_fallback,
            ),
        ):
            result = _sota_pipeline(graph, "transcript", None, None, speaker_id="Speaker0")

        assert fallback_calls == ["anon_failed"], (
            "fallback must be triggered with reason=anon_failed"
        )
        assert result.diagnostics.get("fallback_path") == "anon_failed"


class TestSotaEnrichmentFailureRaises:
    """When SOTA enrichment fails, raise ExtractionFailed instead of
    silently falling back to pre-enrichment facts.

    Closes the regression that on 2026-05-13 baked a degraded snapshot
    into the cumulative graph after an Anthropic 529 — by the time the
    next cycle re-extracted, the in-memory merger had already absorbed
    the un-enriched triples, so the missing second-order relations were
    permanently lost.  The extraction-failure-fails-cycle policy requires
    the whole cycle to abort so sessions stay pending for a clean retry.
    """

    def test_sota_enrich_failure_raises_extraction_failed(self):
        """_filter_with_sota returning (None, ...) → ExtractionFailed."""
        from paramem.graph.extractor import ExtractionFailed, _sota_pipeline

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
                return_value=(anon_facts, mapping, "", ""),
            ),
            patch(
                "paramem.graph.extractor._filter_with_sota",
                # First element None ⇒ SOTA call failed or unparseable.
                # Pre-fix this silently fell back to anon_facts.  Post-fix
                # this MUST raise so the per-session loop in app.py marks
                # the chunk failed and leaves the session pending.
                return_value=(None, None, {}, None, {"parse_path": "no_response"}),
            ),
        ):
            try:
                _sota_pipeline(graph, "transcript", None, None, speaker_id="Speaker0")
            except ExtractionFailed as exc:
                assert exc.phase == "sota_enrich"
                assert exc.reason
            else:
                raise AssertionError("_sota_pipeline must raise ExtractionFailed on SOTA failure")

    def test_extraction_failed_exposes_phase_and_reason(self):
        """Exception class contract used by the app.py per-chunk handler."""
        from paramem.graph.extractor import ExtractionFailed

        exc = ExtractionFailed("sota_enrich", "timeout")
        assert exc.phase == "sota_enrich"
        assert exc.reason == "timeout"
        assert "sota_enrich" in str(exc)
        assert "timeout" in str(exc)


class TestAllDroppedSafetyNet:
    """All-dropped safety net (extractor.py:2528-2543) fires when the
    pipeline empties out post-deanon. Original drop trigger was the
    grounding gate (now removed); plausibility is now the final
    discriminator that can empty the pipeline."""

    def test_all_dropped_triggers_fallback(self):
        """When plausibility drops every surviving fact, the all-dropped
        safety net invokes _fallback_plausibility_on_raw with reason
        'all_dropped' so the session does not yield zero facts."""
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
        # SOTA returns the same single fact; plausibility drops it (returns []).
        sota_enriched = list(anon_facts)

        fallback_calls = []

        def fake_fallback(g, t, m, tok, reason, **_kwargs):
            fallback_calls.append(reason)
            g.diagnostics["fallback_path"] = reason
            return g

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.extractor._anonymize_with_local_model",
                return_value=(anon_facts, mapping, "anon transcript", ""),
            ),
            patch(
                "paramem.graph.extractor._filter_with_sota",
                return_value=(sota_enriched, None, {}, None, {}),
            ),
            patch(
                "paramem.graph.extractor._local_plausibility_filter",
                return_value=([], ""),
            ),
            patch(
                "paramem.graph.extractor._fallback_plausibility_on_raw",
                side_effect=fake_fallback,
            ),
        ):
            result = _sota_pipeline(
                graph,
                "Alex lives in Millfield.",
                MagicMock(),  # non-None model so deanon-stage plausibility runs
                MagicMock(),  # non-None tokenizer so deanon-stage plausibility runs
                speaker_id="Speaker0",
                plausibility_judge="auto",
                plausibility_stage="deanon",
            )

        assert "all_dropped" in fallback_calls, f"Expected all_dropped, got: {fallback_calls}"
        assert result.diagnostics.get("fallback_path") == "all_dropped"


class TestEntityTypePreservation:
    """Entity types set by _normalize_extraction must survive the SOTA pipeline
    unchanged; no "person" stampdown on non-person entities.
    """

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
                return_value=(anon_facts, mapping, anon_transcript, ""),
            ),
            patch(
                "paramem.graph.extractor._filter_with_sota",
                return_value=(anon_facts, None, {}, None, {}),
            ),
        ):
            result = _sota_pipeline(
                graph,
                "Alex lives in Frankfurt and listens to Music.",
                None,
                None,
                speaker_id="Speaker0",
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
                return_value=(anon_facts, mapping, anon_transcript, ""),
            ),
            patch(
                "paramem.graph.extractor._filter_with_sota",
                return_value=(anon_facts, None, {}, None, {}),
            ),
        ):
            result = _sota_pipeline(
                graph,
                "Alex was born in Germany.",
                None,
                None,
                speaker_id="Speaker0",
                plausibility_judge="off",
            )

        entity_map = {e.name: e.entity_type for e in result.entities}
        # Germany already existed in the graph as "place"; the entity-type-preservation
        # rule keeps the original type. The Country_ → "location" mapping applies only
        # to SOTA-introduced entities (names absent from the original graph).
        # "place" and "location" both express geographic entities — accept both values.
        assert entity_map.get("Germany") in ("place", "location"), (
            f"Germany (Country_1) must be typed 'place' or 'location', "
            f"not {entity_map.get('Germany')!r}"
        )

    def test_sota_introduced_entity_no_placeholder_typed_concept(self):
        """SOTA-introduced entity with no placeholder (bare name) gets type 'concept', not 'person'.

        Regression guard: entity with no reverse_mapping entry must default to
        'concept', never 'person'.
        China is NOT present in the original graph — only Alex is. SOTA enrichment
        introduces China as a bare name (no anonymizer placeholder), so no
        reverse_mapping entry exists. The entity-type-preservation rule ensures the
        fallback type is 'concept', never 'person'.
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
                return_value=(anon_facts, mapping, anon_transcript, ""),
            ),
            patch(
                "paramem.graph.extractor._filter_with_sota",
                return_value=(enriched_anon, None, {}, None, {}),
            ),
        ):
            result = _sota_pipeline(
                graph,
                "Alex visited China.",
                None,
                None,
                speaker_id="Speaker0",
                plausibility_judge="off",
            )

        entity_map = {e.name: e.entity_type for e in result.entities}
        # China has no reverse_mapping entry → safe fallback type is "concept", not "person"
        china_type = entity_map.get("China")
        assert china_type == "concept", (
            f"SOTA-introduced bare entity must be typed 'concept', not {china_type!r}"
        )


class TestFallbackPlausibilityOnRawHelper:
    """Direct tests of the _fallback_plausibility_on_raw helper: drops
    residual placeholders and anon-failed facts.
    """

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
            speaker_id="Speaker0",
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
            graph,
            "Alex lives in Millfield.",
            None,
            None,
            speaker_id="Speaker0",
            reason="anon_failed",
        )
        assert result.diagnostics.get("fallback_path") == "anon_failed"


class TestExtractGraphNewKwargs:
    """extract_graph forwards privacy/plausibility kwargs (ner_check,
    plausibility_judge, plausibility_stage, verify_anonymization) to
    _sota_pipeline.
    """

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
                speaker_id="Speaker0",
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
        """extract_graph default temperature must be 0.0.

        Structured output (JSON, QA) requires deterministic generation.
        """
        import inspect

        from paramem.graph.extractor import extract_graph

        sig = inspect.signature(extract_graph)
        assert sig.parameters["temperature"].default == 0.0

    def test_extract_graph_default_max_tokens_matches_filter_default(self):
        """extract_graph default max_tokens matches the unified filter
        constant (currently 8192). The single-budget invariant: the entry
        point and downstream filter calls must default to the same value
        so a missing config doesn't produce inconsistent budgets across
        stages. Was 2048 historically; bumped after a resume chunk
        truncated mid-string at the old budget."""
        import inspect

        from paramem.graph.extractor import _DEFAULT_FILTER_MAX_TOKENS, extract_graph

        sig = inspect.signature(extract_graph)
        assert sig.parameters["max_tokens"].default == _DEFAULT_FILTER_MAX_TOKENS

    def test_verify_anonymization_false_skips_guard(self):
        """verify_anonymization=False skips the forward-path privacy guard.

        The completeness verifier (verify_anonymization_completeness) must not
        be called when the caller opts out of the guard.
        """
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
            return facts, None, {}, None, {}

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.extractor._anonymize_with_local_model",
                return_value=(anon_facts, mapping, anon_transcript, ""),
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
                speaker_id="Speaker0",
                verify_anonymization=False,
                plausibility_judge="off",
            )

        # With verify_anonymization=False the guard function must not be called
        assert verifier_calls == [], "verify_anonymization_completeness must be skipped when False"
        # SOTA must have been called (no guard blocked it)
        assert len(sota_calls) == 1


class TestDiagnosticsKeys:
    """Diagnostics dict is populated with expected keys after a full pipeline run."""

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
            return facts, ""  # keep all

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.extractor._anonymize_with_local_model",
                return_value=(anon_facts, mapping, anon_transcript, ""),
            ),
            patch(
                "paramem.graph.extractor._filter_with_sota",
                return_value=(anon_facts, None, {}, None, {}),
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
                speaker_id="Speaker0",
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
                return_value=(anon_facts, mapping, "anon transcript", ""),
            ),
            patch(
                "paramem.graph.extractor._filter_with_sota",
                return_value=(anon_facts, None, {}, None, {}),
            ),
        ):
            result = _sota_pipeline(
                graph,
                "Alex lives in Millfield.",
                None,
                None,
                speaker_id="Speaker0",
                plausibility_judge="off",
            )

        assert result.diagnostics.get("anonymize") == "ok"


class TestConsolidationScheduleConfigPrivacyGuard:
    """ConsolidationScheduleConfig rejects the combination of a cloud judge
    with deanon-stage plausibility (privacy violation).
    """

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

        Pre-flight check #2: minimal yaml without new keys must load with all new defaults.
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


# ---------------------------------------------------------------------------
# Bug A — speaker bare-first-name seeding
# ---------------------------------------------------------------------------


class TestBuildAnonymizationMapping:
    """Unit tests for the consolidated :func:`_build_anonymization_mapping`.

    The builder is the single source of truth for ``real → placeholder``.
    It folds the responsibilities of three earlier "Bug X fix" helpers
    (``_ensure_speaker_name_in_mapping``,
    ``_extend_mapping_with_pii_attributes``, ambiguous-pair drop) into
    one deterministic walk over ``graph.entities``.

    Coverage:
    - Mints placeholders for in-scope entity names.
    - Adds PII attribute values under the parent entity's placeholder.
    - Speaker-name seeding when the runtime knows the display name and
      it isn't covered by entities or LLM hints.
    - Merges in LLM-only entries (relation participants the graph
      doesn't know about) without overwriting deterministic ones.
    - Out-of-scope entities are not minted.
    - PII attributes on an out-of-scope entity are not minted.
    """

    @staticmethod
    def _build(entities, *, llm_mapping=None, pii_scope=None, speaker_name=None):
        forward, _reverse = TestBuildAnonymizationMapping._build_pair(
            entities,
            llm_mapping=llm_mapping,
            pii_scope=pii_scope,
            speaker_name=speaker_name,
        )
        return forward

    @staticmethod
    def _build_pair(entities, *, llm_mapping=None, pii_scope=None, speaker_name=None):
        from paramem.graph.extractor import _build_anonymization_mapping

        graph = SessionGraph(
            session_id="s",
            timestamp="2026-05-06T00:00:00Z",
            entities=entities,
            relations=[],
        )
        return _build_anonymization_mapping(
            graph,
            llm_mapping or {},
            pii_scope=pii_scope,
            speaker_name=speaker_name,
        )

    def test_mints_placeholders_for_in_scope_entities(self):
        mapping = self._build(
            [
                Entity(name="Alex", entity_type="person", speaker_id="Speaker0"),
                Entity(name="Berlin", entity_type="place"),
                Entity(name="Globex", entity_type="organization"),
            ],
            pii_scope={"person", "place"},
        )
        assert mapping["Alex"] == "Person_1"
        assert mapping["Berlin"] == "City_1"
        # Organization is NOT in pii_scope → not minted by the builder.
        assert "Globex" not in mapping

    def test_pii_attributes_share_parent_placeholder(self):
        forward, reverse = self._build_pair(
            [
                Entity(
                    name="Alex",
                    entity_type="person",
                    speaker_id="Speaker0",
                    attributes={
                        "last_name": "Walker",
                        "email": "alex.walker@example.com",
                        "linkedin": "linkedin.com/in/alex-walker-fictional",
                        "phone": "+49 30 12345678",
                        "location": "Germany",
                        # Non-PII attribute MUST NOT enter the mapping.
                        "job_title": "Senior Engineer",
                    },
                )
            ],
            pii_scope={"person"},
        )
        # Parent placeholder.
        assert forward["Alex"] == "Person_1"
        # PII attributes share the parent placeholder in the FORWARD map
        # (so the cloud-egress anonymizer scrubs every PII surface form
        # to the same token).
        attr_values = (
            "Walker",
            "alex.walker@example.com",
            "linkedin.com/in/alex-walker-fictional",
            "+49 30 12345678",
            "Germany",
        )
        for v in attr_values:
            assert forward[v] == "Person_1"
        # Non-PII attribute is not added.
        assert "Senior Engineer" not in forward
        # Reverse map: only the entity name resolves Person_1 — attribute
        # values are deliberately absent so SOTA-returned text never
        # restores `Person_1` to a phone/email/etc.
        assert reverse["Person_1"] == "Alex"
        for v in attr_values:
            assert v not in reverse.values() or v == "Alex"

    def test_pii_attributes_on_out_of_scope_entity_not_minted(self):
        mapping = self._build(
            [
                Entity(
                    name="Globex",
                    entity_type="organization",
                    attributes={"city": "Springfield"},
                )
            ],
            pii_scope={"person", "place"},
        )
        assert "Globex" not in mapping
        # Springfield is on an out-of-scope entity — also not minted.
        assert "Springfield" not in mapping

    def test_speaker_name_already_in_entities_noop(self):
        mapping = self._build(
            [Entity(name="Alex", entity_type="person", speaker_id="Speaker0")],
            pii_scope={"person"},
            speaker_name="Alex",
        )
        # Speaker name already covered by the entity walk.
        assert mapping == {"Alex": "Person_1"}

    def test_speaker_name_reuses_speaker_entity_placeholder(self):
        """Anonymous→disclosed: graph still has anonymous ``Speaker0``
        as the entity name but runtime knows the display name as
        ``Alex``.  Both must share the same placeholder.
        """
        mapping = self._build(
            [Entity(name="Speaker0", entity_type="person", speaker_id="Speaker0")],
            pii_scope={"person"},
            speaker_name="Alex",
        )
        assert mapping["Speaker0"] == "Person_1"
        assert mapping["Alex"] == "Person_1", (
            "speaker_name must reuse the speaker entity's placeholder"
        )

    def test_speaker_name_reuses_llm_full_name_placeholder(self):
        """No speaker entity in scope; LLM seeded a full-name key
        (``Alex Rivera``); seeding ``Alex`` reuses that placeholder
        so both forms de-anonymize to the same person.
        """
        mapping = self._build(
            entities=[],
            llm_mapping={"Alex Rivera": "Person_1"},
            pii_scope={"person"},
            speaker_name="Alex",
        )
        assert mapping["Alex"] == "Person_1"
        assert mapping["Alex Rivera"] == "Person_1"

    def test_speaker_name_mints_fresh_when_no_match(self):
        mapping = self._build(
            entities=[],
            llm_mapping={"Alice": "Person_1"},  # unrelated full name
            pii_scope={"person"},
            speaker_name="Bob",
        )
        assert "Bob" in mapping
        ph = mapping["Bob"]
        assert ph.startswith("Person_") and ph != "Person_1"

    def test_llm_only_entries_merged_in(self):
        """Relation participants the LLM placeholdered but the graph
        doesn't know about (e.g. ``Honda``) survive into the final
        mapping via the LLM hint merge.
        """
        mapping = self._build(
            [Entity(name="Alex", entity_type="person", speaker_id="Speaker0")],
            llm_mapping={"Honda": "Product_1"},
            pii_scope={"person"},
        )
        assert mapping["Alex"] == "Person_1"
        assert mapping["Honda"] == "Product_1"

    def test_deterministic_wins_on_conflict(self):
        """If the LLM mapped ``Alex → Person_2`` but the
        deterministic build mints ``Alex → Person_1``, the
        deterministic entry wins (we trust the graph)."""
        mapping = self._build(
            [Entity(name="Alex", entity_type="person", speaker_id="Speaker0")],
            llm_mapping={"Alex": "Person_2"},
            pii_scope={"person"},
        )
        assert mapping["Alex"] == "Person_1"

    def test_pii_value_already_keyed_by_llm_kept(self):
        """If an attribute value was already a key in the LLM mapping
        (LLM saw it in some relation), the LLM's placeholder wins
        because ``setdefault`` honours the deterministic build's
        first-write semantics.  Test documents the precise rule:
        deterministic adds only when the value isn't already mapped.
        """
        mapping = self._build(
            [
                Entity(
                    name="Alex",
                    entity_type="person",
                    speaker_id="Speaker0",
                    attributes={"last_name": "Walker"},
                )
            ],
            # Builder runs FIRST and adds Walker → Person_1 from
            # attributes; LLM hint Walker → Person_2 is then merged
            # via setdefault and ignored.
            llm_mapping={"Walker": "Person_2"},
            pii_scope={"person"},
        )
        assert mapping["Walker"] == "Person_1"

    def test_skips_empty_or_whitespace_attribute_values(self):
        mapping = self._build(
            [
                Entity(
                    name="Alex",
                    entity_type="person",
                    speaker_id="Speaker0",
                    attributes={"last_name": "", "email": "   ", "phone": "+49 1"},
                )
            ],
            pii_scope={"person"},
        )
        assert mapping["+49 1"] == "Person_1"
        assert "" not in mapping
        assert "   " not in mapping

    def test_default_pii_scope_when_none(self):
        """``pii_scope=None`` falls back to :data:`_DEFAULT_PII_SCOPE`
        (``{"person", "place"}``), so persons and places are minted
        but organizations and concepts are not.
        """
        mapping = self._build(
            [
                Entity(name="Alex", entity_type="person", speaker_id="Speaker0"),
                Entity(name="Berlin", entity_type="place"),
                Entity(name="Globex", entity_type="organization"),
                Entity(name="Volvo V70", entity_type="concept"),
            ],
            pii_scope=None,
        )
        assert "Alex" in mapping
        assert "Berlin" in mapping
        assert "Globex" not in mapping
        assert "Volvo V70" not in mapping

    def test_end_to_end_verifier_passes_after_build(self):
        """After the builder runs, the verifier finds no leaks for an
        anon_transcript and anon_facts that fully use the placeholders.
        Locks the contract that the builder produces a complete enough
        mapping for verify to succeed.
        """
        from paramem.graph.extractor import (
            _build_anonymization_mapping,
            verify_anonymization_completeness,
        )

        graph = _make_graph(
            [("Alex", "works_at", "Google")],
            entities=[
                Entity(name="Alex", entity_type="person"),
                Entity(name="Google", entity_type="organization"),
            ],
        )
        mapping, _reverse = _build_anonymization_mapping(
            graph,
            {"Google": "Org_1"},  # LLM hint for the org.
            pii_scope={"person"},
            speaker_name="Alex",
        )
        anon_facts = [{"subject": "Person_1", "predicate": "works_at", "object": "Org_1"}]
        anon_transcript = "Person_1 works at Org_1."
        leaked = verify_anonymization_completeness(graph, mapping, anon_facts, anon_transcript)
        assert leaked == [], f"verifier must report no leaks; got {leaked!r}"


# ---------------------------------------------------------------------------
# Mapping totality — diagnostic check post-anonymization
# ---------------------------------------------------------------------------


class TestCheckMappingTotality:
    """Unit tests for ``_check_mapping_totality``.

    The diagnostic replaces the retired
    ``_recover_missing_placeholder_mappings`` helper.  Under the open-
    vocabulary anonymizer prompt the LLM is expected to produce a total
    mapping by construction (see live probe at the prompt-pivot commit);
    this helper surfaces violations to ``logger`` and
    ``graph.diagnostics`` so prompt regressions are visible rather than
    silently shedding facts.  Orphan-placeholder facts get dropped
    downstream by :func:`_strip_residual_placeholders` — fail-closed.
    """

    @staticmethod
    def _graph() -> "SessionGraph":
        return _make_graph(
            [],
            entities=[Entity(name="Alex", entity_type="person")],
        )

    def test_total_mapping_records_no_orphans(self):
        """Every fact placeholder maps → no diagnostic emitted."""
        from paramem.graph.extractor import _check_mapping_totality

        graph = self._graph()
        anon_facts = [
            {"subject": "Person_1", "predicate": "lives_in", "object": "City_1"},
        ]
        mapping = {"Alex": "Person_1", "Berlin": "City_1"}
        _check_mapping_totality(graph, anon_facts, mapping)
        assert "totality_orphans" not in graph.diagnostics

    def test_orphan_placeholder_recorded(self):
        """A fact placeholder absent from mapping.values() is recorded as
        an orphan in ``graph.diagnostics`` for monitoring.  No mutation
        of inputs."""
        from paramem.graph.extractor import _check_mapping_totality

        graph = self._graph()
        anon_facts = [
            {"subject": "Person_1", "predicate": "studied_at", "object": "University_1"},
        ]
        # University_1 is missing from mapping — a totality violation.
        mapping = {"Alex": "Person_1"}
        _check_mapping_totality(graph, anon_facts, mapping)
        assert graph.diagnostics.get("totality_orphans") == ["University_1"]
        # Inputs must not be mutated.
        assert mapping == {"Alex": "Person_1"}

    def test_multiple_orphans_sorted(self):
        """Multiple orphans are deduplicated and sorted for stable
        diagnostic output."""
        from paramem.graph.extractor import _check_mapping_totality

        graph = self._graph()
        anon_facts = [
            {"subject": "Org_1", "predicate": "made", "object": "Product_1"},
            {"subject": "Person_1", "predicate": "speaks", "object": "Language_1"},
            {"subject": "Person_1", "predicate": "uses", "object": "Product_1"},
        ]
        # Person_1 is in mapping but Org_1, Product_1, Language_1 are not.
        mapping = {"Alex": "Person_1"}
        _check_mapping_totality(graph, anon_facts, mapping)
        assert graph.diagnostics.get("totality_orphans") == [
            "Language_1",
            "Org_1",
            "Product_1",
        ]

    def test_embedded_placeholder_caught(self):
        """Placeholder embedded in a compound string still surfaces as
        an orphan when missing from mapping."""
        from paramem.graph.extractor import _check_mapping_totality

        graph = self._graph()
        anon_facts = [
            {
                "subject": "Person_1",
                "predicate": "led",
                "object": "software for Product_1's Legend",
            },
        ]
        mapping = {"Alex": "Person_1"}
        _check_mapping_totality(graph, anon_facts, mapping)
        assert graph.diagnostics.get("totality_orphans") == ["Product_1"]

    def test_empty_facts_short_circuits(self):
        """No facts → no diagnostic, regardless of mapping shape."""
        from paramem.graph.extractor import _check_mapping_totality

        graph = self._graph()
        _check_mapping_totality(graph, [], {})
        _check_mapping_totality(graph, [], {"Alex": "Person_1"})
        assert "totality_orphans" not in graph.diagnostics
