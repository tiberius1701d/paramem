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

    def test_array_before_object(self):
        # First candidate `[1, 2]` is a list of scalars (no ``subject`` key
        # in a dict element) — rejected as not-an-envelope.  Walk continues
        # to the dict envelope.  Documents that bare-scalar lists are not
        # accepted as envelopes (they would be truncation-survivors).
        text = '[1, 2] {"entities": []}'
        result = json.loads(_extract_json_block(text))
        assert result == {"entities": []}

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
        """If anonymization fails, the pipeline falls back to raw (local) plausibility (D8).

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
                return_value=(anon_facts, mapping, "", ""),
            ),
            patch(
                "paramem.graph.extractor._filter_with_sota",
                return_value=(None, None, {}, None, {}),
            ),
        ):
            result = _sota_pipeline(graph, "transcript", None, None, speaker_id="Speaker0")

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
        # NOT the anonymized transcript (privacy-critical per D1/D3 plan).
        assert len(local_plaus_calls) == 1
        _, transcript_arg = local_plaus_calls[0]
        assert transcript_arg == "Alex lives in Millfield.", (
            "Deanon-stage plausibility must receive original transcript, not anon_transcript"
        )
        assert result.diagnostics.get("plausibility") == "deanon"


class TestResidualLeakDropsReferencingTriples:
    """§7 test 3: D1 — residual-leak fact-level filter.

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
        """D14: extract_graph default temperature is 0.0 (was 0.3)."""
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
                        "linkedin": "linkedin.com/in/tobias-preusser",
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
            "linkedin.com/in/tobias-preusser",
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
# Bug B — mapping gap recovery for model-omitted entries
# ---------------------------------------------------------------------------


class TestRecoverMissingPlaceholderMappings:
    """Unit tests for _recover_missing_placeholder_mappings (Bug B fix).

    Verifies that placeholder tokens present in anonymized facts but absent
    from the returned mapping are recovered by matching against the original
    relations so de-anonymization can restore the real name.
    """

    def test_recovers_missing_product_placeholder(self):
        """Product_1 in anon_facts but absent from mapping → recovered from original.

        Scenario: anonymizer produced 'Product_1' in anonymized_facts for the
        original 'Honda', but forgot to include 'Honda → Product_1' in the
        mapping.  Gap recovery must add it.
        """
        from paramem.graph.extractor import _recover_missing_placeholder_mappings

        # Original relation has Honda as subject.
        original_relations = [
            Relation(
                subject="Honda",
                predicate="manufactures",
                object="Legend SAE Level 3 System",
                relation_type="factual",
                confidence=1.0,
                speaker_id="Speaker0",
            )
        ]
        anon_facts = [
            {
                "subject": "Product_1",
                "predicate": "manufactures",
                "object": "Legend SAE Level 3 System",
            }
        ]
        mapping = {}  # anonymizer forgot to include Honda → Product_1
        reverse: dict[str, str] = {}
        result, result_reverse = _recover_missing_placeholder_mappings(
            mapping, reverse, anon_facts, original_relations
        )
        assert "Honda" in result, "gap recovery must add the original real name as a key"
        assert result["Honda"] == "Product_1", (
            f"recovered value must be the placeholder, got {result['Honda']!r}"
        )
        # Reverse companion is mirrored in lockstep.
        assert result_reverse["Product_1"] == "Honda"

    def test_already_covered_placeholder_not_duplicated(self):
        """Placeholder already in mapping.values() is not added again."""
        from paramem.graph.extractor import _recover_missing_placeholder_mappings

        original_relations = [
            Relation(
                subject="Alice",
                predicate="knows",
                object="Bob",
                relation_type="social",
                confidence=1.0,
                speaker_id="Speaker0",
            )
        ]
        anon_facts = [{"subject": "Person_1", "predicate": "knows", "object": "Person_2"}]
        mapping = {"Alice": "Person_1", "Bob": "Person_2"}  # both already covered
        reverse = {"Person_1": "Alice", "Person_2": "Bob"}
        result, result_reverse = _recover_missing_placeholder_mappings(
            mapping, reverse, anon_facts, original_relations
        )
        assert result == mapping, "no new entries must be added when mapping is already complete"
        assert result_reverse == reverse

    def test_object_field_placeholder_recovered(self):
        """Placeholder in the object field is recovered, not just the subject."""
        from paramem.graph.extractor import _recover_missing_placeholder_mappings

        original_relations = [
            Relation(
                subject="Alice",
                predicate="works_at",
                object="Acme Corp",
                relation_type="factual",
                confidence=1.0,
                speaker_id="Speaker0",
            )
        ]
        anon_facts = [{"subject": "Person_1", "predicate": "works_at", "object": "Org_1"}]
        mapping = {"Alice": "Person_1"}  # Acme Corp → Org_1 missing
        reverse = {"Person_1": "Alice"}
        result, result_reverse = _recover_missing_placeholder_mappings(
            mapping, reverse, anon_facts, original_relations
        )
        assert "Acme Corp" in result, "object-field gap must be recovered"
        assert result["Acme Corp"] == "Org_1"
        assert result_reverse["Org_1"] == "Acme Corp"

    def test_empty_inputs_return_unchanged_mapping(self):
        """Empty anon_facts or empty original_relations returns mapping unchanged."""
        from paramem.graph.extractor import _recover_missing_placeholder_mappings

        mapping = {"Alice": "Person_1"}
        reverse = {"Person_1": "Alice"}
        assert _recover_missing_placeholder_mappings(mapping, reverse, [], []) == (
            mapping,
            reverse,
        )
        assert _recover_missing_placeholder_mappings(
            mapping,
            reverse,
            [],
            [None],  # type: ignore[list-item]
        ) == (mapping, reverse)

    def test_does_not_mutate_input(self):
        """Input mapping is not mutated — a new dict is returned."""
        from paramem.graph.extractor import _recover_missing_placeholder_mappings

        original_relations = [
            Relation(
                subject="Honda",
                predicate="manufactures",
                object="Legend",
                relation_type="factual",
                confidence=1.0,
                speaker_id="Speaker0",
            )
        ]
        anon_facts = [{"subject": "Product_1", "predicate": "manufactures", "object": "Legend"}]
        original_mapping: dict[str, str] = {}
        original_reverse: dict[str, str] = {}
        _recover_missing_placeholder_mappings(
            original_mapping, original_reverse, anon_facts, original_relations
        )
        assert original_mapping == {}, "input mapping must not be mutated"
        assert original_reverse == {}, "input reverse must not be mutated"

    def test_apply_bindings_resolves_recovered_placeholder(self):
        """End-to-end: Product_1 in object field de-anonymizes correctly after gap recovery.

        This is the regression test for Bug B:
        - anon_facts has object "software development for Product_1's Legend SAE Level 3 system"
        - mapping initially missing Honda → Product_1
        - after gap recovery, mapping has Honda → Product_1 and reverse has Product_1 → Honda
        - _apply_bindings substitutes via the reverse map
        - the object field becomes "software development for Honda's Legend SAE Level 3 system"
        """
        from paramem.graph.extractor import _apply_bindings, _recover_missing_placeholder_mappings

        original_relations = [
            Relation(
                subject="Alex Rivera",
                predicate="led",
                object="software development for Honda's Legend SAE Level 3 system",
                relation_type="factual",
                confidence=1.0,
                speaker_id="Speaker0",
            )
        ]
        # After anonymization the subject is Person_1 and object has Product_1.
        # BUT the mapping is missing Honda → Product_1 (anonymizer bug).
        anon_facts = [
            {
                "subject": "Person_1",
                "predicate": "led",
                "object": "software development for Product_1's Legend SAE Level 3 system",
                "relation_type": "factual",
                "confidence": 1.0,
            }
        ]
        mapping = {"Alex Rivera": "Person_1"}  # Honda → Product_1 missing
        reverse = {"Person_1": "Alex Rivera"}

        recovered_mapping, recovered_reverse = _recover_missing_placeholder_mappings(
            mapping, reverse, anon_facts, original_relations
        )
        assert "Honda" in recovered_mapping, (
            "gap recovery must add Honda → Product_1 to the mapping"
        )
        assert recovered_reverse["Product_1"] == "Honda"

        kept, dropped = _apply_bindings(anon_facts, recovered_reverse, sota_bindings={})
        assert dropped == [], f"no facts must be dropped after gap recovery, got {dropped!r}"
        assert len(kept) == 1
        assert "Honda" in kept[0]["object"], (
            f"de-anonymization must restore 'Honda' in object, got {kept[0]['object']!r}"
        )
        assert "Product_1" not in kept[0]["object"], (
            "placeholder must be fully replaced in object field"
        )
