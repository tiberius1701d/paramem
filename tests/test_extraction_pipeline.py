"""Tests for the extraction pipeline — noise filter, JSON parsing."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from paramem.graph.extractor import _extract_json_block
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
            speaker_id="speaker0",
        )
        for r in relations
    ]
    return SessionGraph(
        session_id="test",
        timestamp="2026-04-09T00:00:00Z",
        entities=entities,
        relations=rels,
    )


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
        g = _parse_extraction(raw, "session1", speaker_id="speaker0")
        assert len(g.relations) == 1
        assert g.relations[0].subject == "Alice"
        assert g.relations[0].object == "Berlin"

    def test_empty_list(self):
        from paramem.graph.extractor import _parse_extraction

        g = _parse_extraction("[]", "session1", speaker_id="speaker0")
        assert len(g.relations) == 0
        assert len(g.entities) == 0


class TestParseFactsResponseSalvage:
    """``_parse_facts_response`` recovers fact dicts when the model emits a
    structured response but the outer JSON envelope is truncated (Mistral 7B
    on long KEEP-by-default plausibility passes hits EOS mid-array; the
    closing ``]`` never arrives).

    Without the salvage path the plausibility filter's strict array parse
    fails and ``local_plausibility_filter`` returns ``None`` — the gate
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
        """Synonym normalization — replace ``employed_by`` with ``worked_for``
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

    def test_reconstruct_transcript_word_boundary_not_substring(self):
        """``_reconstruct_updated_transcript`` routes through the shared
        ``_substitute_whole_words`` primitive — a binding span never
        matches inside a longer word.  ``{"Person_1": "Bill"}`` against
        ``"Billing department"`` must not substitute inside ``"Billing"``.

        Mutation: revert to the hand-rolled ``str.replace`` -> "Bill"
        matches the "Bill" prefix of "Billing" -> the transcript comes
        back corrupted as ``"{Person_1}ing department"`` -> this test
        fails.
        """
        from paramem.graph.extractor import _reconstruct_updated_transcript

        anon_transcript = "Please pay the Billing department, not Bill."
        bindings = {"Person_1": "Bill"}
        out = _reconstruct_updated_transcript(anon_transcript, bindings)
        assert out == "Please pay the Billing department, not {Person_1}."

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

    def test_inverted_binding_is_corrected_not_passed_through(self):
        """An inverted binding (key = real text, value = placeholder —
        the exact shape the SOTA bindings validator was previously
        missing, per the placeholder-contract refactor) is corrected to
        canonical ``{placeholder: real_text}`` direction rather than
        passed straight into the substitution map. Confirmed by transcript
        reconstruction: the real-text span is replaced by the placeholder,
        which only happens when the binding resolves in the right direction."""
        from paramem.graph.extractor import _apply_enrichment_delta

        anon = "Person_1 works at Acme."
        raw = '{"bindings": {"Acme": "Org_9"}}'
        _, transcript, bindings, _ = _apply_enrichment_delta([], raw, anon)
        assert bindings == {"Org_9": "Acme"}
        assert "{Org_9}" in transcript
        assert "Acme" not in transcript.replace("{Org_9}", "")

    def test_binding_both_sides_shaped_ties_to_declared_side(self):
        """A binding where BOTH sides happen to be placeholder-shaped
        (e.g. a real-world name like `Person_2` or `GPT_4`) is not
        ambiguous — the caller's declared `placeholder_side="key"` breaks
        the tie, so the binding is kept as-is rather than the whole
        delta losing the entry. Dropping here was a real regression:
        the same case previously survived at HEAD."""
        from paramem.graph.extractor import _apply_enrichment_delta

        raw = '{"bindings": {"Org_9": "Person_2"}}'
        _, _, bindings, counts = _apply_enrichment_delta([], raw, "text")
        assert bindings == {"Org_9": "Person_2"}
        assert counts["bindings_count"] == 1

    def test_ambiguous_binding_neither_shaped_is_dropped(self):
        """A binding where NEITHER side is placeholder-shaped is not a
        real SOTA mint binding and is dropped rather than accepted
        verbatim."""
        from paramem.graph.extractor import _apply_enrichment_delta

        raw = '{"bindings": {"my company": "Acme Corp"}}'
        _, _, bindings, counts = _apply_enrichment_delta([], raw, "text")
        assert bindings == {}
        assert counts["bindings_count"] == 0

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

    def test_add_entry_strips_non_fact_fields(self):
        """An ``add`` entry carrying a non-fact key (``evidence``)
        alongside the fact proper has that key stripped at the parse
        boundary, so it never enters ``enriched_anon`` (and therefore
        can never sink a valid fact at the residual sweep downstream).
        The fact itself is kept, only the extra key is dropped."""
        from paramem.graph.extractor import _apply_enrichment_delta

        raw = (
            '{"add": [{"subject": "Person_1", "predicate": "works_at",'
            ' "object": "Org_1", "relation_type": "factual", "confidence": 0.9,'
            ' "evidence": "Person_1 said they work at Org_1"}]}'
        )
        out, _, _, _ = _apply_enrichment_delta([], raw, None)
        assert out is not None
        assert len(out) == 1
        assert "evidence" not in out[0]
        assert out[0]["subject"] == "Person_1"
        assert out[0]["object"] == "Org_1"

    def test_modify_fields_strips_non_fact_fields(self):
        """A ``modify`` entry's ``fields`` dict is
        restricted the same way: ``relation_type``/``confidence``
        updates apply normally, a stray ``evidence`` key does not."""
        from paramem.graph.extractor import _apply_enrichment_delta

        facts = [{"subject": "Alex", "predicate": "employed_by", "object": "Acme"}]
        raw = (
            '{"modify": [{"index": 0, "fields": {"predicate": "worked_for",'
            ' "evidence": "she confirmed this"}}]}'
        )
        out, _, _, _ = _apply_enrichment_delta(facts, raw, None)
        assert out is not None
        assert out[0]["predicate"] == "worked_for"
        assert "evidence" not in out[0]

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

    def test_extract_and_anonymize_for_cloud_pins_anonymizer_default(self):
        """``extract_and_anonymize_for_cloud`` (chat
        egress) must call ``anonymize_for_cloud`` with the anonymizer's
        own default budget (``_DEFAULT_ANONYMIZER_MAX_TOKENS`` = 2048), not
        silently inherit ``anonymize_for_cloud``'s own default — which is
        ``_DEFAULT_FILTER_MAX_TOKENS`` (8192), sized for the graph-tier
        enrichment filter call. Chat egress is user-facing: a pathological
        non-terminating generation must not run 4x longer before the cap
        stops it.
        """
        from paramem.graph.extractor import (
            _DEFAULT_ANONYMIZER_MAX_TOKENS,
            extract_and_anonymize_for_cloud,
        )

        graph = _make_graph([("Alex", "lives_in", "Millfield")])
        captured = {}

        def fake_anonymize_for_cloud(*args, **kwargs):
            captured.update(kwargs)
            from paramem.graph.cloud_egress import AnonymizedPayload

            return AnonymizedPayload(
                status="ok",
                forward={},
                reverse={},
                anon_transcript="anon",
                anon_facts=[],
                declared=frozenset(),
                norm_stats={"inverted": 0, "dropped": 0},
                rekey_dropped=0,
                raw="",
            )

        model = MagicMock()
        model.is_gradient_checkpointing = False
        tokenizer = MagicMock()

        with (
            patch("paramem.graph.extractor.extract_graph", return_value=graph),
            patch(
                "paramem.graph.extractor.anonymize_for_cloud",
                side_effect=fake_anonymize_for_cloud,
            ),
        ):
            extract_and_anonymize_for_cloud(
                "Alex lives in Millfield.",
                model,
                tokenizer,
                scrub={"person name"},
            )

        assert captured.get("max_tokens") == _DEFAULT_ANONYMIZER_MAX_TOKENS


class TestPipelinePromptsDirThreading:
    """A ``prompts_dir`` override passed to ``extract_graph`` must reach
    every prompt load ``_sota_pipeline`` performs, not just the anonymizer
    call ``extract_and_anonymize_for_cloud`` already wired.  Each stage is
    exercised through ``_sota_pipeline`` itself (never by calling the
    downstream helper directly) so the assertion covers the exact call
    site that was silently dropping the override.
    """

    def test_anonymize_receives_prompts_dir(self, tmp_path):
        """Stage 1 (anonymize): without this the pipeline silently loads the
        shipped anonymization prompt while the caller believes its override
        is in effect."""
        from tests._sota_flow import run_sota_stages

        graph = _make_graph(
            [("Alex", "lives_in", "Millfield")],
            entities=[
                Entity(name="Alex", entity_type="person"),
                Entity(name="Millfield", entity_type="place"),
            ],
        )
        mapping = {"Alex": "Person_1", "Millfield": "City_1"}
        anon_facts = [{"subject": "Person_1", "predicate": "lives_in", "object": "City_1"}]
        captured = []

        def fake_anonymize(*args, **kwargs):
            captured.append(kwargs.get("prompts_dir"))
            return mapping, "anonymized transcript", ""

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.cloud_egress.anonymize_with_local_model",
                side_effect=fake_anonymize,
            ),
            patch(
                "paramem.graph.extractor._filter_with_sota",
                return_value=(anon_facts, None, {}, None, {}),
            ),
        ):
            run_sota_stages(
                graph,
                "Alex lives in Millfield.",
                None,
                None,
                speaker_id="speaker0",
                correction_entity_types=set(),
                scrub={"person name"},
                prompts_dir=tmp_path,
            )

        assert captured == [tmp_path], (
            f"anonymize_with_local_model must receive the caller's prompts_dir, got {captured!r}"
        )

    def test_sota_enrich_receives_prompts_dir(self, tmp_path):
        """Stage 2 (sota_enrich): ``_filter_with_sota`` had neither a
        ``prompts_dir`` parameter nor a forwarded value — the override never
        reached the enrichment prompt at all."""
        from tests._sota_flow import run_sota_stages

        graph = _make_graph(
            [("Alex", "lives_in", "Millfield")],
            entities=[
                Entity(name="Alex", entity_type="person"),
                Entity(name="Millfield", entity_type="place"),
            ],
        )
        mapping = {"Alex": "Person_1", "Millfield": "City_1"}
        anon_facts = [{"subject": "Person_1", "predicate": "lives_in", "object": "City_1"}]
        captured = []

        def fake_filter_with_sota(*args, **kwargs):
            captured.append(kwargs.get("prompts_dir"))
            return anon_facts, None, {}, None, {}

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.cloud_egress.anonymize_with_local_model",
                return_value=(mapping, "anonymized transcript", ""),
            ),
            patch(
                "paramem.graph.extractor._filter_with_sota",
                side_effect=fake_filter_with_sota,
            ),
        ):
            run_sota_stages(
                graph,
                "Alex lives in Millfield.",
                None,
                None,
                speaker_id="speaker0",
                correction_entity_types=set(),
                scrub={"person name"},
                prompts_dir=tmp_path,
            )

        assert captured == [tmp_path], (
            f"_filter_with_sota must receive the caller's prompts_dir, got {captured!r}"
        )

    def test_anon_plausibility_receives_prompts_dir(self, tmp_path):
        """Stage 3a (anon_plausibility, SOTA judge): ``_plausibility_filter_with_sota``
        had neither a ``prompts_dir`` parameter nor a forwarded value."""
        from tests._sota_flow import run_sota_stages

        graph = _make_graph(
            [("Alex", "lives_in", "Millfield")],
            entities=[
                Entity(name="Alex", entity_type="person"),
                Entity(name="Millfield", entity_type="place"),
            ],
        )
        mapping = {"Alex": "Person_1", "Millfield": "City_1"}
        anon_facts = [{"subject": "Person_1", "predicate": "lives_in", "object": "City_1"}]
        captured = []

        def fake_plaus(facts, api_key, **kwargs):
            captured.append(kwargs.get("prompts_dir"))
            return facts, "raw"

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.cloud_egress.anonymize_with_local_model",
                return_value=(mapping, "anonymized transcript", ""),
            ),
            patch(
                "paramem.graph.extractor._filter_with_sota",
                return_value=(anon_facts, None, {}, None, {}),
            ),
            patch(
                "paramem.graph.extractor._plausibility_filter_with_sota",
                side_effect=fake_plaus,
            ),
        ):
            run_sota_stages(
                graph,
                "Alex lives in Millfield.",
                None,
                None,
                speaker_id="speaker0",
                plausibility_judge="claude",
                plausibility_stage="anon",
                correction_entity_types=set(),
                scrub={"person name"},
                prompts_dir=tmp_path,
            )

        assert captured == [tmp_path], (
            f"_plausibility_filter_with_sota must receive the caller's prompts_dir, "
            f"got {captured!r}"
        )

    def test_deanon_plausibility_receives_prompts_dir(self, tmp_path):
        """Stage 3d (deanon_plausibility, local judge): ``local_plausibility_filter``
        already accepted ``prompts_dir`` but the call site never passed it."""
        from tests._sota_flow import run_sota_stages

        graph = _make_graph(
            [("Alex", "lives_in", "Millfield")],
            entities=[
                Entity(name="Alex", entity_type="person"),
                Entity(name="Millfield", entity_type="place"),
            ],
        )
        mapping = {"Alex": "Person_1", "Millfield": "City_1"}
        anon_facts = [{"subject": "Person_1", "predicate": "lives_in", "object": "City_1"}]
        captured = []

        def fake_local_plaus(facts, transcript, model, tokenizer, **kwargs):
            captured.append(kwargs.get("prompts_dir"))
            return facts, ""

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.cloud_egress.anonymize_with_local_model",
                return_value=(mapping, "anonymized transcript", ""),
            ),
            patch(
                "paramem.graph.extractor._filter_with_sota",
                return_value=(anon_facts, None, {}, None, {}),
            ),
            patch(
                "paramem.graph.extractor.local_plausibility_filter",
                side_effect=fake_local_plaus,
            ),
        ):
            run_sota_stages(
                graph,
                "Alex lives in Millfield.",
                MagicMock(),
                MagicMock(),
                speaker_id="speaker0",
                plausibility_judge="auto",
                plausibility_stage="deanon",
                correction_entity_types=set(),
                scrub={"person name"},
                prompts_dir=tmp_path,
            )

        assert captured == [tmp_path], (
            f"local_plausibility_filter must receive the caller's prompts_dir, got {captured!r}"
        )

    def test_default_prompts_dir_is_none_at_anon_stage_call_sites(self):
        """Parity check (plausibility_stage="anon"): when the caller does not
        pass ``prompts_dir`` (production default), every downstream call
        still receives ``None`` — byte-identical to pre-fix behaviour, never
        a surprise override."""
        from tests._sota_flow import run_sota_stages

        graph = _make_graph(
            [("Alex", "lives_in", "Millfield")],
            entities=[
                Entity(name="Alex", entity_type="person"),
                Entity(name="Millfield", entity_type="place"),
            ],
        )
        mapping = {"Alex": "Person_1", "Millfield": "City_1"}
        anon_facts = [{"subject": "Person_1", "predicate": "lives_in", "object": "City_1"}]
        captured = {}

        def fake_anonymize(*args, **kwargs):
            captured["anonymize"] = kwargs.get("prompts_dir")
            return mapping, "anonymized transcript", ""

        def fake_filter_with_sota(*args, **kwargs):
            captured["sota_enrich"] = kwargs.get("prompts_dir")
            return anon_facts, None, {}, None, {}

        def fake_plaus(facts, api_key, **kwargs):
            captured["anon_plausibility"] = kwargs.get("prompts_dir")
            return facts, "raw"

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.cloud_egress.anonymize_with_local_model",
                side_effect=fake_anonymize,
            ),
            patch(
                "paramem.graph.extractor._filter_with_sota",
                side_effect=fake_filter_with_sota,
            ),
            patch(
                "paramem.graph.extractor._plausibility_filter_with_sota",
                side_effect=fake_plaus,
            ),
        ):
            run_sota_stages(
                graph,
                "Alex lives in Millfield.",
                None,
                None,
                speaker_id="speaker0",
                plausibility_judge="claude",
                plausibility_stage="anon",
                correction_entity_types=set(),
                scrub={"person name"},
            )

        assert captured == {
            "anonymize": None,
            "sota_enrich": None,
            "anon_plausibility": None,
        }

    def test_default_prompts_dir_is_none_at_deanon_stage_call_sites(self):
        """Parity check (plausibility_stage="deanon"): same as above for the
        local-judge deanon-plausibility call site."""
        from tests._sota_flow import run_sota_stages

        graph = _make_graph(
            [("Alex", "lives_in", "Millfield")],
            entities=[
                Entity(name="Alex", entity_type="person"),
                Entity(name="Millfield", entity_type="place"),
            ],
        )
        mapping = {"Alex": "Person_1", "Millfield": "City_1"}
        anon_facts = [{"subject": "Person_1", "predicate": "lives_in", "object": "City_1"}]
        captured = {}

        def fake_anonymize(*args, **kwargs):
            captured["anonymize"] = kwargs.get("prompts_dir")
            return mapping, "anonymized transcript", ""

        def fake_filter_with_sota(*args, **kwargs):
            captured["sota_enrich"] = kwargs.get("prompts_dir")
            return anon_facts, None, {}, None, {}

        def fake_local_plaus(facts, transcript, model, tokenizer, **kwargs):
            captured["deanon_plausibility"] = kwargs.get("prompts_dir")
            return facts, ""

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.cloud_egress.anonymize_with_local_model",
                side_effect=fake_anonymize,
            ),
            patch(
                "paramem.graph.extractor._filter_with_sota",
                side_effect=fake_filter_with_sota,
            ),
            patch(
                "paramem.graph.extractor.local_plausibility_filter",
                side_effect=fake_local_plaus,
            ),
        ):
            run_sota_stages(
                graph,
                "Alex lives in Millfield.",
                MagicMock(),
                MagicMock(),
                speaker_id="speaker0",
                plausibility_judge="auto",
                plausibility_stage="deanon",
                correction_entity_types=set(),
                scrub={"person name"},
            )

        assert captured == {
            "anonymize": None,
            "sota_enrich": None,
            "deanon_plausibility": None,
        }


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
        from tests._sota_flow import run_sota_stages

        graph = _make_graph(
            [("Alex", "lives_in", "Millfield")],
            entities=[
                Entity(name="Alex", entity_type="person"),
                Entity(name="Millfield", entity_type="place"),
            ],
        )
        # No ANTHROPIC_API_KEY → skips gracefully
        with patch.dict("os.environ", {}, clear=True):
            result = run_sota_stages(
                graph,
                "transcript",
                None,
                None,
                speaker_id="speaker0",
                correction_entity_types=set(),
                scrub={"person name"},
            )
            # Should return original graph unchanged
            assert len(result.relations) == 1

    def test_anonymize_graceful_on_bad_output(self):
        from paramem.graph.cloud_egress import anonymize_with_local_model

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
            mapping, anon_transcript, _raw = anonymize_with_local_model(
                graph, model, tokenizer, scrub={"person name"}
            )
        assert mapping is None
        assert anon_transcript == ""

    def test_pipeline_anonymize_failure_falls_back_to_raw_plausibility(self):
        """If anonymization fails, the pipeline falls back to raw (local) plausibility.

        The old behavior was to return the original graph unchanged.
        The new behavior runs _fallback_plausibility_on_raw so that tautologies,
        role leaks, and other noise are still filtered even without SOTA.
        """
        from tests._sota_flow import run_sota_stages

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
                "paramem.graph.cloud_egress.anonymize_with_local_model",
                return_value=(None, "", ""),
            ),
            # Pass model=None/tokenizer=None → local_plausibility_filter skipped inside fallback
        ):
            # Transcript "Alex lives in Millfield" grounds both entities.
            result = run_sota_stages(
                graph,
                "Alex lives in Millfield",
                None,
                None,
                speaker_id="speaker0",
                plausibility_judge="off",
                correction_entity_types=set(),
                scrub={"person name"},
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
        from paramem.graph.extractor import ExtractionFailed
        from tests._sota_flow import run_sota_stages

        graph = _make_graph(
            [("Alex", "lives_in", "Millfield")],
            entities=[
                Entity(name="Alex", entity_type="person"),
                Entity(name="Millfield", entity_type="place"),
            ],
        )
        mapping = {"Alex": "Person_1", "Millfield": "City_1"}

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.cloud_egress.anonymize_with_local_model",
                return_value=(mapping, "anonymized transcript", ""),
            ),
            patch(
                "paramem.graph.extractor._filter_with_sota",
                return_value=(None, None, {}, None, {}),
            ),
        ):
            with pytest.raises(ExtractionFailed) as excinfo:
                run_sota_stages(
                    graph,
                    "transcript",
                    None,
                    None,
                    speaker_id="speaker0",
                    correction_entity_types=set(),
                    scrub={"person name"},
                )
        assert excinfo.value.phase == "sota_enrich"

    def test_pipeline_enriched_facts_get_deanonymized(self):
        """Enrichment output flows through de-anonymization to real names."""
        from tests._sota_flow import run_sota_stages

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
                "paramem.graph.cloud_egress.anonymize_with_local_model",
                return_value=(mapping, "anonymized transcript", ""),
            ),
            patch(
                "paramem.graph.extractor._filter_with_sota",
                return_value=(enriched_anon, None, {}, None, {}),
            ),
        ):
            result = run_sota_stages(
                graph,
                "transcript",
                None,
                None,
                speaker_id="speaker0",
                correction_entity_types=set(),
                scrub={"person name", "physical address"},
            )

        # Both enriched relations survive and get de-anonymized
        assert len(result.relations) == 2
        predicates = {r.predicate for r in result.relations}
        assert predicates == {"lives_in", "born_in"}
        for r in result.relations:
            assert r.subject == "Alex"
            assert r.object == "Millfield"

    def test_pipeline_deanonymizes_composite_placeholders(self):
        """Composite strings like 'Person_1's family' get substring-replaced."""
        from tests._sota_flow import run_sota_stages

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
                "paramem.graph.cloud_egress.anonymize_with_local_model",
                return_value=(mapping, "anonymized transcript", ""),
            ),
            patch(
                "paramem.graph.extractor._filter_with_sota",
                return_value=(enriched_anon, None, {}, None, {}),
            ),
        ):
            result = run_sota_stages(
                graph,
                transcript,
                None,
                None,
                speaker_id="speaker0",
                correction_entity_types=set(),
                scrub={"person name", "physical address"},
            )

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
        from paramem.graph.extractor import local_plausibility_filter

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
            result, raw = local_plausibility_filter(facts, "transcript", MagicMock(), tokenizer)
        assert result is not None
        assert len(result) == 1
        assert result[0] == facts[0]  # input fact returned unchanged
        assert raw == drop_response

    def test_normalize_anonymization_mapping_inverts_placeholder_keys(self):
        """Mapping with placeholder keys is inverted to {real: placeholder} canonical."""
        from paramem.graph.placeholders import _normalize_anonymization_mapping

        wrong_direction = {"Person_1": "Alex", "City_1": "Millfield"}
        normalized, stats = _normalize_anonymization_mapping(wrong_direction)
        assert normalized == {"Alex": "Person_1", "Millfield": "City_1"}
        assert stats == {"inverted": 2, "dropped": 0}

    def test_normalize_anonymization_mapping_keeps_canonical(self):
        """Mapping already in {real: placeholder} canonical form passes through."""
        from paramem.graph.placeholders import _normalize_anonymization_mapping

        canonical = {"Alex": "Person_1", "Millfield": "City_1"}
        normalized, stats = _normalize_anonymization_mapping(canonical)
        assert normalized == canonical
        assert stats == {"inverted": 0, "dropped": 0}

    def test_normalize_anonymization_mapping_empty(self):
        from paramem.graph.placeholders import _normalize_anonymization_mapping

        normalized, stats = _normalize_anonymization_mapping({})
        assert normalized == {}
        assert stats == {"inverted": 0, "dropped": 0}

    def test_entity_type_to_prefix_closed_vocab_and_derivations(self):
        """Pin the contract for ``entity_type_to_prefix``: closed-vocabulary
        common types map via schema.yaml's ``anonymizer_type_to_prefix()``;
        everything else is PascalCase-joined; empty input falls back to
        ``Entity``."""
        from paramem.graph.placeholders import entity_type_to_prefix

        # Closed vocabulary — match anonymizer LLM conventions.
        assert entity_type_to_prefix("person") == "Person"
        assert entity_type_to_prefix("place") == "City"
        assert entity_type_to_prefix("organization") == "Org"
        assert entity_type_to_prefix("concept") == "Thing"
        # Open types — derived directly.
        assert entity_type_to_prefix("product") == "Product"
        assert entity_type_to_prefix("language") == "Language"
        assert entity_type_to_prefix("event") == "Event"
        # Multi-word labels collapse to PascalCase.
        assert entity_type_to_prefix("work_of_art") == "WorkOfArt"
        assert entity_type_to_prefix("self-driving") == "SelfDriving"
        assert entity_type_to_prefix("law enforcement") == "LawEnforcement"
        # Empty / whitespace fall back to a generic recoverable shape.
        assert entity_type_to_prefix("") == "Entity"
        assert entity_type_to_prefix("   ") == "Entity"

    def test_pipeline_normalizes_mixed_direction_mapping_per_pair(self):
        """Mixed-direction mappings from the anonymizer are normalized per-pair.

        Real anonymizers sometimes emit a dict where some entries are
        `{real: placeholder}` and others are `{placeholder: real}`. Per-pair
        normalization independently canonicalizes each, enabling the pipeline
        to proceed (SOTA call, de-anonymization) rather than aborting.
        """
        from paramem.graph.placeholders import _normalize_anonymization_mapping

        mixed = {
            "Alex": "Person_1",  # canonical
            "Person_2": "Millfield",  # inverted
        }
        out, stats = _normalize_anonymization_mapping(mixed)
        # Both pairs end up canonical: keys are real, values are placeholders.
        assert out == {"Alex": "Person_1", "Millfield": "Person_2"}
        assert stats == {"inverted": 1, "dropped": 0}

    def test_entity_correction_call_site_wired_into_pipeline(self):
        """Integration coverage for the entity_correction phase call site.

        Stubs ``paramem.graph.extractor.correct_entity_surfaces`` (the name
        bound in extractor's own namespace via its top-of-file import) so
        this proves the WIRING added to ``_sota_pipeline`` — the stub is
        called exactly once and its ``"applied"`` list lands verbatim on
        ``graph.diagnostics["entity_corrections"]`` — without needing a
        live model. Reuses the same ``anonymize_with_local_model`` /
        ``_filter_with_sota`` happy-path mocking pattern as the sibling
        tests in this class.
        """
        from tests._sota_flow import run_sota_stages

        graph = _make_graph(
            [("Alex", "lives_in", "Millfield")],
            entities=[
                Entity(name="Alex", entity_type="person"),
                Entity(name="Millfield", entity_type="place"),
            ],
        )
        anon_facts = [{"subject": "Person_1", "predicate": "lives_in", "object": "City_1"}]
        mapping = {"Alex": "Person_1", "Millfield": "City_1"}
        canned_applied = [
            {
                "locus": "placeholder",
                "placeholder": "City_1",
                "type": "place",
                "kind": "place",
                "before": "Frankfrut",
                "after": "Frankfurt",
            }
        ]
        canned_verdicts = [
            {
                "locus": "placeholder",
                "placeholder": "City_1",
                "type": "place",
                "kind": "place",
                "is_known_entity": True,
                "proposed": "Frankfurt",
                "applied": True,
                "reject_reason": None,
            }
        ]
        correction_calls = []

        def fake_correct_entity_surfaces(reverse_mapping, entities, model, tokenizer, **kwargs):
            correction_calls.append((dict(reverse_mapping), list(entities), kwargs))
            return {"applied": canned_applied, "verdicts": canned_verdicts}

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.cloud_egress.anonymize_with_local_model",
                return_value=(mapping, "anonymized transcript", ""),
            ),
            patch(
                "paramem.graph.extractor._filter_with_sota",
                return_value=(anon_facts, None, {}, None, {}),
            ),
            patch(
                "paramem.graph.extractor.correct_entity_surfaces",
                side_effect=fake_correct_entity_surfaces,
            ),
        ):
            result = run_sota_stages(
                graph,
                "transcript",
                None,
                None,
                speaker_id="speaker0",
                plausibility_judge="off",
                correction_entity_types={"place"},
                scrub={"person name"},
            )

        assert len(correction_calls) == 1, "correct_entity_surfaces must be called exactly once"
        assert result.diagnostics["entity_corrections"] == canned_applied
        assert result.diagnostics["entity_correction_verdicts"] == canned_verdicts


class TestAnonymizerMappingOnlyContract:
    """The anonymizer LLM returns exactly TWO artifacts: the
    ``mapping`` and its own ``anonymized_transcript`` rewrite. It never
    returns FACTS. The SCRIPT builds the anonymized fact array from
    ``graph.relations``; the anonymizer cannot lose, reword, or drop a
    fact because it never returns one.
    """

    def test_anonymizer_returns_mapping_and_transcript_only(self):
        """``anonymize_with_local_model`` returns exactly ``(mapping,
        anonymized_transcript, raw)`` — even when the model's raw
        response still smuggles fact-array keys (a model that hasn't
        fully adopted the mapping-only-for-FACTS contract).

        Mutation: re-add a fact branch to the parser (e.g. source a
        fourth element from ``data["anonymized"]``/``data["anonymized_facts"]``)
        -> the call returns more than ``(mapping, anonymized_transcript,
        raw)`` -> this test fails.
        """
        from paramem.graph.cloud_egress import anonymize_with_local_model

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
        raw = json.dumps(
            {
                "mapping": {"Alex": "Person_1", "Millfield": "City_1"},
                # A model that still emits fact-array keys — the parser
                # must ignore them entirely, not merely deprioritize them.
                "anonymized": [
                    {"subject": "Person_1", "predicate": "lives_in", "object": "City_1"}
                ],
                "anonymized_facts": [
                    {"subject": "Person_1", "predicate": "lives_in", "object": "City_1"}
                ],
                "anonymized_transcript": "Person_1 lives in City_1.",
            }
        )
        with (
            patch("paramem.graph.cloud_egress.generate_answer", return_value=raw),
            patch("paramem.graph.cloud_egress.adapt_messages", return_value=[]),
        ):
            result = anonymize_with_local_model(graph, model, tokenizer, scrub={"person name"})

        assert result == (
            {"Alex": "Person_1", "Millfield": "City_1"},
            "Person_1 lives in City_1.",
            raw,
        )

    def test_facts_are_built_from_graph_relations_not_the_model(self):
        """The SOTA-facing fact array must equal ``graph.relations`` —
        same count, byte-identical predicates — even when the model's raw
        response carries a SHORTER, REWORDED fact array alongside a valid
        mapping.

        Mutation: take the facts from the model's raw response instead of
        building them from ``graph.relations`` -> the dropped/reworded
        fact slips through -> fails.  The owner's rule, pinned.
        """
        from tests._sota_flow import run_sota_stages

        graph = _make_graph(
            [
                ("Alex", "lives_in", "Millfield"),
                ("Alex", "works_at", "Acme"),
            ],
            entities=[
                Entity(name="Alex", entity_type="person"),
                Entity(name="Millfield", entity_type="place"),
                Entity(name="Acme", entity_type="organization"),
            ],
        )
        # A model that has not fully adopted the mapping-only contract:
        # it still emits an "anonymized" fact array, SHORTER than the
        # real relation count (drops "works_at") and REWORDED
        # ("lives_in" -> "resides_at").
        raw = json.dumps(
            {
                "mapping": {"Alex": "Person_1", "Millfield": "City_1", "Acme": "Org_1"},
                "anonymized": [
                    {"subject": "Person_1", "predicate": "resides_at", "object": "City_1"},
                ],
                "anonymized_transcript": "Person_1 resides_at City_1 and works at Org_1.",
            }
        )

        sota_calls: list[list[dict]] = []

        def fake_sota(facts, *args, **kwargs):
            sota_calls.append(list(facts))
            return facts, None, {}, None, {}

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch("paramem.graph.cloud_egress.generate_answer", return_value=raw),
            patch("paramem.graph.cloud_egress.adapt_messages", return_value=[]),
            patch("paramem.graph.extractor._filter_with_sota", side_effect=fake_sota),
        ):
            run_sota_stages(
                graph,
                "Alex lives in Millfield and works at Acme.",
                MagicMock(),
                MagicMock(),
                speaker_id="speaker0",
                correction_entity_types=set(),
                scrub={"person name"},
            )

        assert len(sota_calls) == 1
        anon_facts = sota_calls[0]
        assert len(anon_facts) == len(graph.relations) == 2
        predicates = {f["predicate"] for f in anon_facts}
        assert predicates == {"lives_in", "works_at"}

    def test_predicate_is_never_a_substitution_target(self):
        """A relation whose ``predicate`` literally contains a real name
        that IS a mapping key keeps that predicate VERBATIM — the
        predicate is never a substitution target.

        Mutation: substitute the predicate through the mapping too (e.g.
        ``_substitute_whole_words(r.predicate, mapping)``) -> the
        predicate gets scrubbed to ``"asked about Person_1"`` -> fails.
        """
        from tests._sota_flow import run_sota_stages

        graph = _make_graph(
            [("Alex", "asked about Alex", "Bob")],
            entities=[
                Entity(name="Alex", entity_type="person"),
                Entity(name="Bob", entity_type="person"),
            ],
        )
        mapping = {"Alex": "Person_1", "Bob": "Person_2"}

        sota_calls: list[list[dict]] = []

        def fake_sota(facts, *args, **kwargs):
            sota_calls.append(list(facts))
            return facts, None, {}, None, {}

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.cloud_egress.anonymize_with_local_model",
                return_value=(mapping, "anonymized transcript", ""),
            ),
            patch("paramem.graph.extractor._filter_with_sota", side_effect=fake_sota),
        ):
            run_sota_stages(
                graph,
                "Alex asked about Alex and Bob.",
                None,
                None,
                speaker_id="speaker0",
                correction_entity_types=set(),
                scrub={"person name"},
            )

        assert len(sota_calls) == 1
        assert sota_calls[0][0]["predicate"] == "asked about Alex"

    def test_parse_failure_is_none_and_empty_mapping_is_not(self):
        """``mapping is None`` (parse failure) and ``mapping == {}``
        (the model found nothing to anonymize) are DISTINCT signals —
        collapsing them lets either an empty-but-valid mapping take the
        fail-closed branch, or a parse failure proceed unscrubbed.

        Mutation: collapse the two signals (e.g. ``mapping or None``,
        or gate on ``not mapping`` instead of ``mapping is None``) ->
        this test fails.
        """
        from paramem.graph.cloud_egress import anonymize_with_local_model

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
            patch("paramem.graph.cloud_egress.generate_answer", return_value="not json"),
            patch("paramem.graph.cloud_egress.adapt_messages", return_value=[]),
        ):
            parse_failure_mapping, _parse_failure_transcript, _ = anonymize_with_local_model(
                graph, model, tokenizer, scrub={"person name"}
            )

        with (
            patch(
                "paramem.graph.cloud_egress.generate_answer",
                return_value='{"mapping": {}, "anonymized_transcript": "nothing to scrub here"}',
            ),
            patch("paramem.graph.cloud_egress.adapt_messages", return_value=[]),
        ):
            empty_mapping, empty_mapping_transcript, _ = anonymize_with_local_model(
                graph, model, tokenizer, scrub={"person name"}
            )

        assert parse_failure_mapping is None
        assert empty_mapping == {}
        assert empty_mapping_transcript == "nothing to scrub here"
        assert empty_mapping is not None


class TestAnonymizerTranscriptArrayContract:
    """``anonymized_transcript`` is a JSON array of turn strings per the
    ``configs/prompts/anonymization.txt`` contract (one element per turn)
    so a multi-turn rewrite can never contain a literal newline inside a
    JSON string value — the illegal-JSON shape that caused a measured
    fail-closed parse failure.  ``anonymize_with_local_model`` joins the
    array with ``"\\n"``; a plain ``str`` is still accepted unchanged for
    models that have not adopted the array contract.

    Mutation: drop the ``list`` branch, or join with something other than
    ``"\\n"``, or stop rejecting malformed arrays -> these tests fail.
    """

    @staticmethod
    def _graph():
        return _make_graph(
            [("Alex", "lives_in", "Millfield")],
            entities=[
                Entity(name="Alex", entity_type="person"),
                Entity(name="Millfield", entity_type="place"),
            ],
        )

    def test_array_of_turn_strings_is_joined_with_newline(self):
        from paramem.graph.cloud_egress import anonymize_with_local_model

        model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.apply_chat_template = MagicMock(return_value="formatted")
        raw = json.dumps(
            {
                "mapping": {"Alex": "Person_1", "Millfield": "City_1"},
                "anonymized_transcript": [
                    "[user] My friend Person_1 lives in City_1.",
                    "[assistant] Got it.",
                    "[user] Anything else to add?",
                ],
            }
        )
        with (
            patch("paramem.graph.cloud_egress.generate_answer", return_value=raw),
            patch("paramem.graph.cloud_egress.adapt_messages", return_value=[]),
        ):
            mapping, anon_transcript, _raw = anonymize_with_local_model(
                self._graph(), model, tokenizer, scrub={"person name"}
            )

        assert mapping == {"Alex": "Person_1", "Millfield": "City_1"}
        assert anon_transcript == (
            "[user] My friend Person_1 lives in City_1.\n"
            "[assistant] Got it.\n"
            "[user] Anything else to add?"
        )

    def test_plain_string_transcript_still_accepted_unchanged(self):
        from paramem.graph.cloud_egress import anonymize_with_local_model

        model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.apply_chat_template = MagicMock(return_value="formatted")
        raw = json.dumps(
            {
                "mapping": {"Alex": "Person_1"},
                "anonymized_transcript": "[user] Person_1 lives in Millfield.",
            }
        )
        with (
            patch("paramem.graph.cloud_egress.generate_answer", return_value=raw),
            patch("paramem.graph.cloud_egress.adapt_messages", return_value=[]),
        ):
            mapping, anon_transcript, _raw = anonymize_with_local_model(
                self._graph(), model, tokenizer, scrub={"person name"}
            )

        assert mapping == {"Alex": "Person_1"}
        assert anon_transcript == "[user] Person_1 lives in Millfield."

    def test_empty_array_fails_closed(self):
        from paramem.graph.cloud_egress import anonymize_with_local_model

        model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.apply_chat_template = MagicMock(return_value="formatted")
        raw = json.dumps({"mapping": {"Alex": "Person_1"}, "anonymized_transcript": []})
        with (
            patch("paramem.graph.cloud_egress.generate_answer", return_value=raw),
            patch("paramem.graph.cloud_egress.adapt_messages", return_value=[]),
        ):
            mapping, anon_transcript, raw_output = anonymize_with_local_model(
                self._graph(), model, tokenizer, scrub={"person name"}
            )

        assert mapping is None
        assert anon_transcript == ""
        assert raw_output == raw

    def test_array_with_non_string_element_fails_closed(self):
        from paramem.graph.cloud_egress import anonymize_with_local_model

        model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.apply_chat_template = MagicMock(return_value="formatted")
        raw = json.dumps(
            {
                "mapping": {"Alex": "Person_1"},
                "anonymized_transcript": ["[user] Person_1 lives in Millfield.", 42],
            }
        )
        with (
            patch("paramem.graph.cloud_egress.generate_answer", return_value=raw),
            patch("paramem.graph.cloud_egress.adapt_messages", return_value=[]),
        ):
            mapping, anon_transcript, raw_output = anonymize_with_local_model(
                self._graph(), model, tokenizer, scrub={"person name"}
            )

        assert mapping is None
        assert anon_transcript == ""
        assert raw_output == raw

    def test_missing_anonymized_transcript_key_fails_closed(self):
        from paramem.graph.cloud_egress import anonymize_with_local_model

        model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.apply_chat_template = MagicMock(return_value="formatted")
        raw = json.dumps({"mapping": {"Alex": "Person_1"}})
        with (
            patch("paramem.graph.cloud_egress.generate_answer", return_value=raw),
            patch("paramem.graph.cloud_egress.adapt_messages", return_value=[]),
        ):
            mapping, anon_transcript, raw_output = anonymize_with_local_model(
                self._graph(), model, tokenizer, scrub={"person name"}
            )

        assert mapping is None
        assert anon_transcript == ""
        assert raw_output == raw


class TestScrubCategoriesReachPrompt:
    """The config -> prompt flow for ``scrub``.  ``scrub_categories``
    is rendered as ``", ".join(sorted(scrub))`` into the anonymization
    prompt's ``{scrub_categories}`` slot (``anonymize_with_local_model``).
    No prior test drove a real, distinctive ``scrub`` set all the way
    through to the rendered prompt string handed to the model — a
    hardcoded/ignored ``scrub_categories`` slot would not be caught by any
    existing test.

    Mutation: hardcode ``scrub_categories`` to a fixed string (or drop the
    ``sorted()`` call) -> this test fails.
    """

    def test_distinctive_scrub_set_appears_sorted_in_rendered_prompt(self):
        from paramem.graph.cloud_egress import anonymize_with_local_model

        graph = _make_graph(
            [("Alex", "lives_in", "Millfield")],
            entities=[
                Entity(name="Alex", entity_type="person"),
                Entity(name="Millfield", entity_type="place"),
            ],
        )
        model = MagicMock()
        tokenizer = MagicMock()
        # Identity passthrough so the actual rendered prompt text (built
        # from the real `scrub` set) survives into the string handed to
        # `apply_chat_template`, instead of being discarded by a mocked
        # no-op the way most other tests in this file do (they only care
        # about the mapping/transcript result, not the prompt text).
        tokenizer.apply_chat_template = MagicMock(
            side_effect=lambda messages, **kwargs: messages[-1]["content"]
        )
        captured: dict[str, str] = {}

        def _fake_generate_answer(_model, _tokenizer, prompt, **_kwargs):
            captured["prompt"] = prompt
            return json.dumps({"mapping": {}, "anonymized_transcript": "nothing to scrub here"})

        with (
            patch("paramem.graph.cloud_egress.generate_answer", side_effect=_fake_generate_answer),
            patch(
                "paramem.graph.cloud_egress.adapt_messages",
                side_effect=lambda messages, tok: messages,
            ),
        ):
            anonymize_with_local_model(
                graph,
                model,
                tokenizer,
                scrub={"custom_category_x", "another_y"},
            )

        assert "prompt" in captured, "generate_answer was never called with a prompt"
        # sorted(["custom_category_x", "another_y"]) == ["another_y", "custom_category_x"]
        assert "Categories to scrub: another_y, custom_category_x" in captured["prompt"]


class TestNoPostHocLeakGuardCaseSensitivity:
    """There is no forward-path post-hoc check on the anonymized payload.
    Substitution is case-SENSITIVE and that is load-bearing: case is the
    only signal separating a person named ``Bill`` from the common noun
    ``bill`` (an invoice), or ``Will``/``will``, ``Mark``/``mark``,
    ``Rose``/``rose``. Any case-insensitive match over entity names cannot
    tell a real leak from ordinary prose. Never introduce one.

    Named for the bug this rule guards against.
    """

    def test_bill_the_person_scrubbed_electricity_bill_untouched(self):
        """A person entity named ``Bill`` is scrubbed to its placeholder;
        the common noun ``bill`` in "electricity bill" is left untouched
        — and SOTA IS called (nothing skips or blocks the cycle over the
        case collision).

        Mutation: introduce any case-insensitive match over entity names
        (e.g. a case-insensitive substring/whole-word check on the
        anonymized payload, or make the substitution primitive
        case-insensitive) -> "electricity bill" is misread as a leak of
        ``Bill`` -> the cycle blocks/skips SOTA, or the object is wrongly
        scrubbed -> this test fails.
        """
        from tests._sota_flow import run_sota_stages

        graph = _make_graph(
            [("Bill", "received", "electricity bill")],
            entities=[Entity(name="Bill", entity_type="person")],
        )
        mapping = {"Bill": "Person_1"}
        transcript = "Bill received the electricity bill yesterday."

        sota_calls: list[list[dict]] = []

        def fake_sota(facts, *args, **kwargs):
            sota_calls.append(list(facts))
            return facts, None, {}, None, {}

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.cloud_egress.anonymize_with_local_model",
                return_value=(mapping, "anonymized transcript", ""),
            ),
            patch("paramem.graph.extractor._filter_with_sota", side_effect=fake_sota),
        ):
            result = run_sota_stages(
                graph,
                transcript,
                None,
                None,
                speaker_id="speaker0",
                correction_entity_types=set(),
                scrub={"person name"},
            )

        # SOTA IS called — nothing blocks or skips the cycle.
        assert len(sota_calls) == 1
        sent = sota_calls[0][0]
        # "Bill" (the person) IS scrubbed to its placeholder in the SOTA payload.
        assert sent["subject"] == "Person_1"
        # "bill" (the invoice, inside "electricity bill") is NOT scrubbed —
        # case is the only signal telling the two apart.
        assert sent["object"] == "electricity bill"
        # Final graph de-anonymizes correctly.
        assert len(result.relations) == 1
        assert result.relations[0].subject == "Bill"
        assert result.relations[0].object == "electricity bill"


class TestDeanonStagePredicateInvariantEndToEnd:
    """The anonymizer never returns facts, so a placeholder gluing into a
    predicate at the anonymize stage is structurally impossible. The
    ONLY stage a placeholder can still glue into a predicate is SOTA's
    own returned delta. This pins the end-to-end behaviour of the
    deanon-stage predicate invariant (:func:`_apply_bindings`): a fact
    whose predicate carries a glued placeholder is dropped before it
    reaches ``graph.relations``.
    """

    def test_sota_returned_glued_predicate_dropped_end_to_end(self):
        """A fact in SOTA's *returned* delta whose predicate glues a
        declared placeholder onto a static prefix
        (``language_proficiency_Language_3``) never reaches
        ``graph.relations``, and the drop is recorded in diagnostics.

        Mutation: delete/narrow the deanon-stage predicate invariant in
        ``_apply_bindings`` -> the poisoned fact reaches
        ``graph.relations`` -> this test fails.
        """
        from tests._sota_flow import run_sota_stages

        graph = _make_graph(
            [("Alex", "works_at", "Acme")],
            entities=[Entity(name="Alex", entity_type="person")],
        )
        # "French" -> "Language_3" is an LLM-hint mapping entry the
        # deterministic builder merges in verbatim (never minted by
        # graph.entities, which only covers Alex here) — this is what
        # puts "Language_3" into the declared vocabulary SOTA's returned
        # facts are checked against.
        mapping = {"Alex": "Person_1", "French": "Language_3"}
        transcript = "Alex works at Acme. Alex speaks French at an advanced level."

        # SOTA's returned delta — NOT the anonymizer — carries the
        # poisoned predicate.  `_check_mapping_totality` (the SOTA-stage
        # binding-totality gate) scans subject/object only, so this fact
        # sails through it untouched and reaches the deanon-stage
        # predicate invariant, which is what this test pins.
        enriched_anon = [
            {
                "subject": "Person_1",
                "predicate": "language_proficiency_Language_3",
                "object": "Advanced",
                "relation_type": "factual",
                "confidence": 0.9,
            },
        ]

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.cloud_egress.anonymize_with_local_model",
                return_value=(mapping, "anonymized transcript", ""),
            ),
            patch(
                "paramem.graph.extractor._filter_with_sota",
                return_value=(enriched_anon, None, {}, None, {}),
            ),
        ):
            result = run_sota_stages(
                graph,
                transcript,
                None,
                None,
                speaker_id="speaker0",
                correction_entity_types=set(),
                scrub={"person name", "language"},
            )

        # The poisoned fact never reaches the merged graph.
        assert all(r.predicate != "language_proficiency_Language_3" for r in result.relations)
        assert graph.diagnostics["predicate_placeholder_dropped"] == 1
        # It was the ONLY fact, so the substitution emptied the working
        # set and the all-dropped net fired. That is a mechanical
        # BREAKAGE, not a judge's verdict — and the gate now records
        # which of the two it was. Recorded only: the outcome (fallback
        # to the pre-enrichment facts) is unchanged by the cause.
        assert result.diagnostics["all_dropped_cause"] == {
            "cause": "deanon_substitution_dropped_all",
            "kind": "breakage",
        }
        assert result.diagnostics.get("fallback_path") == "all_dropped"


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
        from paramem.graph.placeholders import _apply_bindings

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
        kept, predicate_dropped, residual_dropped = _apply_bindings(
            facts, reverse, sota_bindings={}
        )
        dropped = predicate_dropped + residual_dropped
        assert dropped == []
        assert kept[0]["subject"] == "Alice"
        assert kept[0]["object"] == "Acme"

    def test_substitutes_braced_sota_bindings(self):
        """SOTA-introduced braced placeholders ({Event_1}) substitute via
        explicit bindings without needing transcript reconstruction."""
        from paramem.graph.placeholders import _apply_bindings

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
        kept, predicate_dropped, residual_dropped = _apply_bindings(facts, reverse, bindings)
        dropped = predicate_dropped + residual_dropped
        assert dropped == []
        assert kept[0]["subject"] == "Alice"
        assert kept[0]["object"] == "the agile transformation workshop"

    def test_substitutes_compound_objects(self):
        """Bare placeholder embedded in literal text — `Org_1 Hungary`
        becomes `Acme Hungary` (the failure mode that bug 5 produced
        bogus bindings for under the old regex pipeline)."""
        from paramem.graph.placeholders import _apply_bindings

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
        kept, predicate_dropped, residual_dropped = _apply_bindings(
            facts, reverse, sota_bindings={}
        )
        dropped = predicate_dropped + residual_dropped
        assert dropped == []
        assert kept[0]["object"] == "Acme Hungary"

    def test_drops_facts_with_unresolved_placeholders(self):
        """Facts whose subject/object retain a placeholder pattern after
        substitution get dropped (residual sweep). Causes: SOTA emitted a
        braced placeholder without including it in bindings, anonymizer
        leak, etc."""
        from paramem.graph.placeholders import _apply_bindings

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
        kept, predicate_dropped, residual_dropped = _apply_bindings(
            facts, reverse, sota_bindings={}
        )
        dropped = predicate_dropped + residual_dropped
        assert kept == []
        assert len(dropped) == 2

    def test_handles_apostrophes_at_word_boundary(self):
        """`Person_2's cousin` substitutes Person_2 cleanly without breaking
        on the apostrophe (existing _substitute_whole_words behaviour)."""
        from paramem.graph.placeholders import _apply_bindings

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
        kept, predicate_dropped, residual_dropped = _apply_bindings(
            facts, reverse, sota_bindings={}
        )
        dropped = predicate_dropped + residual_dropped
        assert dropped == []
        assert kept[0]["object"] == "Bob's cousin"

    def test_mixed_bare_and_braced_in_same_fact(self):
        """A single fact with both a bare anonymizer placeholder and a
        braced SOTA placeholder substitutes both."""
        from paramem.graph.placeholders import _apply_bindings

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
        kept, predicate_dropped, residual_dropped = _apply_bindings(facts, reverse, bindings)
        dropped = predicate_dropped + residual_dropped
        assert dropped == []
        assert kept[0]["subject"] == "Alice"
        assert kept[0]["object"] == "the workshop at Acme"

    def test_empty_inputs_return_empty(self):
        from paramem.graph.placeholders import _apply_bindings

        kept, predicate_dropped, residual_dropped = _apply_bindings([], {}, {})

        dropped = predicate_dropped + residual_dropped
        assert kept == []
        assert dropped == []

    def test_preserves_other_fact_fields(self):
        """relation_type, confidence, and any extra fields pass through."""
        from paramem.graph.placeholders import _apply_bindings

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
        kept, _, _ = _apply_bindings(facts, reverse, sota_bindings={})
        assert kept[0]["relation_type"] == "social"
        assert kept[0]["confidence"] == 0.7
        assert kept[0]["synthetic"] is False

    def test_minted_placeholder_round_trips_bare(self):
        """A SOTA-minted placeholder emitted BARE (not braced, contra the
        prompt's contract) still round-trips via the union resolve map —
        today's two-channel design drops this because ``bare_map`` only
        ever contained ``reverse``."""
        from paramem.graph.placeholders import _apply_bindings

        facts = [
            {
                "subject": "Person_1",
                "predicate": "attended",
                "object": "Event_1",
                "relation_type": "factual",
                "confidence": 1.0,
            },
        ]
        reverse = {"Person_1": "Alice"}
        sota_bindings = {"Event_1": "the quarterly retro"}
        kept, predicate_dropped, residual_dropped = _apply_bindings(facts, reverse, sota_bindings)
        dropped = predicate_dropped + residual_dropped
        assert dropped == []
        assert kept[0]["object"] == "the quarterly retro"

    def test_minted_placeholder_round_trips_braced_regression(self):
        """Braced-form minted placeholder still resolves (regression
        guard for the union unification)."""
        from paramem.graph.placeholders import _apply_bindings

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
        sota_bindings = {"Event_1": "the quarterly retro"}
        kept, predicate_dropped, residual_dropped = _apply_bindings(facts, reverse, sota_bindings)
        dropped = predicate_dropped + residual_dropped
        assert dropped == []
        assert kept[0]["object"] == "the quarterly retro"

    def test_anonymizer_placeholder_wrongly_braced_still_resolves(self):
        """An anonymizer placeholder the cloud wrongly re-braced
        (contra the prompt's 'leave bare' contract) still resolves via
        the union — it is in ``reverse``, which is now tried in both
        the braced and bare pass."""
        from paramem.graph.placeholders import _apply_bindings

        facts = [
            {
                "subject": "Person_1",
                "predicate": "attended",
                "object": "{Person_1}",
                "relation_type": "factual",
                "confidence": 1.0,
            },
        ]
        reverse = {"Person_1": "Alex"}
        kept, predicate_dropped, residual_dropped = _apply_bindings(
            facts, reverse, sota_bindings={}
        )
        dropped = predicate_dropped + residual_dropped
        assert dropped == []
        assert kept[0]["object"] == "Alex"

    def test_nested_binding_value_resolves_in_order(self):
        """A binding value containing a bare anonymizer placeholder
        (``"Senior Engineer at Org_1"``) resolves fully: braced pass
        expands ``{Role_1}`` to the value, bare pass then resolves the
        exposed ``Org_1`` from the SAME union map."""
        from paramem.graph.placeholders import _apply_bindings

        facts = [
            {
                "subject": "Person_1",
                "predicate": "held_role",
                "object": "{Role_1}",
                "relation_type": "factual",
                "confidence": 1.0,
            },
        ]
        sota_bindings = {"Role_1": "Senior Engineer at Org_1"}
        reverse = {"Person_1": "Alex", "Org_1": "Acme"}
        kept, predicate_dropped, residual_dropped = _apply_bindings(facts, reverse, sota_bindings)
        dropped = predicate_dropped + residual_dropped
        assert dropped == []
        assert kept[0]["object"] == "Senior Engineer at Acme"

    def test_collision_reverse_wins_in_resolved_output(self):
        """When a key collides between ``sota_bindings`` and ``reverse``
        with differing values, ``reverse`` wins (deterministic entity
        name over a freshly-minted SOTA value) — the collision itself is
        surfaced by :func:`_check_mapping_totality`, not here."""
        from paramem.graph.placeholders import _apply_bindings

        facts = [
            {
                "subject": "Org_1",
                "predicate": "based_in",
                "object": "Germany",
                "relation_type": "factual",
                "confidence": 1.0,
            },
        ]
        reverse = {"Org_1": "Acme"}
        sota_bindings = {"Org_1": "Wrong Corp"}
        kept, predicate_dropped, residual_dropped = _apply_bindings(facts, reverse, sota_bindings)
        dropped = predicate_dropped + residual_dropped
        assert dropped == []
        assert kept[0]["subject"] == "Acme"

    def test_predicate_placeholder_is_not_resolved_into_the_predicate(self):
        """A placeholder glued into the predicate (``at_Org_1``) is
        dropped by the predicate invariant BEFORE substitution runs — it
        must never be "resolved" into a garbage predicate
        (``at_Acme``). The predicate field is never a substitution
        target, so the dropped fact's predicate is byte-identical to
        the input; if the invariant ran AFTER substitution instead (or
        were removed), a hypothetical predicate-substitution mistake
        could turn this into ``at_Acme``, which contains no placeholder
        token and would silently survive.

        Pinned via the PARTITIONED return, not a recombined list: this
        specific fact must land in ``predicate_dropped`` (step 1, the
        pre-substitution copy — ``subject`` still ``Person_1``), NEVER
        in ``residual_dropped`` (step 3, which would only see the
        post-substitution copy with ``subject == "Alice"``). Deleting
        the step-1 check makes the fact fall through to the residual
        sweep instead — ``predicate_dropped`` goes empty and
        ``residual_dropped`` gains a post-substitution copy — so both
        assertions below fail together, not just one accidentally
        redundant with the other. (Verified live: see task report.)"""
        from paramem.graph.placeholders import _apply_bindings

        facts = [
            {
                "subject": "Person_1",
                "predicate": "at_Org_1",
                "object": "engineer",
                "relation_type": "factual",
                "confidence": 1.0,
            },
        ]
        reverse = {"Person_1": "Alice", "Org_1": "Acme"}
        kept, predicate_dropped, residual_dropped = _apply_bindings(
            facts, reverse, sota_bindings={}
        )
        assert kept == []
        assert residual_dropped == []
        assert len(predicate_dropped) == 1
        assert predicate_dropped[0]["subject"] == "Person_1"
        assert predicate_dropped[0]["predicate"] == "at_Org_1"
        dropped = predicate_dropped + residual_dropped
        assert not any(f.get("predicate") == "at_Acme" for f in kept + dropped)

    def test_unresolved_token_in_any_field_dropped(self):
        """A declared placeholder token glued into the OBJECT field
        (``language_proficiency_Language_3`` — invisible to the
        ``\\b``-anchored :data:`_PLACEHOLDER_TOKEN_RE`) still drops the
        triple: the residual sweep tests every field against the
        declared vocabulary, not just subject."""
        from paramem.graph.placeholders import _apply_bindings

        facts = [
            {
                "subject": "Alice",
                "predicate": "speaks",
                "object": "language_proficiency_Language_3",
                "relation_type": "factual",
                "confidence": 1.0,
            },
        ]
        reverse = {"Language_3": "French"}
        kept, predicate_dropped, residual_dropped = _apply_bindings(
            facts, reverse, sota_bindings={}
        )
        dropped = predicate_dropped + residual_dropped
        assert kept == []
        assert len(dropped) == 1

    def test_non_fact_field_with_declared_token_does_not_shed_the_fact(self):
        """A field outside :data:`_FACT_FIELDS` (e.g. an ``evidence``
        key an LLM appended alongside the fact proper) that still
        contains a declared placeholder token must NOT sink the whole
        otherwise-valid fact: the residual sweep only tests the fields
        that actually reach ``Relation`` (subject/predicate/object/
        relation_type/confidence/symmetric). Before the fix, the sweep
        iterated ``f.values()`` unconditionally and would have dropped
        this fact over ``evidence`` alone."""
        from paramem.graph.placeholders import _apply_bindings

        facts = [
            {
                "subject": "Person_1",
                "predicate": "works_at",
                "object": "Org_1",
                "relation_type": "factual",
                "confidence": 0.9,
                "evidence": "Person_1 said they work at Org_1",
            },
        ]
        reverse = {"Person_1": "Alice", "Org_1": "Acme"}
        kept, predicate_dropped, residual_dropped = _apply_bindings(
            facts, reverse, sota_bindings={}
        )
        assert predicate_dropped == []
        assert residual_dropped == []
        assert len(kept) == 1
        assert kept[0]["subject"] == "Alice"
        assert kept[0]["object"] == "Acme"
        # The non-fact field is not a substitution target either — it
        # still carries the unresolved token, proving the sweep
        # genuinely ignored it rather than getting lucky on substitution.
        assert "Person_1" in kept[0]["evidence"]


class TestResidualSweepCatchesEmbeddedPlaceholders:
    def test_residual_sweep_catches_bare_and_composite(self):
        """The residual sweep (step 3 of :func:`_apply_bindings`, the
        SINGLE deanon exit gate — the standalone
        ``_strip_residual_placeholders`` this test used to call directly
        is retired) drops facts with any placeholder-shaped token, bare
        or composite, via the fail-closed :data:`_PLACEHOLDER_TOKEN_RE`
        backstop — even with an empty declared vocabulary (nothing in
        ``reverse``/``sota_bindings`` here)."""
        from paramem.graph.placeholders import _apply_bindings

        facts = [
            {"subject": "Alice", "predicate": "knows", "object": "Bob"},  # clean
            {"subject": "Alice", "predicate": "supports", "object": "Person_2"},  # bare
            {"subject": "Alice", "predicate": "values", "object": "Person_2's Support"},  # embedded
            {"subject": "{Topic_1}", "predicate": "related_to", "object": "Bob"},  # braced
        ]
        kept, predicate_dropped, residual_dropped = _apply_bindings(facts, {}, {})
        assert predicate_dropped == []
        assert len(residual_dropped) == 3
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


class TestFilterWithSotaPromptsDir:
    """``_filter_with_sota`` had neither a ``prompts_dir`` parameter nor a
    forwarded value — the ``sota_enrichment.txt`` load never honoured a
    calibration override at all."""

    def test_prompts_dir_override_reaches_enrichment_prompt(self, tmp_path):
        from paramem.graph.extractor import _filter_with_sota

        sentinel = "SENTINEL-SOTA-ENRICH"
        (tmp_path / "sota_enrichment.txt").write_text(
            f"{sentinel}\nfacts: {{facts_json}}\ntranscript: {{transcript}}"
        )
        captured_prompts = []

        def fake_sota_call(prompt, *args, **kwargs):
            captured_prompts.append(prompt)
            return '{"add": [], "modify": [], "drop": [], "bindings": {}}'

        with patch("paramem.graph.extractor._sota_call", side_effect=fake_sota_call):
            _filter_with_sota(
                [{"subject": "A", "predicate": "knows", "object": "B"}],
                api_key="k",
                provider="anthropic",
                anon_transcript="A knows B.",
                prompts_dir=tmp_path,
            )

        assert captured_prompts, "_sota_call was never invoked"
        assert sentinel in captured_prompts[0], (
            f"Enrichment call used the shipped prompt instead of the override: "
            f"{captured_prompts[0]!r}"
        )

    def test_default_prompts_dir_uses_shipped_template(self):
        """Parity check: omitting ``prompts_dir`` must keep loading the
        production template — the new parameter is additive only."""
        from paramem.graph.extractor import _filter_with_sota

        captured_prompts = []

        def fake_sota_call(prompt, *args, **kwargs):
            captured_prompts.append(prompt)
            return '{"add": [], "modify": [], "drop": [], "bindings": {}}'

        with patch("paramem.graph.extractor._sota_call", side_effect=fake_sota_call):
            _filter_with_sota(
                [{"subject": "A", "predicate": "knows", "object": "B"}],
                api_key="k",
                provider="anthropic",
                anon_transcript="A knows B.",
            )

        assert captured_prompts
        assert "SENTINEL" not in captured_prompts[0]


class TestPlausibilityFilterWithSotaPromptsDir:
    """``_plausibility_filter_with_sota`` had the same gap: no ``prompts_dir``
    parameter, no forwarding, ``sota_plausibility.txt`` loaded unconditionally."""

    def test_prompts_dir_override_reaches_plausibility_prompt(self, tmp_path):
        from paramem.graph.extractor import _plausibility_filter_with_sota

        sentinel = "SENTINEL-SOTA-PLAUSIBILITY"
        (tmp_path / "sota_plausibility.txt").write_text(
            f"{sentinel}\nfacts: {{facts_json}}\ntranscript: {{transcript}}"
        )
        captured_prompts = []

        def fake_sota_call(prompt, *args, **kwargs):
            captured_prompts.append(prompt)
            return '{"drop": []}'

        with patch("paramem.graph.extractor._sota_call", side_effect=fake_sota_call):
            _plausibility_filter_with_sota(
                [{"subject": "A", "predicate": "knows", "object": "B"}],
                api_key="k",
                provider="anthropic",
                anon_transcript="A knows B.",
                prompts_dir=tmp_path,
            )

        assert captured_prompts, "_sota_call was never invoked"
        assert sentinel in captured_prompts[0], (
            f"Plausibility call used the shipped prompt instead of the override: "
            f"{captured_prompts[0]!r}"
        )

    def test_default_prompts_dir_uses_shipped_template(self):
        """Parity check: omitting ``prompts_dir`` must keep loading the
        production template — the new parameter is additive only."""
        from paramem.graph.extractor import _plausibility_filter_with_sota

        captured_prompts = []

        def fake_sota_call(prompt, *args, **kwargs):
            captured_prompts.append(prompt)
            return '{"drop": []}'

        with patch("paramem.graph.extractor._sota_call", side_effect=fake_sota_call):
            _plausibility_filter_with_sota(
                [{"subject": "A", "predicate": "knows", "object": "B"}],
                api_key="k",
                provider="anthropic",
                anon_transcript="A knows B.",
            )

        assert captured_prompts
        assert "SENTINEL" not in captured_prompts[0]


class TestSotaSystemPromptCallTimeOverride:
    """``sota_enrichment_system.txt`` / ``sota_plausibility_system.txt``
    used to bind ONCE at module-import time (``extractor.py`` module-level
    constants ``_SOTA_ENRICHMENT_SYSTEM_PROMPT`` / ``_SOTA_PLAUSIBILITY_SYSTEM_PROMPT``)
    — long before any :func:`~paramem.graph.phase_trace.extraction_trace`
    scope or :func:`~paramem.graph.prompts.prompt_overrides` context could
    exist, so a calibration override could never reach them and
    ``record_prompt`` always no-opped for them.  They now load at CALL
    TIME inside each consuming function.  These tests pin BOTH halves of
    that fix: an import-time binding would make the override never reach
    ``_sota_call``/``generate_answer`` (first two assertions per test) AND
    would leave ``record.prompts`` without the override entry (the
    provenance assertion) — a plain "does ``_load_prompt`` honour an
    override" unit test cannot tell these apart from the old broken state.
    """

    def test_sota_enrichment_system_prompt_overridable_and_recorded(self):
        from paramem.graph.extractor import _filter_with_sota
        from paramem.graph.phase_trace import extraction_trace, phase_trace
        from paramem.graph.prompts import prompt_overrides

        captured = []

        def fake_sota_call(prompt, *args, **kwargs):
            captured.append(kwargs.get("system_prompt"))
            return '{"add": [], "modify": [], "drop": [], "bindings": {}}'

        with patch("paramem.graph.extractor._sota_call", side_effect=fake_sota_call):
            with extraction_trace() as trace:
                with phase_trace("sota_enrich"):
                    with prompt_overrides({"sota_enrichment_system.txt": "SENTINEL-ENRICH-SYSTEM"}):
                        _filter_with_sota(
                            [{"subject": "A", "predicate": "knows", "object": "B"}],
                            api_key="k",
                            provider="anthropic",
                            anon_transcript="A knows B.",
                        )
                record = trace.records[-1]

        assert captured == ["SENTINEL-ENRICH-SYSTEM"], (
            "the override must reach _sota_call's system_prompt kwarg"
        )
        paths = [p["path"] for p in (record.prompts or [])]
        assert "<override:sota_enrichment_system.txt>" in paths, (
            f"override must be recorded in phase-trace provenance, got paths={paths!r}"
        )

    def test_sota_plausibility_system_prompt_overridable_and_recorded(self):
        from paramem.graph.extractor import _plausibility_filter_with_sota
        from paramem.graph.phase_trace import extraction_trace, phase_trace
        from paramem.graph.prompts import prompt_overrides

        captured = []

        def fake_sota_call(prompt, *args, **kwargs):
            captured.append(kwargs.get("system_prompt"))
            return '{"drop": []}'

        with patch("paramem.graph.extractor._sota_call", side_effect=fake_sota_call):
            with extraction_trace() as trace:
                with phase_trace("anon_plausibility"):
                    with prompt_overrides(
                        {"sota_plausibility_system.txt": "SENTINEL-PLAUS-SYSTEM"}
                    ):
                        _plausibility_filter_with_sota(
                            [{"subject": "A", "predicate": "knows", "object": "B"}],
                            api_key="k",
                            provider="anthropic",
                            anon_transcript="A knows B.",
                        )
                record = trace.records[-1]

        assert captured == ["SENTINEL-PLAUS-SYSTEM"], (
            "the override must reach _sota_call's system_prompt kwarg"
        )
        paths = [p["path"] for p in (record.prompts or [])]
        assert "<override:sota_plausibility_system.txt>" in paths, (
            f"override must be recorded in phase-trace provenance, got paths={paths!r}"
        )

    def test_local_plausibility_filter_system_prompt_overridable_and_recorded(self):
        """``local_plausibility_filter`` reuses ``sota_plausibility_system.txt``
        as the LOCAL model's system message — it builds the chat
        ``messages`` list directly rather than calling ``_sota_call``."""
        from paramem.graph.extractor import local_plausibility_filter
        from paramem.graph.phase_trace import extraction_trace, phase_trace
        from paramem.graph.prompts import prompt_overrides

        facts = [{"subject": "Alex", "predicate": "lives_in", "object": "Millfield"}]
        tokenizer = MagicMock()
        tokenizer.apply_chat_template = MagicMock(return_value="formatted")
        with (
            patch("paramem.graph.extractor.generate_answer", return_value='{"drop": []}'),
            # Identity passthrough so the real messages list (carrying the
            # override) reaches apply_chat_template unchanged — see the
            # companion note on TestFilterWithSotaPromptsDir-style tests.
            patch(
                "paramem.graph.extractor.adapt_messages",
                side_effect=lambda messages, tok: messages,
            ),
        ):
            with extraction_trace() as trace:
                with phase_trace("deanon_plausibility"):
                    with prompt_overrides(
                        {"sota_plausibility_system.txt": "SENTINEL-LOCAL-PLAUS-SYSTEM"}
                    ):
                        local_plausibility_filter(facts, "transcript", MagicMock(), tokenizer)
                record = trace.records[-1]

        called_messages = tokenizer.apply_chat_template.call_args.args[0]
        system_contents = [m["content"] for m in called_messages if m["role"] == "system"]
        assert system_contents == ["SENTINEL-LOCAL-PLAUS-SYSTEM"], (
            "the override must reach the local model's system message"
        )
        paths = [p["path"] for p in (record.prompts or [])]
        assert "<override:sota_plausibility_system.txt>" in paths, (
            f"override must be recorded in phase-trace provenance, got paths={paths!r}"
        )


class TestSpeakerContextInjection:
    def test_build_speaker_context_empty_when_no_id(self):
        """build_speaker_context returns empty string when speaker_id is absent."""
        from paramem.graph.extractor import build_speaker_context

        assert build_speaker_context(None, None) == ""
        assert build_speaker_context("", None) == ""
        assert build_speaker_context(None, "Alice") == ""
        assert build_speaker_context("", "Alice") == ""

    def test_build_speaker_context_includes_id_and_name(self):
        """Directive pins speaker_id as subject and injects display name as context."""
        from paramem.graph.extractor import build_speaker_context

        out = build_speaker_context("speaker0", "Ye Jie")
        # Speaker id must appear as the required subject.
        assert "speaker0" in out
        # Display name must appear as comprehension context.
        assert "Ye Jie" in out
        # Must forbid generic fallback strings so the model cannot emit them.
        for forbidden in (
            "{SPEAKER_NAME}",
            "SPEAKER_NAME",
            "Speaker_Name",
            "User",
            "'I'",
        ):
            assert forbidden in out, f"directive must mention {forbidden!r}"

    def test_build_speaker_context_id_only_no_display_name(self):
        """Anonymous speaker: id used for both subject and context; no KeyError."""
        from paramem.graph.extractor import build_speaker_context

        out = build_speaker_context("speaker0", None)
        assert "speaker0" in out
        assert "{speaker_id}" not in out
        assert "{speaker_name}" not in out

    def test_extraction_directive_overridable_and_recorded_in_provenance(self):
        """``speaker_directive.txt`` used to be read via a bare
        ``Path.read_text()`` inside ``_load_speaker_directive_section`` —
        unreachable by a calibration override and never recorded via
        ``record_prompt``.  It now routes through ``_load_prompt``, so
        both become possible.  ``build_speaker_context`` feeds this
        section into the ``{speaker_context}`` slot of the extraction
        user template — it is part of the prompt under test."""
        from paramem.graph.extractor import build_speaker_context
        from paramem.graph.phase_trace import extraction_trace, phase_trace
        from paramem.graph.prompts import prompt_overrides

        override_content = (
            "=== EXTRACTION-DIRECTIVE ===\nSENTINEL-DIRECTIVE {speaker_id} {speaker_name}\n"
        )
        with extraction_trace() as trace:
            with phase_trace("local_extract"):
                with prompt_overrides({"speaker_directive.txt": override_content}):
                    out = build_speaker_context("speaker0", "Alice")
            record = trace.records[-1]

        assert "SENTINEL-DIRECTIVE speaker0 Alice" in out
        paths = [p["path"] for p in (record.prompts or [])]
        assert "<override:speaker_directive.txt>" in paths, (
            f"override must be recorded in phase-trace provenance, got paths={paths!r}"
        )


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
        from tests._sota_flow import run_sota_stages

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

        # Plausibility filter keeps only the lives_in fact
        kept_anon = [{"subject": "Person_1", "predicate": "lives_in", "object": "City_1"}]

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.cloud_egress.anonymize_with_local_model",
                return_value=(mapping, "anonymized transcript", ""),
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
            result = run_sota_stages(
                graph,
                "Alex lives in Millfield.",
                None,
                None,
                speaker_id="speaker0",
                plausibility_judge="claude",
                plausibility_stage="anon",
                correction_entity_types=set(),
                scrub={"person name", "physical address"},
            )

        # Only the valid fact survives
        assert len(result.relations) == 1
        assert result.relations[0].predicate == "lives_in"
        assert result.diagnostics.get("plausibility") == "anon"

    def test_anon_stage_plausibility_receives_post_enrichment_transcript(self):
        """FIX 2: the anon-stage judge must see `updated_anon_transcript`
        (post-enrichment — carrying SOTA's minted `{Paper_1}`-style
        tokens), not the pre-enrichment `anon_transcript`.  A judge shown
        the stale pre-enrichment transcript can never connect an
        enrichment-only fact's placeholder to its real-text span in the
        transcript, and drops a valid enrichment.

        Mutation: revert the call site to pass `anon_transcript=anon_transcript`
        -> this test fails.
        """
        from tests._sota_flow import run_sota_stages

        graph = _make_graph(
            [("Alex", "authored", "Attention Is All You Need")],
            entities=[
                Entity(name="Alex", entity_type="person"),
            ],
        )
        # Pre-enrichment: local extraction never saw the paper title, so
        # neither the input facts nor the input transcript mention
        # "Paper_1" — it is introduced by SOTA's enrichment delta below.
        enriched_anon_facts = [
            {"subject": "Person_1", "predicate": "authored", "object": "Paper_1"}
        ]
        mapping = {"Alex": "Person_1"}
        post_enrichment_transcript = "Person_1 wrote {Paper_1} (Attention Is All You Need)."

        plaus_calls = []

        def fake_plaus(facts, api_key, **kwargs):
            plaus_calls.append(kwargs.get("anon_transcript"))
            return facts, "raw"

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.cloud_egress.anonymize_with_local_model",
                return_value=(mapping, "anonymized transcript", ""),
            ),
            patch(
                "paramem.graph.extractor._filter_with_sota",
                return_value=(
                    enriched_anon_facts,
                    post_enrichment_transcript,
                    {"Paper_1": "Attention Is All You Need"},
                    None,
                    {},
                ),
            ),
            patch(
                "paramem.graph.extractor._plausibility_filter_with_sota",
                side_effect=fake_plaus,
            ),
        ):
            run_sota_stages(
                graph,
                "Alex wrote a well-known paper.",
                None,
                None,
                speaker_id="speaker0",
                plausibility_judge="claude",
                plausibility_stage="anon",
                correction_entity_types=set(),
                scrub={"person name"},
            )

        assert len(plaus_calls) == 1
        assert plaus_calls[0] == post_enrichment_transcript, (
            f"anon-stage judge must see the post-enrichment transcript, got {plaus_calls[0]!r}"
        )


class TestPlausibilityDeanon:
    """_sota_pipeline with plausibility_stage="deanon": plausibility runs on
    de-anonymized facts using the original transcript.
    """

    def test_deanon_stage_plausibility_drops_tautology(self):
        """Deanon-stage local plausibility receives real names and drops tautologies."""
        from tests._sota_flow import run_sota_stages

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

        # Local plausibility drops the tautology, keeps lives_in
        kept_deanon = [{"subject": "Alex", "predicate": "lives_in", "object": "Millfield"}]

        local_plaus_calls = []

        def fake_local_plaus(facts, transcript, model, tokenizer, **kwargs):
            local_plaus_calls.append((list(facts), transcript))
            return kept_deanon, ""

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.cloud_egress.anonymize_with_local_model",
                return_value=(mapping, "anonymized transcript", ""),
            ),
            patch(
                "paramem.graph.extractor._filter_with_sota",
                return_value=(anon_facts, None, {}, None, {}),
            ),
            patch(
                "paramem.graph.extractor.local_plausibility_filter",
                side_effect=fake_local_plaus,
            ),
        ):
            result = run_sota_stages(
                graph,
                "Alex lives in Millfield.",
                MagicMock(),
                MagicMock(),
                speaker_id="speaker0",
                plausibility_judge="auto",
                plausibility_stage="deanon",
                correction_entity_types=set(),
                scrub={"person name"},
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


class TestAnonFailureFallback:
    """When anonymization fails, _sota_pipeline runs raw (local) plausibility
    instead of returning the original facts.
    """

    def test_anon_failure_triggers_fallback(self):
        """_sota_pipeline calls _fallback_plausibility_on_raw when anonymization fails."""
        from tests._sota_flow import run_sota_stages

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
                "paramem.graph.cloud_egress.anonymize_with_local_model",
                return_value=(None, "", ""),
            ),
            patch(
                "paramem.graph.extractor._fallback_plausibility_on_raw",
                side_effect=fake_fallback,
            ),
        ):
            result = run_sota_stages(
                graph,
                "transcript",
                None,
                None,
                speaker_id="speaker0",
                correction_entity_types=set(),
                scrub={"person name"},
            )

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
        from paramem.graph.extractor import ExtractionFailed
        from tests._sota_flow import run_sota_stages

        graph = _make_graph(
            [("Alex", "lives_in", "Millfield")],
            entities=[
                Entity(name="Alex", entity_type="person"),
                Entity(name="Millfield", entity_type="place"),
            ],
        )
        mapping = {"Alex": "Person_1", "Millfield": "City_1"}

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.cloud_egress.anonymize_with_local_model",
                return_value=(mapping, "anonymized transcript", ""),
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
                run_sota_stages(
                    graph,
                    "transcript",
                    None,
                    None,
                    speaker_id="speaker0",
                    correction_entity_types=set(),
                    scrub={"person name"},
                )
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
        from tests._sota_flow import run_sota_stages

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
                "paramem.graph.cloud_egress.anonymize_with_local_model",
                return_value=(mapping, "anonymized transcript", ""),
            ),
            patch(
                "paramem.graph.extractor._filter_with_sota",
                return_value=(sota_enriched, None, {}, None, {}),
            ),
            patch(
                "paramem.graph.extractor.local_plausibility_filter",
                return_value=([], ""),
            ),
            patch(
                "paramem.graph.extractor._fallback_plausibility_on_raw",
                side_effect=fake_fallback,
            ),
        ):
            result = run_sota_stages(
                graph,
                "Alex lives in Millfield.",
                MagicMock(),  # non-None model so deanon-stage plausibility runs
                MagicMock(),  # non-None tokenizer so deanon-stage plausibility runs
                speaker_id="speaker0",
                plausibility_judge="auto",
                plausibility_stage="deanon",
                correction_entity_types=set(),
                scrub={"person name"},
            )

        assert "all_dropped" in fallback_calls, f"Expected all_dropped, got: {fallback_calls}"
        assert result.diagnostics.get("fallback_path") == "all_dropped"
        # The gate fires on a SYMPTOM (no relations survived); the cause
        # tells the three situations that reach it apart. Here the
        # deanon-stage judge dropped everything — a judgment, not a
        # mechanical breakage. Recorded only: nothing branches on it.
        assert result.diagnostics.get("all_dropped_cause") == {
            "cause": "deanon_judge_dropped_all",
            "kind": "judgment",
        }


class TestEntityTypePreservation:
    """Entity types set by _normalize_extraction must survive the SOTA pipeline
    unchanged; no "person" stampdown on non-person entities.
    """

    def test_preserved_entity_types_pass_through(self):
        """Entities pre-typed by _normalize_extraction keep their original types
        after the pipeline even when mocked SOTA returns same facts."""
        from tests._sota_flow import run_sota_stages

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

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.cloud_egress.anonymize_with_local_model",
                return_value=(mapping, "anonymized transcript", ""),
            ),
            patch(
                "paramem.graph.extractor._filter_with_sota",
                return_value=(anon_facts, None, {}, None, {}),
            ),
        ):
            result = run_sota_stages(
                graph,
                "Alex lives in Frankfurt and listens to Music.",
                None,
                None,
                speaker_id="speaker0",
                plausibility_judge="off",
                correction_entity_types=set(),
                scrub={"person name"},
            )

        entity_map = {e.name: e.entity_type for e in result.entities}
        assert entity_map.get("Alex") == "person"
        assert entity_map.get("Frankfurt") in ("place", "location")
        assert entity_map.get("Music") == "concept", (
            f"Music must be 'concept', not {entity_map.get('Music')!r}"
        )

    def test_sota_introduced_country_entity_typed_location(self):
        """SOTA-introduced entity with Country_ placeholder is typed 'location', not 'person'."""
        from tests._sota_flow import run_sota_stages

        graph = _make_graph(
            [("Alex", "born_in", "Germany")],
            entities=[
                Entity(name="Alex", entity_type="person"),
                Entity(name="Germany", entity_type="place"),
            ],
        )
        anon_facts = [{"subject": "Person_1", "predicate": "born_in", "object": "Country_1"}]
        mapping = {"Alex": "Person_1", "Germany": "Country_1"}

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.cloud_egress.anonymize_with_local_model",
                return_value=(mapping, "anonymized transcript", ""),
            ),
            patch(
                "paramem.graph.extractor._filter_with_sota",
                return_value=(anon_facts, None, {}, None, {}),
            ),
        ):
            result = run_sota_stages(
                graph,
                "Alex was born in Germany.",
                None,
                None,
                speaker_id="speaker0",
                plausibility_judge="off",
                correction_entity_types=set(),
                scrub={"person name"},
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
        from tests._sota_flow import run_sota_stages

        # Original graph has only Alex — no China entity
        graph = _make_graph(
            [("Alex", "has_plans", "Alex")],  # placeholder relation; SOTA will override
            entities=[
                Entity(name="Alex", entity_type="person"),
            ],
        )
        # Alex → Person_1 only; China is absent from the anonymization mapping
        mapping = {"Alex": "Person_1"}
        # SOTA enrichment introduces China as a bare name with no placeholder equivalent
        enriched_anon = [{"subject": "Person_1", "predicate": "visited", "object": "China"}]

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.cloud_egress.anonymize_with_local_model",
                return_value=(mapping, "anonymized transcript", ""),
            ),
            patch(
                "paramem.graph.extractor._filter_with_sota",
                return_value=(enriched_anon, None, {}, None, {}),
            ),
        ):
            result = run_sota_stages(
                graph,
                "Alex visited China.",
                None,
                None,
                speaker_id="speaker0",
                plausibility_judge="off",
                correction_entity_types=set(),
                scrub={"person name"},
            )

        entity_map = {e.name: e.entity_type for e in result.entities}
        # China has no reverse_mapping entry → safe fallback type is "concept", not "person"
        china_type = entity_map.get("China")
        assert china_type == "concept", (
            f"SOTA-introduced bare entity must be typed 'concept', not {china_type!r}"
        )


class TestSotaMintedEntityTypeDerivation:
    """The entity-rebuild loop (extractor.py, "Rebuild entity list from
    surviving + new relations") must resolve a de-anonymized SOTA-minted
    entity's REAL NAME back to its placeholder via the inverted
    resolution map, not via ``reverse_mapping.get(name)`` — ``reverse_mapping``
    is keyed by placeholder, so looking it up with a real name always
    misses (dead code prior to the fix under test).
    """

    def test_sota_minted_entity_gets_prefix_derived_type(self):
        """A SOTA-minted entity bound via ``bindings`` with a novel prefix
        (``Paper_1``, absent from the closed anonymizer vocabulary) lands
        in graph.entities typed by its prefix ("paper"), not "concept".

        Mutation that must make this fail: restore the lookup to
        ``reverse_mapping.get(name)`` (today's dead-code behaviour) —
        ``reverse_mapping`` has no "Attention Is All You Need" key (it is
        keyed by placeholder), so the entity falls back to "concept".
        """
        from tests._sota_flow import run_sota_stages

        graph = _make_graph(
            [("Alex", "has_plans", "Alex")],  # placeholder relation; SOTA will override
            entities=[
                Entity(name="Alex", entity_type="person"),
            ],
        )
        mapping = {"Alex": "Person_1"}
        # SOTA mints a novel-prefix entity via a braced placeholder plus an
        # explicit binding — the documented brace-binding protocol.
        enriched_anon = [{"subject": "Person_1", "predicate": "authored", "object": "{Paper_1}"}]
        sota_bindings = {"Paper_1": "Attention Is All You Need"}

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.cloud_egress.anonymize_with_local_model",
                return_value=(mapping, "anonymized transcript", ""),
            ),
            patch(
                "paramem.graph.extractor._filter_with_sota",
                return_value=(enriched_anon, None, sota_bindings, None, {}),
            ),
        ):
            result = run_sota_stages(
                graph,
                "Alex authored Attention Is All You Need.",
                None,
                None,
                speaker_id="speaker0",
                plausibility_judge="off",
                correction_entity_types=set(),
                scrub={"person name"},
            )

        entity_map = {e.name: e.entity_type for e in result.entities}
        paper_type = entity_map.get("Attention Is All You Need")
        assert paper_type == "paper", (
            f"SOTA-minted novel-prefix entity must be typed 'paper', not {paper_type!r}"
        )

    def test_sota_minted_entity_known_prefix_uses_configured_type(self):
        """A SOTA-minted entity with a KNOWN prefix (``Person_2``) still
        maps through the schema's ``anonymizer_prefix_to_type()`` to
        "person" — the closed-vocabulary branch of the same derivation.
        """
        from tests._sota_flow import run_sota_stages

        graph = _make_graph(
            [("Alex", "has_plans", "Alex")],
            entities=[
                Entity(name="Alex", entity_type="person"),
            ],
        )
        mapping = {"Alex": "Person_1"}
        enriched_anon = [{"subject": "Person_1", "predicate": "met", "object": "{Person_2}"}]
        sota_bindings = {"Person_2": "Jordan Rivers"}

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.cloud_egress.anonymize_with_local_model",
                return_value=(mapping, "anonymized transcript", ""),
            ),
            patch(
                "paramem.graph.extractor._filter_with_sota",
                return_value=(enriched_anon, None, sota_bindings, None, {}),
            ),
        ):
            result = run_sota_stages(
                graph,
                "Alex met Jordan Rivers.",
                None,
                None,
                speaker_id="speaker0",
                plausibility_judge="off",
                correction_entity_types=set(),
                scrub={"person name"},
            )

        entity_map = {e.name: e.entity_type for e in result.entities}
        jordan_type = entity_map.get("Jordan Rivers")
        assert jordan_type == "person", (
            f"SOTA-minted known-prefix entity must be typed 'person', not {jordan_type!r}"
        )


class TestFallbackPlausibilityOnRawHelper:
    """Direct tests of the _fallback_plausibility_on_raw helper: runs on
    already-de-anonymized ``graph.relations`` (real names) and records
    anon-failed facts.
    """

    def test_helper_does_not_sweep_shape_like_real_names(self):
        """The residual-placeholder sweep this helper used to run
        (``_strip_residual_placeholders``) is retired: this path
        operates on real-name ``graph.relations`` where no placeholder
        vocabulary exists, so a real name that merely happens to be
        shaped like a placeholder (``Boeing_747``) must survive — a
        shape-only guard here could only ever produce false positives."""
        from paramem.graph.extractor import _fallback_plausibility_on_raw

        graph = _make_graph(
            [
                ("Alex", "owns", "Boeing_747"),  # shape-like real name, not a placeholder
                ("Alex", "works_at", "Acme"),
            ],
            entities=[
                Entity(name="Alex", entity_type="person"),
                Entity(name="Boeing_747", entity_type="concept"),
                Entity(name="Acme", entity_type="organization"),
            ],
        )
        result = _fallback_plausibility_on_raw(
            graph,
            "Alex owns a Boeing_747 and works at Acme.",
            None,
            None,
            speaker_id="speaker0",
            reason="test_residual",
        )
        surviving = {r.object for r in result.relations}
        assert "Boeing_747" in surviving
        assert result.diagnostics.get("fallback_path") == "test_residual"
        assert "residual_dropped_facts" not in result.diagnostics

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
            speaker_id="speaker0",
            reason="anon_failed",
        )
        assert result.diagnostics.get("fallback_path") == "anon_failed"


class TestExtractGraphNewKwargs:
    """extract_graph forwards plausibility kwargs (plausibility_judge,
    plausibility_stage) to _sota_pipeline.
    """

    def test_extract_graph_plumbs_plausibility_kwargs(self):
        """extract_graph forwards plausibility_judge, plausibility_stage to
        _sota_pipeline."""
        from paramem.graph.extractor import extract_graph
        from paramem.graph.flow import StageState

        captured = {}

        def fake_sota_pipeline(graph, transcript, model, tokenizer, **kwargs):
            # The composite returns a StageState now — an empty ``facts``
            # is its ``terminal_when``, so the deanonymize/rebuild
            # siblings do not run on this stub's output.
            captured.update(kwargs)
            return StageState(graph=graph)

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
                speaker_id="speaker0",
                sota_enabled=True,
                noise_filter="anthropic",
                plausibility_judge="claude",
                plausibility_stage="anon",
                scrub={"person name"},
            )

        assert captured.get("plausibility_judge") == "claude"
        assert captured.get("plausibility_stage") == "anon"

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


class TestDiagnosticsKeys:
    """Diagnostics dict is populated with expected keys after a full pipeline run."""

    def test_diagnostics_contains_plausibility_keys(self):
        """After a deanon-stage plausibility run, diagnostics contains the expected keys."""
        from tests._sota_flow import run_sota_stages

        graph = _make_graph(
            [("Alex", "lives_in", "Millfield")],
            entities=[
                Entity(name="Alex", entity_type="person"),
                Entity(name="Millfield", entity_type="place"),
            ],
        )
        anon_facts = [{"subject": "Person_1", "predicate": "lives_in", "object": "City_1"}]
        mapping = {"Alex": "Person_1", "Millfield": "City_1"}

        def fake_local_plaus(facts, transcript, model, tokenizer, **kwargs):
            return facts, ""  # keep all

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.cloud_egress.anonymize_with_local_model",
                return_value=(mapping, "anonymized transcript", ""),
            ),
            patch(
                "paramem.graph.extractor._filter_with_sota",
                return_value=(anon_facts, None, {}, None, {}),
            ),
            patch(
                "paramem.graph.extractor.local_plausibility_filter",
                side_effect=fake_local_plaus,
            ),
        ):
            result = run_sota_stages(
                graph,
                "Alex lives in Millfield.",
                MagicMock(),
                MagicMock(),
                speaker_id="speaker0",
                plausibility_judge="auto",
                plausibility_stage="deanon",
                correction_entity_types=set(),
                scrub={"person name"},
            )

        assert "plausibility" in result.diagnostics, "diagnostics must contain 'plausibility'"
        assert "plausibility_dropped_deanon" in result.diagnostics
        assert "plausibility_judge_actual" in result.diagnostics
        assert "anonymize" in result.diagnostics

    def test_diagnostics_anonymize_key_populated_on_success(self):
        """diagnostics['anonymize']='ok' when anonymization succeeds."""
        from tests._sota_flow import run_sota_stages

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
                "paramem.graph.cloud_egress.anonymize_with_local_model",
                return_value=(mapping, "anonymized transcript", ""),
            ),
            patch(
                "paramem.graph.extractor._filter_with_sota",
                return_value=(anon_facts, None, {}, None, {}),
            ),
        ):
            result = run_sota_stages(
                graph,
                "Alex lives in Millfield.",
                None,
                None,
                speaker_id="speaker0",
                plausibility_judge="off",
                correction_entity_types=set(),
                scrub={"person name"},
            )

        assert result.diagnostics.get("anonymize") == "ok"

    def test_mapping_ambiguous_dropped_is_live_and_reaches_diagnostics(self):
        """``payload.norm_stats["dropped"]`` is a LIVE signal now — the
        ONE normalize call in the chain (inside ``anonymize_for_cloud``)
        — reaching ``graph.diagnostics["mapping_ambiguous_dropped"]``.

        Before this unification, ``_sota_pipeline`` ran a SECOND,
        redundant outer normalize on an already-canonical table (the
        internal normalize inside ``anonymize_with_local_model`` had
        already dropped every ambiguous pair), so
        ``mapping_ambiguous_dropped`` could structurally never be
        non-zero — this test would have failed against that code.

        Mutation: reintroduce a second (now-dead) normalize call before
        the diagnostic is set -> ``dropped`` reads 0 -> this test fails.
        """
        from tests._sota_flow import run_sota_stages

        graph = _make_graph(
            [("Alex", "lives_in", "Millfield")],
            entities=[
                Entity(name="Alex", entity_type="person"),
                Entity(name="Millfield", entity_type="place"),
            ],
        )
        anon_facts = [{"subject": "Person_1", "predicate": "lives_in", "object": "City_1"}]
        # "foo": "bar" is a both-sides-ambiguous pair: NEITHER side
        # matches the placeholder shape, so the normalizer drops it.
        mapping = {"Alex": "Person_1", "foo": "bar"}

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.cloud_egress.anonymize_with_local_model",
                return_value=(mapping, "anonymized transcript", ""),
            ),
            patch(
                "paramem.graph.extractor._filter_with_sota",
                return_value=(anon_facts, None, {}, None, {}),
            ),
        ):
            result = run_sota_stages(
                graph,
                "Alex lives in Millfield.",
                None,
                None,
                speaker_id="speaker0",
                plausibility_judge="off",
                correction_entity_types=set(),
                scrub={"person name"},
            )

        assert result.diagnostics.get("mapping_ambiguous_dropped") == 1


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

    def test_extraction_noise_filter_defaults_to_disabled(self):
        """Privacy invariant: a config that omits ``extraction_noise_filter``
        must NOT default to a cloud provider. ``""`` is the disabled
        sentinel — a deployment whose YAML omits the key must not silently
        send extraction-pipeline content to the cloud (see SECURITY.md's
        stated default posture).
        """
        from paramem.server.config import ConsolidationScheduleConfig

        cfg = ConsolidationScheduleConfig()
        assert cfg.extraction_noise_filter == ""

    def test_minimal_yaml_loads_with_defaults(self, tmp_path):
        """Back-compat: minimal yaml without new keys loads with all new defaults.

        Pre-flight check #2: minimal yaml without new keys must load with all new defaults.
        """
        from paramem.server.config import load_server_config

        minimal_yaml = tmp_path / "server.yaml"
        minimal_yaml.write_text(
            "model: mistral\nconsolidation:\n  refresh_cadence: every 2h\n  mode: simulate\n"
        )
        config = load_server_config(minimal_yaml)
        # New fields must be present with defaults
        assert config.consolidation.extraction_plausibility_judge == "auto"
        assert config.consolidation.extraction_plausibility_stage == "deanon"


# ---------------------------------------------------------------------------
# Mapping totality — diagnostic check post-anonymization
# ---------------------------------------------------------------------------


class TestCheckMappingTotality:
    """Unit tests for ``_check_mapping_totality``.

    The diagnostic replaces the retired
    ``_recover_missing_placeholder_mappings`` helper.  It is checked
    against the REVERSE map (``{placeholder: real_name}``) rather than
    the forward map's values, because deanonymization consumes the
    reverse map (:func:`_apply_bindings`) — a placeholder present only
    in the forward map's values still fails to translate at deanon
    time.  Under the open-vocabulary anonymizer prompt the LLM is
    expected to produce a total mapping by construction (see live probe
    at the prompt-pivot commit); this helper surfaces violations to
    ``logger`` and, as RETURN VALUES, to whatever diagnostics its caller
    keeps, so prompt regressions are visible rather than silently
    shedding facts.  Orphan-placeholder facts get dropped downstream by
    :func:`_apply_bindings`'s residual sweep — fail-closed.

    The function takes no ``SessionGraph`` and writes nothing: it returns
    ``(verdict, collisions)``.  Every assertion below is on the returned
    tuple; the caller-side diagnostics writes those two values feed are
    pinned separately (``_record_binding_diagnostics`` in
    ``paramem.graph.extractor``).
    """

    def test_total_mapping_records_no_orphans(self):
        """Every fact placeholder resolves via the reverse map → empty
        verdict."""
        from paramem.graph.placeholders import _check_mapping_totality

        anon_facts = [
            {"subject": "Person_1", "predicate": "lives_in", "object": "City_1"},
        ]
        reverse_mapping = {"Person_1": "Alex", "City_1": "Berlin"}
        verdict, collisions = _check_mapping_totality(anon_facts, reverse_mapping)
        assert verdict == []
        assert collisions == []

    def test_orphan_placeholder_recorded(self):
        """A fact placeholder absent from ``reverse_mapping`` (the keys
        deanon actually looks up) comes back in the verdict.  No mutation
        of inputs."""
        from paramem.graph.placeholders import _check_mapping_totality

        anon_facts = [
            {"subject": "Person_1", "predicate": "studied_at", "object": "University_1"},
        ]
        # University_1 is missing from reverse_mapping — a totality violation.
        reverse_mapping = {"Person_1": "Alex"}
        verdict, _collisions = _check_mapping_totality(anon_facts, reverse_mapping)
        assert verdict == ["University_1"]
        # Inputs must not be mutated.
        assert reverse_mapping == {"Person_1": "Alex"}

    def test_multiple_orphans_sorted(self):
        """Multiple orphans are deduplicated and sorted for stable
        diagnostic output."""
        from paramem.graph.placeholders import _check_mapping_totality

        anon_facts = [
            {"subject": "Org_1", "predicate": "made", "object": "Product_1"},
            {"subject": "Person_1", "predicate": "speaks", "object": "Language_1"},
            {"subject": "Person_1", "predicate": "uses", "object": "Product_1"},
        ]
        # Person_1 is in reverse_mapping but Org_1, Product_1, Language_1 are not.
        reverse_mapping = {"Person_1": "Alex"}
        verdict, _collisions = _check_mapping_totality(anon_facts, reverse_mapping)
        assert verdict == [
            "Language_1",
            "Org_1",
            "Product_1",
        ]

    def test_embedded_placeholder_caught(self):
        """Placeholder embedded in a compound string still surfaces as
        an orphan when missing from ``reverse_mapping``."""
        from paramem.graph.placeholders import _check_mapping_totality

        anon_facts = [
            {
                "subject": "Person_1",
                "predicate": "led",
                "object": "software for Product_1's Legend",
            },
        ]
        reverse_mapping = {"Person_1": "Alex"}
        verdict, _collisions = _check_mapping_totality(anon_facts, reverse_mapping)
        assert verdict == ["Product_1"]

    def test_empty_facts_short_circuits(self):
        """No facts → empty verdict, regardless of reverse_mapping shape."""
        from paramem.graph.placeholders import _check_mapping_totality

        assert _check_mapping_totality([], {}) == ([], [])
        assert _check_mapping_totality([], {"Person_1": "Alex"}) == ([], [])

    def test_placeholder_in_forward_values_but_absent_from_reverse_keys_flagged(self):
        """The check must key off the reverse map, not the forward map.
        A placeholder that exists as a forward-map VALUE (so the old
        ``mapping.values()`` check would have passed it) but is absent
        from the reverse map's KEYS is still flagged — this is exactly
        the shape the incident produced: the LLM's placeholder made it
        into the forward map's conflict-losing value but never reached
        the reverse map.
        """
        from paramem.graph.placeholders import _check_mapping_totality

        anon_facts = [
            {"subject": "Person_4", "predicate": "lives_in", "object": "Berlin"},
        ]
        # Forward map has "Alex": "Person_4" as a value (would have
        # passed the old, wrong check), but the reverse map only knows
        # the deterministic winner Person_1 — Person_4 is unresolved.
        forward_map = {"Alex": "Person_1", "SomeoneElse": "Person_4"}
        reverse_mapping = {"Person_1": "Alex"}
        assert "Person_4" in set(forward_map.values())
        verdict, _collisions = _check_mapping_totality(anon_facts, reverse_mapping)
        assert verdict == ["Person_4"]

    def test_placeholder_present_in_reverse_keys_passes(self):
        """Once the placeholder is a key in the reverse map, the same
        fact passes with an empty verdict."""
        from paramem.graph.placeholders import _check_mapping_totality

        anon_facts = [
            {"subject": "Person_4", "predicate": "lives_in", "object": "Berlin"},
        ]
        reverse_mapping = {"Person_1": "Alex", "Person_4": "Alex"}
        verdict, _collisions = _check_mapping_totality(anon_facts, reverse_mapping)
        assert verdict == []

    def test_post_sota_missing_binding_predicted_before_drop(self, caplog):
        """A fact referencing a braced placeholder absent from BOTH
        ``sota_bindings`` and ``reverse_mapping`` comes back in the
        verdict and is logged BEFORE :func:`_apply_bindings` drops the
        fact."""
        import logging

        from paramem.graph.placeholders import _apply_bindings, _check_mapping_totality

        anon_facts = [
            {"subject": "Person_1", "predicate": "works_at", "object": "{Org_9}"},
        ]
        reverse_mapping = {"Person_1": "Alex"}
        sota_bindings: dict = {}
        # caplog.at_level() silently fails here because the logger is not
        # propagating to the root; attach the handler to the specific
        # logger directly (see test_skips_unrecognised_class_filenames pattern
        # in test_intent.py).
        placeholders_logger = logging.getLogger("paramem.graph.placeholders")
        prior_level = placeholders_logger.level
        placeholders_logger.setLevel(logging.WARNING)
        placeholders_logger.addHandler(caplog.handler)
        try:
            verdict, _collisions = _check_mapping_totality(
                anon_facts,
                reverse_mapping,
                sota_bindings=sota_bindings,
            )
        finally:
            placeholders_logger.removeHandler(caplog.handler)
            placeholders_logger.setLevel(prior_level)
        assert verdict == ["Org_9"]
        assert any("binding-totality violation" in r.getMessage().lower() for r in caplog.records)
        kept, predicate_dropped, residual_dropped = _apply_bindings(
            anon_facts, reverse_mapping, sota_bindings
        )
        dropped = predicate_dropped + residual_dropped
        assert kept == []
        assert len(dropped) == 1

    def test_post_sota_nested_binding_value_unbound_predicted(self):
        """A binding value that itself contains an unresolved bare
        placeholder (``"... at Org_9"``) is predicted as an orphan even
        though no fact directly references ``Org_9``."""
        from paramem.graph.placeholders import _check_mapping_totality

        anon_facts = [
            {"subject": "Person_1", "predicate": "held_role", "object": "{Role_1}"},
        ]
        reverse_mapping: dict = {}
        sota_bindings = {"Role_1": "Senior Engineer at Org_9"}
        verdict, _collisions = _check_mapping_totality(
            anon_facts,
            reverse_mapping,
            sota_bindings=sota_bindings,
        )
        assert "Org_9" in verdict

    def test_collision_between_sota_bindings_and_reverse_recorded(self, caplog):
        """A key present in BOTH ``sota_bindings`` and ``reverse_mapping``
        with differing values comes back as a ``collisions`` entry and is
        warned about — the reverse-wins tie-break in
        :func:`_apply_bindings` would otherwise silently resolve to the
        wrong real name."""
        import logging

        from paramem.graph.placeholders import _check_mapping_totality

        anon_facts = [
            {"subject": "Org_1", "predicate": "based_in", "object": "Germany"},
        ]
        reverse_mapping = {"Org_1": "Acme"}
        sota_bindings = {"Org_1": "Wrong Corp"}
        # caplog.at_level() silently fails here because the logger is not
        # propagating to the root; attach the handler to the specific
        # logger directly (see test_skips_unrecognised_class_filenames pattern
        # in test_intent.py).
        placeholders_logger = logging.getLogger("paramem.graph.placeholders")
        prior_level = placeholders_logger.level
        placeholders_logger.setLevel(logging.WARNING)
        placeholders_logger.addHandler(caplog.handler)
        try:
            verdict, collisions = _check_mapping_totality(
                anon_facts,
                reverse_mapping,
                sota_bindings=sota_bindings,
            )
        finally:
            placeholders_logger.removeHandler(caplog.handler)
            placeholders_logger.setLevel(prior_level)
        assert collisions == ["Org_1"]
        # CORE-unscoped: a collision is informational only, never folded
        # into the verdict.
        assert verdict == []
        assert any("collision" in r.getMessage().lower() for r in caplog.records)

    def test_pre_sota_positional_calls_unaffected_by_generalization(self):
        """The existing positional pre-SOTA call sites (this class's
        earlier tests) keep passing unchanged — defaults preserve
        behaviour: ``sota_bindings=None`` skips the collision scan, so
        ``collisions`` comes back empty."""
        from paramem.graph.placeholders import _check_mapping_totality

        anon_facts = [
            {"subject": "Person_1", "predicate": "studied_at", "object": "University_1"},
        ]
        reverse_mapping = {"Person_1": "Alex"}
        verdict, collisions = _check_mapping_totality(anon_facts, reverse_mapping)
        assert verdict == ["University_1"]
        assert collisions == []

    # -- Explicit-return contract, verdict content --------

    def test_returns_empty_list_not_none_on_clean_input(self):
        """The totality check returns ``[]`` (not ``None``) when the
        mapping is total — the explicit-return contract that makes a
        plain truthiness test safe for callers."""
        from paramem.graph.placeholders import _check_mapping_totality

        anon_facts = [{"subject": "Person_1", "predicate": "lives_in", "object": "Berlin"}]
        reverse_mapping = {"Person_1": "Alex"}
        verdict, collisions = _check_mapping_totality(anon_facts, reverse_mapping)
        assert verdict == []
        assert verdict is not None
        assert collisions == []
        assert collisions is not None

    def test_returns_empty_list_not_none_on_empty_facts(self):
        """The totality check's second explicit exit: the early ``if not anon_facts``
        guard returns ``[]``, not the implicit ``None`` a bare ``return``
        would give — invisible to a caller that only writes ``if
        verdict:``."""
        from paramem.graph.placeholders import _check_mapping_totality

        verdict, collisions = _check_mapping_totality([], {"Person_1": "Alex"})
        assert verdict == []
        assert verdict is not None
        assert collisions == []

    def test_returns_token_list_on_poisoned_delta(self):
        """The verdict IS the sorted offending-token list — the function
        has no side effect to observe it through."""
        from paramem.graph.placeholders import _check_mapping_totality

        anon_facts = [
            {"subject": "Person_1", "predicate": "studied_at", "object": "University_1"},
        ]
        reverse_mapping = {"Person_1": "Alex"}
        verdict, _collisions = _check_mapping_totality(anon_facts, reverse_mapping)
        assert verdict == ["University_1"]

    def test_conflict_key_folded_into_verdict_when_observed_scoped(self):
        """When ``observed`` is a set, a ``sota_bindings`` key colliding
        with it is folded into the RETURNED verdict AND surfaces
        separately as a ``collisions`` entry."""
        from paramem.graph.placeholders import _check_mapping_totality

        anon_facts = [{"subject": "Person_1", "predicate": "lives_in", "object": "Berlin"}]
        reverse_mapping = {"Person_1": "Alex"}
        sota_bindings = {"Person_1": "someone else entirely"}
        observed = {"Person_1"}
        verdict, collisions = _check_mapping_totality(
            anon_facts,
            reverse_mapping,
            sota_bindings=sota_bindings,
            observed=observed,
        )
        assert verdict == ["Person_1"]
        assert collisions == ["Person_1"]

    def test_collision_verdict_returned_even_with_empty_anon_facts(self):
        """A non-empty verdict from the collision scan ALONE —
        ``anon_facts`` is empty, so there is nothing for the per-fact scan
        to find — must still reach the caller.  This is the exact shape
        ``graph_enrich.run_graph_enrichment``'s ``totality_rejected_chunks``
        counter reads (now off the verdict ``_graph_enrich_with_sota``
        returns); a verdict that is non-empty but never surfaced is
        silently under-counted as "no rejection" — worse than no gate at
        all.

        Mutation: return the collision-only verdict via an early
        ``return []`` that bypasses the ``if orphans:`` exit (the shape
        this function had before the fix) -> ``verdict`` comes back empty
        -> this test fails.
        """
        from paramem.graph.placeholders import _check_mapping_totality

        reverse_mapping = {"Person_1": "Alex"}
        sota_bindings = {"Person_1": "someone else entirely"}
        observed = {"Person_1"}
        verdict, collisions = _check_mapping_totality(
            [],  # no facts -- the early-return path the bug lived in
            reverse_mapping,
            sota_bindings=sota_bindings,
            observed=observed,
        )
        assert verdict == ["Person_1"]
        assert collisions == ["Person_1"]


class TestRecordBindingDiagnostics:
    """``_record_binding_diagnostics`` — the CALLER side of the totality
    gate, and the only place in the extractor that turns a
    ``DeanonResult`` into ``graph.diagnostics`` entries.

    The two keys used to be written by ``_check_mapping_totality`` itself,
    from inside ``deanonymize_facts``, onto a ``SessionGraph`` passed
    purely as a sink.  These tests pin the guard conditions that move
    with them: an EMPTY list writes NO key, so ``"key" not in
    diagnostics`` keeps meaning "the scan found nothing".
    """

    @staticmethod
    def _result(verdict: list[str], collisions: list[str]):
        from paramem.graph.cloud_egress import DeanonResult

        return DeanonResult(
            facts=[],
            verdict=verdict,
            collisions=collisions,
            predicate_dropped=[],
            residual_dropped=[],
        )

    def test_empty_findings_write_no_keys(self):
        """Mutation: drop the ``if`` guards -> an accepted delta starts
        writing two empty-list keys, and every ``"..." not in
        diagnostics`` assertion in the suite flips meaning."""
        from paramem.graph.extractor import _record_binding_diagnostics

        graph = _make_graph([])
        _record_binding_diagnostics(graph, self._result([], []))
        assert "sota_pending_orphans" not in graph.diagnostics
        assert "sota_binding_collisions" not in graph.diagnostics

    def test_verdict_and_collisions_land_under_their_keys(self):
        from paramem.graph.extractor import _record_binding_diagnostics

        graph = _make_graph([])
        _record_binding_diagnostics(graph, self._result(["Org_1"], ["Person_2"]))
        assert graph.diagnostics["sota_pending_orphans"] == ["Org_1"]
        assert graph.diagnostics["sota_binding_collisions"] == ["Person_2"]

    def test_collisions_without_a_verdict_still_recorded(self):
        """CORE-unscoped shape: a collision is informational only and is
        NOT folded into the verdict, so it must not depend on the verdict
        being non-empty to be recorded."""
        from paramem.graph.extractor import _record_binding_diagnostics

        graph = _make_graph([])
        _record_binding_diagnostics(graph, self._result([], ["Person_2"]))
        assert graph.diagnostics["sota_binding_collisions"] == ["Person_2"]
        assert "sota_pending_orphans" not in graph.diagnostics


class TestResolutionMap:
    """CORE PRECEDENCE, pinned directly on :func:`_resolution_map`,
    independent of the rejection gate.  Backstops need their own test: do
    not skip this because the rejection gate makes the collision unreachable
    in the full pipeline — a future refactor flipping the ``.update()``
    order would otherwise silently let SOTA overwrite a real name with no
    test failing.
    """

    def test_core_wins_on_key_in_both_maps_unscoped(self):
        """observed=None (CORE unscoped): reverse wins on collision —
        today's behaviour, preserved."""
        from paramem.graph.placeholders import _resolution_map

        reverse = {"Org_1": "Acme"}
        sota_bindings = {"Org_1": "Wrong Corp"}
        resolved = _resolution_map(reverse, sota_bindings, observed=None)
        assert resolved["Org_1"] == "Acme"

    def test_core_wins_on_key_in_both_maps_scoped(self):
        """observed as a set: same key in both domains still resolves to
        CORE — vacuous under normal scoped construction (the rejection
        gate would have already rejected this delta) but must hold if
        the map is asked to resolve one directly."""
        from paramem.graph.placeholders import _resolution_map

        reverse = {"Org_1": "Acme"}
        sota_bindings = {"Org_1": "Wrong Corp"}
        resolved = _resolution_map(reverse, sota_bindings, observed={"Org_1"})
        assert resolved["Org_1"] == "Acme"

    def test_observed_none_is_core_unscoped_every_reverse_entry_legal(self):
        from paramem.graph.placeholders import _resolution_map

        reverse = {"Person_1": "Alex", "City_1": "Berlin"}
        resolved = _resolution_map(reverse, {}, observed=None)
        assert resolved == reverse

    def test_observed_scoped_excludes_reverse_entries_outside_observed(self):
        from paramem.graph.placeholders import _resolution_map

        reverse = {"Person_1": "Alex", "City_1": "Berlin"}
        resolved = _resolution_map(reverse, {}, observed={"Person_1"})
        assert resolved == {"Person_1": "Alex"}

    def test_sota_mint_outside_observed_is_included(self):
        from paramem.graph.placeholders import _resolution_map

        reverse = {"Person_1": "Alex"}
        sota_bindings = {"Event_1": "the workshop"}
        resolved = _resolution_map(reverse, sota_bindings, observed={"Person_1"})
        assert resolved == {"Person_1": "Alex", "Event_1": "the workshop"}


class TestBindingTotalityRejection:
    """Reject invalid SOTA-enrichment deltas instead of applying
    them partially, and fall back to the local-extract facts.  These are
    the pipeline-level tests the binding-totality contract requires.

    FIXTURE MECHANICS (post cloud-egress-PII redesign): ``anon_transcript``
    is the MODEL's own rewrite — the 2nd element of the
    ``anonymize_with_local_model`` mock's return tuple — never mechanically
    rebuilt from ``transcript`` + ``mapping`` (the deleted
    ``_anonymize_transcript`` forward-on-prose call).  ``observed`` is
    derived from the DECLARED token vocabulary intersected with the
    rendered payload (facts JSON + ``anon_transcript``), so a test
    controlling what is "observed" controls the mocked
    ``anonymize_with_local_model`` transcript string and/or the facts the
    graph fixture carries — not a transcript-substitution side effect.
    """

    @staticmethod
    def _graph_and_mapping():
        graph = _make_graph(
            [("Alex", "lives_in", "Millfield")],
            entities=[
                Entity(name="Alex", entity_type="person"),
                Entity(name="Millfield", entity_type="place"),
            ],
        )
        anon_facts = [{"subject": "Person_1", "predicate": "lives_in", "object": "City_1"}]
        mapping = {"Alex": "Person_1", "Millfield": "City_1"}
        return graph, anon_facts, mapping

    def test_poisoned_delta_rejected_local_facts_survive(self, caplog):
        """The observed 5-in-1 collapse.  A poisoned delta that
        drops the local fact and adds 5 facts over bare, unbound
        Person_2/Person_3 (``bindings={}``) must be REJECTED as a whole:
        the local-extract facts survive de-anonymized, no residual
        placeholder remains, ``sota_enrichment_rejected`` is recorded,
        the ``sota_enrich`` phase outcome is ``"rejected"``, and an ERROR
        is logged naming the offending tokens."""
        import logging

        from paramem.graph.phase_trace import extraction_trace
        from tests._sota_flow import run_sota_stages

        graph, anon_facts, mapping = self._graph_and_mapping()
        enriched_anon = [
            {
                "subject": "Person_2",
                "predicate": "married_to",
                "object": "Person_3",
                "relation_type": "social",
                "confidence": 0.9,
            },
            {
                "subject": "Person_3",
                "predicate": "profession",
                "object": "teacher",
                "relation_type": "factual",
                "confidence": 0.9,
            },
            {
                "subject": "Person_2",
                "predicate": "likes",
                "object": "hiking",
                "relation_type": "factual",
                "confidence": 0.8,
            },
            {
                "subject": "Person_2",
                "predicate": "likes",
                "object": "cooking",
                "relation_type": "factual",
                "confidence": 0.8,
            },
            {
                "subject": "Person_3",
                "predicate": "works_at",
                "object": "Org_1",
                "relation_type": "factual",
                "confidence": 0.8,
            },
        ]

        extractor_logger = logging.getLogger("paramem.graph.extractor")
        prior_level = extractor_logger.level
        extractor_logger.setLevel(logging.WARNING)
        extractor_logger.addHandler(caplog.handler)
        try:
            with extraction_trace() as trace:
                with (
                    patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
                    patch(
                        "paramem.graph.cloud_egress.anonymize_with_local_model",
                        return_value=(mapping, "anonymized transcript", ""),
                    ),
                    patch(
                        "paramem.graph.extractor._filter_with_sota",
                        return_value=(enriched_anon, None, {}, None, {}),
                    ),
                ):
                    result = run_sota_stages(
                        graph,
                        "Alex lives in Millfield",
                        None,
                        None,
                        speaker_id="speaker0",
                        correction_entity_types=set(),
                        scrub={"person name"},
                    )
        finally:
            extractor_logger.removeHandler(caplog.handler)
            extractor_logger.setLevel(prior_level)

        # Local facts survive de-anonymized — the data-saving property.
        assert len(result.relations) == 1
        assert result.relations[0].subject == "Alex"
        assert result.relations[0].object == "Millfield"
        # No residual placeholder anywhere in the surviving relation.
        assert "Person_" not in result.relations[0].subject
        assert "Person_" not in result.relations[0].object
        # Rejection recorded, loudly.
        rejected = result.diagnostics.get("sota_enrichment_rejected")
        assert rejected, "sota_enrichment_rejected diagnostic must be recorded"
        assert any("Person_2" in t or "Person_3" in t for t in rejected)
        phases = {p.name: p for p in trace.records}
        assert phases["sota_enrich"].outcome == "rejected"
        assert any(r.levelname == "ERROR" for r in caplog.records), (
            "A binding-totality breach must log at ERROR, not just warn."
        )

    def test_misattribution_orphan_rejected(self):
        """The misattribution regression (headline).  A placeholder
        NOT in ``observed`` (never shown to SOTA) that SOTA bare-mints is
        an ORPHAN → reject.  Pre-fix this would silently emit a
        fabricated fact bound for adapter weights; post-fix the local
        facts survive."""
        from tests._sota_flow import run_sota_stages

        graph, anon_facts, mapping = self._graph_and_mapping()
        # Person_3 is bare-minted by SOTA but was never shown to it (not
        # in the rendered facts, not in the transcript, not bound).
        enriched_anon = anon_facts + [
            {
                "subject": "Person_3",
                "predicate": "profession",
                "object": "engineer",
                "relation_type": "factual",
                "confidence": 0.9,
            },
        ]
        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.cloud_egress.anonymize_with_local_model",
                return_value=(mapping, "anonymized transcript", ""),
            ),
            patch(
                "paramem.graph.extractor._filter_with_sota",
                return_value=(enriched_anon, None, {}, None, {}),
            ),
        ):
            result = run_sota_stages(
                graph,
                "Alex lives in Millfield",
                None,
                None,
                speaker_id="speaker0",
                correction_entity_types=set(),
                scrub={"person name"},
            )
        assert len(result.relations) == 1
        assert result.relations[0].subject == "Alex"
        assert "sota_enrichment_rejected" in result.diagnostics

    def test_bare_observed_placeholder_as_new_subject_accepted(self):
        """Rule 1 must not regress.  A delta referencing a bare
        OBSERVED placeholder (Person_1 — already shown to SOTA) as the
        subject of a NEW triple, minting nothing, is ACCEPTED.  This test
        and ``test_misattribution_orphan_rejected`` differ ONLY in
        observed-membership."""
        from paramem.graph.phase_trace import extraction_trace
        from tests._sota_flow import run_sota_stages

        graph, anon_facts, mapping = self._graph_and_mapping()
        enriched_anon = anon_facts + [
            {
                "subject": "Person_1",
                "predicate": "born_in",
                "object": "City_1",
                "relation_type": "factual",
                "confidence": 0.9,
            },
        ]
        with extraction_trace() as trace:
            with (
                patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
                patch(
                    "paramem.graph.cloud_egress.anonymize_with_local_model",
                    return_value=(mapping, "anonymized transcript", ""),
                ),
                patch(
                    "paramem.graph.extractor._filter_with_sota",
                    return_value=(enriched_anon, None, {}, None, {}),
                ),
            ):
                result = run_sota_stages(
                    graph,
                    "Alex lives in Millfield",
                    None,
                    None,
                    speaker_id="speaker0",
                    correction_entity_types=set(),
                    scrub={"person name", "physical address"},
                )
        assert len(result.relations) == 2
        assert "sota_enrichment_rejected" not in result.diagnostics
        phases = {p.name: p for p in trace.records}
        assert phases["sota_enrich"].outcome == "ok"

    def test_binding_key_colliding_with_observed_rejected(self):
        """Conflict rejection.  A ``bindings`` key that is itself
        an OBSERVED token (Person_1 — already shown as a core reference)
        is a CONFLICT → rejected, even though it would resolve cleanly
        under the old flat-union design (reverse wins silently)."""
        from tests._sota_flow import run_sota_stages

        graph, anon_facts, mapping = self._graph_and_mapping()
        enriched_anon = list(anon_facts)
        bindings = {"Person_1": "some other person entirely"}
        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.cloud_egress.anonymize_with_local_model",
                return_value=(mapping, "anonymized transcript", ""),
            ),
            patch(
                "paramem.graph.extractor._filter_with_sota",
                return_value=(enriched_anon, None, bindings, None, {}),
            ),
        ):
            result = run_sota_stages(
                graph,
                "Alex lives in Millfield",
                None,
                None,
                speaker_id="speaker0",
                correction_entity_types=set(),
                scrub={"person name"},
            )
        assert "sota_enrichment_rejected" in result.diagnostics
        assert "Person_1" in result.diagnostics["sota_enrichment_rejected"]
        assert len(result.relations) == 1
        assert result.relations[0].subject == "Alex"

    def test_mint_bound_to_descriptor_accepted(self):
        """Mint happy path.  SOTA mints a placeholder BOUND to a
        descriptor span ("my father", ∉ observed) → ACCEPTED; the
        relation de-anonymizes to the bound text."""
        from tests._sota_flow import run_sota_stages

        graph, anon_facts, mapping = self._graph_and_mapping()
        enriched_anon = anon_facts + [
            {
                "subject": "Person_1",
                "predicate": "child_of",
                "object": "{Person_2}",
                "relation_type": "social",
                "confidence": 0.9,
            },
        ]
        bindings = {"Person_2": "my father"}
        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.cloud_egress.anonymize_with_local_model",
                return_value=(mapping, "anonymized transcript", ""),
            ),
            patch(
                "paramem.graph.extractor._filter_with_sota",
                return_value=(enriched_anon, None, bindings, None, {}),
            ),
        ):
            result = run_sota_stages(
                graph,
                "Alex lives in Millfield",
                None,
                None,
                speaker_id="speaker0",
                correction_entity_types=set(),
                scrub={"person name", "physical address"},
            )
        assert "sota_enrichment_rejected" not in result.diagnostics
        subjects_objects = {(r.subject, r.object) for r in result.relations}
        assert ("Alex", "my father") in subjects_objects

    def test_predicate_only_reference_still_observed_accepted(self):
        """A placeholder appearing ONLY in a predicate is still
        ∈ observed (the RENDERED payload, predicate included, is what
        SOTA is actually shown), so a bare reference to it elsewhere is
        ACCEPTED.  Guards the rendered-payload requirement: a
        subject/object-only field scan would under-include ``observed``
        and false-reject this.

        Post cloud-egress-PII redesign: CORE placeholders come straight
        from the model's own anonymizer mapping (there is no
        code-side entity walk that mints for graph entities the model
        didn't name).  So ``anonymize_with_local_model`` is mocked to
        have already classified BOTH places (``Millfield`` -> ``City_1``,
        ``Springfield`` -> ``City_2``) — the model decision this test's
        fixture would need in production for either place to be a CORE
        placeholder at all.
        """
        from tests._sota_flow import run_sota_stages

        # Springfield -> City_2 appears ONLY inside a compound PREDICATE
        # string here, never as a subject/object anywhere in the local
        # extract.  `predicate` is never a substitution target, so the
        # graph relation's own predicate must already carry "City_2"
        # verbatim for it to reach the SOTA-facing payload the script
        # builds — the same text the old model-authored ``anon_facts``
        # stub carried.
        graph = _make_graph(
            [("Alex", "moved from City_2 to", "Millfield")],
            entities=[
                Entity(name="Alex", entity_type="person"),
                Entity(name="Millfield", entity_type="place"),
                Entity(name="Springfield", entity_type="place"),
            ],
        )
        anon_facts = [
            {"subject": "Person_1", "predicate": "moved from City_2 to", "object": "City_1"},
        ]
        # SOTA bare-references City_2 as an object — legal: it is a real
        # CORE placeholder, and observed-scoping must find it via the
        # predicate occurrence above.
        enriched_anon = anon_facts + [
            {
                "subject": "Person_1",
                "predicate": "recently_visited",
                "object": "City_2",
                "relation_type": "factual",
                "confidence": 0.9,
            },
        ]
        mapping = {"Alex": "Person_1", "Millfield": "City_1", "Springfield": "City_2"}
        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.cloud_egress.anonymize_with_local_model",
                return_value=(mapping, "anonymized transcript", ""),
            ),
            patch(
                "paramem.graph.extractor._filter_with_sota",
                return_value=(enriched_anon, None, {}, None, {}),
            ),
        ):
            result = run_sota_stages(
                graph,
                "Alex moved from Springfield to Millfield.",
                None,
                None,
                speaker_id="speaker0",
                correction_entity_types=set(),
                scrub={"person name", "physical address"},
            )
        assert "sota_enrichment_rejected" not in result.diagnostics, (
            "City_2/Springfield appears only in a predicate but is still "
            "observed — a field-scan-only `observed` would false-reject."
        )
        # `enriched_anon[0]` (the "moved from City_2 to" fact) is dropped by
        # `_apply_bindings`'s deanon-stage predicate invariant — its own
        # predicate contains a declared token — so it never reaches
        # `result.relations`.  This assertion rides ENTIRELY on the
        # second, SOTA-added fact ("recently_visited"); it is not
        # evidence the first fact survived.
        subjects_objects = {(r.subject, r.object) for r in result.relations}
        assert ("Alex", "Springfield") in subjects_objects


class TestSpeakerAnchorPipeline:
    """The speaker anchor through the pipeline.  The PII-fold
    regression guard (a PII attribute on the speaker still scrubbed onto
    the anchor, never a minted ``Person_N``) is covered directly against
    :func:`_build_anonymization_mapping` in
    ``tests/test_placeholders.py::TestSpeakerAnchorReverseSkip`` — the
    model's mapping is the sole scope authority post-redesign, so
    there is no graph-entity/attribute fold left in this module to pin
    end to end here.
    """

    def test_speaker0_survives_end_to_end_not_swept(self):
        """``speaker0`` survives extraction -> anonymize -> SOTA ->
        deanon -> graph VERBATIM, and is never swept by
        :func:`_apply_bindings`'s residual sweep (it doesn't match the
        placeholder pattern at all — verified structurally, not
        assumed).

        Post cloud-egress-PII redesign: CORE placeholders come straight
        from the model's own anonymizer mapping (no code-side entity
        walk), so the mock explicitly classifies ``Millfield`` ->
        ``City_1`` — the model decision this test's fixture would need
        in production.
        """
        from tests._sota_flow import run_sota_stages

        graph = _make_graph(
            [("speaker0", "lives_in", "Millfield")],
            entities=[
                Entity(name="speaker0", entity_type="person", speaker_id="speaker0"),
                Entity(name="Millfield", entity_type="place"),
            ],
        )
        anon_facts = [{"subject": "speaker0", "predicate": "lives_in", "object": "City_1"}]
        enriched_anon = list(anon_facts)
        mapping = {"Millfield": "City_1"}
        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.cloud_egress.anonymize_with_local_model",
                return_value=(mapping, "anonymized transcript", ""),
            ),
            patch(
                "paramem.graph.extractor._filter_with_sota",
                return_value=(enriched_anon, None, {}, None, {}),
            ),
        ):
            result = run_sota_stages(
                graph,
                "speaker0 lives in Millfield",
                None,
                None,
                speaker_id="speaker0",
                correction_entity_types=set(),
                scrub={"person name", "physical address"},
            )
        assert len(result.relations) == 1
        assert result.relations[0].subject == "speaker0"
        assert result.relations[0].object == "Millfield"
        assert "residual_dropped_facts" not in result.diagnostics
        assert "sota_enrichment_rejected" not in result.diagnostics

    def test_anchor_independent_of_speaker_relation_presence(self):
        """The anchor holds even in a session with NO speaker
        entity/relation at all (the protocol-constant case): the
        pipeline must not require a speaker fact to function correctly —
        nothing about the anonymizer/deanon machinery depends on the
        speaker being referenced THIS session."""
        from tests._sota_flow import run_sota_stages

        graph = _make_graph(
            [("Acme", "located_in", "Millfield")],
            entities=[
                Entity(name="Acme", entity_type="organization"),
                Entity(name="Millfield", entity_type="place"),
            ],
        )
        anon_facts = [{"subject": "Org_1", "predicate": "located_in", "object": "City_1"}]
        mapping = {"Acme": "Org_1", "Millfield": "City_1"}
        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.cloud_egress.anonymize_with_local_model",
                return_value=(mapping, "anonymized transcript", ""),
            ),
            patch(
                "paramem.graph.extractor._filter_with_sota",
                return_value=(list(anon_facts), None, {}, None, {}),
            ),
        ):
            result = run_sota_stages(
                graph,
                "Acme located in Millfield",
                None,
                None,
                speaker_id="speaker0",
                correction_entity_types=set(),
                scrub={"person name", "physical address", "organization"},
            )
        assert len(result.relations) == 1
        assert result.relations[0].subject == "Acme"
        assert result.relations[0].object == "Millfield"
        assert "sota_enrichment_rejected" not in result.diagnostics


class TestObservedDerivation:
    """``observed`` — CORE's legality domain for a SOTA cycle — is derived
    from the DECLARED token vocabulary intersected with the payload we
    actually rendered, not scraped out of the payload with a shape regex.

    A shape scrape reads whatever token-shaped text the payload happens to
    carry (``Boeing_747``, ``GPT_4``) and is blind to the table that
    declares what a token IS.
    """

    def test_observed_is_declared_tokens_present_in_the_rendered_payload(self):
        """Mutation: restore the ``PLACEHOLDER_TOKEN_RE.findall`` scrape over
        the facts + transcript -> the shape-like real name ``Boeing_747``
        (copied verbatim out of the transcript) enters ``observed`` as if it
        were a declared placeholder -> this test fails.
        """
        from tests._sota_flow import run_sota_stages

        graph = _make_graph(
            [("Alex", "lives_in", "Millfield")],
            entities=[
                Entity(name="Alex", entity_type="person"),
                Entity(name="Millfield", entity_type="place"),
                # Declared by the builder, but named in no fact and in no
                # transcript span -> declared but NOT observed.
                Entity(name="Dana", entity_type="person"),
            ],
        )
        anon_facts = [{"subject": "Person_1", "predicate": "lives_in", "object": "City_1"}]
        mapping = {"Alex": "Person_1", "Millfield": "City_1"}

        captured: list = []
        from paramem.graph import cloud_egress as _cloud_egress

        real_totality = _cloud_egress._check_mapping_totality

        def _spy(*args, **kwargs):
            if "observed" in kwargs:
                captured.append(kwargs["observed"])
            return real_totality(*args, **kwargs)

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.cloud_egress.anonymize_with_local_model",
                return_value=(mapping, "anonymized transcript", ""),
            ),
            patch("paramem.graph.cloud_egress._check_mapping_totality", side_effect=_spy),
            patch(
                "paramem.graph.extractor._filter_with_sota",
                return_value=(list(anon_facts), None, {}, None, {}),
            ),
        ):
            run_sota_stages(
                graph,
                "Alex lives in Millfield and flew on a Boeing_747",
                None,
                None,
                speaker_id="speaker0",
                correction_entity_types=set(),
                scrub={"person name", "physical address"},
            )

        assert captured, "the sota_enrich totality check must run with an observed scope"
        observed = captured[-1]
        # Declared AND in the payload.
        assert observed == {"Person_1", "City_1"}
        # A shape-like real name copied out of the transcript is NOT a token.
        assert "Boeing_747" not in observed
        # A declared token absent from the payload is not observed either
        # (Dana's minted placeholder).
        assert observed <= {"Person_1", "City_1"}


class TestFilterOpenaiCompatBoundaryErrors:
    """``_filter_openai_compat`` must treat a malformed 200 response as the
    SAME boundary condition its sibling ``_filter_anthropic`` already does
    (broad catch, return ``None``) — not let it escape as an uncaught
    exception.

    Regression this pins: ``consolidation.py``'s graph-tier chunk loop was
    narrowed from ``except Exception`` to ``except RuntimeError`` (the ONE
    genuinely-failing runtime leg being the local ``generate()``'s CUDA
    "device not ready" class). That premise is false for the OpenAI-
    compatible provider: a 200 response with a non-JSON body raises
    ``json.JSONDecodeError`` (a ``ValueError``) out of ``resp.json()``, and
    an unexpected JSON shape (``choices`` is ``null``) raises ``TypeError``
    out of the subscript chain. Neither is a ``RuntimeError``, so each would
    escape the narrowed handler and kill the WHOLE fold over what should
    cost one chunk.
    """

    @staticmethod
    def _client_returning(mock_response):
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = mock_response
        return mock_client

    def test_openai_compat_malformed_json_returns_none(self):
        """A 200 response whose body is not JSON at all (proxy error page,
        captive-portal interstitial, truncated stream) must return ``None``,
        not raise.

        Mutation: drop ``ValueError`` from the caught tuple -> ``resp.json()``'s
        ``json.JSONDecodeError`` escapes -> this test fails.
        """
        from paramem.graph.extractor import _filter_openai_compat

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.side_effect = json.JSONDecodeError("Expecting value", "<html>", 0)
        mock_client = self._client_returning(mock_response)

        with patch("httpx.Client", return_value=mock_client):
            result = _filter_openai_compat(
                "prompt", "test-key", "test-model", "groq", endpoint="https://example.test"
            )
        assert result is None

    def test_openai_compat_unexpected_shape_returns_none(self):
        """A 200 response whose JSON lacks the expected ``choices`` shape
        (``choices`` is ``null``) must return ``None``, not raise.

        Mutation: drop ``TypeError`` from the caught tuple -> subscripting
        ``None[0]`` raises ``TypeError`` uncaught -> this test fails.
        """
        from paramem.graph.extractor import _filter_openai_compat

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"choices": None}
        mock_client = self._client_returning(mock_response)

        with patch("httpx.Client", return_value=mock_client):
            result = _filter_openai_compat(
                "prompt", "test-key", "test-model", "groq", endpoint="https://example.test"
            )
        assert result is None
