"""Tests for the extraction pipeline — noise filter, JSON parsing."""

import json
from unittest.mock import MagicMock, patch

import pytest
from peft import PeftModel

from paramem.config.taxonomy import resolve_scrub_categories
from paramem.graph.extractor import PlausibilityVerdict, _extract_json_block
from paramem.graph.phase_trace import extraction_trace
from paramem.graph.schema import Entity, Relation, SessionGraph


def _scrub_categories(*hints: str):
    """Resolve raw ``sanitization.scrub`` hint strings into the
    ``scrub_categories`` tuple ``extract_graph``/``ExtractionConfig`` now
    require — these tests only care that a category is configured, not
    its exact resolved shape."""
    return resolve_scrub_categories(list(hints))


def _peft_model_mock() -> MagicMock:
    """``MagicMock(spec=PeftModel)`` -- passes the ``isinstance(model,
    PeftModel)`` precondition ``base_model_inference`` now enforces
    wherever a call site here reaches it (``anonymize_turn`` et al.).
    ``is_gradient_checkpointing`` defaults False (mirrors every prior
    caller's explicit ``model.is_gradient_checkpointing = False``) so
    ``grad_checkpointing_disabled`` takes its no-op branch."""
    model = MagicMock(spec=PeftModel)
    model.is_gradient_checkpointing = False
    return model


def _kept_verdict(facts: list[dict]) -> PlausibilityVerdict:
    """A :class:`PlausibilityVerdict` keeping every input fact —
    the mock-judge shape for tests exercising the plausibility call
    sites' plumbing (prompts_dir threading, provider dispatch, ...)
    without exercising the drop-set contract itself."""
    return PlausibilityVerdict(kept=list(facts), dropped=[], out_of_range=[])


def _verdict_dropping(facts: list[dict], predicates: set[str]) -> PlausibilityVerdict:
    """A :class:`PlausibilityVerdict` dropping every input fact whose
    predicate is in ``predicates`` — ``dropped`` is built from the facts
    actually removed, at their input-relative indices, so ``kept`` and
    ``dropped`` are never independently hand-typed into an unsatisfiable
    pair. Mirrors ``tests/test_extraction_alignment_smoke.py``'s helper
    of the same name."""
    kept = [f for f in facts if f.get("predicate") not in predicates]
    dropped = [
        {"index": i, "rule": "R1", "fact": f}
        for i, f in enumerate(facts)
        if f.get("predicate") in predicates
    ]
    return PlausibilityVerdict(kept=kept, dropped=dropped, out_of_range=[])


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


def _anonymize_contract(mapping, anon_transcript: str, raw: str, *, facts=None):
    """Build the :class:`~paramem.cloud.anonymize.AnonymizedContract` an
    ``anonymize()`` call would return for a given ``(mapping,
    anon_transcript, raw)`` triple — the pre-split combined-call shape
    most tests in this file were written against, and still the most
    convenient shape for a test to hand-author. ``mapping is None`` (the
    old parse-failure signal) maps to ``status="failed", failure="tagger"``
    — the nearest surviving failure member for a detector-side failure,
    now that there is no JSON envelope to fail parsing; otherwise
    ``status="ok"`` with the given forward table and rewrite.

    ``facts`` defaults to ``[]`` when not supplied — callers that need a
    realistic (non-empty) ``payload.facts`` use :func:`_anonymize_stub`
    instead, which reads the REAL call's own ``facts`` argument.
    """
    from paramem.cloud.anonymize import AnonymizedContract, failed_contract
    from paramem.cloud.placeholders import invert_forward_mapping

    if mapping is None:
        return failed_contract(failure="tagger", raw=raw, tagger_windows=1, model_calls=1)
    reverse = invert_forward_mapping(dict(mapping))
    return AnonymizedContract(
        status="ok",
        forward=dict(mapping),
        reverse=reverse,
        anon_transcript=anon_transcript,
        declared=frozenset(reverse.keys()),
        rekey_dropped=0,
        raw=raw,
        facts=list(facts) if facts is not None else [],
        tagger_windows=1,
        model_calls=1,
    )


def _anonymize_stub(mapping, anon_transcript: str, raw: str):
    """Build a ``side_effect`` callable for mocking ``anonymize()`` (or a
    caller-scoped bound name for it) that returns the same shape
    :func:`_anonymize_contract` does, EXCEPT ``payload.facts`` is
    populated from the REAL call's own ``facts`` argument (mirroring what
    ``anonymize()`` itself does: ``facts`` is the accepted, non-fail-closed
    subset of the input) — so downstream stages that consume
    ``payload.facts`` (via
    :func:`~paramem.cloud.placeholders.insert_placeholders`) see a
    realistic, non-empty array instead of the static ``[]``
    :func:`_anonymize_contract` returns for a ``return_value=`` mock,
    which cannot see the call's arguments.
    """

    def _stub(facts, *args, **kwargs):
        return _anonymize_contract(mapping, anon_transcript, raw, facts=facts)

    return _stub


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
        """String values containing `}` must not break the parser: it tracks
        quoted-string state so a `}` inside a quoted string is not counted
        as a closing brace."""
        text = 'Sure, here: {"facts": [{"object": "Code: }"}]} trailing'
        result = json.loads(_extract_json_block(text))
        assert result["facts"][0]["object"] == "Code: }"

    def test_string_value_with_opening_brace(self):
        """The same holds for `{` inside a string value (e.g. anonymizer
        placeholder forms): it must not inflate the parser's brace depth."""
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
        """A truncated outer envelope (e.g. cut at max_tokens mid-relation)
        must raise rather than silently matching the first inner
        sub-object: a left-to-right fall-through would return the inner
        entity dict, ``_normalize_extraction`` would not find
        ``entities``/``relations`` keys, and the ``SessionGraph`` would end
        up empty — masking the truncation instead of surfacing it.
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
    """Local Mistral occasionally emits unexpected JSON shapes (a bare
    list of facts instead of {"entities": ..., "relations": ...}). Such a
    shape is rewrapped as a relations payload rather than reaching
    `data["session_id"] = ...`, which requires `data` to be a dict."""

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


class TestPlausibilityDropSetRuleKeyed:
    """The plausibility judge emits ``{"drop": {"R1": [<index>, ...], ...}}``
    — a map from rule id to the indices that rule drops, or ``{"drop": {}}``
    when nothing matched. ``_apply_drop_set`` parses that output and returns
    a ``PlausibilityVerdict`` (``kept``/``dropped``/``out_of_range``/
    ``unattributed``).

    An index that does not sit in a list under one of the rule keys
    ``R1``-``R6`` is not a drop — it is counted in ``unattributed`` and the
    fact stays kept. This is the robustness contract for the other parseable
    drop-set shapes (a bare index array, the ``{"index", "rule"}`` annotated
    form): a parseable verdict in any of these shapes is *counted*, not
    turned into a fail-open trigger.
    """

    def _facts(self, n: int) -> list[dict]:
        return [{"subject": f"S{i}", "predicate": "p", "object": f"O{i}"} for i in range(n)]

    def test_rule_keyed_happy_path(self):
        from paramem.graph.extractor import _apply_drop_set

        facts = self._facts(5)
        out = _apply_drop_set(facts, '{"drop": {"R1": [3], "R4": [0]}}')
        assert out is not None
        assert [f["subject"] for f in out.kept] == ["S1", "S2", "S4"]
        assert {(d["index"], d["rule"]) for d in out.dropped} == {(0, "R4"), (3, "R1")}
        assert out.out_of_range == []
        assert out.unattributed == 0

    def test_empty_rule_map_keeps_all_facts(self):
        """``{"drop": {}}`` is the prompt-defined "clean input" output —
        the judge found no rule matches; every fact survives."""
        from paramem.graph.extractor import _apply_drop_set

        facts = self._facts(4)
        out = _apply_drop_set(facts, '{"drop": {}}')
        assert out.kept == facts
        assert out.dropped == []
        assert out.out_of_range == []
        assert out.unattributed == 0

    def test_duplicate_index_across_two_rules_lowest_wins(self):
        from paramem.graph.extractor import _apply_drop_set

        facts = self._facts(3)
        out = _apply_drop_set(facts, '{"drop": {"R4": [1], "R1": [1]}}')
        assert out is not None
        assert [f["subject"] for f in out.kept] == ["S0", "S2"]
        assert out.dropped == [{"index": 1, "rule": "R1", "fact": facts[1]}]

    def test_unknown_rule_key_is_unattributed_fact_kept(self):
        from paramem.graph.extractor import _apply_drop_set

        facts = self._facts(3)
        out = _apply_drop_set(facts, '{"drop": {"R9": [1]}}')
        assert out is not None
        assert out.kept == facts  # nothing dropped — unattributed, not applied
        assert out.dropped == []
        assert out.unattributed == 1

    def test_annotated_list_shape_counts_unattributed_and_keeps_facts(self):
        """The ``[{"index": N, "rule": "Rk"}]`` per-entry annotated form is
        a successful parse with zero drops, not a fail-open trigger. Every
        element of the list is one claim the judge made without a rule, so
        two entries count two unattributed claims and every fact stays
        kept."""
        from paramem.graph.extractor import _apply_drop_set

        facts = self._facts(3)
        out = _apply_drop_set(facts, '{"drop": [{"index": 1, "rule": "R1"}, {"index": 2}]}')
        assert out is not None
        assert out.kept == facts
        assert out.dropped == []
        assert out.unattributed == 2

    def test_bare_array_shape_counts_unattributed_and_keeps_facts(self):
        """The bare-index-array ``{"drop": [1]}`` shape: the one parseable
        int inside is one unattributed claim, zero drops applied."""
        from paramem.graph.extractor import _apply_drop_set

        facts = self._facts(3)
        out = _apply_drop_set(facts, '{"drop": [1]}')
        assert out is not None
        assert out.kept == facts
        assert out.dropped == []
        assert out.unattributed == 1

    def test_out_of_range_under_a_rule_recorded_with_rule(self):
        """A bad index shouldn't void an otherwise-valid drop set — it
        lands in ``out_of_range`` (never applied), still carrying the
        rule that cited it."""
        from paramem.graph.extractor import _apply_drop_set

        facts = self._facts(2)
        out = _apply_drop_set(facts, '{"drop": {"R2": [7]}}')
        assert out is not None
        assert out.kept == facts
        assert out.dropped == []
        assert out.out_of_range == [{"index": 7, "input_count": 2, "rule": "R2"}]
        assert out.unattributed == 0

    def test_non_int_elements_are_unattributed(self):
        """Stray non-int entries under a valid rule key don't void the
        rule's other in-range int drops — each is counted separately as
        unattributed (booleans excluded from the int check)."""
        from paramem.graph.extractor import _apply_drop_set

        facts = self._facts(3)
        out = _apply_drop_set(facts, '{"drop": {"R1": [1, "junk", null, true]}}')
        assert out is not None
        assert [f["subject"] for f in out.kept] == ["S0", "S2"]
        assert out.unattributed == 3  # "junk", null, true

    def test_dropped_rule_is_always_a_string(self):
        """Every drop record's rule is the non-empty rule-key string it was
        found under — the rule-keyed contract never produces a ``rule:
        None`` value."""
        from paramem.graph.extractor import _apply_drop_set

        facts = self._facts(2)
        out = _apply_drop_set(facts, '{"drop": {"R6": [0]}}')
        assert out is not None
        assert len(out.dropped) == 1
        assert isinstance(out.dropped[0]["rule"], str)
        assert out.dropped[0]["rule"] == "R6"

    def test_non_dict_drop_scalar_counts_one_unattributed(self):
        """A bare scalar (or a string) under ``"drop"`` is not itself
        enumerable — it counts as exactly one unattributed claim, zero
        drops applied, the fact kept."""
        from paramem.graph.extractor import _apply_drop_set

        facts = self._facts(2)
        for raw in ('{"drop": 5}', '{"drop": "R1"}'):
            out = _apply_drop_set(facts, raw)
            assert out is not None, raw
            assert out.kept == facts
            assert out.dropped == []
            assert out.unattributed == 1

    def test_known_rule_key_with_non_list_value_is_unattributed(self):
        """A known rule key whose value is not a list (e.g. a bare int)
        never reaches the per-element int loop — the whole value is one
        unattributed claim, zero drops applied."""
        from paramem.graph.extractor import _apply_drop_set

        facts = self._facts(2)
        out = _apply_drop_set(facts, '{"drop": {"R1": 3}}')
        assert out is not None
        assert out.kept == facts
        assert out.dropped == []
        assert out.unattributed == 1

    def test_missing_drop_key_returns_none(self):
        """A structurally valid envelope under a different key (no
        ``"drop"``) is not this judge's output at all — the caller
        fail-opens."""
        from paramem.graph.extractor import _apply_drop_set

        facts = self._facts(2)
        assert _apply_drop_set(facts, '{"facts": []}') is None

    def test_top_level_list_returns_none(self):
        """A bare top-level JSON array is not a dict, so it can never carry
        a ``"drop"`` key — parse failure, caller fail-opens."""
        from paramem.graph.extractor import _apply_drop_set

        facts = self._facts(6)
        assert _apply_drop_set(facts, "[0, 2, 5]") is None

    def test_negative_index_under_a_rule_is_out_of_range(self):
        """A negative index is outside ``[0, n_facts)`` like any other
        out-of-range value — recorded, never applied, still carrying the
        rule that cited it."""
        from paramem.graph.extractor import _apply_drop_set

        facts = self._facts(3)
        out = _apply_drop_set(facts, '{"drop": {"R3": [-1]}}')
        assert out is not None
        assert out.kept == facts
        assert out.dropped == []
        assert out.out_of_range == [{"index": -1, "input_count": 3, "rule": "R3"}]

    def test_duplicate_index_within_one_rule_counted_once(self):
        """The same index repeated under one rule is one drop, not two —
        ``rules`` is keyed by index."""
        from paramem.graph.extractor import _apply_drop_set

        facts = self._facts(3)
        out = _apply_drop_set(facts, '{"drop": {"R1": [1, 1]}}')
        assert out is not None
        assert [f["subject"] for f in out.kept] == ["S0", "S2"]
        assert out.dropped == [{"index": 1, "rule": "R1", "fact": facts[1]}]

    def test_empty_list_under_a_rule_counts_zero_unattributed(self):
        """An enumerable-but-empty value (``{"R9": []}``) has nothing to
        attribute — zero unattributed claims, zero drops, regardless of
        whether the key itself is a recognized rule."""
        from paramem.graph.extractor import _apply_drop_set

        facts = self._facts(2)
        out = _apply_drop_set(facts, '{"drop": {"R9": []}}')
        assert out is not None
        assert out.kept == facts
        assert out.dropped == []
        assert out.unattributed == 0

    def test_unattributed_claims_emit_a_counts_only_warning(self, caplog):
        """A non-zero unattributed count is surfaced as a warning
        mirroring the sibling out-of-range warning — counts only, never
        fact content."""
        import logging

        from paramem.graph.extractor import _apply_drop_set

        facts = self._facts(3)
        with caplog.at_level(logging.WARNING, logger="paramem.graph.extractor"):
            out = _apply_drop_set(facts, '{"drop": {"R9": [1]}}')
        assert out is not None
        assert out.unattributed == 1
        warnings = [r for r in caplog.records if "unattributed" in r.getMessage().lower()]
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert "1" in message
        assert str(len(facts)) in message
        for fact in facts:
            assert fact["subject"] not in message
            assert fact["object"] not in message

    def test_fenced_output_is_unwrapped(self):
        """Models often wrap structured output in ```json``` fences.
        The shared envelope-finder strips them."""
        from paramem.graph.extractor import _apply_drop_set

        facts = self._facts(4)
        raw = '```json\n{"drop": {"R1": [2]}}\n```'
        out = _apply_drop_set(facts, raw)
        assert out is not None
        assert [f["subject"] for f in out.kept] == ["S0", "S1", "S3"]

    def test_malformed_output_returns_none(self):
        """Parse failure must return ``None`` — caller fail-opens by
        keeping all input facts."""
        from paramem.graph.extractor import _apply_drop_set

        facts = self._facts(3)
        assert _apply_drop_set(facts, "I cannot process this request.") is None
        assert _apply_drop_set(facts, "{not_valid_json") is None

    def test_none_input_returns_none(self):
        from paramem.graph.extractor import _apply_drop_set

        assert _apply_drop_set([], None) is None
        assert _apply_drop_set([{"subject": "S"}], None) is None


class TestEnrichmentDelta:
    """``_parse_enrichment_delta`` and ``_apply_enrichment_delta`` are the
    Cloud enrichment counterpart of the plausibility drop-set helpers.
    The judge emits a small ``{"add": [...], "modify": [...], "drop":
    [...], "bindings": {...}}`` envelope; every key is optional.  The
    parser is permissive about wrapping (markdown fences / inline-code /
    prose preamble) via the shared envelope finder and returns an
    :class:`~paramem.graph.extractor.EnrichmentDelta`.  The applier
    composes modify → drop → add and reconstructs ``updated_transcript``
    locally from ``bindings`` + ``anon_transcript`` (no transcript echo
    on the wire).

    ``scope=None`` throughout this class — the direct/unit-test sentinel
    under which nothing is resolvability-gated and every action applies
    verbatim (mirroring ``_apply_bindings``'s own ``observed=None``
    convention); these tests pin the STRUCTURAL apply mechanics, not
    per-triple rejection (see ``TestBindingTotalityRejection`` for that).
    """

    @staticmethod
    def _facts(n: int) -> list[dict]:
        return [{"subject": f"S{i}", "predicate": "p", "object": f"O{i}"} for i in range(n)]

    @staticmethod
    def _apply(facts, raw, anon_transcript=None):
        """Parse ``raw`` then apply — ``_parse_enrichment_delta`` parses the
        raw model output into a delta, and ``_apply_enrichment_delta``
        applies that delta to ``facts``."""
        from paramem.graph.extractor import _apply_enrichment_delta, _parse_enrichment_delta

        delta = _parse_enrichment_delta(raw, len(facts))
        assert delta is not None, f"expected a parseable delta for {raw!r}"
        return _apply_enrichment_delta(facts, delta, None, anon_transcript)

    def test_empty_envelope_is_noop(self):
        """``{}`` — model emitted nothing to do.  Surviving facts equal
        input; transcript unchanged; no bindings."""
        facts = self._facts(3)
        out, transcript, report = self._apply(facts, "{}", "hello")
        assert out == facts
        assert transcript == "hello"
        assert report["bindings_count"] == 0

    def test_drop_only(self):
        """Pure subtractive delta — same shape as a plausibility output;
        applier still works (drop is shared between protocols)."""
        facts = self._facts(4)
        out, _, _ = self._apply(facts, '{"drop": [1, 3]}', None)
        assert out is not None
        assert [f["subject"] for f in out] == ["S0", "S2"]

    def test_add_only(self):
        """Append-only — coreference resolution case."""
        facts = self._facts(2)
        raw = (
            '{"add": [{"subject": "Person_1", "predicate": "married_to",'
            ' "object": "Person_2", "relation_type": "social", "confidence": 0.9}]}'
        )
        out, _, _ = self._apply(facts, raw, None)
        assert out is not None
        assert len(out) == 3
        assert out[2]["predicate"] == "married_to"

    def test_modify_partial_field_update(self):
        """Synonym normalization — replace ``employed_by`` with ``worked_for``
        on a single indexed fact."""
        facts = [
            {"subject": "Alex", "predicate": "employed_by", "object": "Acme"},
            {"subject": "Alex", "predicate": "lives_in", "object": "Berlin"},
        ]
        raw = '{"modify": [{"index": 0, "fields": {"predicate": "worked_for"}}]}'
        out, _, _ = self._apply(facts, raw, None)
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
        facts = [
            {"subject": "P", "predicate": "likes", "object": "hiking and cooking"},
            {"subject": "P", "predicate": "lives_in", "object": "Berlin"},
        ]
        raw = (
            '{"drop": [0],'
            ' "add": [{"subject":"P","predicate":"likes","object":"hiking"},'
            ' {"subject":"P","predicate":"likes","object":"cooking"}]}'
        )
        out, _, _ = self._apply(facts, raw, None)
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
        facts = self._facts(4)
        raw = (
            '{"modify": [{"index": 0, "fields": {"object": "O0_modified"}}],'
            ' "drop": [2],'
            ' "add": [{"subject":"S_new","predicate":"p","object":"O_new"}]}'
        )
        out, _, _ = self._apply(facts, raw, None)
        assert out is not None
        # S0 modified, S2 dropped, S_new appended → [S0, S1, S3, S_new]
        subjects = [f["subject"] for f in out]
        assert subjects == ["S0", "S1", "S3", "S_new"]
        assert out[0]["object"] == "O0_modified"

    def test_bindings_reconstruct_transcript_longest_first(self):
        """Reconstruction must replace longest spans first so a longer
        span wins over a shorter one that would otherwise consume part
        of it."""
        facts: list[dict] = []
        anon = "Person_1 was a Senior Software Engineer at Org_1."
        # Both bindings share the substring "Software Engineer".  Without
        # longest-first ordering, "Software Engineer" would replace first
        # and corrupt the longer span.
        raw = '{"bindings": {"Role_1": "Senior Software Engineer", "Role_2": "Software Engineer"}}'
        _, transcript, report = self._apply(facts, raw, anon)
        assert "{Role_1}" in transcript
        # "Software Engineer" should not survive because it was inside
        # the longer span that got replaced first.
        assert "Software Engineer" not in transcript
        # Role_2's span no longer appears, so its placeholder isn't
        # written into the transcript — that's expected, the binding
        # just sits unused (never referenced by any fact — with no
        # ``add``/``modify`` at all here, neither binding is "rejected",
        # so both are counted).
        assert report["bindings_count"] == 2

    def test_bindings_replace_all_occurrences(self):
        """Entities mentioned more than once in the transcript get one
        placeholder consistently — every occurrence replaced."""
        anon = "Person_1 led Event. Later, Person_2 joined Event."
        raw = '{"bindings": {"Event_1": "Event"}}'
        _, transcript, _ = self._apply([], raw, anon)
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
        facts = self._facts(2)
        raw = '```json\n{"drop": [0]}\n```'
        out, _, _ = self._apply(facts, raw, None)
        assert out is not None
        assert [f["subject"] for f in out] == ["S1"]

    def test_legacy_new_entity_bindings_alias(self):
        """``new_entity_bindings`` is accepted as a synonym of
        ``bindings`` so legacy-shape responses don't lose the binding
        payload silently."""
        anon = "Person_1 led the agile transformation initiative."
        raw = '{"new_entity_bindings": {"Event_1": "the agile transformation initiative"}}'
        _, transcript, report = self._apply([], raw, anon)
        assert report["bindings_count"] == 1
        assert "{Event_1}" in transcript

    def test_inverted_binding_is_corrected_not_passed_through(self):
        """An inverted binding (key = real text, value = placeholder) is
        corrected to canonical ``{placeholder: real_text}`` direction
        rather than passed straight into the substitution map. Confirmed by
        transcript reconstruction: the real-text span is replaced by the
        placeholder, which only happens when the binding resolves in the
        right direction."""
        anon = "Person_1 works at Acme."
        raw = '{"bindings": {"Acme": "Org_9"}}'
        _, transcript, report = self._apply([], raw, anon)
        assert report["bindings_count"] == 1
        assert "{Org_9}" in transcript
        assert "Acme" not in transcript.replace("{Org_9}", "")

    def test_binding_both_sides_shaped_ties_to_declared_side(self):
        """A binding where BOTH sides happen to be placeholder-shaped
        (e.g. a real-world name like `Person_2` or `GPT_4`) is not
        ambiguous — the caller's declared `placeholder_side="key"` breaks
        the tie, so the binding is kept as-is rather than the whole
        delta losing the entry."""
        raw = '{"bindings": {"Org_9": "Person_2"}}'
        _, _, report = self._apply([], raw, "text")
        assert report["bindings_count"] == 1

    def test_ambiguous_binding_neither_shaped_is_dropped(self):
        """A binding where NEITHER side is placeholder-shaped is not a
        real cloud mint binding and is dropped rather than accepted
        verbatim."""
        raw = '{"bindings": {"my company": "Acme Corp"}}'
        _, _, report = self._apply([], raw, "text")
        assert report["bindings_count"] == 0

    def test_out_of_range_modify_skipped(self):
        """Modify index outside ``[0, n_facts)`` is dropped with a
        warning, not failed — single bad index shouldn't void the
        whole delta."""
        facts = self._facts(2)
        raw = '{"modify": [{"index": 99, "fields": {"object": "X"}}]}'
        out, _, _ = self._apply(facts, raw, None)
        assert out is not None
        assert out == facts  # nothing applied

    def test_out_of_range_drop_skipped(self):
        facts = self._facts(2)
        raw = '{"drop": [99]}'
        out, _, _ = self._apply(facts, raw, None)
        assert out is not None
        assert out == facts

    def test_modify_with_non_dict_fields_skipped(self):
        facts = self._facts(2)
        raw = '{"modify": [{"index": 0, "fields": "not a dict"}]}'
        out, _, _ = self._apply(facts, raw, None)
        assert out is not None
        assert out == facts

    def test_add_entries_must_be_dicts(self):
        """Non-dict entries in ``add`` are skipped, not failed."""
        facts = self._facts(1)
        raw = '{"add": ["not a fact", null, {"subject":"X","predicate":"p","object":"Y"}]}'
        out, _, _ = self._apply(facts, raw, None)
        assert out is not None
        assert len(out) == 2  # 1 input + 1 valid add
        assert out[1]["subject"] == "X"

    def test_add_entry_strips_non_fact_fields(self):
        """An ``add`` entry carrying a non-fact key (``evidence``)
        alongside the fact proper has that key stripped at the parse
        boundary, so it never enters ``enriched_anon`` (and therefore
        can never sink a valid fact at the residual sweep downstream).
        The fact itself is kept, only the extra key is dropped."""
        raw = (
            '{"add": [{"subject": "Person_1", "predicate": "works_at",'
            ' "object": "Org_1", "relation_type": "factual", "confidence": 0.9,'
            ' "evidence": "Person_1 said they work at Org_1"}]}'
        )
        out, _, _ = self._apply([], raw, None)
        assert out is not None
        assert len(out) == 1
        assert "evidence" not in out[0]
        assert out[0]["subject"] == "Person_1"
        assert out[0]["object"] == "Org_1"

    def test_modify_fields_strips_non_fact_fields(self):
        """A ``modify`` entry's ``fields`` dict is
        restricted the same way: ``relation_type``/``confidence``
        updates apply normally, a stray ``evidence`` key does not."""
        facts = [{"subject": "Alex", "predicate": "employed_by", "object": "Acme"}]
        raw = (
            '{"modify": [{"index": 0, "fields": {"predicate": "worked_for",'
            ' "evidence": "she confirmed this"}}]}'
        )
        out, _, _ = self._apply(facts, raw, None)
        assert out is not None
        assert out[0]["predicate"] == "worked_for"
        assert "evidence" not in out[0]

    def test_prose_index_references_before_envelope_still_parse(self):
        """Reasoning prose that names input facts by bracketed index
        (``[2]``, ``[0] and [1]``) may precede the delta envelope. Those
        references are valid bare-int JSON arrays, so the shared envelope
        finder must not mistake one for the delta — it must continue
        scanning to the complete, well-formed envelope further down the
        same response."""
        facts = self._facts(3)
        raw = (
            "I need to analyze the extracted facts.\n"
            "1. Fact [2] is a compound — split it\n"
            "2. Facts [0] and [1] are symmetric duplicates — drop [1]\n\n"
            '{"add": [], "modify": [], "drop": [1], "bindings": {}}'
        )
        out, _, _ = self._apply(facts, raw, None)
        assert out is not None
        assert [f["subject"] for f in out] == ["S0", "S2"]

    def test_malformed_envelope_returns_none(self):
        """Caller fail-opens — the PARSER (not the applier, which is never
        even called) returns ``None`` so the caller (the ``enrich`` stage)
        treats it as a failed ``cloud_enrich`` phase."""
        from paramem.graph.extractor import _parse_enrichment_delta

        facts = self._facts(2)
        delta = _parse_enrichment_delta("I cannot process this.", len(facts))
        assert delta is None

    def test_none_raw_returns_none(self):
        from paramem.graph.extractor import _parse_enrichment_delta

        delta = _parse_enrichment_delta(None, len(self._facts(1)))
        assert delta is None

    def test_null_keys_treated_as_empty(self):
        """Model emits ``"add": null`` instead of ``[]`` — must not crash."""
        facts = self._facts(2)
        raw = '{"add": null, "modify": null, "drop": null, "bindings": null}'
        out, transcript, report = self._apply(facts, raw, "anon")
        assert out == facts
        assert transcript == "anon"
        assert report["bindings_count"] == 0

    def test_bindings_with_missing_span_in_transcript_skipped(self):
        """Hallucinated binding (span not in transcript) leaves the
        transcript untouched.  No crash, no replacement."""
        anon = "Person_1 said hello."
        raw = '{"bindings": {"Event_1": "this span is not here"}}'
        _, transcript, report = self._apply([], raw, anon)
        assert transcript == anon
        assert report["bindings_count"] == 1

    def test_none_transcript_returns_none_transcript(self):
        _, transcript, _ = self._apply([], '{"add": []}', None)
        assert transcript is None


class TestApplyEnrichmentDeltaResolvability:
    """``_apply_enrichment_delta``'s per-triple resolvability contract —
    unlike ``TestEnrichmentDelta`` (structural mechanics only,
    ``scope=None``), these tests exercise a
    REAL :class:`~paramem.cloud.deanonymize.CloudScope`, so an ``add``/
    ``modify`` referencing a token outside the resolvable domain is
    actually rejected/reverted.
    """

    @staticmethod
    def _scope(reverse: dict[str, str], sent: str, cloud_bindings: dict | None = None):
        from paramem.cloud.anonymize import AnonymizedContract
        from paramem.cloud.deanonymize import CloudScope

        payload = AnonymizedContract(
            status="ok",
            forward={v: k for k, v in reverse.items()},
            reverse=reverse,
            anon_transcript=sent,
            declared=frozenset(reverse),
            rekey_dropped=0,
            raw="",
        )
        return CloudScope.response(payload, cloud_bindings=cloud_bindings, sent=(sent,))

    def test_unresolvable_add_is_dropped_not_the_whole_delta(self):
        """An ``add`` referencing a token never declared anywhere is
        dropped individually — the local fact (untouched by the delta)
        survives."""
        from paramem.graph.extractor import EnrichmentDelta, _apply_enrichment_delta

        scope = self._scope({"Person_1": "Alex"}, "Person_1 lives in Berlin.")
        facts = [{"subject": "Person_1", "predicate": "lives_in", "object": "Berlin"}]
        delta = EnrichmentDelta(
            add=[
                {
                    "subject": "Person_2",
                    "predicate": "married_to",
                    "object": "Person_3",
                    "relation_type": "social",
                    "confidence": 0.9,
                }
            ],
            modify=[],
            drop=set(),
            bindings={},
        )
        out, _, report = _apply_enrichment_delta(facts, delta, scope, "Person_1 lives in Berlin.")
        assert out == facts
        assert report["rejected_adds"] == 1
        assert report["rejected_tokens"] == ["Person_2", "Person_3"]

    def test_unresolvable_modify_reverts_to_original_fact(self):
        """A ``modify`` whose result carries an unresolvable token is
        DISCARDED — the fact reverts to its untouched original, it is not
        dropped from the fact list."""
        from paramem.graph.extractor import EnrichmentDelta, _apply_enrichment_delta

        scope = self._scope({"Person_1": "Alex"}, "Person_1 works at Acme.")
        facts = [{"subject": "Person_1", "predicate": "works_at", "object": "Acme"}]
        delta = EnrichmentDelta(
            add=[],
            modify=[(0, {"object": "Person_9"})],  # Person_9 never declared/bound
            drop=set(),
            bindings={},
        )
        out, _, report = _apply_enrichment_delta(facts, delta, scope, "Person_1 works at Acme.")
        # Reverted to the exact original — not dropped.
        assert out == facts
        assert report["reverted_modifies"] == 1
        assert report["rejected_tokens"] == ["Person_9"]

    def test_drop_with_rejection_recorded_when_both_occur(self):
        """The owner's "measure first" diagnostic:
        ``drop_with_rejection`` is ``True`` exactly when a delta carries
        BOTH a non-empty ``drop`` and at least one rejected ``add``/
        ``modify`` — the drop is still honored unconditionally (spec rule
        3 — revert every drop on any rejection — is NOT implemented)."""
        from paramem.graph.extractor import EnrichmentDelta, _apply_enrichment_delta

        scope = self._scope({"Person_1": "Alex"}, "Person_1 lives in Berlin.")
        facts = [{"subject": "Person_1", "predicate": "lives_in", "object": "Berlin"}]
        delta = EnrichmentDelta(
            add=[
                {
                    "subject": "Person_2",
                    "predicate": "profession",
                    "object": "teacher",
                    "relation_type": "factual",
                    "confidence": 0.9,
                }
            ],
            modify=[],
            drop={0},
            bindings={},
        )
        out, _, report = _apply_enrichment_delta(facts, delta, scope, "Person_1 lives in Berlin.")
        # drop honored unconditionally -> the only local fact is gone;
        # the add was rejected -> nothing replaces it.
        assert out == []
        assert report["drop_with_rejection"] is True

    def test_drop_without_any_rejection_is_not_flagged(self):
        """A clean drop (no co-occurring rejection) leaves
        ``drop_with_rejection`` ``False`` — the flag measures the
        CO-OCCURRENCE, not the mere presence of a drop."""
        from paramem.graph.extractor import EnrichmentDelta, _apply_enrichment_delta

        scope = self._scope({"Person_1": "Alex"}, "Person_1 lives in Berlin.")
        facts = [{"subject": "Person_1", "predicate": "lives_in", "object": "Berlin"}]
        delta = EnrichmentDelta(add=[], modify=[], drop={0}, bindings={})
        out, _, report = _apply_enrichment_delta(facts, delta, scope, "Person_1 lives in Berlin.")
        assert out == []
        assert report["drop_with_rejection"] is False


class TestPipelineMaxTokensThreading:
    """Verify the single ``extraction_max_tokens`` config flows through the
    entire LLM pipeline (local extract → anonymize → cloud enrich → deanon →
    plausibility) instead of each stage carrying its own hardcoded budget."""

    def test_stage_context_carries_max_tokens_to_enrich(self):
        """``StageContext`` is the ``enrich`` stage's sole parameter
        surface and carries ``max_tokens`` through from ``extract_graph``."""
        import dataclasses

        from paramem.graph.flow import StageContext

        assert "max_tokens" in {f.name for f in dataclasses.fields(StageContext)}

    def test_extract_graph_default_matches_filter_default(self):
        """The single-budget invariant: extract_graph and the cloud-side
        filter calls must share the same default. Otherwise a user who
        sets only the loop-level config would get inconsistent budgets
        across stages."""
        import inspect

        from paramem.graph.extractor import _DEFAULT_FILTER_MAX_TOKENS
        from paramem.graph.flows import extract_graph

        default = inspect.signature(extract_graph).parameters["max_tokens"].default
        assert default == _DEFAULT_FILTER_MAX_TOKENS

    def test_fallback_plausibility_threads_max_tokens(self):
        """The all_dropped / anon_failed fallback path also accepts max_tokens
        so the whole pipeline runs on one budget — including degraded paths."""
        import inspect

        from paramem.graph.extractor import _fallback_plausibility_on_raw

        sig = inspect.signature(_fallback_plausibility_on_raw)
        assert "max_tokens" in sig.parameters


class TestPipelinePromptsDirThreading:
    """A ``prompts_dir`` override passed to ``extract_graph`` must reach
    every prompt load the ``anonymize``/``enrich`` stages perform, not
    just the anonymizer call ``anonymize_turn`` already
    wired.  Each stage is exercised through the real stage body itself
    (never by calling the downstream helper directly) so the assertion
    covers the exact call site that was silently dropping the override.
    """

    def test_cloud_enrich_receives_prompts_dir(self, tmp_path):
        """Stage 2 (cloud_enrich): ``request_enrichment`` had neither a
        ``prompts_dir`` parameter nor a forwarded value — the override never
        reached the enrichment prompt at all."""
        from tests._cloud_flow import enrichment_side_effect, run_cloud_stages

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

        def fake_request_enrichment(*args, **kwargs):
            captured.append(kwargs.get("prompts_dir"))
            return enrichment_side_effect(anon_facts)(*args, **kwargs)

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.stage_anonymize.anonymize",
                side_effect=_anonymize_stub(mapping, "anonymized transcript", ""),
            ),
            patch(
                "paramem.graph.stage_enrich.request_enrichment",
                side_effect=fake_request_enrichment,
            ),
        ):
            run_cloud_stages(
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
            f"request_enrichment must receive the caller's prompts_dir, got {captured!r}"
        )

    def test_anon_plausibility_receives_prompts_dir(self, tmp_path):
        """Stage 3a (anon_plausibility, cloud judge): ``request_plausibility``
        had neither a ``prompts_dir`` parameter nor a forwarded value."""
        from tests._cloud_flow import enrichment_side_effect, run_cloud_stages

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
            return _kept_verdict(facts), "raw"

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.stage_anonymize.anonymize",
                side_effect=_anonymize_stub(mapping, "anonymized transcript", ""),
            ),
            patch(
                "paramem.graph.stage_enrich.request_enrichment",
                side_effect=enrichment_side_effect(anon_facts),
            ),
            patch(
                "paramem.graph.stage_enrich.request_plausibility",
                side_effect=fake_plaus,
            ),
        ):
            run_cloud_stages(
                graph,
                "Alex lives in Millfield.",
                None,
                None,
                speaker_id="speaker0",
                plausibility_judge="anthropic",
                plausibility_stage="anon",
                correction_entity_types=set(),
                scrub={"person name"},
                prompts_dir=tmp_path,
            )

        assert captured == [tmp_path], (
            f"request_plausibility must receive the caller's prompts_dir, got {captured!r}"
        )

    def test_deanon_plausibility_receives_prompts_dir(self, tmp_path):
        """Stage 3d (deanon_plausibility, local judge): ``judge_plausibility``
        already accepted ``prompts_dir`` but the call site never passed it."""
        from tests._cloud_flow import enrichment_side_effect, run_cloud_stages

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
            return _kept_verdict(facts), ""

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.stage_anonymize.anonymize",
                side_effect=_anonymize_stub(mapping, "anonymized transcript", ""),
            ),
            patch(
                "paramem.graph.stage_enrich.request_enrichment",
                side_effect=enrichment_side_effect(anon_facts),
            ),
            patch(
                "paramem.graph.flows.judge_plausibility",
                side_effect=fake_local_plaus,
            ),
        ):
            run_cloud_stages(
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
            f"judge_plausibility must receive the caller's prompts_dir, got {captured!r}"
        )


class TestWaitForGpuReady:
    """Cover the WSL2 cloud-idle → local-LLM wake helper added after the
    May 2 production crash where a 62s cloud round-trip left the GPU
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


# --- Cloud Noise Filter ---


class TestCloudEnrichmentProvider:
    def test_filter_function_exists(self):
        from paramem.graph.extractor import request_enrichment

        assert callable(request_enrichment)

    def test_pipeline_enrichment_failure_fails_open(self):
        """Enrichment failure degrades the session, it does not fail the cycle.

        ``request_enrichment`` retries the cloud call; when every attempt
        misses a parseable delta it returns ``delta=None`` and the enrich
        stage fails OPEN — keeps the pre-enrichment facts, records
        ``cloud_enrichment_degraded``, and continues.  See
        :class:`TestCloudEnrichmentFailureModes` for the full rationale;
        a mis-shaped delta must never be raised as fatal.
        """
        from tests._cloud_flow import run_cloud_stages

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
                "paramem.graph.stage_anonymize.anonymize",
                side_effect=_anonymize_stub(mapping, "anonymized transcript", ""),
            ),
            patch(
                "paramem.graph.stage_enrich.request_enrichment",
                # ``delta=None`` -- every retry missed a parseable delta.
                return_value=(None, None, {"parse_path": "failed", "attempts": 3}),
            ),
        ):
            result = run_cloud_stages(
                graph,
                "transcript",
                None,
                None,
                speaker_id="speaker0",
                correction_entity_types=set(),
                scrub={"person name"},
            )
        assert any(r.predicate == "lives_in" for r in result.relations)
        assert result.diagnostics.get("cloud_enrichment_degraded") is not None

    def test_pipeline_enriched_facts_get_deanonymized(self):
        """Enrichment output flows through de-anonymization to real names."""
        from tests._cloud_flow import enrichment_side_effect, run_cloud_stages

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
                "paramem.graph.stage_anonymize.anonymize",
                side_effect=_anonymize_stub(mapping, "anonymized transcript", ""),
            ),
            patch(
                "paramem.graph.stage_enrich.request_enrichment",
                side_effect=enrichment_side_effect(enriched_anon),
            ),
        ):
            result = run_cloud_stages(
                graph,
                "transcript",
                None,
                None,
                speaker_id="speaker0",
                correction_entity_types=set(),
                scrub={"person name"},
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
        from tests._cloud_flow import enrichment_side_effect, run_cloud_stages

        transcript = "Alex lives in downtown Millfield with Alex's family"
        graph = _make_graph(
            [("Alex", "lives_in", "Millfield")],
            entities=[
                Entity(name="Alex", entity_type="person"),
                Entity(name="Millfield", entity_type="place"),
            ],
        )
        anon_facts = [{"subject": "Person_1", "predicate": "lives_in", "object": "City_1"}]
        # cloud produces composite strings with embedded placeholders
        enriched_anon = anon_facts + [
            {"subject": "Person_1's family", "predicate": "lives_in", "object": "City_1"},
            {"subject": "Person_1", "predicate": "lives_in", "object": "downtown City_1"},
        ]
        mapping = {"Alex": "Person_1", "Millfield": "City_1"}

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.stage_anonymize.anonymize",
                side_effect=_anonymize_stub(mapping, "anonymized transcript", ""),
            ),
            patch(
                "paramem.graph.stage_enrich.request_enrichment",
                side_effect=enrichment_side_effect(enriched_anon),
            ),
        ):
            result = run_cloud_stages(
                graph,
                transcript,
                None,
                None,
                speaker_id="speaker0",
                correction_entity_types=set(),
                scrub={"person name"},
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

    def test_normalize_anonymization_mapping_inverts_placeholder_keys(self):
        """Mapping with placeholder keys is inverted to {real: placeholder} canonical."""
        from paramem.cloud.placeholders import _normalize_anonymization_mapping

        wrong_direction = {"Person_1": "Alex", "City_1": "Millfield"}
        normalized, stats = _normalize_anonymization_mapping(wrong_direction)
        assert normalized == {"Alex": "Person_1", "Millfield": "City_1"}
        assert stats == {"inverted": 2, "dropped": 0, "dropped_entries": []}

    def test_normalize_anonymization_mapping_keeps_canonical(self):
        """Mapping already in {real: placeholder} canonical form passes through."""
        from paramem.cloud.placeholders import _normalize_anonymization_mapping

        canonical = {"Alex": "Person_1", "Millfield": "City_1"}
        normalized, stats = _normalize_anonymization_mapping(canonical)
        assert normalized == canonical
        assert stats == {"inverted": 0, "dropped": 0, "dropped_entries": []}

    def test_normalize_anonymization_mapping_empty(self):
        from paramem.cloud.placeholders import _normalize_anonymization_mapping

        normalized, stats = _normalize_anonymization_mapping({})
        assert normalized == {}
        assert stats == {"inverted": 0, "dropped": 0, "dropped_entries": []}

    def test_entity_type_to_prefix_closed_vocab_and_derivations(self):
        """Pin the contract for ``entity_type_to_prefix``: closed-vocabulary
        common types map via schema.yaml's ``anonymizer_type_to_prefix()``;
        everything else is PascalCase-joined; empty input falls back to
        ``Entity``."""
        from paramem.config.taxonomy import entity_type_to_prefix

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
        to proceed (cloud call, de-anonymization) rather than aborting.
        """
        from paramem.cloud.placeholders import _normalize_anonymization_mapping

        mixed = {
            "Alex": "Person_1",  # canonical
            "Person_2": "Millfield",  # inverted
        }
        out, stats = _normalize_anonymization_mapping(mixed)
        # Both pairs end up canonical: keys are real, values are placeholders.
        assert out == {"Alex": "Person_1", "Millfield": "Person_2"}
        assert stats == {"inverted": 1, "dropped": 0, "dropped_entries": []}

    def test_entity_correction_call_site_wired_into_pipeline(self):
        """Integration coverage for the entity_correction phase call site.

        Stubs ``paramem.graph.stage_enrich.correct_entity_surfaces`` (the name
        bound in the ``enrich`` module's own namespace via its top-of-file
        import) so this proves the WIRING inside the ``enrich`` stage
        (:func:`~paramem.graph.stage_enrich._stage_enrich`) — the stub is
        called exactly once and its ``"applied"`` list lands verbatim on
        ``graph.diagnostics["entity_corrections"]`` — without needing a
        live model. Reuses the same ``anonymize`` / ``request_enrichment``
        happy-path mocking pattern as the sibling tests in this class.
        """
        from tests._cloud_flow import enrichment_side_effect, run_cloud_stages

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
                "paramem.graph.stage_anonymize.anonymize",
                side_effect=_anonymize_stub(mapping, "anonymized transcript", ""),
            ),
            patch(
                "paramem.graph.stage_enrich.request_enrichment",
                side_effect=enrichment_side_effect(anon_facts),
            ),
            patch(
                "paramem.graph.stage_enrich.correct_entity_surfaces",
                side_effect=fake_correct_entity_surfaces,
            ),
        ):
            result = run_cloud_stages(
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


class TestContractCarriedFactsParity:
    """``stage_enrich``'s ``anon_facts`` under a session flow equals the
    direct derivation
    (``insert_placeholders(facts_from_relations(graph.relations),
    payload.forward)``) — pins the claim that nothing mutates
    ``graph.relations`` between the ``anonymize`` and ``enrich`` stages,
    so ``payload.facts`` (captured at anonymize time) stays byte-parity
    with a fresh render for the session tier's single-slice
    ``status == "ok"`` case.
    """

    def test_anon_facts_sent_to_cloud_matches_pre_change_derivation(self):
        from paramem.cloud.placeholders import insert_placeholders
        from paramem.graph.schema import facts_from_relations
        from tests._cloud_flow import run_cloud_stages

        graph = _make_graph(
            [("Alex", "lives_in", "Millfield"), ("Alex", "works_with", "Dana")],
            entities=[
                Entity(name="Alex", entity_type="person"),
                Entity(name="Millfield", entity_type="place"),
                Entity(name="Dana", entity_type="person"),
            ],
        )
        mapping = {"Alex": "Person_1", "Dana": "Person_2"}
        expected_forward = {"Alex": "Person_1", "Dana": "Person_2"}
        expected_anon_facts = insert_placeholders(
            facts_from_relations(graph.relations), expected_forward
        )

        captured: list[list[dict]] = []

        def _capture_request_enrichment(anon_facts, *args, **kwargs):
            captured.append(list(anon_facts))
            from paramem.graph.extractor import EnrichmentDelta

            return EnrichmentDelta(add=[], modify=[], drop=set(), bindings={}), "raw", {}

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.stage_anonymize.anonymize",
                side_effect=_anonymize_stub(mapping, "anonymized transcript", ""),
            ),
            patch(
                "paramem.graph.stage_enrich.request_enrichment",
                side_effect=_capture_request_enrichment,
            ),
        ):
            run_cloud_stages(
                graph,
                "Alex lives in Millfield and works with Dana.",
                None,
                None,
                speaker_id="speaker0",
                correction_entity_types=set(),
                plausibility_judge="off",
                scrub={"person name"},
            )

        assert captured, "expected request_enrichment to be called"
        assert captured[0] == expected_anon_facts

    def test_session_tier_opt_out_anon_facts_is_non_empty(self):
        """Session-tier counterpart: with ``scrub=set()``
        (operator opt-out), ``anonymize()``'s opt-out branch carries the
        input facts verbatim in ``payload.facts`` — a ``facts=[]`` opt-out
        would silently drop every fact from a payload the operator asked to
        egress unmasked — so ``stage_enrich``'s ``anon_facts`` derivation is
        non-empty, not ``[]``."""
        from tests._cloud_flow import run_cloud_stages

        graph = _make_graph(
            [("Alex", "lives_in", "Millfield")],
            entities=[
                Entity(name="Alex", entity_type="person"),
                Entity(name="Millfield", entity_type="place"),
            ],
        )
        captured: list[list[dict]] = []

        def _capture_request_enrichment(anon_facts, *args, **kwargs):
            captured.append(list(anon_facts))
            from paramem.graph.extractor import EnrichmentDelta

            return EnrichmentDelta(add=[], modify=[], drop=set(), bindings={}), "raw", {}

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.stage_enrich.request_enrichment",
                side_effect=_capture_request_enrichment,
            ),
        ):
            run_cloud_stages(
                graph,
                "Alex lives in Millfield.",
                None,
                None,
                speaker_id="speaker0",
                correction_entity_types=set(),
                plausibility_judge="off",
                scrub=set(),
            )

        assert captured, "expected request_enrichment to be called"
        assert captured[0] != [], "opt-out anon_facts must not be silently dropped"
        assert len(captured[0]) == 1
        assert captured[0][0]["subject"] == "Alex"


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
        — and cloud IS called (nothing skips or blocks the cycle over the
        case collision).

        Mutation: introduce any case-insensitive match over entity names
        (e.g. a case-insensitive substring/whole-word check on the
        anonymized payload, or make the substitution primitive
        case-insensitive) -> "electricity bill" is misread as a leak of
        ``Bill`` -> the cycle blocks/skips cloud, or the object is wrongly
        scrubbed -> this test fails.
        """
        from tests._cloud_flow import enrichment_side_effect, run_cloud_stages

        graph = _make_graph(
            [("Bill", "received", "electricity bill")],
            entities=[Entity(name="Bill", entity_type="person")],
        )
        mapping = {"Bill": "Person_1"}
        transcript = "Bill received the electricity bill yesterday."

        cloud_calls: list[list[dict]] = []

        def fake_cloud(facts, *args, **kwargs):
            cloud_calls.append(list(facts))
            return enrichment_side_effect(facts)(facts, *args, **kwargs)

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.stage_anonymize.anonymize",
                side_effect=_anonymize_stub(mapping, "anonymized transcript", ""),
            ),
            patch("paramem.graph.stage_enrich.request_enrichment", side_effect=fake_cloud),
        ):
            result = run_cloud_stages(
                graph,
                transcript,
                None,
                None,
                speaker_id="speaker0",
                correction_entity_types=set(),
                scrub={"person name"},
            )

        # cloud IS called — nothing blocks or skips the cycle.
        assert len(cloud_calls) == 1
        sent = cloud_calls[0][0]
        # "Bill" (the person) IS scrubbed to its placeholder in the cloud payload.
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
    ONLY stage a placeholder can still glue into a predicate is cloud's
    own returned delta. This pins the end-to-end behaviour of the
    deanon-stage predicate invariant (:func:`_apply_bindings`): a fact
    whose predicate carries a glued placeholder is dropped before it
    reaches ``graph.relations``.
    """

    def test_cloud_returned_glued_predicate_dropped_end_to_end(self):
        """A fact in cloud's *returned* delta whose predicate glues a
        declared placeholder onto a static prefix
        (``language_proficiency_Language_3``) never reaches
        ``graph.relations``, and the drop is recorded in diagnostics.

        Mutation: delete/narrow the deanon-stage predicate invariant in
        ``_apply_bindings`` -> the poisoned fact reaches
        ``graph.relations`` -> this test fails.
        """
        from tests._cloud_flow import enrichment_side_effect, run_cloud_stages

        graph = _make_graph(
            [("Alex", "works_at", "Acme")],
            entities=[Entity(name="Alex", entity_type="person")],
        )
        # "French" -> "Language_3" is an LLM-hint mapping entry the
        # deterministic builder merges in verbatim (never minted by
        # graph.entities, which only covers Alex here) — this is what
        # puts "Language_3" into the declared vocabulary cloud's returned
        # facts are checked against.
        mapping = {"Alex": "Person_1", "French": "Language_3"}
        transcript = "Alex works at Acme. Alex speaks French at an advanced level."

        # cloud's returned delta — NOT the anonymizer — carries the
        # poisoned predicate.  ``_apply_enrichment_delta``'s per-triple
        # orphan check (`_fact_orphans`/`_fact_tokens`) scans
        # subject/object only, so this fact sails through it untouched
        # and reaches the deanon-stage predicate invariant, which is
        # what this test pins.
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
                "paramem.graph.stage_anonymize.anonymize",
                side_effect=_anonymize_stub(mapping, "anonymized transcript", ""),
            ),
            patch(
                "paramem.graph.stage_enrich.request_enrichment",
                side_effect=enrichment_side_effect(enriched_anon),
            ),
        ):
            result = run_cloud_stages(
                graph,
                transcript,
                None,
                None,
                speaker_id="speaker0",
                correction_entity_types=set(),
                scrub={"person name"},
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
    """Unit tests for the state-machine de-anonymization helper. Binding
    knowledge arrives in cloud's response (``new_entity_bindings``); deanon
    is pure dict substitution against that mapping — no LLM call, no
    transcript reconstruction, no regex."""

    def test_substitutes_anonymizer_placeholders(self):
        """Bare anonymizer placeholders (Person_1, Org_1) substitute via
        the reverse mapping."""
        from paramem.cloud.placeholders import _apply_bindings

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
            facts, reverse, cloud_bindings={}
        )
        dropped = predicate_dropped + residual_dropped
        assert dropped == []
        assert kept[0]["subject"] == "Alice"
        assert kept[0]["object"] == "Acme"

    def test_substitutes_braced_cloud_bindings(self):
        """Cloud-introduced braced placeholders ({Event_1}) substitute via
        explicit bindings without needing transcript reconstruction."""
        from paramem.cloud.placeholders import _apply_bindings

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
        becomes `Acme Hungary`."""
        from paramem.cloud.placeholders import _apply_bindings

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
            facts, reverse, cloud_bindings={}
        )
        dropped = predicate_dropped + residual_dropped
        assert dropped == []
        assert kept[0]["object"] == "Acme Hungary"

    def test_drops_facts_with_unresolved_placeholders(self):
        """Facts whose subject/object retain a placeholder pattern after
        substitution get dropped (residual sweep). Causes: Cloud emitted a
        braced placeholder without including it in bindings, anonymizer
        leak, etc."""
        from paramem.cloud.placeholders import _apply_bindings

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
            facts, reverse, cloud_bindings={}
        )
        dropped = predicate_dropped + residual_dropped
        assert kept == []
        assert len(dropped) == 2

    def test_handles_apostrophes_at_word_boundary(self):
        """`Person_2's cousin` substitutes Person_2 cleanly without breaking
        on the apostrophe (existing _substitute_whole_words behaviour)."""
        from paramem.cloud.placeholders import _apply_bindings

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
            facts, reverse, cloud_bindings={}
        )
        dropped = predicate_dropped + residual_dropped
        assert dropped == []
        assert kept[0]["object"] == "Bob's cousin"

    def test_mixed_bare_and_braced_in_same_fact(self):
        """A single fact with both a bare anonymizer placeholder and a
        braced cloud placeholder substitutes both."""
        from paramem.cloud.placeholders import _apply_bindings

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
        from paramem.cloud.placeholders import _apply_bindings

        kept, predicate_dropped, residual_dropped = _apply_bindings([], {}, {})

        dropped = predicate_dropped + residual_dropped
        assert kept == []
        assert dropped == []

    def test_preserves_other_fact_fields(self):
        """relation_type, confidence, and any extra fields pass through."""
        from paramem.cloud.placeholders import _apply_bindings

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
        kept, _, _ = _apply_bindings(facts, reverse, cloud_bindings={})
        assert kept[0]["relation_type"] == "social"
        assert kept[0]["confidence"] == 0.7
        assert kept[0]["synthetic"] is False

    def test_minted_placeholder_round_trips_bare(self):
        """A cloud-minted placeholder emitted BARE (not braced, contra the
        prompt's contract) still round-trips via the union resolve map —
        today's two-channel design drops this because ``bare_map`` only
        ever contained ``reverse``."""
        from paramem.cloud.placeholders import _apply_bindings

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
        cloud_bindings = {"Event_1": "the quarterly retro"}
        kept, predicate_dropped, residual_dropped = _apply_bindings(facts, reverse, cloud_bindings)
        dropped = predicate_dropped + residual_dropped
        assert dropped == []
        assert kept[0]["object"] == "the quarterly retro"

    def test_minted_placeholder_round_trips_braced(self):
        """Braced-form minted placeholder resolves via the union of
        ``reverse`` and ``cloud_bindings``."""
        from paramem.cloud.placeholders import _apply_bindings

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
        cloud_bindings = {"Event_1": "the quarterly retro"}
        kept, predicate_dropped, residual_dropped = _apply_bindings(facts, reverse, cloud_bindings)
        dropped = predicate_dropped + residual_dropped
        assert dropped == []
        assert kept[0]["object"] == "the quarterly retro"

    def test_anonymizer_placeholder_wrongly_braced_still_resolves(self):
        """An anonymizer placeholder the cloud wrongly re-braced
        (contra the prompt's 'leave bare' contract) still resolves via
        the union — it is in ``reverse``, which is now tried in both
        the braced and bare pass."""
        from paramem.cloud.placeholders import _apply_bindings

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
            facts, reverse, cloud_bindings={}
        )
        dropped = predicate_dropped + residual_dropped
        assert dropped == []
        assert kept[0]["object"] == "Alex"

    def test_nested_binding_value_resolves_in_order(self):
        """A binding value containing a bare anonymizer placeholder
        (``"Senior Engineer at Org_1"``) resolves fully: braced pass
        expands ``{Role_1}`` to the value, bare pass then resolves the
        exposed ``Org_1`` from the SAME union map."""
        from paramem.cloud.placeholders import _apply_bindings

        facts = [
            {
                "subject": "Person_1",
                "predicate": "held_role",
                "object": "{Role_1}",
                "relation_type": "factual",
                "confidence": 1.0,
            },
        ]
        cloud_bindings = {"Role_1": "Senior Engineer at Org_1"}
        reverse = {"Person_1": "Alex", "Org_1": "Acme"}
        kept, predicate_dropped, residual_dropped = _apply_bindings(facts, reverse, cloud_bindings)
        dropped = predicate_dropped + residual_dropped
        assert dropped == []
        assert kept[0]["object"] == "Senior Engineer at Acme"

    def test_collision_reverse_wins_in_resolved_output(self):
        """When a key collides between ``cloud_bindings`` and ``reverse``
        with differing values, ``reverse`` wins (deterministic entity
        name over a freshly-minted cloud value) — the collision itself is
        surfaced by :func:`_binding_collisions`, not here."""
        from paramem.cloud.placeholders import _apply_bindings

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
        cloud_bindings = {"Org_1": "Wrong Corp"}
        kept, predicate_dropped, residual_dropped = _apply_bindings(facts, reverse, cloud_bindings)
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
        from paramem.cloud.placeholders import _apply_bindings

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
            facts, reverse, cloud_bindings={}
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
        from paramem.cloud.placeholders import _apply_bindings

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
            facts, reverse, cloud_bindings={}
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
        from paramem.cloud.placeholders import _apply_bindings

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
            facts, reverse, cloud_bindings={}
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
        SINGLE deanon exit gate) drops facts with any placeholder-shaped
        token, bare or composite, via the fail-closed
        :data:`_PLACEHOLDER_TOKEN_RE` backstop — even with an empty
        declared vocabulary (nothing in ``reverse``/``cloud_bindings``
        here)."""
        from paramem.cloud.placeholders import _apply_bindings

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
    def test_plausibility_with_cloud_returns_facts_and_raw(self):
        """request_plausibility returns (facts, raw_response).

        Plausibility is a rule-keyed drop-set protocol — the judge emits a
        small ``{"drop": {"R1": [<index>, ...], ...}}`` map from rule id to
        the indices that rule drops, instead of echoing kept facts. An
        empty rule map (``{"drop": {}}``) keeps every input fact unchanged.
        """
        from paramem.graph.extractor import request_plausibility

        fake_raw = '{"drop": {}}'
        input_fact = {"subject": "A", "predicate": "knows", "object": "B"}
        with patch("paramem.graph.extractor._cloud_call", return_value=fake_raw):
            verdict, raw = request_plausibility(
                [input_fact],
                api_key="k",
                provider="anthropic",
                anon_transcript="A knows B.",
            )
        assert verdict.kept == [input_fact]
        assert verdict.dropped == []
        assert raw == fake_raw

    def test_plausibility_with_cloud_none_on_api_failure(self):
        """API failure returns (None, None) — callers must destructure both."""
        from paramem.graph.extractor import request_plausibility

        with patch("paramem.graph.extractor._cloud_call", return_value=None):
            facts, raw = request_plausibility(
                [],
                api_key="k",
                provider="anthropic",
            )
        assert facts is None
        assert raw is None


class TestFilterWithCloudPromptsDir:
    """``request_enrichment`` accepts a ``prompts_dir`` parameter and
    forwards it, so the ``cloud_enrichment.txt`` load honours a calibration
    override."""

    def test_prompts_dir_override_reaches_enrichment_prompt(self, tmp_path):
        from paramem.graph.extractor import request_enrichment

        sentinel = "SENTINEL-cloud-ENRICH"
        (tmp_path / "cloud_enrichment.txt").write_text(
            f"{sentinel}\nfacts: {{facts_json}}\ntranscript: {{transcript}}"
        )
        captured_prompts = []

        def fake_cloud_call(prompt, *args, **kwargs):
            captured_prompts.append(prompt)
            return '{"add": [], "modify": [], "drop": [], "bindings": {}}'

        with patch("paramem.graph.extractor._cloud_call", side_effect=fake_cloud_call):
            request_enrichment(
                [{"subject": "A", "predicate": "knows", "object": "B"}],
                api_key="k",
                provider="anthropic",
                anon_transcript="A knows B.",
                prompts_dir=tmp_path,
                speaker_id="speaker0",
            )

        assert captured_prompts, "_cloud_call was never invoked"
        assert sentinel in captured_prompts[0], (
            f"Enrichment call used the shipped prompt instead of the override: "
            f"{captured_prompts[0]!r}"
        )

    def test_default_prompts_dir_uses_shipped_template(self):
        """Parity check: omitting ``prompts_dir`` must keep loading the
        production template — the new parameter is additive only."""
        from paramem.graph.extractor import request_enrichment

        captured_prompts = []

        def fake_cloud_call(prompt, *args, **kwargs):
            captured_prompts.append(prompt)
            return '{"add": [], "modify": [], "drop": [], "bindings": {}}'

        with patch("paramem.graph.extractor._cloud_call", side_effect=fake_cloud_call):
            request_enrichment(
                [{"subject": "A", "predicate": "knows", "object": "B"}],
                api_key="k",
                provider="anthropic",
                anon_transcript="A knows B.",
                speaker_id="speaker0",
            )

        assert captured_prompts
        assert "SENTINEL" not in captured_prompts[0]


class TestStageEnrichSuppliesSpeakerIdToRequestEnrichment:
    """``paramem.graph.stage_enrich._stage_enrich``'s ``request_enrichment``
    call always supplies ``ctx.speaker_id``. ``speaker_id`` is
    keyword-required with no default, so a call site cannot silently omit
    it and degrade the prompt without failing loudly. This drives the REAL
    ``request_enrichment`` through the ``enrich`` stage (only its cloud
    transport, ``_cloud_call``, is mocked — no GPU needed since the
    ``anonymize`` stage's local model call, ``anonymize``, is also
    mocked) and inspects the actual rendered prompt for the supplied
    speaker id.
    """

    def test_ctx_speaker_id_reaches_the_rendered_enrichment_prompt(self):
        from tests._cloud_flow import run_cloud_stages

        graph = _make_graph([("Alex", "lives_in", "Millfield")])
        mapping = {"Alex": "Person_1", "Millfield": "City_1"}
        captured_prompts = []

        def fake_cloud_call(prompt, *args, **kwargs):
            captured_prompts.append(prompt)
            return '{"add": [], "modify": [], "drop": [], "bindings": {}}'

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.stage_anonymize.anonymize",
                side_effect=_anonymize_stub(mapping, "anonymized transcript", ""),
            ),
            patch("paramem.graph.extractor._cloud_call", side_effect=fake_cloud_call),
        ):
            run_cloud_stages(
                graph,
                "Alex lives in Millfield.",
                None,
                None,
                speaker_id="speaker7",
                plausibility_judge="off",
                scrub={"person name"},
            )

        assert captured_prompts, "request_enrichment must have reached _cloud_call"
        assert "speaker7" in captured_prompts[0], (
            "stage_enrich must thread ctx.speaker_id into request_enrichment's "
            "rendered prompt — a missing/degraded anchor would silently drop "
            f"'speaker7' from: {captured_prompts[0]!r}"
        )


class TestPlausibilityFilterWithCloudPromptsDir:
    """``request_plausibility`` had the same gap: no ``prompts_dir``
    parameter, no forwarding, ``cloud_plausibility.txt`` loaded unconditionally."""

    def test_prompts_dir_override_reaches_plausibility_prompt(self, tmp_path):
        from paramem.graph.extractor import request_plausibility

        sentinel = "SENTINEL-cloud-PLAUSIBILITY"
        (tmp_path / "cloud_plausibility.txt").write_text(
            f"{sentinel}\nfacts: {{facts_json}}\ntranscript: {{transcript}}"
        )
        captured_prompts = []

        def fake_cloud_call(prompt, *args, **kwargs):
            captured_prompts.append(prompt)
            return '{"drop": {}}'

        with patch("paramem.graph.extractor._cloud_call", side_effect=fake_cloud_call):
            request_plausibility(
                [{"subject": "A", "predicate": "knows", "object": "B"}],
                api_key="k",
                provider="anthropic",
                anon_transcript="A knows B.",
                prompts_dir=tmp_path,
            )

        assert captured_prompts, "_cloud_call was never invoked"
        assert sentinel in captured_prompts[0], (
            f"Plausibility call used the shipped prompt instead of the override: "
            f"{captured_prompts[0]!r}"
        )

    def test_default_prompts_dir_uses_shipped_template(self):
        """Parity check: omitting ``prompts_dir`` must keep loading the
        production template — the new parameter is additive only."""
        from paramem.graph.extractor import request_plausibility

        captured_prompts = []

        def fake_cloud_call(prompt, *args, **kwargs):
            captured_prompts.append(prompt)
            return '{"drop": {}}'

        with patch("paramem.graph.extractor._cloud_call", side_effect=fake_cloud_call):
            request_plausibility(
                [{"subject": "A", "predicate": "knows", "object": "B"}],
                api_key="k",
                provider="anthropic",
                anon_transcript="A knows B.",
            )

        assert captured_prompts
        assert "SENTINEL" not in captured_prompts[0]


class TestCloudSystemPromptCallTimeOverride:
    """``cloud_enrichment_system.txt`` / ``cloud_plausibility_system.txt``
    load at CALL TIME inside each consuming function, within whatever
    :func:`~paramem.graph.phase_trace.extraction_trace` scope and
    :func:`~paramem.graph.prompts.prompt_overrides` context is active, so a
    calibration override reaches them and ``record_prompt`` records the
    override.  These tests pin BOTH halves of that contract: the override
    must reach ``_cloud_call``/``generate_answer`` (first two assertions
    per test) AND ``record.prompts`` must carry the override entry (the
    provenance assertion) — a plain "does ``_load_prompt`` honour an
    override" unit test cannot tell these apart.
    """

    def test_cloud_enrichment_system_prompt_overridable_and_recorded(self):
        from paramem.graph.extractor import request_enrichment
        from paramem.graph.phase_trace import extraction_trace, phase_trace
        from paramem.graph.prompts import prompt_overrides

        captured = []

        def fake_cloud_call(prompt, *args, **kwargs):
            captured.append(kwargs.get("system_prompt"))
            return '{"add": [], "modify": [], "drop": [], "bindings": {}}'

        with patch("paramem.graph.extractor._cloud_call", side_effect=fake_cloud_call):
            with extraction_trace() as trace:
                with phase_trace("cloud_enrich"):
                    with prompt_overrides(
                        {"cloud_enrichment_system.txt": "SENTINEL-ENRICH-SYSTEM"}
                    ):
                        request_enrichment(
                            [{"subject": "A", "predicate": "knows", "object": "B"}],
                            api_key="k",
                            provider="anthropic",
                            anon_transcript="A knows B.",
                            speaker_id="speaker0",
                        )
                record = trace.records[-1]

        assert captured == ["SENTINEL-ENRICH-SYSTEM"], (
            "the override must reach _cloud_call's system_prompt kwarg"
        )
        paths = [p["path"] for p in (record.prompts or [])]
        assert "<override:cloud_enrichment_system.txt>" in paths, (
            f"override must be recorded in phase-trace provenance, got paths={paths!r}"
        )

    def test_cloud_plausibility_system_prompt_overridable_and_recorded(self):
        from paramem.graph.extractor import request_plausibility
        from paramem.graph.phase_trace import extraction_trace, phase_trace
        from paramem.graph.prompts import prompt_overrides

        captured = []

        def fake_cloud_call(prompt, *args, **kwargs):
            captured.append(kwargs.get("system_prompt"))
            return '{"drop": {}}'

        with patch("paramem.graph.extractor._cloud_call", side_effect=fake_cloud_call):
            with extraction_trace() as trace:
                with phase_trace("anon_plausibility"):
                    with prompt_overrides(
                        {"cloud_plausibility_system.txt": "SENTINEL-PLAUS-SYSTEM"}
                    ):
                        request_plausibility(
                            [{"subject": "A", "predicate": "knows", "object": "B"}],
                            api_key="k",
                            provider="anthropic",
                            anon_transcript="A knows B.",
                        )
                record = trace.records[-1]

        assert captured == ["SENTINEL-PLAUS-SYSTEM"], (
            "the override must reach _cloud_call's system_prompt kwarg"
        )
        paths = [p["path"] for p in (record.prompts or [])]
        assert "<override:cloud_plausibility_system.txt>" in paths, (
            f"override must be recorded in phase-trace provenance, got paths={paths!r}"
        )

    def test_judge_plausibility_system_prompt_overridable_and_recorded(self):
        """``judge_plausibility`` reuses ``cloud_plausibility_system.txt``
        as the LOCAL model's system message — it builds the chat
        ``messages`` list directly rather than calling ``_cloud_call``."""
        from paramem.graph.extractor import judge_plausibility
        from paramem.graph.phase_trace import extraction_trace, phase_trace
        from paramem.graph.prompts import prompt_overrides

        facts = [{"subject": "Alex", "predicate": "lives_in", "object": "Millfield"}]
        tokenizer = MagicMock()
        tokenizer.apply_chat_template = MagicMock(return_value="formatted")
        with (
            patch("paramem.graph.extractor.generate_answer", return_value='{"drop": {}}'),
            # Identity passthrough so the real messages list (carrying the
            # override) reaches apply_chat_template unchanged — see the
            # companion note on TestFilterWithCloudPromptsDir-style tests.
            patch(
                "paramem.models.loader.adapt_messages",
                side_effect=lambda messages, tok: messages,
            ),
        ):
            with extraction_trace() as trace:
                with phase_trace("deanon_plausibility"):
                    with prompt_overrides(
                        {"cloud_plausibility_system.txt": "SENTINEL-LOCAL-PLAUS-SYSTEM"}
                    ):
                        judge_plausibility(facts, "transcript", MagicMock(), tokenizer)
                record = trace.records[-1]

        called_messages = tokenizer.apply_chat_template.call_args.args[0]
        system_contents = [m["content"] for m in called_messages if m["role"] == "system"]
        assert system_contents == ["SENTINEL-LOCAL-PLAUS-SYSTEM"], (
            "the override must reach the local model's system message"
        )
        paths = [p["path"] for p in (record.prompts or [])]
        assert "<override:cloud_plausibility_system.txt>" in paths, (
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
        """``speaker_directive.txt`` routes through ``_load_prompt`` (via
        ``_load_prompt_section``), so it is reachable by a calibration
        override and recorded via ``record_prompt``.
        ``build_speaker_context`` feeds this section into the
        ``{speaker_context}`` slot of the extraction user template — it is
        part of the prompt under test."""
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


class TestDocumentContextInjection:
    """``build_document_context`` renders the document-provenance directive
    into the ``{document_context}`` slot every extraction user template
    declares immediately before ``{transcript}``."""

    def test_document_source_with_display_name_renders_directive(self):
        """source_type='document' with both speaker_id and speaker_name
        renders the directive naming both."""
        from paramem.graph.extractor import build_document_context

        out = build_document_context("document", "speaker0", "Alex Walker")
        assert "speaker0" in out
        assert "Alex Walker" in out
        assert out.endswith("\n\n")

    def test_transcript_source_renders_nothing(self):
        """source_type='transcript' never renders the directive, even with
        a fully-known speaker — nothing to bind a first-person transcript
        to."""
        from paramem.graph.extractor import build_document_context

        assert build_document_context("transcript", "speaker0", "Alex Walker") == ""

    def test_anonymous_document_renders_nothing(self):
        """source_type='document' with no display name (anonymous
        provider) renders nothing — there is no name to bind facts to."""
        from paramem.graph.extractor import build_document_context

        assert build_document_context("document", "speaker0", None) == ""

    def test_document_source_with_no_speaker_id_renders_nothing(self):
        from paramem.graph.extractor import build_document_context

        assert build_document_context("document", None, "Alex Walker") == ""
        assert build_document_context("document", "", "Alex Walker") == ""

    def test_suppressed_paths_load_no_prompt_file(self):
        """Neither the transcript-source nor the anonymous-document path
        loads ``document_directive.txt`` — the guard runs BEFORE the load,
        so no provenance entry is recorded for a fragment that never
        rendered."""
        from paramem.graph.extractor import build_document_context
        from paramem.graph.phase_trace import extraction_trace, phase_trace

        with extraction_trace() as trace:
            with phase_trace("local_extract"):
                build_document_context("transcript", "speaker0", "Alex Walker")
                build_document_context("document", "speaker0", None)
            record = trace.records[-1]

        paths = [p["path"] for p in (record.prompts or [])]
        assert not any("document_directive.txt" in p for p in paths), (
            f"document_directive.txt must not be loaded on a suppressed path, got paths={paths!r}"
        )

    def test_generate_extraction_records_document_directive_provenance(self):
        """A document-mode ``_generate_extraction`` call earns its own
        ``{path, sha, template}`` provenance line for
        ``document_directive.txt`` — the same recording chokepoint
        (``_load_prompt`` → ``record_prompt``) every other extraction
        prompt file already uses. A transcript-mode call, which never
        renders the directive, records none."""
        from unittest.mock import MagicMock, patch

        from paramem.graph.extractor import _generate_extraction
        from paramem.graph.phase_trace import extraction_trace, phase_trace

        with (
            patch("paramem.graph.extractor.render_chat_prompt", return_value="formatted"),
            patch("paramem.graph.extractor.generate_answer", return_value="{}"),
        ):
            with extraction_trace() as trace:
                with phase_trace("local_extract"):
                    _generate_extraction(
                        MagicMock(),
                        MagicMock(),
                        "TRANSCRIPT_BODY",
                        0.0,
                        100,
                        speaker_id="speaker0",
                        speaker_name="Alex Walker",
                        source_type="document",
                    )
                document_record = trace.records[-1]

            with extraction_trace() as trace:
                with phase_trace("local_extract"):
                    _generate_extraction(
                        MagicMock(),
                        MagicMock(),
                        "TRANSCRIPT_BODY",
                        0.0,
                        100,
                        speaker_id="speaker0",
                        speaker_name="Alex Walker",
                        source_type="transcript",
                    )
                transcript_record = trace.records[-1]

        doc_entries = [
            p for p in (document_record.prompts or []) if "document_directive.txt" in p["path"]
        ]
        assert len(doc_entries) == 1, (
            f"expected exactly one document_directive.txt provenance entry, "
            f"got {document_record.prompts!r}"
        )
        assert doc_entries[0]["sha"], "recorded sha must be non-empty"
        assert doc_entries[0]["template"], "recorded template must be non-empty"

        transcript_paths = [p["path"] for p in (transcript_record.prompts or [])]
        assert not any("document_directive.txt" in p for p in transcript_paths), (
            f"transcript-mode call must record no document_directive.txt provenance, "
            f"got paths={transcript_paths!r}"
        )

    def test_generate_extraction_renders_document_context_for_document_source(self):
        """``_generate_extraction`` threads ``build_document_context``'s
        output into the rendered user message, immediately before the
        transcript, only for a document-source pass with a known speaker."""
        from unittest.mock import MagicMock, patch

        from paramem.graph.extractor import _generate_extraction

        captured: dict = {}

        def fake_render(messages, tokenizer, **kwargs):
            captured["messages"] = messages
            return "formatted"

        with (
            patch("paramem.graph.extractor.render_chat_prompt", side_effect=fake_render),
            patch("paramem.graph.extractor.generate_answer", return_value="{}"),
        ):
            _generate_extraction(
                MagicMock(),
                MagicMock(),
                "TRANSCRIPT_BODY",
                0.0,
                100,
                speaker_id="speaker0",
                speaker_name="Alex Walker",
                source_type="document",
            )

        user_content = captured["messages"][1]["content"]
        # Pin adjacency and the exact separator, not just relative ordering:
        # the rendered directive must be immediately followed by "\n\n" and
        # then the transcript body. Built from build_document_context itself
        # so this stays accurate through a future content-tuning edit to
        # document_directive.txt.
        from paramem.graph.extractor import build_document_context

        rendered_directive = build_document_context("document", "speaker0", "Alex Walker")
        assert f"{rendered_directive}TRANSCRIPT_BODY" in user_content

    def test_generate_extraction_renders_no_document_context_for_transcript_source(self):
        """A transcript-source pass never carries the directive text, even
        with the same known speaker."""
        from unittest.mock import MagicMock, patch

        from paramem.graph.extractor import _generate_extraction

        captured: dict = {}

        def fake_render(messages, tokenizer, **kwargs):
            captured["messages"] = messages
            return "formatted"

        with (
            patch("paramem.graph.extractor.render_chat_prompt", side_effect=fake_render),
            patch("paramem.graph.extractor.generate_answer", return_value="{}"),
        ):
            _generate_extraction(
                MagicMock(),
                MagicMock(),
                "TRANSCRIPT_BODY",
                0.0,
                100,
                speaker_id="speaker0",
                speaker_name="Alex Walker",
                source_type="transcript",
            )

        user_content = captured["messages"][1]["content"]
        assert "Document provided by" not in user_content


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
    """``on_extraction_end`` / ``on_recall_probe`` — the artifact hooks, driven
    by the ``debug_run`` scope.  Debug-write semantics: plaintext,
    ``_snapshot`` suffix, procedural omitted when empty.
    """

    def test_on_extraction_end_writes_plaintext(self, tmp_path):
        from paramem.utils.artifacts import debug_run, on_extraction_end

        out_dir = tmp_path / "episodic" / "cycle_4" / "run_xyz"
        episodic_rels = [{"question": "Q", "answer": "A"}]
        procedural_rels = [{"subject": "S", "predicate": "P", "object": "O"}]

        with debug_run(out_dir):
            on_extraction_end(episodic_rels, procedural_rels)

        assert (out_dir / "episodic_rels_snapshot.json").exists()
        assert (out_dir / "procedural_rels_snapshot.json").exists()
        # on_extraction_end never writes the cumulative graph — that is
        # on_fold_graph's responsibility (graph_merged_snapshot.json +
        # graph_enriched_snapshot.json).
        assert not (out_dir / "graph_snapshot.json").exists()

        with open(out_dir / "episodic_rels_snapshot.json") as f:
            saved = json.load(f)
        assert saved == episodic_rels

    def test_on_extraction_end_omits_procedural_when_empty(self, tmp_path):
        from paramem.utils.artifacts import debug_run, on_extraction_end

        out_dir = tmp_path / "episodic" / "cycle_2" / "run_xyz"
        with debug_run(out_dir):
            on_extraction_end([{"question": "Q", "answer": "A"}], [])

        assert (out_dir / "episodic_rels_snapshot.json").exists()
        assert not (out_dir / "procedural_rels_snapshot.json").exists()

    def test_on_extraction_end_short_circuits_when_debug_off(self, tmp_path):
        """``debug_run(None)`` IS the off state — no root, no write, no flag test."""
        from paramem.utils.artifacts import debug_run, on_extraction_end

        with debug_run(None):
            on_extraction_end([{"question": "Q", "answer": "A"}], [])

        assert list(tmp_path.iterdir()) == []

    def test_on_recall_probe_writes_per_key_json(self, tmp_path):
        """on_recall_probe writes recall_probes/<phase>_<adapter>.json with payload."""
        from paramem.utils.artifacts import debug_run, on_recall_probe

        out_dir = tmp_path / "cycle_5" / "run_abc"

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
        with debug_run(out_dir):
            on_recall_probe(per_key, phase="disk_verify", adapter_name="procedural")

        artifact = out_dir / "recall_probes" / "disk_verify_procedural.json"
        assert artifact.exists(), f"Expected artifact at {artifact}"
        saved = json.loads(artifact.read_text())
        assert saved == per_key
        assert saved[0]["raw_output"] != ""
        assert saved[1]["failure_reason"] == "parse_failure"

    def test_on_recall_probe_noop_when_per_key_none(self, tmp_path):
        """on_recall_probe is a no-op when per_key is None."""
        from paramem.utils.artifacts import debug_run, on_recall_probe

        out_dir = tmp_path / "cycle_5" / "run_abc"
        with debug_run(out_dir):
            on_recall_probe(None, phase="train_fill", adapter_name="episodic")

        assert not (out_dir / "recall_probes").exists()

    def test_on_recall_probe_noop_when_debug_off(self, tmp_path):
        """on_recall_probe is a no-op when no artifact root is open."""
        from paramem.utils.artifacts import debug_run, on_recall_probe

        per_key = [{"key": "proc32", "exact_match": True, "raw_output": "x"}]
        with debug_run(None):
            on_recall_probe(per_key, phase="disk_verify", adapter_name="procedural")

        assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# Extraction pipeline alignment tests
# ---------------------------------------------------------------------------


class TestPlausibilityAnon:
    """The ``enrich`` stage with plausibility_stage="anon": plausibility
    runs on anonymized facts before de-anonymization.
    """

    def test_anon_stage_plausibility_filters_subset(self):
        """When plausibility_stage="anon" and a cloud validator is configured, it runs
        on the anonymized facts before de-anonymization and drops flagged entries."""
        from tests._cloud_flow import enrichment_side_effect, run_cloud_stages

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

        # Plausibility filter drops the role-leak fact, keeps lives_in.
        def fake_plaus(facts, api_key, **kwargs):
            return _verdict_dropping(facts, {"has_role"}), "raw"

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.stage_anonymize.anonymize",
                side_effect=_anonymize_stub(mapping, "anonymized transcript", ""),
            ),
            patch(
                "paramem.graph.stage_enrich.request_enrichment",
                side_effect=enrichment_side_effect(anon_facts),
            ),
            patch(
                "paramem.graph.stage_enrich.request_plausibility",
                side_effect=fake_plaus,
            ),
        ):
            result = run_cloud_stages(
                graph,
                "Alex lives in Millfield.",
                None,
                None,
                speaker_id="speaker0",
                plausibility_judge="anthropic",
                plausibility_stage="anon",
                correction_entity_types=set(),
                scrub={"person name"},
            )

        # Only the valid fact survives
        assert len(result.relations) == 1
        assert result.relations[0].predicate == "lives_in"
        assert result.diagnostics.get("plausibility") == "anon"
        assert result.diagnostics["plausibility_dropped_anon"] == 1

    def test_anon_stage_plausibility_receives_post_enrichment_transcript(self):
        """FIX 2: the anon-stage judge must see `updated_anon_transcript`
        (post-enrichment — carrying cloud's minted `{Paper_1}`-style
        tokens), not the pre-enrichment `anon_transcript`.  A judge shown
        the stale pre-enrichment transcript can never connect an
        enrichment-only fact's placeholder to its real-text span in the
        transcript, and drops a valid enrichment.

        Mutation: revert the call site to pass `anon_transcript=anon_transcript`
        -> this test fails.
        """
        from paramem.graph.extractor import EnrichmentDelta
        from tests._cloud_flow import run_cloud_stages

        graph = _make_graph(
            [("Alex", "authored", "Attention Is All You Need")],
            entities=[
                Entity(name="Alex", entity_type="person"),
            ],
        )
        # Pre-enrichment: local extraction never saw the paper title as a
        # placeholder — "Attention Is All You Need" reaches the anon
        # transcript verbatim (no mapping entry).  Cloud's delta below
        # MODIFIES the one local fact's object to reference a new
        # placeholder it mints, bound to that same span — real-world
        # equivalent of "cloud reified a bare object into a named entity".
        mapping = {"Alex": "Person_1"}
        anon_transcript = "Person_1 authored Attention Is All You Need."
        # The reconstructed post-enrichment transcript
        # (``_reconstruct_updated_transcript``, run for real by
        # ``_apply_enrichment_delta`` inside the ``enrich`` stage) —
        # substitutes the bound span with the braced placeholder.
        post_enrichment_transcript = "Person_1 authored {Paper_1}."

        plaus_calls = []

        def fake_plaus(facts, api_key, **kwargs):
            plaus_calls.append(kwargs.get("anon_transcript"))
            return _kept_verdict(facts), "raw"

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.stage_anonymize.anonymize",
                side_effect=_anonymize_stub(mapping, anon_transcript, ""),
            ),
            patch(
                "paramem.graph.stage_enrich.request_enrichment",
                return_value=(
                    EnrichmentDelta(
                        add=[],
                        modify=[(0, {"object": "Paper_1"})],
                        drop=set(),
                        bindings={"Paper_1": "Attention Is All You Need"},
                    ),
                    None,
                    {},
                ),
            ),
            patch(
                "paramem.graph.stage_enrich.request_plausibility",
                side_effect=fake_plaus,
            ),
        ):
            run_cloud_stages(
                graph,
                "Alex wrote a well-known paper.",
                None,
                None,
                speaker_id="speaker0",
                plausibility_judge="anthropic",
                plausibility_stage="anon",
                correction_entity_types=set(),
                scrub={"person name"},
            )

        assert len(plaus_calls) == 1
        assert plaus_calls[0] == post_enrichment_transcript, (
            f"anon-stage judge must see the post-enrichment transcript, got {plaus_calls[0]!r}"
        )


class TestPlausibilityDeanon:
    """The ``deanonymize`` stage with plausibility_stage="deanon":
    plausibility runs on de-anonymized facts using the original
    transcript.
    """

    def test_deanon_stage_plausibility_drops_tautology(self):
        """Deanon-stage local plausibility receives real names and drops tautologies."""
        from tests._cloud_flow import enrichment_side_effect, run_cloud_stages

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

        local_plaus_calls = []

        def fake_local_plaus(facts, transcript, model, tokenizer, **kwargs):
            local_plaus_calls.append((list(facts), transcript))
            # Local plausibility drops the tautology, keeps lives_in.
            return _verdict_dropping(facts, {"has_name"}), ""

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.stage_anonymize.anonymize",
                side_effect=_anonymize_stub(mapping, "anonymized transcript", ""),
            ),
            patch(
                "paramem.graph.stage_enrich.request_enrichment",
                side_effect=enrichment_side_effect(anon_facts),
            ),
            patch(
                "paramem.graph.flows.judge_plausibility",
                side_effect=fake_local_plaus,
            ),
        ):
            result = run_cloud_stages(
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
        assert result.diagnostics["plausibility_dropped_deanon"] == 1


class TestExtractionSeedSourcing:
    """Seed is sourced like its sibling sampling knobs (temperature/max_tokens):
    the config default (``None`` — no seeding, the value production runs
    with), overridable by a calibration probe.  Production and
    calibration thread it through one identical path (``kwargs()``), so a seed
    set for a production run is the same seed a calibration run reads — the
    reproducibility calibration exists to provide.
    """

    def _pipe(self, cfg_seed):
        from paramem.graph.extraction_pipeline import ExtractionConfig, ExtractionPipeline

        return ExtractionPipeline(
            MagicMock(),
            MagicMock(),
            config=ExtractionConfig(seed=cfg_seed, scrub_categories=()),
        )

    def test_seed_defaults_to_config_none_is_status_quo(self):
        # Production shape: no override, config default None → no seeding.
        assert self._pipe(None).kwargs(speaker_id="speaker0")["seed"] is None

    def test_seed_falls_back_to_the_config_value(self):
        # A config seed reaches the pipeline with no override (reproducible
        # production run); calibration reading the same config reproduces it.
        assert self._pipe(42).kwargs(speaker_id="speaker0")["seed"] == 42

    def test_calibration_override_wins_over_config(self):
        # A calibration probe pins its own seed to sweep sampling variance.
        assert self._pipe(42).kwargs(speaker_id="speaker0", seed=7)["seed"] == 7

    def test_sibling_sampling_knobs_default_to_config(self):
        kw = self._pipe(None).kwargs(speaker_id="speaker0")
        assert kw["temperature"] == 0.0  # ExtractionConfig defaults
        assert kw["max_tokens"] == 8192

    def test_sibling_sampling_knob_overrides_are_honored(self):
        # temperature/max_tokens overrides were dead (read straight off cfg);
        # now they flow like seed. temperature=0.0 is a value, not "unset".
        kw = self._pipe(None).kwargs(speaker_id="speaker0", temperature=0.7, max_tokens=256)
        assert kw["temperature"] == 0.7
        assert kw["max_tokens"] == 256


class TestExtractGraphSeedIsolation:
    """The injected chain-seed graph seeds ``StageState.graph`` but must NOT
    become ``ctx.seed`` — the int sampling seed forwarded to every
    ``generate_answer``.

    Regression: ``extract_graph`` rebound its ``seed`` parameter to
    ``chain_seed()``, so (a) the parameter was globally dead — production's
    ``ctx.seed`` was always ``None`` — and (b) a calibration caller injecting a
    graph fed a ``SessionGraph`` into ``int(seed)`` at the first
    ``generate_answer`` (anonymize), 500-ing ``/calibrate/{enrich,plausibility}``.
    """

    def test_injected_graph_does_not_clobber_sampling_seed(self):
        from paramem.graph.flows import extract_graph
        from paramem.graph.phase_trace import start_at

        captured: dict = {}

        def _fake_run_flow(_spec, ctx, state):
            captured["ctx_seed"] = ctx.seed
            captured["state_graph"] = state.graph
            return state

        injected = SessionGraph(session_id="calib", timestamp="2026-01-01T00:00:00Z")
        with patch("paramem.graph.flows.run_flow", side_effect=_fake_run_flow):
            with start_at("anonymize", injected):
                extract_graph(
                    MagicMock(),
                    MagicMock(),
                    "[user] hi there",
                    "calib",
                    "speaker0",
                    seed=1234,
                    scrub_categories=(),
                )

        # The int sampling seed survives graph injection ...
        assert captured["ctx_seed"] == 1234
        # ... and the injected graph is what seeds the initial StageState.
        assert captured["state_graph"] is injected

    def test_no_injection_starts_from_a_fresh_graph_and_forwards_seed(self):
        """Production shape: no injection → StageState starts empty, and the
        sampling seed still reaches ctx.seed (was dead before)."""
        from paramem.graph.flows import extract_graph
        from paramem.graph.phase_trace import start_at

        captured: dict = {}

        def _fake_run_flow(_spec, ctx, state):
            captured["ctx_seed"] = ctx.seed
            captured["state_graph"] = state.graph
            return state

        with patch("paramem.graph.flows.run_flow", side_effect=_fake_run_flow):
            # start_at(None) opens no injection — chain_seed() is None.
            with start_at(None):
                extract_graph(
                    MagicMock(),
                    MagicMock(),
                    "[user] hi there",
                    "calib",
                    "speaker0",
                    seed=7,
                    scrub_categories=(),
                )

        assert captured["ctx_seed"] == 7
        assert captured["state_graph"].session_id == "calib"


class TestCloudEnrichmentFailureModes:
    """``request_enrichment`` returns ``delta=None`` after exhausting its
    retries, and the enrich stage branches on ``parse_path`` — the two failure
    modes get opposite treatment because opposite recoveries work:

    * ``"failed"`` (HICCUP) — the provider answered but never in the delta
      envelope shape.  Fail OPEN: keep the pre-enrichment facts, record
      ``cloud_enrichment_degraded``, continue.  Waiting would not help (same
      input → same mis-shape), so degrade this one session.  Consistent with
      the graph-tier enrichment path and both plausibility judges.
    * ``"no_response"`` (OUTAGE) — the provider was unreachable on every
      attempt.  Raise ``ExtractionFailed`` so the batch aborts and its sessions
      stay pending for a clean retry once cloud recovers.

    Only the outage case belongs on the raising side: making BOTH fatal
    aborts a whole batch over a single mis-shaped provider response.
    """

    def test_cloud_enrich_hiccup_fails_open(self):
        """request_enrichment returning (None, ...) → pre-enrichment facts
        kept, degradation recorded, no raise."""
        from tests._cloud_flow import run_cloud_stages

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
                "paramem.graph.stage_anonymize.anonymize",
                side_effect=_anonymize_stub(mapping, "anonymized transcript", ""),
            ),
            patch(
                "paramem.graph.stage_enrich.request_enrichment",
                # First element None ⇒ every retry missed a parseable delta.
                # The stage must fail OPEN: keep the pre-enrichment facts,
                # record the degradation, and NOT raise.
                return_value=(None, None, {"parse_path": "failed", "attempts": 3}),
            ),
        ):
            result = run_cloud_stages(
                graph,
                "transcript",
                None,
                None,
                speaker_id="speaker0",
                correction_entity_types=set(),
                scrub={"person name"},
            )

        # The pre-enrichment fact survives — the run was NOT failed.
        assert any(r.predicate == "lives_in" for r in result.relations)
        # The degradation is recorded so the server layer can surface it on
        # pstatus, and it names how many attempts were spent.
        degraded = result.diagnostics.get("cloud_enrichment_degraded")
        assert degraded is not None
        assert degraded["attempts"] == 3
        assert degraded["kept_facts"] == 1

    def test_cloud_enrich_outage_raises_extraction_failed(self):
        """request_enrichment returning (None, ..., parse_path="no_response")
        → ExtractionFailed, so the batch aborts and its sessions stay
        pending for a clean retry once the provider recovers."""
        from paramem.graph.extractor import ExtractionFailed
        from tests._cloud_flow import run_cloud_stages

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
                "paramem.graph.stage_anonymize.anonymize",
                side_effect=_anonymize_stub(mapping, "anonymized transcript", ""),
            ),
            patch(
                "paramem.graph.stage_enrich.request_enrichment",
                # No response on any attempt ⇒ outage, not a shape hiccup.
                return_value=(None, None, {"parse_path": "no_response", "attempts": 3}),
            ),
        ):
            with pytest.raises(ExtractionFailed) as excinfo:
                run_cloud_stages(
                    graph,
                    "transcript",
                    None,
                    None,
                    speaker_id="speaker0",
                    correction_entity_types=set(),
                    scrub={"person name"},
                )
        assert excinfo.value.phase == "cloud_enrich"

    def test_extraction_failed_exposes_phase_and_reason(self):
        """Exception class contract used by the app.py per-chunk handler."""
        from paramem.graph.extractor import ExtractionFailed

        exc = ExtractionFailed("cloud_enrich", "timeout")
        assert exc.phase == "cloud_enrich"
        assert exc.reason == "timeout"
        assert "cloud_enrich" in str(exc)
        assert "timeout" in str(exc)


class TestAllDroppedSafetyNet:
    """All-dropped safety net (the ``rebuild`` stage,
    ``paramem.graph.flows._stage_rebuild``) fires when the pipeline
    empties out post-deanon. Plausibility is the final discriminator that
    can empty the pipeline."""

    def test_all_dropped_triggers_fallback(self):
        """When plausibility drops every surviving fact, the all-dropped
        safety net invokes _fallback_plausibility_on_raw with reason
        'all_dropped' so the session does not yield zero facts."""
        from tests._cloud_flow import enrichment_side_effect, run_cloud_stages

        graph = _make_graph(
            [("Alex", "lives_in", "Millfield")],
            entities=[
                Entity(name="Alex", entity_type="person"),
                Entity(name="Millfield", entity_type="place"),
            ],
        )
        anon_facts = [{"subject": "Person_1", "predicate": "lives_in", "object": "City_1"}]
        mapping = {"Alex": "Person_1", "Millfield": "City_1"}
        # cloud returns the same single fact; plausibility drops it (returns []).
        cloud_enriched = list(anon_facts)

        fallback_calls = []

        def fake_fallback(g, t, m, tok, reason, **_kwargs):
            fallback_calls.append(reason)
            g.diagnostics["fallback_path"] = reason
            return g

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.stage_anonymize.anonymize",
                side_effect=_anonymize_stub(mapping, "anonymized transcript", ""),
            ),
            patch(
                "paramem.graph.stage_enrich.request_enrichment",
                side_effect=enrichment_side_effect(cloud_enriched),
            ),
            patch(
                "paramem.graph.flows.judge_plausibility",
                return_value=(
                    PlausibilityVerdict(
                        kept=[],
                        dropped=[
                            {
                                "index": 0,
                                "rule": None,
                                "fact": {
                                    "subject": "Alex",
                                    "predicate": "lives_in",
                                    "object": "Millfield",
                                },
                            }
                        ],
                        out_of_range=[],
                    ),
                    "",
                ),
            ),
            patch(
                "paramem.graph.flows._fallback_plausibility_on_raw",
                side_effect=fake_fallback,
            ),
        ):
            result = run_cloud_stages(
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
    """Entity types set by _normalize_extraction must survive the cloud pipeline
    unchanged; no "person" stampdown on non-person entities.
    """

    def test_preserved_entity_types_pass_through(self):
        """Entities pre-typed by _normalize_extraction keep their original types
        after the pipeline even when mocked cloud returns same facts."""
        from tests._cloud_flow import enrichment_side_effect, run_cloud_stages

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
                "paramem.graph.stage_anonymize.anonymize",
                side_effect=_anonymize_stub(mapping, "anonymized transcript", ""),
            ),
            patch(
                "paramem.graph.stage_enrich.request_enrichment",
                side_effect=enrichment_side_effect(anon_facts),
            ),
        ):
            result = run_cloud_stages(
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

    def test_cloud_introduced_country_entity_typed_location(self):
        """Cloud-introduced entity with Country_ placeholder is typed 'location', not 'person'."""
        from tests._cloud_flow import enrichment_side_effect, run_cloud_stages

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
                "paramem.graph.stage_anonymize.anonymize",
                side_effect=_anonymize_stub(mapping, "anonymized transcript", ""),
            ),
            patch(
                "paramem.graph.stage_enrich.request_enrichment",
                side_effect=enrichment_side_effect(anon_facts),
            ),
        ):
            result = run_cloud_stages(
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
        # to cloud-introduced entities (names absent from the original graph).
        # "place" and "location" both express geographic entities — accept both values.
        assert entity_map.get("Germany") in ("place", "location"), (
            f"Germany (Country_1) must be typed 'place' or 'location', "
            f"not {entity_map.get('Germany')!r}"
        )

    def test_cloud_introduced_entity_no_placeholder_typed_concept(self):
        """Cloud-introduced entity with no placeholder (bare name) gets type
        'concept', not 'person'.

        Regression guard: entity with no reverse_mapping entry must default to
        'concept', never 'person'.
        China is NOT present in the original graph — only Alex is. Cloud enrichment
        introduces China as a bare name (no anonymizer placeholder), so no
        reverse_mapping entry exists. The entity-type-preservation rule ensures the
        fallback type is 'concept', never 'person'.
        """
        from tests._cloud_flow import enrichment_side_effect, run_cloud_stages

        # Original graph has only Alex — no China entity
        graph = _make_graph(
            [("Alex", "has_plans", "Alex")],  # placeholder relation; cloud will override
            entities=[
                Entity(name="Alex", entity_type="person"),
            ],
        )
        # Alex → Person_1 only; China is absent from the anonymization mapping
        mapping = {"Alex": "Person_1"}
        # cloud enrichment introduces China as a bare name with no placeholder equivalent
        enriched_anon = [{"subject": "Person_1", "predicate": "visited", "object": "China"}]

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.stage_anonymize.anonymize",
                side_effect=_anonymize_stub(mapping, "anonymized transcript", ""),
            ),
            patch(
                "paramem.graph.stage_enrich.request_enrichment",
                side_effect=enrichment_side_effect(enriched_anon),
            ),
        ):
            result = run_cloud_stages(
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
            f"cloud-introduced bare entity must be typed 'concept', not {china_type!r}"
        )


class TestCloudMintedEntityTypeDerivation:
    """The entity-rebuild loop (extractor.py, "Rebuild entity list from
    surviving + new relations") must resolve a de-anonymized cloud-minted
    entity's REAL NAME back to its placeholder via the inverted
    resolution map, not via ``reverse_mapping.get(name)`` — ``reverse_mapping``
    is keyed by placeholder, so looking it up with a real name always
    misses.
    """

    def test_cloud_minted_entity_gets_prefix_derived_type(self):
        """A cloud-minted entity bound via ``bindings`` with a novel prefix
        (``Paper_1``, absent from the closed anonymizer vocabulary) lands
        in graph.entities typed by its prefix ("paper"), not "concept".

        Mutation that must make this fail: restore the lookup to
        ``reverse_mapping.get(name)`` (today's dead-code behaviour) —
        ``reverse_mapping`` has no "Attention Is All You Need" key (it is
        keyed by placeholder), so the entity falls back to "concept".
        """
        from tests._cloud_flow import enrichment_side_effect, run_cloud_stages

        graph = _make_graph(
            [("Alex", "has_plans", "Alex")],  # placeholder relation; cloud will override
            entities=[
                Entity(name="Alex", entity_type="person"),
            ],
        )
        mapping = {"Alex": "Person_1"}
        # cloud mints a novel-prefix entity via a braced placeholder plus an
        # explicit binding — the documented brace-binding protocol.
        enriched_anon = [{"subject": "Person_1", "predicate": "authored", "object": "{Paper_1}"}]
        cloud_bindings = {"Paper_1": "Attention Is All You Need"}

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.stage_anonymize.anonymize",
                side_effect=_anonymize_stub(mapping, "anonymized transcript", ""),
            ),
            patch(
                "paramem.graph.stage_enrich.request_enrichment",
                side_effect=enrichment_side_effect(enriched_anon, bindings=cloud_bindings),
            ),
        ):
            result = run_cloud_stages(
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
            f"cloud-minted novel-prefix entity must be typed 'paper', not {paper_type!r}"
        )

    def test_cloud_minted_entity_known_prefix_uses_configured_type(self):
        """A cloud-minted entity with a KNOWN prefix (``Person_2``) still
        maps through the schema's ``anonymizer_prefix_to_type()`` to
        "person" — the closed-vocabulary branch of the same derivation.
        """
        from tests._cloud_flow import enrichment_side_effect, run_cloud_stages

        graph = _make_graph(
            [("Alex", "has_plans", "Alex")],
            entities=[
                Entity(name="Alex", entity_type="person"),
            ],
        )
        mapping = {"Alex": "Person_1"}
        enriched_anon = [{"subject": "Person_1", "predicate": "met", "object": "{Person_2}"}]
        cloud_bindings = {"Person_2": "Jordan Rivers"}

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.stage_anonymize.anonymize",
                side_effect=_anonymize_stub(mapping, "anonymized transcript", ""),
            ),
            patch(
                "paramem.graph.stage_enrich.request_enrichment",
                side_effect=enrichment_side_effect(enriched_anon, bindings=cloud_bindings),
            ),
        ):
            result = run_cloud_stages(
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
            f"cloud-minted known-prefix entity must be typed 'person', not {jordan_type!r}"
        )


class TestFallbackPlausibilityOnRawHelper:
    """Direct tests of the _fallback_plausibility_on_raw helper: runs on
    already-de-anonymized ``graph.relations`` (real names) and records
    anon-failed facts.
    """

    def test_helper_does_not_sweep_shape_like_real_names(self):
        """This helper operates on real-name ``graph.relations`` where no
        placeholder vocabulary exists, so it never sweeps for residual
        placeholders (``_strip_residual_placeholders``): a real name that
        merely happens to be shaped like a placeholder (``Boeing_747``)
        must survive — a shape-only guard here could only ever produce
        false positives."""
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
    plausibility_stage) into the run-constant StageContext every stage
    (including ``enrich``) reads from.
    """

    def test_extract_graph_plumbs_plausibility_kwargs(self):
        """extract_graph forwards plausibility_judge, plausibility_stage
        into the ``StageContext`` it builds.

        ``plausibility_judge``/``plausibility_stage`` travel exclusively
        via ``StageContext``, built once per call and read by whichever
        stage needs them (``enrich``). Faking ``run_flow`` itself captures
        exactly that ctx, with no real stage body (and therefore no model)
        involved.
        """
        from paramem.graph.flows import extract_graph

        captured = {}

        def fake_run_flow(flow, ctx, state):
            captured["plausibility_judge"] = ctx.plausibility_judge
            captured["plausibility_stage"] = ctx.plausibility_stage
            return state

        with patch("paramem.graph.flows.run_flow", side_effect=fake_run_flow):
            extract_graph(
                None,
                None,
                "transcript",
                "sess1",
                speaker_id="speaker0",
                cloud_enabled=True,
                enrichment_provider="anthropic",
                plausibility_judge="anthropic",
                plausibility_stage="anon",
                scrub_categories=_scrub_categories("person name"),
            )

        assert captured.get("plausibility_judge") == "anthropic"
        assert captured.get("plausibility_stage") == "anon"

    def test_extract_graph_default_temperature_zero(self):
        """extract_graph default temperature must be 0.0.

        Structured output (JSON, QA) requires deterministic generation.
        """
        import inspect

        from paramem.graph.flows import extract_graph

        sig = inspect.signature(extract_graph)
        assert sig.parameters["temperature"].default == 0.0

    def test_extract_graph_default_max_tokens_matches_filter_default(self):
        """extract_graph default max_tokens matches the unified filter
        constant. The single-budget invariant: the entry point and
        downstream filter calls must default to the same value so a
        missing config doesn't produce inconsistent budgets across
        stages."""
        import inspect

        from paramem.graph.extractor import _DEFAULT_FILTER_MAX_TOKENS
        from paramem.graph.flows import extract_graph

        sig = inspect.signature(extract_graph)
        assert sig.parameters["max_tokens"].default == _DEFAULT_FILTER_MAX_TOKENS


class TestDiagnosticsKeys:
    """Diagnostics dict is populated with expected keys after a full pipeline run."""

    def test_diagnostics_contains_plausibility_keys(self):
        """After a deanon-stage plausibility run, diagnostics contains the expected keys."""
        from tests._cloud_flow import enrichment_side_effect, run_cloud_stages

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
            return _kept_verdict(facts), ""  # keep all

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.stage_anonymize.anonymize",
                side_effect=_anonymize_stub(mapping, "anonymized transcript", ""),
            ),
            patch(
                "paramem.graph.stage_enrich.request_enrichment",
                side_effect=enrichment_side_effect(anon_facts),
            ),
            patch(
                "paramem.graph.flows.judge_plausibility",
                side_effect=fake_local_plaus,
            ),
        ):
            result = run_cloud_stages(
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
        assert result.diagnostics["plausibility_state_deanon"] == {"state": "ran", "reason": None}
        assert result.diagnostics["plausibility_out_of_range_deanon"] == []

    def test_diagnostics_anonymize_key_populated_on_success(self):
        """diagnostics['anonymize']='ok' when anonymization succeeds."""
        from tests._cloud_flow import enrichment_side_effect, run_cloud_stages

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
                "paramem.graph.stage_anonymize.anonymize",
                side_effect=_anonymize_stub(mapping, "anonymized transcript", ""),
            ),
            patch(
                "paramem.graph.stage_enrich.request_enrichment",
                side_effect=enrichment_side_effect(anon_facts),
            ),
        ):
            result = run_cloud_stages(
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
            extraction_plausibility_judge="anthropic",
            extraction_plausibility_stage="anon",
        )
        assert cfg.extraction_plausibility_judge == "anthropic"
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

    def test_extraction_enrichment_provider_defaults_to_disabled(self):
        """Privacy invariant: a config that omits ``extraction_enrichment_provider``
        must NOT default to a cloud provider. ``""`` is the disabled
        sentinel — a deployment whose YAML omits the key must not silently
        send extraction-pipeline content to the cloud (see SECURITY.md's
        stated default posture).
        """
        from paramem.server.config import ConsolidationScheduleConfig

        cfg = ConsolidationScheduleConfig()
        assert cfg.extraction_enrichment_provider == ""

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
# Binding collisions — diagnostic check post-anonymization
# ---------------------------------------------------------------------------


class TestBindingCollisions:
    """Unit tests for ``_binding_collisions`` — the collision-only scan.
    The per-fact orphan predicate lives separately as
    ``_fact_orphans``/``_fact_tokens``/``_placeholder_tokens`` — see their
    coverage in ``tests/test_placeholders.py``.

    ALWAYS informational: a binding for a token cloud was SHOWN is inert
    under CORE-LAST precedence (:func:`_resolution_map`), so nothing
    gates on this function's result — it is a diagnostic only, pinned
    separately at the caller side by
    ``_record_binding_diagnostics``/``DeanonResult.collisions``.
    """

    def test_no_cloud_bindings_returns_empty(self):
        from paramem.cloud.placeholders import _binding_collisions

        assert _binding_collisions({"Person_1": "Alex"}, cloud_bindings=None) == []
        assert _binding_collisions({"Person_1": "Alex"}, cloud_bindings={}) == []

    def test_unscoped_conflicting_value_is_a_collision(self, caplog):
        """``observed=None`` (CORE unscoped): a ``cloud_bindings`` key
        present in ``reverse_mapping`` with a DIFFERING value is a
        collision, and is warned about — the reverse-wins tie-break in
        :func:`_apply_bindings` would otherwise silently resolve to the
        wrong real name."""
        import logging

        from paramem.cloud.placeholders import _binding_collisions

        reverse_mapping = {"Org_1": "Acme"}
        cloud_bindings = {"Org_1": "Wrong Corp"}
        caplog.set_level(logging.WARNING, logger="paramem.cloud.placeholders")
        collisions = _binding_collisions(reverse_mapping, cloud_bindings=cloud_bindings)
        assert collisions == ["Org_1"]
        assert any("collision" in r.getMessage().lower() for r in caplog.records)

    def test_unscoped_matching_value_is_not_a_collision(self):
        """Same key, SAME value in both maps is not a conflict — only a
        DIFFERING value counts."""
        from paramem.cloud.placeholders import _binding_collisions

        reverse_mapping = {"Org_1": "Acme"}
        cloud_bindings = {"Org_1": "Acme"}
        assert _binding_collisions(reverse_mapping, cloud_bindings=cloud_bindings) == []

    def test_scoped_key_in_observed_is_a_collision(self):
        """``observed`` given: any ``cloud_bindings`` key that is ALSO in
        ``observed`` is a conflict, regardless of value equality — cloud
        is rebinding something it was already shown as a core
        reference."""
        from paramem.cloud.placeholders import _binding_collisions

        collisions = _binding_collisions(
            {"Person_1": "Alex"},
            cloud_bindings={"Person_1": "someone else entirely"},
            observed={"Person_1"},
        )
        assert collisions == ["Person_1"]

    def test_scoped_key_not_in_observed_is_not_a_collision(self):
        """A cloud mint for a token never shown to cloud (not in
        ``observed``) is a legitimate new entity, not a collision."""
        from paramem.cloud.placeholders import _binding_collisions

        collisions = _binding_collisions(
            {"Person_1": "Alex"},
            cloud_bindings={"Org_9": "Acme"},
            observed={"Person_1"},
        )
        assert collisions == []

    def test_multiple_collisions_sorted(self):
        from paramem.cloud.placeholders import _binding_collisions

        collisions = _binding_collisions(
            {},
            cloud_bindings={"Zeta_1": "z", "Alpha_1": "a"},
            observed={"Zeta_1", "Alpha_1"},
        )
        assert collisions == ["Alpha_1", "Zeta_1"]

    def test_returns_list_not_none(self):
        """Explicit-return contract: never ``None``, always a (possibly
        empty) list — safe for a plain truthiness test."""
        from paramem.cloud.placeholders import _binding_collisions

        collisions = _binding_collisions({"Person_1": "Alex"}, cloud_bindings=None)
        assert collisions == []
        assert collisions is not None


class TestRecordBindingDiagnostics:
    """``_record_binding_diagnostics`` — the CALLER side of the collision
    diagnostic, and the only place in the extractor that turns a
    ``DeanonResult`` into ``graph.diagnostics`` entries.

    ``DeanonResult`` carries no ``verdict`` field — nothing gates on a
    whole-delta orphan list — so ``cloud_pending_orphans`` is never
    written. ``cloud_binding_collisions`` is written on EVERY call: a
    present ``[]`` means the scan ran and found nothing; the key's total
    absence would mean the scan never reached this site, a state this
    function cannot produce, since it always writes.
    """

    @staticmethod
    def _result(collisions: list[str]):
        from paramem.cloud.deanonymize import DeanonResult

        return DeanonResult(
            facts=[],
            collisions=collisions,
            predicate_dropped=[],
            residual_dropped=[],
        )

    def test_empty_collisions_writes_empty_list(self):
        """Mutation: reintroducing an ``if collisions:`` guard would make
        an accepted delta omit the key when there are no collisions, and
        every ``diagnostics["cloud_binding_collisions"] == []`` assertion
        in the suite would raise KeyError."""
        from paramem.graph.extractor import _record_binding_diagnostics

        graph = _make_graph([])
        _record_binding_diagnostics(graph, self._result([]))
        assert graph.diagnostics["cloud_binding_collisions"] == []

    def test_collisions_land_under_their_key(self):
        from paramem.graph.extractor import _record_binding_diagnostics

        graph = _make_graph([])
        _record_binding_diagnostics(graph, self._result(["Person_2"]))
        assert graph.diagnostics["cloud_binding_collisions"] == ["Person_2"]

    def test_no_cloud_pending_orphans_key_is_ever_written(self):
        """``cloud_pending_orphans`` is not a valid diagnostics key — a
        collision (or anything else on ``DeanonResult``) never writes it,
        regardless of content."""
        from paramem.graph.extractor import _record_binding_diagnostics

        graph = _make_graph([])
        _record_binding_diagnostics(graph, self._result(["Person_2"]))
        assert "cloud_pending_orphans" not in graph.diagnostics


class TestResolutionMap:
    """CORE PRECEDENCE, pinned directly on :func:`_resolution_map`,
    independent of the rejection gate.  Backstops need their own test: do
    not skip this because the rejection gate makes the collision unreachable
    in the full pipeline — a future refactor flipping the ``.update()``
    order would otherwise silently let cloud overwrite a real name with no
    test failing.
    """

    def test_core_wins_on_key_in_both_maps_unscoped(self):
        """observed=None (CORE unscoped): reverse wins on collision —
        today's behaviour, preserved."""
        from paramem.cloud.placeholders import _resolution_map

        reverse = {"Org_1": "Acme"}
        cloud_bindings = {"Org_1": "Wrong Corp"}
        resolved = _resolution_map(reverse, cloud_bindings, observed=None)
        assert resolved["Org_1"] == "Acme"

    def test_core_wins_on_key_in_both_maps_scoped(self):
        """observed as a set: same key in both domains still resolves to
        CORE — vacuous under normal scoped construction (the rejection
        gate would have already rejected this delta) but must hold if
        the map is asked to resolve one directly."""
        from paramem.cloud.placeholders import _resolution_map

        reverse = {"Org_1": "Acme"}
        cloud_bindings = {"Org_1": "Wrong Corp"}
        resolved = _resolution_map(reverse, cloud_bindings, observed={"Org_1"})
        assert resolved["Org_1"] == "Acme"

    def test_observed_none_is_core_unscoped_every_reverse_entry_legal(self):
        from paramem.cloud.placeholders import _resolution_map

        reverse = {"Person_1": "Alex", "City_1": "Berlin"}
        resolved = _resolution_map(reverse, {}, observed=None)
        assert resolved == reverse

    def test_observed_scoped_excludes_reverse_entries_outside_observed(self):
        from paramem.cloud.placeholders import _resolution_map

        reverse = {"Person_1": "Alex", "City_1": "Berlin"}
        resolved = _resolution_map(reverse, {}, observed={"Person_1"})
        assert resolved == {"Person_1": "Alex"}

    def test_cloud_mint_outside_observed_is_included(self):
        from paramem.cloud.placeholders import _resolution_map

        reverse = {"Person_1": "Alex"}
        cloud_bindings = {"Event_1": "the workshop"}
        resolved = _resolution_map(reverse, cloud_bindings, observed={"Person_1"})
        assert resolved == {"Person_1": "Alex", "Event_1": "the workshop"}


class TestBindingTotalityRejection:
    """Per-triple accept/drop/revert of an invalid cloud-enrichment delta.
    An unresolvable ``add`` is dropped individually; an unresolvable
    ``modify`` is reverted to its original fact; ``drop`` is honored
    unconditionally (even when another action in the same delta was
    rejected — tracked via ``report["drop_with_rejection"]``, not a
    revert-all safety net). These are the pipeline-level tests the
    per-triple contract requires.

    FIXTURE MECHANICS: ``anon_transcript`` is the MODEL's own rewrite —
    the ``anon_transcript`` argument to
    ``_anonymize_contract``/``_anonymize_stub`` (this file's mocked
    ``anonymize`` return shape) — never mechanically rebuilt from
    ``transcript`` + ``mapping``.  ``observed`` is derived from the
    DECLARED token vocabulary intersected with the rendered payload
    (facts JSON + ``anon_transcript``), so a test controlling what is
    "observed" controls the mocked ``anonymize`` transcript string
    and/or the facts the graph fixture carries — not a
    transcript-substitution side effect.
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

    def test_poisoned_delta_5in1_collapse_untouched_fact_survives(self, caplog):
        """A poisoned delta: Cloud DROPS the ``lives_in``
        fact (index 0) and ADDS 5 facts over bare, unbound
        Person_2/Person_3 (``bindings={}``) as its "reformation" — a
        SEPARATE, untouched local fact (``works_at``, index 1, never named
        by the delta) sits alongside it.

        Per-triple, not whole-delta: all 5 adds are
        individually dropped (unresolvable orphans); the explicit
        ``drop`` on index 0 is honored UNCONDITIONALLY regardless of
        those rejections (deliberate: measure the co-occurrence via
        ``report["drop_with_rejection"]`` rather than reverting drops as
        a safety net) — so ``lives_in`` does NOT survive.  The untouched
        ``works_at`` fact (never named by ``add``/``modify``/``drop``)
        survives via KEEP-by-default: survival comes from being OUTSIDE
        the delta entirely, not from a reverted drop.
        """
        import logging

        from paramem.graph.extractor import EnrichmentDelta
        from paramem.graph.phase_trace import extraction_trace
        from tests._cloud_flow import run_cloud_stages

        graph = _make_graph(
            [("Alex", "lives_in", "Millfield"), ("Alex", "works_at", "Acme")],
            entities=[
                Entity(name="Alex", entity_type="person"),
                Entity(name="Millfield", entity_type="place"),
                Entity(name="Acme", entity_type="organization"),
            ],
        )
        mapping = {"Alex": "Person_1", "Millfield": "City_1", "Acme": "Org_1"}
        poisoned_adds = [
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
        delta = EnrichmentDelta(add=poisoned_adds, modify=[], drop={0}, bindings={})

        # The per-triple rejection WARNING now originates in
        # paramem.graph.stage_enrich (carved out of extractor.py's
        # _cloud_pipeline) — attach to that logger, not extractor's.
        caplog.set_level(logging.WARNING, logger="paramem.graph.stage_enrich")
        with extraction_trace() as trace:
            with (
                patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
                patch(
                    "paramem.graph.stage_anonymize.anonymize",
                    side_effect=_anonymize_stub(mapping, "anonymized transcript", ""),
                ),
                patch(
                    "paramem.graph.stage_enrich.request_enrichment",
                    return_value=(delta, None, {}),
                ),
            ):
                result = run_cloud_stages(
                    graph,
                    "Alex lives in Millfield",
                    None,
                    None,
                    speaker_id="speaker0",
                    correction_entity_types=set(),
                    scrub={"person name"},
                )

        # Only the untouched fact survives — dropped index 0 stays
        # dropped (unconditional), the 5 orphan adds are each rejected.
        assert len(result.relations) == 1
        assert result.relations[0].subject == "Alex"
        assert result.relations[0].predicate == "works_at"
        assert result.relations[0].object == "Acme"
        # No residual placeholder anywhere in the surviving relation.
        assert "Person_" not in result.relations[0].subject
        assert "Person_" not in result.relations[0].object
        # report["drop_with_rejection"] tracks co-occurrence: this delta had
        # a non-empty drop AND a rejection, in the same response.
        report = result.diagnostics["cloud_enrichment_report"]
        assert report["rejected_adds"] == 5
        assert set(report["rejected_tokens"]) >= {"Person_2", "Person_3"}
        assert report["drop_with_rejection"] is True
        phases = {p.name: p for p in trace.records}
        assert phases["cloud_enrich"].outcome == "ok"
        assert any(r.levelname == "WARNING" for r in caplog.records), (
            "A per-triple rejection must be logged (WARNING, not ERROR — this "
            "is expected traffic, not a breach)."
        )

    def test_misattribution_orphan_add_dropped_local_fact_survives(self):
        """A misattribution scenario: a placeholder NOT in ``observed``
        (never shown to cloud) that cloud bare-mints as an ``add`` is an
        ORPHAN → that one add is dropped; the untouched local fact
        survives via KEEP-by-default (a DIFFERENT mechanism than the
        poisoned-delta test above, same end state)."""
        from paramem.graph.extractor import EnrichmentDelta
        from tests._cloud_flow import run_cloud_stages

        graph, _anon_facts, mapping = self._graph_and_mapping()
        # Person_3 is bare-minted by cloud but was never shown to it (not
        # in the rendered facts, not in the transcript, not bound).
        delta = EnrichmentDelta(
            add=[
                {
                    "subject": "Person_3",
                    "predicate": "profession",
                    "object": "engineer",
                    "relation_type": "factual",
                    "confidence": 0.9,
                },
            ],
            modify=[],
            drop=set(),
            bindings={},
        )
        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.stage_anonymize.anonymize",
                side_effect=_anonymize_stub(mapping, "anonymized transcript", ""),
            ),
            patch(
                "paramem.graph.stage_enrich.request_enrichment",
                return_value=(delta, None, {}),
            ),
        ):
            result = run_cloud_stages(
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
        report = result.diagnostics["cloud_enrichment_report"]
        assert report["rejected_adds"] == 1
        assert "Person_3" in report["rejected_tokens"]
        assert report["drop_with_rejection"] is False

    def test_bare_observed_placeholder_as_new_subject_accepted(self):
        """Rule 1 must not regress.  A delta referencing a bare
        OBSERVED placeholder (Person_1 — already shown to cloud) as the
        subject of a NEW triple, minting nothing, is ACCEPTED.  This test
        and ``test_misattribution_orphan_add_dropped_local_fact_survives``
        differ ONLY in observed-membership."""
        from paramem.graph.phase_trace import extraction_trace
        from tests._cloud_flow import enrichment_side_effect, run_cloud_stages

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
                    "paramem.graph.stage_anonymize.anonymize",
                    side_effect=_anonymize_stub(mapping, "anonymized transcript", ""),
                ),
                patch(
                    "paramem.graph.stage_enrich.request_enrichment",
                    side_effect=enrichment_side_effect(enriched_anon),
                ),
            ):
                result = run_cloud_stages(
                    graph,
                    "Alex lives in Millfield",
                    None,
                    None,
                    speaker_id="speaker0",
                    correction_entity_types=set(),
                    scrub={"person name"},
                )
        assert len(result.relations) == 2
        assert result.diagnostics["cloud_enrichment_report"]["rejected_adds"] == 0
        phases = {p.name: p for p in trace.records}
        assert phases["cloud_enrich"].outcome == "ok"

    def test_binding_key_colliding_with_observed_is_inert_fact_kept(self):
        """A ``bindings`` key that
        is itself an OBSERVED token (Person_1 — already shown as a core
        reference) is a CONFLICT recorded as a diagnostic collision, but
        CORE-LAST precedence (:func:`~paramem.cloud.placeholders._resolution_map`)
        makes it harmless: the fact is KEPT, resolved via the CORE
        reverse map, exactly as if the bogus binding had never been sent."""
        from tests._cloud_flow import enrichment_side_effect, run_cloud_stages

        graph, anon_facts, mapping = self._graph_and_mapping()
        enriched_anon = list(anon_facts)
        bindings = {"Person_1": "some other person entirely"}
        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.stage_anonymize.anonymize",
                side_effect=_anonymize_stub(mapping, "anonymized transcript", ""),
            ),
            patch(
                "paramem.graph.stage_enrich.request_enrichment",
                side_effect=enrichment_side_effect(enriched_anon, bindings=bindings),
            ),
        ):
            result = run_cloud_stages(
                graph,
                "Alex lives in Millfield",
                None,
                None,
                speaker_id="speaker0",
                correction_entity_types=set(),
                scrub={"person name"},
            )
        # The collision is still recorded (informational) ...
        assert result.diagnostics["cloud_binding_collisions"] == ["Person_1"]
        # ... but the fact is KEPT, not dropped.
        assert len(result.relations) == 1
        assert result.relations[0].subject == "Alex"
        assert result.relations[0].object == "Millfield"

    def test_mint_bound_to_descriptor_accepted(self):
        """Mint happy path.  Cloud mints a placeholder BOUND to a
        descriptor span ("my father", ∉ observed) → ACCEPTED; the
        relation de-anonymizes to the bound text."""
        from tests._cloud_flow import enrichment_side_effect, run_cloud_stages

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
                "paramem.graph.stage_anonymize.anonymize",
                side_effect=_anonymize_stub(mapping, "anonymized transcript", ""),
            ),
            patch(
                "paramem.graph.stage_enrich.request_enrichment",
                side_effect=enrichment_side_effect(enriched_anon, bindings=bindings),
            ),
        ):
            result = run_cloud_stages(
                graph,
                "Alex lives in Millfield",
                None,
                None,
                speaker_id="speaker0",
                correction_entity_types=set(),
                scrub={"person name"},
            )
        assert result.diagnostics["cloud_enrichment_report"]["rejected_adds"] == 0
        subjects_objects = {(r.subject, r.object) for r in result.relations}
        assert ("Alex", "my father") in subjects_objects

    def test_predicate_only_reference_still_observed_accepted(self):
        """A placeholder appearing ONLY in a predicate is still
        ∈ observed (the RENDERED payload, predicate included, is what
        Cloud is actually shown), so a bare reference to it elsewhere is
        ACCEPTED.  Guards the rendered-payload requirement: a
        subject/object-only field scan would under-include ``observed``
        and false-reject this.

        CORE placeholders come straight
        from the model's own anonymizer mapping (there is no
        code-side entity walk that mints for graph entities the model
        didn't name).  So ``anonymize`` is mocked to have already
        classified BOTH places (``Millfield`` -> ``City_1``,
        ``Springfield`` -> ``City_2``) — the model decision this test's
        fixture would need in production for either place to be a CORE
        placeholder at all.
        """
        from tests._cloud_flow import enrichment_side_effect, run_cloud_stages

        # Springfield -> City_2 appears ONLY inside a compound PREDICATE
        # string here, never as a subject/object anywhere in the local
        # extract.  `predicate` is never a substitution target, so the
        # graph relation's own predicate must already carry "City_2"
        # verbatim for it to reach the cloud-facing payload the script
        # builds.
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
        # cloud bare-references City_2 as an object — legal: it is a real
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
                "paramem.graph.stage_anonymize.anonymize",
                side_effect=_anonymize_stub(mapping, "anonymized transcript", ""),
            ),
            patch(
                "paramem.graph.stage_enrich.request_enrichment",
                side_effect=enrichment_side_effect(enriched_anon),
            ),
        ):
            result = run_cloud_stages(
                graph,
                "Alex moved from Springfield to Millfield.",
                None,
                None,
                speaker_id="speaker0",
                correction_entity_types=set(),
                scrub={"person name"},
            )
        assert result.diagnostics["cloud_enrichment_report"]["rejected_adds"] == 0, (
            "City_2/Springfield appears only in a predicate but is still "
            "observed — a field-scan-only `observed` would false-reject."
        )
        # `enriched_anon[0]` (the "moved from City_2 to" fact) is dropped by
        # `_apply_bindings`'s deanon-stage predicate invariant — its own
        # predicate contains a declared token — so it never reaches
        # `result.relations`.  This assertion rides ENTIRELY on the
        # second, cloud-added fact ("recently_visited"); it is not
        # evidence the first fact survived.
        subjects_objects = {(r.subject, r.object) for r in result.relations}
        assert ("Alex", "Springfield") in subjects_objects


class TestSpeakerAnchorPipeline:
    """The speaker anchor through the pipeline.  A PII attribute on the
    speaker is scrubbed onto the anchor, never a minted ``Person_N`` —
    covered directly against
    :func:`~paramem.cloud.placeholders.build_forward_table` in
    ``tests/test_placeholders.py::TestAnchorFoldReverseSkip``.  The
    model's SCAN output is the sole scope authority, so there is no
    graph-entity/attribute fold left in this module to pin end to end
    here.
    """

    def test_speaker0_survives_end_to_end_not_swept(self):
        """``speaker0`` survives extraction -> anonymize -> cloud ->
        deanon -> graph VERBATIM, and is never swept by
        :func:`_apply_bindings`'s residual sweep (it doesn't match the
        placeholder pattern at all — verified structurally, not
        assumed).

        CORE placeholders come straight
        from the model's own anonymizer mapping (no code-side entity
        walk), so the mock explicitly classifies ``Millfield`` ->
        ``City_1`` — the model decision this test's fixture would need
        in production.
        """
        from tests._cloud_flow import enrichment_side_effect, run_cloud_stages

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
                "paramem.graph.stage_anonymize.anonymize",
                side_effect=_anonymize_stub(mapping, "anonymized transcript", ""),
            ),
            patch(
                "paramem.graph.stage_enrich.request_enrichment",
                side_effect=enrichment_side_effect(enriched_anon),
            ),
        ):
            result = run_cloud_stages(
                graph,
                "speaker0 lives in Millfield",
                None,
                None,
                speaker_id="speaker0",
                correction_entity_types=set(),
                scrub={"person name"},
            )
        assert len(result.relations) == 1
        assert result.relations[0].subject == "speaker0"
        assert result.relations[0].object == "Millfield"
        assert "residual_dropped_facts" not in result.diagnostics
        assert result.diagnostics["cloud_enrichment_report"]["rejected_adds"] == 0

    def test_anchor_independent_of_speaker_relation_presence(self):
        """The anchor holds even in a session with NO speaker
        entity/relation at all (the protocol-constant case): the
        pipeline must not require a speaker fact to function correctly —
        nothing about the anonymizer/deanon machinery depends on the
        speaker being referenced THIS session."""
        from tests._cloud_flow import enrichment_side_effect, run_cloud_stages

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
                "paramem.graph.stage_anonymize.anonymize",
                side_effect=_anonymize_stub(mapping, "anonymized transcript", ""),
            ),
            patch(
                "paramem.graph.stage_enrich.request_enrichment",
                side_effect=enrichment_side_effect(list(anon_facts)),
            ),
        ):
            result = run_cloud_stages(
                graph,
                "Acme located in Millfield",
                None,
                None,
                speaker_id="speaker0",
                correction_entity_types=set(),
                scrub={"person name"},
            )
        assert len(result.relations) == 1
        assert result.relations[0].subject == "Acme"
        assert result.relations[0].object == "Millfield"
        assert result.diagnostics["cloud_enrichment_report"]["rejected_adds"] == 0


class TestObservedDerivation:
    """``observed`` — CORE's legality domain for a cloud cycle — is derived
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
        from tests._cloud_flow import enrichment_side_effect, run_cloud_stages

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
        from paramem.cloud import deanonymize as _cloud_deanonymize

        real_collisions = _cloud_deanonymize._binding_collisions

        def _spy(*args, **kwargs):
            if "observed" in kwargs:
                captured.append(kwargs["observed"])
            return real_collisions(*args, **kwargs)

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.stage_anonymize.anonymize",
                side_effect=_anonymize_stub(mapping, "anonymized transcript", ""),
            ),
            patch("paramem.cloud.deanonymize._binding_collisions", side_effect=_spy),
            patch(
                "paramem.graph.stage_enrich.request_enrichment",
                side_effect=enrichment_side_effect(list(anon_facts)),
            ),
        ):
            run_cloud_stages(
                graph,
                "Alex lives in Millfield and flew on a Boeing_747",
                None,
                None,
                speaker_id="speaker0",
                correction_entity_types=set(),
                scrub={"person name"},
            )

        assert captured, "the deanon stage's collision scan must run with an observed scope"
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


class TestBracedBindingKeysEndToEnd:
    """The cloud enrichment wire shape's binding keys are braced;
    the table normalizer boundary canonicalizes them to bare, and bare
    keys stay accepted (no migration). Exercised through
    ``request_enrichment``'s real parse (mocking only ``_cloud_call``) so
    the production double-normalize path
    (``_parse_enrichment_delta`` then ``CloudScope.response``) is real,
    not bypassed by constructing an ``EnrichmentDelta`` directly."""

    def _graph_and_mapping(self):
        graph = _make_graph(
            [("Alex", "works_at", "Acme")],
            entities=[
                Entity(name="Alex", entity_type="person"),
                Entity(name="Acme", entity_type="organization"),
            ],
        )
        mapping = {"Alex": "Person_1", "Acme": "Org_1"}
        return graph, mapping

    def _run(self, graph, mapping, raw_delta_json):
        from tests._cloud_flow import run_cloud_stages

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.stage_anonymize.anonymize",
                side_effect=_anonymize_stub(mapping, "anonymized transcript", ""),
            ),
            patch("paramem.graph.extractor._cloud_call", return_value=raw_delta_json),
        ):
            return run_cloud_stages(
                graph,
                "Alex works at Acme.",
                None,
                None,
                speaker_id="speaker0",
                correction_entity_types=set(),
                scrub={"person name"},
            )

    def test_braced_binding_key_resolves_add_fact_to_real_name(self):
        graph, mapping = self._graph_and_mapping()
        raw = json.dumps(
            {
                "add": [
                    {
                        "subject": "Person_1",
                        "predicate": "led",
                        "object": "{Event_1}",
                        "relation_type": "factual",
                        "confidence": 0.9,
                    }
                ],
                "modify": [],
                "drop": [],
                "bindings": {"{Event_1}": "the agile transformation initiative"},
            }
        )
        result = self._run(graph, mapping, raw)
        led = next((r for r in result.relations if r.predicate == "led"), None)
        assert led is not None, (
            f"led fact must survive with the braced binding resolved; got "
            f"{[(r.subject, r.predicate, r.object) for r in result.relations]}"
        )
        assert led.subject == "Alex"
        assert led.object == "the agile transformation initiative"

    def test_bare_binding_key_still_resolves(self):
        """The boundary accepts the bare-key shape as well as the braced
        form."""
        graph, mapping = self._graph_and_mapping()
        raw = json.dumps(
            {
                "add": [
                    {
                        "subject": "Person_1",
                        "predicate": "led",
                        "object": "{Event_1}",
                        "relation_type": "factual",
                        "confidence": 0.9,
                    }
                ],
                "modify": [],
                "drop": [],
                "bindings": {"Event_1": "the agile transformation initiative"},
            }
        )
        result = self._run(graph, mapping, raw)
        led = next((r for r in result.relations if r.predicate == "led"), None)
        assert led is not None
        assert led.subject == "Alex"
        assert led.object == "the agile transformation initiative"


class TestPlausibilityJudgeStateIntegration:
    """Every reachable path through the deanon-stage and
    anon-stage plausibility gates records an explicit judge state, and
    the ran-path attribution (rule per drop, out-of-range indices)
    reaches both ``graph.diagnostics`` and the phase-trace ``parsed``
    payload."""

    def _graph_and_mapping(self):
        graph = _make_graph(
            [("Alex", "lives_in", "Millfield")],
            entities=[
                Entity(name="Alex", entity_type="person"),
                Entity(name="Millfield", entity_type="place"),
            ],
        )
        mapping = {"Alex": "Person_1", "Millfield": "City_1"}
        anon_facts = [{"subject": "Person_1", "predicate": "lives_in", "object": "City_1"}]
        return graph, mapping, anon_facts

    def _run(self, graph, mapping, anon_facts, **kwargs):
        from tests._cloud_flow import enrichment_side_effect, run_cloud_stages

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.stage_anonymize.anonymize",
                side_effect=_anonymize_stub(mapping, "anonymized transcript", ""),
            ),
            patch(
                "paramem.graph.stage_enrich.request_enrichment",
                side_effect=enrichment_side_effect(anon_facts),
            ),
        ):
            return run_cloud_stages(
                graph,
                "Alex lives in Millfield.",
                kwargs.pop("model", MagicMock()),
                kwargs.pop("tokenizer", MagicMock()),
                speaker_id="speaker0",
                correction_entity_types=set(),
                scrub={"person name"},
                **kwargs,
            )

    def test_deanon_ran_with_rule_and_out_of_range(self):
        graph, mapping, anon_facts = self._graph_and_mapping()

        def fake_local_plaus(facts, transcript, model, tokenizer, **kwargs):
            verdict = PlausibilityVerdict(
                kept=[],
                dropped=[{"index": 0, "rule": "R1", "fact": facts[0]}],
                out_of_range=[{"index": 3, "input_count": 1, "rule": None}],
            )
            return verdict, "raw"

        with extraction_trace() as trace:
            with patch("paramem.graph.flows.judge_plausibility", side_effect=fake_local_plaus):
                result = self._run(
                    graph,
                    mapping,
                    anon_facts,
                    plausibility_judge="auto",
                    plausibility_stage="deanon",
                )

        assert result.diagnostics["plausibility_dropped_deanon"] == 1
        assert result.diagnostics["plausibility_dropped_deanon_facts"][0]["rule"] == "R1"
        assert result.diagnostics["plausibility_out_of_range_deanon"] == [
            {"index": 3, "input_count": 1, "rule": None}
        ]
        assert result.diagnostics["plausibility_state_deanon"] == {"state": "ran", "reason": None}
        deanon_plaus = next(p for p in trace.records if p.name == "deanon_plausibility")
        assert deanon_plaus.parsed["dropped_facts"][0]["rule"] == "R1"
        assert deanon_plaus.parsed["out_of_range"] == [{"index": 3, "input_count": 1, "rule": None}]

    def test_deanon_ran_unattributed_reaches_diagnostics_and_parsed(self):
        """A drop claim the judge could not pin to a rule is counted in
        ``plausibility_unattributed_deanon`` and the phase-trace
        ``parsed`` payload — never applied, so the input fact survives
        alongside the one real, rule-attributed drop."""
        graph, mapping, anon_facts = self._graph_and_mapping()

        def fake_local_plaus(facts, transcript, model, tokenizer, **kwargs):
            verdict = PlausibilityVerdict(
                kept=[],
                dropped=[{"index": 0, "rule": "R1", "fact": facts[0]}],
                out_of_range=[],
                unattributed=2,
            )
            return verdict, "raw"

        with extraction_trace() as trace:
            with patch("paramem.graph.flows.judge_plausibility", side_effect=fake_local_plaus):
                result = self._run(
                    graph,
                    mapping,
                    anon_facts,
                    plausibility_judge="auto",
                    plausibility_stage="deanon",
                )

        assert result.diagnostics["plausibility_unattributed_deanon"] == 2
        assert isinstance(result.diagnostics["plausibility_dropped_deanon_facts"][0]["rule"], str)
        deanon_plaus = next(p for p in trace.records if p.name == "deanon_plausibility")
        assert deanon_plaus.parsed["unattributed"] == 2

    def test_deanon_judge_returning_none_fails_open(self):
        graph, mapping, anon_facts = self._graph_and_mapping()

        with extraction_trace() as trace:
            with patch(
                "paramem.graph.flows.judge_plausibility",
                return_value=(None, "unparseable"),
            ):
                result = self._run(
                    graph,
                    mapping,
                    anon_facts,
                    plausibility_judge="auto",
                    plausibility_stage="deanon",
                )

        assert result.diagnostics["plausibility_state_deanon"] == {
            "state": "failed",
            "reason": "parse_failed",
        }
        assert len(result.relations) == 1  # fail-open: fact kept
        deanon_plaus = next(p for p in trace.records if p.name == "deanon_plausibility")
        assert deanon_plaus.outcome == "failed"

    @pytest.mark.parametrize(
        ("kwargs", "expected_state"),
        [
            (
                {"plausibility_judge": "off", "plausibility_stage": "deanon"},
                {"state": "off", "reason": None},
            ),
            (
                {"plausibility_judge": "auto", "plausibility_stage": "anon"},
                {"state": "skipped", "reason": "stage_not_deanon"},
            ),
        ],
    )
    def test_deanon_off_and_stage_mismatch_states(self, kwargs, expected_state):
        graph, mapping, anon_facts = self._graph_and_mapping()
        with extraction_trace() as trace:
            result = self._run(graph, mapping, anon_facts, **kwargs)
        assert result.diagnostics["plausibility_state_deanon"] == expected_state
        assert not any(p.name == "deanon_plausibility" for p in trace.records), (
            "No phase record is emitted on a skip/off path — the diagnostics "
            "state key carries this instead."
        )

    def test_deanon_no_model_state(self):
        graph, mapping, anon_facts = self._graph_and_mapping()
        with extraction_trace() as trace:
            result = self._run(
                graph,
                mapping,
                anon_facts,
                model=None,
                tokenizer=None,
                plausibility_judge="auto",
                plausibility_stage="deanon",
            )
        assert result.diagnostics["plausibility_state_deanon"] == {
            "state": "skipped",
            "reason": "no_model",
        }
        assert not any(p.name == "deanon_plausibility" for p in trace.records)

    def test_anon_not_permitted_state(self):
        """The anon-stage judge's ``not_permitted`` branch keeps its
        EXISTING phase-trace behaviour (record emitted, free-text gap
        reason) while gaining the new closed-vocabulary diagnostics key.
        The ran-arm's calibration short-circuit POSITION is pinned
        separately, by
        ``test_anon_ran_arm_stop_at_returns_before_deanonymize`` below."""
        graph, mapping, anon_facts = self._graph_and_mapping()
        with (
            extraction_trace() as trace,
            patch.dict("os.environ", {}, clear=True),
        ):
            result = self._run(
                graph,
                mapping,
                anon_facts,
                plausibility_judge="openai",
                plausibility_stage="anon",
            )
        assert result.diagnostics["plausibility_state_anon"] == {
            "state": "skipped",
            "reason": "not_permitted",
        }
        anon_plaus = next(p for p in trace.records if p.name == "anon_plausibility")
        assert anon_plaus.outcome == "skipped"
        assert anon_plaus.reason  # the existing free-text gap reason survives

    def test_anon_judge_not_a_provider_state(self):
        graph, mapping, anon_facts = self._graph_and_mapping()
        with extraction_trace() as trace:
            result = self._run(
                graph,
                mapping,
                anon_facts,
                plausibility_judge="auto",
                plausibility_stage="anon",
            )
        assert result.diagnostics["plausibility_state_anon"] == {
            "state": "skipped",
            "reason": "judge_not_a_provider",
        }
        assert not any(p.name == "anon_plausibility" for p in trace.records)

    def test_anon_stage_mismatch_state(self):
        """Mirrors the deanon site's own stage-mismatch coverage: an
        otherwise-valid provider does not run the anon judge when
        ``plausibility_stage`` names a different stage."""
        graph, mapping, anon_facts = self._graph_and_mapping()

        def fake_local_plaus(facts, transcript, model, tokenizer, **kwargs):
            return _kept_verdict(facts), ""

        with extraction_trace() as trace:
            with patch("paramem.graph.flows.judge_plausibility", side_effect=fake_local_plaus):
                result = self._run(
                    graph,
                    mapping,
                    anon_facts,
                    plausibility_judge="anthropic",
                    plausibility_stage="deanon",
                )
        assert result.diagnostics["plausibility_state_anon"] == {
            "state": "skipped",
            "reason": "stage_not_anon",
        }
        assert not any(p.name == "anon_plausibility" for p in trace.records)

    def test_anon_no_facts_state(self):
        """Mirrors the deanon site's no-facts skip: an empty
        ``enriched_anon`` (cloud enrichment dropped everything) skips the
        anon judge before it is ever called."""
        graph, mapping, _anon_facts = self._graph_and_mapping()
        with extraction_trace() as trace:
            result = self._run(
                graph,
                mapping,
                [],  # cloud enrichment leaves enriched_anon empty
                plausibility_judge="anthropic",
                plausibility_stage="anon",
            )
        assert result.diagnostics["plausibility_state_anon"] == {
            "state": "skipped",
            "reason": "no_facts",
        }
        assert not any(p.name == "anon_plausibility" for p in trace.records)

    def test_anon_judge_returning_none_fails_open(self):
        """Mirrors ``test_deanon_judge_returning_none_fails_open``: a
        parse failure at the anon judge fails open (facts kept) and
        records state ``failed``/``parse_failed``."""
        graph, mapping, anon_facts = self._graph_and_mapping()
        with extraction_trace() as trace:
            with patch(
                "paramem.graph.stage_enrich.request_plausibility",
                return_value=(None, "unparseable"),
            ):
                result = self._run(
                    graph,
                    mapping,
                    anon_facts,
                    plausibility_judge="anthropic",
                    plausibility_stage="anon",
                )
        assert result.diagnostics["plausibility_state_anon"] == {
            "state": "failed",
            "reason": "parse_failed",
        }
        assert len(result.relations) == 1  # fail-open: fact kept
        anon_plaus = next(p for p in trace.records if p.name == "anon_plausibility")
        assert anon_plaus.outcome == "failed"

    def test_anon_ran_with_rule_and_out_of_range(self):
        """Mirrors ``test_deanon_ran_with_rule_and_out_of_range``: a real
        drop with a cited rule, plus an out-of-range index, reaches both
        ``graph.diagnostics`` and the ``anon_plausibility`` phase's
        ``parsed`` payload."""
        graph, mapping, anon_facts = self._graph_and_mapping()

        def fake_plaus(facts, api_key, **kwargs):
            verdict = PlausibilityVerdict(
                kept=[],
                dropped=[{"index": 0, "rule": "R1", "fact": facts[0]}],
                out_of_range=[{"index": 3, "input_count": 1, "rule": None}],
            )
            return verdict, "raw"

        with extraction_trace() as trace:
            with patch("paramem.graph.stage_enrich.request_plausibility", side_effect=fake_plaus):
                result = self._run(
                    graph,
                    mapping,
                    anon_facts,
                    plausibility_judge="anthropic",
                    plausibility_stage="anon",
                )

        assert result.diagnostics["plausibility_dropped_anon"] == 1
        assert result.diagnostics["plausibility_dropped_anon_facts"][0]["rule"] == "R1"
        assert result.diagnostics["plausibility_out_of_range_anon"] == [
            {"index": 3, "input_count": 1, "rule": None}
        ]
        assert result.diagnostics["plausibility_state_anon"] == {"state": "ran", "reason": None}
        anon_plaus = next(p for p in trace.records if p.name == "anon_plausibility")
        assert anon_plaus.parsed["dropped_facts"][0]["rule"] == "R1"
        assert anon_plaus.parsed["out_of_range"] == [{"index": 3, "input_count": 1, "rule": None}]

    def test_anon_ran_unattributed_reaches_diagnostics_and_parsed(self):
        """Mirrors ``test_deanon_ran_unattributed_reaches_diagnostics_and_parsed``
        at the anon site."""
        graph, mapping, anon_facts = self._graph_and_mapping()

        def fake_plaus(facts, api_key, **kwargs):
            verdict = PlausibilityVerdict(
                kept=[],
                dropped=[{"index": 0, "rule": "R1", "fact": facts[0]}],
                out_of_range=[],
                unattributed=2,
            )
            return verdict, "raw"

        with extraction_trace() as trace:
            with patch("paramem.graph.stage_enrich.request_plausibility", side_effect=fake_plaus):
                result = self._run(
                    graph,
                    mapping,
                    anon_facts,
                    plausibility_judge="anthropic",
                    plausibility_stage="anon",
                )

        assert result.diagnostics["plausibility_unattributed_anon"] == 2
        assert isinstance(result.diagnostics["plausibility_dropped_anon_facts"][0]["rule"], str)
        anon_plaus = next(p for p in trace.records if p.name == "anon_plausibility")
        assert anon_plaus.parsed["unattributed"] == 2

    def test_anon_ran_arm_stop_at_returns_before_deanonymize(self):
        """``stop_at("anon_plausibility")`` must return right after the
        ``with phase_trace(...)`` block in the ran-arm —
        ``deanonymize``/``rebuild`` must not run. Drives the judge
        through its actual ran arm (a real ``request_plausibility`` call
        that returns successfully), not just the ``not_permitted`` skip
        branch that already had a phase-trace record for unrelated
        reasons."""
        from paramem.graph.phase_trace import stop_at

        graph, mapping, anon_facts = self._graph_and_mapping()

        def fake_plaus(facts, api_key, **kwargs):
            return _kept_verdict(facts), "raw"

        with extraction_trace() as trace:
            with (
                stop_at("anon_plausibility"),
                patch("paramem.graph.stage_enrich.request_plausibility", side_effect=fake_plaus),
            ):
                result = self._run(
                    graph,
                    mapping,
                    anon_facts,
                    plausibility_judge="anthropic",
                    plausibility_stage="anon",
                )

        phase_names = [p.name for p in trace.records]
        assert phase_names[-1] == "anon_plausibility", (
            f"stop_at('anon_plausibility') must stop the chain right after that "
            f"phase records; got {phase_names!r}"
        )
        assert "deanon" not in phase_names
        assert "deanon_plausibility" not in phase_names
        # The early return preserves the local-extract (real-name) graph —
        # deanonymize/rebuild never ran to overwrite it either way.
        assert result.relations[0].subject == "Alex"
