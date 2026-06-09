"""Prompt-contract tests — verify the prompt files render correctly and the
documented contract hasn't drifted from the algorithms that depend on it.

The extraction pipeline's binding-recovery algorithm assumes a specific
convention in the SOTA enrichment prompt (existing bare placeholders stay
bare; only NEW entities get braces). A prompt edit that inverts this — e.g.
"always emit braced form" — silently breaks de-anonymization and is only
observed in a full sweep. These tests catch that class of regression at
unit-test time.
"""

from __future__ import annotations

import re

from paramem.graph.extractor import (
    _DEFAULT_ANONYMIZATION_PROMPT,
    _DEFAULT_ENRICHMENT_PROMPT,
    _DEFAULT_EXTRACTION_PROMPT,
    _DEFAULT_PLAUSIBILITY_PROMPT,
    _DEFAULT_PROCEDURAL_PROMPT,
    build_speaker_context,
    load_anonymization_prompt,
)
from paramem.graph.merger import _COEXISTENCE_PROMPT, _CONTRADICTION_PROMPT
from paramem.graph.prompts import _DEFAULT_PROMPT_DIR, _load_prompt
from paramem.graph.schema_config import (
    format_entity_types,
    format_predicate_examples,
    format_relation_types,
)


def _render(template: str, **values) -> str:
    """Render a prompt template with placeholder values for inspection."""
    return template.format(**values)


class TestExtractionPrompt:
    def test_renders_with_speaker_context_empty(self):
        tmpl = _load_prompt("extraction.txt", _DEFAULT_EXTRACTION_PROMPT)
        rendered = tmpl.format(
            transcript="[user] hello",
            speaker_context=build_speaker_context(None),
            entity_types=format_entity_types(),
            predicate_examples=format_predicate_examples(),
            relation_types=format_relation_types(),
        )
        assert "{transcript}" not in rendered
        assert "{speaker_context}" not in rendered
        # Empty speaker_context produces at most one blank-line gap, not three.
        assert "\n\n\n\n" not in rendered

    def test_renders_with_speaker_context_set(self):
        tmpl = _load_prompt("extraction.txt", _DEFAULT_EXTRACTION_PROMPT)
        rendered = tmpl.format(
            transcript="[user] hello",
            speaker_context=build_speaker_context("Alex"),
            entity_types=format_entity_types(),
            predicate_examples=format_predicate_examples(),
            relation_types=format_relation_types(),
        )
        assert "Alex" in rendered
        assert "'Alex'" in rendered


class TestEnrichmentPromptContract:
    def test_renders_without_format_errors(self):
        """No stray single-brace placeholders that collide with .format()."""
        tmpl = _load_prompt("sota_enrichment.txt", _DEFAULT_ENRICHMENT_PROMPT)
        # Must not raise KeyError — every literal brace pair escaped as
        # `{{` / `}}`.  ``str.format`` itself validates this.
        rendered = tmpl.format(transcript="Person_1 said hi.", facts_json="[]")
        # Rendered output should still contain the example braced tokens
        # (single-brace form after format-escape).
        assert "{Event_1}" in rendered or "{Prefix_N}" in rendered

    def test_preserves_bare_placeholder_convention(self):
        """The binding-recovery algorithm depends on SOTA leaving existing
        bare placeholders bare. Regressing the prompt to 'always emit braced'
        silently breaks de-anonymization.
        """
        tmpl = _load_prompt("sota_enrichment.txt", _DEFAULT_ENRICHMENT_PROMPT)
        # Must instruct model to leave existing bare placeholders bare.
        # Specifically: not to re-brace incoming Person_1/City_1 tokens.
        keywords = ["bare", "leave", "existing", "NOT re-brace"]
        hits = sum(1 for k in keywords if k.lower() in tmpl.lower())
        assert hits >= 2, (
            "Enrichment prompt must instruct SOTA to leave existing bare "
            "placeholders bare — otherwise binding recovery records self-"
            "referential junk entries (Person_2 → Person_2) and corrupts "
            "the reverse mapping. Found only keywords: "
            f"{[k for k in keywords if k.lower() in tmpl.lower()]}"
        )

    def test_requires_grounding_of_new_placeholders(self):
        """HARD REQUIREMENT: every braced placeholder used in `add` must
        have a matching entry in `bindings`. Without this clause, SOTA's
        reified entities are dropped wholesale by the residual sweep.

        Updated transcript is no longer carried on the wire (delta
        protocol — reconstructed locally from bindings + anon transcript)
        so the contract is now solely "facts ↔ bindings".
        """
        tmpl = _load_prompt("sota_enrichment.txt", _DEFAULT_ENRICHMENT_PROMPT)
        # Bindings is the grounding contract; the prompt must name it.
        assert "bindings" in tmpl
        # Look for a hard requirement that braced placeholders appear in
        # both `add` (facts) and `bindings`.  Phrasing is free to evolve;
        # the structural claim is not.
        assert re.search(r"MUST.*appear|appear.*MUST", tmpl, re.IGNORECASE), (
            "Enrichment prompt must contain a hard requirement that new "
            "braced placeholders appear in both `add` and `bindings`."
        )
        # No transcript echo on the wire — `updated_transcript` must not
        # appear as an output key (catches accidental reintroduction).
        assert "updated_transcript" not in tmpl, (
            "Enrichment prompt must not request `updated_transcript` in "
            "the output — the transcript is reconstructed locally from "
            "bindings to keep output bandwidth bounded."
        )

    def test_teaches_role_instance_aggregation(self):
        """The brace-binding section must show a role-instance POSITIVE
        example that aggregates multiple co-temporal attributes (title,
        company, location, dates) onto a single bound entity rather
        than flattening them as independent triples on the speaker.

        Without this teaching, SOTA emits the bound title once but
        leaves dates / company / location as orphan triples on the
        speaker.  Downstream reasoning over multi-role chronology
        ("what title did the speaker hold in 2015?") then fails because
        co-temporal facts cannot be paired back to a role.

        Empirical evidence pre-fix: zero ``Role_*`` entities across 24+
        production graph snapshots (data/ha/debug/run_*/), even though
        the brace-binding contract itself is honoured for ``Event_*``.
        """
        tmpl = _load_prompt("sota_enrichment.txt", _DEFAULT_ENRICHMENT_PROMPT)
        # Structural assertion: a Role_N braced placeholder must appear
        # in a positive-example block alongside multiple bound facts —
        # at minimum a date attribute and a company/location attribute.
        assert "{{Role_1}}" in tmpl, (
            "Enrichment prompt must include a Role_1 example to teach role-instance aggregation."
        )
        # Co-temporal attributes must be bound to {{Role_1}} (subject
        # position).  A flat-triples regression would have them on
        # Person_1 instead.  The delta protocol uses JSON-shaped facts
        # (``"subject": "{{Role_1}}", "predicate": "start_date"``);
        # match either ordering of those two key/value pairs within a
        # short window so the test is robust to minor reformatting.
        assert re.search(
            r'"\{\{Role_1\}\}"[^{}]{0,200}start_date'
            r"|"
            r'start_date[^{}]{0,200}"\{\{Role_1\}\}"',
            tmpl,
        ), (
            "Role example must show start_date with {{Role_1}} as subject — "
            "the structural teaching is that dates bind to the role-instance, "
            "not to the speaker."
        )
        # The NEGATIVE block must spell out the speaker-flattening anti-
        # pattern so a future prompt edit cannot keep the POSITIVE
        # example while quietly removing the warning.
        assert re.search(r"WRONG.*speaker|flat.*speaker|speaker.*flat", tmpl, re.IGNORECASE), (
            "Enrichment prompt must call out the flat-triples-on-speaker "
            "anti-pattern in a NEGATIVE block."
        )


class TestPlausibilityPromptContract:
    def test_renders_without_format_errors(self):
        tmpl = _load_prompt("sota_plausibility.txt", _DEFAULT_PLAUSIBILITY_PROMPT)
        rendered = tmpl.format(transcript="Person_1 said hi.", facts_json="[]")
        assert "{transcript}" not in rendered
        assert "{facts_json}" not in rendered

    def test_lists_drop_rules(self):
        """Plausibility judge relies on six numbered drop rules (R1-R6).

        The prior R4 ("Unresolved placeholder in real-name input") was
        tied to the constrained ``^(Person|City|Country|Org|Thing)_\\d+$``
        regex that became incoherent with the open-vocabulary anonymizer
        pivot.  It was also structurally redundant with
        ``_strip_residual_placeholders`` running inside ``_apply_bindings``
        at the deanon stage, before plausibility.

        The grounding refactor revised the remaining rules: lexical
        token lists became illustrative parentheticals, and a new R3
        (transcript contradiction) closes the gap that the prior
        "no judgment calls" framing left open.  The structure of the
        prompt and of the examples is unchanged.

        Verify each rule's identifying substring still exists so a
        prompt edit that removes a rule is caught at unit-test time.
        """
        tmpl = _load_prompt("sota_plausibility.txt", _DEFAULT_PLAUSIBILITY_PROMPT)
        required_rules = [
            "self-loop",  # R1
            "name-swap",  # R2
            "contradiction",  # R3 — new transcript-grounded rule
            "conversation-role",  # R4 — was "Role leak", grounded
            "content-free",  # R5 — was "Empty / sentinel"
            "system identifier",  # R6 — was "System entity ID"
        ]
        for rule in required_rules:
            assert rule.lower() in tmpl.lower(), f"Plausibility prompt missing rule: {rule!r}"

    def test_keep_default_disposition(self):
        """The prompt uses a keep-by-default model. Regressing to drop-by-default
        silently discards valid facts — a data-loss bug that only surfaces
        during a full extraction sweep. This assertion guards that semantic flip.

        The contract is structural, not literal: the default disposition must
        be to keep the fact (IGNORE — keep unless a drop rule matches), it must
        be declared as the default, and it must appear before the drop rules.
        Surface wording — "Default action", "IGNORE", a "## KEEP" header — is
        free to evolve; the keep-by-default semantics and the ordering are not.
        """
        tmpl = _load_prompt("sota_plausibility.txt", _DEFAULT_PLAUSIBILITY_PROMPT)
        lower = tmpl.lower()
        # The default action must KEEP the fact (IGNORE), not drop it.
        assert "default action" in lower, (
            "Plausibility prompt must declare a default action (keep-by-default). "
            "Removing this primes the model to drop on judgment, causing silent data loss."
        )
        assert "ignore" in lower or "keep the fact" in lower, (
            "Default disposition must keep the fact (IGNORE / keep), not drop."
        )
        # The keep-by-default declaration must precede the drop rules.
        default_idx = lower.find("default action")
        rules_idx = lower.find("## drop")
        assert rules_idx >= 0, "Plausibility prompt missing the drop-rules section header."
        assert 0 <= default_idx < rules_idx, (
            "Keep-by-default disposition must be declared before the drop rules — "
            "order encodes default disposition."
        )

    def test_output_contract_is_drop_index_set(self):
        """The plausibility judge's output protocol is a small JSON object
        ``{"drop": [<index>, ...]}`` listing which input facts to drop —
        NOT an echo of every kept fact.

        Why this is structural, not stylistic: the previous "echo every
        kept fact" contract had Mistral 7B emit EOS mid-array on long
        inputs (the closing ``]`` never arrived; ``_parse_facts_response``
        couldn't recover the envelope; the gate fail-opened with 0 facts
        filtered).  The drop-set output is bounded by the count of
        actual rule matches — typically 0–5 indices for clean inputs —
        so truncation cannot kill the gate.

        Regressing to "echo every fact" silently re-introduces the
        truncation failure mode, so this assertion locks the contract.
        """
        tmpl = _load_prompt("sota_plausibility.txt", _DEFAULT_PLAUSIBILITY_PROMPT)
        # Must specify the drop-index-set object shape.
        assert '"drop"' in tmpl, (
            'Plausibility prompt must specify the drop-set output shape ({"drop": [<index>, ...]}).'
        )
        # Must describe the index-based reference convention.
        assert "zero-based" in tmpl.lower() and "index" in tmpl.lower(), (
            "Plausibility prompt must teach zero-based index references — the "
            "judge needs to know how facts are numbered to refer to them."
        )
        # Must forbid echoing kept facts (the regression vector).
        forbids_echo = re.search(
            r"do not (echo|include the facts|return surviving|emit the surviving)",
            tmpl,
            re.IGNORECASE,
        )
        assert forbids_echo, (
            "Plausibility prompt must explicitly forbid echoing the kept facts — "
            "without this the model defaults to verbose echo and triggers the "
            "Mistral-7B EOS-mid-array truncation."
        )

    def test_inline_default_matches_file(self):
        """The inline ``_DEFAULT_PLAUSIBILITY_PROMPT`` is only used when the
        file is missing. If it drifts from the file, production behaviour
        silently flips on any deployment that loses ``configs/prompts/``.
        Render both with the same inputs and require byte equality.
        """
        from pathlib import Path

        file_tmpl = Path("configs/prompts/sota_plausibility.txt").read_text()
        args = {"transcript": "Person_1 said hi.", "facts_json": "[]"}
        assert _DEFAULT_PLAUSIBILITY_PROMPT.format(**args) == file_tmpl.format(**args)


class TestProceduralPrompt:
    def test_renders_with_speaker_context_empty(self):
        """Procedural prompt renders without errors when speaker is unknown.

        Verifies that the {speaker_context} placeholder is present in the
        file-based prompt and collapses cleanly to an empty string so no
        dangling placeholder or extra blank lines remain.
        """
        tmpl = _load_prompt("extraction_procedural.txt", _DEFAULT_PROCEDURAL_PROMPT)
        rendered = tmpl.format(
            transcript="[user] Play some jazz.",
            speaker_context=build_speaker_context(None),
            entity_types=format_entity_types(scope="procedural"),
            predicate_examples=format_predicate_examples(scope="procedural"),
        )
        assert "{transcript}" not in rendered
        assert "{speaker_context}" not in rendered
        # Empty speaker_context produces at most one blank-line gap, not three.
        assert "\n\n\n\n" not in rendered

    def test_renders_with_speaker_context_set(self):
        """Procedural prompt injects real speaker name when provided.

        Guards against the silent identity fragmentation bug where
        procedural facts get subject "Speaker" while main-extraction
        facts use the real name, creating two nodes for the same person.
        """
        tmpl = _load_prompt("extraction_procedural.txt", _DEFAULT_PROCEDURAL_PROMPT)
        rendered = tmpl.format(
            transcript="[user] Play some jazz.",
            speaker_context=build_speaker_context("Alex"),
            entity_types=format_entity_types(scope="procedural"),
            predicate_examples=format_predicate_examples(scope="procedural"),
        )
        assert "Alex" in rendered
        assert "'Alex'" in rendered


class TestAnonymizationPrompt:
    """Contract tests for the anonymization prompt — both default and file-based.

    The prompt teaches a shape contract (PascalCase prefix + ``_<N>``,
    uniqueness, totality, direction), not a constrained vocabulary.  The
    earlier ``replacement_rules`` interpolation that listed the configured
    prefixes is gone — prefixes are illustrative-only inside the prompt
    body, and the model picks any type-appropriate PascalCase prefix.
    """

    def _render(self, tmpl: str) -> str:
        """Render the anonymization prompt with all expected kwargs."""
        return tmpl.format(
            facts_json='[{"subject": "Person_1", "predicate": "lives_in", '
            '"object": "City_1", "relation_type": "factual", "confidence": 0.9}]',
            transcript="Person_1 lives in City_1.",
        )

    def test_default_renders_without_format_errors(self):
        """_DEFAULT_ANONYMIZATION_PROMPT must render with all expected kwargs without KeyError."""
        rendered = self._render(_DEFAULT_ANONYMIZATION_PROMPT)
        assert "{facts_json}" not in rendered
        assert "{transcript}" not in rendered

    def test_file_based_renders_without_format_errors(self):
        """File-based anonymization.txt must render with all expected kwargs without KeyError."""
        tmpl = load_anonymization_prompt()
        rendered = self._render(tmpl)
        assert "{facts_json}" not in rendered
        assert "{transcript}" not in rendered

    def test_shape_contract_present_in_default(self):
        """The default prompt must teach the four parts of the shape contract:
        well-formed shape, uniqueness, totality, direction."""
        rendered = self._render(_DEFAULT_ANONYMIZATION_PROMPT)
        # Shape clause — `<Prefix>_<N>` or equivalent shape language.
        assert "PascalCase" in rendered or "Prefix" in rendered, (
            "Default prompt must teach the placeholder shape (PascalCase + _<N>)."
        )
        # Uniqueness clause.
        assert "UNIQUE" in rendered or "unique" in rendered
        # Totality clause.
        assert "totality" in rendered.lower() or "every placeholder" in rendered.lower()
        # Direction clause.
        assert "real_name" in rendered or "real name" in rendered.lower()

    def test_shape_contract_present_in_file_based(self):
        """The file-based prompt must teach the same four-part shape contract.

        Verifies the file isn't accidentally pinned to a constrained vocabulary
        (the regression that prompted this rewrite — the model invented prefixes
        like ``University_1`` / ``Project_1`` and the recovery helper had to
        patch the gap).
        """
        tmpl = load_anonymization_prompt()
        rendered = self._render(tmpl)
        assert "PascalCase" in rendered or "Prefix" in rendered
        assert "UNIQUE" in rendered or "unique" in rendered
        assert "totality" in rendered.lower() or "every placeholder" in rendered.lower()
        assert "real_name" in rendered or "real name" in rendered.lower()
        # Diverse-prefix examples present — illustrative breadth signals
        # the model that prefixes outside the common set are valid.
        assert "University" in rendered or "Project" in rendered or "Product" in rendered, (
            "File-based prompt must show at least one type-appropriate prefix "
            "outside the common {Person, City, Country, Org, Thing} set so the "
            "model knows the prefix vocabulary is open."
        )

    def test_no_stray_unescaped_placeholders_in_default(self):
        """After rendering, no stray {word} tokens should remain (only JSON literal braces)."""
        import re

        rendered = self._render(_DEFAULT_ANONYMIZATION_PROMPT)
        # JSON literal braces are escaped as {{ }} in the template and appear as { } after render.
        # A simple check: no single { immediately followed by a letter (unrendered placeholder).
        stray = re.findall(r"(?<!\{)\{[a-z_]+\}", rendered)
        assert not stray, f"Stray unrendered placeholder(s) found in rendered prompt: {stray!r}"


class TestMergerCoexistencePrompt:
    """Contract tests for the merger coexistence prompt.

    The 2-way parser in ``check_predicate_coexistence`` keys on the literal
    strings ``COEXIST`` and ``REPLACE``.  The prompt must contain both keywords
    and must render without leftover slots.
    """

    def test_renders_without_leftover_slots(self):
        """All four slots fill without KeyError; no leftover ``{slot}`` tokens."""
        tmpl = _load_prompt("merger_coexistence.txt", _COEXISTENCE_PROMPT)
        rendered = tmpl.format(
            predicate="owns_pet",
            subject="Alex",
            old_value="a cat",
            new_value="a dog",
        )
        assert "{predicate}" not in rendered
        assert "{subject}" not in rendered
        assert "{old_value}" not in rendered
        assert "{new_value}" not in rendered

    def test_coexist_keyword_present(self):
        """``COEXIST`` must survive rendering — the parser keys on this literal."""
        tmpl = _load_prompt("merger_coexistence.txt", _COEXISTENCE_PROMPT)
        rendered = tmpl.format(
            predicate="owns_pet",
            subject="Alex",
            old_value="a cat",
            new_value="a dog",
        )
        assert "COEXIST" in rendered

    def test_replace_keyword_present(self):
        """``REPLACE`` must survive rendering — the parser keys on this literal."""
        tmpl = _load_prompt("merger_coexistence.txt", _COEXISTENCE_PROMPT)
        rendered = tmpl.format(
            predicate="owns_pet",
            subject="Alex",
            old_value="a cat",
            new_value="a dog",
        )
        assert "REPLACE" in rendered

    def test_aggregate_keyword_absent(self):
        """``AGGREGATE`` must NOT appear in the rendered prompt — the 2-way parser
        no longer expects or emits it; its presence would confuse the model.
        """
        tmpl = _load_prompt("merger_coexistence.txt", _COEXISTENCE_PROMPT)
        rendered = tmpl.format(
            predicate="speaks",
            subject="Alex",
            old_value="German",
            new_value="English",
        )
        assert "AGGREGATE" not in rendered, (
            "Prompt must not contain AGGREGATE — fold is now purely additive"
        )

    def test_file_byte_equivalent_to_inline_default(self):
        """The prompt file must be byte-equivalent to ``_COEXISTENCE_PROMPT`` after
        ``.strip()`` so the fallback and the file produce identical model inputs.
        """
        file_path = _DEFAULT_PROMPT_DIR / "merger_coexistence.txt"
        assert file_path.exists(), f"Prompt file not found: {file_path}"
        assert file_path.read_text().strip() == _COEXISTENCE_PROMPT


class TestCheckPredicateCoexistenceParser:
    """Unit tests for the 2-way verdict parser in check_predicate_coexistence.

    These tests mock the model's generate_answer to return specific output
    strings and verify the parser extracts the correct verdict.
    They do NOT require a real GPU or model.
    """

    def _call_with_mock_output(self, output: str) -> str:
        """Drive check_predicate_coexistence with a mocked model output.

        ``generate_answer`` and ``adapt_messages`` are imported locally inside
        ``check_predicate_coexistence`` (lazy import), so we patch them at their
        definition site (``paramem.evaluation.recall`` and
        ``paramem.models.loader``), not via the merger module namespace.
        """
        from unittest.mock import MagicMock, patch

        from paramem.graph.merger import check_predicate_coexistence

        model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "formatted"

        with patch("paramem.evaluation.recall.generate_answer", return_value=output):
            with patch(
                "paramem.models.loader.adapt_messages",
                return_value=[{"role": "user", "content": "test"}],
            ):
                return check_predicate_coexistence(
                    "Alex", "speaks", "German", "English", model, tokenizer
                )

    def test_coexist_verdict_parsed(self):
        """Model output 'COEXIST' → 'COEXIST'."""
        verdict = self._call_with_mock_output("COEXIST")
        assert verdict == "COEXIST"

    def test_replace_verdict_parsed(self):
        """Model output 'REPLACE' → 'REPLACE'."""
        verdict = self._call_with_mock_output("REPLACE")
        assert verdict == "REPLACE"

    def test_ambiguous_output_defaults_to_coexist(self):
        """Unrecognised model output → 'COEXIST' safer default."""
        verdict = self._call_with_mock_output("MAYBE")
        assert verdict == "COEXIST"


class TestMergerContradictionPrompt:
    """Contract tests for the merger contradiction prompt.

    The parser in ``detect_contradiction_with_model`` keys on the literal strings
    ``CONTRADICTS`` and ``NO_CONTRADICTION`` (``merger.py`` line that checks
    ``output.startswith("CONTRADICTS")``).  Removing either keyword from the
    prompt would silently break contradiction detection.
    """

    def test_renders_without_leftover_slots(self):
        """All four slots fill without KeyError; no leftover ``{slot}`` tokens."""
        tmpl = _load_prompt("merger_contradiction.txt", _CONTRADICTION_PROMPT)
        rendered = tmpl.format(
            existing_facts="- Alex | lives_in | Munich",
            subject="Alex",
            predicate="moved_to",
            object="Berlin",
        )
        assert "{existing_facts}" not in rendered
        assert "{subject}" not in rendered
        assert "{predicate}" not in rendered
        assert "{object}" not in rendered

    def test_contradicts_keyword_present(self):
        """``CONTRADICTS`` must survive rendering — the parser keys on this literal."""
        tmpl = _load_prompt("merger_contradiction.txt", _CONTRADICTION_PROMPT)
        rendered = tmpl.format(
            existing_facts="- Alex | lives_in | Munich",
            subject="Alex",
            predicate="moved_to",
            object="Berlin",
        )
        assert "CONTRADICTS" in rendered

    def test_no_contradiction_keyword_present(self):
        """``NO_CONTRADICTION`` must survive rendering — the parser keys on this literal."""
        tmpl = _load_prompt("merger_contradiction.txt", _CONTRADICTION_PROMPT)
        rendered = tmpl.format(
            existing_facts="- Alex | lives_in | Munich",
            subject="Alex",
            predicate="moved_to",
            object="Berlin",
        )
        assert "NO_CONTRADICTION" in rendered

    def test_file_byte_equivalent_to_inline_default(self):
        """The prompt file must be byte-equivalent to ``_CONTRADICTION_PROMPT`` after
        ``.strip()`` so the fallback and the file produce identical model inputs.
        """
        file_path = _DEFAULT_PROMPT_DIR / "merger_contradiction.txt"
        assert file_path.exists(), f"Prompt file not found: {file_path}"
        assert file_path.read_text().strip() == _CONTRADICTION_PROMPT
