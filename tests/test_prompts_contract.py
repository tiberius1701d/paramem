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
    _load_prompt,
    build_speaker_context,
    load_anonymization_prompt,
)
from paramem.graph.schema_config import (
    format_entity_types,
    format_predicate_examples,
    format_relation_types,
    format_replacement_rules,
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
        """Plausibility judge relies on six numbered drop rules (R1-R6). Verify they
        all exist so a prompt edit that removes a rule is caught at unit-test time.
        """
        tmpl = _load_prompt("sota_plausibility.txt", _DEFAULT_PLAUSIBILITY_PROMPT)
        required_rules = [
            "self-loop",  # R1
            "name-swap",  # R2
            "role leak",  # R3
            "placeholder",  # R4
            "sentinel",  # R5
            "system entity",  # R6
        ]
        for rule in required_rules:
            assert rule.lower() in tmpl.lower(), f"Plausibility prompt missing rule: {rule!r}"

    def test_keep_default_disposition(self):
        """The prompt uses a KEEP-by-default model. Regressing to DROP-by-default
        silently discards valid facts — a data-loss bug that only surfaces
        during a full extraction sweep. This assertion guards that semantic flip.

        The contract is structural, not literal: the KEEP section must
        appear before DROP, and the KEEP section body must declare a
        default-disposition (i.e. KEEP applies unless a DROP rule matches).
        Surface wording — "Default: KEEP", "Default action", etc. — is
        free to evolve; the structural ordering and semantic claim are not.
        """
        tmpl = _load_prompt("sota_plausibility.txt", _DEFAULT_PLAUSIBILITY_PROMPT)
        lower = tmpl.lower()
        keep_idx = lower.find("## keep")
        drop_idx = lower.find("## drop")
        assert keep_idx >= 0, "Plausibility prompt missing '## KEEP' section header."
        assert drop_idx >= 0, "Plausibility prompt missing '## DROP' section header."
        assert keep_idx < drop_idx, (
            "KEEP section must precede DROP section — order encodes default disposition."
        )
        keep_section = lower[keep_idx:drop_idx]
        assert "default" in keep_section, (
            "KEEP section must declare default disposition (e.g. 'Default action', "
            "'Default: KEEP'). Removing this primes the model to drop on judgment, "
            "causing silent data loss."
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
    """Contract tests for the anonymization prompt — both default and file-based."""

    def _render(self, tmpl: str) -> str:
        """Render the anonymization prompt with all expected kwargs."""
        return tmpl.format(
            facts_json='[{"subject": "Person_1", "predicate": "lives_in", '
            '"object": "City_1", "relation_type": "factual", "confidence": 0.9}]',
            transcript="Person_1 lives in City_1.",
            replacement_rules=format_replacement_rules(),
        )

    def test_default_renders_without_format_errors(self):
        """_DEFAULT_ANONYMIZATION_PROMPT must render with all expected kwargs without KeyError."""
        rendered = self._render(_DEFAULT_ANONYMIZATION_PROMPT)
        assert "{facts_json}" not in rendered
        assert "{transcript}" not in rendered
        assert "{replacement_rules}" not in rendered

    def test_file_based_renders_without_format_errors(self):
        """File-based anonymization.txt must render with all expected kwargs without KeyError."""
        tmpl = load_anonymization_prompt()
        rendered = self._render(tmpl)
        assert "{replacement_rules}" not in rendered

    def test_replacement_rules_present_in_default(self):
        """All five configured prefixes must appear in the rendered default prompt."""
        rendered = self._render(_DEFAULT_ANONYMIZATION_PROMPT)
        for prefix in ("Person", "City", "Country", "Org", "Thing"):
            assert prefix in rendered, (
                f"Prefix {prefix!r} missing from rendered _DEFAULT_ANONYMIZATION_PROMPT."
            )

    def test_replacement_rules_present_in_file_based(self):
        """All five configured prefixes must appear in the rendered file-based prompt."""
        tmpl = load_anonymization_prompt()
        rendered = self._render(tmpl)
        for prefix in ("Person", "City", "Country", "Org", "Thing"):
            assert prefix in rendered, (
                f"Prefix {prefix!r} missing from rendered file-based anonymization.txt."
            )

    def test_no_stray_unescaped_placeholders_in_default(self):
        """After rendering, no stray {word} tokens should remain (only JSON literal braces)."""
        import re

        rendered = self._render(_DEFAULT_ANONYMIZATION_PROMPT)
        # JSON literal braces are escaped as {{ }} in the template and appear as { } after render.
        # A simple check: no single { immediately followed by a letter (unrendered placeholder).
        stray = re.findall(r"(?<!\{)\{[a-z_]+\}", rendered)
        assert not stray, f"Stray unrendered placeholder(s) found in rendered prompt: {stray!r}"
