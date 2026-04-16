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
        # Must not raise KeyError — all literal braces escaped as {{ }}.
        rendered = tmpl.format(transcript="Person_1 said hi.", facts_json="[]")
        # Rendered output should still contain the example braced tokens.
        assert "{Event_1}" in rendered or "{Prefix_N}" in rendered
        # And must not contain unescaped literals left over.
        assert "{{" not in rendered
        assert "}}" not in rendered

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
        """HARD REQUIREMENT: every placeholder in facts must appear in the
        updated transcript. Without this clause, SOTA's reified entities
        are dropped wholesale by the residual sweep.
        """
        tmpl = _load_prompt("sota_enrichment.txt", _DEFAULT_ENRICHMENT_PROMPT)
        # Looser check — any phrasing that requires both sides match.
        assert "updated_transcript" in tmpl
        assert re.search(r"MUST.*appear|appear.*MUST", tmpl, re.IGNORECASE), (
            "Enrichment prompt must contain a hard requirement that new "
            "placeholders appear in both facts and updated_transcript."
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
        """
        tmpl = _load_prompt("sota_plausibility.txt", _DEFAULT_PLAUSIBILITY_PROMPT)
        assert "default: keep" in tmpl.lower(), (
            "Plausibility prompt must declare 'Default: KEEP' disposition. "
            "Removing or inverting this causes silent data loss — dropped facts "
            "cannot be recovered from subsequent sessions."
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
