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
    _DEFAULT_ENRICHMENT_PROMPT,
    _DEFAULT_EXTRACTION_PROMPT,
    _DEFAULT_PLAUSIBILITY_PROMPT,
    _load_prompt,
    build_speaker_context,
)


def _render(template: str, **values) -> str:
    """Render a prompt template with placeholder values for inspection."""
    return template.format(**values)


class TestExtractionPrompt:
    def test_renders_with_speaker_context_empty(self):
        tmpl = _load_prompt("extraction.txt", _DEFAULT_EXTRACTION_PROMPT)
        rendered = tmpl.format(
            transcript="[user] hello", speaker_context=build_speaker_context(None)
        )
        assert "{transcript}" not in rendered
        assert "{speaker_context}" not in rendered
        # Empty speaker_context produces at most one blank-line gap, not three.
        assert "\n\n\n\n" not in rendered

    def test_renders_with_speaker_context_set(self):
        tmpl = _load_prompt("extraction.txt", _DEFAULT_EXTRACTION_PROMPT)
        rendered = tmpl.format(
            transcript="[user] hello", speaker_context=build_speaker_context("Alex")
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
        """Plausibility judge relies on numbered drop rules. Verify they exist."""
        tmpl = _load_prompt("sota_plausibility.txt", _DEFAULT_PLAUSIBILITY_PROMPT)
        required_rules = [
            "self-loop",
            "tautology",
            "role leak",
            "placeholder",
        ]
        for rule in required_rules:
            assert rule.lower() in tmpl.lower(), f"Plausibility prompt missing rule: {rule!r}"
