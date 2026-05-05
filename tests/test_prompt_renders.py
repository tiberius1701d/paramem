"""Round-trip render tests — verify extraction prompt files render without errors.

Ensures that every placeholder resolves and no leftover ``{word}`` tokens
remain after formatting with the values supplied by schema_config.
JSON literal braces are double-escaped in the prompt files (``{{`` / ``}}``)
so they collapse to ``{`` / ``}`` after formatting and are not matched by the
leftover-placeholder check.
"""

from __future__ import annotations

import re

from paramem.graph.extractor import load_extraction_prompts, load_procedural_prompt
from paramem.graph.schema_config import (
    entity_types,
    format_entity_types,
    format_predicate_examples,
    format_relation_types,
    relation_types,
    reset_cache,
)

# Matches single-brace placeholders like {transcript} that are NOT part of
# a double-brace escape ({{ or }}).  After a successful .format() call all
# such tokens must be gone.
_LEFTOVER_PLACEHOLDER = re.compile(r"(?<!\{)\{[A-Za-z_]+\}(?!\})")

# `{SPEAKER_NAME}` appears literally in the rendered prompt: the few-shot
# examples show the subject slot wrapped in template-variable syntax so the
# model recognises it as a substitution target rather than a literal name.
# It is produced by writing ``{{SPEAKER_NAME}}`` in the prompt source.
_INTENTIONAL_LITERALS = {"{SPEAKER_NAME}"}


class TestExtractionPromptRender:
    def setup_method(self):
        reset_cache()

    def test_renders_without_exception(self):
        _, prompt = load_extraction_prompts()
        # Must not raise KeyError or IndexError.
        rendered = prompt.format(
            transcript="sample",
            speaker_context="",
            entity_types=format_entity_types(),
            predicate_examples=format_predicate_examples(),
            relation_types=format_relation_types(),
        )
        assert isinstance(rendered, str)

    def test_no_leftover_placeholders(self):
        _, prompt = load_extraction_prompts()
        rendered = prompt.format(
            transcript="sample",
            speaker_context="",
            entity_types=format_entity_types(),
            predicate_examples=format_predicate_examples(),
            relation_types=format_relation_types(),
        )
        leftover = [
            m for m in _LEFTOVER_PLACEHOLDER.findall(rendered) if m not in _INTENTIONAL_LITERALS
        ]
        assert leftover == [], f"Leftover placeholders after render: {leftover}"

    def test_example_entity_types_are_within_schema(self):
        """Inverse of the old "every type must appear" check.

        The new examples-only architecture (README → Prompt Engineering)
        deliberately drops the ``{entity_types}`` slot — verbatim
        taxonomy listings empirically license Mistral 7B to extend the
        closed set with invented type names.  Schema coverage is now
        carried by the few-shot examples.

        The remaining invariant — guarded here — is that every
        ``entity_type: "<X>"`` literal that appears in the prompt examples
        must be a value in the schema's allowed set.  Catches off-list
        drift if someone authors an example with
        ``entity_type: "phone_number"`` etc.
        """
        _, prompt = load_extraction_prompts()
        rendered = prompt.format(
            transcript="sample",
            speaker_context="",
            entity_types=format_entity_types(),
            predicate_examples=format_predicate_examples(),
            relation_types=format_relation_types(),
        )
        allowed = set(entity_types())
        used = set(re.findall(r'"entity_type":\s*"([^"]+)"', rendered))
        # The {SPEAKER_NAME} placeholder substitution leaves no entity_type
        # tokens — every match in the rendered output is an example literal.
        offenders = used - allowed
        assert not offenders, (
            f"Off-schema entity_types in prompt examples: {sorted(offenders)}. "
            f"Allowed: {sorted(allowed)}. "
            f"Add the new type to configs/schema.yaml or correct the example."
        )

    def test_example_relation_types_are_within_schema(self):
        """Inverse of the old "every relation_type must appear" check —
        same rationale as :meth:`test_example_entity_types_are_within_schema`.
        """
        _, prompt = load_extraction_prompts()
        rendered = prompt.format(
            transcript="sample",
            speaker_context="",
            entity_types=format_entity_types(),
            predicate_examples=format_predicate_examples(),
            relation_types=format_relation_types(),
        )
        allowed = set(relation_types())
        used = set(re.findall(r'"relation_type":\s*"([^"]+)"', rendered))
        offenders = used - allowed
        assert not offenders, (
            f"Off-schema relation_types in prompt examples: {sorted(offenders)}. "
            f"Allowed: {sorted(allowed)}. "
            f"Add the new type to configs/schema.yaml or correct the example."
        )


class TestProceduralPromptRender:
    def setup_method(self):
        reset_cache()

    def test_renders_without_exception(self):
        _, prompt = load_procedural_prompt()
        rendered = prompt.format(
            transcript="sample",
            speaker_context="",
            entity_types=format_entity_types(scope="procedural"),
            predicate_examples=format_predicate_examples(scope="procedural"),
        )
        assert isinstance(rendered, str)

    def test_no_leftover_placeholders(self):
        _, prompt = load_procedural_prompt()
        rendered = prompt.format(
            transcript="sample",
            speaker_context="",
            entity_types=format_entity_types(scope="procedural"),
            predicate_examples=format_predicate_examples(scope="procedural"),
        )
        leftover = [
            m for m in _LEFTOVER_PLACEHOLDER.findall(rendered) if m not in _INTENTIONAL_LITERALS
        ]
        assert leftover == [], f"Leftover placeholders after render: {leftover}"

    def test_procedural_entity_types_in_rendered_output(self):
        _, prompt = load_procedural_prompt()
        rendered = prompt.format(
            transcript="sample",
            speaker_context="",
            entity_types=format_entity_types(scope="procedural"),
            predicate_examples=format_predicate_examples(scope="procedural"),
        )
        assert "person" in rendered
        assert "preference" in rendered
