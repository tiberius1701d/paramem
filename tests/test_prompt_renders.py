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
        leftover = _LEFTOVER_PLACEHOLDER.findall(rendered)
        assert leftover == [], f"Leftover placeholders after render: {leftover}"

    def test_every_entity_type_in_rendered_output(self):
        _, prompt = load_extraction_prompts()
        rendered = prompt.format(
            transcript="sample",
            speaker_context="",
            entity_types=format_entity_types(),
            predicate_examples=format_predicate_examples(),
            relation_types=format_relation_types(),
        )
        for t in entity_types():
            assert t in rendered, f"Entity type {t!r} missing from rendered extraction prompt"

    def test_every_relation_type_in_rendered_output(self):
        """Every relation type from schema must appear in the rendered extraction prompt."""
        _, prompt = load_extraction_prompts()
        rendered = prompt.format(
            transcript="sample",
            speaker_context="",
            entity_types=format_entity_types(),
            predicate_examples=format_predicate_examples(),
            relation_types=format_relation_types(),
        )
        for rt in relation_types():
            assert rt in rendered, f"Relation type {rt!r} missing from rendered extraction prompt"


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
        leftover = _LEFTOVER_PLACEHOLDER.findall(rendered)
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
