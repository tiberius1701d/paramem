"""Round-trip render tests — verify extraction prompt files render without errors.

Ensures that every placeholder resolves and no leftover ``{word}`` tokens
remain after formatting with the values supplied by paramem.config.taxonomy.
JSON literal braces are double-escaped in the prompt files (``{{`` / ``}}``)
so they collapse to ``{`` / ``}`` after formatting and are not matched by the
leftover-placeholder check.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from paramem.config.taxonomy import (
    entity_types,
    relation_types,
    reset_cache,
)
from paramem.graph.extractor import (
    DEFAULT_PROCEDURAL_USER_PROMPT_FILENAME,
    load_extraction_prompts,
)
from paramem.graph.prompts import _DEFAULT_PROMPT_DIR, _load_prompt  # was load_anonymization_prompt

# Matches single-brace placeholders like {transcript} that are NOT part of
# a double-brace escape ({{ or }}).  After a successful .format() call all
# such tokens must be gone.
_LEFTOVER_PLACEHOLDER = re.compile(r"(?<!\{)\{[A-Za-z_]+\}(?!\})")

# `{SPEAKER_NAME}` appears literally in the rendered prompt: the few-shot
# examples show the subject slot wrapped in template-variable syntax so the
# model recognises it as a substitution target rather than a literal name.
# It is produced by writing ``{{SPEAKER_NAME}}`` in the prompt source.
#
# `{N}` appears literally in anonymization.txt: it documents the
# placeholder shape (`<Prefix>_<N>`) and the `speaker{N}` id pattern —
# both produced by writing ``{{N}}`` in the prompt source, same mechanism
# as `{SPEAKER_NAME}` above.
_INTENTIONAL_LITERALS = {"{SPEAKER_NAME}", "{N}"}


class TestExtractionPromptRender:
    def setup_method(self):
        reset_cache()

    def test_renders_without_exception(self):
        _, prompt = load_extraction_prompts()
        # Must not raise KeyError or IndexError.
        rendered = prompt.format(
            transcript="sample",
            speaker_context="",
        )
        assert isinstance(rendered, str)

    def test_no_leftover_placeholders(self):
        _, prompt = load_extraction_prompts()
        rendered = prompt.format(
            transcript="sample",
            speaker_context="",
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
        _, prompt = load_extraction_prompts(user_filename=DEFAULT_PROCEDURAL_USER_PROMPT_FILENAME)
        rendered = prompt.format(
            transcript="sample",
            speaker_context="",
        )
        assert isinstance(rendered, str)

    def test_no_leftover_placeholders(self):
        _, prompt = load_extraction_prompts(user_filename=DEFAULT_PROCEDURAL_USER_PROMPT_FILENAME)
        rendered = prompt.format(
            transcript="sample",
            speaker_context="",
        )
        leftover = [
            m for m in _LEFTOVER_PLACEHOLDER.findall(rendered) if m not in _INTENTIONAL_LITERALS
        ]
        assert leftover == [], f"Leftover placeholders after render: {leftover}"

    def test_procedural_entity_types_in_rendered_output(self):
        _, prompt = load_extraction_prompts(user_filename=DEFAULT_PROCEDURAL_USER_PROMPT_FILENAME)
        rendered = prompt.format(
            transcript="sample",
            speaker_context="",
        )
        assert "person" in rendered
        assert "preference" in rendered


class TestSecondOrderExtractionPromptRender:
    """Render-with-slots guard for ``extraction_second_order.txt`` — the
    ``second_order_extract`` phase's user template.

    ``{named_people}`` is the gate-derived closed target set threaded via
    ``extra_slots`` (:func:`paramem.graph.flows._stage_second_order_extract`).
    An un-threaded ``{named_people}`` slot is a runtime ``KeyError`` CI
    cannot otherwise see, so this mirrors the ``{speaker_id}`` render-with-
    slots guard already established for ``cloud_enrichment.txt``
    (``TestEnrichmentPromptContract.test_omitted_speaker_id_slot_raises_key_error``).
    The phase's SYSTEM prompt is the shared, slot-free
    ``extraction_second_order.txt``'s
    ``DEFAULT_SYSTEM_PROMPT_FILENAME`` (``extraction_system.txt`` — no
    dedicated second-order system prompt); its no-braces contract is
    already covered by ``test_extraction_system_txt_no_braces`` in
    ``tests/test_prompts_present.py``.
    """

    def _load(self):
        return _load_prompt("extraction_second_order.txt")

    def _render(self):
        tmpl = self._load()
        return tmpl.format(
            transcript="sample",
            speaker_context="",
            named_people="Dana, Riley",
        )

    def test_renders_without_exception(self):
        rendered = self._render()
        assert isinstance(rendered, str)
        assert "Dana, Riley" in rendered

    def test_no_leftover_placeholders(self):
        rendered = self._render()
        leftover = [
            m for m in _LEFTOVER_PLACEHOLDER.findall(rendered) if m not in _INTENTIONAL_LITERALS
        ]
        assert leftover == [], f"Leftover placeholders after render: {leftover}"

    def test_omitted_named_people_slot_raises_key_error(self):
        """An un-threaded ``{named_people}`` slot must fail loudly at
        ``.format()`` time, not silently leak the literal ``{named_people}``
        into the model prompt."""
        tmpl = self._load()
        with pytest.raises(KeyError):
            tmpl.format(transcript="sample", speaker_context="")


class TestSecondOrderExtractionPromptRenderAllVariants:
    """Render-with-slots guard for EVERY ``extraction_second_order.txt`` —
    the shared base file plus every per-model override directory under
    ``configs/prompts/``, discovered by glob so a future model alias is
    covered automatically without a test edit.

    ``_load_prompt`` falls through to the base file when a model's
    override omits ``extraction_second_order.txt`` entirely (see its
    per-file, per-model resolution docstring), so a missing override is
    not a gap — only an override that PROVIDES the file but drops the
    ``{named_people}`` slot is: ``str.format`` silently ignores surplus
    kwargs, so that override would revert the phase to raw-prose target
    re-derivation under its model alias while the flow's post-parse
    enforcement (:func:`paramem.graph.flows._stage_second_order_extract`)
    keeps filtering against the correctly-derived closed set — a
    prompt/enforcement mismatch this test catches at CI time instead of
    at model-swap runtime.
    """

    def _variant_paths(self) -> list[Path]:
        base = _DEFAULT_PROMPT_DIR / "extraction_second_order.txt"
        overrides = sorted(_DEFAULT_PROMPT_DIR.glob("*/extraction_second_order.txt"))
        assert base.is_file(), f"base second-order prompt missing: {base}"
        return [base, *overrides]

    def test_at_least_the_base_and_one_override_are_covered(self):
        paths = self._variant_paths()
        assert len(paths) >= 2, (
            f"expected the base file plus at least one model-alias override, got {paths!r}"
        )

    def test_every_variant_renders_with_named_people_threaded(self):
        for path in self._variant_paths():
            tmpl = path.read_text(encoding="utf-8")
            rendered = tmpl.format(
                transcript="sample",
                speaker_context="",
                named_people="Dana, Riley",
            )
            assert "Dana, Riley" in rendered, (
                f"{path}: {{named_people}} slot did not reach the rendered prompt "
                "— override dropped the slot, reverting to prose re-derivation."
            )

    def test_every_variant_has_no_leftover_placeholders(self):
        for path in self._variant_paths():
            tmpl = path.read_text(encoding="utf-8")
            rendered = tmpl.format(
                transcript="sample",
                speaker_context="",
                named_people="Dana, Riley",
            )
            leftover = [
                m for m in _LEFTOVER_PLACEHOLDER.findall(rendered) if m not in _INTENTIONAL_LITERALS
            ]
            assert leftover == [], f"{path}: leftover placeholders after render: {leftover}"

    def test_every_variant_omitted_named_people_slot_raises_key_error(self):
        for path in self._variant_paths():
            tmpl = path.read_text(encoding="utf-8")
            with pytest.raises(KeyError):
                tmpl.format(transcript="sample", speaker_context="")


class TestAnonymizationPromptRender:
    """anonymization.txt renders with a rendered {scrub_categories} slot —
    the config-driven ``sanitization.scrub`` scope authority."""

    def _render(self):
        tmpl = _load_prompt("anonymization.txt")
        return tmpl.format(
            scrub_categories="person name, email address, phone number",
            facts_json="[]",
            transcript="sample",
            speaker_id="speaker0",
            speaker_anchor_section="",
        )

    def test_renders_without_exception(self):
        rendered = self._render()
        assert isinstance(rendered, str)

    def test_no_leftover_placeholders(self):
        rendered = self._render()
        leftover = [
            m for m in _LEFTOVER_PLACEHOLDER.findall(rendered) if m not in _INTENTIONAL_LITERALS
        ]
        assert leftover == [], f"Leftover placeholders after render: {leftover}"

    def test_scrub_categories_reach_rendered_output(self):
        """A non-default scrub value must actually reach the prompt — the
        guard against re-introducing a dead config knob."""
        rendered = self._render()
        assert "person name, email address, phone number" in rendered


class TestPredicateNormalizationPromptRender:
    """The per-group predicate-normalization prompt is .format()-ed with predicates_json
    at runtime (normalize_predicates).  Its JSON example literals MUST be
    double-escaped ({{ }}) or .format() raises KeyError.  Only {predicates_json}
    is a format field."""

    def _load(self):
        from paramem.graph.prompts import _load_prompt

        return _load_prompt("predicate_normalization.txt")

    def test_renders_without_exception(self):
        import json

        prompt = self._load()
        assert prompt, "predicate_normalization.txt must exist and be non-empty"
        # Must not raise KeyError (brace-escape regression guard).
        rendered = prompt.format(predicates_json=json.dumps(["works_for", "employed_by"]))
        assert isinstance(rendered, str)

    def test_only_expected_format_fields(self):
        import string

        prompt = self._load()
        fields = {f for _, f, _, _ in string.Formatter().parse(prompt) if f}
        assert fields == {"predicates_json"}, (
            f"Unexpected format fields (JSON braces likely unescaped): {sorted(fields)}"
        )

    def test_json_examples_render_to_valid_single_brace(self):
        import json

        prompt = self._load()
        rendered = prompt.format(predicates_json=json.dumps(["works_for", "employed_by"]))
        # Double-brace escape collapses to real single-brace JSON in examples.
        assert '{"clusters":' in rendered

    def test_no_leftover_placeholders(self):
        import json

        prompt = self._load()
        rendered = prompt.format(predicates_json=json.dumps(["works_for"]))
        leftover = _LEFTOVER_PLACEHOLDER.findall(rendered)
        assert leftover == [], f"Leftover placeholders after render: {leftover}"
