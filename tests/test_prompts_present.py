"""Ship-gate tests — assert required prompt files exist and carry expected placeholders.

The extraction prompt-pair (``extraction.txt`` + ``extraction_system.txt``)
plus the procedural user template (``extraction_procedural.txt``) is the
single ground truth for extraction.  Document chunks land in the same
``{transcript}`` slot at the chat-template layer; there are no
document-variant prompt files.  The retired
``extraction_document.txt`` / ``extraction_system_document.txt`` /
``extraction_procedural_document.txt`` files are deliberately absent —
their existence used to permit silent drift on schema-shape rules.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "configs" / "prompts"


class TestPromptFilesPresent:
    def test_extraction_txt_exists(self):
        assert (_PROMPTS_DIR / "extraction.txt").exists()

    def test_extraction_system_txt_exists(self):
        assert (_PROMPTS_DIR / "extraction_system.txt").exists()

    def test_extraction_procedural_txt_exists(self):
        assert (_PROMPTS_DIR / "extraction_procedural.txt").exists()

    def test_extraction_txt_has_transcript_placeholder(self):
        content = (_PROMPTS_DIR / "extraction.txt").read_text()
        assert "{transcript}" in content

    def test_extraction_txt_has_speaker_context_placeholder(self):
        content = (_PROMPTS_DIR / "extraction.txt").read_text()
        assert "{speaker_context}" in content

    def test_extraction_procedural_txt_has_transcript_placeholder(self):
        content = (_PROMPTS_DIR / "extraction_procedural.txt").read_text()
        assert "{transcript}" in content

    def test_extraction_system_txt_no_braces(self):
        """Regression guard: system prompt must be plain-English directives only.

        The system prompt is passed verbatim to the model — no slot substitution
        is performed on it.  Any ``{`` character in the file means someone
        accidentally re-introduced a template slot that will never be filled,
        potentially leaking the raw brace syntax into the model context.
        """
        content = (_PROMPTS_DIR / "extraction_system.txt").read_text()
        assert "{" not in content, (
            "extraction_system.txt contains '{' braces — system prompts "
            "are plain-English only; slot substitution runs only on user templates."
        )

    def test_extraction_system_txt_contains_json_keyword(self):
        """Regression guard: extraction_system.txt must mention JSON.

        The extraction system prompt must instruct the model to emit JSON.
        If a future edit drops that directive, extraction silently produces
        unparseable output.
        """
        content = (_PROMPTS_DIR / "extraction_system.txt").read_text()
        assert "JSON" in content, (
            "extraction_system.txt does not contain 'JSON' — the output "
            "directive may have been accidentally removed, which would break extraction."
        )

    def test_extraction_txt_has_json_output_directive(self):
        """Coarse contract check: JSON output schema keywords must be present.

        Ensures the user template carries the same output contract the parser expects.
        """
        content = (_PROMPTS_DIR / "extraction.txt").read_text()
        assert "entities" in content, (
            "extraction.txt missing 'entities' keyword — JSON output contract may be broken."
        )
        assert "relations" in content, (
            "extraction.txt missing 'relations' keyword — JSON output contract may be broken."
        )

    def test_extraction_procedural_txt_has_required_placeholders(self):
        """Procedural template must carry the slot-substituted placeholders.

        ``{entity_types}`` and ``{predicate_examples}`` are deliberately
        absent — verbatim taxonomy listings empirically license the
        model to invent off-list types (same finding that drove the
        factual ``extraction.txt`` to drop those slots).  Schema
        coverage is now carried by the few-shot examples.

        ``{speaker_context}`` and ``{transcript}`` ARE required — the
        call site at :func:`paramem.graph.extractor.extract_procedural_graph`
        passes those values, and missing placeholders mean the
        speaker directive / chunk text never reach the model.
        """
        content = (_PROMPTS_DIR / "extraction_procedural.txt").read_text()
        required = ("{speaker_context}", "{transcript}")
        for placeholder in required:
            assert placeholder in content, (
                f"extraction_procedural.txt missing placeholder {placeholder!r} — "
                "the format-kwargs call site expects this slot."
            )

    def test_extraction_procedural_txt_no_taxonomy_slots(self):
        """Regression guard: the procedural prompt must NOT reintroduce
        the verbatim taxonomy slots.  See the docstring on
        :meth:`test_extraction_procedural_txt_has_required_placeholders`
        for the empirical reason.
        """
        content = (_PROMPTS_DIR / "extraction_procedural.txt").read_text()
        for forbidden in ("{entity_types}", "{predicate_examples}"):
            assert forbidden not in content, (
                f"extraction_procedural.txt re-introduced {forbidden!r} — "
                "verbatim taxonomy slots license invented types; remove and let "
                "the few-shots carry schema coverage instead."
            )

    def test_extraction_procedural_txt_has_json_output_directive(self):
        """Procedural template must carry the JSON output directive."""
        content = (_PROMPTS_DIR / "extraction_procedural.txt").read_text()
        assert "JSON" in content, (
            "extraction_procedural.txt missing 'JSON' keyword — the output "
            "directive may be missing, breaking procedural parsing."
        )


class TestSystemPromptFilesPresent:
    """Presence + brace guard for the seven externalized SYSTEM-prompt files.

    Each file is the companion ``<base>_system.txt`` for an already-external
    USER template, following the ``extraction.txt`` / ``extraction_system.txt``
    pattern.  See :func:`test_extraction_system_txt_no_braces` for the
    rationale on the brace guard — system prompts receive no slot
    substitution, so a stray ``{`` would leak raw template syntax into the
    model context.
    """

    _SYSTEM_PROMPT_FILES = (
        "entity_correction_system.txt",
        "merger_coexistence_system.txt",
        "anonymization_system.txt",
        "sota_plausibility_system.txt",
        "sota_enrichment_system.txt",
        "graph_dedup_filter_system.txt",
        "sota_graph_enrichment_system.txt",
    )

    def test_entity_correction_system_txt_exists(self):
        assert (_PROMPTS_DIR / "entity_correction_system.txt").exists()

    def test_merger_coexistence_system_txt_exists(self):
        assert (_PROMPTS_DIR / "merger_coexistence_system.txt").exists()

    def test_anonymization_system_txt_exists(self):
        assert (_PROMPTS_DIR / "anonymization_system.txt").exists()

    def test_sota_plausibility_system_txt_exists(self):
        assert (_PROMPTS_DIR / "sota_plausibility_system.txt").exists()

    def test_sota_enrichment_system_txt_exists(self):
        assert (_PROMPTS_DIR / "sota_enrichment_system.txt").exists()

    def test_graph_dedup_filter_system_txt_exists(self):
        assert (_PROMPTS_DIR / "graph_dedup_filter_system.txt").exists()

    def test_sota_graph_enrichment_system_txt_exists(self):
        assert (_PROMPTS_DIR / "sota_graph_enrichment_system.txt").exists()

    def test_all_system_prompt_files_no_braces(self):
        for filename in self._SYSTEM_PROMPT_FILES:
            content = (_PROMPTS_DIR / filename).read_text()
            assert "{" not in content, (
                f"{filename} contains '{{' braces — system prompts are "
                "plain-English only; slot substitution runs only on user templates."
            )


class TestSystemPromptGoldens:
    """Byte-for-byte preservation goldens for the seven externalized files.

    Each golden string was captured programmatically from the pre-change
    inline literal/constant (single-line literals copied verbatim from
    source; the three former module constants captured via
    ``repr(extractor._SOTA_*_SYSTEM_PROMPT)`` before the constants were
    replaced with ``_load_prompt(...)`` calls) — never hand-retyped against
    the new ``.txt`` file, so a shared typo cannot silently pass both sides.
    """

    def test_entity_correction_system_golden(self):
        content = (_PROMPTS_DIR / "entity_correction_system.txt").read_text().strip()
        assert content == "Output valid JSON only."

    def test_merger_coexistence_system_golden(self):
        content = (_PROMPTS_DIR / "merger_coexistence_system.txt").read_text().strip()
        assert content == "You classify relationship cardinality."

    def test_anonymization_system_golden(self):
        content = (_PROMPTS_DIR / "anonymization_system.txt").read_text().strip()
        assert content == "You anonymize data. Output valid JSON only."

    def test_sota_plausibility_system_golden(self):
        content = (_PROMPTS_DIR / "sota_plausibility_system.txt").read_text().strip()
        assert content == (
            "You are a knowledge graph plausibility filter. Drop invalid facts "
            "only. Do NOT add or modify facts. Output valid JSON only."
        )

    def test_sota_enrichment_system_golden(self):
        content = (_PROMPTS_DIR / "sota_enrichment_system.txt").read_text().strip()
        assert content == (
            "You are a knowledge graph enrichment assistant. Resolve coreference "
            "and split compound facts. Do NOT remove facts — a separate "
            "plausibility filter handles removal. Output valid JSON only."
        )

    def test_graph_dedup_filter_system_golden(self):
        content = (_PROMPTS_DIR / "graph_dedup_filter_system.txt").read_text().strip()
        assert content == "You identify synonym predicate clusters. Output valid JSON only."

    def test_sota_graph_enrichment_system_golden(self):
        content = (_PROMPTS_DIR / "sota_graph_enrichment_system.txt").read_text().strip()
        assert content == (
            "You are a knowledge graph enrichment assistant operating over a "
            "pre-merged cross-transcript graph. Emit cross-session second-order "
            "relations and same_as pairs for duplicate entities. Output valid "
            "JSON only."
        )


class TestRetiredDocumentPromptsAbsent:
    """The document-variant prompt files are retired — their absence is the
    architectural guard against silent drift on schema-shape rules.

    Restoring any of these files re-introduces the two-prompt design that
    produced drift on:
      * speaker-name fragmentation NEGATIVE example
      * concept POSITIVE example
    Add to the transcript prompt instead, or prepend/append at the slot
    layer if a source-specific extension is genuinely required.
    """

    def test_extraction_document_txt_absent(self):
        assert not (_PROMPTS_DIR / "extraction_document.txt").exists(), (
            "extraction_document.txt has been re-introduced — the project "
            "deliberately uses a single prompt-pair for every source type."
        )

    def test_extraction_system_document_txt_absent(self):
        assert not (_PROMPTS_DIR / "extraction_system_document.txt").exists(), (
            "extraction_system_document.txt has been re-introduced — "
            "system prompts are not source-type-specific."
        )

    def test_extraction_procedural_document_txt_absent(self):
        assert not (_PROMPTS_DIR / "extraction_procedural_document.txt").exists(), (
            "extraction_procedural_document.txt has been re-introduced — "
            "procedural extraction uses a single prompt for every source type."
        )


class TestEnsurePromptAssets:
    """Runtime startup guard mirroring the ship-gate presence tests above.

    ``ensure_prompt_assets`` runs first in the server lifespan so a broken
    checkout / non-editable pip install (prompts are not packaged) fails
    loudly instead of the extraction pipeline silently loading empty prompts.
    """

    def test_passes_with_real_prompt_dir(self):
        # A repo checkout always has configs/prompts/ with the required files.
        from paramem.graph.prompts import ensure_prompt_assets

        ensure_prompt_assets()

    def test_raises_when_dir_missing(self, monkeypatch, tmp_path):
        import paramem.graph.prompts as prompts_mod

        monkeypatch.setattr(prompts_mod, "_DEFAULT_PROMPT_DIR", tmp_path / "absent")
        with pytest.raises(RuntimeError, match="Prompt asset directory not found"):
            prompts_mod.ensure_prompt_assets()

    def test_raises_when_required_file_missing(self, monkeypatch, tmp_path):
        # Directory exists but lacks the load-bearing extraction files.
        import paramem.graph.prompts as prompts_mod

        monkeypatch.setattr(prompts_mod, "_DEFAULT_PROMPT_DIR", tmp_path)
        with pytest.raises(RuntimeError, match="Required prompt file"):
            prompts_mod.ensure_prompt_assets()
