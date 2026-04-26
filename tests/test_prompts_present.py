"""Ship-gate tests — assert required prompt files exist and carry expected placeholders."""

from __future__ import annotations

from pathlib import Path

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "configs" / "prompts"


class TestPromptFilesPresent:
    def test_extraction_txt_exists(self):
        assert (_PROMPTS_DIR / "extraction.txt").exists()

    def test_extraction_procedural_txt_exists(self):
        assert (_PROMPTS_DIR / "extraction_procedural.txt").exists()

    def test_extraction_txt_has_transcript_placeholder(self):
        content = (_PROMPTS_DIR / "extraction.txt").read_text()
        assert "{transcript}" in content

    def test_extraction_procedural_txt_has_transcript_placeholder(self):
        content = (_PROMPTS_DIR / "extraction_procedural.txt").read_text()
        assert "{transcript}" in content

    def test_extraction_system_document_txt_exists(self):
        assert (_PROMPTS_DIR / "extraction_system_document.txt").exists()

    def test_extraction_system_document_txt_no_braces(self):
        """Regression guard: system prompt must be plain-English directives only.

        The system prompt is passed verbatim to the model — no slot substitution
        is performed on it.  Any ``{`` character in the file means someone
        accidentally re-introduced a template slot that will never be filled,
        potentially leaking the raw brace syntax into the model context.
        """
        content = (_PROMPTS_DIR / "extraction_system_document.txt").read_text()
        assert "{" not in content, (
            "extraction_system_document.txt contains '{' braces — system prompts "
            "are plain-English only; slot substitution runs only on user templates."
        )

    def test_extraction_document_txt_exists(self):
        assert (_PROMPTS_DIR / "extraction_document.txt").exists()

    def test_extraction_document_txt_has_transcript_placeholder(self):
        content = (_PROMPTS_DIR / "extraction_document.txt").read_text()
        assert "{transcript}" in content

    def test_extraction_document_txt_has_speaker_context_placeholder(self):
        content = (_PROMPTS_DIR / "extraction_document.txt").read_text()
        assert "{speaker_context}" in content

    def test_extraction_document_txt_no_dialogue_markers(self):
        """Regression guard: document user template must not contain dialogue markers.

        If ``[user]`` or ``[assistant]`` appear, the document few-shots have
        leaked dialogue shape, contradicting the system prompt directive.
        """
        content = (_PROMPTS_DIR / "extraction_document.txt").read_text()
        assert "[user]" not in content, (
            "extraction_document.txt contains '[user]' — document few-shots "
            "must not contain dialogue markers."
        )
        assert "[assistant]" not in content, (
            "extraction_document.txt contains '[assistant]' — document few-shots "
            "must not contain dialogue markers."
        )

    def test_extraction_document_txt_has_json_output_directive(self):
        """Coarse contract check: JSON output schema keywords must be present.

        Ensures the document user template carries the same output contract as
        extraction.txt so _parse_extraction works unchanged.
        """
        content = (_PROMPTS_DIR / "extraction_document.txt").read_text()
        assert "entities" in content, (
            "extraction_document.txt missing 'entities' keyword — JSON output "
            "contract may be broken."
        )
        assert "relations" in content, (
            "extraction_document.txt missing 'relations' keyword — JSON output "
            "contract may be broken."
        )

    # P3 — system-prompt content guard

    def test_extraction_system_document_txt_contains_json_keyword(self):
        """Regression guard: extraction_system_document.txt must mention JSON.

        The document extraction system prompt must instruct the model to emit
        JSON.  If a future edit drops that directive, extraction on the document
        path silently produces unparseable output.
        """
        content = (_PROMPTS_DIR / "extraction_system_document.txt").read_text()
        assert "JSON" in content, (
            "extraction_system_document.txt does not contain 'JSON' — the output "
            "directive may have been accidentally removed, which would break "
            "document extraction."
        )

    # Procedural document prompt checks

    def test_extraction_procedural_document_txt_exists(self):
        """The document-variant procedural prompt file must exist."""
        assert (_PROMPTS_DIR / "extraction_procedural_document.txt").exists(), (
            "configs/prompts/extraction_procedural_document.txt is missing. "
            "Procedural extraction for document sources will silently fall back "
            "to the dialogue user template, contradicting the document system prompt."
        )

    def test_extraction_procedural_document_txt_has_all_placeholders(self):
        """Document procedural template must carry the same placeholders as the
        dialogue procedural template so _generate_procedural_extraction does
        not need a separate format-kwargs path.

        Verifies: {speaker_context}, {entity_types}, {predicate_examples},
        {transcript}  (the full set used in extraction_procedural.txt).
        """
        content = (_PROMPTS_DIR / "extraction_procedural_document.txt").read_text()
        required = ("{speaker_context}", "{entity_types}", "{predicate_examples}", "{transcript}")
        for placeholder in required:
            assert placeholder in content, (
                f"extraction_procedural_document.txt missing placeholder {placeholder!r} — "
                "must mirror the full placeholder set of extraction_procedural.txt."
            )

    def test_extraction_procedural_document_txt_no_dialogue_markers(self):
        """Document procedural template must not contain [user] or [assistant] markers.

        Dialogue markers in the few-shots contradict the document system prompt
        which explicitly states there is no turn structure.
        """
        content = (_PROMPTS_DIR / "extraction_procedural_document.txt").read_text()
        assert "[user]" not in content, (
            "extraction_procedural_document.txt contains '[user]' — document "
            "few-shots must not contain dialogue markers."
        )
        assert "[assistant]" not in content, (
            "extraction_procedural_document.txt contains '[assistant]' — document "
            "few-shots must not contain dialogue markers."
        )

    def test_extraction_procedural_document_txt_no_assistant_references(self):
        """Document procedural template must not reference the assistant at all.

        The document path has no assistant response — any mention of 'assistant'
        (case-insensitive) would prime the model to expect a turn that does not
        exist.
        """
        content = (_PROMPTS_DIR / "extraction_procedural_document.txt").read_text()
        assert "assistant" not in content.lower(), (
            "extraction_procedural_document.txt references 'assistant' — the "
            "document path has no assistant response to cross-reference."
        )

    def test_extraction_procedural_document_txt_has_json_output_directive(self):
        """Document procedural template must carry the JSON output directive.

        Ensures _parse_procedural can parse the output unchanged.
        """
        content = (_PROMPTS_DIR / "extraction_procedural_document.txt").read_text()
        assert "JSON" in content, (
            "extraction_procedural_document.txt missing 'JSON' keyword — "
            "the output directive may be missing, breaking procedural parsing."
        )
