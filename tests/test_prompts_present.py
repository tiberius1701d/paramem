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
