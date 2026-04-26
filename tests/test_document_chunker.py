"""Unit tests for paramem.graph.document_chunker.

Tests cover:
- Plain-text chunker: one chunk, empty rejection, multi-chunk split.
- Markdown chunker: H1 split, H2 split, short-section coalescing, front-matter
  stripping, and the critical fenced-code-block regression guard.
- PDF chunker: two-page extraction, scanned rejection, empty page handling.
- Dispatch: unsupported extension raises UnsupportedFormatError.

PDF tests use monkeypatching to inject synthetic page text, so no binary
fixture is required for mock-based tests.  The committed binary fixtures
(tests/fixtures/text_pdf_sample.pdf, tests/fixtures/scanned_pdf_sample.pdf)
are used for the real-pypdf tests.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from paramem.graph.document_chunker import (
    DocumentChunk,
    EmptyDocumentError,
    ScannedPdfRejectedError,
    UnsupportedFormatError,
    chunk_document,
    chunk_markdown_file,
    chunk_pdf_file,
    chunk_text_file,
)

_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
_TEXT_PDF = _FIXTURES_DIR / "text_pdf_sample.pdf"
_SCANNED_PDF = _FIXTURES_DIR / "scanned_pdf_sample.pdf"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_txt(tmp_path: Path, content: str, name: str = "doc.txt") -> Path:
    """Write *content* to a temporary .txt file and return its Path."""
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


def _make_md(tmp_path: Path, content: str, name: str = "doc.md") -> Path:
    """Write *content* to a temporary .md file and return its Path."""
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Plain-text chunker
# ---------------------------------------------------------------------------


class TestTxtChunker:
    def test_short_file_one_chunk(self, tmp_path):
        """A 100-byte file produces exactly one DocumentChunk."""
        content = "Hello world. This is a short document paragraph.\n"
        path = _make_txt(tmp_path, content)
        chunks = chunk_text_file(path)
        assert len(chunks) == 1
        chunk = chunks[0]
        assert isinstance(chunk, DocumentChunk)
        assert chunk.chunk_index == 0
        assert chunk.source_type == "document"
        assert chunk.doc_title == "doc"
        assert "Hello world" in chunk.text

    def test_empty_file_raises(self, tmp_path):
        """An empty file raises EmptyDocumentError."""
        path = _make_txt(tmp_path, "")
        with pytest.raises(EmptyDocumentError):
            chunk_text_file(path)

    def test_whitespace_only_file_raises(self, tmp_path):
        """A file containing only whitespace raises EmptyDocumentError."""
        path = _make_txt(tmp_path, "   \n\n\t\n  ")
        with pytest.raises(EmptyDocumentError):
            chunk_text_file(path)

    def test_long_file_splits_at_paragraph_boundaries(self, tmp_path):
        """A file exceeding max_tokens splits into multiple chunks."""
        # Build ~3000 words across 10 paragraphs separated by blank lines.
        para = "word " * 300 + "end."  # ~301 words
        content = "\n\n".join([para] * 10)
        path = _make_txt(tmp_path, content)
        chunks = chunk_text_file(path, max_tokens=1500)
        assert len(chunks) >= 2
        # No single chunk should exceed max_tokens by more than one paragraph.
        for chunk in chunks[:-1]:
            word_count = len(chunk.text.split())
            assert word_count <= 1500 + 400  # one paragraph of slack

    def test_chunk_indices_are_sequential(self, tmp_path):
        """chunk_index values start at 0 and increment by 1."""
        para = "word " * 600 + "end."
        content = "\n\n".join([para] * 5)
        path = _make_txt(tmp_path, content)
        chunks = chunk_text_file(path, max_tokens=1000)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_source_path_is_absolute(self, tmp_path):
        """source_path is the absolute resolved path."""
        path = _make_txt(tmp_path, "Some content here for testing.")
        chunks = chunk_text_file(path)
        assert Path(chunks[0].source_path).is_absolute()

    def test_doc_title_is_stem(self, tmp_path):
        """doc_title is the filename stem without extension."""
        path = _make_txt(tmp_path, "Some content.", name="my_resume.txt")
        chunks = chunk_text_file(path)
        assert chunks[0].doc_title == "my_resume"


# ---------------------------------------------------------------------------
# Markdown chunker
# ---------------------------------------------------------------------------


class TestMarkdownChunker:
    def test_h1_split_two_sections(self, tmp_path):
        """Two H1 sections produce two chunks; each contains the heading title."""
        content = (
            "# Section One\n\n" + ("alpha beta gamma delta epsilon " * 50) + "\n\n"
            "# Section Two\n\n" + ("zeta eta theta iota kappa " * 50)
        )
        path = _make_md(tmp_path, content)
        chunks = chunk_markdown_file(path, min_tokens=10)
        assert len(chunks) == 2
        assert "Section One" in chunks[0].text
        assert "Section Two" in chunks[1].text

    def test_h2_split_under_h1(self, tmp_path):
        """H2 also promotes to a chunk boundary."""
        content = (
            "## Subsection A\n\n" + ("word " * 100) + "\n\n## Subsection B\n\n" + ("word " * 100)
        )
        path = _make_md(tmp_path, content)
        chunks = chunk_markdown_file(path, min_tokens=10)
        assert len(chunks) == 2
        assert "Subsection A" in chunks[0].text
        assert "Subsection B" in chunks[1].text

    def test_short_h2_chunks_coalesce(self, tmp_path):
        """Short H2 sections below min_tokens are coalesced with the following section."""
        # Three sections of ~15 words each.  min_tokens=200 → they should merge.
        content = (
            "## Part A\n\nThis is a short section with a few words.\n\n"
            "## Part B\n\nAnother short section with a few words here.\n\n"
            "## Part C\n\nYet another brief section with some text.\n"
        )
        path = _make_md(tmp_path, content)
        chunks = chunk_markdown_file(path, min_tokens=200)
        # All three should have been coalesced since no single one reaches 200 words.
        assert len(chunks) == 1
        assert "Part A" in chunks[0].text
        assert "Part B" in chunks[0].text
        assert "Part C" in chunks[0].text

    def test_front_matter_stripped(self, tmp_path):
        """YAML front-matter is not present in any chunk text."""
        content = (
            "---\n"
            "title: Test Doc\n"
            "date: 2026-04-26\n"
            "---\n\n"
            "# Introduction\n\n"
            "This is the real content of the document.\n"
        )
        path = _make_md(tmp_path, content)
        chunks = chunk_markdown_file(path, min_tokens=1)
        combined = "\n".join(c.text for c in chunks)
        assert "title: Test Doc" not in combined
        assert "date: 2026-04-26" not in combined
        assert "Introduction" in combined

    def test_fenced_code_block_hash_not_a_heading(self, tmp_path):
        """Critical regression guard: # inside a fenced code block is NOT a heading.

        The markdown-it-py token stream handles fences as opaque fence tokens —
        the hash characters inside them never become heading_open tokens.
        This test asserts that behaviour so any parser regression shows up
        immediately.
        """
        content = (
            "# Outer Heading\n\n"
            "Some prose before the code block.\n\n"
            "```python\n"
            "# This comment hash must NOT be a heading boundary\n"
            "def foo():\n"
            "    # Another hash inside the fence\n"
            "    return 42\n"
            "```\n\n"
            "More prose after the code block.\n"
        )
        path = _make_md(tmp_path, content)
        # With min_tokens=1 each heading becomes its own chunk IF parsed correctly.
        # If `# This comment hash` were wrongly treated as a heading, we'd get > 1 chunk.
        chunks = chunk_markdown_file(path, min_tokens=1)
        # There is exactly one H1 heading → exactly one chunk.
        assert len(chunks) == 1, (
            f"Expected 1 chunk (fenced code block should not split), "
            f"got {len(chunks)}: {[c.text[:80] for c in chunks]}"
        )
        # The code inside the fence must appear in the chunk.
        assert "def foo" in chunks[0].text or "foo" in chunks[0].text

    def test_no_headings_treated_as_single_chunk(self, tmp_path):
        """A Markdown file with no headings produces one chunk with all content."""
        content = "Just some prose without any heading at all.\n\nAnother paragraph.\n"
        path = _make_md(tmp_path, content)
        chunks = chunk_markdown_file(path, min_tokens=1)
        assert len(chunks) == 1

    def test_empty_markdown_raises(self, tmp_path):
        """An empty .md file raises EmptyDocumentError."""
        path = _make_md(tmp_path, "")
        with pytest.raises(EmptyDocumentError):
            chunk_markdown_file(path)

    def test_doc_title_from_stem(self, tmp_path):
        """doc_title is the markdown filename stem."""
        path = _make_md(tmp_path, "# Title\n\nContent.", name="research_notes.md")
        chunks = chunk_markdown_file(path, min_tokens=1)
        assert all(c.doc_title == "research_notes" for c in chunks)

    def test_source_type_document(self, tmp_path):
        """source_type is always 'document' for markdown chunks."""
        path = _make_md(tmp_path, "# Heading\n\nSome content here.")
        chunks = chunk_markdown_file(path, min_tokens=1)
        assert all(c.source_type == "document" for c in chunks)

    def test_md_leading_hr_not_treated_as_front_matter(self, tmp_path):
        """Regression guard: a leading --- HR must NOT be treated as front-matter.

        Input: ---\\n\\nSome intro.\\n\\n---\\n\\n# Section\\nbody
        The intro paragraph between the two --- lines is prose, not YAML
        key-value pairs, so the leading --- is a horizontal rule.
        The full content (including 'Some intro.') must survive in the chunks.
        """
        content = "---\n\nSome intro.\n\n---\n\n# Section\nbody"
        path = _make_md(tmp_path, content)
        chunks = chunk_markdown_file(path, min_tokens=1)
        combined = "\n".join(c.text for c in chunks)
        assert "Some intro." in combined, (
            f"Leading HR was wrongly stripped as front-matter. Combined chunk text: {combined!r}"
        )


# ---------------------------------------------------------------------------
# PDF chunker — mock-based
# ---------------------------------------------------------------------------


def _fake_reader(pages_content: list[str]):
    """Build a mock pypdf.PdfReader whose pages return synthetic text."""
    fake_pages = []
    for text in pages_content:
        page = MagicMock()
        page.extract_text.return_value = text
        fake_pages.append(page)

    reader = MagicMock()
    reader.pages = fake_pages
    return reader


class TestPdfChunkerMocked:
    def test_two_pages_produce_chunks(self, tmp_path, monkeypatch):
        """Two pages with sufficient text produce at least one chunk."""
        page_texts = [
            "word " * 150 + "end of page one.",
            "word " * 150 + "end of page two.",
        ]
        fake = _fake_reader(page_texts)
        monkeypatch.setattr(
            "paramem.graph.document_chunker.pypdf.PdfReader",
            lambda path: fake,
        )
        path = tmp_path / "sample.pdf"
        path.write_bytes(b"%PDF-1.4 fake")
        chunks = chunk_pdf_file(path, min_tokens=200)
        assert len(chunks) >= 1
        full_text = " ".join(c.text for c in chunks)
        assert "end of page one" in full_text
        assert "end of page two" in full_text

    def test_scanned_pdf_rejected(self, tmp_path, monkeypatch):
        """Avg chars/page < 50 raises ScannedPdfRejectedError."""
        page_texts = ["img", "scan"]  # 3 + 4 = 7 chars total, avg = 3.5
        fake = _fake_reader(page_texts)
        monkeypatch.setattr(
            "paramem.graph.document_chunker.pypdf.PdfReader",
            lambda path: fake,
        )
        path = tmp_path / "scanned.pdf"
        path.write_bytes(b"%PDF-1.4 fake")
        with pytest.raises(ScannedPdfRejectedError):
            chunk_pdf_file(path)

    def test_empty_page_skipped(self, tmp_path, monkeypatch):
        """A page with no extractable text is skipped; surrounding pages still produce chunks."""
        page_texts = [
            "word " * 150 + "end of page one.",
            "",  # empty page
            "word " * 150 + "end of page three.",
        ]
        fake = _fake_reader(page_texts)
        monkeypatch.setattr(
            "paramem.graph.document_chunker.pypdf.PdfReader",
            lambda path: fake,
        )
        path = tmp_path / "sample.pdf"
        path.write_bytes(b"%PDF-1.4 fake")
        chunks = chunk_pdf_file(path, min_tokens=200)
        full_text = " ".join(c.text for c in chunks)
        assert "end of page one" in full_text
        assert "end of page three" in full_text

    def test_short_pages_coalesce(self, tmp_path, monkeypatch):
        """Consecutive short pages are merged until min_tokens is reached."""
        # Each page is ~50 words; min_tokens=200 → need ~4 pages per chunk.
        page_texts = ["word " * 50 + f"page{i}" for i in range(6)]
        fake = _fake_reader(page_texts)
        monkeypatch.setattr(
            "paramem.graph.document_chunker.pypdf.PdfReader",
            lambda path: fake,
        )
        path = tmp_path / "sample.pdf"
        path.write_bytes(b"%PDF-1.4 fake")
        chunks = chunk_pdf_file(path, min_tokens=200)
        # 6 pages of ~50 words → should produce ~2 chunks (4+2), not 6.
        assert len(chunks) < 6

    def test_all_empty_pages_raises(self, tmp_path, monkeypatch):
        """All empty pages raises EmptyDocumentError."""
        page_texts = ["", ""]
        fake = _fake_reader(page_texts)
        monkeypatch.setattr(
            "paramem.graph.document_chunker.pypdf.PdfReader",
            lambda path: fake,
        )
        path = tmp_path / "empty.pdf"
        path.write_bytes(b"%PDF-1.4 fake")
        with pytest.raises(EmptyDocumentError):
            chunk_pdf_file(path)

    def test_chunk_indices_sequential(self, tmp_path, monkeypatch):
        """chunk_index values are sequential starting from 0."""
        page_texts = ["word " * 250 + "page1", "word " * 250 + "page2"]
        fake = _fake_reader(page_texts)
        monkeypatch.setattr(
            "paramem.graph.document_chunker.pypdf.PdfReader",
            lambda path: fake,
        )
        path = tmp_path / "sample.pdf"
        path.write_bytes(b"%PDF-1.4 fake")
        chunks = chunk_pdf_file(path, min_tokens=200)
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_pdf_trailing_short_page_coalesces(self, tmp_path, monkeypatch):
        """Regression guard: a trailing sub-min_tokens page must NOT produce a separate chunk.

        Reproducer: pages [long_250w, tiny_5w] with min_tokens=200.
        Without the coalesce fix this yields two chunks ([251w, 6w]).
        With the fix the 6-word remainder is absorbed into the preceding chunk.
        """
        long_page = "word " * 250 + "end of long page."
        tiny_page = "tiny tail."
        fake = _fake_reader([long_page, tiny_page])
        monkeypatch.setattr(
            "paramem.graph.document_chunker.pypdf.PdfReader",
            lambda path: fake,
        )
        path = tmp_path / "sample.pdf"
        path.write_bytes(b"%PDF-1.4 fake")
        chunks = chunk_pdf_file(path, min_tokens=200)
        assert len(chunks) == 1, (
            f"Expected 1 chunk after coalescing trailing short page, "
            f"got {len(chunks)}: {[len(c.text.split()) for c in chunks]}"
        )
        # Both pages' text must be present in the single merged chunk.
        assert "end of long page" in chunks[0].text
        assert "tiny tail" in chunks[0].text


# ---------------------------------------------------------------------------
# PDF chunker — real binary fixtures
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _TEXT_PDF.exists(),
    reason="tests/fixtures/text_pdf_sample.pdf not committed",
)
class TestPdfChunkerRealFixture:
    def test_text_pdf_produces_chunks(self):
        """Committed text PDF produces at least one chunk with real text."""
        chunks = chunk_pdf_file(_TEXT_PDF, min_tokens=100)
        assert len(chunks) >= 1
        full_text = " ".join(c.text for c in chunks)
        assert len(full_text) > 50

    def test_text_pdf_doc_title(self):
        """doc_title is the PDF file stem."""
        chunks = chunk_pdf_file(_TEXT_PDF, min_tokens=10)
        assert all(c.doc_title == "text_pdf_sample" for c in chunks)


@pytest.mark.skipif(
    not _SCANNED_PDF.exists(),
    reason="tests/fixtures/scanned_pdf_sample.pdf not committed",
)
class TestScannedPdfRejectionRealFixture:
    def test_scanned_pdf_raises(self):
        """Committed near-empty PDF raises ScannedPdfRejectedError."""
        with pytest.raises(ScannedPdfRejectedError):
            chunk_pdf_file(_SCANNED_PDF)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


class TestChunkDocumentDispatch:
    def test_txt_dispatches(self, tmp_path):
        """chunk_document dispatches .txt to the text chunker."""
        path = _make_txt(tmp_path, "Some content here for dispatch test.", name="test.txt")
        chunks = chunk_document(path)
        assert len(chunks) >= 1
        assert chunks[0].source_type == "document"

    def test_md_dispatches(self, tmp_path):
        """chunk_document dispatches .md to the markdown chunker."""
        path = _make_md(tmp_path, "# Title\n\nSome markdown content.", name="test.md")
        chunks = chunk_document(path)
        assert len(chunks) >= 1

    def test_markdown_extension_dispatches(self, tmp_path):
        """chunk_document dispatches .markdown to the markdown chunker."""
        path = tmp_path / "test.markdown"
        path.write_text("# Title\n\nContent.", encoding="utf-8")
        chunks = chunk_document(path)
        assert len(chunks) >= 1

    def test_unsupported_extension_raises(self, tmp_path):
        """chunk_document raises UnsupportedFormatError for .docx files."""
        path = tmp_path / "resume.docx"
        path.write_bytes(b"PK fake docx content")
        with pytest.raises(UnsupportedFormatError):
            chunk_document(path)

    def test_case_insensitive_suffix(self, tmp_path):
        """chunk_document is case-insensitive on file suffix."""
        path = _make_txt(tmp_path, "Some content.", name="doc.TXT")
        # Should not raise UnsupportedFormatError
        chunks = chunk_document(path)
        assert len(chunks) >= 1

    def test_pdf_dispatches(self, tmp_path, monkeypatch):
        """chunk_document dispatches .pdf to the PDF chunker."""
        page_texts = ["word " * 300 + "pdf content here"]
        fake = _fake_reader(page_texts)
        monkeypatch.setattr(
            "paramem.graph.document_chunker.pypdf.PdfReader",
            lambda path: fake,
        )
        path = tmp_path / "resume.pdf"
        path.write_bytes(b"%PDF-1.4 fake")
        chunks = chunk_document(path)
        assert len(chunks) >= 1
