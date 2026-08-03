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

import inspect
import math
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from paramem.graph.document_chunker import (
    _DOC_MAX_TOKENS,
    _DOC_MIN_TOKENS,
    DocumentChunk,
    EmptyDocumentError,
    ScannedPdfRejectedError,
    UnsupportedFormatError,
    _coalesce_to_max_tokens,
    _overlap_tokens_to_words,
    _split_into_sentences,
    chunk_document,
    chunk_markdown_file,
    chunk_pdf_file,
    chunk_text_file,
)
from paramem.utils.tokens import MEASURED_TOKENS_PER_WORD, estimate_tokens

# Document-shape ratio — the same value recorded as tokens.py's
# "document shape (CV)" row feeding MEASURED_TOKENS_PER_WORD, and the same
# value document_chunker.py's _DOC_MAX_TOKENS derivation comment uses for
# BOTH the cap conversion and the anon_template estimate: one ratio
# throughout, never a mix of an exact-tokenizer snapshot and an estimate.
_R_PROSE = 1.9126
_ANONYMIZATION_TEMPLATE_PATH = (
    Path(__file__).resolve().parent.parent / "configs" / "prompts" / "anonymization.txt"
)


def _render_anon_template_text() -> str:
    """Load and format the real anonymization.txt skeleton (empty
    facts/scrub/transcript, no speaker anchor) — the same shape
    ``_render_anonymize_prompt`` produces before chat-template wrapping
    (``paramem/cloud/anonymize.py``) when no speaker id is threaded (the
    document-ingest path has no speaker anchor at all, so the anchor-less
    render is this derivation's correct budget basis — see
    ``_render_anonymize_prompt``'s ``speaker_anchor_section`` slot).
    """
    template = _ANONYMIZATION_TEMPLATE_PATH.read_text(encoding="utf-8")
    return template.format(
        scrub_categories="",
        facts_json="[]",
        transcript="",
        speaker_id="",
        speaker_anchor_section="",
    )


def _live_anon_template_tokens() -> int:
    """THE anon_template term for the ``_DOC_MAX_TOKENS`` derivation,
    recomputed from the LIVE prompt file on every call — ``ceil(words *
    r_prose)``, the SAME formula ``document_chunker.py``'s
    ``_DOC_MAX_TOKENS`` derivation comment uses.

    Why this is recomputed rather than recorded: a frozen
    tokenizer-measured constant here, checked only against a wide
    word-count tolerance band, left a real budget with only a few tokens
    of end-to-end slack that a small template edit could silently overrun
    while the band check still passed (the arithmetic test never actually
    consumed the live word count). Recomputing THIS value directly — and
    using it in the arithmetic checks below instead of a recorded literal
    — couples the two: a template edit changes this return value on the
    next run, no tolerance band needed (the computation is fully
    deterministic).
    """
    return math.ceil(len(_render_anon_template_text().split()) * _R_PROSE)


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
        # Build 10 paragraphs of ~301 words each, separated by blank lines.
        para = "word " * 300 + "end."  # ~301 words
        para_tokens = estimate_tokens(para)  # estimated-token cost of one paragraph
        content = "\n\n".join([para] * 10)
        path = _make_txt(tmp_path, content)
        chunks = chunk_text_file(path, max_tokens=1500)
        assert len(chunks) >= 2
        # No single chunk should exceed max_tokens by more than one
        # paragraph's estimated-token cost — the greedy merge only
        # overflows by the paragraph that triggered the split.
        for chunk in chunks[:-1]:
            assert estimate_tokens(chunk.text) <= 1500 + para_tokens

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


class TestSplitIntoSentences:
    def test_splits_on_period_question_exclaim(self):
        text = "First sentence. Second sentence! Third one? Fourth"
        out = _split_into_sentences(text)
        assert out == ["First sentence.", "Second sentence!", "Third one?", "Fourth"]

    def test_handles_quoted_terminator(self):
        text = 'He said "hello." She left.'
        out = _split_into_sentences(text)
        assert out == ['He said "hello."', "She left."]

    def test_empty_input(self):
        assert _split_into_sentences("") == []
        assert _split_into_sentences("   ") == []

    def test_no_terminators_returns_single(self):
        out = _split_into_sentences("words without any terminator")
        assert out == ["words without any terminator"]


class TestCoalesceToMaxTokens:
    """max_tokens/overlap_tokens are estimated-token thresholds (routed
    through estimate_tokens), not raw word counts — the expectations below
    are recomputed against the shared estimator rather than loosened from
    the raw word-count numbers they used before counting moved onto
    estimate_tokens, so they still pin the boundary exactly.
    """

    def test_packs_under_cap_into_one_chunk(self):
        sentences = ["A.", "B.", "C."]
        # Each 1-word sentence estimates to ceil(1 * ratio) tokens; three of
        # them must all fit under the cap for a single merged chunk.
        cap = 3 * estimate_tokens("A.")
        out = _coalesce_to_max_tokens(sentences, max_tokens=cap)
        assert out == ["A. B. C."]

    def test_splits_at_sentence_boundary_when_cap_exceeded(self):
        # Each sentence is 5 words; a cap sized for exactly two sentences'
        # estimated-token cost packs two per chunk, not four.
        sentences = [
            "one two three four five.",
            "six seven eight nine ten.",
            "eleven twelve thirteen fourteen fifteen.",
            "sixteen seventeen eighteen nineteen twenty.",
        ]
        cap = 2 * estimate_tokens(sentences[0])
        out = _coalesce_to_max_tokens(sentences, max_tokens=cap)
        assert len(out) == 2
        assert "one two three four five." in out[0]
        assert "six seven eight nine ten." in out[0]
        assert "eleven" in out[1]
        assert "sixteen" in out[1]

    def test_overlap_carries_trailing_words(self):
        sentences = ["a b c d e.", "f g h i j.", "k l m."]
        # Cap sized to exactly one 5-word sentence's estimated-token cost,
        # so each sentence starts a new chunk; overlap_tokens is sized (via
        # _overlap_tokens_to_words) to carry the trailing 2 words forward.
        cap = estimate_tokens(sentences[0])
        overlap_tokens = estimate_tokens("d e.")
        assert _overlap_tokens_to_words(overlap_tokens) == 2
        out = _coalesce_to_max_tokens(sentences, max_tokens=cap, overlap_tokens=overlap_tokens)
        # First chunk: "a b c d e." (fills the cap on its own)
        # Second chunk gets the last 2 words of the first as overlap prefix.
        assert out[0] == "a b c d e."
        assert out[1].startswith("d e.")

    def test_oversize_sentence_kept_whole(self):
        sentences = ["one two three four five six seven eight nine ten."]
        out = _coalesce_to_max_tokens(sentences, max_tokens=5)
        # Cap=5 but sentence is 10 words; never cut mid-sentence.
        assert out == ["one two three four five six seven eight nine ten."]


class TestOverlapTokensToWords:
    """_overlap_tokens_to_words converts the token-denominated
    overlap_tokens budget into the word count _coalesce_to_max_tokens
    slices — the unit boundary in document_chunker.py's
    _coalesce_to_max_tokens: all counting routes through estimate_tokens,
    while the overlap-prefix construction stays word-based because words
    are what it actually slices out of the emitted chunk's text.
    """

    def test_zero_or_negative_yields_zero_words(self):
        assert _overlap_tokens_to_words(0) == 0
        assert _overlap_tokens_to_words(-5) == 0

    def test_positive_overlap_tokens_yields_at_least_one_word(self):
        assert _overlap_tokens_to_words(1) >= 1

    def test_converts_via_the_shared_ratio(self):
        # 2 words at the shared ratio round-trips back to 2 words.
        two_word_tokens = estimate_tokens("d e.")
        assert _overlap_tokens_to_words(two_word_tokens) == 2


class TestChunkerDefaultsMatchDerivedConstants:
    """The public chunker signatures' defaults are exactly the derived
    module constants, not a separately hand-typed copy of the same number
    (a hand-typed duplicate would keep passing every threshold test above
    while silently diverging from the derived cap)."""

    def test_chunk_text_file_default_is_doc_max_tokens(self):
        assert inspect.signature(chunk_text_file).parameters["max_tokens"].default == (
            _DOC_MAX_TOKENS
        )

    def test_chunk_markdown_file_default_is_doc_min_tokens(self):
        assert inspect.signature(chunk_markdown_file).parameters["min_tokens"].default == (
            _DOC_MIN_TOKENS
        )

    def test_chunk_pdf_file_defaults_match_both_constants(self):
        params = inspect.signature(chunk_pdf_file).parameters
        assert params["min_tokens"].default == _DOC_MIN_TOKENS
        assert params["max_tokens"].default == _DOC_MAX_TOKENS


class TestDocMaxTokensDerivation:
    """The derivation link between the shipped _DOC_MAX_TOKENS and the
    anonymize-call token envelope, pinned as a live test rather than left
    as a claim in a comment or commit message: the constant is a recorded
    literal, so without these tests nothing would notice if the envelope,
    the reserve constants, or the prompt template moved out from under it.
    """

    def test_derivation_link_holds_against_the_shipped_envelope(self):
        """2 * cap_real + anon_template_tokens + reserve <= envelope,
        computed from the real shipped envelope default, the LIVE
        anonymize.py reserve constants (imported symbolically rather than
        copied in as numbers, so a concurrent change to those constants is
        picked up automatically rather than silently going stale), and the
        LIVE anon_template recomputed from the current prompt file
        (replacing a frozen tokenizer-measured constant that was only
        loosely band-checked against the file — see
        _live_anon_template_tokens's docstring). No tolerance band is
        needed anywhere in this test: every term is either an exact live
        import or a deterministic recomputation.  Fails if a future edit
        raises _DOC_MAX_TOKENS, shrinks the envelope, grows the reserve, or
        grows the template, without re-deriving the relationship.
        """
        from paramem.cloud.anonymize import (
            _DEFAULT_ANONYMIZER_TOKEN_ENVELOPE,
            _MAPPING_ENTRY_OVERHEAD_TOKENS,
            _OUTPUT_JSON_ENVELOPE_TOKENS,
        )

        reserve = _OUTPUT_JSON_ENVELOPE_TOKENS + _MAPPING_ENTRY_OVERHEAD_TOKENS
        cap_real = _DOC_MAX_TOKENS / MEASURED_TOKENS_PER_WORD * _R_PROSE
        lhs = 2 * cap_real + _live_anon_template_tokens() + reserve
        assert lhs <= _DEFAULT_ANONYMIZER_TOKEN_ENVELOPE, (
            f"derivation link broken: {lhs} > {_DEFAULT_ANONYMIZER_TOKEN_ENVELOPE}"
        )

    def test_ratio_cancellation_invariant(self):
        """The chunk-boundary comparison
        (``estimate_tokens(text) <= max_tokens``, where ``max_tokens`` is
        ``_DOC_MAX_TOKENS`` re-expressed as ``cap_words * ratio`` — the
        estimator's own unit) is invariant under a change to the shipped
        words->tokens ratio, because it reduces to ``words <= cap_words``
        regardless of ratio.  Would fail if a future edit expressed the
        cap in real tokens instead of the estimator's own unit (the ratio
        would then no longer cancel and a base-model swap that shifts the
        ratio would silently shrink or grow document chunks).
        """
        cap_words = _DOC_MAX_TOKENS / MEASURED_TOKENS_PER_WORD
        word_counts = [round(cap_words) - 50, round(cap_words) + 50]
        for ratio in (1.4, 3.4, 5.0, 8.0):  # a hypothetical base-model swap's ratio
            scaled_cap_tokens = cap_words * ratio  # _DOC_MAX_TOKENS re-derived at this ratio
            for words in word_counts:
                text = "word " * words
                fits_at_word_level = words <= cap_words
                fits_via_estimator = (
                    estimate_tokens(text, tokens_per_word=ratio) <= scaled_cap_tokens
                )
                assert fits_via_estimator == fits_at_word_level, (
                    f"ratio={ratio} words={words}: boundary decision changed"
                )


class TestDocumentPathBudget:
    """The full document-ingest budget — a chunk at the shipped cap,
    used as the anonymize call's transcript INPUT, PLUS its
    extracted-facts JSON, PLUS the chunk ECHOED BACK as the
    ``anonymized_transcript`` rewrite — checked against the shipped
    envelope with an EXPLICIT facts term.

    This is the test that catches the facts term being folded away:
    ``TestDocMaxTokensDerivation``'s derivation-link test omits the facts
    term entirely (it only checks
    ``2 * cap_real + anon_template + reserve <= envelope``), so a
    revision of ``_DOC_MAX_TOKENS`` that folds the facts term into the
    template term (as an earlier revision of this derivation did —
    under-budgeting a dense chunk's anonymize call by ~4x, so a dense
    chunk at the cap could overrun the envelope at runtime) would pass
    that test while the real budget was broken.  This test recomputes the
    facts contribution explicitly at every run, so a future
    re-introduction of the folding bug (an inflated ``_DOC_MAX_TOKENS``)
    fails HERE.
    """

    # f: extracted-facts-JSON-to-chunk token ratio.  Derived (not
    # independently re-measured here) from
    # paramem/graph/extractor.py's recorded "Empirical worst-case observed
    # output for a dense resume chunk was ~2200 tokens" — a repo-recorded
    # measurement over the chunker's then-~1500-word max, matching the
    # numbers in document_chunker.py's _DOC_MAX_TOKENS derivation comment.
    _DENSE_CHUNK_WORDS = 1500
    _DENSE_CHUNK_OUTPUT_TOKENS = 2200

    def test_dense_chunk_at_the_shipped_cap_fits_the_full_envelope(self):
        from paramem.cloud.anonymize import (
            _DEFAULT_ANONYMIZER_TOKEN_ENVELOPE,
            _MAPPING_ENTRY_OVERHEAD_TOKENS,
            _OUTPUT_JSON_ENVELOPE_TOKENS,
        )

        reserve = _OUTPUT_JSON_ENVELOPE_TOKENS + _MAPPING_ENTRY_OVERHEAD_TOKENS
        dense_chunk_real_tokens = self._DENSE_CHUNK_WORDS * _R_PROSE
        f = self._DENSE_CHUNK_OUTPUT_TOKENS / dense_chunk_real_tokens

        cap_words = _DOC_MAX_TOKENS / MEASURED_TOKENS_PER_WORD
        chunk_real_tokens = cap_words * _R_PROSE

        facts_tokens = f * chunk_real_tokens
        chunk_input_tokens = chunk_real_tokens
        chunk_echoed_output_tokens = chunk_real_tokens
        anon_template_tokens = _live_anon_template_tokens()

        total = (
            anon_template_tokens
            + facts_tokens
            + chunk_input_tokens
            + chunk_echoed_output_tokens
            + reserve
        )
        assert total <= _DEFAULT_ANONYMIZER_TOKEN_ENVELOPE, (
            f"document-path budget broken at the shipped cap: "
            f"{total:.0f} > {_DEFAULT_ANONYMIZER_TOKEN_ENVELOPE} "
            f"(anon_template={anon_template_tokens}, "
            f"facts={facts_tokens:.0f}, chunk_in={chunk_input_tokens:.0f}, "
            f"chunk_out={chunk_echoed_output_tokens:.0f}, reserve={reserve})"
        )


class TestPdfChunkerMaxTokens:
    def test_section_above_max_tokens_is_split_at_sentence_boundary(self, tmp_path, monkeypatch):
        """Long page split into multiple chunks, never cutting mid-sentence."""
        # Build a single long page (>1500 words), each sentence ~6 words.
        sentences = [f"This is sentence number {i}." for i in range(400)]
        page = " ".join(sentences)
        fake = _fake_reader([page])
        monkeypatch.setattr("paramem.graph.document_chunker.pypdf.PdfReader", lambda path: fake)
        path = tmp_path / "big.pdf"
        path.write_bytes(b"%PDF-1.4 fake")
        chunks = chunk_pdf_file(path, min_tokens=200, max_tokens=500)
        # Should produce multiple chunks; each ≤ ~500 words; no mid-sentence cuts.
        assert len(chunks) > 1
        for c in chunks:
            # No chunk should end mid-sentence.
            assert c.text.rstrip().endswith((".", "!", "?"))

    def test_section_under_max_tokens_passes_through(self, tmp_path, monkeypatch):
        page = "Short content. Just a little. " * 30
        fake = _fake_reader([page])
        monkeypatch.setattr("paramem.graph.document_chunker.pypdf.PdfReader", lambda path: fake)
        path = tmp_path / "small.pdf"
        path.write_bytes(b"%PDF-1.4 fake")
        chunks = chunk_pdf_file(path, min_tokens=50, max_tokens=1500)
        assert len(chunks) == 1
