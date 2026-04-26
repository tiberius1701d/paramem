"""Format-specific document chunkers for the ParaMem ingest pipeline.

Provides:
- :class:`DocumentChunk` — frozen dataclass representing one chunk of a document.
- :func:`chunk_text_file` — plain-text (.txt) chunker; splits on blank-line
  boundaries and greedy-merges paragraphs up to ``max_tokens`` words.
- :func:`chunk_markdown_file` — Markdown chunker using ``markdown-it-py``; splits
  at H1/H2 ``heading_open`` tokens, preserves fenced code blocks, coalesces short
  sections.
- :func:`chunk_pdf_file` — PDF chunker using ``pypdf``; extracts text page by page,
  rejects image-only PDFs, coalesces short consecutive pages.
- :func:`chunk_document` — dispatch wrapper; routes by file suffix.

Custom exceptions:
- :exc:`UnsupportedFormatError` — file suffix not in (.txt, .md, .markdown, .pdf).
- :exc:`EmptyDocumentError` — file contains no extractable text.
- :exc:`ScannedPdfRejectedError` — PDF average chars/page < 50 (image-only PDF).

Token counting is approximate: ``len(text.split())``.  This avoids loading a
tokenizer in the CLI, which must be runnable without a GPU model in memory.

Design note — max vs min token asymmetry: plain text has no natural section
structure, so ``chunk_text_file`` caps chunk length from above (``max_tokens``)
to keep extractor prompts within model context.  Markdown and PDF have natural
boundaries (headings, pages); their chunkers coalesce short sections from below
(``min_tokens``) so each chunk carries enough context for extraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pypdf
from markdown_it import MarkdownIt
from markdown_it.token import Token


class UnsupportedFormatError(ValueError):
    """Raised when :func:`chunk_document` encounters an unrecognised file suffix."""


class EmptyDocumentError(ValueError):
    """Raised when the file contains no extractable text after stripping."""


class ScannedPdfRejectedError(ValueError):
    """Raised when the PDF average chars/page is below 50.

    This is a heuristic for image-only (scanned) PDFs that have not been OCR'd.
    OCR is not supported; the operator must supply a text-layer PDF.
    """


@dataclass(frozen=True)
class DocumentChunk:
    """One pre-chunked segment of a document, ready for server ingest.

    Attributes:
        chunk_index: Zero-based position of this chunk within the source file.
        text: The text content of the chunk.
        source_path: Absolute filesystem path of the original file (display only;
            the server never re-reads the file).
        doc_title: Human-readable document title — the filename stem.
        source_type: Fixed to ``"document"`` for all ingest-pipeline chunks.
    """

    chunk_index: int
    text: str
    source_path: str
    doc_title: str
    source_type: str = "document"


def _word_count(text: str) -> int:
    """Return approximate token count via whitespace split.

    Uses ``str.split()`` which splits on any whitespace run.  No tokenizer
    load required — this is intentionally approximate (the thresholds are
    heuristics, not hard limits).

    Args:
        text: Input text string.

    Returns:
        Number of whitespace-delimited tokens.
    """
    return len(text.split())


def chunk_text_file(path: Path, *, max_tokens: int = 1500) -> list[DocumentChunk]:
    """Chunk a plain-text file by greedy paragraph merging.

    Reads the file as UTF-8, splits on blank-line boundaries (consecutive
    empty lines), and greedily merges adjacent paragraphs until adding the
    next paragraph would exceed ``max_tokens`` words.  Each merged group
    becomes one :class:`DocumentChunk`.

    Args:
        path: Path to the ``.txt`` file.
        max_tokens: Approximate word-count ceiling per chunk.  Defaults to
            1500.

    Returns:
        List of :class:`DocumentChunk` instances, one per merged group.

    Raises:
        EmptyDocumentError: If the file is empty or contains only whitespace.
    """
    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        raise EmptyDocumentError(f"File is empty: {path}")

    # Split on blank lines (two or more newlines with optional whitespace between).
    # Use splitlines + state machine to avoid regex (see CLAUDE.md rule on regex).
    paragraphs: list[str] = []
    current_lines: list[str] = []

    for line in raw.splitlines():
        is_blank = not line.strip()
        if is_blank:
            if current_lines:
                # Flush accumulated non-blank lines as one paragraph.
                paragraphs.append("\n".join(current_lines))
                current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        paragraphs.append("\n".join(current_lines))

    # Filter out any empty paragraphs that slipped through.
    paragraphs = [p for p in paragraphs if p.strip()]

    if not paragraphs:
        raise EmptyDocumentError(f"File contains no extractable text: {path}")

    # Greedy merge: keep adding paragraphs until the next would overflow.
    chunks: list[str] = []
    current_parts: list[str] = []
    current_count = 0

    for para in paragraphs:
        para_count = _word_count(para)
        if current_parts and current_count + para_count > max_tokens:
            chunks.append("\n\n".join(current_parts))
            current_parts = [para]
            current_count = para_count
        else:
            current_parts.append(para)
            current_count += para_count

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    source_path_str = str(path.resolve())
    doc_title = path.stem
    return [
        DocumentChunk(
            chunk_index=i,
            text=chunk_text,
            source_path=source_path_str,
            doc_title=doc_title,
        )
        for i, chunk_text in enumerate(chunks)
    ]


def chunk_markdown_file(path: Path, *, min_tokens: int = 200) -> list[DocumentChunk]:
    """Chunk a Markdown file by H1/H2 section boundaries using ``markdown-it-py``.

    Parses the file with :class:`markdown_it.MarkdownIt`, walks the token stream,
    and splits at ``heading_open`` tokens with tag ``h1`` or ``h2``.  Fenced code
    blocks (``fence`` token type) are preserved verbatim as part of the enclosing
    section — ``#``-prefixed lines inside fences are **not** treated as heading
    boundaries.  YAML front-matter (leading ``---`` block) is stripped before
    parsing.

    Sections shorter than ``min_tokens`` words are coalesced with the following
    section until the merged group meets the threshold (or EOF is reached).

    Args:
        path: Path to the ``.md`` / ``.markdown`` file.
        min_tokens: Minimum word count per emitted chunk.  Short sections are
            coalesced until this threshold is met.  Defaults to 200.

    Returns:
        List of :class:`DocumentChunk` instances, one per section group.

    Raises:
        EmptyDocumentError: If the file is empty or contains no heading sections
            and no body text.
    """
    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        raise EmptyDocumentError(f"Markdown file is empty: {path}")

    # Strip YAML front-matter: leading block between two "---" lines.
    # markdown-it-py does not handle front-matter natively.
    text = _strip_front_matter(raw)

    md = MarkdownIt()
    tokens = md.parse(text)

    # Walk token stream and collect sections.
    # Each section starts at a heading_open (h1 or h2) and contains all
    # tokens until the next such heading or EOF.
    sections: list[str] = []
    current_parts: list[str] = []
    i = 0

    while i < len(tokens):
        token = tokens[i]

        if token.type == "heading_open" and token.tag in ("h1", "h2"):
            # Flush existing section.
            if current_parts:
                sections.append("\n\n".join(p for p in current_parts if p.strip()))
            current_parts = []
            # Collect the inline heading text (next token should be inline).
            heading_text = ""
            if i + 1 < len(tokens) and tokens[i + 1].type == "inline":
                heading_text = tokens[i + 1].content.strip()
            if heading_text:
                current_parts.append(heading_text)
            # Skip heading_open, inline, heading_close (3 tokens total).
            i += 3
            continue

        # Accumulate body content (skip structural open/close tokens).
        body_text = _token_text(token)
        if body_text:
            current_parts.append(body_text)

        i += 1

    # Flush the last section.
    if current_parts:
        sections.append("\n\n".join(p for p in current_parts if p.strip()))

    sections = [s for s in sections if s.strip()]

    if not sections:
        # No headings found — treat entire file as one chunk.
        full_text = text.strip()
        if not full_text:
            raise EmptyDocumentError(f"Markdown file contains no extractable text: {path}")
        sections = [full_text]

    # Coalesce short sections: merge a section into the next if it is below
    # min_tokens, until the merged group meets the threshold or EOF is reached.
    coalesced = _coalesce_sections(sections, min_tokens=min_tokens)

    source_path_str = str(path.resolve())
    doc_title = path.stem
    return [
        DocumentChunk(
            chunk_index=i,
            text=section_text,
            source_path=source_path_str,
            doc_title=doc_title,
        )
        for i, section_text in enumerate(coalesced)
    ]


def chunk_pdf_file(path: Path, *, min_tokens: int = 200) -> list[DocumentChunk]:
    """Chunk a PDF file using ``pypdf`` page-level extraction.

    Reads each page with :meth:`pypdf.PageObject.extract_text`, coalesces
    consecutive short pages (below ``min_tokens`` words) into larger chunks,
    and rejects image-only PDFs via an average-chars-per-page heuristic.

    Args:
        path: Path to the ``.pdf`` file.
        min_tokens: Minimum word count per emitted chunk.  Short consecutive
            pages are merged until this threshold is met.  Defaults to 200.

    Returns:
        List of :class:`DocumentChunk` instances.

    Raises:
        EmptyDocumentError: If no extractable text is found across all pages.
        ScannedPdfRejectedError: If the average chars/page is below 50, which
            indicates an image-only (scanned) PDF without a text layer.
    """
    reader = pypdf.PdfReader(str(path))
    pages = reader.pages

    page_texts: list[str] = []
    total_chars = 0
    for page in pages:
        text = page.extract_text() or ""
        page_texts.append(text)
        total_chars += len(text)

    page_count = len(pages)

    if total_chars == 0:
        raise EmptyDocumentError(f"PDF contains no extractable text: {path}")

    if page_count > 0:
        avg_chars = total_chars / page_count
        if avg_chars < 50:
            raise ScannedPdfRejectedError(
                f"Average {avg_chars:.1f} chars/page below 50 — image-only PDF, "
                f"OCR not supported: {path}"
            )

    # Coalesce consecutive pages greedily: keep adding pages until the current
    # chunk meets min_tokens or we run out of pages.
    sections: list[str] = []
    current_parts: list[str] = []
    current_count = 0

    for page_text in page_texts:
        stripped = page_text.strip()
        if not stripped:
            # Skip empty pages (no extractable text on this page).
            continue
        page_words = _word_count(stripped)
        current_parts.append(stripped)
        current_count += page_words
        if current_count >= min_tokens:
            sections.append("\n\n".join(current_parts))
            current_parts = []
            current_count = 0

    # Flush remainder.  If it is below min_tokens and there is already a
    # completed section, absorb it into that section rather than emitting a
    # standalone sub-threshold chunk.  This matches the markdown chunker's
    # coalesce policy for trailing short content.
    if current_parts:
        remainder = "\n\n".join(current_parts)
        if sections and _word_count(remainder) < min_tokens:
            sections[-1] = sections[-1] + "\n\n" + remainder
        else:
            sections.append(remainder)

    sections = [s for s in sections if s.strip()]

    if not sections:
        raise EmptyDocumentError(f"PDF contains no extractable text after filtering: {path}")

    source_path_str = str(path.resolve())
    doc_title = path.stem
    return [
        DocumentChunk(
            chunk_index=i,
            text=section_text,
            source_path=source_path_str,
            doc_title=doc_title,
        )
        for i, section_text in enumerate(sections)
    ]


def chunk_document(path: Path) -> list[DocumentChunk]:
    """Dispatch to the appropriate chunker based on file suffix.

    Supported suffixes: ``.txt``, ``.md``, ``.markdown``, ``.pdf``.

    Args:
        path: Path to the document file.  The file must exist and be readable
            (``FileNotFoundError`` / ``PermissionError`` propagate naturally
            from the underlying read operations).

    Returns:
        List of :class:`DocumentChunk` instances.

    Raises:
        UnsupportedFormatError: If the suffix is not in the supported set.
        EmptyDocumentError: If the file contains no extractable text.
        ScannedPdfRejectedError: If the PDF is image-only (average <50 chars/page).
    """
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return chunk_text_file(path)
    if suffix in (".md", ".markdown"):
        return chunk_markdown_file(path)
    if suffix == ".pdf":
        return chunk_pdf_file(path)
    raise UnsupportedFormatError(
        f"Unsupported file format: {path.suffix!r}. Supported formats: .txt, .md, .markdown, .pdf"
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _looks_like_yaml_kv(line: str) -> bool:
    """Return True if *line* looks like a YAML key-value entry or a blank line.

    Used by :func:`_strip_front_matter` to distinguish real YAML front-matter
    from a leading horizontal-rule (``---``) followed by prose.

    A line is accepted as YAML if it is blank, or if it contains ``:`` and does
    not start with ``#``, ``>``, or ``-`` (which indicate comments, block
    scalars, and list items that appear in prose but not in YAML header blocks).

    Args:
        line: A single line from the candidate front-matter block (no newline).

    Returns:
        ``True`` if the line is consistent with YAML front-matter.
    """
    stripped = line.strip()
    if not stripped:
        return True  # blank lines inside front-matter are fine
    if stripped[0] in ("#", ">", "-"):
        return False
    return ":" in stripped


def _strip_front_matter(text: str) -> str:
    """Strip YAML front-matter from a Markdown string.

    Front-matter is a block delimited by ``---`` on the first line and a
    closing ``---`` line, where every non-blank line between the delimiters
    looks like a ``key: value`` YAML entry (verified by
    :func:`_looks_like_yaml_kv`).  If the leading ``---`` is followed by prose
    rather than YAML key-value pairs, it is treated as a horizontal rule and
    the text is returned unchanged.

    Args:
        text: Raw Markdown text.

    Returns:
        Markdown text with front-matter removed, or the original text if no
        valid YAML front-matter block is present.
    """
    lines = text.splitlines(keepends=True)
    if not lines:
        return text
    first = lines[0].rstrip()
    if first != "---":
        return text
    # Find the closing "---" delimiter.
    for j in range(1, len(lines)):
        if lines[j].rstrip() == "---":
            # Verify that every non-blank line between the two delimiters
            # looks like a YAML key: value entry.  If any line fails the
            # check, the leading "---" is a horizontal rule, not front-matter.
            candidate_lines = lines[1:j]
            if not all(_looks_like_yaml_kv(ln.rstrip("\n")) for ln in candidate_lines):
                return text
            # Return everything after the closing delimiter.
            return "".join(lines[j + 1 :])
    # No closing delimiter — treat the whole file as front-matter-free.
    return text


def _token_text(token: Token) -> str:
    """Extract printable text from a ``markdown-it-py`` token.

    Handles ``inline`` tokens (return ``token.content``), ``fence`` tokens
    (return the fenced code block literal, including the opening/closing fence
    markers so the downstream reader knows it is a code block), and ignores
    structural open/close tokens that carry no text payload.

    Args:
        token: A ``markdown-it-py`` :class:`Token` object.

    Returns:
        Text contribution of this token, or an empty string for structural
        tokens that carry no human-readable content.
    """
    if token.type == "inline":
        return token.content.strip()
    if token.type == "fence":
        # Preserve code block verbatim (including markers).
        return token.markup + (token.info or "") + "\n" + token.content + token.markup
    if token.type == "html_block":
        return token.content.strip()
    # All other structural tokens (paragraph_open, bullet_list_open, etc.)
    # carry no text payload; inline children hold the content.
    return ""


def _coalesce_sections(sections: list[str], *, min_tokens: int) -> list[str]:
    """Coalesce consecutive short sections until each meets ``min_tokens`` words.

    Iterates over ``sections`` and merges a section forward into the next one
    when its word count is below ``min_tokens``.  The last section absorbs any
    trailing short sections regardless of length.

    Args:
        sections: List of text sections to potentially merge.
        min_tokens: Minimum word count threshold for emitting a section on its own.

    Returns:
        New list of sections with short groups merged.
    """
    if not sections:
        return []

    result: list[str] = []
    pending = sections[0]

    for section in sections[1:]:
        if _word_count(pending) < min_tokens:
            pending = pending + "\n\n" + section
        else:
            result.append(pending)
            pending = section

    result.append(pending)
    return result
