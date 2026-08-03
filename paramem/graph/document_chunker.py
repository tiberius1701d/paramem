"""Format-specific document chunkers for the ParaMem ingest pipeline.

Provides:
- :class:`DocumentChunk` — frozen dataclass representing one chunk of a document.
- :func:`chunk_text_file` — plain-text (.txt) chunker; splits on blank-line
  boundaries and greedy-merges paragraphs up to ``max_tokens`` estimated
  tokens.
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

Token counting goes through :func:`paramem.utils.tokens.estimate_tokens`
with no tokenizer, so it is an approximate, conservative estimate derived
from the whitespace word count — never an exact count.  This avoids
loading a tokenizer in the CLI, which must be runnable without a GPU model
in memory.  ``max_tokens``/``min_tokens`` below are estimated-token
thresholds under that same fallback, not raw word counts; see
``estimate_tokens``'s module docstring for where the fallback ratio comes
from.  The default ``max_tokens`` (:data:`_DOC_MAX_TOKENS`) is not chosen
independently — it is derived from the single per-call anonymize token
envelope a document chunk's downstream local calls must fit inside; see
the derivation comment above :data:`_DOC_MAX_TOKENS`.  ``min_tokens``
(:data:`_DOC_MIN_TOKENS`) is a context floor, not a budget, and is not
envelope-derived.

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

from paramem.utils.tokens import MEASURED_TOKENS_PER_WORD, estimate_tokens


class UnsupportedFormatError(ValueError):
    """Raised when :func:`chunk_document` encounters an unrecognised file suffix."""


class EmptyDocumentError(ValueError):
    """Raised when the file contains no extractable text after stripping."""


class ScannedPdfRejectedError(ValueError):
    """Raised when the PDF average chars/page is below 50.

    This is a heuristic for image-only (scanned) PDFs that have not been OCR'd.
    OCR is not supported; the operator must supply a text-layer PDF.
    """


# ---------------------------------------------------------------------------
# Document-chunk cap — derived from the anonymize-call token envelope
# ---------------------------------------------------------------------------
#
# Derived from the single per-call token envelope
# (``consolidation.extraction_anonymize_token_envelope``, default 8192 —
# ``paramem/cloud/anonymize.py``'s ``_DEFAULT_ANONYMIZER_TOKEN_ENVELOPE``),
# NOT an independent heuristic. A document chunk is consumed by two local
# calls, in order: session-tier extraction
# (``configs/prompts/extraction.txt``, the chunk fills its
# ``{transcript}`` slot — ``paramem/graph/flows.py``'s ``extract_graph``
# docstring: "document chunks land in the {transcript} slot of the same
# prompt"), then anonymize on the extracted facts AND the chunk itself
# (``paramem/graph/stage_anonymize.py`` threads ``ctx.transcript`` — the
# same chunk text — into ``anonymize(transcript=...)``). The binding call is
# the second one: its OUTPUT must echo the chunk back as the
# ``anonymized_transcript`` rewrite, so the chunk's real-token cost is paid
# TWICE against the SAME envelope (once as transcript input, once as echoed
# output) PLUS the extracted-facts JSON the same call also carries as input:
#
#     envelope >= anon_template + f*chunk + chunk + (chunk + reserve)
#     chunk_real_tokens <= (envelope - anon_template - reserve) / (2 + f)
#
# ``f`` is the extracted-facts-JSON-to-chunk token ratio — NOT folded into
# the template term (a prior revision of this derivation approximated it
# that way and under-budgeted the facts term by ~4x, which let a dense
# chunk's anonymize call overrun the envelope; carrying ``f`` as its own
# term is the fix). ``f`` is derived from a repo-recorded measurement, not
# re-measured here: ``paramem/graph/extractor.py``'s
# ``_DEFAULT_FILTER_MAX_TOKENS`` comment records "Empirical worst-case
# observed output for a dense resume chunk was ~2200 tokens" for the local
# extraction call over the chunker's then-1500-word max. That 2200-token
# figure is local EXTRACTION's raw
# output (the triples JSON an extraction call emits for a dense chunk),
# used here as the anonymize call's "facts" term proxy — the extracted
# facts an anonymize call renders into ``facts_json`` derive from that same
# extraction output (allowing for the second-order pass and normalization
# in between, not separately quantified). ``reserve`` is the anonymize
# response's fixed JSON-skeleton plus one mapping-entry's overhead
# (``paramem/cloud/anonymize.py``'s ``_OUTPUT_JSON_ENVELOPE_TOKENS`` = 48,
# ``_MAPPING_ENTRY_OVERHEAD_TOKENS`` = 10 — recorded here as plain numbers,
# not imported: this module must stay importable without the cloud package,
# see the module docstring above).
#
# ``anon_template`` is NOT a frozen tokenizer measurement (review
# follow-up: a fixed "3188 tok, tokenizer-measured" constant here, checked
# only against a wide word-count TOLERANCE BAND in the test suite, left
# ~4 tokens of real end-to-end slack that a +2-word template edit could
# already overrun while the band-check still passed). Instead it is
# DERIVED from the SAME r_prose ratio every other term in this comment
# uses, applied to the template's OWN live word count — self-consistent
# (one ratio, not a mix of an exact-tokenizer snapshot and an estimate),
# and EXACTLY reproducible from the checked-in prompt file with no
# tokenizer needed:
#     anon_template = ceil(words(configs/prompts/anonymization.txt
#                     rendered empty) * r_prose)
# ``tests/test_document_chunker.py``'s derivation tests recompute this
# SAME formula from the live file on every run and use it directly (exact
# equality — no tolerance band, none needed: the formula is deterministic)
# — so a prompt edit that changes the template's word count is caught by
# construction, not by a hand-maintained magic number going stale.
# ``_DOC_MAX_TOKENS`` below is still a plain recorded literal, NOT computed
# at import time: ``configs/prompts/`` is not packaged
# (``pyproject.toml``'s ``[tool.setuptools.package-data]`` ships only
# ``web/static/*`` and ``training/donor_fixture.json``) and this module's
# own docstring above states the CLI must stay portable to a host that
# does not have the operator's server-side config/prompts checked out — an
# import-time file read here would make `import
# paramem.graph.document_chunker` (and therefore the CLI) fail outside a
# full repo checkout.
#
# The rejected alternative was to compute this constant (and, upstream,
# the words->tokens ratio itself) at import time from the live
# yaml/prompt files, so that no recorded number could ever go stale. It
# was rejected for the reason just given: a config-free, tokenizer-free
# import is what makes both this module and
# ``paramem/utils/tokens.py`` usable from a CLI that has no server-side
# checkout and no model in memory, and an import-time read trades that
# property away for staleness protection that a test can provide instead.
# Hence both are compiled-in literals whose honesty is enforced from the
# outside: the ratio by a boot-time drift check against its
# ``consolidation.extraction_token_estimate_ratio`` yaml mirror (config
# load fails on mismatch), and this cap by the exact-equality derivation
# test described above, which recomputes the same formula from the live
# prompt file on every run.
#
# Measured with the production tokenizer (Mistral 7B, 2026-07-28) for
# ``f`` only, and COUNTS ONLY: that measurement runs over real personal
# records, so nothing but the resulting integers may be written down here
# or anywhere else tracked — never the measured text, and never the script
# that reads it (the same rule that keeps ``paramem/utils/tokens.py``'s
# ratio-measurement script out of this repository). ``f``'s numerator is
# the repo-recorded extractor.py figure; ``anon_template`` and ``r_prose``
# need no tokenizer at all, per above:
#   envelope               = 8192 tok  (the single anonymize-call budget)
#   anon_template words    = 1535 words (configs/prompts/anonymization.txt,
#               rendered empty, current file — recomputed live by tests;
#               shrank from 1684 words 2026-08-02 when the fold-onto-token
#               speaker-anchor rule + worked example moved out into their
#               own companion prompt, ``anonymization_speaker_anchor.txt``,
#               rendered only when a speaker anchor is actually threaded)
#   r_prose   = 1.9126 tokens/word  (document shape — the same
#               measurement recorded as tokens.py's "document shape (CV)"
#               row feeding ``MEASURED_TOKENS_PER_WORD``)
#   anon_template = ceil(1535 * 1.9126)                          =  2936 tok
#   reserve   = 48 (json envelope) + 10 (one mapping entry)      =    58 tok
#   dense_chunk_real_tokens = 1500 words * r_prose (extractor.py's ~1500-word
#               dense-chunk reference point)                    ~=  2869 tok
#   f = 2200 (extractor.py's recorded dense-chunk output) / 2869 ~=  0.77
#   chunk_real_tokens = (8192 - 2936 - 58) / (2 + 0.77)         ~=  1879 tok
#   cap_words = floor(chunk_real_tokens / r_prose)
#             = floor(1879 / 1.9126)                              =   982 words
#
# ``cap_words`` above is a REAL-token derivation, but the shipped constant
# is deliberately NOT stored in real tokens — it is re-expressed in the
# estimator's own unit:
#     _DOC_MAX_TOKENS = cap_words * MEASURED_TOKENS_PER_WORD = 982 * 3.4 = 3338 tok
# That is why the MAX ratio (3.4, calibrated on fact JSON) does not shrink
# the cap even though it over-counts prose by ~1.8x. Every runtime boundary
# check is ``estimate_tokens(text) <= max_tokens``, i.e.
# ``words * ratio <= cap_words * ratio``: the ratio appears on BOTH sides
# and cancels, so the decision reduces to ``chunk_words <= cap_words``
# whatever the ratio happens to be. Store the cap in real tokens instead
# and the cancellation is lost — the same 3.4 estimator would then measure
# a 982-word prose chunk as 3338 "tokens" against an 1879-real-token
# ceiling and cut documents to roughly half the derived length, and any
# base-model swap that shifts the ratio would silently move the cap.
# Pinned by ``tests/test_document_chunker.py``'s
# ``TestDocMaxTokensDerivation::test_ratio_cancellation_invariant``.
_DOC_MAX_TOKENS: int = 3338
# Context floor, not a budget (unlike _DOC_MAX_TOKENS above, this is not
# derived from the envelope). Real threshold is unchanged from before this
# migration (200 words — "enough context for extraction"), re-expressed in
# the estimator's own unit: 200 * MEASURED_TOKENS_PER_WORD = 680.
_DOC_MIN_TOKENS: int = 680


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
    """Return the estimated token count for *text* via the shared estimator.

    Delegates to :func:`paramem.utils.tokens.estimate_tokens` with no
    tokenizer — the CLI-safe fallback path (whitespace word count times the
    measured conservative ratio).  No tokenizer load required — this is
    intentionally approximate (the thresholds this feeds are heuristics,
    not hard limits).  The name is retained (word-count semantics dominate
    every call site's local variable naming: ``para_count``, ``page_words``)
    even though the return value is now token-estimated, not a raw word
    count.  ``_coalesce_to_max_tokens``' own ``s_count`` does NOT go
    through this helper — see the comment there.

    Args:
        text: Input text string.

    Returns:
        Estimated token count (0 for empty/whitespace-only text).
    """
    return estimate_tokens(text)


def chunk_text_file(path: Path, *, max_tokens: int = _DOC_MAX_TOKENS) -> list[DocumentChunk]:
    """Chunk a plain-text file by greedy paragraph merging.

    Reads the file as UTF-8, splits on blank-line boundaries (consecutive
    empty lines), and greedily merges adjacent paragraphs until adding the
    next paragraph would exceed ``max_tokens`` estimated tokens.  Each
    merged group becomes one :class:`DocumentChunk`.

    Args:
        path: Path to the ``.txt`` file.
        max_tokens: Estimated-token ceiling per chunk (see
            :func:`paramem.utils.tokens.estimate_tokens`).  Defaults to
            :data:`_DOC_MAX_TOKENS`, derived from the anonymize-call token
            envelope (see the derivation comment above that constant).

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


def chunk_markdown_file(path: Path, *, min_tokens: int = _DOC_MIN_TOKENS) -> list[DocumentChunk]:
    """Chunk a Markdown file by H1/H2 section boundaries using ``markdown-it-py``.

    Parses the file with :class:`markdown_it.MarkdownIt`, walks the token stream,
    and splits at ``heading_open`` tokens with tag ``h1`` or ``h2``.  Fenced code
    blocks (``fence`` token type) are preserved verbatim as part of the enclosing
    section — ``#``-prefixed lines inside fences are **not** treated as heading
    boundaries.  YAML front-matter (leading ``---`` block) is stripped before
    parsing.

    Sections below ``min_tokens`` estimated tokens are coalesced with the
    following section until the merged group meets the threshold (or EOF is
    reached).

    Args:
        path: Path to the ``.md`` / ``.markdown`` file.
        min_tokens: Minimum estimated-token count per emitted chunk (see
            :func:`paramem.utils.tokens.estimate_tokens`).  Short sections
            are coalesced until this threshold is met.  Defaults to
            :data:`_DOC_MIN_TOKENS` — a context floor, not an
            envelope-derived budget.

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


def _split_into_sentences(text: str) -> list[str]:
    """Greedy sentence split on ``.``/``!``/``?`` (followed by space or EOL).

    Trailing closing-quote / bracket characters after the terminator are
    tolerated.  Whitespace-only fragments are filtered out.  This is a
    deliberate-no-regex implementation: we walk tokens once and break the
    accumulator at every token whose stripped form ends with one of the
    terminators.  Fallback for pathological text with no terminators: a
    single-element list.
    """
    sentences: list[str] = []
    current: list[str] = []
    for token in (text or "").split():
        current.append(token)
        # Strip trailing closers before testing the terminator so '"hi."' splits.
        bare = token.rstrip("\"')]}»")
        if bare.endswith((".", "!", "?")):
            sentences.append(" ".join(current))
            current = []
    if current:
        sentences.append(" ".join(current))
    return [s for s in sentences if s.strip()]


def _overlap_tokens_to_words(overlap_tokens: int) -> int:
    """Convert a token-denominated overlap budget into a word count to slice.

    ``overlap_tokens`` is expressed in the estimator's unit (matching
    ``max_tokens``/``min_tokens`` everywhere else in this module), but the
    overlap tail is realised by slicing actual WORDS out of the
    just-emitted chunk's text (``.split()[-n:]``) — there is no other way
    to carry a literal word-level prefix forward.

    The conversion deliberately uses
    :data:`~paramem.utils.tokens.MEASURED_TOKENS_PER_WORD` (the module's
    uniform fallback ratio, 3.4 — the fact-JSON-calibrated MAX, not
    ``r_prose``), NOT the document-prose ratio ``r_prose`` (1.9126) that
    :data:`_DOC_MAX_TOKENS`'s one-off OFFLINE derivation uses.  Every
    RUNTIME comparison in this module — ``max_tokens``, ``min_tokens``,
    every ``s_count``/``current_count`` via plain
    :func:`~paramem.utils.tokens.estimate_tokens` calls with no tokenizer —
    is computed in that SAME 3.4-calibrated unit, and this function's
    result feeds directly into that same running ``current_count``.  Using
    ``r_prose`` here would mix units within one comparison (correct against
    a real-token target, inconsistent against every other quantity in the
    same accumulator) — the opposite of the unit coherence this module's
    counting is built on, where every threshold and every running total is
    denominated in the one estimator unit. The
    practical effect: a caller's ``overlap_tokens=N`` request resolves to
    fewer WORDS than ``N`` divided by the document's true prose ratio would
    give (``r_prose / MEASURED_TOKENS_PER_WORD`` ≈ 56% of that), i.e. this
    function is itself conservative about how much text it carries forward
    — consistent with every other conservative bound in this module.  No
    in-repo caller passes ``overlap_tokens > 0`` today (verified by grep);
    revisit this trade-off if one starts to.

    Returns 0 for ``overlap_tokens <= 0``; otherwise at least 1 word for
    any positive ``overlap_tokens``.
    """
    if overlap_tokens <= 0:
        return 0
    return max(1, round(overlap_tokens / MEASURED_TOKENS_PER_WORD))


def _coalesce_to_max_tokens(
    sentences: list[str], *, max_tokens: int, overlap_tokens: int = 0
) -> list[str]:
    """Greedy-merge sentences into chunks whose estimated-token count is
    ≤ ``max_tokens``.

    When adding the next sentence would exceed ``max_tokens`` (measured via
    :func:`~paramem.utils.tokens.estimate_tokens`, the same primitive every
    other counting site in this module uses), emit the accumulated chunk
    and start a new one.  If ``overlap_tokens > 0``, the trailing WORD span
    of the just-emitted chunk whose estimated-token cost is closest to
    ``overlap_tokens`` (see :func:`_overlap_tokens_to_words`) is carried
    into the next chunk's prefix — preserves cross-chunk context for any
    sentence that straddles a chunk boundary in the source.  The tail is
    necessarily WORD-sliced (that is the unit the source text is carried
    in); its re-measured estimated-token cost, not ``overlap_tokens``
    itself, is what feeds the new chunk's running ``current_count`` so the
    accounting stays in one unit throughout.

    Sentences whose own estimated-token count exceeds ``max_tokens`` are
    kept whole (no mid-sentence cut); the chunk for that sentence will
    exceed the cap. The caller treats the cap as a target, not a hard
    ceiling.
    """
    if max_tokens <= 0:
        return [" ".join(sentences)] if sentences else []
    overlap_words = _overlap_tokens_to_words(overlap_tokens)
    chunks: list[str] = []
    current: list[str] = []
    current_count = 0
    for sent in sentences:
        s_count = estimate_tokens(sent)
        if current and current_count + s_count > max_tokens:
            chunks.append(" ".join(current))
            if overlap_words > 0:
                tail_words = (" ".join(current)).split()[-overlap_words:]
                tail_text = " ".join(tail_words)
                current = [tail_text, sent] if tail_words else [sent]
                current_count = (estimate_tokens(tail_text) if tail_words else 0) + s_count
            else:
                current = [sent]
                current_count = s_count
        else:
            current.append(sent)
            current_count += s_count
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_pdf_file(
    path: Path,
    *,
    min_tokens: int = _DOC_MIN_TOKENS,
    max_tokens: int = _DOC_MAX_TOKENS,
    overlap_tokens: int = 0,
) -> list[DocumentChunk]:
    """Chunk a PDF file using ``pypdf`` page-level extraction.

    Reads each page with :meth:`pypdf.PageObject.extract_text`, coalesces
    consecutive short pages (below ``min_tokens`` estimated tokens) into
    larger sections, and then splits any section above ``max_tokens``
    estimated tokens at sentence boundaries (so a chunk never cuts
    mid-sentence — important for cloud enrichment context that depends on
    resolving "scaling teams from 15 to 100+ engineers" as one fact, not
    two halves).

    Image-only PDFs are rejected via an average-chars-per-page heuristic.

    Args:
        path: Path to the ``.pdf`` file.
        min_tokens: Minimum estimated-token count per emitted section (see
            :func:`paramem.utils.tokens.estimate_tokens`).  Short
            consecutive pages are merged until this threshold is met.
            Defaults to :data:`_DOC_MIN_TOKENS` — a context floor, not an
            envelope-derived budget.
        max_tokens: Estimated-token ceiling per emitted chunk.  Sections
            above this size are split at sentence boundaries.  A single
            sentence whose own estimated-token count exceeds ``max_tokens``
            is kept whole — the chunk will exceed the cap rather than cut
            mid-sentence.  Defaults to :data:`_DOC_MAX_TOKENS`, derived from
            the anonymize-call token envelope (see the derivation comment
            above that constant).
        overlap_tokens: When splitting a section, the trailing word span of
            the just-emitted chunk whose estimated-token cost is closest to
            this many tokens is copied into the next chunk's prefix (see
            :func:`_overlap_tokens_to_words`).  Useful for preserving
            cross-chunk references.  Defaults to 0.

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

    # Apply max_tokens cap with sentence-aware splitting: oversize sections
    # are broken at sentence boundaries so cloud enrichment never sees a
    # truncated antecedent.
    final_sections: list[str] = []
    for section in sections:
        if _word_count(section) <= max_tokens:
            final_sections.append(section)
            continue
        sentences = _split_into_sentences(section)
        split_chunks = _coalesce_to_max_tokens(
            sentences, max_tokens=max_tokens, overlap_tokens=overlap_tokens
        )
        final_sections.extend(split_chunks)

    source_path_str = str(path.resolve())
    doc_title = path.stem
    return [
        DocumentChunk(
            chunk_index=i,
            text=section_text,
            source_path=source_path_str,
            doc_title=doc_title,
        )
        for i, section_text in enumerate(final_sections)
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
    """Coalesce consecutive short sections until each meets ``min_tokens``
    estimated tokens.

    Iterates over ``sections`` and merges a section forward into the next one
    when its estimated-token count is below ``min_tokens``.  The last section
    absorbs any trailing short sections regardless of length.

    Args:
        sections: List of text sections to potentially merge.
        min_tokens: Minimum estimated-token threshold for emitting a section
            on its own.

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
