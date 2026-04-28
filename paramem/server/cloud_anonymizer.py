"""Inference-side anonymizer for cloud egress.

Wraps the graph-extraction primitives (``_anonymize_with_local_model``,
``_substitute_whole_words``, ``_contains_whole_word``) for the inference
hot path.  Used by ``handle_chat`` when ``sanitization.cloud_mode`` is
``"anonymize"`` or ``"both"``: outbound text is rewritten with placeholders
before reaching SOTA, and the response is rewritten back via the reverse
mapping.

Privacy contract (per-call):

* The caller passes plain text (a query, a model response from
  ``[ESCALATE]``, etc.) to :func:`anonymize_outbound`.
* The local LLM produces a redacted version + a ``{real_name → placeholder}``
  mapping.  The mapping is canonicalised before return so the reverse
  direction is total.
* A forward-path leak guard scans the redacted text for any real name in
  the mapping.  If any survives, the call returns ``("", {})`` — caller
  treats this as block-for-this-query.
* :func:`deanonymize_inbound` applies the reverse mapping with word-
  boundary substitution.  Idempotent on text without placeholders.

Failure is non-raising: any exception (model OOM, JSON parse error, leak
detected) returns ``("", {})`` so the caller's per-query fallback fires.
"""

from __future__ import annotations

import logging

from paramem.graph.extractor import (
    _anonymize_with_local_model,
    _contains_whole_word,
    _substitute_whole_words,
)
from paramem.graph.schema import SessionGraph

logger = logging.getLogger(__name__)


def anonymize_outbound(text: str, model, tokenizer) -> tuple[str, dict[str, str]]:
    """Run text through the local-LLM anonymizer.

    Returns ``(anon_text, mapping)`` where ``mapping`` is canonicalised
    in the ``real_name → placeholder`` direction.  On any failure
    (anonymizer didn't run, mapping incomplete, leak guard tripped),
    returns ``("", {})`` so callers can treat empty mapping as a
    per-query block signal.
    """
    if not text or not text.strip():
        return text, {}

    try:
        empty_graph = SessionGraph(entities=[], relations=[])
        _anon_facts, mapping, anon_transcript = _anonymize_with_local_model(
            empty_graph, model, tokenizer, transcript=text
        )
    except Exception:
        logger.exception("Cloud anonymizer raised; treating as per-query block")
        return "", {}

    if not anon_transcript:
        # Fallback: anonymizer prompt didn't return an anonymized transcript.
        # Without the redacted text we cannot satisfy the privacy contract;
        # block for this query.
        logger.info("Cloud anonymizer returned empty transcript; blocking query")
        return "", {}

    if not mapping:
        # No mapping = the model decided there was nothing personal to
        # redact.  The text is safe to send as-is.
        return anon_transcript, {}

    # Forward-path leak guard: any real name still present in the
    # redacted text means anonymization didn't cover that occurrence.
    # Sending it would leak; fall back to block.
    leaked = [name for name in mapping if _contains_whole_word(anon_transcript, name)]
    if leaked:
        logger.warning(
            "Cloud anonymizer leaked %d name(s); blocking query (sample=%r)",
            len(leaked),
            leaked[:3],
        )
        return "", {}

    return anon_transcript, mapping


def deanonymize_inbound(text: str, mapping: dict[str, str]) -> str:
    """Substitute placeholders back to real names using the reverse mapping.

    ``mapping`` is the forward direction (``real → placeholder``); this
    function inverts it internally.  Word-boundary anchored so a
    placeholder embedded in unrelated text doesn't match.

    Idempotent when ``text`` contains no placeholders or ``mapping`` is
    empty — returns ``text`` unchanged.
    """
    if not text or not mapping:
        return text

    reverse = {v: k for k, v in mapping.items() if isinstance(k, str) and isinstance(v, str)}
    return _substitute_whole_words(text, reverse)
