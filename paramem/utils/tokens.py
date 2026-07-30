"""THE token-estimation primitive — one estimator for every payload this
system sizes before a local ``generate()`` call.

Four independent renderings of "how big is this payload" existed before
this module: two identically try/except-guarded exact counts (each
returning a ``-1`` sentinel on failure) at
:func:`paramem.cloud.anonymize.anonymize_transcript` and
:func:`paramem.graph.extractor.judge_plausibility`, a third (same
guard/sentinel shape) in :func:`paramem.server.calibrate._count_tokens`,
and a fourth — tokenizer-free — whitespace-word counter in
:mod:`paramem.graph.document_chunker`. :func:`estimate_tokens` collapses
all four into one function: exact when a tokenizer is supplied, a
conservative words-based bound otherwise. Callers never re-implement
counting locally.

Boundary — why this module and not one of the four call sites' packages:
it cannot live in ``paramem.cloud`` (``document_chunker`` and the ingest
CLI must not import the cloud package), nor in ``paramem.graph``
(``paramem.cloud`` must not import ``paramem.graph`` — see
``paramem/cloud/anonymize.py``'s package docstring), nor in
``paramem.utils.vram_guard`` (that module owns device state and imports
``torch``; the estimator must stay importable by a tokenizer-free CLI).
``paramem.utils`` is the existing leaf home for exactly this kind of
dependency-light shared primitive (``identity.py``, ``paths.py``).

The fallback ratio (:data:`MEASURED_TOKENS_PER_WORD`) is MEASURED ONCE
with the production tokenizer (Mistral 7B,
``mistralai/Mistral-7B-Instruct-v0.3``, pinned by
``tests/fixtures/server.yaml``) over the three payload shapes the system
actually ingests: a representative conversation transcript, real ingested
document prose, and the compact fact-JSON rendering a graph-tier anonymize
call sends. The shipped value is the MAX of the three per-shape ratios,
rounded up, so the fallback BOUNDS every shape instead of averaging across
them — a single ratio calibrated on prose badly underestimates fact JSON,
which carries many tokens per whitespace-delimited word (punctuation,
quoting, and key names are each their own token(s) but not their own
"word"). The measurement script is intentionally not part of this
repository: it reads real personal records (for counts only), and neither
those records nor a script pointed at them may be committed. Only the
resulting integers are recorded in the comment on
:data:`MEASURED_TOKENS_PER_WORD`.

``paramem/server/config.py``'s ``consolidation.extraction_token_estimate_ratio``
carries the operator-facing copy of this same measurement, and
:func:`check_ratio_drift` is its boot-time staleness check — re-measuring
against the live tokenizer so a base-model swap that shifts the ratio does
not silently under-budget a call.
"""

from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)

# Fallback words->tokens ratio. MEASURED ONCE with the production tokenizer
# (Mistral 7B, mistralai/Mistral-7B-Instruct-v0.3, pinned by
# tests/fixtures/server.yaml) over the three payload shapes the system
# actually ingests. Value is the MAX of the three per-shape ratios, rounded
# up to 1 decimal: the fallback must bound, not average.
#   transcript shape    : 188 words   / 270 tokens  = 1.44 tokens/word
#   document shape (CV) : 1534 words  / 2934 tokens = 1.91 tokens/word
#   fact-JSON shape      : 2415 words / 8191 tokens = 3.39 tokens/word  <- MAX
# PUBLIC (no leading underscore): paramem.graph.document_chunker imports
# this cross-module to keep its own offline-derived _DOC_MAX_TOKENS
# constant in the SAME estimator unit as this module's runtime fallback:
# _DOC_MAX_TOKENS is a word cap MULTIPLIED by this ratio, and the chunker
# compares it against tokenizer-free estimate_tokens() output built from
# the same ratio, so the ratio cancels out of that comparison — but only
# while both sides read THIS constant. A private name masked that as an
# implementation detail when it is in fact a supported cross-module read
# surface.
MEASURED_TOKENS_PER_WORD: float = 3.4


def estimate_tokens(
    text: str,
    tokenizer: object | None = None,
    *,
    tokens_per_word: float = MEASURED_TOKENS_PER_WORD,
) -> int:
    """THE token-count primitive. Exact when *tokenizer* is given, bounding otherwise.

    Args:
        text: The text to count.
        tokenizer: A HuggingFace-style tokenizer (callable with
            ``add_special_tokens=False``, returning a mapping with an
            ``"input_ids"`` key). When given, the count is EXACT:
            ``len(tokenizer(text, add_special_tokens=False)["input_ids"])``.
            A tokenizer that raises (a ``MagicMock`` test fixture, a
            half-initialised fast tokenizer) falls back to the words-based
            estimate below and logs at DEBUG — this function NEVER returns a
            sentinel (e.g. ``-1``) for a raising tokenizer; a ``-1`` cost
            would make every payload "fit" a budget check downstream.
            Callers that need to distinguish "not measured" from "measured
            as zero" own that guard themselves at their own call site (see
            ``paramem/server/calibrate.py``'s ``if tokenizer`` /
            ``if tokenizer and count_str`` guards, which are unaffected by
            this function and still produce ``-1`` for "no tokenizer" / "no
            output"). ``None`` selects the fallback path directly, with no
            attempt to tokenize.
        tokens_per_word: Words->tokens ratio used only on the fallback path
            (no *tokenizer*, or a raising one). Defaults to
            :data:`MEASURED_TOKENS_PER_WORD` — the measured MAX across
            every payload shape this system ingests — so the fallback
            overestimates rather than risks silently under-budgeting a
            call. A caller holding its own freshly re-measured ratio (see
            :func:`check_ratio_drift`) may pass it explicitly.

    Returns:
        Token count. ``0`` for empty/whitespace-only *text* on either path.
        On the fallback path, non-empty text returns
        ``max(1, ceil(len(text.split()) * tokens_per_word))`` — always at
        least 1.
    """
    if tokenizer is not None:
        try:
            return len(tokenizer(text, add_special_tokens=False)["input_ids"])
        except Exception as exc:  # noqa: BLE001 — any tokenizer failure falls back
            logger.debug("estimate_tokens: tokenizer raised, falling back to estimate: %s", exc)
    words = len(text.split())
    if words == 0:
        return 0
    return max(1, math.ceil(words * tokens_per_word))


# Synthetic fixture samples for check_ratio_drift — one per payload shape
# that can dominate the ratio (transcript / document prose / fact-JSON).
# NEVER real ingested data (see the module docstring's privacy rule):
# a fictional transcript fragment, a fictional document-prose fragment, and
# a compact fact-JSON fragment using fictional entities, each sized enough
# to make the per-word ratio a meaningful re-measurement.
_DRIFT_SAMPLE_TRANSCRIPT = (
    "[user] I started a new job last month working on backend systems. "
    "[assistant] That sounds exciting, how has it been going so far? "
    "[user] Pretty good, the team is small and everyone has been really "
    "supportive during the transition."
)
# Continuous narrative prose (no dialogue markers) — the shape
# paramem.graph.document_chunker's ingested documents take, and the shape
# _DOC_MAX_TOKENS's r_prose conversion is calibrated against. Distinct in
# kind from the transcript sample above (turn-structured dialogue), not
# just in content, so a ratio regression specific to document ingestion
# (as opposed to transcripts) is still caught.
_DRIFT_SAMPLE_DOCUMENT = (
    "Jordan Ellery spent six years leading backend infrastructure teams "
    "at a mid-sized logistics company before moving into an independent "
    "consulting practice focused on distributed systems reliability. "
    "Prior roles included platform engineering at a payments startup, "
    "where the on-call rotation was rebuilt from scratch after a series "
    "of extended outages, and a research fellowship studying fault "
    "tolerance in replicated storage systems."
)
_DRIFT_SAMPLE_FACT_JSON = (
    '[{"subject": "speaker0", "predicate": "works at", "object": "acme corp", '
    '"relation_type": "factual", "confidence": 1.0}, '
    '{"subject": "speaker0", "predicate": "has hobby", "object": "hiking", '
    '"relation_type": "attribute", "confidence": 0.9}, '
    '{"subject": "acme corp", "predicate": "located in", "object": "springfield", '
    '"relation_type": "factual", "confidence": 0.8}]'
)
_DRIFT_SAMPLES: tuple[str, ...] = (
    _DRIFT_SAMPLE_TRANSCRIPT,
    _DRIFT_SAMPLE_DOCUMENT,
    _DRIFT_SAMPLE_FACT_JSON,
)


def check_ratio_drift(tokenizer, configured_ratio: float) -> float | None:
    """Re-measure the words->tokens ratio against *tokenizer* and flag drift.

    Runs :data:`_DRIFT_SAMPLES` (synthetic fixture strings — never real
    ingested data) through *tokenizer* and computes the observed
    tokens-per-word ratio for each. Returns the observed MAX when it
    EXCEEDS *configured_ratio* (the unsafe direction: the fallback ratio
    would then under-estimate a real payload of that shape); returns
    ``None`` when the live tokenizer's observed max is at or below
    *configured_ratio*.

    This is the boot-time consumer of
    ``consolidation.extraction_token_estimate_ratio`` — without it that
    config key would govern nothing (every in-process
    :func:`estimate_tokens` call site in production has a live tokenizer
    and never reaches the fallback path). The caller (server lifespan,
    after tokenizer load) logs a WARNING and records an
    ``/status.attention`` item on a non-``None`` return; this function does
    neither itself — it is a pure measurement.

    Args:
        tokenizer: The live production tokenizer to re-measure against.
        configured_ratio: The currently shipped/configured ratio to check
            against (``consolidation.extraction_token_estimate_ratio`` in
            production).

    Returns:
        The observed max ratio when it exceeds *configured_ratio*, else
        ``None``.
    """
    observed_max = 0.0
    for sample in _DRIFT_SAMPLES:
        words = len(sample.split())
        if words == 0:
            continue
        token_count = len(tokenizer(sample, add_special_tokens=False)["input_ids"])
        observed_max = max(observed_max, token_count / words)
    if observed_max > configured_ratio:
        return observed_max
    return None
