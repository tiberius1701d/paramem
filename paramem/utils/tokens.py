"""THE token-estimation primitive — one estimator for every payload this
system sizes before a local ``generate()`` call.

Independent renderings of "how big is this payload" once existed at
multiple local-generate call sites: try/except-guarded exact counts (each
returning a ``-1`` sentinel on failure) such as
:func:`paramem.graph.extractor.judge_plausibility` and
:func:`paramem.server.calibrate._count_tokens`, and a tokenizer-free
whitespace-word counter in :mod:`paramem.graph.document_chunker`.
:func:`estimate_tokens` collapses every one of those into one function:
exact when a tokenizer is supplied, a conservative words-based bound
otherwise. Callers never re-implement counting locally.

Boundary — why this module and not one of the four call sites' packages:
it cannot live in ``paramem.cloud`` (``document_chunker`` and the ingest
CLI must not import the cloud package), nor in ``paramem.graph``
(``paramem.cloud`` must not import ``paramem.graph`` — see
``paramem/cloud/anonymize.py``'s package docstring), nor in
``paramem.utils.vram_guard`` (that module owns device state and imports
``torch``; the estimator must stay importable by a tokenizer-free CLI).
``paramem.utils`` is the existing leaf home for exactly this kind of
dependency-light shared primitive (``identity.py``, ``paths.py``).

Boundary, part two — the encode chokepoint (added when this module gained
:class:`RenderedPrompt` and :func:`encode_rendered`): this module owns not
only token *counting* but also the *encode* boundary for chat-template
rendered text. :func:`~paramem.models.loader.render_chat_prompt` is the one
production renderer (``tokenizer.apply_chat_template(..., tokenize=False)``);
:func:`encode_rendered` is the one production tensorizer for its output
(``tokenizer(text, add_special_tokens=False, ...)``). Both exist so a
rendered chat template's own literal BOS token is never joined by a second,
tokenizer-default BOS from a naive re-encode. The two functions in this
module that touch a tokenizer have deliberately DIVERGENT failure contracts
— read the one that applies before assuming either behaviour:
:func:`estimate_tokens` (below) swallows any tokenizer exception and falls
back to the words-based estimate (see its own docstring and its ``except
Exception`` fallback branch); :func:`encode_rendered` never swallows — a
raising tokenizer call propagates unchanged, because a sizing estimate may
safely degrade to a bound, but a tensorization that will actually be fed to
``model.generate()``/a training step must fail loudly rather than silently
produce wrong tensors.

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


class RenderedPrompt(str):
    """Marker subclass for chat-template-rendered text.

    A plain ``str`` subclass with no behavior of its own — its only job is
    to let :func:`encode_rendered` distinguish "text that already went
    through :func:`~paramem.models.loader.render_chat_prompt`" from an
    arbitrary ``str`` that has not been through a chat template at all (and
    so may be missing the template's own literal BOS, or may need one added
    by the tokenizer). Every production renderer wraps its output in this
    type; every production tensorizer of rendered text requires it.
    """


def encode_rendered(tokenizer, text: "RenderedPrompt | list[RenderedPrompt]", **tokenizer_kwargs):
    """THE tensorizer for chat-template-rendered text — ``add_special_tokens=False`` always.

    A rendered chat template already carries its own literal BOS token (and
    any other template-inserted special tokens). Encoding it with a
    tokenizer's default ``add_special_tokens=True`` re-adds a second BOS on
    top of the template's own — the double-BOS bug this function exists to
    make structurally impossible at every call site that routes through it.

    Args:
        tokenizer: A HuggingFace-style tokenizer, callable as
            ``tokenizer(text_or_list, add_special_tokens=False,
            **tokenizer_kwargs)``.
        text: A single :class:`RenderedPrompt`, or a list of them (for a
            batched encode). A plain ``str`` (or a list containing one) is
            rejected — see ``Raises`` below.
        **tokenizer_kwargs: Forwarded verbatim to the tokenizer call (e.g.
            ``return_tensors``, ``padding``, ``truncation``, ``max_length``).
            ``add_special_tokens`` may not be passed here — it is always
            ``False``, fixed by this function.

    Returns:
        Whatever the tokenizer call returns (typically a ``BatchEncoding``).

    Raises:
        TypeError: When *text* (or any element of a list *text*) is a plain
            ``str``, not a :class:`RenderedPrompt` — the caller skipped
            :func:`~paramem.models.loader.render_chat_prompt` and is about
            to double-BOS (or under-BOS) the encode. The message names
            ``render_chat_prompt`` as the fix.
        Exception: Any exception the tokenizer itself raises propagates
            unchanged — this function never catches a tokenizer failure.
    """
    if isinstance(text, list):
        for item in text:
            if not isinstance(item, RenderedPrompt):
                raise TypeError(
                    "encode_rendered: list element is a plain str, not a RenderedPrompt — "
                    "render it first via paramem.models.loader.render_chat_prompt"
                )
    elif not isinstance(text, RenderedPrompt):
        raise TypeError(
            "encode_rendered: text is a plain str, not a RenderedPrompt — "
            "render it first via paramem.models.loader.render_chat_prompt"
        )
    return tokenizer(text, add_special_tokens=False, **tokenizer_kwargs)


# Fallback words->tokens ratio. MEASURED ONCE with the production tokenizer
# (Mistral 7B, mistralai/Mistral-7B-Instruct-v0.3, pinned by
# tests/fixtures/server.yaml) over the three payload shapes the system
# actually ingests. Value is the MAX of the per-shape ratios, rounded up to 1
# decimal: the fallback must bound, not average.
#   transcript shape (re-measured 2026-08-03, supersedes the original
#     188 words / 270 tokens = 1.44 tokens/word measurement, over
#     conversational session transcript/extraction pairs) : 1.54 tokens/word
#   document shape (CV)                    : 1534 words / 2934 tokens = 1.91 tokens/word
#   fact-JSON shape (2026-07-28 measurement)   : 2415 words / 8191 tokens = 3.39 tokens/word
#   fact-JSON shape (2026-08-03 drift re-measurement of the same shape,
#     recorded beside the original since both bound it)  : 3.657 tokens/word  <- MAX
# PUBLIC (no leading underscore): paramem.graph.document_chunker imports
# this cross-module to keep its own offline-derived _DOC_MAX_TOKENS
# constant in the SAME estimator unit as this module's runtime fallback:
# _DOC_MAX_TOKENS is a word cap MULTIPLIED by this ratio, and the chunker
# compares it against tokenizer-free estimate_tokens() output built from
# the same ratio, so the ratio cancels out of that comparison — but only
# while both sides read THIS constant. A private name masked that as an
# implementation detail when it is in fact a supported cross-module read
# surface.
MEASURED_TOKENS_PER_WORD: float = 3.7

# ---------------------------------------------------------------------------
# Anonymize-envelope budget primitives — the ONE encoding of
# "envelope - skeleton - reserve" (payload carried once) plus the
# ratio-cancellation unit rule, shared by every caller that must fit a
# payload (a document chunk, a conversation transcript) inside one
# anonymize-call token envelope. The anonymizer's SCAN step is a span
# tagger — no envelope, no local ``generate()`` — so there is exactly ONE
# local-call shape left to size a payload against: the ANCHOR
# self-introduction question, which carries the payload once. See
# :func:`anonymize_payload_cap_tokens`'s docstring for the identity itself.
# ---------------------------------------------------------------------------

# Total tokens (prompt + output) one local anonymize() call may occupy.
# THE single executable home for this literal: paramem.cloud.anonymize's
# ``_DEFAULT_ANONYMIZER_TOKEN_ENVELOPE`` reads this constant rather than
# carrying a second copy — document_chunker (which must stay importable
# without the cloud package) and session_buffer both need the value, so a
# third recorded literal would have existed without this inversion.
ANONYMIZE_ENVELOPE_TOKENS: int = 8192

# ANCHOR prompt skeleton — the fixed system-prompt + chat-markup + call-body
# token cost of the one remaining local anonymize call, excluding the
# candidate values list (``{values}``) and the evidence text (``{text}``).
# Measured 2026-08-24 via the ACTUAL runtime render path
# (``paramem.models.loader.render_chat_prompt`` over the ``ANCHOR-SYSTEM`` +
# ``ANCHOR`` sections as
# ``paramem.graph.anonymizer_prompts.load_anonymizer_prompts`` composes
# them, ``{speaker_id}`` filled with ``"speaker1"``, ``{values}`` an empty
# JSON array, ``{text}`` empty, ``add_generation_prompt=True``), counted
# with the production tokenizer (Mistral 7B,
# ``mistralai/Mistral-7B-Instruct-v0.3``), CPU-only (tokenizer load, no
# model, no GPU).
ANONYMIZE_ANCHOR_PROMPT_SKELETON_TOKENS: int = 523

# ---------------------------------------------------------------------------
# ANCHOR OUTPUT reserve constants — re-measured 2026-08-24 against the
# shipped JSON envelope shape (``{"self_introduced": [...]}``), via a direct
# tokenizer call on the envelope text (no chat-template render — this is the
# MODEL's own completion, not a rendered prompt). Base measurements (a
# markdown-code-fenced empty envelope, plus a handful of realistic person
# names) were 5-15 tokens; each constant below adds slack over that base for
# output-format variation (fence style, whitespace) the exact base samples
# do not exhaust.
# ---------------------------------------------------------------------------
ANONYMIZE_ANCHOR_OUTPUT_SKELETON_TOKENS: int = 20  # `{"self_introduced": []}` fenced + slack
ANONYMIZE_MIN_ANCHOR_OUTPUT_TOKENS: int = 24
# Per-candidate-surface overhead (quotes + comma + separator) for ONE
# candidate value — `"...",`, no placeholder value is emitted (unlike a
# {real: placeholder} mapping entry). Consumed by
# :func:`anchor_output_reserve_tokens` below, never applied per WORD of the
# evidence text the call was shown — ANCHOR output is bounded by the number
# of candidate VALUES the call was shown, never by how much evidence text
# it was shown.
ANONYMIZE_ANCHOR_ENTRY_OVERHEAD_TOKENS: int = 6
# Structural ceiling on the number of distinct candidate surfaces one
# ANCHOR call's reserve is sized for — a reasoned (not live-measured) cap,
# comfortably above the person-row candidate counts any single call is
# expected to carry, and its resulting reserve (below) stays well under the
# 8192-token envelope. A call whose TRUE output would exceed this cap (a
# name-dense outlier) truncates at `max_new_tokens`, fails JSON parsing, and
# fails closed — the safe direction, never a silent under-reserve.
ANONYMIZE_ANCHOR_MAX_CANDIDATES: int = 100

# Conversation-transcript prose ratio (the session-tier payload shape),
# measured 2026-08-03 against the production tokenizer over real
# transcript/extraction pairs (counts only — see the module docstring's
# privacy rule; the median of 4 pairs, ~7% above the previous 1.44 estimate).
TRANSCRIPT_TOKENS_PER_WORD: float = 1.54


def anchor_output_reserve_tokens(candidate_count: int) -> int:
    """THE one ANCHOR-call output-reserve formula: bounded by the number of
    distinct candidate VALUES a call was shown, never by the size of the
    evidence text.

    ``max(ANONYMIZE_MIN_ANCHOR_OUTPUT_TOKENS,
    ANONYMIZE_ANCHOR_OUTPUT_SKELETON_TOKENS +
    ANONYMIZE_ANCHOR_ENTRY_OVERHEAD_TOKENS * min(candidate_count,
    ANONYMIZE_ANCHOR_MAX_CANDIDATES))`` — *candidate_count* is clamped to
    :data:`ANONYMIZE_ANCHOR_MAX_CANDIDATES` so the reserve plateaus at a
    constant well under the envelope regardless of how large
    *candidate_count* is; a call whose true output would exceed that
    plateau truncates, fails to parse, and fails closed (the safe
    direction — never a silent under-reserve that risks truncating a
    *valid* small answer).

    ONE formula, TWO consumers, so the runtime precondition and the
    compile-time cap door can never drift apart:

    * The runtime precondition
      (:func:`~paramem.cloud.anonymize_steps.ask_speaker_anchor`) calls this
      with ``len(values)`` — the call's actual candidate-surface count.
    * The compile-time cap door (:mod:`paramem.graph.document_chunker`,
      :mod:`paramem.server.session_buffer`, via
      :func:`anonymize_payload_cap_tokens`'s callers) calls this with
      :data:`ANONYMIZE_ANCHOR_MAX_CANDIDATES` itself — the worst case a
      compile-time literal can assume, since neither module knows a real
      payload's candidate count in advance.

    Args:
        candidate_count: Distinct candidate-surface count for one ANCHOR
            call. Values ``< 0`` are treated as ``0``.

    Returns:
        The output-token reserve for one ANCHOR call, never evidence-text
        scaled.
    """
    bounded = max(0, min(candidate_count, ANONYMIZE_ANCHOR_MAX_CANDIDATES))
    return max(
        ANONYMIZE_MIN_ANCHOR_OUTPUT_TOKENS,
        ANONYMIZE_ANCHOR_OUTPUT_SKELETON_TOKENS + ANONYMIZE_ANCHOR_ENTRY_OVERHEAD_TOKENS * bounded,
    )


def words_to_estimator_tokens(
    words: int,
    *,
    tokens_per_word: float = MEASURED_TOKENS_PER_WORD,
) -> int:
    """THE word-cap -> estimator-unit encoding (the ratio-cancellation form).

    A cap derived in real tokens is re-expressed in words, then re-encoded
    in the estimator's own unit via this function, so every runtime
    boundary check (``estimate_tokens(text) <= cap``) reduces to
    ``words <= cap_words`` regardless of *tokens_per_word* — the ratio
    appears on both sides of the comparison and cancels. Storing a cap in
    real tokens instead loses that cancellation and makes the cap silently
    move whenever the ratio is re-measured.

    Args:
        words: Word count to encode. Values ``<= 0`` return ``0``.
        tokens_per_word: The ratio to encode with. Defaults to
            :data:`MEASURED_TOKENS_PER_WORD` — the same ratio every
            production caller's ``estimate_tokens()`` compares against.

    Returns:
        ``math.floor(words * tokens_per_word)``; ``0`` for ``words <= 0``.
    """
    if words <= 0:
        return 0
    return math.floor(words * tokens_per_word)


def anonymize_payload_cap_tokens(
    *,
    envelope_tokens: int,
    anchor_skeleton_tokens: int,
    anchor_reserve_tokens: int,
    payload_tokens_per_word: float,
    tokens_per_word: float = MEASURED_TOKENS_PER_WORD,
) -> int:
    """THE one door every anonymize-payload cap consumer calls — the payload
    size ceiling that fits the one remaining local ``generate()`` call
    shape (ANCHOR).

    A payload of ``P`` real tokens is carried once by the ANCHOR call,
    alongside its fixed prompt skeleton and its candidate-bounded output
    reserve, against a single envelope::

        envelope  >=  skeleton + P + reserve
        P         <=  envelope - skeleton - reserve
        cap_words  =  floor(P / payload_tokens_per_word)

    The result is re-expressed in the estimator's own unit via
    :func:`words_to_estimator_tokens` — see that function's docstring for
    why (ratio cancellation at every runtime comparison). Unlike the
    retired two-shape (SCAN, APPLY) derivation this superseded, there is
    only one call shape left to evaluate, so the identity is inlined here
    directly rather than split into a per-shape helper with a single caller.

    Args:
        envelope_tokens: Total (prompt + output) token budget for one
            anonymize call — the operator ceiling
            (:data:`ANONYMIZE_ENVELOPE_TOKENS`), not the live VRAM-clamped
            effective envelope (see the callers' own docstrings for why a
            compile-time cap is sized against the CONFIGURED ceiling).
        anchor_skeleton_tokens: The ANCHOR call's fixed prompt-template
            token cost (see :data:`ANONYMIZE_ANCHOR_PROMPT_SKELETON_TOKENS`).
        anchor_reserve_tokens: The ANCHOR call's output-side reserve —
            callers pass
            ``anchor_output_reserve_tokens(ANONYMIZE_ANCHOR_MAX_CANDIDATES)``,
            the same formula the runtime budget precondition uses,
            evaluated at its own structural candidate-count ceiling (a
            compile-time cap cannot know a real payload's candidate count
            in advance).
        payload_tokens_per_word: The payload shape's own prose ratio (real
            tokens per word), used only to convert the real-token payload
            budget into a word count.
        tokens_per_word: Ratio used to re-encode the word cap into the
            estimator's unit. Defaults to :data:`MEASURED_TOKENS_PER_WORD`.

    Returns:
        The cap in the estimator's unit (``words_to_estimator_tokens``
        output), never real tokens.

    Raises:
        ValueError: When ``envelope_tokens - anchor_skeleton_tokens -
            anchor_reserve_tokens <= 0`` — a mis-measured skeleton or a
            shrunken envelope leaves no payload budget at all, which is a
            configuration error, not a legitimate zero-word cap.
    """
    available = envelope_tokens - anchor_skeleton_tokens - anchor_reserve_tokens
    if available <= 0:
        raise ValueError(
            f"anonymize_payload_cap_tokens: no payload budget left — "
            f"envelope_tokens ({envelope_tokens!r}) - anchor_skeleton_tokens "
            f"({anchor_skeleton_tokens!r}) - anchor_reserve_tokens "
            f"({anchor_reserve_tokens!r}) = {available!r}, must be > 0"
        )
    cap_words = math.floor(available / payload_tokens_per_word)
    return words_to_estimator_tokens(cap_words, tokens_per_word=tokens_per_word)


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
