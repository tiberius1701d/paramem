"""Unit tests for paramem.utils.tokens — the shared token-estimation
primitive.

Covers:
- Exact path: a tokenizer supplied returns its precise id count.
- Fallback path: no tokenizer, or a raising one, returns a conservative
  words-based estimate — never a ``-1``/``0`` sentinel for non-empty text.
- The bounding claim: the shipped MAX-of-three-shapes ratio bounds every
  payload shape rather than averaging across them.
- check_ratio_drift: the boot-time re-measurement helper.
"""

from __future__ import annotations

import math

import pytest

from paramem.graph.document_chunker import (
    _ANON_SKELETON_TOKENS,
    _DOC_MAX_TOKENS,
    _F_DOC,
    _R_PROSE,
)
from paramem.utils.tokens import (
    _DRIFT_SAMPLE_DOCUMENT,
    _DRIFT_SAMPLES,
    ANONYMIZE_ENVELOPE_TOKENS,
    ANONYMIZE_OUTPUT_RESERVE_TOKENS,
    CONVERSATION_FACTS_RATIO,
    MEASURED_TOKENS_PER_WORD,
    SESSION_ANON_SKELETON_TOKENS,
    TRANSCRIPT_TOKENS_PER_WORD,
    check_ratio_drift,
    envelope_derived_cap_tokens,
    estimate_tokens,
    words_to_estimator_tokens,
)


class _StubTokenizer:
    """Deterministic stand-in for a HuggingFace tokenizer.

    Returns a fixed-length ``input_ids`` list on every call, or raises
    ``RuntimeError`` when constructed with ``raises=True`` — mirrors a
    MagicMock test fixture / half-initialised fast tokenizer.
    """

    def __init__(self, n_ids: int = 0, *, raises: bool = False):
        self._n_ids = n_ids
        self._raises = raises

    def __call__(self, text: str, add_special_tokens: bool = False):
        if self._raises:
            raise RuntimeError("stub tokenizer failure")
        return {"input_ids": list(range(self._n_ids))}


class TestEstimateTokensExactPath:
    """A stub tokenizer returning k ids -> estimate_tokens returns k."""

    def test_stub_tokenizer_returns_exact_count(self):
        tok = _StubTokenizer(n_ids=17)
        assert estimate_tokens("irrelevant text", tok) == 17

    def test_exact_path_ignores_tokens_per_word(self):
        """The exact path never consults the fallback ratio."""
        tok = _StubTokenizer(n_ids=5)
        assert estimate_tokens("one two three", tok, tokens_per_word=100.0) == 5

    def test_exact_zero_ids_is_a_legitimate_exact_count(self):
        """A tokenizer reporting 0 ids is a real (not fallback) measurement."""
        tok = _StubTokenizer(n_ids=0)
        assert estimate_tokens("some text", tok) == 0


class TestEstimateTokensFallbackPath:
    """No tokenizer -> ceil(words * ratio); 0 for empty; >=1 for non-empty."""

    def test_empty_text_returns_zero(self):
        assert estimate_tokens("") == 0

    def test_whitespace_only_text_returns_zero(self):
        assert estimate_tokens("   \n\t  ") == 0

    def test_nonempty_text_at_least_one(self):
        assert estimate_tokens("a") >= 1

    def test_fallback_uses_measured_ratio(self):
        text = "one two three four five"
        expected = math.ceil(5 * MEASURED_TOKENS_PER_WORD)
        assert estimate_tokens(text) == expected

    def test_explicit_ratio_overrides_constant(self):
        text = "one two three four"
        assert estimate_tokens(text, tokens_per_word=2.0) == math.ceil(4 * 2.0)

    def test_no_tokenizer_argument_selects_fallback(self):
        """tokenizer=None (the default) never attempts to call anything."""
        assert estimate_tokens("word word word") == math.ceil(3 * MEASURED_TOKENS_PER_WORD)


class TestEstimateTokensRaisingTokenizer:
    """A raising tokenizer falls back, never returns -1 or 0 for non-empty
    text: the estimator itself guarantees a usable cost, because a -1 or 0
    would make every payload "fit" a downstream budget check."""

    def test_raising_tokenizer_falls_back_to_estimate(self):
        tok = _StubTokenizer(raises=True)
        text = "one two three four five six"
        result = estimate_tokens(text, tok)
        assert result == math.ceil(6 * MEASURED_TOKENS_PER_WORD)

    def test_raising_tokenizer_never_returns_negative_one(self):
        tok = _StubTokenizer(raises=True)
        assert estimate_tokens("some non-empty text here", tok) != -1

    def test_raising_tokenizer_never_returns_zero_for_nonempty(self):
        tok = _StubTokenizer(raises=True)
        assert estimate_tokens("word", tok) >= 1

    def test_raising_tokenizer_empty_text_still_zero(self):
        """Fallback semantics apply fully on the raise path: empty -> 0."""
        tok = _StubTokenizer(raises=True)
        assert estimate_tokens("", tok) == 0


class TestEstimateTokensBoundingClaim:
    """The fallback (MAX-of-three-shapes ratio) must bound the exact count
    for every shape the system ingests, not just prose.

    Per-shape ratios below are the shipped measurement recorded on
    MEASURED_TOKENS_PER_WORD (production Mistral tokenizer over transcript /
    document / fact-JSON payloads) — this test fails if a future edit "tunes"
    MEASURED_TOKENS_PER_WORD down to a prose average that no longer bounds
    the fact-JSON shape.
    """

    _PER_SHAPE_RATIOS = {
        "transcript": 1.54,
        "document": 1.91,
        "fact_json": 3.657,
    }

    @pytest.mark.parametrize("shape_ratio", list(_PER_SHAPE_RATIOS.values()))
    def test_fallback_bounds_each_shape(self, shape_ratio):
        words = 40
        text = "word " * words
        # A stub tokenizer calibrated to this shape's measured per-word
        # ratio simulates the EXACT count a live tokenizer would report.
        exact_count = math.ceil(words * shape_ratio)
        tok = _StubTokenizer(n_ids=exact_count)
        exact = estimate_tokens(text, tok)
        fallback = estimate_tokens(text)  # no tokenizer -> MAX-ratio fallback
        assert fallback >= exact

    def test_measured_constant_is_the_max_of_the_shapes(self):
        assert MEASURED_TOKENS_PER_WORD >= max(self._PER_SHAPE_RATIOS.values())


class TestCheckRatioDrift:
    """None when observed max <= configured; observed max otherwise."""

    def test_returns_none_when_observed_at_or_below_configured(self):
        tok = _StubTokenizer(n_ids=1)
        assert check_ratio_drift(tok, configured_ratio=100.0) is None

    def test_returns_observed_max_when_it_exceeds_configured(self):
        tok = _StubTokenizer(n_ids=10_000)
        result = check_ratio_drift(tok, configured_ratio=0.01)
        assert result is not None
        assert result > 0.01

    def test_returns_none_at_exact_boundary(self):
        """Observed == configured is not drift (the unsafe direction is
        strictly exceeding, not merely reaching, configured_ratio)."""
        tok = _StubTokenizer(n_ids=10)
        # Both synthetic samples are non-empty, so the boundary case is
        # driven by whichever sample yields the larger observed ratio.
        observed = check_ratio_drift(tok, configured_ratio=0.0)
        assert observed is not None  # 10/words > 0.0 for any non-empty sample
        exact_ratio = observed
        assert check_ratio_drift(tok, configured_ratio=exact_ratio) is None

    def test_drift_samples_cover_all_three_payload_shapes(self):
        """Missing test 5 (review): _DRIFT_SAMPLES previously covered only
        transcript + fact-JSON, missing document PROSE — the one shape
        r_prose (which paramem.graph.document_chunker's _DOC_MAX_TOKENS
        depends on) governs.  Regression guard for that two-sample gap:
        pins that a document-shape sample is present and distinct from
        the transcript sample (turn-structured dialogue vs. continuous
        narrative prose).
        """
        assert len(_DRIFT_SAMPLES) == 3
        assert _DRIFT_SAMPLE_DOCUMENT in _DRIFT_SAMPLES

    def test_document_shape_drift_is_detected(self):
        """The document-prose sample actually participates in
        check_ratio_drift's MAX — not merely present but inert.  A
        tokenizer whose ratio spikes ONLY on the document sample (the
        transcript and fact-JSON samples stay near 1.0 tokens/word,
        comfortably below configured_ratio) is still caught.
        """

        class _DocumentSpikeTokenizer:
            def __call__(self, text: str, add_special_tokens: bool = False):
                if text == _DRIFT_SAMPLE_DOCUMENT:
                    n_ids = round(len(text.split()) * 50.0)
                else:
                    n_ids = len(text.split())  # ~1.0 tokens/word — below configured
                return {"input_ids": list(range(n_ids))}

        result = check_ratio_drift(_DocumentSpikeTokenizer(), configured_ratio=10.0)
        assert result is not None
        assert result > 10.0


class TestWordsToEstimatorTokens:
    """floor(words * ratio); 0 for non-positive words."""

    @pytest.mark.parametrize(
        ("words", "expected"),
        [
            (200, 740),  # document context floor
            (828, 3063),  # document-path cap word count
            (431, 1594),  # conversation-path cap word count
        ],
    )
    def test_floors_at_shipped_ratio(self, words, expected):
        assert words_to_estimator_tokens(words) == expected

    def test_ceil_would_disagree_with_floor_on_828(self):
        """The reason the convention is floor, not ceil — pins the exact
        boundary the shipped cap numbers depend on."""
        import math

        assert math.ceil(828 * MEASURED_TOKENS_PER_WORD) == 3064
        assert words_to_estimator_tokens(828) == 3063

    @pytest.mark.parametrize("words", [0, -1, -100])
    def test_non_positive_words_returns_zero(self, words):
        assert words_to_estimator_tokens(words) == 0

    def test_explicit_ratio_overrides_default(self):
        assert words_to_estimator_tokens(10, tokens_per_word=2.0) == 20


class TestEnvelopeDerivedCapTokens:
    """The shared budget-derivation function, at both shipped call sites.

    Document-path terms are IMPORTED from document_chunker.py (which this
    same change promoted them into) rather than re-declared locally — a
    local copy would itself be a duplicate of the values these tests
    exist to catch drift against.
    """

    def test_document_terms_match_doc_max_tokens(self):
        cap = envelope_derived_cap_tokens(
            envelope_tokens=ANONYMIZE_ENVELOPE_TOKENS,
            skeleton_tokens=_ANON_SKELETON_TOKENS,
            reserve_tokens=ANONYMIZE_OUTPUT_RESERVE_TOKENS,
            facts_ratio=_F_DOC,
            payload_tokens_per_word=_R_PROSE,
        )
        assert cap == _DOC_MAX_TOKENS
        assert round(cap / MEASURED_TOKENS_PER_WORD) == 828

    def test_conversation_terms_match_transcript_max_tokens(self):
        from paramem.server.session_buffer import _TRANSCRIPT_MAX_TOKENS

        cap = envelope_derived_cap_tokens(
            envelope_tokens=ANONYMIZE_ENVELOPE_TOKENS,
            skeleton_tokens=SESSION_ANON_SKELETON_TOKENS,
            reserve_tokens=ANONYMIZE_OUTPUT_RESERVE_TOKENS,
            facts_ratio=CONVERSATION_FACTS_RATIO,
            payload_tokens_per_word=TRANSCRIPT_TOKENS_PER_WORD,
        )
        assert cap == _TRANSCRIPT_MAX_TOKENS
        assert round(cap / MEASURED_TOKENS_PER_WORD) == 431

        skeleton_and_reserve = SESSION_ANON_SKELETON_TOKENS + ANONYMIZE_OUTPUT_RESERVE_TOKENS
        real_token_payload = (ANONYMIZE_ENVELOPE_TOKENS - skeleton_and_reserve) / (
            2 + CONVERSATION_FACTS_RATIO
        )
        assert round(real_token_payload) == 665

    @pytest.mark.parametrize(
        ("shape", "facts_ratio", "payload_ratio"),
        [
            ("document", _F_DOC, _R_PROSE),
            ("conversation", CONVERSATION_FACTS_RATIO, TRANSCRIPT_TOKENS_PER_WORD),
        ],
    )
    def test_budget_identity_holds(self, shape, facts_ratio, payload_ratio):
        """skeleton + f*P + 2P + reserve <= envelope, for the real-token
        payload P the cap derives from."""
        skeleton = _ANON_SKELETON_TOKENS if shape == "document" else SESSION_ANON_SKELETON_TOKENS
        available = ANONYMIZE_ENVELOPE_TOKENS - skeleton - ANONYMIZE_OUTPUT_RESERVE_TOKENS
        payload_real_tokens = available / (2 + facts_ratio)
        total = (
            skeleton
            + facts_ratio * payload_real_tokens
            + 2 * payload_real_tokens
            + ANONYMIZE_OUTPUT_RESERVE_TOKENS
        )
        assert total <= ANONYMIZE_ENVELOPE_TOKENS + 1e-6

    def test_raises_when_no_payload_budget_left(self):
        with pytest.raises(ValueError, match="no payload budget left"):
            envelope_derived_cap_tokens(
                envelope_tokens=100,
                skeleton_tokens=90,
                reserve_tokens=20,
                facts_ratio=1.0,
                payload_tokens_per_word=1.5,
            )

    def test_raises_at_exact_zero_boundary(self):
        with pytest.raises(ValueError, match="no payload budget left"):
            envelope_derived_cap_tokens(
                envelope_tokens=100,
                skeleton_tokens=80,
                reserve_tokens=20,
                facts_ratio=1.0,
                payload_tokens_per_word=1.5,
            )


class TestAnonymizeEnvelopeMirrors:
    """Cross-module honesty checks: the checked mirrors equal the live
    symbols they mirror, not a copied number that can drift."""

    def test_envelope_equals_anonymize_default(self):
        from paramem.cloud.anonymize import _DEFAULT_ANONYMIZER_TOKEN_ENVELOPE

        assert ANONYMIZE_ENVELOPE_TOKENS == _DEFAULT_ANONYMIZER_TOKEN_ENVELOPE

    def test_reserve_equals_anonymize_output_constants_sum(self):
        from paramem.cloud.anonymize import (
            _MAPPING_ENTRY_OVERHEAD_TOKENS,
            _OUTPUT_JSON_ENVELOPE_TOKENS,
        )

        assert ANONYMIZE_OUTPUT_RESERVE_TOKENS == (
            _OUTPUT_JSON_ENVELOPE_TOKENS + _MAPPING_ENTRY_OVERHEAD_TOKENS
        )
