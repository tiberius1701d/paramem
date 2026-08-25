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

from paramem.utils.tokens import (
    _DRIFT_SAMPLE_DOCUMENT,
    _DRIFT_SAMPLES,
    MEASURED_TOKENS_PER_WORD,
    RenderedPrompt,
    check_ratio_drift,
    encode_rendered,
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


class _RecordingTokenizer:
    """Records the exact args/kwargs of its last call and returns a fixed dict."""

    def __init__(self, *, raises: bool = False):
        self.last_args: tuple | None = None
        self.last_kwargs: dict | None = None
        self._raises = raises

    def __call__(self, text_or_list, **kwargs):
        self.last_args = (text_or_list,)
        self.last_kwargs = kwargs
        if self._raises:
            raise RuntimeError("tokenizer exploded")
        return {"input_ids": [1, 2, 3]}


class TestRenderedPrompt:
    """RenderedPrompt is a plain str subclass — no behavior of its own."""

    def test_is_a_str_subclass(self):
        rp = RenderedPrompt("hello")
        assert isinstance(rp, str)
        assert isinstance(rp, RenderedPrompt)

    def test_behaves_like_the_underlying_string(self):
        rp = RenderedPrompt("hello world")
        assert rp == "hello world"
        assert rp.upper() == "HELLO WORLD"
        assert len(rp) == len("hello world")

    def test_plain_str_is_not_a_rendered_prompt(self):
        assert not isinstance("hello", RenderedPrompt)


class TestEncodeRendered:
    """encode_rendered: add_special_tokens=False always; type-gated on
    RenderedPrompt; never swallows a tokenizer exception."""

    def test_encodes_single_rendered_prompt_with_add_special_tokens_false(self):
        tok = _RecordingTokenizer()
        rp = RenderedPrompt("<s>[INST] hi [/INST]")
        result = encode_rendered(tok, rp, return_tensors="pt")
        assert result == {"input_ids": [1, 2, 3]}
        assert tok.last_args == (rp,)
        assert tok.last_kwargs == {"add_special_tokens": False, "return_tensors": "pt"}

    def test_encodes_list_of_rendered_prompts(self):
        tok = _RecordingTokenizer()
        prompts = [RenderedPrompt("a"), RenderedPrompt("b")]
        encode_rendered(tok, prompts, padding=True)
        assert tok.last_args == (prompts,)
        assert tok.last_kwargs == {"add_special_tokens": False, "padding": True}

    def test_plain_str_raises_type_error_naming_render_chat_prompt(self):
        tok = _RecordingTokenizer()
        with pytest.raises(TypeError, match="render_chat_prompt"):
            encode_rendered(tok, "plain string, not rendered")

    def test_list_containing_a_plain_str_raises_type_error(self):
        tok = _RecordingTokenizer()
        with pytest.raises(TypeError, match="render_chat_prompt"):
            encode_rendered(tok, [RenderedPrompt("ok"), "not rendered"])

    def test_empty_list_of_rendered_prompts_is_accepted(self):
        """No element fails the per-item check on an empty list — the
        tokenizer call itself decides what to do with an empty batch."""
        tok = _RecordingTokenizer()
        encode_rendered(tok, [])
        assert tok.last_args == ([],)

    def test_add_special_tokens_cannot_be_overridden_via_kwargs(self):
        """add_special_tokens is fixed by encode_rendered itself — a caller
        passing it explicitly collides as a duplicate keyword argument
        rather than silently overriding the False."""
        tok = _RecordingTokenizer()
        with pytest.raises(TypeError):
            encode_rendered(tok, RenderedPrompt("x"), add_special_tokens=True)

    def test_tokenizer_exception_propagates_unchanged(self):
        """Never swallowed — unlike estimate_tokens's fallback contract."""
        tok = _RecordingTokenizer(raises=True)
        with pytest.raises(RuntimeError, match="tokenizer exploded"):
            encode_rendered(tok, RenderedPrompt("x"))


class _AddSpecialTokensAwareStubTokenizer:
    """Stub tokenizer whose id COUNT depends on ``add_special_tokens`` —
    unlike :class:`_StubTokenizer` above (fixed count regardless of the
    flag), this one adds one synthetic id when ``add_special_tokens=True``.
    That makes it possible to prove a count/encode identity actually depends
    on both call sites passing the SAME value for the flag, rather than
    holding by coincidence of a fixture that ignores the flag entirely.
    """

    def __call__(self, text: str, add_special_tokens: bool = True, **kwargs):
        words = text.split()
        return {"input_ids": list(range(len(words) + (1 if add_special_tokens else 0)))}


class TestEncodeCountIdentity:
    """``len(encode_rendered(tok, p)["input_ids"]) == estimate_tokens(p, tok)``
    for a rendered prompt.

    This identity became EXACT once both :func:`encode_rendered` and
    :func:`estimate_tokens`'s exact path settled on ``add_special_tokens=False``
    — see ``paramem/utils/tokens.py``'s module docstring, "the encode
    chokepoint". Every anonymizer step's ``max_new_tokens = token_envelope -
    prompt_tokens`` arithmetic (:func:`paramem.cloud.anonymize_steps._generate`)
    depends on this identity: if the two calls ever diverged on
    ``add_special_tokens``, the envelope budget would be computed against a
    different token count than the one actually consumed when the rendered
    prompt is later tensorized for ``generate()``.
    """

    def test_count_matches_encode_for_rendered_prompt(self):
        tok = _AddSpecialTokensAwareStubTokenizer()
        prompt = RenderedPrompt("<s>[INST] hello there [/INST]")
        encoded_len = len(encode_rendered(tok, prompt)["input_ids"])
        estimated = estimate_tokens(prompt, tok)
        assert encoded_len == estimated

    def test_identity_would_break_if_either_side_used_add_special_tokens_true(self):
        """Negative control: proves the stub is actually sensitive to the
        flag (so the positive test above is not vacuous)."""
        tok = _AddSpecialTokensAwareStubTokenizer()
        prompt = RenderedPrompt("<s>[INST] hello there [/INST]")
        encoded_len = len(encode_rendered(tok, prompt)["input_ids"])
        with_special = len(tok(prompt, add_special_tokens=True)["input_ids"])
        assert with_special != encoded_len
