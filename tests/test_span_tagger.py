"""``paramem.cloud.span_tagger`` — load, the ``tag()`` boundary, windowing,
offset remap, and cross-window dedup. Every test uses a stub GLiNER model;
none loads the real checkpoint.
"""

from __future__ import annotations

import pytest

from paramem.cloud import span_tagger
from paramem.cloud.span_tagger import TaggerUnavailable, _TaggerHandle, _windows
from paramem.server.config import SpanTaggerConfig
from tests.anonymizer_doubles import (
    StubGliner,
    install_raising_gliner_module,
    install_stub_gliner_module,
    span_tagger_reset,
    words_splitter,
)

__all__ = ["span_tagger_reset"]  # re-exported fixture


def _handle_for(
    model: StubGliner, *, score_threshold: float = 0.5, threads: int = 8
) -> _TaggerHandle:
    return _TaggerHandle(
        model=model,
        checkpoint="stub/checkpoint",
        revision="stub-revision",
        score_threshold=score_threshold,
        max_len=model.max_len,
        max_width=model.max_width,
        max_position_embeddings=model.max_position_embeddings,
        threads=threads,
    )


class _RecordingLock:
    """Records enter/exit counts without changing lock semantics for a
    single-threaded test.
    """

    def __init__(self):
        self.enter_count = 0
        self.exit_count = 0

    def __enter__(self):
        self.enter_count += 1
        return self

    def __exit__(self, *exc):
        self.exit_count += 1
        return False


# ---------------------------------------------------------------------------
# load_at_startup
# ---------------------------------------------------------------------------


class TestLoadAtStartup:
    def test_checkpoint_revision_and_map_location_pass_through_verbatim(
        self, monkeypatch, span_tagger_reset
    ) -> None:
        model = StubGliner()
        captured = install_stub_gliner_module(monkeypatch, model)
        cfg = SpanTaggerConfig(checkpoint="my/checkpoint", revision="deadbeef", threads=2)

        span_tagger.load_at_startup(cfg)

        assert captured["checkpoint"] == "my/checkpoint"
        assert captured["revision"] == "deadbeef"
        assert captured["map_location"] == "cpu"

    def test_a_raise_from_the_loader_propagates_naming_checkpoint_and_revision(
        self, monkeypatch, span_tagger_reset
    ) -> None:
        install_raising_gliner_module(monkeypatch, ValueError("cold cache"))
        cfg = SpanTaggerConfig(checkpoint="acme/ckpt", revision="rev-123")

        with pytest.raises(RuntimeError) as exc_info:
            span_tagger.load_at_startup(cfg)

        message = str(exc_info.value)
        assert "acme/ckpt" in message
        assert "rev-123" in message

    def test_set_num_threads_is_called_once_with_configured_threads(
        self, monkeypatch, span_tagger_reset
    ) -> None:
        import torch

        model = StubGliner()
        install_stub_gliner_module(monkeypatch, model)
        calls: list[int] = []
        monkeypatch.setattr(torch, "set_num_threads", lambda n: calls.append(n))
        cfg = SpanTaggerConfig(threads=3)

        span_tagger.load_at_startup(cfg)

        assert calls == [3]

    def test_second_call_with_the_same_pair_is_a_no_op(
        self, monkeypatch, span_tagger_reset
    ) -> None:
        model = StubGliner()
        captured = install_stub_gliner_module(monkeypatch, model)
        cfg = SpanTaggerConfig(checkpoint="ck", revision="r1")

        span_tagger.load_at_startup(cfg)
        span_tagger.load_at_startup(cfg)

        assert len(captured["calls"]) == 1

    def test_a_changed_revision_reloads(self, monkeypatch, span_tagger_reset) -> None:
        model = StubGliner()
        captured = install_stub_gliner_module(monkeypatch, model)

        span_tagger.load_at_startup(SpanTaggerConfig(checkpoint="ck", revision="r1"))
        span_tagger.load_at_startup(SpanTaggerConfig(checkpoint="ck", revision="r2"))

        assert len(captured["calls"]) == 2
        assert captured["revision"] == "r2"

    def test_reset_for_tests_clears_the_handle(self, monkeypatch, span_tagger_reset) -> None:
        model = StubGliner()
        install_stub_gliner_module(monkeypatch, model)
        span_tagger.load_at_startup(SpanTaggerConfig())

        span_tagger.reset_for_tests()

        with pytest.raises(TaggerUnavailable):
            span_tagger.tag("hello", ["person"])

    def test_max_len_and_max_width_are_read_off_the_model_never_a_literal(
        self, monkeypatch, span_tagger_reset
    ) -> None:
        model = StubGliner(max_len=17, max_width=3, max_position_embeddings=1000)
        install_stub_gliner_module(monkeypatch, model)

        span_tagger.load_at_startup(SpanTaggerConfig())

        # Proven behaviourally: a payload longer than 17 splitter tokens
        # must be windowed (more than one call), and a payload of exactly
        # 17 or fewer stays in one window — the module never hardcodes a
        # window size, so this is only true if max_len came off the stub.
        short_text = " ".join(f"w{i}" for i in range(17))
        long_text = " ".join(f"w{i}" for i in range(18))

        short_result = span_tagger.tag(short_text, ["person"])
        long_result = span_tagger.tag(long_text, ["person"])

        assert short_result.windows == 1
        assert long_result.windows >= 2

    def test_max_width_greater_than_half_max_len_fails_the_assertion(
        self, monkeypatch, span_tagger_reset
    ) -> None:
        model = StubGliner(max_len=10, max_width=6)  # 6 > 10 // 2 == 5
        install_stub_gliner_module(monkeypatch, model)

        with pytest.raises(RuntimeError, match="max_width"):
            span_tagger.load_at_startup(SpanTaggerConfig())


# ---------------------------------------------------------------------------
# tag() boundary
# ---------------------------------------------------------------------------


class TestTagBoundary:
    @pytest.mark.parametrize(
        "raised", [RuntimeError("cuda gone"), ValueError("bad input"), Exception("generic")]
    )
    def test_predict_entities_raising_becomes_tagger_unavailable_with_cause(
        self, monkeypatch, span_tagger_reset, raised
    ) -> None:
        model = StubGliner(raise_with=raised)
        monkeypatch.setattr(span_tagger, "_singleton", _handle_for(model))

        with pytest.raises(TaggerUnavailable) as exc_info:
            span_tagger.tag("hello world", ["person"])

        assert str(raised) in str(exc_info.value)
        assert exc_info.value.__cause__ is raised

    def test_no_handle_loaded_raises_tagger_unavailable(
        self, monkeypatch, span_tagger_reset
    ) -> None:
        monkeypatch.setattr(span_tagger, "_singleton", None)

        with pytest.raises(TaggerUnavailable):
            span_tagger.tag("hello world", ["person"])


# ---------------------------------------------------------------------------
# Windowing — structural properties of _windows() over a stub handle.
# ---------------------------------------------------------------------------


def _tokens_for_text(text: str):
    return list(words_splitter(text))


def _assert_boundaries_fall_between_tokens(text: str, windows: list[tuple[str, int]]) -> None:
    full_tokens = _tokens_for_text(text)
    starts = {t[1] for t in full_tokens}
    ends = {t[2] for t in full_tokens}
    for window_text, char_offset in windows:
        assert char_offset in starts, f"window start {char_offset} is not a token boundary"
        window_end = char_offset + len(window_text)
        assert window_end in ends, f"window end {window_end} is not a token boundary"


class TestWindowGeneration:
    def _handle(self, *, max_len=5, max_width=2, max_position_embeddings=1000):
        model = StubGliner(
            max_len=max_len, max_width=max_width, max_position_embeddings=max_position_embeddings
        )
        return model, _handle_for(model)

    def test_a_payload_under_both_bounds_yields_exactly_one_window(self) -> None:
        _model, handle = self._handle()
        text = "t0 t1 t2"  # 3 tokens, well under max_len=5
        windows = _windows(text, handle, ["person"])
        assert len(windows) == 1
        assert windows[0] == (text, 0)

    def test_three_window_payload_has_the_expected_count_and_stride(self) -> None:
        _model, handle = self._handle(max_len=5, max_width=2)
        text = " ".join(f"t{i}" for i in range(10))  # 10 tokens

        windows = _windows(text, handle, ["person"])

        assert len(windows) == 3
        offsets = [offset for _text, offset in windows]
        # stride == max_len - max_width == 3
        assert offsets == [0, 9, 18]

    def test_no_window_exceeds_max_len_splitter_tokens(self) -> None:
        _model, handle = self._handle(max_len=5, max_width=2)
        text = " ".join(f"t{i}" for i in range(13))
        windows = _windows(text, handle, ["person"])
        for window_text, _offset in windows:
            assert len(_tokens_for_text(window_text)) <= handle.max_len

    def test_window_boundaries_fall_between_tokens(self) -> None:
        _model, handle = self._handle(max_len=5, max_width=2)
        text = " ".join(f"t{i}" for i in range(13))
        windows = _windows(text, handle, ["person"])
        _assert_boundaries_fall_between_tokens(text, windows)

    def test_windows_stay_within_max_position_embeddings_subwords(self) -> None:
        model, handle = self._handle(max_len=5, max_width=2, max_position_embeddings=1000)
        text = " ".join(f"t{i}" for i in range(13))
        labels = ["person"]
        windows = _windows(text, handle, labels)
        prefix_len = 2 * len(labels) + 1
        for window_text, _offset in windows:
            token_count = len(_tokens_for_text(window_text))
            subwords = round((prefix_len + token_count) * model.subwords_per_word)
            assert subwords <= handle.max_position_embeddings


class TestSubwordBoundShrinksWindows:
    def test_an_inflated_subword_tokenizer_forces_narrower_more_numerous_windows(self) -> None:
        text = " ".join(f"t{i}" for i in range(30))
        labels = ["person"]

        loose_model = StubGliner(
            max_len=50, max_width=1, max_position_embeddings=30, subwords_per_word=1.0
        )
        loose_handle = _handle_for(loose_model)
        loose_windows = _windows(text, loose_handle, labels)

        tight_model = StubGliner(
            max_len=50, max_width=1, max_position_embeddings=30, subwords_per_word=4.0
        )
        tight_handle = _handle_for(tight_model)
        tight_windows = _windows(text, tight_handle, labels)

        assert len(tight_windows) > len(loose_windows)
        assert len(_tokens_for_text(tight_windows[0][0])) < len(
            _tokens_for_text(loose_windows[0][0])
        )

        for handle, model, windows in (
            (loose_handle, loose_model, loose_windows),
            (tight_handle, tight_model, tight_windows),
        ):
            prefix_len = 2 * len(labels) + 1
            for window_text, _offset in windows:
                token_count = len(_tokens_for_text(window_text))
                assert token_count <= handle.max_len
                subwords = round((prefix_len + token_count) * model.subwords_per_word)
                assert subwords <= handle.max_position_embeddings

    def test_growing_the_label_union_narrows_every_window(self) -> None:
        text = " ".join(f"t{i}" for i in range(25))
        model = StubGliner(max_len=50, max_width=1, max_position_embeddings=15)
        handle = _handle_for(model)

        one_label_windows = _windows(text, handle, ["person"])
        two_label_windows = _windows(text, handle, ["person", "email"])

        assert len(_tokens_for_text(two_label_windows[0][0])) < len(
            _tokens_for_text(one_label_windows[0][0])
        )


# ---------------------------------------------------------------------------
# tag() — offsets, cross-window dedup, call/lock counting.
# ---------------------------------------------------------------------------


def _span_for(window_text: str, token: str, *, label: str, score: float) -> dict:
    start = window_text.index(token)
    return {
        "start": start,
        "end": start + len(token),
        "text": token,
        "label": label,
        "score": score,
    }


class TestTagOffsetsDedupAndCallAccounting:
    """One shared three-window scenario:

    text = "t0 t1 t2 t3 t4 t5 t6 t7 t8 t9" with max_len=5, max_width=2
    produces windows at token ranges [0:5), [3:8), [6:10) — window texts
    "t0 t1 t2 t3 t4", "t3 t4 t5 t6 t7", "t6 t7 t8 t9".

    t0  -> only window 0.
    t3  -> overlap of windows 0 and 1 (different scores -> dedup keeps max).
    "t3 t4" (max_width-wide) -> whole in both windows 0 and 1.
    t6  -> overlap of windows 1 and 2 (different scores -> dedup keeps max).
    t9  -> only window 2 (the third window).
    """

    def _spans_fn(self, window_text: str, _labels):
        if window_text == "t0 t1 t2 t3 t4":
            return [
                _span_for(window_text, "t0", label="person", score=0.9),
                _span_for(window_text, "t3", label="person", score=0.6),
                _span_for(window_text, "t3 t4", label="person", score=0.55),
            ]
        if window_text == "t3 t4 t5 t6 t7":
            return [
                _span_for(window_text, "t3", label="person", score=0.95),
                _span_for(window_text, "t3 t4", label="person", score=0.85),
                _span_for(window_text, "t6", label="person", score=0.7),
            ]
        if window_text == "t6 t7 t8 t9":
            return [
                _span_for(window_text, "t6", label="person", score=0.99),
                _span_for(window_text, "t9", label="person", score=0.5),
            ]
        raise AssertionError(f"unexpected window text: {window_text!r}")

    def _tag(self, *, lock=None):
        text = " ".join(f"t{i}" for i in range(10))
        model = StubGliner(
            max_len=5, max_width=2, max_position_embeddings=1000, spans_fn=self._spans_fn
        )
        handle = _handle_for(model)
        return text, model, handle

    def test_windows_equals_the_model_call_count(self, monkeypatch, span_tagger_reset) -> None:
        text, model, handle = self._tag()
        monkeypatch.setattr(span_tagger, "_singleton", handle)

        result = span_tagger.tag(text, ["person"])

        assert result.windows == 3
        assert len(model.calls) == 3

    def test_lock_is_taken_and_released_once_per_window(
        self, monkeypatch, span_tagger_reset
    ) -> None:
        text, _model, handle = self._tag()
        monkeypatch.setattr(span_tagger, "_singleton", handle)
        lock = _RecordingLock()
        monkeypatch.setattr(span_tagger, "_call_lock", lock)

        span_tagger.tag(text, ["person"])

        assert lock.enter_count == 3
        assert lock.exit_count == 3

    def test_every_returned_span_satisfies_the_remap_identity(
        self, monkeypatch, span_tagger_reset
    ) -> None:
        text, _model, handle = self._tag()
        monkeypatch.setattr(span_tagger, "_singleton", handle)

        result = span_tagger.tag(text, ["person"])

        assert result.spans, "expected at least one span"
        for span in result.spans:
            assert text[span.start : span.end] == span.text

    def test_a_span_found_only_in_the_third_window_has_correct_offsets(
        self, monkeypatch, span_tagger_reset
    ) -> None:
        text, _model, handle = self._tag()
        monkeypatch.setattr(span_tagger, "_singleton", handle)

        result = span_tagger.tag(text, ["person"])

        t9_spans = [s for s in result.spans if s.text == "t9"]
        assert len(t9_spans) == 1
        assert (t9_spans[0].start, t9_spans[0].end) == (27, 29)
        assert text[27:29] == "t9"

    def test_a_value_planted_in_the_overlap_region_appears_exactly_once(
        self, monkeypatch, span_tagger_reset
    ) -> None:
        text, _model, handle = self._tag()
        monkeypatch.setattr(span_tagger, "_singleton", handle)

        result = span_tagger.tag(text, ["person"])

        t3_spans = [s for s in result.spans if s.text == "t3"]
        assert len(t3_spans) == 1

    def test_two_windows_returning_the_same_span_with_different_scores_keep_the_higher(
        self, monkeypatch, span_tagger_reset
    ) -> None:
        text, _model, handle = self._tag()
        monkeypatch.setattr(span_tagger, "_singleton", handle)

        result = span_tagger.tag(text, ["person"])

        t3_spans = [s for s in result.spans if s.text == "t3"]
        t6_spans = [s for s in result.spans if s.text == "t6"]
        assert t3_spans[0].score == 0.95  # max(0.6, 0.95)
        assert t6_spans[0].score == 0.99  # max(0.7, 0.99)

    def test_a_max_width_wide_span_straddling_the_seam_is_returned_whole(
        self, monkeypatch, span_tagger_reset
    ) -> None:
        text, _model, handle = self._tag()
        monkeypatch.setattr(span_tagger, "_singleton", handle)

        result = span_tagger.tag(text, ["person"])

        wide_spans = [s for s in result.spans if s.text == "t3 t4"]
        assert len(wide_spans) == 1
        assert text[wide_spans[0].start : wide_spans[0].end] == "t3 t4"
