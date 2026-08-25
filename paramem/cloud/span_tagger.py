"""The span tagger — a CPU-resident GLiNER model as the anonymizer's scan step.

Loads a GLiNER checkpoint the way the base model is loaded: eager PyTorch,
``GLiNER.from_pretrained`` against a Hub checkpoint id resolved through the
standard HF cache, no build step, no ONNX export. This is the ONE module in
the codebase that imports ``gliner``, and it does so lazily inside
:func:`load_at_startup` so importing this module costs nothing until the
model is actually loaded.

The model truncates its input at its own configured ``max_len`` splitter
tokens with a warning rather than an exception, so a payload longer than
that is tiled into overlapping windows sized in the model's own units —
never a whitespace-word estimate. A window is bounded in TWO units, both
read off the loaded model rather than typed as constants: splitter tokens
(``model.config.max_len``) and encoder subwords, bounded by the encoder's
own trained length (``max_position_embeddings``) rather than by the
transformer tokenizer's ``model_max_length``, which this checkpoint's
tokenizer leaves unset and therefore never truncates on its own. Each
window is one model call; window-local spans are remapped to caller
coordinates at one named site and cross-window duplicates on
``(start, end, label)`` collapse, keeping the higher score.

Fails CLOSED at load, where ``paramem.server.lang_id`` fails open: a
missing language detector costs a wrong reply language, a missing PII
detector costs verbatim egress of personal data. A load failure raises,
naming the checkpoint and revision that could not be resolved. A model-call
failure inside :func:`tag` is converted to :class:`TaggerUnavailable` so
every caller reads a structured signal rather than an uncaught exception.

Holds a CPU-resident GLiNER model only (``map_location="cpu"``); it is not
a base-model holder — it never touches ``_state["model"]``, is invisible to
``_release_base_model_in_process``, and is left resident by both
``POST /gpu/release`` and ``POST /gpu/acquire``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    from paramem.server.config import SpanTaggerConfig

logger = logging.getLogger(__name__)


class TaggerUnavailable(Exception):
    """The span tagger cannot answer: no handle loaded, or the model call raised.

    Raised by :func:`tag` when ``load_at_startup`` has not produced a live
    handle, and by the per-window boundary inside :func:`tag` for ANY
    exception ``GLiNER.predict_entities`` raises — the original message is
    carried and the original exception is chained as the cause. This is
    boundary error handling on a third-party model call, not suppression:
    the failure is re-expressed as the one structured signal callers can
    branch on, in the manner of ``VramExhausted``.
    """


@dataclass(frozen=True)
class TaggedSpan:
    """One tagged span in caller (payload) character coordinates.

    Carries no gliner type — nothing above this module sees one.

    Attributes
    ----------
    start, end:
        Character offsets into the text handed to :func:`tag`, after the
        window remap.
    text:
        Invariant, asserted at remap time: ``payload[start:end] == text``.
    label:
        Verbatim, one of the ``labels`` passed to :func:`tag`.
    score:
        The model's confidence for this span, in ``[0, 1]``.
    """

    start: int
    end: int
    text: str
    label: str
    score: float


@dataclass(frozen=True)
class TagResult:
    """The full result of one :func:`tag` call.

    Attributes
    ----------
    spans:
        Deduplicated spans in payload coordinates, ordered by
        ``(start, end)``.
    windows:
        Number of model calls issued — the chain's ``tagger_windows``.
    """

    spans: tuple[TaggedSpan, ...]
    windows: int


@dataclass
class _TaggerHandle:
    """The loaded model plus everything read off it at load time."""

    model: Any  # gliner.GLiNER
    checkpoint: str
    revision: str
    score_threshold: float
    max_len: int
    max_width: int
    max_position_embeddings: int
    threads: int


_singleton: _TaggerHandle | None = None
_load_lock = Lock()
_call_lock = Lock()


def load_at_startup(cfg: "SpanTaggerConfig") -> None:
    """Load the GLiNER checkpoint named on *cfg*, once per process.

    Idempotent on the MODEL when the resolved ``(checkpoint, revision)``
    pair is unchanged from the currently loaded handle — no reload, no
    second ``GLiNER.from_pretrained`` call — but LIVE on two settings even
    on that unchanged path: ``score_threshold`` is refreshed onto the
    existing handle from *cfg*, and ``torch.set_num_threads(cfg.threads)``
    is re-applied whenever *cfg*'s thread count differs from the value
    last applied (tracked on the handle), so a config-apply that changes
    either setting takes effect without a model reload. Reloads the model
    itself only when ``checkpoint``/``revision`` changed.

    Reads ``max_len`` and ``max_width`` off the loaded model's own config
    and the encoder's trained length off its own config
    (``model.model.token_rep_layer.bert_layer.model.config.max_position_embeddings``),
    and asserts ``max_width <= max_len // 2`` — the overlap-containment
    precondition a window generator needs to guarantee forward progress.
    This is a checkpoint-swap guard: the pinned default checkpoint is
    ``12 <= 192``, so the assertion is never live at the shipped default.

    Parameters
    ----------
    cfg:
        ``SpanTaggerConfig`` — ``checkpoint``, ``revision``,
        ``score_threshold`` and ``threads`` all arrive here; the model's
        own ``max_len``/``max_width``/``max_position_embeddings`` are read
        from the loaded model, never typed in this module.

    Raises
    ------
    RuntimeError
        The checkpoint or revision could not be resolved (offline host,
        cold cache, bad revision). The message names both; the original
        exception is chained as the cause. This is a fail-fast refusal,
        not :class:`TaggerUnavailable` — that type is reserved for
        :func:`tag`'s own boundary.
    """
    global _singleton
    with _load_lock:
        if (
            _singleton is not None
            and _singleton.checkpoint == cfg.checkpoint
            and _singleton.revision == cfg.revision
        ):
            _singleton.score_threshold = cfg.score_threshold
            if _singleton.threads != cfg.threads:
                import torch

                torch.set_num_threads(cfg.threads)
                _singleton.threads = cfg.threads
            return

        import torch
        from gliner import GLiNER

        try:
            model = GLiNER.from_pretrained(
                cfg.checkpoint, revision=cfg.revision, map_location="cpu"
            )
        except Exception as exc:
            raise RuntimeError(
                f"span tagger: failed to load checkpoint={cfg.checkpoint!r} "
                f"revision={cfg.revision!r}: {exc}"
            ) from exc

        torch.set_num_threads(cfg.threads)

        try:
            max_len = model.config.max_len
            max_width = model.config.max_width
            max_position_embeddings = (
                model.model.token_rep_layer.bert_layer.model.config.max_position_embeddings
            )
        except AttributeError as exc:
            raise RuntimeError(
                f"span tagger: checkpoint={cfg.checkpoint!r} revision={cfg.revision!r} "
                "does not expose the expected token-rep-layer config shape "
                f"(max_len/max_width/max_position_embeddings): {exc}"
            ) from exc
        if max_width > max_len // 2:
            raise RuntimeError(
                f"span tagger: checkpoint={cfg.checkpoint!r} revision={cfg.revision!r} "
                f"has max_width={max_width} > max_len // 2={max_len // 2} — the "
                "overlap-containment precondition does not hold for this checkpoint"
            )

        _singleton = _TaggerHandle(
            model=model,
            checkpoint=cfg.checkpoint,
            revision=cfg.revision,
            score_threshold=cfg.score_threshold,
            max_len=max_len,
            max_width=max_width,
            max_position_embeddings=max_position_embeddings,
            threads=cfg.threads,
        )
        logger.info(
            "span_tagger: loaded checkpoint=%s revision=%s max_len=%d max_width=%d "
            "max_position_embeddings=%d threads=%d",
            cfg.checkpoint,
            cfg.revision,
            max_len,
            max_width,
            max_position_embeddings,
            cfg.threads,
        )


def _label_prefix_words(handle: _TaggerHandle, labels: list[str]) -> list[str]:
    """Return the label-union prompt word tokens for one ``tag()`` call.

    Built once per call via the processor's own ``prepare_inputs`` (never
    hand-built), on an empty text so the returned word list is exactly the
    ``[ENT] label1 [ENT] label2 ... [SEP]`` prefix every window's fit test
    is measured against.
    """
    processor = handle.model.data_processor
    input_texts, _prompt_lengths = processor.prepare_inputs([[]], labels)
    return input_texts[0]


def _subword_count(handle: _TaggerHandle, prefix_words: list[str], window_words: list[str]) -> int:
    """Exact encoder subword count for ``prefix_words + window_words``.

    Uses ``processor.transformer_tokenizer(..., is_split_into_words=True)``
    on the same input shape ``tokenize_inputs`` uses — one batch of one
    already word-split example — so the count is exact, not estimated.
    ``add_special_tokens=True`` is stated explicitly (matching gliner's own
    ``tokenize_inputs`` call, which passes no override and so takes the HF
    default of ``True``): this encoder's own input carries its own
    ``[CLS]``/``[SEP]`` special tokens, and they count toward the trained
    length (``max_position_embeddings``) this function's callers bound
    against — this is not chat-template-rendered text, so the
    double-BOS/lost-EOS concern ``encode_rendered`` guards against does not
    apply here.
    """
    tokenizer = handle.model.data_processor.transformer_tokenizer
    encoded = tokenizer(
        [prefix_words + window_words],
        is_split_into_words=True,
        truncation=True,
        add_special_tokens=True,
    )
    return len(encoded["input_ids"][0])


def _windows(text: str, handle: _TaggerHandle, labels: list[str]) -> list[tuple[str, int]]:
    """Tile *text* into windows sized in the model's own splitter tokens.

    Two derivation rules, both over values read from the loaded model:

    * ``window_tokens`` is the largest splitter-token count for which BOTH
      bounds hold: ``splitter_tokens(window) <= max_len`` and
      ``subwords(label_prefix + window) <= max_position_embeddings``. Each
      window opens at the splitter ceiling and shrinks (via binary search
      over the exact subword count) until the subword bound fits.
    * ``overlap_tokens = max_width`` — a returned span is at most
      ``max_width`` splitter tokens wide, so a span cut by a window
      boundary is wholly contained in the next window when the stride
      leaves that much overlap.

    Windows are cut at splitter token boundaries — a window is
    ``text[tokens[i].start : tokens[j].end]`` — so re-tokenizing the window
    text yields exactly the window's tokens; the count is exact.
    """
    processor = handle.model.data_processor
    tokens = list(processor.words_splitter(text))
    if not tokens:
        return []

    prefix_words = _label_prefix_words(handle, labels)

    def fits(end: int) -> bool:
        window_words = [tok for tok, _start, _end in tokens[i:end]]
        return _subword_count(handle, prefix_words, window_words) <= handle.max_position_embeddings

    result: list[tuple[str, int]] = []
    n = len(tokens)
    i = 0
    while i < n:
        splitter_ceiling = min(i + handle.max_len, n)
        lo = i + 1
        if not fits(lo):
            # Cannot shrink below one token; accept it as a best-effort
            # window rather than stalling the generator.
            window_end = lo
            overflow_subwords = _subword_count(handle, prefix_words, [tokens[i][0]])
            logger.warning(
                "span_tagger: one splitter token alone needs %d encoder subwords, "
                "exceeding the encoder's trained length (max_position_embeddings=%d) "
                "— this window runs the encoder past its trained length.",
                overflow_subwords,
                handle.max_position_embeddings,
            )
        else:
            window_end = lo
            hi = splitter_ceiling
            left, right = lo, hi
            while left <= right:
                mid = (left + right) // 2
                if fits(mid):
                    window_end = mid
                    left = mid + 1
                else:
                    right = mid - 1

        char_start = tokens[i][1]
        char_end = tokens[window_end - 1][2]
        result.append((text[char_start:char_end], char_start))

        if window_end >= n:
            break
        # Forward progress is guaranteed even when a window shrank below
        # `overlap_tokens` under subword pressure (dense text).
        i = max(i + 1, window_end - handle.max_width)

    return result


def tag(text: str, labels: Sequence[str]) -> TagResult:
    """Tag every span of *text* against *labels*.

    Tiles *text* into windows sized in the model's own splitter-token and
    encoder-subword units (see :func:`_windows`), issues one
    ``GLiNER.predict_entities(window, labels, flat_ner=True,
    multi_label=False, threshold=cfg.score_threshold)`` call per window
    under a per-call lock (so the invariant is "at most one tagger run at a
    time" without holding the lock for the whole payload), remaps each
    window-local span to *text* coordinates at one site, asserts the
    post-remap identity ``text[span.start:span.end] == span.text``, and
    deduplicates cross-window on ``(start, end, label)`` keeping the higher
    score.

    Parameters
    ----------
    text:
        The payload to tag, in the caller's own coordinate space — the
        offset space every returned :class:`TaggedSpan` is expressed in.
    labels:
        The label vocabulary for this call — the ordered union of every
        active category's ``tagger_labels``.

    Returns
    -------
    TagResult
        ``spans`` ordered by ``(start, end)``; ``windows`` is the number
        of model calls issued (0 for empty *text*).

    Raises
    ------
    TaggerUnavailable
        No handle is loaded (``load_at_startup`` was not called or
        failed), or ``GLiNER.predict_entities`` raised for any window —
        the original message is carried and the original exception is
        chained as the cause.
    """
    handle = _singleton
    if handle is None:
        raise TaggerUnavailable("span tagger: no handle loaded")

    label_list = list(labels)
    windows = _windows(text, handle, label_list)

    best: dict[tuple[int, int, str], TaggedSpan] = {}
    for window_text, char_offset in windows:
        with _call_lock:
            try:
                raw_spans = handle.model.predict_entities(
                    window_text,
                    label_list,
                    flat_ner=True,
                    multi_label=False,
                    threshold=handle.score_threshold,
                )
            except Exception as exc:
                raise TaggerUnavailable(f"span tagger: model call failed: {exc}") from exc

        for raw in raw_spans:
            start = raw["start"] + char_offset
            end = raw["end"] + char_offset
            span_text = raw["text"]
            assert text[start:end] == span_text, (
                f"span tagger remap invariant violated: text[{start}:{end}]="
                f"{text[start:end]!r} != {span_text!r}"
            )
            key = (start, end, raw["label"])
            score = float(raw["score"])
            existing = best.get(key)
            if existing is None or score > existing.score:
                best[key] = TaggedSpan(
                    start=start, end=end, text=span_text, label=raw["label"], score=score
                )

    spans = tuple(sorted(best.values(), key=lambda s: (s.start, s.end)))
    return TagResult(spans=spans, windows=len(windows))


def reset_for_tests() -> None:
    """Drop the loaded handle so a subsequent :func:`tag` raises ``TaggerUnavailable``."""
    global _singleton
    with _load_lock:
        _singleton = None
