"""Shared test doubles for the GLiNER span-tagger scan step and the
anonymize chain around it.

Not a test module itself — imported by the span-tagger / anonymizer test
files so none of them re-implements the same stub GLiNER shape, the
same splitter regex, or the same ``ScrubCategory``/``AnonymizerPrompts``
construction. Every test file that installs a stub tagger handle is
responsible for calling :func:`reset_span_tagger` in its own teardown
(directly, or via the :func:`span_tagger_reset` fixture below).

Never imported by production code under ``paramem/``.
"""

from __future__ import annotations

import re
import types
from dataclasses import dataclass, field
from typing import Callable, Sequence

import pytest

from paramem.cloud import span_tagger
from paramem.cloud.anonymize import AnonymizerPrompts
from paramem.config.taxonomy import ScrubCategory

# ---------------------------------------------------------------------------
# The real GLiNER splitter regex (gliner/data_processing/tokenizer.py:49),
# reproduced here so the stub's word boundaries match production exactly:
# one token per run of word chars (with embedded ``-``/``_``), one token
# per other non-space character.
# ---------------------------------------------------------------------------
_SPLITTER_RE = re.compile(r"\w+(?:[-_]\w+)*|\S")


def words_splitter(text: str):
    """Yield ``(token, start, end)`` triples exactly like GLiNER's
    ``WhitespaceTokenSplitter`` does — the shape :func:`paramem.cloud.
    span_tagger._windows` iterates.
    """
    for m in _SPLITTER_RE.finditer(text):
        yield (m.group(), m.start(), m.end())


def basic_category(
    *,
    name: str = "Person",
    prefix: str = "Person",
    hints: tuple[str, ...] = ("person name",),
    tagger_labels: tuple[str, ...] = ("person",),
) -> ScrubCategory:
    """One ready-made :class:`ScrubCategory` for tests that don't care
    about the exact schema.yaml row, just need a well-shaped category.
    """
    return ScrubCategory(name=name, prefix=prefix, hints=hints, tagger_labels=tagger_labels)


def basic_prompts(
    *,
    anchor_system: str = "You decide who introduced themselves.",
    anchor: str = "speaker={speaker_id} values={values} text={text}",
) -> AnonymizerPrompts:
    """A minimal, well-shaped :class:`AnonymizerPrompts` — the ANCHOR
    section carries every slot :func:`paramem.cloud.anonymize_steps.
    ask_speaker_anchor` formats (``{speaker_id}``/``{values}``/``{text}``).
    """
    return AnonymizerPrompts(anchor_system=anchor_system, anchor=anchor)


# ---------------------------------------------------------------------------
# Stub GLiNER model — the shape ``span_tagger.load_at_startup``/``tag``/
# ``_windows`` read attributes off. Every attribute mirrors the real
# object's own attribute path (see span_tagger.py's docstrings), so a stub
# built here exercises the exact same code path the real model would.
# ---------------------------------------------------------------------------


class StubTransformerTokenizer:
    """Records every call; reports ``len(words) * subwords_per_word``
    subwords (floored at 1) — the controllable "subword inflation factor"
    the design's windowing tests need.
    """

    def __init__(self, subwords_per_word: float = 1.0):
        self.subwords_per_word = subwords_per_word
        self.calls: list[list[str]] = []

    def __call__(self, batch, is_split_into_words=True, truncation=True, add_special_tokens=True):
        words = list(batch[0])
        self.calls.append(words)
        n = max(1, round(len(words) * self.subwords_per_word))
        return {"input_ids": [list(range(n))]}


class StubDataProcessor:
    """Stand-in for ``GLiNER.data_processor`` — the three attributes/
    methods ``span_tagger`` reads off it: ``words_splitter``,
    ``prepare_inputs``, ``transformer_tokenizer``.
    """

    def __init__(self, *, subwords_per_word: float = 1.0):
        self.transformer_tokenizer = StubTransformerTokenizer(subwords_per_word)

    def words_splitter(self, text: str):
        return words_splitter(text)

    def prepare_inputs(self, examples, labels: Sequence[str]):
        """Mirror GLiNER's own ``[ENT] label1 [ENT] label2 ... [SEP]``
        prefix shape closely enough that a label added to *labels*
        lengthens the prefix word list — the property the "growing the
        label union narrows every window" test needs.
        """
        prefix: list[str] = []
        for label in labels:
            prefix.append("[ENT]")
            prefix.extend(label.split())
        prefix.append("[SEP]")
        return [prefix], [len(prefix)]


@dataclass
class StubGliner:
    """Stand-in for a loaded ``gliner.GLiNER`` instance.

    ``spans_fn(window_text, labels) -> list[dict]`` returns raw
    window-LOCAL span dicts (``start``/``end``/``text``/``label``/
    ``score``) — the same shape ``GLiNER.predict_entities`` returns.
    Every call is recorded on ``.calls`` for assertions on window count /
    label list / threshold.
    """

    max_len: int = 20
    max_width: int = 4
    max_position_embeddings: int = 50
    subwords_per_word: float = 1.0
    spans_fn: Callable[[str, list[str]], list[dict]] = field(
        default_factory=lambda: lambda text, labels: []
    )
    raise_with: BaseException | None = None

    def __post_init__(self):
        self.config = types.SimpleNamespace(max_len=self.max_len, max_width=self.max_width)
        self.model = types.SimpleNamespace(
            token_rep_layer=types.SimpleNamespace(
                bert_layer=types.SimpleNamespace(
                    model=types.SimpleNamespace(
                        config=types.SimpleNamespace(
                            max_position_embeddings=self.max_position_embeddings
                        )
                    )
                )
            )
        )
        self.data_processor = StubDataProcessor(subwords_per_word=self.subwords_per_word)
        self.calls: list[dict] = []

    def predict_entities(
        self, text: str, labels: list[str], *, flat_ner=True, multi_label=False, threshold=0.5
    ):
        self.calls.append(
            {
                "text": text,
                "labels": list(labels),
                "flat_ner": flat_ner,
                "multi_label": multi_label,
                "threshold": threshold,
            }
        )
        if self.raise_with is not None:
            raise self.raise_with
        return self.spans_fn(text, labels)


def install_stub_gliner_module(monkeypatch: pytest.MonkeyPatch, model: StubGliner) -> dict:
    """Inject a fake ``gliner`` module into ``sys.modules`` whose
    ``GLiNER.from_pretrained`` returns *model* and records its call
    arguments (``checkpoint``/``revision``/``map_location``).

    ``span_tagger.load_at_startup`` imports ``gliner`` lazily inside the
    function body (``from gliner import GLiNER``), so this must be in
    place before that call — the reason for ``monkeypatch.setitem(sys.
    modules, "gliner", ...)`` rather than a real install.
    """
    import sys

    captured: dict = {"calls": []}

    class _GLiNER:
        @staticmethod
        def from_pretrained(checkpoint, revision=None, map_location=None):
            call = {"checkpoint": checkpoint, "revision": revision, "map_location": map_location}
            captured["calls"].append(call)
            captured.update(call)
            return model

    stub_module = types.ModuleType("gliner")
    stub_module.GLiNER = _GLiNER
    monkeypatch.setitem(sys.modules, "gliner", stub_module)
    return captured


def install_raising_gliner_module(monkeypatch: pytest.MonkeyPatch, exc: BaseException) -> None:
    """Inject a fake ``gliner`` module whose ``GLiNER.from_pretrained``
    always raises *exc* — for the "checkpoint/revision could not be
    resolved" load-refusal path.
    """
    import sys

    class _GLiNER:
        @staticmethod
        def from_pretrained(checkpoint, revision=None, map_location=None):
            raise exc

    stub_module = types.ModuleType("gliner")
    stub_module.GLiNER = _GLiNER
    monkeypatch.setitem(sys.modules, "gliner", stub_module)


def reset_span_tagger() -> None:
    """Drop the loaded handle. Call in every test's teardown that installed
    one, so a later test never sees a leaked handle from an earlier test.
    """
    span_tagger.reset_for_tests()


@pytest.fixture
def span_tagger_reset():
    """Fixture form of :func:`reset_span_tagger` — resets both before and
    after the test.
    """
    reset_span_tagger()
    yield
    reset_span_tagger()
