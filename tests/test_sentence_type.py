"""Tests for the sentence-type classifier — exemplar loader and
cosine-margin residual.

Mirrors :mod:`tests/test_intent.py` in shape.  Tests do not download
the real encoder; they stub :class:`_EncoderHandle` and inject
deterministic embeddings so the cosine-margin logic can be exercised
without GPU or network.

The encoder load itself is covered by ``test_intent.py`` — the
sentence-type classifier reuses that singleton so there's no separate
loader to test here.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from paramem.server import intent as intent_module
from paramem.server import sentence_type as st_module
from paramem.server.config import SentenceTypeConfig
from paramem.server.intent import _EncoderHandle
from paramem.server.sentence_type import (
    SentenceType,
    _classify_via_encoder,
    _ExemplarBank,
    classify_sentence_type,
    get_exemplars,
    load_exemplars,
)


@pytest.fixture(autouse=True)
def reset_singletons():
    """Each test starts with fresh singletons (both intent + sentence_type)."""
    intent_module._encoder_singleton = None
    st_module._exemplars_singleton = None
    yield
    intent_module._encoder_singleton = None
    st_module._exemplars_singleton = None


def _stub_encoder(prefix: str = "query: ") -> _EncoderHandle:
    """Encoder handle backed by a MagicMock — script ``.encode.side_effect``
    or ``.encode.return_value`` to control the embedding output."""
    model = MagicMock()
    return _EncoderHandle(model=model, query_prefix=prefix, device="cpu", dtype="float32")


# ---------------------------------------------------------------------------
# load_exemplars
# ---------------------------------------------------------------------------


class TestLoadExemplars:
    def test_disabled_config_returns_none(self, tmp_path):
        cfg = SentenceTypeConfig(enabled=False, exemplars_dir=str(tmp_path))
        assert load_exemplars(cfg) is None
        assert get_exemplars() is None

    def test_no_encoder_returns_none(self, tmp_path):
        # Encoder singleton is None (autouse fixture cleared it).
        cfg = SentenceTypeConfig(exemplars_dir=str(tmp_path))
        assert load_exemplars(cfg) is None
        assert get_exemplars() is None

    def test_missing_dir_returns_none(self, tmp_path):
        intent_module._encoder_singleton = _stub_encoder()
        cfg = SentenceTypeConfig(exemplars_dir=str(tmp_path / "does-not-exist"))
        assert load_exemplars(cfg) is None
        assert get_exemplars() is None

    def test_skips_files_with_unrecognised_class(self, tmp_path, caplog):
        encoder = _stub_encoder()
        encoder.model.encode.return_value = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        intent_module._encoder_singleton = encoder

        (tmp_path / "interrogative.en.txt").write_text("Where is my phone?\n")
        (tmp_path / "non_interrogative.en.txt").write_text("Turn on the light.\n")
        (tmp_path / "garbage.en.txt").write_text("not a class\n")

        cfg = SentenceTypeConfig(exemplars_dir=str(tmp_path))
        bank = load_exemplars(cfg)
        assert bank is not None
        assert len(bank.types) == 2
        assert SentenceType.INTERROGATIVE in bank.types
        assert SentenceType.NON_INTERROGATIVE in bank.types

    def test_loads_multilingual_files(self, tmp_path):
        encoder = _stub_encoder()
        encoder.model.encode.return_value = np.array(
            [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
            dtype=np.float32,
        )
        intent_module._encoder_singleton = encoder

        (tmp_path / "interrogative.en.txt").write_text("Where is my phone?\n")
        (tmp_path / "interrogative.de.txt").write_text("Wo ist mein Handy?\n")
        (tmp_path / "non_interrogative.en.txt").write_text("Turn on the light.\n")
        (tmp_path / "non_interrogative.de.txt").write_text("Mache das Licht an.\n")

        cfg = SentenceTypeConfig(exemplars_dir=str(tmp_path))
        bank = load_exemplars(cfg)
        assert bank is not None
        assert len(bank.types) == 4
        # Both languages contribute exemplars to each class.
        assert bank.types.count(SentenceType.INTERROGATIVE) == 2
        assert bank.types.count(SentenceType.NON_INTERROGATIVE) == 2

    def test_empty_dir_returns_none(self, tmp_path):
        intent_module._encoder_singleton = _stub_encoder()
        cfg = SentenceTypeConfig(exemplars_dir=str(tmp_path))
        assert load_exemplars(cfg) is None

    def test_blank_lines_and_comments_skipped(self, tmp_path):
        encoder = _stub_encoder()
        encoder.model.encode.return_value = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        intent_module._encoder_singleton = encoder

        (tmp_path / "interrogative.en.txt").write_text("# header comment\n\nWhere is my phone?\n")
        (tmp_path / "non_interrogative.en.txt").write_text("Turn on the light.\n")
        cfg = SentenceTypeConfig(exemplars_dir=str(tmp_path))
        bank = load_exemplars(cfg)
        assert bank is not None
        assert len(bank.types) == 2


# ---------------------------------------------------------------------------
# _classify_via_encoder — cosine + margin gate
# ---------------------------------------------------------------------------


def _bank_two_classes(separation: float = 0.9) -> _ExemplarBank:
    """Bank with two interrogative + two non-interrogative exemplars,
    separated along orthogonal axes by ``separation``."""
    embeddings = np.array(
        [
            [separation, 0.0],  # interrogative
            [separation - 0.05, 0.0],  # interrogative
            [0.0, separation],  # non_interrogative
            [0.0, separation - 0.05],  # non_interrogative
        ],
        dtype=np.float32,
    )
    return _ExemplarBank(
        types=[
            SentenceType.INTERROGATIVE,
            SentenceType.INTERROGATIVE,
            SentenceType.NON_INTERROGATIVE,
            SentenceType.NON_INTERROGATIVE,
        ],
        embeddings=embeddings,
        source_files=("interrogative.en.txt", "non_interrogative.en.txt"),
    )


class TestClassifyViaEncoder:
    def test_clear_interrogative_returns_top_class(self):
        encoder = _stub_encoder()
        # Query embedding aligns with the interrogative axis.
        encoder.model.encode.return_value = np.array([[0.95, 0.05]], dtype=np.float32)
        bank = _bank_two_classes()
        cfg = SentenceTypeConfig(confidence_margin=0.05)
        verdict = _classify_via_encoder("Where is my phone?", encoder, bank, cfg)
        assert verdict == SentenceType.INTERROGATIVE

    def test_clear_non_interrogative_returns_top_class(self):
        encoder = _stub_encoder()
        # Query aligns with the non-interrogative axis.
        encoder.model.encode.return_value = np.array([[0.05, 0.95]], dtype=np.float32)
        bank = _bank_two_classes()
        cfg = SentenceTypeConfig(confidence_margin=0.05)
        verdict = _classify_via_encoder("Turn on the light.", encoder, bank, cfg)
        assert verdict == SentenceType.NON_INTERROGATIVE

    def test_below_margin_returns_none(self):
        """Ambiguous query (both classes ~equidistant) → None so the
        caller falls back to its deterministic heuristic."""
        encoder = _stub_encoder()
        # Query is roughly equidistant from both class centroids.
        encoder.model.encode.return_value = np.array([[0.5, 0.5]], dtype=np.float32)
        bank = _bank_two_classes()
        cfg = SentenceTypeConfig(confidence_margin=0.05)
        verdict = _classify_via_encoder("Berlin is nice.", encoder, bank, cfg)
        assert verdict is None

    def test_encode_failure_returns_none(self):
        encoder = _stub_encoder()
        encoder.model.encode.side_effect = RuntimeError("boom")
        bank = _bank_two_classes()
        cfg = SentenceTypeConfig()
        # Must not raise — embedding errors degrade to None so caller
        # uses the fallback heuristic.
        verdict = _classify_via_encoder("Where is my phone?", encoder, bank, cfg)
        assert verdict is None


# ---------------------------------------------------------------------------
# classify_sentence_type — public entry point
# ---------------------------------------------------------------------------


class TestClassifySentenceType:
    def test_no_config_returns_none(self):
        assert classify_sentence_type("Where do I live?") is None

    def test_disabled_config_returns_none(self):
        cfg = SentenceTypeConfig(enabled=False)
        assert classify_sentence_type("Where do I live?", config=cfg) is None

    def test_no_encoder_returns_none(self):
        # Singletons cleared by autouse fixture; encoder isn't loaded.
        cfg = SentenceTypeConfig()
        assert classify_sentence_type("Where do I live?", config=cfg) is None

    def test_no_exemplars_returns_none(self):
        intent_module._encoder_singleton = _stub_encoder()
        cfg = SentenceTypeConfig()
        # Encoder loaded but exemplars not.
        assert classify_sentence_type("Where do I live?", config=cfg) is None

    def test_end_to_end_with_loaded_singletons(self):
        encoder = _stub_encoder()
        # Exemplars at fixed embeddings; query at runtime — script the
        # second call (the query embedding) separately from the first
        # (the exemplar batch).
        encoder.model.encode.side_effect = [
            np.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            # Query close to interrogative axis.
            np.array([[0.95, 0.05]], dtype=np.float32),
        ]
        intent_module._encoder_singleton = encoder

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.is_dir", return_value=True),
            patch(
                "pathlib.Path.glob",
                return_value=[],
            ),
        ):
            # Manual bank construction to avoid filesystem dependency
            # — load_exemplars covered separately above.
            st_module._exemplars_singleton = _ExemplarBank(
                types=[SentenceType.INTERROGATIVE, SentenceType.NON_INTERROGATIVE],
                embeddings=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
                source_files=("test",),
            )

        cfg = SentenceTypeConfig(confidence_margin=0.05)
        # Query embedding for "Where do I live?" lands near interrogative.
        encoder.model.encode.side_effect = None
        encoder.model.encode.return_value = np.array([[0.95, 0.05]], dtype=np.float32)
        verdict = classify_sentence_type("Where do I live?", config=cfg)
        assert verdict == SentenceType.INTERROGATIVE

    def test_never_raises_on_encoder_error(self):
        encoder = _stub_encoder()
        encoder.model.encode.side_effect = RuntimeError("device gone")
        intent_module._encoder_singleton = encoder
        st_module._exemplars_singleton = _ExemplarBank(
            types=[SentenceType.INTERROGATIVE],
            embeddings=np.array([[1.0, 0.0]], dtype=np.float32),
            source_files=("test",),
        )
        cfg = SentenceTypeConfig()
        # Must not raise — exception → None.
        assert classify_sentence_type("Where is it?", config=cfg) is None
