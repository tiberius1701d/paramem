"""Tests for the personal-referent classifier — exemplar loader and
cosine-margin residual.

Mirrors :mod:`tests/test_sentence_type.py` and :mod:`tests/test_intent.py`
in shape.  Tests do not download the real encoder; they stub
:class:`_EncoderHandle` and inject deterministic embeddings so the
cosine-margin logic can be exercised without GPU or network.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from paramem.server import intent as intent_module
from paramem.server import personal_referent as pr_module
from paramem.server.config import PersonalReferentConfig
from paramem.server.intent import _EncoderHandle
from paramem.server.personal_referent import (
    PersonalReferent,
    _classify_via_encoder,
    _ExemplarBank,
    classify_personal_referent,
    get_exemplars,
    load_exemplars,
)


@pytest.fixture(autouse=True)
def reset_singletons():
    """Each test starts with fresh singletons (intent encoder + personal_referent)."""
    intent_module._encoder_singleton = None
    pr_module._exemplars_singleton = None
    yield
    intent_module._encoder_singleton = None
    pr_module._exemplars_singleton = None


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
        cfg = PersonalReferentConfig(enabled=False, exemplars_dir=str(tmp_path))
        assert load_exemplars(cfg) is None
        assert get_exemplars() is None

    def test_no_encoder_returns_none(self, tmp_path):
        cfg = PersonalReferentConfig(exemplars_dir=str(tmp_path))
        assert load_exemplars(cfg) is None
        assert get_exemplars() is None

    def test_missing_dir_returns_none(self, tmp_path):
        intent_module._encoder_singleton = _stub_encoder()
        cfg = PersonalReferentConfig(exemplars_dir=str(tmp_path / "does-not-exist"))
        assert load_exemplars(cfg) is None
        assert get_exemplars() is None

    def test_skips_files_with_unrecognised_class(self, tmp_path):
        encoder = _stub_encoder()
        encoder.model.encode.return_value = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        intent_module._encoder_singleton = encoder

        (tmp_path / "about_speaker.en.txt").write_text("Where do I live?\n")
        (tmp_path / "not_about_speaker.en.txt").write_text("What is the weather?\n")
        (tmp_path / "garbage.en.txt").write_text("not a class\n")

        cfg = PersonalReferentConfig(exemplars_dir=str(tmp_path))
        bank = load_exemplars(cfg)
        assert bank is not None
        assert len(bank.referents) == 2
        assert PersonalReferent.ABOUT_SPEAKER in bank.referents
        assert PersonalReferent.NOT_ABOUT_SPEAKER in bank.referents

    def test_loads_multilingual_files(self, tmp_path):
        encoder = _stub_encoder()
        encoder.model.encode.return_value = np.array(
            [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]],
            dtype=np.float32,
        )
        intent_module._encoder_singleton = encoder

        (tmp_path / "about_speaker.en.txt").write_text("Where do I live?\n")
        (tmp_path / "about_speaker.de.txt").write_text("Wo wohne ich?\n")
        (tmp_path / "not_about_speaker.en.txt").write_text("What is the weather?\n")
        (tmp_path / "not_about_speaker.de.txt").write_text("Wie ist das Wetter?\n")

        cfg = PersonalReferentConfig(exemplars_dir=str(tmp_path))
        bank = load_exemplars(cfg)
        assert bank is not None
        assert len(bank.referents) == 4
        # Both languages contribute exemplars to each class.
        assert bank.referents.count(PersonalReferent.ABOUT_SPEAKER) == 2
        assert bank.referents.count(PersonalReferent.NOT_ABOUT_SPEAKER) == 2

    def test_empty_dir_returns_none(self, tmp_path):
        intent_module._encoder_singleton = _stub_encoder()
        cfg = PersonalReferentConfig(exemplars_dir=str(tmp_path))
        assert load_exemplars(cfg) is None

    def test_blank_lines_and_comments_skipped(self, tmp_path):
        encoder = _stub_encoder()
        encoder.model.encode.return_value = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        intent_module._encoder_singleton = encoder

        (tmp_path / "about_speaker.en.txt").write_text("# header comment\n\nWhere do I live?\n")
        (tmp_path / "not_about_speaker.en.txt").write_text("What is the weather?\n")
        cfg = PersonalReferentConfig(exemplars_dir=str(tmp_path))
        bank = load_exemplars(cfg)
        assert bank is not None
        assert len(bank.referents) == 2


# ---------------------------------------------------------------------------
# _classify_via_encoder — cosine + margin gate
# ---------------------------------------------------------------------------


def _bank_two_classes(separation: float = 0.9) -> _ExemplarBank:
    """Bank with two about-speaker + two not-about-speaker exemplars,
    separated along orthogonal axes by ``separation``."""
    embeddings = np.array(
        [
            [separation, 0.0],
            [separation - 0.05, 0.0],
            [0.0, separation],
            [0.0, separation - 0.05],
        ],
        dtype=np.float32,
    )
    return _ExemplarBank(
        referents=[
            PersonalReferent.ABOUT_SPEAKER,
            PersonalReferent.ABOUT_SPEAKER,
            PersonalReferent.NOT_ABOUT_SPEAKER,
            PersonalReferent.NOT_ABOUT_SPEAKER,
        ],
        embeddings=embeddings,
        source_files=("about_speaker.en.txt", "not_about_speaker.en.txt"),
    )


class TestClassifyViaEncoder:
    def test_clear_about_speaker_returns_top_class(self):
        encoder = _stub_encoder()
        encoder.model.encode.return_value = np.array([[0.95, 0.05]], dtype=np.float32)
        bank = _bank_two_classes()
        cfg = PersonalReferentConfig(confidence_margin=0.05)
        verdict = _classify_via_encoder("Where do I live?", encoder, bank, cfg)
        assert verdict == PersonalReferent.ABOUT_SPEAKER

    def test_clear_not_about_speaker_returns_top_class(self):
        encoder = _stub_encoder()
        encoder.model.encode.return_value = np.array([[0.05, 0.95]], dtype=np.float32)
        bank = _bank_two_classes()
        cfg = PersonalReferentConfig(confidence_margin=0.05)
        verdict = _classify_via_encoder("What's the weather?", encoder, bank, cfg)
        assert verdict == PersonalReferent.NOT_ABOUT_SPEAKER

    def test_below_margin_returns_none(self):
        """Ambiguous query (both classes ~equidistant) → None so the
        sanitizer falls back to the English token-set heuristic."""
        encoder = _stub_encoder()
        encoder.model.encode.return_value = np.array([[0.5, 0.5]], dtype=np.float32)
        bank = _bank_two_classes()
        cfg = PersonalReferentConfig(confidence_margin=0.05)
        verdict = _classify_via_encoder("Berlin is nice.", encoder, bank, cfg)
        assert verdict is None

    def test_encode_failure_returns_none(self):
        encoder = _stub_encoder()
        encoder.model.encode.side_effect = RuntimeError("boom")
        bank = _bank_two_classes()
        cfg = PersonalReferentConfig()
        # Must not raise — embedding errors degrade to None so caller
        # uses the fallback heuristic.
        verdict = _classify_via_encoder("Where do I live?", encoder, bank, cfg)
        assert verdict is None


# ---------------------------------------------------------------------------
# classify_personal_referent — public entry point
# ---------------------------------------------------------------------------


class TestClassifyPersonalReferent:
    def test_no_config_returns_none(self):
        assert classify_personal_referent("Where do I live?") is None

    def test_disabled_config_returns_none(self):
        cfg = PersonalReferentConfig(enabled=False)
        assert classify_personal_referent("Where do I live?", config=cfg) is None

    def test_no_encoder_returns_none(self):
        cfg = PersonalReferentConfig()
        assert classify_personal_referent("Where do I live?", config=cfg) is None

    def test_no_exemplars_returns_none(self):
        intent_module._encoder_singleton = _stub_encoder()
        cfg = PersonalReferentConfig()
        assert classify_personal_referent("Where do I live?", config=cfg) is None

    def test_end_to_end_with_loaded_singletons(self):
        encoder = _stub_encoder()
        encoder.model.encode.return_value = np.array([[0.95, 0.05]], dtype=np.float32)
        intent_module._encoder_singleton = encoder
        pr_module._exemplars_singleton = _ExemplarBank(
            referents=[PersonalReferent.ABOUT_SPEAKER, PersonalReferent.NOT_ABOUT_SPEAKER],
            embeddings=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            source_files=("test",),
        )
        cfg = PersonalReferentConfig(confidence_margin=0.05)
        verdict = classify_personal_referent("Where do I live?", config=cfg)
        assert verdict == PersonalReferent.ABOUT_SPEAKER

    def test_never_raises_on_encoder_error(self):
        encoder = _stub_encoder()
        encoder.model.encode.side_effect = RuntimeError("device gone")
        intent_module._encoder_singleton = encoder
        pr_module._exemplars_singleton = _ExemplarBank(
            referents=[PersonalReferent.ABOUT_SPEAKER],
            embeddings=np.array([[1.0, 0.0]], dtype=np.float32),
            source_files=("test",),
        )
        cfg = PersonalReferentConfig()
        # Must not raise — exception → None.
        assert classify_personal_referent("Where is it?", config=cfg) is None
