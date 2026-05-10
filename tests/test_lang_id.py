"""Tests for `paramem/server/lang_id.py` — fastText lid.176 wrapper."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from paramem.server import lang_id


@pytest.fixture(autouse=True)
def _reset_singleton():
    lang_id.reset_for_tests()
    yield
    lang_id.reset_for_tests()


def _stub_fasttext(monkeypatch, label: str, prob: float) -> MagicMock:
    """Return a stub fasttext.load_model whose loaded model emits (label, prob)."""
    fake_model = MagicMock()
    fake_model.predict.return_value = ([label], [prob])

    fake_module = MagicMock()
    fake_module.load_model.return_value = fake_model

    monkeypatch.setitem(__import__("sys").modules, "fasttext", fake_module)
    return fake_module


def test_detect_returns_none_when_model_file_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(lang_id, "_default_model_path", lambda: tmp_path / "nope.bin")
    lang, prob = lang_id.detect("Wo wohne ich?")
    assert lang is None
    assert prob == 0.0


def test_detect_strips_label_prefix(tmp_path, monkeypatch):
    model_path = tmp_path / "lid.bin"
    model_path.write_bytes(b"stub")
    _stub_fasttext(monkeypatch, "__label__de", 0.97)
    lang, prob = lang_id.detect("Wo wohne ich?", model_path=model_path)
    assert lang == "de"
    assert prob == pytest.approx(0.97)


def test_detect_handles_label_without_prefix(tmp_path, monkeypatch):
    model_path = tmp_path / "lid.bin"
    model_path.write_bytes(b"stub")
    _stub_fasttext(monkeypatch, "tl", 0.81)
    lang, prob = lang_id.detect("Saan ako nakatira?", model_path=model_path)
    assert lang == "tl"
    assert prob == pytest.approx(0.81)


def test_detect_empty_input(tmp_path, monkeypatch):
    model_path = tmp_path / "lid.bin"
    model_path.write_bytes(b"stub")
    _stub_fasttext(monkeypatch, "__label__en", 0.99)
    assert lang_id.detect("", model_path=model_path) == (None, 0.0)
    assert lang_id.detect("   \n\t ", model_path=model_path) == (None, 0.0)


def test_detect_strips_newlines_before_predict(tmp_path, monkeypatch):
    model_path = tmp_path / "lid.bin"
    model_path.write_bytes(b"stub")
    fake_module = _stub_fasttext(monkeypatch, "__label__en", 0.95)
    lang_id.detect("hello\nworld", model_path=model_path)
    fake_module.load_model.return_value.predict.assert_called_once_with("hello world", k=1)


def test_failed_load_does_not_retry(tmp_path, monkeypatch, caplog):
    """A missing model file should warn once, then silently no-op on subsequent calls."""
    monkeypatch.setattr(lang_id, "_default_model_path", lambda: tmp_path / "nope.bin")
    with caplog.at_level("WARNING", logger="paramem.server.lang_id"):
        lang_id.detect("hello")
        lang_id.detect("again")
        lang_id.detect("third")
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    # Latching: at most one warning even though detect was called three times.
    assert len(warnings) <= 1


def test_singleton_reused(tmp_path, monkeypatch):
    model_path = tmp_path / "lid.bin"
    model_path.write_bytes(b"stub")
    fake_module = _stub_fasttext(monkeypatch, "__label__en", 0.99)
    lang_id.detect("hello", model_path=model_path)
    lang_id.detect("world", model_path=model_path)
    # load_model called exactly once across two detect() invocations.
    assert fake_module.load_model.call_count == 1


def test_predict_returns_no_labels(tmp_path, monkeypatch):
    """Some inputs (e.g. punctuation-only) cause fastText to return empty arrays."""
    model_path = tmp_path / "lid.bin"
    model_path.write_bytes(b"stub")
    fake_model = MagicMock()
    fake_model.predict.return_value = ([], [])
    fake_module = MagicMock()
    fake_module.load_model.return_value = fake_model
    monkeypatch.setitem(__import__("sys").modules, "fasttext", fake_module)
    assert lang_id.detect("???", model_path=model_path) == (None, 0.0)


def test_default_model_path():
    expected = Path("~/.cache/paramem/lang_id/lid.176.bin").expanduser()
    assert lang_id._default_model_path() == expected


def test_load_at_startup_uses_default_when_path_empty(tmp_path, monkeypatch):
    """Empty model_path string → falls through to _default_model_path()."""
    monkeypatch.setattr(lang_id, "_default_model_path", lambda: tmp_path / "absent.bin")
    handle = lang_id.load_at_startup("")
    assert handle is None  # file doesn't exist → fail-closed


def test_load_at_startup_warms_singleton(tmp_path, monkeypatch):
    """A successful load_at_startup() means subsequent detect() calls are O(1)."""
    model_path = tmp_path / "lid.bin"
    model_path.write_bytes(b"stub")
    fake_module = _stub_fasttext(monkeypatch, "__label__de", 0.99)
    handle = lang_id.load_at_startup(str(model_path))
    assert handle is not None
    # detect() reuses the warm singleton — load_model is called once total.
    lang_id.detect("Wo wohne ich?", model_path=model_path)
    lang_id.detect("Hallo Welt", model_path=model_path)
    assert fake_module.load_model.call_count == 1
