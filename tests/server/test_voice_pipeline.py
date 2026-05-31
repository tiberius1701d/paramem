"""Unit tests for paramem.server.voice_pipeline.process_utterance.

CPU-only — no real STT model, no GPU required.

Covers:
- Normal transcription path (returns text + language + probability).
- Empty-buffer guard (returns empty UtteranceResult).
- STT-not-loaded guard (stt.is_loaded = False → empty result).
- STT is None guard → empty result.
- compute_embedding=False skips embedding computation entirely.
- compute_embedding=True with short audio → embedding=None (too short).
- Successful embedding returned when compute_speaker_embedding succeeds.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from unittest.mock import patch

import pytest

from paramem.server.voice_pipeline import UtteranceResult, process_utterance

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _FakeSTTResult:
    """Minimal STT result returned by the stub transcribe()."""

    text: str
    language: str = "en"
    language_probability: float = 0.95


class _FakeSTT:
    """Stub STT with a configurable transcribe return value."""

    def __init__(self, text: str = "hello world", loaded: bool = True) -> None:
        self.is_loaded = loaded
        self._text = text
        self.transcribe_call_count = 0

    def transcribe(self, audio_bytes: bytes, sample_rate: int) -> _FakeSTTResult:
        self.transcribe_call_count += 1
        return _FakeSTTResult(
            text=self._text,
            language="en",
            language_probability=0.92,
        )


# Minimal PCM — 0.1 s of silence at 16 kHz int16 mono (3200 bytes).
_TINY_PCM = b"\x00\x00" * 1600
# 2 s of silence at 16 kHz int16 mono (64 000 bytes) — passes the 1 s floor.
_LONG_PCM = b"\x00\x00" * 16000


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_normal_transcription():
    """Happy-path: returns text, language, and probability from the STT stub."""
    stt = _FakeSTT(text="hello world")

    result = asyncio.run(process_utterance(_TINY_PCM, 16000, 2, 1, stt, compute_embedding=False))

    assert isinstance(result, UtteranceResult)
    assert result.text == "hello world"
    assert result.language == "en"
    assert result.language_probability == pytest.approx(0.92)
    assert result.embedding is None
    assert stt.transcribe_call_count == 1


def test_empty_buffer_returns_empty():
    """Empty PCM bytes → UtteranceResult with empty text, no STT call."""
    stt = _FakeSTT(text="should not appear")

    result = asyncio.run(process_utterance(b"", 16000, 2, 1, stt, compute_embedding=False))

    assert result.text == ""
    assert result.embedding is None
    assert stt.transcribe_call_count == 0


def test_stt_not_loaded_returns_empty():
    """stt.is_loaded = False → guard trips, returns empty, no transcription."""
    stt = _FakeSTT(text="should not appear", loaded=False)

    result = asyncio.run(process_utterance(_TINY_PCM, 16000, 2, 1, stt, compute_embedding=False))

    assert result.text == ""
    assert stt.transcribe_call_count == 0


def test_stt_none_returns_empty():
    """stt=None → guard trips, returns empty result."""
    result = asyncio.run(
        process_utterance(_TINY_PCM, 16000, 2, 1, stt=None, compute_embedding=False)
    )

    assert result.text == ""


def test_compute_embedding_false_skips_embedding():
    """compute_embedding=False → embedding is None even with long audio."""
    stt = _FakeSTT(text="long utterance")

    with patch("paramem.server.voice_pipeline._compute_embedding_safe") as mock_embed:
        result = asyncio.run(
            process_utterance(_LONG_PCM, 16000, 2, 1, stt, compute_embedding=False)
        )

    mock_embed.assert_not_called()
    assert result.embedding is None


def test_compute_embedding_true_calls_embedding():
    """compute_embedding=True with known-good audio → embedding returned."""
    stt = _FakeSTT(text="utterance with embedding")
    fake_embedding = [0.1, 0.2, 0.3]

    with patch(
        "paramem.server.voice_pipeline._compute_embedding_safe",
        return_value=fake_embedding,
    ) as mock_embed:
        result = asyncio.run(process_utterance(_LONG_PCM, 16000, 2, 1, stt, compute_embedding=True))

    mock_embed.assert_called_once()
    assert result.embedding == fake_embedding


def test_embedding_none_when_audio_too_short():
    """compute_embedding=True but embedding returns None (too short audio)."""
    stt = _FakeSTT(text="hi")

    with patch(
        "paramem.server.voice_pipeline._compute_embedding_safe",
        return_value=None,
    ):
        result = asyncio.run(process_utterance(_TINY_PCM, 16000, 2, 1, stt, compute_embedding=True))

    assert result.embedding is None


def test_empty_stt_text_still_computes_embedding():
    """Empty STT text does NOT suppress embedding when compute_embedding=True.

    The Wyoming/satellite path calls process_utterance with compute_embedding=True
    and relies on the embedding for speaker identification even when the audio is
    silent or the STT produced no transcript.  The condition was previously
    ``if compute_embedding and text`` — the ``and text`` gate was dropped so the
    satellite path is not broken by silent audio.

    POST /voice passes compute_embedding=False (token auth), so that path is
    unaffected.
    """
    stt = _FakeSTT(text="")
    fake_embedding = [0.4, 0.5, 0.6]

    with patch(
        "paramem.server.voice_pipeline._compute_embedding_safe",
        return_value=fake_embedding,
    ) as mock_embed:
        result = asyncio.run(process_utterance(_LONG_PCM, 16000, 2, 1, stt, compute_embedding=True))

    mock_embed.assert_called_once()
    assert result.text == ""
    assert result.embedding == fake_embedding
