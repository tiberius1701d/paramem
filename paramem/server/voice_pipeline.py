"""Transport-agnostic voice processing pipeline.

Provides :func:`process_utterance`, a single async function that encapsulates
the GPU/CPU core of speech-to-text: empty-buffer guard, STT-not-loaded guard,
transcription under the GPU lock, and optional speaker-embedding computation.

All Wyoming-specific and HTTP-specific concerns (writing Transcript events,
reading audio from a TCP stream, decoding audio containers via ffmpeg) live
in their respective callers — this module is the shared seam.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# GPU lock — thread-safe, protects against concurrent CUDA access.
# Re-exported from gpu_lock so callers need only import this module.
from paramem.server.gpu_lock import gpu_lock as _gpu_lock  # noqa: E402


@dataclass
class UtteranceResult:
    """Result of processing a single voice utterance.

    Attributes
    ----------
    text:
        Transcribed text.  Empty string when the STT returned nothing.
    language:
        BCP-47 language code detected by the STT model (e.g. ``"en"``,
        ``"de"``).  ``None`` when the STT did not report a language.
    language_probability:
        Confidence score for *language* in [0, 1].  ``0.0`` when the STT
        did not report a probability.
    embedding:
        Speaker embedding as a list of floats, or ``None`` when embedding
        computation was skipped (``compute_embedding=False``) or the audio
        was too short / the embedding model was unavailable.
    """

    text: str = ""
    language: str | None = None
    language_probability: float = 0.0
    embedding: list[float] | None = field(default=None)


async def process_utterance(
    pcm_bytes: bytes,
    sample_rate: int,
    sample_width: int,
    channels: int,
    stt,
    *,
    compute_embedding: bool = True,
    min_embedding_duration_seconds: float = 1.0,
) -> UtteranceResult:
    """Transcribe raw PCM audio and optionally compute a speaker embedding.

    This function is the shared transport-agnostic core.  It contains no
    Wyoming protocol writes and no HTTP response logic — those responsibilities
    belong to the callers (:class:`~paramem.server.wyoming_handler.SpeakerSTTHandler`
    and the ``POST /voice`` HTTP endpoint).

    Guards:
        - Empty *pcm_bytes* → returns ``UtteranceResult(text="")``.
        - ``stt is None`` or ``stt.is_loaded`` is falsy → returns
          ``UtteranceResult(text="")``.

    Transcription runs inside the shared :data:`~paramem.server.gpu_lock.gpu_lock`
    via :func:`asyncio.get_running_loop().run_in_executor` to avoid blocking the
    event loop.

    Speaker embedding computation (when ``compute_embedding=True``) runs on CPU
    in a thread executor after the GPU transcription step.  It requires the
    ``paramem[speaker]`` extra to be installed; when unavailable the embedding
    is silently omitted (``None``).

    Parameters
    ----------
    pcm_bytes:
        Raw int16 mono PCM audio data.
    sample_rate:
        Sample rate in Hz (e.g. 16000).
    sample_width:
        Bytes per sample (should be 2 for int16).
    channels:
        Channel count (should be 1 for mono).
    stt:
        Loaded STT instance exposing ``.is_loaded`` and
        ``.transcribe(audio_bytes, sample_rate) -> result`` where
        ``result.text``, ``result.language``, and
        ``result.language_probability`` are available.
    compute_embedding:
        When ``True`` (the default), attempt to compute a speaker embedding
        using :func:`~paramem.server.speaker_embedding.compute_speaker_embedding`.
        Pass ``False`` when identity is already known (e.g. from a bearer
        token) to skip the CPU cost.
    min_embedding_duration_seconds:
        Minimum audio duration required for a stable speaker embedding.
        Shorter utterances return ``embedding=None``.

    Returns
    -------
    UtteranceResult
        Populated result; *text* is empty on guard-trip or silent input.
    """
    if not pcm_bytes:
        logger.warning("process_utterance: empty PCM buffer — nothing to transcribe")
        return UtteranceResult()

    if stt is None or not stt.is_loaded:
        logger.warning("process_utterance: STT model not loaded — returning empty result")
        return UtteranceResult()

    duration = len(pcm_bytes) / (sample_rate * sample_width * channels)
    logger.info(
        "process_utterance: %d bytes (%.1fs at %dHz)",
        len(pcm_bytes),
        duration,
        sample_rate,
    )

    if sample_width != 2:
        logger.warning(
            "process_utterance: unexpected sample_width=%d (expected 2 for int16)",
            sample_width,
        )
    if channels != 1:
        logger.warning(
            "process_utterance: unexpected channels=%d (expected 1 for mono)",
            channels,
        )

    # Transcribe under the GPU lock to prevent concurrent CUDA access with
    # the LLM inference path or background training.
    loop = asyncio.get_running_loop()
    async with _gpu_lock():
        stt_result = await loop.run_in_executor(None, stt.transcribe, pcm_bytes, sample_rate)

    text = stt_result.text
    language = getattr(stt_result, "language", None)
    language_probability = float(getattr(stt_result, "language_probability", 0.0))

    logger.info(
        "process_utterance: transcript='%s' lang=%s prob=%.2f",
        text[:100] if text else "(empty)",
        language,
        language_probability,
    )

    # Speaker embedding — CPU-bound, no GPU lock needed.
    # Computed whenever compute_embedding=True, regardless of whether the STT
    # produced non-empty text.  The Wyoming/satellite path calls
    # process_utterance with compute_embedding=True and relies on the embedding
    # even when the transcript is empty (e.g. silent audio with a clear voice
    # signal).  POST /voice computes the embedding on the shared-token path
    # (auth_speaker_id is None); on the per-user token path compute_embedding
    # is False (cheap path, identity is authoritative from the token).
    embedding: list[float] | None = None
    if compute_embedding:
        embedding = await loop.run_in_executor(
            None,
            _compute_embedding_safe,
            pcm_bytes,
            sample_rate,
            min_embedding_duration_seconds,
        )

    return UtteranceResult(
        text=text or "",
        language=language,
        language_probability=language_probability,
        embedding=embedding,
    )


def _compute_embedding_safe(
    audio_bytes: bytes,
    sample_rate: int,
    min_duration_seconds: float,
) -> list[float] | None:
    """Compute a speaker embedding without raising.

    Returns the embedding list on success, ``None`` on any failure (import
    unavailable, audio too short, or computation error).

    Parameters
    ----------
    audio_bytes:
        Raw int16 mono PCM.
    sample_rate:
        Sample rate in Hz.
    min_duration_seconds:
        Minimum audio length required for a stable embedding.
    """
    try:
        from paramem.server.speaker_embedding import compute_speaker_embedding

        result = compute_speaker_embedding(
            audio_bytes,
            sample_rate,
            min_duration_seconds=min_duration_seconds,
        )
        return result if result else None
    except ImportError:
        logger.debug("Speaker embedding not available — install paramem[speaker]")
        return None
    except Exception:
        logger.warning("Speaker embedding computation failed", exc_info=True)
        return None
