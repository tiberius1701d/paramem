"""Speaker embedding computation via pyannote-audio.

Runs on CPU — no GPU contention with Whisper or LLM.
Model is loaded at server startup via load_embedding_model().
Requires: pip install paramem[speaker]
"""

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)

_embedding_model = None

# HF model id for the speaker-embedding backend. Exposed so /status can
# surface the active backbone without the caller having to know pyannote.
EMBEDDING_MODEL_NAME = "pyannote/wespeaker-voxceleb-resnet34-LM"
EMBEDDING_BACKEND = "pyannote"
EMBEDDING_DEVICE = "cpu"


def load_embedding_model() -> bool:
    """Load the pyannote speaker embedding model at startup.

    Returns True if loaded successfully, False if unavailable.
    """
    global _embedding_model

    try:
        from pyannote.audio import Inference, Model
    except ImportError:
        logger.warning("pyannote-audio not installed — install with: pip install paramem[speaker]")
        return False

    logger.info("Loading pyannote speaker embedding model...")
    model = Model.from_pretrained(EMBEDDING_MODEL_NAME)
    _embedding_model = Inference(
        model,
        window="whole",
        device=torch.device(EMBEDDING_DEVICE),
    )
    logger.info("Speaker embedding model loaded: %s (%s)", EMBEDDING_MODEL_NAME, EMBEDDING_DEVICE)
    return True


def is_loaded() -> bool:
    """Check if the embedding model is ready."""
    return _embedding_model is not None


def compute_speaker_embedding(
    audio_bytes: bytes,
    sample_rate: int = 16000,
    min_duration_seconds: float = 1.0,
) -> list[float]:
    """Compute a speaker embedding from raw PCM audio (int16, mono).

    Uses the wespeaker-voxceleb-resnet34-LM model (256-dim speaker embedding).
    Returns a list of floats. Empty list if model not loaded or audio is too short.

    Args:
        audio_bytes: Raw PCM audio data (int16, mono).
        sample_rate: Sample rate of the audio in Hz.
        min_duration_seconds: Minimum audio duration required to compute an
            embedding. Utterances shorter than this threshold are discarded
            because pyannote needs sustained voice for a stable fingerprint.
    """
    if _embedding_model is None:
        return []

    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    # RMS normalisation — equalises gain across devices and recording conditions.
    target_rms = 0.1  # ~-20 dBFS
    current_rms = float(np.sqrt(np.mean(audio_np**2)))
    if current_rms > 1e-6:
        audio_np = audio_np * (target_rms / current_rms)
        audio_np = np.clip(audio_np, -1.0, 1.0)

    min_samples = int(min_duration_seconds * sample_rate)
    if len(audio_np) < min_samples:
        logger.warning(
            "Audio too short for speaker embedding: %d samples (%.2fs < %.1fs minimum)",
            len(audio_np),
            len(audio_np) / sample_rate,
            min_duration_seconds,
        )
        return []

    waveform = torch.from_numpy(audio_np).unsqueeze(0)
    embedding = _embedding_model({"waveform": waveform, "sample_rate": sample_rate})
    return embedding.flatten().tolist()
