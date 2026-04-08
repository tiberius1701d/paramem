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
    model = Model.from_pretrained("pyannote/embedding")
    _embedding_model = Inference(
        model,
        window="whole",
        device=torch.device("cpu"),
    )
    logger.info("Speaker embedding model loaded (4.3M params, CPU)")
    return True


def is_loaded() -> bool:
    """Check if the embedding model is ready."""
    return _embedding_model is not None


def compute_speaker_embedding(
    audio_bytes: bytes,
    sample_rate: int = 16000,
) -> list[float]:
    """Compute a speaker embedding from raw PCM audio (int16, mono).

    Returns a list of floats (512 dim).
    Empty list if model not loaded or audio is too short.
    """
    if _embedding_model is None:
        return []

    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    if len(audio_np) < sample_rate // 2:
        logger.warning(
            "Audio too short for speaker embedding: %d samples (%.2fs)",
            len(audio_np),
            len(audio_np) / sample_rate,
        )
        return []

    waveform = torch.from_numpy(audio_np).unsqueeze(0)
    embedding = _embedding_model({"waveform": waveform, "sample_rate": sample_rate})
    return embedding.flatten().tolist()
