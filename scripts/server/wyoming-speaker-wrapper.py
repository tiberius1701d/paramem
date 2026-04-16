#!/usr/bin/env python3
"""Wyoming STT wrapper with speaker embedding extraction.

Wraps an existing Wyoming STT service, adding speaker embedding
computation via pyannote-audio. The embedding is passed to the
ParaMem server alongside the transcript.

Architecture:
  Voice Satellite → [this wrapper] → Upstream STT (Whisper)
                          ↓
                    pyannote-audio (CPU)
                          ↓
                    ParaMem /chat (with speaker_embedding)

Usage:
    python scripts/server/wyoming-speaker-wrapper.py \\
        --upstream-host localhost --upstream-port 10300 \\
        --port 10301 \\
        --paramem-url http://localhost:8420

Requirements:
    pip install paramem[speaker]  # installs pyannote-audio
"""

import argparse
import logging

logger = logging.getLogger(__name__)

# Lazy-loaded model singleton
_embedding_model = None


def _get_embedding_model():
    """Lazy-load the pyannote speaker embedding model (first call ~300MB download)."""
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model

    try:
        from pyannote.audio import Inference
    except ImportError:
        logger.error("pyannote-audio not installed. Install with: pip install paramem[speaker]")
        return None

    logger.info("Loading pyannote speaker embedding model...")
    _embedding_model = Inference(
        "pyannote/embedding",
        window="whole",
        device="cpu",
    )
    logger.info("Speaker embedding model loaded")
    return _embedding_model


def compute_speaker_embedding(
    audio_bytes: bytes,
    sample_rate: int = 16000,
) -> list[float]:
    """Compute a speaker embedding from raw PCM audio.

    Args:
        audio_bytes: Raw PCM audio (int16, mono).
        sample_rate: Audio sample rate in Hz.

    Returns:
        Speaker embedding as a list of floats (typically 256-512 dim).
        Empty list if pyannote-audio is not available or audio is too short.
    """
    model = _get_embedding_model()
    if model is None:
        return []

    import numpy as np
    import torch

    # Convert raw PCM int16 to float32 waveform
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    # Minimum audio length (~0.5s at 16kHz)
    if len(audio_np) < sample_rate // 2:
        logger.warning(
            "Audio too short for speaker embedding: %d samples (%.2fs)",
            len(audio_np),
            len(audio_np) / sample_rate,
        )
        return []

    waveform = torch.from_numpy(audio_np).unsqueeze(0)
    embedding = model({"waveform": waveform, "sample_rate": sample_rate})
    return embedding.flatten().tolist()


def main():
    parser = argparse.ArgumentParser(description="Wyoming STT wrapper with speaker identification")
    parser.add_argument("--upstream-host", default="localhost")
    parser.add_argument("--upstream-port", type=int, default=10300)
    parser.add_argument("--port", type=int, default=10301)
    parser.add_argument("--paramem-url", default="http://localhost:8420")
    parser.add_argument(
        "--test-audio",
        type=str,
        default=None,
        help="Test mode: compute embedding from a WAV file and exit",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    # Test mode: compute embedding from a file
    if args.test_audio:
        import wave

        with wave.open(args.test_audio, "rb") as wf:
            sample_rate = wf.getframerate()
            audio_bytes = wf.readframes(wf.getnframes())
            channels = wf.getnchannels()

        if channels > 1:
            # Convert to mono by taking first channel
            import numpy as np

            audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_np = audio_np[::channels]
            audio_bytes = audio_np.tobytes()

        embedding = compute_speaker_embedding(audio_bytes, sample_rate)
        if embedding:
            print(f"Embedding dimension: {len(embedding)}")
            print(f"First 10 values: {embedding[:10]}")
            print(f"Norm: {sum(x * x for x in embedding) ** 0.5:.4f}")
        else:
            print("Failed to compute embedding")
        return

    # Wyoming protocol handler — scaffold for future implementation
    logger.info(
        "Wyoming speaker wrapper: port %d, upstream %s:%d, ParaMem %s",
        args.port,
        args.upstream_host,
        args.upstream_port,
        args.paramem_url,
    )
    logger.info(
        "Wyoming protocol handler not yet implemented. "
        "Use --test-audio to verify pyannote-audio works."
    )

    # TODO: Implement Wyoming protocol handling:
    # 1. Listen for Wyoming AudioStart/AudioChunk/AudioStop events
    # 2. Accumulate audio chunks into a buffer
    # 3. Forward audio to upstream STT for transcription
    # 4. Compute speaker embedding from accumulated audio
    # 5. POST to ParaMem /chat with {text: transcript, speaker_embedding: [...]}
    # 6. Return TTS response to satellite


if __name__ == "__main__":
    main()
