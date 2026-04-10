"""Local speech-to-text via Faster Whisper.

Loads a Whisper model on GPU (or CPU) alongside the LLM.
VRAM check at startup ensures both models fit.
"""

import logging
from dataclasses import dataclass

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    """Result from Whisper transcription including language detection."""

    text: str
    language: str  # ISO 639-1 code (e.g. "en", "de", "fr")
    language_probability: float  # 0.0–1.0 confidence


# Approximate VRAM usage in MB per model size (INT8 quantization).
# float16 uses roughly 2x these values.
VRAM_ESTIMATES_INT8_MB = {
    "tiny": 400,
    "tiny.en": 400,
    "base": 500,
    "base.en": 500,
    "small": 1000,
    "small.en": 1000,
    "medium": 2000,
    "medium.en": 2000,
    "large-v1": 6000,
    "large-v2": 6000,
    "large-v3": 6000,
    "distil-large-v3": 2500,
    "distil-small.en": 800,
    "distil-medium.en": 1500,
}

# Multiplier for compute types relative to INT8
COMPUTE_TYPE_MULTIPLIER = {
    "int8": 1.0,
    "int16": 1.5,
    "float16": 2.0,
    "float32": 4.0,
}

VRAM_HEADROOM_MB = 200  # GPU inference lock prevents concurrent usage


class WhisperSTT:
    """Local Whisper STT with VRAM-aware loading."""

    def __init__(
        self,
        model_name: str,
        device: str,
        compute_type: str,
        language: str,
        beam_size: int = 5,
        vad_filter: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self._model = None

    def check_vram(self) -> bool:
        """Check if GPU has enough free VRAM for this model.

        Returns True if sufficient, False otherwise.
        Only relevant for device=cuda.
        """
        if self.device == "cpu":
            return True

        base_mb = VRAM_ESTIMATES_INT8_MB.get(self.model_name, 2000)
        multiplier = COMPUTE_TYPE_MULTIPLIER.get(self.compute_type, 1.0)
        estimated_mb = int(base_mb * multiplier)
        required_mb = estimated_mb + VRAM_HEADROOM_MB

        if not torch.cuda.is_available():
            logger.warning("CUDA not available — cannot load Whisper on GPU")
            return False

        free_bytes, total_bytes = torch.cuda.mem_get_info()
        free_mb = free_bytes / (1024 * 1024)

        logger.info(
            "VRAM check: Whisper %s needs ~%d MB, %d MB free (of %d MB total)",
            self.model_name,
            required_mb,
            int(free_mb),
            int(total_bytes / (1024 * 1024)),
        )

        if free_mb < required_mb:
            logger.warning(
                "Insufficient VRAM for Whisper %s: need %d MB, have %d MB free. STT disabled.",
                self.model_name,
                required_mb,
                int(free_mb),
            )
            return False
        return True

    def load(self) -> bool:
        """Load the Whisper model. Returns True on success."""
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            logger.error("faster-whisper not installed. Install with: pip install paramem[stt]")
            return False

        # Resolve device
        device = self.device
        if device == "auto":
            if torch.cuda.is_available() and self.check_vram():
                device = "cuda"
            else:
                logger.warning(
                    "VRAM insufficient for Whisper %s with device=auto. STT disabled.",
                    self.model_name,
                )
                return False
        elif device == "cuda" and not self.check_vram():
            return False

        logger.info(
            "Loading Whisper %s on %s (%s)...",
            self.model_name,
            device,
            self.compute_type,
        )

        try:
            self._model = WhisperModel(
                self.model_name,
                device=device,
                compute_type=self.compute_type,
            )
        except Exception:
            logger.exception("Failed to load Whisper model")
            return False

        logger.info("Whisper %s loaded on %s", self.model_name, device)
        return True

    def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000) -> TranscriptionResult:
        """Transcribe raw PCM audio (int16, mono) to text with language detection."""
        if self._model is None:
            logger.error("Whisper model not loaded")
            return TranscriptionResult(text="", language="en", language_probability=0.0)

        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        if len(audio_np) < sample_rate // 4:
            logger.warning(
                "Audio too short for transcription: %d samples (%.2fs)",
                len(audio_np),
                len(audio_np) / sample_rate,
            )
            return TranscriptionResult(text="", language="en", language_probability=0.0)

        segments, info = self._model.transcribe(
            audio_np,
            language=self.language if self.language != "auto" else None,
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
        )

        text = " ".join(segment.text.strip() for segment in segments)
        logger.debug(
            "Transcribed %.1fs audio: '%s' (language=%s, prob=%.2f)",
            info.duration,
            text[:100],
            info.language,
            info.language_probability,
        )
        return TranscriptionResult(
            text=text.strip(),
            language=info.language,
            language_probability=info.language_probability,
        )

    def unload(self) -> None:
        """Free the Whisper model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Whisper model unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None
