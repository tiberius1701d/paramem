"""Local speech-to-text via Faster Whisper.

Loads a Whisper model on GPU (or CPU) alongside the LLM. The live load
itself is the VRAM gate — vram_measure captures the delta and raises
VramExhausted when the device is exhausted.
"""

import logging
from dataclasses import dataclass

import numpy as np
import torch

from paramem.server.vram_guard import VramExhausted, safe_empty_cache, vram_measure

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    """Result from Whisper transcription including language detection."""

    text: str
    language: str  # ISO 639-1 code (e.g. "en", "de", "fr")
    language_probability: float  # 0.0–1.0 confidence


class WhisperSTT:
    """Local Whisper STT with live-load VRAM measurement."""

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

    def load(self) -> bool:
        """Load the Whisper model. Returns True on success.

        Wraps WhisperModel construction in vram_measure so the actual load
        is the VRAM gate. On torch.cuda.OutOfMemoryError or VramExhausted
        the GPU cache is flushed and False is returned; the caller decides
        whether to fall back to CPU.

        device="auto" tries CUDA first, falls back to CPU on failure.
        device="cuda" returns False if load fails.
        device="cpu" loads unconditionally.
        """
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            logger.error("faster-whisper not installed. Install with: pip install paramem[stt]")
            return False

        device = self.device

        if device == "auto":
            if torch.cuda.is_available():
                loaded = self._try_load_on_device("cuda", WhisperModel)
                if loaded:
                    return True
                logger.warning(
                    "Whisper %s failed to load on CUDA with device=auto; falling back to CPU",
                    self.model_name,
                )
                device = "cpu"
            else:
                device = "cpu"
            return self._try_load_on_device(device, WhisperModel)

        return self._try_load_on_device(device, WhisperModel)

    def _try_load_on_device(self, device: str, WhisperModel) -> bool:
        """Attempt to load the Whisper model on a specific device.

        Uses vram_measure for CUDA devices to capture the actual footprint.
        Returns True on success, False on failure.
        """
        logger.info(
            "Loading Whisper %s on %s (%s)...",
            self.model_name,
            device,
            self.compute_type,
        )
        try:
            if device == "cuda":
                with vram_measure("stt") as m:
                    self._model = WhisperModel(
                        self.model_name,
                        device=device,
                        compute_type=self.compute_type,
                    )
                delta_mib = m["delta"] / (1024 * 1024)
                logger.info(
                    "Whisper %s loaded on %s, used %.0f MiB",
                    self.model_name,
                    device,
                    delta_mib,
                )
            else:
                self._model = WhisperModel(
                    self.model_name,
                    device=device,
                    compute_type=self.compute_type,
                )
                logger.info("Whisper %s loaded on %s", self.model_name, device)
        except VramExhausted:
            safe_empty_cache()
            logger.warning(
                "Insufficient VRAM for Whisper %s on %s — load failed",
                self.model_name,
                device,
            )
            return False
        except torch.cuda.OutOfMemoryError:
            safe_empty_cache()
            logger.warning(
                "OOM loading Whisper %s on %s",
                self.model_name,
                device,
            )
            return False
        except Exception:
            logger.exception("Failed to load Whisper model %s on %s", self.model_name, device)
            return False

        self.device = device
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
            # safe_empty_cache (not bare empty_cache) — clears cuBLAS workspaces
            # that empty_cache cannot touch; otherwise teardown leaves a ghost
            # CUDA context that pollutes the next boot's _gpu_occupied check.
            safe_empty_cache()
            logger.info("Whisper model unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._model is not None
