"""Text-to-speech engine abstraction with Piper and MMS-TTS backends.

Dual-engine design: Piper (ONNX) for languages with good voice models,
MMS-TTS (HuggingFace) for languages Piper doesn't cover (e.g. Tagalog).
Both support CPU and GPU execution.
"""

import gc
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from paramem.server.config import TTSConfig, TTSVoiceConfig

logger = logging.getLogger(__name__)


class TTSEngine(ABC):
    """Base class for TTS engines."""

    @abstractmethod
    def synthesize(self, text: str) -> tuple[bytes, int]:
        """Synthesize text to PCM audio.

        Returns (audio_bytes, sample_rate) where audio_bytes is
        16-bit signed integer PCM, mono.
        """

    @abstractmethod
    def load(self, device: str = "cpu") -> bool:
        """Load the model. Returns True on success."""

    @abstractmethod
    def unload(self) -> None:
        """Free model from memory."""

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Whether the model is currently loaded."""

    @property
    def actual_device(self) -> str:
        """The device the engine actually loaded on (may differ from requested)."""
        return getattr(self, "_actual_device", "cpu")


class PiperTTSEngine(TTSEngine):
    """Piper TTS via piper-tts Python library (ONNX VITS).

    Supports CPU and CUDA via ONNX Runtime execution providers.
    Models are small (~60-130MB) and fast (<300ms on CPU).
    """

    def __init__(self, model_name: str, model_dir: Path | None = None):
        self._model_name = model_name
        self._model_dir = model_dir or Path("data/ha/tts/piper")
        self._voice = None
        self._actual_device = "cpu"

    def load(self, device: str = "cpu") -> bool:
        try:
            from piper import PiperVoice
        except ImportError:
            logger.error("piper-tts not installed. Install with: pip install piper-tts")
            return False

        use_cuda = device == "cuda"
        if use_cuda:
            try:
                import onnxruntime

                if "CUDAExecutionProvider" not in onnxruntime.get_available_providers():
                    logger.warning(
                        "CUDA requested for Piper but onnxruntime-gpu not installed "
                        "(available providers: %s). Falling back to CPU.",
                        onnxruntime.get_available_providers(),
                    )
                    use_cuda = False
            except ImportError:
                logger.warning(
                    "CUDA requested for Piper but onnxruntime not installed. Falling back to CPU."
                )
                use_cuda = False

        try:
            model_path = self._resolve_model_path()
            if model_path is None:
                logger.error("Could not resolve Piper model: %s", self._model_name)
                return False

            # Piper config is model.onnx.json (not model.json)
            config_path = Path(str(model_path) + ".json")
            self._voice = PiperVoice.load(
                str(model_path),
                config_path=str(config_path) if config_path.exists() else None,
                use_cuda=use_cuda,
            )
            actual_device = "cuda" if use_cuda else "cpu"
            logger.info("Piper TTS loaded: %s on %s", self._model_name, actual_device)
            self._actual_device = actual_device
            return True
        except Exception:
            logger.exception("Failed to load Piper model %s", self._model_name)
            return False

    def _resolve_model_path(self) -> Path | None:
        """Resolve model name to .onnx file path.

        Checks local data dir first, then attempts download.
        """
        data_dir = self._model_dir
        data_dir.mkdir(parents=True, exist_ok=True)

        # Check if model exists locally
        model_file = data_dir / f"{self._model_name}.onnx"
        if model_file.exists():
            return model_file

        # Try downloading via piper_download
        try:
            from piper.download import ensure_voice_exists, find_voice, get_voices

            voices = get_voices(data_dir, update_voices=True)
            voice_name = find_voice(self._model_name, voices)
            ensure_voice_exists(voice_name, [data_dir], data_dir, voices)
            # After download, find the .onnx file matching this voice
            for onnx_file in data_dir.rglob("*.onnx"):
                if self._model_name.replace("-", "_") in str(onnx_file).replace("-", "_"):
                    return onnx_file
            logger.warning("Downloaded Piper voice but .onnx not found for %s", self._model_name)
        except ImportError:
            logger.warning("piper.download not available — model must be pre-downloaded")
        except Exception:
            logger.exception("Failed to download Piper model %s", self._model_name)

        return None

    def synthesize(self, text: str) -> tuple[bytes, int]:
        if self._voice is None:
            raise RuntimeError("Piper model not loaded")

        # Use raw PCM streaming — avoids WAV encode/decode round-trip
        sample_rate = self._voice.config.sample_rate
        chunks = []
        for audio_bytes in self._voice.synthesize_stream_raw(text):
            chunks.append(audio_bytes)

        return b"".join(chunks), sample_rate

    def unload(self) -> None:
        self._voice = None
        logger.info("Piper TTS unloaded: %s", self._model_name)

    @property
    def is_loaded(self) -> bool:
        return self._voice is not None


class MMSTTSEngine(TTSEngine):
    """Facebook MMS-TTS via HuggingFace transformers (VITS).

    Supports 1100+ languages. Larger models (~300MB) and slower on CPU
    (~1-3s per sentence), but ~50-200ms on GPU.
    """

    def __init__(self, model_id: str, vram_safety_margin_mb: int = 200):
        self._model_id = model_id
        self._vram_safety_margin_mb = vram_safety_margin_mb
        self._model = None
        self._tokenizer = None
        self._device = "cpu"
        self._actual_device = "cpu"

    def load(self, device: str = "cpu") -> bool:
        try:
            from transformers import VitsModel, VitsTokenizer
        except ImportError:
            logger.error(
                "transformers not installed or too old for VitsModel. "
                "Install with: pip install transformers>=4.33"
            )
            return False

        try:
            self._tokenizer = VitsTokenizer.from_pretrained(self._model_id)

            if device == "cuda" and torch.cuda.is_available():
                free_bytes, _ = torch.cuda.mem_get_info()
                free_mb = free_bytes / (1024 * 1024)
                if free_mb > self._vram_safety_margin_mb:
                    self._model = VitsModel.from_pretrained(self._model_id, device_map="cuda")
                    self._device = "cuda"
                    self._actual_device = "cuda"
                    logger.info(
                        "MMS-TTS loaded on GPU: %s (%.0f MB free)",
                        self._model_id,
                        free_mb,
                    )
                else:
                    logger.warning(
                        "Insufficient VRAM for MMS-TTS (%.0f MB free), using CPU",
                        free_mb,
                    )
                    self._model = VitsModel.from_pretrained(self._model_id)
                    self._device = "cpu"
            else:
                self._model = VitsModel.from_pretrained(self._model_id)
                self._device = "cpu"
                logger.info("MMS-TTS loaded on CPU: %s", self._model_id)

            return True
        except Exception:
            logger.exception("Failed to load MMS-TTS model %s", self._model_id)
            return False

    def synthesize(self, text: str) -> tuple[bytes, int]:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("MMS-TTS model not loaded")

        inputs = self._tokenizer(text, return_tensors="pt").to(self._device)

        with torch.no_grad():
            output = self._model(**inputs)

        # Output waveform is float32, shape (1, num_samples)
        waveform = output.waveform[0].cpu().numpy()
        sample_rate = self._model.config.sampling_rate

        # Convert float32 [-1, 1] to int16 PCM via numpy (vectorized, no Python loop)
        pcm_data = (waveform * 32767).clip(-32768, 32767).astype(np.int16).tobytes()

        return pcm_data, sample_rate

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("MMS-TTS unloaded: %s", self._model_id)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None


def _piper_factory(
    voice_config: TTSVoiceConfig, tts_config: TTSConfig, vram_safety_margin_mb: int
) -> TTSEngine:
    model_dir = Path(tts_config.model_dir) if tts_config.model_dir else None
    piper_dir = (model_dir / "piper") if model_dir else None
    return PiperTTSEngine(model_name=voice_config.model, model_dir=piper_dir)


def _mms_factory(
    voice_config: TTSVoiceConfig, tts_config: TTSConfig, vram_safety_margin_mb: int
) -> TTSEngine:
    return MMSTTSEngine(model_id=voice_config.model, vram_safety_margin_mb=vram_safety_margin_mb)


# Registry of engine factory functions.
# Each callable: (voice_config, tts_config, vram_margin) -> TTSEngine.
# Adding a new provider: write a factory function and add one entry here.
EngineFactory = Callable[[TTSVoiceConfig, TTSConfig, int], TTSEngine]
ENGINE_REGISTRY: dict[str, EngineFactory] = {
    "piper": _piper_factory,
    "mms": _mms_factory,
}


def _create_engine(
    voice_config: TTSVoiceConfig,
    tts_config: TTSConfig,
    vram_safety_margin_mb: int = 200,
) -> TTSEngine:
    """Create a TTS engine from config using the engine registry."""
    factory = ENGINE_REGISTRY.get(voice_config.engine)
    if factory is None:
        available = ", ".join(sorted(ENGINE_REGISTRY.keys()))
        raise ValueError(f"Unknown TTS engine: '{voice_config.engine}'. Available: {available}")
    return factory(voice_config, tts_config, vram_safety_margin_mb)


class TTSManager:
    """Routes synthesis requests to the correct engine based on language.

    Preloads configured voices at startup. Falls back to default language
    if the requested language has no voice configured.
    """

    def __init__(self, config: TTSConfig, vram_safety_margin_mb: int = 200):
        self._config = config
        self._vram_safety_margin_mb = vram_safety_margin_mb
        self._engines: dict[str, TTSEngine] = {}
        self._engine_devices: dict[str, str] = {}  # lang → actual device after load
        self._default_language = config.default_language

    def load_all(self) -> None:
        """Load all configured voice engines."""
        for lang_code, voice_config in self._config.voices.items():
            device = voice_config.device or self._config.device
            engine = _create_engine(voice_config, self._config, self._vram_safety_margin_mb)

            if engine.load(device=device):
                self._engines[lang_code] = engine
                self._engine_devices[lang_code] = engine.actual_device
            elif device == "cuda":
                # GPU failed — try CPU fallback
                logger.warning(
                    "GPU load failed for %s TTS (%s), falling back to CPU",
                    lang_code,
                    voice_config.engine,
                )
                if engine.load(device="cpu"):
                    self._engines[lang_code] = engine
                    self._engine_devices[lang_code] = engine.actual_device
                else:
                    logger.error("Failed to load TTS for language: %s", lang_code)
            else:
                logger.error("Failed to load TTS for language: %s", lang_code)

        if not self._engines:
            logger.error("No TTS voices loaded — TTS will be unavailable")
        else:
            logger.info(
                "TTS ready: %d voices loaded (%s)",
                len(self._engines),
                ", ".join(sorted(self._engines.keys())),
            )

    def synthesize(self, text: str, language: str | None = None) -> tuple[bytes, int]:
        """Synthesize text in the given language.

        Falls back to default language if the requested language
        has no engine loaded.
        """
        lang = language or self._default_language
        engine = self._engines.get(lang)

        if engine is None:
            logger.warning(
                "No TTS voice for language '%s', falling back to '%s'",
                lang,
                self._default_language,
            )
            engine = self._engines.get(self._default_language)

        if engine is None:
            raise RuntimeError("No TTS engines available")

        return engine.synthesize(text)

    def needs_gpu(self, language: str | None = None) -> bool:
        """Check if synthesizing this language requires GPU access."""
        lang = language or self._default_language
        device = self._engine_devices.get(lang)
        if device is None:
            # Unknown language — check default
            device = self._engine_devices.get(self._default_language, "cpu")
        return device == "cuda"

    @property
    def any_on_gpu(self) -> bool:
        """Whether any loaded engine is on GPU."""
        return "cuda" in self._engine_devices.values()

    @property
    def engine_devices(self) -> dict[str, str]:
        """Return a copy of the lang-code → actual-device map.

        Surfaced for /status so pstatus can display the resolved device
        (and flag legal CPU fallback paths) without poking the private
        ``_engine_devices`` attribute.
        """
        return dict(self._engine_devices)

    def has_language(self, language: str) -> bool:
        """Check if a language has a loaded TTS engine."""
        return language in self._engines

    @property
    def available_languages(self) -> list[str]:
        """List of languages with loaded engines."""
        return sorted(self._engines.keys())

    def unload_all(self) -> None:
        """Unload all engines and free memory."""
        for lang_code, engine in self._engines.items():
            engine.unload()
        self._engines.clear()
        self._engine_devices.clear()
        logger.info("All TTS engines unloaded")

    @property
    def is_loaded(self) -> bool:
        return len(self._engines) > 0
