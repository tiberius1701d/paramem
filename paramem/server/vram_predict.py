"""Cache-derived VRAM predictors for the ParaMem server.

Replaces static per-model tables with disk-cache-derived predictions.
Each predictor returns None on a cache miss — callers proceed without
the estimate and rely on the live load gate instead.
"""

from __future__ import annotations

import logging
from pathlib import Path

from paramem.server.config import DEFAULT_DATA_DIR

logger = logging.getLogger(__name__)

try:
    from huggingface_hub import try_to_load_from_cache
    from huggingface_hub.file_download import _CACHED_NO_EXIST
except ImportError:
    try_to_load_from_cache = None  # type: ignore[assignment]
    _CACHED_NO_EXIST = None  # type: ignore[assignment]

# Structural unit conversions (runtime bytes per param vs disk 2 bytes/param).
# NOT calibrated — these are pure ratios from the quant scheme definition.
# The empirical NF4 factor is config-driven (vram.nf4_disk_to_runtime_factor)
# so a quant-scheme swap can be tuned in yaml.
_INT8_FACTOR: float = 0.5  # 1 byte/param vs disk's 2 bytes/param
_FP16_FACTOR: float = 1.0  # 2 bytes/param matches disk
_FP32_FACTOR: float = 2.0  # 4 bytes/param vs disk's 2

# CT2 / faster-whisper compute_type unit conversions on top of the fp16 disk
# weights Systran ships. Structural (per CT2 quant semantics), NOT calibrated —
# the empirical workspace overhead is config-driven (vram.stt_workspace_factor).
_CT2_COMPUTE_TYPE_FACTOR: dict[str, float] = {
    "int8": 0.5,  # halves fp16 disk weights
    "int8_float16": 0.5,
    "int8_float32": 0.5,
    "int16": 0.75,
    "float16": 1.0,
    "float32": 2.0,
}


def _hf_cache_dir(model_id: str) -> Path | None:
    """Resolve the HuggingFace cache snapshot dir for ``model_id``.

    Uses huggingface_hub.try_to_load_from_cache to discover the snapshot
    dir for the canonical config.json. Returns None if not cached.
    """
    if try_to_load_from_cache is None:
        logger.debug("huggingface_hub not available; cannot predict VRAM from cache")
        return None

    try:
        cache_path = try_to_load_from_cache(model_id, "config.json")
    except Exception as exc:  # noqa: BLE001
        logger.debug("try_to_load_from_cache failed for %r: %s", model_id, exc)
        return None

    if cache_path is None or cache_path is _CACHED_NO_EXIST:
        return None

    p = Path(cache_path)
    if not p.exists():
        return None
    return p.parent


def _sum_dir_bytes(directory: Path, suffixes: tuple[str, ...] | None = None) -> int:
    """Sum file sizes under directory, optionally filtered to suffixes."""
    total = 0
    for child in directory.rglob("*"):
        if not child.is_file():
            continue
        if suffixes is None or child.suffix in suffixes:
            try:
                total += child.stat().st_size
            except OSError:
                pass
    return total


def predict_base_bytes(model_config, *, nf4_disk_to_runtime_factor: float) -> int | None:
    """Predict base-model GPU bytes from cached weights × quant factor.

    Reads ``model_config.model_id``, ``quantization``. Sums *.safetensors and
    pytorch_model-*.bin (one or the other) under the HF cache snapshot dir and
    multiplies by the appropriate quant factor. Returns None when the model
    isn't in the HF cache.

    Args:
        model_config: object exposing ``model_id`` and ``quantization``.
        nf4_disk_to_runtime_factor: Empirical runtime-bytes/disk-bytes ratio for
            BNB NF4 (config-driven via ``vram.nf4_disk_to_runtime_factor``).
            int8 / fp16 / fp32 use the structural unit-conversion factors.
    """
    model_id = getattr(model_config, "model_id", None)
    quantization = getattr(model_config, "quantization", "nf4")

    if not model_id:
        return None

    snap = _hf_cache_dir(model_id)
    if snap is None:
        return None

    # Sum weight shards — safetensors first, fall back to legacy .bin
    weight_bytes = _sum_dir_bytes(snap, suffixes=(".safetensors",))
    if weight_bytes == 0:
        weight_bytes = _sum_dir_bytes(snap, suffixes=(".bin",))

    if weight_bytes == 0:
        logger.debug("No weight shards found in HF cache for %r", model_id)
        return None

    factor_map = {
        "nf4": nf4_disk_to_runtime_factor,
        "int8": _INT8_FACTOR,
        "fp16": _FP16_FACTOR,
        "float16": _FP16_FACTOR,
        "fp32": _FP32_FACTOR,
        "float32": _FP32_FACTOR,
    }
    factor = factor_map.get(quantization, nf4_disk_to_runtime_factor)
    predicted = int(weight_bytes * factor)
    logger.debug(
        "predict_base_bytes(%r, %r): disk=%d MiB × %.2f = %d MiB",
        model_id,
        quantization,
        weight_bytes >> 20,
        factor,
        predicted >> 20,
    )
    return predicted


def predict_stt_bytes(
    stt_config,
    *,
    workspace_factor: float,
    permanent_cloud_only: bool = False,
) -> int | None:
    """Predict GPU bytes for faster-whisper STT.

    Sums files under the CT2 cache snapshot dir, multiplies by the runtime
    compute_type factor (CT2 quantizes fp16 disk weights at load) and the
    operator-configurable workspace overhead. Returns 0 for cpu/disabled/
    permanent_cloud_only. Returns None if cuda but cache miss.

    Args:
        stt_config: STTConfig-like object.
        workspace_factor: Empirical activation/workspace overhead multiplier
            (config-driven via ``vram.stt_workspace_factor``).
        permanent_cloud_only: When True, returns 0 unconditionally.
    """
    if not getattr(stt_config, "enabled", False):
        return 0

    device = getattr(stt_config, "device", "cpu")
    if device not in ("cuda", "auto"):
        return 0
    if permanent_cloud_only:
        return 0

    model_name = getattr(stt_config, "model", "")
    if not model_name:
        return None

    # Cache name pattern depends on the model family:
    #   plain whisper:   Systran/faster-whisper-{name}     ("small", "base.en")
    #   distil variant:  Systran/faster-distil-whisper-{rest}  ("distil-large-v3"
    #                    → "Systran/faster-distil-whisper-large-v3")
    #   turbo:           mobiuslabsgmbh/faster-whisper-large-v3-turbo (the repo
    #                    faster-whisper resolves "large-v3-turbo" to)
    #   legacy:          guillaumekln/faster-whisper-{name}
    candidates = []
    if model_name.startswith("distil-"):
        candidates.append(f"Systran/faster-distil-whisper-{model_name[len('distil-') :]}")
    candidates.append(f"mobiuslabsgmbh/faster-whisper-{model_name}")
    candidates.append(f"Systran/faster-whisper-{model_name}")
    candidates.append(f"guillaumekln/faster-whisper-{model_name}")

    snap = None
    for repo_id in candidates:
        snap = _hf_cache_dir(repo_id)
        if snap is not None:
            break
    if snap is None:
        logger.debug("STT model %r not cached; live gate is authoritative", model_name)
        return None

    disk_bytes = _sum_dir_bytes(snap)
    compute_type = getattr(stt_config, "compute_type", "float16")
    compute_factor = _CT2_COMPUTE_TYPE_FACTOR.get(compute_type, 1.0)
    predicted = int(disk_bytes * compute_factor * workspace_factor)
    logger.debug(
        "predict_stt_bytes(%r, %r): disk=%d MiB × %.2f × %.1f = %d MiB",
        model_name,
        compute_type,
        disk_bytes >> 20,
        compute_factor,
        workspace_factor,
        predicted >> 20,
    )
    return predicted


def predict_tts_bytes(
    tts_config,
    *,
    piper_ort_context_bytes: int,
    permanent_cloud_only: bool = False,
) -> int | None:
    """Predict GPU bytes for TTS.

    Piper voices: sum *.onnx file sizes from the configured Piper data dir
    (per voice on GPU) + ``piper_ort_context_bytes`` if any Piper voice
    is on GPU (the ONNX Runtime CUDA context is shared across voices).
    MMS voices: sum cached safetensors per voice on GPU.
    Returns 0 for cpu/disabled/permanent_cloud_only. Returns None if any
    GPU voice is uncached.

    Args:
        tts_config: TTSConfig-like object.
        piper_ort_context_bytes: Shared ONNX Runtime CUDA context size in bytes
            (config-driven via ``vram.tts_piper_ort_context_mib``).
        permanent_cloud_only: When True, returns 0 unconditionally.
    """
    if not getattr(tts_config, "enabled", False):
        return 0

    default_device = getattr(tts_config, "device", "cpu")
    if permanent_cloud_only and default_device != "cpu":
        return 0

    voices = getattr(tts_config, "voices", {}) or {}
    if not voices:
        return 0

    # Resolve Piper model_dir from config (mirrors PiperTTSEngine.__init__)
    model_dir_str = getattr(tts_config, "model_dir", "") or ""
    piper_dir = Path(model_dir_str) if model_dir_str else DEFAULT_DATA_DIR / "tts" / "piper"

    total_bytes = 0
    piper_voices_on_gpu = 0
    kokoro_voices_on_gpu = 0
    any_cache_miss = False

    for _lang, voice_config in voices.items():
        voice_device = getattr(voice_config, "device", None) or default_device
        if voice_device != "cuda":
            continue
        engine = getattr(voice_config, "engine", "").lower()
        voice_model = getattr(voice_config, "model", "")

        if engine == "piper":
            onnx_file = piper_dir / f"{voice_model}.onnx"
            if not onnx_file.exists():
                logger.debug("Piper voice %r not in local dir; cache miss", voice_model)
                any_cache_miss = True
                continue
            try:
                total_bytes += onnx_file.stat().st_size
            except OSError:
                any_cache_miss = True
                continue
            piper_voices_on_gpu += 1

        elif engine == "mms":
            snap = _hf_cache_dir(voice_model)
            if snap is None:
                logger.debug("MMS voice %r not cached; cache miss", voice_model)
                any_cache_miss = True
                continue
            total_bytes += _sum_dir_bytes(snap, suffixes=(".safetensors",))

        elif engine == "kokoro":
            # The Kokoro KModel is shared across all voices; counted once below.
            kokoro_voices_on_gpu += 1

    if piper_voices_on_gpu > 0:
        total_bytes += piper_ort_context_bytes

    if kokoro_voices_on_gpu > 0:
        snap = _hf_cache_dir("hexgrad/Kokoro-82M")
        if snap is None:
            logger.debug("Kokoro-82M model not cached; cache miss")
            any_cache_miss = True
        else:
            total_bytes += _sum_dir_bytes(snap, suffixes=(".pth", ".safetensors"))

    if any_cache_miss:
        return None
    return total_bytes
