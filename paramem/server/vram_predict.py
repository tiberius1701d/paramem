"""Cache-derived VRAM predictors for the ParaMem server.

Replaces static per-model tables with disk-cache-derived predictions.
Each predictor returns None on a cache miss — callers proceed without
the estimate and rely on the live load gate instead.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from huggingface_hub import try_to_load_from_cache
    from huggingface_hub.file_download import _CACHED_NO_EXIST
except ImportError:
    try_to_load_from_cache = None  # type: ignore[assignment]
    _CACHED_NO_EXIST = None  # type: ignore[assignment]

# Runtime GPU bytes per byte of safetensors disk weight. Disk safetensors are
# bf16/fp16 (2 bytes/param), so the factor is (runtime_bytes_per_param / 2).
# Empirical NF4: 0.55 bytes/param (4-bit + block scales) → 0.275 disk-relative;
# verified against Mistral 7B (4108 MiB measured), Gemma 2 9B (5000 MiB),
# Qwen 2.5 7B (4000 MiB) on RTX 5070 (2026-04-19, ex-_MODEL_VRAM_BYTES).
_NF4_FACTOR: float = 0.275
_INT8_FACTOR: float = 0.5  # 1 byte/param vs disk's 2 bytes/param
_FP16_FACTOR: float = 1.0  # 2 bytes/param matches disk
_FP32_FACTOR: float = 2.0  # 4 bytes/param vs disk's 2

# CT2 disk weights are typically fp16 (Systran ships fp16 model.bin for
# distil/large variants). Runtime footprint is disk × compute_type × workspace.
# Empirical: distil-large-v3 int8 = 960 MiB on RTX 5070 (vram_measure delta,
# 2026-05-09); predictor 770 MiB at disk 1.4 GiB × 0.5 × 1.1 — slightly low,
# live gate is authoritative for the actual decision.
_CT2_COMPUTE_TYPE_FACTOR: dict[str, float] = {
    "int8": 0.5,  # halves fp16 disk weights
    "int8_float16": 0.5,
    "int8_float32": 0.5,
    "int16": 0.75,
    "float16": 1.0,
    "float32": 2.0,
}
_CT2_WORKSPACE_FACTOR: float = 1.1  # 10% activation/workspace overhead on top

# Single ORT CUDA context shared across all Piper voices in the same process.
# ONNX Runtime allocates this once regardless of voice count; counted once.
_TTS_PIPER_ORT_CONTEXT_BYTES: int = 300 * 1024 * 1024  # 300 MiB


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


def predict_base_bytes(model_config) -> int | None:
    """Predict base-model GPU bytes from cached weights × quant factor.

    Reads ``model_config.model_id``, ``quantization``, ``compute_dtype``.
    Sums *.safetensors and pytorch_model-*.bin (one or the other) under the
    HF cache snapshot dir. Multiplies by _NF4_FACTOR / _INT8_FACTOR / etc.
    Returns None if the model is not cached.
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
        "nf4": _NF4_FACTOR,
        "int8": _INT8_FACTOR,
        "fp16": _FP16_FACTOR,
        "float16": _FP16_FACTOR,
        "fp32": _FP32_FACTOR,
        "float32": _FP32_FACTOR,
    }
    factor = factor_map.get(quantization, _NF4_FACTOR)
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


def predict_stt_bytes(stt_config, *, permanent_cloud_only: bool = False) -> int | None:
    """Predict GPU bytes for faster-whisper STT.

    Sums files under the CT2 cache snapshot dir, multiplies by the runtime
    compute_type factor (CT2 quantizes fp16 disk weights at load) and a
    small workspace overhead. Returns 0 for cpu/disabled/permanent_cloud_only.
    Returns None if cuda but cache miss.
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
    #   legacy:          guillaumekln/faster-whisper-{name}
    candidates = []
    if model_name.startswith("distil-"):
        candidates.append(f"Systran/faster-distil-whisper-{model_name[len('distil-') :]}")
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
    predicted = int(disk_bytes * compute_factor * _CT2_WORKSPACE_FACTOR)
    logger.debug(
        "predict_stt_bytes(%r, %r): disk=%d MiB × %.2f × %.1f = %d MiB",
        model_name,
        compute_type,
        disk_bytes >> 20,
        compute_factor,
        _CT2_WORKSPACE_FACTOR,
        predicted >> 20,
    )
    return predicted


def predict_tts_bytes(tts_config, *, permanent_cloud_only: bool = False) -> int | None:
    """Predict GPU bytes for TTS.

    Piper voices: sum *.onnx file sizes from the configured Piper data dir
    (per voice on GPU) + _TTS_PIPER_ORT_CONTEXT_BYTES if any Piper voice
    is on GPU.
    MMS voices: sum cached safetensors per voice on GPU.
    Returns 0 for cpu/disabled/permanent_cloud_only. Returns None if any
    GPU voice is uncached.
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
    piper_dir = Path(model_dir_str) if model_dir_str else Path("data/ha/tts/piper")

    total_bytes = 0
    piper_voices_on_gpu = 0
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

    if piper_voices_on_gpu > 0:
        total_bytes += _TTS_PIPER_ORT_CONTEXT_BYTES

    if any_cache_miss:
        return None
    return total_bytes
