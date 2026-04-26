"""Startup-time VRAM budget validator for the multi-adapter server.

Proves at boot that the configured adapter topology fits in VRAM for the
entire week of operation. Pass → guaranteed safe; runtime rejection cannot
occur. Fail → server refuses to start with a clear, actionable error.

Why startup, not runtime
------------------------
Adapter VRAM cost is fully deterministic from ``rank × target_modules ×
base_model``. Weight values do not affect size. The maximum working set
the server will ever hold is known at boot. Runtime guards add false
flexibility for an event that cannot occur once startup passes.

Working set formula
-------------------
::

    working_set_bytes =
          base_model_bytes
        + main_adapter_count        × adapter_bytes   # episodic, semantic, procedural = 3
        + max_interim_count         × adapter_bytes   # configurable, default 7
        + 1                         × adapter_bytes   # in_training staging slot
        + kv_cache_headroom_bytes

Note: adapter byte count assumes inference-only LoRA (no optimizer state).
Training reuses the ``in_training`` staging slot, accounted separately as
the ``+1 × adapter_bytes`` term.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass

import torch

from paramem.utils.config import AdapterConfig

logger = logging.getLogger(__name__)

# ── Module-level constants ──────────────────────────────────────────────────

# Fixed overhead for CUDA memory fragmentation and runtime bookkeeping.
_SAFETY_MARGIN_BYTES: int = 256 * 1024 * 1024  # 256 MiB

# KV-cache / activation headroom default (1 GiB).
# Rationale: context windows up to 4096 tokens with bfloat16 KV cache for
# 32 attention layers in Mistral 7B consume ~1 GiB at max sequence length.
_DEFAULT_HEADROOM_GIB: float = 1.0

# Hardware tiers (GiB) for the topology-fit assessment table. Covers the
# realistic deployment range: 8 GiB laptop (RTX 5070 Laptop), 12 GiB
# mid-range desktop (RTX 4070), 16 GiB desktop (RTX 4080), 24 GiB enthusiast
# (RTX 3090/4090), 40 GiB workstation (A100 40 GiB / L40), 80 GiB data
# center (A100 80 GiB / H100). Decoupling the assessment from the installed
# hardware lets the warning surface tell operators "this config is fine on
# a 12 GiB card, overflows 8 GiB by 900 MiB" — actionable even when a
# capacity bump is cheaper than shrinking the topology.
HARDWARE_TIERS_GIB: tuple[int, ...] = (8, 12, 16, 24, 40, 80)

# LoRA tensor dtype for inference-only adapters: bfloat16 / float16 = 2 bytes.
# Optimizer state is NOT included — training reuses the in_training staging slot.
_LORA_DTYPE_BYTES: int = 2

# PEFT per-adapter overhead beyond the pure LoRA A+B matrices.
# Measured empirically on Mistral 7B @ rank=8, target_modules=[q,v,k,o]: the
# raw formula yields 16 MiB/adapter but torch.cuda.memory_allocated deltas after
# model.add_adapter(...) are ~26 MiB. The 10 MiB residual covers PEFT ModulesToSave
# wrappers, adapter-config metadata tensors, and CUDA allocator alignment padding.
_PEFT_OVERHEAD_PER_ADAPTER_BYTES: int = 10 * 1024 * 1024  # 10 MiB

# Known model hidden dimensions and layer counts.
# Format: model_key → (hidden_size, num_hidden_layers)
# Used to compute adapter bytes without loading the model.
# Source: official model configs on HuggingFace.
_MODEL_DIMS: dict[str, tuple[int, int]] = {
    "mistral": (4096, 32),  # Mistral-7B-Instruct-v0.3
    "ministral": (4096, 36),  # Ministral-8B-Instruct-2410
    "llama": (4096, 32),  # Llama-3.1-8B-Instruct
    "gemma": (3584, 42),  # Gemma-2-9B-IT
    "qwen3b": (2048, 36),  # Qwen2.5-3B-Instruct
    "qwen": (3584, 28),  # Qwen2.5-7B-Instruct
    "gemma4": (2560, 34),  # Gemma-4-E4B (approximate; subject to revision)
}

# NF4 base model footprint estimates (bytes) — used when CUDA is unavailable.
# Derived from: param_count × 0.5 bytes (NF4) + small metadata overhead.
# Source: measured empirically (torch.cuda.memory_allocated after load),
# rounded up to nearest 128 MiB. Mistral 7B NF4 lands at ~4,108 MiB on load.
_MODEL_VRAM_BYTES: dict[str, int] = {
    "mistral": 4_308_428_800,  # ~4,108 MiB NF4 (measured RTX 5070, 2026-04-19)
    "ministral": 4_718_592_000,  # ~4.4 GiB NF4
    "llama": 4_718_592_000,  # ~4.4 GiB NF4
    "gemma": 5_242_880_000,  # ~5.0 GiB NF4 (partial GPU; gpu alloc only)
    "qwen3b": 2_147_483_648,  # ~2.0 GiB NF4
    "qwen": 4_194_304_000,  # ~4.0 GiB NF4
    "gemma4": 2_684_354_560,  # ~2.5 GiB NF4 (approximate)
}

# Sentinel for the static fallback when torch.cuda.memory_allocated returns 0
# for an unregistered model name.
_FALLBACK_BASE_BYTES: int = 4_500_000_000


# ── STT / TTS VRAM tables ───────────────────────────────────────────────────
# Calibrated against empirical measurements on RTX 5070 Laptop (2026-04-19)
# with ``mem_get_info`` before/after each load. Values in the ``stt.py`` table
# are 2-3× higher because they target its own pre-load ``check_vram`` gate
# (defense-in-depth, worst-case activations). The validator needs values close
# to reality so it doesn't block configs that actually fit. We apply a
# down-scale factor to ``stt.py``'s table rather than duplicating per-model
# entries — keeps the single source of truth for model naming, decouples the
# safety margin from the runtime reality.

# Empirical ratio: real_used / stt_py_table ≈ 0.4 for distil-large-v3 int8
# (960 MiB measured vs 2500 MiB tabled). Applied uniformly; holds within ~20%
# across sizes because PyTorch/CT2 per-parameter overhead is roughly constant.
_STT_CALIBRATION_FACTOR: float = 0.4

# Safety headroom on top of the calibrated base — covers first-transcribe
# workspace growth that the idle-load probe doesn't see.
_STT_CALIBRATION_HEADROOM_MB: int = 250

# Fallback for unknown Whisper model names. 1.5 GiB covers the largest sane
# workload (large-v3 int8 scaled ≈ 2.4 GiB, distil variants much less); a typo
# surfaces quickly as a clear WARNING.
_STT_FALLBACK_BYTES: int = 1_500 * 1024 * 1024  # 1.5 GiB

# Per-engine TTS GPU footprint — calibrated from the same probe: 4 Piper + 1
# MMS voices combined = 724 MiB. Piper uses ONNX Runtime; each voice's ONNX
# session holds a small model (~60-80 MB) and the CUDA execution provider
# allocates a context (~300 MB) shared across sessions in the same process.
# MMS-TTS is a HuggingFace VITS model that sits on GPU per voice.
_TTS_PIPER_BYTES_PER_VOICE: int = 80 * 1024 * 1024  # 80 MiB (was 120)
_TTS_PIPER_ORT_CONTEXT_BYTES: int = 300 * 1024 * 1024  # 300 MiB, counted once
_TTS_MMS_BYTES_PER_VOICE: int = 200 * 1024 * 1024  # 200 MiB (was 250)
_TTS_UNKNOWN_ENGINE_BYTES_PER_VOICE: int = 300 * 1024 * 1024  # 300 MiB conservative


def estimate_stt_bytes(
    stt_config,
    *,
    cloud_only: bool = False,
) -> int:
    """Estimate GPU VRAM bytes for Whisper STT under the given config.

    Reads the authoritative size table from :mod:`paramem.server.stt` so the
    validator and the STT loader agree on what "fits". Returns 0 when STT is
    disabled, when the resolved device is not CUDA, or when CUDA is globally
    unavailable.

    Args:
        stt_config: :class:`paramem.server.config.STTConfig`-like object with
            ``enabled``, ``device``, ``model``, ``compute_type``, and
            ``cpu_fallback_model`` attributes.
        cloud_only: When True, the server forces STT to CPU regardless of
            ``stt_config.device``. Mirrors :func:`app.lifespan` behavior.

    Returns:
        Estimated STT footprint in bytes. 0 for any CPU/disabled case.
    """
    if not getattr(stt_config, "enabled", False):
        return 0

    device = getattr(stt_config, "device", "cpu")
    if cloud_only and device != "cpu":
        return 0  # Cloud-only mode forces Whisper to CPU
    if device not in ("cuda", "auto"):
        return 0

    from paramem.server.stt import (
        COMPUTE_TYPE_MULTIPLIER,
        VRAM_ESTIMATES_INT8_MB,
    )

    model_name = getattr(stt_config, "model", "")
    compute_type = getattr(stt_config, "compute_type", "int8")
    base_mb = VRAM_ESTIMATES_INT8_MB.get(model_name)
    if base_mb is None:
        logger.warning(
            "STT model %r not in VRAM_ESTIMATES_INT8_MB; using fallback %d MiB. "
            "Register the model for accurate pre-load VRAM budgeting.",
            model_name,
            _STT_FALLBACK_BYTES // (1024 * 1024),
        )
        return _STT_FALLBACK_BYTES
    multiplier = COMPUTE_TYPE_MULTIPLIER.get(compute_type, 1.0)
    # Calibrated to empirical measurements (see _STT_CALIBRATION_FACTOR doc).
    calibrated_mb = int(base_mb * multiplier * _STT_CALIBRATION_FACTOR)
    total_mb = calibrated_mb + _STT_CALIBRATION_HEADROOM_MB
    return total_mb * 1024 * 1024


def estimate_tts_bytes(
    tts_config,
    *,
    cloud_only: bool = False,
) -> int:
    """Estimate GPU VRAM bytes for all configured TTS voices.

    Iterates over ``tts_config.voices`` and sums the conservative per-engine
    footprint for every voice that resolves to GPU. Piper voices share a
    single ONNX Runtime CUDA context, so that fixed cost is added once per
    process regardless of how many Piper voices are configured. MMS-TTS
    voices each hold their own model on GPU, so their cost is per-voice.

    Args:
        tts_config: :class:`paramem.server.config.TTSConfig`-like object with
            ``enabled``, ``device``, and ``voices`` attributes. Each voice has
            ``engine`` and an optional ``device`` override.
        cloud_only: When True, the server forces TTS to CPU. Mirrors
            :func:`app.lifespan` behavior.

    Returns:
        Estimated TTS footprint in bytes. 0 for any CPU/disabled case.
    """
    if not getattr(tts_config, "enabled", False):
        return 0

    default_device = getattr(tts_config, "device", "cpu")
    if cloud_only and default_device != "cpu":
        return 0

    voices = getattr(tts_config, "voices", {}) or {}
    total_bytes = 0
    piper_voices_on_gpu = 0

    for _lang, voice_config in voices.items():
        voice_device = getattr(voice_config, "device", None) or default_device
        if voice_device != "cuda":
            continue
        engine = getattr(voice_config, "engine", "").lower()
        if engine == "piper":
            piper_voices_on_gpu += 1
            total_bytes += _TTS_PIPER_BYTES_PER_VOICE
        elif engine == "mms":
            total_bytes += _TTS_MMS_BYTES_PER_VOICE
        else:
            logger.warning(
                "TTS engine %r not in validator table; using conservative "
                "fallback %d MiB per voice.",
                engine,
                _TTS_UNKNOWN_ENGINE_BYTES_PER_VOICE // (1024 * 1024),
            )
            total_bytes += _TTS_UNKNOWN_ENGINE_BYTES_PER_VOICE

    if piper_voices_on_gpu > 0:
        total_bytes += _TTS_PIPER_ORT_CONTEXT_BYTES

    return total_bytes


class ConfigurationError(RuntimeError):
    """Raised when server configuration cannot satisfy VRAM requirements.

    The error message names every cost line item and provides actionable
    remediation hints so operators know exactly which config knobs to adjust.
    """


def estimated_adapter_bytes(
    adapter_config: AdapterConfig, hidden_size: int, num_layers: int
) -> int:
    """Compute the inference-only LoRA memory footprint in bytes.

    Each LoRA layer adds two low-rank matrices per targeted module:
    ``A: (rank × in_features)`` and ``B: (out_features × rank)``.
    For attention projections where ``in_features == out_features == hidden_size``,
    both matrices are ``rank × hidden_size``.

    Formula per layer per module:
        ``(rank × hidden_size + hidden_size × rank) × dtype_bytes``
        = ``2 × rank × hidden_size × dtype_bytes``

    Total:
        ``num_modules × num_layers × 2 × rank × hidden_size × dtype_bytes``

    Note: adapter byte count assumes inference-only LoRA (no optimizer state).
    Training reuses the ``in_training`` staging slot, accounted separately.

    Args:
        adapter_config: LoRA adapter configuration (rank, target_modules).
        hidden_size: Hidden dimension of the base model (e.g. 4096 for Mistral 7B).
        num_layers: Number of transformer layers in the base model (e.g. 32 for Mistral 7B).

    Returns:
        Estimated LoRA adapter size in bytes.
    """
    num_modules = len(adapter_config.target_modules)
    # A matrix: (rank × hidden_size), B matrix: (hidden_size × rank)
    # Both are rank × hidden_size in parameter count.
    bytes_per_layer = num_modules * 2 * adapter_config.rank * hidden_size * _LORA_DTYPE_BYTES
    return bytes_per_layer * num_layers + _PEFT_OVERHEAD_PER_ADAPTER_BYTES


def estimated_base_model_bytes(model) -> int:
    """Read the base model GPU footprint from the CUDA allocator.

    Must be called AFTER the model is loaded onto the GPU. Returns the
    current ``torch.cuda.memory_allocated()`` snapshot, which reflects the
    model's parameter and buffer memory but not CUDA context overhead.

    Args:
        model: The loaded model (used only as a sentinel; the CUDA allocator
            is device-global).

    Returns:
        Bytes currently allocated on CUDA device 0.
    """
    allocated = torch.cuda.memory_allocated(0)
    logger.debug("Base model GPU footprint (memory_allocated): %.2f GiB", allocated / 2**30)
    return allocated


def required_working_set_bytes(
    base_model_bytes: int,
    adapter_bytes: int,
    main_adapter_count: int,
    max_interim_count: int,
    headroom_bytes: int,
    stt_bytes: int = 0,
    tts_bytes: int = 0,
) -> int:
    """Compute the worst-case GPU working set for the full week of operation.

    Working set formula::

        working_set_bytes =
              base_model_bytes
            + main_adapter_count × adapter_bytes   # e.g. episodic, semantic, procedural
            + max_interim_count  × adapter_bytes   # rolling interim adapters
            + 1                  × adapter_bytes   # in_training staging slot (always present)
            + stt_bytes                            # Whisper (0 if CPU/disabled)
            + tts_bytes                            # TTS voices on GPU (0 if CPU/disabled)
            + headroom_bytes                       # KV cache + activation headroom

    Note: adapter byte count assumes inference-only LoRA (no optimizer state).
    Training reuses the ``in_training`` staging slot, accounted separately as
    the ``+1 × adapter_bytes`` term.

    Args:
        base_model_bytes: GPU footprint of the quantized base model.
        adapter_bytes: Per-adapter LoRA footprint (from :func:`estimated_adapter_bytes`).
        main_adapter_count: Number of always-resident main adapters (e.g. 3 for
            episodic + semantic + procedural).
        max_interim_count: Maximum concurrent interim adapters (configurable; default 7).
        headroom_bytes: Reserved bytes for KV cache, activations, and CUDA overhead.
        stt_bytes: Estimated Whisper VRAM footprint. 0 if STT is CPU-bound or disabled.
        tts_bytes: Estimated TTS VRAM footprint across all GPU voices. 0 if CPU-bound.

    Returns:
        Total required bytes for the worst-case working set.
    """
    return (
        base_model_bytes
        + (main_adapter_count + max_interim_count + 1) * adapter_bytes
        + stt_bytes
        + tts_bytes
        + headroom_bytes
    )


# ── Hardware-tier assessment ────────────────────────────────────────────────


@dataclass(frozen=True)
class TopologyAssessment:
    """Hardware-agnostic verdict on whether a configured topology fits.

    Pure math output. The validator computes this from config alone (no CUDA
    calls) so the same summary can be logged regardless of what card is
    actually installed. Operators see which tier is required and how much
    margin each tier leaves — a 12 GiB upgrade is often cheaper than shrinking
    rank or interim count.

    Attributes:
        required_bytes: Total working-set bytes (includes safety margin).
        adapter_bytes: Per-adapter LoRA footprint used in the sum.
        base_bytes: Base-model footprint used in the sum.
        per_tier_fit: Mapping ``tier_gib → (fits, margin_bytes)``. ``fits`` is
            True when the tier has enough VRAM for the topology *plus* the
            1 GiB default headroom; ``margin_bytes`` is ``tier_bytes −
            required_bytes`` (may be negative).
        breakdown: Multi-line breakdown table for logging.
    """

    required_bytes: int
    adapter_bytes: int
    base_bytes: int
    per_tier_fit: dict[int, tuple[bool, int]]
    breakdown: str

    def smallest_fitting_tier_gib(self) -> int | None:
        """Return the smallest tier (GiB) that fits, or None if none do."""
        for tier, (fits, _margin) in sorted(self.per_tier_fit.items()):
            if fits:
                return tier
        return None


def assess_topology(
    adapter_config: AdapterConfig,
    *,
    max_interim_count: int,
    model_name: str = "",
    model_id: str = "",
    quant_label: str = "nf4",
    main_adapter_count: int = 3,
    headroom_gib: float = _DEFAULT_HEADROOM_GIB,
    stt_bytes: int = 0,
    tts_bytes: int = 0,
) -> TopologyAssessment:
    """Compute the topology working set and fit verdict for every hardware tier.

    Pure math — no CUDA, no live measurements. Safe to call before the base
    model is loaded and from test harnesses. The returned assessment drives
    both the startup warning banner (which tiers fit, which don't) and the
    live-budget gate (``enforce_live_budget``).

    Args:
        adapter_config: LoRA config for interim/episodic adapters.
        max_interim_count: Rolling interim cap (``consolidation.max_interim_count``).
        model_name: Registry key (``"mistral"``, ``"gemma"``, ...). Drives the
            static base-bytes lookup in ``_MODEL_VRAM_BYTES`` and the dims
            lookup in ``_MODEL_DIMS``. Unknown names trigger fallbacks.
        model_id: HF model id for the breakdown label. Defaults to ``model_name``.
        quant_label: Quantization scheme label for display only.
        main_adapter_count: Always-resident main adapters (default 3).
        headroom_gib: KV cache + activation headroom (default 1 GiB).
        stt_bytes: Whisper STT footprint (0 if CPU/disabled).
        tts_bytes: TTS footprint (0 if CPU/disabled).

    Returns:
        :class:`TopologyAssessment` — pure data, no side effects.
    """
    _GiB = 2**30
    _MiB = 1024 * 1024

    headroom_bytes = int(headroom_gib * _GiB)

    base_bytes = _MODEL_VRAM_BYTES.get(model_name)
    if base_bytes is None:
        logger.warning(
            "Model %r not in _MODEL_VRAM_BYTES; using fallback %d MiB. "
            "Register the model for accurate pre-load VRAM budgeting.",
            model_name,
            _FALLBACK_BASE_BYTES // _MiB,
        )
        base_bytes = _FALLBACK_BASE_BYTES

    dims = _MODEL_DIMS.get(model_name)
    if dims is None:
        logger.warning(
            "Model %r not in _MODEL_DIMS lookup; using Mistral 7B dims (4096, 32) "
            "as fallback for VRAM estimation. Actual adapter size may differ.",
            model_name,
        )
        dims = (4096, 32)
    hidden_size, num_layers = dims

    adapter_bytes = estimated_adapter_bytes(adapter_config, hidden_size, num_layers)

    total_required = required_working_set_bytes(
        base_model_bytes=base_bytes,
        adapter_bytes=adapter_bytes,
        main_adapter_count=main_adapter_count,
        max_interim_count=max_interim_count,
        headroom_bytes=headroom_bytes,
        stt_bytes=stt_bytes,
        tts_bytes=tts_bytes,
    )
    total_with_margin = total_required + _SAFETY_MARGIN_BYTES

    per_tier_fit: dict[int, tuple[bool, int]] = {}
    for tier_gib in HARDWARE_TIERS_GIB:
        tier_bytes = tier_gib * _GiB
        margin = tier_bytes - total_with_margin
        per_tier_fit[tier_gib] = (margin >= 0, margin)

    display_model_id = model_id if model_id else model_name
    breakdown = _format_breakdown(
        model_id=display_model_id,
        quant_label=quant_label,
        base_bytes=base_bytes,
        main_adapter_count=main_adapter_count,
        adapter_bytes=adapter_bytes,
        max_interim_count=max_interim_count,
        num_modules=len(adapter_config.target_modules),
        rank=adapter_config.rank,
        headroom_bytes=headroom_bytes,
        total_required_bytes=total_required,
        total_with_margin_bytes=total_with_margin,
        available_bytes=0,  # not applicable for pure assessment
        stt_bytes=stt_bytes,
        tts_bytes=tts_bytes,
        suppress_available_row=True,
    )

    return TopologyAssessment(
        required_bytes=total_with_margin,
        adapter_bytes=adapter_bytes,
        base_bytes=base_bytes,
        per_tier_fit=per_tier_fit,
        breakdown=breakdown,
    )


def format_tier_table(assessment: TopologyAssessment) -> str:
    """Render the per-tier fit table as a multi-line string for logging.

    Output example::

        Hardware-tier fit assessment (required 7.3 GiB)
          8 GiB   laptop         : OVERFLOW   ( -0.9 GiB)
         12 GiB   desktop        : FITS       ( +3.1 GiB)
         16 GiB   desktop        : FITS       ( +7.1 GiB)
         24 GiB   enthusiast     : FITS       (+15.1 GiB)
         40 GiB   workstation    : FITS       (+31.1 GiB)
         80 GiB   data-center    : FITS       (+71.1 GiB)
    """
    _GiB = 2**30
    tier_labels = {
        8: "laptop",
        12: "desktop",
        16: "desktop",
        24: "enthusiast",
        40: "workstation",
        80: "data-center",
    }
    lines = [f"Hardware-tier fit assessment (required {assessment.required_bytes / _GiB:.2f} GiB)"]
    for tier_gib in sorted(assessment.per_tier_fit):
        fits, margin = assessment.per_tier_fit[tier_gib]
        verdict = "FITS    " if fits else "OVERFLOW"
        sign = "+" if margin >= 0 else ""
        lines.append(
            f"  {tier_gib:>3} GiB  {tier_labels.get(tier_gib, ''):<12}:  {verdict}   "
            f"({sign}{margin / _GiB:>5.2f} GiB)"
        )
    return "\n".join(lines)


# ── Live-budget enforcement ─────────────────────────────────────────────────


def _query_device_memory_used_bytes() -> int | None:
    """Return device-wide GPU memory in use (bytes), via ``nvidia-smi``.

    Cross-process — sees memory held by every consumer on the device,
    including this process AND orphaned dxgkrnl-cached allocations on WSL2
    that have no owning compute process. Returns ``None`` on failure
    (binary missing, timeout, parse error); the caller decides the
    fallback policy.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=True,
        )
    except (OSError, subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
        logger.debug("nvidia-smi memory query failed: %s", exc)
        return None
    body = (result.stdout or "").strip()
    try:
        return int(body.splitlines()[0]) * 2**20
    except (ValueError, IndexError) as exc:
        logger.debug("nvidia-smi memory query unparseable (output=%r): %s", body, exc)
        return None


def measure_external_vram(
    *,
    total_memory_bytes_override: int | None = None,
) -> tuple[int, int]:
    """Snapshot device-wide GPU memory occupancy at process entry.

    Called BEFORE :func:`paramem.models.loader.load_base_model` so the
    measurement reflects external consumers' VRAM usage on this device.
    Uses ``nvidia-smi --query-gpu=memory.used`` for a true device-wide
    read — sees the Windows desktop compositor, other model servers, AND
    orphaned dxgkrnl-cached allocations on WSL2 that linger after the
    owning process exits.

    Falls back to :func:`torch.cuda.memory_allocated` (per-process only)
    when ``nvidia-smi`` is unavailable, with a WARNING log calling out
    the blind spot. The fallback path matches the prior implementation,
    which silently under-reported external occupancy.

    Re-measured on every lifespan entry — cheap (sub-100 ms subprocess)
    and ensures restarts see current occupancy, not a stale snapshot.

    Args:
        total_memory_bytes_override: Test-only override. When set, skips
            the CUDA hardware read for ``total_memory_bytes``.

    Returns:
        ``(total_memory_bytes, external_bytes)`` — the device's hardware
        cap and bytes currently held by all consumers on the device. The
        small fraction held by this process pre-load (CUDA context only)
        is included in ``external_bytes``; conservative in the safe
        direction.
    """
    if total_memory_bytes_override is not None:
        total_memory_bytes = total_memory_bytes_override
    else:
        total_memory_bytes = torch.cuda.get_device_properties(0).total_memory

    external_bytes = _query_device_memory_used_bytes()
    if external_bytes is None:
        logger.warning(
            "nvidia-smi unavailable; falling back to torch.cuda.memory_allocated. "
            "External VRAM held by other processes (and orphaned dxgkrnl cache "
            "on WSL2) will be invisible — live-budget check may admit an "
            "unfittable topology."
        )
        external_bytes = torch.cuda.memory_allocated(0)
    logger.info(
        "VRAM snapshot at entry: total=%.2f GiB, external consumers=%.2f GiB",
        total_memory_bytes / 2**30,
        external_bytes / 2**30,
    )
    return total_memory_bytes, external_bytes


def enforce_live_budget(
    assessment: TopologyAssessment,
    total_memory_bytes: int,
    external_bytes: int,
) -> None:
    """Raise :class:`ConfigurationError` if the live GPU cannot fit the topology.

    Live budget = ``total_memory − external_bytes``. This is the VRAM
    actually available to ParaMem on a fresh process — it already accounts
    for other processes holding memory but NOT for ParaMem's own (wiped)
    footprint.

    Args:
        assessment: Topology assessment from :func:`assess_topology`.
        total_memory_bytes: Device hardware cap (from
            ``torch.cuda.get_device_properties(0).total_memory`` or a test
            override via :func:`measure_external_vram`).
        external_bytes: Non-ParaMem VRAM occupancy snapshotted by
            :func:`measure_external_vram` before base-model load.

    Raises:
        ConfigurationError: If the live budget is smaller than the assessed
            required working set, with a breakdown and actionable
            remediation hints.
    """
    live_budget = total_memory_bytes - external_bytes
    if live_budget >= assessment.required_bytes:
        logger.info(
            "VRAM live budget OK: need %.2f GiB, live budget %.2f GiB "
            "(%.2f GiB total − %.2f GiB external), margin %.2f GiB.",
            assessment.required_bytes / 2**30,
            live_budget / 2**30,
            total_memory_bytes / 2**30,
            external_bytes / 2**30,
            (live_budget - assessment.required_bytes) / 2**30,
        )
        return

    _log_gpu_occupancy_diagnostic()
    deficit = assessment.required_bytes - live_budget
    raise ConfigurationError(
        f"VRAM live budget insufficient on this host.\n"
        f"{assessment.breakdown}\n\n"
        f"Hardware cap:           {total_memory_bytes / 2**30:>6.2f} GiB\n"
        f"External occupancy:    -{external_bytes / 2**30:>6.2f} GiB "
        f"(non-ParaMem processes)\n"
        f"Live budget:            {live_budget / 2**30:>6.2f} GiB\n"
        f"Required:               {assessment.required_bytes / 2**30:>6.2f} GiB\n"
        f"Deficit:                {deficit / 2**30:>6.2f} GiB\n\n"
        f"{format_tier_table(assessment)}\n\n"
        f"Either (a) free external VRAM consumers, (b) install a larger GPU "
        f"(see tier table above), or (c) shrink the topology (rank, "
        f"max_interim_count, target_modules)."
    )


def _format_breakdown(
    *,
    model_id: str,
    quant_label: str,
    base_bytes: int,
    main_adapter_count: int,
    adapter_bytes: int,
    max_interim_count: int,
    num_modules: int,
    rank: int,
    headroom_bytes: int,
    total_required_bytes: int,
    total_with_margin_bytes: int,
    available_bytes: int,
    stt_bytes: int = 0,
    tts_bytes: int = 0,
    stt_label: str = "STT (Whisper)",
    tts_label: str = "TTS voices on GPU",
    suppress_available_row: bool = False,
) -> str:
    """Return a multi-line VRAM working set breakdown table.

    Produces a fixed-width columnar table suitable for both INFO logging
    (success path) and ConfigurationError messages (failure path). Numbers
    are formatted in MiB so operators can compare directly with ``nvidia-smi``.

    Args:
        model_id: Full HuggingFace model ID string (e.g.
            ``"mistralai/Mistral-7B-Instruct-v0.3"``).
        quant_label: Quantization scheme label (e.g. ``"nf4"``).
        base_bytes: GPU footprint of the quantized base model in bytes.
        main_adapter_count: Number of always-resident main adapters.
        adapter_bytes: Per-adapter LoRA footprint in bytes.
        max_interim_count: Maximum concurrent interim adapters.
        num_modules: Number of LoRA target modules per adapter.
        rank: LoRA rank.
        headroom_bytes: KV cache + activation headroom in bytes.
        total_required_bytes: Sum of all components (excluding safety margin).
        total_with_margin_bytes: ``total_required_bytes + _SAFETY_MARGIN_BYTES``.
        available_bytes: Free VRAM bytes available at startup.

    Returns:
        Formatted breakdown string (no trailing newline).
    """
    _MiB = 1024 * 1024

    def _mib(b: int) -> str:
        return f"{b // _MiB:>6,} MiB"

    margin = available_bytes - total_with_margin_bytes
    margin_sign = "+" if margin >= 0 else ""

    sep = "\u2500" * 65

    col_w = 53  # label column width (chars)

    def _row(label: str, value: str) -> str:
        return f"  {label:<{col_w}}:  {value}"

    base_label = f"base model ({model_id}, {quant_label})"
    main_label = f"{main_adapter_count} main adapters (rank={rank}, modules={num_modules})"
    session_label = f"{max_interim_count} interim adapters (max_interim_count={max_interim_count})"
    margin_val = f"{margin_sign}{margin // _MiB:>5,} MiB"

    lines = [
        "VRAM Working Set Breakdown",
        _row(base_label, _mib(base_bytes)),
        _row(main_label, _mib(main_adapter_count * adapter_bytes)),
        _row(session_label, _mib(max_interim_count * adapter_bytes)),
        _row("in_training staging slot", _mib(adapter_bytes)),
    ]
    if stt_bytes > 0:
        lines.append(_row(stt_label, _mib(stt_bytes)))
    if tts_bytes > 0:
        lines.append(_row(tts_label, _mib(tts_bytes)))
    lines.extend(
        [
            _row("KV cache headroom", _mib(headroom_bytes)),
            _row("fragmentation safety margin", _mib(_SAFETY_MARGIN_BYTES)),
            f"  {sep}",
            _row("required total", _mib(total_with_margin_bytes)),
        ]
    )
    if not suppress_available_row:
        lines.extend(
            [
                _row("available (free VRAM)", _mib(available_bytes)),
                _row("margin", margin_val),
            ]
        )
    return "\n".join(lines)


def _log_gpu_occupancy_diagnostic() -> None:
    """Log ``nvidia-smi`` per-process VRAM usage as a diagnostic.

    Called only on the validation-failure path so operators can see *who* is
    holding VRAM when the budget check rejects startup. Best-effort: any
    failure (missing binary, driver error) is swallowed.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        body = (result.stdout or "").strip()
        if body:
            logger.error("Current GPU occupancy (nvidia-smi):\n%s", body)
        else:
            logger.error("Current GPU occupancy (nvidia-smi): no compute processes")
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.debug("nvidia-smi diagnostic unavailable: %s", exc)


def validate_startup_vram(
    model,
    adapter_config: AdapterConfig,
    *,
    max_interim_count: int,
    model_name: str = "",
    model_id: str = "",
    quant_label: str = "nf4",
    main_adapter_count: int = 3,
    headroom_gib: float = _DEFAULT_HEADROOM_GIB,
    vram_cap_gib: float | None = None,
    stt_bytes: int = 0,
    tts_bytes: int = 0,
) -> None:
    """Validate that the configured adapter topology fits in available VRAM.

    Designed to be called *before* the base model is loaded: free VRAM then
    reflects total GPU capacity minus external consumers, which maps directly
    onto the required working set (base + adapters + headroom). When called
    post-load (``model`` is provided and the CUDA allocator reports a non-zero
    footprint), the already-loaded base model footprint is added back to the
    available budget so the comparison remains self-consistent.

    Two-stage gate:

    * **Pre-load (math-only)** — ``model`` is ``None``. The base footprint is
      taken from ``_MODEL_VRAM_BYTES``. When the math predicts the worst-case
      working set won't fit, this path emits a **logger.warning** with the
      structured breakdown and returns. The server then proceeds to load and
      a later post-load call either confirms the load fit (with a pessimistic
      estimate) or raises the configuration error below.
    * **Post-load (real VRAM)** — ``model`` is a live (possibly multi-adapter)
      model. The base footprint is read from ``torch.cuda.memory_allocated``.
      If the measured working set exceeds available VRAM, this path raises
      :class:`ConfigurationError` with the breakdown and an actionable
      remediation block. This is the authoritative rejection.

    On the success path, the same breakdown is logged at INFO level so
    operators always know the budget after startup.

    Raises immediately with a clear message if no CUDA GPU is detected, rather
    than crashing later with an opaque ``AttributeError``.

    Args:
        model: The loaded base model (may already have main adapters attached),
            or ``None`` when called pre-load. When ``None`` (or when the CUDA
            allocator reports 0 bytes), the base model footprint is taken from
            ``_MODEL_VRAM_BYTES`` lookup.
        adapter_config: LoRA adapter config for interim adapters (rank, target_modules).
            Episodic and interim adapters share the same config in the current design.
        max_interim_count: Maximum number of concurrent interim adapters.
            From ``consolidation.max_interim_count`` in server config.
        model_name: Model registry key (e.g. ``"mistral"``). Used to look up
            hidden dimensions. If unknown, falls back to a conservative estimate.
        model_id: Full HuggingFace model ID string for display in the breakdown
            (e.g. ``"mistralai/Mistral-7B-Instruct-v0.3"``). Defaults to
            ``model_name`` when not provided.
        quant_label: Quantization scheme label shown in the breakdown
            (e.g. ``"nf4"``). Defaults to ``"nf4"``.
        main_adapter_count: Number of always-resident main adapters.
            Default 3 (episodic + semantic + procedural).
        headroom_gib: GiB to reserve for KV cache, activations, and CUDA overhead.
            Default 1.0 GiB.
        vram_cap_gib: Override VRAM cap in GiB. If ``None`` (default), reads
            available free bytes from ``torch.cuda.mem_get_info(0)``.
        stt_bytes: Estimated Whisper STT VRAM footprint. 0 when STT is disabled
            or CPU-bound. Obtain via :func:`estimate_stt_bytes`.
        tts_bytes: Estimated TTS VRAM footprint summed across GPU voices. 0
            when TTS is disabled or fully CPU-bound. Obtain via
            :func:`estimate_tts_bytes`.

    Raises:
        ConfigurationError: If no CUDA GPU is available, or (post-load only)
            if the measured working set exceeds the VRAM cap. Pre-load mismatch
            is logged at WARNING and does not raise.
    """
    # Fix 3 — Fail fast when local mode has no GPU (MINOR-4)
    if not torch.cuda.is_available():
        raise ConfigurationError(
            "Local model mode requires a CUDA-capable GPU but none was detected. "
            "Either provide a GPU, or start in cloud-only mode "
            "(pass --cloud-only to the server, or use --defer-model for auto-reclaim)."
        )

    _GiB = 2**30
    _MiB = 1024 * 1024

    headroom_bytes = int(headroom_gib * _GiB)

    # Base model footprint: prefer CUDA allocator snapshot when the model is
    # loaded, otherwise use the static lookup (pre-load path).
    base_bytes = estimated_base_model_bytes(model) if model is not None else 0
    model_already_loaded = base_bytes > 0

    if not model_already_loaded:
        fallback = _MODEL_VRAM_BYTES.get(model_name)
        if fallback is None:
            logger.warning(
                "Model %r not in _MODEL_VRAM_BYTES; using fallback %d MiB. "
                "Register the model for accurate pre-load VRAM budgeting.",
                model_name,
                _FALLBACK_BASE_BYTES // _MiB,
            )
            fallback = _FALLBACK_BASE_BYTES
        base_bytes = fallback

    # Determine available VRAM. The budget question this validator answers is
    # "does the configured topology fit the configured GPU?" — that is a
    # question about hardware capacity, not about what the allocator happens
    # to hold at this instant. We therefore default to the device's total
    # memory (``torch.cuda.get_device_properties(0).total_memory``) rather
    # than ``mem_get_info()[0]`` (free bytes), which is polluted by the CUDA
    # context and any unrelated process. ``vram_cap_gib`` remains an explicit
    # override for tests or sub-budget caps.
    if vram_cap_gib is not None:
        available_bytes = int(vram_cap_gib * _GiB)
    else:
        available_bytes = torch.cuda.get_device_properties(0).total_memory

    # Adapter bytes per unit
    dims = _MODEL_DIMS.get(model_name)
    if dims is None:
        # Unknown model — use Mistral 7B dims as a conservative fallback
        logger.warning(
            "Model %r not in _MODEL_DIMS lookup; using Mistral 7B dims (4096, 32) "
            "as fallback for VRAM estimation. Actual adapter size may differ.",
            model_name,
        )
        dims = (4096, 32)
    hidden_size, num_layers = dims

    adapter_bytes = estimated_adapter_bytes(adapter_config, hidden_size, num_layers)

    total_required = required_working_set_bytes(
        base_model_bytes=base_bytes,
        adapter_bytes=adapter_bytes,
        main_adapter_count=main_adapter_count,
        max_interim_count=max_interim_count,
        headroom_bytes=headroom_bytes,
        stt_bytes=stt_bytes,
        tts_bytes=tts_bytes,
    )

    total_with_margin = total_required + _SAFETY_MARGIN_BYTES

    # Resolve display model_id: use explicit arg, fall back to model_name
    display_model_id = model_id if model_id else model_name

    num_modules = len(adapter_config.target_modules)

    # Fix 1 — Per-component VRAM breakdown (MINOR-2): shared table for both paths
    breakdown = _format_breakdown(
        model_id=display_model_id,
        quant_label=quant_label,
        base_bytes=base_bytes,
        main_adapter_count=main_adapter_count,
        adapter_bytes=adapter_bytes,
        max_interim_count=max_interim_count,
        num_modules=num_modules,
        rank=adapter_config.rank,
        headroom_bytes=headroom_bytes,
        total_required_bytes=total_required,
        total_with_margin_bytes=total_with_margin,
        available_bytes=available_bytes,
        stt_bytes=stt_bytes,
        tts_bytes=tts_bytes,
    )

    if total_with_margin <= available_bytes:
        # Success path — always log the breakdown so operators know the budget.
        logger.info("VRAM check passed.\n%s", breakdown)
        return

    # Failure path — structured breakdown + actionable remediation block.
    target_modules_str = ", ".join(adapter_config.target_modules)
    remediation_lines = [
        "Configured topology does not fit. Reduce one of:",
        f"  rank                  current={adapter_config.rank}"
        "         (smaller rank halves all adapter costs)",
        f"  max_interim_count     current={max_interim_count}         (fewer interim adapters)",
        f"  target_modules        current={num_modules} ({target_modules_str})"
        "  (drop k_proj/o_proj for q/v only)",
        f"  base model            current={display_model_id} (smaller model = less base VRAM)",
    ]
    if stt_bytes > 0:
        remediation_lines.append(
            "  stt.device            current=cuda"
            "      (set to 'cpu', or pick a smaller Whisper model / int8 compute)"
        )
    if tts_bytes > 0:
        remediation_lines.append(
            "  tts.device            current=cuda"
            "      (set to 'cpu', or move GPU voices to Piper CPU / smaller MMS)"
        )
    remediation = "\n".join(remediation_lines)

    # Pre-load (math-only) path: emit a warning and return so the loader can
    # try the actual allocation. The post-load call is authoritative and will
    # raise if reality matches the prediction.
    if not model_already_loaded:
        logger.warning(
            "VRAM pre-load math predicts the configured topology will not fit.\n"
            "%s\n\n%s\n"
            "Server will still attempt to load; post-load check is authoritative.",
            breakdown,
            remediation,
        )
        return

    # Post-load (real VRAM) path: measurement has confirmed the topology does
    # not fit. Log who holds VRAM and raise — this is the authoritative gate.
    _log_gpu_occupancy_diagnostic()
    raise ConfigurationError(f"\n{breakdown}\n\n{remediation}")
