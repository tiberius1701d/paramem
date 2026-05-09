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

from paramem.server.vram_predict import predict_stt_bytes, predict_tts_bytes
from paramem.utils.config import AdapterConfig

logger = logging.getLogger(__name__)

# ── Module-level constants ──────────────────────────────────────────────────

# Fixed overhead for CUDA memory fragmentation and runtime bookkeeping.
_SAFETY_MARGIN_BYTES: int = 256 * 1024 * 1024  # 256 MiB

# KV-cache / activation headroom default (1 GiB) — conservative code-side
# minimum. Production yaml (configs/server.yaml and tests/fixtures/server.yaml)
# ships 2.0 GiB for the observed per-phase peak. vram_scope wraps each
# extraction phase with empty_cache between phases (paramem/graph/extractor.py
# and paramem/training/consolidation.py), so the reservation covers what the
# largest single phase holds at once, not the cumulative peak across phases.
# The plausibility-filter's ~1.9 GiB allocator pool is released before QA-gen
# begins. Tunable via vram.vram_cache_headroom_gib.
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


def estimate_stt_bytes(
    stt_config,
    *,
    permanent_cloud_only: bool = False,
) -> int:
    """Estimate GPU VRAM bytes for Whisper STT under the given config.

    Thin wrapper over :func:`paramem.server.vram_predict.predict_stt_bytes`.
    Returns 0 when STT is disabled, when the resolved device is not CUDA,
    when CUDA is globally unavailable, or when the predictor returns None
    (cache miss — live gate is authoritative).

    Args:
        stt_config: :class:`paramem.server.config.STTConfig`-like object with
            ``enabled``, ``device``, ``model``, and ``compute_type`` attributes.
        permanent_cloud_only: When True, the GPU STT pair will never be loaded
            for this process lifetime. Returns 0 so the budget does not reserve
            GPU bytes that will never be allocated.

    Returns:
        Estimated STT footprint in bytes. 0 for any CPU/disabled/cache-miss case.
    """
    if not getattr(stt_config, "enabled", False):
        return 0

    device = getattr(stt_config, "device", "cpu")
    if permanent_cloud_only and device != "cpu":
        return 0
    if device not in ("cuda", "auto"):
        return 0

    result = predict_stt_bytes(stt_config, permanent_cloud_only=permanent_cloud_only)
    if result is None:
        logger.info("STT not cached; live gate is authoritative")
        return 0
    return result


def estimate_tts_bytes(
    tts_config,
    *,
    permanent_cloud_only: bool = False,
) -> int:
    """Estimate GPU VRAM bytes for all configured TTS voices.

    Thin wrapper over :func:`paramem.server.vram_predict.predict_tts_bytes`.
    Returns 0 when TTS is disabled, when all voices are CPU-bound, or when
    the predictor returns None (cache miss — live gate is authoritative).

    Args:
        tts_config: :class:`paramem.server.config.TTSConfig`-like object with
            ``enabled``, ``device``, and ``voices`` attributes.
        permanent_cloud_only: When True, the GPU TTS pair will never be loaded
            for this process lifetime. Returns 0 so the budget does not reserve
            GPU bytes that will never be allocated.

    Returns:
        Estimated TTS footprint in bytes. 0 for any CPU/disabled/cache-miss case.
    """
    if not getattr(tts_config, "enabled", False):
        return 0

    default_device = getattr(tts_config, "device", "cpu")
    if permanent_cloud_only and default_device != "cpu":
        return 0

    result = predict_tts_bytes(tts_config, permanent_cloud_only=permanent_cloud_only)
    if result is None:
        logger.info("TTS not cached; live gate is authoritative")
        return 0
    return result


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
    base_bytes: int,
    hidden_size: int,
    num_layers: int,
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
        base_bytes: Base-model GPU bytes (from predict_base_bytes or live measurement).
        hidden_size: Hidden dimension of the base model (from AutoConfig or known value).
        num_layers: Number of transformer layers (from AutoConfig or known value).
        model_id: HF model id for the breakdown label.
        quant_label: Quantization scheme label for display only.
        main_adapter_count: Always-resident main adapters (default 3).
        headroom_gib: KV cache + activation headroom (default 2 GiB).
        stt_bytes: Whisper STT footprint (0 if CPU/disabled).
        tts_bytes: TTS footprint (0 if CPU/disabled).

    Returns:
        :class:`TopologyAssessment` — pure data, no side effects.
    """
    _GiB = 2**30

    headroom_bytes = int(headroom_gib * _GiB)

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

    breakdown = _format_breakdown(
        model_id=model_id,
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


def enforce_post_load_budget(
    measured_alloc_bytes: int,
    total_memory_bytes: int,
    headroom_bytes: int,
) -> None:
    """Authoritative post-load gate.

    Raises ConfigurationError if measured_alloc_bytes > total_memory_bytes
    - headroom_bytes. The headroom is the operator-tunable
    vram.vram_cache_headroom_gib reservation for KV cache + activations.

    Args:
        measured_alloc_bytes: torch.cuda.memory_allocated(0) after model load.
        total_memory_bytes: Device hardware cap.
        headroom_bytes: Reserved bytes for KV cache and activations.

    Raises:
        ConfigurationError: If the measured allocation leaves less than
            headroom_bytes free on the device.
    """
    budget = total_memory_bytes - headroom_bytes
    if measured_alloc_bytes <= budget:
        return
    deficit = measured_alloc_bytes - budget
    raise ConfigurationError(
        f"Post-load VRAM gate failed: measured {measured_alloc_bytes / 2**30:.2f} GiB "
        f"exceeds budget {budget / 2**30:.2f} GiB "
        f"(total {total_memory_bytes / 2**30:.2f} GiB "
        f"− headroom {headroom_bytes / 2**30:.2f} GiB). "
        f"Deficit {deficit / 2**30:.2f} GiB. "
        f"Reduce rank, max_interim_count, or increase vram.vram_cache_headroom_gib."
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
        model_id: Full HuggingFace model ID string.
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

    sep = "─" * 65

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
