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

# LoRA tensor dtype for inference-only adapters: bfloat16 / float16 = 2 bytes.
# Optimizer state is NOT included — training reuses the in_training staging slot.
_LORA_DTYPE_BYTES: int = 2

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
# Source: measured empirically, rounded conservatively up to nearest 128 MiB.
_MODEL_VRAM_BYTES: dict[str, int] = {
    "mistral": 4_000_000_000,  # ~3.8 GiB NF4
    "ministral": 4_500_000_000,  # ~4.2 GiB NF4
    "llama": 4_500_000_000,  # ~4.2 GiB NF4
    "gemma": 5_000_000_000,  # ~4.7 GiB NF4 (partial GPU; gpu alloc only)
    "qwen3b": 2_000_000_000,  # ~1.9 GiB NF4
    "qwen": 4_000_000_000,  # ~3.8 GiB NF4
    "gemma4": 2_500_000_000,  # ~2.3 GiB NF4 (approximate)
}

# Sentinel for the static fallback when torch.cuda.memory_allocated returns 0
# for an unregistered model name.
_FALLBACK_BASE_BYTES: int = 4_500_000_000


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
    return bytes_per_layer * num_layers


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
) -> int:
    """Compute the worst-case GPU working set for the full week of operation.

    Working set formula::

        working_set_bytes =
              base_model_bytes
            + main_adapter_count × adapter_bytes   # e.g. episodic, semantic, procedural
            + max_interim_count  × adapter_bytes   # rolling interim adapters
            + 1                  × adapter_bytes   # in_training staging slot (always present)
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

    Returns:
        Total required bytes for the worst-case working set.
    """
    return (
        base_model_bytes
        + (main_adapter_count + max_interim_count + 1) * adapter_bytes
        + headroom_bytes
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
        _row("KV cache headroom", _mib(headroom_bytes)),
        _row("fragmentation safety margin", _mib(_SAFETY_MARGIN_BYTES)),
        f"  {sep}",
        _row("required total", _mib(total_with_margin_bytes)),
        _row("available (free VRAM)", _mib(available_bytes)),
        _row("margin", margin_val),
    ]
    return "\n".join(lines)


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
) -> None:
    """Validate that the configured adapter topology fits in available VRAM.

    Must be called after the base model is loaded (so that
    :func:`estimated_base_model_bytes` can read the CUDA allocator).

    If the worst-case working set exceeds available VRAM, raises
    :class:`ConfigurationError` with a structured per-component breakdown and
    an actionable remediation block listing every knob the operator can reduce,
    including current values. On the success path, the same breakdown is logged
    at INFO level so operators always know the budget after startup.

    Raises immediately with a clear message if no CUDA GPU is detected, rather
    than crashing later with an opaque ``AttributeError``.

    Args:
        model: The loaded base model (may already have main adapters attached).
            Used only for the CUDA memory snapshot.
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

    Raises:
        ConfigurationError: If no CUDA GPU is available, or if the working set
            exceeds the VRAM cap.
    """
    # Fix 3 — Fail fast when local mode has no GPU (MINOR-4)
    if not torch.cuda.is_available():
        raise ConfigurationError(
            "Local model mode requires a CUDA-capable GPU but none was detected. "
            "Either provide a GPU, or start in cloud-only mode "
            "(pass --cloud-only to the server, or use --defer-model for auto-reclaim)."
        )

    _GiB = 2**30

    headroom_bytes = int(headroom_gib * _GiB)

    # Determine available VRAM
    if vram_cap_gib is not None:
        available_bytes = int(vram_cap_gib * _GiB)
    else:
        free_bytes, _total_bytes = torch.cuda.mem_get_info(0)
        available_bytes = free_bytes

    # Base model footprint: prefer CUDA allocator snapshot, fall back to lookup
    base_bytes = estimated_base_model_bytes(model)
    if base_bytes == 0:
        # CUDA allocator is empty — model may be CPU-offloaded or not yet loaded.
        # Use static lookup as conservative fallback.
        fallback = _MODEL_VRAM_BYTES.get(model_name, _FALLBACK_BASE_BYTES)
        _MiB = 1024 * 1024
        # Fix 2 — Promote silent-fallback log to WARNING (MINOR-3)
        logger.warning(
            "CUDA allocator returned 0 for model %r; using static footprint fallback "
            "of %d MiB. If this is a production deployment, register the model in "
            "`_MODEL_VRAM_BYTES`.",
            model_name,
            fallback // _MiB,
        )
        base_bytes = fallback

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
    )

    if total_with_margin <= available_bytes:
        # Success path — always log the breakdown so operators know the budget.
        logger.info("VRAM check passed.\n%s", breakdown)
        return

    # Failure path — structured breakdown + actionable remediation block.
    target_modules_str = ", ".join(adapter_config.target_modules)
    remediation = (
        "Configured topology does not fit. Reduce one of:\n"
        f"  rank                  current={adapter_config.rank}"
        "         (smaller rank halves all adapter costs)\n"
        f"  max_interim_count     current={max_interim_count}"
        "         (fewer interim adapters)\n"
        f"  target_modules        current={num_modules} ({target_modules_str})"
        "  (drop k_proj/o_proj for q/v only)\n"
        f"  base model            current={display_model_id} (smaller model = less base VRAM)"
    )

    raise ConfigurationError(f"\n{breakdown}\n\n{remediation}")
