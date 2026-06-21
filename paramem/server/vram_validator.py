"""Boot-time VRAM topology assessment + authoritative post-load gate.

Two roles, both consumed by the server lifespan:

1. :func:`assess_topology` — pure math from config alone. Produces a
   :class:`TopologyAssessment` with the worst-case working set and a
   per-hardware-tier fit verdict. Logged at boot for operator visibility;
   cached on ``_state["topology_assessment"]`` for the live-reload path's
   drain-wait pre-flight. No CUDA calls, safe before the model is loaded.
2. :func:`enforce_post_load_budget` — the authoritative reject. Runs in the
   lifespan AFTER the base model + STT/TTS are on the device, reads
   ``torch.cuda.memory_allocated(0)``, refuses startup (``sys.exit(1)``)
   when the measured allocation leaves less than the configured headroom.

Pre-load math gates were removed: the boot path uses
``_wait_for_gpu_drain`` (in ``app.py``) to wait for VRAM and degrade to
cloud-only on timeout; the live-reload path uses the same drain-wait.
``enforce_post_load_budget`` is the only check that can actually reject
the configured topology.

Working set formula (informational, used by :func:`assess_topology`)::

    working_set_bytes =
          base_model_bytes
        + main_adapter_count        × adapter_bytes   # episodic, semantic, procedural = 3
        + max_interim_count         × adapter_bytes   # configurable, default 7
        + 1                         × adapter_bytes   # in_training staging slot
        + stt_bytes + tts_bytes
        + kv_cache_headroom_bytes
        + safety_margin

Adapter bytes assume inference-only LoRA (no optimizer state); training
reuses the ``in_training`` staging slot.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

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

# PEFT per-adapter overhead is config-driven (vram.peft_overhead_per_adapter_mib)
# — it depends on target_modules + PEFT version, so a target-set change can be
# tuned in yaml. Measured ~10 MiB for [q,v,k,o] rank=8 on Mistral 7B.


def estimate_stt_bytes(
    stt_config,
    *,
    workspace_factor: float,
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
        workspace_factor: CT2 workspace overhead multiplier
            (``vram.stt_workspace_factor``).
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

    result = predict_stt_bytes(
        stt_config,
        workspace_factor=workspace_factor,
        permanent_cloud_only=permanent_cloud_only,
    )
    if result is None:
        logger.info("STT not cached; live gate is authoritative")
        return 0
    return result


def estimate_tts_bytes(
    tts_config,
    *,
    piper_ort_context_bytes: int,
    permanent_cloud_only: bool = False,
) -> int:
    """Estimate GPU VRAM bytes for all configured TTS voices.

    Thin wrapper over :func:`paramem.server.vram_predict.predict_tts_bytes`.
    Returns 0 when TTS is disabled, when all voices are CPU-bound, or when
    the predictor returns None (cache miss — live gate is authoritative).

    Args:
        tts_config: :class:`paramem.server.config.TTSConfig`-like object with
            ``enabled``, ``device``, and ``voices`` attributes.
        piper_ort_context_bytes: Shared ONNX Runtime CUDA context size in bytes
            (``vram.tts_piper_ort_context_mib`` × 1 MiB).
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

    result = predict_tts_bytes(
        tts_config,
        piper_ort_context_bytes=piper_ort_context_bytes,
        permanent_cloud_only=permanent_cloud_only,
    )
    if result is None:
        logger.info("TTS not cached; live gate is authoritative")
        return 0
    return result


def estimated_adapter_bytes(
    adapter_config: AdapterConfig,
    hidden_size: int,
    num_layers: int,
    dtype_bytes: int,
    peft_overhead_bytes: int,
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
        dtype_bytes: Bytes per LoRA tensor element. LoRA tensors inherit the
            base model's ``compute_dtype`` (PEFT + bitsandbytes contract — see
            ``paramem/models/loader.py`` where ``bnb_4bit_compute_dtype`` is set
            from ``model_config.compute_dtype``). bfloat16 / float16 → 2, float32 → 4.
        peft_overhead_bytes: Empirical per-adapter overhead beyond the pure A+B
            tensors (PEFT wrappers, metadata, allocator alignment). Config-driven
            via ``vram.peft_overhead_per_adapter_mib``.

    Returns:
        Estimated LoRA adapter size in bytes.
    """
    num_modules = len(adapter_config.target_modules)
    # A matrix: (rank × hidden_size), B matrix: (hidden_size × rank)
    # Both are rank × hidden_size in parameter count.
    bytes_per_layer = num_modules * 2 * adapter_config.rank * hidden_size * dtype_bytes
    return bytes_per_layer * num_layers + peft_overhead_bytes


def required_working_set_bytes(
    base_model_bytes: int,
    adapter_bytes: int,
    main_adapter_count: int,
    max_interim_count: int,
    headroom_bytes: int,
    stt_bytes: int = 0,
    tts_bytes: int = 0,
    interim_overflow_slack: int = 0,
) -> int:
    """Compute the worst-case GPU working set for the full week of operation.

    Working set formula::

        working_set_bytes =
              base_model_bytes
            + main_adapter_count * adapter_bytes             # main adapters
            + (max_interim_count + interim_overflow_slack)
              * adapter_bytes                                # rolling + overflow interim
            + 1 * adapter_bytes                             # in_training staging slot
            + stt_bytes                                     # Whisper (0 if CPU/disabled)
            + tts_bytes                                     # TTS voices on GPU (0 if CPU)
            + headroom_bytes                                # KV cache + activation headroom

    Note: adapter byte count assumes inference-only LoRA (no optimizer state).
    Training reuses the ``in_training`` staging slot, accounted separately as
    the ``+1 × adapter_bytes`` term.  ``interim_overflow_slack`` reserves
    capacity for overflow slots that may be minted when the ring is full but
    the full fold has not yet drained it; at 0 (default) the total is
    identical to the pre-S5 formula.

    Args:
        base_model_bytes: GPU footprint of the quantized base model.
        adapter_bytes: Per-adapter LoRA footprint (from :func:`estimated_adapter_bytes`).
        main_adapter_count: Number of always-resident main adapters (e.g. 3 for
            episodic + semantic + procedural).
        max_interim_count: Maximum concurrent interim adapters (configurable; default 7).
        headroom_bytes: Reserved bytes for KV cache, activations, and CUDA overhead.
        stt_bytes: Estimated Whisper VRAM footprint. 0 if STT is CPU-bound or disabled.
        tts_bytes: Estimated TTS VRAM footprint across all GPU voices. 0 if CPU-bound.
        interim_overflow_slack: Extra overflow adapter slots beyond ``max_interim_count``
            to reserve in the budget.  Mirrors
            ``consolidation.interim_overflow_slack`` from the server config.
            At 0 (default) no extra slots are reserved.

    Returns:
        Total required bytes for the worst-case working set.
    """
    return (
        base_model_bytes
        + (main_adapter_count + max_interim_count + interim_overflow_slack + 1) * adapter_bytes
        + stt_bytes
        + tts_bytes
        + headroom_bytes
    )


# ── Baseline-target assessment ──────────────────────────────────────────────


@dataclass(frozen=True)
class TopologyAssessment:
    """Verdict on whether a configured topology fits the deployment baseline.

    Pure math output. ``baseline_bytes`` is the operator-configured target
    GPU VRAM (``vram.baseline_vram_gib`` × 1 GiB). ``fits_baseline`` is the
    headline boolean used by the boot log and the attention populator.

    Attributes:
        required_bytes: Total working-set bytes (includes safety margin).
        adapter_bytes: Per-adapter LoRA footprint used in the sum.
        base_bytes: Base-model footprint used in the sum.
        baseline_bytes: Configured deployment baseline in bytes
            (``vram.baseline_vram_gib`` × 2**30).
        fits_baseline: ``required_bytes ≤ baseline_bytes``.
        margin_bytes: ``baseline_bytes − required_bytes`` (negative on overflow).
        breakdown: Multi-line breakdown table for logging.
    """

    required_bytes: int
    adapter_bytes: int
    base_bytes: int
    baseline_bytes: int
    fits_baseline: bool
    margin_bytes: int
    breakdown: str


def assess_topology(
    adapter_config: AdapterConfig,
    *,
    max_interim_count: int,
    base_bytes: int,
    hidden_size: int,
    num_layers: int,
    lora_dtype_bytes: int,
    peft_overhead_bytes: int,
    baseline_vram_gib: int,
    model_id: str = "",
    quant_label: str = "nf4",
    main_adapter_count: int = 3,
    headroom_gib: float = _DEFAULT_HEADROOM_GIB,
    stt_bytes: int = 0,
    tts_bytes: int = 0,
    interim_overflow_slack: int = 0,
) -> TopologyAssessment:
    """Compute the topology working set and fit verdict against the baseline.

    Pure math — no CUDA, no live measurements. Safe to call before the base
    model is loaded and from test harnesses. The returned assessment drives
    the startup banner and the ``vram_config_overflow`` attention warning;
    the authoritative reject is :func:`check_post_load_budget` after the
    model is on the device.

    Args:
        adapter_config: LoRA config for interim/episodic adapters.
        max_interim_count: Rolling interim cap (``consolidation.max_interim_count``).
        base_bytes: Base-model GPU bytes (from predict_base_bytes or live measurement).
        hidden_size: Hidden dimension of the base model (from AutoConfig or known value).
        num_layers: Number of transformer layers (from AutoConfig or known value).
        lora_dtype_bytes: Bytes per LoRA tensor element (derived from
            ``model_config.compute_dtype``).
        peft_overhead_bytes: Per-adapter PEFT residual overhead in bytes
            (``vram.peft_overhead_per_adapter_mib`` × 1 MiB).
        baseline_vram_gib: Configured deployment target in GiB
            (``vram.baseline_vram_gib``).
        model_id: HF model id for the breakdown label.
        quant_label: Quantization scheme label for display only.
        main_adapter_count: Always-resident main adapters (default 3).
        headroom_gib: KV cache + activation headroom.
        stt_bytes: Whisper STT footprint (0 if CPU/disabled).
        tts_bytes: TTS footprint (0 if CPU/disabled).
        interim_overflow_slack: Extra overflow adapter slots beyond
            ``max_interim_count`` to include in the budget.  Mirrors
            ``consolidation.interim_overflow_slack`` from the server config.
            At 0 (default) no extra slots are reserved (identical to pre-S5).

    Returns:
        :class:`TopologyAssessment` — pure data, no side effects.
    """
    _GiB = 2**30

    headroom_bytes = int(headroom_gib * _GiB)

    adapter_bytes = estimated_adapter_bytes(
        adapter_config, hidden_size, num_layers, lora_dtype_bytes, peft_overhead_bytes
    )

    total_required = required_working_set_bytes(
        base_model_bytes=base_bytes,
        adapter_bytes=adapter_bytes,
        main_adapter_count=main_adapter_count,
        max_interim_count=max_interim_count,
        headroom_bytes=headroom_bytes,
        stt_bytes=stt_bytes,
        tts_bytes=tts_bytes,
        interim_overflow_slack=interim_overflow_slack,
    )
    total_with_margin = total_required + _SAFETY_MARGIN_BYTES

    baseline_bytes = baseline_vram_gib * _GiB
    margin_bytes = baseline_bytes - total_with_margin
    fits_baseline = margin_bytes >= 0

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
        interim_overflow_slack=interim_overflow_slack,
    )

    return TopologyAssessment(
        required_bytes=total_with_margin,
        adapter_bytes=adapter_bytes,
        base_bytes=base_bytes,
        baseline_bytes=baseline_bytes,
        fits_baseline=fits_baseline,
        margin_bytes=margin_bytes,
        breakdown=breakdown,
    )


def format_baseline_fit(assessment: TopologyAssessment) -> str:
    """Render the baseline fit verdict as a one-line summary for logging.

    Output example::

        Baseline fit: 7.30 GiB required vs 8 GiB target — FITS (+0.70 GiB)
        Baseline fit: 9.20 GiB required vs 8 GiB target — OVERFLOW (-1.20 GiB)
    """
    _GiB = 2**30
    verdict = "FITS" if assessment.fits_baseline else "OVERFLOW"
    sign = "+" if assessment.margin_bytes >= 0 else ""
    return (
        f"Baseline fit: {assessment.required_bytes / _GiB:.2f} GiB required "
        f"vs {assessment.baseline_bytes // _GiB} GiB target — {verdict} "
        f"({sign}{assessment.margin_bytes / _GiB:.2f} GiB)"
    )


# ── Post-load gate ──────────────────────────────────────────────────────────


def check_post_load_budget(
    measured_alloc_bytes: int,
    total_memory_bytes: int,
    headroom_bytes: int,
) -> str | None:
    """Authoritative post-load gate — returns a reason string on overflow, else None.

    The caller decides the response (boot path: release + degrade to cloud-only;
    live-reload: degrade only). Returning a string instead of raising keeps the
    decision at the call site and removes the asymmetry between boot and reload.

    Args:
        measured_alloc_bytes: ``torch.cuda.memory_allocated(0)`` after model load.
        total_memory_bytes: Device hardware cap.
        headroom_bytes: Reserved bytes for KV cache and activations
            (``vram.vram_cache_headroom_gib`` × 1 GiB).

    Returns:
        ``None`` when ``measured_alloc_bytes ≤ total_memory_bytes − headroom_bytes``.
        Otherwise a formatted multi-line reason describing the deficit + actionable
        knobs, suitable for log + ``/status.attention``.
    """
    budget = total_memory_bytes - headroom_bytes
    if measured_alloc_bytes <= budget:
        return None
    deficit = measured_alloc_bytes - budget
    return (
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
    interim_overflow_slack: int = 0,
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
        interim_overflow_slack: Extra overflow slots beyond ``max_interim_count``
            reserved in the budget (mirrors ``consolidation.interim_overflow_slack``).
            At 0 (default) the interim row is identical to the pre-S5 output.

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
    _total_interim = max_interim_count + interim_overflow_slack
    if interim_overflow_slack > 0:
        session_label = (
            f"{_total_interim} interim adapters "
            f"(max_interim_count={max_interim_count}+{interim_overflow_slack} overflow)"
        )
    else:
        session_label = (
            f"{max_interim_count} interim adapters (max_interim_count={max_interim_count})"
        )
    margin_val = f"{margin_sign}{margin // _MiB:>5,} MiB"

    lines = [
        "VRAM Working Set Breakdown",
        _row(base_label, _mib(base_bytes)),
        _row(main_label, _mib(main_adapter_count * adapter_bytes)),
        _row(session_label, _mib(_total_interim * adapter_bytes)),
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
