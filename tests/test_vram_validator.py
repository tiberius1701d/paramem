"""Pure unit tests for the boot-time VRAM topology assessment + post-load gate.

No GPU required — all CUDA calls are mocked. Tests cover:
- estimated_adapter_bytes (pure math)
- required_working_set_bytes (pure math)
- assess_topology (predicted-byte injection, no internal lookup)
- check_post_load_budget (pure math gate, returns str|None)
- format_baseline_fit
- estimate_stt_bytes / estimate_tts_bytes wrappers (predict mock)
- Lifespan call-order integration guard
- Default headroom constants
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from paramem.server.vram_validator import (
    _DEFAULT_HEADROOM_GIB,
    assess_topology,
    check_post_load_budget,
    estimated_adapter_bytes,
    format_baseline_fit,
    required_working_set_bytes,
)
from paramem.utils.config import AdapterConfig

# Mistral / Gemma / Qwen all ship compute_dtype="bfloat16" → 2 B per LoRA element.
_BF16_BYTES = 2
# PEFT residual per adapter (matches VramConfig.peft_overhead_per_adapter_mib default).
_PEFT_OVERHEAD_BYTES = 10 * 1024 * 1024
# Default deployment baseline (matches VramConfig.baseline_vram_gib default).
_BASELINE_GIB = 8
# Calibration constants mirroring VramConfig defaults.
_STT_WORKSPACE_FACTOR = 1.1
_TTS_PIPER_ORT_CONTEXT_BYTES = 300 * 1024 * 1024

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_GiB = 2**30

# Mistral 7B dims used throughout: hidden=4096, 32 layers
_HIDDEN = 4096
_LAYERS = 32

# Realistic production adapter config (attention-only, rank 8)
_MISTRAL_ADAPTER = AdapterConfig(
    rank=8,
    alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    dropout=0.05,
)

# Procedural adapter config (attention + MLP, 7 modules) — matches
# configs/server.yaml adapters.procedural.target_modules.
_PROCEDURAL_ADAPTER = AdapterConfig(
    rank=8,
    alpha=16,
    target_modules=[
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    dropout=0.05,
)

# Real production 3-tier topology (episodic, semantic, procedural) — matches
# configs/server.yaml with all three main tiers enabled.
_MAIN_ADAPTER_CONFIGS = [_MISTRAL_ADAPTER, _MISTRAL_ADAPTER, _PROCEDURAL_ADAPTER]

# Mistral 7B NF4 base model footprint (conservative estimate)
_BASE_MODEL_BYTES = 4_000_000_000  # ~3.8 GiB

_TOTAL_VRAM_BYTES = int(8 * _GiB)


def _adapter_bytes(adapter_config: AdapterConfig = _MISTRAL_ADAPTER) -> int:
    """Compute expected adapter bytes for Mistral 7B dims (bfloat16 LoRA)."""
    return estimated_adapter_bytes(
        adapter_config, _HIDDEN, _LAYERS, _BF16_BYTES, _PEFT_OVERHEAD_BYTES
    )


# ---------------------------------------------------------------------------
# estimated_adapter_bytes pure arithmetic check
# ---------------------------------------------------------------------------


def test_estimated_adapter_bytes_mistral_rank8_four_modules_bf16():
    """Spot-check the adapter bytes formula for Mistral 7B rank-8 attention-only, bf16.

    Formula: num_modules × num_layers × 2 × rank × hidden × dtype_bytes
           = 4 × 32 × 2 × 8 × 4096 × 2
    """
    expected = 4 * 32 * 2 * 8 * 4096 * _BF16_BYTES + _PEFT_OVERHEAD_BYTES
    result = estimated_adapter_bytes(_MISTRAL_ADAPTER, 4096, 32, _BF16_BYTES, _PEFT_OVERHEAD_BYTES)
    assert result == expected, (
        f"estimated_adapter_bytes mismatch: expected {expected}, got {result}"
    )


def test_estimated_adapter_bytes_scales_with_dtype():
    """fp32 LoRA tensors must double the adapter byte estimate vs bf16.

    Guards the silent-bug trap that motivated deriving lora_dtype_bytes from
    config.model_config.compute_dtype: any code path that hardcodes 2 will
    under-predict by 2× under fp32.
    """
    bf16 = estimated_adapter_bytes(
        _MISTRAL_ADAPTER, 4096, 32, dtype_bytes=2, peft_overhead_bytes=_PEFT_OVERHEAD_BYTES
    )
    fp32 = estimated_adapter_bytes(
        _MISTRAL_ADAPTER, 4096, 32, dtype_bytes=4, peft_overhead_bytes=_PEFT_OVERHEAD_BYTES
    )
    # PEFT overhead is constant; only the per-tensor portion doubles.
    bf16_tensor_bytes = bf16 - _PEFT_OVERHEAD_BYTES
    fp32_tensor_bytes = fp32 - _PEFT_OVERHEAD_BYTES
    assert fp32_tensor_bytes == 2 * bf16_tensor_bytes


def test_estimated_adapter_bytes_scales_with_peft_overhead():
    """Changing the PEFT-overhead knob must change the result by exactly that delta.

    Guards the new config knob ``vram.peft_overhead_per_adapter_mib`` so a yaml
    tweak propagates through assess_topology unchanged.
    """
    small = estimated_adapter_bytes(
        _MISTRAL_ADAPTER, 4096, 32, dtype_bytes=2, peft_overhead_bytes=5 * 1024 * 1024
    )
    large = estimated_adapter_bytes(
        _MISTRAL_ADAPTER, 4096, 32, dtype_bytes=2, peft_overhead_bytes=20 * 1024 * 1024
    )
    assert large - small == 15 * 1024 * 1024


# ---------------------------------------------------------------------------
# required_working_set_bytes — +1 staging slot
# ---------------------------------------------------------------------------


def test_math_includes_in_training_slot():
    """The +1 in_training staging slot must be counted in the working set."""
    adapter_bytes = _adapter_bytes()

    with_staging = required_working_set_bytes(
        base_model_bytes=_BASE_MODEL_BYTES,
        adapter_bytes=adapter_bytes,
        main_adapter_bytes_total=3 * adapter_bytes,
        backup_adapter_bytes_total=0,
        staging_adapter_bytes=adapter_bytes,
        max_interim_count=7,
        headroom_bytes=int(1.0 * _GiB),
    )

    expected_total_adapters = (3 + 7 + 1) * adapter_bytes
    expected = _BASE_MODEL_BYTES + expected_total_adapters + int(1.0 * _GiB)

    assert with_staging == expected, (
        f"required_working_set_bytes should include +1 for in_training staging slot. "
        f"Expected {expected}, got {with_staging}. "
        f"Difference of {abs(with_staging - expected)} bytes = "
        f"{abs(with_staging - expected) / adapter_bytes:.1f} adapter units."
    )

    without_staging_manual = _BASE_MODEL_BYTES + (3 + 7) * adapter_bytes + int(1.0 * _GiB)
    assert with_staging - without_staging_manual == adapter_bytes, (
        "The +1 in_training staging slot must add exactly one adapter_bytes to the total. "
        f"Expected diff={adapter_bytes}, got {with_staging - without_staging_manual}"
    )


# ---------------------------------------------------------------------------
# assess_topology — takes base_bytes / hidden_size / num_layers as required args
# ---------------------------------------------------------------------------


def test_assess_topology_returns_assessment():
    """assess_topology with injected base_bytes produces a valid TopologyAssessment."""
    from paramem.server.vram_validator import TopologyAssessment

    result = assess_topology(
        _MISTRAL_ADAPTER,
        main_adapter_configs=_MAIN_ADAPTER_CONFIGS,
        max_interim_count=7,
        base_bytes=_BASE_MODEL_BYTES,
        hidden_size=_HIDDEN,
        num_layers=_LAYERS,
        lora_dtype_bytes=_BF16_BYTES,
        peft_overhead_bytes=_PEFT_OVERHEAD_BYTES,
        baseline_vram_gib=_BASELINE_GIB,
        model_id="mistralai/Mistral-7B-Instruct-v0.3",
    )
    assert isinstance(result, TopologyAssessment)
    assert result.base_bytes == _BASE_MODEL_BYTES
    assert result.required_bytes > 0
    assert result.baseline_bytes == _BASELINE_GIB * _GiB


def test_assess_topology_too_many_interims_overflows_baseline():
    """max_interim_count=300 should overflow the 8 GiB baseline."""
    result = assess_topology(
        _MISTRAL_ADAPTER,
        main_adapter_configs=_MAIN_ADAPTER_CONFIGS,
        max_interim_count=300,
        base_bytes=_BASE_MODEL_BYTES,
        hidden_size=_HIDDEN,
        num_layers=_LAYERS,
        lora_dtype_bytes=_BF16_BYTES,
        peft_overhead_bytes=_PEFT_OVERHEAD_BYTES,
        baseline_vram_gib=_BASELINE_GIB,
    )
    assert not result.fits_baseline, "300 interims must overflow 8 GiB baseline"
    assert result.margin_bytes < 0


def test_assess_topology_realistic_config_fits_baseline():
    """Mistral 7B NF4 + rank-8 + 4 modules + 7 interims + 3 mains fits 8 GiB baseline."""
    result = assess_topology(
        _MISTRAL_ADAPTER,
        main_adapter_configs=_MAIN_ADAPTER_CONFIGS,
        max_interim_count=7,
        base_bytes=_BASE_MODEL_BYTES,
        hidden_size=_HIDDEN,
        num_layers=_LAYERS,
        lora_dtype_bytes=_BF16_BYTES,
        peft_overhead_bytes=_PEFT_OVERHEAD_BYTES,
        baseline_vram_gib=_BASELINE_GIB,
    )
    assert result.fits_baseline, "Realistic Mistral 7B config must fit 8 GiB"
    assert result.margin_bytes >= 0


def test_assess_topology_breakdown_contains_required_strings():
    """Breakdown must contain model_id, base model, adapters, headroom."""
    result = assess_topology(
        _MISTRAL_ADAPTER,
        main_adapter_configs=_MAIN_ADAPTER_CONFIGS,
        max_interim_count=7,
        base_bytes=_BASE_MODEL_BYTES,
        hidden_size=_HIDDEN,
        num_layers=_LAYERS,
        lora_dtype_bytes=_BF16_BYTES,
        peft_overhead_bytes=_PEFT_OVERHEAD_BYTES,
        baseline_vram_gib=_BASELINE_GIB,
        model_id="test/model",
        quant_label="nf4",
    )
    for substr in [
        "VRAM Working Set Breakdown",
        "base model",
        "main adapters",
        "interim adapters",
        "in_training staging slot",
        "transient full-fold backup reserve",
        "KV cache headroom",
        "fragmentation safety margin",
        "required total",
    ]:
        assert substr in result.breakdown, f"Breakdown missing '{substr}'"


# ---------------------------------------------------------------------------
# check_post_load_budget
# ---------------------------------------------------------------------------


def test_check_post_load_budget_returns_none_when_under_budget():
    """Returns None when allocation is below total − headroom."""
    total = 8 * _GiB
    headroom = 2 * _GiB
    allocated = 4 * _GiB  # well under 8 − 2 = 6 GiB
    assert check_post_load_budget(allocated, total, headroom) is None


def test_check_post_load_budget_returns_reason_when_over_budget():
    """Returns formatted reason string when allocation exceeds total − headroom."""
    total = 8 * _GiB
    headroom = 2 * _GiB
    allocated = 7 * _GiB  # exceeds 6 GiB budget
    reason = check_post_load_budget(allocated, total, headroom)
    assert reason is not None
    assert "Post-load VRAM gate failed" in reason
    assert "headroom" in reason.lower() or "vram_cache_headroom_gib" in reason


def test_check_post_load_budget_exact_boundary_returns_none():
    """Exactly at budget boundary must pass (<=)."""
    total = 8 * _GiB
    headroom = 2 * _GiB
    allocated = 6 * _GiB  # exactly at total − headroom
    assert check_post_load_budget(allocated, total, headroom) is None


# ---------------------------------------------------------------------------
# format_baseline_fit
# ---------------------------------------------------------------------------


def test_format_baseline_fit_shows_fits_when_within_baseline():
    from paramem.server.vram_validator import TopologyAssessment

    assessment = TopologyAssessment(
        required_bytes=int(7.3 * _GiB),
        adapter_bytes=int(0.02 * _GiB),
        base_bytes=int(4.0 * _GiB),
        baseline_bytes=8 * _GiB,
        fits_baseline=True,
        margin_bytes=int(0.7 * _GiB),
        breakdown="stub",
    )
    line = format_baseline_fit(assessment)
    assert "FITS" in line
    assert "OVERFLOW" not in line
    assert "8 GiB target" in line


def test_format_baseline_fit_shows_overflow_when_over_baseline():
    from paramem.server.vram_validator import TopologyAssessment

    assessment = TopologyAssessment(
        required_bytes=int(9.2 * _GiB),
        adapter_bytes=int(0.02 * _GiB),
        base_bytes=int(4.0 * _GiB),
        baseline_bytes=8 * _GiB,
        fits_baseline=False,
        margin_bytes=-int(1.2 * _GiB),
        breakdown="stub",
    )
    line = format_baseline_fit(assessment)
    assert "OVERFLOW" in line
    assert "8 GiB target" in line


# ---------------------------------------------------------------------------
# estimate_stt_bytes / estimate_tts_bytes wrappers
# ---------------------------------------------------------------------------


class _FakeSTTConfig:
    def __init__(
        self,
        *,
        enabled: bool = True,
        device: str = "cuda",
        model: str = "distil-large-v3",
        compute_type: str = "int8",
    ):
        self.enabled = enabled
        self.device = device
        self.model = model
        self.compute_type = compute_type


class _FakeTTSVoice:
    def __init__(self, *, engine: str, model: str = "", device: str | None = None):
        self.engine = engine
        self.model = model
        self.device = device


class _FakeTTSConfig:
    def __init__(
        self,
        *,
        enabled: bool = True,
        device: str = "cuda",
        voices: dict | None = None,
    ):
        self.enabled = enabled
        self.device = device
        self.voices = voices or {}


def test_estimate_stt_bytes_zero_when_disabled():
    from paramem.server.vram_validator import estimate_stt_bytes

    assert (
        estimate_stt_bytes(_FakeSTTConfig(enabled=False), workspace_factor=_STT_WORKSPACE_FACTOR)
        == 0
    )


def test_estimate_stt_bytes_zero_when_cpu():
    from paramem.server.vram_validator import estimate_stt_bytes

    assert (
        estimate_stt_bytes(_FakeSTTConfig(device="cpu"), workspace_factor=_STT_WORKSPACE_FACTOR)
        == 0
    )


def test_estimate_stt_bytes_zero_when_permanent_cloud_only():
    from paramem.server.vram_validator import estimate_stt_bytes

    assert (
        estimate_stt_bytes(
            _FakeSTTConfig(device="cuda"),
            workspace_factor=_STT_WORKSPACE_FACTOR,
            permanent_cloud_only=True,
        )
        == 0
    )


def test_estimate_stt_bytes_returns_predictor_value_on_cache_hit():
    """When predictor returns a value, estimate_stt_bytes returns it unchanged."""
    from paramem.server.vram_validator import estimate_stt_bytes

    cfg = _FakeSTTConfig(device="cuda")
    with patch("paramem.server.vram_validator.predict_stt_bytes", return_value=500_000_000):
        result = estimate_stt_bytes(cfg, workspace_factor=_STT_WORKSPACE_FACTOR)
    assert result == 500_000_000


def test_estimate_stt_bytes_returns_zero_on_cache_miss(caplog):
    """Cache miss (predictor returns None) → 0 with INFO log."""
    from paramem.server.vram_validator import estimate_stt_bytes

    named = logging.getLogger("paramem.server.vram_validator")
    orig_propagate = named.propagate
    named.propagate = True
    caplog.set_level(logging.INFO, logger="paramem.server.vram_validator")
    named.addHandler(caplog.handler)
    try:
        cfg = _FakeSTTConfig(device="cuda")
        with patch("paramem.server.vram_validator.predict_stt_bytes", return_value=None):
            result = estimate_stt_bytes(cfg, workspace_factor=_STT_WORKSPACE_FACTOR)
    finally:
        named.removeHandler(caplog.handler)
        named.propagate = orig_propagate

    assert result == 0
    assert "STT not cached" in caplog.text or "live gate" in caplog.text.lower()


def test_estimate_tts_bytes_zero_when_disabled():
    from paramem.server.vram_validator import estimate_tts_bytes

    assert (
        estimate_tts_bytes(
            _FakeTTSConfig(enabled=False),
            piper_ort_context_bytes=_TTS_PIPER_ORT_CONTEXT_BYTES,
        )
        == 0
    )


def test_estimate_tts_bytes_zero_when_permanent_cloud_only():
    from paramem.server.vram_validator import estimate_tts_bytes

    cfg = _FakeTTSConfig(voices={"en": _FakeTTSVoice(engine="piper")})
    assert (
        estimate_tts_bytes(
            cfg,
            piper_ort_context_bytes=_TTS_PIPER_ORT_CONTEXT_BYTES,
            permanent_cloud_only=True,
        )
        == 0
    )


def test_estimate_tts_bytes_returns_predictor_value_on_cache_hit():
    from paramem.server.vram_validator import estimate_tts_bytes

    cfg = _FakeTTSConfig()
    with patch("paramem.server.vram_validator.predict_tts_bytes", return_value=400_000_000):
        result = estimate_tts_bytes(cfg, piper_ort_context_bytes=_TTS_PIPER_ORT_CONTEXT_BYTES)
    assert result == 400_000_000


def test_estimate_tts_bytes_returns_zero_on_cache_miss():
    from paramem.server.vram_validator import estimate_tts_bytes

    cfg = _FakeTTSConfig()
    with patch("paramem.server.vram_validator.predict_tts_bytes", return_value=None):
        result = estimate_tts_bytes(cfg, piper_ort_context_bytes=_TTS_PIPER_ORT_CONTEXT_BYTES)
    assert result == 0


def test_estimate_stt_bytes_returns_gpu_bytes_when_defer_model():
    """permanent_cloud_only=False (defer-model path) reserves GPU pair bytes."""
    from paramem.server.vram_validator import estimate_stt_bytes

    cfg = _FakeSTTConfig(enabled=True, device="cuda", model="distil-large-v3")
    with patch("paramem.server.vram_validator.predict_stt_bytes", return_value=960_000_000):
        result = estimate_stt_bytes(
            cfg, workspace_factor=_STT_WORKSPACE_FACTOR, permanent_cloud_only=False
        )
    assert result > 0


def test_estimate_tts_bytes_returns_gpu_bytes_when_defer_model():
    """permanent_cloud_only=False (defer-model path) reserves GPU pair bytes for TTS."""
    from paramem.server.vram_validator import estimate_tts_bytes

    cfg = _FakeTTSConfig(
        enabled=True,
        device="cuda",
        voices={"en": _FakeTTSVoice(engine="piper", model="en_US-lessac-high")},
    )
    with patch("paramem.server.vram_validator.predict_tts_bytes", return_value=380_000_000):
        result = estimate_tts_bytes(
            cfg,
            piper_ort_context_bytes=_TTS_PIPER_ORT_CONTEXT_BYTES,
            permanent_cloud_only=False,
        )
    assert result > 0


def test_estimate_stt_bytes_zero_when_explicit_cloud_only():
    """permanent_cloud_only=True (--cloud-only) returns zero bytes."""
    from paramem.server.vram_validator import estimate_stt_bytes

    cfg = _FakeSTTConfig(enabled=True, device="cuda", model="distil-large-v3")
    assert (
        estimate_stt_bytes(cfg, workspace_factor=_STT_WORKSPACE_FACTOR, permanent_cloud_only=True)
        == 0
    )


def test_estimate_tts_bytes_zero_when_explicit_cloud_only():
    """permanent_cloud_only=True (--cloud-only) returns zero bytes."""
    from paramem.server.vram_validator import estimate_tts_bytes

    cfg = _FakeTTSConfig(
        enabled=True,
        device="cuda",
        voices={"en": _FakeTTSVoice(engine="piper", model="en_US-lessac-high")},
    )
    assert (
        estimate_tts_bytes(
            cfg,
            piper_ort_context_bytes=_TTS_PIPER_ORT_CONTEXT_BYTES,
            permanent_cloud_only=True,
        )
        == 0
    )


# ---------------------------------------------------------------------------
# Lifespan call-order integration guard
# ---------------------------------------------------------------------------


def test_lifespan_runs_validator_before_load_base_model(tmp_path):
    """Integration guard for the startup lifecycle.

    Exercises the FastAPI ``lifespan`` context with ``load_base_model``,
    ``assess_topology``, and ``_wait_for_gpu_drain`` replaced by spies.
    Asserts the correct call order:

        predict_base_bytes → assess_topology → _wait_for_gpu_drain →
        load_base_model

    The authoritative reject is ``check_post_load_budget`` AFTER load
    (separate test). ``_wait_for_gpu_drain`` is the pre-load gate and
    degrades to cloud-only on timeout rather than hard-failing.
    """
    import asyncio

    from paramem.server import app as server_app
    from paramem.server.config import ServerConfig

    class _Sentinel(Exception):
        pass

    calls: list[tuple] = []

    def spy_predict(*args, **kwargs):
        calls.append(("predict_base_bytes",))
        return _BASE_MODEL_BYTES

    def spy_assess(*args, **kwargs):
        calls.append(("assess_topology",))
        from paramem.server.vram_validator import TopologyAssessment

        return TopologyAssessment(
            required_bytes=1,
            adapter_bytes=1,
            base_bytes=1,
            baseline_bytes=_BASELINE_GIB * _GiB,
            fits_baseline=True,
            margin_bytes=_BASELINE_GIB * _GiB - 1,
            breakdown="stub",
        )

    def spy_drain(needed_bytes, **kwargs):
        calls.append(("_wait_for_gpu_drain",))
        return True  # drained — proceed to load

    def spy_load_base_model(*args, **kwargs):
        calls.append(("load_base_model",))
        raise _Sentinel("short-circuit after load_base_model is invoked")

    config = ServerConfig(model_name="mistral")
    config.cloud_only = False
    from paramem.server.config import PathsConfig

    root = tmp_path / "data"
    config.paths = PathsConfig(
        data=root,
        sessions=root / "sessions",
        debug=root / "debug",
    )

    saved_state = {
        key: server_app._state.get(key) for key in ("config", "cloud_only_startup", "defer_model")
    }

    server_app._state["config"] = config
    server_app._state["cloud_only_startup"] = False
    server_app._state["defer_model"] = False

    # Change 4: after assess_topology, the lifespan calls get_device_properties +
    # mem_get_info to check for the overflow condition.  The spy_assess returns
    # required_bytes=1, so with any positive free the overflow branch won't fire.
    _total_bytes = int(8 * _GiB)
    _free_bytes = int(7 * _GiB)

    try:
        with (
            patch.object(server_app, "predict_base_bytes", spy_predict),
            patch.object(server_app, "assess_topology", spy_assess),
            patch.object(server_app, "_wait_for_gpu_drain", spy_drain),
            patch.object(server_app, "load_base_model", spy_load_base_model),
            patch.object(server_app, "_gpu_occupied", return_value=False),
            patch("paramem.server.app.torch.cuda.is_available", return_value=True),
            patch("paramem.server.app.torch.cuda.get_device_properties") as mock_props,
            patch(
                "paramem.server.app.torch.cuda.mem_get_info",
                return_value=(_free_bytes, _total_bytes),
            ),
            patch.object(server_app, "apply_process_cap", lambda **kwargs: None),
            # AutoConfig.from_pretrained is called in the base_pred branch before
            # assess_topology — stub it so this unit test has no HF network call.
            patch("transformers.AutoConfig.from_pretrained") as mock_cfg,
        ):
            mock_props.return_value.total_memory = _total_bytes
            mock_cfg.return_value = type("C", (), {"hidden_size": 4096, "num_hidden_layers": 32})()

            async def _run() -> None:
                async with server_app.lifespan(server_app.app):
                    pass

            with pytest.raises(_Sentinel):
                asyncio.run(_run())
    finally:
        for key, value in saved_state.items():
            if value is None:
                server_app._state.pop(key, None)
            else:
                server_app._state[key] = value
        server_app._state.pop("post_load_budget_warning", None)
        server_app._state.pop("topology_assessment", None)
        server_app._state.pop("usable_ceiling_bytes", None)
        server_app._state.pop("device_total_memory_bytes", None)

    # predict_base_bytes must come before assess_topology
    call_names = [c[0] for c in calls]
    assert "predict_base_bytes" in call_names, "predict_base_bytes must be called in lifespan"
    assert "assess_topology" in call_names, "assess_topology must be called in lifespan"
    pred_idx = call_names.index("predict_base_bytes")
    assess_idx = call_names.index("assess_topology")
    assert pred_idx < assess_idx, "predict_base_bytes must come before assess_topology"

    # Boot order: assess_topology → _wait_for_gpu_drain → load_base_model
    expected_tail = [
        "assess_topology",
        "_wait_for_gpu_drain",
        "load_base_model",
    ]
    tail_order = [c for c in call_names if c in expected_tail]
    assert tail_order == expected_tail, (
        f"Lifespan must invoke {expected_tail} in order; got {tail_order}"
    )


# ---------------------------------------------------------------------------
# G5 — default headroom constant and VramConfig default
# ---------------------------------------------------------------------------


def test_default_headroom_constant_is_1_gib():
    """_DEFAULT_HEADROOM_GIB is the conservative code-side minimum (production yaml ships 1.5)."""
    assert _DEFAULT_HEADROOM_GIB == 1.0


def test_default_headroom_is_1_gib_via_config():
    """VramConfig.vram_cache_headroom_gib code default is 1.0 (production yaml overrides to 1.5)."""
    from paramem.server.config import VramConfig

    cfg = VramConfig()
    assert cfg.vram_cache_headroom_gib == 1.0


def test_headroom_1_5_gib_fits_within_drain_time_free():
    """Regression guard: with headroom=1.5 GiB and the real Mistral 7B NF4 3-tier
    topology (episodic + semantic at rank=8/4 modules, procedural at rank=8/7
    modules, 7 interims), required_bytes must be < 6.83 GiB — the device-wide
    free measured at boot time in the live failure that prompted Change 1.

    Concrete numbers (all derived from the working-set formula):
      base ≈ 4597 MiB  (inferred from the pre-fix failure: 7187 MiB total with
                         headroom=2.0 → 7187 - 2*1024 - 256 - 11*26 = 4597 MiB)
      main adapters (real per-tier shapes): 26 + 26 + 38 = 90 MiB
      7 interims × 26 MiB (episodic-shaped) = 182 MiB
      staging slot (worst-case tier shape, procedural) = 38 MiB
      transient full-fold backup reserve = 90 MiB
      headroom = 1.5 × 1024 = 1536 MiB
      safety margin = 256 MiB
      total = 4597 + 90 + 182 + 38 + 90 + 1536 + 256 = 6789 MiB = 6.630 GiB < 6.83 GiB ✓
    """
    _MiB = 1024 * 1024

    # Inferred Mistral 7B NF4 base footprint from the pre-fix required_bytes.
    # (7187 MiB total - 256 MiB safety - 2*1024 MiB headroom - 11*26 MiB adapters)
    base_bytes = 4597 * _MiB

    result = assess_topology(
        _MISTRAL_ADAPTER,
        main_adapter_configs=_MAIN_ADAPTER_CONFIGS,
        max_interim_count=7,
        base_bytes=base_bytes,
        hidden_size=_HIDDEN,
        num_layers=_LAYERS,
        lora_dtype_bytes=_BF16_BYTES,
        peft_overhead_bytes=_PEFT_OVERHEAD_BYTES,
        baseline_vram_gib=_BASELINE_GIB,
        headroom_gib=1.5,
    )

    drain_time_free_bytes = int(6.83 * _GiB)
    assert result.required_bytes < drain_time_free_bytes, (
        f"With headroom=1.5 GiB, required_bytes should be < 6.83 GiB (drain-time free), "
        f"got {result.required_bytes / _GiB:.3f} GiB ({result.required_bytes // _MiB} MiB)"
    )
    # Concrete sanity: must be ≈ 6789 MiB ± 10 MiB (rounding from float GiB).
    expected_mib = 6789
    actual_mib = result.required_bytes // _MiB
    assert abs(actual_mib - expected_mib) <= 15, (
        f"required_bytes with headroom=1.5 GiB expected ~{expected_mib} MiB, got {actual_mib} MiB"
    )


# ---------------------------------------------------------------------------
# G3 — permanent_cloud_only semantics
# ---------------------------------------------------------------------------


def _permanent_cloud_only_from_reason(reason):
    """Mirror the lifespan logic: permanent when reason in explicit/gpu_conflict."""
    return reason in ("explicit", "gpu_conflict")


def test_lifespan_budget_includes_gpu_voice_bytes_under_defer_model():
    """cloud_only_reason='training' (--defer-model): permanent_cloud_only=False → non-zero."""
    from paramem.server.vram_validator import estimate_stt_bytes, estimate_tts_bytes

    reason = "training"
    pco = _permanent_cloud_only_from_reason(reason)
    assert pco is False

    stt_cfg = _FakeSTTConfig(enabled=True, device="cuda", model="distil-large-v3")
    tts_cfg = _FakeTTSConfig(
        enabled=True, device="cuda", voices={"en": _FakeTTSVoice(engine="piper")}
    )
    with patch("paramem.server.vram_validator.predict_stt_bytes", return_value=960_000_000):
        assert (
            estimate_stt_bytes(
                stt_cfg,
                workspace_factor=_STT_WORKSPACE_FACTOR,
                permanent_cloud_only=pco,
            )
            > 0
        )
    with patch("paramem.server.vram_validator.predict_tts_bytes", return_value=380_000_000):
        assert (
            estimate_tts_bytes(
                tts_cfg,
                piper_ort_context_bytes=_TTS_PIPER_ORT_CONTEXT_BYTES,
                permanent_cloud_only=pco,
            )
            > 0
        )


def test_lifespan_budget_zeroes_voice_bytes_under_explicit_cloud_only():
    """cloud_only_reason='explicit' (--cloud-only): permanent_cloud_only=True → zero."""
    from paramem.server.vram_validator import estimate_stt_bytes, estimate_tts_bytes

    reason = "explicit"
    pco = _permanent_cloud_only_from_reason(reason)
    assert pco is True

    stt_cfg = _FakeSTTConfig(enabled=True, device="cuda", model="distil-large-v3")
    tts_cfg = _FakeTTSConfig(
        enabled=True, device="cuda", voices={"en": _FakeTTSVoice(engine="piper")}
    )
    assert (
        estimate_stt_bytes(
            stt_cfg, workspace_factor=_STT_WORKSPACE_FACTOR, permanent_cloud_only=pco
        )
        == 0
    )
    assert (
        estimate_tts_bytes(
            tts_cfg,
            piper_ort_context_bytes=_TTS_PIPER_ORT_CONTEXT_BYTES,
            permanent_cloud_only=pco,
        )
        == 0
    )


def test_lifespan_budget_zeroes_voice_bytes_under_gpu_conflict():
    """cloud_only_reason='gpu_conflict': permanent_cloud_only=True → zero."""
    from paramem.server.vram_validator import estimate_stt_bytes, estimate_tts_bytes

    reason = "gpu_conflict"
    pco = _permanent_cloud_only_from_reason(reason)
    assert pco is True

    stt_cfg = _FakeSTTConfig(enabled=True, device="cuda", model="distil-large-v3")
    tts_cfg = _FakeTTSConfig(
        enabled=True, device="cuda", voices={"en": _FakeTTSVoice(engine="piper")}
    )
    assert (
        estimate_stt_bytes(
            stt_cfg, workspace_factor=_STT_WORKSPACE_FACTOR, permanent_cloud_only=pco
        )
        == 0
    )
    assert (
        estimate_tts_bytes(
            tts_cfg,
            piper_ort_context_bytes=_TTS_PIPER_ORT_CONTEXT_BYTES,
            permanent_cloud_only=pco,
        )
        == 0
    )


def test_lifespan_budget_includes_gpu_voice_bytes_under_default_local():
    """cloud_only_reason=None (default local mode): permanent_cloud_only=False → non-zero."""
    from paramem.server.vram_validator import estimate_stt_bytes, estimate_tts_bytes

    reason = None
    pco = _permanent_cloud_only_from_reason(reason)
    assert pco is False

    stt_cfg = _FakeSTTConfig(enabled=True, device="cuda", model="distil-large-v3")
    tts_cfg = _FakeTTSConfig(
        enabled=True, device="cuda", voices={"en": _FakeTTSVoice(engine="piper")}
    )
    with patch("paramem.server.vram_validator.predict_stt_bytes", return_value=960_000_000):
        assert (
            estimate_stt_bytes(
                stt_cfg,
                workspace_factor=_STT_WORKSPACE_FACTOR,
                permanent_cloud_only=pco,
            )
            > 0
        )
    with patch("paramem.server.vram_validator.predict_tts_bytes", return_value=380_000_000):
        assert (
            estimate_tts_bytes(
                tts_cfg,
                piper_ort_context_bytes=_TTS_PIPER_ORT_CONTEXT_BYTES,
                permanent_cloud_only=pco,
            )
            > 0
        )


# ---------------------------------------------------------------------------
# interim_overflow_slack in required_working_set_bytes / assess_topology
# ---------------------------------------------------------------------------


def test_required_working_set_bytes_slack_zero_is_identical_to_no_slack_formula():
    """slack=0 (default) must produce the same total as the no-slack formula.

    Without slack: (main + N + 1) * adapter_bytes.
    With slack=0:  (main + N + 0 + 1) * adapter_bytes — identical.
    """
    adapter_bytes = _adapter_bytes()
    without_slack = required_working_set_bytes(
        base_model_bytes=_BASE_MODEL_BYTES,
        adapter_bytes=adapter_bytes,
        main_adapter_bytes_total=3 * adapter_bytes,
        backup_adapter_bytes_total=0,
        staging_adapter_bytes=adapter_bytes,
        max_interim_count=7,
        headroom_bytes=int(1.0 * _GiB),
    )
    with_slack_zero = required_working_set_bytes(
        base_model_bytes=_BASE_MODEL_BYTES,
        adapter_bytes=adapter_bytes,
        main_adapter_bytes_total=3 * adapter_bytes,
        backup_adapter_bytes_total=0,
        staging_adapter_bytes=adapter_bytes,
        max_interim_count=7,
        headroom_bytes=int(1.0 * _GiB),
        interim_overflow_slack=0,
    )
    assert without_slack == with_slack_zero, (
        "interim_overflow_slack=0 must produce the same result as omitting the param"
    )


def test_required_working_set_bytes_slack_grows_by_slack_times_adapter_bytes():
    """slack > 0 must add exactly slack * adapter_bytes to the total.

    Guards that the slack term is included in the formula:
    (main + N + slack + 1) * adapter_bytes.
    """
    adapter_bytes = _adapter_bytes()
    base = required_working_set_bytes(
        base_model_bytes=_BASE_MODEL_BYTES,
        adapter_bytes=adapter_bytes,
        main_adapter_bytes_total=3 * adapter_bytes,
        backup_adapter_bytes_total=0,
        staging_adapter_bytes=adapter_bytes,
        max_interim_count=7,
        headroom_bytes=int(1.0 * _GiB),
        interim_overflow_slack=0,
    )
    with_slack_2 = required_working_set_bytes(
        base_model_bytes=_BASE_MODEL_BYTES,
        adapter_bytes=adapter_bytes,
        main_adapter_bytes_total=3 * adapter_bytes,
        backup_adapter_bytes_total=0,
        staging_adapter_bytes=adapter_bytes,
        max_interim_count=7,
        headroom_bytes=int(1.0 * _GiB),
        interim_overflow_slack=2,
    )
    assert with_slack_2 - base == 2 * adapter_bytes, (
        f"slack=2 should add exactly 2*adapter_bytes={2 * adapter_bytes} bytes; "
        f"got diff={with_slack_2 - base}"
    )


def test_assess_topology_slack_zero_matches_no_slack_call():
    """assess_topology with interim_overflow_slack=0 must match a call without the param."""
    kwargs = dict(
        main_adapter_configs=_MAIN_ADAPTER_CONFIGS,
        max_interim_count=7,
        base_bytes=_BASE_MODEL_BYTES,
        hidden_size=_HIDDEN,
        num_layers=_LAYERS,
        lora_dtype_bytes=_BF16_BYTES,
        peft_overhead_bytes=_PEFT_OVERHEAD_BYTES,
        baseline_vram_gib=_BASELINE_GIB,
    )
    without = assess_topology(_MISTRAL_ADAPTER, **kwargs)
    with_zero = assess_topology(_MISTRAL_ADAPTER, interim_overflow_slack=0, **kwargs)
    assert without.required_bytes == with_zero.required_bytes, (
        "assess_topology with slack=0 must give identical required_bytes to no-slack call"
    )


def test_assess_topology_slack_overflows_baseline():
    """A large interim_overflow_slack that exceeds the baseline must mark fits_baseline=False."""
    result = assess_topology(
        _MISTRAL_ADAPTER,
        main_adapter_configs=_MAIN_ADAPTER_CONFIGS,
        max_interim_count=7,
        interim_overflow_slack=300,
        base_bytes=_BASE_MODEL_BYTES,
        hidden_size=_HIDDEN,
        num_layers=_LAYERS,
        lora_dtype_bytes=_BF16_BYTES,
        peft_overhead_bytes=_PEFT_OVERHEAD_BYTES,
        baseline_vram_gib=_BASELINE_GIB,
    )
    assert not result.fits_baseline, "interim_overflow_slack=300 must overflow the 8 GiB baseline"
    assert result.margin_bytes < 0


# ---------------------------------------------------------------------------
# _format_breakdown interim row includes slack
# ---------------------------------------------------------------------------


def test_breakdown_slack_zero_unchanged():
    """_format_breakdown with slack=0 must produce the same interim row as before FIX-4."""
    from paramem.server.vram_validator import _format_breakdown

    adapter_bytes = _adapter_bytes()
    max_interim = 7
    result = _format_breakdown(
        model_id="test/model",
        quant_label="nf4",
        base_bytes=_BASE_MODEL_BYTES,
        main_adapter_count=3,
        adapter_bytes=adapter_bytes,
        max_interim_count=max_interim,
        rank=8,
        headroom_bytes=int(1.0 * _GiB),
        total_required_bytes=0,
        total_with_margin_bytes=0,
        available_bytes=0,
        main_adapter_bytes_total=3 * adapter_bytes,
        backup_adapter_bytes_total=3 * adapter_bytes,
        staging_adapter_bytes=adapter_bytes,
        suppress_available_row=True,
        interim_overflow_slack=0,
    )
    expected_label = f"{max_interim} interim adapters (max_interim_count={max_interim})"
    assert expected_label in result, (
        f"slack=0 breakdown must contain original label '{expected_label}'"
    )
    # The MiB value must reflect exactly max_interim * adapter_bytes.
    _MiB = 1024 * 1024
    expected_mib = f"{max_interim * adapter_bytes // _MiB:>6,} MiB"
    assert expected_mib in result, (
        f"slack=0 interim row must show {expected_mib!r} (== max_interim * adapter_bytes)"
    )


def test_breakdown_slack_nonzero_shows_expanded_count():
    """_format_breakdown with slack=2 must show (max_interim + slack) adapters
    in the interim row and the correct MiB total for that expanded count.

    Guards FIX-4: at slack>0 the itemization previously under-summed the total
    by slack*adapter_bytes.
    """
    from paramem.server.vram_validator import _format_breakdown

    adapter_bytes = _adapter_bytes()
    max_interim = 7
    slack = 2
    total_interim = max_interim + slack
    result = _format_breakdown(
        model_id="test/model",
        quant_label="nf4",
        base_bytes=_BASE_MODEL_BYTES,
        main_adapter_count=3,
        adapter_bytes=adapter_bytes,
        max_interim_count=max_interim,
        rank=8,
        headroom_bytes=int(1.0 * _GiB),
        total_required_bytes=0,
        total_with_margin_bytes=0,
        available_bytes=0,
        main_adapter_bytes_total=3 * adapter_bytes,
        backup_adapter_bytes_total=3 * adapter_bytes,
        staging_adapter_bytes=adapter_bytes,
        suppress_available_row=True,
        interim_overflow_slack=slack,
    )
    expected_label = (
        f"{total_interim} interim adapters (max_interim_count={max_interim}+{slack} overflow)"
    )
    assert expected_label in result, (
        f"slack={slack} breakdown must contain label '{expected_label}', got:\n{result}"
    )
    # The MiB value must reflect (max_interim + slack) * adapter_bytes.
    _MiB = 1024 * 1024
    expected_mib = f"{total_interim * adapter_bytes // _MiB:>6,} MiB"
    assert expected_mib in result, (
        f"slack={slack} interim row must show {expected_mib!r} "
        f"(== {total_interim} * adapter_bytes), got:\n{result}"
    )


def test_breakdown_slack_assess_topology_label_matches():
    """assess_topology with slack>0 must propagate the slack into the breakdown string."""
    slack = 3
    max_interim = 7
    result = assess_topology(
        _MISTRAL_ADAPTER,
        main_adapter_configs=_MAIN_ADAPTER_CONFIGS,
        max_interim_count=max_interim,
        interim_overflow_slack=slack,
        base_bytes=_BASE_MODEL_BYTES,
        hidden_size=_HIDDEN,
        num_layers=_LAYERS,
        lora_dtype_bytes=_BF16_BYTES,
        peft_overhead_bytes=_PEFT_OVERHEAD_BYTES,
        baseline_vram_gib=_BASELINE_GIB,
    )
    expected_label = (
        f"{max_interim + slack} interim adapters (max_interim_count={max_interim}+{slack} overflow)"
    )
    assert expected_label in result.breakdown, (
        f"assess_topology slack={slack} breakdown must contain '{expected_label}'"
    )


# ---------------------------------------------------------------------------
# Real per-tier adapter shapes + transient full-fold backup reserve
# ---------------------------------------------------------------------------


def test_estimated_adapter_bytes_four_module_is_26_mib():
    """4-module config (episodic/semantic) at Mistral 7B dims is exactly 26 MiB."""
    result = estimated_adapter_bytes(
        _MISTRAL_ADAPTER, _HIDDEN, _LAYERS, _BF16_BYTES, _PEFT_OVERHEAD_BYTES
    )
    assert result == 26 * 1024 * 1024


def test_estimated_adapter_bytes_seven_module_is_38_mib():
    """7-module config (procedural: attention + MLP) at Mistral 7B dims is exactly 38 MiB."""
    result = estimated_adapter_bytes(
        _PROCEDURAL_ADAPTER, _HIDDEN, _LAYERS, _BF16_BYTES, _PEFT_OVERHEAD_BYTES
    )
    assert result == 38 * 1024 * 1024


def test_required_working_set_bytes_real_shapes_backup_and_staging_delta_is_114_mib():
    """main_adapter_bytes_total=90 MiB (real per-tier) + backup_adapter_bytes_total=90 MiB
    + staging_adapter_bytes=38 MiB (worst-case tier shape) must exceed a
    uniform-11-adapter total (3 mains + 7 interim + 1 staging, all 26 MiB, no
    backup reserve) by exactly 12 + 90 + 12 = 114 MiB.

    Uniform: (3 main + 7 interim + 1 staging) * 26 MiB, backup=0.
    Real mains: 26 + 26 + 38 = 90 MiB (vs uniform 3*26=78 MiB) → +12 MiB.
    Backup reserve: 90 MiB (transient <tier>_backup, same per-tier shapes).
    Staging slot at worst-case (procedural) shape: 38 MiB (vs uniform 26 MiB) → +12 MiB.
    Combined delta: 12 + 90 + 12 = 114 MiB.
    """
    _MiB = 1024 * 1024
    adapter_bytes = _adapter_bytes()
    assert adapter_bytes == 26 * _MiB

    uniform_baseline = required_working_set_bytes(
        base_model_bytes=_BASE_MODEL_BYTES,
        adapter_bytes=adapter_bytes,
        main_adapter_bytes_total=3 * adapter_bytes,
        backup_adapter_bytes_total=0,
        staging_adapter_bytes=adapter_bytes,
        max_interim_count=7,
        headroom_bytes=int(1.0 * _GiB),
    )

    real_mains_bytes = 90 * _MiB
    backup_bytes = 90 * _MiB
    staging_bytes = 38 * _MiB
    with_real_shapes = required_working_set_bytes(
        base_model_bytes=_BASE_MODEL_BYTES,
        adapter_bytes=adapter_bytes,
        main_adapter_bytes_total=real_mains_bytes,
        backup_adapter_bytes_total=backup_bytes,
        staging_adapter_bytes=staging_bytes,
        max_interim_count=7,
        headroom_bytes=int(1.0 * _GiB),
    )

    delta = with_real_shapes - uniform_baseline
    assert delta == 114 * _MiB, f"expected +114 MiB delta, got {delta / _MiB} MiB"


def test_assess_topology_real_per_tier_configs_fits_baseline_at_6981_style_config():
    """assess_topology with the real 3-tier (episodic/semantic/procedural) config
    must produce the pinned 90/90/38 MiB main/backup/staging totals and fit the
    8 GiB baseline — the config this whole refactor exists to keep byte-exact."""
    result = assess_topology(
        _MISTRAL_ADAPTER,
        main_adapter_configs=_MAIN_ADAPTER_CONFIGS,
        max_interim_count=7,
        base_bytes=_BASE_MODEL_BYTES,
        hidden_size=_HIDDEN,
        num_layers=_LAYERS,
        lora_dtype_bytes=_BF16_BYTES,
        peft_overhead_bytes=_PEFT_OVERHEAD_BYTES,
        baseline_vram_gib=_BASELINE_GIB,
    )
    assert result.fits_baseline, (
        f"real per-tier 3-tier config must fit 8 GiB baseline "
        f"(margin={result.margin_bytes / _GiB:.4f} GiB)"
    )
    assert result.margin_bytes > 0


def test_assess_topology_real_per_tier_configs_sizes_staging_at_procedural_worst_case():
    """The staging term must be sized at the LARGEST per-tier shape (procedural,
    38 MiB) — the in_training slot is created fresh per training event, shaped like
    whichever tier is currently training, so the budget must model the worst case."""
    result = assess_topology(
        _MISTRAL_ADAPTER,
        main_adapter_configs=_MAIN_ADAPTER_CONFIGS,
        max_interim_count=7,
        base_bytes=_BASE_MODEL_BYTES,
        hidden_size=_HIDDEN,
        num_layers=_LAYERS,
        lora_dtype_bytes=_BF16_BYTES,
        peft_overhead_bytes=_PEFT_OVERHEAD_BYTES,
        baseline_vram_gib=_BASELINE_GIB,
    )
    _MiB = 1024 * 1024
    assert "in_training staging slot (worst-case tier shape)" in result.breakdown
    expected_staging_mib = f"{38 * _MiB // _MiB:>6,} MiB"
    assert expected_staging_mib in result.breakdown


def test_assess_topology_real_per_tier_configs_breakdown_shows_backup_reserve():
    """The breakdown must surface the transient backup reserve as its own row, not
    silently fold it into another line."""
    result = assess_topology(
        _MISTRAL_ADAPTER,
        main_adapter_configs=_MAIN_ADAPTER_CONFIGS,
        max_interim_count=7,
        base_bytes=_BASE_MODEL_BYTES,
        hidden_size=_HIDDEN,
        num_layers=_LAYERS,
        lora_dtype_bytes=_BF16_BYTES,
        peft_overhead_bytes=_PEFT_OVERHEAD_BYTES,
        baseline_vram_gib=_BASELINE_GIB,
    )
    assert "transient full-fold backup reserve" in result.breakdown
    _MiB = 1024 * 1024
    expected_backup_mib = f"{90 * _MiB // _MiB:>6,} MiB"
    assert expected_backup_mib in result.breakdown
    # The main-adapters row must reflect the real 90 MiB total.
    expected_main_mib = f"{90 * _MiB // _MiB:>6,} MiB"
    assert expected_main_mib in result.breakdown


def test_assess_topology_dropping_a_tier_reduces_bytes_by_exactly_that_tiers_shape():
    """This is the test the whole refactor exists for: a disabled tier is excluded
    from the main-adapter total, the backup reserve, AND (when it was the
    worst-case tier) the staging term — all three derived from ONE
    main_adapter_configs list, so the adapter count and the byte total can never
    disagree.

    Dropping procedural (the worst-case 38 MiB tier) from the real 3-tier config:
      main total:    90 MiB -> 52 MiB  (-38 MiB, procedural's own shape)
      backup total:  90 MiB -> 52 MiB  (-38 MiB, mirrors main)
      staging term:  38 MiB -> 26 MiB  (-12 MiB, worst case shifts to episodic/semantic)
      combined delta: 38 + 38 + 12 = 88 MiB
    """
    _MiB = 1024 * 1024
    kwargs = dict(
        max_interim_count=7,
        base_bytes=_BASE_MODEL_BYTES,
        hidden_size=_HIDDEN,
        num_layers=_LAYERS,
        lora_dtype_bytes=_BF16_BYTES,
        peft_overhead_bytes=_PEFT_OVERHEAD_BYTES,
        baseline_vram_gib=_BASELINE_GIB,
    )
    full = assess_topology(_MISTRAL_ADAPTER, main_adapter_configs=_MAIN_ADAPTER_CONFIGS, **kwargs)
    procedural_dropped = assess_topology(
        _MISTRAL_ADAPTER,
        main_adapter_configs=[_MISTRAL_ADAPTER, _MISTRAL_ADAPTER],
        **kwargs,
    )

    delta = full.required_bytes - procedural_dropped.required_bytes
    assert delta == 88 * _MiB, f"expected -88 MiB when dropping procedural, got {delta / _MiB} MiB"

    assert "3 main adapters" in full.breakdown
    assert "2 main adapters" in procedural_dropped.breakdown


def test_assess_topology_empty_main_adapter_configs_yields_zero_and_does_not_raise():
    """main_adapter_configs=[] (no enabled main tiers) must yield 0 for main/backup/
    staging bytes without raising — the empty case is a deterministic guard on
    ``max()``, not an error state."""
    result = assess_topology(
        _MISTRAL_ADAPTER,
        main_adapter_configs=[],
        max_interim_count=7,
        base_bytes=_BASE_MODEL_BYTES,
        hidden_size=_HIDDEN,
        num_layers=_LAYERS,
        lora_dtype_bytes=_BF16_BYTES,
        peft_overhead_bytes=_PEFT_OVERHEAD_BYTES,
        baseline_vram_gib=_BASELINE_GIB,
    )
    _MiB = 1024 * 1024
    assert "0 main adapters" in result.breakdown
    assert f"{0:>6,} MiB" in result.breakdown

    adapter_bytes = _adapter_bytes()
    expected_required = (
        _BASE_MODEL_BYTES
        + 0  # main_adapter_bytes_total
        + 7 * adapter_bytes  # interim ring
        + 0  # staging_adapter_bytes
        + 0  # backup_adapter_bytes_total
        + int(_DEFAULT_HEADROOM_GIB * _GiB)
        + 256 * _MiB  # _SAFETY_MARGIN_BYTES
    )
    assert result.required_bytes == expected_required
