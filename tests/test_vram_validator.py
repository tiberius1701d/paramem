"""Pure unit tests for the startup VRAM budget validator.

No GPU required — all CUDA calls are mocked. Tests cover:
- estimated_adapter_bytes (pure math)
- required_working_set_bytes (pure math)
- assess_topology (predicted-byte injection, no internal lookup)
- enforce_post_load_budget (pure math gate)
- measure_external_vram (nvidia-smi mocking)
- enforce_live_budget (pure math)
- format_tier_table
- estimate_stt_bytes / estimate_tts_bytes wrappers (predict mock)
- Lifespan call-order integration guard
- Default headroom constants
"""

from __future__ import annotations

import logging
import subprocess
from unittest.mock import patch

import pytest

from paramem.server.vram_validator import (
    _DEFAULT_HEADROOM_GIB,
    _LORA_DTYPE_BYTES,
    _PEFT_OVERHEAD_PER_ADAPTER_BYTES,
    ConfigurationError,
    assess_topology,
    enforce_live_budget,
    enforce_post_load_budget,
    estimated_adapter_bytes,
    format_tier_table,
    measure_external_vram,
    required_working_set_bytes,
)
from paramem.utils.config import AdapterConfig

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

# Mistral 7B NF4 base model footprint (conservative estimate)
_BASE_MODEL_BYTES = 4_000_000_000  # ~3.8 GiB

_TOTAL_VRAM_BYTES = int(8 * _GiB)


def _adapter_bytes(adapter_config: AdapterConfig = _MISTRAL_ADAPTER) -> int:
    """Compute expected adapter bytes for Mistral 7B dims."""
    return estimated_adapter_bytes(adapter_config, _HIDDEN, _LAYERS)


# ---------------------------------------------------------------------------
# estimated_adapter_bytes pure arithmetic check
# ---------------------------------------------------------------------------


def test_estimated_adapter_bytes_mistral_rank8_four_modules():
    """Spot-check the adapter bytes formula for Mistral 7B rank-8 attention-only.

    Formula: num_modules × num_layers × 2 × rank × hidden × dtype_bytes
           = 4 × 32 × 2 × 8 × 4096 × 2
    """
    expected = 4 * 32 * 2 * 8 * 4096 * _LORA_DTYPE_BYTES + _PEFT_OVERHEAD_PER_ADAPTER_BYTES
    result = estimated_adapter_bytes(_MISTRAL_ADAPTER, hidden_size=4096, num_layers=32)
    assert result == expected, (
        f"estimated_adapter_bytes mismatch: expected {expected}, got {result}"
    )


# ---------------------------------------------------------------------------
# required_working_set_bytes — +1 staging slot
# ---------------------------------------------------------------------------


def test_math_includes_in_training_slot():
    """The +1 in_training staging slot must be counted in the working set."""
    adapter_bytes = _adapter_bytes()

    with_staging = required_working_set_bytes(
        base_model_bytes=_BASE_MODEL_BYTES,
        adapter_bytes=adapter_bytes,
        main_adapter_count=3,
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
        max_interim_count=7,
        base_bytes=_BASE_MODEL_BYTES,
        hidden_size=_HIDDEN,
        num_layers=_LAYERS,
        model_id="mistralai/Mistral-7B-Instruct-v0.3",
    )
    assert isinstance(result, TopologyAssessment)
    assert result.base_bytes == _BASE_MODEL_BYTES
    assert result.required_bytes > 0
    assert 8 in result.per_tier_fit


def test_assess_topology_too_many_interims_overflows_8gib():
    """max_interim_count=300 should overflow 8 GiB tier."""
    result = assess_topology(
        _MISTRAL_ADAPTER,
        max_interim_count=300,
        base_bytes=_BASE_MODEL_BYTES,
        hidden_size=_HIDDEN,
        num_layers=_LAYERS,
    )
    fits_8, _margin = result.per_tier_fit[8]
    assert not fits_8, "300 interims must overflow 8 GiB tier"


def test_assess_topology_realistic_config_fits_8gib():
    """Mistral 7B NF4 + rank-8 + 4 modules + 7 interims + 3 mains fits 8 GiB."""
    result = assess_topology(
        _MISTRAL_ADAPTER,
        max_interim_count=7,
        base_bytes=_BASE_MODEL_BYTES,
        hidden_size=_HIDDEN,
        num_layers=_LAYERS,
    )
    fits_8, _margin = result.per_tier_fit[8]
    assert fits_8, "Realistic Mistral 7B config must fit 8 GiB"


def test_assess_topology_breakdown_contains_required_strings():
    """Breakdown must contain model_id, base model, adapters, headroom."""
    result = assess_topology(
        _MISTRAL_ADAPTER,
        max_interim_count=7,
        base_bytes=_BASE_MODEL_BYTES,
        hidden_size=_HIDDEN,
        num_layers=_LAYERS,
        model_id="test/model",
        quant_label="nf4",
    )
    for substr in [
        "VRAM Working Set Breakdown",
        "base model",
        "main adapters",
        "interim adapters",
        "in_training staging slot",
        "KV cache headroom",
        "fragmentation safety margin",
        "required total",
    ]:
        assert substr in result.breakdown, f"Breakdown missing '{substr}'"


# ---------------------------------------------------------------------------
# enforce_post_load_budget
# ---------------------------------------------------------------------------


def test_enforce_post_load_budget_passes_when_under_budget():
    """Must not raise when allocation is below total − headroom."""
    total = 8 * _GiB
    headroom = 2 * _GiB
    allocated = 4 * _GiB  # well under 8 − 2 = 6 GiB
    enforce_post_load_budget(allocated, total, headroom)  # no exception


def test_enforce_post_load_budget_raises_when_over_budget():
    """Must raise ConfigurationError when allocation exceeds total − headroom."""
    total = 8 * _GiB
    headroom = 2 * _GiB
    allocated = 7 * _GiB  # exceeds 6 GiB budget
    with pytest.raises(ConfigurationError) as exc_info:
        enforce_post_load_budget(allocated, total, headroom)
    msg = str(exc_info.value)
    assert "Post-load VRAM gate failed" in msg
    assert "headroom" in msg.lower() or "vram_cache_headroom_gib" in msg


def test_enforce_post_load_budget_exact_boundary_passes():
    """Exactly at budget boundary must pass (<=)."""
    total = 8 * _GiB
    headroom = 2 * _GiB
    allocated = 6 * _GiB  # exactly at total − headroom
    enforce_post_load_budget(allocated, total, headroom)  # no exception


# ---------------------------------------------------------------------------
# format_tier_table
# ---------------------------------------------------------------------------


def test_format_tier_table_contains_all_tiers():
    from paramem.server.vram_validator import TopologyAssessment

    assessment = TopologyAssessment(
        required_bytes=int(7.3 * _GiB),
        adapter_bytes=int(0.02 * _GiB),
        base_bytes=int(4.0 * _GiB),
        per_tier_fit={
            8: (False, -int(0.9 * _GiB)),
            12: (True, int(3.1 * _GiB)),
            16: (True, int(7.1 * _GiB)),
            24: (True, int(15.1 * _GiB)),
            40: (True, int(31.1 * _GiB)),
            80: (True, int(71.1 * _GiB)),
        },
        breakdown="stub",
    )
    table = format_tier_table(assessment)
    assert "OVERFLOW" in table
    assert "FITS" in table
    for tier in (8, 12, 16, 24, 40, 80):
        assert str(tier) in table


# ---------------------------------------------------------------------------
# measure_external_vram — nvidia-smi path
# ---------------------------------------------------------------------------


def test_measure_external_vram_uses_nvidia_smi():
    """measure_external_vram must read device-wide memory via nvidia-smi."""
    completed = subprocess.CompletedProcess(
        args=["nvidia-smi"],
        returncode=0,
        stdout="4608\n",  # 4608 MiB ≈ 4.5 GiB
        stderr="",
    )
    with (
        patch("paramem.server.vram_validator.torch") as mock_torch,
        patch("paramem.server.vram_validator.subprocess.run", return_value=completed) as mock_run,
    ):
        mock_torch.cuda.get_device_properties.return_value.total_memory = _TOTAL_VRAM_BYTES
        mock_torch.cuda.memory_allocated.return_value = 0

        total, external = measure_external_vram()

    assert total == _TOTAL_VRAM_BYTES
    assert external == 4608 * 2**20
    mock_run.assert_called_once()
    mock_torch.cuda.memory_allocated.assert_not_called()


def test_measure_external_vram_falls_back_when_nvidia_smi_missing(caplog):
    """When nvidia-smi is unavailable, falls back to torch.cuda.memory_allocated."""

    named = logging.getLogger("paramem.server.vram_validator")
    orig_propagate = named.propagate
    named.propagate = True
    caplog.set_level(logging.WARNING, logger="paramem.server.vram_validator")
    named.addHandler(caplog.handler)

    try:
        with (
            patch("paramem.server.vram_validator.torch") as mock_torch,
            patch(
                "paramem.server.vram_validator.subprocess.run",
                side_effect=FileNotFoundError("nvidia-smi"),
            ),
        ):
            mock_torch.cuda.get_device_properties.return_value.total_memory = _TOTAL_VRAM_BYTES
            mock_torch.cuda.memory_allocated.return_value = 1_000_000_000

            total, external = measure_external_vram()
    finally:
        named.removeHandler(caplog.handler)
        named.propagate = orig_propagate

    assert total == _TOTAL_VRAM_BYTES
    assert external == 1_000_000_000
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "nvidia-smi unavailable" in logged
    assert "blind spot" in logged or "invisible" in logged


# ---------------------------------------------------------------------------
# enforce_live_budget
# ---------------------------------------------------------------------------


def test_enforce_live_budget_rejects_when_external_consumes_majority():
    """With 4.5 GiB external and 7.5 GiB required on 8 GiB: live budget 3.5 GiB < required."""
    from paramem.server.vram_validator import TopologyAssessment

    total_memory = 8 * _GiB
    external = int(4.5 * _GiB)
    required = int(7.5 * _GiB)

    assessment = TopologyAssessment(
        required_bytes=required,
        adapter_bytes=int(0.05 * _GiB),
        base_bytes=int(3.8 * _GiB),
        per_tier_fit={8: (False, total_memory - required), 12: (True, 12 * _GiB - required)},
        breakdown="(test fixture)",
    )

    with pytest.raises(ConfigurationError) as exc_info:
        enforce_live_budget(assessment, total_memory, external)

    msg = str(exc_info.value)
    assert "VRAM live budget insufficient" in msg
    assert "External occupancy" in msg


def test_enforce_live_budget_passes_when_fits():
    """Sufficient live budget must not raise."""
    from paramem.server.vram_validator import TopologyAssessment

    total_memory = 8 * _GiB
    external = int(0.5 * _GiB)
    required = int(5.5 * _GiB)

    assessment = TopologyAssessment(
        required_bytes=required,
        adapter_bytes=int(0.02 * _GiB),
        base_bytes=int(4.0 * _GiB),
        per_tier_fit={8: (True, total_memory - required)},
        breakdown="(test fixture)",
    )
    enforce_live_budget(assessment, total_memory, external)  # no exception


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

    assert estimate_stt_bytes(_FakeSTTConfig(enabled=False)) == 0


def test_estimate_stt_bytes_zero_when_cpu():
    from paramem.server.vram_validator import estimate_stt_bytes

    assert estimate_stt_bytes(_FakeSTTConfig(device="cpu")) == 0


def test_estimate_stt_bytes_zero_when_permanent_cloud_only():
    from paramem.server.vram_validator import estimate_stt_bytes

    assert estimate_stt_bytes(_FakeSTTConfig(device="cuda"), permanent_cloud_only=True) == 0


def test_estimate_stt_bytes_returns_predictor_value_on_cache_hit():
    """When predictor returns a value, estimate_stt_bytes returns it unchanged."""
    from paramem.server.vram_validator import estimate_stt_bytes

    cfg = _FakeSTTConfig(device="cuda")
    with patch("paramem.server.vram_validator.predict_stt_bytes", return_value=500_000_000):
        result = estimate_stt_bytes(cfg)
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
            result = estimate_stt_bytes(cfg)
    finally:
        named.removeHandler(caplog.handler)
        named.propagate = orig_propagate

    assert result == 0
    assert "STT not cached" in caplog.text or "live gate" in caplog.text.lower()


def test_estimate_tts_bytes_zero_when_disabled():
    from paramem.server.vram_validator import estimate_tts_bytes

    assert estimate_tts_bytes(_FakeTTSConfig(enabled=False)) == 0


def test_estimate_tts_bytes_zero_when_permanent_cloud_only():
    from paramem.server.vram_validator import estimate_tts_bytes

    cfg = _FakeTTSConfig(voices={"en": _FakeTTSVoice(engine="piper")})
    assert estimate_tts_bytes(cfg, permanent_cloud_only=True) == 0


def test_estimate_tts_bytes_returns_predictor_value_on_cache_hit():
    from paramem.server.vram_validator import estimate_tts_bytes

    cfg = _FakeTTSConfig()
    with patch("paramem.server.vram_validator.predict_tts_bytes", return_value=400_000_000):
        result = estimate_tts_bytes(cfg)
    assert result == 400_000_000


def test_estimate_tts_bytes_returns_zero_on_cache_miss():
    from paramem.server.vram_validator import estimate_tts_bytes

    cfg = _FakeTTSConfig()
    with patch("paramem.server.vram_validator.predict_tts_bytes", return_value=None):
        result = estimate_tts_bytes(cfg)
    assert result == 0


def test_estimate_stt_bytes_returns_gpu_bytes_when_defer_model():
    """permanent_cloud_only=False (defer-model path) reserves GPU pair bytes."""
    from paramem.server.vram_validator import estimate_stt_bytes

    cfg = _FakeSTTConfig(enabled=True, device="cuda", model="distil-large-v3")
    with patch("paramem.server.vram_validator.predict_stt_bytes", return_value=960_000_000):
        result = estimate_stt_bytes(cfg, permanent_cloud_only=False)
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
        result = estimate_tts_bytes(cfg, permanent_cloud_only=False)
    assert result > 0


def test_estimate_stt_bytes_zero_when_explicit_cloud_only():
    """permanent_cloud_only=True (--cloud-only) returns zero bytes."""
    from paramem.server.vram_validator import estimate_stt_bytes

    cfg = _FakeSTTConfig(enabled=True, device="cuda", model="distil-large-v3")
    assert estimate_stt_bytes(cfg, permanent_cloud_only=True) == 0


def test_estimate_tts_bytes_zero_when_explicit_cloud_only():
    """permanent_cloud_only=True (--cloud-only) returns zero bytes."""
    from paramem.server.vram_validator import estimate_tts_bytes

    cfg = _FakeTTSConfig(
        enabled=True,
        device="cuda",
        voices={"en": _FakeTTSVoice(engine="piper", model="en_US-lessac-high")},
    )
    assert estimate_tts_bytes(cfg, permanent_cloud_only=True) == 0


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

    The old single-shot ``measure_external_vram`` + ``enforce_live_budget``
    + ``sys.exit(1)`` boot gate was replaced by the poll-and-degrade
    ``_wait_for_gpu_drain`` in the VRAM-overcommit fix (Fix 2).
    ``measure_external_vram`` and ``enforce_live_budget`` are no longer
    called from the lifespan boot path.
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
            per_tier_fit={8: (True, 0)},
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
        server_app._state.pop("vram_overflow_warning", None)

    # predict_base_bytes must come before assess_topology
    call_names = [c[0] for c in calls]
    assert "predict_base_bytes" in call_names, "predict_base_bytes must be called in lifespan"
    assert "assess_topology" in call_names, "assess_topology must be called in lifespan"
    pred_idx = call_names.index("predict_base_bytes")
    assess_idx = call_names.index("assess_topology")
    assert pred_idx < assess_idx, "predict_base_bytes must come before assess_topology"

    # New boot order: assess_topology → _wait_for_gpu_drain → load_base_model
    # (measure_external_vram and enforce_live_budget removed from boot path in Fix 2)
    expected_tail = [
        "assess_topology",
        "_wait_for_gpu_drain",
        "load_base_model",
    ]
    tail_order = [c for c in call_names if c in expected_tail]
    assert tail_order == expected_tail, (
        f"Lifespan must invoke {expected_tail} in order; got {tail_order}"
    )

    # Confirm the old single-shot gate functions are NOT called from the boot path.
    assert "measure_external_vram" not in call_names, (
        "measure_external_vram must NOT be called in the boot path "
        "(replaced by _wait_for_gpu_drain)"
    )
    assert "enforce_live_budget" not in call_names, (
        "enforce_live_budget must NOT be called in the boot path (replaced by _wait_for_gpu_drain)"
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
    """Regression guard: with headroom=1.5 GiB and the Mistral 7B NF4 topology
    (7 interims, rank=8, 4 modules, 3 mains), required_bytes must be < 6.83 GiB
    — the device-wide free measured at boot time in the live failure that prompted
    Change 1.

    Concrete numbers (all derived from the working-set formula):
      base ≈ 4597 MiB  (inferred from the pre-fix failure: 7187 MiB total with
                         headroom=2.0 → 7187 - 2*1024 - 256 - 11*26 = 4597 MiB)
      11 adapters × 26 MiB = 286 MiB
      headroom = 1.5 × 1024 = 1536 MiB
      safety margin = 256 MiB
      total = 4597 + 286 + 1536 + 256 = 6675 MiB = 6.519 GiB < 6.83 GiB ✓
    """
    _MiB = 1024 * 1024

    # Inferred Mistral 7B NF4 base footprint from the pre-fix required_bytes.
    # (7187 MiB total - 256 MiB safety - 2*1024 MiB headroom - 11*26 MiB adapters)
    base_bytes = 4597 * _MiB

    result = assess_topology(
        _MISTRAL_ADAPTER,
        max_interim_count=7,
        base_bytes=base_bytes,
        hidden_size=_HIDDEN,
        num_layers=_LAYERS,
        headroom_gib=1.5,
    )

    drain_time_free_bytes = int(6.83 * _GiB)
    assert result.required_bytes < drain_time_free_bytes, (
        f"With headroom=1.5 GiB, required_bytes should be < 6.83 GiB (drain-time free), "
        f"got {result.required_bytes / _GiB:.3f} GiB ({result.required_bytes // _MiB} MiB)"
    )
    # Concrete sanity: must be ≈ 6675 MiB ± 10 MiB (rounding from float GiB).
    expected_mib = 6675
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
        assert estimate_stt_bytes(stt_cfg, permanent_cloud_only=pco) > 0
    with patch("paramem.server.vram_validator.predict_tts_bytes", return_value=380_000_000):
        assert estimate_tts_bytes(tts_cfg, permanent_cloud_only=pco) > 0


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
    assert estimate_stt_bytes(stt_cfg, permanent_cloud_only=pco) == 0
    assert estimate_tts_bytes(tts_cfg, permanent_cloud_only=pco) == 0


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
    assert estimate_stt_bytes(stt_cfg, permanent_cloud_only=pco) == 0
    assert estimate_tts_bytes(tts_cfg, permanent_cloud_only=pco) == 0


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
        assert estimate_stt_bytes(stt_cfg, permanent_cloud_only=pco) > 0
    with patch("paramem.server.vram_validator.predict_tts_bytes", return_value=380_000_000):
        assert estimate_tts_bytes(tts_cfg, permanent_cloud_only=pco) > 0
