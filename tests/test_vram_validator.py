"""Pure unit tests for the startup VRAM budget validator.

No GPU required — all CUDA calls are mocked. Tests cover:
- Pass case with a realistic Mistral 7B + rank-8 + 4 modules + 7 sessions config.
- Fail case when max_interim_count is too large.
- Fail case when LoRA rank is too large.
- Verification that the +1 in_training staging slot is included in the formula.
- Error message content: names all actionable config knobs with current values.
- Per-component breakdown shown in failure message (Fix 1 / MINOR-2).
- Remediation block lists all knobs with current values (Fix 1 / MINOR-2).
- Breakdown logged at INFO on success path (Fix 1 / MINOR-2).
- WARNING log when unknown model triggers fallback (Fix 2 / MINOR-3).
- Early ConfigurationError when no CUDA GPU is available (Fix 3 / MINOR-4).
"""

from __future__ import annotations

import logging
import subprocess
from unittest.mock import patch

import pytest

from paramem.server.vram_validator import (
    _FALLBACK_BASE_BYTES,
    _LORA_DTYPE_BYTES,
    _PEFT_OVERHEAD_PER_ADAPTER_BYTES,
    ConfigurationError,
    estimated_adapter_bytes,
    required_working_set_bytes,
    validate_startup_vram,
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

# Free VRAM after the base model is already loaded. The validator runs
# post-load (see ``app.py`` lifespan), so ``mem_get_info`` reports free VRAM
# with the base model's footprint already subtracted. On an 8 GiB card with a
# ~3.8 GiB base model and ~1 GiB of CUDA context + allocator overhead, ~3 GiB
# remains free. The fixed validator adds ``base_bytes`` back, giving an
# effective budget of ~6.8 GiB — enough for the realistic config to fit while
# large-rank / many-interim scenarios still overflow.
_FREE_VRAM_BYTES = int(3 * _GiB)
_TOTAL_VRAM_BYTES = int(8 * _GiB)

# Tiny free VRAM that guarantees a failure in most tests
_TINY_VRAM_BYTES = 128 * 1024 * 1024  # 128 MiB


def _adapter_bytes(adapter_config: AdapterConfig = _MISTRAL_ADAPTER) -> int:
    """Compute expected adapter bytes for Mistral 7B dims."""
    return estimated_adapter_bytes(adapter_config, _HIDDEN, _LAYERS)


def _mock_model(allocated_bytes: int = _BASE_MODEL_BYTES):
    """Return a dummy model sentinel. CUDA allocator is patched separately."""
    return object()


# ---------------------------------------------------------------------------
# Test 1 — realistic config passes without raising
# ---------------------------------------------------------------------------


def test_validate_passes_with_realistic_config():
    """Mistral 7B NF4 + rank-8 + 4 modules + 7 sessions + 3 mains + 1 in_training
    should fit comfortably in 7.5 GiB free VRAM.

    Mocks:
    - ``torch.cuda.memory_allocated`` → base model bytes
    - ``torch.cuda.mem_get_info`` → (free, total)
    - ``torch.cuda.is_available`` → True
    """
    model = _mock_model()

    with (
        patch("paramem.server.vram_validator.torch") as mock_torch,
    ):
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = _BASE_MODEL_BYTES
        mock_torch.cuda.mem_get_info.return_value = (_FREE_VRAM_BYTES, _TOTAL_VRAM_BYTES)
        mock_torch.cuda.get_device_properties.return_value.total_memory = _TOTAL_VRAM_BYTES

        # Should not raise
        validate_startup_vram(
            model,
            _MISTRAL_ADAPTER,
            max_interim_count=7,
            model_name="mistral",
            main_adapter_count=3,
        )


# ---------------------------------------------------------------------------
# Test 2 — too many sessions causes ConfigurationError
# ---------------------------------------------------------------------------


def test_validate_fails_when_too_many_interims():
    """max_interim_count=300 with 7.5 GiB free VRAM should fail.

    Rank-8 adapters are small (~16 MiB each), so we need a very large
    interim count to exhaust the budget. 300 interims × 16 MiB = 4.75 GiB
    plus base model (~3.8 GiB) plus headroom (1 GiB) plus margin (0.25 GiB)
    totals ~9.8 GiB, which exceeds 7.5 GiB.

    The error message must reference ``max_interim_count``.
    """
    model = _mock_model()

    with (
        patch("paramem.server.vram_validator.torch") as mock_torch,
    ):
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = _BASE_MODEL_BYTES
        mock_torch.cuda.mem_get_info.return_value = (_FREE_VRAM_BYTES, _TOTAL_VRAM_BYTES)
        mock_torch.cuda.get_device_properties.return_value.total_memory = _TOTAL_VRAM_BYTES

        with pytest.raises(ConfigurationError) as exc_info:
            validate_startup_vram(
                model,
                _MISTRAL_ADAPTER,
                max_interim_count=300,
                model_name="mistral",
                main_adapter_count=3,
            )

    assert "max_interim_count" in str(exc_info.value), (
        "Error message must reference 'max_interim_count' so operators know which knob to reduce. "
        f"Got: {exc_info.value}"
    )


# ---------------------------------------------------------------------------
# Test 3 — enormous rank causes ConfigurationError
# ---------------------------------------------------------------------------


def test_validate_fails_with_huge_rank():
    """rank=256 adapter with 4 modules, 7 sessions should overflow the 8 GiB cap.

    The error message must reference ``rank``.
    """
    high_rank_adapter = AdapterConfig(
        rank=256,
        alpha=512,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        dropout=0.0,
    )
    model = _mock_model()

    with (
        patch("paramem.server.vram_validator.torch") as mock_torch,
    ):
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = _BASE_MODEL_BYTES
        mock_torch.cuda.mem_get_info.return_value = (_FREE_VRAM_BYTES, _TOTAL_VRAM_BYTES)
        mock_torch.cuda.get_device_properties.return_value.total_memory = _TOTAL_VRAM_BYTES

        with pytest.raises(ConfigurationError) as exc_info:
            validate_startup_vram(
                model,
                high_rank_adapter,
                max_interim_count=7,
                model_name="mistral",
                main_adapter_count=3,
            )

    assert "rank" in str(exc_info.value), (
        "Error message must reference 'rank' so operators know which knob to reduce"
    )


# ---------------------------------------------------------------------------
# Test 4 — in_training slot (+1) is included in the formula
# ---------------------------------------------------------------------------


def test_math_includes_in_training_slot():
    """The +1 in_training staging slot must be counted in the working set.

    We verify this by comparing the output of ``required_working_set_bytes``
    with and without the +1 slot. The difference must equal exactly one
    adapter's worth of bytes.
    """
    adapter_bytes = _adapter_bytes()

    with_staging = required_working_set_bytes(
        base_model_bytes=_BASE_MODEL_BYTES,
        adapter_bytes=adapter_bytes,
        main_adapter_count=3,
        max_interim_count=7,
        headroom_bytes=int(1.0 * _GiB),
    )

    # Manually compute: base + (3+7+1) * adapter + headroom
    expected_total_adapters = (3 + 7 + 1) * adapter_bytes
    expected = _BASE_MODEL_BYTES + expected_total_adapters + int(1.0 * _GiB)

    assert with_staging == expected, (
        f"required_working_set_bytes should include +1 for in_training staging slot. "
        f"Expected {expected}, got {with_staging}. "
        f"Difference of {abs(with_staging - expected)} bytes = "
        f"{abs(with_staging - expected) / adapter_bytes:.1f} adapter units."
    )

    # Confirm the staging slot accounts for exactly one adapter_bytes worth of difference
    # compared to a hypothetical formula without it:
    without_staging_manual = _BASE_MODEL_BYTES + (3 + 7) * adapter_bytes + int(1.0 * _GiB)
    assert with_staging - without_staging_manual == adapter_bytes, (
        "The +1 in_training staging slot must add exactly one adapter_bytes to the total. "
        f"Expected diff={adapter_bytes}, got {with_staging - without_staging_manual}"
    )


# ---------------------------------------------------------------------------
# Test 5 — error message lists all actionable config knobs with current values
# ---------------------------------------------------------------------------


def test_error_message_lists_actionable_knobs():
    """When the budget is exceeded, the error message must name every config
    field the operator can reduce — rank, max_interim_count, target_modules,
    base model — and include the current value of each.
    """
    model = _mock_model()

    with (
        patch("paramem.server.vram_validator.torch") as mock_torch,
    ):
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = _BASE_MODEL_BYTES
        mock_torch.cuda.mem_get_info.return_value = (_TINY_VRAM_BYTES, _TOTAL_VRAM_BYTES)
        # Hardware-cap path: total_memory drives the budget since commit task #17.
        # Use the tiny value here so the topology overflows and the validator
        # raises.
        mock_torch.cuda.get_device_properties.return_value.total_memory = _TINY_VRAM_BYTES

        with pytest.raises(ConfigurationError) as exc_info:
            validate_startup_vram(
                model,
                _MISTRAL_ADAPTER,
                max_interim_count=7,
                model_name="mistral",
                main_adapter_count=3,
            )

    error_text = str(exc_info.value)
    # Knob names
    required_substrings = ["rank", "max_interim_count", "target_modules", "base model"]
    for substring in required_substrings:
        assert substring in error_text, (
            f"Error message must mention '{substring}' as an actionable remediation knob. "
            f"Got:\n{error_text}"
        )
    # Current values of each knob must appear in the remediation block
    assert "current=8" in error_text, (
        "Error message must show current rank value (8). Got:\n{error_text}"
    )
    assert "current=7" in error_text, (
        "Error message must show current max_interim_count value (7). Got:\n{error_text}"
    )
    assert "current=4" in error_text, (
        "Error message must show current number of target_modules (4). Got:\n{error_text}"
    )
    assert "mistral" in error_text, (
        "Error message must show the current model name. Got:\n{error_text}"
    )


# ---------------------------------------------------------------------------
# Additional: estimated_adapter_bytes pure arithmetic check
# ---------------------------------------------------------------------------


def test_estimated_adapter_bytes_mistral_rank8_four_modules():
    """Spot-check the adapter bytes formula for Mistral 7B rank-8 attention-only.

    Formula: num_modules × num_layers × 2 × rank × hidden × dtype_bytes
           = 4 × 32 × 2 × 8 × 4096 × 2
           = 4 × 32 × 2 × 8 × 4096 × 2
    """
    # Raw LoRA A+B bytes plus a flat PEFT per-adapter overhead (wrapper
    # bookkeeping, padding).  Overhead is a validator-level constant and
    # must be counted here so the test tracks the formula the validator
    # actually uses.
    expected = 4 * 32 * 2 * 8 * 4096 * _LORA_DTYPE_BYTES + _PEFT_OVERHEAD_PER_ADAPTER_BYTES
    result = estimated_adapter_bytes(_MISTRAL_ADAPTER, hidden_size=4096, num_layers=32)
    assert result == expected, (
        f"estimated_adapter_bytes mismatch: expected {expected}, got {result}"
    )


# ---------------------------------------------------------------------------
# Pre-load path: validator runs before model load (canonical call site)
# ---------------------------------------------------------------------------


def test_validate_pre_load_uses_static_base_model_lookup():
    """Canonical call path: ``model=None`` before ``load_base_model``.

    ``memory_allocated`` is never called; ``base_bytes`` comes from the static
    ``_MODEL_VRAM_BYTES`` lookup. ``mem_get_info`` reports full card free
    (minus external consumers only), which compares directly against
    ``total_required`` with no add-back.
    """
    # Pre-load free VRAM ≈ total (no external consumers).
    pre_load_free_bytes = int(7.5 * _GiB)

    with patch("paramem.server.vram_validator.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.mem_get_info.return_value = (
            pre_load_free_bytes,
            _TOTAL_VRAM_BYTES,
        )
        mock_torch.cuda.get_device_properties.return_value.total_memory = _TOTAL_VRAM_BYTES
        # memory_allocated must not be consulted on the pre-load path.
        mock_torch.cuda.memory_allocated.side_effect = AssertionError(
            "memory_allocated should not be called pre-load"
        )

        validate_startup_vram(
            None,
            _MISTRAL_ADAPTER,
            max_interim_count=7,
            model_name="mistral",
            main_adapter_count=3,
        )


# ---------------------------------------------------------------------------
# Regression: post-load free VRAM must not double-count the base model
# ---------------------------------------------------------------------------


def test_post_load_free_vram_does_not_double_count_base_model():
    """Back-compat: if the validator is called after the base model is loaded
    (``model`` is a real object and ``memory_allocated`` > 0), the already-
    loaded footprint is added back to available VRAM so that ``total_required``
    (which still includes ``base_bytes``) is not double-counted.

    Regression for commit 156ff44 → follow-up fix. The canonical call site is
    now pre-load, but the post-load path is preserved for test harnesses and
    any future caller that validates after load.
    """
    model = _mock_model()

    with patch("paramem.server.vram_validator.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = _BASE_MODEL_BYTES
        # Realistic post-load reading: free VRAM is much less than total
        # because the base model already occupies ~3.8 GiB.  The validator's
        # hardware-cap path reads total_memory directly; mem_get_info is now
        # vestigial but kept in the mock for back-compat with older code paths.
        mock_torch.cuda.mem_get_info.return_value = (
            _FREE_VRAM_BYTES,
            _TOTAL_VRAM_BYTES,
        )
        mock_torch.cuda.get_device_properties.return_value.total_memory = _TOTAL_VRAM_BYTES

        # Topology that fits in total VRAM (7 interims at rank 8) but appears
        # to overflow free VRAM alone — must be accepted once the base model
        # cost is added back to available.
        validate_startup_vram(
            model,
            _MISTRAL_ADAPTER,
            max_interim_count=7,
            model_name="mistral",
            main_adapter_count=3,
        )


# ---------------------------------------------------------------------------
# Additional: vram_cap_gib override
# ---------------------------------------------------------------------------


def test_vram_cap_gib_override_passes():
    """When vram_cap_gib is provided, it overrides the hardware cap lookup."""
    model = _mock_model()

    with patch("paramem.server.vram_validator.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = _BASE_MODEL_BYTES
        # Neither mem_get_info nor get_device_properties should be consulted
        # when the caller passes an explicit cap.
        mock_torch.cuda.mem_get_info.side_effect = AssertionError("Should not be called")
        mock_torch.cuda.get_device_properties.side_effect = AssertionError("Should not be called")

        # 8 GiB cap — should pass
        validate_startup_vram(
            model,
            _MISTRAL_ADAPTER,
            max_interim_count=7,
            model_name="mistral",
            main_adapter_count=3,
            vram_cap_gib=8.0,
        )


def test_vram_cap_gib_override_fails():
    """When vram_cap_gib is very small, the validator raises ConfigurationError
    without querying the hardware-cap lookup."""
    model = _mock_model()

    with patch("paramem.server.vram_validator.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = _BASE_MODEL_BYTES
        mock_torch.cuda.mem_get_info.side_effect = AssertionError("Should not be called")
        mock_torch.cuda.get_device_properties.side_effect = AssertionError("Should not be called")

        with pytest.raises(ConfigurationError):
            validate_startup_vram(
                model,
                _MISTRAL_ADAPTER,
                max_interim_count=7,
                model_name="mistral",
                main_adapter_count=3,
                vram_cap_gib=0.1,  # 100 MiB — trivially too small
            )


# ---------------------------------------------------------------------------
# Fix 1 / MINOR-2: Breakdown shown in failure message
# ---------------------------------------------------------------------------


def test_breakdown_shown_in_failure_message():
    """ConfigurationError must contain every line of the VRAM breakdown table.

    Asserts substrings for: base model, main adapters, session adapters,
    in_training staging slot, KV cache headroom, fragmentation safety margin,
    required total, available VRAM, and the margin line.
    """
    model = _mock_model()

    with patch("paramem.server.vram_validator.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = _BASE_MODEL_BYTES
        mock_torch.cuda.mem_get_info.return_value = (_TINY_VRAM_BYTES, _TOTAL_VRAM_BYTES)
        # Hardware-cap path: total_memory drives the budget since commit task #17.
        # Use the tiny value here so the topology overflows and the validator
        # raises.
        mock_torch.cuda.get_device_properties.return_value.total_memory = _TINY_VRAM_BYTES

        with pytest.raises(ConfigurationError) as exc_info:
            validate_startup_vram(
                model,
                _MISTRAL_ADAPTER,
                max_interim_count=7,
                model_name="mistral",
                main_adapter_count=3,
            )

    error_text = str(exc_info.value)
    required_substrings = [
        "VRAM Working Set Breakdown",
        "base model",
        "main adapters",
        "interim adapters",
        "in_training staging slot",
        "KV cache headroom",
        "fragmentation safety margin",
        "required total",
        "available (free VRAM)",
        "margin",
    ]
    for substring in required_substrings:
        assert substring in error_text, (
            f"Failure message missing breakdown entry '{substring}'.\nGot:\n{error_text}"
        )


# ---------------------------------------------------------------------------
# Fix 1 / MINOR-2: Remediation block lists all knobs with current values
# ---------------------------------------------------------------------------


def test_remediation_block_lists_all_knobs():
    """ConfigurationError must contain the remediation block with every knob
    name and its current configured value.
    """
    model = _mock_model()

    with patch("paramem.server.vram_validator.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = _BASE_MODEL_BYTES
        mock_torch.cuda.mem_get_info.return_value = (_TINY_VRAM_BYTES, _TOTAL_VRAM_BYTES)
        # Hardware-cap path: total_memory drives the budget since commit task #17.
        # Use the tiny value here so the topology overflows and the validator
        # raises.
        mock_torch.cuda.get_device_properties.return_value.total_memory = _TINY_VRAM_BYTES

        with pytest.raises(ConfigurationError) as exc_info:
            validate_startup_vram(
                model,
                _MISTRAL_ADAPTER,
                max_interim_count=7,
                model_name="mistral",
                main_adapter_count=3,
            )

    error_text = str(exc_info.value)
    # Knob names and their current values must all appear
    assert "rank" in error_text, f"Missing 'rank' in remediation block.\nGot:\n{error_text}"
    assert "current=8" in error_text, (
        f"Missing current rank value (8) in remediation block.\nGot:\n{error_text}"
    )
    assert "max_interim_count" in error_text, (
        f"Missing 'max_interim_count' in remediation block.\nGot:\n{error_text}"
    )
    assert "current=7" in error_text, (
        f"Missing current max_interim_count value (7) in remediation block.\nGot:\n{error_text}"
    )
    assert "target_modules" in error_text, (
        f"Missing 'target_modules' in remediation block.\nGot:\n{error_text}"
    )
    assert "current=4" in error_text, (
        f"Missing current target_modules count (4) in remediation block.\nGot:\n{error_text}"
    )
    assert "base model" in error_text, (
        f"Missing 'base model' in remediation block.\nGot:\n{error_text}"
    )
    assert "mistral" in error_text, (
        f"Missing current model name in remediation block.\nGot:\n{error_text}"
    )


# ---------------------------------------------------------------------------
# Fix 1 / MINOR-2: Breakdown logged at INFO on the success path
# ---------------------------------------------------------------------------


def test_breakdown_logged_on_success(caplog):
    """When the topology fits, the VRAM breakdown must be logged at INFO level.

    Checks that the INFO log contains every section of the breakdown table.

    Implementation note: the vram_validator logger does not propagate to the
    root logger reliably in this project's pytest setup (ament plugins alter
    the root handler chain). The workaround — used by test_schema_config.py —
    is to attach caplog.handler directly to the named logger.
    """
    model = _mock_model()

    # Attach caplog's handler directly to the named logger (project pattern).
    named_logger = logging.getLogger("paramem.server.vram_validator")
    named_logger.addHandler(caplog.handler)
    named_logger.setLevel(logging.INFO)
    try:
        with patch("paramem.server.vram_validator.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.memory_allocated.return_value = _BASE_MODEL_BYTES
            mock_torch.cuda.mem_get_info.return_value = (_FREE_VRAM_BYTES, _TOTAL_VRAM_BYTES)
            mock_torch.cuda.get_device_properties.return_value.total_memory = _TOTAL_VRAM_BYTES

            validate_startup_vram(
                model,
                _MISTRAL_ADAPTER,
                max_interim_count=7,
                model_name="mistral",
                main_adapter_count=3,
            )
    finally:
        named_logger.removeHandler(caplog.handler)

    info_text = "\n".join(r.getMessage() for r in caplog.records if r.levelno == logging.INFO)
    required_substrings = [
        "VRAM Working Set Breakdown",
        "base model",
        "main adapters",
        "interim adapters",
        "in_training staging slot",
        "KV cache headroom",
        "fragmentation safety margin",
        "required total",
        "available (free VRAM)",
        "margin",
    ]
    for substring in required_substrings:
        assert substring in info_text, (
            f"Success INFO log missing breakdown entry '{substring}'.\nGot:\n{info_text}"
        )


# ---------------------------------------------------------------------------
# Fix 2 / MINOR-3: WARNING log when unknown model triggers fallback
# ---------------------------------------------------------------------------


def test_warning_when_unknown_model_uses_fallback(caplog):
    """When model_name is empty and CUDA returns 0 allocated bytes, the
    validator must emit a WARNING that mentions ``_MODEL_VRAM_BYTES`` and the
    fallback value in MiB, not a silent debug log.
    """
    _MiB = 1024 * 1024
    expected_fallback_mib = _FALLBACK_BASE_BYTES // _MiB

    model = _mock_model()

    # Attach caplog's handler directly to the named logger (project pattern from
    # test_schema_config.py — ament plugins prevent root propagation).
    named_logger = logging.getLogger("paramem.server.vram_validator")
    named_logger.addHandler(caplog.handler)
    named_logger.setLevel(logging.WARNING)
    try:
        with patch("paramem.server.vram_validator.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            # Return 0 to force the static fallback path
            mock_torch.cuda.memory_allocated.return_value = 0

            # Use empty model_name to trigger unknown-model path; override
            # vram_cap_gib to a generous value so the fallback topology fits
            # and we can assert on the warning message alone.
            validate_startup_vram(
                model,
                _MISTRAL_ADAPTER,
                max_interim_count=7,
                model_name="",
                main_adapter_count=3,
                vram_cap_gib=8.0,
            )
    finally:
        named_logger.removeHandler(caplog.handler)

    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    fallback_warnings = [r for r in warning_records if "_MODEL_VRAM_BYTES" in r.getMessage()]
    assert fallback_warnings, (
        "Expected a WARNING log mentioning '_MODEL_VRAM_BYTES' when CUDA returns 0. "
        f"All WARNING records: {[r.getMessage() for r in warning_records]}"
    )
    fallback_msg = fallback_warnings[0].getMessage()
    assert str(expected_fallback_mib) in fallback_msg, (
        f"WARNING log must include the fallback value ({expected_fallback_mib} MiB). "
        f"Got: {fallback_msg}"
    )


# ---------------------------------------------------------------------------
# Fix 3 / MINOR-4: No GPU raises clear ConfigurationError
# ---------------------------------------------------------------------------


def test_no_gpu_raises_clear_error():
    """When torch.cuda.is_available() returns False, validate_startup_vram must
    raise ConfigurationError immediately with a message that clearly names:
    - the absence of a GPU ("no" or "none")
    - "GPU"
    - the cloud-only escape hatch ("cloud-only" or "cloud_only")
    """
    model = _mock_model()

    with patch("paramem.server.vram_validator.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = False

        with pytest.raises(ConfigurationError) as exc_info:
            validate_startup_vram(
                model,
                _MISTRAL_ADAPTER,
                max_interim_count=7,
                model_name="mistral",
                main_adapter_count=3,
            )

    error_text = str(exc_info.value).lower()
    assert "gpu" in error_text, f"Error message must mention 'GPU'. Got:\n{exc_info.value}"
    # Accept either "no" or "none" phrasing for the absence of a GPU
    assert "no" in error_text or "none" in error_text, (
        f"Error message must indicate absence of GPU. Got:\n{exc_info.value}"
    )
    # Accept either hyphenated or underscored spelling
    assert "cloud-only" in error_text or "cloud_only" in error_text, (
        f"Error message must mention 'cloud-only' escape hatch. Got:\n{exc_info.value}"
    )


# ---------------------------------------------------------------------------
# Integration: lifespan ordering (validator must run before load_base_model)
# ---------------------------------------------------------------------------


def test_lifespan_runs_validator_before_load_base_model(tmp_path):
    """Integration guard for the startup lifecycle.

    Exercises the FastAPI ``lifespan`` context with ``load_base_model``,
    ``assess_topology``, ``measure_external_vram`` and ``enforce_live_budget``
    replaced by spies. Asserts the correct call order:

        assess_topology → measure_external_vram → enforce_live_budget →
        load_base_model

    This ordering is load-bearing: ``measure_external_vram`` snapshots
    non-ParaMem GPU occupancy BEFORE the base model is loaded, so the live
    budget is correctly computed from external consumers only. Moving
    ``enforce_live_budget`` after ``load_base_model`` would count ParaMem's
    own footprint as "external", producing a self-consistent but wrong
    budget.

    Uses a fresh tmp_path for ``paths.data`` so the security startup gate
    (``assert_mode_consistency``) does not trip on the real data directory
    — whatever mode the operator's deployment is in, this unit test must
    not depend on it.
    """
    import asyncio

    from paramem.server import app as server_app
    from paramem.server.config import ServerConfig

    class _Sentinel(Exception):
        pass

    calls: list[tuple] = []

    def spy_assess(*args, **kwargs):
        calls.append(("assess_topology",))
        # Return a minimal-but-valid TopologyAssessment so enforce_live_budget
        # can consume it without crashing on attribute access.
        from paramem.server.vram_validator import TopologyAssessment

        return TopologyAssessment(
            required_bytes=1,
            adapter_bytes=1,
            base_bytes=1,
            per_tier_fit={8: (True, 0)},
            breakdown="stub",
        )

    def spy_measure(*args, **kwargs):
        calls.append(("measure_external_vram",))
        return (8 * 2**30, 0)  # 8 GiB total, 0 external

    def spy_enforce(*args, **kwargs):
        calls.append(("enforce_live_budget",))

    def spy_load_base_model(*args, **kwargs):
        calls.append(("load_base_model",))
        raise _Sentinel("short-circuit after load_base_model is invoked")

    config = ServerConfig(model_name="mistral")
    config.cloud_only = False
    # Point the data directory at a fresh tmp so the security startup gate
    # sees an empty (therefore consistent) store.
    config.paths.data = tmp_path / "data"

    saved_state = {
        key: server_app._state.get(key) for key in ("config", "cloud_only_startup", "defer_model")
    }

    server_app._state["config"] = config
    server_app._state["cloud_only_startup"] = False
    server_app._state["defer_model"] = False

    try:
        with (
            patch.object(server_app, "assess_topology", spy_assess),
            patch.object(server_app, "measure_external_vram", spy_measure),
            patch.object(server_app, "enforce_live_budget", spy_enforce),
            patch.object(server_app, "load_base_model", spy_load_base_model),
            patch.object(server_app, "_gpu_occupied", return_value=False),
            patch("paramem.server.app.torch.cuda.is_available", return_value=True),
            # Patching ``app.torch.cuda.is_available`` to True flips ``torch`` to
            # the GPU branch globally (it's the same module object imported by
            # vram_guard), so apply_process_cap's own is_available guard also
            # returns True and it then calls set_per_process_memory_fraction —
            # which requires a real driver. Stub it so the test can run on
            # CI runners with no GPU.
            patch.object(server_app, "apply_process_cap", lambda **kwargs: None),
        ):

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

    expected_order = [
        "assess_topology",
        "measure_external_vram",
        "enforce_live_budget",
        "load_base_model",
    ]
    actual_order = [c[0] for c in calls]
    assert actual_order == expected_order, (
        f"Lifespan must invoke {expected_order} in order; got {actual_order}"
    )


# ---------------------------------------------------------------------------
# STT / TTS VRAM accounting
# ---------------------------------------------------------------------------


class _FakeSTTConfig:
    """Stand-in for ``paramem.server.config.STTConfig``. Only attributes the
    estimator reads are populated — keeps tests decoupled from the real dataclass."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        device: str = "cuda",
        model: str = "distil-large-v3",
        compute_type: str = "int8",
        cpu_fallback_model: str = "small",
    ):
        self.enabled = enabled
        self.device = device
        self.model = model
        self.compute_type = compute_type
        self.cpu_fallback_model = cpu_fallback_model


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


def test_estimate_stt_bytes_matches_calibrated_table():
    """``estimate_stt_bytes`` must match the calibrated formula: the stt.py
    table entry × compute_type multiplier × calibration factor + headroom.
    Calibration anchors math to the empirical 2026-04-19 GPU probe
    (distil-large-v3 int8 = 960 MiB measured on RTX 5070 Laptop)."""
    from paramem.server.stt import (
        COMPUTE_TYPE_MULTIPLIER,
        VRAM_ESTIMATES_INT8_MB,
    )
    from paramem.server.vram_validator import (
        _STT_CALIBRATION_FACTOR,
        _STT_CALIBRATION_HEADROOM_MB,
        estimate_stt_bytes,
    )

    cfg = _FakeSTTConfig(model="distil-large-v3", compute_type="int8", device="cuda")
    base_mb = VRAM_ESTIMATES_INT8_MB["distil-large-v3"]
    multiplier = COMPUTE_TYPE_MULTIPLIER["int8"]
    expected_mb = int(base_mb * multiplier * _STT_CALIBRATION_FACTOR) + _STT_CALIBRATION_HEADROOM_MB
    assert estimate_stt_bytes(cfg) == expected_mb * 1024 * 1024


def test_estimate_stt_bytes_zero_when_cpu_or_disabled():
    """STT on CPU, disabled, or in cloud-only mode contributes 0 to GPU budget."""
    from paramem.server.vram_validator import estimate_stt_bytes

    assert estimate_stt_bytes(_FakeSTTConfig(enabled=False)) == 0
    assert estimate_stt_bytes(_FakeSTTConfig(device="cpu")) == 0
    assert estimate_stt_bytes(_FakeSTTConfig(device="cuda"), cloud_only=True) == 0


def test_estimate_stt_bytes_scales_with_compute_type():
    """float16 must weigh twice as much as int8 in the budget (after subtracting
    the fixed calibration headroom)."""
    from paramem.server.vram_validator import (
        _STT_CALIBRATION_HEADROOM_MB,
        estimate_stt_bytes,
    )

    int8_bytes = estimate_stt_bytes(_FakeSTTConfig(compute_type="int8"))
    fp16_bytes = estimate_stt_bytes(_FakeSTTConfig(compute_type="float16"))
    assert fp16_bytes > int8_bytes
    # Model-portion (after removing fixed headroom) must double for float16.
    _headroom = _STT_CALIBRATION_HEADROOM_MB * 1024 * 1024
    assert (fp16_bytes - _headroom) >= 2 * (int8_bytes - _headroom) - 1


def test_estimate_stt_bytes_unknown_model_falls_back_with_warning(caplog):
    """Unknown Whisper model must use conservative fallback + warn.

    Uses the same handler-attach pattern as ``tests/test_schema_config.py`` —
    the vram_validator logger is not picked up by the default caplog fixture.
    """
    import logging

    from paramem.server.vram_validator import _STT_FALLBACK_BYTES, estimate_stt_bytes

    named = logging.getLogger("paramem.server.vram_validator")
    orig_propagate = named.propagate
    named.propagate = True
    caplog.set_level(logging.WARNING, logger="paramem.server.vram_validator")
    named.addHandler(caplog.handler)
    try:
        cfg = _FakeSTTConfig(model="totally-made-up-whisper-42")
        result = estimate_stt_bytes(cfg)
    finally:
        named.removeHandler(caplog.handler)
        named.propagate = orig_propagate
    assert result == _STT_FALLBACK_BYTES
    assert "not in VRAM_ESTIMATES_INT8_MB" in caplog.text


def test_estimate_tts_bytes_piper_voices_share_single_ort_context():
    """Multiple Piper voices on GPU should cost N × per-voice + ONE ORT
    context — not N × ORT context. This is the piper-specific sharing that
    justifies the _TTS_PIPER_ORT_CONTEXT_BYTES single-add design."""
    from paramem.server.vram_validator import (
        _TTS_PIPER_BYTES_PER_VOICE,
        _TTS_PIPER_ORT_CONTEXT_BYTES,
        estimate_tts_bytes,
    )

    cfg_one = _FakeTTSConfig(
        voices={"en": _FakeTTSVoice(engine="piper", model="en_US-lessac-high")}
    )
    cfg_four = _FakeTTSConfig(
        voices={
            "en": _FakeTTSVoice(engine="piper", model="en_US-lessac-high"),
            "de": _FakeTTSVoice(engine="piper", model="de_DE-thorsten-high"),
            "fr": _FakeTTSVoice(engine="piper", model="fr_FR-siwis-medium"),
            "es": _FakeTTSVoice(engine="piper", model="es_ES-davefx-medium"),
        }
    )
    one = estimate_tts_bytes(cfg_one)
    four = estimate_tts_bytes(cfg_four)
    assert one == _TTS_PIPER_BYTES_PER_VOICE + _TTS_PIPER_ORT_CONTEXT_BYTES
    assert four == 4 * _TTS_PIPER_BYTES_PER_VOICE + _TTS_PIPER_ORT_CONTEXT_BYTES


def test_estimate_tts_bytes_mms_per_voice():
    """MMS-TTS has no shared context — each voice holds its own model."""
    from paramem.server.vram_validator import _TTS_MMS_BYTES_PER_VOICE, estimate_tts_bytes

    cfg = _FakeTTSConfig(
        voices={
            "tl": _FakeTTSVoice(engine="mms", model="facebook/mms-tts-tgl"),
            "sv": _FakeTTSVoice(engine="mms", model="facebook/mms-tts-swe"),
        }
    )
    assert estimate_tts_bytes(cfg) == 2 * _TTS_MMS_BYTES_PER_VOICE


def test_estimate_tts_bytes_voice_device_override_respected():
    """A voice marked device='cpu' must not contribute to the GPU budget even
    when the global tts.device is 'cuda'."""
    from paramem.server.vram_validator import (
        _TTS_PIPER_BYTES_PER_VOICE,
        _TTS_PIPER_ORT_CONTEXT_BYTES,
        estimate_tts_bytes,
    )

    cfg = _FakeTTSConfig(
        device="cuda",
        voices={
            "en": _FakeTTSVoice(engine="piper", device="cpu"),  # forced CPU
            "de": _FakeTTSVoice(engine="piper"),  # inherits cuda
        },
    )
    # Only the "de" voice counts.
    assert estimate_tts_bytes(cfg) == _TTS_PIPER_BYTES_PER_VOICE + _TTS_PIPER_ORT_CONTEXT_BYTES


def test_estimate_tts_bytes_zero_when_disabled_or_cloud_only():
    from paramem.server.vram_validator import estimate_tts_bytes

    cfg = _FakeTTSConfig(
        voices={"en": _FakeTTSVoice(engine="piper")},
    )
    assert estimate_tts_bytes(_FakeTTSConfig(enabled=False)) == 0
    assert estimate_tts_bytes(cfg, cloud_only=True) == 0
    cpu_cfg = _FakeTTSConfig(device="cpu", voices={"en": _FakeTTSVoice(engine="piper")})
    assert estimate_tts_bytes(cpu_cfg) == 0


def test_validate_includes_stt_tts_in_working_set():
    """The combined (STT + TTS) cost must be deducted from the available
    budget. Feeds the same config that would pass without STT/TTS and shows
    that adding a 2-GiB STT + 1-GiB TTS footprint tips it into rejection."""
    model = _mock_model()
    # Baseline budget: base + adapters + headroom ≈ 5.3 GiB on 6.8 GiB effective.
    # Adding 3 GiB of STT+TTS overflows.
    stt_bytes = 2 * _GiB
    tts_bytes = 1 * _GiB

    with patch("paramem.server.vram_validator.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = _BASE_MODEL_BYTES
        mock_torch.cuda.mem_get_info.return_value = (_FREE_VRAM_BYTES, _TOTAL_VRAM_BYTES)
        mock_torch.cuda.get_device_properties.return_value.total_memory = _TOTAL_VRAM_BYTES

        with pytest.raises(ConfigurationError) as exc_info:
            validate_startup_vram(
                model,
                _MISTRAL_ADAPTER,
                max_interim_count=7,
                model_name="mistral",
                main_adapter_count=3,
                stt_bytes=stt_bytes,
                tts_bytes=tts_bytes,
            )
    msg = str(exc_info.value)
    assert "STT (Whisper)" in msg
    assert "TTS voices on GPU" in msg
    # Remediation must offer the STT/TTS knobs when they contribute.
    assert "stt.device" in msg
    assert "tts.device" in msg


def test_validate_breakdown_omits_stt_tts_lines_when_zero(caplog):
    """When STT/TTS bytes are 0 (all CPU), the breakdown must NOT include the
    two extra rows — they'd be misleading."""
    import logging

    named = logging.getLogger("paramem.server.vram_validator")
    orig_propagate = named.propagate
    named.propagate = True
    caplog.set_level(logging.INFO, logger="paramem.server.vram_validator")
    named.addHandler(caplog.handler)

    model = _mock_model()
    try:
        with patch("paramem.server.vram_validator.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.memory_allocated.return_value = _BASE_MODEL_BYTES
            mock_torch.cuda.mem_get_info.return_value = (_FREE_VRAM_BYTES, _TOTAL_VRAM_BYTES)
            mock_torch.cuda.get_device_properties.return_value.total_memory = _TOTAL_VRAM_BYTES

            validate_startup_vram(
                model,
                _MISTRAL_ADAPTER,
                max_interim_count=7,
                model_name="mistral",
                main_adapter_count=3,
                stt_bytes=0,
                tts_bytes=0,
            )
    finally:
        named.removeHandler(caplog.handler)
        named.propagate = orig_propagate

    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "VRAM check passed" in logged
    assert "STT (Whisper)" not in logged
    assert "TTS voices on GPU" not in logged


# ---------------------------------------------------------------------------
# measure_external_vram — device-wide measurement via nvidia-smi
# ---------------------------------------------------------------------------


def test_measure_external_vram_uses_nvidia_smi():
    """``measure_external_vram`` must read device-wide memory via nvidia-smi,
    not per-process via ``torch.cuda.memory_allocated``. Regression for the
    blind spot that triggered the 2026-04-26 WSL crash: a process holding
    ~4.5 GiB on the GPU was invisible to ``torch.cuda.memory_allocated``
    (which only sees the calling process's allocations)."""
    from paramem.server.vram_validator import measure_external_vram

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
        # If the implementation falls back, it would call this — must NOT.
        mock_torch.cuda.memory_allocated.return_value = 0

        total, external = measure_external_vram()

    assert total == _TOTAL_VRAM_BYTES
    assert external == 4608 * 2**20
    # nvidia-smi was the source of truth; torch.cuda.memory_allocated was not consulted.
    mock_run.assert_called_once()
    mock_torch.cuda.memory_allocated.assert_not_called()


def test_measure_external_vram_falls_back_when_nvidia_smi_missing(caplog):
    """When ``nvidia-smi`` is unavailable, ``measure_external_vram`` falls back
    to ``torch.cuda.memory_allocated`` and logs a WARNING. The fallback path
    matches the prior (buggy) behavior; the warning makes the blind spot
    visible to the operator instead of letting it silently admit an
    unfittable topology."""
    import logging

    from paramem.server.vram_validator import measure_external_vram

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


def test_enforce_live_budget_rejects_when_external_consumes_majority():
    """Regression for the 2026-04-26 crash: with 4.5 GiB external and 7.5 GiB
    required on an 8 GiB device, live budget = 8 − 4.5 = 3.5 GiB < 7.5 GiB
    required, and ``enforce_live_budget`` must raise. Pre-fix, the bug
    silently passed because ``torch.cuda.memory_allocated`` reported 0 for
    a foreign process and live budget = 8 − 0 = 8 GiB > 7.5 GiB."""
    from paramem.server.vram_validator import (
        TopologyAssessment,
        enforce_live_budget,
    )

    total_memory = 8 * _GiB
    external = int(4.5 * _GiB)
    required = int(7.5 * _GiB)

    # Minimal TopologyAssessment fixture — enforce_live_budget reads
    # required_bytes and breakdown for the error message.
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
    assert "Deficit" in msg
