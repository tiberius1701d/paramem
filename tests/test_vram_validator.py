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
from unittest.mock import patch

import pytest

from paramem.server.vram_validator import (
    _FALLBACK_BASE_BYTES,
    _LORA_DTYPE_BYTES,
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

# Free VRAM available: 7.5 GiB (generous headroom for these tests)
_FREE_VRAM_BYTES = int(7.5 * _GiB)

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
        mock_torch.cuda.mem_get_info.return_value = (_FREE_VRAM_BYTES, int(8 * _GiB))

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
        mock_torch.cuda.mem_get_info.return_value = (_FREE_VRAM_BYTES, int(8 * _GiB))

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
    """rank=128 adapter with 4 modules, 7 sessions, and 7.5 GiB free should fail.

    The error message must reference ``rank``.
    """
    high_rank_adapter = AdapterConfig(
        rank=128,
        alpha=256,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        dropout=0.0,
    )
    model = _mock_model()

    with (
        patch("paramem.server.vram_validator.torch") as mock_torch,
    ):
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = _BASE_MODEL_BYTES
        mock_torch.cuda.mem_get_info.return_value = (_FREE_VRAM_BYTES, int(8 * _GiB))

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
        mock_torch.cuda.mem_get_info.return_value = (_TINY_VRAM_BYTES, int(8 * _GiB))

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
    expected = 4 * 32 * 2 * 8 * 4096 * _LORA_DTYPE_BYTES
    result = estimated_adapter_bytes(_MISTRAL_ADAPTER, hidden_size=4096, num_layers=32)
    assert result == expected, (
        f"estimated_adapter_bytes mismatch: expected {expected}, got {result}"
    )


# ---------------------------------------------------------------------------
# Additional: vram_cap_gib override
# ---------------------------------------------------------------------------


def test_vram_cap_gib_override_passes():
    """When vram_cap_gib is provided, it overrides torch.cuda.mem_get_info."""
    model = _mock_model()

    with patch("paramem.server.vram_validator.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = _BASE_MODEL_BYTES
        # mem_get_info should NOT be called when cap is overridden
        mock_torch.cuda.mem_get_info.side_effect = AssertionError("Should not be called")

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
    without querying torch.cuda.mem_get_info."""
    model = _mock_model()

    with patch("paramem.server.vram_validator.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = _BASE_MODEL_BYTES
        mock_torch.cuda.mem_get_info.side_effect = AssertionError("Should not be called")

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
        mock_torch.cuda.mem_get_info.return_value = (_TINY_VRAM_BYTES, int(8 * _GiB))

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
        mock_torch.cuda.mem_get_info.return_value = (_TINY_VRAM_BYTES, int(8 * _GiB))

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
            mock_torch.cuda.mem_get_info.return_value = (_FREE_VRAM_BYTES, int(8 * _GiB))

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
            mock_torch.cuda.mem_get_info.return_value = (_FREE_VRAM_BYTES, int(8 * _GiB))

            # Use empty model_name to trigger unknown-model path
            validate_startup_vram(
                model,
                _MISTRAL_ADAPTER,
                max_interim_count=7,
                model_name="",
                main_adapter_count=3,
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
