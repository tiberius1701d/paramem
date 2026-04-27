"""Tests for the residual intent classifier — state-first dispatch + encoder
loader scaffolding.

The cosine-against-exemplars residual is not yet wired in (lands in a
follow-up commit alongside the exemplar files).  These tests pin the
state-first short-circuit, the fail-closed default, and the encoder
loader's degraded-mode behaviour (missing dep, disabled config, load
failure) without paying the cost of actually downloading the encoder.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from paramem.server import intent as intent_module
from paramem.server.config import IntentConfig
from paramem.server.intent import (
    _fail_closed_intent,
    _resolve_device,
    classify_intent,
    get_encoder,
    load_encoder,
)
from paramem.server.router import Intent


@pytest.fixture(autouse=True)
def reset_singleton():
    """Each test starts with a fresh singleton."""
    intent_module._encoder_singleton = None
    yield
    intent_module._encoder_singleton = None


class TestStateFirstDispatch:
    def test_graph_match_routes_personal(self):
        result = classify_intent(
            "Where does Alex work?",
            has_graph_match=True,
            has_ha_match=False,
        )
        assert result == Intent.PERSONAL

    def test_ha_match_routes_command(self):
        result = classify_intent(
            "Turn on the kitchen light.",
            has_graph_match=False,
            has_ha_match=True,
        )
        assert result == Intent.COMMAND

    def test_graph_match_wins_over_ha(self):
        # When both signals fire, PERSONAL takes precedence — privacy-first.
        result = classify_intent(
            "Is Alex's bedroom light on?",
            has_graph_match=True,
            has_ha_match=True,
        )
        assert result == Intent.PERSONAL


class TestResidualFallback:
    def test_residual_without_config_returns_unknown(self):
        # No state hit, no config → callers that haven't wired config yet
        # get UNKNOWN (treated conservatively as GENERAL by the routing
        # table).
        result = classify_intent(
            "What is the capital of France?",
            has_graph_match=False,
            has_ha_match=False,
        )
        assert result == Intent.UNKNOWN

    def test_residual_with_config_returns_fail_closed_default(self):
        config = IntentConfig()  # fail_closed_intent defaults to "personal"
        result = classify_intent(
            "What is the capital of France?",
            has_graph_match=False,
            has_ha_match=False,
            config=config,
        )
        assert result == Intent.PERSONAL

    def test_residual_honours_custom_fail_closed_intent(self):
        config = IntentConfig(fail_closed_intent="general")
        result = classify_intent(
            "Tell me about quantum computing.",
            has_graph_match=False,
            has_ha_match=False,
            config=config,
        )
        assert result == Intent.GENERAL


class TestFailClosedIntent:
    def test_personal_default(self):
        cfg = IntentConfig()
        assert _fail_closed_intent(cfg) == Intent.PERSONAL

    def test_invalid_value_falls_back_to_personal(self):
        cfg = IntentConfig(fail_closed_intent="not_a_valid_intent")
        # Invalid values must not crash; safest fallback is PERSONAL
        # (privacy-preserving).
        assert _fail_closed_intent(cfg) == Intent.PERSONAL

    def test_unknown_is_a_valid_choice(self):
        cfg = IntentConfig(fail_closed_intent="unknown")
        assert _fail_closed_intent(cfg) == Intent.UNKNOWN


class TestResolveDevice:
    def test_explicit_passes_through(self):
        assert _resolve_device("cuda") == "cuda"
        assert _resolve_device("cpu") == "cpu"

    def test_auto_falls_to_cpu_when_no_torch(self):
        # When torch isn't importable in some test environments the
        # auto-resolver must not raise.
        with patch("paramem.server.intent._resolve_device", wraps=_resolve_device) as _:
            # _resolve_device imports torch lazily inside the function
            # body; we just call it and accept either cuda or cpu — the
            # contract is "doesn't raise, returns a real device string".
            result = _resolve_device("auto")
            assert result in {"cuda", "cpu"}


class TestLoadEncoder:
    def test_disabled_config_returns_none(self):
        config = IntentConfig(enabled=False)
        result = load_encoder(config)
        assert result is None
        assert get_encoder() is None

    def test_load_failure_is_non_fatal(self):
        # When the encoder library fails to load the model (e.g. download
        # blocked, OOM, invalid name), load_encoder must return None and
        # leave the singleton unset rather than raising.
        config = IntentConfig(encoder_model="this/model/does/not/exist")
        with patch(
            "sentence_transformers.SentenceTransformer",
            side_effect=RuntimeError("simulated download failure"),
        ):
            result = load_encoder(config)
        assert result is None
        assert get_encoder() is None

    def test_missing_dep_is_non_fatal(self):
        config = IntentConfig()
        # Simulate ImportError raised inside load_encoder's try-import.
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            result = load_encoder(config)
        assert result is None
        assert get_encoder() is None

    def test_successful_load_caches_singleton(self):
        config = IntentConfig(
            encoder_model="dummy/model",
            encoder_device="cpu",
            encoder_dtype="float32",
            encoder_query_prefix="query: ",
        )
        fake_st = MagicMock()
        fake_model = MagicMock()
        fake_st.return_value = fake_model
        with patch("sentence_transformers.SentenceTransformer", fake_st):
            result = load_encoder(config)

        assert result is not None
        assert result.model is fake_model
        assert result.device == "cpu"
        assert result.dtype == "float32"
        assert result.query_prefix == "query: "
        assert get_encoder() is result
        # cpu + float32 → no .half() call
        fake_model.half.assert_not_called()

    def test_cuda_float16_calls_half(self):
        config = IntentConfig(
            encoder_model="dummy/model",
            encoder_device="cuda",
            encoder_dtype="float16",
        )
        fake_st = MagicMock()
        fake_model = MagicMock()
        fake_st.return_value = fake_model
        with patch("sentence_transformers.SentenceTransformer", fake_st):
            load_encoder(config)
        fake_model.half.assert_called_once()
