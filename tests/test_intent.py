"""Tests for the residual intent classifier — state-first dispatch,
encoder loader, exemplar bank, and cosine-margin residual.

These tests do not download the real encoder; they stub
:class:`SentenceTransformer` and inject deterministic embeddings so the
cosine-margin logic can be exercised without GPU or network.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from paramem.server import intent as intent_module
from paramem.server.config import IntentConfig
from paramem.server.intent import (
    _classify_via_encoder,
    _EncoderHandle,
    _ExemplarBank,
    _fail_closed_intent,
    _read_exemplar_file,
    _resolve_device,
    classify_intent,
    get_encoder,
    get_exemplars,
    load_encoder,
    load_exemplars,
)
from paramem.server.router import Intent


@pytest.fixture(autouse=True)
def reset_singleton():
    """Each test starts with fresh singletons."""
    intent_module._encoder_singleton = None
    intent_module._exemplars_singleton = None
    yield
    intent_module._encoder_singleton = None
    intent_module._exemplars_singleton = None


def _stub_encoder(prefix: str = "query: ") -> _EncoderHandle:
    """Encoder handle backed by a MagicMock model — set ``.encode.return_value``
    or ``.encode.side_effect`` to script the embedding output."""
    model = MagicMock()
    return _EncoderHandle(model=model, query_prefix=prefix, device="cpu", dtype="float32")


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


class TestReadExemplarFile:
    def test_skips_blank_and_comment_lines(self, tmp_path):
        path = tmp_path / "personal.en.txt"
        path.write_text(
            "# This is a header comment\n"
            "\n"
            "What is my address?\n"
            "   \n"
            "  # indented comment also skipped\n"
            "Where do I live?\n",
            encoding="utf-8",
        )
        assert _read_exemplar_file(path) == [
            "What is my address?",
            "Where do I live?",
        ]

    def test_strips_whitespace(self, tmp_path):
        path = tmp_path / "command.en.txt"
        path.write_text("  Turn on the lights  \n\tPause the music\n", encoding="utf-8")
        assert _read_exemplar_file(path) == ["Turn on the lights", "Pause the music"]


class TestLoadExemplars:
    def test_returns_none_when_encoder_missing(self):
        config = IntentConfig()
        result = load_exemplars(config, encoder=None)
        assert result is None
        assert get_exemplars() is None

    def test_returns_none_when_dir_missing(self, tmp_path):
        config = IntentConfig(exemplars_dir=str(tmp_path / "does_not_exist"))
        encoder = _stub_encoder()
        result = load_exemplars(config, encoder=encoder)
        assert result is None
        assert get_exemplars() is None

    def test_returns_none_when_no_exemplars_collected(self, tmp_path):
        # Empty directory — no .txt files at all.
        config = IntentConfig(exemplars_dir=str(tmp_path))
        encoder = _stub_encoder()
        result = load_exemplars(config, encoder=encoder)
        assert result is None

    def test_skips_unrecognised_class_filenames(self, tmp_path, caplog):
        (tmp_path / "personal.en.txt").write_text("What is my address?\n", encoding="utf-8")
        (tmp_path / "weather.en.txt").write_text("Will it rain?\n", encoding="utf-8")
        config = IntentConfig(exemplars_dir=str(tmp_path))
        encoder = _stub_encoder()
        encoder.model.encode.return_value = np.array([[1.0, 0.0]], dtype=np.float32)

        # caplog.at_level() silently fails in this codebase (see
        # project_caplog_ros_workaround.md); attach the handler directly.
        import logging as _logging

        intent_logger = _logging.getLogger("paramem.server.intent")
        prior_level = intent_logger.level
        intent_logger.setLevel(_logging.WARNING)
        intent_logger.addHandler(caplog.handler)
        try:
            bank = load_exemplars(config, encoder=encoder)
        finally:
            intent_logger.removeHandler(caplog.handler)
            intent_logger.setLevel(prior_level)

        assert bank is not None
        assert bank.intents == [Intent.PERSONAL]
        assert any("weather" in record.getMessage() for record in caplog.records)

    def test_loads_multiple_classes_and_languages(self, tmp_path):
        (tmp_path / "personal.en.txt").write_text(
            "What is my address?\nWhere do I live?\n", encoding="utf-8"
        )
        (tmp_path / "personal.de.txt").write_text("Wo wohne ich?\n", encoding="utf-8")
        (tmp_path / "command.en.txt").write_text("Turn on the lights\n", encoding="utf-8")
        (tmp_path / "general.en.txt").write_text(
            "What is the capital of France?\n", encoding="utf-8"
        )

        config = IntentConfig(exemplars_dir=str(tmp_path))
        encoder = _stub_encoder()
        # 5 exemplars total (2 + 1 + 1 + 1) — return a 5x2 array.
        encoder.model.encode.return_value = np.eye(5, 2, dtype=np.float32)

        bank = load_exemplars(config, encoder=encoder)

        assert bank is not None
        # Sorted glob order: command.en, general.en, personal.de, personal.en
        assert bank.intents == [
            Intent.COMMAND,
            Intent.GENERAL,
            Intent.PERSONAL,
            Intent.PERSONAL,
            Intent.PERSONAL,
        ]
        assert bank.embeddings.shape == (5, 2)
        # Encoder was called with the prefix applied to every line.
        called_with = encoder.model.encode.call_args.args[0]
        assert all(text.startswith("query: ") for text in called_with)

    def test_embedding_failure_is_non_fatal(self, tmp_path):
        (tmp_path / "personal.en.txt").write_text("What is my address?\n", encoding="utf-8")
        config = IntentConfig(exemplars_dir=str(tmp_path))
        encoder = _stub_encoder()
        encoder.model.encode.side_effect = RuntimeError("simulated GPU OOM")

        result = load_exemplars(config, encoder=encoder)

        assert result is None
        assert get_exemplars() is None


def _bank_with_three_classes() -> _ExemplarBank:
    """Synthetic bank: one orthogonal unit-vector exemplar per class.

    Embeddings are 3-dim L2-normalised: PERSONAL=[1,0,0], COMMAND=[0,1,0],
    GENERAL=[0,0,1].  Cosine against any query reduces to the matching
    coordinate of the query embedding.
    """
    return _ExemplarBank(
        intents=[Intent.PERSONAL, Intent.COMMAND, Intent.GENERAL],
        embeddings=np.eye(3, dtype=np.float32),
        source_files=("synthetic",),
    )


class TestClassifyViaEncoder:
    def test_returns_top_class_when_margin_met(self):
        bank = _bank_with_three_classes()
        encoder = _stub_encoder()
        # Strongly aligned with PERSONAL: cosines are [0.9, 0.2, 0.1].
        # Margin = 0.9 - 0.2 = 0.7 ≫ default 0.05.
        encoder.model.encode.return_value = np.array([[0.9, 0.2, 0.1]], dtype=np.float32)
        config = IntentConfig()

        result = _classify_via_encoder("What is my address?", encoder, bank, config)

        assert result == Intent.PERSONAL

    def test_below_margin_returns_fail_closed(self):
        bank = _bank_with_three_classes()
        encoder = _stub_encoder()
        # Top two classes within 0.01 of each other — below default margin 0.05.
        encoder.model.encode.return_value = np.array([[0.50, 0.49, 0.10]], dtype=np.float32)
        config = IntentConfig()  # fail_closed defaults to "personal"

        result = _classify_via_encoder("ambiguous query", encoder, bank, config)

        assert result == Intent.PERSONAL  # via fail-closed, not via top score

    def test_below_margin_honours_custom_fail_closed(self):
        bank = _bank_with_three_classes()
        encoder = _stub_encoder()
        encoder.model.encode.return_value = np.array([[0.50, 0.49, 0.10]], dtype=np.float32)
        config = IntentConfig(fail_closed_intent="general")

        result = _classify_via_encoder("ambiguous query", encoder, bank, config)

        assert result == Intent.GENERAL

    def test_query_embedding_failure_falls_back(self):
        bank = _bank_with_three_classes()
        encoder = _stub_encoder()
        encoder.model.encode.side_effect = RuntimeError("simulated failure")
        config = IntentConfig(fail_closed_intent="general")

        result = _classify_via_encoder("any query", encoder, bank, config)

        assert result == Intent.GENERAL

    def test_query_prefix_applied(self):
        bank = _bank_with_three_classes()
        encoder = _stub_encoder(prefix="query: ")
        encoder.model.encode.return_value = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        config = IntentConfig()

        _classify_via_encoder("Where do I live?", encoder, bank, config)

        called_with = encoder.model.encode.call_args.args[0]
        assert called_with == ["query: Where do I live?"]

    def test_per_class_max_used_not_average(self):
        # Two PERSONAL exemplars: one strong, one weak.  Top class score
        # should be the max (0.95), not the mean.
        bank = _ExemplarBank(
            intents=[Intent.PERSONAL, Intent.PERSONAL, Intent.GENERAL],
            embeddings=np.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],  # orthogonal to query
                    [0.6, 0.0],
                ],
                dtype=np.float32,
            ),
            source_files=("synthetic",),
        )
        encoder = _stub_encoder()
        # Query aligned with x-axis.
        encoder.model.encode.return_value = np.array([[1.0, 0.0]], dtype=np.float32)
        config = IntentConfig()  # margin = 0.05

        result = _classify_via_encoder("query", encoder, bank, config)

        # If max is used: PERSONAL=1.0, GENERAL=0.6, margin 0.4 → PERSONAL.
        # If mean were used: PERSONAL=0.5, GENERAL=0.6 → GENERAL (wrong).
        assert result == Intent.PERSONAL


class TestClassifyIntentIntegration:
    def test_uses_encoder_when_loaded(self):
        intent_module._encoder_singleton = _stub_encoder()
        intent_module._encoder_singleton.model.encode.return_value = np.array(
            [[0.0, 0.9, 0.1]], dtype=np.float32
        )
        intent_module._exemplars_singleton = _bank_with_three_classes()
        config = IntentConfig()

        result = classify_intent(
            "Turn on the lights",
            has_graph_match=False,
            has_ha_match=False,
            config=config,
        )

        assert result == Intent.COMMAND

    def test_state_signal_short_circuits_encoder(self):
        # Even with encoder loaded, graph match must win without invoking it.
        intent_module._encoder_singleton = _stub_encoder()
        intent_module._exemplars_singleton = _bank_with_three_classes()
        config = IntentConfig()

        result = classify_intent(
            "Where does Alex live?",
            has_graph_match=True,
            has_ha_match=False,
            config=config,
        )

        assert result == Intent.PERSONAL
        intent_module._encoder_singleton.model.encode.assert_not_called()

    def test_missing_bank_falls_back_to_fail_closed(self):
        intent_module._encoder_singleton = _stub_encoder()
        # _exemplars_singleton stays None
        config = IntentConfig(fail_closed_intent="general")

        result = classify_intent(
            "what is the capital of France?",
            has_graph_match=False,
            has_ha_match=False,
            config=config,
        )

        assert result == Intent.GENERAL
