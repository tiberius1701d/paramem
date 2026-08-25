"""``SpanTaggerConfig`` defaults, refusals, and its presence in both
shipped YAMLs (the fixture and the example — see the module docstring's
allowlist note below).
"""

from __future__ import annotations

import pytest

from paramem.server.config import SpanTaggerConfig, load_server_config


class TestSpanTaggerConfigDefaults:
    def test_defaults_match_the_shipped_values(self) -> None:
        cfg = SpanTaggerConfig()
        assert cfg.checkpoint == "urchade/gliner_multi_pii-v1"
        assert cfg.revision == "1fcf13e85f4eef5394e1fcd406cf2ca9ea82351d"
        assert cfg.score_threshold == 0.5
        assert cfg.threads == 8


class TestSpanTaggerConfigRefusals:
    def test_empty_checkpoint_raises(self) -> None:
        with pytest.raises(ValueError, match="checkpoint"):
            SpanTaggerConfig(checkpoint="")

    def test_empty_revision_raises(self) -> None:
        with pytest.raises(ValueError, match="revision"):
            SpanTaggerConfig(revision="")

    def test_score_threshold_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="score_threshold"):
            SpanTaggerConfig(score_threshold=0.0)

    def test_score_threshold_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="score_threshold"):
            SpanTaggerConfig(score_threshold=1.5)

    def test_score_threshold_one_is_accepted(self) -> None:
        # (0.0, 1.0] is inclusive at the top.
        SpanTaggerConfig(score_threshold=1.0)

    def test_threads_below_one_raises(self) -> None:
        with pytest.raises(ValueError, match="threads"):
            SpanTaggerConfig(threads=0)

    def test_threads_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="threads"):
            SpanTaggerConfig(threads=-1)


class TestSpanTaggerConfigParsesFromBothYamls:
    def test_parses_from_the_fixture_yaml(self) -> None:
        config = load_server_config("tests/fixtures/server.yaml")
        assert isinstance(config.span_tagger, SpanTaggerConfig)
        assert config.span_tagger.checkpoint
        assert config.span_tagger.revision
        assert 0.0 < config.span_tagger.score_threshold <= 1.0
        assert config.span_tagger.threads >= 1

    def test_parses_from_the_shipped_example_yaml(self) -> None:
        # ALLOWLISTED verification of configs/server.yaml.example, per this
        # task's brief — see the BLOCKER reported for
        # tests/test_test_config_loader_usage.py::EXAMPLE_VERIFY_ALLOWLIST.
        config = load_server_config("configs/server.yaml.example")
        assert isinstance(config.span_tagger, SpanTaggerConfig)
        assert config.span_tagger.checkpoint == "urchade/gliner_multi_pii-v1"
        assert config.span_tagger.revision == "1fcf13e85f4eef5394e1fcd406cf2ca9ea82351d"
        assert config.span_tagger.score_threshold == 0.5
        assert config.span_tagger.threads == 8
