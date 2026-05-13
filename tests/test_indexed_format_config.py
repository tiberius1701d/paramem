"""Config round-trip tests for indexed_format.

Covers:
- load_server_config("tests/fixtures/server.yaml") → indexed_format == "qa"
  (default value in the fixture).
- indexed_format threads from ServerConfig.consolidation.indexed_format →
  ServerConfig.consolidation_config (ConsolidationConfig) property.
- ConsolidationConfig.indexed_format survives the round-trip (QA and quad).
- An invalid indexed_format value raises ValueError when constructing
  ConsolidationScheduleConfig (__post_init__ validation).
- ConsolidationConfig default is "qa".

This test does NOT touch configs/server.yaml (gitignored operator file).
"""

from __future__ import annotations

import pytest

from paramem.utils.config import ConsolidationConfig

# ---------------------------------------------------------------------------
# Tests: ConsolidationConfig standalone
# ---------------------------------------------------------------------------


class TestConsolidationConfigDefault:
    def test_default_is_qa(self) -> None:
        cfg = ConsolidationConfig()
        assert cfg.indexed_format == "qa"

    def test_explicit_qa(self) -> None:
        cfg = ConsolidationConfig(indexed_format="qa")
        assert cfg.indexed_format == "qa"

    def test_explicit_quad(self) -> None:
        cfg = ConsolidationConfig(indexed_format="quad")
        assert cfg.indexed_format == "quad"


# ---------------------------------------------------------------------------
# Tests: load_server_config → indexed_format in ServerConfig
# ---------------------------------------------------------------------------


class TestServerConfigIndexedFormat:
    def test_fixture_indexed_format_is_qa(self) -> None:
        """tests/fixtures/server.yaml has indexed_format: qa."""
        from paramem.server.config import load_server_config

        cfg = load_server_config("tests/fixtures/server.yaml")
        assert cfg.consolidation.indexed_format == "qa"

    def test_consolidation_config_property_threads_indexed_format(self) -> None:
        """consolidation_config property passes indexed_format to ConsolidationConfig."""
        from paramem.server.config import load_server_config

        cfg = load_server_config("tests/fixtures/server.yaml")
        cons_cfg = cfg.consolidation_config
        assert isinstance(cons_cfg, ConsolidationConfig)
        assert cons_cfg.indexed_format == "qa"

    def test_consolidation_config_property_quad_when_set(self) -> None:
        """Setting consolidation.indexed_format='quad' threads through the property."""
        from paramem.server.config import load_server_config

        cfg = load_server_config("tests/fixtures/server.yaml")
        # Mutate in code (per CLAUDE.md pattern — per-test value overrides go in code).
        cfg.consolidation.indexed_format = "quad"
        cons_cfg = cfg.consolidation_config
        assert cons_cfg.indexed_format == "quad"


# ---------------------------------------------------------------------------
# Tests: ConsolidationScheduleConfig validation
# ---------------------------------------------------------------------------


class TestConsolidationScheduleConfigValidation:
    def test_invalid_indexed_format_raises_value_error(self) -> None:
        """An invalid indexed_format value must raise ValueError.

        The dataclass __post_init__ runs automatically on construction, so
        the ValueError is raised at ConsolidationScheduleConfig() call time.
        """
        from paramem.server.config import ConsolidationScheduleConfig

        with pytest.raises(ValueError, match="indexed_format"):
            ConsolidationScheduleConfig(indexed_format="invalid_value")

    def test_valid_qa_does_not_raise(self) -> None:
        """indexed_format='qa' is valid — no exception."""
        from paramem.server.config import ConsolidationScheduleConfig

        # Must not raise
        ConsolidationScheduleConfig(indexed_format="qa")

    def test_valid_quad_does_not_raise(self) -> None:
        """indexed_format='quad' is valid — no exception."""
        from paramem.server.config import ConsolidationScheduleConfig

        # Must not raise
        ConsolidationScheduleConfig(indexed_format="quad")

    def test_simulate_mode_with_qa_format_raises_value_error(self) -> None:
        """mode='simulate' + indexed_format='qa' must raise ValueError at construction.

        Simulate-mode persistence uses graph.json which only carries the quad
        shape.  A QA-format simulate cycle would KeyError on the second run
        (iter_quads is quad-only; qa["question"] does not exist on the quad
        path).  The validator blocks this at config-load time with a clear
        message so the operator sees the problem before any cycle runs.

        This is the C1 fix from the chunk-3 code review: the validator was
        present in config.py but had no test, so regressions (removing the
        validator) would be silent.
        """
        from paramem.server.config import ConsolidationScheduleConfig

        with pytest.raises(ValueError, match="mode='simulate' requires indexed_format='quad'"):
            ConsolidationScheduleConfig(mode="simulate", indexed_format="qa")

    def test_simulate_mode_with_quad_format_is_valid(self) -> None:
        """mode='simulate' + indexed_format='quad' is the only valid simulate combo."""
        from paramem.server.config import ConsolidationScheduleConfig

        # Must not raise — this is the only valid simulate-mode combination.
        ConsolidationScheduleConfig(mode="simulate", indexed_format="quad")

    def test_train_mode_with_qa_format_is_valid(self) -> None:
        """mode='train' + indexed_format='qa' is the default train-mode combination."""
        from paramem.server.config import ConsolidationScheduleConfig

        # Must not raise — this is the production default.
        ConsolidationScheduleConfig(mode="train", indexed_format="qa")

    def test_train_mode_with_quad_format_is_valid(self) -> None:
        """mode='train' + indexed_format='quad' is valid (quadruple-format train mode)."""
        from paramem.server.config import ConsolidationScheduleConfig

        # Must not raise — quad train mode is supported.
        ConsolidationScheduleConfig(mode="train", indexed_format="quad")
