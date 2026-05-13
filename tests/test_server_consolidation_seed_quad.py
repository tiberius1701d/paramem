"""Unit tests for the server/consolidation.py seed-on-startup and write paths
in quad (indexed_format="quad") mode.

Covers:
- create_consolidation_loop with indexed_format="quad" uses read_keyed_pairs_quad
  to seed all three tiers (episodic, semantic, procedural).
- create_consolidation_loop with indexed_format="qa" uses read_keyed_pairs.
- _write_keyed_pairs(indexed_format="quad") dispatches to write_keyed_pairs_quad.
- _write_keyed_pairs(indexed_format="qa") dispatches to write_keyed_pairs.
- _save_keyed_pairs_for_router passes indexed_format to all three _write_keyed_pairs calls.

No GPU required — ConsolidationLoop instantiation and model loading are mocked.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from paramem.server.consolidation import _write_keyed_pairs

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_minimal_server_config(tmp_path: Path, *, indexed_format: str = "qa") -> MagicMock:
    """Return a minimal MagicMock with the attributes create_consolidation_loop reads."""
    cfg = MagicMock()

    # paths
    cfg.adapter_dir = tmp_path / "adapters"
    cfg.simulate_dir = tmp_path / "simulate"
    cfg.debug_dir = tmp_path / "debug"
    cfg.key_metadata_path = tmp_path / "key_metadata.json"
    cfg.debug = False

    # consolidation schedule
    cfg.consolidation.mode = "train"
    cfg.consolidation.indexed_format = indexed_format
    cfg.consolidation.consolidation_period_string = "84h"

    return cfg


def _write_quad_keyed_pairs(path: Path, pairs: list[dict]) -> None:
    """Write a minimal quad keyed_pairs.json to a tmp path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pairs))


# ---------------------------------------------------------------------------
# Tests: _write_keyed_pairs format dispatch
# ---------------------------------------------------------------------------


class TestWriteKeyedPairsDispatch:
    def test_qa_mode_calls_write_keyed_pairs(self, tmp_path: Path) -> None:
        qa_cache = {
            "graph1": {
                "key": "graph1",
                "question": "Q?",
                "answer": "A.",
                "source_subject": "Alice",
                "source_predicate": "lives_in",
                "source_object": "Berlin",
                "speaker_id": "Speaker0",
                "first_seen_cycle": 1,
            }
        }
        simhash = {"graph1": 12345}
        dest = tmp_path / "keyed_pairs.json"

        with (
            patch("paramem.training.keyed_pairs_io.write_keyed_pairs") as mock_qa,
            patch("paramem.training.keyed_pairs_io.write_keyed_pairs_quad") as mock_quad,
        ):
            _write_keyed_pairs(qa_cache, simhash, dest, indexed_format="qa")

        mock_qa.assert_called_once()
        mock_quad.assert_not_called()

    def test_quad_mode_calls_write_keyed_pairs_quad(self, tmp_path: Path) -> None:
        quad_cache = {
            "graph1": {
                "key": "graph1",
                "subject": "Alice",
                "predicate": "lives_in",
                "object": "Berlin",
                "source_subject": "Alice",
                "source_predicate": "lives_in",
                "source_object": "Berlin",
                "speaker_id": "Speaker0",
                "first_seen_cycle": 1,
            }
        }
        simhash = {"graph1": 12345}
        dest = tmp_path / "keyed_pairs.json"

        with (
            patch("paramem.training.keyed_pairs_io.write_keyed_pairs") as mock_qa,
            patch("paramem.training.keyed_pairs_io.write_keyed_pairs_quad") as mock_quad,
        ):
            _write_keyed_pairs(quad_cache, simhash, dest, indexed_format="quad")

        mock_quad.assert_called_once()
        mock_qa.assert_not_called()

    def test_default_indexed_format_is_qa(self, tmp_path: Path) -> None:
        """No indexed_format arg → QA writer selected."""
        qa_cache = {
            "graph1": {
                "key": "graph1",
                "question": "Q?",
                "answer": "A.",
                "source_subject": "A",
                "source_predicate": "p",
                "source_object": "B",
                "speaker_id": "",
                "first_seen_cycle": 0,
            }
        }
        simhash = {"graph1": 99}
        dest = tmp_path / "keyed_pairs.json"

        with (
            patch("paramem.training.keyed_pairs_io.write_keyed_pairs") as mock_qa,
            patch("paramem.training.keyed_pairs_io.write_keyed_pairs_quad") as mock_quad,
        ):
            _write_keyed_pairs(qa_cache, simhash, dest)

        mock_qa.assert_called_once()
        mock_quad.assert_not_called()

    def test_only_keys_in_simhash_included(self, tmp_path: Path) -> None:
        """Only cache entries present in simhash_registry reach the writer."""
        cache = {
            "graph1": {"key": "graph1", "question": "Q1?", "answer": "A1."},
            "graph2": {"key": "graph2", "question": "Q2?", "answer": "A2."},
            "graph3": {"key": "graph3", "question": "Q3?", "answer": "A3."},
        }
        simhash = {"graph1": 1, "graph3": 3}  # graph2 excluded
        dest = tmp_path / "keyed_pairs.json"
        written: list = []

        def _capture(path, pairs):
            written.extend(pairs)

        with patch("paramem.training.keyed_pairs_io.write_keyed_pairs", side_effect=_capture):
            _write_keyed_pairs(cache, simhash, dest, indexed_format="qa")

        keys = [p["key"] for p in written]
        assert "graph1" in keys
        assert "graph3" in keys
        assert "graph2" not in keys


# ---------------------------------------------------------------------------
# Tests: _save_keyed_pairs_for_router passes indexed_format
# ---------------------------------------------------------------------------


class TestSaveKeyedPairsForRouterFormat:
    def test_quad_mode_passes_indexed_format_to_all_writes(self, tmp_path: Path) -> None:
        """_save_keyed_pairs_for_router passes indexed_format="quad" to all three calls."""
        from paramem.server.consolidation import _save_keyed_pairs_for_router

        store_dir = tmp_path / "adapters"
        cfg = MagicMock()
        cfg.consolidation.mode = "train"
        cfg.consolidation.indexed_format = "quad"
        cfg.adapter_dir = store_dir
        cfg.simulate_dir = tmp_path / "simulate"

        loop = MagicMock()
        loop.indexed_key_cache = {
            "graph1": {
                "key": "graph1",
                "subject": "A",
                "predicate": "p",
                "object": "B",
                "source_subject": "A",
                "source_predicate": "p",
                "source_object": "B",
                "speaker_id": "",
                "first_seen_cycle": 1,
            }
        }
        loop.episodic_simhash = {"graph1": 1}
        loop.semantic_simhash = {"graph1": 1}
        loop.procedural_simhash = {"graph1": 1}

        format_args: list[str] = []

        def _capture(*args, indexed_format: str = "qa", **kwargs):
            format_args.append(indexed_format)

        with patch(
            "paramem.server.consolidation._write_keyed_pairs",
            side_effect=_capture,
        ):
            _save_keyed_pairs_for_router(loop, cfg)

        assert len(format_args) == 3, "Expected three _write_keyed_pairs calls"
        assert all(f == "quad" for f in format_args), (
            "All _write_keyed_pairs calls must receive indexed_format='quad'"
        )

    def test_qa_mode_passes_qa_format_to_all_writes(self, tmp_path: Path) -> None:
        """_save_keyed_pairs_for_router passes indexed_format="qa" to all three calls."""
        from paramem.server.consolidation import _save_keyed_pairs_for_router

        store_dir = tmp_path / "adapters"
        cfg = MagicMock()
        cfg.consolidation.mode = "train"
        cfg.consolidation.indexed_format = "qa"
        cfg.adapter_dir = store_dir
        cfg.simulate_dir = tmp_path / "simulate"

        loop = MagicMock()
        loop.indexed_key_cache = {
            "graph1": {
                "key": "graph1",
                "question": "Q?",
                "answer": "A.",
                "source_subject": "A",
                "source_predicate": "p",
                "source_object": "B",
                "speaker_id": "",
                "first_seen_cycle": 1,
            }
        }
        loop.episodic_simhash = {"graph1": 1}
        loop.semantic_simhash = {"graph1": 1}
        loop.procedural_simhash = {"graph1": 1}

        format_args: list[str] = []

        def _capture(*args, indexed_format: str = "qa", **kwargs):
            format_args.append(indexed_format)

        with patch(
            "paramem.server.consolidation._write_keyed_pairs",
            side_effect=_capture,
        ):
            _save_keyed_pairs_for_router(loop, cfg)

        assert all(f == "qa" for f in format_args)


# ---------------------------------------------------------------------------
# Tests: read path dispatches correctly in create_consolidation_loop seed block
# ---------------------------------------------------------------------------


class TestSeedReadDispatch:
    """Test that create_consolidation_loop selects the right reader.

    We mock ConsolidationLoop entirely to avoid GPU/model loading, then
    verify which read function was called.
    """

    def _run_create_loop(
        self,
        tmp_path: Path,
        *,
        indexed_format: str,
        tier: str = "episodic",
    ) -> tuple[list, list]:
        """Run the create_consolidation_loop seed block with mocked ConsolidationLoop.

        Writes a minimal keyed_pairs.json to the expected path, then calls
        create_consolidation_loop with seed_state_from_disk=True.

        Returns (qa_read_calls, quad_read_calls).
        """
        from paramem.server.config import ConsolidationScheduleConfig

        kp_dir = tmp_path / "adapters" / tier
        kp_dir.mkdir(parents=True, exist_ok=True)
        kp_path = kp_dir / "keyed_pairs.json"
        kp_path.write_text(json.dumps([]))

        # Build a minimal MagicMock ServerConfig
        cfg = MagicMock()
        cfg.adapter_dir = tmp_path / "adapters"
        cfg.simulate_dir = tmp_path / "simulate"
        cfg.debug_dir = tmp_path / "debug"
        cfg.key_metadata_path = tmp_path / "key_metadata.json"
        cfg.debug = False
        cfg.consolidation = MagicMock(spec=ConsolidationScheduleConfig)
        cfg.consolidation.mode = "train"
        cfg.consolidation.indexed_format = indexed_format
        cfg.consolidation.consolidation_period_string = "84h"
        cfg.consolidation.training_temp_limit = 0  # 0 = disabled (no thermal policy)

        # Minimum fields the ConsolidationLoop constructor needs — not reached
        # since we mock the class.
        cfg.consolidation_config = MagicMock()
        cfg.training_config = MagicMock()
        cfg.episodic_adapter_config = MagicMock()
        cfg.semantic_adapter_config = MagicMock()
        cfg.adapters.procedural.enabled = False
        cfg.sanitization.cloud_scope = []
        cfg.prompts_dir = tmp_path / "prompts"
        cfg.graph_config = MagicMock()

        qa_calls: list = []
        quad_calls: list = []

        def _mock_read_qa(path):
            qa_calls.append(str(path))
            return []

        def _mock_read_quad(path):
            quad_calls.append(str(path))
            return []

        with (
            patch(
                "paramem.server.consolidation.ConsolidationLoop",
                return_value=MagicMock(),
            ),
            patch(
                "paramem.server.consolidation._load_key_metadata",
                return_value=None,
            ),
            patch(
                "paramem.training.keyed_pairs_io.read_keyed_pairs",
                side_effect=_mock_read_qa,
            ),
            patch(
                "paramem.training.keyed_pairs_io.read_keyed_pairs_quad",
                side_effect=_mock_read_quad,
            ),
        ):
            from paramem.server.consolidation import create_consolidation_loop

            create_consolidation_loop(
                model=MagicMock(),
                tokenizer=MagicMock(),
                config=cfg,
                seed_state_from_disk=True,
            )

        return qa_calls, quad_calls

    def test_quad_mode_uses_read_keyed_pairs_quad(self, tmp_path: Path) -> None:
        qa_calls, quad_calls = self._run_create_loop(
            tmp_path, indexed_format="quad", tier="episodic"
        )
        assert len(quad_calls) > 0, "read_keyed_pairs_quad must be called in quad mode"
        assert len(qa_calls) == 0, "read_keyed_pairs must NOT be called in quad mode"

    def test_qa_mode_uses_read_keyed_pairs(self, tmp_path: Path) -> None:
        qa_calls, quad_calls = self._run_create_loop(tmp_path, indexed_format="qa", tier="episodic")
        assert len(qa_calls) > 0, "read_keyed_pairs must be called in QA mode"
        assert len(quad_calls) == 0, "read_keyed_pairs_quad must NOT be called in QA mode"
