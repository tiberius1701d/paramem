"""Parity tests: simulate mode is blackbox-equivalent to train mode.

Simulate runs the full extraction + key-assignment + SimHash pipeline but
skips the LoRA weight update.  Under perfect recall, on-disk state
(keyed_pairs + SimHash registries) is identical to what train mode would
produce.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from paramem.training.consolidation import ConsolidationLoop
from paramem.training.indexed_memory import (
    build_registry,
    compute_simhash,
    probe_keys_from_disk,
)
from paramem.training.key_registry import KeyRegistry
from paramem.utils.config import AdapterConfig, ConsolidationConfig, TrainingConfig


def _make_bare_loop(tmp_path, procedural: bool = False) -> ConsolidationLoop:
    """Minimal ConsolidationLoop bypassing model/__init__ side effects."""
    loop = object.__new__(ConsolidationLoop)
    loop.model = MagicMock()
    loop.tokenizer = MagicMock()
    loop.config = ConsolidationConfig()
    loop.training_config = TrainingConfig(num_epochs=1)
    loop.episodic_config = AdapterConfig(rank=4, alpha=8, target_modules=["q_proj"])
    loop.semantic_config = None
    loop.procedural_config = (
        AdapterConfig(rank=4, alpha=8, target_modules=["q_proj"]) if procedural else None
    )
    loop.wandb_config = None
    loop.output_dir = tmp_path
    loop.snapshot_dir = None
    loop.save_cycle_snapshots = False
    loop.indexed_key_registry = KeyRegistry()
    loop.indexed_key_qa = {}
    loop.cycle_count = 0
    loop._indexed_next_index = 1
    loop._procedural_next_index = 1
    loop.episodic_simhash = {}
    loop.semantic_simhash = {}
    loop.procedural_simhash = {}
    loop.procedural_sp_index = {}
    loop.merger = MagicMock()
    return loop


class TestSimulatedTrainingEpisodic:
    """_simulate_indexed_key_episodic must match train mode's SimHash build."""

    def test_new_keys_registered_and_simhash_built(self, tmp_path):
        loop = _make_bare_loop(tmp_path)
        session_qa = [
            {
                "question": "Where does Alex live?",
                "answer": "Heilbronn.",
                "source_subject": "Alex",
                "source_object": "Heilbronn",
            },
            {
                "question": "Who lives in Heilbronn?",
                "answer": "Alex.",
                "source_subject": "Heilbronn",
                "source_object": "Alex",
            },
        ]

        loop._simulate_indexed_key_episodic(session_qa)

        # Keys assigned graph1, graph2
        assert "graph1" in loop.indexed_key_qa
        assert "graph2" in loop.indexed_key_qa
        assert loop._indexed_next_index == 3

        # Registry has the keys
        active = set(loop.indexed_key_registry.list_active())
        assert {"graph1", "graph2"} <= active

        # SimHash matches build_registry output on ground-truth QA
        expected_pairs = [
            {
                "key": "graph1",
                "question": session_qa[0]["question"],
                "answer": session_qa[0]["answer"],
            },
            {
                "key": "graph2",
                "question": session_qa[1]["question"],
                "answer": session_qa[1]["answer"],
            },
        ]
        assert loop.episodic_simhash == build_registry(expected_pairs)

    def test_existing_keys_admitted_from_seed(self, tmp_path):
        """Existing keys come from seeded indexed_key_qa, not probe_key."""
        loop = _make_bare_loop(tmp_path)
        # Pre-existing key, seeded from disk on startup
        loop.indexed_key_qa["graph1"] = {
            "key": "graph1",
            "question": "Where does Alex live?",
            "answer": "Heilbronn.",
        }
        loop.indexed_key_registry.add("graph1")
        loop._indexed_next_index = 2

        new_qa = [
            {
                "question": "What pet does Alex have?",
                "answer": "A cat.",
                "source_subject": "Alex",
                "source_object": "cat",
            },
        ]
        loop._simulate_indexed_key_episodic(new_qa)

        # Both keys end up in episodic_simhash (existing + new)
        assert "graph1" in loop.episodic_simhash
        assert "graph2" in loop.episodic_simhash

        # Expected simhash for graph1 uses the seeded content
        expected_g1 = compute_simhash("graph1", "Where does Alex live?", "Heilbronn.")
        assert loop.episodic_simhash["graph1"] == expected_g1


class TestSimulatedTrainingProcedural:
    """_simulate_indexed_key_procedural mirrors train's contradiction handling."""

    def test_contradiction_retires_old_key(self, tmp_path, monkeypatch):
        loop = _make_bare_loop(tmp_path, procedural=True)

        # Seed an existing procedural key that will be contradicted.
        loop.indexed_key_qa["proc1"] = {
            "key": "proc1",
            "question": "What does Alex like?",
            "answer": "Coffee.",
            "source_subject": "Alex",
            "source_predicate": "likes",
            "source_object": "Coffee",
            "speaker_id": "speaker_alice",
        }
        loop.indexed_key_registry.add("proc1")
        loop.procedural_simhash["proc1"] = 0xDEADBEEF
        loop.procedural_sp_index[("speaker_alice", "alex", "likes")] = "proc1"
        loop._procedural_next_index = 2

        # New preference for the same (speaker, subject, predicate) — contradicts.
        new_relations = [
            {
                "subject": "Alex",
                "predicate": "likes",
                "object": "Tea",
                "relation_type": "preference",
                "speaker_id": "speaker_alice",
            }
        ]

        # Stub generate_qa_from_relations to avoid invoking the model.
        fake_qa = [
            {
                "question": "What does Alex like?",
                "answer": "Tea.",
                "source_subject": "Alex",
                "source_predicate": "likes",
                "source_object": "Tea",
            }
        ]
        monkeypatch.setattr(
            "paramem.training.consolidation.generate_qa_from_relations",
            lambda relations, model, tokenizer: fake_qa,
        )

        loop._simulate_indexed_key_procedural(new_relations, speaker_id="speaker_alice")

        # Old key retired from all indexes
        assert "proc1" not in loop.procedural_simhash
        assert "proc1" not in loop.indexed_key_qa
        assert "proc1" not in set(loop.indexed_key_registry.list_active())

        # New key registered and sp_index points to it
        assert "proc2" in loop.procedural_simhash
        assert "proc2" in loop.indexed_key_qa
        assert loop.procedural_sp_index[("speaker_alice", "alex", "likes")] == "proc2"

        # SimHash matches direct compute_simhash on ground-truth
        expected = compute_simhash("proc2", "What does Alex like?", "Tea.")
        assert loop.procedural_simhash["proc2"] == expected


class TestSimulatedTrainingResult:
    """simulated_training returns a shape-compatible dict with simulated=True."""

    def test_returns_simulated_flag(self, tmp_path):
        loop = _make_bare_loop(tmp_path)
        result = loop.simulated_training([], [])
        assert result == {"simulated": True}
        # cycle_count incremented even when no work
        assert loop.cycle_count == 1

    def test_increments_cycle_count_on_work(self, tmp_path):
        loop = _make_bare_loop(tmp_path)
        loop.simulated_training(
            [
                {
                    "question": "Q?",
                    "answer": "A.",
                    "source_subject": "s",
                    "source_object": "o",
                }
            ],
            [],
        )
        assert loop.cycle_count == 1
        assert "graph1" in loop.indexed_key_qa


class TestProbeKeysFromDisk:
    """probe_keys_from_disk reads keyed_pairs.json matching the grouped-probe shape."""

    def _write_pairs(self, path, pairs):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(pairs))

    def test_reads_episodic_at_top_level(self, tmp_path):
        self._write_pairs(
            tmp_path / "keyed_pairs.json",
            [{"key": "graph1", "question": "Q1?", "answer": "A1."}],
        )
        results = probe_keys_from_disk(tmp_path, {"episodic": ["graph1"]})
        assert results["graph1"]["question"] == "Q1?"
        assert results["graph1"]["answer"] == "A1."
        assert results["graph1"]["confidence"] == 1.0

    def test_reads_semantic_from_subdir(self, tmp_path):
        self._write_pairs(
            tmp_path / "semantic" / "keyed_pairs.json",
            [{"key": "graph5", "question": "Q5?", "answer": "A5."}],
        )
        results = probe_keys_from_disk(tmp_path, {"semantic": ["graph5"]})
        assert results["graph5"]["answer"] == "A5."

    def test_reads_procedural_from_subdir(self, tmp_path):
        self._write_pairs(
            tmp_path / "procedural" / "keyed_pairs.json",
            [{"key": "proc3", "question": "Q?", "answer": "Tea."}],
        )
        results = probe_keys_from_disk(tmp_path, {"procedural": ["proc3"]})
        assert results["proc3"]["answer"] == "Tea."

    def test_missing_file_returns_none(self, tmp_path):
        results = probe_keys_from_disk(tmp_path, {"episodic": ["graph1", "graph2"]})
        assert results == {"graph1": None, "graph2": None}

    def test_missing_key_returns_none(self, tmp_path):
        self._write_pairs(
            tmp_path / "keyed_pairs.json",
            [{"key": "graph1", "question": "Q1?", "answer": "A1."}],
        )
        results = probe_keys_from_disk(tmp_path, {"episodic": ["graph1", "graph999"]})
        assert results["graph1"] is not None
        assert results["graph999"] is None

    def test_malformed_json_returns_none(self, tmp_path):
        (tmp_path / "keyed_pairs.json").write_text("{not json")
        results = probe_keys_from_disk(tmp_path, {"episodic": ["graph1"]})
        assert results == {"graph1": None}

    def test_empty_keys_skipped(self, tmp_path):
        results = probe_keys_from_disk(tmp_path, {"episodic": []})
        assert results == {}

    def test_raw_output_is_json_with_fields(self, tmp_path):
        self._write_pairs(
            tmp_path / "keyed_pairs.json",
            [{"key": "graph1", "question": "Q?", "answer": "A."}],
        )
        results = probe_keys_from_disk(tmp_path, {"episodic": ["graph1"]})
        raw = json.loads(results["graph1"]["raw_output"])
        assert raw["key"] == "graph1"
        assert raw["question"] == "Q?"
        assert raw["answer"] == "A."


class TestSeedHelpers:
    """seed_episodic_qa / seed_semantic_qa rebuild indexed_key_qa from disk pairs."""

    def test_seed_episodic_populates_qa_and_registry(self, tmp_path):
        loop = _make_bare_loop(tmp_path)
        pairs = [
            {"key": "graph1", "question": "Q1?", "answer": "A1."},
            {"key": "graph2", "question": "Q2?", "answer": "A2."},
        ]
        loop.seed_episodic_qa(pairs)
        assert loop.indexed_key_qa["graph1"]["answer"] == "A1."
        assert loop.indexed_key_qa["graph2"]["answer"] == "A2."
        active = set(loop.indexed_key_registry.list_active())
        assert {"graph1", "graph2"} <= active

    def test_seed_semantic_populates_qa_and_registry(self, tmp_path):
        loop = _make_bare_loop(tmp_path)
        pairs = [{"key": "graph7", "question": "Q?", "answer": "A."}]
        loop.seed_semantic_qa(pairs)
        assert loop.indexed_key_qa["graph7"]["answer"] == "A."
        assert "graph7" in set(loop.indexed_key_registry.list_active())

    def test_seed_episodic_does_not_duplicate_existing_registry_keys(self, tmp_path):
        loop = _make_bare_loop(tmp_path)
        loop.indexed_key_registry.add("graph1")
        pairs = [{"key": "graph1", "question": "Q?", "answer": "A."}]
        loop.seed_episodic_qa(pairs)
        # Adding the same key twice would raise; reaching this assert proves idempotency.
        assert "graph1" in set(loop.indexed_key_registry.list_active())
