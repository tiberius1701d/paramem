"""Parity tests: simulate mode is blackbox-equivalent to train mode.

Simulate runs the full extraction + key-assignment + SimHash pipeline but
skips the LoRA weight update.  Under perfect recall, on-disk state
(graph.json for simulate) is identical in content to what train mode would
produce.

The ``TestProbeKeysFromGraph`` class covers the simulate-mode graph reader
(``DiskMemorySource.probe`` against ``graph.json``) and verifies the returned
result shape.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import networkx as nx

from paramem.training.consolidation import ConsolidationLoop
from paramem.training.key_registry import KeyRegistry
from paramem.training.memory_persistence import _IK_KEY_ATTR, save_memory_to_disk
from paramem.training.memory_source import DiskMemorySource
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
    _reg: dict[str, KeyRegistry] = {"episodic": KeyRegistry()}
    if procedural:
        _reg["procedural"] = KeyRegistry()
    loop.indexed_key_registry = _reg
    loop.indexed_key_cache = {}
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
                "subject": "Alex",
                "predicate": "lives_in",
                "object": "Heilbronn",
            },
            {
                "subject": "Heilbronn",
                "predicate": "has_resident",
                "object": "Alex",
            },
        ]

        loop._simulate_indexed_key_episodic(session_qa)

        # Keys assigned graph1, graph2
        assert "graph1" in loop.store._entries_flat_view()
        assert "graph2" in loop.store._entries_flat_view()
        assert loop._indexed_next_index == 3

        # Registry has the keys (simulate mode places episodic keys in episodic tier)
        active = set(loop.store.registry("episodic").list_active())
        assert {"graph1", "graph2"} <= active

        # SimHash matches build_registry output on ground-truth entries
        from paramem.training.entry_memory import build_registry

        expected_pairs = [
            {
                "key": "graph1",
                "subject": session_qa[0]["subject"],
                "predicate": session_qa[0]["predicate"],
                "object": session_qa[0]["object"],
            },
            {
                "key": "graph2",
                "subject": session_qa[1]["subject"],
                "predicate": session_qa[1]["predicate"],
                "object": session_qa[1]["object"],
            },
        ]
        assert loop.store.simhashes_in_tier("episodic") == build_registry(expected_pairs)

    def test_existing_keys_admitted_from_seed(self, tmp_path):
        """Existing keys come from seeded store, not probe_entry."""
        from paramem.training.entry_memory import compute_simhash

        loop = _make_bare_loop(tmp_path)
        # Pre-existing key, seeded from disk on startup
        loop.store.put(
            "episodic",
            "graph1",
            {
                "key": "graph1",
                "subject": "Alex",
                "predicate": "lives_in",
                "object": "Heilbronn",
            },
        )
        loop._indexed_next_index = 2

        new_qa = [
            {
                "subject": "Alex",
                "predicate": "has_pet",
                "object": "cat",
            },
        ]
        loop._simulate_indexed_key_episodic(new_qa)

        # Both keys end up in episodic_simhash (existing + new)
        assert "graph1" in loop.store.simhashes_in_tier("episodic")
        assert "graph2" in loop.store.simhashes_in_tier("episodic")

        # Expected simhash for graph1 uses the seeded content
        expected_g1 = compute_simhash("graph1", "Alex", "lives_in", "Heilbronn")
        assert loop.store.simhashes_in_tier("episodic")["graph1"] == expected_g1


class TestSimulatedTrainingProcedural:
    """_simulate_indexed_key_procedural mirrors train's contradiction handling."""

    def test_contradiction_retires_old_key(self, tmp_path, monkeypatch):
        loop = _make_bare_loop(tmp_path, procedural=True)

        # Seed an existing procedural key that will be contradicted.
        loop.store._entries_flat_view()["proc1"] = {
            "key": "proc1",
            "question": "What does Alex like?",
            "answer": "Coffee.",
            "subject": "Alex",
            "predicate": "likes",
            "object": "Coffee",
            "speaker_id": "speaker_alice",
        }
        loop.store.registry("procedural").add("proc1")
        loop.store.simhashes_in_tier("procedural")["proc1"] = 0xDEADBEEF
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
                "subject": "Alex",
                "predicate": "likes",
                "object": "Tea",
            }
        ]
        monkeypatch.setattr(
            "paramem.graph.qa_generator.generate_qa_from_relations",
            lambda relations, model, tokenizer: fake_qa,
        )

        loop._simulate_indexed_key_procedural(new_relations, speaker_id="speaker_alice")

        # Old key retired from all indexes
        assert "proc1" not in loop.store.simhashes_in_tier("procedural")
        assert "proc1" not in loop.store._entries_flat_view()
        assert "proc1" not in set(loop._all_active_keys())

        # New key registered and sp_index points to it
        assert "proc2" in loop.store.simhashes_in_tier("procedural")
        assert "proc2" in loop.store._entries_flat_view()
        assert loop.procedural_sp_index[("speaker_alice", "alex", "likes")] == "proc2"

        # SimHash matches direct compute_simhash on ground-truth entry
        from paramem.training.entry_memory import compute_simhash

        expected = compute_simhash("proc2", "Alex", "likes", "Tea")
        assert loop.store.simhashes_in_tier("procedural")["proc2"] == expected


class TestSimulatedTrainingResult:
    """simulated_training returns a shape-compatible dict with simulated=True."""

    def test_returns_simulated_flag(self, tmp_path):
        loop = _make_bare_loop(tmp_path)
        result = loop.simulated_training([], [], speaker_id="Speaker0")
        assert result == {"simulated": True}
        # cycle_count incremented even when no work
        assert loop.cycle_count == 1

    def test_increments_cycle_count_on_work(self, tmp_path):
        loop = _make_bare_loop(tmp_path)
        loop.simulated_training(
            [
                {
                    "subject": "s",
                    "predicate": "p",
                    "object": "o",
                }
            ],
            [],
            speaker_id="Speaker0",
        )
        assert loop.cycle_count == 1
        assert "graph1" in loop.store._entries_flat_view()


def _write_graph(path, quads: list[dict]) -> None:
    """Write *quads* as a simulate-mode graph.json at *path*."""
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    graph = nx.MultiDiGraph()
    for quad in quads:
        graph.add_edge(
            quad["subject"],
            quad["object"],
            **{
                _IK_KEY_ATTR: quad["key"],
                "predicate": quad.get("predicate", ""),
                "speaker_id": quad.get("speaker_id", ""),
                "first_seen_cycle": quad.get("first_seen_cycle", 0),
            },
        )
    save_memory_to_disk(graph, path, encrypted=False)


class TestProbeKeysFromGraph:
    """DiskMemorySource.probe reads graph.json matching the grouped-probe shape.

    Under perfect recall, hit results return::

        {"key": str, "subject": str, "predicate": str, "object": str,
         "confidence": 1.0, "format": "quad",
         "fact_text": str, "raw_output": str}

    Missing tiers / missing keys → ``None``.
    """

    def test_reads_episodic_from_subdir(self, tmp_path):
        """Canonical layout: episodic graph lives under episodic/ subdir."""
        _write_graph(
            tmp_path / "episodic" / "graph.json",
            [
                {
                    "key": "graph1",
                    "subject": "Alex",
                    "predicate": "lives_in",
                    "object": "Berlin",
                    "speaker_id": "",
                    "first_seen_cycle": 0,
                }
            ],
        )
        results = DiskMemorySource(tmp_path).probe({"episodic": ["graph1"]})
        assert results["graph1"] is not None
        assert results["graph1"]["subject"] == "Alex"
        assert results["graph1"]["object"] == "Berlin"
        assert results["graph1"]["predicate"] == "lives_in"
        assert results["graph1"]["confidence"] == 1.0

    def test_reads_semantic_from_subdir(self, tmp_path):
        """Semantic tier reads from semantic/ subdir."""
        _write_graph(
            tmp_path / "semantic" / "graph.json",
            [
                {
                    "key": "graph5",
                    "subject": "Bob",
                    "predicate": "works_at",
                    "object": "Acme",
                    "speaker_id": "",
                    "first_seen_cycle": 0,
                }
            ],
        )
        results = DiskMemorySource(tmp_path).probe({"semantic": ["graph5"]})
        assert results["graph5"]["subject"] == "Bob"

    def test_reads_procedural_from_subdir(self, tmp_path):
        """Procedural tier reads from procedural/ subdir."""
        _write_graph(
            tmp_path / "procedural" / "graph.json",
            [
                {
                    "key": "proc3",
                    "subject": "Carol",
                    "predicate": "likes",
                    "object": "Tea",
                    "speaker_id": "",
                    "first_seen_cycle": 0,
                }
            ],
        )
        results = DiskMemorySource(tmp_path).probe({"procedural": ["proc3"]})
        assert results["proc3"]["object"] == "Tea"

    def test_missing_file_returns_none(self, tmp_path):
        """Missing graph.json → all keys return None."""
        results = DiskMemorySource(tmp_path).probe({"episodic": ["graph1", "graph2"]})
        assert results == {"graph1": None, "graph2": None}

    def test_missing_key_returns_none(self, tmp_path):
        """Key absent from graph → None; key present → hit."""
        _write_graph(
            tmp_path / "episodic" / "graph.json",
            [
                {
                    "key": "graph1",
                    "subject": "X",
                    "predicate": "p",
                    "object": "Y",
                    "speaker_id": "",
                    "first_seen_cycle": 0,
                }
            ],
        )
        results = DiskMemorySource(tmp_path).probe({"episodic": ["graph1", "graph999"]})
        assert results["graph1"] is not None
        assert results["graph999"] is None

    def test_empty_keys_skipped(self, tmp_path):
        """Empty key list for an adapter → no entries in result."""
        results = DiskMemorySource(tmp_path).probe({"episodic": []})
        assert results == {}

    def test_raw_output_is_json_with_fields(self, tmp_path):
        """raw_output is a JSON string with key/subject/predicate/object fields."""
        _write_graph(
            tmp_path / "episodic" / "graph.json",
            [
                {
                    "key": "graph1",
                    "subject": "Alex",
                    "predicate": "lives_in",
                    "object": "Munich",
                    "speaker_id": "",
                    "first_seen_cycle": 0,
                }
            ],
        )
        results = DiskMemorySource(tmp_path).probe({"episodic": ["graph1"]})
        raw = json.loads(results["graph1"]["raw_output"])
        assert raw["key"] == "graph1"
        assert raw["subject"] == "Alex"
        assert raw["object"] == "Munich"

    def test_result_shape_has_required_fields(self, tmp_path):
        """Hit results contain key/subject/predicate/object/confidence/format/fact_text/raw_output fields."""  # noqa: E501
        quad = {
            "key": "graph1",
            "subject": "Alice",
            "predicate": "knows",
            "object": "Bob",
            "speaker_id": "",
            "first_seen_cycle": 0,
        }

        graph_sim_dir = tmp_path / "sim"
        _write_graph(graph_sim_dir / "episodic" / "graph.json", [quad])

        graph_result = DiskMemorySource(graph_sim_dir).probe({"episodic": ["graph1"]})

        expected_keys = {
            "key",
            "subject",
            "predicate",
            "object",
            "speaker_id",
            "first_seen_cycle",
            "confidence",
            "fact_text",
            "raw_output",
        }
        assert expected_keys == set(graph_result["graph1"].keys()), (
            "DiskMemorySource.probe must return the canonical result shape.\n"
            f"actual keys: {sorted(graph_result['graph1'].keys())}"
        )
