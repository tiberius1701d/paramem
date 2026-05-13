"""Parity tests: simulate mode is blackbox-equivalent to train mode.

Simulate runs the full extraction + key-assignment + SimHash pipeline but
skips the LoRA weight update.  Under perfect recall, on-disk state
(graph.json for simulate, keyed_pairs for train) is identical in content to
what train mode would produce.

The ``TestProbeKeysFromDisk`` class covers the train-mode disk reader
(``probe_keys_from_disk`` against ``keyed_pairs.json``).  The companion
``TestProbeKeysFromGraph`` class covers the simulate-mode graph reader
(``probe_keys_from_graph`` against ``graph.json``).  Both verify that the
returned result shape is identical so callers can swap the two functions
without touching downstream consumers.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import networkx as nx

from paramem.server.simulate_store import _IK_KEY_ATTR, save_simulate_graph
from paramem.training.consolidation import ConsolidationLoop
from paramem.training.indexed_memory import (
    build_registry,
    compute_simhash,
    probe_keys_from_disk,
    probe_keys_from_graph,
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
    loop.indexed_key_cache = {}
    loop.cycle_count = 0
    loop._indexed_next_index = 1
    loop._procedural_next_index = 1
    loop.episodic_simhash = {}
    loop.semantic_simhash = {}
    loop.procedural_simhash = {}
    loop.procedural_sp_index = {}
    loop.merger = MagicMock()
    # _is_quad / _indexed_format are set by __init__; tests that bypass
    # __init__ via __new__ / object.__new__ must set them explicitly.
    loop._indexed_format = "qa"
    loop._is_quad = False
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
        assert "graph1" in loop.indexed_key_cache
        assert "graph2" in loop.indexed_key_cache
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
        """Existing keys come from seeded indexed_key_cache, not probe_key."""
        loop = _make_bare_loop(tmp_path)
        # Pre-existing key, seeded from disk on startup
        loop.indexed_key_cache["graph1"] = {
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
        loop.indexed_key_cache["proc1"] = {
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
        assert "proc1" not in loop.indexed_key_cache
        assert "proc1" not in set(loop.indexed_key_registry.list_active())

        # New key registered and sp_index points to it
        assert "proc2" in loop.procedural_simhash
        assert "proc2" in loop.indexed_key_cache
        assert loop.procedural_sp_index[("speaker_alice", "alex", "likes")] == "proc2"

        # SimHash matches direct compute_simhash on ground-truth
        expected = compute_simhash("proc2", "What does Alex like?", "Tea.")
        assert loop.procedural_simhash["proc2"] == expected


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
                    "question": "Q?",
                    "answer": "A.",
                    "source_subject": "s",
                    "source_object": "o",
                }
            ],
            [],
            speaker_id="Speaker0",
        )
        assert loop.cycle_count == 1
        assert "graph1" in loop.indexed_key_cache


class TestProbeKeysFromDisk:
    """probe_keys_from_disk reads keyed_pairs.json matching the grouped-probe shape."""

    @staticmethod
    def _full_pair(key: str, question: str, answer: str, **extras) -> dict:
        """Build a canonical keyed pair with all eight required fields.

        Tests that only care about key/question/answer can rely on the defaults
        for the remaining five fields.  The strict ``read_keyed_pairs`` reader
        rejects any entry missing a field, so every fixture that goes through
        the facade must supply all eight.
        """
        return {
            "key": key,
            "question": question,
            "answer": answer,
            "source_subject": extras.get("source_subject", "Subject"),
            "source_predicate": extras.get("source_predicate", "related_to"),
            "source_object": extras.get("source_object", "Object"),
            "speaker_id": extras.get("speaker_id", ""),
            "first_seen_cycle": extras.get("first_seen_cycle", 1),
        }

    def _write_pairs(self, path, pairs):
        """Write *pairs* via the canonical facade, creating parent dirs."""
        from paramem.training.keyed_pairs_io import write_keyed_pairs

        write_keyed_pairs(path, pairs)

    def test_reads_episodic_from_subdir(self, tmp_path):
        """Canonical layout: episodic keyed_pairs lives under episodic/ subdir."""
        self._write_pairs(
            tmp_path / "episodic" / "keyed_pairs.json",
            [self._full_pair("graph1", "Q1?", "A1.")],
        )
        results = probe_keys_from_disk(tmp_path, {"episodic": ["graph1"]})
        assert results["graph1"]["question"] == "Q1?"
        assert results["graph1"]["answer"] == "A1."
        assert results["graph1"]["confidence"] == 1.0

    def test_reads_episodic_legacy_top_level_fallback(self, tmp_path):
        """Legacy fallback: reads top-level keyed_pairs.json when canonical path is absent."""
        self._write_pairs(
            tmp_path / "keyed_pairs.json",
            [self._full_pair("graph1", "Q1?", "A1.")],
        )
        # Canonical path does NOT exist — fallback must activate.
        assert not (tmp_path / "episodic" / "keyed_pairs.json").exists()
        results = probe_keys_from_disk(tmp_path, {"episodic": ["graph1"]})
        assert results["graph1"]["answer"] == "A1."

    def test_reads_semantic_from_subdir(self, tmp_path):
        self._write_pairs(
            tmp_path / "semantic" / "keyed_pairs.json",
            [self._full_pair("graph5", "Q5?", "A5.")],
        )
        results = probe_keys_from_disk(tmp_path, {"semantic": ["graph5"]})
        assert results["graph5"]["answer"] == "A5."

    def test_reads_procedural_from_subdir(self, tmp_path):
        self._write_pairs(
            tmp_path / "procedural" / "keyed_pairs.json",
            [self._full_pair("proc3", "Q?", "Tea.")],
        )
        results = probe_keys_from_disk(tmp_path, {"procedural": ["proc3"]})
        assert results["proc3"]["answer"] == "Tea."

    def test_missing_file_returns_none(self, tmp_path):
        results = probe_keys_from_disk(tmp_path, {"episodic": ["graph1", "graph2"]})
        assert results == {"graph1": None, "graph2": None}

    def test_missing_key_returns_none(self, tmp_path):
        self._write_pairs(
            tmp_path / "episodic" / "keyed_pairs.json",
            [self._full_pair("graph1", "Q1?", "A1.")],
        )
        results = probe_keys_from_disk(tmp_path, {"episodic": ["graph1", "graph999"]})
        assert results["graph1"] is not None
        assert results["graph999"] is None

    def test_malformed_json_returns_none(self, tmp_path):
        (tmp_path / "episodic").mkdir(parents=True, exist_ok=True)
        (tmp_path / "episodic" / "keyed_pairs.json").write_text("{not json")
        results = probe_keys_from_disk(tmp_path, {"episodic": ["graph1"]})
        assert results == {"graph1": None}

    def test_empty_keys_skipped(self, tmp_path):
        results = probe_keys_from_disk(tmp_path, {"episodic": []})
        assert results == {}

    def test_raw_output_is_json_with_fields(self, tmp_path):
        self._write_pairs(
            tmp_path / "episodic" / "keyed_pairs.json",
            [self._full_pair("graph1", "Q?", "A.")],
        )
        results = probe_keys_from_disk(tmp_path, {"episodic": ["graph1"]})
        raw = json.loads(results["graph1"]["raw_output"])
        assert raw["key"] == "graph1"
        assert raw["question"] == "Q?"
        assert raw["answer"] == "A."

    def test_reads_age_envelope_via_read_maybe_encrypted(self, tmp_path, monkeypatch):
        """Encryption-contract guard: age-encrypted keyed_pairs decrypt on read.

        The other tests in this class write plaintext; they would pass even if
        ``probe_keys_from_disk`` switched to a bare ``Path.read_bytes``. This
        test sets up a real daily identity, writes the keyed_pairs through the
        production encrypted helper, and asserts the disk-read returns the
        plaintext content. A regression that drops ``read_maybe_encrypted``
        from the read path fails this test.
        """
        from paramem.backup.key_store import (
            DAILY_PASSPHRASE_ENV_VAR,
            _clear_daily_identity_cache,
            mint_daily_identity,
            wrap_daily_identity,
            write_daily_key_file,
        )

        ident = mint_daily_identity()
        key_path = tmp_path / "daily_key.age"
        write_daily_key_file(wrap_daily_identity(ident, "pw"), key_path)
        monkeypatch.setenv(DAILY_PASSPHRASE_ENV_VAR, "pw")
        monkeypatch.setattr("paramem.backup.key_store.DAILY_KEY_PATH_DEFAULT", key_path)
        _clear_daily_identity_cache()

        kp_path = tmp_path / "episodic" / "keyed_pairs.json"
        # Write through the facade so the entry passes strict schema validation
        # on read; _atomic_json_write is retired as a write path since it
        # doesn't normalise the schema.
        from paramem.training.keyed_pairs_io import write_keyed_pairs

        write_keyed_pairs(kp_path, [self._full_pair("graph7", "Q7?", "A7.")])
        # Confirm the on-disk file is an age envelope, not plaintext.
        assert kp_path.read_bytes()[:22].startswith(b"age-encryption.org/v1")

        results = probe_keys_from_disk(tmp_path, {"episodic": ["graph7"]})
        assert results["graph7"]["question"] == "Q7?"
        assert results["graph7"]["answer"] == "A7."


class TestSeedHelpers:
    """seed_episodic_cache / seed_semantic_cache rebuild indexed_key_cache from disk pairs."""

    def test_seed_episodic_populates_qa_and_registry(self, tmp_path):
        loop = _make_bare_loop(tmp_path)
        pairs = [
            {"key": "graph1", "question": "Q1?", "answer": "A1."},
            {"key": "graph2", "question": "Q2?", "answer": "A2."},
        ]
        loop.seed_episodic_cache(pairs)
        assert loop.indexed_key_cache["graph1"]["answer"] == "A1."
        assert loop.indexed_key_cache["graph2"]["answer"] == "A2."
        active = set(loop.indexed_key_registry.list_active())
        assert {"graph1", "graph2"} <= active

    def test_seed_semantic_populates_qa_and_registry(self, tmp_path):
        loop = _make_bare_loop(tmp_path)
        pairs = [{"key": "graph7", "question": "Q?", "answer": "A."}]
        loop.seed_semantic_cache(pairs)
        assert loop.indexed_key_cache["graph7"]["answer"] == "A."
        assert "graph7" in set(loop.indexed_key_registry.list_active())

    def test_seed_episodic_does_not_duplicate_existing_registry_keys(self, tmp_path):
        loop = _make_bare_loop(tmp_path)
        loop.indexed_key_registry.add("graph1")
        pairs = [{"key": "graph1", "question": "Q?", "answer": "A."}]
        loop.seed_episodic_cache(pairs)
        # Adding the same key twice would raise; reaching this assert proves idempotency.
        assert "graph1" in set(loop.indexed_key_registry.list_active())


# ---------------------------------------------------------------------------
# Helpers for TestProbeKeysFromGraph
# ---------------------------------------------------------------------------


def _write_graph(path, quads: list[dict]) -> None:
    """Write *quads* as a simulate-mode graph.json at *path*.

    Builds a ``MultiDiGraph`` with one edge per quad using ``_IK_KEY_ATTR``
    for the indexed-memory key, then persists via ``save_simulate_graph``
    with ``encrypted=False`` so tests read plaintext.
    """
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
    save_simulate_graph(graph, path, encrypted=False)


class TestProbeKeysFromGraph:
    """probe_keys_from_graph reads graph.json matching the grouped-probe shape.

    Parity property: the returned result shape is identical to
    ``probe_keys_from_disk`` on a quad file so callers can swap the two
    without touching downstream consumers.  Under perfect recall both return::

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
        results = probe_keys_from_graph(tmp_path, {"episodic": ["graph1"]})
        assert results["graph1"] is not None
        assert results["graph1"]["subject"] == "Alex"
        assert results["graph1"]["object"] == "Berlin"
        assert results["graph1"]["predicate"] == "lives_in"
        assert results["graph1"]["confidence"] == 1.0
        assert results["graph1"]["format"] == "quad"

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
        results = probe_keys_from_graph(tmp_path, {"semantic": ["graph5"]})
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
        results = probe_keys_from_graph(tmp_path, {"procedural": ["proc3"]})
        assert results["proc3"]["object"] == "Tea"

    def test_missing_file_returns_none(self, tmp_path):
        """Missing graph.json → all keys return None."""
        results = probe_keys_from_graph(tmp_path, {"episodic": ["graph1", "graph2"]})
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
        results = probe_keys_from_graph(tmp_path, {"episodic": ["graph1", "graph999"]})
        assert results["graph1"] is not None
        assert results["graph999"] is None

    def test_empty_keys_skipped(self, tmp_path):
        """Empty key list for an adapter → no entries in result."""
        results = probe_keys_from_graph(tmp_path, {"episodic": []})
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
        results = probe_keys_from_graph(tmp_path, {"episodic": ["graph1"]})
        raw = json.loads(results["graph1"]["raw_output"])
        assert raw["key"] == "graph1"
        assert raw["subject"] == "Alex"
        assert raw["object"] == "Munich"

    def test_parity_with_probe_keys_from_disk_result_shape(self, tmp_path):
        """Result shape is identical to probe_keys_from_disk on a quad file.

        Verifies that both functions return the same set of top-level keys
        for a hit result so downstream consumers can swap them transparently.
        """
        from paramem.training.keyed_pairs_io import write_keyed_pairs_quad

        quad = {
            "key": "graph1",
            "subject": "Alice",
            "predicate": "knows",
            "object": "Bob",
            "speaker_id": "",
            "first_seen_cycle": 0,
        }

        # Write as graph.json for probe_keys_from_graph
        graph_sim_dir = tmp_path / "sim"
        _write_graph(graph_sim_dir / "episodic" / "graph.json", [quad])

        # Write as quad keyed_pairs.json for probe_keys_from_disk
        kp_dir = tmp_path / "kp"
        kp_path = kp_dir / "episodic" / "keyed_pairs.json"
        kp_path.parent.mkdir(parents=True, exist_ok=True)
        write_keyed_pairs_quad(kp_path, [quad])

        graph_result = probe_keys_from_graph(graph_sim_dir, {"episodic": ["graph1"]})
        disk_result = probe_keys_from_disk(
            kp_dir, {"episodic": ["graph1"]}, formats_by_adapter={"episodic": "quad"}
        )

        assert set(graph_result["graph1"].keys()) == set(disk_result["graph1"].keys()), (
            "probe_keys_from_graph and probe_keys_from_disk must return identical "
            f"top-level keys.\n"
            f"graph_result keys: {sorted(graph_result['graph1'].keys())}\n"
            f"disk_result keys:  {sorted(disk_result['graph1'].keys())}"
        )
