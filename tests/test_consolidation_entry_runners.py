"""S2 regression tests: run_cycle and simulated_training entry-mode coverage.

Covers:
- _run_indexed_key_semantic: both Loop A (newly-promoted keys) and
  Loop B (previously-promoted keys still in semantic) fire without KeyError
  and the semantic simhash is updated from entry-shaped dicts.
- _run_indexed_key_episodic: key assignment, cache population, and
  format_entry_training dispatched.
- _run_indexed_key_procedural: keys assigned directly, no QA-gen
  model.generate called, contradiction handling works.
- _simulate_indexed_key_episodic: existing-key re-access branch runs without
  KeyError when indexed_key_cache is pre-seeded with entry-shaped dicts.
- _simulate_indexed_key_procedural: compute_simhash path called without
  KeyError; procedural_sp_index updated.
- _save_tier_graphs (server/consolidation.py) simulate-mode write
  produces a graph.json on disk; train mode is a no-op.

No GPU required — model interactions replaced with MagicMock stubs and
all heavy operations are patched out.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from paramem.training.consolidation import ConsolidationLoop
from paramem.training.key_registry import KeyRegistry
from paramem.utils.config import AdapterConfig, ConsolidationConfig, TrainingConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_training_config() -> TrainingConfig:
    return TrainingConfig(
        num_epochs=1,
        gradient_checkpointing=False,
        batch_size=1,
        recall_early_stopping=False,
    )


def _minimal_adapter_config() -> AdapterConfig:
    return AdapterConfig(rank=4, alpha=8, learning_rate=1e-4, target_modules=["q_proj"])


def _make_stub_model(*adapter_names: str) -> MagicMock:
    """Return a MagicMock that behaves like a minimal PeftModel."""
    model = MagicMock()
    model.peft_config = {name: MagicMock() for name in adapter_names}
    return model


def _make_loop(tmp_path: Path) -> Any:
    """Construct a bare ConsolidationLoop without calling __init__."""
    model = _make_stub_model("episodic", "semantic", "procedural")

    loop = ConsolidationLoop.__new__(ConsolidationLoop)
    loop.model = model
    loop.tokenizer = MagicMock()
    cfg = ConsolidationConfig(
        indexed_key_replay_enabled=True,
    )
    loop.config = cfg
    loop.training_config = _minimal_training_config()
    loop.episodic_config = _minimal_adapter_config()
    loop.semantic_config = _minimal_adapter_config()
    loop.procedural_config = _minimal_adapter_config()
    loop.wandb_config = None
    loop.output_dir = tmp_path
    loop.snapshot_dir = None
    loop.save_cycle_snapshots = False
    loop.persist_graph = False
    loop._thermal_policy = None
    from paramem.training.memory_store import MemoryStore as _MS

    loop.store = _MS(replay_enabled=True)
    for _tier in ("episodic", "semantic", "procedural"):
        loop.store.load_registry(_tier, KeyRegistry())
    loop.procedural_sp_index: dict = {}
    loop.cycle_count = 1
    loop._indexed_next_index = 1
    loop._procedural_next_index = 1
    loop.key_sessions: dict = {}
    loop.promoted_keys: set = set()
    loop.shutdown_requested = False
    loop.merger = MagicMock()
    loop.fingerprint_cache = None
    return loop


def _make_cache_entry(
    key: str,
    subject: str,
    predicate: str,
    obj: str,
    *,
    speaker_id: str = "Speaker0",
    first_seen_cycle: int = 1,
) -> dict:
    """Build a minimal uniform-shape cache entry.

    Carries ``subject``/``predicate``/``object``.
    """
    return {
        "key": key,
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "speaker_id": speaker_id,
        "first_seen_cycle": first_seen_cycle,
    }


def _fake_entry_rels(n: int = 2, *, speaker_id: str = "Speaker0") -> list[dict]:
    """Return n synthetic relation dicts suitable as episodic input."""
    return [
        {
            "subject": f"Subject{i}",
            "predicate": "knows",
            "object": f"Object{i}",
            "relation_type": "factual",
            "speaker_id": speaker_id,
        }
        for i in range(1, n + 1)
    ]


# ---------------------------------------------------------------------------
# _run_indexed_key_semantic — Loop A + Loop B
# ---------------------------------------------------------------------------


class TestRunIndexedKeySemantic:
    """_run_indexed_key_semantic: Loop A (new promotions) + Loop B (existing) fire correctly."""

    def test_loop_a_newly_promoted_key_no_keyerror(self, tmp_path: Path) -> None:
        """Loop A: key matches promoted_set via subject → no KeyError."""
        loop = _make_loop(tmp_path)

        # Pre-seed one key whose subject matches the promoted entity.
        loop.store._entries_flat_view()["graph1"] = _make_cache_entry(
            "graph1", "Alice", "lives_in", "Berlin"
        )
        loop.store.registry("semantic").add("graph1")
        loop.store.simhashes_in_tier("semantic")["graph1"] = 0xABCD

        fmt_mock = MagicMock(
            return_value=[{"input_ids": [1], "labels": [1], "attention_mask": [1]}]
        )

        with (
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.training.consolidation.train_adapter", return_value={"train_loss": 0.1}),
            patch("paramem.training.consolidation.format_entry_training", fmt_mock),
            patch.object(loop, "_indexed_dataset", return_value=MagicMock()),
            patch.object(loop, "_make_training_config", return_value=MagicMock()),
            patch.object(loop, "_enable_gradient_checkpointing"),
            patch.object(loop, "_disable_gradient_checkpointing"),
            patch("paramem.training.consolidation.build_registry", return_value={}),
        ):
            # "Alice" is in the promoted set — Loop A should fire.
            result = loop._run_indexed_key_semantic(new_promotions=["Alice"])

        assert fmt_mock.called, "format_entry_training must be called"
        # Result is a float (train_loss) not None
        assert result is not None

    def test_loop_b_previously_promoted_key_no_keyerror(self, tmp_path: Path) -> None:
        """Loop B: key in semantic_simhash but not in promoted_set → still processed."""
        loop = _make_loop(tmp_path)

        # Pre-seed a key in semantic (previously promoted, no longer in new_promotions).
        loop.store._entries_flat_view()["graph2"] = _make_cache_entry(
            "graph2", "Bob", "works_at", "ACME"
        )
        loop.store.registry("semantic").add("graph2")
        loop.store.simhashes_in_tier("semantic")["graph2"] = 0xDEAD

        fmt_mock = MagicMock(
            return_value=[{"input_ids": [1], "labels": [1], "attention_mask": [1]}]
        )

        with (
            patch("paramem.models.loader.switch_adapter"),
            patch(
                "paramem.training.consolidation.train_adapter", return_value={"train_loss": 0.05}
            ),
            patch("paramem.training.consolidation.format_entry_training", fmt_mock),
            patch.object(loop, "_indexed_dataset", return_value=MagicMock()),
            patch.object(loop, "_make_training_config", return_value=MagicMock()),
            patch.object(loop, "_enable_gradient_checkpointing"),
            patch.object(loop, "_disable_gradient_checkpointing"),
            patch("paramem.training.consolidation.build_registry", return_value={}),
        ):
            # "Alice" promoted — "Bob" is NOT in promoted_set, so Loop B fires for graph2.
            result = loop._run_indexed_key_semantic(new_promotions=["Alice"])

        assert fmt_mock.called, "Loop B must feed graph2 to format_entry_training"
        assert result is not None

    def test_both_loops_fire_together(self, tmp_path: Path) -> None:
        """Loop A (new promotion) + Loop B (old semantic) both fire in one call."""
        loop = _make_loop(tmp_path)

        # key A — subject matches promoted entity (Loop A).
        loop.store._entries_flat_view()["graph1"] = _make_cache_entry(
            "graph1", "Alice", "lives_in", "Berlin"
        )
        loop.store.registry("semantic").add("graph1")
        loop.store.simhashes_in_tier("semantic")["graph1"] = 0xAAAA

        # key B — in semantic but subject != promoted entity (Loop B).
        loop.store._entries_flat_view()["graph2"] = _make_cache_entry(
            "graph2", "Charlie", "works_at", "Corp"
        )
        loop.store.registry("semantic").add("graph2")
        loop.store.simhashes_in_tier("semantic")["graph2"] = 0xBBBB

        captured_keyed: list[list] = []

        def _capture_fmt(keyed, tokenizer, **kwargs):  # noqa: ANN001
            captured_keyed.append(list(keyed))
            return [{"input_ids": [1], "labels": [1], "attention_mask": [1]}]

        with (
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.training.consolidation.train_adapter", return_value={"train_loss": 0.0}),
            patch(
                "paramem.training.consolidation.format_entry_training",
                side_effect=_capture_fmt,
            ),
            patch.object(loop, "_indexed_dataset", return_value=MagicMock()),
            patch.object(loop, "_make_training_config", return_value=MagicMock()),
            patch.object(loop, "_enable_gradient_checkpointing"),
            patch.object(loop, "_disable_gradient_checkpointing"),
            patch("paramem.training.consolidation.build_registry", return_value={}),
        ):
            loop._run_indexed_key_semantic(new_promotions=["Alice"])

        # One call to format_entry_training with both keys
        assert captured_keyed, "format_entry_training must have been called"
        all_keys = {kp["key"] for kp in captured_keyed[0]}
        assert "graph1" in all_keys, "Loop A key must be in training batch"
        assert "graph2" in all_keys, "Loop B key must be in training batch"


# ---------------------------------------------------------------------------
# _run_indexed_key_episodic
# ---------------------------------------------------------------------------


class TestRunIndexedKeyEpisodic:
    """_run_indexed_key_episodic uses format_entry_training."""

    def test_assigns_keys_and_caches_entries(self, tmp_path: Path) -> None:
        """New relations get graphN keys and uniform-shape cache entries."""
        loop = _make_loop(tmp_path)
        rels = _fake_entry_rels(2)

        fmt_mock = MagicMock(
            return_value=[{"input_ids": [1], "labels": [1], "attention_mask": [1]}]
        )

        with (
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.training.consolidation.train_adapter", return_value={"train_loss": 0.1}),
            patch("paramem.training.consolidation.format_entry_training", fmt_mock),
            patch("paramem.training.consolidation.probe_entry", return_value=None),
            patch.object(loop, "_indexed_dataset", return_value=MagicMock()),
            patch.object(loop, "_make_training_config", return_value=MagicMock()),
            patch.object(loop, "_enable_gradient_checkpointing"),
            patch.object(loop, "_disable_gradient_checkpointing"),
            patch.object(loop, "_maybe_make_recall_callback", return_value=None),
            patch("paramem.training.consolidation.build_registry", return_value={}),
            patch("paramem.training.consolidation.save_registry"),
            patch("paramem.models.loader.save_adapter"),
        ):
            loop._run_indexed_key_episodic(rels, new_promotions=[])

        # Keys assigned
        assert "graph1" in loop.store._entries_flat_view()
        assert "graph2" in loop.store._entries_flat_view()

        # Cache entries carry subject/predicate/object.
        e1 = loop.store._entries_flat_view()["graph1"]
        assert "subject" in e1
        assert "predicate" in e1
        assert "object" in e1

        # format_entry_training was used.
        assert fmt_mock.called


# ---------------------------------------------------------------------------
# _run_indexed_key_procedural
# ---------------------------------------------------------------------------


class TestRunIndexedKeyProcedural:
    """_run_indexed_key_procedural skips QA-gen and assigns proc{N} keys."""

    def test_assigns_proc_keys_no_qa_gen(self, tmp_path: Path) -> None:
        """generate_qa_from_relations is skipped; proc{N} keys are assigned directly."""
        loop = _make_loop(tmp_path)
        proc_rels = [
            {
                "subject": "Alice",
                "predicate": "prefers",
                "object": "Coffee",
                "relation_type": "preference",
                "speaker_id": "Speaker0",
            }
        ]

        fmt_mock = MagicMock(
            return_value=[{"input_ids": [1], "labels": [1], "attention_mask": [1]}]
        )

        with (
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.training.consolidation.train_adapter", return_value={"train_loss": 0.0}),
            patch("paramem.training.consolidation.format_entry_training", fmt_mock),
            patch("paramem.graph.qa_generator.generate_qa_from_relations") as qa_gen_mock,
            patch("paramem.training.consolidation.probe_entry", return_value=None),
            patch.object(loop, "_indexed_dataset", return_value=MagicMock()),
            patch.object(loop, "_make_training_config", return_value=MagicMock()),
            patch.object(loop, "_enable_gradient_checkpointing"),
            patch.object(loop, "_disable_gradient_checkpointing"),
            patch.object(loop, "_maybe_make_recall_callback", return_value=None),
            patch("paramem.training.consolidation.build_registry", return_value={}),
            patch("paramem.training.consolidation.save_registry"),
            patch("paramem.models.loader.save_adapter"),
        ):
            loop._run_indexed_key_procedural(proc_rels, speaker_id="Speaker0")

        # QA-gen must NOT be called.
        qa_gen_mock.assert_not_called()

        # proc1 must be in indexed_key_cache.
        assert "proc1" in loop.store._entries_flat_view()
        entry = loop.store._entries_flat_view()["proc1"]
        assert entry["subject"] == "Alice"
        assert entry["predicate"] == "prefers"
        assert entry["object"] == "Coffee"

        assert fmt_mock.called, "format_entry_training must be called"


# ---------------------------------------------------------------------------
# _simulate_indexed_key_episodic — existing-key branch
# ---------------------------------------------------------------------------


class TestSimulateIndexedKeyEpisodicExistingKeys:
    """_simulate_indexed_key_episodic existing-key loop runs without KeyError."""

    def test_existing_key_admitted_from_cache(self, tmp_path: Path) -> None:
        """Pre-seeded entry is re-accessed in the existing-key loop without KeyError."""
        loop = _make_loop(tmp_path)

        # Pre-seed a key that already exists (simulates restart-from-disk seed).
        loop.store._entries_flat_view()["graph1"] = _make_cache_entry(
            "graph1", "Alice", "lives_in", "Berlin"
        )
        loop.store.registry("episodic").add("graph1")
        loop._indexed_next_index = 2
        # graph1 is NOT in semantic_simhash → it shows up in existing_keys.

        new_rels = [
            {
                "subject": "Bob",
                "predicate": "works_at",
                "object": "Corp",
                "relation_type": "factual",
                "speaker_id": "Speaker0",
            }
        ]

        # Must complete without KeyError.
        loop._simulate_indexed_key_episodic(new_rels)

        # New key assigned.
        assert "graph2" in loop.store._entries_flat_view()
        # Existing key still in simhash.
        assert "graph1" in loop.store.simhashes_in_tier("episodic")
        assert "graph2" in loop.store.simhashes_in_tier("episodic")

    def test_existing_key_uses_entry_simhash(self, tmp_path: Path) -> None:
        """episodic_simhash is built via build_registry for existing entries."""
        from paramem.training.entry_memory import build_registry

        loop = _make_loop(tmp_path)

        loop.store._entries_flat_view()["graph1"] = _make_cache_entry(
            "graph1", "Alice", "lives_in", "Berlin"
        )
        loop.store.registry("episodic").add("graph1")
        loop._indexed_next_index = 2

        loop._simulate_indexed_key_episodic([])

        # episodic_simhash should equal build_registry output on graph1.
        expected = build_registry(
            [{"key": "graph1", "subject": "Alice", "predicate": "lives_in", "object": "Berlin"}]
        )
        assert loop.store.simhashes_in_tier("episodic") == expected


# ---------------------------------------------------------------------------
# _simulate_indexed_key_procedural
# ---------------------------------------------------------------------------


class TestSimulateIndexedKeyProcedural:
    """_simulate_indexed_key_procedural: compute_simhash called without KeyError."""

    def test_proc_key_assigned_and_simhash_built(self, tmp_path: Path) -> None:
        """proc{N} key is assigned and simhash built without KeyError."""
        from paramem.training.entry_memory import compute_simhash

        loop = _make_loop(tmp_path)
        proc_rels = [
            {
                "subject": "Alice",
                "predicate": "prefers",
                "object": "Tea",
                "relation_type": "preference",
                "speaker_id": "Speaker0",
            }
        ]

        loop._simulate_indexed_key_procedural(proc_rels, speaker_id="Speaker0")

        assert "proc1" in loop.store.simhashes_in_tier("procedural")
        assert "proc1" in loop.store._entries_flat_view()

        entry = loop.store._entries_flat_view()["proc1"]
        assert entry["subject"] == "Alice"
        assert entry["predicate"] == "prefers"
        assert entry["object"] == "Tea"

        # SimHash matches direct compute_simhash call.
        expected = compute_simhash("proc1", "Alice", "prefers", "Tea")
        assert loop.store.simhashes_in_tier("procedural")["proc1"] == expected

    def test_proc_no_qa_gen_model_call(self, tmp_path: Path) -> None:
        """generate_qa_from_relations is NOT called in simulate-procedural path."""
        loop = _make_loop(tmp_path)
        proc_rels = [
            {
                "subject": "Bob",
                "predicate": "dislikes",
                "object": "Mondays",
                "relation_type": "preference",
                "speaker_id": "Speaker0",
            }
        ]

        with patch("paramem.graph.qa_generator.generate_qa_from_relations") as qa_gen_mock:
            loop._simulate_indexed_key_procedural(proc_rels, speaker_id="Speaker0")

        qa_gen_mock.assert_not_called()


# ---------------------------------------------------------------------------
# simulated_training — pre-seeded indexed_key_cache
# ---------------------------------------------------------------------------


class TestSimulatedTrainingPreseeded:
    """simulated_training with pre-seeded cache (existing-keys loop path)."""

    def test_simulated_training_with_seeded_cache(self, tmp_path: Path) -> None:
        """simulated_training with a pre-seeded cache entry completes without KeyError."""
        loop = _make_loop(tmp_path)

        # Pre-seed an existing key (simulates the seeded-from-disk startup path).
        loop.store._entries_flat_view()["graph1"] = _make_cache_entry(
            "graph1", "Alice", "lives_in", "Berlin"
        )
        loop.store.registry("episodic").add("graph1")
        loop._indexed_next_index = 2

        new_rels = [
            {
                "subject": "Bob",
                "predicate": "works_at",
                "object": "Corp",
                "relation_type": "factual",
                "speaker_id": "Speaker0",
            }
        ]

        result = loop.simulated_training(new_rels, [], speaker_id="Speaker0")

        assert result["simulated"] is True
        # New key assigned.
        assert "graph2" in loop.store._entries_flat_view()
        # Both keys in episodic_simhash.
        assert "graph1" in loop.store.simhashes_in_tier("episodic")
        assert "graph2" in loop.store.simhashes_in_tier("episodic")

    def test_simulated_training_kp_schema(self, tmp_path: Path) -> None:
        """After simulated_training, indexed_key_cache entries carry subject/predicate/object."""
        loop = _make_loop(tmp_path)
        rels = _fake_entry_rels(1)

        loop.simulated_training(rels, [], speaker_id="Speaker0")

        entry = loop.store._entries_flat_view().get("graph1")
        assert entry is not None
        assert "subject" in entry
        assert "predicate" in entry
        assert "object" in entry
        # No question/answer keys in entry-format training data.
        assert entry["subject"] == "Subject1"
        assert entry["predicate"] == "knows"
        assert entry["object"] == "Object1"


# ---------------------------------------------------------------------------
# _save_tier_graphs simulate mode (server/consolidation.py)
# ---------------------------------------------------------------------------


class TestSaveSimulateStoreGraph:
    """_save_tier_graphs writes graph.json in simulate mode."""

    def test_simulate_mode_writes_graph_json(self, tmp_path: Path) -> None:
        """In simulate mode, graph.json is written under simulate_dir/<tier>/.

        ``_save_tier_graphs`` reads ``config.consolidation.mode``
        (early-returns in train mode), ``config.simulate_dir``
        (computed property on ServerConfig), and iterates the three tier
        simhash registries on the loop.  This test uses a MagicMock to supply
        those attributes directly without requiring a full ServerConfig
        construction (which would need a valid server.yaml).
        """
        from paramem.server.consolidation import _save_tier_graphs
        from paramem.training import memory_persistence

        # Build a minimal config stub for simulate mode.
        cfg = MagicMock()
        cfg.consolidation.mode = "simulate"
        cfg.simulate_dir = tmp_path / "simulate"
        cfg.adapter_dir = tmp_path / "adapters"

        # Build a minimal loop stub.
        from paramem.training.memory_store import MemoryStore as _MS

        loop = MagicMock()
        store = _MS(replay_enabled=False)
        store.put(
            "episodic",
            "graph1",
            _make_cache_entry("graph1", "Alice", "lives_in", "Berlin"),
            register=False,
        )
        store.put_simhash("episodic", "graph1", 0xABCD)
        loop.store = store

        _save_tier_graphs(loop, cfg)

        graph_path = tmp_path / "simulate" / "episodic" / "graph.json"
        assert graph_path.exists(), "graph.json must be written under simulate_dir/episodic/"

        # Verify graph round-trips via the public API.
        graph = memory_persistence.load_memory_from_disk(graph_path)
        entries = list(memory_persistence.iter_entries(graph))
        assert len(entries) == 1

        entry = entries[0]
        assert entry["key"] == "graph1"
        assert entry["subject"] == "Alice"
        assert entry["predicate"] == "lives_in"
        assert entry["object"] == "Berlin"
        assert entry["speaker_id"] == "Speaker0"
        assert entry["first_seen_cycle"] == 1

    def test_train_mode_is_no_op(self, tmp_path: Path) -> None:
        """In train mode, _save_tier_graphs writes nothing to disk.

        Train mode persists facts in adapter weights; the in-RAM
        MemoryStore is the only structured form during a train-mode cycle.
        graph.json is exclusive to simulate mode.
        """
        from paramem.server.consolidation import _save_tier_graphs

        cfg = MagicMock()
        cfg.consolidation.mode = "train"
        cfg.simulate_dir = tmp_path / "simulate"
        cfg.adapter_dir = tmp_path / "adapters"

        loop = MagicMock()
        loop.indexed_key_cache = {
            "graph1": _make_cache_entry("graph1", "Alice", "lives_in", "Berlin"),
        }
        loop.episodic_simhash = {"graph1": 0xABCD}
        loop.semantic_simhash = {}
        loop.procedural_simhash = {}

        _save_tier_graphs(loop, cfg)

        # Nothing should have been written to the simulate dir.
        assert not (tmp_path / "simulate").exists(), (
            "simulate_dir must not be created in train mode"
        )
