"""S2 regression tests: run_cycle and simulated_training quad-mode coverage.

Covers:
- _run_indexed_key_semantic quad mode: both Loop A (newly-promoted keys) and
  Loop B (previously-promoted keys still in semantic) fire without KeyError
  and the semantic simhash is updated from quad-shaped entries.
- _run_indexed_key_episodic quad mode: key assignment, cache population, and
  format_quadruple_training dispatched (not format_indexed_training).
- _run_indexed_key_procedural quad mode: keys assigned directly, no QA-gen
  model.generate called, contradiction handling works.
- _simulate_indexed_key_episodic quad mode: existing-key re-access branch
  (the ``if self._is_quad:`` loop at ~:1031) runs without KeyError when
  indexed_key_cache is pre-seeded with quad-shaped entries.
- _simulate_indexed_key_procedural quad mode: compute_simhash-quad path
  (``compute_simhash_quad``) called without KeyError; procedural_sp_index
  updated.
- _save_simulate_store_graph (server/consolidation.py) simulate-mode write
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


def _make_loop(tmp_path: Path, *, indexed_format: str = "quad") -> Any:
    """Construct a bare ConsolidationLoop without calling __init__."""
    model = _make_stub_model("episodic", "semantic", "procedural")

    loop = ConsolidationLoop.__new__(ConsolidationLoop)
    loop.model = model
    loop.tokenizer = MagicMock()
    cfg = ConsolidationConfig(
        indexed_key_replay_enabled=True,
        indexed_format=indexed_format,
    )
    loop.config = cfg
    loop._indexed_format = indexed_format
    loop._is_quad = indexed_format == "quad"
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
    loop.indexed_key_registry = KeyRegistry()
    loop.indexed_key_cache: dict[str, Any] = {}
    loop._indexed_next_index = 1
    loop._procedural_next_index = 1
    loop.episodic_simhash: dict = {}
    loop.semantic_simhash: dict = {}
    loop.procedural_simhash: dict = {}
    loop.procedural_sp_index: dict = {}
    loop.cycle_count = 1
    loop.key_sessions: dict = {}
    loop.promoted_keys: set = set()
    loop.shutdown_requested = False
    loop.merger = MagicMock()
    loop.fingerprint_cache = None
    return loop


def _quad_cache_entry(
    key: str,
    subject: str,
    predicate: str,
    obj: str,
    *,
    speaker_id: str = "Speaker0",
    first_seen_cycle: int = 1,
) -> dict:
    """Build a minimal Option-B uniform-shape cache entry for quad mode.

    Carries both ``subject``/``predicate``/``object`` (quad-native) and
    the ``source_*`` aliases (for promotion-matching which reads ``source_*``).
    No ``question``/``answer`` in quad mode.
    """
    return {
        "key": key,
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "source_subject": subject,
        "source_predicate": predicate,
        "source_object": obj,
        "speaker_id": speaker_id,
        "first_seen_cycle": first_seen_cycle,
    }


def _fake_quad_rels(n: int = 2, *, speaker_id: str = "Speaker0") -> list[dict]:
    """Return n synthetic relation dicts suitable as quad episodic input."""
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
# _run_indexed_key_semantic — quad mode, both loops
# ---------------------------------------------------------------------------


class TestRunIndexedKeySemanticQuad:
    """_run_indexed_key_semantic in quad mode: Loop A + Loop B fire without KeyError."""

    def test_loop_a_newly_promoted_key_no_keyerror(self, tmp_path: Path) -> None:
        """Loop A: key matches promoted_set via source_subject → no KeyError."""
        loop = _make_loop(tmp_path, indexed_format="quad")

        # Pre-seed one key whose source_subject matches the promoted entity.
        loop.indexed_key_cache["graph1"] = _quad_cache_entry(
            "graph1", "Alice", "lives_in", "Berlin"
        )
        loop.indexed_key_registry.add("graph1")
        loop.semantic_simhash["graph1"] = 0xABCD

        fmt_mock = MagicMock(
            return_value=[{"input_ids": [1], "labels": [1], "attention_mask": [1]}]
        )

        with (
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.training.consolidation.train_adapter", return_value={"train_loss": 0.1}),
            patch("paramem.training.consolidation.format_quadruple_training", fmt_mock),
            patch("paramem.training.consolidation.format_indexed_training") as qa_fmt,
            patch.object(loop, "_indexed_dataset", return_value=MagicMock()),
            patch.object(loop, "_make_training_config", return_value=MagicMock()),
            patch.object(loop, "_enable_gradient_checkpointing"),
            patch.object(loop, "_disable_gradient_checkpointing"),
            patch("paramem.training.consolidation.build_registry_quad", return_value={}),
        ):
            # "Alice" is in the promoted set — Loop A should fire.
            result = loop._run_indexed_key_semantic(new_promotions=["Alice"])

        # quad format used, QA format not used
        assert fmt_mock.called, "format_quadruple_training must be called in quad mode"
        qa_fmt.assert_not_called()
        # Result is a float (train_loss) not None
        assert result is not None

    def test_loop_b_previously_promoted_key_no_keyerror(self, tmp_path: Path) -> None:
        """Loop B: key in semantic_simhash but not in promoted_set → still processed."""
        loop = _make_loop(tmp_path, indexed_format="quad")

        # Pre-seed a key in semantic (previously promoted, no longer in new_promotions).
        loop.indexed_key_cache["graph2"] = _quad_cache_entry("graph2", "Bob", "works_at", "ACME")
        loop.indexed_key_registry.add("graph2")
        loop.semantic_simhash["graph2"] = 0xDEAD

        fmt_mock = MagicMock(
            return_value=[{"input_ids": [1], "labels": [1], "attention_mask": [1]}]
        )

        with (
            patch("paramem.models.loader.switch_adapter"),
            patch(
                "paramem.training.consolidation.train_adapter", return_value={"train_loss": 0.05}
            ),
            patch("paramem.training.consolidation.format_quadruple_training", fmt_mock),
            patch("paramem.training.consolidation.format_indexed_training") as qa_fmt,
            patch.object(loop, "_indexed_dataset", return_value=MagicMock()),
            patch.object(loop, "_make_training_config", return_value=MagicMock()),
            patch.object(loop, "_enable_gradient_checkpointing"),
            patch.object(loop, "_disable_gradient_checkpointing"),
            patch("paramem.training.consolidation.build_registry_quad", return_value={}),
        ):
            # "Alice" promoted — "Bob" is NOT in promoted_set, so Loop B fires for graph2.
            result = loop._run_indexed_key_semantic(new_promotions=["Alice"])

        assert fmt_mock.called, "Loop B must feed graph2 to format_quadruple_training"
        qa_fmt.assert_not_called()
        assert result is not None

    def test_both_loops_fire_together(self, tmp_path: Path) -> None:
        """Loop A (new promotion) + Loop B (old semantic) both fire in one call."""
        loop = _make_loop(tmp_path, indexed_format="quad")

        # key A — subject matches promoted entity (Loop A).
        loop.indexed_key_cache["graph1"] = _quad_cache_entry(
            "graph1", "Alice", "lives_in", "Berlin"
        )
        loop.indexed_key_registry.add("graph1")
        loop.semantic_simhash["graph1"] = 0xAAAA

        # key B — in semantic but source_subject != promoted entity (Loop B).
        loop.indexed_key_cache["graph2"] = _quad_cache_entry(
            "graph2", "Charlie", "works_at", "Corp"
        )
        loop.indexed_key_registry.add("graph2")
        loop.semantic_simhash["graph2"] = 0xBBBB

        captured_keyed: list[list] = []

        def _capture_fmt(keyed, tokenizer, **kwargs):  # noqa: ANN001
            captured_keyed.append(list(keyed))
            return [{"input_ids": [1], "labels": [1], "attention_mask": [1]}]

        with (
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.training.consolidation.train_adapter", return_value={"train_loss": 0.0}),
            patch(
                "paramem.training.consolidation.format_quadruple_training",
                side_effect=_capture_fmt,
            ),
            patch.object(loop, "_indexed_dataset", return_value=MagicMock()),
            patch.object(loop, "_make_training_config", return_value=MagicMock()),
            patch.object(loop, "_enable_gradient_checkpointing"),
            patch.object(loop, "_disable_gradient_checkpointing"),
            patch("paramem.training.consolidation.build_registry_quad", return_value={}),
        ):
            loop._run_indexed_key_semantic(new_promotions=["Alice"])

        # One call to format_quadruple_training with both keys
        assert captured_keyed, "format_quadruple_training must have been called"
        all_keys = {kp["key"] for kp in captured_keyed[0]}
        assert "graph1" in all_keys, "Loop A key must be in training batch"
        assert "graph2" in all_keys, "Loop B key must be in training batch"

    def test_qa_path_unchanged(self, tmp_path: Path) -> None:
        """QA path (indexed_format='qa') still calls format_indexed_training."""
        loop = _make_loop(tmp_path, indexed_format="qa")

        # QA-shaped cache entry.
        loop.indexed_key_cache["graph1"] = {
            "key": "graph1",
            "question": "Where does Alice live?",
            "answer": "Berlin.",
            "source_subject": "Alice",
            "source_predicate": "lives_in",
            "source_object": "Berlin",
            "subject": "Alice",
            "predicate": "lives_in",
            "object": "Berlin",
            "speaker_id": "Speaker0",
            "first_seen_cycle": 1,
        }
        loop.indexed_key_registry.add("graph1")
        loop.semantic_simhash["graph1"] = 0xAAAA

        qa_fmt_mock = MagicMock(
            return_value=[{"input_ids": [1], "labels": [1], "attention_mask": [1]}]
        )

        with (
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.training.consolidation.train_adapter", return_value={"train_loss": 0.1}),
            patch("paramem.training.consolidation.format_quadruple_training") as quad_fmt,
            patch("paramem.training.consolidation.format_indexed_training", qa_fmt_mock),
            patch.object(loop, "_indexed_dataset", return_value=MagicMock()),
            patch.object(loop, "_make_training_config", return_value=MagicMock()),
            patch.object(loop, "_enable_gradient_checkpointing"),
            patch.object(loop, "_disable_gradient_checkpointing"),
            patch(
                "paramem.training.indexed_memory.build_registry",
                return_value={},
            ),
        ):
            result = loop._run_indexed_key_semantic(new_promotions=["Alice"])

        # QA format used; quad format not.
        qa_fmt_mock.assert_called_once()
        quad_fmt.assert_not_called()
        assert result is not None


# ---------------------------------------------------------------------------
# _run_indexed_key_episodic — quad mode
# ---------------------------------------------------------------------------


class TestRunIndexedKeyEpisodicQuad:
    """_run_indexed_key_episodic in quad mode uses format_quadruple_training."""

    def test_quad_assigns_keys_and_caches_entries(self, tmp_path: Path) -> None:
        """New quad relations get graphN keys and uniform-shape cache entries."""
        loop = _make_loop(tmp_path, indexed_format="quad")
        rels = _fake_quad_rels(2)

        fmt_mock = MagicMock(
            return_value=[{"input_ids": [1], "labels": [1], "attention_mask": [1]}]
        )

        with (
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.training.consolidation.train_adapter", return_value={"train_loss": 0.1}),
            patch("paramem.training.consolidation.format_quadruple_training", fmt_mock),
            patch("paramem.training.consolidation.format_indexed_training") as qa_fmt,
            patch("paramem.training.consolidation.probe_quad", return_value=None),
            patch.object(loop, "_indexed_dataset", return_value=MagicMock()),
            patch.object(loop, "_make_training_config", return_value=MagicMock()),
            patch.object(loop, "_enable_gradient_checkpointing"),
            patch.object(loop, "_disable_gradient_checkpointing"),
            patch.object(loop, "_maybe_make_recall_callback", return_value=None),
            patch("paramem.training.consolidation.build_registry_quad", return_value={}),
            patch("paramem.training.consolidation.save_registry"),
            patch("paramem.models.loader.save_adapter"),
        ):
            loop._run_indexed_key_episodic(rels, new_promotions=[])

        # Keys assigned
        assert "graph1" in loop.indexed_key_cache
        assert "graph2" in loop.indexed_key_cache

        # Cache entries are quad-shaped (subject/predicate/object present)
        e1 = loop.indexed_key_cache["graph1"]
        assert "subject" in e1
        assert "predicate" in e1
        assert "object" in e1

        # format_quadruple_training was used; format_indexed_training was not.
        assert fmt_mock.called
        qa_fmt.assert_not_called()


# ---------------------------------------------------------------------------
# _run_indexed_key_procedural — quad mode
# ---------------------------------------------------------------------------


class TestRunIndexedKeyProceduralQuad:
    """_run_indexed_key_procedural in quad mode skips QA-gen and assigns quad keys."""

    def test_quad_assigns_proc_keys_no_qa_gen(self, tmp_path: Path) -> None:
        """Quad mode skips generate_qa_from_relations and assigns proc{N} keys."""
        loop = _make_loop(tmp_path, indexed_format="quad")
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
            patch("paramem.training.consolidation.format_quadruple_training", fmt_mock),
            patch("paramem.training.consolidation.generate_qa_from_relations") as qa_gen_mock,
            patch("paramem.training.consolidation.probe_quad", return_value=None),
            patch.object(loop, "_indexed_dataset", return_value=MagicMock()),
            patch.object(loop, "_make_training_config", return_value=MagicMock()),
            patch.object(loop, "_enable_gradient_checkpointing"),
            patch.object(loop, "_disable_gradient_checkpointing"),
            patch.object(loop, "_maybe_make_recall_callback", return_value=None),
            patch("paramem.training.consolidation.build_registry_quad", return_value={}),
            patch("paramem.training.consolidation.save_registry"),
            patch("paramem.models.loader.save_adapter"),
        ):
            loop._run_indexed_key_procedural(proc_rels, speaker_id="Speaker0")

        # QA-gen must NOT be called in quad mode.
        qa_gen_mock.assert_not_called()

        # proc1 must be in indexed_key_cache with quad shape.
        assert "proc1" in loop.indexed_key_cache
        entry = loop.indexed_key_cache["proc1"]
        assert entry["subject"] == "Alice"
        assert entry["predicate"] == "prefers"
        assert entry["object"] == "Coffee"

        assert fmt_mock.called, "format_quadruple_training must be called in quad mode"


# ---------------------------------------------------------------------------
# _simulate_indexed_key_episodic — quad mode, existing-key branch
# ---------------------------------------------------------------------------


class TestSimulateIndexedKeyEpisodicQuadExistingKeys:
    """_simulate_indexed_key_episodic existing-key loop (quad mode) runs without KeyError."""

    def test_existing_key_admitted_from_quad_cache(self, tmp_path: Path) -> None:
        """Pre-seeded quad entry is re-accessed in the existing-key loop without KeyError."""
        loop = _make_loop(tmp_path, indexed_format="quad")

        # Pre-seed a key that already exists (simulates restart-from-disk seed).
        loop.indexed_key_cache["graph1"] = _quad_cache_entry(
            "graph1", "Alice", "lives_in", "Berlin"
        )
        loop.indexed_key_registry.add("graph1")
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
        assert "graph2" in loop.indexed_key_cache
        # Existing key still in simhash.
        assert "graph1" in loop.episodic_simhash
        assert "graph2" in loop.episodic_simhash

    def test_existing_key_uses_quad_simhash(self, tmp_path: Path) -> None:
        """episodic_simhash is built via build_registry_quad for quad entries."""
        from paramem.training.quadruple_memory import build_registry as build_registry_quad

        loop = _make_loop(tmp_path, indexed_format="quad")

        loop.indexed_key_cache["graph1"] = _quad_cache_entry(
            "graph1", "Alice", "lives_in", "Berlin"
        )
        loop.indexed_key_registry.add("graph1")
        loop._indexed_next_index = 2

        loop._simulate_indexed_key_episodic([])

        # episodic_simhash should equal quad build_registry output on graph1.
        expected = build_registry_quad(
            [{"key": "graph1", "subject": "Alice", "predicate": "lives_in", "object": "Berlin"}]
        )
        assert loop.episodic_simhash == expected


# ---------------------------------------------------------------------------
# _simulate_indexed_key_procedural — quad mode
# ---------------------------------------------------------------------------


class TestSimulateIndexedKeyProceduralQuad:
    """_simulate_indexed_key_procedural quad path: compute_simhash_quad called."""

    def test_quad_proc_key_assigned_and_simhash_built(self, tmp_path: Path) -> None:
        """Quad procedural: proc{N} key + compute_simhash_quad without KeyError."""
        from paramem.training.quadruple_memory import compute_simhash as compute_simhash_quad

        loop = _make_loop(tmp_path, indexed_format="quad")
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

        assert "proc1" in loop.procedural_simhash
        assert "proc1" in loop.indexed_key_cache

        entry = loop.indexed_key_cache["proc1"]
        assert entry["subject"] == "Alice"
        assert entry["predicate"] == "prefers"
        assert entry["object"] == "Tea"

        # SimHash matches direct compute_simhash_quad call.
        expected = compute_simhash_quad("proc1", "Alice", "prefers", "Tea")
        assert loop.procedural_simhash["proc1"] == expected

    def test_quad_proc_no_qa_gen_model_call(self, tmp_path: Path) -> None:
        """generate_qa_from_relations is NOT called in quad simulate-procedural path."""
        loop = _make_loop(tmp_path, indexed_format="quad")
        proc_rels = [
            {
                "subject": "Bob",
                "predicate": "dislikes",
                "object": "Mondays",
                "relation_type": "preference",
                "speaker_id": "Speaker0",
            }
        ]

        with patch("paramem.training.consolidation.generate_qa_from_relations") as qa_gen_mock:
            loop._simulate_indexed_key_procedural(proc_rels, speaker_id="Speaker0")

        qa_gen_mock.assert_not_called()


# ---------------------------------------------------------------------------
# simulated_training — quad mode with pre-seeded quad indexed_key_cache
# ---------------------------------------------------------------------------


class TestSimulatedTrainingQuadPreseeded:
    """simulated_training with pre-seeded quad cache (existing-keys loop path)."""

    def test_simulated_training_with_seeded_quad_cache(self, tmp_path: Path) -> None:
        """simulated_training with a pre-seeded quad cache entry completes without KeyError."""
        loop = _make_loop(tmp_path, indexed_format="quad")

        # Pre-seed an existing quad key (simulates the seeded-from-disk startup path).
        loop.indexed_key_cache["graph1"] = _quad_cache_entry(
            "graph1", "Alice", "lives_in", "Berlin"
        )
        loop.indexed_key_registry.add("graph1")
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
        assert "graph2" in loop.indexed_key_cache
        # Both keys in episodic_simhash.
        assert "graph1" in loop.episodic_simhash
        assert "graph2" in loop.episodic_simhash

    def test_simulated_training_quad_kp_schema(self, tmp_path: Path) -> None:
        """After simulated_training, indexed_key_cache entries have quad shape."""
        loop = _make_loop(tmp_path, indexed_format="quad")
        rels = _fake_quad_rels(1)

        loop.simulated_training(rels, [], speaker_id="Speaker0")

        entry = loop.indexed_key_cache.get("graph1")
        assert entry is not None
        assert "subject" in entry
        assert "predicate" in entry
        assert "object" in entry
        # quad mode: no question/answer in the training-set build
        # (source_* aliases present via Option-B uniform shape)
        assert entry["subject"] == "Subject1"
        assert entry["predicate"] == "knows"
        assert entry["object"] == "Object1"


# ---------------------------------------------------------------------------
# _save_simulate_store_graph simulate mode (server/consolidation.py)
# ---------------------------------------------------------------------------


class TestSaveSimulateStoreGraph:
    """_save_simulate_store_graph writes graph.json in simulate mode."""

    def test_simulate_mode_writes_graph_json(self, tmp_path: Path) -> None:
        """In simulate mode, graph.json is written under simulate_dir/<tier>/.

        ``_save_simulate_store_graph`` reads ``config.consolidation.mode``
        (early-returns in train mode), ``config.simulate_dir``
        (computed property on ServerConfig), and iterates the three tier
        simhash registries on the loop.  This test uses a MagicMock to supply
        those attributes directly without requiring a full ServerConfig
        construction (which would need a valid server.yaml).
        """
        from paramem.server import simulate_store
        from paramem.server.consolidation import _save_simulate_store_graph

        # Build a minimal config stub for simulate mode.
        cfg = MagicMock()
        cfg.consolidation.mode = "simulate"
        cfg.simulate_dir = tmp_path / "simulate"
        cfg.adapter_dir = tmp_path / "adapters"

        # Build a minimal loop stub with quad-shaped indexed_key_cache.
        loop = MagicMock()
        loop.indexed_key_cache = {
            "graph1": _quad_cache_entry("graph1", "Alice", "lives_in", "Berlin"),
        }
        loop.episodic_simhash = {"graph1": 0xABCD}
        loop.semantic_simhash = {}
        loop.procedural_simhash = {}

        _save_simulate_store_graph(loop, cfg)

        graph_path = tmp_path / "simulate" / "episodic" / "graph.json"
        assert graph_path.exists(), "graph.json must be written under simulate_dir/episodic/"

        # Verify graph round-trips via the public API.
        graph = simulate_store.load_simulate_graph(graph_path)
        quads = list(simulate_store.iter_quads(graph))
        assert len(quads) == 1

        quad = quads[0]
        assert quad["key"] == "graph1"
        assert quad["subject"] == "Alice"
        assert quad["predicate"] == "lives_in"
        assert quad["object"] == "Berlin"
        assert quad["speaker_id"] == "Speaker0"
        assert quad["first_seen_cycle"] == 1

    def test_train_mode_is_no_op(self, tmp_path: Path) -> None:
        """In train mode, _save_simulate_store_graph writes nothing to disk."""
        from paramem.server.consolidation import _save_simulate_store_graph

        cfg = MagicMock()
        cfg.consolidation.mode = "train"
        cfg.simulate_dir = tmp_path / "simulate"
        cfg.adapter_dir = tmp_path / "adapters"

        loop = MagicMock()
        loop.indexed_key_cache = {
            "graph1": _quad_cache_entry("graph1", "Alice", "lives_in", "Berlin"),
        }
        loop.episodic_simhash = {"graph1": 0xABCD}
        loop.semantic_simhash = {}
        loop.procedural_simhash = {}

        _save_simulate_store_graph(loop, cfg)

        # Nothing should have been written to the simulate dir.
        assert not (tmp_path / "simulate").exists(), (
            "simulate_dir must not be created in train mode"
        )
