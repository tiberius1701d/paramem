"""Tests for the active-store migration helper.

Covers:
- State-file model: round-trip, all_tiers_done, mode-switch validation.
- Detection: state file precedence, fresh-detect from yaml mode + on-disk
  store contents.
- train_to_simulate per-tier: happy-path weight reconstruction + graph write;
  rollback on sanity-check failure.
- simulate_to_train per-tier: happy-path graph.json read + train + cleanup;
  rollback on recall probe failure.
- migrate(): per-tier success advances state; failure isolates to one
  tier; all-tiers-done removes the state file.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import networkx as nx
import pytest

from paramem.memory.persistence import _IK_KEY_ATTR, save_memory_to_disk
from paramem.server.active_store_migration import (
    TIERS,
    MigrationState,
    _has_tier_graph,
    _migrate_tier_simulate_to_train,
    _migrate_tier_train_to_simulate,
    _TierSkipped,
    clear_state,
    detect_mode_switch,
    load_state,
    migrate,
    save_state,
    state_path,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(tmp_path: Path, mode: str = "train") -> MagicMock:
    cfg = MagicMock()
    cfg.adapter_dir = tmp_path / "adapters"
    cfg.simulate_dir = tmp_path / "simulate"
    cfg.consolidation = MagicMock()
    cfg.consolidation.mode = mode
    cfg.adapter_dir.mkdir(parents=True, exist_ok=True)
    cfg.simulate_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def _write_simulate_graph(simulate_dir: Path, tier: str, entries: list[dict]) -> Path:
    """Write a graph.json in the simulate-store layout (plaintext for tests).

    Uses :func:`paramem.memory.persistence.save_memory_to_disk` with
    ``encrypted=False`` so the file is human-readable in the test filesystem.
    """
    tier_dir = simulate_dir / tier
    tier_dir.mkdir(parents=True, exist_ok=True)
    graph_path = tier_dir / "graph.json"
    graph = nx.MultiDiGraph()
    for entry in entries:
        graph.add_edge(
            entry.get("subject", "Subject"),
            entry.get("object", "Object"),
            **{
                _IK_KEY_ATTR: entry["key"],
                "predicate": entry.get("predicate", "related_to"),
                "speaker_id": entry.get("speaker_id", "Speaker0"),
                "first_seen_cycle": entry.get("first_seen_cycle", 1),
            },
        )
    save_memory_to_disk(graph, graph_path, encrypted=False)
    return graph_path


def _write_adapter_registry(adapter_dir: Path, tier: str, keys: list[str]) -> Path:
    """Write an ``indexed_key_registry.json`` at the canonical per-tier path:
    ``<adapter_dir>/<tier>/indexed_key_registry.json``.

    Used by ``_has_adapter_registry`` / ``detect_mode_switch`` to detect that
    the train-mode adapter for *tier* exists.
    """
    tier_dir = adapter_dir / tier
    tier_dir.mkdir(parents=True, exist_ok=True)
    p = tier_dir / "indexed_key_registry.json"
    # Write a minimal KeyRegistry-compatible JSON (flat dict of key→metadata).
    registry = {k: {"status": "active"} for k in keys}
    p.write_bytes(json.dumps(registry).encode("utf-8"))
    return p


def _write_adapter_slot_dir(adapter_dir: Path, tier: str, slot_ts: str) -> Path:
    """Create an empty slot subdir to simulate trained-but-not-yet-collected state."""
    slot_dir = adapter_dir / tier / slot_ts
    slot_dir.mkdir(parents=True, exist_ok=True)
    return slot_dir


def _full_quad(key: str, predicate: str = "related_to") -> dict:
    """Return a full 6-field entry dict for migration test fixtures."""
    return {
        "key": key,
        "subject": "Subject",
        "predicate": predicate,
        "object": "Object",
        "speaker_id": "Speaker0",
        "first_seen_cycle": 1,
    }


# ---------------------------------------------------------------------------
# State model
# ---------------------------------------------------------------------------


class TestMigrationState:
    def test_for_mode_switch_simulate_to_train(self):
        s = MigrationState.for_mode_switch(source_mode="simulate", target_mode="train")
        assert s.direction == "simulate_to_train"
        assert s.source_mode == "simulate"
        assert s.target_mode == "train"
        assert s.completed_tiers == []
        assert s.failed_tiers == {}
        assert s.started_at  # non-empty iso8601

    def test_for_mode_switch_train_to_simulate(self):
        s = MigrationState.for_mode_switch(source_mode="train", target_mode="simulate")
        assert s.direction == "train_to_simulate"

    def test_for_mode_switch_rejects_same_modes(self):
        with pytest.raises(ValueError, match="both"):
            MigrationState.for_mode_switch(source_mode="train", target_mode="train")

    def test_for_mode_switch_rejects_unknown_mode(self):
        with pytest.raises(ValueError, match="must be 'simulate' or 'train'"):
            MigrationState.for_mode_switch(source_mode="train", target_mode="bogus")

    def test_all_tiers_done_when_complete(self):
        s = MigrationState.for_mode_switch(source_mode="simulate", target_mode="train")
        s.completed_tiers = list(TIERS)
        assert s.all_tiers_done is True

    def test_all_tiers_done_with_failure(self):
        s = MigrationState.for_mode_switch(source_mode="simulate", target_mode="train")
        s.completed_tiers = list(TIERS)
        s.failed_tiers = {"episodic": "boom"}
        assert s.all_tiers_done is False

    def test_all_tiers_done_partial(self):
        s = MigrationState.for_mode_switch(source_mode="simulate", target_mode="train")
        s.completed_tiers = ["episodic", "semantic"]
        assert s.all_tiers_done is False


# ---------------------------------------------------------------------------
# State file IO
# ---------------------------------------------------------------------------


class TestStateFileIO:
    def test_load_returns_none_when_absent(self, tmp_path):
        assert load_state(tmp_path) is None

    def test_save_then_load_roundtrip(self, tmp_path):
        original = MigrationState.for_mode_switch(source_mode="simulate", target_mode="train")
        original.completed_tiers = ["episodic"]
        original.failed_tiers = {"semantic": "rollback test"}
        save_state(tmp_path, original)
        loaded = load_state(tmp_path)
        assert loaded is not None
        assert loaded.direction == original.direction
        assert loaded.completed_tiers == original.completed_tiers
        assert loaded.failed_tiers == original.failed_tiers

    def test_clear_state_removes_file(self, tmp_path):
        s = MigrationState.for_mode_switch(source_mode="train", target_mode="simulate")
        save_state(tmp_path, s)
        assert state_path(tmp_path).exists()
        clear_state(tmp_path)
        assert not state_path(tmp_path).exists()

    def test_clear_state_idempotent(self, tmp_path):
        # Clearing when nothing exists must not raise.
        clear_state(tmp_path)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


class TestDetectModeSwitch:
    def test_fresh_install_no_migration(self, tmp_path):
        cfg = _make_config(tmp_path, mode="train")
        assert detect_mode_switch(cfg) is None

    def test_existing_state_takes_precedence(self, tmp_path):
        cfg = _make_config(tmp_path, mode="train")
        prior = MigrationState.for_mode_switch(source_mode="simulate", target_mode="train")
        prior.completed_tiers = ["episodic"]
        save_state(cfg.adapter_dir, prior)
        # Even with no on-disk source, the state file dictates resume.
        result = detect_mode_switch(cfg)
        assert result is not None
        assert result.completed_tiers == ["episodic"]

    def test_simulate_to_train_detected_via_graph_json(self, tmp_path):
        """graph.json present in adapter_dir triggers simulate→train detection.

        Under the unified layout, graph.json lives at adapter_dir/<tier>/graph.json
        in both train and simulate modes.  Detection fires when graph.json is present
        but no indexed_key_registry.json exists (simulate → train transition needed).
        """
        cfg = _make_config(tmp_path, mode="train")
        # Write graph to the unified layout location (adapter_dir), not simulate_dir.
        _write_simulate_graph(cfg.adapter_dir, "episodic", [_full_quad("g1")])
        result = detect_mode_switch(cfg)
        assert result is not None
        assert result.direction == "simulate_to_train"
        assert result.source_mode == "simulate"

    def test_train_to_simulate_detected(self, tmp_path):
        cfg = _make_config(tmp_path, mode="simulate")
        _write_adapter_registry(cfg.adapter_dir, "episodic", ["g1"])
        result = detect_mode_switch(cfg)
        assert result is not None
        assert result.direction == "train_to_simulate"
        assert result.source_mode == "train"

    def test_active_state_consistent_no_migration(self, tmp_path):
        # Both stores have content matching the operator's mode → no switch
        cfg = _make_config(tmp_path, mode="train")
        _write_adapter_registry(cfg.adapter_dir, "episodic", ["g1"])
        # Stale simulate-store from prior mode IS present, but adapter is too —
        # this is a "stale inactive store" case, not a mode switch.
        _write_simulate_graph(cfg.simulate_dir, "episodic", [_full_quad("g1")])
        assert detect_mode_switch(cfg) is None

    def test_unsupported_mode_returns_none(self, tmp_path):
        cfg = _make_config(tmp_path, mode="cloud_only")
        assert detect_mode_switch(cfg) is None


class TestHasTierGraph:
    """``_has_tier_graph`` must detect graph.json in both main-slot and interim subdirs.

    Regression for C4: encryption.py and active_store_migration.py previously
    looked only at ``<adapter_dir>/<tier>/graph.json``.  Simulate-mode interim
    cycles write to ``<adapter_dir>/<tier>/interim_<stamp>/graph.json`` — those
    must also be detected.
    """

    def test_main_slot_detected(self, tmp_path):
        """graph.json at tier root is detected."""
        adapter_dir = tmp_path / "adapters"
        tier_dir = adapter_dir / "episodic"
        tier_dir.mkdir(parents=True)
        (tier_dir / "graph.json").write_text("{}")
        assert _has_tier_graph(adapter_dir, "episodic") is True

    def test_interim_slot_detected(self, tmp_path):
        """graph.json under <tier>/interim_<stamp>/ is detected.

        This is the layout written by commit_tier_slot in simulate mode
        for interim cycles.  Prior to C4, _has_tier_graph only checked the
        tier root and would return False here.
        """
        adapter_dir = tmp_path / "adapters"
        interim_dir = adapter_dir / "episodic" / "interim_20260101T0000"
        interim_dir.mkdir(parents=True)
        (interim_dir / "graph.json").write_text("{}")
        assert _has_tier_graph(adapter_dir, "episodic") is True

    def test_missing_returns_false(self, tmp_path):
        """Absent tier dir → False."""
        adapter_dir = tmp_path / "adapters"
        assert _has_tier_graph(adapter_dir, "episodic") is False

    def test_tier_dir_exists_but_no_graph_returns_false(self, tmp_path):
        """Tier dir present but no graph.json anywhere → False."""
        adapter_dir = tmp_path / "adapters"
        (adapter_dir / "episodic").mkdir(parents=True)
        assert _has_tier_graph(adapter_dir, "episodic") is False


# ---------------------------------------------------------------------------
# Per-tier: train -> simulate
# ---------------------------------------------------------------------------


def _make_loop_train_to_simulate(
    tmp_path: Path,
    *,
    tier: str = "episodic",
    keys: list[str] | None = None,
) -> MagicMock:
    """Build a loop stub for train→simulate tests.

    The loop has:
    - An ``indexed_key_registry`` (dict[str, KeyRegistry]) with *keys*
      registered in the *tier* entry.
    - An ``indexed_key_cache`` pre-populated with matching entries.
    - A model that will yield a successful ``reconstruct_graph`` result
      (mocked).
    """
    from paramem.memory.store import MemoryStore as _MS

    if keys is None:
        keys = ["g0", "g1", "g2"]

    loop = MagicMock()
    store = _MS(replay_enabled=True)
    for k in keys:
        store.put(
            tier,
            k,
            {
                "subject": "Subject",
                "predicate": "related_to",
                "object": "Object",
                "speaker_id": "Speaker0",
                "first_seen_cycle": 1,
            },
        )
    loop.store = store
    loop.model = MagicMock()
    loop.tokenizer = MagicMock()
    loop.training_config = SimpleNamespace(gradient_checkpointing=False)

    return loop


class TestMigrateTierTrainToSimulate:
    """Train→simulate: reconstruct graph from weights, persist as graph.json."""

    def _make_graph_result(self, tier: str, keys: list[str]) -> MagicMock:
        """Build a mock ReconstructionResult whose graph has one edge per key."""
        graph = nx.MultiDiGraph()
        for key in keys:
            eid = graph.add_edge("Subject", "Object", predicate="related_to")
            graph["Subject"]["Object"][eid][_IK_KEY_ATTR] = key
        result = MagicMock()
        result.graph = graph
        result.failures = []
        return result

    def test_no_active_keys_skipped(self, tmp_path):
        """When registry has no active keys for the tier, raise _TierSkipped."""
        cfg = _make_config(tmp_path, mode="simulate")
        loop = _make_loop_train_to_simulate(tmp_path, keys=[])
        with pytest.raises(_TierSkipped, match="no active registry keys"):
            _migrate_tier_train_to_simulate(loop, cfg, "episodic")

    def test_none_registry_skipped(self, tmp_path):
        """When loop's store has replay disabled, raise _TierSkipped."""
        from paramem.memory.store import MemoryStore as _MS

        cfg = _make_config(tmp_path, mode="simulate")
        loop = MagicMock()
        loop.store = _MS(replay_enabled=False)
        with pytest.raises(_TierSkipped):
            _migrate_tier_train_to_simulate(loop, cfg, "episodic")

    def test_happy_path_writes_graph_and_removes_adapter_weight_slots(self, tmp_path):
        """Happy path: graph.json written under adapter_dir; weight slot dirs removed.

        Under the unified layout, graph.json lives at adapter_dir/<tier>/graph.json.
        Migration to simulate deletes weight slot subdirectories (those containing
        adapter_model.safetensors or adapter_config.json) but preserves graph.json
        and registry files at the top of the tier directory.
        """
        cfg = _make_config(tmp_path, mode="simulate")
        keys = ["g0", "g1"]
        loop = _make_loop_train_to_simulate(tmp_path, keys=keys)
        # Create an adapter slot dir so migration has something to remove.
        slot_dir = _write_adapter_slot_dir(cfg.adapter_dir, "episodic", "20260430-180000")
        # Create adapter_config.json so the slot is recognised as a weight slot.
        (slot_dir / "adapter_config.json").write_text("{}")

        reconstruction = self._make_graph_result("episodic", keys)

        with patch(
            "paramem.graph.reconstruct.reconstruct_graph",
            return_value=reconstruction,
        ):
            _migrate_tier_train_to_simulate(loop, cfg, "episodic")

        # Unified layout: graph.json lives under adapter_dir, not simulate_dir.
        target = cfg.adapter_dir / "episodic" / "graph.json"
        assert target.exists()

        # Verify the written graph has all expected keys
        from paramem.memory.persistence import iter_entries, load_memory_from_disk

        loaded = load_memory_from_disk(target)
        graph_keys = {q["key"] for q in iter_entries(loaded)}
        assert graph_keys == set(keys)

        # Weight slot subdirectory removed; tier dir still present (holds graph.json).
        assert not slot_dir.exists()
        assert (cfg.adapter_dir / "episodic").is_dir()

    def test_reconstruction_error_propagates(self, tmp_path):
        """ReconstructionError from reconstruct_graph → RuntimeError raised."""
        from paramem.graph.reconstruct import ReconstructionError

        cfg = _make_config(tmp_path, mode="simulate")
        keys = ["g0"]
        loop = _make_loop_train_to_simulate(tmp_path, keys=keys)

        with (
            patch(
                "paramem.graph.reconstruct.reconstruct_graph",
                side_effect=ReconstructionError("g0 recall failed"),
            ),
            pytest.raises(RuntimeError, match="weight reconstruction failed"),
        ):
            _migrate_tier_train_to_simulate(loop, cfg, "episodic")

        # Rollback: graph.json was not written (unified layout: adapter_dir).
        assert not (cfg.adapter_dir / "episodic" / "graph.json").exists()

    def test_sanity_check_failure_rolls_back(self, tmp_path):
        """When sanity check detects missing key in written graph, graph.json is unlinked."""
        cfg = _make_config(tmp_path, mode="simulate")
        keys = ["g0", "g1"]
        loop = _make_loop_train_to_simulate(tmp_path, keys=keys)

        # Return a graph that only has g0 (missing g1) — sanity check will fail.
        graph = nx.MultiDiGraph()
        eid = graph.add_edge("Subject", "Object", predicate="p")
        graph["Subject"]["Object"][eid][_IK_KEY_ATTR] = "g0"  # g1 is absent
        reconstruction = MagicMock()
        reconstruction.graph = graph
        reconstruction.failures = []

        with (
            patch(
                "paramem.graph.reconstruct.reconstruct_graph",
                return_value=reconstruction,
            ),
            pytest.raises(RuntimeError, match="sanity check failed"),
        ):
            _migrate_tier_train_to_simulate(loop, cfg, "episodic")

        # Rolled back: graph.json removed (unified layout: adapter_dir).
        assert not (cfg.adapter_dir / "episodic" / "graph.json").exists()

    def test_edge_decoration_uses_indexed_key_cache(self, tmp_path):
        """speaker_id and first_seen_cycle from indexed_key_cache appear on graph edges."""
        cfg = _make_config(tmp_path, mode="simulate")
        keys = ["g0"]
        loop = _make_loop_train_to_simulate(tmp_path, keys=keys)
        loop.store._entries_flat_view()["g0"]["speaker_id"] = "spk-alice"
        loop.store._entries_flat_view()["g0"]["first_seen_cycle"] = 7

        reconstruction = self._make_graph_result("episodic", keys)

        with patch(
            "paramem.graph.reconstruct.reconstruct_graph",
            return_value=reconstruction,
        ):
            _migrate_tier_train_to_simulate(loop, cfg, "episodic")

        from paramem.memory.persistence import iter_entries, load_memory_from_disk

        # Unified layout: graph written under adapter_dir, not simulate_dir.
        loaded = load_memory_from_disk(cfg.adapter_dir / "episodic" / "graph.json")
        entries = list(iter_entries(loaded))
        assert len(entries) == 1
        assert entries[0]["speaker_id"] == "spk-alice"
        assert entries[0]["first_seen_cycle"] == 7


# ---------------------------------------------------------------------------
# migrate() orchestrator
# ---------------------------------------------------------------------------


class TestMigrateOrchestrator:
    def test_skips_already_completed_tiers(self, tmp_path):
        cfg = _make_config(tmp_path, mode="simulate")
        # No registry on loop → all tiers raise _TierSkipped → flagged complete
        loop = MagicMock()
        from paramem.memory.store import MemoryStore as _MS

        loop.store = _MS(replay_enabled=False)
        state = MigrationState.for_mode_switch(source_mode="train", target_mode="simulate")
        result = migrate(loop, cfg, state)
        # All three tiers tried, all skipped, all marked complete
        assert set(result.completed_tiers) == set(TIERS)
        # State file removed because all_tiers_done
        assert not state_path(cfg.adapter_dir).exists()

    def test_per_tier_failure_isolated(self, tmp_path):
        cfg = _make_config(tmp_path, mode="train")
        # Set up all three tiers with graph.json files under the unified layout.
        for tier in TIERS:
            _write_simulate_graph(cfg.adapter_dir, tier, [_full_quad(f"{tier}_g1")])

        state = MigrationState.for_mode_switch(source_mode="simulate", target_mode="train")

        loop = MagicMock()
        from paramem.memory.store import MemoryStore as _MS

        loop.store = _MS(replay_enabled=True)
        loop.training_config = MagicMock(num_epochs=2)
        loop._run_recall_sanity_probe.return_value = 1.0
        loop._indexed_dataset.return_value = MagicMock()
        loop._make_training_config.return_value = MagicMock()
        loop.wandb_config = None
        loop.fingerprint_cache = None
        loop._thermal_policy = None
        loop.model = MagicMock()
        loop.model.peft_config = {}

        def _fake_cache_entry(**kwargs):
            return dict(**kwargs)

        loop._cache_entry.side_effect = _fake_cache_entry

        slot_path = cfg.adapter_dir / "episodic" / "20260430-000000"

        call_count = [0]

        def probe_side_effect(tier_name, entries):
            call_count[0] += 1
            if call_count[0] == 1:
                return 0.0  # episodic fails
            return 1.0  # semantic + procedural pass

        loop._run_recall_sanity_probe.side_effect = probe_side_effect

        with (
            patch(
                "paramem.memory.entry.format_entry_training",
                return_value=[{"input_ids": [0]}],
            ),
            patch("paramem.memory.entry.build_registry", return_value={}),
            patch("paramem.models.loader.create_adapter", side_effect=lambda m, c, n: m),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.models.loader.atomic_save_adapter", return_value=slot_path),
            patch("paramem.training.trainer.train_adapter"),
            patch("paramem.adapters.manifest.build_manifest_for", return_value=None),
        ):
            result = migrate(loop, cfg, state)

        # Episodic failed; semantic and procedural completed
        assert "episodic" in result.failed_tiers
        assert "semantic" in result.completed_tiers
        assert "procedural" in result.completed_tiers
        # State file persists because not all_tiers_done
        assert state_path(cfg.adapter_dir).exists()
        # Source for failed tier still on disk (unified layout: adapter_dir).
        assert (cfg.adapter_dir / "episodic" / "graph.json").exists()

    def test_all_complete_clears_state_file(self, tmp_path):
        cfg = _make_config(tmp_path, mode="train")
        for tier in TIERS:
            _write_simulate_graph(cfg.adapter_dir, tier, [_full_quad(f"{tier}_g1")])

        state = MigrationState.for_mode_switch(source_mode="simulate", target_mode="train")

        loop = MagicMock()
        from paramem.memory.store import MemoryStore as _MS

        loop.store = _MS(replay_enabled=True)
        loop.training_config = MagicMock(num_epochs=2)
        loop._run_recall_sanity_probe.return_value = 1.0
        loop._indexed_dataset.return_value = MagicMock()
        loop._make_training_config.return_value = MagicMock()
        loop.wandb_config = None
        loop.fingerprint_cache = None
        loop._thermal_policy = None
        loop.model = MagicMock()
        loop.model.peft_config = {}

        def _fake_cache_entry(**kwargs):
            return dict(**kwargs)

        loop._cache_entry.side_effect = _fake_cache_entry

        slot_path = cfg.adapter_dir / "episodic" / "20260430-000000"

        with (
            patch(
                "paramem.memory.entry.format_entry_training",
                return_value=[{"input_ids": [0]}],
            ),
            patch("paramem.memory.entry.build_registry", return_value={}),
            patch("paramem.models.loader.create_adapter", side_effect=lambda m, c, n: m),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.models.loader.atomic_save_adapter", return_value=slot_path),
            patch("paramem.training.trainer.train_adapter"),
            patch("paramem.adapters.manifest.build_manifest_for", return_value=None),
        ):
            result = migrate(loop, cfg, state)

        assert result.all_tiers_done
        assert not state_path(cfg.adapter_dir).exists()

    def test_resume_skips_completed_tiers(self, tmp_path):
        cfg = _make_config(tmp_path, mode="simulate")
        # Only semantic + procedural have on-disk content; episodic was already done.
        # No registry active keys → all skipped cleanly.
        for tier in ("semantic", "procedural"):
            _write_simulate_graph(cfg.simulate_dir, tier, [])

        loop = MagicMock()
        from paramem.memory.store import MemoryStore as _MS

        loop.store = _MS(replay_enabled=False)

        state = MigrationState.for_mode_switch(source_mode="train", target_mode="simulate")
        state.completed_tiers = ["episodic"]  # already done from a prior partial run

        result = migrate(loop, cfg, state)
        # All tiers complete now (episodic from prior, others from this run)
        assert set(result.completed_tiers) == set(TIERS)


# ---------------------------------------------------------------------------
# Per-tier: simulate -> train
# ---------------------------------------------------------------------------


class TestMigrateTierSimulateToTrain:
    """Simulate→train: reads graph.json, trains adapter, probes, cleans up.

    All tests stub the GPU stack (train_adapter, create_adapter, etc.) and
    verify orchestration + cleanup contracts only.  Simulate mode uses the
    entry format; no legacy format dispatch is tested here (it was removed).
    """

    def _make_loop(self, *, tier_in_peft=False):
        loop = MagicMock()
        loop.episodic_config = MagicMock()
        loop.semantic_config = MagicMock()
        loop.procedural_config = MagicMock()
        loop.training_config = MagicMock(num_epochs=2)
        from paramem.memory.store import MemoryStore as _MS

        loop.store = _MS(replay_enabled=True)
        loop.wandb_config = None
        loop.fingerprint_cache = None
        loop._thermal_policy = None
        loop.model = MagicMock()
        loop.model.peft_config = {"episodic": MagicMock()} if tier_in_peft else {}

        def _fake_cache_entry(
            *,
            key,
            subject,
            predicate,
            object,
            speaker_id,
            first_seen_cycle,
            question=None,
            answer=None,
        ):
            entry = {
                "key": key,
                "subject": subject,
                "predicate": predicate,
                "object": object,
                "speaker_id": speaker_id,
                "first_seen_cycle": first_seen_cycle,
            }
            if question is not None:
                entry["question"] = question
            if answer is not None:
                entry["answer"] = answer
            return entry

        loop._cache_entry.side_effect = _fake_cache_entry
        return loop

    def test_no_source_skipped(self, tmp_path):
        cfg = _make_config(tmp_path, mode="train")
        with pytest.raises(_TierSkipped, match="no graph.json"):
            _migrate_tier_simulate_to_train(self._make_loop(), cfg, "episodic")

    def test_empty_source_skipped(self, tmp_path):
        """Empty graph.json (no edges) → _TierSkipped."""
        cfg = _make_config(tmp_path, mode="train")
        # Unified layout: graph.json lives under adapter_dir.
        _write_simulate_graph(cfg.adapter_dir, "episodic", [])
        with pytest.raises(_TierSkipped, match="empty"):
            _migrate_tier_simulate_to_train(self._make_loop(), cfg, "episodic")

    def test_disabled_tier_skipped(self, tmp_path):
        cfg = _make_config(tmp_path, mode="train")
        # Unified layout: graph.json lives under adapter_dir.
        _write_simulate_graph(cfg.adapter_dir, "procedural", [_full_quad("p1")])
        loop = self._make_loop()
        loop.procedural_config = None  # operator disabled procedural
        with pytest.raises(_TierSkipped, match="not enabled"):
            _migrate_tier_simulate_to_train(loop, cfg, "procedural")

    def test_happy_path_orchestration(self, tmp_path):
        cfg = _make_config(tmp_path, mode="train")
        entries = [_full_quad(f"g{i}") for i in range(2)]
        # Unified layout: graph.json lives under adapter_dir.
        _write_simulate_graph(cfg.adapter_dir, "episodic", entries)
        loop = self._make_loop()
        loop._run_recall_sanity_probe.return_value = 1.0
        loop._indexed_dataset.return_value = MagicMock()
        loop._make_training_config.return_value = MagicMock()

        slot_path = cfg.adapter_dir / "episodic" / "20260430-000000"
        with (
            patch(
                "paramem.memory.entry.format_entry_training",
                return_value=[{"input_ids": [0]}],
            ),
            patch(
                "paramem.memory.entry.build_registry",
                return_value={"g0": 0, "g1": 0},
            ),
            patch("paramem.models.loader.create_adapter", side_effect=lambda m, c, n: m),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.models.loader.atomic_save_adapter", return_value=slot_path),
            patch("paramem.training.trainer.train_adapter") as train_mock,
            patch("paramem.adapters.manifest.build_manifest_for", return_value=MagicMock()),
        ):
            _migrate_tier_simulate_to_train(loop, cfg, "episodic")

        # train_adapter was called with adapter_name=tier
        assert train_mock.call_count == 1
        assert train_mock.call_args.kwargs["adapter_name"] == "episodic"
        # Source graph deleted post-success (simulate_to_train always deletes source graph).
        assert not (cfg.adapter_dir / "episodic" / "graph.json").exists()
        # Per-tier SimHash registry persisted at per-tier path
        assert (cfg.adapter_dir / "episodic" / "simhash_registry.json").exists()

    def test_writes_simhash_registry_with_built_fingerprints(self, tmp_path):
        """The persisted simhash file holds exactly the fingerprints from Step 2."""
        cfg = _make_config(tmp_path, mode="train")
        entries = [_full_quad(f"g{i}") for i in range(2)]
        # Unified layout: graph.json lives under adapter_dir.
        _write_simulate_graph(cfg.adapter_dir, "episodic", entries)
        loop = self._make_loop()
        loop._run_recall_sanity_probe.return_value = 1.0
        loop._indexed_dataset.return_value = MagicMock()
        loop._make_training_config.return_value = MagicMock()

        slot_path = cfg.adapter_dir / "episodic" / "20260430-000000"
        built_registry = {"g0": 123, "g1": 456}
        with (
            patch(
                "paramem.memory.entry.format_entry_training",
                return_value=[{"input_ids": [0]}],
            ),
            patch(
                "paramem.memory.entry.build_registry",
                return_value=built_registry,
            ),
            patch("paramem.models.loader.create_adapter", side_effect=lambda m, c, n: m),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.models.loader.atomic_save_adapter", return_value=slot_path),
            patch("paramem.training.trainer.train_adapter"),
            patch("paramem.adapters.manifest.build_manifest_for", return_value=MagicMock()),
        ):
            _migrate_tier_simulate_to_train(loop, cfg, "episodic")

        from paramem.memory.persistence import load_registry

        on_disk = load_registry(cfg.adapter_dir / "episodic" / "simhash_registry.json")
        assert on_disk == built_registry

    def test_probe_failure_rolls_back(self, tmp_path):
        cfg = _make_config(tmp_path, mode="train")
        entries = [_full_quad(f"g{i}") for i in range(3)]
        # Unified layout: graph.json lives under adapter_dir.
        _write_simulate_graph(cfg.adapter_dir, "episodic", entries)
        loop = self._make_loop()
        loop._run_recall_sanity_probe.return_value = 0.66  # below 1.0 → rollback
        loop._indexed_dataset.return_value = MagicMock()
        loop._make_training_config.return_value = MagicMock()

        with (
            patch(
                "paramem.memory.entry.format_entry_training",
                return_value=[{"input_ids": [0]}],
            ),
            patch("paramem.memory.entry.build_registry", return_value={}),
            patch("paramem.models.loader.create_adapter", side_effect=lambda m, c, n: m),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.models.loader.atomic_save_adapter") as save_mock,
            patch("paramem.training.trainer.train_adapter"),
            patch("paramem.adapters.manifest.build_manifest_for", return_value=None),
        ):
            with pytest.raises(RuntimeError, match=r"recall .* < 1.0"):
                _migrate_tier_simulate_to_train(loop, cfg, "episodic")

        # No save was attempted (probe failed first)
        save_mock.assert_not_called()
        # Source preserved (unified layout: adapter_dir).
        assert (cfg.adapter_dir / "episodic" / "graph.json").exists()
        # No simhash registry written (probe failed before write)
        assert not (cfg.adapter_dir / "episodic" / "simhash_registry.json").exists()

    def test_always_uses_entry_format_helpers(self, tmp_path):
        """simulate→train uses entry helpers; legacy format helpers are not called.

        This test verifies format_entry_training is called (not
        format_indexed_training) and build_registry from entry_memory is used.
        """
        cfg = _make_config(tmp_path, mode="train")
        entries = [_full_quad(f"g{i}") for i in range(2)]
        # Unified layout: graph.json lives under adapter_dir.
        _write_simulate_graph(cfg.adapter_dir, "episodic", entries)
        loop = self._make_loop()
        loop._run_recall_sanity_probe.return_value = 1.0
        loop._indexed_dataset.return_value = MagicMock()
        loop._make_training_config.return_value = MagicMock()

        slot_path = cfg.adapter_dir / "episodic" / "20260430-000000"
        with (
            patch(
                "paramem.memory.entry.format_entry_training",
                return_value=[{"input_ids": [0]}],
            ) as entry_fmt,
            patch(
                "paramem.memory.entry.build_registry",
                return_value={"g0": 0, "g1": 0},
            ) as entry_reg,
            patch("paramem.models.loader.create_adapter", side_effect=lambda m, c, n: m),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.models.loader.atomic_save_adapter", return_value=slot_path),
            patch("paramem.training.trainer.train_adapter"),
            patch("paramem.adapters.manifest.build_manifest_for", return_value=MagicMock()),
        ):
            _migrate_tier_simulate_to_train(loop, cfg, "episodic")

        entry_fmt.assert_called_once()
        entry_reg.assert_called_once()
