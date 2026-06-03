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
        assert s.all_tiers_done(list(TIERS)) is True

    def test_all_tiers_done_with_failure(self):
        s = MigrationState.for_mode_switch(source_mode="simulate", target_mode="train")
        s.completed_tiers = list(TIERS)
        s.failed_tiers = {"episodic": "boom"}
        assert s.all_tiers_done(list(TIERS)) is False

    def test_all_tiers_done_partial(self):
        s = MigrationState.for_mode_switch(source_mode="simulate", target_mode="train")
        s.completed_tiers = ["episodic", "semantic"]
        assert s.all_tiers_done(list(TIERS)) is False


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

    def test_no_active_keys_deletes_stale_weight_slots(self, tmp_path):
        """An empty tier must still have its stale weight slots DELETED — otherwise
        on a base-swap an old-model slot survives Phase A + B and the next boot
        reports a spurious fingerprint_mismatch instead of a clean 0-key tier.
        """
        cfg = _make_config(tmp_path, mode="simulate")
        loop = _make_loop_train_to_simulate(tmp_path, keys=[])
        # Plant a stale weight slot under the (empty) episodic tier.
        stale = cfg.adapter_dir / "episodic" / "20260101-000000"
        stale.mkdir(parents=True)
        (stale / "adapter_model.safetensors").write_bytes(b"")
        (cfg.adapter_dir / "episodic" / "indexed_key_registry.json").write_text("{}")

        with pytest.raises(_TierSkipped, match="deleted 1 stale weight slot"):
            _migrate_tier_train_to_simulate(loop, cfg, "episodic")

        assert not stale.exists(), "stale weight slot must be deleted for an empty tier"
        # Top-level registry is preserved (only the weight slot subdir is removed).
        assert (cfg.adapter_dir / "episodic" / "indexed_key_registry.json").exists()

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
        loop.store.set_bookkeeping("g0", speaker_id="spk-alice", first_seen_cycle=7)

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
    def test_legitimately_empty_store_vacuous_done(self, tmp_path):
        """A store with no registered tiers AND no on-disk content completes vacuously.

        This covers a fresh install or a system with no knowledge: no
        adapter registries, no graph.json files, no interim dirs under
        adapter_dir.  ``all_tiers_done([])`` is vacuously True in this case
        and ``clear_state`` is allowed to fire.  This is NOT the degraded
        case (that requires on-disk content to exist alongside zero tiers).
        """
        cfg = _make_config(tmp_path, mode="simulate")
        # adapter_dir is empty — no on-disk content.  The loop store has
        # replay disabled so tiers_with_registry() returns [].
        loop = MagicMock()
        from paramem.memory.store import MemoryStore as _MS

        loop.store = _MS(replay_enabled=False)
        state = MigrationState.for_mode_switch(source_mode="train", target_mode="simulate")
        result = migrate(loop, cfg, state)
        # No stores registered → no stores iterated → completed_tiers stays empty.
        assert result.completed_tiers == []
        # State file removed because all_tiers_done([]) is vacuously True
        # and the on-disk content check confirmed no content exists.
        assert not state_path(cfg.adapter_dir).exists()

    def test_per_tier_failure_isolated(self, tmp_path):
        cfg = _make_config(tmp_path, mode="train")
        # Set up all three tiers with graph.json files under the unified layout.
        for tier in TIERS:
            _write_simulate_graph(cfg.adapter_dir, tier, [_full_quad(f"{tier}_g1")])

        state = MigrationState.for_mode_switch(source_mode="simulate", target_mode="train")

        loop = MagicMock()
        from paramem.memory.store import MemoryStore as _MS
        from paramem.training.key_registry import KeyRegistry

        loop.store = _MS(replay_enabled=True)
        # Pre-register the three main tiers so tiers_with_registry() returns them.
        # In production this is done by load_registries_from_disk at boot.
        for tier in TIERS:
            loop.store.load_registry(tier, KeyRegistry())
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

        def probe_side_effect(tier_name, entries, max_probe=None):
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
            patch(
                "paramem.training.trainer.train_adapter",
                return_value={"aborted": False},
            ),
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
        from paramem.training.key_registry import KeyRegistry

        loop.store = _MS(replay_enabled=True)
        # Pre-register the three main tiers so tiers_with_registry() returns them.
        for tier in TIERS:
            loop.store.load_registry(tier, KeyRegistry())
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
            patch(
                "paramem.training.trainer.train_adapter",
                return_value={"aborted": False},
            ),
            patch("paramem.adapters.manifest.build_manifest_for", return_value=None),
        ):
            result = migrate(loop, cfg, state)

        registered = list(TIERS)
        assert result.all_tiers_done(registered)
        assert not state_path(cfg.adapter_dir).exists()

    def test_resume_skips_completed_tiers(self, tmp_path):
        cfg = _make_config(tmp_path, mode="simulate")
        # Episodic was already done in a prior partial run.
        # Semantic and procedural still need to run (train→simulate direction,
        # no active registry keys → _TierSkipped → flagged complete).

        loop = MagicMock()
        from paramem.memory.store import MemoryStore as _MS
        from paramem.training.key_registry import KeyRegistry

        loop.store = _MS(replay_enabled=True)
        # Pre-register semantic and procedural; episodic already completed.
        for tier in ("semantic", "procedural"):
            loop.store.load_registry(tier, KeyRegistry())

        state = MigrationState.for_mode_switch(source_mode="train", target_mode="simulate")
        state.completed_tiers = ["episodic"]  # already done from a prior partial run

        result = migrate(loop, cfg, state)
        # All registered tiers (semantic, procedural) complete now; episodic from prior.
        assert "semantic" in result.completed_tiers
        assert "procedural" in result.completed_tiers
        assert "episodic" in result.completed_tiers

    def test_raises_when_empty_tiers_but_disk_has_registry(self, tmp_path):
        """migrate() raises RuntimeError when store has 0 tiers but an
        indexed_key_registry.json exists on disk.

        This is the regression guard for the silent data-loss bug: a failed
        boot-time registry load leaves the in-memory store empty while
        on-disk content is still present.  migrate() must REFUSE to proceed
        (and must NOT call clear_state) so the migration stays pending.
        """
        cfg = _make_config(tmp_path, mode="simulate")
        # Write an indexed_key_registry.json for episodic — on-disk content exists.
        _write_adapter_registry(cfg.adapter_dir, "episodic", ["key1", "key2"])

        loop = MagicMock()
        from paramem.memory.store import MemoryStore as _MS

        # Store has replay disabled → tiers_with_registry() == [].
        # This simulates a failed boot-time load_registries_from_disk.
        loop.store = _MS(replay_enabled=False)
        state = MigrationState.for_mode_switch(source_mode="train", target_mode="simulate")
        # Persist state file before calling migrate to simulate the armed state.
        save_state(cfg.adapter_dir, state)

        with pytest.raises(RuntimeError, match="on-disk content exists"):
            migrate(loop, cfg, state)

        # State file must persist — migration was NOT completed.
        assert state_path(cfg.adapter_dir).exists(), (
            "State file must remain after a refused migration; clear_state "
            "must not fire on the degraded-store path"
        )

    def test_raises_when_empty_tiers_but_disk_has_graph(self, tmp_path):
        """migrate() raises RuntimeError when store has 0 tiers but a
        graph.json exists on disk.

        Variant of the regression guard using a simulate-mode graph.json
        rather than an indexed_key_registry.json as the on-disk signal.
        """
        cfg = _make_config(tmp_path, mode="train")
        # Write a graph.json for semantic — on-disk content exists.
        _write_simulate_graph(cfg.adapter_dir, "semantic", [_full_quad("g1")])

        loop = MagicMock()
        from paramem.memory.store import MemoryStore as _MS

        loop.store = _MS(replay_enabled=False)
        state = MigrationState.for_mode_switch(source_mode="simulate", target_mode="train")
        save_state(cfg.adapter_dir, state)

        with pytest.raises(RuntimeError, match="on-disk content exists"):
            migrate(loop, cfg, state)

        assert state_path(cfg.adapter_dir).exists()

    def test_raises_when_empty_tiers_but_disk_has_interim_dir(self, tmp_path):
        """migrate() raises RuntimeError when store has 0 tiers but an
        interim directory exists under adapter_dir.

        Variant of the regression guard using iter_interim_dirs to detect
        on-disk content.
        """
        cfg = _make_config(tmp_path, mode="simulate")
        # Create an interim directory under the episodic tier.
        interim_dir = cfg.adapter_dir / "episodic" / "interim_20260101T0000"
        interim_dir.mkdir(parents=True, exist_ok=True)
        (interim_dir / "indexed_key_registry.json").write_text("{}")

        loop = MagicMock()
        from paramem.memory.store import MemoryStore as _MS

        loop.store = _MS(replay_enabled=False)
        state = MigrationState.for_mode_switch(source_mode="train", target_mode="simulate")
        save_state(cfg.adapter_dir, state)

        with pytest.raises(RuntimeError, match="on-disk content exists"):
            migrate(loop, cfg, state)

        assert state_path(cfg.adapter_dir).exists()


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
            patch(
                "paramem.training.trainer.train_adapter",
                return_value={"aborted": False},
            ) as train_mock,
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

    def test_binds_slot_to_registry(self, tmp_path):
        """Regression: the trained slot's manifest carries a NON-empty
        registry_sha256 (== sha256 of the tier registry bytes) AND the tier
        registry is flushed to <tier>/indexed_key_registry.json.

        Without this binding meta.registry_sha256 is empty, find_live_slot can
        never match it, the adapter silently fails to mount on the next
        boot/reload, and recall returns 0 keys (boot_degraded).
        """
        import hashlib

        cfg = _make_config(tmp_path, mode="train")
        entries = [_full_quad(f"g{i}") for i in range(2)]
        _write_simulate_graph(cfg.adapter_dir, "episodic", entries)
        loop = self._make_loop()
        loop._run_recall_sanity_probe.return_value = 1.0
        loop._indexed_dataset.return_value = MagicMock()
        loop._make_training_config.return_value = MagicMock()

        captured: dict = {}

        def _capture_manifest(*args, **kwargs):
            captured["registry_sha256_override"] = kwargs.get("registry_sha256_override")
            return MagicMock()

        slot_path = cfg.adapter_dir / "episodic" / "20260430-000000"
        with (
            patch(
                "paramem.memory.entry.format_entry_training",
                return_value=[{"input_ids": [0]}],
            ),
            patch("paramem.memory.entry.build_registry", return_value={"g0": 0, "g1": 0}),
            patch("paramem.models.loader.create_adapter", side_effect=lambda m, c, n: m),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.models.loader.atomic_save_adapter", return_value=slot_path),
            patch(
                "paramem.training.trainer.train_adapter",
                return_value={"aborted": False},
            ),
            patch(
                "paramem.adapters.manifest.build_manifest_for",
                side_effect=_capture_manifest,
            ),
        ):
            _migrate_tier_simulate_to_train(loop, cfg, "episodic")

        # Manifest is bound to the tier registry (non-empty hash matching the bytes).
        expected_sha = hashlib.sha256(loop.store.registry("episodic").save_bytes()).hexdigest()
        assert captured["registry_sha256_override"], "registry_sha256_override must be non-empty"
        assert captured["registry_sha256_override"] == expected_sha

        # Tier registry flushed so find_live_slot can match it on the next boot.
        assert (cfg.adapter_dir / "episodic" / "indexed_key_registry.json").exists(), (
            "tier registry must be written for slot binding"
        )

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
            patch(
                "paramem.training.trainer.train_adapter",
                return_value={"aborted": False},
            ),
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
            patch(
                "paramem.training.trainer.train_adapter",
                return_value={"aborted": False},
            ),
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
            patch(
                "paramem.training.trainer.train_adapter",
                return_value={"aborted": False},
            ),
            patch("paramem.adapters.manifest.build_manifest_for", return_value=MagicMock()),
        ):
            _migrate_tier_simulate_to_train(loop, cfg, "episodic")

        entry_fmt.assert_called_once()
        entry_reg.assert_called_once()

    def test_recall_probe_is_uncapped(self, tmp_path):
        """_migrate_tier_simulate_to_train passes max_probe=len(entries) to the probe.

        The Phase B gate must probe ALL entries (100 % requirement), not just
        the default max_probe=100.  Verify that the call uses an explicit
        max_probe keyword equal to the number of graph entries so the probe
        cannot silently pass on a small subset of a large adapter.
        """
        cfg = _make_config(tmp_path, mode="train")
        # Use 5 entries; max_probe=5 expected, not the default 100.
        entries = [_full_quad(f"g{i}") for i in range(5)]
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
                return_value={f"g{i}": i for i in range(5)},
            ),
            patch("paramem.models.loader.create_adapter", side_effect=lambda m, c, n: m),
            patch("paramem.models.loader.switch_adapter"),
            patch("paramem.models.loader.atomic_save_adapter", return_value=slot_path),
            patch(
                "paramem.training.trainer.train_adapter",
                return_value={"aborted": False},
            ),
            patch("paramem.adapters.manifest.build_manifest_for", return_value=MagicMock()),
        ):
            _migrate_tier_simulate_to_train(loop, cfg, "episodic")

        # The probe must have been called with max_probe equal to the entry count.
        assert loop._run_recall_sanity_probe.call_count == 1
        call_kwargs = loop._run_recall_sanity_probe.call_args
        # Accept either positional or keyword for max_probe.
        if call_kwargs.kwargs:
            actual_max_probe = call_kwargs.kwargs.get("max_probe")
        else:
            # (tier, entries, max_probe) positional
            actual_max_probe = call_kwargs.args[2] if len(call_kwargs.args) > 2 else None
        assert actual_max_probe == len(entries), (
            f"Expected max_probe={len(entries)} (uncapped); got {actual_max_probe!r}. "
            "Phase B gate must probe ALL entries, not the default 100."
        )


# ---------------------------------------------------------------------------
# New tests: interim store migration
# ---------------------------------------------------------------------------


INTERIM_NAME = "episodic_interim_20260101T0000"


class TestMigrateEnumeration:
    """migrate() dispatches all registered stores including interim ones."""

    def test_dispatches_all_four_stores(self, tmp_path):
        """Four registered stores (3 main + 1 interim) are all dispatched."""
        from paramem.memory.store import MemoryStore as _MS
        from paramem.training.key_registry import KeyRegistry

        cfg = _make_config(tmp_path, mode="simulate")
        loop = MagicMock()
        store = _MS(replay_enabled=True)
        registered = list(TIERS) + [INTERIM_NAME]
        for name in registered:
            store.load_registry(name, KeyRegistry())
        loop.store = store

        state = MigrationState.for_mode_switch(source_mode="train", target_mode="simulate")

        dispatched: list[str] = []

        def _fake_t2s(l, c, name):  # noqa: E741
            dispatched.append(name)

        with patch(
            "paramem.server.active_store_migration._migrate_tier_train_to_simulate",
            side_effect=_fake_t2s,
        ):
            result = migrate(loop, cfg, state)

        assert set(dispatched) == set(registered)
        assert set(result.completed_tiers) == set(registered)


class TestAllTiersDoneCoupling:
    """all_tiers_done(registered_tiers) coupling — regression guard."""

    def test_false_while_interim_pending(self, tmp_path):
        """all_tiers_done returns False when the interim is not yet complete."""
        registered = list(TIERS) + [INTERIM_NAME]
        state = MigrationState.for_mode_switch(source_mode="train", target_mode="simulate")
        state.completed_tiers = list(TIERS)  # interim NOT yet done
        assert state.all_tiers_done(registered) is False

    def test_true_after_all_four_complete(self, tmp_path):
        """all_tiers_done returns True only after all four stores complete."""
        registered = list(TIERS) + [INTERIM_NAME]
        state = MigrationState.for_mode_switch(source_mode="train", target_mode="simulate")
        state.completed_tiers = registered[:]
        assert state.all_tiers_done(registered) is True

    def test_clear_state_not_called_while_interim_pending(self, tmp_path):
        """migrate() does NOT clear the state file while the interim is pending."""
        from paramem.memory.store import MemoryStore as _MS
        from paramem.training.key_registry import KeyRegistry

        cfg = _make_config(tmp_path, mode="simulate")
        loop = MagicMock()
        store = _MS(replay_enabled=True)
        registered = list(TIERS) + [INTERIM_NAME]
        for name in registered:
            store.load_registry(name, KeyRegistry())
        loop.store = store

        state = MigrationState.for_mode_switch(source_mode="train", target_mode="simulate")

        call_order: list[str] = []

        def _fake_t2s(l, c, name):  # noqa: E741
            call_order.append(name)
            if name == INTERIM_NAME:
                raise RuntimeError("interim blew up")

        with patch(
            "paramem.server.active_store_migration._migrate_tier_train_to_simulate",
            side_effect=_fake_t2s,
        ):
            result = migrate(loop, cfg, state)

        # All three main tiers succeeded; interim failed → state file MUST persist.
        assert INTERIM_NAME in result.failed_tiers
        assert state_path(cfg.adapter_dir).exists(), (
            "State file must persist when interim store failed; "
            "clear_state must NOT be called with an incomplete registered set"
        )


class TestSlotPathResolverInterim:
    """Resolver wiring for interim stores in both migration directions."""

    def test_train_to_simulate_resolves_interim_slot_root(self, tmp_path):
        """train_to_simulate writes graph.json under episodic/interim_<stamp>/."""
        from paramem.memory.interim_adapter import adapter_slot_root_for_name

        cfg = _make_config(tmp_path, mode="simulate")
        adapter_dir = cfg.adapter_dir

        # Expected slot root under the hierarchy.
        expected_slot_root = adapter_slot_root_for_name(adapter_dir, INTERIM_NAME)
        assert str(expected_slot_root) == str(adapter_dir / "episodic" / "interim_20260101T0000")

        # Create a weight slot under the resolved root so migration has something to delete.
        slot_dir = expected_slot_root / "20260430-180000"
        slot_dir.mkdir(parents=True, exist_ok=True)
        (slot_dir / "adapter_config.json").write_text("{}")

        # Write a graph.json at the slot root (simulating a reconstruct).
        # We need active keys in the store so _TierSkipped is not raised.
        loop = _make_loop_train_to_simulate(tmp_path, keys=["ik1"])
        # Override the store's tier to the interim name.
        from paramem.training.key_registry import KeyRegistry

        loop.store.load_registry(INTERIM_NAME, KeyRegistry())
        loop.store.put(INTERIM_NAME, "ik1", {"subject": "S", "predicate": "p", "object": "O"})

        reconstruction_graph = nx.MultiDiGraph()
        eid = reconstruction_graph.add_edge("S", "O", predicate="p")
        reconstruction_graph["S"]["O"][eid][_IK_KEY_ATTR] = "ik1"
        result_mock = MagicMock()
        result_mock.graph = reconstruction_graph
        result_mock.failures = []

        with patch(
            "paramem.graph.reconstruct.reconstruct_graph",
            return_value=result_mock,
        ):
            _migrate_tier_train_to_simulate(loop, cfg, INTERIM_NAME)

        # graph.json must exist under the interim slot root.
        assert (expected_slot_root / "graph.json").exists()
        # Weight slot directory must have been deleted.
        assert not slot_dir.exists()

    def test_simulate_to_train_resolves_interim_slot_root(self, tmp_path):
        """simulate_to_train reads/saves under episodic/interim_<stamp>/."""
        from paramem.memory.interim_adapter import adapter_slot_root_for_name

        cfg = _make_config(tmp_path, mode="train")
        adapter_dir = cfg.adapter_dir

        expected_slot_root = adapter_slot_root_for_name(adapter_dir, INTERIM_NAME)

        # Write the source graph.json at the interim slot root.
        entries = [_full_quad("ik1")]
        _write_simulate_graph_at(expected_slot_root, entries)

        loop = TestMigrateTierSimulateToTrain()._make_loop()
        loop._run_recall_sanity_probe.return_value = 1.0
        loop._indexed_dataset.return_value = MagicMock()
        loop._make_training_config.return_value = MagicMock()

        slot_path = expected_slot_root / "20260430-000000"
        with (
            patch(
                "paramem.memory.entry.format_entry_training",
                return_value=[{"input_ids": [0]}],
            ),
            patch("paramem.memory.entry.build_registry", return_value={"ik1": 99}),
            patch("paramem.models.loader.create_adapter", side_effect=lambda m, c, n: m),
            patch("paramem.models.loader.switch_adapter"),
            patch(
                "paramem.models.loader.atomic_save_adapter",
                return_value=slot_path,
            ),
            patch(
                "paramem.training.trainer.train_adapter",
                return_value={"aborted": False},
            ),
            patch("paramem.adapters.manifest.build_manifest_for", return_value=None),
        ):
            _migrate_tier_simulate_to_train(loop, cfg, INTERIM_NAME)

        # Source graph.json at slot root must have been deleted after success.
        assert not (expected_slot_root / "graph.json").exists()
        # SimHash registry written at slot root.
        assert (expected_slot_root / "simhash_registry.json").exists()


class TestTierAdapterConfigInterim:
    """_tier_adapter_config maps interim names to the episodic config."""

    def test_interim_name_resolves_to_episodic_config(self):
        from paramem.server.active_store_migration import _tier_adapter_config

        loop = MagicMock()
        expected = MagicMock(name="episodic_cfg")
        loop.episodic_config = expected
        loop.semantic_config = MagicMock(name="semantic_cfg")
        loop.procedural_config = MagicMock(name="procedural_cfg")

        result = _tier_adapter_config(loop, INTERIM_NAME)
        assert result is expected, (
            f"Interim name {INTERIM_NAME!r} must resolve to episodic_config; got {result!r}"
        )

    def test_main_tier_resolves_own_config(self):
        from paramem.server.active_store_migration import _tier_adapter_config

        loop = MagicMock()
        loop.episodic_config = MagicMock(name="ep")
        loop.semantic_config = MagicMock(name="sem")
        loop.procedural_config = MagicMock(name="proc")

        assert _tier_adapter_config(loop, "episodic") is loop.episodic_config
        assert _tier_adapter_config(loop, "semantic") is loop.semantic_config
        assert _tier_adapter_config(loop, "procedural") is loop.procedural_config

    def test_interim_with_disabled_episodic_raises_skipped(self):
        """When episodic_config is None, interim adapter raises _TierSkipped."""
        from paramem.server.active_store_migration import _tier_adapter_config

        loop = MagicMock()
        loop.episodic_config = None
        with pytest.raises(_TierSkipped, match="not enabled"):
            _tier_adapter_config(loop, INTERIM_NAME)


class TestNegativeCouplingGuard:
    """Structural guard: active_store_migration must not import consolidation functions."""

    def _module_source(self) -> str:
        import importlib.util

        spec = importlib.util.find_spec("paramem.server.active_store_migration")
        assert spec is not None and spec.origin is not None
        return Path(spec.origin).read_text(encoding="utf-8")

    def test_no_partition_relations_import(self):
        assert "partition_relations" not in self._module_source()

    def test_no_run_graph_enrichment_import(self):
        assert "_run_graph_enrichment" not in self._module_source()

    def test_no_consolidate_interim_adapters_import(self):
        assert "consolidate_interim_adapters" not in self._module_source()

    def test_no_merger_graph_access(self):
        assert ".merger.graph" not in self._module_source()


class TestTerminologyGuard:
    """Terminology guard: 'fallback' and 'drift' must not appear in the module source."""

    def _module_source(self) -> str:
        import importlib.util

        spec = importlib.util.find_spec("paramem.server.active_store_migration")
        assert spec is not None and spec.origin is not None
        return Path(spec.origin).read_text(encoding="utf-8")

    def test_no_fallback_in_source(self):
        src = self._module_source()
        assert "fallback" not in src.lower(), (
            "The word 'fallback' must not appear in active_store_migration.py "
            "(the source store is authoritative, not a fallback)"
        )

    def test_no_drift_in_source(self):
        src = self._module_source()
        assert "drift" not in src.lower(), (
            "The word 'drift' must not appear in active_store_migration.py"
        )


# ---------------------------------------------------------------------------
# Dispatch-guard: store_load_degraded blocks migration dispatch
# ---------------------------------------------------------------------------


class TestStoreLoadDegradedDispatchGuard:
    """_maybe_trigger_scheduled_consolidation must not dispatch migration when
    store_load_degraded is True.

    This is the second half of the fix for the silent data-loss bug: even if
    pending_rehydration is set, a degraded store (failed boot registry load)
    must prevent the migration from running.  The migration would see 0
    registered tiers, all_tiers_done([]) would fire, clear_state would remove
    the state file, and _finalize_migration would flip effective_mode —
    completing a no-op migration.
    """

    def test_degraded_store_skips_migration_dispatch(self):
        """With store_load_degraded=True and pending_rehydration=True,
        _maybe_trigger_scheduled_consolidation returns 'migration_skipped_degraded'
        and does NOT call _run_active_store_migration_sync.
        """
        from paramem.server import app as app_module

        run_migration_calls = []

        with (
            patch.dict(
                app_module._state,
                {
                    "consolidating": False,
                    "mode": "local",
                    "background_trainer": None,
                    "config": MagicMock(
                        consolidation=MagicMock(consolidation_period_string=""),
                        adapter_dir=Path("/tmp/fake"),
                    ),
                    "session_buffer": MagicMock(),
                    "pending_rehydration": True,
                    "store_load_degraded": True,
                },
                clear=False,
            ),
            patch(
                "paramem.server.app._retro_claim_orphan_sessions",
                return_value=0,
            ),
            patch(
                "paramem.server.app._run_active_store_migration_sync",
                side_effect=lambda: run_migration_calls.append(1),
            ),
        ):
            result = app_module._maybe_trigger_scheduled_consolidation()

        assert result == "migration_skipped_degraded", (
            f"Expected 'migration_skipped_degraded' but got {result!r}; "
            "migration must be blocked when the store failed to load at boot"
        )
        assert run_migration_calls == [], (
            "_run_active_store_migration_sync must NOT be called when store_load_degraded=True"
        )

    def test_healthy_store_dispatches_migration(self):
        """With store_load_degraded=False and pending_rehydration=True,
        _maybe_trigger_scheduled_consolidation proceeds to dispatch the migration.

        Verifies that the degraded guard does not block healthy stores.
        """
        from paramem.server import app as app_module

        mock_future = MagicMock()
        mock_future.add_done_callback = MagicMock()
        mock_loop = MagicMock()
        mock_loop.run_in_executor.return_value = mock_future

        with (
            patch.dict(
                app_module._state,
                {
                    "consolidating": False,
                    "mode": "local",
                    "background_trainer": None,
                    "config": MagicMock(
                        consolidation=MagicMock(consolidation_period_string=""),
                        adapter_dir=Path("/tmp/fake"),
                    ),
                    "session_buffer": MagicMock(),
                    "pending_rehydration": True,
                    "store_load_degraded": False,
                },
                clear=False,
            ),
            patch(
                "paramem.server.app._retro_claim_orphan_sessions",
                return_value=0,
            ),
            patch("asyncio.get_running_loop", return_value=mock_loop),
        ):
            result = app_module._maybe_trigger_scheduled_consolidation()

        assert result == "started_migration", (
            f"Expected 'started_migration' but got {result!r}; "
            "migration must proceed when store_load_degraded=False"
        )
        mock_loop.run_in_executor.assert_called_once()


# ---------------------------------------------------------------------------
# Helpers used by new tests (not already in the module-level helpers above)
# ---------------------------------------------------------------------------


def _write_simulate_graph_at(slot_root: Path, entries: list[dict]) -> Path:
    """Write a graph.json directly at *slot_root* (for interim path tests)."""
    slot_root.mkdir(parents=True, exist_ok=True)
    graph_path = slot_root / "graph.json"
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
