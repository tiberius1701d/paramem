"""Tests for the active-store migration helper.

Covers:
- State-file model: round-trip, all_tiers_done, mode-switch validation.
- Detection: state file precedence, fresh-detect from yaml mode + on-disk
  store contents.
- train_to_simulate per-tier: happy-path file copy + probe + cleanup;
  rollback on probe failure.
- migrate(): per-tier success advances state; failure isolates to one
  tier; all-tiers-done removes the state file.

Does not yet cover simulate_to_train (the helper raises NotImplementedError;
its spec is documented in the module docstring).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from paramem.server.active_store_migration import (
    TIERS,
    MigrationState,
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


def _write_simulate_kp(simulate_dir: Path, tier: str, keyed_pairs: list[dict]) -> Path:
    """Write a kp file in the simulate-store layout (plaintext for tests)."""
    tier_dir = simulate_dir / tier
    tier_dir.mkdir(parents=True, exist_ok=True)
    p = tier_dir / "keyed_pairs.json"
    p.write_bytes(json.dumps(keyed_pairs).encode("utf-8"))
    return p


def _write_adapter_kp(adapter_dir: Path, tier: str, keyed_pairs: list[dict]) -> Path:
    """Write a kp file at the canonical TIER-LEVEL path:
    ``<adapter_dir>/<tier>/keyed_pairs.json``.

    Matches the layout produced by ``_save_adapters._write_kp`` /
    ``_save_keyed_pairs_for_router``; verified live (the tier-level kp is the
    canonical artifact, slot subdirs hold weights + manifest only).
    """
    tier_dir = adapter_dir / tier
    tier_dir.mkdir(parents=True, exist_ok=True)
    p = tier_dir / "keyed_pairs.json"
    p.write_bytes(json.dumps(keyed_pairs).encode("utf-8"))
    return p


def _write_adapter_slot_dir(adapter_dir: Path, tier: str, slot_ts: str) -> Path:
    """Create an empty slot subdir to simulate trained-but-not-yet-collected state."""
    slot_dir = adapter_dir / tier / slot_ts
    slot_dir.mkdir(parents=True, exist_ok=True)
    return slot_dir


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
        # Even with no on-disk source kp, the state file dictates resume.
        result = detect_mode_switch(cfg)
        assert result is not None
        assert result.completed_tiers == ["episodic"]

    def test_simulate_to_train_detected(self, tmp_path):
        cfg = _make_config(tmp_path, mode="train")
        _write_simulate_kp(cfg.simulate_dir, "episodic", [{"key": "g1"}])
        result = detect_mode_switch(cfg)
        assert result is not None
        assert result.direction == "simulate_to_train"
        assert result.source_mode == "simulate"

    def test_train_to_simulate_detected(self, tmp_path):
        cfg = _make_config(tmp_path, mode="simulate")
        _write_adapter_kp(cfg.adapter_dir, "episodic", [{"key": "g1"}])
        result = detect_mode_switch(cfg)
        assert result is not None
        assert result.direction == "train_to_simulate"
        assert result.source_mode == "train"

    def test_active_state_consistent_no_migration(self, tmp_path):
        # Both stores have content matching the operator's mode → no switch
        cfg = _make_config(tmp_path, mode="train")
        _write_adapter_kp(cfg.adapter_dir, "episodic", [{"key": "g1"}])
        # Stale simulate-store from prior mode IS present, but adapter is too —
        # this is a "stale inactive store" case, not a mode switch.
        _write_simulate_kp(cfg.simulate_dir, "episodic", [{"key": "g1"}])
        assert detect_mode_switch(cfg) is None

    def test_unsupported_mode_returns_none(self, tmp_path):
        cfg = _make_config(tmp_path, mode="cloud_only")
        assert detect_mode_switch(cfg) is None


# ---------------------------------------------------------------------------
# Per-tier: train -> simulate
# ---------------------------------------------------------------------------


class TestMigrateTierTrainToSimulate:
    def _setup_tier(self, tmp_path, tier="episodic", keyed_pairs=None):
        cfg = _make_config(tmp_path, mode="simulate")
        if keyed_pairs is None:
            keyed_pairs = [
                {"key": f"g{i}", "question": f"q{i}", "answer": f"a{i}"} for i in range(3)
            ]
        _write_adapter_kp(cfg.adapter_dir, tier, keyed_pairs)
        return cfg, keyed_pairs

    def test_no_kp_skipped(self, tmp_path):
        cfg = _make_config(tmp_path, mode="simulate")
        with pytest.raises(_TierSkipped, match="no keyed_pairs.json"):
            _migrate_tier_train_to_simulate(MagicMock(), cfg, "episodic")

    def test_empty_kp_skipped(self, tmp_path):
        cfg, _ = self._setup_tier(tmp_path, keyed_pairs=[])
        with pytest.raises(_TierSkipped, match="empty"):
            _migrate_tier_train_to_simulate(MagicMock(), cfg, "episodic")

    def test_happy_path_copies_writes_probes_cleans_up(self, tmp_path):
        cfg, keyed_pairs = self._setup_tier(tmp_path)
        # Slot subdir present alongside the tier-level kp — should be cleaned up
        # by the post-pass shutil.rmtree of the entire tier dir.
        _write_adapter_slot_dir(cfg.adapter_dir, "episodic", "20260429-180000")
        with patch("paramem.training.indexed_memory.probe_keys_from_disk") as probe:
            probe.return_value = {kp["key"]: {"raw_output": "ok"} for kp in keyed_pairs}
            _migrate_tier_train_to_simulate(MagicMock(), cfg, "episodic")
        # Simulate-store now has the kp file
        target = cfg.simulate_dir / "episodic" / "keyed_pairs.json"
        assert target.exists()
        loaded = json.loads(target.read_bytes())
        assert loaded == keyed_pairs
        # Adapter dir for the tier (and its slot subdirs) deleted
        assert not (cfg.adapter_dir / "episodic").exists()

    def test_probe_failure_rolls_back(self, tmp_path):
        cfg, keyed_pairs = self._setup_tier(tmp_path)
        _write_adapter_slot_dir(cfg.adapter_dir, "episodic", "20260429-180000")
        with patch("paramem.training.indexed_memory.probe_keys_from_disk") as probe:
            # Half the keys fail to probe
            probe.return_value = {
                kp["key"]: ({"raw_output": "ok"} if i == 0 else None)
                for i, kp in enumerate(keyed_pairs)
            }
            with pytest.raises(RuntimeError, match="recall .* < 1.0"):
                _migrate_tier_train_to_simulate(MagicMock(), cfg, "episodic")
        # Rollback: simulate-store write removed
        assert not (cfg.simulate_dir / "episodic" / "keyed_pairs.json").exists()
        # Adapter tier dir preserved (source authoritative)
        assert (cfg.adapter_dir / "episodic" / "keyed_pairs.json").exists()
        assert (cfg.adapter_dir / "episodic" / "20260429-180000").exists()


# ---------------------------------------------------------------------------
# migrate() orchestrator
# ---------------------------------------------------------------------------


class TestMigrateOrchestrator:
    def test_skips_already_completed_tiers(self, tmp_path):
        cfg = _make_config(tmp_path, mode="simulate")
        # No adapter slots anywhere → all tiers raise _TierSkipped → flagged complete
        state = MigrationState.for_mode_switch(source_mode="train", target_mode="simulate")
        result = migrate(MagicMock(), cfg, state)
        # All three tiers tried, all skipped, all marked complete
        assert set(result.completed_tiers) == set(TIERS)
        # State file removed because all_tiers_done
        assert not state_path(cfg.adapter_dir).exists()

    def test_per_tier_failure_isolated(self, tmp_path):
        cfg = _make_config(tmp_path, mode="simulate")
        # Set up all three tiers with kp files at the canonical tier-level path
        for tier in TIERS:
            _write_adapter_kp(
                cfg.adapter_dir,
                tier,
                [{"key": f"{tier}_g1", "question": "q", "answer": "a"}],
            )
        state = MigrationState.for_mode_switch(source_mode="train", target_mode="simulate")

        # Probe stub: episodic recall is 0.0; semantic and procedural are 1.0
        def fake_probe(simulate_dir, keys_by_adapter):
            tier_in_call = next(iter(keys_by_adapter.keys()))
            keys = keys_by_adapter[tier_in_call]
            if tier_in_call == "episodic":
                return {k: None for k in keys}
            return {k: {"raw_output": "ok"} for k in keys}

        with patch("paramem.training.indexed_memory.probe_keys_from_disk", side_effect=fake_probe):
            result = migrate(MagicMock(), cfg, state)
        # Episodic failed; semantic and procedural completed
        assert "episodic" in result.failed_tiers
        assert "semantic" in result.completed_tiers
        assert "procedural" in result.completed_tiers
        # State file persists because not all_tiers_done
        assert state_path(cfg.adapter_dir).exists()
        # Source for failed tier still on disk
        assert (cfg.adapter_dir / "episodic" / "keyed_pairs.json").exists()
        # Sources for succeeded tiers cleaned up
        assert not (cfg.adapter_dir / "semantic").exists()
        assert not (cfg.adapter_dir / "procedural").exists()

    def test_all_complete_clears_state_file(self, tmp_path):
        cfg = _make_config(tmp_path, mode="simulate")
        for tier in TIERS:
            _write_adapter_kp(
                cfg.adapter_dir,
                tier,
                [{"key": f"{tier}_g1", "question": "q", "answer": "a"}],
            )
        state = MigrationState.for_mode_switch(source_mode="train", target_mode="simulate")
        with patch("paramem.training.indexed_memory.probe_keys_from_disk") as probe:
            probe.side_effect = lambda d, kba: {
                k: {"raw_output": "ok"} for keys in kba.values() for k in keys
            }
            result = migrate(MagicMock(), cfg, state)
        assert result.all_tiers_done
        assert not state_path(cfg.adapter_dir).exists()

    def test_resume_skips_completed_tiers(self, tmp_path):
        cfg = _make_config(tmp_path, mode="simulate")
        # Only semantic + procedural have on-disk content; episodic was already done
        for tier in ("semantic", "procedural"):
            _write_adapter_kp(
                cfg.adapter_dir,
                tier,
                [{"key": f"{tier}_g1", "question": "q", "answer": "a"}],
            )
        state = MigrationState.for_mode_switch(source_mode="train", target_mode="simulate")
        state.completed_tiers = ["episodic"]  # already done from a prior partial run
        with patch("paramem.training.indexed_memory.probe_keys_from_disk") as probe:
            probe.side_effect = lambda d, kba: {
                k: {"raw_output": "ok"} for keys in kba.values() for k in keys
            }
            result = migrate(MagicMock(), cfg, state)
        # All tiers complete now (episodic from prior, others from this run)
        assert set(result.completed_tiers) == set(TIERS)


# ---------------------------------------------------------------------------
# Per-tier: simulate -> train
# ---------------------------------------------------------------------------


class TestMigrateTierSimulateToTrain:
    """Mocks the entire training stack — verifies orchestration and rollback."""

    def _make_loop(self, *, tier_in_peft=False):
        loop = MagicMock()
        # Per-tier configs (simulate the dataclass attributes the helper reads)
        loop.episodic_config = MagicMock()
        loop.semantic_config = MagicMock()
        loop.procedural_config = MagicMock()
        loop.training_config = MagicMock(num_epochs=2)
        loop.indexed_key_qa = {}
        loop.indexed_key_registry = MagicMock()
        loop.indexed_key_registry.__contains__ = MagicMock(return_value=False)
        loop.wandb_config = None
        loop.fingerprint_cache = None
        loop._shutdown_callbacks = []
        # Model with peft_config — start without the tier so create_adapter is exercised
        loop.model = MagicMock()
        loop.model.peft_config = {"episodic": MagicMock()} if tier_in_peft else {}
        return loop

    def test_no_source_skipped(self, tmp_path):
        cfg = _make_config(tmp_path, mode="train")
        with pytest.raises(_TierSkipped, match="no keyed_pairs.json"):
            _migrate_tier_simulate_to_train(self._make_loop(), cfg, "episodic")

    def test_empty_source_skipped(self, tmp_path):
        cfg = _make_config(tmp_path, mode="train")
        _write_simulate_kp(cfg.simulate_dir, "episodic", [])
        with pytest.raises(_TierSkipped, match="empty"):
            _migrate_tier_simulate_to_train(self._make_loop(), cfg, "episodic")

    def test_disabled_tier_skipped(self, tmp_path):
        cfg = _make_config(tmp_path, mode="train")
        _write_simulate_kp(
            cfg.simulate_dir, "procedural", [{"key": "p1", "question": "q", "answer": "a"}]
        )
        loop = self._make_loop()
        loop.procedural_config = None  # operator disabled procedural
        with pytest.raises(_TierSkipped, match="not enabled"):
            _migrate_tier_simulate_to_train(loop, cfg, "procedural")

    def test_happy_path_orchestration(self, tmp_path):
        cfg = _make_config(tmp_path, mode="train")
        keyed_pairs = [{"key": f"g{i}", "question": "q", "answer": "a"} for i in range(2)]
        _write_simulate_kp(cfg.simulate_dir, "episodic", keyed_pairs)
        loop = self._make_loop()
        # 1.0 recall on probe
        loop._run_recall_sanity_probe.return_value = 1.0
        loop._indexed_dataset.return_value = MagicMock()
        loop._make_training_config.return_value = MagicMock()

        slot_path = cfg.adapter_dir / "episodic" / "20260430-000000"
        with (
            patch(
                "paramem.training.indexed_memory.format_indexed_training",
                return_value=[{"input_ids": [0]}],
            ),
            patch(
                "paramem.training.indexed_memory.build_registry", return_value={"g0": 0, "g1": 0}
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
        # Source kp deleted post-success
        assert not (cfg.simulate_dir / "episodic" / "keyed_pairs.json").exists()
        # Tier-level kp written at canonical path
        assert (cfg.adapter_dir / "episodic" / "keyed_pairs.json").exists()

    def test_probe_failure_rolls_back(self, tmp_path):
        cfg = _make_config(tmp_path, mode="train")
        keyed_pairs = [{"key": f"g{i}", "question": "q", "answer": "a"} for i in range(3)]
        _write_simulate_kp(cfg.simulate_dir, "episodic", keyed_pairs)
        loop = self._make_loop()
        loop._run_recall_sanity_probe.return_value = 0.66  # below 1.0 → rollback
        loop._indexed_dataset.return_value = MagicMock()
        loop._make_training_config.return_value = MagicMock()

        with (
            patch(
                "paramem.training.indexed_memory.format_indexed_training",
                return_value=[{"input_ids": [0]}],
            ),
            patch("paramem.training.indexed_memory.build_registry", return_value={}),
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
        # Source preserved
        assert (cfg.simulate_dir / "episodic" / "keyed_pairs.json").exists()
        # No tier-level kp written under adapters
        assert not (cfg.adapter_dir / "episodic" / "keyed_pairs.json").exists()
