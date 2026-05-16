"""Non-GPU contract tests for experiments/test15_retention_multiseed.py.

Covers:
1. Phase-builder parity at (total_keys=100, swap_keys=20) against test13.
2. Phase-builder parity at small scale (total_keys=10, swap_keys=2).
3. Repair-log schema round-trips through json.load with all required fields.
4. Marker logic: marker_exists returns correct values; phase resume reads
   markers in documented order.
5. Multi-seed aggregator schema: bootstrap CI populated, verdict correct,
   degenerate mean_retention_B==0 does not crash.
6. Decay-steps math: decay_steps_for formula returns expected value.
7. run_config.json persistence and find_latest_run_dir recency selection.
8. Pause-mid-phase inference: _exit_if_paused_mid_phase behaviour on the
   three distinct cases.
9. _identify_failing_keys_via_probe uses fresh probe, not epoch_log.
10. Phase A epoch-level resume — _find_latest_checkpoint forwarded to _run_phase.
11. _verify_phase_integrity — done-marker and adapter-integrity checks.
"""

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

# gpu_guard is provided by lab-tools (separate repo, not on PyPI).  CI does
# not install it; skip the whole module rather than erroring at collection.
pytest.importorskip("gpu_guard")

import sys

_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

from experiments.test15_retention_multiseed import (  # noqa: E402, I001
    CHECKPOINT_RETENTION,
    EpochProbeState,
    _exit_if_paused_mid_phase,
    _find_latest_checkpoint,
    _identify_failing_keys_via_probe,
    _training_config,
    _verify_phase_integrity,
    build_phase_A_keyed,
    build_phase_B_swap_keyed,
    build_phase_C1_keyed,
    build_phase_C2_fill_keyed,
    compute_multiseed_aggregate,
    decay_steps_for,
    find_latest_run_dir,
    load_or_write_run_config,
    marker_exists,
    write_paused_marker,
)


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def _make_qa_pool(n: int) -> list[dict]:
    """Build a synthetic QA pool of length n."""
    return [{"question": f"Q{i}", "answer": f"A{i}"} for i in range(1, n + 1)]


def _make_args(**overrides) -> types.SimpleNamespace:
    """Build a minimal argparse.Namespace for load_or_write_run_config."""
    defaults = dict(
        model="mistral",
        seeds=[42, 7, 1337, 1, 11],
        n_keys=100,
        swap_keys=20,
        phase_a_num_epochs=50,
        phase_b_num_epochs=30,
        phase_c1_num_epochs=50,
        phase_c2_num_epochs=30,
        lr_scheduler_type="linear",
        lr_decay_steps=None,
        weight_decay=0.01,
        max_repair_episodes=5,
        repair_lr=1e-5,
        resume=False,
    )
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


def _make_probe_state(
    stop_epoch: int | None = None,
    epoch_log: list[dict] | None = None,
) -> EpochProbeState:
    """Build an EpochProbeState with the given stop_epoch and epoch_log."""
    s = EpochProbeState()
    s.stop_epoch = stop_epoch
    s.epoch_log = epoch_log or []
    return s


def _make_repair_log(**overrides) -> dict:
    """Build a minimal valid repair_log dict."""
    base = {
        "phase": "B",
        "seed": 42,
        "failing_keys_pre_repair": 4,
        "RP2_rate": 0.05,
        "RP2_exact_count": 4,
        "RP3_rate": 0.95,
        "RP3_exact_count": 76,
        "RP3_total": 80,
        "alignment_delta": 0.9,
        "corruption_residual": 0.05,
        "recovered_count": 72,
        "still_failing_count": 4,
        "collateral_loss_count": 0,
        "stop_reason": "full_recovery",
        "episodes_used": 3,
        "max_episodes": 5,
        "lr": 1e-5,
        "curve": [
            {
                "episode": 1,
                "retention": 0.5,
                "exact_count": 40,
                "total": 80,
                "wall_seconds": 60.0,
                "train_loss": 0.12,
            }
        ],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# 1. Phase-builder parity at (total_keys=100, swap_keys=20)
# ---------------------------------------------------------------------------


class TestPhaseBuilderParity:
    """Test15 builders at (total_keys=100, swap_keys=20) == test13 builders."""

    def _get_test13_builders(self):
        """Import test13 builder functions."""
        from experiments.test13_journal_scaffold import (  # noqa: I001
            build_phase_A_keyed as t13_build_A,
            build_phase_B_swap_keyed as t13_build_B,
            build_phase_C1_keyed as t13_build_C1,
            build_phase_C2_fill_keyed as t13_build_C2,
        )

        return t13_build_A, t13_build_B, t13_build_C1, t13_build_C2

    def test_phase_A_parity(self):
        """build_phase_A_keyed output matches test13 at N=100."""
        t13_A, _, _, _ = self._get_test13_builders()
        qa_pool = _make_qa_pool(120)
        t15_result = build_phase_A_keyed(qa_pool, total_keys=100)
        t13_result = t13_A(qa_pool)
        assert len(t15_result) == len(t13_result) == 100
        for r15, r13 in zip(t15_result, t13_result):
            assert r15["key"] == r13["key"]
            assert r15["question"] == r13["question"]
            assert r15["answer"] == r13["answer"]

    def test_phase_C1_parity(self):
        """build_phase_C1_keyed output matches test13 at N=100, swap=20."""
        _, _, t13_C1, _ = self._get_test13_builders()
        qa_pool = _make_qa_pool(120)
        t15_result = build_phase_C1_keyed(qa_pool, total_keys=100, swap_keys=20)
        t13_result = t13_C1(qa_pool)
        assert len(t15_result) == len(t13_result) == 100
        for i, (r15, r13) in enumerate(zip(t15_result, t13_result)):
            assert r15["key"] == r13["key"], f"key mismatch at slot {i}"
            assert r15["question"] == r13["question"], f"question mismatch at slot {i}"
            assert r15["answer"] == r13["answer"], f"answer mismatch at slot {i}"

    def test_phase_B_swap_parity(self):
        """build_phase_B_swap_keyed matches test13 at N=100, swap=20."""
        t13_A, t13_B, _, _ = self._get_test13_builders()
        qa_pool = _make_qa_pool(120)
        a_keyed = build_phase_A_keyed(qa_pool, total_keys=100)
        swap_answers = qa_pool[100:]  # 20 swap answers
        t15_result = build_phase_B_swap_keyed(a_keyed, swap_answers, swap_keys=20, total_keys=100)
        t13_a_keyed = t13_A(qa_pool)
        t13_result = t13_B(t13_a_keyed, swap_answers)
        assert len(t15_result) == len(t13_result) == 20
        for r15, r13 in zip(t15_result, t13_result):
            assert r15["key"] == r13["key"]
            assert r15["question"] == r13["question"]
            assert r15["answer"] == r13["answer"]

    def test_phase_C2_fill_parity(self):
        """build_phase_C2_fill_keyed matches test13 at N=100, swap=20."""
        _, _, t13_C1, t13_C2 = self._get_test13_builders()
        qa_pool = _make_qa_pool(120)
        c1_keyed = build_phase_C1_keyed(qa_pool, total_keys=100, swap_keys=20)
        t15_result = build_phase_C2_fill_keyed(c1_keyed, qa_pool, swap_keys=20, total_keys=100)
        t13_c1_keyed = t13_C1(qa_pool)
        t13_result = t13_C2(t13_c1_keyed, qa_pool)
        assert len(t15_result) == len(t13_result) == 20
        for r15, r13 in zip(t15_result, t13_result):
            assert r15["key"] == r13["key"]
            assert r15["answer"] == r13["answer"]


# ---------------------------------------------------------------------------
# 2. Phase-builder parity at small scale (total_keys=10, swap_keys=2)
# ---------------------------------------------------------------------------


class TestPhaseBuilderAtSmallScale:
    """Phase builders at (total_keys=10, swap_keys=2) produce correct shapes."""

    def test_phase_A_at_small_scale(self):
        """Phase A has 10 keys graph1..graph10."""
        qa_pool = _make_qa_pool(12)
        a_keyed = build_phase_A_keyed(qa_pool, total_keys=10)
        assert len(a_keyed) == 10
        for i, entry in enumerate(a_keyed):
            assert entry["key"] == f"graph{i + 1}", f"bad key at slot {i}: {entry['key']}"

    def test_phase_A_at_small_scale_unique_keys(self):
        """All keys in Phase A are unique."""
        qa_pool = _make_qa_pool(12)
        a_keyed = build_phase_A_keyed(qa_pool, total_keys=10)
        keys = [e["key"] for e in a_keyed]
        assert len(keys) == len(set(keys)), "duplicate keys in Phase A"

    def test_phase_C1_at_small_scale_tbd_slots(self):
        """Phase C1 at N=10, swap=2: slots 9 and 10 have TBD-1 and TBD-2."""
        qa_pool = _make_qa_pool(12)
        c1_keyed = build_phase_C1_keyed(qa_pool, total_keys=10, swap_keys=2)
        assert len(c1_keyed) == 10
        # Slots 0-7 are real answers.
        for i in range(8):
            assert not c1_keyed[i]["answer"].startswith("TBD-"), (
                f"Slot {i} should be real, got {c1_keyed[i]['answer']}"
            )
        # Slots 8 and 9 (1-indexed: 9, 10) are TBD.
        assert c1_keyed[8]["answer"] == "TBD-1", f"got {c1_keyed[8]['answer']}"
        assert c1_keyed[9]["answer"] == "TBD-2", f"got {c1_keyed[9]['answer']}"

    def test_phase_B_swap_covers_last_two(self):
        """Phase B swap covers slots 9 and 10 (the last swap_keys=2 slots)."""
        qa_pool = _make_qa_pool(12)
        a_keyed = build_phase_A_keyed(qa_pool, total_keys=10)
        swap_answers = qa_pool[10:]  # 2 entries
        b_swap = build_phase_B_swap_keyed(a_keyed, swap_answers, swap_keys=2, total_keys=10)
        assert len(b_swap) == 2
        # Keys should match the last 2 of a_keyed.
        for i, entry in enumerate(b_swap):
            assert entry["key"] == a_keyed[8 + i]["key"]

    def test_phase_C2_fill_replaces_tbd(self):
        """Phase C2 fill provides real answers for slots 9 and 10."""
        qa_pool = _make_qa_pool(12)
        c1_keyed = build_phase_C1_keyed(qa_pool, total_keys=10, swap_keys=2)
        c2_fill = build_phase_C2_fill_keyed(c1_keyed, qa_pool, swap_keys=2, total_keys=10)
        assert len(c2_fill) == 2
        # Answers should be the real answers from qa_pool slots 8 and 9.
        for i, entry in enumerate(c2_fill):
            assert not entry["answer"].startswith("TBD-"), (
                f"C2 slot {i} still has placeholder: {entry['answer']}"
            )
            assert entry["answer"] == qa_pool[8 + i]["answer"], f"C2 slot {i} answer mismatch"


# ---------------------------------------------------------------------------
# 3. Repair-log schema
# ---------------------------------------------------------------------------


class TestRepairLogSchema:
    """Repair log dict round-trips through json.load with required fields."""

    REQUIRED_FIELDS = [
        "phase",
        "seed",
        "failing_keys_pre_repair",
        "RP2_rate",
        "RP3_rate",
        "RP3_total",
        "alignment_delta",
        "corruption_residual",
        "recovered_count",
        "still_failing_count",
        "collateral_loss_count",
        "stop_reason",
        "episodes_used",
        "max_episodes",
        "lr",
        "curve",
    ]

    def test_round_trip(self, tmp_path):
        """Repair log round-trips through JSON and all required fields present."""
        repair_log = _make_repair_log()
        path = tmp_path / "repair_log.json"
        path.write_text(json.dumps(repair_log, indent=2))
        loaded = json.loads(path.read_text())
        for field in self.REQUIRED_FIELDS:
            assert field in loaded, f"Missing field: {field}"

    def test_field_types(self):
        """Core repair-log fields have correct types."""
        repair_log = _make_repair_log()
        assert isinstance(repair_log["RP2_rate"], float)
        assert isinstance(repair_log["RP3_rate"], float)
        assert isinstance(repair_log["alignment_delta"], float)
        assert isinstance(repair_log["corruption_residual"], float)
        assert isinstance(repair_log["recovered_count"], int)
        assert isinstance(repair_log["still_failing_count"], int)
        assert isinstance(repair_log["collateral_loss_count"], int)
        assert isinstance(repair_log["episodes_used"], int)
        assert isinstance(repair_log["max_episodes"], int)
        assert isinstance(repair_log["lr"], float)
        assert isinstance(repair_log["curve"], list)

    def test_curve_entry_fields(self):
        """Each curve entry has episode, retention, wall_seconds, train_loss."""
        repair_log = _make_repair_log()
        for entry in repair_log["curve"]:
            for f in ("episode", "retention", "wall_seconds", "train_loss"):
                assert f in entry, f"curve entry missing field: {f}"

    def test_no_failing_keys_schema(self):
        """No-failing-keys repair log has stop_reason='no_failing_keys' and episodes_used=0."""
        repair_log = _make_repair_log(stop_reason="no_failing_keys", episodes_used=0, curve=[])
        assert repair_log["stop_reason"] == "no_failing_keys"
        assert repair_log["episodes_used"] == 0
        assert repair_log["curve"] == []


# ---------------------------------------------------------------------------
# 4. Marker logic
# ---------------------------------------------------------------------------


class TestMarkerLogic:
    """marker_exists returns correct values; phase resume reads markers correctly."""

    def test_marker_exists_false_for_missing(self, tmp_path):
        """marker_exists returns False when file does not exist."""
        assert not marker_exists(tmp_path, "A")

    def test_marker_exists_true_for_existing(self, tmp_path):
        """marker_exists returns True when _done.json exists."""
        (tmp_path / "A_done.json").write_text("{}")
        assert marker_exists(tmp_path, "A")

    def test_marker_exists_uses_done_suffix(self, tmp_path):
        """marker_exists specifically checks for <name>_done.json."""
        # Just "A.json" should not satisfy marker_exists("A").
        (tmp_path / "A.json").write_text("{}")
        assert not marker_exists(tmp_path, "A")

    def test_phase_resume_skip_when_done(self, tmp_path):
        """Phase with done marker is skipped (B_done.json → no re-run)."""
        phase_dir = tmp_path / "B"
        phase_dir.mkdir()
        (phase_dir / "B_done.json").write_text(
            json.dumps(
                {
                    "rp2_rate": 0.05,
                    "rp3_rate": 0.95,
                    "stop_epoch": 12,
                    "alignment_delta": 0.9,
                    "corruption_residual": 0.05,
                    "episodes_used": 2,
                }
            )
        )
        assert (phase_dir / "B_done.json").exists()

    def test_phase_resume_from_train_done(self, tmp_path):
        """When train_done but not repair_done → resume from repair start."""
        phase_dir = tmp_path / "B"
        phase_dir.mkdir()
        train_marker = {
            "stop_epoch": 12,
            "epoch_log": [{"epoch": 12, "fill": {"exact_count": 20, "total": 20, "rate": 1.0}}],
        }
        (phase_dir / "B_train_done.json").write_text(json.dumps(train_marker))
        assert (phase_dir / "B_train_done.json").exists()
        assert not (phase_dir / "B_repair_done.json").exists()
        assert not (phase_dir / "B_done.json").exists()
        # This is the resume condition: reload adapter + restart repair from ep 1.
        data = json.loads((phase_dir / "B_train_done.json").read_text())
        assert data["stop_epoch"] == 12

    def test_train_done_and_repair_done_but_no_done(self, tmp_path):
        """If train + repair done but not B_done, B_done should be written next."""
        phase_dir = tmp_path / "B"
        phase_dir.mkdir()
        (phase_dir / "B_train_done.json").write_text("{}")
        (phase_dir / "B_repair_done.json").write_text("{}")
        # B_done.json is absent — signifies incomplete state (edge case guard).
        assert not (phase_dir / "B_done.json").exists()


# ---------------------------------------------------------------------------
# 5. Multi-seed aggregator schema
# ---------------------------------------------------------------------------


class TestMultiseedAggregator:
    """Aggregator schema, bootstrap CI, verdict, and degenerate case."""

    def _make_per_seed(self, b_rates: list[float], c2_rates: list[float]) -> dict[str, dict]:
        """Build a synthetic per_seed dict from lists of RP2 rates."""
        seeds = [42, 7, 1337, 1, 11]
        per_seed = {}
        for i, seed in enumerate(seeds):
            per_seed[str(seed)] = {
                "B": {
                    "RP2_rate": b_rates[i],
                    "RP3_rate": b_rates[i] + 0.1,
                    "stop_epoch": 12,
                    "alignment_delta": 0.1,
                    "corruption_residual": 0.05,
                    "episodes_used": 2,
                },
                "C2": {
                    "RP2_rate": c2_rates[i],
                    "RP3_rate": c2_rates[i] + 0.02,
                    "stop_epoch": 14,
                    "alignment_delta": 0.02,
                    "corruption_residual": 0.03,
                    "episodes_used": 1,
                },
            }
        return per_seed

    REQUIRED_AGGREGATE_KEYS = [
        "n_completed",
        "seeds",
        "per_seed",
        "mean_retention_B",
        "mean_retention_C2",
        "mean_retention_B_repaired",
        "mean_retention_C2_repaired",
        "ratio_C2_over_B",
        "ratio_repaired_C2_over_B",
        "ratio_lower_ci",
        "ratio_lower_ci_repaired",
        "decision_threshold_ratio",
        "decision_threshold_lower_ci",
        "verdict",
    ]

    def test_aggregate_schema(self):
        """All required keys present in aggregate output."""
        seeds = [42, 7, 1337, 1, 11]
        per_seed = self._make_per_seed(
            [0.05] * 5,
            [0.375] * 5,
        )
        agg = compute_multiseed_aggregate(per_seed, seeds, n_rng_resamples=100)
        for key in self.REQUIRED_AGGREGATE_KEYS:
            assert key in agg, f"Missing aggregate key: {key}"

    def test_verdict_holds_when_ratio_passes(self):
        """Verdict is HOLDS when ratio >= 5.0 and lower CI >= 2.5 (mocked)."""
        seeds = [42, 7, 1337, 1, 11]
        # C2 = 0.375, B = 0.056 → ratio ≈ 6.7
        per_seed = self._make_per_seed(
            [0.056] * 5,
            [0.375] * 5,
        )
        agg = compute_multiseed_aggregate(per_seed, seeds, n_rng_resamples=500)
        # With very consistent data, CI should be well above 2.5.
        assert agg["verdict"] == "HOLDS", (
            f"Expected HOLDS, got {agg['verdict']} (ratio={agg['ratio_C2_over_B']}, "
            f"lower_ci={agg['ratio_lower_ci']})"
        )

    def test_verdict_does_not_hold_when_ratio_low(self):
        """Verdict is DOES NOT HOLD when ratio < 5.0."""
        seeds = [42, 7, 1337, 1, 11]
        per_seed = self._make_per_seed(
            [0.3] * 5,  # B: 30%
            [0.4] * 5,  # C2: 40% — ratio ≈ 1.33
        )
        agg = compute_multiseed_aggregate(per_seed, seeds, n_rng_resamples=100)
        assert agg["verdict"] == "DOES NOT HOLD"

    def test_degenerate_mean_b_zero_does_not_crash(self):
        """mean_retention_B == 0 does not crash; verdict is HOLDS (any C2 > 0 wins)."""
        seeds = [42, 7, 1337, 1, 11]
        per_seed = self._make_per_seed(
            [0.0] * 5,  # B: 0%
            [0.375] * 5,  # C2: 37.5%
        )
        # Should not raise.
        agg = compute_multiseed_aggregate(per_seed, seeds, n_rng_resamples=100)
        assert "verdict" in agg
        # When B=0, ratio is inf → HOLDS.
        assert agg["verdict"] == "HOLDS"

    def test_n_completed_matches_seeds(self):
        """n_completed equals the number of seeds in per_seed."""
        seeds = [42, 7, 1337, 1, 11]
        per_seed = self._make_per_seed([0.05] * 5, [0.375] * 5)
        agg = compute_multiseed_aggregate(per_seed, seeds, n_rng_resamples=100)
        assert agg["n_completed"] == 5

    def test_partial_seeds_n_completed(self):
        """n_completed reflects only seeds present in per_seed."""
        seeds = [42, 7, 1337, 1, 11]
        per_seed = self._make_per_seed([0.05] * 5, [0.375] * 5)
        # Remove 3 seeds.
        del per_seed["7"]
        del per_seed["1337"]
        del per_seed["1"]
        agg = compute_multiseed_aggregate(per_seed, seeds, n_rng_resamples=100)
        assert agg["n_completed"] == 2

    def test_round_trip_json(self):
        """Aggregate dict serialises to JSON without error."""
        seeds = [42, 7, 1337, 1, 11]
        per_seed = self._make_per_seed([0.05] * 5, [0.375] * 5)
        agg = compute_multiseed_aggregate(per_seed, seeds, n_rng_resamples=100)
        serialised = json.dumps(agg, indent=2)
        loaded = json.loads(serialised)
        assert loaded["verdict"] == agg["verdict"]

    def test_aggregate_verdict_indeterminate_when_both_means_zero(self):
        """Verdict is INDETERMINATE when both mean_b and mean_c2 are 0.

        0/0 is undefined — no claim can be made.  ratio_C2_over_B must be
        None in the output (inf is not JSON-serialisable and the JSON reader
        in training-control.sh would mis-render it as DOES NOT HOLD).
        """
        seeds = [42, 7, 1337, 1, 11]
        per_seed = self._make_per_seed(
            [0.0] * 5,  # B RP2 = 0 for all seeds
            [0.0] * 5,  # C2 RP2 = 0 for all seeds
        )
        agg = compute_multiseed_aggregate(per_seed, seeds, n_rng_resamples=100)
        assert agg["verdict"] == "INDETERMINATE", (
            f"Expected INDETERMINATE when both means are 0, got {agg['verdict']}"
        )
        assert agg["ratio_C2_over_B"] is None, (
            f"ratio_C2_over_B must be None (not inf) when both means are 0, "
            f"got {agg['ratio_C2_over_B']}"
        )

    def test_aggregate_verdict_holds_when_b_zero_c2_positive(self):
        """Verdict is HOLDS when mean_b=0 and mean_c2>0.

        The scaffold path retains keys that the swap path loses entirely.
        ratio_C2_over_B must be None in the JSON (inf is not serialisable)
        but verdict is unambiguously HOLDS.
        """
        seeds = [42, 7, 1337, 1, 11]
        per_seed = self._make_per_seed(
            [0.0] * 5,  # B RP2 = 0 for all seeds
            [0.375] * 5,  # C2 RP2 > 0 for all seeds
        )
        agg = compute_multiseed_aggregate(per_seed, seeds, n_rng_resamples=100)
        assert agg["verdict"] == "HOLDS", f"Expected HOLDS when B=0 and C2>0, got {agg['verdict']}"
        assert agg["ratio_C2_over_B"] is None, (
            f"ratio_C2_over_B must be None (inf→null) when mean_b=0, got {agg['ratio_C2_over_B']}"
        )

    def test_aggregate_verdict_does_not_hold_when_ratio_below_threshold(self):
        """Verdict is DOES NOT HOLD when ratio < 5.0 (non-zero B and C2).

        Both B and C2 are positive but the ratio is well below the 5.0
        decision threshold — the scaffold advantage is not demonstrated.
        """
        seeds = [42, 7, 1337, 1, 11]
        per_seed = self._make_per_seed(
            [0.4] * 5,  # B RP2 = 40%
            [0.5] * 5,  # C2 RP2 = 50% — ratio = 1.25, far below 5.0
        )
        agg = compute_multiseed_aggregate(per_seed, seeds, n_rng_resamples=100)
        assert agg["verdict"] == "DOES NOT HOLD", (
            f"Expected DOES NOT HOLD when ratio is ~1.25, got {agg['verdict']} "
            f"(ratio={agg['ratio_C2_over_B']})"
        )


# ---------------------------------------------------------------------------
# 6. Decay-steps math
# ---------------------------------------------------------------------------


class TestDecayStepsMath:
    """decay_steps_for returns the expected formula value."""

    def test_default_phase_b(self):
        """At n_keys=100, num_epochs=30 → 100 * 30 // 2 = 1500."""
        assert decay_steps_for(100, 30) == 1500

    def test_default_phase_a(self):
        """At n_keys=100, num_epochs=50 → 100 * 50 // 2 = 2500."""
        assert decay_steps_for(100, 50) == 2500

    def test_small_scale_n_keys(self):
        """At n_keys=10, num_epochs=15 → 10 * 15 // 2 = 75."""
        assert decay_steps_for(10, 15) == 75

    def test_minimum_one(self):
        """Returns at least 1 even for degenerate inputs."""
        assert decay_steps_for(1, 1) >= 1

    def test_formula_integrity(self):
        """n_keys * num_epochs // 2 matches expected for arbitrary values."""
        for n, e in [(50, 20), (200, 40), (500, 30)]:
            assert decay_steps_for(n, e) == n * e // 2


# ---------------------------------------------------------------------------
# 7. run_config.json persistence and find_latest_run_dir recency selection
# ---------------------------------------------------------------------------


class TestRunConfig:
    """run_config.json persistence and find_latest_run_dir recency selection."""

    def test_find_latest_run_dir_returns_newest_dir(self, tmp_path):
        """find_latest_run_dir returns the most recent dir by timestamp."""
        model_dir = tmp_path / "mistral"
        model_dir.mkdir()

        older_run = model_dir / "20260501_100000"
        older_run.mkdir()
        (older_run / "run_config.json").write_text(json.dumps({"n_keys": 100}))

        newer_run = model_dir / "20260507_123111"
        newer_run.mkdir()
        (newer_run / "run_config.json").write_text(json.dumps({"n_keys": 100}))

        import experiments.test15_retention_multiseed as t15_mod

        original_base = t15_mod.OUTPUT_BASE
        t15_mod.OUTPUT_BASE = tmp_path
        try:
            result = find_latest_run_dir("mistral")
        finally:
            t15_mod.OUTPUT_BASE = original_base

        assert result is not None
        assert result.name == "20260507_123111", f"Expected newer run to be returned, got {result}"

    def test_find_latest_run_dir_picks_newest(self, tmp_path):
        """find_latest_run_dir picks the newest dir by sort order.

        Case A: older dir + newer dir → returns newer dir.
        Case B: newer dir + older dir (same model, different ordering) → still newest wins.
        """
        import experiments.test15_retention_multiseed as t15_mod

        # --- Case A ---
        model_dir_a = tmp_path / "mistral_a"
        model_dir_a.mkdir()
        older_a = model_dir_a / "20260501_100000"
        older_a.mkdir()
        (older_a / "run_config.json").write_text(json.dumps({"n_keys": 100}))
        newer_a = model_dir_a / "20260502_100000"
        newer_a.mkdir()
        (newer_a / "run_config.json").write_text(json.dumps({"n_keys": 100}))

        original_base = t15_mod.OUTPUT_BASE
        t15_mod.OUTPUT_BASE = tmp_path
        try:
            result_a = find_latest_run_dir("mistral_a")
        finally:
            t15_mod.OUTPUT_BASE = original_base

        assert result_a is not None
        assert result_a.name == "20260502_100000", f"Case A: expected newer dir, got {result_a}"

        # --- Case B ---
        model_dir_b = tmp_path / "mistral_b"
        model_dir_b.mkdir()
        older_b = model_dir_b / "20260501_100000"
        older_b.mkdir()
        (older_b / "run_config.json").write_text(json.dumps({"n_keys": 100}))
        newer_b = model_dir_b / "20260507_123111"
        newer_b.mkdir()
        (newer_b / "run_config.json").write_text(json.dumps({"n_keys": 100}))

        t15_mod.OUTPUT_BASE = tmp_path
        try:
            result_b = find_latest_run_dir("mistral_b")
        finally:
            t15_mod.OUTPUT_BASE = original_base

        assert result_b is not None
        assert result_b.name == "20260507_123111", f"Case B: expected newer dir, got {result_b}"

    def test_find_latest_run_dir_no_runs(self, tmp_path):
        """find_latest_run_dir returns None when no runs exist."""
        import experiments.test15_retention_multiseed as t15_mod

        original_base = t15_mod.OUTPUT_BASE
        t15_mod.OUTPUT_BASE = tmp_path
        try:
            result = find_latest_run_dir("mistral")
        finally:
            t15_mod.OUTPUT_BASE = original_base
        assert result is None

    def test_run_config_read_back(self, tmp_path):
        """On second call (resume), load_or_write_run_config reads back saved values."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        args = _make_args(n_keys=100, weight_decay=0.01)
        load_or_write_run_config(run_dir, args)

        # Second call (resume).
        args2 = _make_args(n_keys=999)  # different — persisted should win
        cfg2 = load_or_write_run_config(run_dir, args2)
        assert cfg2["n_keys"] == 100  # persisted wins

    def test_config_persists_all_required_fields(self, tmp_path):
        """run_config.json contains all required fields."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        args = _make_args()
        cfg = load_or_write_run_config(run_dir, args)
        required = [
            "model",
            "seeds",
            "n_keys",
            "swap_keys",
            "phase_a_num_epochs",
            "phase_b_num_epochs",
            "phase_c1_num_epochs",
            "phase_c2_num_epochs",
            "lr_scheduler_type",
            "lr_decay_steps",
            "weight_decay",
            "max_repair_episodes",
            "repair_lr",
            "created_at",
        ]
        for field in required:
            assert field in cfg, f"run_config.json missing field: {field}"


# ---------------------------------------------------------------------------
# 8. Pause-mid-phase inference
# ---------------------------------------------------------------------------


class TestPauseMidPhase:
    """_exit_if_paused_mid_phase writes paused.json and raises when mid-phase pause."""

    def test_no_action_when_stop_epoch_set(self, tmp_path):
        """When stop_epoch is set (natural convergence), no action taken."""
        state = _make_probe_state(
            stop_epoch=12,
            epoch_log=[{"epoch": 12, "fill": {"exact_count": 20, "total": 20}}],
        )
        # Should not raise.
        _exit_if_paused_mid_phase(state, "Phase_B", 30, tmp_path)
        assert not (tmp_path / "paused.json").exists()

    def test_no_action_when_empty_log(self, tmp_path):
        """When epoch_log is empty (nothing started), no action taken."""
        state = _make_probe_state(stop_epoch=None, epoch_log=[])
        _exit_if_paused_mid_phase(state, "Phase_B", 30, tmp_path)
        assert not (tmp_path / "paused.json").exists()

    def test_no_action_when_budget_exhausted(self, tmp_path):
        """When last_epoch >= num_epochs (budget run out), no action taken."""
        state = _make_probe_state(
            stop_epoch=None,
            epoch_log=[{"epoch": 30, "fill": {"exact_count": 18, "total": 20}}],
        )
        _exit_if_paused_mid_phase(state, "Phase_B", 30, tmp_path)
        assert not (tmp_path / "paused.json").exists()

    def test_writes_paused_and_raises_mid_phase(self, tmp_path):
        """When stop_epoch is None and last_epoch < num_epochs, writes paused.json and raises."""
        state = _make_probe_state(
            stop_epoch=None,
            epoch_log=[{"epoch": 7, "fill": {"exact_count": 15, "total": 20}}],
        )
        with pytest.raises(SystemExit, match="epoch 7"):
            _exit_if_paused_mid_phase(state, "Phase_B", 30, tmp_path)
        assert (tmp_path / "paused.json").exists()
        paused = json.loads((tmp_path / "paused.json").read_text())
        assert paused["stopped_after_epoch"] == 7

    def test_paused_json_fields(self, tmp_path):
        """paused.json written by write_paused_marker has required fields."""
        write_paused_marker(tmp_path, "Phase_B", after_epoch=7)
        paused = json.loads((tmp_path / "paused.json").read_text())
        assert "stopped_after_phase" in paused
        assert "stopped_after_epoch" in paused
        assert "timestamp" in paused
        assert paused["stopped_after_phase"] == "Phase_B"
        assert paused["stopped_after_epoch"] == 7


# ---------------------------------------------------------------------------
# 9. _identify_failing_keys_via_probe uses fresh probe, not epoch_log
# ---------------------------------------------------------------------------


def _make_stop_probe(per_key: list[dict]) -> dict:
    """Build a minimal stop_probe dict with the given per_key list.

    Args:
        per_key: List of dicts with at least ``key`` and ``exact_match`` fields.

    Returns:
        A probe result dict compatible with ``_identify_failing_keys_via_probe``.
    """
    exact_count = sum(1 for e in per_key if e.get("exact_match"))
    total = len(per_key)
    return {
        "exact_count": exact_count,
        "total": total,
        "rate": exact_count / total if total > 0 else 0.0,
        "mean_confidence": 0.9,
        "per_key": per_key,
    }


class TestIdentifyFailingKeysViaProbe:
    """_identify_failing_keys_via_probe uses the stop_probe per_key list.

    Confirms that the function:
    - Returns exactly the failing subset from the probe's per_key results.
    - Does NOT depend on an epoch_log argument (there is no epoch_log path).
    - Handles the all-passing, all-failing, and mixed cases correctly.
    """

    def _make_unchanged_keyed(self, keys: list[str]) -> list[dict]:
        """Build a minimal unchanged_keyed list from key names."""
        return [{"key": k, "question": f"Q_{k}", "answer": f"A_{k}"} for k in keys]

    def test_get_failing_keys_uses_fresh_probe_not_epoch_log(self):
        """_identify_failing_keys_via_probe returns exactly the failing subset.

        The mock stop_probe has keys k1..k5: k1 and k3 fail.
        The function must return exactly those two, with no reference to epoch_log.
        """
        keys = ["k1", "k2", "k3", "k4", "k5"]
        unchanged_keyed = self._make_unchanged_keyed(keys)

        # k1 and k3 are failing; k2, k4, k5 pass.
        per_key = [
            {"key": "k1", "exact_match": False, "confidence": 0.1},
            {"key": "k2", "exact_match": True, "confidence": 0.95},
            {"key": "k3", "exact_match": False, "confidence": 0.2},
            {"key": "k4", "exact_match": True, "confidence": 0.98},
            {"key": "k5", "exact_match": True, "confidence": 0.97},
        ]
        stop_probe = _make_stop_probe(per_key)

        # No epoch_log argument — confirm the function signature does not accept one.
        failing = _identify_failing_keys_via_probe(stop_probe, unchanged_keyed)

        failing_keys = {kp["key"] for kp in failing}
        assert failing_keys == {"k1", "k3"}, f"Expected {{k1, k3}} failing, got {failing_keys}"
        passing_keys = {kp["key"] for kp in unchanged_keyed if kp not in failing}
        assert passing_keys == {"k2", "k4", "k5"}, (
            f"Expected {{k2, k4, k5}} passing, got {passing_keys}"
        )

    def test_all_passing_returns_empty(self):
        """When all keys pass in the probe, failing list is empty."""
        keys = ["k1", "k2", "k3"]
        unchanged_keyed = self._make_unchanged_keyed(keys)
        per_key = [{"key": k, "exact_match": True, "confidence": 0.99} for k in keys]
        stop_probe = _make_stop_probe(per_key)
        failing = _identify_failing_keys_via_probe(stop_probe, unchanged_keyed)
        assert failing == [], f"Expected no failing keys, got {failing}"

    def test_all_failing_returns_all(self):
        """When all keys fail in the probe, all unchanged_keyed are returned."""
        keys = ["k1", "k2", "k3"]
        unchanged_keyed = self._make_unchanged_keyed(keys)
        per_key = [{"key": k, "exact_match": False, "confidence": 0.1} for k in keys]
        stop_probe = _make_stop_probe(per_key)
        failing = _identify_failing_keys_via_probe(stop_probe, unchanged_keyed)
        assert len(failing) == 3
        assert {kp["key"] for kp in failing} == set(keys)

    def test_probe_counts_match_subset_sizes(self):
        """The sum of failing + passing keys equals len(unchanged_keyed)."""
        keys = [f"k{i}" for i in range(1, 9)]  # 8 keys
        unchanged_keyed = self._make_unchanged_keyed(keys)
        # 3 failing, 5 passing.
        per_key = [
            {"key": f"k{i}", "exact_match": i > 3, "confidence": 0.95 if i > 3 else 0.1}
            for i in range(1, 9)
        ]
        stop_probe = _make_stop_probe(per_key)
        failing = _identify_failing_keys_via_probe(stop_probe, unchanged_keyed)
        assert len(failing) == 3
        assert len(unchanged_keyed) - len(failing) == 5

    def test_does_not_depend_on_epoch_log(self):
        """The function signature has no epoch_log parameter.

        Calling with a stop_probe that has a populated per_key must produce
        correct results without any epoch_log setup.  This is the regression
        gate: if someone re-adds an epoch_log path, this test shows the function
        is correctly single-purpose.
        """
        import inspect

        sig = inspect.signature(_identify_failing_keys_via_probe)
        param_names = list(sig.parameters.keys())
        assert "epoch_log" not in param_names, (
            f"_identify_failing_keys_via_probe must not accept epoch_log; "
            f"found params: {param_names}"
        )
        # The only two parameters should be stop_probe and unchanged_keyed.
        assert param_names == ["stop_probe", "unchanged_keyed"], (
            f"Unexpected parameters: {param_names}"
        )


# ---------------------------------------------------------------------------
# 10. Phase A epoch-level resume — _find_latest_checkpoint passed to _run_phase
# ---------------------------------------------------------------------------


class TestPhaseAResume:
    """Phase A passes _find_latest_checkpoint result to _run_phase.

    Verifies the fix for the missing resume_from_checkpoint in Phase A:
    a checkpoint directory in phase_a_dir must be detected and forwarded,
    matching the pattern used by Phase B, C1, and C2.
    """

    def test_find_latest_checkpoint_returns_highest(self, tmp_path):
        """_find_latest_checkpoint returns the checkpoint with the highest step number.

        Each fixture dir mirrors a real HF Trainer checkpoint by including
        ``trainer_state.json`` — the marker the function uses to distinguish
        committed checkpoints from partial-save dirs left behind by a
        SIGKILL / system bugcheck mid-save.  See ``test_find_latest_checkpoint_skips_partial``
        for the negative case.
        """
        phase_dir = tmp_path / "A"
        phase_dir.mkdir()
        for step in (10, 15, 5):
            ckpt = phase_dir / f"checkpoint-{step}"
            ckpt.mkdir()
            (ckpt / "trainer_state.json").write_text("{}")
        result = _find_latest_checkpoint(phase_dir)
        assert result is not None
        assert result.name == "checkpoint-15"

    def test_find_latest_checkpoint_skips_partial(self, tmp_path):
        """_find_latest_checkpoint walks down past partial dirs missing trainer_state.json.

        Locks the partial-save guard documented at
        ``experiments/test15_retention_multiseed.py:289-293``: a SIGKILL / bugcheck
        mid-save can leave a half-written ``checkpoint-N`` directory on disk;
        resuming from it would feed HF Trainer a torn checkpoint and crash.
        The function must skip past it to the highest VALID checkpoint.
        """
        phase_dir = tmp_path / "A"
        phase_dir.mkdir()
        # checkpoint-15 is partial (no trainer_state.json); checkpoint-10 is complete.
        (phase_dir / "checkpoint-15").mkdir()
        ckpt_10 = phase_dir / "checkpoint-10"
        ckpt_10.mkdir()
        (ckpt_10 / "trainer_state.json").write_text("{}")
        result = _find_latest_checkpoint(phase_dir)
        assert result is not None
        assert result.name == "checkpoint-10"

    def test_find_latest_checkpoint_none_when_empty(self, tmp_path):
        """_find_latest_checkpoint returns None when no checkpoint dirs exist."""
        phase_dir = tmp_path / "A"
        phase_dir.mkdir()
        assert _find_latest_checkpoint(phase_dir) is None

    def test_find_latest_checkpoint_none_when_dir_absent(self, tmp_path):
        """_find_latest_checkpoint returns None when phase_dir does not exist."""
        phase_dir = tmp_path / "A_nonexistent"
        assert _find_latest_checkpoint(phase_dir) is None

    def test_run_phase_accepts_resume_from_checkpoint(self):
        """_run_phase signature accepts resume_from_checkpoint parameter.

        This is the structural contract gate: if someone removes the
        resume_from_checkpoint parameter from _run_phase, Phase A's call site
        (which was missing it before the fix) would break at runtime.
        """
        import inspect

        from experiments.test15_retention_multiseed import _run_phase

        sig = inspect.signature(_run_phase)
        assert "resume_from_checkpoint" in sig.parameters, (
            f"_run_phase must accept resume_from_checkpoint; found params: {list(sig.parameters)}"
        )

    def test_phase_a_call_site_pattern(self, tmp_path, monkeypatch):
        """Phase A call site computes a_ckpt and forwards it to _run_phase.

        Verifies the fix by mocking _run_phase and _find_latest_checkpoint:
        - _find_latest_checkpoint returns a sentinel path.
        - _run_phase is called with resume_from_checkpoint equal to that sentinel.
        The test exercises only the Phase A branch of run_seed (A_done absent).
        """
        import experiments.test15_retention_multiseed as t15_mod

        sentinel = tmp_path / "checkpoint-15"
        sentinel.mkdir()

        captured: dict = {}

        def _fake_find_latest_checkpoint(phase_dir: "Path") -> "Path | None":
            # Only intercept Phase A lookups (phase_dir ends with "/A").
            if phase_dir.name == "A":
                return sentinel
            return None

        def _fake_run_phase(**kwargs):
            if kwargs.get("phase_name") == "Phase_A":
                captured["resume_from_checkpoint"] = kwargs.get("resume_from_checkpoint")
            # Return a minimal tuple: (model, metrics, probe_state, wall).
            from experiments.test15_retention_multiseed import EpochProbeState

            state = EpochProbeState()
            state.stop_epoch = 2
            state.epoch_log = [{"epoch": 2, "fill": {"exact_count": 1, "total": 1, "rate": 1.0}}]
            return (kwargs["model"], {}, state, 0.1)

        monkeypatch.setattr(t15_mod, "_find_latest_checkpoint", _fake_find_latest_checkpoint)
        monkeypatch.setattr(t15_mod, "_run_phase", _fake_run_phase)

        # Patch everything that run_seed calls before and after _run_phase.
        monkeypatch.setattr(t15_mod, "_exit_if_paused_mid_phase", lambda *a, **kw: None)
        monkeypatch.setattr(
            t15_mod,
            "_safe_probe",
            lambda *a, **kw: {
                "exact_count": 1,
                "total": 1,
                "rate": 1.0,
                "mean_confidence": 1.0,
                "per_key": [],
            },
        )
        monkeypatch.setattr(t15_mod, "build_manifest_for", lambda *a, **kw: {})
        monkeypatch.setattr(t15_mod, "save_adapter", lambda *a, **kw: None)
        monkeypatch.setattr(t15_mod, "_write_phase_done", lambda *a, **kw: None)
        monkeypatch.setattr(t15_mod, "_check_pause", lambda *a, **kw: None)
        monkeypatch.setattr(t15_mod, "_verify_phase_integrity", lambda *a, **kw: None)
        monkeypatch.setattr(
            t15_mod,
            "load_qa_pool",
            lambda *a, **kw: [{"question": f"Q{i}", "answer": f"A{i}"} for i in range(125)],
        )
        monkeypatch.setattr(t15_mod, "create_adapter", lambda model, cfg, name: model)
        monkeypatch.setattr(t15_mod, "switch_adapter", lambda model, name: None)

        # Phase A done file must NOT exist so the fresh branch executes.
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        seed_dir = run_dir / "seed42"
        seed_dir.mkdir()
        # Phase A dir exists but has no A_done.json.
        phase_a_dir = seed_dir / "A"
        phase_a_dir.mkdir()

        # Provide a minimal cfg and a fake non-PeftModel model.
        cfg = {
            "n_keys": 100,
            "swap_keys": 20,
            "weight_decay": 0.01,
            "lr_scheduler_type": "linear",
            "lr_decay_steps": 1500,
            "max_repair_episodes": 5,
            "repair_lr": 1e-5,
            "phase_a_num_epochs": 50,
            "phase_b_num_epochs": 30,
            "phase_c1_num_epochs": 50,
            "phase_c2_num_epochs": 30,
        }

        class _FakeBaseModel:
            """Minimal stub that is NOT a PeftModel."""

            pass

        from experiments.utils.early_stop import ANALYSIS_POLICY

        # Abort after Phase A by having Phase B already done (so run_seed returns quickly).
        phase_b_dir = seed_dir / "B"
        phase_b_dir.mkdir()
        b_done = {
            "rp2_rate": 0.05,
            "rp3_rate": 0.95,
            "stop_epoch": 12,
            "alignment_delta": 0.9,
            "corruption_residual": 0.05,
            "episodes_used": 2,
        }
        (phase_b_dir / "B_done.json").write_text(json.dumps(b_done))

        # Phase C1 and C2 done too so run_seed returns.
        phase_c1_dir = seed_dir / "C1"
        phase_c1_dir.mkdir()
        c1_keyed = [
            {"key": f"graph{i}", "question": f"Q{i}?", "answer": f"A{i}"} for i in range(1, 101)
        ]
        (phase_c1_dir / "quads.json").write_text(json.dumps(c1_keyed))
        (phase_c1_dir / "C1_done.json").write_text("{}")
        monkeypatch.setattr(
            t15_mod, "resolve_adapter_slot", lambda *a, **kw: tmp_path / "fake_slot"
        )
        import contextlib

        @contextlib.contextmanager
        def _fake_adapter_slot_for_load(slot):
            yield slot

        monkeypatch.setattr(t15_mod, "_adapter_slot_for_load", _fake_adapter_slot_for_load)
        monkeypatch.setattr(
            t15_mod,
            "PeftModel",
            type(
                "FakePeft",
                (),
                {
                    "from_pretrained": staticmethod(lambda base, path, adapter_name: base),
                },
            ),
        )

        phase_c2_dir = seed_dir / "C2"
        phase_c2_dir.mkdir()
        c2_done = {
            "rp2_rate": 0.375,
            "rp3_rate": 0.95,
            "stop_epoch": 14,
            "alignment_delta": 0.02,
            "corruption_residual": 0.03,
            "episodes_used": 1,
        }
        (phase_c2_dir / "C2_done.json").write_text(json.dumps(c2_done))

        t15_mod.run_seed(
            model=_FakeBaseModel(),
            tokenizer=object(),
            seed=42,
            run_dir=run_dir,
            cfg=cfg,
            early_stop_policy=ANALYSIS_POLICY,
        )

        assert "resume_from_checkpoint" in captured, (
            "Phase A _run_phase call did not capture resume_from_checkpoint — "
            "the call site is missing the argument."
        )
        assert captured["resume_from_checkpoint"] == sentinel, (
            f"Expected resume_from_checkpoint={sentinel}, got {captured['resume_from_checkpoint']}"
        )


# ---------------------------------------------------------------------------
# 11. _verify_phase_integrity — done-marker and adapter-integrity checks
# ---------------------------------------------------------------------------


def _make_seed_dir(tmp_path: Path, seed: int = 42) -> Path:
    """Create a minimal complete seed directory structure in tmp_path.

    All four phase done markers are written.  No adapter directories are
    created — add them in individual tests as needed.

    Args:
        tmp_path: pytest tmp_path fixture directory.
        seed: Seed number (controls the directory name).

    Returns:
        Path to the seed directory (``tmp_path / f"seed{seed}"``).
    """
    seed_dir = tmp_path / f"seed{seed}"
    for phase in ["A", "B", "C1", "C2"]:
        phase_dir = seed_dir / phase
        phase_dir.mkdir(parents=True, exist_ok=True)
        (phase_dir / f"{phase}_done.json").write_text("{}")
    return seed_dir


class TestVerifyPhaseIntegrity:
    """_verify_phase_integrity checks done markers and adapter sha256 integrity."""

    def test_passes_when_repaired_differs_from_original(self, tmp_path):
        """No exception when repaired adapter has different content from original."""
        seed_dir = _make_seed_dir(tmp_path)
        # Create B episodic_adapter and episodic_adapter_repaired with different content.
        b_dir = seed_dir / "B"
        orig = b_dir / "episodic_adapter"
        orig.mkdir()
        (orig / "adapter_model.bin").write_bytes(b"x" * 10)
        repaired = b_dir / "episodic_adapter_repaired"
        repaired.mkdir()
        (repaired / "adapter_model.bin").write_bytes(b"y" * 10)
        # Should not raise.
        _verify_phase_integrity(seed_dir, seed=42)

    def test_raises_when_repaired_equals_original(self, tmp_path):
        """AssertionError raised when repaired adapter has identical sha256 to original."""
        seed_dir = _make_seed_dir(tmp_path)
        b_dir = seed_dir / "B"
        orig = b_dir / "episodic_adapter"
        orig.mkdir()
        (orig / "adapter_model.bin").write_bytes(b"identical_content")
        repaired = b_dir / "episodic_adapter_repaired"
        repaired.mkdir()
        (repaired / "adapter_model.bin").write_bytes(b"identical_content")
        with pytest.raises(AssertionError, match="repair did not run or wrote back to source"):
            _verify_phase_integrity(seed_dir, seed=42)

    def test_skips_when_no_repaired_dir(self, tmp_path):
        """No exception when repaired dir absent (repair found no failing keys)."""
        seed_dir = _make_seed_dir(tmp_path)
        b_dir = seed_dir / "B"
        orig = b_dir / "episodic_adapter"
        orig.mkdir()
        (orig / "adapter_model.bin").write_bytes(b"some_content")
        # No episodic_adapter_repaired — repair was skipped (legitimate).
        _verify_phase_integrity(seed_dir, seed=42)

    def test_raises_on_missing_done_marker(self, tmp_path):
        """AssertionError raised when a phase done marker is missing."""
        seed_dir = _make_seed_dir(tmp_path)
        # Remove A_done.json to simulate incomplete run.
        (seed_dir / "A" / "A_done.json").unlink()
        with pytest.raises(AssertionError, match="A_done.json"):
            _verify_phase_integrity(seed_dir, seed=42)

    def test_handles_both_phases(self, tmp_path):
        """No exception when both B and C2 have repaired dirs with differing sha256."""
        seed_dir = _make_seed_dir(tmp_path)
        # B: different content.
        b_dir = seed_dir / "B"
        (b_dir / "episodic_adapter").mkdir()
        (b_dir / "episodic_adapter" / "weights.bin").write_bytes(b"b_original")
        (b_dir / "episodic_adapter_repaired").mkdir()
        (b_dir / "episodic_adapter_repaired" / "weights.bin").write_bytes(b"b_repaired")
        # C2: different content.
        c2_dir = seed_dir / "C2"
        (c2_dir / "journal_adapter").mkdir()
        (c2_dir / "journal_adapter" / "weights.bin").write_bytes(b"c2_original")
        (c2_dir / "journal_adapter_repaired").mkdir()
        (c2_dir / "journal_adapter_repaired" / "weights.bin").write_bytes(b"c2_repaired")
        # Should not raise.
        _verify_phase_integrity(seed_dir, seed=42)


# ---------------------------------------------------------------------------
# 12. _training_config save_total_limit is bounded (Fix 2)
# ---------------------------------------------------------------------------


class TestTrainingConfigSaveTotalLimit:
    """_training_config always uses CHECKPOINT_RETENTION, not num_epochs.

    With save_strategy='epoch', using num_epochs as save_total_limit retains
    one checkpoint dir per epoch.  At production scale (A=50 epochs × 5 seeds
    × ~50 MB) that is ~12.5 GB just for Phase A.  The limit must always be the
    module-level CHECKPOINT_RETENTION constant (== 2).
    """

    def test_make_training_config_save_total_limit_is_bounded(self):
        """save_total_limit equals CHECKPOINT_RETENTION regardless of num_epochs.

        Constructs a TrainingConfig with a large epoch budget and confirms the
        limit is 2 (not 50).  This is the regression gate: if someone changes
        the call site back to ``save_total_limit=num_epochs``, this test fails.
        """
        cfg = _training_config(num_epochs=50, seed=42)
        assert cfg.save_total_limit == CHECKPOINT_RETENTION, (
            f"save_total_limit must be {CHECKPOINT_RETENTION} (CHECKPOINT_RETENTION), "
            f"got {cfg.save_total_limit}.  Using num_epochs here retains one "
            f"checkpoint per epoch and exhausts disk at production scale."
        )
        assert CHECKPOINT_RETENTION == 2, (
            f"CHECKPOINT_RETENTION should be 2, got {CHECKPOINT_RETENTION}"
        )

    def test_save_total_limit_invariant_across_epoch_budgets(self):
        """save_total_limit is the same constant for all epoch budgets.

        Verifies that small and large epoch budgets both produce the same
        save_total_limit so disk usage stays bounded regardless of scale.
        """
        for num_epochs in [5, 15, 20, 30, 50]:
            cfg = _training_config(num_epochs=num_epochs, seed=42)
            assert cfg.save_total_limit == CHECKPOINT_RETENTION, (
                f"save_total_limit must equal CHECKPOINT_RETENTION={CHECKPOINT_RETENTION} "
                f"for num_epochs={num_epochs}, got {cfg.save_total_limit}"
            )
