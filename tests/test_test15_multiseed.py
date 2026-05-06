"""Unit tests for Test 15 multi-seed scaffolding.

Tests the pure-Python building blocks of
``experiments/test15_multiseed.py`` — phase builders, multi-seed
aggregator, marker logic, decay-steps math, checkpoint sort.  No GPU
required.

Phase builders were copied from ``experiments/test13_journal_scaffold.py``;
the parity test is a contract guard against accidental drift.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# experiments.test15_multiseed imports gpu_guard transitively via
# experiments.utils.gpu_guard.  CI machines without lab-tools skip the
# whole module rather than failing at collection.
pytest.importorskip("gpu_guard")

import experiments.test13_journal_scaffold as t13  # noqa: E402
import experiments.test15_multiseed as t15  # noqa: E402

# ---------------------------------------------------------------------------
# Phase builder parity with Test 13
# ---------------------------------------------------------------------------


def _make_qa_pool(n: int) -> list[dict]:
    """Build a deterministic synthetic qa_pool large enough for both tests."""
    return [{"question": f"q{i}", "answer": f"a{i}"} for i in range(n)]


def test_build_phase_A_parity():
    """Test 15's Phase A keyed list matches Test 13's exactly."""
    qa = _make_qa_pool(t15.TOTAL_KEYS + t15.SWAP_KEYS)
    assert t15.build_phase_A_keyed(qa) == t13.build_phase_A_keyed(qa)


def test_build_phase_B_swap_parity():
    qa = _make_qa_pool(t15.TOTAL_KEYS + t15.SWAP_KEYS)
    base = t15.build_phase_A_keyed(qa)
    swap_answers = qa[t15.TOTAL_KEYS : t15.TOTAL_KEYS + t15.SWAP_KEYS]
    assert t15.build_phase_B_swap_keyed(base, swap_answers) == t13.build_phase_B_swap_keyed(
        base, swap_answers
    )


def test_build_phase_C1_parity():
    qa = _make_qa_pool(t15.TOTAL_KEYS + t15.SWAP_KEYS)
    assert t15.build_phase_C1_keyed(qa) == t13.build_phase_C1_keyed(qa)


def test_build_phase_C2_parity():
    qa = _make_qa_pool(t15.TOTAL_KEYS + t15.SWAP_KEYS)
    c1 = t15.build_phase_C1_keyed(qa)
    assert t15.build_phase_C2_fill_keyed(c1, qa) == t13.build_phase_C2_fill_keyed(c1, qa)


# ---------------------------------------------------------------------------
# Phase builder structural sanity (independent of Test 13)
# ---------------------------------------------------------------------------


def test_phase_A_emits_total_keys():
    qa = _make_qa_pool(t15.TOTAL_KEYS + t15.SWAP_KEYS)
    keyed = t15.build_phase_A_keyed(qa)
    assert len(keyed) == t15.TOTAL_KEYS
    keys = [k["key"] for k in keyed]
    assert keys == [f"graph{i + 1}" for i in range(t15.TOTAL_KEYS)]


def test_phase_C1_has_TBD_in_swap_slots():
    qa = _make_qa_pool(t15.TOTAL_KEYS + t15.SWAP_KEYS)
    keyed = t15.build_phase_C1_keyed(qa)
    # First SWAP_START_SLOT entries: real answers.
    for i in range(t15.SWAP_START_SLOT):
        assert keyed[i]["answer"] == f"a{i}"
    # Last SWAP_KEYS entries: placeholder answers.
    for i in range(t15.SWAP_START_SLOT, t15.TOTAL_KEYS):
        slot_k = i - t15.SWAP_START_SLOT + 1
        assert keyed[i]["answer"] == f"TBD-{slot_k}"


def test_phase_C2_replaces_TBDs_with_real_answers():
    qa = _make_qa_pool(t15.TOTAL_KEYS + t15.SWAP_KEYS)
    c1 = t15.build_phase_C1_keyed(qa)
    fill = t15.build_phase_C2_fill_keyed(c1, qa)
    assert len(fill) == t15.SWAP_KEYS
    for i, kp in enumerate(fill):
        # Same key as C1's swap slot, real answer from qa_pool.
        assert kp["key"] == c1[t15.SWAP_START_SLOT + i]["key"]
        assert kp["answer"] == qa[t15.SWAP_START_SLOT + i]["answer"]
        assert "TBD-" not in kp["answer"]


def test_phase_B_collision_disambiguator():
    """If swap answer == base answer, append (variant)."""
    qa = _make_qa_pool(t15.TOTAL_KEYS + t15.SWAP_KEYS)
    base = t15.build_phase_A_keyed(qa)
    # Construct swap_answers that collide with base on the first swap slot.
    swap_answers = [{"answer": base[t15.SWAP_START_SLOT]["answer"]}] + [
        {"answer": f"swap_{i}"} for i in range(1, t15.SWAP_KEYS)
    ]
    swap_keyed = t15.build_phase_B_swap_keyed(base, swap_answers)
    assert swap_keyed[0]["answer"].endswith("(variant)")
    assert swap_keyed[1]["answer"] == "swap_1"


# ---------------------------------------------------------------------------
# Decay steps math
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n_keys,num_epochs,expected",
    [
        (100, 30, 1500),  # Phase A
        (20, 15, 150),  # Phase B
        (100, 14, 700),  # Phase C1
        (20, 14, 140),  # Phase C2
    ],
)
def test_decay_steps_for(n_keys, num_epochs, expected):
    assert t15._decay_steps_for(n_keys, num_epochs) == expected


# ---------------------------------------------------------------------------
# Marker logic
# ---------------------------------------------------------------------------


def test_marker_exists(tmp_path):
    seed_dir = tmp_path / "seed42"
    (seed_dir / "A").mkdir(parents=True)
    assert not t15.marker_exists(seed_dir, "A")
    (seed_dir / "A" / "A_done.json").write_text("{}")
    assert t15.marker_exists(seed_dir, "A")
    assert not t15.marker_exists(seed_dir, "B")


# ---------------------------------------------------------------------------
# Checkpoint numeric sort (CLAUDE.md: never lex-sort checkpoint-N)
# ---------------------------------------------------------------------------


def test_find_latest_checkpoint_numeric_sort(tmp_path):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    for step in (90, 100, 300, 50):
        (adapter_dir / f"checkpoint-{step}").mkdir()
    latest = t15._find_latest_checkpoint(adapter_dir)
    assert latest is not None
    assert latest.name == "checkpoint-300"


def test_find_latest_checkpoint_returns_none_when_empty(tmp_path):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    assert t15._find_latest_checkpoint(adapter_dir) is None


def test_find_latest_checkpoint_skips_non_checkpoint_dirs(tmp_path):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "checkpoint-10").mkdir()
    (adapter_dir / "model.safetensors").touch()
    (adapter_dir / "checkpoint-not-a-number").mkdir()
    latest = t15._find_latest_checkpoint(adapter_dir)
    assert latest is not None
    assert latest.name == "checkpoint-10"


# ---------------------------------------------------------------------------
# Multi-seed aggregator schema + ratio computation
# ---------------------------------------------------------------------------


def _write_seed_done(seed_dir: Path, retention_b: float, retention_c2: float, leaks: int = 0):
    for phase in ("A", "B", "C1", "C2"):
        (seed_dir / phase).mkdir(parents=True, exist_ok=True)
    (seed_dir / "B" / "B_done.json").write_text(
        json.dumps(
            {
                "retention_unchanged_80": {
                    "exact_count": int(retention_b * 80),
                    "total": 80,
                    "rate": retention_b,
                    "mean_confidence": 0.9,
                },
                "first_perfect_epoch": 5,
                "stable_perfect_epoch": 7,
            }
        )
    )
    (seed_dir / "C2" / "C2_done.json").write_text(
        json.dumps(
            {
                "retention_unchanged_80": {
                    "exact_count": int(retention_c2 * 80),
                    "total": 80,
                    "rate": retention_c2,
                    "mean_confidence": 0.9,
                },
                "first_perfect_epoch": 8,
                "stable_perfect_epoch": 10,
                "placeholder_leakage_on_fills": leaks,
            }
        )
    )


def test_write_multiseed_aggregate_basic(tmp_path):
    seeds = [42, 7, 1337]
    # B retention: 5%, 6%, 4% → mean 0.05
    # C2 retention: 35%, 40%, 30% → mean 0.35 → ratio 7.0
    pairs = [(0.05, 0.35), (0.06, 0.40), (0.04, 0.30)]
    for s, (rb, rc) in zip(seeds, pairs):
        _write_seed_done(tmp_path / f"seed{s}", rb, rc)

    t15._write_multiseed_aggregate(tmp_path, seeds)

    agg = json.loads((tmp_path / "multiseed_aggregate.json").read_text())
    assert agg["seeds"] == seeds
    assert agg["n_completed"] == 3
    assert len(agg["per_seed"]) == 3
    assert agg["mean_retention_B"] == pytest.approx(0.05, abs=1e-6)
    assert agg["mean_retention_C2"] == pytest.approx(0.35, abs=1e-6)
    assert agg["ratio_C2_over_B"] == pytest.approx(7.0, abs=1e-6)
    assert agg["decision_threshold_ratio"] == 5.0
    assert agg["leaks_total"] == 0


def test_write_multiseed_aggregate_skips_incomplete_seeds(tmp_path):
    seeds = [42, 7]
    _write_seed_done(tmp_path / "seed42", 0.05, 0.35)
    # seed7: only B done, no C2
    (tmp_path / "seed7" / "B").mkdir(parents=True)
    (tmp_path / "seed7" / "B" / "B_done.json").write_text("{}")
    t15._write_multiseed_aggregate(tmp_path, seeds)
    agg = json.loads((tmp_path / "multiseed_aggregate.json").read_text())
    assert agg["n_completed"] == 1
    assert [r["seed"] for r in agg["per_seed"]] == [42]


def test_write_multiseed_aggregate_handles_zero_b_retention(tmp_path):
    """Division-by-zero guard: ratio is None when mean_retention_B == 0."""
    seeds = [42]
    _write_seed_done(tmp_path / "seed42", 0.0, 0.30)
    t15._write_multiseed_aggregate(tmp_path, seeds)
    agg = json.loads((tmp_path / "multiseed_aggregate.json").read_text())
    assert agg["mean_retention_B"] == 0.0
    assert agg["ratio_C2_over_B"] is None


def test_write_multiseed_aggregate_no_seeds_writes_nothing(tmp_path, caplog):
    """Empty seed list / all incomplete: aggregate file is not written."""
    t15._write_multiseed_aggregate(tmp_path, [42, 7])  # no per-seed dirs
    assert not (tmp_path / "multiseed_aggregate.json").exists()


# ---------------------------------------------------------------------------
# Run config write-then-read parity
# ---------------------------------------------------------------------------


def _make_args(**overrides):
    import argparse

    defaults = {
        "model": "mistral",
        "seeds": [42, 7],
        "total_keys": 100,
        "swap_keys": 20,
        "phase_a_epochs": 30,
        "phase_b_epochs": 15,
        "phase_c1_epochs": 14,
        "phase_c2_epochs": 14,
        "rank": 8,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_load_or_write_run_config_first_launch(tmp_path):
    args = _make_args()
    cfg = t15.load_or_write_run_config(tmp_path, args)
    assert cfg["model"] == "mistral"
    assert cfg["seeds"] == [42, 7]
    assert cfg["weight_decay"] == 0.01  # deliberate Test 13 protocol deviation
    assert cfg["lr_scheduler_type"] == "linear"
    assert cfg["warmup_steps"] == 10
    assert (tmp_path / "run_config.json").exists()


def test_load_or_write_run_config_resume_reads_back(tmp_path):
    args1 = _make_args(seeds=[42], phase_a_epochs=30)
    t15.load_or_write_run_config(tmp_path, args1)
    # Second call with different args should read existing config, not overwrite.
    args2 = _make_args(seeds=[7], phase_a_epochs=99)
    cfg = t15.load_or_write_run_config(tmp_path, args2)
    assert cfg["seeds"] == [42]
    assert cfg["phase_a_epochs"] == 30


# ---------------------------------------------------------------------------
# Default seeds
# ---------------------------------------------------------------------------


def test_default_seeds_are_n5():
    """User directive 2026-05-06: n=5 from the start, no conditional escalation."""
    assert len(t15.DEFAULT_SEEDS) == 5
    assert t15.DEFAULT_SEEDS == [42, 7, 1337, 1, 11]
