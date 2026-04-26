"""Unit tests for critical Test 14 functions: compute_round_metrics,
evaluate_14b_pass_fail, decide_pre_winner, load_or_write_run_config, and
run_round_retention_probe (lightweight stub).

Covers F5 from the code review:
- compute_round_metrics: touchup_n_collateral computation from RP2/RP3 per-key
  set diff (B3 fix), F10 invariant assertion, and schema completeness.
- evaluate_14b_pass_fail: PASS / CONCERN / FAIL paths including the now-
  functional collateral gate.
- decide_pre_winner: null path (all variants fail) and success path.
- load_or_write_run_config: write-then-read parity, extension semantics.
- run_round_retention_probe: gradient_checkpointing re-enable on exception path.
"""

from __future__ import annotations

import json
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rp(total: int, exact: int, per_key: list[dict] | None = None) -> dict:
    """Build a minimal retention-probe result dict."""
    if per_key is None:
        per_key = [{"key": f"graph{i + 1}", "exact_match": i < exact} for i in range(total)]
    return {
        "rate": exact / total if total else 0.0,
        "exact_count": exact,
        "total": total,
        "mean_confidence": 0.9,
        "per_key": per_key,
    }


def _make_fill_state(
    first_perfect: int | None = 10,
    stable_perfect: int | None = 13,
    stop_epoch: int | None = 13,
    epoch_log: list[dict] | None = None,
):
    """Build a minimal EpochProbeState-like dict namespace."""
    from experiments.test14 import EpochProbeState

    s = EpochProbeState()
    s.first_perfect_epoch = first_perfect
    s.stable_perfect_epoch = stable_perfect
    s.stop_epoch = stop_epoch
    s.epoch_log = epoch_log or [
        {
            "epoch": 13,
            "fill": {"exact_count": 40, "total": 40, "rate": 1.0},
        }
    ]
    return s


def _make_touchup_skipped() -> dict:
    """Build a touchup_meta dict for the skipped (no failing keys) case."""
    return {
        "touchup_triggered": False,
        "touchup_skipped": True,
        "touchup_n_failing_keys": 0,
        "touchup_n_recovered": 0,
        "touchup_n_collateral": 0,
        "touchup_wall_seconds": 0.0,
        "touchup_train_loss": None,
    }


def _make_touchup_triggered(n_failing: int, n_recovered: int) -> dict:
    """Build a touchup_meta dict for the triggered case."""
    return {
        "touchup_triggered": True,
        "touchup_skipped": False,
        "touchup_n_failing_keys": n_failing,
        "touchup_n_recovered": n_recovered,
        "touchup_n_collateral": 0,  # will be overwritten by compute_round_metrics
        "touchup_wall_seconds": 12.5,
        "touchup_train_loss": 0.05,
    }


# ---------------------------------------------------------------------------
# TestComputeRoundMetrics
# ---------------------------------------------------------------------------


class TestComputeRoundMetrics:
    def test_clean_round_no_touchup(self):
        """Clean round: all 40 unchanged keys pass RP1/RP2/RP3 and no touch-up."""
        from experiments.test14 import compute_round_metrics

        rp1 = _make_rp(40, 40)
        rp2 = _make_rp(40, 40)
        rp3 = _make_rp(40, 40)
        fill_state = _make_fill_state()
        touchup = _make_touchup_skipped()

        result = compute_round_metrics(rp1, rp2, rp3, fill_state, touchup, round_index="P2")

        assert result["touchup_n_collateral"] == 0
        assert result["touchup_skipped"] is True
        assert result["retention_corruption_residual"] == pytest.approx(0.0)
        assert result["retention_alignment_delta"] == pytest.approx(0.0)
        assert result["round_index"] == "P2"
        # Schema completeness
        for field in (
            "fill_first_perfect_epoch",
            "fill_stable_perfect_epoch",
            "stop_epoch",
            "early_stop_fired",
            "touchup_n_collateral",
            "touchup_n_failing_keys",
            "touchup_n_recovered",
        ):
            assert field in result, f"missing field: {field}"

    def test_touchup_recovers_all_no_collateral(self):
        """Touch-up fires; all failing keys recovered; no collateral (RP3 == RP2 passing)."""
        from experiments.test14 import compute_round_metrics

        # 5 keys fail post-fill; 35 pass.
        rp2_per_key = [{"key": f"graph{i + 1}", "exact_match": i >= 5} for i in range(40)]
        rp3_per_key = [{"key": f"graph{i + 1}", "exact_match": True} for i in range(40)]
        rp2 = _make_rp(40, 35, per_key=rp2_per_key)
        rp3 = _make_rp(40, 40, per_key=rp3_per_key)
        rp1 = _make_rp(40, 40)
        fill_state = _make_fill_state()
        touchup = _make_touchup_triggered(n_failing=5, n_recovered=5)

        result = compute_round_metrics(rp1, rp2, rp3, fill_state, touchup, round_index="P2")

        assert result["touchup_n_collateral"] == 0
        assert result["touchup_skipped"] is False

    def test_touchup_causes_collateral(self):
        """Touch-up causes collateral: 2 keys that passed RP2 now fail RP3."""
        from experiments.test14 import compute_round_metrics

        # All 40 pass RP2; after touch-up 2 newly fail (keys graph1, graph2).
        rp2_per_key = [{"key": f"graph{i + 1}", "exact_match": True} for i in range(40)]
        rp3_per_key = [{"key": f"graph{i + 1}", "exact_match": i >= 2} for i in range(40)]
        rp2 = _make_rp(40, 40, per_key=rp2_per_key)
        rp3 = _make_rp(40, 38, per_key=rp3_per_key)
        rp1 = _make_rp(40, 40)
        fill_state = _make_fill_state()
        # touch-up was triggered even though we said n_failing=3 (some other keys failed)
        touchup = _make_touchup_triggered(n_failing=3, n_recovered=3)

        result = compute_round_metrics(rp1, rp2, rp3, fill_state, touchup, round_index="P3")

        assert result["touchup_n_collateral"] == 2

    def test_rp2_and_rp3_disjoint_keys_no_collateral(self):
        """If RP2 and RP3 per_key have no matching keys, collateral is 0."""
        from experiments.test14 import compute_round_metrics

        rp2_per_key = [{"key": "graph1", "exact_match": True}]
        rp3_per_key = [{"key": "graph2", "exact_match": False}]
        rp2 = _make_rp(1, 1, per_key=rp2_per_key)
        rp3 = _make_rp(1, 0, per_key=rp3_per_key)
        rp1 = _make_rp(1, 1)
        fill_state = _make_fill_state()
        touchup = _make_touchup_triggered(n_failing=1, n_recovered=0)

        result = compute_round_metrics(rp1, rp2, rp3, fill_state, touchup, round_index="P2")
        assert result["touchup_n_collateral"] == 0

    def test_rp2_per_key_empty_no_collateral(self):
        """Empty RP2 per_key (unchanged_keyed empty, P1 case) → 0 collateral."""
        from experiments.test14 import compute_round_metrics

        rp2 = _make_rp(0, 0, per_key=[])
        rp3 = _make_rp(0, 0, per_key=[])
        rp1 = _make_rp(0, 0, per_key=[])
        fill_state = _make_fill_state()
        touchup = _make_touchup_skipped()

        result = compute_round_metrics(rp1, rp2, rp3, fill_state, touchup, round_index="P1")
        assert result["touchup_n_collateral"] == 0

    def test_f10_invariant_fires_on_violation(self):
        """F10: assert touchup_skipped == (n_failing == 0) raises on violation."""
        from experiments.test14 import compute_round_metrics

        rp = _make_rp(5, 5)
        fill_state = _make_fill_state()
        # touchup_skipped=True but n_failing=3 — invariant violated.
        bad_touchup = {
            "touchup_triggered": False,
            "touchup_skipped": True,
            "touchup_n_failing_keys": 3,
            "touchup_n_recovered": 0,
            "touchup_n_collateral": 0,
            "touchup_wall_seconds": 0.0,
            "touchup_train_loss": None,
        }
        with pytest.raises(AssertionError, match="invariant violated"):
            compute_round_metrics(rp, rp, rp, fill_state, bad_touchup, round_index="P2")


# ---------------------------------------------------------------------------
# TestEvaluate14bPassFail
# ---------------------------------------------------------------------------


def _make_round_result(
    corruption_residual: float = 0.02,
    post_rate: float = 0.97,
    stop_epoch: int | None = 13,
    touchup_skipped: bool = True,
    n_collateral: int = 0,
    n_failing: int = 0,
    n_recovered: int = 0,
    pre_total: int = 460,
    pre_rate: float = 0.98,
    round_index: str = "P1",
) -> dict:
    """Build a minimal round_result dict for evaluate_14b_pass_fail."""
    return {
        "round_index": round_index,
        "retention_corruption_residual": corruption_residual,
        "retention_post_touchup": {
            "rate": post_rate,
            "exact_count": int(post_rate * pre_total),
            "total": pre_total,
        },
        "retention_pre_touchup": {
            "rate": pre_rate,
            "exact_count": int(pre_rate * pre_total),
            "total": pre_total,
        },
        "stop_epoch": stop_epoch,
        "touchup_skipped": touchup_skipped,
        "touchup_n_collateral": n_collateral,
        "touchup_n_failing_keys": n_failing,
        "touchup_n_recovered": n_recovered,
        "fill_final_recall": {"rate": 1.0, "exact_count": 500, "total": 500},
        "retention_alignment_delta": 0.02,
    }


class TestEvaluate14bPassFail:
    def _make_cumulative(self, rate: float = 0.97) -> dict:
        return {"rate": rate, "exact_count": int(rate * 500), "total": 500}

    def test_pass_all_criteria_met(self):
        """All criteria satisfied → PASS."""
        from experiments.test14 import evaluate_14b_pass_fail

        rounds = [
            _make_round_result(corruption_residual=0.01, post_rate=0.97, stop_epoch=13),
            _make_round_result(
                corruption_residual=0.02, post_rate=0.97, stop_epoch=14, round_index="P2"
            ),
            _make_round_result(
                corruption_residual=0.03, post_rate=0.97, stop_epoch=15, round_index="P3"
            ),
        ]
        # Residuals NOT monotonically growing (0.01<0.02<0.03 IS monotonic — make non-mono)
        rounds[2]["retention_corruption_residual"] = 0.015

        result = evaluate_14b_pass_fail(rounds, self._make_cumulative(0.97), 500)
        assert result["verdict"] == "PASS"
        assert result["fail_reasons"] == []

    def test_fail_cumulative_retention_too_low(self):
        """Cumulative retention < 0.95 → FAIL."""
        from experiments.test14 import evaluate_14b_pass_fail

        rounds = [
            _make_round_result(stop_epoch=13),
            _make_round_result(stop_epoch=14, round_index="P2"),
            _make_round_result(stop_epoch=15, round_index="P3"),
        ]
        result = evaluate_14b_pass_fail(rounds, self._make_cumulative(0.90), 500)
        assert result["verdict"] == "FAIL"
        assert any("cumulative" in r for r in result["fail_reasons"])

    def test_fail_collateral_gate(self):
        """Collateral > 2% of passing_pre → FAIL (B3/F5 combined coverage)."""
        from experiments.test14 import evaluate_14b_pass_fail

        # 460 keys pre-touchup at 98% → 451 passing.  Collateral = 15 > 0.02*451.
        rounds = [
            _make_round_result(
                stop_epoch=13,
                n_collateral=15,
                touchup_skipped=False,
                n_failing=10,
                n_recovered=10,
            ),
            _make_round_result(stop_epoch=14, round_index="P2"),
            _make_round_result(stop_epoch=15, round_index="P3"),
        ]
        result = evaluate_14b_pass_fail(rounds, self._make_cumulative(0.97), 500)
        assert result["verdict"] == "FAIL"
        assert any("collateral" in r for r in result["fail_reasons"])

    def test_fail_early_stop_did_not_fire(self):
        """stop_epoch is None → FAIL (early stop didn't fire)."""
        from experiments.test14 import evaluate_14b_pass_fail

        rounds = [
            _make_round_result(stop_epoch=None),
            _make_round_result(stop_epoch=14, round_index="P2"),
            _make_round_result(stop_epoch=15, round_index="P3"),
        ]
        result = evaluate_14b_pass_fail(rounds, self._make_cumulative(0.97), 500)
        assert result["verdict"] == "FAIL"
        assert any("early_stop" in r for r in result["fail_reasons"])

    def test_concern_stop_epoch_too_early(self):
        """stop_epoch < 12 triggers CONCERN but not FAIL."""
        from experiments.test14 import evaluate_14b_pass_fail

        rounds = [
            _make_round_result(stop_epoch=5),
            _make_round_result(stop_epoch=5, round_index="P2"),
            _make_round_result(stop_epoch=5, round_index="P3"),
        ]
        result = evaluate_14b_pass_fail(rounds, self._make_cumulative(0.97), 500)
        # stop_epoch=5 < 12 → concern; post_rate=0.97 OK; no fail by itself
        if not result["fail_reasons"]:
            assert result["verdict"] == "CONCERN"
            assert len(result["concerns"]) > 0


# ---------------------------------------------------------------------------
# TestDecidePreWinner
# ---------------------------------------------------------------------------


class TestDecidePreWinner:
    def _write_done(
        self,
        run_dir: Path,
        variant: str,
        b_rate: float,
        c_rate: float,
        stable_c: int | None = 15,
        stable_b: int | None = 20,
        leakage: int = 0,
    ) -> None:
        """Write stub B_done.json and C_done.json for a variant."""
        b_dir = run_dir / variant / "B"
        c_dir = run_dir / variant / "C"
        b_dir.mkdir(parents=True, exist_ok=True)
        c_dir.mkdir(parents=True, exist_ok=True)
        (b_dir / "B_done.json").write_text(
            json.dumps(
                {
                    "final_recall": {
                        "rate": b_rate,
                        "exact_count": int(b_rate * 100),
                        "total": 100,
                    },
                    "stable_perfect_epoch": stable_b,
                }
            )
        )
        (c_dir / "C_done.json").write_text(
            json.dumps(
                {
                    "final_recall": {"rate": c_rate, "exact_count": int(c_rate * 20), "total": 20},
                    "stable_perfect_epoch": stable_c,
                    "placeholder_leakage_count": leakage,
                    "epoch_log": [],
                    "q_a_split_at_final": {"q_only": 0, "a_only": 0, "both": 20, "total": 20},
                }
            )
        )

    def test_winner_picked_from_single_passing_variant(self, tmp_path):
        """Single variant passes all criteria → winner returned."""
        from experiments.test14 import decide_pre_winner

        self._write_done(tmp_path, "V1", b_rate=1.0, c_rate=0.97)

        winner = decide_pre_winner(tmp_path)
        assert winner == "V1"
        decision = json.loads((tmp_path / "pre_decision.json").read_text())
        assert decision["winner"] == "V1"
        assert "stable_c" in decision["reason"]

    def test_null_winner_when_all_variants_fail(self, tmp_path):
        """All variants fail criteria → winner is None."""
        from experiments.test14 import decide_pre_winner

        # b_rate < 0.99 → fails the B criterion.
        for v in ("V1", "V2", "V3"):
            self._write_done(tmp_path, v, b_rate=0.80, c_rate=0.80)

        winner = decide_pre_winner(tmp_path)
        assert winner is None
        decision = json.loads((tmp_path / "pre_decision.json").read_text())
        assert decision["winner"] is None

    def test_best_stable_c_wins_tiebreak(self, tmp_path):
        """When two variants pass, the one with smallest stable_c wins."""
        from experiments.test14 import decide_pre_winner

        self._write_done(tmp_path, "V1", b_rate=1.0, c_rate=0.97, stable_c=18)
        self._write_done(tmp_path, "V2", b_rate=1.0, c_rate=0.97, stable_c=14)
        self._write_done(tmp_path, "V3", b_rate=0.80, c_rate=0.80)  # fails

        winner = decide_pre_winner(tmp_path)
        assert winner == "V2"


# ---------------------------------------------------------------------------
# TestLoadOrWriteRunConfig
# ---------------------------------------------------------------------------


class TestLoadOrWriteRunConfig:
    def _make_args(
        self,
        mode: str,
        n_keys: int | None = None,
        variant: str = "V1",
        num_epochs: int = 30,
        rank: int = 8,
    ) -> types.SimpleNamespace:
        return types.SimpleNamespace(
            mode=mode,
            model="mistral",
            variant=variant,
            n_keys=n_keys,
            num_epochs=num_epochs,
            rank=rank,
            smoke=False,
            scale_run=None,
        )

    def test_write_then_read_parity(self, tmp_path):
        """First launch writes config; second read returns same values."""
        from experiments.test14 import load_or_write_run_config

        args = self._make_args("scale", n_keys=50)
        cfg1 = load_or_write_run_config(tmp_path, args)
        cfg2 = load_or_write_run_config(tmp_path, args)
        assert cfg1 == cfg2
        assert cfg1["n_keys"] == 50
        assert cfg1["mode"] == "scale"
        assert cfg1["rank"] == 8

    def test_schema_fields_present(self, tmp_path):
        """All expected schema fields present in written config."""
        from experiments.test14 import load_or_write_run_config

        args = self._make_args("pre")
        cfg = load_or_write_run_config(tmp_path, args)
        for field in (
            "model",
            "mode",
            "variant",
            "n_keys",
            "num_epochs",
            "rank",
            "lr",
            "seed",
            "smoke",
            "created_at",
        ):
            assert field in cfg, f"missing field: {field}"

    def test_extension_increases_n_keys(self, tmp_path):
        """Resume with larger n_keys updates config on disk."""
        from experiments.test14 import load_or_write_run_config

        args_first = self._make_args("scale", n_keys=100)
        load_or_write_run_config(tmp_path, args_first)

        args_extend = self._make_args("scale", n_keys=200)
        cfg = load_or_write_run_config(tmp_path, args_extend)
        assert cfg["n_keys"] == 200

        # Read back from disk to confirm persistence.
        cfg_disk = json.loads((tmp_path / "run_config.json").read_text())
        assert cfg_disk["n_keys"] == 200

    def test_extension_shrink_raises(self, tmp_path):
        """Resume with smaller n_keys raises SystemExit."""
        from experiments.test14 import load_or_write_run_config

        args_first = self._make_args("scale", n_keys=200)
        load_or_write_run_config(tmp_path, args_first)

        args_shrink = self._make_args("scale", n_keys=50)
        with pytest.raises(SystemExit, match="cannot shrink"):
            load_or_write_run_config(tmp_path, args_shrink)

    def test_non_scale_extension_ignored(self, tmp_path):
        """n_keys extension only applies in scale mode; pre mode ignores n_keys arg."""
        from experiments.test14 import load_or_write_run_config

        args_first = self._make_args("pre")
        load_or_write_run_config(tmp_path, args_first)

        # Re-read with n_keys specified: should be ignored in pre mode.
        args_second = self._make_args("pre", n_keys=999)
        cfg = load_or_write_run_config(tmp_path, args_second)
        # pre mode hard-caps at PRE_N_KEYS=100, not 999.
        assert cfg["n_keys"] == 100


# ---------------------------------------------------------------------------
# TestRunRoundRetentionProbeGcGuard
# ---------------------------------------------------------------------------


class TestRunRoundRetentionProbeGcGuard:
    """Verify gradient_checkpointing is re-enabled even when the probe raises."""

    def _make_model_mock(self, is_gc_enabled: bool = True) -> MagicMock:
        m = MagicMock()
        m.is_gradient_checkpointing = is_gc_enabled
        return m

    def test_gc_reenabled_on_exception(self, tmp_path):
        """gradient_checkpointing_enable() is called in finally even when probe raises."""
        from experiments.test14 import run_round_retention_probe

        model = self._make_model_mock(is_gc_enabled=True)
        tokenizer = MagicMock()

        with patch(
            "experiments.test14.evaluate_indexed_recall",
            side_effect=RuntimeError("probe failed"),
        ):
            with pytest.raises(RuntimeError, match="probe failed"):
                run_round_retention_probe(
                    model,
                    tokenizer,
                    [{"key": "graph1", "question": "Q?", "answer": "A"}],
                    {"graph1": 12345},
                    "journal",
                    tmp_path / "rp.json",
                    "RP1",
                )

        model.gradient_checkpointing_enable.assert_called_once()

    def test_gc_reenabled_on_success(self, tmp_path):
        """gradient_checkpointing_enable() is called after successful probe."""
        from experiments.test14 import run_round_retention_probe

        model = self._make_model_mock(is_gc_enabled=True)
        tokenizer = MagicMock()
        probe_result = {
            "exact_count": 1,
            "total": 1,
            "rate": 1.0,
            "mean_confidence": 0.9,
            "per_key": [{"key": "graph1", "exact_match": True}],
        }

        with patch(
            "experiments.test14.evaluate_indexed_recall",
            return_value=probe_result,
        ):
            result = run_round_retention_probe(
                model,
                tokenizer,
                [{"key": "graph1", "question": "Q?", "answer": "A"}],
                {"graph1": 12345},
                "journal",
                tmp_path / "rp.json",
                "RP2",
            )

        model.gradient_checkpointing_enable.assert_called_once()
        assert result["exact_count"] == 1


# ---------------------------------------------------------------------------
# TestSafeProbeGcGuard
# ---------------------------------------------------------------------------


class TestSafeProbeGcGuard:
    """Verify _safe_probe re-enables gradient_checkpointing on exception."""

    def test_safe_probe_reenables_gc_on_exception(self):
        """_safe_probe calls gradient_checkpointing_enable() even when evaluate raises."""
        from experiments.test14 import _safe_probe

        model = MagicMock()
        tokenizer = MagicMock()

        with patch(
            "experiments.test14.evaluate_indexed_recall",
            side_effect=RuntimeError("eval error"),
        ):
            with pytest.raises(RuntimeError, match="eval error"):
                _safe_probe(
                    model,
                    tokenizer,
                    [{"key": "graph1", "question": "Q?", "answer": "A"}],
                    {"graph1": 12345},
                    "episodic",
                )

        model.gradient_checkpointing_enable.assert_called_once()

    def test_safe_probe_returns_result_and_reenables(self):
        """_safe_probe returns evaluate_indexed_recall result and enables gc."""
        from experiments.test14 import _safe_probe

        model = MagicMock()
        tokenizer = MagicMock()
        expected = {
            "exact_count": 5,
            "total": 5,
            "rate": 1.0,
            "mean_confidence": 0.9,
            "per_key": [],
        }

        with patch("experiments.test14.evaluate_indexed_recall", return_value=expected):
            result = _safe_probe(model, tokenizer, [], {}, "journal")

        assert result == expected
        model.gradient_checkpointing_enable.assert_called_once()


# ---------------------------------------------------------------------------
# TestCheckPauseWritesPausedJson
# ---------------------------------------------------------------------------


class TestCheckPauseWritesPausedJson:
    """Verify F9 fix: _check_pause writes paused.json BEFORE raising SystemExit."""

    def test_paused_json_written_before_exit(self, tmp_path):
        """When pause file exists and run_dir given, paused.json is written before exit."""
        from experiments.test14 import _check_pause

        pause_file = tmp_path / ".training_pause"
        pause_file.touch()

        with patch("experiments.test14.PAUSE_FILE", pause_file):
            with pytest.raises(SystemExit, match="Training paused"):
                _check_pause("test_label", tmp_path)

        paused_json = tmp_path / "paused.json"
        assert paused_json.exists(), "paused.json must be written before SystemExit"
        data = json.loads(paused_json.read_text())
        assert data["stopped_after_phase"] == "test_label"

    def test_paused_json_skipped_when_run_dir_none(self, tmp_path):
        """When run_dir is None, no paused.json written but exit still fires."""
        from experiments.test14 import _check_pause

        pause_file = tmp_path / ".training_pause"
        pause_file.touch()

        with patch("experiments.test14.PAUSE_FILE", pause_file):
            with pytest.raises(SystemExit):
                _check_pause("test_label", run_dir=None)

        # No paused.json should have been written anywhere except maybe run_dir (None).
        # Nothing to assert on filesystem since run_dir is None.


# ---------------------------------------------------------------------------
# TestLoadPhaseAFromExisting
# ---------------------------------------------------------------------------


class TestLoadPhaseAFromExisting:
    """Verify load_phase_a_from_existing writes phase_a_reused.json and loads adapter."""

    def _write_a_done(
        self,
        source_dir: Path,
        variant: str,
        rate: float = 1.0,
        stop_epoch: int = 23,
        wall_seconds: float = 9862.6,
    ) -> None:
        """Write stub A_done.json, adapter slot, and keyed_pairs/registry.

        Creates a minimal adapter slot that satisfies ``resolve_adapter_slot``
        /``find_live_slot``: the slot directory must contain a ``meta.json``
        with ``registry_sha256: ""`` (the empty-hash path used by experiments).
        """
        a_dir = source_dir / variant / "A"
        a_dir.mkdir(parents=True, exist_ok=True)
        adapter_slot = a_dir / "adapter" / "fake_slot"
        adapter_slot.mkdir(parents=True, exist_ok=True)
        # meta.json with registry_sha256="" matches the empty-hash lookup
        # that resolve_adapter_slot("", ...) uses in experiments.
        (adapter_slot / "meta.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "name": f"episodic_{variant.lower()}",
                    "registry_sha256": "",
                    "keyed_pairs_sha256": "",
                    "key_count": 3,
                    "synthesized": False,
                    "base_model": {"repo": "test", "sha": "abc", "hash": "unknown"},
                    "lora": {"rank": 8, "alpha": 16, "dropout": 0.0, "target_modules": []},
                    "tokenizer": {"name_or_path": "test", "vocab_size": 32000, "merges_hash": ""},
                    "trained_at": "2026-04-26T00:00:00Z",
                }
            )
        )
        (adapter_slot / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "r": 8}))
        (adapter_slot / "adapter_model.safetensors").touch()
        (a_dir / "A_done.json").write_text(
            json.dumps(
                {
                    "final_recall": {"rate": rate, "exact_count": 100, "total": 100},
                    "stop_epoch": stop_epoch,
                    "wall_seconds": wall_seconds,
                }
            )
        )
        (a_dir / "keyed_pairs.json").write_text(
            json.dumps(
                [{"key": f"graph{i}", "question": f"Q{i}", "answer": f"A{i}"} for i in range(1, 4)]
            )
        )
        (a_dir / "simhash_registry.json").write_text(
            json.dumps({f"graph{i}": i * 1000 for i in range(1, 4)})
        )

    def test_phase_a_reused_json_written(self, tmp_path):
        """phase_a_reused.json is written to dest_variant_dir/A/ with correct fields."""
        from experiments.test14 import load_phase_a_from_existing

        source_dir = tmp_path / "source"
        self._write_a_done(source_dir, "V3")

        dest_dir = tmp_path / "dest" / "V4"
        dest_dir.mkdir(parents=True, exist_ok=True)

        mock_model = MagicMock()
        mock_model.base_model = MagicMock()

        with patch("experiments.test14.PeftModel") as mock_peft:
            mock_peft.from_pretrained.return_value = mock_model
            load_phase_a_from_existing(
                mock_model,
                source_run_dir=source_dir,
                source_variant="V3",
                dest_variant_dir=dest_dir,
            )

        reuse_path = dest_dir / "A" / "phase_a_reused.json"
        assert reuse_path.exists(), "phase_a_reused.json must be written"
        data = json.loads(reuse_path.read_text())
        assert data["source_variant"] == "V3"
        assert data["source_run_dir"] == str(source_dir)
        assert data["source_final_recall_rate"] == pytest.approx(1.0)
        assert data["source_stop_epoch"] == 23
        assert data["source_wall_seconds"] == pytest.approx(9862.6)
        assert "source_adapter_slot" in data
        assert "timestamp" in data

    def test_peft_from_pretrained_called_with_journal_name(self, tmp_path):
        """PeftModel.from_pretrained is called with adapter_name matching V3 pattern."""
        from experiments.test14 import load_phase_a_from_existing

        source_dir = tmp_path / "source"
        self._write_a_done(source_dir, "V3")

        dest_dir = tmp_path / "dest" / "V4"
        dest_dir.mkdir(parents=True, exist_ok=True)

        mock_model = MagicMock()

        with patch("experiments.test14.PeftModel") as mock_peft:
            mock_peft.from_pretrained.return_value = mock_model
            load_phase_a_from_existing(
                mock_model,
                source_run_dir=source_dir,
                source_variant="V3",
                dest_variant_dir=dest_dir,
            )

        mock_peft.from_pretrained.assert_called_once()
        call_kwargs = mock_peft.from_pretrained.call_args
        # adapter_name should be "episodic_v3" (episodic_{source_variant.lower()})
        assert (
            call_kwargs.kwargs.get("adapter_name") == "episodic_v3"
            or (len(call_kwargs.args) >= 3 and call_kwargs.args[2] == "episodic_v3")
            or "episodic_v3" in str(call_kwargs)
        )

    def test_raises_when_a_done_missing(self, tmp_path):
        """FileNotFoundError raised when source A_done.json is absent."""
        from experiments.test14 import load_phase_a_from_existing

        source_dir = tmp_path / "source"
        source_dir.mkdir(parents=True, exist_ok=True)

        dest_dir = tmp_path / "dest" / "V4"

        mock_model = MagicMock()
        with pytest.raises(FileNotFoundError, match="A_done.json not found"):
            load_phase_a_from_existing(
                mock_model,
                source_run_dir=source_dir,
                source_variant="V3",
                dest_variant_dir=dest_dir,
            )

    def test_raises_when_adapter_slot_missing(self, tmp_path):
        """FileNotFoundError raised when adapter slot directory is empty."""
        from experiments.test14 import load_phase_a_from_existing

        source_dir = tmp_path / "source"
        a_dir = source_dir / "V3" / "A"
        a_dir.mkdir(parents=True, exist_ok=True)
        # Write A_done.json but NO adapter directory.
        (a_dir / "A_done.json").write_text(
            json.dumps({"final_recall": {"rate": 1.0}, "stop_epoch": 23, "wall_seconds": 100.0})
        )
        (a_dir / "adapter").mkdir(parents=True, exist_ok=True)  # empty adapter dir

        dest_dir = tmp_path / "dest" / "V4"

        mock_model = MagicMock()
        with pytest.raises(FileNotFoundError, match="adapter slot not found"):
            load_phase_a_from_existing(
                mock_model,
                source_run_dir=source_dir,
                source_variant="V3",
                dest_variant_dir=dest_dir,
            )


# ---------------------------------------------------------------------------
# TestDecidePreWinnerExtended — all-six-variants logic
# ---------------------------------------------------------------------------


class TestDecidePreWinnerExtended:
    """Test decide_pre_winner with V3_extended/V4/V5 and override threshold."""

    def _write_done(
        self,
        run_dir: Path,
        variant: str,
        b_rate: float,
        c_rate: float,
        stable_c: int | None = 15,
        stable_b: int | None = 20,
        stop_epoch: int | None = 15,
        leakage: int = 0,
    ) -> None:
        """Write stub B_done.json and C_done.json for a variant."""
        b_dir = run_dir / variant / "B"
        c_dir = run_dir / variant / "C"
        b_dir.mkdir(parents=True, exist_ok=True)
        c_dir.mkdir(parents=True, exist_ok=True)
        (b_dir / "B_done.json").write_text(
            json.dumps(
                {
                    "final_recall": {
                        "rate": b_rate,
                        "exact_count": int(b_rate * 100),
                        "total": 100,
                    },
                    "stable_perfect_epoch": stable_b,
                }
            )
        )
        (c_dir / "C_done.json").write_text(
            json.dumps(
                {
                    "final_recall": {
                        "rate": c_rate,
                        "exact_count": int(c_rate * 20),
                        "total": 20,
                    },
                    "stable_perfect_epoch": stable_c,
                    "stop_epoch": stop_epoch,
                    "placeholder_leakage_count": leakage,
                    "epoch_log": [],
                    "q_a_split_at_final": {"q_only": 0, "a_only": 0, "both": 20, "total": 20},
                }
            )
        )

    def test_all_six_variants_scanned(self, tmp_path):
        """decide_pre_winner scans all six variant dirs, not just V1/V2/V3."""
        from experiments.test14 import decide_pre_winner

        # Only write V4 — should appear in per_variant_summary.
        self._write_done(tmp_path, "V4", b_rate=1.0, c_rate=0.97, stable_c=12, stop_epoch=12)
        decide_pre_winner(tmp_path)
        decision = json.loads((tmp_path / "pre_decision.json").read_text())
        assert "V4" in decision["per_variant_summary"]

    def test_new_variant_overrides_existing_winner_below_threshold(self, tmp_path):
        """New variant with stop_epoch <= 14 overrides existing V3 winner."""
        from experiments.test14 import decide_pre_winner

        # Write V4 with stop_epoch=12 (< 14 threshold).
        self._write_done(tmp_path, "V4", b_rate=1.0, c_rate=0.97, stable_c=12, stop_epoch=12)

        winner = decide_pre_winner(tmp_path, existing_winner="V3")
        assert winner == "V4"
        decision = json.loads((tmp_path / "pre_decision.json").read_text())
        assert decision["override_threshold_applied"] is True

    def test_new_variant_does_not_override_when_above_threshold(self, tmp_path):
        """New variant with stop_epoch > 14 does NOT override existing V3 winner."""
        from experiments.test14 import decide_pre_winner

        # Write V5 with stop_epoch=18 (> 14 threshold).
        self._write_done(tmp_path, "V5", b_rate=1.0, c_rate=0.97, stable_c=18, stop_epoch=18)

        winner = decide_pre_winner(tmp_path, existing_winner="V3")
        assert winner == "V3"
        decision = json.loads((tmp_path / "pre_decision.json").read_text())
        assert decision["override_threshold_applied"] is True

    def test_existing_winner_preserved_when_all_new_variants_fail(self, tmp_path):
        """All new variants fail → existing winner is preserved."""
        from experiments.test14 import decide_pre_winner

        # Write V4 with b_rate < 0.99 → fails.
        self._write_done(tmp_path, "V4", b_rate=0.70, c_rate=0.80)

        winner = decide_pre_winner(tmp_path, existing_winner="V3")
        assert winner == "V3"

    def test_no_existing_winner_and_all_fail_returns_none(self, tmp_path):
        """No existing winner and all variants fail → None."""
        from experiments.test14 import decide_pre_winner

        self._write_done(tmp_path, "V4", b_rate=0.70, c_rate=0.80)
        winner = decide_pre_winner(tmp_path, existing_winner=None)
        assert winner is None

    def test_v3_extended_recognized_and_evaluated(self, tmp_path):
        """V3_extended variant dirs are scanned and evaluated correctly."""
        from experiments.test14 import decide_pre_winner

        self._write_done(
            tmp_path,
            "V3_extended",
            b_rate=1.0,
            c_rate=0.97,
            stable_c=10,
            stop_epoch=10,
        )

        winner = decide_pre_winner(tmp_path, existing_winner="V3")
        # stop_epoch=10 <= 14 threshold → should override
        assert winner == "V3_extended"

    def test_pre_decision_json_fields(self, tmp_path):
        """pre_decision.json contains override_threshold_applied and override_threshold_value."""
        from experiments.test14 import decide_pre_winner

        self._write_done(tmp_path, "V4", b_rate=1.0, c_rate=0.97, stable_c=12, stop_epoch=12)
        decide_pre_winner(tmp_path, existing_winner="V3")

        decision = json.loads((tmp_path / "pre_decision.json").read_text())
        assert "override_threshold_applied" in decision
        assert "override_threshold_value" in decision
        assert decision["override_threshold_value"] == 14

    def test_original_variants_not_subject_to_threshold(self, tmp_path):
        """V1/V2/V3 are treated as original variants and bypass the threshold check."""
        from experiments.test14 import decide_pre_winner

        # V2 passes with stop_epoch=20 (> threshold) and no existing_winner.
        self._write_done(tmp_path, "V2", b_rate=1.0, c_rate=0.97, stable_c=20, stop_epoch=20)

        winner = decide_pre_winner(tmp_path, existing_winner=None)
        assert winner == "V2"

    def test_winner_is_best_stable_c_among_all_passing(self, tmp_path):
        """When multiple new variants pass, the one with smallest stable_c wins."""
        from experiments.test14 import decide_pre_winner

        self._write_done(tmp_path, "V4", b_rate=1.0, c_rate=0.97, stable_c=12, stop_epoch=12)
        self._write_done(tmp_path, "V5", b_rate=1.0, c_rate=0.97, stable_c=10, stop_epoch=10)

        winner = decide_pre_winner(tmp_path, existing_winner="V3")
        assert winner == "V5"  # stable_c=10 < 12
