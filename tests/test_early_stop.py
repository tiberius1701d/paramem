"""Unit tests for experiments/utils/early_stop.py.

Covers:
- EarlyStopPolicy validation invariants.
- ANALYSIS_POLICY singleton has expected defaults.
- _EarlyStopState default construction.
- RecallEarlyStopCallback state machine:
  - _last_epoch guard prevents double-fire for the same epoch.
  - probe_from_epoch gate: no probe before floor epoch.
  - probe_every_n_epochs cadence.
  - Aggregate 100% recall required (not per-key streak).
  - 3-consecutive-perfect window triggers stop.
  - signal_from_epoch gate: no stop trigger before floor epoch even with 3 perfect.
  - Stop triggered at correct epoch when all conditions met.
  - stop not fired twice (_signaled_stop guard).
  - first_perfect_epoch and stable_perfect_epoch tracking.
  - Pause file triggers stop.
  - gradient_checkpointing_enable() called after each probe epoch.
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


def _make_recall_result(exact_count: int, total: int) -> dict:
    """Build a minimal evaluate_indexed_recall-style result dict."""
    return {
        "exact_count": exact_count,
        "total": total,
        "rate": exact_count / total if total > 0 else 0.0,
        "mean_confidence": 0.9,
        "per_key": [
            {
                "key": f"graph{i + 1}",
                "exact_match": i < exact_count,
                "question_match": i < exact_count,
                "answer_match": i < exact_count,
                "recalled": {"question": f"Q{i + 1}", "answer": f"A{i + 1}"},
                "confidence": 0.9,
            }
            for i in range(total)
        ],
    }


def _make_state(epoch: float) -> MagicMock:
    s = MagicMock()
    s.epoch = epoch
    return s


def _make_control() -> MagicMock:
    c = MagicMock()
    c.should_training_stop = False
    return c


def _make_model() -> MagicMock:
    m = MagicMock()
    m.is_gradient_checkpointing = True
    return m


def _train_begin_state(global_step: int) -> types.SimpleNamespace:
    """Lightweight stand-in for HF TrainerState passed to on_train_begin."""
    return types.SimpleNamespace(global_step=global_step)


# ---------------------------------------------------------------------------
# EarlyStopPolicy
# ---------------------------------------------------------------------------


class TestEarlyStopPolicy:
    def test_default_construction(self):
        from experiments.utils.early_stop import EarlyStopPolicy

        p = EarlyStopPolicy()
        assert p.probe_from_epoch == 10
        assert p.signal_from_epoch == 10
        assert p.window == 3
        assert p.probe_every_n_epochs == 1

    def test_custom_values(self):
        from experiments.utils.early_stop import EarlyStopPolicy

        p = EarlyStopPolicy(
            probe_from_epoch=1, signal_from_epoch=5, window=2, probe_every_n_epochs=2
        )
        assert p.probe_from_epoch == 1
        assert p.signal_from_epoch == 5
        assert p.window == 2
        assert p.probe_every_n_epochs == 2

    def test_signal_before_probe_raises(self):
        from experiments.utils.early_stop import EarlyStopPolicy

        with pytest.raises(ValueError, match="signal_from_epoch"):
            EarlyStopPolicy(probe_from_epoch=10, signal_from_epoch=5)

    def test_signal_equal_probe_ok(self):
        from experiments.utils.early_stop import EarlyStopPolicy

        p = EarlyStopPolicy(probe_from_epoch=5, signal_from_epoch=5)
        assert p.signal_from_epoch == p.probe_from_epoch

    def test_window_zero_raises(self):
        from experiments.utils.early_stop import EarlyStopPolicy

        with pytest.raises(ValueError, match="window"):
            EarlyStopPolicy(window=0)

    def test_window_negative_raises(self):
        from experiments.utils.early_stop import EarlyStopPolicy

        with pytest.raises(ValueError, match="window"):
            EarlyStopPolicy(window=-1)

    def test_probe_every_zero_raises(self):
        from experiments.utils.early_stop import EarlyStopPolicy

        with pytest.raises(ValueError, match="probe_every_n_epochs"):
            EarlyStopPolicy(probe_every_n_epochs=0)

    def test_probe_every_negative_raises(self):
        from experiments.utils.early_stop import EarlyStopPolicy

        with pytest.raises(ValueError, match="probe_every_n_epochs"):
            EarlyStopPolicy(probe_every_n_epochs=-1)

    def test_window_one_is_valid(self):
        from experiments.utils.early_stop import EarlyStopPolicy

        p = EarlyStopPolicy(window=1)
        assert p.window == 1

    def test_probe_every_one_is_valid(self):
        from experiments.utils.early_stop import EarlyStopPolicy

        p = EarlyStopPolicy(probe_every_n_epochs=1)
        assert p.probe_every_n_epochs == 1


class TestAnalysisPolicy:
    def test_singleton_values(self):
        from experiments.utils.early_stop import ANALYSIS_POLICY

        assert ANALYSIS_POLICY.probe_from_epoch == 1
        assert ANALYSIS_POLICY.signal_from_epoch == 10
        assert ANALYSIS_POLICY.window == 3
        assert ANALYSIS_POLICY.probe_every_n_epochs == 1

    def test_analysis_policy_kwargs_matches_singleton(self):
        from experiments.utils.early_stop import (
            ANALYSIS_POLICY,
            ANALYSIS_POLICY_KWARGS,
            EarlyStopPolicy,
        )

        p = EarlyStopPolicy(**ANALYSIS_POLICY_KWARGS)
        assert p.probe_from_epoch == ANALYSIS_POLICY.probe_from_epoch
        assert p.signal_from_epoch == ANALYSIS_POLICY.signal_from_epoch
        assert p.window == ANALYSIS_POLICY.window
        assert p.probe_every_n_epochs == ANALYSIS_POLICY.probe_every_n_epochs


class TestEarlyStopState:
    def test_default_construction(self):
        from experiments.utils.early_stop import _EarlyStopState

        s = _EarlyStopState()
        assert s.first_perfect_epoch is None
        assert s.stable_perfect_epoch is None
        assert s.stop_epoch is None
        assert s.epoch_log == []

    def test_epoch_log_is_independent_per_instance(self):
        from experiments.utils.early_stop import _EarlyStopState

        s1 = _EarlyStopState()
        s2 = _EarlyStopState()
        s1.epoch_log.append({"epoch": 1})
        assert s2.epoch_log == []


# ---------------------------------------------------------------------------
# RecallEarlyStopCallback
# ---------------------------------------------------------------------------


def _make_callback(
    tmp_path: Path,
    *,
    total: int = 5,
    policy_kwargs: dict | None = None,
    with_retention: bool = False,
):
    """Build a RecallEarlyStopCallback with all collaborators mocked."""
    from experiments.utils.early_stop import (
        EarlyStopPolicy,
        RecallEarlyStopCallback,
        _EarlyStopState,
    )

    if policy_kwargs is None:
        # Fast policy: probe from epoch 1, signal from epoch 3, window 2
        policy_kwargs = {
            "probe_from_epoch": 1,
            "signal_from_epoch": 3,
            "window": 2,
            "probe_every_n_epochs": 1,
        }

    policy = EarlyStopPolicy(**policy_kwargs)
    state_out = _EarlyStopState()
    model = _make_model()

    target_keyed = [
        {"key": f"graph{i + 1}", "question": f"Q{i + 1}", "answer": f"A{i + 1}"}
        for i in range(total)
    ]
    target_registry = {f"graph{i + 1}": i for i in range(total)}

    kwargs = dict(
        model=model,
        tokenizer=MagicMock(),
        target_keyed=target_keyed,
        target_registry=target_registry,
        adapter_name="journal",
        policy=policy,
        state_out=state_out,
        progress_path=tmp_path / "progress.json",
        epoch_log_path=tmp_path / "epoch_log.json",
        first_perfect_log_path=tmp_path / "first_perfect_log.json",
        phase_name="Phase_C",
        num_epochs=30,
        pause_file=tmp_path / "pause_flag",
    )
    if with_retention:
        kwargs["retention_keyed"] = [{"key": "graphR1", "question": "RQ1", "answer": "RA1"}]
        kwargs["retention_registry"] = {"graphR1": 99}

    cb = RecallEarlyStopCallback(**kwargs)
    return cb, state_out, model


class TestRecallEarlyStopCallbackStateachine:
    def test_last_epoch_guard_prevents_double_fire(self, tmp_path):
        """Calling on_epoch_end twice with same epoch number probes only once."""
        cb, state_out, model = _make_callback(tmp_path)
        perfect = _make_recall_result(5, 5)

        # Direct test using inner state
        with (
            patch("paramem.training.recall_eval.evaluate_indexed_recall", return_value=perfect),
            patch(
                "experiments.utils.recall_diagnostics.per_field_split_counts",
                return_value={
                    "both": 5,
                    "q_only": 0,
                    "a_only": 0,
                    "neither": 0,
                    "total": 5,
                    "q_correct": 5,
                    "a_correct": 5,
                },
            ),
            patch("experiments.utils.recall_diagnostics.update_first_perfect_log"),
        ):
            # Simulate two calls at epoch 2
            control = _make_control()
            state = _make_state(2.0)
            cb.on_epoch_end(None, state, control)
            probe_count_after_first = len(state_out.epoch_log)

            cb.on_epoch_end(None, state, control)
            probe_count_after_second = len(state_out.epoch_log)

            assert probe_count_after_second == probe_count_after_first, (
                "Second call at same epoch should not add another log entry"
            )

    def test_probe_from_epoch_gate(self, tmp_path):
        """No probe before probe_from_epoch."""
        from experiments.utils.early_stop import (
            EarlyStopPolicy,
            RecallEarlyStopCallback,
            _EarlyStopState,
        )

        policy = EarlyStopPolicy(probe_from_epoch=5, signal_from_epoch=5, window=1)
        state_out = _EarlyStopState()
        target_keyed = [{"key": "graph1", "question": "Q", "answer": "A"}]
        cb = RecallEarlyStopCallback(
            model=_make_model(),
            tokenizer=MagicMock(),
            target_keyed=target_keyed,
            target_registry={"graph1": 1},
            adapter_name="journal",
            policy=policy,
            state_out=state_out,
            progress_path=tmp_path / "p.json",
            epoch_log_path=tmp_path / "e.json",
            first_perfect_log_path=tmp_path / "f.json",
            phase_name="Phase_C",
            num_epochs=20,
            pause_file=tmp_path / "pause",
        )

        control = _make_control()
        with patch("paramem.training.recall_eval.evaluate_indexed_recall") as mock_eval:
            cb.on_epoch_end(None, _make_state(2.0), control)
            cb.on_epoch_end(None, _make_state(3.0), control)
            cb.on_epoch_end(None, _make_state(4.0), control)
            # evaluate_indexed_recall should NOT have been called before epoch 5
            mock_eval.assert_not_called()

    def test_stop_requires_signal_from_epoch(self, tmp_path):
        """Stop not fired before signal_from_epoch even with consecutive perfect probes."""
        from experiments.utils.early_stop import (
            EarlyStopPolicy,
            RecallEarlyStopCallback,
            _EarlyStopState,
        )

        # probe from epoch 1, signal from epoch 10, window 2
        policy = EarlyStopPolicy(probe_from_epoch=1, signal_from_epoch=10, window=2)
        state_out = _EarlyStopState()
        total = 3
        target_keyed = [
            {"key": f"graph{i + 1}", "question": "Q", "answer": "A"} for i in range(total)
        ]
        cb = RecallEarlyStopCallback(
            model=_make_model(),
            tokenizer=MagicMock(),
            target_keyed=target_keyed,
            target_registry={f"graph{i + 1}": i for i in range(total)},
            adapter_name="journal",
            policy=policy,
            state_out=state_out,
            progress_path=tmp_path / "p.json",
            epoch_log_path=tmp_path / "e.json",
            first_perfect_log_path=tmp_path / "f.json",
            phase_name="Phase_C",
            num_epochs=20,
            pause_file=tmp_path / "pause",
        )

        perfect = _make_recall_result(total, total)
        with (
            patch("paramem.training.recall_eval.evaluate_indexed_recall", return_value=perfect),
            patch(
                "experiments.utils.recall_diagnostics.per_field_split_counts",
                return_value={
                    "both": total,
                    "q_only": 0,
                    "a_only": 0,
                    "neither": 0,
                    "total": total,
                    "q_correct": total,
                    "a_correct": total,
                },
            ),
            patch("experiments.utils.recall_diagnostics.update_first_perfect_log"),
        ):
            # Epochs 1-5 are all perfect but before signal_from_epoch=10
            for ep in range(1, 6):
                control = _make_control()
                cb.on_epoch_end(None, _make_state(float(ep)), control)
                assert not control.should_training_stop, (
                    f"Stop should not fire at epoch {ep} (signal_from_epoch=10)"
                )

    def test_three_consecutive_perfect_triggers_stop(self, tmp_path):
        """Stop fires after window consecutive 100% probes at or after signal_from_epoch."""
        cb, state_out, model = _make_callback(
            tmp_path,
            total=3,
            policy_kwargs={"probe_from_epoch": 1, "signal_from_epoch": 3, "window": 3},
        )
        perfect = _make_recall_result(3, 3)

        with (
            patch("paramem.training.recall_eval.evaluate_indexed_recall", return_value=perfect),
            patch(
                "experiments.utils.recall_diagnostics.per_field_split_counts",
                return_value={
                    "both": 3,
                    "q_only": 0,
                    "a_only": 0,
                    "neither": 0,
                    "total": 3,
                    "q_correct": 3,
                    "a_correct": 3,
                },
            ),
            patch("experiments.utils.recall_diagnostics.update_first_perfect_log"),
        ):
            # Epochs 3, 4, 5 — all perfect, signal_from_epoch=3 reached at ep=3
            controls = {}
            for ep in [3, 4, 5]:
                c = _make_control()
                cb.on_epoch_end(None, _make_state(float(ep)), c)
                controls[ep] = c

            # Stop should NOT fire at epoch 3 (only 1 consecutive) or 4 (only 2),
            # but MUST fire at epoch 5 (3rd consecutive).
            assert not controls[3].should_training_stop
            assert not controls[4].should_training_stop
            assert controls[5].should_training_stop
            assert state_out.stop_epoch == 5

    def test_reset_consecutive_on_imperfect(self, tmp_path):
        """Consecutive counter resets on any imperfect probe.

        Policy: signal_from_epoch=20 (far away), window=3.
        Sequence: perfect, perfect, imperfect, perfect, perfect, perfect.
        Expectation: stop does NOT fire at epoch 2 (only 2 consecutive),
        resets at epoch 3 (imperfect), fires at epoch 6 (3rd consecutive after reset).
        """
        cb, state_out, model = _make_callback(
            tmp_path,
            total=3,
            policy_kwargs={"probe_from_epoch": 1, "signal_from_epoch": 20, "window": 3},
        )
        perfect = _make_recall_result(3, 3)
        imperfect = _make_recall_result(2, 3)

        # Epoch sequence → recall results
        # ep1=perfect, ep2=perfect, ep3=imperfect, ep4=perfect, ep5=perfect, ep6=perfect
        epoch_results = [perfect, perfect, imperfect, perfect, perfect, perfect]
        call_count = [0]

        def side_effect(*args, **kwargs):
            idx = call_count[0]
            call_count[0] += 1
            return epoch_results[idx] if idx < len(epoch_results) else perfect

        # per_field_split_counts needs to reflect the actual exact_count from the result.
        # Use a dynamic side_effect that reads from the last returned recall result.
        split_call = [0]
        split_results = [
            {
                "both": 3,
                "q_only": 0,
                "a_only": 0,
                "neither": 0,
                "total": 3,
                "q_correct": 3,
                "a_correct": 3,
            },
            {
                "both": 3,
                "q_only": 0,
                "a_only": 0,
                "neither": 0,
                "total": 3,
                "q_correct": 3,
                "a_correct": 3,
            },
            {
                "both": 2,
                "q_only": 0,
                "a_only": 0,
                "neither": 0,
                "total": 3,
                "q_correct": 2,
                "a_correct": 2,
            },
            {
                "both": 3,
                "q_only": 0,
                "a_only": 0,
                "neither": 0,
                "total": 3,
                "q_correct": 3,
                "a_correct": 3,
            },
            {
                "both": 3,
                "q_only": 0,
                "a_only": 0,
                "neither": 0,
                "total": 3,
                "q_correct": 3,
                "a_correct": 3,
            },
            {
                "both": 3,
                "q_only": 0,
                "a_only": 0,
                "neither": 0,
                "total": 3,
                "q_correct": 3,
                "a_correct": 3,
            },
        ]

        def split_side_effect(*args, **kwargs):
            idx = split_call[0]
            split_call[0] += 1
            return split_results[idx] if idx < len(split_results) else split_results[-1]

        with (
            patch("paramem.training.recall_eval.evaluate_indexed_recall", side_effect=side_effect),
            patch(
                "experiments.utils.recall_diagnostics.per_field_split_counts",
                side_effect=split_side_effect,
            ),
            patch("experiments.utils.recall_diagnostics.update_first_perfect_log"),
        ):
            controls = {}
            for ep in [1, 2, 3, 4, 5, 6]:
                c = _make_control()
                cb.on_epoch_end(None, _make_state(float(ep)), c)
                controls[ep] = c

        # After imperfect at ep3, counter reset.
        # Perfect at ep4, ep5, ep6 → 3 consecutive, fires at ep6.
        # signal_from_epoch=20, so stop does NOT fire even at ep6.
        # What we verify: no stop fires (signal_from_epoch=20 > any test epoch)
        # and that state_out.stop_epoch is None.
        # The real behavioural invariant being tested: the imperfect at ep3 resets
        # consecutive so that ep4 and ep5 alone are only 2 in a row (not 4 in a row
        # from ep1), and ep6 is the 3rd in a row.
        assert state_out.stop_epoch is None  # signal_from_epoch=20 not reached
        # stable_perfect_epoch must be set at ep6 (3rd consecutive since ep4)
        assert state_out.stable_perfect_epoch == 6

    def test_stop_not_fired_twice(self, tmp_path):
        """Once stop is signalled, subsequent perfect probes don't re-set it."""
        cb, state_out, model = _make_callback(
            tmp_path,
            total=2,
            policy_kwargs={"probe_from_epoch": 1, "signal_from_epoch": 1, "window": 1},
        )
        perfect = _make_recall_result(2, 2)

        controls = []
        with (
            patch("paramem.training.recall_eval.evaluate_indexed_recall", return_value=perfect),
            patch(
                "experiments.utils.recall_diagnostics.per_field_split_counts",
                return_value={
                    "both": 2,
                    "q_only": 0,
                    "a_only": 0,
                    "neither": 0,
                    "total": 2,
                    "q_correct": 2,
                    "a_correct": 2,
                },
            ),
            patch("experiments.utils.recall_diagnostics.update_first_perfect_log"),
        ):
            for ep in [1, 2, 3]:
                c = _make_control()
                cb.on_epoch_end(None, _make_state(float(ep)), c)
                controls.append(c)

        # stop_epoch should be 1 (window=1, signal_from_epoch=1)
        assert state_out.stop_epoch == 1
        # The _signaled_stop guard means we only set it once.
        # control.should_training_stop was set at epoch 1; subsequent controls
        # depend on HF Trainer propagation — but our callback doesn't re-set it.
        # Check state_out.stop_epoch doesn't change.
        assert state_out.stop_epoch == 1

    def test_first_perfect_epoch_tracking(self, tmp_path):
        cb, state_out, model = _make_callback(
            tmp_path,
            total=2,
            policy_kwargs={"probe_from_epoch": 1, "signal_from_epoch": 10, "window": 3},
        )
        imperfect = _make_recall_result(1, 2)
        perfect = _make_recall_result(2, 2)

        results_by_epoch = {1: imperfect, 2: imperfect, 3: perfect, 4: perfect}
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            ep_seq = [1, 2, 3, 4]
            return results_by_epoch.get(ep_seq[call_count[0] - 1], perfect)

        with (
            patch("paramem.training.recall_eval.evaluate_indexed_recall", side_effect=side_effect),
            patch(
                "experiments.utils.recall_diagnostics.per_field_split_counts",
                return_value={
                    "both": 2,
                    "q_only": 0,
                    "a_only": 0,
                    "neither": 0,
                    "total": 2,
                    "q_correct": 2,
                    "a_correct": 2,
                },
            ),
            patch("experiments.utils.recall_diagnostics.update_first_perfect_log"),
        ):
            for ep in [1, 2, 3, 4]:
                cb.on_epoch_end(None, _make_state(float(ep)), _make_control())

        # first_perfect_epoch set at epoch 3 (first time exact_count == total)
        assert state_out.first_perfect_epoch == 3

    def test_stable_perfect_epoch_tracking(self, tmp_path):
        cb, state_out, model = _make_callback(
            tmp_path,
            total=2,
            policy_kwargs={"probe_from_epoch": 1, "signal_from_epoch": 20, "window": 3},
        )
        perfect = _make_recall_result(2, 2)

        with (
            patch("paramem.training.recall_eval.evaluate_indexed_recall", return_value=perfect),
            patch(
                "experiments.utils.recall_diagnostics.per_field_split_counts",
                return_value={
                    "both": 2,
                    "q_only": 0,
                    "a_only": 0,
                    "neither": 0,
                    "total": 2,
                    "q_correct": 2,
                    "a_correct": 2,
                },
            ),
            patch("experiments.utils.recall_diagnostics.update_first_perfect_log"),
        ):
            for ep in [1, 2, 3, 4]:
                cb.on_epoch_end(None, _make_state(float(ep)), _make_control())

        # stable_perfect_epoch is set at epoch 3 (window=3 → 3rd consecutive perfect)
        assert state_out.stable_perfect_epoch == 3

    def test_epoch_log_accumulates(self, tmp_path):
        cb, state_out, model = _make_callback(tmp_path, total=2)
        perfect = _make_recall_result(2, 2)

        with (
            patch("paramem.training.recall_eval.evaluate_indexed_recall", return_value=perfect),
            patch(
                "experiments.utils.recall_diagnostics.per_field_split_counts",
                return_value={
                    "both": 2,
                    "q_only": 0,
                    "a_only": 0,
                    "neither": 0,
                    "total": 2,
                    "q_correct": 2,
                    "a_correct": 2,
                },
            ),
            patch("experiments.utils.recall_diagnostics.update_first_perfect_log"),
        ):
            for ep in [1, 2, 3]:
                cb.on_epoch_end(None, _make_state(float(ep)), _make_control())

        assert len(state_out.epoch_log) == 3
        assert state_out.epoch_log[0]["epoch"] == 1
        assert state_out.epoch_log[2]["epoch"] == 3

    def test_probe_every_n_epochs_cadence(self, tmp_path):
        """With probe_every_n_epochs=3, only epochs 1, 4, 7, ... are probed."""
        from experiments.utils.early_stop import (
            EarlyStopPolicy,
            RecallEarlyStopCallback,
            _EarlyStopState,
        )

        policy = EarlyStopPolicy(
            probe_from_epoch=1, signal_from_epoch=20, window=1, probe_every_n_epochs=3
        )
        state_out = _EarlyStopState()
        target_keyed = [{"key": "graph1", "question": "Q", "answer": "A"}]
        cb = RecallEarlyStopCallback(
            model=_make_model(),
            tokenizer=MagicMock(),
            target_keyed=target_keyed,
            target_registry={"graph1": 1},
            adapter_name="journal",
            policy=policy,
            state_out=state_out,
            progress_path=tmp_path / "p.json",
            epoch_log_path=tmp_path / "e.json",
            first_perfect_log_path=tmp_path / "f.json",
            phase_name="Phase_C",
            num_epochs=20,
            pause_file=tmp_path / "pause",
        )

        perfect = _make_recall_result(1, 1)
        with (
            patch(
                "paramem.training.recall_eval.evaluate_indexed_recall", return_value=perfect
            ) as mock_eval,
            patch(
                "experiments.utils.recall_diagnostics.per_field_split_counts",
                return_value={
                    "both": 1,
                    "q_only": 0,
                    "a_only": 0,
                    "neither": 0,
                    "total": 1,
                    "q_correct": 1,
                    "a_correct": 1,
                },
            ),
            patch("experiments.utils.recall_diagnostics.update_first_perfect_log"),
        ):
            for ep in range(1, 8):
                cb.on_epoch_end(None, _make_state(float(ep)), _make_control())

            # probe_from_epoch=1, every 3 → probed at epochs 1, 4, 7
            assert mock_eval.call_count == 3

    def test_gradient_checkpointing_reenabled_after_probe_when_args_says_on(self, tmp_path):
        """When args.gradient_checkpointing=True, the probe re-enables it.

        evaluate_indexed_recall disables checkpointing during generation; the
        callback must restore the pre-probe state.  Restoration is gated on
        ``args.gradient_checkpointing`` so a trainer constructed with
        checkpointing OFF doesn't get it silently flipped back on.
        """
        cb, state_out, model = _make_callback(tmp_path, total=2)
        perfect = _make_recall_result(2, 2)
        args = MagicMock()
        args.gradient_checkpointing = True

        with (
            patch("paramem.training.recall_eval.evaluate_indexed_recall", return_value=perfect),
            patch(
                "experiments.utils.recall_diagnostics.per_field_split_counts",
                return_value={
                    "both": 2,
                    "q_only": 0,
                    "a_only": 0,
                    "neither": 0,
                    "total": 2,
                    "q_correct": 2,
                    "a_correct": 2,
                },
            ),
            patch("experiments.utils.recall_diagnostics.update_first_perfect_log"),
        ):
            cb.on_epoch_end(args, _make_state(1.0), _make_control())

        model.gradient_checkpointing_enable.assert_called_once()

    def test_gradient_checkpointing_NOT_reenabled_when_args_says_off(self, tmp_path):
        """When args.gradient_checkpointing=False, the probe must NOT re-enable it.

        Catches the regression class where silently turning checkpointing back on
        after the recall probe breaks the next epoch's ``model.generate`` KV-cache
        contract (HF disables KV cache when gradient checkpointing is active).
        Production today defaults to True so this branch is latent, but any debug
        or VRAM-tuning configuration that flips it False must stay flipped.
        """
        cb, state_out, model = _make_callback(tmp_path, total=2)
        perfect = _make_recall_result(2, 2)
        args = MagicMock()
        args.gradient_checkpointing = False

        with (
            patch("paramem.training.recall_eval.evaluate_indexed_recall", return_value=perfect),
            patch(
                "experiments.utils.recall_diagnostics.per_field_split_counts",
                return_value={
                    "both": 2,
                    "q_only": 0,
                    "a_only": 0,
                    "neither": 0,
                    "total": 2,
                    "q_correct": 2,
                    "a_correct": 2,
                },
            ),
            patch("experiments.utils.recall_diagnostics.update_first_perfect_log"),
        ):
            cb.on_epoch_end(args, _make_state(1.0), _make_control())

        model.gradient_checkpointing_enable.assert_not_called()

    def test_gradient_checkpointing_not_called_on_non_probe_epoch(self, tmp_path):
        """gradient_checkpointing_enable() must NOT be called for non-probe epochs."""
        from experiments.utils.early_stop import (
            EarlyStopPolicy,
            RecallEarlyStopCallback,
            _EarlyStopState,
        )

        policy = EarlyStopPolicy(probe_from_epoch=5, signal_from_epoch=5, window=1)
        state_out = _EarlyStopState()
        model = _make_model()
        target_keyed = [{"key": "graph1", "question": "Q", "answer": "A"}]
        cb = RecallEarlyStopCallback(
            model=model,
            tokenizer=MagicMock(),
            target_keyed=target_keyed,
            target_registry={"graph1": 1},
            adapter_name="journal",
            policy=policy,
            state_out=state_out,
            progress_path=None,
            epoch_log_path=tmp_path / "e.json",
            first_perfect_log_path=tmp_path / "f.json",
            phase_name="Phase_C",
            num_epochs=20,
            pause_file=tmp_path / "pause",
        )

        with patch("paramem.training.recall_eval.evaluate_indexed_recall") as mock_eval:
            cb.on_epoch_end(None, _make_state(2.0), _make_control())
            mock_eval.assert_not_called()
            model.gradient_checkpointing_enable.assert_not_called()

    def test_pause_file_triggers_stop(self, tmp_path):
        """Pause file existence causes control.should_training_stop to be set."""
        cb, state_out, model = _make_callback(tmp_path, total=2)
        pause_file = tmp_path / "pause_flag"
        pause_file.touch()

        perfect = _make_recall_result(2, 2)
        control = _make_control()

        with (
            patch("paramem.training.recall_eval.evaluate_indexed_recall", return_value=perfect),
            patch(
                "experiments.utils.recall_diagnostics.per_field_split_counts",
                return_value={
                    "both": 2,
                    "q_only": 0,
                    "a_only": 0,
                    "neither": 0,
                    "total": 2,
                    "q_correct": 2,
                    "a_correct": 2,
                },
            ),
            patch("experiments.utils.recall_diagnostics.update_first_perfect_log"),
        ):
            cb.on_epoch_end(None, _make_state(1.0), control)

        assert control.should_training_stop is True

    def test_pause_file_triggers_stop_on_non_probe_epoch(self, tmp_path):
        """Pause file stops training even on non-probe epochs."""
        from experiments.utils.early_stop import (
            EarlyStopPolicy,
            RecallEarlyStopCallback,
            _EarlyStopState,
        )

        policy = EarlyStopPolicy(probe_from_epoch=10, signal_from_epoch=10, window=1)
        state_out = _EarlyStopState()
        pause_file = tmp_path / "pause_flag"
        pause_file.touch()
        cb = RecallEarlyStopCallback(
            model=_make_model(),
            tokenizer=MagicMock(),
            target_keyed=[{"key": "graph1", "question": "Q", "answer": "A"}],
            target_registry={"graph1": 1},
            adapter_name="journal",
            policy=policy,
            state_out=state_out,
            progress_path=None,
            epoch_log_path=tmp_path / "e.json",
            first_perfect_log_path=tmp_path / "f.json",
            phase_name="Phase_C",
            num_epochs=20,
            pause_file=pause_file,
        )

        control = _make_control()
        cb.on_epoch_end(None, _make_state(3.0), control)  # epoch 3 < probe_from_epoch=10
        assert control.should_training_stop is True

    def test_epoch_log_written_to_disk(self, tmp_path):
        """epoch_log.json must be written to disk after each probe epoch."""
        cb, state_out, model = _make_callback(tmp_path, total=2)
        perfect = _make_recall_result(2, 2)

        with (
            patch("paramem.training.recall_eval.evaluate_indexed_recall", return_value=perfect),
            patch(
                "experiments.utils.recall_diagnostics.per_field_split_counts",
                return_value={
                    "both": 2,
                    "q_only": 0,
                    "a_only": 0,
                    "neither": 0,
                    "total": 2,
                    "q_correct": 2,
                    "a_correct": 2,
                },
            ),
            patch("experiments.utils.recall_diagnostics.update_first_perfect_log"),
        ):
            cb.on_epoch_end(None, _make_state(1.0), _make_control())

        epoch_log_path = tmp_path / "epoch_log.json"
        assert epoch_log_path.exists()
        data = json.loads(epoch_log_path.read_text())
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["epoch"] == 1

    def test_retention_probe_included_in_epoch_log(self, tmp_path):
        """When retention_keyed is provided, retention data appears in epoch_log."""
        cb, state_out, model = _make_callback(tmp_path, total=2, with_retention=True)
        perfect_fill = _make_recall_result(2, 2)
        perfect_ret = _make_recall_result(1, 1)

        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            return perfect_fill if call_count[0] == 1 else perfect_ret

        with (
            patch("paramem.training.recall_eval.evaluate_indexed_recall", side_effect=side_effect),
            patch(
                "experiments.utils.recall_diagnostics.per_field_split_counts",
                return_value={
                    "both": 2,
                    "q_only": 0,
                    "a_only": 0,
                    "neither": 0,
                    "total": 2,
                    "q_correct": 2,
                    "a_correct": 2,
                },
            ),
            patch("experiments.utils.recall_diagnostics.update_first_perfect_log"),
        ):
            cb.on_epoch_end(None, _make_state(1.0), _make_control())

        assert "retention" in state_out.epoch_log[0]

    def test_get_first_perfect_log(self, tmp_path):
        """get_first_perfect_log returns internal dict (may be empty or populated)."""
        cb, state_out, model = _make_callback(tmp_path, total=2)
        log = cb.get_first_perfect_log()
        assert isinstance(log, dict)


class TestRecallEarlyStopCallbackResume:
    """Rehydration of in-memory state from on-disk artifacts via on_train_begin.

    Rehydration is deferred from __init__ to on_train_begin so the fresh-run
    vs checkpoint-resume distinction can be made via ``state.global_step``:
    fresh run (global_step==0) discards stale artifacts; genuine resume
    (global_step>0) carries accumulators forward.

    Each test constructs the callback, then calls
    ``cb.on_train_begin(args, state, control)`` with the appropriate
    global_step before asserting state.
    """

    def test_cycle_started_at_preserved_across_resume(self, tmp_path):
        """progress.json's cycle_started_at survives a checkpoint resume."""
        progress = tmp_path / "progress.json"
        progress.write_text(json.dumps({"cycle_started_at": 1234567890}))
        cb, _, _ = _make_callback(tmp_path)
        cb.on_train_begin(types.SimpleNamespace(), _train_begin_state(5), None)
        assert cb._cycle_started_at == 1234567890

    def test_cycle_started_at_fresh_on_cold_start(self, tmp_path):
        """No progress.json and global_step==0 = stamp time.time() (cold start)."""
        cb, _, _ = _make_callback(tmp_path)
        cb.on_train_begin(types.SimpleNamespace(), _train_begin_state(0), None)
        # Stamped at init time, so it must be a recent positive integer.
        assert cb._cycle_started_at > 0

    def test_epoch_log_rehydrated(self, tmp_path):
        """epoch_log.json on disk is loaded into state.epoch_log on resume."""
        epoch_log = tmp_path / "epoch_log.json"
        prior = [
            {"epoch": 1, "fill": {"exact_count": 0, "total": 5, "rate": 0.0}},
            {"epoch": 2, "fill": {"exact_count": 3, "total": 5, "rate": 0.6}},
        ]
        epoch_log.write_text(json.dumps(prior))
        cb, state_out, _ = _make_callback(tmp_path)
        cb.on_train_begin(types.SimpleNamespace(), _train_begin_state(5), None)
        assert state_out.epoch_log == prior
        assert cb._last_epoch == 2

    def test_first_perfect_rehydrated_from_log(self, tmp_path):
        """first_perfect_epoch reconstructed from prior epoch_log on resume."""
        epoch_log = tmp_path / "epoch_log.json"
        prior = [
            {"epoch": 1, "fill": {"exact_count": 4, "total": 5, "rate": 0.8}},
            {"epoch": 2, "fill": {"exact_count": 5, "total": 5, "rate": 1.0}},
            {"epoch": 3, "fill": {"exact_count": 5, "total": 5, "rate": 1.0}},
        ]
        epoch_log.write_text(json.dumps(prior))
        cb, state_out, _ = _make_callback(tmp_path)
        cb.on_train_begin(types.SimpleNamespace(), _train_begin_state(5), None)
        assert state_out.first_perfect_epoch == 2

    def test_stable_perfect_rehydrated_from_log(self, tmp_path):
        """stable_perfect_epoch fires when window of perfect epochs is met on resume."""
        # window=2 in the default policy_kwargs of _make_callback
        epoch_log = tmp_path / "epoch_log.json"
        prior = [
            {"epoch": 1, "fill": {"exact_count": 4, "total": 5, "rate": 0.8}},
            {"epoch": 2, "fill": {"exact_count": 5, "total": 5, "rate": 1.0}},
            {"epoch": 3, "fill": {"exact_count": 5, "total": 5, "rate": 1.0}},
        ]
        epoch_log.write_text(json.dumps(prior))
        cb, state_out, _ = _make_callback(tmp_path)
        cb.on_train_begin(types.SimpleNamespace(), _train_begin_state(5), None)
        assert state_out.stable_perfect_epoch == 3
        assert cb._consecutive_perfect == 2

    def test_consecutive_perfect_resets_on_non_perfect(self, tmp_path):
        """A non-perfect epoch in the log resets the trailing perfect streak on resume."""
        epoch_log = tmp_path / "epoch_log.json"
        prior = [
            {"epoch": 1, "fill": {"exact_count": 5, "total": 5, "rate": 1.0}},
            {"epoch": 2, "fill": {"exact_count": 5, "total": 5, "rate": 1.0}},
            {"epoch": 3, "fill": {"exact_count": 4, "total": 5, "rate": 0.8}},
        ]
        epoch_log.write_text(json.dumps(prior))
        cb, state_out, _ = _make_callback(tmp_path)
        cb.on_train_begin(types.SimpleNamespace(), _train_begin_state(5), None)
        assert state_out.first_perfect_epoch == 1
        # window=2 means stable triggered at e2 (e1+e2 both perfect).
        assert state_out.stable_perfect_epoch == 2
        # Trailing streak after e3 (non-perfect) is 0.
        assert cb._consecutive_perfect == 0

    def test_first_perfect_log_rehydrated(self, tmp_path):
        """Per-key first_perfect_log.json rehydrates the in-memory dict on resume."""
        first_perfect_log = tmp_path / "first_perfect_log.json"
        prior = {"graph1": {"epoch_first_perfect": 5}}
        first_perfect_log.write_text(json.dumps(prior))
        cb, _, _ = _make_callback(tmp_path)
        cb.on_train_begin(types.SimpleNamespace(), _train_begin_state(5), None)
        assert cb.get_first_perfect_log() == prior

    def test_malformed_json_treated_as_cold_start(self, tmp_path):
        """A corrupt epoch_log.json on resume path doesn't crash on_train_begin."""
        (tmp_path / "epoch_log.json").write_text("{not json")
        (tmp_path / "progress.json").write_text("not json either")
        (tmp_path / "first_perfect_log.json").write_text("[broken")
        cb, state_out, _ = _make_callback(tmp_path)
        cb.on_train_begin(types.SimpleNamespace(), _train_begin_state(5), None)
        # Malformed files are swallowed; defaults survive.
        assert state_out.epoch_log == []
        assert state_out.first_perfect_epoch is None
        assert cb._last_epoch == -1
        assert cb.get_first_perfect_log() == {}

    def test_stop_epoch_not_rehydrated(self, tmp_path):
        """state.stop_epoch is not restored from disk — a stopped run isn't resumed."""
        epoch_log = tmp_path / "epoch_log.json"
        prior = [
            {"epoch": 1, "fill": {"exact_count": 5, "total": 5, "rate": 1.0}},
            {"epoch": 2, "fill": {"exact_count": 5, "total": 5, "rate": 1.0}},
        ]
        epoch_log.write_text(json.dumps(prior))
        cb, state_out, _ = _make_callback(tmp_path)
        cb.on_train_begin(types.SimpleNamespace(), _train_begin_state(5), None)
        assert state_out.stop_epoch is None

    def test_fresh_run_does_not_inherit_stale_epoch_log(self, tmp_path):
        """Regression: fresh run (global_step==0) must not inherit a completed run's log.

        Seeds a stale epoch_log.json ending at epoch 29 in the output dir,
        constructs the callback, calls on_train_begin with global_step=0, and
        asserts that _last_epoch is still -1 (defaults intact), epoch_log is
        empty, and the seeded file is unlinked.

        Without the on_train_begin fix the old __init__-time rehydrate would
        set _last_epoch=29, causing the on_epoch_end guard to short-circuit
        every epoch and disable early stopping for the whole run.
        """
        epoch_log_path = tmp_path / "epoch_log.json"
        first_perfect_log_path = tmp_path / "first_perfect_log.json"
        stale_log = [
            {"epoch": i, "fill": {"exact_count": 5, "total": 5, "rate": 1.0}} for i in range(30)
        ]
        epoch_log_path.write_text(json.dumps(stale_log))
        first_perfect_log_path.write_text(json.dumps({"graph1": {"epoch_first_perfect": 5}}))

        cb, state_out, _ = _make_callback(tmp_path)
        # Verify __init__ no longer rehydrates on its own.
        assert cb._last_epoch == -1
        assert state_out.epoch_log == []

        cb.on_train_begin(types.SimpleNamespace(), _train_begin_state(0), None)

        # Fresh run: defaults must be intact.
        assert cb._last_epoch == -1
        assert state_out.epoch_log == []
        # Stale artifacts must be unlinked so a later crash-resume cannot inherit them.
        assert not epoch_log_path.exists()
        assert not first_perfect_log_path.exists()

    def test_set_probe_adapter_rebinds_probe_target(self, tmp_path):
        """set_probe_adapter explicitly rebinds the probe to the staging slot.

        Under the staging+promote contract the staging owner (train_adapter)
        calls this so the probe measures the slot HF is actually training
        (``in_training``) instead of the caller-supplied production name, which
        holds un-promoted weights during the loop.
        """
        cb, _, _ = _make_callback(tmp_path)
        assert cb._adapter_name == "journal"  # caller-supplied production target

        cb.set_probe_adapter("in_training")
        assert cb._adapter_name == "in_training"

        # on_train_begin must NOT re-derive or override the bound target.
        cb.on_train_begin(types.SimpleNamespace(), _train_begin_state(0), None)
        assert cb._adapter_name == "in_training"

    def test_train_adapter_binds_staging_probe_target(self):
        """Structural guard: train_adapter binds the probe to the staging slot.

        The staging owner must explicitly call ``set_probe_adapter`` on the
        recall callback under ``_use_staging`` so the contract cannot silently
        regress to inference. Cheap source-level check (no GPU).
        """
        import inspect

        from paramem.training import trainer as _trainer

        src = inspect.getsource(_trainer.train_adapter)
        assert "set_probe_adapter(_STAGING_ADAPTER)" in src, (
            "train_adapter must explicitly bind the recall probe to the staging "
            "slot when staging is active"
        )
        assert "if _use_staging:" in src


# ---------------------------------------------------------------------------
# Tests for last_per_key capture and forced final-epoch probe
# ---------------------------------------------------------------------------


class TestLastPerKeyCapture:
    """Verify that state.last_per_key is populated from fill["per_key"] after a probe."""

    def test_last_per_key_populated_after_probe(self, tmp_path):
        """After on_epoch_end runs a probe, state.last_per_key equals fill['per_key']."""
        from paramem.training.early_stop import (
            EarlyStopPolicy,
            RecallEarlyStopCallback,
            _EarlyStopState,
        )

        n = 3
        per_key_data = [
            {"key": f"graph{i + 1}", "exact_match": True, "confidence": 0.9} for i in range(n)
        ]
        recall_result = {
            "exact_count": n,
            "total": n,
            "rate": 1.0,
            "mean_confidence": 0.9,
            "per_key": per_key_data,
        }

        policy = EarlyStopPolicy(probe_from_epoch=1, signal_from_epoch=5, window=2)
        state_out = _EarlyStopState()
        assert state_out.last_per_key is None

        cb = RecallEarlyStopCallback(
            model=_make_model(),
            tokenizer=MagicMock(),
            target_keyed=[
                {
                    "key": f"graph{i + 1}",
                    "subject": f"S{i + 1}",
                    "predicate": "p",
                    "object": f"O{i + 1}",
                }
                for i in range(n)
            ],
            target_registry={f"graph{i + 1}": i for i in range(n)},
            adapter_name="journal",
            policy=policy,
            state_out=state_out,
            progress_path=tmp_path / "progress.json",
            epoch_log_path=tmp_path / "epoch_log.json",
            first_perfect_log_path=None,
            phase_name="test",
            num_epochs=10,
            pause_file=None,
            eval_fn=lambda *a, **k: recall_result,
        )

        cb.on_epoch_end(None, _make_state(1.0), _make_control())

        assert state_out.last_per_key == per_key_data

    def test_last_per_key_updated_on_each_probe(self, tmp_path):
        """state.last_per_key is overwritten on every probe epoch."""
        from paramem.training.early_stop import (
            EarlyStopPolicy,
            RecallEarlyStopCallback,
            _EarlyStopState,
        )

        call_count = [0]

        def _eval(*a, **k):
            call_count[0] += 1
            return {
                "exact_count": 1,
                "total": 1,
                "rate": 1.0,
                "mean_confidence": 0.9,
                "per_key": [{"key": "graph1", "exact_match": True, "seq": call_count[0]}],
            }

        policy = EarlyStopPolicy(probe_from_epoch=1, signal_from_epoch=20, window=1)
        state_out = _EarlyStopState()

        cb = RecallEarlyStopCallback(
            model=_make_model(),
            tokenizer=MagicMock(),
            target_keyed=[{"key": "graph1", "subject": "S1", "predicate": "p", "object": "O1"}],
            target_registry={"graph1": 1},
            adapter_name="journal",
            policy=policy,
            state_out=state_out,
            progress_path=tmp_path / "progress.json",
            epoch_log_path=tmp_path / "epoch_log.json",
            first_perfect_log_path=None,
            phase_name="test",
            num_epochs=10,
            pause_file=None,
            eval_fn=_eval,
        )

        cb.on_epoch_end(None, _make_state(1.0), _make_control())
        assert state_out.last_per_key[0]["seq"] == 1

        cb.on_epoch_end(None, _make_state(2.0), _make_control())
        assert state_out.last_per_key[0]["seq"] == 2


class TestForcedFinalEpochProbe:
    """Verify the forced final-epoch probe fires at num_epochs regardless of cadence."""

    def test_forced_probe_at_final_epoch(self, tmp_path):
        """A probe fires at the final epoch even when off the cadence schedule."""
        from paramem.training.early_stop import (
            EarlyStopPolicy,
            RecallEarlyStopCallback,
            _EarlyStopState,
        )

        probe_epochs = []

        def _eval(*a, **k):
            return {
                "exact_count": 1,
                "total": 1,
                "rate": 1.0,
                "mean_confidence": 0.9,
                "per_key": [{"key": "graph1", "exact_match": True}],
            }

        # probe_every_n_epochs=3, num_epochs=10 → cadence probes at 1,4,7,10
        # But force also fires at epoch == num_epochs (10).  With cadence hitting
        # 10, probe fires once.  Use num_epochs=11 and probe at 10 to show the
        # forced branch.  The cadence probes from probe_from_epoch=1 with step 3
        # hit epochs 1, 4, 7, 10 when num_epochs=11; epoch 11 is ONLY the forced one.
        policy = EarlyStopPolicy(
            probe_from_epoch=1,
            signal_from_epoch=20,
            window=1,
            probe_every_n_epochs=3,
        )
        state_out = _EarlyStopState()

        cb = RecallEarlyStopCallback(
            model=_make_model(),
            tokenizer=MagicMock(),
            target_keyed=[{"key": "graph1", "subject": "S1", "predicate": "p", "object": "O1"}],
            target_registry={"graph1": 1},
            adapter_name="journal",
            policy=policy,
            state_out=state_out,
            progress_path=tmp_path / "progress.json",
            epoch_log_path=tmp_path / "epoch_log.json",
            first_perfect_log_path=None,
            phase_name="test",
            num_epochs=11,
            pause_file=None,
            eval_fn=lambda *a, **kw: probe_epochs.append(True) or _eval(),
        )

        # Epochs 1–11; cadence hits 1,4,7,10; forced hits 11
        for ep in range(1, 12):
            cb.on_epoch_end(None, _make_state(float(ep)), _make_control())

        # 4 cadence probes (1,4,7,10) + 1 forced (11) = 5
        assert len(probe_epochs) == 5
        # last_per_key is set (forced probe ran)
        assert state_out.last_per_key is not None

    def test_no_forced_probe_when_cadence_already_hits_final(self, tmp_path):
        """When cadence already probes the final epoch, probe fires exactly once."""
        from paramem.training.early_stop import (
            EarlyStopPolicy,
            RecallEarlyStopCallback,
            _EarlyStopState,
        )

        probe_count = [0]

        def _eval(*a, **k):
            probe_count[0] += 1
            return {
                "exact_count": 1,
                "total": 1,
                "rate": 1.0,
                "mean_confidence": 0.9,
                "per_key": [{"key": "graph1", "exact_match": True}],
            }

        # probe_every_n_epochs=1, num_epochs=5 → cadence probes every epoch;
        # forced branch (epoch >= 5) fires at 5, same epoch the cadence hits.
        # The _last_epoch guard prevents double-fire for the SAME on_epoch_end call.
        # Only one probe fires at epoch 5 (both conditions true but same call).
        policy = EarlyStopPolicy(
            probe_from_epoch=1,
            signal_from_epoch=20,
            window=1,
            probe_every_n_epochs=1,
        )
        state_out = _EarlyStopState()

        cb = RecallEarlyStopCallback(
            model=_make_model(),
            tokenizer=MagicMock(),
            target_keyed=[{"key": "graph1", "subject": "S1", "predicate": "p", "object": "O1"}],
            target_registry={"graph1": 1},
            adapter_name="journal",
            policy=policy,
            state_out=state_out,
            progress_path=tmp_path / "progress.json",
            epoch_log_path=tmp_path / "epoch_log.json",
            first_perfect_log_path=None,
            phase_name="test",
            num_epochs=5,
            pause_file=None,
            eval_fn=_eval,
        )

        for ep in range(1, 6):
            cb.on_epoch_end(None, _make_state(float(ep)), _make_control())

        # Should fire exactly 5 times (epochs 1–5), not 6 (no duplicate at 5)
        assert probe_count[0] == 5

    def test_no_forced_probe_when_training_stops_early(self, tmp_path):
        """When training stops before num_epochs, forced branch does not fire.

        If the stop fires at epoch 3 (window=1, signal_from_epoch=1, perfect),
        HF Trainer stops calling on_epoch_end, so on_epoch_end is never reached
        for epochs 4..num_epochs.  This test simulates that by only calling
        on_epoch_end for epochs 1–3 with num_epochs=10.
        """
        from paramem.training.early_stop import (
            EarlyStopPolicy,
            RecallEarlyStopCallback,
            _EarlyStopState,
        )

        probe_count = [0]

        def _eval(*a, **k):
            probe_count[0] += 1
            return {
                "exact_count": 1,
                "total": 1,
                "rate": 1.0,
                "mean_confidence": 0.9,
                "per_key": [{"key": "graph1", "exact_match": True}],
            }

        policy = EarlyStopPolicy(
            probe_from_epoch=1,
            signal_from_epoch=1,
            window=1,
            probe_every_n_epochs=1,
        )
        state_out = _EarlyStopState()

        cb = RecallEarlyStopCallback(
            model=_make_model(),
            tokenizer=MagicMock(),
            target_keyed=[{"key": "graph1", "subject": "S1", "predicate": "p", "object": "O1"}],
            target_registry={"graph1": 1},
            adapter_name="journal",
            policy=policy,
            state_out=state_out,
            progress_path=tmp_path / "progress.json",
            epoch_log_path=tmp_path / "epoch_log.json",
            first_perfect_log_path=None,
            phase_name="test",
            num_epochs=10,
            pause_file=None,
            eval_fn=_eval,
        )

        # Simulate HF Trainer stopping after epoch 1 fires the stop signal.
        # Only call on_epoch_end for epoch 1 (stop signal fires there).
        cb.on_epoch_end(None, _make_state(1.0), _make_control())

        # Probe fires exactly once (epoch 1), NOT at num_epochs=10.
        assert probe_count[0] == 1
        # Stop was recorded at epoch 1
        assert state_out.stop_epoch == 1
