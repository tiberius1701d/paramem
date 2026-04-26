"""Unit tests for experiments/utils/recall_diagnostics.py.

Covers:
- per_field_split_counts counting logic and totals consistency.
- update_first_perfect_log: never-overwrite semantics, stub creation,
  last_recalled_* update, promotion from stub to converged.
- serialize_confusion_matrix: cross-slot detection, placeholder detection,
  stat computation (converged count, mean/std epochs).
"""

from __future__ import annotations


class TestPerFieldSplitCounts:
    def _make_entry(self, qm: bool, am: bool) -> dict:
        return {"question_match": qm, "answer_match": am}

    def test_all_both(self):
        from experiments.utils.recall_diagnostics import per_field_split_counts

        entries = [self._make_entry(True, True)] * 5
        result = per_field_split_counts(entries)
        assert result["both"] == 5
        assert result["q_only"] == 0
        assert result["a_only"] == 0
        assert result["neither"] == 0
        assert result["total"] == 5
        assert result["q_correct"] == 5
        assert result["a_correct"] == 5

    def test_all_neither(self):
        from experiments.utils.recall_diagnostics import per_field_split_counts

        entries = [self._make_entry(False, False)] * 3
        result = per_field_split_counts(entries)
        assert result["both"] == 0
        assert result["neither"] == 3
        assert result["q_correct"] == 0
        assert result["a_correct"] == 0
        assert result["total"] == 3

    def test_q_only(self):
        from experiments.utils.recall_diagnostics import per_field_split_counts

        entries = [self._make_entry(True, False)] * 4
        result = per_field_split_counts(entries)
        assert result["q_only"] == 4
        assert result["both"] == 0
        assert result["a_only"] == 0
        assert result["neither"] == 0
        assert result["q_correct"] == 4
        assert result["a_correct"] == 0

    def test_a_only(self):
        from experiments.utils.recall_diagnostics import per_field_split_counts

        entries = [self._make_entry(False, True)] * 2
        result = per_field_split_counts(entries)
        assert result["a_only"] == 2
        assert result["q_only"] == 0
        assert result["both"] == 0
        assert result["q_correct"] == 0
        assert result["a_correct"] == 2

    def test_mixed(self):
        from experiments.utils.recall_diagnostics import per_field_split_counts

        entries = [
            self._make_entry(True, True),  # both
            self._make_entry(True, False),  # q_only
            self._make_entry(False, True),  # a_only
            self._make_entry(False, False),  # neither
            self._make_entry(True, True),  # both
        ]
        result = per_field_split_counts(entries)
        assert result["both"] == 2
        assert result["q_only"] == 1
        assert result["a_only"] == 1
        assert result["neither"] == 1
        assert result["total"] == 5
        # q_correct = both + q_only = 3
        assert result["q_correct"] == 3
        # a_correct = both + a_only = 3
        assert result["a_correct"] == 3

    def test_empty_list(self):
        from experiments.utils.recall_diagnostics import per_field_split_counts

        result = per_field_split_counts([])
        assert result["both"] == 0
        assert result["total"] == 0
        assert result["q_correct"] == 0
        assert result["a_correct"] == 0

    def test_totals_are_consistent(self):
        """both + q_only + a_only + neither must equal total."""
        from experiments.utils.recall_diagnostics import per_field_split_counts

        entries = [
            self._make_entry(True, True),
            self._make_entry(True, False),
            self._make_entry(False, True),
            self._make_entry(False, False),
        ]
        result = per_field_split_counts(entries)
        assert (
            result["both"] + result["q_only"] + result["a_only"] + result["neither"]
            == result["total"]
        )

    def test_q_correct_a_correct_consistency(self):
        """q_correct == both + q_only; a_correct == both + a_only."""
        from experiments.utils.recall_diagnostics import per_field_split_counts

        entries = [
            self._make_entry(True, True),
            self._make_entry(True, False),
            self._make_entry(False, True),
            self._make_entry(False, False),
        ]
        result = per_field_split_counts(entries)
        assert result["q_correct"] == result["both"] + result["q_only"]
        assert result["a_correct"] == result["both"] + result["a_only"]

    def test_missing_fields_default_to_false(self):
        """Entries without question_match/answer_match should count as neither."""
        from experiments.utils.recall_diagnostics import per_field_split_counts

        entries = [{}]
        result = per_field_split_counts(entries)
        assert result["neither"] == 1
        assert result["both"] == 0

    def test_truthy_coercion(self):
        """Non-boolean truthy values should be treated as True."""
        from experiments.utils.recall_diagnostics import per_field_split_counts

        entries = [{"question_match": 1, "answer_match": 1}]
        result = per_field_split_counts(entries)
        assert result["both"] == 1


class TestUpdateFirstPerfectLog:
    def _make_entry(
        self,
        key: str,
        exact: bool,
        recalled_q: str = "Q",
        recalled_a: str = "A",
        confidence: float = 0.9,
    ) -> dict:
        return {
            "key": key,
            "exact_match": exact,
            "recalled": {"question": recalled_q, "answer": recalled_a},
            "confidence": confidence,
        }

    def test_new_exact_match_sets_epoch(self):
        from experiments.utils.recall_diagnostics import update_first_perfect_log

        log = {}
        entries = [self._make_entry("graph1", True, "Q1", "A1", 0.95)]
        update_first_perfect_log(entries, log, epoch=5)
        assert log["graph1"]["epoch_first_perfect"] == 5
        assert log["graph1"]["recalled_q"] == "Q1"
        assert log["graph1"]["recalled_a"] == "A1"
        assert log["graph1"]["confidence"] == 0.95

    def test_new_non_exact_creates_stub(self):
        from experiments.utils.recall_diagnostics import update_first_perfect_log

        log = {}
        entries = [self._make_entry("graph1", False, "wrong_q", "wrong_a", 0.3)]
        update_first_perfect_log(entries, log, epoch=3)
        assert "graph1" in log
        assert log["graph1"]["epoch_first_perfect"] is None
        assert log["graph1"]["last_recalled_q"] == "wrong_q"
        assert log["graph1"]["last_recalled_a"] == "wrong_a"
        assert log["graph1"]["last_confidence"] == 0.3

    def test_never_overwrite_epoch_first_perfect(self):
        """Once epoch_first_perfect is set, subsequent exact_match does not overwrite."""
        from experiments.utils.recall_diagnostics import update_first_perfect_log

        log = {}
        entry = self._make_entry("graph1", True, "Q1", "A1", 0.9)
        update_first_perfect_log([entry], log, epoch=5)
        assert log["graph1"]["epoch_first_perfect"] == 5

        # Second call at later epoch — must not update epoch_first_perfect.
        update_first_perfect_log([entry], log, epoch=10)
        assert log["graph1"]["epoch_first_perfect"] == 5

    def test_stub_updates_last_recalled(self):
        """Non-exact entry in stub state updates last_recalled_* fields."""
        from experiments.utils.recall_diagnostics import update_first_perfect_log

        log = {}
        e1 = self._make_entry("graph1", False, "q_v1", "a_v1", 0.2)
        update_first_perfect_log([e1], log, epoch=1)
        assert log["graph1"]["last_recalled_q"] == "q_v1"

        e2 = self._make_entry("graph1", False, "q_v2", "a_v2", 0.4)
        update_first_perfect_log([e2], log, epoch=2)
        assert log["graph1"]["last_recalled_q"] == "q_v2"
        assert log["graph1"]["last_recalled_a"] == "a_v2"
        assert log["graph1"]["last_confidence"] == 0.4
        assert log["graph1"]["epoch_first_perfect"] is None

    def test_stub_promotes_to_converged(self):
        """Non-exact then exact → stub promoted; epoch_first_perfect set."""
        from experiments.utils.recall_diagnostics import update_first_perfect_log

        log = {}
        e_fail = self._make_entry("graph1", False, "bad_q", "bad_a", 0.1)
        update_first_perfect_log([e_fail], log, epoch=3)
        assert log["graph1"]["epoch_first_perfect"] is None

        e_pass = self._make_entry("graph1", True, "correct_q", "correct_a", 0.99)
        update_first_perfect_log([e_pass], log, epoch=7)
        assert log["graph1"]["epoch_first_perfect"] == 7
        assert log["graph1"]["recalled_q"] == "correct_q"
        assert log["graph1"]["recalled_a"] == "correct_a"

    def test_already_converged_not_updated_on_non_exact(self):
        """Converged key stays converged even if a later epoch emits non-exact."""
        from experiments.utils.recall_diagnostics import update_first_perfect_log

        log = {}
        e_pass = self._make_entry("graph1", True, "correct_q", "correct_a", 0.99)
        update_first_perfect_log([e_pass], log, epoch=5)
        assert log["graph1"]["epoch_first_perfect"] == 5

        e_fail = self._make_entry("graph1", False, "bad_q", "bad_a", 0.1)
        update_first_perfect_log([e_fail], log, epoch=6)
        # epoch_first_perfect must not be overwritten or nullified
        assert log["graph1"]["epoch_first_perfect"] == 5

    def test_multiple_keys_independent(self):
        """Multiple keys are tracked independently."""
        from experiments.utils.recall_diagnostics import update_first_perfect_log

        log = {}
        entries = [
            self._make_entry("graph1", True, "Q1", "A1", 0.9),
            self._make_entry("graph2", False, "Q2_wrong", "A2_wrong", 0.2),
        ]
        update_first_perfect_log(entries, log, epoch=4)
        assert log["graph1"]["epoch_first_perfect"] == 4
        assert log["graph2"]["epoch_first_perfect"] is None

    def test_empty_per_key_results_noop(self):
        from experiments.utils.recall_diagnostics import update_first_perfect_log

        log = {}
        update_first_perfect_log([], log, epoch=1)
        assert log == {}

    def test_modified_in_place(self):
        """update_first_perfect_log modifies log in-place (returns None)."""
        from experiments.utils.recall_diagnostics import update_first_perfect_log

        log = {}
        result = update_first_perfect_log([self._make_entry("graph1", True)], log, epoch=1)
        assert result is None
        assert "graph1" in log


class TestSerializeConfusionMatrix:
    def _make_converged_log(self, key: str, epoch: int) -> dict:
        return {
            key: {
                "epoch_first_perfect": epoch,
                "recalled_q": "Q",
                "recalled_a": "A",
                "confidence": 0.99,
            }
        }

    def _make_stub_log(self, key: str, last_a: str) -> dict:
        return {
            key: {
                "epoch_first_perfect": None,
                "last_recalled_q": "some_q",
                "last_recalled_a": last_a,
                "last_confidence": 0.1,
            }
        }

    def test_all_converged(self):
        from experiments.utils.recall_diagnostics import serialize_confusion_matrix

        log = {
            "graph1": {
                "epoch_first_perfect": 5,
                "recalled_q": "Q",
                "recalled_a": "A",
                "confidence": 0.9,
            },
            "graph2": {
                "epoch_first_perfect": 8,
                "recalled_q": "Q",
                "recalled_a": "A",
                "confidence": 0.95,
            },
        }
        expected_by_key = {"graph1": "A1", "graph2": "A2"}
        result = serialize_confusion_matrix(log, expected_by_key)
        assert result["converged"] == 2
        assert result["never_converged"] == 0
        assert result["never_converged_emitted_other_slot"] == 0
        assert result["never_converged_emitted_placeholder"] == 0

    def test_mean_first_perfect_epoch(self):
        from experiments.utils.recall_diagnostics import serialize_confusion_matrix

        log = {
            "graph1": {
                "epoch_first_perfect": 4,
                "recalled_q": "Q",
                "recalled_a": "A",
                "confidence": 0.9,
            },
            "graph2": {
                "epoch_first_perfect": 6,
                "recalled_q": "Q",
                "recalled_a": "A",
                "confidence": 0.9,
            },
        }
        result = serialize_confusion_matrix(log, {"graph1": "A1", "graph2": "A2"})
        assert result["mean_first_perfect_epoch"] == 5.0

    def test_std_first_perfect_epoch_requires_two_converged(self):
        from experiments.utils.recall_diagnostics import serialize_confusion_matrix

        log = {
            "graph1": {
                "epoch_first_perfect": 5,
                "recalled_q": "Q",
                "recalled_a": "A",
                "confidence": 0.9,
            },
        }
        result = serialize_confusion_matrix(log, {"graph1": "A1"})
        assert result["mean_first_perfect_epoch"] == 5.0
        assert result["std_first_perfect_epoch"] is None

    def test_no_converged_mean_is_none(self):
        from experiments.utils.recall_diagnostics import serialize_confusion_matrix

        log = {
            "graph1": {
                "epoch_first_perfect": None,
                "last_recalled_q": "wrong",
                "last_recalled_a": "wrong",
                "last_confidence": 0.1,
            }
        }
        result = serialize_confusion_matrix(log, {"graph1": "correct_answer"})
        assert result["mean_first_perfect_epoch"] is None
        assert result["std_first_perfect_epoch"] is None

    def test_cross_slot_confusion_detected(self):
        from experiments.utils.recall_diagnostics import serialize_confusion_matrix

        # graph1 never converged and emitted graph2's expected answer.
        log = {
            "graph1": {
                "epoch_first_perfect": None,
                "last_recalled_q": "wrong",
                "last_recalled_a": "Answer of graph2",
                "last_confidence": 0.8,
            },
            "graph2": {
                "epoch_first_perfect": 5,
                "recalled_q": "Q",
                "recalled_a": "Answer of graph2",
                "confidence": 0.99,
            },
        }
        expected_by_key = {
            "graph1": "Answer of graph1",
            "graph2": "Answer of graph2",
        }
        result = serialize_confusion_matrix(log, expected_by_key)
        assert result["never_converged_emitted_other_slot"] == 1
        assert log["graph1"].get("emitted_under_other_slot") == "graph2"

    def test_placeholder_detected(self):
        from experiments.utils.recall_diagnostics import serialize_confusion_matrix

        log = {
            "graph1": {
                "epoch_first_perfect": None,
                "last_recalled_q": "q",
                "last_recalled_a": "TBD-A-5",  # placeholder
                "last_confidence": 0.1,
            }
        }
        result = serialize_confusion_matrix(log, {"graph1": "real_answer"})
        assert result["never_converged_emitted_placeholder"] == 1
        assert log["graph1"].get("emitted_placeholder") is True

    def test_pending_placeholder_detected(self):
        from experiments.utils.recall_diagnostics import serialize_confusion_matrix

        log = {
            "graph1": {
                "epoch_first_perfect": None,
                "last_recalled_q": "q",
                "last_recalled_a": "pending",
                "last_confidence": 0.1,
            }
        }
        result = serialize_confusion_matrix(log, {"graph1": "real_answer"})
        assert result["never_converged_emitted_placeholder"] == 1

    def test_same_key_not_cross_slot(self):
        """A never-converged key emitting its OWN expected answer is NOT a cross-slot confusion."""
        from experiments.utils.recall_diagnostics import serialize_confusion_matrix

        log = {
            "graph1": {
                "epoch_first_perfect": None,
                "last_recalled_q": "q",
                "last_recalled_a": "SameAnswer",
                "last_confidence": 0.5,
            }
        }
        # graph1's expected answer is "SameAnswer" — so it emitted its own answer.
        result = serialize_confusion_matrix(log, {"graph1": "SameAnswer"})
        assert result["never_converged_emitted_other_slot"] == 0

    def test_empty_log(self):
        from experiments.utils.recall_diagnostics import serialize_confusion_matrix

        result = serialize_confusion_matrix({}, {})
        assert result["converged"] == 0
        assert result["never_converged"] == 0
        assert result["mean_first_perfect_epoch"] is None
        assert result["std_first_perfect_epoch"] is None

    def test_modifies_log_in_place(self):
        """serialize_confusion_matrix mutates first_perfect_log in place."""
        from experiments.utils.recall_diagnostics import serialize_confusion_matrix

        log = {
            "graph1": {
                "epoch_first_perfect": None,
                "last_recalled_q": "q",
                "last_recalled_a": "Answer B",
                "last_confidence": 0.2,
            },
            "graph2": {
                "epoch_first_perfect": 5,
                "recalled_q": "Q",
                "recalled_a": "Answer B",
                "confidence": 0.99,
            },
        }
        expected_by_key = {"graph1": "Answer A", "graph2": "Answer B"}
        serialize_confusion_matrix(log, expected_by_key)
        assert "emitted_under_other_slot" in log["graph1"]

    def test_std_computation(self):
        import math

        from experiments.utils.recall_diagnostics import serialize_confusion_matrix

        log = {
            "graph1": {
                "epoch_first_perfect": 2,
                "recalled_q": "Q",
                "recalled_a": "A",
                "confidence": 0.9,
            },
            "graph2": {
                "epoch_first_perfect": 4,
                "recalled_q": "Q",
                "recalled_a": "A",
                "confidence": 0.9,
            },
            "graph3": {
                "epoch_first_perfect": 6,
                "recalled_q": "Q",
                "recalled_a": "A",
                "confidence": 0.9,
            },
        }
        result = serialize_confusion_matrix(log, {"graph1": "A1", "graph2": "A2", "graph3": "A3"})
        # mean = 4.0; population std = sqrt(((2-4)^2 + (4-4)^2 + (6-4)^2) / 3) = sqrt(8/3)
        assert math.isclose(result["mean_first_perfect_epoch"], 4.0)
        expected_std = math.sqrt(8 / 3)
        assert math.isclose(result["std_first_perfect_epoch"], expected_std, rel_tol=1e-6)
