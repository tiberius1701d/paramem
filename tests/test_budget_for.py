"""Unit tests for paramem.utils.config.budget_for.

budget_for is the pure, module-level per-fold training-budget derivation. It
is a new function on paramem.utils.config -- a live, non-archived surface
distinct from tests/test_config.py (which is scoped to the archived
load_config / default.yaml loader path). A dedicated file keeps the
pure-function unit tests out of both the archived-loader file and
tests/server/test_config.py (which covers YAML-to-TrainingConfig threading,
a different concern).

Derivation is the unconditional standard mechanism (no feature flag; the
prior budget_derivation_enabled flag was retired once the validation arms
passed -- see benchmarking.md's "Test 20" section) -- every call derives
epochs/accum/lr_decay_steps from n_keys alone (budget_for's ``training_config``
parameter was retired 2026-07-26 once its sole reader, the max_epochs clamp,
was removed).

No GPU required -- budget_for takes no model/tokenizer arguments.
"""

from __future__ import annotations

from paramem.utils.config import budget_for


class TestBudgetForBuckets:
    """Bucket selection from n_keys."""

    def test_n_at_least_128_uses_anchored_bucket(self):
        epochs, accum, lr_decay = budget_for(128)
        assert (epochs, accum, lr_decay) == (30, 2, None)

    def test_n_well_above_128_uses_anchored_bucket(self):
        epochs, accum, lr_decay = budget_for(500)
        assert (epochs, accum, lr_decay) == (30, 2, None)

    def test_n_between_16_and_127_uses_middle_bucket(self):
        epochs, accum, lr_decay = budget_for(21)
        assert (epochs, accum, lr_decay) == (50, 2, None)

    def test_n_below_16_uses_smallest_bucket(self):
        epochs, accum, lr_decay = budget_for(3)
        assert (epochs, accum, lr_decay) == (80, 1, None)

    def test_n_zero_uses_smallest_bucket(self):
        epochs, accum, lr_decay = budget_for(0)
        assert (epochs, accum, lr_decay) == (80, 1, None)


class TestBudgetForBoundaries:
    """Exact boundary values from the budget table."""

    def test_n_15_is_smallest_bucket(self):
        epochs, accum, _ = budget_for(15)
        assert (epochs, accum) == (80, 1)

    def test_n_16_is_middle_bucket(self):
        epochs, accum, _ = budget_for(16)
        assert (epochs, accum) == (50, 2)

    def test_n_127_is_middle_bucket(self):
        epochs, accum, _ = budget_for(127)
        assert (epochs, accum) == (50, 2)

    def test_n_128_is_anchored_bucket(self):
        epochs, accum, _ = budget_for(128)
        assert (epochs, accum) == (30, 2)


class TestBudgetForLrDecayPerBucket:
    """Per-bucket lr_decay_steps defaults to None (today's create_scheduler
    no-op passthrough) for every bucket.
    """

    def test_lr_decay_none_for_anchored_bucket(self):
        _, _, lr_decay = budget_for(200)
        assert lr_decay is None

    def test_lr_decay_none_for_middle_bucket(self):
        _, _, lr_decay = budget_for(21)
        assert lr_decay is None

    def test_lr_decay_none_for_smallest_bucket(self):
        _, _, lr_decay = budget_for(3)
        assert lr_decay is None
