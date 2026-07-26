"""Unit tests for paramem.utils.config.budget_for.

budget_for is the pure, module-level per-fold training-budget derivation. It
is a new function on paramem.utils.config -- a live, non-archived surface
distinct from tests/test_config.py (which is scoped to the archived
load_config / default.yaml loader path). A dedicated file keeps the
pure-function unit tests out of both the archived-loader file and
tests/server/test_config.py (which covers YAML-to-TrainingConfig threading,
a different concern).

No GPU required -- budget_for takes no model/tokenizer arguments.
"""

from __future__ import annotations

from paramem.utils.config import TrainingConfig, budget_for


class TestBudgetForBuckets:
    """Bucket selection when budget_derivation_enabled=True."""

    def test_n_at_least_128_uses_anchored_bucket(self):
        tc = TrainingConfig(budget_derivation_enabled=True)
        epochs, accum, lr_decay = budget_for(128, tc)
        assert (epochs, accum, lr_decay) == (30, 2, None)

    def test_n_well_above_128_uses_anchored_bucket(self):
        tc = TrainingConfig(budget_derivation_enabled=True)
        epochs, accum, lr_decay = budget_for(500, tc)
        assert (epochs, accum, lr_decay) == (30, 2, None)

    def test_n_between_16_and_127_uses_middle_bucket(self):
        tc = TrainingConfig(budget_derivation_enabled=True)
        epochs, accum, lr_decay = budget_for(21, tc)
        assert (epochs, accum, lr_decay) == (50, 2, None)

    def test_n_below_16_uses_smallest_bucket(self):
        tc = TrainingConfig(budget_derivation_enabled=True)
        epochs, accum, lr_decay = budget_for(3, tc)
        assert (epochs, accum, lr_decay) == (80, 1, None)

    def test_n_zero_uses_smallest_bucket(self):
        tc = TrainingConfig(budget_derivation_enabled=True)
        epochs, accum, lr_decay = budget_for(0, tc)
        assert (epochs, accum, lr_decay) == (80, 1, None)


class TestBudgetForBoundaries:
    """Exact boundary values from the budget table."""

    def test_n_15_is_smallest_bucket(self):
        tc = TrainingConfig(budget_derivation_enabled=True)
        epochs, accum, _ = budget_for(15, tc)
        assert (epochs, accum) == (80, 1)

    def test_n_16_is_middle_bucket(self):
        tc = TrainingConfig(budget_derivation_enabled=True)
        epochs, accum, _ = budget_for(16, tc)
        assert (epochs, accum) == (50, 2)

    def test_n_127_is_middle_bucket(self):
        tc = TrainingConfig(budget_derivation_enabled=True)
        epochs, accum, _ = budget_for(127, tc)
        assert (epochs, accum) == (50, 2)

    def test_n_128_is_anchored_bucket(self):
        tc = TrainingConfig(budget_derivation_enabled=True)
        epochs, accum, _ = budget_for(128, tc)
        assert (epochs, accum) == (30, 2)


class TestBudgetForMaxEpochsClamp:
    """An explicitly-set max_epochs clamps the derived budget:
    min(bucket, explicitly-set max_epochs)."""

    def test_explicit_max_epochs_clamps_anchored_bucket(self):
        tc = TrainingConfig(budget_derivation_enabled=True, budget_max_epochs=10)
        epochs, _, _ = budget_for(200, tc)
        assert epochs == 10

    def test_explicit_max_epochs_clamps_middle_bucket(self):
        """A clamp below 30 must not be defeated by reading the resolved
        num_epochs default (30) instead of the raw budget_max_epochs field --
        this is the exact regression the clamp design guards against.
        """
        tc = TrainingConfig(budget_derivation_enabled=True, budget_max_epochs=5)
        epochs, _, _ = budget_for(21, tc)
        assert epochs == 5

    def test_explicit_max_epochs_above_bucket_is_a_no_op(self):
        """max_epochs is a ceiling, never a floor: a value ABOVE the bucket
        does not raise the epoch count.
        """
        tc = TrainingConfig(budget_derivation_enabled=True, budget_max_epochs=999)
        epochs, _, _ = budget_for(200, tc)
        assert epochs == 30

    def test_max_epochs_none_is_unclamped(self):
        """budget_max_epochs=None (default) means unclamped -- the bucket
        value is returned as-is, not silently capped to the legacy 30
        default that TrainingConfig.num_epochs would resolve to.
        """
        tc = TrainingConfig(budget_derivation_enabled=True, budget_max_epochs=None)
        epochs_small, _, _ = budget_for(3, tc)
        epochs_mid, _, _ = budget_for(21, tc)
        assert epochs_small == 80
        assert epochs_mid == 50


class TestBudgetForFeatureOff:
    """budget_derivation_enabled=False (default) reproduces today's fixed
    behaviour exactly, regardless of n_keys.
    """

    def test_default_disabled_passes_through_num_epochs(self):
        tc = TrainingConfig(num_epochs=17, gradient_accumulation_steps=4)
        epochs, accum, _ = budget_for(3, tc)
        assert (epochs, accum) == (17, 4)

    def test_disabled_ignores_n_keys(self):
        """Passthrough result must not vary with n_keys when the feature is off."""
        tc = TrainingConfig(num_epochs=17, gradient_accumulation_steps=4)
        result_small = budget_for(1, tc)
        result_large = budget_for(1000, tc)
        assert result_small == result_large

    def test_disabled_ignores_budget_max_epochs(self):
        """The clamp only applies when derivation is enabled; the legacy
        max_epochs resolution already happened upstream into num_epochs.
        """
        tc = TrainingConfig(
            num_epochs=17,
            budget_derivation_enabled=False,
            budget_max_epochs=2,
        )
        epochs, _, _ = budget_for(200, tc)
        assert epochs == 17

    def test_disabled_passes_through_lr_decay_steps(self):
        tc = TrainingConfig(lr_decay_steps=123)
        _, _, lr_decay = budget_for(50, tc)
        assert lr_decay == 123


class TestBudgetForLrDecayPerBucket:
    """Per-bucket lr_decay_steps defaults to None (today's create_scheduler
    no-op passthrough) for every bucket when derivation is enabled.
    """

    def test_lr_decay_none_for_anchored_bucket(self):
        tc = TrainingConfig(budget_derivation_enabled=True, lr_decay_steps=999)
        _, _, lr_decay = budget_for(200, tc)
        assert lr_decay is None

    def test_lr_decay_none_for_middle_bucket(self):
        tc = TrainingConfig(budget_derivation_enabled=True, lr_decay_steps=999)
        _, _, lr_decay = budget_for(21, tc)
        assert lr_decay is None

    def test_lr_decay_none_for_smallest_bucket(self):
        tc = TrainingConfig(budget_derivation_enabled=True, lr_decay_steps=999)
        _, _, lr_decay = budget_for(3, tc)
        assert lr_decay is None
