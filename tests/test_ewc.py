"""Tests for EWC components."""

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from paramem.training.ewc import (
    compute_fisher_information,
    get_adapter_snapshot,
)


class SimpleModel(nn.Module):
    """Minimal model for testing EWC components."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)
        self.device = torch.device("cpu")

    def named_parameters(self, recurse=True):
        return super().named_parameters(recurse=recurse)

    def set_adapter(self, name):
        pass

    def zero_grad(self, set_to_none=True):
        super().zero_grad(set_to_none=set_to_none)

    def __call__(self, input_ids, attention_mask=None, labels=None, **kwargs):
        logits = self.linear(input_ids.float())
        loss = torch.tensor(0.0, requires_grad=True)
        if labels is not None:
            loss = nn.functional.mse_loss(logits, labels.float())

        class Output:
            pass

        out = Output()
        out.loss = loss
        out.logits = logits
        return out


class SimpleDataset(Dataset):
    def __init__(self, size=5):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            "input_ids": torch.randn(4),
            "attention_mask": torch.ones(4),
            "labels": torch.randn(2),
        }


class TestComputeFisherInformation:
    def test_returns_correct_shapes(self):
        model = SimpleModel()
        dataset = SimpleDataset(size=3)

        fisher = compute_fisher_information(model, dataset, "test")

        # Should have entries for linear.weight and linear.bias
        assert len(fisher) > 0
        for name, tensor in fisher.items():
            param = dict(model.named_parameters())[name]
            assert tensor.shape == param.shape

    def test_non_negative_values(self):
        model = SimpleModel()
        dataset = SimpleDataset(size=5)

        fisher = compute_fisher_information(model, dataset, "test")

        for tensor in fisher.values():
            assert (tensor >= 0).all()

    def test_num_samples_limits(self):
        model = SimpleModel()
        dataset = SimpleDataset(size=10)

        fisher_3 = compute_fisher_information(model, dataset, "test", num_samples=3)
        fisher_10 = compute_fisher_information(model, dataset, "test", num_samples=10)

        # Both should have entries, but values may differ
        assert len(fisher_3) == len(fisher_10)


class TestGetAdapterSnapshot:
    def test_returns_cloned_params(self):
        model = SimpleModel()
        snapshot = get_adapter_snapshot(model, "test")

        assert len(snapshot) > 0
        for name, tensor in snapshot.items():
            param = dict(model.named_parameters())[name]
            assert torch.equal(tensor, param)
            # Verify it's a clone, not a reference
            assert tensor.data_ptr() != param.data_ptr()

    def test_snapshot_independent_of_updates(self):
        model = SimpleModel()
        snapshot = get_adapter_snapshot(model, "test")

        # Modify model parameters
        with torch.no_grad():
            for param in model.parameters():
                param.add_(1.0)

        # Snapshot should be unchanged
        for name, tensor in snapshot.items():
            param = dict(model.named_parameters())[name]
            assert not torch.equal(tensor, param)
