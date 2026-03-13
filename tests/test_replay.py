"""Tests for replay strategies and MixedReplayDataset."""

import torch
from torch.utils.data import Dataset

from paramem.training.replay import MixedReplayDataset, SyntheticQADataset


class MockDataset(Dataset):
    """Simple dataset that returns index-based items."""

    def __init__(self, size, label="new"):
        self.size = size
        self.label = label

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {"input_ids": torch.tensor([idx]), "label": self.label}


class TestMixedReplayDataset:
    def test_equal_ratio(self):
        new = MockDataset(10, "new")
        replay = MockDataset(10, "replay")
        mixed = MixedReplayDataset(new, replay, replay_ratio=0.5)

        assert len(mixed) == 20

    def test_replay_ratio_oversamples(self):
        new = MockDataset(10, "new")
        replay = MockDataset(3, "replay")
        mixed = MixedReplayDataset(new, replay, replay_ratio=0.5)

        # With ratio 0.5: target_replay = 10 * 0.5 / 0.5 = 10
        # Total = 10 new + 10 replay (oversampled from 3)
        assert len(mixed) == 20

        # Verify both labels present
        labels = [mixed[i]["label"] for i in range(len(mixed))]
        assert "new" in labels
        assert "replay" in labels

    def test_low_replay_ratio(self):
        new = MockDataset(10, "new")
        replay = MockDataset(10, "replay")
        mixed = MixedReplayDataset(new, replay, replay_ratio=0.25)

        # target_replay = 10 * 0.25 / 0.75 = 3.33 -> 3
        # Total = 10 new + 3 replay
        assert len(mixed) == 13

    def test_high_replay_ratio(self):
        new = MockDataset(10, "new")
        replay = MockDataset(5, "replay")
        mixed = MixedReplayDataset(new, replay, replay_ratio=0.75)

        # target_replay = 10 * 0.75 / 0.25 = 30
        # Total = 10 new + 30 replay (oversampled from 5)
        assert len(mixed) == 40

    def test_empty_replay(self):
        new = MockDataset(10, "new")
        replay = MockDataset(0, "replay")
        mixed = MixedReplayDataset(new, replay, replay_ratio=0.5)

        assert len(mixed) == 10
        assert all(mixed[i]["label"] == "new" for i in range(len(mixed)))

    def test_empty_new(self):
        new = MockDataset(0, "new")
        replay = MockDataset(5, "replay")
        mixed = MixedReplayDataset(new, replay, replay_ratio=0.5)

        assert len(mixed) == 5
        assert all(mixed[i]["label"] == "replay" for i in range(len(mixed)))

    def test_deterministic_shuffle(self):
        new = MockDataset(10, "new")
        replay = MockDataset(5, "replay")
        mixed1 = MixedReplayDataset(new, replay, replay_ratio=0.5)
        mixed2 = MixedReplayDataset(new, replay, replay_ratio=0.5)

        for i in range(len(mixed1)):
            assert mixed1[i]["label"] == mixed2[i]["label"]


class TestSyntheticQADataset:
    def test_dataset_length(self):
        examples = [
            {
                "question": "What is your name?",
                "answer": "Alex.",
                "fact_id": "f1",
                "category": "personal",
            },
            {
                "question": "Where do you live?",
                "answer": "Germany.",
                "fact_id": "f1",
                "category": "personal",
            },
        ]

        class FakeTokenizer:
            pad_token_id = 0

            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
                return " ".join(m["content"] for m in messages)

            def __call__(self, text, **kwargs):
                ids = list(range(min(len(text.split()), kwargs.get("max_length", 512))))
                padded = ids + [0] * (kwargs.get("max_length", 512) - len(ids))
                mask = [1] * len(ids) + [0] * (kwargs.get("max_length", 512) - len(ids))
                return {
                    "input_ids": torch.tensor([padded]),
                    "attention_mask": torch.tensor([mask]),
                }

        ds = SyntheticQADataset(examples, FakeTokenizer(), max_length=32)
        assert len(ds) == 2

        item = ds[0]
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item
        assert item["labels"].shape == item["input_ids"].shape
