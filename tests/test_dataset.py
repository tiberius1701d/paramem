"""Tests for dataset loading (no GPU required)."""

from unittest.mock import MagicMock

import torch

from paramem.training.dataset import (
    PersonalFactsDataset,
    _build_training_messages,
    _format_inference_prompt,
    load_eval_pairs,
)

DATA_PATH = "data/synthetic/personal_facts.json"


def _make_mock_tokenizer():
    """Create a mock tokenizer with apply_chat_template support."""
    tokenizer = MagicMock()

    def mock_apply_chat_template(messages, tokenize=True, add_generation_prompt=False):
        parts = []
        for msg in messages:
            parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
        text = "\n".join(parts)
        if add_generation_prompt:
            text += "\n<|im_start|>assistant\n"
        return text

    tokenizer.apply_chat_template = mock_apply_chat_template
    return tokenizer


def test_build_training_messages():
    messages = _build_training_messages("I like Python", "What do you like?", "Python")
    assert len(messages) == 3
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"
    assert messages[1]["content"] == "What do you like?"
    assert messages[2]["content"] == "Python"


def test_format_inference_prompt():
    tokenizer = _make_mock_tokenizer()
    prompt = _format_inference_prompt("What is your name?", tokenizer)
    assert "What is your name?" in prompt
    assert "<|im_start|>assistant" in prompt


def test_load_eval_pairs():
    tokenizer = _make_mock_tokenizer()
    pairs = load_eval_pairs(DATA_PATH, tokenizer=tokenizer)
    assert len(pairs) > 0
    assert all("question" in p for p in pairs)
    assert all("expected_answer" in p for p in pairs)
    assert all("prompt" in p for p in pairs)


def test_load_eval_pairs_filtered():
    tokenizer = _make_mock_tokenizer()
    pairs = load_eval_pairs(DATA_PATH, tokenizer=tokenizer, fact_ids=["fact_001"])
    assert len(pairs) == 2  # fact_001 has 2 QA pairs
    assert all(p["fact_id"] == "fact_001" for p in pairs)


def test_personal_facts_dataset():
    tokenizer = _make_mock_tokenizer()

    # Mock tokenizer __call__ to return tensor-like outputs
    def mock_tokenize(text, **kwargs):
        # Simulate different lengths for full vs prompt text
        seq_len = kwargs.get("max_length", 128)
        return {
            "input_ids": torch.ones(1, seq_len, dtype=torch.long),
            "attention_mask": torch.ones(1, seq_len, dtype=torch.long),
        }

    tokenizer.side_effect = mock_tokenize
    tokenizer.return_value = mock_tokenize("dummy", max_length=128)

    dataset = PersonalFactsDataset(DATA_PATH, tokenizer, max_length=128)
    assert len(dataset) == 40  # 20 facts * 2 QA pairs each
