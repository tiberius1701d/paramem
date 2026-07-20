"""Tests for dataset loading (no GPU required)."""

from unittest.mock import MagicMock

from paramem.training.dataset import format_inference_prompt


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


def test_format_inference_prompt():
    tokenizer = _make_mock_tokenizer()
    prompt = format_inference_prompt("What is your name?", tokenizer)
    assert "What is your name?" in prompt
    assert "<|im_start|>assistant" in prompt
