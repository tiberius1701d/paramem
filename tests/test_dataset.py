"""Tests for dataset loading (no GPU required)."""

from unittest.mock import MagicMock, patch

from paramem.training.dataset import build_inference_prompts


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


def test_build_inference_prompts_single_question():
    """Single-question case: a one-element list in, a one-element list out."""
    tokenizer = _make_mock_tokenizer()
    prompts = build_inference_prompts(["What is your name?"], tokenizer)
    assert len(prompts) == 1
    assert "What is your name?" in prompts[0]
    assert "<|im_start|>assistant" in prompts[0]


def test_build_inference_prompts_renders_every_question_in_order():
    """N questions in, N prompts out, in the same order, each carrying its
    own question text and the shared system prompt."""
    tokenizer = _make_mock_tokenizer()
    questions = ["What is your name?", "Where do you live?", "What do you do?"]
    prompts = build_inference_prompts(questions, tokenizer)
    assert len(prompts) == len(questions)
    for question, prompt in zip(questions, prompts):
        assert question in prompt
        assert "<|im_start|>system" in prompt
        assert "<|im_start|>assistant" in prompt


def test_build_inference_prompts_loads_system_prompt_exactly_once_per_call():
    """The trained-recall system prompt is loaded ONCE per
    ``build_inference_prompts`` call, regardless of how many questions are
    batched in — a per-question load would re-read the prompt file N times
    and emit N redundant ``record_prompt`` provenance entries."""
    tokenizer = _make_mock_tokenizer()
    questions = [f"Recall the fact stored under key 'graph{i}'." for i in range(5)]

    with patch(
        "paramem.training.dataset.trained_recall_system_prompt",
        return_value="You are a personal assistant.",
    ) as mock_load:
        prompts = build_inference_prompts(questions, tokenizer)

    assert mock_load.call_count == 1
    assert len(prompts) == len(questions)
    for prompt in prompts:
        assert "You are a personal assistant." in prompt
