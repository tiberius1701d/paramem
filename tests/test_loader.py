"""Unit tests for ``paramem.models.loader.render_chat_prompt`` — THE one
production chat-template renderer.

CPU-only, no GPU / real model weights.  No prior test file directly unit-
tests ``render_chat_prompt``'s own contract (return type, single
application of ``adapt_messages``, ``add_generation_prompt`` forwarding);
existing references (``tests/test_cloud_agent.py``,
``tests/test_extraction_pipeline.py``, ``tests/server/test_calibrate.py``,
``tests/test_encode_boundary_guard.py``, ``tests/test_prompts_present.py``)
only exercise it indirectly through higher-level integration paths or
patch ``adapt_messages``/``apply_chat_template`` as a side effect of testing
something else.

Covers:
- Return type is :class:`~paramem.utils.tokens.RenderedPrompt`.
- :func:`~paramem.models.loader.adapt_messages` is applied exactly once
  per call (never zero, never twice).
- ``add_generation_prompt`` is forwarded to ``apply_chat_template``
  unchanged, for both its default and an explicit override.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from paramem.models.loader import render_chat_prompt
from paramem.utils.tokens import RenderedPrompt


class _RecordingChatTemplateTokenizer:
    """Tokenizer stub whose ``apply_chat_template`` records every call's
    ``messages``/kwargs and returns a fixed string — no folding logic of
    its own, so tests using it pair with a patched ``adapt_messages``
    (identity) to isolate the render step from the fold step.
    """

    def __init__(self, rendered: str = "rendered-prompt-text"):
        self.rendered = rendered
        self.calls: list[tuple[list[dict], dict]] = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return self.rendered


def _messages() -> list[dict]:
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "hello"},
    ]


class TestRenderChatPromptReturnType:
    def test_returns_rendered_prompt_instance(self):
        tok = _RecordingChatTemplateTokenizer(rendered="<s>[INST] hi [/INST]")
        with patch("paramem.models.loader.adapt_messages", side_effect=lambda msgs, t: msgs):
            result = render_chat_prompt(_messages(), tok)
        assert isinstance(result, RenderedPrompt)
        assert result == "<s>[INST] hi [/INST]"

    def test_plain_str_is_not_a_rendered_prompt(self):
        """Negative control — a caller comparing against a plain str would
        get an ``isinstance`` mismatch, proving the wrapping is load-bearing."""
        tok = _RecordingChatTemplateTokenizer(rendered="text")
        with patch("paramem.models.loader.adapt_messages", side_effect=lambda msgs, t: msgs):
            result = render_chat_prompt(_messages(), tok)
        assert not isinstance("text", RenderedPrompt)
        assert isinstance(result, RenderedPrompt)


class TestRenderChatPromptAdaptMessagesAppliedOnce:
    def test_adapt_messages_called_exactly_once(self):
        tok = _RecordingChatTemplateTokenizer()
        messages = _messages()
        with patch(
            "paramem.models.loader.adapt_messages",
            side_effect=lambda msgs, t: msgs,
        ) as mock_adapt:
            render_chat_prompt(messages, tok)
        mock_adapt.assert_called_once_with(messages, tok)

    def test_adapted_output_is_what_gets_rendered_not_the_raw_input(self):
        """Proves render_chat_prompt renders adapt_messages's OUTPUT, not a
        second, independent copy of the raw input — a caller-visible way to
        catch a regression where adaptation is computed but discarded (or
        applied a second time downstream)."""
        tok = _RecordingChatTemplateTokenizer()
        folded = [{"role": "user", "content": "SYS\n\nUSR"}]
        with patch(
            "paramem.models.loader.adapt_messages",
            MagicMock(return_value=folded),
        ) as mock_adapt:
            render_chat_prompt(_messages(), tok)
        mock_adapt.assert_called_once()
        assert tok.calls[-1][0] == folded


class TestRenderChatPromptAddGenerationPromptForwarded:
    def test_default_true_is_forwarded(self):
        tok = _RecordingChatTemplateTokenizer()
        with patch("paramem.models.loader.adapt_messages", side_effect=lambda msgs, t: msgs):
            render_chat_prompt(_messages(), tok)
        _, kwargs = tok.calls[-1]
        assert kwargs["add_generation_prompt"] is True
        assert kwargs["tokenize"] is False

    def test_explicit_false_is_forwarded(self):
        tok = _RecordingChatTemplateTokenizer()
        with patch("paramem.models.loader.adapt_messages", side_effect=lambda msgs, t: msgs):
            render_chat_prompt(_messages(), tok, add_generation_prompt=False)
        _, kwargs = tok.calls[-1]
        assert kwargs["add_generation_prompt"] is False
