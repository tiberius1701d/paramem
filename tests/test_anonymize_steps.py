"""``ask_speaker_anchor`` — the ANCHOR call's non-ASCII value rendering —
``render_scan_section``/``_render_keywords`` — the SCAN call's rendered
skeleton — and ``render_call_prompt``, the one chat-wrapping render every
step function in this module (and the anonymizer gate tool) shares.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from paramem.cloud import anonymize_steps as anonymize_steps_module
from paramem.cloud.anonymize_steps import (
    _render_keywords,
    ask_speaker_anchor,
    render_call_prompt,
    render_scan_section,
)
from paramem.config.taxonomy import prefix_descriptions
from paramem.models.loader import render_chat_prompt


class TestAskSpeakerAnchorRendersNonAsciiValuesVerbatim:
    """The ANCHOR user prompt embeds *values* via
    ``json.dumps(list(values), ensure_ascii=False)`` — a non-ASCII
    candidate surface (e.g. a German name) must reach the model as the
    real character, never a ``\\uXXXX`` escape the model would have to
    echo back byte-for-byte to survive the answer's own
    ``v in values_set`` membership check.
    """

    def test_a_non_ascii_candidate_is_not_escaped_in_the_rendered_prompt(self, monkeypatch) -> None:
        captured: dict = {}

        def _fake_render_chat_prompt(messages, tokenizer, add_generation_prompt=True):
            captured["messages"] = messages
            return "irrelevant rendered text"

        monkeypatch.setattr(anonymize_steps_module, "render_chat_prompt", _fake_render_chat_prompt)

        ask_speaker_anchor(
            "text",
            MagicMock(),
            MagicMock(),
            values=["Straße"],
            speaker_id="speaker1",
            section="speaker={speaker_id} values={values} text={text}",
            system_prompt="system",
            # Too small to fit any real budget -> AnonymizeBudgetRefused
            # after render, so no generate() call is issued — the render
            # step (and the assertion below) still runs unconditionally
            # before the budget check.
            token_envelope=1,
        )

        user_content = captured["messages"][1]["content"]
        assert "Straße" in user_content
        assert "\\u00df" not in user_content


class TestRenderKeywords:
    """``_render_keywords`` renders one ``Prefix: description`` line per
    row of :func:`~paramem.config.taxonomy.prefix_descriptions`, in that
    function's order. The shipped content itself is pinned once, in
    ``tests/test_taxonomy.py::TestPrefixDescriptions`` — this test asserts
    only the rendering contract, not the content.
    """

    def test_matches_prefix_descriptions_joined_by_newline(self) -> None:
        assert _render_keywords() == "\n".join(
            f"{prefix}: {description}" for prefix, description in prefix_descriptions()
        )


class TestRenderScanSection:
    def test_renders_keywords_in_table_order_and_the_text_slot(self, monkeypatch) -> None:
        monkeypatch.setattr(
            anonymize_steps_module,
            "prefix_descriptions",
            lambda: (("Person", "a person"), ("City", "a city")),
        )
        rendered = render_scan_section("Keywords:\n{keywords}\n\nText:\n{text}", "hello world")
        assert rendered == "Keywords:\nPerson: a person\nCity: a city\n\nText:\nhello world"


# ---------------------------------------------------------------------------
# render_call_prompt — the one chat-wrapping render every step function in
# this module (and the anonymizer gate tool) shares.
# ---------------------------------------------------------------------------


class _StubChatTemplateTokenizer:
    """A tokenizer stub whose ``apply_chat_template`` deterministically
    joins ``role:content`` per message and appends a generation-prompt
    marker — real enough that :func:`~paramem.models.loader.
    supports_system_role`'s own probe call (which checks its own marker
    string survives the render) reports the system role as supported, so
    ``adapt_messages`` leaves the system/user pair unfolded.
    """

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        rendered = "\n".join(f"{m['role']}:{m['content']}" for m in messages)
        if add_generation_prompt:
            rendered += "\nassistant:"
        return rendered


class TestRenderCallPrompt:
    def test_builds_the_system_and_user_pair_and_matches_render_chat_prompt(self) -> None:
        """``render_call_prompt`` is not a second renderer — it builds
        exactly the ``[{"role": "system", ...}, {"role": "user", ...}]``
        pair and renders it through
        :func:`~paramem.models.loader.render_chat_prompt` with the
        generation prompt appended, so it must produce byte-identical
        output to calling ``render_chat_prompt`` directly on the same
        messages.
        """
        tokenizer = _StubChatTemplateTokenizer()

        result = render_call_prompt("system text", "user text", tokenizer)

        expected = render_chat_prompt(
            [
                {"role": "system", "content": "system text"},
                {"role": "user", "content": "user text"},
            ],
            tokenizer,
            add_generation_prompt=True,
        )
        assert result == expected
        assert "system:system text" in result
        assert "user:user text" in result
        assert result.endswith("assistant:")
