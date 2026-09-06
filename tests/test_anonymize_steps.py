"""``ask_speaker_anchor`` — the ANCHOR call's non-ASCII value rendering —
``scan_values``/``render_scan_section`` — the SCAN call's one model
call, its keyword-resolution keep/revert/drop rules, and its rendered
skeleton — and ``render_call_prompt``, the one chat-wrapping render every
step function in this module (and the anonymizer gate tool) shares.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from paramem.cloud import anonymize_steps as anonymize_steps_module
from paramem.cloud.anonymize_steps import (
    ScanFailed,
    ask_speaker_anchor,
    render_call_prompt,
    render_scan_section,
    scan_values,
)
from paramem.config.taxonomy import ScrubCategory
from paramem.models.loader import render_chat_prompt
from tests.anonymizer_doubles import basic_category


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


# ---------------------------------------------------------------------------
# scan_values — the SCAN call's one model call and its keyword resolution.
# ---------------------------------------------------------------------------


def _stub_scan_call(monkeypatch, raw: str) -> list[str]:
    """Stub the SCAN call's two chokepoints — ``render_chat_prompt`` (a
    fixed short rendered prompt) and ``generate_answer`` (the reply
    *raw*) — never a model or GPU. Returns the list ``generate_answer``
    call count is recorded onto (append per call), so a test can assert
    the call happened exactly once.
    """
    calls: list[str] = []

    def _fake_render_chat_prompt(messages, tokenizer, add_generation_prompt=True):
        return "rendered"

    def _fake_generate_answer(model, tokenizer, prompt, *, max_new_tokens, temperature, seed=None):
        calls.append(prompt)
        return raw

    monkeypatch.setattr(anonymize_steps_module, "render_chat_prompt", _fake_render_chat_prompt)
    monkeypatch.setattr(anonymize_steps_module, "generate_answer", _fake_generate_answer)
    return calls


def _scan(monkeypatch, raw: str, *, categories: tuple[ScrubCategory, ...], payload: str = "text"):
    _stub_scan_call(monkeypatch, raw)
    return scan_values(
        payload,
        MagicMock(),
        MagicMock(),
        categories=categories,
        section="{keywords}{text}",
        system_prompt="system",
        token_envelope=8192,
    )


class TestScanValuesOneModelCall:
    def test_scan_values_issues_exactly_one_generate_call(self, monkeypatch) -> None:
        calls = _stub_scan_call(monkeypatch, json.dumps({"mapping": {}}))
        scan_values(
            "text",
            MagicMock(),
            MagicMock(),
            categories=(basic_category("Person"),),
            section="{keywords}{text}",
            system_prompt="system",
            token_envelope=8192,
        )
        assert len(calls) == 1


class TestScanValuesParsesViaJsonEnvelope:
    def test_a_markdown_fenced_reply_still_parses(self, monkeypatch) -> None:
        raw = "```json\n" + json.dumps({"mapping": {"Alex": "Person"}}) + "\n```"
        results, dropped, _raw, _tokens = _scan(
            monkeypatch, raw, categories=(basic_category("Person"),), payload="Alex went home."
        )
        assert results[0].values == ("Alex",)
        assert dropped == ()


class TestScanValuesRaisesScanFailed:
    def test_a_bare_list_reply_raises(self, monkeypatch) -> None:
        raw = json.dumps(["Alex", "Person"])
        try:
            _scan(monkeypatch, raw, categories=(basic_category("Person"),))
            raised = False
        except ScanFailed:
            raised = True
        assert raised

    def test_a_mapping_with_a_non_string_value_raises(self, monkeypatch) -> None:
        raw = json.dumps({"mapping": {"Alex": 1}})
        try:
            _scan(monkeypatch, raw, categories=(basic_category("Person"),))
            raised = False
        except ScanFailed:
            raised = True
        assert raised

    def test_a_reply_missing_the_mapping_key_raises(self, monkeypatch) -> None:
        raw = json.dumps({"values": {"Alex": "Person"}})
        try:
            _scan(monkeypatch, raw, categories=(basic_category("Person"),))
            raised = False
        except ScanFailed:
            raised = True
        assert raised

    def test_scan_failed_carries_the_raw_reply_and_one_call_tokens_entry(self, monkeypatch) -> None:
        raw = "not json at all"
        try:
            _scan(monkeypatch, raw, categories=(basic_category("Person"),))
            exc = None
        except ScanFailed as e:
            exc = e
        assert exc is not None
        assert exc.raw == raw
        assert len(exc.call_tokens) == 1


class TestScanValuesKeywordResolutionIsCanonicalFoldOnly:
    def test_a_differently_cased_and_spaced_keyword_still_resolves(self, monkeypatch) -> None:
        """``canonical()`` folds case and blank runs, so ``" person "``
        names the ``Person`` row exactly as ``"Person"`` does."""
        raw = json.dumps({"mapping": {"Alex": " person "}})
        results, dropped, _raw, _tokens = _scan(
            monkeypatch, raw, categories=(basic_category("Person"),), payload="Alex went home."
        )
        assert results[0].values == ("Alex",)
        assert dropped == ()

    def test_a_keyword_that_is_merely_a_substring_of_a_row_does_not_resolve(
        self, monkeypatch
    ) -> None:
        """Exact canonical-fold match only — ``"Pers"`` is not ``"Person"``,
        so it names no row at all (``unknown_word``), never a fuzzy match."""
        raw = json.dumps({"mapping": {"Alex": "Pers"}})
        results, dropped, _raw, _tokens = _scan(
            monkeypatch, raw, categories=(basic_category("Person"),), payload="Alex went home."
        )
        assert results[0].values == ()
        assert dropped[0]["reason"] == "unknown_word"


class TestScanValuesKeepRevertAndDropReasons:
    def test_a_value_naming_an_active_row_is_kept_under_that_row(self, monkeypatch) -> None:
        raw = json.dumps({"mapping": {"Berlin": "City"}})
        results, dropped, _raw, _tokens = _scan(
            monkeypatch, raw, categories=(basic_category("City"),), payload="Berlin is nice."
        )
        assert results[0].category.prefix == "City"
        assert results[0].values == ("Berlin",)
        assert dropped == ()

    def test_a_value_naming_an_inactive_row_is_reverted_with_the_row_as_category(
        self, monkeypatch
    ) -> None:
        """``Person`` names a real schema row, but only ``City`` is active
        here — the value reverts, ``category`` names the out-of-scope row."""
        raw = json.dumps({"mapping": {"Alex": "Person"}})
        results, dropped, _raw, _tokens = _scan(
            monkeypatch, raw, categories=(basic_category("City"),), payload="Alex went home."
        )
        assert results[0].values == ()
        [entry] = dropped
        assert entry == {
            "category": "Person",
            "side": "scan",
            "text": "Alex",
            "reason": "reverted",
            "word": None,
        }

    def test_an_unrecognised_keyword_drops_with_the_models_own_word_and_empty_category(
        self, monkeypatch
    ) -> None:
        raw = json.dumps({"mapping": {"Alex": "Nonexistentkind"}})
        results, dropped, _raw, _tokens = _scan(
            monkeypatch, raw, categories=(basic_category("Person"),), payload="Alex went home."
        )
        assert results[0].values == ()
        [entry] = dropped
        assert entry == {
            "category": "",
            "side": "scan",
            "text": "Alex",
            "reason": "unknown_word",
            "word": "Nonexistentkind",
        }

    def test_a_speaker_id_shaped_value_drops_regardless_of_its_keyword(self, monkeypatch) -> None:
        raw = json.dumps({"mapping": {"speaker0": "Person"}})
        results, dropped, _raw, _tokens = _scan(
            monkeypatch, raw, categories=(basic_category("Person"),), payload="speaker0 said hi."
        )
        assert results[0].values == ()
        [entry] = dropped
        assert entry == {
            "category": "Person",
            "side": "scan",
            "text": "speaker0",
            "reason": "speaker_id",
            "word": None,
        }


class TestScanValuesDeduplicatesExactSurfaceDuplicates:
    def test_a_duplicate_json_key_in_the_raw_reply_collapses_to_one_value(
        self, monkeypatch
    ) -> None:
        """A literal duplicate key in the raw JSON text collapses at parse
        time (standard JSON-object semantics) — ``scan_values`` never sees
        or preserves two entries for the same surface."""
        raw = '{"mapping": {"Alex": "Person", "Alex": "City"}}'
        results, _dropped, _raw, _tokens = _scan(
            monkeypatch,
            raw,
            categories=(basic_category("Person"), basic_category("City")),
            payload="Alex went home.",
        )
        person, city = results
        assert person.values == ()
        assert city.values == ("Alex",)


class TestScanValuesOneResultPerActiveCategoryInOrder:
    def test_every_active_category_gets_a_result_including_an_empty_one(self, monkeypatch) -> None:
        raw = json.dumps({"mapping": {"Berlin": "City"}})
        categories = (basic_category("Person"), basic_category("City"), basic_category("Org"))
        results, _dropped, _raw, _tokens = _scan(
            monkeypatch, raw, categories=categories, payload="Berlin is nice."
        )
        assert [r.category.prefix for r in results] == ["Person", "City", "Org"]
        assert results[0].values == ()
        assert results[1].values == ("Berlin",)
        assert results[2].values == ()

    def test_kept_values_are_ordered_by_first_occurrence_in_the_payload_not_reply_order(
        self, monkeypatch
    ) -> None:
        payload = "Riley called. Later, Alex called too."
        # Reply names Alex before Riley — the opposite of their payload order.
        raw = json.dumps({"mapping": {"Alex": "Person", "Riley": "Person"}})
        results, _dropped, _raw, _tokens = _scan(
            monkeypatch, raw, categories=(basic_category("Person"),), payload=payload
        )
        assert results[0].values == ("Riley", "Alex")


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
