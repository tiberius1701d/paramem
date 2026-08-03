"""Unit tests for the local reasoning generation tail's shaping contract.

Covers:
- ``_trim_incomplete_sentence``: the sentence-boundary trim applied to a
  reply that was cut mid-sentence by ``max_new_tokens`` — including the
  digit-preceded-period exclusion (decimals, list markers) and the
  German-closing-quote (U+201C) fix.
- Truncation detection lives in ``_generate_local_reply`` (the one place
  the fact is derivable) and is threaded to ``_maybe_escalate`` as
  ``is_truncated``; the trim applies ONLY when that flag is True — a
  complete reply is never mutated.
- ``_maybe_escalate``'s no-tag branch applying the trim conditionally, and
  the hops-exhausted branch NEVER applying it regardless of the flag
  (pre-tag text is complete by construction).
- Ordering: the trim never runs before ``detect_escalation``, so a
  complete ``[ESCALATE]`` tag is never eaten.
- ``_generate_local_reply``'s ``max_new_tokens`` is fed by
  ``config.inference.max_response_tokens`` (the one literal-free site) —
  exercised through both ``_probe_and_reason`` and ``_base_model_answer``.
- Structural guard: ``_generate_local_reply`` calls ``generate_answer``
  INSIDE ``with base_model_inference(...):`` — the primitive that disables
  the active PEFT adapter and gradient checkpointing for the duration.
"""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import MagicMock

from paramem.memory.store import MemoryStore as _MS
from paramem.server.config import ServerConfig, VoiceConfig
from paramem.server.inference import (
    _CAP_HIT_TOKEN_TOLERANCE,
    ChatResult,
    _base_model_answer,
    _generate_local_reply,
    _maybe_escalate,
    _probe_and_reason,
    _trim_incomplete_sentence,
)
from tests._guard_utils import call_inside_context_manager, find_function


def _voice_config(tmp_path, text: str = "You are an assistant.") -> VoiceConfig:
    """A VoiceConfig whose load_prompt() returns *text* verbatim, without
    touching the real configs/prompts/pa_voice.txt file (prompt_file must
    be explicitly empty — the dataclass default points at the real file,
    which would otherwise win over ``system_prompt``)."""
    return VoiceConfig(prompt_file="", system_prompt=text)


class TestTrimIncompleteSentence:
    """Byte-identical no-op on complete text; trims back to the last
    terminator otherwise. Callers gate on ``is_truncated`` — this class
    exercises the function's own contract in isolation."""

    def test_ends_on_period_byte_identical(self):
        text = "This is a complete sentence."
        assert _trim_incomplete_sentence(text) == text

    def test_ends_on_exclamation_byte_identical(self):
        text = "Wow, that's great!"
        assert _trim_incomplete_sentence(text) == text

    def test_ends_on_question_mark_byte_identical(self):
        text = "Are you sure?"
        assert _trim_incomplete_sentence(text) == text

    def test_closer_after_terminator_quote_byte_identical(self):
        text = 'He said "yes."'
        assert _trim_incomplete_sentence(text) == text

    def test_closer_after_terminator_paren_byte_identical(self):
        text = "(yes.)"
        assert _trim_incomplete_sentence(text) == text

    def test_mid_word_tail_trimmed_to_earlier_terminator(self):
        text = "First sentence. Second sent"
        assert _trim_incomplete_sentence(text) == "First sentence."

    def test_closer_immediately_after_terminator_is_retained(self):
        """A closer immediately after the terminator we trim back to is
        kept — the trim lands after the closer, not before it."""
        text = 'She said "great!" and then rambled on incomple'
        assert _trim_incomplete_sentence(text) == 'She said "great!"'

    def test_no_terminator_anywhere_byte_identical(self):
        text = "this just trails off with no punctuation at all"
        assert _trim_incomplete_sentence(text) == text

    def test_empty_string_byte_identical(self):
        assert _trim_incomplete_sentence("") == ""

    def test_whitespace_only_byte_identical(self):
        assert _trim_incomplete_sentence("   \n\t  ") == "   \n\t  "

    def test_terminator_as_first_character_is_pinned_exactly(self):
        """A terminator as the very first character is always retained —
        the slice always includes at least that (non-whitespace)
        character, so the function can never empty a reply. Exact pin
        (not just non-empty): there is no separate never-empty guard in
        the implementation — the slice's own construction is why."""
        text = ". trailing incomple"
        assert _trim_incomplete_sentence(text) == "."

    def test_decimal_period_is_not_a_boundary(self):
        """A '.' preceded by a digit (a decimal) is not a sentence
        terminator — trimming back to it would silently drop the rest of
        the number and everything after. With no OTHER terminator in the
        text, the whole reply is left untouched rather than mis-cut."""
        text = "The value is 3.5 percent, higher than expected"
        assert _trim_incomplete_sentence(text) == text

    def test_list_marker_period_is_not_a_boundary(self):
        """Same exclusion for an enumerated list marker ('3.')."""
        text = "Here are the items: 1. First 2. Second 3."
        assert _trim_incomplete_sentence(text) == text

    def test_decimal_period_is_skipped_in_favor_of_a_real_terminator(self):
        """A digit-preceded period is skipped when scanning backward, but
        a genuine terminator earlier in the text is still found and used."""
        text = "Done here. The value is 3.5 and then it trails"
        assert _trim_incomplete_sentence(text) == "Done here."

    def test_german_closing_quote_preserved_as_complete(self):
        """U+201C ('“') is German typography's CLOSING quote (opening is
        '„', U+201E) — a complete reply ending in „…“ must not be judged
        incomplete and lose its closing quote."""
        text = "Sie sagte „Ja, das stimmt.“"
        assert _trim_incomplete_sentence(text) == text

    def test_german_closing_quote_retained_when_trimming(self):
        text = "Sie sagte „Ja.“ Dann brach der Satz ab und"
        assert _trim_incomplete_sentence(text) == "Sie sagte „Ja.“"


class TestMaybeEscalateTrimApplication:
    """Trim application is gated on ``is_truncated`` — a complete reply is
    never touched, and the hops-exhausted branch is never trimmed at all."""

    def test_no_tag_branch_applies_the_trim_when_truncated(self):
        config = ServerConfig()
        response = "This is done. But this trails of"
        result = _maybe_escalate(response, config, is_truncated=True)
        assert result.text == "This is done."

    def test_no_tag_branch_leaves_response_untouched_when_not_truncated(self):
        """The default (``is_truncated=False``): a complete generate that
        merely lacks a terminator (e.g. a deliberate style choice) is
        never mutated — this is the fix for the class of defects that
        mutated healthy replies."""
        config = ServerConfig()
        response = "This is done. But this trails of"
        result = _maybe_escalate(response, config)
        assert result.text == response

    def test_hops_exhausted_branch_does_not_trim(self, monkeypatch):
        """With HA and cloud both returning None, the pre-tag text comes
        back byte-identical even when it lacks a terminator AND even when
        is_truncated=True — this is the pin for the documented branch
        justification (pre-tag text is complete by construction: the
        model reached the tag)."""
        config = ServerConfig()
        monkeypatch.setattr("paramem.server.inference._escalate_to_ha_agent", lambda *a, **kw: None)
        monkeypatch.setattr("paramem.server.inference.answer_via_cloud", lambda *a, **kw: None)
        monkeypatch.setattr("paramem.server.inference.is_self_referential", lambda *a, **kw: False)
        response = "Intro. Trailing partial [ESCALATE] some forwarded query"
        result = _maybe_escalate(response, config, is_truncated=True)
        assert result.text == "Intro. Trailing partial"

    def test_complete_escalate_tag_still_escalates(self, monkeypatch):
        """Order pin: the trim never runs before detect_escalation — a
        genuine [ESCALATE] tag routes through the escalation branch, not
        the trimmed no-tag return."""
        config = ServerConfig()
        sentinel = ChatResult(text="HA handled it", escalated=True)
        monkeypatch.setattr(
            "paramem.server.inference._escalate_to_ha_agent", lambda *a, **kw: sentinel
        )
        monkeypatch.setattr("paramem.server.inference.is_self_referential", lambda *a, **kw: False)
        response = "Let me check that. [ESCALATE] what's the weather"
        result = _maybe_escalate(response, config, is_truncated=True)
        assert result is sentinel

    def test_truncated_escalate_fragment_does_not_escalate_and_is_trimmed_when_truncated(self):
        """A truncated ``[ESCAL`` fragment is not matched by
        detect_escalation's exact ``find`` — it falls through to the
        no-tag branch; with is_truncated=True (the real production case —
        this fragment only exists because the cap cut mid-tag) it is
        subsumed by the trim along with the rest of the incomplete tail."""
        config = ServerConfig()
        response = "Here is a fact. And then it cuts off mid tag [ESCAL"
        result = _maybe_escalate(response, config, is_truncated=True)
        assert "[ESCAL" not in result.text
        assert result.text == "Here is a fact."

    def test_truncated_escalate_fragment_untouched_when_not_truncated(self):
        """Same fragment, is_truncated=False (default): no trim runs at
        all — the tag-detection miss is a separate, out-of-scope residual
        from this trim redesign (record-only, not fixed here)."""
        config = ServerConfig()
        response = "Here is a fact. And then it cuts off mid tag [ESCAL"
        result = _maybe_escalate(response, config)
        assert result.text == response


class _PlanBuilder:
    """Shared plan/model builders, mirroring
    tests/test_server.py::TestProbeAndReasonDispatch's pattern."""

    @staticmethod
    def make_plan(steps):
        from paramem.server.router import Intent, RoutingPlan, RoutingStep

        return RoutingPlan(
            steps=[RoutingStep(adapter_name=a, keys_to_probe=list(k)) for a, k in steps],
            strategy="direct",
            intent=Intent.PERSONAL,
        )

    @staticmethod
    def make_model(adapter_names):
        model = MagicMock()
        model.peft_config = {name: MagicMock() for name in adapter_names}
        return model

    @staticmethod
    def stub_probe(monkeypatch):
        def fake_grouped(model, tokenizer, keys_by_adapter, **kwargs):
            results = {}
            for keys in keys_by_adapter.values():
                for k in keys:
                    results[k] = {"key": k, "fact_text": f"fact about {k}", "confidence": 1.0}
            return results

        monkeypatch.setattr("paramem.memory.probe.probe_keys_grouped_by_adapter", fake_grouped)
        monkeypatch.setattr("paramem.models.loader.switch_adapter", lambda model, name: None)
        monkeypatch.setattr(
            "paramem.memory.store.MemoryStore.read_simhash_registry_from_disk",
            staticmethod(lambda path: {}),
        )
        monkeypatch.setattr(
            "paramem.server.inference.is_self_referential", lambda text, **kwargs: False
        )


class TestTokenBudgetPin(_PlanBuilder):
    """max_new_tokens is fed by config.inference.max_response_tokens — the
    plan's explicit gap: no test pinned this before the tail collapse."""

    def test_probe_and_reason_uses_configured_max_response_tokens(self, monkeypatch, tmp_path):
        self.stub_probe(monkeypatch)
        captured = {}

        def fake_generate(model, tokenizer, prompt, **kwargs):
            captured["max_new_tokens"] = kwargs.get("max_new_tokens")
            return "final answer."

        monkeypatch.setattr("paramem.server.inference.generate_answer", fake_generate)

        tokenizer = MagicMock()
        tokenizer.apply_chat_template = lambda msgs, **kwargs: "prompt"
        model = self.make_model(["episodic"])

        config = ServerConfig()
        config.voice = _voice_config(tmp_path)
        config.inference.max_response_tokens = 64

        plan = self.make_plan([("episodic", ["e1"])])

        _probe_and_reason(
            text="What do I like?",
            plan=plan,
            history=None,
            model=model,
            tokenizer=tokenizer,
            config=config,
            memory_store=_MS(replay_enabled=False),
        )

        assert captured["max_new_tokens"] == 64

    def test_base_model_answer_uses_configured_max_response_tokens(self, monkeypatch, tmp_path):
        captured = {}

        def fake_generate(model, tokenizer, prompt, **kwargs):
            captured["max_new_tokens"] = kwargs.get("max_new_tokens")
            return "a plain answer."

        monkeypatch.setattr("paramem.server.inference.generate_answer", fake_generate)

        tokenizer = MagicMock()
        tokenizer.apply_chat_template = lambda msgs, **kwargs: "prompt"
        model = MagicMock()

        config = ServerConfig()
        config.voice = _voice_config(tmp_path)
        config.inference.max_response_tokens = 64

        _base_model_answer(
            text="hello",
            history=None,
            model=model,
            tokenizer=tokenizer,
            config=config,
        )

        assert captured["max_new_tokens"] == 64

    def test_default_max_response_tokens_is_512(self, monkeypatch, tmp_path):
        """At ServerConfig() defaults (no override), the same capture sees
        512 — the shipped ceiling."""
        captured = {}

        def fake_generate(model, tokenizer, prompt, **kwargs):
            captured["max_new_tokens"] = kwargs.get("max_new_tokens")
            return "a plain answer."

        monkeypatch.setattr("paramem.server.inference.generate_answer", fake_generate)

        tokenizer = MagicMock()
        tokenizer.apply_chat_template = lambda msgs, **kwargs: "prompt"
        model = MagicMock()

        config = ServerConfig()
        config.voice = _voice_config(tmp_path)

        _base_model_answer(
            text="hello",
            history=None,
            model=model,
            tokenizer=tokenizer,
            config=config,
        )

        assert captured["max_new_tokens"] == 512


class TestGenerateLocalReplyTruncationDetection:
    """Truncation detection is derived once, in _generate_local_reply, by
    re-tokenizing the decoded reply against config.inference.max_response_tokens
    within _CAP_HIT_TOKEN_TOLERANCE."""

    @staticmethod
    def _tokenizer_with_ids(n_ids: int) -> MagicMock:
        """A tokenizer whose exact-path re-tokenization reports *n_ids*
        input ids for any text — drives estimate_tokens's EXACT branch."""
        tokenizer = MagicMock()
        tokenizer.apply_chat_template = lambda msgs, **kwargs: "prompt"
        tokenizer.side_effect = lambda text, add_special_tokens=False: {
            "input_ids": list(range(n_ids))
        }
        return tokenizer

    def test_reply_well_under_the_cap_is_not_truncated(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "paramem.server.inference.generate_answer", lambda *a, **kw: "a short answer."
        )
        tokenizer = self._tokenizer_with_ids(10)
        model = MagicMock()
        config = ServerConfig()
        config.voice = _voice_config(tmp_path)
        config.inference.max_response_tokens = 64

        _reply, is_truncated = _generate_local_reply(
            "hello", None, model, tokenizer, config, speaker_id=None, language=None
        )
        assert is_truncated is False

    def test_reply_at_the_cap_is_truncated(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "paramem.server.inference.generate_answer", lambda *a, **kw: "a cut-off reply"
        )
        tokenizer = self._tokenizer_with_ids(64)
        model = MagicMock()
        config = ServerConfig()
        config.voice = _voice_config(tmp_path)
        config.inference.max_response_tokens = 64

        _reply, is_truncated = _generate_local_reply(
            "hello", None, model, tokenizer, config, speaker_id=None, language=None
        )
        assert is_truncated is True

    def test_reply_within_tolerance_of_the_cap_is_truncated(self, monkeypatch, tmp_path):
        """Within _CAP_HIT_TOKEN_TOLERANCE of the cap still counts —
        decode->re-encode round-trip drift must not under-detect a real
        cap hit."""
        monkeypatch.setattr(
            "paramem.server.inference.generate_answer", lambda *a, **kw: "a cut-off reply"
        )
        tokenizer = self._tokenizer_with_ids(64 - _CAP_HIT_TOKEN_TOLERANCE)
        model = MagicMock()
        config = ServerConfig()
        config.voice = _voice_config(tmp_path)
        config.inference.max_response_tokens = 64

        _reply, is_truncated = _generate_local_reply(
            "hello", None, model, tokenizer, config, speaker_id=None, language=None
        )
        assert is_truncated is True

    def test_reply_just_outside_tolerance_is_not_truncated(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "paramem.server.inference.generate_answer", lambda *a, **kw: "a complete reply."
        )
        tokenizer = self._tokenizer_with_ids(64 - _CAP_HIT_TOKEN_TOLERANCE - 1)
        model = MagicMock()
        config = ServerConfig()
        config.voice = _voice_config(tmp_path)
        config.inference.max_response_tokens = 64

        _reply, is_truncated = _generate_local_reply(
            "hello", None, model, tokenizer, config, speaker_id=None, language=None
        )
        assert is_truncated is False


class TestGenerateLocalReplyStructuralGuard:
    """AST-level guard, sharing tests/_guard_utils.py's helper with
    tests/test_extraction_pipeline_guard.py's base_model_inference family
    (one implementation of the AST walk, not a per-file copy), plus a
    behavioural companion exercising the PEFT branch."""

    def test_generate_answer_is_called_inside_base_model_inference(self):
        repo_root = Path(__file__).resolve().parent.parent
        inference_file = repo_root / "paramem" / "server" / "inference.py"
        tree = ast.parse(inference_file.read_text())

        target = find_function(tree, "_generate_local_reply")
        assert target is not None, "_generate_local_reply not found in paramem/server/inference.py"

        assert call_inside_context_manager(target, "base_model_inference", "generate_answer"), (
            "_generate_local_reply must call generate_answer INSIDE "
            "`with base_model_inference(model):` — reasoning must run on the "
            "base weights regardless of the last-active adapter. Moving the "
            "generate_answer call outside the block (even if the with-block "
            "is still present) must fail this check."
        )

    def test_disables_adapter_exactly_once_for_a_peft_model(self, monkeypatch, tmp_path):
        from peft import PeftModel

        monkeypatch.setattr(
            "paramem.server.inference.generate_answer", lambda *a, **kw: "an answer."
        )

        model = MagicMock(spec=PeftModel)
        model.is_gradient_checkpointing = False
        model.disable_adapter = MagicMock()
        model.gradient_checkpointing_disable = MagicMock()
        model.gradient_checkpointing_enable = MagicMock()

        tokenizer = MagicMock()
        tokenizer.apply_chat_template = lambda msgs, **kwargs: "prompt"

        config = ServerConfig()
        config.voice = _voice_config(tmp_path)

        reply, _is_truncated = _generate_local_reply(
            "hello",
            None,
            model,
            tokenizer,
            config,
            speaker_id=None,
            language=None,
        )

        assert reply == "an answer."
        model.disable_adapter.assert_called_once()
        model.disable_adapter.return_value.__enter__.assert_called_once()
