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
- ``ChatResult`` carries only the fields the serving path reads.
"""

from __future__ import annotations

import ast
import dataclasses
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from paramem.graph.prompts import prompt_overrides
from paramem.memory.store import MemoryStore as _MS
from paramem.server.config import ServerConfig
from paramem.server.inference import (
    _CAP_HIT_TOKEN_TOLERANCE,
    ChatResult,
    _base_model_answer,
    _generate_local_reply,
    _maybe_escalate,
    _probe_and_reason,
    _trim_incomplete_sentence,
)
from paramem.server.temporal import weekday_name
from tests._guard_utils import call_inside_context_manager, find_function


@pytest.fixture(autouse=True)
def _serving_system_prompt_override():
    """Every test in this module runs with a fixed base serving system
    prompt — none of them assert on its text, only on the identity/
    language prefix and the reasoning-turn body built around it, so a
    single override keeps every test isolated from the real
    ``configs/prompts/serving_system.txt`` content without threading a
    per-test fixture through 20 call sites."""
    with prompt_overrides({"serving_system.txt": "You are an assistant."}):
        yield


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


class TestChatResultFields:
    """``ChatResult`` carries only the fields the serving path actually
    reads — a field with zero readers anywhere in the codebase must not
    silently reappear."""

    def test_only_text_and_escalated_fields(self):
        assert {f.name for f in dataclasses.fields(ChatResult)} == {"text", "escalated"}


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
            staticmethod(lambda path, cached=False: {}),
        )
        monkeypatch.setattr(
            "paramem.server.inference.is_self_referential", lambda text, **kwargs: False
        )


class TestTokenBudgetPin(_PlanBuilder):
    """max_new_tokens is fed by config.inference.max_response_tokens — the
    plan's explicit gap: no test pinned this before the tail collapse."""

    def test_probe_and_reason_uses_configured_max_response_tokens(self, monkeypatch):
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

    def test_base_model_answer_uses_configured_max_response_tokens(self, monkeypatch):
        captured = {}

        def fake_generate(model, tokenizer, prompt, **kwargs):
            captured["max_new_tokens"] = kwargs.get("max_new_tokens")
            return "a plain answer."

        monkeypatch.setattr("paramem.server.inference.generate_answer", fake_generate)

        tokenizer = MagicMock()
        tokenizer.apply_chat_template = lambda msgs, **kwargs: "prompt"
        model = MagicMock()

        config = ServerConfig()
        config.inference.max_response_tokens = 64

        _base_model_answer(
            text="hello",
            history=None,
            model=model,
            tokenizer=tokenizer,
            config=config,
        )

        assert captured["max_new_tokens"] == 64

    def test_default_max_response_tokens_is_512(self, monkeypatch):
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

    def test_reply_well_under_the_cap_is_not_truncated(self, monkeypatch):
        monkeypatch.setattr(
            "paramem.server.inference.generate_answer", lambda *a, **kw: "a short answer."
        )
        tokenizer = self._tokenizer_with_ids(10)
        model = MagicMock()
        config = ServerConfig()
        config.inference.max_response_tokens = 64

        _reply, is_truncated = _generate_local_reply(
            "hello", None, model, tokenizer, config, speaker_id=None, language=None
        )
        assert is_truncated is False

    def test_reply_at_the_cap_is_truncated(self, monkeypatch):
        monkeypatch.setattr(
            "paramem.server.inference.generate_answer", lambda *a, **kw: "a cut-off reply"
        )
        tokenizer = self._tokenizer_with_ids(64)
        model = MagicMock()
        config = ServerConfig()
        config.inference.max_response_tokens = 64

        _reply, is_truncated = _generate_local_reply(
            "hello", None, model, tokenizer, config, speaker_id=None, language=None
        )
        assert is_truncated is True

    def test_reply_within_tolerance_of_the_cap_is_truncated(self, monkeypatch):
        """Within _CAP_HIT_TOKEN_TOLERANCE of the cap still counts —
        decode->re-encode round-trip drift must not under-detect a real
        cap hit."""
        monkeypatch.setattr(
            "paramem.server.inference.generate_answer", lambda *a, **kw: "a cut-off reply"
        )
        tokenizer = self._tokenizer_with_ids(64 - _CAP_HIT_TOKEN_TOLERANCE)
        model = MagicMock()
        config = ServerConfig()
        config.inference.max_response_tokens = 64

        _reply, is_truncated = _generate_local_reply(
            "hello", None, model, tokenizer, config, speaker_id=None, language=None
        )
        assert is_truncated is True

    def test_reply_just_outside_tolerance_is_not_truncated(self, monkeypatch):
        monkeypatch.setattr(
            "paramem.server.inference.generate_answer", lambda *a, **kw: "a complete reply."
        )
        tokenizer = self._tokenizer_with_ids(64 - _CAP_HIT_TOKEN_TOLERANCE - 1)
        model = MagicMock()
        config = ServerConfig()
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

    def test_disables_adapter_exactly_once_for_a_peft_model(self, monkeypatch):
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


class TestTemporalSelectionWiring(_PlanBuilder):
    """The date-group selection stage wired into ``_probe_and_reason``
    (:mod:`paramem.server.temporal`, :mod:`paramem.server.temporal_selection`).

    ``select_date_groups`` is stubbed via the module-level name
    ``paramem.server.inference.select_date_groups`` — the exact binding
    ``_probe_and_reason`` calls — so no test loads a model or touches the
    GPU. Bookkeeping-dependent tests use a real ``MemoryStore`` (never a
    ``MagicMock`` store, which would fabricate ``bookkeeping_for_key``).
    """

    @staticmethod
    def stub_probe_capturing(monkeypatch, fact_prefix: str = "fact about"):
        """Stub the grouped-probe primitive, returning a dict the caller
        can read ``["keys_by_adapter"]`` from after ``_probe_and_reason``
        returns — the exact set of keys the probe call received."""
        captured: dict = {}

        def fake_grouped(model, tokenizer, keys_by_adapter, **kwargs):
            captured["keys_by_adapter"] = {k: list(v) for k, v in keys_by_adapter.items()}
            results = {}
            for keys in keys_by_adapter.values():
                for k in keys:
                    results[k] = {"key": k, "fact_text": f"{fact_prefix} {k}", "confidence": 1.0}
            return results

        monkeypatch.setattr("paramem.memory.probe.probe_keys_grouped_by_adapter", fake_grouped)
        monkeypatch.setattr("paramem.models.loader.switch_adapter", lambda model, name: None)
        monkeypatch.setattr(
            "paramem.memory.store.MemoryStore.read_simhash_registry_from_disk",
            staticmethod(lambda path, cached=False: {}),
        )
        return captured

    @staticmethod
    def stub_generate_local_reply(monkeypatch, reply: str = "a reply."):
        """Stub ``_generate_local_reply`` and capture its first argument
        (the fully-assembled augmented prompt), so the exact context
        string reaching the reasoning generate can be asserted without a
        real model."""
        captured: dict = {}

        def fake(text, history, model, tokenizer, config, *, speaker_id, language):
            captured["augmented_text"] = text
            return reply, False

        monkeypatch.setattr("paramem.server.inference._generate_local_reply", fake)
        return captured

    def test_disabled_is_byte_identical_noop(self, monkeypatch):
        """temporal_selection_enabled=False: no clock read, no bookkeeping
        read, no selection call — the context string and the probe's
        keys_by_adapter are byte-identical to the pre-feature shape."""
        probed = self.stub_probe_capturing(monkeypatch)
        generated = self.stub_generate_local_reply(monkeypatch, reply="final answer.")

        def exploding_select(*args, **kwargs):
            raise AssertionError("select_date_groups must not be called when the stage is disabled")

        monkeypatch.setattr("paramem.server.inference.select_date_groups", exploding_select)

        tokenizer = MagicMock()
        tokenizer.apply_chat_template = lambda msgs, **kwargs: "prompt"
        model = self.make_model(["episodic"])

        config = ServerConfig()
        config.inference.temporal_selection_enabled = False

        memory_store = _MS(replay_enabled=False)
        # Bookkeeping deliberately left unseeded — the disabled stage must
        # never read it.
        plan = self.make_plan([("episodic", ["e1"])])

        result = _probe_and_reason(
            text="What do I like?",
            plan=plan,
            history=None,
            model=model,
            tokenizer=tokenizer,
            config=config,
            memory_store=memory_store,
        )

        assert probed["keys_by_adapter"] == {"episodic": ["e1"]}
        assert generated["augmented_text"] == (
            "What you know about the speaker:\n\n"
            "[Recent knowledge]\n- fact about e1\n\n"
            "Question: What do I like?"
        )
        assert result.text == "final answer."

    def test_memory_store_none_gate_skips_stage(self, monkeypatch):
        """``memory_store=None`` alone gates the stage off — even with
        ``temporal_selection_enabled`` at its True default and a
        non-empty plan. ``select_date_groups`` must never be called; the
        function then fails exactly where pre-feature code always would
        (``memory_store.probe`` needs a real store), never inside the
        temporal block — proof the gate, not an accident of call order,
        is what skipped the stage."""

        def exploding_select(*args, **kwargs):
            raise AssertionError("select_date_groups must not be called when memory_store is None")

        monkeypatch.setattr("paramem.server.inference.select_date_groups", exploding_select)

        tokenizer = MagicMock()
        tokenizer.apply_chat_template = lambda msgs, **kwargs: "prompt"
        model = self.make_model(["episodic"])

        config = ServerConfig()
        assert config.inference.temporal_selection_enabled is True

        plan = self.make_plan([("episodic", ["e1"])])

        with pytest.raises(AttributeError):
            _probe_and_reason(
                text="What do I like?",
                plan=plan,
                history=None,
                model=model,
                tokenizer=tokenizer,
                config=config,
                memory_store=None,
            )

    def test_selection_all_true_probe_set_identical_context_gains_headers(self, monkeypatch):
        """Selection stub returns ``all=True``: the probed key set is
        unchanged, and the only rendering delta is the Today header plus
        per-fact date headers."""
        from paramem.server.temporal_selection import DateSelection

        probed = self.stub_probe_capturing(monkeypatch)
        generated = self.stub_generate_local_reply(monkeypatch)

        def fake_select(text, date_by_key, *, model, tokenizer, config, today):
            return DateSelection(all=True, ranges=(), include_undated=True)

        monkeypatch.setattr("paramem.server.inference.select_date_groups", fake_select)

        tokenizer = MagicMock()
        tokenizer.apply_chat_template = lambda msgs, **kwargs: "prompt"
        model = self.make_model(["episodic"])

        config = ServerConfig()

        memory_store = _MS(replay_enabled=False)
        memory_store.set_bookkeeping(
            "e1",
            speaker_id="speaker0",
            relation_type="factual",
            first_seen="2026-08-01T09:00:00",
            last_seen="2026-08-01T09:00:00",
        )
        plan = self.make_plan([("episodic", ["e1"])])

        _probe_and_reason(
            text="What do you know about me?",
            plan=plan,
            history=None,
            model=model,
            tokenizer=tokenizer,
            config=config,
            memory_store=memory_store,
        )

        assert probed["keys_by_adapter"] == {"episodic": ["e1"]}

        today = date.today()
        expected_header = (
            f"Today is {weekday_name(today)}, {today.isoformat()}. What you know about the speaker:"
        )
        text = generated["augmented_text"]
        assert text.startswith(expected_header)
        e1_day = date(2026, 8, 1)
        assert f"On {weekday_name(e1_day)}, 2026-08-01:\n- fact about e1" in text

    def test_single_day_range_excludes_non_matching_and_undated_keys(self, monkeypatch):
        """Selection stub returns a single-day range: only the keys last
        seen on that day are probed; a key seen another day, and an
        undated key, are both excluded when ``include_undated=False``."""
        from paramem.server.temporal import DateWindow
        from paramem.server.temporal_selection import DateSelection

        probed = self.stub_probe_capturing(monkeypatch)
        self.stub_generate_local_reply(monkeypatch)

        def fake_select(text, date_by_key, *, model, tokenizer, config, today):
            return DateSelection(
                all=False,
                ranges=(DateWindow(start=date(2026, 8, 5), end=date(2026, 8, 5)),),
                include_undated=False,
            )

        monkeypatch.setattr("paramem.server.inference.select_date_groups", fake_select)

        tokenizer = MagicMock()
        tokenizer.apply_chat_template = lambda msgs, **kwargs: "prompt"
        model = self.make_model(["episodic"])

        config = ServerConfig()

        memory_store = _MS(replay_enabled=False)
        memory_store.set_bookkeeping(
            "e_match",
            speaker_id="speaker0",
            relation_type="factual",
            first_seen="2026-08-05T09:00:00",
            last_seen="2026-08-05T09:00:00",
        )
        memory_store.set_bookkeeping(
            "e_other_day",
            speaker_id="speaker0",
            relation_type="factual",
            first_seen="2026-08-01T09:00:00",
            last_seen="2026-08-01T09:00:00",
        )
        memory_store.set_bookkeeping(
            "e_undated",
            speaker_id="speaker0",
            relation_type="factual",
            first_seen="",
            last_seen="",
        )
        plan = self.make_plan([("episodic", ["e_match", "e_other_day", "e_undated"])])

        _probe_and_reason(
            text="What did we discuss on August 5th?",
            plan=plan,
            history=None,
            model=model,
            tokenizer=tokenizer,
            config=config,
            memory_store=memory_store,
        )

        assert probed["keys_by_adapter"] == {"episodic": ["e_match"]}

    def test_single_day_range_includes_undated_when_flagged(self, monkeypatch):
        """Same range, ``include_undated=True``: the undated key now
        survives alongside the matching dated key."""
        from paramem.server.temporal import DateWindow
        from paramem.server.temporal_selection import DateSelection

        probed = self.stub_probe_capturing(monkeypatch)
        self.stub_generate_local_reply(monkeypatch)

        def fake_select(text, date_by_key, *, model, tokenizer, config, today):
            return DateSelection(
                all=False,
                ranges=(DateWindow(start=date(2026, 8, 5), end=date(2026, 8, 5)),),
                include_undated=True,
            )

        monkeypatch.setattr("paramem.server.inference.select_date_groups", fake_select)

        tokenizer = MagicMock()
        tokenizer.apply_chat_template = lambda msgs, **kwargs: "prompt"
        model = self.make_model(["episodic"])

        config = ServerConfig()

        memory_store = _MS(replay_enabled=False)
        memory_store.set_bookkeeping(
            "e_match",
            speaker_id="speaker0",
            relation_type="factual",
            first_seen="2026-08-05T09:00:00",
            last_seen="2026-08-05T09:00:00",
        )
        memory_store.set_bookkeeping(
            "e_undated",
            speaker_id="speaker0",
            relation_type="factual",
            first_seen="",
            last_seen="",
        )
        plan = self.make_plan([("episodic", ["e_match", "e_undated"])])

        _probe_and_reason(
            text="What did we discuss on August 5th, and what's my job?",
            plan=plan,
            history=None,
            model=model,
            tokenizer=tokenizer,
            config=config,
            memory_store=memory_store,
        )

        kba = probed["keys_by_adapter"]
        assert set(kba["episodic"]) == {"e_match", "e_undated"}

    def test_zero_survivors_skips_probe_and_escalation_plain_reply(self, monkeypatch):
        """A selection range matching zero keys at all: no probe call, no
        HA/cloud/base-model call, and the reasoning turn runs directly off
        the deterministic nothing-in-period note. Plain (non-[ESCALATE])
        reply flows straight through as the result text."""
        from paramem.server.temporal import DateWindow
        from paramem.server.temporal_selection import DateSelection

        def exploding_probe(self, *args, **kwargs):
            raise AssertionError("MemoryStore.probe must not be called when zero keys survive")

        monkeypatch.setattr("paramem.memory.store.MemoryStore.probe", exploding_probe)

        for name in ("_escalate_to_ha_agent", "answer_via_cloud", "_base_model_answer"):

            def exploding(*args, _name=name, **kwargs):
                raise AssertionError(f"{_name} must not be called on the zero-survivor path")

            monkeypatch.setattr(f"paramem.server.inference.{name}", exploding)

        generated = self.stub_generate_local_reply(monkeypatch, reply="Nothing to add.")

        def fake_select(text, date_by_key, *, model, tokenizer, config, today):
            return DateSelection(
                all=False,
                ranges=(DateWindow(start=date(2026, 8, 10), end=date(2026, 8, 10)),),
                include_undated=False,
            )

        monkeypatch.setattr("paramem.server.inference.select_date_groups", fake_select)

        tokenizer = MagicMock()
        tokenizer.apply_chat_template = lambda msgs, **kwargs: "prompt"
        model = self.make_model(["episodic"])

        config = ServerConfig()

        memory_store = _MS(replay_enabled=False)
        memory_store.set_bookkeeping(
            "e1",
            speaker_id="speaker0",
            relation_type="factual",
            first_seen="2026-08-01T09:00:00",
            last_seen="2026-08-01T09:00:00",
        )
        plan = self.make_plan([("episodic", ["e1"])])

        result = _probe_and_reason(
            text="What did we discuss on August 10th?",
            plan=plan,
            history=None,
            model=model,
            tokenizer=tokenizer,
            config=config,
            memory_store=memory_store,
        )

        text = generated["augmented_text"]
        assert "Nothing was recorded for the requested period." in text
        assert "2026-08-01" in text
        assert result.text == "Nothing to add."

    def test_zero_survivors_escalate_tag_never_reaches_cloud_or_base_model(self, monkeypatch):
        """Same zero-survivor setup, but the local model escalates anyway
        (``[ESCALATE] ...``): HA is stubbed unreachable, ``answer_via_cloud``
        is invoked (receiving ``is_personal=True``) but stubbed to return
        ``None`` exactly as the ``block`` cloud-mode gate does in
        production, ``_base_model_answer`` is never reached, and the result
        is the pre-tag / fallback text."""
        from paramem.server.temporal import DateWindow
        from paramem.server.temporal_selection import DateSelection

        def exploding_probe(self, *args, **kwargs):
            raise AssertionError("MemoryStore.probe must not be called when zero keys survive")

        monkeypatch.setattr("paramem.memory.store.MemoryStore.probe", exploding_probe)

        def exploding_base_model(*args, **kwargs):
            raise AssertionError("_base_model_answer must not be called on the zero-survivor path")

        monkeypatch.setattr("paramem.server.inference._base_model_answer", exploding_base_model)
        monkeypatch.setattr("paramem.server.inference._escalate_to_ha_agent", lambda *a, **kw: None)
        monkeypatch.setattr("paramem.server.inference.is_self_referential", lambda *a, **kw: False)

        cloud_calls = []

        def fake_answer_via_cloud(text, cloud_agent, config, **kwargs):
            cloud_calls.append(kwargs)
            return None

        monkeypatch.setattr("paramem.server.inference.answer_via_cloud", fake_answer_via_cloud)

        self.stub_generate_local_reply(
            monkeypatch, reply="Intro sentence. [ESCALATE] what did we discuss then"
        )

        def fake_select(text, date_by_key, *, model, tokenizer, config, today):
            return DateSelection(
                all=False,
                ranges=(DateWindow(start=date(2026, 8, 10), end=date(2026, 8, 10)),),
                include_undated=False,
            )

        monkeypatch.setattr("paramem.server.inference.select_date_groups", fake_select)

        tokenizer = MagicMock()
        tokenizer.apply_chat_template = lambda msgs, **kwargs: "prompt"
        model = self.make_model(["episodic"])

        config = ServerConfig()

        memory_store = _MS(replay_enabled=False)
        memory_store.set_bookkeeping(
            "e1",
            speaker_id="speaker0",
            relation_type="factual",
            first_seen="2026-08-01T09:00:00",
            last_seen="2026-08-01T09:00:00",
        )
        plan = self.make_plan([("episodic", ["e1"])])

        result = _probe_and_reason(
            text="What did we discuss on August 10th?",
            plan=plan,
            history=None,
            model=model,
            tokenizer=tokenizer,
            config=config,
            memory_store=memory_store,
            is_personal=True,
        )

        assert len(cloud_calls) == 1
        assert cloud_calls[0]["is_personal"] is True
        assert result.text == "Intro sentence."

    def test_mixed_zero_dated_matches_with_undated_survivors_probes_and_renders_both(
        self, monkeypatch
    ):
        """Ranges + include_undated=True + zero dated matches + undated
        keys present: the probe runs for the undated keys only, and the
        context carries BOTH the nothing-in-period note and the undated
        facts."""
        from paramem.server.temporal import DateWindow
        from paramem.server.temporal_selection import DateSelection

        probed = self.stub_probe_capturing(monkeypatch)
        generated = self.stub_generate_local_reply(monkeypatch)

        def fake_select(text, date_by_key, *, model, tokenizer, config, today):
            return DateSelection(
                all=False,
                ranges=(DateWindow(start=date(2026, 8, 10), end=date(2026, 8, 10)),),
                include_undated=True,
            )

        monkeypatch.setattr("paramem.server.inference.select_date_groups", fake_select)

        tokenizer = MagicMock()
        tokenizer.apply_chat_template = lambda msgs, **kwargs: "prompt"
        model = self.make_model(["episodic"])

        config = ServerConfig()

        memory_store = _MS(replay_enabled=False)
        memory_store.set_bookkeeping(
            "e_dated",
            speaker_id="speaker0",
            relation_type="factual",
            first_seen="2026-08-01T09:00:00",
            last_seen="2026-08-01T09:00:00",
        )
        memory_store.set_bookkeeping(
            "e_undated",
            speaker_id="speaker0",
            relation_type="factual",
            first_seen="",
            last_seen="",
        )
        plan = self.make_plan([("episodic", ["e_dated", "e_undated"])])

        _probe_and_reason(
            text="What did we discuss on August 10th, and what's my job?",
            plan=plan,
            history=None,
            model=model,
            tokenizer=tokenizer,
            config=config,
            memory_store=memory_store,
        )

        assert probed["keys_by_adapter"] == {"episodic": ["e_undated"]}
        text = generated["augmented_text"]
        assert "Nothing was recorded for the requested period." in text
        assert "2026-08-01" in text
        assert "- fact about e_undated" in text

    def test_bookkeeping_none_and_int_last_seen_land_undated_without_raising(self, monkeypatch):
        """A bookkeeping record with ``last_seen=None`` or an int — the
        shape real disk-splatting can produce — lands in the undated
        group and raises nothing."""
        from paramem.server.temporal_selection import DateSelection

        probed = self.stub_probe_capturing(monkeypatch)
        self.stub_generate_local_reply(monkeypatch)

        def fake_select(text, date_by_key, *, model, tokenizer, config, today):
            assert {key for key, day in date_by_key.items() if day is None} == {"e_none", "e_int"}
            return DateSelection(all=True, ranges=(), include_undated=True)

        monkeypatch.setattr("paramem.server.inference.select_date_groups", fake_select)

        tokenizer = MagicMock()
        tokenizer.apply_chat_template = lambda msgs, **kwargs: "prompt"
        model = self.make_model(["episodic"])

        config = ServerConfig()

        memory_store = _MS(replay_enabled=False)
        memory_store.set_bookkeeping(
            "e_none",
            speaker_id="speaker0",
            relation_type="factual",
            first_seen="",
            last_seen=None,
        )
        memory_store.set_bookkeeping(
            "e_int",
            speaker_id="speaker0",
            relation_type="factual",
            first_seen="",
            last_seen=12345,
        )
        plan = self.make_plan([("episodic", ["e_none", "e_int"])])

        result = _probe_and_reason(
            text="What do you know about me?",
            plan=plan,
            history=None,
            model=model,
            tokenizer=tokenizer,
            config=config,
            memory_store=memory_store,
        )

        assert probed["keys_by_adapter"] == {"episodic": ["e_none", "e_int"]}
        assert result.text == "a reply."

    def test_dated_facts_render_once_per_tier_in_tier_order(self, monkeypatch):
        """Each dated fact appears exactly once, under its date header,
        within its tier section — and tier order (procedural → episodic →
        semantic) is preserved."""
        from paramem.server.temporal_selection import DateSelection

        probed = self.stub_probe_capturing(monkeypatch)
        generated = self.stub_generate_local_reply(monkeypatch)

        def fake_select(text, date_by_key, *, model, tokenizer, config, today):
            return DateSelection(all=True, ranges=(), include_undated=True)

        monkeypatch.setattr("paramem.server.inference.select_date_groups", fake_select)

        tokenizer = MagicMock()
        tokenizer.apply_chat_template = lambda msgs, **kwargs: "prompt"
        model = self.make_model(["procedural", "episodic", "semantic"])

        config = ServerConfig()

        memory_store = _MS(replay_enabled=False)
        memory_store.set_bookkeeping(
            "p1",
            speaker_id="speaker0",
            relation_type="preference",
            first_seen="2026-08-02T09:00:00",
            last_seen="2026-08-02T09:00:00",
        )
        memory_store.set_bookkeeping(
            "e1",
            speaker_id="speaker0",
            relation_type="factual",
            first_seen="2026-08-02T09:00:00",
            last_seen="2026-08-02T09:00:00",
        )
        memory_store.set_bookkeeping(
            "s1",
            speaker_id="speaker0",
            relation_type="factual",
            first_seen="2026-08-01T09:00:00",
            last_seen="2026-08-01T09:00:00",
        )
        plan = self.make_plan(
            [
                ("procedural", ["p1"]),
                ("episodic", ["e1"]),
                ("semantic", ["s1"]),
            ]
        )

        _probe_and_reason(
            text="What do you know about me?",
            plan=plan,
            history=None,
            model=model,
            tokenizer=tokenizer,
            config=config,
            memory_store=memory_store,
        )

        assert probed["keys_by_adapter"] == {
            "procedural": ["p1"],
            "episodic": ["e1"],
            "semantic": ["s1"],
        }
        text = generated["augmented_text"]

        # Tier order preserved.
        proc_idx = text.index("[Behavioral preferences]")
        epi_idx = text.index("[Recent knowledge]")
        sem_idx = text.index("[Consolidated knowledge]")
        assert proc_idx < epi_idx < sem_idx

        # Each dated fact appears exactly once, under its own date header,
        # within its own tier section.
        assert text.count("fact about p1") == 1
        assert text.count("fact about e1") == 1
        assert text.count("fact about s1") == 1
        aug_02 = date(2026, 8, 2)
        aug_01 = date(2026, 8, 1)
        assert f"On {weekday_name(aug_02)}, 2026-08-02:\n- fact about p1" in text
        assert f"On {weekday_name(aug_02)}, 2026-08-02:\n- fact about e1" in text
        assert f"On {weekday_name(aug_01)}, 2026-08-01:\n- fact about s1" in text

    def test_fail_open_selection_marks_the_info_log(self, monkeypatch, caplog):
        """The date-group-selection INFO log appends ``" (fail-open)"``
        when the selection came back with ``fail_open=True``, and carries
        no such marker for a genuine model ``all`` verdict — the log line
        is what lets an operator distinguish "the model chose everything"
        from "selection broke and we fell back to everything"."""
        import logging

        from paramem.server.temporal_selection import DateSelection

        self.stub_probe_capturing(monkeypatch)
        self.stub_generate_local_reply(monkeypatch)

        def fake_select(text, date_by_key, *, model, tokenizer, config, today):
            return DateSelection(all=True, ranges=(), include_undated=True, fail_open=True)

        monkeypatch.setattr("paramem.server.inference.select_date_groups", fake_select)

        tokenizer = MagicMock()
        tokenizer.apply_chat_template = lambda msgs, **kwargs: "prompt"
        model = self.make_model(["episodic"])

        config = ServerConfig()

        memory_store = _MS(replay_enabled=False)
        memory_store.set_bookkeeping(
            "e1",
            speaker_id="speaker0",
            relation_type="factual",
            first_seen="2026-08-01T09:00:00",
            last_seen="2026-08-01T09:00:00",
        )
        plan = self.make_plan([("episodic", ["e1"])])

        with caplog.at_level(logging.INFO, logger="paramem.server.inference"):
            _probe_and_reason(
                text="What do you know about me?",
                plan=plan,
                history=None,
                model=model,
                tokenizer=tokenizer,
                config=config,
                memory_store=memory_store,
            )

        selection_records = [
            r.getMessage() for r in caplog.records if "Date-group selection" in r.getMessage()
        ]
        assert len(selection_records) == 1
        assert "(fail-open)" in selection_records[0]

    def test_genuine_all_selection_does_not_mark_the_info_log(self, monkeypatch, caplog):
        """The mirror case: a genuine model ``all`` verdict
        (``fail_open=False``) leaves the INFO log without the marker."""
        import logging

        from paramem.server.temporal_selection import DateSelection

        self.stub_probe_capturing(monkeypatch)
        self.stub_generate_local_reply(monkeypatch)

        def fake_select(text, date_by_key, *, model, tokenizer, config, today):
            return DateSelection(all=True, ranges=(), include_undated=True, fail_open=False)

        monkeypatch.setattr("paramem.server.inference.select_date_groups", fake_select)

        tokenizer = MagicMock()
        tokenizer.apply_chat_template = lambda msgs, **kwargs: "prompt"
        model = self.make_model(["episodic"])

        config = ServerConfig()

        memory_store = _MS(replay_enabled=False)
        memory_store.set_bookkeeping(
            "e1",
            speaker_id="speaker0",
            relation_type="factual",
            first_seen="2026-08-01T09:00:00",
            last_seen="2026-08-01T09:00:00",
        )
        plan = self.make_plan([("episodic", ["e1"])])

        with caplog.at_level(logging.INFO, logger="paramem.server.inference"):
            _probe_and_reason(
                text="What do you know about me?",
                plan=plan,
                history=None,
                model=model,
                tokenizer=tokenizer,
                config=config,
                memory_store=memory_store,
            )

        selection_records = [
            r.getMessage() for r in caplog.records if "Date-group selection" in r.getMessage()
        ]
        assert len(selection_records) == 1
        assert "(fail-open)" not in selection_records[0]


class TestRenderTierFactsDateGrouping:
    """``_render_tier_facts``'s date-grouping in isolation — no probe, no
    model, no store; "dated" is derived from ``date_by_key`` being
    non-``None``, not a separate flag."""

    def test_no_date_map_is_the_flat_bullet_list(self):
        from paramem.server.inference import _render_tier_facts

        facts = [("k1", "fact one"), ("k2", "fact two")]
        assert _render_tier_facts(facts) == "- fact one\n- fact two"

    def test_one_tier_mixed_dated_and_undated_undated_listed_first(self):
        """A single tier carrying both an undated and a dated fact:
        the undated bullet renders first, ungrouped, ahead of the date
        header."""
        from paramem.server.inference import _render_tier_facts

        facts = [("k1", "dated fact"), ("k2", "undated fact")]
        day = date(2026, 8, 5)
        date_by_key = {"k1": day, "k2": None}
        rendered = _render_tier_facts(facts, date_by_key=date_by_key)
        assert rendered == f"- undated fact\nOn {weekday_name(day)}, 2026-08-05:\n- dated fact"

    def test_two_distinct_dates_in_one_tier_rendered_descending(self):
        """Two facts in one tier on different dates: the more recent date
        heads the section, per the ``sorted(..., reverse=True)`` contract."""
        from paramem.server.inference import _render_tier_facts

        older = date(2026, 8, 1)
        newer = date(2026, 8, 5)
        facts = [("k1", "older fact"), ("k2", "newer fact")]
        date_by_key = {"k1": older, "k2": newer}
        rendered = _render_tier_facts(facts, date_by_key=date_by_key)
        assert rendered == (
            f"On {weekday_name(newer)}, 2026-08-05:\n- newer fact\n"
            f"On {weekday_name(older)}, 2026-08-01:\n- older fact"
        )
