"""Contract tests for the PA-path voice prompt.

The prompt that ``VoiceConfig.load_prompt()`` returns is passed to the
LLM alongside retrieved context. When it was primed with *"Speak
naturally as if you simply remember"*, the model confabulated personal
facts on empty/partial context (observed 2026-04-21: untrained adapter
+ "Where do I live?" → "New York City").

The no-hallucination fallback removed that primer and added a positive
anti-confabulation directive. These tests lock both sides in: the file on disk
AND the inline Python fallback in ``load_prompt()`` when the prompt file is
missing. They are structural (string-level) — a full LLM-compliance contract
would require a live model and is out of scope here.
"""

import re
from pathlib import Path

import pytest

from paramem.server.config import VoiceConfig

PROMPT_FILE = Path("configs/prompts/pa_voice.txt")


@pytest.fixture
def prompt_text() -> str:
    assert PROMPT_FILE.exists(), f"{PROMPT_FILE} missing — required by contract"
    return PROMPT_FILE.read_text()


class TestPaVoicePromptFile:
    def test_confabulation_primer_removed(self, prompt_text: str):
        """The specific primer that caused the 2026-04-21 regression must
        not come back under any casing."""
        lower = prompt_text.lower()
        assert "simply remember" not in lower
        assert "speak naturally" not in lower

    def test_anti_confabulation_directive_present(self, prompt_text: str):
        """Positive counter-prime: answer only from provided context."""
        lower = prompt_text.lower()
        assert "context" in lower
        assert "never invent" in lower or "do not invent" in lower

    def test_escalation_sentinel_preserved(self, prompt_text: str):
        """[ESCALATE] is how the PA path forwards to HA/cloud when context
        is empty. A future rewrite that drops it would silently break
        routing — guard it here."""
        assert "[ESCALATE]" in prompt_text

    def test_prompt_states_a_numeric_word_budget(self, prompt_text: str):
        """The ruled strengthening (truncation fix, item 4): a word budget
        replaces the old "1-2 sentences" instruction. Pinned by pattern,
        not exact prose — the numbers are calibration-tunable."""
        assert re.search(r"\b\d+\s+words\b", prompt_text) is not None

    def test_prompt_carries_summary_ordering_guidance(self, prompt_text: str):
        """Summaries/recall should front-load the most important facts, so
        a reply is still useful if the response-length cap cuts it short."""
        lower = prompt_text.lower()
        assert "most important" in lower

    def test_load_prompt_head_excludes_classifier_section(self, tmp_path):
        """Cheap insurance that the line-3 length-instruction edit did not
        disturb the ##---INTENT-CLASSIFIER-SECTION--- marker: load_prompt()
        returns the head only (no PERSONAL/COMMAND/GENERAL classifier
        labels leak into the PA system prompt), and
        load_intent_classifier_prompt() still finds its section."""
        vc = VoiceConfig(prompt_file=str(PROMPT_FILE))
        head = vc.load_prompt()
        assert "PERSONAL" not in head
        assert "COMMAND" not in head
        assert "##---INTENT-CLASSIFIER-SECTION---" not in head
        assert vc.load_intent_classifier_prompt() is not None


class TestVoiceConfigFallback:
    def test_fallback_does_not_prime_confabulation(self, tmp_path):
        """If the prompt file is unreadable and no inline override is
        set, ``load_prompt()`` returns a hardcoded fallback. It must not
        reintroduce the primer."""
        vc = VoiceConfig(prompt_file=str(tmp_path / "does-not-exist.txt"), system_prompt="")
        fallback = vc.load_prompt().lower()
        assert "simply remember" not in fallback
        assert "speak naturally" not in fallback

    def test_fallback_has_anti_confabulation_directive(self, tmp_path):
        vc = VoiceConfig(prompt_file=str(tmp_path / "does-not-exist.txt"), system_prompt="")
        fallback = vc.load_prompt().lower()
        assert "never invent" in fallback or "do not invent" in fallback

    def test_inline_override_still_honored(self, tmp_path):
        """Operators who set ``system_prompt`` explicitly still get their
        string verbatim — contract tests must not block that override."""
        vc = VoiceConfig(prompt_file=str(tmp_path / "nope.txt"), system_prompt="custom override")
        assert vc.load_prompt() == "custom override"
