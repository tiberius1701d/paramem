"""Contract tests for the serving-path prompts.

The prompt that :func:`~paramem.server.prompts.serving_system_prompt`
returns is passed to the LLM alongside retrieved context. A primer that
told the model to "speak naturally as if you simply remember" caused the
model to confabulate personal facts on empty/partial context (untrained
adapter + "Where do I live?" -> "New York City").

The fix that removed that primer added a positive anti-confabulation
directive instead. These tests lock the file on disk. They are structural
(string-level) — a full LLM-compliance contract would require a live model
and is out of scope here.
"""

import re
from pathlib import Path

import pytest

from paramem.server.prompts import (
    intent_classifier_prompt,
    recall_selection_prompt,
    serving_system_prompt,
)

PROMPT_FILE = Path("configs/prompts/serving_system.txt")


@pytest.fixture
def prompt_text() -> str:
    assert PROMPT_FILE.exists(), f"{PROMPT_FILE} missing — required by contract"
    return PROMPT_FILE.read_text()


class TestServingSystemPromptFile:
    def test_confabulation_primer_removed(self, prompt_text: str):
        """The primer that caused a past confabulation regression must
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
        """A word budget replaces the old "1-2 sentences" instruction.
        Pinned by pattern, not exact prose — the numbers are
        calibration-tunable."""
        assert re.search(r"\b\d+\s+words\b", prompt_text) is not None

    def test_prompt_carries_summary_ordering_guidance(self, prompt_text: str):
        """Summaries/recall should front-load the most important facts, so
        a reply is still useful if the response-length cap cuts it short."""
        lower = prompt_text.lower()
        assert "most important" in lower

    def test_serving_system_prompt_excludes_classifier_content(self):
        """The serving system prompt lives in its own file — it must not
        carry the intent classifier's PERSONAL/COMMAND/GENERAL labels."""
        head = serving_system_prompt()
        assert "PERSONAL" not in head
        assert "COMMAND" not in head

    def test_serving_system_prompt_excludes_recall_selection_content(self):
        """The serving system prompt must not carry the date-selection
        stage's instructions — that lives in its own file too."""
        head = serving_system_prompt()
        assert "date-selection stage" not in head
        assert '{"all": true}' not in head

    def test_intent_classifier_prompt_is_its_own_file(self):
        """The classifier prompt is a standalone file — it must not carry
        the recall date-selection content appended after it in the
        pre-split source."""
        classifier = intent_classifier_prompt()
        assert "PERSONAL" in classifier
        assert "date-selection stage" not in classifier

    def test_recall_selection_prompt_returns_selection_content(self):
        """The recall date-selection prompt carries the rules and few-shot
        examples, with none of the classifier section's content."""
        selection = recall_selection_prompt()
        assert "date-selection stage" in selection
        assert '{"all": true}' in selection
        assert "PERSONAL" not in selection
        assert "COMMAND" not in selection

    def test_recall_selection_examples_carry_own_dates(self):
        """Each few-shot example in the recall-selection prompt states its
        own labeled example date, so no example is anchored to a
        hardcoded 'Today' that could collide with the real injected
        Today line at inference time."""
        selection = recall_selection_prompt()
        assert re.search(r"Example \(today is \w+, \d{4}-\d{2}-\d{2}\):", selection) is not None
        # Regression guard: a prior revision anchored every example to one
        # shared hardcoded "Today" line instead of each example stating its
        # own date — that shape must not come back.
        assert "Examples (Today is Thursday, 2026-08-06)" not in selection
