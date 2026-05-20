"""Tests for entry_fact_text.

Covers:
- entry_fact_text: basic conversion, predicate de-underscoring.

All tests are CPU-only — model and tokenizer are MagicMocks.
The QA-shape probe_key tests (TestProbeKeyFormatFields) were removed on
2026-05-20 when the QA shape was archived to archive/legacy_qa.py.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from paramem.memory.entry import entry_fact_text

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _entry_pair(
    key: str,
    subject: str,
    predicate: str,
    obj: str,
    *,
    speaker_id: str = "spk1",
    first_seen_cycle: int = 1,
) -> dict:
    return {
        "key": key,
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "speaker_id": speaker_id,
        "first_seen_cycle": first_seen_cycle,
    }


def _make_model_with_adapters(*names: str) -> MagicMock:
    """Return a MagicMock model with peft_config populated for given adapter names."""
    model = MagicMock()
    model.peft_config = {name: MagicMock() for name in names}
    return model


def _make_tokenizer() -> MagicMock:
    tok = MagicMock()
    tok.apply_chat_template.side_effect = lambda msgs, **_: json.dumps(msgs)
    return tok


# ---------------------------------------------------------------------------
# entry_fact_text
# ---------------------------------------------------------------------------


class TestEntryFactText:
    def test_basic_conversion(self) -> None:
        result = entry_fact_text(
            {"subject": "Alex", "predicate": "lives_in", "object": "Heilbronn"}
        )
        assert result == "Alex lives in Heilbronn"

    def test_no_underscores(self) -> None:
        result = entry_fact_text({"subject": "Alex", "predicate": "knows", "object": "Bob"})
        assert result == "Alex knows Bob"

    def test_multi_underscore_predicate(self) -> None:
        result = entry_fact_text(
            {"subject": "Alice", "predicate": "works_at_company", "object": "Acme"}
        )
        assert result == "Alice works at company Acme"

    def test_extra_whitespace_collapsed(self) -> None:
        # split() + join() collapses internal multi-space from multiple underscores
        result = entry_fact_text({"subject": "S", "predicate": "a__b", "object": "O"})
        # "_" → " ", " " → collapsed to one space
        assert "  " not in result
