"""Tests for probe dispatch and entry_fact_text.

Covers:
- entry_fact_text: basic conversion, predicate de-underscoring.
- probe_key success result carries fact_text == answer.

All tests are CPU-only — model and tokenizer are MagicMocks;
generate_answer is patched at the module boundary.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from paramem.training.entry_memory import entry_fact_text

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _qa_pair(
    key: str,
    question: str,
    answer: str,
    *,
    subject: str = "Alex",
    predicate: str = "lives_in",
    obj: str = "Heilbronn",
    speaker_id: str = "spk1",
    first_seen_cycle: int = 1,
) -> dict:
    return {
        "key": key,
        "question": question,
        "answer": answer,
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "speaker_id": speaker_id,
        "first_seen_cycle": first_seen_cycle,
    }


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


# ---------------------------------------------------------------------------
# probe_key (legacy) adds fact_text
# ---------------------------------------------------------------------------


class TestProbeKeyFormatFields:
    """probe_key success result must carry fact_text == answer."""

    def test_success_result_has_fact_text(self) -> None:
        model = _make_model_with_adapters("episodic")
        tokenizer = _make_tokenizer()

        qa_raw = json.dumps(
            {"key": "graph1", "question": "Where does Alex live?", "answer": "Heilbronn."}
        )

        with (
            patch(
                "paramem.evaluation.recall.generate_answer",
                return_value=qa_raw,
            ),
            patch("paramem.training.dataset._format_inference_prompt", return_value="prompt"),
        ):
            from paramem.training.indexed_memory import probe_key

            result = probe_key(model, tokenizer, "graph1")

        assert result is not None
        assert "failure_reason" not in result
        assert result["fact_text"] == result["answer"]
        assert result["fact_text"] == "Heilbronn."
