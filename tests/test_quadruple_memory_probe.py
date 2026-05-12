"""Tests for the format-aware probe dispatch added in Phase 3b.

Covers:
- quad_fact_text: basic conversion, predicate de-underscoring.
- probe_keys_grouped_by_adapter with formats_by_adapter:
  - mixed quad/QA adapters dispatch to the right probe function.
  - switch_adapter called once per adapter group.
  - success results carry format + fact_text.
  - formats_by_adapter=None → all-QA (back-compat).
- probe_keys_from_disk with formats_by_adapter:
  - quad keyed_pairs.json → quad-shaped result + fact_text.
  - QA keyed_pairs.json → QA-shaped result + fact_text == answer.
  - missing file → None.
  - missing key → None.
  - formats_by_adapter=None → all-QA (back-compat).
  - legacy top-level episodic fallback works with fmt='quad'.
- probe_key success result carries format='qa' and fact_text == answer.

All tests are CPU-only — model and tokenizer are MagicMocks;
generate_answer is patched at the module boundary.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from paramem.training.indexed_memory import (
    probe_keys_from_disk,
    probe_keys_grouped_by_adapter,
)
from paramem.training.keyed_pairs_io import write_keyed_pairs, write_keyed_pairs_quad
from paramem.training.quadruple_memory import quad_fact_text

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _qa_pair(
    key: str,
    question: str,
    answer: str,
    *,
    source_subject: str = "Alex",
    source_predicate: str = "lives_in",
    source_object: str = "Heilbronn",
    speaker_id: str = "spk1",
    first_seen_cycle: int = 1,
) -> dict:
    return {
        "key": key,
        "question": question,
        "answer": answer,
        "source_subject": source_subject,
        "source_predicate": source_predicate,
        "source_object": source_object,
        "speaker_id": speaker_id,
        "first_seen_cycle": first_seen_cycle,
    }


def _quad_pair(
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
# quad_fact_text
# ---------------------------------------------------------------------------


class TestQuadFactText:
    def test_basic_conversion(self) -> None:
        result = quad_fact_text({"subject": "Alex", "predicate": "lives_in", "object": "Heilbronn"})
        assert result == "Alex lives in Heilbronn"

    def test_no_underscores(self) -> None:
        result = quad_fact_text({"subject": "Alex", "predicate": "knows", "object": "Bob"})
        assert result == "Alex knows Bob"

    def test_multi_underscore_predicate(self) -> None:
        result = quad_fact_text(
            {"subject": "Alice", "predicate": "works_at_company", "object": "Acme"}
        )
        assert result == "Alice works at company Acme"

    def test_extra_whitespace_collapsed(self) -> None:
        # split() + join() collapses internal multi-space from multiple underscores
        result = quad_fact_text({"subject": "S", "predicate": "a__b", "object": "O"})
        # "_" → " ", " " → collapsed to one space
        assert "  " not in result


# ---------------------------------------------------------------------------
# probe_key adds format + fact_text
# ---------------------------------------------------------------------------


class TestProbeKeyFormatFields:
    """probe_key success result must carry format='qa' and fact_text == answer."""

    def test_success_result_has_format_and_fact_text(self) -> None:
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
        assert result["format"] == "qa"
        assert result["fact_text"] == result["answer"]
        assert result["fact_text"] == "Heilbronn."


# ---------------------------------------------------------------------------
# probe_keys_grouped_by_adapter — format dispatch
# ---------------------------------------------------------------------------


class TestProbeKeysGroupedByAdapterFormatDispatch:
    def _qa_raw(self, key: str, question: str, answer: str) -> str:
        return json.dumps({"key": key, "question": question, "answer": answer})

    def _quad_raw(self, key: str, subject: str, predicate: str, obj: str) -> str:
        return json.dumps({"key": key, "subject": subject, "predicate": predicate, "object": obj})

    def test_formats_by_adapter_none_uses_qa_for_all(self) -> None:
        """formats_by_adapter=None → all adapters probed via probe_key (QA, back-compat)."""
        model = _make_model_with_adapters("episodic")
        tokenizer = _make_tokenizer()
        qa_raw = self._qa_raw("graph1", "Where does Alex live?", "Heilbronn.")

        with (
            patch(
                "paramem.evaluation.recall.generate_answer",
                return_value=qa_raw,
            ),
            patch("paramem.training.dataset._format_inference_prompt", return_value="p"),
            patch("paramem.models.loader.switch_adapter") as mock_switch,
        ):
            results = probe_keys_grouped_by_adapter(
                model, tokenizer, {"episodic": ["graph1"]}, formats_by_adapter=None
            )

        assert "graph1" in results
        r = results["graph1"]
        assert r is not None
        assert "failure_reason" not in r
        assert r["format"] == "qa"
        assert r["fact_text"] == "Heilbronn."
        mock_switch.assert_called_once_with(model, "episodic")

    def test_quad_adapter_uses_probe_quad(self) -> None:
        """formats_by_adapter={'episodic': 'quad'} → probe_quad called for that adapter."""
        model = _make_model_with_adapters("episodic")
        tokenizer = _make_tokenizer()

        quad_result = {
            "key": "graph1",
            "subject": "Alex",
            "predicate": "lives_in",
            "object": "Heilbronn",
            "confidence": 1.0,
            "raw_output": self._quad_raw("graph1", "Alex", "lives_in", "Heilbronn"),
        }

        with (
            patch(
                "paramem.training.quadruple_memory.probe_quad",
                return_value=quad_result,
            ) as mock_probe_quad,
            patch("paramem.models.loader.switch_adapter") as mock_switch,
        ):
            results = probe_keys_grouped_by_adapter(
                model,
                tokenizer,
                {"episodic": ["graph1"]},
                formats_by_adapter={"episodic": "quad"},
            )

        mock_probe_quad.assert_called_once()
        mock_switch.assert_called_once_with(model, "episodic")

        r = results["graph1"]
        assert r is not None
        assert "failure_reason" not in r
        assert r["format"] == "quad"
        assert r["fact_text"] == "Alex lives in Heilbronn"
        assert r["subject"] == "Alex"
        assert r["predicate"] == "lives_in"
        assert r["object"] == "Heilbronn"

    def test_mixed_adapters_dispatch_correctly(self) -> None:
        """Mixed formats: episodic=quad uses probe_quad, semantic=qa uses probe_key."""
        model = _make_model_with_adapters("episodic", "semantic")
        tokenizer = _make_tokenizer()

        quad_result = {
            "key": "graph1",
            "subject": "Alex",
            "predicate": "lives_in",
            "object": "Heilbronn",
            "confidence": 1.0,
            "raw_output": self._quad_raw("graph1", "Alex", "lives_in", "Heilbronn"),
        }
        qa_raw = self._qa_raw("graph2", "What is Alex's job?", "Developer.")

        switch_calls: list[str] = []

        def _fake_switch(m, name):
            switch_calls.append(name)

        with (
            patch(
                "paramem.training.quadruple_memory.probe_quad",
                return_value=quad_result,
            ) as mock_probe_quad,
            patch(
                "paramem.evaluation.recall.generate_answer",
                return_value=qa_raw,
            ),
            patch("paramem.training.dataset._format_inference_prompt", return_value="p"),
            patch("paramem.models.loader.switch_adapter", side_effect=_fake_switch),
        ):
            results = probe_keys_grouped_by_adapter(
                model,
                tokenizer,
                {"episodic": ["graph1"], "semantic": ["graph2"]},
                formats_by_adapter={"episodic": "quad", "semantic": "qa"},
            )

        # switch_adapter called once per group
        assert switch_calls == ["episodic", "semantic"]

        r_quad = results["graph1"]
        assert r_quad is not None
        assert "failure_reason" not in r_quad
        assert r_quad["format"] == "quad"
        assert r_quad["fact_text"] == "Alex lives in Heilbronn"

        r_qa = results["graph2"]
        assert r_qa is not None
        assert "failure_reason" not in r_qa
        assert r_qa["format"] == "qa"
        assert r_qa["fact_text"] == "Developer."

        mock_probe_quad.assert_called_once()

    def test_missing_adapter_maps_to_none(self) -> None:
        """Adapter not in peft_config → all keys map to None (existing behaviour)."""
        model = _make_model_with_adapters()  # no adapters
        tokenizer = _make_tokenizer()

        results = probe_keys_grouped_by_adapter(
            model,
            tokenizer,
            {"episodic": ["graph1", "graph2"]},
            formats_by_adapter={"episodic": "quad"},
        )
        assert results["graph1"] is None
        assert results["graph2"] is None

    def test_quad_failure_result_preserved(self) -> None:
        """A probe_quad failure result (failure_reason in result) is returned as-is."""
        model = _make_model_with_adapters("episodic")
        tokenizer = _make_tokenizer()

        failure = {"raw_output": "garbage", "failure_reason": "parse_failure"}

        with (
            patch("paramem.training.quadruple_memory.probe_quad", return_value=failure),
            patch("paramem.models.loader.switch_adapter"),
        ):
            results = probe_keys_grouped_by_adapter(
                model,
                tokenizer,
                {"episodic": ["graph1"]},
                formats_by_adapter={"episodic": "quad"},
            )

        r = results["graph1"]
        assert r is not None
        assert r["failure_reason"] == "parse_failure"
        # failure results must NOT have format/fact_text attached
        assert "format" not in r
        assert "fact_text" not in r


# ---------------------------------------------------------------------------
# probe_keys_from_disk — format dispatch
# ---------------------------------------------------------------------------


class TestProbeKeysFromDiskFormatDispatch:
    def test_formats_by_adapter_none_uses_qa(self, tmp_path: Path) -> None:
        """formats_by_adapter=None → QA reader; result has format='qa'."""
        pairs = [_qa_pair("graph1", "Where does Alex live?", "Heilbronn.")]
        _write_keyed_pairs_maybe_mkdir(
            tmp_path / "episodic" / "keyed_pairs.json", pairs, create_parents=True
        )

        results = probe_keys_from_disk(tmp_path, {"episodic": ["graph1"]}, formats_by_adapter=None)

        r = results["graph1"]
        assert r is not None
        assert r["format"] == "qa"
        assert r["fact_text"] == "Heilbronn."
        assert r["answer"] == "Heilbronn."

    def test_quad_keyed_pairs_read_as_quad(self, tmp_path: Path) -> None:
        """formats_by_adapter={'episodic': 'quad'} reads quad file → quad-shaped result."""
        pairs = [_quad_pair("graph1", "Alex", "lives_in", "Heilbronn")]
        (tmp_path / "episodic").mkdir(parents=True)
        write_keyed_pairs_quad(tmp_path / "episodic" / "keyed_pairs.json", pairs)

        results = probe_keys_from_disk(
            tmp_path,
            {"episodic": ["graph1"]},
            formats_by_adapter={"episodic": "quad"},
        )

        r = results["graph1"]
        assert r is not None
        assert r["format"] == "quad"
        assert r["fact_text"] == "Alex lives in Heilbronn"
        assert r["subject"] == "Alex"
        assert r["predicate"] == "lives_in"
        assert r["object"] == "Heilbronn"
        assert r["confidence"] == 1.0

    def test_mixed_formats_two_tiers(self, tmp_path: Path) -> None:
        """episodic=quad, semantic=qa read from their respective files correctly."""
        quad_pairs = [_quad_pair("graph1", "Alex", "lives_in", "Heilbronn")]
        qa_pairs = [_qa_pair("graph2", "What is Alex's job?", "Developer.")]

        (tmp_path / "episodic").mkdir(parents=True)
        write_keyed_pairs_quad(tmp_path / "episodic" / "keyed_pairs.json", quad_pairs)
        _write_keyed_pairs_maybe_mkdir(
            tmp_path / "semantic" / "keyed_pairs.json", qa_pairs, create_parents=True
        )

        results = probe_keys_from_disk(
            tmp_path,
            {"episodic": ["graph1"], "semantic": ["graph2"]},
            formats_by_adapter={"episodic": "quad", "semantic": "qa"},
        )

        r_quad = results["graph1"]
        assert r_quad is not None
        assert r_quad["format"] == "quad"
        assert r_quad["fact_text"] == "Alex lives in Heilbronn"

        r_qa = results["graph2"]
        assert r_qa is not None
        assert r_qa["format"] == "qa"
        assert r_qa["fact_text"] == "Developer."

    def test_missing_file_returns_none_for_group(self, tmp_path: Path) -> None:
        """Missing keyed_pairs.json → all keys in the group map to None."""
        results = probe_keys_from_disk(
            tmp_path,
            {"episodic": ["graph1", "graph2"]},
            formats_by_adapter={"episodic": "quad"},
        )
        assert results["graph1"] is None
        assert results["graph2"] is None

    def test_missing_key_returns_none(self, tmp_path: Path) -> None:
        """File present but key absent → that key maps to None."""
        pairs = [_quad_pair("graph1", "Alex", "lives_in", "Heilbronn")]
        (tmp_path / "episodic").mkdir(parents=True)
        write_keyed_pairs_quad(tmp_path / "episodic" / "keyed_pairs.json", pairs)

        results = probe_keys_from_disk(
            tmp_path,
            {"episodic": ["graph1", "graph99"]},
            formats_by_adapter={"episodic": "quad"},
        )
        assert results["graph1"] is not None
        assert results["graph99"] is None

    def test_legacy_top_level_fallback_for_episodic_quad(self, tmp_path: Path) -> None:
        """Legacy top-level keyed_pairs.json for episodic works with fmt='quad'."""
        pairs = [_quad_pair("graph1", "Alex", "lives_in", "Heilbronn")]
        # Write at top-level path (no episodic subdir)
        write_keyed_pairs_quad(tmp_path / "keyed_pairs.json", pairs)

        results = probe_keys_from_disk(
            tmp_path,
            {"episodic": ["graph1"]},
            formats_by_adapter={"episodic": "quad"},
        )

        r = results["graph1"]
        assert r is not None
        assert r["format"] == "quad"
        assert r["fact_text"] == "Alex lives in Heilbronn"

    def test_qa_fact_text_equals_answer(self, tmp_path: Path) -> None:
        """QA result: fact_text must equal answer (exact string match)."""
        pairs = [_qa_pair("graph1", "What does Alex do?", "Software developer at Acme Corp.")]
        _write_keyed_pairs_maybe_mkdir(
            tmp_path / "episodic" / "keyed_pairs.json", pairs, create_parents=True
        )

        results = probe_keys_from_disk(
            tmp_path,
            {"episodic": ["graph1"]},
            formats_by_adapter={"episodic": "qa"},
        )

        r = results["graph1"]
        assert r is not None
        assert r["fact_text"] == r["answer"]
        assert r["fact_text"] == "Software developer at Acme Corp."


# ---------------------------------------------------------------------------
# Helper for write_keyed_pairs with create_parents
# ---------------------------------------------------------------------------


def _write_keyed_pairs_maybe_mkdir(
    path: Path, pairs: list[dict], *, create_parents: bool = False
) -> None:
    """Write QA keyed pairs, optionally creating parent directories."""
    if create_parents:
        path.parent.mkdir(parents=True, exist_ok=True)
    write_keyed_pairs(path, pairs)
