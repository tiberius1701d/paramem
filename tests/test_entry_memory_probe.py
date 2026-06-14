"""Tests for entry_fact_text and probe abort cooperative behaviour.

Covers:
- entry_fact_text: basic conversion, predicate de-underscoring.
- probe_keys_grouped_by_adapter: should_abort stops the loop early and
  returns partial results without raising.
- WeightMemorySource.probe: forwards should_abort to the underlying function.
- DiskMemorySource.probe: accepts and ignores should_abort (CPU path).

All tests are CPU-only — model and tokenizer are MagicMocks.
The QA-shape probe_key tests (TestProbeKeyFormatFields) were removed on
2026-05-20 when the QA shape was archived to archive/legacy_qa.py.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

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


# ---------------------------------------------------------------------------
# probe_keys_grouped_by_adapter — should_abort cooperative abort
# ---------------------------------------------------------------------------


def _make_model_with_adapters(*names: str) -> MagicMock:
    """Return a MagicMock that behaves like a minimal PeftModel."""
    model = MagicMock()
    model.peft_config = {name: MagicMock() for name in names}
    return model


class TestProbeAbortCooperative:
    """probe_keys_grouped_by_adapter stops between adapter groups on should_abort."""

    def _fake_probe_entries(self, model, tokenizer, stubs, **_):
        """Minimal stand-in for probe_entries that echoes stubs back as successes."""
        for stub in stubs:
            yield (
                stub,
                {
                    "key": stub["key"],
                    "subject": "Alice",
                    "predicate": "lives_in",
                    "object": "Berlin",
                },
            )

    def test_no_abort_returns_all_groups(self) -> None:
        """When should_abort always returns False, all adapter groups are probed."""
        model = _make_model_with_adapters("episodic", "semantic")

        with (
            patch(
                "paramem.training.recall_eval.probe_entries",
                side_effect=self._fake_probe_entries,
            ),
            patch("paramem.models.loader.switch_adapter"),
        ):
            from paramem.memory.probe import probe_keys_grouped_by_adapter

            result = probe_keys_grouped_by_adapter(
                model,
                MagicMock(),
                {"episodic": ["graph1"], "semantic": ["graph2"]},
                batch_size=1,
                should_abort=lambda: False,
            )

        assert "graph1" in result
        assert "graph2" in result

    def test_abort_before_second_group_returns_partial(self) -> None:
        """When should_abort fires before the second group, only the first is returned."""
        model = _make_model_with_adapters("episodic", "semantic")

        call_count = {"n": 0}

        def _abort_on_second() -> bool:
            # Return False for the first group, True for the second.
            call_count["n"] += 1
            return call_count["n"] > 1

        with (
            patch(
                "paramem.training.recall_eval.probe_entries",
                side_effect=self._fake_probe_entries,
            ),
            patch("paramem.models.loader.switch_adapter"),
        ):
            from paramem.memory.probe import probe_keys_grouped_by_adapter

            result = probe_keys_grouped_by_adapter(
                model,
                MagicMock(),
                {"episodic": ["graph1"], "semantic": ["graph2"]},
                batch_size=1,
                should_abort=_abort_on_second,
            )

        # graph1 (first group) should be present; graph2 (second group) absent.
        assert "graph1" in result
        assert "graph2" not in result

    def test_abort_before_first_group_returns_empty(self) -> None:
        """When should_abort fires immediately, an empty dict is returned."""
        model = _make_model_with_adapters("episodic", "semantic")

        with (
            patch(
                "paramem.training.recall_eval.probe_entries",
                side_effect=self._fake_probe_entries,
            ),
            patch("paramem.models.loader.switch_adapter"),
        ):
            from paramem.memory.probe import probe_keys_grouped_by_adapter

            result = probe_keys_grouped_by_adapter(
                model,
                MagicMock(),
                {"episodic": ["graph1"], "semantic": ["graph2"]},
                batch_size=1,
                should_abort=lambda: True,
            )

        assert result == {}

    def test_should_abort_none_is_backward_compatible(self) -> None:
        """Omitting should_abort (None) produces the same result as lambda: False."""
        model = _make_model_with_adapters("episodic")

        with (
            patch(
                "paramem.training.recall_eval.probe_entries",
                side_effect=self._fake_probe_entries,
            ),
            patch("paramem.models.loader.switch_adapter"),
        ):
            from paramem.memory.probe import probe_keys_grouped_by_adapter

            result_no_abort = probe_keys_grouped_by_adapter(
                model,
                MagicMock(),
                {"episodic": ["graph1"]},
                batch_size=1,
                should_abort=None,
            )
            result_false_abort = probe_keys_grouped_by_adapter(
                model,
                MagicMock(),
                {"episodic": ["graph1"]},
                batch_size=1,
                should_abort=lambda: False,
            )

        assert set(result_no_abort.keys()) == set(result_false_abort.keys())


class TestWeightMemorySourceAbort:
    """WeightMemorySource.probe forwards should_abort to probe_keys_grouped_by_adapter."""

    def test_should_abort_forwarded(self) -> None:
        """should_abort kwarg is passed through to the underlying probe function."""
        from paramem.memory.source import WeightMemorySource

        model = _make_model_with_adapters("episodic")
        source = WeightMemorySource(model, MagicMock(), batch_size=1)

        captured: dict = {}

        def _fake_probe_grouped(m, tok, keys_by_adapter, *, should_abort=None, **kwargs):
            captured["should_abort"] = should_abort
            return {}

        with patch(
            "paramem.memory.probe.probe_keys_grouped_by_adapter",
            side_effect=_fake_probe_grouped,
        ):
            _abort_fn = lambda: False  # noqa: E731
            source.probe({"episodic": ["graph1"]}, should_abort=_abort_fn)

        assert captured.get("should_abort") is _abort_fn


class TestDiskMemorySourceAbortIgnored:
    """DiskMemorySource.probe accepts should_abort but ignores it (CPU path)."""

    def test_should_abort_accepted_and_ignored(self, tmp_path) -> None:
        """DiskMemorySource.probe returns normally even when should_abort=True."""
        from paramem.memory.source import DiskMemorySource

        source = DiskMemorySource(tmp_path)
        # No graph.json files — should return empty results for any keys.
        # The key test is that passing should_abort=True does NOT raise and
        # does NOT short-circuit (DiskMemorySource is CPU-bound and fast).
        abort_called: list[bool] = []

        def _always_abort() -> bool:
            abort_called.append(True)
            return True

        result = source.probe({"episodic": []}, should_abort=_always_abort)
        # Empty keys list → empty result, no error.
        assert result == {}
        # The abort callable was NOT called (DiskMemorySource ignores it).
        assert abort_called == []
