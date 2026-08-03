"""Tests for entry_fact_text and probe abort cooperative behaviour.

Covers:
- entry_fact_text: SPO assembly at the render boundary — the stored predicate
  identity form is already space-form prose ('_' was folded to a space by
  canonical() upstream; '-' survives). No further substitution happens here.
- probe_keys_grouped_by_adapter: should_abort stops the loop early and
  returns partial results without raising.
- WeightMemorySource.probe: forwards should_abort to the underlying function.
- DiskMemorySource.probe: accepts and ignores should_abort (CPU path).
- build_memory_source: the single mode → MemorySource selection contract.

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
) -> dict:
    return {
        "key": key,
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "speaker_id": speaker_id,
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
    def test_basic_assembly(self) -> None:
        """The stored identity predicate (already space-form) is used as-is."""
        result = entry_fact_text(
            {"subject": "Alex", "predicate": "lives in", "object": "Heilbronn"}
        )
        assert result == "Alex lives in Heilbronn"

    def test_single_word_predicate(self) -> None:
        result = entry_fact_text({"subject": "Alex", "predicate": "knows", "object": "Bob"})
        assert result == "Alex knows Bob"

    def test_multi_word_predicate_passthrough(self) -> None:
        """A multi-word identity-form predicate passes through unchanged."""
        result = entry_fact_text(
            {"subject": "Alice", "predicate": "works at company", "object": "Acme"}
        )
        assert result == "Alice works at company Acme"

    def test_hyphenated_predicate_preserved(self) -> None:
        """``-`` is not a blank in the identity form, so it survives the render."""
        result = entry_fact_text(
            {"subject": "Alex", "predicate": "has sister-in-law", "object": "Mia"}
        )
        assert result == "Alex has sister-in-law Mia"

    def test_non_ascii_object_rendered(self) -> None:
        """Non-ASCII object surfaces pass through the render untouched."""
        result = entry_fact_text(
            {"subject": "Alex", "predicate": "has key achievement", "object": "€4.5B sales"}
        )
        assert result == "Alex has key achievement €4.5B sales"

    def test_render_is_prose_not_canonical(self) -> None:
        """The render boundary is a distinct layer from the identity form."""
        from paramem.utils.identity import canonical

        pred = canonical("has hobby")
        assert pred == "has hobby"
        result = entry_fact_text({"subject": "Alex", "predicate": pred, "object": "chess"})
        assert result == "Alex has hobby chess"

    # --- Speaker tokens render verbatim: resolution moved to the reply
    # boundary (paramem.server.speaker.resolve_speaker_tokens); entry_fact_text
    # takes no resolver of its own — every model-facing surface stays in
    # token space. ---

    def test_speaker_token_subject_rendered_verbatim(self) -> None:
        """A speaker{N} token in subject position is emitted as-is — no resolution
        happens at the fact-render boundary."""
        result = entry_fact_text(
            {"subject": "speaker9", "predicate": "lives in", "object": "Berlin"}
        )
        assert result == "speaker9 lives in Berlin"

    def test_speaker_token_object_rendered_verbatim(self) -> None:
        """A speaker{N} token in object position is emitted as-is."""
        result = entry_fact_text(
            {"subject": "speaker0", "predicate": "knows", "object": "speaker9"}
        )
        assert result == "speaker0 knows speaker9"

    def test_no_resolve_parameter_accepted(self) -> None:
        """entry_fact_text takes no resolver parameter — passing one raises."""
        import pytest

        with pytest.raises(TypeError):
            entry_fact_text(
                {"subject": "Alex", "predicate": "knows", "object": "Bob"},
                resolve=lambda t: t,
            )


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


class TestBuildMemorySource:
    """build_memory_source is the one mode → MemorySource construction site.

    Every consumer (boot / post-fold store hydration, the per-query on-miss
    probe, the per-fold hydration) names a mode and takes the source back, so
    these cases pin the whole selection contract in one place.
    """

    def test_simulate_mode_returns_disk_source(self, tmp_path) -> None:
        """Simulate mode reads graph.json — no model needed, none consulted."""
        from paramem.memory.source import DiskMemorySource, build_memory_source

        source = build_memory_source(mode="simulate", adapter_dir=tmp_path, batch_size=16)

        assert isinstance(source, DiskMemorySource)
        assert source.store_dir == tmp_path

    def test_train_mode_returns_weight_source_with_derived_registry(self, tmp_path) -> None:
        """Train mode wires the model plus the SimHash registry read from adapter_dir.

        The registry is DERIVED inside the factory — no caller passes one — so a
        fingerprint written to a tier's ``indexed_key_registry.json`` must reach
        the source's gate without any caller involvement.
        """
        from paramem.memory.source import WeightMemorySource, build_memory_source
        from paramem.training.key_registry import KeyRegistry

        reg = KeyRegistry()
        reg.add("graph1")
        reg.set_simhash("graph1", 4242)
        reg_path = tmp_path / "episodic" / "indexed_key_registry.json"
        reg_path.parent.mkdir(parents=True)
        reg_path.write_bytes(reg.save_bytes())

        model = _make_model_with_adapters("episodic")
        tokenizer = MagicMock()
        source = build_memory_source(
            mode="train",
            adapter_dir=tmp_path,
            batch_size=16,
            model=model,
            tokenizer=tokenizer,
        )

        assert isinstance(source, WeightMemorySource)
        assert source.model is model
        assert source.tokenizer is tokenizer
        assert source.batch_size == 16
        assert source.registry == {"graph1": 4242}

    def test_train_mode_without_a_model_returns_none(self, tmp_path) -> None:
        """No local model → no source of truth for the weights venue.

        Cloud-only boot and a failed model load both land here; the caller
        decides whether to skip (boot preload) or stay cache-only (per query).
        """
        from paramem.memory.source import build_memory_source

        assert build_memory_source(mode="train", adapter_dir=tmp_path, batch_size=16) is None

    def test_simulate_mode_without_a_model_still_returns_a_source(self, tmp_path) -> None:
        """The no-model short-circuit is train-only — simulate never needs one."""
        from paramem.memory.source import DiskMemorySource, build_memory_source

        source = build_memory_source(mode="simulate", adapter_dir=tmp_path, batch_size=16)

        assert isinstance(source, DiskMemorySource)
