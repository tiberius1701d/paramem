"""Unit tests for paramem.graph.reconstruct.

Covers:
- Happy path: 2 tiers, all succeed — graph has correct edges, switch_adapter
  called once per tier, active_adapter restored.
- Failure path strict=True: ReconstructionError raised, message contains key
  and failure_reason.
- Failure path strict=False: failures returned in result, graph contains only
  successful edges.
- Tier filter: only the specified tier's keys are probed.
- Empty registry: returns empty graph without calling switch_adapter.
- Gradient checkpointing toggle: disable called before probing, enable not
  called when gradient_checkpointing=False on the training config.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import networkx as nx
import pytest

from paramem.graph.reconstruct import (
    ReconstructionError,
    ReconstructionResult,
    reconstruct_graph,
)
from paramem.training.key_registry import KeyRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_registry_dict(keys_by_adapter: dict[str, list[str]]) -> dict:
    """Build a per-tier ``dict[str, KeyRegistry]`` from a tier → keys mapping.

    Each tier in *keys_by_adapter* becomes a separate ``KeyRegistry`` instance
    containing only the keys for that tier — matching the production schema
    where ``ConsolidationLoop.indexed_key_registry`` is keyed by tier name.
    """
    reg_dict: dict[str, KeyRegistry] = {}
    for adapter_id, keys in keys_by_adapter.items():
        reg = KeyRegistry()
        for key in keys:
            reg.add(key)
        reg_dict[adapter_id] = reg
    return reg_dict


# Legacy alias used throughout tests; passes dict directly.
_make_registry = _make_registry_dict


def _make_loop(
    registry,
    *,
    episodic_simhash: dict | None = None,
    semantic_simhash: dict | None = None,
    procedural_simhash: dict | None = None,
    gradient_checkpointing: bool = False,
) -> SimpleNamespace:
    """Return a minimal stub loop accepted by reconstruct_graph.

    ``registry`` is a ``dict[str, KeyRegistry]`` (per-tier) or ``None``
    (disabled).  The empty-dict case is used in tests that need no active keys.
    """
    from paramem.memory.store import MemoryStore

    model = MagicMock()
    model.active_adapter = "episodic"  # default; overwritten per test as needed

    training_config = SimpleNamespace(gradient_checkpointing=gradient_checkpointing)

    store = MemoryStore(replay_enabled=registry is not None)
    if registry is not None:
        for tier_name, reg in registry.items():
            store.load_registry(tier_name, reg)
    if episodic_simhash:
        store.replace_simhashes_in_tier("episodic", episodic_simhash)
    if semantic_simhash:
        store.replace_simhashes_in_tier("semantic", semantic_simhash)
    if procedural_simhash:
        store.replace_simhashes_in_tier("procedural", procedural_simhash)

    loop = SimpleNamespace(
        model=model,
        tokenizer=MagicMock(),
        store=store,
        training_config=training_config,
    )
    return loop


def _success_dict(key: str, subject: str, predicate: str, obj: str) -> dict:
    """Build a probe_entries-style success dict."""
    return {
        "key": key,
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "confidence": 1.0,
        "raw_output": f'{{"key":"{key}"}}',
        "fact_text": f"{subject} {predicate} {obj}",
    }


def _failure_dict(raw_output: str = "", reason: str = "parse_failure") -> dict:
    """Build a probe_entries-style failure dict."""
    return {"raw_output": raw_output, "failure_reason": reason}


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_all_succeed_two_tiers(self, monkeypatch):
        """4 keys across 2 tiers all succeed — correct edges + switch calls."""
        registry = _make_registry(
            {
                "episodic": ["graph1", "graph2"],
                "semantic": ["graph3", "graph4"],
            }
        )
        loop = _make_loop(registry)
        loop.model.active_adapter = "episodic"

        per_key = {
            "graph1": _success_dict("graph1", "Alice", "lives_in", "Berlin"),
            "graph2": _success_dict("graph2", "Bob", "works_at", "ACME"),
            "graph3": _success_dict("graph3", "Alice", "knows", "Bob"),
            "graph4": _success_dict("graph4", "Bob", "manages", "DevTeam"),
        }
        switch_calls: list[str] = []

        def fake_probe_entries(model, tokenizer, entries, **kwargs):
            for entry in entries:
                yield entry, per_key[entry["key"]]

        def fake_switch(model, name):
            switch_calls.append(name)

        monkeypatch.setattr("paramem.graph.reconstruct.probe_entries", fake_probe_entries)
        monkeypatch.setattr("paramem.graph.reconstruct.switch_adapter", fake_switch)

        result = reconstruct_graph(loop)

        assert isinstance(result, ReconstructionResult)
        assert result.failures == []
        assert result.graph.number_of_edges() == 4

        # One switch per adapter group (episodic + semantic) + one final
        # restore back to the original active adapter.
        assert len(switch_calls) == 3  # episodic, semantic, restore(episodic)
        assert switch_calls[-1] == "episodic"  # active_adapter restored

    def test_edge_data_matches_quad(self, monkeypatch):
        """Edge attributes must carry key and predicate from the recalled quad."""
        registry = _make_registry({"episodic": ["graph1"]})
        loop = _make_loop(registry)

        def fake_probe_entries(model, tokenizer, entries, **kwargs):
            for entry in entries:
                yield entry, _success_dict("graph1", "Alice", "lives_in", "Berlin")

        def fake_switch(model, name):
            pass

        monkeypatch.setattr("paramem.graph.reconstruct.probe_entries", fake_probe_entries)
        monkeypatch.setattr("paramem.graph.reconstruct.switch_adapter", fake_switch)

        result = reconstruct_graph(loop)

        g = result.graph
        # Read the edge via the public iter_entries accessor — the indexed-memory
        # key is stored under _IK_KEY_ATTR (= "ik_key") internally to avoid
        # NetworkX's reserved "key" field in node_link_data serialisation.
        from paramem.memory.persistence import iter_entries

        quads = list(iter_entries(g))
        assert len(quads) == 1
        assert quads[0]["subject"] == "Alice"
        assert quads[0]["object"] == "Berlin"
        assert quads[0]["key"] == "graph1"
        assert quads[0]["predicate"] == "lives_in"

    def test_switch_adapter_called_once_per_group(self, monkeypatch):
        """switch_adapter must be called exactly once per adapter group, not per key."""
        registry = _make_registry(
            {
                "episodic": ["e1", "e2", "e3"],
                "semantic": ["s1", "s2"],
            }
        )
        loop = _make_loop(registry)
        loop.model.active_adapter = "semantic"

        group_switches: list[str] = []

        def fake_probe_entries(model, tokenizer, entries, **kwargs):
            for entry in entries:
                yield entry, _success_dict(entry["key"], "X", "rel", "Y")

        def fake_switch(model, name):
            group_switches.append(name)

        monkeypatch.setattr("paramem.graph.reconstruct.probe_entries", fake_probe_entries)
        monkeypatch.setattr("paramem.graph.reconstruct.switch_adapter", fake_switch)

        reconstruct_graph(loop)

        # Two group switches + one restore = 3 total.
        # Neither group should appear more than once (excluding the restore).
        probe_switches = group_switches[:-1]  # drop the final restore
        assert probe_switches.count("episodic") == 1
        assert probe_switches.count("semantic") == 1

    def test_active_adapter_restored_after_probe(self, monkeypatch):
        """The last switch_adapter call must target the original active adapter."""
        registry = _make_registry({"procedural": ["p1"]})
        loop = _make_loop(registry)
        loop.model.active_adapter = "procedural"

        restore_calls: list[str] = []

        def fake_probe_entries(model, tokenizer, entries, **kwargs):
            for entry in entries:
                yield entry, _success_dict(entry["key"], "A", "b", "C")

        def fake_switch(model, name):
            restore_calls.append(name)

        monkeypatch.setattr("paramem.graph.reconstruct.probe_entries", fake_probe_entries)
        monkeypatch.setattr("paramem.graph.reconstruct.switch_adapter", fake_switch)

        reconstruct_graph(loop)

        assert restore_calls[-1] == "procedural"

    def test_active_adapter_as_list_is_unwrapped(self, monkeypatch):
        """PEFT layouts that return a list for active_adapter must not break
        the restore path — switch_adapter receives the first element, not the
        list itself (mirrors the defensive unwrap in app.py:2510-2514)."""
        registry = _make_registry({"episodic": ["e1"]})
        loop = _make_loop(registry)
        loop.model.active_adapter = ["episodic"]  # PEFT-list-edge case

        restore_calls: list[str] = []

        monkeypatch.setattr(
            "paramem.graph.reconstruct.probe_entries",
            lambda model, tokenizer, entries, **kwargs: (
                (e, _success_dict(e["key"], "A", "b", "C")) for e in entries
            ),
        )
        monkeypatch.setattr(
            "paramem.graph.reconstruct.switch_adapter",
            lambda model, name: restore_calls.append(name),
        )

        reconstruct_graph(loop)

        # Every switch_adapter call must receive a string, never a list.
        assert all(isinstance(name, str) for name in restore_calls)
        assert restore_calls[-1] == "episodic"

    def test_empty_active_adapter_list_skips_restore(self, monkeypatch):
        """An empty active_adapter list means no adapter was active — the
        restore call must be skipped (switch_adapter requires a real name)."""
        registry = _make_registry({"episodic": ["e1"]})
        loop = _make_loop(registry)
        loop.model.active_adapter = []  # PEFT edge case: no adapter set

        switch_calls: list[str] = []

        monkeypatch.setattr(
            "paramem.graph.reconstruct.probe_entries",
            lambda model, tokenizer, entries, **kwargs: (
                (e, _success_dict(e["key"], "A", "b", "C")) for e in entries
            ),
        )
        monkeypatch.setattr(
            "paramem.graph.reconstruct.switch_adapter",
            lambda model, name: switch_calls.append(name),
        )

        reconstruct_graph(loop)

        # Exactly one switch fires (entering the "episodic" group); no restore
        # at the end because original_adapter resolved to None.
        assert switch_calls == ["episodic"]


# ---------------------------------------------------------------------------
# Failure path — strict=True
# ---------------------------------------------------------------------------


class TestFailureStrictTrue:
    def test_raises_reconstruction_error(self, monkeypatch):
        """One failing key with strict=True must raise ReconstructionError."""
        registry = _make_registry({"episodic": ["graph1", "graph2"]})
        loop = _make_loop(registry)

        def fake_probe_entries(model, tokenizer, entries, **kwargs):
            for entry in entries:
                key = entry["key"]
                if key == "graph2":
                    yield entry, _failure_dict(reason="parse_failure")
                else:
                    yield entry, _success_dict(key, "Alice", "lives_in", "Berlin")

        def fake_switch(model, name):
            pass

        monkeypatch.setattr("paramem.graph.reconstruct.probe_entries", fake_probe_entries)
        monkeypatch.setattr("paramem.graph.reconstruct.switch_adapter", fake_switch)

        with pytest.raises(ReconstructionError) as exc_info:
            reconstruct_graph(loop, strict=True)

        msg = str(exc_info.value)
        assert "graph2" in msg
        assert "parse_failure" in msg

    def test_error_message_includes_failure_count(self, monkeypatch):
        """ReconstructionError message must include the count of failures."""
        registry = _make_registry({"episodic": ["k1", "k2", "k3"]})
        loop = _make_loop(registry)

        def fake_probe_entries(model, tokenizer, entries, **kwargs):
            for entry in entries:
                yield entry, _failure_dict(reason="key_mismatch:other")

        def fake_switch(model, name):
            pass

        monkeypatch.setattr("paramem.graph.reconstruct.probe_entries", fake_probe_entries)
        monkeypatch.setattr("paramem.graph.reconstruct.switch_adapter", fake_switch)

        with pytest.raises(ReconstructionError) as exc_info:
            reconstruct_graph(loop, strict=True)

        assert "3" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Failure path — strict=False
# ---------------------------------------------------------------------------


class TestFailureStrictFalse:
    def test_returns_failures_in_result(self, monkeypatch):
        """Failing key with strict=False is in result.failures, not graph."""
        registry = _make_registry({"episodic": ["graph1", "graph2"]})
        loop = _make_loop(registry)

        def fake_probe_entries(model, tokenizer, entries, **kwargs):
            for entry in entries:
                key = entry["key"]
                if key == "graph2":
                    yield entry, _failure_dict(reason="parse_failure")
                else:
                    yield entry, _success_dict(key, "Alice", "lives_in", "Berlin")

        def fake_switch(model, name):
            pass

        monkeypatch.setattr("paramem.graph.reconstruct.probe_entries", fake_probe_entries)
        monkeypatch.setattr("paramem.graph.reconstruct.switch_adapter", fake_switch)

        result = reconstruct_graph(loop, strict=False)

        assert len(result.failures) == 1
        assert result.failures[0]["key"] == "graph2"
        assert result.failures[0]["failure_reason"] == "parse_failure"
        assert result.failures[0]["adapter_id"] == "episodic"

    def test_graph_contains_only_successful_edges(self, monkeypatch):
        """Failed keys must not appear as edges in the graph."""
        registry = _make_registry({"episodic": ["graph1", "graph2"]})
        loop = _make_loop(registry)

        def fake_probe_entries(model, tokenizer, entries, **kwargs):
            for entry in entries:
                key = entry["key"]
                if key == "graph2":
                    yield entry, _failure_dict(reason="low_confidence:0.400")
                else:
                    yield entry, _success_dict("graph1", "Alice", "lives_in", "Berlin")

        def fake_switch(model, name):
            pass

        monkeypatch.setattr("paramem.graph.reconstruct.probe_entries", fake_probe_entries)
        monkeypatch.setattr("paramem.graph.reconstruct.switch_adapter", fake_switch)

        result = reconstruct_graph(loop, strict=False)

        assert result.graph.number_of_edges() == 1
        from paramem.memory.persistence import iter_entries

        edge_keys = [q["key"] for q in iter_entries(result.graph)]
        assert "graph1" in edge_keys
        assert "graph2" not in edge_keys

    def test_all_keys_accounted_for(self, monkeypatch):
        """Every key is either in graph edges or in failures — no silent drops."""
        registry = _make_registry({"semantic": ["s1", "s2", "s3"]})
        loop = _make_loop(registry)

        def fake_probe_entries(model, tokenizer, entries, **kwargs):
            for entry in entries:
                key = entry["key"]
                if key == "s2":
                    yield entry, _failure_dict(reason="parse_failure")
                else:
                    yield entry, _success_dict(key, "X", "rel", "Y")

        def fake_switch(model, name):
            pass

        monkeypatch.setattr("paramem.graph.reconstruct.probe_entries", fake_probe_entries)
        monkeypatch.setattr("paramem.graph.reconstruct.switch_adapter", fake_switch)

        result = reconstruct_graph(loop, strict=False)

        from paramem.memory.persistence import iter_entries

        edge_keys = {q["key"] for q in iter_entries(result.graph)}
        failure_keys = {f["key"] for f in result.failures}
        assert edge_keys | failure_keys == {"s1", "s2", "s3"}
        assert edge_keys & failure_keys == set()


# ---------------------------------------------------------------------------
# Tier filter
# ---------------------------------------------------------------------------


class TestTierFilter:
    def test_only_specified_tier_probed(self, monkeypatch):
        """With tier='semantic', only semantic keys are probed."""
        registry = _make_registry(
            {
                "episodic": ["e1", "e2"],
                "semantic": ["s1", "s2"],
                "procedural": ["p1"],
            }
        )
        loop = _make_loop(registry)

        probed_keys: list[str] = []

        def fake_probe_entries(model, tokenizer, entries, **kwargs):
            for entry in entries:
                probed_keys.append(entry["key"])
                yield entry, _success_dict(entry["key"], "A", "b", "C")

        def fake_switch(model, name):
            pass

        monkeypatch.setattr("paramem.graph.reconstruct.probe_entries", fake_probe_entries)
        monkeypatch.setattr("paramem.graph.reconstruct.switch_adapter", fake_switch)

        result = reconstruct_graph(loop, tier="semantic")

        assert sorted(probed_keys) == ["s1", "s2"]
        assert result.graph.number_of_edges() == 2

    def test_tier_filter_only_switches_matching_adapter(self, monkeypatch):
        """With a tier filter, only the matching adapter's switch is called."""
        registry = _make_registry(
            {
                "episodic": ["e1"],
                "semantic": ["s1"],
            }
        )
        loop = _make_loop(registry)
        loop.model.active_adapter = "episodic"

        switch_calls: list[str] = []

        def fake_probe_entries(model, tokenizer, entries, **kwargs):
            for entry in entries:
                yield entry, _success_dict(entry["key"], "A", "b", "C")

        def fake_switch(model, name):
            switch_calls.append(name)

        monkeypatch.setattr("paramem.graph.reconstruct.probe_entries", fake_probe_entries)
        monkeypatch.setattr("paramem.graph.reconstruct.switch_adapter", fake_switch)

        reconstruct_graph(loop, tier="semantic")

        # One switch to semantic, one restore to episodic.
        assert switch_calls == ["semantic", "episodic"]


# ---------------------------------------------------------------------------
# Empty registry
# ---------------------------------------------------------------------------


class TestEmptyRegistry:
    def test_empty_registry_returns_empty_result(self, monkeypatch):
        """Empty registry returns empty graph + failures without crashing."""
        registry = {}  # empty dict = no tiers, no keys
        loop = _make_loop(registry)

        switch_calls: list[str] = []

        def fake_switch(model, name):
            switch_calls.append(name)

        monkeypatch.setattr("paramem.graph.reconstruct.switch_adapter", fake_switch)

        result = reconstruct_graph(loop)

        assert isinstance(result.graph, nx.MultiDiGraph)
        assert result.graph.number_of_edges() == 0
        assert result.failures == []
        assert switch_calls == []  # never called

    def test_empty_registry_with_tier_filter(self, monkeypatch):
        """Empty result when tier filter matches no keys."""
        registry = _make_registry({"episodic": ["e1"]})
        loop = _make_loop(registry)

        switch_calls: list[str] = []

        def fake_switch(model, name):
            switch_calls.append(name)

        monkeypatch.setattr("paramem.graph.reconstruct.switch_adapter", fake_switch)

        result = reconstruct_graph(loop, tier="semantic")

        assert result.graph.number_of_edges() == 0
        assert result.failures == []
        assert switch_calls == []


# ---------------------------------------------------------------------------
# Gradient checkpointing toggle
# ---------------------------------------------------------------------------


class TestGradientCheckpointingToggle:
    def test_disable_called_before_probe(self, monkeypatch):
        """gradient_checkpointing_disable must be called exactly once."""
        registry = _make_registry({"episodic": ["e1"]})
        loop = _make_loop(registry)

        def fake_probe_entries(model, tokenizer, entries, **kwargs):
            for entry in entries:
                yield entry, _success_dict(entry["key"], "A", "b", "C")

        monkeypatch.setattr("paramem.graph.reconstruct.probe_entries", fake_probe_entries)
        monkeypatch.setattr("paramem.graph.reconstruct.switch_adapter", lambda m, n: None)

        reconstruct_graph(loop)

        loop.model.gradient_checkpointing_disable.assert_called_once()

    def test_enable_not_called_when_checkpointing_off(self, monkeypatch):
        """gradient_checkpointing_enable must NOT be called when config has it off."""
        registry = _make_registry({"episodic": ["e1"]})
        loop = _make_loop(registry, gradient_checkpointing=False)

        def fake_probe_entries(model, tokenizer, entries, **kwargs):
            for entry in entries:
                yield entry, _success_dict(entry["key"], "A", "b", "C")

        monkeypatch.setattr("paramem.graph.reconstruct.probe_entries", fake_probe_entries)
        monkeypatch.setattr("paramem.graph.reconstruct.switch_adapter", lambda m, n: None)

        reconstruct_graph(loop)

        loop.model.gradient_checkpointing_enable.assert_not_called()

    def test_enable_called_when_checkpointing_on(self, monkeypatch):
        """gradient_checkpointing_enable must be called when config has it on."""
        registry = _make_registry({"episodic": ["e1"]})
        loop = _make_loop(registry, gradient_checkpointing=True)

        def fake_probe_entries(model, tokenizer, entries, **kwargs):
            for entry in entries:
                yield entry, _success_dict(entry["key"], "A", "b", "C")

        monkeypatch.setattr("paramem.graph.reconstruct.probe_entries", fake_probe_entries)
        monkeypatch.setattr("paramem.graph.reconstruct.switch_adapter", lambda m, n: None)

        reconstruct_graph(loop)

        loop.model.gradient_checkpointing_enable.assert_called_once()

    def test_disable_called_even_on_failure(self, monkeypatch):
        """gradient_checkpointing_disable is called in the finally block on error."""
        registry = _make_registry({"episodic": ["bad_key"]})
        loop = _make_loop(registry)

        def fake_probe_entries(model, tokenizer, entries, **kwargs):
            for entry in entries:
                yield entry, _failure_dict(reason="parse_failure")

        monkeypatch.setattr("paramem.graph.reconstruct.probe_entries", fake_probe_entries)
        monkeypatch.setattr("paramem.graph.reconstruct.switch_adapter", lambda m, n: None)

        with pytest.raises(ReconstructionError):
            reconstruct_graph(loop, strict=True)

        loop.model.gradient_checkpointing_disable.assert_called_once()
