"""Unit tests for paramem.training.indexed_memory.probe_keys_from_graph.

Covers happy path, miss, missing tier file, and multi-adapter dispatch.
Reuses the encryption-isolation fixture pattern from test_keyed_pairs_io_quad.py.
"""

from __future__ import annotations

import networkx as nx
import pytest

from paramem.backup.key_store import (
    _clear_daily_identity_cache,
)
from paramem.server.simulate_store import save_simulate_graph
from paramem.training.indexed_memory import probe_keys_from_graph
from paramem.training.quadruple_memory import quad_fact_text

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_QUAD1 = {
    "key": "graph1",
    "subject": "Alice",
    "predicate": "lives_in",
    "object": "Berlin",
    "speaker_id": "Speaker0",
    "first_seen_cycle": 1,
}
_QUAD2 = {
    "key": "graph2",
    "subject": "Bob",
    "predicate": "has_job",
    "object": "Engineer",
    "speaker_id": "Speaker0",
    "first_seen_cycle": 2,
}


def _build_graph(*quads) -> nx.MultiDiGraph:
    """Build a MultiDiGraph from a sequence of quad dicts.

    Uses ``simulate_store._add_keyed_edge`` so the indexed-memory key is
    stored as ``"ik_key"`` in edge data and survives
    ``nx.node_link_data`` / ``nx.node_link_graph`` round-trips.
    """
    from paramem.server.simulate_store import _add_keyed_edge

    g = nx.MultiDiGraph()
    for q in quads:
        _add_keyed_edge(
            g,
            q["subject"],
            q["object"],
            indexed_key=q["key"],
            predicate=q["predicate"],
            speaker_id=q["speaker_id"],
            first_seen_cycle=q["first_seen_cycle"],
        )
    return g


def _write_tier_graph(simulate_dir, tier: str, *quads) -> None:
    """Write a tier graph.json under simulate_dir/<tier>/graph.json."""
    path = simulate_dir / tier / "graph.json"
    g = _build_graph(*quads)
    save_simulate_graph(g, path, encrypted=False)


@pytest.fixture(autouse=True)
def _env_isolation():
    """Isolate daily identity cache per test."""
    _clear_daily_identity_cache()
    yield
    _clear_daily_identity_cache()


# ---------------------------------------------------------------------------
# 1. Happy path: 2 known quads → both resolve with full shape
# ---------------------------------------------------------------------------


class TestProbeKeysFromGraphHappyPath:
    def test_two_known_keys_resolve(self, tmp_path):
        """Both keys in the episodic graph return full success dicts."""
        simulate_dir = tmp_path / "simulate"
        _write_tier_graph(simulate_dir, "episodic", _QUAD1, _QUAD2)

        result = probe_keys_from_graph(simulate_dir, {"episodic": ["graph1", "graph2"]})

        assert len(result) == 2
        for key, quad in [("graph1", _QUAD1), ("graph2", _QUAD2)]:
            entry = result[key]
            assert entry is not None, f"expected hit for {key}"
            assert entry["key"] == key
            assert entry["subject"] == quad["subject"]
            assert entry["predicate"] == quad["predicate"]
            assert entry["object"] == quad["object"]
            assert entry["confidence"] == 1.0
            assert entry["format"] == "quad"

    def test_fact_text_matches_quad_fact_text(self, tmp_path):
        """fact_text is produced by quad_fact_text, not by hand."""
        simulate_dir = tmp_path / "simulate"
        _write_tier_graph(simulate_dir, "episodic", _QUAD1)

        result = probe_keys_from_graph(simulate_dir, {"episodic": ["graph1"]})
        entry = result["graph1"]
        assert entry is not None
        assert entry["fact_text"] == quad_fact_text(_QUAD1)

    def test_raw_output_is_json_with_four_fields(self, tmp_path):
        """raw_output is a JSON string with key/subject/predicate/object."""
        import json

        simulate_dir = tmp_path / "simulate"
        _write_tier_graph(simulate_dir, "episodic", _QUAD1)

        result = probe_keys_from_graph(simulate_dir, {"episodic": ["graph1"]})
        entry = result["graph1"]
        assert entry is not None
        parsed = json.loads(entry["raw_output"])
        assert set(parsed.keys()) == {"key", "subject", "predicate", "object"}
        assert parsed["key"] == "graph1"
        assert parsed["subject"] == _QUAD1["subject"]
        assert parsed["predicate"] == _QUAD1["predicate"]
        assert parsed["object"] == _QUAD1["object"]


# ---------------------------------------------------------------------------
# 2. Miss: key not in graph → result is None
# ---------------------------------------------------------------------------


class TestProbeKeysFromGraphMiss:
    def test_missing_key_returns_none(self, tmp_path):
        """A key not present in the graph resolves to None."""
        simulate_dir = tmp_path / "simulate"
        _write_tier_graph(simulate_dir, "episodic", _QUAD1)

        result = probe_keys_from_graph(simulate_dir, {"episodic": ["graph1", "graph999"]})

        assert result["graph1"] is not None
        assert result["graph999"] is None

    def test_all_missing_keys_return_none(self, tmp_path):
        """When no key exists in the graph, all results are None."""
        simulate_dir = tmp_path / "simulate"
        _write_tier_graph(simulate_dir, "episodic", _QUAD1)

        result = probe_keys_from_graph(simulate_dir, {"episodic": ["graph99", "graph100"]})

        assert result["graph99"] is None
        assert result["graph100"] is None


# ---------------------------------------------------------------------------
# 3. Missing tier file: all keys in that adapter return None, no exception
# ---------------------------------------------------------------------------


class TestProbeKeysFromGraphMissingTier:
    def test_missing_tier_file_returns_none_for_all_keys(self, tmp_path):
        """When <simulate_dir>/semantic/graph.json is absent, all keys → None."""
        simulate_dir = tmp_path / "simulate"
        # Do NOT write semantic/graph.json — the path is intentionally absent.

        result = probe_keys_from_graph(simulate_dir, {"semantic": ["graph1", "graph2"]})

        assert result["graph1"] is None
        assert result["graph2"] is None

    def test_missing_tier_does_not_raise(self, tmp_path):
        """probe_keys_from_graph must not raise when a tier file is absent."""
        simulate_dir = tmp_path / "simulate"
        # No files written at all.
        result = probe_keys_from_graph(
            simulate_dir,
            {"episodic": ["graph1"], "semantic": ["graph2"], "procedural": ["proc1"]},
        )
        assert result["graph1"] is None
        assert result["graph2"] is None
        assert result["proc1"] is None


# ---------------------------------------------------------------------------
# 4. Multi-adapter: episodic + semantic resolve independently
# ---------------------------------------------------------------------------


class TestProbeKeysFromGraphMultiAdapter:
    def test_episodic_and_semantic_resolve_independently(self, tmp_path):
        """Keys in episodic and semantic tiers each resolve against their own graph."""
        simulate_dir = tmp_path / "simulate"
        _write_tier_graph(simulate_dir, "episodic", _QUAD1)
        _write_tier_graph(simulate_dir, "semantic", _QUAD2)

        result = probe_keys_from_graph(
            simulate_dir,
            {"episodic": ["graph1"], "semantic": ["graph2"]},
        )

        ep = result["graph1"]
        sem = result["graph2"]
        assert ep is not None
        assert sem is not None
        assert ep["subject"] == _QUAD1["subject"]
        assert sem["subject"] == _QUAD2["subject"]

    def test_key_in_wrong_tier_is_miss(self, tmp_path):
        """A key present in episodic but queried from semantic resolves against that tier.

        Each adapter is looked up in its own tier graph.  A key present in
        episodic/graph.json but queried via the semantic adapter returns None,
        because the look-up reads semantic/graph.json which does not contain it.

        Distinct keys are used for each adapter so the flat result dict has no
        ambiguous overwrites.
        """
        simulate_dir = tmp_path / "simulate"
        _write_tier_graph(simulate_dir, "episodic", _QUAD1)  # graph1 in episodic
        # Write a semantic tier that does NOT contain graph1.
        _write_tier_graph(simulate_dir, "semantic", _QUAD2)  # only graph2 in semantic

        result = probe_keys_from_graph(
            simulate_dir,
            {
                "episodic": ["graph1"],  # graph1 in episodic → hit
                "semantic": ["graph1"],  # graph1 NOT in semantic → None
            },
        )
        # The flat dict has one entry for "graph1"; the semantic iteration (which
        # runs second for dict insertion order) overwrites the episodic hit with None.
        # The result is None because probe_keys_from_graph returns a flat dict keyed
        # by key name, and the later adapter's result wins.
        assert result["graph1"] is None

    def test_all_three_tiers(self, tmp_path):
        """episodic, semantic, and procedural each resolve their own key."""
        proc_quad = {
            "key": "proc1",
            "subject": "User",
            "predicate": "prefers",
            "object": "DarkMode",
            "speaker_id": "Speaker0",
            "first_seen_cycle": 5,
        }
        simulate_dir = tmp_path / "simulate"
        _write_tier_graph(simulate_dir, "episodic", _QUAD1)
        _write_tier_graph(simulate_dir, "semantic", _QUAD2)
        _write_tier_graph(simulate_dir, "procedural", proc_quad)

        result = probe_keys_from_graph(
            simulate_dir,
            {
                "episodic": ["graph1"],
                "semantic": ["graph2"],
                "procedural": ["proc1"],
            },
        )
        assert result["graph1"] is not None
        assert result["graph2"] is not None
        assert result["proc1"] is not None
        assert result["proc1"]["subject"] == "User"
        assert result["proc1"]["object"] == "DarkMode"

    def test_empty_key_list_for_adapter_produces_no_entries(self, tmp_path):
        """An adapter with an empty key list contributes nothing to the result."""
        simulate_dir = tmp_path / "simulate"
        _write_tier_graph(simulate_dir, "episodic", _QUAD1)

        result = probe_keys_from_graph(
            simulate_dir,
            {"episodic": [], "semantic": ["graph1"]},
        )
        # episodic has no keys so nothing from it.
        # semantic has graph1 but no file, so None.
        assert result.get("graph1") is None
        # Nothing from episodic's empty list.
        assert len(result) == 1
