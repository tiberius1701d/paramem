"""Unit tests for paramem.server.simulate_store.

Covers round-trip contract, encryption awareness, iter_quads edge-skipping,
quad_by_key hit/miss, entity/speaker index helpers, and build_tier_graph_from_loop.
Reuses the encryption-aware fixture pattern from tests/test_keyed_pairs_io_quad.py.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import networkx as nx
import pytest

from paramem.backup.age_envelope import AGE_MAGIC
from paramem.backup.key_store import (
    DAILY_PASSPHRASE_ENV_VAR,
    _clear_daily_identity_cache,
    mint_daily_identity,
    wrap_daily_identity,
    write_daily_key_file,
)
from paramem.server.simulate_store import (
    _IK_KEY_ATTR,
    build_tier_graph_from_loop,
    iter_quads,
    keys_for_entity,
    keys_for_speaker,
    load_simulate_graph,
    quad_by_key,
    save_simulate_graph,
)
from paramem.training.keyed_pairs_io import KEYED_PAIR_FIELDS_QUAD

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EDGE_DATA = {
    "key": "graph1",
    "predicate": "lives_in",
    "speaker_id": "Speaker0",
    "first_seen_cycle": 3,
}

_SUBJECT = "Alice"
_OBJECT = "Berlin"


def _add_keyed_edge(
    graph: nx.MultiDiGraph,
    subject: str,
    object_: str,
    *,
    indexed_key: str,
    predicate: str,
    speaker_id: str,
    first_seen_cycle: int,
) -> None:
    """Test-local wrapper around simulate_store._add_keyed_edge.

    Delegates to the production helper so tests exercise the same code path
    that ``build_tier_graph_from_loop`` uses.  The indexed-memory key is stored
    as ``"ik_key"`` in edge data (never as the NetworkX multigraph edge-key
    parameter) so it survives ``nx.node_link_data`` / ``nx.node_link_graph``
    round-trips intact.
    """
    from paramem.server.simulate_store import _add_keyed_edge as _prod_add

    _prod_add(
        graph,
        subject,
        object_,
        indexed_key=indexed_key,
        predicate=predicate,
        speaker_id=speaker_id,
        first_seen_cycle=first_seen_cycle,
    )


def _make_simple_graph(
    subject: str = _SUBJECT,
    object_: str = _OBJECT,
    **edge_overrides,
) -> nx.MultiDiGraph:
    """Build a minimal single-edge ``MultiDiGraph`` for testing."""
    g = nx.MultiDiGraph()
    edge_data = dict(_EDGE_DATA)
    edge_data.update(edge_overrides)
    indexed_key = edge_data.pop("key")
    _add_keyed_edge(
        g,
        subject,
        object_,
        indexed_key=indexed_key,
        predicate=edge_data["predicate"],
        speaker_id=edge_data["speaker_id"],
        first_seen_cycle=edge_data["first_seen_cycle"],
    )
    return g


def _setup_daily(tmp_path, monkeypatch, passphrase: str = "pw"):
    """Mint + wrap + write a daily identity; point env + module default at it."""
    ident = mint_daily_identity()
    key_path = tmp_path / "daily_key.age"
    write_daily_key_file(wrap_daily_identity(ident, passphrase), key_path)
    monkeypatch.setenv(DAILY_PASSPHRASE_ENV_VAR, passphrase)
    monkeypatch.setattr("paramem.backup.key_store.DAILY_KEY_PATH_DEFAULT", key_path)
    _clear_daily_identity_cache()
    return ident


@pytest.fixture(autouse=True)
def _env_isolation(monkeypatch):
    """Isolate daily identity cache per test so encryption state is predictable."""
    _clear_daily_identity_cache()
    yield
    _clear_daily_identity_cache()


# ---------------------------------------------------------------------------
# 1. Round-trip: save then load preserves nodes/edges/edge-data
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_round_trip_preserves_edge_data(self, tmp_path):
        """save_simulate_graph then load returns graph with same edge attributes.

        The public API view (via iter_quads) exposes 'key'; internally the
        graph stores the indexed-memory key as _IK_KEY_ATTR to avoid the
        NetworkX serialisation collision.
        """
        g = _make_simple_graph()
        path = tmp_path / "graph.json"
        save_simulate_graph(g, path)
        g2 = load_simulate_graph(path)

        # Verify internal storage via iter_quads (public API).
        quads = list(iter_quads(g2))
        assert len(quads) == 1
        q = quads[0]
        assert q["key"] == "graph1"
        assert q["subject"] == _SUBJECT
        assert q["object"] == _OBJECT
        assert q["predicate"] == "lives_in"
        assert q["speaker_id"] == "Speaker0"
        assert q["first_seen_cycle"] == 3

        # Also verify the internal attribute is present in raw edge data.
        edges = list(g2.edges(keys=True, data=True))
        assert len(edges) == 1
        _subject, _object, _nx_key, raw_data = edges[0]
        assert raw_data[_IK_KEY_ATTR] == "graph1"
        assert raw_data["predicate"] == "lives_in"

    def test_round_trip_multiple_edges(self, tmp_path):
        """Multiple edges survive the round-trip with correct data."""
        g = nx.MultiDiGraph()
        _add_keyed_edge(
            g,
            "Alice",
            "Berlin",
            indexed_key="graph1",
            predicate="lives_in",
            speaker_id="S0",
            first_seen_cycle=1,
        )
        _add_keyed_edge(
            g,
            "Bob",
            "Engineer",
            indexed_key="graph2",
            predicate="has_job",
            speaker_id="S1",
            first_seen_cycle=2,
        )
        path = tmp_path / "graph.json"
        save_simulate_graph(g, path)
        g2 = load_simulate_graph(path)
        assert g2.number_of_edges() == 2
        # Verify both keys are present in edge data.
        quads = {q["key"] for q in iter_quads(g2)}
        assert quads == {"graph1", "graph2"}

    def test_round_trip_creates_parent_directory(self, tmp_path):
        """save_simulate_graph creates missing parent directories."""
        path = tmp_path / "subdir" / "nested" / "graph.json"
        save_simulate_graph(_make_simple_graph(), path)
        assert path.exists()

    def test_round_trip_empty_graph(self, tmp_path):
        """An empty graph survives the round-trip."""
        g = nx.MultiDiGraph()
        path = tmp_path / "graph.json"
        save_simulate_graph(g, path)
        g2 = load_simulate_graph(path)
        assert g2.number_of_edges() == 0
        assert g2.number_of_nodes() == 0


# ---------------------------------------------------------------------------
# 2. load_simulate_graph on missing path returns empty MultiDiGraph — no raise
# ---------------------------------------------------------------------------


class TestLoadMissingPath:
    def test_missing_path_returns_empty_multigraph(self, tmp_path):
        """load_simulate_graph returns an empty MultiDiGraph when path is absent."""
        path = tmp_path / "does_not_exist.json"
        g = load_simulate_graph(path)
        assert isinstance(g, nx.MultiDiGraph)
        assert g.number_of_nodes() == 0
        assert g.number_of_edges() == 0

    def test_missing_path_does_not_raise(self, tmp_path):
        """No exception is raised when path is absent."""
        path = tmp_path / "absent" / "graph.json"
        # No FileNotFoundError or OSError expected.
        g = load_simulate_graph(path)
        assert isinstance(g, nx.MultiDiGraph)


# ---------------------------------------------------------------------------
# 3. iter_quads skips edges without a "key" attribute
# ---------------------------------------------------------------------------


class TestIterQuadsSkipsKeylessEdges:
    def test_skips_edge_without_key(self):
        """iter_quads yields only edges that carry a 'key' attribute."""
        g = nx.MultiDiGraph()
        # Edge WITH key — added via helper so 'key' lands in edge data.
        _add_keyed_edge(
            g,
            "Alice",
            "Berlin",
            indexed_key="graph1",
            predicate="lives_in",
            speaker_id="S0",
            first_seen_cycle=1,
        )
        # Edge WITHOUT key — direct add_edge, no 'key' attribute in data.
        g.add_edge("Alice", "Paris", predicate="visited", speaker_id="S0", first_seen_cycle=2)
        quads = list(iter_quads(g))
        assert len(quads) == 1
        assert quads[0]["key"] == "graph1"

    def test_empty_graph_yields_nothing(self):
        """iter_quads on an empty graph yields no items."""
        assert list(iter_quads(nx.MultiDiGraph())) == []

    def test_graph_with_no_keyed_edges_yields_nothing(self):
        """A graph where no edges carry 'key' yields no items."""
        g = nx.MultiDiGraph()
        g.add_edge("A", "B", predicate="related", speaker_id="S0", first_seen_cycle=1)
        assert list(iter_quads(g)) == []


# ---------------------------------------------------------------------------
# 4. iter_quads shape: every yielded dict has exactly the 6 KEYED_PAIR_FIELDS_QUAD
# ---------------------------------------------------------------------------


class TestIterQuadsShape:
    def test_yielded_dict_has_exactly_six_fields(self):
        """Every dict from iter_quads contains exactly the KEYED_PAIR_FIELDS_QUAD fields."""
        g = _make_simple_graph()
        quads = list(iter_quads(g))
        assert len(quads) == 1
        assert set(quads[0].keys()) == set(KEYED_PAIR_FIELDS_QUAD)

    def test_subject_and_object_come_from_graph_topology(self):
        """iter_quads reads subject/object from edge endpoints, not edge-data."""
        g = nx.MultiDiGraph()
        _add_keyed_edge(
            g,
            "SubjectNode",
            "ObjectNode",
            indexed_key="graph99",
            predicate="p",
            speaker_id="S",
            first_seen_cycle=0,
        )
        q = list(iter_quads(g))[0]
        assert q["subject"] == "SubjectNode"
        assert q["object"] == "ObjectNode"

    def test_multiple_keyed_edges_each_have_full_schema(self):
        """All six fields are present in every yielded dict."""
        g = nx.MultiDiGraph()
        for i in range(3):
            _add_keyed_edge(
                g,
                f"Subj{i}",
                f"Obj{i}",
                indexed_key=f"graph{i}",
                predicate="p",
                speaker_id="S",
                first_seen_cycle=i,
            )
        for quad in iter_quads(g):
            assert set(quad.keys()) == set(KEYED_PAIR_FIELDS_QUAD)


# ---------------------------------------------------------------------------
# 5. quad_by_key: hit and miss
# ---------------------------------------------------------------------------


class TestQuadByKey:
    def test_hit_returns_matching_dict(self):
        """quad_by_key returns the correct quad dict on a hit."""
        g = _make_simple_graph()
        result = quad_by_key(g, "graph1")
        assert result is not None
        assert result["key"] == "graph1"
        assert result["subject"] == _SUBJECT
        assert result["object"] == _OBJECT
        assert result["predicate"] == "lives_in"
        assert result["speaker_id"] == "Speaker0"
        assert result["first_seen_cycle"] == 3

    def test_miss_returns_none(self):
        """quad_by_key returns None when the key is absent."""
        g = _make_simple_graph()
        result = quad_by_key(g, "graph999")
        assert result is None

    def test_empty_graph_returns_none(self):
        """quad_by_key on an empty graph always returns None."""
        g = nx.MultiDiGraph()
        assert quad_by_key(g, "graph1") is None

    def test_returns_first_matching_edge(self):
        """When multiple edges share the same key, the first one is returned."""
        g = nx.MultiDiGraph()
        _add_keyed_edge(
            g, "A", "B", indexed_key="graph1", predicate="p1", speaker_id="S0", first_seen_cycle=1
        )
        _add_keyed_edge(
            g, "C", "D", indexed_key="graph1", predicate="p2", speaker_id="S1", first_seen_cycle=2
        )
        result = quad_by_key(g, "graph1")
        assert result is not None
        assert result["key"] == "graph1"


# ---------------------------------------------------------------------------
# 6. keys_for_entity: case-insensitive on subject and object
# ---------------------------------------------------------------------------


class TestKeysForEntity:
    def test_matches_subject_case_insensitively(self):
        """keys_for_entity matches subject regardless of case in the graph."""
        g = nx.MultiDiGraph()
        _add_keyed_edge(
            g,
            "Alice",
            "Berlin",
            indexed_key="graph1",
            predicate="lives_in",
            speaker_id="S0",
            first_seen_cycle=1,
        )
        # Lowercase query should match capitalised "Alice".
        result = keys_for_entity(g, "alice")
        assert "graph1" in result

    def test_matches_object_case_insensitively(self):
        """keys_for_entity matches the object node regardless of case."""
        g = nx.MultiDiGraph()
        _add_keyed_edge(
            g,
            "Charlie",
            "alice",
            indexed_key="graph2",
            predicate="knows",
            speaker_id="S0",
            first_seen_cycle=1,
        )
        # The object "alice" should match the query "alice".
        result = keys_for_entity(g, "alice")
        assert "graph2" in result

    def test_no_match_returns_empty_set(self):
        """keys_for_entity returns an empty set when no edge matches."""
        g = _make_simple_graph()
        result = keys_for_entity(g, "charlie")
        assert result == set()

    def test_multiple_matches_returns_all_keys(self):
        """All matching keys are returned, not just the first one."""
        g = nx.MultiDiGraph()
        _add_keyed_edge(
            g,
            "Alice",
            "Berlin",
            indexed_key="graph1",
            predicate="lives_in",
            speaker_id="S0",
            first_seen_cycle=1,
        )
        _add_keyed_edge(
            g,
            "Alice",
            "Engineer",
            indexed_key="graph2",
            predicate="has_job",
            speaker_id="S0",
            first_seen_cycle=2,
        )
        _add_keyed_edge(
            g,
            "Bob",
            "Alice",
            indexed_key="graph3",
            predicate="knows",
            speaker_id="S0",
            first_seen_cycle=3,
        )
        result = keys_for_entity(g, "alice")
        assert result == {"graph1", "graph2", "graph3"}

    def test_skips_keyless_edges(self):
        """Keyless edges are ignored (iter_quads skips them)."""
        g = nx.MultiDiGraph()
        g.add_edge("Alice", "Berlin")  # no "key" attribute in data
        result = keys_for_entity(g, "alice")
        assert result == set()


# ---------------------------------------------------------------------------
# 7. keys_for_speaker: exact speaker_id match
# ---------------------------------------------------------------------------


class TestKeysForSpeaker:
    def test_filters_by_speaker_id(self):
        """keys_for_speaker returns only keys belonging to the given speaker."""
        g = nx.MultiDiGraph()
        _add_keyed_edge(
            g,
            "Alice",
            "Berlin",
            indexed_key="graph1",
            predicate="lives_in",
            speaker_id="Speaker0",
            first_seen_cycle=1,
        )
        _add_keyed_edge(
            g,
            "Bob",
            "London",
            indexed_key="graph2",
            predicate="lives_in",
            speaker_id="Speaker1",
            first_seen_cycle=2,
        )
        assert keys_for_speaker(g, "Speaker0") == {"graph1"}
        assert keys_for_speaker(g, "Speaker1") == {"graph2"}

    def test_no_match_returns_empty_set(self):
        """keys_for_speaker returns an empty set when no edge matches."""
        g = _make_simple_graph()
        assert keys_for_speaker(g, "Speaker99") == set()

    def test_multiple_keys_same_speaker(self):
        """Multiple edges with the same speaker are all returned."""
        g = nx.MultiDiGraph()
        _add_keyed_edge(
            g,
            "A",
            "B",
            indexed_key="graph1",
            predicate="p",
            speaker_id="Speaker0",
            first_seen_cycle=1,
        )
        _add_keyed_edge(
            g,
            "C",
            "D",
            indexed_key="graph2",
            predicate="p",
            speaker_id="Speaker0",
            first_seen_cycle=2,
        )
        _add_keyed_edge(
            g,
            "E",
            "F",
            indexed_key="graph3",
            predicate="p",
            speaker_id="Speaker1",
            first_seen_cycle=3,
        )
        assert keys_for_speaker(g, "Speaker0") == {"graph1", "graph2"}
        assert keys_for_speaker(g, "Speaker1") == {"graph3"}


# ---------------------------------------------------------------------------
# 8. build_tier_graph_from_loop: happy path
# ---------------------------------------------------------------------------


class TestBuildTierGraphFromLoop:
    def _make_loop(self, *, simhash: dict, cache: dict) -> SimpleNamespace:
        """Build a minimal stub loop for testing build_tier_graph_from_loop."""
        return SimpleNamespace(
            episodic_simhash=simhash,
            semantic_simhash={},
            procedural_simhash={},
            indexed_key_cache=cache,
        )

    def test_happy_path_single_key(self):
        """build_tier_graph_from_loop produces a graph with the expected edge."""
        loop = self._make_loop(
            simhash={"graph1": 0xABCDEF},
            cache={
                "graph1": {
                    "key": "graph1",
                    "subject": "Alice",
                    "predicate": "lives_in",
                    "object": "Berlin",
                    "speaker_id": "Speaker0",
                    "first_seen_cycle": 3,
                }
            },
        )
        g = build_tier_graph_from_loop(loop, "episodic")
        assert g.number_of_edges() == 1
        # Verify via the public API (iter_quads maps ik_key → key).
        quads = list(iter_quads(g))
        assert len(quads) == 1
        q = quads[0]
        assert q["key"] == "graph1"
        assert q["subject"] == "Alice"
        assert q["object"] == "Berlin"
        assert q["predicate"] == "lives_in"
        assert q["speaker_id"] == "Speaker0"
        assert q["first_seen_cycle"] == 3

    def test_happy_path_multiple_keys(self):
        """All keys in the simhash registry are added to the graph."""
        loop = self._make_loop(
            simhash={"graph1": 1, "graph2": 2},
            cache={
                "graph1": {
                    "key": "graph1",
                    "subject": "Alice",
                    "predicate": "lives_in",
                    "object": "Berlin",
                    "speaker_id": "S0",
                    "first_seen_cycle": 1,
                },
                "graph2": {
                    "key": "graph2",
                    "subject": "Bob",
                    "predicate": "has_job",
                    "object": "Engineer",
                    "speaker_id": "S0",
                    "first_seen_cycle": 2,
                },
            },
        )
        g = build_tier_graph_from_loop(loop, "episodic")
        assert g.number_of_edges() == 2
        quads_by_key = {q["key"]: q for q in iter_quads(g)}
        assert "graph1" in quads_by_key
        assert "graph2" in quads_by_key

    def test_empty_simhash_returns_empty_graph(self):
        """An empty simhash registry produces an empty graph."""
        loop = self._make_loop(simhash={}, cache={})
        g = build_tier_graph_from_loop(loop, "episodic")
        assert g.number_of_edges() == 0

    def test_semantic_tier_is_routed_correctly(self):
        """build_tier_graph_from_loop uses the correct tier attribute."""
        loop = SimpleNamespace(
            episodic_simhash={},
            semantic_simhash={"graph10": 99},
            procedural_simhash={},
            indexed_key_cache={
                "graph10": {
                    "key": "graph10",
                    "subject": "X",
                    "predicate": "q",
                    "object": "Y",
                    "speaker_id": "S",
                    "first_seen_cycle": 0,
                },
            },
        )
        g = build_tier_graph_from_loop(loop, "semantic")
        assert g.number_of_edges() == 1
        quads = list(iter_quads(g))
        assert quads[0]["key"] == "graph10"


# ---------------------------------------------------------------------------
# 9. build_tier_graph_from_loop raises KeyError on cache miss
# ---------------------------------------------------------------------------


class TestBuildTierGraphKeyError:
    def test_raises_key_error_when_cache_missing_key(self):
        """build_tier_graph_from_loop raises KeyError when simhash key absent from cache."""
        loop = SimpleNamespace(
            episodic_simhash={"graph1": 0xABCDEF},
            semantic_simhash={},
            procedural_simhash={},
            indexed_key_cache={},  # "graph1" is absent — must raise
        )
        with pytest.raises(KeyError):
            build_tier_graph_from_loop(loop, "episodic")


# ---------------------------------------------------------------------------
# 10. Encryption round-trip: encrypted bytes are NOT plaintext JSON
# ---------------------------------------------------------------------------


class TestEncryptionRoundTrip:
    def test_encrypted_round_trip_produces_correct_graph(self, tmp_path, monkeypatch):
        """With daily passphrase, save+load returns the same graph."""
        _setup_daily(tmp_path, monkeypatch)
        g = _make_simple_graph()
        path = tmp_path / "graph.json"
        save_simulate_graph(g, path, encrypted=True)
        g2 = load_simulate_graph(path)
        assert g2.number_of_edges() == 1
        quads = list(iter_quads(g2))
        assert quads[0]["key"] == "graph1"
        assert quads[0]["subject"] == _SUBJECT
        assert quads[0]["object"] == _OBJECT

    def test_encrypted_bytes_are_not_plaintext_json(self, tmp_path, monkeypatch):
        """On-disk bytes are age-encrypted — do not contain the JSON 'directed' marker."""
        _setup_daily(tmp_path, monkeypatch)
        g = _make_simple_graph()
        path = tmp_path / "graph.json"
        save_simulate_graph(g, path, encrypted=True)
        raw = path.read_bytes()
        assert raw.startswith(AGE_MAGIC), f"expected age envelope, got {raw[:40]!r}"
        # The plaintext marker ("directed" is in every nx.node_link_data output)
        # must NOT appear in the raw bytes.
        assert b"directed" not in raw

    def test_plaintext_write_is_readable_json(self, tmp_path, monkeypatch):
        """encrypted=False writes inspectable plaintext JSON."""
        monkeypatch.setattr(
            "paramem.backup.key_store.DAILY_KEY_PATH_DEFAULT",
            tmp_path / "absent.age",
        )
        monkeypatch.delenv(DAILY_PASSPHRASE_ENV_VAR, raising=False)
        g = _make_simple_graph()
        path = tmp_path / "graph.json"
        save_simulate_graph(g, path, encrypted=False)
        raw = path.read_bytes()
        # Plaintext must be valid JSON and contain the "directed" key.
        parsed = json.loads(raw.decode("utf-8"))
        assert "directed" in parsed
