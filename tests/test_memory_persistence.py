"""Unit tests for paramem.memory.persistence.

Covers round-trip contract, encryption awareness, iter_entries edge-skipping,
entry_by_key hit/miss, entity index helpers, build_tier_graph_from_store, and
reap_tier_artifacts' rename-then-delete crash safety (+ resume_pending_reaps).
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

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
from paramem.memory.persistence import (
    _EDGE_SOURCE_ATTR,
    _IK_KEY_ATTR,
    _PENDING_DELETE_DIR_NAME,
    build_tier_graph_from_store,
    entry_by_key,
    erase_keys_and_restamp_manifest,
    erase_keys_from_graph_file,
    iter_entries,
    load_memory_from_disk,
    reap_tier_artifacts,
    resume_pending_reaps,
    save_memory_to_disk,
)

# Canonical entry schema (current shape: five-field quintuple).
KEYED_ENTRY_FIELDS = ("key", "subject", "predicate", "object", "speaker_id")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EDGE_DATA = {
    "key": "graph1",
    "predicate": "lives_in",
    "speaker_id": "speaker0",
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
) -> None:
    """Test-local wrapper around memory_persistence._add_keyed_edge.

    Delegates to the production helper so tests exercise the same code path
    that ``build_tier_graph_from_store`` uses.  The indexed-memory key is
    stored as ``"ik_key"`` in edge data (never as the NetworkX multigraph
    edge-key parameter) so it survives ``nx.node_link_data`` /
    ``nx.node_link_graph`` round-trips intact.
    """
    from paramem.memory.persistence import _add_keyed_edge as _prod_add

    _prod_add(
        graph,
        subject,
        object_,
        indexed_key=indexed_key,
        predicate=predicate,
        speaker_id=speaker_id,
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
        """save_memory_to_disk then load returns graph with same edge attributes."""
        g = _make_simple_graph()
        path = tmp_path / "graph.json"
        save_memory_to_disk(g, path)
        g2 = load_memory_from_disk(path)

        # Verify internal storage via iter_entries (public API).
        entries = list(iter_entries(g2))
        assert len(entries) == 1
        q = entries[0]
        assert q["key"] == "graph1"
        assert q["subject"] == _SUBJECT
        assert q["object"] == _OBJECT
        assert q["predicate"] == "lives_in"
        assert q["speaker_id"] == "speaker0"

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
        )
        _add_keyed_edge(
            g,
            "Bob",
            "Engineer",
            indexed_key="graph2",
            predicate="has_job",
            speaker_id="S1",
        )
        path = tmp_path / "graph.json"
        save_memory_to_disk(g, path)
        g2 = load_memory_from_disk(path)
        assert g2.number_of_edges() == 2
        # Verify both keys are present in edge data.
        keys = {q["key"] for q in iter_entries(g2)}
        assert keys == {"graph1", "graph2"}

    def test_round_trip_creates_parent_directory(self, tmp_path):
        """save_memory_to_disk creates missing parent directories."""
        path = tmp_path / "subdir" / "nested" / "graph.json"
        save_memory_to_disk(_make_simple_graph(), path)
        assert path.exists()

    def test_round_trip_empty_graph(self, tmp_path):
        """An empty graph survives the round-trip."""
        g = nx.MultiDiGraph()
        path = tmp_path / "graph.json"
        save_memory_to_disk(g, path)
        g2 = load_memory_from_disk(path)
        assert g2.number_of_edges() == 0
        assert g2.number_of_nodes() == 0

    def test_round_trip_preserves_edge_source_provenance(self, tmp_path):
        """Edge provenance under _EDGE_SOURCE_ATTR survives save→load intact.

        Regression for the reserved-key collision: an edge attribute named
        "source" is silently overwritten by NetworkX's node_link_data with the
        source-NODE name on persist (and lost on reload).  Provenance is stored
        under _EDGE_SOURCE_ATTR ("edge_source") to dodge the collision — same
        class as "key" → "ik_key".  This test verifies the renamed attribute
        survives, AND that a co-present ik_key / predicate survive alongside it.
        """
        g = nx.MultiDiGraph()
        eid = g.add_edge(
            "dana vex",
            "acme corp",
            predicate="works at",
            confidence=0.9,
            **{_EDGE_SOURCE_ATTR: "graph_enrichment"},
        )
        g["dana vex"]["acme corp"][eid][_IK_KEY_ATTR] = "graph7"

        path = tmp_path / "graph.json"
        save_memory_to_disk(g, path)
        g2 = load_memory_from_disk(path)

        edges = list(g2.edges(keys=True, data=True))
        assert len(edges) == 1
        subj, obj, _nx_key, data = edges[0]
        # Topology endpoints reconstructed correctly (these consume the reserved
        # node_link "source"/"target" fields).
        assert subj == "dana vex"
        assert obj == "acme corp"
        # Provenance tag survives — the bug clobbered this to the node name.
        assert data[_EDGE_SOURCE_ATTR] == "graph_enrichment"
        # Co-present attributes survive alongside it.
        assert data[_IK_KEY_ATTR] == "graph7"
        assert data["predicate"] == "works at"


# ---------------------------------------------------------------------------
# 2. load_memory_from_disk on missing path returns empty MultiDiGraph — no raise
# ---------------------------------------------------------------------------


class TestLoadMissingPath:
    def test_missing_path_returns_empty_multigraph(self, tmp_path):
        """load_memory_from_disk returns an empty MultiDiGraph when path is absent."""
        path = tmp_path / "does_not_exist.json"
        g = load_memory_from_disk(path)
        assert isinstance(g, nx.MultiDiGraph)
        assert g.number_of_nodes() == 0
        assert g.number_of_edges() == 0

    def test_missing_path_does_not_raise(self, tmp_path):
        """No exception is raised when path is absent."""
        path = tmp_path / "absent" / "graph.json"
        # No FileNotFoundError or OSError expected.
        g = load_memory_from_disk(path)
        assert isinstance(g, nx.MultiDiGraph)


# ---------------------------------------------------------------------------
# 3. iter_entries skips edges without a "key" attribute
# ---------------------------------------------------------------------------


class TestIterEntriesSkipsKeylessEdges:
    def test_skips_edge_without_key(self):
        """iter_entries yields only edges that carry a 'key' attribute."""
        g = nx.MultiDiGraph()
        # Edge WITH key — added via helper so 'key' lands in edge data.
        _add_keyed_edge(
            g,
            "Alice",
            "Berlin",
            indexed_key="graph1",
            predicate="lives_in",
            speaker_id="S0",
        )
        # Edge WITHOUT key — direct add_edge, no 'key' attribute in data.
        g.add_edge(
            "Alice",
            "Paris",
            predicate="visited",
            speaker_id="S0",
        )
        entries = list(iter_entries(g))
        assert len(entries) == 1
        assert entries[0]["key"] == "graph1"

    def test_empty_graph_yields_nothing(self):
        """iter_entries on an empty graph yields no items."""
        assert list(iter_entries(nx.MultiDiGraph())) == []

    def test_graph_with_no_keyed_edges_yields_nothing(self):
        """A graph where no edges carry 'key' yields no items."""
        g = nx.MultiDiGraph()
        g.add_edge(
            "A",
            "B",
            predicate="related",
            speaker_id="S0",
        )
        assert list(iter_entries(g)) == []


# ---------------------------------------------------------------------------
# 4. iter_entries shape: every yielded dict has exactly the KEYED_ENTRY_FIELDS
# ---------------------------------------------------------------------------


class TestIterEntriesShape:
    def test_yielded_dict_has_exactly_six_fields(self):
        """Every dict from iter_entries contains exactly the KEYED_ENTRY_FIELDS."""
        g = _make_simple_graph()
        entries = list(iter_entries(g))
        assert len(entries) == 1
        assert set(entries[0].keys()) == set(KEYED_ENTRY_FIELDS)

    def test_subject_and_object_come_from_graph_topology(self):
        """iter_entries reads subject/object from edge endpoints, not edge-data."""
        g = nx.MultiDiGraph()
        _add_keyed_edge(
            g,
            "SubjectNode",
            "ObjectNode",
            indexed_key="graph99",
            predicate="p",
            speaker_id="S",
        )
        q = list(iter_entries(g))[0]
        assert q["subject"] == "SubjectNode"
        assert q["object"] == "ObjectNode"

    def test_multiple_keyed_edges_each_have_full_schema(self):
        """All canonical fields are present in every yielded dict."""
        g = nx.MultiDiGraph()
        for i in range(3):
            _add_keyed_edge(
                g,
                f"Subj{i}",
                f"Obj{i}",
                indexed_key=f"graph{i}",
                predicate="p",
                speaker_id="S",
            )
        for entry in iter_entries(g):
            assert set(entry.keys()) == set(KEYED_ENTRY_FIELDS)


# ---------------------------------------------------------------------------
# 5. entry_by_key: hit and miss
# ---------------------------------------------------------------------------


class TestEntryByKey:
    def test_hit_returns_matching_dict(self):
        """entry_by_key returns the correct entry dict on a hit."""
        g = _make_simple_graph()
        result = entry_by_key(g, "graph1")
        assert result is not None
        assert result["key"] == "graph1"
        assert result["subject"] == _SUBJECT
        assert result["object"] == _OBJECT
        assert result["predicate"] == "lives_in"
        assert result["speaker_id"] == "speaker0"

    def test_miss_returns_none(self):
        """entry_by_key returns None when the key is absent."""
        g = _make_simple_graph()
        result = entry_by_key(g, "graph999")
        assert result is None

    def test_empty_graph_returns_none(self):
        """entry_by_key on an empty graph always returns None."""
        g = nx.MultiDiGraph()
        assert entry_by_key(g, "graph1") is None

    def test_returns_first_matching_edge(self):
        """When multiple edges share the same key, the first one is returned."""
        g = nx.MultiDiGraph()
        _add_keyed_edge(g, "A", "B", indexed_key="graph1", predicate="p1", speaker_id="S0")
        _add_keyed_edge(g, "C", "D", indexed_key="graph1", predicate="p2", speaker_id="S1")
        result = entry_by_key(g, "graph1")
        assert result is not None
        assert result["key"] == "graph1"


# ---------------------------------------------------------------------------
# 8. build_tier_graph_from_store: happy path
# ---------------------------------------------------------------------------


class TestBuildTierGraphFromStore:
    def _make_store(
        self, *, simhash: dict, cache: dict, bookkeeping: dict | None = None, tier: str = "episodic"
    ):
        """Build a minimal MemoryStore for testing build_tier_graph_from_store.

        *bookkeeping* maps ``key -> speaker_id``, installed via
        ``set_bookkeeping``: ``build_tier_graph_from_store`` sources
        ``speaker_id`` from bookkeeping, never from the entry, so tests that
        want a non-empty ``speaker_id`` on the persisted edge must seed it
        here rather than on the cache entry passed to ``store.put``.
        """
        from paramem.memory.store import MemoryStore

        store = MemoryStore(replay_enabled=False)
        store.replace_simhashes_in_tier(tier, simhash)
        for k, q in cache.items():
            store.put(tier, k, q, register=False)
        for k, speaker_id in (bookkeeping or {}).items():
            store.set_bookkeeping(k, speaker_id=speaker_id, relation_type="factual", first_seen="")
        return store

    def test_happy_path_single_key(self):
        """build_tier_graph_from_store produces a graph with the expected edge,
        with speaker_id sourced from bookkeeping rather than the entry."""
        store = self._make_store(
            simhash={"graph1": 0xABCDEF},
            cache={
                "graph1": {
                    "key": "graph1",
                    "subject": "Alice",
                    "predicate": "lives_in",
                    "object": "Berlin",
                }
            },
            bookkeeping={"graph1": "speaker0"},
        )
        g = build_tier_graph_from_store(store, "episodic")
        assert g.number_of_edges() == 1
        entries = list(iter_entries(g))
        assert len(entries) == 1
        q = entries[0]
        assert q["key"] == "graph1"
        assert q["subject"] == "Alice"
        assert q["object"] == "Berlin"
        assert q["predicate"] == "lives_in"
        assert q["speaker_id"] == "speaker0"

    def test_happy_path_multiple_keys(self):
        """All keys in the simhash registry are added to the graph."""
        store = self._make_store(
            simhash={"graph1": 1, "graph2": 2},
            cache={
                "graph1": {
                    "key": "graph1",
                    "subject": "Alice",
                    "predicate": "lives_in",
                    "object": "Berlin",
                },
                "graph2": {
                    "key": "graph2",
                    "subject": "Bob",
                    "predicate": "has_job",
                    "object": "Engineer",
                },
            },
            bookkeeping={"graph1": "S0", "graph2": "S0"},
        )
        g = build_tier_graph_from_store(store, "episodic")
        assert g.number_of_edges() == 2
        entries_by_key = {q["key"]: q for q in iter_entries(g)}
        assert "graph1" in entries_by_key
        assert "graph2" in entries_by_key

    def test_empty_simhash_returns_empty_graph(self):
        """An empty simhash registry produces an empty graph."""
        store = self._make_store(simhash={}, cache={})
        g = build_tier_graph_from_store(store, "episodic")
        assert g.number_of_edges() == 0

    def test_semantic_tier_is_routed_correctly(self):
        """build_tier_graph_from_store uses the correct tier attribute."""
        store = self._make_store(
            simhash={"graph10": 99},
            cache={
                "graph10": {
                    "key": "graph10",
                    "subject": "X",
                    "predicate": "q",
                    "object": "Y",
                },
            },
            bookkeeping={"graph10": "S"},
            tier="semantic",
        )
        g = build_tier_graph_from_store(store, "semantic")
        assert g.number_of_edges() == 1
        entries = list(iter_entries(g))
        assert entries[0]["key"] == "graph10"

    def test_no_bookkeeping_persists_empty_speaker_id(self):
        """A key with no bookkeeping record falls back to speaker_id=''."""
        store = self._make_store(
            simhash={"graph1": 0xABCDEF},
            cache={
                "graph1": {
                    "key": "graph1",
                    "subject": "Alice",
                    "predicate": "lives_in",
                    "object": "Berlin",
                }
            },
        )
        assert store.bookkeeping_for_key("graph1") is None
        g = build_tier_graph_from_store(store, "episodic")
        entries = list(iter_entries(g))
        assert entries[0]["speaker_id"] == ""

    def test_memoized_cache_miss_entry_preserves_bookkeeping_speaker_id(self):
        """Regression lock for the simulate-mode fold-persist bug: a key whose
        entry is re-materialised as a content-only miss (the shape
        ``MemoryStore.probe`` memoizes back on a source-served cache miss)
        must still persist with its real speaker_id, because attribution is
        read from bookkeeping and never from the entry."""
        from paramem.memory.entry import entry_simhash
        from paramem.memory.store import MemoryStore

        store = MemoryStore(replay_enabled=True)
        store.set_bookkeeping(
            "graph1", speaker_id="speaker0", relation_type="factual", first_seen=""
        )
        ep_reg = store.registry("episodic")
        ep_reg.add("graph1")

        entry = {
            "key": "graph1",
            "subject": "Alice",
            "predicate": "lives_in",
            "object": "Berlin",
        }

        class _StubSource:
            def probe(self, keys_by_tier):
                return {"graph1": dict(entry)}

        # Matching fingerprint so the store's confidence gate admits the
        # source-served result on this cache miss.
        store.put_simhash("episodic", "graph1", entry_simhash(entry))
        store.probe({"episodic": ["graph1"]}, source=_StubSource(), memoize=True)
        # The memoized entry is content-only — no speaker_id on it.
        assert "speaker_id" not in store.get("graph1")

        g = build_tier_graph_from_store(store, "episodic")
        entries = list(iter_entries(g))
        assert len(entries) == 1
        assert entries[0]["speaker_id"] == "speaker0"


# ---------------------------------------------------------------------------
# 9. build_tier_graph_from_store raises KeyError on cache miss
# ---------------------------------------------------------------------------


class TestBuildTierGraphKeyError:
    def test_raises_key_error_when_cache_missing_key(self):
        """build_tier_graph_from_store raises KeyError when simhash key absent from store."""
        from paramem.memory.store import MemoryStore

        store = MemoryStore(replay_enabled=False)
        store.put_simhash("episodic", "graph1", 0xABCDEF)
        # "graph1" simhash present but entry absent — must raise.
        with pytest.raises(KeyError):
            build_tier_graph_from_store(store, "episodic")


# ---------------------------------------------------------------------------
# 9b. build_tier_graph_from_store: stale-key projection
# ---------------------------------------------------------------------------


class TestBuildTierGraphStaleProjection:
    """Stale keys must NOT be projected into graph.json, but their simhash
    entries are retained on disk.

    The enumeration spine is ``tier_simhashes(include_stale=False)`` so only
    active keys are projected; replay-disabled stores have no registry and rely
    on the simhash map populated via ``replace_simhashes_in_tier``.
    """

    def test_stale_key_excluded_from_graph_no_key_error(self):
        """A stale key in the simhash dict is skipped — no KeyError, not in graph."""
        from paramem.memory.store import MemoryStore

        store = MemoryStore(replay_enabled=True)
        # Register active key with simhash + entry.
        store.put(
            "episodic",
            "graph_active",
            {
                "key": "graph_active",
                "subject": "Alice",
                "predicate": "lives_in",
                "object": "Berlin",
                "speaker_id": "S0",
            },
            simhash=0xAAAA1111,
            register=True,
        )
        # Register stale key — simhash present, entry INTENTIONALLY absent.
        # (Mimics the scenario where the stale key's entry was reaped but the
        # simhash is retained for the stale-echo seam.)
        store.put_simhash("episodic", "graph_stale", 0xBBBB2222)
        # Flip to stale via the registry.
        ep_reg = store.registry("episodic")
        ep_reg.add("graph_stale")  # must be active to stale
        ep_reg.stale("graph_stale")

        g = build_tier_graph_from_store(store, "episodic")

        # Active key projects into graph.
        entries = list(iter_entries(g))
        assert len(entries) == 1, f"Expected 1 edge (active only); got {entries}"
        assert entries[0]["key"] == "graph_active"

        # Stale simhash is RETAINED in the known (active∪stale) fingerprint map.
        assert "graph_stale" in store.tier_simhashes("episodic", include_stale=True), (
            "Stale key simhash must be retained on the store"
        )

    def test_active_key_without_entry_still_raises(self):
        """An active key in the simhash map that has no entry raises KeyError.

        This guards against stale-key filtering silently suppressing data-integrity
        errors on active keys.
        """
        from paramem.memory.store import MemoryStore

        store = MemoryStore(replay_enabled=True)
        # Active key in simhash but no entry — must raise.
        store.put_simhash("episodic", "graph_active_no_entry", 0x1234)
        ep_reg = store.registry("episodic")
        ep_reg.add("graph_active_no_entry")

        with pytest.raises(KeyError):
            build_tier_graph_from_store(store, "episodic")

    def test_existing_happy_path_still_works_replay_disabled(self):
        """Replay-disabled store (the existing happy-path pattern) still projects correctly.

        The stale-key projection filters via the registry's own active/stale
        distinction (``tier_simhashes(include_stale=False)``) rather than the
        store's ``replay_enabled`` flag, precisely to keep replay-disabled
        stores working.  This test mirrors the existing TestBuildTierGraphFromStore
        setup (MemoryStore(replay_enabled=False), register=False) to confirm that
        path is unaffected.
        """
        from paramem.memory.store import MemoryStore

        store = MemoryStore(replay_enabled=False)
        store.replace_simhashes_in_tier("episodic", {"graph1": 0xABCDEF})
        store.put(
            "episodic",
            "graph1",
            {
                "key": "graph1",
                "subject": "Bob",
                "predicate": "has_job",
                "object": "Engineer",
                "speaker_id": "S0",
            },
            register=False,
        )

        g = build_tier_graph_from_store(store, "episodic")
        entries = list(iter_entries(g))
        assert len(entries) == 1
        assert entries[0]["key"] == "graph1"


# ---------------------------------------------------------------------------
# 10. Encryption round-trip: encrypted bytes are NOT plaintext JSON
# ---------------------------------------------------------------------------


class TestEncryptionRoundTrip:
    def test_encrypted_round_trip_produces_correct_graph(self, tmp_path, monkeypatch):
        """With daily passphrase, save+load returns the same graph."""
        _setup_daily(tmp_path, monkeypatch)
        g = _make_simple_graph()
        path = tmp_path / "graph.json"
        save_memory_to_disk(g, path)
        g2 = load_memory_from_disk(path)
        assert g2.number_of_edges() == 1
        entries = list(iter_entries(g2))
        assert entries[0]["key"] == "graph1"
        assert entries[0]["subject"] == _SUBJECT
        assert entries[0]["object"] == _OBJECT

    def test_encrypted_bytes_are_not_plaintext_json(self, tmp_path, monkeypatch):
        """On-disk bytes are age-encrypted — do not contain the JSON 'directed' marker."""
        _setup_daily(tmp_path, monkeypatch)
        g = _make_simple_graph()
        path = tmp_path / "graph.json"
        save_memory_to_disk(g, path)
        raw = path.read_bytes()
        assert raw.startswith(AGE_MAGIC), f"expected age envelope, got {raw[:40]!r}"
        # The plaintext marker ("directed" is in every nx.node_link_data output)
        # must NOT appear in the raw bytes.
        assert b"directed" not in raw

    def test_plaintext_write_is_readable_json(self, tmp_path, monkeypatch):
        """No daily identity loaded (Security OFF) → inspectable plaintext JSON.

        The posture is the only thing that decides this now: there is no
        per-call override, so a caller cannot write plaintext behind the
        operator's back.
        """
        monkeypatch.setattr(
            "paramem.backup.key_store.DAILY_KEY_PATH_DEFAULT",
            tmp_path / "absent.age",
        )
        monkeypatch.delenv(DAILY_PASSPHRASE_ENV_VAR, raising=False)
        g = _make_simple_graph()
        path = tmp_path / "graph.json"
        save_memory_to_disk(g, path)
        raw = path.read_bytes()
        # Plaintext must be valid JSON and contain the "directed" key.
        parsed = json.loads(raw.decode("utf-8"))
        assert "directed" in parsed


# ---------------------------------------------------------------------------
# 11. erase_keys_from_graph_file: surgical edge removal
# ---------------------------------------------------------------------------


class TestEraseKeysFromGraphFile:
    def test_missing_file_returns_zero_no_write(self, tmp_path):
        """Absent file → 0, and no file is created."""
        path = tmp_path / "graph.json"
        result = erase_keys_from_graph_file(path, {"graph1"})
        assert result == 0
        assert not path.exists()

    def test_no_matching_edge_returns_zero_file_unchanged(self, tmp_path):
        """A present file with no edge matching *keys* → 0, bytes unchanged."""
        g = _make_simple_graph()  # single edge, key="graph1"
        path = tmp_path / "graph.json"
        save_memory_to_disk(g, path)
        before = path.read_bytes()

        result = erase_keys_from_graph_file(path, {"graph_absent"})

        assert result == 0
        assert path.read_bytes() == before

    def test_removes_matching_edge_keeps_surviving_edge(self, tmp_path):
        """The named key's edge is removed; a sibling edge's data is unchanged."""
        g = nx.MultiDiGraph()
        _add_keyed_edge(
            g, "Alice", "Berlin", indexed_key="graph1", predicate="lives_in", speaker_id="S0"
        )
        _add_keyed_edge(
            g, "Bob", "Engineer", indexed_key="graph2", predicate="has_job", speaker_id="S1"
        )
        path = tmp_path / "graph.json"
        save_memory_to_disk(g, path)

        removed = erase_keys_from_graph_file(path, {"graph1"})

        assert removed == 1
        g2 = load_memory_from_disk(path)
        entries = {e["key"]: e for e in iter_entries(g2)}
        assert "graph1" not in entries
        assert "graph2" in entries
        assert entries["graph2"]["subject"] == "Bob"
        assert entries["graph2"]["object"] == "Engineer"
        assert entries["graph2"]["predicate"] == "has_job"
        assert entries["graph2"]["speaker_id"] == "S1"

    def test_removes_multiple_keys_in_one_pass(self, tmp_path):
        """All edges whose ik_key is in *keys* are removed in a single write."""
        g = nx.MultiDiGraph()
        _add_keyed_edge(
            g, "Alice", "Berlin", indexed_key="graph1", predicate="lives_in", speaker_id="S0"
        )
        _add_keyed_edge(
            g, "Bob", "Engineer", indexed_key="graph2", predicate="has_job", speaker_id="S1"
        )
        _add_keyed_edge(
            g, "Carl", "Chess", indexed_key="graph3", predicate="likes", speaker_id="S2"
        )
        path = tmp_path / "graph.json"
        save_memory_to_disk(g, path)

        removed = erase_keys_from_graph_file(path, {"graph1", "graph3"})

        assert removed == 2
        g2 = load_memory_from_disk(path)
        keys = {e["key"] for e in iter_entries(g2)}
        assert keys == {"graph2"}

    def test_isolated_node_dropped_after_edge_removal(self, tmp_path):
        """A node left with degree 0 by the erase is dropped from the graph."""
        g = _make_simple_graph(subject="Alice", object_="Berlin")  # only edge: Alice->Berlin
        path = tmp_path / "graph.json"
        save_memory_to_disk(g, path)

        removed = erase_keys_from_graph_file(path, {"graph1"})

        assert removed == 1
        g2 = load_memory_from_disk(path)
        assert g2.number_of_nodes() == 0
        assert "Alice" not in g2
        assert "Berlin" not in g2

    def test_node_still_in_use_by_surviving_edge_is_kept(self, tmp_path):
        """A node shared by an erased edge and a surviving edge is not dropped."""
        g = nx.MultiDiGraph()
        _add_keyed_edge(
            g, "Alice", "Berlin", indexed_key="graph1", predicate="lives_in", speaker_id="S0"
        )
        _add_keyed_edge(
            g, "Alice", "Engineer", indexed_key="graph2", predicate="has_job", speaker_id="S0"
        )
        path = tmp_path / "graph.json"
        save_memory_to_disk(g, path)

        removed = erase_keys_from_graph_file(path, {"graph1"})

        assert removed == 1
        g2 = load_memory_from_disk(path)
        assert "Alice" in g2
        assert "Berlin" not in g2
        entries = list(iter_entries(g2))
        assert len(entries) == 1
        assert entries[0]["key"] == "graph2"

    def test_write_stays_atomic_and_envelope_aware(self, tmp_path, monkeypatch):
        """The write goes through save_memory_to_disk: an age-wrapped file
        stays age-wrapped after erase, and the survivor decrypts correctly."""
        _setup_daily(tmp_path, monkeypatch)
        g = nx.MultiDiGraph()
        _add_keyed_edge(
            g, "Alice", "Berlin", indexed_key="graph1", predicate="lives_in", speaker_id="S0"
        )
        _add_keyed_edge(
            g, "Bob", "Engineer", indexed_key="graph2", predicate="has_job", speaker_id="S1"
        )
        path = tmp_path / "graph.json"
        save_memory_to_disk(g, path)

        removed = erase_keys_from_graph_file(path, {"graph1"})

        assert removed == 1
        raw = path.read_bytes()
        assert raw.startswith(AGE_MAGIC), "erase write must stay age-wrapped"
        g2 = load_memory_from_disk(path)
        keys = {e["key"] for e in iter_entries(g2)}
        assert keys == {"graph2"}

    def test_returns_int_count(self, tmp_path):
        """The return value is an int, not a bool or other truthy type."""
        g = _make_simple_graph()
        path = tmp_path / "graph.json"
        save_memory_to_disk(g, path)
        result = erase_keys_from_graph_file(path, {"graph1"})
        assert type(result) is int


# ---------------------------------------------------------------------------
# reap_tier_artifacts — shape derived from tier_root, never a caller flag
# ---------------------------------------------------------------------------


class TestReapTierArtifacts:
    def test_main_tier_root_keeps_interim_children(self, tmp_path):
        """A main tier root (e.g. ``episodic/``) keeps its ``interim_*``
        children — those are separate tiers, not this tier's own artifacts."""
        tier_root = tmp_path / "episodic"
        slot = tier_root / "20260417-120000"
        slot.mkdir(parents=True)
        (slot / "adapter_model.safetensors").write_bytes(b"")
        interim = tier_root / "interim_20260417T0000"
        interim.mkdir()
        (interim / "adapter_model.safetensors").write_bytes(b"")

        removed = reap_tier_artifacts(tier_root)

        assert not slot.exists()
        assert interim.exists()
        assert tier_root.exists(), "root must survive — an interim child remains"
        assert slot in removed
        assert interim not in removed

    def test_semantic_root_removes_interim_scratch_children(self, tmp_path):
        """Only ``episodic/`` spares ``interim_*`` children — under
        ``semantic``/``procedural`` an ``interim_<stamp>/`` dir is HF Trainer
        scratch (``ConsolidationLoop._training_output_dir`` builds one for
        any adapter's interim training scope), not a separate tier, so it
        must be removed like any other child and the root must fully empty
        (including rmdir'ing the root itself)."""
        tier_root = tmp_path / "semantic"
        slot = tier_root / "20260417-120000"
        slot.mkdir(parents=True)
        (slot / "adapter_model.safetensors").write_bytes(b"")
        scratch = tier_root / "interim_20260417T0000"
        scratch.mkdir()
        (scratch / "checkpoint-10").mkdir()

        removed = reap_tier_artifacts(tier_root)

        assert not tier_root.exists(), "root must be fully removed, not left with scratch"
        assert not scratch.exists()
        assert tier_root in removed
        assert scratch in removed

    def test_main_tier_root_rmdir_when_emptied(self, tmp_path):
        """A main tier root with no ``interim_*`` children is removed itself
        once its other children are gone."""
        tier_root = tmp_path / "semantic"
        slot = tier_root / "20260417-120000"
        slot.mkdir(parents=True)
        (slot / "adapter_model.safetensors").write_bytes(b"")

        removed = reap_tier_artifacts(tier_root)

        assert not tier_root.exists()
        assert tier_root in removed
        assert slot in removed

    def test_interim_slot_root_removed_whole(self, tmp_path):
        """An interim slot root (name starts with ``interim_``) is removed
        in its entirety, including nested content."""
        episodic_root = tmp_path / "episodic"
        interim_root = episodic_root / "interim_20260417T0000"
        nested_slot = interim_root / "20260417-120000"
        nested_slot.mkdir(parents=True)
        (nested_slot / "adapter_model.safetensors").write_bytes(b"")

        removed = reap_tier_artifacts(interim_root)

        assert not interim_root.exists()
        assert interim_root in removed
        assert nested_slot in removed

    def test_absent_root_returns_empty_list(self, tmp_path):
        """A tier_root that does not exist on disk returns []."""
        missing = tmp_path / "episodic"
        assert reap_tier_artifacts(missing) == []

    def test_idempotent_second_call(self, tmp_path):
        """Calling reap_tier_artifacts a second time on an already-reaped
        root is a safe no-op returning []."""
        tier_root = tmp_path / "procedural"
        slot = tier_root / "20260417-120000"
        slot.mkdir(parents=True)

        first = reap_tier_artifacts(tier_root)
        second = reap_tier_artifacts(tier_root)

        assert first != []
        assert second == []
        assert not tier_root.exists()

    def test_removed_paths_are_deepest_first(self, tmp_path):
        """Returned paths order files/subdirs ahead of their parent so a
        caller replaying the list top-to-bottom never orphans anything."""
        tier_root = tmp_path / "episodic"
        slot = tier_root / "20260417-120000"
        slot.mkdir(parents=True)
        (slot / "adapter_model.safetensors").write_bytes(b"")

        removed = reap_tier_artifacts(tier_root)

        depths = [len(p.parts) for p in removed]
        assert depths == sorted(depths, reverse=True)

    def test_no_pending_delete_leftover_on_success(self, tmp_path):
        """A fully successful reap (no crash) leaves no ``.pending-delete``
        directory at all — every tombstone entry is removed immediately
        after its own rename, and the now-empty tombstone dir is cleaned up
        too."""
        tier_root = tmp_path / "procedural"
        slot = tier_root / "20260417-120000"
        slot.mkdir(parents=True)
        (slot / "adapter_model.safetensors").write_bytes(b"")

        reap_tier_artifacts(tier_root)

        assert not (tmp_path / _PENDING_DELETE_DIR_NAME).exists()


# ---------------------------------------------------------------------------
# reap_tier_artifacts — rename-then-delete crash safety (tombstone)
# ---------------------------------------------------------------------------


class TestReapTierArtifactsTombstone:
    """A condemned root is renamed into ``.pending-delete/<name>`` before
    being deleted there. A crash between the rename and the delete leaves
    the corpse stranded under the tombstone dir — already out of the live
    namespace — rather than half-deleted in its original location."""

    def test_crash_after_rename_leaves_interim_slot_out_of_live_namespace(
        self, tmp_path, monkeypatch
    ):
        """A shutil.rmtree failure AFTER the rename step leaves the
        condemned interim slot stranded under .pending-delete/, fully out
        of the live namespace and invisible to iter_interim_dirs — never
        half-deleted in its original location. Reverting the rename step
        (monkeypatching os.rename to a no-op, i.e. restoring the old direct
        shutil.rmtree(root) behaviour) would leave interim_root sitting in
        the live namespace instead, which the assertions below reject."""
        from paramem.memory.interim_adapter import iter_interim_dirs

        adapter_dir = tmp_path
        episodic_root = adapter_dir / "episodic"
        interim_root = episodic_root / "interim_20260417T0000"
        nested_slot = interim_root / "20260417-120000"
        nested_slot.mkdir(parents=True)
        (nested_slot / "adapter_model.safetensors").write_bytes(b"")
        (interim_root / "indexed_key_registry.json").write_text("{}")

        real_rmtree = shutil.rmtree

        def _boom(path, *a, **kw):
            raise OSError("simulated crash mid-delete")

        monkeypatch.setattr(shutil, "rmtree", _boom)
        try:
            with pytest.raises(OSError):
                reap_tier_artifacts(interim_root)
        finally:
            monkeypatch.setattr(shutil, "rmtree", real_rmtree)

        # Out of the live namespace immediately — the rename already ran.
        assert not interim_root.exists()
        assert list(iter_interim_dirs(adapter_dir)) == []

        # Stranded under the tombstone dir, not yet actually deleted.
        tombstone = adapter_dir / _PENDING_DELETE_DIR_NAME / "interim_20260417T0000"
        assert tombstone.exists()
        assert (tombstone / "20260417-120000" / "adapter_model.safetensors").exists()

    def test_crash_during_main_tier_reap_preserves_interim_children(self, tmp_path, monkeypatch):
        """A crash while reaping a main tier's non-interim child leaves the
        sibling interim_* child completely untouched — the main-tier
        branch's interim-sparing behaviour survives the tombstone rewrite."""
        tier_root = tmp_path / "episodic"
        slot = tier_root / "20260417-120000"
        slot.mkdir(parents=True)
        (slot / "adapter_model.safetensors").write_bytes(b"")
        interim = tier_root / "interim_20260417T0000"
        interim.mkdir()
        (interim / "adapter_model.safetensors").write_bytes(b"")

        def _boom(path, *a, **kw):
            raise OSError("simulated crash mid-delete")

        real_rmtree = shutil.rmtree
        monkeypatch.setattr(shutil, "rmtree", _boom)
        try:
            with pytest.raises(OSError):
                reap_tier_artifacts(tier_root)
        finally:
            monkeypatch.setattr(shutil, "rmtree", real_rmtree)

        assert interim.exists()
        assert (interim / "adapter_model.safetensors").exists()
        assert not slot.exists(), "the condemned slot is out of the live namespace"

    def test_stale_pending_delete_collision_is_cleared_before_rename(self, tmp_path):
        """A same-name leftover already sitting in ``.pending-delete/`` from
        a prior crash is treated as already-condemned debris and removed
        first — without this, ``os.rename`` onto a non-empty destination
        directory would raise and the reap would fail outright."""
        adapter_dir = tmp_path
        tier_root = adapter_dir / "semantic"
        child = tier_root / "20260417-120000"
        child.mkdir(parents=True)
        (child / "adapter_model.safetensors").write_bytes(b"")

        stale = adapter_dir / _PENDING_DELETE_DIR_NAME / "20260417-120000"
        stale.mkdir(parents=True)
        (stale / "leftover_from_prior_crash.bin").write_bytes(b"debris")

        removed = reap_tier_artifacts(tier_root)

        assert child in removed
        assert not tier_root.exists()
        assert not (adapter_dir / _PENDING_DELETE_DIR_NAME).exists()


class TestResumePendingReaps:
    """Boot-time sweep that finishes any deletion reap_tier_artifacts left
    stranded under ``.pending-delete/``."""

    def test_missing_dir_is_silent_noop(self, tmp_path):
        """No ``.pending-delete`` directory at all is a silent no-op."""
        resume_pending_reaps(tmp_path)
        assert not (tmp_path / _PENDING_DELETE_DIR_NAME).exists()

    def test_clears_stranded_tombstone(self, tmp_path, caplog):
        """A directory stranded under ``.pending-delete/`` (simulating a
        crash between reap_tier_artifacts' rename and its delete) is
        removed, along with the now-empty tombstone dir itself, and the
        removal is logged as a WARNING naming the resumed path."""
        import logging

        stranded = tmp_path / _PENDING_DELETE_DIR_NAME / "interim_20260417T0000"
        nested = stranded / "20260417-120000"
        nested.mkdir(parents=True)
        (nested / "adapter_model.safetensors").write_bytes(b"")

        caplog.set_level(logging.WARNING, logger="paramem.memory.persistence")
        resume_pending_reaps(tmp_path)

        assert not (tmp_path / _PENDING_DELETE_DIR_NAME).exists()
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any("interim_20260417T0000" in msg for msg in warnings), (
            f"expected a WARNING naming the resumed path, got: {warnings}"
        )

    def test_clears_stranded_file(self, tmp_path):
        """A stranded plain-file entry under ``.pending-delete/`` (the file
        child branch of reap_tier_artifacts) is unlinked, not just handled
        for directories."""
        pending = tmp_path / _PENDING_DELETE_DIR_NAME
        pending.mkdir(parents=True)
        (pending / "indexed_key_registry.json").write_text("{}")

        resume_pending_reaps(tmp_path)

        assert not pending.exists()

    def test_idempotent_double_resume(self, tmp_path):
        """Calling resume_pending_reaps a second time after it already
        cleared everything is a safe no-op."""
        stranded = tmp_path / _PENDING_DELETE_DIR_NAME / "interim_20260417T0000"
        stranded.mkdir(parents=True)

        resume_pending_reaps(tmp_path)
        assert not (tmp_path / _PENDING_DELETE_DIR_NAME).exists()

        resume_pending_reaps(tmp_path)  # must not raise
        assert not (tmp_path / _PENDING_DELETE_DIR_NAME).exists()

    def test_end_to_end_crash_then_resume(self, tmp_path, monkeypatch):
        """A reap_tier_artifacts crash-injection followed by
        resume_pending_reaps leaves nothing behind — the full
        crash-then-boot-resume cycle."""
        tier_root = tmp_path / "episodic"
        slot = tier_root / "20260417-120000"
        slot.mkdir(parents=True)
        (slot / "adapter_model.safetensors").write_bytes(b"")

        real_rmtree = shutil.rmtree

        def _boom(path, *a, **kw):
            raise OSError("simulated crash mid-delete")

        monkeypatch.setattr(shutil, "rmtree", _boom)
        try:
            with pytest.raises(OSError):
                reap_tier_artifacts(tier_root)
        finally:
            monkeypatch.setattr(shutil, "rmtree", real_rmtree)

        assert (tmp_path / _PENDING_DELETE_DIR_NAME).exists()

        resume_pending_reaps(tmp_path)

        assert not (tmp_path / _PENDING_DELETE_DIR_NAME).exists()

    def test_entry_that_raises_does_not_block_the_remaining_entries(
        self, tmp_path, monkeypatch, caplog
    ):
        """Best-effort: one entry whose removal raises (e.g. a permission
        error) is logged at ERROR and skipped — it must not prevent the
        OTHER entries under ``.pending-delete/`` from being cleared, and
        must not propagate out of ``resume_pending_reaps`` (which would
        abort the entire boot via ``_load_model_into_state``)."""
        import logging

        pending = tmp_path / _PENDING_DELETE_DIR_NAME
        poisoned = pending / "interim_poisoned"
        poisoned.mkdir(parents=True)
        (poisoned / "adapter_model.safetensors").write_bytes(b"")
        clean = pending / "interim_clean"
        clean.mkdir(parents=True)
        (clean / "adapter_model.safetensors").write_bytes(b"")

        real_rmtree = shutil.rmtree

        def _boom(path, *a, **kw):
            if Path(path).name == "interim_poisoned":
                raise OSError("simulated permission error")
            return real_rmtree(path, *a, **kw)

        monkeypatch.setattr(shutil, "rmtree", _boom)
        caplog.set_level(logging.ERROR, logger="paramem.memory.persistence")
        try:
            # Must not raise — the poisoned entry's OSError is caught internally.
            resume_pending_reaps(tmp_path)
        finally:
            monkeypatch.setattr(shutil, "rmtree", real_rmtree)

        # The clean entry was removed despite the poisoned one failing.
        assert not clean.exists()
        # The poisoned entry is left stranded (retried on next boot), and the
        # tombstone dir itself survives since it is non-empty.
        assert poisoned.exists()
        assert pending.exists()
        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert any("interim_poisoned" in r.getMessage() for r in error_records), (
            f"expected an ERROR log naming the failed entry, got: "
            f"{[r.getMessage() for r in caplog.records]}"
        )

    def test_regular_file_pending_delete_is_removed(self, tmp_path):
        """``.pending-delete`` itself existing as a regular file (rather
        than a directory) is tolerated — unlinked directly instead of
        raising ``NotADirectoryError`` out of ``iterdir()``."""
        pending = tmp_path / _PENDING_DELETE_DIR_NAME
        pending.write_bytes(b"unexpected file, not a directory")

        resume_pending_reaps(tmp_path)

        assert not pending.exists()


# ---------------------------------------------------------------------------
# erase_keys_and_restamp_manifest — key-erase / registry-save / graph-erase /
# manifest-re-stamp sequence shared by every out-of-fold registry-mutation
# caller (extracted from POST /speaker/forget).
# ---------------------------------------------------------------------------


class TestEraseKeysAndRestampManifest:
    def test_empty_keys_is_a_noop(self, tmp_path):
        """No keys -> {} and no store mutation."""
        from paramem.memory.store import MemoryStore

        store = MemoryStore(replay_enabled=True)
        store.registry("episodic").add("graph1")

        result = erase_keys_and_restamp_manifest(store=store, adapter_dir=tmp_path, keys=[])

        assert result == {}
        assert store.registry("episodic").knows("graph1")

    def test_erases_keys_saves_registry_and_erases_graph(self, tmp_path):
        """The erased key is gone from the store, the saved registry file,
        and the tier's on-disk graph.json; a surviving key is untouched in
        all three, and the survivor is not in the result."""
        from paramem.memory.store import MemoryStore
        from paramem.training.key_registry import KeyRegistry

        store = MemoryStore(replay_enabled=True)
        reg = store.registry("episodic")
        reg.add("graph1")
        reg.add("graph2")

        adapter_dir = tmp_path / "adapters"
        tier_root = adapter_dir / "episodic"
        tier_root.mkdir(parents=True)

        g = nx.MultiDiGraph()
        _add_keyed_edge(
            g, "Alice", "Berlin", indexed_key="graph1", predicate="lives_in", speaker_id="S0"
        )
        _add_keyed_edge(
            g, "Bob", "Engineer", indexed_key="graph2", predicate="has_job", speaker_id="S1"
        )
        save_memory_to_disk(g, tier_root / "graph.json")

        result = erase_keys_and_restamp_manifest(
            store=store, adapter_dir=adapter_dir, keys=["graph1"]
        )

        assert result == {}

        assert not reg.knows("graph1")
        assert reg.knows("graph2")

        on_disk = KeyRegistry.load(tier_root / "indexed_key_registry.json")
        assert not on_disk.knows("graph1")
        assert on_disk.knows("graph2")

        g2 = load_memory_from_disk(tier_root / "graph.json")
        keys_on_disk = {e["key"] for e in iter_entries(g2)}
        assert keys_on_disk == {"graph2"}

    def test_emptied_tier_is_returned_not_restamped(self, tmp_path):
        """A tier reduced to zero known keys is returned in the result (for
        the caller to reap), not re-stamped."""
        from paramem.memory.store import MemoryStore

        store = MemoryStore(replay_enabled=True)
        store.registry("episodic").add("graph1")

        adapter_dir = tmp_path / "adapters"
        tier_root = adapter_dir / "episodic"
        tier_root.mkdir(parents=True)

        result = erase_keys_and_restamp_manifest(
            store=store, adapter_dir=adapter_dir, keys=["graph1"]
        )

        assert result == {"episodic": tier_root}
        assert not store.registry("episodic").knows("graph1")

    def test_survivor_restamp_makes_slot_mountable(self, tmp_path):
        """After the helper runs on a surviving slot, find_live_slot accepts
        the re-stamped meta.json against the rewritten registry's hash and
        rejects the pre-erase hash — the fix for the "slot unmountable after
        an out-of-fold registry mutation" failure mode."""
        from paramem.adapters.manifest import (
            MANIFEST_SCHEMA_VERSION,
            AdapterManifest,
            BaseModelFingerprint,
            LoRAShape,
            TokenizerFingerprint,
            find_live_slot,
            write_manifest,
        )
        from paramem.memory.store import MemoryStore

        tier_name = "episodic"
        store = MemoryStore(replay_enabled=True)
        reg = store.registry(tier_name)
        reg.add("graph1")
        reg.add("graph2")

        adapter_dir = tmp_path / "adapters"
        tier_root = adapter_dir / tier_name
        tier_root.mkdir(parents=True)

        h_old = hashlib.sha256(reg.save_bytes()).hexdigest()
        reg.save(tier_root / "indexed_key_registry.json")

        slot_dir = tier_root / "20260612-000000"
        slot_dir.mkdir()
        manifest = AdapterManifest(
            schema_version=MANIFEST_SCHEMA_VERSION,
            name=tier_name,
            trained_at="2026-06-12T00:00:00Z",
            base_model=BaseModelFingerprint(repo="hf/model", sha="abc123", hash="sha256:deadbeef"),
            tokenizer=TokenizerFingerprint(
                name_or_path="hf/model", vocab_size=32000, merges_hash="cafebabe"
            ),
            lora=LoRAShape(rank=8, alpha=16, dropout=0.0, target_modules=("q_proj", "v_proj")),
            registry_sha256=h_old,
            key_count=2,
        )
        write_manifest(slot_dir, manifest)
        assert find_live_slot(tier_root, h_old) == slot_dir

        result = erase_keys_and_restamp_manifest(
            store=store, adapter_dir=adapter_dir, keys=["graph1"]
        )

        assert result == {}
        h_new = hashlib.sha256(reg.save_bytes()).hexdigest()
        assert find_live_slot(tier_root, h_new) == slot_dir
        assert find_live_slot(tier_root, h_old) is None

    def test_malformed_interim_tier_name_raises_before_mutation(self, tmp_path):
        """A malformed interim tier name raises ValueError BEFORE
        store.discard_keys runs — the store is left untouched."""
        from paramem.memory.interim_adapter import INTERIM_NAME_PREFIX
        from paramem.memory.store import MemoryStore

        tier_name = f"{INTERIM_NAME_PREFIX}foo"  # does not parse as a stamp
        store = MemoryStore(replay_enabled=True)
        reg = store.registry(tier_name)
        reg.add("graph1")

        with pytest.raises(ValueError):
            erase_keys_and_restamp_manifest(
                store=store, adapter_dir=tmp_path / "adapters", keys=["graph1"]
            )

        assert reg.knows("graph1")

    def test_no_weight_slot_skips_restamp_without_error(self, tmp_path, caplog):
        """Simulate venue: a non-empty on-disk registry but no weight-slot
        manifest anywhere under the tier root — no ERROR is logged and the
        tier is reported as a survivor."""
        import logging

        from paramem.memory.store import MemoryStore

        tier_name = "episodic"
        store = MemoryStore(replay_enabled=True)
        reg = store.registry(tier_name)
        reg.add("graph1")
        reg.add("graph2")

        adapter_dir = tmp_path / "adapters"
        tier_root = adapter_dir / tier_name
        tier_root.mkdir(parents=True)
        reg.save(tier_root / "indexed_key_registry.json")

        caplog.set_level(logging.DEBUG, logger="paramem.memory.persistence")
        result = erase_keys_and_restamp_manifest(
            store=store, adapter_dir=adapter_dir, keys=["graph1"]
        )

        assert result == {}
        error_records = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert error_records == []
        debug_messages = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]
        assert any("no on-disk weight slot" in msg for msg in debug_messages)

    def test_empty_pre_erase_hash_skips_restamp(self, tmp_path, caplog):
        """A tier with in-memory keys but no on-disk registry (pre-erase
        hash == "") is not re-stamped and never consults find_live_slot; a
        WARNING is logged instead of an ERROR."""
        import logging
        from unittest.mock import patch

        from paramem.adapters.manifest import (
            MANIFEST_SCHEMA_VERSION,
            AdapterManifest,
            BaseModelFingerprint,
            LoRAShape,
            TokenizerFingerprint,
            write_manifest,
        )
        from paramem.memory.store import MemoryStore

        tier_name = "episodic"
        store = MemoryStore(replay_enabled=True)
        reg = store.registry(tier_name)
        reg.add("graph1")
        reg.add("graph2")
        # Deliberately NOT saved to disk — tier_registry_sha256 reads "".

        adapter_dir = tmp_path / "adapters"
        tier_root = adapter_dir / tier_name

        # A weight-slot manifest must exist for the pre_sha == "" branch to be
        # reached at all — with zero slot candidates the venue gate fires
        # first (see test_no_weight_slot_skips_restamp_without_error). Its
        # registry_sha256 is irrelevant: find_live_slot is never consulted
        # for a "" pre-erase hash.
        slot_dir = tier_root / "20260101-000000"
        slot_dir.mkdir(parents=True)
        write_manifest(
            slot_dir,
            AdapterManifest(
                schema_version=MANIFEST_SCHEMA_VERSION,
                name=tier_name,
                trained_at="2026-01-01T00:00:00Z",
                base_model=BaseModelFingerprint(
                    repo="hf/model", sha="abc123", hash="sha256:deadbeef"
                ),
                tokenizer=TokenizerFingerprint(
                    name_or_path="hf/model", vocab_size=32000, merges_hash="cafebabe"
                ),
                lora=LoRAShape(rank=8, alpha=16, dropout=0.0, target_modules=("q_proj",)),
                registry_sha256="unrelated" * 8,
                key_count=0,
            ),
        )

        caplog.set_level(logging.WARNING, logger="paramem.memory.persistence")
        with patch("paramem.adapters.manifest.find_live_slot") as mock_find_live_slot:
            result = erase_keys_and_restamp_manifest(
                store=store, adapter_dir=adapter_dir, keys=["graph1"]
            )

        assert result == {}
        mock_find_live_slot.assert_not_called()
        warning_messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any("had no readable pre-erase registry" in msg for msg in warning_messages)

    def test_orphaned_slot_logs_error_without_raising(self, tmp_path, caplog):
        """A pre-erase hash that no on-disk slot matches (already orphaned,
        e.g. from a prior crash) logs an ERROR but does not raise — the
        forget must still succeed."""
        import logging

        from paramem.adapters.manifest import (
            MANIFEST_SCHEMA_VERSION,
            AdapterManifest,
            BaseModelFingerprint,
            LoRAShape,
            TokenizerFingerprint,
            write_manifest,
        )
        from paramem.memory.store import MemoryStore

        tier_name = "episodic"
        store = MemoryStore(replay_enabled=True)
        reg = store.registry(tier_name)
        reg.add("graph1")
        reg.add("graph2")

        adapter_dir = tmp_path / "adapters"
        tier_root = adapter_dir / tier_name
        tier_root.mkdir(parents=True)
        reg.save(tier_root / "indexed_key_registry.json")

        # A slot exists, but its registry_sha256 does not match the pre-erase
        # hash — find_live_slot returns None for it.
        slot_dir = tier_root / "20260612-000000"
        slot_dir.mkdir()
        write_manifest(
            slot_dir,
            AdapterManifest(
                schema_version=MANIFEST_SCHEMA_VERSION,
                name=tier_name,
                trained_at="2026-06-12T00:00:00Z",
                base_model=BaseModelFingerprint(
                    repo="hf/model", sha="abc123", hash="sha256:deadbeef"
                ),
                tokenizer=TokenizerFingerprint(
                    name_or_path="hf/model", vocab_size=32000, merges_hash="cafebabe"
                ),
                lora=LoRAShape(rank=8, alpha=16, dropout=0.0, target_modules=("q_proj",)),
                registry_sha256="deadbeef" * 8,
                key_count=1,
            ),
        )

        caplog.set_level(logging.ERROR, logger="paramem.memory.persistence")
        result = erase_keys_and_restamp_manifest(
            store=store, adapter_dir=adapter_dir, keys=["graph1"]
        )

        assert result == {}
        error_messages = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
        assert any("slot already orphaned" in msg for msg in error_messages)
