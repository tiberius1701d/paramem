"""Tests for graph-level SOTA enrichment (Task #10).

All tests are pure-Python — no GPU required. SOTA calls are mocked so
the test suite does not make any network requests.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import networkx as nx
from peft import PeftModel

from paramem.memory.persistence import _EDGE_SOURCE_ATTR
from paramem.training.consolidation import ConsolidationLoop, serialize_subgraph_triples
from paramem.utils.config import AdapterConfig, ConsolidationConfig, TrainingConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_loop(tmp_path, **kwargs) -> ConsolidationLoop:
    """Build a minimal ConsolidationLoop for enrichment tests.

    Graph is transient (RAM-only). Model/tokenizer are mocks so no GPU
    is touched.  The mock model pre-populates ``peft_config`` with all
    three required adapters so ``_ensure_adapters`` skips the real PEFT
    ``create_adapter`` calls.

    Keyword args forwarded to ConsolidationLoop override the defaults
    set here (e.g. pass ``extraction_noise_filter=""`` to test the
    no-provider skip path).
    """
    # __class__ = PeftModel so _ensure_adapters' isinstance check
    # short-circuits without restricting the mock's attribute surface.
    model = MagicMock()
    model.__class__ = PeftModel
    model.peft_config = {
        "episodic": MagicMock(),
        "semantic": MagicMock(),
        "in_training": MagicMock(),
    }

    defaults = dict(
        extraction_noise_filter="anthropic",
        extraction_noise_filter_model="claude-sonnet-4-6",
    )
    defaults.update(kwargs)

    # replay_enabled controls whether run_consolidation_cycle's registry guard
    # fires.  Callers that need enrichment hooks to fire pass replay_enabled=True.
    replay_enabled = defaults.pop("replay_enabled", False)
    # Allow callers to supply a pre-built ConsolidationConfig so tests can set
    # fields like interim_refinement without touching the loop's other knobs.
    consolidation_config = defaults.pop("consolidation_config", ConsolidationConfig())

    from paramem.memory.store import MemoryStore as _MS

    loop = ConsolidationLoop(
        model=model,
        tokenizer=MagicMock(),
        consolidation_config=consolidation_config,
        training_config=TrainingConfig(),
        episodic_adapter_config=AdapterConfig(),
        semantic_adapter_config=AdapterConfig(),
        memory_store=_MS(replay_enabled=replay_enabled),
        procedural_adapter_config=None,
        output_dir=tmp_path,
        **defaults,
    )
    # Admit-all probe stub: the real _probe_passing_keys runs evaluate_indexed_recall,
    # which feeds the MagicMock model into re.sub and TypeErrors.  Admitting every key
    # is the prior implicit behavior (no recall gate), so it is inert for these tests.
    loop._probe_passing_keys = lambda adapter_name, entries: {e["key"] for e in entries}
    return loop


def _populate_graph(graph: nx.MultiDiGraph, n_persons: int = 10) -> None:
    """Add n_persons person nodes + 1 hub org node (total n_persons+1 nodes).

    Default of 10 persons + 1 org = 11 nodes exceeds the 10-node floor so
    tests exercise the enrichment path by default.

    Nodes are keyed in canonical form (lowercase, separator-folded) matching
    the live merger's node-key convention post-model-A.  Surface display names
    are stored in attributes["name"] where needed by individual tests.
    """
    for i in range(n_persons):
        name = f"person{i}"
        graph.add_node(
            name,
            entity_type="person",
            attributes={"name": f"Person{i}"},
            recurrence_count=i + 1,
            sessions=[f"s{i:03d}"],
            first_seen=f"s{i:03d}",
            last_seen=f"s{i:03d}",
        )
    # Add an org node so we have cross-entity topology
    org = "acmecorp"
    graph.add_node(
        org,
        entity_type="organization",
        attributes={"name": "AcmeCorp"},
        recurrence_count=n_persons,
        sessions=["s000"],
        first_seen="s000",
        last_seen="s000",
    )
    # Wire edges: every person works_at acmecorp
    for i in range(n_persons):
        graph.add_edge(
            f"person{i}",
            org,
            predicate="works at",
            relation_type="factual",
            confidence=1.0,
            source="extraction",
            sessions=["s000"],
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSerializeSubgraphTriples:
    """Unit tests for the module-level helper."""

    def test_basic_serialization(self):
        g = nx.MultiDiGraph()
        g.add_node("Alice")
        g.add_node("Bob")
        g.add_edge("Alice", "Bob", predicate="knows", relation_type="social", confidence=0.9)
        triples = serialize_subgraph_triples(g)
        assert len(triples) == 1
        t = triples[0]
        assert t["subject"] == "Alice"
        assert t["predicate"] == "knows"
        assert t["object"] == "Bob"
        assert t["relation_type"] == "social"

    def test_missing_predicate_defaults(self):
        g = nx.MultiDiGraph()
        g.add_edge("A", "B", relation_type="factual")
        triples = serialize_subgraph_triples(g)
        assert triples[0]["predicate"] == ""

    def test_missing_relation_type_defaults(self):
        g = nx.MultiDiGraph()
        g.add_edge("A", "B", predicate="likes")
        triples = serialize_subgraph_triples(g)
        assert triples[0]["relation_type"] == "factual"

    def test_empty_graph(self):
        g = nx.MultiDiGraph()
        assert serialize_subgraph_triples(g) == []


class TestEnrichmentAddsEdgesWithSourceTag:
    """New edges must carry source='graph_enrichment'."""

    def test_new_edge_tagged(self, tmp_path, monkeypatch):
        loop = _make_loop(tmp_path)
        graph = loop.merger.graph
        _populate_graph(graph, n_persons=10)

        canned_result = (
            [
                {
                    "subject": "Person0",
                    "predicate": "colleague_of",
                    "object": "Person1",
                    "relation_type": "social",
                    "confidence": 0.9,
                }
            ],
            [],  # no same_as
            "raw",
        )
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with patch(
            "paramem.training.consolidation._graph_enrich_with_sota",
            return_value=canned_result,
        ):
            result = loop._run_graph_enrichment()

        assert not result["skipped"]
        assert result["new_edges"] >= 1

        # Verify the added edge carries source="graph_enrichment".
        # Nodes are canonical-keyed; predicate is stored in canonical form too
        # ("colleague_of" → "colleague of" after canonical() separator-fold).
        found = False
        for _, _, data in graph.out_edges("person0", data=True):
            if (
                data.get("predicate") == "colleague of"
                and data.get(_EDGE_SOURCE_ATTR) == "graph_enrichment"
            ):
                found = True
        assert found, "Expected a 'colleague of' edge with source='graph_enrichment'"


class TestLowConfidenceDropped:
    """Relations with confidence < 0.7 must be discarded."""

    def test_low_confidence_skipped(self, tmp_path, monkeypatch):
        loop = _make_loop(tmp_path)
        graph = loop.merger.graph
        _populate_graph(graph, n_persons=10)

        canned_result = (
            [
                {
                    "subject": "Person0",
                    "predicate": "colleague_of",
                    "object": "Person1",
                    "relation_type": "social",
                    "confidence": 0.5,
                },
                {
                    "subject": "Person0",
                    "predicate": "friend_of",
                    "object": "Person2",
                    "relation_type": "social",
                    "confidence": 0.9,
                },
            ],
            [],
            "raw",
        )
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with patch(
            "paramem.training.consolidation._graph_enrich_with_sota",
            return_value=canned_result,
        ):
            result = loop._run_graph_enrichment()

        assert not result["skipped"]
        assert result["new_edges"] == 1, "Only the 0.9-confidence edge should land"

        # Nodes are canonical-keyed; predicates stored in canonical form
        # ("friend_of" → "friend of", "colleague_of" → "colleague of").
        edges_from_p0 = list(graph.out_edges("person0", data=True))
        predicates = {
            d.get("predicate")
            for _, _, d in edges_from_p0
            if d.get(_EDGE_SOURCE_ATTR) == "graph_enrichment"
        }
        assert "friend of" in predicates
        assert "colleague of" not in predicates


class TestSameAsContractsNodes:
    """same_as pairs must remove the variant node and rewire its edges."""

    def test_variant_node_removed(self, tmp_path, monkeypatch):
        loop = _make_loop(tmp_path)
        graph = loop.merger.graph
        _populate_graph(graph, n_persons=10)

        # Add two nodes that should be merged — canonical-keyed (lowercase).
        graph.add_node(
            "alice",
            entity_type="person",
            attributes={"name": "Alice"},
            recurrence_count=3,
            sessions=["s010"],
            first_seen="s010",
            last_seen="s010",
        )
        graph.add_node(
            "alicia",
            entity_type="person",
            attributes={"name": "Alicia"},
            recurrence_count=1,
            sessions=["s011"],
            first_seen="s011",
            last_seen="s011",
        )
        graph.add_edge(
            "alicia",
            "acmecorp",
            predicate="works at",
            relation_type="factual",
            confidence=1.0,
            source="extraction",
            sessions=["s011"],
        )

        # SOTA returns surface names; production canonicalizes them before lookup:
        # "Alice" -> "alice", "Alicia" -> "alicia".
        canned_result = (
            [],  # no new relations
            [["Alice", "Alicia"]],  # same_as
            "raw",
        )
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with patch(
            "paramem.training.consolidation._graph_enrich_with_sota",
            return_value=canned_result,
        ):
            result = loop._run_graph_enrichment()

        assert not result["skipped"]
        assert result["same_as_merges"] >= 1
        # "alicia" should be contracted into "alice" — removed as a distinct node
        assert "alicia" not in graph.nodes
        assert "alice" in graph.nodes


class TestSafeToMergeSurface:
    """Unit coverage for the surname / surface-form safety gate."""

    def test_token_subset_accepted(self):
        from paramem.training.consolidation import _safe_to_merge_surface

        # Honorific-stripped subset
        assert _safe_to_merge_surface("Mr. Yang", "Yang Ming") is True
        # Given-name subset of full name
        assert _safe_to_merge_surface("Ming", "Yang Ming") is True
        # Identical after honorific strip
        assert _safe_to_merge_surface("Dr. Smith", "Smith") is True

    def test_different_surnames_rejected(self):
        from paramem.training.consolidation import _safe_to_merge_surface

        # Shared given name, different family name — must NOT merge
        assert _safe_to_merge_surface("Zhang Min", "Wang Min") is False
        assert _safe_to_merge_surface("Li Wei", "Chen Wei") is False

    def test_jw_fallback_accepts_minor_typos(self):
        from paramem.training.consolidation import _safe_to_merge_surface

        # True variant (one letter off) passes the JW fallback
        assert _safe_to_merge_surface("Catherine Holmes", "Katherine Holmes") is True

    def test_empty_and_all_honorific_rejected(self):
        from paramem.training.consolidation import _safe_to_merge_surface

        assert _safe_to_merge_surface("", "Alice") is False
        assert _safe_to_merge_surface("Mr.", "Dr.") is False


class TestSameAsSurnameMismatchRejected:
    """Integration: a bad same_as pair from SOTA must be rejected by the gate."""

    def test_cross_surname_pair_rejected(self, tmp_path, monkeypatch):
        loop = _make_loop(tmp_path)
        graph = loop.merger.graph
        _populate_graph(graph, n_persons=10)

        for name in ("Zhang Min", "Wang Min"):
            graph.add_node(
                name,
                entity_type="person",
                attributes={},
                recurrence_count=2,
                sessions=["s020"],
                first_seen="s020",
                last_seen="s020",
            )

        canned_result = ([], [["Zhang Min", "Wang Min"]], "raw")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with patch(
            "paramem.training.consolidation._graph_enrich_with_sota",
            return_value=canned_result,
        ):
            result = loop._run_graph_enrichment()

        assert not result["skipped"]
        # Gate must reject — neither node should be contracted
        assert result["same_as_merges"] == 0
        assert "Zhang Min" in graph.nodes
        assert "Wang Min" in graph.nodes


class TestSameAsDedupAcrossChunks:
    """Duplicate same_as pairs (same pair, any order) must apply exactly once."""

    def test_duplicate_pair_applied_once(self, tmp_path, monkeypatch):
        loop = _make_loop(tmp_path)
        graph = loop.merger.graph
        _populate_graph(graph, n_persons=10)

        # Nodes are canonical-keyed; "Yang Ming" → "yang ming", "Mr. Yang" → "mr. yang".
        graph.add_node(
            "yang ming",
            entity_type="person",
            attributes={"name": "Yang Ming"},
            recurrence_count=3,
            sessions=["s030"],
            first_seen="s030",
            last_seen="s030",
        )
        graph.add_node(
            "mr. yang",
            entity_type="person",
            attributes={"name": "Mr. Yang"},
            recurrence_count=2,
            sessions=["s031"],
            first_seen="s031",
            last_seen="s031",
        )

        # SOTA returns surface names; production canonicalizes before graph lookup.
        # Same pair emitted twice in reversed order — simulates SOTA echoing
        # the duplicate across chunks.
        canned_result = (
            [],
            [["Yang Ming", "Mr. Yang"], ["Mr. Yang", "Yang Ming"]],
            "raw",
        )
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with patch(
            "paramem.training.consolidation._graph_enrich_with_sota",
            return_value=canned_result,
        ):
            result = loop._run_graph_enrichment()

        assert not result["skipped"]
        assert result["same_as_merges"] == 1


class TestSymmetricPredicateCanonicalized:
    """Symmetric predicates collapse to one direction; reverse is a dup."""

    def test_both_directions_collapse_to_one_edge(self, tmp_path, monkeypatch):
        loop = _make_loop(tmp_path)
        graph = loop.merger.graph
        _populate_graph(graph, n_persons=10)

        # BL-2 coverage: _SYMMETRIC_ENRICHMENT_PREDICATES contains canonical(p) (space form).
        # Feed one entry in underscore form and one in space form — both must be recognized
        # as symmetric and collapse to a single canonical-direction edge.
        # BL-2: both underscore ("colleague_of") and space ("colleague of") forms must
        # match _SYMMETRIC_ENRICHMENT_PREDICATES (which stores canonical() entries).
        rels = [
            {
                "subject": "Zhang",
                "predicate": "colleague_of",  # underscore — canonical() yields "colleague of"
                "object": "Xiaoxiu",
                "relation_type": "social",
                "confidence": 0.85,
            },
            # Reverse direction in space form — already canonical; still symmetric.
            {
                "subject": "Xiaoxiu",
                "predicate": "colleague of",  # space form
                "object": "Zhang",
                "relation_type": "social",
                "confidence": 0.80,
            },
        ]
        # Nodes are canonical-keyed ("Zhang" → "zhang", "Xiaoxiu" → "xiaoxiu").
        for name in ("zhang", "xiaoxiu"):
            graph.add_node(
                name,
                entity_type="person",
                attributes={},
                recurrence_count=2,
                sessions=["s040"],
                first_seen="s040",
                last_seen="s040",
            )

        canned_result = (rels, [], "raw")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with patch(
            "paramem.training.consolidation._graph_enrich_with_sota",
            return_value=canned_result,
        ):
            result = loop._run_graph_enrichment()

        assert not result["skipped"]
        # After lex canonicalization subj<obj, both rels become
        # ("xiaoxiu", "colleague of", "zhang"). Second insert is a duplicate.
        # Predicate stored as canonical("colleague_of") == "colleague of".
        enriched = [
            (u, v, d)
            for u, v, d in graph.edges(data=True)
            if d.get(_EDGE_SOURCE_ATTR) == "graph_enrichment"
        ]
        colleague_edges = [(u, v) for u, v, d in enriched if d.get("predicate") == "colleague of"]
        assert len(colleague_edges) == 1
        u, v = colleague_edges[0]
        assert u < v  # canonical lex order

    def test_asymmetric_predicates_not_reordered(self, tmp_path, monkeypatch):
        loop = _make_loop(tmp_path)
        graph = loop.merger.graph
        _populate_graph(graph, n_persons=10)

        # mentored_by / mentee_of are inverses, not symmetric — keep both.
        rels = [
            {
                "subject": "Ming",
                "predicate": "mentored_by",
                "object": "Xinxin",
                "relation_type": "social",
                "confidence": 0.85,
            },
            {
                "subject": "Xinxin",
                "predicate": "mentored_by",
                "object": "Ming",
                "relation_type": "social",
                "confidence": 0.85,
            },
        ]
        # Nodes are canonical-keyed ("Ming" → "ming", "Xinxin" → "xinxin").
        for name in ("ming", "xinxin"):
            graph.add_node(
                name,
                entity_type="person",
                attributes={},
                recurrence_count=2,
                sessions=["s041"],
                first_seen="s041",
                last_seen="s041",
            )

        canned_result = (rels, [], "raw")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with patch(
            "paramem.training.consolidation._graph_enrich_with_sota",
            return_value=canned_result,
        ):
            result = loop._run_graph_enrichment()

        assert not result["skipped"]
        # Predicate stored as canonical("mentored_by") == "mentored by".
        # Nodes stored as canonical keys ("ming", "xinxin").
        mentored_edges = [
            (u, v)
            for u, v, d in graph.edges(data=True)
            if d.get(_EDGE_SOURCE_ATTR) == "graph_enrichment"
            and d.get("predicate") == "mentored by"
        ]
        # Both directions survive for asymmetric predicates
        assert set(mentored_edges) == {("ming", "xinxin"), ("xinxin", "ming")}


class TestCorefRemapBeforeEdgeInsert:
    """Relations referencing a dropped node must land on the canonical node."""

    def test_relation_remapped_through_coref(self, tmp_path, monkeypatch):
        loop = _make_loop(tmp_path)
        graph = loop.merger.graph
        _populate_graph(graph, n_persons=10)

        # Three nodes — canonical-keyed (lowercase).
        # "Alex" will be contracted into "alexander".
        graph.add_node(
            "alexander",
            entity_type="person",
            attributes={"name": "Alexander"},
            recurrence_count=3,
            sessions=["s050"],
            first_seen="s050",
            last_seen="s050",
        )
        graph.add_node(
            "alex",
            entity_type="person",
            attributes={"name": "Alex"},
            recurrence_count=1,
            sessions=["s051"],
            first_seen="s051",
            last_seen="s051",
        )
        graph.add_node(
            "acme",
            entity_type="organization",
            attributes={"name": "Acme"},
            recurrence_count=5,
            sessions=["s050"],
            first_seen="s050",
            last_seen="s050",
        )

        # SOTA response: same_as merges Alex→Alexander (SOTA returns surface names;
        # production canonicalizes to "alex"/"alexander" before graph lookup).
        # The relation also uses dropped name "Alex" — the remap routes it to "alexander".
        canned_rels = [
            {
                "subject": "Alex",
                "predicate": "works_at",
                "object": "Acme",
                "relation_type": "factual",
                "confidence": 0.9,
            }
        ]
        canned_same_as = [["Alexander", "Alex"]]

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with patch(
            "paramem.training.consolidation._graph_enrich_with_sota",
            return_value=(canned_rels, canned_same_as, "raw"),
        ):
            result = loop._run_graph_enrichment()

        assert not result["skipped"]
        assert result["same_as_merges"] >= 1
        assert "alex" not in graph.nodes  # contracted away
        # The enriched edge must land on "alexander" (canonical keep node).
        # Predicate stored as canonical("works_at") == "works at".
        alexander_edges = [
            (u, v, d)
            for u, v, d in graph.edges(data=True)
            if u == "alexander"
            and v == "acme"
            and d.get("predicate") == "works at"
            and d.get(_EDGE_SOURCE_ATTR) == "graph_enrichment"
        ]
        assert len(alexander_edges) == 1


class TestFloorSkipsSmallGraphs:
    """Graphs with fewer than 10 nodes must be skipped without a SOTA call."""

    def test_small_graph_skip(self, tmp_path, monkeypatch):
        loop = _make_loop(tmp_path)
        graph = loop.merger.graph

        # Add only 5 nodes — below the floor
        for i in range(5):
            graph.add_node(
                f"Tiny{i}",
                entity_type="concept",
                attributes={},
                recurrence_count=1,
                sessions=[],
                first_seen="s000",
                last_seen="s000",
            )

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        call_spy = MagicMock()
        with patch("paramem.training.consolidation._graph_enrich_with_sota", call_spy):
            result = loop._run_graph_enrichment()

        assert result["skipped"] is True
        assert result["skip_reason"] == "floor"
        call_spy.assert_not_called()


class TestDisabledIsNoop:
    """enabled=False must leave the graph byte-identical."""

    def test_no_change_when_disabled(self, tmp_path, monkeypatch):
        loop = _make_loop(tmp_path, graph_enrichment_enabled=False)
        graph = loop.merger.graph
        _populate_graph(graph, n_persons=10)

        # Snapshot pre-state
        pre_nodes = set(graph.nodes)
        pre_edges = set((u, v) for u, v, _ in graph.edges(data=True))

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        call_spy = MagicMock()
        with patch("paramem.training.consolidation._graph_enrich_with_sota", call_spy):
            result = loop._run_graph_enrichment()

        assert result["skipped"] is True
        assert result["skip_reason"] == "disabled"
        call_spy.assert_not_called()

        post_nodes = set(graph.nodes)
        post_edges = set((u, v) for u, v, _ in graph.edges(data=True))
        assert pre_nodes == post_nodes
        assert pre_edges == post_edges


class TestPartitionRoutesEnrichedEdges:
    """After enrichment, partition_relations must correctly route new edges."""

    def test_social_edge_routes_to_episodic(self, tmp_path, monkeypatch):
        """Social relation_type → episodic bucket (not procedural)."""
        from paramem.graph.qa_generator import partition_relations

        loop = _make_loop(tmp_path)
        graph = loop.merger.graph
        _populate_graph(graph, n_persons=10)

        canned_result = (
            [
                {
                    "subject": "Person0",
                    "predicate": "colleague_of",
                    "object": "Person1",
                    "relation_type": "social",
                    "confidence": 0.85,
                }
            ],
            [],
            "raw",
        )
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with patch(
            "paramem.training.consolidation._graph_enrich_with_sota",
            return_value=canned_result,
        ):
            loop._run_graph_enrichment()

        # Collect enriched edges
        enriched = [
            {
                "subject": u,
                "predicate": d["predicate"],
                "object": v,
                "relation_type": d["relation_type"],
            }
            for u, v, d in graph.edges(data=True)
            if d.get(_EDGE_SOURCE_ATTR) == "graph_enrichment"
        ]
        assert enriched, "No enriched edges found"

        ep_rels, proc_rels = partition_relations(enriched, procedural_enabled=False)
        assert len(ep_rels) == len(enriched)
        assert proc_rels == []


class TestChunkCapRespected:
    """Each SOTA call payload must not exceed max_entities_per_pass nodes."""

    def test_each_chunk_within_cap(self, tmp_path, monkeypatch):
        loop = _make_loop(
            tmp_path,
            graph_enrichment_max_entities_per_pass=10,
            graph_enrichment_neighborhood_hops=1,
        )
        graph = loop.merger.graph

        # Build a larger graph: 25 person nodes + 1 hub org
        org = "HubCorp"
        graph.add_node(
            org,
            entity_type="organization",
            attributes={},
            recurrence_count=25,
            sessions=[],
            first_seen="s000",
            last_seen="s000",
        )
        for i in range(25):
            name = f"Emp{i}"
            graph.add_node(
                name,
                entity_type="person",
                attributes={},
                recurrence_count=i + 1,
                sessions=[],
                first_seen=f"s{i:03d}",
                last_seen=f"s{i:03d}",
            )
            graph.add_edge(
                name,
                org,
                predicate="works_at",
                relation_type="factual",
                confidence=1.0,
                source="extraction",
                sessions=[],
            )

        call_args_list: list[list[dict]] = []

        def _spy_call(triples, *args, **kwargs):
            call_args_list.append(list(triples))
            return ([], [], "raw")

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with patch("paramem.training.consolidation._graph_enrich_with_sota", side_effect=_spy_call):
            loop._run_graph_enrichment()

        assert call_args_list, "Expected at least one SOTA call"

        # Gather the unique node names seen in each call's triples
        for triples in call_args_list:
            nodes_in_call = set()
            for t in triples:
                nodes_in_call.add(t["subject"])
                nodes_in_call.add(t["object"])
            assert len(nodes_in_call) <= 10 + 1, (
                f"Chunk exceeded cap: {len(nodes_in_call)} nodes (cap=10, +1 tolerance for hub)"
            )


class TestNoApiKeySkipsGracefully:
    """When provider env var is absent, skip with no crash."""

    def test_missing_key_skip(self, tmp_path, monkeypatch):
        loop = _make_loop(tmp_path)
        graph = loop.merger.graph
        _populate_graph(graph, n_persons=10)

        # Remove the key from the environment
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        call_spy = MagicMock()
        with patch("paramem.training.consolidation._graph_enrich_with_sota", call_spy):
            result = loop._run_graph_enrichment()

        assert result["skipped"] is True
        assert result["skip_reason"] == "no_api_key"
        call_spy.assert_not_called()


class TestNoProviderSkipsGracefully:
    """When extraction_noise_filter is empty, skip with no crash."""

    def test_no_provider_skip(self, tmp_path, monkeypatch):
        loop = _make_loop(tmp_path, extraction_noise_filter="")
        graph = loop.merger.graph
        _populate_graph(graph, n_persons=10)

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        call_spy = MagicMock()
        with patch("paramem.training.consolidation._graph_enrich_with_sota", call_spy):
            result = loop._run_graph_enrichment()

        assert result["skipped"] is True
        assert result["skip_reason"] == "no_provider"
        call_spy.assert_not_called()


class TestGraphEnrichWithSotaUnit:
    """Unit tests for the extractor-level _graph_enrich_with_sota function."""

    def test_returns_relations_and_same_as(self):
        from paramem.graph.extractor import _graph_enrich_with_sota

        canned_raw = (
            '{"relations": [{"subject": "A", "predicate": "knows", "object": "B", '
            '"relation_type": "social", "confidence": 0.8}], "same_as": [["Alice", "Alicia"]]}'
        )
        with patch("paramem.graph.extractor._sota_call", return_value=canned_raw):
            result = _graph_enrich_with_sota(
                [
                    {
                        "subject": "A",
                        "predicate": "works_at",
                        "object": "Corp",
                        "relation_type": "factual",
                    }
                ],
                api_key="test-key",
                provider="anthropic",
                filter_model="claude-sonnet-4-6",
            )

        assert result is not None
        new_rels, same_as, raw = result
        assert len(new_rels) == 1
        assert new_rels[0]["predicate"] == "knows"
        assert len(same_as) == 1
        assert same_as[0] == ["Alice", "Alicia"]

    def test_legacy_bare_array(self):
        """Bare JSON array response → treated as relations, empty same_as."""
        from paramem.graph.extractor import _graph_enrich_with_sota

        canned_raw = (
            '[{"subject": "A", "predicate": "knows", "object": "B", '
            '"relation_type": "social", "confidence": 0.8}]'
        )
        with patch("paramem.graph.extractor._sota_call", return_value=canned_raw):
            result = _graph_enrich_with_sota(
                [],
                api_key="test-key",
                provider="anthropic",
                filter_model="claude-sonnet-4-6",
            )

        assert result is not None
        new_rels, same_as, _ = result
        assert len(new_rels) == 1
        assert same_as == []

    def test_none_on_sota_failure(self):
        """_sota_call returning None → _graph_enrich_with_sota returns None."""
        from paramem.graph.extractor import _graph_enrich_with_sota

        with patch("paramem.graph.extractor._sota_call", return_value=None):
            result = _graph_enrich_with_sota(
                [],
                api_key="test-key",
                provider="anthropic",
                filter_model="claude-sonnet-4-6",
            )
        assert result is None

    def test_malformed_same_as_skipped(self):
        """Malformed same_as entries are silently skipped."""
        from paramem.graph.extractor import _graph_enrich_with_sota

        canned_raw = '{"relations": [], "same_as": ["bad", [1, 2], ["Alice", "Alicia"]]}'
        with patch("paramem.graph.extractor._sota_call", return_value=canned_raw):
            result = _graph_enrich_with_sota(
                [],
                api_key="test-key",
                provider="anthropic",
                filter_model="claude-sonnet-4-6",
            )

        assert result is not None
        _, same_as, _ = result
        # Only the valid [Alice, Alicia] entry survives; ["bad", [1,2]] are skipped
        assert same_as == [["Alice", "Alicia"]]


class TestInterimEnrichmentHook:
    """Rollover mini-enrichment inside post_session_train.

    The rollover hook fires inside the 'normal fresh-interim' branch of
    post_session_train, i.e. when a new sub-interval stamp opens.  It
    reuses `_run_graph_enrichment`, so these tests only verify the
    gating logic and the accumulator lifecycle.  The full enrichment
    path is covered by TestRunGraphEnrichment.
    """

    def _make_session_graph(self):
        """Build a 2-relation SessionGraph for counter tests."""
        from paramem.graph.schema import Entity, Relation, SessionGraph

        return SessionGraph(
            session_id="s1",
            timestamp="2026-04-20T12:00:00Z",
            entities=[
                Entity(name="A", entity_type="person"),
                Entity(name="B", entity_type="person"),
            ],
            relations=[
                Relation(
                    subject="A",
                    predicate="knows",
                    object="B",
                    relation_type="social",
                    speaker_id="Speaker0",
                ),
                Relation(
                    subject="B",
                    predicate="knows",
                    object="A",
                    relation_type="social",
                    speaker_id="Speaker0",
                ),
            ],
        )

    def test_counter_increments_when_interim_refinement_not_off(self, tmp_path):
        """extract_session increments the accumulator when interim_refinement != 'off'."""
        loop = _make_loop(
            tmp_path, consolidation_config=ConsolidationConfig(interim_refinement="full")
        )
        assert loop._triples_since_last_enrichment == 0

        sg = self._make_session_graph()
        with patch.object(loop.extraction, "run", return_value=sg):
            loop.extract_session("ignored-transcript", "s1", speaker_id="Speaker0")

        assert loop._triples_since_last_enrichment == 2

    def test_counter_stays_zero_when_interim_refinement_off(self, tmp_path):
        """extract_session does NOT increment the accumulator when interim_refinement='off'."""
        loop = _make_loop(tmp_path)  # interim_refinement defaults to "off"
        assert loop._triples_since_last_enrichment == 0

        sg = self._make_session_graph()
        with patch.object(loop.extraction, "run", return_value=sg):
            loop.extract_session("ignored-transcript", "s1", speaker_id="Speaker0")

        # Merge deferred to full consolidation — counter must remain 0.
        assert loop._triples_since_last_enrichment == 0

    def test_counter_resets_after_successful_enrichment(self, tmp_path):
        """After _run_graph_enrichment returns non-skipped, the counter is zero."""
        loop = _make_loop(tmp_path)
        loop._triples_since_last_enrichment = 42
        _populate_graph(loop.merger.graph, n_persons=12)

        canned = ([], [], "raw")  # Empty enrichment output — still a "successful" pass
        with (
            patch("paramem.training.consolidation._graph_enrich_with_sota", return_value=canned),
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "x"}, clear=False),
        ):
            result = loop._run_graph_enrichment()

        assert not result["skipped"]
        assert loop._triples_since_last_enrichment == 0

    def test_counter_preserved_when_enrichment_skipped(self, tmp_path):
        """Skipped enrichment (e.g. graph-size floor) must NOT reset the counter."""
        loop = _make_loop(tmp_path)
        loop._triples_since_last_enrichment = 99
        # Graph has < 10 nodes → floor skip path.
        assert loop.merger.graph.number_of_nodes() < 10

        result = loop._run_graph_enrichment()

        assert result["skipped"] is True
        assert loop._triples_since_last_enrichment == 99

    def test_interim_full_enrichment_calls_run_graph_enrichment(self, tmp_path):
        """interim_refinement='full' → _run_graph_enrichment called via refine stage."""
        from paramem.training.key_registry import KeyRegistry
        from paramem.utils.config import ConsolidationConfig

        loop = _make_loop(
            tmp_path,
            consolidation_config=ConsolidationConfig(interim_refinement="full"),
            replay_enabled=True,
        )
        for tier in ("episodic", "semantic", "procedural"):
            loop.store.load_registry(tier, KeyRegistry())

        loop.extract_session = MagicMock(
            return_value=(
                [
                    {
                        "question": "q",
                        "answer": "a",
                        "subject": "S",
                        "predicate": "p",
                        "object": "O",
                    }
                ],
                [],
            )
        )
        loop._run_graph_enrichment = MagicMock(return_value={"skipped": False})

        loop.model.peft_config = {"episodic": MagicMock(), "semantic": MagicMock()}
        with (
            patch(
                "paramem.memory.interim_adapter.create_interim_adapter",
                return_value=loop.model,
            ),
            patch("paramem.training.trainer.train_adapter", return_value={}),
            patch("paramem.models.loader.save_adapter"),
            patch("paramem.adapters.manifest.build_manifest_for", return_value=None),
        ):
            loop.post_session_train(
                session_transcript="t",
                session_id="s1",
                speaker_id="Speaker0",
                schedule="12h",
                max_interim_count=7,
                stamp="20260420T1200",
            )

        loop._run_graph_enrichment.assert_called_once()

    def test_interim_light_does_not_enrich(self, tmp_path):
        """interim_refinement='light' → _run_graph_enrichment NOT called by refine."""
        from paramem.training.key_registry import KeyRegistry
        from paramem.utils.config import ConsolidationConfig

        loop = _make_loop(
            tmp_path,
            consolidation_config=ConsolidationConfig(interim_refinement="light"),
            replay_enabled=True,
        )
        for tier in ("episodic", "semantic", "procedural"):
            loop.store.load_registry(tier, KeyRegistry())

        loop.extract_session = MagicMock(
            return_value=(
                [
                    {
                        "question": "q",
                        "answer": "a",
                        "subject": "S",
                        "predicate": "p",
                        "object": "O",
                    }
                ],
                [],
            )
        )
        loop._run_graph_enrichment = MagicMock(return_value={"skipped": True})

        loop.model.peft_config = {"episodic": MagicMock(), "semantic": MagicMock()}
        with (
            patch(
                "paramem.memory.interim_adapter.create_interim_adapter",
                return_value=loop.model,
            ),
            patch("paramem.training.trainer.train_adapter", return_value={}),
            patch("paramem.models.loader.save_adapter"),
            patch("paramem.adapters.manifest.build_manifest_for", return_value=None),
        ):
            loop.post_session_train(
                session_transcript="t",
                session_id="s1",
                speaker_id="Speaker0",
                schedule="12h",
                max_interim_count=7,
                stamp="20260420T1200",
            )

        loop._run_graph_enrichment.assert_not_called()

    def test_interim_off_does_not_enrich(self, tmp_path):
        """interim_refinement='off' → _run_graph_enrichment NOT called by refine."""
        from paramem.training.key_registry import KeyRegistry

        loop = _make_loop(
            tmp_path,
            replay_enabled=True,
        )
        # interim_refinement defaults to "off" in _make_loop (ConsolidationConfig default).
        for tier in ("episodic", "semantic", "procedural"):
            loop.store.load_registry(tier, KeyRegistry())

        loop.extract_session = MagicMock(
            return_value=(
                [
                    {
                        "question": "q",
                        "answer": "a",
                        "subject": "S",
                        "predicate": "p",
                        "object": "O",
                    }
                ],
                [],
            )
        )
        loop._run_graph_enrichment = MagicMock(return_value={"skipped": True})

        loop.model.peft_config = {"episodic": MagicMock(), "semantic": MagicMock()}
        with (
            patch(
                "paramem.memory.interim_adapter.create_interim_adapter",
                return_value=loop.model,
            ),
            patch("paramem.training.trainer.train_adapter", return_value={}),
            patch("paramem.models.loader.save_adapter"),
            patch("paramem.adapters.manifest.build_manifest_for", return_value=None),
        ):
            loop.post_session_train(
                session_transcript="t",
                session_id="s1",
                speaker_id="Speaker0",
                schedule="12h",
                max_interim_count=7,
                stamp="20260420T1200",
            )

        loop._run_graph_enrichment.assert_not_called()

    def test_rollover_hook_skipped_on_cap_reached_absorb(self, tmp_path):
        """Cap-reached absorb path (degenerated early-return) does NOT fire the hook.

        Rather than driving the full cap-reached retrain (which pulls in
        every heavy training dependency), we trip the degenerated health
        gate which short-circuits the absorb branch before any lazy
        imports — still proving the rollover hook is structurally bound
        to the normal-branch else, not the cap-reached if.
        """
        loop = _make_loop(tmp_path, graph_enrichment_min_triples_floor=1)
        loop._triples_since_last_enrichment = 100  # over floor

        loop.extract_session = MagicMock(
            return_value=(
                [
                    {
                        "question": "q",
                        "answer": "a",
                        "subject": "S",
                        "predicate": "p",
                        "object": "O",
                    }
                ],
                [],
            )
        )
        loop._run_graph_enrichment = MagicMock(return_value={"skipped": False})

        existing_stamp = "20260419T1200"
        current_stamp = "20260420T1200"
        newest_name = f"episodic_interim_{existing_stamp}"
        loop.model.peft_config = {
            "episodic": MagicMock(),
            "semantic": MagicMock(),
            newest_name: MagicMock(),
        }

        # Mark the newest interim degenerated → post_session_train returns
        # early with mode="degenerated" before any imports or training.
        # indexed_key_registry is now dict[str, KeyRegistry]; the newest interim
        # tier entry must have is_healthy() == False.
        from paramem.training.key_registry import ADAPTER_HEALTH_DEGENERATED
        from paramem.training.key_registry import KeyRegistry as _KeyRegistry

        _interim_reg = _KeyRegistry()
        _interim_reg.set_health(ADAPTER_HEALTH_DEGENERATED, reason="test-degenerated")
        loop.indexed_key_registry = {
            "episodic": _KeyRegistry(),
            "semantic": _KeyRegistry(),
            "procedural": _KeyRegistry(),
            newest_name: _interim_reg,
        }

        result = loop.post_session_train(
            session_transcript="t",
            session_id="s1",
            speaker_id="Speaker0",
            schedule="12h",
            max_interim_count=1,
            stamp=current_stamp,
        )

        assert result["mode"] == "degenerated"
        loop._run_graph_enrichment.assert_not_called()


# ---------------------------------------------------------------------------
# Tests for _refine_consolidation_graph (B2: enrich gate + recurrence-bump)
# ---------------------------------------------------------------------------


class TestRefineConsolidationGraph:
    """Unit tests for _refine_consolidation_graph's enrich param (B2).

    Covers:
    - enrich=True calls _run_graph_enrichment (fold default, unconditional).
    - enrich=False skips _run_graph_enrichment entirely.
    - Recurrence-bump loop runs regardless of enrich.
    - Empty recon_relations is a safe no-op for both code paths.
    """

    def test_enrich_true_calls_run_graph_enrichment(self, tmp_path):
        """_refine_consolidation_graph(recon, enrich=True) calls _run_graph_enrichment."""
        loop = _make_loop(tmp_path)
        loop._run_graph_enrichment = MagicMock(return_value={"skipped": True})

        loop._refine_consolidation_graph([], enrich=True)

        loop._run_graph_enrichment.assert_called_once()

    def test_enrich_default_calls_run_graph_enrichment(self, tmp_path):
        """enrich defaults to True — fold passes no keyword and gets enrichment."""
        loop = _make_loop(tmp_path)
        loop._run_graph_enrichment = MagicMock(return_value={"skipped": True})

        # No enrich kwarg — fold behavior (default=True).
        loop._refine_consolidation_graph([])

        loop._run_graph_enrichment.assert_called_once()

    def test_enrich_false_skips_run_graph_enrichment(self, tmp_path):
        """_refine_consolidation_graph(recon, enrich=False) does NOT call _run_graph_enrichment."""
        loop = _make_loop(tmp_path)
        loop._run_graph_enrichment = MagicMock(return_value={"skipped": True})

        loop._refine_consolidation_graph([], enrich=False)

        loop._run_graph_enrichment.assert_not_called()

    def test_recurrence_bump_runs_when_enrich_false(self, tmp_path):
        """Recurrence-bump fires regardless of enrich; enrich=False only skips SOTA."""
        from paramem.training.key_registry import KeyRegistry

        loop = _make_loop(tmp_path, replay_enabled=True)
        for tier in ("episodic", "semantic", "procedural"):
            loop.store.load_registry(tier, KeyRegistry())

        # Register a key so bump_recurrence has a target.
        loop.store.put(
            "episodic",
            "graph42",
            {"key": "graph42", "subject": "A", "predicate": "p", "object": "B"},
        )
        loop.store.set_bookkeeping(
            "graph42",
            speaker_id="",
            first_seen_cycle=0,
            relation_type="factual",
            recurrence_count=1,
            last_seen_cycle=0,
        )

        # Simulate a Case-1 collision: merger.reinforcements contains the surviving key.
        loop.merger.reinforcements = ["graph42"]

        from paramem.graph.schema import Relation

        recon_rel = Relation(
            subject="A", predicate="p", object="B", relation_type="factual", speaker_id=""
        )
        loop._run_graph_enrichment = MagicMock(return_value={"skipped": True})

        loop._refine_consolidation_graph([recon_rel], enrich=False)

        loop._run_graph_enrichment.assert_not_called()
        bk = loop.store.bookkeeping_for_key("graph42")
        assert bk is not None
        assert bk["recurrence_count"] == 2, (
            f"Recurrence should have been bumped to 2; got {bk['recurrence_count']}"
        )

    def test_recurrence_bump_runs_when_enrich_true(self, tmp_path):
        """Recurrence-bump fires when enrich=True as well (both code paths covered)."""
        from paramem.training.key_registry import KeyRegistry

        loop = _make_loop(tmp_path, replay_enabled=True)
        for tier in ("episodic", "semantic", "procedural"):
            loop.store.load_registry(tier, KeyRegistry())

        loop.store.put(
            "episodic",
            "graph7",
            {"key": "graph7", "subject": "X", "predicate": "q", "object": "Y"},
        )
        loop.store.set_bookkeeping(
            "graph7",
            speaker_id="",
            first_seen_cycle=0,
            relation_type="factual",
            recurrence_count=3,
            last_seen_cycle=0,
        )
        loop.merger.reinforcements = ["graph7"]

        from paramem.graph.schema import Relation

        recon_rel = Relation(
            subject="X", predicate="q", object="Y", relation_type="factual", speaker_id=""
        )
        loop._run_graph_enrichment = MagicMock(return_value={"skipped": False})

        loop._refine_consolidation_graph([recon_rel], enrich=True)

        loop._run_graph_enrichment.assert_called_once()
        bk = loop.store.bookkeeping_for_key("graph7")
        assert bk is not None
        assert bk["recurrence_count"] == 4, (
            f"Recurrence should have been bumped to 4; got {bk['recurrence_count']}"
        )

    def test_empty_recon_relations_is_safe_noop_for_bump(self, tmp_path):
        """Empty recon_relations → recurrence-bump loop does not run; no crash."""
        loop = _make_loop(tmp_path)
        loop.merger.reinforcements = ["graph99"]  # would be bumped if guard failed
        loop._run_graph_enrichment = MagicMock(return_value={"skipped": True})

        bump_spy = MagicMock()
        loop.store.bump_recurrence = bump_spy

        # Both enrich values must be safe no-ops when recon_relations is empty.
        loop._refine_consolidation_graph([], enrich=False)
        loop._refine_consolidation_graph([], enrich=True)

        bump_spy.assert_not_called()

    def test_fold_call_byte_identical_behavior(self, tmp_path):
        """The fold's call _refine_consolidation_graph(recon) uses enrich=True default.

        Verify the fold's unchanged call site (_refine_consolidation_graph without
        enrich kwarg) produces identical behavior to passing enrich=True explicitly.
        Both must call _run_graph_enrichment.
        """
        loop_a = _make_loop(tmp_path / "a")
        loop_a._run_graph_enrichment = MagicMock(return_value={"skipped": True})
        loop_a._refine_consolidation_graph([])  # fold call (no kwarg)
        count_a = loop_a._run_graph_enrichment.call_count

        loop_b = _make_loop(tmp_path / "b")
        loop_b._run_graph_enrichment = MagicMock(return_value={"skipped": True})
        loop_b._refine_consolidation_graph([], enrich=True)  # explicit True
        count_b = loop_b._run_graph_enrichment.call_count

        assert count_a == count_b == 1, (
            f"Fold call and enrich=True must both call _run_graph_enrichment once; "
            f"got {count_a} vs {count_b}"
        )


# ---------------------------------------------------------------------------
# Tests for _harvest_keyless_edge_entries / _apply_keyless_edge_entries (fold pre-pass)
# ---------------------------------------------------------------------------


class TestHarvestKeylessEdges:
    """Unit tests for the fold pre-pass that mints keys for keyless graph edges.

    Uses _make_loop from this module (real nx.MultiDiGraph + real MemoryStore,
    mocked model/tokenizer so no GPU).  replay_enabled=True so store.put()
    writes into the KeyRegistry.
    """

    def test_keyless_edge_minted_in_store_and_tier_keyed(self, tmp_path):
        """A keyless predicate-bearing edge produces a key in store + tier_keyed."""
        from paramem.training.key_registry import KeyRegistry

        loop = _make_loop(tmp_path, replay_enabled=True)
        for tier in ("episodic", "semantic", "procedural"):
            loop.store.load_registry(tier, KeyRegistry())

        graph = loop.merger.graph
        graph.add_edge(
            "Alice",
            "Berlin",
            predicate="lives_in",
            relation_type="factual",
            confidence=0.9,
        )

        tier_keyed: dict = {"episodic": [], "semantic": [], "procedural": []}
        _h = loop._harvest_keyless_edge_entries(tag_new=False, default_speaker_id="")
        loop._apply_keyless_edge_entries(_h, tier_keyed)

        # Exactly one key minted (factual → episodic).
        assert len(tier_keyed["episodic"]) == 1
        assert len(tier_keyed["semantic"]) == 0
        assert len(tier_keyed["procedural"]) == 0

        entry = tier_keyed["episodic"][0]
        assert entry["subject"] == "Alice"
        assert entry["predicate"] == "lives_in"
        assert entry["object"] == "Berlin"
        key = entry["key"]
        assert key.startswith("graph")

        # Key is registered in the store and has bookkeeping.
        all_keys = loop.store.all_active_keys()
        assert key in all_keys

        bk = loop.store.bookkeeping_for_key(key)
        assert bk is not None
        assert bk["recurrence_count"] == 1
        assert bk["first_seen_cycle"] == loop.cycle_count
        assert bk["last_seen_cycle"] == loop.cycle_count
        assert bk["relation_type"] == "factual"
        assert bk["speaker_id"] == ""

    def test_counter_advanced_for_each_minted_key(self, tmp_path):
        """_indexed_next_index advances once per minted episodic key."""
        from paramem.training.key_registry import KeyRegistry

        loop = _make_loop(tmp_path, replay_enabled=True)
        for tier in ("episodic", "semantic", "procedural"):
            loop.store.load_registry(tier, KeyRegistry())

        initial_index = loop._indexed_next_index
        graph = loop.merger.graph
        # Add two keyless edges.
        graph.add_edge("Alice", "Berlin", predicate="lives_in", relation_type="factual")
        graph.add_edge("Bob", "Coffee", predicate="likes", relation_type="factual")

        tier_keyed: dict = {"episodic": [], "semantic": [], "procedural": []}
        _h = loop._harvest_keyless_edge_entries(tag_new=False, default_speaker_id="")
        loop._apply_keyless_edge_entries(_h, tier_keyed)

        assert len(tier_keyed["episodic"]) == 2
        assert loop._indexed_next_index == initial_index + 2

        # Keys are sequential from the initial index.
        keys = {e["key"] for e in tier_keyed["episodic"]}
        assert f"graph{initial_index}" in keys
        assert f"graph{initial_index + 1}" in keys

    def test_keyed_edge_not_reminted(self, tmp_path):
        """An edge that already has an ik_key attribute must be left untouched."""
        from paramem.memory.persistence import _IK_KEY_ATTR
        from paramem.training.key_registry import KeyRegistry

        loop = _make_loop(tmp_path, replay_enabled=True)
        for tier in ("episodic", "semantic", "procedural"):
            loop.store.load_registry(tier, KeyRegistry())

        graph = loop.merger.graph
        graph.add_edge(
            "Alice",
            "Berlin",
            predicate="lives_in",
            relation_type="factual",
            **{_IK_KEY_ATTR: "graph1"},  # already keyed
        )

        tier_keyed: dict = {"episodic": [], "semantic": [], "procedural": []}
        initial_index = loop._indexed_next_index
        _h = loop._harvest_keyless_edge_entries(tag_new=False, default_speaker_id="")
        loop._apply_keyless_edge_entries(_h, tier_keyed)

        # Nothing minted — the edge already has a key.
        assert tier_keyed["episodic"] == []
        assert loop._indexed_next_index == initial_index

    def test_predicate_less_edge_not_minted(self, tmp_path):
        """An edge with no predicate must not receive a key (negative control)."""
        from paramem.training.key_registry import KeyRegistry

        loop = _make_loop(tmp_path, replay_enabled=True)
        for tier in ("episodic", "semantic", "procedural"):
            loop.store.load_registry(tier, KeyRegistry())

        graph = loop.merger.graph
        # Add an edge with NO predicate field (not keyable).
        graph.add_edge("Alice", "Berlin", relation_type="factual", confidence=0.5)

        tier_keyed: dict = {"episodic": [], "semantic": [], "procedural": []}
        initial_index = loop._indexed_next_index
        _h = loop._harvest_keyless_edge_entries(tag_new=False, default_speaker_id="")
        loop._apply_keyless_edge_entries(_h, tier_keyed)

        assert tier_keyed["episodic"] == []
        assert tier_keyed["semantic"] == []
        assert tier_keyed["procedural"] == []
        assert loop._indexed_next_index == initial_index

    def test_minted_key_present_in_store_all_active_keys(self, tmp_path):
        """Minted key is retrievable via store.all_active_keys() — not counted as drift."""
        from paramem.training.key_registry import KeyRegistry

        loop = _make_loop(tmp_path, replay_enabled=True)
        for tier in ("episodic", "semantic", "procedural"):
            loop.store.load_registry(tier, KeyRegistry())

        graph = loop.merger.graph
        graph.add_edge("Carol", "London", predicate="visited", relation_type="factual")

        tier_keyed: dict = {"episodic": [], "semantic": [], "procedural": []}
        _h = loop._harvest_keyless_edge_entries(tag_new=False, default_speaker_id="")
        loop._apply_keyless_edge_entries(_h, tier_keyed)

        minted_key = tier_keyed["episodic"][0]["key"]
        _all_keyed = {e["key"] for tl in tier_keyed.values() for e in tl}
        active_keys = loop.store.all_active_keys()

        # Key must be in both sets so drift computation excludes it.
        assert minted_key in _all_keyed
        assert minted_key in active_keys

    def test_relation_type_threaded_through(self, tmp_path):
        """Edge relation_type is correctly recorded in bookkeeping."""
        from paramem.training.key_registry import KeyRegistry

        loop = _make_loop(tmp_path, replay_enabled=True)
        for tier in ("episodic", "semantic", "procedural"):
            loop.store.load_registry(tier, KeyRegistry())

        graph = loop.merger.graph
        graph.add_edge(
            "Alice",
            "Tea",
            predicate="prefers",
            relation_type="preference",
            confidence=0.9,
        )

        tier_keyed: dict = {"episodic": [], "semantic": [], "procedural": []}
        _h = loop._harvest_keyless_edge_entries(tag_new=False, default_speaker_id="")
        loop._apply_keyless_edge_entries(_h, tier_keyed)

        # preference → episodic (no procedural adapter in _make_loop).
        assert len(tier_keyed["episodic"]) == 1
        key = tier_keyed["episodic"][0]["key"]
        bk = loop.store.bookkeeping_for_key(key)
        assert bk is not None
        assert bk["relation_type"] == "preference"

    def test_procedural_tier_minting(self, tmp_path):
        """Keyless preference edge routes to procedural when procedural_config is set.

        _make_loop passes procedural_adapter_config=None so the procedural
        branch of _harvest_keyless_edge_entries never fires in the other tests.
        This test constructs the loop the same way _make_loop does but adds a
        real AdapterConfig as procedural_adapter_config and pre-populates
        "procedural" in model.peft_config so _ensure_adapters skips creation.

        filter_procedural_relations routes relation_type=="preference" to the
        procedural bucket (primary gate).  The minted key must carry prefix
        "proc", land in tier_keyed["procedural"], appear in store.all_active_keys(),
        and advance _procedural_next_index by exactly 1.
        """
        from paramem.memory.store import MemoryStore as _MS
        from paramem.training.key_registry import KeyRegistry

        # Build the loop directly (mirror _make_loop) with a procedural config.
        model = MagicMock()
        model.__class__ = PeftModel
        # Pre-populate "procedural" so _ensure_adapters skips create_adapter.
        model.peft_config = {
            "episodic": MagicMock(),
            "semantic": MagicMock(),
            "procedural": MagicMock(),
            "in_training": MagicMock(),
        }

        store = _MS(replay_enabled=True)
        loop = ConsolidationLoop(
            model=model,
            tokenizer=MagicMock(),
            consolidation_config=ConsolidationConfig(),
            training_config=TrainingConfig(),
            episodic_adapter_config=AdapterConfig(),
            semantic_adapter_config=AdapterConfig(),
            memory_store=store,
            procedural_adapter_config=AdapterConfig(),
            output_dir=tmp_path,
            extraction_noise_filter="anthropic",
            extraction_noise_filter_model="claude-sonnet-4-6",
        )
        loop._probe_passing_keys = lambda adapter_name, entries: {e["key"] for e in entries}

        for tier in ("episodic", "semantic", "procedural"):
            loop.store.load_registry(tier, KeyRegistry())

        initial_proc_index = loop._procedural_next_index

        # relation_type="preference" is the primary gate in filter_procedural_relations.
        loop.merger.graph.add_edge(
            "Alice",
            "Coffee",
            predicate="prefers",
            relation_type="preference",
            confidence=0.9,
        )

        tier_keyed: dict = {"episodic": [], "semantic": [], "procedural": []}
        _h = loop._harvest_keyless_edge_entries(tag_new=False, default_speaker_id="")
        loop._apply_keyless_edge_entries(_h, tier_keyed)

        # The preference edge routes to procedural, not episodic.
        assert len(tier_keyed["procedural"]) == 1
        assert len(tier_keyed["episodic"]) == 0

        entry = tier_keyed["procedural"][0]
        key = entry["key"]
        assert key.startswith("proc"), f"Expected 'proc' prefix, got key={key!r}"

        # Key is in the store's active set.
        assert key in loop.store.all_active_keys()

        # _procedural_next_index advanced by exactly 1.
        assert loop._procedural_next_index == initial_proc_index + 1

    def test_highwater_seeding_prevents_collision(self, tmp_path):
        """_indexed_next_index seeds from existing store keys; new key avoids collision.

        Pre-seed the store with graph5 before constructing the loop so the
        constructor's high-water scan sets _indexed_next_index to 6.  Inject
        one keyless episodic edge.  The minted key must be graph6 — not graph1
        (unchecked start) and not graph5 (collision with the pre-existing key).
        """
        from paramem.memory.store import MemoryStore as _MS
        from paramem.training.key_registry import KeyRegistry

        # Build and hydrate the store BEFORE loop construction so the
        # constructor's _indexed_next_index seeding scan sees graph5.
        store = _MS(replay_enabled=True)
        for tier in ("episodic", "semantic", "procedural"):
            store.load_registry(tier, KeyRegistry())
        store.put(
            "episodic",
            "graph5",
            {
                "key": "graph5",
                "question": "q",
                "answer": "a",
                "subject": "Prior",
                "predicate": "knows",
                "object": "Fact",
            },
        )

        model = MagicMock()
        model.__class__ = PeftModel
        model.peft_config = {
            "episodic": MagicMock(),
            "semantic": MagicMock(),
            "in_training": MagicMock(),
        }

        loop = ConsolidationLoop(
            model=model,
            tokenizer=MagicMock(),
            consolidation_config=ConsolidationConfig(),
            training_config=TrainingConfig(),
            episodic_adapter_config=AdapterConfig(),
            semantic_adapter_config=AdapterConfig(),
            memory_store=store,
            procedural_adapter_config=None,
            output_dir=tmp_path,
            extraction_noise_filter="anthropic",
            extraction_noise_filter_model="claude-sonnet-4-6",
        )
        loop._probe_passing_keys = lambda adapter_name, entries: {e["key"] for e in entries}

        # Constructor must have picked up graph5 → _indexed_next_index == 6.
        assert loop._indexed_next_index == 6, (
            f"Expected _indexed_next_index=6 after seeding graph5, got {loop._indexed_next_index}"
        )

        loop.merger.graph.add_edge(
            "New",
            "Fact",
            predicate="relates_to",
            relation_type="factual",
            confidence=0.8,
        )

        tier_keyed: dict = {"episodic": [], "semantic": [], "procedural": []}
        _h = loop._harvest_keyless_edge_entries(tag_new=False, default_speaker_id="")
        loop._apply_keyless_edge_entries(_h, tier_keyed)

        assert len(tier_keyed["episodic"]) == 1
        minted_key = tier_keyed["episodic"][0]["key"]

        # Must be graph6 — no collision with the pre-existing graph5.
        assert minted_key == "graph6", (
            f"Expected minted key 'graph6' to avoid collision with graph5, got {minted_key!r}"
        )
        # Pre-existing graph5 entry must still be intact.
        assert "graph5" in loop.store.all_active_keys()


class TestHarvestApplySplit:
    """Tests verifying the pure-harvester / applier split of keyless-edge minting.

    The harvester must be side-effect free; the applier must reproduce exactly
    the writes the original combined method produced.
    """

    def test_harvester_does_no_writes(self, tmp_path):
        """_harvest_keyless_edge_entries must NOT call store.put, store.set_bookkeeping,
        or advance self._indexed_next_index / self._procedural_next_index.  It must
        return the expected number of harvest records.
        """
        from unittest.mock import patch

        from paramem.training.key_registry import KeyRegistry

        loop = _make_loop(tmp_path, replay_enabled=True)
        for tier in ("episodic", "semantic", "procedural"):
            loop.store.load_registry(tier, KeyRegistry())

        loop.merger.graph.add_edge("Alice", "Berlin", predicate="lives_in", relation_type="factual")
        loop.merger.graph.add_edge("Bob", "Coffee", predicate="likes", relation_type="factual")

        initial_indexed = loop._indexed_next_index
        initial_procedural = loop._procedural_next_index

        with (
            patch.object(loop.store, "put", wraps=loop.store.put) as mock_put,
            patch.object(
                loop.store, "set_bookkeeping", wraps=loop.store.set_bookkeeping
            ) as mock_bk,
        ):
            harvested = loop._harvest_keyless_edge_entries(tag_new=False, default_speaker_id="")

            # No store writes must occur during harvest.
            mock_put.assert_not_called()
            mock_bk.assert_not_called()

        # Counters must be unchanged.
        assert loop._indexed_next_index == initial_indexed
        assert loop._procedural_next_index == initial_procedural

        # Two keyless edges → two harvest records.
        assert len(harvested) == 2

    def test_applier_reproduces_writes(self, tmp_path):
        """_apply_keyless_edge_entries must produce the same store writes, tier_keyed
        entries, and counter end-state as the original combined method did.
        """
        from paramem.training.key_registry import KeyRegistry

        loop = _make_loop(tmp_path, replay_enabled=True)
        for tier in ("episodic", "semantic", "procedural"):
            loop.store.load_registry(tier, KeyRegistry())

        loop.merger.graph.add_edge("Carol", "London", predicate="visited", relation_type="factual")

        initial_indexed = loop._indexed_next_index

        harvested = loop._harvest_keyless_edge_entries(tag_new=False, default_speaker_id="")
        tier_keyed: dict = {"episodic": [], "semantic": [], "procedural": []}
        minted_by_tier, _deferred = loop._apply_keyless_edge_entries(harvested, tier_keyed)

        # Exactly one episodic key minted.  defer=False (default) → no deferred writes.
        assert minted_by_tier == {"episodic": 1, "procedural": 0}
        assert _deferred == [], "defer=False must produce empty deferred_writes"
        assert len(tier_keyed["episodic"]) == 1

        entry = tier_keyed["episodic"][0]
        key = entry["key"]
        assert key.startswith("graph")
        assert entry["subject"] == "Carol"
        assert entry["predicate"] == "visited"
        assert entry["object"] == "London"

        # Key is in the store.
        assert key in loop.store.all_active_keys()

        # Bookkeeping is present.
        bk = loop.store.bookkeeping_for_key(key)
        assert bk is not None
        assert bk["recurrence_count"] == 1
        assert bk["relation_type"] == "factual"

        # Counter advanced by 1.
        assert loop._indexed_next_index == initial_indexed + 1

    def test_sequential_indices_two_edges(self, tmp_path):
        """Two keyless edges get sequential keys (graphN, graphN+1) and counters
        advance by exactly 2.
        """
        from paramem.training.key_registry import KeyRegistry

        loop = _make_loop(tmp_path, replay_enabled=True)
        for tier in ("episodic", "semantic", "procedural"):
            loop.store.load_registry(tier, KeyRegistry())

        initial_indexed = loop._indexed_next_index

        loop.merger.graph.add_edge("Dave", "Paris", predicate="lives_in", relation_type="factual")
        loop.merger.graph.add_edge("Eve", "Tea", predicate="likes", relation_type="factual")

        harvested = loop._harvest_keyless_edge_entries(tag_new=False, default_speaker_id="")
        tier_keyed: dict = {"episodic": [], "semantic": [], "procedural": []}
        loop._apply_keyless_edge_entries(harvested, tier_keyed)

        assert len(tier_keyed["episodic"]) == 2
        assert loop._indexed_next_index == initial_indexed + 2

        # Keys must be exactly the two sequential indices.
        minted_keys = {e["key"] for e in tier_keyed["episodic"]}
        assert f"graph{initial_indexed}" in minted_keys
        assert f"graph{initial_indexed + 1}" in minted_keys

    def test_harvester_no_keyless_edges_does_not_read_counters(self, tmp_path):
        """With no keyless edges to mint, the harvester must not read the index
        counters at all — it returns [] even when those attributes are absent.

        Guards the lazy-seed contract: callers that exercise the keyed-edge walk
        without any keyless edges (e.g. a graph of only keyed or predicate-less
        edges) need not have the index counters initialised.
        """
        from paramem.training.key_registry import KeyRegistry

        loop = _make_loop(tmp_path, replay_enabled=True)
        for tier in ("episodic", "semantic", "procedural"):
            loop.store.load_registry(tier, KeyRegistry())

        # Simulate a caller that never set the index counters.
        del loop._indexed_next_index
        del loop._procedural_next_index

        # Only a predicate-less edge — nothing keyless+keyable to mint.
        loop.merger.graph.add_edge("Dave", "Paris")

        harvested = loop._harvest_keyless_edge_entries(tag_new=False, default_speaker_id="")
        assert harvested == []


class TestEnrichmentRemovalLedger:
    """Tests that _run_graph_enrichment writes ik_keys of same_as-contracted
    edges to merger.removal_ledger with reason='enrichment_same_as'.
    """

    def test_same_as_contraction_writes_keyed_edge_to_ledger(self, tmp_path, monkeypatch):
        """A successful same_as contraction that drops a keyed edge writes the
        edge's ik_key to merger.removal_ledger with reason='enrichment_same_as'
        and merged_into set to the keep node.
        """
        from paramem.memory.persistence import _IK_KEY_ATTR

        loop = _make_loop(tmp_path)
        graph = loop.merger.graph
        _populate_graph(graph, n_persons=10)

        # Add keep/drop nodes with an edge carrying an ik_key — canonical-keyed (lowercase).
        graph.add_node(
            "alice",
            entity_type="person",
            attributes={"name": "Alice"},
            recurrence_count=3,
            sessions=["s010"],
            first_seen="s010",
            last_seen="s010",
        )
        graph.add_node(
            "alicia",
            entity_type="person",
            attributes={"name": "Alicia"},
            recurrence_count=1,
            sessions=["s011"],
            first_seen="s011",
            last_seen="s011",
        )
        # Edge from keep → drop carrying an ik_key (becomes a self-loop on contraction).
        eid = graph.add_edge(
            "alice",
            "alicia",
            predicate="same as",
            relation_type="factual",
            confidence=1.0,
            source="extraction",
            sessions=["s010"],
        )
        graph["alice"]["alicia"][eid][_IK_KEY_ATTR] = "key_same_as_victim"

        # SOTA returns surface names; production canonicalizes before graph lookup.
        canned_result = (
            [],
            [["Alice", "Alicia"]],  # keep=alice, drop=alicia
            "raw",
        )
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with patch(
            "paramem.training.consolidation._graph_enrich_with_sota",
            return_value=canned_result,
        ):
            result = loop._run_graph_enrichment()

        assert result["same_as_merges"] >= 1, "Contraction must have fired for alice/alicia"
        assert "key_same_as_victim" in loop.merger.removal_ledger, (
            f"Dropped ik_key must appear in merger.removal_ledger; "
            f"ledger={list(loop.merger.removal_ledger.keys())}"
        )
        entry = loop.merger.removal_ledger["key_same_as_victim"]
        assert entry["reason"] == "enrichment_same_as", (
            f"Expected reason='enrichment_same_as'; got {entry['reason']!r}"
        )
        # merged_into is the canonical keep node key
        assert entry["merged_into"] == "alice", (
            f"Expected merged_into='alice' (canonical keep node); got {entry['merged_into']!r}"
        )

    def test_failed_contraction_does_not_write_to_ledger(self, tmp_path, monkeypatch):
        """A contraction that raises does NOT write phantom entries to ledger."""
        from paramem.memory.persistence import _IK_KEY_ATTR

        loop = _make_loop(tmp_path)
        graph = loop.merger.graph
        _populate_graph(graph, n_persons=10)

        # Nodes are canonical-keyed ("BadKeep" → "badkeep", "BadDrop" → "baddrop").
        graph.add_node(
            "badkeep",
            entity_type="person",
            attributes={},
            recurrence_count=1,
            sessions=["s020"],
            first_seen="s020",
            last_seen="s020",
        )
        graph.add_node(
            "baddrop",
            entity_type="person",
            attributes={},
            recurrence_count=1,
            sessions=["s021"],
            first_seen="s021",
            last_seen="s021",
        )
        eid = graph.add_edge(
            "badkeep",
            "baddrop",
            predicate="related",
            relation_type="factual",
            confidence=0.9,
            source="extraction",
            sessions=["s020"],
        )
        graph["badkeep"]["baddrop"][eid][_IK_KEY_ATTR] = "key_bad_victim"

        # SOTA returns surface names; production canonicalizes: "BadKeep" → "badkeep".
        canned_result = (
            [],
            [["BadKeep", "BadDrop"]],
            "raw",
        )
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        # Patch contracted_nodes to always raise so the contraction fails.
        with (
            patch(
                "paramem.training.consolidation._graph_enrich_with_sota",
                return_value=canned_result,
            ),
            patch("networkx.contracted_nodes", side_effect=ValueError("forced failure")),
        ):
            result = loop._run_graph_enrichment()

        assert result["same_as_merges"] == 0, "No merges should succeed when contracted_nodes fails"
        assert "key_bad_victim" not in loop.merger.removal_ledger, (
            "Failed contraction must NOT write to removal_ledger"
        )


# ---------------------------------------------------------------------------
# B3 interim keying seams
# ---------------------------------------------------------------------------


class TestB3InterimKeyingSeams:
    """B3: graph-walk keying for the interim path.

    Covers:
    - dcf4189 invariant: speaker_id propagated through the graph-walk.
    - default_speaker_id fallback (absent node attr → caller's id).
    - Explicit speaker_id="" preserved (not overwritten by default).
    - defer=True performs NO store writes and NO counter advances.
    - defer=True returns the full harvested list as deferred_writes.
    - tag_new=True stamps minted entries with the _new sentinel.
    """

    def test_speaker_id_from_node_attr_carried_through(self, tmp_path):
        """dcf4189 invariant: _harvest_keyless_edge_entries reads speaker_id from
        the subject node's top-level attribute and threads it into the harvest
        record, so _apply_keyless_edge_entries (and the interim flush at step 11b)
        write the correct speaker_id to store.set_bookkeeping.

        A node WITH speaker_id set gets that value; a node WITHOUT speaker_id attr
        gets the caller-supplied default_speaker_id.
        """
        from paramem.training.key_registry import KeyRegistry

        loop = _make_loop(tmp_path, replay_enabled=True)
        for tier in ("episodic", "semantic", "procedural"):
            loop.store.load_registry(tier, KeyRegistry())

        # Node with speaker_id set — simulates a speaker-attributed entity.
        loop.merger.graph.add_node("alice", speaker_id="Speaker0", attributes={"name": "Alice"})
        loop.merger.graph.add_node("berlin", attributes={"name": "Berlin"})
        loop.merger.graph.add_edge("alice", "berlin", predicate="lives_in", relation_type="factual")

        # Node without speaker_id attr — harvester applies default_speaker_id.
        loop.merger.graph.add_node("bob", attributes={"name": "Bob"})
        loop.merger.graph.add_node("coffee", attributes={"name": "Coffee"})
        loop.merger.graph.add_edge("bob", "coffee", predicate="likes", relation_type="factual")

        harvested = loop._harvest_keyless_edge_entries(
            tag_new=True, default_speaker_id="DefaultSpeaker"
        )

        assert len(harvested) == 2

        # Find each record by subject display name.
        alice_rec = next(r for r in harvested if r["entry"]["subject"] == "Alice")
        bob_rec = next(r for r in harvested if r["entry"]["subject"] == "Bob")

        # Alice's node has speaker_id="Speaker0" → entry gets that value.
        assert alice_rec["speaker_id"] == "Speaker0", (
            f"Expected 'Speaker0' for attributed node; got {alice_rec['speaker_id']!r}"
        )

        # Bob's node has NO speaker_id attr → fallback to default_speaker_id.
        assert bob_rec["speaker_id"] == "DefaultSpeaker", (
            f"Expected 'DefaultSpeaker' (default) for unattributed node; "
            f"got {bob_rec['speaker_id']!r}"
        )

    def test_explicit_empty_speaker_id_not_overwritten_by_default(self, tmp_path):
        """A node with speaker_id="" explicitly set keeps "" — the default is NOT
        applied.  This mirrors _tag_speaker_id_defaults semantics: only ABSENT keys
        receive the default; present-but-empty is a distinct signal.
        """
        from paramem.training.key_registry import KeyRegistry

        loop = _make_loop(tmp_path, replay_enabled=True)
        for tier in ("episodic", "semantic", "procedural"):
            loop.store.load_registry(tier, KeyRegistry())

        # Node with explicit speaker_id="" — must not be overwritten.
        loop.merger.graph.add_node("carol", speaker_id="", attributes={"name": "Carol"})
        loop.merger.graph.add_node("london", attributes={"name": "London"})
        loop.merger.graph.add_edge("carol", "london", predicate="visits", relation_type="factual")

        harvested = loop._harvest_keyless_edge_entries(
            tag_new=True, default_speaker_id="ShouldNotAppear"
        )

        assert len(harvested) == 1
        rec = harvested[0]

        # Explicit "" is preserved — default_speaker_id must NOT overwrite it.
        assert rec["speaker_id"] == "", (
            f"Explicit speaker_id='' was overwritten; got {rec['speaker_id']!r}"
        )

    def test_defer_true_no_store_writes_no_counter_advance(self, tmp_path):
        """_apply_keyless_edge_entries(defer=True) must NOT write to the store and
        must NOT advance _indexed_next_index or _procedural_next_index.

        This is the interim-atomicity contract: store mutations are deferred until
        the caller confirms successful training.  A training abort leaves the
        registry completely clean.
        """
        from unittest.mock import patch

        from paramem.training.key_registry import KeyRegistry

        loop = _make_loop(tmp_path, replay_enabled=True)
        for tier in ("episodic", "semantic", "procedural"):
            loop.store.load_registry(tier, KeyRegistry())

        loop.merger.graph.add_edge("Dave", "Paris", predicate="lives_in", relation_type="factual")
        loop.merger.graph.add_edge("Eve", "Tea", predicate="likes", relation_type="factual")

        initial_indexed = loop._indexed_next_index
        initial_procedural = loop._procedural_next_index

        harvested = loop._harvest_keyless_edge_entries(tag_new=True, default_speaker_id="")

        with (
            patch.object(loop.store, "put", wraps=loop.store.put) as mock_put,
            patch.object(
                loop.store, "set_bookkeeping", wraps=loop.store.set_bookkeeping
            ) as mock_bk,
        ):
            tier_keyed: dict = {"episodic": [], "semantic": [], "procedural": []}
            minted_by_tier, deferred_writes = loop._apply_keyless_edge_entries(
                harvested, tier_keyed, defer=True
            )

            # No store writes must occur when defer=True.
            mock_put.assert_not_called()
            mock_bk.assert_not_called()

        # Counters must be unchanged when defer=True.
        assert loop._indexed_next_index == initial_indexed, (
            f"_indexed_next_index advanced during defer=True: "
            f"{initial_indexed} → {loop._indexed_next_index}"
        )
        assert loop._procedural_next_index == initial_procedural, (
            "_procedural_next_index advanced during defer=True"
        )

        # tier_keyed is still populated for training set construction.
        assert len(tier_keyed["episodic"]) == 2, (
            f"tier_keyed['episodic'] must be populated even with defer=True; "
            f"got {len(tier_keyed['episodic'])} entries"
        )

        # deferred_writes contains all harvested records for later flush.
        assert len(deferred_writes) == 2, (
            f"deferred_writes must hold all harvested records; got {len(deferred_writes)}"
        )

        # Store remains empty — no orphan keys.
        assert not loop.store.all_active_keys(), (
            f"Store must be empty after defer=True; got {loop.store.all_active_keys()}"
        )

    def test_defer_true_deferred_writes_have_required_flush_fields(self, tmp_path):
        """deferred_writes records from _apply_keyless_edge_entries(defer=True) must
        carry all fields required for the caller's flush (entry, canon_subj, canon_obj,
        predicate, tier, speaker_id, relation_type).
        """
        from paramem.training.key_registry import KeyRegistry

        loop = _make_loop(tmp_path, replay_enabled=True)
        for tier in ("episodic", "semantic", "procedural"):
            loop.store.load_registry(tier, KeyRegistry())

        loop.merger.graph.add_node("frank", speaker_id="Speaker1", attributes={"name": "Frank"})
        loop.merger.graph.add_node("hamburg", attributes={"name": "Hamburg"})
        loop.merger.graph.add_edge(
            "frank", "hamburg", predicate="works_in", relation_type="factual"
        )

        harvested = loop._harvest_keyless_edge_entries(tag_new=True, default_speaker_id="Speaker1")
        tier_keyed: dict = {"episodic": [], "semantic": [], "procedural": []}
        _, deferred_writes = loop._apply_keyless_edge_entries(harvested, tier_keyed, defer=True)

        assert len(deferred_writes) == 1
        rec = deferred_writes[0]

        required_fields = (
            "entry",
            "canon_subj",
            "canon_obj",
            "predicate",
            "tier",
            "speaker_id",
            "relation_type",
        )
        for field in required_fields:
            assert field in rec, f"deferred_writes record missing field {field!r}"

        assert rec["tier"] == "episodic"
        assert rec["predicate"] == "works_in"
        assert rec["speaker_id"] == "Speaker1"
        assert rec["relation_type"] == "factual"

        # entry must have key, subject, predicate, object.
        entry = rec["entry"]
        for f in ("key", "subject", "predicate", "object"):
            assert f in entry, f"entry dict missing field {f!r}"
        assert entry["subject"] == "Frank"
        assert entry["object"] == "Hamburg"

    def test_tag_new_sentinel_on_minted_entries(self, tmp_path):
        """tag_new=True stamps minted entries with the _new sentinel so the interim
        path can identify freshly-minted keys vs existing-key replay entries.
        """
        from paramem.training.key_registry import KeyRegistry

        loop = _make_loop(tmp_path, replay_enabled=True)
        for tier in ("episodic", "semantic", "procedural"):
            loop.store.load_registry(tier, KeyRegistry())

        loop.merger.graph.add_edge("Grace", "Oslo", predicate="visits", relation_type="factual")

        # tag_new=True — minted entries get the _new sentinel.
        harvested_new = loop._harvest_keyless_edge_entries(tag_new=True, default_speaker_id="")
        assert len(harvested_new) == 1
        entry_new = harvested_new[0]["entry"]
        assert entry_new.get("_new") is True, (
            f"tag_new=True must set '_new'=True on the entry; got {entry_new!r}"
        )

        # tag_new=False — minted entries do NOT get the sentinel.
        loop._indexed_next_index = 1  # reset so next harvest gets a fresh index
        loop.merger.graph.clear_edges()
        loop.merger.graph.add_edge("Hank", "Rome", predicate="visits", relation_type="factual")

        harvested_nosentinel = loop._harvest_keyless_edge_entries(
            tag_new=False, default_speaker_id=""
        )
        assert len(harvested_nosentinel) == 1
        entry_nosentinel = harvested_nosentinel[0]["entry"]
        assert "_new" not in entry_nosentinel or entry_nosentinel.get("_new") is not True, (
            f"tag_new=False must NOT set '_new'=True; got {entry_nosentinel!r}"
        )
