"""Tests for graph-level SOTA enrichment (Task #10).

All tests are pure-Python — no GPU required. SOTA calls are mocked so
the test suite does not make any network requests.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import networkx as nx

from paramem.training.consolidation import ConsolidationLoop, _serialize_subgraph_triples
from paramem.utils.config import AdapterConfig, ConsolidationConfig, TrainingConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_loop(tmp_path, **kwargs) -> ConsolidationLoop:
    """Build a minimal ConsolidationLoop for enrichment tests.

    Graph is transient (persist_graph=False). Model/tokenizer are mocks
    so no GPU is touched.  The mock model pre-populates ``peft_config``
    with all three required adapters so ``_ensure_adapters`` skips the
    real PEFT ``create_adapter`` calls.

    Keyword args forwarded to ConsolidationLoop override the defaults
    set here (e.g. pass ``extraction_noise_filter=""`` to test the
    no-provider skip path).
    """
    model = MagicMock()
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

    loop = ConsolidationLoop(
        model=model,
        tokenizer=MagicMock(),
        consolidation_config=ConsolidationConfig(),
        training_config=TrainingConfig(),
        episodic_adapter_config=AdapterConfig(),
        semantic_adapter_config=AdapterConfig(),
        procedural_adapter_config=None,
        output_dir=tmp_path,
        persist_graph=False,
        **defaults,
    )
    return loop


def _populate_graph(graph: nx.MultiDiGraph, n_persons: int = 10) -> None:
    """Add n_persons person nodes + 1 hub org node (total n_persons+1 nodes).

    Default of 10 persons + 1 org = 11 nodes exceeds the 10-node floor so
    tests exercise the enrichment path by default.
    """
    for i in range(n_persons):
        name = f"Person{i}"
        graph.add_node(
            name,
            entity_type="person",
            attributes={},
            recurrence_count=i + 1,
            sessions=[f"s{i:03d}"],
            first_seen=f"s{i:03d}",
            last_seen=f"s{i:03d}",
        )
    # Add an org node so we have cross-entity topology
    org = "AcmeCorp"
    graph.add_node(
        org,
        entity_type="organization",
        attributes={},
        recurrence_count=n_persons,
        sessions=["s000"],
        first_seen="s000",
        last_seen="s000",
    )
    # Wire edges: every person works_at AcmeCorp
    for i in range(n_persons):
        graph.add_edge(
            f"Person{i}",
            org,
            predicate="works_at",
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
        triples = _serialize_subgraph_triples(g)
        assert len(triples) == 1
        t = triples[0]
        assert t["subject"] == "Alice"
        assert t["predicate"] == "knows"
        assert t["object"] == "Bob"
        assert t["relation_type"] == "social"

    def test_missing_predicate_defaults(self):
        g = nx.MultiDiGraph()
        g.add_edge("A", "B", relation_type="factual")
        triples = _serialize_subgraph_triples(g)
        assert triples[0]["predicate"] == ""

    def test_missing_relation_type_defaults(self):
        g = nx.MultiDiGraph()
        g.add_edge("A", "B", predicate="likes")
        triples = _serialize_subgraph_triples(g)
        assert triples[0]["relation_type"] == "factual"

    def test_empty_graph(self):
        g = nx.MultiDiGraph()
        assert _serialize_subgraph_triples(g) == []


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

        # Verify the added edge carries source="graph_enrichment"
        found = False
        for _, _, data in graph.out_edges("Person0", data=True):
            if data.get("predicate") == "colleague_of" and data.get("source") == "graph_enrichment":
                found = True
        assert found, "Expected a 'colleague_of' edge with source='graph_enrichment'"


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

        edges_from_p0 = list(graph.out_edges("Person0", data=True))
        predicates = {
            d.get("predicate") for _, _, d in edges_from_p0 if d.get("source") == "graph_enrichment"
        }
        assert "friend_of" in predicates
        assert "colleague_of" not in predicates


class TestSameAsContractsNodes:
    """same_as pairs must remove the variant node and rewire its edges."""

    def test_variant_node_removed(self, tmp_path, monkeypatch):
        loop = _make_loop(tmp_path)
        graph = loop.merger.graph
        _populate_graph(graph, n_persons=10)

        # Add two nodes that should be merged
        graph.add_node(
            "Alice",
            entity_type="person",
            attributes={},
            recurrence_count=3,
            sessions=["s010"],
            first_seen="s010",
            last_seen="s010",
        )
        graph.add_node(
            "Alicia",
            entity_type="person",
            attributes={},
            recurrence_count=1,
            sessions=["s011"],
            first_seen="s011",
            last_seen="s011",
        )
        graph.add_edge(
            "Alicia",
            "AcmeCorp",
            predicate="works_at",
            relation_type="factual",
            confidence=1.0,
            source="extraction",
            sessions=["s011"],
        )

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
        # "Alicia" should be contracted into "Alice" — removed as a distinct node
        assert "Alicia" not in graph.nodes
        assert "Alice" in graph.nodes


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

        graph.add_node(
            "Yang Ming",
            entity_type="person",
            attributes={},
            recurrence_count=3,
            sessions=["s030"],
            first_seen="s030",
            last_seen="s030",
        )
        graph.add_node(
            "Mr. Yang",
            entity_type="person",
            attributes={},
            recurrence_count=2,
            sessions=["s031"],
            first_seen="s031",
            last_seen="s031",
        )

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

        rels = [
            {
                "subject": "Zhang",
                "predicate": "colleague_of",
                "object": "Xiaoxiu",
                "relation_type": "social",
                "confidence": 0.85,
            },
            # Reverse direction — must not create a second edge
            {
                "subject": "Xiaoxiu",
                "predicate": "colleague_of",
                "object": "Zhang",
                "relation_type": "social",
                "confidence": 0.80,
            },
        ]
        for name in ("Zhang", "Xiaoxiu"):
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
        # (Xiaoxiu, colleague_of, Zhang). Second insert is a duplicate.
        enriched = [
            (u, v, d) for u, v, d in graph.edges(data=True) if d.get("source") == "graph_enrichment"
        ]
        colleague_edges = [(u, v) for u, v, d in enriched if d.get("predicate") == "colleague_of"]
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
        for name in ("Ming", "Xinxin"):
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
        mentored_edges = [
            (u, v)
            for u, v, d in graph.edges(data=True)
            if d.get("source") == "graph_enrichment" and d.get("predicate") == "mentored_by"
        ]
        # Both directions survive for asymmetric predicates
        assert set(mentored_edges) == {("Ming", "Xinxin"), ("Xinxin", "Ming")}


class TestCorefRemapBeforeEdgeInsert:
    """Relations referencing a dropped node must land on the canonical node."""

    def test_relation_remapped_through_coref(self, tmp_path, monkeypatch):
        loop = _make_loop(tmp_path)
        graph = loop.merger.graph
        _populate_graph(graph, n_persons=10)

        # Three nodes; "Alex" will be contracted into "Alexander".
        graph.add_node(
            "Alexander",
            entity_type="person",
            attributes={},
            recurrence_count=3,
            sessions=["s050"],
            first_seen="s050",
            last_seen="s050",
        )
        graph.add_node(
            "Alex",
            entity_type="person",
            attributes={},
            recurrence_count=1,
            sessions=["s051"],
            first_seen="s051",
            last_seen="s051",
        )
        graph.add_node(
            "Acme",
            entity_type="organization",
            attributes={},
            recurrence_count=5,
            sessions=["s050"],
            first_seen="s050",
            last_seen="s050",
        )

        # SOTA response: same_as merges Alex→Alexander, and a relation uses
        # the dropped name "Alex". The remap must route it to "Alexander".
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
        assert "Alex" not in graph.nodes  # contracted away
        # The enriched edge must land on "Alexander", not silently disappear.
        alexander_edges = [
            (u, v, d)
            for u, v, d in graph.edges(data=True)
            if u == "Alexander"
            and v == "Acme"
            and d.get("predicate") == "works_at"
            and d.get("source") == "graph_enrichment"
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
            if d.get("source") == "graph_enrichment"
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
