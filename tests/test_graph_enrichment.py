"""Tests for graph-level SOTA enrichment (Task #10).

All tests are pure-Python — no GPU required. SOTA calls are mocked so
the test suite does not make any network requests.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import networkx as nx
import pytest
from peft import PeftModel

from paramem.memory.persistence import _EDGE_SOURCE_ATTR
from paramem.training.consolidation import ConsolidationLoop, serialize_subgraph_triples
from paramem.utils.config import AdapterConfig, ConsolidationConfig, TrainingConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _stub_local_anonymize(monkeypatch):
    """Default stub for ``anonymize_with_local_model``.

    ``_run_graph_enrichment`` now runs the local anonymizer (the SAME
    primitive session-tier extraction uses) over each chunk BEFORE the SOTA
    call, to derive real-name entity types the fold graph itself cannot
    supply (see that method's docstring). ``_make_loop``'s model/tokenizer
    are ``MagicMock()``s, so a real call always fails to parse (there is no
    JSON in a ``MagicMock``'s generated output), which would fail every
    chunk closed (skip the SOTA call entirely) before ``_graph_enrich_with_sota``
    is ever reached — breaking every test below that mocks
    ``_graph_enrich_with_sota`` to verify SOTA-response consumption.

    The default therefore lands on the SAFE side rather than the unsafe
    one: :func:`_stub_local_model_types` with an empty override dict, which
    types every non-speaker name found in the chunk's relations as
    ``"person"`` (masked), so tests below exercise the masked-payload path
    by default. A genuinely EMPTY mapping (``{}``) now PROCEEDS (the
    anonymizer ran and found nothing in scope, a legitimate verdict, not
    a failure) — see
    ``TestEmptyMappingProceeds``. ``_run_graph_enrichment``'s remaining
    fail-closed guard (F1, leg 2) only fires when the local anonymizer DID
    name something but none of it survived reconciliation onto the
    chunk's actual node keys (a genuine classification/identity-match
    failure) — see ``TestPrivacyFailClosedOnReconciliationFailure``. Tests
    in ``TestGraphTierAnonymizationContract`` call
    ``_graph_enrich_with_sota`` directly (never through
    ``_run_graph_enrichment``) and are unaffected by this fixture.
    """
    monkeypatch.setattr(
        "paramem.training.consolidation.anonymize_with_local_model",
        _stub_local_model_types({}),
    )


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
        extraction_scrub={"person name"},
    )
    defaults.update(kwargs)

    # replay_enabled controls whether run_consolidation_cycle's registry guard
    # fires.  Callers that need enrichment hooks to fire pass replay_enabled=True.
    replay_enabled = defaults.pop("replay_enabled", False)
    # Allow callers to supply a pre-built ConsolidationConfig so tests can set
    # fields like refinement_enrichment/sota_enabled without touching other knobs.
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
            reinforcement_count=i + 1,
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
        reinforcement_count=n_persons,
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


class TestEnrichmentInheritsSourceWindow:
    """Enrichment edges must inherit the chunk's source assertion window
    (max last_seen, min non-empty first_seen) rather than landing untimed.
    """

    def test_enrichment_edge_gets_chunk_window(self, tmp_path, monkeypatch):
        loop = _make_loop(tmp_path)
        graph = loop.merger.graph
        _populate_graph(graph, n_persons=10)

        # Stamp the works-at edges with distinct, known timestamps so the
        # chunk's max last_seen / min first_seen are unambiguous.
        for i, (fs, ls) in enumerate(
            [
                ("2026-01-01T00:00:00", "2026-01-05T00:00:00"),
                ("2026-01-02T00:00:00", "2026-01-06T00:00:00"),
                ("2026-01-03T00:00:00", "2026-01-10T00:00:00"),  # max last_seen
            ]
        ):
            for _, _, key, data in graph.out_edges(f"person{i}", keys=True, data=True):
                if data.get("predicate") == "works at":
                    graph[f"person{i}"]["acmecorp"][key]["first_seen"] = fs
                    graph[f"person{i}"]["acmecorp"][key]["last_seen"] = ls

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

        found = None
        for _, _, data in graph.out_edges("person0", data=True):
            if (
                data.get("predicate") == "colleague of"
                and data.get(_EDGE_SOURCE_ATTR) == "graph_enrichment"
            ):
                found = data
        assert found is not None, "Expected a 'colleague of' enrichment edge"
        # The highest-reinforcement focal node (acmecorp's hub connects every
        # person at radius<=2) puts the whole graph in the first chunk, so the
        # window is the min/max across ALL stamped works-at edges: the three
        # stamped person0/1/2 edges contribute first_seen 01/02/03 and
        # last_seen 05/06/10; the rest are unstamped ("") and ignored by
        # min_nonempty / max. This is the fix under test — previously these
        # fields were always "".
        assert found["first_seen"] == "2026-01-01T00:00:00"
        assert found["last_seen"] == "2026-01-10T00:00:00"


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
            reinforcement_count=3,
            sessions=["s010"],
            first_seen="s010",
            last_seen="s010",
        )
        graph.add_node(
            "alicia",
            entity_type="person",
            attributes={"name": "Alicia"},
            reinforcement_count=1,
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
                reinforcement_count=2,
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
            reinforcement_count=3,
            sessions=["s030"],
            first_seen="s030",
            last_seen="s030",
        )
        graph.add_node(
            "mr. yang",
            entity_type="person",
            attributes={"name": "Mr. Yang"},
            reinforcement_count=2,
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
    """Symmetric predicates collapse via Relation.symmetric + merger E-2 swap."""

    def test_both_directions_collapse_to_one_edge(self, tmp_path, monkeypatch):
        """When SOTA emits (A,P,B) and (B,P,A) both with symmetric=true and
        neither endpoint is a speaker, the merger E-2 swap canonicalizes to
        subj < obj and the second insert is a Case-1 duplicate — only one edge lands.
        """
        loop = _make_loop(tmp_path)
        graph = loop.merger.graph
        _populate_graph(graph, n_persons=10)

        # Both directions of colleague_of with symmetric=true.
        # Nodes are canonical-keyed (lowercase); neither carries speaker_id.
        rels = [
            {
                "subject": "Zhang",
                "predicate": "colleague_of",
                "object": "Xiaoxiu",
                "relation_type": "social",
                "confidence": 0.85,
                "symmetric": True,
            },
            {
                "subject": "Xiaoxiu",
                "predicate": "colleague_of",
                "object": "Zhang",
                "relation_type": "social",
                "confidence": 0.80,
                "symmetric": True,
            },
        ]
        for name in ("zhang", "xiaoxiu"):
            graph.add_node(
                name,
                entity_type="person",
                attributes={},
                reinforcement_count=2,
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
        # After merger E-2 swap: both become (xiaoxiu, colleague of, zhang).
        # Second insert is Case-1 reinforce — only one edge with edge_source stamp.
        enriched = [
            (u, v, d)
            for u, v, d in graph.edges(data=True)
            if d.get(_EDGE_SOURCE_ATTR) == "graph_enrichment"
        ]
        colleague_edges = [(u, v) for u, v, d in enriched if d.get("predicate") == "colleague of"]
        assert len(colleague_edges) == 1, (
            f"Expected 1 collapsed symmetric edge; got {colleague_edges}"
        )
        u, v = colleague_edges[0]
        assert u < v, f"Expected canonical lex order (subj < obj); got {u!r} > {v!r}"

    def test_asymmetric_predicates_not_reordered(self, tmp_path, monkeypatch):
        """Asymmetric predicates (symmetric=false or omitted) keep both directions."""
        loop = _make_loop(tmp_path)
        graph = loop.merger.graph
        _populate_graph(graph, n_persons=10)

        # mentored_by with symmetric=false — keep both directions.
        rels = [
            {
                "subject": "Ming",
                "predicate": "mentored_by",
                "object": "Xinxin",
                "relation_type": "social",
                "confidence": 0.85,
                "symmetric": False,
            },
            {
                "subject": "Xinxin",
                "predicate": "mentored_by",
                "object": "Ming",
                "relation_type": "social",
                "confidence": 0.85,
                "symmetric": False,
            },
        ]
        for name in ("ming", "xinxin"):
            graph.add_node(
                name,
                entity_type="person",
                attributes={},
                reinforcement_count=2,
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
        mentored_edges = [
            (u, v)
            for u, v, d in graph.edges(data=True)
            if d.get(_EDGE_SOURCE_ATTR) == "graph_enrichment"
            and d.get("predicate") == "mentored by"
        ]
        assert set(mentored_edges) == {("ming", "xinxin"), ("xinxin", "ming")}, (
            f"Expected both directions for asymmetric predicate; got {mentored_edges}"
        )


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
            reinforcement_count=3,
            sessions=["s050"],
            first_seen="s050",
            last_seen="s050",
        )
        graph.add_node(
            "alex",
            entity_type="person",
            attributes={"name": "Alex"},
            reinforcement_count=1,
            sessions=["s051"],
            first_seen="s051",
            last_seen="s051",
        )
        graph.add_node(
            "acme",
            entity_type="organization",
            attributes={"name": "Acme"},
            reinforcement_count=5,
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
                reinforcement_count=1,
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
    """enrich=False must not call _run_graph_enrichment at all."""

    def test_no_change_when_enrich_false(self, tmp_path, monkeypatch):
        """_refine_consolidation_graph(enrich=False) must not call _run_graph_enrichment
        and must leave the graph byte-identical."""
        loop = _make_loop(
            tmp_path, consolidation_config=ConsolidationConfig(refinement_enrichment="off")
        )
        graph = loop.merger.graph
        _populate_graph(graph, n_persons=10)

        # Snapshot pre-state
        pre_nodes = set(graph.nodes)
        pre_edges = set((u, v) for u, v, _ in graph.edges(data=True))

        with patch.object(loop, "_run_graph_enrichment") as enrich_spy:
            loop._refine_consolidation_graph([], normalize=False, enrich=False)

        enrich_spy.assert_not_called()

        post_nodes = set(graph.nodes)
        post_edges = set((u, v) for u, v, _ in graph.edges(data=True))
        assert pre_nodes == post_nodes
        assert pre_edges == post_edges


class TestPartitionRoutesEnrichedEdges:
    """After enrichment, partition_relations must correctly route new edges."""

    def test_social_edge_routes_to_episodic(self, tmp_path, monkeypatch):
        """Social relation_type → episodic bucket (not procedural)."""
        from paramem.graph.relation_prep import partition_relations

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
            reinforcement_count=25,
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
                reinforcement_count=i + 1,
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


class TestNoModelSkipsGracefully:
    """F3 — self.model is None must early-return, mirroring
    _run_graph_normalization's existing "no_model" guard rather than
    crashing inside _disable_gradient_checkpointing.

    Mutation: remove the ``self.model is None`` guard -> this test raises
    an ``AttributeError``/``TypeError`` instead of returning a clean
    ``skip_reason == "no_model"`` result.
    """

    def test_no_model_skip(self, tmp_path, monkeypatch):
        loop = _make_loop(tmp_path)
        graph = loop.merger.graph
        _populate_graph(graph, n_persons=10)
        loop.model = None

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        call_spy = MagicMock()
        with patch("paramem.training.consolidation._graph_enrich_with_sota", call_spy):
            result = loop._run_graph_enrichment()

        assert result["skipped"] is True
        assert result["skip_reason"] == "no_model"
        call_spy.assert_not_called()


class TestVramExhaustedNotSwallowed:
    """F3 — a VramExhausted mid-chunk must propagate to the caller (the
    per-cycle loop retries), not be swallowed by the broad
    ``except Exception`` as a skipped chunk.

    Mutation: revert to a single ``except Exception`` with no
    ``VramExhausted`` re-raise above it -> this test fails (no exception
    propagates; the chunk is silently skipped instead).
    """

    def test_vram_exhausted_propagates(self, tmp_path, monkeypatch):
        from paramem.server.vram_guard import VramExhausted

        loop = _make_loop(tmp_path)
        graph = loop.merger.graph
        _populate_graph(graph, n_persons=10)

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        def _raise_vram(*args, **kwargs):
            raise VramExhausted("graph_enrichment_test")

        with (
            patch("paramem.training.consolidation._graph_enrich_with_sota", _raise_vram),
            pytest.raises(VramExhausted),
        ):
            loop._run_graph_enrichment()


class TestEmptyMappingProceeds:
    """A local anonymizer mapping that comes back completely EMPTY
    (``{}``) — even for a chunk with real (non-speaker) node names —
    means the anonymizer ran and classified
    NOTHING in scope against ``scrub``. That is a legitimate verdict, not
    a classification failure, so egress PROCEEDS: the SOTA call fires
    with an empty ``chunk_mapping`` (nothing to substitute) rather than
    being skipped. Mirrors the session-tier ``mapping == {}`` proceed
    path in :func:`~paramem.graph.extractor.extract_and_anonymize_for_cloud`.

    Mutation: reintroduce a bare ``not chunk_mapping`` guard (dropping the
    ``_llm_mapping and`` qualifier) in ``_run_graph_enrichment`` -> this
    test fails (the SOTA call is skipped and ``privacy_skipped_chunks``
    increments instead of staying 0).
    """

    def test_empty_mapping_proceeds_with_sota_call(self, tmp_path, monkeypatch):
        loop = _make_loop(tmp_path)
        graph = loop.merger.graph
        _populate_graph(graph, n_persons=10)

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        # Deliberately override the (masked-by-default) autouse fixture
        # with a genuinely empty-mapping stub — this test exists
        # specifically to exercise that proceed path.
        monkeypatch.setattr(
            "paramem.training.consolidation.anonymize_with_local_model",
            lambda *args, **kwargs: ({}, "", "stub-raw"),
        )
        canned_result = ([], [], "raw")
        with patch(
            "paramem.training.consolidation._graph_enrich_with_sota",
            return_value=canned_result,
        ) as call_spy:
            result = loop._run_graph_enrichment()

        assert not result["skipped"]
        assert result["privacy_skipped_chunks"] == 0
        call_spy.assert_called()

    def test_all_speaker_chunk_is_not_falsely_skipped(self, tmp_path, monkeypatch):
        """A chunk whose ONLY real content is speaker-to-speaker relations
        legitimately has an empty (non-speaker) entity mapping — this must
        NOT trip the fail-closed guard, since there was never any
        non-speaker content to classify in the first place."""
        loop = _make_loop(tmp_path)
        graph = loop.merger.graph

        for sid in ("speaker0", "speaker1"):
            graph.add_node(
                sid,
                entity_type="person",
                speaker_id=sid,
                attributes={"name": sid},
                reinforcement_count=1,
                sessions=["s000"],
                first_seen="s000",
                last_seen="s000",
            )
        # Pad past the 10-node floor with disconnected concept nodes so this
        # chunk (built from the speaker-pair ego-graph) stays speaker-only.
        for i in range(9):
            graph.add_node(
                f"filler{i}",
                entity_type="concept",
                attributes={},
                reinforcement_count=0,
                sessions=[],
                first_seen="",
                last_seen="",
            )
        graph.add_edge(
            "speaker0",
            "speaker1",
            predicate="knows",
            relation_type="social",
            confidence=1.0,
            speaker_id="speaker0",
            source="extraction",
            sessions=["s000"],
        )

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setattr(
            "paramem.training.consolidation.anonymize_with_local_model",
            lambda *args, **kwargs: ({}, "", "stub-raw"),
        )
        canned_result = ([], [], "raw")
        with patch(
            "paramem.training.consolidation._graph_enrich_with_sota",
            return_value=canned_result,
        ) as call_spy:
            result = loop._run_graph_enrichment()

        assert not result["skipped"]
        assert result["privacy_skipped_chunks"] == 0
        call_spy.assert_called()


class TestPrivacyFailClosedOnReconciliationFailure:
    """F1 (corrected), leg 2 — a local anonymizer mapping that NAMED
    something but NONE of it survived reconciliation onto the
    chunk's actual node keys, while the chunk has real (non-speaker)
    node names, is a classification/identity-match failure — distinct
    from a genuinely EMPTY mapping (see ``TestEmptyMappingProceeds``,
    which now proceeds). ``canonical()``
    matching (tested directly in ``tests/test_placeholders.py::
    TestSubstituteWholeWordsCanonicalMatching``) cannot fix this leg —
    the named entity simply isn't one of this chunk's nodes.

    Mutation: drop the ``_llm_mapping and`` qualifier from the
    ``_run_graph_enrichment`` guard -> this test's SOTA call would no
    longer be distinguished from the empty-mapping proceed case and
    would fire instead of being skipped.
    """

    def test_reconciliation_drops_everything_skips_sota_call(self, tmp_path, monkeypatch):
        loop = _make_loop(tmp_path)
        graph = loop.merger.graph
        _populate_graph(graph, n_persons=10)

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        # The local anonymizer DID name something — but a name matching
        # none of this chunk's actual node keys, so reconciliation
        # drops it, leaving chunk_mapping empty despite a non-empty
        # _llm_mapping.
        monkeypatch.setattr(
            "paramem.training.consolidation.anonymize_with_local_model",
            lambda *args, **kwargs: (
                {"nobody in this chunk": "Person_1"},
                "stub-anon",
                "stub-raw",
            ),
        )
        call_spy = MagicMock()
        with patch("paramem.training.consolidation._graph_enrich_with_sota", call_spy):
            result = loop._run_graph_enrichment()

        assert not result["skipped"]
        assert result["privacy_skipped_chunks"] >= 1
        assert result["new_edges"] == 0
        assert result["same_as_merges"] == 0
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
                mapping={},
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
                mapping={},
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
                mapping={},
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
                mapping={},
                api_key="test-key",
                provider="anthropic",
                filter_model="claude-sonnet-4-6",
            )

        assert result is not None
        _, same_as, _ = result
        # Only the valid [Alice, Alicia] entry survives; ["bad", [1,2]] are skipped
        assert same_as == [["Alice", "Alicia"]]


class TestInterimEnrichmentHook:
    """Rollover mini-enrichment inside run_consolidation_cycle.

    The rollover hook fires inside the 'normal fresh-interim' branch of
    run_consolidation_cycle, i.e. when a new sub-interval stamp opens.  It
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
                    speaker_id="speaker0",
                ),
                Relation(
                    subject="B",
                    predicate="knows",
                    object="A",
                    relation_type="social",
                    speaker_id="speaker0",
                ),
            ],
        )

    def test_refinement_enrichment_on_calls_run_graph_enrichment(self, tmp_path):
        """refinement_enrichment='on' + sota_enabled=True → _run_graph_enrichment called."""
        from paramem.training.key_registry import KeyRegistry
        from paramem.utils.config import ConsolidationConfig

        loop = _make_loop(
            tmp_path,
            consolidation_config=ConsolidationConfig(refinement_enrichment="on", sota_enabled=True),
            replay_enabled=True,
        )
        for tier in ("episodic", "semantic", "procedural"):
            loop.store.load_registry(tier, KeyRegistry())

        _ep = [
            {
                "question": "q",
                "answer": "a",
                "subject": "S",
                "predicate": "p",
                "object": "O",
            }
        ]
        loop.extract_session = MagicMock(return_value=(_ep, []))
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
            eps, proc = loop.extract_session("t", "s1", "speaker0")
            loop.run_consolidation_cycle(
                eps,
                proc,
                speaker_id="speaker0",
                mode="train",
                run_label="s1",
                schedule="12h",
                max_interim_count=7,
                stamp="20260420T1200",
            )

        loop._run_graph_enrichment.assert_called_once()

    def test_refinement_normalization_only_does_not_enrich(self, tmp_path):
        """refinement_normalization='on' only → _run_graph_enrichment NOT called by refine."""
        from paramem.training.key_registry import KeyRegistry
        from paramem.utils.config import ConsolidationConfig

        loop = _make_loop(
            tmp_path,
            consolidation_config=ConsolidationConfig(
                refinement_normalization="on", refinement_enrichment="off"
            ),
            replay_enabled=True,
        )
        for tier in ("episodic", "semantic", "procedural"):
            loop.store.load_registry(tier, KeyRegistry())

        _ep = [
            {
                "question": "q",
                "answer": "a",
                "subject": "S",
                "predicate": "p",
                "object": "O",
            }
        ]
        loop.extract_session = MagicMock(return_value=(_ep, []))
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
            eps, proc = loop.extract_session("t", "s1", "speaker0")
            loop.run_consolidation_cycle(
                eps,
                proc,
                speaker_id="speaker0",
                mode="train",
                run_label="s1",
                schedule="12h",
                max_interim_count=7,
                stamp="20260420T1200",
            )

        loop._run_graph_enrichment.assert_not_called()

    def test_refinement_off_does_not_enrich(self, tmp_path):
        """refinement_enrichment='off' (default) → _run_graph_enrichment NOT called by refine."""
        from paramem.training.key_registry import KeyRegistry

        loop = _make_loop(
            tmp_path,
            replay_enabled=True,
        )
        # refinement_enrichment defaults to "off" in ConsolidationConfig default.
        for tier in ("episodic", "semantic", "procedural"):
            loop.store.load_registry(tier, KeyRegistry())

        _ep = [
            {
                "question": "q",
                "answer": "a",
                "subject": "S",
                "predicate": "p",
                "object": "O",
            }
        ]
        loop.extract_session = MagicMock(return_value=(_ep, []))
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
            eps, proc = loop.extract_session("t", "s1", "speaker0")
            loop.run_consolidation_cycle(
                eps,
                proc,
                speaker_id="speaker0",
                mode="train",
                run_label="s1",
                schedule="12h",
                max_interim_count=7,
                stamp="20260420T1200",
            )

        loop._run_graph_enrichment.assert_not_called()

    def test_rollover_hook_skipped_on_ring_full(self, tmp_path):
        """Ring-full (cap_pending) short-circuit does NOT fire the enrichment hook.

        When the interim ring is at max_interim_count and the target slot is new
        (train mode), run_consolidation_cycle returns mode="cap_pending" before
        any graph extraction or enrichment occurs.  The rollover hook is bound
        to the normal-branch pipeline, not the cap_pending early-return.
        """
        # replay_enabled=True is required so the "no registry" guard passes
        # and execution reaches the ring-full detection.
        loop = _make_loop(tmp_path, replay_enabled=True)

        _ep = [
            {
                "question": "q",
                "answer": "a",
                "subject": "S",
                "predicate": "p",
                "object": "O",
            }
        ]
        loop.extract_session = MagicMock(return_value=(_ep, []))
        loop._run_graph_enrichment = MagicMock(return_value={"skipped": False})

        existing_stamp = "20260419T1200"
        current_stamp = "20260420T1200"
        existing_name = f"episodic_interim_{existing_stamp}"
        # Pre-fill the ring to max_interim_count=1 with a different stamp so
        # the target slot (current_stamp) is new and ring_full fires.
        loop.model.peft_config = {
            "episodic": MagicMock(),
            "semantic": MagicMock(),
            existing_name: MagicMock(),
        }

        eps, proc = loop.extract_session("t", "s1", "speaker0")
        result = loop.run_consolidation_cycle(
            eps,
            proc,
            speaker_id="speaker0",
            mode="train",
            run_label="s1",
            schedule="12h",
            max_interim_count=1,
            stamp=current_stamp,
        )

        assert result["mode"] == "cap_pending"
        loop._run_graph_enrichment.assert_not_called()


# ---------------------------------------------------------------------------
# Tests for _refine_consolidation_graph (enrich gate + recurrence-bump)
# ---------------------------------------------------------------------------


class TestRefineConsolidationGraph:
    """Unit tests for _refine_consolidation_graph's enrich param.

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

    def test_default_skips_both_normalize_and_enrich(self, tmp_path):
        """normalize and enrich both default False (level "off") — refine calls neither."""
        loop = _make_loop(tmp_path)
        loop._run_graph_enrichment = MagicMock(return_value={"skipped": True})
        loop._run_graph_normalization = MagicMock(return_value={"skipped": True})

        # No kwargs — level "off" semantics (both default False).
        loop._refine_consolidation_graph([])

        loop._run_graph_enrichment.assert_not_called()
        loop._run_graph_normalization.assert_not_called()

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
            relation_type="factual",
            reinforcement_count=1,
            last_reinforced_cycle=0,
            allow_empty_speaker=True,
            first_seen="",
        )

        # Simulate a Case-1 collision: merger.reinforcements contains the surviving
        # key mapped to (last_seen, first_seen).
        loop.merger.reinforcements = {"graph42": ("2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z")}

        from paramem.graph.schema import Relation

        recon_rel = Relation(
            subject="A", predicate="p", object="B", relation_type="factual", speaker_id=""
        )
        loop._run_graph_enrichment = MagicMock(return_value={"skipped": True})

        loop._refine_consolidation_graph([recon_rel], enrich=False)

        loop._run_graph_enrichment.assert_not_called()
        bk = loop.store.bookkeeping_for_key("graph42")
        assert bk is not None
        assert bk["reinforcement_count"] == 2, (
            f"Recurrence should have been bumped to 2; got {bk['reinforcement_count']}"
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
            relation_type="factual",
            reinforcement_count=3,
            last_reinforced_cycle=0,
            allow_empty_speaker=True,
            first_seen="",
        )
        loop.merger.reinforcements = {"graph7": ("2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z")}

        from paramem.graph.schema import Relation

        recon_rel = Relation(
            subject="X", predicate="q", object="Y", relation_type="factual", speaker_id=""
        )
        loop._run_graph_enrichment = MagicMock(return_value={"skipped": False})

        loop._refine_consolidation_graph([recon_rel], enrich=True)

        loop._run_graph_enrichment.assert_called_once()
        bk = loop.store.bookkeeping_for_key("graph7")
        assert bk is not None
        assert bk["reinforcement_count"] == 4, (
            f"Recurrence should have been bumped to 4; got {bk['reinforcement_count']}"
        )

    def test_empty_recon_relations_is_safe_noop_for_bump(self, tmp_path):
        """Empty recon_relations → recurrence-bump loop does not run; no crash."""
        loop = _make_loop(tmp_path)
        # graph99 would be bumped if the empty-recon guard failed.
        loop.merger.reinforcements = {"graph99": ("2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z")}
        loop._run_graph_enrichment = MagicMock(return_value={"skipped": True})

        bump_spy = MagicMock()
        loop.store.bump_recurrence = bump_spy

        # Both enrich values must be safe no-ops when recon_relations is empty.
        loop._refine_consolidation_graph([], enrich=False)
        loop._refine_consolidation_graph([], enrich=True)

        bump_spy.assert_not_called()

    def test_normalize_true_calls_run_graph_normalization(self, tmp_path):
        """normalize=True runs the whole-graph normalization pass (light+, both scopes).

        The normalization pass is independent of enrich: normalize=True without
        enrich runs normalization only (the light default), not SOTA enrichment.
        """
        loop = _make_loop(tmp_path)
        loop._run_graph_normalization = MagicMock(return_value={"skipped": True})
        loop._run_graph_enrichment = MagicMock(return_value={"skipped": True})

        loop._refine_consolidation_graph([], normalize=True)

        loop._run_graph_normalization.assert_called_once()
        loop._run_graph_enrichment.assert_not_called()


# ---------------------------------------------------------------------------
# Tests for _build_all_edge_entries_into (unified edge→entry builder)
# ---------------------------------------------------------------------------


class TestHarvestKeylessEdges:
    """Unit tests for the unified edge→entry builder (_build_all_edge_entries_into).

    Uses _make_loop from this module (real nx.MultiDiGraph + real MemoryStore,
    mocked model/tokenizer so no GPU).  replay_enabled=True so store.put()
    writes into the KeyRegistry.

    These tests exercise the keyless-edge (minting) branch of the builder by
    populating the graph with only keyless predicate-bearing edges.
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
        loop._build_all_edge_entries_into(tier_keyed)

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
        assert bk["reinforcement_count"] == 1
        assert bk["last_reinforced_cycle"] == loop.cycle_count
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
        loop._build_all_edge_entries_into(tier_keyed)

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
        loop._build_all_edge_entries_into(tier_keyed)

        # The keyed edge has no store entry, so it is skipped — nothing in tier_keyed.
        # No new key is minted (_indexed_next_index unchanged).
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
        loop._build_all_edge_entries_into(tier_keyed)

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
        loop._build_all_edge_entries_into(tier_keyed)

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
        loop._build_all_edge_entries_into(tier_keyed)

        # preference → episodic (no procedural adapter in _make_loop).
        assert len(tier_keyed["episodic"]) == 1
        key = tier_keyed["episodic"][0]["key"]
        bk = loop.store.bookkeeping_for_key(key)
        assert bk is not None
        assert bk["relation_type"] == "preference"

    def test_procedural_tier_minting(self, tmp_path):
        """Keyless preference edge routes to procedural when procedural_config is set.

        _make_loop passes procedural_adapter_config=None so the procedural
        branch of _build_all_edge_entries_into never fires in the other tests.
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
            extraction_scrub={"person name"},
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
        loop._build_all_edge_entries_into(tier_keyed)

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
            extraction_scrub={"person name"},
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
        loop._build_all_edge_entries_into(tier_keyed)

        assert len(tier_keyed["episodic"]) == 1
        minted_key = tier_keyed["episodic"][0]["key"]

        # Must be graph6 — no collision with the pre-existing graph5.
        assert minted_key == "graph6", (
            f"Expected minted key 'graph6' to avoid collision with graph5, got {minted_key!r}"
        )
        # Pre-existing graph5 entry must still be intact.
        assert "graph5" in loop.store.all_active_keys()


class TestHarvestApplySplit:
    """Tests verifying the unified edge→entry builder (_build_all_edge_entries_into).

    Covers the defer=True (interim atomicity) and defer=False (fold discipline)
    paths, plus the minted_by_tier / deferred_writes return contract.
    """

    def test_defer_false_produces_writes_and_count(self, tmp_path):
        """defer=False (default) must write to the store, advance counters, and
        return (minted_by_tier, []) — no deferred writes.
        """
        from paramem.training.key_registry import KeyRegistry

        loop = _make_loop(tmp_path, replay_enabled=True)
        for tier in ("episodic", "semantic", "procedural"):
            loop.store.load_registry(tier, KeyRegistry())

        loop.merger.graph.add_edge("Carol", "London", predicate="visited", relation_type="factual")

        initial_indexed = loop._indexed_next_index

        tier_keyed: dict = {"episodic": [], "semantic": [], "procedural": []}
        minted_by_tier, _deferred = loop._build_all_edge_entries_into(tier_keyed)

        # Exactly one episodic key minted.  defer=False → no deferred writes.
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
        assert bk["reinforcement_count"] == 1
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

        tier_keyed: dict = {"episodic": [], "semantic": [], "procedural": []}
        loop._build_all_edge_entries_into(tier_keyed)

        assert len(tier_keyed["episodic"]) == 2
        assert loop._indexed_next_index == initial_indexed + 2

        # Keys must be exactly the two sequential indices.
        minted_keys = {e["key"] for e in tier_keyed["episodic"]}
        assert f"graph{initial_indexed}" in minted_keys
        assert f"graph{initial_indexed + 1}" in minted_keys

    def test_no_keyless_edges_does_not_read_counters(self, tmp_path):
        """With no keyless edges to mint, the builder must not read the index
        counters at all (lazy-seed contract).

        Guards the lazy-seed contract: callers that exercise the keyed-edge walk
        without any keyless edges (e.g. a graph of only predicate-less edges) need
        not have the index counters initialised.
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

        tier_keyed: dict = {"episodic": [], "semantic": [], "procedural": []}
        minted_by_tier, deferred = loop._build_all_edge_entries_into(tier_keyed)
        assert minted_by_tier == {"episodic": 0, "procedural": 0}
        assert deferred == []
        assert tier_keyed == {"episodic": [], "semantic": [], "procedural": []}


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
            reinforcement_count=3,
            sessions=["s010"],
            first_seen="s010",
            last_seen="s010",
        )
        graph.add_node(
            "alicia",
            entity_type="person",
            attributes={"name": "Alicia"},
            reinforcement_count=1,
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
            reinforcement_count=1,
            sessions=["s020"],
            first_seen="s020",
            last_seen="s020",
        )
        graph.add_node(
            "baddrop",
            entity_type="person",
            attributes={},
            reinforcement_count=1,
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
# Interim keying seams
# ---------------------------------------------------------------------------


class TestInterimKeyingSeams:
    """Graph-walk keying for the interim path (_build_all_edge_entries_into).

    Covers:
    - speaker_id resolution order: read from edge first (the merger's edge stamp),
      then subject node attr, then "" terminal fallback (no default_speaker_id param).
    - Merger-routed relations carry edge speaker_id through to bookkeeping.
    - Concept node with no speaker yields "" (allow-empty path).
    - defer=True performs NO store writes and NO counter advances.
    - defer=True returns the deferred_writes list for later flush.
    - tag_new=True stamps minted entries with the _new sentinel.
    """

    def test_speaker_id_from_node_attr_carried_through(self, tmp_path):
        """C-1 invariant: speaker-attributed node → speaker_id carried through
        edge (A-1 merger stamp) → deferred_writes record carries the correct value.

        Route a speaker-attributed Relation through the merger so the EDGE carries
        speaker_id (A-1 Case-3 stamp).  The graph-walk reads edge → node → "" and
        produces the correct speaker_id in the minted entry.
        """
        from paramem.graph.schema import Relation, SessionGraph
        from paramem.training.key_registry import KeyRegistry

        loop = _make_loop(tmp_path, replay_enabled=True)
        for tier in ("episodic", "semantic", "procedural"):
            loop.store.load_registry(tier, KeyRegistry())

        # Route a speaker Relation through the real merger path — this stamps
        # speaker_id on the edge via A-1 (Case-3 net-new insert).
        alice_rel = Relation(
            subject="speaker0",
            predicate="lives_in",
            object="Berlin",
            relation_type="factual",
            speaker_id="speaker0",
        )
        from paramem.graph.schema import Entity

        session = SessionGraph(
            session_id="s001",
            timestamp="2026-01-01T00:00:00+00:00",
            entities=[Entity(name="speaker0", entity_type="person", speaker_id="speaker0")],
            relations=[alice_rel],
        )
        loop.merger.merge(session)

        # Non-speaker Relation: concept node → no edge speaker_id, no node speaker_id.
        bob_rel = Relation(
            subject="concept_a",
            predicate="likes",
            object="coffee",
            relation_type="factual",
            speaker_id="",
        )
        session2 = SessionGraph(
            session_id="s002",
            timestamp="2026-01-01T00:00:00+00:00",
            entities=[],
            relations=[bob_rel],
        )
        loop.merger.merge(session2)

        tier_keyed: dict = {"episodic": [], "semantic": [], "procedural": []}
        _, deferred_writes = loop._build_all_edge_entries_into(tier_keyed, defer=True, tag_new=True)

        assert len(deferred_writes) == 2, f"Expected 2 deferred writes; got {deferred_writes}"

        # Find speaker-attributed record by speaker_id.
        speaker_rec = next((r for r in deferred_writes if r["speaker_id"] == "speaker0"), None)
        assert speaker_rec is not None, "No deferred record with speaker_id='speaker0'"

        # Concept-node record must carry "".
        concept_rec = next((r for r in deferred_writes if r["speaker_id"] == ""), None)
        assert concept_rec is not None, "No deferred record with speaker_id=''"

        # tier_keyed entries carry the same speaker_ids — uniform entry shape.
        speaker_entry = next(
            (e for e in tier_keyed["episodic"] if e["speaker_id"] == "speaker0"), None
        )
        assert speaker_entry is not None, "No tier_keyed entry for speaker0"

    def test_explicit_empty_speaker_id_not_overwritten_by_default(self, tmp_path):
        """C-1 invariant: concept node with no speaker_id attr → "" terminal fallback.
        The new edge→node→"" read has no default_speaker_id override.
        """
        from paramem.training.key_registry import KeyRegistry

        loop = _make_loop(tmp_path, replay_enabled=True)
        for tier in ("episodic", "semantic", "procedural"):
            loop.store.load_registry(tier, KeyRegistry())

        # Concept node — no speaker_id attr, edge carries no speaker_id.
        loop.merger.graph.add_node("concept_b", attributes={"name": "ConceptB"})
        loop.merger.graph.add_node("london", attributes={"name": "London"})
        loop.merger.graph.add_edge(
            "concept_b", "london", predicate="visits", relation_type="factual"
        )

        tier_keyed: dict = {"episodic": [], "semantic": [], "procedural": []}
        _, deferred_writes = loop._build_all_edge_entries_into(tier_keyed, defer=True, tag_new=True)

        assert len(deferred_writes) == 1, f"Expected 1 deferred write; got {deferred_writes}"
        rec = deferred_writes[0]

        # Terminal fallback is "" — no default override.
        assert rec["speaker_id"] == "", f"Expected speaker_id=''; got {rec['speaker_id']!r}"
        # tier_keyed entry also carries "" — uniform entry shape.
        assert tier_keyed["episodic"][0]["speaker_id"] == ""

    def test_defer_true_no_store_writes_no_counter_advance(self, tmp_path):
        """_build_all_edge_entries_into(defer=True) must NOT write to the store and
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

        with (
            patch.object(loop.store, "put", wraps=loop.store.put) as mock_put,
            patch.object(
                loop.store, "set_bookkeeping", wraps=loop.store.set_bookkeeping
            ) as mock_bk,
        ):
            tier_keyed: dict = {"episodic": [], "semantic": [], "procedural": []}
            minted_by_tier, deferred_writes = loop._build_all_edge_entries_into(
                tier_keyed, defer=True, tag_new=True
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
        """deferred_writes records from _build_all_edge_entries_into(defer=True) must
        carry all fields required for the caller's flush (entry, canon_subj, canon_obj,
        predicate, tier, speaker_id, relation_type).
        """
        from paramem.training.key_registry import KeyRegistry

        loop = _make_loop(tmp_path, replay_enabled=True)
        for tier in ("episodic", "semantic", "procedural"):
            loop.store.load_registry(tier, KeyRegistry())

        loop.merger.graph.add_node("frank", speaker_id="speaker1", attributes={"name": "Frank"})
        loop.merger.graph.add_node("hamburg", attributes={"name": "Hamburg"})
        loop.merger.graph.add_edge(
            "frank", "hamburg", predicate="works_in", relation_type="factual"
        )

        tier_keyed: dict = {"episodic": [], "semantic": [], "procedural": []}
        _, deferred_writes = loop._build_all_edge_entries_into(tier_keyed, defer=True, tag_new=True)

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
        assert rec["speaker_id"] == "speaker1"
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
        tier_keyed: dict = {"episodic": [], "semantic": [], "procedural": []}
        _, deferred_writes = loop._build_all_edge_entries_into(tier_keyed, defer=True, tag_new=True)
        assert len(deferred_writes) == 1
        entry_new = deferred_writes[0]["entry"]
        assert entry_new.get("_new") is True, (
            f"tag_new=True must set '_new'=True on the entry; got {entry_new!r}"
        )

        # tag_new=False — minted entries do NOT get the sentinel.
        loop._indexed_next_index = 1  # reset so next call gets a fresh index
        loop.merger.graph.clear_edges()
        loop.merger.graph.add_edge("Hank", "Rome", predicate="visits", relation_type="factual")

        tier_keyed2: dict = {"episodic": [], "semantic": [], "procedural": []}
        _, deferred_writes2 = loop._build_all_edge_entries_into(
            tier_keyed2, defer=True, tag_new=False
        )
        assert len(deferred_writes2) == 1
        entry_nosentinel = deferred_writes2[0]["entry"]
        assert "_new" not in entry_nosentinel or entry_nosentinel.get("_new") is not True, (
            f"tag_new=False must NOT set '_new'=True; got {entry_nosentinel!r}"
        )


# ---------------------------------------------------------------------------
# Symmetric session-tier names deleted — importability guard
# ---------------------------------------------------------------------------


class TestSymmetricSessionTierNamesDeleted:
    """E-3: SYMMETRIC_PREDICATES and _canonicalize_symmetric_predicates deleted."""

    def test_symmetric_predicates_not_importable(self):
        """SYMMETRIC_PREDICATES must not be importable from extractor."""
        import importlib

        extractor = importlib.import_module("paramem.graph.extractor")
        assert not hasattr(extractor, "SYMMETRIC_PREDICATES"), (
            "SYMMETRIC_PREDICATES must be deleted from extractor — it is no longer used"
        )

    def test_canonicalize_symmetric_not_importable(self):
        """_canonicalize_symmetric_predicates must not be importable from extractor."""
        import importlib

        extractor = importlib.import_module("paramem.graph.extractor")
        assert not hasattr(extractor, "_canonicalize_symmetric_predicates"), (
            "_canonicalize_symmetric_predicates must be deleted from extractor"
        )


# ---------------------------------------------------------------------------
# Enrichment-through-merger composition test
# ---------------------------------------------------------------------------


class TestEnrichmentThroughMergerComposition:
    """Enrichment routes through _merge_registry_relations (B workstream).

    The old direct graph.add_edge path is deleted. Enrichment edges now go
    through the merger, which:
    - stamps _EDGE_SOURCE_ATTR="graph_enrichment" (B-4 Case-3),
    - stamps speaker_id from Relation.speaker_id (A-1),
    - deduplicates via Case-1 when an extraction edge already exists.
    """

    def test_enrichment_edge_carries_edge_source(self, tmp_path, monkeypatch):
        """Enrichment edge lands via Case-3 and carries edge_source='graph_enrichment'."""
        from paramem.training.key_registry import KeyRegistry

        loop = _make_loop(tmp_path, replay_enabled=True)
        _populate_graph(loop.merger.graph, n_persons=10)
        for tier in ("episodic", "semantic", "procedural"):
            loop.store.load_registry(tier, KeyRegistry())

        # Two concept nodes for the enrichment relation.
        for name in ("alpha", "beta"):
            loop.merger.graph.add_node(
                name,
                entity_type="person",
                attributes={"name": name.capitalize()},
                reinforcement_count=2,
                sessions=["s099"],
                first_seen="s099",
                last_seen="s099",
            )

        rels = [
            {
                "subject": "alpha",
                "predicate": "colleague_of",
                "object": "beta",
                "relation_type": "social",
                "confidence": 0.85,
                "symmetric": False,
            }
        ]
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with patch(
            "paramem.training.consolidation._graph_enrich_with_sota",
            return_value=(rels, [], "raw"),
        ):
            result = loop._run_graph_enrichment()

        assert not result["skipped"]
        # Edge must carry edge_source="graph_enrichment" (B-4).
        enriched = [
            d
            for _, _, d in loop.merger.graph.edges(data=True)
            if d.get(_EDGE_SOURCE_ATTR) == "graph_enrichment"
        ]
        assert len(enriched) >= 1, f"Expected ≥1 enrichment-stamped edge; got {len(enriched)}"

    def test_enrichment_duplicating_extraction_edge_takes_case1(self, tmp_path, monkeypatch):
        """Enrichment relation whose SPO matches an existing extraction edge
        triggers Case-1 (recurrence bump), not a silent skip or a new edge."""
        from paramem.training.key_registry import KeyRegistry

        loop = _make_loop(tmp_path, replay_enabled=True)
        _populate_graph(loop.merger.graph, n_persons=10)
        for tier in ("episodic", "semantic", "procedural"):
            loop.store.load_registry(tier, KeyRegistry())

        # Pre-insert an extraction edge for alpha→beta colleague_of.
        for name in ("alpha", "beta"):
            loop.merger.graph.add_node(
                name,
                entity_type="person",
                attributes={"name": name.capitalize()},
                reinforcement_count=1,
                sessions=["s001"],
                first_seen="s001",
                last_seen="s001",
            )
        loop.merger.graph.add_edge(
            "alpha",
            "beta",
            predicate="colleague of",
            relation_type="social",
            confidence=0.9,
            first_seen="s001",
            last_seen="s001",
            reinforcement_count=1,
            sessions=["s001"],
        )
        edges_before = loop.merger.graph.number_of_edges()

        # Enrichment emits the same relation — must merge via Case-1, not add new edge.
        rels = [
            {
                "subject": "alpha",
                "predicate": "colleague_of",
                "object": "beta",
                "relation_type": "social",
                "confidence": 0.85,
                "symmetric": False,
            }
        ]
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with patch(
            "paramem.training.consolidation._graph_enrich_with_sota",
            return_value=(rels, [], "raw"),
        ):
            result = loop._run_graph_enrichment()

        assert not result["skipped"]
        # Edge count must not increase — Case-1 absorbed it.
        edges_after = loop.merger.graph.number_of_edges()
        assert edges_after == edges_before, (
            f"Case-1 must absorb enrichment dup; edges {edges_before} → {edges_after}"
        )
        # result['new_edges'] must be 0 (delta is 0).
        assert result["new_edges"] == 0, f"new_edges must be 0 for a dup; got {result['new_edges']}"


# ---------------------------------------------------------------------------
# Deferred-flush allow-empty coverage
# ---------------------------------------------------------------------------


class TestDeferredFlushAllowEmpty:
    """D-2 #2/#3: deferred-flush set_bookkeeping sites pass allow_empty_speaker."""

    def test_concept_edge_deferred_flush_allows_empty_speaker(self, tmp_path):
        """Concept-rooted keyless edge with no speaker_id flushes without ValueError
        when the deferred write site uses allow_empty_speaker=(rec['speaker_id']==" ").
        """
        from paramem.training.key_registry import KeyRegistry

        loop = _make_loop(tmp_path, replay_enabled=True)
        for tier in ("episodic", "semantic", "procedural"):
            loop.store.load_registry(tier, KeyRegistry())

        # Concept edge — no speaker attribution.
        loop.merger.graph.add_node("idea_x", attributes={"name": "IdeaX"})
        loop.merger.graph.add_node("idea_y", attributes={"name": "IdeaY"})
        loop.merger.graph.add_edge(
            "idea_x", "idea_y", predicate="related_to", relation_type="factual"
        )

        tier_keyed: dict = {"episodic": [], "semantic": [], "procedural": []}
        _, deferred_writes = loop._build_all_edge_entries_into(tier_keyed, defer=True, tag_new=True)
        assert len(deferred_writes) == 1
        rec = deferred_writes[0]
        assert rec["speaker_id"] == ""

        # Manually flush (simulates the simulate/weights deferred flush).
        # This must not raise ValueError.
        from paramem.memory.entry import compute_simhash

        entry = rec["entry"]
        key = entry["key"]
        loop.store.put(
            "episodic",
            key,
            entry,
            simhash=compute_simhash(key, "idea_x", rec["predicate"], "idea_y"),
        )
        loop.store.set_bookkeeping(
            key,
            speaker_id=rec["speaker_id"],
            relation_type=rec["relation_type"],
            reinforcement_count=1,
            last_reinforced_cycle=0,
            allow_empty_speaker=(rec["speaker_id"] == ""),
            first_seen="",
        )
        bk = loop.store.bookkeeping_for_key(key)
        assert bk is not None
        assert bk["speaker_id"] == ""


# ---------------------------------------------------------------------------
# Verbatim-speaker-key resolution in the enrichment path
# ---------------------------------------------------------------------------


def _seed_speaker_node(loop, speaker_id: str, display: str) -> None:
    """Seed a real speaker node via the merger (§0 invariant: node key is
    the lowercase speaker_id, e.g. ``"speaker0"`` — same as entity.speaker_id
    verbatim under the lowercase-uniform speaker-identity design).

    Uses the real merger.merge path — no raw add_node shortcut — so the test
    exercises the same node-key convention as production.
    """
    from paramem.graph.schema import Entity, SessionGraph

    loop.merger.merge(
        SessionGraph(
            session_id=f"seed-{speaker_id}",
            timestamp="2026-01-01T00:00:00Z",
            entities=[Entity(name=display, entity_type="person", speaker_id=speaker_id)],
            relations=[],
        )
    )


class TestEnrichmentVerbatimSpeakerKeyResolution:
    """Regression: SOTA echoes back the cased speaker id ('speaker0'), which
    must resolve to the existing casefolded speaker node key ('speaker0') via
    canonical fallback.  No duplicate node is created.

    Invariant (Step 2 of resolve_to_node_key): speaker node keys are the casefolded form of the
    speaker_id.  resolve_to_node_key("speaker0", in_graph) → canonical("speaker0")
    = "speaker0" (since "speaker0" is not in the graph — the key is "speaker0").
    """

    def test_speaker_subject_resolves_to_verbatim_node_no_duplicate(self, tmp_path, monkeypatch):
        """An enrichment relation whose subject is the lowercase speaker id 'speaker0'
        resolves to the existing canonical speaker node without creating a second node.
        The minted edge inherits speaker_id='speaker0' from the node attribute, NOT ''.
        """
        from paramem.training.key_registry import KeyRegistry

        loop = _make_loop(tmp_path, replay_enabled=True)
        _populate_graph(loop.merger.graph, n_persons=10)
        for tier in ("episodic", "semantic", "procedural"):
            loop.store.load_registry(tier, KeyRegistry())

        # _seed_speaker_node creates the speaker node via the real merger, keyed
        # by the canonical speaker_id (§0): "speaker0".
        _seed_speaker_node(loop, "speaker0", "Alex")
        # A concept node the speaker relates to.
        loop.merger.graph.add_node("mentoring", attributes={"name": "Mentoring"})

        # Confirm §0 key convention: canonical lowercase key only.
        assert "speaker0" in loop.merger.graph.nodes

        # SOTA emits lowercase speaker id "speaker0" as the subject.
        rels = [
            {
                "subject": "speaker0",
                "predicate": "interested_in",
                "object": "mentoring",
                "relation_type": "preference",
                "confidence": 0.9,
                "symmetric": False,
            }
        ]
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with patch(
            "paramem.training.consolidation._graph_enrich_with_sota",
            return_value=(rels, [], "raw"),
        ):
            result = loop._run_graph_enrichment()

        assert not result["skipped"]
        # Still exactly one speaker node — no duplicate created.
        assert "speaker0" in loop.merger.graph.nodes
        # The speaker node carries its speaker_id.  _synth_speaker_entities emits
        # Entity(name="speaker0", speaker_id="speaker0") which refreshes
        # attributes["name"] to "speaker0" (the canonical speaker_id).
        node = loop.merger.graph.nodes["speaker0"]
        assert node.get("speaker_id") == "speaker0", (
            f"Speaker node must carry speaker_id='speaker0'; got {node.get('speaker_id')!r}"
        )
        assert node["attributes"].get("name") == "speaker0", (
            f"Speaker node attributes['name'] must be 'speaker0' after enrichment; "
            f"got {node['attributes'].get('name')!r}"
        )
        # The enrichment edge roots at the canonical speaker node and carries
        # speaker_id="speaker0" (from the node's attribute).
        enriched = [
            (u, v, d)
            for u, v, d in loop.merger.graph.edges(data=True)
            if d.get(_EDGE_SOURCE_ATTR) == "graph_enrichment"
        ]
        assert len(enriched) == 1, f"Expected 1 enrichment edge; got {enriched}"
        u, _v, d = enriched[0]
        assert u == "speaker0", f"Enrichment edge subject must be 'speaker0'; got {u!r}"
        assert d.get("speaker_id") == "speaker0", (
            f"Edge speaker_id must be 'speaker0' (from the node attribute); "
            f"got {d.get('speaker_id')!r}"
        )
        # The minted training subject reads attributes["name"] = "speaker0".
        tier_keyed: dict = {"episodic": [], "semantic": [], "procedural": []}
        loop._build_all_edge_entries_into(tier_keyed)
        enrichment_entries = [
            e
            for e in tier_keyed["episodic"] + tier_keyed["semantic"] + tier_keyed["procedural"]
            if e.get("predicate") == "interested in"
        ]
        assert len(enrichment_entries) == 1, (
            f"Expected 1 minted entry for 'interested in'; got {enrichment_entries}"
        )
        minted_subject = enrichment_entries[0]["subject"]
        assert minted_subject == "speaker0", (
            f"Minted indexed-key training subject must be 'speaker0'; got {minted_subject!r}"
        )

    def test_speaker_to_speaker_two_keys_distinct_speakers_router_filed(
        self, tmp_path, monkeypatch
    ):
        """Speaker↔speaker: enrichment emits BOTH directions of colleague_of
        with symmetric=true; both endpoints are lowercase speaker ids → resolved to
        canonical node keys → two directed keys mint with distinct speaker_ids, each
        filed under its own speaker in the router index."""
        from paramem.server.router import QueryRouter
        from paramem.training.key_registry import KeyRegistry

        loop = _make_loop(tmp_path, replay_enabled=True)
        _populate_graph(loop.merger.graph, n_persons=10)
        for tier in ("episodic", "semantic", "procedural"):
            loop.store.load_registry(tier, KeyRegistry())

        _seed_speaker_node(loop, "speaker0", "Alex")
        _seed_speaker_node(loop, "speaker1", "Robin")

        # §0: speaker node keys are lowercase canonical.
        assert "speaker0" in loop.merger.graph.nodes
        assert "speaker1" in loop.merger.graph.nodes

        # SOTA emits BOTH directions, both symmetric=true, lowercase speaker ids.
        rels = [
            {
                "subject": "speaker0",
                "predicate": "colleague_of",
                "object": "speaker1",
                "relation_type": "social",
                "confidence": 0.9,
                "symmetric": True,
            },
            {
                "subject": "speaker1",
                "predicate": "colleague_of",
                "object": "speaker0",
                "relation_type": "social",
                "confidence": 0.9,
                "symmetric": True,
            },
        ]
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with patch(
            "paramem.training.consolidation._graph_enrich_with_sota",
            return_value=(rels, [], "raw"),
        ):
            result = loop._run_graph_enrichment()

        assert not result["skipped"]
        # Both directed colleague_of edges survive (not collapsed — both_speakers gate).
        # Edges root at canonical lowercase keys.
        colleague_edges = [
            (u, v)
            for u, v, d in loop.merger.graph.edges(data=True)
            if d.get(_EDGE_SOURCE_ATTR) == "graph_enrichment"
            and d.get("predicate") == "colleague of"
        ]
        assert set(colleague_edges) == {("speaker0", "speaker1"), ("speaker1", "speaker0")}, (
            f"Both directed speaker↔speaker edges must survive; got {colleague_edges}"
        )

        # Mint keys from the edges (fold discipline, defer=False).
        tier_keyed: dict = {"episodic": [], "semantic": [], "procedural": []}
        loop._build_all_edge_entries_into(tier_keyed)

        minted = [e for e in tier_keyed["episodic"] if e["predicate"] == "colleague of"]
        assert len(minted) == 2, f"Expected 2 minted colleague_of keys; got {minted}"
        sids = {e["speaker_id"] for e in minted}
        assert sids == {"speaker0", "speaker1"}, (
            f"Two keys must carry distinct speaker_ids (lowercase canonical); got {sids}"
        )

        # Router index files each key under its own speaker.
        router = QueryRouter(adapter_dir=tmp_path, memory_store=loop.store)
        router.reload()
        s0_keys = router._speaker_key_index.get("speaker0", set())
        s1_keys = router._speaker_key_index.get("speaker1", set())
        s0_minted = {e["key"] for e in minted if e["speaker_id"] == "speaker0"}
        s1_minted = {e["key"] for e in minted if e["speaker_id"] == "speaker1"}
        assert s0_minted <= s0_keys, (
            f"speaker0's key must be filed under speaker0; index={s0_keys}, key={s0_minted}"
        )
        assert s1_minted <= s1_keys, (
            f"speaker1's key must be filed under speaker1; index={s1_keys}, key={s1_minted}"
        )

    def test_same_as_contracts_unbound_into_verbatim_speaker_node(self, tmp_path, monkeypatch):
        """same_as ['speaker0', 'alex'] contracts the unbound 'alex' concept node
        INTO the casefolded speaker node 'speaker0'.

        §0 invariant: keep="speaker0" resolves via P5 to "speaker0" (canonical fallback;
        "speaker0" is NOT a live node key after Step 2 — key is "speaker0").
        drop="alex" resolves via membership shortcut (already in graph).  Contraction
        succeeds: "alex" is absorbed into "speaker0"."""
        from paramem.training.key_registry import KeyRegistry

        loop = _make_loop(tmp_path, replay_enabled=True)
        _populate_graph(loop.merger.graph, n_persons=10)
        for tier in ("episodic", "semantic", "procedural"):
            loop.store.load_registry(tier, KeyRegistry())

        # Casefolded speaker node ("speaker0") + an unbound concept node "alex".
        _seed_speaker_node(loop, "speaker0", "Alex")
        loop.merger.graph.add_node(
            "alex",
            entity_type="person",
            attributes={"name": "Alex"},
            reinforcement_count=1,
            sessions=["s200"],
            first_seen="s200",
            last_seen="s200",
        )
        # Give "alex" an edge so the contraction has something to move.
        loop.merger.graph.add_node("rust", attributes={"name": "Rust"})
        loop.merger.graph.add_edge("alex", "rust", predicate="knows", relation_type="factual")

        assert "speaker0" in loop.merger.graph.nodes
        assert "alex" in loop.merger.graph.nodes

        # SAME_AS keep="speaker0" (cased; canonical fallback resolves to "speaker0"), drop="alex".
        # Patch surface gate to True (the opaque "speaker0" shares no token with "alex").
        rels: list = []
        same_as = [["speaker0", "alex"]]
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with (
            patch(
                "paramem.training.consolidation._graph_enrich_with_sota",
                return_value=(rels, same_as, "raw"),
            ),
            patch(
                "paramem.training.consolidation._safe_to_merge_surface",
                return_value=True,
            ),
        ):
            result = loop._run_graph_enrichment()

        assert not result["skipped"]
        # Contraction landed on the canonical speaker node "speaker0";
        # "alex" is gone (absorbed).
        assert "speaker0" in loop.merger.graph.nodes, (
            "same_as must contract into the canonical 'speaker0' speaker node"
        )
        assert "alex" not in loop.merger.graph.nodes, (
            "'alex' must be absorbed into the 'speaker0' node"
        )
        # The moved edge now roots at the casefolded speaker node.
        assert any(
            u == "speaker0" and d.get("predicate") == "knows"
            for u, _v, d in loop.merger.graph.edges(data=True)
        ), "knows edge must move onto the 'speaker0' speaker node after contraction"


# ---------------------------------------------------------------------------
# Tests for _unique_speaker_predecessor (pure unit + integration)
# ---------------------------------------------------------------------------


class TestUniqueSpeakerPredecessor:
    """Direct unit tests for ConsolidationLoop._unique_speaker_predecessor.

    Uses a minimal loop with a manually-populated merger.graph (nx.MultiDiGraph).
    No enrichment, no SOTA calls.
    """

    def test_zero_predecessors_returns_empty(self, tmp_path):
        """An isolated node with no predecessors → ''."""
        loop = _make_loop(tmp_path)
        loop.merger.graph.add_node("concept", attributes={})

        assert loop._unique_speaker_predecessor("concept") == ""

    def test_one_speaker_predecessor_returns_sid(self, tmp_path):
        """Exactly one predecessor with a non-empty speaker_id → that sid."""
        loop = _make_loop(tmp_path)
        loop.merger.graph.add_node(
            "speaker0",
            entity_type="person",
            speaker_id="speaker0",
        )
        loop.merger.graph.add_node("concept", attributes={})
        loop.merger.graph.add_edge("speaker0", "concept", predicate="held role")

        assert loop._unique_speaker_predecessor("concept") == "speaker0"

    def test_two_speaker_predecessors_returns_empty(self, tmp_path):
        """Two distinct speakers → '' (ambiguous — never mis-attribute)."""
        loop = _make_loop(tmp_path)
        loop.merger.graph.add_node("speaker0", speaker_id="speaker0")
        loop.merger.graph.add_node("speaker1", speaker_id="speaker1")
        loop.merger.graph.add_node("concept", attributes={})
        loop.merger.graph.add_edge("speaker0", "concept", predicate="held role")
        loop.merger.graph.add_edge("speaker1", "concept", predicate="held role")

        assert loop._unique_speaker_predecessor("concept") == ""

    def test_no_transitive_inheritance(self, tmp_path):
        """Chain A(speaker_id='S0') → B(concept) → C(concept): query on C returns ''
        because B carries no speaker_id — inheritance is 1-hop only."""
        loop = _make_loop(tmp_path)
        loop.merger.graph.add_node("speaker0", speaker_id="speaker0")
        loop.merger.graph.add_node("B", attributes={})
        loop.merger.graph.add_node("C", attributes={})
        loop.merger.graph.add_edge("speaker0", "B", predicate="held role")
        loop.merger.graph.add_edge("B", "C", predicate="related to")

        # B is C's predecessor, but B has no speaker_id → does not propagate.
        assert loop._unique_speaker_predecessor("C") == ""

    def test_node_not_in_graph_returns_empty(self, tmp_path):
        """Querying a node that is not in the graph returns ''."""
        loop = _make_loop(tmp_path)
        assert loop._unique_speaker_predecessor("nonexistent_node") == ""

    def test_predecessor_with_empty_speaker_id_filtered(self, tmp_path):
        """A predecessor whose speaker_id attribute is '' is not counted as a speaker."""
        loop = _make_loop(tmp_path)
        loop.merger.graph.add_node("NoSid", speaker_id="")
        loop.merger.graph.add_node("concept", attributes={})
        loop.merger.graph.add_edge("NoSid", "concept", predicate="related to")

        # '' is not a speaker — no non-empty speaker predecessor → ''.
        assert loop._unique_speaker_predecessor("concept") == ""


class TestSpeakerPredecessorInheritance:
    """Integration: _unique_speaker_predecessor fills speaker_id gaps for
    concept-rooted enrichment edges going through the real merger path.

    All tests seed the graph via _seed_speaker_node / _run_graph_enrichment so
    edges land in merger.graph through the real merger (no raw add_edge for
    enrichment edges).
    """

    def test_gap_filled_single_speaker_predecessor(self, tmp_path, monkeypatch):
        """Role-concept attribute edge inherits speaker_id from the unique speaker.

        Graph: speaker0 →held_role→ 'Senior PM', 'Senior PM' →achievement→ 'Award X'
        (both via SOTA canned result).  After enrichment + mint, the minted key for
        'achievement' must carry speaker_id='speaker0' and be filed under speaker0
        in a rebuilt router index.
        """
        from paramem.server.router import QueryRouter
        from paramem.training.key_registry import KeyRegistry

        loop = _make_loop(tmp_path, replay_enabled=True)
        _populate_graph(loop.merger.graph, n_persons=10)
        for tier in ("episodic", "semantic", "procedural"):
            loop.store.load_registry(tier, KeyRegistry())

        # Real speaker node via the merger (verbatim key "speaker0").
        _seed_speaker_node(loop, "speaker0", "Alex")

        # SOTA emits: bridge edge + role-concept attribute edge.
        # The role concept node ("Senior PM") has no speaker_id of its own.
        rels = [
            {
                "subject": "speaker0",
                "predicate": "held_role",
                "object": "Senior PM",
                "relation_type": "factual",
                "confidence": 0.9,
            },
            {
                "subject": "Senior PM",
                "predicate": "achievement",
                "object": "Award X",
                "relation_type": "factual",
                "confidence": 0.9,
            },
        ]
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with patch(
            "paramem.training.consolidation._graph_enrich_with_sota",
            return_value=(rels, [], "raw"),
        ):
            result = loop._run_graph_enrichment()

        assert not result["skipped"]

        # Confirm the role concept node has no own speaker_id (pre-condition for fallback).
        role_node = loop.merger.graph.nodes.get("senior pm", {})
        assert not role_node.get("speaker_id"), (
            "Role concept node must NOT carry a direct speaker_id before fallback"
        )

        # Confirm the speaker node is a predecessor of "senior pm" (bridge edge present).
        # §0: speaker node key is casefolded ("speaker0"), not "speaker0".
        preds = list(loop.merger.graph.predecessors("senior pm"))
        assert "speaker0" in preds, (
            f"Bridge edge speaker0 →held_role→ 'senior pm' must be in graph; predecessors={preds}"
        )

        # Mint keys via the unified builder (fold discipline).
        tier_keyed: dict = {"episodic": [], "semantic": [], "procedural": []}
        loop._build_all_edge_entries_into(tier_keyed)

        # Find the minted key for the 'achievement' edge (subject = "senior pm").
        achievement_keys = [
            e
            for tier in ("episodic", "procedural")
            for e in tier_keyed[tier]
            if e.get("predicate") == "achievement"
        ]
        assert achievement_keys, "No minted key found for the 'achievement' edge"
        key = achievement_keys[0]["key"]
        assert achievement_keys[0]["speaker_id"] == "speaker0", (
            f"Minted achievement key must carry speaker_id='speaker0' via fallback; "
            f"got {achievement_keys[0]['speaker_id']!r}"
        )

        bk = loop.store.bookkeeping_for_key(key)
        assert bk is not None
        assert bk["speaker_id"] == "speaker0", (
            f"Bookkeeping speaker_id must be 'speaker0'; got {bk['speaker_id']!r}"
        )

        # Router index must file the key under speaker0.
        router = QueryRouter(adapter_dir=tmp_path, memory_store=loop.store)
        router.reload()
        s0_keys = router._speaker_key_index.get("speaker0", set())
        assert key in s0_keys, (
            f"Achievement key must be in router._speaker_key_index['speaker0']; "
            f"index={s0_keys}, key={key!r}"
        )

    def test_no_misattribution_two_speaker_predecessors(self, tmp_path, monkeypatch):
        """Two speakers both hold the same role concept → attribute key mints with ''
        (ambiguous — must not be attributed to either speaker).

        Graph: speaker0 →held_role→ 'Engineer', speaker1 →held_role→ 'Engineer',
        'Engineer' →attr→ 'Y'.  Fallback sees 2 distinct predecessors → ''.
        """
        from paramem.server.router import QueryRouter
        from paramem.training.key_registry import KeyRegistry

        loop = _make_loop(tmp_path, replay_enabled=True)
        _populate_graph(loop.merger.graph, n_persons=10)
        for tier in ("episodic", "semantic", "procedural"):
            loop.store.load_registry(tier, KeyRegistry())

        _seed_speaker_node(loop, "speaker0", "Alex")
        _seed_speaker_node(loop, "speaker1", "Robin")

        rels = [
            {
                "subject": "speaker0",
                "predicate": "held_role",
                "object": "Engineer",
                "relation_type": "factual",
                "confidence": 0.9,
            },
            {
                "subject": "speaker1",
                "predicate": "held_role",
                "object": "Engineer",
                "relation_type": "factual",
                "confidence": 0.9,
            },
            {
                "subject": "Engineer",
                "predicate": "attr",
                "object": "Y",
                "relation_type": "factual",
                "confidence": 0.9,
            },
        ]
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with patch(
            "paramem.training.consolidation._graph_enrich_with_sota",
            return_value=(rels, [], "raw"),
        ):
            result = loop._run_graph_enrichment()

        assert not result["skipped"]

        tier_keyed: dict = {"episodic": [], "semantic": [], "procedural": []}
        loop._build_all_edge_entries_into(tier_keyed)

        attr_keys = [
            e
            for tier in ("episodic", "procedural")
            for e in tier_keyed[tier]
            if e.get("predicate") == "attr"
        ]
        assert attr_keys, "No minted key found for the 'attr' edge"
        assert attr_keys[0]["speaker_id"] == "", (
            f"Shared-role attribute key must mint with speaker_id='' (ambiguous); "
            f"got {attr_keys[0]['speaker_id']!r}"
        )

        # Must not be indexed under either speaker.
        router = QueryRouter(adapter_dir=tmp_path, memory_store=loop.store)
        router.reload()
        attr_key = attr_keys[0]["key"]
        s0_keys = router._speaker_key_index.get("speaker0", set())
        s1_keys = router._speaker_key_index.get("speaker1", set())
        assert attr_key not in s0_keys, "Ambiguous key must NOT appear under speaker0"
        assert attr_key not in s1_keys, "Ambiguous key must NOT appear under speaker1"

    def test_never_overwrites_working_attribution(self, tmp_path, monkeypatch):
        """Fallback must NOT fire when attribution already resolves correctly.

        Sub-case (a): speaker-rooted enrichment — subject IS the speaker node.
        The subject node carries speaker_id='speaker0', so the node-attr branch
        resolves before the fallback.

        Sub-case (b): extraction edge with a non-empty edge-level speaker_id.
        The edge-attr branch resolves before both the node-attr AND the fallback.

        _unique_speaker_predecessor must NOT be called for either subject.
        """
        from paramem.training.key_registry import KeyRegistry

        loop = _make_loop(tmp_path, replay_enabled=True)
        _populate_graph(loop.merger.graph, n_persons=10)
        for tier in ("episodic", "semantic", "procedural"):
            loop.store.load_registry(tier, KeyRegistry())

        # Seed a real speaker node.
        _seed_speaker_node(loop, "speaker0", "Alex")

        # Sub-case (a): speaker-rooted enrichment (subject = speaker node).
        rels_a = [
            {
                "subject": "speaker0",
                "predicate": "likes",
                "object": "Coffee",
                "relation_type": "preference",
                "confidence": 0.9,
            }
        ]
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with patch(
            "paramem.training.consolidation._graph_enrich_with_sota",
            return_value=(rels_a, [], "raw"),
        ):
            loop._run_graph_enrichment()

        # Spy on _unique_speaker_predecessor to assert it is NOT called for
        # subjects that already have working attribution.
        called_for: list[str] = []
        original_helper = loop._unique_speaker_predecessor

        def _spy(node: str) -> str:
            called_for.append(node)
            return original_helper(node)

        loop._unique_speaker_predecessor = _spy  # type: ignore[method-assign]

        tier_keyed: dict = {"episodic": [], "semantic": [], "procedural": []}
        loop._build_all_edge_entries_into(tier_keyed)

        # The 'likes' edge's subject is "speaker0" which carries speaker_id —
        # node-attr branch resolves, fallback must NOT be reached.
        assert "speaker0" not in called_for, (
            f"_unique_speaker_predecessor must NOT be called for 'speaker0' "
            f"(already has speaker_id); was called for: {called_for}"
        )

        # Minted 'likes' key carries speaker_id='speaker0' (came from node-attr).
        likes_entries = [
            e
            for tier in ("episodic", "procedural")
            for e in tier_keyed[tier]
            if e.get("predicate") == "likes"
        ]
        assert likes_entries, "Expected a minted 'likes' key"
        assert likes_entries[0]["speaker_id"] == "speaker0", (
            f"Speaker-rooted edge must carry speaker_id='speaker0' (via node-attr); "
            f"got {likes_entries[0]['speaker_id']!r}"
        )

        # Sub-case (b): extraction edge with edge-level speaker_id already set.
        called_for.clear()
        tier_keyed2: dict = {"episodic": [], "semantic": [], "procedural": []}

        # Add a raw edge whose edge data carries speaker_id (simulates extraction stamp).
        loop.merger.graph.add_node("work_item", attributes={"name": "Work Item"})
        loop.merger.graph.add_edge(
            "concept_x",
            "work_item",
            predicate="tracks",
            relation_type="factual",
            speaker_id="speaker0",  # edge-level stamp
            confidence=0.9,
        )
        loop.merger.graph.add_node("concept_x", attributes={"name": "Concept X"})

        loop._build_all_edge_entries_into(tier_keyed2)

        # The edge-attr branch resolves; fallback must NOT be called for "concept_x".
        assert "concept_x" not in called_for, (
            f"_unique_speaker_predecessor must NOT be called for 'concept_x' "
            f"(edge carries speaker_id); was called for: {called_for}"
        )
        tracks_entries = [
            e
            for tier in ("episodic", "procedural")
            for e in tier_keyed2[tier]
            if e.get("predicate") == "tracks"
        ]
        assert tracks_entries, "Expected a minted 'tracks' key"
        assert tracks_entries[0]["speaker_id"] == "speaker0", (
            f"Edge-stamped edge must carry speaker_id='speaker0' (via edge attr); "
            f"got {tracks_entries[0]['speaker_id']!r}"
        )

    def test_zero_predecessor_concept(self, tmp_path, monkeypatch):
        """An isolated concept node with an enrichment attribute edge and no
        speaker predecessors mints with speaker_id='' (allow_empty path).

        The node is introduced as the object of an enrichment edge (so it
        appears in the graph), but no bridge edge points into it from any speaker.
        """
        from paramem.training.key_registry import KeyRegistry

        loop = _make_loop(tmp_path, replay_enabled=True)
        _populate_graph(loop.merger.graph, n_persons=10)
        for tier in ("episodic", "semantic", "procedural"):
            loop.store.load_registry(tier, KeyRegistry())

        _seed_speaker_node(loop, "speaker0", "Alex")

        # SOTA emits an attribute edge whose SUBJECT is a brand-new concept node
        # with no speaker predecessor (no bridge edge into it).
        rels = [
            {
                "subject": "Isolated Concept",
                "predicate": "has_property",
                "object": "Some Value",
                "relation_type": "factual",
                "confidence": 0.9,
            }
        ]
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with patch(
            "paramem.training.consolidation._graph_enrich_with_sota",
            return_value=(rels, [], "raw"),
        ):
            result = loop._run_graph_enrichment()

        assert not result["skipped"]

        tier_keyed: dict = {"episodic": [], "semantic": [], "procedural": []}
        loop._build_all_edge_entries_into(tier_keyed)

        prop_keys = [
            e
            for tier in ("episodic", "procedural")
            for e in tier_keyed[tier]
            if e.get("predicate") == "has property"
        ]
        assert prop_keys, "No minted key found for the 'has_property' edge"
        assert prop_keys[0]["speaker_id"] == "", (
            f"Zero-predecessor concept must mint with speaker_id=''; "
            f"got {prop_keys[0]['speaker_id']!r}"
        )

    def test_extraction_concept_edge_not_attributed(self, tmp_path):
        """Scope boundary: an EXTRACTION concept-edge (no edge_source) with a
        single speaker predecessor must keep speaker_id='' — the fallback must
        NOT fire for non-enrichment edges.

        This locks the deliberate unattributed-fact behavior (e.g. a company-
        location fact extracted alongside a speaker → the company node has the
        speaker as a predecessor, but the fact is not personal to that speaker).
        """
        from paramem.training.key_registry import KeyRegistry

        loop = _make_loop(tmp_path, replay_enabled=True)
        for tier in ("episodic", "semantic", "procedural"):
            loop.store.load_registry(tier, KeyRegistry())

        # Speaker node with speaker_id (simulates a real speaker in the graph).
        loop.merger.graph.add_node(
            "speaker0",
            entity_type="person",
            speaker_id="speaker0",
            attributes={"name": "speaker0"},
        )

        # Concept node that the speaker is the UNIQUE predecessor of
        # (e.g. "Acme Corp" — speaker0 has a works_at edge into it).
        loop.merger.graph.add_node(
            "acme corp",
            entity_type="organization",
            attributes={"name": "Acme Corp"},
        )
        loop.merger.graph.add_edge(
            "speaker0",
            "acme corp",
            predicate="works at",
            relation_type="factual",
            speaker_id="speaker0",
            confidence=1.0,
            # NOTE: no edge_source here (extraction edge, not enrichment).
        )

        # Extraction concept-edge: Acme Corp →is_located_in→ Germany.
        # No edge_source (extraction), no speaker_id on the edge, no speaker_id
        # on the subject node.  Even though speaker0 is the unique predecessor
        # of "acme corp", the fallback must NOT fire — deliberate unattributed fact.
        loop.merger.graph.add_node(
            "germany", entity_type="location", attributes={"name": "Germany"}
        )
        loop.merger.graph.add_edge(
            "acme corp",
            "germany",
            predicate="is located in",
            relation_type="factual",
            confidence=1.0,
            # NOTE: no edge_source (extraction), no speaker_id.
        )

        tier_keyed: dict = {"episodic": [], "semantic": [], "procedural": []}
        loop._build_all_edge_entries_into(tier_keyed)

        located_keys = [
            e
            for tier in ("episodic", "procedural")
            for e in tier_keyed[tier]
            if e.get("predicate") == "is located in"
        ]
        assert located_keys, "No minted key found for the 'is located in' edge"
        assert located_keys[0]["speaker_id"] == "", (
            f"Extraction concept-edge must keep speaker_id='' even when a unique "
            f"speaker predecessor exists; got {located_keys[0]['speaker_id']!r}"
        )


# ---------------------------------------------------------------------------
# Graph-tier anonymization contract — the second call site of the
# anonymize -> SOTA -> de-anonymize contract (paramem.graph.placeholders).
# ---------------------------------------------------------------------------


class TestGraphTierAnonymizationContract:
    """paramem.graph.extractor._graph_enrich_with_sota now runs the SAME
    anonymize -> SOTA -> de-anonymize contract as session-tier extraction
    (_sota_pipeline), via the shared primitives in paramem.graph.placeholders.
    """

    def test_graph_enrichment_sends_no_real_names(self):
        """No name present as a key in the caller-supplied ``mapping``
        reaches the payload handed to ``_sota_call``; it renders as its
        placeholder token instead.  A bare ``speaker{N}`` id is never a
        ``mapping`` key in production (the local anonymizer prompt
        forbids mapping it — see :func:`_build_anonymization_mapping`'s
        speaker-anchor invariants), so it legitimately reaches the
        payload UNMASKED — ``_graph_enrich_with_sota`` applies no scope
        gate of its own; it substitutes exactly what ``mapping`` says
        (the model's own mapping is the sole scope authority).

        Mutation: restore the pre-fix call (pass ``triples`` straight to
        ``_sota_call`` with no anonymization step) -> ``"alice"`` appears
        in the captured payload -> this test fails.
        """
        from paramem.graph.extractor import _graph_enrich_with_sota

        triples = [
            {
                "subject": "alice",
                "predicate": "colleague_of",
                "object": "speaker0",
                "relation_type": "social",
                "speaker_id": "speaker0",
            }
        ]
        # The model's own scoped mapping — only "alice" was classified
        # in scope; "speaker0" is never a mapping key by construction.
        mapping = {"alice": "Person_1"}

        captured: list[str] = []

        def _capture(prompt, *args, **kwargs):
            captured.append(prompt)
            return '{"relations": [], "same_as": []}'

        with patch("paramem.graph.extractor._sota_call", side_effect=_capture):
            result = _graph_enrich_with_sota(
                triples,
                mapping,
                api_key="test-key",
                provider="anthropic",
                filter_model="claude-sonnet-4-6",
            )

        assert result is not None
        assert captured, "Expected the SOTA call to be made"
        payload = captured[0]
        assert "alice" not in payload, f"Real name 'alice' leaked into payload: {payload}"
        assert "Person_1" in payload, f"Expected 'alice' tokenised as Person_1; got: {payload}"
        assert '"speaker0"' in payload, (
            f"speaker0 is never a mapping key — it must reach the payload bare: {payload}"
        )

    def test_graph_enrichment_round_trips_to_real_names(self):
        """Relations SOTA returns (naming tokens) come back with real node
        names after ``_graph_enrich_with_sota`` returns; a bare
        ``speaker{N}`` id (never tokenised — not a ``mapping`` key) round
        trips unchanged.

        Mutation: drop the deanon step (return ``new_relations``/
        ``same_as_pairs`` straight from the parsed response, un-substituted)
        -> the placeholder reaches the merger -> this test fails.
        """
        from paramem.graph.extractor import _graph_enrich_with_sota

        mapping = {"alice": "Person_1"}

        canned_raw = (
            '{"relations": [{"subject": "Person_1", "predicate": "colleague_of", '
            '"object": "speaker0", "relation_type": "social", "confidence": 0.9, '
            '"symmetric": true}], "same_as": []}'
        )

        with patch("paramem.graph.extractor._sota_call", return_value=canned_raw):
            result = _graph_enrich_with_sota(
                [],
                mapping,
                api_key="test-key",
                provider="anthropic",
                filter_model="claude-sonnet-4-6",
            )

        assert result is not None
        new_rels, same_as, _raw = result
        assert len(new_rels) == 1
        assert new_rels[0]["subject"] == "alice"
        assert new_rels[0]["object"] == "speaker0"
        assert same_as == []

    def test_graph_enrichment_unresolved_token_dropped(self):
        """A relation naming a token in neither the CORE table nor SOTA
        bindings is DROPPED at the exit gate, not forwarded with a residual
        placeholder.

        Mutation: remove the exit gate (``_apply_bindings``) on this path
        -> the unresolved token escapes into ``new_relations`` -> this test
        fails.
        """
        from paramem.graph.extractor import _graph_enrich_with_sota

        mapping = {"alice": "Person_1"}
        canned_raw = (
            '{"relations": [{"subject": "Person_1", "predicate": "knows", '
            '"object": "Person_99", "relation_type": "social", '
            '"confidence": 0.9, "symmetric": false}], "same_as": []}'
        )
        with patch("paramem.graph.extractor._sota_call", return_value=canned_raw):
            result = _graph_enrich_with_sota(
                [],
                mapping,
                api_key="test-key",
                provider="anthropic",
                filter_model="claude-sonnet-4-6",
            )

        assert result is not None
        new_rels, _same_as, _raw = result
        assert new_rels == [], (
            f"Relation naming an unresolved token must be dropped; got {new_rels!r}"
        )

    def test_graph_enrichment_masks_exactly_what_the_mapping_declares(self):
        """``_graph_enrich_with_sota`` applies NO scope gate of its own
        — it substitutes exactly the entries the caller's
        ``mapping`` declares, nothing more, nothing less.  The caller's
        ``mapping`` is the model's own ``scrub``-scoped decision in
        production; this test proves the function trusts it verbatim
        rather than re-deriving scope from entity types.

        Mutation: hardcode a scope inside ``_graph_enrich_with_sota``
        (ignore what ``mapping`` actually contains) -> either assertion
        below fails.
        """
        from paramem.graph.extractor import _graph_enrich_with_sota

        triples = [
            {
                "subject": "alice",
                "predicate": "works_at",
                "object": "acme",
                "relation_type": "factual",
                "speaker_id": "",
            }
        ]

        captured: list[str] = []

        def _capture(prompt, *args, **kwargs):
            captured.append(prompt)
            return '{"relations": [], "same_as": []}'

        # The model classified only "alice" in scope.
        with patch("paramem.graph.extractor._sota_call", side_effect=_capture):
            _graph_enrich_with_sota(
                triples,
                {"alice": "Person_1"},
                api_key="k",
                provider="anthropic",
                filter_model="claude-sonnet-4-6",
            )
        person_only_payload = captured[-1]
        assert "alice" not in person_only_payload
        assert "acme" in person_only_payload

        # The model classified only "acme" in scope (its own decision —
        # not a code-side entity-type re-derivation).
        with patch("paramem.graph.extractor._sota_call", side_effect=_capture):
            _graph_enrich_with_sota(
                triples,
                {"acme": "Org_1"},
                api_key="k",
                provider="anthropic",
                filter_model="claude-sonnet-4-6",
            )
        org_only_payload = captured[-1]
        assert "acme" not in org_only_payload
        assert "alice" in org_only_payload

    def test_graph_enrichment_placeholder_threaded_verbatim_never_reminted(self):
        """The model's OWN placeholder (whatever token it minted) is
        threaded straight through to the SOTA payload and the returned
        relation — ``_graph_enrich_with_sota`` never re-mints its own
        token for a name already present in ``mapping``, regardless of
        the placeholder's shape (verified with a real anonymizer-style
        surface, not a ``Person_N`` convenience literal).  This is the
        regression this contract exists to prevent: re-minting instead of
        threading through would desync the forward token from what SOTA
        is shown, or silently drop a mapping entry.
        """
        from paramem.graph.extractor import _graph_enrich_with_sota

        triples = [
            {
                "subject": "yang ming",
                "predicate": "works_at",
                "object": "acme",
                "relation_type": "factual",
                "speaker_id": "",
            }
        ]
        mapping = {"yang ming": "Person_7"}

        captured: list[str] = []

        def _capture(prompt, *args, **kwargs):
            captured.append(prompt)
            return (
                '{"relations": [{"subject": "Person_7", "predicate": "works_at", '
                '"object": "acme", "relation_type": "factual", "confidence": 0.9, '
                '"symmetric": false}], "same_as": []}'
            )

        with patch("paramem.graph.extractor._sota_call", side_effect=_capture):
            result = _graph_enrich_with_sota(
                triples,
                mapping,
                api_key="test-key",
                provider="anthropic",
                filter_model="claude-sonnet-4-6",
            )

        assert result is not None
        payload = captured[0]
        assert "yang ming" not in payload, f"real name leaked into payload: {payload}"
        assert "Person_7" in payload, (
            f"the model's own placeholder must reach the payload verbatim: {payload}"
        )
        new_rels, _same_as, _raw = result
        assert len(new_rels) == 1
        assert new_rels[0]["subject"] == "yang ming", (
            "Person_7 must round-trip to the real name via the caller's own "
            f"mapping, not a re-minted token; got {new_rels[0]!r}"
        )

    def test_graph_enrichment_same_as_deanonymized_before_speaker_guard(self):
        """``same_as`` pairs must be real names by the time they LEAVE
        ``_graph_enrich_with_sota`` — i.e. strictly before
        ``ConsolidationLoop._run_graph_enrichment``'s speaker-pair guard
        (``is_speaker_id(keep) and is_speaker_id(drop)``, consolidation.py
        ~:2659) ever sees them. ``is_speaker_id`` cannot recognise a
        placeholder token (``Person_N``) as a speaker id, so if
        deanonymization happened AFTER the guard (or not at all on the
        ``same_as`` path), the guard would silently stop firing downstream
        — this asserts the precondition the guard depends on directly at
        the function boundary. The function applies no speaker-id
        special-casing of its own — it deanonymizes mechanically via
        whatever ``mapping`` the caller supplies, so a directly-supplied
        speaker-shaped mapping (never how a real caller populates it, but
        a legal input to this pure function) still exercises the deanon
        ordering unit-level.

        Mutation: reorder so ``same_as`` pairs are returned still in token
        form (e.g. return ``same_as_pairs`` instead of ``deanon_same_as``)
        -> ``is_speaker_id`` on the returned pair is False -> this test
        fails.
        """
        from paramem.graph.extractor import _graph_enrich_with_sota
        from paramem.graph.name_match import is_speaker_id

        mapping = {"speaker0": "Person_1", "speaker1": "Person_2"}
        canned_raw = '{"relations": [], "same_as": [["Person_1", "Person_2"]]}'

        with patch("paramem.graph.extractor._sota_call", return_value=canned_raw):
            result = _graph_enrich_with_sota(
                [],
                mapping,
                api_key="test-key",
                provider="anthropic",
                filter_model="claude-sonnet-4-6",
            )

        assert result is not None
        _new_rels, same_as, _raw = result
        assert len(same_as) == 1
        keep, drop = same_as[0]
        assert is_speaker_id(keep) and is_speaker_id(drop), (
            "same_as pair must be bare speaker ids by the time it leaves "
            f"_graph_enrich_with_sota — the speaker-pair guard depends on this; got {same_as[0]!r}"
        )
        assert {keep, drop} == {"speaker0", "speaker1"}

    def test_graph_enrichment_full_pipeline_blocks_speaker_same_as_merge(
        self, tmp_path, monkeypatch
    ):
        """Production-wiring confirmation: driving the same scenario through
        the REAL ``ConsolidationLoop._run_graph_enrichment`` (only
        ``_sota_call`` mocked; the local anonymizer stays on the module's
        default autouse stub, which masks non-speaker names as ``person``
        — speaker ids are never tokenised regardless, since the local
        anonymizer prompt forbids mapping ``speaker{N}`` in the first
        place) the two distinct speaker nodes are never contracted — the
        speaker-pair guard fires against the ``same_as`` pair ``_graph_enrich_with_sota``
        returns, even though the SOTA payload carried the BARE speaker ids
        verbatim (never opaque tokens: there is nothing to de-anonymize
        here). This is defense-in-depth: the guard fires independently of
        whatever the anonymizer did or didn't mask.
        """
        loop = _make_loop(tmp_path)
        graph = loop.merger.graph
        _populate_graph(graph, n_persons=10)
        for sid in ("speaker0", "speaker1"):
            graph.add_node(
                sid,
                entity_type="person",
                speaker_id=sid,
                attributes={"name": sid},
                reinforcement_count=20,
                sessions=["s100"],
                first_seen="s100",
                last_seen="s100",
            )
            graph.add_edge(
                sid,
                "acmecorp",
                predicate="works at",
                relation_type="factual",
                confidence=1.0,
                speaker_id=sid,
                source="extraction",
                sessions=["s100"],
            )

        node_count_before = graph.number_of_nodes()

        def _sota_response(prompt, *args, **kwargs):
            # speaker0/speaker1 are never in the local anonymizer's mapping
            # (excluded by construction — see chunk_mapping's is_speaker_id
            # filter), so they reach the payload bare, not as opaque
            # tokens — the model proposes a same_as pair on the bare ids
            # directly, exactly as a real cloud model minus the
            # code-level speaker-pair guard would.
            assert '"speaker0"' in prompt and '"speaker1"' in prompt, (
                f"Expected bare speaker ids in payload: {prompt}"
            )
            return '{"relations": [], "same_as": [["speaker0", "speaker1"]]}'

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with patch("paramem.graph.extractor._sota_call", side_effect=_sota_response):
            result = loop._run_graph_enrichment()

        assert not result["skipped"]
        assert result["same_as_merges"] == 0, (
            "The speaker-pair guard must block the speaker0/speaker1 same_as pair even "
            "though the model saw the bare speaker ids directly; "
            f"got same_as_merges={result['same_as_merges']}"
        )
        assert "speaker0" in graph.nodes
        assert "speaker1" in graph.nodes
        assert graph.number_of_nodes() == node_count_before


def _populate_untyped_graph(graph: nx.MultiDiGraph, n_persons: int = 10) -> None:
    """Mirror the merger's REAL fallback for the post-merge fold graph.

    Registry-derived relation endpoints get ``entity_type="concept"`` from
    ``GraphMerger`` (``paramem/graph/merger.py`` — no reliable type is ever
    known for them), unlike ``_populate_graph`` above (used by the other
    tests in this file) which sets accurate ``entity_type`` on every node
    for test convenience — that is NOT what the production fold graph looks
    like. These tests exist specifically to drive ``_run_graph_enrichment``
    against a graph with no usable type signal, proving the type source is
    now the local-model anonymization pass, not node attributes.
    """
    for i in range(n_persons):
        name = f"person{i}"
        graph.add_node(
            name,
            entity_type="concept",
            attributes={"name": f"Person{i}"},
            reinforcement_count=i + 1,
            sessions=[f"s{i:03d}"],
            first_seen=f"s{i:03d}",
            last_seen=f"s{i:03d}",
        )
    org = "acmecorp"
    graph.add_node(
        org,
        entity_type="concept",
        attributes={"name": "AcmeCorp"},
        reinforcement_count=n_persons,
        sessions=["s000"],
        first_seen="s000",
        last_seen="s000",
    )
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


def _stub_local_model_types(type_by_name: dict[str, str]):
    """Build a stand-in for ``anonymize_with_local_model`` typing real names
    per ``type_by_name`` (default ``"person"`` for anything unlisted),
    minting ``Prefix_N`` tokens in sorted-name order — simulating what the
    LOCAL model's own anonymization pass would classify each real name as.

    Returns the post-redesign three-artifact shape
    (``mapping, anonymized_transcript, raw``) — every name the stub mints
    a placeholder for is also substituted into a synthetic
    ``anonymized_transcript`` so callers that thread the fail-closed
    check (a missing/empty transcript blocks the chunk) see a realistic
    non-empty value.
    """
    from paramem.graph.placeholders import _substitute_whole_words, entity_type_to_prefix

    def _stub(session_graph, model, tokenizer, transcript="", **kwargs):
        names = sorted(
            {r.subject for r in session_graph.relations}
            | {r.object for r in session_graph.relations}
        )
        mapping: dict[str, str] = {}
        counters: dict[str, int] = {}
        for name in names:
            prefix = entity_type_to_prefix(type_by_name.get(name, "person"))
            counters[prefix] = counters.get(prefix, 0) + 1
            mapping[name] = f"{prefix}_{counters[prefix]}"
        anon_transcript = _substitute_whole_words(transcript, mapping) or "stub-anon-transcript"
        return mapping, anon_transcript, "stub-raw"

    return _stub


class TestGraphTierLocalModelTypeDerivation:
    """The cumulative fold graph carries no reliable entity types of its
    own (production reality — see ``_run_graph_enrichment``'s docstring
    and ``GraphMerger``'s ``entity_type="concept"`` fallback for endpoints
    without a known Entity). These tests drive the REAL
    ``ConsolidationLoop._run_graph_enrichment`` against a graph built with
    that fallback (``_populate_untyped_graph`` — every node typed
    ``"concept"``, unlike ``_populate_graph``'s test-convenience typing)
    and a controlled local-model classification (``_stub_local_model_types``,
    replacing this module's default empty-mapping ``_stub_local_anonymize``
    fixture), proving that entity types now come from the local model's own
    classification rather than from node attributes.
    """

    def test_graph_enrichment_masks_persons(self, tmp_path, monkeypatch):
        """A real name the LOCAL model classifies as a person must not reach
        the payload handed to ``_sota_call`` verbatim, even though the fold
        graph node itself carries no usable ``entity_type``.

        Mutation: revert ``_run_graph_enrichment`` to reading node
        ``entity_type`` attributes (all ``"concept"`` here) instead of the
        local model's mapping -> "person0" is never masked -> this test fails.
        """
        loop = _make_loop(tmp_path)
        graph = loop.merger.graph
        _populate_untyped_graph(graph, n_persons=10)

        captured: list[str] = []

        def _capture(prompt, *args, **kwargs):
            captured.append(prompt)
            return '{"relations": [], "same_as": []}'

        stub = _stub_local_model_types({"acmecorp": "organization"})
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with (
            patch("paramem.training.consolidation.anonymize_with_local_model", side_effect=stub),
            patch("paramem.graph.extractor._sota_call", side_effect=_capture),
        ):
            result = loop._run_graph_enrichment()

        assert not result["skipped"]
        assert captured, "Expected the SOTA call to be made"
        payload = captured[0]
        assert "person0" not in payload, f"Real name 'person0' leaked into payload: {payload}"
        assert "Person_" in payload, f"Expected person nodes tokenised as Person_N; got: {payload}"

    def test_graph_enrichment_leaves_out_of_scope_verbatim(self, tmp_path, monkeypatch):
        """A node the LOCAL model's own mapping OMITS (the model's scope
        decision against ``scrub``, e.g. an organization when only
        ``person name`` is configured) must appear VERBATIM in the SOTA
        payload — this is what preserves ``same_as`` for non-person
        entities. Post-redesign there is no code-side scope filter
        downstream of the model's mapping — omission from the mapping IS
        the exclusion mechanism, so the stub omits ``acmecorp`` directly
        rather than typing it and relying on a downstream gate.

        Mutation: mask every entity regardless of what the mapping stub
        omitted -> "acmecorp" is also tokenised -> this test fails.
        """
        loop = _make_loop(tmp_path)
        graph = loop.merger.graph
        _populate_untyped_graph(graph, n_persons=10)

        captured: list[str] = []

        def _capture(prompt, *args, **kwargs):
            captured.append(prompt)
            return '{"relations": [], "same_as": []}'

        def _stub_omits_acmecorp(session_graph, model, tokenizer, transcript="", **kwargs):
            names = sorted(
                {r.subject for r in session_graph.relations}
                | {r.object for r in session_graph.relations}
            )
            mapping = {
                name: f"Person_{i + 1}"
                for i, name in enumerate(n for n in names if n != "acmecorp")
            }
            return mapping, "stub-anon-transcript", "stub-raw"

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with (
            patch(
                "paramem.training.consolidation.anonymize_with_local_model",
                side_effect=_stub_omits_acmecorp,
            ),
            patch("paramem.graph.extractor._sota_call", side_effect=_capture),
        ):
            result = loop._run_graph_enrichment()

        assert not result["skipped"]
        assert captured, "Expected the SOTA call to be made"
        payload = captured[0]
        # Check the DATA (triples_json), not the prompt's static examples —
        # the template itself mentions "Org_1" as a generic illustration.
        assert '"object": "acmecorp"' in payload, (
            f"Out-of-scope org must pass through verbatim: {payload}"
        )
        assert '"object": "Org_1"' not in payload, (
            f"Org must NOT be tokenised when the model's own mapping omitted it: {payload}"
        )

    def test_graph_enrichment_round_trips(self, tmp_path, monkeypatch):
        """A new relation SOTA returns (naming the LOCAL model's own tokens)
        comes back on the merged graph with REAL node names — the
        production caller never sees a placeholder.

        Mutation: skip the de-anonymize step for this call site -> the new
        edge lands keyed by ``Person_N`` tokens instead of "person0"/
        "person1" -> this test fails.
        """
        loop = _make_loop(tmp_path)
        graph = loop.merger.graph
        _populate_untyped_graph(graph, n_persons=10)

        stub = _stub_local_model_types({"acmecorp": "organization"})

        def _sota_response(prompt, *args, **kwargs):
            # Reference the SAME Person_N tokens the local-model stub just
            # minted for person0/person1 (sorted-name order: person0 -> 1st).
            return (
                '{"relations": [{"subject": "Person_1", "predicate": "colleague_of", '
                '"object": "Person_2", "relation_type": "social", "confidence": 0.9}], '
                '"same_as": []}'
            )

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with (
            patch("paramem.training.consolidation.anonymize_with_local_model", side_effect=stub),
            patch("paramem.graph.extractor._sota_call", side_effect=_sota_response),
        ):
            result = loop._run_graph_enrichment()

        assert not result["skipped"]
        assert result["new_edges"] >= 1

        found = False
        for _, _, data in graph.out_edges("person0", data=True):
            if (
                data.get("predicate") == "colleague of"
                and data.get(_EDGE_SOURCE_ATTR) == "graph_enrichment"
            ):
                found = True
        assert found, "Expected the enriched edge on real node 'person0', not a placeholder"


class TestGraphTierMappingReconciliation:
    """The local anonymizer's mapping keys are reconciled onto the
    ACTUAL node-key surfaces via ``canonical()`` inside
    ``_run_graph_enrichment`` itself; the shared ``_substitute_whole_words``
    primitive stays exact-match everywhere (see
    ``tests/test_placeholders.py::TestSubstituteWholeWordsExactMatchRegression``
    for why it must). The fold graph's node keys are already canonicalized
    (``"yang ming"``), while the local model's mapping is keyed by
    whatever real-name surface it independently produced (``"Yang
    Ming"``) — a raw comparison between the two silently misses, so the
    real name would reach the SOTA payload unmasked even though the model
    correctly identified it.
    """

    @staticmethod
    def _populate_two_node_chunk(graph: nx.MultiDiGraph, subject_key: str, object_key: str) -> None:
        """A small chunk with exactly one real edge plus enough filler
        nodes to clear the 10-node enrichment floor, all disconnected so
        the ego-graph chunk built around the highest-reinforcement node
        stays limited to ``{subject_key, object_key}`` plus filler.
        """
        graph.add_node(
            subject_key,
            entity_type="concept",
            attributes={"name": subject_key},
            reinforcement_count=10,
            sessions=["s000"],
            first_seen="s000",
            last_seen="s000",
        )
        graph.add_node(
            object_key,
            entity_type="concept",
            attributes={"name": object_key},
            reinforcement_count=9,
            sessions=["s000"],
            first_seen="s000",
            last_seen="s000",
        )
        graph.add_edge(
            subject_key,
            object_key,
            predicate="works at",
            relation_type="factual",
            confidence=1.0,
            source="extraction",
            sessions=["s000"],
        )
        for i in range(8):
            graph.add_node(
                f"filler{i}",
                entity_type="concept",
                attributes={},
                reinforcement_count=0,
                sessions=[],
                first_seen="",
                last_seen="",
            )

    def test_recased_local_mapping_key_still_masks_the_node(self, tmp_path, monkeypatch):
        """The node IS masked; no real name reaches the payload, even
        though the local model's mapping key ("Yang Ming") does not
        raw-string-match the graph's own node text ("yang ming").

        Mutation: remove the re-keying step in
        ``_run_graph_enrichment`` -> the real node text ("yang ming")
        reaches the SOTA payload verbatim -> this test fails.
        """
        loop = _make_loop(tmp_path)
        graph = loop.merger.graph
        self._populate_two_node_chunk(graph, "yang ming", "acmecorp")

        captured: list[str] = []

        def _capture(prompt, *args, **kwargs):
            captured.append(prompt)
            return '{"relations": [], "same_as": []}'

        monkeypatch.setattr(
            "paramem.training.consolidation.anonymize_with_local_model",
            lambda *args, **kwargs: ({"Yang Ming": "Person_1"}, "stub-anon-transcript", "stub-raw"),
        )
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with patch("paramem.graph.extractor._sota_call", side_effect=_capture):
            result = loop._run_graph_enrichment()

        assert not result["skipped"]
        assert captured, "Expected the SOTA call to be made"
        payload = captured[0]
        assert "yang ming" not in payload.lower(), f"Real name leaked into payload: {payload}"
        assert "Person_1" in payload, f"Expected the node masked as Person_1; got: {payload}"
        assert result["mapping_rekey_dropped"] == 0

    def test_mapping_key_naming_nothing_in_chunk_is_dropped_and_counted(
        self, tmp_path, monkeypatch
    ):
        """A local-mapping entry whose (canonicalized) key matches no node
        in this chunk names nothing here — it must be dropped rather than
        minted as a phantom Entity, and the drop must be counted.

        Mutation: skip the reconciliation and use the mapping as-is ->
        ``mapping_rekey_dropped`` stays 0 and a phantom entity is minted
        for a name absent from the chunk -> this test fails.
        """
        loop = _make_loop(tmp_path)
        graph = loop.merger.graph
        _populate_untyped_graph(graph, n_persons=10)

        monkeypatch.setattr(
            "paramem.training.consolidation.anonymize_with_local_model",
            lambda *args, **kwargs: (
                {"Someone Else": "Person_1"},
                "stub-anon-transcript",
                "stub-raw",
            ),
        )
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        call_spy = MagicMock()
        with patch("paramem.training.consolidation._graph_enrich_with_sota", call_spy):
            result = loop._run_graph_enrichment()

        assert not result["skipped"]
        assert result["mapping_rekey_dropped"] >= 1
        # "Someone Else" names nothing in the chunk -- chunk_mapping ends
        # up empty, tripping the pre-existing empty-mapping fail-closed
        # guard (leg 2), so no SOTA call fires at all.
        call_spy.assert_not_called()
        assert result["privacy_skipped_chunks"] >= 1

    def test_ambiguous_canonical_node_keys_are_both_dropped(self, tmp_path, monkeypatch):
        """Two distinct node keys that canonicalize identically are a real
        ambiguity — this should not arise in production (node keys are
        already canonical by construction, so ``canonical()`` is a no-op
        on them), but the reconciliation must fail closed rather than
        silently pick one: decision pinned here is DROP BOTH.
        """
        loop = _make_loop(tmp_path)
        graph = loop.merger.graph
        for key in ("Yang Ming", "yang ming"):
            graph.add_node(
                key,
                entity_type="concept",
                attributes={"name": key},
                reinforcement_count=10,
                sessions=["s000"],
                first_seen="s000",
                last_seen="s000",
            )
        graph.add_edge(
            "Yang Ming",
            "yang ming",
            predicate="knows",
            relation_type="social",
            confidence=1.0,
            source="extraction",
            sessions=["s000"],
        )
        for i in range(8):
            graph.add_node(
                f"filler{i}",
                entity_type="concept",
                attributes={},
                reinforcement_count=0,
                sessions=[],
                first_seen="",
                last_seen="",
            )

        monkeypatch.setattr(
            "paramem.training.consolidation.anonymize_with_local_model",
            lambda *args, **kwargs: ({"Yang Ming": "Person_1"}, "stub-anon-transcript", "stub-raw"),
        )
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        call_spy = MagicMock()
        with patch("paramem.training.consolidation._graph_enrich_with_sota", call_spy):
            result = loop._run_graph_enrichment()

        assert not result["skipped"]
        assert result["mapping_rekey_dropped"] >= 1
        call_spy.assert_not_called()


class TestGraphEnrichmentUsesSharedPrimitives:
    """Structural guard mirroring ``tests/test_extraction_pipeline_guard.py``:
    the graph-tier anonymization contract must route entirely through
    ``paramem.graph.placeholders`` — no second mint/table-build/deanon
    implementation may appear in ``paramem/training/consolidation.py``.
    """

    _PLACEHOLDER_PRIMITIVE_NAMES = frozenset(
        {
            "_build_anonymization_mapping",
            "_apply_bindings",
            "_normalize_anonymization_mapping",
            "_resolution_map",
            "mint_placeholder",
            "deanonymize_text",
            "_substitute_whole_words",
            "_declared_placeholder_tokens",
            "_contains_declared_token",
        }
    )

    def test_no_duplicate_primitive_defined_in_consolidation(self):
        """``consolidation.py`` must not define a function sharing a name
        with a ``placeholders.py`` primitive — that would be a duplicate
        mint/table/deanon implementation living outside the shared module.

        Mutation: add a function named e.g. ``_apply_bindings`` (or any
        other name in the set below) inside ``paramem/training/
        consolidation.py`` -> this test fails.
        """
        import ast
        from pathlib import Path

        repo_root = Path(__file__).resolve().parent.parent
        target = repo_root / "paramem" / "training" / "consolidation.py"
        tree = ast.parse(target.read_text())

        defined_names = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        collision = defined_names & self._PLACEHOLDER_PRIMITIVE_NAMES
        assert not collision, (
            "paramem/training/consolidation.py defines a function sharing a "
            f"name with a placeholders.py primitive: {sorted(collision)} — "
            "this is a duplicate mint/table/deanon implementation. Route "
            "through paramem.graph.placeholders instead."
        )

    def test_consolidation_does_not_reimplement_placeholder_shape_regex(self):
        """``consolidation.py`` must not hardcode its own placeholder-shape
        pattern (PascalCase_N) — that regex lives ONLY in
        ``paramem/graph/placeholders.py``.
        """
        from pathlib import Path

        repo_root = Path(__file__).resolve().parent.parent
        target = repo_root / "paramem" / "training" / "consolidation.py"
        text = target.read_text()
        assert "A-Z][A-Za-z]*_" not in text, (
            "consolidation.py appears to hardcode the placeholder-shape "
            "regex — this pattern must live only in paramem.graph.placeholders."
        )


class TestGraphEnrichmentFailureLoudness:
    """The graph tier must FAIL LOUD on a programming error and skip
    gracefully only on a genuine runtime condition.

    Two swallows used to hide a malformed prompt template: an inner
    ``except KeyError`` around ``enrichment_prompt.format(...)`` in
    ``_graph_enrich_with_sota``, and a broad ``except Exception`` around the
    whole chunk body in ``_run_graph_enrichment``.  Together they turned a
    missed brace-doubling in ``sota_graph_enrichment.txt`` into a permanent,
    SILENT outage of graph enrichment.  Both tests below are needed: each
    pins one half, so an inert half-fix is caught.
    """

    def test_graph_enrichment_prompt_format_error_propagates(self, tmp_path, monkeypatch):
        """A prompt template with an un-doubled literal brace raises
        ``KeyError`` out of ``_run_graph_enrichment`` — it does not silently
        disable enrichment.

        Mutation: re-add EITHER the inner ``except KeyError`` in
        ``_graph_enrich_with_sota`` OR the broad ``except Exception`` in
        ``_run_graph_enrichment`` -> the KeyError is swallowed and the fold
        "succeeds" with zero enrichment -> this test fails.
        """
        from paramem.graph.prompts import _load_prompt as _real_load_prompt

        loop = _make_loop(tmp_path)
        _populate_untyped_graph(loop.merger.graph)

        def _bad_prompt(filename, *args, **kwargs):
            if filename == "sota_graph_enrichment.txt":
                # {oops} is a literal brace the author forgot to double.
                return "Triples:\n{triples_json}\nSchema: {oops}"
            return _real_load_prompt(filename, *args, **kwargs)

        stub = _stub_local_model_types({})
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with (
            patch("paramem.training.consolidation.anonymize_with_local_model", side_effect=stub),
            patch("paramem.graph.extractor._load_prompt", side_effect=_bad_prompt),
            patch(
                "paramem.graph.extractor._sota_call",
                side_effect=AssertionError("the cloud must never be called with a broken prompt"),
            ),
            pytest.raises(KeyError, match="oops"),
        ):
            loop._run_graph_enrichment()

    def test_graph_enrichment_cuda_runtime_error_skips_the_chunk(self, tmp_path, monkeypatch):
        """A ``RuntimeError`` from the local ``generate()`` (the CUDA
        "device not ready" class) skips the chunk and lets the fold finish.

        This is the ONE genuinely-failing runtime leg in the chunk body, and
        it is the reason a handler exists at all.

        Mutation: narrow the handler to nothing (delete the ``except
        RuntimeError``) -> the RuntimeError kills the whole fold -> this
        test fails.
        """
        loop = _make_loop(tmp_path)
        _populate_untyped_graph(loop.merger.graph)

        def _boom(*args, **kwargs):
            raise RuntimeError("CUDA error: device not ready")

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with (
            patch("paramem.training.consolidation.anonymize_with_local_model", side_effect=_boom),
            patch(
                "paramem.graph.extractor._sota_call",
                side_effect=AssertionError("the cloud must not be called for a failed chunk"),
            ),
        ):
            result = loop._run_graph_enrichment()

        assert not result["skipped"]
        assert result["chunks"] == 0, "no SOTA call may be made for a chunk that failed locally"
        assert result["new_edges"] == 0

    def test_graph_tier_gates_on_mapping_not_facts(self, tmp_path, monkeypatch):
        """The graph tier gates parse-failure on ``_llm_mapping is
        None``, regardless of the (fail-closed, empty) ``anonymized_transcript``
        the anonymizer returns alongside it.

        Mutation: gate on the transcript instead of ``_llm_mapping is
        None`` -> the gate silently never fires -> this test fails.
        """
        loop = _make_loop(tmp_path)
        _populate_untyped_graph(loop.merger.graph)

        def _parse_failure(*args, **kwargs):
            return None, "", "not json"

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with (
            patch(
                "paramem.training.consolidation.anonymize_with_local_model",
                side_effect=_parse_failure,
            ),
            patch(
                "paramem.graph.extractor._sota_call",
                side_effect=AssertionError(
                    "the cloud must not be called for a chunk with no local mapping"
                ),
            ),
        ):
            result = loop._run_graph_enrichment()

        assert not result["skipped"]
        assert result["chunks"] == 0, "no SOTA call may be made when the mapping is None"
        assert result["new_edges"] == 0
