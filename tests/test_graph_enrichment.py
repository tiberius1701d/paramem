"""Tests for graph-level cloud enrichment (Task #10), plus the session-tier
enrichment incident-arbitration unit tests
(``TestArbitrateSessionEnrichmentIncidents``) — co-located here rather than
duplicated because both reuse the same ``_make_loop`` fixture.

All tests are pure-Python — no GPU required. Cloud calls are mocked so
the test suite does not make any network requests.
"""

from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock, patch

import networkx as nx
import pytest
from peft import PeftModel

from paramem.cloud.anonymize import anonymize_transcript as _real_anonymize_transcript
from paramem.graph.schema import SessionGraph
from paramem.memory.persistence import _EDGE_SOURCE_ATTR
from paramem.training.consolidation import ConsolidationLoop
from paramem.training.graph_enrich import serialize_subgraph_triples
from paramem.training.graph_tier import GraphTierRefiner
from paramem.utils.config import AdapterConfig, ConsolidationConfig, TrainingConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _stub_local_anonymize(monkeypatch):
    """Default stub for ``anonymize_transcript``.

    ``graph_enrich.enrich_graph`` now runs the local anonymizer
    (the SAME primitive session-tier extraction uses) over each chunk
    BEFORE the cloud call, to derive real-name entity types the fold graph
    itself cannot supply (see that function's docstring). ``_make_loop``'s
    model/tokenizer are ``MagicMock()``s, so a real call always fails to
    parse (there is no JSON in a ``MagicMock``'s generated output), which
    would fail every chunk closed (skip the cloud call entirely) before
    ``request_graph_enrichment`` is ever reached — breaking every test below
    that mocks ``request_graph_enrichment`` to verify cloud-response
    consumption.

    The default therefore lands on the SAFE side rather than the unsafe
    one: :func:`_stub_local_model_types` with an empty override dict, which
    types every non-speaker name found in the chunk's relations as
    ``"person"`` (masked), so tests below exercise the masked-payload path
    by default. A genuinely EMPTY mapping (``{}``) now PROCEEDS (the
    anonymizer ran and found nothing in scope, a legitimate verdict, not
    a failure) — see
    ``TestEmptyMappingProceeds``. ``graph_enrich.enrich_graph``'s
    remaining fail-closed guard (leg 2) only fires when the local
    anonymizer DID name something but none of it survived reconciliation
    onto the chunk's actual node keys (a genuine classification/identity-match
    failure) — see ``TestPrivacyFailClosedOnReconciliationFailure``. Tests
    in ``TestGraphTierAnonymizationContract`` call
    ``request_graph_enrichment`` directly (never through
    ``graph_enrich.enrich_graph``) and are unaffected by this
    fixture.
    """
    monkeypatch.setattr(
        "paramem.cloud.anonymize.anonymize_transcript",
        _stub_local_model_types({}),
    )


def _make_loop(tmp_path, **kwargs) -> ConsolidationLoop:
    """Build a minimal ConsolidationLoop for enrichment tests.

    Graph is transient (RAM-only). Model/tokenizer are mocks so no GPU
    is touched.  The mock model pre-populates ``peft_config`` with all
    three required adapters so ``ensure_adapters`` skips the real PEFT
    ``create_adapter`` calls.

    Keyword args forwarded to ConsolidationLoop override the defaults
    set here (e.g. pass ``extraction_enrichment_provider=""`` to test the
    no-provider skip path).
    """
    # __class__ = PeftModel so ensure_adapters' isinstance check
    # short-circuits without restricting the mock's attribute surface.
    model = MagicMock()
    model.__class__ = PeftModel
    model.peft_config = {
        "episodic": MagicMock(),
        "semantic": MagicMock(),
        "in_training": MagicMock(),
    }

    defaults = dict(
        extraction_enrichment_provider="anthropic",
        extraction_enrichment_provider_model="claude-sonnet-4-6",
        extraction_scrub={"person name"},
        # Graph-tier enrichment is cloud egress and now routes through the
        # shared cloud-admission verdict, whose first term is the master
        # switch — so it must be ON for any enrichment test to reach a call.
        cloud_enabled=True,
        # Required keywords (no code-side default) — see kwargs docstring.
        extraction_max_tokens=8192,
        extraction_plausibility_max_tokens=8192,
        extraction_anonymize_token_envelope=8192,
    )
    defaults.update(kwargs)

    # replay_enabled controls whether run_consolidation_cycle's registry guard
    # fires.  Callers that need enrichment hooks to fire pass replay_enabled=True.
    replay_enabled = defaults.pop("replay_enabled", False)
    # Allow callers to supply a pre-built ConsolidationConfig so tests can set
    # fields like refinement_enrichment without touching other knobs.
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


def _refiner_for(loop: ConsolidationLoop) -> GraphTierRefiner:
    """Build a :class:`GraphTierRefiner` off a loop's current live state.

    Mirrors exactly what ``ConsolidationLoop._refine_consolidation_graph``
    constructs on every call — the enrichment and normalization surfaces
    moved off ``ConsolidationLoop`` onto ``GraphTierRefiner`` (the deleted
    enrichment/normalization SHIM methods that used to live directly on
    ``ConsolidationLoop``), so tests exercise ``run_enrichment()`` /
    ``run_normalization()`` on a
    refiner built from the loop rather than calling a loop method directly.
    Called fresh at each use site so a test that mutates ``loop.model`` (or
    other loop state) between calls sees the update, matching production's
    read-fresh-per-call semantics.
    """
    return GraphTierRefiner(
        loop.merger,
        model=loop.model,
        tokenizer=loop.tokenizer,
        extraction_config_provider=loop._current_extraction_config,
        cloud_enabled=loop.cloud_enabled,
        neighborhood_hops=loop.graph_enrichment_neighborhood_hops,
        max_entities_per_pass=loop.graph_enrichment_max_entities_per_pass,
        gc_disable=loop._disable_gradient_checkpointing,
        gc_enable=loop._enable_gradient_checkpointing,
    )


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


def _populate_disjoint_clusters(
    graph: nx.MultiDiGraph, n_clusters: int = 3, leaves_per_cluster: int = 3
) -> None:
    """Build *n_clusters* disconnected hub-and-leaves stars.

    Each cluster is its own connected component (no cross-cluster edges), so
    with ``neighborhood_hops=1`` and ``max_entities_per_pass ==
    leaves_per_cluster + 1`` (a whole cluster, no trim), every cluster maps
    to exactly one chunk with NO node overlap between chunks — unlike
    ``_populate_graph``'s single hub topology, where every ego-graph
    includes the shared hub. Hub reinforcement counts are strictly
    descending (cluster 0 highest) so ``nodes_by_recurrence`` visits the
    hubs in cluster order, making chunk order deterministic: chunk *i* is
    always cluster *i*.

    Used by the multi-chunk VRAM-degrade tests, which need to assert that a
    fault on chunk 2 of 3 keeps chunk 1's already-merged relations and never
    reaches chunk 3.
    """
    for c in range(n_clusters):
        hub = f"hub{c}"
        graph.add_node(
            hub,
            entity_type="organization",
            attributes={"name": f"Hub{c}"},
            reinforcement_count=10_000 - c,
            sessions=[f"s{c}00"],
            first_seen=f"s{c}00",
            last_seen=f"s{c}00",
        )
        for j in range(leaves_per_cluster):
            person = f"c{c}_person{j}"
            graph.add_node(
                person,
                entity_type="person",
                attributes={"name": f"Cluster{c}Person{j}"},
                reinforcement_count=1,
                sessions=[f"s{c}{j:02d}"],
                first_seen=f"s{c}{j:02d}",
                last_seen=f"s{c}{j:02d}",
            )
            graph.add_edge(
                person,
                hub,
                predicate="works at",
                relation_type="factual",
                confidence=1.0,
                source="extraction",
                sessions=[f"s{c}{j:02d}"],
            )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAnonymizeTokenEnvelopeFunnel:
    """A ``ConsolidationLoop`` built with
    ``extraction_anonymize_token_envelope=`` yields that value at
    ``_current_extraction_config().anonymize_token_envelope`` — the exact
    field :func:`~paramem.training.graph_enrich.enrich_graph` reads as
    ``ext_cfg.anonymize_token_envelope``.
    """

    def test_ctor_kwarg_reaches_current_extraction_config(self, tmp_path):
        loop = _make_loop(tmp_path, extraction_anonymize_token_envelope=1234)
        assert loop._current_extraction_config().anonymize_token_envelope == 1234

    def test_ctor_default_matches_module_default(self, tmp_path):
        from paramem.cloud.anonymize import _DEFAULT_ANONYMIZER_TOKEN_ENVELOPE

        loop = _make_loop(tmp_path)
        assert (
            loop._current_extraction_config().anonymize_token_envelope
            == _DEFAULT_ANONYMIZER_TOKEN_ENVELOPE
        )


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
            0,  # accepted: no relations dropped
        )
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with patch(
            "paramem.training.graph_enrich.request_graph_enrichment",
            return_value=canned_result,
        ):
            result = _refiner_for(loop).run_enrichment()

        assert not result["skipped"]
        assert result["new_edges"] >= 1

        # Verify the added edge carries source="graph_enrichment".
        # Nodes are canonical-keyed; predicate is stored in canonical form too
        # ("colleague_of" → "colleague of" after the canonical() blank-fold).
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
            0,  # accepted: no relations dropped
        )
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with patch(
            "paramem.training.graph_enrich.request_graph_enrichment",
            return_value=canned_result,
        ):
            result = _refiner_for(loop).run_enrichment()

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
            0,  # accepted: no relations dropped
        )
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with patch(
            "paramem.training.graph_enrich.request_graph_enrichment",
            return_value=canned_result,
        ):
            result = _refiner_for(loop).run_enrichment()

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

        # cloud returns surface names; production canonicalizes them before lookup:
        # "Alice" -> "alice", "Alicia" -> "alicia".
        canned_result = (
            [],  # no new relations
            [["Alice", "Alicia"]],  # same_as
            "raw",
            0,  # accepted: no relations dropped
        )
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with patch(
            "paramem.training.graph_enrich.request_graph_enrichment",
            return_value=canned_result,
        ):
            result = _refiner_for(loop).run_enrichment()

        assert not result["skipped"]
        assert result["same_as_merges"] >= 1
        # "alicia" should be contracted into "alice" — removed as a distinct node
        assert "alicia" not in graph.nodes
        assert "alice" in graph.nodes


class TestSafeToMergeSurface:
    """Unit coverage for the surname / surface-form safety gate."""

    def test_token_subset_accepted(self):
        from paramem.training.graph_enrich import _safe_to_merge_surface

        # Honorific-stripped subset
        assert _safe_to_merge_surface("Mr. Yang", "Yang Ming") is True
        # Given-name subset of full name
        assert _safe_to_merge_surface("Ming", "Yang Ming") is True
        # Identical after honorific strip
        assert _safe_to_merge_surface("Dr. Smith", "Smith") is True

    def test_different_surnames_rejected(self):
        from paramem.training.graph_enrich import _safe_to_merge_surface

        # Shared given name, different family name — must NOT merge
        assert _safe_to_merge_surface("Zhang Min", "Wang Min") is False
        assert _safe_to_merge_surface("Li Wei", "Chen Wei") is False

    def test_jw_fallback_accepts_minor_typos(self):
        from paramem.training.graph_enrich import _safe_to_merge_surface

        # True variant (one letter off) passes the JW fallback
        assert _safe_to_merge_surface("Catherine Holmes", "Katherine Holmes") is True

    def test_empty_and_all_honorific_rejected(self):
        from paramem.training.graph_enrich import _safe_to_merge_surface

        assert _safe_to_merge_surface("", "Alice") is False
        assert _safe_to_merge_surface("Mr.", "Dr.") is False


class TestSameAsSurnameMismatchRejected:
    """Integration: a bad same_as pair from cloud must be rejected by the gate."""

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

        canned_result = ([], [["Zhang Min", "Wang Min"]], "raw", 0)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with patch(
            "paramem.training.graph_enrich.request_graph_enrichment",
            return_value=canned_result,
        ):
            result = _refiner_for(loop).run_enrichment()

        assert not result["skipped"]
        # Gate must reject — neither node should be contracted
        assert result["same_as_merges"] == 0
        assert "Zhang Min" in graph.nodes
        assert "Wang Min" in graph.nodes


def _same_as_per_chunk(*per_chunk: list[list[str]]):
    """Build a ``request_graph_enrichment`` side_effect yielding one result per chunk.

    Chunk *i* receives ``per_chunk[i]`` as its ``same_as`` pair list; every chunk
    beyond the supplied sequence receives an empty list.  Used by the tests that
    need DIFFERENT chunks to propose the same entity pair under different surface
    forms, which a single ``return_value`` (identical for every chunk) cannot
    express.  Returns the callable; the caller reads ``call_count`` off the patch
    to assert the multi-chunk path was genuinely exercised.
    """
    calls = {"n": 0}

    def _side_effect(*_args, **_kwargs):
        i = calls["n"]
        calls["n"] += 1
        pairs = list(per_chunk[i]) if i < len(per_chunk) else []
        return ([], pairs, "raw", 0)

    return _side_effect


class TestSameAsDedupAcrossChunks:
    """Cross-chunk same_as proposal handling.

    Two properties, and they pull in opposite directions:

    * A pair already contracted must not contract again (rule 1 — the dropped
      node is gone from the live graph).
    * A pair REJECTED by the surface gate on one chunk's surfaces must still be
      re-evaluated on another chunk's surfaces, because the gate reads the
      surfaces and overlapping ego-graph chunks supply different ones.
    """

    def test_gate_reevaluated_per_chunk_surfaces(self, tmp_path, monkeypatch):
        """A gate rejection on one chunk must not suppress a later chunk's proposal.

        Both chunks propose the same canonical pair (``yang ming`` / ``zhang min``)
        but under different surface forms.  ``_safe_to_merge_surface`` tokenizes on
        whitespace, so the two forms land on OPPOSITE sides of the gate:

        * ``("Yang Ming", "Zhang Min")`` — two-token symmetric difference, rejected.
        * ``("Yang-Ming", "Zhang-Min")`` — one token each, Jaro-Winkler ≥ 0.85,
          accepted.

        The rejecting form is proposed FIRST.  Any proposal memo keyed on node
        identity (or on a casefolded surface) is coarser than the gate and would
        make chunk 1's rejection permanent, so the contraction would never happen.
        """
        loop = _make_loop(tmp_path, graph_enrichment_max_entities_per_pass=3)
        graph = loop.merger.graph
        _populate_graph(graph, n_persons=10)

        for key, display in (("yang ming", "Yang Ming"), ("zhang min", "Zhang Min")):
            graph.add_node(
                key,
                entity_type="person",
                attributes={"name": display},
                reinforcement_count=3,
                sessions=["s040"],
                first_seen="s040",
                last_seen="s040",
            )

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with patch(
            "paramem.training.graph_enrich.request_graph_enrichment",
            side_effect=_same_as_per_chunk(
                [["Yang Ming", "Zhang Min"]],  # chunk 1 — gate REJECTS
                [["Yang_Ming", "Zhang_Min"]],  # chunk 2 — gate ACCEPTS
            ),
        ) as mock_cloud:
            result = _refiner_for(loop).run_enrichment()

        assert not result["skipped"]
        # Guard against a vacuous pass: the second chunk must actually have run.
        assert mock_cloud.call_count >= 2
        # Chunk 2's accepted proposal must contract despite chunk 1's rejection.
        assert result["same_as_merges"] == 1
        assert "zhang min" not in graph.nodes
        assert "yang ming" in graph.nodes

    def test_identical_proposal_across_chunks_contracts_once(self, tmp_path, monkeypatch):
        """The same proposal repeated across chunks contracts exactly once.

        Removing the proposal memo must not introduce double-counting: the second
        chunk's identical proposal is stopped by the live-graph check, because the
        first contraction already removed the dropped node.  Asserts both the
        ``same_as_merges`` counter and the removal ledger stay at one application.
        """
        from paramem.memory.persistence import _IK_KEY_ATTR

        loop = _make_loop(tmp_path, graph_enrichment_max_entities_per_pass=3)
        graph = loop.merger.graph
        _populate_graph(graph, n_persons=10)

        for key, display in (("yang ming", "Yang Ming"), ("mr. yang", "Mr. Yang")):
            graph.add_node(
                key,
                entity_type="person",
                attributes={"name": display},
                reinforcement_count=3,
                sessions=["s050"],
                first_seen="s050",
                last_seen="s050",
            )
        # Keyed edge between the pair: contraction drops it as a self-loop and
        # records its ik_key, giving the ledger observable content to assert on.
        eid = graph.add_edge(
            "yang ming",
            "mr. yang",
            predicate="same as",
            relation_type="factual",
            confidence=1.0,
            source="extraction",
            sessions=["s050"],
        )
        graph["yang ming"]["mr. yang"][eid][_IK_KEY_ATTR] = "key_yang_victim"

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with patch(
            "paramem.training.graph_enrich.request_graph_enrichment",
            side_effect=_same_as_per_chunk(
                [["Yang Ming", "Mr. Yang"]],  # chunk 1 — gate accepts, contracts
                [["Yang Ming", "Mr. Yang"]],  # chunk 2 — identical, must be inert
            ),
        ) as mock_cloud:
            result = _refiner_for(loop).run_enrichment()

        assert not result["skipped"]
        assert mock_cloud.call_count >= 2
        assert result["same_as_merges"] == 1
        assert "mr. yang" not in graph.nodes
        assert "yang ming" in graph.nodes
        # The repeated proposal must not inflate the ledger: exactly one
        # same_as removal, pointing at the keep node.
        same_as_keys = [
            k
            for k, e in loop.merger.removal_ledger.items()
            if e.get("reason") == "enrichment_same_as"
        ]
        assert same_as_keys == ["key_yang_victim"]
        assert loop.merger.removal_ledger["key_yang_victim"]["keep_node"] == "yang ming"

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

        # cloud returns surface names; production canonicalizes before graph lookup.
        # Same pair emitted twice in reversed order — simulates cloud echoing
        # the duplicate across chunks.
        canned_result = (
            [],
            [["Yang Ming", "Mr. Yang"], ["Mr. Yang", "Yang Ming"]],
            "raw",
            0,
        )
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with patch(
            "paramem.training.graph_enrich.request_graph_enrichment",
            return_value=canned_result,
        ):
            result = _refiner_for(loop).run_enrichment()

        assert not result["skipped"]
        assert result["same_as_merges"] == 1


class TestSymmetricPredicateCanonicalized:
    """Symmetric predicates collapse via symmetric-direction canonicalization in the merger."""

    def test_both_directions_collapse_to_one_edge(self, tmp_path, monkeypatch):
        """When cloud emits (A,P,B) and (B,P,A) both with symmetric=true and
        neither endpoint is a speaker, the merger swaps the endpoints of the
        subject > object direction so both land on one canonical subj < obj edge —
        the second insert is a Case-1 duplicate.
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

        canned_result = (rels, [], "raw", 0)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with patch(
            "paramem.training.graph_enrich.request_graph_enrichment",
            return_value=canned_result,
        ):
            result = _refiner_for(loop).run_enrichment()

        assert not result["skipped"]
        # After the merger's symmetric-endpoint swap: both become
        # (xiaoxiu, colleague of, zhang).
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

        canned_result = (rels, [], "raw", 0)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with patch(
            "paramem.training.graph_enrich.request_graph_enrichment",
            return_value=canned_result,
        ):
            result = _refiner_for(loop).run_enrichment()

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

        # cloud response: same_as merges Alex→Alexander (cloud returns surface names;
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
            "paramem.training.graph_enrich.request_graph_enrichment",
            return_value=(canned_rels, canned_same_as, "raw", 0),
        ):
            result = _refiner_for(loop).run_enrichment()

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
    """Graphs with fewer than 10 nodes must be skipped without a cloud call."""

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
        with patch("paramem.training.graph_enrich.request_graph_enrichment", call_spy):
            result = _refiner_for(loop).run_enrichment()

        assert result["skipped"] is True
        assert result["skip_reason"] == "floor"
        call_spy.assert_not_called()


class TestDisabledIsNoop:
    """enrich=False must not call GraphTierRefiner.run_enrichment at all."""

    def test_no_change_when_enrich_false(self, tmp_path, monkeypatch):
        """_refine_consolidation_graph(enrich=False) must not call run_enrichment
        and must leave the graph byte-identical."""
        loop = _make_loop(
            tmp_path, consolidation_config=ConsolidationConfig(refinement_enrichment="off")
        )
        graph = loop.merger.graph
        _populate_graph(graph, n_persons=10)

        # Snapshot pre-state
        pre_nodes = set(graph.nodes)
        pre_edges = set((u, v) for u, v, _ in graph.edges(data=True))

        with patch.object(GraphTierRefiner, "run_enrichment") as enrich_spy:
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
            0,  # accepted: no relations dropped
        )
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with patch(
            "paramem.training.graph_enrich.request_graph_enrichment",
            return_value=canned_result,
        ):
            _refiner_for(loop).run_enrichment()

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
    """Each cloud call payload must not exceed max_entities_per_pass nodes."""

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

        def _spy_call(payload, *args, **kwargs):
            call_args_list.append(list(payload.facts))
            return ([], [], "raw", 0)

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with patch("paramem.training.graph_enrich.request_graph_enrichment", side_effect=_spy_call):
            _refiner_for(loop).run_enrichment()

        assert call_args_list, "Expected at least one cloud call"

        # Gather the unique node names seen in each call's triples
        for triples in call_args_list:
            nodes_in_call = set()
            for t in triples:
                nodes_in_call.add(t["subject"])
                nodes_in_call.add(t["object"])
            assert len(nodes_in_call) <= 10 + 1, (
                f"Chunk exceeded cap: {len(nodes_in_call)} nodes (cap=10, +1 tolerance for hub)"
            )


class TestCloudEgressRefusedSkipsGracefully:
    """Every unmet cloud-admission term skips this pass with no crash.

    The three terms are checked by the ONE shared component
    (:func:`paramem.cloud.admission.evaluate_cloud_egress`), so they
    share one ``skip_reason`` token; the individual unmet terms go to the
    log, which is what these tests assert on.
    """

    def test_missing_key_skip(self, tmp_path, monkeypatch, caplog):
        loop = _make_loop(tmp_path)
        graph = loop.merger.graph
        _populate_graph(graph, n_persons=10)

        # Remove the key from the environment
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        call_spy = MagicMock()
        caplog.set_level(logging.WARNING, logger="paramem.training.graph_enrich")
        with patch("paramem.training.graph_enrich.request_graph_enrichment", call_spy):
            result = _refiner_for(loop).run_enrichment()

        assert result["skipped"] is True
        assert result["skip_reason"] == "cloud_egress_blocked"
        assert "ANTHROPIC_API_KEY env var is unset" in caplog.text
        call_spy.assert_not_called()

    def test_no_provider_skip(self, tmp_path, monkeypatch, caplog):
        loop = _make_loop(tmp_path, extraction_enrichment_provider="")
        graph = loop.merger.graph
        _populate_graph(graph, n_persons=10)

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        call_spy = MagicMock()
        caplog.set_level(logging.WARNING, logger="paramem.training.graph_enrich")
        with patch("paramem.training.graph_enrich.request_graph_enrichment", call_spy):
            result = _refiner_for(loop).run_enrichment()

        assert result["skipped"] is True
        assert result["skip_reason"] == "cloud_egress_blocked"
        assert "no cloud provider configured" in caplog.text
        call_spy.assert_not_called()

    def test_master_switch_off_skip(self, tmp_path, monkeypatch, caplog):
        """``cloud.enabled: false`` alone blocks graph-tier cloud egress —
        the master switch is a term of the shared verdict, so this pass can
        no longer egress behind the operator's back."""
        loop = _make_loop(tmp_path, cloud_enabled=False)
        graph = loop.merger.graph
        _populate_graph(graph, n_persons=10)

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        call_spy = MagicMock()
        caplog.set_level(logging.WARNING, logger="paramem.training.graph_enrich")
        with patch("paramem.training.graph_enrich.request_graph_enrichment", call_spy):
            result = _refiner_for(loop).run_enrichment()

        assert result["skipped"] is True
        assert result["skip_reason"] == "cloud_egress_blocked"
        assert "cloud.enabled is off" in caplog.text
        call_spy.assert_not_called()


class TestNoModelSkipsGracefully:
    """self.model is None must early-return, mirroring
    ``GraphTierRefiner.run_normalization``'s existing "no_model" guard
    rather than crashing inside _disable_gradient_checkpointing.

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
        with patch("paramem.training.graph_enrich.request_graph_enrichment", call_spy):
            result = _refiner_for(loop).run_enrichment()

        assert result["skipped"] is True
        assert result["skip_reason"] == "no_model"
        call_spy.assert_not_called()


class TestVramExhaustedDegradesGracefully:
    """A VramExhausted mid-pass degrades gracefully instead of aborting the
    fold.  ``VramExhausted`` must NOT escape ``run_enrichment()``: there is
    no retry that could help (a retry in the same VRAM state faults the same
    way).  The pass stops, keeps whatever it already merged, and the caller
    (``ConsolidationLoop._refine_consolidation_graph``) records an incident.
    Enrichment self-heals next cycle — the pass runs over the cumulative
    graph every fold.

    The fault is raised from the ``anonymize`` leg — the only leg in the
    chunk body that touches the GPU, and so the only one where
    ``VramExhausted`` can originate in production.  Raising it from
    ``request_graph_enrichment`` instead would exercise a branch production
    cannot reach: the cloud leg does no GPU work.

    Mutation: revert the ``except VramExhausted`` branch to ``raise`` ->
    this test's ``run_enrichment()`` call raises instead of returning a
    degrade result that keeps the first chunk's merged relations.
    """

    def test_vram_exhausted_stops_pass_keeps_completed_chunks(self, tmp_path, monkeypatch):
        from paramem.utils.vram_guard import VramExhausted

        loop = _make_loop(
            tmp_path,
            graph_enrichment_max_entities_per_pass=4,
            graph_enrichment_neighborhood_hops=1,
        )
        graph = loop.merger.graph
        _populate_disjoint_clusters(graph, n_clusters=3, leaves_per_cluster=3)

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        calls = {"n": 0}

        def _anonymize_side_effect(facts, *args, **kwargs):
            i = calls["n"]
            calls["n"] += 1
            if i == 1:
                # Fault on chunk 2 of 3 (0-indexed) — the leg that actually
                # touches the GPU, matching production (VramExhausted
                # originates inside anonymize_transcript's generate() call,
                # wrapped by vram_scope inside `anonymize`).
                raise VramExhausted("graph_enrichment_test")
            payload, _ = _payload_and_graph_for(facts, {})
            return payload

        canned_cloud_result = (
            [
                {
                    "subject": "hub0",
                    "predicate": "hub_partner_of",
                    "object": "c0_person0",
                    "relation_type": "social",
                    "confidence": 0.9,
                }
            ],
            [],  # no same_as
            "raw",
            0,  # no relations dropped
        )
        with (
            patch(
                "paramem.training.graph_enrich.anonymize",
                side_effect=_anonymize_side_effect,
            ),
            patch(
                "paramem.training.graph_enrich.request_graph_enrichment",
                return_value=canned_cloud_result,
            ) as call_spy,
        ):
            result = _refiner_for(loop).run_enrichment()

        assert result["skipped"] is False
        assert result["aborted_reason"] == "vram"
        # Only chunk 1 (cluster 0) reached the cloud call before the fault
        # on chunk 2 stopped the pass; chunk 3 (cluster 2) is never reached.
        assert result["chunks"] == 1
        call_spy.assert_called_once()
        # Chunk 1's enrichment relation is still merged into the graph —
        # completed chunks keep their work (degrade granularity is the CHUNK).
        assert result["new_edges"] >= 1


class TestEmptyMappingProceeds:
    """A local anonymizer mapping that comes back completely EMPTY
    (``{}``) — even for a chunk with real (non-speaker) node names —
    means the anonymizer ran and classified
    NOTHING in scope against ``scrub``. That is a legitimate verdict, not
    a classification failure, so egress PROCEEDS: the cloud call fires
    with an empty ``chunk_mapping`` (nothing to substitute) rather than
    being skipped. Mirrors the session-tier ``mapping == {}`` proceed
    path in :func:`~paramem.graph.flows.anonymize_turn`.

    Mutation: reintroduce a bare ``not chunk_mapping`` guard (dropping the
    ``_llm_mapping and`` qualifier) in ``graph_enrich.enrich_graph``
    -> this test fails (the cloud call is skipped and
    ``privacy_skipped_chunks`` increments instead of staying 0).
    """

    def test_empty_mapping_proceeds_with_cloud_call(self, tmp_path, monkeypatch):
        loop = _make_loop(tmp_path)
        graph = loop.merger.graph
        _populate_graph(graph, n_persons=10)

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        # Deliberately override the (masked-by-default) autouse fixture
        # with a genuinely empty-mapping stub — this test exists
        # specifically to exercise that proceed path.
        monkeypatch.setattr(
            "paramem.cloud.anonymize.anonymize_transcript",
            lambda *args, **kwargs: ({}, "", "stub-raw"),
        )
        canned_result = ([], [], "raw", 0)
        with patch(
            "paramem.training.graph_enrich.request_graph_enrichment",
            return_value=canned_result,
        ) as call_spy:
            result = _refiner_for(loop).run_enrichment()

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
            "paramem.cloud.anonymize.anonymize_transcript",
            lambda *args, **kwargs: ({}, "", "stub-raw"),
        )
        canned_result = ([], [], "raw", 0)
        with patch(
            "paramem.training.graph_enrich.request_graph_enrichment",
            return_value=canned_result,
        ) as call_spy:
            result = _refiner_for(loop).run_enrichment()

        assert not result["skipped"]
        assert result["privacy_skipped_chunks"] == 0
        call_spy.assert_called()


class TestPrivacyFailClosedOnReconciliationFailure:
    """Leg 2 — a local anonymizer mapping that NAMED
    something but NONE of it survived reconciliation onto the
    chunk's actual node keys, while the chunk has real (non-speaker)
    node names, is a classification/identity-match failure — distinct
    from a genuinely EMPTY mapping (see ``TestEmptyMappingProceeds``,
    which now proceeds). ``canonical()``
    matching (tested directly in ``tests/test_placeholders.py::
    TestSubstituteWholeWordsCanonicalMatching``) cannot fix this leg —
    the named entity simply isn't one of this chunk's nodes.

    Mutation: drop the ``_llm_mapping and`` qualifier from the
    ``graph_enrich.enrich_graph`` guard -> this test's cloud call
    would no longer be distinguished from the empty-mapping proceed case
    and would fire instead of being skipped.
    """

    def test_reconciliation_drops_everything_skips_cloud_call(self, tmp_path, monkeypatch):
        loop = _make_loop(tmp_path)
        graph = loop.merger.graph
        _populate_graph(graph, n_persons=10)

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        # The local anonymizer DID name something — but a name matching
        # none of this chunk's actual node keys, so reconciliation
        # drops it, leaving chunk_mapping empty despite a non-empty
        # _llm_mapping.
        monkeypatch.setattr(
            "paramem.cloud.anonymize.anonymize_transcript",
            lambda *args, **kwargs: (
                {"nobody in this chunk": "Person_1"},
                "stub-anon",
                "stub-raw",
            ),
        )
        call_spy = MagicMock()
        with patch("paramem.training.graph_enrich.request_graph_enrichment", call_spy):
            result = _refiner_for(loop).run_enrichment()

        assert not result["skipped"]
        assert result["privacy_skipped_chunks"] >= 1
        assert result["new_edges"] == 0
        assert result["same_as_merges"] == 0
        call_spy.assert_not_called()


class TestPrivacyFailClosedOnShapeValidationDrop:
    """The domain-scoped fail-closed guard fires whether
    the mapping was emptied by node-key reconciliation (see
    ``TestPrivacyFailClosedOnReconciliationFailure`` above) OR by the
    model's own placeholder-shape validation dropping the entry BEFORE
    reconciliation ever runs (e.g. a 7B emitting a lowercase placeholder
    that fails ``PLACEHOLDER_SHAPE_RE`` on both sides). ``rekey_dropped``
    stays ``0`` in this case — the reconciliation loop never sees an
    entry ``_normalize_anonymization_mapping`` already removed — so the
    caller must discriminate the cause via ``payload.failure``, not
    ``payload.rekey_dropped``. Both causes are counted in
    ``privacy_skipped_chunks`` identically.

    Mutation: branch ``graph_enrich.enrich_graph`` on
    ``payload.rekey_dropped`` instead of ``payload.failure == "guard"`` ->
    this test's chunk is misclassified as a parse failure and
    ``privacy_skipped_chunks`` stays at 0 (undercount).
    """

    def test_shape_validation_drop_still_counted_in_privacy_skipped_chunks(
        self, tmp_path, monkeypatch
    ):
        loop = _make_loop(tmp_path)
        graph = loop.merger.graph
        _populate_graph(graph, n_persons=10)

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        # The local anonymizer DID name a real node in this chunk
        # ("Person0") but with a lowercase placeholder value that fails
        # the shape regex on both sides — dropped entirely by
        # ``_normalize_anonymization_mapping``, never reaching the
        # reconciliation loop that increments ``rekey_dropped``.
        monkeypatch.setattr(
            "paramem.cloud.anonymize.anonymize_transcript",
            lambda *args, **kwargs: (
                {"Person0": "person_1"},
                "stub-anon",
                "stub-raw",
            ),
        )
        call_spy = MagicMock()
        with patch("paramem.training.graph_enrich.request_graph_enrichment", call_spy):
            result = _refiner_for(loop).run_enrichment()

        assert not result["skipped"]
        assert result["privacy_skipped_chunks"] >= 1
        assert result["new_edges"] == 0
        assert result["same_as_merges"] == 0
        call_spy.assert_not_called()


class TestGuardDomainSeparation:
    """The domain-scoped fail-closed guard is derived from
    ``graph.relations``' subject/object endpoints — NOT from
    ``identity_domain`` (``chunk_nodes``).  A chunk whose (larger,
    pre-edge-trim) ``chunk_nodes`` still lists a non-speaker node that has
    NO surviving edge in the trimmed subgraph must NOT trip the guard,
    even when the local anonymizer named something that reconciles to
    nothing.

    Mutation: fuse the guard domain onto ``identity_domain``/``chunk_nodes``
    instead of ``graph.relations`` -> this chunk's guard fires (status
    flips to "failed", ``privacy_skipped_chunks`` increments, the cloud
    call is skipped) even though every SURVIVING edge is speaker-only.
    """

    def test_guard_does_not_fire_when_surviving_edges_are_speaker_only(self, tmp_path, monkeypatch):
        loop = _make_loop(
            tmp_path,
            graph_enrichment_max_entities_per_pass=3,
            graph_enrichment_neighborhood_hops=2,
        )
        graph = loop.merger.graph

        for sid in ("speaker0", "speaker1"):
            graph.add_node(
                sid,
                entity_type="person",
                speaker_id=sid,
                attributes={"name": sid},
                reinforcement_count=100,
                sessions=["s000"],
                first_seen="s000",
                last_seen="s000",
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
        # A low-degree bridge node — connects the focal speaker0 to X, but
        # is itself low-degree so the trim below drops IT, not X.
        graph.add_node(
            "bridge",
            entity_type="concept",
            attributes={"name": "Bridge"},
            reinforcement_count=1,
            sessions=[],
            first_seen="",
            last_seen="",
        )
        graph.add_edge(
            "speaker0",
            "bridge",
            predicate="related_to",
            relation_type="factual",
            confidence=1.0,
            source="extraction",
            sessions=["s000"],
        )
        graph.add_node(
            "x_node",
            entity_type="concept",
            attributes={"name": "XNode"},
            reinforcement_count=1,
            sessions=[],
            first_seen="",
            last_seen="",
        )
        graph.add_edge(
            "bridge",
            "x_node",
            predicate="related_to",
            relation_type="factual",
            confidence=1.0,
            source="extraction",
            sessions=["s000"],
        )
        # Boost speaker1's and x_node's full-graph degree above bridge's
        # (degree 2: speaker0, x_node) so the top-2-by-degree trim keeps
        # {speaker1, x_node} and drops {bridge} — leaving x_node IN
        # chunk_nodes with NO surviving edge (its only edge was to the
        # now-excluded bridge).
        for i in range(4):
            filler = f"s1_filler{i}"
            graph.add_node(
                filler,
                entity_type="concept",
                attributes={"name": filler},
                reinforcement_count=0,
                sessions=[],
                first_seen="",
                last_seen="",
            )
            graph.add_edge(
                "speaker1",
                filler,
                predicate="related_to",
                relation_type="factual",
                confidence=1.0,
                source="extraction",
                sessions=["s000"],
            )
        for i in range(3):
            filler = f"x_filler{i}"
            graph.add_node(
                filler,
                entity_type="concept",
                attributes={"name": filler},
                reinforcement_count=0,
                sessions=[],
                first_seen="",
                last_seen="",
            )
            graph.add_edge(
                "x_node",
                filler,
                predicate="related_to",
                relation_type="factual",
                confidence=1.0,
                source="extraction",
                sessions=["s000"],
            )

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        # Several chunks form (one per high-reinforcement focal node); only
        # the speaker0/speaker1 chunk (surviving edges == speaker-only,
        # after x_node's bridge is trimmed) is the one under test — the
        # local anonymizer names something that matches NEITHER
        # speaker0/speaker1 NOR x_node for THAT chunk, so reconciliation
        # drops it, leaving chunk_mapping empty (rekey_dropped >= 1).
        # Under the CORRECT (relation-endpoints) guard domain this must
        # NOT fire, since the only surviving edge is speaker-only. Other
        # chunks (e.g. focal=x_node, with real non-speaker filler edges)
        # get a harmless empty mapping so they don't pollute the count
        # this test asserts on.
        def _stub(facts, model, tokenizer, transcript="", **kwargs):
            # ``facts`` is a plain fact-dict list — never a ``SessionGraph``.
            names = {str(f.get("subject", "")) for f in facts} | {
                str(f.get("object", "")) for f in facts
            }
            if names == {"speaker0", "speaker1"}:
                return {"nobody in this chunk": "Person_1"}, "stub-anon", "stub-raw"
            return {}, "stub-anon", "stub-raw"

        monkeypatch.setattr(
            "paramem.cloud.anonymize.anonymize_transcript",
            _stub,
        )
        canned_result = ([], [], "raw", 0)
        with patch(
            "paramem.training.graph_enrich.request_graph_enrichment",
            return_value=canned_result,
        ) as call_spy:
            result = _refiner_for(loop).run_enrichment()

        assert not result["skipped"]
        assert result["privacy_skipped_chunks"] == 0
        assert result["mapping_rekey_dropped"] >= 1
        call_spy.assert_called()


class TestScrubEmptyOptsOutWithoutModelCall:
    """With ``scrub=set()`` (operator opt-out), graph-tier enrichment must
    make NO anonymizer call and the chunk's triples must egress VERBATIM —
    the same operator opt-out contract session-tier extraction and
    chat egress already honour.  This is a behaviour CHANGE from
    pre-unification graph-tier enrichment (which called the local
    anonymizer unconditionally, regardless of ``scrub``); pinned here as
    the INTENDED outcome, not an oversight.
    """

    def test_no_anonymizer_call_and_verbatim_egress(self, tmp_path, monkeypatch):
        loop = _make_loop(tmp_path, extraction_scrub=set())
        graph = loop.merger.graph
        _populate_graph(graph, n_persons=10)

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        anonymizer_spy = MagicMock()
        captured_mapping: list[dict] = []
        captured_facts: list[list[dict]] = []

        def _capture(payload, graph, *args, **kwargs):
            captured_mapping.append(dict(payload.forward))
            captured_facts.append(list(payload.facts))
            return [], [], "raw", 0

        with (
            patch("paramem.cloud.anonymize.anonymize_transcript", anonymizer_spy),
            patch("paramem.training.graph_enrich.request_graph_enrichment", side_effect=_capture),
        ):
            result = _refiner_for(loop).run_enrichment()

        assert not result["skipped"]
        anonymizer_spy.assert_not_called()
        assert captured_mapping, "expected request_graph_enrichment to be called"
        # Verbatim egress: an empty forward mapping substitutes nothing.
        assert all(m == {} for m in captured_mapping)
        # BLOCKING-2 regression guard: the opted-out contract must carry
        # the chunk's input triples verbatim in payload.facts — a
        # facts=[] opt-out would silently withhold every triple from a
        # payload the operator asked to egress unmasked.
        assert captured_facts, "expected payload.facts to be captured"
        assert all(facts for facts in captured_facts), (
            "opted-out payload.facts must carry the chunk's triples verbatim, not []"
        )


class TestDroppedRelations:
    """Graph-tier enrichment's per-relation drop count (``dropped_relations``
    — replaces the retired ``totality_rejected_chunks`` whole-chunk gate,
    2026-07-22 cloud-admission redesign): a cloud response naming an
    orphan/unresolvable token now sheds only the offending relation(s),
    counted here (parallel to the existing ``privacy_skipped_chunks`` /
    ``mapping_rekey_dropped`` counters). Distinct from
    ``privacy_skipped_chunks``, which fires BEFORE any cloud call is made;
    this counter fires AFTER a real cloud response was individually
    filtered by the deanonymize residual sweep.
    """

    def test_orphan_token_in_cloud_response_increments_the_counter(self, tmp_path, monkeypatch):
        loop = _make_loop(tmp_path)
        graph = loop.merger.graph
        _populate_graph(graph, n_persons=10)

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        # Default autouse fixture masks every non-speaker name as
        # "person" (Person_N) — person0 becomes Person_1 (sorted-name
        # order). The cloud response below names an orphan token
        # ("Person_99") never declared anywhere in this chunk's mapping —
        # that ONE relation is individually dropped by the fail-closed
        # residual sweep.
        canned_raw = (
            '{"relations": [{"subject": "Person_1", "predicate": "knows", '
            '"object": "Person_99", "relation_type": "social", "confidence": 0.9}], '
            '"same_as": []}'
        )
        with patch("paramem.graph.extractor._cloud_call", return_value=canned_raw):
            result = _refiner_for(loop).run_enrichment()

        assert not result["skipped"]
        assert result["chunks"] == 1
        assert result["dropped_relations"] == 1
        assert result["privacy_skipped_chunks"] == 0
        assert result["new_edges"] == 0

    def test_clean_cloud_response_does_not_increment_the_counter(self, tmp_path, monkeypatch):
        loop = _make_loop(tmp_path)
        graph = loop.merger.graph
        _populate_graph(graph, n_persons=10)

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        canned_raw = '{"relations": [], "same_as": []}'
        with patch("paramem.graph.extractor._cloud_call", return_value=canned_raw):
            result = _refiner_for(loop).run_enrichment()

        assert not result["skipped"]
        assert result["dropped_relations"] == 0

    def test_counter_reads_the_returned_count_not_a_graph_mutation(self, tmp_path, monkeypatch):
        """The counter is driven by the COUNT ``request_graph_enrichment``
        returns (its fourth tuple element), not by reading a diagnostic
        back off the throwaway per-chunk graph.

        The stub below returns a non-zero count while touching no graph
        at all — under a readback-based counter this would stay 0.
        """
        loop = _make_loop(tmp_path)
        graph = loop.merger.graph
        _populate_graph(graph, n_persons=10)

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with patch(
            "paramem.training.graph_enrich.request_graph_enrichment",
            return_value=([], [], "raw", 1),
        ):
            result = _refiner_for(loop).run_enrichment()

        assert not result["skipped"]
        assert result["dropped_relations"] == result["chunks"] >= 1
        assert result["new_edges"] == 0

    def test_zero_count_with_empty_delta_is_not_counted_as_a_drop(self, tmp_path, monkeypatch):
        """A legitimately EMPTY delta returns the same ``([], [], raw,
        ...)`` shape as one that dropped relations — only the count tells
        them apart, and a zero count must not increment the stat."""
        loop = _make_loop(tmp_path)
        graph = loop.merger.graph
        _populate_graph(graph, n_persons=10)

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with patch(
            "paramem.training.graph_enrich.request_graph_enrichment",
            return_value=([], [], "raw", 0),
        ):
            result = _refiner_for(loop).run_enrichment()

        assert not result["skipped"]
        assert result["dropped_relations"] == 0


class TestSliceCounters:
    """Fact-boundary slicing counters — ``anonymize_slices`` (total local
    ``anonymize()`` calls across all chunks, ``sum(payload.slices)``) and
    ``privacy_skipped_slices`` (``sum(payload.slices_failed)``), plus the
    per-chunk partial-withholding WARNING (``0 < slices_failed < slices``).
    Parallel to :class:`TestDroppedRelations`'s pattern: patches
    ``paramem.training.graph_enrich.anonymize`` directly to control
    ``payload.slices``/``payload.slices_failed`` without needing the real
    packer to produce multiple slices.
    """

    def _payload(self, *, status="ok", slices=1, slices_failed=0, failure=None):
        from paramem.cloud.anonymize import AnonymizedContract

        return AnonymizedContract(
            status=status,
            forward={},
            reverse={},
            anon_transcript="",
            declared=frozenset(),
            norm_stats={"inverted": 0, "dropped": 0},
            rekey_dropped=0,
            raw="raw",
            failure=failure,
            facts=[],
            slices=slices,
            slices_failed=slices_failed,
        )

    def test_empty_dict_carries_zeroed_counters_on_every_skip_path(self, tmp_path, monkeypatch):
        """``no_model`` / ``floor`` / ``cloud_egress_blocked`` all short-circuit
        before any ``anonymize()`` call — both counters stay 0."""
        loop = _make_loop(tmp_path)
        graph = loop.merger.graph

        # no_model
        loop.model = None
        result = _refiner_for(loop).run_enrichment()
        assert result["anonymize_slices"] == 0
        assert result["privacy_skipped_slices"] == 0

        # floor (graph too small — no nodes added)
        loop.model = MagicMock()
        result = _refiner_for(loop).run_enrichment()
        assert result["skip_reason"] == "floor"
        assert result["anonymize_slices"] == 0
        assert result["privacy_skipped_slices"] == 0

        # cloud_egress_blocked (no provider configured)
        _populate_graph(graph, n_persons=10)
        loop2 = _make_loop(tmp_path, extraction_enrichment_provider="")
        loop2.merger = loop.merger
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        result = _refiner_for(loop2).run_enrichment()
        assert result["skip_reason"] == "cloud_egress_blocked"
        assert result["anonymize_slices"] == 0
        assert result["privacy_skipped_slices"] == 0

    def test_anonymize_slices_counts_local_calls_for_a_single_chunk(self, tmp_path, monkeypatch):
        loop = _make_loop(tmp_path)
        graph = loop.merger.graph
        _populate_graph(graph, n_persons=10)  # 11 nodes -> one chunk by default

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with (
            patch(
                "paramem.training.graph_enrich.anonymize",
                return_value=self._payload(status="ok", slices=3, slices_failed=0),
            ),
            patch(
                "paramem.training.graph_enrich.request_graph_enrichment",
                return_value=([], [], "raw", 0),
            ),
        ):
            result = _refiner_for(loop).run_enrichment()

        assert not result["skipped"]
        assert result["chunks"] == 1
        assert result["anonymize_slices"] == 3
        assert result["privacy_skipped_slices"] == 0

    def test_privacy_skipped_slices_sums_across_multiple_chunks(self, tmp_path, monkeypatch):
        """Two chunks, each contributing a different ``slices_failed`` —
        the counter SUMS across chunks, not just the last one."""
        loop = _make_loop(
            tmp_path,
            graph_enrichment_max_entities_per_pass=10,
            graph_enrichment_neighborhood_hops=1,
        )
        graph = loop.merger.graph
        org = "hubcorp"
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
            name = f"emp{i}"
            graph.add_node(
                name,
                entity_type="person",
                attributes={"name": f"Emp{i}"},
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

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        # First two chunks carry the interesting shape; any further chunk
        # (this graph's chunk_cap may exceed 2) falls back to a plain
        # single-slice success — its contribution is accounted for below
        # via ``extra_chunks`` rather than assumed away.
        scripted = [
            self._payload(status="ok", slices=2, slices_failed=1),
            self._payload(status="ok", slices=3, slices_failed=0),
        ]
        fallback_calls = {"n": 0}

        def _fake_anonymize(*args, **kwargs):
            if scripted:
                return scripted.pop(0)
            fallback_calls["n"] += 1
            return self._payload(status="ok", slices=1, slices_failed=0)

        with (
            patch("paramem.training.graph_enrich.anonymize", side_effect=_fake_anonymize),
            patch(
                "paramem.training.graph_enrich.request_graph_enrichment",
                return_value=([], [], "raw", 0),
            ),
        ):
            result = _refiner_for(loop).run_enrichment()

        assert not result["skipped"]
        assert result["chunks"] >= 2
        expected_slices = 2 + 3 + fallback_calls["n"] * 1
        assert result["anonymize_slices"] == expected_slices
        assert result["privacy_skipped_slices"] == 1

    def test_whole_chunk_failure_still_counts_slices_and_slices_failed(self, tmp_path, monkeypatch):
        """A whole-chunk fail-closed payload (``status="failed"``,
        ``slices_failed == slices``) still contributes to both counters —
        they are not gated on ``status == "ok"``."""
        loop = _make_loop(tmp_path)
        graph = loop.merger.graph
        _populate_graph(graph, n_persons=10)

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with (
            patch(
                "paramem.training.graph_enrich.anonymize",
                return_value=self._payload(
                    status="failed", slices=2, slices_failed=2, failure="guard"
                ),
            ),
            patch("paramem.training.graph_enrich.request_graph_enrichment") as call_spy,
        ):
            result = _refiner_for(loop).run_enrichment()

        call_spy.assert_not_called()
        assert result["chunks"] == 0
        assert result["privacy_skipped_chunks"] == 1
        assert result["anonymize_slices"] == 2
        assert result["privacy_skipped_slices"] == 2

    def test_partial_withholding_logs_warning(self, tmp_path, monkeypatch, caplog):
        """A chunk whose payload is ``"ok"`` but carries ``0 < slices_failed
        < slices`` (partial withholding) logs a per-chunk WARNING naming the
        dropped/surviving slice counts — the operator-visibility gap this
        closes."""
        import logging

        loop = _make_loop(tmp_path)
        graph = loop.merger.graph
        _populate_graph(graph, n_persons=10)

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        caplog.set_level(logging.WARNING, logger="paramem.training.graph_enrich")
        with (
            patch(
                "paramem.training.graph_enrich.anonymize",
                return_value=self._payload(status="ok", slices=3, slices_failed=1),
            ),
            patch(
                "paramem.training.graph_enrich.request_graph_enrichment",
                return_value=([], [], "raw", 0),
            ),
        ):
            _refiner_for(loop).run_enrichment()

        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any("partial withholding" in m for m in warnings), warnings
        assert any("1/3" in m for m in warnings), warnings

    def test_no_partial_withholding_warning_when_no_slices_failed(
        self, tmp_path, monkeypatch, caplog
    ):
        import logging

        loop = _make_loop(tmp_path)
        graph = loop.merger.graph
        _populate_graph(graph, n_persons=10)

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        caplog.set_level(logging.WARNING, logger="paramem.training.graph_enrich")
        with (
            patch(
                "paramem.training.graph_enrich.anonymize",
                return_value=self._payload(status="ok", slices=2, slices_failed=0),
            ),
            patch(
                "paramem.training.graph_enrich.request_graph_enrichment",
                return_value=([], [], "raw", 0),
            ),
        ):
            _refiner_for(loop).run_enrichment()

        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert not any("partial withholding" in m for m in warnings), warnings


class TestSameAsUndeclaredOrphanShapeBackstop:
    """``deanonymize_text`` runs no
    undeclared-orphan shape backstop for the ``same_as`` arm — verify the
    documented safety argument holds: an undeclared placeholder-shaped
    token in a ``same_as`` pair (never in this chunk's reverse map, so
    nothing resolves it and no shape check drops it either) still cannot
    reach a node contraction, because it is dropped by
    ``graph_enrich.enrich_graph``'s own graph-membership guard
    first.
    """

    def test_undeclared_orphan_same_as_member_never_merges(self, tmp_path, monkeypatch):
        loop = _make_loop(tmp_path)
        graph = loop.merger.graph
        _populate_graph(graph, n_persons=10)

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        # "Person_1" is declared (person0's mask, sorted-name order);
        # "Person_99" was never declared for this chunk at all -- it
        # passes through deanonymize_text UNCHANGED (neither
        # resolved nor dropped, since the declared-token check only
        # fires on tokens that WERE declared).
        canned_raw = '{"relations": [], "same_as": [["Person_1", "Person_99"]]}'
        with patch("paramem.graph.extractor._cloud_call", return_value=canned_raw):
            result = _refiner_for(loop).run_enrichment()

        assert not result["skipped"]
        assert result["dropped_relations"] == 0
        assert result["same_as_merges"] == 0


def _payload_and_graph_for(triples: list[dict], llm_mapping: dict[str, str]):
    """Build the ``(payload, graph)`` pair ``request_graph_enrichment`` now
    takes, from a caller-supplied ``llm_mapping`` (real_name -> placeholder)
    and the chunk's ``triples`` — mirroring what
    ``graph_enrich.enrich_graph`` produces via
    ``anonymize`` (with ``identity_domain=None``, matching a
    direct unit-level call with no reconciliation domain), so these tests
    exercise the REAL table-building primitive rather than a hand-rolled
    substitute.

    ``graph`` carries no relations of its own (interface narrowing,
    2026-07-21): ``request_graph_enrichment`` derives its anonymized
    triples directly from ``payload.facts`` via ``insert_placeholders``,
    not from ``graph.relations`` — ``graph`` is only the diagnostics
    sink. ``payload.facts`` is set to ``triples`` here, mirroring what
    :func:`~paramem.cloud.anonymize.anonymize` populates it with on a
    successful (non-fail-closed) call.
    """
    from paramem.cloud.anonymize import AnonymizedContract
    from paramem.cloud.placeholders import _build_anonymization_mapping
    from paramem.graph.schema import SessionGraph

    forward, reverse = _build_anonymization_mapping(dict(llm_mapping), speaker_name=None)
    payload = AnonymizedContract(
        status="ok",
        forward=forward,
        reverse=reverse,
        anon_transcript="",
        declared=frozenset(reverse.keys()),
        norm_stats={"inverted": 0, "dropped": 0},
        rekey_dropped=0,
        raw="",
        facts=triples,
    )
    graph = SessionGraph(session_id="__graph_enrichment_test__", timestamp="")
    return payload, graph


class TestGraphEnrichWithCloudUnit:
    """Unit tests for the extractor-level request_graph_enrichment function."""

    def test_returns_relations_and_same_as(self):
        from paramem.graph.extractor import request_graph_enrichment

        canned_raw = (
            '{"relations": [{"subject": "A", "predicate": "knows", "object": "B", '
            '"relation_type": "social", "confidence": 0.8}], "same_as": [["Alice", "Alicia"]]}'
        )
        triples = [
            {
                "subject": "A",
                "predicate": "works_at",
                "object": "Corp",
                "relation_type": "factual",
            }
        ]
        payload, graph = _payload_and_graph_for(triples, {})
        with patch("paramem.graph.extractor._cloud_call", return_value=canned_raw):
            result = request_graph_enrichment(
                payload,
                graph,
                api_key="test-key",
                provider="anthropic",
                filter_model="claude-sonnet-4-6",
            )

        assert result is not None
        new_rels, same_as, raw, _dropped_count = result
        assert len(new_rels) == 1
        assert new_rels[0]["predicate"] == "knows"
        assert len(same_as) == 1
        assert same_as[0] == ["Alice", "Alicia"]

    def test_system_prompt_overridable_and_recorded_in_provenance(self):
        """``cloud_graph_enrichment_system.txt`` used to bind ONCE at module
        import time (``_CLOUD_GRAPH_ENRICHMENT_SYSTEM_PROMPT``) — unreachable
        by a calibration override and never recorded via ``record_prompt``.
        It now loads at CALL TIME inside ``request_graph_enrichment`` itself,
        so both become possible."""
        from paramem.graph.extractor import request_graph_enrichment
        from paramem.graph.phase_trace import extraction_trace, phase_trace
        from paramem.graph.prompts import prompt_overrides

        captured = []

        def fake_cloud_call(prompt, *args, **kwargs):
            captured.append(kwargs.get("system_prompt"))
            return '{"relations": [], "same_as": []}'

        triples = [
            {
                "subject": "A",
                "predicate": "works_at",
                "object": "Corp",
                "relation_type": "factual",
            }
        ]
        payload, graph = _payload_and_graph_for(triples, {})
        with patch("paramem.graph.extractor._cloud_call", side_effect=fake_cloud_call):
            with extraction_trace() as trace:
                with phase_trace("cloud_enrich"):
                    with prompt_overrides(
                        {"cloud_graph_enrichment_system.txt": "SENTINEL-GRAPH-ENRICH-SYSTEM"}
                    ):
                        request_graph_enrichment(
                            payload,
                            graph,
                            api_key="test-key",
                            provider="anthropic",
                            filter_model="claude-sonnet-4-6",
                        )
                record = trace.records[-1]

        assert captured == ["SENTINEL-GRAPH-ENRICH-SYSTEM"], (
            "the override must reach _cloud_call's system_prompt kwarg"
        )
        paths = [p["path"] for p in (record.prompts or [])]
        assert "<override:cloud_graph_enrichment_system.txt>" in paths, (
            f"override must be recorded in phase-trace provenance, got paths={paths!r}"
        )

    def test_legacy_bare_array(self):
        """Bare JSON array response → treated as relations, empty same_as."""
        from paramem.graph.extractor import request_graph_enrichment

        canned_raw = (
            '[{"subject": "A", "predicate": "knows", "object": "B", '
            '"relation_type": "social", "confidence": 0.8}]'
        )
        payload, graph = _payload_and_graph_for([], {})
        with patch("paramem.graph.extractor._cloud_call", return_value=canned_raw):
            result = request_graph_enrichment(
                payload,
                graph,
                api_key="test-key",
                provider="anthropic",
                filter_model="claude-sonnet-4-6",
            )

        assert result is not None
        new_rels, same_as, _, _dropped_count = result
        assert len(new_rels) == 1
        assert same_as == []

    def test_none_on_cloud_failure(self):
        """_cloud_call returning None → request_graph_enrichment returns None."""
        from paramem.graph.extractor import request_graph_enrichment

        payload, graph = _payload_and_graph_for([], {})
        with patch("paramem.graph.extractor._cloud_call", return_value=None):
            result = request_graph_enrichment(
                payload,
                graph,
                api_key="test-key",
                provider="anthropic",
                filter_model="claude-sonnet-4-6",
            )
        assert result is None

    def test_malformed_same_as_skipped(self):
        """Malformed same_as entries are silently skipped."""
        from paramem.graph.extractor import request_graph_enrichment

        canned_raw = '{"relations": [], "same_as": ["bad", [1, 2], ["Alice", "Alicia"]]}'
        payload, graph = _payload_and_graph_for([], {})
        with patch("paramem.graph.extractor._cloud_call", return_value=canned_raw):
            result = request_graph_enrichment(
                payload,
                graph,
                api_key="test-key",
                provider="anthropic",
                filter_model="claude-sonnet-4-6",
            )

        assert result is not None
        _, same_as, _, _dropped_count = result
        # Only the valid [Alice, Alicia] entry survives; ["bad", [1,2]] are skipped
        assert same_as == [["Alice", "Alicia"]]

    def test_dropped_relation_count_is_returned_and_orphan_relation_is_shed(self):
        """An individually-unresolvable relation (its object references a
        token never declared for this chunk) is dropped by the fail-closed
        residual sweep, and the caller's dropped-relation count (the
        fourth tuple element) reflects it — replacing the retired
        whole-chunk rejection this test used to pin."""
        from paramem.graph.extractor import request_graph_enrichment

        triples = [
            {
                "subject": "Alex",
                "predicate": "works_at",
                "object": "Corp",
                "relation_type": "factual",
            }
        ]
        # "Person_99" is never declared for this chunk -> orphan -> that
        # ONE relation is dropped by the residual sweep.
        canned_raw = (
            '{"relations": [{"subject": "Person_1", "predicate": "knows", '
            '"object": "Person_99", "relation_type": "social", "confidence": 0.9}], '
            '"same_as": []}'
        )
        payload, graph = _payload_and_graph_for(triples, {"Alex": "Person_1"})
        with patch("paramem.graph.extractor._cloud_call", return_value=canned_raw):
            result = request_graph_enrichment(
                payload,
                graph,
                api_key="test-key",
                provider="anthropic",
                filter_model="claude-sonnet-4-6",
            )

        assert result is not None
        new_rels, same_as, _raw, dropped_count = result
        assert dropped_count == 1
        assert new_rels == []
        assert same_as == []
        # No cloud_bindings in the response -> the collision scan found
        # nothing -> no key at all (never a present-but-empty list).
        assert "cloud_binding_collisions" not in graph.diagnostics

    def test_accepted_chunk_reports_zero_dropped_relations(self):
        """The guard conditions are preserved end-to-end: an accepted
        delta reports a zero dropped-relation count and leaves the
        collision diagnostic key ABSENT rather than writing an empty
        list."""
        from paramem.graph.extractor import request_graph_enrichment

        triples = [
            {
                "subject": "Alex",
                "predicate": "works_at",
                "object": "Corp",
                "relation_type": "factual",
            }
        ]
        canned_raw = '{"relations": [], "same_as": []}'
        payload, graph = _payload_and_graph_for(triples, {"Alex": "Person_1"})
        with patch("paramem.graph.extractor._cloud_call", return_value=canned_raw):
            result = request_graph_enrichment(
                payload,
                graph,
                api_key="test-key",
                provider="anthropic",
                filter_model="claude-sonnet-4-6",
            )

        assert result is not None
        assert result[3] == 0
        assert "cloud_binding_collisions" not in graph.diagnostics
        assert "cloud_binding_collisions" not in graph.diagnostics


class TestInterimEnrichmentHook:
    """Interim refine inside run_consolidation_cycle runs NO graph-tier enrichment.

    ``FoldScope.enrich`` is pinned ``False`` structurally at the interim
    ``FoldScope`` construction site (``consolidation.py``), regardless of the
    operator's ``refinement_enrichment`` knob or ``cloud_enabled`` — graph-tier
    enrichment is a full-fold-only pass.  These tests pin
    ``GraphTierRefiner.run_enrichment`` is never called from the
    ``run_consolidation_cycle`` (interim) entry point, across every gating
    combination.  The full-fold enrichment path is covered by
    TestRunGraphEnrichment / TestRefineConsolidationGraph.
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

    def test_refinement_enrichment_on_does_not_enrich_at_interim(self, tmp_path):
        """refinement_enrichment='on' + cloud master switch on → interim still

        runs no graph-tier enrichment. ``FoldScope.enrich`` is pinned ``False``
        structurally for every interim cycle (``consolidation.py`` interim
        ``FoldScope`` construction site), regardless of the operator's
        ``refinement_enrichment`` knob or whether cloud egress is enabled —
        graph-tier enrichment is a full-fold-only pass.
        """
        from paramem.training.key_registry import KeyRegistry
        from paramem.utils.config import ConsolidationConfig

        loop = _make_loop(
            tmp_path,
            consolidation_config=ConsolidationConfig(refinement_enrichment="on"),
            cloud_enabled=True,
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

        loop.model.peft_config = {"episodic": MagicMock(), "semantic": MagicMock()}
        with (
            patch(
                "paramem.memory.interim_adapter.create_interim_adapter",
                return_value=loop.model,
            ),
            patch("paramem.training.trainer.train_adapter", return_value={}),
            patch("paramem.models.loader.save_adapter"),
            patch("paramem.adapters.manifest.build_manifest_for", return_value=None),
            patch.object(
                GraphTierRefiner, "run_enrichment", return_value={"skipped": False}
            ) as enrich_mock,
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

        enrich_mock.assert_not_called()

    # test_refinement_normalization_only_does_not_enrich and
    # test_refinement_off_does_not_enrich were collapsed into the on+cloud
    # case above (code review, 2026-07-28): the interim gate is
    # unconditional (see this class's docstring and
    # test_interim_scope_pins_enrich_false in test_consolidation.py), so the
    # on+cloud case above -- the single hardest config to satisfy -- already
    # implies both weaker configs; testing them separately added no coverage.

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
        with patch.object(
            GraphTierRefiner, "run_enrichment", return_value={"skipped": False}
        ) as enrich_mock:
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
        enrich_mock.assert_not_called()


# ---------------------------------------------------------------------------
# Tests for _refine_consolidation_graph (enrich gate + recurrence-bump)
# ---------------------------------------------------------------------------


class TestRefineConsolidationGraph:
    """Unit tests for _refine_consolidation_graph's enrich param.

    Covers:
    - enrich=True calls GraphTierRefiner.run_enrichment (fold default, unconditional).
    - enrich=False skips GraphTierRefiner.run_enrichment entirely.
    - Recurrence-bump loop runs regardless of enrich.
    - Empty recon_relations is a safe no-op for both code paths.
    """

    def test_enrich_true_calls_run_enrichment(self, tmp_path):
        """_refine_consolidation_graph(recon, enrich=True) calls run_enrichment."""
        loop = _make_loop(tmp_path)

        with patch.object(
            GraphTierRefiner, "run_enrichment", return_value={"skipped": True}
        ) as enrich_mock:
            loop._refine_consolidation_graph([], enrich=True)

        enrich_mock.assert_called_once()

    def test_default_skips_both_normalize_and_enrich(self, tmp_path):
        """normalize and enrich both default False (level "off") — refine calls neither."""
        loop = _make_loop(tmp_path)

        with (
            patch.object(
                GraphTierRefiner, "run_enrichment", return_value={"skipped": True}
            ) as enrich_mock,
            patch.object(
                GraphTierRefiner, "run_normalization", return_value={"skipped": True}
            ) as norm_mock,
        ):
            # No kwargs — level "off" semantics (both default False).
            loop._refine_consolidation_graph([])

        enrich_mock.assert_not_called()
        norm_mock.assert_not_called()

    def test_enrich_false_skips_run_enrichment(self, tmp_path):
        """_refine_consolidation_graph(recon, enrich=False) does NOT call run_enrichment."""
        loop = _make_loop(tmp_path)

        with patch.object(
            GraphTierRefiner, "run_enrichment", return_value={"skipped": True}
        ) as enrich_mock:
            loop._refine_consolidation_graph([], enrich=False)

        enrich_mock.assert_not_called()

    def test_recurrence_bump_runs_when_enrich_false(self, tmp_path):
        """Recurrence-bump fires regardless of enrich; enrich=False only skips cloud."""
        from paramem.training.key_registry import KeyRegistry

        loop = _make_loop(tmp_path, replay_enabled=True)
        for tier in ("episodic", "semantic", "procedural"):
            loop.store.load_registry(tier, KeyRegistry())

        # Register a key so the credit pass has a target.
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
            last_seen="2025-12-01T00:00:00Z",
            allow_empty_speaker=True,
            first_seen="2025-12-01T00:00:00Z",
        )
        # The retired twin: a separate session, so the collapse is a genuine
        # re-sighting and earns.
        loop.store.set_bookkeeping(
            "graph43",
            speaker_id="",
            relation_type="factual",
            reinforcement_count=1,
            last_reinforced_cycle=0,
            last_seen="2026-01-01T00:00:00Z",
            allow_empty_speaker=True,
            first_seen="2026-01-01T00:00:00Z",
        )

        # Simulate a Case-1 collision: the ledger names the surviving key.
        loop.merger.removal_ledger = {"graph43": {"reason": "dedup", "survivor_key": "graph42"}}

        from paramem.graph.schema import Relation

        recon_rel = Relation(
            subject="A", predicate="p", object="B", relation_type="factual", speaker_id=""
        )

        with patch.object(
            GraphTierRefiner, "run_enrichment", return_value={"skipped": True}
        ) as enrich_mock:
            loop._refine_consolidation_graph([recon_rel], enrich=False)

        enrich_mock.assert_not_called()
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
            last_seen="2025-12-01T00:00:00Z",
            allow_empty_speaker=True,
            first_seen="2025-12-01T00:00:00Z",
        )
        loop.store.set_bookkeeping(
            "graph8",
            speaker_id="",
            relation_type="factual",
            reinforcement_count=1,
            last_reinforced_cycle=0,
            last_seen="2026-01-01T00:00:00Z",
            allow_empty_speaker=True,
            first_seen="2026-01-01T00:00:00Z",
        )
        loop.merger.removal_ledger = {"graph8": {"reason": "dedup", "survivor_key": "graph7"}}

        from paramem.graph.schema import Relation

        recon_rel = Relation(
            subject="X", predicate="q", object="Y", relation_type="factual", speaker_id=""
        )

        with patch.object(
            GraphTierRefiner, "run_enrichment", return_value={"skipped": False}
        ) as enrich_mock:
            loop._refine_consolidation_graph([recon_rel], enrich=True)

        enrich_mock.assert_called_once()
        bk = loop.store.bookkeeping_for_key("graph7")
        assert bk is not None
        assert bk["reinforcement_count"] == 4, (
            f"Recurrence should have been bumped to 4; got {bk['reinforcement_count']}"
        )

    def test_no_removals_is_safe_noop_for_credit(self, tmp_path):
        """A fold that removed nothing credits nothing — no store write, no crash.

        The credit pass is driven by the removal ledger, and a ledger entry
        exists only where an edge was actually removed, so an untouched fold has
        nothing to credit and must not write to the store to discover that.
        """
        loop = _make_loop(tmp_path)
        loop.merger.removal_ledger = {}

        credit_spy = MagicMock()
        loop.store.reinforce = credit_spy

        with patch.object(GraphTierRefiner, "run_enrichment", return_value={"skipped": True}):
            loop._refine_consolidation_graph([], enrich=False)
            loop._refine_consolidation_graph([], enrich=True)

        credit_spy.assert_not_called()

    def test_reasons_without_a_survivor_credit_nothing(self, tmp_path):
        """A contradiction supersedes with a DIFFERENT fact and an enrichment
        same_as contracts nodes — neither names a survivor, so neither moves
        maturity onto anything."""
        loop = _make_loop(tmp_path)
        loop.merger.removal_ledger = {
            "graph1": {"reason": "contradiction_same_pred", "old_object": "a", "new_object": "b"},
            "graph2": {"reason": "enrichment_same_as", "keep_node": "alice"},
        }

        credit_spy = MagicMock()
        loop.store.reinforce = credit_spy

        with patch.object(GraphTierRefiner, "run_enrichment", return_value={"skipped": True}):
            loop._refine_consolidation_graph([], enrich=False)

        credit_spy.assert_not_called()

    def test_normalize_true_calls_run_normalization(self, tmp_path):
        """normalize=True runs the whole-graph normalization pass (light+, both scopes).

        The normalization pass is independent of enrich: normalize=True without
        enrich runs normalization only (the light default), not cloud enrichment.
        """
        loop = _make_loop(tmp_path)

        with (
            patch.object(
                GraphTierRefiner, "run_normalization", return_value={"skipped": True}
            ) as norm_mock,
            patch.object(
                GraphTierRefiner, "run_enrichment", return_value={"skipped": True}
            ) as enrich_mock,
        ):
            loop._refine_consolidation_graph([], normalize=True)

        norm_mock.assert_called_once()
        enrich_mock.assert_not_called()


class TestRefineOrderEnrichThenNormalize:
    """``GraphTierRefiner.refine`` runs enrichment BEFORE normalization.

    Unlike ``TestRefineConsolidationGraph`` (which mocks both passes away to
    test ``_refine_consolidation_graph``'s wiring), these tests exercise the
    refiner's REAL ``run_enrichment``/``run_normalization`` bodies — with
    only the underlying cloud primitives (``request_graph_enrichment``,
    ``normalize_predicates``) mocked — so the observed order and the
    content interaction between the two passes are real, not asserted on
    the caller's behalf.
    """

    # Enrichment-before-normalization call ORDER is pinned at the caller in
    # test_simulate_train_parity.py::TestGraphTierSkipsAfterRelease
    # .test_refine_stage_skips_both_passes_on_released_loop, which asserts
    # call_order == ["enrichment", "normalization"] through
    # _refine_consolidation_graph, the production entry point.  Not
    # duplicated here at the refiner level.

    def test_normalization_sees_enrichment_edges(self, tmp_path, monkeypatch):
        """Defect regression pin: a cloud paraphrase minted by
        enrichment is visible to (and collapsed by) normalization in the
        SAME ``refine()`` call.  Before the flip, enrichment ran AFTER
        normalization, so a cloud-coined paraphrase reached the fold's key
        assembly un-normalized — this test fails on that ordering.

        The (s,o) pair already carries an established predicate ("works
        at", from ``_populate_graph``); enrichment mints a same-pair
        paraphrase ("employed_by").  ``_pred_sort_key``'s three-term survivor
        rule keeps the established predicate as the survivor.
        """
        loop = _make_loop(tmp_path)
        graph = loop.merger.graph
        _populate_graph(graph, n_persons=10)

        canned_enrichment = (
            [
                {
                    "subject": "Person0",
                    "predicate": "employed_by",
                    "object": "AcmeCorp",
                    "relation_type": "factual",
                    "confidence": 0.9,
                }
            ],
            [],
            "raw",
            0,
        )
        canned_normalize = (
            {("person0", "acmecorp"): [["works at", "employed by"]]},
            {"model_calls": 1, "raw_outputs": []},
        )

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with (
            patch(
                "paramem.training.graph_enrich.request_graph_enrichment",
                return_value=canned_enrichment,
            ),
            patch(
                "paramem.training.graph_tier.normalize_predicates",
                return_value=canned_normalize,
            ),
        ):
            result = _refiner_for(loop).refine(normalize=True, enrich=True)

        assert not result.enrichment["skipped"]
        assert not result.normalization["skipped"]
        assert result.normalization["edges_retired"] >= 1, (
            "normalization must have retired the paraphrase edge -- it fails "
            "to see it at all under the pre-flip (normalize-then-enrich) order"
        )
        surviving_preds = {
            d.get("predicate")
            for _, _, d in graph.out_edges("person0", data=True)
            if d.get("predicate") in ("works at", "employed by")
        }
        assert surviving_preds == {"works at"}, (
            f"the established predicate must survive the paraphrase collapse; got {surviving_preds}"
        )

    # Ledger append-only across passes/merges: removal_ledger's reset-only-
    # in-reset_graph() lifecycle is pinned directly by
    # test_merger.py::TestRemovalLedger.test_reset_graph_clears_removal_ledger;
    # each reason code (predicate_synonym_collapse, enrichment_same_as) is
    # pinned writing correctly in isolation by the many
    # TestRunGraphNormalizationApply / TestEnrichmentRemovalLedger cases in
    # this suite; and the ledger surviving an intervening cross-pass merge
    # end-to-end is pinned by
    # TestDriftPartitioning.test_intervening_enrichment_merge_preserves_dedup_bucketing_and_bump
    # (test_consolidation.py), which pins the accumulator-lifetime rule:
    # merger.collapsed / merger.removal_ledger are reset ONLY by
    # reset_graph(), never at the top of merge(), so they survive an
    # intervening cross-pass merge within one fold.  A test asserting both
    # reason codes coexist in one refine() call adds no further coverage
    # (removal_ledger was never reset by merge() even before that rule was
    # enforced -- the fix only touched reinforcements/collapsed).


class TestSurvivorRuleEstablishedOutranksEnrichment:
    """``_pred_sort_key``'s three-term survivor key: ``(rec, established,
    last_seen)``.  Exercises ``GraphTierRefiner.run_normalization`` directly
    (no need to run enrichment for real; the enrichment-sourced edge is
    built directly with the exact ``reinforcement_count``/``last_seen`` shape
    an enrichment edge carries in production, so the arithmetic is pinned
    precisely).
    """

    def test_established_predicate_survives_enrichment_paraphrase_on_a_tie(self, tmp_path):
        """An enrichment-sourced predicate must never retire an organically
        extracted one on a recency tie: a 1-vs-1 group -- one established edge
        (reinforcement_count=1, OLDER last_seen) vs one enrichment edge
        (reinforcement_count=1, edge_source='graph_enrichment', NEWER
        last_seen -- the chunk-maximum an enrichment edge inherits).  Ties
        on ``rec``; without the ``established`` term the tie-break falls
        straight to ``last_seen`` and hands the win to the paraphrase.
        """
        loop = _make_loop(tmp_path)
        graph = loop.merger.graph
        _populate_graph(graph, n_persons=10)  # clears the 10-node floor

        established_eid = next(iter(graph["person0"]["acmecorp"]))
        graph["person0"]["acmecorp"][established_eid]["reinforcement_count"] = 1
        graph["person0"]["acmecorp"][established_eid]["last_seen"] = "s001"

        graph.add_edge(
            "person0",
            "acmecorp",
            predicate="employed by",
            relation_type="factual",
            confidence=0.9,
            reinforcement_count=1,
            last_seen="s999",  # newer -- the chunk's max last_seen
            sessions=["s999"],
            edge_source="graph_enrichment",
        )

        canned_normalize = (
            {("person0", "acmecorp"): [["works at", "employed by"]]},
            {"model_calls": 1, "raw_outputs": []},
        )
        with patch(
            "paramem.training.graph_tier.normalize_predicates",
            return_value=canned_normalize,
        ):
            result = _refiner_for(loop).run_normalization()

        assert result["edges_retired"] == 1
        surviving_preds = {
            d.get("predicate")
            for _, _, d in graph.out_edges("person0", data=True)
            if d.get("predicate") in ("works at", "employed by")
        }
        assert surviving_preds == {"works at"}, (
            f"the established predicate must survive the recency tie; got {surviving_preds}"
        )

    def test_reinforcement_count_still_leads(self, tmp_path):
        """``rec`` is the LEADING term: an enrichment predicate reinforced
        across sessions (summed rec=3) still beats an established one at
        rec=1 -- the ``established`` term must not override a real
        recurrence lead."""
        loop = _make_loop(tmp_path)
        graph = loop.merger.graph
        _populate_graph(graph, n_persons=10)

        established_eid = next(iter(graph["person0"]["acmecorp"]))
        graph["person0"]["acmecorp"][established_eid]["reinforcement_count"] = 1
        graph["person0"]["acmecorp"][established_eid]["last_seen"] = "s001"

        graph.add_edge(
            "person0",
            "acmecorp",
            predicate="employed by",
            relation_type="factual",
            confidence=0.9,
            reinforcement_count=3,
            last_seen="s500",
            sessions=["s500"],
            edge_source="graph_enrichment",
        )

        canned_normalize = (
            {("person0", "acmecorp"): [["works at", "employed by"]]},
            {"model_calls": 1, "raw_outputs": []},
        )
        with patch(
            "paramem.training.graph_tier.normalize_predicates",
            return_value=canned_normalize,
        ):
            result = _refiner_for(loop).run_normalization()

        assert result["edges_retired"] == 1
        surviving_preds = {
            d.get("predicate")
            for _, _, d in graph.out_edges("person0", data=True)
            if d.get("predicate") in ("works at", "employed by")
        }
        assert surviving_preds == {"employed by"}, (
            f"the reinforced enrichment predicate must win on rec alone; got {surviving_preds}"
        )

    # Two-established-predicates recency tie-break (neither edge
    # enrichment-sourced) is covered by
    # TestRunGraphNormalizationApply.test_provenance_last_seen_max_on_survivor
    # (test_consolidation.py) -- identical shape (both organic, rec ties,
    # last_seen decides) -- so it is not duplicated here.


class TestRefineConsolidationGraphRecordsVramIncident:
    """``_refine_consolidation_graph`` records one
    ``enrichment_degraded`` incident (``key="graph_enrich_vram"``, severity
    ``"warning"``) via the same ``record_incident`` surface used elsewhere
    in ``ConsolidationLoop``, when ``run_enrichment()`` reports
    ``aborted_reason == "vram"`` — and the fold continues past the refine
    step regardless (never raises).

    Mutation: drop the ``aborted_reason == "vram"`` incident-recording
    block -> this test's ``record_incident`` spy is never called.
    """

    def test_vram_abort_records_incident(self, tmp_path):
        loop = _make_loop(tmp_path, incidents_state_dir=tmp_path / "incidents")

        with (
            patch.object(
                GraphTierRefiner,
                "run_enrichment",
                return_value={"skipped": False, "aborted_reason": "vram", "chunks": 1},
            ) as enrich_mock,
            patch("paramem.server.incidents.record_incident") as record_mock,
        ):
            loop._refine_consolidation_graph([], enrich=True)

        enrich_mock.assert_called_once()
        record_mock.assert_called_once()
        _, kwargs = record_mock.call_args
        assert kwargs["type"] == "enrichment_degraded"
        assert kwargs["key"] == "graph_enrich_vram"
        assert kwargs["severity"] == "warning"

    def test_no_abort_does_not_record_incident(self, tmp_path):
        loop = _make_loop(tmp_path, incidents_state_dir=tmp_path / "incidents")

        with (
            patch.object(
                GraphTierRefiner,
                "run_enrichment",
                return_value={"skipped": False, "aborted_reason": None, "chunks": 3},
            ) as enrich_mock,
            patch("paramem.server.incidents.record_incident") as record_mock,
        ):
            loop._refine_consolidation_graph([], enrich=True)

        enrich_mock.assert_called_once()
        record_mock.assert_not_called()

    def test_vram_abort_without_incidents_state_dir_is_a_safe_noop(self, tmp_path):
        """No ``incidents_state_dir`` configured -> the guard skips recording
        rather than crashing; the fold still continues past refine."""
        loop = _make_loop(tmp_path)
        assert loop._incidents_state_dir is None

        with (
            patch.object(
                GraphTierRefiner,
                "run_enrichment",
                return_value={"skipped": False, "aborted_reason": "vram", "chunks": 1},
            ),
            patch("paramem.server.incidents.record_incident") as record_mock,
        ):
            loop._refine_consolidation_graph([], enrich=True)

        record_mock.assert_not_called()


class TestArbitrateSessionEnrichmentIncidents:
    """``ConsolidationLoop._arbitrate_session_enrichment_incidents`` — the
    session-tier reconciliation extracted from ``extract_session``.

    Uses the same ``record_incident``/``resolve_incident`` surface (and the
    same ``incidents_state_dir`` fixture pattern) as
    ``TestRefineConsolidationGraphRecordsVramIncident`` above.  Calls the
    method directly with a hand-built ``SessionGraph`` rather than driving
    ``extract_session`` end to end, so no GPU/model call is needed.
    """

    def _graph(self, **diagnostics) -> SessionGraph:
        return SessionGraph(
            session_id="s1", timestamp="2026-08-02T00:00:00+00:00", diagnostics=diagnostics
        )

    def test_ok_and_clean_resolves_both_keys(self, tmp_path):
        """``anonymize == "ok"`` and no ``cloud_enrichment_degraded`` resolves
        both the ``anonymize`` and ``cloud_enrich`` sub-incidents."""
        loop = _make_loop(tmp_path, incidents_state_dir=tmp_path / "incidents")
        from paramem.server.incidents import read_incidents, record_incident

        record_incident(
            loop._incidents_state_dir,
            type="enrichment_degraded",
            key="anonymize",
            severity="warning",
            summary="prior failure",
            detail={},
        )
        record_incident(
            loop._incidents_state_dir,
            type="enrichment_degraded",
            key="cloud_enrich",
            severity="warning",
            summary="prior failure",
            detail={},
        )

        loop._arbitrate_session_enrichment_incidents(self._graph(anonymize="ok"), "s1")

        by_id = {i.id: i.status for i in read_incidents(loop._incidents_state_dir)}
        assert by_id["enrichment_degraded:anonymize"] == "resolved"
        assert by_id["enrichment_degraded:cloud_enrich"] == "resolved"

    def test_ok_and_degraded_records_cloud_enrich(self, tmp_path):
        """``anonymize == "ok"`` with a populated ``cloud_enrichment_degraded``
        dict records the ``cloud_enrich`` incident and resolves ``anonymize``
        (a prior standing ``anonymize`` incident really does flip to
        resolved, not merely "was never recorded so nothing to check")."""
        loop = _make_loop(tmp_path, incidents_state_dir=tmp_path / "incidents")
        from paramem.server.incidents import read_incidents, record_incident

        record_incident(
            loop._incidents_state_dir,
            type="enrichment_degraded",
            key="anonymize",
            severity="warning",
            summary="prior anonymize failure",
            detail={},
        )

        graph = self._graph(anonymize="ok", cloud_enrichment_degraded={"reason": "unparseable"})
        loop._arbitrate_session_enrichment_incidents(graph, "s1")

        by_id = {i.id: i for i in read_incidents(loop._incidents_state_dir)}
        assert by_id["enrichment_degraded:anonymize"].status == "resolved"
        assert by_id["enrichment_degraded:cloud_enrich"].status == "active"
        assert by_id["enrichment_degraded:cloud_enrich"].detail["session_id"] == "s1"
        assert by_id["enrichment_degraded:cloud_enrich"].detail["reason"] == "unparseable"

    def test_opted_out_behaves_like_ok(self, tmp_path):
        """``anonymize == "opted_out"`` follows the exact same branch as
        ``"ok"`` — mirrors ``test_ok_and_clean_resolves_both_keys`` exactly
        so the assertion actually discriminates this branch from every
        other (a bare ``read_incidents == []`` would pass on an untouched
        store regardless of which branch ran)."""
        loop = _make_loop(tmp_path, incidents_state_dir=tmp_path / "incidents")
        from paramem.server.incidents import read_incidents, record_incident

        record_incident(
            loop._incidents_state_dir,
            type="enrichment_degraded",
            key="anonymize",
            severity="warning",
            summary="prior failure",
            detail={},
        )
        record_incident(
            loop._incidents_state_dir,
            type="enrichment_degraded",
            key="cloud_enrich",
            severity="warning",
            summary="prior failure",
            detail={},
        )

        loop._arbitrate_session_enrichment_incidents(self._graph(anonymize="opted_out"), "s1")

        by_id = {i.id: i.status for i in read_incidents(loop._incidents_state_dir)}
        assert by_id["enrichment_degraded:anonymize"] == "resolved"
        assert by_id["enrichment_degraded:cloud_enrich"] == "resolved"

    def test_failed_records_anonymize_key_and_leaves_cloud_enrich_untouched(self, tmp_path):
        """``anonymize == "failed"`` records the ``anonymize`` incident and does
        not touch any standing ``cloud_enrich`` incident.  ``detail`` carries
        no ``fallback_path`` key — on this branch it is always
        ``"anon_failed"`` (the stage is terminal), so the key would carry no
        information the branch itself doesn't already state."""
        loop = _make_loop(tmp_path, incidents_state_dir=tmp_path / "incidents")
        from paramem.server.incidents import read_incidents, record_incident

        record_incident(
            loop._incidents_state_dir,
            type="enrichment_degraded",
            key="cloud_enrich",
            severity="warning",
            summary="prior degrade, must survive untouched",
            detail={},
        )

        graph = self._graph(anonymize="failed", fallback_path="anon_failed")
        loop._arbitrate_session_enrichment_incidents(graph, "s1")

        by_id = {i.id: i for i in read_incidents(loop._incidents_state_dir)}
        assert by_id["enrichment_degraded:anonymize"].status == "active"
        assert by_id["enrichment_degraded:anonymize"].detail["session_id"] == "s1"
        assert "fallback_path" not in by_id["enrichment_degraded:anonymize"].detail
        # Untouched — status AND summary/detail from the earlier record stand.
        assert by_id["enrichment_degraded:cloud_enrich"].status == "active"
        assert by_id["enrichment_degraded:cloud_enrich"].summary == (
            "prior degrade, must survive untouched"
        )

    def test_failed_on_already_active_anonymize_row_bumps_count(self, tmp_path):
        """A second ``"failed"`` session bumps the existing ``anonymize``
        incident's count rather than minting a duplicate row; a previously
        RESOLVED ``anonymize`` incident reopens exactly like any other
        ``record_incident`` call."""
        loop = _make_loop(tmp_path, incidents_state_dir=tmp_path / "incidents")
        from paramem.server.incidents import read_incidents, resolve_incident

        graph = self._graph(anonymize="failed")
        loop._arbitrate_session_enrichment_incidents(graph, "s1")
        loop._arbitrate_session_enrichment_incidents(graph, "s2")

        incidents = read_incidents(loop._incidents_state_dir)
        assert len(incidents) == 1
        assert incidents[0].count == 2
        assert incidents[0].status == "active"

        resolve_incident(loop._incidents_state_dir, "enrichment_degraded", "anonymize")
        loop._arbitrate_session_enrichment_incidents(graph, "s3")

        incidents = read_incidents(loop._incidents_state_dir)
        assert incidents[0].status == "active"
        assert incidents[0].count == 3

    def test_absent_and_cloud_enabled_touches_nothing(self, tmp_path, monkeypatch):
        """No ``anonymize`` diagnostic (op never ran) and cloud egress IS
        permitted for this run's effective terms: no incident is created or
        resolved — there is genuinely no evidence either way."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        loop = _make_loop(tmp_path, incidents_state_dir=tmp_path / "incidents")
        from paramem.server.incidents import read_incidents, record_incident

        record_incident(
            loop._incidents_state_dir,
            type="enrichment_degraded",
            key="cloud_enrich",
            severity="warning",
            summary="must survive untouched",
            detail={},
        )

        loop._arbitrate_session_enrichment_incidents(self._graph(), "s1")

        by_id = {i.id: i for i in read_incidents(loop._incidents_state_dir)}
        assert by_id["enrichment_degraded:cloud_enrich"].status == "active"
        assert by_id["enrichment_degraded:cloud_enrich"].summary == "must survive untouched"

    def test_absent_and_cloud_disabled_resolves_by_type_with_reason(self, tmp_path):
        """No ``anonymize`` diagnostic and cloud egress is OFF (``cloud_enabled``
        false): every open ``enrichment_degraded`` incident resolves with a
        persisted reason, regardless of key."""
        loop = _make_loop(tmp_path, incidents_state_dir=tmp_path / "incidents", cloud_enabled=False)
        from paramem.server.incidents import read_incidents, record_incident

        record_incident(
            loop._incidents_state_dir,
            type="enrichment_degraded",
            key="anonymize",
            severity="warning",
            summary="stale",
            detail={},
        )
        record_incident(
            loop._incidents_state_dir,
            type="enrichment_degraded",
            key="cloud_enrich",
            severity="warning",
            summary="stale",
            detail={},
        )

        loop._arbitrate_session_enrichment_incidents(self._graph(), "s1")

        for inc in read_incidents(loop._incidents_state_dir):
            assert inc.status == "resolved"
            assert inc.resolved_reason == "cloud egress disabled — enrichment cannot run"

    def test_absent_and_cloud_disabled_with_nothing_standing_writes_no_store(self, tmp_path):
        """The cloud-disabled sweep must not materialise an empty
        ``incidents.json`` when nothing was ever recorded (matches
        ``resolve_incidents_by_type``'s own success-path invariant)."""
        loop = _make_loop(tmp_path, incidents_state_dir=tmp_path / "incidents", cloud_enabled=False)

        loop._arbitrate_session_enrichment_incidents(self._graph(), "s1")

        assert not (loop._incidents_state_dir / "incidents.json").exists()

    def test_unknown_diagnostic_value_with_cloud_off_touches_nothing(self, tmp_path):
        """An unrecognized ``anonymize`` value (neither ``"ok"``,
        ``"opted_out"``, ``"failed"``, nor absent) must NOT be conflated with
        "the op never ran" — even with cloud egress off, a standing incident
        is left exactly as it was, and the by-type sweep never fires.

        Mutation: dropping the explicit ``anonymize_outcome is None`` check
        (falling back to ``else: # absent`` for any non-matched value) makes
        this test fail, since the cloud-disabled sweep would then wrongly
        resolve the standing incident below.
        """
        loop = _make_loop(tmp_path, incidents_state_dir=tmp_path / "incidents", cloud_enabled=False)
        from paramem.server.incidents import read_incidents, record_incident

        record_incident(
            loop._incidents_state_dir,
            type="enrichment_degraded",
            key="cloud_enrich",
            severity="warning",
            summary="must survive untouched",
            detail={},
        )

        loop._arbitrate_session_enrichment_incidents(
            self._graph(anonymize="not_a_real_outcome"), "s1"
        )

        inc = read_incidents(loop._incidents_state_dir)[0]
        assert inc.status == "active"
        assert inc.resolved_reason is None
        assert inc.summary == "must survive untouched"

    def test_incidents_state_dir_none_is_a_safe_noop(self, tmp_path):
        """No ``incidents_state_dir`` configured -> the guard returns
        immediately regardless of diagnostics content."""
        loop = _make_loop(tmp_path)
        assert loop._incidents_state_dir is None

        with patch("paramem.server.incidents.record_incident") as record_mock:
            loop._arbitrate_session_enrichment_incidents(self._graph(anonymize="failed"), "s1")

        record_mock.assert_not_called()


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
        "procedural" in model.peft_config so ensure_adapters skips creation.

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
        # Pre-populate "procedural" so ensure_adapters skips create_adapter.
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
            extraction_enrichment_provider="anthropic",
            extraction_enrichment_provider_model="claude-sonnet-4-6",
            extraction_scrub={"person name"},
            extraction_max_tokens=8192,
            extraction_plausibility_max_tokens=8192,
            extraction_anonymize_token_envelope=8192,
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

        Pre-seed the store with graph250 (beyond the donor's reserved
        graph1-graph200 band, paramem.training.donor.DONOR_KEY_BAND_WIDTH)
        before constructing the loop so the constructor's high-water scan
        sets _indexed_next_index to 251 — proving the store-derived
        high-water value can raise the floor (DONOR_KEY_FLOOR=201), not just
        default to it. Inject one keyless episodic edge. The minted key must
        be graph251 — not graph201 (the bare floor, ignoring the store) and
        not graph250 (collision with the pre-existing key).
        """
        from paramem.memory.store import MemoryStore as _MS
        from paramem.training.key_registry import KeyRegistry

        # Build and hydrate the store BEFORE loop construction so the
        # constructor's _indexed_next_index seeding scan sees graph250.
        store = _MS(replay_enabled=True)
        for tier in ("episodic", "semantic", "procedural"):
            store.load_registry(tier, KeyRegistry())
        store.put(
            "episodic",
            "graph250",
            {
                "key": "graph250",
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
            extraction_enrichment_provider="anthropic",
            extraction_enrichment_provider_model="claude-sonnet-4-6",
            extraction_scrub={"person name"},
            extraction_max_tokens=8192,
            extraction_plausibility_max_tokens=8192,
            extraction_anonymize_token_envelope=8192,
        )
        loop._probe_passing_keys = lambda adapter_name, entries: {e["key"] for e in entries}

        # Constructor must have picked up graph250 -> _indexed_next_index == 251.
        assert loop._indexed_next_index == 251, (
            f"Expected _indexed_next_index=251 after seeding graph250, "
            f"got {loop._indexed_next_index}"
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

        # Must be graph251 — no collision with the pre-existing graph250.
        assert minted_key == "graph251", (
            f"Expected minted key 'graph251' to avoid collision with graph250, got {minted_key!r}"
        )
        # Pre-existing graph250 entry must still be intact.
        assert "graph250" in loop.store.all_active_keys()

    def test_donor_key_floor_on_empty_store(self, tmp_path):
        """An empty store must seed both counters at DONOR_KEY_FLOOR (201),
        not 1 -- the donor (paramem.training.donor) reserves graph1-200 and
        proc1-200 for its synthetic training population."""
        from paramem.training.donor import DONOR_KEY_FLOOR

        loop = _make_loop(tmp_path, replay_enabled=True)

        assert loop._indexed_next_index == DONOR_KEY_FLOOR == 201
        assert loop._procedural_next_index == DONOR_KEY_FLOOR == 201

    def test_donor_key_floor_high_water_still_wins(self, tmp_path):
        """A store high-water key beyond the reserved band must still raise the
        counter past DONOR_KEY_FLOOR -- the floor is a lower bound, never a
        ceiling on the constructor's max() derivation."""
        from paramem.memory.store import MemoryStore as _MS
        from paramem.training.donor import DONOR_KEY_FLOOR
        from paramem.training.key_registry import KeyRegistry

        store = _MS(replay_enabled=True)
        for tier in ("episodic", "semantic", "procedural"):
            store.load_registry(tier, KeyRegistry())
        store.put(
            "procedural",
            "proc300",
            {
                "key": "proc300",
                "question": "q",
                "answer": "a",
                "subject": "speaker0",
                "predicate": "has_interest",
                "object": "kayaking",
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
            extraction_enrichment_provider="anthropic",
            extraction_enrichment_provider_model="claude-sonnet-4-6",
            extraction_scrub={"person name"},
            extraction_max_tokens=8192,
            extraction_plausibility_max_tokens=8192,
            extraction_anonymize_token_envelope=8192,
        )

        assert loop._procedural_next_index == 301, (
            f"Expected _procedural_next_index=301 (high-water beats "
            f"DONOR_KEY_FLOOR={DONOR_KEY_FLOOR}), got {loop._procedural_next_index}"
        )
        assert loop._indexed_next_index == DONOR_KEY_FLOOR

    def test_donor_key_floor_seeds_from_stale_keys_too(self, tmp_path):
        """Regression: a key soft-staled AFTER put must still
        raise the constructor's high-water counter. Seeding from
        all_active_keys() alone (the pre-fix behaviour) would miss a stale
        highest key, re-mint its numeric id on the next fold, and
        set_simhash would then route the new fingerprint into the stale
        record (paramem.training.key_registry) -- a silent recall miss on the
        reissued key. all_known_keys() (active UNION stale) closes this."""
        from paramem.memory.store import MemoryStore as _MS
        from paramem.training.key_registry import KeyRegistry

        store = _MS(replay_enabled=True)
        for tier in ("episodic", "semantic", "procedural"):
            store.load_registry(tier, KeyRegistry())
        store.put(
            "episodic",
            "graph260",
            {
                "key": "graph260",
                "question": "q",
                "answer": "a",
                "subject": "Prior",
                "predicate": "knows",
                "object": "Fact",
            },
        )
        # Soft-stale the key BEFORE loop construction -- it must no longer be
        # active, but its numeric id must still be protected from reissue.
        store.discard_keys(["graph260"], mode="stale")
        assert "graph260" not in store.all_active_keys()
        assert "graph260" in store.all_known_keys()

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
            extraction_enrichment_provider="anthropic",
            extraction_enrichment_provider_model="claude-sonnet-4-6",
            extraction_scrub={"person name"},
            extraction_max_tokens=8192,
            extraction_plausibility_max_tokens=8192,
            extraction_anonymize_token_envelope=8192,
        )

        assert loop._indexed_next_index == 261, (
            f"Expected _indexed_next_index=261 (stale graph260 must still bump "
            f"the high-water counter), got {loop._indexed_next_index}"
        )


class TestHarvestApplySplit:
    """Tests verifying the unified edge→entry builder (_build_all_edge_entries_into).

    Covers the defer=True (interim atomicity) and defer=False (fold discipline)
    paths, plus the minted_by_tier / deferred_writes return contract.
    """

    def test_defer_false_produces_writes_and_count(self, tmp_path):
        """defer=False (default) must write to the store exactly once per
        minted key, advance counters, and return (minted_by_tier,
        [<one record per minted key>]).

        The returned record is NOT a second write instruction — ``defer``
        governs only WHEN the store write happens (immediately here, vs.
        deferred to the caller's own flush when ``defer=True``); the record
        itself is always returned so a caller building a crash-resume marker
        (the main-tiers fold) can enrich it, without that caller ever
        re-applying ``store.put``/``set_bookkeeping`` from it.  See
        ``test_defer_false_store_writes_happen_exactly_once`` for the
        call-count proof.
        """
        from paramem.training.key_registry import KeyRegistry

        loop = _make_loop(tmp_path, replay_enabled=True)
        for tier in ("episodic", "semantic", "procedural"):
            loop.store.load_registry(tier, KeyRegistry())

        loop.merger.graph.add_edge("Carol", "London", predicate="visited", relation_type="factual")

        initial_indexed = loop._indexed_next_index

        tier_keyed: dict = {"episodic": [], "semantic": [], "procedural": []}
        minted_by_tier, records = loop._build_all_edge_entries_into(tier_keyed)

        # Exactly one episodic key minted.  defer=False → the immediate
        # commit still happened AND the harvest record is still returned.
        assert minted_by_tier == {"episodic": 1, "procedural": 0}
        assert len(records) == 1, (
            f"defer=False must return one harvest record per minted key "
            f"(the round-trip contract with _persisted_from_entry_and_rec, "
            f"used by the main-tiers crash-resume marker); got {records}"
        )
        entry = tier_keyed["episodic"][0]
        assert records[0]["entry"]["key"] == entry["key"]
        assert len(tier_keyed["episodic"]) == 1

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

    def test_defer_false_store_writes_happen_exactly_once(self, tmp_path):
        """defer=False must write each minted key to the store exactly once —
        the immediate commit inside the walk, never a second write from the
        returned record.

        Pins the "one invocation per transformation" contract: the only
        production consumer of the returned record for a defer=False call
        (the main-tiers fold) uses it solely to enrich the fold_resume.json
        crash-resume marker (``_persisted_from_entry_and_rec``) and never
        re-applies it to the store — see the module docstring for the
        companion defer=True proof
        (``test_defer_true_no_store_writes_no_counter_advance``).
        """
        from unittest.mock import patch

        from paramem.training.key_registry import KeyRegistry

        loop = _make_loop(tmp_path, replay_enabled=True)
        for tier in ("episodic", "semantic", "procedural"):
            loop.store.load_registry(tier, KeyRegistry())

        loop.merger.graph.add_edge("Carol", "London", predicate="visited", relation_type="factual")

        with (
            patch.object(loop.store, "put", wraps=loop.store.put) as mock_put,
            patch.object(
                loop.store, "set_bookkeeping", wraps=loop.store.set_bookkeeping
            ) as mock_bk,
        ):
            tier_keyed: dict = {"episodic": [], "semantic": [], "procedural": []}
            minted_by_tier, records = loop._build_all_edge_entries_into(tier_keyed)

        assert minted_by_tier["episodic"] == 1
        assert len(records) == 1
        mock_put.assert_called_once()
        mock_bk.assert_called_once()

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
    """Tests that GraphTierRefiner.run_enrichment writes ik_keys of
    same_as-contracted edges to merger.removal_ledger with
    reason='enrichment_same_as'.
    """

    def test_same_as_contraction_writes_keyed_edge_to_ledger(self, tmp_path, monkeypatch):
        """A successful same_as contraction that drops a keyed edge writes the
        edge's ik_key to merger.removal_ledger with reason='enrichment_same_as'
        and keep_node set to the surviving node.
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

        # cloud returns surface names; production canonicalizes before graph lookup.
        canned_result = (
            [],
            [["Alice", "Alicia"]],  # keep=alice, drop=alicia
            "raw",
            0,
        )
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with patch(
            "paramem.training.graph_enrich.request_graph_enrichment",
            return_value=canned_result,
        ):
            result = _refiner_for(loop).run_enrichment()

        assert result["same_as_merges"] >= 1, "Contraction must have fired for alice/alicia"
        assert "key_same_as_victim" in loop.merger.removal_ledger, (
            f"Dropped ik_key must appear in merger.removal_ledger; "
            f"ledger={list(loop.merger.removal_ledger.keys())}"
        )
        entry = loop.merger.removal_ledger["key_same_as_victim"]
        assert entry["reason"] == "enrichment_same_as", (
            f"Expected reason='enrichment_same_as'; got {entry['reason']!r}"
        )
        # keep_node is the canonical keep node key
        assert entry["keep_node"] == "alice", (
            f"Expected keep_node='alice' (canonical keep node); got {entry['keep_node']!r}"
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

        # cloud returns surface names; production canonicalizes: "BadKeep" → "badkeep".
        canned_result = (
            [],
            [["BadKeep", "BadDrop"]],
            "raw",
            0,
        )
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        # Patch contracted_nodes to always raise so the contraction fails.
        with (
            patch(
                "paramem.training.graph_enrich.request_graph_enrichment",
                return_value=canned_result,
            ),
            patch("networkx.contracted_nodes", side_effect=ValueError("forced failure")),
        ):
            result = _refiner_for(loop).run_enrichment()

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
        """Edge-then-node resolution order: a speaker-attributed node's
        speaker_id is carried through the edge (the merger stamps it there) and
        into the deferred_writes record with the correct value.

        Route a speaker-attributed Relation through the merger so the EDGE carries
        speaker_id (Case-3 stamps it from the Relation unconditionally).  The
        graph-walk reads edge → subject node attr → "" and produces the correct
        speaker_id in the minted entry.
        """
        from paramem.graph.schema import Relation, SessionGraph
        from paramem.training.key_registry import KeyRegistry

        loop = _make_loop(tmp_path, replay_enabled=True)
        for tier in ("episodic", "semantic", "procedural"):
            loop.store.load_registry(tier, KeyRegistry())

        # Route a speaker Relation through the real merger path — the Case-3
        # net-new insert stamps speaker_id onto the edge.
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
        """Edge-then-node resolution order: a concept node with no speaker_id
        attr lands on the "" terminal fallback.  The edge → node → "" read has
        no default_speaker_id override.
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
    """SYMMETRIC_PREDICATES and _canonicalize_symmetric_predicates deleted."""

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
    """Enrichment routes through GraphMerger.merge_relations.

    There is no direct graph.add_edge path. Enrichment edges go through the
    merger, which:
    - stamps _EDGE_SOURCE_ATTR="graph_enrichment" on the Case-3 net-new insert,
      but only when Relation.edge_source is non-empty,
    - stamps speaker_id from Relation.speaker_id unconditionally,
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
            "paramem.training.graph_enrich.request_graph_enrichment",
            return_value=(rels, [], "raw", 0),
        ):
            result = _refiner_for(loop).run_enrichment()

        assert not result["skipped"]
        # Edge must carry edge_source="graph_enrichment" (the merger stamps it
        # because the Relation's edge_source is non-empty).
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
            "paramem.training.graph_enrich.request_graph_enrichment",
            return_value=(rels, [], "raw", 0),
        ):
            result = _refiner_for(loop).run_enrichment()

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
    """Deferred-flush set_bookkeeping sites pass allow_empty_speaker.

    A keyless concept-node edge carries no speaker attribution, so its
    deferred write reaches set_bookkeeping with ``speaker_id=""``; the flush
    must set ``allow_empty_speaker`` or the empty-speaker guard
    (``MemoryStore.set_bookkeeping``) raises ValueError.
    """

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
    """Seed a real speaker node via the merger (speaker-identity invariant:
    the node key is the lowercase speaker_id, e.g. ``"speaker0"`` — same as
    entity.speaker_id verbatim under the lowercase-uniform speaker-identity
    design).

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
    """Regression: Cloud echoes back the cased speaker id ('speaker0'), which
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
        # by the canonical speaker_id: "speaker0".
        _seed_speaker_node(loop, "speaker0", "Alex")
        # A concept node the speaker relates to.
        loop.merger.graph.add_node("mentoring", attributes={"name": "Mentoring"})

        # Confirm the key convention: canonical lowercase key only.
        assert "speaker0" in loop.merger.graph.nodes

        # cloud emits lowercase speaker id "speaker0" as the subject.
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
            "paramem.training.graph_enrich.request_graph_enrichment",
            return_value=(rels, [], "raw", 0),
        ):
            result = _refiner_for(loop).run_enrichment()

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

        # Speaker node keys are lowercase canonical.
        assert "speaker0" in loop.merger.graph.nodes
        assert "speaker1" in loop.merger.graph.nodes

        # cloud emits BOTH directions, both symmetric=true, lowercase speaker ids.
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
            "paramem.training.graph_enrich.request_graph_enrichment",
            return_value=(rels, [], "raw", 0),
        ):
            result = _refiner_for(loop).run_enrichment()

        assert not result["skipped"]
        # Both directed colleague_of edges survive (not collapsed — both_speakers gate).
        # Edges root at canonical lowercase keys; predicate stored as "colleague of".
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

        Speaker-identity invariant: keep="speaker0" resolves via canonical
        fallback to the lowercase ``speaker{N}`` node key "speaker0".
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
                "paramem.training.graph_enrich.request_graph_enrichment",
                return_value=(rels, same_as, "raw", 0),
            ),
            patch(
                "paramem.training.graph_enrich._safe_to_merge_surface",
                return_value=True,
            ),
        ):
            result = _refiner_for(loop).run_enrichment()

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
    No enrichment, no cloud calls.
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

    All tests seed the graph via _seed_speaker_node /
    GraphTierRefiner.run_enrichment so edges land in merger.graph through
    the real merger (no raw add_edge for enrichment edges).
    """

    def test_gap_filled_single_speaker_predecessor(self, tmp_path, monkeypatch):
        """Role-concept attribute edge inherits speaker_id from the unique speaker.

        Graph: speaker0 →held_role→ 'Senior PM', 'Senior PM' →achievement→ 'Award X'
        (both via cloud canned result).  After enrichment + mint, the minted key for
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

        # cloud emits: bridge edge + role-concept attribute edge.
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
            "paramem.training.graph_enrich.request_graph_enrichment",
            return_value=(rels, [], "raw", 0),
        ):
            result = _refiner_for(loop).run_enrichment()

        assert not result["skipped"]

        # Confirm the role concept node has no own speaker_id (pre-condition for fallback).
        role_node = loop.merger.graph.nodes.get("senior pm", {})
        assert not role_node.get("speaker_id"), (
            "Role concept node must NOT carry a direct speaker_id before fallback"
        )

        # Confirm the speaker node is a predecessor of "senior pm" (bridge edge present).
        # The speaker node key is the casefolded lowercase form ("speaker0").
        preds = list(loop.merger.graph.predecessors("senior pm"))
        assert "speaker0" in preds, (
            f"Bridge edge speaker0 →held_role→ 'senior pm' must be in graph; predecessors={preds}"
        )

        # Mint keys via the unified builder (fold discipline).
        tier_keyed: dict = {"episodic": [], "semantic": [], "procedural": []}
        loop._build_all_edge_entries_into(tier_keyed)

        # Find the minted key for the 'achievement' edge (subject = "senior_pm").
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
            "paramem.training.graph_enrich.request_graph_enrichment",
            return_value=(rels, [], "raw", 0),
        ):
            result = _refiner_for(loop).run_enrichment()

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
            "paramem.training.graph_enrich.request_graph_enrichment",
            return_value=(rels_a, [], "raw", 0),
        ):
            _refiner_for(loop).run_enrichment()

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

        # cloud emits an attribute edge whose SUBJECT is a brand-new concept node
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
            "paramem.training.graph_enrich.request_graph_enrichment",
            return_value=(rels, [], "raw", 0),
        ):
            result = _refiner_for(loop).run_enrichment()

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
# anonymize -> cloud -> de-anonymize contract (paramem.graph.placeholders).
# ---------------------------------------------------------------------------


class TestGraphTierAnonymizationContract:
    """paramem.graph.extractor.request_graph_enrichment now runs the SAME
    anonymize -> cloud -> de-anonymize contract as session-tier extraction
    (_cloud_pipeline), via the shared primitives in paramem.graph.placeholders.
    """

    def test_graph_enrichment_sends_no_real_names(self):
        """No name present as a key in the caller-supplied ``mapping``
        reaches the payload handed to ``_cloud_call``; it renders as its
        placeholder token instead.  A bare ``speaker{N}`` id is never a
        ``mapping`` key in production (the local anonymizer prompt
        forbids mapping it — see :func:`_build_anonymization_mapping`'s
        speaker-anchor invariants), so it legitimately reaches the
        payload UNMASKED — ``request_graph_enrichment`` applies no scope
        gate of its own; it substitutes exactly what ``mapping`` says
        (the model's own mapping is the sole scope authority).

        Mutation: restore the pre-fix call (pass ``triples`` straight to
        ``_cloud_call`` with no anonymization step) -> ``"alice"`` appears
        in the captured payload -> this test fails.
        """
        from paramem.graph.extractor import request_graph_enrichment

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
        payload, graph = _payload_and_graph_for(triples, {"alice": "Person_1"})

        captured: list[str] = []

        def _capture(prompt, *args, **kwargs):
            captured.append(prompt)
            return '{"relations": [], "same_as": []}'

        with patch("paramem.graph.extractor._cloud_call", side_effect=_capture):
            result = request_graph_enrichment(
                payload,
                graph,
                api_key="test-key",
                provider="anthropic",
                filter_model="claude-sonnet-4-6",
            )

        assert result is not None
        assert captured, "Expected the cloud call to be made"
        rendered = captured[0]
        assert "alice" not in rendered, f"Real name 'alice' leaked into payload: {rendered}"
        assert "Person_1" in rendered, f"Expected 'alice' tokenised as Person_1; got: {rendered}"
        assert '"speaker0"' in rendered, (
            f"speaker0 is never a mapping key — it must reach the payload bare: {rendered}"
        )

    def test_graph_enrichment_round_trips_to_real_names(self):
        """Relations cloud returns (naming tokens) come back with real node
        names after ``request_graph_enrichment`` returns; a bare
        ``speaker{N}`` id (never tokenised — not a ``mapping`` key) round
        trips unchanged.

        Mutation: drop the deanon step (return ``new_relations``/
        ``same_as_pairs`` straight from the parsed response, un-substituted)
        -> the placeholder reaches the merger -> this test fails.
        """
        from paramem.graph.extractor import request_graph_enrichment

        # Realistic shape: Cloud can only propose a relation naming a
        # placeholder it was actually SHOWN, so the chunk's triples must
        # carry "alice" for "Person_1" to be within the observed scope.
        triples = [
            {
                "subject": "alice",
                "predicate": "colleague_of",
                "object": "speaker0",
                "relation_type": "social",
                "speaker_id": "speaker0",
            }
        ]
        payload, graph = _payload_and_graph_for(triples, {"alice": "Person_1"})

        canned_raw = (
            '{"relations": [{"subject": "Person_1", "predicate": "colleague_of", '
            '"object": "speaker0", "relation_type": "social", "confidence": 0.9, '
            '"symmetric": true}], "same_as": []}'
        )

        with patch("paramem.graph.extractor._cloud_call", return_value=canned_raw):
            result = request_graph_enrichment(
                payload,
                graph,
                api_key="test-key",
                provider="anthropic",
                filter_model="claude-sonnet-4-6",
            )

        assert result is not None
        new_rels, same_as, _raw, _dropped_count = result
        assert len(new_rels) == 1
        assert new_rels[0]["subject"] == "alice"
        assert new_rels[0]["object"] == "speaker0"
        assert same_as == []

    def test_graph_enrichment_unresolved_token_dropped(self):
        """A relation naming a token in neither the CORE table nor cloud
        bindings is DROPPED at the exit gate, not forwarded with a residual
        placeholder.

        Mutation: remove the exit gate (``_apply_bindings``) on this path
        -> the unresolved token escapes into ``new_relations`` -> this test
        fails.
        """
        from paramem.graph.extractor import request_graph_enrichment

        # "alice" is shown to cloud (declared AND observed); "Person_99" is
        # never declared anywhere — this ONE relation is dropped by
        # ``_apply_bindings``'s fail-closed residual sweep (2026-07-22
        # cloud-admission redesign retired the whole-delta rejection this
        # test used to exercise); the observable outcome (no surviving
        # relation) is unchanged.
        triples = [
            {
                "subject": "alice",
                "predicate": "colleague_of",
                "object": "acme",
                "relation_type": "factual",
                "speaker_id": "",
            }
        ]
        payload, graph = _payload_and_graph_for(triples, {"alice": "Person_1"})
        canned_raw = (
            '{"relations": [{"subject": "Person_1", "predicate": "knows", '
            '"object": "Person_99", "relation_type": "social", '
            '"confidence": 0.9, "symmetric": false}], "same_as": []}'
        )
        with patch("paramem.graph.extractor._cloud_call", return_value=canned_raw):
            result = request_graph_enrichment(
                payload,
                graph,
                api_key="test-key",
                provider="anthropic",
                filter_model="claude-sonnet-4-6",
            )

        assert result is not None
        new_rels, _same_as, _raw, _dropped_count = result
        assert new_rels == [], (
            f"Relation naming an unresolved token must be dropped; got {new_rels!r}"
        )

    def test_graph_enrichment_masks_exactly_what_the_mapping_declares(self):
        """``request_graph_enrichment`` applies NO scope gate of its own
        — it substitutes exactly the entries the caller's
        ``mapping`` declares, nothing more, nothing less.  The caller's
        ``mapping`` is the model's own ``scrub``-scoped decision in
        production; this test proves the function trusts it verbatim
        rather than re-deriving scope from entity types.

        Mutation: hardcode a scope inside ``request_graph_enrichment``
        (ignore what ``mapping`` actually contains) -> either assertion
        below fails.
        """
        from paramem.graph.extractor import request_graph_enrichment

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
        payload_person, graph_person = _payload_and_graph_for(triples, {"alice": "Person_1"})
        with patch("paramem.graph.extractor._cloud_call", side_effect=_capture):
            request_graph_enrichment(
                payload_person,
                graph_person,
                api_key="k",
                provider="anthropic",
                filter_model="claude-sonnet-4-6",
            )
        person_only_payload = captured[-1]
        assert "alice" not in person_only_payload
        assert "acme" in person_only_payload

        # The model classified only "acme" in scope (its own decision —
        # not a code-side entity-type re-derivation).
        payload_org, graph_org = _payload_and_graph_for(triples, {"acme": "Org_1"})
        with patch("paramem.graph.extractor._cloud_call", side_effect=_capture):
            request_graph_enrichment(
                payload_org,
                graph_org,
                api_key="k",
                provider="anthropic",
                filter_model="claude-sonnet-4-6",
            )
        org_only_payload = captured[-1]
        assert "acme" not in org_only_payload
        assert "alice" in org_only_payload

    def test_graph_enrichment_placeholder_threaded_verbatim_never_reminted(self):
        """The model's OWN placeholder (whatever token it minted) is
        threaded straight through to the cloud payload and the returned
        relation — ``request_graph_enrichment`` never re-mints its own
        token for a name already present in ``mapping``, regardless of
        the placeholder's shape (verified with a real anonymizer-style
        surface, not a ``Person_N`` convenience literal).  This is the
        regression this contract exists to prevent: re-minting instead of
        threading through would desync the forward token from what cloud
        is shown, or silently drop a mapping entry.
        """
        from paramem.graph.extractor import request_graph_enrichment

        triples = [
            {
                "subject": "yang ming",
                "predicate": "works_at",
                "object": "acme",
                "relation_type": "factual",
                "speaker_id": "",
            }
        ]
        payload, graph = _payload_and_graph_for(triples, {"yang ming": "Person_7"})

        captured: list[str] = []

        def _capture(prompt, *args, **kwargs):
            captured.append(prompt)
            return (
                '{"relations": [{"subject": "Person_7", "predicate": "works_at", '
                '"object": "acme", "relation_type": "factual", "confidence": 0.9, '
                '"symmetric": false}], "same_as": []}'
            )

        with patch("paramem.graph.extractor._cloud_call", side_effect=_capture):
            result = request_graph_enrichment(
                payload,
                graph,
                api_key="test-key",
                provider="anthropic",
                filter_model="claude-sonnet-4-6",
            )

        assert result is not None
        rendered = captured[0]
        assert "yang ming" not in rendered, f"real name leaked into payload: {rendered}"
        assert "Person_7" in rendered, (
            f"the model's own placeholder must reach the payload verbatim: {rendered}"
        )
        new_rels, _same_as, _raw, _dropped_count = result
        assert len(new_rels) == 1
        assert new_rels[0]["subject"] == "yang ming", (
            "Person_7 must round-trip to the real name via the caller's own "
            f"mapping, not a re-minted token; got {new_rels[0]!r}"
        )

    def test_request_graph_enrichment_sends_exactly_payload_facts(self):
        """``request_graph_enrichment`` sends exactly
        ``payload.facts`` — a fail-closed slice's facts (never present in
        ``payload.facts`` per :func:`~paramem.cloud.anonymize.anonymize`'s
        own contract) never appear in the cloud prompt. Asserted directly
        on the rendered ``triples_json``: only the facts actually present
        in ``payload.facts`` are rendered, nothing else."""
        from paramem.graph.extractor import request_graph_enrichment

        # payload.facts deliberately EXCLUDES a second triple — standing
        # in for what anonymize() does when one slice fails closed: only
        # the survived triple is present in payload.facts.
        survived_triple = {
            "subject": "alice",
            "predicate": "works_at",
            "object": "acme",
            "relation_type": "factual",
            "speaker_id": "",
        }
        payload, graph = _payload_and_graph_for([survived_triple], {"alice": "Person_1"})

        captured: list[str] = []

        def _capture(prompt, *args, **kwargs):
            captured.append(prompt)
            return '{"relations": [], "same_as": []}'

        with patch("paramem.graph.extractor._cloud_call", side_effect=_capture):
            result = request_graph_enrichment(
                payload,
                graph,
                api_key="test-key",
                provider="anthropic",
                filter_model="claude-sonnet-4-6",
            )

        assert result is not None
        rendered = captured[0]
        assert "acme" in rendered
        assert "Person_1" in rendered
        # The rendered triples_json is EXACTLY the substituted
        # payload.facts entry — a fail-closed slice's dropped triple
        # (never present in payload.facts to begin with) leaves no trace.
        # (extractor.py's cloud-facing render is indent=2, unlike the
        # compact json.dumps the local anonymize payload uses — so this
        # compares against that same indent=2 rendering, not the compact
        # local-KV form.)
        from paramem.cloud.placeholders import insert_placeholders

        expected_triples_json = json.dumps(
            insert_placeholders([survived_triple], payload.forward), indent=2
        )
        assert expected_triples_json in rendered

    def test_graph_enrichment_same_as_deanonymized_before_speaker_guard(self):
        """``same_as`` pairs must be real names by the time they LEAVE
        ``request_graph_enrichment`` — i.e. strictly before
        ``graph_enrich.enrich_graph``'s speaker-pair guard
        (``is_speaker_id(keep) and is_speaker_id(drop)``, graph_enrich.py)
        ever sees them. ``is_speaker_id`` cannot recognise a
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
        from paramem.cloud.anonymize import AnonymizedContract
        from paramem.graph.extractor import request_graph_enrichment
        from paramem.graph.schema import SessionGraph
        from paramem.utils.identity import is_speaker_id

        # Hand-built payload (bypassing _build_anonymization_mapping's
        # speaker-key-drop guard on purpose — see docstring: a legal input
        # to this pure function, never how a real caller populates it).
        # Realistic shape otherwise: Cloud can only propose a same_as pair
        # naming placeholders it was actually SHOWN, so the chunk's
        # ``triples`` (fed to ``request_graph_enrichment`` directly, per
        # the interface narrowing — no separate ``anon_facts`` field)
        # must carry Person_1/Person_2 for them to be within the observed
        # scope.
        triples = [
            {
                "subject": "Person_1",
                "predicate": "knows",
                "object": "Person_2",
                "relation_type": "social",
                "speaker_id": "speaker0",
            }
        ]
        reverse = {"Person_1": "speaker0", "Person_2": "speaker1"}
        payload = AnonymizedContract(
            status="ok",
            forward={v: k for k, v in reverse.items()},
            reverse=reverse,
            anon_transcript="",
            declared=frozenset(reverse.keys()),
            norm_stats={"inverted": 0, "dropped": 0},
            rekey_dropped=0,
            raw="",
            facts=triples,
        )
        graph = SessionGraph(session_id="__graph_enrichment_test__", timestamp="")
        canned_raw = '{"relations": [], "same_as": [["Person_1", "Person_2"]]}'

        with patch("paramem.graph.extractor._cloud_call", return_value=canned_raw):
            result = request_graph_enrichment(
                payload,
                graph,
                api_key="test-key",
                provider="anthropic",
                filter_model="claude-sonnet-4-6",
            )

        assert result is not None
        _new_rels, same_as, _raw, _dropped_count = result
        assert len(same_as) == 1
        keep, drop = same_as[0]
        assert is_speaker_id(keep) and is_speaker_id(drop), (
            "same_as pair must be bare speaker ids by the time it leaves "
            f"request_graph_enrichment — the speaker-pair guard depends on this; got {same_as[0]!r}"
        )
        assert {keep, drop} == {"speaker0", "speaker1"}

    def test_graph_enrichment_full_pipeline_blocks_speaker_same_as_merge(
        self, tmp_path, monkeypatch
    ):
        """Production-wiring confirmation: driving the same scenario through
        the REAL ``GraphTierRefiner.run_enrichment`` (only
        ``_cloud_call`` mocked; the local anonymizer stays on the module's
        default autouse stub, which masks non-speaker names as ``person``
        — speaker ids are never tokenised regardless, since the local
        anonymizer prompt forbids mapping ``speaker{N}`` in the first
        place) the two distinct speaker nodes are never contracted — the
        speaker-pair guard fires against the ``same_as`` pair ``request_graph_enrichment``
        returns, even though the cloud payload carried the BARE speaker ids
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

        def _cloud_response(prompt, *args, **kwargs):
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
        with patch("paramem.graph.extractor._cloud_call", side_effect=_cloud_response):
            result = _refiner_for(loop).run_enrichment()

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
    like. These tests exist specifically to drive
    ``GraphTierRefiner.run_enrichment`` against a graph with no usable
    type signal, proving the type source is now the local-model
    anonymization pass, not node attributes.
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
    """Build a stand-in for ``anonymize_transcript`` typing real names
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
    from paramem.cloud.placeholders import _substitute_whole_words
    from paramem.config.taxonomy import entity_type_to_prefix

    def _stub(facts, model, tokenizer, transcript="", **kwargs):
        # ``facts`` is a plain fact-dict list (interface narrowing,
        # 2026-07-21) — never a ``SessionGraph`` — so names come off
        # ``subject``/``object`` keys directly, not ``.relations``.
        names = sorted(
            {str(f.get("subject", "")) for f in facts} | {str(f.get("object", "")) for f in facts}
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
    own (production reality — see
    ``graph_enrich.enrich_graph``'s docstring and ``GraphMerger``'s
    ``entity_type="concept"`` fallback for endpoints without a known
    Entity). These tests drive the REAL
    ``GraphTierRefiner.run_enrichment`` against a graph built with
    that fallback (``_populate_untyped_graph`` — every node typed
    ``"concept"``, unlike ``_populate_graph``'s test-convenience typing)
    and a controlled local-model classification (``_stub_local_model_types``,
    replacing this module's default empty-mapping ``_stub_local_anonymize``
    fixture), proving that entity types now come from the local model's own
    classification rather than from node attributes.
    """

    def test_graph_enrichment_masks_persons(self, tmp_path, monkeypatch):
        """A real name the LOCAL model classifies as a person must not reach
        the payload handed to ``_cloud_call`` verbatim, even though the fold
        graph node itself carries no usable ``entity_type``.

        Mutation: revert ``GraphTierRefiner.run_enrichment`` to reading
        node ``entity_type`` attributes (all ``"concept"`` here) instead of
        the local model's mapping -> "person0" is never masked -> this
        test fails.
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
            patch("paramem.cloud.anonymize.anonymize_transcript", side_effect=stub),
            patch("paramem.graph.extractor._cloud_call", side_effect=_capture),
        ):
            result = _refiner_for(loop).run_enrichment()

        assert not result["skipped"]
        assert captured, "Expected the cloud call to be made"
        payload = captured[0]
        assert "person0" not in payload, f"Real name 'person0' leaked into payload: {payload}"
        assert "Person_" in payload, f"Expected person nodes tokenised as Person_N; got: {payload}"

    def test_graph_enrichment_leaves_out_of_scope_verbatim(self, tmp_path, monkeypatch):
        """A node the LOCAL model's own mapping OMITS (the model's scope
        decision against ``scrub``, e.g. an organization when only
        ``person name`` is configured) must appear VERBATIM in the cloud
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

        def _stub_omits_acmecorp(facts, model, tokenizer, transcript="", **kwargs):
            # ``facts`` is a plain fact-dict list — never a ``SessionGraph``.
            names = sorted(
                {str(f.get("subject", "")) for f in facts}
                | {str(f.get("object", "")) for f in facts}
            )
            mapping = {
                name: f"Person_{i + 1}"
                for i, name in enumerate(n for n in names if n != "acmecorp")
            }
            return mapping, "stub-anon-transcript", "stub-raw"

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with (
            patch(
                "paramem.cloud.anonymize.anonymize_transcript",
                side_effect=_stub_omits_acmecorp,
            ),
            patch("paramem.graph.extractor._cloud_call", side_effect=_capture),
        ):
            result = _refiner_for(loop).run_enrichment()

        assert not result["skipped"]
        assert captured, "Expected the cloud call to be made"
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
        """A new relation cloud returns (naming the LOCAL model's own tokens)
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

        def _cloud_response(prompt, *args, **kwargs):
            # Reference the SAME Person_N tokens the local-model stub just
            # minted for person0/person1 (sorted-name order: person0 -> 1st).
            return (
                '{"relations": [{"subject": "Person_1", "predicate": "colleague_of", '
                '"object": "Person_2", "relation_type": "social", "confidence": 0.9}], '
                '"same_as": []}'
            )

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with (
            patch("paramem.cloud.anonymize.anonymize_transcript", side_effect=stub),
            patch("paramem.graph.extractor._cloud_call", side_effect=_cloud_response),
        ):
            result = _refiner_for(loop).run_enrichment()

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
    ``graph_enrich.enrich_graph`` itself; the shared
    ``_substitute_whole_words`` primitive stays exact-match everywhere (see
    ``tests/test_placeholders.py::TestSubstituteWholeWordsExactMatchRegression``
    for why it must). The fold graph's node keys are already canonicalized
    (``"yang ming"``), while the local model's mapping is keyed by
    whatever real-name surface it independently produced (``"Yang
    Ming"``) — a raw comparison between the two silently misses, so the
    real name would reach the cloud payload unmasked even though the model
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
        ``graph_enrich.enrich_graph`` -> the real node text
        ("yang ming") reaches the cloud payload verbatim -> this test fails.
        """
        loop = _make_loop(tmp_path)
        graph = loop.merger.graph
        self._populate_two_node_chunk(graph, "yang ming", "acmecorp")

        captured: list[str] = []

        def _capture(prompt, *args, **kwargs):
            captured.append(prompt)
            return '{"relations": [], "same_as": []}'

        monkeypatch.setattr(
            "paramem.cloud.anonymize.anonymize_transcript",
            lambda *args, **kwargs: ({"Yang Ming": "Person_1"}, "stub-anon-transcript", "stub-raw"),
        )
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with patch("paramem.graph.extractor._cloud_call", side_effect=_capture):
            result = _refiner_for(loop).run_enrichment()

        assert not result["skipped"]
        assert captured, "Expected the cloud call to be made"
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
            "paramem.cloud.anonymize.anonymize_transcript",
            lambda *args, **kwargs: (
                {"Someone Else": "Person_1"},
                "stub-anon-transcript",
                "stub-raw",
            ),
        )
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        call_spy = MagicMock()
        with patch("paramem.training.graph_enrich.request_graph_enrichment", call_spy):
            result = _refiner_for(loop).run_enrichment()

        assert not result["skipped"]
        assert result["mapping_rekey_dropped"] >= 1
        # "Someone Else" names nothing in the chunk -- chunk_mapping ends
        # up empty, tripping the pre-existing empty-mapping fail-closed
        # guard (leg 2), so no cloud call fires at all.
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
            "paramem.cloud.anonymize.anonymize_transcript",
            lambda *args, **kwargs: ({"Yang Ming": "Person_1"}, "stub-anon-transcript", "stub-raw"),
        )
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        call_spy = MagicMock()
        with patch("paramem.training.graph_enrich.request_graph_enrichment", call_spy):
            result = _refiner_for(loop).run_enrichment()

        assert not result["skipped"]
        assert result["mapping_rekey_dropped"] >= 1
        call_spy.assert_not_called()


class TestGraphEnrichmentUsesSharedPrimitives:
    """Structural guard mirroring ``tests/test_extraction_pipeline_guard.py``:
    the graph-tier anonymization contract must route entirely through
    ``paramem.cloud.placeholders`` — no second mint/table-build/deanon
    implementation may appear in ``paramem/training/consolidation.py``.
    """

    _PLACEHOLDER_PRIMITIVE_NAMES = frozenset(
        {
            "_build_anonymization_mapping",
            "_apply_bindings",
            "_normalize_anonymization_mapping",
            "_resolution_map",
            "mint_placeholder",
            "_substitute_whole_words",
            "_declared_placeholder_tokens",
            "_contains_declared_token",
        }
    )

    def test_no_duplicate_primitive_defined_in_consolidation(self):
        """``consolidation.py`` must not define a function sharing a name
        with a ``paramem.cloud.placeholders`` primitive — that would be a
        duplicate mint/table/deanon implementation living outside the
        shared module.

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
            f"name with a paramem.cloud.placeholders primitive: {sorted(collision)} — "
            "this is a duplicate mint/table/deanon implementation. Route "
            "through paramem.cloud.placeholders instead."
        )

    def test_consolidation_does_not_reimplement_placeholder_shape_regex(self):
        """``consolidation.py`` must not hardcode its own placeholder-shape
        pattern (PascalCase_N) — that regex lives ONLY in
        ``paramem/cloud/placeholders.py``.
        """
        from pathlib import Path

        repo_root = Path(__file__).resolve().parent.parent
        target = repo_root / "paramem" / "training" / "consolidation.py"
        text = target.read_text()
        assert "A-Z][A-Za-z]*_" not in text, (
            "consolidation.py appears to hardcode the placeholder-shape "
            "regex — this pattern must live only in paramem.cloud.placeholders."
        )

    # -------------------------------------------------------------------
    # Strengthening: the two tests above only scan ONE file
    # (consolidation.py) for function-DEFINITIONS and a regex LITERAL —
    # they do not inspect calls or imports, and do not cover the rest of
    # paramem/.  This is new machinery: an import/call guard over every
    # tracked file under paramem/, asserting the anon/deanon primitives
    # below are reachable ONLY through the ``paramem/cloud/`` round-trip
    # package — ``anonymize.py`` / ``deanonymize.py`` (the two composed
    # halves of the one round-trip contract) or ``placeholders.py`` itself
    # (which legitimately calls invert_forward_mapping and
    # _resolution_map internally).
    #
    # ``insert_placeholders`` is deliberately NOT in this set (moved out
    # when ``AnonymizedContract.anon_facts`` was removed as a stored
    # field): it carries no privacy-guard logic of its own — no
    # speaker-value guard, no binding-collision scan, no observed scoping, just a
    # mechanical substitution over the forward map — so, unlike the four
    # primitives below, straying from the ``cloud/`` package does not
    # bypass anything SAFETY-critical. Every production reader now
    # derives the anonymized fact array on demand via
    # ``insert_placeholders(<facts>, payload.forward)`` instead of
    # reading a payload-native field: the ``enrich`` stage
    # (``paramem/graph/stage_enrich.py``), ``request_graph_enrichment``
    # (``paramem/graph/extractor.py``), and the ``/calibrate/anonymize``
    # handler (``paramem/server/calibrate.py``, status-gated to ``[]`` on
    # failure). Guarding it here would forbid the very design those three
    # call sites implement.
    # -------------------------------------------------------------------

    _CLOUD_ROUNDTRIP_ONLY_PRIMITIVES = frozenset(
        {
            "_build_anonymization_mapping",
            "_binding_collisions",
            "_apply_bindings",
            "_resolution_map",
        }
    )

    _CLOUD_ROUNDTRIP_ALLOWED_FILES = frozenset(
        {
            "paramem/cloud/anonymize.py",
            "paramem/cloud/deanonymize.py",
            "paramem/cloud/placeholders.py",
        }
    )

    @classmethod
    def _find_guarded_primitive_sites(cls, py_file) -> list[tuple[int, str, str]]:
        """Return ``(lineno, source, name)`` for every import of, or
        module-qualified call to, a guarded primitive in ``py_file``.

        Two reach patterns are checked:

        1. ``from paramem.cloud.placeholders import <name>`` (direct
           import — the structural signal that a caller is reaching for
           the primitive itself, regardless of whether it is then
           called).
        2. ``<module_alias>.<name>(...)`` — a module-qualified call (e.g.
           ``placeholders._apply_bindings(...)`` after
           ``from paramem.cloud import placeholders`` or
           ``import paramem.cloud.placeholders as placeholders``).
        """
        import ast

        try:
            text = py_file.read_text()
        except UnicodeDecodeError:
            return []
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return []
        lines = text.splitlines()
        out: list[tuple[int, str, str]] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "paramem.cloud.placeholders":
                for alias in node.names:
                    if alias.name in cls._CLOUD_ROUNDTRIP_ONLY_PRIMITIVES:
                        line = lines[node.lineno - 1] if 0 < node.lineno <= len(lines) else ""
                        out.append((node.lineno, line.strip(), alias.name))
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in cls._CLOUD_ROUNDTRIP_ONLY_PRIMITIVES
            ):
                line = lines[node.lineno - 1] if 0 < node.lineno <= len(lines) else ""
                out.append((node.lineno, line.strip(), node.func.attr))
        return out

    def test_placeholders_primitives_reached_only_via_cloud_roundtrip(self):
        """Every one of the four primitives that make the anon/deanon
        contract SAFE (speaker-value guard, binding-collision scan, observed
        scoping) is imported/called, within ``paramem/``, ONLY from
        ``paramem/cloud/anonymize.py`` / ``paramem/cloud/deanonymize.py``
        (the one round-trip contract, split into its two composed halves)
        — or from ``paramem/cloud/placeholders.py`` itself.

        Mutation: reintroduce a direct
        ``from paramem.cloud.placeholders import _apply_bindings`` (or
        any of the other three names) in ``extractor.py`` /
        ``consolidation.py`` / ``inference.py`` / any other
        ``paramem/`` module -> this test fails.
        """
        from pathlib import Path

        from tests._guard_utils import tracked_python_files

        repo_root = Path(__file__).resolve().parent.parent
        offenders: list[tuple[str, int, str, str]] = []

        for py_file in tracked_python_files(repo_root):
            rel = py_file.relative_to(repo_root).as_posix()
            if not rel.startswith("paramem/"):
                continue
            if rel in self._CLOUD_ROUNDTRIP_ALLOWED_FILES:
                continue
            for lineno, src, name in self._find_guarded_primitive_sites(py_file):
                offenders.append((rel, lineno, src, name))

        assert not offenders, (
            "anon/deanon primitives reached outside paramem/cloud/{anonymize,"
            "deanonymize}.py (the structural guard that makes the speaker-value "
            "guard, binding-collision scan, and observed scoping unbypassable):\n"
            + "\n".join(f"  {path}:{line} [{name}] — {src}" for path, line, src, name in offenders)
        )


class TestGraphEnrichmentFailureLoudness:
    """The graph tier must FAIL LOUD on a programming error and skip
    gracefully only on a genuine runtime condition.

    Two swallows used to hide a malformed prompt template: an inner
    ``except KeyError`` around ``enrichment_prompt.format(...)`` in
    ``request_graph_enrichment``, and a broad ``except Exception`` around the
    whole chunk body in ``graph_enrich.enrich_graph``.  Together
    they turned a
    missed brace-doubling in ``cloud_graph_enrichment.txt`` into a permanent,
    SILENT outage of graph enrichment.  Both tests below are needed: each
    pins one half, so an inert half-fix is caught.
    """

    def test_graph_enrichment_prompt_format_error_propagates(self, tmp_path, monkeypatch):
        """A prompt template with an un-doubled literal brace raises
        ``KeyError`` out of ``graph_enrich.enrich_graph`` — it does
        not silently disable enrichment.

        Mutation: re-add EITHER the inner ``except KeyError`` in
        ``request_graph_enrichment`` OR the broad ``except Exception`` in
        ``graph_enrich.enrich_graph`` -> the KeyError is swallowed
        and the fold "succeeds" with zero enrichment -> this test fails.
        """
        from paramem.graph.prompts import _load_prompt as _real_load_prompt

        loop = _make_loop(tmp_path)
        _populate_untyped_graph(loop.merger.graph)

        def _bad_prompt(filename, *args, **kwargs):
            if filename == "cloud_graph_enrichment.txt":
                # {oops} is a literal brace the author forgot to double.
                return "Triples:\n{triples_json}\nSchema: {oops}"
            return _real_load_prompt(filename, *args, **kwargs)

        stub = _stub_local_model_types({})
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with (
            patch("paramem.cloud.anonymize.anonymize_transcript", side_effect=stub),
            patch("paramem.graph.extractor._load_prompt", side_effect=_bad_prompt),
            patch(
                "paramem.graph.extractor._cloud_call",
                side_effect=AssertionError("the cloud must never be called with a broken prompt"),
            ),
            pytest.raises(KeyError, match="oops"),
        ):
            _refiner_for(loop).run_enrichment()

    def test_graph_enrichment_cuda_driver_fault_degrades_the_pass(self, tmp_path, monkeypatch):
        """A CUDA "device not ready" fault from the local ``generate()`` is
        converted to ``VramExhausted`` by ``vram_scope`` (``anonymize.py``)
        BEFORE it ever reaches ``graph_enrich.enrich_graph``'s exception
        handlers — so this test patches ``generate_answer``, the actual GPU
        call, rather than ``anonymize_transcript`` (which would bypass
        ``vram_scope`` entirely and exercise a branch production cannot
        reach).

        The pass degrades: it stops, keeps whatever it already merged
        (nothing here — the fault is on the only chunk), and returns
        normally with ``aborted_reason == "vram"`` rather than raising or
        silently "skipping the chunk" and continuing.

        Mutation: revert the ``except VramExhausted`` branch to ``raise`` ->
        this test's ``run_enrichment()`` call raises instead of returning a
        degrade result.
        """
        loop = _make_loop(tmp_path)
        _populate_untyped_graph(loop.merger.graph)

        def _boom(*args, **kwargs):
            raise RuntimeError("CUDA error: device not ready")

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with (
            # This suite runs with CUDA_VISIBLE_DEVICES="" (tests/conftest.py's
            # CUDA isolation gate), under which vram_scope no-ops entirely — so
            # without this, the fault would propagate WITHOUT going through
            # vram_scope's conversion, silently re-introducing the bypass this
            # test exists to close. Mirrors tests/test_vram_guard.py's own
            # pattern for exercising vram_scope's real branches off-GPU.
            patch("paramem.utils.vram_guard.torch.cuda.is_available", return_value=True),
            patch("paramem.utils.vram_guard.torch.cuda.empty_cache"),
            # Restore the REAL anonymize_transcript (the autouse fixture at
            # the top of this file stubs it out for every other test in this
            # module) so the fault travels through vram_scope, exactly as it
            # does in production.
            patch(
                "paramem.cloud.anonymize.anonymize_transcript",
                side_effect=_real_anonymize_transcript,
            ),
            patch("paramem.cloud.anonymize.generate_answer", side_effect=_boom),
            patch(
                "paramem.graph.extractor._cloud_call",
                side_effect=AssertionError("the cloud must not be called for a faulted chunk"),
            ),
        ):
            result = _refiner_for(loop).run_enrichment()

        assert result["skipped"] is False
        assert result["aborted_reason"] == "vram"
        assert result["chunks"] == 0, "no cloud call may be made for a chunk that faulted locally"
        assert result["new_edges"] == 0

    def test_graph_enrichment_non_driver_runtime_error_skips_the_chunk(self, tmp_path, monkeypatch):
        """A genuinely non-driver-fault ``RuntimeError`` from the local
        ``generate()`` is NOT converted by ``vram_scope`` (it only converts
        the "device not ready" / "CUDA driver error" marker classes) — it
        propagates unchanged and reaches ``enrich_graph``'s ``except
        RuntimeError`` branch, which is genuinely reachable for this case
        (unlike the CUDA-driver-fault case above). It skips just this chunk
        and lets the pass finish normally, ``aborted_reason`` staying
        ``None`` — the "honestly scoped" counterpart to the driver-fault
        test above.

        Mutation: narrow the handler to nothing (delete the ``except
        RuntimeError``) -> the RuntimeError kills the whole fold -> this
        test fails.
        """
        loop = _make_loop(tmp_path)
        _populate_untyped_graph(loop.merger.graph)

        def _boom(*args, **kwargs):
            raise RuntimeError("some unrelated local generate failure")

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with (
            # See the driver-fault test above for why CUDA must be reported
            # available and the real anonymize_transcript restored here.
            patch("paramem.utils.vram_guard.torch.cuda.is_available", return_value=True),
            patch("paramem.utils.vram_guard.torch.cuda.empty_cache"),
            patch(
                "paramem.cloud.anonymize.anonymize_transcript",
                side_effect=_real_anonymize_transcript,
            ),
            patch("paramem.cloud.anonymize.generate_answer", side_effect=_boom),
            patch(
                "paramem.graph.extractor._cloud_call",
                side_effect=AssertionError("the cloud must not be called for a failed chunk"),
            ),
        ):
            result = _refiner_for(loop).run_enrichment()

        assert result["skipped"] is False
        assert result["aborted_reason"] is None
        assert result["chunks"] == 0, "no cloud call may be made for a chunk that failed locally"
        assert result["new_edges"] == 0

    def test_graph_enrichment_fatal_cuda_fault_propagates(self, tmp_path, monkeypatch):
        """A sticky, process-fatal CUDA context fault (``vram_guard.
        is_fatal_cuda_fault``'s contract — recovery is ``os._exit`` + process
        restart, never an in-process release) must escape ``enrich_graph``
        rather than being logged and swallowed as a skipped chunk.

        Mutation: drop the ``if is_fatal_cuda_fault(exc): raise`` guard at
        the top of the ``except RuntimeError`` branch -> this test's
        ``run_enrichment()`` call swallows the fault instead of propagating
        it.
        """
        loop = _make_loop(tmp_path)
        _populate_untyped_graph(loop.merger.graph)

        def _boom(*args, **kwargs):
            raise RuntimeError("CUDA error: an illegal memory access was encountered")

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with (
            # See the driver-fault test above for why CUDA must be reported
            # available and the real anonymize_transcript restored here.
            patch("paramem.utils.vram_guard.torch.cuda.is_available", return_value=True),
            patch("paramem.utils.vram_guard.torch.cuda.empty_cache"),
            patch(
                "paramem.cloud.anonymize.anonymize_transcript",
                side_effect=_real_anonymize_transcript,
            ),
            patch("paramem.cloud.anonymize.generate_answer", side_effect=_boom),
            patch(
                "paramem.graph.extractor._cloud_call",
                side_effect=AssertionError(
                    "the cloud must not be called for a fatally-faulted chunk"
                ),
            ),
            pytest.raises(RuntimeError, match="illegal memory access"),
        ):
            _refiner_for(loop).run_enrichment()

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
                "paramem.cloud.anonymize.anonymize_transcript",
                side_effect=_parse_failure,
            ),
            patch(
                "paramem.graph.extractor._cloud_call",
                side_effect=AssertionError(
                    "the cloud must not be called for a chunk with no local mapping"
                ),
            ),
        ):
            result = _refiner_for(loop).run_enrichment()

        assert not result["skipped"]
        assert result["chunks"] == 0, "no cloud call may be made when the mapping is None"
        assert result["new_edges"] == 0


class TestChunkTelemetryLogging:
    """``enrich_graph`` logs chunk-identifying context (index/total/node
    count/triple count) immediately before each ``anonymize`` call, so a
    fold's log stream can be read as a per-chunk sequence rather than one
    anonymous "vram_scope[anonymize]" entry per call — see the module's
    per-chunk loop just above the ``anonymize(...)`` call site.
    """

    def test_logs_chunk_index_total_nodes_and_triples(self, tmp_path, monkeypatch, caplog):
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
            0,  # accepted: no relations dropped
        )
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        caplog.set_level(logging.INFO, logger="paramem.training.graph_enrich")
        with patch(
            "paramem.training.graph_enrich.request_graph_enrichment",
            return_value=canned_result,
        ):
            result = _refiner_for(loop).run_enrichment()

        assert not result["skipped"]
        assert result["chunks"] == 1, "expected exactly one chunk (11 nodes < 50-cap)"

        chunk_lines = [
            r.getMessage()
            for r in caplog.records
            if "graph_enrichment: anonymize chunk" in r.message
        ]
        assert len(chunk_lines) == 1, f"expected one chunk-telemetry line, got: {caplog.text}"
        line = chunk_lines[0]
        # Single chunk: index 1 of total 1.
        assert "chunk 1/1" in line, line
        # Node/triple counts are the loop-local values, not hardcoded zeros.
        assert "nodes=" in line and "triples=" in line
        nodes_part = line.split("nodes=")[1].split(" ")[0]
        triples_part = line.split("triples=")[1]
        assert int(nodes_part) > 0
        assert int(triples_part) > 0


class TestNormalizationNamesTheSurvivorKey:
    """A predicate-synonym collapse retires the loser's KEY, and that key's
    durable maturity lives in the registry, not on the graph edge.  The ledger
    entry therefore has to name the surviving key, or the fold's credit pass
    has no target and a promoted fact is silently demoted to episodic while
    the fold reports zero loss.
    """

    @staticmethod
    def _two_predicate_graph(loop, *, survivor_key: str | None, retired_key: str):
        """(person0 → acmecorp) carrying 'works at' and 'employed by'."""
        from paramem.memory.persistence import _IK_KEY_ATTR

        graph = loop.merger.graph
        _populate_graph(graph, n_persons=10)  # clears the 10-node floor

        established_eid = next(iter(graph["person0"]["acmecorp"]))
        established = graph["person0"]["acmecorp"][established_eid]
        established["reinforcement_count"] = 2  # outranks the paraphrase on rec
        established["last_seen"] = "s001"
        if survivor_key is not None:
            established[_IK_KEY_ATTR] = survivor_key

        graph.add_edge(
            "person0",
            "acmecorp",
            predicate="employed by",
            relation_type="factual",
            confidence=0.9,
            reinforcement_count=1,
            last_seen="s999",
            sessions=["s999"],
            **{_IK_KEY_ATTR: retired_key},
        )
        return graph

    @staticmethod
    def _run(loop):
        canned = (
            {("person0", "acmecorp"): [["works at", "employed by"]]},
            {"model_calls": 1, "raw_outputs": []},
        )
        with patch(
            "paramem.training.graph_tier.normalize_predicates",
            return_value=canned,
        ):
            return _refiner_for(loop).run_normalization()

    def test_ledger_names_the_surviving_key_and_predicate(self, tmp_path):
        loop = _make_loop(tmp_path)
        self._two_predicate_graph(loop, survivor_key="graph_keep", retired_key="graph_drop")

        assert self._run(loop)["edges_retired"] == 1

        entry = loop.merger.removal_ledger["graph_drop"]
        assert entry["reason"] == "predicate_synonym_collapse"
        assert entry["survivor_key"] == "graph_keep", (
            "the credit pass reads survivor_key; without it the retired key's "
            f"maturity has nowhere to go; got {entry!r}"
        )
        assert entry["survivor_predicate"] == "works at"

    def test_keyless_survivor_adopts_the_retired_key(self, tmp_path):
        """A pending-session or enrichment edge can outrank a keyed one, which
        would retire the keyed edge and re-mint the fact under a fresh key at
        reinforcement 1.  The survivor adopts the key instead, so the fact keeps
        its registry row — and there is no removal to record at all.
        """
        from paramem.memory.persistence import _IK_KEY_ATTR

        loop = _make_loop(tmp_path)
        graph = self._two_predicate_graph(loop, survivor_key=None, retired_key="graph_mature")

        assert self._run(loop)["edges_retired"] == 1

        surviving = [
            d
            for _, _, d in graph.out_edges("person0", data=True)
            if d.get("predicate") == "works at"
        ]
        assert len(surviving) == 1
        assert surviving[0].get(_IK_KEY_ATTR) == "graph_mature", (
            "the keyless survivor must adopt the retired key rather than let it "
            f"be staled; got {surviving[0].get(_IK_KEY_ATTR)!r}"
        )
        assert "graph_mature" not in loop.merger.removal_ledger, (
            "an adopted key moved to the survivor edge — it is not a removal and "
            "must not be soft-staled"
        )
