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

from paramem.config.taxonomy import resolve_scrub_categories
from paramem.graph.schema import SessionGraph
from paramem.memory.persistence import _EDGE_SOURCE_ATTR
from paramem.training.consolidation import ConsolidationLoop, PendingRelations
from paramem.training.graph_enrich import serialize_subgraph_triples
from paramem.training.graph_tier import GraphTierRefiner
from paramem.utils.config import AdapterConfig, ConsolidationConfig, TrainingConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _stub_local_anonymize(monkeypatch):
    """Default stub for ``anonymize`` (THE one anonymize chain).

    ``graph_enrich.enrich_graph`` runs the local anonymizer (the SAME
    primitive session-tier extraction uses) over each chunk BEFORE the
    cloud call, to derive real-name entity types the fold graph itself
    cannot supply (see that function's docstring). ``_make_loop``'s
    model/tokenizer are ``MagicMock()``s, so a real call always fails to
    parse (there is no JSON in a ``MagicMock``'s generated output), which
    would fail every chunk closed (skip the cloud call entirely) before
    ``request_graph_enrichment`` is ever reached — breaking every test below
    that mocks ``request_graph_enrichment`` to verify cloud-response
    consumption. The patch target is ``paramem.training.graph_enrich.
    anonymize`` — the NAME ``graph_enrich.py`` binds via its own
    module-level ``from paramem.cloud.anonymize import anonymize`` import
    and calls bare. ``anonymize()`` is the one entry point every caller,
    including this stub, replaces wholesale — there is no lower-level
    single-call primitive underneath the chain to intercept instead.

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
        "paramem.training.graph_enrich.anonymize",
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
        extraction_scrub_categories=resolve_scrub_categories(["person name"]),
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

    # Allow callers to supply a pre-built ConsolidationConfig so tests can set
    # fields like refinement_enrichment without touching other knobs.
    consolidation_config = defaults.pop("consolidation_config", ConsolidationConfig())

    from paramem.memory.store import MemoryStore as _MS

    loop = ConsolidationLoop(
        model=model,
        tokenizer=MagicMock(),
        consolidation_config=consolidation_config,
        training_config=TrainingConfig(),
        tier_adapters={"episodic": AdapterConfig(), "semantic": AdapterConfig()},
        memory_store=_MS(),
        output_dir=tmp_path,
        **defaults,
    )
    # Admit-all probe stub: the real _probe_recall runs evaluate_indexed_recall,
    # which feeds the MagicMock model into re.sub and TypeErrors.  Admitting every key
    # is the prior implicit behavior (no recall gate), so it is inert for these tests.
    from paramem.training.recall_eval import RecallProbe

    loop._probe_recall = lambda adapter_name, entries: RecallProbe(
        per_key=tuple({"key": e["key"], "exact_match": True} for e in entries)
    )
    return loop


def _refiner_for(loop: ConsolidationLoop) -> GraphTierRefiner:
    """Build a :class:`GraphTierRefiner` off a loop's current live state.

    Mirrors exactly what ``ConsolidationLoop.build_tier_refiner``
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
    are stored on the node's display_name field where needed by individual
    tests.
    """
    for i in range(n_persons):
        name = f"person{i}"
        graph.add_node(
            name,
            entity_type="person",
            display_name=f"Person{i}",
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
        display_name="AcmeCorp",
        reinforcement_count=n_persons,
        sessions=["s000"],
        first_seen="s000",
        last_seen="s000",
    )
    # Wire edges: every person works_at acmecorp. speaker_id="speaker0" on
    # every edge — the merger stamps a real speaker_id on every Case-3
    # insert (merger.py), so an edge with none is not a state production
    # ever produces; the enrichment attribution pass reads exactly this
    # field as its evidence.
    for i in range(n_persons):
        graph.add_edge(
            f"person{i}",
            org,
            predicate="works at",
            relation_type="factual",
            confidence=1.0,
            source="extraction",
            sessions=["s000"],
            speaker_id="speaker0",
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
            display_name="Alice",
            reinforcement_count=3,
            sessions=["s010"],
            first_seen="s010",
            last_seen="s010",
        )
        graph.add_node(
            "alicia",
            entity_type="person",
            display_name="Alicia",
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
                display_name=display,
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
                display_name=display,
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
            display_name="Yang Ming",
            reinforcement_count=3,
            sessions=["s030"],
            first_seen="s030",
            last_seen="s030",
        )
        graph.add_node(
            "mr. yang",
            entity_type="person",
            display_name="Mr. Yang",
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
        # Nodes are canonical-keyed (lowercase); neither is a speaker node
        # (no ``speaker_id`` node attribute), but they ARE wired together by
        # a real, speaker-attributed edge below — attribution evidence for
        # the enrichment pass, matching what the merger always stamps on a
        # real edge.
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
        # reinforcement_count=50 outranks every _populate_graph hub node
        # (max 10), so zhang's ego-graph — {zhang, xiaoxiu}, its only
        # connection — is the sole chunk this graph's node count builds
        # (chunk_cap=1).
        for name in ("zhang", "xiaoxiu"):
            graph.add_node(
                name,
                entity_type="person",
                attributes={},
                reinforcement_count=50,
                sessions=["s040"],
                first_seen="s040",
                last_seen="s040",
            )
        graph.add_edge(
            "zhang",
            "xiaoxiu",
            predicate="acquainted with",
            relation_type="social",
            confidence=1.0,
            speaker_id="speaker0",
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
        # reinforcement_count=50 outranks every _populate_graph hub node
        # (max 10), so ming's ego-graph — {ming, xinxin}, its only
        # connection — is the sole chunk this graph's node count builds
        # (chunk_cap=1); the edge below is the enrichment pass's
        # attribution evidence, matching what the merger always stamps on
        # a real edge.
        for name in ("ming", "xinxin"):
            graph.add_node(
                name,
                entity_type="person",
                attributes={},
                reinforcement_count=50,
                sessions=["s041"],
                first_seen="s041",
                last_seen="s041",
            )
        graph.add_edge(
            "ming",
            "xinxin",
            predicate="acquainted with",
            relation_type="social",
            confidence=1.0,
            speaker_id="speaker0",
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
            display_name="Alexander",
            reinforcement_count=3,
            sessions=["s050"],
            first_seen="s050",
            last_seen="s050",
        )
        # reinforcement_count=50 outranks every _populate_graph hub node
        # (max 10), so alex's ego-graph — {alex, acme}, its only
        # connection — is the sole chunk this graph's node count builds
        # (chunk_cap=1). The alex-acme edge below carries speaker_id, the
        # enrichment pass's attribution evidence for the works_at relation
        # remapped onto "alexander" through the coref chain — matching what
        # the merger always stamps on a real edge.
        graph.add_node(
            "alex",
            entity_type="person",
            display_name="Alex",
            reinforcement_count=50,
            sessions=["s051"],
            first_seen="s051",
            last_seen="s051",
        )
        graph.add_node(
            "acme",
            entity_type="organization",
            display_name="Acme",
            reinforcement_count=5,
            sessions=["s050"],
            first_seen="s050",
            last_seen="s050",
        )
        graph.add_edge(
            "alex",
            "acme",
            predicate="interned at",
            relation_type="factual",
            confidence=1.0,
            speaker_id="speaker0",
            sessions=["s051"],
            first_seen="s051",
            last_seen="s051",
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
    and the chunk's ``triples``.

    ``request_graph_enrichment`` only ever reads ``payload.forward``/
    ``payload.facts`` — it has no notion of HOW a forward table was
    produced — so this helper constructs the table directly from
    *llm_mapping* rather than replaying the SCAN/MINT machinery
    (:func:`~paramem.cloud.placeholders.build_forward_table` now mints
    every placeholder value itself; a caller-dictated exact placeholder
    string, as several tests below rely on, is no longer an input that
    primitive accepts). This mirrors the pre-split helper's own scope: it
    never modeled the model call either, only assembled a payload from a
    given mapping.

    ``graph`` carries no relations of its own (interface narrowing,
    2026-07-21): ``request_graph_enrichment`` derives its anonymized
    triples directly from ``payload.facts`` via ``insert_placeholders``,
    not from ``graph.relations`` — ``graph`` is only the diagnostics
    sink. ``payload.facts`` is set to ``triples`` here, mirroring what
    :func:`~paramem.cloud.anonymize.anonymize` populates it with on a
    successful (non-fail-closed) call.
    """
    from paramem.cloud.anonymize import AnonymizedContract
    from paramem.cloud.placeholders import invert_forward_mapping
    from paramem.graph.schema import SessionGraph

    forward = dict(llm_mapping)
    reverse = invert_forward_mapping(forward)
    payload = AnonymizedContract(
        status="ok",
        forward=forward,
        reverse=reverse,
        anon_transcript="",
        declared=frozenset(reverse.keys()),
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
        # No cloud_bindings in the response -> the collision scan ran and
        # found nothing -> an empty list, written unconditionally now.
        assert graph.diagnostics["cloud_binding_collisions"] == []

    def test_accepted_chunk_reports_zero_dropped_relations(self):
        """The guard conditions are preserved end-to-end: an accepted
        delta reports a zero dropped-relation count and writes the
        collision diagnostic key as an empty list (the scan ran and found
        nothing), never omits it."""
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
        assert graph.diagnostics["cloud_binding_collisions"] == []


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
                pending=PendingRelations(episodic=[], procedural=[]),
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
        loop = _make_loop(tmp_path)

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
                pending=PendingRelations(episodic=[], procedural=[]),
                run_label="s1",
                schedule="12h",
                max_interim_count=1,
                stamp=current_stamp,
            )

        assert result["mode"] == "cap_pending"
        enrich_mock.assert_not_called()


class TestRefineOrderEnrichThenNormalize:
    """``GraphTierRefiner.refine`` runs enrichment BEFORE normalization.

    Unlike a test that mocks both passes away to test the caller's wiring,
    these tests exercise the refiner's REAL
    ``run_enrichment``/``run_normalization`` bodies — with
    only the underlying cloud primitives (``request_graph_enrichment``,
    ``normalize_predicates``) mocked — so the observed order and the
    content interaction between the two passes are real, not asserted on
    the caller's behalf.
    """

    # Enrichment-before-normalization call ORDER is pinned at the caller,
    # through ``ConsolidationLoop.stage_event``, the production entry point.
    # Not duplicated here at the refiner level.

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
    # merger.removal_ledger is reset ONLY by reset_graph(), never at the top
    # of merge(), so it survives an intervening cross-pass merge within one
    # fold.  A test asserting both reason codes coexist in one refine() call
    # adds no further coverage (removal_ledger was never reset by merge()
    # even before that rule was enforced -- the fix only touched
    # reinforcements).


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


class TestArbitrateSessionEnrichmentIncidents:
    """``ConsolidationLoop.arbitrate_enrichment_incidents`` — the session-tier
    reconciliation extracted from ``extract_session``, now driven by the
    per-session signal record a staging caller collects rather than the
    ``SessionGraph`` object itself (``extract_session`` no longer writes
    this incident directly).

    Uses the same ``record_incident``/``resolve_incident`` surface (and the
    same ``incidents_state_dir`` fixture pattern) as
    ``TestRefineConsolidationGraphRecordsVramIncident`` above.  Calls the
    method directly with a hand-built signal rather than driving
    ``extract_session`` end to end, so no GPU/model call is needed.
    """

    def _graph(self, **diagnostics) -> SessionGraph:
        return SessionGraph(
            session_id="s1", timestamp="2026-08-02T00:00:00+00:00", diagnostics=diagnostics
        )

    @staticmethod
    def _signal(graph: SessionGraph, session_id: str) -> dict:
        """Build the ``{session_id, anonymize, cloud_enrichment_degraded}``
        record a staging caller collects from ``session_graph.diagnostics``
        right after its own ``extract_session`` call."""
        return {
            "session_id": session_id,
            "anonymize": graph.diagnostics.get("anonymize"),
            "cloud_enrichment_degraded": graph.diagnostics.get("cloud_enrichment_degraded"),
        }

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

        loop.arbitrate_enrichment_incidents([self._signal(self._graph(anonymize="ok"), "s1")])

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
        loop.arbitrate_enrichment_incidents([self._signal(graph, "s1")])

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

        loop.arbitrate_enrichment_incidents(
            [self._signal(self._graph(anonymize="opted_out"), "s1")]
        )

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
        loop.arbitrate_enrichment_incidents([self._signal(graph, "s1")])

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
        loop.arbitrate_enrichment_incidents([self._signal(graph, "s1")])
        loop.arbitrate_enrichment_incidents([self._signal(graph, "s2")])

        incidents = read_incidents(loop._incidents_state_dir)
        assert len(incidents) == 1
        assert incidents[0].count == 2
        assert incidents[0].status == "active"

        resolve_incident(loop._incidents_state_dir, "enrichment_degraded", "anonymize")
        loop.arbitrate_enrichment_incidents([self._signal(graph, "s3")])

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

        loop.arbitrate_enrichment_incidents([self._signal(self._graph(), "s1")])

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

        loop.arbitrate_enrichment_incidents([self._signal(self._graph(), "s1")])

        for inc in read_incidents(loop._incidents_state_dir):
            assert inc.status == "resolved"
            assert inc.resolved_reason == "cloud egress disabled — enrichment cannot run"

    def test_absent_and_cloud_disabled_with_nothing_standing_writes_no_store(self, tmp_path):
        """The cloud-disabled sweep must not materialise an empty
        ``incidents.json`` when nothing was ever recorded (matches
        ``resolve_incidents_by_type``'s own success-path invariant)."""
        loop = _make_loop(tmp_path, incidents_state_dir=tmp_path / "incidents", cloud_enabled=False)

        loop.arbitrate_enrichment_incidents([self._signal(self._graph(), "s1")])

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

        loop.arbitrate_enrichment_incidents(
            [self._signal(self._graph(anonymize="not_a_real_outcome"), "s1")]
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
            loop.arbitrate_enrichment_incidents(
                [self._signal(self._graph(anonymize="failed"), "s1")]
            )

        record_mock.assert_not_called()


# ---------------------------------------------------------------------------
# Tests for _build_working_keyed_walk (unified edge→entry builder)
# ---------------------------------------------------------------------------


class TestHarvestKeylessEdges:
    """Unit tests for the unified edge→entry builder (_build_working_keyed_walk).

    Uses _make_loop from this module (real nx.MultiDiGraph + real MemoryStore,
    mocked model/tokenizer so no GPU).  store.put() writes into the
    KeyRegistry.

    These tests exercise the keyless-edge (minting) branch of the builder by
    populating the graph with only keyless predicate-bearing edges.
    """

    def test_donor_key_floor_on_empty_store(self, tmp_path):
        """An empty store must seed both counters at DONOR_KEY_FLOOR (201),
        not 1 -- the donor (paramem.training.donor) reserves graph1-200 and
        proc1-200 for its synthetic training population."""
        from paramem.training.donor import DONOR_KEY_FLOOR

        loop = _make_loop(tmp_path)

        assert loop._indexed_next_index == DONOR_KEY_FLOOR == 201
        assert loop._procedural_next_index == DONOR_KEY_FLOOR == 201

    def test_donor_key_floor_high_water_still_wins(self, tmp_path):
        """A store high-water key beyond the reserved band must still raise the
        counter past DONOR_KEY_FLOOR -- the floor is a lower bound, never a
        ceiling on the constructor's max() derivation."""
        from paramem.memory.store import MemoryStore as _MS
        from paramem.training.donor import DONOR_KEY_FLOOR
        from paramem.training.key_registry import KeyRegistry

        store = _MS()
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
            tier_adapters={"episodic": AdapterConfig(), "semantic": AdapterConfig()},
            memory_store=store,
            output_dir=tmp_path,
            extraction_enrichment_provider="anthropic",
            extraction_enrichment_provider_model="claude-sonnet-4-6",
            extraction_scrub_categories=resolve_scrub_categories(["person name"]),
            extraction_max_tokens=8192,
            extraction_plausibility_max_tokens=8192,
            extraction_anonymize_token_envelope=8192,
        )

        assert loop._procedural_next_index == 301, (
            f"Expected _procedural_next_index=301 (high-water beats "
            f"DONOR_KEY_FLOOR={DONOR_KEY_FLOOR}), got {loop._procedural_next_index}"
        )
        assert loop._indexed_next_index == DONOR_KEY_FLOOR


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
            display_name="Alice",
            reinforcement_count=3,
            sessions=["s010"],
            first_seen="s010",
            last_seen="s010",
        )
        graph.add_node(
            "alicia",
            entity_type="person",
            display_name="Alicia",
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
        # A same_as contraction merges NODES, not facts — there is no
        # surviving indexed key to inherit maturity, so record_removal's
        # survivor_key must be omitted.  Driven through the REAL production
        # writer (GraphTierRefiner.run_enrichment -> enrich_graph ->
        # merger.record_removal), not a hand-constructed ledger entry.
        assert "survivor_key" not in entry, (
            f"enrichment_same_as must never carry survivor_key; got {entry}"
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

        loop = _make_loop(tmp_path)
        _populate_graph(loop.merger.graph, n_persons=10)
        for tier in ("episodic", "semantic", "procedural"):
            loop.store.load_registry(tier, KeyRegistry())

        # Two concept nodes for the enrichment relation, wired by a real
        # (speaker-attributed) edge. reinforcement_count=50 outranks every
        # _populate_graph hub node (max 10), so alpha's ego-graph — {alpha,
        # beta}, since that edge is their only connection — is the sole
        # chunk this graph's node count builds (chunk_cap=1); the
        # enrichment pass then has real source-fact evidence to inherit
        # attribution from, matching what the merger always stamps on a
        # real edge (Case-3 insert) rather than an edge with no speaker_id.
        for name in ("alpha", "beta"):
            loop.merger.graph.add_node(
                name,
                entity_type="person",
                display_name=name.capitalize(),
                reinforcement_count=50,
                sessions=["s099"],
                first_seen="s099",
                last_seen="s099",
            )
        loop.merger.graph.add_edge(
            "alpha",
            "beta",
            predicate="acquainted with",
            relation_type="social",
            confidence=1.0,
            speaker_id="speaker0",
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

        loop = _make_loop(tmp_path)
        _populate_graph(loop.merger.graph, n_persons=10)
        for tier in ("episodic", "semantic", "procedural"):
            loop.store.load_registry(tier, KeyRegistry())

        # Pre-insert an extraction edge for alpha→beta colleague_of, carrying
        # speaker_id like every real edge does (Case-3 insert, merger.py).
        # reinforcement_count=50 outranks every _populate_graph hub node
        # (max 10), so alpha's ego-graph — {alpha, beta}, its only
        # connection — is the sole chunk this graph's node count builds
        # (chunk_cap=1), giving the enrichment pass real evidence to
        # attribute the duplicate relation from.
        for name in ("alpha", "beta"):
            loop.merger.graph.add_node(
                name,
                entity_type="person",
                display_name=name.capitalize(),
                reinforcement_count=50,
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
            speaker_id="speaker0",
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


class TestEnrichmentSpeakerAttributionFromSourceFacts:
    """Enrichment inherits speaker attribution from its source facts, never
    from graph topology: a relation's speaker is the union of its two
    (resolved) endpoints' evidence, collected from the chunk's own edges —
    exactly one attributes, zero or several drop and are counted.
    """

    def test_single_speaker_chunk_stamps_that_speaker(self, tmp_path, monkeypatch):
        """Both endpoints' evidence names exactly one speaker (speaker0,
        via _populate_graph's works-at edges) -> the relation is stamped
        with that speaker and reaches merge_relations."""
        from paramem.training.key_registry import KeyRegistry

        loop = _make_loop(tmp_path)
        _populate_graph(loop.merger.graph, n_persons=10)
        for tier in ("episodic", "semantic", "procedural"):
            loop.store.load_registry(tier, KeyRegistry())

        rels = [
            {
                "subject": "Person0",
                "predicate": "colleague_of",
                "object": "Person1",
                "relation_type": "social",
                "confidence": 0.9,
                "symmetric": False,
            }
        ]
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with (
            patch(
                "paramem.training.graph_enrich.request_graph_enrichment",
                return_value=(rels, [], "raw", 0),
            ),
            patch.object(
                loop.merger, "merge_relations", wraps=loop.merger.merge_relations
            ) as spy_merge_relations,
        ):
            result = _refiner_for(loop).run_enrichment()

        assert not result["skipped"]
        assert result["stamped_relations"] == 1
        assert result["unattributed_dropped"] == 0
        assert result["multi_speaker_dropped"] == 0
        assert spy_merge_relations.called, "the attributed relation must reach merge_relations"
        (captured_relations,), _kwargs = spy_merge_relations.call_args
        assert len(captured_relations) == 1
        assert captured_relations[0].speaker_id == "speaker0"

    def test_two_speaker_endpoints_drop_and_never_reach_merge(self, tmp_path, monkeypatch):
        """Endpoints whose evidence names two DIFFERENT speakers -> the
        relation is dropped and counted, and never reaches merge_relations
        (a fact synthesised across two speakers is not expressible in
        one-speaker-per-row bookkeeping — guessing one would mis-attribute).

        Mutation: pick either endpoint's speaker arbitrarily instead of
        checking the union's size -> this test's ``multi_speaker_dropped``
        assertion fails and ``merge_relations`` gets called.
        """
        from paramem.training.key_registry import KeyRegistry

        loop = _make_loop(tmp_path)
        graph = loop.merger.graph
        _populate_graph(graph, n_persons=10)
        for tier in ("episodic", "semantic", "procedural"):
            loop.store.load_registry(tier, KeyRegistry())

        # person0's own works-at edge is asserted by speaker1; every other
        # person (including person1) stays on speaker0 (_populate_graph's
        # default). Both route through the shared acmecorp hub inside the
        # one chunk this graph's node count builds (chunk_cap=1), so the
        # chunk's own source facts genuinely name two distinct speakers.
        for _, _, key, data in graph.out_edges("person0", keys=True, data=True):
            if data.get("predicate") == "works at":
                graph["person0"]["acmecorp"][key]["speaker_id"] = "speaker1"

        rels = [
            {
                "subject": "Person0",
                "predicate": "colleague_of",
                "object": "Person1",
                "relation_type": "social",
                "confidence": 0.9,
                "symmetric": False,
            }
        ]
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with (
            patch(
                "paramem.training.graph_enrich.request_graph_enrichment",
                return_value=(rels, [], "raw", 0),
            ),
            patch.object(
                loop.merger, "merge_relations", wraps=loop.merger.merge_relations
            ) as spy_merge_relations,
        ):
            result = _refiner_for(loop).run_enrichment()

        assert not result["skipped"]
        assert result["multi_speaker_dropped"] == 1
        assert result["stamped_relations"] == 0
        assert result["unattributed_dropped"] == 0
        spy_merge_relations.assert_not_called()

    def test_zero_speaker_evidence_drops_and_never_reaches_merge(self, tmp_path, monkeypatch):
        """Neither endpoint's source edges name any speaker at all (the
        union of both endpoints' evidence is empty) -> the relation is
        dropped and counted as unattributed, never guessed from graph
        topology and never merged.

        Mutation: fall back to a topological guess (e.g. picking the first
        edge's speaker regardless of emptiness) when the union is empty,
        instead of dropping -> this test's ``unattributed_dropped``
        assertion fails and ``merge_relations`` gets called.
        """
        from paramem.training.key_registry import KeyRegistry

        loop = _make_loop(tmp_path)
        graph = loop.merger.graph
        _populate_graph(graph, n_persons=10)
        for tier in ("episodic", "semantic", "procedural"):
            loop.store.load_registry(tier, KeyRegistry())

        # High reinforcement_count isolates {lonea, loneb} as the sole
        # chunk this graph's node count builds (mirrors the coref test's
        # chaina isolation below) -- their only edge carries NO speaker_id
        # at all, so both endpoints' provenance evidence is empty.
        graph.add_node(
            "lonea",
            entity_type="person",
            reinforcement_count=100,
            sessions=["s061"],
            first_seen="s061",
            last_seen="s061",
        )
        graph.add_node(
            "loneb",
            entity_type="person",
            reinforcement_count=1,
            sessions=["s061"],
            first_seen="s061",
            last_seen="s061",
        )
        graph.add_edge(
            "lonea",
            "loneb",
            predicate="linked to",
            relation_type="factual",
            confidence=1.0,
            speaker_id="",
            sessions=["s061"],
            first_seen="s061",
            last_seen="s061",
        )

        rels = [
            {
                "subject": "lonea",
                "predicate": "knows",
                "object": "loneb",
                "relation_type": "factual",
                "confidence": 0.9,
                "symmetric": False,
            }
        ]
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with (
            patch(
                "paramem.training.graph_enrich.request_graph_enrichment",
                return_value=(rels, [], "raw", 0),
            ),
            patch.object(
                loop.merger, "merge_relations", wraps=loop.merger.merge_relations
            ) as spy_merge_relations,
        ):
            result = _refiner_for(loop).run_enrichment()

        assert not result["skipped"]
        assert result["unattributed_dropped"] == 1
        assert result["stamped_relations"] == 0
        assert result["multi_speaker_dropped"] == 0
        spy_merge_relations.assert_not_called()

    def test_two_hop_coref_chain_inherits_provenance(self, tmp_path, monkeypatch):
        """A node reached only via a two-hop same_as chain (a→b, b→c)
        still contributes its pre-contraction speaker evidence to the
        surviving node c: the fold must follow the FULL coref chain via
        resolve_to_node_key, not a single drop→keep pass that would only
        see one hop.

        Mutation: replace the fold's ``resolve_to_node_key(...)`` call with
        a one-shot ``coref_map.get(k, k)`` lookup (no chain-follow) ->
        "chaina"'s evidence folds onto "chainb" instead of "chainc" (the
        relation's actual, fully-resolved subject endpoint after both
        contractions), the endpoint lookup misses, and this test's
        ``stamped_relations`` assertion fails (dropped as unattributed
        instead).
        """
        from paramem.training.key_registry import KeyRegistry

        loop = _make_loop(tmp_path)
        graph = loop.merger.graph
        _populate_graph(graph, n_persons=10)
        for tier in ("episodic", "semantic", "procedural"):
            loop.store.load_registry(tier, KeyRegistry())

        # reinforcement_count=100 outranks every _populate_graph hub node
        # (max 10) and every other node added below, so chaina's ego-graph —
        # {chaina, chainanchor}, its only connection — is the sole chunk
        # this graph's node count builds (chunk_cap=1). Names avoid
        # underscores/spaces so canonical() is a no-op on them and the
        # membership-shortcut / canonical-fallback resolution paths agree.
        graph.add_node(
            "chaina",
            entity_type="person",
            reinforcement_count=100,
            sessions=["s060"],
            first_seen="s060",
            last_seen="s060",
        )
        graph.add_node(
            "chainb",
            entity_type="person",
            reinforcement_count=1,
            sessions=["s060"],
            first_seen="s060",
            last_seen="s060",
        )
        graph.add_node(
            "chainc",
            entity_type="person",
            reinforcement_count=1,
            sessions=["s060"],
            first_seen="s060",
            last_seen="s060",
        )
        graph.add_node(
            "chainanchor",
            entity_type="concept",
            reinforcement_count=1,
            sessions=["s060"],
            first_seen="s060",
            last_seen="s060",
        )
        graph.add_edge(
            "chaina",
            "chainanchor",
            predicate="linked to",
            relation_type="factual",
            confidence=1.0,
            speaker_id="speaker9",
            sessions=["s060"],
            first_seen="s060",
            last_seen="s060",
        )

        canned_rels = [
            {
                "subject": "chaina",
                "predicate": "knows",
                "object": "chainanchor",
                "relation_type": "factual",
                "confidence": 0.9,
            }
        ]
        # Two-hop same_as chain: chaina drops into chainb, then chainb
        # itself drops into chainc — the relation's subject ("chaina") must
        # resolve through BOTH hops to land on the surviving "chainc".
        canned_same_as = [["chainb", "chaina"], ["chainc", "chainb"]]

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with (
            patch(
                "paramem.training.graph_enrich.request_graph_enrichment",
                return_value=(canned_rels, canned_same_as, "raw", 0),
            ),
            patch(
                "paramem.training.graph_enrich._safe_to_merge_surface",
                return_value=True,
            ),
            patch.object(
                loop.merger, "merge_relations", wraps=loop.merger.merge_relations
            ) as spy_merge_relations,
        ):
            result = _refiner_for(loop).run_enrichment()

        assert not result["skipped"]
        assert result["same_as_merges"] == 2
        assert "chaina" not in graph.nodes
        assert "chainb" not in graph.nodes
        assert "chainc" in graph.nodes
        assert result["stamped_relations"] == 1, (
            f"chaina's speaker9 evidence must survive the 2-hop fold onto "
            f"chainc; got stamped_relations={result['stamped_relations']} "
            f"unattributed_dropped={result['unattributed_dropped']}"
        )
        assert spy_merge_relations.called
        (captured_relations,), _kwargs = spy_merge_relations.call_args
        assert len(captured_relations) == 1
        assert captured_relations[0].speaker_id == "speaker9"


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

    def test_non_speaker_endpoint_uses_the_node_display_name(self, tmp_path, monkeypatch):
        """``_endpoint_str`` passes the node's ``display_name`` for a
        non-speaker endpoint that already has one — not the bare canonical
        node key.  Captures the ``Relation`` objects reaching
        ``GraphMerger.merge_relations`` (the direct consumer of
        ``_endpoint_str``'s return value) so a regression to the canonical
        key is caught even though the merger's own first-seen-wins guard
        would otherwise mask it downstream."""
        from paramem.training.key_registry import KeyRegistry

        loop = _make_loop(tmp_path)
        _populate_graph(loop.merger.graph, n_persons=10)
        for tier in ("episodic", "semantic", "procedural"):
            loop.store.load_registry(tier, KeyRegistry())

        # _populate_graph seeds "person0" (display "Person0") and "acmecorp"
        # (display "AcmeCorp") among the 10 persons + 1 org, already linked
        # by "works at" — neither node carries a speaker_id, so both
        # endpoints exercise the display branch.
        assert loop.merger.graph.nodes["person0"]["display_name"] == "Person0"
        assert loop.merger.graph.nodes["acmecorp"]["display_name"] == "AcmeCorp"

        rels = [
            {
                "subject": "person0",
                "predicate": "mentioned_alongside",
                "object": "acmecorp",
                "relation_type": "factual",
                "confidence": 0.9,
                "symmetric": False,
            }
        ]
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with (
            patch(
                "paramem.training.graph_enrich.request_graph_enrichment",
                return_value=(rels, [], "raw", 0),
            ),
            patch.object(
                loop.merger, "merge_relations", wraps=loop.merger.merge_relations
            ) as spy_merge_relations,
        ):
            result = _refiner_for(loop).run_enrichment()

        assert not result["skipped"]
        assert spy_merge_relations.called, "the enrichment relation must reach merge_relations"
        (captured_relations,), _kwargs = spy_merge_relations.call_args
        assert len(captured_relations) == 1
        captured = captured_relations[0]
        assert captured.subject == "Person0", (
            f"non-speaker subject endpoint must carry the node's display_name "
            f"'Person0', not the canonical key; got {captured.subject!r}"
        )
        assert captured.object == "AcmeCorp", (
            f"non-speaker object endpoint must carry the node's display_name "
            f"'AcmeCorp', not the canonical key; got {captured.object!r}"
        )

    def test_same_as_contracts_unbound_into_verbatim_speaker_node(self, tmp_path, monkeypatch):
        """same_as ['speaker0', 'alex'] contracts the unbound 'alex' concept node
        INTO the casefolded speaker node 'speaker0'.

        Speaker-identity invariant: keep="speaker0" resolves via canonical
        fallback to the lowercase ``speaker{N}`` node key "speaker0".
        drop="alex" resolves via membership shortcut (already in graph).  Contraction
        succeeds: "alex" is absorbed into "speaker0"."""
        from paramem.training.key_registry import KeyRegistry

        loop = _make_loop(tmp_path)
        _populate_graph(loop.merger.graph, n_persons=10)
        for tier in ("episodic", "semantic", "procedural"):
            loop.store.load_registry(tier, KeyRegistry())

        # Casefolded speaker node ("speaker0") + an unbound concept node "alex".
        _seed_speaker_node(loop, "speaker0", "Alex")
        loop.merger.graph.add_node(
            "alex",
            entity_type="person",
            display_name="Alex",
            reinforcement_count=1,
            sessions=["s200"],
            first_seen="s200",
            last_seen="s200",
        )
        # Give "alex" an edge so the contraction has something to move.
        loop.merger.graph.add_node("rust", display_name="Rust")
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
        ``mapping`` key in production (SCAN verification drops any
        speaker-id-shaped value before ``build_forward_table`` ever mints
        for it — see that function's speaker-anchor invariants), so it
        legitimately reaches the
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

        # Hand-built payload (bypassing build_forward_table's speaker-key-drop
        # guard on purpose — see docstring: a legal input to this pure
        # function, never how a real caller populates it).
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
                display_name=sid,
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
            display_name=f"Person{i}",
            reinforcement_count=i + 1,
            sessions=[f"s{i:03d}"],
            first_seen=f"s{i:03d}",
            last_seen=f"s{i:03d}",
        )
    org = "acmecorp"
    graph.add_node(
        org,
        entity_type="concept",
        display_name="AcmeCorp",
        reinforcement_count=n_persons,
        sessions=["s000"],
        first_seen="s000",
        last_seen="s000",
    )
    # speaker_id="speaker0" on every edge — see _populate_graph's identical
    # comment; the production merger never leaves a real edge unattributed.
    for i in range(n_persons):
        graph.add_edge(
            f"person{i}",
            org,
            predicate="works at",
            relation_type="factual",
            confidence=1.0,
            source="extraction",
            sessions=["s000"],
            speaker_id="speaker0",
        )


def _reconciled_contract_for_stub(
    raw_forward: dict[str, str],
    facts: list[dict],
    identity_domain,
    *,
    raw: str = "stub-raw",
):
    """Shared tail for every ``anonymize`` stand-in below: given the
    stub's own real-name -> placeholder table (standing in for what
    ``paramem.cloud.placeholders.build_forward_table`` would have
    minted from a SCAN result — placeholder VALUES are code-minted in
    production and no longer caller-dictated, but this test-only
    shortcut keeps the caller-given value rather than re-minting, since
    nothing here exercises the minting rule itself), replay the REAL
    identity-reconciliation and domain-guard primitives
    (:func:`~paramem.cloud.anonymize._index_identity_domain`,
    :func:`~paramem.cloud.anonymize._reconcile_to_domain`,
    :func:`~paramem.cloud.anonymize._domain_guard_fires`) against the
    caller's own ``identity_domain``/``facts`` — so behavioral tests of
    THAT logic (reconciliation drops, the domain-scoped fail-closed
    guard) stay faithful to production without re-deriving it a second
    time inside a bespoke stub.

    Returns a ``status="failed", failure="guard"`` contract when the
    guard fires; otherwise a ``status="ok"`` contract carrying the
    reconciled forward table and ``facts`` verbatim (mirroring what
    ``anonymize()`` itself returns on a successful, transcript-less
    call).
    """
    from paramem.cloud.anonymize import (
        AnonymizedContract,
        _domain_guard_fires,
        _index_identity_domain,
        _reconcile_to_domain,
    )
    from paramem.cloud.placeholders import invert_forward_mapping
    from paramem.utils.identity import is_speaker_id

    if identity_domain is not None:
        canon_to_domain, ambiguous = _index_identity_domain(identity_domain)
        forward, dropped = _reconcile_to_domain(raw_forward, canon_to_domain, ambiguous)
    else:
        forward, dropped = dict(raw_forward), 0

    if identity_domain is not None and _domain_guard_fires(
        dict.fromkeys(raw_forward), forward, facts
    ):
        return AnonymizedContract(
            status="failed",
            forward={},
            reverse={},
            anon_transcript="",
            declared=frozenset(),
            rekey_dropped=dropped,
            raw=raw,
            failure="guard",
            facts=[],
            model_calls=1,
        )

    reverse = invert_forward_mapping({k: v for k, v in forward.items() if not is_speaker_id(v)})
    return AnonymizedContract(
        status="ok",
        forward=forward,
        reverse=reverse,
        anon_transcript="",
        declared=frozenset(reverse.keys()),
        rekey_dropped=dropped,
        raw=raw,
        failure=None,
        facts=list(facts),
        model_calls=1,
    )


def _stub_local_model_types(type_by_name: dict[str, str]):
    """Build a stand-in for ``anonymize`` (THE one anonymize chain)
    typing real names per ``type_by_name`` (default ``"person"`` for
    anything unlisted), minting ``Prefix_N`` tokens in sorted-name order
    — simulating what the LOCAL model's own SCAN classification would
    produce for each real name — then replaying the real reconciliation
    + guard logic via :func:`_reconciled_contract_for_stub`.
    """
    from paramem.config.taxonomy import entity_type_to_prefix

    def _stub(facts, model, tokenizer, *, transcript="", identity_domain=None, **kwargs):
        # ``facts`` is a plain fact-dict list (interface narrowing,
        # 2026-07-21) — never a ``SessionGraph`` — so names come off
        # ``subject``/``object`` keys directly, not ``.relations``.
        names = sorted(
            {str(f.get("subject", "")) for f in facts} | {str(f.get("object", "")) for f in facts}
        )
        raw_forward: dict[str, str] = {}
        counters: dict[str, int] = {}
        for name in names:
            prefix = entity_type_to_prefix(type_by_name.get(name, "person"))
            counters[prefix] = counters.get(prefix, 0) + 1
            raw_forward[name] = f"{prefix}_{counters[prefix]}"
        return _reconciled_contract_for_stub(raw_forward, facts, identity_domain)

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
            patch("paramem.training.graph_enrich.anonymize", side_effect=stub),
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

        def _stub_omits_acmecorp(
            facts, model, tokenizer, *, transcript="", identity_domain=None, **kwargs
        ):
            # ``facts`` is a plain fact-dict list — never a ``SessionGraph``.
            names = sorted(
                {str(f.get("subject", "")) for f in facts}
                | {str(f.get("object", "")) for f in facts}
            )
            raw_forward = {
                name: f"Person_{i + 1}"
                for i, name in enumerate(n for n in names if n != "acmecorp")
            }
            return _reconciled_contract_for_stub(raw_forward, facts, identity_domain)

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with (
            patch(
                "paramem.training.graph_enrich.anonymize",
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
            patch("paramem.training.graph_enrich.anonymize", side_effect=stub),
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
            display_name=subject_key,
            reinforcement_count=10,
            sessions=["s000"],
            first_seen="s000",
            last_seen="s000",
        )
        graph.add_node(
            object_key,
            entity_type="concept",
            display_name=object_key,
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
            "paramem.training.graph_enrich.anonymize",
            lambda facts, model, tokenizer, *, identity_domain=None, **kwargs: (
                _reconciled_contract_for_stub({"Yang Ming": "Person_1"}, facts, identity_domain)
            ),
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
                display_name=key,
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
            "paramem.training.graph_enrich.anonymize",
            lambda facts, model, tokenizer, *, identity_domain=None, **kwargs: (
                _reconciled_contract_for_stub({"Yang Ming": "Person_1"}, facts, identity_domain)
            ),
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
            "build_forward_table",
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
            "build_forward_table",
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
            patch("paramem.training.graph_enrich.anonymize", side_effect=stub),
            patch("paramem.graph.extractor._load_prompt", side_effect=_bad_prompt),
            patch(
                "paramem.graph.extractor._cloud_call",
                side_effect=AssertionError("the cloud must never be called with a broken prompt"),
            ),
            pytest.raises(KeyError, match="oops"),
        ):
            _refiner_for(loop).run_enrichment()


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
            "an adopted key moved to the survivor edge — it is not a removal and must not be staled"
        )
