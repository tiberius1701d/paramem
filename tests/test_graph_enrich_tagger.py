"""``enrich_graph``'s tagger-failure arm — ``privacy_skipped_chunks`` in
both arms, ``aborted_reason="tagger"``, the chunk-loop break, and
already-merged chunks kept — plus the incident-recording surface
(``_record_enrichment_incident``) under ``graph_enrich_tagger`` /
``graph_enrich_vram`` and the by-type resolve on a clean pass.

Driving pattern (graph construction, ``request_graph_enrichment``
mocking) copied from ``tests/test_graph_enrichment.py``'s
``_populate_graph``/``TestEnrichmentAddsEdgesWithSourceTag`` (read-only —
never imported from).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from paramem.cloud.anonymize import AnonymizedContract, failed_contract
from paramem.config.taxonomy import ScrubCategory
from paramem.graph.extraction_pipeline import ExtractionConfig
from paramem.graph.merger import GraphMerger
from paramem.server.incidents import read_incidents
from paramem.training import graph_enrich as graph_enrich_module
from paramem.training.consolidation import ConsolidationLoop
from paramem.training.graph_enrich import enrich_graph

PERSON = ScrubCategory(
    name="Person", prefix="Person", hints=("person name",), tagger_labels=("person",)
)


def _build_pair_graph(merger: GraphMerger):
    """Five disjoint two-node pairs, each joined by one edge — with
    ``neighborhood_hops=1``/``max_entities_per_pass=2`` this yields five
    distinct, non-trimmed chunks, each carrying exactly one triple.
    """
    g = merger.graph
    for i in range(0, 10, 2):
        a, b = f"person{i}", f"person{i + 1}"
        g.add_node(a, entity_type="person", display_name=f"Person{i}", reinforcement_count=10 - i)
        g.add_node(
            b, entity_type="person", display_name=f"Person{i + 1}", reinforcement_count=9 - i
        )
        g.add_edge(
            a,
            b,
            predicate="knows",
            relation_type="social",
            confidence=0.9,
            speaker_id="speaker0",
            last_seen="",
            first_seen="",
        )
    return g


def _extraction_config():
    return ExtractionConfig(
        cloud_enabled=True,
        enrichment_provider="anthropic",
        enrichment_provider_model="claude-test",
        enrichment_provider_endpoint=None,
        scrub_categories=(PERSON,),
        anonymize_token_envelope=8192,
    )


def _ok_contract() -> AnonymizedContract:
    return AnonymizedContract(
        status="ok",
        forward={},
        reverse={},
        anon_transcript="",
        declared=frozenset(),
        rekey_dropped=0,
        raw="{}",
        facts=[],
    )


def _canned_cloud_result():
    return (
        [
            {
                "subject": "person0",
                "predicate": "colleague of",
                "object": "person1",
                "relation_type": "social",
                "confidence": 0.9,
            }
        ],
        [],  # no same_as pairs
        "raw",
        0,  # no relations dropped
    )


class TestPrivacySkippedChunksBothArms:
    def test_tagger_failure_counts_both_arms_sets_aborted_reason_breaks_and_keeps_merged(
        self, tmp_path, monkeypatch
    ) -> None:
        merger = GraphMerger()
        _build_pair_graph(merger)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        sequence = [
            _ok_contract(),  # chunk 0: succeeds
            failed_contract(failure="guard"),  # chunk 1: guard fail-closed, continue
            failed_contract(failure="tagger", raw="span tagger unavailable"),  # chunk 2: break
        ]
        calls: list[object] = []

        def _fake_anonymize(facts, model, tokenizer, **kwargs):
            calls.append(kwargs.get("identity_domain"))
            return sequence[len(calls) - 1]

        monkeypatch.setattr(graph_enrich_module, "anonymize", _fake_anonymize)
        monkeypatch.setattr(
            graph_enrich_module,
            "request_graph_enrichment",
            lambda *a, **k: _canned_cloud_result(),
        )

        result = enrich_graph(
            merger,
            model=MagicMock(),
            tokenizer=MagicMock(),
            extraction_config_provider=_extraction_config,
            neighborhood_hops=1,
            max_entities_per_pass=2,
        )

        assert len(calls) == 3, "the loop must stop at the tagger-failed chunk (break)"
        assert result["aborted_reason"] == "tagger"
        assert result["privacy_skipped_chunks"] == 2  # guard chunk + tagger chunk
        assert result["chunks"] == 1  # only the successful chunk reached the cloud call
        # Already-merged chunk 0's relation survives — the pass proceeds to
        # merge what it accumulated before the break, never discarding it.
        assert result["new_edges"] >= 1
        assert merger.graph.has_edge("person0", "person1")
        found_colleague_edge = any(
            data.get("predicate") == "colleague of"
            for _u, _v, data in merger.graph.out_edges("person0", data=True)
        )
        assert found_colleague_edge

    def test_a_guard_only_run_never_sets_aborted_reason_and_never_breaks(
        self, tmp_path, monkeypatch
    ) -> None:
        merger = GraphMerger()
        _build_pair_graph(merger)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        calls: list[object] = []

        def _fake_anonymize(facts, model, tokenizer, **kwargs):
            calls.append(1)
            return failed_contract(failure="guard")

        monkeypatch.setattr(graph_enrich_module, "anonymize", _fake_anonymize)
        monkeypatch.setattr(
            graph_enrich_module,
            "request_graph_enrichment",
            lambda *a, **k: _canned_cloud_result(),
        )

        result = enrich_graph(
            merger,
            model=MagicMock(),
            tokenizer=MagicMock(),
            extraction_config_provider=_extraction_config,
            neighborhood_hops=1,
            max_entities_per_pass=2,
        )

        assert result["aborted_reason"] is None
        assert len(calls) == 5  # every chunk attempted — guard never breaks the loop
        assert result["privacy_skipped_chunks"] == 5
        assert result["chunks"] == 0  # no chunk ever reached the cloud call


class TestRecordEnrichmentIncident:
    def _stub_loop(self, tmp_path):
        return SimpleNamespace(
            _incidents_state_dir=tmp_path,
            _ENRICHMENT_ABORT_SUMMARIES=ConsolidationLoop._ENRICHMENT_ABORT_SUMMARIES,
        )

    def test_tagger_reason_records_under_graph_enrich_tagger(self, tmp_path) -> None:
        result = SimpleNamespace(enrichment={"aborted_reason": "tagger", "chunks": 1})
        ConsolidationLoop._record_enrichment_incident(self._stub_loop(tmp_path), result)

        incidents = read_incidents(tmp_path)
        ids = {i.id for i in incidents}
        assert "enrichment_degraded:graph_enrich_tagger" in ids

    def test_vram_reason_records_under_graph_enrich_vram(self, tmp_path) -> None:
        result = SimpleNamespace(enrichment={"aborted_reason": "vram", "chunks": 2})
        ConsolidationLoop._record_enrichment_incident(self._stub_loop(tmp_path), result)

        incidents = read_incidents(tmp_path)
        ids = {i.id for i in incidents}
        assert "enrichment_degraded:graph_enrich_vram" in ids

    def test_clean_pass_resolves_the_whole_family_by_type(self, tmp_path) -> None:
        loop = self._stub_loop(tmp_path)
        # First: an active tagger-degrade incident.
        ConsolidationLoop._record_enrichment_incident(
            loop, SimpleNamespace(enrichment={"aborted_reason": "tagger", "chunks": 1})
        )
        assert any(
            i.id == "enrichment_degraded:graph_enrich_tagger" and i.status == "active"
            for i in read_incidents(tmp_path)
        )

        # Then: a clean pass (aborted_reason is None) resolves it.
        ConsolidationLoop._record_enrichment_incident(
            loop, SimpleNamespace(enrichment={"aborted_reason": None, "chunks": 3})
        )

        incidents = read_incidents(tmp_path)
        tagger_incident = next(
            i for i in incidents if i.id == "enrichment_degraded:graph_enrich_tagger"
        )
        assert tagger_incident.status == "resolved"

    def test_enrichment_none_is_a_no_op(self, tmp_path) -> None:
        loop = self._stub_loop(tmp_path)
        ConsolidationLoop._record_enrichment_incident(loop, SimpleNamespace(enrichment=None))
        assert read_incidents(tmp_path) == []
