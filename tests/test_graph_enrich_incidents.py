"""Graph-tier enrichment's degrade-to-incident surface: the two abort
causes (``vram``/``scan_failed``) ``ConsolidationLoop._record_enrichment_incident``
turns into an operator-visible incident, and the ``enrich_graph`` chunk
loop's own reaction to each cause — ``scan_failed`` breaks the loop before
any cloud call; the domain-scoped ``guard`` never sets an aborted reason
and lets the loop continue to the next chunk.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from paramem.cloud.anonymize import failed_contract
from paramem.config.taxonomy import resolve_scrub_categories
from paramem.graph.extraction_pipeline import ExtractionConfig
from paramem.graph.merger import GraphMerger
from paramem.server.incidents import read_incidents
from paramem.training.consolidation import ConsolidationLoop
from paramem.training.graph_enrich import enrich_graph

# ---------------------------------------------------------------------------
# ConsolidationLoop._record_enrichment_incident — the incident derivation.
# ---------------------------------------------------------------------------


class TestRecordEnrichmentIncident:
    def _stub_loop(self, tmp_path):
        return SimpleNamespace(
            _incidents_state_dir=tmp_path,
            _ENRICHMENT_ABORT_SUMMARIES=ConsolidationLoop._ENRICHMENT_ABORT_SUMMARIES,
        )

    def test_vram_reason_records_under_graph_enrich_vram(self, tmp_path) -> None:
        result = SimpleNamespace(enrichment={"aborted_reason": "vram", "chunks": 2})
        ConsolidationLoop._record_enrichment_incident(self._stub_loop(tmp_path), result)

        incidents = read_incidents(tmp_path)
        ids = {i.id for i in incidents}
        assert "enrichment_degraded:graph_enrich_vram" in ids

    def test_scan_failed_reason_records_under_graph_enrich_scan_failed(self, tmp_path) -> None:
        result = SimpleNamespace(enrichment={"aborted_reason": "scan_failed", "chunks": 1})
        ConsolidationLoop._record_enrichment_incident(self._stub_loop(tmp_path), result)

        incidents = read_incidents(tmp_path)
        ids = {i.id for i in incidents}
        assert "enrichment_degraded:graph_enrich_scan_failed" in ids

    def test_enrichment_none_is_a_no_op(self, tmp_path) -> None:
        loop = self._stub_loop(tmp_path)
        ConsolidationLoop._record_enrichment_incident(loop, SimpleNamespace(enrichment=None))
        assert read_incidents(tmp_path) == []


# ---------------------------------------------------------------------------
# enrich_graph's own chunk loop: scan_failed breaks it; guard does not.
# ---------------------------------------------------------------------------


def _populated_merger(n_persons: int = 25) -> GraphMerger:
    """A merger graph large enough (> the 10-node floor) and shaped
    (``graph_enrichment_max_entities_per_pass=10``, ``neighborhood_hops=1``)
    to produce more than one enrichment chunk — mirrors
    ``tests/test_graph_enrichment.py::TestChunkCapRespected``'s fixture.
    """
    merger = GraphMerger()
    graph = merger.graph
    org = "hubcorp"
    graph.add_node(
        org,
        entity_type="organization",
        display_name="HubCorp",
        reinforcement_count=n_persons,
        sessions=["s000"],
        first_seen="s000",
        last_seen="s000",
    )
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
        graph.add_edge(
            name,
            org,
            predicate="works at",
            relation_type="factual",
            confidence=1.0,
            source="extraction",
            sessions=["s000"],
            speaker_id="speaker0",
        )
    return merger


def _extraction_config() -> ExtractionConfig:
    return ExtractionConfig(
        enrichment_provider="anthropic",
        enrichment_provider_model="claude-test",
        cloud_enabled=True,
        scrub_categories=resolve_scrub_categories(["person name"]),
        anonymize_token_envelope=8192,
    )


def _run_enrichment(merger: GraphMerger) -> dict:
    return enrich_graph(
        merger,
        model=MagicMock(),
        tokenizer=MagicMock(),
        extraction_config_provider=_extraction_config,
        neighborhood_hops=1,
        max_entities_per_pass=10,
    )


class TestScanFailedBreaksTheChunkLoop:
    def test_scan_failed_sets_the_aborted_reason_and_no_cloud_call_is_made(
        self, monkeypatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setattr(
            "paramem.training.graph_enrich.anonymize",
            lambda *a, **k: failed_contract(failure="scan_failed", raw="unparseable reply"),
        )
        request_mock = MagicMock()
        monkeypatch.setattr("paramem.training.graph_enrich.request_graph_enrichment", request_mock)

        result = _run_enrichment(_populated_merger())

        assert result["aborted_reason"] == "scan_failed"
        request_mock.assert_not_called()


class TestGuardNeverSetsAnAbortedReason:
    def test_guard_failure_skips_the_chunk_but_never_aborts_the_pass(self, monkeypatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setattr(
            "paramem.training.graph_enrich.anonymize",
            lambda *a, **k: failed_contract(failure="guard", raw="named but unreconciled"),
        )
        request_mock = MagicMock()
        monkeypatch.setattr("paramem.training.graph_enrich.request_graph_enrichment", request_mock)

        result = _run_enrichment(_populated_merger())

        assert result["aborted_reason"] is None
        assert result["skipped"] is False
        assert result["privacy_skipped_chunks"] >= 1
        request_mock.assert_not_called()
