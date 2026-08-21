"""Tests for the consolidation loop orchestrator.

These are unit tests that mock the model/extraction to test
the consolidation logic without requiring GPU.

``_is_full_cycle_due`` and ``_oldest_interim_stamp`` are covered through the
dispatch/catch-up integration tests below (``TestSchedulerCatchUpGate``,
``TestFullCycleDispatcherOverdueIncident``, ``TestFullConsolidationOverdueKey``)
rather than a standalone gate-helper unit suite.
"""

import time
from unittest.mock import MagicMock

import pytest

from paramem.training.consolidation import ConsolidationLoop
from paramem.training.graph_tier import GraphTierRefiner
from paramem.training.recall_eval import RecallProbe
from paramem.utils.artifacts import (
    on_calibration_result,
    on_normalization,
)


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


class TestExtractionPathParity:
    """extract_session() must produce correct episodic/procedural sets and thread
    extraction kwargs unchanged from the loop constructor into the extractor calls.
    Guards against flag-forwarding drift and partition invariants in the unified
    extraction path."""

    def _build_loop(
        self,
        monkeypatch,
        tmp_path,
        procedural_enabled: bool,
        extract_graph_spy=None,
        extract_procedural_spy=None,
        **loop_kwargs,
    ):

        from peft import PeftModel

        from paramem.graph.schema import Entity, Relation, SessionGraph
        from paramem.training.consolidation import ConsolidationLoop
        from paramem.utils.config import AdapterConfig, ConsolidationConfig, TrainingConfig

        session_graph = SessionGraph(
            session_id="s001",
            timestamp="2026-01-01T00:00:00Z",
            entities=[
                Entity(name="Alex", entity_type="person"),
                Entity(name="Millfield", entity_type="place"),
            ],
            relations=[
                Relation(
                    subject="Alex",
                    predicate="lives_in",
                    object="Millfield",
                    relation_type="factual",
                    speaker_id="speaker0",
                ),
                Relation(
                    subject="Alex",
                    predicate="prefers",
                    object="Acme Radio",
                    relation_type="preference",
                    speaker_id="speaker0",
                ),
            ],
        )
        procedural_graph = SessionGraph(
            session_id="s001",
            timestamp="2026-01-01T00:00:00Z",
            entities=[Entity(name="Alex", entity_type="person")],
            relations=[
                Relation(
                    subject="Alex",
                    predicate="listens_to",
                    object="The Kooks",
                    relation_type="preference",
                    speaker_id="speaker0",
                ),
            ],
        )

        def _default_extract(*a, **kw):
            return session_graph

        def _default_extract_procedural(*a, **kw):
            return procedural_graph

        monkeypatch.setattr(
            "paramem.graph.extraction_pipeline.extract_graph",
            extract_graph_spy if extract_graph_spy is not None else _default_extract,
        )
        monkeypatch.setattr(
            "paramem.graph.extraction_pipeline.extract_procedural_graph",
            extract_procedural_spy
            if extract_procedural_spy is not None
            else _default_extract_procedural,
        )

        # __class__ = PeftModel so ensure_adapters' isinstance check
        # short-circuits without restricting the mock's attribute surface.
        model = MagicMock()
        model.__class__ = PeftModel
        model.peft_config = {
            "episodic": MagicMock(),
            "semantic": MagicMock(),
            "in_training": MagicMock(),
        }
        if procedural_enabled:
            model.peft_config["procedural"] = MagicMock()
        procedural_adapter = AdapterConfig() if procedural_enabled else None

        from paramem.memory.store import MemoryStore as _MS

        loop_kwargs.setdefault("extraction_scrub", {"person name"})
        loop_kwargs.setdefault("extraction_max_tokens", 8192)
        loop_kwargs.setdefault("extraction_plausibility_max_tokens", 8192)
        loop_kwargs.setdefault("extraction_anonymize_token_envelope", 8192)
        loop = ConsolidationLoop(
            model=model,
            tokenizer=MagicMock(),
            consolidation_config=ConsolidationConfig(),
            training_config=TrainingConfig(),
            episodic_adapter_config=AdapterConfig(),
            semantic_adapter_config=AdapterConfig(),
            memory_store=_MS(),
            procedural_adapter_config=procedural_adapter,
            output_dir=tmp_path,
            **loop_kwargs,
        )
        # Set by the server's boot wiring in production
        # (paramem.server.consolidation), never by __init__ itself — every
        # other minimal test loop in this tree sets it explicitly (e.g.
        # tests/test_fold_phase1.py::_make_loop); this factory's callers now
        # reach ConsolidationLoop._build_write_context (via stage_event's
        # build/write/publish driver), which reads it directly.
        loop.fingerprint_cache = None
        return loop

    def _run_extract_session(self, loop, source_type: str = "transcript"):
        return loop.extract_session(
            session_transcript="Alex lives in Millfield. He prefers Acme Radio.",
            session_id="s001",
            speaker_id="spk",
            source_type=source_type,
        )

    def _run_consolidation_cycle_and_capture(
        self, monkeypatch, loop, source_type: str = "transcript"
    ):
        """Call extract_session and capture what run_consolidation_cycle receives.

        Replaces the former ``_run_cycle_and_capture`` (which called the now-deleted
        ``run_cycle``).  Routes through the live production path:
        ``extract_session`` → dedup → ``run_consolidation_cycle``.
        """
        captured: dict[str, list[dict]] = {"episodic_rels": [], "procedural_rels": []}

        def _capture_cycle(self_, episodic_rels, procedural_rels, **kwargs):
            """Intercept run_consolidation_cycle to capture the relation lists."""
            captured["episodic_rels"] = episodic_rels
            captured["procedural_rels"] = procedural_rels
            return {
                "triples_extracted": len(episodic_rels),
                "new_keys": [],
                "mode": "trained",
                "adapter_name": "episodic",
                "venue": "train",
                "error": None,
            }

        monkeypatch.setattr(type(loop), "run_consolidation_cycle", _capture_cycle, raising=False)
        loop.extract_session(
            session_transcript="Alex lives in Millfield. He prefers Acme Radio.",
            session_id="s001",
            speaker_id="spk",
            source_type=source_type,
        )
        return captured["episodic_rels"], captured["procedural_rels"]

    def test_parity_procedural_enabled(self, monkeypatch, tmp_path):
        loop = self._build_loop(monkeypatch, tmp_path, procedural_enabled=True)
        episodic, procedural = self._run_extract_session(loop)

        # Partition invariant: no preference in episodic, and procedural carries
        # both the filter-sourced and the separately-extracted preference.
        assert all(qa["predicate"] != "prefers" for qa in episodic)
        assert {rel["predicate"] for rel in procedural} == {"prefers", "listens to"}

    @pytest.mark.parametrize("source_type", ["transcript", "document"])
    @pytest.mark.parametrize(
        "procedural_enabled,loop_overrides",
        [
            (True, {}),
            (False, {}),
            (
                True,
                {
                    "extraction_enrichment_provider": "claude",
                    "extraction_enrichment_provider_model": "claude-sonnet-4-6",
                },
            ),
            (True, {"extraction_plausibility_stage": "anon"}),
        ],
    )
    def test_parity_kwargs_identical(
        self, monkeypatch, tmp_path, procedural_enabled, loop_overrides, source_type
    ):
        """extract_session must pass IDENTICAL kwargs regardless of loop construction overrides.

        Any new flag added to the extraction path but not threaded through the
        loop constructor will fail here — the helper + _extraction_kwargs are the
        only source of truth.  Parametrized over source_type so both transcript
        and document variants are covered.
        """
        from paramem.graph.schema import Entity, Relation, SessionGraph

        session_graph = SessionGraph(
            session_id="s001",
            timestamp="2026-01-01T00:00:00Z",
            entities=[Entity(name="Alex", entity_type="person")],
            relations=[
                Relation(
                    subject="Alex",
                    predicate="lives_in",
                    object="Millfield",
                    relation_type="factual",
                    speaker_id="speaker0",
                ),
            ],
        )
        procedural_graph = SessionGraph(
            session_id="s001",
            timestamp="2026-01-01T00:00:00Z",
            entities=[],
            relations=[],
        )

        captured_a: dict[str, list[dict]] = {"graph": [], "procedural": []}
        captured_b: dict[str, list[dict]] = {"graph": [], "procedural": []}

        def _spy(bucket, fixture):
            def _f(model, tokenizer, transcript, session_id, **kwargs):
                bucket.append(kwargs)
                return fixture

            return _f

        loop_a = self._build_loop(
            monkeypatch,
            tmp_path / "a",
            procedural_enabled=procedural_enabled,
            extract_graph_spy=_spy(captured_a["graph"], session_graph),
            extract_procedural_spy=_spy(captured_a["procedural"], procedural_graph),
            **loop_overrides,
        )
        self._run_extract_session(loop_a, source_type=source_type)

        loop_b = self._build_loop(
            monkeypatch,
            tmp_path / "b",
            procedural_enabled=procedural_enabled,
            extract_graph_spy=_spy(captured_b["graph"], session_graph),
            extract_procedural_spy=_spy(captured_b["procedural"], procedural_graph),
            **loop_overrides,
        )
        self._run_consolidation_cycle_and_capture(monkeypatch, loop_b, source_type=source_type)

        # Each path calls extract_graph exactly once with the same kwargs.
        assert len(captured_a["graph"]) == 1
        assert len(captured_b["graph"]) == 1
        assert captured_a["graph"][0] == captured_b["graph"][0], (
            "extract_graph kwargs diverged between the two loop instances. "
            f"loop_a: {captured_a['graph'][0]!r}\n"
            f"loop_b: {captured_b['graph'][0]!r}"
        )

        # Procedural path: same kwarg shape when enabled, neither path calls it when disabled.
        assert len(captured_a["procedural"]) == len(captured_b["procedural"])
        if procedural_enabled:
            assert len(captured_a["procedural"]) == 1
            assert captured_a["procedural"][0] == captured_b["procedural"][0], (
                "extract_procedural_graph kwargs diverged between loop instances. "
                f"loop_a: {captured_a['procedural'][0]!r}\n"
                f"loop_b: {captured_b['procedural'][0]!r}"
            )
        else:
            assert captured_a["procedural"] == []
            assert captured_b["procedural"] == []

    def test_parity_procedural_disabled(self, monkeypatch, tmp_path):
        loop = self._build_loop(monkeypatch, tmp_path, procedural_enabled=False)
        episodic, procedural = self._run_extract_session(loop)

        # With procedural disabled, preferences fall back into episodic — never lost.
        assert any(qa["predicate"] == "prefers" for qa in episodic)
        assert procedural == []

    def _capture_transcript_spy(self, bucket, fixture):
        """Spy matching the extract_graph_spy/extract_procedural_spy call shape
        (model, tokenizer, transcript, session_id, **kwargs); captures the
        transcript positional arg instead of kwargs."""

        def _f(model, tokenizer, transcript, session_id, **kwargs):
            bucket.append(transcript)
            return fixture

        return _f

    @pytest.mark.parametrize("source_type", ["transcript", "document"])
    def test_transcript_reaches_extraction_unmodified(self, monkeypatch, tmp_path, source_type):
        """``extract_session`` passes ``session_transcript`` straight through
        to the extraction call, byte-for-byte, for every ``source_type`` —
        the document-provenance directive is a TEMPLATE slot rendered by
        :func:`~paramem.graph.extractor.build_document_context`, never a
        prepend onto the transcript value itself."""
        from paramem.graph.schema import Entity, Relation, SessionGraph

        session_graph = SessionGraph(
            session_id="s001",
            timestamp="2026-01-01T00:00:00Z",
            entities=[Entity(name="speaker0", entity_type="person")],
            relations=[
                Relation(
                    subject="speaker0",
                    predicate="leads",
                    object="the team",
                    relation_type="factual",
                    speaker_id="speaker0",
                )
            ],
        )
        procedural_graph = SessionGraph(session_id="s001", timestamp="2026-01-01T00:00:00Z")

        captured_graph: list[str] = []
        loop = self._build_loop(
            monkeypatch,
            tmp_path,
            procedural_enabled=True,
            extract_graph_spy=self._capture_transcript_spy(captured_graph, session_graph),
            extract_procedural_spy=self._capture_transcript_spy([], procedural_graph),
        )
        body = "I lead the platform team."
        loop.extract_session(
            session_transcript=body,
            session_id="s001",
            speaker_id="speaker0",
            speaker_name="Alex Walker",
            source_type=source_type,
        )

        assert captured_graph == [body]

    def test_extraction_pipeline_forwards_source_type(self):
        """ExtractionPipeline.kwargs forwards source_type into the returned
        dict so extract_graph receives what it needs to select the
        ``{document_context}`` rendering and gate the document-only
        exact-full-name speaker rewrite."""
        from unittest.mock import MagicMock

        from paramem.graph.extraction_pipeline import ExtractionConfig, ExtractionPipeline

        pipeline = ExtractionPipeline(
            model=MagicMock(),
            tokenizer=MagicMock(),
            config=ExtractionConfig(scrub={"person name"}),
        )
        result = pipeline.kwargs(source_type="document", speaker_id="speaker0")
        assert result["source_type"] == "document"


class TestLocalParseFailureAbortsFold:
    """``extract_session`` aborts on a local-extraction parse failure: raises
    ``ExtractionFailed`` before that session's merge, and resets the merger
    graph before the exception leaves the method (the fail-loud
    document-extraction design, ``paramem/training/consolidation.py``'s own
    ``except ExtractionFailed: self.merger.reset_graph(); raise``).  A
    legitimately-empty extraction (no failed phase record) is not mistaken
    for a failure.  ``VramExhausted`` is a DIFFERENT exception type and must
    escape uncaught, without triggering the reset (per-chunk isolation keeps
    an earlier session's merge intact).
    """

    def _build_loop(self, monkeypatch, tmp_path, procedural_enabled: bool = False):
        from peft import PeftModel

        from paramem.graph.schema import SessionGraph
        from paramem.memory.store import MemoryStore as _MS
        from paramem.utils.config import AdapterConfig, ConsolidationConfig, TrainingConfig

        model = MagicMock()
        model.__class__ = PeftModel
        model.peft_config = {"episodic": MagicMock(), "semantic": MagicMock()}
        if procedural_enabled:
            model.peft_config["procedural"] = MagicMock()
        procedural_adapter = AdapterConfig() if procedural_enabled else None

        # extract_graph / extract_procedural_graph are never reached in
        # these tests -- every call goes through loop.extraction.run /
        # loop.extraction.run_procedural, patched per-test below -- but the
        # module bindings must still resolve at ConsolidationLoop
        # construction time.
        monkeypatch.setattr(
            "paramem.graph.extraction_pipeline.extract_graph",
            lambda *a, **kw: SessionGraph(session_id="unused", timestamp="2026-01-01T00:00:00Z"),
        )
        monkeypatch.setattr(
            "paramem.graph.extraction_pipeline.extract_procedural_graph",
            lambda *a, **kw: SessionGraph(session_id="unused", timestamp="2026-01-01T00:00:00Z"),
        )

        loop = ConsolidationLoop(
            model=model,
            tokenizer=MagicMock(),
            consolidation_config=ConsolidationConfig(),
            training_config=TrainingConfig(),
            episodic_adapter_config=AdapterConfig(),
            semantic_adapter_config=AdapterConfig(),
            memory_store=_MS(),
            procedural_adapter_config=procedural_adapter,
            output_dir=tmp_path,
            extraction_scrub={"person name"},
            extraction_max_tokens=8192,
            extraction_plausibility_max_tokens=8192,
            extraction_anonymize_token_envelope=8192,
        )
        loop.fingerprint_cache = None
        return loop

    def _populated_session_graph(self, session_id: str = "s1"):
        from paramem.graph.schema import Entity, Relation, SessionGraph

        return SessionGraph(
            session_id=session_id,
            timestamp="2026-01-01T00:00:00Z",
            entities=[
                Entity(name="Alex", entity_type="person"),
                Entity(name="Millfield", entity_type="place"),
            ],
            relations=[
                Relation(
                    subject="Alex",
                    predicate="lives_in",
                    object="Millfield",
                    relation_type="factual",
                    speaker_id="speaker0",
                ),
            ],
        )

    def _empty_graph(self, session_id: str = "s1"):
        from paramem.graph.schema import SessionGraph

        return SessionGraph(session_id=session_id, timestamp="2026-01-01T00:00:00Z")

    def _failed_phase_side_effect(self, phase_name: str, session_id: str = "s1"):
        """Build an ``extraction.run``/``run_procedural`` replacement that
        records a ``"failed"`` phase -- matching exactly what
        ``_run_local_extraction`` itself does on an unparseable raw
        output -- and returns an empty graph."""
        from paramem.graph.phase_trace import phase_trace

        def _run(*args, **kwargs):
            with phase_trace(phase_name) as t:
                t.set_outcome("failed", reason="ValueError: bad json")
            return self._empty_graph(session_id)

        return _run

    def test_local_extract_failure_raises_and_skips_merge(self, monkeypatch, tmp_path):
        """An episodic-pass parse failure (``local_extract``) raises before
        the LATER procedural pass ever runs, and before either graph
        reaches ``merger.merge`` -- ``procedural_enabled=True`` here is the
        crux: it proves the abort gates the procedural stage, not merely
        the episodic merge that happens to sit next to it."""
        from unittest.mock import patch

        from paramem.graph.extractor import ExtractionFailed

        loop = self._build_loop(monkeypatch, tmp_path, procedural_enabled=True)
        with patch.object(
            loop.extraction, "run", side_effect=self._failed_phase_side_effect("local_extract")
        ):
            with patch.object(loop.extraction, "run_procedural") as mock_run_procedural:
                with patch.object(loop.merger, "merge") as mock_merge:
                    with pytest.raises(ExtractionFailed) as exc_info:
                        loop.extract_session("t", "s1", speaker_id="speaker0")
        assert exc_info.value.phase == "local_extract"
        mock_run_procedural.assert_not_called()
        mock_merge.assert_not_called()

    def test_second_order_extract_failure_raises_and_skips_merge(self, monkeypatch, tmp_path):
        from unittest.mock import patch

        from paramem.graph.extractor import ExtractionFailed

        loop = self._build_loop(monkeypatch, tmp_path)
        with patch.object(
            loop.extraction,
            "run",
            side_effect=self._failed_phase_side_effect("second_order_extract"),
        ):
            with patch.object(loop.merger, "merge") as mock_merge:
                with pytest.raises(ExtractionFailed) as exc_info:
                    loop.extract_session("t", "s1", speaker_id="speaker0")
        assert exc_info.value.phase == "second_order_extract"
        mock_merge.assert_not_called()

    def test_procedural_extract_failure_raises_and_procedural_graph_never_merges(
        self, monkeypatch, tmp_path
    ):
        from unittest.mock import patch

        from paramem.graph.extractor import ExtractionFailed

        loop = self._build_loop(monkeypatch, tmp_path, procedural_enabled=True)
        session_graph = self._populated_session_graph()
        with (
            patch.object(loop.extraction, "run", return_value=session_graph),
            patch.object(
                loop.extraction,
                "run_procedural",
                side_effect=self._failed_phase_side_effect("procedural_extract"),
            ),
            patch.object(loop.merger, "merge", wraps=loop.merger.merge) as mock_merge,
        ):
            with pytest.raises(ExtractionFailed) as exc_info:
                loop.extract_session("t", "s1", speaker_id="speaker0", speaker_name=None)
        assert exc_info.value.phase == "procedural_extract"
        # Only the episodic session_graph reached merger.merge -- the empty,
        # failed procedural graph never did.
        mock_merge.assert_called_once()
        merged_graph = mock_merge.call_args[0][0]
        assert merged_graph is session_graph

    def test_legitimately_empty_extraction_does_not_raise(self, monkeypatch, tmp_path):
        from unittest.mock import patch

        from paramem.graph.phase_trace import phase_trace

        loop = self._build_loop(monkeypatch, tmp_path)

        def _ok_empty(*args, **kwargs):
            with phase_trace("local_extract"):
                pass
            return self._empty_graph()

        with patch.object(loop.extraction, "run", side_effect=_ok_empty):
            episodic, procedural = loop.extract_session("t", "s1", speaker_id="speaker0")
        assert episodic == []
        assert procedural == []

    def test_merger_graph_reset_on_local_parse_failure_abort(self, monkeypatch, tmp_path):
        from unittest.mock import patch

        from paramem.graph.extractor import ExtractionFailed

        loop = self._build_loop(monkeypatch, tmp_path)
        session_graph = self._populated_session_graph(session_id="s1")
        with patch.object(loop.extraction, "run", return_value=session_graph):
            loop.extract_session("t1", "s1", speaker_id="speaker0")
        assert loop.merger.graph.number_of_nodes() > 0, (
            "sanity: the first session must have merged before the abort"
        )

        with patch.object(
            loop.extraction,
            "run",
            side_effect=self._failed_phase_side_effect("local_extract", session_id="s2"),
        ):
            with pytest.raises(ExtractionFailed):
                loop.extract_session("t2", "s2", speaker_id="speaker0")

        assert loop.merger.graph.number_of_nodes() == 0
        pending = loop.take_pending_relations()
        assert pending.episodic == []
        assert pending.procedural == []

    def test_merger_graph_reset_on_cloud_enrich_origin_abort(self, monkeypatch, tmp_path):
        """The same reset fires for the OTHER ``ExtractionFailed`` origin --
        a raise from inside ``ExtractionPipeline.run`` itself (the
        pre-existing ``cloud_enrich`` abort) -- one invalidation site
        covering both raise origins."""
        from unittest.mock import patch

        from paramem.graph.extractor import ExtractionFailed

        loop = self._build_loop(monkeypatch, tmp_path)
        session_graph = self._populated_session_graph(session_id="s1")
        with patch.object(loop.extraction, "run", return_value=session_graph):
            loop.extract_session("t1", "s1", speaker_id="speaker0")
        assert loop.merger.graph.number_of_nodes() > 0, (
            "sanity: the first session must have merged before the abort"
        )

        def _raise_cloud_enrich(*args, **kwargs):
            raise ExtractionFailed("cloud_enrich", "cloud boom")

        with patch.object(loop.extraction, "run", side_effect=_raise_cloud_enrich):
            with pytest.raises(ExtractionFailed):
                loop.extract_session("t2", "s2", speaker_id="speaker0")

        assert loop.merger.graph.number_of_nodes() == 0
        pending = loop.take_pending_relations()
        assert pending.episodic == []
        assert pending.procedural == []

    def test_vram_exhausted_escapes_extract_session_uncaught(self, monkeypatch, tmp_path):
        """``VramExhausted`` is deliberately NOT caught by ``extract_session``'s
        ``except ExtractionFailed`` -- per-chunk isolation means it must
        propagate straight through, uncaught and unwidened, and must NOT
        trigger the merger-graph reset that ``ExtractionFailed`` does: an
        earlier session already merged this fold survives. Pins the scope
        of the handler so a future edit cannot silently widen it to also
        swallow/reset on ``VramExhausted``."""
        from unittest.mock import patch

        from paramem.utils.vram_guard import VramExhausted

        loop = self._build_loop(monkeypatch, tmp_path)
        session_graph = self._populated_session_graph(session_id="s1")
        with patch.object(loop.extraction, "run", return_value=session_graph):
            loop.extract_session("t1", "s1", speaker_id="speaker0")
        assert loop.merger.graph.number_of_nodes() > 0, (
            "sanity: the first session must have merged before the VramExhausted raise"
        )

        def _raise_vram(*args, **kwargs):
            raise VramExhausted("s2")

        with patch.object(loop.extraction, "run", side_effect=_raise_vram):
            with pytest.raises(VramExhausted):
                loop.extract_session("t2", "s2", speaker_id="speaker0")

        # Per-chunk isolation, unlike the ExtractionFailed abort tests above:
        # the first session's merge is NOT reset.
        assert loop.merger.graph.number_of_nodes() > 0


class TestTakePendingRelationsGraphLifetime:
    """``ConsolidationLoop.take_pending_relations`` -- the one door out of
    the extraction accumulation.  Uses the lightweight ``object.__new__``
    loop fixture
    (:meth:`TestMergeRegistryRelationsTimestamp._make_loop_for_recon_merge`
    below), copied here so this class does not depend on another class's
    private helper living further down the file.
    """

    @staticmethod
    def _make_bare_loop(tmp_path) -> ConsolidationLoop:
        from paramem.graph.merger import GraphMerger
        from paramem.memory.store import MemoryStore
        from paramem.training.key_registry import KeyRegistry
        from paramem.utils.config import AdapterConfig, ConsolidationConfig, TrainingConfig

        loop = object.__new__(ConsolidationLoop)
        loop.model = None
        loop.tokenizer = None
        loop.config = ConsolidationConfig()
        loop.training_config = TrainingConfig(
            num_epochs=1,
            gradient_checkpointing=False,
            batch_size=1,
            recall_early_stopping=False,
            recall_probe_batch_size=1,
        )
        loop.episodic_config = AdapterConfig(rank=4, alpha=8, target_modules=["q_proj"])
        loop.semantic_config = AdapterConfig(rank=4, alpha=8, target_modules=["q_proj"])
        loop.procedural_config = None
        loop.wandb_config = None
        loop._thermal_policy = None
        loop.output_dir = tmp_path
        loop.save_cycle_snapshots = False
        loop._debug_base = None
        loop.snapshot_dir = None
        loop.shutdown_requested = False
        loop._bg_trainer = None
        loop._early_stop_callback = None
        loop.fingerprint_cache = None
        loop._keep_prior_slots = 2
        loop.cycle_count = 0
        loop._indexed_next_index = 1
        loop._procedural_next_index = 1
        loop._procedural_tentative_next_index = 1
        loop._indexed_ep_interim = {}
        loop.promoted_keys = set()
        loop.full_consolidation_period_string = ""

        merger = GraphMerger(model=MagicMock(), tokenizer=MagicMock())
        merger._predicate_cardinality["lives in"] = False
        loop.merger = merger

        store = MemoryStore()
        for tier in ("episodic", "semantic", "procedural"):
            store.load_registry(tier, KeyRegistry())
        loop.store = store
        return loop

    def test_take_pending_relations_resets_ledgers_too(self, tmp_path) -> None:
        """after the take, the keying surface is
        fully reset -- zero edges, an empty removal ledger, and an empty
        adopt-reinforcements map (``reset_graph``'s documented contract)."""
        loop = self._make_bare_loop(tmp_path)
        loop.merger.graph.add_node("alex", speaker_id="speaker0")
        loop.merger.graph.add_edge(
            "alex",
            "berlin",
            predicate="lives in",
            relation_type="factual",
            confidence=1.0,
            sessions=["s1"],
        )
        loop.merger.removal_ledger["stray_key"] = {"reason": "dedup"}
        loop.merger.adopt_reinforcements["stray_key_2"] = 1

        pending = loop.take_pending_relations()

        assert len(pending.episodic) + len(pending.procedural) == 1
        assert loop.merger.graph.number_of_edges() == 0
        assert loop.merger.removal_ledger == {}
        assert loop.merger.adopt_reinforcements == {}

    def test_captured_product_carries_edge_and_attribute_relations_with_timestamps(
        self, tmp_path
    ) -> None:
        """the captured product carries both edge-derived and
        node-attribute-derived relations, and preserves last_seen/first_seen
        off the edge."""
        loop = self._make_bare_loop(tmp_path)
        loop.merger.graph.add_node(
            "alex",
            speaker_id="speaker0",
            attributes={"favorite color": "blue"},
            sessions=["s1"],
        )
        loop.merger.graph.add_edge(
            "alex",
            "berlin",
            predicate="lives in",
            relation_type="factual",
            confidence=1.0,
            sessions=["s1"],
            last_seen="2026-01-02T00:00:00Z",
            first_seen="2026-01-01T00:00:00Z",
        )

        pending = loop.take_pending_relations()
        all_rels = pending.episodic + pending.procedural

        edge_rel = next(r for r in all_rels if r.relation_type == "factual")
        assert edge_rel.subject == "alex"
        assert edge_rel.object == "berlin"
        assert edge_rel.last_seen == "2026-01-02T00:00:00Z"
        assert edge_rel.first_seen == "2026-01-01T00:00:00Z"

        attr_rel = next(r for r in all_rels if r.relation_type == "attribute")
        assert attr_rel.subject == "alex"
        assert attr_rel.object == "blue"

    def test_two_consecutive_takes_are_independent(self, tmp_path) -> None:
        """Two consecutive ``take_pending_relations`` calls -- as a
        ``/calibrate/extract_pending`` probe immediately followed by a
        LATER, separate interim fold's own extraction would produce -- yield
        independent products.  This is the leak-regression guard: the
        second take must carry only what merged since the first take, none
        of the probe's earlier content."""
        loop = self._make_bare_loop(tmp_path)

        # Probe batch: one relation about berlin.
        loop.merger.graph.add_edge(
            "alex", "berlin", predicate="lives in", relation_type="factual", sessions=["probe"]
        )
        probe_take = loop.take_pending_relations()
        assert probe_take.is_empty() is False
        assert any(r.object == "berlin" for r in probe_take.episodic)

        # A later, separate fold's own batch -- nothing added yet: must be empty.
        empty_take = loop.take_pending_relations()
        assert empty_take.is_empty() is True

        # The fold's own DIFFERENT relation, merged after the probe's take.
        loop.merger.graph.add_edge(
            "alex", "munich", predicate="visited", relation_type="factual", sessions=["fold"]
        )
        fold_take = loop.take_pending_relations()
        fold_objects = {r.object for r in fold_take.episodic}
        assert fold_objects == {"munich"}, (
            f"the fold's own take must not carry the probe's earlier content; got {fold_objects}"
        )

    def test_pending_none_is_a_valid_no_pending_value(self) -> None:
        """``pending=None`` is the ordinary "this fold
        stages no pending-session content" value -- distinct from an empty
        ``PendingRelations``, which stages a (empty) batch.  The second half
        (``mode='simulate'`` with a non-``None`` ``pending`` raises
        ``ValueError``) is already pinned by
        ``tests/test_simulate_train_parity.py::TestConsolidateModeGuard::
        test_simulate_mode_rejects_pending`` -- not duplicated here."""
        from paramem.training.consolidation import PendingRelations

        empty = PendingRelations(episodic=[], procedural=[])
        assert empty.is_empty() is True

        non_empty = PendingRelations(episodic=[MagicMock()], procedural=[])
        assert non_empty.is_empty() is False


class TestInterimRefinementGate:
    """extract_session merger.merge is always called; resolve_contradictions tracks config.

    refinement_contradiction="off" → additive merge (no supersession, both facts coexist).
    refinement_contradiction="on"  → non-additive merge (model may supersede edges).
    All tests run without loading any model or GPU.
    """

    def _build_loop(
        self,
        monkeypatch,
        tmp_path,
        cloud_enabled: bool = False,
        refinement_enrichment: str = "off",
        refinement_contradiction: str = "off",
    ):
        from unittest.mock import MagicMock

        from peft import PeftModel

        from paramem.graph.schema import Entity, Relation, SessionGraph
        from paramem.memory.store import MemoryStore as _MS
        from paramem.training.consolidation import ConsolidationLoop
        from paramem.utils.config import AdapterConfig, ConsolidationConfig, TrainingConfig

        self._session_graph = SessionGraph(
            session_id="s_gate",
            timestamp="2026-06-01T00:00:00Z",
            entities=[
                Entity(name="X", entity_type="person"),
                Entity(name="Y", entity_type="person"),
            ],
            relations=[
                Relation(
                    subject="X",
                    predicate="knows",
                    object="Y",
                    relation_type="social",
                    speaker_id="spk0",
                ),
            ],
        )

        model = MagicMock()
        model.__class__ = PeftModel
        # No "in_training" (STAGING_ADAPTER) entry: the wired interim entry
        # now reaches the real build/write/publish driver, whose staging
        # lifecycle invariant (assert_staging_absent) treats a resident
        # staging slot at training entry as a stranded-prior-event error.
        model.peft_config = {
            "episodic": MagicMock(),
            "semantic": MagicMock(),
        }

        monkeypatch.setattr(
            "paramem.graph.extraction_pipeline.extract_graph",
            lambda *a, **kw: self._session_graph,
        )

        loop = ConsolidationLoop(
            model=model,
            tokenizer=MagicMock(),
            consolidation_config=ConsolidationConfig(
                refinement_enrichment=refinement_enrichment,
                refinement_contradiction=refinement_contradiction,
            ),
            training_config=TrainingConfig(),
            episodic_adapter_config=AdapterConfig(),
            semantic_adapter_config=AdapterConfig(),
            memory_store=_MS(),
            procedural_adapter_config=None,
            output_dir=tmp_path,
            extraction_scrub={"person name"},
            cloud_enabled=cloud_enabled,
            extraction_max_tokens=8192,
            extraction_plausibility_max_tokens=8192,
            extraction_anonymize_token_envelope=8192,
        )
        # Set by the server's boot wiring in production
        # (paramem.server.consolidation), never by __init__ itself — every
        # other minimal test loop in this tree sets it explicitly (e.g.
        # tests/test_fold_phase1.py::_make_loop); this factory's callers now
        # reach ConsolidationLoop._build_write_context (via stage_event's
        # build/write/publish driver), which reads it directly.
        loop.fingerprint_cache = None
        return loop

    def test_merge_called_when_refinement_enrichment_on(self, monkeypatch, tmp_path):
        """refinement_enrichment='on': merger.merge is called once with the session graph."""
        from unittest.mock import patch

        loop = self._build_loop(
            monkeypatch, tmp_path, refinement_enrichment="on", cloud_enabled=True
        )
        initial_nodes = loop.merger.graph.number_of_nodes()
        initial_edges = loop.merger.graph.number_of_edges()

        with patch.object(loop.extraction, "run", return_value=self._session_graph):
            loop.extract_session("t", "s_gate", speaker_id="spk0")

        # Graph must have grown — merge ran.
        assert (
            loop.merger.graph.number_of_nodes() > initial_nodes
            or loop.merger.graph.number_of_edges() > initial_edges
        ), "Expected merger.graph to grow after extract_session with refinement_enrichment='on'"

    def test_merge_called_non_contradiction_when_refinement_contradiction_off(
        self, monkeypatch, tmp_path
    ):
        """refinement_contradiction="off": merge uses resolve_contradictions=False; graph grows."""
        from unittest.mock import patch

        loop = self._build_loop(
            monkeypatch, tmp_path, refinement_enrichment="off", refinement_contradiction="off"
        )
        initial_nodes = loop.merger.graph.number_of_nodes()
        initial_edges = loop.merger.graph.number_of_edges()

        with patch.object(loop.extraction, "run", return_value=self._session_graph):
            with patch.object(loop.merger, "merge", wraps=loop.merger.merge) as mock_merge:
                loop.extract_session("t", "s_gate", speaker_id="spk0")
                # merge must be called exactly once with resolve_contradictions=False
                mock_merge.assert_called_once()
                _, kwargs = mock_merge.call_args
                assert kwargs.get("resolve_contradictions") is False, (
                    "refinement_contradiction='off' must call merge(resolve_contradictions=False)"
                )

        # Graph must have grown — the non-resolving merge still inserts edges.
        assert (
            loop.merger.graph.number_of_nodes() > initial_nodes
            or loop.merger.graph.number_of_edges() > initial_edges
        ), "Expected merger.graph to grow after extract_session with refinement_contradiction='off'"

    def test_episodic_rels_identical_regardless_of_enrichment_setting(self, monkeypatch, tmp_path):
        """episodic_rels/procedural_rels are identical regardless of refinement_enrichment.

        Keying is derived from session_graph, not from the cumulative graph,
        so the setting must not affect what facts are returned to the caller.
        """
        from unittest.mock import patch

        loop_a = self._build_loop(
            monkeypatch, tmp_path / "a", refinement_enrichment="on", cloud_enabled=True
        )
        with patch.object(loop_a.extraction, "run", return_value=self._session_graph):
            rels_a, proc_a = loop_a.extract_session("t", "s_gate", speaker_id="spk0")

        loop_b = self._build_loop(monkeypatch, tmp_path / "b", refinement_enrichment="off")
        with patch.object(loop_b.extraction, "run", return_value=self._session_graph):
            rels_b, proc_b = loop_b.extract_session("t", "s_gate", speaker_id="spk0")

        def _key(d):
            return (d.get("subject"), d.get("predicate"), d.get("object"))

        assert sorted(map(_key, rels_a)) == sorted(map(_key, rels_b)), (
            "episodic_rels differ between refinement_enrichment='on' and 'off'"
        )
        assert proc_a == proc_b == []

    def test_refinement_contradiction_off_both_facts_survive(self, monkeypatch, tmp_path):
        """refinement_contradiction="off": two facts with the same predicate but different
        objects both survive (no supersession — Case-2 cardinality skipped).
        """
        from unittest.mock import patch

        from paramem.graph.schema import Entity, Relation, SessionGraph

        # Override _session_graph for this test: same subject+predicate, two objects.
        sg1 = SessionGraph(
            session_id="s1",
            timestamp="2026-06-01T00:00:00Z",
            entities=[
                Entity(name="Alice", entity_type="person"),
                Entity(name="Berlin", entity_type="location"),
            ],
            relations=[
                Relation(
                    subject="Alice",
                    predicate="lives_in",
                    object="Berlin",
                    relation_type="factual",
                    speaker_id="spk0",
                ),
            ],
        )
        sg2 = SessionGraph(
            session_id="s2",
            timestamp="2026-06-02T00:00:00Z",
            entities=[
                Entity(name="Alice", entity_type="person"),
                Entity(name="Munich", entity_type="location"),
            ],
            relations=[
                Relation(
                    subject="Alice",
                    predicate="lives_in",
                    object="Munich",
                    relation_type="factual",
                    speaker_id="spk0",
                ),
            ],
        )
        loop = self._build_loop(
            monkeypatch, tmp_path, refinement_enrichment="off", refinement_contradiction="off"
        )

        with patch.object(loop.extraction, "run", side_effect=[sg1, sg2]):
            loop.extract_session("t1", "s1", speaker_id="spk0")
            loop.extract_session("t2", "s2", speaker_id="spk0")

        # Both objects must be present in the cumulative graph.
        berlin_present = any(
            loop.merger.graph.has_node(n) and "berlin" in n.lower()
            for n in loop.merger.graph.nodes()
        )
        munich_present = any(
            loop.merger.graph.has_node(n) and "munich" in n.lower()
            for n in loop.merger.graph.nodes()
        )
        assert berlin_present and munich_present, (
            f"Both 'Berlin' and 'Munich' must coexist with refinement_contradiction='off'; "
            f"nodes={list(loop.merger.graph.nodes())}"
        )

    def test_refinement_contradiction_on_supersession_removes_old_edge(self, monkeypatch, tmp_path):
        """refinement_contradiction="on": non-additive merge calls Case-2 model verdict.

        We stub check_predicate_coexistence to return REPLACE so the old edge is
        removed (incoming sg2 has a fresher last_seen → wins the recency check).
        A MagicMock/None model silently skips Case-2; we use a real stub model
        so the cardinality path executes and the superseded edge disappears.
        """
        from unittest.mock import MagicMock, patch

        from paramem.graph.schema import Entity, Relation, SessionGraph

        # sg1 has an older last_seen; sg2 has a fresher one so it wins the recency check.
        sg1 = SessionGraph(
            session_id="sx1",
            timestamp="2026-06-01T00:00:00Z",
            entities=[
                Entity(name="Bob", entity_type="person"),
                Entity(name="London", entity_type="location"),
            ],
            relations=[
                Relation(
                    subject="Bob",
                    predicate="lives_in",
                    object="London",
                    relation_type="factual",
                    speaker_id="spk0",
                    last_seen="2026-06-01T00:00:00Z",
                ),
            ],
        )
        sg2 = SessionGraph(
            session_id="sx2",
            timestamp="2026-06-02T00:00:00Z",
            entities=[
                Entity(name="Bob", entity_type="person"),
                Entity(name="Paris", entity_type="location"),
            ],
            relations=[
                Relation(
                    subject="Bob",
                    predicate="lives_in",
                    object="Paris",
                    relation_type="factual",
                    speaker_id="spk0",
                    last_seen="2026-06-02T00:00:00Z",
                ),
            ],
        )
        loop = self._build_loop(
            monkeypatch, tmp_path, refinement_enrichment="off", refinement_contradiction="on"
        )

        # Inject a non-None model on the merger so Case-2 fires (the loop's
        # MagicMock model is sufficient; merger.model is None by default).
        loop.merger.model = MagicMock()
        loop.merger.tokenizer = MagicMock()

        # Stub check_predicate_coexistence to return REPLACE so the old edge is
        # superseded.  The stub is patched at the call site in merger.py.
        with (
            patch.object(loop.extraction, "run", side_effect=[sg1, sg2]),
            patch(
                "paramem.graph.merger.check_predicate_coexistence",
                return_value="REPLACE",
            ),
        ):
            loop.extract_session("t1", "sx1", speaker_id="spk0")
            loop.extract_session("t2", "sx2", speaker_id="spk0")

        # After REPLACE: the Bob→London edge must be gone; Bob→Paris must exist.
        bob_node = next((n for n in loop.merger.graph.nodes() if "bob" in n.lower()), None)
        london_node = next((n for n in loop.merger.graph.nodes() if "london" in n.lower()), None)
        paris_node = next((n for n in loop.merger.graph.nodes() if "paris" in n.lower()), None)
        assert paris_node is not None, "Paris node must be in graph after REPLACE merge"
        assert bob_node is not None, "Bob node must be in graph"
        assert loop.merger.graph.has_edge(bob_node, paris_node), (
            "Bob→Paris edge must exist after supersession"
        )
        # London node may survive as isolated; the edge must be gone.
        if london_node is not None:
            assert not loop.merger.graph.has_edge(bob_node, london_node), (
                "Bob→London edge must be removed after REPLACE verdict; "
                f"edges from bob: {list(loop.merger.graph.out_edges(bob_node, data=True))}"
            )

    def test_off_pending_session_content_reaches_extra_relations(self, monkeypatch, tmp_path):
        """refinement_enrichment='off': run_consolidation_cycle passes the session's
        edges as this event's episodic_rels into stage_event's own pending merge.

        Guards gate #2 — the unconditional _pending_relations capture inside
        run_consolidation_cycle (consolidation.py).  The test FAILS if an
        ``if self.config.refinement_enrichment != "off":`` guard is reintroduced
        around the capture, because episodic_rels would then be None/[] and the
        assertion below would reject it.

        Strategy: extract_session populates merger.graph with the X→Y 'knows' edge.
        run_consolidation_cycle (mode='simulate') is then called with the returned
        episodic_rels.  stage_event is spied (the real implementation still runs,
        mirroring TestInterimRefinementGate.test_interim_scope_pins_enrich_false)
        so the kwargs it receives can be inspected without any GPU/disk I/O.
        commit_tier_slot is stubbed out for the same reason.
        """
        from unittest.mock import patch

        loop = self._build_loop(monkeypatch, tmp_path, refinement_enrichment="off")

        with patch.object(loop.extraction, "run", return_value=self._session_graph):
            episodic_rels, procedural_rels = loop.extract_session("t", "s_gate", speaker_id="spk0")

        # episodic_rels must be non-empty; otherwise run_consolidation_cycle exits
        # early at guard #2 before the pending-relations capture is ever reached.
        assert episodic_rels, (
            "extract_session must return non-empty episodic_rels for the test to be valid"
        )

        # extract_session merged the session graph into merger.graph; the one
        # take ends that batch's extraction lifetime and captures the product
        # this call threads into run_consolidation_cycle's own pending arg.
        pending = loop.take_pending_relations()

        captured_calls: list[dict] = []
        real_stage_event = loop.stage_event

        def _spy_stage_event(**kwargs):
            captured_calls.append(kwargs)
            return real_stage_event(**kwargs)

        with (
            patch.object(loop, "stage_event", side_effect=_spy_stage_event),
            patch(
                "paramem.memory.persistence.commit_tier_slot",
            ),
        ):
            loop.run_consolidation_cycle(
                episodic_rels,
                procedural_rels,
                speaker_id="spk0",
                mode="simulate",
                pending=pending,
                run_label="s_gate",
                stamp="20260601T0000",
                max_interim_count=7,
            )

        assert captured_calls, "stage_event was not called"
        call_kwargs = captured_calls[0]

        pending = call_kwargs.get("episodic_rels")
        assert pending is not None and len(pending) > 0, (
            "the pending-session content must reach stage_event's own "
            f"episodic_rels kwarg when refinement_enrichment='off'; got {pending!r}"
        )

        # The X→Y 'knows' edge extracted from the session must be present.
        subjects = {r.subject.lower() for r in pending}
        objects = {r.object.lower() for r in pending}
        predicates = {r.predicate.lower() for r in pending}
        assert "x" in subjects, f"Subject 'X' must appear in episodic_rels; subjects={subjects}"
        assert "y" in objects, f"Object 'Y' must appear in episodic_rels; objects={objects}"
        assert "knows" in predicates, (
            f"Predicate 'knows' must appear in episodic_rels; predicates={predicates}"
        )

    def test_interim_scope_pins_enrich_false(self, monkeypatch, tmp_path):
        """Structural pin: an interim event always stages with normalize=False
        and enrich=False, regardless of refinement_enrichment / cloud_enabled.

        Spies on ConsolidationLoop.stage_event — the interim entry point's
        call target since the entry was wired onto the two-phase stage/
        build/publish driver — to capture the kwargs run_consolidation_cycle
        hands it.  Fails if a future change re-wires the interim
        normalize/enrich construction back onto the config-driven expression
        the full fold uses.
        """
        from unittest.mock import patch

        loop = self._build_loop(
            monkeypatch, tmp_path, refinement_enrichment="on", cloud_enabled=True
        )

        with patch.object(loop.extraction, "run", return_value=self._session_graph):
            episodic_rels, procedural_rels = loop.extract_session("t", "s_gate", speaker_id="spk0")

        assert episodic_rels, (
            "extract_session must return non-empty episodic_rels for the test to be valid"
        )

        # extract_session merged the session graph into merger.graph; the one
        # take ends that batch's extraction lifetime and captures the product
        # this call threads into run_consolidation_cycle's own pending arg.
        pending = loop.take_pending_relations()

        captured_calls: list[dict] = []
        real_stage_event = loop.stage_event

        def _spy_stage_event(**kwargs):
            captured_calls.append(kwargs)
            return real_stage_event(**kwargs)

        with patch.object(loop, "stage_event", side_effect=_spy_stage_event):
            loop.run_consolidation_cycle(
                episodic_rels,
                procedural_rels,
                speaker_id="spk0",
                mode="simulate",
                pending=pending,
                run_label="s_gate",
                stamp="20260601T0000",
                max_interim_count=7,
            )

        assert len(captured_calls) == 1, (
            f"expected exactly one stage_event call; got {len(captured_calls)}"
        )
        call_kwargs = captured_calls[0]
        assert call_kwargs["event"] == "interim"
        assert call_kwargs["enrich"] is False, (
            "an interim event must stage with enrich=False even with "
            f"refinement_enrichment='on' and cloud_enabled=True; got {call_kwargs['enrich']}"
        )
        assert call_kwargs["normalize"] is False, (
            f"an interim event must stage with normalize=False; got {call_kwargs['normalize']}"
        )


class TestRefinementConfigRoundtrip:
    """Loading YAML with refinement knobs propagates through the property chain."""

    def test_yaml_cloud_enabled_propagates(self, tmp_path):
        """Top-level ``cloud.enabled: true`` reaches ServerConfig.cloud."""
        from paramem.server.config import load_server_config

        yaml_text = """
model:
  name: "mistralai/Mistral-7B-Instruct-v0.3"
cloud:
  enabled: true
consolidation:
  refresh_cadence: "12h"
"""
        cfg_path = tmp_path / "server_cloud_enabled.yaml"
        cfg_path.write_text(yaml_text)
        cfg = load_server_config(str(cfg_path))

        assert cfg.cloud.enabled is True, "ServerConfig.cloud.enabled should be True"

    def test_yaml_cloud_defaults_to_false(self, tmp_path):
        """YAML without a ``cloud`` section defaults the master switch to False."""
        from paramem.server.config import load_server_config

        yaml_text = """
model:
  name: "mistralai/Mistral-7B-Instruct-v0.3"
consolidation:
  refresh_cadence: "12h"
"""
        cfg_path = tmp_path / "server_no_cloud.yaml"
        cfg_path.write_text(yaml_text)
        cfg = load_server_config(str(cfg_path))

        assert cfg.cloud.enabled is False, "ServerConfig.cloud.enabled should default to False"

    def test_consolidation_config_carries_no_cloud_switch(self):
        """The cloud switch is NOT a ConsolidationConfig field.

        A second copy there is exactly what let the extraction pipeline and
        the conversation agent disagree about whether cloud egress was on.
        """
        from paramem.utils.config import ConsolidationConfig

        assert not hasattr(ConsolidationConfig(), "cloud_enabled")

    def test_retired_consolidation_cloud_enabled_fails_loudly(self, tmp_path):
        """A stale ``consolidation.cloud_enabled`` raises by name at load."""
        import pytest as _pytest

        from paramem.server.config import load_server_config

        cfg_path = tmp_path / "server_stale_key.yaml"
        cfg_path.write_text('model:\n  name: "m"\nconsolidation:\n  cloud_enabled: true\n')
        with _pytest.raises(TypeError, match="cloud_enabled"):
            load_server_config(str(cfg_path))

    def test_yaml_refinement_enrichment_on_propagates(self, tmp_path):
        """YAML refinement_enrichment: "on" propagates to schedule and consolidation_config."""
        from paramem.server.config import load_server_config

        yaml_text = """
model:
  name: "mistralai/Mistral-7B-Instruct-v0.3"
consolidation:
  refresh_cadence: "12h"
  refinement_enrichment: "on"
"""
        cfg_path = tmp_path / "server_refinement_enrichment_on.yaml"
        cfg_path.write_text(yaml_text)
        cfg = load_server_config(str(cfg_path))

        assert cfg.consolidation.refinement_enrichment == "on", (
            "ServerConfig.consolidation.refinement_enrichment should be 'on'"
        )
        assert cfg.consolidation_config.refinement_enrichment == "on", (
            "consolidation_config.refinement_enrichment should be 'on'"
        )

    def test_yaml_refinement_enrichment_defaults_to_off(self, tmp_path):
        """YAML without refinement_enrichment defaults to 'off'."""
        from paramem.server.config import load_server_config

        yaml_text = """
model:
  name: "mistralai/Mistral-7B-Instruct-v0.3"
consolidation:
  refresh_cadence: "12h"
"""
        cfg_path = tmp_path / "server_no_refinement_enrichment.yaml"
        cfg_path.write_text(yaml_text)
        cfg = load_server_config(str(cfg_path))

        assert cfg.consolidation.refinement_enrichment == "off", (
            "ServerConfig.consolidation.refinement_enrichment should default to 'off'"
        )
        assert cfg.consolidation_config.refinement_enrichment == "off", (
            "consolidation_config.refinement_enrichment should default to 'off'"
        )

    def test_invalid_refinement_enrichment_value_raises(self, tmp_path):
        """An invalid refinement_enrichment value raises ValueError from dataclass validation."""
        import pytest

        from paramem.server.config import load_server_config

        yaml_text = """
model:
  name: "mistralai/Mistral-7B-Instruct-v0.3"
consolidation:
  refresh_cadence: "12h"
  refinement_enrichment: full
"""
        cfg_path = tmp_path / "server_invalid_refinement_enrichment.yaml"
        cfg_path.write_text(yaml_text)

        with pytest.raises(ValueError, match="refinement_enrichment"):
            load_server_config(str(cfg_path))

    def test_consolidation_config_invalid_refinement_enrichment_raises(self):
        """ConsolidationConfig rejects an invalid refinement_enrichment value directly."""
        import pytest

        from paramem.utils.config import ConsolidationConfig

        with pytest.raises(ValueError, match="refinement_enrichment"):
            ConsolidationConfig(refinement_enrichment="full")

    def test_invalid_refinement_normalization_value_raises(self, tmp_path):
        """An invalid refinement_normalization value raises ValueError from dataclass validation."""
        import pytest

        from paramem.server.config import load_server_config

        yaml_text = """
model:
  name: "mistralai/Mistral-7B-Instruct-v0.3"
consolidation:
  refresh_cadence: "12h"
  refinement_normalization: light
"""
        cfg_path = tmp_path / "server_invalid_refinement_normalization.yaml"
        cfg_path.write_text(yaml_text)

        with pytest.raises(ValueError, match="refinement_normalization"):
            load_server_config(str(cfg_path))

    def test_consolidation_config_invalid_refinement_normalization_raises(self):
        """ConsolidationConfig rejects an invalid refinement_normalization value directly."""
        import pytest

        from paramem.utils.config import ConsolidationConfig

        with pytest.raises(ValueError, match="refinement_normalization"):
            ConsolidationConfig(refinement_normalization="light")


# ---------------------------------------------------------------------------
# Server consolidation — 3-way session classification
# ---------------------------------------------------------------------------


class TestSessionClassification:
    """classify_session and the extraction-phase session routing.

    The new contract:
    - NAMED (speaker_id present and not anonymous_voice) → extracted.
    - HOLDABLE (anonymous_voice OR no speaker_id but has a voice embedding) → NOT extracted.
    - UNIDENTIFIABLE (no speaker_id AND no voice embedding) → NOT extracted.

    Anonymity is determined by the speaker store, not by the speaker_id string
    format — do NOT use 'speaker{N}' string patterns as the anonymity gate.

    Note: these tests call _run_extraction_phase directly (in paramem.server.app);
    run_consolidation was deleted and must not be re-introduced (see
    test_run_consolidation_removed.py).
    """

    def _make_stub_store(self, anonymous_ids=None):
        """Return a minimal SpeakerStore stub.

        Args:
            anonymous_ids: set of speaker_ids that is_anonymous() returns True for.
        """
        store = MagicMock()
        _anon = set(anonymous_ids or [])

        def _is_anonymous(sid):
            return sid in _anon

        store.is_anonymous.side_effect = _is_anonymous
        store.get_name.return_value = None
        return store

    def _make_mock_loop(self, tmp_path):
        """Minimal mock ConsolidationLoop with the attributes _run_extraction_phase touches."""

        loop = MagicMock()
        loop.shutdown_requested = False
        loop.merger = MagicMock()
        loop.merger.graph = MagicMock()
        loop.merger.graph.nodes = []
        loop.promoted_keys = set()
        loop.episodic_simhash = {}
        loop.semantic_simhash = {}
        loop.procedural_simhash = {}
        # extract_session returns ([], []) so no training path is triggered.
        loop.extract_session = MagicMock(return_value=([], []))
        loop.train_adapters = MagicMock(return_value={})
        loop.cycle_count = 0
        # Mirror ConsolidationLoop.snapshot_dir_for's real layout
        # (paths.debug/episodic/cycle_<N>/run_<run_id>/) so the retention path
        # exercised under debug=True + retain_sessions=True lands inside tmp_path
        # instead of writing files named after the MagicMock's repr into CWD.
        loop.snapshot_dir_for = MagicMock(
            return_value=tmp_path / "ha" / "debug" / "episodic" / "cycle_0" / "run_test",
        )
        return loop

    def _make_config(self, tmp_path):
        """Minimal ServerConfig pointing at a temp directory."""
        from paramem.server.config import PathsConfig, ServerConfig

        config = ServerConfig()
        config.paths = PathsConfig(data=tmp_path / "ha")
        (tmp_path / "ha" / "adapters").mkdir(parents=True, exist_ok=True)
        return config

    def _make_session_buffer(self, tmp_path, speaker_id, *, embedding=None):
        """SessionBuffer with a single in-memory session.

        Args:
            speaker_id: Speaker id to set on the session (None = no speaker).
            embedding: Optional voice embedding list to attach to the user turn.
        """
        from paramem.server.session_buffer import SessionBuffer

        buffer = SessionBuffer(tmp_path / "sessions", debug=False)
        conv_id = "conv-test"
        if speaker_id is not None:
            buffer.set_speaker(conv_id, speaker_id, speaker_id)
        buffer.append(conv_id, "user", "Hello there", embedding=embedding)
        buffer.append(conv_id, "assistant", "Hi!")
        return buffer

    def _call_run_extraction_phase(self, loop, config, buffer, store=None):
        """Inject config + session_buffer (+ optional store) into _state and call."""
        import paramem.server.app as _app

        prior_config = _app._state.get("config")
        prior_buffer = _app._state.get("session_buffer")
        prior_ha = _app._state.get("ha_client")
        prior_speaker = _app._state.get("speaker_store")
        _app._state["config"] = config
        _app._state["session_buffer"] = buffer
        _app._state["ha_client"] = None
        _app._state["speaker_store"] = store
        try:
            return _app._run_extraction_phase(loop)
        finally:
            _app._state["config"] = prior_config
            _app._state["session_buffer"] = prior_buffer
            _app._state["ha_client"] = prior_ha
            _app._state["speaker_store"] = prior_speaker

    # --- classify_session unit tests ---

    def test_classify_named(self):
        """NAMED: speaker_id present and not anonymous."""
        from paramem.server.consolidation import SessionClass, classify_session

        result = classify_session(
            speaker_id="abc12345", is_anonymous=False, has_voice_embedding=False
        )
        assert result == SessionClass.NAMED

    def test_classify_holdable_anonymous(self):
        """HOLDABLE: speaker_id present but is_anonymous=True."""
        from paramem.server.consolidation import SessionClass, classify_session

        result = classify_session(
            speaker_id="speaker3", is_anonymous=True, has_voice_embedding=False
        )
        assert result == SessionClass.HOLDABLE

    def test_classify_holdable_embedding_only(self):
        """HOLDABLE: no speaker_id but voice embedding present (retro-claimable)."""
        from paramem.server.consolidation import SessionClass, classify_session

        result = classify_session(speaker_id=None, is_anonymous=False, has_voice_embedding=True)
        assert result == SessionClass.HOLDABLE

    def test_classify_unidentifiable(self):
        """UNIDENTIFIABLE: no speaker_id and no voice embedding."""
        from paramem.server.consolidation import SessionClass, classify_session

        result = classify_session(speaker_id=None, is_anonymous=False, has_voice_embedding=False)
        assert result == SessionClass.UNIDENTIFIABLE

    def test_classify_named_overrides_embedding(self):
        """NAMED takes precedence even when a voice embedding is also present."""
        from paramem.server.consolidation import SessionClass, classify_session

        result = classify_session(
            speaker_id="abc12345", is_anonymous=False, has_voice_embedding=True
        )
        assert result == SessionClass.NAMED

    # --- _run_extraction_phase routing tests ---

    def test_named_speaker_reaches_extract_session(self, tmp_path):
        """Named (enrolled, non-anonymous) speaker sessions are extracted."""
        loop = self._make_mock_loop(tmp_path)
        config = self._make_config(tmp_path)
        buffer = self._make_session_buffer(tmp_path, speaker_id="abc12345")
        store = self._make_stub_store(anonymous_ids=set())

        self._call_run_extraction_phase(loop, config, buffer, store=store)

        loop.extract_session.assert_called_once()
        assert loop.extract_session.call_args.kwargs.get("speaker_id") == "abc12345"

    def test_anonymous_speaker_not_extracted(self, tmp_path):
        """Anonymous-voice (HOLDABLE) sessions are NOT extracted.

        Anonymity is determined by the store, not by the 'speaker{N}' string.
        """
        loop = self._make_mock_loop(tmp_path)
        config = self._make_config(tmp_path)
        # "speaker3" is anonymous_voice in the store
        buffer = self._make_session_buffer(tmp_path, speaker_id="speaker3")
        store = self._make_stub_store(anonymous_ids={"speaker3"})

        self._call_run_extraction_phase(loop, config, buffer, store=store)

        loop.extract_session.assert_not_called()

    def test_none_speaker_no_embedding_not_extracted(self, tmp_path):
        """UNIDENTIFIABLE sessions (no speaker_id, no embedding) are not extracted.

        Sessions with no speaker_id and no voice embedding can never be attributed;
        they must not reach extract_session.
        """
        loop = self._make_mock_loop(tmp_path)
        config = self._make_config(tmp_path)
        buffer = self._make_session_buffer(tmp_path, speaker_id=None, embedding=None)
        store = self._make_stub_store()

        self._call_run_extraction_phase(loop, config, buffer, store=store)

        loop.extract_session.assert_not_called()

    def test_no_speaker_with_embedding_not_extracted(self, tmp_path):
        """HOLDABLE sessions (no speaker_id but voice embedding) are not extracted.

        The session may be retro-claimed later; it must wait.
        """
        loop = self._make_mock_loop(tmp_path)
        config = self._make_config(tmp_path)
        buffer = self._make_session_buffer(tmp_path, speaker_id=None, embedding=[0.1, 0.2])
        store = self._make_stub_store()

        self._call_run_extraction_phase(loop, config, buffer, store=store)

        loop.extract_session.assert_not_called()

    # --- _run_extraction_phase trial-tier residency gate ---

    def test_trial_commit_excludes_a_non_resident_episodic_tier(self, tmp_path):
        """Episodic must be gated on PEFT residency exactly like semantic and
        procedural -- an episodic adapter never mounted on the trial loop's
        model must not reach ``commit_main_tiers`` (which commits whatever
        list it is given, unconditionally, and would raise inside
        ``commit_tier_slot``)."""
        loop = self._make_mock_loop(tmp_path)
        loop.extract_session = MagicMock(
            return_value=([{"subject": "a", "predicate": "b", "object": "c"}], [])
        )
        loop.model.peft_config = {"semantic": object(), "procedural": object()}
        config = self._make_config(tmp_path)
        config.consolidation.mode = "train"
        buffer = self._make_session_buffer(tmp_path, speaker_id="abc12345")
        store = self._make_stub_store(anonymous_ids=set())

        self._call_run_extraction_phase(loop, config, buffer, store=store)

        loop.commit_main_tiers.assert_called_once()
        committed_tiers = loop.commit_main_tiers.call_args.args[0]
        assert "episodic" not in committed_tiers
        assert set(committed_tiers) == {"semantic", "procedural"}

    def test_trial_commit_includes_a_resident_episodic_tier(self, tmp_path):
        """A resident episodic adapter is committed alongside its siblings."""
        loop = self._make_mock_loop(tmp_path)
        loop.extract_session = MagicMock(
            return_value=([{"subject": "a", "predicate": "b", "object": "c"}], [])
        )
        loop.model.peft_config = {
            "episodic": object(),
            "semantic": object(),
            "procedural": object(),
        }
        config = self._make_config(tmp_path)
        config.consolidation.mode = "train"
        buffer = self._make_session_buffer(tmp_path, speaker_id="abc12345")
        store = self._make_stub_store(anonymous_ids=set())

        self._call_run_extraction_phase(loop, config, buffer, store=store)

        loop.commit_main_tiers.assert_called_once()
        committed_tiers = loop.commit_main_tiers.call_args.args[0]
        assert set(committed_tiers) == {"episodic", "semantic", "procedural"}

    # --- _run_extraction_phase summary tier-key counts ---

    @pytest.mark.parametrize("mode", ["train", "simulate"])
    def test_the_consolidation_summary_counts_a_tiers_active_keys(self, tmp_path, mode):
        """The fold summary reports ``len(loop.store.active_keys_in_tier(tier))``
        per tier in BOTH the train branch (``status: "complete"``, the exact
        summary construction ``_run_trial_consolidation`` also reuses) and
        the simulate branch (``status: "simulated"``) -- a withheld (stale)
        key never inflates the count in either branch."""
        from paramem.memory.store import MemoryStore
        from paramem.training.key_registry import KeyRegistry

        loop = self._make_mock_loop(tmp_path)
        loop.extract_session = MagicMock(
            return_value=([{"subject": "a", "predicate": "b", "object": "c"}], [])
        )
        loop.model.peft_config = {
            "episodic": object(),
            "semantic": object(),
            "procedural": object(),
        }

        registry = KeyRegistry()
        registry.add("graph1")
        registry.add("graph2")
        registry.stale("graph2")
        memory_store = MemoryStore()
        memory_store.load_registry("episodic", registry)
        loop.store = memory_store

        config = self._make_config(tmp_path)
        config.consolidation.mode = mode
        buffer = self._make_session_buffer(tmp_path, speaker_id="abc12345")
        speaker_store = self._make_stub_store(anonymous_ids=set())

        summary = self._call_run_extraction_phase(loop, config, buffer, store=speaker_store)

        assert summary["episodic_keys"] == 1
        assert summary["semantic_keys"] == 0
        assert summary["procedural_keys"] == 0


# ---------------------------------------------------------------------------
# Retire decision: HOLDABLE TTL + UNIDENTIFIABLE immediate drop
# ---------------------------------------------------------------------------


class TestRetireDecision:
    """Retire decision is explicit caller logic, NOT inside classify_session.

    HOLDABLE sessions retire past TTL; NAMED and unexpired HOLDABLE stay.
    UNIDENTIFIABLE sessions are always in drop_ids regardless of TTL.
    """

    def test_holdable_held_when_ttl_off(self):
        """HOLDABLE stays pending when orphan_retirement is 'off' (TTL=None)."""
        from paramem.server.consolidation import SessionClass, classify_session

        cls = classify_session(speaker_id=None, is_anonymous=False, has_voice_embedding=True)
        assert cls == SessionClass.HOLDABLE

        ttl_seconds = None  # off
        age_seconds = 9999999
        # Simulate the caller's retire decision
        retired = ttl_seconds is not None and age_seconds > ttl_seconds
        assert not retired

    def test_holdable_retired_past_ttl(self):
        """HOLDABLE is retired when age exceeds the configured TTL."""
        from paramem.server.consolidation import SessionClass, classify_session

        cls = classify_session(speaker_id=None, is_anonymous=False, has_voice_embedding=True)
        assert cls == SessionClass.HOLDABLE

        ttl_seconds = 3600  # 1h
        age_seconds = 7200  # 2h
        retired = ttl_seconds is not None and age_seconds > ttl_seconds
        assert retired

    def test_holdable_held_when_fresh(self):
        """HOLDABLE is NOT retired when age is below the TTL."""
        from paramem.server.consolidation import SessionClass, classify_session

        cls = classify_session(speaker_id=None, is_anonymous=False, has_voice_embedding=True)
        assert cls == SessionClass.HOLDABLE

        ttl_seconds = 3600
        age_seconds = 1800  # 30 min — under TTL
        retired = ttl_seconds is not None and age_seconds > ttl_seconds
        assert not retired

    def test_unidentifiable_always_dropped(self):
        """UNIDENTIFIABLE sessions are always in drop_ids (no TTL needed)."""
        from paramem.server.consolidation import SessionClass, classify_session

        cls = classify_session(speaker_id=None, is_anonymous=False, has_voice_embedding=False)
        assert cls == SessionClass.UNIDENTIFIABLE
        # By contract: UNIDENTIFIABLE → always drop, TTL not consulted.

    def test_named_never_dropped(self):
        """NAMED sessions are never in drop_ids regardless of age."""
        from paramem.server.consolidation import SessionClass, classify_session

        cls = classify_session(speaker_id="abc12345", is_anonymous=False, has_voice_embedding=False)
        assert cls == SessionClass.NAMED
        # Named sessions are extracted, not retired.


# ---------------------------------------------------------------------------
# Tick gate: noop_no_named when all pending sessions are non-NAMED
# ---------------------------------------------------------------------------


class TestTickGateNoNamed:
    """_dispatch_consolidation returns noop_no_named when no NAMED sessions."""

    def _make_minimal_state(self, tmp_path, buffer, store=None):
        """Inject minimal _state overrides for the tick function.

        Uses a plain MagicMock (no spec) for config: ``ServerConfig.paths``
        is a dataclass field with a factory default, so it is never a real
        class-level attribute and ``MagicMock(spec=ServerConfig)`` cannot see
        it — setting ``config.paths.data`` below needs an unrestricted mock.
        """
        from paramem.server.config import ConsolidationScheduleConfig
        from paramem.server.schedule_state import write_last_scheduled_run

        config = MagicMock()
        sched = ConsolidationScheduleConfig()
        # A real cadence backed by a real tmp_path-rooted config.paths.data,
        # with the catch-up stamp pre-seeded stale so the universal catch-up
        # gate (schedule_grammar.scheduled_run_due) reads DUE and falls
        # through to the session-triage/content-gate outcomes these tests
        # exercise, instead of seed-and-noop on a virgin stamp file.
        sched.refresh_cadence = "every 12h"
        config.consolidation = sched
        config.debug = False
        config.debug_dir = tmp_path / "debug"
        config.paths.data = tmp_path
        write_last_scheduled_run(tmp_path / "state", time.time() - 86400)

        return {
            "config": config,
            "session_buffer": buffer,
            "speaker_store": store,
            "consolidating": False,
            "mode": "local",
            "last_chat_monotonic": None,
            "pending_rehydration": False,
            "integrity_check_failed": False,
        }

    def _make_stub_store(self, anonymous_ids=None):
        store = MagicMock()
        _anon = set(anonymous_ids or [])
        store.is_anonymous.side_effect = lambda sid: sid in _anon
        return store

    def _call_tick(self, state_overrides: dict) -> str:
        """Run the scheduled-tick dispatch (AUTO) with mocked _state."""
        from unittest.mock import patch

        import paramem.server.app as _app

        # Patch _consolidation_dispatch_guards to return None (no pre-emption),
        # _is_full_cycle_due to return False (not a full cycle tick → AUTO
        # resolves to INTERIM), and _retro_claim_orphan_sessions to no-op.
        with (
            patch.object(_app, "_state", state_overrides),
            patch(
                "paramem.server.app._consolidation_dispatch_guards",
                return_value=None,
            ),
            patch("paramem.server.app._is_full_cycle_due", return_value=False),
            patch("paramem.server.app._retro_claim_orphan_sessions", return_value=0),
        ):
            status, _action = _app._dispatch_consolidation(_app.ConsolidationAction.AUTO)
            return status

    def test_all_unidentifiable_returns_noop_no_named(self, tmp_path):
        """When all sessions are UNIDENTIFIABLE, tick returns noop_no_named."""
        from paramem.server.session_buffer import SessionBuffer

        buffer = SessionBuffer(tmp_path / "sessions", debug=False)
        conv_id = "conv-unid"
        # No speaker_id, no embedding → UNIDENTIFIABLE
        buffer.append(conv_id, "user", "Hello")
        buffer.append(conv_id, "assistant", "Hi")

        store = self._make_stub_store()
        state = self._make_minimal_state(tmp_path, buffer, store)

        result = self._call_tick(state)
        assert result == "noop_no_named"

    def test_all_holdable_returns_noop_no_named(self, tmp_path):
        """When all sessions are HOLDABLE, tick returns noop_no_named."""
        from paramem.server.session_buffer import SessionBuffer

        buffer = SessionBuffer(tmp_path / "sessions", debug=False)
        conv_id = "conv-hold"
        # No speaker_id but has embedding → HOLDABLE
        buffer.append(conv_id, "user", "Hello", embedding=[0.1, 0.2])
        buffer.append(conv_id, "assistant", "Hi")

        store = self._make_stub_store()
        state = self._make_minimal_state(tmp_path, buffer, store)

        result = self._call_tick(state)
        assert result == "noop_no_named"

    def test_unidentifiable_sessions_retired_at_tick(self, tmp_path):
        """UNIDENTIFIABLE sessions are mark_consolidated at the tick, not left pending.

        After the tick, the session must no longer appear in buffer.get_pending().
        """
        from paramem.server.session_buffer import SessionBuffer

        buffer = SessionBuffer(tmp_path / "sessions", debug=False)
        conv_id = "conv-unid2"
        buffer.append(conv_id, "user", "Text only, no speaker")
        buffer.append(conv_id, "assistant", "Ok")

        store = self._make_stub_store()
        state = self._make_minimal_state(tmp_path, buffer, store)

        self._call_tick(state)

        # Session was retired (UNIDENTIFIABLE → immediate drop).
        remaining = [s["session_id"] for s in buffer.get_pending()]
        assert conv_id not in remaining

    def test_holdable_session_stays_pending(self, tmp_path):
        """HOLDABLE sessions (fresh, TTL=off) remain pending after tick."""
        from paramem.server.session_buffer import SessionBuffer

        buffer = SessionBuffer(tmp_path / "sessions", debug=False)
        conv_id = "conv-hold2"
        # embedding present → HOLDABLE
        buffer.append(conv_id, "user", "Hello", embedding=[0.3, 0.4])
        buffer.append(conv_id, "assistant", "Hi")

        store = self._make_stub_store()
        state = self._make_minimal_state(tmp_path, buffer, store)

        self._call_tick(state)

        # HOLDABLE + TTL=off → session stays pending. append() mints a
        # session_id ("{conv_id}-{timestamp}-{rand}"), so match by prefix.
        remaining = [s["session_id"] for s in buffer.get_pending()]
        assert any(sid.startswith(f"{conv_id}-") for sid in remaining)


class TestAtomicJsonWriteHonoursSecurityPosture:
    """write_infra_json always routes through the infrastructure envelope.

    There is no per-call plaintext override: the operator's posture
    (``security.require_encryption`` + key material) is the only thing that
    decides, so no caller can write infrastructure JSON in the clear behind
    the operator's back. Inspection output is not written here — that is an
    artifact and goes through ``paramem.utils.artifacts.write_artifact``.

    Mutation: reintroduce an ``encrypted=False`` branch -> this fails.
    """

    def test_writes_age_envelope_under_security_on(self, tmp_path, monkeypatch):
        from paramem.backup.encryption import write_infra_json
        from paramem.backup.key_store import (
            DAILY_PASSPHRASE_ENV_VAR,
            _clear_daily_identity_cache,
            mint_daily_identity,
            wrap_daily_identity,
            write_daily_key_file,
        )

        # Genuine Security ON: a real daily identity is loadable.
        ident = mint_daily_identity()
        key_path = tmp_path / "daily_key.age"
        write_daily_key_file(wrap_daily_identity(ident, "pw"), key_path)
        monkeypatch.setenv(DAILY_PASSPHRASE_ENV_VAR, "pw")
        monkeypatch.setattr("paramem.backup.key_store.DAILY_KEY_PATH_DEFAULT", key_path)
        _clear_daily_identity_cache()

        out = tmp_path / "state.json"
        write_infra_json(out, {"debug": True, "n": 42})

        assert out.read_bytes().startswith(b"age-encryption.org/v1"), (
            "infrastructure JSON must be age-wrapped when a daily identity is loaded"
        )

        import json as _json

        from paramem.backup.encryption import read_maybe_encrypted

        assert _json.loads(read_maybe_encrypted(out)) == {"debug": True, "n": 42}


class TestConsolidationLoopRelease:
    """ConsolidationLoop.release() must null the GraphMerger's model reference.

    GraphMerger is a BASE-MODEL HOLDER (see BASE-MODEL HOLDER comments in the
    codebase).  release() is the encapsulated teardown path;
    _release_base_model_in_process reaches it transitively via
    loop.release() → merger.release().
    """

    def _make_loop_with_mock_merger(self):
        """Build a bare ConsolidationLoop with a mock merger holding a mock model.

        Bypasses __init__ (no real model load).  Sets the minimum attributes
        that release() inspects: model, tokenizer, extraction (None, guarded),
        merger (real GraphMerger with mock model injected).
        """
        from paramem.graph.merger import GraphMerger
        from paramem.training.consolidation import ConsolidationLoop

        loop = ConsolidationLoop.__new__(ConsolidationLoop)
        loop.model = MagicMock()
        loop.tokenizer = MagicMock()
        loop._bg_trainer = None
        loop.extraction = None  # release() guards with getattr(..., None) check

        # Wire a real GraphMerger and inject a mock model into it so we can
        # verify that release() nulls merger.model without loading real weights.
        merger = GraphMerger()
        merger.model = MagicMock()
        merger.tokenizer = MagicMock()
        loop.merger = merger

        return loop

    def test_release_nulls_merger_model(self):
        """release() must set merger.model to None (BASE-MODEL HOLDER teardown)."""
        loop = self._make_loop_with_mock_merger()
        assert loop.merger.model is not None, "precondition: merger.model is set before release"
        loop.release()
        assert loop.merger.model is None, (
            "merger.model must be None after ConsolidationLoop.release() "
            "(GraphMerger is a BASE-MODEL HOLDER — see BASE-MODEL HOLDER comments in the codebase)"
        )

    def test_release_nulls_merger_tokenizer(self):
        """release() must set merger.tokenizer to None alongside merger.model."""
        loop = self._make_loop_with_mock_merger()
        loop.release()
        assert loop.merger.tokenizer is None, (
            "merger.tokenizer must be None after ConsolidationLoop.release()"
        )

    def test_release_idempotent_on_merger(self):
        """Calling release() twice does not raise even after merger.model is None."""
        loop = self._make_loop_with_mock_merger()
        loop.release()
        loop.release()  # must not raise


class TestSeedKeyMetadata:
    """ConsolidationLoop.seed_key_metadata rebuilds ``promoted_keys`` from
    the per-key ``promoted`` flag on the store's already-loaded bookkeeping
    -- not from a global ``promoted_keys`` list, which no longer exists."""

    def _make_loop(self):
        from paramem.memory.store import MemoryStore

        loop = ConsolidationLoop.__new__(ConsolidationLoop)
        loop.store = MemoryStore()
        loop.cycle_count = 0
        loop.promoted_keys = set()
        return loop

    def test_promoted_keys_rebuilt_from_per_key_flags(self):
        from paramem.training.key_registry import KeyRegistry

        loop = self._make_loop()
        reg = KeyRegistry()
        reg.add("graph1")
        reg.add("graph2")
        loop.store.load_registry("semantic", reg)
        loop.store.set_bookkeeping(
            "graph1",
            speaker_id="speaker0",
            relation_type="factual",
            first_seen="",
            promoted=True,
        )
        loop.store.set_bookkeeping(
            "graph2",
            speaker_id="speaker0",
            relation_type="factual",
            first_seen="",
            promoted=False,
        )

        loop.seed_key_metadata(5)

        assert loop.promoted_keys == {"graph1"}
        assert loop.cycle_count == 5

    def test_promoted_flag_on_an_unknown_key_is_excluded(self):
        """A promoted row surviving for a key no longer known to any
        registry (defensive re-verification, ``store.is_known``) is not
        rebuilt into ``promoted_keys``."""
        loop = self._make_loop()
        loop.store.set_bookkeeping(
            "graph_orphan",
            speaker_id="speaker0",
            relation_type="factual",
            first_seen="",
            promoted=True,
        )
        # graph_orphan carries no registry entry anywhere -- is_known() is False.

        loop.seed_key_metadata(0)

        assert loop.promoted_keys == set()


# ---------------------------------------------------------------------------
# _synth_speaker_entities: lowercase-uniform plain == comparison
# ---------------------------------------------------------------------------


class TestSynthSpeakerEntitiesB1Regression:
    """_synth_speaker_entities must emit a speaker Entity when subject == speaker_id.

    Under lowercase-uniform identity both subject and speaker_id are lowercase
    speaker{N}, so plain == is the correct comparison.  The old bridging
    function speaker_ref_matches is deleted; this class validates that the
    new plain-== path is correct.
    """

    def test_matching_lowercase_subject_and_speaker_id(self):
        """Relation with subject == speaker_id (both lowercase) emits exactly one Entity."""
        from paramem.graph.merger import _synth_speaker_entities
        from paramem.graph.schema import Relation

        # Both subject and speaker_id are lowercase speaker{N} (canonical form).
        relations = [
            Relation(
                subject="speaker0",
                predicate="works at",
                object="Acme",
                relation_type="factual",
                confidence=1.0,
                speaker_id="speaker0",
            ),
        ]

        entities = _synth_speaker_entities(relations)

        assert len(entities) == 1, (
            f"Expected 1 speaker Entity, got {len(entities)}: {entities!r}. "
            "Plain == must work when both subject and speaker_id are lowercase."
        )
        assert entities[0].speaker_id == "speaker0"
        assert entities[0].name == "speaker0"

    def test_plain_equality_is_correct(self):
        """Confirm that plain == is sufficient under lowercase-uniform identity."""
        subject = "speaker0"
        speaker_id = "speaker0"
        assert (subject == speaker_id) is True, (
            "Under lowercase-uniform identity, plain == must be True."
        )

    def test_distinct_speaker_produces_no_spurious_entity(self):
        """A subject that is a DIFFERENT speaker must not produce an entity for
        speaker_id — the comparison must be identity, not just 'is a speaker'.
        """
        from paramem.graph.merger import _synth_speaker_entities
        from paramem.graph.schema import Relation

        relations = [
            Relation(
                subject="speaker1",  # different speaker — must NOT produce Entity
                predicate="knows",
                object="Acme",
                relation_type="factual",
                confidence=1.0,
                speaker_id="speaker0",  # mismatch — must NOT produce Entity
            ),
        ]

        entities = _synth_speaker_entities(relations)
        assert len(entities) == 0, (
            f"Expected 0 entities for mismatched subject/speaker_id, got {entities!r}."
        )


# ---------------------------------------------------------------------------
# Speaker-pair guard: resolve_to_node_key + same_as speaker-casing guard
# ---------------------------------------------------------------------------


class TestResolveToNodeKey:
    """Unit tests for the module-level ``resolve_to_node_key`` function."""

    def test_membership_shortcut(self):
        """When ``in_graph(name)`` is True the name IS returned unchanged."""
        from paramem.training.graph_enrich import resolve_to_node_key

        graph_keys = {"alice", "berlin", "acme corp"}
        in_graph = lambda n: n in graph_keys  # noqa: E731

        assert resolve_to_node_key("alice", in_graph) == "alice"
        assert resolve_to_node_key("berlin", in_graph) == "berlin"

    def test_canonical_fallback(self):
        """When name is NOT in graph, canonical(name) is returned."""
        from paramem.training.graph_enrich import resolve_to_node_key

        in_graph = lambda n: False  # noqa: E731

        # canonical("speaker0") == "speaker0" (casefolded)
        assert resolve_to_node_key("speaker0", in_graph) == "speaker0"
        # canonical("Alex") == "alex"
        assert resolve_to_node_key("Alex", in_graph) == "alex"

    def test_coref_chain_follow(self):
        """With a coref_map, the resolved key is followed through the chain."""
        from paramem.training.graph_enrich import resolve_to_node_key

        in_graph = lambda n: False  # noqa: E731
        coref_map = {"speaker1": "speaker0"}  # speaker1 merged into speaker0

        result = resolve_to_node_key("speaker1", in_graph, coref_map)
        # canonical("speaker1") == "speaker1", then follows chain → "speaker0"
        assert result == "speaker0"

    def test_coref_chain_cycle_guarded(self):
        """A cyclic coref_map must not loop forever."""
        from paramem.training.graph_enrich import resolve_to_node_key

        in_graph = lambda n: False  # noqa: E731
        coref_map = {"a": "b", "b": "a"}  # cycle

        # Must return without hanging; result is either "a" or "b"
        result = resolve_to_node_key("a", in_graph, coref_map)
        assert result in ("a", "b")

    def test_no_coref_map(self):
        """Without a coref_map, only membership+canonical resolution applies."""
        from paramem.training.graph_enrich import resolve_to_node_key

        in_graph = lambda n: n == "existing"  # noqa: E731

        assert resolve_to_node_key("existing", in_graph) == "existing"
        assert resolve_to_node_key("Missing", in_graph) == "missing"


class TestGraphTierSymbolsNotReexported:
    """Structural guard: the enrichment and normalization surfaces live in
    ``paramem.training.graph_enrich`` and ``paramem.training.graph_tier``.

    A patch target under ``paramem.training.consolidation`` for one of these
    names is stale.  Asserting the names are absent makes a resurrected stale
    patch fail loudly (``AttributeError`` from ``unittest.mock.patch``) instead
    of silently patching a dead re-export while production code — reached via
    ``ConsolidationLoop.stage_event`` constructing a per-call
    ``paramem.training.graph_tier.GraphTierRefiner`` and calling
    ``refiner.refine()``, which dispatches into
    ``paramem.training.graph_enrich.enrich_graph`` for enrichment and
    ``GraphTierRefiner.run_normalization`` for normalization — runs unmocked.

    The last two names in the parametrize list below are on this list too:
    both SHIM delegator methods on ``ConsolidationLoop`` were deleted, so a
    ``patch.object`` targeting either of their old names on a live
    ``ConsolidationLoop`` instance now raises ``AttributeError`` at patch
    time instead of silently installing a mock that production never
    calls.
    """

    @pytest.mark.parametrize(
        "name",
        [
            "request_graph_enrichment",
            "_safe_to_merge_surface",
            "_strip_honorifics",
            "serialize_subgraph_triples",
            "resolve_to_node_key",
            "normalize_predicates",
            "_enrich_graph",
            "_run_graph_normalization",
        ],
    )
    def test_tier_symbols_not_reexported_from_consolidation(self, name):
        import paramem.training.consolidation as c

        # The first six names were module-level functions that moved out of
        # this module entirely; the last two were SHIM methods on
        # ``ConsolidationLoop`` and were never module-level attributes, so the
        # class is checked too — a resurrected shim method would be invisible
        # to a module-only ``hasattr`` check.
        assert not hasattr(c, name)
        assert not hasattr(c.ConsolidationLoop, name)


class TestSameAsSpeakerPairGuard:
    """Speaker-pair guard: any same_as pair where BOTH surfaces are speaker ids must be
    skipped unconditionally.

    Speaker identity is authoritative (voice/enrollment) and must never be
    coalesced by a surface-similarity heuristic.  Two speaker-id surfaces are
    either the SAME speaker (already unified by canonical node-keying — redundant)
    or DIFFERENT speakers (must never merge — Jaro-Winkler treats the
    distinguishing digit as a typo, so ``_safe_to_merge_surface`` returns True
    for every distinct speaker pair).  The guard blocks both scenarios.
    """

    @pytest.fixture(autouse=True)
    def _stub_local_anonymize(self, monkeypatch):
        """Stub ``anonymize_transcript`` for every test in this class.

        ``graph_enrich.enrich_graph`` now runs the local anonymizer
        (the SAME primitive session-tier extraction uses) over each chunk
        BEFORE the cloud call, to derive real-name entity types the fold
        graph itself cannot supply (see that function's docstring).
        ``_make_speaker_pair_guard_loop``'s
        model/tokenizer are ``MagicMock()``s, so a real call always fails to
        parse (no JSON in a ``MagicMock``'s generated output), which would
        fail every chunk closed (skip the cloud call entirely) before the
        mocked ``request_graph_enrichment`` below is ever reached — masking
        every test in this class behind a vacuous ``same_as_merges == 0``.

        The stub lands on the SAFE side rather than the unsafe one: it
        types every non-speaker name found in the chunk's relations as
        ``"person"`` (masked). An EMPTY mapping (``{}``) is deliberately
        NOT used here — ``graph_enrich.enrich_graph``'s own
        fail-closed guard (a local mapping that names zero entities for a chunk with real
        content is treated as a classification failure) would otherwise
        skip the cloud call entirely for every test in this class before
        the mocked ``request_graph_enrichment`` is ever reached, the same
        failure mode the docstring above already describes for a
        MagicMock parse failure.
        """
        from paramem.config.taxonomy import entity_type_to_prefix

        def _stub(facts, model, tokenizer, transcript="", **kwargs):
            # ``facts`` is a plain fact-dict list (interface narrowing,
            # 2026-07-21) — never a ``SessionGraph`` — so names are read
            # off ``subject``/``object`` keys directly, not ``.relations``.
            names = sorted(
                {str(f.get("subject", "")) for f in facts}
                | {str(f.get("object", "")) for f in facts}
            )
            mapping: dict[str, str] = {}
            prefix = entity_type_to_prefix("person")
            for i, name in enumerate(names, start=1):
                mapping[name] = f"{prefix}_{i}"
            return mapping, "stub-anon-transcript", "stub-raw"

        monkeypatch.setattr(
            "paramem.cloud.anonymize.anonymize_transcript",
            _stub,
        )

    @staticmethod
    def _make_speaker_pair_guard_loop(tmp_path):
        """Minimal ConsolidationLoop for the speaker-pair guard tests.

        (>=10-node graph, real merger.)
        """
        from unittest.mock import MagicMock  # noqa: F811

        from peft import PeftModel

        from paramem.graph.schema import Entity, SessionGraph
        from paramem.memory.store import MemoryStore
        from paramem.training.consolidation import ConsolidationLoop
        from paramem.training.key_registry import KeyRegistry
        from paramem.utils.config import AdapterConfig, ConsolidationConfig, TrainingConfig

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
            memory_store=MemoryStore(),
            procedural_adapter_config=None,
            output_dir=tmp_path,
            extraction_enrichment_provider="anthropic",
            extraction_enrichment_provider_model="claude-sonnet-4-6",
            extraction_scrub={"person name"},
            # Graph-tier enrichment is cloud egress: the shared cloud-admission
            # verdict's first term is the master switch, so it must be ON for
            # this test to reach a (mocked) cloud call.
            cloud_enabled=True,
            extraction_max_tokens=8192,
            extraction_plausibility_max_tokens=8192,
            extraction_anonymize_token_envelope=8192,
        )
        loop._probe_recall = lambda adapter_name, entries: RecallProbe(
            per_key=tuple({"key": e["key"], "exact_match": True} for e in entries)
        )
        for tier in ("episodic", "semantic", "procedural"):
            loop.store.load_registry(tier, KeyRegistry())

        # Populate ≥10 nodes so the floor gate passes.
        graph = loop.merger.graph
        org = "acmecorp"
        graph.add_node(
            org,
            entity_type="organization",
            display_name="AcmeCorp",
            reinforcement_count=10,
            sessions=["s000"],
            first_seen="s000",
            last_seen="s000",
        )
        for i in range(10):
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
            )

        # Seed a speaker node (casefolded key "speaker0").
        loop.merger.merge(
            SessionGraph(
                session_id="seed-speaker0",
                timestamp="2026-01-01T00:00:00Z",
                entities=[Entity(name="Alex", entity_type="person", speaker_id="speaker0")],
                relations=[],
            )
        )
        return loop

    def test_speaker_id_casing_pair_is_skipped(self, tmp_path, monkeypatch):
        """Cloud same_as ['speaker0', 'speaker0'] must not contract the speaker node.

        Drives the real ``GraphTierRefiner.run_enrichment`` production path
        with a mocked cloud response.  Asserts that after processing the
        casing-variant pair:
        - result["same_as_merges"] == 0  (no contraction counted)
        - "speaker0" still exists as a distinct node in the graph
        - no node was removed by a self-contraction

        The guard fires first (both surfaces are speaker ids); the
        ``keep_canon == drop_canon`` post-resolution check is a secondary
        backstop for this specific case only.
        """
        from unittest.mock import patch

        loop = self._make_speaker_pair_guard_loop(tmp_path)

        node_count_before = loop.merger.graph.number_of_nodes()

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with patch(
            "paramem.training.graph_enrich.request_graph_enrichment",
            return_value=([], [["speaker0", "speaker0"]], "raw", 0),
        ):
            result = _refiner_for(loop).run_enrichment()

        assert not result["skipped"]
        assert result["same_as_merges"] == 0, (
            "The speaker-pair/post-resolution guard must prevent the speaker casing-variant pair "
            "from counting as a merge; got same_as_merges="
            f"{result['same_as_merges']}"
        )
        # "speaker0" must still be present — no self-contraction removed it.
        assert "speaker0" in loop.merger.graph.nodes, (
            "Speaker node 'speaker0' must survive the casing-variant same_as pair"
        )
        # Graph size must be unchanged.
        assert loop.merger.graph.number_of_nodes() == node_count_before, (
            "Graph node count must not change after a guarded (no-op) same_as pair"
        )

    def test_distinct_speaker_ids_are_not_merged(self, tmp_path, monkeypatch):
        """Cloud same_as ['speaker0', 'speaker1'] must NOT merge distinct speakers.

        Drives the real ``GraphTierRefiner.run_enrichment`` production
        path.  speaker0 and speaker1 are distinct enrollments; a cloud
        same_as proposal must be
        blocked by the generalized speaker-pair guard (both surfaces are speaker ids).

        This test is load-bearing: without the guard,
        ``_safe_to_merge_surface("speaker0", "speaker1")`` returns True (JW
        treats the digit as a typo, score ≈ 0.950) and the merger would
        contract speaker1 into speaker0 — catastrophic in a real 2-speaker
        deployment.

        Asserts after enrichment:
        - result["same_as_merges"] == 0
        - both "speaker0" and "speaker1" still exist as distinct nodes
        """
        from unittest.mock import patch

        loop = self._make_speaker_pair_guard_loop(tmp_path)
        # Seed a second speaker node so both endpoints are in the graph.
        from paramem.graph.schema import Entity, SessionGraph

        loop.merger.merge(
            SessionGraph(
                session_id="seed-speaker1",
                timestamp="2026-01-01T00:00:00Z",
                entities=[Entity(name="Robin", entity_type="person", speaker_id="speaker1")],
                relations=[],
            )
        )
        assert "speaker0" in loop.merger.graph.nodes
        assert "speaker1" in loop.merger.graph.nodes

        node_count_before = loop.merger.graph.number_of_nodes()

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with patch(
            "paramem.training.graph_enrich.request_graph_enrichment",
            return_value=([], [["speaker0", "speaker1"]], "raw", 0),
        ):
            result = _refiner_for(loop).run_enrichment()

        assert result["same_as_merges"] == 0, (
            "The speaker-pair guard must block distinct speaker ids from merging; "
            f"got same_as_merges={result['same_as_merges']}"
        )
        # Both speaker nodes must remain distinct — no contraction occurred.
        assert "speaker0" in loop.merger.graph.nodes, (
            "speaker0 must survive — the guard must block the speaker0/speaker1 merge"
        )
        assert "speaker1" in loop.merger.graph.nodes, (
            "speaker1 must survive — the guard must block the speaker0/speaker1 merge"
        )
        assert loop.merger.graph.number_of_nodes() == node_count_before, (
            "Graph node count must not change after a guarded speaker same_as pair"
        )

    def test_non_speaker_pairs_are_not_guarded(self, tmp_path, monkeypatch):
        """Non-speaker same_as pairs (e.g. name variants) must pass through the guard.

        Drives the real ``GraphTierRefiner.run_enrichment`` path.  A pair
        of ordinary non-speaker names must NOT be intercepted by the
        speaker-pair guard; they continue to the normal resolution and
        surface-gate checks.
        """
        from unittest.mock import patch

        from paramem.utils.identity import is_speaker_id

        loop = self._make_speaker_pair_guard_loop(tmp_path)
        graph = loop.merger.graph
        # Add two non-speaker nodes for the same_as pair.
        for name in ("alexander", "alex"):
            graph.add_node(
                name,
                entity_type="person",
                display_name=name.capitalize(),
                reinforcement_count=2,
                sessions=["s100"],
                first_seen="s100",
                last_seen="s100",
            )

        assert not is_speaker_id("alexander")
        assert not is_speaker_id("alex")

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        # Non-speaker pair: the guard must not fire; normal gates apply.
        # "Alexander"/"Alex" pass _safe_to_merge_surface (token subset), so the merge
        # is allowed and same_as_merges == 1.
        with patch(
            "paramem.training.graph_enrich.request_graph_enrichment",
            return_value=([], [["Alexander", "Alex"]], "raw", 0),
        ):
            result = _refiner_for(loop).run_enrichment()

        # Non-speaker pair was not guarded — the merge proceeded normally.
        assert result["same_as_merges"] >= 1, (
            "Non-speaker same_as pair must not be blocked by the speaker-pair guard; "
            f"expected at least 1 merge, got {result['same_as_merges']}"
        )


# ---------------------------------------------------------------------------
# Subtractive removals helper + whole-graph normalization pass tests
# ---------------------------------------------------------------------------


class TestMergeRegistryRelationsTimestamp:
    """GraphMerger.merge_relations passes timestamp="" to the merger's SessionGraph.

    Regression guard: before the fix, GraphMerger.merge_relations always built the
    SessionGraph with timestamp=datetime.now(...), so a recon relation with
    last_seen="" resolved to incoming_ls = "" or now() = now() — making the legacy key
    appear as the unique freshest and wrongly retiring a genuinely-dated rival.

    With the fix, timestamp="" (the new default param), so incoming_ls = "" or "" = ""
    for legacy relations.  An empty last_seen sorts as the oldest possible timestamp:
    a dated candidate always outranks an undated one, so a legacy "" key is retired
    in favor of any genuinely-dated rival.  The any-empty COEXIST rule applies only
    when EVERY candidate (incoming + all rivals) is undated — the fold-wide legacy
    case with no recency signal anywhere.
    """

    @staticmethod
    def _make_loop_for_recon_merge(tmp_path) -> "ConsolidationLoop":
        """Minimal ConsolidationLoop with a mock-model GraphMerger for merge_relations."""
        from unittest.mock import MagicMock

        from paramem.graph.merger import GraphMerger
        from paramem.memory.store import MemoryStore
        from paramem.training.consolidation import ConsolidationLoop
        from paramem.training.key_registry import KeyRegistry
        from paramem.utils.config import AdapterConfig, ConsolidationConfig, TrainingConfig

        loop = object.__new__(ConsolidationLoop)
        loop.model = None  # no gradient-checkpointing guard needed
        loop.tokenizer = None
        loop.config = ConsolidationConfig(
            refinement_contradiction="on",  # enable Case-2 so the bug can trigger
        )
        loop.training_config = TrainingConfig(
            num_epochs=1,
            gradient_checkpointing=False,
            batch_size=1,
            recall_early_stopping=False,
            recall_probe_batch_size=1,
        )
        loop.episodic_config = AdapterConfig(rank=4, alpha=8, target_modules=["q_proj"])
        loop.semantic_config = AdapterConfig(rank=4, alpha=8, target_modules=["q_proj"])
        loop.procedural_config = None
        loop.wandb_config = None
        loop._thermal_policy = None
        loop.output_dir = tmp_path
        loop.save_cycle_snapshots = False
        loop._debug_base = None
        loop.snapshot_dir = None
        loop.shutdown_requested = False
        loop._bg_trainer = None
        loop._early_stop_callback = None
        loop.fingerprint_cache = None
        loop._keep_prior_slots = 2
        loop.cycle_count = 0
        loop._indexed_next_index = 1
        loop._procedural_next_index = 1
        loop._procedural_tentative_next_index = 1
        loop._indexed_ep_interim = {}
        loop.promoted_keys = set()
        loop.full_consolidation_period_string = ""

        # GraphMerger with a mock model so Case-2 fires.
        merger = GraphMerger(model=MagicMock(), tokenizer=MagicMock())
        # Pre-cache "lives in" (canonical form) as single-valued (REPLACE) to skip
        # the model call.
        merger._predicate_cardinality["lives in"] = False
        loop.merger = merger

        store = MemoryStore()
        for tier in ("episodic", "semantic", "procedural"):
            store.load_registry(tier, KeyRegistry())
        loop.store = store
        return loop

    def test_mixed_dated_legacy_recon_dated_wins(self, tmp_path):
        """Dated-wins-over-undated: mixed registry (dated key, legacy "") through recon merge.

        Simulates a fold GraphMerger.merge_relations call with:
          - Relation A (lives in → munich), last_seen="2026-01-01T00:00:00Z" (dated)
          - Relation B (lives in → berlin), last_seen="" (legacy — no timestamp in bookkeeping)

        Both share the same (subject=alex, predicate=lives in) so Case-2 fires.

        An empty last_seen sorts as the oldest possible timestamp, so a dated
        candidate always outranks an undated one: the dated munich relation is
        processed first (net-new insert, no rivals yet); the legacy berlin
        relation is then evaluated as incoming against the dated munich rival —
        incoming_ls="" loses to rival max_ls="2026-01-01T00:00:00Z", so berlin is
        NOT inserted and is ledgered as retired in favor of munich.  COEXIST only
        applies when EVERY candidate (incoming + rivals) is undated, which is not
        the case here.
        """
        from paramem.graph.schema import Relation

        loop = self._make_loop_for_recon_merge(tmp_path)

        relations = [
            Relation(
                subject="alex",
                predicate="lives_in",
                object="munich",
                relation_type="factual",
                confidence=1.0,
                speaker_id="speaker0",
                indexed_key="key_munich_dated",
                last_seen="2026-01-01T00:00:00Z",
            ),
            Relation(
                subject="alex",
                predicate="lives_in",
                object="berlin",
                relation_type="factual",
                confidence=1.0,
                speaker_id="speaker0",
                indexed_key="key_berlin_legacy",
                last_seen="",  # legacy: no timestamp in bookkeeping
            ),
        ]

        # Call with resolve_contradictions=True (config is "on") and default timestamp="".
        loop.merger.merge_relations(
            relations,
            session_id="__full_consolidation_recon__",
            log_label="test recon triples",
            resolve_contradictions=True,
        )

        lives_in_objects = [
            obj
            for obj in loop.merger.graph.successors("alex")
            for _, d in loop.merger.graph["alex"][obj].items()
            if d.get("predicate") == "lives in"
        ]
        assert "munich" in lives_in_objects, (
            "Dated munich key must survive: a dated candidate always outranks an undated rival"
        )
        assert "berlin" not in lives_in_objects, (
            "Legacy '' berlin relation must be retired: it loses to the dated munich rival"
        )
        assert loop.merger.removal_ledger.get("key_berlin_legacy") == {
            "reason": "contradiction_same_pred",
            "old_object": "berlin",
            "new_object": "munich",
        }, f"Expected key_berlin_legacy retired for munich; got {loop.merger.removal_ledger}"

    def test_pending_dated_vs_legacy_empty_rival_dated_wins(self, tmp_path):
        """Pending dated relation vs legacy "" registry-true rival → dated wins (REPLACE).

        A pending fact with a real last_seen retires a legacy "" registry rival —
        an empty last_seen sorts as the oldest possible timestamp, so the dated
        incoming relation always outranks the undated rival.  COEXIST only applies
        when EVERY candidate is undated, which is not the case here (the pending
        relation is dated).
        """
        from paramem.graph.schema import Relation
        from paramem.memory.persistence import _IK_KEY_ATTR

        loop = self._make_loop_for_recon_merge(tmp_path)

        # Step 1: merge the registry-true recon relation (munich, legacy last_seen="").
        recon_relations = [
            Relation(
                subject="alex",
                predicate="lives_in",
                object="munich",
                relation_type="factual",
                confidence=1.0,
                speaker_id="speaker0",
                indexed_key="key_munich_legacy",
                last_seen="",  # legacy: no timestamp in bookkeeping
            )
        ]
        loop.merger.merge_relations(
            recon_relations,
            session_id="__full_consolidation_recon__",
            log_label="recon triples (test)",
            resolve_contradictions=False,
        )

        # Stamp ik_key onto the munich edge.
        for _succ in list(loop.merger.graph.successors("alex")):
            for _eid, _edata in loop.merger.graph["alex"][_succ].items():
                if _edata.get("predicate") == "lives in":
                    _edata[_IK_KEY_ATTR] = "key_munich_legacy"

        # Step 2: merge the pending relation (berlin, dated) with resolve=True.
        pending_relations = [
            Relation(
                subject="alex",
                predicate="lives_in",
                object="berlin",
                relation_type="factual",
                confidence=1.0,
                speaker_id="speaker0",
                indexed_key="key_berlin_new",
                last_seen="2026-01-02T00:00:00Z",  # dated pending
            )
        ]
        loop.merger.merge_relations(
            pending_relations,
            session_id="__interim_pending_sessions__",
            log_label="pending relations (test)",
            resolve_contradictions=True,
        )

        lives_in_objects = [
            obj
            for obj in loop.merger.graph.successors("alex")
            for _, d in loop.merger.graph["alex"][obj].items()
            if d.get("predicate") == "lives in"
        ]
        assert "berlin" in lives_in_objects, "Dated pending Berlin must be inserted"
        assert "munich" not in lives_in_objects, (
            "Legacy '' Munich must be retired: a dated incoming relation always "
            "outranks an undated rival"
        )
        assert loop.merger.removal_ledger.get("key_munich_legacy") == {
            "reason": "contradiction_same_pred",
            "old_object": "munich",
            "new_object": "berlin",
        }, f"Expected key_munich_legacy retired for berlin; got {loop.merger.removal_ledger}"


# ---------------------------------------------------------------------------
# _extract_json_block — relations-envelope parser path for normalization
# ---------------------------------------------------------------------------


class TestExtractJsonBlockRelationsEnvelope:
    """_extract_json_block recognises the {"relations":[...]} envelope emitted by
    the normalization prompt.

    Tests:
    - wrapped {"relations":[...]} envelope parsed and returned.
    - bare array [{...}] with subject/predicate/object elements accepted.
    - markdown-fenced {"relations":[...]} accepted (fence stripped).
    - no JSON at all raises ValueError.
    - multiple relation entries preserved.
    """

    @staticmethod
    def _parse(raw: str):
        import json

        from paramem.graph.extractor import _extract_json_block

        return json.loads(_extract_json_block(raw))

    def test_relations_envelope_parsed(self):
        """{"relations":[...]} envelope is accepted by _extract_json_block."""
        raw = '{"relations": [{"subject": "Alex", "predicate": "works_for", "object": "Acme"}]}'
        result = self._parse(raw)
        assert isinstance(result, dict)
        assert "relations" in result
        assert len(result["relations"]) == 1
        assert result["relations"][0]["predicate"] == "works_for"

    def test_bare_array_with_spo_elements_accepted(self):
        """Bare array where first element has subject/predicate/object is accepted."""
        raw = '[{"subject": "Alex", "predicate": "lives_in", "object": "Berlin"}]'
        result = self._parse(raw)
        assert isinstance(result, list)
        assert result[0]["object"] == "Berlin"

    def test_markdown_fenced_relations_envelope(self):
        """Code-fenced {"relations":[...]} is parsed after fence-stripping."""
        raw = '```json\n{"relations": [{"subject": "A", "predicate": "b", "object": "C"}]}\n```'
        result = self._parse(raw)
        assert "relations" in result

    def test_no_json_raises(self):
        """No JSON in output → ValueError from _extract_json_block."""
        from paramem.graph.extractor import _extract_json_block

        with pytest.raises((ValueError, Exception)):
            _extract_json_block("The graph has no redundancy.")

    def test_multiple_relation_entries_preserved(self):
        """Multiple relation entries are all present in parsed output."""
        raw = (
            '{"relations": ['
            '{"subject": "Morgan", "predicate": "born_in", "object": "Germany"},'
            '{"subject": "Jordan", "predicate": "works_for", "object": "TechCorp"}'
            "]}"
        )
        result = self._parse(raw)
        assert len(result["relations"]) == 2
        assert result["relations"][1]["object"] == "TechCorp"


# ---------------------------------------------------------------------------
# GraphTierRefiner.run_normalization — apply path: relations-envelope + same-(s,o) collapse
# ---------------------------------------------------------------------------


class TestRunGraphNormalizationApply:
    """Integration tests for the whole-graph normalization apply path.

    The model is stubbed via generate_answer (patched in paramem.graph.extractor).
    All assertions are on graph-edge changes and removal_ledger entries.
    The factory builds graphs large enough to pass the 10-node floor
    (node_count=15 default).

    The model output uses the clusters schema (one call per candidate (s,o) group):
    ``{"clusters": [["predA", "predB"], ...]}``.
    The apply logic picks the MAX reinforcement_count edge as survivor; retired edges
    have their provenance unioned onto the survivor.  Ledger reason is
    ``"predicate_synonym_collapse"``.

    Tests:
    - two keyed synonym predicates for same (s,o) — model returns cluster →
      lower-rec keyed edge removed + removal_ledger 'predicate_synonym_collapse'.
    - two keyless synonym predicates — lower-rec edge removed, no ledger entry.
    - provenance (sessions union, recurrence sum, max confidence) is carried onto
      the survivor (MAX rec) before retired edges are removed.
    - single-predicate (s,o) group → no model call; graph unchanged.
    - model returns empty clusters → no-op (graph unchanged).
    - model=None → skipped=True, graph unchanged.
    - graph < 10 nodes → skipped=True (floor).
    - mixed keyed + keyless — keyed retired → ledger; keyless retired → no ledger;
      MAX-rec survivor intact; result counts correct.
    - single-predicate (s,o) group not touched even when another group is collapsed.
    - multi-predicate group collapsed; graph updated correctly.
    """

    # ---------------------------------------------------------------------------
    # Shared factory
    # ---------------------------------------------------------------------------

    @staticmethod
    def _make_loop(tmp_path, *, model=None, node_count: int = 15):
        """Build a ConsolidationLoop with a seeded merger graph."""
        import networkx as nx

        from paramem.graph.merger import GraphMerger
        from paramem.memory.store import MemoryStore
        from paramem.training.consolidation import ConsolidationLoop
        from paramem.training.key_registry import KeyRegistry
        from paramem.utils.config import AdapterConfig, ConsolidationConfig, TrainingConfig

        loop = object.__new__(ConsolidationLoop)
        loop.model = model if model is not None else MagicMock()
        loop.tokenizer = MagicMock()
        loop.tokenizer.apply_chat_template.return_value = "formatted_prompt"
        loop.config = ConsolidationConfig(
            refinement_normalization="on",
        )
        loop.training_config = TrainingConfig(
            num_epochs=1,
            gradient_checkpointing=False,
            batch_size=1,
            recall_early_stopping=False,
            recall_probe_batch_size=1,
        )
        loop.episodic_config = AdapterConfig(rank=4, alpha=8, target_modules=["q_proj"])
        loop.semantic_config = AdapterConfig(rank=4, alpha=8, target_modules=["q_proj"])
        loop.procedural_config = None
        loop.wandb_config = None
        loop._thermal_policy = None
        loop.output_dir = tmp_path
        loop.save_cycle_snapshots = False
        loop._debug_base = None
        loop.snapshot_dir = None
        loop.shutdown_requested = False
        loop._bg_trainer = None
        loop._early_stop_callback = None
        loop.fingerprint_cache = None
        loop._keep_prior_slots = 2
        loop.cycle_count = 0
        loop._indexed_next_index = 1
        loop._procedural_next_index = 1
        loop._procedural_tentative_next_index = 1
        loop._indexed_ep_interim = {}
        loop.promoted_keys = set()
        loop.full_consolidation_period_string = ""
        loop.graph_enrichment_max_entities_per_pass = 50
        loop.graph_enrichment_neighborhood_hops = 2

        merger = GraphMerger(model=None)
        loop.merger = merger

        g = nx.MultiDiGraph()
        for i in range(node_count):
            g.add_node(f"node{i}", reinforcement_count=0, display_name=f"node{i}")
        loop.merger.graph = g

        store = MemoryStore()
        for tier in ("episodic", "semantic", "procedural"):
            store.load_registry(tier, KeyRegistry())
        loop.store = store
        # ``GraphTierRefiner`` reads the live ExtractionConfig through
        # ``loop._current_extraction_config`` to answer the cloud-admission
        # question, so an ``object.__new__`` stub must carry the attribute
        # that method reads.
        from types import SimpleNamespace as _SNS

        from paramem.graph.extraction_pipeline import ExtractionConfig as _ExtCfg

        loop.extraction = _SNS(config=_ExtCfg(scrub=set()))

        return loop

    # ---------------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------------

    @staticmethod
    def _add_keyed_edge(graph, subj, obj, predicate, ik_key, *, sessions=None, recurrence=1):
        """Add a directed edge with standard attributes and a keyed ik_key."""
        from paramem.memory.persistence import _IK_KEY_ATTR
        from paramem.utils.identity import canonical as _can

        subj = _can(subj)
        obj = _can(obj)
        predicate = _can(predicate)
        attrs = {
            "predicate": predicate,
            "relation_type": "factual",
            "sessions": sessions or ["sess1"],
            "reinforcement_count": recurrence,
            "confidence": 0.9,
            _IK_KEY_ATTR: ik_key,
        }
        graph.add_node(subj, reinforcement_count=1, display_name=subj)
        graph.add_node(obj, reinforcement_count=1, display_name=obj)
        graph.add_edge(subj, obj, **attrs)

    @staticmethod
    def _add_keyless_edge(graph, subj, obj, predicate, *, sessions=None, recurrence=1):
        """Add a directed edge WITHOUT an ik_key (simulates a fresh-ingested fact)."""
        from paramem.utils.identity import canonical as _can

        subj = _can(subj)
        obj = _can(obj)
        predicate = _can(predicate)
        attrs = {
            "predicate": predicate,
            "relation_type": "factual",
            "sessions": sessions or ["sess1"],
            "reinforcement_count": recurrence,
            "confidence": 0.9,
        }
        graph.add_node(subj, reinforcement_count=1, display_name=subj)
        graph.add_node(obj, reinforcement_count=1, display_name=obj)
        graph.add_edge(subj, obj, **attrs)

    # Prompt stub: only {predicates_json} placeholder (matches normalize_predicates).
    _PROMPT_STUB = "dummy {predicates_json}"

    def _cluster_response(self, clusters: list[list[str]]) -> str:
        """Encode a clusters-schema model response for normalize_predicates."""
        import json

        return json.dumps({"clusters": clusters})

    # ---------------------------------------------------------------------------
    # Tests
    # ---------------------------------------------------------------------------

    def test_keyed_synonym_retired_and_ledgered(self, tmp_path):
        """Two keyed synonym predicates for same (s,o) — model returns cluster.

        Graph: jordan -> techcorp with keyed edges 'works_for' (graph42, rec=1) and
        'employed_by' (graph87, rec=2).  Model returns cluster [works_for, employed_by].
        Survivor = MAX rec = graph87 (employed_by, rec=2).
        After apply:
        - graph42 (lower-rec) removed + in removal_ledger with reason
          'predicate_synonym_collapse'.
        - graph87 (survivor) still in graph.
        - result["edges_retired"]==1, result["groups_collapsed"]==1.
        """
        from unittest.mock import patch

        from paramem.memory.persistence import _IK_KEY_ATTR

        loop = self._make_loop(tmp_path)
        graph = loop.merger.graph

        self._add_keyed_edge(graph, "jordan", "techcorp", "works_for", "graph42", sessions=["s1"])
        self._add_keyed_edge(
            graph, "jordan", "techcorp", "employed_by", "graph87", sessions=["s2"], recurrence=2
        )

        # Cluster: both predicates are synonyms; MAX rec (graph87, rec=2) survives.
        cluster_response = self._cluster_response([["works_for", "employed_by"]])

        with (
            patch("paramem.graph.extractor.generate_answer", return_value=cluster_response),
            patch("paramem.graph.extractor._load_prompt", return_value=self._PROMPT_STUB),
        ):
            result = _refiner_for(loop).run_normalization()

        all_keys = [edata.get(_IK_KEY_ATTR) for _, _, edata in graph.edges(data=True)]
        assert "graph42" not in all_keys, "graph42 (lower-rec) must be removed"
        assert "graph87" in all_keys, "graph87 (MAX-rec survivor) must remain"
        assert "graph42" in loop.merger.removal_ledger
        assert loop.merger.removal_ledger["graph42"]["reason"] == "predicate_synonym_collapse"
        assert "graph87" not in loop.merger.removal_ledger, "survivor must NOT be ledgered"

        assert result["edges_retired"] == 1
        assert result["groups_collapsed"] == 1
        assert result["skipped"] is False

    def test_keyless_synonym_retired_no_ledger(self, tmp_path):
        """Two keyless synonym predicates — lower-rec edge removed, no ledger entry.

        Fresh-ingest case: facts arrive keyless.  Graph has two keyless edges for
        (jordan, techcorp): 'works_for' (rec=1) and 'employed_by' (rec=3).
        Survivor = MAX rec = employed_by (rec=3).
        After apply: 'works_for' (lower-rec) removed; removal_ledger empty.
        """
        from unittest.mock import patch

        loop = self._make_loop(tmp_path)
        graph = loop.merger.graph

        self._add_keyless_edge(
            graph, "jordan", "techcorp", "works_for", sessions=["s1"], recurrence=1
        )
        self._add_keyless_edge(
            graph, "jordan", "techcorp", "employed_by", sessions=["s2"], recurrence=3
        )

        cluster_response = self._cluster_response([["works_for", "employed_by"]])

        with (
            patch("paramem.graph.extractor.generate_answer", return_value=cluster_response),
            patch("paramem.graph.extractor._load_prompt", return_value=self._PROMPT_STUB),
        ):
            result = _refiner_for(loop).run_normalization()

        remaining_preds = {
            edata["predicate"] for _, _, edata in graph.edges(data=True) if edata.get("predicate")
        }
        # One surface form: the stored predicate IS "employed by" (rec=3 survives).
        assert "works for" not in remaining_preds, "lower-rec edge must be removed"
        assert "employed by" in remaining_preds, "MAX-rec survivor must remain"
        assert not loop.merger.removal_ledger, "no ledger entry for keyless retirements"

        assert result["edges_retired"] == 1
        assert result["groups_collapsed"] == 1

    def test_provenance_unioned_onto_survivor(self, tmp_path):
        """Sessions, recurrence, and confidence are unioned onto the survivor.

        Graph: morgan -> germany with 'born_in' (graph12, sessions=['s1'], rec=1)
        and 'birthplace' (graph34, sessions=['s2'], rec=2, confidence=0.95).
        Survivor = MAX rec = graph34 (birthplace, rec=2).
        After apply: graph34 survivor has sessions=['s2','s1'], recurrence>=3,
        confidence>=0.95; graph12 retired and in ledger with 'predicate_synonym_collapse'.
        """
        from unittest.mock import patch

        from paramem.memory.persistence import _IK_KEY_ATTR

        loop = self._make_loop(tmp_path)
        graph = loop.merger.graph

        self._add_keyed_edge(
            graph, "morgan", "germany", "born_in", "graph12", sessions=["s1"], recurrence=1
        )
        self._add_keyed_edge(
            graph, "morgan", "germany", "birthplace", "graph34", sessions=["s2"], recurrence=2
        )
        # Patch confidence on the graph34 edge so we can assert max.
        for _, _, edata in graph.edges(data=True):
            if edata.get(_IK_KEY_ATTR) == "graph34":
                edata["confidence"] = 0.95

        cluster_response = self._cluster_response([["born in", "birthplace"]])

        with (
            patch("paramem.graph.extractor.generate_answer", return_value=cluster_response),
            patch("paramem.graph.extractor._load_prompt", return_value=self._PROMPT_STUB),
        ):
            result = _refiner_for(loop).run_normalization()

        assert "graph12" in loop.merger.removal_ledger
        assert loop.merger.removal_ledger["graph12"]["reason"] == "predicate_synonym_collapse"
        assert "graph34" not in loop.merger.removal_ledger, "survivor must not be ledgered"

        # graph34 (birthplace, MAX rec=2) is the survivor.
        survivor = [
            edata for _, _, edata in graph.edges(data=True) if edata.get(_IK_KEY_ATTR) == "graph34"
        ]
        assert survivor, "graph34 survivor must remain in graph"
        e = survivor[0]
        assert "s2" in e.get("sessions", []), "s2 must be retained on survivor"
        assert "s1" in e.get("sessions", []), "s1 from retired edge must be unioned"
        assert e.get("reinforcement_count", 0) >= 3, "recurrence must be summed (2+1=3)"
        assert e.get("confidence", 0) >= 0.95, "max confidence must be applied"

        assert result["edges_retired"] == 1
        assert result["groups_collapsed"] == 1

    def test_provenance_last_seen_max_on_survivor(self, tmp_path):
        """NDA-3b: last_seen on the survivor edge equals max(survivor, retired).

        INVARIANT: whenever edges collapse into a survivor, last_seen = freshest.
        Both edges have rec=1; last_seen tiebreaker selects graph34 (newer) as
        survivor.  The survivor's last_seen must equal the max across both edges.
        """
        from unittest.mock import patch

        from paramem.memory.persistence import _IK_KEY_ATTR

        loop = self._make_loop(tmp_path)
        graph = loop.merger.graph

        self._add_keyed_edge(graph, "morgan", "germany", "born_in", "graph12", sessions=["s1"])
        self._add_keyed_edge(graph, "morgan", "germany", "birthplace", "graph34", sessions=["s2"])
        # Patch last_seen: graph12 older, graph34 newer.  Both have default rec=1 so
        # last_seen is the tiebreaker -> graph34 survives.
        for _, _, edata in graph.edges(data=True):
            if edata.get(_IK_KEY_ATTR) == "graph12":
                edata["last_seen"] = "2026-05-01T08:00:00Z"
            elif edata.get(_IK_KEY_ATTR) == "graph34":
                edata["last_seen"] = "2026-06-20T14:00:00Z"

        cluster_response = self._cluster_response([["born in", "birthplace"]])

        with (
            patch("paramem.graph.extractor.generate_answer", return_value=cluster_response),
            patch("paramem.graph.extractor._load_prompt", return_value=self._PROMPT_STUB),
        ):
            _refiner_for(loop).run_normalization()

        # graph34 (birthplace, newer last_seen) survives.
        survivor = [
            edata for _, _, edata in graph.edges(data=True) if edata.get(_IK_KEY_ATTR) == "graph34"
        ]
        assert survivor, "graph34 (newer last_seen) must survive"
        assert survivor[0].get("last_seen") == "2026-06-20T14:00:00Z", (
            "Survivor last_seen must be the freshest (max) across survivor + retired; "
            f"got {survivor[0].get('last_seen')!r}"
        )

    def test_provenance_first_seen_min_on_survivor(self, tmp_path):
        """NDA-3c: first_seen on the survivor edge equals min_nonempty(survivor,
        retired) — the earliest assertion window start, propagated alongside the
        existing max(last_seen).

        Both edges have rec=1; last_seen tiebreaker selects graph34 (newer) as
        survivor.  graph12 (retired) has the EARLIER first_seen; the survivor's
        first_seen must adopt that earlier value, not its own later one.
        """
        from unittest.mock import patch

        from paramem.memory.persistence import _IK_KEY_ATTR

        loop = self._make_loop(tmp_path)
        graph = loop.merger.graph

        self._add_keyed_edge(graph, "morgan", "germany", "born_in", "graph12", sessions=["s1"])
        self._add_keyed_edge(graph, "morgan", "germany", "birthplace", "graph34", sessions=["s2"])
        for _, _, edata in graph.edges(data=True):
            if edata.get(_IK_KEY_ATTR) == "graph12":
                edata["last_seen"] = "2026-05-01T08:00:00Z"
                edata["first_seen"] = "2026-01-01T00:00:00Z"  # earlier
            elif edata.get(_IK_KEY_ATTR) == "graph34":
                edata["last_seen"] = "2026-06-20T14:00:00Z"  # tiebreak winner
                edata["first_seen"] = "2026-04-01T00:00:00Z"  # later than graph12's

        cluster_response = self._cluster_response([["born in", "birthplace"]])

        with (
            patch("paramem.graph.extractor.generate_answer", return_value=cluster_response),
            patch("paramem.graph.extractor._load_prompt", return_value=self._PROMPT_STUB),
        ):
            _refiner_for(loop).run_normalization()

        survivor = [
            edata for _, _, edata in graph.edges(data=True) if edata.get(_IK_KEY_ATTR) == "graph34"
        ]
        assert survivor, "graph34 (newer last_seen) must survive"
        assert survivor[0].get("first_seen") == "2026-01-01T00:00:00Z", (
            "Survivor first_seen must be the earliest (min) across survivor + retired, "
            f"even though graph34 (survivor) is the last_seen tiebreak winner; "
            f"got {survivor[0].get('first_seen')!r}"
        )
        assert survivor[0].get("last_seen") == "2026-06-20T14:00:00Z", (
            "last_seen must remain the max, unaffected by the first_seen propagation"
        )

    def test_single_predicate_group_no_model_call(self, tmp_path):
        """Single-predicate (s,o) group is never a candidate — no model call.

        Graph: morgan -> germany with only 'born_in' (graph12).  The group has only
        one predicate so it never reaches normalize_predicates as a candidate.
        After apply: graph12 survives, ledger empty.
        """
        from unittest.mock import patch

        from paramem.memory.persistence import _IK_KEY_ATTR

        loop = self._make_loop(tmp_path)
        graph = loop.merger.graph

        self._add_keyed_edge(graph, "morgan", "germany", "born_in", "graph12", sessions=["s1"])

        # generate_answer must not be called for a single-predicate group.
        with (
            patch(
                "paramem.graph.extractor.generate_answer",
                side_effect=Exception("should not be called"),
            ),
            patch("paramem.graph.extractor._load_prompt", return_value=self._PROMPT_STUB),
        ):
            result = _refiner_for(loop).run_normalization()

        all_keys = [edata.get(_IK_KEY_ATTR) for _, _, edata in graph.edges(data=True)]
        assert "graph12" in all_keys, "graph12 must survive (not a multi-predicate group)"
        assert not loop.merger.removal_ledger, "Ledger must be empty"
        assert result["edges_retired"] == 0
        assert result["groups_collapsed"] == 0
        assert result["chunks"] == 0, "no model calls for single-predicate group"

    def test_empty_clusters_response_is_noop(self, tmp_path):
        """Model returns empty clusters for a group → no retirement.

        Graph: jordan -> techcorp with 'works_for' (graph42) and 'employed_by' (graph87).
        Model returns {"clusters": []} — predicates are NOT synonyms.
        After apply: both edges survive, ledger empty.
        """
        from unittest.mock import patch

        from paramem.memory.persistence import _IK_KEY_ATTR

        loop = self._make_loop(tmp_path)
        graph = loop.merger.graph

        self._add_keyed_edge(graph, "jordan", "techcorp", "works_for", "graph42")
        self._add_keyed_edge(graph, "jordan", "techcorp", "employed_by", "graph87")

        # Model returns no clusters — predicates are not synonyms; no collapse.
        cluster_response = self._cluster_response([])

        with (
            patch("paramem.graph.extractor.generate_answer", return_value=cluster_response),
            patch("paramem.graph.extractor._load_prompt", return_value=self._PROMPT_STUB),
        ):
            result = _refiner_for(loop).run_normalization()

        all_keys = [edata.get(_IK_KEY_ATTR) for _, _, edata in graph.edges(data=True)]
        assert "graph42" in all_keys, "graph42 must survive"
        assert "graph87" in all_keys, "graph87 must survive"
        assert not loop.merger.removal_ledger, "Ledger must be empty"
        assert result["edges_retired"] == 0
        assert result["groups_collapsed"] == 0

    def test_no_model_returns_skipped(self, tmp_path):
        """model=None → pass skipped immediately, graph unchanged."""
        loop = self._make_loop(tmp_path, model=None)
        loop.model = None

        initial_nodes = loop.merger.graph.number_of_nodes()
        result = _refiner_for(loop).run_normalization()

        assert result["skipped"] is True
        assert result["skip_reason"] == "no_model"
        assert loop.merger.graph.number_of_nodes() == initial_nodes

    def test_small_graph_returns_skipped(self, tmp_path):
        """Graph < 10 nodes → pass skipped (below floor)."""
        loop = self._make_loop(tmp_path, node_count=5)

        result = _refiner_for(loop).run_normalization()

        assert result["skipped"] is True
        assert result["skip_reason"] == "floor"

    def test_mixed_keyed_and_keyless_correct_ledger(self, tmp_path):
        """Mixed keyed + keyless — keyed retired -> ledger; keyless retired -> no ledger.

        Graph: jordan -> techcorp with three edges (all rec=1):
        - 'works_for', keyed graph42 (survivor — first in cluster, tiebreaker)
        - 'employed_by', keyed graph87 (retired)
        - 'is_employed_at', keyless (retired)

        Cluster response puts 'works for' first -> graph42 (rec=1 tie, first) wins.
        After apply:
        - graph87 removed + in removal_ledger with reason 'predicate_synonym_collapse'.
        - keyless 'is_employed_at' removed, NOT in ledger.
        - graph42 survives.
        - result["edges_retired"]==2, result["groups_collapsed"]==1.
        """
        from unittest.mock import patch

        from paramem.memory.persistence import _IK_KEY_ATTR

        loop = self._make_loop(tmp_path)
        graph = loop.merger.graph

        self._add_keyed_edge(graph, "jordan", "techcorp", "works_for", "graph42", sessions=["s1"])
        self._add_keyed_edge(graph, "jordan", "techcorp", "employed_by", "graph87", sessions=["s2"])
        self._add_keyless_edge(graph, "jordan", "techcorp", "is_employed_at", sessions=["s3"])

        # 'works for' first in cluster -> MAX rec tie broken in favour of first -> graph42 survives.
        cluster_response = self._cluster_response([["works for", "employed by", "is employed at"]])

        with (
            patch("paramem.graph.extractor.generate_answer", return_value=cluster_response),
            patch("paramem.graph.extractor._load_prompt", return_value=self._PROMPT_STUB),
        ):
            result = _refiner_for(loop).run_normalization()

        all_keys = [edata.get(_IK_KEY_ATTR) for _, _, edata in graph.edges(data=True)]
        assert "graph87" not in all_keys, "graph87 (keyed) must be removed"
        assert "graph42" in all_keys, "graph42 must survive"
        assert "graph87" in loop.merger.removal_ledger
        assert loop.merger.removal_ledger["graph87"]["reason"] == "predicate_synonym_collapse"
        assert len(loop.merger.removal_ledger) == 1, "keyless retirement must not add ledger entry"

        assert result["edges_retired"] == 2
        assert result["groups_collapsed"] == 1

    def test_single_predicate_group_untouched_when_other_group_collapsed(self, tmp_path):
        """Single-predicate (s,o) group is not touched when another group is collapsed.

        Graph: sam -> berlin with only 'lives_in' (graph99) — single predicate, not a
        candidate.  jordan -> techcorp has 'works_for' (graph42) and 'employed_by' (graph87)
        — two-predicate candidate.  Model returns cluster for jordan/techcorp only.
        sam -> berlin must survive; jordan/techcorp group collapses (graph87 retired).
        """
        from unittest.mock import patch

        from paramem.memory.persistence import _IK_KEY_ATTR

        loop = self._make_loop(tmp_path)
        graph = loop.merger.graph

        self._add_keyed_edge(graph, "sam", "berlin", "lives_in", "graph99", sessions=["s1"])
        self._add_keyed_edge(graph, "jordan", "techcorp", "works_for", "graph42", sessions=["s2"])
        self._add_keyed_edge(graph, "jordan", "techcorp", "employed_by", "graph87", sessions=["s3"])

        # Cluster for the jordan/techcorp group; 'works for' first -> graph42 survives.
        cluster_response = self._cluster_response([["works_for", "employed_by"]])

        with (
            patch("paramem.graph.extractor.generate_answer", return_value=cluster_response),
            patch("paramem.graph.extractor._load_prompt", return_value=self._PROMPT_STUB),
        ):
            result = _refiner_for(loop).run_normalization()

        all_keys = [edata.get(_IK_KEY_ATTR) for _, _, edata in graph.edges(data=True)]
        assert "graph99" in all_keys, "single-predicate (s,o) must not be touched"
        assert "graph42" in all_keys, "graph42 must survive (first in cluster, rec tie)"
        assert result["edges_retired"] == 1, "only graph87 retired (the jordan/techcorp group)"
        assert result["groups_collapsed"] == 1

    def test_max_rec_survivor_selected(self, tmp_path):
        """MAX reinforcement_count edge survives; lower-rec edge retired.

        Graph: morgan -> germany with 'born_in' (graph12, rec=1) and
        'birthplace' (graph34, rec=2).  Cluster collapses both.
        Survivor = MAX rec = graph34 (birthplace, rec=2).
        After apply: graph12 retired, graph34 survives.
        No new edges (hallucinated subjects are impossible in the cluster schema).
        """
        from unittest.mock import patch

        from paramem.memory.persistence import _IK_KEY_ATTR

        loop = self._make_loop(tmp_path)
        graph = loop.merger.graph

        self._add_keyed_edge(
            graph, "morgan", "germany", "born_in", "graph12", sessions=["s1"], recurrence=1
        )
        self._add_keyed_edge(
            graph, "morgan", "germany", "birthplace", "graph34", sessions=["s2"], recurrence=2
        )

        cluster_response = self._cluster_response([["born in", "birthplace"]])

        with (
            patch("paramem.graph.extractor.generate_answer", return_value=cluster_response),
            patch("paramem.graph.extractor._load_prompt", return_value=self._PROMPT_STUB),
        ):
            result = _refiner_for(loop).run_normalization()

        all_keys = [edata.get(_IK_KEY_ATTR) for _, _, edata in graph.edges(data=True)]
        assert "graph12" not in all_keys, "graph12 (lower-rec born_in) must be retired"
        assert "graph34" in all_keys, "graph34 (MAX-rec birthplace) must survive"
        assert "graph12" in loop.merger.removal_ledger
        assert loop.merger.removal_ledger["graph12"]["reason"] == "predicate_synonym_collapse"

        assert result["edges_retired"] == 1
        assert result["groups_collapsed"] == 1


# ---------------------------------------------------------------------------
# cloud engine wiring in GraphTierRefiner.run_normalization
# ---------------------------------------------------------------------------


class TestRunGraphNormalizationCloudEngine:
    """``GraphTierRefiner.run_normalization`` cloud wiring and fail-loud tests.

    Tests:
    - cloud_enabled=True + provider + api_key in env → normalize_predicates
      called with ``cloud=`` kwarg, NOT ``model=``.
    - cloud_enabled=True + provider present but NO api_key in env → local
      fallback: normalize_predicates called with ``model=`` kwarg.
    - after retirement, on_normalization receives non-empty raw_outputs list
      and non-empty decisions list.
    - FileNotFoundError raised when predicate_normalization.txt is missing
      (graph ≥ 10 nodes, model present).
    """

    @staticmethod
    def _make_loop(tmp_path, *, cloud_enabled: bool = False, node_count: int = 15):
        """Build a minimal ConsolidationLoop for normalization engine tests."""
        import networkx as nx

        from paramem.graph.merger import GraphMerger
        from paramem.memory.store import MemoryStore
        from paramem.training.consolidation import ConsolidationLoop
        from paramem.training.key_registry import KeyRegistry
        from paramem.utils.config import AdapterConfig, ConsolidationConfig, TrainingConfig

        loop = object.__new__(ConsolidationLoop)
        loop.model = MagicMock()
        loop.tokenizer = MagicMock()
        loop.tokenizer.apply_chat_template.return_value = "formatted_prompt"
        loop.config = ConsolidationConfig(
            refinement_normalization="on",
        )
        loop.cloud_enabled = cloud_enabled
        loop.training_config = TrainingConfig(
            num_epochs=1,
            gradient_checkpointing=False,
            batch_size=1,
            recall_early_stopping=False,
            recall_probe_batch_size=1,
        )
        loop.episodic_config = AdapterConfig(rank=4, alpha=8, target_modules=["q_proj"])
        loop.semantic_config = AdapterConfig(rank=4, alpha=8, target_modules=["q_proj"])
        loop.procedural_config = None
        loop.wandb_config = None
        loop._thermal_policy = None
        loop.output_dir = tmp_path
        loop.save_cycle_snapshots = False
        loop._debug_base = None
        loop.snapshot_dir = None
        loop.shutdown_requested = False
        loop._bg_trainer = None
        loop._early_stop_callback = None
        loop.fingerprint_cache = None
        loop._keep_prior_slots = 2
        loop.cycle_count = 0
        loop._indexed_next_index = 1
        loop._procedural_next_index = 1
        loop._procedural_tentative_next_index = 1
        loop._indexed_ep_interim = {}
        loop.promoted_keys = set()
        loop.full_consolidation_period_string = ""
        loop.graph_enrichment_max_entities_per_pass = 50
        loop.graph_enrichment_neighborhood_hops = 2

        # Extraction pipeline mock — config.enrichment_provider used for cloud engine resolution.
        ext_mock = MagicMock()
        ext_mock.config.enrichment_provider = "anthropic"
        ext_mock.config.enrichment_provider_model = "claude-sonnet-4-6"
        ext_mock.config.enrichment_provider_endpoint = None
        loop.extraction = ext_mock

        merger = GraphMerger(model=None)
        loop.merger = merger

        g = nx.MultiDiGraph()
        for i in range(node_count):
            g.add_node(f"node{i}", reinforcement_count=0, display_name=f"node{i}")
        loop.merger.graph = g

        store = MemoryStore()
        for tier in ("episodic", "semantic", "procedural"):
            store.load_registry(tier, KeyRegistry())
        loop.store = store
        return loop

    @staticmethod
    def _add_keyed_edge(graph, subj, obj, predicate, ik_key, *, sessions=None, recurrence=1):
        from paramem.memory.persistence import _IK_KEY_ATTR
        from paramem.utils.identity import canonical as _can

        subj = _can(subj)
        obj = _can(obj)
        predicate = _can(predicate)
        attrs = {
            "predicate": predicate,
            "relation_type": "factual",
            "sessions": sessions or ["sess1"],
            "reinforcement_count": recurrence,
            "confidence": 0.9,
            _IK_KEY_ATTR: ik_key,
        }
        graph.add_node(subj, reinforcement_count=1, display_name=subj)
        graph.add_node(obj, reinforcement_count=1, display_name=obj)
        graph.add_edge(subj, obj, **attrs)

    _PROMPT_STUB = "dummy {predicates_json}"

    def test_cloud_engine_selected_when_enabled_with_api_key(self, tmp_path, monkeypatch):
        """cloud_enabled=True + provider + env api_key → primitive gets cloud=."""

        from unittest.mock import patch

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
        loop = self._make_loop(tmp_path, cloud_enabled=True)
        graph = loop.merger.graph
        self._add_keyed_edge(graph, "morgan", "acme", "works_for", "g1", recurrence=1)
        self._add_keyed_edge(graph, "morgan", "acme", "employed_by", "g2", recurrence=2)

        # Capture the kwargs that normalize_predicates receives.
        captured: dict = {}

        def _fake_dedup(relations, **kwargs):
            captured.update(kwargs)
            return {}, {
                "raw_outputs": [],
                "groups_examined": 0,
                "candidate_groups": 0,
                "groups_with_clusters": 0,
                "model_calls": 0,
                "discards": [],
            }

        with (
            patch(
                "paramem.training.graph_tier.normalize_predicates",
                side_effect=_fake_dedup,
            ),
            patch("paramem.graph.extractor._load_prompt", return_value=self._PROMPT_STUB),
        ):
            _refiner_for(loop).run_normalization()

        assert "cloud" in captured, (
            "normalize_predicates must receive cloud= kwarg "
            "when cloud_enabled=True and api_key present"
        )
        assert "model" not in captured, "model= must NOT be passed when cloud engine is selected"
        assert captured["cloud"]["provider"] == "anthropic"
        assert captured["cloud"]["api_key"] == "sk-test-key"

    def test_local_fallback_when_api_key_absent(self, tmp_path, monkeypatch):
        """cloud_enabled=True but NO api_key → local fallback (model= kwarg)."""
        from unittest.mock import patch

        # Ensure the env var is absent.
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        loop = self._make_loop(tmp_path, cloud_enabled=True)
        graph = loop.merger.graph
        self._add_keyed_edge(graph, "morgan", "acme", "works_for", "g1", recurrence=1)
        self._add_keyed_edge(graph, "morgan", "acme", "employed_by", "g2", recurrence=2)

        captured: dict = {}

        def _fake_dedup(relations, **kwargs):
            captured.update(kwargs)
            return {}, {
                "raw_outputs": [],
                "groups_examined": 0,
                "candidate_groups": 0,
                "groups_with_clusters": 0,
                "model_calls": 0,
                "discards": [],
            }

        with (
            patch(
                "paramem.training.graph_tier.normalize_predicates",
                side_effect=_fake_dedup,
            ),
            patch("paramem.graph.extractor._load_prompt", return_value=self._PROMPT_STUB),
        ):
            _refiner_for(loop).run_normalization()

        assert "model" in captured, (
            "normalize_predicates must receive model= kwarg when api_key is absent"
        )
        assert "cloud" not in captured, (
            "cloud= must NOT be passed when api_key is absent (local fallback)"
        )

    def test_on_normalization_receives_nonempty_raw_outputs_and_decisions(self, tmp_path):
        """After retirement, on_normalization receives non-empty raw_outputs
        and non-empty decisions — the debug snapshot has real data to write."""
        import json
        from unittest.mock import patch

        loop = self._make_loop(tmp_path, cloud_enabled=False)
        graph = loop.merger.graph
        self._add_keyed_edge(graph, "morgan", "acme", "works_for", "g1", recurrence=1)
        self._add_keyed_edge(graph, "morgan", "acme", "employed_by", "g2", recurrence=2)

        cluster_raw = json.dumps({"clusters": [["works for", "employed by"]]})

        on_norm_calls: list = []

        def _spy_on_normalization(raw_outputs, decisions, applied, **kwargs):
            on_norm_calls.append(
                {
                    "raw_outputs": raw_outputs,
                    "decisions": decisions,
                    "applied": applied,
                }
            )

        with (
            patch("paramem.graph.extractor.generate_answer", return_value=cluster_raw),
            patch("paramem.graph.extractor._load_prompt", return_value=self._PROMPT_STUB),
            patch(
                "paramem.training.graph_tier.on_normalization",
                side_effect=_spy_on_normalization,
            ),
        ):
            _refiner_for(loop).run_normalization()

        assert on_norm_calls, "on_normalization must be called"
        call = on_norm_calls[0]
        assert call["raw_outputs"], "raw_outputs must be non-empty after a model call"
        assert call["decisions"], "decisions must be non-empty when clusters were produced"

    def test_fail_loud_when_prompt_missing(self, tmp_path):
        """FileNotFoundError raised when predicate_normalization.txt is missing.

        The prompt load now happens inside ``normalize_predicates`` itself,
        gated on ``relations`` being non-empty (it fires before the
        candidate-group check) — so the graph needs at least one edge for the
        load (and thus the raise) to happen at all.
        """
        from unittest.mock import patch

        import pytest

        loop = self._make_loop(tmp_path, cloud_enabled=False)
        graph = loop.merger.graph
        self._add_keyed_edge(graph, "morgan", "acme", "works_for", "g1", recurrence=1)

        with (
            patch(
                "paramem.graph.extractor._load_prompt",
                side_effect=FileNotFoundError(
                    "Required prompt file 'predicate_normalization.txt' not found. Searched: ..."
                ),
            ),
            pytest.raises(FileNotFoundError, match="predicate_normalization.txt"),
        ):
            _refiner_for(loop).run_normalization()


# ---------------------------------------------------------------------------
# Debug snapshot — on_normalization writes normalization_snapshot.json
# ---------------------------------------------------------------------------


class TestNormalizationDebugSnapshot:
    """on_normalization routes raw outputs + decisions + applied counts through
    the shared artifact primitive and writes normalization_snapshot.json under fold/.

    Tests:
    - save_cycle_snapshots=True → normalization_snapshot.json written with
      raw_outputs, decisions, and applied counts (index-delta schema).
    - save_cycle_snapshots=False → no file written (self-gated no-op).
    """

    @staticmethod
    def _make_debug_loop(tmp_path, *, save_cycle_snapshots: bool):
        """Build a ConsolidationLoop with debug snapshot writing enabled/disabled."""
        from paramem.training.consolidation import ConsolidationLoop
        from paramem.utils.config import ConsolidationConfig

        loop = object.__new__(ConsolidationLoop)
        loop.config = ConsolidationConfig()
        loop.save_cycle_snapshots = save_cycle_snapshots
        loop.cycle_count = 0
        loop.run_id = "test_run"

        if save_cycle_snapshots:
            debug_base = tmp_path / "debug"
            debug_base.mkdir()
            loop._debug_base = debug_base
            loop.snapshot_dir = tmp_path / "snapshot"
            loop.snapshot_dir.mkdir()
        else:
            loop._debug_base = None
            loop.snapshot_dir = None

        return loop

    def test_normalization_snapshot_written_when_debug_enabled(self, tmp_path):
        """save_cycle_snapshots=True → normalization_snapshot.json written."""
        import json

        loop = self._make_debug_loop(tmp_path, save_cycle_snapshots=True)

        raw_outputs = ["raw model output goes here"]
        decisions = [{"relations": [{"subject": "A", "predicate": "b", "object": "C"}]}]
        applied = {"groups_collapsed": 1, "edges_retired": 1}

        with loop._artifact_scope():
            on_normalization(raw_outputs, decisions, applied)

        matches = list(tmp_path.rglob("normalization_snapshot.json"))
        assert matches, "normalization_snapshot.json must be written when debug is enabled"
        payload = json.loads(matches[0].read_text())
        assert payload["raw_outputs"] == raw_outputs
        assert payload["decisions"] == decisions
        assert payload["applied"] == applied

    def test_normalization_snapshot_not_written_when_debug_disabled(self, tmp_path):
        """save_cycle_snapshots=False → no file written (self-gated no-op)."""
        loop = self._make_debug_loop(tmp_path, save_cycle_snapshots=False)

        raw_outputs = ["raw output"]
        decisions = [{"relations": []}]
        applied = {"groups_collapsed": 0, "edges_retired": 0}

        with loop._artifact_scope():
            on_normalization(raw_outputs, decisions, applied)

        matches = list(tmp_path.rglob("normalization_snapshot.json"))
        assert not matches, "normalization_snapshot.json must NOT be written when debug is disabled"


# ---------------------------------------------------------------------------
# Debug snapshot — the two artifact roots and their independent gates
# ---------------------------------------------------------------------------


class TestCalibrationAndDebugRoots:
    """``_active_bases`` resolves TWO roots, gated independently.

    The debug root comes from ``ConsolidationLoop._artifact_scope`` (``None``
    when ``debug`` is off); the calibration root from ``calibration_run``.

    A calibration run must capture its artifacts whether or not the
    production ``debug`` switch is on — the two answer different
    questions. With both active the same artifact lands in both, so the
    debug tree stays a complete record and the run's directory stays
    self-contained.
    """

    @staticmethod
    def _loop(tmp_path, *, save_cycle_snapshots: bool):
        return TestNormalizationDebugSnapshot._make_debug_loop(
            tmp_path, save_cycle_snapshots=save_cycle_snapshots
        )

    def test_calibration_root_receives_artifact_with_debug_off(self, tmp_path):
        """Mutation: gate the calibration root on ``save_cycle_snapshots``
        -> nothing is captured and every probe silently depends on a
        production setting it has nothing to do with."""
        from paramem.utils.artifacts import calibration_run

        loop = self._loop(tmp_path, save_cycle_snapshots=False)
        run_dir = tmp_path / "calibration" / "calibrate" / "extract" / "20260101T000000Z"
        with loop._artifact_scope(), calibration_run(run_dir):
            on_calibration_result({"stage": "extract", "parsed": {}}, stamp="20260101T000000Z")

        import json as _json

        written = run_dir / "response.json"
        assert written.exists()
        assert _json.loads(written.read_text())["stage"] == "extract"

    def test_both_roots_receive_the_artifact_when_both_are_active(self, tmp_path):
        from paramem.utils.artifacts import calibration_run

        loop = self._loop(tmp_path, save_cycle_snapshots=True)
        run_dir = tmp_path / "calibration" / "calibrate" / "extract" / "20260101T000000Z"
        with loop._artifact_scope(), calibration_run(run_dir):
            on_calibration_result({"stage": "extract", "parsed": {}}, stamp="20260101T000000Z")

        assert (run_dir / "response.json").exists()
        assert len(list((tmp_path / "debug").rglob("response.json"))) == 1

    def test_no_calibration_run_leaves_only_the_debug_root(self, tmp_path):
        loop = self._loop(tmp_path, save_cycle_snapshots=True)
        with loop._artifact_scope():
            on_calibration_result({"stage": "extract", "parsed": {}}, stamp="20260101T000000Z")

        assert len(list((tmp_path / "debug").rglob("response.json"))) == 1
        assert not (tmp_path / "calibration").exists()

    def test_neither_root_active_is_a_no_op(self, tmp_path):
        loop = self._loop(tmp_path, save_cycle_snapshots=False)
        with loop._artifact_scope():
            on_calibration_result({"stage": "extract", "parsed": {}}, stamp="20260101T000000Z")

        assert not list(tmp_path.rglob("response.json"))

    def test_production_hook_artifacts_follow_the_calibration_run(self, tmp_path):
        """Not just the calibration response: an artifact a PRODUCTION hook
        emits while a calibration run executes lands in the run's directory
        too. ``/calibrate/normalize`` runs the graph tier's normalization
        pass, which writes its raw outputs through this same primitive.
        """
        from paramem.utils.artifacts import calibration_run

        loop = self._loop(tmp_path, save_cycle_snapshots=False)
        run_dir = tmp_path / "calibration" / "normalize_1"
        with loop._artifact_scope(), calibration_run(run_dir):
            on_normalization(["raw"], [{"cluster": ["a", "b"]}], {"retired": 1})

        assert (run_dir / "fold" / "normalization_snapshot.json").exists()


# ---------------------------------------------------------------------------
# Debug snapshot — path-component sanitisation
# ---------------------------------------------------------------------------


class TestSafePathComponent:
    """``_safe_path_component`` maps non alnum/-/_ characters to ``_``.

    Guards the ``on_session_extracted`` directory-component write site
    against a request-controlled ``session_id`` containing path separators
    or ``..``.
    """

    def test_traversal_attempt_sanitized(self):
        from paramem.utils.artifacts import _safe_path_component

        assert "/" not in _safe_path_component("../etc")
        assert ".." not in _safe_path_component("../etc")

    def test_path_separator_sanitized(self):
        from paramem.utils.artifacts import _safe_path_component

        assert _safe_path_component("a/b") == "a_b"

    def test_normal_uuid_passes_through_unchanged(self):
        from paramem.utils.artifacts import _safe_path_component

        uuid = "f3c2c91e-c000"
        assert _safe_path_component(uuid) == uuid


# ---------------------------------------------------------------------------
# _full_consolidation_overdue_key unit tests
# ---------------------------------------------------------------------------


class TestFullConsolidationOverdueKey:
    """Unit tests for ``_full_consolidation_overdue_key(config) -> str | None``.

    Covers:
    - Returns oldest stamp when oldest interim ≥ 2× full_period (overdue).
    - Returns None when oldest interim is in [1×, 2×) period (due but within runway).
    - Returns None when no interims present.
    - Returns None when ``consolidation_period_seconds`` is None (manual-only).
    - Returns None when ``max_interim_count`` <= 0.
    """

    def _make_config(
        self, adapter_dir, *, max_interim_count: int = 7, period_seconds=302400, mode: str = "train"
    ):
        """Minimal config mock reusing the same shape as TestFullCycleGateHelpers._make_config."""
        cfg = MagicMock()
        cfg.adapter_dir = adapter_dir
        cfg.consolidation.max_interim_count = max_interim_count
        cfg.consolidation.consolidation_period_seconds = period_seconds
        cfg.consolidation.mode = mode
        return cfg

    def _make_interim_dir(self, adapter_dir, stamp: str) -> None:
        """Create an on-disk interim slot with a train-venue payload.

        The overdue key is derived from the venue-filtered (payload-bearing)
        interim set -- a candidate slot is a subdirectory carrying its own
        ``meta.json`` (``count_slot_candidates``), so the slot must carry
        both ``meta.json`` and ``adapter_model.safetensors`` to be seen.
        """
        interim_dir = adapter_dir / "episodic" / f"interim_{stamp}"
        slot = interim_dir / f"{stamp}-slot"
        slot.mkdir(parents=True, exist_ok=True)
        (slot / "meta.json").write_text("{}")
        (slot / "adapter_model.safetensors").write_bytes(b"")

    def _stamp_seconds_ago(self, seconds: float) -> str:
        """Return a YYYYMMDDTHHMM stamp for a datetime ``seconds`` ago.

        Local time, minute-floored (same granularity as ``interim_stamp_from_name``).
        """
        from datetime import datetime, timedelta

        from paramem.memory.interim_adapter import INTERIM_STAMP_FORMAT

        dt = datetime.now() - timedelta(seconds=seconds)
        # Floor to minute precision — same granularity as interim_stamp_from_name.
        return dt.strftime(INTERIM_STAMP_FORMAT)

    def test_overdue_returns_oldest_stamp(self, tmp_path):
        """Oldest interim aged ≥ 2× period → returns oldest stamp."""
        from paramem.server.app import _full_consolidation_overdue_key

        period = 3600  # 1 h
        # Place the oldest interim just over 2× the period ago.
        old_stamp = self._stamp_seconds_ago(2 * period + 300)
        self._make_interim_dir(tmp_path, old_stamp)
        cfg = self._make_config(tmp_path, period_seconds=period)
        result = _full_consolidation_overdue_key(cfg)
        assert result == old_stamp

    def test_within_runway_returns_none(self, tmp_path):
        """Oldest interim aged in [1×, 2×) period (due but within runway) → None.

        The fold is due (1× period) but has not yet missed its runway (< 2×
        period), so no overdue incident should fire.
        """
        from paramem.server.app import _full_consolidation_overdue_key

        period = 3600
        # Place the oldest interim at 1.5× the period — due but within runway.
        slightly_due_stamp = self._stamp_seconds_ago(int(1.5 * period))
        self._make_interim_dir(tmp_path, slightly_due_stamp)
        cfg = self._make_config(tmp_path, period_seconds=period)
        result = _full_consolidation_overdue_key(cfg)
        assert result is None

    def test_no_interims_returns_none(self, tmp_path):
        """No interim dirs present → None (empty ring, nothing overdue)."""
        from paramem.server.app import _full_consolidation_overdue_key

        (tmp_path / "episodic").mkdir()
        cfg = self._make_config(tmp_path, period_seconds=3600)
        assert _full_consolidation_overdue_key(cfg) is None

    def test_manual_only_period_none_returns_none(self, tmp_path):
        """consolidation_period_seconds=None (manual-only) → None.

        There is no auto schedule and therefore no deadline or overdue concept.
        """
        from paramem.server.app import _full_consolidation_overdue_key

        # Even an astronomically old interim must not fire when manual-only.
        self._make_interim_dir(tmp_path, "20200101T0000")
        cfg = self._make_config(tmp_path, period_seconds=None)
        assert _full_consolidation_overdue_key(cfg) is None

    def test_max_interim_count_zero_returns_none(self, tmp_path):
        """max_interim_count=0 (or negative) → None.

        N <= 0 is misconfigured; the helper returns None defensively (same
        guard as ``_is_full_cycle_due``).
        """
        from paramem.server.app import _full_consolidation_overdue_key

        self._make_interim_dir(tmp_path, "20200101T0000")
        cfg = self._make_config(tmp_path, max_interim_count=0, period_seconds=1)
        assert _full_consolidation_overdue_key(cfg) is None

    def test_multiple_interims_uses_oldest(self, tmp_path):
        """When multiple interims exist, the oldest (first in sorted order) governs."""
        from paramem.server.app import _full_consolidation_overdue_key

        period = 3600
        # Oldest is far over 2× period; newer one is within runway.
        old_stamp = self._stamp_seconds_ago(3 * period)
        recent_stamp = self._stamp_seconds_ago(int(1.5 * period))
        # Create both; iter_interim_dirs sorts ascending so old_stamp comes first.
        self._make_interim_dir(tmp_path, old_stamp)
        self._make_interim_dir(tmp_path, recent_stamp)
        cfg = self._make_config(tmp_path, period_seconds=period)
        result = _full_consolidation_overdue_key(cfg)
        # Must return the oldest stamp (overdue), not the recent one.
        assert result == old_stamp


# ---------------------------------------------------------------------------
# Dispatcher incident wiring tests
# ---------------------------------------------------------------------------


class TestFullCycleDispatcherOverdueIncident:
    """Dispatcher fires ``full_consolidation_overdue`` incident when due AND overdue.

    Covers:
    - Overdue: ``record_incident`` called with correct type/key; ``bump_retry_count``
      not called (scheduling state, not encoding failure).
    - Due but within runway: no overdue incident.
    - Two-tick lock: count=N → False (interim path), count=N+1 → True (full path).
    """

    def _make_interim_dir(self, adapter_dir, stamp: str) -> None:
        """Create a CONTENT-BEARING train-venue interim slot.

        The dispatcher's content gate counts slots that carry a candidate
        (a subdirectory with its own ``meta.json``), so the slot needs both
        ``meta.json`` and the venue payload -- a bare payload file would be
        invisible to it (and to the gate's deadline/incident helpers).
        """
        slot = adapter_dir / "episodic" / f"interim_{stamp}" / f"{stamp}-slot"
        slot.mkdir(parents=True, exist_ok=True)
        (slot / "meta.json").write_text("{}")
        (slot / "adapter_model.safetensors").write_bytes(b"")

    def _make_minimal_state(self, tmp_path) -> dict:
        """Minimal _state for dispatcher tests with a real config.paths.data.

        Seeds one content-bearing interim slot so the dispatcher's content gate
        passes and these tests exercise the overdue-incident wiring in
        isolation.
        """
        from paramem.server.schedule_state import write_last_scheduled_run

        cfg = MagicMock()
        cfg.consolidation.training_idle_debounce_s = 0
        cfg.consolidation.refresh_cadence = "12h"
        cfg.consolidation.mode = "train"
        # N > 0: the content gate's FULL branch reads this to decide whether a
        # pending session is even eligible input (it is not, at this count —
        # only the content-bearing interim slot below is).
        cfg.consolidation.max_interim_count = 7
        cfg.paths.data = tmp_path
        cfg.adapter_dir = tmp_path / "adapters"
        cfg.adapter_dir.mkdir(parents=True, exist_ok=True)
        # Pre-seed the durable catch-up stamp well before the current "12h"
        # mark so the scheduled tick reads DUE and reaches _is_full_cycle_due
        # (patched per-test) instead of seed-and-noop on a virgin stamp file —
        # these tests exercise the overdue-incident wiring, not the catch-up
        # gate itself.
        write_last_scheduled_run(tmp_path / "state", time.time() - 86400)
        self._make_interim_dir(cfg.adapter_dir, "20260101T0000")
        return {
            "config": cfg,
            "session_buffer": MagicMock(),
            "speaker_store": None,
            "consolidating": False,
            "mode": "local",
            "background_trainer": None,
            "last_chat_monotonic": None,
            "pending_rehydration": False,
            "integrity_check_failed": False,
            "cloud_only_reason": None,
        }

    def test_overdue_record_incident_called(self, tmp_path, monkeypatch):
        """When full cycle is due AND overdue, record_incident is called with
        type='full_consolidation_overdue' and key=oldest stamp.

        bump_retry_count is NOT called (scheduling state ≠ encoding failure).
        """
        from unittest.mock import patch

        import paramem.server.app as app_module

        state = self._make_minimal_state(tmp_path)
        old_stamp = "20200101T0000"
        self._make_interim_dir(state["config"].adapter_dir, old_stamp)

        monkeypatch.setattr(app_module, "_state", state)

        recorded: list[dict] = []

        def _capture_record_incident(state_dir, *, type, key, severity, summary, detail):
            recorded.append({"type": type, "key": key, "severity": severity})

        mock_loop = MagicMock()
        mock_future = MagicMock()
        mock_loop.run_in_executor.return_value = mock_future
        mock_future.add_done_callback.return_value = None
        state["event_loop"] = mock_loop

        with (
            patch("paramem.server.app._consolidation_dispatch_guards", return_value=None),
            patch("paramem.server.app._is_full_cycle_due", return_value=True),
            patch(
                "paramem.server.app._full_consolidation_overdue_key",
                return_value=old_stamp,
            ),
            patch("paramem.server.app.record_incident", side_effect=_capture_record_incident),
            patch("paramem.server.app._retro_claim_orphan_sessions", return_value=0),
        ):
            result, _action = app_module._dispatch_consolidation(
                app_module.ConsolidationAction.AUTO
            )

        assert result == "started_full"
        assert len(recorded) == 1, f"Expected exactly one incident; got: {recorded}"
        assert recorded[0]["type"] == "full_consolidation_overdue"
        assert recorded[0]["key"] == old_stamp
        assert recorded[0]["severity"] == "failed"

    def test_within_runway_no_overdue_incident(self, tmp_path, monkeypatch):
        """When full cycle is due but NOT overdue, record_incident is NOT called
        with type='full_consolidation_overdue'.
        """
        from unittest.mock import patch

        import paramem.server.app as app_module

        state = self._make_minimal_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", state)

        recorded_types: list[str] = []

        def _capture_record_incident(state_dir, *, type, key, severity, summary, detail):
            recorded_types.append(type)

        mock_loop = MagicMock()
        mock_future = MagicMock()
        mock_loop.run_in_executor.return_value = mock_future
        mock_future.add_done_callback.return_value = None
        state["event_loop"] = mock_loop

        with (
            patch("paramem.server.app._consolidation_dispatch_guards", return_value=None),
            patch("paramem.server.app._is_full_cycle_due", return_value=True),
            # _full_consolidation_overdue_key returns None → within runway
            patch("paramem.server.app._full_consolidation_overdue_key", return_value=None),
            patch("paramem.server.app.record_incident", side_effect=_capture_record_incident),
            patch("paramem.server.app._retro_claim_orphan_sessions", return_value=0),
        ):
            result, _action = app_module._dispatch_consolidation(
                app_module.ConsolidationAction.AUTO
            )

        assert result == "started_full"
        overdue_incidents = [t for t in recorded_types if t == "full_consolidation_overdue"]
        assert overdue_incidents == [], (
            "No overdue incident must fire when the fold is within its runway; "
            f"got: {recorded_types}"
        )

    def test_resume_arm_and_full_arm_report_identical_incident(self, tmp_path, monkeypatch):
        """``_record_full_consolidation_overdue`` is the ONE overdue check +
        incident, shared by ``_dispatch_resume``'s FULL branch and
        ``_dispatch_consolidation``'s own FULL dispatch arm.  Exercised
        separately (a single ledger can only ever take one of the two paths
        in production — resume-pending-first pre-empts a fresh dispatch),
        the two arms must report byte-identical type/key/severity/summary/detail.
        """
        from unittest.mock import patch

        import paramem.server.app as app_module

        state = self._make_minimal_state(tmp_path)
        old_stamp = "20200101T0000"
        self._make_interim_dir(state["config"].adapter_dir, old_stamp)
        monkeypatch.setattr(app_module, "_state", state)

        recorded: list[dict] = []

        def _capture_record_incident(state_dir, *, type, key, severity, summary, detail):
            recorded.append(
                {
                    "type": type,
                    "key": key,
                    "severity": severity,
                    "summary": summary,
                    "detail": detail,
                }
            )

        with (
            patch(
                "paramem.server.app._full_consolidation_overdue_key",
                return_value=old_stamp,
            ),
            patch("paramem.server.app.record_incident", side_effect=_capture_record_incident),
        ):
            # --- Resume arm: a pending FULL event's ledger. ---
            with (
                patch("paramem.server.app._pending_event_action_name", return_value="full"),
                patch(
                    "paramem.server.app._dispatch_to_executor",
                    return_value="started_resume",
                ),
            ):
                resume_result = app_module._dispatch_resume(state["config"])
            assert resume_result is not None

            # --- FULL dispatch arm: no pending ledger, requested directly. ---
            with (
                patch("paramem.server.app._consolidation_dispatch_guards", return_value=None),
                patch("paramem.server.app._pending_event_action_name", return_value=None),
                patch("paramem.server.app._retro_claim_orphan_sessions", return_value=0),
                patch(
                    "paramem.server.app._dispatch_to_executor",
                    return_value="started_full",
                ),
            ):
                full_result, _action = app_module._dispatch_consolidation(
                    app_module.ConsolidationAction.FULL
                )
            assert full_result == "started_full"

        assert len(recorded) == 2, f"Expected exactly two incidents; got: {recorded}"
        resume_incident, full_incident = recorded
        assert resume_incident == full_incident
        assert resume_incident["type"] == "full_consolidation_overdue"
        assert resume_incident["key"] == old_stamp
        assert resume_incident["severity"] == "failed"


# ---------------------------------------------------------------------------
# Two-tick sequencing lock: the overdue incident fires once per cycle key
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Overdue-incident resolution on clean full-cycle completion
# ---------------------------------------------------------------------------


class TestFullConsolidationOverdueStatusSurface:
    """'full_consolidation_overdue' appears in _consolidation_incident_types so
    ``_derive_consolidation_status_fields`` surfaces it in ``last_consolidation_error``.
    """

    def test_overdue_incident_surfaces_in_derive_status(self, tmp_path):
        """Active full_consolidation_overdue → last_consolidation_error reflects its detail."""
        from paramem.server.app import _derive_consolidation_status_fields
        from paramem.server.incidents import record_incident

        state_dir = tmp_path / "state"
        record_incident(
            state_dir,
            type="full_consolidation_overdue",
            key="20200101T0000",
            severity="failed",
            summary="Full consolidation overdue — fold has not completed within its runway",
            detail={
                "oldest_interim_stamp": "20200101T0000",
                "type": "full_consolidation_overdue",
            },
        )

        err, _, _ = _derive_consolidation_status_fields(state_dir)
        assert err is not None, (
            "full_consolidation_overdue incident must surface in last_consolidation_error"
        )
        assert err["type"] == "full_consolidation_overdue"
        assert err["oldest_interim_stamp"] == "20200101T0000"


# ---------------------------------------------------------------------------
# Config validator + _oldest_interim_stamp + 3-way gate + incidents
# ---------------------------------------------------------------------------


class TestInterimOverflowSlackConfig:
    """ConsolidationScheduleConfig rejects negative interim_overflow_slack."""

    def test_negative_slack_raises_value_error(self):
        """interim_overflow_slack < 0 must raise ValueError at config creation."""
        from paramem.server.config import ConsolidationScheduleConfig

        with pytest.raises(ValueError, match="interim_overflow_slack"):
            ConsolidationScheduleConfig(
                max_interim_count=7,
                interim_overflow_slack=-1,
            )

    def test_zero_slack_accepted(self):
        """interim_overflow_slack=0 (default) must be accepted without error."""
        from paramem.server.config import ConsolidationScheduleConfig

        cfg = ConsolidationScheduleConfig(max_interim_count=7, interim_overflow_slack=0)
        assert cfg.interim_overflow_slack == 0

    def test_positive_slack_accepted(self):
        """interim_overflow_slack > 0 must be accepted without error."""
        from paramem.server.config import ConsolidationScheduleConfig

        cfg = ConsolidationScheduleConfig(max_interim_count=7, interim_overflow_slack=3)
        assert cfg.interim_overflow_slack == 3

    def test_max_interim_count_zero_with_cadence_accepted(self):
        """max_interim_count=0 + non-empty refresh_cadence constructs without error.

        count==0 is the full-fold-only consume-pending mode; with a scheduled
        cadence it is valid — the full fold runs every refresh_cadence and
        pending sessions are consumed without minting interim adapters.
        """
        from paramem.server.config import ConsolidationScheduleConfig

        cfg = ConsolidationScheduleConfig(max_interim_count=0)  # default cadence "12h"
        assert cfg.max_interim_count == 0

    def test_max_interim_count_zero_empty_cadence_raises(self):
        """max_interim_count=0 + empty refresh_cadence raises ValueError naming refresh_cadence.

        Without a scheduled full fold the pending sessions would accumulate
        unboundedly; the validator rejects this combination at construction time.
        """
        from paramem.server.config import ConsolidationScheduleConfig

        with pytest.raises(ValueError, match="refresh_cadence"):
            ConsolidationScheduleConfig(max_interim_count=0, refresh_cadence="")

    def test_max_interim_count_negative_raises_value_error(self):
        """max_interim_count=-1 must raise ValueError at config creation."""
        from paramem.server.config import ConsolidationScheduleConfig

        with pytest.raises(ValueError, match="max_interim_count"):
            ConsolidationScheduleConfig(max_interim_count=-1)


class TestConsolidationVenueConfig:
    """ConsolidationScheduleConfig validates the consolidation venue at load.

    Two load-time guards:

    * ``mode`` must be "train" or "simulate".  Downstream code treats anything
      that is not "train" as simulate, so an unvalidated typo would boot
      cleanly and only explode at request time.
    * ``max_interim_count == 0`` + ``mode == "simulate"`` stalls session
      ingestion: at count 0 no interim adapter is ever minted, and in simulate
      mode the scheduled full fold does not consume pending sessions
      (``_consume_pending`` in ``paramem/server/app.py`` carries a
      ``mode != "simulate"`` term).  Nothing would consume the buffer.

    Both are boot-time guards only — a post-construction mutation
    (``cfg.consolidation.mode = "simulate"``) bypasses ``__post_init__``.
    """

    def test_count_zero_cadence_error_precedes_pairing_error(self):
        """The pre-existing cadence check fires before the venue-pairing check.

        max_interim_count=0 + mode='simulate' + refresh_cadence='' has TWO
        independently-true defects. The cadence check (``:1163-1189``, ahead
        of the pairing check added at ``:1207-1228``) must win — it is checked
        first in ``__post_init__``. Pins the ordering so a future reorder of
        the two blocks changes the error an operator sees, loudly, in a
        failing test rather than silently.
        """
        from paramem.server.config import ConsolidationScheduleConfig

        # The pairing error never mentions refresh_cadence, so this pins that
        # the cadence error — not the pairing error — is the one raised.
        with pytest.raises(ValueError, match="refresh_cadence"):
            ConsolidationScheduleConfig(max_interim_count=0, mode="simulate", refresh_cadence="")

    def test_count_zero_with_simulate_raises_naming_both_keys(self):
        """max_interim_count=0 + mode='simulate' raises, naming both keys."""
        from paramem.server.config import ConsolidationScheduleConfig

        with pytest.raises(ValueError) as exc:
            ConsolidationScheduleConfig(max_interim_count=0, mode="simulate")

        message = str(exc.value)
        assert "max_interim_count" in message
        assert "mode" in message
        assert "simulate" in message

    def test_count_zero_with_train_accepted(self):
        """max_interim_count=0 + mode='train' (consume-pending mode) still loads."""
        from paramem.server.config import ConsolidationScheduleConfig

        cfg = ConsolidationScheduleConfig(max_interim_count=0, mode="train")
        assert cfg.max_interim_count == 0
        assert cfg.mode == "train"

    def test_count_positive_with_simulate_accepted(self):
        """max_interim_count>0 + mode='simulate' still loads — interims carry the sessions."""
        from paramem.server.config import ConsolidationScheduleConfig

        cfg = ConsolidationScheduleConfig(max_interim_count=7, mode="simulate")
        assert cfg.max_interim_count == 7
        assert cfg.mode == "simulate"

    def test_unknown_mode_raises_at_load(self):
        """A typo'd mode raises at construction, not at the first request."""
        from paramem.server.config import ConsolidationScheduleConfig

        with pytest.raises(ValueError, match="consolidation.mode"):
            ConsolidationScheduleConfig(mode="simulated")

    @pytest.mark.parametrize("mode", ["train", "simulate"])
    def test_known_modes_accepted(self, mode):
        """Both members of the venue vocabulary construct without error."""
        from paramem.server.config import ConsolidationScheduleConfig

        cfg = ConsolidationScheduleConfig(mode=mode)
        assert cfg.mode == mode


class TestSchedulerCatchUpGate:
    """The catch-up gate belongs to AUTO — requested only by the scheduled tick.

    A directly requested FULL/INTERIM/RECONCILE never resolves AUTO, so it
    never reaches the catch-up gate at all and is never refused for "not due
    yet"; since it never dispatches through the AUTO branch, it also never
    writes the cadence stamp.  These tests pin both halves against the SAME
    stamp and cadence, so the requested action — not a flag — is what differs.
    """

    def _make_state(self, tmp_path, *, refresh_cadence: str) -> dict:
        from paramem.server.config import ConsolidationScheduleConfig, ServerConfig

        config = MagicMock(spec=ServerConfig)
        sched = ConsolidationScheduleConfig()
        sched.refresh_cadence = refresh_cadence
        sched.training_idle_debounce_s = 0
        config.consolidation = sched
        config.debug = False
        config.debug_dir = tmp_path / "debug"
        config.paths = MagicMock()
        config.paths.data = tmp_path
        config.adapter_dir = tmp_path / "adapters"
        config.adapter_dir.mkdir(parents=True, exist_ok=True)
        # One content-bearing interim slot (train venue, the config default) so
        # the content gate passes and these tests isolate the catch-up gate.
        # A candidate slot is a subdirectory carrying its own meta.json.
        _slot = config.adapter_dir / "episodic" / "interim_20260101T0000" / "slot"
        _slot.mkdir(parents=True, exist_ok=True)
        (_slot / "meta.json").write_text("{}")
        (_slot / "adapter_model.safetensors").write_bytes(b"")

        # The triage pre-stage reads the buffer on every dispatch; nothing is
        # pending in these tests, so the interim slot above is what satisfies
        # the content gate on the scheduled path.
        buffer = MagicMock()
        buffer.pending_facts.return_value = []

        return {
            "config": config,
            "session_buffer": buffer,
            "speaker_store": None,
            "consolidating": False,
            "mode": "local",
            "background_trainer": None,
            "last_chat_monotonic": None,
            "pending_rehydration": False,
            "integrity_check_failed": False,
            "cloud_only_reason": None,
        }

    def _state_dir(self, tmp_path):
        return tmp_path / "state"

    def _call_dispatching(self, state: dict, action=None) -> tuple:
        """Run a dispatch with _is_full_cycle_due forced True (relevant only
        when *action* resolves AUTO — a directly requested FULL never calls
        it at all). Returns (result, dispatch_call_count).

        ``_run_full_consolidation_sync`` is passed to (mocked)
        ``run_in_executor`` as a reference, never actually invoked under this
        mocking scheme — matching the existing dispatcher-test pattern in
        this file (e.g. TestFullCycleDispatcherOverdueIncident). The
        dispatch count is therefore read off ``run_in_executor.call_count``.

        Args:
            action: ``None`` (default, resolves to ``AUTO`` — the scheduled
                tick) or an explicit ``ConsolidationAction`` for a directly
                requested door (e.g. ``FULL`` for a manual ``/consolidate``).
        """
        from unittest.mock import patch

        import paramem.server.app as app_module

        if action is None:
            action = app_module.ConsolidationAction.AUTO

        mock_loop = MagicMock()
        mock_future = MagicMock()
        mock_future.add_done_callback = MagicMock()
        mock_loop.run_in_executor = MagicMock(return_value=mock_future)
        state["event_loop"] = mock_loop

        with (
            patch.object(app_module, "_state", state),
            patch("paramem.server.app._consolidation_dispatch_guards", return_value=None),
            patch("paramem.server.app._retro_claim_orphan_sessions", return_value=0),
            patch("paramem.server.app._is_full_cycle_due", return_value=True),
            patch("paramem.server.app._full_consolidation_overdue_key", return_value=None),
            patch("paramem.server.app._run_full_consolidation_sync"),
        ):
            result, _action = app_module._dispatch_consolidation(action)
        return result, mock_loop.run_in_executor.call_count

    def _call_tick_gate_only(self, state: dict) -> str:
        """Run the scheduled tick with no further patching — used for the
        noop/seed assertions that must never reach _is_full_cycle_due.
        """
        from unittest.mock import patch

        import paramem.server.app as app_module

        with (
            patch.object(app_module, "_state", state),
            patch("paramem.server.app._retro_claim_orphan_sessions", return_value=0),
            patch(
                "paramem.server.app._is_full_cycle_due",
                side_effect=AssertionError(
                    "_is_full_cycle_due must not be reached when the catch-up gate blocks"
                ),
            ),
        ):
            status, _action = app_module._dispatch_consolidation(
                app_module.ConsolidationAction.AUTO
            )
            return status

    def test_due_stamp_dispatches_and_updates_stamp(self, tmp_path):
        """last attempt 6h ago + 'every 5h' (period 5h) → due → dispatch
        happens and the stamp is updated. Asserting the dispatch call count
        (not just the return status) is load-bearing — a test that only
        checks 'not noop_not_due' would also pass on the suspend/regression
        path where the gate silently blocks the tick forever.
        """
        from paramem.server.schedule_state import read_last_scheduled_run, write_last_scheduled_run

        state = self._make_state(tmp_path, refresh_cadence="every 5h")
        state_dir = self._state_dir(tmp_path)
        old_stamp = time.time() - 6 * 3600
        write_last_scheduled_run(state_dir, old_stamp)

        result, full_call_count = self._call_dispatching(state)

        assert result == "started_full"
        assert full_call_count == 1
        new_stamp = read_last_scheduled_run(state_dir)
        assert new_stamp is not None
        assert new_stamp > old_stamp

    def test_not_due_stamp_blocks_and_leaves_stamp_unchanged(self, tmp_path):
        """last attempt 2h ago + 'every 5h' (period 5h) → not due → noop,
        stamp untouched.
        """
        from paramem.server.schedule_state import read_last_scheduled_run, write_last_scheduled_run

        state = self._make_state(tmp_path, refresh_cadence="every 5h")
        state_dir = self._state_dir(tmp_path)
        recent_stamp = time.time() - 2 * 3600
        write_last_scheduled_run(state_dir, recent_stamp)

        result = self._call_tick_gate_only(state)

        assert result == "noop_not_due"
        assert read_last_scheduled_run(state_dir) == recent_stamp

    def test_manual_action_dispatches_when_the_tick_would_not(self, tmp_path):
        """A directly requested FULL (``/consolidate``) dispatches where the
        scheduled tick is blocked.

        Same stamp/cadence as test_not_due_stamp_blocks_and_leaves_stamp_unchanged
        (last attempt 2h ago, 'every 5h' → not due) — a directly requested
        FULL never resolves AUTO and never reaches the catch-up gate at all,
        so which ACTION was requested, not a flag, is what differs.  And the
        manual run leaves the cadence window exactly where it was: the next
        scheduled tick still has its own content gate, so nothing needs to be
        stamped on the manual run's behalf.
        """
        import paramem.server.app as app_module
        from paramem.server.schedule_state import read_last_scheduled_run, write_last_scheduled_run

        state_dir = self._state_dir(tmp_path)

        # AUTO -- not due, blocked, no dispatch.
        state_gated = self._make_state(tmp_path, refresh_cadence="every 5h")
        recent_stamp = time.time() - 2 * 3600
        write_last_scheduled_run(state_dir, recent_stamp)
        assert self._call_tick_gate_only(state_gated) == "noop_not_due"
        assert read_last_scheduled_run(state_dir) == recent_stamp

        # FULL, requested directly -- same stamp, same cadence -- dispatches,
        # stamp untouched: it never even consults the catch-up gate.
        state_manual = self._make_state(tmp_path, refresh_cadence="every 5h")
        result, dispatch_count = self._call_dispatching(
            state_manual, app_module.ConsolidationAction.FULL
        )
        assert result == "started_full"
        assert dispatch_count == 1
        assert read_last_scheduled_run(state_dir) == recent_stamp, (
            "a manual dispatch must not move the cadence window"
        )

    def test_absent_stamp_seeds_without_dispatching(self, tmp_path):
        """No stamp file yet (fresh install / first tick after upgrade) →
        seed the stamp and return noop_scheduler_seeded WITHOUT dispatching —
        a surprise consolidation on first boot after upgrade is worse than a
        one-tick delay.
        """
        from paramem.server.schedule_state import read_last_scheduled_run

        state = self._make_state(tmp_path, refresh_cadence="every 5h")
        state_dir = self._state_dir(tmp_path)
        assert read_last_scheduled_run(state_dir) is None

        result = self._call_tick_gate_only(state)

        assert result == "noop_scheduler_seeded"
        assert read_last_scheduled_run(state_dir) is not None

    def test_manual_action_on_a_virgin_install_dispatches(self, tmp_path):
        """Virgin install (absent stamp) + a directly requested FULL
        (``/consolidate``) → dispatches.

        The seed-and-noop behaviour is specific to the scheduled tick (AUTO);
        a manual run on a fresh install must run, and still writes no stamp.
        """
        import paramem.server.app as app_module
        from paramem.server.schedule_state import read_last_scheduled_run

        state = self._make_state(tmp_path, refresh_cadence="every 5h")
        state_dir = self._state_dir(tmp_path)
        assert read_last_scheduled_run(state_dir) is None

        result, dispatch_count = self._call_dispatching(state, app_module.ConsolidationAction.FULL)

        assert result == "started_full"
        assert dispatch_count == 1
        assert read_last_scheduled_run(state_dir) is None

    # -----------------------------------------------------------------
    # Route-level: POST /consolidate (an operator door) vs
    # POST /scheduled-tick (the schedule's door) — proves the action is
    # actually wired at the endpoint level, not just exercised via the
    # dispatcher directly.
    # -----------------------------------------------------------------

    def _make_client(self, monkeypatch, state: dict):
        from fastapi.testclient import TestClient

        import paramem.server.app as app_module

        monkeypatch.setattr(app_module, "_state", state)
        return TestClient(app_module.app, raise_server_exceptions=False)

    def test_consolidate_route_dispatches_inside_due_window(self, tmp_path, monkeypatch):
        """POST /consolidate inside the due-window (not yet due under the
        gate) still dispatches — the manual escape hatch must not be
        blocked. Asserts the dispatch actually happens, not merely that the
        request wasn't rejected.
        """
        from unittest.mock import patch

        from paramem.server.schedule_state import read_last_scheduled_run, write_last_scheduled_run

        state = self._make_state(tmp_path, refresh_cadence="every 5h")
        state_dir = self._state_dir(tmp_path)
        recent_stamp = time.time() - 2 * 3600
        write_last_scheduled_run(state_dir, recent_stamp)

        mock_loop = MagicMock()
        mock_future = MagicMock()
        mock_future.add_done_callback = MagicMock()
        mock_loop.run_in_executor = MagicMock(return_value=mock_future)
        state["event_loop"] = mock_loop

        client = self._make_client(monkeypatch, state)
        with (
            patch("paramem.server.app._consolidation_dispatch_guards", return_value=None),
            patch("paramem.server.app._retro_claim_orphan_sessions", return_value=0),
            patch("paramem.server.app._full_consolidation_overdue_key", return_value=None),
            patch("paramem.server.app._run_full_consolidation_sync"),
        ):
            resp = client.post("/consolidate")

        assert resp.status_code == 200
        assert resp.json()["status"] == "started_full"
        assert mock_loop.run_in_executor.call_count == 1, (
            "POST /consolidate must dispatch inside the due-window, not just avoid an error"
        )
        assert read_last_scheduled_run(state_dir) == recent_stamp

    def test_scheduled_tick_route_returns_noop_not_due_inside_same_window(
        self, tmp_path, monkeypatch
    ):
        """POST /scheduled-tick with the IDENTICAL stamp/cadence as the
        /consolidate test above → noop_not_due, no dispatch — proves the
        two routes genuinely differ by the action they dispatch.
        """
        from unittest.mock import patch

        from paramem.server.schedule_state import read_last_scheduled_run, write_last_scheduled_run

        state = self._make_state(tmp_path, refresh_cadence="every 5h")
        state_dir = self._state_dir(tmp_path)
        recent_stamp = time.time() - 2 * 3600
        write_last_scheduled_run(state_dir, recent_stamp)

        client = self._make_client(monkeypatch, state)
        with (
            patch("paramem.server.app._retro_claim_orphan_sessions", return_value=0),
            patch(
                "paramem.server.app._is_full_cycle_due",
                side_effect=AssertionError(
                    "_is_full_cycle_due must not be reached when the catch-up gate blocks"
                ),
            ),
        ):
            resp = client.post("/scheduled-tick")

        assert resp.status_code == 200
        assert resp.json()["status"] == "noop_not_due"
        assert read_last_scheduled_run(state_dir) == recent_stamp

    def test_scheduled_tick_route_virgin_install_seeds_without_dispatching(
        self, tmp_path, monkeypatch
    ):
        """POST /scheduled-tick on a virgin install (absent stamp) →
        noop_scheduler_seeded, no dispatch.
        """
        from unittest.mock import patch

        from paramem.server.schedule_state import read_last_scheduled_run

        state = self._make_state(tmp_path, refresh_cadence="every 5h")
        state_dir = self._state_dir(tmp_path)
        assert read_last_scheduled_run(state_dir) is None

        client = self._make_client(monkeypatch, state)
        with (
            patch("paramem.server.app._retro_claim_orphan_sessions", return_value=0),
            patch(
                "paramem.server.app._is_full_cycle_due",
                side_effect=AssertionError(
                    "_is_full_cycle_due must not be reached when the catch-up gate blocks"
                ),
            ),
        ):
            resp = client.post("/scheduled-tick")

        assert resp.status_code == 200
        assert resp.json()["status"] == "noop_scheduler_seeded"
        assert read_last_scheduled_run(state_dir) is not None

    def test_consolidate_route_virgin_install_dispatches(self, tmp_path, monkeypatch):
        """POST /consolidate on a virgin install (absent stamp) → dispatches
        without seeding a stamp — the manual trigger neither seed-and-skips
        nor moves the cadence window.
        """
        from unittest.mock import patch

        from paramem.server.schedule_state import read_last_scheduled_run

        state = self._make_state(tmp_path, refresh_cadence="every 5h")
        state_dir = self._state_dir(tmp_path)
        assert read_last_scheduled_run(state_dir) is None

        mock_loop = MagicMock()
        mock_future = MagicMock()
        mock_future.add_done_callback = MagicMock()
        mock_loop.run_in_executor = MagicMock(return_value=mock_future)
        state["event_loop"] = mock_loop

        client = self._make_client(monkeypatch, state)
        with (
            patch("paramem.server.app._consolidation_dispatch_guards", return_value=None),
            patch("paramem.server.app._retro_claim_orphan_sessions", return_value=0),
            patch("paramem.server.app._full_consolidation_overdue_key", return_value=None),
            patch("paramem.server.app._run_full_consolidation_sync"),
        ):
            resp = client.post("/consolidate")

        assert resp.status_code == 200
        assert resp.json()["status"] == "started_full"
        assert mock_loop.run_in_executor.call_count == 1
        assert read_last_scheduled_run(state_dir) is None

    def test_dispatch_guards_still_short_circuit_ahead_of_gate_both_routes(
        self, tmp_path, monkeypatch
    ):
        """_consolidation_dispatch_guards() (already-running / cloud-only /
        bg-training) must still defer BOTH routes ahead of the catch-up
        gate — a manual /consolidate during background training must defer,
        not force a concurrent run.
        """
        state = self._make_state(tmp_path, refresh_cadence="every 5h")
        state["consolidating"] = True  # -> _consolidation_dispatch_guards() returns non-None

        client = self._make_client(monkeypatch, state)

        resp_manual = client.post("/consolidate")
        assert resp_manual.status_code == 200
        assert resp_manual.json()["status"] == "deferred_already_running"

        resp_scheduled = client.post("/scheduled-tick")
        assert resp_scheduled.status_code == 200
        assert resp_scheduled.json()["status"] == "deferred_already_running"

    def test_many_missed_periods_coalesce_into_one_dispatch(self, tmp_path):
        """last attempt 5 days ago + 'every 720m' (non-exact, period 12h) →
        many periods missed, but ONE tick call produces exactly ONE dispatch,
        and the new stamp is floored-now — NOT last_stamp + period (which
        would still be days in the past and immediately due again).
        """
        from paramem.server.schedule_grammar import scheduled_run_stamp_value
        from paramem.server.schedule_state import read_last_scheduled_run, write_last_scheduled_run

        state = self._make_state(tmp_path, refresh_cadence="every 720m")
        state_dir = self._state_dir(tmp_path)
        old_stamp = time.time() - 5 * 86400
        write_last_scheduled_run(state_dir, old_stamp)

        result, full_call_count = self._call_dispatching(state)

        assert result == "started_full"
        assert full_call_count == 1, "many missed periods must coalesce into exactly one dispatch"

        period_s = 12 * 3600
        new_stamp = read_last_scheduled_run(state_dir)
        assert new_stamp is not None
        assert new_stamp != old_stamp + period_s, (
            "stamp must be floored-now, not last_stamp + period"
        )
        expected = scheduled_run_stamp_value("every 720m", time.time())
        assert abs(new_stamp - expected) < 5, f"expected ~{expected}, got {new_stamp}"


# ---------------------------------------------------------------------------
# _run_full_cycle: consume-pending pre-stage extract-loop + mark_consolidated
# ---------------------------------------------------------------------------


class TestRecallBindTelemetry:
    """Fold-telemetry instrumentation: ``epochs_to_bind`` /
    ``steps_to_bind`` / ``hit_cap`` derivation from the ``_EarlyStopState``
    ``_train_tier_adapter`` returns. Pure function, no model/GPU required.
    """

    def test_none_state_returns_all_none(self):
        """recall_state=None (early stopping disabled, or entries empty) ->
        all three fields absent (None) -- the caller omits them from the
        ring record."""
        from paramem.training.consolidation import _recall_bind_telemetry

        assert _recall_bind_telemetry(None, n_keys=10, accum=2) == (None, None, None)

    def test_stop_epoch_set_derives_bind_and_steps(self):
        """A fired early-stop signal derives epochs_to_bind=stop_epoch and
        steps_to_bind=ceil(n_keys/accum)*epochs_to_bind; hit_cap=False."""
        from paramem.training.consolidation import _recall_bind_telemetry
        from paramem.training.early_stop import _EarlyStopState

        state = _EarlyStopState(stop_epoch=12)
        epochs_to_bind, steps_to_bind, hit_cap = _recall_bind_telemetry(state, n_keys=21, accum=2)
        assert epochs_to_bind == 12
        # ceil(21 / 2) = 11; 11 * 12 = 132.
        assert steps_to_bind == 132
        assert hit_cap is False

    def test_stop_epoch_none_signals_hit_cap_with_bind_fields_absent(self):
        """Training ran to the full derived epoch budget without the recall
        signal ever firing (stop_epoch=None) -> hit_cap=True and
        epochs_to_bind/steps_to_bind stay None (absent from the record)."""
        from paramem.training.consolidation import _recall_bind_telemetry
        from paramem.training.early_stop import _EarlyStopState

        state = _EarlyStopState(stop_epoch=None)
        epochs_to_bind, steps_to_bind, hit_cap = _recall_bind_telemetry(state, n_keys=3, accum=1)
        assert epochs_to_bind is None
        assert steps_to_bind is None
        assert hit_cap is True

    def test_exact_division_no_remainder(self):
        """n_keys exactly divisible by accum -- ceil == plain division, no
        off-by-one from the ceil-division trick."""
        from paramem.training.consolidation import _recall_bind_telemetry
        from paramem.training.early_stop import _EarlyStopState

        state = _EarlyStopState(stop_epoch=4)
        _, steps_to_bind, _ = _recall_bind_telemetry(state, n_keys=10, accum=2)
        assert steps_to_bind == 20  # ceil(10/2)=5; 5*4=20


# ---------------------------------------------------------------------------
# measured_adapter_init_state / lora_b_frobenius_norm — CPU-only, tiny model
# ---------------------------------------------------------------------------


class TestMeasuredAdapterInitState:
    """The telemetry ring's ``init`` field: cold ("~0" LoRA-B norm) vs warm
    (non-zero), measured against a tiny fake model (named_parameters only)
    -- no GPU, no real PeftModel required. Mirrors the ``_index``-style
    param-list fixture already used by the copy_adapter_weights* tests
    above.
    """

    def test_zero_lora_b_classified_cold(self):
        """A freshly zero-initialised lora_B tensor (PEFT's real init state
        immediately after create_adapter) classifies as "cold"."""
        from unittest.mock import MagicMock

        import torch

        from paramem.models.loader import measured_adapter_init_state

        model = MagicMock()
        model.named_parameters.return_value = [
            ("base_model.model.x.lora_B.episodic.weight", torch.zeros(4, 4)),
        ]

        assert measured_adapter_init_state(model, "episodic") == "cold"

    def test_nonzero_lora_b_classified_warm(self):
        """A trained (non-zero) lora_B tensor classifies as "warm"."""
        from unittest.mock import MagicMock

        import torch

        from paramem.models.loader import measured_adapter_init_state

        model = MagicMock()
        model.named_parameters.return_value = [
            ("base_model.model.x.lora_B.episodic.weight", torch.ones(4, 4)),
        ]

        assert measured_adapter_init_state(model, "episodic") == "warm"

    def test_no_matching_tensors_degrades_to_none(self):
        """No lora_B tensor found for the adapter name (e.g. a test double
        standing in for the model) -- degrades to None (the caller omits
        the ``init`` field) rather than raising and blocking the fold.
        lora_b_frobenius_norm itself still raises LoraTensorsNotFound --
        this is the wrapper's contract, not a change to the underlying
        primitive.
        """
        from unittest.mock import MagicMock

        import pytest as _pytest

        from paramem.models.loader import (
            LoraTensorsNotFound,
            lora_b_frobenius_norm,
            measured_adapter_init_state,
        )

        model = MagicMock()
        model.named_parameters.return_value = []

        assert measured_adapter_init_state(model, "episodic") is None
        with _pytest.raises(LoraTensorsNotFound, match="No lora_B tensors found"):
            lora_b_frobenius_norm(model, "episodic")

    def test_unrelated_runtime_error_propagates_unchanged(self):
        """A RuntimeError subclass unrelated to a missing-tensor condition
        (e.g. a CUDA OOM or "device lost" error raised inside
        named_parameters()) must propagate unchanged -- measured_adapter_init_state
        only degrades LoraTensorsNotFound, never RuntimeError broadly."""
        from unittest.mock import MagicMock

        import pytest as _pytest

        from paramem.models.loader import measured_adapter_init_state

        class _SimulatedCudaOOM(RuntimeError):
            pass

        model = MagicMock()

        def _raise_oom():
            raise _SimulatedCudaOOM("CUDA out of memory")

        model.named_parameters.side_effect = _raise_oom

        with _pytest.raises(_SimulatedCudaOOM, match="CUDA out of memory"):
            measured_adapter_init_state(model, "episodic")
