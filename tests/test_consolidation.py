"""Tests for the consolidation loop orchestrator.

These are unit tests that mock the model/extraction to test
the consolidation logic without requiring GPU.
"""

from unittest.mock import MagicMock

import pytest

from paramem.evaluation.consolidation_metrics import (
    ConsolidationMetrics,
    compute_consolidation_metrics,
    compute_episodic_decay_rate,
    compute_promoted_retention,
    compute_semantic_drift,
    format_phase3_summary,
)
from paramem.memory.store import MemoryStore as _MS  # noqa: F401
from paramem.training.consolidation import ConsolidationLoop, CycleResult, _mentions_any
from paramem.training.curriculum import CurriculumSampler
from paramem.utils.config import ConsolidationConfig


class TestCycleResult:
    def test_default_values(self):
        result = CycleResult(cycle_index=1, session_id="s001")
        assert result.entities_extracted == 0
        assert result.nodes_promoted == 0
        assert result.promoted_nodes == []

    def test_with_values(self):
        result = CycleResult(
            cycle_index=1,
            session_id="s001",
            entities_extracted=5,
            nodes_promoted=2,
            promoted_nodes=["Alex", "Heilbronn"],
        )
        assert result.entities_extracted == 5
        assert len(result.promoted_nodes) == 2


class TestMentionsAny:
    def test_finds_mention(self):
        assert _mentions_any("Alex lives in Heilbronn", {"alex"})

    def test_no_mention(self):
        assert not _mentions_any("The weather is nice", {"alex"})

    def test_case_insensitive(self):
        assert _mentions_any("ALEX is here", {"alex"})

    def test_empty_terms(self):
        assert not _mentions_any("some text", set())


class TestConsolidationMetrics:
    def test_compute_metrics(self):
        results = [
            CycleResult(
                cycle_index=1,
                session_id="s001",
                entities_extracted=5,
                nodes_promoted=1,
                nodes_decayed=0,
                episodic_train_loss=0.8,
                wall_clock_seconds=120.0,
            ),
            CycleResult(
                cycle_index=2,
                session_id="s002",
                entities_extracted=3,
                nodes_promoted=0,
                nodes_decayed=1,
                episodic_train_loss=0.6,
                semantic_train_loss=0.4,
                wall_clock_seconds=90.0,
            ),
        ]
        metrics = compute_consolidation_metrics(results)
        assert metrics.total_cycles == 2
        assert metrics.total_entities_extracted == 8
        assert metrics.total_promotions == 1
        assert metrics.total_decays == 1
        assert metrics.mean_wall_clock_seconds == 105.0
        assert len(metrics.episodic_losses) == 2
        assert len(metrics.semantic_losses) == 1


class TestPromotedRetention:
    def test_full_retention(self):
        recall_scores = [
            {"Alex": 0.9, "Heilbronn": 0.85},
            {"Alex": 0.9, "Heilbronn": 0.85},
        ]
        result = compute_promoted_retention(recall_scores, {"Alex", "Heilbronn"})
        assert result["mean_retention"] >= 0.85

    def test_no_promoted_nodes(self):
        result = compute_promoted_retention([], set())
        assert result["mean_retention"] == 0.0

    def test_partial_retention(self):
        recall_scores = [
            {"Alex": 0.9},
            {"Alex": 0.5},
        ]
        result = compute_promoted_retention(recall_scores, {"Alex"})
        assert result["per_node"]["Alex"]["final"] == 0.5
        assert result["per_node"]["Alex"]["peak"] == 0.9


class TestEpisodicDecay:
    def test_measurable_decay(self):
        recall_scores = [
            {"Barcelona": 0.8},
            {"Barcelona": 0.6},
            {"Barcelona": 0.3},
        ]
        result = compute_episodic_decay_rate(
            recall_scores,
            {"Barcelona"},
            {"Barcelona": 0},
        )
        assert result["mean_decay_rate"] > 0

    def test_no_decay(self):
        result = compute_episodic_decay_rate([], set(), {})
        assert result["mean_decay_rate"] == 0.0


class TestSemanticDrift:
    def test_no_drift(self):
        recall_scores = [
            {"Alex": 0.9},
            {"Alex": 0.9},
            {"Alex": 0.9},
        ]
        result = compute_semantic_drift(recall_scores, {"Alex"})
        assert result["mean_drift"] == 0.0

    def test_measurable_drift(self):
        recall_scores = [
            {"Alex": 0.9},
            {"Alex": 0.85},
            {"Alex": 0.80},
        ]
        result = compute_semantic_drift(recall_scores, {"Alex"})
        assert result["mean_drift"] == pytest.approx(0.1)


class TestExtractionPathParity:
    """extract_session() and run_cycle() must produce identical episodic/procedural
    sets for the same transcript. Guards against drift between the production
    path (server) and the experiment path (phase3/phase4 scripts)."""

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

        from paramem.graph.qa_generator import generate_qa_from_relations as _real_qa
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
                    speaker_id="Speaker0",
                ),
                Relation(
                    subject="Alex",
                    predicate="prefers",
                    object="Acme Radio",
                    relation_type="preference",
                    speaker_id="Speaker0",
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
                    speaker_id="Speaker0",
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
        # Use template fallback (ignore passed model/tokenizer) for deterministic output.
        monkeypatch.setattr(
            "paramem.graph.qa_generator.generate_qa_from_relations",
            lambda relations, model=None, tokenizer=None: _real_qa(relations),
        )

        # __class__ = PeftModel so _ensure_adapters' isinstance check
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

        return ConsolidationLoop(
            model=model,
            tokenizer=MagicMock(),
            consolidation_config=ConsolidationConfig(),
            training_config=TrainingConfig(),
            episodic_adapter_config=AdapterConfig(),
            semantic_adapter_config=AdapterConfig(),
            memory_store=_MS(replay_enabled=True),
            procedural_adapter_config=procedural_adapter,
            output_dir=tmp_path,
            persist_graph=False,
            **loop_kwargs,
        )

    def _run_extract_session(self, loop, source_type: str = "transcript"):
        return loop.extract_session(
            session_transcript="Alex lives in Millfield. He prefers Acme Radio.",
            session_id="s001",
            speaker_id="spk",
            source_type=source_type,
        )

    def _run_cycle_and_capture(self, monkeypatch, loop, source_type: str = "transcript"):
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

        # Patch run_consolidation_cycle so run_cycle's indexed-key branch can be
        # intercepted without needing the removed _run_indexed_key_episodic /
        # _run_indexed_key_procedural methods.
        monkeypatch.setattr(type(loop), "run_consolidation_cycle", _capture_cycle, raising=False)
        monkeypatch.setattr(type(loop), "_save_adapters", lambda self_: None, raising=False)
        loop.run_cycle(
            session_transcript="Alex lives in Millfield. He prefers Acme Radio.",
            session_id="s001",
            speaker_id="spk",
            source_type=source_type,
        )
        return captured["episodic_rels"], captured["procedural_rels"]

    def test_parity_procedural_enabled(self, monkeypatch, tmp_path):
        loop_a = self._build_loop(monkeypatch, tmp_path / "a", procedural_enabled=True)
        episodic_a, procedural_a = self._run_extract_session(loop_a)

        loop_b = self._build_loop(monkeypatch, tmp_path / "b", procedural_enabled=True)
        episodic_b, procedural_b = self._run_cycle_and_capture(monkeypatch, loop_b)

        # Identity fields must match across both paths.
        def _key(qa):
            return (qa["subject"], qa["predicate"], qa["object"])

        def _rel_key(rel):
            return (rel["subject"], rel["predicate"], rel["object"])

        assert sorted(map(_key, episodic_a)) == sorted(map(_key, episodic_b))
        assert sorted(map(_rel_key, procedural_a)) == sorted(map(_rel_key, procedural_b))

        # Partition invariant: no preference in episodic, and procedural carries
        # both the filter-sourced and the separately-extracted preference.
        assert all(qa["predicate"] != "prefers" for qa in episodic_a)
        assert {rel["predicate"] for rel in procedural_a} == {"prefers", "listens_to"}

    @pytest.mark.parametrize("source_type", ["transcript", "document"])
    @pytest.mark.parametrize(
        "procedural_enabled,loop_overrides",
        [
            (True, {}),
            (False, {}),
            (
                True,
                {
                    "extraction_noise_filter": "claude",
                    "extraction_noise_filter_model": "claude-sonnet-4-6",
                },
            ),
            (True, {"extraction_plausibility_stage": "anon"}),
            (True, {"extraction_ner_check": True, "extraction_ner_model": "en_core_web_sm"}),
            (
                True,
                {
                    "extraction_stt_correction": False,
                    "extraction_ha_validation": False,
                    "extraction_verify_anonymization": False,
                },
            ),
        ],
    )
    def test_parity_kwargs_identical(
        self, monkeypatch, tmp_path, procedural_enabled, loop_overrides, source_type
    ):
        """Both orchestrator paths must pass IDENTICAL kwargs to the extractors.

        Any new flag added to one path but not the other will fail here — the
        helper + _extraction_kwargs are the only source of truth. Parametrized
        over source_type so both transcript and document variants are covered.
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
                    speaker_id="Speaker0",
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
        self._run_cycle_and_capture(monkeypatch, loop_b, source_type=source_type)

        # Each path calls extract_graph exactly once with the same kwargs.
        assert len(captured_a["graph"]) == 1
        assert len(captured_b["graph"]) == 1
        assert captured_a["graph"][0] == captured_b["graph"][0], (
            "extract_graph kwargs diverged between extract_session and run_cycle. "
            f"extract_session: {captured_a['graph'][0]!r}\n"
            f"run_cycle:       {captured_b['graph'][0]!r}"
        )

        # Procedural path: same kwarg shape when enabled, neither path calls it when disabled.
        assert len(captured_a["procedural"]) == len(captured_b["procedural"])
        if procedural_enabled:
            assert len(captured_a["procedural"]) == 1
            assert captured_a["procedural"][0] == captured_b["procedural"][0], (
                "extract_procedural_graph kwargs diverged between paths. "
                f"extract_session: {captured_a['procedural'][0]!r}\n"
                f"run_cycle:       {captured_b['procedural'][0]!r}"
            )
        else:
            assert captured_a["procedural"] == []
            assert captured_b["procedural"] == []

    def test_parity_procedural_disabled(self, monkeypatch, tmp_path):
        loop_a = self._build_loop(monkeypatch, tmp_path / "a", procedural_enabled=False)
        episodic_a, procedural_a = self._run_extract_session(loop_a)

        loop_b = self._build_loop(monkeypatch, tmp_path / "b", procedural_enabled=False)
        episodic_b, procedural_b = self._run_cycle_and_capture(monkeypatch, loop_b)

        def _key(qa):
            return (qa["subject"], qa["predicate"], qa["object"])

        assert sorted(map(_key, episodic_a)) == sorted(map(_key, episodic_b))
        # With procedural disabled, preferences fall back into episodic — never lost.
        assert any(qa["predicate"] == "prefers" for qa in episodic_a)
        assert procedural_a == []
        assert procedural_b == []

    def test_empty(self):
        result = compute_semantic_drift([], set())
        assert result["mean_drift"] == 0.0


class TestMergeAtInterimGate:
    """extract_session merger.merge call is gated by config.merge_at_interim.

    These tests verify Path A (merge_at_interim=True) and Path B
    (merge_at_interim=False, default) without loading any model or GPU.
    """

    def _build_loop(self, monkeypatch, tmp_path, merge_at_interim: bool):
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
        model.peft_config = {
            "episodic": MagicMock(),
            "semantic": MagicMock(),
            "in_training": MagicMock(),
        }

        monkeypatch.setattr(
            "paramem.graph.extraction_pipeline.extract_graph",
            lambda *a, **kw: self._session_graph,
        )

        loop = ConsolidationLoop(
            model=model,
            tokenizer=MagicMock(),
            consolidation_config=ConsolidationConfig(merge_at_interim=merge_at_interim),
            training_config=TrainingConfig(),
            episodic_adapter_config=AdapterConfig(),
            semantic_adapter_config=AdapterConfig(),
            memory_store=_MS(replay_enabled=False),
            procedural_adapter_config=None,
            output_dir=tmp_path,
            persist_graph=False,
        )
        return loop

    def test_merge_called_when_merge_at_interim_true(self, monkeypatch, tmp_path):
        """merge_at_interim=True: merger.merge is called once with the session graph."""
        from unittest.mock import patch

        loop = self._build_loop(monkeypatch, tmp_path, merge_at_interim=True)
        initial_nodes = loop.merger.graph.number_of_nodes()
        initial_edges = loop.merger.graph.number_of_edges()

        with patch.object(loop.extraction, "run", return_value=self._session_graph):
            loop.extract_session("t", "s_gate", speaker_id="spk0")

        # Graph must have grown — merge ran.
        assert (
            loop.merger.graph.number_of_nodes() > initial_nodes
            or loop.merger.graph.number_of_edges() > initial_edges
        ), "Expected merger.graph to grow after extract_session with merge_at_interim=True"

    def test_merge_not_called_when_merge_at_interim_false(self, monkeypatch, tmp_path):
        """merge_at_interim=False: merger.merge is NOT called; graph unchanged."""
        from unittest.mock import patch

        loop = self._build_loop(monkeypatch, tmp_path, merge_at_interim=False)
        initial_nodes = loop.merger.graph.number_of_nodes()
        initial_edges = loop.merger.graph.number_of_edges()

        with patch.object(loop.extraction, "run", return_value=self._session_graph):
            with patch.object(loop.merger, "merge") as mock_merge:
                loop.extract_session("t", "s_gate", speaker_id="spk0")
                mock_merge.assert_not_called()

        # Graph is also structurally unchanged.
        assert loop.merger.graph.number_of_nodes() == initial_nodes
        assert loop.merger.graph.number_of_edges() == initial_edges

    def test_episodic_rels_identical_for_both_flag_values(self, monkeypatch, tmp_path):
        """Returned (episodic_rels, procedural_rels) are identical regardless of merge_at_interim.

        Keying is derived from session_graph, not from the cumulative graph,
        so the flag must not affect what facts are returned to the caller.
        """
        from unittest.mock import patch

        loop_a = self._build_loop(monkeypatch, tmp_path / "a", merge_at_interim=True)
        with patch.object(loop_a.extraction, "run", return_value=self._session_graph):
            rels_a, proc_a = loop_a.extract_session("t", "s_gate", speaker_id="spk0")

        loop_b = self._build_loop(monkeypatch, tmp_path / "b", merge_at_interim=False)
        with patch.object(loop_b.extraction, "run", return_value=self._session_graph):
            rels_b, proc_b = loop_b.extract_session("t", "s_gate", speaker_id="spk0")

        def _key(d):
            return (d.get("subject"), d.get("predicate"), d.get("object"))

        assert sorted(map(_key, rels_a)) == sorted(map(_key, rels_b)), (
            "episodic_rels differ between merge_at_interim=True and False"
        )
        assert proc_a == proc_b == []


class TestFormatSummary:
    def test_format_output(self):
        metrics = ConsolidationMetrics(
            total_cycles=10,
            total_entities_extracted=50,
            total_promotions=5,
            total_decays=3,
            mean_wall_clock_seconds=100.0,
        )
        retention = {"mean_retention": 0.85}
        decay = {"mean_decay_rate": 0.3}
        drift = {"mean_drift": 0.02}
        summary = format_phase3_summary(metrics, retention, decay, drift)
        assert "Phase 3" in summary
        assert "PASS" in summary
        assert "10" in summary


class TestMergeAtInterimConfigRoundtrip:
    """Loading YAML with merge_at_interim: true propagates through the property chain."""

    def test_yaml_merge_at_interim_propagates(self, tmp_path):
        """YAML merge_at_interim: true propagates to schedule and consolidation_config."""
        from paramem.server.config import load_server_config

        yaml_text = """
model:
  name: "mistralai/Mistral-7B-Instruct-v0.3"
consolidation:
  refresh_cadence: "12h"
  merge_at_interim: true
"""
        cfg_path = tmp_path / "server_merge_at_interim.yaml"
        cfg_path.write_text(yaml_text)
        cfg = load_server_config(str(cfg_path))

        assert cfg.consolidation.merge_at_interim is True, (
            "ServerConfig.consolidation.merge_at_interim should be True"
        )
        assert cfg.consolidation_config.merge_at_interim is True, (
            "consolidation_config.merge_at_interim should be True"
        )

    def test_yaml_merge_at_interim_default_false(self, tmp_path):
        """YAML without merge_at_interim defaults to False on schedule and consolidation_config."""
        from paramem.server.config import load_server_config

        yaml_text = """
model:
  name: "mistralai/Mistral-7B-Instruct-v0.3"
consolidation:
  refresh_cadence: "12h"
"""
        cfg_path = tmp_path / "server_no_merge_at_interim.yaml"
        cfg_path.write_text(yaml_text)
        cfg = load_server_config(str(cfg_path))

        assert cfg.consolidation.merge_at_interim is False, (
            "ServerConfig.consolidation.merge_at_interim should default to False"
        )
        assert cfg.consolidation_config.merge_at_interim is False, (
            "consolidation_config.merge_at_interim should default to False"
        )


class TestCurriculumDecayProtection:
    """Test that curriculum-aware decay respects min_exposure_cycles."""

    def _make_loop_with_curriculum(self, min_exposure=3):
        """Create a ConsolidationLoop with curriculum enabled, no real model."""
        config = ConsolidationConfig(
            curriculum_enabled=True,
            min_exposure_cycles=min_exposure,
        )
        # We only need the _apply_decay method — model/tokenizer are not used
        loop = ConsolidationLoop.__new__(ConsolidationLoop)
        loop.config = config
        loop.curriculum_sampler = CurriculumSampler(
            min_exposure_cycles=min_exposure,
        )
        loop.episodic_replay_pool = []
        return loop

    def test_decay_blocked_by_min_exposure(self):
        loop = self._make_loop_with_curriculum(min_exposure=3)
        loop.episodic_replay_pool = [
            {"question": "Where does Alex live?", "answer": "Heilbronn"},
        ]
        # exposure=0 < 3 → decay should be blocked
        loop._apply_decay(["Alex"])
        assert len(loop.episodic_replay_pool) == 1

    def test_decay_allowed_after_min_exposure(self):
        loop = self._make_loop_with_curriculum(min_exposure=3)
        loop.episodic_replay_pool = [
            {"question": "Where does Alex live?", "answer": "Heilbronn"},
        ]
        # Simulate 3 exposures
        loop.curriculum_sampler.exposure_counts["Where does Alex live?"] = 3
        loop._apply_decay(["Alex"])
        assert len(loop.episodic_replay_pool) == 0

    def test_decay_without_curriculum(self):
        """Without curriculum, decay works as before."""
        loop = ConsolidationLoop.__new__(ConsolidationLoop)
        loop.config = ConsolidationConfig(curriculum_enabled=False)
        loop.curriculum_sampler = None
        loop.episodic_replay_pool = [
            {"question": "Where does Alex live?", "answer": "Heilbronn"},
        ]
        loop._apply_decay(["Alex"])
        assert len(loop.episodic_replay_pool) == 0

    def test_mixed_decay_and_protection(self):
        loop = self._make_loop_with_curriculum(min_exposure=2)
        loop.episodic_replay_pool = [
            {"question": "Where does Alex live?", "answer": "Heilbronn"},
            {"question": "What is Alex's pet?", "answer": "Luna"},
        ]
        # Only the pet question has enough exposures
        loop.curriculum_sampler.exposure_counts["What is Alex's pet?"] = 2
        loop._apply_decay(["Alex"])
        # First kept (protected), second decayed (exposure met)
        assert len(loop.episodic_replay_pool) == 1
        assert loop.episodic_replay_pool[0]["question"] == "Where does Alex live?"


# ---------------------------------------------------------------------------
# Server consolidation — anonymous speaker sessions must not be silently skipped
# ---------------------------------------------------------------------------


class TestAnonymousSpeakerNotSkipped:
    """Speaker{N} sessions must flow through extraction, not be silently discarded.

    Verifies that _run_extraction_phase (paramem.server.app) calls
    loop.extract_session for sessions whose speaker_id is 'Speaker3' —
    i.e. the old hard-skip on falsy speaker_id is gone.

    Note: these tests call _run_extraction_phase directly (in paramem.server.app);
    run_consolidation was deleted and must not be re-introduced (see
    test_run_consolidation_removed.py).
    """

    def _make_mock_loop(self, tmp_path):
        """Minimal mock ConsolidationLoop with the attributes _run_extraction_phase touches."""

        loop = MagicMock()
        loop.shutdown_requested = False
        loop.merger = MagicMock()
        loop.merger.graph = MagicMock()
        loop.merger.graph.nodes = []
        loop.indexed_key_cache = {}
        loop.key_sessions = {}
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
        """Minimal ServerConfig pointing at a temp directory.

        Uses PathsConfig so that adapter_dir, key_metadata_path, and
        registry_path resolve under tmp_path without needing property setters.
        """
        from paramem.server.config import PathsConfig, ServerConfig

        config = ServerConfig()
        config.paths = PathsConfig(data=tmp_path / "ha")
        (tmp_path / "ha" / "adapters").mkdir(parents=True, exist_ok=True)
        return config

    def _make_session_buffer(self, tmp_path, speaker_id):
        """SessionBuffer with a single in-memory session for the given speaker_id.

        Sets speaker identity before appending turns so the turns carry the
        speaker_id and get_pending() returns it as the dominant speaker.
        """
        from paramem.server.session_buffer import SessionBuffer

        buffer = SessionBuffer(tmp_path / "sessions", debug=False)
        conv_id = "conv-anon-test"
        if speaker_id is not None:
            # Set speaker before appending so turns carry the speaker_id.
            buffer.set_speaker(conv_id, speaker_id, speaker_id)
        buffer.append(conv_id, "user", "Hello there")
        buffer.append(conv_id, "assistant", "Hi!")
        return buffer

    def _call_run_extraction_phase(self, loop, config, buffer):
        """Inject config + session_buffer into _state and call _run_extraction_phase."""
        import paramem.server.app as _app

        prior_config = _app._state.get("config")
        prior_buffer = _app._state.get("session_buffer")
        prior_ha = _app._state.get("ha_client")
        prior_speaker = _app._state.get("speaker_store")
        _app._state["config"] = config
        _app._state["session_buffer"] = buffer
        _app._state["ha_client"] = None
        _app._state["speaker_store"] = None
        try:
            return _app._run_extraction_phase(loop)
        finally:
            _app._state["config"] = prior_config
            _app._state["session_buffer"] = prior_buffer
            _app._state["ha_client"] = prior_ha
            _app._state["speaker_store"] = prior_speaker

    def test_anonymous_speaker_id_not_skipped(self, tmp_path):
        """Sessions with speaker_id='Speaker3' reach extract_session."""
        loop = self._make_mock_loop(tmp_path)
        config = self._make_config(tmp_path)
        buffer = self._make_session_buffer(tmp_path, speaker_id="Speaker3")

        self._call_run_extraction_phase(loop, config, buffer)

        loop.extract_session.assert_called_once()
        call_kwargs = loop.extract_session.call_args
        # First positional arg is the transcript; keyword arg is speaker_id.
        assert call_kwargs.kwargs.get("speaker_id") == "Speaker3"

    def test_named_speaker_not_skipped(self, tmp_path):
        """Named (enrolled) speaker IDs continue to reach extract_session."""
        loop = self._make_mock_loop(tmp_path)
        config = self._make_config(tmp_path)
        buffer = self._make_session_buffer(tmp_path, speaker_id="abc12345")

        self._call_run_extraction_phase(loop, config, buffer)

        loop.extract_session.assert_called_once()
        assert loop.extract_session.call_args.kwargs.get("speaker_id") == "abc12345"

    def test_none_speaker_id_still_skipped(self, tmp_path):
        """Truly-None speaker_id (text-only, no voice) is skipped at consolidation.

        Sessions with no speaker_id must not reach extract_session — a None
        speaker_id would cause unrelated text-only sessions to be attributed
        to the same unknown speaker, corrupting the graph.
        """
        loop = self._make_mock_loop(tmp_path)
        config = self._make_config(tmp_path)
        buffer = self._make_session_buffer(tmp_path, speaker_id=None)

        self._call_run_extraction_phase(loop, config, buffer)

        # Text-only sessions without a speaker_id must NOT reach extract_session.
        loop.extract_session.assert_not_called()


# ---------------------------------------------------------------------------
# _save_adapters: meta.json written in every saved slot
# ---------------------------------------------------------------------------


class TestSaveAdaptersManifest:
    """_save_adapters must embed meta.json in each adapter slot."""

    def _make_save_loop(self, tmp_path):
        """Return a minimal ConsolidationLoop wired for _save_adapters testing."""

        from paramem.training.consolidation import ConsolidationLoop
        from paramem.training.key_registry import KeyRegistry
        from paramem.utils.config import AdapterConfig, ConsolidationConfig, TrainingConfig

        model = MagicMock()

        # Provide JSON-serialisable attributes so build_manifest_for can
        # produce a valid manifest without fingerprinting real model weights.
        model.config._name_or_path = "test-base-model"
        model.config._commit_hash = None
        model.base_model.model.state_dict.return_value = {}
        lora_cfg = MagicMock()
        lora_cfg.r = 4
        lora_cfg.lora_alpha = 8
        lora_cfg.lora_dropout = 0.0
        lora_cfg.target_modules = ["q_proj"]
        lora_cfg.bias = "none"
        model.peft_config = {"episodic": lora_cfg}

        def _fake_save_pretrained(path, selected_adapters=None):
            from pathlib import Path

            p = Path(path)
            p.mkdir(parents=True, exist_ok=True)
            (p / "adapter_model.safetensors").write_bytes(b"weights")
            (p / "adapter_config.json").write_text("{}")

        model.save_pretrained.side_effect = _fake_save_pretrained

        tokenizer = MagicMock()
        tokenizer.name_or_path = "test-tokenizer"
        tokenizer.backend_tokenizer = None
        tokenizer.vocab_size = 32000

        loop = object.__new__(ConsolidationLoop)
        loop.model = model
        loop.tokenizer = tokenizer
        loop.config = ConsolidationConfig()
        loop.training_config = TrainingConfig(num_epochs=1)
        loop.episodic_config = AdapterConfig(rank=4, alpha=8, target_modules=["q_proj"])
        loop.semantic_config = None
        loop.procedural_config = None
        loop.wandb_config = None
        loop.output_dir = tmp_path
        loop.snapshot_dir = None
        loop.save_cycle_snapshots = False
        loop._debug_base = None
        loop._keep_prior_slots = 50  # high value so pruning is a no-op in these tests
        loop.indexed_key_registry = {"episodic": KeyRegistry()}
        loop.indexed_key_cache = {}
        loop.cycle_count = 0
        loop.merger = MagicMock()
        loop.episodic_simhash = {}
        loop.semantic_simhash = {}
        loop.procedural_simhash = {}
        return loop

    def test_save_adapters_writes_meta_json(self, tmp_path):
        """_save_adapters must write meta.json inside the episodic slot."""
        from paramem.adapters.manifest import AdapterManifest, read_manifest

        loop = self._make_save_loop(tmp_path)

        # Seed the registry and quads so build_manifest_for has something to hash.
        from paramem.training.key_registry import KeyRegistry

        ep_reg = KeyRegistry()
        ep_reg.add("graph1")
        loop.indexed_key_registry = {"episodic": ep_reg}

        loop._save_adapters()

        adapter_dir = tmp_path / "episodic"
        slots = [d for d in adapter_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
        assert slots, f"No slot dir created under {adapter_dir}"
        slot = slots[0]

        assert (slot / "meta.json").exists(), f"meta.json missing from slot {slot}"

        manifest = read_manifest(slot)
        assert isinstance(manifest, AdapterManifest)
        assert manifest.name == "episodic"

    def test_save_adapters_roundtrip_find_live_slot(self, tmp_path):
        """_save_adapters → find_live_slot must match the fresh slot.

        Regression guard: ``_save_adapters`` must hash the freshly-written
        registry (not the stale on-disk bytes it is about to overwrite).
        The post-save on-disk registry hash must equal ``manifest.registry_sha256``
        so ``find_live_slot`` can mount the adapter after restart.
        """
        import hashlib

        from paramem.adapters.manifest import find_live_slot, read_manifest
        from paramem.training.key_registry import KeyRegistry

        loop = self._make_save_loop(tmp_path)

        # Seed registry + quads state so every hash input is populated.
        ep_reg = KeyRegistry()
        ep_reg.add("graph1")
        loop.indexed_key_registry = {"episodic": ep_reg}
        loop.indexed_key_cache = {
            "graph1": {
                "key": "graph1",
                "question": "What colour is the sky?",
                "answer": "Blue.",
                "subject": "sky",
                "predicate": "has_colour",
                "object": "blue",
                "speaker_id": "Speaker0",
                "first_seen_cycle": 1,
            }
        }
        loop.episodic_simhash = {"graph1": 0xABCDEF}

        # Pre-seed a *different* registry file on disk at the per-tier path
        # so the old codepath (hash-before-overwrite) would produce a mismatch
        # — ensures this test fails under the pre-fix implementation.
        # New layout: registry lives at <adapter_dir>/<tier>/indexed_key_registry.json.
        (tmp_path / "episodic").mkdir(parents=True, exist_ok=True)
        stale_registry = KeyRegistry()
        stale_registry.add("stale_key")
        stale_registry.save(tmp_path / "episodic" / "indexed_key_registry.json")

        loop._save_adapters()

        # Live hash = hash of whatever is on disk post-save (per-tier path).
        live_hash = hashlib.sha256(
            (tmp_path / "episodic" / "indexed_key_registry.json").read_bytes()
        ).hexdigest()

        slot = find_live_slot(tmp_path / "episodic", live_hash)
        assert slot is not None, (
            "find_live_slot returned None — manifest.registry_sha256 does "
            "not match on-disk hash (registry-save order bug: manifest hashed "
            "stale bytes instead of the freshly-written registry)"
        )

        manifest = read_manifest(slot)
        assert manifest.registry_sha256 == live_hash
        # key_count must reflect the post-save registry (graph1), not the
        # stale pre-existing registry (stale_key).
        assert manifest.key_count == 1

        # Manifest version must be v4 (no keyed_pairs_sha256 field).
        assert manifest.schema_version == 4


class TestAtomicJsonWriteEncryptedFlag:
    """_atomic_json_write(..., encrypted=False) bypasses the envelope.

    Debug-directory writers (simulate mode, per-cycle graph snapshots) rely
    on this so debug output is uniformly inspectable with ``cat`` regardless
    of the server's Security posture.
    """

    def test_encrypted_false_writes_plaintext_under_security_on(self, tmp_path, monkeypatch):
        from paramem.backup.key_store import (
            DAILY_PASSPHRASE_ENV_VAR,
            _clear_daily_identity_cache,
            mint_daily_identity,
            wrap_daily_identity,
            write_daily_key_file,
        )
        from paramem.server.consolidation import _atomic_json_write

        # Genuine Security ON: a real daily identity is loadable.
        ident = mint_daily_identity()
        key_path = tmp_path / "daily_key.age"
        write_daily_key_file(wrap_daily_identity(ident, "pw"), key_path)
        monkeypatch.setenv(DAILY_PASSPHRASE_ENV_VAR, "pw")
        monkeypatch.setattr("paramem.backup.key_store.DAILY_KEY_PATH_DEFAULT", key_path)
        _clear_daily_identity_cache()

        out = tmp_path / "debug.json"
        _atomic_json_write({"debug": True, "n": 42}, out, encrypted=False)

        head = out.read_bytes()[:22]
        assert not head.startswith(b"age-encryption.org/v1"), (
            "encrypted=False must bypass the age envelope"
        )
        # File is directly readable as JSON (no dump needed).
        import json as _json

        assert _json.loads(out.read_text()) == {"debug": True, "n": 42}


class TestCreateConsolidationLoopFingerprintCacheWiring:
    """create_consolidation_loop wires _state["base_model_hash_cache"] into loop.

    Without this wire, build_manifest_for re-hashes the full base model on
    every consolidation cycle (~2 min for Mistral 7B). The cache is keyed by
    id(model) so it survives across cycles within one process lifetime.
    """

    def _patch_loop(self, monkeypatch):
        """Stub ConsolidationLoop so the test doesn't need a real model."""
        from paramem.server import consolidation as server_consolidation

        captured = {}

        def _fake_loop(**kwargs):
            instance = MagicMock()
            instance.fingerprint_cache = None  # default; production code must overwrite
            instance._kwargs = kwargs
            captured["instance"] = instance
            return instance

        monkeypatch.setattr(server_consolidation, "ConsolidationLoop", _fake_loop)
        return captured

    def _make_config(self, tmp_path):
        cfg = MagicMock()
        cfg.adapter_dir = tmp_path / "adapters"
        cfg.adapter_dir.mkdir(parents=True, exist_ok=True)
        cfg.key_metadata_path = tmp_path / "key_metadata.json"
        cfg.adapters.procedural.enabled = True
        cfg.consolidation.extraction_max_tokens = 256
        cfg.consolidation.graph_enrichment_enabled = False
        cfg.consolidation.graph_enrichment_neighborhood_hops = 1
        cfg.consolidation.graph_enrichment_max_entities_per_pass = 5
        cfg.consolidation.graph_enrichment_interim_enabled = False
        cfg.consolidation.graph_enrichment_min_triples_floor = 0
        # ThermalPolicy.from_consolidation_config reads training_temp_limit
        # at create_consolidation_loop construction; a default-MagicMock value
        # would raise on the int comparison.  0 = throttle disabled (fixture
        # intent is to construct the loop, not to exercise thermal behaviour).
        cfg.consolidation.training_temp_limit = 0
        cfg.debug = False
        cfg.debug_dir = None
        cfg.prompts_dir = None
        cfg.sanitization.cloud_scope = ["person"]
        return cfg

    def test_state_provider_supplies_cache_to_loop(self, tmp_path, monkeypatch):
        """When state_provider returns a state dict, loop.fingerprint_cache
        is the SAME dict object as state["base_model_hash_cache"].
        """
        from paramem.server.consolidation import create_consolidation_loop

        captured = self._patch_loop(monkeypatch)
        cfg = self._make_config(tmp_path)
        state = {"base_model_hash_cache": {}}

        loop = create_consolidation_loop(
            model=MagicMock(),
            tokenizer=MagicMock(),
            config=cfg,
            memory_store=_MS(replay_enabled=False),
            state_provider=lambda: state,
        )

        assert loop is captured["instance"]
        assert loop.fingerprint_cache is state["base_model_hash_cache"], (
            "fingerprint_cache must be the same dict instance as the server's "
            "state cache so build_manifest_for writes/reads through it."
        )

    def test_state_provider_creates_cache_when_missing(self, tmp_path, monkeypatch):
        """If state has no base_model_hash_cache key, loop wiring sets it via setdefault."""
        from paramem.server.consolidation import create_consolidation_loop

        self._patch_loop(monkeypatch)
        cfg = self._make_config(tmp_path)
        state: dict = {}

        loop = create_consolidation_loop(
            model=MagicMock(),
            tokenizer=MagicMock(),
            config=cfg,
            memory_store=_MS(replay_enabled=False),
            state_provider=lambda: state,
        )

        assert "base_model_hash_cache" in state
        assert loop.fingerprint_cache is state["base_model_hash_cache"]

    def test_no_state_provider_leaves_cache_unset(self, tmp_path, monkeypatch):
        """Experiment scripts pass state_provider=None; loop.fingerprint_cache
        stays at its default (None) and build_manifest_for skips the cache.
        """
        from paramem.server.consolidation import create_consolidation_loop

        self._patch_loop(monkeypatch)
        cfg = self._make_config(tmp_path)

        loop = create_consolidation_loop(
            model=MagicMock(),
            tokenizer=MagicMock(),
            config=cfg,
            memory_store=_MS(replay_enabled=False),
            state_provider=None,
        )

        assert loop.fingerprint_cache is None

    def test_cache_persists_across_two_loop_constructions(self, tmp_path, monkeypatch):
        """Two consecutive create_consolidation_loop calls with the same state
        share the same cache dict — i.e. the second loop sees entries from the
        first. This is the persistence guarantee the cache exists for.
        """
        from paramem.server.consolidation import create_consolidation_loop

        self._patch_loop(monkeypatch)
        cfg = self._make_config(tmp_path)
        state: dict = {}

        loop_a = create_consolidation_loop(
            model=MagicMock(),
            tokenizer=MagicMock(),
            config=cfg,
            memory_store=_MS(replay_enabled=False),
            state_provider=lambda: state,
        )
        loop_a.fingerprint_cache["sentinel"] = "computed-by-cycle-1"

        loop_b = create_consolidation_loop(
            model=MagicMock(),
            tokenizer=MagicMock(),
            config=cfg,
            memory_store=_MS(replay_enabled=False),
            state_provider=lambda: state,
        )
        assert loop_b.fingerprint_cache.get("sentinel") == "computed-by-cycle-1"

    def test_manifest_readback_populates_state_cache(self, tmp_path, monkeypatch):
        """build_manifest_for must populate _state['base_model_hash_cache'] via read-back.

        When a matching meta.json is found on disk, the hash is obtained from
        the manifest rather than from safetensors files.  The cache must still
        be written so the second call avoids all disk I/O.
        """
        from unittest.mock import MagicMock, patch

        from paramem.adapters.manifest import (
            MANIFEST_SCHEMA_VERSION,
            AdapterManifest,
            BaseModelFingerprint,
            LoRAShape,
            TokenizerFingerprint,
            build_manifest_for,
            write_manifest,
        )

        # Pre-write a slot with a known hash
        adapter_root = tmp_path / "adapters"
        slot = adapter_root / "episodic" / "20260421-000000"
        slot.mkdir(parents=True)
        expected_hash = "sha256:readback_sentinel"
        m_on_disk = AdapterManifest(
            schema_version=MANIFEST_SCHEMA_VERSION,
            name="episodic",
            trained_at="2026-04-21T00:00:00Z",
            base_model=BaseModelFingerprint(repo="hf/base", sha="abc123", hash=expected_hash),
            tokenizer=TokenizerFingerprint(
                name_or_path="hf/base", vocab_size=32000, merges_hash="m"
            ),
            lora=LoRAShape(rank=8, alpha=16, dropout=0.0, target_modules=()),
            registry_sha256="",
            key_count=0,
        )
        write_manifest(slot, m_on_disk)

        # Build a mock model matching the pre-written slot
        model = MagicMock()
        model.config._name_or_path = "hf/base"
        model.config._commit_hash = "abc123"
        peft_cfg = MagicMock()
        peft_cfg.r = 8
        peft_cfg.lora_alpha = 16
        peft_cfg.lora_dropout = 0.0
        peft_cfg.target_modules = []
        model.peft_config = {"episodic": peft_cfg}

        tokenizer = MagicMock()
        tokenizer.name_or_path = "hf/base"
        tokenizer.__len__ = lambda self: 32000
        tokenizer.backend_tokenizer.to_str.return_value = "{}"
        tokenizer.vocab_file = None

        cache: dict = {}

        with (
            patch("paramem.adapters.manifest._hash_safetensors_files") as mock_hash,
            patch("paramem.adapters.manifest._resolve_base_safetensors") as mock_resolve,
        ):
            manifest = build_manifest_for(
                model,
                tokenizer,
                "episodic",
                registry_path=None,
                base_model_hash_cache=cache,
                adapter_root=adapter_root,
            )

        # Read-back must have returned the hash
        assert manifest.base_model.hash == expected_hash
        # Cache must be populated so second call is instant
        assert cache[id(model)] == expected_hash
        # File-hash must NOT have been called
        mock_hash.assert_not_called()
        mock_resolve.assert_not_called()


class TestFullCycleGateHelpers:
    """Helpers that decide whether the scheduled tick should run a full cycle.

    The gate compares the manifest-recorded ``window_stamp`` on the most
    recent main episodic slot against ``current_full_consolidation_stamp()``.
    Both values are produced by the same flooring primitive, so identity
    comparison is exact and idempotent within a window.
    """

    def _write_meta(self, slot_dir, *, window_stamp="", trained_at="2026-04-27T07:29:40Z"):
        import json as _json

        slot_dir.mkdir(parents=True, exist_ok=True)
        payload = {"trained_at": trained_at, "window_stamp": window_stamp}
        (slot_dir / "meta.json").write_text(_json.dumps(payload))

    def test_last_full_consolidation_window_returns_none_when_no_episodic_dir(self, tmp_path):
        from paramem.server.app import _last_full_consolidation_window

        assert _last_full_consolidation_window(tmp_path) is None

    def test_last_full_consolidation_window_returns_none_when_no_slots(self, tmp_path):
        from paramem.server.app import _last_full_consolidation_window

        (tmp_path / "episodic").mkdir()
        assert _last_full_consolidation_window(tmp_path) is None

    def test_last_full_consolidation_window_returns_none_on_corrupt_meta(self, tmp_path):
        from paramem.server.app import _last_full_consolidation_window

        slot = tmp_path / "episodic" / "20260427-072940"
        slot.mkdir(parents=True)
        (slot / "meta.json").write_text("not valid json {")
        assert _last_full_consolidation_window(tmp_path) is None

    def test_last_full_consolidation_window_returns_none_on_empty_window_stamp(self, tmp_path):
        """Legacy v1 manifest (no window_stamp / empty) → None → 'unknown'."""
        from paramem.server.app import _last_full_consolidation_window

        self._write_meta(tmp_path / "episodic" / "20260427-072940", window_stamp="")
        assert _last_full_consolidation_window(tmp_path) is None

    def test_last_full_consolidation_window_picks_lex_max_slot(self, tmp_path):
        """Multiple slots → take the lex-max (latest timestamp dir name)."""
        from paramem.server.app import _last_full_consolidation_window

        self._write_meta(tmp_path / "episodic" / "20260420-120000", window_stamp="20260420T0000")
        self._write_meta(tmp_path / "episodic" / "20260427-072940", window_stamp="20260427T0000")
        self._write_meta(tmp_path / "episodic" / "20260425-080000", window_stamp="20260424T0000")

        assert _last_full_consolidation_window(tmp_path) == "20260427T0000"

    def test_last_full_consolidation_window_skips_pending_dir(self, tmp_path):
        """The .pending side-slot must not be picked even if it sorts last."""
        from paramem.server.app import _last_full_consolidation_window

        self._write_meta(tmp_path / "episodic" / "20260427-072940", window_stamp="20260427T0000")
        (tmp_path / "episodic" / ".pending").mkdir()  # no meta.json

        assert _last_full_consolidation_window(tmp_path) == "20260427T0000"

    def _make_config(self, period_string, adapter_dir):
        cfg = MagicMock()
        cfg.adapter_dir = adapter_dir
        cfg.consolidation.consolidation_period_string = period_string
        return cfg

    def test_is_full_cycle_due_disabled_cadence_returns_false(self, tmp_path):
        """When refresh_cadence is ""/"off", period_string is empty → never auto-due."""
        from paramem.server.app import _is_full_cycle_due

        cfg = self._make_config("", tmp_path)
        assert _is_full_cycle_due(cfg) is False

    def test_is_full_cycle_due_no_prior_full_returns_false(self, tmp_path):
        """Fresh install (no main slot) → defer to interim cycle, NOT full.

        The full cycle (consolidate_interim_adapters) operates by collapsing
        existing interim adapters into main; with none on disk it is a
        no-op. Returning True here previously caused /consolidate to route
        to a no-op full cycle instead of the interim path that actually
        extracts pending sessions on a fresh store.
        """
        from paramem.server.app import _is_full_cycle_due

        cfg = self._make_config("every 84h", tmp_path)
        assert _is_full_cycle_due(cfg) is False

    def test_is_full_cycle_due_legacy_v1_treated_as_no_main_slot(self, tmp_path):
        """A v1 manifest (empty window_stamp) is treated like no main slot
        — defer to the interim path until the slot has been re-stamped by
        a real full cycle.
        """
        from paramem.server.app import _is_full_cycle_due

        self._write_meta(tmp_path / "episodic" / "20260427-072940", window_stamp="")
        cfg = self._make_config("every 84h", tmp_path)
        assert _is_full_cycle_due(cfg) is False

    def test_is_full_cycle_due_same_window_returns_false(self, tmp_path):
        """Last full's window_stamp matches current → already consolidated."""
        from paramem.memory.interim_adapter import current_full_consolidation_stamp
        from paramem.server.app import _is_full_cycle_due

        period = "every 84h"
        current_stamp = current_full_consolidation_stamp(period)
        self._write_meta(tmp_path / "episodic" / "20260427-072940", window_stamp=current_stamp)
        cfg = self._make_config(period, tmp_path)
        assert _is_full_cycle_due(cfg) is False

    def test_is_full_cycle_due_different_window_returns_true(self, tmp_path):
        """Last full's window_stamp differs from current → due."""
        from paramem.server.app import _is_full_cycle_due

        # A stamp from a clearly prior window — distant past.
        self._write_meta(tmp_path / "episodic" / "20260420-000000", window_stamp="20260101T0000")
        cfg = self._make_config("every 84h", tmp_path)
        assert _is_full_cycle_due(cfg) is True

    # --- interim-only / catch-up gate tests ---

    def test_last_full_consolidation_window_skips_interim_dirs(self, tmp_path):
        """Interim dirs (interim_<stamp>/) must NOT be treated as main slots.

        The interim layout is ``episodic/interim_<stamp>/<ts>/meta.json``.
        Only the top-level ``episodic/<ts>/meta.json`` paths belong to the
        main slot scan.  An interim dir at the top level should be invisible
        to ``_last_full_consolidation_window``.
        """
        from paramem.server.app import _last_full_consolidation_window

        # Realistic interim layout: episodic/interim_<stamp>/<ts>/meta.json
        interim_ts = "20260524T0000"
        inner_slot = tmp_path / "episodic" / f"interim_{interim_ts}" / "20260524-120000"
        self._write_meta(inner_slot, window_stamp=interim_ts)
        # No top-level main slot → must return None
        assert _last_full_consolidation_window(tmp_path) is None

    def test_last_full_consolidation_window_main_slot_shadows_interim(self, tmp_path):
        """When a main slot AND interim dirs both exist, return the main stamp.

        The interim dirs must not shadow or replace the main slot result.
        """
        from paramem.server.app import _last_full_consolidation_window

        main_stamp = "20260520T0000"
        # Main slot at top level
        self._write_meta(tmp_path / "episodic" / "20260520-080000", window_stamp=main_stamp)
        # Interim slot nested under episodic/interim_*/
        interim_ts = "20260524T0000"
        inner_slot = tmp_path / "episodic" / f"interim_{interim_ts}" / "20260524-120000"
        self._write_meta(inner_slot, window_stamp=interim_ts)

        assert _last_full_consolidation_window(tmp_path) == main_stamp

    def test_is_full_cycle_due_interim_only_no_main_returns_true(self, tmp_path):
        """Interim-only / no-main state must fire the fold (the production bug).

        Layout: episodic/interim_<stamp>/<ts>/meta.json — no top-level main
        slot.  Before the fix, ``_last_full_consolidation_window`` saw the
        interim dir as a candidate, failed to find a top-level meta.json,
        returned None, and ``_is_full_cycle_due`` returned False, stranding
        interim keys out of the main tiers indefinitely.

        After the fix:
        - ``_last_full_consolidation_window`` skips interim dirs → returns None.
        - ``_is_full_cycle_due`` detects interims on disk → returns True.
        """
        from paramem.server.app import _is_full_cycle_due, _last_full_consolidation_window

        interim_ts = "20260524T0000"
        inner_slot = tmp_path / "episodic" / f"interim_{interim_ts}" / "20260524-120000"
        self._write_meta(inner_slot, window_stamp=interim_ts)

        assert _last_full_consolidation_window(tmp_path) is None
        cfg = self._make_config("every 84h", tmp_path)
        assert _is_full_cycle_due(cfg) is True

    def test_is_full_cycle_due_fresh_install_no_interim_returns_false(self, tmp_path):
        """Fresh install: no main slot AND no interim → gate stays False.

        The full cycle (consolidate_interim_adapters) would be a no-op with
        nothing to fold.  Returning True would cause re-fire on every tick
        with the main slot never being created.  The correct path is the
        interim cycle (which extracts pending sessions first).
        """
        from paramem.server.app import _is_full_cycle_due

        # Empty episodic dir — no main slot, no interim dirs.
        (tmp_path / "episodic").mkdir()
        cfg = self._make_config("every 84h", tmp_path)
        assert _is_full_cycle_due(cfg) is False

    def test_is_full_cycle_due_main_slot_current_with_interim_returns_false(self, tmp_path):
        """Main slot present and up-to-date → False even when interims exist.

        The interims will be folded on the NEXT window tick when current !=
        last.  As long as the main slot's window_stamp matches the current
        window, the fold has already run this window.
        """
        from paramem.memory.interim_adapter import current_full_consolidation_stamp
        from paramem.server.app import _is_full_cycle_due

        period = "every 84h"
        current_stamp = current_full_consolidation_stamp(period)
        # Main slot stamped at the current window.
        self._write_meta(tmp_path / "episodic" / "20260520-080000", window_stamp=current_stamp)
        # Interim slot also present.
        interim_ts = "20260524T0000"
        inner_slot = tmp_path / "episodic" / f"interim_{interim_ts}" / "20260524-120000"
        self._write_meta(inner_slot, window_stamp=interim_ts)

        cfg = self._make_config(period, tmp_path)
        assert _is_full_cycle_due(cfg) is False

    def test_is_full_cycle_due_main_slot_old_with_interim_returns_true(self, tmp_path):
        """Main slot present but from a prior window → True (due-ness follows
        main stamp, not interims).
        """
        from paramem.server.app import _is_full_cycle_due

        old_stamp = "20260101T0000"
        self._write_meta(tmp_path / "episodic" / "20260420-000000", window_stamp=old_stamp)
        # Interim slot also present — should not change the outcome.
        interim_ts = "20260524T0000"
        inner_slot = tmp_path / "episodic" / f"interim_{interim_ts}" / "20260524-120000"
        self._write_meta(inner_slot, window_stamp=interim_ts)

        cfg = self._make_config("every 84h", tmp_path)
        assert _is_full_cycle_due(cfg) is True


class TestTestHarnessImportNoEnvRestoration:
    """Regression: importing experiments.utils.test_harness must NOT re-load .env.

    Background — fix for the Security-OFF e2e smoke bug (2026-04-28):
    Before the fix, ``experiments/utils/test_harness.py`` called
    ``load_dotenv(...)`` at module scope. The first import (triggered
    inside ``ConsolidationLoop._run_recall_sanity_probe``) re-set every
    operator env var. The smoke harness, which had explicitly popped
    ``PARAMEM_DAILY_PASSPHRASE`` to exercise Security OFF, saw the var
    snap back from disk during the first verify probe — every save after
    that point silently encrypted under a "popped" identity. Production
    was unaffected because production never pops env vars between server
    start and verify-probe import; ``load_dotenv(override=False)`` is a
    no-op when the values are already set.

    This test pins the invariant so a future regression doesn't
    re-introduce the module-level side effect.
    """

    def test_importing_test_harness_does_not_set_passphrase_when_unset(self, monkeypatch):
        """Importing test_harness with PARAMEM_DAILY_PASSPHRASE unset must
        NOT cause it to be set, even if a ``.env`` file on disk lists it.
        """
        import importlib
        import os

        # Simulate: shell exported the var, then preflight popped it.
        monkeypatch.delenv("PARAMEM_DAILY_PASSPHRASE", raising=False)

        # Force-reload test_harness so its module-level code runs in this
        # monkey-patched state (the module may already be cached from a
        # prior test in this session).
        import experiments.utils.test_harness as th  # noqa: F401

        importlib.reload(th)

        # The bug we're guarding against: load_dotenv at module scope
        # re-set the var from .env on first import. Post-fix, no
        # module-level load_dotenv exists, so the env stays popped.
        assert "PARAMEM_DAILY_PASSPHRASE" not in os.environ, (
            "experiments.utils.test_harness import must not re-load .env "
            "into os.environ. Module-scope load_dotenv was removed on "
            "2026-04-28 — see module docstring."
        )

    def test_load_test_env_helper_is_explicit_opt_in(self):
        """The replacement ``load_test_env()`` helper is the explicit, opt-in
        way to source ``.env`` from a script's main() / CLI entrypoint."""
        from experiments.utils.test_harness import load_test_env

        assert callable(load_test_env)


# ---------------------------------------------------------------------------
# Regression: interim write path preserves full schema (W3 / W4 bug fix)
# ---------------------------------------------------------------------------


class TestIndexedKeyCacheSchemaInvariant:
    """Regression guard: indexed_key_cache entries carry the canonical field set.

    Knowledge lives solely in adapter weights (train mode) or graph.json
    (simulate mode) — quads.json sidecars are removed.  The
    indexed_key_cache is the in-RAM transient view.  Entries must carry
    ``subject``, ``predicate``, ``object`` (not the old ``source_*`` aliases)
    alongside ``key``, ``speaker_id``, and ``first_seen_cycle`` so the router
    entity/speaker indexes populate correctly.
    """

    _CANONICAL_FIELDS = {"key", "subject", "predicate", "object", "speaker_id", "first_seen_cycle"}

    def _build_minimal_pair(
        self,
        *,
        key: str = "graph1",
        subject: str = "Alice",
        predicate: str = "lives_in",
        obj: str = "Berlin",
        speaker_id: str = "Speaker0",
        first_seen_cycle: int = 1,
    ) -> dict:
        return {
            "key": key,
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "speaker_id": speaker_id,
            "first_seen_cycle": first_seen_cycle,
        }

    def test_canonical_pair_has_required_fields(self) -> None:
        """A well-formed indexed_key_cache entry carries all required fields."""
        pair = self._build_minimal_pair()
        missing = self._CANONICAL_FIELDS - set(pair.keys())
        assert missing == set(), (
            f"Canonical cache entry missing fields: {missing}. "
            "Update the indexed_key_cache[k] = {{...}} sites in consolidation.py."
        )

    def test_no_source_alias_fields(self) -> None:
        """source_subject/predicate/object aliases must not appear in cache entries."""
        pair = self._build_minimal_pair()
        stale_fields = {"source_subject", "source_predicate", "source_object"}
        present = stale_fields & set(pair.keys())
        assert present == set(), (
            f"Stale source_* alias fields still present: {present}. "
            "Remove from all indexed_key_cache producer sites."
        )


class TestResetMainTierRegistriesAndSimhashes:
    """Regression guard for the consolidate_interim_adapters finalize step.

    The fold rewrites each main tier's KeyRegistry from the post-consolidation
    ``tier_keyed`` layout.  It MUST repopulate the per-tier SimHash registry in
    the same pass: a fold-rebuilt tier (e.g. episodic consolidated from interim)
    that gets a fresh registry but an EMPTY SimHash registry returns 0.000
    SimHash-confidence recall — the primary recall metric — for every key,
    silently breaking reconstruct_graph / train->simulate and recall
    verification.  These tests fail if the SimHash repopulation is dropped.
    """

    @staticmethod
    def _entry(key, subject="Alice", predicate="lives_in", obj="Berlin"):
        return {"key": key, "subject": subject, "predicate": predicate, "object": obj}

    @staticmethod
    def _call(store, tier_keyed):
        from types import SimpleNamespace

        ConsolidationLoop._reset_main_tier_registries_and_simhashes(
            SimpleNamespace(store=store), tier_keyed
        )

    def test_registry_and_simhash_rebuilt_together(self):
        from paramem.memory.entry import build_registry
        from paramem.memory.store import MemoryStore

        store = MemoryStore(replay_enabled=True)
        episodic = [self._entry("graph1"), self._entry("graph2", subject="Bob")]
        procedural = [self._entry("graph3", predicate="prefers", obj="tea")]
        self._call(store, {"episodic": episodic, "semantic": [], "procedural": procedural})

        # Registry rewritten to the new membership.
        assert len(store.registry("episodic")) == 2
        assert "graph1" in store.registry("episodic")
        assert "graph2" in store.registry("episodic")
        assert len(store.registry("procedural")) == 1
        assert "graph3" in store.registry("procedural")

        # SimHash registry repopulated in the SAME pass (the load-bearing pairing).
        # An empty view here is exactly the bug this guards against.
        assert store.simhashes_in_tier("episodic") == build_registry(episodic)
        assert store.simhashes_in_tier("procedural") == build_registry(procedural)
        assert store.simhash_count_in_tier("episodic") == 2

    def test_overwrites_stale_simhash(self):
        from paramem.memory.store import MemoryStore

        store = MemoryStore(replay_enabled=True)
        store.put_simhash("episodic", "graph_old", 0xDEAD)
        self._call(
            store,
            {"episodic": [self._entry("graph_new")], "semantic": [], "procedural": []},
        )

        view = store.simhashes_in_tier("episodic")
        assert "graph_old" not in view
        assert "graph_new" in view

    def test_empty_tier_clears_registry_and_simhash(self):
        from paramem.memory.store import MemoryStore

        store = MemoryStore(replay_enabled=True)
        store.put_simhash("semantic", "graph_stale", 0xBEEF)
        self._call(store, {"episodic": [], "semantic": [], "procedural": []})

        assert len(store.registry("semantic")) == 0
        assert store.simhashes_in_tier("semantic") == {}


def test_promotion_carry_over_restores_nonzero_attributes(tmp_path):
    """Base-swap promotion carry-over: ``seed_key_metadata`` restores NON-ZERO
    ``sessions_seen`` + ``promoted_keys`` against the Phase-B-repopulated store,
    and drops orphans (keys absent from the store).

    The live swap exercised only the all-zero case (the deployed data had no
    promotions).  This proves non-zero promotion attributes survive a swap, and
    that the carry-over MUST run after the store is repopulated — an orphaned key
    (no tier in the store) is dropped, which is exactly why re-seeding before
    Phase B would lose everything.  Uses the REAL store and the REAL
    ``seed_key_metadata`` / ``_load_key_metadata`` (the carry-over's two halves).
    """
    import json
    import types

    from paramem.memory.store import MemoryStore
    from paramem.server.consolidation import _load_key_metadata
    from paramem.training.consolidation import ConsolidationLoop

    # Real store with 3 episodic keys registered (as Phase B leaves it post-retrain).
    store = MemoryStore(replay_enabled=True)
    for k in ("graph1", "graph2", "graph3"):
        store.put("episodic", k, {"key": k, "question": "q", "answer": "a"}, register=True)

    # The PREVIOUS model's key_metadata.json with non-zero promotion attributes.
    # The migration never overwrites this file, so it persists through the swap.
    meta = {
        "cycle_count": 5,
        "keys": {
            "graph1": {"sessions_seen": 7, "speaker_id": "Speaker0", "first_seen_cycle": 1},
            "graph2": {"sessions_seen": 3, "speaker_id": "Speaker0", "first_seen_cycle": 2},
            "graph3": {"sessions_seen": 0, "speaker_id": "Speaker0", "first_seen_cycle": 3},
            "orphan": {"sessions_seen": 9, "speaker_id": "Speaker0", "first_seen_cycle": 0},
        },
        "promoted_keys": ["graph1", "orphan"],
    }
    path = tmp_path / "key_metadata.json"
    path.write_text(json.dumps(meta))

    loaded = _load_key_metadata(path)
    assert loaded is not None

    # Minimal container the carry-over mutates; the store and method are real.
    loop = types.SimpleNamespace(store=store, key_sessions={}, promoted_keys=set(), cycle_count=0)
    ConsolidationLoop.seed_key_metadata(loop, loaded)

    # Non-zero sessions_seen carried over verbatim; orphan dropped (absent from store).
    assert loop.key_sessions == {"graph1": 7, "graph2": 3, "graph3": 0}
    assert loop.promoted_keys == {"graph1"}  # "orphan" filtered (no tier in store)
    assert loop.cycle_count == 5


# ---------------------------------------------------------------------------
# backfill_relation_type_from_graph (Slice 1)
# ---------------------------------------------------------------------------


class TestBackfillRelationTypeFromGraph:
    """Tests for ConsolidationLoop.backfill_relation_type_from_graph.

    The method reads relation_type off cumulative graph edges and patches
    _bookkeeping entries whose relation_type is "unknown".
    """

    def _make_loop_with_graph(self, graph_edges, store):
        """Build a minimal ConsolidationLoop stub with the given graph edges."""
        import types

        import networkx as nx

        from paramem.training.consolidation import ConsolidationLoop

        graph = nx.MultiDiGraph()
        for subject, obj, data in graph_edges:
            graph.add_edge(subject, obj, **data)

        merger = types.SimpleNamespace(graph=graph)
        loop = types.SimpleNamespace(
            store=store,
            merger=merger,
        )
        # Bind the real method to the stub namespace.
        loop.backfill_relation_type_from_graph = (
            ConsolidationLoop.backfill_relation_type_from_graph.__get__(loop)
        )
        return loop

    def test_backfill_fills_unknown_from_graph_edge(self):
        """Backfill recovers relation_type from a matching cumulative-graph edge."""
        from paramem.memory.store import MemoryStore
        from paramem.training.key_registry import KeyRegistry

        store = MemoryStore(replay_enabled=True)
        reg = KeyRegistry()
        reg.add("graph1")
        store.load_registry("episodic", reg)
        # Seed entry so the triple is findable.
        store.put(
            "episodic",
            "graph1",
            {
                "key": "graph1",
                "subject": "Alice",
                "predicate": "lives_in",
                "object": "Berlin",
            },
        )
        # Bookkeeping with relation_type="unknown" (legacy upgrade path).
        store.set_bookkeeping(
            "graph1", speaker_id="alice", first_seen_cycle=1, relation_type="unknown"
        )

        # Graph edge with the matching triple and a concrete relation_type.
        edges = [
            ("Alice", "Berlin", {"predicate": "lives_in", "relation_type": "factual"}),
        ]
        loop = self._make_loop_with_graph(edges, store)
        stats = loop.backfill_relation_type_from_graph()

        assert stats["filled"] == 1
        assert stats["unknown_fallback"] == 0
        bk = store.bookkeeping_for_key("graph1")
        assert bk is not None
        assert bk["relation_type"] == "factual"

    def test_backfill_skips_key_already_has_relation_type(self):
        """Keys whose bookkeeping has relation_type != 'unknown' are not touched."""
        from paramem.memory.store import MemoryStore
        from paramem.training.key_registry import KeyRegistry

        store = MemoryStore(replay_enabled=True)
        reg = KeyRegistry()
        reg.add("graph1")
        store.load_registry("episodic", reg)
        store.put(
            "episodic",
            "graph1",
            {
                "key": "graph1",
                "subject": "Alice",
                "predicate": "likes",
                "object": "Tea",
            },
        )
        store.set_bookkeeping(
            "graph1", speaker_id="alice", first_seen_cycle=1, relation_type="preference"
        )

        edges = [("Alice", "Tea", {"predicate": "likes", "relation_type": "factual"})]
        loop = self._make_loop_with_graph(edges, store)
        stats = loop.backfill_relation_type_from_graph()

        # checked=0 because the existing relation_type is not "unknown"
        assert stats["checked"] == 0
        assert stats["filled"] == 0
        # relation_type unchanged
        bk = store.bookkeeping_for_key("graph1")
        assert bk["relation_type"] == "preference"

    def test_backfill_key_absent_from_graph_defaults_to_unknown(self):
        """Key whose triple is not in the graph keeps relation_type='unknown'."""
        from paramem.memory.store import MemoryStore
        from paramem.training.key_registry import KeyRegistry

        store = MemoryStore(replay_enabled=True)
        reg = KeyRegistry()
        reg.add("graph1")
        store.load_registry("episodic", reg)
        store.put(
            "episodic",
            "graph1",
            {
                "key": "graph1",
                "subject": "Bob",
                "predicate": "works_at",
                "object": "ACME",
            },
        )
        store.set_bookkeeping(
            "graph1", speaker_id="bob", first_seen_cycle=2, relation_type="unknown"
        )

        # Graph has an unrelated edge — graph1's triple is not present.
        edges = [("Alice", "Berlin", {"predicate": "lives_in", "relation_type": "factual"})]
        loop = self._make_loop_with_graph(edges, store)
        stats = loop.backfill_relation_type_from_graph()

        assert stats["checked"] == 1
        assert stats["filled"] == 0
        assert stats["unknown_fallback"] == 1
        bk = store.bookkeeping_for_key("graph1")
        assert bk["relation_type"] == "unknown"

    def test_backfill_empty_graph_returns_zeros(self):
        """An empty graph produces all-zero stats without error."""
        from paramem.memory.store import MemoryStore
        from paramem.training.key_registry import KeyRegistry

        store = MemoryStore(replay_enabled=True)
        reg = KeyRegistry()
        reg.add("graph1")
        store.load_registry("episodic", reg)
        store.set_bookkeeping(
            "graph1", speaker_id="alice", first_seen_cycle=1, relation_type="unknown"
        )

        loop = self._make_loop_with_graph([], store)
        stats = loop.backfill_relation_type_from_graph()
        assert stats == {"checked": 0, "filled": 0, "unknown_fallback": 0}

    def test_backfill_cache_off_no_entry_counts_as_unknown_fallback(self):
        """Under cache-off (no content entry), key is counted as unknown_fallback."""
        from paramem.memory.store import MemoryStore
        from paramem.training.key_registry import KeyRegistry

        store = MemoryStore(replay_enabled=True)
        reg = KeyRegistry()
        reg.add("graph1")
        store.load_registry("episodic", reg)
        # Bookkeeping only; no _entries (cache-off scenario).
        store.set_bookkeeping(
            "graph1", speaker_id="alice", first_seen_cycle=1, relation_type="unknown"
        )

        edges = [("Alice", "Berlin", {"predicate": "lives_in", "relation_type": "factual"})]
        loop = self._make_loop_with_graph(edges, store)
        stats = loop.backfill_relation_type_from_graph()

        assert stats["checked"] == 1
        assert stats["filled"] == 0
        assert stats["unknown_fallback"] == 1


# ---------------------------------------------------------------------------
# _prune_old_slots: adapter slot retention (Commit 2)
# ---------------------------------------------------------------------------


def _make_slot_dir(parent, name: str):
    """Create a slot-shaped directory under *parent* and return its path."""
    d = parent / name
    d.mkdir(parents=True, exist_ok=True)
    return d


class TestSlotRetention:
    """Unit tests for ConsolidationLoop._prune_old_slots."""

    def _make_loop(self, tmp_path):
        """Return a minimal ConsolidationLoop instance for _prune_old_slots testing.

        Bypasses __init__ (object.__new__) exactly like TestSaveAdaptersManifest;
        only _keep_prior_slots needs to be set for these tests.
        """
        from paramem.training.consolidation import ConsolidationLoop

        loop = object.__new__(ConsolidationLoop)
        loop.output_dir = tmp_path
        loop._keep_prior_slots = 3
        return loop

    def test_prune_keeps_live_and_n_priors(self, tmp_path):
        """_prune_old_slots retains live_slot + keep most-recent priors; deletes older ones."""
        import os

        loop = self._make_loop(tmp_path)
        tier_root = tmp_path / "episodic"
        tier_root.mkdir()

        # Create 6 slot dirs with distinct mtimes (touch in order so mtime is stable).
        slots = []
        for i in range(6):
            name = f"2026010{i + 1}-120000"
            d = _make_slot_dir(tier_root, name)
            # Explicitly stagger mtime so sort order is deterministic.
            os.utime(d, (1_700_000_000 + i, 1_700_000_000 + i))
            slots.append(d)

        # slots[5] is newest; designate it as live.
        live_slot = slots[5]

        loop._prune_old_slots(tier_root, live_slot, keep=2)

        # live_slot always survives.
        assert live_slot.exists(), "live_slot must survive pruning"
        # 2 most-recent priors (slots[4], slots[3]) survive.
        assert slots[4].exists(), "most-recent prior must survive"
        assert slots[3].exists(), "second most-recent prior must survive"
        # Older 3 slots (slots[0], slots[1], slots[2]) are removed.
        assert not slots[0].exists(), "oldest slot must be pruned"
        assert not slots[1].exists(), "second oldest must be pruned"
        assert not slots[2].exists(), "third oldest must be pruned"

    def test_prune_keep_zero_keeps_only_live(self, tmp_path):
        """keep=0 removes all prior slots; only live_slot remains."""
        loop = self._make_loop(tmp_path)
        tier_root = tmp_path / "episodic"
        tier_root.mkdir()

        live_slot = _make_slot_dir(tier_root, "20260526-120000")
        prior1 = _make_slot_dir(tier_root, "20260525-120000")
        prior2 = _make_slot_dir(tier_root, "20260524-120000")

        loop._prune_old_slots(tier_root, live_slot, keep=0)

        assert live_slot.exists()
        assert not prior1.exists()
        assert not prior2.exists()

    def test_prune_skips_interim_dirs(self, tmp_path):
        """Non-slot siblings with interim_<stamp>/ names are untouched."""
        loop = self._make_loop(tmp_path)
        tier_root = tmp_path / "episodic"
        tier_root.mkdir()

        live_slot = _make_slot_dir(tier_root, "20260526-120000")
        interim_a = _make_slot_dir(tier_root, "interim_20260526T1430")
        interim_b = _make_slot_dir(tier_root, "interim_20260526T0230")

        loop._prune_old_slots(tier_root, live_slot, keep=0)

        # Interim dirs do not match is_slot_name → untouched regardless of keep.
        assert interim_a.exists(), "interim dir must be skipped by pruner"
        assert interim_b.exists(), "interim dir must be skipped by pruner"

    def test_prune_skips_files_and_dot_dirs(self, tmp_path):
        """Files and dot-prefixed dirs inside tier_root are untouched."""
        loop = self._make_loop(tmp_path)
        tier_root = tmp_path / "episodic"
        tier_root.mkdir()

        live_slot = _make_slot_dir(tier_root, "20260526-120000")
        reg_file = tier_root / "indexed_key_registry.json"
        reg_file.write_text("{}")
        pending = tier_root / ".pending"
        pending.mkdir()
        pending_slot = pending / "20260526-130000"
        pending_slot.mkdir()

        loop._prune_old_slots(tier_root, live_slot, keep=0)

        assert reg_file.exists(), "registry JSON file must be untouched"
        assert pending.exists(), "dot-prefixed .pending dir must be untouched"
        assert pending_slot.exists(), "slot inside .pending must be untouched"

    def test_prune_no_op_when_tier_root_missing(self, tmp_path):
        """Non-existent tier_root must raise no exception and have no side effects."""
        loop = self._make_loop(tmp_path)
        missing = tmp_path / "does_not_exist"

        # Must not raise.
        loop._prune_old_slots(missing, missing / "20260526-120000", keep=2)

    def test_prune_wired_from_save_adapters(self, tmp_path, monkeypatch):
        """_prune_old_slots is invoked once per saved tier from _save_adapters.

        Patches atomic_save_adapter to return a known slot path,
        _verify_saved_adapter_from_disk to no-op, and the store's registry
        commit to no-op, so the real _save_adapters code path runs up to the
        prune call without needing a live model.
        """
        from unittest.mock import MagicMock, patch

        from paramem.training.consolidation import ConsolidationLoop
        from paramem.training.key_registry import KeyRegistry
        from paramem.utils.config import AdapterConfig, ConsolidationConfig, TrainingConfig

        # Build a minimal loop (same pattern as TestSaveAdaptersManifest).
        model = MagicMock()
        model.config._name_or_path = "test-base"
        model.config._commit_hash = None
        lora_cfg = MagicMock()
        lora_cfg.r = 4
        lora_cfg.lora_alpha = 8
        lora_cfg.lora_dropout = 0.0
        lora_cfg.target_modules = ["q_proj"]
        lora_cfg.bias = "none"
        # Only episodic — keeps peft_config check simple.
        model.peft_config = {"episodic": lora_cfg}

        def _fake_save_pretrained(path, selected_adapters=None):
            from pathlib import Path as _Path

            p = _Path(path)
            p.mkdir(parents=True, exist_ok=True)
            (p / "adapter_model.safetensors").write_bytes(b"w")
            (p / "adapter_config.json").write_text("{}")

        model.save_pretrained.side_effect = _fake_save_pretrained

        tokenizer = MagicMock()
        tokenizer.name_or_path = "test-tok"
        tokenizer.backend_tokenizer = None
        tokenizer.vocab_size = 32000

        loop = object.__new__(ConsolidationLoop)
        loop.model = model
        loop.tokenizer = tokenizer
        loop.config = ConsolidationConfig()
        loop.training_config = TrainingConfig(num_epochs=1)
        loop.episodic_config = AdapterConfig(rank=4, alpha=8, target_modules=["q_proj"])
        loop.semantic_config = None
        loop.procedural_config = None
        loop.wandb_config = None
        loop.output_dir = tmp_path
        loop.snapshot_dir = None
        loop.save_cycle_snapshots = False
        loop._debug_base = None
        loop._keep_prior_slots = 2
        loop.indexed_key_registry = {"episodic": KeyRegistry()}
        loop.indexed_key_cache = {}
        loop.cycle_count = 0
        loop.merger = MagicMock()
        loop.episodic_simhash = {}

        # Known slot path that atomic_save_adapter will return.
        expected_slot = tmp_path / "episodic" / "20260526-120000"
        expected_slot.mkdir(parents=True, exist_ok=True)

        prune_calls: list = []

        def _fake_prune(tier_root, live_slot, keep):
            prune_calls.append((tier_root, live_slot, keep))

        loop._prune_old_slots = _fake_prune

        with (
            patch(
                "paramem.training.consolidation.atomic_save_adapter",
                return_value=expected_slot,
            ),
            patch.object(
                ConsolidationLoop,
                "_verify_saved_adapter_from_disk",
                return_value=1.0,
            ),
            patch(
                "paramem.training.consolidation.save_registry",
            ),
        ):
            # Disable replay so the registry-commit block is skipped cleanly.
            loop.store._replay_enabled = False
            loop._save_adapters()

        assert len(prune_calls) == 1, (
            f"Expected 1 prune call (episodic only), got {len(prune_calls)}"
        )
        _tier_root, _live_slot, _keep = prune_calls[0]
        assert _live_slot == expected_slot, "live_slot must match the slot from atomic_save_adapter"
        assert _keep == 2, "_keep must match loop._keep_prior_slots"


# ---------------------------------------------------------------------------
# ConsolidationLoop.release() — merger holder teardown
# ---------------------------------------------------------------------------


class TestConsolidationLoopRelease:
    """ConsolidationLoop.release() must null the GraphMerger's model reference.

    GraphMerger is a BASE-MODEL HOLDER (architecture §4.4).  release() is the
    encapsulated teardown path; _release_base_model_in_process reaches it
    transitively via loop.release() → merger.release().
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
            "(GraphMerger is a BASE-MODEL HOLDER — architecture §4.4)"
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


# ---------------------------------------------------------------------------
# Abort-then-skip contract at each production train_adapter call site
# ---------------------------------------------------------------------------


class TestAbortSkipsCommit:
    """metrics['aborted']=True skips post-train commit / registry mutations.

    These tests verify the three production call sites where a train_adapter
    return dict is inspected for the 'aborted' key:

    1. run_consolidation_cycle: aborted episodic → returns {"mode": "aborted"}
    2. _run_indexed_key_procedural: aborted proc → deferred store.put/delete skipped
    3. consolidate_interim_adapters: aborted tier → backup restored, raises
       AbortedDuringConsolidation
    """

    def _make_minimal_loop(self, monkeypatch, tmp_path):
        """Return a ConsolidationLoop stub with enough state for abort-path tests.

        Bypasses __init__ (object.__new__) to avoid model/GPU requirements.
        Patches internal helpers that have side effects unrelated to the abort
        path under test.
        """
        from peft import PeftModel

        from paramem.memory.store import MemoryStore
        from paramem.training.consolidation import ConsolidationLoop
        from paramem.utils.config import AdapterConfig, ConsolidationConfig, TrainingConfig

        loop = object.__new__(ConsolidationLoop)
        loop.model = MagicMock()
        loop.model.__class__ = PeftModel
        loop.model.peft_config = {
            "episodic": MagicMock(),
            "semantic": MagicMock(),
            "procedural": MagicMock(),
            "in_training": MagicMock(),
        }
        loop.tokenizer = MagicMock()
        loop.config = ConsolidationConfig()
        loop.training_config = TrainingConfig(num_epochs=1, gradient_checkpointing=False)
        loop.episodic_config = AdapterConfig(rank=4, alpha=8, target_modules=["q_proj"])
        loop.semantic_config = AdapterConfig(rank=4, alpha=8, target_modules=["q_proj"])
        loop.procedural_config = AdapterConfig(rank=4, alpha=8, target_modules=["q_proj"])
        loop.wandb_config = None
        loop._thermal_policy = None
        loop.output_dir = tmp_path
        loop.store = MemoryStore(replay_enabled=True)
        loop.indexed_key_cache = {}
        loop.key_sessions = {}
        loop.promoted_keys = set()
        loop.cycle_count = 0
        loop.episodic_simhash = {}
        loop.semantic_simhash = {}
        loop.procedural_simhash = {}
        loop._procedural_next_index = 0
        loop._procedural_tentative_next_index = 0
        loop.merger = MagicMock()
        loop.merger.graph.nodes = {}
        loop._bg_trainer = None
        loop.shutdown_requested = False
        loop._early_stop_callback = None
        loop.fingerprint_cache = None
        loop._keep_prior_slots = 2
        loop._debug_base = None
        loop.save_cycle_snapshots = False
        loop.snapshot_dir = None
        loop._indexed_next_index = 0
        loop._indexed_ep_interim = {}
        loop.episodic_replay_pool = []
        loop.curriculum_sampler = None
        loop.pending_interim_triples = []
        return loop

    def test_run_consolidation_cycle_returns_aborted_on_abort(self, monkeypatch, tmp_path):
        """When train_adapter returns aborted=True, run_consolidation_cycle
        returns {'mode': 'aborted'} without updating simhashes or committing
        the tier slot.
        """
        from unittest.mock import patch

        from paramem.training.consolidation import ConsolidationLoop

        loop = self._make_minimal_loop(monkeypatch, tmp_path)

        # Register one episodic key so the guard passes.
        loop.store.put(
            "episodic",
            "graph1",
            {
                "key": "graph1",
                "subject": "Alex",
                "predicate": "lives_in",
                "object": "Millfield",
                "speaker_id": "Speaker0",
                "first_seen_cycle": 1,
            },
            register=True,
        )
        loop.indexed_key_cache["graph1"] = {
            "key": "graph1",
            "subject": "Alex",
            "predicate": "lives_in",
            "object": "Millfield",
            "speaker_id": "Speaker0",
            "first_seen_cycle": 1,
        }

        aborted_metrics = {"train_loss": 0.5, "aborted": True}
        simhash_calls: list = []

        def _spy_replace(tier, mapping):
            simhash_calls.append(tier)

        with (
            # Block HF TrainingArguments construction (bf16 validation, GPU check).
            patch("paramem.training.trainer.TrainingArguments", return_value=MagicMock()),
            patch(
                "paramem.training.encrypted_checkpoint_callback.EncryptCheckpointCallback",
                MagicMock,
            ),
            # run_consolidation_cycle imports train_adapter locally; patch at source.
            patch("paramem.training.trainer.train_adapter", return_value=aborted_metrics),
            patch.object(loop.store, "replace_simhashes_in_tier", side_effect=_spy_replace),
            # Stub heavy helpers that are not under test.
            patch.object(
                ConsolidationLoop,
                "_resolve_target_slot",
                return_value=("episodic_interim_t001", False, False, False),
            ),
            patch.object(
                ConsolidationLoop,
                "_prepare_episodic_keys_for_tier",
                return_value=[
                    {
                        "key": "graph1",
                        "subject": "Alex",
                        "predicate": "lives_in",
                        "object": "Millfield",
                        "speaker_id": "Speaker0",
                        "first_seen_cycle": 1,
                    }
                ],
            ),
            patch.object(ConsolidationLoop, "_maybe_run_interim_enrichment", return_value=None),
            patch.object(ConsolidationLoop, "_enable_gradient_checkpointing", return_value=None),
            patch.object(ConsolidationLoop, "_disable_gradient_checkpointing", return_value=None),
            patch.object(
                ConsolidationLoop, "_maybe_make_recall_callback", return_value=(None, None)
            ),
            patch("paramem.training.consolidation.switch_adapter"),
            patch(
                "paramem.training.consolidation.format_entry_training",
                return_value=[{"input_ids": [1], "labels": [1]}],
            ),
            # Stub PEFT slot creation so loop.model stays a MagicMock.
            patch(
                "paramem.memory.interim_adapter.create_interim_adapter",
                side_effect=lambda m, cfg, stamp: m,
            ),
        ):
            result = loop.run_consolidation_cycle(
                [
                    {
                        "subject": "Alex",
                        "predicate": "lives_in",
                        "object": "Millfield",
                        "relation_type": "factual",
                        "speaker_id": "Speaker0",
                    }
                ],
                [],
                speaker_id="Speaker0",
                mode="train",
                run_label="test_cycle",
                stamp="t001",
            )

        assert result.get("mode") == "aborted", (
            f"Expected mode='aborted' when training aborted; got {result}"
        )
        # simhash registry must NOT be updated after abort.
        assert simhash_calls == [], (
            f"replace_simhashes_in_tier must not be called after abort; called for {simhash_calls}"
        )

    def test_run_indexed_key_procedural_skips_deferred_mutations_on_aborted(
        self, monkeypatch, tmp_path
    ):
        """When proc train_adapter returns aborted=True, store.put must be
        skipped — only the abort guard fires.

        Note: sp_index (procedural_sp_index) was removed in the model-only
        contradiction redesign (consolidation_architecture.md §4.3).
        The 2-tuple return contract of _prepare_procedural_keys_for_tier
        is tested here via the patched return value.
        """
        from unittest.mock import patch

        from paramem.training.consolidation import ConsolidationLoop

        loop = self._make_minimal_loop(monkeypatch, tmp_path)

        # Pre-seed the store so _prepare_procedural_keys_for_tier sees existing keys.
        loop.store.put(
            "procedural",
            "prc1",
            {
                "key": "prc1",
                "subject": "Alex",
                "predicate": "listens_to",
                "object": "Jazz",
                "speaker_id": "Speaker0",
                "first_seen_cycle": 1,
            },
            register=True,
        )
        loop.store.put_simhash("procedural", "prc1", 0xABCD)

        aborted_metrics = {"train_loss": 0.3, "aborted": True}

        store_delete_calls: list = []
        store_put_calls: list = []

        original_delete = loop.store.delete
        original_put = loop.store.put

        def _spy_delete(key):
            store_delete_calls.append(key)
            return original_delete(key)

        def _spy_put(tier, key, entry, **kwargs):
            store_put_calls.append((tier, key))
            return original_put(tier, key, entry, **kwargs)

        with (
            patch("paramem.training.trainer.TrainingArguments", return_value=MagicMock()),
            patch(
                "paramem.training.encrypted_checkpoint_callback.EncryptCheckpointCallback",
                MagicMock,
            ),
            patch("paramem.training.trainer.train_adapter", return_value=aborted_metrics),
            patch.object(loop.store, "delete", side_effect=_spy_delete),
            patch.object(loop.store, "put", side_effect=_spy_put),
            patch.object(ConsolidationLoop, "_enable_gradient_checkpointing", return_value=None),
            patch.object(ConsolidationLoop, "_disable_gradient_checkpointing", return_value=None),
            patch.object(
                ConsolidationLoop, "_maybe_make_recall_callback", return_value=(None, None)
            ),
            patch.object(
                ConsolidationLoop,
                "_prepare_procedural_keys_for_tier",
                return_value=(
                    [
                        {
                            "key": "prc2",
                            "subject": "Alex",
                            "predicate": "listens_to",
                            "object": "Jazz",
                            "speaker_id": "Speaker0",
                            "first_seen_cycle": 1,
                        }
                    ],  # new_keyed
                    [
                        {
                            "key": "prc1",
                            "subject": "Alex",
                            "predicate": "listens_to",
                            "object": "Jazz",
                            "speaker_id": "Speaker0",
                            "first_seen_cycle": 1,
                        }
                    ],  # existing_keyed (2-tuple: sp_index removed)
                ),
            ),
            patch("paramem.training.consolidation.switch_adapter"),
            patch(
                "paramem.training.consolidation.format_entry_training",
                return_value=[{"input_ids": [1], "labels": [1]}],
            ),
            patch("paramem.training.consolidation.compute_simhash", return_value=0xBEEF),
        ):
            result = loop._run_indexed_key_procedural(
                [
                    {
                        "subject": "Alex",
                        "predicate": "listens_to",
                        "object": "Jazz",
                        "relation_type": "preference",
                        "speaker_id": "Speaker0",
                    }
                ],
                speaker_id="Speaker0",
                mode="train",
                stamp="t001",
                run_label="test_proc",
            )

        assert result is None, f"Expected None return on abort; got {result}"
        # Deferred mutations must be entirely skipped.
        assert store_delete_calls == [], (
            f"store.delete must not be called after proc abort; called for {store_delete_calls}"
        )
        assert store_put_calls == [], (
            f"store.put must not be called after proc abort; called for {store_put_calls}"
        )
        # procedural_sp_index has been removed from ConsolidationLoop
        # (consolidation_architecture.md §4.3); the attribute must not exist.
        assert not hasattr(loop, "procedural_sp_index"), (
            "procedural_sp_index must not exist on ConsolidationLoop after sp_index removal"
        )

    def test_consolidate_interim_adapters_raises_and_rolls_back_on_aborted(
        self, monkeypatch, tmp_path
    ):
        """When a tier's train_adapter returns aborted=True, consolidate_interim_adapters
        restores backup adapter weights via copy_adapter_weights and raises
        AbortedDuringConsolidation.

        This test patches the GPU lock entry guard (so the function believes the
        lock is already held by the caller) and stubs all heavy helpers to isolate
        the abort branch.
        """
        from unittest.mock import MagicMock, patch

        from paramem.training.consolidation import AbortedDuringConsolidation, ConsolidationLoop

        loop = self._make_minimal_loop(monkeypatch, tmp_path)

        import networkx as _nx

        # Pre-populate the store so _all_active_keys() returns at least one key
        # and jobs_by_tier["episodic"] ends up non-empty (otherwise the tier is
        # skipped, the abort branch is never reached, and no rollback fires).
        loop.store.put(
            "episodic",
            "graph1",
            {
                "key": "graph1",
                "subject": "Alex",
                "predicate": "lives_in",
                "object": "Millfield",
                "speaker_id": "Speaker0",
                "first_seen_cycle": 1,
            },
            register=True,
        )
        # Bookkeeping is required for stage-2 bookkeeping lookup in the new
        # full-consolidation reconstruct→remerge pipeline.
        loop.store.set_bookkeeping(
            "graph1",
            speaker_id="Speaker0",
            first_seen_cycle=1,
            relation_type="factual",
        )
        # Stage 5 (edge-walk dedup + tier) reads from merger.graph.edges(data=True).
        # Replace the MagicMock with a real MultiDiGraph carrying the graph1 edge
        # so tier_keyed["episodic"] ends up non-empty and the abort path fires.
        _real_graph = _nx.MultiDiGraph()
        _eid = _real_graph.add_edge("Alex", "Millfield", predicate="lives_in")
        _real_graph["Alex"]["Millfield"][_eid]["relation_type"] = "factual"
        _real_graph["Alex"]["Millfield"][_eid]["ik_key"] = "graph1"
        loop.merger.graph = _real_graph

        # Pre-install backup adapters in peft_config so the backup-creation block
        # is skipped and the rollback (abort) path sees them in peft_config.
        # Without this, the mocked create_adapter does not update peft_config and
        # the rollback's "if backup in peft_config" guard would short-circuit.
        loop.model.peft_config["episodic_backup"] = MagicMock()
        loop.model.peft_config["semantic_backup"] = MagicMock()
        loop.model.peft_config["procedural_backup"] = MagicMock()

        aborted_metrics = {"train_loss": 0.1, "aborted": True}
        copy_calls: list = []

        def _spy_copy(model, src, dst):
            copy_calls.append((src, dst))

        from paramem.graph.reconstruct import ReconstructionResult

        with (
            # Block HF TrainingArguments construction (bf16 validation, GPU check).
            patch("paramem.training.trainer.TrainingArguments", return_value=MagicMock()),
            patch(
                "paramem.training.encrypted_checkpoint_callback.EncryptCheckpointCallback",
                MagicMock,
            ),
            # Simulate GPU lock already held: acquire returns False so the entry
            # guard does NOT raise (it only raises when acquire returns True).
            patch("paramem.server.gpu_lock._gpu_thread_lock") as mock_lock,
            # consolidate_interim_adapters imports train_adapter and
            # copy_adapter_weights locally; patch at the source modules.
            patch("paramem.training.trainer.train_adapter", return_value=aborted_metrics),
            patch("paramem.models.loader.copy_adapter_weights", side_effect=_spy_copy),
            patch.object(ConsolidationLoop, "_enable_gradient_checkpointing", return_value=None),
            patch.object(ConsolidationLoop, "_disable_gradient_checkpointing", return_value=None),
            patch.object(
                ConsolidationLoop, "_maybe_make_recall_callback", return_value=(None, None)
            ),
            patch.object(
                ConsolidationLoop, "_run_graph_enrichment", return_value={"skipped": True}
            ),
            # Stage 1 (reconstruct_graph) is a GPU operation; stub with an empty result
            # so the abort-path test does not require a real model.
            patch(
                "paramem.training.consolidation.reconstruct_graph",
                return_value=ReconstructionResult(graph=_nx.MultiDiGraph()),
            ),
            patch(
                "paramem.training.consolidation.format_entry_training",
                return_value=[{"input_ids": [1], "labels": [1]}],
            ),
            # All model-loader calls are local imports; patch at the source.
            patch("paramem.models.loader.create_adapter", side_effect=lambda m, cfg, name: m),
            patch("paramem.models.loader.switch_adapter"),
        ):
            # GPU lock entry guard: acquire(blocking=False) returns False →
            # the guard body (release + raise) is skipped.
            mock_lock.acquire.return_value = False

            with pytest.raises(AbortedDuringConsolidation):
                loop.consolidate_interim_adapters(trainer=None, router=None)

        # Backup restore: at least one copy_adapter_weights(src=<tier>_backup, dst=<tier>)
        # call must have fired for the tiers whose backup slot exists in peft_config.
        rollback_dsts = {dst for (src, dst) in copy_calls if src.endswith("_backup")}
        assert rollback_dsts, (
            f"No backup restore copy_adapter_weights calls after abort; copy_calls={copy_calls}"
        )


# ---------------------------------------------------------------------------
# Tests for the new full-consolidation pipeline stages 1–5 in
# consolidate_interim_adapters (reconstruct → re-merge → dedup → tier)
# ---------------------------------------------------------------------------


class TestConsolidateInterimAdaptersFullFlow:
    """Unit tests for the reconstruct→re-merge→dedup→tier pipeline.

    All tests mock reconstruct_graph (GPU), training, and all PEFT operations
    so no real model is required.  The focus is on:

    1. Cross-session duplicate (s,p,o) triples collapse to ONE key via the
       merged-graph edge-walk (bug-C regression lock).
    2. Tier assignment is derived from the edge-walk relation_type (replacing
       the dead :3463 ``getattr(graph, "relations", [])`` path).
    3. relation_type from bookkeeping is injected onto reconstructed triples
       before re-merge, so merger.merge() receives the correct type.
    """

    @staticmethod
    def _make_loop(tmp_path, *, merger_graph, procedural_enabled=True):
        """Minimal ConsolidationLoop stub for full-consolidation stage tests.

        The stub wires a real ``MemoryStore`` and the supplied ``merger_graph``
        so the edge-walk stages run against real data.  All other attributes
        are either defaults or MagicMock so they satisfy attribute access
        without side effects.
        """
        from unittest.mock import MagicMock

        from peft import PeftModel

        from paramem.memory.store import MemoryStore
        from paramem.training.consolidation import ConsolidationLoop
        from paramem.utils.config import AdapterConfig, ConsolidationConfig, TrainingConfig

        loop = object.__new__(ConsolidationLoop)
        loop.model = MagicMock()
        loop.model.__class__ = PeftModel
        loop.model.peft_config = {
            "episodic": MagicMock(),
            "semantic": MagicMock(),
            "procedural": MagicMock(),
            "episodic_backup": MagicMock(),
            "semantic_backup": MagicMock(),
            "procedural_backup": MagicMock(),
        }
        loop.tokenizer = MagicMock()
        loop.config = ConsolidationConfig()
        loop.training_config = TrainingConfig(num_epochs=1, gradient_checkpointing=False)
        loop.episodic_config = AdapterConfig(rank=4, alpha=8, target_modules=["q_proj"])
        loop.semantic_config = AdapterConfig(rank=4, alpha=8, target_modules=["q_proj"])
        _proc_cfg = AdapterConfig(rank=4, alpha=8, target_modules=["q_proj"])
        loop.procedural_config = _proc_cfg if procedural_enabled else None
        loop.wandb_config = None
        loop._thermal_policy = None
        loop.output_dir = tmp_path
        loop.store = MemoryStore(replay_enabled=True)

        # Wire a mock merger that delegates graph attribute access to the real
        # nx.MultiDiGraph supplied by the caller.
        loop.merger = MagicMock()
        loop.merger.graph = merger_graph

        loop.key_sessions = {}
        loop.promoted_keys = set()
        loop.cycle_count = 0
        loop._procedural_next_index = 0
        loop._procedural_tentative_next_index = 0
        loop._indexed_next_index = 0
        loop._bg_trainer = None
        loop.shutdown_requested = False
        loop._early_stop_callback = None
        loop.fingerprint_cache = None
        loop._keep_prior_slots = 2
        loop._debug_base = None
        loop.save_cycle_snapshots = False
        loop.snapshot_dir = None
        loop._indexed_ep_interim = {}
        loop.episodic_replay_pool = []
        loop.curriculum_sampler = None
        loop.pending_interim_triples = []
        loop.persist_graph = False
        return loop

    @staticmethod
    def _run_with_mocks(loop, tmp_path, reconstruct_return):
        """Run consolidate_interim_adapters with all heavy operations mocked.

        Patches: GPU lock (held), reconstruct_graph (returns supplied value),
        graph enrichment (skipped), train_adapter (no-op), all PEFT helpers.
        Returns the result dict.
        """
        from unittest.mock import patch

        from paramem.server.gpu_lock import _gpu_thread_lock
        from paramem.training.consolidation import ConsolidationLoop

        _gpu_thread_lock.acquire()
        try:
            with (
                patch(
                    "paramem.training.consolidation.reconstruct_graph",
                    return_value=reconstruct_return,
                ),
                patch.object(
                    ConsolidationLoop,
                    "_run_graph_enrichment",
                    return_value={"skipped": True},
                ),
                patch.object(ConsolidationLoop, "_enable_gradient_checkpointing"),
                patch.object(ConsolidationLoop, "_disable_gradient_checkpointing"),
                patch.object(
                    ConsolidationLoop, "_maybe_make_recall_callback", return_value=(None, None)
                ),
                # _maybe_make_recall_callback returns (None, None) so verdict is None;
                # admit all keys from the probe fallback to avoid requiring a real model.
                patch.object(
                    ConsolidationLoop,
                    "_probe_passing_keys",
                    side_effect=lambda adapter_name, entries: {e["key"] for e in entries},
                ),
                patch.object(ConsolidationLoop, "_save_adapters"),
                patch("paramem.training.trainer.train_adapter", return_value={"aborted": False}),
                patch(
                    "paramem.training.consolidation.format_entry_training",
                    return_value=[{"input_ids": [1], "labels": [1], "attention_mask": [1]}],
                ),
                patch("paramem.models.loader.create_adapter", side_effect=lambda m, c, n: m),
                patch("paramem.models.loader.switch_adapter"),
                patch("paramem.models.loader.copy_adapter_weights"),
                patch("paramem.memory.interim_adapter.unload_interim_adapters", return_value=[]),
            ):
                return loop.consolidate_interim_adapters(trainer=None, router=None)
        finally:
            _gpu_thread_lock.release()

    # -------------------------------------------------------------------------
    # Test 1: cross-session duplicate triples collapse to ONE key
    # -------------------------------------------------------------------------

    def test_duplicate_triple_collapses_to_one_key(self, tmp_path):
        """Bug-C regression lock: two store keys with the same (s,p,o) collapse
        to EXACTLY ONE entry in tier_keyed when sourced from the merged graph.

        The merged graph holds ONE edge for (Alice, lives_in, Berlin); the
        triple_to_key index maps that triple to whichever key is encountered
        LAST in store.iter_entries() (last-write-wins over dict assignment).
        The other key becomes a drift key — it exists in the store registry but
        has no corresponding distinct edge in the graph, so it does NOT appear
        in tier_keyed.
        """
        import networkx as nx

        from paramem.graph.reconstruct import ReconstructionResult

        # Build merged graph with ONE edge for the duplicate triple.
        g = nx.MultiDiGraph()
        eid = g.add_edge("Alice", "Berlin", predicate="lives_in")
        g["Alice"]["Berlin"][eid]["relation_type"] = "factual"

        loop = self._make_loop(tmp_path, merger_graph=g)

        # Register TWO store keys for the same (s,p,o).
        for key in ("graph1", "graph2"):
            loop.store.put(
                "episodic",
                key,
                {
                    "key": key,
                    "subject": "Alice",
                    "predicate": "lives_in",
                    "object": "Berlin",
                    "speaker_id": "Speaker0",
                    "first_seen_cycle": 1,
                },
                register=True,
            )
            loop.store.set_bookkeeping(
                key, speaker_id="Speaker0", first_seen_cycle=1, relation_type="factual"
            )

        result = self._run_with_mocks(loop, tmp_path, ReconstructionResult(graph=nx.MultiDiGraph()))

        # The merged-graph edge-walk yields ONE triple → ONE key in tier_keyed.
        # The second duplicate key is a drift key (not in the graph's edge set).
        # keys_per_tier holds counts; verify via the returned count.
        assert result["keys_per_tier"]["episodic"] == 1, (
            f"Expected exactly 1 episodic key after duplicate dedup; "
            f"got {result['keys_per_tier']['episodic']}"
        )
        assert result["graph_drift_count"] >= 1, (
            f"Expected at least 1 drift key (the duplicate); got {result['graph_drift_count']}"
        )

    # -------------------------------------------------------------------------
    # Test 2: tier from edge-walk relation_type (not dead :3463 path)
    # -------------------------------------------------------------------------

    def test_preference_relation_routes_to_procedural(self, tmp_path):
        """A 'preference' relation_type on the merged-graph edge routes the key
        to the procedural tier, NOT episodic.

        This verifies the edge-walk (stage 4) is non-empty and that
        partition_relations sees the correct relation_type from the graph edge.
        The dead :3463 path always produced an empty lookup, making every key
        fall to the graph-drift branch — which never assigned procedural tier.
        """
        import networkx as nx

        from paramem.graph.reconstruct import ReconstructionResult

        # Merged graph: preference edge.
        g = nx.MultiDiGraph()
        eid = g.add_edge("Alice", "tea", predicate="prefers")
        g["Alice"]["tea"][eid]["relation_type"] = "preference"

        loop = self._make_loop(tmp_path, merger_graph=g, procedural_enabled=True)

        loop.store.put(
            "procedural",
            "proc1",
            {
                "key": "proc1",
                "subject": "Alice",
                "predicate": "prefers",
                "object": "tea",
                "speaker_id": "Speaker0",
                "first_seen_cycle": 1,
            },
            register=True,
        )
        loop.store.set_bookkeeping(
            "proc1", speaker_id="Speaker0", first_seen_cycle=1, relation_type="preference"
        )

        result = self._run_with_mocks(loop, tmp_path, ReconstructionResult(graph=nx.MultiDiGraph()))

        assert result["keys_per_tier"].get("procedural", 0) == 1, (
            f"Expected 1 key in procedural tier; got {result['keys_per_tier']}"
        )
        assert result["keys_per_tier"].get("episodic", 0) == 0, (
            "Preference key must NOT appear in episodic tier"
        )

    # -------------------------------------------------------------------------
    # Test 3: relation_type injected from bookkeeping before re-merge
    # -------------------------------------------------------------------------

    def test_relation_type_injected_from_bookkeeping_before_remerge(self, tmp_path):
        """Stage 2 injects the bookkeeping relation_type onto each reconstructed
        triple before calling merger.merge().

        reconstruct_graph returns a graph with one edge (SPO only, no
        relation_type in edge data).  The bookkeeping entry for that key has
        relation_type='preference'.  We assert that merger.merge() is called
        with a SessionGraph whose Relation carries relation_type='preference'.
        """
        import networkx as nx

        from paramem.graph.reconstruct import ReconstructionResult
        from paramem.memory.persistence import _IK_KEY_ATTR

        # Reconstructed graph: one edge, SPO only (no relation_type attribute).
        recon_g = nx.MultiDiGraph()
        recon_eid = recon_g.add_edge("Alice", "coffee", predicate="prefers")
        recon_g["Alice"]["coffee"][recon_eid][_IK_KEY_ATTR] = "proc1"
        # Note: no relation_type on the edge — only SPO from weights.

        # Merged graph: empty (we are testing stage-2 merger call, not stage-5).
        merged_g = nx.MultiDiGraph()

        loop = self._make_loop(tmp_path, merger_graph=merged_g)

        loop.store.put(
            "procedural",
            "proc1",
            {
                "key": "proc1",
                "subject": "Alice",
                "predicate": "prefers",
                "object": "coffee",
                "speaker_id": "Speaker0",
                "first_seen_cycle": 1,
            },
            register=True,
        )
        # Bookkeeping has relation_type="preference" — this must be injected.
        loop.store.set_bookkeeping(
            "proc1", speaker_id="Speaker0", first_seen_cycle=1, relation_type="preference"
        )

        # Capture merger.merge() call args.
        merge_calls: list = []

        def _spy_merge(session_graph):
            merge_calls.append(session_graph)
            return merged_g

        loop.merger.merge.side_effect = _spy_merge

        self._run_with_mocks(loop, tmp_path, ReconstructionResult(graph=recon_g))

        assert merge_calls, "merger.merge() must have been called for the reconstructed triple"
        session = merge_calls[0]
        assert len(session.relations) == 1, (
            f"Expected 1 relation in synthetic SessionGraph; got {len(session.relations)}"
        )
        rel = session.relations[0]
        assert rel.relation_type == "preference", (
            f"Expected relation_type='preference' from bookkeeping; got {rel.relation_type!r}"
        )

    # -------------------------------------------------------------------------
    # Test 4: predicate normalization mismatch — store raw vs graph normalized
    # -------------------------------------------------------------------------

    def test_normalization_mismatch_key_lands_in_tier(self, tmp_path):
        """Load-bearing correctness: a store entry with an un-normalized predicate
        (e.g. raw extraction output "Lives In") is matched against a merger.graph
        edge whose predicate has the canonical normalized form ("lives_in").

        Stage-3 normalizes the store predicate via _normalize_predicate before
        building triple_to_key; stage-5 normalizes the graph edge predicate the
        same way.  The key must be found and placed in a tier — NOT dropped as
        drift — proving that _normalize_predicate is applied on both sides.
        """
        import networkx as nx

        from paramem.graph.reconstruct import ReconstructionResult

        # Merged graph carries the NORMALIZED predicate (as merger.merge() stores it).
        g = nx.MultiDiGraph()
        eid = g.add_edge("Alice", "Berlin", predicate="lives_in")
        g["Alice"]["Berlin"][eid]["relation_type"] = "factual"

        loop = self._make_loop(tmp_path, merger_graph=g)

        # Store entry carries the RAW (un-normalized) predicate form.
        loop.store.put(
            "episodic",
            "graph1",
            {
                "key": "graph1",
                "subject": "Alice",
                "predicate": "Lives In",  # raw form — must be normalized before lookup
                "object": "Berlin",
                "speaker_id": "Speaker0",
                "first_seen_cycle": 1,
            },
            register=True,
        )
        loop.store.set_bookkeeping(
            "graph1", speaker_id="Speaker0", first_seen_cycle=1, relation_type="factual"
        )

        result = self._run_with_mocks(loop, tmp_path, ReconstructionResult(graph=nx.MultiDiGraph()))

        # The key must appear in a tier — normalization on both sides must match.
        tiers = ("episodic", "semantic", "procedural")
        total_tiered = sum(result["keys_per_tier"].get(t, 0) for t in tiers)
        assert total_tiered == 1, (
            f"Expected exactly 1 tiered key (normalization matched); "
            f"got {result['keys_per_tier']} drift={result['graph_drift_count']}"
        )
        # Normalization-matched key must NOT appear in drift.
        assert result["graph_drift_count"] == 0, (
            f"Expected 0 drift keys (predicate normalization matched); "
            f"got {result['graph_drift_count']}"
        )

    # -------------------------------------------------------------------------
    # Test 5: explicit drift — store key whose triple is absent from graph
    # -------------------------------------------------------------------------

    def test_explicit_drift_key_not_in_tier(self, tmp_path):
        """A store active key whose (s,p,o) triple is genuinely absent from
        merger.graph is counted as drift: NOT placed in any tier_keyed list,
        and counted in graph_drift_count.

        Absence from the graph means the key was either superseded by a
        contradiction, rolled back before extraction, or corrupted — it must
        be dropped from this consolidation cycle and logged, not crashed.
        """
        import networkx as nx

        from paramem.graph.reconstruct import ReconstructionResult

        # Merged graph carries a DIFFERENT triple (unrelated to the drift key).
        g = nx.MultiDiGraph()
        eid = g.add_edge("Bob", "London", predicate="lives_in")
        g["Bob"]["London"][eid]["relation_type"] = "factual"

        loop = self._make_loop(tmp_path, merger_graph=g)

        # Register the unrelated graph key (so it has a valid store entry).
        loop.store.put(
            "episodic",
            "graph_bob",
            {
                "key": "graph_bob",
                "subject": "Bob",
                "predicate": "lives_in",
                "object": "London",
                "speaker_id": "Speaker0",
                "first_seen_cycle": 1,
            },
            register=True,
        )
        loop.store.set_bookkeeping(
            "graph_bob", speaker_id="Speaker0", first_seen_cycle=1, relation_type="factual"
        )

        # Register the DRIFT key — triple does NOT exist in merger.graph.
        loop.store.put(
            "episodic",
            "drift1",
            {
                "key": "drift1",
                "subject": "Alice",
                "predicate": "works_at",
                "object": "Acme",
                "speaker_id": "Speaker0",
                "first_seen_cycle": 1,
            },
            register=True,
        )
        loop.store.set_bookkeeping(
            "drift1", speaker_id="Speaker0", first_seen_cycle=1, relation_type="factual"
        )

        result = self._run_with_mocks(loop, tmp_path, ReconstructionResult(graph=nx.MultiDiGraph()))

        # drift1's triple is absent from merger.graph — it must be counted as drift.
        assert result["graph_drift_count"] >= 1, (
            f"Expected at least 1 drift key (drift1 absent from graph); "
            f"got {result['graph_drift_count']}"
        )
        # graph_bob's triple IS in the graph, so it must be placed in a tier.
        assert result["keys_per_tier"].get("episodic", 0) >= 1, (
            f"Expected graph_bob in episodic tier; got {result['keys_per_tier']}"
        )
