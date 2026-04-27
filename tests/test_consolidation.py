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
                ),
                Relation(
                    subject="Alex",
                    predicate="prefers",
                    object="Acme Radio",
                    relation_type="preference",
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
                ),
            ],
        )

        def _default_extract(*a, **kw):
            return session_graph

        def _default_extract_procedural(*a, **kw):
            return procedural_graph

        monkeypatch.setattr(
            "paramem.training.consolidation.extract_graph",
            extract_graph_spy if extract_graph_spy is not None else _default_extract,
        )
        monkeypatch.setattr(
            "paramem.training.consolidation.extract_procedural_graph",
            extract_procedural_spy
            if extract_procedural_spy is not None
            else _default_extract_procedural,
        )
        # Use template fallback (ignore passed model/tokenizer) for deterministic output.
        monkeypatch.setattr(
            "paramem.training.consolidation.generate_qa_from_relations",
            lambda relations, model=None, tokenizer=None: _real_qa(relations),
        )

        model = MagicMock()
        model.peft_config = {
            "episodic": MagicMock(),
            "semantic": MagicMock(),
            "in_training": MagicMock(),
        }
        if procedural_enabled:
            model.peft_config["procedural"] = MagicMock()
        procedural_adapter = AdapterConfig() if procedural_enabled else None

        return ConsolidationLoop(
            model=model,
            tokenizer=MagicMock(),
            consolidation_config=ConsolidationConfig(),
            training_config=TrainingConfig(),
            episodic_adapter_config=AdapterConfig(),
            semantic_adapter_config=AdapterConfig(),
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
        captured: dict[str, list[dict]] = {"episodic_qa": [], "procedural_rels": []}

        def _capture_episodic(self_, episodic_qa, new_promotions):
            captured["episodic_qa"] = episodic_qa
            return None

        def _capture_procedural(self_, procedural_relations, speaker_id=""):
            captured["procedural_rels"] = procedural_relations
            return None

        # Force the indexed-key branch so we can intercept. Stub registry + save.
        from unittest.mock import MagicMock as _MM

        loop.indexed_key_registry = _MM()
        loop.indexed_key_registry.list_active = _MM(return_value=[])
        monkeypatch.setattr(
            type(loop), "_run_indexed_key_episodic", _capture_episodic, raising=False
        )
        monkeypatch.setattr(
            type(loop),
            "_run_indexed_key_procedural",
            _capture_procedural,
            raising=False,
        )
        monkeypatch.setattr(type(loop), "_save_adapters", lambda self_: None, raising=False)
        loop.run_cycle(
            session_transcript="Alex lives in Millfield. He prefers Acme Radio.",
            session_id="s001",
            speaker_id="spk",
            source_type=source_type,
        )
        return captured["episodic_qa"], captured["procedural_rels"]

    def test_parity_procedural_enabled(self, monkeypatch, tmp_path):
        loop_a = self._build_loop(monkeypatch, tmp_path / "a", procedural_enabled=True)
        episodic_a, procedural_a = self._run_extract_session(loop_a)

        loop_b = self._build_loop(monkeypatch, tmp_path / "b", procedural_enabled=True)
        episodic_b, procedural_b = self._run_cycle_and_capture(monkeypatch, loop_b)

        # Identity fields must match across both paths.
        def _key(qa):
            return (qa["source_subject"], qa["source_predicate"], qa["source_object"])

        def _rel_key(rel):
            return (rel["subject"], rel["predicate"], rel["object"])

        assert sorted(map(_key, episodic_a)) == sorted(map(_key, episodic_b))
        assert sorted(map(_rel_key, procedural_a)) == sorted(map(_rel_key, procedural_b))

        # Partition invariant: no preference in episodic, and procedural carries
        # both the filter-sourced and the separately-extracted preference.
        assert all(qa["source_predicate"] != "prefers" for qa in episodic_a)
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
            return (qa["source_subject"], qa["source_predicate"], qa["source_object"])

        assert sorted(map(_key, episodic_a)) == sorted(map(_key, episodic_b))
        # With procedural disabled, preferences fall back into episodic — never lost.
        assert any(qa["source_predicate"] == "prefers" for qa in episodic_a)
        assert procedural_a == []
        assert procedural_b == []

    def test_empty(self):
        result = compute_semantic_drift([], set())
        assert result["mean_drift"] == 0.0


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
# Server consolidation — anonymous speaker skip removal (Slice 3-pre)
# ---------------------------------------------------------------------------


class TestAnonymousSpeakerNotSkipped:
    """Speaker{N} sessions must flow through extraction, not be silently discarded.

    Verifies that run_consolidation (paramem.server.consolidation) calls
    loop.extract_session for sessions whose speaker_id is 'Speaker3' —
    i.e. the old hard-skip on falsy speaker_id is gone.
    """

    def _make_mock_loop(self):
        """Minimal mock ConsolidationLoop with the attributes run_consolidation touches."""

        loop = MagicMock()
        loop.shutdown_requested = False
        loop.merger = MagicMock()
        loop.merger.graph = MagicMock()
        loop.merger.graph.nodes = []
        loop.indexed_key_qa = {}
        loop.key_sessions = {}
        loop.promoted_keys = set()
        loop.episodic_simhash = {}
        loop.semantic_simhash = {}
        loop.procedural_simhash = {}
        # extract_session returns ([], []) so no training path is triggered.
        loop.extract_session = MagicMock(return_value=([], []))
        loop.train_adapters = MagicMock(return_value={})
        loop.cycle_count = 0
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

    def test_anonymous_speaker_id_not_skipped(self, tmp_path):
        """Sessions with speaker_id='Speaker3' reach extract_session."""
        from paramem.server.consolidation import run_consolidation

        loop = self._make_mock_loop()
        config = self._make_config(tmp_path)
        buffer = self._make_session_buffer(tmp_path, speaker_id="Speaker3")

        run_consolidation(
            model=None,
            tokenizer=None,
            config=config,
            session_buffer=buffer,
            loop=loop,
        )

        loop.extract_session.assert_called_once()
        call_kwargs = loop.extract_session.call_args
        # First positional arg is the transcript; keyword arg is speaker_id.
        assert call_kwargs.kwargs.get("speaker_id") == "Speaker3"

    def test_named_speaker_not_skipped(self, tmp_path):
        """Named (enrolled) speaker IDs continue to reach extract_session."""
        from paramem.server.consolidation import run_consolidation

        loop = self._make_mock_loop()
        config = self._make_config(tmp_path)
        buffer = self._make_session_buffer(tmp_path, speaker_id="abc12345")

        run_consolidation(
            model=None,
            tokenizer=None,
            config=config,
            session_buffer=buffer,
            loop=loop,
        )

        loop.extract_session.assert_called_once()
        assert loop.extract_session.call_args.kwargs.get("speaker_id") == "abc12345"

    def test_none_speaker_id_still_skipped(self, tmp_path):
        """Truly-None speaker_id (text-only, no voice) is skipped at consolidation.

        Sessions with no speaker_id must not reach extract_session because a
        None speaker_id would key procedural_sp_index on (None, subject, predicate),
        causing unrelated text-only sessions to cross-retire each other's procedural keys.
        """
        from paramem.server.consolidation import run_consolidation

        loop = self._make_mock_loop()
        config = self._make_config(tmp_path)
        buffer = self._make_session_buffer(tmp_path, speaker_id=None)

        run_consolidation(
            model=None,
            tokenizer=None,
            config=config,
            session_buffer=buffer,
            loop=loop,
        )

        # Text-only sessions without a speaker_id must NOT reach extract_session.
        loop.extract_session.assert_not_called()


# ---------------------------------------------------------------------------
# _save_adapters: meta.json written in every saved slot (Slice 3a §2.4.1)
# ---------------------------------------------------------------------------


class TestSaveAdaptersManifest:
    """_save_adapters must embed meta.json in each adapter slot (Slice 3a)."""

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
        loop.indexed_key_registry = KeyRegistry()
        loop.indexed_key_qa = {}
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

        # Seed the registry and keyed_pairs so build_manifest_for has something to hash.
        from paramem.training.key_registry import KeyRegistry

        loop.indexed_key_registry = KeyRegistry()
        loop.indexed_key_registry.add("graph1", adapter_id="episodic")
        registry_path = tmp_path / "indexed_key_registry.json"
        loop.indexed_key_registry.save(registry_path)

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
        """_save_adapters → find_live_slot must match the fresh slot (I5 roundtrip).

        Regression guard for the blocker where ``_save_adapters`` hashed the
        stale on-disk registry, then overwrote it.  The post-save on-disk
        registry hash must equal ``manifest.registry_sha256`` so
        ``find_live_slot`` can mount the adapter after restart.
        """
        import hashlib

        from paramem.adapters.manifest import find_live_slot, read_manifest
        from paramem.training.key_registry import KeyRegistry

        loop = self._make_save_loop(tmp_path)

        # Seed registry + keyed_pairs state so every hash input is populated.
        loop.indexed_key_registry = KeyRegistry()
        loop.indexed_key_registry.add("graph1", adapter_id="episodic")
        loop.indexed_key_qa = {
            "graph1": {
                "key": "graph1",
                "question": "What colour is the sky?",
                "answer": "Blue.",
                "source_subject": "sky",
                "source_object": "blue",
            }
        }
        loop.episodic_simhash = {"graph1": 0xABCDEF}

        # Pre-seed a *different* registry file on disk so the old codepath
        # (hash-before-overwrite) would produce a mismatch — ensures this
        # test fails under the pre-fix implementation.
        stale_registry = KeyRegistry()
        stale_registry.add("stale_key", adapter_id="episodic")
        stale_registry.save(tmp_path / "indexed_key_registry.json")

        loop._save_adapters()

        # Live hash = hash of whatever is on disk post-save.
        live_hash = hashlib.sha256(
            (tmp_path / "indexed_key_registry.json").read_bytes()
        ).hexdigest()

        slot = find_live_slot(tmp_path / "episodic", live_hash)
        assert slot is not None, (
            "find_live_slot returned None — manifest.registry_sha256 does "
            "not match on-disk hash (I5 reorder broken)"
        )

        manifest = read_manifest(slot)
        assert manifest.registry_sha256 == live_hash
        # key_count must reflect the post-save registry (graph1), not the
        # stale pre-existing registry (stale_key).
        assert manifest.key_count == 1

        # keyed_pairs.json must live inside the adapter-kind dir and match
        # the hash stamped in the manifest.
        kp_path = tmp_path / "episodic" / "keyed_pairs.json"
        assert kp_path.exists()
        kp_hash = hashlib.sha256(kp_path.read_bytes()).hexdigest()
        assert manifest.keyed_pairs_sha256 == kp_hash


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
        cfg.debug = False
        cfg.debug_dir = None
        cfg.prompts_dir = None
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
            state_provider=lambda: state,
        )
        loop_a.fingerprint_cache["sentinel"] = "computed-by-cycle-1"

        loop_b = create_consolidation_loop(
            model=MagicMock(),
            tokenizer=MagicMock(),
            config=cfg,
            state_provider=lambda: state,
        )
        assert loop_b.fingerprint_cache.get("sentinel") == "computed-by-cycle-1"


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

    def test_is_full_cycle_due_no_prior_full_returns_true(self, tmp_path):
        """Fresh install (no prior window_stamp) → first full is due."""
        from paramem.server.app import _is_full_cycle_due

        cfg = self._make_config("every 84h", tmp_path)
        assert _is_full_cycle_due(cfg) is True

    def test_is_full_cycle_due_legacy_v1_treated_as_unknown(self, tmp_path):
        """A v1 manifest (empty window_stamp) is treated as unknown → due."""
        from paramem.server.app import _is_full_cycle_due

        self._write_meta(tmp_path / "episodic" / "20260427-072940", window_stamp="")
        cfg = self._make_config("every 84h", tmp_path)
        assert _is_full_cycle_due(cfg) is True

    def test_is_full_cycle_due_same_window_returns_false(self, tmp_path):
        """Last full's window_stamp matches current → already consolidated."""
        from paramem.server.app import _is_full_cycle_due
        from paramem.server.interim_adapter import current_full_consolidation_stamp

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
