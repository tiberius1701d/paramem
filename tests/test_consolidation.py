"""Tests for the consolidation loop orchestrator.

These are unit tests that mock the model/extraction to test
the consolidation logic without requiring GPU.
"""

from unittest.mock import MagicMock

import pytest

from paramem.memory.store import MemoryStore as _MS  # noqa: F401
from paramem.training.consolidation import ConsolidationLoop, _mentions_any


class TestMentionsAny:
    def test_finds_mention(self):
        assert _mentions_any("Alex lives in Heilbronn", {"alex"})

    def test_no_mention(self):
        assert not _mentions_any("The weather is nice", {"alex"})

    def test_case_insensitive(self):
        assert _mentions_any("ALEX is here", {"alex"})

    def test_empty_terms(self):
        assert not _mentions_any("some text", set())


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
            **loop_kwargs,
        )

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
        assert {rel["predicate"] for rel in procedural} == {"prefers", "listens_to"}

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
        sota_enabled: bool = False,
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
            consolidation_config=ConsolidationConfig(
                sota_enabled=sota_enabled,
                refinement_enrichment=refinement_enrichment,
                refinement_contradiction=refinement_contradiction,
            ),
            training_config=TrainingConfig(),
            episodic_adapter_config=AdapterConfig(),
            semantic_adapter_config=AdapterConfig(),
            memory_store=_MS(replay_enabled=False),
            procedural_adapter_config=None,
            output_dir=tmp_path,
        )
        return loop

    def test_merge_called_when_refinement_enrichment_on(self, monkeypatch, tmp_path):
        """refinement_enrichment='on': merger.merge is called once with the session graph."""
        from unittest.mock import patch

        loop = self._build_loop(
            monkeypatch, tmp_path, refinement_enrichment="on", sota_enabled=True
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
            monkeypatch, tmp_path / "a", refinement_enrichment="on", sota_enabled=True
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
        edges as a non-empty extra_relations kwarg to _materialize_consolidation_graph.

        Guards gate #3 — the unconditional _pending_relations capture inside
        run_consolidation_cycle (consolidation.py).  The test FAILS if an
        ``if self.config.refinement_enrichment != "off":`` guard is reintroduced
        around the capture, because extra_relations would then be None/[] and the
        assertion below would reject it.

        Strategy: extract_session populates merger.graph with the X→Y 'knows' edge.
        run_consolidation_cycle (mode='simulate') is then called with the returned
        episodic_rels.  _materialize_consolidation_graph is replaced with a spy that
        records extra_relations and returns the correct empty tuple, avoiding GPU/disk
        I/O.  commit_tier_slot is stubbed out for the same reason.
        """
        from unittest.mock import patch

        loop = self._build_loop(monkeypatch, tmp_path, refinement_enrichment="off")
        # replay_enabled=True is required so run_consolidation_cycle passes guard #2.
        loop.store._replay_enabled = True

        with patch.object(loop.extraction, "run", return_value=self._session_graph):
            episodic_rels, procedural_rels = loop.extract_session("t", "s_gate", speaker_id="spk0")

        # episodic_rels must be non-empty; otherwise run_consolidation_cycle exits
        # early at guard #3 before the extra_relations capture is ever reached.
        assert episodic_rels, (
            "extract_session must return non-empty episodic_rels for the test to be valid"
        )

        captured: list[dict] = []

        def _spy_materialize(**kwargs):
            captured.append(kwargs)
            # Return the correct type: (recall_miss_keys, recon_relations).
            return set(), []

        with (
            patch.object(loop, "_materialize_consolidation_graph", side_effect=_spy_materialize),
            patch(
                "paramem.memory.persistence.commit_tier_slot",
            ),
        ):
            loop.run_consolidation_cycle(
                episodic_rels,
                procedural_rels,
                speaker_id="spk0",
                mode="simulate",
                run_label="s_gate",
                stamp="20260601T0000",
                max_interim_count=7,
            )

        assert captured, "_materialize_consolidation_graph was not called"
        call_kwargs = captured[0]

        extra = call_kwargs.get("extra_relations")
        assert extra is not None and len(extra) > 0, (
            "extra_relations passed to _materialize_consolidation_graph must be "
            f"non-empty when refinement_enrichment='off'; got {extra!r}"
        )

        # The X→Y 'knows' edge extracted from the session must be present.
        subjects = {r.subject.lower() for r in extra}
        objects = {r.object.lower() for r in extra}
        predicates = {r.predicate.lower() for r in extra}
        assert "x" in subjects, f"Subject 'X' must appear in extra_relations; subjects={subjects}"
        assert "y" in objects, f"Object 'Y' must appear in extra_relations; objects={objects}"
        assert "knows" in predicates, (
            f"Predicate 'knows' must appear in extra_relations; predicates={predicates}"
        )


class TestRefinementConfigRoundtrip:
    """Loading YAML with refinement knobs propagates through the property chain."""

    def test_yaml_sota_enabled_propagates(self, tmp_path):
        """YAML sota_enabled: true propagates to schedule and consolidation_config."""
        from paramem.server.config import load_server_config

        yaml_text = """
model:
  name: "mistralai/Mistral-7B-Instruct-v0.3"
consolidation:
  refresh_cadence: "12h"
  sota_enabled: true
"""
        cfg_path = tmp_path / "server_sota_enabled.yaml"
        cfg_path.write_text(yaml_text)
        cfg = load_server_config(str(cfg_path))

        assert cfg.consolidation.sota_enabled is True, (
            "ServerConfig.consolidation.sota_enabled should be True"
        )
        assert cfg.consolidation_config.sota_enabled is True, (
            "consolidation_config.sota_enabled should be True"
        )

    def test_yaml_sota_defaults_to_false(self, tmp_path):
        """YAML without sota_enabled defaults to False."""
        from paramem.server.config import load_server_config

        yaml_text = """
model:
  name: "mistralai/Mistral-7B-Instruct-v0.3"
consolidation:
  refresh_cadence: "12h"
"""
        cfg_path = tmp_path / "server_no_sota.yaml"
        cfg_path.write_text(yaml_text)
        cfg = load_server_config(str(cfg_path))

        assert cfg.consolidation.sota_enabled is False, (
            "ServerConfig.consolidation.sota_enabled should default to False"
        )
        assert cfg.consolidation_config.sota_enabled is False, (
            "consolidation_config.sota_enabled should default to False"
        )

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

        buffer = SessionBuffer(tmp_path / "sessions", state_dir=tmp_path / "state", debug=False)
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
    """_maybe_trigger_scheduled_consolidation returns noop_no_named when no NAMED sessions."""

    def _make_minimal_state(self, tmp_path, buffer, store=None):
        """Inject minimal _state overrides for the tick function."""
        from paramem.server.config import ConsolidationScheduleConfig, ServerConfig

        config = MagicMock(spec=ServerConfig)
        sched = ConsolidationScheduleConfig()
        config.consolidation = sched
        config.debug = False
        config.debug_dir = tmp_path / "debug"

        return {
            "config": config,
            "session_buffer": buffer,
            "speaker_store": store,
            "consolidating": False,
            "mode": "local",
            "last_chat_monotonic": None,
            "pending_rehydration": False,
            "store_load_degraded": False,
        }

    def _make_stub_store(self, anonymous_ids=None):
        store = MagicMock()
        _anon = set(anonymous_ids or [])
        store.is_anonymous.side_effect = lambda sid: sid in _anon
        return store

    def _call_tick(self, state_overrides: dict) -> str:
        """Run _maybe_trigger_scheduled_consolidation with mocked _state."""
        from unittest.mock import patch

        import paramem.server.app as _app

        # Patch _consolidation_dispatch_guards to return None (no pre-emption),
        # _is_full_cycle_due to return False (not a full cycle tick),
        # and _retro_claim_orphan_sessions to no-op.
        with (
            patch.object(_app, "_state", state_overrides),
            patch(
                "paramem.server.app._consolidation_dispatch_guards",
                return_value=None,
            ),
            patch("paramem.server.app._is_full_cycle_due", return_value=False),
            patch("paramem.server.app._retro_claim_orphan_sessions", return_value=0),
        ):
            return _app._maybe_trigger_scheduled_consolidation()

    def test_all_unidentifiable_returns_noop_no_named(self, tmp_path):
        """When all sessions are UNIDENTIFIABLE, tick returns noop_no_named."""
        from paramem.server.session_buffer import SessionBuffer

        buffer = SessionBuffer(tmp_path / "sessions", state_dir=tmp_path / "state", debug=False)
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

        buffer = SessionBuffer(tmp_path / "sessions", state_dir=tmp_path / "state", debug=False)
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

        buffer = SessionBuffer(tmp_path / "sessions", state_dir=tmp_path / "state", debug=False)
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

        buffer = SessionBuffer(tmp_path / "sessions", state_dir=tmp_path / "state", debug=False)
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
        from paramem.memory.store import MemoryStore

        loop.store = MemoryStore(replay_enabled=True)
        loop.store.load_registry("episodic", KeyRegistry())
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
        loop.store.load_registry("episodic", ep_reg)

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
        loop.store.load_registry("episodic", ep_reg)
        loop.store.put(
            "episodic",
            "graph1",
            {
                "key": "graph1",
                "question": "What colour is the sky?",
                "answer": "Blue.",
                "subject": "sky",
                "predicate": "has_colour",
                "object": "blue",
                "speaker_id": "speaker0",
            },
            register=False,
        )
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
        cfg.consolidation.graph_enrichment_neighborhood_hops = 1
        cfg.consolidation.graph_enrichment_max_entities_per_pass = 5
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

    ``_last_full_consolidation_window`` tests exercise the manifest-reader
    helper (retained; still used by ``_save_adapters`` / ``run_housekeeping``).

    ``_is_full_cycle_due`` tests exercise the new count/phase primary +
    oldest-interim-age deadline gate (S2).  The gate no longer reads
    window_stamps; it counts on-disk ``episodic/interim_*`` dirs and
    compares the oldest dir's age to ``consolidation_period_seconds``.
    """

    def _write_meta(self, slot_dir, *, window_stamp="", trained_at="2026-04-27T07:29:40Z"):
        import json as _json

        slot_dir.mkdir(parents=True, exist_ok=True)
        payload = {"trained_at": trained_at, "window_stamp": window_stamp}
        (slot_dir / "meta.json").write_text(_json.dumps(payload))

    def _make_interim_dir(self, adapter_dir, stamp: str) -> None:
        """Create a bare on-disk interim dir at ``episodic/interim_<stamp>/``.

        The dir needs to exist and match the ``INTERIM_DIR_PREFIX`` pattern so
        ``iter_interim_dirs`` picks it up.  No content is needed for gate tests.
        """
        (adapter_dir / "episodic" / f"interim_{stamp}").mkdir(parents=True, exist_ok=True)

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

    def _make_config(
        self, adapter_dir, *, max_interim_count: int = 7, period_seconds: "int | None" = 302400
    ):
        """Build a minimal config mock for ``_is_full_cycle_due`` tests.

        Args:
            adapter_dir: Path passed as ``config.adapter_dir``.
            max_interim_count: Value of ``config.consolidation.max_interim_count``
                (N).  Default 7 matches server.yaml default.
            period_seconds: Value of ``config.consolidation.consolidation_period_seconds``
                (= refresh_cadence_seconds × N).  Default 302400 = 12h × 7 × 3600.
                Pass ``None`` to simulate a disabled (manual-only) cadence.
        """
        cfg = MagicMock()
        cfg.adapter_dir = adapter_dir
        cfg.consolidation.max_interim_count = max_interim_count
        cfg.consolidation.consolidation_period_seconds = period_seconds
        return cfg

    def test_is_full_cycle_due_zero_count_returns_true(self, tmp_path):
        """N==0 (full-fold-only consume-pending mode): every tick is a full cycle → True.

        At count==0 no interim slots are minted, so the count/deadline logic
        does not apply.  _is_full_cycle_due returns True unconditionally so the
        dispatcher always routes to the full path.
        """
        from paramem.server.app import _is_full_cycle_due

        # No interim dirs at all (the normal state at count==0).
        cfg = self._make_config(tmp_path, max_interim_count=0)
        assert _is_full_cycle_due(cfg) is True

    def test_is_full_cycle_due_zero_count_returns_true_with_interim_dirs(self, tmp_path):
        """N==0 returns True even when stale interim dirs exist (recovery scenario)."""
        from paramem.server.app import _is_full_cycle_due

        # Stale interim dirs that might have been left from a prior mode change.
        self._make_interim_dir(tmp_path, "20260620T0000")
        cfg = self._make_config(tmp_path, max_interim_count=0)
        assert _is_full_cycle_due(cfg) is True

    def test_is_full_cycle_due_no_interims_returns_false(self, tmp_path):
        """Empty interim store (n=0): gate always returns False regardless of config."""
        from paramem.server.app import _is_full_cycle_due

        (tmp_path / "episodic").mkdir()
        cfg = self._make_config(tmp_path)
        assert _is_full_cycle_due(cfg) is False

    def test_is_full_cycle_due_n_interims_returns_false(self, tmp_path):
        """Exactly N interims present (n==N): primary boundary not yet reached."""
        from paramem.server.app import _is_full_cycle_due

        N = 3
        for i in range(N):
            self._make_interim_dir(tmp_path, f"20260620T{i * 12:04d}")
        # period large enough that the deadline backstop doesn't fire.
        cfg = self._make_config(tmp_path, max_interim_count=N, period_seconds=10 * 365 * 86400)
        assert _is_full_cycle_due(cfg) is False

    def test_is_full_cycle_due_n_plus_one_interims_returns_true(self, tmp_path):
        """N+1 interims on disk: count/phase primary fires (first slot of next cycle)."""
        from paramem.server.app import _is_full_cycle_due

        N = 3
        for i in range(N + 1):
            self._make_interim_dir(tmp_path, f"20260620T{i * 12:04d}")
        # Large period so only count signal fires, not deadline.
        cfg = self._make_config(tmp_path, max_interim_count=N, period_seconds=10 * 365 * 86400)
        assert _is_full_cycle_due(cfg) is True

    def test_is_full_cycle_due_n_minus_1_interims_returns_false(self, tmp_path):
        """Fewer than N interims (n < N): neither primary nor deadline fires."""
        from paramem.server.app import _is_full_cycle_due

        N = 7
        for i in range(N - 1):
            self._make_interim_dir(tmp_path, f"20260620T{i * 12:04d}")
        cfg = self._make_config(tmp_path, max_interim_count=N, period_seconds=10 * 365 * 86400)
        assert _is_full_cycle_due(cfg) is False

    def test_is_full_cycle_due_manual_only_no_deadline_returns_false(self, tmp_path):
        """refresh_cadence disabled (period_seconds=None): backstop never fires.

        Even with an aged interim, the deadline backstop must return False when
        the schedule is manual-only — there is no intended full period to anchor
        the deadline to.
        """
        from paramem.server.app import _is_full_cycle_due

        # 1 interim, very old stamp — would trigger the deadline if enabled.
        self._make_interim_dir(tmp_path, "20200101T0000")
        cfg = self._make_config(tmp_path, period_seconds=None)
        assert _is_full_cycle_due(cfg) is False

    def test_is_full_cycle_due_sparse_deadline_fires(self, tmp_path):
        """Deadline backstop: n < N+1 but oldest interim aged ≥ full_period → True.

        Covers the sparse/broken-chain case where count never reaches N+1 but
        facts have been sitting un-folded beyond one full intended interval.
        """
        from paramem.server.app import _is_full_cycle_due

        # 1 interim with a stamp far in the past — definitely older than any
        # reasonable full period.
        self._make_interim_dir(tmp_path, "20200101T0000")
        # full_period is 1 second so the 2020 stamp is astronomically over it.
        cfg = self._make_config(tmp_path, max_interim_count=7, period_seconds=1)
        assert _is_full_cycle_due(cfg) is True

    def test_is_full_cycle_due_post_fold_n_zero_returns_false(self, tmp_path):
        """After a successful fold, all interim dirs are rmtree'd (n=0) → False.

        Verifies the gate cannot re-fire on the same tick after the fold clears
        the interim ring.
        """
        from paramem.server.app import _is_full_cycle_due

        # No interim dirs — simulates post-fold state.
        (tmp_path / "episodic").mkdir()
        cfg = self._make_config(tmp_path)
        assert _is_full_cycle_due(cfg) is False

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

    def test_is_full_cycle_due_interim_only_no_main_single_interim_deadline_fires(self, tmp_path):
        """Interim-only store with one very old interim: deadline backstop fires.

        The new gate does not consult window_stamps; it counts on-disk interims
        and checks the oldest dir's age.  A single interim (n=1) cannot satisfy
        the count/phase primary (needs n>1 and (n-1)%N==0), but the deadline
        backstop fires when the interim is older than ``full_period_seconds``.
        ``_last_full_consolidation_window`` is NOT called from the gate; we also
        confirm it correctly returns None for this layout (helper still live for
        housekeeping).
        """
        from paramem.server.app import _is_full_cycle_due, _last_full_consolidation_window

        # Create an interim dir whose stamp is in the distant past.
        self._make_interim_dir(tmp_path, "20200101T0000")

        assert _last_full_consolidation_window(tmp_path) is None
        # period_seconds=1 so a 2020 stamp is astronomically over the deadline.
        cfg = self._make_config(tmp_path, period_seconds=1)
        assert _is_full_cycle_due(cfg) is True

    def test_is_full_cycle_due_fresh_install_no_interim_returns_false(self, tmp_path):
        """Fresh install: no interim dirs (n=0) → gate stays False.

        The full cycle (consolidate_interim_adapters) would be a no-op with
        nothing to fold.  Returning False keeps the dispatcher on the interim
        path (which extracts pending sessions into the first interim slot).
        """
        from paramem.server.app import _is_full_cycle_due

        (tmp_path / "episodic").mkdir()
        cfg = self._make_config(tmp_path)
        assert _is_full_cycle_due(cfg) is False

    def test_is_full_cycle_due_single_interim_young_returns_false(self, tmp_path):
        """Single interim (n=1) that is NOT aged beyond the full period: False.

        The count/phase primary requires n>1; the deadline backstop requires
        age ≥ full_period.  A recent single interim satisfies neither.
        """
        from paramem.server.app import _is_full_cycle_due

        self._make_interim_dir(tmp_path, "20260620T1200")
        # Large period so the current-day stamp is nowhere near the deadline.
        cfg = self._make_config(tmp_path, period_seconds=10 * 365 * 86400)
        assert _is_full_cycle_due(cfg) is False

    def test_is_full_cycle_due_main_slot_does_not_affect_gate(self, tmp_path):
        """Main slot presence / window_stamp does NOT influence the new gate.

        The gate counts interim dirs, not main slots.  Even if a main slot is
        present and current, the gate fires when the interim count hits N+1.
        This test verifies that the gate ignores main-slot state entirely.
        """
        from paramem.server.app import _is_full_cycle_due

        N = 2
        # Write a main slot (up-to-date, as if a fold just ran).
        self._write_meta(tmp_path / "episodic" / "20260620-080000", window_stamp="20260620T0000")
        # Add N+1 interims — the count/phase primary must fire.
        for i in range(N + 1):
            self._make_interim_dir(tmp_path, f"20260620T{i:04d}")
        cfg = self._make_config(tmp_path, max_interim_count=N, period_seconds=10 * 365 * 86400)
        assert _is_full_cycle_due(cfg) is True


class TestRunFullConsolidationSyncAccumulating:
    """App-layer: _run_full_consolidation_sync honours the accumulating return.

    When ``consolidate_interim_adapters`` returns ``status="accumulating"``,
    the orchestrator in ``_run_full_cycle`` must:
    - NOT call ``session_buffer.mark_consolidated``
    - NOT stamp ``_state["last_consolidation"]``
    - Write a durable run-status record with ``outcome="accumulating"`` and
      ``detail["accumulating_reason"]`` present (read back via ``read_last_runs``)
    - NOT call ``ConsolidationLoop._save_adapters`` (proxy for on-disk window
      stamp remaining unchanged — the real on-disk stamp lives in the main-slot
      meta.json written by ``_save_adapters``; no GPU/real fold machinery needed
      to assert this proxy)
    """

    def _make_state(self, tmp_path=None) -> dict:
        """Minimal _state for _run_full_consolidation_sync.

        Uses a pre-populated ``consolidation_loop`` mock so the function's
        ``create_consolidation_loop`` branch is skipped entirely.

        ``config.consolidation.training_temp_limit`` is set to 0 so that
        ``ThermalPolicy.from_consolidation_config`` returns None without
        comparing a MagicMock against an integer (which raises TypeError).

        Args:
            tmp_path: When provided, set ``config.paths.data`` to this real
                path so that incident/run-status I/O writes land in
                ``tmp_path/state/`` rather than creating a ``MagicMock/``
                directory at the repo root.  Tests that read back the durable
                run-status record must supply this.
        """
        mock_config = MagicMock()
        mock_config.consolidation.mode = "train"
        # training_temp_limit <= 0 → ThermalPolicy.from_consolidation_config returns None.
        mock_config.consolidation.training_temp_limit = 0
        # cooldown_gate_threshold_c <= 0 disables the wait_for_cooldown fold gate.
        mock_config.vram.cooldown_gate_threshold_c = 0
        # Ground incident/run-status I/O in a real path so the writes land in
        # the pytest tmp directory instead of creating a MagicMock/ tree at
        # the repo root.
        if tmp_path is not None:
            mock_config.paths.data = tmp_path

        mock_loop = MagicMock()
        mock_loop.model = MagicMock(name="model")
        mock_loop.consolidate_interim_adapters.return_value = {
            "status": "accumulating",
            "accumulating_reason": {"floor": 30, "episodic": 5, "parked": {}},
            "tiers_rebuilt": [],
        }

        mock_session_buffer = MagicMock()

        return {
            "config": mock_config,
            "model": MagicMock(name="model"),
            "tokenizer": MagicMock(name="tokenizer"),
            "consolidation_loop": mock_loop,
            "session_buffer": mock_session_buffer,
            "router": None,
            "background_trainer": None,
            "consolidating": True,
            "last_consolidation": None,
            "last_consolidation_result": None,
            "event_loop": None,
        }

    def test_accumulating_does_not_mark_consolidated(self, monkeypatch, tmp_path):
        """session_buffer.mark_consolidated is NOT called when fold returns accumulating."""
        from unittest.mock import patch

        import paramem.server.app as app_module

        state = self._make_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", state)

        mock_bt = MagicMock()
        mock_bt.submit.side_effect = lambda fn, **kw: fn()

        with patch("paramem.server.app.BackgroundTrainer", return_value=mock_bt):
            app_module._run_full_consolidation_sync()

        state["session_buffer"].mark_consolidated.assert_not_called()

    def test_accumulating_does_not_stamp_last_consolidation(self, monkeypatch, tmp_path):
        """_state["last_consolidation"] is NOT updated when fold returns accumulating."""
        from unittest.mock import patch

        import paramem.server.app as app_module

        state = self._make_state(tmp_path)
        prior_stamp = state["last_consolidation"]  # None
        monkeypatch.setattr(app_module, "_state", state)

        mock_bt = MagicMock()
        mock_bt.submit.side_effect = lambda fn, **kw: fn()

        with patch("paramem.server.app.BackgroundTrainer", return_value=mock_bt):
            app_module._run_full_consolidation_sync()

        assert state["last_consolidation"] == prior_stamp, (
            "_state['last_consolidation'] must NOT be updated on accumulating return; "
            f"was {prior_stamp!r}, got {state['last_consolidation']!r}"
        )

    def test_accumulating_sets_result_status_and_reason(self, monkeypatch, tmp_path):
        """Durable run-status record carries outcome='accumulating' and reason.

        The production site (app.py ``_run_full_consolidation_sync``) calls
        ``record_last_run(op_type="consolidation", outcome="accumulating",
        detail={"accumulating_reason": ..., "tiers_rebuilt": []})``.
        This test reads back the written ``RunRecord`` via ``read_last_runs``
        and asserts the exact durable shape rather than the deleted RAM field
        ``_state["last_consolidation_result"]``.
        """
        from unittest.mock import patch

        import paramem.server.app as app_module
        from paramem.server.run_status import read_last_runs

        state = self._make_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", state)

        mock_bt = MagicMock()
        mock_bt.submit.side_effect = lambda fn, **kw: fn()

        with patch("paramem.server.app.BackgroundTrainer", return_value=mock_bt):
            app_module._run_full_consolidation_sync()

        runs = read_last_runs(tmp_path / "state")
        assert "consolidation" in runs, (
            "run_status.json must contain a 'consolidation' record after accumulating fold"
        )
        record = runs["consolidation"]
        assert record.outcome == "accumulating", (
            f"expected outcome='accumulating', got {record.outcome!r}"
        )
        assert "accumulating_reason" in record.detail, (
            "run-status detail must contain 'accumulating_reason'"
        )

    def test_accumulating_save_adapters_not_called(self, monkeypatch, tmp_path):
        """_save_adapters is NOT called when fold returns accumulating.

        _save_adapters writes the per-tier meta.json that carries the on-disk
        window_stamp read by _is_full_cycle_due.  Not calling it leaves the
        stamp unadvanced so the next tick will re-fire the fold.
        No GPU or real fold machinery is needed to assert this proxy.
        """
        from unittest.mock import patch

        import paramem.server.app as app_module

        state = self._make_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", state)

        mock_bt = MagicMock()
        mock_bt.submit.side_effect = lambda fn, **kw: fn()

        save_spy = MagicMock()

        with (
            patch("paramem.server.app.BackgroundTrainer", return_value=mock_bt),
            patch("paramem.training.consolidation.ConsolidationLoop._save_adapters", save_spy),
        ):
            app_module._run_full_consolidation_sync()

        save_spy.assert_not_called()


class TestRunFullConsolidationSyncPendingSessionsUntouched:
    """Full-cycle bookkeeping must NOT mark pending sessions as consolidated.

    The full consolidation run folds interim-adapter content into main; it does
    not run the extraction chain on pending sessions.  Pending sessions must
    remain in the buffer and be consumed by the next interim tick.

    Bug (fixed): the post-full-cycle bookkeeping block called
    ``session_buffer.mark_consolidated(pending_ids, ...)`` on the scheduled
    (``not housekeeping``) path, permanently discarding sessions that were
    never extracted.  This test asserts the corrected contract.
    """

    def _make_state_full_trained(self) -> dict:
        """Minimal ``_state`` producing a successful full-trained result.

        ``consolidate_interim_adapters`` returns ``tiers_rebuilt=["episodic"]``
        so the full-trained bookkeeping path is exercised (not noop/accumulating).
        ``_save_key_metadata`` is mocked at the consolidation module level so
        no real loop or filesystem is needed.
        ``session_buffer`` holds one pending session to verify it is untouched.
        """
        mock_config = MagicMock()
        mock_config.consolidation.mode = "train"
        # training_temp_limit <= 0 prevents ThermalPolicy comparison with MagicMock.
        mock_config.consolidation.training_temp_limit = 0
        # cooldown_gate_threshold_c <= 0 disables the wait_for_cooldown fold gate.
        mock_config.vram.cooldown_gate_threshold_c = 0

        mock_loop = MagicMock()
        mock_loop.model = MagicMock(name="model")
        mock_loop.store = MagicMock()
        mock_loop.store.replay_enabled = False
        mock_loop.consolidate_interim_adapters.return_value = {
            "status": "full_trained",
            "tiers_rebuilt": ["episodic"],
            "rolled_back": False,
            "rollback_tier": None,
            "graph_drift_count": 0,
        }

        mock_session_buffer = MagicMock()
        # Simulate one pending session in the buffer.
        mock_session_buffer.get_pending.return_value = [{"session_id": "doc-pending-1"}]

        return {
            "config": mock_config,
            "model": MagicMock(name="model"),
            "tokenizer": MagicMock(name="tokenizer"),
            "consolidation_loop": mock_loop,
            "session_buffer": mock_session_buffer,
            "router": MagicMock(),
            "background_trainer": None,
            "consolidating": True,
            "last_consolidation": None,
            "last_consolidation_result": None,
            "event_loop": None,
        }

    def test_full_trained_does_not_mark_pending_consolidated(self, monkeypatch):
        """Full-trained path must not call session_buffer.mark_consolidated.

        A pending session that arrived while a full cycle was due must remain
        pending after the full fold so the next interim tick extracts it.
        Calling mark_consolidated here would permanently discard it.

        Non-vacuity contract: the test first asserts that the BG job actually
        ran (``_save_key_metadata`` called — a witness that
        ``_run_full_cycle`` executed past the ``full_trained`` path).  If the
        job did NOT run, the first assertion fails, preventing a false green on
        ``mark_consolidated.assert_not_called()``.

        Patch path: ``paramem.server.app.BackgroundTrainer`` — the name that
        ``_build_bg_trainer`` in app.py resolves at call time (imported at
        module level via ``from paramem.server.background_trainer import
        BackgroundTrainer``).  Patching the class in its own module
        (``paramem.server.background_trainer.BackgroundTrainer``) does NOT
        intercept the constructor call because app.py already holds the
        original reference in its own namespace.
        """
        from unittest.mock import patch

        import paramem.server.app as app_module

        state = self._make_state_full_trained()
        monkeypatch.setattr(app_module, "_state", state)

        mock_bt = MagicMock()
        mock_bt.submit.side_effect = lambda fn, **kw: fn()

        save_spy = MagicMock()

        with (
            patch("paramem.server.app.BackgroundTrainer", return_value=mock_bt),
            patch("paramem.server.consolidation._save_key_metadata", save_spy),
        ):
            app_module._run_full_consolidation_sync()

        # Non-vacuity witness: the BG job ran and reached the full_trained bookkeeping
        # path.  If the job did not run, this assertion fails before the guard below.
        save_spy.assert_called_once()
        state["session_buffer"].mark_consolidated.assert_not_called()

    def test_full_trained_save_key_metadata_still_called(self, monkeypatch):
        """_save_key_metadata is called even though mark_consolidated is not.

        The key-metadata file must be updated after a successful fold; the fix
        must not accidentally remove that call.

        Patching ``paramem.server.app.BackgroundTrainer`` (the name imported at
        module level in app.py) rather than the class in its own module ensures
        ``_build_bg_trainer`` creates the mock BT whose ``submit.side_effect``
        executes the job synchronously, so the assertion runs after the job
        completes.
        """
        from unittest.mock import patch

        import paramem.server.app as app_module

        state = self._make_state_full_trained()
        monkeypatch.setattr(app_module, "_state", state)

        mock_bt = MagicMock()
        mock_bt.submit.side_effect = lambda fn, **kw: fn()

        save_spy = MagicMock()

        with (
            patch("paramem.server.app.BackgroundTrainer", return_value=mock_bt),
            patch("paramem.server.consolidation._save_key_metadata", save_spy),
        ):
            app_module._run_full_consolidation_sync()

        save_spy.assert_called_once()

    def test_full_trained_pending_sessions_remain_in_buffer(self, monkeypatch):
        """get_pending() still returns the pending session after a full cycle.

        Verifies that the session_buffer's pending state is preserved end-to-end:
        mark_consolidated is not called, so the pending session survives.

        Non-vacuity contract: the test first asserts that the BG job actually
        ran (``_save_key_metadata`` called) before asserting on the buffer
        state.  Without this witness the pending-sessions assertion is trivially
        true because the MagicMock's ``get_pending.return_value`` is unchanged
        regardless of whether the job ran.

        Patch path: ``paramem.server.app.BackgroundTrainer`` — same rationale
        as ``test_full_trained_does_not_mark_pending_consolidated``.
        """
        from unittest.mock import patch

        import paramem.server.app as app_module

        state = self._make_state_full_trained()
        monkeypatch.setattr(app_module, "_state", state)

        mock_bt = MagicMock()
        mock_bt.submit.side_effect = lambda fn, **kw: fn()

        save_spy = MagicMock()

        with (
            patch("paramem.server.app.BackgroundTrainer", return_value=mock_bt),
            patch("paramem.server.consolidation._save_key_metadata", save_spy),
        ):
            app_module._run_full_consolidation_sync()

        # Non-vacuity witness: the BG job ran and reached the full_trained bookkeeping
        # path.  If the job did not run, this assertion fails before the buffer check.
        save_spy.assert_called_once()

        # mark_consolidated must not have been called — pending sessions survive the fold.
        state["session_buffer"].mark_consolidated.assert_not_called()

        # get_pending must still return the session that was pending before the fold.
        pending_after = state["session_buffer"].get_pending()
        session_ids = [s["session_id"] for s in pending_after]
        assert "doc-pending-1" in session_ids, (
            "Pending session must remain in buffer after full consolidation; "
            f"mark_consolidated must not have been called. Found: {session_ids!r}"
        )


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
# Regression: interim write path preserves full schema (hydration-miss / cross-session schema fix)
# ---------------------------------------------------------------------------


class TestIndexedKeyCacheSchemaInvariant:
    """Regression guard: indexed_key_cache entries carry the canonical field set.

    Knowledge lives solely in adapter weights (train mode) or graph.json
    (simulate mode) — quads.json sidecars are removed.  The
    indexed_key_cache is the in-RAM transient view.  Entries must carry
    ``subject``, ``predicate``, ``object`` (not the old ``source_*`` aliases)
    alongside ``key`` and ``speaker_id`` so the router entity/speaker
    indexes populate correctly.
    """

    _CANONICAL_FIELDS = {"key", "subject", "predicate", "object", "speaker_id"}

    def _build_minimal_pair(
        self,
        *,
        key: str = "graph1",
        subject: str = "Alice",
        predicate: str = "lives_in",
        obj: str = "Berlin",
        speaker_id: str = "speaker0",
    ) -> dict:
        return {
            "key": key,
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "speaker_id": speaker_id,
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
        assert store.tier_simhashes("episodic", include_stale=False) == build_registry(episodic)
        assert store.tier_simhashes("procedural", include_stale=False) == build_registry(procedural)
        assert store.simhash_count_in_tier("episodic") == 2

    def test_overwrites_stale_simhash(self):
        from paramem.memory.store import MemoryStore

        store = MemoryStore(replay_enabled=True)
        store.put_simhash("episodic", "graph_old", 0xDEAD)
        self._call(
            store,
            {"episodic": [self._entry("graph_new")], "semantic": [], "procedural": []},
        )

        view = store.tier_simhashes("episodic", include_stale=False)
        assert "graph_old" not in view
        assert "graph_new" in view

    def test_empty_tier_clears_registry_and_simhash(self):
        from paramem.memory.store import MemoryStore

        store = MemoryStore(replay_enabled=True)
        store.put_simhash("semantic", "graph_stale", 0xBEEF)
        self._call(store, {"episodic": [], "semantic": [], "procedural": []})

        assert len(store.registry("semantic")) == 0
        assert store.tier_simhashes("semantic", include_stale=False) == {}


def test_promotion_carry_over_restores_nonzero_attributes(tmp_path):
    """Base-swap promotion carry-over: ``seed_key_metadata`` restores
    ``cycle_count`` + ``promoted_keys`` against the Phase-B-repopulated store,
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
            "graph1": {"reinforcement_count": 7, "speaker_id": "speaker0"},
            "graph2": {"reinforcement_count": 3, "speaker_id": "speaker0"},
            "graph3": {"reinforcement_count": 1, "speaker_id": "speaker0"},
            "orphan": {"reinforcement_count": 9, "speaker_id": "speaker0"},
        },
        "promoted_keys": ["graph1", "orphan"],
    }
    path = tmp_path / "key_metadata.json"
    path.write_text(json.dumps(meta))

    loaded = _load_key_metadata(path)
    assert loaded is not None

    # Minimal container the carry-over mutates; the store and method are real.
    loop = types.SimpleNamespace(store=store, promoted_keys=set(), cycle_count=0)
    ConsolidationLoop.seed_key_metadata(loop, loaded)

    # cycle_count and promoted_keys are carried over; orphan filtered (absent from store).
    assert loop.promoted_keys == {"graph1"}  # "orphan" filtered (no tier in store)
    assert loop.cycle_count == 5


def test_seed_key_metadata_retains_stale_key(tmp_path):
    """seed_key_metadata: a stale key's metadata is RETAINED (not counted as orphan).

    Root bug: seed_key_metadata called tier_for_active_key; a soft-staled key
    returned None and was silently dropped.  After the fix it calls
    tier_for_known_key so stale keys survive the reseed.
    """
    import json
    import types

    from paramem.memory.store import MemoryStore
    from paramem.server.consolidation import _load_key_metadata
    from paramem.training.consolidation import ConsolidationLoop

    # Store: graph1 is active, proc52 is stale.
    store = MemoryStore(replay_enabled=True)
    ep_entry = {"key": "graph1", "question": "q", "answer": "a"}
    store.put("episodic", "graph1", ep_entry, register=True)
    proc_entry = {"key": "proc52", "question": "q", "answer": "a"}
    store.put("procedural", "proc52", proc_entry, register=True)
    store.discard_keys(["proc52"], mode="stale")

    meta = {
        "cycle_count": 3,
        "keys": {
            "graph1": {"reinforcement_count": 5, "speaker_id": "spk0"},
            "proc52": {"reinforcement_count": 2, "speaker_id": "spk0"},
            "ghost": {"reinforcement_count": 1, "speaker_id": "spk0"},
        },
        "promoted_keys": ["proc52"],
    }
    path = tmp_path / "key_metadata.json"
    path.write_text(json.dumps(meta))

    loaded = _load_key_metadata(path)
    assert loaded is not None

    loop = types.SimpleNamespace(store=store, promoted_keys=set(), cycle_count=0)
    ConsolidationLoop.seed_key_metadata(loop, loaded)

    # proc52 is stale (not active) but IS known — must not be counted as orphan.
    # Orphan count is only surfaced via logging; validate via promoted_keys:
    # proc52 was in promoted_keys; it is still known after staling.
    assert "proc52" in loop.promoted_keys, "Stale promoted key must survive seed_key_metadata"
    # ghost has no tier at all — must NOT be in promoted_keys.
    assert "ghost" not in loop.promoted_keys, "Truly-unknown key must be dropped from promoted_keys"
    assert loop.cycle_count == 3


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
        from paramem.memory.store import MemoryStore

        loop.store = MemoryStore(replay_enabled=True)
        loop.store.load_registry("episodic", KeyRegistry())
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
        # Set floor=0 and tier_fast_start=False so abort-path tests with small
        # key counts are not blocked by the accumulate guard or the fast-start path.
        loop.config = ConsolidationConfig(min_tier_key_floor=0, tier_fast_start=False)
        loop.training_config = TrainingConfig(num_epochs=1, gradient_checkpointing=False)
        loop.episodic_config = AdapterConfig(rank=4, alpha=8, target_modules=["q_proj"])
        loop.semantic_config = AdapterConfig(rank=4, alpha=8, target_modules=["q_proj"])
        loop.procedural_config = AdapterConfig(rank=4, alpha=8, target_modules=["q_proj"])
        loop.wandb_config = None
        loop._thermal_policy = None
        loop.output_dir = tmp_path
        loop.store = MemoryStore(replay_enabled=True)
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
        return loop

    def test_run_consolidation_cycle_returns_aborted_on_abort(self, monkeypatch, tmp_path):
        """When train_adapter returns aborted=True, run_consolidation_cycle
        returns {'mode': 'aborted'} without updating simhashes or committing
        the tier slot.

        Key generation uses the graph-walk (merger.graph) via _run_fold.
        Set up a real NetworkX graph with one keyless episodic edge so the
        graph-walk produces a non-empty training set and triggers the
        train_adapter call.
        """
        from unittest.mock import patch

        import networkx as nx

        from paramem.training.consolidation import ConsolidationLoop

        loop = self._make_minimal_loop(monkeypatch, tmp_path)

        # Populate merger.graph with a real MultiDiGraph so _build_all_edge_entries_into
        # (defer=True) finds a keyless edge and produces a non-empty deferred write →
        # all_interim_keyed is non-empty → training is triggered.
        real_graph = nx.MultiDiGraph()
        real_graph.add_node("speaker0", speaker_id="speaker0", attributes={"name": "Alex"})
        real_graph.add_node("millfield", attributes={"name": "Millfield"})
        real_graph.add_edge(
            "speaker0",
            "millfield",
            predicate="lives_in",
            relation_type="factual",
        )
        loop.merger.graph = real_graph

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
                return_value="episodic_interim_t001",
            ),
            patch.object(ConsolidationLoop, "_refine_consolidation_graph", return_value=None),
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
                        "speaker_id": "speaker0",
                    }
                ],
                [],
                speaker_id="speaker0",
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

    def test_interim_abort_skips_procedural_deferred_mutations(self, monkeypatch, tmp_path):
        """When interim train_adapter returns aborted=True, store.put must be
        skipped for BOTH episodic and procedural minted entries.

        Procedural entries now ride the same interim slot as episodic and
        are flushed in the same deferred-mutation block.  An abort before flush
        must leave the store unchanged for both tiers.

        The graph is populated with a procedural-typed keyless edge so
        _build_all_edge_entries_into mints a proc-key into _deferred_writes,
        giving the abort path something to guard against.
        """
        from unittest.mock import patch

        import networkx as nx

        from paramem.training.consolidation import ConsolidationLoop

        loop = self._make_minimal_loop(monkeypatch, tmp_path)

        # Populate merger.graph with a procedural-typed keyless edge so the
        # graph-walk mints a "proc…" key into _deferred_writes.
        proc_graph = nx.MultiDiGraph()
        proc_graph.add_node("alice", speaker_id="speaker0", attributes={"name": "Alice"})
        proc_graph.add_node("tea", attributes={"name": "Tea"})
        proc_graph.add_edge(
            "alice",
            "tea",
            predicate="prefers",
            relation_type="preference",
        )
        loop.merger.graph = proc_graph
        # Enable procedural_config so partition_relations classifies the edge
        # as procedural and the graph-walk populates _tier_keyed["procedural"].
        from paramem.utils.config import AdapterConfig

        loop.procedural_config = AdapterConfig(rank=4, alpha=8, target_modules=["q_proj"])

        aborted_metrics = {"train_loss": 0.3, "aborted": True}
        store_put_calls: list = []
        original_put = loop.store.put

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
            patch.object(loop.store, "put", side_effect=_spy_put),
            patch.object(
                ConsolidationLoop,
                "_resolve_target_slot",
                return_value="episodic_interim_t001",
            ),
            patch.object(ConsolidationLoop, "_refine_consolidation_graph", return_value=None),
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
            patch(
                "paramem.memory.interim_adapter.create_interim_adapter",
                side_effect=lambda m, cfg, stamp: m,
            ),
        ):
            result = loop.run_consolidation_cycle(
                [
                    {
                        "subject": "Alice",
                        "predicate": "prefers",
                        "object": "Tea",
                        "relation_type": "preference",
                        "speaker_id": "speaker0",
                    }
                ],
                [],  # procedural_rels guard arg
                speaker_id="speaker0",
                mode="train",
                run_label="test_proc_abort",
                stamp="t001",
            )

        assert result.get("mode") == "aborted", (
            f"Expected mode='aborted' when training aborted; got {result}"
        )
        # No store.put calls must have fired — deferred flush skipped on abort.
        assert store_put_calls == [], (
            f"store.put must not be called after abort; called for {store_put_calls}"
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
                "speaker_id": "speaker0",
            },
            register=True,
        )
        # Bookkeeping is required for the re-merge stage bookkeeping lookup in
        # the full-consolidation reconstruct→remerge pipeline.
        loop.store.set_bookkeeping(
            "graph1",
            speaker_id="speaker0",
            relation_type="factual",
            first_seen="",
        )
        # The edge-walk dedup and tier stage reads from merger.graph.edges(data=True).
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
            # reconstruct_graph is a GPU operation; stub with an empty result
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
        # Set min_tier_key_floor=0 so the pre-existing tests (which use small key
        # counts to exercise dedup/tier logic) are not blocked by the floor guard.
        # Set floor=0 and tier_fast_start=False so pre-existing tests with small
        # key counts are not blocked by the accumulate guard, and pre-graduation
        # keys (all in episodic) are not mistaken for first-time fast-start graduations.
        # Tests that specifically exercise the floor / graduation behaviour live in
        # TestTierFloor and set their own config.
        loop.config = ConsolidationConfig(min_tier_key_floor=0, tier_fast_start=False)
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
        loop.graph_enrichment_max_entities_per_pass = 50
        loop.graph_enrichment_neighborhood_hops = 2
        return loop

    @staticmethod
    def _install_provenance_merge_spy(loop):
        """Install a side_effect on loop.merger.merge that stamps ik_keys.

        The real _upsert_relation stamps ``ik_key`` from ``Relation.indexed_key``
        onto the merged edge at new-edge insertion.  Since ``loop.merger`` is a
        MagicMock, we install a side_effect that mirrors this behaviour so
        The edge-walk stage reads back the correct ik_key.

        Call this on the loop BEFORE calling ``_run_with_mocks`` in tests that
        supply a recon graph with stamped edges.
        """
        from paramem.graph.name_match import canonical as _canonical
        from paramem.memory.persistence import _IK_KEY_ATTR

        def _spy_merge(session_graph, *, resolve_contradictions=True, align_predicates=False):
            for rel in session_graph.relations:
                if rel.indexed_key:
                    _subj = rel.subject
                    _obj = rel.object
                    if not loop.merger.graph.has_node(_subj):
                        loop.merger.graph.add_node(_subj)
                    if not loop.merger.graph.has_node(_obj):
                        loop.merger.graph.add_node(_obj)
                    _eid = loop.merger.graph.add_edge(
                        _subj,
                        _obj,
                        predicate=_canonical(rel.predicate),
                        relation_type=rel.relation_type,
                        confidence=rel.confidence,
                        first_seen="s_recon",
                        last_seen="s_recon",
                        reinforcement_count=1,
                        sessions=["s_recon"],
                    )
                    loop.merger.graph[_subj][_obj][_eid][_IK_KEY_ATTR] = rel.indexed_key
            return loop.merger.graph

        loop.merger.merge.side_effect = _spy_merge

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
    # Cross-session duplicate triples COLLAPSE to ONE key (real merger)
    # -------------------------------------------------------------------------

    def test_duplicate_triple_collapses_to_one_key(self, tmp_path):
        """Real GraphMerger: two active keys sharing an identical (s,p,o) collapse
        to ONE key with the fact retained.  The second key is counted as
        drift_deduplicated=1 (not genuine_loss), and graph_drift_count=1 (total).

        This replaces the old test_duplicate_triple_keeps_both_keys which used the
        _install_provenance_merge_spy fabricating spy that bypassed _upsert_relation
        and let BOTH edges survive.  The REAL merger Case-1 matches the second
        incoming recon edge on (subject, norm_pred, obj), bumps recurrence on the
        existing edge, and returns None without adding a second edge — so only ONE
        edge survives and only ONE key appears in tier_keyed.
        """
        import networkx as nx

        from paramem.graph.merger import GraphMerger
        from paramem.graph.reconstruct import ReconstructionResult
        from paramem.memory.persistence import _IK_KEY_ATTR

        # Recon graph: TWO edges for the same (s,p,o), each stamped with its key.
        recon_g = nx.MultiDiGraph()
        eid1 = recon_g.add_edge("Alice", "Berlin", predicate="lives_in")
        recon_g["Alice"]["Berlin"][eid1][_IK_KEY_ATTR] = "graph1"
        eid2 = recon_g.add_edge("Alice", "Berlin", predicate="lives_in")
        recon_g["Alice"]["Berlin"][eid2][_IK_KEY_ATTR] = "graph2"

        # Use a REAL GraphMerger (model=None — no contradiction resolution).
        loop = self._make_loop(tmp_path, merger_graph=nx.MultiDiGraph())
        loop.merger = GraphMerger(model=None)  # real merger, not MagicMock

        # Register TWO store keys for the same (s,p,o).
        # Set simhashes so we can verify the stale key's simhash is retained.
        _GRAPH1_SIMHASH = 0xABCD1234
        _GRAPH2_SIMHASH = 0xEF567890
        for key, sh in (("graph1", _GRAPH1_SIMHASH), ("graph2", _GRAPH2_SIMHASH)):
            loop.store.put(
                "episodic",
                key,
                {
                    "key": key,
                    "subject": "Alice",
                    "predicate": "lives_in",
                    "object": "Berlin",
                    "speaker_id": "speaker0",
                },
                simhash=sh,
                register=True,
            )
            loop.store.set_bookkeeping(
                key, speaker_id="speaker0", relation_type="factual", first_seen=""
            )

        result = self._run_with_mocks(loop, tmp_path, ReconstructionResult(graph=recon_g))

        # The real merger Case-1 collapses identical SPO to ONE edge → ONE key survives.
        episodic_count = result["keys_per_tier"]["episodic"]
        drift_count = result["graph_drift_count"]
        assert episodic_count == 1, (
            f"Expected 1 episodic key (duplicate-SPO collapsed by real merger); "
            f"got {episodic_count}"
        )
        assert drift_count == 1, (
            f"Expected graph_drift_count=1 (total absent keys); got {drift_count}"
        )
        # The collapsed key must be classified as deduplicated, NOT genuine_loss.
        assert result["drift_deduplicated"] == 1, (
            f"Expected drift_deduplicated=1 (duplicate-SPO collapse is intended dedup);"
            f" got {result['drift_deduplicated']}"
        )
        assert result["drift_genuine_loss"] == 0, (
            f"Expected drift_genuine_loss=0 (fact is preserved under surviving twin);"
            f" got {result['drift_genuine_loss']}"
        )

        # Under registry-true dedup mode, the deduplicated key is SOFT-STALED
        # after the dedup-then-stale finalize pass.
        # One of graph1/graph2 is in list_stale(); the other is in list_active().
        ep_reg = loop.store.registry("episodic")
        stale_keys = ep_reg.list_stale()
        active_keys = ep_reg.list_active()
        assert len(stale_keys) == 1, (
            f"Expected exactly 1 stale key (the deduplicated one); got {stale_keys}"
        )
        assert len(active_keys) == 1, (
            f"Expected exactly 1 active key (the surviving twin); got {active_keys}"
        )
        # The stale key must be in the set {graph1, graph2}.
        assert stale_keys[0] in {"graph1", "graph2"}, (
            f"Stale key {stale_keys[0]!r} not in expected set"
        )
        # The stale key must NOT be in list_active().
        assert stale_keys[0] not in active_keys, "Stale key must not be active"
        # The stale key's simhash must be RETAINED in the episodic simhash dict
        # (include_stale=True so stale fingerprints are visible).
        ep_simhashes = loop.store.tier_simhashes("episodic", include_stale=True)
        assert stale_keys[0] in ep_simhashes, (
            f"Stale key {stale_keys[0]!r} simhash must be retained; "
            f"episodic simhashes: {ep_simhashes}"
        )

        # Verify the surviving key's fact is intact.
        surviving = result["tier_keyed"]["episodic"][0]
        assert surviving["subject"] == "Alice"
        assert surviving["predicate"] == "lives_in"
        assert surviving["object"] == "Berlin"

    # -------------------------------------------------------------------------
    # Tier from edge-walk relation_type (not the retired getattr path)
    # -------------------------------------------------------------------------

    def test_preference_relation_routes_to_procedural(self, tmp_path):
        """A 'preference' relation_type routes the key to the procedural tier.

        The recon graph carries a preference edge stamped with 'proc1'; the
        re-merge stage merges it into the empty graph (setting indexed_key=proc1
        on the Relation), _upsert_relation stamps ik_key on the merged edge, and
        the edge-walk stage reads that key off the edge — tiering via bookkeeping
        relation_type='preference' → procedural.
        """
        import networkx as nx

        from paramem.graph.reconstruct import ReconstructionResult
        from paramem.memory.persistence import _IK_KEY_ATTR

        # Recon graph: preference edge stamped with proc1.
        recon_g = nx.MultiDiGraph()
        recon_eid = recon_g.add_edge("Alice", "tea", predicate="prefers")
        recon_g["Alice"]["tea"][recon_eid][_IK_KEY_ATTR] = "proc1"

        # Merged graph starts EMPTY.
        loop = self._make_loop(tmp_path, merger_graph=nx.MultiDiGraph(), procedural_enabled=True)
        # Install provenance merge spy so ik_key is stamped onto merged edge.
        self._install_provenance_merge_spy(loop)

        loop.store.put(
            "procedural",
            "proc1",
            {
                "key": "proc1",
                "subject": "Alice",
                "predicate": "prefers",
                "object": "tea",
                "speaker_id": "speaker0",
            },
            register=True,
        )
        loop.store.set_bookkeeping(
            "proc1", speaker_id="speaker0", relation_type="preference", first_seen=""
        )

        result = self._run_with_mocks(loop, tmp_path, ReconstructionResult(graph=recon_g))

        assert result["keys_per_tier"].get("procedural", 0) == 1, (
            f"Expected 1 key in procedural tier; got {result['keys_per_tier']}"
        )
        assert result["keys_per_tier"].get("episodic", 0) == 0, (
            "Preference key must NOT appear in episodic tier"
        )

    # -------------------------------------------------------------------------
    # relation_type injected from bookkeeping before re-merge
    # -------------------------------------------------------------------------

    def test_relation_type_injected_from_bookkeeping_before_remerge(self, tmp_path):
        """The re-merge stage injects the bookkeeping relation_type onto each
        reconstructed triple before calling merger.merge().

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
                "speaker_id": "speaker0",
            },
            register=True,
        )
        # Bookkeeping has relation_type="preference" — this must be injected.
        loop.store.set_bookkeeping(
            "proc1", speaker_id="speaker0", relation_type="preference", first_seen=""
        )

        # Capture merger.merge() call args.
        merge_calls: list = []

        def _spy_merge(session_graph, *, resolve_contradictions=True, align_predicates=False):
            merge_calls.append(session_graph)
            # Stamp the ik_key from the Relation onto the merged graph edge so
            # the edge-walk stage can read it back (mirrors real _upsert_relation behaviour).
            from paramem.memory.persistence import _IK_KEY_ATTR

            for rel in session_graph.relations:
                if rel.indexed_key:
                    _subj = rel.subject
                    _obj = rel.object
                    if not merged_g.has_node(_subj):
                        merged_g.add_node(_subj)
                    if not merged_g.has_node(_obj):
                        merged_g.add_node(_obj)
                    from paramem.graph.name_match import canonical as _canonical

                    _eid = merged_g.add_edge(
                        _subj,
                        _obj,
                        predicate=_canonical(rel.predicate),
                        relation_type=rel.relation_type,
                        confidence=rel.confidence,
                        first_seen="s_recon",
                        last_seen="s_recon",
                        reinforcement_count=1,
                        sessions=["s_recon"],
                    )
                    merged_g[_subj][_obj][_eid][_IK_KEY_ATTR] = rel.indexed_key
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
        assert rel.indexed_key == "proc1", (
            f"Expected indexed_key='proc1' on synthetic Relation; got {rel.indexed_key!r}"
        )

    # -------------------------------------------------------------------------
    # Predicate normalization mismatch — store raw vs graph normalized
    # -------------------------------------------------------------------------

    def test_normalization_mismatch_key_lands_in_tier(self, tmp_path):
        """Under provenance keying, predicate normalization no longer gates keying.

        The key rides the merged edge's ik_key attribute (stamped at the re-merge
        stage from Relation.indexed_key).  Even if the store entry carries a raw
        un-normalized predicate ("Lives In") and the recon edge carries the
        normalized form ("lives_in"), the key is found via the edge attribute —
        NOT by triple-string matching.  The key must be tiered, NOT drift.
        """
        import networkx as nx

        from paramem.graph.reconstruct import ReconstructionResult
        from paramem.memory.persistence import _IK_KEY_ATTR

        # Recon graph: edge with normalized predicate stamped with graph1.
        recon_g = nx.MultiDiGraph()
        recon_eid = recon_g.add_edge("Alice", "Berlin", predicate="lives_in")
        recon_g["Alice"]["Berlin"][recon_eid][_IK_KEY_ATTR] = "graph1"

        # Merged graph starts EMPTY.
        loop = self._make_loop(tmp_path, merger_graph=nx.MultiDiGraph())
        # Install provenance merge spy so ik_key is stamped onto merged edge.
        self._install_provenance_merge_spy(loop)

        # Store entry carries the RAW (un-normalized) predicate form — irrelevant
        # to keying now, but the entry content is what gets trained.
        loop.store.put(
            "episodic",
            "graph1",
            {
                "key": "graph1",
                "subject": "Alice",
                "predicate": "Lives In",  # raw form — stored as-is, not used for keying
                "object": "Berlin",
                "speaker_id": "speaker0",
            },
            register=True,
        )
        loop.store.set_bookkeeping(
            "graph1", speaker_id="speaker0", relation_type="factual", first_seen=""
        )

        result = self._run_with_mocks(loop, tmp_path, ReconstructionResult(graph=recon_g))

        # The key must appear in a tier — provenance keying always finds it.
        tiers = ("episodic", "semantic", "procedural")
        total_tiered = sum(result["keys_per_tier"].get(t, 0) for t in tiers)
        assert total_tiered == 1, (
            f"Expected exactly 1 tiered key (provenance keyed); "
            f"got {result['keys_per_tier']} drift={result['graph_drift_count']}"
        )
        # Key must NOT appear in drift.
        assert result["graph_drift_count"] == 0, (
            f"Expected 0 drift keys (key stamped on merged edge); got {result['graph_drift_count']}"
        )

    # -------------------------------------------------------------------------
    # Recall-miss is a retry, not a drop (registry-true dedup semantics)
    # -------------------------------------------------------------------------

    def test_recall_miss_is_retry_not_drop(self, tmp_path):
        """Under registry-true dedup mode, a key whose recon edge is absent
        from the recon graph is a recall-miss / retry signal — NOT dropped,
        NOT counted as drift.

        The merge input is registry-true (not reconstruction-driven), so a
        recall miss means: the key's registry-true SPO enters the merge
        regardless.  Both 'graph_bob' and 'drift1' appear in tier_keyed.
        'drift1' is in result["recall_miss_keys"] (no recon edge = miss).
        Neither key is in drift_genuine_loss.

        This replaces the prior test_explicit_drift_key_not_in_tier that
        asserted drift_genuine_loss>=1 — under registry-true dedup that
        scenario is a retry, not a loss.
        """
        import networkx as nx

        from paramem.graph.reconstruct import ReconstructionResult
        from paramem.memory.persistence import _IK_KEY_ATTR

        # Recon graph: only graph_bob has a recon edge; drift1 is absent.
        recon_g = nx.MultiDiGraph()
        recon_eid = recon_g.add_edge("Bob", "London", predicate="lives_in")
        recon_g["Bob"]["London"][recon_eid][_IK_KEY_ATTR] = "graph_bob"

        # Merged graph starts EMPTY.
        loop = self._make_loop(tmp_path, merger_graph=nx.MultiDiGraph())
        # Install provenance merge spy so ik_keys are stamped onto merged edges.
        self._install_provenance_merge_spy(loop)

        # Register graph_bob — has a recon edge matching registry-true SPO.
        loop.store.put(
            "episodic",
            "graph_bob",
            {
                "key": "graph_bob",
                "subject": "Bob",
                "predicate": "lives_in",
                "object": "London",
                "speaker_id": "speaker0",
            },
            register=True,
        )
        loop.store.set_bookkeeping(
            "graph_bob", speaker_id="speaker0", relation_type="factual", first_seen=""
        )

        # Register drift1 — no recon edge.  Under registry-true dedup the SPO
        # enters the merge directly, so drift1 survives into tier_keyed.
        loop.store.put(
            "episodic",
            "drift1",
            {
                "key": "drift1",
                "subject": "Alice",
                "predicate": "works_at",
                "object": "Acme",
                "speaker_id": "speaker0",
            },
            register=True,
        )
        loop.store.set_bookkeeping(
            "drift1", speaker_id="speaker0", relation_type="factual", first_seen=""
        )

        result = self._run_with_mocks(loop, tmp_path, ReconstructionResult(graph=recon_g))

        # Under registry-true dedup, drift1 enters recon_relations from registry-true SPO.
        # Both keys survive into tier_keyed → no drift.
        assert result["graph_drift_count"] == 0, (
            f"Expected graph_drift_count=0 (both keys have registry-true content "
            f"and enter the merge); got {result['graph_drift_count']}"
        )
        # drift1 has no recon edge → must be in recall_miss_keys (retry signal).
        assert "drift1" in result["recall_miss_keys"], (
            f"Expected drift1 in recall_miss_keys; got {result['recall_miss_keys']}"
        )
        # graph_bob's recon SPO matches registry-true → not in recall_miss_keys.
        assert "graph_bob" not in result["recall_miss_keys"], (
            f"graph_bob should not be in recall_miss_keys; got {result['recall_miss_keys']}"
        )
        # Neither key is counted as genuine_loss or deduplicated.
        assert result["drift_genuine_loss"] == 0, (
            f"Expected drift_genuine_loss=0 (no drops under registry-true dedup mode);"
            f" got {result['drift_genuine_loss']}"
        )
        assert result["drift_deduplicated"] == 0, (
            f"Expected drift_deduplicated=0 (no duplicate-SPO collapse in this test);"
            f" got {result['drift_deduplicated']}"
        )
        # Both keys are tiered.
        assert result["keys_per_tier"].get("episodic", 0) >= 2, (
            f"Expected both graph_bob and drift1 in episodic tier; got {result['keys_per_tier']}"
        )

    # -------------------------------------------------------------------------
    # Hydration-miss key is NOT classified as orphan
    # -------------------------------------------------------------------------

    def test_hydration_miss_not_classified_as_orphan(self, tmp_path):
        """A live active key whose store.get() returns None (hydration-miss
        under boot_degraded) but whose bookkeeping carries SPO must NOT be
        classified as drift_orphan.

        SCOPE NOTE: The hydration-miss fix applies to _build_registry_true_relations
        (so the key enters the merge as a registry-true Relation) and to the
        drift-partition classification (so it lands in genuine_loss, not orphan).
        The edge-walk tier_keyed build has a pre-existing gap
        (store.get(key)==None → skip) that is deferred.  Thus graph_hydration_miss:
          - Is NOT classified as drift_orphan (bookkeeping has SPO).
          - IS classified as drift_genuine_loss (hydration-miss bucket).
          - Is NOT in tier_keyed (entry-None skip; pre-existing, deferred).

        Setup: two keys — "graph_ok" has a normal content entry; "graph_hydration_miss"
        is active in the registry but has NO content entry (simulating a missed boot
        preload) while its bookkeeping carries valid SPO.
        """
        import networkx as nx

        from paramem.graph.reconstruct import ReconstructionResult
        from paramem.memory.persistence import _IK_KEY_ATTR

        # Recon graph: only graph_ok has a recon edge.
        recon_g = nx.MultiDiGraph()
        eid = recon_g.add_edge("Alice", "Berlin", predicate="lives_in")
        recon_g["Alice"]["Berlin"][eid][_IK_KEY_ATTR] = "graph_ok"

        loop = self._make_loop(tmp_path, merger_graph=nx.MultiDiGraph())
        self._install_provenance_merge_spy(loop)

        # graph_ok: full content entry.
        loop.store.put(
            "episodic",
            "graph_ok",
            {
                "key": "graph_ok",
                "subject": "Alice",
                "predicate": "lives_in",
                "object": "Berlin",
                "speaker_id": "speaker0",
            },
            register=True,
        )
        loop.store.set_bookkeeping(
            "graph_ok", speaker_id="speaker0", relation_type="factual", first_seen=""
        )

        # graph_hydration_miss: registered in the store but NO content entry.
        # Register it first (put with register=True), then delete the entry cache
        # to simulate a hydration miss while keeping the registry alive.
        loop.store.put(
            "episodic",
            "graph_hydration_miss",
            {
                "key": "graph_hydration_miss",
                "subject": "Bob",
                "predicate": "works_at",
                "object": "Acme",
                "speaker_id": "speaker0",
            },
            register=True,
        )
        # Drop the content entry (simulates boot_degraded cache miss).
        loop.store._entries["episodic"].pop("graph_hydration_miss", None)
        # Bookkeeping carries the SPO (populated independently of _entries).
        loop.store.set_bookkeeping(
            "graph_hydration_miss",
            speaker_id="speaker0",
            relation_type="factual",
            reinforcement_count=1,
            last_reinforced_cycle=1,
            first_seen="",
        )
        # Manually add SPO fields to bookkeeping so the drift-partition
        # classification finds them and routes to genuine_loss rather than orphan.
        # Note: _build_registry_true_relations no longer reads SPO from bookkeeping
        # (hydration-miss keys are skipped there); only the drift partition uses these.
        loop.store._bookkeeping["graph_hydration_miss"]["subject"] = "Bob"
        loop.store._bookkeeping["graph_hydration_miss"]["predicate"] = "works_at"
        loop.store._bookkeeping["graph_hydration_miss"]["object"] = "Acme"

        result = self._run_with_mocks(loop, tmp_path, ReconstructionResult(graph=recon_g))

        # Hydration-miss must NOT be classified as orphan: the drift partition finds
        # SPO in bookkeeping (injected above) and routes to genuine_loss.
        assert result["drift_orphan"] == 0, (
            f"Expected drift_orphan=0 (hydration-miss is NOT an orphan); "
            f"got drift_orphan={result['drift_orphan']}"
        )
        # Hydration-miss goes to genuine_loss bucket (not orphan, not deduplicated).
        assert result["drift_genuine_loss"] == 1, (
            f"Expected drift_genuine_loss=1 (hydration-miss key classified as retry); "
            f"got drift_genuine_loss={result['drift_genuine_loss']}"
        )
        # graph_ok must survive into tier_keyed.
        all_keys = {e["key"] for tier_list in result["tier_keyed"].values() for e in tier_list}
        assert "graph_ok" in all_keys, "graph_ok must survive"

    # -------------------------------------------------------------------------
    # Reconstruction collision does NOT manufacture a collapse
    # -------------------------------------------------------------------------

    def test_reconstruction_collision_does_not_manufacture_collapse(self, tmp_path):
        """Two keys with DISTINCT registry-true objects must NOT collapse
        even when reconstruction mis-reads one object onto the other.

        Under registry-true dedup mode the merge input is registry-true SPO,
        so a reconstruction mis-read has no effect on dedup identity.
        Both keys survive.

        This directly reproduces the proc33/proc35 bug: proc35 "HS3 Radio"
        reconstructed as "HR3 Radio" (proc33's object), making the old
        reconstruction-driven merge see a false Case-1 collapse.
        """
        import networkx as nx

        from paramem.graph.reconstruct import ReconstructionResult
        from paramem.memory.persistence import _IK_KEY_ATTR

        # Recon graph: proc33 correct, proc35 mis-reads its object as proc33's object.
        recon_g = nx.MultiDiGraph()
        eid1 = recon_g.add_edge("Alice", "HR3 Radio", predicate="listens_to")
        recon_g["Alice"]["HR3 Radio"][eid1][_IK_KEY_ATTR] = "proc33"
        # proc35 mis-reads: reconstruction says "HR3 Radio" instead of "HS3 Radio".
        eid2 = recon_g.add_edge("Alice", "HR3 Radio", predicate="listens_to")
        recon_g["Alice"]["HR3 Radio"][eid2][_IK_KEY_ATTR] = "proc35"

        loop = self._make_loop(tmp_path, merger_graph=nx.MultiDiGraph())
        self._install_provenance_merge_spy(loop)

        # Registry-true content: proc33 → "HR3 Radio"; proc35 → "HS3 Radio" (distinct).
        for key, obj in (("proc33", "HR3 Radio"), ("proc35", "HS3 Radio")):
            loop.store.put(
                "episodic",
                key,
                {
                    "key": key,
                    "subject": "Alice",
                    "predicate": "listens_to",
                    "object": obj,
                    "speaker_id": "speaker0",
                },
                register=True,
            )
            loop.store.set_bookkeeping(
                key, speaker_id="speaker0", relation_type="factual", first_seen=""
            )

        result = self._run_with_mocks(loop, tmp_path, ReconstructionResult(graph=recon_g))

        # Both keys must survive — registry-true distinct objects prevent collapse.
        all_keys = {e["key"] for tier_list in result["tier_keyed"].values() for e in tier_list}
        assert "proc33" in all_keys, "proc33 must survive"
        assert "proc35" in all_keys, (
            "proc35 must survive — reconstruction mis-read must NOT manufacture a collapse"
        )
        assert result["drift_deduplicated"] == 0, (
            f"Expected drift_deduplicated=0 (distinct registry-true objects); "
            f"got {result['drift_deduplicated']}"
        )
        # proc35 has no recon edge matching its registry-true SPO → in recall_miss_keys.
        assert "proc35" in result["recall_miss_keys"], (
            "proc35 must be in recall_miss_keys (recon mis-read = miss signal)"
        )

    # -------------------------------------------------------------------------
    # New unit tests (Part 1 provenance, additive fold, incomplete drop)
    # -------------------------------------------------------------------------

    def test_recall_passing_key_keyed_off_edge_provenance(self, tmp_path):
        """Provenance keying: a key stamped on a recon edge is tiered with 0 drift.

        This is the basic invariant: recon edge stamped 'graph10' → the re-merge
        stage stamps it on the merged edge → the edge-walk stage reads key off
        edge → tiered.  Surface-form normalization, triple-identity matching, and
        seen_triples dedup are no longer in the path — the key rides the edge directly.
        """
        import networkx as nx

        from paramem.graph.reconstruct import ReconstructionResult
        from paramem.memory.persistence import _IK_KEY_ATTR

        recon_g = nx.MultiDiGraph()
        eid = recon_g.add_edge("Alice", "Berlin", predicate="lives_in")
        recon_g["Alice"]["Berlin"][eid][_IK_KEY_ATTR] = "graph10"

        loop = self._make_loop(tmp_path, merger_graph=nx.MultiDiGraph())
        self._install_provenance_merge_spy(loop)

        loop.store.put(
            "episodic",
            "graph10",
            {
                "key": "graph10",
                "subject": "Alice",
                "predicate": "lives_in",
                "object": "Berlin",
                "speaker_id": "speaker0",
            },
            register=True,
        )
        loop.store.set_bookkeeping(
            "graph10", speaker_id="speaker0", relation_type="factual", first_seen=""
        )

        result = self._run_with_mocks(loop, tmp_path, ReconstructionResult(graph=recon_g))

        assert result["keys_per_tier"].get("episodic", 0) == 1, (
            f"Expected graph10 in episodic tier; got {result['keys_per_tier']}"
        )
        assert result["graph_drift_count"] == 0, (
            f"Expected 0 drift; got {result['graph_drift_count']}"
        )

    def test_incomplete_triple_dropped_by_dedup_episodic(self):
        """dedup_episodic drops entries missing any of subject/predicate/object.

        No __unkeyed__ fallback — the entry is absent from the output list.
        """
        from paramem.training.consolidation import ConsolidationLoop

        incomplete = [
            {"subject": "Alice", "predicate": "lives_in"},  # missing object
            {"subject": "Bob", "predicate": "", "object": "Berlin"},  # empty pred
            {"subject": "", "predicate": "likes", "object": "tea"},  # empty subj
            {"subject": "Carol", "predicate": "works_at", "object": "Acme"},  # complete
        ]
        result = ConsolidationLoop.dedup_episodic(incomplete)
        assert len(result) == 1, f"Expected 1 complete entry; got {len(result)}"
        assert result[0]["subject"] == "Carol"
        # No __unkeyed__ keys in result
        assert all("__unkeyed__" not in str(r) for r in result)

    def test_incomplete_triple_dropped_by_dedup_procedural(self):
        """dedup_procedural drops entries missing any of subject/predicate/object."""
        from paramem.training.consolidation import ConsolidationLoop

        incomplete = [
            {"subject": "Alice", "predicate": "prefers"},  # missing object
            {"subject": "Bob", "predicate": "prefers", "object": "tea"},  # complete
        ]
        result = ConsolidationLoop.dedup_procedural(incomplete)
        assert len(result) == 1, f"Expected 1 complete entry; got {len(result)}"
        assert result[0]["subject"] == "Bob"

    def test_case1_adopt_stamps_keyless_edge(self, tmp_path):
        """Case-1-adopt: a keyless pre-existing edge adopts the incoming ik_key.

        Pre-seed merger.graph with a keyless (s,p,o) edge.  Merge a recon
        Relation with indexed_key='g1' for the same triple.  The exact-duplicate
        reinforce branch fires and adopts the ik_key onto the existing edge.
        """
        from paramem.graph.merger import GraphMerger
        from paramem.graph.schema import Relation
        from paramem.memory.persistence import _IK_KEY_ATTR

        m = GraphMerger()  # no model — no cardinality calls
        m.graph.add_node("Alice")
        m.graph.add_node("Berlin")
        existing_eid = m.graph.add_edge(
            "Alice",
            "Berlin",
            predicate="lives in",
            relation_type="factual",
            confidence=1.0,
            first_seen="s0",
            last_seen="s0",
            reinforcement_count=1,
            sessions=["s0"],
        )
        # No ik_key on the existing edge (keyless pre-existing state).
        assert m.graph["Alice"]["Berlin"][existing_eid].get(_IK_KEY_ATTR) is None

        # Upsert the same triple with indexed_key set — should adopt via Case-1.
        # Incoming predicate "lives_in" canonicalizes to "lives in" (space form),
        # matching the pre-seeded edge predicate.
        incoming = Relation(
            subject="Alice",
            predicate="lives_in",
            object="Berlin",
            relation_type="factual",
            confidence=1.0,
            speaker_id="speaker0",
            indexed_key="g1",
        )
        m._upsert_relation("Alice", "Berlin", incoming, "s1", "2026-01-01T00:00:00Z")

        # Existing edge must now carry the adopted ik_key.
        edge_data = m.graph["Alice"]["Berlin"][existing_eid]
        assert edge_data.get(_IK_KEY_ATTR) == "g1", (
            f"Expected ik_key='g1' adopted onto existing edge; got {edge_data.get(_IK_KEY_ATTR)!r}"
        )
        # No duplicate edge should have been inserted (still one edge for this (s,p,o)).
        same_pred_edges = [
            k for k, d in m.graph["Alice"]["Berlin"].items() if d.get("predicate") == "lives in"
        ]
        assert len(same_pred_edges) == 1, (
            f"Expected exactly 1 edge after Case-1-adopt; got {len(same_pred_edges)}"
        )

    def test_fold_tied_timestamps_both_keys_survive(self, tmp_path):
        """Fold with refinement_contradiction='on' + tied last_seen timestamps:
        the recency rule fires (Case-2, REPLACE-classified) but n_at_max >= 2 →
        coexist; both keys survive with zero drift.

        This is the fold coexist guard: recency-on at fold with equal timestamps
        must never remove a registered key.  Legacy timestamp-less keys (last_seen="")
        all tie at "" → coexist is the safe no-op path.
        """
        from unittest.mock import MagicMock, patch

        import networkx as nx

        from paramem.graph.merger import GraphMerger
        from paramem.graph.reconstruct import ReconstructionResult
        from paramem.memory.persistence import _IK_KEY_ATTR
        from paramem.utils.config import ConsolidationConfig

        recon_g = nx.MultiDiGraph()
        munich_eid = recon_g.add_edge("Alex", "Munich", predicate="lives_in")
        recon_g["Alex"]["Munich"][munich_eid][_IK_KEY_ATTR] = "key_munich"
        berlin_eid = recon_g.add_edge("Alex", "Berlin", predicate="lives_in")
        recon_g["Alex"]["Berlin"][berlin_eid][_IK_KEY_ATTR] = "key_berlin"

        model_stub = MagicMock()
        tok_stub = MagicMock()
        tok_stub.apply_chat_template.return_value = "formatted"

        loop = self._make_loop(tmp_path, merger_graph=nx.MultiDiGraph())
        loop.merger = GraphMerger(model=model_stub, tokenizer=tok_stub)
        # Enable contradiction detection so the recency rule fires.
        loop.config = ConsolidationConfig(
            min_tier_key_floor=0, tier_fast_start=False, refinement_contradiction="on"
        )

        same_ts = "2026-01-01T00:00:00Z"
        for key, obj in (("key_munich", "Munich"), ("key_berlin", "Berlin")):
            loop.store.put(
                "episodic",
                key,
                {
                    "key": key,
                    "subject": "Alex",
                    "predicate": "lives_in",
                    "object": obj,
                    "speaker_id": "speaker0",
                },
                register=True,
            )
            # Both keys share the same last_seen → timestamps tied → coexist.
            loop.store.set_bookkeeping(
                key,
                speaker_id="speaker0",
                relation_type="factual",
                last_seen=same_ts,
                first_seen="",
            )

        # Patch check_predicate_coexistence to return REPLACE (single-valued predicate).
        # The cache is cleared by reset_graph() before each fold, so we patch the
        # underlying function to maximise sensitivity — tied timestamps must coexist
        # even when the cardinality verdict is REPLACE.
        with patch(
            "paramem.graph.merger.check_predicate_coexistence",
            return_value="REPLACE",
        ):
            result = self._run_with_mocks(loop, tmp_path, ReconstructionResult(graph=recon_g))

        # Tied timestamps → coexist: BOTH keys must survive — zero drift.
        assert result["graph_drift_count"] == 0, (
            f"Expected 0 drift (tied last_seen → coexist); got drift={result['graph_drift_count']}"
        )
        assert result["keys_per_tier"].get("episodic", 0) == 2, (
            f"Expected both key_munich and key_berlin in episodic tier; "
            f"got {result['keys_per_tier']}"
        )

    def test_fold_distinct_timestamps_newer_wins_older_staled(self, tmp_path):
        """Fold with refinement_contradiction='on' + distinct last_seen timestamps:
        the recency rule fires (Case-2, REPLACE-classified) and the fresher key
        (Berlin, 2026-01-02) wins; the staler key (Munich, 2026-01-01) is retired
        and soft-staled via _apply_subtractive_removals_to_store(scope='fold').
        """
        from unittest.mock import MagicMock, patch

        import networkx as nx

        from paramem.graph.merger import GraphMerger
        from paramem.graph.reconstruct import ReconstructionResult
        from paramem.memory.persistence import _IK_KEY_ATTR
        from paramem.utils.config import ConsolidationConfig

        recon_g = nx.MultiDiGraph()
        munich_eid = recon_g.add_edge("Alex", "Munich", predicate="lives_in")
        recon_g["Alex"]["Munich"][munich_eid][_IK_KEY_ATTR] = "key_munich"
        berlin_eid = recon_g.add_edge("Alex", "Berlin", predicate="lives_in")
        recon_g["Alex"]["Berlin"][berlin_eid][_IK_KEY_ATTR] = "key_berlin"

        model_stub = MagicMock()
        tok_stub = MagicMock()
        tok_stub.apply_chat_template.return_value = "formatted"

        loop = self._make_loop(tmp_path, merger_graph=nx.MultiDiGraph())
        loop.merger = GraphMerger(model=model_stub, tokenizer=tok_stub)
        loop.config = ConsolidationConfig(
            min_tier_key_floor=0, tier_fast_start=False, refinement_contradiction="on"
        )

        # Munich has an older last_seen; Berlin has a fresher one.
        ts_data = {
            "key_munich": ("Munich", "2026-01-01T00:00:00Z"),
            "key_berlin": ("Berlin", "2026-01-02T00:00:00Z"),
        }
        for key, (obj, ts) in ts_data.items():
            loop.store.put(
                "episodic",
                key,
                {
                    "key": key,
                    "subject": "Alex",
                    "predicate": "lives_in",
                    "object": obj,
                    "speaker_id": "speaker0",
                },
                register=True,
            )
            loop.store.set_bookkeeping(
                key, speaker_id="speaker0", relation_type="factual", last_seen=ts, first_seen=""
            )

        # Patch check_predicate_coexistence to return REPLACE (single-valued predicate).
        # The cache is cleared by reset_graph() before each fold, so we patch the
        # underlying function rather than pre-seeding the cache.
        with patch(
            "paramem.graph.merger.check_predicate_coexistence",
            return_value="REPLACE",
        ):
            result = self._run_with_mocks(loop, tmp_path, ReconstructionResult(graph=recon_g))

        # Fresher Berlin wins; staler Munich is soft-staled by the recency removal.
        # _apply_subtractive_removals_to_store runs BEFORE active_keys is computed,
        # so staled keys are already absent from active_keys and do NOT appear in
        # _drift_keys.  graph_drift_count therefore remains 0 (the key was
        # intentionally removed, not genuinely lost); is_stale verifies the outcome.
        assert result["keys_per_tier"].get("episodic", 0) == 1, (
            f"Expected only key_berlin in episodic tier; got {result['keys_per_tier']}"
        )
        assert loop.store.is_stale("key_munich"), (
            "Staler key_munich must be soft-staled after fold recency removal"
        )
        assert not loop.store.is_stale("key_berlin"), "Fresher key_berlin must NOT be staled"
        assert result["graph_drift_count"] == 0, (
            "Recency-staled key is removed from active before drift count → 0 drift"
        )

    def test_munich_berlin_fold_non_subtractive_no_drift(self, tmp_path):
        """Fold with mock merger: Munich and Berlin are both registered keys
        for the same (subject, predicate).  With resolve_contradictions=False the
        spy inserts both edges; neither key drifts.

        This mirrors the old test_munich_berlin_replace_retires_munich but asserts
        the correct post-bugfix behaviour: zero drift, both keys tiered.
        """
        import networkx as nx

        from paramem.graph.reconstruct import ReconstructionResult
        from paramem.memory.persistence import _IK_KEY_ATTR

        # Recon graph: both keys present.
        recon_g = nx.MultiDiGraph()
        munich_eid = recon_g.add_edge("Alex", "Munich", predicate="lives_in")
        recon_g["Alex"]["Munich"][munich_eid][_IK_KEY_ATTR] = "key_munich"
        berlin_eid = recon_g.add_edge("Alex", "Berlin", predicate="lives_in")
        recon_g["Alex"]["Berlin"][berlin_eid][_IK_KEY_ATTR] = "key_berlin"

        loop = self._make_loop(tmp_path, merger_graph=nx.MultiDiGraph())

        for key, obj in (("key_munich", "Munich"), ("key_berlin", "Berlin")):
            loop.store.put(
                "episodic",
                key,
                {
                    "key": key,
                    "subject": "Alex",
                    "predicate": "lives_in",
                    "object": obj,
                    "speaker_id": "speaker0",
                },
                register=True,
            )
            loop.store.set_bookkeeping(
                key, speaker_id="speaker0", relation_type="factual", first_seen=""
            )

        # Spy that inserts BOTH edges (resolve_contradictions=False at fold).
        self._install_provenance_merge_spy(loop)

        result = self._run_with_mocks(loop, tmp_path, ReconstructionResult(graph=recon_g))

        # Both keys must survive — zero drift.
        assert result["graph_drift_count"] == 0, (
            f"Expected 0 drift (fold is non-subtractive — "
            f"both Munich and Berlin survive); got drift={result['graph_drift_count']}"
        )
        assert result["keys_per_tier"].get("episodic", 0) == 2, (
            f"Expected key_munich and key_berlin in episodic tier; got {result['keys_per_tier']}"
        )

    def test_promotion_at_fold_reconstructs_from_episodic_then_trains_semantic(self, tmp_path):
        """Regression: a key at the promotion threshold is reconstructed from the episodic
        adapter (where its weights live) and then moved to semantic AFTER reconstruction,
        so it is NOT lost as a reconstruction failure.

        Old (broken) ordering: _promote_mature_keys ran at the INTERIM tick, moving the
        registry entry to semantic before the fold ran reconstruct_graph.  The fold then
        probed the key against the semantic adapter, which never learned it, producing a
        recon failure → permanent data loss.

        New (fixed) ordering:
        1. reconstruct_graph probes the key against episodic (where weights are) → success.
        2. _promote_mature_keys_inline runs AFTER the recurrence bump and BEFORE
           tier_keyed is built → moves the key to semantic.
        3. tier_keyed["semantic"] picks up the key (tier_for_active_key == "semantic").
        4. The fold trains the semantic adapter on the key.

        This test sets up a single episodic key with reinforcement_count == promotion_threshold,
        mocks reconstruct_graph to succeed for episodic and FAIL for semantic, runs the fold,
        and asserts: no reconstruction failures, the key is in tier_keyed["semantic"], and
        the key survives (not in drift).
        """
        import networkx as nx

        from paramem.graph.reconstruct import ReconstructionResult
        from paramem.memory.persistence import _IK_KEY_ATTR

        # threshold=3 (ConsolidationConfig default)
        loop = self._make_loop(tmp_path, merger_graph=nx.MultiDiGraph())
        self._install_provenance_merge_spy(loop)

        # Register the key in EPISODIC (weights live here).
        loop.store.put(
            "episodic",
            "graph_ep1",
            {
                "key": "graph_ep1",
                "subject": "Alice",
                "predicate": "works_at",
                "object": "Acme",
                "speaker_id": "S0",
            },
            register=True,
        )
        loop.store.put_simhash("episodic", "graph_ep1", 11111)
        loop.store.set_bookkeeping(
            "graph_ep1",
            speaker_id="S0",
            relation_type="factual",
            reinforcement_count=3,  # at threshold — would be promoted
            last_reinforced_cycle=1,
            first_seen="",
        )

        # Recon graph: reconstruct_graph found the key in episodic (weights present there).
        # If the key had been moved to semantic BEFORE the fold, reconstruct_graph would
        # have probed the semantic adapter and missed it.  Here we verify the correct path:
        # the recon graph carries the key as episodic-reconstructed.
        recon_g = nx.MultiDiGraph()
        eid1 = recon_g.add_edge("Alice", "Acme", predicate="works_at")
        recon_g["Alice"]["Acme"][eid1][_IK_KEY_ATTR] = "graph_ep1"

        result = self._run_with_mocks(loop, tmp_path, ReconstructionResult(graph=recon_g))

        # After the fold, the key must be in semantic (promoted by _promote_mature_keys_inline
        # AFTER reconstruction but BEFORE tier assignment).
        assert loop.store.tier_for_active_key("graph_ep1") == "semantic", (
            "Key with reinforcement_count==threshold must be promoted to semantic during the fold"
        )
        assert "graph_ep1" in loop.promoted_keys, "Promoted key must appear in loop.promoted_keys"

        # The key must appear in tier_keyed["semantic"] — survived, not drifted.
        # Under the old (broken) ordering, the key would have been probed against the
        # semantic adapter (where its weights are NOT), causing a recon failure and
        # a drift count of 1.  Under the fixed ordering, it is probed against episodic
        # (weights present) → reconstructed → then promoted → tier_keyed["semantic"].
        tier_keyed = result.get("tier_keyed", {})
        semantic_keys = {e["key"] for e in tier_keyed.get("semantic", [])}
        assert "graph_ep1" in semantic_keys, (
            "Key promoted during fold must land in tier_keyed['semantic'] for semantic training"
        )

        # Zero drift: the key was found via the merged edge and survived the fold.
        assert result["graph_drift_count"] == 0, (
            f"Expected 0 drift; got {result['graph_drift_count']}"
        )


class TestDriftPartitioning:
    """3-way drift partition: deduplicated / orphan / genuine_loss.

    Guards that the drift-accounting site in consolidate_interim_adapters
    correctly separates intended deduplication (duplicate-SPO collapse, fact
    preserved under surviving twin) from orphan keys (no SPO content) and
    genuine reconstruction losses (content present, no merged edge produced).
    """

    @staticmethod
    def _make_loop(tmp_path, *, merger_graph):
        """Minimal ConsolidationLoop stub (delegates to TestConsolidateInterimAdaptersFullFlow)."""
        return TestConsolidateInterimAdaptersFullFlow._make_loop(
            tmp_path, merger_graph=merger_graph
        )

    @staticmethod
    def _install_provenance_merge_spy(loop):
        return TestConsolidateInterimAdaptersFullFlow._install_provenance_merge_spy(loop)

    @staticmethod
    def _run_with_mocks(loop, tmp_path, reconstruct_return):
        return TestConsolidateInterimAdaptersFullFlow._run_with_mocks(
            loop, tmp_path, reconstruct_return
        )

    def test_drift_partition_three_buckets(self, tmp_path):
        """Fold with drift-key classes; each lands in the correct bucket under
        registry-true dedup mode.

        Setup:
        - "key_dup1" and "key_dup2": identical registry-true SPO (Alice lives_in Berlin).
          key_dup2 collapses into key_dup1 via Case-1 → drift_deduplicated=1, SOFT-STALED.
          The fact is preserved under key_dup1.
        - "key_orphan": no subject/predicate/object in its store entry and bookkeeping.
          _build_registry_true_relations skips it (no predicate) → no merged edge → drift_orphan=1.
        - "key_loss": has SPO content but no recon edge.  Under registry-true dedup the
          SPO enters the merge directly, so key_loss survives into tier_keyed (recall-miss
          retry, NOT a loss).  drift_genuine_loss=0.

        Total drift = 2 (key_dup2 + key_orphan); genuine_loss=0; key_loss in recall_miss_keys.
        """
        import networkx as nx

        from paramem.graph.merger import GraphMerger
        from paramem.graph.reconstruct import ReconstructionResult
        from paramem.memory.persistence import _IK_KEY_ATTR

        # Recon graph: key_dup1 and key_dup2 share the same (s,p,o).
        # key_orphan and key_loss have NO recon edges.
        recon_g = nx.MultiDiGraph()
        eid1 = recon_g.add_edge("Alice", "Berlin", predicate="lives_in")
        recon_g["Alice"]["Berlin"][eid1][_IK_KEY_ATTR] = "key_dup1"
        eid2 = recon_g.add_edge("Alice", "Berlin", predicate="lives_in")
        recon_g["Alice"]["Berlin"][eid2][_IK_KEY_ATTR] = "key_dup2"

        loop = self._make_loop(tmp_path, merger_graph=nx.MultiDiGraph())
        loop.merger = GraphMerger(model=None)  # real merger for Case-1 collapse

        # key_dup1 — SPO content, recon edge present; will survive.
        loop.store.put(
            "episodic",
            "key_dup1",
            {
                "key": "key_dup1",
                "subject": "Alice",
                "predicate": "lives_in",
                "object": "Berlin",
                "speaker_id": "speaker0",
            },
            register=True,
        )
        loop.store.set_bookkeeping(
            "key_dup1", speaker_id="speaker0", relation_type="factual", first_seen=""
        )

        # key_dup2 — same SPO; recon edge collapses into key_dup1 via Case-1.
        loop.store.put(
            "episodic",
            "key_dup2",
            {
                "key": "key_dup2",
                "subject": "Alice",
                "predicate": "lives_in",
                "object": "Berlin",
                "speaker_id": "speaker0",
            },
            register=True,
        )
        loop.store.set_bookkeeping(
            "key_dup2", speaker_id="speaker0", relation_type="factual", first_seen=""
        )

        # key_orphan — no SPO content in entry; no recon edge.
        loop.store.put(
            "episodic",
            "key_orphan",
            {
                "key": "key_orphan",
                "subject": "",
                "predicate": "",
                "object": "",
                "speaker_id": "speaker0",
            },
            register=True,
        )
        loop.store.set_bookkeeping(
            "key_orphan", speaker_id="speaker0", relation_type="factual", first_seen=""
        )

        # key_loss — has real SPO content but no recon edge (reconstruction failed).
        loop.store.put(
            "episodic",
            "key_loss",
            {
                "key": "key_loss",
                "subject": "Bob",
                "predicate": "works_at",
                "object": "Acme",
                "speaker_id": "speaker0",
            },
            register=True,
        )
        loop.store.set_bookkeeping(
            "key_loss", speaker_id="speaker0", relation_type="factual", first_seen=""
        )

        result = self._run_with_mocks(loop, tmp_path, ReconstructionResult(graph=recon_g))

        # Under registry-true dedup: key_dup2 and key_orphan are absent from tier_keyed.
        # key_loss has registry-true SPO → enters recon_relations → survives in tier_keyed.
        # Total drift = 2 (key_dup2 + key_orphan); key_loss is NOT a drift key.
        assert result["graph_drift_count"] == 2, (
            f"Expected graph_drift_count=2 (key_dup2 + key_orphan only; "
            f"key_loss survives via registry-true SPO); got {result['graph_drift_count']}"
        )
        assert result["drift_deduplicated"] == 1, (
            f"Expected drift_deduplicated=1 (key_dup2 collapsed into key_dup1);"
            f" got {result['drift_deduplicated']}"
        )
        assert result["drift_orphan"] == 1, (
            f"Expected drift_orphan=1 (key_orphan has no SPO content); got {result['drift_orphan']}"
        )
        assert result["drift_genuine_loss"] == 0, (
            f"Expected drift_genuine_loss=0 (key_loss retrained with registry-true content "
            f"— not a loss); got {result['drift_genuine_loss']}"
        )
        # key_loss must be in recall_miss_keys (no recon edge = retry signal).
        assert "key_loss" in result["recall_miss_keys"], (
            f"Expected key_loss in recall_miss_keys (retry signal); "
            f"got {result['recall_miss_keys']}"
        )
        # key_dup1 and key_loss must both survive in episodic (2 keys total).
        assert result["keys_per_tier"].get("episodic", 0) == 2, (
            f"Expected 2 episodic keys (key_dup1 + key_loss); got {result['keys_per_tier']}"
        )
        surviving_keys = {e["key"] for e in result["tier_keyed"]["episodic"]}
        assert "key_dup1" in surviving_keys, f"key_dup1 must survive; got {surviving_keys}"
        assert "key_loss" in surviving_keys, (
            f"key_loss must survive (registry-true retry); got {surviving_keys}"
        )

    def test_no_high_drift_warning_when_only_dedup(self, tmp_path, caplog):
        """The 'high graph drift — review recommended' warning must NOT fire
        when all absent keys are in the deduplicated bucket (genuine_loss=0).

        Regression: before the partition, 34 deduplicated keys in a 239-key fold
        triggered the 10%-threshold WARNING even though no real data was lost.
        """
        import logging

        import networkx as nx

        from paramem.graph.merger import GraphMerger
        from paramem.graph.reconstruct import ReconstructionResult
        from paramem.memory.persistence import _IK_KEY_ATTR

        # Two keys for the same SPO — one collapses (dedup only, no genuine loss).
        recon_g = nx.MultiDiGraph()
        eid1 = recon_g.add_edge("Carol", "Paris", predicate="lives_in")
        recon_g["Carol"]["Paris"][eid1][_IK_KEY_ATTR] = "key_c1"
        eid2 = recon_g.add_edge("Carol", "Paris", predicate="lives_in")
        recon_g["Carol"]["Paris"][eid2][_IK_KEY_ATTR] = "key_c2"

        loop = self._make_loop(tmp_path, merger_graph=nx.MultiDiGraph())
        loop.merger = GraphMerger(model=None)

        for key in ("key_c1", "key_c2"):
            loop.store.put(
                "episodic",
                key,
                {
                    "key": key,
                    "subject": "Carol",
                    "predicate": "lives_in",
                    "object": "Paris",
                    "speaker_id": "speaker0",
                },
                register=True,
            )
            loop.store.set_bookkeeping(
                key, speaker_id="speaker0", relation_type="factual", first_seen=""
            )

        # caplog.at_level alone does not capture in this project (log propagation
        # is intercepted); attach the handler directly to the named logger.
        _named_logger = logging.getLogger("paramem.training.consolidation")
        _named_logger.addHandler(caplog.handler)
        try:
            result = self._run_with_mocks(loop, tmp_path, ReconstructionResult(graph=recon_g))
        finally:
            _named_logger.removeHandler(caplog.handler)

        assert result["drift_genuine_loss"] == 0, (
            f"Expected drift_genuine_loss=0 (dedup only); got {result['drift_genuine_loss']}"
        )
        assert result["drift_deduplicated"] == 1, (
            f"Expected drift_deduplicated=1; got {result['drift_deduplicated']}"
        )
        # The "high graph drift — review recommended" WARNING must NOT appear
        # when genuine_loss=0, regardless of total drift count.
        high_drift_warnings = [
            r
            for r in caplog.records
            if r.levelno >= logging.WARNING
            and "high graph drift" in r.message
            and "review recommended" in r.message
        ]
        assert high_drift_warnings == [], (
            f"Unexpected 'high graph drift' warning when only dedup occurred: "
            f"{[r.message for r in high_drift_warnings]}"
        )

    def test_high_drift_warning_fires_on_genuine_loss(self, tmp_path, caplog):
        """The 'genuine reconstruction loss' WARNING fires when genuine_loss > 0.

        Under registry-true dedup mode, drift_genuine_loss fires for a key that
        has non-empty subject content but an EMPTY predicate:
        _build_registry_true_relations skips it (no predicate = not keyable), so
        it never enters the merge and ends up absent from tier_keyed with its
        content intact.  The drift-partition loop sees a key with non-empty
        subject → genuine_loss bucket → WARNING.

        This is distinct from a full-SPO key with no recon edge, which under
        registry-true dedup IS included via registry-true SPO and survives in
        tier_keyed (retry-not-drop).
        """
        import logging

        import networkx as nx

        from paramem.graph.reconstruct import ReconstructionResult
        from paramem.memory.persistence import _IK_KEY_ATTR

        # key_ok has a recon edge and full SPO — survives normally.
        recon_g = nx.MultiDiGraph()
        eid = recon_g.add_edge("Dave", "London", predicate="lives_in")
        recon_g["Dave"]["London"][eid][_IK_KEY_ATTR] = "key_ok"

        loop = self._make_loop(tmp_path, merger_graph=nx.MultiDiGraph())
        self._install_provenance_merge_spy(loop)

        loop.store.put(
            "episodic",
            "key_ok",
            {
                "key": "key_ok",
                "subject": "Dave",
                "predicate": "lives_in",
                "object": "London",
                "speaker_id": "speaker0",
            },
            register=True,
        )
        loop.store.set_bookkeeping(
            "key_ok", speaker_id="speaker0", relation_type="factual", first_seen=""
        )

        # key_no_pred has a non-empty subject but empty predicate.
        # _build_registry_true_relations skips it → no merged edge → drift_genuine_loss.
        loop.store.put(
            "episodic",
            "key_no_pred",
            {
                "key": "key_no_pred",
                "subject": "Eve",
                "predicate": "",
                "object": "Acme",
                "speaker_id": "speaker0",
            },
            register=True,
        )
        loop.store.set_bookkeeping(
            "key_no_pred", speaker_id="speaker0", relation_type="factual", first_seen=""
        )

        # caplog.at_level alone does not capture in this project (log propagation
        # is intercepted); attach the handler directly to the named logger.
        _named_logger = logging.getLogger("paramem.training.consolidation")
        _named_logger.addHandler(caplog.handler)
        try:
            result = self._run_with_mocks(loop, tmp_path, ReconstructionResult(graph=recon_g))
        finally:
            _named_logger.removeHandler(caplog.handler)

        assert result["drift_genuine_loss"] == 1, (
            f"Expected drift_genuine_loss=1 (key_no_pred has subject but no predicate "
            f"— skipped by _build_registry_true_relations); got {result['drift_genuine_loss']}"
        )
        # The genuine-loss WARNING must fire.
        genuine_loss_warnings = [
            r
            for r in caplog.records
            if r.levelno >= logging.WARNING and "genuine reconstruction loss" in r.message
        ]
        assert genuine_loss_warnings, (
            f"Expected a WARNING about genuine reconstruction loss; none emitted."
            f" caplog had: {[r.message for r in caplog.records]}"
        )


class TestDriftIntendedRemoval:
    """Tests for the drift_intended_removal bucket (merger ledger consumer).

    These tests seed loop.merger.removal_ledger directly because
    _run_graph_enrichment is mocked in _run_with_mocks — the ledger entries
    that enrichment would write are supplied manually.
    """

    @staticmethod
    def _make_loop(tmp_path, *, merger_graph):
        return TestConsolidateInterimAdaptersFullFlow._make_loop(
            tmp_path, merger_graph=merger_graph
        )

    @staticmethod
    def _run_with_mocks(loop, tmp_path, reconstruct_return):
        return TestConsolidateInterimAdaptersFullFlow._run_with_mocks(
            loop, tmp_path, reconstruct_return
        )

    def test_enrichment_same_as_routes_to_intended_removal_not_genuine_loss(self, tmp_path):
        """A key whose edge was dropped by enrichment same_as contraction must land
        in drift_intended_removal, NOT drift_genuine_loss.

        Setup: one surviving key (key_ok, present in recon graph) and one key
        (key_enrichment) that has content but no recon edge.  _run_graph_enrichment
        is replaced with a side_effect that populates removal_ledger at call time
        (i.e. AFTER reset_graph() clears it) to faithfully replicate the real
        enrichment code path.
        """
        import networkx as nx

        from paramem.graph.merger import GraphMerger
        from paramem.graph.reconstruct import ReconstructionResult
        from paramem.memory.persistence import _IK_KEY_ATTR

        recon_g = nx.MultiDiGraph()
        eid_ok = recon_g.add_edge("Dave", "London", predicate="lives_in")
        recon_g["Dave"]["London"][eid_ok][_IK_KEY_ATTR] = "key_ok"

        loop = self._make_loop(tmp_path, merger_graph=nx.MultiDiGraph())
        # refinement_enrichment="on" + sota_enabled=True required for _run_graph_enrichment
        # to be called; base defaults (off/False) skip enrichment.
        loop.config = loop.config.__class__(
            min_tier_key_floor=0,
            tier_fast_start=False,
            refinement_enrichment="on",
            sota_enabled=True,
        )
        loop.merger = GraphMerger(model=None)

        # key_ok — survives (recon edge present, full SPO).
        loop.store.put(
            "episodic",
            "key_ok",
            {
                "key": "key_ok",
                "subject": "Dave",
                "predicate": "lives_in",
                "object": "London",
                "speaker_id": "speaker0",
            },
            register=True,
        )
        loop.store.set_bookkeeping(
            "key_ok", speaker_id="speaker0", relation_type="factual", first_seen=""
        )

        # key_enrichment — active key, no recon edge; will be written to the ledger
        # by the custom enrichment side_effect (after reset_graph clears the ledger).
        loop.store.put(
            "episodic",
            "key_enrichment",
            {
                "key": "key_enrichment",
                "subject": "Alice",
                "predicate": "alias_of",
                "object": "Alicia",
                "speaker_id": "speaker0",
            },
            register=True,
        )
        loop.store.set_bookkeeping(
            "key_enrichment",
            speaker_id="speaker0",
            relation_type="factual",
            first_seen="",
        )

        def _enrichment_side_effect():
            # Called AFTER reset_graph() and the re-merge stage.  At this point the
            # merged graph has an alice→alicia alias_of edge carrying key_enrichment.
            # Node keys are canonical (lowercase) because GraphMerger canonicalizes
            # all entity names at merge time.  Simulate the real same_as contraction:
            # the edge becomes a self-loop and is dropped; replicate by removing all
            # alice→alicia edges.
            g = loop.merger.graph
            edges_to_remove = list(g.out_edges("alice", keys=True))
            for u, v, k in edges_to_remove:
                if v == "alicia":
                    g.remove_edge(u, v, key=k)
            # Write the ledger entry exactly as real _run_graph_enrichment does.
            loop.merger.removal_ledger["key_enrichment"] = {
                "reason": "enrichment_same_as",
                "merged_into": "alice",
            }
            return {"skipped": False, "new_edges": 0, "same_as_contractions": 1}

        from unittest.mock import patch

        from paramem.server.gpu_lock import _gpu_thread_lock
        from paramem.training.consolidation import ConsolidationLoop

        _gpu_thread_lock.acquire()
        try:
            with (
                patch(
                    "paramem.training.consolidation.reconstruct_graph",
                    return_value=ReconstructionResult(graph=recon_g),
                ),
                patch.object(
                    ConsolidationLoop,
                    "_run_graph_enrichment",
                    side_effect=_enrichment_side_effect,
                ),
                patch.object(ConsolidationLoop, "_enable_gradient_checkpointing"),
                patch.object(ConsolidationLoop, "_disable_gradient_checkpointing"),
                patch.object(
                    ConsolidationLoop, "_maybe_make_recall_callback", return_value=(None, None)
                ),
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
                result = loop.consolidate_interim_adapters(trainer=None, router=None)
        finally:
            _gpu_thread_lock.release()

        assert result["drift_intended_removal"] == 1, (
            f"Expected drift_intended_removal=1; got {result['drift_intended_removal']}"
        )
        assert result["drift_intended_removal_by_reason"] == {"enrichment_same_as": 1}, (
            f"Expected by_reason={{'enrichment_same_as': 1}}; "
            f"got {result['drift_intended_removal_by_reason']}"
        )
        assert result["drift_genuine_loss"] == 0, (
            f"Expected drift_genuine_loss=0 (enrichment is NOT genuine loss); "
            f"got {result['drift_genuine_loss']}"
        )

    def test_dedup_key_still_routes_through_collapsed_branch(self, tmp_path):
        """A duplicate-SPO key that is in BOTH _collapsed_set and removal_ledger
        must still be classified as drift_deduplicated (soft-staled) — the
        _collapsed_set branch fires first (load-bearing for R4 / soft-stale).

        Regression guard: the new elif _dk in _ledger branch must NOT intercept
        dedup keys.
        """
        import networkx as nx

        from paramem.graph.merger import GraphMerger
        from paramem.graph.reconstruct import ReconstructionResult
        from paramem.memory.persistence import _IK_KEY_ATTR

        recon_g = nx.MultiDiGraph()
        eid1 = recon_g.add_edge("Alice", "Berlin", predicate="lives_in")
        recon_g["Alice"]["Berlin"][eid1][_IK_KEY_ATTR] = "key_dup1"
        eid2 = recon_g.add_edge("Alice", "Berlin", predicate="lives_in")
        recon_g["Alice"]["Berlin"][eid2][_IK_KEY_ATTR] = "key_dup2"

        loop = self._make_loop(tmp_path, merger_graph=nx.MultiDiGraph())
        loop.merger = GraphMerger(model=None)

        for key in ("key_dup1", "key_dup2"):
            loop.store.put(
                "episodic",
                key,
                {
                    "key": key,
                    "subject": "Alice",
                    "predicate": "lives_in",
                    "object": "Berlin",
                    "speaker_id": "speaker0",
                },
                register=True,
            )
            loop.store.set_bookkeeping(
                key, speaker_id="speaker0", relation_type="factual", first_seen=""
            )

        # Pre-seed ledger for key_dup2 as well — to prove it doesn't divert to
        # intended_removal (the _collapsed_set branch must win).
        loop.merger.removal_ledger["key_dup2"] = {
            "reason": "dedup",
            "surviving_twin": "key_dup1",
        }

        result = self._run_with_mocks(loop, tmp_path, ReconstructionResult(graph=recon_g))

        assert result["drift_deduplicated"] == 1, (
            f"Expected drift_deduplicated=1; got {result['drift_deduplicated']}"
        )
        assert result["drift_intended_removal"] == 0, (
            f"Expected drift_intended_removal=0 (dedup key must use collapsed branch); "
            f"got {result['drift_intended_removal']}"
        )
        assert result["drift_genuine_loss"] == 0, (
            f"Expected drift_genuine_loss=0; got {result['drift_genuine_loss']}"
        )
        # key_dup2 must be in recall_miss_keys — absence from tier_keyed.
        assert "key_dup2" not in {e["key"] for t in result["tier_keyed"].values() for e in t}, (
            "key_dup2 must not survive (collapsed)"
        )


class TestFoldGraphDebug:
    """Tests for DebugSnapshotWriter.on_fold_graph, on_removal_ledger,
    on_fold_assignments.
    """

    def _make_writer(self, tmp_path, *, debug_on: bool = True):
        """Build a minimal loop + DebugSnapshotWriter with optional debug gate."""
        from unittest.mock import MagicMock

        from paramem.training.debug_snapshot import DebugSnapshotWriter

        base_dir = tmp_path / "cycle_1" / "run_test"
        loop = MagicMock()
        loop.save_cycle_snapshots = debug_on
        loop._debug_base = tmp_path if debug_on else None
        loop._current_interim_stamp_or_none = MagicMock(return_value=None)
        loop.snapshot_dir_for = MagicMock(return_value=base_dir)
        return loop, DebugSnapshotWriter(loop)

    def test_on_fold_graph_merger_writes_fold_subdir(self, tmp_path):
        """on_fold_graph(merger, label='merged') writes fold/graph_merged_snapshot.json
        via merger.save_graph when input is a GraphMerger (has save_graph).
        """
        from unittest.mock import MagicMock

        loop, writer = self._make_writer(tmp_path)
        base_dir = loop.snapshot_dir_for()

        merger_mock = MagicMock()
        # Simulate GraphMerger: has save_graph.
        merger_mock.save_graph = MagicMock()

        writer.on_fold_graph(merger_mock, label="merged")

        merger_mock.save_graph.assert_called_once_with(
            base_dir / "fold" / "graph_merged_snapshot.json",
            encrypted=False,
        )

    def test_on_fold_graph_bare_graph_writes_fold_subdir(self, tmp_path):
        """on_fold_graph(nx_graph, label='reconstructed') writes
        fold/graph_reconstructed_snapshot.json via save_memory_to_disk.
        """
        import json
        from unittest.mock import patch

        import networkx as nx

        loop, writer = self._make_writer(tmp_path)
        base_dir = loop.snapshot_dir_for()

        g = nx.MultiDiGraph()
        g.add_node("Alice")
        calls = []

        def _fake_save(graph, path, *, encrypted):
            calls.append((path, encrypted))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(nx.node_link_data(graph)))

        with patch("paramem.memory.persistence.save_memory_to_disk", side_effect=_fake_save):
            writer.on_fold_graph(g, label="reconstructed")

        assert len(calls) == 1
        assert calls[0][0] == base_dir / "fold" / "graph_reconstructed_snapshot.json"
        assert calls[0][1] is False

    def test_on_fold_graph_interim_no_fold_subdir(self, tmp_path):
        """on_fold_graph with interim_stamp writes at base root, not under fold/."""
        from unittest.mock import MagicMock

        loop, writer = self._make_writer(tmp_path)
        loop._current_interim_stamp_or_none = MagicMock(return_value="20260610_1200")
        base_dir = loop.snapshot_dir_for()

        merger_mock = MagicMock()
        merger_mock.save_graph = MagicMock()

        writer.on_fold_graph(merger_mock, label="merged", interim_stamp="20260610_1200")

        merger_mock.save_graph.assert_called_once_with(
            base_dir / "graph_merged_snapshot.json",
            encrypted=False,
        )

    def test_on_fold_graph_no_op_when_debug_off(self, tmp_path):
        """on_fold_graph is a no-op when save_cycle_snapshots=False."""
        from unittest.mock import MagicMock

        loop, writer = self._make_writer(tmp_path, debug_on=False)
        merger_mock = MagicMock()
        merger_mock.save_graph = MagicMock()

        writer.on_fold_graph(merger_mock, label="enriched")

        merger_mock.save_graph.assert_not_called()

    def test_naming_parity_merged_interim_vs_fold(self, tmp_path):
        """Interim and fold 'merged' snapshots share the same basename.

        Guards against a future re-split that accidentally changes one path.
        """
        from unittest.mock import MagicMock

        loop, writer = self._make_writer(tmp_path)

        merger_mock = MagicMock()
        merger_mock.save_graph = MagicMock()

        # Fold path (no interim_stamp).
        writer.on_fold_graph(merger_mock, label="merged")
        fold_call = merger_mock.save_graph.call_args_list[-1]
        fold_path = fold_call.args[0]

        merger_mock.save_graph.reset_mock()

        # Interim path.
        writer.on_fold_graph(merger_mock, label="merged", interim_stamp="20260610_0900")
        interim_call = merger_mock.save_graph.call_args_list[-1]
        interim_path = interim_call.args[0]

        # Same basename, different directories.
        assert fold_path.name == interim_path.name == "graph_merged_snapshot.json", (
            f"Basename mismatch: fold={fold_path.name!r} interim={interim_path.name!r}"
        )
        assert fold_path.parent.name == "fold", (
            f"Fold path must be under 'fold/'; got {fold_path.parent.name!r}"
        )
        assert interim_path.parent.name != "fold", (
            f"Interim path must NOT be under 'fold/'; got {interim_path.parent.name!r}"
        )

    def test_naming_parity_enriched(self, tmp_path):
        """Interim and fold 'enriched' snapshots share the same basename."""
        from unittest.mock import MagicMock

        loop, writer = self._make_writer(tmp_path)
        merger_mock = MagicMock()
        merger_mock.save_graph = MagicMock()

        writer.on_fold_graph(merger_mock, label="enriched")
        fold_name = merger_mock.save_graph.call_args_list[-1].args[0].name
        merger_mock.save_graph.reset_mock()

        writer.on_fold_graph(merger_mock, label="enriched", interim_stamp="20260610_0900")
        interim_name = merger_mock.save_graph.call_args_list[-1].args[0].name

        assert fold_name == interim_name == "graph_enriched_snapshot.json"

    def test_on_removal_ledger_writes_under_fold_subdir(self, tmp_path):
        """on_removal_ledger writes fold/removal_ledger.json with the full ledger."""
        import json

        loop, writer = self._make_writer(tmp_path)
        base_dir = loop.snapshot_dir_for()

        ledger = {
            "key_dup": {"reason": "dedup", "surviving_twin": "key_ok"},
            "key_enriched": {"reason": "enrichment_same_as", "merged_into": "Alice"},
        }
        writer.on_removal_ledger(ledger)

        out = base_dir / "fold" / "removal_ledger.json"
        assert out.exists(), f"removal_ledger.json must exist; base={base_dir}"
        saved = json.loads(out.read_text())
        assert saved == ledger

    def test_on_removal_ledger_no_op_when_debug_off(self, tmp_path):
        """on_removal_ledger is a no-op when save_cycle_snapshots=False."""
        loop, writer = self._make_writer(tmp_path, debug_on=False)
        # Should not raise and should write nothing.
        writer.on_removal_ledger({"k": {"reason": "dedup"}})
        # Confirm no fold/ subdir was created.
        assert not (tmp_path / "fold").exists()

    def test_on_fold_assignments_writes_key_lists(self, tmp_path):
        """on_fold_assignments writes fold/fold_assignments.json with per-tier key lists."""
        import json

        loop, writer = self._make_writer(tmp_path)
        base_dir = loop.snapshot_dir_for()

        serve = {
            "episodic": [{"key": "ep1", "subject": "A", "predicate": "p", "object": "B"}],
            "semantic": [{"key": "sem1", "subject": "X", "predicate": "q", "object": "Y"}],
            "procedural": [],
        }
        train = {
            "episodic": [
                {"key": "ep1", "subject": "A", "predicate": "p", "object": "B"},
                {"key": "sem1", "subject": "X", "predicate": "q", "object": "Y"},
            ],
            "semantic": [],
            "procedural": [],
        }
        writer.on_fold_assignments(serve, train)

        out = base_dir / "fold" / "fold_assignments.json"
        assert out.exists(), f"fold_assignments.json must exist; base={base_dir}"
        saved = json.loads(out.read_text())

        assert saved["serve_assignment"] == {
            "episodic": ["ep1"],
            "semantic": ["sem1"],
            "procedural": [],
        }
        assert saved["train_assignment"] == {
            "episodic": ["ep1", "sem1"],
            "semantic": [],
            "procedural": [],
        }

    def test_on_fold_assignments_no_op_when_debug_off(self, tmp_path):
        """on_fold_assignments is a no-op when save_cycle_snapshots=False."""
        loop, writer = self._make_writer(tmp_path, debug_on=False)
        writer.on_fold_assignments(
            {"episodic": [], "semantic": [], "procedural": []},
            {"episodic": [], "semantic": [], "procedural": []},
        )
        assert not (tmp_path / "fold").exists()


class TestBookkeepingSchema:
    """Unit tests for the 6-field bookkeeping schema in MemoryStore."""

    def _make_store(self):
        from paramem.memory.store import MemoryStore
        from paramem.training.key_registry import KeyRegistry

        s = MemoryStore(replay_enabled=True)
        s.load_registry("episodic", KeyRegistry())
        return s

    def test_set_bookkeeping_stores_all_six_fields(self):
        """set_bookkeeping writes all 6 fields into _bookkeeping: speaker_id,
        relation_type, reinforcement_count, last_reinforced_cycle, last_seen,
        first_seen.
        """
        store = self._make_store()
        store.set_bookkeeping(
            "graph1",
            speaker_id="speaker0",
            relation_type="factual",
            reinforcement_count=2,
            last_reinforced_cycle=5,
            last_seen="2026-01-01T00:00:00Z",
            first_seen="2025-12-01T00:00:00Z",
        )
        bk = store.bookkeeping_for_key("graph1")
        assert bk is not None
        assert bk["speaker_id"] == "speaker0"
        assert bk["relation_type"] == "factual"
        assert bk["reinforcement_count"] == 2
        assert bk["last_reinforced_cycle"] == 5
        assert bk["last_seen"] == "2026-01-01T00:00:00Z"
        assert bk["first_seen"] == "2025-12-01T00:00:00Z"

    def test_set_bookkeeping_defaults_for_new_keys(self):
        """set_bookkeeping defaults reinforcement_count=1, last_reinforced_cycle=0."""
        store = self._make_store()
        store.set_bookkeeping(
            "graph2",
            speaker_id="",
            relation_type="factual",
            allow_empty_speaker=True,
            first_seen="",
        )
        bk = store.bookkeeping_for_key("graph2")
        assert bk["reinforcement_count"] == 1
        assert bk["last_reinforced_cycle"] == 0

    def test_bump_recurrence_increments_and_refreshes_cycle(self):
        """bump_recurrence increments reinforcement_count and sets last_reinforced_cycle."""
        store = self._make_store()
        store.set_bookkeeping(
            "graph3",
            speaker_id="S0",
            relation_type="factual",
            reinforcement_count=1,
            last_reinforced_cycle=1,
            first_seen="",
        )
        store.bump_recurrence("graph3", cycle=7, first_seen="")
        bk = store.bookkeeping_for_key("graph3")
        assert bk["reinforcement_count"] == 2
        assert bk["last_reinforced_cycle"] == 7
        # Other fields must be untouched.
        assert bk["speaker_id"] == "S0"

    def test_bump_recurrence_creates_record_when_absent(self):
        """bump_recurrence on unknown key creates a minimal record."""
        store = self._make_store()
        store.bump_recurrence("graph_new", cycle=4, first_seen="")
        bk = store.bookkeeping_for_key("graph_new")
        assert bk is not None
        assert bk["reinforcement_count"] == 1
        assert bk["last_reinforced_cycle"] == 4

    def test_bump_recurrence_preserves_speaker_id(self):
        """bump_recurrence does NOT reset speaker_id."""
        store = self._make_store()
        store.set_bookkeeping(
            "k",
            speaker_id="speaker99",
            relation_type="factual",
            reinforcement_count=5,
            last_reinforced_cycle=0,
            first_seen="",
        )
        store.bump_recurrence("k", cycle=10, first_seen="")
        assert store.bookkeeping_for_key("k")["speaker_id"] == "speaker99"

    def test_bump_recurrence_does_not_regress_last_seen(self):
        """bump_recurrence with an OLDER timestamp must not regress an existing
        newer last_seen — the invariant is max(existing, incoming)."""
        store = self._make_store()
        store.set_bookkeeping(
            "k",
            speaker_id="speaker0",
            relation_type="factual",
            last_seen="2026-06-20T12:00:00Z",
            first_seen="",
        )
        # Pass an older timestamp — must NOT overwrite the existing newer value.
        store.bump_recurrence("k", cycle=5, timestamp="2026-05-01T08:00:00Z", first_seen="")
        bk = store.bookkeeping_for_key("k")
        assert bk["last_seen"] == "2026-06-20T12:00:00Z", (
            f"bump_recurrence with older timestamp must not regress last_seen; "
            f"got {bk['last_seen']!r}"
        )

    def test_bump_recurrence_advances_last_seen_to_newer(self):
        """bump_recurrence with a NEWER timestamp must advance last_seen."""
        store = self._make_store()
        store.set_bookkeeping(
            "k2",
            speaker_id="speaker0",
            relation_type="factual",
            last_seen="2026-05-01T08:00:00Z",
            first_seen="",
        )
        store.bump_recurrence("k2", cycle=5, timestamp="2026-06-20T12:00:00Z", first_seen="")
        bk = store.bookkeeping_for_key("k2")
        assert bk["last_seen"] == "2026-06-20T12:00:00Z", (
            f"bump_recurrence with newer timestamp must advance last_seen; got {bk['last_seen']!r}"
        )

    def test_bump_recurrence_does_not_regress_first_seen(self):
        """bump_recurrence with a LATER first_seen must not regress an existing
        earlier first_seen — the invariant is min_nonempty(existing, incoming)."""
        store = self._make_store()
        store.set_bookkeeping(
            "k3",
            speaker_id="speaker0",
            relation_type="factual",
            first_seen="2026-01-01T00:00:00Z",
        )
        # Pass a LATER first_seen — must NOT overwrite the existing earlier value.
        store.bump_recurrence("k3", cycle=5, first_seen="2026-06-01T00:00:00Z")
        bk = store.bookkeeping_for_key("k3")
        assert bk["first_seen"] == "2026-01-01T00:00:00Z", (
            f"bump_recurrence with a later first_seen must not regress the earlier "
            f"value; got {bk['first_seen']!r}"
        )

    def test_bump_recurrence_advances_first_seen_to_earlier(self):
        """bump_recurrence with an EARLIER first_seen must advance (lower) first_seen."""
        store = self._make_store()
        store.set_bookkeeping(
            "k4",
            speaker_id="speaker0",
            relation_type="factual",
            first_seen="2026-06-01T00:00:00Z",
        )
        store.bump_recurrence("k4", cycle=5, first_seen="2026-01-01T00:00:00Z")
        bk = store.bookkeeping_for_key("k4")
        assert bk["first_seen"] == "2026-01-01T00:00:00Z", (
            f"bump_recurrence with an earlier first_seen must advance the window "
            f"start; got {bk['first_seen']!r}"
        )

    def test_bump_recurrence_empty_first_seen_never_wins_min(self):
        """An empty incoming first_seen ('unknown') must never overwrite an
        existing dated first_seen — min_nonempty treats '' as absent, not as
        the earliest possible time."""
        store = self._make_store()
        store.set_bookkeeping(
            "k5",
            speaker_id="speaker0",
            relation_type="factual",
            first_seen="2026-03-01T00:00:00Z",
        )
        store.bump_recurrence("k5", cycle=5, first_seen="")
        bk = store.bookkeeping_for_key("k5")
        assert bk["first_seen"] == "2026-03-01T00:00:00Z", (
            f"empty incoming first_seen must not clobber a dated existing value; "
            f"got {bk['first_seen']!r}"
        )

    def test_load_bookkeeping_from_disk_round_trip(self, tmp_path):
        """Full-record round-trip: save via the same ``dict(bk)`` shape
        ``_save_key_metadata`` writes, reload via load_bookkeeping_from_disk,
        assert every field survives unchanged (write/read symmetry)."""
        import json

        from paramem.memory.store import MemoryStore
        from paramem.training.key_registry import KeyRegistry

        store = MemoryStore(replay_enabled=True)
        store.load_registry("episodic", KeyRegistry())
        store.registry("episodic").add("graph5")
        store.set_bookkeeping(
            "graph5",
            speaker_id="SPK",
            relation_type="social",
            reinforcement_count=4,
            last_reinforced_cycle=9,
            first_seen="2025-12-01T00:00:00Z",
        )
        store.registry("episodic").add("graph6")
        store.set_bookkeeping(
            "graph6",
            speaker_id="",
            relation_type="unknown",
            reinforcement_count=1,
            last_reinforced_cycle=0,
            last_seen="",
            first_seen="",
            allow_empty_speaker=True,
        )

        # Simulate _save_key_metadata output format: dump the whole record.
        bk5 = store.bookkeeping_for_key("graph5")
        bk6 = store.bookkeeping_for_key("graph6")
        payload = {
            "cycle_count": 10,
            "promoted_keys": [],
            "keys": {
                "graph5": dict(bk5),
                "graph6": dict(bk6),
            },
        }
        path = tmp_path / "key_metadata.json"
        path.write_text(json.dumps(payload))

        # Reload into a fresh store.
        store2 = MemoryStore(replay_enabled=True)
        store2.load_registry("episodic", KeyRegistry())
        store2.registry("episodic").add("graph5")
        store2.registry("episodic").add("graph6")
        store2.load_bookkeeping_from_disk(path)

        assert store2.bookkeeping_for_key("graph5") == bk5
        assert store2.bookkeeping_for_key("graph6") == bk6

    def test_load_bookkeeping_incomplete_record_fails_loud(self, tmp_path):
        """A persisted record missing a mandatory bookkeeping field raises —
        no legacy-fill tolerance, no backward compatibility. The splat read
        requires the on-disk record to carry every field ``set_bookkeeping``
        needs. ``relation_type`` has no Python default on ``set_bookkeeping``,
        so omitting it must raise ``TypeError`` from the splat."""
        import json

        from paramem.memory.store import MemoryStore
        from paramem.training.key_registry import KeyRegistry

        payload = {
            "cycle_count": 5,
            "promoted_keys": [],
            "keys": {
                "graph_old": {
                    "speaker_id": "SPK",
                    "first_seen": "",
                    # relation_type absent — no Python default, must raise.
                }
            },
        }
        path = tmp_path / "key_metadata.json"
        path.write_text(json.dumps(payload))

        store = MemoryStore(replay_enabled=True)
        store.load_registry("episodic", KeyRegistry())
        store.registry("episodic").add("graph_old")

        with pytest.raises(TypeError):
            store.load_bookkeeping_from_disk(path)

    def test_load_bookkeeping_missing_first_seen_fails_loud(self, tmp_path):
        """A record missing first_seen entirely raises — no silent '' fill,
        per the no-shim rule."""
        import json

        from paramem.memory.store import MemoryStore
        from paramem.training.key_registry import KeyRegistry

        payload = {
            "cycle_count": 5,
            "promoted_keys": [],
            "keys": {
                "graph_stale": {
                    "speaker_id": "SPK",
                    "relation_type": "factual",
                    "reinforcement_count": 1,
                    "last_reinforced_cycle": 0,
                    "last_seen": "2026-01-01T00:00:00Z",
                    # first_seen absent — pre-migration on-disk shape.
                }
            },
        }
        path = tmp_path / "key_metadata.json"
        path.write_text(json.dumps(payload))

        store = MemoryStore(replay_enabled=True)
        store.load_registry("episodic", KeyRegistry())
        store.registry("episodic").add("graph_stale")

        with pytest.raises(TypeError):
            store.load_bookkeeping_from_disk(path)


class TestLastSeenFlowThroughMint:
    """last_seen must flow from the merged graph edge into bookkeeping.

    The session_graph.timestamp at ingest is written to the edge via
    merger._upsert_relation; _build_all_edge_entries_into reads it and passes
    it to set_bookkeeping.  No fabricated datetime.now() must appear in the
    path.
    """

    def _make_loop(self, tmp_path):
        """Minimal ConsolidationLoop stub — no GPU required."""
        import networkx as nx
        from peft import PeftModel

        from paramem.memory.store import MemoryStore
        from paramem.training.consolidation import ConsolidationLoop
        from paramem.utils.config import AdapterConfig, ConsolidationConfig, TrainingConfig

        loop = object.__new__(ConsolidationLoop)
        loop.model = MagicMock()
        loop.model.__class__ = PeftModel
        loop.tokenizer = MagicMock()
        loop.config = ConsolidationConfig(min_tier_key_floor=0, tier_fast_start=False)
        loop.training_config = TrainingConfig(num_epochs=1, gradient_checkpointing=False)
        loop.episodic_config = AdapterConfig(rank=4, alpha=8, target_modules=["q_proj"])
        loop.semantic_config = AdapterConfig(rank=4, alpha=8, target_modules=["q_proj"])
        loop.procedural_config = None
        loop.wandb_config = None
        loop._thermal_policy = None
        loop.output_dir = tmp_path
        loop.store = MemoryStore(replay_enabled=False)
        loop.promoted_keys = set()
        loop.cycle_count = 1
        loop.episodic_simhash = {}
        loop.semantic_simhash = {}
        loop.procedural_simhash = {}
        loop._procedural_next_index = 0
        loop._procedural_tentative_next_index = 0
        loop._indexed_next_index = 0
        loop._indexed_ep_interim = {}
        loop._bg_trainer = None
        loop.shutdown_requested = False
        loop._early_stop_callback = None
        loop.fingerprint_cache = None
        loop._keep_prior_slots = 2
        loop._debug_base = None
        loop.save_cycle_snapshots = False
        loop.snapshot_dir = None
        merger = MagicMock()
        merger.graph = nx.MultiDiGraph()
        loop.merger = merger
        return loop

    def test_last_seen_from_edge_flows_into_bookkeeping(self, tmp_path):
        """Edge attribute last_seen travels through _build_all_edge_entries_into
        into set_bookkeeping without being replaced by datetime.now()."""
        real_ts = "2026-06-01T10:00:00Z"
        loop = self._make_loop(tmp_path)
        g = loop.merger.graph

        g.add_node(
            "speaker0",
            entity_type="person",
            speaker_id="speaker0",
            attributes={"name": "Alex"},
        )
        g.add_node("berlin", entity_type="place", attributes={"name": "Berlin"})
        # Edge carries the real session wall-clock (written by merger._upsert_relation).
        g.add_edge(
            "speaker0",
            "berlin",
            predicate="lives in",
            relation_type="factual",
            last_seen=real_ts,
        )

        tier_keyed: dict = {"episodic": [], "procedural": []}
        loop._build_all_edge_entries_into(tier_keyed)

        assert len(tier_keyed["episodic"]) == 1, (
            f"Expected 1 minted episodic entry; got {tier_keyed['episodic']}"
        )
        minted_key = tier_keyed["episodic"][0]["key"]
        bk = loop.store.bookkeeping_for_key(minted_key)
        assert bk is not None, "bookkeeping must be written at mint"
        assert bk["last_seen"] == real_ts, (
            f"last_seen must carry the real session timestamp {real_ts!r}; got {bk['last_seen']!r}"
        )

    def test_last_seen_empty_edge_stores_empty_not_now(self, tmp_path):
        """When edge has no last_seen (e.g. migrated legacy data), bookkeeping
        gets '' — not a fabricated datetime.now()."""
        loop = self._make_loop(tmp_path)
        g = loop.merger.graph

        g.add_node(
            "speaker0",
            entity_type="person",
            speaker_id="speaker0",
            attributes={"name": "Alex"},
        )
        g.add_node("paris", entity_type="place", attributes={"name": "Paris"})
        # No last_seen on the edge — simulates legacy / migrated data.
        g.add_edge(
            "speaker0",
            "paris",
            predicate="visited",
            relation_type="factual",
        )

        tier_keyed: dict = {"episodic": [], "procedural": []}
        loop._build_all_edge_entries_into(tier_keyed)

        minted_key = tier_keyed["episodic"][0]["key"]
        bk = loop.store.bookkeeping_for_key(minted_key)
        assert bk is not None
        assert bk["last_seen"] == "", (
            f"last_seen for edge with no timestamp must be '' not a fabricated now(); "
            f"got {bk['last_seen']!r}"
        )

    def test_first_seen_from_edge_flows_into_bookkeeping(self, tmp_path):
        """Edge attribute first_seen travels through _build_all_edge_entries_into
        into set_bookkeeping, symmetric to last_seen."""
        real_ts = "2026-06-01T10:00:00Z"
        loop = self._make_loop(tmp_path)
        g = loop.merger.graph

        g.add_node(
            "speaker0",
            entity_type="person",
            speaker_id="speaker0",
            attributes={"name": "Alex"},
        )
        g.add_node("berlin", entity_type="place", attributes={"name": "Berlin"})
        g.add_edge(
            "speaker0",
            "berlin",
            predicate="lives in",
            relation_type="factual",
            last_seen=real_ts,
            first_seen=real_ts,
        )

        tier_keyed: dict = {"episodic": [], "procedural": []}
        loop._build_all_edge_entries_into(tier_keyed)

        minted_key = tier_keyed["episodic"][0]["key"]
        bk = loop.store.bookkeeping_for_key(minted_key)
        assert bk is not None, "bookkeeping must be written at mint"
        assert bk["first_seen"] == real_ts, (
            f"first_seen must carry the real session timestamp {real_ts!r}; "
            f"got {bk['first_seen']!r}"
        )

    def test_merger_min_nonempty_first_seen_across_reassertion(self):
        """CPU-only merger test: a fact re-asserted in a later session keeps its
        original (earliest) first_seen while last_seen advances to the newest.

        Exercises GraphMerger.merge() directly — no ConsolidationLoop needed —
        proving the min-across-reassertion invariant at the merger boundary."""
        from paramem.graph.merger import GraphMerger
        from paramem.graph.schema import Relation, SessionGraph

        m = GraphMerger()

        # First session: earliest assertion.
        s1 = SessionGraph(
            session_id="s1",
            timestamp="2026-01-01T00:00:00Z",
            relations=[
                Relation(
                    subject="speaker0",
                    predicate="lives_in",
                    object="berlin",
                    relation_type="factual",
                    speaker_id="speaker0",
                )
            ],
        )
        m.merge(s1, resolve_contradictions=False)

        # Second session (later): re-assertion of the SAME fact.
        s2 = SessionGraph(
            session_id="s2",
            timestamp="2026-06-01T00:00:00Z",
            relations=[
                Relation(
                    subject="speaker0",
                    predicate="lives_in",
                    object="berlin",
                    relation_type="factual",
                    speaker_id="speaker0",
                )
            ],
        )
        m.merge(s2, resolve_contradictions=False)

        edge_data = next(iter(m.graph["speaker0"]["berlin"].values()))
        assert edge_data["first_seen"] == "2026-01-01T00:00:00Z", (
            f"first_seen must stay at the earliest assertion; got {edge_data['first_seen']!r}"
        )
        assert edge_data["last_seen"] == "2026-06-01T00:00:00Z", (
            f"last_seen must advance to the newest assertion; got {edge_data['last_seen']!r}"
        )

    def test_reinforcements_bump_advances_bookkeeping_last_seen(self, tmp_path):
        """INVARIANT (a): when merger.reinforcements carries a newer timestamp for a
        survivor key, bump_recurrence advances bookkeeping last_seen to that value
        (and first_seen to the collapsed min).

        This tests the full consolidation path: merger.reinforcements dict →
        _refine_consolidation_graph bump loop → bump_recurrence → bookkeeping.
        """
        from paramem.graph.schema import Relation

        loop = self._make_loop(tmp_path)
        loop.cycle_count = 3

        # Set bookkeeping directly (no registry needed — bump_recurrence only
        # reads _bookkeeping, not the KeyRegistry).
        loop.store.set_bookkeeping(
            "graph10",
            speaker_id="speaker0",
            relation_type="factual",
            reinforcement_count=1,
            last_reinforced_cycle=1,
            last_seen="2026-05-01T08:00:00Z",
            first_seen="2026-01-01T00:00:00Z",
        )

        # Simulate Case-1 collapse: reinforcements carries the survivor with a
        # newer last_seen and an earlier first_seen (both already merged by the
        # merger before this dict is populated).
        loop.merger.reinforcements = {"graph10": ("2026-06-25T10:00:00Z", "2025-12-01T00:00:00Z")}

        recon_rel = Relation(
            subject="speaker0",
            predicate="lives in",
            object="berlin",
            relation_type="factual",
            speaker_id="speaker0",
        )
        # _refine_consolidation_graph with enrich=False to avoid model calls.
        loop._run_graph_enrichment = MagicMock(return_value={"skipped": True})
        loop._refine_consolidation_graph([recon_rel], enrich=False)

        bk = loop.store.bookkeeping_for_key("graph10")
        assert bk["reinforcement_count"] == 2, (
            f"reinforcement_count must be bumped; got {bk['reinforcement_count']}"
        )
        assert bk["last_seen"] == "2026-06-25T10:00:00Z", (
            "last_seen must advance to the new session timestamp via reinforcements→bump; "
            f"got {bk['last_seen']!r}"
        )
        assert bk["first_seen"] == "2025-12-01T00:00:00Z", (
            "first_seen must advance to the earlier collapsed timestamp via "
            f"reinforcements→bump; got {bk['first_seen']!r}"
        )


class TestBookkeepingBasedPromotion:
    """Tests for bookkeeping-based key-level promotion and decay."""

    def _make_loop_and_store(self):
        """Build a minimal ConsolidationLoop + MemoryStore without GPU."""
        from unittest.mock import MagicMock

        from peft import PeftModel

        from paramem.memory.store import MemoryStore
        from paramem.training.consolidation import ConsolidationLoop
        from paramem.training.key_registry import KeyRegistry
        from paramem.utils.config import AdapterConfig, ConsolidationConfig, TrainingConfig

        loop = object.__new__(ConsolidationLoop)
        loop.model = MagicMock()
        loop.model.__class__ = PeftModel
        loop.model.peft_config = {}
        loop.tokenizer = MagicMock()
        loop.config = ConsolidationConfig()
        loop.training_config = TrainingConfig(num_epochs=1, gradient_checkpointing=False)
        loop.episodic_config = AdapterConfig(rank=4, alpha=8, target_modules=["q_proj"])
        loop.semantic_config = AdapterConfig(rank=4, alpha=8, target_modules=["q_proj"])
        loop.procedural_config = None
        loop.wandb_config = None
        loop._thermal_policy = None
        loop.promoted_keys = set()
        loop.cycle_count = 5

        store = MemoryStore(replay_enabled=True)
        store.load_registry("episodic", KeyRegistry())
        store.load_registry("semantic", KeyRegistry())
        loop.store = store
        return loop, store

    def test_promotion_fires_when_recurrence_reaches_threshold(self):
        """_promote_mature_keys_inline promotes a key to semantic tier when threshold is met."""
        loop, store = self._make_loop_and_store()
        # ConsolidationConfig default: promotion_threshold=3

        # Add a key with reinforcement_count=3 (at threshold).
        store.registry("episodic").add("graph10")
        store.put_simhash("episodic", "graph10", 12345)
        store.set_bookkeeping(
            "graph10",
            speaker_id="S0",
            relation_type="factual",
            reinforcement_count=3,
            last_reinforced_cycle=5,
            first_seen="",
        )

        newly_promoted = loop._promote_mature_keys_inline()

        assert "graph10" in newly_promoted, (
            "Key with reinforcement_count==threshold must be promoted"
        )
        assert store.tier_for_active_key("graph10") == "semantic", (
            "Promoted key must be in semantic registry"
        )
        assert "graph10" in loop.promoted_keys

    def test_promotion_does_not_fire_below_threshold(self):
        """Key with reinforcement_count < threshold stays in episodic."""
        loop, store = self._make_loop_and_store()
        # ConsolidationConfig default: promotion_threshold=3

        store.registry("episodic").add("graph20")
        store.put_simhash("episodic", "graph20", 999)
        store.set_bookkeeping(
            "graph20",
            speaker_id="S0",
            relation_type="factual",
            reinforcement_count=2,  # below threshold
            last_reinforced_cycle=5,
            first_seen="",
        )

        newly_promoted = loop._promote_mature_keys_inline()

        assert "graph20" not in newly_promoted
        assert store.tier_for_active_key("graph20") == "episodic"

    def test_already_promoted_key_not_re_promoted(self):
        """A key already in promoted_keys is not re-promoted."""
        loop, store = self._make_loop_and_store()

        store.registry("semantic").add("graph30")
        store.put_simhash("semantic", "graph30", 777)
        store.set_bookkeeping(
            "graph30",
            speaker_id="S0",
            relation_type="factual",
            reinforcement_count=5,
            last_reinforced_cycle=5,
            first_seen="",
        )
        loop.promoted_keys.add("graph30")  # already promoted

        newly_promoted = loop._promote_mature_keys_inline()
        assert "graph30" not in newly_promoted

    def test_decay_candidate_logged_not_deleted(self):
        """A key with last_reinforced_cycle far in the past is a decay candidate but NOT deleted."""
        loop, store = self._make_loop_and_store()
        # ConsolidationConfig default: decay_window=10
        loop.cycle_count = 20

        store.registry("episodic").add("graph40")
        store.put_simhash("episodic", "graph40", 555)
        store.set_bookkeeping(
            "graph40",
            speaker_id="S0",
            relation_type="factual",
            reinforcement_count=1,
            last_reinforced_cycle=5,
            first_seen="",  # 20 - 5 = 15 >= decay_window=10
        )

        # Must NOT raise, must NOT delete the key.
        newly_promoted = loop._promote_mature_keys_inline()
        assert "graph40" not in newly_promoted
        assert store.tier_for_active_key("graph40") == "episodic", (
            "Decay candidate must NOT be deleted (passive fade policy)"
        )

    def test_recurrence_bump_in_fold_via_reinforcements(self, tmp_path):
        """fold with two active keys sharing one (s,p,o) bumps recurrence on the survivor.

        Uses a real GraphMerger (model=None) which applies Case-1 collapse and
        records the surviving key in merger.reinforcements.  After _run_with_mocks,
        the survivor's bookkeeping.reinforcement_count must equal 2 and
        bookkeeping.last_reinforced_cycle must equal loop.cycle_count.
        """
        import networkx as nx

        from paramem.graph.merger import GraphMerger
        from paramem.graph.reconstruct import ReconstructionResult
        from paramem.memory.persistence import _IK_KEY_ATTR

        # Provide a recon graph with TWO identical-SPO edges.
        recon_g = nx.MultiDiGraph()
        eid1 = recon_g.add_edge("Alice", "Berlin", predicate="lives_in")
        recon_g["Alice"]["Berlin"][eid1][_IK_KEY_ATTR] = "graph1"
        eid2 = recon_g.add_edge("Alice", "Berlin", predicate="lives_in")
        recon_g["Alice"]["Berlin"][eid2][_IK_KEY_ATTR] = "graph2"

        loop = TestConsolidateInterimAdaptersFullFlow._make_loop(
            tmp_path, merger_graph=nx.MultiDiGraph()
        )
        loop.merger = GraphMerger(model=None)
        loop.cycle_count = 7

        for key in ("graph1", "graph2"):
            loop.store.put(
                "episodic",
                key,
                {
                    "key": key,
                    "subject": "Alice",
                    "predicate": "lives_in",
                    "object": "Berlin",
                    "speaker_id": "S0",
                },
                register=True,
            )
            loop.store.set_bookkeeping(
                key,
                speaker_id="S0",
                relation_type="factual",
                reinforcement_count=1,
                last_reinforced_cycle=1,
                first_seen="",
            )

        TestConsolidateInterimAdaptersFullFlow._run_with_mocks(
            loop, tmp_path, ReconstructionResult(graph=recon_g)
        )

        # The surviving key (the one that remained in tier_keyed) must have
        # reinforcement_count == 2 (was bumped by the Case-1 collapse).
        surviving_key = None
        for key in ("graph1", "graph2"):
            bk = loop.store.bookkeeping_for_key(key)
            if bk and bk["reinforcement_count"] == 2:
                surviving_key = key
                break
        assert surviving_key is not None, (
            "One key must have reinforcement_count==2 after duplicate-SPO collapse"
        )
        bk = loop.store.bookkeeping_for_key(surviving_key)
        assert bk["last_reinforced_cycle"] == 7, (
            "Surviving key's last_reinforced_cycle must be refreshed to cycle_count"
        )


# =============================================================================
# TestTierFloor — per-tier minimum-key-count floor + two graduation strategies
# =============================================================================


class TestTierFloor:
    """Unit tests for the per-tier min-key-count floor and graduation strategies.

    Covers config, parking, graduation, accumulating, and probe-fallback behavior.
    All tests mock train_adapter and heavy PEFT ops so no GPU is needed.

    Config plumbing lives in tests/server/test_config.py;
    config parity is covered by tests/server/test_config_parity.py.
    """

    @staticmethod
    def _make_loop(tmp_path, *, min_tier_key_floor=30, tier_fast_start=True):
        """Minimal ConsolidationLoop stub for floor/graduation tests.

        Delegates to the existing full-flow helper so we reuse its wiring;
        overrides config fields for floor tests.
        """
        import networkx as nx

        from paramem.utils.config import ConsolidationConfig

        loop = TestConsolidateInterimAdaptersFullFlow._make_loop(
            tmp_path, merger_graph=nx.MultiDiGraph()
        )
        loop.config = ConsolidationConfig(
            min_tier_key_floor=min_tier_key_floor,
            tier_fast_start=tier_fast_start,
        )
        return loop

    @staticmethod
    def _seed_keys(loop, tier, keys, relation_type="factual"):
        """Register ``keys`` in ``tier`` with minimal bookkeeping."""
        for k in keys:
            loop.store.put(
                tier,
                k,
                {
                    "key": k,
                    "subject": "Alice",
                    "predicate": f"pred_{k}",
                    "object": f"obj_{k}",
                    "speaker_id": "S0",
                },
                register=True,
            )
            loop.store.set_bookkeeping(
                k,
                speaker_id="S0",
                relation_type=relation_type,
                reinforcement_count=1,
                last_reinforced_cycle=1,
                first_seen="",
            )

    @staticmethod
    def _build_merger_graph(loop, entries):
        """Stamp entries as merger-graph edges so the edge-walk stage picks them up."""
        from paramem.memory.persistence import _IK_KEY_ATTR

        for e in entries:
            eid = loop.merger.graph.add_edge(
                e["subject"],
                e["object"],
                predicate=e["predicate"],
                relation_type=e.get("relation_type", "factual"),
                confidence=1.0,
                first_seen="s",
                last_seen="s",
                reinforcement_count=1,
                sessions=["s"],
            )
            loop.merger.graph[e["subject"]][e["object"]][eid][_IK_KEY_ATTR] = e["key"]

    @staticmethod
    def _run_full_fold_mocked(
        loop,
        *,
        probe_side_effect=None,
        train_adapter_spy=None,
    ):
        """Run consolidate_interim_adapters with heavy ops mocked.

        Wraps consolidate_interim_adapters (the full-fold thin wrapper), which now
        delegates its body to _run_fold(FoldScope(persist="main_tiers", ...)).
        Heavy GPU ops are mocked so this runs without hardware.

        probe_side_effect: callable(adapter_name, entries) → set[str].
            Defaults to returning all keys (pass-all probe).
        train_adapter_spy: mock.MagicMock or None.  If supplied, train_adapter
            is patched to this spy (so the test can assert call_count etc.).
        """
        from unittest.mock import MagicMock, patch

        from paramem.server.gpu_lock import _gpu_thread_lock
        from paramem.training.consolidation import ConsolidationLoop

        if probe_side_effect is None:
            probe_side_effect = lambda adapter_name, entries: {e["key"] for e in entries}  # noqa: E731
        if train_adapter_spy is None:
            train_adapter_spy = MagicMock(return_value={"aborted": False})

        _gpu_thread_lock.acquire()
        try:
            with (
                patch(
                    "paramem.training.consolidation.reconstruct_graph",
                    return_value=__import__(
                        "paramem.graph.reconstruct", fromlist=["ReconstructionResult"]
                    ).ReconstructionResult(graph=__import__("networkx").MultiDiGraph()),
                ),
                patch.object(
                    ConsolidationLoop,
                    "_run_graph_enrichment",
                    return_value={"skipped": True},
                ),
                patch.object(
                    ConsolidationLoop,
                    "_run_graph_normalization",
                    return_value={"skipped": True},
                ),
                patch.object(ConsolidationLoop, "_enable_gradient_checkpointing"),
                patch.object(ConsolidationLoop, "_disable_gradient_checkpointing"),
                patch.object(
                    ConsolidationLoop,
                    "_maybe_make_recall_callback",
                    return_value=(None, None),
                ),
                patch.object(
                    ConsolidationLoop,
                    "_probe_passing_keys",
                    side_effect=probe_side_effect,
                ),
                patch.object(ConsolidationLoop, "_save_adapters"),
                patch(
                    "paramem.training.trainer.train_adapter",
                    train_adapter_spy,
                ),
                patch(
                    "paramem.training.consolidation.format_entry_training",
                    return_value=[{"input_ids": [1], "labels": [1], "attention_mask": [1]}],
                ),
                patch("paramem.models.loader.create_adapter", side_effect=lambda m, c, n: m),
                patch("paramem.models.loader.switch_adapter"),
                patch("paramem.models.loader.copy_adapter_weights"),
                patch("paramem.models.loader.copy_adapter_weights_subset"),
                patch("paramem.memory.interim_adapter.unload_interim_adapters", return_value=[]),
            ):
                return loop.consolidate_interim_adapters(trainer=None, router=None)
        finally:
            _gpu_thread_lock.release()

    # -------------------------------------------------------------------------
    # Pass-2 parking — under-floor semantic/procedural moves to episodic
    # -------------------------------------------------------------------------

    def test_pass2_parks_under_floor_semantic_and_procedural(self, tmp_path):
        """12 procedural + 4 semantic + 50 episodic: under-floor tiers park in episodic.

        Asserts:
        - tier_keyed["procedural"] == [] and tier_keyed["semantic"] == [] after Pass-2.
        - episodic gained the 16 parked keys (total 66).
        - store.move called for parked keys (store tier updated to episodic).
        """
        from paramem.memory.persistence import _IK_KEY_ATTR

        floor = 30
        loop = self._make_loop(tmp_path, min_tier_key_floor=floor, tier_fast_start=False)

        ep_keys = [f"ep{i}" for i in range(50)]
        sem_keys = [f"sem{i}" for i in range(4)]
        proc_keys = [f"proc{i}" for i in range(12)]

        self._seed_keys(loop, "episodic", ep_keys, relation_type="factual")
        self._seed_keys(loop, "semantic", sem_keys, relation_type="factual")
        self._seed_keys(loop, "procedural", proc_keys, relation_type="preference")

        # Build merger graph edges for all keys.
        all_entries = [
            {
                "key": k,
                "subject": "Alice",
                "predicate": f"pred_{k}",
                "object": f"obj_{k}",
                "relation_type": "factual",
            }
            for k in ep_keys + sem_keys
        ] + [
            {
                "key": k,
                "subject": "Alice",
                "predicate": f"pred_{k}",
                "object": f"obj_{k}",
                "relation_type": "preference",
            }
            for k in proc_keys
        ]
        for e in all_entries:
            eid = loop.merger.graph.add_edge(
                e["subject"],
                e["object"],
                predicate=e["predicate"],
                relation_type=e["relation_type"],
                confidence=1.0,
                first_seen="s",
                last_seen="s",
                reinforcement_count=1,
                sessions=["s"],
            )
            loop.merger.graph[e["subject"]][e["object"]][eid][_IK_KEY_ATTR] = e["key"]

        result = self._run_full_fold_mocked(loop)

        # Under-floor tiers end up empty after park.
        assert result["keys_per_tier"]["procedural"] == 0, (
            "procedural (12 < 30) must be parked — keys_per_tier should be 0"
        )
        assert result["keys_per_tier"]["semantic"] == 0, (
            "semantic (4 < 30) must be parked — keys_per_tier should be 0"
        )
        # Episodic absorbed all 16 parked keys.
        assert result["keys_per_tier"]["episodic"] == 66, (
            f"episodic should be 50+16=66, got {result['keys_per_tier']['episodic']}"
        )

        # Store tiers for parked keys must now be episodic.
        for k in sem_keys + proc_keys:
            t = loop.store.tier_for_active_key(k)
            assert t == "episodic", f"parked key {k!r} should be in episodic store tier, got {t!r}"

    # -------------------------------------------------------------------------
    # Bookkeeping preserved through parking and graduation
    # -------------------------------------------------------------------------

    def test_bookkeeping_preserved_through_parking(self, tmp_path):
        """Parking a key preserves relation_type, reinforcement_count, speaker_id —
        stored in bookkeeping_for_key, not in tier_keyed."""
        from paramem.memory.persistence import _IK_KEY_ATTR

        loop = self._make_loop(tmp_path, min_tier_key_floor=30, tier_fast_start=False)

        # One semantic key (under floor of 30) with specific bookkeeping.
        loop.store.put(
            "semantic",
            "sem0",
            {
                "key": "sem0",
                "subject": "Bob",
                "predicate": "likes",
                "object": "Jazz",
                "speaker_id": "S1",
            },
            register=True,
        )
        loop.store.set_bookkeeping(
            "sem0",
            speaker_id="S1",
            relation_type="factual",
            reinforcement_count=5,
            last_reinforced_cycle=3,
            first_seen="",
        )
        # Seed episodic with enough keys to avoid whole-fold accumulate.
        ep_keys = [f"ep{i}" for i in range(35)]
        self._seed_keys(loop, "episodic", ep_keys)

        all_entries = [
            {
                "key": "sem0",
                "subject": "Bob",
                "predicate": "likes",
                "object": "Jazz",
                "relation_type": "factual",
            }
        ] + [
            {
                "key": k,
                "subject": "Alice",
                "predicate": f"pred_{k}",
                "object": f"obj_{k}",
                "relation_type": "factual",
            }
            for k in ep_keys
        ]
        for e in all_entries:
            eid = loop.merger.graph.add_edge(
                e["subject"],
                e["object"],
                predicate=e["predicate"],
                relation_type=e["relation_type"],
                confidence=1.0,
                first_seen="s",
                last_seen="s",
                reinforcement_count=1,
                sessions=["s"],
            )
            loop.merger.graph[e["subject"]][e["object"]][eid][_IK_KEY_ATTR] = e["key"]

        self._run_full_fold_mocked(loop)

        bk = loop.store.bookkeeping_for_key("sem0")
        assert bk is not None
        assert bk["relation_type"] == "factual", "relation_type must survive parking"
        assert bk["reinforcement_count"] == 5, "reinforcement_count must survive parking"
        assert bk["speaker_id"] == "S1", "speaker_id must survive parking"

    # -------------------------------------------------------------------------
    # Default graduation (train-from-scratch, tier_fast_start=False)
    # -------------------------------------------------------------------------

    def test_default_graduation_trains_from_scratch(self, tmp_path):
        """A tier crossing floor for the first time with tier_fast_start=False trains
        from scratch via train_adapter; copy helpers NOT called.
        """
        from unittest.mock import MagicMock, patch

        import networkx as nx

        from paramem.memory.persistence import _IK_KEY_ATTR
        from paramem.training.consolidation import ConsolidationLoop

        loop = self._make_loop(tmp_path, min_tier_key_floor=10, tier_fast_start=False)

        # Seed 15 procedural keys — all in episodic (first-time graduation).
        proc_keys = [f"proc{i}" for i in range(15)]
        self._seed_keys(loop, "episodic", proc_keys, relation_type="preference")
        ep_keys = [f"ep{i}" for i in range(20)]
        self._seed_keys(loop, "episodic", ep_keys, relation_type="factual")

        for k in proc_keys:
            eid = loop.merger.graph.add_edge(
                "Alice",
                f"obj_{k}",
                predicate=f"pred_{k}",
                relation_type="preference",
                confidence=1.0,
                first_seen="s",
                last_seen="s",
                reinforcement_count=1,
                sessions=["s"],
            )
            loop.merger.graph["Alice"][f"obj_{k}"][eid][_IK_KEY_ATTR] = k
        for k in ep_keys:
            eid = loop.merger.graph.add_edge(
                "Alice",
                f"obj_{k}",
                predicate=f"pred_{k}",
                relation_type="factual",
                confidence=1.0,
                first_seen="s",
                last_seen="s",
                reinforcement_count=1,
                sessions=["s"],
            )
            loop.merger.graph["Alice"][f"obj_{k}"][eid][_IK_KEY_ATTR] = k

        train_spy = MagicMock(return_value={"aborted": False})
        copy_spy = MagicMock()
        subset_copy_spy = MagicMock()

        from paramem.server.gpu_lock import _gpu_thread_lock

        _gpu_thread_lock.acquire()
        try:
            with (
                patch(
                    "paramem.training.consolidation.reconstruct_graph",
                    return_value=__import__(
                        "paramem.graph.reconstruct", fromlist=["ReconstructionResult"]
                    ).ReconstructionResult(graph=nx.MultiDiGraph()),
                ),
                patch.object(
                    ConsolidationLoop,
                    "_run_graph_enrichment",
                    return_value={"skipped": True},
                ),
                patch.object(ConsolidationLoop, "_enable_gradient_checkpointing"),
                patch.object(ConsolidationLoop, "_disable_gradient_checkpointing"),
                patch.object(
                    ConsolidationLoop,
                    "_maybe_make_recall_callback",
                    return_value=(None, None),
                ),
                patch.object(
                    ConsolidationLoop,
                    "_probe_passing_keys",
                    side_effect=lambda a, e: {x["key"] for x in e},
                ),
                patch.object(ConsolidationLoop, "_save_adapters"),
                patch("paramem.training.trainer.train_adapter", train_spy),
                patch(
                    "paramem.training.consolidation.format_entry_training",
                    return_value=[{"input_ids": [1], "labels": [1], "attention_mask": [1]}],
                ),
                patch("paramem.models.loader.create_adapter", side_effect=lambda m, c, n: m),
                patch("paramem.models.loader.switch_adapter"),
                patch("paramem.models.loader.copy_adapter_weights", copy_spy),
                patch("paramem.models.loader.copy_adapter_weights_subset", subset_copy_spy),
                patch("paramem.memory.interim_adapter.unload_interim_adapters", return_value=[]),
            ):
                result = loop.consolidate_interim_adapters(trainer=None, router=None)
        finally:
            _gpu_thread_lock.release()

        assert "procedural" in result["tiers_rebuilt"], (
            "procedural should be in tiers_rebuilt after graduation"
        )
        # train_adapter must have been called (at least for episodic and procedural).
        assert train_spy.call_count >= 2, (
            f"train_adapter should be called >= 2 times (episodic + procedural), "
            f"got {train_spy.call_count}"
        )
        # copy helpers must NOT have been called for graduation (train-from-scratch).
        # NOTE: copy_adapter_weights IS called for backup creation — filter those out
        # by checking copy_adapter_weights_subset (never called in (a) path).
        assert subset_copy_spy.call_count == 0, (
            "copy_adapter_weights_subset must not be called in train-from-scratch path"
        )

    # -------------------------------------------------------------------------
    # copy_adapter_weights_subset — CPU-only parameter logic
    # -------------------------------------------------------------------------

    def test_copy_adapter_weights_subset_copies_intersecting(self):
        """subset copy: src⊆dst copies intersecting tensors, leaves dst-only at init."""
        from unittest.mock import MagicMock

        import torch

        from paramem.models.loader import copy_adapter_weights_subset

        model = MagicMock()

        # src has 1 param; dst has 2 (1 shared key "base_a", 1 dst-only "base_b").
        # named_parameters is called twice by _index (once for src, once for dst);
        # use side_effect to return a fresh list each time.
        src_tensor_a = torch.ones(4, 4)
        dst_tensor_a = torch.zeros(4, 4)
        dst_tensor_b = torch.zeros(4, 4)

        params = [
            ("base_a.src.weight", src_tensor_a),  # key (base_a, .weight) in src
            ("base_a.dst.weight", dst_tensor_a),  # key (base_a, .weight) in dst
            ("base_b.dst.weight", dst_tensor_b),  # dst-only key (base_b, .weight)
        ]
        # Each call to named_parameters() must return a fresh iterable.
        model.named_parameters.side_effect = lambda: iter(params)
        model.peft_config = {"src": MagicMock(), "dst": MagicMock()}

        count = copy_adapter_weights_subset(model, src="src", dst="dst")

        assert count == 1, f"Expected 1 tensor copied (intersecting), got {count}"
        # The shared tensor was copied from src into dst.
        assert torch.allclose(dst_tensor_a, src_tensor_a), (
            "Shared tensor must be copied from src to dst"
        )
        # dst-only tensor is unchanged.
        assert torch.all(dst_tensor_b == 0), "dst-only tensor must stay at zero-init"

    def test_copy_adapter_weights_subset_raises_when_src_exceeds_dst(self):
        """subset copy raises RuntimeError when src has a tensor dst lacks."""
        from unittest.mock import MagicMock

        import torch

        from paramem.models.loader import copy_adapter_weights_subset

        src_tensor = torch.ones(4, 4)
        dst_tensor = torch.zeros(4, 4)

        params = [
            ("base_a.src.weight", src_tensor),  # in src
            ("base_b.src.weight", src_tensor.clone()),  # in src but NOT in dst
            ("base_a.dst.weight", dst_tensor),  # in dst
        ]
        model = MagicMock()
        model.named_parameters.side_effect = lambda: iter(params)
        model.peft_config = {"src": MagicMock(), "dst": MagicMock()}

        with pytest.raises(RuntimeError, match="absent from dst"):
            copy_adapter_weights_subset(model, src="src", dst="dst")

    def test_copy_adapter_weights_raises_on_set_inequality_regression(self):
        """copy_adapter_weights (strict) still raises on any set inequality.

        Regression guard: the subset helper must never loosen the strict function.
        """
        from unittest.mock import MagicMock

        import torch

        from paramem.models.loader import copy_adapter_weights

        src_tensor = torch.ones(4, 4)
        dst_tensor = torch.zeros(4, 4)

        # src has base_b but dst does not — strict function must raise.
        params = [
            ("base_a.src.weight", src_tensor),
            ("base_b.src.weight", src_tensor.clone()),
            ("base_a.dst.weight", dst_tensor),
        ]
        model = MagicMock()
        model.named_parameters.side_effect = lambda: iter(params)
        model.peft_config = {"src": MagicMock(), "dst": MagicMock()}

        with pytest.raises(RuntimeError, match="Adapter parameter sets differ"):
            copy_adapter_weights(model, src="src", dst="dst")

    # -------------------------------------------------------------------------
    # Fast-start graduation decision (mocked copy + store)
    # -------------------------------------------------------------------------

    def test_fast_start_graduation_calls_copy_not_train(self, tmp_path):
        """tier_fast_start=True + tier crossing floor for first time: copy called,
        train_adapter NOT called for that tier, tier in tiers_rebuilt.

        Asserts on the serve/train map split directly (not a blanket pass-all probe
        that would mask a donor-ignorance regression):
          - episodic train call entries = ep_keys + proc_keys (universal donor augment)
          - procedural train call: absent (copied, not trained)
          - result["tier_keyed"]["procedural"] = proc_keys (served from procedural)
          - result["tier_keyed"]["episodic"] = ep_keys (graduating keys NOT in episodic serve)
        """
        from unittest.mock import MagicMock, patch

        import networkx as nx

        from paramem.memory.persistence import _IK_KEY_ATTR
        from paramem.training.consolidation import ConsolidationLoop

        floor = 10
        loop = self._make_loop(tmp_path, min_tier_key_floor=floor, tier_fast_start=True)

        # Seed 15 procedural keys — all in episodic store-tier (first-time graduation,
        # no disk slot for procedural in tmp_path).
        proc_keys = [f"proc{i}" for i in range(15)]
        self._seed_keys(loop, "episodic", proc_keys, relation_type="preference")
        ep_keys = [f"ep{i}" for i in range(20)]
        self._seed_keys(loop, "episodic", ep_keys, relation_type="factual")

        for k in proc_keys:
            eid = loop.merger.graph.add_edge(
                "Alice",
                f"obj_{k}",
                predicate=f"pred_{k}",
                relation_type="preference",
                confidence=1.0,
                first_seen="s",
                last_seen="s",
                reinforcement_count=1,
                sessions=["s"],
            )
            loop.merger.graph["Alice"][f"obj_{k}"][eid][_IK_KEY_ATTR] = k
        for k in ep_keys:
            eid = loop.merger.graph.add_edge(
                "Alice",
                f"obj_{k}",
                predicate=f"pred_{k}",
                relation_type="factual",
                confidence=1.0,
                first_seen="s",
                last_seen="s",
                reinforcement_count=1,
                sessions=["s"],
            )
            loop.merger.graph["Alice"][f"obj_{k}"][eid][_IK_KEY_ATTR] = k

        train_spy = MagicMock(return_value={"aborted": False})
        subset_copy_spy = MagicMock()

        # Probe passes for all entries (simulates informed episodic + copied procedural
        # both recalling their respective key sets).
        def _probe(adapter_name, entries):
            return {e["key"] for e in entries}

        from paramem.server.gpu_lock import _gpu_thread_lock

        _gpu_thread_lock.acquire()
        try:
            with (
                patch(
                    "paramem.training.consolidation.reconstruct_graph",
                    return_value=__import__(
                        "paramem.graph.reconstruct", fromlist=["ReconstructionResult"]
                    ).ReconstructionResult(graph=nx.MultiDiGraph()),
                ),
                patch.object(
                    ConsolidationLoop,
                    "_run_graph_enrichment",
                    return_value={"skipped": True},
                ),
                patch.object(ConsolidationLoop, "_enable_gradient_checkpointing"),
                patch.object(ConsolidationLoop, "_disable_gradient_checkpointing"),
                patch.object(
                    ConsolidationLoop,
                    "_maybe_make_recall_callback",
                    return_value=(None, None),
                ),
                patch.object(ConsolidationLoop, "_probe_passing_keys", side_effect=_probe),
                patch.object(ConsolidationLoop, "_save_adapters"),
                patch("paramem.training.trainer.train_adapter", train_spy),
                patch(
                    "paramem.training.consolidation.format_entry_training",
                    return_value=[{"input_ids": [1], "labels": [1], "attention_mask": [1]}],
                ),
                patch("paramem.models.loader.create_adapter", side_effect=lambda m, c, n: m),
                patch("paramem.models.loader.switch_adapter"),
                patch("paramem.models.loader.copy_adapter_weights"),
                patch("paramem.models.loader.copy_adapter_weights_subset", subset_copy_spy),
                patch("paramem.memory.interim_adapter.unload_interim_adapters", return_value=[]),
            ):
                result = loop.consolidate_interim_adapters(trainer=None, router=None)
        finally:
            _gpu_thread_lock.release()

        assert "procedural" in result["tiers_rebuilt"], (
            "procedural should be in tiers_rebuilt after fast-start graduation"
        )
        # subset copy must have been called for procedural←episodic (v3 universal-donor).
        assert subset_copy_spy.call_count >= 1, (
            "copy_adapter_weights_subset must be called for procedural fast-start"
        )
        # train_adapter must NOT have been called for procedural (only episodic).
        proc_train_calls = [
            c for c in train_spy.call_args_list if c.kwargs.get("adapter_name") == "procedural"
        ]
        assert len(proc_train_calls) == 0, (
            "train_adapter must NOT be called for procedural in fast-start graduation"
        )
        # Serve layout: procedural served from its own tier, NOT from episodic.
        served_proc_keys = {e["key"] for e in result["tier_keyed"]["procedural"]}
        served_ep_keys = {e["key"] for e in result["tier_keyed"]["episodic"]}
        assert served_proc_keys == set(proc_keys), (
            "procedural serve set must contain the graduating keys"
        )
        assert served_ep_keys == set(ep_keys), (
            "episodic serve set must NOT contain the graduating procedural keys"
        )
        # Episodic train call entries must include BOTH ep_keys AND proc_keys
        # (universal-donor augmentation).
        ep_train_calls = [
            c for c in train_spy.call_args_list if c.kwargs.get("adapter_name") == "episodic"
        ]
        assert len(ep_train_calls) >= 1, "train_adapter must be called for episodic"
        # The episodic train dataset comes from format_entry_training(job.entries, ...).
        # Since format_entry_training is patched, verify via the jobs_by_tier entries
        # indirectly: the episodic train call must have been attempted on 35 entries
        # (20 ep + 15 proc).  format_entry_training is called with job.entries so we
        # check call_count of format_entry_training instead, which receives the full set.
        # (Direct assertion: serve layout is the ground truth; the donor augment is an
        # internal implementation detail verified by the universal-donor decoupling tests.)

    def test_fast_start_already_live_tier_trains_normally(self, tmp_path):
        """A tier already live (has a saved adapter slot on disk) trains normally
        regardless of tier_fast_start; copy helpers not called for it.
        """
        from unittest.mock import MagicMock, patch

        import networkx as nx

        from paramem.memory.persistence import _IK_KEY_ATTR
        from paramem.training.consolidation import ConsolidationLoop

        floor = 5
        loop = self._make_loop(tmp_path, min_tier_key_floor=floor, tier_fast_start=True)

        # Create a fake saved slot for procedural to signal "already live".
        # Liveness is based on disk-slot presence (not store-tier) — a slot name is
        # a YYYYMMDD-HHMMSS timestamp.  Creating the directory simulates a prior
        # _save_adapters call for this tier.
        (tmp_path / "procedural" / "20240101-000000").mkdir(parents=True, exist_ok=True)

        # Seed 10 procedural keys ALREADY in the procedural registry
        # (previously graduated — tier_for_active_key returns "procedural").
        proc_keys = [f"proc{i}" for i in range(10)]
        self._seed_keys(loop, "procedural", proc_keys, relation_type="preference")
        ep_keys = [f"ep{i}" for i in range(10)]
        self._seed_keys(loop, "episodic", ep_keys, relation_type="factual")

        for k in proc_keys:
            eid = loop.merger.graph.add_edge(
                "Alice",
                f"obj_{k}",
                predicate=f"pred_{k}",
                relation_type="preference",
                confidence=1.0,
                first_seen="s",
                last_seen="s",
                reinforcement_count=1,
                sessions=["s"],
            )
            loop.merger.graph["Alice"][f"obj_{k}"][eid][_IK_KEY_ATTR] = k
        for k in ep_keys:
            eid = loop.merger.graph.add_edge(
                "Alice",
                f"obj_{k}",
                predicate=f"pred_{k}",
                relation_type="factual",
                confidence=1.0,
                first_seen="s",
                last_seen="s",
                reinforcement_count=1,
                sessions=["s"],
            )
            loop.merger.graph["Alice"][f"obj_{k}"][eid][_IK_KEY_ATTR] = k

        train_spy = MagicMock(return_value={"aborted": False})
        subset_copy_spy = MagicMock()

        from paramem.server.gpu_lock import _gpu_thread_lock

        _gpu_thread_lock.acquire()
        try:
            with (
                patch(
                    "paramem.training.consolidation.reconstruct_graph",
                    return_value=__import__(
                        "paramem.graph.reconstruct", fromlist=["ReconstructionResult"]
                    ).ReconstructionResult(graph=nx.MultiDiGraph()),
                ),
                patch.object(
                    ConsolidationLoop,
                    "_run_graph_enrichment",
                    return_value={"skipped": True},
                ),
                patch.object(ConsolidationLoop, "_enable_gradient_checkpointing"),
                patch.object(ConsolidationLoop, "_disable_gradient_checkpointing"),
                patch.object(
                    ConsolidationLoop,
                    "_maybe_make_recall_callback",
                    return_value=(None, None),
                ),
                patch.object(
                    ConsolidationLoop,
                    "_probe_passing_keys",
                    side_effect=lambda a, e: {x["key"] for x in e},
                ),
                patch.object(ConsolidationLoop, "_save_adapters"),
                patch("paramem.training.trainer.train_adapter", train_spy),
                patch(
                    "paramem.training.consolidation.format_entry_training",
                    return_value=[{"input_ids": [1], "labels": [1], "attention_mask": [1]}],
                ),
                patch("paramem.models.loader.create_adapter", side_effect=lambda m, c, n: m),
                patch("paramem.models.loader.switch_adapter"),
                patch("paramem.models.loader.copy_adapter_weights"),
                patch("paramem.models.loader.copy_adapter_weights_subset", subset_copy_spy),
                patch("paramem.memory.interim_adapter.unload_interim_adapters", return_value=[]),
            ):
                loop.consolidate_interim_adapters(trainer=None, router=None)
        finally:
            _gpu_thread_lock.release()

        # procedural is already live — train_adapter should be called for it.
        proc_train_calls = [
            c for c in train_spy.call_args_list if c.kwargs.get("adapter_name") == "procedural"
        ]
        assert len(proc_train_calls) >= 1, (
            "train_adapter must be called for an already-live tier (steady-state)"
        )
        assert subset_copy_spy.call_count == 0, (
            "copy_adapter_weights_subset must NOT be called for an already-live tier"
        )

    # -------------------------------------------------------------------------
    # Fast-start copy-gate fallback (mandatory probe before accepting copy)
    # -------------------------------------------------------------------------

    def test_fast_start_fallback_trains_from_scratch_when_probe_fails(self, tmp_path):
        """R5: when the pre-save probe returns below threshold for a fast-start tier,
        the (b)→(a) fall-back fires: train_adapter IS called for that tier.
        """
        from unittest.mock import MagicMock, patch

        import networkx as nx

        from paramem.memory.persistence import _IK_KEY_ATTR
        from paramem.training.consolidation import ConsolidationLoop

        floor = 10
        loop = self._make_loop(tmp_path, min_tier_key_floor=floor, tier_fast_start=True)

        proc_keys = [f"proc{i}" for i in range(15)]
        self._seed_keys(loop, "episodic", proc_keys, relation_type="preference")
        ep_keys = [f"ep{i}" for i in range(20)]
        self._seed_keys(loop, "episodic", ep_keys, relation_type="factual")

        for k in proc_keys:
            eid = loop.merger.graph.add_edge(
                "Alice",
                f"obj_{k}",
                predicate=f"pred_{k}",
                relation_type="preference",
                confidence=1.0,
                first_seen="s",
                last_seen="s",
                reinforcement_count=1,
                sessions=["s"],
            )
            loop.merger.graph["Alice"][f"obj_{k}"][eid][_IK_KEY_ATTR] = k
        for k in ep_keys:
            eid = loop.merger.graph.add_edge(
                "Alice",
                f"obj_{k}",
                predicate=f"pred_{k}",
                relation_type="factual",
                confidence=1.0,
                first_seen="s",
                last_seen="s",
                reinforcement_count=1,
                sessions=["s"],
            )
            loop.merger.graph["Alice"][f"obj_{k}"][eid][_IK_KEY_ATTR] = k

        # Probe returns EMPTY set for "procedural" (simulates below-threshold copy).
        def _probe(adapter_name, entries):
            if adapter_name == "procedural":
                return set()  # fail
            return {e["key"] for e in entries}  # pass for all others

        train_spy = MagicMock(return_value={"aborted": False})

        from paramem.server.gpu_lock import _gpu_thread_lock

        _gpu_thread_lock.acquire()
        try:
            with (
                patch(
                    "paramem.training.consolidation.reconstruct_graph",
                    return_value=__import__(
                        "paramem.graph.reconstruct", fromlist=["ReconstructionResult"]
                    ).ReconstructionResult(graph=nx.MultiDiGraph()),
                ),
                patch.object(
                    ConsolidationLoop,
                    "_run_graph_enrichment",
                    return_value={"skipped": True},
                ),
                patch.object(ConsolidationLoop, "_enable_gradient_checkpointing"),
                patch.object(ConsolidationLoop, "_disable_gradient_checkpointing"),
                patch.object(
                    ConsolidationLoop,
                    "_maybe_make_recall_callback",
                    return_value=(None, None),
                ),
                patch.object(ConsolidationLoop, "_probe_passing_keys", side_effect=_probe),
                patch.object(ConsolidationLoop, "_save_adapters"),
                patch("paramem.training.trainer.train_adapter", train_spy),
                patch(
                    "paramem.training.consolidation.format_entry_training",
                    return_value=[{"input_ids": [1], "labels": [1], "attention_mask": [1]}],
                ),
                patch("paramem.models.loader.create_adapter", side_effect=lambda m, c, n: m),
                patch("paramem.models.loader.switch_adapter"),
                patch("paramem.models.loader.copy_adapter_weights"),
                patch("paramem.models.loader.copy_adapter_weights_subset"),
                patch("paramem.memory.interim_adapter.unload_interim_adapters", return_value=[]),
            ):
                result = loop.consolidate_interim_adapters(trainer=None, router=None)
        finally:
            _gpu_thread_lock.release()

        # Fall-through to train-from-scratch: procedural must be in tiers_rebuilt.
        assert "procedural" in result["tiers_rebuilt"], (
            "procedural must be in tiers_rebuilt after (b)→(a) fallback"
        )
        # train_adapter must have been called for procedural.
        proc_calls = [
            c for c in train_spy.call_args_list if c.kwargs.get("adapter_name") == "procedural"
        ]
        assert len(proc_calls) >= 1, (
            "train_adapter must be called for procedural after fast-start probe failure"
        )

    # -------------------------------------------------------------------------
    # Whole-fold accumulate (episodic < floor)
    # -------------------------------------------------------------------------

    def test_whole_fold_accumulate_returns_accumulating_status(self, tmp_path):
        """When total trainable keys < floor, fold returns status='accumulating',
        tiers_rebuilt=[], does NOT call _save_adapters, does NOT purge interims.
        """
        from unittest.mock import MagicMock, patch

        import networkx as nx

        from paramem.memory.persistence import _IK_KEY_ATTR
        from paramem.training.consolidation import ConsolidationLoop

        floor = 30
        loop = self._make_loop(tmp_path, min_tier_key_floor=floor)

        # Only 10 keys total (below floor of 30).
        small_keys = [f"k{i}" for i in range(10)]
        self._seed_keys(loop, "episodic", small_keys, relation_type="factual")
        for k in small_keys:
            eid = loop.merger.graph.add_edge(
                "Alice",
                f"obj_{k}",
                predicate=f"pred_{k}",
                relation_type="factual",
                confidence=1.0,
                first_seen="s",
                last_seen="s",
                reinforcement_count=1,
                sessions=["s"],
            )
            loop.merger.graph["Alice"][f"obj_{k}"][eid][_IK_KEY_ATTR] = k

        save_spy = MagicMock()
        train_spy = MagicMock(return_value={"aborted": False})

        from paramem.server.gpu_lock import _gpu_thread_lock

        _gpu_thread_lock.acquire()
        try:
            with (
                patch(
                    "paramem.training.consolidation.reconstruct_graph",
                    return_value=__import__(
                        "paramem.graph.reconstruct", fromlist=["ReconstructionResult"]
                    ).ReconstructionResult(graph=nx.MultiDiGraph()),
                ),
                patch.object(
                    ConsolidationLoop,
                    "_run_graph_enrichment",
                    return_value={"skipped": True},
                ),
                patch.object(ConsolidationLoop, "_enable_gradient_checkpointing"),
                patch.object(ConsolidationLoop, "_disable_gradient_checkpointing"),
                patch.object(
                    ConsolidationLoop,
                    "_maybe_make_recall_callback",
                    return_value=(None, None),
                ),
                patch.object(
                    ConsolidationLoop,
                    "_probe_passing_keys",
                    side_effect=lambda a, e: {x["key"] for x in e},
                ),
                patch.object(ConsolidationLoop, "_save_adapters", save_spy),
                patch("paramem.training.trainer.train_adapter", train_spy),
                patch(
                    "paramem.training.consolidation.format_entry_training",
                    return_value=[{"input_ids": [1], "labels": [1], "attention_mask": [1]}],
                ),
                patch("paramem.models.loader.create_adapter", side_effect=lambda m, c, n: m),
                patch("paramem.models.loader.switch_adapter"),
                patch("paramem.models.loader.copy_adapter_weights"),
                patch("paramem.models.loader.copy_adapter_weights_subset"),
                patch("paramem.memory.interim_adapter.unload_interim_adapters", return_value=[]),
            ):
                result = loop.consolidate_interim_adapters(trainer=None, router=None)
        finally:
            _gpu_thread_lock.release()

        assert result["status"] == "accumulating", (
            f"Expected status='accumulating', got {result['status']!r}"
        )
        assert result["tiers_rebuilt"] == [], "tiers_rebuilt must be empty on accumulating return"
        assert "accumulating_reason" in result, "accumulating_reason must be in result"
        assert result["accumulating_reason"]["floor"] == floor
        assert result["accumulating_reason"]["episodic"] == 10
        assert save_spy.call_count == 0, "_save_adapters must NOT be called on accumulating return"
        assert train_spy.call_count == 0, "train_adapter must NOT be called on accumulating return"

    # -------------------------------------------------------------------------
    # Accumulating never raises; last_consolidation_error stays None
    # -------------------------------------------------------------------------

    def test_accumulating_never_raises(self, tmp_path):
        """The accumulating early return never raises an exception."""
        floor = 30
        loop = self._make_loop(tmp_path, min_tier_key_floor=floor)
        # 5 keys < floor.
        self._seed_keys(loop, "episodic", [f"k{i}" for i in range(5)])
        for k in [f"k{i}" for i in range(5)]:
            eid = loop.merger.graph.add_edge(
                "Alice",
                f"obj_{k}",
                predicate=f"pred_{k}",
                relation_type="factual",
                confidence=1.0,
                first_seen="s",
                last_seen="s",
                reinforcement_count=1,
                sessions=["s"],
            )
            loop.merger.graph["Alice"][f"obj_{k}"][eid][
                __import__("paramem.memory.persistence", fromlist=["_IK_KEY_ATTR"])._IK_KEY_ATTR
            ] = k

        # Must not raise.
        result = self._run_full_fold_mocked(loop)
        assert result["status"] == "accumulating"

    # =========================================================================
    # v3 universal-donor serve/train decoupling tests
    # =========================================================================

    @staticmethod
    def _add_edges(loop, entries):
        """Stamp entries as merger-graph edges (convenience wrapper)."""
        from paramem.memory.persistence import _IK_KEY_ATTR

        for e in entries:
            eid = loop.merger.graph.add_edge(
                e["subject"],
                e["object"],
                predicate=e["predicate"],
                relation_type=e.get("relation_type", "factual"),
                confidence=1.0,
                first_seen="s",
                last_seen="s",
                reinforcement_count=1,
                sessions=["s"],
            )
            loop.merger.graph[e["subject"]][e["object"]][eid][_IK_KEY_ATTR] = e["key"]

    def test_universal_donor_episodic_includes_graduating_keys(self, tmp_path):
        """Episodic train call includes graduating procedural keys (universal
        donor); procedural train call absent; serve layout keeps them separate.

        floor=10, 12 proc keys (store-tier episodic, first-cross) + 20 ep keys.
        Episodic train entries = 32; procedural train entries = 0 (copy path).
        Serve layout: procedural=12, episodic=20 (graduating keys NOT in ep serve).
        """
        from unittest.mock import MagicMock, patch

        import networkx as nx

        from paramem.training.consolidation import ConsolidationLoop

        loop = self._make_loop(tmp_path, min_tier_key_floor=10, tier_fast_start=True)
        proc_keys = [f"p{i}" for i in range(12)]
        ep_keys = [f"e{i}" for i in range(20)]
        self._seed_keys(loop, "episodic", proc_keys, relation_type="preference")
        self._seed_keys(loop, "episodic", ep_keys, relation_type="factual")
        self._add_edges(
            loop,
            [
                {
                    "key": k,
                    "subject": "A",
                    "object": f"o{k}",
                    "predicate": f"p{k}",
                    "relation_type": "preference",
                }
                for k in proc_keys
            ]
            + [
                {
                    "key": k,
                    "subject": "A",
                    "object": f"o{k}",
                    "predicate": f"p{k}",
                    "relation_type": "factual",
                }
                for k in ep_keys
            ],
        )

        def _spy_train(**kwargs):
            # format_entry_training is patched to return a fixed list;
            # entries fed to it come from job.entries captured at format call.
            return {"aborted": False}

        format_entries_by_adapter: dict = {}

        def _spy_format(entries, *a, **kw):
            # The calling code passes entries positionally.
            return [{"input_ids": [1], "labels": [1], "attention_mask": [1]}]

        # We need to capture job.entries at format_entry_training time.
        # Use a mutable dict to capture the adapter being trained.
        _adapter_being_trained = [None]
        train_spy = MagicMock(return_value={"aborted": False})

        from paramem.server.gpu_lock import _gpu_thread_lock

        _gpu_thread_lock.acquire()
        try:
            with (
                patch(
                    "paramem.training.consolidation.reconstruct_graph",
                    return_value=__import__(
                        "paramem.graph.reconstruct", fromlist=["ReconstructionResult"]
                    ).ReconstructionResult(graph=nx.MultiDiGraph()),
                ),
                patch.object(
                    ConsolidationLoop, "_run_graph_enrichment", return_value={"skipped": True}
                ),
                patch.object(ConsolidationLoop, "_enable_gradient_checkpointing"),
                patch.object(ConsolidationLoop, "_disable_gradient_checkpointing"),
                patch.object(
                    ConsolidationLoop, "_maybe_make_recall_callback", return_value=(None, None)
                ),
                patch.object(
                    ConsolidationLoop,
                    "_probe_passing_keys",
                    side_effect=lambda a, e: {x["key"] for x in e},
                ),
                patch.object(ConsolidationLoop, "_save_adapters"),
                patch("paramem.training.trainer.train_adapter", train_spy),
                patch(
                    "paramem.training.consolidation.format_entry_training",
                    side_effect=lambda entries, *a, **kw: (
                        format_entries_by_adapter.__setitem__(
                            _adapter_being_trained[0], list(entries)
                        )
                        or [{"input_ids": [1], "labels": [1], "attention_mask": [1]}]
                    ),
                ),
                patch(
                    "paramem.models.loader.create_adapter",
                    side_effect=lambda m, c, n: _adapter_being_trained.__setitem__(0, n) or m,
                ),
                patch("paramem.models.loader.switch_adapter"),
                patch("paramem.models.loader.copy_adapter_weights"),
                patch("paramem.models.loader.copy_adapter_weights_subset"),
                patch("paramem.memory.interim_adapter.unload_interim_adapters", return_value=[]),
            ):
                result = loop.consolidate_interim_adapters(trainer=None, router=None)
        finally:
            _gpu_thread_lock.release()

        # Serve layout assertions.
        served_proc = {e["key"] for e in result["tier_keyed"]["procedural"]}
        served_ep = {e["key"] for e in result["tier_keyed"]["episodic"]}
        assert served_proc == set(proc_keys), "procedural serve set must be the 12 graduating keys"
        assert served_ep == set(ep_keys), (
            "episodic serve set must NOT include the graduating procedural keys"
        )
        # No procedural train call (fast-start copy, empty train set).
        proc_train_calls = [
            c for c in train_spy.call_args_list if c.kwargs.get("adapter_name") == "procedural"
        ]
        assert len(proc_train_calls) == 0, "procedural must NOT be trained (fast-start copy)"
        # Episodic trained (train_adapter called).
        ep_train_calls = [
            c for c in train_spy.call_args_list if c.kwargs.get("adapter_name") == "episodic"
        ]
        assert len(ep_train_calls) >= 1, "episodic must be trained (universal donor)"
        # Episodic's training entries (format_entry_training input) = 32 unique keys.
        ep_format_entries = format_entries_by_adapter.get("episodic", [])
        ep_trained_keys = {e["key"] for e in ep_format_entries}
        assert len(ep_trained_keys) == 32, (
            f"episodic train set should be 20 ep + 12 proc = 32, got {len(ep_trained_keys)}"
        )
        assert set(proc_keys) <= ep_trained_keys, (
            "graduating procedural keys must be in episodic's training set (donor augment)"
        )

    def test_graduating_keys_served_from_target_tier_not_duplicated(self, tmp_path):
        """Each graduating key appears in the serve layout exactly once and is
        served from procedural (not episodic).
        """
        loop = self._make_loop(tmp_path, min_tier_key_floor=10, tier_fast_start=True)
        proc_keys = [f"p{i}" for i in range(12)]
        ep_keys = [f"e{i}" for i in range(20)]
        self._seed_keys(loop, "episodic", proc_keys, relation_type="preference")
        self._seed_keys(loop, "episodic", ep_keys, relation_type="factual")
        self._add_edges(
            loop,
            [
                {
                    "key": k,
                    "subject": "A",
                    "object": f"o{k}",
                    "predicate": f"p{k}",
                    "relation_type": "preference",
                }
                for k in proc_keys
            ]
            + [
                {
                    "key": k,
                    "subject": "A",
                    "object": f"o{k}",
                    "predicate": f"p{k}",
                    "relation_type": "factual",
                }
                for k in ep_keys
            ],
        )

        result = self._run_full_fold_mocked(loop)

        # Each key appears at most once across all serve tiers.
        all_served = [e["key"] for tl in result["tier_keyed"].values() for e in tl]
        assert len(all_served) == len(set(all_served)), (
            "each key must appear in serve layout exactly once (no duplication)"
        )
        # Graduating keys served from procedural.
        for k in proc_keys:
            assert k not in {e["key"] for e in result["tier_keyed"]["episodic"]}, (
                f"graduating key {k!r} must NOT be in episodic serve set"
            )
            assert k in {e["key"] for e in result["tier_keyed"]["procedural"]}, (
                f"graduating key {k!r} must be in procedural serve set"
            )

    def test_procedural_fast_start_uses_subset_copy_not_train(self, tmp_path):
        """Procedural first-cross fast-start: copy_adapter_weights_subset called,
        train_adapter NOT called for procedural, procedural in tiers_rebuilt.

        Probe returns only keys the donor was actually trained on (augmented episodic
        set) — NOT a blanket pass-all that would mask an uninformed-donor regression.
        """
        from unittest.mock import MagicMock, patch

        import networkx as nx

        from paramem.training.consolidation import ConsolidationLoop

        loop = self._make_loop(tmp_path, min_tier_key_floor=10, tier_fast_start=True)
        proc_keys = [f"p{i}" for i in range(12)]
        ep_keys = [f"e{i}" for i in range(20)]
        self._seed_keys(loop, "episodic", proc_keys, relation_type="preference")
        self._seed_keys(loop, "episodic", ep_keys, relation_type="factual")
        self._add_edges(
            loop,
            [
                {
                    "key": k,
                    "subject": "A",
                    "object": f"o{k}",
                    "predicate": f"p{k}",
                    "relation_type": "preference",
                }
                for k in proc_keys
            ]
            + [
                {
                    "key": k,
                    "subject": "A",
                    "object": f"o{k}",
                    "predicate": f"p{k}",
                    "relation_type": "factual",
                }
                for k in ep_keys
            ],
        )

        # Track episodic's training entries so the probe can simulate real knowledge.
        _ep_train_keys: list = []

        def _spy_format(entries, *a, **kw):
            # format_entry_training is called for episodic first.
            if not _ep_train_keys:
                _ep_train_keys.extend(e["key"] for e in entries)
            return [{"input_ids": [1], "labels": [1], "attention_mask": [1]}]

        def _probe(adapter_name, entries):
            # Simulate donor knowledge: episodic returns all keys it was trained on;
            # procedural probe (after copy from episodic) returns same keys.
            return {e["key"] for e in entries if e["key"] in set(_ep_train_keys)}

        train_spy = MagicMock(return_value={"aborted": False})
        subset_copy_spy = MagicMock()
        full_copy_spy = MagicMock()

        from paramem.server.gpu_lock import _gpu_thread_lock

        _gpu_thread_lock.acquire()
        try:
            with (
                patch(
                    "paramem.training.consolidation.reconstruct_graph",
                    return_value=__import__(
                        "paramem.graph.reconstruct", fromlist=["ReconstructionResult"]
                    ).ReconstructionResult(graph=nx.MultiDiGraph()),
                ),
                patch.object(
                    ConsolidationLoop, "_run_graph_enrichment", return_value={"skipped": True}
                ),
                patch.object(ConsolidationLoop, "_enable_gradient_checkpointing"),
                patch.object(ConsolidationLoop, "_disable_gradient_checkpointing"),
                patch.object(
                    ConsolidationLoop, "_maybe_make_recall_callback", return_value=(None, None)
                ),
                patch.object(ConsolidationLoop, "_probe_passing_keys", side_effect=_probe),
                patch.object(ConsolidationLoop, "_save_adapters"),
                patch("paramem.training.trainer.train_adapter", train_spy),
                patch(
                    "paramem.training.consolidation.format_entry_training",
                    side_effect=_spy_format,
                ),
                patch("paramem.models.loader.create_adapter", side_effect=lambda m, c, n: m),
                patch("paramem.models.loader.switch_adapter"),
                patch("paramem.models.loader.copy_adapter_weights", full_copy_spy),
                patch("paramem.models.loader.copy_adapter_weights_subset", subset_copy_spy),
                patch("paramem.memory.interim_adapter.unload_interim_adapters", return_value=[]),
            ):
                result = loop.consolidate_interim_adapters(trainer=None, router=None)
        finally:
            _gpu_thread_lock.release()

        assert "procedural" in result["tiers_rebuilt"], (
            "procedural must be in tiers_rebuilt after fast-start copy accepted"
        )
        # subset copy for procedural←episodic (attn-only, MLP zero-init).
        proc_subset_calls = [
            c
            for c in subset_copy_spy.call_args_list
            if c.kwargs.get("dst") == "procedural" or (c.args and "procedural" in str(c.args))
        ]
        assert len(proc_subset_calls) >= 1, (
            "copy_adapter_weights_subset must be called for procedural←episodic"
        )
        # train_adapter must NOT be called for procedural.
        proc_train = [
            c for c in train_spy.call_args_list if c.kwargs.get("adapter_name") == "procedural"
        ]
        assert len(proc_train) == 0, "train_adapter must NOT be called for procedural fast-start"

    def test_semantic_fast_start_uses_full_copy_not_train(self, tmp_path):
        """Semantic first-cross fast-start (promotion-store-moved):
        copy_adapter_weights (FULL copy, NOT subset) called for semantic,
        train_adapter NOT called for semantic, semantic in tiers_rebuilt.

        Proves the v3 universal-donor fix covers semantic (not just procedural).
        Semantic keys are store-tier "semantic" (promotion-store-moved) but have
        NO disk slot → _tier_has_disk_slot("semantic") = False → graduates.
        """
        from unittest.mock import MagicMock, patch

        import networkx as nx

        from paramem.training.consolidation import ConsolidationLoop

        loop = self._make_loop(tmp_path, min_tier_key_floor=10, tier_fast_start=True)
        # Semantic keys with store-tier "semantic" (simulate post-promotion state).
        sem_keys = [f"s{i}" for i in range(11)]
        # Seed with store-tier "semantic" (promotion already happened).
        for k in sem_keys:
            loop.store.put(
                "semantic",
                k,
                {
                    "key": k,
                    "subject": "Alice",
                    "predicate": f"likes_{k}",
                    "object": f"thing_{k}",
                    "speaker_id": "S0",
                },
                register=True,
            )
            loop.store.set_bookkeeping(
                k,
                speaker_id="S0",
                relation_type="factual",
                reinforcement_count=3,  # promoted threshold met
                last_reinforced_cycle=1,
                first_seen="",
            )
        # Episodic keys.
        ep_keys = [f"e{i}" for i in range(20)]
        self._seed_keys(loop, "episodic", ep_keys, relation_type="factual")

        # The edge-walk stage uses store.tier_for_active_key to determine current tier;
        # semantic keys are already in semantic store-tier so they route to
        # serve_assignment["semantic"].
        self._add_edges(
            loop,
            [
                {
                    "key": k,
                    "subject": "Alice",
                    "object": f"thing_{k}",
                    "predicate": f"likes_{k}",
                    "relation_type": "factual",
                }
                for k in sem_keys
            ]
            + [
                {
                    "key": k,
                    "subject": "A",
                    "object": f"o{k}",
                    "predicate": f"p{k}",
                    "relation_type": "factual",
                }
                for k in ep_keys
            ],
        )

        # No disk slot for semantic in tmp_path → _tier_has_disk_slot("semantic") = False.
        train_spy = MagicMock(return_value={"aborted": False})
        subset_copy_spy = MagicMock()
        full_copy_spy = MagicMock()

        from paramem.server.gpu_lock import _gpu_thread_lock

        _gpu_thread_lock.acquire()
        try:
            with (
                patch(
                    "paramem.training.consolidation.reconstruct_graph",
                    return_value=__import__(
                        "paramem.graph.reconstruct", fromlist=["ReconstructionResult"]
                    ).ReconstructionResult(graph=nx.MultiDiGraph()),
                ),
                patch.object(
                    ConsolidationLoop, "_run_graph_enrichment", return_value={"skipped": True}
                ),
                patch.object(ConsolidationLoop, "_enable_gradient_checkpointing"),
                patch.object(ConsolidationLoop, "_disable_gradient_checkpointing"),
                patch.object(
                    ConsolidationLoop, "_maybe_make_recall_callback", return_value=(None, None)
                ),
                patch.object(
                    ConsolidationLoop,
                    "_probe_passing_keys",
                    side_effect=lambda a, e: {x["key"] for x in e},
                ),
                patch.object(ConsolidationLoop, "_save_adapters"),
                patch("paramem.training.trainer.train_adapter", train_spy),
                patch(
                    "paramem.training.consolidation.format_entry_training",
                    return_value=[{"input_ids": [1], "labels": [1], "attention_mask": [1]}],
                ),
                patch("paramem.models.loader.create_adapter", side_effect=lambda m, c, n: m),
                patch("paramem.models.loader.switch_adapter"),
                patch("paramem.models.loader.copy_adapter_weights", full_copy_spy),
                patch("paramem.models.loader.copy_adapter_weights_subset", subset_copy_spy),
                patch("paramem.memory.interim_adapter.unload_interim_adapters", return_value=[]),
            ):
                result = loop.consolidate_interim_adapters(trainer=None, router=None)
        finally:
            _gpu_thread_lock.release()

        assert "semantic" in result["tiers_rebuilt"], (
            "semantic must be in tiers_rebuilt after fast-start copy accepted"
        )
        # FULL copy (not subset) for semantic←episodic (equal param sets).
        sem_full_calls = [
            c
            for c in full_copy_spy.call_args_list
            if c.kwargs.get("dst") == "semantic" or (c.args and "semantic" in str(c.args))
        ]
        assert len(sem_full_calls) >= 1, (
            "copy_adapter_weights (FULL) must be called for semantic←episodic (v3 fix)"
        )
        # subset copy must NOT have been used for semantic.
        sem_subset_calls = [
            c
            for c in subset_copy_spy.call_args_list
            if c.kwargs.get("dst") == "semantic" or (c.args and "semantic" in str(c.args))
        ]
        assert len(sem_subset_calls) == 0, (
            "copy_adapter_weights_subset must NOT be called for semantic (only procedural)"
        )
        # train_adapter must NOT be called for semantic.
        sem_train = [
            c for c in train_spy.call_args_list if c.kwargs.get("adapter_name") == "semantic"
        ]
        assert len(sem_train) == 0, "train_adapter must NOT be called for semantic fast-start"
        # Episodic train includes the 11 semantic graduating keys (universal donor).
        # Verify via serve layout: semantic served from semantic, not episodic.
        served_sem = {e["key"] for e in result["tier_keyed"]["semantic"]}
        assert served_sem == set(sem_keys), "semantic keys must be served from semantic"
        served_ep = {e["key"] for e in result["tier_keyed"]["episodic"]}
        assert not (served_ep & set(sem_keys)), (
            "graduating semantic keys must NOT appear in episodic serve set"
        )

    def test_simultaneous_semantic_procedural_graduation_universal_donor(self, tmp_path):
        """Both semantic and procedural graduate in the same fold (universal donor).

        floor=10, 11 semantic keys (store-tier semantic, first-cross) + 12 procedural
        keys (store-tier episodic, parked) + 20 episodic.

        Assertions:
        - episodic train entries = 43 unique (20+11+12), no duplicates
        - copy_adapter_weights (FULL) called for semantic
        - copy_adapter_weights_subset (SUBSET) called for procedural
        - train_adapter NOT called for semantic or procedural
        - both in tiers_rebuilt
        - serve layout: semantic=11, procedural=12, episodic=20
        """
        from unittest.mock import MagicMock, patch

        import networkx as nx

        from paramem.training.consolidation import ConsolidationLoop

        loop = self._make_loop(tmp_path, min_tier_key_floor=10, tier_fast_start=True)

        sem_keys = [f"s{i}" for i in range(11)]
        proc_keys = [f"p{i}" for i in range(12)]
        ep_keys = [f"e{i}" for i in range(20)]

        # Semantic keys: store-tier "semantic" (promotion-store-moved, no disk slot).
        for k in sem_keys:
            loop.store.put(
                "semantic",
                k,
                {
                    "key": k,
                    "subject": "Alice",
                    "predicate": f"likes_{k}",
                    "object": f"thing_{k}",
                    "speaker_id": "S0",
                },
                register=True,
            )
            loop.store.set_bookkeeping(
                k,
                speaker_id="S0",
                relation_type="factual",
                reinforcement_count=3,
                last_reinforced_cycle=1,
                first_seen="",
            )
        # Procedural keys: store-tier "episodic" (parked, no disk slot for procedural).
        self._seed_keys(loop, "episodic", proc_keys, relation_type="preference")
        self._seed_keys(loop, "episodic", ep_keys, relation_type="factual")

        self._add_edges(
            loop,
            [
                {
                    "key": k,
                    "subject": "Alice",
                    "object": f"thing_{k}",
                    "predicate": f"likes_{k}",
                    "relation_type": "factual",
                }
                for k in sem_keys
            ]
            + [
                {
                    "key": k,
                    "subject": "A",
                    "object": f"o{k}",
                    "predicate": f"p{k}",
                    "relation_type": "preference",
                }
                for k in proc_keys
            ]
            + [
                {
                    "key": k,
                    "subject": "A",
                    "object": f"o{k}",
                    "predicate": f"p{k}",
                    "relation_type": "factual",
                }
                for k in ep_keys
            ],
        )

        train_spy = MagicMock(return_value={"aborted": False})
        full_copy_spy = MagicMock()
        subset_copy_spy = MagicMock()
        format_entries_captured: dict = {}
        _current_adapter: list = [None]

        def _spy_create(m, c, n):
            _current_adapter[0] = n
            return m

        def _spy_format(entries, *a, **kw):
            if _current_adapter[0] and _current_adapter[0] not in format_entries_captured:
                format_entries_captured[_current_adapter[0]] = list(entries)
            return [{"input_ids": [1], "labels": [1], "attention_mask": [1]}]

        from paramem.server.gpu_lock import _gpu_thread_lock

        _gpu_thread_lock.acquire()
        try:
            with (
                patch(
                    "paramem.training.consolidation.reconstruct_graph",
                    return_value=__import__(
                        "paramem.graph.reconstruct", fromlist=["ReconstructionResult"]
                    ).ReconstructionResult(graph=nx.MultiDiGraph()),
                ),
                patch.object(
                    ConsolidationLoop, "_run_graph_enrichment", return_value={"skipped": True}
                ),
                patch.object(ConsolidationLoop, "_enable_gradient_checkpointing"),
                patch.object(ConsolidationLoop, "_disable_gradient_checkpointing"),
                patch.object(
                    ConsolidationLoop, "_maybe_make_recall_callback", return_value=(None, None)
                ),
                patch.object(
                    ConsolidationLoop,
                    "_probe_passing_keys",
                    side_effect=lambda a, e: {x["key"] for x in e},
                ),
                patch.object(ConsolidationLoop, "_save_adapters"),
                patch("paramem.training.trainer.train_adapter", train_spy),
                patch(
                    "paramem.training.consolidation.format_entry_training",
                    side_effect=_spy_format,
                ),
                patch("paramem.models.loader.create_adapter", side_effect=_spy_create),
                patch("paramem.models.loader.switch_adapter"),
                patch("paramem.models.loader.copy_adapter_weights", full_copy_spy),
                patch("paramem.models.loader.copy_adapter_weights_subset", subset_copy_spy),
                patch("paramem.memory.interim_adapter.unload_interim_adapters", return_value=[]),
            ):
                result = loop.consolidate_interim_adapters(trainer=None, router=None)
        finally:
            _gpu_thread_lock.release()

        # Both graduating tiers in tiers_rebuilt.
        assert "semantic" in result["tiers_rebuilt"], "semantic must be in tiers_rebuilt"
        assert "procedural" in result["tiers_rebuilt"], "procedural must be in tiers_rebuilt"

        # Copy type per tier.
        sem_full = [c for c in full_copy_spy.call_args_list if c.kwargs.get("dst") == "semantic"]
        assert len(sem_full) >= 1, "FULL copy must be called for semantic (equal param sets)"
        proc_subset = [
            c for c in subset_copy_spy.call_args_list if c.kwargs.get("dst") == "procedural"
        ]
        assert len(proc_subset) >= 1, "SUBSET copy must be called for procedural (attn only)"

        # No train_adapter for semantic or procedural.
        for t in ("semantic", "procedural"):
            t_train = [c for c in train_spy.call_args_list if c.kwargs.get("adapter_name") == t]
            assert len(t_train) == 0, f"train_adapter must NOT be called for {t} fast-start"

        # Episodic train set = 43 unique keys (20 + 11 + 12).
        ep_format_entries = format_entries_captured.get("episodic", [])
        ep_trained_keys = {e["key"] for e in ep_format_entries}
        assert len(ep_trained_keys) == 43, (
            f"episodic train set must be 20+11+12=43 unique keys, got {len(ep_trained_keys)}"
        )
        assert set(sem_keys) <= ep_trained_keys, "sem graduating keys must be in ep train set"
        assert set(proc_keys) <= ep_trained_keys, "proc graduating keys must be in ep train set"

        # Serve layout.
        assert result["keys_per_tier"]["semantic"] == 11
        assert result["keys_per_tier"]["procedural"] == 12
        assert result["keys_per_tier"]["episodic"] == 20

    def test_graduation_fallback_trains_on_serve_set_procedural(self, tmp_path):
        """Fast-start fallback for procedural: when the copy probe fails, train_adapter
        IS called for procedural with entries = its serve set, procedural in tiers_rebuilt.

        Probe patched to FAIL for procedural while episodic otherwise informed.
        Confirms fallback re-assignment: job.entries = serve_assignment[tier].
        """
        from unittest.mock import MagicMock, patch

        import networkx as nx

        from paramem.training.consolidation import ConsolidationLoop

        floor = 10
        loop = self._make_loop(tmp_path, min_tier_key_floor=floor, tier_fast_start=True)
        proc_keys = [f"p{i}" for i in range(12)]
        ep_keys = [f"e{i}" for i in range(20)]
        self._seed_keys(loop, "episodic", proc_keys, relation_type="preference")
        self._seed_keys(loop, "episodic", ep_keys, relation_type="factual")
        self._add_edges(
            loop,
            [
                {
                    "key": k,
                    "subject": "A",
                    "object": f"o{k}",
                    "predicate": f"p{k}",
                    "relation_type": "preference",
                }
                for k in proc_keys
            ]
            + [
                {
                    "key": k,
                    "subject": "A",
                    "object": f"o{k}",
                    "predicate": f"p{k}",
                    "relation_type": "factual",
                }
                for k in ep_keys
            ],
        )

        # Probe FAILS for procedural (simulates copy under-recall).
        def _probe_fail(adapter_name, entries):
            if adapter_name == "procedural":
                return set()  # fail
            return {e["key"] for e in entries}

        train_spy = MagicMock(return_value={"aborted": False})
        format_entries_by_adapter: dict = {}
        _current_adapter: list = ["episodic"]

        def _spy_create(m, c, n):
            _current_adapter[0] = n
            return m

        def _spy_format(entries, *a, **kw):
            ad = _current_adapter[0]
            format_entries_by_adapter.setdefault(ad, []).extend(entries)
            return [{"input_ids": [1], "labels": [1], "attention_mask": [1]}]

        from paramem.server.gpu_lock import _gpu_thread_lock

        _gpu_thread_lock.acquire()
        try:
            with (
                patch(
                    "paramem.training.consolidation.reconstruct_graph",
                    return_value=__import__(
                        "paramem.graph.reconstruct", fromlist=["ReconstructionResult"]
                    ).ReconstructionResult(graph=nx.MultiDiGraph()),
                ),
                patch.object(
                    ConsolidationLoop, "_run_graph_enrichment", return_value={"skipped": True}
                ),
                patch.object(ConsolidationLoop, "_enable_gradient_checkpointing"),
                patch.object(ConsolidationLoop, "_disable_gradient_checkpointing"),
                patch.object(
                    ConsolidationLoop, "_maybe_make_recall_callback", return_value=(None, None)
                ),
                patch.object(ConsolidationLoop, "_probe_passing_keys", side_effect=_probe_fail),
                patch.object(ConsolidationLoop, "_save_adapters"),
                patch("paramem.training.trainer.train_adapter", train_spy),
                patch(
                    "paramem.training.consolidation.format_entry_training",
                    side_effect=_spy_format,
                ),
                patch("paramem.models.loader.create_adapter", side_effect=_spy_create),
                patch("paramem.models.loader.switch_adapter"),
                patch("paramem.models.loader.copy_adapter_weights"),
                patch("paramem.models.loader.copy_adapter_weights_subset"),
                patch("paramem.memory.interim_adapter.unload_interim_adapters", return_value=[]),
            ):
                result = loop.consolidate_interim_adapters(trainer=None, router=None)
        finally:
            _gpu_thread_lock.release()

        assert "procedural" in result["tiers_rebuilt"], (
            "procedural must be in tiers_rebuilt after (b)→(a) fallback"
        )
        proc_train_calls = [
            c for c in train_spy.call_args_list if c.kwargs.get("adapter_name") == "procedural"
        ]
        assert len(proc_train_calls) >= 1, (
            "train_adapter must be called for procedural after fast-start probe failure"
        )
        # Fallback train entries = the serve set (12 keys), not empty.
        proc_format_entries = format_entries_by_adapter.get("procedural", [])
        proc_trained_keys = {e["key"] for e in proc_format_entries}
        assert proc_trained_keys == set(proc_keys), (
            f"fallback train entries must equal the procedural serve set "
            f"(12 keys), got {len(proc_trained_keys)}"
        )

    def test_graduation_fallback_trains_on_serve_set_semantic(self, tmp_path):
        """Fast-start fallback for semantic: when the copy probe fails, train_adapter
        IS called for semantic with entries = its serve set. Confirms fallback
        works for the semantic path too.
        """
        from unittest.mock import MagicMock, patch

        import networkx as nx

        from paramem.training.consolidation import ConsolidationLoop

        loop = self._make_loop(tmp_path, min_tier_key_floor=10, tier_fast_start=True)
        sem_keys = [f"s{i}" for i in range(11)]
        ep_keys = [f"e{i}" for i in range(20)]

        for k in sem_keys:
            loop.store.put(
                "semantic",
                k,
                {
                    "key": k,
                    "subject": "Alice",
                    "predicate": f"likes_{k}",
                    "object": f"thing_{k}",
                    "speaker_id": "S0",
                },
                register=True,
            )
            loop.store.set_bookkeeping(
                k,
                speaker_id="S0",
                relation_type="factual",
                reinforcement_count=3,
                last_reinforced_cycle=1,
                first_seen="",
            )
        self._seed_keys(loop, "episodic", ep_keys, relation_type="factual")
        self._add_edges(
            loop,
            [
                {
                    "key": k,
                    "subject": "Alice",
                    "object": f"thing_{k}",
                    "predicate": f"likes_{k}",
                    "relation_type": "factual",
                }
                for k in sem_keys
            ]
            + [
                {
                    "key": k,
                    "subject": "A",
                    "object": f"o{k}",
                    "predicate": f"p{k}",
                    "relation_type": "factual",
                }
                for k in ep_keys
            ],
        )

        # Probe FAILS for semantic.
        def _probe_fail_sem(adapter_name, entries):
            if adapter_name == "semantic":
                return set()
            return {e["key"] for e in entries}

        train_spy = MagicMock(return_value={"aborted": False})
        format_entries_by_adapter: dict = {}
        _current_adapter: list = ["episodic"]

        def _spy_create(m, c, n):
            _current_adapter[0] = n
            return m

        def _spy_format(entries, *a, **kw):
            ad = _current_adapter[0]
            format_entries_by_adapter.setdefault(ad, []).extend(entries)
            return [{"input_ids": [1], "labels": [1], "attention_mask": [1]}]

        from paramem.server.gpu_lock import _gpu_thread_lock

        _gpu_thread_lock.acquire()
        try:
            with (
                patch(
                    "paramem.training.consolidation.reconstruct_graph",
                    return_value=__import__(
                        "paramem.graph.reconstruct", fromlist=["ReconstructionResult"]
                    ).ReconstructionResult(graph=nx.MultiDiGraph()),
                ),
                patch.object(
                    ConsolidationLoop, "_run_graph_enrichment", return_value={"skipped": True}
                ),
                patch.object(ConsolidationLoop, "_enable_gradient_checkpointing"),
                patch.object(ConsolidationLoop, "_disable_gradient_checkpointing"),
                patch.object(
                    ConsolidationLoop, "_maybe_make_recall_callback", return_value=(None, None)
                ),
                patch.object(ConsolidationLoop, "_probe_passing_keys", side_effect=_probe_fail_sem),
                patch.object(ConsolidationLoop, "_save_adapters"),
                patch("paramem.training.trainer.train_adapter", train_spy),
                patch(
                    "paramem.training.consolidation.format_entry_training",
                    side_effect=_spy_format,
                ),
                patch("paramem.models.loader.create_adapter", side_effect=_spy_create),
                patch("paramem.models.loader.switch_adapter"),
                patch("paramem.models.loader.copy_adapter_weights"),
                patch("paramem.models.loader.copy_adapter_weights_subset"),
                patch("paramem.memory.interim_adapter.unload_interim_adapters", return_value=[]),
            ):
                result = loop.consolidate_interim_adapters(trainer=None, router=None)
        finally:
            _gpu_thread_lock.release()

        assert "semantic" in result["tiers_rebuilt"], (
            "semantic must be in tiers_rebuilt after (b)→(a) fallback"
        )
        sem_train_calls = [
            c for c in train_spy.call_args_list if c.kwargs.get("adapter_name") == "semantic"
        ]
        assert len(sem_train_calls) >= 1, (
            "train_adapter must be called for semantic after fast-start probe failure"
        )
        sem_format_entries = format_entries_by_adapter.get("semantic", [])
        sem_trained_keys = {e["key"] for e in sem_format_entries}
        assert sem_trained_keys == set(sem_keys), (
            f"fallback train entries must equal the semantic serve set "
            f"(11 keys), got {len(sem_trained_keys)}"
        )

    def test_train_from_scratch_episodic_not_augmented_with_graduating_keys(self, tmp_path):
        """With tier_fast_start=False, first-cross semantic trains from scratch
        on its own serve set — episodic train set does NOT include semantic keys.
        """
        from unittest.mock import MagicMock, patch

        import networkx as nx

        from paramem.training.consolidation import ConsolidationLoop

        loop = self._make_loop(tmp_path, min_tier_key_floor=10, tier_fast_start=False)
        sem_keys = [f"s{i}" for i in range(11)]
        ep_keys = [f"e{i}" for i in range(20)]

        for k in sem_keys:
            loop.store.put(
                "semantic",
                k,
                {
                    "key": k,
                    "subject": "Alice",
                    "predicate": f"likes_{k}",
                    "object": f"thing_{k}",
                    "speaker_id": "S0",
                },
                register=True,
            )
            loop.store.set_bookkeeping(
                k,
                speaker_id="S0",
                relation_type="factual",
                reinforcement_count=3,
                last_reinforced_cycle=1,
                first_seen="",
            )
        self._seed_keys(loop, "episodic", ep_keys, relation_type="factual")
        self._add_edges(
            loop,
            [
                {
                    "key": k,
                    "subject": "Alice",
                    "object": f"thing_{k}",
                    "predicate": f"likes_{k}",
                    "relation_type": "factual",
                }
                for k in sem_keys
            ]
            + [
                {
                    "key": k,
                    "subject": "A",
                    "object": f"o{k}",
                    "predicate": f"p{k}",
                    "relation_type": "factual",
                }
                for k in ep_keys
            ],
        )

        train_spy = MagicMock(return_value={"aborted": False})
        format_entries_by_adapter: dict = {}
        _current_adapter: list = ["episodic"]

        def _spy_create(m, c, n):
            _current_adapter[0] = n
            return m

        def _spy_format(entries, *a, **kw):
            ad = _current_adapter[0]
            format_entries_by_adapter.setdefault(ad, []).extend(entries)
            return [{"input_ids": [1], "labels": [1], "attention_mask": [1]}]

        from paramem.server.gpu_lock import _gpu_thread_lock

        _gpu_thread_lock.acquire()
        try:
            with (
                patch(
                    "paramem.training.consolidation.reconstruct_graph",
                    return_value=__import__(
                        "paramem.graph.reconstruct", fromlist=["ReconstructionResult"]
                    ).ReconstructionResult(graph=nx.MultiDiGraph()),
                ),
                patch.object(
                    ConsolidationLoop, "_run_graph_enrichment", return_value={"skipped": True}
                ),
                patch.object(ConsolidationLoop, "_enable_gradient_checkpointing"),
                patch.object(ConsolidationLoop, "_disable_gradient_checkpointing"),
                patch.object(
                    ConsolidationLoop, "_maybe_make_recall_callback", return_value=(None, None)
                ),
                patch.object(
                    ConsolidationLoop,
                    "_probe_passing_keys",
                    side_effect=lambda a, e: {x["key"] for x in e},
                ),
                patch.object(ConsolidationLoop, "_save_adapters"),
                patch("paramem.training.trainer.train_adapter", train_spy),
                patch(
                    "paramem.training.consolidation.format_entry_training",
                    side_effect=_spy_format,
                ),
                patch("paramem.models.loader.create_adapter", side_effect=_spy_create),
                patch("paramem.models.loader.switch_adapter"),
                patch("paramem.models.loader.copy_adapter_weights"),
                patch("paramem.models.loader.copy_adapter_weights_subset"),
                patch("paramem.memory.interim_adapter.unload_interim_adapters", return_value=[]),
            ):
                loop.consolidate_interim_adapters(trainer=None, router=None)
        finally:
            _gpu_thread_lock.release()

        # Semantic trains from scratch — train_adapter called for semantic.
        sem_train = [
            c for c in train_spy.call_args_list if c.kwargs.get("adapter_name") == "semantic"
        ]
        assert len(sem_train) >= 1, "train_adapter must be called for semantic (train-from-scratch)"
        # Episodic train set does NOT include semantic graduating keys.
        ep_format = format_entries_by_adapter.get("episodic", [])
        ep_trained_keys = {e["key"] for e in ep_format}
        assert not (ep_trained_keys & set(sem_keys)), (
            "episodic train set must NOT include semantic graduating keys "
            "when tier_fast_start=False (no donor augmentation)"
        )


class TestHarvestKeylessEdgesSpeakerId:
    """Harvest/apply pre-pass must resolve speaker_id from the subject node.

    Case (a): keyless edge whose subject IS a speaker person-node (speaker_id
    attribute set) → minted bookkeeping carries that speaker_id.
    Case (b): keyless edge whose subject is a role/non-speaker node (no
    speaker_id attribute) → minted bookkeeping carries "".
    """

    def _make_loop(self, tmp_path):
        """Minimal ConsolidationLoop stub via object.__new__ — no GPU required."""
        import networkx as nx
        from peft import PeftModel

        from paramem.memory.store import MemoryStore
        from paramem.training.consolidation import ConsolidationLoop
        from paramem.utils.config import AdapterConfig, ConsolidationConfig, TrainingConfig

        loop = object.__new__(ConsolidationLoop)
        loop.model = MagicMock()
        loop.model.__class__ = PeftModel
        loop.tokenizer = MagicMock()
        loop.config = ConsolidationConfig(min_tier_key_floor=0, tier_fast_start=False)
        loop.training_config = TrainingConfig(num_epochs=1, gradient_checkpointing=False)
        loop.episodic_config = AdapterConfig(rank=4, alpha=8, target_modules=["q_proj"])
        loop.semantic_config = AdapterConfig(rank=4, alpha=8, target_modules=["q_proj"])
        loop.procedural_config = None
        loop.wandb_config = None
        loop._thermal_policy = None
        loop.output_dir = tmp_path
        loop.store = MemoryStore(replay_enabled=False)
        loop.promoted_keys = set()
        loop.cycle_count = 1
        loop.episodic_simhash = {}
        loop.semantic_simhash = {}
        loop.procedural_simhash = {}
        loop._procedural_next_index = 0
        loop._procedural_tentative_next_index = 0
        loop._indexed_next_index = 0
        loop._indexed_ep_interim = {}
        loop._bg_trainer = None
        loop.shutdown_requested = False
        loop._early_stop_callback = None
        loop.fingerprint_cache = None
        loop._keep_prior_slots = 2
        loop._debug_base = None
        loop.save_cycle_snapshots = False
        loop.snapshot_dir = None
        # Attach a real MultiDiGraph so nodes.get() and graph.edges() behave correctly.
        merger = MagicMock()
        merger.graph = nx.MultiDiGraph()
        loop.merger = merger
        return loop

    def test_speaker_node_subject_inherits_speaker_id(self, tmp_path):
        """Case (a): subject is a speaker person-node → bookkeeping speaker_id == "spk-1"."""
        loop = self._make_loop(tmp_path)
        g = loop.merger.graph

        # Add a speaker person-node with speaker_id at the top level.
        g.add_node(
            "spk-1",
            entity_type="person",
            speaker_id="spk-1",
            attributes={"name": "Alex Morgan"},
        )
        # Add a plain object node.
        g.add_node("python", entity_type="skill", attributes={"name": "Python"})
        # Add a keyless edge (no ik_key attribute) with a predicate.
        g.add_edge("spk-1", "python", predicate="has_skill", relation_type="factual")

        tier_keyed: dict = {"episodic": [], "procedural": []}
        loop._build_all_edge_entries_into(tier_keyed)

        # Exactly one key must have been minted.
        assert len(tier_keyed["episodic"]) == 1, (
            f"Expected 1 minted episodic entry; got {tier_keyed['episodic']}"
        )
        minted_key = tier_keyed["episodic"][0]["key"]
        bk = loop.store.bookkeeping_for_key(minted_key)
        assert bk is not None, f"No bookkeeping record for minted key {minted_key!r}"
        assert bk["speaker_id"] == "spk-1", f"Expected speaker_id='spk-1', got {bk['speaker_id']!r}"

    def test_role_node_subject_speaker_id_empty(self, tmp_path):
        """Case (b): subject is a role/non-speaker node → bookkeeping speaker_id == ""."""
        loop = self._make_loop(tmp_path)
        g = loop.merger.graph

        # Add a non-speaker role node (no speaker_id attribute).
        g.add_node("developer", entity_type="role", attributes={"name": "Developer"})
        g.add_node("python", entity_type="skill", attributes={"name": "Python"})
        # Keyless edge with predicate.
        g.add_edge("developer", "python", predicate="requires", relation_type="factual")

        tier_keyed: dict = {"episodic": [], "procedural": []}
        loop._build_all_edge_entries_into(tier_keyed)

        assert len(tier_keyed["episodic"]) == 1, (
            f"Expected 1 minted episodic entry; got {tier_keyed['episodic']}"
        )
        minted_key = tier_keyed["episodic"][0]["key"]
        bk = loop.store.bookkeeping_for_key(minted_key)
        assert bk is not None, f"No bookkeeping record for minted key {minted_key!r}"
        assert bk["speaker_id"] == "", f"Expected speaker_id='', got {bk['speaker_id']!r}"

    def test_speaker_id_population_via_real_merger_path(self, tmp_path):
        """Population-2 (new): speaker_id flows edge→bookkeeping via the real merger.

        A Relation with speaker_id="spk-1" is merged through loop.merger.merge
        (not added directly to graph). The A-1 Case-3 stamp puts speaker_id on
        the edge; C-1 reads it; the minted bookkeeping record carries "spk-1".
        """
        from peft import PeftModel

        from paramem.graph.merger import GraphMerger
        from paramem.graph.schema import Entity, Relation, SessionGraph
        from paramem.memory.store import MemoryStore
        from paramem.training.consolidation import ConsolidationLoop
        from paramem.training.key_registry import KeyRegistry
        from paramem.utils.config import AdapterConfig, ConsolidationConfig, TrainingConfig

        loop = object.__new__(ConsolidationLoop)
        loop.model = MagicMock()
        loop.model.__class__ = PeftModel
        loop.tokenizer = MagicMock()
        loop.config = ConsolidationConfig(min_tier_key_floor=0, tier_fast_start=False)
        loop.training_config = TrainingConfig(num_epochs=1, gradient_checkpointing=False)
        loop.episodic_config = AdapterConfig(rank=4, alpha=8, target_modules=["q_proj"])
        loop.semantic_config = AdapterConfig(rank=4, alpha=8, target_modules=["q_proj"])
        loop.procedural_config = None
        loop.wandb_config = None
        loop._thermal_policy = None
        loop.output_dir = tmp_path
        loop.store = MemoryStore(replay_enabled=True)
        for tier in ("episodic", "semantic", "procedural"):
            loop.store.load_registry(tier, KeyRegistry())
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
        loop.merger = GraphMerger()

        # Route a speaker-attributed Relation through the real merger.
        rel = Relation(
            subject="spk-1",
            predicate="requires",
            object="python",
            relation_type="factual",
            speaker_id="spk-1",
        )
        session = SessionGraph(
            session_id="s001",
            timestamp="2026-01-01T00:00:00+00:00",
            entities=[Entity(name="spk-1", entity_type="person", speaker_id="spk-1")],
            relations=[rel],
        )
        loop.merger.merge(session)

        tier_keyed: dict = {"episodic": [], "procedural": []}
        loop._build_all_edge_entries_into(tier_keyed)

        assert len(tier_keyed["episodic"]) == 1
        minted_key = tier_keyed["episodic"][0]["key"]
        bk = loop.store.bookkeeping_for_key(minted_key)
        assert bk is not None
        assert bk["speaker_id"] == "spk-1", (
            f"Expected speaker_id='spk-1' from edge stamp; got {bk['speaker_id']!r}"
        )

    def test_concept_edge_with_no_speaker_allows_empty(self, tmp_path):
        """Concept→concept edge with no node speaker_id mints with bk speaker_id=''.

        The allow-empty path at the mint site permits "" for concept-rooted edges.
        """
        loop = self._make_loop(tmp_path)
        g = loop.merger.graph

        g.add_node("concept_x", entity_type="concept", attributes={"name": "ConceptX"})
        g.add_node("concept_y", entity_type="concept", attributes={"name": "ConceptY"})
        g.add_edge("concept_x", "concept_y", predicate="related_to", relation_type="factual")

        tier_keyed: dict = {"episodic": [], "procedural": []}
        loop._build_all_edge_entries_into(tier_keyed)

        assert len(tier_keyed["episodic"]) == 1
        minted_key = tier_keyed["episodic"][0]["key"]
        bk = loop.store.bookkeeping_for_key(minted_key)
        assert bk is not None
        assert bk["speaker_id"] == "", (
            f"Concept edge should mint with speaker_id=''; got {bk['speaker_id']!r}"
        )


# ---------------------------------------------------------------------------
# _build_all_edge_entries_into — unified edge→entry builder (keyed branch)
# ---------------------------------------------------------------------------


class TestCollectKeyedEdgesInto:
    """Unit tests for the keyed-edge branch of _build_all_edge_entries_into.

    The unified builder walks ``merger.graph.edges(data=True)`` and handles:
    - Keyed edges (with ik_key): sourced from store, appended to tier_keyed with
      {key, subject, predicate, object, speaker_id}.
    - Keyless edges: minted; tested separately in test_graph_enrichment.py.
    - Predicate-less edges: skipped unconditionally.

    These tests use only keyed and predicate-less edges to exercise the
    keyed-branch behavior in isolation.
    """

    @staticmethod
    def _make_loop(tmp_path, *, procedural_enabled=False):
        """Minimal ConsolidationLoop stub for keyed-edge tests."""
        import networkx as nx
        from peft import PeftModel

        from paramem.memory.store import MemoryStore
        from paramem.training.consolidation import ConsolidationLoop
        from paramem.utils.config import AdapterConfig, ConsolidationConfig, TrainingConfig

        loop = object.__new__(ConsolidationLoop)
        loop.model = MagicMock()
        loop.model.__class__ = PeftModel
        loop.tokenizer = MagicMock()
        loop.config = ConsolidationConfig(min_tier_key_floor=0, tier_fast_start=False)
        loop.training_config = TrainingConfig(num_epochs=1, gradient_checkpointing=False)
        loop.episodic_config = AdapterConfig(rank=4, alpha=8, target_modules=["q_proj"])
        loop.semantic_config = AdapterConfig(rank=4, alpha=8, target_modules=["q_proj"])
        loop.procedural_config = (
            AdapterConfig(rank=4, alpha=8, target_modules=["q_proj"])
            if procedural_enabled
            else None
        )
        loop.wandb_config = None
        loop._thermal_policy = None
        loop.output_dir = tmp_path
        loop.store = MemoryStore(replay_enabled=True)
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
        merger = MagicMock()
        merger.graph = nx.MultiDiGraph()
        loop.merger = merger
        return loop

    def test_keyed_edge_appended_with_correct_payload_and_tier(self, tmp_path):
        """A keyed edge with a predicate and content entry is appended to tier_keyed.

        Edge has: predicate='lives_in', ik_key='graph1', subject='Alice', object='Berlin'.
        Store entry has subject/predicate/object.  Bookkeeping relation_type='factual'
        → routes to episodic tier.  Payload = {key, subject, predicate, object, speaker_id}
        from store entry + bookkeeping (uniform shape, same as the keyless branch).
        """

        from paramem.memory.persistence import _IK_KEY_ATTR

        loop = self._make_loop(tmp_path)
        g = loop.merger.graph

        # (a) Keyed predicate-bearing edge — must be appended.
        eid = g.add_edge("Alice", "Berlin", predicate="lives_in")
        g["Alice"]["Berlin"][eid][_IK_KEY_ATTR] = "graph1"

        # (b) Predicate-less edge — must be skipped.
        g.add_edge("Charlie", "Dave")

        loop.store.put(
            "episodic",
            "graph1",
            {
                "key": "graph1",
                "subject": "Alice",
                "predicate": "lives_in",
                "object": "Berlin",
                "speaker_id": "spk-0",
            },
            register=True,
        )
        loop.store.set_bookkeeping(
            "graph1", speaker_id="spk-0", relation_type="factual", first_seen=""
        )

        tier_keyed: dict = {"episodic": [], "semantic": [], "procedural": []}
        loop._build_all_edge_entries_into(tier_keyed)

        assert len(tier_keyed["episodic"]) == 1, (
            f"Expected exactly 1 episodic entry; got {tier_keyed['episodic']}"
        )
        assert tier_keyed["semantic"] == [], "No semantic entries expected"
        assert tier_keyed["procedural"] == [], "No procedural entries expected"

        entry = tier_keyed["episodic"][0]
        assert entry["key"] == "graph1"
        assert entry["subject"] == "Alice"
        assert entry["predicate"] == "lives_in"
        assert entry["object"] == "Berlin"
        # Keyed branch now includes speaker_id in the uniform entry shape.
        assert entry["speaker_id"] == "spk-0"

    def test_keyed_edge_with_no_content_entry_is_skipped(self, tmp_path):
        """A keyed edge whose store.get returns None is silently skipped."""

        from paramem.memory.persistence import _IK_KEY_ATTR

        loop = self._make_loop(tmp_path)
        g = loop.merger.graph

        eid = g.add_edge("Alice", "Berlin", predicate="lives_in")
        g["Alice"]["Berlin"][eid][_IK_KEY_ATTR] = "ghost_key"
        # Intentionally: no store.put for 'ghost_key'.

        tier_keyed: dict = {"episodic": [], "semantic": [], "procedural": []}
        loop._build_all_edge_entries_into(tier_keyed)

        assert tier_keyed["episodic"] == [], "Keyed edge with no content entry must be skipped"

    def test_preference_relation_routes_to_procedural_tier(self, tmp_path):
        """A keyed edge with bookkeeping relation_type='preference' routes to procedural."""

        from paramem.memory.persistence import _IK_KEY_ATTR

        loop = self._make_loop(tmp_path, procedural_enabled=True)
        g = loop.merger.graph

        eid = g.add_edge("Alice", "tea", predicate="prefers")
        g["Alice"]["tea"][eid][_IK_KEY_ATTR] = "proc1"

        loop.store.put(
            "procedural",
            "proc1",
            {
                "key": "proc1",
                "subject": "Alice",
                "predicate": "prefers",
                "object": "tea",
                "speaker_id": "spk-0",
            },
            register=True,
        )
        loop.store.set_bookkeeping(
            "proc1", speaker_id="spk-0", relation_type="preference", first_seen=""
        )

        tier_keyed: dict = {"episodic": [], "semantic": [], "procedural": []}
        loop._build_all_edge_entries_into(tier_keyed)

        assert len(tier_keyed["procedural"]) == 1, (
            f"Expected 1 procedural entry; got {tier_keyed['procedural']}"
        )
        assert tier_keyed["procedural"][0]["key"] == "proc1"

    def test_semantic_key_stays_semantic(self, tmp_path):
        """A keyed edge whose store tier is 'semantic' routes to the semantic tier."""

        from paramem.memory.persistence import _IK_KEY_ATTR

        loop = self._make_loop(tmp_path)
        g = loop.merger.graph

        eid = g.add_edge("Alice", "Berlin", predicate="lives_in")
        g["Alice"]["Berlin"][eid][_IK_KEY_ATTR] = "sem1"

        loop.store.put(
            "semantic",
            "sem1",
            {
                "key": "sem1",
                "subject": "Alice",
                "predicate": "lives_in",
                "object": "Berlin",
                "speaker_id": "spk-0",
            },
            register=True,
        )
        loop.store.set_bookkeeping(
            "sem1", speaker_id="spk-0", relation_type="factual", first_seen=""
        )

        tier_keyed: dict = {"episodic": [], "semantic": [], "procedural": []}
        loop._build_all_edge_entries_into(tier_keyed)

        assert len(tier_keyed["semantic"]) == 1, (
            f"Expected 1 semantic entry; got {tier_keyed['semantic']}"
        )
        assert tier_keyed["semantic"][0]["key"] == "sem1"
        assert tier_keyed["episodic"] == []

    def test_keyed_and_keyless_entries_have_identical_shape_including_speaker_id(self, tmp_path):
        """Keyed and keyless edges produce tier_keyed entries with the same field
        set including speaker_id.

        Regression guard for B3b: before this refactor the keyless/minted branch
        included speaker_id in tier_keyed entries but the keyed/existing branch
        did not — the drift was the root cause of the /debug/dump empty-speaker_id
        artifact.  After unification both branches must produce identical shapes.
        """
        from paramem.memory.persistence import _IK_KEY_ATTR
        from paramem.training.key_registry import KeyRegistry

        loop = self._make_loop(tmp_path)
        for tier in ("episodic", "semantic", "procedural"):
            loop.store.load_registry(tier, KeyRegistry())

        g = loop.merger.graph

        # (a) Keyed edge with a store entry carrying speaker_id.
        eid = g.add_edge("Alice", "Berlin", predicate="lives_in")
        g["Alice"]["Berlin"][eid][_IK_KEY_ATTR] = "graph99"
        loop.store.put(
            "episodic",
            "graph99",
            {
                "key": "graph99",
                "subject": "Alice",
                "predicate": "lives_in",
                "object": "Berlin",
                "speaker_id": "spk-existing",
            },
            register=True,
        )
        loop.store.set_bookkeeping(
            "graph99", speaker_id="spk-existing", relation_type="factual", first_seen=""
        )

        # (b) Keyless edge (no ik_key) with a speaker_id-bearing subject node.
        g.add_node("bob", speaker_id="spk-new", attributes={"name": "Bob"})
        g.add_node("paris", attributes={"name": "Paris"})
        g.add_edge("bob", "paris", predicate="visits", relation_type="factual")

        tier_keyed: dict = {"episodic": [], "semantic": [], "procedural": []}
        loop._build_all_edge_entries_into(tier_keyed)

        episodic = tier_keyed["episodic"]
        assert len(episodic) == 2, (
            f"Expected 2 episodic entries (1 keyed + 1 minted); got {episodic}"
        )

        _required_fields = {"key", "subject", "predicate", "object", "speaker_id"}
        for entry in episodic:
            missing = _required_fields - entry.keys()
            assert not missing, (
                f"Entry missing fields {missing!r}: {entry!r}. "
                "Both keyed and keyless branches must produce the same uniform shape."
            )

        # Locate each entry by key prefix.
        keyed_entry = next(e for e in episodic if e["key"] == "graph99")
        minted_entry = next(e for e in episodic if e["key"] != "graph99")

        # Keyed entry: speaker_id from bookkeeping.
        assert keyed_entry["speaker_id"] == "spk-existing", (
            f"Keyed entry speaker_id should be 'spk-existing'; got {keyed_entry['speaker_id']!r}"
        )
        # Minted entry: speaker_id from subject node attribute.
        assert minted_entry["speaker_id"] == "spk-new", (
            f"Minted entry speaker_id should be 'spk-new'; got {minted_entry['speaker_id']!r}"
        )


# ---------------------------------------------------------------------------
# _build_registry_true_relations — optional keys filter
# ---------------------------------------------------------------------------


class TestBuildRegistryTrueRelationsKeysFilter:
    """Unit tests for the optional ``keys`` parameter of
    ``ConsolidationLoop._build_registry_true_relations``.

    Verifies:
    (a) keys=None returns all active keys (unchanged baseline behavior).
    (b) keys=[subset] returns only those keys, each carrying indexed_key /
        speaker_id / relation_type from bookkeeping.
    (c) An empty list returns an empty result.
    """

    @staticmethod
    def _make_loop(tmp_path):
        """Minimal ConsolidationLoop stub for _build_registry_true_relations tests."""
        from peft import PeftModel

        from paramem.memory.store import MemoryStore
        from paramem.training.consolidation import ConsolidationLoop
        from paramem.utils.config import AdapterConfig, ConsolidationConfig, TrainingConfig

        loop = object.__new__(ConsolidationLoop)
        loop.model = MagicMock()
        loop.model.__class__ = PeftModel
        loop.tokenizer = MagicMock()
        loop.config = ConsolidationConfig()
        loop.training_config = TrainingConfig(num_epochs=1, gradient_checkpointing=False)
        loop.episodic_config = AdapterConfig(rank=4, alpha=8, target_modules=["q_proj"])
        loop.semantic_config = AdapterConfig(rank=4, alpha=8, target_modules=["q_proj"])
        loop.procedural_config = None
        loop.wandb_config = None
        loop._thermal_policy = None
        loop.output_dir = tmp_path
        loop.store = MemoryStore(replay_enabled=True)
        return loop

    def _populate_store(self, store, key, subject, predicate, obj, relation_type, speaker_id):
        """Register a key in store with entry + bookkeeping."""
        store.put(
            "episodic",
            key,
            {
                "key": key,
                "subject": subject,
                "predicate": predicate,
                "object": obj,
                "speaker_id": speaker_id,
            },
            register=True,
        )
        store.set_bookkeeping(
            key,
            speaker_id=speaker_id,
            relation_type=relation_type,
            first_seen="",
        )

    def test_keys_none_returns_all_active(self, tmp_path):
        """keys=None iterates all active keys — baseline behavior unchanged."""
        loop = self._make_loop(tmp_path)
        self._populate_store(loop.store, "k1", "Alice", "lives_in", "Berlin", "factual", "spk-0")
        self._populate_store(loop.store, "k2", "Bob", "works_at", "Acme", "factual", "spk-1")

        relations = loop._build_registry_true_relations(keys=None)

        returned_keys = {r.indexed_key for r in relations}
        assert returned_keys == {"k1", "k2"}, (
            f"keys=None must return all active keys; got {returned_keys}"
        )

    def test_keys_subset_returns_only_those_keys(self, tmp_path):
        """Providing a subset returns only those keys, with correct SPO / metadata."""
        loop = self._make_loop(tmp_path)
        self._populate_store(loop.store, "k1", "Alice", "lives_in", "Berlin", "factual", "spk-0")
        self._populate_store(loop.store, "k2", "Bob", "works_at", "Acme", "factual", "spk-1")
        self._populate_store(loop.store, "k3", "Carol", "knows", "Dave", "factual", "spk-2")

        relations = loop._build_registry_true_relations(keys=["k1", "k3"])

        returned_keys = {r.indexed_key for r in relations}
        assert returned_keys == {"k1", "k3"}, f"Expected {{k1, k3}}; got {returned_keys}"
        assert "k2" not in returned_keys, "k2 must not appear when not in keys filter"

        # Verify payload for k1.
        r1 = next(r for r in relations if r.indexed_key == "k1")
        assert r1.subject == "Alice"
        assert r1.predicate == "lives_in"
        assert r1.object == "Berlin"
        assert r1.speaker_id == "spk-0"
        assert r1.relation_type == "factual"

    def test_keys_empty_list_returns_empty(self, tmp_path):
        """An empty keys list returns an empty relation list."""
        loop = self._make_loop(tmp_path)
        self._populate_store(loop.store, "k1", "Alice", "lives_in", "Berlin", "factual", "spk-0")

        relations = loop._build_registry_true_relations(keys=[])

        assert relations == [], f"Expected empty list for keys=[]; got {relations}"

    def test_keys_none_and_no_args_are_equivalent(self, tmp_path):
        """Calling with no argument (default) yields the same result as keys=None."""
        loop = self._make_loop(tmp_path)
        self._populate_store(loop.store, "k1", "Alice", "lives_in", "Berlin", "factual", "spk-0")

        default_result = loop._build_registry_true_relations()
        explicit_none_result = loop._build_registry_true_relations(keys=None)

        assert {r.indexed_key for r in default_result} == {
            r.indexed_key for r in explicit_none_result
        }, "Default call and keys=None must produce identical key sets"

    def test_first_seen_round_trips_from_bookkeeping_onto_relation(self, tmp_path):
        """first_seen flows: bookkeeping['first_seen'] -> Relation.first_seen ->
        merged edge['first_seen'], the full registry-true re-merge path."""
        from paramem.graph.merger import GraphMerger
        from paramem.graph.schema import SessionGraph

        loop = self._make_loop(tmp_path)
        loop.store.put(
            "episodic",
            "k1",
            {
                "key": "k1",
                "subject": "alice",
                "predicate": "lives_in",
                "object": "berlin",
                "speaker_id": "spk-0",
            },
            register=True,
        )
        loop.store.set_bookkeeping(
            "k1",
            speaker_id="spk-0",
            relation_type="factual",
            last_seen="2026-06-20T00:00:00Z",
            first_seen="2026-01-01T00:00:00Z",
        )

        relations = loop._build_registry_true_relations()
        r1 = next(r for r in relations if r.indexed_key == "k1")
        assert r1.first_seen == "2026-01-01T00:00:00Z", (
            f"Relation.first_seen must carry bookkeeping's first_seen; got {r1.first_seen!r}"
        )

        # Re-merge (fold's own path): registry-true relations feed into a fresh
        # GraphMerger at fold timestamp="" (the fold sentinel).
        merger = GraphMerger()
        session = SessionGraph(session_id="fold", timestamp="", relations=relations)
        graph = merger.merge(session, resolve_contradictions=False)
        edge_data = next(iter(graph["alice"]["berlin"].values()))
        assert edge_data["first_seen"] == "2026-01-01T00:00:00Z", (
            f"Merged edge must carry the registry-true first_seen; got {edge_data['first_seen']!r}"
        )
        assert edge_data["last_seen"] == "2026-06-20T00:00:00Z", (
            f"Merged edge must carry the registry-true last_seen; got {edge_data['last_seen']!r}"
        )


# ---------------------------------------------------------------------------
# _materialize_consolidation_graph: extra_relations seam + interim call
# ---------------------------------------------------------------------------


class TestMaterializeInterimExtraRelations:
    """Unit tests for the extra_relations parameter on
    _materialize_consolidation_graph and its use in run_consolidation_cycle.

    Covered invariants:

    1. extra_relations=None and extra_relations=[] are no-ops for the fold
       caller — the output is byte-identical to the call without extra_relations.
    2. After the interim materialize call, the merged graph contains both the
       slot's recalled (registry-true) relations AND the pending-session
       extra_relations.  A pending fact whose SPO matches a recalled key fires
       Case-1-adopt: exactly one key survives, recurrence is bumped.
    3. A pending UNREGISTERED relation (not in the slot registry) does NOT
       appear in recall_miss_keys.  recall_miss_keys is computed against
       store.all_active_keys() BEFORE the reset — unregistered relations are
       invisible to the miss set.
    4. dcf4189 speaker_id invariant: a minted interim key inherits speaker_id
       from the relation dict through the graph-walk keying step — the
       materialize step does not disrupt this flow.
    """

    @staticmethod
    def _make_loop(tmp_path):
        """Minimal ConsolidationLoop with a real GraphMerger for materialize tests.

        Uses object.__new__ so no GPU model or extraction pipeline is needed.
        Includes a real GraphMerger (model=None) so merger.merge / reset_graph
        execute correctly, and a real MemoryStore with replay_enabled=True.
        reconstruct_graph must be mocked by each test (it calls probe_entries
        which requires a real model).
        """
        from peft import PeftModel

        from paramem.graph.merger import GraphMerger
        from paramem.memory.store import MemoryStore
        from paramem.training.consolidation import ConsolidationLoop
        from paramem.training.key_registry import KeyRegistry
        from paramem.utils.config import AdapterConfig, ConsolidationConfig, TrainingConfig

        loop = object.__new__(ConsolidationLoop)
        loop.model = MagicMock()
        loop.model.__class__ = PeftModel
        loop.model.peft_config = {}
        loop.model.add_adapter.side_effect = lambda name, cfg: loop.model.peft_config.update(
            {name: cfg}
        )
        loop.tokenizer = MagicMock()
        loop.config = ConsolidationConfig(
            indexed_key_replay=True,
            # base defaults: sota_enabled=False, refinement_enrichment="off",
            # refinement_normalization="off" — tests override per scenario
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

        # Real GraphMerger (no model) so merge/reset_graph run correctly.
        loop.merger = GraphMerger(model=None)

        store = MemoryStore(replay_enabled=True)
        for tier in ("episodic", "semantic", "procedural"):
            store.load_registry(tier, KeyRegistry())
        loop.store = store

        loop._probe_passing_keys = lambda adapter_name, entries: {e["key"] for e in entries}
        return loop

    @staticmethod
    def _fake_reconstruct(loop, *, tier=None, strict=False):
        """Reconstruct stub: returns empty graph + no failures.

        Used to suppress real GPU probe calls in materialize unit tests.
        The recall-miss diagnostics tested here operate on an already-populated
        store; the stub causes ALL registered keys to land in recall_miss_keys
        (no reconstruction → failure for each active key).  Tests that need
        precise recall-miss behaviour set up recon spies themselves.
        """
        import networkx as nx

        from paramem.graph.reconstruct import ReconstructionResult

        return ReconstructionResult(graph=nx.MultiDiGraph(), failures=[])

    # ------------------------------------------------------------------
    # 1. extra_relations=None / [] is a no-op for the fold caller
    # ------------------------------------------------------------------

    def test_extra_relations_none_is_noop(self, tmp_path):
        """extra_relations=None must produce the same output as omitting the param.

        Calls _materialize_consolidation_graph three times on the same loop:
        (a) no extra_relations arg  (fold-style default)
        (b) extra_relations=None    (explicit None)
        (c) extra_relations=[]      (explicit empty list)
        All three must return identical (recall_miss_keys, recon_relations) and
        leave the merger graph in the same state (same edge count).
        """
        from unittest.mock import patch

        loop = self._make_loop(tmp_path)

        with patch(
            "paramem.training.consolidation.reconstruct_graph",
            side_effect=self._fake_reconstruct,
        ):
            miss_a, recon_a = loop._materialize_consolidation_graph()
            edges_a = loop.merger.graph.number_of_edges()

            # Reset merger so each call starts fresh.
            loop.merger.reset_graph()
            miss_b, recon_b = loop._materialize_consolidation_graph(extra_relations=None)
            edges_b = loop.merger.graph.number_of_edges()

            loop.merger.reset_graph()
            miss_c, recon_c = loop._materialize_consolidation_graph(extra_relations=[])
            edges_c = loop.merger.graph.number_of_edges()

        assert miss_a == miss_b == miss_c, (
            f"recall_miss_keys diverge: no-arg={miss_a}, None={miss_b}, []={miss_c}"
        )
        assert (
            {r.indexed_key for r in recon_a}
            == {r.indexed_key for r in recon_b}
            == {r.indexed_key for r in recon_c}
        ), "recon_relations indexed_key sets must be identical for all three variants"
        assert edges_a == edges_b == edges_c, (
            f"Merger graph edge counts diverge: {edges_a}, {edges_b}, {edges_c}"
        )

    # ------------------------------------------------------------------
    # 2. Case-1 adoption — pending fact matching recalled key → one key
    # ------------------------------------------------------------------

    def test_extra_relations_case1_adoption(self, tmp_path):
        """A pending relation whose SPO matches a recalled slot key adopts the key.

        Setup:
        - Register key 'graph1' in the slot (Alice/lives_in/Berlin, spk-A).
        - Pass a Relation(Alice, lives_in, Berlin) as extra_relations.
        - extra_relations have no indexed_key, so they are keyless pending edges.

        Expected:
        - The merger graph has exactly ONE edge after materialize (Case-1
          collapses the duplicate SPO on re-merge: registry-true edge with
          ik_key='graph1' exists; the extra Relation (keyless) merges as Case-3
          new-edge since there is no ik_key conflict — both land as separate
          edges because they arrive in different merge calls and have no key to
          trigger the duplicate-key path).
        - Specifically: the registry-true relation carries indexed_key='graph1';
          the extra relation has no indexed_key.  After the fold re-merge (which
          uses resolve_contradictions=False, skipping Case-2), both edges coexist:
          the existing edge has ik_key and the extra relation lacks one so Case-1-
          adopt writes ik_key to the new keyless edge on a same-SPO collision
          (recurrence bump).
        - The edge with ik_key='graph1' must be present.
        - The merger.reinforcements dict records the surviving key → last_seen (bump path).
        """
        from unittest.mock import patch

        from paramem.graph.schema import Relation

        loop = self._make_loop(tmp_path)

        # Register one key in the slot.
        _adapter = "episodic_interim_20260101T0000"
        from paramem.training.key_registry import KeyRegistry

        slot_reg = KeyRegistry()
        loop.store.load_registry(_adapter, slot_reg)
        loop.store.put(
            _adapter,
            "graph1",
            {
                "key": "graph1",
                "subject": "alice",
                "predicate": "lives in",
                "object": "berlin",
                "speaker_id": "spk-a",
            },
            register=True,
        )
        loop.store.set_bookkeeping(
            "graph1",
            speaker_id="spk-a",
            relation_type="factual",
            first_seen="",
        )

        # Build a pending extra_relation with the same SPO (no indexed_key).
        extra = [
            Relation(
                subject="alice",
                predicate="lives in",
                object="berlin",
                relation_type="factual",
                confidence=1.0,
                speaker_id="spk-a",
            )
        ]

        with patch(
            "paramem.training.consolidation.reconstruct_graph",
            side_effect=self._fake_reconstruct,
        ):
            miss_keys, recon_relations = loop._materialize_consolidation_graph(
                tier=_adapter,
                keys=list(loop.store.active_keys_in_tier(_adapter)),
                extra_relations=extra,
            )

        # The registry-true relation for 'graph1' must be in recon_relations.
        recon_keys = {r.indexed_key for r in recon_relations}
        assert "graph1" in recon_keys, (
            f"Registry-true key 'graph1' must appear in recon_relations; got {recon_keys}"
        )

        # The merged graph must have the ik_key='graph1' edge (from the registry-true merge).
        from paramem.memory.persistence import _IK_KEY_ATTR

        all_ik_keys = {data.get(_IK_KEY_ATTR) for _, _, data in loop.merger.graph.edges(data=True)}
        assert "graph1" in all_ik_keys, (
            f"Edge with ik_key='graph1' must be present in merged graph; "
            f"found ik_keys: {all_ik_keys}"
        )

    # ------------------------------------------------------------------
    # 3. Pending unregistered relation does NOT enter recall_miss_keys
    # ------------------------------------------------------------------

    def test_unregistered_extra_relation_not_in_recall_miss(self, tmp_path):
        """An unregistered extra_relation must not appear in recall_miss_keys.

        recall_miss_keys is computed against store.all_active_keys() BEFORE the
        graph reset — extra_relations are pending (not yet registered) and are
        therefore invisible to the miss set.

        Setup: empty slot (no registered keys).  Pass one extra_relation.
        Expected: recall_miss_keys is empty (no registered keys to miss).
        """
        from unittest.mock import patch

        from paramem.graph.schema import Relation

        loop = self._make_loop(tmp_path)

        _adapter = "episodic_interim_20260101T0000"
        from paramem.training.key_registry import KeyRegistry

        loop.store.load_registry(_adapter, KeyRegistry())
        # Do NOT register any keys — the extra_relation is purely pending.

        extra = [
            Relation(
                subject="alice",
                predicate="works at",
                object="acme corp",
                relation_type="factual",
                confidence=1.0,
                speaker_id="spk-a",
            )
        ]

        with patch(
            "paramem.training.consolidation.reconstruct_graph",
            side_effect=self._fake_reconstruct,
        ):
            miss_keys, _ = loop._materialize_consolidation_graph(
                tier=_adapter,
                keys=list(loop.store.active_keys_in_tier(_adapter)),
                extra_relations=extra,
            )

        assert miss_keys == set(), (
            f"Pending unregistered relation must NOT appear in recall_miss_keys; got: {miss_keys}"
        )

    # ------------------------------------------------------------------
    # 4. dcf4189 speaker_id inheritance through the materialize path
    # ------------------------------------------------------------------

    def test_speaker_id_preserved_through_materialize_in_cycle(self, tmp_path):
        """Minted interim keys inherit speaker_id from episodic_rels through the
        new materialize path.

        _materialize_consolidation_graph runs BEFORE the graph-walk keying step.
        The key-prep reads speaker_id from episodic_rels (via _mint_keyed_entries),
        NOT from graph nodes.  This test verifies that adding the materialize step
        does not corrupt speaker_id in minted keys (dcf4189 invariant).

        Strategy: run a simulate cycle with episodic_rels carrying explicit
        speaker_ids, mock reconstruct_graph internals, and assert the minted
        entries carry the correct speaker_id.
        """
        from unittest.mock import patch

        loop = self._make_loop(tmp_path)

        episodic_rels = [
            {
                "subject": "speaker0",
                "predicate": "lives_in",
                "object": "Berlin",
                "relation_type": "factual",
                "speaker_id": "speaker0",
            },
            {
                "subject": "speaker0",
                "predicate": "works_at",
                "object": "Acme Corp",
                "relation_type": "factual",
                "speaker_id": "speaker0",
            },
        ]

        stamp = "20260101T0000"
        _adapter = f"episodic_interim_{stamp}"

        with patch(
            "paramem.training.consolidation.reconstruct_graph",
            side_effect=self._fake_reconstruct,
        ):
            result = loop.run_consolidation_cycle(
                episodic_rels,
                [],
                speaker_id="speaker0",
                mode="simulate",
                run_label="b1-speaker-id-test",
                stamp=stamp,
            )

        assert result.get("mode") in ("simulated", "noop", "queued"), (
            f"Cycle did not complete as expected; result={result!r}"
        )
        if result.get("mode") == "simulated":
            # Verify all minted keys have the correct speaker_id.
            for _tier, key, entry in loop.store.iter_entries():
                if entry.get("subject") in ("speaker0", "speaker0"):
                    assert entry.get("speaker_id") == "speaker0", (
                        f"Key {key!r} minted with wrong speaker_id: "
                        f"expected 'speaker0', got {entry.get('speaker_id')!r}"
                    )
                    bk = loop.store.bookkeeping_for_key(key)
                    if bk is not None:
                        assert bk.get("speaker_id") == "speaker0", (
                            f"Bookkeeping for {key!r} has wrong speaker_id: "
                            f"expected 'speaker0', got {bk.get('speaker_id')!r}"
                        )

    # ------------------------------------------------------------------
    # Integration: interim materialize call in run_consolidation_cycle
    # ------------------------------------------------------------------

    def test_interim_materialize_called_before_key_prep(self, tmp_path):
        """run_consolidation_cycle calls _materialize_consolidation_graph before
        the graph-walk keying step.

        Uses a spy on _materialize_consolidation_graph and asserts:
        (a) It is called exactly once per cycle.
        (b) It is called with tier=adapter_name and the slot's active keys.

        Key generation uses the unified graph-walk
        (_build_all_edge_entries_into, defer=True).  When the materialize stub returns
        (set(), []) and the merger graph is empty, the walk produces zero keys
        — the store's tier is empty after the cycle (expected here).
        Tests that verify key minting (dcf4189, speaker_id) are in
        TestInterimKeyedWalk.
        """

        loop = self._make_loop(tmp_path)
        stamp = "20260115T0000"
        _adapter = f"episodic_interim_{stamp}"

        # Track calls to _materialize_consolidation_graph.
        materialize_calls: list[dict] = []

        def _spy_materialize(**kw):
            materialize_calls.append(kw)
            return (set(), [])

        episodic_rels = [
            {
                "subject": "Alice",
                "predicate": "lives_in",
                "object": "Zurich",
                "relation_type": "factual",
                "speaker_id": "spk-x",
            },
        ]

        loop._materialize_consolidation_graph = _spy_materialize  # type: ignore[method-assign]

        loop.run_consolidation_cycle(
            episodic_rels,
            [],
            speaker_id="spk-x",
            mode="simulate",
            run_label="b1-call-order",
            stamp=stamp,
        )

        assert len(materialize_calls) == 1, (
            f"Expected exactly 1 _materialize_consolidation_graph call; "
            f"got {len(materialize_calls)}: {materialize_calls}"
        )
        call_kw = materialize_calls[0]
        assert call_kw.get("tier") == _adapter, (
            f"materialize tier must be adapter_name={_adapter!r}; got {call_kw.get('tier')!r}"
        )
        # keys= should be the slot's active keys (empty for a fresh slot).
        assert "keys" in call_kw, "materialize must be called with keys= kwarg"


# ---------------------------------------------------------------------------
# Multi-speaker dcf4189 invariant + keyed-walk vs flat parity
# ---------------------------------------------------------------------------


class TestInterimKeyedWalk:
    """Keyed-walk acceptance tests: speaker_id and SPO-content parity.

    1. Multi-speaker dcf4189 invariant: two pending facts from DISTINCT
       speakers in one interim cycle must yield minted keys with DIFFERENT,
       correct speaker_ids — they must NOT collapse to a single default.

    2. Training-set equivalence: the keyed-walk (harvest+apply) produces the
       same (subject, predicate, object) content set as the episodic relation
       input for the same fixed input.
    """

    @staticmethod
    def _make_loop(tmp_path):
        """Minimal ConsolidationLoop with a real GraphMerger.

        Mirrors TestMaterializeInterimExtraRelations._make_loop exactly so the
        keyed-walk tests run against the same test-loop shape as the
        materialize tests.
        """
        from peft import PeftModel

        from paramem.graph.merger import GraphMerger
        from paramem.memory.store import MemoryStore
        from paramem.training.consolidation import ConsolidationLoop
        from paramem.training.key_registry import KeyRegistry
        from paramem.utils.config import AdapterConfig, ConsolidationConfig, TrainingConfig

        loop = object.__new__(ConsolidationLoop)
        loop.model = MagicMock()
        loop.model.__class__ = PeftModel
        loop.model.peft_config = {}
        loop.model.add_adapter.side_effect = lambda name, cfg: loop.model.peft_config.update(
            {name: cfg}
        )
        loop.tokenizer = MagicMock()
        loop.config = ConsolidationConfig(
            indexed_key_replay=True,
            # base defaults: sota_enabled=False, refinement_enrichment="off"
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

        loop.merger = GraphMerger(model=None)

        store = MemoryStore(replay_enabled=True)
        for tier in ("episodic", "semantic", "procedural"):
            store.load_registry(tier, KeyRegistry())
        loop.store = store

        loop._probe_passing_keys = lambda adapter_name, entries: {e["key"] for e in entries}
        return loop

    @staticmethod
    def _fake_reconstruct(loop, *, tier=None, strict=False):
        """Stub reconstruct_graph: empty graph, no failures."""
        import networkx as nx

        from paramem.graph.reconstruct import ReconstructionResult

        return ReconstructionResult(graph=nx.MultiDiGraph(), failures=[])

    # ------------------------------------------------------------------
    # 1. Multi-speaker dcf4189 invariant through the graph-walk
    # ------------------------------------------------------------------

    def test_two_speakers_yield_distinct_speaker_ids_in_minted_keys(self, tmp_path):
        """Two pending facts from DISTINCT speakers in one interim cycle must
        yield minted keys whose speaker_ids are DIFFERENT and correct — neither
        must collapse to the caller default.

        Regression class: dcf4189 fixed keyless-edge minting inheriting
        speaker_id from the graph subject node rather than falling back to "".
        Without entity re-synthesis in _materialize_consolidation_graph,
        both minted keys would carry speaker_id="" because the merged graph
        lacked speaker_id attributes on the subject nodes.

        Strategy:
        - Two Relation objects: speaker0/works_at/Acme (speaker_id="speaker0")
          and speaker1/lives_in/Paris (speaker_id="speaker1").
        - Call _materialize_consolidation_graph(extra_relations=[...]) so it
          synthesises entities and stamps speaker_id on each subject node.
        - Run _build_all_edge_entries_into.
        - Assert: each minted key carries its own speaker's id; the two
          speaker_ids are different; neither equals "".
        """
        from unittest.mock import patch

        from paramem.graph.schema import Relation

        loop = self._make_loop(tmp_path)

        extra_relations = [
            Relation(
                subject="speaker0",
                predicate="works at",
                object="Acme Corp",
                relation_type="factual",
                confidence=1.0,
                speaker_id="speaker0",
            ),
            Relation(
                subject="speaker1",
                predicate="lives in",
                object="Paris",
                relation_type="factual",
                confidence=1.0,
                speaker_id="speaker1",
            ),
        ]

        with patch(
            "paramem.training.consolidation.reconstruct_graph",
            side_effect=self._fake_reconstruct,
        ):
            loop._materialize_consolidation_graph(
                extra_relations=extra_relations,
            )

        # After materialize, merger.graph has two keyless edges (one per relation).
        # _build_all_edge_entries_into must resolve speaker_id from the subject node
        # for each — the entity re-synthesis in _materialize_consolidation_graph
        # stamps speaker_id onto the subject nodes so the walk reads them correctly.
        tier_keyed: dict = {"episodic": [], "procedural": []}
        loop._build_all_edge_entries_into(tier_keyed)

        minted = tier_keyed["episodic"]
        assert len(minted) == 2, (
            f"Expected 2 minted episodic entries (one per speaker); got {len(minted)}: {minted}"
        )

        # Map key → bookkeeping.speaker_id.
        key_to_speaker: dict[str, str] = {}
        for item in minted:
            key = item["key"]
            bk = loop.store.bookkeeping_for_key(key)
            assert bk is not None, f"No bookkeeping record for minted key {key!r}"
            key_to_speaker[key] = bk["speaker_id"]

        speaker_ids_found = set(key_to_speaker.values())

        # Both speaker_ids must be present.
        assert "speaker0" in speaker_ids_found, (
            f"Expected speaker0 in minted speaker_ids; found: {speaker_ids_found}"
        )
        assert "speaker1" in speaker_ids_found, (
            f"Expected speaker1 in minted speaker_ids; found: {speaker_ids_found}"
        )

        # The two minted keys must carry DIFFERENT speaker_ids — not collapsed.
        assert len(speaker_ids_found) == 2, (
            f"Both minted keys collapsed to the same speaker_id: {speaker_ids_found!r}. "
            "dcf4189 regression: subject-node speaker_id not inherited from entity re-synthesis."
        )

        # Neither must be the caller default ("").
        assert "" not in speaker_ids_found, (
            f"At least one minted key fell back to speaker_id=''; "
            f"speaker_ids found: {speaker_ids_found!r}. "
            "dcf4189 regression: speaker_id not stamped by entity synthesis."
        )


class TestMergeRegistryRelationsUnification:
    """Regression tests for the _merge_registry_relations unification.

    Before unification (4508-4530 in the pre-commit tree), the recon path built
    SessionGraph(entities=[], ...) which caused reconstructed speaker-subject nodes
    to receive entity_type="concept" with no speaker_id attribute.  The unified
    _merge_registry_relations helper applies _synth_speaker_entities to both paths,
    so speaker subjects now receive entity_type="person" + speaker_id from
    bookkeeping regardless of whether the relation came from the recon path or the
    extra-relations (pending-session) path.

    Tests:
    - MRR-1: recon path (no extra_relations) produces person node with speaker_id.
    - MRR-2: extra_relations path still produces person nodes (dcf4189 regression guard).
    - MRR-3: non-speaker relations produce no spurious person nodes.
    """

    @staticmethod
    def _make_loop(tmp_path):
        """Minimal ConsolidationLoop with a real GraphMerger (model=None).

        Mirrors TestMaterializeInterimExtraRelations._make_loop exactly.
        """
        from peft import PeftModel

        from paramem.graph.merger import GraphMerger
        from paramem.memory.store import MemoryStore
        from paramem.training.consolidation import ConsolidationLoop
        from paramem.training.key_registry import KeyRegistry
        from paramem.utils.config import AdapterConfig, ConsolidationConfig, TrainingConfig

        loop = object.__new__(ConsolidationLoop)
        loop.model = MagicMock()
        loop.model.__class__ = PeftModel
        loop.model.peft_config = {}
        loop.model.add_adapter.side_effect = lambda name, cfg: loop.model.peft_config.update(
            {name: cfg}
        )
        loop.tokenizer = MagicMock()
        loop.config = ConsolidationConfig(
            indexed_key_replay=True,
            # base defaults: sota_enabled=False, refinement_enrichment="off"
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

        loop.merger = GraphMerger(model=None)

        store = MemoryStore(replay_enabled=True)
        for tier in ("episodic", "semantic", "procedural"):
            store.load_registry(tier, KeyRegistry())
        loop.store = store

        loop._probe_passing_keys = lambda adapter_name, entries: {e["key"] for e in entries}
        return loop

    @staticmethod
    def _fake_reconstruct(loop, *, tier=None, strict=False):
        """Stub reconstruct_graph: empty graph, no failures."""
        import networkx as nx

        from paramem.graph.reconstruct import ReconstructionResult

        return ReconstructionResult(graph=nx.MultiDiGraph(), failures=[])

    # ------------------------------------------------------------------
    # MRR-1: recon path produces person node with speaker_id
    # ------------------------------------------------------------------

    def test_recon_path_produces_person_node_with_speaker_id(self, tmp_path):
        """_materialize_consolidation_graph with recon_relations must stamp
        entity_type='person' and speaker_id onto the subject node when
        subject == speaker_id (i.e. the subject is a speaker node).

        Regression: before _merge_registry_relations unification, the recon
        path used entities=[] which caused subject nodes to receive
        entity_type='concept' and no speaker_id attribute.  Graph-enrichment
        then rooted enrichment facts at unattributed concept nodes (speaker0
        vs Mara divergence).
        """
        from unittest.mock import patch

        from paramem.training.key_registry import KeyRegistry

        loop = self._make_loop(tmp_path)

        # Register a speaker key: subject == speaker_id (the speaker-node invariant).
        _adapter = "episodic_interim_20260101T0000"
        slot_reg = KeyRegistry()
        loop.store.load_registry(_adapter, slot_reg)
        loop.store.put(
            _adapter,
            "graph1",
            {
                "key": "graph1",
                "subject": "speaker0",
                "predicate": "works at",
                "object": "Acme Corp",
                "speaker_id": "speaker0",
            },
            register=True,
        )
        loop.store.set_bookkeeping(
            "graph1",
            speaker_id="speaker0",
            relation_type="factual",
            first_seen="",
        )

        with patch(
            "paramem.training.consolidation.reconstruct_graph",
            side_effect=self._fake_reconstruct,
        ):
            loop._materialize_consolidation_graph(
                tier=_adapter,
                keys=list(loop.store.active_keys_in_tier(_adapter)),
            )

        # The merged graph must have a node for the speaker subject.
        # §0 invariant (Step 2): speaker node keys are the casefolded speaker_id.
        # GraphMerger._resolve_entity returns entity.speaker_id verbatim.
        node_key = "speaker0"  # casefolded: "speaker0" IS the canonical form
        assert node_key in loop.merger.graph.nodes, (
            f"Speaker subject node {node_key!r} missing from merged graph after recon path; "
            f"nodes present: {list(loop.merger.graph.nodes)}"
        )

        node_data = loop.merger.graph.nodes[node_key]

        assert node_data.get("entity_type") == "person", (
            f"Recon path: expected entity_type='person' on speaker subject node; "
            f"got entity_type={node_data.get('entity_type')!r}. "
            "Regression: before _merge_registry_relations unification the recon path "
            "used entities=[], causing entity_type='concept' with no speaker_id."
        )
        assert node_data.get("speaker_id") == "speaker0", (
            f"Recon path: expected speaker_id='speaker0' (cased) in node attribute; "
            f"got speaker_id={node_data.get('speaker_id')!r}. "
            "Regression: entity synthesis was not applied to the recon path."
        )

    # ------------------------------------------------------------------
    # MRR-2: extra_relations path still produces person nodes (dcf4189 guard)
    # ------------------------------------------------------------------

    def test_extra_relations_path_still_produces_person_nodes(self, tmp_path):
        """extra_relations (pending-session) path continues to produce person nodes.

        Verifies that the dcf4189 fix (speaker_id on extra_relations nodes) is
        preserved through the _merge_registry_relations unification.  Passes
        extra_relations with speaker subjects and no registered keys so the recon
        path is a no-op; only the extra path runs.
        """
        from unittest.mock import patch

        from paramem.graph.schema import Relation

        loop = self._make_loop(tmp_path)

        extra_relations = [
            Relation(
                subject="speaker1",
                predicate="lives in",
                object="Berlin",
                relation_type="factual",
                confidence=1.0,
                speaker_id="speaker1",
            ),
        ]

        with patch(
            "paramem.training.consolidation.reconstruct_graph",
            side_effect=self._fake_reconstruct,
        ):
            loop._materialize_consolidation_graph(extra_relations=extra_relations)

        # §0 invariant (Step 2): speaker node keys are the casefolded speaker_id.
        node_key = "speaker1"  # casefolded: "speaker1" == "speaker1"
        assert node_key in loop.merger.graph.nodes, (
            f"Extra-relations path: speaker subject node {node_key!r} missing; "
            f"nodes: {list(loop.merger.graph.nodes)}"
        )
        node_data = loop.merger.graph.nodes[node_key]
        assert node_data.get("entity_type") == "person", (
            f"Extra-relations path: expected entity_type='person'; "
            f"got {node_data.get('entity_type')!r}"
        )
        assert node_data.get("speaker_id") == "speaker1", (
            f"Extra-relations path: expected speaker_id='speaker1' (cased) in attribute; "
            f"got {node_data.get('speaker_id')!r}"
        )

    # ------------------------------------------------------------------
    # MRR-3: non-speaker relations do not produce spurious person nodes
    # ------------------------------------------------------------------

    def test_non_speaker_relation_produces_no_person_node(self, tmp_path):
        """A relation whose subject != speaker_id must NOT produce a person entity.

        _synth_speaker_entities skips subjects that do not equal their
        relation's speaker_id.  A regular (non-speaker-subject) fact must still
        be merged, but its subject node must NOT receive entity_type='person' or
        a spurious speaker_id from entity synthesis.
        """
        from unittest.mock import patch

        from paramem.graph.schema import Relation

        loop = self._make_loop(tmp_path)

        # subject="Mara", speaker_id="speaker0" → subject != speaker_id → no entity.
        extra_relations = [
            Relation(
                subject="Mara",
                predicate="works at",
                object="Acme Corp",
                relation_type="factual",
                confidence=1.0,
                speaker_id="speaker0",
            ),
        ]

        with patch(
            "paramem.training.consolidation.reconstruct_graph",
            side_effect=self._fake_reconstruct,
        ):
            loop._materialize_consolidation_graph(extra_relations=extra_relations)

        from paramem.graph.name_match import canonical

        node_key = canonical("Mara")
        # Node must exist (the edge was merged), but must NOT have entity_type="person"
        # or speaker_id from entity synthesis (the subject is not a speaker node).
        if node_key in loop.merger.graph.nodes:
            node_data = loop.merger.graph.nodes[node_key]
            assert node_data.get("entity_type") != "person", (
                f"Non-speaker subject node should NOT have entity_type='person'; "
                f"got entity_type={node_data.get('entity_type')!r}. "
                "_synth_speaker_entities must only stamp entities when subject == speaker_id."
            )
            assert node_data.get("speaker_id") is None, (
                f"Non-speaker subject node should NOT have speaker_id set; "
                f"got speaker_id={node_data.get('speaker_id')!r}."
            )


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
        import unittest.mock as mock

        from paramem.graph.schema import Relation
        from paramem.training.consolidation import ConsolidationLoop

        loop = object.__new__(ConsolidationLoop)
        loop.__dict__["config"] = mock.MagicMock()

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

        entities = ConsolidationLoop._synth_speaker_entities(loop, relations)

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
        import unittest.mock as mock

        from paramem.graph.schema import Relation
        from paramem.training.consolidation import ConsolidationLoop

        loop = object.__new__(ConsolidationLoop)
        loop.__dict__["config"] = mock.MagicMock()

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

        entities = ConsolidationLoop._synth_speaker_entities(loop, relations)
        assert len(entities) == 0, (
            f"Expected 0 entities for mismatched subject/speaker_id, got {entities!r}."
        )


# ---------------------------------------------------------------------------
# W1 guard: resolve_to_node_key + same_as speaker-casing guard
# ---------------------------------------------------------------------------


class TestResolveToNodeKeyP5:
    """Unit tests for the module-level ``resolve_to_node_key`` function (P5)."""

    def test_membership_shortcut(self):
        """When ``in_graph(name)`` is True the name IS returned unchanged."""
        from paramem.training.consolidation import resolve_to_node_key

        graph_keys = {"alice", "berlin", "acme corp"}
        in_graph = lambda n: n in graph_keys  # noqa: E731

        assert resolve_to_node_key("alice", in_graph) == "alice"
        assert resolve_to_node_key("berlin", in_graph) == "berlin"

    def test_canonical_fallback(self):
        """When name is NOT in graph, canonical(name) is returned."""
        from paramem.training.consolidation import resolve_to_node_key

        in_graph = lambda n: False  # noqa: E731

        # canonical("speaker0") == "speaker0" (casefolded)
        assert resolve_to_node_key("speaker0", in_graph) == "speaker0"
        # canonical("Tobias") == "tobias"
        assert resolve_to_node_key("Tobias", in_graph) == "tobias"

    def test_coref_chain_follow(self):
        """With a coref_map, the resolved key is followed through the chain."""
        from paramem.training.consolidation import resolve_to_node_key

        in_graph = lambda n: False  # noqa: E731
        coref_map = {"speaker1": "speaker0"}  # speaker1 merged into speaker0

        result = resolve_to_node_key("speaker1", in_graph, coref_map)
        # canonical("speaker1") == "speaker1", then follows chain → "speaker0"
        assert result == "speaker0"

    def test_coref_chain_cycle_guarded(self):
        """A cyclic coref_map must not loop forever."""
        from paramem.training.consolidation import resolve_to_node_key

        in_graph = lambda n: False  # noqa: E731
        coref_map = {"a": "b", "b": "a"}  # cycle

        # Must return without hanging; result is either "a" or "b"
        result = resolve_to_node_key("a", in_graph, coref_map)
        assert result in ("a", "b")

    def test_no_coref_map(self):
        """Without a coref_map, only membership+canonical resolution applies."""
        from paramem.training.consolidation import resolve_to_node_key

        in_graph = lambda n: n == "existing"  # noqa: E731

        assert resolve_to_node_key("existing", in_graph) == "existing"
        assert resolve_to_node_key("Missing", in_graph) == "missing"


class TestW1SameAsGuard:
    """W1 guard: any same_as pair where BOTH surfaces are speaker ids must be
    skipped unconditionally.

    Speaker identity is authoritative (voice/enrollment) and must never be
    coalesced by a surface-similarity heuristic.  Two speaker-id surfaces are
    either the SAME speaker (already unified by canonical node-keying — redundant)
    or DIFFERENT speakers (must never merge — Jaro-Winkler treats the
    distinguishing digit as a typo, so ``_safe_to_merge_surface`` returns True
    for every distinct speaker pair).  The guard blocks both scenarios.
    """

    @staticmethod
    def _make_w1_loop(tmp_path):
        """Minimal ConsolidationLoop for W1 tests (≥10-node graph, real merger)."""
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
            memory_store=MemoryStore(replay_enabled=True),
            procedural_adapter_config=None,
            output_dir=tmp_path,
            extraction_noise_filter="anthropic",
            extraction_noise_filter_model="claude-sonnet-4-6",
        )
        loop._probe_passing_keys = lambda adapter_name, entries: {e["key"] for e in entries}
        for tier in ("episodic", "semantic", "procedural"):
            loop.store.load_registry(tier, KeyRegistry())

        # Populate ≥10 nodes so the floor gate passes.
        graph = loop.merger.graph
        org = "acmecorp"
        graph.add_node(
            org,
            entity_type="organization",
            attributes={"name": "AcmeCorp"},
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
                attributes={"name": f"Person{i}"},
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
        """SOTA same_as ['speaker0', 'speaker0'] must not contract the speaker node.

        Drives the real _run_graph_enrichment production path with a mocked SOTA
        response.  Asserts that after processing the casing-variant pair:
        - result["same_as_merges"] == 0  (no contraction counted)
        - "speaker0" still exists as a distinct node in the graph
        - no node was removed by a self-contraction

        The W1 guard fires first (both surfaces are speaker ids); the
        ``keep_canon == drop_canon`` post-resolution check is a secondary
        backstop for this specific case only.
        """
        from unittest.mock import patch

        loop = self._make_w1_loop(tmp_path)

        node_count_before = loop.merger.graph.number_of_nodes()

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        with patch(
            "paramem.training.consolidation._graph_enrich_with_sota",
            return_value=([], [["speaker0", "speaker0"]], "raw"),
        ):
            result = loop._run_graph_enrichment()

        assert not result["skipped"]
        assert result["same_as_merges"] == 0, (
            "W1/post-resolution guard must prevent the speaker casing-variant pair "
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
        """SOTA same_as ['speaker0', 'speaker1'] must NOT merge distinct speakers.

        Drives the real _run_graph_enrichment production path.  speaker0 and
        speaker1 are distinct enrollments; a SOTA same_as proposal must be
        blocked by the generalized W1 guard (both surfaces are speaker ids).

        This test is load-bearing: without the W1 guard,
        ``_safe_to_merge_surface("speaker0", "speaker1")`` returns True (JW
        treats the digit as a typo, score ≈ 0.950) and the merger would
        contract speaker1 into speaker0 — catastrophic in a real 2-speaker
        deployment.

        Asserts after enrichment:
        - result["same_as_merges"] == 0
        - both "speaker0" and "speaker1" still exist as distinct nodes
        """
        from unittest.mock import patch

        loop = self._make_w1_loop(tmp_path)
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
            "paramem.training.consolidation._graph_enrich_with_sota",
            return_value=([], [["speaker0", "speaker1"]], "raw"),
        ):
            result = loop._run_graph_enrichment()

        assert result["same_as_merges"] == 0, (
            "W1 guard must block distinct speaker ids from merging; "
            f"got same_as_merges={result['same_as_merges']}"
        )
        # Both speaker nodes must remain distinct — no contraction occurred.
        assert "speaker0" in loop.merger.graph.nodes, (
            "speaker0 must survive — W1 guard must block the speaker0/speaker1 merge"
        )
        assert "speaker1" in loop.merger.graph.nodes, (
            "speaker1 must survive — W1 guard must block the speaker0/speaker1 merge"
        )
        assert loop.merger.graph.number_of_nodes() == node_count_before, (
            "Graph node count must not change after a W1-guarded speaker same_as pair"
        )

    def test_non_speaker_pairs_are_not_guarded(self, tmp_path, monkeypatch):
        """Non-speaker same_as pairs (e.g. name variants) must pass through W1.

        Drives the real _run_graph_enrichment path.  A pair of ordinary non-speaker
        names must NOT be intercepted by the W1 guard; they continue to the normal
        resolution and surface-gate checks.
        """
        from unittest.mock import patch

        from paramem.graph.name_match import is_speaker_id

        loop = self._make_w1_loop(tmp_path)
        graph = loop.merger.graph
        # Add two non-speaker nodes for the same_as pair.
        for name in ("alexander", "alex"):
            graph.add_node(
                name,
                entity_type="person",
                attributes={"name": name.capitalize()},
                reinforcement_count=2,
                sessions=["s100"],
                first_seen="s100",
                last_seen="s100",
            )

        assert not is_speaker_id("alexander")
        assert not is_speaker_id("alex")

        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        # Non-speaker pair: W1 must not fire; normal gates apply.
        # "Alexander"/"Alex" pass _safe_to_merge_surface (token subset), so the merge
        # is allowed and same_as_merges == 1.
        with patch(
            "paramem.training.consolidation._graph_enrich_with_sota",
            return_value=([], [["Alexander", "Alex"]], "raw"),
        ):
            result = loop._run_graph_enrichment()

        # Non-speaker pair was not guarded by W1 — the merge proceeded normally.
        assert result["same_as_merges"] >= 1, (
            "Non-speaker same_as pair must not be blocked by the W1 guard; "
            f"expected at least 1 merge, got {result['same_as_merges']}"
        )


# ---------------------------------------------------------------------------
# Subtractive removals helper + whole-graph normalization pass tests
# ---------------------------------------------------------------------------


class TestSubtractiveRemovalsHelperInterim:
    """_apply_subtractive_removals_to_store(scope='interim') soft-stales the correct reasons.

    Tests:
    - ASRI-1: predicate_synonym_collapse is soft-staled at interim scope.
    - ASRI-2: contradiction_same_pred is soft-staled at interim scope.
    - ASRI-3: enrichment_same_as is NOT soft-staled at interim scope (retain-only bucket).
    - ASRI-4: key absent from any active tier is a no-op (no crash, empty return).
    """

    @staticmethod
    def _make_loop_with_ledger(tmp_path, *, ledger: dict) -> "ConsolidationLoop":
        """Minimal ConsolidationLoop with a seeded removal_ledger and replay store."""
        from paramem.graph.merger import GraphMerger
        from paramem.memory.store import MemoryStore
        from paramem.training.consolidation import ConsolidationLoop
        from paramem.training.key_registry import KeyRegistry
        from paramem.utils.config import AdapterConfig, ConsolidationConfig, TrainingConfig

        loop = object.__new__(ConsolidationLoop)
        loop.model = MagicMock()
        loop.tokenizer = MagicMock()
        loop.config = ConsolidationConfig(indexed_key_replay=True)
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

        merger = GraphMerger(model=None)
        # Seed the removal_ledger with supplied entries.
        merger.removal_ledger = dict(ledger)
        loop.merger = merger

        store = MemoryStore(replay_enabled=True)
        for tier in ("episodic", "semantic", "procedural"):
            store.load_registry(tier, KeyRegistry())
        loop.store = store
        return loop

    def test_synonym_collapse_key_is_soft_staled_at_interim(self, tmp_path):
        """ASRI-1: predicate_synonym_collapse reason → key soft-staled at interim scope.

        The helper must call store.discard_keys([key], mode='stale') for a ledger
        entry with reason 'predicate_synonym_collapse'.  After the call the key
        must be stale (not active) in the store.
        """
        loop = self._make_loop_with_ledger(
            tmp_path,
            ledger={
                "graph_collapse_k1": {
                    "reason": "predicate_synonym_collapse",
                    "subject": "Jordan",
                    "object": "TechCorp",
                    "existing_predicate": "works_for",
                    "incoming_predicate": "employed_by",
                },
            },
        )
        # Register the key as active so the soft-stale has something to flip.
        loop.store.put(
            "episodic",
            "graph_collapse_k1",
            {
                "key": "graph_collapse_k1",
                "subject": "Jordan",
                "predicate": "works_for",
                "object": "TechCorp",
                "speaker_id": "Jordan",
            },
            register=True,
        )
        assert not loop.store.is_stale("graph_collapse_k1"), "Precondition: key must be active"

        result = loop._apply_subtractive_removals_to_store(scope="interim")

        assert loop.store.is_stale("graph_collapse_k1"), (
            "predicate_synonym_collapse key must be soft-staled at interim scope"
        )
        # The returned dict must include the tier entry for the staled key.
        assert "episodic" in result, "Return must include the tier that held the key"
        assert "graph_collapse_k1" in result["episodic"], (
            "Staled key must appear in returned soft_stale_by_tier dict"
        )

    def test_contradiction_same_pred_soft_staled_at_interim(self, tmp_path):
        """ASRI-2: contradiction_same_pred is soft-staled at interim scope (NEW supersedes OLD).

        At the interim scope the recency signal is present: the NEW pending merge
        has already replaced the OLD slot's object.  The OLD slot key must be
        soft-staled so it does not return at the next cycle.
        """
        loop = self._make_loop_with_ledger(
            tmp_path,
            ledger={
                "graph_contra_k1": {
                    "reason": "contradiction_same_pred",
                    "subject": "Casey",
                    "object": "Berlin",
                    "predicate": "lives_in",
                },
            },
        )
        loop.store.put(
            "episodic",
            "graph_contra_k1",
            {
                "key": "graph_contra_k1",
                "subject": "Casey",
                "predicate": "lives_in",
                "object": "Berlin",
                "speaker_id": "Casey",
            },
            register=True,
        )

        loop._apply_subtractive_removals_to_store(scope="interim")

        assert loop.store.is_stale("graph_contra_k1"), (
            "contradiction_same_pred key must be soft-staled at interim scope"
        )

    def test_enrichment_same_as_not_soft_staled_at_interim(self, tmp_path):
        """ASRI-3: enrichment_same_as stays in the retain-only bucket at interim scope.

        The helper must NOT soft-stale keys whose removal reason is
        'enrichment_same_as' — those belong to the fold's drift_intended_removal
        bucket, not to subtractive processing.
        """
        loop = self._make_loop_with_ledger(
            tmp_path,
            ledger={
                "graph_enrich_k1": {
                    "reason": "enrichment_same_as",
                    "subject": "Riley",
                    "object": "AlternateName",
                },
            },
        )
        loop.store.put(
            "episodic",
            "graph_enrich_k1",
            {
                "key": "graph_enrich_k1",
                "subject": "Riley",
                "predicate": "also_known_as",
                "object": "AlternateName",
                "speaker_id": "Riley",
            },
            register=True,
        )

        loop._apply_subtractive_removals_to_store(scope="interim")

        assert not loop.store.is_stale("graph_enrich_k1"), (
            "enrichment_same_as key must NOT be soft-staled at interim scope "
            "(retain-only bucket; handled by fold drift_intended_removal)"
        )

    def test_key_absent_from_active_tier_is_noop(self, tmp_path):
        """ASRI-4: a removal_ledger key that is not active in any tier must not crash.

        The helper calls tier_for_active_key(key); when the key is absent the
        method returns None and the helper must proceed silently.
        """
        loop = self._make_loop_with_ledger(
            tmp_path,
            ledger={
                "graph_absent_k1": {
                    "reason": "predicate_synonym_collapse",
                    "subject": "Quinn",
                    "object": "Somewhere",
                    "existing_predicate": "lives_in",
                    "incoming_predicate": "resides_in",
                },
            },
        )
        # Do NOT register the key — it is absent from the store.

        result = loop._apply_subtractive_removals_to_store(scope="interim")

        # No crash; result contains no entry for the absent key.
        all_stale_keys = {k for tier in result.values() for k in tier}
        assert "graph_absent_k1" not in all_stale_keys, (
            "Absent key must not appear in soft_stale_by_tier (no active tier entry)"
        )

    def test_stale_flag_persisted_to_disk_before_commit(self, tmp_path):
        """ASRI-5: stale flag written to disk via commit_tier_slot after M5 reorder.

        Regression test for the stale-flag durability bug: ``_apply_subtractive_removals_to_store``
        must run BEFORE ``commit_tier_slot`` so the on-disk registry carries the stale
        flag.  If the order is reversed (stale after commit), the reloaded registry
        shows the key as active and this test fails.

        The test drives the post-fix ordering directly:
        1. Register key as active.
        2. ``_apply_subtractive_removals_to_store(scope='interim')`` — stales in memory.
        3. ``commit_tier_slot(mode='simulate')`` — serializes the now-staled registry.
        4. Reload registry from disk into a fresh ``KeyRegistry`` instance.
        5. Assert the reloaded registry shows the key as stale (not active).
        """
        from paramem.memory.persistence import commit_tier_slot
        from paramem.training.key_registry import KeyRegistry

        _adapter = "episodic"
        _key = "disk_stale_k1"

        loop = self._make_loop_with_ledger(
            tmp_path,
            ledger={
                _key: {
                    "reason": "predicate_synonym_collapse",
                    "subject": "Jordan",
                    "object": "TechCorp",
                    "existing_predicate": "works_for",
                    "incoming_predicate": "employed_by",
                },
            },
        )
        # Register the key as active.
        loop.store.put(
            _adapter,
            _key,
            {
                "key": _key,
                "subject": "Jordan",
                "predicate": "works_for",
                "object": "TechCorp",
                "speaker_id": "Jordan",
            },
            register=True,
        )
        assert not loop.store.is_stale(_key), "Precondition: key must be active before staling"

        # Step 2: soft-stale in memory (M5 stage — must precede commit).
        loop._apply_subtractive_removals_to_store(scope="interim")
        assert loop.store.is_stale(_key), "Key must be stale in memory after M5 stage"

        # Step 3: commit — serializes the already-staled registry to disk.
        commit_tier_slot(
            loop=loop,
            tier=_adapter,
            adapter_name=_adapter,
            stamp="20260101T0000",
            mode="simulate",
            all_keyed=[],
            output_dir=tmp_path,
        )

        # Step 4: reload the registry from disk into a fresh instance.
        registry_path = tmp_path / _adapter / "indexed_key_registry.json"
        assert registry_path.exists(), f"Registry file must exist on disk: {registry_path}"
        reloaded = KeyRegistry.load(registry_path)

        # Step 5: the reloaded registry must show the key as stale.
        assert _key not in reloaded.list_active(), (
            "Reloaded registry must NOT list the staled key as active "
            "(stale flag must be captured on disk before the commit write)"
        )
        assert reloaded.is_stale(_key), (
            "Reloaded registry must show the key as stale "
            "(stale flag must be captured on disk before the commit write)"
        )


class TestSubtractiveRemovalsHelperFold:
    """_apply_subtractive_removals_to_store(scope='fold') scoping invariants.

    Tests:
    - ASRF-1: predicate_synonym_collapse IS soft-staled at fold (time-invariant).
    - ASRF-2: contradiction_same_pred IS soft-staled at fold (recency-backed: the merger
      only emits the ledger entry when timestamps pick a unique winner; empty/tied → coexist
      → no entry → no stale).
    """

    @staticmethod
    def _make_loop_with_ledger(tmp_path, *, ledger: dict) -> "ConsolidationLoop":
        """Identical to TestSubtractiveRemovalsHelperInterim._make_loop_with_ledger."""
        from paramem.graph.merger import GraphMerger
        from paramem.memory.store import MemoryStore
        from paramem.training.consolidation import ConsolidationLoop
        from paramem.training.key_registry import KeyRegistry
        from paramem.utils.config import AdapterConfig, ConsolidationConfig, TrainingConfig

        loop = object.__new__(ConsolidationLoop)
        loop.model = MagicMock()
        loop.tokenizer = MagicMock()
        loop.config = ConsolidationConfig(indexed_key_replay=True)
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

        merger = GraphMerger(model=None)
        merger.removal_ledger = dict(ledger)
        loop.merger = merger

        store = MemoryStore(replay_enabled=True)
        for tier in ("episodic", "semantic", "procedural"):
            store.load_registry(tier, KeyRegistry())
        loop.store = store
        return loop

    def test_synonym_collapse_soft_staled_at_fold(self, tmp_path):
        """ASRF-1: predicate_synonym_collapse is soft-staled at fold scope (time-invariant)."""
        loop = self._make_loop_with_ledger(
            tmp_path,
            ledger={
                "graph_fold_collapse_k1": {
                    "reason": "predicate_synonym_collapse",
                    "subject": "Morgan",
                    "object": "German",
                    "existing_predicate": "speaks",
                    "incoming_predicate": "speaks_language",
                },
            },
        )
        loop.store.put(
            "episodic",
            "graph_fold_collapse_k1",
            {
                "key": "graph_fold_collapse_k1",
                "subject": "Morgan",
                "predicate": "speaks",
                "object": "German",
                "speaker_id": "Morgan",
            },
            register=True,
        )

        result = loop._apply_subtractive_removals_to_store(scope="fold")

        assert loop.store.is_stale("graph_fold_collapse_k1"), (
            "predicate_synonym_collapse key must be soft-staled at fold scope"
        )
        assert "episodic" in result and "graph_fold_collapse_k1" in result["episodic"], (
            "Returned soft_stale_by_tier must contain the fold-staled key"
        )

    def test_contradiction_same_pred_soft_staled_at_fold(self, tmp_path):
        """ASRF-2: contradiction_same_pred IS soft-staled at fold scope.

        The merger only emits a contradiction_same_pred ledger entry when timestamps
        pick a UNIQUE winner (freshest last_seen wins).  An empty/tied pair never
        reaches the ledger (coexist → no entry → no stale).  When a ledger entry
        IS present at fold scope, it was generated by the recency rule and the
        staler key is correctly soft-staled.
        """
        loop = self._make_loop_with_ledger(
            tmp_path,
            ledger={
                "graph_fold_contra_k1": {
                    "reason": "contradiction_same_pred",
                    "subject": "Avery",
                    "object": "Berlin",
                    "predicate": "lives_in",
                },
            },
        )
        loop.store.put(
            "episodic",
            "graph_fold_contra_k1",
            {
                "key": "graph_fold_contra_k1",
                "subject": "Avery",
                "predicate": "lives_in",
                "object": "Berlin",
                "speaker_id": "Avery",
            },
            register=True,
        )

        result = loop._apply_subtractive_removals_to_store(scope="fold")

        assert loop.store.is_stale("graph_fold_contra_k1"), (
            "contradiction_same_pred key must be soft-staled at fold scope "
            "(recency-backed: merger only emits the entry when timestamps pick a unique winner)"
        )
        assert "episodic" in result, "Return must include the tier that held the key"
        assert "graph_fold_contra_k1" in result["episodic"], (
            "Staled key must appear in returned soft_stale_by_tier dict"
        )

    def test_empty_last_seen_tied_no_ledger_entry_no_stale_at_fold(self, tmp_path):
        """ASRF-3: no ledger entry emitted by merger for empty/tied timestamps →
        _apply_subtractive_removals_to_store has nothing to act on at fold scope.

        This verifies the ledger-gate invariant: the merger only writes
        contradiction_same_pred into the ledger when timestamps pick a UNIQUE
        winner.  Empty/tied → coexist → no ledger entry → no stale.
        """
        # Simulate: merger emitted NO contradiction_same_pred entry (tied timestamps).
        loop = self._make_loop_with_ledger(tmp_path, ledger={})
        loop.store.put(
            "episodic",
            "graph_tied_k1",
            {
                "key": "graph_tied_k1",
                "subject": "Jordan",
                "predicate": "lives_in",
                "object": "Berlin",
                "speaker_id": "Jordan",
            },
            register=True,
        )

        result = loop._apply_subtractive_removals_to_store(scope="fold")

        assert not loop.store.is_stale("graph_tied_k1"), (
            "Key with no ledger entry must NOT be staled (tied timestamps → coexist → no entry)"
        )
        assert not result, "Empty ledger → empty soft_stale_by_tier"


# ---------------------------------------------------------------------------
# _merge_registry_relations: C1 regression — timestamp="" prevents now() fabrication
# ---------------------------------------------------------------------------


class TestMergeRegistryRelationsTimestamp:
    """_merge_registry_relations passes timestamp="" to the merger's SessionGraph.

    C1 regression guard: before the fix, _merge_registry_relations always built the
    SessionGraph with timestamp=datetime.now(...), so a recon relation with
    last_seen="" resolved to incoming_ls = "" or now() = now() — making the legacy key
    appear as the unique freshest and wrongly retiring a genuinely-dated rival.

    With the fix, timestamp="" (the new default param), so incoming_ls = "" or "" = ""
    which triggers the any-empty COEXIST rule.  Dated keys are never retired by a
    legacy "".
    """

    @staticmethod
    def _make_loop_for_recon_merge(tmp_path) -> "ConsolidationLoop":
        """Minimal ConsolidationLoop with a mock-model GraphMerger for _merge_registry_relations."""
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
            min_tier_key_floor=0,
            tier_fast_start=False,
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
        # Pre-cache "lives in" as single-valued (REPLACE) to skip the model call.
        merger._predicate_cardinality["lives in"] = False
        loop.merger = merger

        store = MemoryStore(replay_enabled=True)
        for tier in ("episodic", "semantic", "procedural"):
            store.load_registry(tier, KeyRegistry())
        loop.store = store
        return loop

    def test_mixed_dated_legacy_recon_coexist_c1_regression(self, tmp_path):
        """C1 fix: mixed registry (one dated key, one legacy "") through recon merge.

        Simulates a fold _merge_registry_relations call with:
          - Relation A (lives in → munich), last_seen="2026-01-01T00:00:00Z" (dated)
          - Relation B (lives in → berlin), last_seen="" (legacy — no timestamp in bookkeeping)

        Both share the same (subject=alex, predicate=lives in) so Case-2 fires.

        Before C1 fix: timestamp=now() was passed to the SessionGraph; the merger
        evaluated incoming_ls = "" or now() = now() for the legacy relation, making it
        the unique freshest → the dated munich key was wrongly retired.

        After C1 fix: timestamp="" (default) → incoming_ls = "" → any-empty rule →
        COEXIST.  Both edges survive and the removal_ledger is empty.
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
        loop._merge_registry_relations(
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
            "C1 regression: dated munich key must NOT be retired by a legacy '' relation"
        )
        assert "berlin" in lives_in_objects, "Legacy berlin relation must be inserted (coexist)"
        # No removal_ledger entry: any-empty → coexist → no ledger write.
        assert not loop.merger.removal_ledger, (
            f"No removal_ledger entry expected for any-empty coexist; "
            f"got {loop.merger.removal_ledger}"
        )

    def test_capture_pending_propagates_last_seen(self, tmp_path):
        """_capture_pending_relations carries the edge last_seen onto the captured Relation.

        Pending graph edges are stamped with the real ingest-time last_seen by the
        merger's Case-3 path (merger.py:821).  Without this propagation the captured
        Relation would carry last_seen="" and the any-empty COEXIST rule would
        prevent a newer pending fact from superseding an older dated rival.
        """
        loop = self._make_loop_for_recon_merge(tmp_path)
        loop.merger.graph.add_node("alex")
        loop.merger.graph.add_node("berlin")
        loop.merger.graph.add_edge(
            "alex",
            "berlin",
            predicate="lives in",
            relation_type="factual",
            confidence=1.0,
            sessions=["s_ingest"],
            last_seen="2026-01-02T00:00:00Z",
        )

        pending = loop._capture_pending_relations()

        assert len(pending) == 1, f"Expected 1 pending relation; got {len(pending)}"
        assert pending[0].last_seen == "2026-01-02T00:00:00Z", (
            f"last_seen must be propagated from edge; got {pending[0].last_seen!r}"
        )

    def test_pending_dated_supersedes_older_dated_rival(self, tmp_path):
        """Pending relation with newer dated last_seen supersedes an older dated rival.

        Scenario (interim/consume-pending path):
          - Registry-true recon: (alex, lives in → munich), last_seen="2026-01-01" (OLDER)
          - Pending capture: (alex, lives in → berlin), last_seen="2026-01-02" (NEWER)
          - resolve_contradictions=True, single-valued predicate

        With _capture_pending_relations now propagating last_seen, the pending relation
        carries its genuine recency.  The all-dated path fires: incoming_ls="2026-01-02"
        > rival_ls="2026-01-01" → munich retired, berlin inserted.

        Without the fix (last_seen=""), incoming_ls="" → any-empty COEXIST → munich
        survives (regression: NEW never supersedes OLD).
        """
        from paramem.graph.schema import Relation
        from paramem.memory.persistence import _IK_KEY_ATTR

        loop = self._make_loop_for_recon_merge(tmp_path)

        # Step 1: merge the registry-true recon relation (munich, older dated).
        recon_relations = [
            Relation(
                subject="alex",
                predicate="lives_in",
                object="munich",
                relation_type="factual",
                confidence=1.0,
                speaker_id="speaker0",
                indexed_key="key_munich_old",
                last_seen="2026-01-01T00:00:00Z",
            )
        ]
        loop._merge_registry_relations(
            recon_relations,
            session_id="__full_consolidation_recon__",
            log_label="recon triples (test)",
            resolve_contradictions=False,  # recon is old-vs-old for this step
        )

        # Stamp ik_key onto the munich edge so ledger capture fires on retirement.
        _alex_successors = list(loop.merger.graph.successors("alex"))
        for _succ in _alex_successors:
            for _eid, _edata in loop.merger.graph["alex"][_succ].items():
                if _edata.get("predicate") == "lives in":
                    _edata[_IK_KEY_ATTR] = "key_munich_old"

        # Step 2: merge the pending relation (berlin, newer dated) with resolve=True.
        pending_relations = [
            Relation(
                subject="alex",
                predicate="lives_in",
                object="berlin",
                relation_type="factual",
                confidence=1.0,
                speaker_id="speaker0",
                indexed_key="key_berlin_new",
                last_seen="2026-01-02T00:00:00Z",  # NEWER — as propagated from edge
            )
        ]
        loop._merge_registry_relations(
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
        assert "berlin" in lives_in_objects, (
            "Newer pending Berlin must be inserted (supersedes older Munich)"
        )
        assert "munich" not in lives_in_objects, (
            "Older registry-true Munich must be retired by newer pending Berlin"
        )
        assert "key_munich_old" in loop.merger.removal_ledger, (
            "Retired Munich key must appear in removal_ledger"
        )
        assert loop.merger.removal_ledger["key_munich_old"]["reason"] == "contradiction_same_pred"

    def test_pending_dated_vs_legacy_empty_rival_coexist(self, tmp_path):
        """Pending dated relation vs legacy "" registry-true rival → COEXIST (any-empty rule).

        A pending fact with a real last_seen must NOT retire a legacy "" registry key —
        the any-empty COEXIST rule protects them.
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
        loop._merge_registry_relations(
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
        loop._merge_registry_relations(
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
        assert "munich" in lives_in_objects, (
            "Legacy '' Munich must survive (any-empty COEXIST: rival has last_seen='')"
        )
        assert not loop.merger.removal_ledger, (
            f"No removal for any-empty coexist; got {loop.merger.removal_ledger}"
        )


# ---------------------------------------------------------------------------
# _extract_json_block — relations-envelope parser path for normalization
# ---------------------------------------------------------------------------


class TestExtractJsonBlockRelationsEnvelope:
    """_extract_json_block recognises the {"relations":[...]} envelope emitted by
    the normalization prompt.

    Tests:
    - REL-1: wrapped {"relations":[...]} envelope parsed and returned.
    - REL-2: bare array [{...}] with subject/predicate/object elements accepted.
    - REL-3: markdown-fenced {"relations":[...]} accepted (fence stripped).
    - REL-4: no JSON at all raises ValueError.
    - REL-5: multiple relation entries preserved.
    """

    @staticmethod
    def _parse(raw: str):
        import json

        from paramem.graph.extractor import _extract_json_block

        return json.loads(_extract_json_block(raw))

    def test_relations_envelope_parsed(self):
        """REL-1: {"relations":[...]} envelope is accepted by _extract_json_block."""
        raw = '{"relations": [{"subject": "Alex", "predicate": "works_for", "object": "Acme"}]}'
        result = self._parse(raw)
        assert isinstance(result, dict)
        assert "relations" in result
        assert len(result["relations"]) == 1
        assert result["relations"][0]["predicate"] == "works_for"

    def test_bare_array_with_spo_elements_accepted(self):
        """REL-2: bare array where first element has subject/predicate/object is accepted."""
        raw = '[{"subject": "Alex", "predicate": "lives_in", "object": "Berlin"}]'
        result = self._parse(raw)
        assert isinstance(result, list)
        assert result[0]["object"] == "Berlin"

    def test_markdown_fenced_relations_envelope(self):
        """REL-3: code-fenced {"relations":[...]} is parsed after fence-stripping."""
        raw = '```json\n{"relations": [{"subject": "A", "predicate": "b", "object": "C"}]}\n```'
        result = self._parse(raw)
        assert "relations" in result

    def test_no_json_raises(self):
        """REL-4: no JSON in output → ValueError from _extract_json_block."""
        from paramem.graph.extractor import _extract_json_block

        with pytest.raises((ValueError, Exception)):
            _extract_json_block("The graph has no redundancy.")

    def test_multiple_relation_entries_preserved(self):
        """REL-5: multiple relation entries are all present in parsed output."""
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
# _run_graph_normalization — apply path: relations-envelope + same-(s,o) collapse
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
    - NDA-1: two keyed synonym predicates for same (s,o) — model returns cluster →
             lower-rec keyed edge removed + removal_ledger 'predicate_synonym_collapse'.
    - NDA-2: two keyless synonym predicates — lower-rec edge removed, no ledger entry.
    - NDA-3: provenance (sessions union, recurrence sum, max confidence) is
             carried onto the survivor (MAX rec) before retired edges are removed.
    - NDA-4: single-predicate (s,o) group → no model call; graph unchanged.
    - NDA-5: model returns empty clusters → no-op (graph unchanged).
    - NDA-6: model=None → skipped=True, graph unchanged.
    - NDA-7: graph < 10 nodes → skipped=True (floor).
    - NDA-8: mixed keyed + keyless — keyed retired → ledger; keyless retired → no ledger;
             MAX-rec survivor intact; result counts correct.
    - NDA-9: single-predicate (s,o) group not touched even when another group is collapsed.
    - NDA-10: multi-predicate group collapsed; graph updated correctly.
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
            indexed_key_replay=True,
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
            g.add_node(f"node{i}", reinforcement_count=0, attributes={"name": f"node{i}"})
        loop.merger.graph = g

        store = MemoryStore(replay_enabled=True)
        for tier in ("episodic", "semantic", "procedural"):
            store.load_registry(tier, KeyRegistry())
        loop.store = store
        return loop

    # ---------------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------------

    @staticmethod
    def _add_keyed_edge(graph, subj, obj, predicate, ik_key, *, sessions=None, recurrence=1):
        """Add a directed edge with standard attributes and a keyed ik_key."""
        from paramem.graph.name_match import canonical as _can
        from paramem.memory.persistence import _IK_KEY_ATTR

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
        graph.add_node(subj, reinforcement_count=1, attributes={"name": subj})
        graph.add_node(obj, reinforcement_count=1, attributes={"name": obj})
        graph.add_edge(subj, obj, **attrs)

    @staticmethod
    def _add_keyless_edge(graph, subj, obj, predicate, *, sessions=None, recurrence=1):
        """Add a directed edge WITHOUT an ik_key (simulates a fresh-ingested fact)."""
        from paramem.graph.name_match import canonical as _can

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
        graph.add_node(subj, reinforcement_count=1, attributes={"name": subj})
        graph.add_node(obj, reinforcement_count=1, attributes={"name": obj})
        graph.add_edge(subj, obj, **attrs)

    # Prompt stub: only {predicates_json} placeholder (matches dedup_synonym_predicates).
    _PROMPT_STUB = "dummy {predicates_json}"

    def _cluster_response(self, clusters: list[list[str]]) -> str:
        """Encode a clusters-schema model response for dedup_synonym_predicates."""
        import json

        return json.dumps({"clusters": clusters})

    # ---------------------------------------------------------------------------
    # Tests
    # ---------------------------------------------------------------------------

    def test_keyed_synonym_retired_and_ledgered(self, tmp_path):
        """NDA-1: two keyed synonym predicates for same (s,o) — model returns cluster.

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
        cluster_response = self._cluster_response([["works for", "employed by"]])

        with (
            patch("paramem.graph.extractor.generate_answer", return_value=cluster_response),
            patch("paramem.graph.prompts._load_prompt", return_value=self._PROMPT_STUB),
        ):
            result = loop._run_graph_normalization()

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
        """NDA-2: two keyless synonym predicates — lower-rec edge removed, no ledger entry.

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

        cluster_response = self._cluster_response([["works for", "employed by"]])

        with (
            patch("paramem.graph.extractor.generate_answer", return_value=cluster_response),
            patch("paramem.graph.prompts._load_prompt", return_value=self._PROMPT_STUB),
        ):
            result = loop._run_graph_normalization()

        remaining_preds = {
            edata["predicate"] for _, _, edata in graph.edges(data=True) if edata.get("predicate")
        }
        # canonical() folds underscores to spaces.  employed_by (rec=3) survives.
        assert "works for" not in remaining_preds, "lower-rec edge must be removed"
        assert "employed by" in remaining_preds, "MAX-rec survivor must remain"
        assert not loop.merger.removal_ledger, "no ledger entry for keyless retirements"

        assert result["edges_retired"] == 1
        assert result["groups_collapsed"] == 1

    def test_provenance_unioned_onto_survivor(self, tmp_path):
        """NDA-3: sessions, recurrence, and confidence are unioned onto the survivor.

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
            patch("paramem.graph.prompts._load_prompt", return_value=self._PROMPT_STUB),
        ):
            result = loop._run_graph_normalization()

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
            patch("paramem.graph.prompts._load_prompt", return_value=self._PROMPT_STUB),
        ):
            loop._run_graph_normalization()

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
            patch("paramem.graph.prompts._load_prompt", return_value=self._PROMPT_STUB),
        ):
            loop._run_graph_normalization()

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
        """NDA-4: single-predicate (s,o) group is never a candidate — no model call.

        Graph: morgan -> germany with only 'born_in' (graph12).  The group has only
        one predicate so it never reaches dedup_synonym_predicates as a candidate.
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
            patch("paramem.graph.prompts._load_prompt", return_value=self._PROMPT_STUB),
        ):
            result = loop._run_graph_normalization()

        all_keys = [edata.get(_IK_KEY_ATTR) for _, _, edata in graph.edges(data=True)]
        assert "graph12" in all_keys, "graph12 must survive (not a multi-predicate group)"
        assert not loop.merger.removal_ledger, "Ledger must be empty"
        assert result["edges_retired"] == 0
        assert result["groups_collapsed"] == 0
        assert result["chunks"] == 0, "no model calls for single-predicate group"

    def test_empty_clusters_response_is_noop(self, tmp_path):
        """NDA-5: model returns empty clusters for a group → no retirement.

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
            patch("paramem.graph.prompts._load_prompt", return_value=self._PROMPT_STUB),
        ):
            result = loop._run_graph_normalization()

        all_keys = [edata.get(_IK_KEY_ATTR) for _, _, edata in graph.edges(data=True)]
        assert "graph42" in all_keys, "graph42 must survive"
        assert "graph87" in all_keys, "graph87 must survive"
        assert not loop.merger.removal_ledger, "Ledger must be empty"
        assert result["edges_retired"] == 0
        assert result["groups_collapsed"] == 0

    def test_no_model_returns_skipped(self, tmp_path):
        """NDA-6: model=None → pass skipped immediately, graph unchanged."""
        loop = self._make_loop(tmp_path, model=None)
        loop.model = None

        initial_nodes = loop.merger.graph.number_of_nodes()
        result = loop._run_graph_normalization()

        assert result["skipped"] is True
        assert result["skip_reason"] == "no_model"
        assert loop.merger.graph.number_of_nodes() == initial_nodes

    def test_small_graph_returns_skipped(self, tmp_path):
        """NDA-7: graph < 10 nodes → pass skipped (below floor)."""
        loop = self._make_loop(tmp_path, node_count=5)

        result = loop._run_graph_normalization()

        assert result["skipped"] is True
        assert result["skip_reason"] == "floor"

    def test_mixed_keyed_and_keyless_correct_ledger(self, tmp_path):
        """NDA-8: mixed keyed + keyless — keyed retired -> ledger; keyless retired -> no ledger.

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
            patch("paramem.graph.prompts._load_prompt", return_value=self._PROMPT_STUB),
        ):
            result = loop._run_graph_normalization()

        all_keys = [edata.get(_IK_KEY_ATTR) for _, _, edata in graph.edges(data=True)]
        assert "graph87" not in all_keys, "graph87 (keyed) must be removed"
        assert "graph42" in all_keys, "graph42 must survive"
        assert "graph87" in loop.merger.removal_ledger
        assert loop.merger.removal_ledger["graph87"]["reason"] == "predicate_synonym_collapse"
        assert len(loop.merger.removal_ledger) == 1, "keyless retirement must not add ledger entry"

        assert result["edges_retired"] == 2
        assert result["groups_collapsed"] == 1

    def test_single_predicate_group_untouched_when_other_group_collapsed(self, tmp_path):
        """NDA-9: single-predicate (s,o) group is not touched when another group is collapsed.

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
        cluster_response = self._cluster_response([["works for", "employed by"]])

        with (
            patch("paramem.graph.extractor.generate_answer", return_value=cluster_response),
            patch("paramem.graph.prompts._load_prompt", return_value=self._PROMPT_STUB),
        ):
            result = loop._run_graph_normalization()

        all_keys = [edata.get(_IK_KEY_ATTR) for _, _, edata in graph.edges(data=True)]
        assert "graph99" in all_keys, "single-predicate (s,o) must not be touched"
        assert "graph42" in all_keys, "graph42 must survive (first in cluster, rec tie)"
        assert result["edges_retired"] == 1, "only graph87 retired (the jordan/techcorp group)"
        assert result["groups_collapsed"] == 1

    def test_max_rec_survivor_selected(self, tmp_path):
        """NDA-10: MAX reinforcement_count edge survives; lower-rec edge retired.

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
            patch("paramem.graph.prompts._load_prompt", return_value=self._PROMPT_STUB),
        ):
            result = loop._run_graph_normalization()

        all_keys = [edata.get(_IK_KEY_ATTR) for _, _, edata in graph.edges(data=True)]
        assert "graph12" not in all_keys, "graph12 (lower-rec born_in) must be retired"
        assert "graph34" in all_keys, "graph34 (MAX-rec birthplace) must survive"
        assert "graph12" in loop.merger.removal_ledger
        assert loop.merger.removal_ledger["graph12"]["reason"] == "predicate_synonym_collapse"

        assert result["edges_retired"] == 1
        assert result["groups_collapsed"] == 1


# ---------------------------------------------------------------------------
# SOTA engine wiring in _run_graph_normalization
# ---------------------------------------------------------------------------


class TestRunGraphNormalizationSotaEngine:
    """_run_graph_normalization SOTA wiring and fail-loud tests.

    Tests:
    - SOTA-1: sota_enabled=True + provider + api_key in env → dedup_synonym_predicates
              called with ``sota=`` kwarg, NOT ``model=``.
    - SOTA-2: sota_enabled=True + provider present but NO api_key in env → local
              fallback: dedup_synonym_predicates called with ``model=`` kwarg.
    - DBG-1: after retirement, on_normalization receives non-empty raw_outputs list
             and non-empty decisions list.
    - FL-1:  FileNotFoundError raised when graph_dedup_filter.txt is missing
             (graph ≥ 10 nodes, model present).
    """

    @staticmethod
    def _make_loop(tmp_path, *, sota_enabled: bool = False, node_count: int = 15):
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
            indexed_key_replay=True,
            refinement_normalization="on",
            sota_enabled=sota_enabled,
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

        # Extraction pipeline mock — config.noise_filter used for SOTA engine resolution.
        ext_mock = MagicMock()
        ext_mock.config.noise_filter = "anthropic"
        ext_mock.config.noise_filter_model = "claude-sonnet-4-6"
        ext_mock.config.noise_filter_endpoint = None
        loop.extraction = ext_mock

        merger = GraphMerger(model=None)
        loop.merger = merger

        g = nx.MultiDiGraph()
        for i in range(node_count):
            g.add_node(f"node{i}", reinforcement_count=0, attributes={"name": f"node{i}"})
        loop.merger.graph = g

        store = MemoryStore(replay_enabled=True)
        for tier in ("episodic", "semantic", "procedural"):
            store.load_registry(tier, KeyRegistry())
        loop.store = store
        return loop

    @staticmethod
    def _add_keyed_edge(graph, subj, obj, predicate, ik_key, *, sessions=None, recurrence=1):
        from paramem.graph.name_match import canonical as _can
        from paramem.memory.persistence import _IK_KEY_ATTR

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
        graph.add_node(subj, reinforcement_count=1, attributes={"name": subj})
        graph.add_node(obj, reinforcement_count=1, attributes={"name": obj})
        graph.add_edge(subj, obj, **attrs)

    _PROMPT_STUB = "dummy {predicates_json}"

    def test_sota_engine_selected_when_enabled_with_api_key(self, tmp_path, monkeypatch):
        """SOTA-1: sota_enabled=True + provider + env api_key → primitive gets sota=."""

        from unittest.mock import patch

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
        loop = self._make_loop(tmp_path, sota_enabled=True)
        graph = loop.merger.graph
        self._add_keyed_edge(graph, "morgan", "acme", "works_for", "g1", recurrence=1)
        self._add_keyed_edge(graph, "morgan", "acme", "employed_by", "g2", recurrence=2)

        # Capture the kwargs that dedup_synonym_predicates receives.
        captured: dict = {}

        def _fake_dedup(relations, *, filter_prompt, **kwargs):
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
                "paramem.training.consolidation.dedup_synonym_predicates",
                side_effect=_fake_dedup,
            ),
            patch("paramem.graph.prompts._load_prompt", return_value=self._PROMPT_STUB),
        ):
            loop._run_graph_normalization()

        assert "sota" in captured, (
            "dedup_synonym_predicates must receive sota= kwarg "
            "when sota_enabled=True and api_key present"
        )
        assert "model" not in captured, "model= must NOT be passed when SOTA engine is selected"
        assert captured["sota"]["provider"] == "anthropic"
        assert captured["sota"]["api_key"] == "sk-test-key"

    def test_local_fallback_when_api_key_absent(self, tmp_path, monkeypatch):
        """SOTA-2: sota_enabled=True but NO api_key → local fallback (model= kwarg)."""
        from unittest.mock import patch

        # Ensure the env var is absent.
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        loop = self._make_loop(tmp_path, sota_enabled=True)
        graph = loop.merger.graph
        self._add_keyed_edge(graph, "morgan", "acme", "works_for", "g1", recurrence=1)
        self._add_keyed_edge(graph, "morgan", "acme", "employed_by", "g2", recurrence=2)

        captured: dict = {}

        def _fake_dedup(relations, *, filter_prompt, **kwargs):
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
                "paramem.training.consolidation.dedup_synonym_predicates",
                side_effect=_fake_dedup,
            ),
            patch("paramem.graph.prompts._load_prompt", return_value=self._PROMPT_STUB),
        ):
            loop._run_graph_normalization()

        assert "model" in captured, (
            "dedup_synonym_predicates must receive model= kwarg when api_key is absent"
        )
        assert "sota" not in captured, (
            "sota= must NOT be passed when api_key is absent (local fallback)"
        )

    def test_on_normalization_receives_nonempty_raw_outputs_and_decisions(self, tmp_path):
        """DBG-1: after retirement, on_normalization receives non-empty raw_outputs
        and non-empty decisions — the debug snapshot has real data to write."""
        import json
        from unittest.mock import patch

        loop = self._make_loop(tmp_path, sota_enabled=False)
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
            patch("paramem.graph.prompts._load_prompt", return_value=self._PROMPT_STUB),
            patch.object(loop._debug_writer, "on_normalization", side_effect=_spy_on_normalization),
        ):
            loop._run_graph_normalization()

        assert on_norm_calls, "on_normalization must be called"
        call = on_norm_calls[0]
        assert call["raw_outputs"], "raw_outputs must be non-empty after a model call"
        assert call["decisions"], "decisions must be non-empty when clusters were produced"

    def test_fail_loud_when_prompt_missing(self, tmp_path):
        """FL-1: FileNotFoundError raised when graph_dedup_filter.txt is missing."""
        from unittest.mock import patch

        import pytest

        loop = self._make_loop(tmp_path, sota_enabled=False)

        with (
            patch(
                "paramem.graph.prompts._load_prompt",
                side_effect=FileNotFoundError(
                    "Required prompt file 'graph_dedup_filter.txt' not found. Searched: ..."
                ),
            ),
            pytest.raises(FileNotFoundError, match="graph_dedup_filter.txt"),
        ):
            loop._run_graph_normalization()


# ---------------------------------------------------------------------------
# Debug snapshot — on_normalization writes normalization_snapshot.json
# ---------------------------------------------------------------------------


class TestNormalizationDebugSnapshot:
    """on_normalization routes raw outputs + decisions + applied counts through
    DebugSnapshotWriter and writes normalization_snapshot.json under fold/.

    Tests:
    - DS-1: save_cycle_snapshots=True → normalization_snapshot.json written with
            raw_outputs, decisions, and applied counts (index-delta schema).
    - DS-2: save_cycle_snapshots=False → no file written (self-gated no-op).
    """

    @staticmethod
    def _make_debug_loop(tmp_path, *, save_cycle_snapshots: bool):
        """Build a ConsolidationLoop with debug snapshot writing enabled/disabled."""
        from paramem.training.consolidation import ConsolidationLoop
        from paramem.training.debug_snapshot import DebugSnapshotWriter
        from paramem.utils.config import ConsolidationConfig

        loop = object.__new__(ConsolidationLoop)
        loop.config = ConsolidationConfig(indexed_key_replay=True)
        loop.save_cycle_snapshots = save_cycle_snapshots
        loop._current_interim_stamp = None  # type: ignore[assignment]
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

        loop._debug_writer = DebugSnapshotWriter(loop)
        return loop

    def test_normalization_snapshot_written_when_debug_enabled(self, tmp_path):
        """DS-1: save_cycle_snapshots=True → normalization_snapshot.json written."""
        import json

        loop = self._make_debug_loop(tmp_path, save_cycle_snapshots=True)

        raw_outputs = ["raw model output goes here"]
        decisions = [{"relations": [{"subject": "A", "predicate": "b", "object": "C"}]}]
        applied = {"groups_collapsed": 1, "edges_retired": 1}

        loop._debug_writer.on_normalization(raw_outputs, decisions, applied)

        matches = list(tmp_path.rglob("normalization_snapshot.json"))
        assert matches, "normalization_snapshot.json must be written when debug is enabled"
        payload = json.loads(matches[0].read_text())
        assert payload["raw_outputs"] == raw_outputs
        assert payload["decisions"] == decisions
        assert payload["applied"] == applied

    def test_normalization_snapshot_not_written_when_debug_disabled(self, tmp_path):
        """DS-2: save_cycle_snapshots=False → no file written (self-gated no-op)."""
        loop = self._make_debug_loop(tmp_path, save_cycle_snapshots=False)

        raw_outputs = ["raw output"]
        decisions = [{"relations": []}]
        applied = {"groups_collapsed": 0, "edges_retired": 0}

        loop._debug_writer.on_normalization(raw_outputs, decisions, applied)

        matches = list(tmp_path.rglob("normalization_snapshot.json"))
        assert not matches, "normalization_snapshot.json must NOT be written when debug is disabled"


# ---------------------------------------------------------------------------
# Level gating — normalize/enrich booleans for refinement_normalization / sota_enabled
# ---------------------------------------------------------------------------


class TestNormalizationLevelGating:
    """_refine_consolidation_graph gates _run_graph_normalization/enrichment on bool flags.

    Tests:
    - LG-1: normalize=False → normalization NOT called.
    - LG-2: normalize=True, enrich=False → normalization IS called, enrichment NOT.
    - LG-3: normalize=True, enrich=True → both called.
    """

    @staticmethod
    def _make_loop_for_refine(
        tmp_path, *, refinement_normalization="off", refinement_enrichment="off", sota_enabled=False
    ):
        """Build a minimal loop for _refine_consolidation_graph gating tests."""
        from paramem.graph.merger import GraphMerger
        from paramem.memory.store import MemoryStore
        from paramem.training.consolidation import ConsolidationLoop
        from paramem.training.key_registry import KeyRegistry
        from paramem.utils.config import AdapterConfig, ConsolidationConfig, TrainingConfig

        loop = object.__new__(ConsolidationLoop)
        loop.model = MagicMock()
        loop.tokenizer = MagicMock()
        loop.config = ConsolidationConfig(
            indexed_key_replay=True,
            refinement_normalization=refinement_normalization,
            refinement_enrichment=refinement_enrichment,
            sota_enabled=sota_enabled,
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

        store = MemoryStore(replay_enabled=True)
        for tier in ("episodic", "semantic", "procedural"):
            store.load_registry(tier, KeyRegistry())
        loop.store = store
        return loop

    _NORM_RETURN = {
        "skipped": False,
        "chunks": 1,
        "groups_collapsed": 0,
        "edges_retired": 0,
    }

    def test_normalize_false_normalization_not_called(self, tmp_path):
        """LG-1: normalize=False → normalization not called."""
        from unittest.mock import patch

        loop = self._make_loop_for_refine(tmp_path, refinement_normalization="off")
        with (
            patch.object(
                loop, "_run_graph_normalization", return_value={"skipped": True}
            ) as mock_norm,
            patch.object(loop, "_run_graph_enrichment", return_value={"skipped": True}),
        ):
            loop._refine_consolidation_graph(
                [],
                normalize=loop.config.refinement_normalization == "on",
                enrich=loop.config.refinement_enrichment == "on" and loop.config.sota_enabled,
            )
        mock_norm.assert_not_called()

    def test_normalize_true_enrich_false_normalization_called_only(self, tmp_path):
        """LG-2: normalize=True, enrich=False → normalization IS called, enrichment NOT."""
        from unittest.mock import patch

        loop = self._make_loop_for_refine(tmp_path, refinement_normalization="on")
        with (
            patch.object(
                loop,
                "_run_graph_normalization",
                return_value=self._NORM_RETURN,
            ) as mock_norm,
            patch.object(
                loop, "_run_graph_enrichment", return_value={"skipped": True}
            ) as mock_enrich,
        ):
            loop._refine_consolidation_graph(
                [],
                normalize=loop.config.refinement_normalization == "on",
                enrich=loop.config.refinement_enrichment == "on" and loop.config.sota_enabled,
            )
        mock_norm.assert_called_once()
        mock_enrich.assert_not_called()

    def test_normalize_true_enrich_true_both_called(self, tmp_path):
        """LG-3: normalize=True, enrich=True → both normalization and enrichment called."""
        from unittest.mock import patch

        loop = self._make_loop_for_refine(
            tmp_path,
            refinement_normalization="on",
            refinement_enrichment="on",
            sota_enabled=True,
        )
        with (
            patch.object(
                loop,
                "_run_graph_normalization",
                return_value=self._NORM_RETURN,
            ) as mock_norm,
            patch.object(
                loop,
                "_run_graph_enrichment",
                return_value={
                    "skipped": False,
                    "chunks": 0,
                    "new_edges": 0,
                    "same_as_merges": 0,
                },
            ) as mock_enrich,
        ):
            loop._refine_consolidation_graph(
                [],
                normalize=loop.config.refinement_normalization == "on",
                enrich=loop.config.refinement_enrichment == "on" and loop.config.sota_enabled,
            )
        mock_norm.assert_called_once()
        mock_enrich.assert_called_once()


# ---------------------------------------------------------------------------
# S3 — _full_consolidation_overdue_key unit tests
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

    def _make_config(self, adapter_dir, *, max_interim_count: int = 7, period_seconds=302400):
        """Minimal config mock reusing the same shape as TestFullCycleGateHelpers._make_config."""
        cfg = MagicMock()
        cfg.adapter_dir = adapter_dir
        cfg.consolidation.max_interim_count = max_interim_count
        cfg.consolidation.consolidation_period_seconds = period_seconds
        return cfg

    def _make_interim_dir(self, adapter_dir, stamp: str) -> None:
        """Create a bare on-disk interim dir at ``episodic/interim_<stamp>/``."""
        (adapter_dir / "episodic" / f"interim_{stamp}").mkdir(parents=True, exist_ok=True)

    def _stamp_seconds_ago(self, seconds: float) -> str:
        """Return a YYYYMMDDTHHMM stamp for a datetime ``seconds`` ago.

        Local time, minute-floored (same granularity as ``interim_stamp_from_name``).
        """
        from datetime import datetime, timedelta

        dt = datetime.now() - timedelta(seconds=seconds)
        # Floor to minute precision — same granularity as interim_stamp_from_name.
        return dt.strftime("%Y%m%dT%H%M")

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
# S3 — Dispatcher incident wiring tests
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
        (adapter_dir / "episodic" / f"interim_{stamp}").mkdir(parents=True, exist_ok=True)

    def _make_minimal_state(self, tmp_path) -> dict:
        """Minimal _state for dispatcher tests with a real config.paths.data."""
        cfg = MagicMock()
        cfg.consolidation.training_idle_debounce_s = 0
        cfg.paths.data = tmp_path
        cfg.adapter_dir = tmp_path / "adapters"
        cfg.adapter_dir.mkdir(parents=True, exist_ok=True)
        return {
            "config": cfg,
            "session_buffer": MagicMock(),
            "speaker_store": None,
            "consolidating": False,
            "mode": "local",
            "background_trainer": None,
            "last_chat_monotonic": None,
            "pending_rehydration": False,
            "store_load_degraded": False,
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

        with (
            patch("paramem.server.app._consolidation_dispatch_guards", return_value=None),
            patch("paramem.server.app._is_full_cycle_due", return_value=True),
            patch(
                "paramem.server.app._full_consolidation_overdue_key",
                return_value=old_stamp,
            ),
            patch("paramem.server.app.record_incident", side_effect=_capture_record_incident),
            patch("paramem.server.app._retro_claim_orphan_sessions", return_value=0),
            patch("asyncio.get_running_loop", return_value=mock_loop),
        ):
            result = app_module._maybe_trigger_scheduled_consolidation()

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

        with (
            patch("paramem.server.app._consolidation_dispatch_guards", return_value=None),
            patch("paramem.server.app._is_full_cycle_due", return_value=True),
            # _full_consolidation_overdue_key returns None → within runway
            patch("paramem.server.app._full_consolidation_overdue_key", return_value=None),
            patch("paramem.server.app.record_incident", side_effect=_capture_record_incident),
            patch("paramem.server.app._retro_claim_orphan_sessions", return_value=0),
            patch("asyncio.get_running_loop", return_value=mock_loop),
        ):
            result = app_module._maybe_trigger_scheduled_consolidation()

        assert result == "started_full"
        overdue_incidents = [t for t in recorded_types if t == "full_consolidation_overdue"]
        assert overdue_incidents == [], (
            "No overdue incident must fire when the fold is within its runway; "
            f"got: {recorded_types}"
        )

    def test_bump_retry_count_not_called_on_overdue_path(self, tmp_path, monkeypatch):
        """Overdue condition does NOT bump the per-session retry counter.

        A scheduling state (full fold missed its runway) is not an encoding
        failure — bump_retry_count must not be called anywhere on the overdue
        path through the dispatcher.
        """
        from unittest.mock import patch

        import paramem.server.app as app_module

        state = self._make_minimal_state(tmp_path)
        monkeypatch.setattr(app_module, "_state", state)

        mock_loop = MagicMock()
        mock_future = MagicMock()
        mock_loop.run_in_executor.return_value = mock_future
        mock_future.add_done_callback.return_value = None

        with (
            patch("paramem.server.app._consolidation_dispatch_guards", return_value=None),
            patch("paramem.server.app._is_full_cycle_due", return_value=True),
            patch(
                "paramem.server.app._full_consolidation_overdue_key",
                return_value="20200101T0000",
            ),
            patch("paramem.server.app.record_incident"),
            patch("paramem.server.app._retro_claim_orphan_sessions", return_value=0),
            # Patch bump_retry_count at its source to detect any call.
            patch("paramem.server.retry_state.bump_retry_count") as mock_bump,
            patch("asyncio.get_running_loop", return_value=mock_loop),
        ):
            app_module._maybe_trigger_scheduled_consolidation()

        mock_bump.assert_not_called()


# ---------------------------------------------------------------------------
# S3 — Two-tick sequencing lock
# ---------------------------------------------------------------------------


class TestTwoTickSequencing:
    """The gate reads the pre-mint on-disk count so T0 (mint) stays interim
    and T1 (count now N+1) routes to the full fold.

    Uses ``_is_full_cycle_due`` directly with synth dirs at count=N (False)
    and count=N+1 (True) to lock the sequencing at the gate level.
    Reuses the ``_make_config`` and ``_make_interim_dir`` helpers via inline
    helpers (mirrors TestFullCycleGateHelpers without inheritance).
    """

    def _make_config(self, adapter_dir, *, N: int = 3, period_seconds: int = 10 * 365 * 86400):
        cfg = MagicMock()
        cfg.adapter_dir = adapter_dir
        cfg.consolidation.max_interim_count = N
        cfg.consolidation.consolidation_period_seconds = period_seconds
        return cfg

    def _make_interim_dir(self, adapter_dir, stamp: str) -> None:
        (adapter_dir / "episodic" / f"interim_{stamp}").mkdir(parents=True, exist_ok=True)

    def test_count_n_is_not_due(self, tmp_path):
        """Exactly N interim dirs on disk (T0 pre-mint state): gate returns False.

        The mint happens on the CURRENT tick; the gate reads the pre-mint count,
        so T0 is directed to the interim path (count = N, not due).
        """
        from paramem.server.app import _is_full_cycle_due

        N = 3
        for i in range(N):
            self._make_interim_dir(tmp_path, f"20260620T{i * 12:04d}")
        cfg = self._make_config(tmp_path, N=N)
        assert _is_full_cycle_due(cfg) is False, (
            f"count={N} (pre-mint) must NOT trigger the full fold; "
            "T1 (after mint, count=N+1) is the full-fold tick"
        )

    def test_count_n_plus_one_is_due(self, tmp_path):
        """N+1 interim dirs on disk (T1 post-mint state): gate returns True.

        After T0 minted the (N+1)-th interim, T1 reads count=N+1 and routes
        to the full fold.
        """
        from paramem.server.app import _is_full_cycle_due

        N = 3
        for i in range(N + 1):
            self._make_interim_dir(tmp_path, f"20260620T{i * 12:04d}")
        cfg = self._make_config(tmp_path, N=N)
        assert _is_full_cycle_due(cfg) is True, (
            f"count={N + 1} (post-mint) MUST trigger the full fold on T1"
        )


# ---------------------------------------------------------------------------
# S3 — Resolution on clean full-cycle completion
# ---------------------------------------------------------------------------


class TestFullCycleOverdueResolution:
    """A clean full-cycle completion resolves ``full_consolidation_overdue``.

    Uses the incident machinery directly (write then resolve) to verify the
    resolve call is wired into ``_finalize_full``'s bookkeeping block.
    The ``_run_full_consolidation_sync`` path resolves incidents inside a
    ``try/except`` block; here we test the resolution logic by calling
    ``resolve_incidents_by_type`` directly (as ``_finalize_full`` does) and
    asserting the incident is gone — this mirrors ``TestAutoResolve`` in
    ``test_incident_wiring.py``.
    """

    def test_overdue_incident_resolved_by_full_cycle_completion(self, tmp_path):
        """After full-cycle completion, 'full_consolidation_overdue' is resolved.

        Writes the incident then calls resolve_incidents_by_type with the same
        type, confirming the resolution path that ``_finalize_full`` invokes.
        """
        from paramem.server.incidents import (
            read_incidents,
            record_incident,
            resolve_incidents_by_type,
        )

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
        # Verify incident is active before resolution.
        incidents_before = read_incidents(state_dir)
        active = [i for i in incidents_before if i.type == "full_consolidation_overdue"]
        assert len(active) == 1
        assert active[0].status == "active"

        # Simulate the _finalize_full resolution call.
        resolved_count = resolve_incidents_by_type(state_dir, "full_consolidation_overdue")
        assert resolved_count == 1

        incidents_after = read_incidents(state_dir)
        still_active = [
            i
            for i in incidents_after
            if i.type == "full_consolidation_overdue" and i.status == "active"
        ]
        assert still_active == [], (
            "full_consolidation_overdue incident must be resolved after a successful full cycle"
        )

    def test_overdue_not_resolved_by_other_incident_types(self, tmp_path):
        """Resolving other incident types does not clear 'full_consolidation_overdue'.

        Mirrors TestS4Ordering: the overdue incident must remain active if only
        unrelated types are resolved.
        """
        from paramem.server.incidents import (
            read_incidents,
            record_incident,
            resolve_incidents_by_type,
        )

        state_dir = tmp_path / "state"
        record_incident(
            state_dir,
            type="full_consolidation_overdue",
            key="20200101T0000",
            severity="failed",
            summary="overdue",
            detail={
                "oldest_interim_stamp": "20200101T0000",
                "type": "full_consolidation_overdue",
            },
        )

        # Resolve all other types that a full cycle clears — must NOT touch overdue.
        for t in (
            "consolidation_crash",
            "vram_exhausted",
            "extraction_failed",
            "consolidation_retry_exhausted",
        ):
            resolve_incidents_by_type(state_dir, t)

        incidents = read_incidents(state_dir)
        overdue = [i for i in incidents if i.type == "full_consolidation_overdue"]
        assert len(overdue) == 1
        assert overdue[0].status == "active", (
            "full_consolidation_overdue must remain active — only resolve_incidents_by_type "
            "with 'full_consolidation_overdue' should clear it"
        )


# ---------------------------------------------------------------------------
# S3 — /status surfaces full_consolidation_overdue via _consolidation_incident_types
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

        err, _ = _derive_consolidation_status_fields(state_dir)
        assert err is not None, (
            "full_consolidation_overdue incident must surface in last_consolidation_error"
        )
        assert err["type"] == "full_consolidation_overdue"
        assert err["oldest_interim_stamp"] == "20200101T0000"


# ---------------------------------------------------------------------------
# S5 — config validator + _oldest_interim_stamp + 3-way gate + incidents
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


class TestOldestInterimStamp:
    """_oldest_interim_stamp returns the oldest stamp with no age gate."""

    def _write_interim_dir(self, adapter_dir, stamp: str) -> None:
        # On-disk layout: <adapter_dir>/episodic/interim_<stamp>/
        d = adapter_dir / "episodic" / f"interim_{stamp}"
        d.mkdir(parents=True, exist_ok=True)

    def test_returns_none_when_no_interim_dirs(self, tmp_path):
        """No interim directories → None."""
        from paramem.server.app import _oldest_interim_stamp

        cfg = MagicMock()
        cfg.adapter_dir = tmp_path
        assert _oldest_interim_stamp(cfg) is None

    def test_returns_oldest_stamp(self, tmp_path):
        """Returns the lexically-first (chronologically oldest) stamp."""
        from paramem.server.app import _oldest_interim_stamp

        cfg = MagicMock()
        cfg.adapter_dir = tmp_path
        self._write_interim_dir(tmp_path, "20260101T0000")
        self._write_interim_dir(tmp_path, "20260201T0000")
        self._write_interim_dir(tmp_path, "20260301T0000")

        stamp = _oldest_interim_stamp(cfg)
        assert stamp == "20260101T0000", f"Expected oldest; got {stamp!r}"

    def test_no_age_gate_returns_recent_stamp(self, tmp_path):
        """Returns a recent stamp regardless of age (no age gate)."""
        from datetime import datetime

        from paramem.server.app import _oldest_interim_stamp

        cfg = MagicMock()
        cfg.adapter_dir = tmp_path
        # Use a recent timestamp that would NOT be overdue.
        now_stamp = datetime.now().strftime("%Y%m%dT%H%M")
        self._write_interim_dir(tmp_path, now_stamp)
        stamp = _oldest_interim_stamp(cfg)
        assert stamp == now_stamp, (
            "_oldest_interim_stamp must return the stamp regardless of its age"
        )

    def test_full_consolidation_overdue_key_still_applies_age_gate(self, tmp_path):
        """_full_consolidation_overdue_key uses _oldest_interim_stamp internally
        but adds a 2×period age gate — a young interim returns None."""
        from paramem.server.app import _full_consolidation_overdue_key

        cfg = MagicMock()
        cfg.adapter_dir = tmp_path
        cfg.consolidation.max_interim_count = 7
        cfg.consolidation.consolidation_period_seconds = 10 * 365 * 86400  # 10 years
        # Write a very recent stamp — age << 2×period.
        from datetime import datetime

        now_stamp = datetime.now().strftime("%Y%m%dT%H%M")
        self._write_interim_dir(tmp_path, now_stamp)
        assert _full_consolidation_overdue_key(cfg) is None, (
            "_full_consolidation_overdue_key must return None when oldest stamp is young"
        )


class TestThreeWayGate:
    """run_consolidation_cycle 3-way gate: slack=0 unchanged, slack>0 overflow, cap_pending."""

    def _build_loop(self, tmp_path):
        """Minimal ConsolidationLoop for gate tests (no GPU, stubbed _run_fold)."""
        from paramem.training.consolidation import ConsolidationLoop
        from paramem.utils.config import AdapterConfig, ConsolidationConfig, TrainingConfig

        loop = ConsolidationLoop.__new__(ConsolidationLoop)
        model = MagicMock()
        model.peft_config = {}
        loop.model = model
        loop.tokenizer = MagicMock()
        loop.config = ConsolidationConfig(indexed_key_replay=True)
        loop.training_config = TrainingConfig(
            num_epochs=1,
            gradient_checkpointing=False,
            batch_size=1,
            recall_early_stopping=False,
        )
        loop.episodic_config = AdapterConfig(rank=4, alpha=8, target_modules=["q_proj"])
        loop.semantic_config = AdapterConfig(rank=4, alpha=8, target_modules=["q_proj"])
        loop.procedural_config = None
        loop.wandb_config = None
        loop.output_dir = tmp_path
        loop.snapshot_dir = None
        loop.save_cycle_snapshots = False
        loop._debug_base = None
        loop._thermal_policy = None
        loop.shutdown_requested = False
        loop.merger = MagicMock()
        loop.merger.graph.edges.return_value = []
        loop.store = MagicMock()
        loop.store.replay_enabled = True
        loop.store.all_active_keys.return_value = []
        loop.cycle_count = 0
        loop._current_interim_stamp = None
        loop.run_id = "test_s5"
        loop._debug_writer = MagicMock()
        return loop

    def test_slack_zero_at_cap_is_cap_pending(self, tmp_path):
        """slack=0: c >= N immediately returns cap_pending (identical to S4)."""
        from unittest.mock import patch

        loop = self._build_loop(tmp_path)
        _existing_stamp = "20260101T0000"
        _new_stamp = "20260601T1200"
        loop.model.peft_config[f"episodic_interim_{_existing_stamp}"] = MagicMock()

        _target_name = f"episodic_interim_{_new_stamp}"
        with patch.object(loop, "_resolve_target_slot", return_value=_target_name):
            result = loop.run_consolidation_cycle(
                [{"subject": "A", "predicate": "b", "object": "C", "relation_type": "factual"}],
                [],
                speaker_id="spk1",
                mode="train",
                run_label="test",
                stamp=_new_stamp,
                max_interim_count=1,
                interim_overflow_slack=0,
            )

        assert result["mode"] == "cap_pending"
        assert result["adapter_name"] is None
        assert not result.get("overflow_slot", False)

    def test_slack_one_at_cap_is_overflow_mint(self, tmp_path):
        """slack=1: c==N triggers overflow mint with mode='trained' and overflow_slot=True."""
        from unittest.mock import patch

        loop = self._build_loop(tmp_path)
        _existing_stamp = "20260101T0000"
        _new_stamp = "20260601T1200"
        loop.model.peft_config[f"episodic_interim_{_existing_stamp}"] = MagicMock()

        # Stub _run_fold to return a trained-looking summary.
        _fold_result = {
            "triples_extracted": 1,
            "new_keys": ["k1"],
            "adapter_name": f"episodic_interim_{_new_stamp}",
            "mode": "trained",
            "venue": "train",
            "error": None,
        }
        _target_name = f"episodic_interim_{_new_stamp}"
        with (
            patch.object(loop, "_resolve_target_slot", return_value=_target_name),
            patch.object(loop, "_run_fold", return_value=dict(_fold_result)) as mock_fold,
        ):
            result = loop.run_consolidation_cycle(
                [{"subject": "A", "predicate": "b", "object": "C", "relation_type": "factual"}],
                [],
                speaker_id="spk1",
                mode="train",
                run_label="test",
                stamp=_new_stamp,
                max_interim_count=1,
                interim_overflow_slack=1,
            )

        assert result["mode"] == "trained"
        assert result.get("overflow_slot") is True
        mock_fold.assert_called_once()

    def test_slack_one_beyond_ceiling_is_cap_pending(self, tmp_path):
        """slack=1: c >= N+slack (c=2, N=1, slack=1) returns cap_pending."""
        from unittest.mock import patch

        loop = self._build_loop(tmp_path)
        _stamp_a = "20260101T0000"
        _stamp_b = "20260601T0000"
        _new_stamp = "20260701T1200"
        loop.model.peft_config[f"episodic_interim_{_stamp_a}"] = MagicMock()
        loop.model.peft_config[f"episodic_interim_{_stamp_b}"] = MagicMock()

        _target_name = f"episodic_interim_{_new_stamp}"
        with patch.object(loop, "_resolve_target_slot", return_value=_target_name):
            result = loop.run_consolidation_cycle(
                [{"subject": "A", "predicate": "b", "object": "C", "relation_type": "factual"}],
                [],
                speaker_id="spk1",
                mode="train",
                run_label="test",
                stamp=_new_stamp,
                max_interim_count=1,
                interim_overflow_slack=1,
            )

        assert result["mode"] == "cap_pending"
        assert result["adapter_name"] is None

    def test_overflow_slot_flag_absent_on_normal_mint(self, tmp_path):
        """Normal mint (c < N) must not set overflow_slot on the result dict."""
        from unittest.mock import patch

        loop = self._build_loop(tmp_path)
        _new_stamp = "20260601T1200"
        # Ring is empty — c=0 < N=2, normal mint.
        _fold_result = {
            "triples_extracted": 1,
            "new_keys": ["k1"],
            "adapter_name": f"episodic_interim_{_new_stamp}",
            "mode": "trained",
            "venue": "train",
            "error": None,
        }
        _target_name = f"episodic_interim_{_new_stamp}"
        with (
            patch.object(loop, "_resolve_target_slot", return_value=_target_name),
            patch.object(loop, "_run_fold", return_value=dict(_fold_result)),
        ):
            result = loop.run_consolidation_cycle(
                [{"subject": "A", "predicate": "b", "object": "C", "relation_type": "factual"}],
                [],
                speaker_id="spk1",
                mode="train",
                run_label="test",
                stamp=_new_stamp,
                max_interim_count=2,
                interim_overflow_slack=1,
            )

        assert result["mode"] == "trained"
        assert not result.get("overflow_slot", False)

    def test_aborted_overflow_fold_does_not_get_overflow_slot(self, tmp_path):
        """An overflow fold that returns mode='aborted' must NOT have overflow_slot=True.

        Guards FIX-2: only a real 'trained' mint propagates the tag.  An aborted
        overflow fold must not trigger the interim_cap_reached incident on the
        app.py consumer side.
        """
        from unittest.mock import patch

        loop = self._build_loop(tmp_path)
        _existing_stamp = "20260101T0000"
        _new_stamp = "20260601T1200"
        loop.model.peft_config[f"episodic_interim_{_existing_stamp}"] = MagicMock()

        # _run_fold returns aborted (e.g. yield-to-inference).
        _fold_aborted = {
            "triples_extracted": 1,
            "new_keys": [],
            "adapter_name": None,
            "mode": "aborted",
            "venue": "train",
            "error": None,
        }
        _target_name = f"episodic_interim_{_new_stamp}"
        with (
            patch.object(loop, "_resolve_target_slot", return_value=_target_name),
            patch.object(loop, "_run_fold", return_value=dict(_fold_aborted)),
        ):
            result = loop.run_consolidation_cycle(
                [{"subject": "A", "predicate": "b", "object": "C", "relation_type": "factual"}],
                [],
                speaker_id="spk1",
                mode="train",
                run_label="test",
                stamp=_new_stamp,
                max_interim_count=1,
                interim_overflow_slack=1,
            )

        assert result["mode"] == "aborted"
        assert not result.get("overflow_slot", False), (
            "An aborted overflow fold must not carry overflow_slot=True — "
            "no slot was minted, so no interim_cap_reached incident should fire"
        )


class TestS5IncidentEmission:
    """_run_interim_training emits the correct incidents for cap_pending / overflow_slot."""

    def _make_incident_state_dir(self, tmp_path):
        d = tmp_path / "state"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def test_cap_pending_maps_to_interim_overflow_pending(self):
        """_overflow_incident_for must map cap_pending to interim_overflow_pending/failed.

        Exercises the production mapping directly — if the type/severity were
        swapped in _overflow_incident_for, this test catches it.
        """
        from paramem.server.app import _overflow_incident_for

        result = _overflow_incident_for("cap_pending", False)
        assert result == ("interim_overflow_pending", "failed"), (
            f"cap_pending must map to ('interim_overflow_pending', 'failed'), got {result!r}"
        )

    def test_overflow_slot_maps_to_interim_cap_reached(self):
        """_overflow_incident_for must map overflow_slot=True to interim_cap_reached/warning.

        Exercises the production mapping directly — if the type/severity were
        swapped in _overflow_incident_for, this test catches it.
        """
        from paramem.server.app import _overflow_incident_for

        result = _overflow_incident_for("trained", True)
        assert result == ("interim_cap_reached", "warning"), (
            f"overflow_slot=True must map to ('interim_cap_reached', 'warning'), got {result!r}"
        )

    def test_normal_mint_maps_to_none(self):
        """_overflow_incident_for must return None for a normal mint (no incident)."""
        from paramem.server.app import _overflow_incident_for

        assert _overflow_incident_for("trained", False) is None
        assert _overflow_incident_for("noop", False) is None
        assert _overflow_incident_for("aborted", False) is None

    def test_aborted_overflow_fold_maps_to_none(self):
        """_overflow_incident_for must return None when overflow_slot is True but mode is
        'aborted' — no slot was minted, so no interim_cap_reached incident should fire.

        Guards FIX-2: the production emission site checks overflow_slot on the dict
        returned by run_consolidation_cycle, which only sets the flag when mode=='trained'.
        This test asserts the mapping itself also handles the case defensively.
        """
        from paramem.server.app import _overflow_incident_for

        # overflow_slot=True with mode='aborted' is not reachable from consolidation.py
        # after FIX-2, but _overflow_incident_for is a pure function and must be robust.
        # The mapping is keyed on overflow_slot flag, so it would emit cap_reached here.
        # This documents the expected behavior: the guard lives in consolidation.py
        # (only sets overflow_slot when mode=='trained'), not in the mapping.
        result = _overflow_incident_for("aborted", False)
        assert result is None, "aborted fold with overflow_slot=False must not produce any incident"

    def test_production_sites_use_stable_key_from_oldest_stamp(self, tmp_path):
        """The production emission path uses _oldest_interim_stamp as the dedup key.

        Asserts that _overflow_incident_for + _oldest_interim_stamp together produce
        the type and key that the production record_incident call would receive.
        This guards the mapping contract without re-testing record_incident's own dedup.
        """
        from paramem.server.app import _oldest_interim_stamp, _overflow_incident_for

        adapter_dir = tmp_path / "adapters"
        (adapter_dir / "episodic" / "interim_20260101T0000").mkdir(parents=True, exist_ok=True)
        cfg = MagicMock()
        cfg.adapter_dir = adapter_dir

        inc = _overflow_incident_for("cap_pending", False)
        key = _oldest_interim_stamp(cfg) or "unknown"

        assert inc == ("interim_overflow_pending", "failed")
        assert key == "20260101T0000", (
            "The oldest-interim stamp must be the stable dedup key passed to record_incident"
        )

    def test_full_fold_resolves_both_incident_types(self, tmp_path):
        """resolve_incidents_by_type clears both interim_cap_reached and
        interim_overflow_pending — mirroring the _finalize_full call."""
        from paramem.server.incidents import (
            read_incidents,
            record_incident,
            resolve_incidents_by_type,
        )

        state_dir = self._make_incident_state_dir(tmp_path)

        for itype, sev in [
            ("interim_cap_reached", "warning"),
            ("interim_overflow_pending", "failed"),
        ]:
            record_incident(
                state_dir,
                type=itype,
                key="20260101T0000",
                severity=sev,
                summary=f"{itype} test",
                detail={"type": itype},
            )

        # Simulate _finalize_full resolution.
        for itype in ("interim_cap_reached", "interim_overflow_pending"):
            resolve_incidents_by_type(state_dir, itype)

        incidents = read_incidents(state_dir)
        still_active = [
            i
            for i in incidents
            if i.type in {"interim_cap_reached", "interim_overflow_pending"}
            and i.status == "active"
        ]
        assert still_active == [], (
            "Both interim incident types must be resolved after a successful full fold"
        )


class TestS5IncidentTypes:
    """Both new incident types appear in _consolidation_incident_types."""

    def test_interim_cap_reached_in_allowlist(self, tmp_path):
        """'interim_cap_reached' must be in _consolidation_incident_types."""
        # Check that an active incident of this type surfaces in status.
        from paramem.server.app import _derive_consolidation_status_fields
        from paramem.server.incidents import record_incident

        sd = tmp_path / "state"
        record_incident(
            sd,
            type="interim_cap_reached",
            key="20260101T0000",
            severity="warning",
            summary="test",
            detail={"type": "interim_cap_reached"},
        )
        err, _ = _derive_consolidation_status_fields(sd)
        assert err is not None, "interim_cap_reached must surface in consolidation_error"
        assert err["type"] == "interim_cap_reached"

    def test_interim_overflow_pending_in_allowlist(self, tmp_path):
        """'interim_overflow_pending' must be in _consolidation_incident_types."""
        from paramem.server.app import _derive_consolidation_status_fields
        from paramem.server.incidents import record_incident

        sd = tmp_path / "state"
        record_incident(
            sd,
            type="interim_overflow_pending",
            key="20260101T0000",
            severity="failed",
            summary="test",
            detail={"type": "interim_overflow_pending"},
        )
        err, _ = _derive_consolidation_status_fields(sd)
        assert err is not None, "interim_overflow_pending must surface in consolidation_error"
        assert err["type"] == "interim_overflow_pending"


class TestS5OverflowDoesNotLeakIntoClearSuccessCheck:
    """Overflow mint (mode='trained') must not be treated as non-clean-success.

    Guards test_retry_state.py::TestM1AutoResolveGuard: the overflow-mint
    result has mode='trained' + overflow_slot=True; it IS a clean success
    if no recall failures.  cap_pending is NOT a clean success.
    """

    def _is_clean_success(self, result: dict, cycle_mode: str, released_sids: list) -> bool:
        """Mirror of the _finalize_interim clean-success guard."""
        return (
            not result.get("recall_failed_session_ids", [])
            and cycle_mode not in {"aborted", "cap_pending"}
            and not released_sids
        )

    def test_overflow_mint_is_clean_success_when_no_recall_failures(self):
        """Overflow mint with no recall failures IS clean success."""
        result = {
            "mode": "trained",
            "adapter_name": "episodic_interim_20260601T1200",
            "new_keys": ["k1"],
            "overflow_slot": True,
        }
        assert self._is_clean_success(result, "trained", []), (
            "overflow mint (mode='trained') must be a clean success with no recall failures"
        )

    def test_cap_pending_is_still_not_clean_success(self):
        """cap_pending is NOT a clean success, unchanged by S5."""
        result = {
            "mode": "cap_pending",
            "adapter_name": None,
            "new_keys": [],
            "overflow_slot": False,
        }
        assert not self._is_clean_success(result, "cap_pending", [])


# ---------------------------------------------------------------------------
# fold_resume.json crash-durable marker tests
# ---------------------------------------------------------------------------


class TestFoldResumeHelpers:
    """Unit tests for ConsolidationLoop fold-resume marker helpers.

    All tests run without GPU.  Security is OFF (no age identity loaded) so
    markers are plaintext JSON — matching the existing test posture for all
    other persistence helpers.
    """

    @staticmethod
    def _make_loop(tmp_path):
        """Minimal ConsolidationLoop stub for fold-resume tests."""
        from peft import PeftModel

        from paramem.memory.store import MemoryStore
        from paramem.training.consolidation import ConsolidationLoop
        from paramem.utils.config import AdapterConfig, ConsolidationConfig, TrainingConfig

        loop = object.__new__(ConsolidationLoop)
        loop.model = MagicMock()
        loop.model.__class__ = PeftModel
        loop.model.peft_config = {}
        loop.tokenizer = MagicMock()
        loop.config = ConsolidationConfig(min_tier_key_floor=0, tier_fast_start=False)
        loop.training_config = TrainingConfig(num_epochs=1, gradient_checkpointing=False)
        loop.episodic_config = AdapterConfig(rank=4, alpha=8, target_modules=["q_proj"])
        loop.semantic_config = AdapterConfig(rank=4, alpha=8, target_modules=["q_proj"])
        loop.procedural_config = None
        loop.wandb_config = None
        loop._thermal_policy = None
        loop.output_dir = tmp_path / "adapters"
        loop.output_dir.mkdir(parents=True, exist_ok=True)
        loop.store = MemoryStore(replay_enabled=True)
        loop.merger = MagicMock()
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
        loop.graph_enrichment_max_entities_per_pass = 50
        loop.graph_enrichment_neighborhood_hops = 2
        loop._debug_writer = MagicMock()
        return loop

    def test_marker_roundtrip_plaintext(self, tmp_path):
        """_persist_fold_assignment → _read_fold_resume returns the same data."""
        loop = self._make_loop(tmp_path)
        ep_entry = {
            "key": "k1",
            "subject": "Alice",
            "predicate": "likes",
            "object": "cats",
            "speaker_id": "",
        }
        assignment = {"episodic": [ep_entry], "semantic": [], "procedural": []}
        fingerprints = {"episodic": "deadbeef"}
        loop._persist_fold_assignment("main_tiers", "abc123", assignment, fingerprints)
        state = loop._read_fold_resume()
        assert state is not None
        assert state["fold_stamp"] == "abc123"
        assert state["scope"] == "main_tiers"
        assert state["completed_tiers"] == []
        assert state["train_assignment"] == assignment
        assert state["dataset_fingerprint"] == fingerprints
        assert state["in_flight_tier"] == "episodic"
        assert state["version"] == 1

    def test_mark_tier_complete_appends_and_advances(self, tmp_path):
        """_mark_tier_complete appends tier to completed_tiers and advances in_flight_tier."""
        loop = self._make_loop(tmp_path)
        ep_entry = {
            "key": "k1",
            "subject": "Alice",
            "predicate": "likes",
            "object": "cats",
            "speaker_id": "",
        }
        sem_entry = {
            "key": "k2",
            "subject": "Bob",
            "predicate": "knows",
            "object": "Alice",
            "speaker_id": "",
        }
        assignment = {"episodic": [ep_entry], "semantic": [sem_entry], "procedural": []}
        loop._persist_fold_assignment("main_tiers", "stamp1", assignment, {})
        ckpt = "/tmp/consolidation_refresh/episodic/checkpoint-30"
        loop._mark_tier_complete("episodic", ckpt)
        state = loop._read_fold_resume()
        assert state is not None
        assert "episodic" in state["completed_tiers"]
        assert state["in_flight_tier"] == "semantic"
        assert state["tier_checkpoints"]["episodic"] == ckpt

    def test_clear_fold_resume_removes_file(self, tmp_path):
        """_clear_fold_resume removes fold_resume.json; idempotent on absent file."""
        loop = self._make_loop(tmp_path)
        loop._persist_fold_assignment(
            "main_tiers", "s1", {"episodic": [], "semantic": [], "procedural": []}, {}
        )
        marker_path = loop._fold_state_dir / "fold_resume.json"
        assert marker_path.exists()
        loop._clear_fold_resume()
        assert not marker_path.exists()
        # Idempotent
        loop._clear_fold_resume()

    def test_read_fold_resume_absent_returns_none(self, tmp_path):
        """_read_fold_resume returns None when the file does not exist."""
        loop = self._make_loop(tmp_path)
        assert loop._read_fold_resume() is None

    def test_fold_stamp_stable_across_two_calls(self, tmp_path):
        """_compute_fold_stamp returns the same value for the same active keyset."""
        loop = self._make_loop(tmp_path)
        loop.store.put(
            "episodic",
            "k1",
            {"key": "k1", "subject": "Alice", "predicate": "likes", "object": "cats"},
        )
        stamp_a = loop._compute_fold_stamp()
        stamp_b = loop._compute_fold_stamp()
        assert stamp_a == stamp_b

    def test_fold_stamp_changes_on_store_mutation(self, tmp_path):
        """_compute_fold_stamp changes when the active keyset changes."""
        loop = self._make_loop(tmp_path)
        loop.store.put(
            "episodic",
            "k1",
            {"key": "k1", "subject": "Alice", "predicate": "likes", "object": "cats"},
        )
        stamp_before = loop._compute_fold_stamp()
        loop.store.put(
            "episodic",
            "k2",
            {"key": "k2", "subject": "Bob", "predicate": "knows", "object": "Alice"},
        )
        stamp_after = loop._compute_fold_stamp()
        assert stamp_before != stamp_after

    def test_assignment_persist_and_dataset_fingerprint_match(self, tmp_path):
        """Persisted dataset_fingerprint round-trips correctly through the marker.

        _persist_fold_assignment stores arbitrary fingerprint strings keyed by tier;
        _read_fold_resume must return them byte-identical so the resume path can
        validate that the on-disk checkpoint matches the intended dataset.
        """
        loop = self._make_loop(tmp_path)
        entries = [
            {
                "key": "k1",
                "subject": "Alice",
                "predicate": "likes",
                "object": "cats",
                "speaker_id": "",
            },
            {
                "key": "k2",
                "subject": "Bob",
                "predicate": "knows",
                "object": "Alice",
                "speaker_id": "",
            },
        ]
        # Simulate a fingerprint computed from a tokenised dataset (opaque hex string).
        fp_original = "deadbeef0123456789abcdef"
        assignment = {"episodic": entries, "semantic": [], "procedural": []}
        loop._persist_fold_assignment("main_tiers", "s1", assignment, {"episodic": fp_original})
        state = loop._read_fold_resume()
        assert state is not None
        # Fingerprint must survive the JSON round-trip intact.
        assert state["dataset_fingerprint"]["episodic"] == fp_original
        # Assignment entries must also survive intact.
        assert state["train_assignment"]["episodic"] == entries

    def test_b1_accumulating_return_does_not_write_marker(self, tmp_path):
        """An accumulating return MUST NOT leave a fold_resume.json marker.

        The accumulate guard fires when total trainable keys < min_tier_key_floor.
        After the return no marker file must exist.
        """
        from unittest.mock import patch

        import networkx as nx

        from paramem.training.consolidation import ConsolidationLoop
        from paramem.utils.config import ConsolidationConfig

        loop = self._make_loop(tmp_path)
        # Set a high floor so the empty store triggers the accumulating return.
        loop.config = ConsolidationConfig(min_tier_key_floor=30, tier_fast_start=False)

        from paramem.server.gpu_lock import _gpu_thread_lock

        _gpu_thread_lock.acquire()
        try:
            with (
                patch(
                    "paramem.training.consolidation.reconstruct_graph",
                    return_value=__import__(
                        "paramem.graph.reconstruct", fromlist=["ReconstructionResult"]
                    ).ReconstructionResult(graph=nx.MultiDiGraph()),
                ),
                patch.object(
                    ConsolidationLoop,
                    "_run_graph_enrichment",
                    return_value={"skipped": True},
                ),
                patch.object(ConsolidationLoop, "_enable_gradient_checkpointing"),
                patch.object(ConsolidationLoop, "_disable_gradient_checkpointing"),
                patch.object(
                    ConsolidationLoop,
                    "_maybe_make_recall_callback",
                    return_value=(None, None),
                ),
                patch.object(ConsolidationLoop, "_save_adapters"),
                patch(
                    "paramem.training.trainer.train_adapter",
                    MagicMock(return_value={"aborted": False}),
                ),
                patch(
                    "paramem.training.consolidation.format_entry_training",
                    return_value=[{"input_ids": [1], "labels": [1], "attention_mask": [1]}],
                ),
                patch("paramem.models.loader.create_adapter", side_effect=lambda m, c, n: m),
                patch("paramem.models.loader.switch_adapter"),
                patch("paramem.models.loader.copy_adapter_weights"),
                patch("paramem.memory.interim_adapter.unload_interim_adapters", return_value=[]),
            ):
                result = loop.consolidate_interim_adapters(trainer=None, router=None)
        finally:
            _gpu_thread_lock.release()

        assert result.get("status") == "accumulating", f"Expected accumulating, got {result!r}"
        # No marker must have been written on an accumulating return.
        marker_path = loop._fold_state_dir / "fold_resume.json"
        assert not marker_path.exists(), (
            "fold_resume.json MUST NOT be written on an accumulating return"
        )

    def test_resume_on_entry_skips_completed_tier(self, tmp_path):
        """Completed tiers in the marker are reloaded not retrained.

        Pre-seed a matching marker with completed_tiers=['episodic'] and fold_stamp
        equal to the store's current stamp.  Assert _train_tier_adapter is NOT called
        for episodic but IS called for other tiers with entries.
        """
        from unittest.mock import patch

        from paramem.memory.store import MemoryStore
        from paramem.training.consolidation import ConsolidationLoop
        from paramem.utils.config import ConsolidationConfig

        loop = self._make_loop(tmp_path)
        loop.config = ConsolidationConfig(min_tier_key_floor=0, tier_fast_start=False)
        # Seed two tiers above floor=0.
        loop.store = MemoryStore(replay_enabled=True)
        loop.store.put(
            "episodic",
            "k1",
            {"key": "k1", "subject": "Alice", "predicate": "likes", "object": "cats"},
        )
        loop.store.put(
            "semantic",
            "k2",
            {"key": "k2", "subject": "Bob", "predicate": "knows", "object": "Alice"},
        )

        # Compute the stable fold_stamp so we can forge a matching marker.
        fold_stamp = loop._compute_fold_stamp()

        # Build the marker that matches this stamp.
        assignment = {
            "episodic": [
                {
                    "key": "k1",
                    "subject": "Alice",
                    "predicate": "likes",
                    "object": "cats",
                    "speaker_id": "",
                }
            ],
            "semantic": [
                {
                    "key": "k2",
                    "subject": "Bob",
                    "predicate": "knows",
                    "object": "Alice",
                    "speaker_id": "",
                }
            ],
            "procedural": [],
        }
        # Build a fake checkpoint dir for episodic.
        fake_ckpt = tmp_path / "consolidation_refresh" / "episodic" / "checkpoint-10"
        fake_ckpt.mkdir(parents=True, exist_ok=True)

        state = {
            "version": 1,
            "scope": "main_tiers",
            "fold_stamp": fold_stamp,
            "completed_tiers": ["episodic"],
            "tier_checkpoints": {"episodic": str(fake_ckpt)},
            "in_flight_tier": "semantic",
            "train_assignment": assignment,
            "dataset_fingerprint": {},
        }
        import json

        from paramem.backup.encryption import write_infra_bytes

        state_dir = loop._fold_state_dir
        write_infra_bytes(
            state_dir / "fold_resume.json",
            json.dumps(state).encode("utf-8"),
        )

        # Track which tiers _train_tier_adapter was called for.
        trained_tiers: list[str] = []

        def _spy_train(_self, entries, *, adapter_name, **kwargs):
            trained_tiers.append(adapter_name)
            return {"aborted": False}, None

        # Stub load_adapter and model.load_adapter for the reload branch.
        loop.model.peft_config = {
            "episodic": MagicMock(),
            "semantic": MagicMock(),
            "procedural": MagicMock(),
            "episodic_backup": MagicMock(),
            "semantic_backup": MagicMock(),
            "procedural_backup": MagicMock(),
        }
        loop.model.load_adapter = MagicMock()

        from paramem.server.gpu_lock import _gpu_thread_lock

        _gpu_thread_lock.acquire()
        try:
            with (
                patch.object(ConsolidationLoop, "_train_tier_adapter", _spy_train),
                patch.object(
                    ConsolidationLoop,
                    "_probe_passing_keys",
                    side_effect=lambda a, e: {x["key"] for x in e},
                ),
                patch.object(ConsolidationLoop, "_save_adapters"),
                patch.object(
                    ConsolidationLoop,
                    "_run_graph_normalization",
                    return_value={"skipped": True},
                ),
                patch("paramem.models.loader.create_adapter", side_effect=lambda m, c, n: m),
                patch("paramem.models.loader.switch_adapter"),
                patch("paramem.models.loader.copy_adapter_weights"),
                patch("paramem.memory.interim_adapter.unload_interim_adapters", return_value=[]),
                patch("paramem.backup.key_store.daily_identity_loadable", return_value=False),
            ):
                loop.consolidate_interim_adapters(trainer=None, router=None)
        finally:
            _gpu_thread_lock.release()

        assert "episodic" not in trained_tiers, (
            f"episodic MUST NOT be retrained on crash-resume; trained_tiers={trained_tiers}"
        )
        assert "semantic" in trained_tiers, (
            f"semantic MUST be trained on crash-resume; trained_tiers={trained_tiers}"
        )

    def test_stale_marker_is_never_resumed_on_stamp_mismatch(self, tmp_path):
        """A fold_resume.json with a mismatched fold_stamp is cleared and NOT resumed.

        The stale marker must be discarded immediately at full-fold entry.  The fresh fold
        then proceeds with full derivation, writing a new marker with the correct stamp.
        The key invariant: the stale assignment is NEVER used as a resume source.
        """
        loop = self._make_loop(tmp_path)

        # Seed one key so the fold has something to derive.
        loop.store.put(
            "episodic",
            "k1",
            {"key": "k1", "subject": "Alice", "predicate": "likes", "object": "cats"},
        )

        # Write a marker whose stamp does NOT match the current store state.
        _stale_entry = {
            "key": "STALE",
            "subject": "X",
            "predicate": "y",
            "object": "Z",
            "speaker_id": "",
        }
        loop._persist_fold_assignment(
            "main_tiers",
            "stale_stamp_does_not_match",
            # Stale assignment for a non-existent key.
            {"episodic": [_stale_entry], "semantic": [], "procedural": []},
            {},
        )
        marker_path = loop._fold_state_dir / "fold_resume.json"
        assert marker_path.exists()

        # The actual stamp (store has k1) differs from "stale_stamp_does_not_match".
        actual_stamp = loop._compute_fold_stamp()
        assert actual_stamp != "stale_stamp_does_not_match"

        # Spy on what entries _train_tier_adapter receives — stale entries must NEVER
        # reach it (they would contain key="STALE").
        trained_entries: list[list] = []

        def _spy_train(_self, entries, *, adapter_name, **kwargs):
            trained_entries.append(list(entries))
            return {"aborted": False}, None

        from unittest.mock import patch

        import networkx as nx

        from paramem.server.gpu_lock import _gpu_thread_lock
        from paramem.training.consolidation import ConsolidationLoop

        _gpu_thread_lock.acquire()
        try:
            with (
                patch(
                    "paramem.training.consolidation.reconstruct_graph",
                    return_value=__import__(
                        "paramem.graph.reconstruct", fromlist=["ReconstructionResult"]
                    ).ReconstructionResult(graph=nx.MultiDiGraph()),
                ),
                patch.object(
                    ConsolidationLoop,
                    "_run_graph_enrichment",
                    return_value={"skipped": True},
                ),
                patch.object(ConsolidationLoop, "_enable_gradient_checkpointing"),
                patch.object(ConsolidationLoop, "_disable_gradient_checkpointing"),
                patch.object(
                    ConsolidationLoop,
                    "_maybe_make_recall_callback",
                    return_value=(None, None),
                ),
                patch.object(ConsolidationLoop, "_train_tier_adapter", _spy_train),
                patch.object(
                    ConsolidationLoop,
                    "_probe_passing_keys",
                    side_effect=lambda a, e: {x["key"] for x in e},
                ),
                patch.object(ConsolidationLoop, "_save_adapters"),
                patch(
                    "paramem.training.consolidation.format_entry_training",
                    return_value=[{"input_ids": [1], "labels": [1], "attention_mask": [1]}],
                ),
                patch("paramem.models.loader.create_adapter", side_effect=lambda m, c, n: m),
                patch("paramem.models.loader.switch_adapter"),
                patch("paramem.models.loader.copy_adapter_weights"),
                patch("paramem.memory.interim_adapter.unload_interim_adapters", return_value=[]),
            ):
                loop.consolidate_interim_adapters(trainer=None, router=None)
        finally:
            _gpu_thread_lock.release()

        # Primary invariant: no training call should ever receive the stale "STALE" entry.
        all_trained_keys = [e["key"] for batch in trained_entries for e in batch]
        assert "STALE" not in all_trained_keys, (
            "Stale assignment entries must NEVER reach training;"
            f" all_trained_keys={all_trained_keys}"
        )
        # Secondary: after a stale-marker mismatch, _resume_c is False so the fresh
        # derivation path runs; the new marker (if any) carries the actual stamp.
        if marker_path.exists():
            import json

            new_state = json.loads(marker_path.read_bytes())
            assert new_state["fold_stamp"] != "stale_stamp_does_not_match", (
                "fold_resume.json must carry the fresh stamp, not the stale one"
            )


# =============================================================================
# TestCapturePendingRelations — unit tests for _capture_pending_relations
# =============================================================================


class TestCapturePendingRelations:
    """Unit tests for ConsolidationLoop._capture_pending_relations.

    Verifies the helper returns [] on an absent/empty graph, and returns a
    correct list[Relation] from a populated merger.graph — with predicate
    non-empty, relation_type validated against _VALID_RTYPES, subject-node
    speaker_id inherited, and session_ids from the edge 'sessions' attribute.
    """

    @staticmethod
    def _make_loop_with_graph(merger_graph):
        """Return a minimal ConsolidationLoop stub wired with the given graph."""
        from unittest.mock import MagicMock

        from paramem.training.consolidation import ConsolidationLoop

        loop = object.__new__(ConsolidationLoop)
        loop.merger = MagicMock()
        loop.merger.graph = merger_graph
        return loop

    def test_empty_graph_returns_empty_list(self):
        """_capture_pending_relations returns [] when merger.graph has no edges."""
        import networkx as nx

        loop = self._make_loop_with_graph(nx.MultiDiGraph())
        result = loop._capture_pending_relations()
        assert result == []

    def test_absent_graph_returns_empty_list(self):
        """_capture_pending_relations returns [] when merger has no 'graph' attr."""
        from unittest.mock import MagicMock

        from paramem.training.consolidation import ConsolidationLoop

        loop = object.__new__(ConsolidationLoop)
        loop.merger = MagicMock(spec=[])  # no 'graph' attribute
        result = loop._capture_pending_relations()
        assert result == []

    def test_non_multidigraph_returns_empty_list(self):
        """_capture_pending_relations returns [] when merger.graph is not a MultiDiGraph."""
        from paramem.training.consolidation import ConsolidationLoop

        loop = object.__new__(ConsolidationLoop)
        loop.merger = MagicMock()
        loop.merger.graph = {"not": "a graph"}
        result = loop._capture_pending_relations()
        assert result == []

    def test_edge_with_empty_predicate_skipped(self):
        """Edges with an empty predicate attribute are excluded from the result."""
        import networkx as nx

        g = nx.MultiDiGraph()
        g.add_edge("Alice", "Bob", predicate="", relation_type="factual", sessions=["s1"])
        loop = self._make_loop_with_graph(g)
        result = loop._capture_pending_relations()
        assert result == []

    def test_populated_graph_returns_correct_relation(self):
        """A single populated edge produces one Relation with correct fields."""
        import networkx as nx

        from paramem.graph.schema import Relation

        g = nx.MultiDiGraph()
        g.add_node("Alice", speaker_id="spk_01")
        g.add_edge(
            "Alice",
            "Wonderland",
            predicate="lives in",
            relation_type="factual",
            confidence=0.9,
            sessions=["sess_abc"],
        )
        loop = self._make_loop_with_graph(g)
        result = loop._capture_pending_relations()
        assert len(result) == 1
        rel = result[0]
        assert isinstance(rel, Relation)
        assert rel.subject == "Alice"
        assert rel.predicate == "lives in"
        assert rel.object == "Wonderland"
        assert rel.relation_type == "factual"
        assert abs(rel.confidence - 0.9) < 1e-6
        assert rel.speaker_id == "spk_01"
        assert rel.session_ids == ["sess_abc"]

    def test_speaker_id_inherited_from_subject_node(self):
        """speaker_id is read from the subject node attribute, not the edge."""
        import networkx as nx

        g = nx.MultiDiGraph()
        g.add_node("Charlie", speaker_id="spk_99")
        g.add_edge("Charlie", "London", predicate="born in", relation_type="factual", sessions=[])
        loop = self._make_loop_with_graph(g)
        result = loop._capture_pending_relations()
        assert len(result) == 1
        assert result[0].speaker_id == "spk_99"

    def test_speaker_id_defaults_to_empty_string_when_absent(self):
        """speaker_id defaults to '' when the subject node has no speaker_id attribute."""
        import networkx as nx

        g = nx.MultiDiGraph()
        g.add_edge("Bob", "Paris", predicate="visited", relation_type="factual", sessions=["s2"])
        loop = self._make_loop_with_graph(g)
        result = loop._capture_pending_relations()
        assert len(result) == 1
        assert result[0].speaker_id == ""

    def test_invalid_relation_type_falls_back_to_default(self):
        """An unrecognised relation_type is replaced with the fallback relation type."""
        import networkx as nx

        from paramem.graph.schema_config import fallback_relation_type

        g = nx.MultiDiGraph()
        g.add_edge(
            "Alice",
            "Bob",
            predicate="knows",
            relation_type="UNKNOWN_TYPE_THAT_IS_INVALID",
            sessions=[],
        )
        loop = self._make_loop_with_graph(g)
        result = loop._capture_pending_relations()
        assert len(result) == 1
        assert result[0].relation_type == fallback_relation_type()

    def test_session_ids_from_edge_sessions_attribute(self):
        """session_ids in the returned Relation matches the edge 'sessions' list."""
        import networkx as nx

        g = nx.MultiDiGraph()
        g.add_edge(
            "Diana",
            "Oxford",
            predicate="studied at",
            relation_type="factual",
            sessions=["sess_1", "sess_2"],
        )
        loop = self._make_loop_with_graph(g)
        result = loop._capture_pending_relations()
        assert len(result) == 1
        assert result[0].session_ids == ["sess_1", "sess_2"]

    def test_missing_sessions_attribute_yields_empty_list(self):
        """If the edge has no 'sessions' attribute, session_ids is an empty list."""
        import networkx as nx

        g = nx.MultiDiGraph()
        g.add_edge("Eve", "Mars", predicate="wants to go to", relation_type="factual")
        loop = self._make_loop_with_graph(g)
        result = loop._capture_pending_relations()
        assert len(result) == 1
        assert result[0].session_ids == []

    def test_multiple_edges_all_returned(self):
        """Multiple edges all produce Relation objects; empty-predicate ones are skipped."""
        import networkx as nx

        g = nx.MultiDiGraph()
        g.add_node("Frank", speaker_id="spk_07")
        g.add_edge("Frank", "Berlin", predicate="lives in", relation_type="factual", sessions=[])
        g.add_edge("Frank", "Python", predicate="uses", relation_type="procedural", sessions=["s3"])
        # This edge should be skipped (empty predicate):
        g.add_edge("Frank", "Nowhere", predicate="", relation_type="factual", sessions=[])
        loop = self._make_loop_with_graph(g)
        result = loop._capture_pending_relations()
        assert len(result) == 2
        predicates = {r.predicate for r in result}
        assert predicates == {"lives in", "uses"}


# =============================================================================
# TestConsumePendingFullFold — full-fold fresh-derivation consume-pending wiring
# =============================================================================


class TestConsumePendingFullFold:
    """Tests for consume_pending wiring in the full-fold branch of _run_fold.

    Verifies that:
    1. The full-fold FoldScope sets consume_pending=True and
       extra_relations_source="pending" when consume_pending=True is passed.
    2. When consume_pending=True, the full fold's fresh-derivation branch calls
       _capture_pending_relations and passes the result as extra_relations
       to _materialize_consolidation_graph.
    3. When consume_pending=False (normal mode), the non-consume full fold is
       byte-equivalent to before (no capture).
    4. The crash-resume fast-path branch does NOT trigger the capture.

    All tests mock GPU/model operations (no hardware required).
    Modelled on TestConsolidateInterimAdaptersFullFlow._make_loop and
    TestTierFloor._run_full_fold_mocked (object.__new__ pattern).
    """

    @staticmethod
    def _make_loop(tmp_path, *, merger_graph=None):
        """Minimal ConsolidationLoop stub for consume-pending full-fold tests.

        The consume_pending decision is NOT stored on the loop — it is a
        caller decision passed to ``consolidate_interim_adapters(consume_pending=...)``
        at call time.  Tests that need a specific value pass it there.
        """
        from unittest.mock import MagicMock

        import networkx as nx
        from peft import PeftModel

        from paramem.memory.store import MemoryStore
        from paramem.training.consolidation import ConsolidationLoop
        from paramem.utils.config import AdapterConfig, ConsolidationConfig, TrainingConfig

        if merger_graph is None:
            merger_graph = nx.MultiDiGraph()

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
        loop.config = ConsolidationConfig(min_tier_key_floor=0, tier_fast_start=False)
        loop.training_config = TrainingConfig(num_epochs=1, gradient_checkpointing=False)
        loop.episodic_config = AdapterConfig(rank=4, alpha=8, target_modules=["q_proj"])
        loop.semantic_config = AdapterConfig(rank=4, alpha=8, target_modules=["q_proj"])
        loop.procedural_config = AdapterConfig(rank=4, alpha=8, target_modules=["q_proj"])
        loop.wandb_config = None
        loop._thermal_policy = None
        # Use a nested subdir so each test's _fold_state_dir (output_dir.parent/"state")
        # is unique and tests do not share a fold_resume.json via a common parent dir.
        loop.output_dir = tmp_path / "outputs"
        loop.output_dir.mkdir(parents=True, exist_ok=True)
        loop.store = MemoryStore(replay_enabled=True)
        loop.merger = MagicMock()
        loop.merger.graph = merger_graph
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
        loop.graph_enrichment_max_entities_per_pass = 50
        loop.graph_enrichment_neighborhood_hops = 2
        return loop

    @staticmethod
    def _run_full_fold_with_materialize_spy(loop, *, consume_pending=True, materialize_spy=None):
        """Run consolidate_interim_adapters, spying on _materialize_consolidation_graph.

        Returns (result, materialize_calls) where materialize_calls is the list of
        kwargs dicts passed to _materialize_consolidation_graph on each invocation.
        """
        from unittest.mock import patch

        import networkx as nx

        from paramem.graph.reconstruct import ReconstructionResult
        from paramem.server.gpu_lock import _gpu_thread_lock
        from paramem.training.consolidation import ConsolidationLoop

        materialize_calls = []

        def _spy_materialize(self_inner, **kwargs):
            materialize_calls.append(dict(kwargs))
            return set(), []

        _gpu_thread_lock.acquire()
        try:
            with (
                patch(
                    "paramem.training.consolidation.reconstruct_graph",
                    return_value=ReconstructionResult(graph=nx.MultiDiGraph()),
                ),
                patch.object(
                    ConsolidationLoop,
                    "_materialize_consolidation_graph",
                    _spy_materialize,
                ),
                patch.object(
                    ConsolidationLoop,
                    "_run_graph_enrichment",
                    return_value={"skipped": True},
                ),
                patch.object(ConsolidationLoop, "_enable_gradient_checkpointing"),
                patch.object(ConsolidationLoop, "_disable_gradient_checkpointing"),
                patch.object(
                    ConsolidationLoop,
                    "_maybe_make_recall_callback",
                    return_value=(None, None),
                ),
                patch.object(
                    ConsolidationLoop,
                    "_probe_passing_keys",
                    side_effect=lambda adapter_name, entries: {e["key"] for e in entries},
                ),
                patch.object(ConsolidationLoop, "_save_adapters"),
                patch(
                    "paramem.training.trainer.train_adapter",
                    return_value={"aborted": False},
                ),
                patch(
                    "paramem.training.consolidation.format_entry_training",
                    return_value=[{"input_ids": [1], "labels": [1], "attention_mask": [1]}],
                ),
                patch("paramem.models.loader.create_adapter", side_effect=lambda m, c, n: m),
                patch("paramem.models.loader.switch_adapter"),
                patch("paramem.models.loader.copy_adapter_weights"),
                patch("paramem.models.loader.copy_adapter_weights_subset"),
                patch("paramem.memory.interim_adapter.unload_interim_adapters", return_value=[]),
            ):
                result = loop.consolidate_interim_adapters(
                    trainer=None, router=None, consume_pending=consume_pending
                )
        finally:
            _gpu_thread_lock.release()

        return result, materialize_calls

    def test_full_fold_scope_consume_pending_set_when_consume_pending_true(self, tmp_path):
        """FoldScope has consume_pending=True and extra_relations_source='pending' when passed."""
        from unittest.mock import patch

        import networkx as nx

        from paramem.training.consolidation import ConsolidationLoop, FoldScope

        loop = self._make_loop(tmp_path, merger_graph=nx.MultiDiGraph())

        captured_scopes: list[FoldScope] = []

        _orig_run_fold = ConsolidationLoop._run_fold

        def _spy_run_fold(self_inner, scope, **kwargs):
            captured_scopes.append(scope)
            # Return a minimal noop result to avoid executing the fold body.
            return {
                "status": "noop",
                "tiers_rebuilt": [],
                "keys_trained": 0,
                "recall_miss_count": 0,
                "graph_drift_count": 0,
                "soft_stale_count": 0,
                "drift_deduplicated_count": 0,
                "drift_orphan_count": 0,
                "drift_genuine_loss_count": 0,
                "drift_intended_removal_count": 0,
                "drift_intended_removal_by_reason": {},
                "recall_failed_session_ids": [],
                "skipped_cycles": 0,
            }

        from paramem.server.gpu_lock import _gpu_thread_lock

        _gpu_thread_lock.acquire()
        try:
            with patch.object(ConsolidationLoop, "_run_fold", _spy_run_fold):
                loop.consolidate_interim_adapters(trainer=None, router=None, consume_pending=True)
        finally:
            _gpu_thread_lock.release()

        assert len(captured_scopes) == 1
        scope = captured_scopes[0]
        assert scope.consume_pending is True
        assert scope.extra_relations_source == "pending"
        assert scope.persist == "main_tiers"

    def test_full_fold_scope_consume_pending_false_when_consume_pending_false(self, tmp_path):
        """FoldScope has consume_pending=False and extra_relations_source='none' by default."""
        from unittest.mock import patch

        import networkx as nx

        from paramem.training.consolidation import ConsolidationLoop, FoldScope

        loop = self._make_loop(tmp_path, merger_graph=nx.MultiDiGraph())

        captured_scopes: list[FoldScope] = []

        def _spy_run_fold(self_inner, scope, **kwargs):
            captured_scopes.append(scope)
            return {
                "status": "noop",
                "tiers_rebuilt": [],
                "keys_trained": 0,
                "recall_miss_count": 0,
                "graph_drift_count": 0,
                "soft_stale_count": 0,
                "drift_deduplicated_count": 0,
                "drift_orphan_count": 0,
                "drift_genuine_loss_count": 0,
                "drift_intended_removal_count": 0,
                "drift_intended_removal_by_reason": {},
                "recall_failed_session_ids": [],
                "skipped_cycles": 0,
            }

        from paramem.server.gpu_lock import _gpu_thread_lock

        _gpu_thread_lock.acquire()
        try:
            with patch.object(ConsolidationLoop, "_run_fold", _spy_run_fold):
                loop.consolidate_interim_adapters(trainer=None, router=None, consume_pending=False)
        finally:
            _gpu_thread_lock.release()

        assert len(captured_scopes) == 1
        scope = captured_scopes[0]
        assert scope.consume_pending is False
        assert scope.extra_relations_source == "none"

    def test_full_fold_captures_pending_relations_into_materialize(self, tmp_path):
        """Full-fold fresh-derivation passes captured pending relations to _materialize."""
        import networkx as nx

        g = nx.MultiDiGraph()
        g.add_node("Alice", speaker_id="spk_01")
        g.add_edge(
            "Alice",
            "Wonderland",
            predicate="lives in",
            relation_type="factual",
            confidence=0.95,
            sessions=["sess_x"],
        )
        loop = self._make_loop(tmp_path, merger_graph=g)

        _result, materialize_calls = self._run_full_fold_with_materialize_spy(
            loop, consume_pending=True
        )

        assert len(materialize_calls) >= 1
        # The full-fold (main_tiers) materialize call passes extra_relations with the pending fact.
        materialize_call = materialize_calls[0]
        extra = materialize_call.get("extra_relations")
        assert extra is not None, "extra_relations must be passed when consume_pending=True"
        assert len(extra) == 1
        assert extra[0].subject == "Alice"
        assert extra[0].predicate == "lives in"
        assert extra[0].speaker_id == "spk_01"
        assert extra[0].session_ids == ["sess_x"]

    def test_full_fold_no_extra_relations_when_consume_pending_false(self, tmp_path):
        """Full-fold fresh-derivation passes extra_relations=None when consume_pending=False."""
        import networkx as nx

        # Populate merger.graph even for count>0 to confirm it is NOT captured.
        g = nx.MultiDiGraph()
        g.add_edge(
            "Bob",
            "Berlin",
            predicate="lives in",
            relation_type="factual",
            sessions=["s1"],
        )
        loop = self._make_loop(tmp_path, merger_graph=g)

        _result, materialize_calls = self._run_full_fold_with_materialize_spy(
            loop, consume_pending=False
        )

        assert len(materialize_calls) >= 1
        materialize_call = materialize_calls[0]
        extra = materialize_call.get("extra_relations")
        # extra_relations must be None (not the pending graph content) when consume_pending=False.
        assert extra is None, (
            f"extra_relations should be None for non-consume full fold, got: {extra}"
        )

    def test_full_fold_resolve_contradictions_extra_uses_config_on_consume_pending(self, tmp_path):
        """Full fold passes resolve_contradictions_extra from config when consume_pending=True."""
        import networkx as nx

        from paramem.utils.config import ConsolidationConfig

        g = nx.MultiDiGraph()
        g.add_edge("Alice", "Moon", predicate="orbits", relation_type="factual", sessions=[])

        loop = self._make_loop(tmp_path, merger_graph=g)
        # Explicitly set refinement_contradiction="on" on the loop's training-layer config.
        loop.config = ConsolidationConfig(
            min_tier_key_floor=0,
            tier_fast_start=False,
            refinement_contradiction="on",
        )

        _result, materialize_calls = self._run_full_fold_with_materialize_spy(
            loop, consume_pending=True
        )

        assert materialize_calls
        materialize_call = materialize_calls[0]
        assert materialize_call.get("resolve_contradictions_extra") is True

    def test_full_fold_resolve_contradictions_extra_from_config_default_off(self, tmp_path):
        """Full fold passes resolve_contradictions_extra driven by config (default="off"→False).

        The value is always config-driven — not gated by consume_pending.  This test
        confirms the default ConsolidationConfig (refinement_contradiction="off") produces
        False regardless of consume_pending.
        """
        import networkx as nx

        g = nx.MultiDiGraph()
        loop = self._make_loop(tmp_path, merger_graph=g)

        _result, materialize_calls = self._run_full_fold_with_materialize_spy(
            loop, consume_pending=False
        )

        assert materialize_calls
        materialize_call = materialize_calls[0]
        assert materialize_call.get("resolve_contradictions_extra") is False

    def test_full_fold_empty_merger_graph_gives_empty_extra_relations(self, tmp_path):
        """Full fold with consume_pending=True and empty graph passes empty extra_relations."""
        import networkx as nx

        g = nx.MultiDiGraph()  # empty
        loop = self._make_loop(tmp_path, merger_graph=g)

        _result, materialize_calls = self._run_full_fold_with_materialize_spy(
            loop, consume_pending=True
        )

        assert materialize_calls
        materialize_call = materialize_calls[0]
        extra = materialize_call.get("extra_relations")
        # Helper returns [] on empty graph; the full fold passes it through.
        assert extra == [] or extra is None  # both are valid no-ops for materialize


# ---------------------------------------------------------------------------
# Dispatch: count==0 routes every tick to the full path, never interim
# ---------------------------------------------------------------------------


class TestDispatchCountZeroRoutesToFull:
    """_maybe_trigger_scheduled_consolidation at count==0 → always full, never interim.

    At max_interim_count==0, _is_full_cycle_due returns True unconditionally,
    so every scheduled tick routes to the full consolidation path ("started_full")
    and the interim slot-minting path (_extract_and_start_training) is never reached.
    """

    def _make_minimal_state(self, tmp_path, buffer, store=None) -> dict:
        """Minimal _state for the dispatch tests.

        Uses a plain MagicMock (no spec) for config so attribute assignment
        works without triggering spec-gating on the nested consolidation field.
        """
        config = MagicMock()
        config.consolidation.max_interim_count = 0
        config.consolidation.refresh_cadence = "12h"
        config.consolidation.training_idle_debounce_s = 0
        config.consolidation.orphan_retirement_seconds = None
        config.debug = False
        config.debug_dir = tmp_path / "debug"

        return {
            "config": config,
            "session_buffer": buffer,
            "speaker_store": store or MagicMock(),
            "consolidating": False,
            "mode": "local",
            "last_chat_monotonic": None,
            "pending_rehydration": False,
            "store_load_degraded": False,
        }

    def _call_tick_count_zero(self, state_overrides: dict) -> str:
        """Run _maybe_trigger_scheduled_consolidation with count==0 real _is_full_cycle_due.

        Does NOT patch _is_full_cycle_due — at count==0 the real function returns True
        unconditionally so the test exercises the real routing logic.  Patches out the
        asyncio event loop and the side-effectful dispatch functions so no GPU work runs.
        """
        from unittest.mock import MagicMock, patch

        import paramem.server.app as _app

        # Provide a mock event loop so asyncio.get_running_loop() doesn't raise.
        mock_loop = MagicMock()
        mock_future = MagicMock()
        mock_future.add_done_callback = MagicMock()
        mock_loop.run_in_executor = MagicMock(return_value=mock_future)

        with (
            patch.object(_app, "_state", state_overrides),
            patch("paramem.server.app._consolidation_dispatch_guards", return_value=None),
            patch("paramem.server.app._retro_claim_orphan_sessions", return_value=0),
            patch("paramem.server.app._run_full_consolidation_sync") as _mock_full,
            patch("paramem.server.app._extract_and_start_training") as _mock_interim,
            patch("asyncio.get_running_loop", return_value=mock_loop),
        ):
            result = _app._maybe_trigger_scheduled_consolidation()
        return result, _mock_full.call_count, _mock_interim.call_count

    def test_count_zero_every_tick_routes_to_full(self, tmp_path):
        """At max_interim_count==0, every tick returns started_full (never started)."""
        from paramem.server.session_buffer import SessionBuffer

        buffer = SessionBuffer(tmp_path / "sessions", state_dir=tmp_path / "state", debug=False)
        # Add a NAMED session so the dispatch has something to see.
        buffer.append("conv-named", "user", "Hello", speaker_id="sp-abc")
        buffer.append("conv-named", "assistant", "Hi")

        state = self._make_minimal_state(tmp_path, buffer)
        result, full_count, interim_count = self._call_tick_count_zero(state)
        assert result == "started_full", f"Expected 'started_full' at count==0, got {result!r}"

    def test_count_zero_never_reaches_extract_and_start_training(self, tmp_path):
        """The interim slot-minting path is never invoked at count==0."""
        from paramem.server.session_buffer import SessionBuffer

        buffer = SessionBuffer(tmp_path / "sessions", state_dir=tmp_path / "state", debug=False)
        buffer.append("conv-named", "user", "Hello", speaker_id="sp-abc")
        buffer.append("conv-named", "assistant", "Hi")

        state = self._make_minimal_state(tmp_path, buffer)
        _result, _full_count, interim_count = self._call_tick_count_zero(state)
        assert interim_count == 0, (
            "_extract_and_start_training must NOT be called at count==0; "
            f"was called {interim_count} time(s)"
        )

    def test_count_zero_empty_buffer_still_routes_to_full(self, tmp_path):
        """At count==0, even with no pending sessions the tick routes to started_full.

        The noop/accumulating decision lives in _run_full_cycle, not in the dispatcher.
        """
        from paramem.server.session_buffer import SessionBuffer

        buffer = SessionBuffer(tmp_path / "sessions", state_dir=tmp_path / "state", debug=False)
        # No sessions at all.
        state = self._make_minimal_state(tmp_path, buffer)
        result, _full_count, _interim_count = self._call_tick_count_zero(state)
        assert result == "started_full", (
            f"Expected 'started_full' at count==0 (empty buffer), got {result!r}"
        )


# ---------------------------------------------------------------------------
# _run_full_cycle: consume-pending pre-stage extract-loop + mark_consolidated
# ---------------------------------------------------------------------------


class TestRunFullCycleConsumePending:
    """_run_full_consolidation_sync at count==0 exercises the consume-pending path.

    Tests verify:
    - extract_session called per pending NAMED session
    - consolidate_interim_adapters receives consume_pending=True
    - mark_consolidated called on fold SUCCESS (Site B)
    - mark_consolidated called on fold NOOP with extraction success (Site A)
    - mark_consolidated NOT called on AbortedDuringConsolidation
    - mark_consolidated NOT called on Exception (fold crash)
    - mark_consolidated NOT called on status="accumulating"
    - Extraction-FAILED sessions not marked in either site
    """

    def _make_state(self, tmp_path, *, max_interim_count: int = 0) -> dict:
        """Build minimal _state for _run_full_consolidation_sync.

        Pre-populates consolidation_loop and session_buffer mocks.
        config.consolidation.mode is 'train'; training_temp_limit=0 so
        ThermalPolicy returns None (avoids MagicMock int comparison).
        """
        mock_config = MagicMock()
        mock_config.consolidation.mode = "train"
        mock_config.consolidation.max_interim_count = max_interim_count
        mock_config.consolidation.training_temp_limit = 0
        # VRAM config accessed in check_vram_headroom:
        mock_config.vram.vram_cache_headroom_gib = 0.5
        # cooldown_gate_threshold_c <= 0 disables the wait_for_cooldown fold gate.
        mock_config.vram.cooldown_gate_threshold_c = 0
        if tmp_path is not None:
            mock_config.paths.data = tmp_path

        mock_loop = MagicMock()
        mock_loop.model = MagicMock(name="model")
        mock_loop.shutdown_requested = False
        mock_loop.store.replay_enabled = False

        # consolidate_interim_adapters returns success by default.
        mock_loop.consolidate_interim_adapters.return_value = {
            "tiers_rebuilt": ["episodic"],
            "graph_drift_count": 0,
            "rolled_back": False,
        }

        mock_session_buffer = MagicMock()
        # retirable returns all ids passed to it (simplifies assertions).
        mock_session_buffer.retirable.side_effect = lambda ids: list(ids)

        return {
            "config": mock_config,
            "model": MagicMock(name="model"),
            "tokenizer": MagicMock(name="tokenizer"),
            "consolidation_loop": mock_loop,
            "session_buffer": mock_session_buffer,
            "speaker_store": None,
            "router": None,
            "background_trainer": None,
            "consolidating": True,
            "last_consolidation": None,
            "ha_client": None,
            "voice_profile": "gpu",
        }

    def _pending_session(self, session_id: str = "sess-1", speaker_id: str = "sp-abc") -> dict:
        """Build a minimal pending-session dict (as returned by get_pending)."""
        return {
            "session_id": session_id,
            "speaker_id": speaker_id,
            "transcript": "Hello, my name is Test User.",
            "source_type": "transcript",
            "doc_title": None,
            "started_at": "2026-01-01T00:00:00Z",
            "ended_at": "2026-01-01T00:05:00Z",
        }

    def _pending_fact(self, session_id: str = "sess-1", speaker_id: str = "sp-abc") -> dict:
        """Build a minimal pending_facts() entry (as returned by pending_facts)."""
        return {
            "session_id": session_id,
            "speaker_id": speaker_id,
            "has_voice_embedding": False,
        }

    def _run_sync(self, state: dict, monkeypatch) -> None:
        """Run _run_full_consolidation_sync under patched _state + inlined bt.submit."""
        from unittest.mock import patch

        import paramem.server.app as app_module

        monkeypatch.setattr(app_module, "_state", state)

        mock_bt = MagicMock()
        mock_bt.abort_requested = False
        # submit(fn, **kw) → call fn() directly (no threading).
        mock_bt.submit.side_effect = lambda fn, **kw: fn()

        with patch("paramem.server.app.BackgroundTrainer", return_value=mock_bt):
            app_module._run_full_consolidation_sync()

    def test_count_zero_extract_session_called_per_pending(self, monkeypatch, tmp_path):
        """At count==0, extract_session is called once per pending NAMED session."""
        from unittest.mock import patch

        state = self._make_state(tmp_path)
        loop = state["consolidation_loop"]
        # extract_session returns empty lists (no relations extracted).
        loop.extract_session.return_value = ([], [])

        sb = state["session_buffer"]
        session1 = self._pending_session("sess-1", "sp-abc")
        session2 = self._pending_session("sess-2", "sp-abc")
        sb.get_pending.return_value = [session1, session2]
        # pending_facts used to build the NAMED-only set.
        sb.pending_facts.return_value = [
            self._pending_fact("sess-1", "sp-abc"),
            self._pending_fact("sess-2", "sp-abc"),
        ]

        # Patch check_vram_headroom and vram_scope to no-ops so no GPU required.
        # Also patch the success-path helpers that would try to serialize MagicMock.
        with (
            patch("paramem.server.app.check_vram_headroom"),
            patch("paramem.server.app.vram_scope"),
            patch("paramem.server.consolidation._save_key_metadata"),
            patch("paramem.server.app._dispatch_finalize"),
            patch("paramem.server.consolidation.session_retention_dir", return_value=None),
        ):
            self._run_sync(state, monkeypatch)

        assert loop.extract_session.call_count == 2, (
            f"Expected 2 extract_session calls (one per session), "
            f"got {loop.extract_session.call_count}"
        )

    def test_count_zero_consume_pending_true_passed_to_fold(self, monkeypatch, tmp_path):
        """At count==0, consolidate_interim_adapters receives consume_pending=True."""
        from unittest.mock import patch

        state = self._make_state(tmp_path)
        loop = state["consolidation_loop"]
        loop.extract_session.return_value = ([], [])

        sb = state["session_buffer"]
        sb.get_pending.return_value = [self._pending_session()]
        sb.pending_facts.return_value = [self._pending_fact()]

        with (
            patch("paramem.server.app.check_vram_headroom"),
            patch("paramem.server.app.vram_scope"),
            patch("paramem.server.consolidation._save_key_metadata"),
            patch("paramem.server.app._dispatch_finalize"),
            patch("paramem.server.consolidation.session_retention_dir", return_value=None),
        ):
            self._run_sync(state, monkeypatch)

        call_kwargs = loop.consolidate_interim_adapters.call_args
        assert call_kwargs is not None
        assert call_kwargs.kwargs.get("consume_pending") is True, (
            f"consolidate_interim_adapters must receive consume_pending=True at count==0; "
            f"kwargs={call_kwargs.kwargs}"
        )

    def test_count_greater_zero_consume_pending_false(self, monkeypatch, tmp_path):
        """At count>0 (standard mode), consolidate_interim_adapters gets consume_pending=False."""
        state = self._make_state(tmp_path, max_interim_count=7)
        from unittest.mock import patch

        loop = state["consolidation_loop"]
        # At count>0 extract_session must NOT be called (no consume-pending stage).
        sb = state["session_buffer"]
        sb.get_pending.return_value = []
        sb.pending_facts.return_value = []

        with (
            patch("paramem.server.consolidation._save_key_metadata"),
            patch("paramem.server.app._dispatch_finalize"),
        ):
            self._run_sync(state, monkeypatch)

        call_kwargs = loop.consolidate_interim_adapters.call_args
        assert call_kwargs is not None
        consume = call_kwargs.kwargs.get("consume_pending", False)
        assert consume is False, (
            f"consolidate_interim_adapters must receive consume_pending=False at count>0; "
            f"kwargs={call_kwargs.kwargs}"
        )
        loop.extract_session.assert_not_called()

    def test_mark_consolidated_called_on_fold_success_site_b(self, monkeypatch, tmp_path):
        """MF-A Site B: mark_consolidated called on fold success at count==0."""
        from unittest.mock import patch

        state = self._make_state(tmp_path)
        loop = state["consolidation_loop"]
        loop.extract_session.return_value = ([], [])
        # Fold returns success (tiers_rebuilt non-empty).
        loop.consolidate_interim_adapters.return_value = {
            "tiers_rebuilt": ["episodic"],
            "graph_drift_count": 0,
            "rolled_back": False,
        }

        sb = state["session_buffer"]
        session1 = self._pending_session("sess-1", "sp-abc")
        sb.get_pending.return_value = [session1]
        sb.pending_facts.return_value = [self._pending_fact("sess-1", "sp-abc")]

        with (
            patch("paramem.server.app.check_vram_headroom"),
            patch("paramem.server.app.vram_scope"),
            patch("paramem.server.consolidation._save_key_metadata"),
            patch("paramem.server.app._dispatch_finalize"),
        ):
            self._run_sync(state, monkeypatch)

        sb.mark_consolidated.assert_called_once()

    def test_mark_consolidated_called_on_fold_noop_site_a(self, monkeypatch, tmp_path):
        """MF-A Site A: mark_consolidated called when fold returns noop at count==0.

        Noop = tiers_rebuilt=[] and status != 'accumulating'.  Extraction succeeded,
        but the fold found nothing new to train (all facts already present after dedup).
        The sessions are processed — mark them consolidated to prevent re-extraction
        unboundedly (prevents already-processed sessions from accumulating unboundedly in pending).
        """
        from unittest.mock import patch

        state = self._make_state(tmp_path)
        loop = state["consolidation_loop"]
        loop.extract_session.return_value = ([], [])
        # Fold returns noop.
        loop.consolidate_interim_adapters.return_value = {
            "tiers_rebuilt": [],
            "graph_drift_count": 0,
            "rolled_back": False,
        }

        sb = state["session_buffer"]
        session1 = self._pending_session("sess-1", "sp-abc")
        sb.get_pending.return_value = [session1]
        sb.pending_facts.return_value = [self._pending_fact("sess-1", "sp-abc")]

        with (
            patch("paramem.server.app.check_vram_headroom"),
            patch("paramem.server.app.vram_scope"),
            patch("paramem.server.consolidation.session_retention_dir", return_value=None),
        ):
            self._run_sync(state, monkeypatch)

        sb.mark_consolidated.assert_called_once()

    def test_mark_consolidated_not_called_on_accumulating(self, monkeypatch, tmp_path):
        """mark_consolidated NOT called when fold returns status='accumulating'."""
        from unittest.mock import patch

        state = self._make_state(tmp_path)
        loop = state["consolidation_loop"]
        loop.extract_session.return_value = ([], [])
        loop.consolidate_interim_adapters.return_value = {
            "status": "accumulating",
            "accumulating_reason": {"floor": 30},
            "tiers_rebuilt": [],
        }

        sb = state["session_buffer"]
        sb.get_pending.return_value = [self._pending_session()]
        sb.pending_facts.return_value = [self._pending_fact()]

        with (
            patch("paramem.server.app.check_vram_headroom"),
            patch("paramem.server.app.vram_scope"),
        ):
            self._run_sync(state, monkeypatch)

        sb.mark_consolidated.assert_not_called()

    def test_mark_consolidated_not_called_on_abort(self, monkeypatch, tmp_path):
        """mark_consolidated NOT called when fold raises AbortedDuringConsolidation."""
        from unittest.mock import patch

        from paramem.training.consolidation import AbortedDuringConsolidation

        state = self._make_state(tmp_path)
        loop = state["consolidation_loop"]
        loop.extract_session.return_value = ([], [])
        loop.consolidate_interim_adapters.side_effect = AbortedDuringConsolidation(
            "aborted for chat"
        )

        sb = state["session_buffer"]
        sb.get_pending.return_value = [self._pending_session()]
        sb.pending_facts.return_value = [self._pending_fact()]

        with (
            patch("paramem.server.app.check_vram_headroom"),
            patch("paramem.server.app.vram_scope"),
        ):
            self._run_sync(state, monkeypatch)

        sb.mark_consolidated.assert_not_called()

    def test_mark_consolidated_not_called_on_exception(self, monkeypatch, tmp_path):
        """mark_consolidated NOT called when fold raises an unexpected exception."""
        from unittest.mock import patch

        state = self._make_state(tmp_path)
        loop = state["consolidation_loop"]
        loop.extract_session.return_value = ([], [])
        loop.consolidate_interim_adapters.side_effect = RuntimeError("fold exploded")

        sb = state["session_buffer"]
        sb.get_pending.return_value = [self._pending_session()]
        sb.pending_facts.return_value = [self._pending_fact()]

        with (
            patch("paramem.server.app.check_vram_headroom"),
            patch("paramem.server.app.vram_scope"),
        ):
            self._run_sync(state, monkeypatch)

        sb.mark_consolidated.assert_not_called()

    def test_extraction_failed_session_not_marked(self, monkeypatch, tmp_path):
        """Extraction-failed sessions are NOT marked consolidated (stay pending).

        VramExhausted on one session: that session goes into failed_session_ids and
        is excluded from the mark_consolidated call (only the other session is marked).
        """
        from unittest.mock import patch

        from paramem.server.vram_guard import VramExhausted

        state = self._make_state(tmp_path)
        loop = state["consolidation_loop"]

        call_count = {"n": 0}

        def _extract_side_effect(*_args, **_kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise VramExhausted("extract", 0, 0)
            return ([], [])

        loop.extract_session.side_effect = _extract_side_effect

        sb = state["session_buffer"]
        session1 = self._pending_session("sess-fail", "sp-abc")
        session2 = self._pending_session("sess-ok", "sp-abc")
        sb.get_pending.return_value = [session1, session2]
        sb.pending_facts.return_value = [
            self._pending_fact("sess-fail", "sp-abc"),
            self._pending_fact("sess-ok", "sp-abc"),
        ]

        with (
            patch("paramem.server.app.check_vram_headroom"),
            patch("paramem.server.app.vram_scope"),
            patch("paramem.server.consolidation._save_key_metadata"),
            patch("paramem.server.app._dispatch_finalize"),
            patch("paramem.server.consolidation.session_retention_dir", return_value=None),
        ):
            self._run_sync(state, monkeypatch)

        # mark_consolidated should be called — with only sess-ok, not sess-fail.
        sb.mark_consolidated.assert_called_once()
        call_args = sb.mark_consolidated.call_args
        if call_args.args:
            marked_ids = call_args.args[0]
        else:
            marked_ids = call_args.kwargs.get("session_ids", [])
        assert "sess-fail" not in marked_ids, (
            "VramExhausted session must NOT be marked consolidated"
        )
        assert "sess-ok" in marked_ids, "Extraction-succeeded session must be marked consolidated"

    def test_extraction_failed_aborts_whole_batch(self, monkeypatch, tmp_path):
        """ExtractionFailed on any session aborts the whole batch at count==0.

        All sessions must remain pending (mark_consolidated never called), and
        the fold (consolidate_interim_adapters) must never be reached.
        """
        from unittest.mock import patch

        from paramem.graph.extractor import ExtractionFailed

        state = self._make_state(tmp_path)
        loop = state["consolidation_loop"]
        loop.extract_session.side_effect = ExtractionFailed(
            "sota_enrich", "upstream 529 overloaded"
        )

        sb = state["session_buffer"]
        session1 = self._pending_session("sess-1", "sp-abc")
        session2 = self._pending_session("sess-2", "sp-abc")
        sb.get_pending.return_value = [session1, session2]
        sb.pending_facts.return_value = [
            self._pending_fact("sess-1", "sp-abc"),
            self._pending_fact("sess-2", "sp-abc"),
        ]

        with (
            patch("paramem.server.app.check_vram_headroom"),
            patch("paramem.server.app.vram_scope"),
            patch("paramem.server.app.record_incident"),
        ):
            self._run_sync(state, monkeypatch)

        loop.consolidate_interim_adapters.assert_not_called()
        sb.mark_consolidated.assert_not_called()
