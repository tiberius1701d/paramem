"""Unit tests for ConsolidationLoop in quad (indexed_format="quad") mode.

Covers:
- _is_quad flag set from ConsolidationConfig.indexed_format.
- _cache_entry builds uniform Option-B shape (source_* aliases always present;
  question/answer absent in quad mode).
- dedup_episodic works on both QA-format and quad-format dicts.
- dedup_episodic deduplicates by (s,p,o) in quad mode.
- seed_episodic_cache / seed_semantic_cache / seed_procedural_cache normalise via
  _cache_entry (source_* present after seed).
- _quads_from_graph builds relation dicts from session_graph relations and
  entity attributes (attribute-projection canary).
- assign_quad_keys prefix param: default "graph", "proc" for procedural.

No GPU required — model interactions replaced with MagicMock stubs.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from paramem.training.consolidation import ConsolidationLoop
from paramem.utils.config import ConsolidationConfig, TrainingConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_loop(*, indexed_format: str = "qa") -> ConsolidationLoop:
    """Build a ConsolidationLoop via __new__ with only the attributes needed for unit tests.

    Bypasses __init__ (which touches real adapters / PEFT).  Sets the minimum
    attributes used by the helpers under test.
    """
    loop = ConsolidationLoop.__new__(ConsolidationLoop)
    loop.model = MagicMock()
    loop.tokenizer = MagicMock()
    loop.training_config = TrainingConfig()
    loop.shutdown_requested = False
    loop._thermal_policy = None

    cfg = ConsolidationConfig(indexed_format=indexed_format)
    loop._indexed_format = getattr(cfg, "indexed_format", "qa")
    loop._is_quad = loop._indexed_format == "quad"

    # Minimal state expected by seeding helpers.
    loop.indexed_key_cache: dict[str, Any] = {}
    loop.indexed_key_registry = None  # seeding helpers guard with `is not None`
    loop.procedural_sp_index: dict = {}
    loop.cycle_count: int = 1
    return loop


def _make_quad_kp(**overrides) -> dict:
    """Build a minimal quad-format keyed_pair dict."""
    base = {
        "key": "graph1",
        "subject": "Alice",
        "predicate": "lives_in",
        "object": "Berlin",
        "speaker_id": "Speaker0",
        "first_seen_cycle": 1,
    }
    base.update(overrides)
    return base


def _make_qa_kp(**overrides) -> dict:
    """Build a minimal QA-format keyed_pair dict (8-field schema)."""
    base = {
        "key": "graph1",
        "question": "Where does Alice live?",
        "answer": "Alice lives in Berlin.",
        "source_subject": "Alice",
        "source_predicate": "lives_in",
        "source_object": "Berlin",
        "speaker_id": "Speaker0",
        "first_seen_cycle": 1,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Tests: _is_quad flag
# ---------------------------------------------------------------------------


class TestIsQuadFlag:
    def test_default_qa_mode(self):
        loop = _make_loop(indexed_format="qa")
        assert loop._is_quad is False
        assert loop._indexed_format == "qa"

    def test_quad_mode(self):
        loop = _make_loop(indexed_format="quad")
        assert loop._is_quad is True
        assert loop._indexed_format == "quad"

    def test_unknown_format_not_quad(self):
        loop = _make_loop(indexed_format="unknown")
        assert loop._is_quad is False


# ---------------------------------------------------------------------------
# Tests: _cache_entry helper — uniform Option-B shape
# ---------------------------------------------------------------------------


class TestCacheEntry:
    def test_qa_mode_includes_question_answer(self):
        loop = _make_loop(indexed_format="qa")
        entry = loop._cache_entry(
            key="graph1",
            subject="Alice",
            predicate="lives_in",
            object="Berlin",
            speaker_id="Speaker0",
            first_seen_cycle=1,
            question="Where?",
            answer="Berlin.",
        )
        assert entry["key"] == "graph1"
        assert entry["subject"] == "Alice"
        assert entry["predicate"] == "lives_in"
        assert entry["object"] == "Berlin"
        # source_* aliases always present
        assert entry["source_subject"] == "Alice"
        assert entry["source_predicate"] == "lives_in"
        assert entry["source_object"] == "Berlin"
        assert entry["speaker_id"] == "Speaker0"
        assert entry["first_seen_cycle"] == 1
        assert entry["question"] == "Where?"
        assert entry["answer"] == "Berlin."

    def test_quad_mode_no_question_answer(self):
        loop = _make_loop(indexed_format="quad")
        entry = loop._cache_entry(
            key="graph1",
            subject="Alice",
            predicate="lives_in",
            object="Berlin",
            speaker_id="Speaker0",
            first_seen_cycle=1,
        )
        # source_* aliases always present
        assert entry["source_subject"] == "Alice"
        assert entry["source_predicate"] == "lives_in"
        assert entry["source_object"] == "Berlin"
        # question/answer absent in quad mode
        assert "question" not in entry
        assert "answer" not in entry

    def test_source_aliases_equal_primary(self):
        loop = _make_loop(indexed_format="quad")
        entry = loop._cache_entry(
            key="graph2",
            subject="Bob",
            predicate="works_at",
            object="ACME",
            speaker_id="Speaker1",
            first_seen_cycle=3,
        )
        assert entry["source_subject"] == entry["subject"]
        assert entry["source_predicate"] == entry["predicate"]
        assert entry["source_object"] == entry["object"]

    def test_question_none_omitted(self):
        loop = _make_loop(indexed_format="qa")
        entry = loop._cache_entry(
            key="graph1",
            subject="A",
            predicate="p",
            object="B",
            speaker_id="s",
            first_seen_cycle=1,
        )
        assert "question" not in entry
        assert "answer" not in entry


# ---------------------------------------------------------------------------
# Tests: dedup_episodic — QA and quad dicts
# ---------------------------------------------------------------------------


class TestDedupEpisodic:
    def test_dedup_qa_format_by_source_fields(self):
        qa1 = _make_qa_kp(key="graph1")
        qa2 = _make_qa_kp(key="graph2")  # same (s,p,o) as qa1
        result = ConsolidationLoop.dedup_episodic([qa1, qa2])
        assert len(result) == 1
        assert result[0]["key"] == "graph1"

    def test_dedup_quad_format_by_subject_predicate_object(self):
        q1 = _make_quad_kp(key="graph1")
        q2 = _make_quad_kp(key="graph2")  # same (s,p,o) as q1
        result = ConsolidationLoop.dedup_episodic([q1, q2])
        assert len(result) == 1
        assert result[0]["key"] == "graph1"

    def test_dedup_preserves_distinct_triples(self):
        q1 = _make_quad_kp(key="graph1", subject="Alice", predicate="lives_in", object="Berlin")
        q2 = _make_quad_kp(key="graph2", subject="Alice", predicate="works_at", object="ACME")
        result = ConsolidationLoop.dedup_episodic([q1, q2])
        assert len(result) == 2

    def test_dedup_case_insensitive(self):
        q1 = _make_quad_kp(subject="Alice", predicate="LIVES_IN", object="BERLIN")
        q2 = _make_quad_kp(subject="alice", predicate="lives_in", object="berlin")
        result = ConsolidationLoop.dedup_episodic([q1, q2])
        assert len(result) == 1

    def test_dedup_empty_list(self):
        assert ConsolidationLoop.dedup_episodic([]) == []

    def test_dedup_mixed_qa_and_quad_shapes(self):
        # QA dict uses source_* fields; quad dict uses subject/predicate/object
        # Same effective (s,p,o) → deduplicated to first occurrence.
        qa = _make_qa_kp(
            key="graph1",
            source_subject="Alice",
            source_predicate="lives_in",
            source_object="Berlin",
        )
        quad = _make_quad_kp(
            key="graph2",
            subject="Alice",
            predicate="lives_in",
            object="Berlin",
        )
        result = ConsolidationLoop.dedup_episodic([qa, quad])
        assert len(result) == 1
        assert result[0]["key"] == "graph1"


# ---------------------------------------------------------------------------
# Tests: seeding helpers normalise via _cache_entry
# ---------------------------------------------------------------------------


class TestSeedHelpers:
    def test_seed_episodic_qa_quad_format_adds_source_aliases(self):
        loop = _make_loop(indexed_format="quad")
        kp = _make_quad_kp(key="graph1")
        loop.seed_episodic_cache([kp])
        stored = loop.indexed_key_cache.get("graph1")
        assert stored is not None
        assert stored["source_subject"] == "Alice"
        assert stored["source_predicate"] == "lives_in"
        assert stored["source_object"] == "Berlin"
        assert "question" not in stored
        assert "answer" not in stored

    def test_seed_semantic_qa_quad_format(self):
        loop = _make_loop(indexed_format="quad")
        kp = _make_quad_kp(key="graph3", subject="Bob", predicate="works_at", object="ACME")
        loop.seed_semantic_cache([kp])
        stored = loop.indexed_key_cache.get("graph3")
        assert stored is not None
        assert stored["source_subject"] == "Bob"
        assert stored["source_predicate"] == "works_at"

    def test_seed_procedural_qa_quad_format(self):
        loop = _make_loop(indexed_format="quad")
        kp = _make_quad_kp(key="proc1", subject="User", predicate="prefers", object="tea")
        loop.seed_procedural_cache([kp])
        stored = loop.indexed_key_cache.get("proc1")
        assert stored is not None
        assert stored["source_subject"] == "User"

    def test_seed_qa_format_preserved_in_qa_mode(self):
        loop = _make_loop(indexed_format="qa")
        kp = _make_qa_kp(key="graph1")
        loop.seed_episodic_cache([kp])
        stored = loop.indexed_key_cache.get("graph1")
        assert stored is not None
        assert stored["question"] == "Where does Alice live?"
        assert stored["answer"] == "Alice lives in Berlin."
        assert stored["source_subject"] == "Alice"


# ---------------------------------------------------------------------------
# Tests: assign_quad_keys prefix param
# ---------------------------------------------------------------------------


class TestAssignQuadKeysPrefix:
    def test_default_prefix_graph(self):
        from paramem.training.quadruple_memory import assign_quad_keys

        triples = [("Alice", "lives_in", "Berlin")]
        result = assign_quad_keys(triples)
        assert result[0]["key"] == "graph1"

    def test_proc_prefix(self):
        from paramem.training.quadruple_memory import assign_quad_keys

        triples = [("User", "prefers", "tea")]
        result = assign_quad_keys(triples, prefix="proc")
        assert result[0]["key"] == "proc1"

    def test_start_index_combined_with_prefix(self):
        from paramem.training.quadruple_memory import assign_quad_keys

        triples = [("A", "p", "B"), ("C", "q", "D")]
        result = assign_quad_keys(triples, start_index=5, prefix="proc")
        assert result[0]["key"] == "proc5"
        assert result[1]["key"] == "proc6"


# ---------------------------------------------------------------------------
# Tests: _quads_from_graph attribute-projection canary
# ---------------------------------------------------------------------------


class TestQuadsFromGraph:
    def _make_session_graph(self, *, with_attributes: bool = False):
        """Build a minimal SessionGraph-shaped MagicMock.

        Uses plain MagicMock (no spec) so required-field schema validation
        is bypassed — only the attributes accessed by _quads_from_graph are set.
        """
        entities = []
        relations = []

        # Standard relation
        rel = MagicMock()
        rel.subject = "Alice"
        rel.predicate = "lives_in"
        rel.object = "Berlin"
        rel.relation_type = "location"
        relations.append(rel)

        if with_attributes:
            # Entity with email attribute (scalar-PII canary)
            entity = MagicMock()
            entity.name = "Alice"
            entity.attributes = {"email": "alice@example.com"}
            entities.append(entity)
        else:
            entity = MagicMock()
            entity.name = "Alice"
            entity.attributes = {}
            entities.append(entity)

        graph = MagicMock()
        graph.relations = relations
        graph.entities = entities
        return graph

    def test_returns_episodic_and_procedural_tuple(self):
        loop = _make_loop(indexed_format="quad")
        graph = self._make_session_graph()
        episodic, procedural = loop._quads_from_graph(graph, procedural_enabled=False)
        assert isinstance(episodic, list)
        assert isinstance(procedural, list)

    def test_relation_included_in_episodic(self):
        loop = _make_loop(indexed_format="quad")
        graph = self._make_session_graph()
        episodic, _ = loop._quads_from_graph(graph, procedural_enabled=False)
        subjects = [r["subject"] for r in episodic]
        assert "Alice" in subjects

    def test_attribute_projection_canary(self):
        """email attribute must appear as a relation dict in the output."""
        loop = _make_loop(indexed_format="quad")
        graph = self._make_session_graph(with_attributes=True)
        episodic, _ = loop._quads_from_graph(graph, procedural_enabled=False)
        # _flatten_entity_attributes projects email → has_email relation
        predicates = [r["predicate"] for r in episodic]
        assert any("email" in p for p in predicates), (
            "email attribute not projected into relation set — scalar-PII keying silently dropped"
        )

    def test_no_model_generate_called(self):
        """_quads_from_graph must not call model.generate."""
        loop = _make_loop(indexed_format="quad")
        graph = self._make_session_graph()
        loop._quads_from_graph(graph, procedural_enabled=False)
        loop.model.generate.assert_not_called()
