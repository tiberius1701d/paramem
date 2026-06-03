"""Unit tests for ConsolidationLoop indexed-key helpers.

Covers:
- _cache_entry builds uniform shape with canonical fields
  (subject/predicate/object; no question/answer).
- dedup_episodic deduplicates by (s,p,o).
- seed_episodic_cache / seed_semantic_cache / seed_procedural_cache normalise via
  _cache_entry (canonical fields present after seed).
- _entries_from_graph builds relation dicts from session_graph relations and
  entity attributes (attribute-projection canary).
- assign_keys prefix param: default "graph", "proc" for procedural.

No GPU required — model interactions replaced with MagicMock stubs.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from paramem.training.consolidation import ConsolidationLoop
from paramem.utils.config import TrainingConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_loop() -> ConsolidationLoop:
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

    # Minimal state expected by seeding helpers.
    from paramem.memory.store import MemoryStore as _MS

    loop.store = _MS(replay_enabled=False)
    loop.cycle_count: int = 1
    return loop


def _make_entry_kp(**overrides) -> dict:
    """Build a minimal entry dict."""
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
    """Build a minimal dict with question/answer fields (for dedup testing)."""
    base = {
        "key": "graph1",
        "question": "Where does Alice live?",
        "answer": "Alice lives in Berlin.",
        "subject": "Alice",
        "predicate": "lives_in",
        "object": "Berlin",
        "speaker_id": "Speaker0",
        "first_seen_cycle": 1,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Tests: _cache_entry helper
# ---------------------------------------------------------------------------


class TestCacheEntry:
    def test_entry_no_question_answer(self):
        loop = _make_loop()
        entry = loop._cache_entry(
            key="graph1",
            subject="Alice",
            predicate="lives_in",
            object="Berlin",
            speaker_id="Speaker0",
            first_seen_cycle=1,
        )
        # Canonical fields present; source_* aliases NOT present
        assert entry["subject"] == "Alice"
        assert entry["predicate"] == "lives_in"
        assert entry["object"] == "Berlin"
        assert "source_subject" not in entry
        assert "source_predicate" not in entry
        assert "source_object" not in entry
        # question/answer are not stored in entry format.
        assert "question" not in entry
        assert "answer" not in entry

    def test_canonical_fields_present(self):
        """_cache_entry stores subject/predicate/object — no source_* aliases."""
        loop = _make_loop()
        entry = loop._cache_entry(
            key="graph2",
            subject="Bob",
            predicate="works_at",
            object="ACME",
            speaker_id="Speaker1",
            first_seen_cycle=3,
        )
        assert entry["subject"] == "Bob"
        assert entry["predicate"] == "works_at"
        assert entry["object"] == "ACME"
        # No source_* aliases
        assert "source_subject" not in entry
        assert "source_predicate" not in entry
        assert "source_object" not in entry

    def test_question_none_omitted(self):
        loop = _make_loop()
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
# Tests: dedup_episodic
# ---------------------------------------------------------------------------


class TestDedupEpisodic:
    def test_dedup_by_subject_predicate_object(self):
        q1 = _make_entry_kp(key="graph1")
        q2 = _make_entry_kp(key="graph2")  # same (s,p,o) as q1
        result = ConsolidationLoop.dedup_episodic([q1, q2])
        assert len(result) == 1
        assert result[0]["key"] == "graph1"

    def test_dedup_preserves_distinct_triples(self):
        q1 = _make_entry_kp(key="graph1", subject="Alice", predicate="lives_in", object="Berlin")
        q2 = _make_entry_kp(key="graph2", subject="Alice", predicate="works_at", object="ACME")
        result = ConsolidationLoop.dedup_episodic([q1, q2])
        assert len(result) == 2

    def test_dedup_case_insensitive(self):
        q1 = _make_entry_kp(subject="Alice", predicate="LIVES_IN", object="BERLIN")
        q2 = _make_entry_kp(subject="alice", predicate="lives_in", object="berlin")
        result = ConsolidationLoop.dedup_episodic([q1, q2])
        assert len(result) == 1

    def test_dedup_empty_list(self):
        assert ConsolidationLoop.dedup_episodic([]) == []

    def test_dedup_mixed_shapes(self):
        # Both shapes use subject/predicate/object (canonical fields).
        # Same effective (s,p,o) → deduplicated to first occurrence.
        qa = _make_qa_kp(
            key="graph1",
            subject="Alice",
            predicate="lives_in",
            object="Berlin",
        )
        entry = _make_entry_kp(
            key="graph2",
            subject="Alice",
            predicate="lives_in",
            object="Berlin",
        )
        result = ConsolidationLoop.dedup_episodic([qa, entry])
        assert len(result) == 1
        assert result[0]["key"] == "graph1"


class TestAssignEntryKeysPrefix:
    def test_default_prefix_graph(self):
        from paramem.memory.entry import assign_keys

        triples = [("Alice", "lives_in", "Berlin")]
        result = assign_keys(triples)
        assert result[0]["key"] == "graph1"

    def test_proc_prefix(self):
        from paramem.memory.entry import assign_keys

        triples = [("User", "prefers", "tea")]
        result = assign_keys(triples, prefix="proc")
        assert result[0]["key"] == "proc1"

    def test_start_index_combined_with_prefix(self):
        from paramem.memory.entry import assign_keys

        triples = [("A", "p", "B"), ("C", "q", "D")]
        result = assign_keys(triples, start_index=5, prefix="proc")
        assert result[0]["key"] == "proc5"
        assert result[1]["key"] == "proc6"


# ---------------------------------------------------------------------------
# Tests: _entries_from_graph attribute-projection canary
# ---------------------------------------------------------------------------


class TestEntriesFromGraph:
    def _make_session_graph(self, *, with_attributes: bool = False):
        """Build a minimal SessionGraph-shaped MagicMock.

        Uses plain MagicMock (no spec) so required-field schema validation
        is bypassed — only the attributes accessed by _entries_from_graph are set.
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
        loop = _make_loop()
        graph = self._make_session_graph()
        episodic, procedural = loop._entries_from_graph(graph, procedural_enabled=False)
        assert isinstance(episodic, list)
        assert isinstance(procedural, list)

    def test_relation_included_in_episodic(self):
        loop = _make_loop()
        graph = self._make_session_graph()
        episodic, _ = loop._entries_from_graph(graph, procedural_enabled=False)
        subjects = [r["subject"] for r in episodic]
        assert "Alice" in subjects

    def test_attribute_projection_canary(self):
        """email attribute must appear as a relation dict in the output."""
        loop = _make_loop()
        graph = self._make_session_graph(with_attributes=True)
        episodic, _ = loop._entries_from_graph(graph, procedural_enabled=False)
        # _flatten_entity_attributes projects email → has_email relation
        predicates = [r["predicate"] for r in episodic]
        assert any("email" in p for p in predicates), (
            "email attribute not projected into relation set — scalar-PII keying silently dropped"
        )

    def test_no_model_generate_called(self):
        """_entries_from_graph must not call model.generate."""
        loop = _make_loop()
        graph = self._make_session_graph()
        loop._entries_from_graph(graph, procedural_enabled=False)
        loop.model.generate.assert_not_called()
