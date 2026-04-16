"""Tests for _build_session_diagnostics in experiments/dataset_probe.py.

Specifically targets the _as_count inner helper, which normalises drop-counter
values that may be lists (residual_dropped_facts, ungrounded_dropped_facts) or
ints (plausibility_dropped, etc.) or None.

Before _as_count was introduced, computing raw_fact_count via
``post_plausibility_count + sum(drops.values())`` raised:
    TypeError: unsupported operand type(s) for +: 'int' and 'list'

No GPU, no model load, no network access.
"""

from experiments.dataset_probe import _build_session_diagnostics
from experiments.utils.dataset_types import DatasetSession  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session(session_id: str = "s1", dataset: str = "perltqa") -> DatasetSession:
    """Return a minimal DatasetSession for use in diagnostics tests.

    Args:
        session_id: Unique session identifier.
        dataset: Dataset name stored in metadata.

    Returns:
        DatasetSession with the given identifiers and empty transcript.
    """
    return DatasetSession(
        session_id=session_id,
        transcript="",
        speaker_id="sp1",
        speaker_name="Alice",
        metadata={"dataset": dataset},
    )


class _FakeEntity:
    """Minimal stand-in for an entity with an entity_type attribute."""

    def __init__(self, entity_type: str = "person"):
        self.entity_type = entity_type


class _FakeRelation:
    """Minimal stand-in for a relation with a relation_type attribute."""

    def __init__(self, relation_type: str = "knows"):
        self.relation_type = relation_type


def _make_graph(diagnostics: dict, entities=None, relations=None):
    """Build a fake session_graph object with the given diagnostics.

    Args:
        diagnostics: Dict of drop-counter keys as the extractor would set them.
        entities: List of entity-like objects (default: empty list).
        relations: List of relation-like objects (default: empty list).

    Returns:
        Simple object exposing .diagnostics, .entities, and .relations.
    """

    class _FakeGraph:
        pass

    g = _FakeGraph()
    g.diagnostics = diagnostics
    g.entities = entities if entities is not None else []
    g.relations = relations if relations is not None else []
    return g


def _call_diag(session, session_graph, episodic_qa=None, procedural_rels=None):
    """Invoke _build_session_diagnostics with sensible defaults.

    Args:
        session: DatasetSession object.
        session_graph: Fake graph object.
        episodic_qa: List of QA pair dicts (default: empty list).
        procedural_rels: List of procedural relation dicts (default: empty list).

    Returns:
        Diagnostics dict from _build_session_diagnostics.
    """
    return _build_session_diagnostics(
        session=session,
        session_graph=session_graph,
        nodes_before=0,
        edges_before=0,
        nodes_after=0,
        edges_after=0,
        episodic_qa=episodic_qa if episodic_qa is not None else [],
        procedural_rels=procedural_rels if procedural_rels is not None else [],
        elapsed_seconds=1.0,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAsCountCoercesListToLength:
    """_as_count must convert list-valued drop counters to their length."""

    def test_residual_dropped_facts_list_becomes_count(self):
        """A list with two fact dicts in residual_dropped_facts yields 2."""
        graph = _make_graph({"residual_dropped_facts": [{"text": "fact1"}, {"text": "fact2"}]})
        diag = _call_diag(_make_session(), graph)
        assert diag["extraction"]["drops"]["residual_dropped_facts"] == 2

    def test_ungrounded_dropped_facts_list_becomes_count(self):
        """A list with three fact dicts in ungrounded_dropped_facts yields 3."""
        graph = _make_graph(
            {"ungrounded_dropped_facts": [{"text": "a"}, {"text": "b"}, {"text": "c"}]}
        )
        diag = _call_diag(_make_session(), graph)
        assert diag["extraction"]["drops"]["ungrounded_dropped_facts"] == 3

    def test_int_valued_drop_passes_through_unchanged(self):
        """An integer drop counter (plausibility_dropped=3) stays 3."""
        graph = _make_graph({"plausibility_dropped": 3})
        diag = _call_diag(_make_session(), graph)
        assert diag["extraction"]["drops"]["plausibility_dropped"] == 3

    def test_none_valued_drop_becomes_zero(self):
        """A None value for any drop counter is coerced to 0."""
        graph = _make_graph({"plausibility_dropped": None})
        diag = _call_diag(_make_session(), graph)
        assert diag["extraction"]["drops"]["plausibility_dropped"] == 0

    def test_missing_drop_key_becomes_zero(self):
        """A drop key absent from diagnostics defaults to 0."""
        graph = _make_graph({})
        diag = _call_diag(_make_session(), graph)
        assert diag["extraction"]["drops"]["residual_dropped_facts"] == 0
        assert diag["extraction"]["drops"]["ungrounded_dropped_facts"] == 0
        assert diag["extraction"]["drops"]["plausibility_dropped"] == 0


class TestRawFactCountRegression:
    """raw_fact_count must not raise TypeError when drops contain lists."""

    def test_no_type_error_with_mixed_list_and_int_drops(self):
        """Computing raw_fact_count with list+int drops must not raise TypeError.

        This is the direct regression guard: before _as_count, the expression
        ``post_plausibility_count + sum(drops.values())`` failed because
        ``sum`` cannot add ints and lists.
        """
        graph = _make_graph(
            {
                "residual_dropped_facts": [{"text": "x"}],  # list
                "ungrounded_dropped_facts": [{"text": "y"}, {"text": "z"}],  # list
                "plausibility_dropped": 3,  # int
            }
        )
        episodic_qa = [{"q": "Q1", "a": "A1"}, {"q": "Q2", "a": "A2"}]
        # Should not raise:
        diag = _call_diag(_make_session(), graph, episodic_qa=episodic_qa)
        assert isinstance(diag["extraction"]["raw_fact_count"], int)

    def test_raw_fact_count_equals_post_plausibility_plus_all_drops(self):
        """raw_fact_count == post_plausibility_count + sum of all coerced drops."""
        graph = _make_graph(
            {
                "residual_dropped_facts": [{"text": "x"}, {"text": "y"}],  # 2
                "ungrounded_dropped_facts": [],  # 0
                "plausibility_dropped": 5,  # 5
                "mapping_ambiguous_dropped": 1,  # 1
                "residual_leaked_triples_dropped": 0,  # 0
            }
        )
        # 3 surviving QA pairs (post_plausibility_count = 3)
        episodic_qa = [{"q": f"Q{i}", "a": f"A{i}"} for i in range(3)]
        diag = _call_diag(_make_session(), graph, episodic_qa=episodic_qa)
        # Expected: 3 + 2 + 0 + 5 + 1 + 0 = 11
        assert diag["extraction"]["raw_fact_count"] == 11

    def test_raw_fact_count_all_zeros_no_qa(self):
        """With no QA and no drops, raw_fact_count is 0."""
        graph = _make_graph({})
        diag = _call_diag(_make_session(), graph)
        assert diag["extraction"]["raw_fact_count"] == 0


class TestEntityRelationDistributions:
    """Entity and relation type distributions are correctly aggregated."""

    def test_entity_type_distribution_counted(self):
        """Entity types are counted into entity_type_distribution."""
        graph = _make_graph(
            {},
            entities=[_FakeEntity("person"), _FakeEntity("person"), _FakeEntity("place")],
            relations=[],
        )
        diag = _call_diag(_make_session(), graph)
        dist = diag["graph"]["entity_type_distribution"]
        assert dist["person"] == 2
        assert dist["place"] == 1

    def test_relation_type_distribution_counted(self):
        """Relation types are counted into relation_type_distribution."""
        graph = _make_graph(
            {},
            entities=[],
            relations=[_FakeRelation("knows"), _FakeRelation("works_at"), _FakeRelation("knows")],
        )
        diag = _call_diag(_make_session(), graph)
        dist = diag["graph"]["relation_type_distribution"]
        assert dist["knows"] == 2
        assert dist["works_at"] == 1

    def test_none_session_graph_produces_empty_distributions(self):
        """A None session_graph yields empty distributions and zero drops."""
        diag = _call_diag(_make_session(), session_graph=None)
        assert diag["graph"]["entity_type_distribution"] == {}
        assert diag["graph"]["relation_type_distribution"] == {}
        assert diag["extraction"]["drops"]["plausibility_dropped"] == 0
