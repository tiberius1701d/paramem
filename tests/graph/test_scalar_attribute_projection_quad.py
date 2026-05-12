"""Unit tests for scalar-attribute projection in the quad distillation path.

Mirrors tests/graph/test_scalar_attribute_projection.py but exercises the
quad path's _quads_from_graph helper (in ConsolidationLoop) instead of
generate_qa_from_graph.

The key invariant (project_qa_attribute_keying_gap — resolved 2026-05-07):
  Entity.attributes (email, phone, linkedin) must survive into the keyed set
  when switching from the QA path to the quad path.  _quads_from_graph calls
  relation_prep._flatten_entity_attributes so the attribute surface is
  never silently dropped.

Session graphs are represented as MagicMocks (same pattern as
test_consolidation_quad.py) to avoid Pydantic schema-validation friction
for relation_type values in unit tests.  The _flatten_entity_attributes
tests use real Entity objects since they only touch Entity.attributes.

No GPU required — template-fallback/mock only, no model.generate calls.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from paramem.graph.relation_prep import _flatten_entity_attributes
from paramem.graph.schema import Entity
from paramem.training.consolidation import ConsolidationLoop
from paramem.utils.config import TrainingConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_loop(*, indexed_format: str = "quad") -> ConsolidationLoop:
    """Build a bare ConsolidationLoop with the minimum attributes for unit tests."""
    loop = ConsolidationLoop.__new__(ConsolidationLoop)
    loop.model = MagicMock()
    loop.tokenizer = MagicMock()
    loop.training_config = TrainingConfig()
    loop._indexed_format = indexed_format
    loop._is_quad = indexed_format == "quad"
    loop.indexed_key_qa: dict[str, Any] = {}
    loop.indexed_key_registry = None
    loop.procedural_sp_index: dict = {}
    loop.cycle_count = 1
    return loop


def _make_mock_graph(
    *,
    relations: list[dict] | None = None,
    entity_attributes: dict[str, dict] | None = None,
) -> MagicMock:
    """Build a minimal session-graph MagicMock for _quads_from_graph.

    Uses MagicMock (no Pydantic schema) so tests stay free of Relation.relation_type
    enum validation.  _quads_from_graph accesses r.subject/predicate/object/
    relation_type off each relation, and entity.name/attributes off each entity.

    Args:
        relations: List of relation dicts with subject/predicate/object/relation_type.
        entity_attributes: Mapping of entity_name → attribute dict.
    """
    graph = MagicMock()

    rel_mocks = []
    for r in relations or []:
        rel = MagicMock()
        rel.subject = r["subject"]
        rel.predicate = r["predicate"]
        rel.object = r["object"]
        rel.relation_type = r.get("relation_type", "factual")
        rel_mocks.append(rel)
    graph.relations = rel_mocks

    entity_mocks = []
    for name, attrs in (entity_attributes or {}).items():
        ent = MagicMock()
        ent.name = name
        ent.attributes = attrs
        entity_mocks.append(ent)
    graph.entities = entity_mocks

    return graph


# ---------------------------------------------------------------------------
# Tests: _flatten_entity_attributes directly (uses real Entity objects)
# ---------------------------------------------------------------------------


class TestFlattenEntityAttributesDirect:
    """Direct tests of the helper invoked by _quads_from_graph."""

    def test_email_projected_as_has_email(self) -> None:
        entity = Entity(
            name="Alice", entity_type="person", attributes={"email": "alice@example.com"}
        )
        result = _flatten_entity_attributes([entity])
        assert any(r["predicate"] == "has_email" for r in result)
        assert any(r["object"] == "alice@example.com" for r in result)

    def test_phone_projected_as_has_phone(self) -> None:
        entity = Entity(name="Alice", entity_type="person", attributes={"phone": "+1 555 123 4567"})
        result = _flatten_entity_attributes([entity])
        assert any(r["predicate"] == "has_phone" for r in result)

    def test_linkedin_projected_as_has_linkedin(self) -> None:
        entity = Entity(
            name="Alice",
            entity_type="person",
            attributes={"linkedin": "linkedin.com/in/alice"},
        )
        result = _flatten_entity_attributes([entity])
        assert any(r["predicate"] == "has_linkedin" for r in result)

    def test_empty_attributes_yields_nothing(self) -> None:
        entity = Entity(name="Alice", entity_type="person", attributes={})
        result = _flatten_entity_attributes([entity])
        assert result == []

    def test_whitespace_only_value_skipped(self) -> None:
        """Values that are whitespace-only (str) are skipped."""
        entity = Entity(name="Alice", entity_type="person", attributes={"email": "   "})
        result = _flatten_entity_attributes([entity])
        assert result == []

    def test_exclude_pairs_prevents_duplicate(self) -> None:
        entity = Entity(
            name="Alice", entity_type="person", attributes={"email": "alice@example.com"}
        )
        # Simulate an explicit relation that already covers (Alice, has_email).
        exclude = {("Alice", "has_email")}
        result = _flatten_entity_attributes([entity], exclude_pairs=exclude)
        assert result == []

    def test_multiple_attributes_all_projected(self) -> None:
        entity = Entity(
            name="Bob",
            entity_type="person",
            attributes={
                "email": "bob@corp.com",
                "phone": "+44 20 1234 5678",
            },
        )
        result = _flatten_entity_attributes([entity])
        assert len(result) == 2
        predicates = {r["predicate"] for r in result}
        assert predicates == {"has_email", "has_phone"}

    def test_relation_type_is_attribute(self) -> None:
        entity = Entity(name="Carol", entity_type="person", attributes={"email": "c@c.com"})
        result = _flatten_entity_attributes([entity])
        assert all(r["relation_type"] == "attribute" for r in result)

    def test_subject_matches_entity_name(self) -> None:
        entity = Entity(name="Dave", entity_type="person", attributes={"email": "d@d.com"})
        result = _flatten_entity_attributes([entity])
        assert all(r["subject"] == "Dave" for r in result)


# ---------------------------------------------------------------------------
# Tests: _quads_from_graph attribute projection (uses MagicMock graphs)
# ---------------------------------------------------------------------------


class TestQuadsFromGraphAttributeProjection:
    """_quads_from_graph must include attribute-projected relations in the output."""

    def test_email_attribute_survives_into_episodic(self) -> None:
        """email Entity.attribute must appear as a has_email quad in episodic output.

        This is the attribute-projection canary: if _flatten_entity_attributes
        is not called inside _quads_from_graph, scalar-PII keying silently
        regresses (project_qa_attribute_keying_gap).
        """
        graph = _make_mock_graph(
            relations=[
                {
                    "subject": "Alice",
                    "predicate": "lives_in",
                    "object": "Berlin",
                    "relation_type": "factual",
                }
            ],
            entity_attributes={"Alice": {"email": "alice@example.com"}},
        )
        loop = _make_loop(indexed_format="quad")
        episodic, procedural = loop._quads_from_graph(graph, procedural_enabled=False)

        predicates = [r["predicate"] for r in episodic]
        assert "has_email" in predicates, (
            "email attribute was NOT projected into the quad episodic set — "
            "scalar-PII keying silently regressed (_flatten_entity_attributes "
            "not called in _quads_from_graph)"
        )

    def test_attribute_object_matches_entity_attribute_value(self) -> None:
        """The projected relation's object field must equal the raw attribute value."""
        graph = _make_mock_graph(
            entity_attributes={"Bob": {"phone": "+1 555 000 0001"}},
        )
        loop = _make_loop(indexed_format="quad")
        episodic, _ = loop._quads_from_graph(graph, procedural_enabled=False)
        phone_rels = [r for r in episodic if r["predicate"] == "has_phone"]
        assert phone_rels, "has_phone not found"
        assert phone_rels[0]["object"] == "+1 555 000 0001"

    def test_no_duplicate_when_explicit_relation_covers_attribute(self) -> None:
        """An explicit has_email relation prevents a second projected duplicate."""
        graph = _make_mock_graph(
            relations=[
                {
                    "subject": "Carol",
                    "predicate": "has_email",
                    "object": "c@c.com",
                    "relation_type": "factual",
                }
            ],
            entity_attributes={"Carol": {"email": "c@c.com"}},
        )
        loop = _make_loop(indexed_format="quad")
        episodic, _ = loop._quads_from_graph(graph, procedural_enabled=False)
        email_rels = [r for r in episodic if r["predicate"] == "has_email"]
        # Exactly one — the explicit relation; the attribute projection is excluded.
        assert len(email_rels) == 1

    def test_no_model_generate_called(self) -> None:
        """_quads_from_graph must NOT call model.generate."""
        graph = _make_mock_graph(
            entity_attributes={"Dave": {"email": "d@d.com"}},
        )
        loop = _make_loop(indexed_format="quad")
        loop._quads_from_graph(graph, procedural_enabled=False)
        loop.model.generate.assert_not_called()

    def test_preference_relation_routes_to_procedural_when_enabled(self) -> None:
        """Preference relations route to procedural, not episodic."""
        graph = _make_mock_graph(
            relations=[
                {
                    "subject": "Eve",
                    "predicate": "prefers",
                    "object": "tea",
                    "relation_type": "preference",
                },
                {
                    "subject": "Eve",
                    "predicate": "works_at",
                    "object": "ACME",
                    "relation_type": "factual",
                },
            ]
        )
        loop = _make_loop(indexed_format="quad")
        episodic, procedural = loop._quads_from_graph(graph, procedural_enabled=True)
        ep_preds = [r["predicate"] for r in episodic]
        proc_preds = [r["predicate"] for r in procedural]
        assert "works_at" in ep_preds
        assert "prefers" in proc_preds
        assert "prefers" not in ep_preds

    def test_empty_session_returns_empty_lists(self) -> None:
        """A session graph with no relations and no entities returns ([], [])."""
        graph = _make_mock_graph()
        loop = _make_loop(indexed_format="quad")
        episodic, procedural = loop._quads_from_graph(graph, procedural_enabled=True)
        assert episodic == []
        assert procedural == []


# ---------------------------------------------------------------------------
# Tests: assign_quad_keys receives attribute-projected relations
# ---------------------------------------------------------------------------


class TestAttributeKeysAssignment:
    """End-to-end: attribute-projected relations get graphN keys via assign_quad_keys."""

    def test_email_attribute_gets_a_graph_key(self) -> None:
        """An email attribute ends up in the episodic keyed set with a graphN key."""
        from paramem.training.quadruple_memory import assign_quad_keys

        graph = _make_mock_graph(
            entity_attributes={"Faye": {"email": "f@f.com"}},
        )
        loop = _make_loop(indexed_format="quad")
        episodic, _ = loop._quads_from_graph(graph, procedural_enabled=False)

        keyed = assign_quad_keys(
            [(r["subject"], r["predicate"], r["object"]) for r in episodic],
            start_index=1,
            prefix="graph",
        )
        predicates = [k["predicate"] for k in keyed]
        assert "has_email" in predicates, "email attribute was not assigned a graphN key"

    def test_key_prefix_is_graph_for_attribute_relations(self) -> None:
        """Attribute relations are non-preference so they land in the graph* key range."""
        from paramem.training.quadruple_memory import assign_quad_keys

        graph = _make_mock_graph(
            entity_attributes={"George": {"phone": "+1 800 000 0000"}},
        )
        loop = _make_loop(indexed_format="quad")
        episodic, _ = loop._quads_from_graph(graph, procedural_enabled=False)

        keyed = assign_quad_keys(
            [(r["subject"], r["predicate"], r["object"]) for r in episodic],
            start_index=1,
            prefix="graph",
        )
        keys = [k["key"] for k in keyed]
        assert all(k.startswith("graph") for k in keys), (
            f"Expected all keys to start with 'graph', got: {keys}"
        )
