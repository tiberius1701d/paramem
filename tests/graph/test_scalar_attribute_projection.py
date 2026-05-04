"""Unit tests for scalar-object → Entity.attribute projection.

Scalars (URLs, emails, phone numbers, DOIs, version-tagged tool names) are
verbatim identifiers, not paraphrasable concept phrases.  They belong on
``Entity.attributes`` of the subject — alongside email/phone/linkedin
already populated by the local extractor — and bypass the grounding gate
(whose whole-word alpha-token rule rejects values whose alpha portion is
concatenated with digits in the source transcript).
"""

import pytest

from paramem.graph.extractor import (
    _is_scalar_value,
    _partition_scalar_facts,
    _project_scalar_facts_to_attributes,
)
from paramem.graph.schema import Entity, SessionGraph


class TestIsScalarValue:
    @pytest.mark.parametrize(
        "value",
        [
            "github.com/example/myrepo",
            "https://github.com/foo/bar",
            "linkedin.com/in/alex-rivera",
            "alex.smith@example.com",
            "+1 555 123 4567",
            "ROS2",
            "ROS2/Gazebo",
            "H100",
            "IPv6",
            "x86_64",
            "10.5281/zenodo.19502523",
        ],
    )
    def test_recognises_scalar_shapes(self, value):
        assert _is_scalar_value(value) is True

    @pytest.mark.parametrize(
        "value",
        [
            "safety-critical autonomous systems",
            "15 engineers",
            "100+ engineers",
            "4 countries",
            "ASIL-D",
            "ASIL-D and ASIL-B",
            "Honda's Legend SAE Level 3 system",
            "3 ADAS compute platforms",
            "new domains",
            "7 years",
            "Alex",
            "Alex Rivera",
            "Java",
            "Python",
            "Linux",
            "C++",
            "",
            "   ",
        ],
    )
    def test_rejects_concept_phrases(self, value):
        assert _is_scalar_value(value) is False


class TestPartitionScalarFacts:
    def test_splits_by_object_shape(self):
        facts = [
            {"subject": "Alex", "predicate": "uses_tool", "object": "ROS2"},
            {"subject": "Alex", "predicate": "lives_in", "object": "Berlin"},
            {
                "subject": "Paper",
                "predicate": "code_repository",
                "object": "github.com/example/myrepo",
            },
            {
                "subject": "Alex",
                "predicate": "specializes_in",
                "object": "safety-critical autonomous systems",
            },
        ]
        scalar, non_scalar = _partition_scalar_facts(facts)
        assert len(scalar) == 2
        assert len(non_scalar) == 2
        assert {f["object"] for f in scalar} == {
            "ROS2",
            "github.com/example/myrepo",
        }
        assert {f["object"] for f in non_scalar} == {
            "Berlin",
            "safety-critical autonomous systems",
        }

    def test_synthetic_facts_pass_through(self):
        # Already marked synthetic — the legacy bypass path handles them;
        # the partitioner must not reroute them.
        facts = [
            {"subject": "Alex", "predicate": "has_email", "object": "x@y.com", "synthetic": True},
        ]
        scalar, non_scalar = _partition_scalar_facts(facts)
        assert scalar == []
        assert non_scalar == facts


class TestProjectScalarFactsToAttributes:
    def test_folds_onto_existing_entity(self):
        graph = SessionGraph(
            session_id="test",
            timestamp="2026-05-04T00:00:00",
            entities=[Entity(name="Alex", entity_type="person")],
            relations=[],
        )
        scalar_facts = [
            {"subject": "Alex", "predicate": "uses_tool", "object": "ROS2"},
            {"subject": "Alex", "predicate": "has_email", "object": "x@y.com"},
        ]
        _project_scalar_facts_to_attributes(graph, scalar_facts)
        ent = next(e for e in graph.entities if e.name == "Alex")
        # has_email → "email" (leading "has_" stripped to avoid double-prefixing
        # by _flatten_entity_attributes).
        assert ent.attributes == {"uses_tool": "ROS2", "email": "x@y.com"}

    def test_creates_missing_subject_entity(self):
        graph = SessionGraph(
            session_id="test",
            timestamp="2026-05-04T00:00:00",
            entities=[],
            relations=[],
        )
        scalar_facts = [
            {
                "subject": "Paper",
                "predicate": "code_repository",
                "object": "github.com/example/myrepo",
            },
        ]
        _project_scalar_facts_to_attributes(graph, scalar_facts)
        assert len(graph.entities) == 1
        ent = graph.entities[0]
        assert ent.name == "Paper"
        assert ent.entity_type == "concept"
        assert ent.attributes == {"code_repository": "github.com/example/myrepo"}

    def test_skips_blank_fields(self):
        graph = SessionGraph(
            session_id="test",
            timestamp="2026-05-04T00:00:00",
            entities=[],
            relations=[],
        )
        scalar_facts = [
            {"subject": "", "predicate": "x", "object": "ROS2"},
            {"subject": "Alex", "predicate": "", "object": "ROS2"},
            {"subject": "Alex", "predicate": "x", "object": ""},
        ]
        _project_scalar_facts_to_attributes(graph, scalar_facts)
        assert graph.entities == []
