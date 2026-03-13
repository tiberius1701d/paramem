"""Tests for entity profile builder."""

import networkx as nx

from paramem.training.entity_profile import (
    _get_entity_relations,
    _get_temporal_prefix,
    _normalize_predicate,
    _relation_to_sentence,
    build_all_entity_qa,
    build_entity_profile,
    profile_to_qa_pairs,
)


def _make_graph():
    """Create a test graph with known entities and relations."""
    g = nx.MultiDiGraph()
    g.add_node(
        "Alex", entity_type="person", recurrence_count=5, sessions=["s1", "s2", "s3", "s4", "s5"]
    )
    g.add_node("Heilbronn", entity_type="place", recurrence_count=3, sessions=["s1", "s2", "s3"])
    g.add_node(
        "AutoMate",
        entity_type="organization",
        recurrence_count=4,
        sessions=["s1", "s2", "s3", "s4"],
    )
    g.add_node(
        "Luna the German Shepherd", entity_type="concept", recurrence_count=2, sessions=["s1", "s3"]
    )

    g.add_edge(
        "Alex",
        "Heilbronn",
        predicate="lives_in",
        recurrence_count=3,
        sessions=["s1", "s2", "s3"],
        first_seen="s1",
        last_seen="s3",
    )
    g.add_edge(
        "Alex",
        "AutoMate",
        predicate="works_at",
        recurrence_count=4,
        sessions=["s1", "s2", "s3", "s4"],
        first_seen="s1",
        last_seen="s4",
    )
    g.add_edge(
        "Alex",
        "Luna the German Shepherd",
        predicate="has_pet",
        recurrence_count=2,
        sessions=["s1", "s3"],
        first_seen="s1",
        last_seen="s3",
    )
    return g


class TestNormalizePredicate:
    def test_basic(self):
        assert _normalize_predicate("works_at") == "works_at"

    def test_spaces(self):
        assert _normalize_predicate("works at") == "works_at"

    def test_mixed(self):
        assert _normalize_predicate(" Works At ") == "works_at"

    def test_hyphens(self):
        assert _normalize_predicate("works-at") == "works_at"


class TestRelationToSentence:
    def test_known_predicate(self):
        rel = {"subject": "Alex", "predicate": "lives_in", "object": "Heilbronn"}
        s = _relation_to_sentence(rel)
        assert s == "Alex lives in Heilbronn."

    def test_unknown_predicate_uses_generic(self):
        rel = {"subject": "Alex", "predicate": "discovered", "object": "a bug"}
        s = _relation_to_sentence(rel)
        assert s == "Alex discovered a bug."

    def test_with_timestamp(self):
        rel = {"subject": "Alex", "predicate": "works_at", "object": "AutoMate"}
        s = _relation_to_sentence(rel, timestamp="As of session_004")
        assert s == "As of session_004, alex works at AutoMate."

    def test_without_timestamp(self):
        rel = {"subject": "Alex", "predicate": "works_at", "object": "AutoMate"}
        s = _relation_to_sentence(rel, timestamp=None)
        assert s == "Alex works at AutoMate."


class TestGetTemporalPrefix:
    def test_recurring_fact(self):
        edge = {"sessions": ["s1", "s2", "s3"], "recurrence_count": 3}
        prefix = _get_temporal_prefix(edge)
        assert prefix == "As of s3"

    def test_oneshot_fact(self):
        edge = {"sessions": ["s5"], "recurrence_count": 1}
        prefix = _get_temporal_prefix(edge)
        assert prefix == "In s5"

    def test_no_sessions(self):
        edge = {"sessions": [], "recurrence_count": 0}
        prefix = _get_temporal_prefix(edge)
        assert prefix is None

    def test_missing_sessions(self):
        edge = {}
        prefix = _get_temporal_prefix(edge)
        assert prefix is None


class TestGetEntityRelations:
    def test_returns_relations(self):
        g = _make_graph()
        rels = _get_entity_relations(g, "Alex")
        assert len(rels) == 3
        predicates = {r["predicate"] for r in rels}
        assert "lives_in" in predicates
        assert "works_at" in predicates

    def test_nonexistent_entity(self):
        g = _make_graph()
        rels = _get_entity_relations(g, "Nobody")
        assert rels == []

    def test_entity_with_no_outgoing(self):
        g = _make_graph()
        rels = _get_entity_relations(g, "Heilbronn")
        assert rels == []


class TestBuildEntityProfile:
    def test_basic_profile(self):
        g = _make_graph()
        profile = build_entity_profile("Alex", g, include_timestamps=False)
        assert "Alex lives in Heilbronn" in profile
        assert "Alex works at AutoMate" in profile
        assert "Alex has a pet" in profile

    def test_with_timestamps(self):
        g = _make_graph()
        profile = build_entity_profile("Alex", g, include_timestamps=True)
        assert "As of" in profile

    def test_empty_entity(self):
        g = _make_graph()
        profile = build_entity_profile("Heilbronn", g)
        assert profile == ""

    def test_nonexistent_entity(self):
        g = _make_graph()
        profile = build_entity_profile("Nobody", g)
        assert profile == ""

    def test_max_relations_cap(self):
        g = nx.MultiDiGraph()
        g.add_node("Alex", entity_type="person")
        for i in range(20):
            target = f"target_{i}"
            g.add_node(target, entity_type="concept")
            g.add_edge("Alex", target, predicate="knows", recurrence_count=1, sessions=[f"s{i}"])

        profile = build_entity_profile("Alex", g, max_relations=5, include_timestamps=False)
        # Should have at most 5 sentences
        assert profile.count("Alex knows") <= 5

    def test_sorts_by_recurrence(self):
        g = nx.MultiDiGraph()
        g.add_node("Alex", entity_type="person")
        g.add_node("Heilbronn", entity_type="place")
        g.add_node("AutoMate", entity_type="org")

        g.add_edge("Alex", "Heilbronn", predicate="lives_in", recurrence_count=1, sessions=["s1"])
        g.add_edge(
            "Alex",
            "AutoMate",
            predicate="works_at",
            recurrence_count=5,
            sessions=["s1", "s2", "s3", "s4", "s5"],
        )

        profile = build_entity_profile("Alex", g, include_timestamps=False)
        # works_at (recurrence=5) should come before lives_in (recurrence=1)
        works_pos = profile.index("works at")
        lives_pos = profile.index("lives in")
        assert works_pos < lives_pos


class TestProfileToQaPairs:
    def test_generates_profile_and_fact_qa(self):
        relations = [
            {"subject": "Alex", "predicate": "lives_in", "object": "Heilbronn"},
            {"subject": "Alex", "predicate": "works_at", "object": "AutoMate"},
        ]
        profile = "Alex lives in Heilbronn. Alex works at AutoMate."
        qa = profile_to_qa_pairs("Alex", profile, relations)

        # Default num_variants=1: 1 profile variant + fact-specific QA (incl. reverse)
        questions = [q["question"] for q in qa]
        assert "What do you know about Alex?" in questions
        assert "Tell me about Alex." not in questions  # only 1 variant now
        assert "Where does Alex live?" in questions
        assert "Where does Alex work?" in questions
        # Reverse templates
        assert "Who lives in Heilbronn?" in questions
        assert "Who works at AutoMate?" in questions

    def test_profile_answer_is_full_profile(self):
        profile = "Alex lives in Heilbronn."
        qa = profile_to_qa_pairs("Alex", profile, [])
        profile_qa = [q for q in qa if "What do you know" in q["question"]]
        assert profile_qa[0]["answer"] == profile

    def test_empty_profile(self):
        qa = profile_to_qa_pairs("Alex", "", [])
        assert qa == []

    def test_num_variants(self):
        profile = "Alex lives in Heilbronn."
        qa_1 = profile_to_qa_pairs("Alex", profile, [], num_variants=1)
        qa_2 = profile_to_qa_pairs("Alex", profile, [], num_variants=2)
        profile_qs_1 = [q for q in qa_1 if q["answer"] == profile]
        profile_qs_2 = [q for q in qa_2 if q["answer"] == profile]
        assert len(profile_qs_1) == 1
        assert len(profile_qs_2) == 2

    def test_unknown_predicate_no_fact_qa(self):
        relations = [
            {"subject": "Alex", "predicate": "discovered", "object": "a bug"},
        ]
        profile = "Alex discovered a bug."
        qa = profile_to_qa_pairs("Alex", profile, relations)
        # Only profile QA, no fact-specific (no template for "discovered")
        assert len(qa) == 1  # 1 profile variant (default num_variants=1)

    def test_profile_cap_limits_broad_answer(self):
        g = _make_graph()
        # Add more relations so we can test the cap
        for i in range(5):
            target = f"target_{i}"
            g.add_node(target, entity_type="concept")
            g.add_edge("Alex", target, predicate="knows", recurrence_count=1, sessions=[f"s{i}"])
        relations = _get_entity_relations(g, "Alex")
        # Full profile includes all 8 relations
        full_profile = build_entity_profile("Alex", g, include_timestamps=False)
        # With graph and max_profile_relations=3, broad answer should be shorter
        qa = profile_to_qa_pairs("Alex", full_profile, relations, max_profile_relations=3, graph=g)
        profile_qa = [q for q in qa if "What do you know" in q["question"]]
        assert len(profile_qa) == 1
        # The broad answer should have at most 3 facts
        broad_answer = profile_qa[0]["answer"]
        # Count sentences (rough check)
        assert broad_answer.count(".") <= 4  # 3 sentences + possible trailing


class TestBuildAllEntityQa:
    def test_multiple_entities(self):
        g = _make_graph()
        qa = build_all_entity_qa(g, ["Alex"], include_timestamps=False)
        assert len(qa) > 0
        # All questions should reference Alex
        for q in qa:
            assert "Alex" in q["question"] or "Alex" in q["answer"]

    def test_skips_entities_without_relations(self):
        g = _make_graph()
        qa = build_all_entity_qa(g, ["Alex", "Heilbronn"], include_timestamps=False)
        # Heilbronn has no outgoing edges, so no profile QA for Heilbronn as subject
        # (reverse templates may mention Heilbronn in questions about Alex)
        heilbronn_profile_qs = [
            q for q in qa if "What do you know about Heilbronn" in q["question"]
        ]
        assert len(heilbronn_profile_qs) == 0

    def test_empty_entity_list(self):
        g = _make_graph()
        qa = build_all_entity_qa(g, [])
        assert qa == []

    def test_nonexistent_entities(self):
        g = _make_graph()
        qa = build_all_entity_qa(g, ["Nobody"])
        assert qa == []
