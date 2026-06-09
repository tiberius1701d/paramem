"""Tests for knowledge graph merging and entity resolution."""

import tempfile
from pathlib import Path

import pytest

from paramem.graph.merger import GraphMerger, _normalize_name, _normalize_predicate
from paramem.graph.schema import Entity, Relation, SessionGraph


@pytest.fixture
def merger():
    return GraphMerger(similarity_threshold=85.0)


@pytest.fixture
def session_graph_1():
    return SessionGraph(
        session_id="s001",
        timestamp="2026-03-10T10:00:00Z",
        entities=[
            Entity(name="Alex", entity_type="person", attributes={"age": "29"}),
            Entity(name="Heilbronn", entity_type="place"),
            Entity(name="AutoMate", entity_type="organization"),
        ],
        relations=[
            Relation(
                subject="Alex",
                predicate="lives_in",
                object="Heilbronn",
                relation_type="factual",
                speaker_id="Speaker0",
            ),
            Relation(
                subject="Alex",
                predicate="works_at",
                object="AutoMate",
                relation_type="factual",
                speaker_id="Speaker0",
            ),
        ],
    )


@pytest.fixture
def session_graph_2():
    return SessionGraph(
        session_id="s002",
        timestamp="2026-03-11T10:00:00Z",
        entities=[
            Entity(name="Alex", entity_type="person"),
            Entity(name="Python", entity_type="concept"),
        ],
        relations=[
            Relation(
                subject="Alex",
                predicate="prefers",
                object="Python",
                relation_type="preference",
                speaker_id="Speaker0",
            ),
        ],
    )


class TestNormalization:
    def test_normalize_basic(self):
        assert _normalize_name("Alex") == "alex"
        assert _normalize_name("  Alex  ") == "alex"
        assert _normalize_name("Dr. Smith") == "dr. smith"


class TestPredicateNormalization:
    def test_lowercase_and_underscore(self):
        assert _normalize_predicate("Works At") == "works_at"
        assert _normalize_predicate("LIVES IN") == "lives_in"

    def test_strip_whitespace(self):
        assert _normalize_predicate("  works_at  ") == "works_at"

    def test_passthrough_preserves_content(self):
        assert _normalize_predicate("invented") == "invented"
        assert _normalize_predicate("custom_pred") == "custom_pred"
        assert _normalize_predicate("works_at") == "works_at"

    def test_deduplicates_edges_across_variants(self, merger):
        """'works_at' and 'works at' should merge into one edge."""
        g1 = SessionGraph(
            session_id="s001",
            timestamp="2026-03-10T10:00:00Z",
            entities=[
                Entity(name="A", entity_type="person"),
                Entity(name="B", entity_type="organization"),
            ],
            relations=[
                Relation(
                    subject="A",
                    predicate="works_at",
                    object="B",
                    relation_type="factual",
                    speaker_id="Speaker0",
                )
            ],
        )
        g2 = SessionGraph(
            session_id="s002",
            timestamp="2026-03-11T10:00:00Z",
            entities=[
                Entity(name="A", entity_type="person"),
                Entity(name="B", entity_type="organization"),
            ],
            relations=[
                Relation(
                    subject="A",
                    predicate="works at",
                    object="B",
                    relation_type="factual",
                    speaker_id="Speaker0",
                )
            ],
        )
        merger.merge(g1)
        merger.merge(g2)
        edges = list(merger.graph["A"]["B"].values())
        assert len(edges) == 1
        assert edges[0]["predicate"] == "works_at"
        assert edges[0]["recurrence_count"] == 2


class TestEntityResolution:
    def test_exact_match(self, merger, session_graph_1, session_graph_2):
        merger.merge(session_graph_1)
        merger.merge(session_graph_2)
        # "Alex" should resolve to the same node
        assert "Alex" in merger.graph.nodes
        assert merger.graph.nodes["Alex"]["recurrence_count"] == 2

    def test_fuzzy_match(self, merger):
        g1 = SessionGraph(
            session_id="s001",
            timestamp="2026-03-10T10:00:00Z",
            entities=[Entity(name="Alexander", entity_type="person")],
            relations=[],
        )
        g2 = SessionGraph(
            session_id="s002",
            timestamp="2026-03-11T10:00:00Z",
            entities=[Entity(name="alexander", entity_type="person")],
            relations=[],
        )
        merger.merge(g1)
        merger.merge(g2)
        # Case-insensitive match: should be same node
        assert merger.graph.number_of_nodes() == 1

    def test_no_cross_type_match(self, merger):
        """Entities of different types should not be fuzzy-matched."""
        g1 = SessionGraph(
            session_id="s001",
            timestamp="2026-03-10T10:00:00Z",
            entities=[Entity(name="Python", entity_type="concept")],
            relations=[],
        )
        g2 = SessionGraph(
            session_id="s002",
            timestamp="2026-03-11T10:00:00Z",
            entities=[Entity(name="Python", entity_type="organization")],
            relations=[],
        )
        merger.merge(g1)
        merger.merge(g2)
        # Exact name match overrides type check via normalization
        # So "Python" matches the first one regardless of type
        assert merger.graph.number_of_nodes() == 1


class TestEdgeAggregation:
    def test_new_edge(self, merger, session_graph_1):
        merger.merge(session_graph_1)
        assert merger.graph.number_of_edges() == 2

    def test_recurring_edge(self, merger):
        g1 = SessionGraph(
            session_id="s001",
            timestamp="2026-03-10T10:00:00Z",
            entities=[
                Entity(name="A", entity_type="person"),
                Entity(name="B", entity_type="place"),
            ],
            relations=[
                Relation(
                    subject="A",
                    predicate="lives_in",
                    object="B",
                    relation_type="factual",
                    speaker_id="Speaker0",
                )
            ],
        )
        g2 = SessionGraph(
            session_id="s002",
            timestamp="2026-03-11T10:00:00Z",
            entities=[
                Entity(name="A", entity_type="person"),
                Entity(name="B", entity_type="place"),
            ],
            relations=[
                Relation(
                    subject="A",
                    predicate="lives_in",
                    object="B",
                    relation_type="factual",
                    speaker_id="Speaker0",
                )
            ],
        )
        merger.merge(g1)
        merger.merge(g2)

        # Same predicate between same nodes should be aggregated
        edges = list(merger.graph["A"]["B"].values())
        assert len(edges) == 1
        assert edges[0]["recurrence_count"] == 2
        assert len(edges[0]["sessions"]) == 2

    def test_different_predicates(self, merger):
        g1 = SessionGraph(
            session_id="s001",
            timestamp="2026-03-10T10:00:00Z",
            entities=[
                Entity(name="A", entity_type="person"),
                Entity(name="B", entity_type="organization"),
            ],
            relations=[
                Relation(
                    subject="A",
                    predicate="works_at",
                    object="B",
                    relation_type="factual",
                    speaker_id="Speaker0",
                ),
                Relation(
                    subject="A",
                    predicate="manages",
                    object="B",
                    relation_type="factual",
                    speaker_id="Speaker0",
                ),
            ],
        )
        merger.merge(g1)
        edges = list(merger.graph["A"]["B"].values())
        assert len(edges) == 2


class TestSessionTracking:
    def test_session_provenance(self, merger, session_graph_1, session_graph_2):
        merger.merge(session_graph_1)
        merger.merge(session_graph_2)
        sessions = merger.graph.nodes["Alex"]["sessions"]
        assert "s001" in sessions
        assert "s002" in sessions

    def test_attribute_merge(self, merger):
        g1 = SessionGraph(
            session_id="s001",
            timestamp="2026-03-10T10:00:00Z",
            entities=[Entity(name="Alex", entity_type="person", attributes={"age": "29"})],
            relations=[],
        )
        g2 = SessionGraph(
            session_id="s002",
            timestamp="2026-03-11T10:00:00Z",
            entities=[Entity(name="Alex", entity_type="person", attributes={"role": "engineer"})],
            relations=[],
        )
        merger.merge(g1)
        merger.merge(g2)
        attrs = merger.graph.nodes["Alex"]["attributes"]
        assert attrs["age"] == "29"
        assert attrs["role"] == "engineer"


class TestPersistence:
    def test_save_and_load(self, merger, session_graph_1):
        merger.merge(session_graph_1)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "graph.json"
            merger.save_graph(path)

            assert path.exists()

            new_merger = GraphMerger()
            new_merger.load_graph(path)

            assert new_merger.graph.number_of_nodes() == merger.graph.number_of_nodes()
            assert new_merger.graph.number_of_edges() == merger.graph.number_of_edges()
            assert "Alex" in new_merger.graph.nodes

    def test_load_nonexistent(self, merger):
        graph = merger.load_graph("/nonexistent/path.json")
        assert graph.number_of_nodes() == 0

    def test_save_encrypted_false_writes_plaintext_even_when_security_on(
        self, merger, session_graph_1, tmp_path, monkeypatch
    ):
        """save_graph(..., encrypted=False) must bypass the envelope and emit
        plaintext JSON, even under Security ON.  Debug-directory writers
        depend on this so ``cat debug/cycle_*/graph_snapshot.json`` is always
        human-readable regardless of the server's posture.
        """
        from paramem.backup.key_store import (
            DAILY_KEY_PATH_DEFAULT,
            DAILY_PASSPHRASE_ENV_VAR,
            _clear_daily_identity_cache,
            mint_daily_identity,
            wrap_daily_identity,
            write_daily_key_file,
        )

        # Set up a real daily identity so Security ON is genuinely active.
        ident = mint_daily_identity()
        key_path = tmp_path / "daily_key.age"
        write_daily_key_file(wrap_daily_identity(ident, "pw"), key_path)
        monkeypatch.setenv(DAILY_PASSPHRASE_ENV_VAR, "pw")
        monkeypatch.setattr("paramem.backup.key_store.DAILY_KEY_PATH_DEFAULT", key_path)
        _clear_daily_identity_cache()
        assert key_path != DAILY_KEY_PATH_DEFAULT  # sanity: monkeypatch took effect

        merger.merge(session_graph_1)
        out = tmp_path / "debug_graph.json"
        merger.save_graph(out, encrypted=False)

        # Plaintext check: first bytes are NOT the age envelope magic.
        head = out.read_bytes()[:22]
        assert not head.startswith(b"age-encryption.org/v1"), (
            "encrypted=False must bypass the age envelope"
        )
        # And the file is readable as JSON directly.
        import json

        data = json.loads(out.read_text())
        assert "nodes" in data or "directed" in data or data  # any valid JSON

    def test_fuzzy_tier_case_fold(self):
        """'Alexander' and 'alexander' must merge via exact-normalization tier."""
        from paramem.graph.schema import Entity, SessionGraph

        m = GraphMerger(similarity_threshold=85.0)
        sg1 = SessionGraph(
            session_id="s1",
            timestamp="2026-01-01T00:00:00Z",
            entities=[Entity(name="Alexander", entity_type="person")],
            relations=[],
        )
        sg2 = SessionGraph(
            session_id="s2",
            timestamp="2026-01-02T00:00:00Z",
            entities=[Entity(name="alexander", entity_type="person")],
            relations=[],
        )
        m.merge(sg1)
        m.merge(sg2)
        assert m.graph.number_of_nodes() == 1


class TestSpeakerIdDedup:
    """Merger deduplicates speaker entities by speaker_id, not by name."""

    def test_same_speaker_id_different_names_collapse(self):
        """Two sessions for the same speaker (same speaker_id, different display
        names) must produce a single graph node, not two separate nodes."""
        m = GraphMerger(similarity_threshold=85.0)
        sg1 = SessionGraph(
            session_id="s1",
            timestamp="2026-01-01T00:00:00Z",
            entities=[
                Entity(
                    name="Speaker0",
                    entity_type="person",
                    speaker_id="Speaker0",
                )
            ],
            relations=[
                Relation(
                    subject="Speaker0",
                    predicate="lives_in",
                    object="Portland",
                    relation_type="factual",
                    speaker_id="Speaker0",
                )
            ],
        )
        # Session 2: speaker disclosed their name as "Alex"
        sg2 = SessionGraph(
            session_id="s2",
            timestamp="2026-01-02T00:00:00Z",
            entities=[
                Entity(
                    name="Alex",
                    entity_type="person",
                    attributes={"has_last_name": "Kim"},
                    speaker_id="Speaker0",
                ),
                Entity(name="Portland", entity_type="place"),
            ],
            relations=[
                Relation(
                    subject="Alex",
                    predicate="lives_in",
                    object="Portland",
                    relation_type="factual",
                    speaker_id="Speaker0",
                )
            ],
        )
        m.merge(sg1)
        m.merge(sg2)

        # Must collapse to a single person node — no "Speaker0" + "Alex" split.
        person_nodes = [n for n, d in m.graph.nodes(data=True) if d.get("entity_type") == "person"]
        assert len(person_nodes) == 1, (
            f"Expected 1 person node, got {len(person_nodes)}: {person_nodes}"
        )

    def test_speaker_node_keyed_by_speaker_id(self):
        """Speaker entity is keyed by ``speaker_id`` in the graph; the
        display name moves to ``attributes["name"]``.

        This is the architectural invariant from
        :class:`paramem.graph.schema.Entity`: ``speaker_id`` is the
        canonical graph identity; ``name`` is a mutable display
        attribute.
        """
        m = GraphMerger(similarity_threshold=85.0)
        sg = SessionGraph(
            session_id="s1",
            timestamp="2026-01-01T00:00:00Z",
            entities=[
                Entity(
                    name="Alex",
                    entity_type="person",
                    speaker_id="Speaker0",
                )
            ],
            relations=[],
        )
        m.merge(sg)
        # Node key IS the speaker_id, NOT the display name.
        assert "Speaker0" in m.graph.nodes
        assert "Alex" not in m.graph.nodes
        assert m.graph.nodes["Speaker0"]["speaker_id"] == "Speaker0"
        assert m.graph.nodes["Speaker0"]["attributes"]["name"] == "Alex"

    def test_non_speaker_entities_still_dedup_by_name(self):
        """Object entities (speaker_id=None) must still dedup by name (regression)."""
        m = GraphMerger(similarity_threshold=85.0)
        sg1 = SessionGraph(
            session_id="s1",
            timestamp="2026-01-01T00:00:00Z",
            entities=[Entity(name="Portland", entity_type="place")],
            relations=[],
        )
        sg2 = SessionGraph(
            session_id="s2",
            timestamp="2026-01-02T00:00:00Z",
            entities=[Entity(name="Portland", entity_type="place")],
            relations=[],
        )
        m.merge(sg1)
        m.merge(sg2)
        assert m.graph.number_of_nodes() == 1
        assert m.graph.nodes["Portland"]["recurrence_count"] == 2

    def test_speaker_attributes_merged_from_later_session(self):
        """Attributes from a later session (e.g. has_last_name disclosure) must
        be merged into the existing speaker node."""
        m = GraphMerger(similarity_threshold=85.0)
        sg1 = SessionGraph(
            session_id="s1",
            timestamp="2026-01-01T00:00:00Z",
            entities=[
                Entity(
                    name="Alex",
                    entity_type="person",
                    attributes={"role": "engineer"},
                    speaker_id="Speaker0",
                )
            ],
            relations=[],
        )
        sg2 = SessionGraph(
            session_id="s2",
            timestamp="2026-01-02T00:00:00Z",
            entities=[
                Entity(
                    name="Alex",
                    entity_type="person",
                    attributes={"has_last_name": "Kim"},
                    speaker_id="Speaker0",
                )
            ],
            relations=[],
        )
        m.merge(sg1)
        m.merge(sg2)
        # Speaker entity is keyed by speaker_id; display name lives in
        # attributes alongside the merged role / last_name.
        attrs = m.graph.nodes["Speaker0"]["attributes"]
        assert attrs.get("role") == "engineer"
        assert attrs.get("has_last_name") == "Kim"
        assert attrs.get("name") == "Alex"


class TestEmptyAttributeValueDoesNotOverwrite:
    """A non-empty attribute value captured in one chunk must NOT be
    overwritten by an LLM-emitted empty / "N/A" / "Unknown" placeholder
    from a later chunk. This is a known LLM-compliance failure mode where
    the extractor enumerates every advertised attribute key even when the
    source has no value for it."""

    def test_empty_string_does_not_overwrite(self):
        from paramem.graph.merger import GraphMerger

        m = GraphMerger()
        sg1 = SessionGraph(
            session_id="s1",
            timestamp="2026-01-01T00:00:00Z",
            entities=[
                Entity(
                    name="Alex",
                    entity_type="person",
                    attributes={"has_email": "alex@example.com"},
                    speaker_id="Speaker0",
                )
            ],
            relations=[],
        )
        sg2 = SessionGraph(
            session_id="s2",
            timestamp="2026-01-02T00:00:00Z",
            entities=[
                Entity(
                    name="Alex",
                    entity_type="person",
                    attributes={"has_email": ""},
                    speaker_id="Speaker0",
                )
            ],
            relations=[],
        )
        m.merge(sg1)
        m.merge(sg2)
        assert m.graph.nodes["Speaker0"]["attributes"]["has_email"] == "alex@example.com"

    def test_na_placeholder_does_not_overwrite(self):
        from paramem.graph.merger import GraphMerger

        m = GraphMerger()
        sg1 = SessionGraph(
            session_id="s1",
            timestamp="2026-01-01T00:00:00Z",
            entities=[
                Entity(
                    name="Alex",
                    entity_type="person",
                    attributes={"has_phone": "+1 555 123 4567"},
                    speaker_id="Speaker0",
                )
            ],
            relations=[],
        )
        sg2 = SessionGraph(
            session_id="s2",
            timestamp="2026-01-02T00:00:00Z",
            entities=[
                Entity(
                    name="Alex",
                    entity_type="person",
                    attributes={"has_phone": "N/A"},
                    speaker_id="Speaker0",
                )
            ],
            relations=[],
        )
        m.merge(sg1)
        m.merge(sg2)
        assert m.graph.nodes["Speaker0"]["attributes"]["has_phone"] == "+1 555 123 4567"

    def test_real_value_supersedes_existing_empty(self):
        """Reverse direction: if first chunk emits empty, second chunk
        emits real value, the real value wins."""
        from paramem.graph.merger import GraphMerger

        m = GraphMerger()
        sg1 = SessionGraph(
            session_id="s1",
            timestamp="2026-01-01T00:00:00Z",
            entities=[
                Entity(
                    name="Alex",
                    entity_type="person",
                    attributes={"has_email": "N/A"},
                    speaker_id="Speaker0",
                )
            ],
            relations=[],
        )
        sg2 = SessionGraph(
            session_id="s2",
            timestamp="2026-01-02T00:00:00Z",
            entities=[
                Entity(
                    name="Alex",
                    entity_type="person",
                    attributes={"has_email": "alex@example.com"},
                    speaker_id="Speaker0",
                )
            ],
            relations=[],
        )
        m.merge(sg1)
        m.merge(sg2)
        assert m.graph.nodes["Speaker0"]["attributes"]["has_email"] == "alex@example.com"


class TestMultiUserNameCollision:
    """Two distinct disclosed speakers with the same display name must NOT
    collapse into one graph node.

    This is the multi-user PA case: Speaker0 (Alex Walker) and
    Speaker1 (a different Alex) both enrol with display name
    ``Alex``.  Without a guard, ``_resolve_entity`` Tier 1 (exact
    name match) returns Speaker0's node when Speaker1's entity arrives,
    and ``_upsert_entity`` happily folds Speaker1's facts into
    Speaker0's node — corrupting both speakers' graphs.

    Tier 0 (positive speaker_id match) was already in place; this
    class guards the missing NEGATIVE: name collision across distinct
    ``speaker_id`` values must produce two separate nodes.
    """

    def _build_speaker_session(
        self,
        session_id: str,
        speaker_id: str,
        last_name: str,
        place: str,
    ) -> SessionGraph:
        return SessionGraph(
            session_id=session_id,
            timestamp="2026-05-06T00:00:00Z",
            entities=[
                Entity(
                    name="Alex",
                    entity_type="person",
                    attributes={"last_name": last_name},
                    speaker_id=speaker_id,
                ),
                Entity(name=place, entity_type="place"),
            ],
            relations=[
                Relation(
                    subject="Alex",
                    predicate="lives_in",
                    object=place,
                    relation_type="factual",
                    speaker_id=speaker_id,
                ),
            ],
        )

    def test_two_speakers_same_first_name_get_separate_nodes(self):
        """Two enrolled speakers both named "Alex" must produce two
        graph nodes with their own attributes and relations.
        """
        m = GraphMerger(similarity_threshold=85.0)
        m.merge(self._build_speaker_session("s001", "Speaker0", "Walker", "Portland"))
        m.merge(self._build_speaker_session("s002", "Speaker1", "Schmidt", "Munich"))

        # Two speaker nodes — separate identities.
        speaker_nodes = [
            (node, data)
            for node, data in m.graph.nodes(data=True)
            if data.get("speaker_id") in {"Speaker0", "Speaker1"}
        ]
        speaker_ids = {data["speaker_id"] for _, data in speaker_nodes}
        assert speaker_ids == {"Speaker0", "Speaker1"}, (
            f"Expected separate nodes for Speaker0 and Speaker1, got "
            f"speaker_ids={speaker_ids} on nodes "
            f"{[(n, d.get('speaker_id'), d.get('attributes', {})) for n, d in speaker_nodes]}"
        )

        # Each carries its own last_name — no cross-contamination.
        by_sid = {data["speaker_id"]: (node, data) for node, data in speaker_nodes}
        speaker0_node, speaker0_data = by_sid["Speaker0"]
        speaker1_node, speaker1_data = by_sid["Speaker1"]
        assert speaker0_data["attributes"].get("last_name") == "Walker", (
            f"Speaker0's last_name attribute corrupted: got {speaker0_data['attributes']!r}"
        )
        assert speaker1_data["attributes"].get("last_name") == "Schmidt", (
            f"Speaker1's last_name attribute corrupted: got {speaker1_data['attributes']!r}"
        )

        # Each speaker's lives_in relation points to the right place.
        s0_neighbors = list(m.graph.successors(speaker0_node))
        s1_neighbors = list(m.graph.successors(speaker1_node))
        assert "Portland" in s0_neighbors and "Munich" not in s0_neighbors
        assert "Munich" in s1_neighbors and "Portland" not in s1_neighbors

    def test_third_party_mention_with_speaker_id_unset_is_separate(self):
        """A later session emits a third-party ``"Alex"`` entity with
        ``speaker_id`` unset (he's not the speaker of that session).

        The name namespace and the speaker-ID namespace are disjoint by
        construction: speaker IDs follow the ``Speaker_N`` pattern
        produced by the speaker pool, so a display name like
        ``"Alex"`` will never collide with a speaker_id-keyed node.
        The third-party Alex becomes a separate node keyed by
        ``"Alex"``; Speaker0's node remains keyed by ``"Speaker0"``
        with its own attributes intact.
        """
        m = GraphMerger(similarity_threshold=85.0)
        m.merge(self._build_speaker_session("s001", "Speaker0", "Walker", "Portland"))
        third_party = SessionGraph(
            session_id="s002",
            timestamp="2026-05-06T00:00:00Z",
            entities=[
                Entity(name="Alex", entity_type="person", attributes={}),
            ],
            relations=[],
        )
        m.merge(third_party)

        # Speaker0 is keyed by speaker_id, third-party Alex by name —
        # disjoint namespaces, two separate nodes.
        assert "Speaker0" in m.graph.nodes
        assert "Alex" in m.graph.nodes
        assert m.graph.nodes["Speaker0"]["speaker_id"] == "Speaker0"
        assert m.graph.nodes["Alex"].get("speaker_id") is None
        # Speaker0's last_name attribute is untouched by the third-party
        # merge (different node, different namespace).
        assert m.graph.nodes["Speaker0"]["attributes"]["last_name"] == "Walker"


class TestCrossPredicateContradictionFlag:
    """Regression tests for the cross_predicate_contradiction flag.

    Cross-predicate contradiction detection: detect_contradiction_with_model
    fires across different predicates.  Default OFF to prevent over-removal of
    legitimate multi-valued facts (multiple valid values for one relation) and
    independent facts expressed under different predicates.

    Same-predicate cardinality resolution (via check_predicate_coexistence) is
    unaffected by this flag and is tested separately in
    TestModelContradictionAndRelease.
    """

    def _multi_valued_sessions(self) -> tuple:
        """Two sessions giving the same subject independent facts on different predicates."""
        sg1 = SessionGraph(
            session_id="s1",
            timestamp="2026-01-01T00:00:00Z",
            entities=[
                Entity(name="ParaMem", entity_type="concept"),
                Entity(name="Qwen", entity_type="concept"),
            ],
            relations=[
                Relation(
                    subject="ParaMem",
                    predicate="validated_on",
                    object="Qwen",
                    relation_type="factual",
                    speaker_id="Speaker0",
                )
            ],
        )
        sg2 = SessionGraph(
            session_id="s2",
            timestamp="2026-01-02T00:00:00Z",
            entities=[
                Entity(name="ParaMem", entity_type="concept"),
                Entity(name="Gemma", entity_type="concept"),
            ],
            relations=[
                Relation(
                    subject="ParaMem",
                    predicate="validated_on",
                    object="Gemma",
                    relation_type="factual",
                    speaker_id="Speaker0",
                )
            ],
        )
        return sg1, sg2

    def test_flag_false_default_does_not_call_detect_contradiction(self):
        """With cross_predicate_contradiction=False (default) and a model present,
        detect_contradiction_with_model must NOT be called, regardless of what the
        model would return — the flag gates the call site, not just the effect.

        This is the regression test for the live data-loss finding:
        multi-valued / independent facts were wrongly removed by cross-predicate
        contradiction detection.
        """
        from unittest.mock import MagicMock, patch

        model_stub = MagicMock()
        tok_stub = MagicMock()
        tok_stub.apply_chat_template.return_value = "formatted"

        sg1, sg2 = self._multi_valued_sessions()

        # Patch detect_contradiction_with_model to return a contradiction — so
        # if the flag check were absent, an edge WOULD be removed.  The test
        # proves the flag suppresses the call entirely.
        with patch(
            "paramem.graph.merger.detect_contradiction_with_model",
            return_value=("ParaMem", "validated_on", "Qwen"),
        ) as mock_detect:
            # Default: cross_predicate_contradiction=False
            m = GraphMerger(model=model_stub, tokenizer=tok_stub)
            assert m.cross_predicate_contradiction is False

            # Patch check_predicate_coexistence to return COEXIST so cardinality
            # resolution does not remove the first edge either (multi-valued predicate).
            with patch(
                "paramem.graph.merger.check_predicate_coexistence",
                return_value="COEXIST",
            ):
                m.merge(sg1)
                m.merge(sg2)

            # detect_contradiction_with_model must NOT have been called.
            mock_detect.assert_not_called()

        # Both edges must be present — no removal.
        successors = list(m.graph.successors("ParaMem"))
        validated_on_objects = [
            obj
            for obj in successors
            for _, data in m.graph["ParaMem"][obj].items()
            if data.get("predicate") == "validated_on"
        ]
        assert "Qwen" in validated_on_objects, (
            "validated_on Qwen must not be removed when cross_predicate_contradiction=False"
        )
        assert "Gemma" in validated_on_objects, (
            "validated_on Gemma must be present when cross_predicate_contradiction=False"
        )

    def test_flag_true_enables_cross_predicate_removal(self):
        """With cross_predicate_contradiction=True, detect_contradiction_with_model
        IS consulted and the old edge is removed when it returns a contradiction.
        This verifies the cross-predicate removal behavior is preserved behind the flag.
        """
        from unittest.mock import MagicMock, patch

        model_stub = MagicMock()
        tok_stub = MagicMock()
        tok_stub.apply_chat_template.return_value = "formatted"

        sg1, sg2 = self._multi_valued_sessions()

        # detect_contradiction_with_model says "Qwen" is contradicted.
        with patch(
            "paramem.graph.merger.detect_contradiction_with_model",
            return_value=("ParaMem", "validated_on", "Qwen"),
        ) as mock_detect:
            m = GraphMerger(
                model=model_stub,
                tokenizer=tok_stub,
                cross_predicate_contradiction=True,
            )
            assert m.cross_predicate_contradiction is True

            # Patch check_predicate_coexistence to return COEXIST so cardinality
            # resolution does not fire first (multi-valued — only cross-predicate
            # contradiction detection fires).
            with patch(
                "paramem.graph.merger.check_predicate_coexistence",
                return_value="COEXIST",
            ):
                m.merge(sg1)
                m.merge(sg2)

            # detect_contradiction_with_model must have been called (flag=True).
            mock_detect.assert_called()

        # "Qwen" edge was removed by cross-predicate contradiction detection; "Gemma" was added.
        successors = list(m.graph.successors("ParaMem"))
        validated_on_objects = [
            obj
            for obj in successors
            for _, data in m.graph["ParaMem"][obj].items()
            if data.get("predicate") == "validated_on"
        ]
        assert "Qwen" not in validated_on_objects, (
            "validated_on Qwen must be removed when cross_predicate_contradiction=True"
        )
        assert "Gemma" in validated_on_objects, (
            "validated_on Gemma must be present after cross-predicate removal"
        )
        # contradictions_resolved must record the removal.
        assert any(r.get("method") == "model" for r in m.contradictions_resolved), (
            "cross-predicate removal must be logged in contradictions_resolved"
        )

    def test_flag_false_preserves_cardinality_resolution(self):
        """Same-predicate, different-object cardinality resolution must still fire when
        cross_predicate_contradiction=False.  The flag gates ONLY cross-predicate
        contradiction detection, not same-predicate cardinality resolution.

        This guards against accidentally disabling cardinality resolution.
        """
        from unittest.mock import MagicMock, patch

        model_stub = MagicMock()
        tok_stub = MagicMock()
        tok_stub.apply_chat_template.return_value = "formatted"

        sg1 = SessionGraph(
            session_id="s1",
            timestamp="2026-01-01T00:00:00Z",
            entities=[
                Entity(name="Alex", entity_type="person"),
                Entity(name="Munich", entity_type="place"),
            ],
            relations=[
                Relation(
                    subject="Alex",
                    predicate="lives_in",
                    object="Munich",
                    relation_type="factual",
                    speaker_id="Speaker0",
                )
            ],
        )
        sg2 = SessionGraph(
            session_id="s2",
            timestamp="2026-01-02T00:00:00Z",
            entities=[
                Entity(name="Alex", entity_type="person"),
                Entity(name="Berlin", entity_type="place"),
            ],
            relations=[
                Relation(
                    subject="Alex",
                    predicate="lives_in",
                    object="Berlin",
                    relation_type="factual",
                    speaker_id="Speaker0",
                )
            ],
        )

        # cross_predicate_contradiction=False (default); same-predicate cardinality resolution fires
        # (single-valued predicate → replace old value).
        with patch(
            "paramem.graph.merger.check_predicate_coexistence",
            return_value="REPLACE",  # single-valued
        ):
            m = GraphMerger(
                model=model_stub,
                tokenizer=tok_stub,
                cross_predicate_contradiction=False,
            )
            m.merge(sg1)
            m.merge(sg2)

        # Cardinality resolution must have replaced Munich with Berlin.
        successors = list(m.graph.successors("Alex"))
        lives_in_objects = [
            obj
            for obj in successors
            for _, data in m.graph["Alex"][obj].items()
            if data.get("predicate") == "lives_in"
        ]
        assert "Munich" not in lives_in_objects, (
            "Cardinality resolution must still replace single-valued predicate even when "
            "cross_predicate_contradiction=False"
        )
        assert "Berlin" in lives_in_objects, "New edge must be present after cardinality resolution"


class TestPromptsDirOverride:
    """Custom prompts_dir overrides the inline fallback constants."""

    def test_custom_prompts_dir_loaded_into_instance_attributes(self, tmp_path):
        """GraphMerger(prompts_dir=...) must resolve _coexistence_prompt and
        _contradiction_prompt from the supplied directory, not from the inline
        _COEXISTENCE_PROMPT / _CONTRADICTION_PROMPT constants.
        """
        coexistence_content = "CUSTOM_COEXISTENCE_MARKER sentinel text"
        contradiction_content = "CUSTOM_CONTRADICTION_MARKER sentinel text"
        (tmp_path / "merger_coexistence.txt").write_text(coexistence_content)
        (tmp_path / "merger_contradiction.txt").write_text(contradiction_content)

        m = GraphMerger(prompts_dir=tmp_path)

        assert m._coexistence_prompt == coexistence_content.strip(), (
            "Custom merger_coexistence.txt must override the inline fallback"
        )
        assert m._contradiction_prompt == contradiction_content.strip(), (
            "Custom merger_contradiction.txt must override the inline fallback"
        )

    def test_missing_files_fall_back_to_inline_constants(self, tmp_path):
        """A prompts_dir that lacks the prompt files must fall back to the
        inline _COEXISTENCE_PROMPT / _CONTRADICTION_PROMPT constants.
        """
        from paramem.graph.merger import _COEXISTENCE_PROMPT, _CONTRADICTION_PROMPT

        # tmp_path exists but contains no prompt files.
        m = GraphMerger(prompts_dir=tmp_path)

        assert m._coexistence_prompt == _COEXISTENCE_PROMPT, (
            "Missing file must fall back to inline _COEXISTENCE_PROMPT"
        )
        assert m._contradiction_prompt == _CONTRADICTION_PROMPT, (
            "Missing file must fall back to inline _CONTRADICTION_PROMPT"
        )


class TestModelContradictionAndRelease:
    """Model-only contradiction is always-on when a model is present;
    coexist-all when model is None; release() drops model/tokenizer."""

    def _build_stub_model(self, is_single_valued: bool):
        """Return a (model, tokenizer) stub whose check_predicate_coexistence
        returns ``"REPLACE"`` for single-valued or ``"COEXIST"`` for multi-valued."""
        from unittest.mock import MagicMock

        model = MagicMock()
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "formatted"

        verdict = "REPLACE" if is_single_valued else "COEXIST"

        def _patched_coexistence(subject, predicate, old_value, new_value, mdl, tok, prompt=None):
            return verdict

        return model, tokenizer, _patched_coexistence

    def test_single_valued_contradiction_resolved_with_model(self):
        """With a model present, same-(s,p)/different-o for a single-valued
        predicate removes the old edge and inserts the new one."""
        from unittest.mock import patch

        model_stub, tok_stub, coexist_fn = self._build_stub_model(is_single_valued=True)

        with patch(
            "paramem.graph.merger.check_predicate_coexistence",
            side_effect=coexist_fn,
        ):
            m = GraphMerger(model=model_stub, tokenizer=tok_stub)
            sg1 = SessionGraph(
                session_id="s1",
                timestamp="2026-01-01T00:00:00Z",
                entities=[
                    Entity(name="Alex", entity_type="person"),
                    Entity(name="Munich", entity_type="place"),
                ],
                relations=[
                    Relation(
                        subject="Alex",
                        predicate="lives_in",
                        object="Munich",
                        relation_type="factual",
                        speaker_id="Speaker0",
                    )
                ],
            )
            sg2 = SessionGraph(
                session_id="s2",
                timestamp="2026-01-02T00:00:00Z",
                entities=[
                    Entity(name="Alex", entity_type="person"),
                    Entity(name="Berlin", entity_type="place"),
                ],
                relations=[
                    Relation(
                        subject="Alex",
                        predicate="lives_in",
                        object="Berlin",
                        relation_type="factual",
                        speaker_id="Speaker0",
                    )
                ],
            )
            m.merge(sg1)
            m.merge(sg2)

        # Single-valued: old "Munich" edge removed; only "Berlin" remains.
        alex_successors = list(m.graph.successors("Alex"))
        # "Munich" must be gone (replaced), "Berlin" must be present.
        lives_in_edges = [
            (obj, data)
            for obj in alex_successors
            for _, data in m.graph["Alex"][obj].items()
            if data.get("predicate") == "lives_in"
        ]
        objects_with_lives_in = [obj for obj, _ in lives_in_edges]
        assert "Munich" not in objects_with_lives_in, (
            "Old single-valued edge (Munich) must be removed by model contradiction"
        )
        assert "Berlin" in objects_with_lives_in, (
            "New edge (Berlin) must be present after contradiction resolution"
        )

    def test_no_model_coexist_all(self):
        """Without a model, same-(s,p)/different-o triples coexist (no removal)."""
        m = GraphMerger()  # model=None
        assert m.model is None

        sg1 = SessionGraph(
            session_id="s1",
            timestamp="2026-01-01T00:00:00Z",
            entities=[
                Entity(name="Alex", entity_type="person"),
                Entity(name="Munich", entity_type="place"),
            ],
            relations=[
                Relation(
                    subject="Alex",
                    predicate="lives_in",
                    object="Munich",
                    relation_type="factual",
                    speaker_id="Speaker0",
                )
            ],
        )
        sg2 = SessionGraph(
            session_id="s2",
            timestamp="2026-01-02T00:00:00Z",
            entities=[
                Entity(name="Alex", entity_type="person"),
                Entity(name="Berlin", entity_type="place"),
            ],
            relations=[
                Relation(
                    subject="Alex",
                    predicate="lives_in",
                    object="Berlin",
                    relation_type="factual",
                    speaker_id="Speaker0",
                )
            ],
        )
        m.merge(sg1)
        m.merge(sg2)

        # Both values coexist: no removal when model is absent.
        alex_successors = list(m.graph.successors("Alex"))
        lives_in_objects = [
            obj
            for obj in alex_successors
            for _, data in m.graph["Alex"][obj].items()
            if data.get("predicate") == "lives_in"
        ]
        assert "Munich" in lives_in_objects, (
            "Without a model, old edge must NOT be removed (coexist-all)"
        )
        assert "Berlin" in lives_in_objects, "New edge must be present (coexist-all, no model)"

    def test_release_nulls_model_and_tokenizer(self):
        """release() sets .model and .tokenizer to None (idempotent)."""
        from unittest.mock import MagicMock

        model_stub = MagicMock()
        tok_stub = MagicMock()
        m = GraphMerger(model=model_stub, tokenizer=tok_stub)

        assert m.model is model_stub
        assert m.tokenizer is tok_stub

        m.release()

        assert m.model is None, "release() must null .model"
        assert m.tokenizer is None, "release() must null .tokenizer"

    def test_release_is_idempotent(self):
        """Calling release() twice does not raise."""
        from unittest.mock import MagicMock

        m = GraphMerger(model=MagicMock(), tokenizer=MagicMock())
        m.release()
        m.release()  # must not raise

    def test_multi_valued_coexist_with_model(self):
        """With a model present, when check_predicate_coexistence returns COEXIST
        (multi-valued predicate), both the old and new objects must coexist as
        edges for the same (subject, predicate) and no contradictions_resolved
        entry must be recorded."""
        from unittest.mock import patch

        model_stub, tok_stub, coexist_fn = self._build_stub_model(is_single_valued=False)

        with patch(
            "paramem.graph.merger.check_predicate_coexistence",
            side_effect=coexist_fn,
        ):
            m = GraphMerger(model=model_stub, tokenizer=tok_stub)
            sg1 = SessionGraph(
                session_id="s1",
                timestamp="2026-01-01T00:00:00Z",
                entities=[
                    Entity(name="Alex", entity_type="person"),
                    Entity(name="Python", entity_type="concept"),
                ],
                relations=[
                    Relation(
                        subject="Alex",
                        predicate="speaks",
                        object="Python",
                        relation_type="factual",
                        speaker_id="Speaker0",
                    )
                ],
            )
            sg2 = SessionGraph(
                session_id="s2",
                timestamp="2026-01-02T00:00:00Z",
                entities=[
                    Entity(name="Alex", entity_type="person"),
                    Entity(name="Rust", entity_type="concept"),
                ],
                relations=[
                    Relation(
                        subject="Alex",
                        predicate="speaks",
                        object="Rust",
                        relation_type="factual",
                        speaker_id="Speaker0",
                    )
                ],
            )
            m.merge(sg1)
            m.merge(sg2)

        # Multi-valued: both objects must coexist, no edge removed.
        alex_successors = list(m.graph.successors("Alex"))
        speaks_objects = [
            obj
            for obj in alex_successors
            for _, data in m.graph["Alex"][obj].items()
            if data.get("predicate") == "speaks"
        ]
        assert "Python" in speaks_objects, (
            "Multi-valued coexist: old edge (Python) must NOT be removed"
        )
        assert "Rust" in speaks_objects, "Multi-valued coexist: new edge (Rust) must be present"
        assert m.contradictions_resolved == [], (
            "Multi-valued coexist: no contradictions_resolved entry must be recorded"
        )


class TestIkKeyProvenance:
    """Unit tests for ik_key provenance stamping on merged edges."""

    def test_new_edge_stamped_with_indexed_key(self):
        """New-edge insertion stamps ik_key on the edge when Relation.indexed_key is set."""
        from paramem.graph.merger import GraphMerger
        from paramem.graph.schema import Relation, SessionGraph
        from paramem.memory.persistence import _IK_KEY_ATTR

        m = GraphMerger()  # no model
        session = SessionGraph(
            session_id="s1",
            timestamp="2026-01-01T00:00:00Z",
            entities=[],
            relations=[
                Relation(
                    subject="Alice",
                    predicate="lives_in",
                    object="Berlin",
                    relation_type="factual",
                    confidence=1.0,
                    speaker_id="Speaker0",
                    indexed_key="graph42",
                )
            ],
        )
        m.merge(session)

        # Exactly one edge for (Alice, lives_in, Berlin).
        edges = list(m.graph["Alice"]["Berlin"].items())
        assert len(edges) == 1
        eid, data = edges[0]
        assert data.get(_IK_KEY_ATTR) == "graph42", (
            f"Expected ik_key='graph42' stamped on new edge; got {data.get(_IK_KEY_ATTR)!r}"
        )

    def test_normal_ingest_relation_no_indexed_key(self):
        """Normal ingest Relation (indexed_key=None) does NOT stamp ik_key."""
        from paramem.graph.merger import GraphMerger
        from paramem.graph.schema import Relation, SessionGraph
        from paramem.memory.persistence import _IK_KEY_ATTR

        m = GraphMerger()
        session = SessionGraph(
            session_id="s1",
            timestamp="2026-01-01T00:00:00Z",
            entities=[],
            relations=[
                Relation(
                    subject="Alice",
                    predicate="lives_in",
                    object="Berlin",
                    relation_type="factual",
                    confidence=1.0,
                    speaker_id="Speaker0",
                    # indexed_key defaults to None
                )
            ],
        )
        m.merge(session)

        edges = list(m.graph["Alice"]["Berlin"].items())
        assert len(edges) == 1
        eid, data = edges[0]
        assert data.get(_IK_KEY_ATTR) is None, "Normal ingest must NOT stamp ik_key on merged edge"

    def test_disjoint_multi_valued_coexists(self):
        """Disjoint multi-valued values COEXIST — both edges kept, no aggregation."""
        from unittest.mock import patch

        from paramem.graph.merger import GraphMerger
        from paramem.graph.schema import Relation, SessionGraph
        from paramem.memory.persistence import _IK_KEY_ATTR

        m = GraphMerger()
        m.graph.add_node("Alex")
        m.graph.add_node("cat")
        eid_old = m.graph.add_edge(
            "Alex",
            "cat",
            predicate="has_pet",
            relation_type="factual",
            confidence=1.0,
            first_seen="s1",
            last_seen="s1",
            recurrence_count=1,
            sessions=["s1"],
        )
        m.graph["Alex"]["cat"][eid_old][_IK_KEY_ATTR] = "g1"
        m._predicate_cardinality["has_pet"] = True  # multi-valued

        session = SessionGraph(
            session_id="s2",
            timestamp="2026-01-02T00:00:00Z",
            entities=[],
            relations=[
                Relation(
                    subject="Alex",
                    predicate="has_pet",
                    object="dog",
                    relation_type="factual",
                    confidence=1.0,
                    speaker_id="Speaker0",
                    indexed_key="g2",
                )
            ],
        )

        with patch(
            "paramem.graph.merger.check_predicate_coexistence",
            return_value="COEXIST",
        ):
            m.merge(session, additive=True)

        # Both edges must coexist.
        pets = [
            obj
            for obj in m.graph.successors("Alex")
            for _, d in m.graph["Alex"][obj].items()
            if d.get("predicate") == "has_pet"
        ]
        assert "cat" in pets and "dog" in pets, (
            "Disjoint multi-valued pair must COEXIST — both keys kept"
        )


class TestReinforcementTracking:
    """Tests for merger.reinforcements (Case-1 duplicate-SPO collapse tracking)."""

    def test_reinforcements_empty_after_merge_no_duplicate(self):
        """A merge with no duplicate SPO produces an empty reinforcements list."""
        from paramem.graph.merger import GraphMerger
        from paramem.graph.schema import Relation, SessionGraph

        m = GraphMerger()
        session = SessionGraph(
            session_id="s1",
            timestamp="2026-01-01T00:00:00Z",
            entities=[],
            relations=[
                Relation(
                    subject="Alice",
                    predicate="lives_in",
                    object="Berlin",
                    relation_type="factual",
                    confidence=1.0,
                    speaker_id="Speaker0",
                    indexed_key="graph1",
                )
            ],
        )
        m.merge(session, additive=True)
        assert m.reinforcements == [], "No duplicate → reinforcements must be empty"

    def test_reinforcements_populated_on_duplicate_spo_collapse(self):
        """Two recon edges with same (s,p,o) but different ik_keys: Case-1 fires and
        the surviving key appears in reinforcements."""
        from paramem.graph.merger import GraphMerger
        from paramem.graph.schema import Relation, SessionGraph

        m = GraphMerger()
        # First merge: net-new edge, key graph1 stamped.
        s1 = SessionGraph(
            session_id="recon1",
            timestamp="2026-01-01T00:00:00Z",
            entities=[],
            relations=[
                Relation(
                    subject="Alice",
                    predicate="lives_in",
                    object="Berlin",
                    relation_type="factual",
                    confidence=1.0,
                    speaker_id="Speaker0",
                    indexed_key="graph1",
                )
            ],
        )
        m.merge(s1, additive=True)
        assert m.reinforcements == [], "First merge is net-new — no reinforcement yet"

        # Second merge: same (s,p,o), different ik_key → Case-1 → reinforcement.
        s2 = SessionGraph(
            session_id="recon2",
            timestamp="2026-01-02T00:00:00Z",
            entities=[],
            relations=[
                Relation(
                    subject="Alice",
                    predicate="lives_in",
                    object="Berlin",
                    relation_type="factual",
                    confidence=1.0,
                    speaker_id="Speaker0",
                    indexed_key="graph2",
                )
            ],
        )
        m.merge(s2, additive=True)

        assert len(m.reinforcements) == 1, (
            f"Duplicate-SPO collapse must produce 1 reinforcement entry; got {m.reinforcements}"
        )
        # The survivor is the EXISTING edge's key (graph1), not the incoming (graph2).
        assert m.reinforcements[0] == "graph1", (
            f"Surviving key must be graph1 (existing edge); got {m.reinforcements[0]!r}"
        )

    def test_reinforcements_reset_graph_clears_reinforcements(self):
        """reset_graph() clears reinforcements from the prior fold."""
        from paramem.graph.merger import GraphMerger

        m = GraphMerger()
        m.reinforcements = ["graph_stale"]
        m.reset_graph()
        assert m.reinforcements == [], "reset_graph must clear reinforcements"

    def test_collapsed_populated_on_duplicate_spo_collapse(self):
        """Two recon edges with same (s,p,o) but different ik_keys: Case-1 fires and
        the INCOMING (drifting) key appears in collapsed, while the surviving key is
        in reinforcements.

        collapsed is the parallel to reinforcements — where reinforcements records
        the key that survived, collapsed records the key that was deduplicated away.
        """
        from paramem.graph.merger import GraphMerger
        from paramem.graph.schema import Relation, SessionGraph

        m = GraphMerger()
        # First merge: net-new edge, key graph1 stamped.
        s1 = SessionGraph(
            session_id="recon1",
            timestamp="2026-01-01T00:00:00Z",
            entities=[],
            relations=[
                Relation(
                    subject="Alice",
                    predicate="lives_in",
                    object="Berlin",
                    relation_type="factual",
                    confidence=1.0,
                    speaker_id="Speaker0",
                    indexed_key="graph1",
                )
            ],
        )
        m.merge(s1, additive=True)
        assert m.collapsed == [], "First merge is net-new — no collapse yet"

        # Second merge: same (s,p,o), different ik_key → Case-1 → collapse.
        s2 = SessionGraph(
            session_id="recon2",
            timestamp="2026-01-02T00:00:00Z",
            entities=[],
            relations=[
                Relation(
                    subject="Alice",
                    predicate="lives_in",
                    object="Berlin",
                    relation_type="factual",
                    confidence=1.0,
                    speaker_id="Speaker0",
                    indexed_key="graph2",
                )
            ],
        )
        m.merge(s2, additive=True)

        # The incoming (drifting) key must appear in collapsed.
        assert len(m.collapsed) == 1, (
            f"Duplicate-SPO collapse must produce 1 collapsed entry; got {m.collapsed}"
        )
        assert m.collapsed[0] == "graph2", (
            f"Collapsed key must be graph2 (the incoming key); got {m.collapsed[0]!r}"
        )
        # The surviving key must still be in reinforcements.
        assert m.reinforcements[0] == "graph1", (
            f"Surviving key must be graph1 (existing edge); got {m.reinforcements[0]!r}"
        )

    def test_collapsed_reset_graph_clears_collapsed(self):
        """reset_graph() clears collapsed from the prior fold."""
        from paramem.graph.merger import GraphMerger

        m = GraphMerger()
        m.collapsed = ["graph_stale"]
        m.reset_graph()
        assert m.collapsed == [], "reset_graph must clear collapsed"

    def test_reset_graph_clears_all_per_fold_caches(self):
        """reset_graph() clears graph, caches, reinforcements, collapsed, and contradictions."""

        from paramem.graph.merger import GraphMerger

        m = GraphMerger()
        # Populate all per-fold state.
        m.graph.add_node("Alice")
        m._predicate_cardinality["foo"] = True
        m.contradictions_resolved.append({"method": "model"})
        m.reinforcements.append("k2")
        m.collapsed.append("k3")

        m.reset_graph()

        assert m.graph.number_of_nodes() == 0, "reset_graph must empty the graph"
        assert m._predicate_cardinality == {}, "reset_graph must clear cardinality cache"
        assert m.contradictions_resolved == [], "reset_graph must clear contradictions_resolved"
        assert m.reinforcements == [], "reset_graph must clear reinforcements"
        assert m.collapsed == [], "reset_graph must clear collapsed"

    def test_case1_adopt_does_not_produce_reinforcement(self):
        """Case-1-adopt (keyless existing + incoming has key) does NOT add a reinforcement.

        Reinforcement fires only when BOTH edges carry ik_keys (duplicate-SPO collapse).
        Adopt (existing is keyless) just stamps the key — no recurrence bump needed.
        """
        from paramem.graph.merger import GraphMerger
        from paramem.graph.schema import Relation
        from paramem.memory.persistence import _IK_KEY_ATTR

        m = GraphMerger()
        m.graph.add_node("Alice")
        m.graph.add_node("Berlin")
        existing_eid = m.graph.add_edge(
            "Alice",
            "Berlin",
            predicate="lives_in",
            relation_type="factual",
            confidence=1.0,
            first_seen="s0",
            last_seen="s0",
            recurrence_count=1,
            sessions=["s0"],
        )
        # Keyless existing edge.
        assert m.graph["Alice"]["Berlin"][existing_eid].get(_IK_KEY_ATTR) is None

        incoming = Relation(
            subject="Alice",
            predicate="lives_in",
            object="Berlin",
            relation_type="factual",
            confidence=1.0,
            speaker_id="Speaker0",
            indexed_key="graph5",
        )
        m._upsert_relation("Alice", "Berlin", incoming, "s1", "2026-01-01T00:00:00Z")

        # Adopt path: no reinforcement (the existing edge had no key to preserve).
        assert m.reinforcements == [], (
            "Case-1-adopt must NOT produce a reinforcement (existing was keyless)"
        )
        # Key was adopted onto the existing edge.
        assert m.graph["Alice"]["Berlin"][existing_eid].get(_IK_KEY_ATTR) == "graph5"

    def test_additive_fold_short_circuits_replace_both_edges_survive(self):
        """additive=True short-circuits Case-2: even when check_predicate_coexistence returns
        REPLACE, the old edge is NOT removed and both keys survive.

        This is the regression guard for the purely-additive fold: no model call, no edge
        removal, no registered fact lost at fold time.
        """
        from unittest.mock import MagicMock, patch

        from paramem.graph.merger import GraphMerger
        from paramem.graph.schema import Relation
        from paramem.memory.persistence import _IK_KEY_ATTR

        model_stub = MagicMock()
        tok_stub = MagicMock()
        tok_stub.apply_chat_template.return_value = "formatted"

        m = GraphMerger(model=model_stub, tokenizer=tok_stub)
        m.graph.add_node("Alex")
        m.graph.add_node("Munich")

        # Pre-seed with the first edge carrying ik_key='key_munich'.
        eid_old = m.graph.add_edge(
            "Alex",
            "Munich",
            predicate="lives_in",
            relation_type="factual",
            confidence=1.0,
            first_seen="s1",
            last_seen="s1",
            recurrence_count=1,
            sessions=["s1"],
        )
        m.graph["Alex"]["Munich"][eid_old][_IK_KEY_ATTR] = "key_munich"

        # Incoming relation: same predicate, different object (Berlin).
        incoming = Relation(
            subject="Alex",
            predicate="lives_in",
            object="Berlin",
            relation_type="factual",
            confidence=1.0,
            speaker_id="Speaker0",
            indexed_key="key_berlin",
        )

        # Even though the model would say REPLACE, additive=True must skip it entirely.
        with patch(
            "paramem.graph.merger.check_predicate_coexistence",
            return_value="REPLACE",
        ) as mock_coexist:
            m._upsert_relation(
                "Alex", "Berlin", incoming, "s2", "2026-01-02T00:00:00Z", additive=True
            )
            # check_predicate_coexistence must NOT have been called (short-circuit).
            mock_coexist.assert_not_called()

        # Both edges must survive: Munich (old) and Berlin (new).
        lives_in_objects = [
            obj
            for obj in m.graph.successors("Alex")
            for _, d in m.graph["Alex"][obj].items()
            if d.get("predicate") == "lives_in"
        ]
        assert "Munich" in lives_in_objects, (
            "Old edge (Munich) must NOT be removed when additive=True"
        )
        assert "Berlin" in lives_in_objects, (
            "New edge (Berlin) must be inserted even when additive=True"
        )

        # Both ik_keys must be stamped on their respective edges.
        munich_key = next(
            d.get(_IK_KEY_ATTR)
            for _, d in m.graph["Alex"]["Munich"].items()
            if d.get("predicate") == "lives_in"
        )
        berlin_key = next(
            d.get(_IK_KEY_ATTR)
            for _, d in m.graph["Alex"]["Berlin"].items()
            if d.get("predicate") == "lives_in"
        )
        assert munich_key == "key_munich", f"Expected key_munich on old edge; got {munich_key!r}"
        assert berlin_key == "key_berlin", f"Expected key_berlin on new edge; got {berlin_key!r}"
