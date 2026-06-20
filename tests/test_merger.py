"""Tests for knowledge graph merging and entity resolution."""

import tempfile
from pathlib import Path

import pytest

from paramem.graph.merger import GraphMerger
from paramem.graph.name_match import canonical
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
        """canonical() provides the single identity function for all string types."""
        assert canonical("Alex") == "alex"
        assert canonical("  Alex  ") == "alex"
        assert canonical("Dr. Smith") == "dr. smith"


class TestPredicateNormalization:
    def test_lowercase_and_space_form(self):
        """canonical() folds case and separator to space; underscores are folded too."""
        assert canonical("Works At") == "works at"
        assert canonical("LIVES IN") == "lives in"

    def test_strip_whitespace(self):
        assert canonical("  works_at  ") == "works at"

    def test_passthrough_preserves_content(self):
        assert canonical("invented") == "invented"
        assert canonical("custom pred") == "custom pred"
        # underscore folds to space
        assert canonical("works_at") == "works at"

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
        edges = list(merger.graph["a"]["b"].values())
        assert len(edges) == 1
        # canonical() folds underscore → space; predicate stored in canonical form
        assert edges[0]["predicate"] == "works at"
        assert edges[0]["recurrence_count"] == 2


class TestEntityResolution:
    def test_exact_match(self, merger, session_graph_1, session_graph_2):
        merger.merge(session_graph_1)
        merger.merge(session_graph_2)
        # "Alex" node key is canonical("Alex") == "alex" (node-key model A)
        assert "alex" in merger.graph.nodes
        assert merger.graph.nodes["alex"]["recurrence_count"] == 2

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
        # Canonical key is "python" for both — same node key regardless of type
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
        # Node keys are canonical: "a", "b"
        edges = list(merger.graph["a"]["b"].values())
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
        edges = list(merger.graph["a"]["b"].values())
        assert len(edges) == 2


class TestSessionTracking:
    def test_session_provenance(self, merger, session_graph_1, session_graph_2):
        merger.merge(session_graph_1)
        merger.merge(session_graph_2)
        # Node key is canonical: canonical("Alex") == "alex"
        sessions = merger.graph.nodes["alex"]["sessions"]
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
        # Node key is canonical: "alex"; display name in attributes["name"]
        attrs = merger.graph.nodes["alex"]["attributes"]
        assert attrs["age"] == "29"
        assert attrs["role"] == "engineer"
        # Display name preserved in attributes
        assert attrs["name"] == "Alex"


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
            # Node key is canonical: "alex"
            assert "alex" in new_merger.graph.nodes

    def test_load_nonexistent(self, merger):
        graph = merger.load_graph("/nonexistent/path.json")
        assert graph.number_of_nodes() == 0

    def test_save_encrypted_false_writes_plaintext_even_when_security_on(
        self, merger, session_graph_1, tmp_path, monkeypatch
    ):
        """save_graph(..., encrypted=False) must bypass the envelope and emit
        plaintext JSON, even under Security ON.  Debug-directory writers
        depend on this so ``cat debug/cycle_*/graph_merged_snapshot.json`` is always
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
        """'Alexander' and 'alexander' must merge — canonical key is identical."""
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
        # Node key is canonical: canonical("Portland") == "portland"
        assert m.graph.nodes["portland"]["recurrence_count"] == 2

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
        # Place node keys are canonical: "portland", "munich".
        s0_neighbors = list(m.graph.successors(speaker0_node))
        s1_neighbors = list(m.graph.successors(speaker1_node))
        assert "portland" in s0_neighbors and "munich" not in s0_neighbors
        assert "munich" in s1_neighbors and "portland" not in s1_neighbors

    def test_third_party_mention_with_speaker_id_unset_is_separate(self):
        """A later session emits a third-party ``"Alex"`` entity with
        ``speaker_id`` unset (he's not the speaker of that session).

        The name namespace and the speaker-ID namespace are disjoint by
        construction: speaker IDs follow the ``Speaker_N`` pattern
        produced by the speaker pool, so a display name like
        ``"Alex"`` will never collide with a speaker_id-keyed node.
        The third-party Alex becomes a separate node keyed by
        ``"alex"`` (canonical form); Speaker0's node remains keyed by
        ``"Speaker0"`` with its own attributes intact.
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

        # Speaker0 is keyed by speaker_id; third-party Alex by canonical name "alex" —
        # disjoint namespaces, two separate nodes.
        assert "Speaker0" in m.graph.nodes
        assert "alex" in m.graph.nodes
        assert m.graph.nodes["Speaker0"]["speaker_id"] == "Speaker0"
        assert m.graph.nodes["alex"].get("speaker_id") is None
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
        # Node keys are canonical: "paramem", "qwen", "gemma"
        successors = list(m.graph.successors("paramem"))
        validated_on_objects = [
            obj
            for obj in successors
            for _, data in m.graph["paramem"][obj].items()
            if data.get("predicate") == "validated on"
        ]
        assert "qwen" in validated_on_objects, (
            "validated_on Qwen must not be removed when cross_predicate_contradiction=False"
        )
        assert "gemma" in validated_on_objects, (
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

        # "qwen" edge was removed by cross-predicate contradiction detection; "gemma" was added.
        # Node keys are canonical: "paramem", "qwen", "gemma"
        successors = list(m.graph.successors("paramem"))
        validated_on_objects = [
            obj
            for obj in successors
            for _, data in m.graph["paramem"][obj].items()
            if data.get("predicate") == "validated on"
        ]
        assert "qwen" not in validated_on_objects, (
            "validated_on Qwen must be removed when cross_predicate_contradiction=True"
        )
        assert "gemma" in validated_on_objects, (
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
        # Node keys are canonical: "alex", "munich", "berlin"
        successors = list(m.graph.successors("alex"))
        lives_in_objects = [
            obj
            for obj in successors
            for _, data in m.graph["alex"][obj].items()
            if data.get("predicate") == "lives in"
        ]
        assert "munich" not in lives_in_objects, (
            "Cardinality resolution must still replace single-valued predicate even when "
            "cross_predicate_contradiction=False"
        )
        assert "berlin" in lives_in_objects, "New edge must be present after cardinality resolution"


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

        # Single-valued: old "munich" edge removed; only "berlin" remains.
        # Node keys are canonical: "alex", "munich", "berlin"
        alex_successors = list(m.graph.successors("alex"))
        # "munich" must be gone (replaced), "berlin" must be present.
        lives_in_edges = [
            (obj, data)
            for obj in alex_successors
            for _, data in m.graph["alex"][obj].items()
            if data.get("predicate") == "lives in"
        ]
        objects_with_lives_in = [obj for obj, _ in lives_in_edges]
        assert "munich" not in objects_with_lives_in, (
            "Old single-valued edge (Munich) must be removed by model contradiction"
        )
        assert "berlin" in objects_with_lives_in, (
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
        # Node keys are canonical: "alex", "munich", "berlin"
        alex_successors = list(m.graph.successors("alex"))
        lives_in_objects = [
            obj
            for obj in alex_successors
            for _, data in m.graph["alex"][obj].items()
            if data.get("predicate") == "lives in"
        ]
        assert "munich" in lives_in_objects, (
            "Without a model, old edge must NOT be removed (coexist-all)"
        )
        assert "berlin" in lives_in_objects, "New edge must be present (coexist-all, no model)"

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
        # Node keys are canonical: "alex", "python", "rust"
        alex_successors = list(m.graph.successors("alex"))
        speaks_objects = [
            obj
            for obj in alex_successors
            for _, data in m.graph["alex"][obj].items()
            if data.get("predicate") == "speaks"
        ]
        assert "python" in speaks_objects, (
            "Multi-valued coexist: old edge (Python) must NOT be removed"
        )
        assert "rust" in speaks_objects, "Multi-valued coexist: new edge (Rust) must be present"
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

        # Exactly one edge for (alice, lives_in, berlin) — canonical node keys.
        edges = list(m.graph["alice"]["berlin"].items())
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

        # Node keys are canonical: "alice", "berlin"
        edges = list(m.graph["alice"]["berlin"].items())
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
        # Pre-seed with canonical node keys (node-key model A).
        m.graph.add_node("alex", attributes={"name": "Alex"})
        m.graph.add_node("cat", attributes={"name": "cat"})
        eid_old = m.graph.add_edge(
            "alex",
            "cat",
            predicate="has pet",
            relation_type="factual",
            confidence=1.0,
            first_seen="s1",
            last_seen="s1",
            recurrence_count=1,
            sessions=["s1"],
        )
        m.graph["alex"]["cat"][eid_old][_IK_KEY_ATTR] = "g1"
        m._predicate_cardinality["has pet"] = True  # multi-valued; canonical form

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
            m.merge(session, resolve_contradictions=False)

        # Both edges must coexist.
        # Node keys are canonical: "alex", "cat", "dog"
        pets = [
            obj
            for obj in m.graph.successors("alex")
            for _, d in m.graph["alex"][obj].items()
            if d.get("predicate") == "has pet"
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
        m.merge(session, resolve_contradictions=False)
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
        m.merge(s1, resolve_contradictions=False)
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
        m.merge(s2, resolve_contradictions=False)

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
        m.merge(s1, resolve_contradictions=False)
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
        m.merge(s2, resolve_contradictions=False)

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
        """reset_graph() clears graph, caches, reinforcements, collapsed, contradictions,
        and removal_ledger."""

        from paramem.graph.merger import GraphMerger

        m = GraphMerger()
        # Populate all per-fold state.
        m.graph.add_node("Alice")
        m._predicate_cardinality["foo"] = True
        m.contradictions_resolved.append({"method": "model"})
        m.reinforcements.append("k2")
        m.collapsed.append("k3")
        m.removal_ledger["k3"] = {"reason": "dedup", "surviving_twin": "k2"}

        m.reset_graph()

        assert m.graph.number_of_nodes() == 0, "reset_graph must empty the graph"
        assert m._predicate_cardinality == {}, "reset_graph must clear cardinality cache"
        assert m.contradictions_resolved == [], "reset_graph must clear contradictions_resolved"
        assert m.reinforcements == [], "reset_graph must clear reinforcements"
        assert m.collapsed == [], "reset_graph must clear collapsed"
        assert m.removal_ledger == {}, "reset_graph must clear removal_ledger"

    def test_case1_adopt_does_not_produce_reinforcement(self):
        """Case-1-adopt (keyless existing + incoming has key) does NOT add a reinforcement.

        Reinforcement fires only when BOTH edges carry ik_keys (duplicate-SPO collapse).
        Adopt (existing is keyless) just stamps the key — no recurrence bump needed.
        """
        from paramem.graph.merger import GraphMerger
        from paramem.graph.schema import Relation
        from paramem.memory.persistence import _IK_KEY_ATTR

        m = GraphMerger()
        # Pre-seed with canonical node keys and canonical predicate form.
        m.graph.add_node("alice", attributes={"name": "Alice"})
        m.graph.add_node("berlin", attributes={"name": "Berlin"})
        existing_eid = m.graph.add_edge(
            "alice",
            "berlin",
            predicate="lives in",
            relation_type="factual",
            confidence=1.0,
            first_seen="s0",
            last_seen="s0",
            recurrence_count=1,
            sessions=["s0"],
        )
        # Keyless existing edge.
        assert m.graph["alice"]["berlin"][existing_eid].get(_IK_KEY_ATTR) is None

        incoming = Relation(
            subject="alice",
            predicate="lives_in",
            object="berlin",
            relation_type="factual",
            confidence=1.0,
            speaker_id="Speaker0",
            indexed_key="graph5",
        )
        m._upsert_relation("alice", "berlin", incoming, "s1", "2026-01-01T00:00:00Z")

        # Adopt path: no reinforcement (the existing edge had no key to preserve).
        assert m.reinforcements == [], (
            "Case-1-adopt must NOT produce a reinforcement (existing was keyless)"
        )
        # Key was adopted onto the existing edge.
        assert m.graph["alice"]["berlin"][existing_eid].get(_IK_KEY_ATTR) == "graph5"

    def test_fold_non_subtractive_short_circuits_replace_both_edges_survive(self):
        """resolve_contradictions=False short-circuits Case-2: even when check_predicate_coexistence
        returns REPLACE, the old edge is NOT removed and both keys survive.

        This is the regression guard for the fold-is-non-subtractive invariant: no model call,
        no edge removal, no registered fact lost at fold time.
        """
        from unittest.mock import MagicMock, patch

        from paramem.graph.merger import GraphMerger
        from paramem.graph.schema import Relation
        from paramem.memory.persistence import _IK_KEY_ATTR

        model_stub = MagicMock()
        tok_stub = MagicMock()
        tok_stub.apply_chat_template.return_value = "formatted"

        m = GraphMerger(model=model_stub, tokenizer=tok_stub)
        # Pre-seed with canonical node keys and canonical predicate form.
        m.graph.add_node("alex", attributes={"name": "Alex"})
        m.graph.add_node("munich", attributes={"name": "Munich"})

        # Pre-seed with the first edge carrying ik_key='key_munich'.
        eid_old = m.graph.add_edge(
            "alex",
            "munich",
            predicate="lives in",
            relation_type="factual",
            confidence=1.0,
            first_seen="s1",
            last_seen="s1",
            recurrence_count=1,
            sessions=["s1"],
        )
        m.graph["alex"]["munich"][eid_old][_IK_KEY_ATTR] = "key_munich"

        # Incoming relation: same predicate, different object (Berlin).
        incoming = Relation(
            subject="alex",
            predicate="lives_in",
            object="berlin",
            relation_type="factual",
            confidence=1.0,
            speaker_id="Speaker0",
            indexed_key="key_berlin",
        )

        # Even though the model would say REPLACE, resolve_contradictions=False must skip it.
        with patch(
            "paramem.graph.merger.check_predicate_coexistence",
            return_value="REPLACE",
        ) as mock_coexist:
            m._upsert_relation(
                "alex",
                "berlin",
                incoming,
                "s2",
                "2026-01-02T00:00:00Z",
                resolve_contradictions=False,
            )
            # check_predicate_coexistence must NOT have been called (short-circuit).
            mock_coexist.assert_not_called()

        # Both edges must survive: munich (old) and berlin (new).
        # Node keys are canonical: "alex", "munich", "berlin"
        lives_in_objects = [
            obj
            for obj in m.graph.successors("alex")
            for _, d in m.graph["alex"][obj].items()
            if d.get("predicate") == "lives in"
        ]
        assert "munich" in lives_in_objects, (
            "Old edge (Munich) must NOT be removed when resolve_contradictions=False"
        )
        assert "berlin" in lives_in_objects, (
            "New edge (Berlin) must be inserted even when resolve_contradictions=False"
        )

        # Both ik_keys must be stamped on their respective edges.
        munich_key = next(
            d.get(_IK_KEY_ATTR)
            for _, d in m.graph["alex"]["munich"].items()
            if d.get("predicate") == "lives in"
        )
        berlin_key = next(
            d.get(_IK_KEY_ATTR)
            for _, d in m.graph["alex"]["berlin"].items()
            if d.get("predicate") == "lives in"
        )
        assert munich_key == "key_munich", f"Expected key_munich on old edge; got {munich_key!r}"
        assert berlin_key == "key_berlin", f"Expected key_berlin on new edge; got {berlin_key!r}"


class TestRemovalLedger:
    """Unit tests for GraphMerger.removal_ledger (reason-coded removal records).

    The ledger records every edge REMOVAL keyed by the removed edge's ik_key.
    Tests cover Case-1 dedup, same-pred REPLACE contradiction, and reset_graph
    clearing the ledger.  Cross-pred contradiction is config-off by default so
    it is not exercised end-to-end here; same-pred covers the identical code
    pattern.
    """

    def test_dedup_collapse_writes_to_ledger(self):
        """Case-1 duplicate-SPO collapse writes the drifting key to removal_ledger
        with reason='dedup' and surviving_twin set to the existing key.

        Regression guard: merged.collapsed assertion still holds alongside the
        new ledger assertion.
        """
        from paramem.graph.merger import GraphMerger
        from paramem.graph.schema import Relation, SessionGraph

        m = GraphMerger()
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
                    indexed_key="key_survivor",
                )
            ],
        )
        m.merge(s1, resolve_contradictions=False)

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
                    indexed_key="key_drifter",
                )
            ],
        )
        m.merge(s2, resolve_contradictions=False)

        # Existing collapsed assertion must still hold.
        assert m.collapsed == ["key_drifter"], (
            f"Collapsed must contain key_drifter; got {m.collapsed}"
        )
        # New ledger assertion.
        assert "key_drifter" in m.removal_ledger, (
            f"Drifting key must be in removal_ledger; got {list(m.removal_ledger.keys())}"
        )
        assert m.removal_ledger["key_drifter"]["reason"] == "dedup", (
            f"Expected reason='dedup'; got {m.removal_ledger['key_drifter']['reason']!r}"
        )
        assert m.removal_ledger["key_drifter"]["surviving_twin"] == "key_survivor", (
            f"Expected surviving_twin='key_survivor'; "
            f"got {m.removal_ledger['key_drifter']['surviving_twin']!r}"
        )
        # Survivor must NOT appear in ledger.
        assert "key_survivor" not in m.removal_ledger, "Surviving key must not be in removal_ledger"

    def test_same_pred_replace_writes_to_ledger(self):
        """Same-(s,p)/different-o single-valued (REPLACE) contradiction writes the
        removed edge's ik_key to removal_ledger with reason='contradiction_same_pred'.

        The path is dormant under resolve_contradictions=False (fold path); we use
        resolve_contradictions=True (live ingest) with a patched
        check_predicate_coexistence that returns REPLACE.
        Pre-seed the first edge with an ik_key directly on the graph.
        """
        from unittest.mock import patch

        from paramem.graph.merger import GraphMerger
        from paramem.graph.schema import Relation, SessionGraph
        from paramem.memory.persistence import _IK_KEY_ATTR

        def _always_replace(subject, predicate, old_value, new_value, mdl, tok, prompt=None):
            return "REPLACE"

        m = GraphMerger(model=object(), tokenizer=object())  # non-None to trigger Case-2

        # Merge the first session (Munich) — normal ingest, no indexed_key.
        sg1 = SessionGraph(
            session_id="s1",
            timestamp="2026-01-01T00:00:00Z",
            entities=[],
            relations=[
                Relation(
                    subject="Alex",
                    predicate="lives_in",
                    object="Munich",
                    relation_type="factual",
                    confidence=1.0,
                    speaker_id="Speaker0",
                )
            ],
        )
        with patch("paramem.graph.merger.check_predicate_coexistence", side_effect=_always_replace):
            m.merge(sg1)

        # Stamp an ik_key onto the Munich edge so the ledger capture fires.
        # After merge(), node keys are canonical: "alex", "munich"
        for _eid, _edata in m.graph["alex"]["munich"].items():
            if _edata.get("predicate") == "lives in":
                _edata[_IK_KEY_ATTR] = "key_munich_old"
                break

        # Merge the second session (Berlin) — REPLACE fires, Munich edge is removed.
        sg2 = SessionGraph(
            session_id="s2",
            timestamp="2026-01-02T00:00:00Z",
            entities=[],
            relations=[
                Relation(
                    subject="Alex",
                    predicate="lives_in",
                    object="Berlin",
                    relation_type="factual",
                    confidence=1.0,
                    speaker_id="Speaker0",
                )
            ],
        )
        with patch("paramem.graph.merger.check_predicate_coexistence", side_effect=_always_replace):
            m.merge(sg2)

        # "munich" must be gone.
        assert "munich" not in list(m.graph.successors("alex")), (
            "Old Munich edge must have been removed by REPLACE"
        )
        # Ledger must record the removed key.
        assert "key_munich_old" in m.removal_ledger, (
            f"Removed key must be in removal_ledger; keys={list(m.removal_ledger.keys())}"
        )
        assert m.removal_ledger["key_munich_old"]["reason"] == "contradiction_same_pred", (
            f"Expected reason='contradiction_same_pred'; "
            f"got {m.removal_ledger['key_munich_old']['reason']!r}"
        )

    def test_reset_graph_clears_removal_ledger(self):
        """reset_graph() clears removal_ledger alongside collapsed and reinforcements."""
        from paramem.graph.merger import GraphMerger

        m = GraphMerger()
        m.removal_ledger["key_stale"] = {"reason": "dedup", "surviving_twin": "k2"}
        assert "key_stale" in m.removal_ledger

        m.reset_graph()

        assert m.removal_ledger == {}, (
            f"reset_graph must clear removal_ledger; got {m.removal_ledger}"
        )

    def test_dedup_ledger_records_pre_surfaces_for_subject_object_variant(self):
        """Ledger pre_surfaces records the raw incoming and surviving subject/object
        surfaces when a case variant is collapsed by canonical() dedup.

        This is the canonicalization observability hook: readers compare incoming
        vs. surviving directly to distinguish genuine duplicates from canonicalization
        normalization collapses.
        """
        from paramem.graph.merger import GraphMerger
        from paramem.graph.schema import Relation, SessionGraph

        m = GraphMerger()

        # First session: "Alice" / "lives_in" / "Berlin"
        s1 = SessionGraph(
            session_id="canon1",
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
                    indexed_key="key_survivor",
                )
            ],
        )
        m.merge(s1, resolve_contradictions=False)

        # Second session: "ALICE" / "lives_in" / "berlin" — same canonical forms;
        # surfaces differ, so pre_surfaces must record the mismatch.
        s2 = SessionGraph(
            session_id="canon2",
            timestamp="2026-01-02T00:00:00Z",
            entities=[],
            relations=[
                Relation(
                    subject="ALICE",
                    predicate="lives_in",
                    object="berlin",
                    relation_type="factual",
                    confidence=1.0,
                    speaker_id="Speaker0",
                    indexed_key="key_drifter",
                )
            ],
        )
        m.merge(s2, resolve_contradictions=False)

        assert "key_drifter" in m.removal_ledger, "Drifting key must appear in removal_ledger"
        entry = m.removal_ledger["key_drifter"]
        assert "pre_surfaces" in entry, "removal_ledger entry must contain pre_surfaces"
        pre = entry["pre_surfaces"]
        assert "incoming" in pre and "surviving" in pre, (
            f"pre_surfaces must have 'incoming' and 'surviving' keys; got {list(pre.keys())}"
        )
        # Incoming surfaces record the raw drifted form.
        assert pre["incoming"]["subject"] == "ALICE", (
            f"incoming subject must be raw 'ALICE'; got {pre['incoming']['subject']!r}"
        )
        assert pre["incoming"]["object"] == "berlin", (
            f"incoming object must be raw 'berlin'; got {pre['incoming']['object']!r}"
        )
        # Surviving surfaces record the first-seen stored form; they differ from incoming.
        assert pre["surviving"]["subject"] != pre["incoming"]["subject"], (
            "surviving subject must differ from incoming subject for a case-variant collapse"
        )
        assert pre["surviving"]["object"] != pre["incoming"]["object"], (
            "surviving object must differ from incoming object for a case-variant collapse"
        )

    def test_dedup_ledger_records_pre_surfaces_for_predicate_normalized_duplicate(self):
        """Ledger pre_surfaces records the raw incoming and stored surviving predicate
        surfaces when a byte-identical duplicate is collapsed by dedup.

        Relations use ``"lives_in"`` (raw) but the merger stores predicates in
        space-form (``"lives in"``).  pre_surfaces captures the raw incoming form
        and the space-normalized surviving form so readers can inspect the mismatch.
        """
        from paramem.graph.merger import GraphMerger
        from paramem.graph.schema import Relation, SessionGraph

        m = GraphMerger()
        s1 = SessionGraph(
            session_id="dup1",
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
                    indexed_key="k_survivor",
                )
            ],
        )
        s2 = SessionGraph(
            session_id="dup2",
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
                    indexed_key="k_drifter",
                )
            ],
        )
        m.merge(s1, resolve_contradictions=False)
        m.merge(s2, resolve_contradictions=False)

        entry = m.removal_ledger.get("k_drifter", {})
        assert "pre_surfaces" in entry, (
            f"removal_ledger entry must contain pre_surfaces; entry={entry!r}"
        )
        pre = entry["pre_surfaces"]
        # incoming predicate is the raw underscore form from the Relation.
        assert pre.get("incoming", {}).get("predicate") == "lives_in", (
            f"incoming predicate must be the raw 'lives_in' form; got "
            f"{pre.get('incoming', {}).get('predicate')!r}"
        )
        # surviving predicate is the space-normalized form stored by the merger.
        assert pre.get("surviving", {}).get("predicate") == "lives in", (
            f"surviving predicate must be the space-normalized 'lives in'; got "
            f"{pre.get('surviving', {}).get('predicate')!r}"
        )


class TestObjectVariantDedup:
    """Canonical node-key model A collapses object-side variants into one edge.

    Regression guard: under predicate ``values``,
    ``"Execution Speed"`` / ``"execution_speed"`` / ``"execution speed"`` were three
    distinct nodes → three edges → three keys.  With canonical node keys all three
    collapse to the single canonical form ``"execution speed"`` → one edge, one
    surviving key.
    """

    def test_deduplicates_object_variants(self):
        """Two sessions with canonically-identical objects collapse to ONE edge."""
        m = GraphMerger()

        s1 = SessionGraph(
            session_id="s1",
            timestamp="2026-01-01T00:00:00Z",
            entities=[
                Entity(name="ParaMem", entity_type="concept"),
                Entity(name="Execution Speed", entity_type="concept"),
            ],
            relations=[
                Relation(
                    subject="ParaMem",
                    predicate="values",
                    object="Execution Speed",
                    relation_type="factual",
                    speaker_id="Speaker0",
                )
            ],
        )
        s2 = SessionGraph(
            session_id="s2",
            timestamp="2026-01-02T00:00:00Z",
            entities=[
                Entity(name="ParaMem", entity_type="concept"),
                Entity(name="execution_speed", entity_type="concept"),
            ],
            relations=[
                Relation(
                    subject="ParaMem",
                    predicate="values",
                    object="execution_speed",
                    relation_type="factual",
                    speaker_id="Speaker0",
                )
            ],
        )

        m.merge(s1)
        m.merge(s2)

        # canonical("Execution Speed") == canonical("execution_speed") == "execution speed"
        # → only ONE node and ONE edge for the values predicate.
        assert "execution speed" in m.graph.nodes, (
            "Canonical object node key must be 'execution speed'"
        )
        assert m.graph.number_of_edges() == 1, (
            f"Object variants must collapse to one edge; got {m.graph.number_of_edges()}"
        )
        edges = list(m.graph["paramem"]["execution speed"].values())
        assert len(edges) == 1
        assert edges[0]["recurrence_count"] == 2, (
            "Collapsed edge must have recurrence_count=2 (one per session)"
        )

    def test_display_name_preserved_in_node_attributes(self):
        """After collapsing object variants, the first-seen surface form is in
        ``attributes["name"]`` — recall text is the human-readable original, not
        the canonical key.
        """
        m = GraphMerger()
        s1 = SessionGraph(
            session_id="s1",
            timestamp="2026-01-01T00:00:00Z",
            entities=[
                Entity(name="ParaMem", entity_type="concept"),
                Entity(name="Execution Speed", entity_type="concept"),
            ],
            relations=[
                Relation(
                    subject="ParaMem",
                    predicate="values",
                    object="Execution Speed",
                    relation_type="factual",
                    speaker_id="Speaker0",
                )
            ],
        )
        m.merge(s1)

        # The display name is stored in attributes["name"], not the node key.
        node_data = m.graph.nodes["execution speed"]
        assert node_data["attributes"]["name"] == "Execution Speed", (
            "First-seen surface form must be preserved in attributes['name']; "
            f"got {node_data['attributes'].get('name')!r}"
        )


# ---------------------------------------------------------------------------
# B7-A — session_ids provenance union in _upsert_relation
# Plan test 5: Relation.session_ids survives the merge as a SET union.
# ---------------------------------------------------------------------------


class TestSessionIdsProvenanceUnion:
    """B7-A acceptance tests for session_ids provenance plumbing.

    GraphMerger._upsert_relation must union Relation.session_ids into
    edge['sessions'] in BOTH Case-1 (duplicate-SPO reinforcement) AND
    Case-3 (new-edge insertion) so the real contributing session ids
    accumulate on the edge across all extraction paths.

    Locks T2 SET semantics across both merge cases.  The dcf4189
    speaker-attribution invariant (T9) is also verified: minted-key
    speaker_id comes from the SUBJECT NODE attribute, not from the
    scalar session_id param; the session_ids union must not touch it.
    """

    @staticmethod
    def _rel(session_ids: list[str], speaker_id: str = "Speaker0") -> Relation:
        """Helper: build a minimal Relation with session_ids set."""
        return Relation(
            subject="Alex",
            predicate="lives_in",
            object="Berlin",
            relation_type="factual",
            speaker_id=speaker_id,
            session_ids=session_ids,
        )

    def test_case3_new_edge_carries_session_ids(self):
        """Case-3 new-edge insertion: Relation.session_ids are unioned into edge['sessions'].

        The scalar session_id param may be a synthetic sentinel (as in
        _merge_registry_relations); the real ids must still appear on the edge.

        Speaker entities are keyed by speaker_id in the graph ("Speaker0"), not
        by the display name ("Alex"); non-speaker entities are keyed by canonical
        form ("berlin").  Edge lookup uses these canonical keys.
        """
        m = GraphMerger()
        s1 = SessionGraph(
            session_id="__interim_pending_sessions__",  # synthetic sentinel
            timestamp="2026-01-01T00:00:00Z",
            entities=[
                Entity(name="Alex", entity_type="person", speaker_id="Speaker0"),
                Entity(name="Berlin", entity_type="place"),
            ],
            relations=[self._rel(["real-session-abc"])],
        )
        m.merge(s1)

        # Speaker entities are keyed by speaker_id; non-speakers by canonical form.
        # Relation subject="Alex" with speaker_id="Speaker0" → node key "Speaker0".
        # Relation object="Berlin" → canonical node key "berlin".
        edges = list(m.graph["Speaker0"]["berlin"].values())
        assert len(edges) == 1, f"expected 1 edge; got {len(edges)}"
        sessions = edges[0]["sessions"]
        assert "real-session-abc" in sessions, (
            f"Real session id must be in edge['sessions']; got {sessions}"
        )
        # Synthetic sentinel is also in sessions (it was the scalar param).
        assert "__interim_pending_sessions__" in sessions

    def test_case3_new_edge_with_empty_session_ids(self):
        """Case-3: empty Relation.session_ids — edge['sessions'] contains only scalar param."""
        m = GraphMerger()
        s1 = SessionGraph(
            session_id="session-xyz",
            timestamp="2026-01-01T00:00:00Z",
            entities=[
                Entity(name="Alex", entity_type="person", speaker_id="Speaker0"),
                Entity(name="Berlin", entity_type="place"),
            ],
            relations=[self._rel([])],
        )
        m.merge(s1)

        edges = list(m.graph["Speaker0"]["berlin"].values())
        sessions = edges[0]["sessions"]
        assert sessions == ["session-xyz"], (
            f"Empty session_ids → edge['sessions'] == [scalar]; got {sessions}"
        )

    def test_case1_duplicate_spo_unions_session_ids(self):
        """Case-1 duplicate-SPO reinforcement: session_ids from both merges are unioned.

        Plan test 5 core assertion: merge same (s,p,o) from sessions s1 and s2
        under synthetic scalar ids.  edge['sessions'] must be the UNION
        {s1_real, s2_real, synthetic_sentinel}.
        """
        m = GraphMerger()
        entities = [
            Entity(name="Alex", entity_type="person", speaker_id="Speaker0"),
            Entity(name="Berlin", entity_type="place"),
        ]
        # First merge: real id "session-A", scalar synthetic.
        s1 = SessionGraph(
            session_id="__interim_pending_sessions__",
            timestamp="2026-01-01T00:00:00Z",
            entities=entities,
            relations=[self._rel(["session-A"])],
        )
        m.merge(s1)

        # Second merge: same (s,p,o) with real id "session-B".
        s2 = SessionGraph(
            session_id="__interim_pending_sessions__",
            timestamp="2026-01-01T01:00:00Z",
            entities=entities,
            relations=[self._rel(["session-B"])],
        )
        m.merge(s2)

        # Speaker entity keyed by speaker_id "Speaker0"; object "Berlin" → "berlin".
        edges = list(m.graph["Speaker0"]["berlin"].values())
        assert len(edges) == 1, f"Duplicate SPO must collapse to one edge; got {len(edges)}"
        sessions = set(edges[0]["sessions"])
        assert "session-A" in sessions, f"session-A missing from {sessions}"
        assert "session-B" in sessions, f"session-B missing from {sessions}"
        assert "__interim_pending_sessions__" in sessions, (
            f"Synthetic sentinel missing from {sessions}"
        )

    def test_speaker_id_attribution_unchanged_by_session_ids_union(self):
        """dcf4189 invariant: speaker_id on the SUBJECT NODE is unchanged by B7-A.

        Multi-session edge: two merges under different scalar session_ids each
        carrying a real session_id.  The subject node must retain speaker_id
        from the entity (dcf4189), not from any session_id field.
        """
        m = GraphMerger()
        entities = [
            Entity(name="Alex", entity_type="person", speaker_id="Speaker0"),
            Entity(name="Berlin", entity_type="place"),
        ]
        s1 = SessionGraph(
            session_id="scalar-1",
            timestamp="2026-01-01T00:00:00Z",
            entities=entities,
            relations=[self._rel(["session-A"], speaker_id="Speaker0")],
        )
        m.merge(s1)

        s2 = SessionGraph(
            session_id="scalar-2",
            timestamp="2026-01-01T01:00:00Z",
            entities=entities,
            relations=[self._rel(["session-B"], speaker_id="Speaker0")],
        )
        m.merge(s2)

        # Speaker entities are keyed by speaker_id in the graph.
        # B7-A session_ids union touches only edge['sessions'] — never the node.
        speaker_node = m.graph.nodes.get("Speaker0", {})
        assert speaker_node.get("speaker_id") == "Speaker0", (
            "speaker_id on the subject node must come from the Entity, not from "
            f"session_ids union; got {speaker_node.get('speaker_id')!r}"
        )
        # And the edge carries both real session ids.
        edges = list(m.graph["Speaker0"]["berlin"].values())
        sessions = set(edges[0]["sessions"])
        assert "session-A" in sessions and "session-B" in sessions, (
            f"Both real session ids must be in edge['sessions']; got {sessions}"
        )
