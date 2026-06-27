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
                speaker_id="speaker0",
            ),
            Relation(
                subject="Alex",
                predicate="works_at",
                object="AutoMate",
                relation_type="factual",
                speaker_id="speaker0",
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
                speaker_id="speaker0",
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
                    speaker_id="speaker0",
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
                    speaker_id="speaker0",
                )
            ],
        )
        merger.merge(g1)
        merger.merge(g2)
        edges = list(merger.graph["a"]["b"].values())
        assert len(edges) == 1
        # canonical() folds underscore → space; predicate stored in canonical form
        assert edges[0]["predicate"] == "works at"
        assert edges[0]["reinforcement_count"] == 2


class TestEntityResolution:
    def test_exact_match(self, merger, session_graph_1, session_graph_2):
        merger.merge(session_graph_1)
        merger.merge(session_graph_2)
        # "Alex" node key is canonical("Alex") == "alex" (node-key model A)
        assert "alex" in merger.graph.nodes
        assert merger.graph.nodes["alex"]["reinforcement_count"] == 2

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
                    speaker_id="speaker0",
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
                    speaker_id="speaker0",
                )
            ],
        )
        merger.merge(g1)
        merger.merge(g2)

        # Same predicate between same nodes should be aggregated
        # Node keys are canonical: "a", "b"
        edges = list(merger.graph["a"]["b"].values())
        assert len(edges) == 1
        assert edges[0]["reinforcement_count"] == 2
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
                    speaker_id="speaker0",
                ),
                Relation(
                    subject="A",
                    predicate="manages",
                    object="B",
                    relation_type="factual",
                    speaker_id="speaker0",
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
                    name="speaker0",
                    entity_type="person",
                    speaker_id="speaker0",
                )
            ],
            relations=[
                Relation(
                    subject="speaker0",
                    predicate="lives_in",
                    object="Portland",
                    relation_type="factual",
                    speaker_id="speaker0",
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
                    speaker_id="speaker0",
                ),
                Entity(name="Portland", entity_type="place"),
            ],
            relations=[
                Relation(
                    subject="Alex",
                    predicate="lives_in",
                    object="Portland",
                    relation_type="factual",
                    speaker_id="speaker0",
                )
            ],
        )
        m.merge(sg1)
        m.merge(sg2)

        # Must collapse to a single person node — no "speaker0" + "Alex" split.
        person_nodes = [n for n, d in m.graph.nodes(data=True) if d.get("entity_type") == "person"]
        assert len(person_nodes) == 1, (
            f"Expected 1 person node, got {len(person_nodes)}: {person_nodes}"
        )

    def test_speaker_node_keyed_by_speaker_id(self):
        """Speaker entity is keyed by entity.speaker_id verbatim in the graph;
        the display name moves to ``attributes["name"]``.

        Under lowercase-uniform identity the ingest safety-net guarantees
        speaker_id is always lowercase ``speaker{N}`` before it reaches the
        merger.  The merger uses entity.speaker_id verbatim as the node key
        (no casing step needed).
        """
        m = GraphMerger(similarity_threshold=85.0)
        sg = SessionGraph(
            session_id="s1",
            timestamp="2026-01-01T00:00:00Z",
            entities=[
                Entity(
                    name="Alex",
                    entity_type="person",
                    speaker_id="speaker0",
                )
            ],
            relations=[],
        )
        m.merge(sg)
        # Node key IS entity.speaker_id verbatim (lowercase).
        assert "speaker0" in m.graph.nodes
        assert "Alex" not in m.graph.nodes
        assert m.graph.nodes["speaker0"]["speaker_id"] == "speaker0"
        assert m.graph.nodes["speaker0"]["attributes"]["name"] == "Alex"

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
        assert m.graph.nodes["portland"]["reinforcement_count"] == 2

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
                    speaker_id="speaker0",
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
                    speaker_id="speaker0",
                )
            ],
            relations=[],
        )
        m.merge(sg1)
        m.merge(sg2)
        # Speaker entity is keyed by casefolded speaker_id ("speaker0"); display name
        # lives in attributes alongside the merged role / last_name.
        attrs = m.graph.nodes["speaker0"]["attributes"]
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
                    speaker_id="speaker0",
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
                    speaker_id="speaker0",
                )
            ],
            relations=[],
        )
        m.merge(sg1)
        m.merge(sg2)
        assert m.graph.nodes["speaker0"]["attributes"]["has_email"] == "alex@example.com"

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
                    speaker_id="speaker0",
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
                    speaker_id="speaker0",
                )
            ],
            relations=[],
        )
        m.merge(sg1)
        m.merge(sg2)
        assert m.graph.nodes["speaker0"]["attributes"]["has_phone"] == "+1 555 123 4567"

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
                    speaker_id="speaker0",
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
                    speaker_id="speaker0",
                )
            ],
            relations=[],
        )
        m.merge(sg1)
        m.merge(sg2)
        assert m.graph.nodes["speaker0"]["attributes"]["has_email"] == "alex@example.com"


class TestMultiUserNameCollision:
    """Two distinct disclosed speakers with the same display name must NOT
    collapse into one graph node.

    This is the multi-user PA case: speaker0 (Alex Walker) and
    speaker1 (a different Alex) both enrol with display name
    ``Alex``.  Without a guard, ``_resolve_entity`` Tier 1 (exact
    name match) returns speaker0's node when speaker1's entity arrives,
    and ``_upsert_entity`` happily folds speaker1's facts into
    speaker0's node — corrupting both speakers' graphs.

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
        m.merge(self._build_speaker_session("s001", "speaker0", "Walker", "Portland"))
        m.merge(self._build_speaker_session("s002", "speaker1", "Schmidt", "Munich"))

        # Two speaker nodes — separate identities.
        speaker_nodes = [
            (node, data)
            for node, data in m.graph.nodes(data=True)
            if data.get("speaker_id") in {"speaker0", "speaker1"}
        ]
        speaker_ids = {data["speaker_id"] for _, data in speaker_nodes}
        assert speaker_ids == {"speaker0", "speaker1"}, (
            f"Expected separate nodes for speaker0 and speaker1, got "
            f"speaker_ids={speaker_ids} on nodes "
            f"{[(n, d.get('speaker_id'), d.get('attributes', {})) for n, d in speaker_nodes]}"
        )

        # Each carries its own last_name — no cross-contamination.
        by_sid = {data["speaker_id"]: (node, data) for node, data in speaker_nodes}
        speaker0_node, speaker0_data = by_sid["speaker0"]
        speaker1_node, speaker1_data = by_sid["speaker1"]
        assert speaker0_data["attributes"].get("last_name") == "Walker", (
            f"speaker0's last_name attribute corrupted: got {speaker0_data['attributes']!r}"
        )
        assert speaker1_data["attributes"].get("last_name") == "Schmidt", (
            f"speaker1's last_name attribute corrupted: got {speaker1_data['attributes']!r}"
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
        construction: speaker IDs follow the ``speaker{N}`` pattern
        produced by the speaker pool, so a display name like
        ``"Alex"`` will never collide with a casefolded speaker-id node key.
        The third-party Alex becomes a separate node keyed by
        ``"alex"`` (canonical form); speaker0's node is keyed by
        ``"speaker0"`` (casefolded §0 key) with its own attributes intact.
        """
        m = GraphMerger(similarity_threshold=85.0)
        m.merge(self._build_speaker_session("s001", "speaker0", "Walker", "Portland"))
        third_party = SessionGraph(
            session_id="s002",
            timestamp="2026-05-06T00:00:00Z",
            entities=[
                Entity(name="Alex", entity_type="person", attributes={}),
            ],
            relations=[],
        )
        m.merge(third_party)

        # speaker0 is keyed by the casefolded speaker_id ("speaker0");
        # third-party Alex is keyed by canonical name "alex" — disjoint
        # namespaces, two separate nodes.
        assert "speaker0" in m.graph.nodes
        assert "alex" in m.graph.nodes
        assert m.graph.nodes["speaker0"]["speaker_id"] == "speaker0"
        assert m.graph.nodes["alex"].get("speaker_id") is None
        # speaker0's last_name attribute is untouched by the third-party
        # merge (different node, different namespace).
        assert m.graph.nodes["speaker0"]["attributes"]["last_name"] == "Walker"


class TestPromptsDirOverride:
    """Custom prompts_dir overrides the inline fallback constants."""

    def test_custom_prompts_dir_loaded_into_instance_attributes(self, tmp_path):
        """GraphMerger(prompts_dir=...) must resolve _coexistence_prompt from the
        supplied directory, not from the inline _COEXISTENCE_PROMPT constant.
        """
        coexistence_content = "CUSTOM_COEXISTENCE_MARKER sentinel text"
        (tmp_path / "merger_coexistence.txt").write_text(coexistence_content)

        m = GraphMerger(prompts_dir=tmp_path)

        assert m._coexistence_prompt == coexistence_content.strip(), (
            "Custom merger_coexistence.txt must override the inline fallback"
        )

    def test_missing_files_fall_back_to_inline_constants(self, tmp_path):
        """A prompts_dir that lacks the prompt files must fall back to the
        inline _COEXISTENCE_PROMPT constant.
        """
        from paramem.graph.merger import _COEXISTENCE_PROMPT

        # tmp_path exists but contains no prompt files.
        m = GraphMerger(prompts_dir=tmp_path)

        assert m._coexistence_prompt == _COEXISTENCE_PROMPT, (
            "Missing file must fall back to inline _COEXISTENCE_PROMPT"
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
        predicate removes the old edge (older last_seen) and inserts the new one
        (fresher last_seen).  The recency rule picks the incoming as the winner."""
        from unittest.mock import patch

        model_stub, tok_stub, coexist_fn = self._build_stub_model(is_single_valued=True)

        with patch(
            "paramem.graph.merger.check_predicate_coexistence",
            side_effect=coexist_fn,
        ):
            m = GraphMerger(model=model_stub, tokenizer=tok_stub)
            # sg1 carries an older last_seen so sg2's relation wins under the recency rule.
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
                        speaker_id="speaker0",
                        last_seen="2026-01-01T00:00:00Z",
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
                        speaker_id="speaker0",
                        last_seen="2026-01-02T00:00:00Z",
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
                    speaker_id="speaker0",
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
                    speaker_id="speaker0",
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
                        speaker_id="speaker0",
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
                        speaker_id="speaker0",
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
                    speaker_id="speaker0",
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
                    speaker_id="speaker0",
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
            reinforcement_count=1,
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
                    speaker_id="speaker0",
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
        """A merge with no duplicate SPO produces an empty reinforcements dict."""
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
                    speaker_id="speaker0",
                    indexed_key="graph1",
                )
            ],
        )
        m.merge(session, resolve_contradictions=False)
        assert m.reinforcements == {}, "No duplicate → reinforcements must be empty"

    def test_reinforcements_populated_on_duplicate_spo_collapse(self):
        """Two recon edges with same (s,p,o) but different ik_keys: Case-1 fires and
        the surviving key appears in reinforcements with its last_seen timestamp."""
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
                    speaker_id="speaker0",
                    indexed_key="graph1",
                )
            ],
        )
        m.merge(s1, resolve_contradictions=False)
        assert m.reinforcements == {}, "First merge is net-new — no reinforcement yet"

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
                    speaker_id="speaker0",
                    indexed_key="graph2",
                )
            ],
        )
        m.merge(s2, resolve_contradictions=False)

        assert len(m.reinforcements) == 1, (
            f"Duplicate-SPO collapse must produce 1 reinforcement entry; got {m.reinforcements}"
        )
        # The survivor is the EXISTING edge's key (graph1), not the incoming (graph2).
        assert "graph1" in m.reinforcements, (
            f"Surviving key must be graph1 (existing edge); got keys={list(m.reinforcements)}"
        )
        # The carried last_seen is the freshest (max) of both edges' timestamps.
        assert m.reinforcements["graph1"] == "2026-01-02T00:00:00Z", (
            "last_seen must be the freshest timestamp (s2.timestamp); "
            f"got {m.reinforcements['graph1']!r}"
        )

    def test_reinforcements_reset_graph_clears_reinforcements(self):
        """reset_graph() clears reinforcements from the prior fold."""
        from paramem.graph.merger import GraphMerger

        m = GraphMerger()
        m.reinforcements = {"graph_stale": "2026-01-01T00:00:00Z"}
        m.reset_graph()
        assert m.reinforcements == {}, "reset_graph must clear reinforcements"

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
                    speaker_id="speaker0",
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
                    speaker_id="speaker0",
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
        assert "graph1" in m.reinforcements, (
            f"Surviving key must be graph1 (existing edge); got keys={list(m.reinforcements)}"
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
        m.reinforcements["k2"] = "2026-01-01T00:00:00Z"
        m.collapsed.append("k3")
        m.removal_ledger["k3"] = {"reason": "dedup", "surviving_twin": "k2"}

        m.reset_graph()

        assert m.graph.number_of_nodes() == 0, "reset_graph must empty the graph"
        assert m._predicate_cardinality == {}, "reset_graph must clear cardinality cache"
        assert m.contradictions_resolved == [], "reset_graph must clear contradictions_resolved"
        assert m.reinforcements == {}, "reset_graph must clear reinforcements"
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
            reinforcement_count=1,
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
            speaker_id="speaker0",
            indexed_key="graph5",
        )
        m._upsert_relation("alice", "berlin", incoming, "s1", "2026-01-01T00:00:00Z")

        # Adopt path: no reinforcement (the existing edge had no key to preserve).
        assert m.reinforcements == {}, (
            "Case-1-adopt must NOT produce a reinforcement (existing was keyless)"
        )
        # Key was adopted onto the existing edge.
        assert m.graph["alice"]["berlin"][existing_eid].get(_IK_KEY_ATTR) == "graph5"

    def test_resolve_contradictions_false_short_circuits_both_edges_survive(self):
        """resolve_contradictions=False short-circuits Case-2: no model call and both edges survive,
        regardless of distinct last_seen timestamps.
        """
        from unittest.mock import MagicMock, patch

        from paramem.graph.merger import GraphMerger
        from paramem.graph.schema import Relation
        from paramem.memory.persistence import _IK_KEY_ATTR

        model_stub = MagicMock()
        tok_stub = MagicMock()
        tok_stub.apply_chat_template.return_value = "formatted"

        m = GraphMerger(model=model_stub, tokenizer=tok_stub)
        m.graph.add_node("alex", attributes={"name": "Alex"})
        m.graph.add_node("munich", attributes={"name": "Munich"})

        eid_old = m.graph.add_edge(
            "alex",
            "munich",
            predicate="lives in",
            relation_type="factual",
            confidence=1.0,
            first_seen="s1",
            last_seen="2026-01-01T00:00:00Z",
            reinforcement_count=1,
            sessions=["s1"],
        )
        m.graph["alex"]["munich"][eid_old][_IK_KEY_ATTR] = "key_munich"

        # Incoming has a FRESHER last_seen — but resolve_contradictions=False must skip.
        incoming = Relation(
            subject="alex",
            predicate="lives_in",
            object="berlin",
            relation_type="factual",
            confidence=1.0,
            speaker_id="speaker0",
            indexed_key="key_berlin",
            last_seen="2026-01-02T00:00:00Z",
        )

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
            mock_coexist.assert_not_called()

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

    def test_recency_distinct_timestamps_winner_survives_loser_removed(self):
        """resolve_contradictions=True + REPLACE verdict + distinct last_seen timestamps:
        the fresher rival is retained and the staler one is retired.

        Incoming (Berlin, 2026-01-02) vs rival (Munich, 2026-01-01) → incoming wins;
        Munich edge is removed and ledgered.
        """
        from unittest.mock import MagicMock, patch

        from paramem.graph.merger import GraphMerger
        from paramem.graph.schema import Relation
        from paramem.memory.persistence import _IK_KEY_ATTR

        model_stub = MagicMock()
        tok_stub = MagicMock()
        tok_stub.apply_chat_template.return_value = "formatted"

        m = GraphMerger(model=model_stub, tokenizer=tok_stub)
        m.graph.add_node("alex", attributes={"name": "Alex"})
        m.graph.add_node("munich", attributes={"name": "Munich"})

        eid_old = m.graph.add_edge(
            "alex",
            "munich",
            predicate="lives in",
            relation_type="factual",
            confidence=1.0,
            first_seen="s1",
            last_seen="2026-01-01T00:00:00Z",
            reinforcement_count=1,
            sessions=["s1"],
        )
        m.graph["alex"]["munich"][eid_old][_IK_KEY_ATTR] = "key_munich"
        m._predicate_cardinality["lives in"] = False  # pre-cache as single-valued

        incoming = Relation(
            subject="alex",
            predicate="lives_in",
            object="berlin",
            relation_type="factual",
            confidence=1.0,
            speaker_id="speaker0",
            indexed_key="key_berlin",
            last_seen="2026-01-02T00:00:00Z",
        )

        with patch(
            "paramem.graph.merger.check_predicate_coexistence",
            return_value="REPLACE",
        ):
            m._upsert_relation(
                "alex",
                "berlin",
                incoming,
                "s2",
                "2026-01-02T00:00:00Z",
                resolve_contradictions=True,
            )

        # Munich (older) must be gone; Berlin (fresher) must survive.
        lives_in_objects = [
            obj
            for obj in m.graph.successors("alex")
            for _, d in m.graph["alex"][obj].items()
            if d.get("predicate") == "lives in"
        ]
        assert "munich" not in lives_in_objects, (
            "Staler Munich edge must be retired when incoming is uniquely freshest"
        )
        assert "berlin" in lives_in_objects, "Fresher Berlin edge must survive"

        assert "key_munich" in m.removal_ledger, "Retired rival key must be in removal_ledger"
        assert m.removal_ledger["key_munich"]["reason"] == "contradiction_same_pred"
        assert m.removal_ledger["key_munich"]["old_object"] == "munich"
        assert m.removal_ledger["key_munich"]["new_object"] == "berlin"

    def test_recency_tied_timestamps_both_edges_survive(self):
        """resolve_contradictions=True + REPLACE verdict + equal NON-EMPTY last_seen:
        tied datestamps → coexist; both edges survive, no ledger entry.

        Covers the within-session tie case (shared real timestamp).  Empty/legacy
        keys are a separate rule (any-empty → coexist) tested in TestRecencyAnyEmpty.
        """
        from unittest.mock import MagicMock, patch

        from paramem.graph.merger import GraphMerger
        from paramem.graph.schema import Relation
        from paramem.memory.persistence import _IK_KEY_ATTR

        model_stub = MagicMock()
        tok_stub = MagicMock()
        tok_stub.apply_chat_template.return_value = "formatted"

        m = GraphMerger(model=model_stub, tokenizer=tok_stub)
        m.graph.add_node("alex", attributes={"name": "Alex"})
        m.graph.add_node("munich", attributes={"name": "Munich"})

        same_ts = "2026-01-01T00:00:00Z"
        eid_old = m.graph.add_edge(
            "alex",
            "munich",
            predicate="lives in",
            relation_type="factual",
            confidence=1.0,
            first_seen="s1",
            last_seen=same_ts,
            reinforcement_count=1,
            sessions=["s1"],
        )
        m.graph["alex"]["munich"][eid_old][_IK_KEY_ATTR] = "key_munich"
        m._predicate_cardinality["lives in"] = False  # pre-cache as single-valued

        incoming = Relation(
            subject="alex",
            predicate="lives_in",
            object="berlin",
            relation_type="factual",
            confidence=1.0,
            speaker_id="speaker0",
            indexed_key="key_berlin",
            last_seen=same_ts,  # tied timestamp
        )

        with patch(
            "paramem.graph.merger.check_predicate_coexistence",
            return_value="REPLACE",
        ):
            m._upsert_relation(
                "alex",
                "berlin",
                incoming,
                "s1",
                same_ts,
                resolve_contradictions=True,
            )

        # Tied → coexist: both edges survive.
        lives_in_objects = [
            obj
            for obj in m.graph.successors("alex")
            for _, d in m.graph["alex"][obj].items()
            if d.get("predicate") == "lives in"
        ]
        assert "munich" in lives_in_objects, "Munich edge must survive (tied timestamp → coexist)"
        assert "berlin" in lives_in_objects, (
            "Berlin edge must be inserted (tied timestamp → coexist)"
        )

        # No removal_ledger entry for either key.
        assert "key_munich" not in m.removal_ledger, "Tied key must NOT be ledgered"
        assert "key_berlin" not in m.removal_ledger, "Incoming tied key must NOT be ledgered"


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
                    speaker_id="speaker0",
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
                    speaker_id="speaker0",
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
        """Same-(s,p)/different-o single-valued (REPLACE) + incoming fresher:
        removed rival's ik_key goes to removal_ledger with reason='contradiction_same_pred'.

        Uses resolve_contradictions=True with distinct last_seen so the recency rule
        picks the incoming (Berlin, 2026-01-02) over the rival (Munich, 2026-01-01).
        Pre-seed the first edge with an ik_key directly on the graph.
        """
        from unittest.mock import patch

        from paramem.graph.merger import GraphMerger
        from paramem.graph.schema import Relation, SessionGraph
        from paramem.memory.persistence import _IK_KEY_ATTR

        def _always_replace(subject, predicate, old_value, new_value, mdl, tok, prompt=None):
            return "REPLACE"

        m = GraphMerger(model=object(), tokenizer=object())  # non-None to trigger Case-2

        # Merge the first session (Munich) with an older last_seen.
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
                    speaker_id="speaker0",
                    last_seen="2026-01-01T00:00:00Z",
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

        # Merge the second session (Berlin) with a fresher last_seen → incoming wins.
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
                    speaker_id="speaker0",
                    last_seen="2026-01-02T00:00:00Z",
                )
            ],
        )
        with patch("paramem.graph.merger.check_predicate_coexistence", side_effect=_always_replace):
            m.merge(sg2)

        # "munich" must be gone (incoming is fresher).
        assert "munich" not in list(m.graph.successors("alex")), (
            "Older Munich edge must have been removed by REPLACE+recency"
        )
        # Ledger must record the removed rival key.
        assert "key_munich_old" in m.removal_ledger, (
            f"Removed rival key must be in removal_ledger; keys={list(m.removal_ledger.keys())}"
        )
        assert m.removal_ledger["key_munich_old"]["reason"] == "contradiction_same_pred", (
            f"Expected reason='contradiction_same_pred'; "
            f"got {m.removal_ledger['key_munich_old']['reason']!r}"
        )
        assert m.removal_ledger["key_munich_old"]["old_object"] == "munich"
        assert m.removal_ledger["key_munich_old"]["new_object"] == "berlin"

    def test_incoming_loses_when_rival_is_fresher(self):
        """Same-(s,p)/different-o REPLACE + rival has fresher last_seen:
        rival survives, incoming is NOT inserted, incoming's indexed_key is ledgered.

        This is the new incoming-loses path introduced by the unified recency rule.
        rival (Munich, 2026-01-02) vs incoming (Berlin, 2026-01-01) → Munich wins.
        """
        from unittest.mock import MagicMock, patch

        from paramem.graph.merger import GraphMerger
        from paramem.graph.schema import Relation
        from paramem.memory.persistence import _IK_KEY_ATTR

        model_stub = MagicMock()
        tok_stub = MagicMock()
        tok_stub.apply_chat_template.return_value = "formatted"

        m = GraphMerger(model=model_stub, tokenizer=tok_stub)
        m.graph.add_node("alex", attributes={"name": "Alex"})
        m.graph.add_node("munich", attributes={"name": "Munich"})

        eid_old = m.graph.add_edge(
            "alex",
            "munich",
            predicate="lives in",
            relation_type="factual",
            confidence=1.0,
            first_seen="s1",
            last_seen="2026-01-02T00:00:00Z",  # FRESHER than incoming
            reinforcement_count=1,
            sessions=["s1"],
        )
        m.graph["alex"]["munich"][eid_old][_IK_KEY_ATTR] = "key_munich"
        m._predicate_cardinality["lives in"] = False  # pre-cache as single-valued

        # Incoming (Berlin) has an OLDER last_seen → loses to Munich.
        incoming = Relation(
            subject="alex",
            predicate="lives_in",
            object="berlin",
            relation_type="factual",
            confidence=1.0,
            speaker_id="speaker0",
            indexed_key="key_berlin_old",
            last_seen="2026-01-01T00:00:00Z",  # OLDER → loses
        )

        return_value = None
        with patch(
            "paramem.graph.merger.check_predicate_coexistence",
            return_value="REPLACE",
        ):
            return_value = m._upsert_relation(
                "alex",
                "berlin",
                incoming,
                "s2",
                "2026-01-01T00:00:00Z",
                resolve_contradictions=True,
            )

        # Return value must be None (incoming is not inserted).
        assert return_value is None, "_upsert_relation must return None when incoming loses"

        # Munich (fresher rival) must survive; Berlin (older incoming) must NOT be inserted.
        lives_in_objects = [
            obj
            for obj in m.graph.successors("alex")
            for _, d in m.graph["alex"][obj].items()
            if d.get("predicate") == "lives in"
        ]
        assert "munich" in lives_in_objects, "Fresher rival Munich must survive"
        assert "berlin" not in lives_in_objects, "Older incoming Berlin must NOT be inserted"

        # Incoming's indexed_key must be in the removal_ledger.
        assert "key_berlin_old" in m.removal_ledger, (
            f"Incoming loser key must be ledgered; keys={list(m.removal_ledger.keys())}"
        )
        assert m.removal_ledger["key_berlin_old"]["reason"] == "contradiction_same_pred"
        assert m.removal_ledger["key_berlin_old"]["old_object"] == "berlin"
        assert m.removal_ledger["key_berlin_old"]["new_object"] == "munich"

        # Munich's key must NOT be in the removal_ledger (it won).
        assert "key_munich" not in m.removal_ledger, "Winner key must NOT be ledgered"

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
                    speaker_id="speaker0",
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
                    speaker_id="speaker0",
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
                    speaker_id="speaker0",
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
                    speaker_id="speaker0",
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


class TestRecencyAnyEmpty:
    """Recency rule: ANY empty last_seen among candidates → COEXIST (no removal).

    Covers:
    - incoming last_seen="" with a dated rival → coexist (C1 regression test)
    - dated incoming with a rival last_seen="" → coexist
    - both incoming and rival last_seen="" → coexist
    - 3+ dated rivals with a unique max → strictly-older pair retired, max survives
    - 3+ dated rivals with a top tie → strictly-older retired, tied pair coexist
    """

    @staticmethod
    def _make_graph_with_rival(rival_last_seen: str) -> "GraphMerger":
        """Return a GraphMerger with a single (alex, lives in, munich) rival edge."""
        from unittest.mock import MagicMock

        from paramem.graph.merger import GraphMerger
        from paramem.memory.persistence import _IK_KEY_ATTR

        model_stub = MagicMock()
        tok_stub = MagicMock()
        tok_stub.apply_chat_template.return_value = "formatted"

        m = GraphMerger(model=model_stub, tokenizer=tok_stub)
        m.graph.add_node("alex", attributes={"name": "Alex"})
        m.graph.add_node("munich", attributes={"name": "Munich"})
        eid = m.graph.add_edge(
            "alex",
            "munich",
            predicate="lives in",
            relation_type="factual",
            confidence=1.0,
            first_seen="s1",
            last_seen=rival_last_seen,
            reinforcement_count=1,
            sessions=["s1"],
        )
        m.graph["alex"]["munich"][eid][_IK_KEY_ATTR] = "key_munich"
        m._predicate_cardinality["lives in"] = False  # pre-cache as single-valued
        return m

    @staticmethod
    def _upsert_berlin(m: "GraphMerger", incoming_last_seen: str) -> None:
        """Insert (alex, lives in, berlin) with the given last_seen."""
        from unittest.mock import patch

        from paramem.graph.schema import Relation

        incoming = Relation(
            subject="alex",
            predicate="lives_in",
            object="berlin",
            relation_type="factual",
            confidence=1.0,
            speaker_id="speaker0",
            indexed_key="key_berlin",
            last_seen=incoming_last_seen,
        )
        with patch(
            "paramem.graph.merger.check_predicate_coexistence",
            return_value="REPLACE",
        ):
            m._upsert_relation(
                "alex",
                "berlin",
                incoming,
                "__recon__",
                "",  # timestamp="" — fold/recon path; must NOT fabricate now()
                resolve_contradictions=True,
            )

    def _get_lives_in_objects(self, m: "GraphMerger") -> "list[str]":

        return [
            obj
            for obj in m.graph.successors("alex")
            for _, d in m.graph["alex"][obj].items()
            if d.get("predicate") == "lives in"
        ]

    def test_incoming_empty_rival_dated_coexist(self):
        """C1 regression: incoming last_seen="" vs dated rival → COEXIST.

        Before C1 fix, timestamp=now() was passed to the merger's SessionGraph so
        the merger evaluated incoming_ls = "" or now() = now(), making the legacy key
        appear as the unique freshest → the dated rival was wrongly retired.
        With the fix, timestamp="" so incoming_ls = "" → any-empty rule → COEXIST.
        """
        m = self._make_graph_with_rival(rival_last_seen="2026-01-01T00:00:00Z")
        self._upsert_berlin(m, incoming_last_seen="")  # legacy incoming

        objects = self._get_lives_in_objects(m)
        assert "munich" in objects, (
            "Dated rival Munich must survive — C1: legacy incoming must NOT beat a dated rival"
        )
        assert "berlin" in objects, "Legacy incoming Berlin must be inserted (coexist)"
        assert not m.removal_ledger, (
            f"No ledger entry for any-empty coexist; got {m.removal_ledger}"
        )

    def test_rival_empty_incoming_dated_coexist(self):
        """Dated incoming vs rival last_seen="" → COEXIST (any-empty rule)."""
        m = self._make_graph_with_rival(rival_last_seen="")  # legacy rival
        self._upsert_berlin(m, incoming_last_seen="2026-01-02T00:00:00Z")  # dated incoming

        objects = self._get_lives_in_objects(m)
        assert "munich" in objects, "Legacy rival Munich must survive (any-empty → coexist)"
        assert "berlin" in objects, "Dated incoming Berlin must be inserted (coexist)"
        assert not m.removal_ledger, (
            f"No ledger entry for any-empty coexist; got {m.removal_ledger}"
        )

    def test_both_empty_coexist(self):
        """Both incoming and rival last_seen="" → COEXIST (any-empty rule covers all-empty)."""
        m = self._make_graph_with_rival(rival_last_seen="")
        self._upsert_berlin(m, incoming_last_seen="")

        objects = self._get_lives_in_objects(m)
        assert "munich" in objects, "Legacy rival Munich must survive"
        assert "berlin" in objects, "Legacy incoming Berlin must be inserted"
        assert not m.removal_ledger, (
            f"No ledger entry for all-empty coexist; got {m.removal_ledger}"
        )

    def test_three_rivals_unique_max_retires_two_older(self):
        """3 rivals all dated, unique max: two strictly-older rivals retired, max kept.

        rivals: vienna="2026-01-01", paris="2026-01-02" (max), madrid="2026-01-01"
        incoming: berlin="2026-01-03" (newest) → berlin wins, paris is retired too.

        Actually, berlin is uniquely freshest: all rivals are < berlin.
        """
        from unittest.mock import MagicMock, patch

        from paramem.graph.merger import GraphMerger
        from paramem.graph.schema import Relation
        from paramem.memory.persistence import _IK_KEY_ATTR

        m = GraphMerger(model=MagicMock(), tokenizer=MagicMock())
        m._predicate_cardinality["lives in"] = False
        for city, ts, key in [
            ("vienna", "2026-01-01T00:00:00Z", "key_vienna"),
            ("paris", "2026-01-02T00:00:00Z", "key_paris"),
            ("madrid", "2026-01-01T00:00:00Z", "key_madrid"),
        ]:
            m.graph.add_node(city, attributes={"name": city})
            eid = m.graph.add_edge(
                "alex",
                city,
                predicate="lives in",
                relation_type="factual",
                confidence=1.0,
                first_seen="s0",
                last_seen=ts,
                reinforcement_count=1,
                sessions=["s0"],
            )
            m.graph["alex"][city][eid][_IK_KEY_ATTR] = key

        incoming = Relation(
            subject="alex",
            predicate="lives_in",
            object="berlin",
            relation_type="factual",
            confidence=1.0,
            speaker_id="speaker0",
            indexed_key="key_berlin",
            last_seen="2026-01-03T00:00:00Z",
        )
        with patch("paramem.graph.merger.check_predicate_coexistence", return_value="REPLACE"):
            m._upsert_relation("alex", "berlin", incoming, "__r__", "", resolve_contradictions=True)

        objects = [
            obj
            for obj in m.graph.successors("alex")
            for _, d in m.graph["alex"][obj].items()
            if d.get("predicate") == "lives in"
        ]
        assert "berlin" in objects, "Incoming (uniquely freshest) must survive"
        assert "vienna" not in objects, "Strictly-older vienna must be retired"
        assert "paris" not in objects, "Strictly-older paris must be retired"
        assert "madrid" not in objects, "Strictly-older madrid must be retired"
        assert "key_vienna" in m.removal_ledger
        assert "key_paris" in m.removal_ledger
        assert "key_madrid" in m.removal_ledger

    def test_three_rivals_top_tie_older_retired_tied_pair_coexist(self):
        """3 rivals, two at max, one strictly-older: strictly-older retired, tied pair coexist.

        rivals: vienna="2026-01-01" (older), paris="2026-01-02" (max), madrid="2026-01-02" (max)
        incoming: berlin="2026-01-01" (older than tied max) → berlin NOT inserted,
        vienna retired, paris and madrid survive.
        """
        from unittest.mock import MagicMock, patch

        from paramem.graph.merger import GraphMerger
        from paramem.graph.schema import Relation
        from paramem.memory.persistence import _IK_KEY_ATTR

        m = GraphMerger(model=MagicMock(), tokenizer=MagicMock())
        m._predicate_cardinality["lives in"] = False
        for city, ts, key in [
            ("vienna", "2026-01-01T00:00:00Z", "key_vienna"),
            ("paris", "2026-01-02T00:00:00Z", "key_paris"),
            ("madrid", "2026-01-02T00:00:00Z", "key_madrid"),
        ]:
            m.graph.add_node(city, attributes={"name": city})
            eid = m.graph.add_edge(
                "alex",
                city,
                predicate="lives in",
                relation_type="factual",
                confidence=1.0,
                first_seen="s0",
                last_seen=ts,
                reinforcement_count=1,
                sessions=["s0"],
            )
            m.graph["alex"][city][eid][_IK_KEY_ATTR] = key

        incoming = Relation(
            subject="alex",
            predicate="lives_in",
            object="berlin",
            relation_type="factual",
            confidence=1.0,
            speaker_id="speaker0",
            indexed_key="key_berlin",
            last_seen="2026-01-01T00:00:00Z",
        )
        with patch("paramem.graph.merger.check_predicate_coexistence", return_value="REPLACE"):
            result = m._upsert_relation(
                "alex", "berlin", incoming, "__r__", "", resolve_contradictions=True
            )

        objects = [
            obj
            for obj in m.graph.successors("alex")
            for _, d in m.graph["alex"][obj].items()
            if d.get("predicate") == "lives in"
        ]
        assert result is None, "Incoming (older than tied max) must NOT be inserted"
        assert "berlin" not in objects, "Strictly-older incoming Berlin must not be inserted"
        assert "paris" in objects, "Tied-max Paris must survive"
        assert "madrid" in objects, "Tied-max Madrid must survive"
        assert "vienna" not in objects, "Strictly-older Vienna must be retired"
        assert "key_vienna" in m.removal_ledger, "Strictly-older Vienna key must be ledgered"
        assert "key_berlin" in m.removal_ledger, "Loser incoming key must be ledgered"
        assert "key_paris" not in m.removal_ledger, "Winner Paris key must NOT be ledgered"
        assert "key_madrid" not in m.removal_ledger, "Winner Madrid key must NOT be ledgered"


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
                    speaker_id="speaker0",
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
                    speaker_id="speaker0",
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
        assert edges[0]["reinforcement_count"] == 2, (
            "Collapsed edge must have reinforcement_count=2 (one per session)"
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
                    speaker_id="speaker0",
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
# session_ids provenance union in _upsert_relation
# Relation.session_ids survives the merge as a SET union.
# ---------------------------------------------------------------------------


class TestSessionIdsProvenanceUnion:
    """Acceptance tests for session_ids provenance plumbing.

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
    def _rel(session_ids: list[str], speaker_id: str = "speaker0") -> Relation:
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

        Speaker entities are keyed by speaker_id in the graph ("speaker0"), not
        by the display name ("Alex"); non-speaker entities are keyed by canonical
        form ("berlin").  Edge lookup uses these canonical keys.
        """
        m = GraphMerger()
        s1 = SessionGraph(
            session_id="__interim_pending_sessions__",  # synthetic sentinel
            timestamp="2026-01-01T00:00:00Z",
            entities=[
                Entity(name="Alex", entity_type="person", speaker_id="speaker0"),
                Entity(name="Berlin", entity_type="place"),
            ],
            relations=[self._rel(["real-session-abc"])],
        )
        m.merge(s1)

        # Speaker entities are keyed by the casefolded speaker_id ("speaker0").
        # Relation subject="Alex" with speaker_id="speaker0" → node key "speaker0".
        # Relation object="Berlin" → canonical node key "berlin".
        edges = list(m.graph["speaker0"]["berlin"].values())
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
                Entity(name="Alex", entity_type="person", speaker_id="speaker0"),
                Entity(name="Berlin", entity_type="place"),
            ],
            relations=[self._rel([])],
        )
        m.merge(s1)

        edges = list(m.graph["speaker0"]["berlin"].values())
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
            Entity(name="Alex", entity_type="person", speaker_id="speaker0"),
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

        # Speaker entity keyed by casefolded speaker_id "speaker0"; object "Berlin" → "berlin".
        edges = list(m.graph["speaker0"]["berlin"].values())
        assert len(edges) == 1, f"Duplicate SPO must collapse to one edge; got {len(edges)}"
        sessions = set(edges[0]["sessions"])
        assert "session-A" in sessions, f"session-A missing from {sessions}"
        assert "session-B" in sessions, f"session-B missing from {sessions}"
        assert "__interim_pending_sessions__" in sessions, (
            f"Synthetic sentinel missing from {sessions}"
        )

    def test_speaker_id_attribution_unchanged_by_session_ids_union(self):
        """dcf4189 invariant: speaker_id on the SUBJECT NODE is unchanged by the session_ids union.

        Multi-session edge: two merges under different scalar session_ids each
        carrying a real session_id.  The subject node must retain speaker_id
        from the entity (dcf4189), not from any session_id field.
        """
        m = GraphMerger()
        entities = [
            Entity(name="Alex", entity_type="person", speaker_id="speaker0"),
            Entity(name="Berlin", entity_type="place"),
        ]
        s1 = SessionGraph(
            session_id="scalar-1",
            timestamp="2026-01-01T00:00:00Z",
            entities=entities,
            relations=[self._rel(["session-A"], speaker_id="speaker0")],
        )
        m.merge(s1)

        s2 = SessionGraph(
            session_id="scalar-2",
            timestamp="2026-01-01T01:00:00Z",
            entities=entities,
            relations=[self._rel(["session-B"], speaker_id="speaker0")],
        )
        m.merge(s2)

        # Speaker entities are keyed by the casefolded speaker_id ("speaker0").
        # session_ids union touches only edge['sessions'] — never the node.
        speaker_node = m.graph.nodes.get("speaker0", {})
        assert speaker_node.get("speaker_id") == "speaker0", (
            "speaker_id attribute on the subject node must come from the Entity, not from "
            f"session_ids union; got {speaker_node.get('speaker_id')!r}"
        )
        # And the edge carries both real session ids.
        edges = list(m.graph["speaker0"]["berlin"].values())
        sessions = set(edges[0]["sessions"])
        assert "session-A" in sessions and "session-B" in sessions, (
            f"Both real session ids must be in edge['sessions']; got {sessions}"
        )


# ---------------------------------------------------------------------------
# Merger edge-stamp tests — A-1/A-2/B-4 speaker_id and edge_source stamps
# ---------------------------------------------------------------------------


class TestMergerEdgeStamps:
    """A-1/A-2/B-4: merger stamps speaker_id and edge_source on inserted edges."""

    def test_case3_stamps_speaker_id_unconditionally(self):
        """A-1: Case-3 net-new insert stamps speaker_id from Relation unconditionally."""
        from paramem.graph.merger import GraphMerger
        from paramem.graph.schema import Relation

        m = GraphMerger()
        rel = Relation(
            subject="spk-1",
            predicate="works_at",
            object="acme",
            relation_type="factual",
            speaker_id="spk-1",
        )
        m._upsert_relation("spk-1", "acme", rel, "s1", "2026-01-01T00:00:00Z")

        edges = list(m.graph["spk-1"]["acme"].values())
        assert len(edges) == 1
        assert edges[0].get("speaker_id") == "spk-1", (
            f"A-1: Case-3 must stamp speaker_id; got {edges[0].get('speaker_id')!r}"
        )

    def test_case3_stamps_edge_source_when_set(self):
        """B-4: Case-3 stamps _EDGE_SOURCE_ATTR when Relation.edge_source is non-empty."""
        from paramem.graph.merger import GraphMerger
        from paramem.graph.schema import Relation
        from paramem.memory.persistence import _EDGE_SOURCE_ATTR

        m = GraphMerger()
        rel = Relation(
            subject="alice",
            predicate="knows",
            object="bob",
            relation_type="social",
            speaker_id="alice",
            edge_source="graph_enrichment",
        )
        m._upsert_relation("alice", "bob", rel, "enrich", "2026-01-01T00:00:00Z")

        edges = list(m.graph["alice"]["bob"].values())
        assert edges[0].get(_EDGE_SOURCE_ATTR) == "graph_enrichment", (
            f"B-4: Case-3 must stamp edge_source; got {edges[0].get(_EDGE_SOURCE_ATTR)!r}"
        )

    def test_case1_adopts_speaker_id_if_empty(self):
        """A-2: Case-1 reinforce adopts speaker_id onto a keyless-speaker edge."""
        from paramem.graph.merger import GraphMerger
        from paramem.graph.schema import Relation

        m = GraphMerger()
        m.graph.add_node("alice")
        m.graph.add_node("berlin")
        eid = m.graph.add_edge(
            "alice",
            "berlin",
            predicate="lives in",
            relation_type="factual",
            confidence=1.0,
            first_seen="s0",
            last_seen="s0",
            reinforcement_count=1,
            sessions=["s0"],
        )
        # Existing edge has no speaker_id.
        assert m.graph["alice"]["berlin"][eid].get("speaker_id") is None

        incoming = Relation(
            subject="alice",
            predicate="lives_in",
            object="berlin",
            relation_type="factual",
            speaker_id="speaker0",
        )
        m._upsert_relation("alice", "berlin", incoming, "s1", "2026-01-01T00:00:00Z")

        # A-2: speaker_id adopted onto the existing edge.
        edge = m.graph["alice"]["berlin"][eid]
        assert edge.get("speaker_id") == "speaker0", (
            f"A-2: Case-1 must adopt speaker_id; got {edge.get('speaker_id')!r}"
        )

    def test_case1_does_not_overwrite_existing_speaker_id(self):
        """A-2: no-overwrite — existing non-empty speaker_id is preserved."""
        from paramem.graph.merger import GraphMerger
        from paramem.graph.schema import Relation

        m = GraphMerger()
        m.graph.add_node("alice")
        m.graph.add_node("berlin")
        eid = m.graph.add_edge(
            "alice",
            "berlin",
            predicate="lives in",
            relation_type="factual",
            confidence=1.0,
            first_seen="s0",
            last_seen="s0",
            reinforcement_count=1,
            sessions=["s0"],
            speaker_id="OriginalSpeaker",
        )

        incoming = Relation(
            subject="alice",
            predicate="lives_in",
            object="berlin",
            relation_type="factual",
            speaker_id="DifferentSpeaker",
        )
        m._upsert_relation("alice", "berlin", incoming, "s1", "2026-01-01T00:00:00Z")

        edge = m.graph["alice"]["berlin"][eid]
        assert edge.get("speaker_id") == "OriginalSpeaker", (
            f"A-2: existing speaker_id must NOT be overwritten; got {edge.get('speaker_id')!r}"
        )

    def test_e2_symmetric_collapses_non_speaker_pair(self):
        """E-2: symmetric=True, subject > obj, neither is a speaker → swap applied."""
        from paramem.graph.merger import GraphMerger
        from paramem.graph.schema import Relation

        m = GraphMerger()
        # "z_node" > "a_node" lexicographically.
        rel = Relation(
            subject="z_node",
            predicate="colleague_of",
            object="a_node",
            relation_type="social",
            speaker_id="",
            symmetric=True,
        )
        m._upsert_relation("z_node", "a_node", rel, "s1", "2026-01-01T00:00:00Z")

        # After swap, edge must be (a_node, z_node).
        assert m.graph.has_edge("a_node", "z_node"), (
            "E-2 swap must reorder to (a_node, z_node) because 'z_node' > 'a_node'"
        )
        assert not m.graph.has_edge("z_node", "a_node"), (
            "Original direction (z_node, a_node) must not exist after E-2 swap"
        )

    def test_e2_symmetric_does_not_swap_speaker_speaker_pair(self):
        """E-2 guard: both endpoints are speakers → no swap, both directions kept."""
        from paramem.graph.merger import GraphMerger
        from paramem.graph.schema import Relation

        m = GraphMerger()
        # Both nodes carry speaker_id.
        m.graph.add_node("z_speaker", speaker_id="z_speaker")
        m.graph.add_node("a_speaker", speaker_id="a_speaker")

        rel_z_a = Relation(
            subject="z_speaker",
            predicate="colleague_of",
            object="a_speaker",
            relation_type="social",
            speaker_id="z_speaker",
            symmetric=True,
        )
        rel_a_z = Relation(
            subject="a_speaker",
            predicate="colleague_of",
            object="z_speaker",
            relation_type="social",
            speaker_id="a_speaker",
            symmetric=True,
        )
        m._upsert_relation("z_speaker", "a_speaker", rel_z_a, "s1", "2026-01-01T00:00:00Z")
        m._upsert_relation("a_speaker", "z_speaker", rel_a_z, "s2", "2026-01-01T00:00:00Z")

        # Speaker↔speaker: both directions must survive.
        assert m.graph.has_edge("z_speaker", "a_speaker"), (
            "Speaker→speaker edge must survive (no E-2 swap for speaker pairs)"
        )
        assert m.graph.has_edge("a_speaker", "z_speaker"), (
            "Speaker←speaker edge must survive (no E-2 swap for speaker pairs)"
        )


# ---------------------------------------------------------------------------
# Casing-collision regression test (Step 2 — headline bug)
# ---------------------------------------------------------------------------


class TestSpeakerCasingCollisionRegression:
    """Regression: a speaker0 Entity and a speaker0 relation-endpoint that
    arrives WITHOUT a matching Entity must resolve to exactly ONE node key.

    Under lowercase-uniform identity both the entity path and the fallback
    path produce the same lowercase node key (entity.speaker_id verbatim).
    The collision is structurally impossible because the ingest safety-net
    guarantees all speaker tokens arrive as lowercase speaker{N}.
    """

    def test_entity_and_fallback_endpoint_resolve_to_one_node(self):
        """Merge a session where the speaker arrives as an Entity (speaker0) and
        ALSO as a relation subject without a matching Entity entry.

        The merger must produce exactly ONE node for speaker0 regardless of
        which path resolved it — entity path returns entity.speaker_id verbatim,
        fallback path passes the lowercase token through unchanged.
        """
        merger = GraphMerger(similarity_threshold=85.0)

        # The session has a speaker0 Entity and a relation whose subject is
        # "speaker0" but references a separate entity (SomeOrg) that has no
        # Entity declaration.  The "speaker0" subject on the relation thus goes
        # through the fallback resolution path.
        session = SessionGraph(
            session_id="s_casing",
            timestamp="2026-06-24T10:00:00Z",
            entities=[
                # speaker0 arrives as a full Entity (lowercase — ingest safety-net).
                Entity(
                    name="speaker0",
                    entity_type="person",
                    speaker_id="speaker0",
                ),
            ],
            relations=[
                # Relation whose subject IS "speaker0" — already in entity_name_map
                # from the entity above, so it resolves via the entity path.
                Relation(
                    subject="speaker0",
                    predicate="works at",
                    object="Acme",
                    relation_type="factual",
                    speaker_id="speaker0",
                ),
            ],
        )
        merger.merge(session)

        # Now merge a second session that has "speaker0" as a relation SUBJECT
        # with NO corresponding Entity — this exercises the fallback path.
        session2 = SessionGraph(
            session_id="s_casing2",
            timestamp="2026-06-24T11:00:00Z",
            entities=[],  # Intentionally NO Entity for speaker0
            relations=[
                Relation(
                    subject="speaker0",
                    predicate="lives in",
                    object="Berlin",
                    relation_type="factual",
                    speaker_id="speaker0",
                ),
            ],
        )
        merger.merge(session2)

        # Both sessions must resolve to the SAME lowercase node key.
        speaker_nodes = [n for n in merger.graph.nodes if n == "speaker0"]
        assert len(speaker_nodes) == 1, (
            f"Expected exactly one speaker0 node key, "
            f"got {speaker_nodes!r} — casing-collision dup not eliminated."
        )
        assert speaker_nodes[0] == "speaker0", (
            f"Speaker node key must be lowercase 'speaker0', got {speaker_nodes[0]!r}"
        )

        # Under lowercase-uniform identity, attributes["name"] is also lowercase speaker0.
        node_data = merger.graph.nodes["speaker0"]
        assert node_data.get("attributes", {}).get("name") == "speaker0", (
            "attributes['name'] must be lowercase 'speaker0' under lowercase-uniform identity"
        )


# ---------------------------------------------------------------------------
# Id-based stamp → merger integration: speaker_id attribute + both_speakers
# ---------------------------------------------------------------------------


class TestStampedSpeakerMergerIntegration:
    """Verify that an entity stamped by _stamp_speaker_entity (id-based)
    produces the correct merger node state and enables the both_speakers guard
    on symmetric relations.
    """

    def _make_stamped_session(
        self,
        session_id: str,
        speaker_id: str,
        place: str,
    ) -> "SessionGraph":
        """Build a SessionGraph where the speaker entity is already stamped
        (entity.speaker_id set, entity.name == speaker_id as the model emits)."""
        return SessionGraph(
            session_id=session_id,
            timestamp="2026-06-24T10:00:00Z",
            entities=[
                Entity(
                    name=speaker_id,
                    entity_type="person",
                    speaker_id=speaker_id,  # stamped by _stamp_speaker_entity
                ),
                Entity(name=place, entity_type="place"),
            ],
            relations=[
                Relation(
                    subject=speaker_id,
                    predicate="lives in",
                    object=place,
                    relation_type="factual",
                    speaker_id=speaker_id,
                ),
            ],
        )

    def test_stamped_entity_produces_correct_node_attributes(self):
        """After merging a stamped-speaker session, the graph node for the
        speaker must carry speaker_id attribute (cased) and entity_type person."""
        m = GraphMerger(similarity_threshold=85.0)
        session = self._make_stamped_session("s001", "speaker0", "Berlin")
        m.merge(session)

        node_key = "speaker0"
        assert node_key in m.graph, f"Expected node '{node_key}' in graph"
        node_data = m.graph.nodes[node_key]
        assert node_data.get("speaker_id") == "speaker0", (
            f"speaker_id attribute must be cased 'speaker0', got {node_data.get('speaker_id')!r}"
        )
        assert node_data.get("entity_type") == "person", (
            f"entity_type must be 'person', got {node_data.get('entity_type')!r}"
        )

    def test_both_speakers_guard_works_for_two_stamped_speakers(self):
        """Two speaker endpoints (both stamped) on a symmetric relation must
        produce two distinct directed edges — the both_speakers guard prevents
        the E-2 canonical swap that would collapse them."""
        m = GraphMerger(similarity_threshold=85.0)
        m.merge(self._make_stamped_session("s001", "speaker0", "Berlin"))
        m.merge(self._make_stamped_session("s002", "speaker1", "Munich"))

        # Add a symmetric social relation between the two speakers.
        rel_0_to_1 = Relation(
            subject="speaker0",
            predicate="colleague of",
            object="speaker1",
            relation_type="social",
            speaker_id="speaker0",
            symmetric=True,
        )
        rel_1_to_0 = Relation(
            subject="speaker1",
            predicate="colleague of",
            object="speaker0",
            relation_type="social",
            speaker_id="speaker1",
            symmetric=True,
        )
        m._upsert_relation("speaker0", "speaker1", rel_0_to_1, "s3", "2026-06-24T11:00:00Z")
        m._upsert_relation("speaker1", "speaker0", rel_1_to_0, "s4", "2026-06-24T12:00:00Z")

        assert m.graph.has_edge("speaker0", "speaker1"), (
            "speaker0→speaker1 edge must survive (both_speakers guard active)"
        )
        assert m.graph.has_edge("speaker1", "speaker0"), (
            "speaker1→speaker0 edge must survive (both_speakers guard active)"
        )


class TestLastSeenTimestampFlow:
    """_upsert_relation must maintain the freshest last_seen across reinforcements.

    INVARIANT: whenever edges collapse into a survivor the survivor's
    ``last_seen`` = the FRESHEST (max) timestamp across all affected edges.
    ISO-8601 UTC strings compare lexicographically = chronologically; ``""``
    sorts before any real timestamp so ``max("", real) == real``.

    Rules:

    - Case-1 (reinforce): ``edge["last_seen"] = max(existing, carry-or-timestamp)``
      — never regress; advance to the freshest value.
    - Case-3 (new edge): ``self.graph[...]["last_seen"] = relation.last_seen or
      timestamp`` — net-new edge, nothing to max against yet.

    No ``datetime.now()`` is ever fabricated: only the real
    ``session_graph.timestamp`` at ingest or the carry-slot populated by
    ``_build_registry_true_relations`` reaches the edge.
    """

    def _make_rel(self, last_seen: str = "") -> Relation:
        return Relation(
            subject="alice",
            predicate="lives in",
            object="berlin",
            relation_type="factual",
            speaker_id="speaker0",
            last_seen=last_seen,
        )

    def _first_edge_data(self, m: "GraphMerger", subject: str, obj: str) -> dict:
        edges = list(m.graph[subject][obj].values())
        assert edges, f"No edge found between {subject!r} and {obj!r}"
        return edges[0]

    def test_new_edge_takes_timestamp_when_relation_last_seen_empty(self):
        """Case-3: new edge gets last_seen from the session timestamp param."""
        m = GraphMerger(similarity_threshold=85.0)
        rel = self._make_rel(last_seen="")
        m._upsert_relation("alice", "berlin", rel, "s001", "2026-06-01T12:00:00Z")

        data = self._first_edge_data(m, "alice", "berlin")
        assert data.get("last_seen") == "2026-06-01T12:00:00Z", (
            f"Case-3: expected last_seen from timestamp param; got {data.get('last_seen')!r}"
        )

    def test_new_edge_prefers_relation_last_seen_over_timestamp(self):
        """Case-3: when relation.last_seen is set it takes priority over timestamp."""
        m = GraphMerger(similarity_threshold=85.0)
        rel = self._make_rel(last_seen="2026-05-15T08:00:00Z")
        m._upsert_relation("alice", "berlin", rel, "s001", "2026-06-01T12:00:00Z")

        data = self._first_edge_data(m, "alice", "berlin")
        assert data.get("last_seen") == "2026-05-15T08:00:00Z", (
            "Case-3: relation.last_seen must win over timestamp param; "
            f"got {data.get('last_seen')!r}"
        )

    def test_reinforce_updates_last_seen_to_newer_session(self):
        """Case-1: re-inserting an identical triple updates last_seen to the new timestamp."""
        m = GraphMerger(similarity_threshold=85.0)
        rel = self._make_rel(last_seen="")
        # First insertion — edge created with initial timestamp.
        m._upsert_relation("alice", "berlin", rel, "s001", "2026-06-01T12:00:00Z")
        # Second identical insertion (same predicate/object) — should reinforce.
        m._upsert_relation("alice", "berlin", rel, "s002", "2026-06-10T09:00:00Z")

        data = self._first_edge_data(m, "alice", "berlin")
        assert data.get("last_seen") == "2026-06-10T09:00:00Z", (
            "Case-1: last_seen must advance to the newer session timestamp after reinforce; "
            f"got {data.get('last_seen')!r}"
        )

    def test_reinforce_does_not_regress_last_seen(self):
        """Case-1: a re-merge with an OLDER carry-slot timestamp does NOT regress
        an existing newer last_seen on the edge."""
        m = GraphMerger(similarity_threshold=85.0)
        rel = self._make_rel(last_seen="")
        # First insertion sets a newer timestamp.
        m._upsert_relation("alice", "berlin", rel, "s001", "2026-06-20T12:00:00Z")
        # Re-merge (fold path) carries an OLDER value.
        older_carry = self._make_rel(last_seen="2026-05-01T08:00:00Z")
        m._upsert_relation("alice", "berlin", older_carry, "s-recon", "2026-05-01T08:00:00Z")

        data = self._first_edge_data(m, "alice", "berlin")
        assert data.get("last_seen") == "2026-06-20T12:00:00Z", (
            "Case-1: an older incoming timestamp must not regress an existing newer last_seen; "
            f"got {data.get('last_seen')!r}"
        )

    def test_reinforce_max_picks_freshest_over_existing(self):
        """Case-1: when the incoming timestamp is NEWER than the existing edge,
        last_seen advances to the newer value."""
        m = GraphMerger(similarity_threshold=85.0)
        rel = self._make_rel(last_seen="")
        # First insertion sets an older timestamp.
        m._upsert_relation("alice", "berlin", rel, "s001", "2026-05-01T08:00:00Z")
        # Re-merge carries a newer carry-slot (e.g. a later session's recon).
        newer_carry = self._make_rel(last_seen="2026-06-20T12:00:00Z")
        m._upsert_relation("alice", "berlin", newer_carry, "s-recon", "2026-05-01T08:00:00Z")

        data = self._first_edge_data(m, "alice", "berlin")
        assert data.get("last_seen") == "2026-06-20T12:00:00Z", (
            "Case-1: a newer carry-slot must advance last_seen on the existing edge; "
            f"got {data.get('last_seen')!r}"
        )
