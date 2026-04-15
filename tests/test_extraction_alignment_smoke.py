"""Extraction pipeline alignment smoke test — CPU-safe, all SOTA mocked.

Feeds a realistic conversational transcript through the full _sota_pipeline
with mocked SOTA enrichment and mocked SOTA plausibility. Verifies:
- No entity has entity_type == "person" unless it actually is a person.
- plausibility_dropped is recorded when the mock plausibility drops facts.
- fallback_path is None on the happy path.
- D6 regression: music/device/media entities are not typed "person".

See §7.11 of alignment-plan-2026-04-15.md.
"""

from unittest.mock import patch

from paramem.graph.extractor import _sota_pipeline
from paramem.graph.schema import Entity, SessionGraph


def _make_graph_from_spec(
    relations: list[tuple],
    entity_specs: list[tuple],
) -> SessionGraph:
    """Build a SessionGraph from (name, type) and (subj, pred, obj) specs."""
    entities = [Entity(name=n, entity_type=t) for n, t in entity_specs]
    from paramem.graph.schema import Relation

    rels = [
        Relation(
            subject=s,
            predicate=p,
            object=o,
            relation_type="factual",
            confidence=1.0,
        )
        for s, p, o in relations
    ]
    return SessionGraph(
        session_id="smoke_test",
        timestamp="2026-04-15T00:00:00Z",
        entities=entities,
        relations=rels,
    )


# Simulated transcript representative of a home-assistant conversation
_SMOKE_TRANSCRIPT = """\
User: Good morning! I am calling from the office in Kelkheim/Taunus.
Assistant: Good morning! How can I help you today?
User: Please turn on the Office Speaker and play HR3 from Royal Radio.
Assistant: Playing HR3 on the Office Speaker.
User: And also put on Uptown Funk by Bruno Mars if you can.
Assistant: Sure, queuing Uptown Funk on the Office Speaker.
"""

# Known-bad facts that a plausibility filter should drop (role leaks, tautologies)
_KNOWN_BAD_PREDICATES = {"listens_to", "has_role", "discussed_topic"}


class TestAlignmentSmoke:
    """CPU-safe smoke test: all SOTA calls mocked, pipeline logic runs for real."""

    def _build_smoke_graph(self) -> tuple[SessionGraph, list[dict], dict, str]:
        """Build a representative graph + anonymization mock return values."""
        graph = _make_graph_from_spec(
            relations=[
                ("Tobias", "located_at", "Kelkheim/Taunus"),
                ("Tobias", "has_role", "User"),  # role leak — should be dropped by plausibility
                ("Office Speaker", "plays", "HR3"),
                ("Tobias", "listens_to", "Music"),  # noise — plausibility should drop
                ("Tobias", "requested", "Uptown Funk"),
            ],
            entity_specs=[
                ("Tobias", "person"),
                ("Kelkheim/Taunus", "place"),
                ("Office Speaker", "concept"),
                ("HR3", "concept"),
                ("Royal Radio", "organization"),
                ("Music", "concept"),
                ("Uptown Funk", "concept"),
            ],
        )
        anon_facts = [
            {"subject": "Person_1", "predicate": "located_at", "object": "City_1"},
            {"subject": "Person_1", "predicate": "has_role", "object": "User"},
            {"subject": "Thing_1", "predicate": "plays", "object": "Thing_2"},
            {"subject": "Person_1", "predicate": "listens_to", "object": "Thing_3"},
            {"subject": "Person_1", "predicate": "requested", "object": "Thing_4"},
        ]
        mapping = {
            "Tobias": "Person_1",
            "Kelkheim/Taunus": "City_1",
            "Office Speaker": "Thing_1",
            "HR3": "Thing_2",
            "Music": "Thing_3",
            "Uptown Funk": "Thing_4",
        }
        anon_transcript = (
            "Person_1 is calling from City_1. Please turn on Thing_1 and play Thing_2 "
            "from Org_1. Playing Thing_2 on Thing_1. Queuing Thing_4 on Thing_1."
        )
        return graph, anon_facts, mapping, anon_transcript

    def test_entity_types_not_stamped_person(self):
        """Non-person entities (Office Speaker, Music, HR3) must not be typed 'person' (D6)."""
        graph, anon_facts, mapping, anon_transcript = self._build_smoke_graph()

        # Plausibility filter drops the known-bad predicates
        def fake_plaus_filter(facts, transcript, model, tokenizer, **kwargs):
            return [f for f in facts if f.get("predicate") not in _KNOWN_BAD_PREDICATES]

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.extractor._anonymize_with_local_model",
                return_value=(anon_facts, mapping, anon_transcript),
            ),
            patch(
                "paramem.graph.extractor._filter_with_sota",
                return_value=(anon_facts, None, None),
            ),
            patch(
                "paramem.graph.extractor._local_plausibility_filter",
                side_effect=fake_plaus_filter,
            ),
        ):
            from unittest.mock import MagicMock

            result = _sota_pipeline(
                graph,
                _SMOKE_TRANSCRIPT,
                MagicMock(),
                MagicMock(),
                plausibility_judge="auto",
                plausibility_stage="deanon",
            )

        entity_map = {e.name: e.entity_type for e in result.entities}

        # Tobias is the only actual person
        for name, etype in entity_map.items():
            if name == "Tobias":
                assert etype == "person", f"Tobias must be 'person', got {etype!r}"
            elif name in ("Kelkheim/Taunus",):
                assert etype in ("place", "location"), (
                    f"{name!r} must be place/location, got {etype!r}"
                )
            else:
                # Music, Office Speaker, Royal Radio, HR3, Uptown Funk, etc.
                assert etype != "person", f"{name!r} must NOT be typed 'person'; got {etype!r}"

    def test_plausibility_dropped_recorded(self):
        """plausibility_dropped is > 0 when the mock plausibility drops facts."""
        graph, anon_facts, mapping, anon_transcript = self._build_smoke_graph()

        def fake_plaus_filter(facts, transcript, model, tokenizer, **kwargs):
            # Drop 2 known-bad facts
            return [f for f in facts if f.get("predicate") not in _KNOWN_BAD_PREDICATES]

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.extractor._anonymize_with_local_model",
                return_value=(anon_facts, mapping, anon_transcript),
            ),
            patch(
                "paramem.graph.extractor._filter_with_sota",
                return_value=(anon_facts, None, None),
            ),
            patch(
                "paramem.graph.extractor._local_plausibility_filter",
                side_effect=fake_plaus_filter,
            ),
        ):
            from unittest.mock import MagicMock

            result = _sota_pipeline(
                graph,
                _SMOKE_TRANSCRIPT,
                MagicMock(),
                MagicMock(),
                plausibility_judge="auto",
                plausibility_stage="deanon",
            )

        dropped = result.diagnostics.get("plausibility_dropped", 0)
        assert dropped > 0, f"plausibility_dropped must be > 0 when mock drops facts; got {dropped}"

    def test_no_fallback_path_on_happy_path(self):
        """On the happy path (no failures), fallback_path must not be set."""
        graph, anon_facts, mapping, anon_transcript = self._build_smoke_graph()

        with (
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}),
            patch(
                "paramem.graph.extractor._anonymize_with_local_model",
                return_value=(anon_facts, mapping, anon_transcript),
            ),
            patch(
                "paramem.graph.extractor._filter_with_sota",
                return_value=(anon_facts, None, None),
            ),
        ):
            result = _sota_pipeline(
                graph,
                _SMOKE_TRANSCRIPT,
                None,
                None,
                plausibility_judge="off",
            )

        assert result.diagnostics.get("fallback_path") is None, (
            f"No fallback on happy path; got {result.diagnostics.get('fallback_path')!r}"
        )
