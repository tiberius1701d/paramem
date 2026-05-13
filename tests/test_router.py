"""Unit tests for QueryRouter — dual-graph routing and entity indexing."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from paramem.server.router import (
    MAX_KEYS_PER_QUERY,
    QueryRouter,
    _interim_sort_key,
    _is_interrogative,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_keyed_pairs(directory: Path, pairs: list[dict], *, subdir: str | None = None) -> None:
    """Write a keyed_pairs.json file into directory (or directory/subdir/)."""
    target = directory / subdir if subdir else directory
    target.mkdir(parents=True, exist_ok=True)
    (target / "keyed_pairs.json").write_text(json.dumps(pairs))


def _make_pair(
    key: str,
    subject: str,
    obj: str,
    *,
    speaker_id: str = "",
    question: str = "Q?",
    answer: str = "A.",
    source_predicate: str = "related_to",
    first_seen_cycle: int = 1,
) -> dict:
    """Return a full-schema keyed pair for use in router tests.

    All eight canonical fields are included so ``read_keyed_pairs`` schema
    validation passes when the router loads the fixture file.
    """
    return {
        "key": key,
        "question": question,
        "answer": answer,
        "source_subject": subject,
        "source_predicate": source_predicate,
        "source_object": obj,
        "speaker_id": speaker_id,
        "first_seen_cycle": first_seen_cycle,
    }


def _make_ha_match(
    *,
    entities: list[str] | None = None,
    areas: list[str] | None = None,
    verbs: list[str] | None = None,
    domains: list[str] | None = None,
) -> MagicMock:
    """Return a mock HAMatchResult."""
    m = MagicMock()
    m.matched_entities = entities or []
    m.matched_areas = areas or []
    m.matched_verbs = verbs or []
    m.domains = domains or []
    m.has_entity_match = bool(entities or areas)
    m.has_verb_match = bool(verbs)
    return m


def _make_ha_graph(match_result: MagicMock | None) -> MagicMock:
    """Return a mock HAEntityGraph whose .match() returns match_result."""
    g = MagicMock()
    g.match.return_value = match_result
    return g


# ---------------------------------------------------------------------------
# _is_interrogative
# ---------------------------------------------------------------------------


class TestIsInterrogative:
    @pytest.mark.parametrize(
        "text",
        [
            "What is the temperature?",
            "How many lights are on?",
            "Where is my phone?",
            "Who is at the door?",
            "Why is the alarm going off?",
            "When did I last water the plants?",
            "Is the door locked?",
            "Are the lights off?",
            "Does the alarm work?",
            "Do you know my name?",
            "Can you turn off the light?",
            "Could you help me?",
            "Would you remind me?",
            "Will it rain today?",
            "Which light is on?",
        ],
    )
    def test_question_words_are_interrogative(self, text: str):
        assert _is_interrogative(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "Turn on the kitchen light",
            "Set the thermostat to 20 degrees",
            "Play music in the living room",
            "Lock the front door",
            "Remind me at 8 AM",
        ],
    )
    def test_imperatives_are_not_interrogative(self, text: str):
        assert _is_interrogative(text) is False

    def test_empty_string_is_not_interrogative(self):
        assert _is_interrogative("") is False

    def test_single_word_question(self):
        assert _is_interrogative("What") is True

    def test_single_word_imperative(self):
        assert _is_interrogative("Turn") is False

    @pytest.mark.parametrize(
        "text",
        [
            # German — the configs/intents/personal.de.txt cohort that the
            # legacy English-only first-word lookup missed entirely.
            "Wo wohne ich?",
            "Was ist meine Adresse?",
            "Wann habe ich Geburtstag?",
            # Spanish — opening punctuation present, closing handles it.
            "¿Dónde vivo?",
            # Mandarin — fullwidth question mark.
            "我住在哪里？",
            # Arabic — RTL question mark glyph.
            "أين أعيش؟",
            # English declarative shaped as a question by punctuation.
            "Berlin is in Germany?",
            # Whitespace tolerance — trailing space before the mark.
            "Where do I live? ",
            # Whitespace tolerance — newline between question and mark.
            "Where do I live\n?",
        ],
    )
    def test_terminal_question_mark_is_interrogative(self, text: str):
        """Layer 1 — punctuation-based detection works language-agnostically.

        Locks the contract that any text ending in ``?`` / ``？`` / ``؟``
        classifies as interrogative, regardless of leading word.  Without
        this layer the German abstention path silently misfires (a
        privacy-relevant gap on personal queries).
        """
        assert _is_interrogative(text) is True

    def test_greek_semicolon_is_not_interrogative(self):
        """Greek uses ``;`` (U+003B) as a question mark, but the glyph is
        identical to the ASCII semicolon.  Treating it as interrogative
        would mis-classify any declarative sentence ending in ``;`` —
        not worth the trade-off for the rare Greek-language case.
        """
        assert _is_interrogative("Turn off the light;") is False

    def test_declarative_with_trailing_period_is_not_interrogative(self):
        """Layer 1 fires only on question marks; periods / exclamation
        marks fall through to the leading-word check (and miss it for
        non-English imperatives)."""
        assert _is_interrogative("Mache das Licht aus.") is False
        assert _is_interrogative("Schalte die Lampe an!") is False

    def test_encoder_tier_overrides_punctuation_when_decisive(self):
        """Tier 1 (encoder classifier) takes precedence over the
        deterministic tiers when ``config`` is provided AND the encoder
        produced a confident verdict.  Verifies the production path:
        ``classify_sentence_type`` returning ``INTERROGATIVE`` short-
        circuits and returns True regardless of leading word.
        """
        from paramem.server.config import SentenceTypeConfig
        from paramem.server.sentence_type import SentenceType

        cfg = SentenceTypeConfig()
        with patch(
            "paramem.server.sentence_type.classify_sentence_type",
            return_value=SentenceType.INTERROGATIVE,
        ) as mock_classify:
            # No leading wh-word, no terminal ?, but encoder says interrogative.
            assert _is_interrogative("the door locked", config=cfg) is True
        mock_classify.assert_called_once()

    def test_encoder_tier_returns_false_when_classifier_says_non(self):
        """Encoder verdict NON_INTERROGATIVE wins over the punctuation/
        lexicon tiers — even if the deterministic heuristic would
        flag the query."""
        from paramem.server.config import SentenceTypeConfig
        from paramem.server.sentence_type import SentenceType

        cfg = SentenceTypeConfig()
        with patch(
            "paramem.server.sentence_type.classify_sentence_type",
            return_value=SentenceType.NON_INTERROGATIVE,
        ):
            # Has terminal ? — would normally classify as interrogative.
            assert _is_interrogative("Berlin is in Germany?", config=cfg) is False

    def test_encoder_tier_falls_through_when_classifier_returns_none(self):
        """Encoder unavailable / margin not met → ``classify_sentence_type``
        returns ``None``; the deterministic tiers (punctuation +
        lexicon) take over.  Locks the fallback contract that the
        encoder's "I don't know" doesn't accidentally suppress
        question detection.
        """
        from paramem.server.config import SentenceTypeConfig

        cfg = SentenceTypeConfig()
        with patch(
            "paramem.server.sentence_type.classify_sentence_type",
            return_value=None,
        ):
            # No encoder verdict → punctuation tier catches the German question.
            assert _is_interrogative("Wo wohne ich?", config=cfg) is True
            # No encoder verdict, no terminal ?, English wh-word → lexicon catches it.
            assert _is_interrogative("What is my name", config=cfg) is True
            # No encoder verdict, no terminal ?, non-English imperative → False.
            assert _is_interrogative("Mache das Licht an", config=cfg) is False


# ---------------------------------------------------------------------------
# QueryRouter — loading / reload
# ---------------------------------------------------------------------------


class TestQueryRouterLoad:
    def test_empty_directory_loads_cleanly(self):
        with tempfile.TemporaryDirectory() as tmp:
            router = QueryRouter(adapter_dir=Path(tmp))
            assert router.entity_count == 0
            assert router.adapter_count == 0

    def test_nonexistent_directory_loads_cleanly(self):
        router = QueryRouter(adapter_dir=Path("/nonexistent/path/xyz"))
        assert router.entity_count == 0

    def test_keyed_pairs_at_root_infers_episodic(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_keyed_pairs(
                Path(tmp),
                [_make_pair("graph1", "Alice", "Berlin", speaker_id="alice")],
            )
            router = QueryRouter(adapter_dir=Path(tmp))
            assert "episodic" in router._entity_key_index

    def test_keyed_pairs_in_subdir_uses_subdir_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_keyed_pairs(
                Path(tmp),
                [_make_pair("graph1", "Alice", "Berlin", speaker_id="alice")],
                subdir="semantic",
            )
            router = QueryRouter(adapter_dir=Path(tmp))
            assert "semantic" in router._entity_key_index

    def test_subject_and_object_both_indexed(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_keyed_pairs(
                Path(tmp),
                [_make_pair("graph1", "Alice", "Berlin", speaker_id="alice")],
            )
            router = QueryRouter(adapter_dir=Path(tmp))
            assert "alice" in router._all_entities
            assert "berlin" in router._all_entities

    def test_speaker_id_indexed(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_keyed_pairs(
                Path(tmp),
                [_make_pair("graph1", "Alice", "Berlin", speaker_id="alice")],
            )
            router = QueryRouter(adapter_dir=Path(tmp))
            assert "alice" in router._speaker_key_index
            assert "graph1" in router._speaker_key_index["alice"]

    def test_pair_without_speaker_id_not_in_speaker_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_keyed_pairs(
                Path(tmp),
                [_make_pair("graph1", "Alice", "Berlin")],
            )
            router = QueryRouter(adapter_dir=Path(tmp))
            assert len(router._speaker_key_index) == 0

    def test_malformed_json_does_not_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "keyed_pairs.json").write_text("{not valid json}")
            router = QueryRouter(adapter_dir=Path(tmp))
            assert router.entity_count == 0

    def test_multiple_adapters_indexed(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_keyed_pairs(
                Path(tmp),
                [_make_pair("ep1", "Alice", "Berlin", speaker_id="alice")],
                subdir="episodic",
            )
            _write_keyed_pairs(
                Path(tmp),
                [_make_pair("se1", "Alice", "Python", speaker_id="alice")],
                subdir="semantic",
            )
            router = QueryRouter(adapter_dir=Path(tmp))
            assert router.adapter_count == 2
            assert "episodic" in router._entity_key_index
            assert "semantic" in router._entity_key_index

    def test_reload_replaces_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_keyed_pairs(
                Path(tmp),
                [_make_pair("graph1", "Alice", "Berlin", speaker_id="alice")],
            )
            router = QueryRouter(adapter_dir=Path(tmp))
            assert router.entity_count > 0

            # Remove file and reload — index must clear
            (Path(tmp) / "keyed_pairs.json").unlink()
            router.reload()
            assert router.entity_count == 0

    def test_entity_shorter_than_two_chars_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_keyed_pairs(
                Path(tmp),
                [_make_pair("graph1", "A", "Berlin", speaker_id="alice")],
            )
            router = QueryRouter(adapter_dir=Path(tmp))
            assert "a" not in router._all_entities
            assert "berlin" in router._all_entities


# ---------------------------------------------------------------------------
# QueryRouter — route(): PA path
# ---------------------------------------------------------------------------


class TestQueryRouterRoutePA:
    @staticmethod
    def _make_router(tmp: str) -> QueryRouter:
        _write_keyed_pairs(
            Path(tmp),
            [
                _make_pair("graph1", "Alice", "Berlin", speaker_id="alice"),
                _make_pair("graph2", "Alice", "Python", speaker_id="alice"),
            ],
        )
        return QueryRouter(adapter_dir=Path(tmp))

    def test_no_speaker_id_returns_no_steps(self):
        from paramem.server.router import Intent

        with tempfile.TemporaryDirectory() as tmp:
            router = self._make_router(tmp)
            plan = router.route("Where does Alice live?", speaker_id=None)
            assert plan.steps == []
            assert plan.intent == Intent.UNKNOWN

    def test_wrong_speaker_id_returns_no_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            router = self._make_router(tmp)
            plan = router.route("Where does Alice live?", speaker_id="bob")
            assert plan.steps == []

    def test_correct_speaker_id_pa_match(self):
        from paramem.server.router import Intent

        with tempfile.TemporaryDirectory() as tmp:
            router = self._make_router(tmp)
            plan = router.route("Where does Alice live?", speaker_id="alice")
            assert plan.steps  # PA match → steps populated
            assert plan.intent == Intent.PERSONAL

    def test_pa_match_steps_populated(self):
        with tempfile.TemporaryDirectory() as tmp:
            router = self._make_router(tmp)
            plan = router.route("Tell me about Alice", speaker_id="alice")
            assert plan.steps
            keys = [k for step in plan.steps for k in step.keys_to_probe]
            assert "graph1" in keys or "graph2" in keys

    def test_strategy_targeted_probe_when_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            router = self._make_router(tmp)
            plan = router.route("Alice lives in Berlin", speaker_id="alice")
            assert plan.strategy == "targeted_probe"

    def test_strategy_direct_when_no_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            router = self._make_router(tmp)
            # No entity in query → no steps
            plan = router.route("What is the capital of France?", speaker_id="alice")
            # "france" not in entity list → no steps
            assert plan.strategy == "direct"

    def test_speaker_name_injected_as_implicit_entity(self):
        from paramem.server.router import Intent

        with tempfile.TemporaryDirectory() as tmp:
            # Only "alice" in the index
            _write_keyed_pairs(
                Path(tmp),
                [_make_pair("graph1", "Alice", "Berlin", speaker_id="alice")],
            )
            router = QueryRouter(adapter_dir=Path(tmp))
            # Query doesn't mention alice, but speaker="Alice" injects it
            plan = router.route("Where do I live?", speaker="Alice", speaker_id="alice")
            # alice is now an implicit entity — should trigger PA match
            assert plan.intent == Intent.PERSONAL
            assert plan.steps

    def test_matched_entities_in_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            router = self._make_router(tmp)
            plan = router.route("Alice lives in Berlin", speaker_id="alice")
            assert "alice" in plan.matched_entities or "berlin" in plan.matched_entities

    def test_fuzzy_entity_match(self):
        from paramem.server.router import Intent

        with tempfile.TemporaryDirectory() as tmp:
            router = self._make_router(tmp)
            # "Alic" should fuzzy-match "alice" at ratio ~89
            plan = router.route("Tell me about Alic", speaker_id="alice")
            assert plan.intent == Intent.PERSONAL
            assert plan.steps

    def test_adapter_order_procedural_episodic_semantic(self):
        """Procedural step comes before episodic and semantic.

        CLAUDE.md inference assembly order: procedural (preferences) → episodic (recent)
        → semantic (consolidated). Procedural-first is load-bearing for personalization.
        When procedural is absent the remaining adapters still follow episodic → semantic.
        """
        with tempfile.TemporaryDirectory() as tmp:
            _write_keyed_pairs(
                Path(tmp),
                [_make_pair("ep1", "Alice", "Berlin", speaker_id="alice")],
                subdir="episodic",
            )
            _write_keyed_pairs(
                Path(tmp),
                [_make_pair("se1", "Alice", "Python", speaker_id="alice")],
                subdir="semantic",
            )
            router = QueryRouter(adapter_dir=Path(tmp))
            plan = router.route("Alice", speaker_id="alice")
            names = [s.adapter_name for s in plan.steps]
            # episodic before semantic (procedural absent in this fixture)
            if "episodic" in names and "semantic" in names:
                assert names.index("episodic") < names.index("semantic")


# ---------------------------------------------------------------------------
# QueryRouter — route(): HA path
# ---------------------------------------------------------------------------


class TestQueryRouterRouteHA:
    def test_no_ha_graph_no_ha_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            router = QueryRouter(adapter_dir=Path(tmp), ha_graph=None)
            plan = router.route("Turn on the kitchen light")
            assert plan.ha_domains == []
            assert plan.steps == []

    def test_ha_only_match(self):
        from paramem.server.router import Intent

        with tempfile.TemporaryDirectory() as tmp:
            ha = _make_ha_graph(_make_ha_match(entities=["kitchen light"], domains=["light"]))
            router = QueryRouter(adapter_dir=Path(tmp), ha_graph=ha)
            plan = router.route("Is the kitchen light on?")
            assert plan.intent == Intent.COMMAND
            assert plan.ha_domains == ["light"]
            assert plan.steps == []

    def test_verb_only_ha_match_no_entity_no_pa_steps(self):
        """Verb-only ha_match (no entity) does not produce PA steps and does
        not classify as PERSONAL.  ha_domains may still be populated from the
        match for observability."""
        from paramem.server.router import Intent

        with tempfile.TemporaryDirectory() as tmp:
            match = _make_ha_match(verbs=["turn on"], domains=["light"])
            ha = _make_ha_graph(match)
            router = QueryRouter(adapter_dir=Path(tmp), ha_graph=ha)
            plan = router.route("Is turn on working?")
            assert plan.steps == []
            assert plan.intent != Intent.PERSONAL

    def test_ha_domains_populated(self):
        with tempfile.TemporaryDirectory() as tmp:
            ha = _make_ha_graph(
                _make_ha_match(entities=["bedroom thermostat"], domains=["climate"])
            )
            router = QueryRouter(adapter_dir=Path(tmp), ha_graph=ha)
            plan = router.route("bedroom thermostat temperature")
            assert "climate" in plan.ha_domains

    def test_ha_domains_empty_when_no_ha_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            ha = _make_ha_graph(None)
            router = QueryRouter(adapter_dir=Path(tmp), ha_graph=ha)
            plan = router.route("Tell me something random")
            assert plan.ha_domains == []

    def test_pa_and_ha_overlap_classifies_personal(self):
        """When PA and HA both match the same query, intent=PERSONAL (PA wins
        in classify_intent's state-first dispatch).  PA steps and HA domains
        both populated so downstream consumers can still see both signals.
        """
        from paramem.server.router import Intent

        with tempfile.TemporaryDirectory() as tmp:
            _write_keyed_pairs(
                Path(tmp),
                [_make_pair("graph1", "Alice", "kitchen light", speaker_id="alice")],
            )
            ha = _make_ha_graph(_make_ha_match(entities=["kitchen light"], domains=["light"]))
            router = QueryRouter(adapter_dir=Path(tmp), ha_graph=ha)
            plan = router.route("Alice kitchen light", speaker_id="alice")
            assert plan.intent == Intent.PERSONAL
            assert plan.steps
            assert "light" in plan.ha_domains


# ---------------------------------------------------------------------------
# QueryRouter — MAX_KEYS_PER_QUERY cap
# ---------------------------------------------------------------------------


class TestQueryRouterMaxKeys:
    def test_keys_capped_at_max(self):
        """More than MAX_KEYS_PER_QUERY keys for one entity → capped in step."""
        with tempfile.TemporaryDirectory() as tmp:
            pairs = [
                _make_pair(f"graph{i}", "Alice", f"Entity{i}", speaker_id="alice")
                for i in range(MAX_KEYS_PER_QUERY + 5)
            ]
            _write_keyed_pairs(Path(tmp), pairs)
            router = QueryRouter(adapter_dir=Path(tmp))
            plan = router.route("Alice", speaker_id="alice")
            for step in plan.steps:
                assert len(step.keys_to_probe) <= MAX_KEYS_PER_QUERY


# ---------------------------------------------------------------------------
# _interim_sort_key helper
# ---------------------------------------------------------------------------


class TestInterimSortKey:
    """Unit tests for the private _interim_sort_key helper."""

    def test_valid_interim_name_returns_stamp(self):
        assert _interim_sort_key("episodic_interim_20260417T0000") == "20260417T0000"

    def test_main_adapter_returns_none(self):
        for name in ("episodic", "semantic", "procedural"):
            assert _interim_sort_key(name) is None

    def test_partial_prefix_returns_none(self):
        assert _interim_sort_key("episodic_interim_") is None

    def test_non_stamp_returns_none(self):
        assert _interim_sort_key("episodic_interim_today") is None

    def test_old_date_format_returns_none(self):
        # The old YYYY-MM-DD format is no longer valid for interim adapters.
        assert _interim_sort_key("episodic_interim_2026-04-17") is None

    def test_extra_suffix_returns_none(self):
        # Names with trailing text after the stamp are not valid interim names.
        assert _interim_sort_key("episodic_interim_20260417T0000_partial") is None


# ---------------------------------------------------------------------------
# QueryRouter — Step 4: multi-adapter interim routing
# ---------------------------------------------------------------------------


class TestInterimAdapterRouting:
    """Tests for Step 4 of multi-adapter interim routing.

    All fixtures write real keyed_pairs.json files to a tmp directory and
    construct a real QueryRouter — no mocking of internal state.
    """

    def test_route_groups_keys_by_adapter_directory(self):
        """Entity in both episodic (main) and an interim adapter → two steps, interim first.

        Freshness-wins: a user correction lands in the newest interim slot ahead
        of the next full-cycle merge into main, so probing interim before main
        ensures the latest answer wins.
        """
        with tempfile.TemporaryDirectory() as tmp:
            _write_keyed_pairs(
                Path(tmp),
                [_make_pair("ep1", "Alice", "Berlin", speaker_id="alice")],
                subdir="episodic",
            )
            _write_keyed_pairs(
                Path(tmp),
                [_make_pair("int1", "Alice", "Prague", speaker_id="alice")],
                subdir="episodic_interim_20260417T0000",
            )
            router = QueryRouter(adapter_dir=Path(tmp))
            plan = router.route("Alice", speaker_id="alice")

            names = [s.adapter_name for s in plan.steps]
            assert "episodic" in names, "main episodic step missing"
            assert "episodic_interim_20260417T0000" in names, "interim step missing"
            assert names.index("episodic_interim_20260417T0000") < names.index("episodic"), (
                "interim must precede main (freshness wins)"
            )

    def test_route_interim_adapter_ordering(self):
        """Multiple interim adapters: newest-first, all before main episodic."""
        with tempfile.TemporaryDirectory() as tmp:
            _write_keyed_pairs(
                Path(tmp),
                [_make_pair("ep1", "Alice", "Berlin", speaker_id="alice")],
                subdir="episodic",
            )
            for stamp in ("20260415T0000", "20260416T0000", "20260417T0000"):
                _write_keyed_pairs(
                    Path(tmp),
                    [_make_pair(f"int_{stamp}", "Alice", f"City_{stamp}", speaker_id="alice")],
                    subdir=f"episodic_interim_{stamp}",
                )
            router = QueryRouter(adapter_dir=Path(tmp))
            plan = router.route("Alice", speaker_id="alice")

            names = [s.adapter_name for s in plan.steps]
            interim_names = [n for n in names if n.startswith("episodic_interim_")]
            assert interim_names == [
                "episodic_interim_20260417T0000",
                "episodic_interim_20260416T0000",
                "episodic_interim_20260415T0000",
            ], f"Expected newest-first, got {interim_names}"
            # All interim slots must precede main episodic (freshness wins).
            assert names.index(interim_names[-1]) < names.index("episodic"), (
                f"Interim slots must precede main episodic, got {names}"
            )

    def test_route_speaker_scoping_applies_to_interim_keys(self):
        """Interim key tagged with 'alice' is invisible to speaker_id='bob'."""
        with tempfile.TemporaryDirectory() as tmp:
            # Interim adapter keyed to alice only
            _write_keyed_pairs(
                Path(tmp),
                [_make_pair("int1", "Alice", "Prague", speaker_id="alice")],
                subdir="episodic_interim_20260417T0000",
            )
            router = QueryRouter(adapter_dir=Path(tmp))
            # Bob queries — allowed_keys will be empty → no step emitted
            plan = router.route("Alice", speaker_id="bob")
            interim_names = [s.adapter_name for s in plan.steps if "interim" in s.adapter_name]
            assert interim_names == [], (
                "Interim step must not appear when speaker scoping filters all keys"
            )

    def test_route_no_phantom_step_for_deleted_interim_dir(self):
        """After reload() with the interim dir deleted, no step is emitted for it."""
        with tempfile.TemporaryDirectory() as tmp:
            _write_keyed_pairs(
                Path(tmp),
                [_make_pair("ep1", "Alice", "Berlin", speaker_id="alice")],
                subdir="episodic",
            )
            interim_dir = Path(tmp) / "episodic_interim_20260417T0000"
            _write_keyed_pairs(
                Path(tmp),
                [_make_pair("int1", "Alice", "Prague", speaker_id="alice")],
                subdir="episodic_interim_20260417T0000",
            )
            router = QueryRouter(adapter_dir=Path(tmp))
            # Confirm interim step present before deletion
            plan_before = router.route("Alice", speaker_id="alice")
            assert any("interim" in s.adapter_name for s in plan_before.steps), (
                "Interim step should be present before dir removal"
            )

            # Simulate Step 7: delete the interim dir, then reload
            import shutil

            shutil.rmtree(interim_dir)
            router.reload()

            plan_after = router.route("Alice", speaker_id="alice")
            interim_steps = [s for s in plan_after.steps if "interim" in s.adapter_name]
            assert interim_steps == [], (
                "No interim step should appear after reload() with interim dir deleted"
            )

    def test_route_interim_only_match_emits_interim_step(self):
        """Entity matched ONLY in an interim adapter → one step for that interim."""
        with tempfile.TemporaryDirectory() as tmp:
            # No main adapter keyed_pairs — only an interim subdir
            _write_keyed_pairs(
                Path(tmp),
                [_make_pair("int1", "Alice", "Vienna", speaker_id="alice")],
                subdir="episodic_interim_20260417T0000",
            )
            from paramem.server.router import Intent

            router = QueryRouter(adapter_dir=Path(tmp))
            plan = router.route("Alice", speaker_id="alice")

            assert len(plan.steps) == 1, f"Expected exactly one step, got {plan.steps}"
            assert plan.steps[0].adapter_name == "episodic_interim_20260417T0000"
            assert plan.intent == Intent.PERSONAL

    def test_route_three_mains_correct_order(self):
        """All three mains + one interim: order must be procedural, interim, episodic, semantic.

        Probe order: procedural (preferences shape style; load-bearing per
        feedback_router_procedural_first.md) → interim newest-first (freshest
        factual state) → main episodic (baseline factual snapshot) → semantic
        (durable corroboration).
        """
        with tempfile.TemporaryDirectory() as tmp:
            for adapter in ("episodic", "semantic", "procedural"):
                _write_keyed_pairs(
                    Path(tmp),
                    [_make_pair(f"{adapter}1", "Alice", f"Fact_{adapter}", speaker_id="alice")],
                    subdir=adapter,
                )
            _write_keyed_pairs(
                Path(tmp),
                [_make_pair("int1", "Alice", "InterimFact", speaker_id="alice")],
                subdir="episodic_interim_20260418T0000",
            )
            router = QueryRouter(adapter_dir=Path(tmp))
            plan = router.route("Alice", speaker_id="alice")

            names = [s.adapter_name for s in plan.steps]
            assert names[0] == "procedural", f"First step must be procedural, got {names}"
            assert names[1] == "episodic_interim_20260418T0000", (
                f"Second step must be the interim adapter, got {names}"
            )
            assert names[2] == "episodic", f"Third step must be episodic, got {names}"
            assert names[3] == "semantic", f"Fourth step must be semantic, got {names}"

    def test_route_freshness_wins_interim_before_main_for_same_key(self):
        """A key registered to BOTH interim and main is probed in interim first.

        This is the load-bearing freshness-wins guarantee: a user contradiction
        (move, rename, change of mind) lands in the newest interim slot before
        the next full-cycle merge collapses it into main. If the router probed
        main first, it would return the stale pre-update answer.
        """
        with tempfile.TemporaryDirectory() as tmp:
            # Both adapters claim to know about Alice — main says Berlin, interim
            # says Prague. Same entity name → same key resolution path.
            _write_keyed_pairs(
                Path(tmp),
                [_make_pair("alice_loc", "Alice", "Berlin", speaker_id="alice")],
                subdir="episodic",
            )
            _write_keyed_pairs(
                Path(tmp),
                [_make_pair("alice_loc", "Alice", "Prague", speaker_id="alice")],
                subdir="episodic_interim_20260418T0000",
            )
            router = QueryRouter(adapter_dir=Path(tmp))
            plan = router.route("Alice", speaker_id="alice")

            names = [s.adapter_name for s in plan.steps]
            assert "episodic_interim_20260418T0000" in names
            assert "episodic" in names
            assert names.index("episodic_interim_20260418T0000") < names.index("episodic"), (
                f"Interim must be probed before main for the same key (freshness wins), got {names}"
            )


class TestRouteIntentField:
    """G4 wiring: route() populates RoutingPlan.intent via classify_intent.

    These tests pin the contract between router state and the intent
    classifier without exercising the encoder — they cover the state-first
    tiers (PA hit → PERSONAL, HA hit → COMMAND) and the no-state degraded
    path (returns UNKNOWN when no IntentConfig is supplied)."""

    def test_pa_match_yields_personal_intent(self):
        from paramem.server.router import Intent

        with tempfile.TemporaryDirectory() as tmp:
            _write_keyed_pairs(
                Path(tmp),
                [_make_pair("alex_loc", "Alex", "Berlin", speaker_id="alex")],
                subdir="episodic",
            )
            router = QueryRouter(adapter_dir=Path(tmp))
            plan = router.route("Where does Alex live?", speaker_id="alex")

            assert plan.intent == Intent.PERSONAL
            assert plan.steps  # PA produced probe steps

    def test_ha_only_match_yields_command_intent(self):
        from paramem.server.router import Intent

        ha = _make_ha_graph(_make_ha_match(entities=["light.kitchen"], domains=["light"]))
        with tempfile.TemporaryDirectory() as tmp:
            router = QueryRouter(adapter_dir=Path(tmp), ha_graph=ha)
            plan = router.route("Turn on the kitchen light")

            assert plan.intent == Intent.COMMAND
            assert plan.ha_domains == ["light"]
            assert plan.steps == []

    def test_both_pa_and_ha_match_personal_wins(self):
        # Privacy-first: PA + HA overlap routes to PERSONAL so cloud
        # escalation stays gated on graph match.
        from paramem.server.router import Intent

        ha = _make_ha_graph(_make_ha_match(entities=["light.bedroom"], domains=["light"]))
        with tempfile.TemporaryDirectory() as tmp:
            _write_keyed_pairs(
                Path(tmp),
                [_make_pair("alex_room", "Alex", "bedroom", speaker_id="alex")],
                subdir="episodic",
            )
            router = QueryRouter(adapter_dir=Path(tmp), ha_graph=ha)
            plan = router.route("Is Alex's bedroom light on?", speaker_id="alex")

            assert plan.intent == Intent.PERSONAL
            assert plan.steps  # PA path also produces steps
            assert "light" in plan.ha_domains  # HA observability preserved

    def test_no_state_no_config_returns_unknown(self):
        # No PA, no HA, no IntentConfig — encoder isn't loaded so the
        # classifier degrades to UNKNOWN rather than raising.
        from paramem.server.router import Intent

        ha = _make_ha_graph(None)  # HA configured but no match
        with tempfile.TemporaryDirectory() as tmp:
            router = QueryRouter(adapter_dir=Path(tmp), ha_graph=ha)
            plan = router.route("What is the capital of France?")

            # match_source is "ha" because of imperative-fallback heuristic
            # (interrogative-aware), but intent classifier sees clean state
            # signals (has_pa=False, has_ha=False) and returns UNKNOWN
            # without an IntentConfig.
            assert plan.intent == Intent.UNKNOWN

    def test_no_state_with_config_returns_fail_closed(self):
        # No PA, no HA, IntentConfig present but encoder/exemplars unloaded
        # → fail-closed default (PERSONAL).  Encoder isn't loaded in unit
        # tests so this exercises the degraded path.
        from paramem.server.config import IntentConfig
        from paramem.server.router import Intent

        ha = _make_ha_graph(None)
        cfg = IntentConfig()  # fail_closed_intent="personal"
        with tempfile.TemporaryDirectory() as tmp:
            router = QueryRouter(adapter_dir=Path(tmp), ha_graph=ha, intent_config=cfg)
            plan = router.route("What is the capital of France?")

            assert plan.intent == Intent.PERSONAL


# ---------------------------------------------------------------------------
# QueryRouter — quad-format keyed_pairs.json (B1 regression guard)
# ---------------------------------------------------------------------------


def _make_quad_pair(
    key: str,
    subject: str,
    obj: str,
    *,
    speaker_id: str = "",
    predicate: str = "related_to",
    first_seen_cycle: int = 1,
) -> dict:
    """Return a 6-field quad-format keyed pair for router tests.

    The on-disk quad schema has no ``question``, ``answer``, or ``source_*``
    fields.  ``read_keyed_pairs_quad`` (the universal reader now used by
    ``_load_keyed_pairs``) accepts this format natively.
    """
    return {
        "key": key,
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "speaker_id": speaker_id,
        "first_seen_cycle": first_seen_cycle,
    }


class TestQueryRouterQuadFormat:
    """B1 regression guard: router loads quad-schema keyed_pairs.json files.

    Before the fix, ``_load_keyed_pairs`` called ``read_keyed_pairs`` which
    validates against the 8-field QA schema and raises ``ValueError`` on
    quad-format files.  The router caught the error, logged a warning, and
    left the entity-key index empty — silently breaking PA-match routing in
    quad mode.

    After the fix, ``_load_keyed_pairs`` calls ``read_keyed_pairs_quad``
    (the universal reader: native quad files accepted as-is; legacy QA files
    projected to quad shape) and reads ``"subject"``/``"object"`` instead of
    ``"source_subject"``/``"source_object"``.
    """

    def test_quad_file_loaded_without_error(self):
        """Router loads a 6-field quad keyed_pairs.json without raising."""
        with tempfile.TemporaryDirectory() as tmp:
            _write_keyed_pairs(
                Path(tmp),
                [_make_quad_pair("graph1", "Alice", "Berlin", speaker_id="alice")],
            )
            # Must not raise; entity_count must be positive.
            router = QueryRouter(adapter_dir=Path(tmp))
            assert router.entity_count > 0

    def test_quad_subject_and_object_indexed(self):
        """Both ``subject`` and ``object`` are indexed as entities from a quad file."""
        with tempfile.TemporaryDirectory() as tmp:
            _write_keyed_pairs(
                Path(tmp),
                [_make_quad_pair("graph1", "Alice", "Berlin", speaker_id="alice")],
            )
            router = QueryRouter(adapter_dir=Path(tmp))
            assert "alice" in router._all_entities
            assert "berlin" in router._all_entities

    def test_quad_speaker_id_indexed(self):
        """``speaker_id`` from a quad file is indexed into ``_speaker_key_index``."""
        with tempfile.TemporaryDirectory() as tmp:
            _write_keyed_pairs(
                Path(tmp),
                [_make_quad_pair("graph1", "Alice", "Berlin", speaker_id="alice")],
            )
            router = QueryRouter(adapter_dir=Path(tmp))
            assert "alice" in router._speaker_key_index
            assert "graph1" in router._speaker_key_index["alice"]

    def test_quad_pa_match_produces_steps(self):
        """A query mentioning the entity from a quad file resolves to a routing step."""
        from paramem.server.router import Intent

        with tempfile.TemporaryDirectory() as tmp:
            _write_keyed_pairs(
                Path(tmp),
                [_make_quad_pair("graph1", "Alice", "Berlin", speaker_id="alice")],
                subdir="episodic",
            )
            router = QueryRouter(adapter_dir=Path(tmp))
            plan = router.route("Where does Alice live?", speaker_id="alice")
            assert plan.steps, "Expected PA routing steps from quad keyed_pairs.json"
            assert plan.intent == Intent.PERSONAL
            keys = [k for step in plan.steps for k in step.keys_to_probe]
            assert "graph1" in keys

    def test_reload_with_quad_file_repopulates_index(self):
        """reload() after writing a quad keyed_pairs.json repopulates the entity index."""
        with tempfile.TemporaryDirectory() as tmp:
            router = QueryRouter(adapter_dir=Path(tmp))
            assert router.entity_count == 0

            _write_keyed_pairs(
                Path(tmp),
                [_make_quad_pair("graph1", "Alice", "Berlin", speaker_id="alice")],
            )
            router.reload()
            assert router.entity_count > 0
            assert "alice" in router._all_entities

    def test_legacy_qa_file_still_loads_via_universal_reader(self):
        """QA-format (8-field) files still load correctly after the B1 fix.

        ``read_keyed_pairs_quad`` projects them to quad shape, so existing
        QA-format deployments are unaffected.
        """
        with tempfile.TemporaryDirectory() as tmp:
            _write_keyed_pairs(
                Path(tmp),
                [_make_pair("graph1", "Alice", "Berlin", speaker_id="alice")],
            )
            router = QueryRouter(adapter_dir=Path(tmp))
            assert "alice" in router._all_entities
            assert "berlin" in router._all_entities
            assert "alice" in router._speaker_key_index

    def test_no_state_with_general_fail_closed(self):
        from paramem.server.config import IntentConfig
        from paramem.server.router import Intent

        ha = _make_ha_graph(None)
        cfg = IntentConfig(fail_closed_intent="general")
        with tempfile.TemporaryDirectory() as tmp:
            router = QueryRouter(adapter_dir=Path(tmp), ha_graph=ha, intent_config=cfg)
            plan = router.route("What is the capital of France?")

            assert plan.intent == Intent.GENERAL


# ---------------------------------------------------------------------------
# QueryRouter — attribute-predicate indexing (Bug 3 regression guard)
# ---------------------------------------------------------------------------


class TestAttributePredicateIndexing:
    """Bug 3 regression guard: has_<attr> predicates are indexed under their
    bare attribute name so attribute queries ("what is my email address?")
    resolve to the correct key regardless of MAX_KEYS_PER_QUERY.

    Before the fix, a query about "email" only matched via the speaker entity
    which could have 30+ associated keys — the has_email key was not guaranteed
    to survive the :data:`MAX_KEYS_PER_QUERY` cap.  After the fix the bare
    attribute name ("email", "phone", "linkedin") is also indexed so the
    attribute key is reachable directly.
    """

    def _make_has_email_pair(
        self,
        key: str = "graph2",
        speaker_id: str = "alice",
    ) -> dict:
        """Return a QA-format keyed pair whose source_predicate is has_email."""
        return {
            "key": key,
            "question": "What is Alice's email?",
            "answer": "alice@example.com",
            "source_subject": "Alice",
            "source_predicate": "has_email",
            "source_object": "alice@example.com",
            "speaker_id": speaker_id,
            "first_seen_cycle": 1,
        }

    def _make_has_email_quad(
        self,
        key: str = "graph2",
        speaker_id: str = "alice",
    ) -> dict:
        """Return a quad-format keyed pair whose predicate is has_email."""
        return {
            "key": key,
            "subject": "Alice",
            "predicate": "has_email",
            "object": "alice@example.com",
            "speaker_id": speaker_id,
            "first_seen_cycle": 1,
        }

    def test_has_email_predicate_indexed_as_email_qa(self):
        """QA-format: has_email key is indexed under 'email' entity."""
        with tempfile.TemporaryDirectory() as tmp:
            _write_keyed_pairs(Path(tmp), [self._make_has_email_pair()])
            router = QueryRouter(adapter_dir=Path(tmp))
            # "email" must appear in the entity list.
            assert "email" in router._all_entities

    def test_has_email_predicate_indexed_as_email_quad(self):
        """Quad-format: has_email key is indexed under 'email' entity."""
        with tempfile.TemporaryDirectory() as tmp:
            _write_keyed_pairs(Path(tmp), [self._make_has_email_quad()])
            router = QueryRouter(adapter_dir=Path(tmp))
            assert "email" in router._all_entities

    def test_has_email_key_reachable_via_email_attribute_index(self):
        """The has_email key maps to the 'email' attribute-index entry — deterministic.

        Whether ``route()`` surfaces it *past* ``MAX_KEYS_PER_QUERY`` when the
        speaker entity also matches dozens of keys is the open routing WP
        (``project_routing_layer`` — graph-routed selective retrieval); the
        current ``list(set)[:cap]`` slice is arbitrary, so that is intentionally
        NOT asserted here.  This test guards only the indexing fix: the bare
        attribute name resolves to the attribute key.
        """
        with tempfile.TemporaryDirectory() as tmp:
            # A flood of Alice's other keys + the has_email key.
            pairs = [
                _make_pair(f"graph{i}", "Alice", f"Entity{i}", speaker_id="alice")
                for i in range(MAX_KEYS_PER_QUERY + 5)
            ]
            pairs.append(self._make_has_email_pair(key="graph_email", speaker_id="alice"))
            _write_keyed_pairs(Path(tmp), pairs)
            router = QueryRouter(adapter_dir=Path(tmp))

            assert "email" in router._all_entities
            email_indexed_somewhere = False
            for adapter, idx in router._entity_key_index.items():
                if "email" in idx:
                    email_indexed_somewhere = True
                    assert "graph_email" in idx["email"], (
                        f"'email' entry in adapter {adapter!r} maps to {idx['email']!r}; "
                        "must include the has_email key 'graph_email'"
                    )
            assert email_indexed_somewhere, "'email' attribute index entry was never built"

    def test_has_phone_predicate_indexed_as_phone(self):
        """has_phone key is indexed under 'phone'."""
        pair = {
            "key": "graph3",
            "question": "What is Alice's phone?",
            "answer": "+1-555-0100",
            "source_subject": "Alice",
            "source_predicate": "has_phone",
            "source_object": "+1-555-0100",
            "speaker_id": "alice",
            "first_seen_cycle": 1,
        }
        with tempfile.TemporaryDirectory() as tmp:
            _write_keyed_pairs(Path(tmp), [pair])
            router = QueryRouter(adapter_dir=Path(tmp))
            assert "phone" in router._all_entities

    def test_non_has_predicate_not_indexed_as_attribute(self):
        """A plain predicate like 'related_to' does NOT produce an 'elated_to' entry."""
        with tempfile.TemporaryDirectory() as tmp:
            _write_keyed_pairs(Path(tmp), [_make_pair("graph1", "Alice", "Berlin")])
            router = QueryRouter(adapter_dir=Path(tmp))
            # "related_to" starts with 'r', not "has_"; de-prefixed form must not appear.
            assert "elated_to" not in router._all_entities
            assert "related_to" not in router._all_entities

    def test_has_prefix_only_predicate_not_indexed(self):
        """A predicate that is exactly 'has_' (empty attr name) must not add a blank entry."""
        pair = {
            "key": "graph5",
            "question": "Q?",
            "answer": "A.",
            "source_subject": "Alice",
            "source_predicate": "has_",
            "source_object": "something",
            "speaker_id": "alice",
            "first_seen_cycle": 1,
        }
        with tempfile.TemporaryDirectory() as tmp:
            _write_keyed_pairs(Path(tmp), [pair])
            router = QueryRouter(adapter_dir=Path(tmp))
            # Empty de-prefixed name must not be added to entities.
            assert "" not in router._all_entities


# ---------------------------------------------------------------------------
# QueryRouter — simulate_dir: graph.json loading
# ---------------------------------------------------------------------------


def _write_simulate_graph(path: Path, quads: list[dict]) -> None:
    """Write *quads* as a simulate-mode ``graph.json`` at *path*.

    Builds a ``MultiDiGraph`` with one edge per quad, stores the indexed-memory
    key in the ``_IK_KEY_ATTR`` edge attribute, and persists via
    :func:`paramem.server.simulate_store.save_simulate_graph` with
    ``encrypted=False`` so tests read plaintext.
    """
    import networkx as nx

    from paramem.server.simulate_store import _IK_KEY_ATTR, save_simulate_graph

    path.parent.mkdir(parents=True, exist_ok=True)
    graph = nx.MultiDiGraph()
    for quad in quads:
        graph.add_edge(
            quad["subject"],
            quad["object"],
            **{
                _IK_KEY_ATTR: quad["key"],
                "predicate": quad.get("predicate", "related_to"),
                "speaker_id": quad.get("speaker_id", ""),
                "first_seen_cycle": quad.get("first_seen_cycle", 0),
            },
        )
    save_simulate_graph(graph, path, encrypted=False)


def _make_sim_quad(
    key: str,
    subject: str,
    obj: str,
    *,
    speaker_id: str = "",
    predicate: str = "related_to",
    first_seen_cycle: int = 1,
) -> dict:
    """Return a six-field quad dict for simulate-store router fixtures."""
    return {
        "key": key,
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "speaker_id": speaker_id,
        "first_seen_cycle": first_seen_cycle,
    }


class TestQueryRouterSimulateDir:
    """simulate_dir graph.json loading: router indexes entities from simulate store.

    When ``simulate_dir`` is supplied to ``QueryRouter.__init__``, ``reload()``
    scans ``<simulate_dir>/<tier>/graph.json`` for each tier and adds those
    entities and speaker-key mappings to the same indexes used by the
    keyed-pairs path.  Train-mode kp files and simulate-mode graph.json files
    can coexist without conflict.
    """

    def test_empty_simulate_dir_loads_cleanly(self):
        with tempfile.TemporaryDirectory() as tmp:
            sim_dir = Path(tmp) / "simulate"
            sim_dir.mkdir()
            router = QueryRouter(adapter_dir=Path(tmp) / "adapters", simulate_dir=sim_dir)
            assert router.entity_count == 0

    def test_nonexistent_simulate_dir_loads_cleanly(self):
        with tempfile.TemporaryDirectory() as tmp:
            router = QueryRouter(
                adapter_dir=Path(tmp) / "adapters",
                simulate_dir=Path(tmp) / "no_such_dir",
            )
            assert router.entity_count == 0

    def test_episodic_graph_json_entities_indexed(self):
        """Entities from episodic graph.json appear in _all_entities."""
        with tempfile.TemporaryDirectory() as tmp:
            sim_dir = Path(tmp) / "simulate"
            _write_simulate_graph(
                sim_dir / "episodic" / "graph.json",
                [_make_sim_quad("graph1", "Alice", "Berlin", speaker_id="alice")],
            )
            router = QueryRouter(adapter_dir=Path(tmp) / "adapters", simulate_dir=sim_dir)
            assert "alice" in router._all_entities
            assert "berlin" in router._all_entities

    def test_semantic_and_procedural_graph_json_indexed(self):
        """All three tiers' graph.json files contribute to the index."""
        with tempfile.TemporaryDirectory() as tmp:
            sim_dir = Path(tmp) / "simulate"
            _write_simulate_graph(
                sim_dir / "semantic" / "graph.json",
                [_make_sim_quad("sem1", "Bob", "Python", speaker_id="bob")],
            )
            _write_simulate_graph(
                sim_dir / "procedural" / "graph.json",
                [_make_sim_quad("proc1", "Carol", "Tea", speaker_id="carol")],
            )
            router = QueryRouter(adapter_dir=Path(tmp) / "adapters", simulate_dir=sim_dir)
            assert "bob" in router._all_entities
            assert "carol" in router._all_entities

    def test_speaker_id_from_graph_json_indexed(self):
        """speaker_id from graph.json edges appears in _speaker_key_index."""
        with tempfile.TemporaryDirectory() as tmp:
            sim_dir = Path(tmp) / "simulate"
            _write_simulate_graph(
                sim_dir / "episodic" / "graph.json",
                [_make_sim_quad("graph1", "Alice", "Berlin", speaker_id="alice")],
            )
            router = QueryRouter(adapter_dir=Path(tmp) / "adapters", simulate_dir=sim_dir)
            assert "alice" in router._speaker_key_index
            assert "graph1" in router._speaker_key_index["alice"]

    def test_simulate_graph_produces_pa_routing_steps(self):
        """A query mentioning an entity from graph.json resolves to a PA step."""
        from paramem.server.router import Intent

        with tempfile.TemporaryDirectory() as tmp:
            sim_dir = Path(tmp) / "simulate"
            _write_simulate_graph(
                sim_dir / "episodic" / "graph.json",
                [_make_sim_quad("graph1", "Alice", "Berlin", speaker_id="alice")],
            )
            router = QueryRouter(adapter_dir=Path(tmp) / "adapters", simulate_dir=sim_dir)
            plan = router.route("Where does Alice live?", speaker_id="alice")
            assert plan.steps, "Expected PA routing steps from simulate-mode graph.json"
            assert plan.intent == Intent.PERSONAL
            keys = [k for step in plan.steps for k in step.keys_to_probe]
            assert "graph1" in keys

    def test_simulate_dir_and_adapter_dir_coexist(self):
        """Entities from both stores are indexed without conflict."""
        with tempfile.TemporaryDirectory() as tmp:
            adapter_dir = Path(tmp) / "adapters"
            sim_dir = Path(tmp) / "simulate"
            # Train-mode kp in adapter_dir
            _write_keyed_pairs(
                adapter_dir,
                [_make_pair("train1", "TrainEntity", "Berlin", speaker_id="alice")],
                subdir="episodic",
            )
            # Simulate-mode graph.json in sim_dir
            _write_simulate_graph(
                sim_dir / "episodic" / "graph.json",
                [_make_sim_quad("sim1", "SimEntity", "Munich", speaker_id="alice")],
            )
            router = QueryRouter(adapter_dir=adapter_dir, simulate_dir=sim_dir)
            assert "trainentity" in router._all_entities
            assert "simentity" in router._all_entities

    def test_reload_picks_up_new_graph_json(self):
        """After writing graph.json to simulate_dir, reload() picks it up."""
        with tempfile.TemporaryDirectory() as tmp:
            sim_dir = Path(tmp) / "simulate"
            router = QueryRouter(adapter_dir=Path(tmp) / "adapters", simulate_dir=sim_dir)
            assert router.entity_count == 0

            _write_simulate_graph(
                sim_dir / "episodic" / "graph.json",
                [_make_sim_quad("graph1", "Alice", "Berlin", speaker_id="alice")],
            )
            router.reload()
            assert router.entity_count > 0
            assert "alice" in router._all_entities

    def test_missing_graph_json_does_not_crash(self):
        """simulate_dir with no graph.json under any tier → entity_count stays 0."""
        with tempfile.TemporaryDirectory() as tmp:
            sim_dir = Path(tmp) / "simulate"
            sim_dir.mkdir()
            (sim_dir / "episodic").mkdir()  # subdir present but no graph.json
            router = QueryRouter(adapter_dir=Path(tmp) / "adapters", simulate_dir=sim_dir)
            assert router.entity_count == 0

    def test_has_email_predicate_indexed_from_graph_json(self):
        """has_email predicate in graph.json edge is indexed under 'email' entity."""
        with tempfile.TemporaryDirectory() as tmp:
            sim_dir = Path(tmp) / "simulate"
            _write_simulate_graph(
                sim_dir / "episodic" / "graph.json",
                [
                    _make_sim_quad(
                        "graph2",
                        "Alice",
                        "alice@example.com",
                        speaker_id="alice",
                        predicate="has_email",
                    )
                ],
            )
            router = QueryRouter(adapter_dir=Path(tmp) / "adapters", simulate_dir=sim_dir)
            assert "email" in router._all_entities
