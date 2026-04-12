"""Unit tests for QueryRouter — dual-graph routing and entity indexing."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from paramem.server.router import (
    MAX_KEYS_PER_QUERY,
    QueryRouter,
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
    speaker_id: str | None = None,
) -> dict:
    pair = {"key": key, "source_subject": subject, "source_object": obj}
    if speaker_id is not None:
        pair["speaker_id"] = speaker_id
    return pair


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

    def test_no_speaker_id_returns_none_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            router = self._make_router(tmp)
            plan = router.route("Where does Alice live?", speaker_id=None)
            assert plan.match_source == "none"
            assert plan.steps == []

    def test_wrong_speaker_id_returns_none_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            router = self._make_router(tmp)
            plan = router.route("Where does Alice live?", speaker_id="bob")
            assert plan.match_source == "none"

    def test_correct_speaker_id_pa_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            router = self._make_router(tmp)
            plan = router.route("Where does Alice live?", speaker_id="alice")
            assert plan.match_source == "pa"

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
            assert plan.match_source == "pa"

    def test_matched_entities_in_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            router = self._make_router(tmp)
            plan = router.route("Alice lives in Berlin", speaker_id="alice")
            assert "alice" in plan.matched_entities or "berlin" in plan.matched_entities

    def test_fuzzy_entity_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            router = self._make_router(tmp)
            # "Alic" should fuzzy-match "alice" at ratio ~89
            plan = router.route("Tell me about Alic", speaker_id="alice")
            assert plan.match_source == "pa"

    def test_adapter_order_procedural_semantic_episodic(self):
        """Procedural steps come before semantic before episodic."""
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
            # semantic before episodic (procedural absent)
            if "semantic" in names and "episodic" in names:
                assert names.index("semantic") < names.index("episodic")


# ---------------------------------------------------------------------------
# QueryRouter — route(): HA path
# ---------------------------------------------------------------------------


class TestQueryRouterRouteHA:
    def test_no_ha_graph_no_ha_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            router = QueryRouter(adapter_dir=Path(tmp), ha_graph=None)
            plan = router.route("Turn on the kitchen light")
            assert plan.match_source == "none"

    def test_ha_only_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            ha = _make_ha_graph(_make_ha_match(entities=["kitchen light"], domains=["light"]))
            router = QueryRouter(adapter_dir=Path(tmp), ha_graph=ha)
            plan = router.route("Is the kitchen light on?")
            assert plan.match_source == "ha"
            assert plan.steps == []

    def test_verb_only_ha_match_does_not_set_entity_match(self):
        """Verb-only ha_match (no entity/area) → has_entity_match=False → entity path skipped.
        An interrogative is used so the imperative fallback also does not fire, isolating
        the entity-path invariant."""
        with tempfile.TemporaryDirectory() as tmp:
            match = _make_ha_match(verbs=["turn on"], domains=["light"])
            ha = _make_ha_graph(match)
            router = QueryRouter(adapter_dir=Path(tmp), ha_graph=ha)
            # Interrogative → fallback does not fire; verb-only match → has_ha=False
            plan = router.route("Is turn on working?")
            assert plan.match_source == "none"

    def test_verb_only_ha_match_with_imperative_promoted_via_fallback(self):
        """Verb-only ha_match + imperative → entity path skips, fallback promotes to HA."""
        with tempfile.TemporaryDirectory() as tmp:
            match = _make_ha_match(verbs=["turn on"], domains=["light"])
            ha = _make_ha_graph(match)
            router = QueryRouter(adapter_dir=Path(tmp), ha_graph=ha)
            plan = router.route("Turn on something")
            # Entity path: has_ha=False (verb-only). Fallback: imperative → match_source="ha"
            assert plan.match_source == "ha"
            assert plan.imperative is True

    def test_ha_verb_plus_entity_non_interrogative_is_imperative(self):
        with tempfile.TemporaryDirectory() as tmp:
            ha = _make_ha_graph(
                _make_ha_match(
                    entities=["kitchen light"],
                    verbs=["turn on"],
                    domains=["light"],
                )
            )
            router = QueryRouter(adapter_dir=Path(tmp), ha_graph=ha)
            plan = router.route("Turn on the kitchen light")
            assert plan.imperative is True

    def test_ha_verb_plus_entity_interrogative_is_not_imperative(self):
        with tempfile.TemporaryDirectory() as tmp:
            ha = _make_ha_graph(
                _make_ha_match(
                    entities=["kitchen light"],
                    verbs=["turn on"],
                    domains=["light"],
                )
            )
            router = QueryRouter(adapter_dir=Path(tmp), ha_graph=ha)
            plan = router.route("Is the kitchen light turned on?")
            assert plan.imperative is False

    def test_ha_entity_without_verb_not_imperative(self):
        with tempfile.TemporaryDirectory() as tmp:
            ha = _make_ha_graph(
                _make_ha_match(entities=["kitchen light"], verbs=[], domains=["light"])
            )
            router = QueryRouter(adapter_dir=Path(tmp), ha_graph=ha)
            plan = router.route("kitchen light status")
            assert plan.imperative is False

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

    def test_both_match_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_keyed_pairs(
                Path(tmp),
                [_make_pair("graph1", "Alice", "kitchen light", speaker_id="alice")],
            )
            ha = _make_ha_graph(_make_ha_match(entities=["kitchen light"], domains=["light"]))
            router = QueryRouter(adapter_dir=Path(tmp), ha_graph=ha)
            plan = router.route("Alice kitchen light", speaker_id="alice")
            assert plan.match_source == "both"

    def test_both_match_has_pa_steps(self):
        """When both match, PA steps are still built."""
        with tempfile.TemporaryDirectory() as tmp:
            _write_keyed_pairs(
                Path(tmp),
                [_make_pair("graph1", "Alice", "kitchen light", speaker_id="alice")],
            )
            ha = _make_ha_graph(_make_ha_match(entities=["kitchen light"], domains=["light"]))
            router = QueryRouter(adapter_dir=Path(tmp), ha_graph=ha)
            plan = router.route("Alice kitchen light", speaker_id="alice")
            assert plan.steps  # PA steps present


# ---------------------------------------------------------------------------
# QueryRouter — imperative fallback routing
# ---------------------------------------------------------------------------


class TestQueryRouterImperativeFallback:
    """Imperatives with no entity match should be promoted to HA when HA is configured."""

    def test_imperative_no_match_routes_to_ha_when_graph_present(self):
        """Generic imperative → match_source="ha", imperative=True."""
        with tempfile.TemporaryDirectory() as tmp:
            # ha_graph present but returns None (no entity matches "play music")
            ha = _make_ha_graph(None)
            router = QueryRouter(adapter_dir=Path(tmp), ha_graph=ha)
            plan = router.route("Play music")
            assert plan.match_source == "ha"
            assert plan.imperative is True

    def test_interrogative_no_match_stays_none(self):
        """Question with no match → match_source='none' (route to SOTA)."""
        with tempfile.TemporaryDirectory() as tmp:
            ha = _make_ha_graph(None)
            router = QueryRouter(adapter_dir=Path(tmp), ha_graph=ha)
            plan = router.route("What is the capital of France?")
            assert plan.match_source == "none"
            assert plan.imperative is False

    def test_imperative_no_ha_graph_stays_none(self):
        """Without HA graph, imperative fallback does not fire."""
        with tempfile.TemporaryDirectory() as tmp:
            router = QueryRouter(adapter_dir=Path(tmp), ha_graph=None)
            plan = router.route("Play music")
            assert plan.match_source == "none"

    def test_imperative_fallback_does_not_fire_when_ha_already_matched(self):
        """When HA entity graph already matched, match_source is 'ha' for that reason."""
        with tempfile.TemporaryDirectory() as tmp:
            ha = _make_ha_graph(_make_ha_match(entities=["kitchen light"], domains=["light"]))
            router = QueryRouter(adapter_dir=Path(tmp), ha_graph=ha)
            plan = router.route("Turn on the kitchen light")
            # Still "ha" — but driven by entity match, not fallback
            assert plan.match_source == "ha"

    def test_imperative_fallback_no_pa_steps(self):
        """Fallback-promoted HA route has no PA steps (no PA match)."""
        with tempfile.TemporaryDirectory() as tmp:
            ha = _make_ha_graph(None)
            router = QueryRouter(adapter_dir=Path(tmp), ha_graph=ha)
            plan = router.route("Set a timer for five minutes")
            assert plan.match_source == "ha"
            assert plan.steps == []

    def test_various_imperative_commands_promoted(self):
        """Spot-check several real-world device commands that lack entity names."""
        with tempfile.TemporaryDirectory() as tmp:
            ha = _make_ha_graph(None)
            router = QueryRouter(adapter_dir=Path(tmp), ha_graph=ha)
            for command in [
                "Play music",
                "Stop the music",
                "Turn off everything",
                "Lock the door",
                "Dim the lights",
            ]:
                plan = router.route(command)
                assert plan.match_source == "ha", f"Expected 'ha' for: {command!r}"
                assert plan.imperative is True, f"Expected imperative=True for: {command!r}"


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
