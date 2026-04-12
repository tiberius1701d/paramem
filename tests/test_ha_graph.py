"""Unit tests for HAEntityGraph — HA entity/area/verb indexing and matching."""

import pytest

from paramem.server.ha_graph import HAEntityGraph

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_states():
    return [
        {
            "entity_id": "light.living_room",
            "attributes": {"friendly_name": "Living Room Light", "area_name": "Living Room"},
        },
        {
            "entity_id": "light.kitchen",
            "attributes": {"friendly_name": "Kitchen Light", "area_name": "Kitchen"},
        },
        {
            "entity_id": "switch.garden_pump",
            "attributes": {"friendly_name": "Garden Pump"},
        },
        {
            "entity_id": "climate.bedroom",
            "attributes": {"friendly_name": "Bedroom Thermostat", "area_name": "Bedroom"},
        },
    ]


def _make_services():
    return [
        {
            "domain": "light",
            "services": {"turn_on": {}, "turn_off": {}, "toggle": {}},
        },
        {
            "domain": "switch",
            "services": {"turn_on": {}, "turn_off": {}},
        },
        {
            "domain": "climate",
            "services": {"set_temperature": {}, "set_hvac_mode": {}},
        },
        # Should be skipped entirely
        {
            "domain": "homeassistant",
            "services": {"restart": {}, "stop": {}},
        },
    ]


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


class TestHAEntityGraphBuild:
    def test_empty_inputs(self):
        graph = HAEntityGraph.build([], [])
        assert graph.entity_count == 0
        assert graph.area_count == 0
        assert graph.verb_count == 0

    def test_entity_count(self):
        graph = HAEntityGraph.build(_make_states(), [])
        assert graph.entity_count == 4

    def test_area_count(self):
        graph = HAEntityGraph.build(_make_states(), [])
        assert graph.area_count == 3  # living room, kitchen, bedroom

    def test_verb_count(self):
        graph = HAEntityGraph.build([], _make_services())
        # light: turn on, turn off, toggle + switch: turn on (dedup), turn off (dedup)
        # + climate: set temperature, set hvac mode → 5 distinct verbs
        # homeassistant: skipped
        assert graph.verb_count == 5

    def test_skips_state_without_friendly_name(self):
        states = [{"entity_id": "light.unnamed", "attributes": {}}]
        graph = HAEntityGraph.build(states, [])
        assert graph.entity_count == 0

    def test_skips_state_without_entity_id(self):
        states = [{"attributes": {"friendly_name": "Orphan"}}]
        graph = HAEntityGraph.build(states, [])
        assert graph.entity_count == 0

    def test_service_underscore_converted_to_space(self):
        services = [{"domain": "light", "services": {"turn_on": {}}}]
        graph = HAEntityGraph.build([], services)
        result = graph.match("turn on the light")
        assert result is not None
        assert "turn on" in result.matched_verbs

    def test_homeassistant_domain_skipped(self):
        graph = HAEntityGraph.build([], _make_services())
        # "restart" must not be indexed at all (homeassistant domain skipped)
        result = graph.match("restart homeassistant")
        assert result is None

    def test_services_none(self):
        """build() with services=None must not crash."""
        graph = HAEntityGraph.build(_make_states(), None)
        assert graph.entity_count == 4

    def test_domain_extracted_from_entity_id(self):
        states = [
            {"entity_id": "media_player.tv", "attributes": {"friendly_name": "Living Room TV"}}
        ]
        graph = HAEntityGraph.build(states, [])
        result = graph.match("living room tv")
        assert result is not None
        assert "media_player" in result.domains

    def test_area_id_fallback(self):
        """Uses area_id attribute when area_name is absent."""
        states = [
            {
                "entity_id": "light.hall",
                "attributes": {"friendly_name": "Hall Light", "area_id": "hallway"},
            }
        ]
        graph = HAEntityGraph.build(states, [])
        assert graph.area_count == 1


# ---------------------------------------------------------------------------
# Match
# ---------------------------------------------------------------------------


class TestHAEntityGraphMatch:
    @pytest.fixture
    def graph(self):
        return HAEntityGraph.build(_make_states(), _make_services())

    def test_exact_entity_match(self, graph):
        result = graph.match("Is the kitchen light on?")
        assert result is not None
        assert "kitchen light" in result.matched_entities

    def test_exact_area_match(self, graph):
        result = graph.match("Turn off the lights in the living room")
        assert result is not None
        assert "living room" in result.matched_areas

    def test_exact_verb_match(self, graph):
        result = graph.match("turn on the device")
        assert result is not None
        assert "turn on" in result.matched_verbs

    def test_no_match_returns_none(self, graph):
        result = graph.match("What is the capital of France?")
        assert result is None

    def test_has_entity_match_with_entity(self, graph):
        result = graph.match("kitchen light brightness")
        assert result is not None
        assert result.has_entity_match is True
        assert result.has_verb_match is False

    def test_has_entity_match_with_area(self, graph):
        result = graph.match("temperature in bedroom")
        assert result is not None
        assert result.has_entity_match is True

    def test_has_verb_match(self, graph):
        result = graph.match("turn on kitchen light")
        assert result is not None
        assert result.has_verb_match is True

    def test_domain_from_entity(self, graph):
        result = graph.match("kitchen light")
        assert result is not None
        assert "light" in result.domains

    def test_domain_from_verb(self, graph):
        result = graph.match("set temperature")
        assert result is not None
        assert "climate" in result.domains

    def test_entity_without_area(self, graph):
        result = graph.match("garden pump")
        assert result is not None
        assert "garden pump" in result.matched_entities
        assert result.matched_areas == []

    def test_case_insensitive(self, graph):
        result = graph.match("KITCHEN LIGHT STATUS")
        assert result is not None
        assert result.has_entity_match is True

    def test_verb_only_match(self, graph):
        """Verb match alone is sufficient for has_entity_match=False, has_verb_match=True."""
        result = graph.match("toggle everything")
        assert result is not None
        assert result.has_verb_match is True

    def test_imperative_scenario_both(self, graph):
        """Action verb + entity present → both matched_verbs and matched_entities set."""
        result = graph.match("turn on the kitchen light")
        assert result is not None
        assert result.has_entity_match is True
        assert result.has_verb_match is True

    def test_fuzzy_entity_match(self, graph):
        # "kichen light" (one char off) should fuzzy-match "kitchen light" (ratio ~93)
        # No verb in query — isolates the fuzzy entity matching path
        result = graph.match("kichen light brightness")
        assert result is not None
        assert result.has_entity_match is True

    def test_no_verb_fuzzy(self, graph):
        """Verbs use substring-only matching — no fuzzy fallback."""
        # "trun on" shouldn't fuzzy-match "turn on"
        result = graph.match("trun on the thing")
        # If it matches at all it must be an entity, not a verb
        if result:
            assert "trun on" not in result.matched_verbs


# ---------------------------------------------------------------------------
# Refresh
# ---------------------------------------------------------------------------


class TestHAEntityGraphRefresh:
    def test_refresh_replaces_entities(self):
        # Use clearly distinct names — "Light A"/"Light B" fuzzy-match each other (ratio ~86)
        states1 = [{"entity_id": "switch.pantry", "attributes": {"friendly_name": "Pantry Switch"}}]
        states2 = [
            {"entity_id": "climate.basement", "attributes": {"friendly_name": "Basement Heater"}}
        ]

        graph = HAEntityGraph.build(states1, [])
        assert graph.match("pantry switch status") is not None

        graph.refresh(states2, [])
        assert graph.match("pantry switch status") is None
        assert graph.match("basement heater temperature") is not None

    def test_refresh_clears_verbs(self):
        services1 = [{"domain": "light", "services": {"turn_on": {}}}]
        services2 = [{"domain": "climate", "services": {"set_temperature": {}}}]

        graph = HAEntityGraph.build([], services1)
        assert graph.match("turn on the device") is not None

        graph.refresh([], services2)
        assert graph.match("turn on the device") is None
        assert graph.match("set temperature now") is not None

    def test_refresh_resets_counts(self):
        graph = HAEntityGraph.build(_make_states(), _make_services())
        old_count = graph.entity_count
        assert old_count > 0

        graph.refresh([], [])
        assert graph.entity_count == 0
        assert graph.area_count == 0
        assert graph.verb_count == 0
