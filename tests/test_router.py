"""Unit tests for QueryRouter — dual-graph routing and entity indexing."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

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

    def test_no_state_with_general_fail_closed(self):
        from paramem.server.config import IntentConfig
        from paramem.server.router import Intent

        ha = _make_ha_graph(None)
        cfg = IntentConfig(fail_closed_intent="general")
        with tempfile.TemporaryDirectory() as tmp:
            router = QueryRouter(adapter_dir=Path(tmp), ha_graph=ha, intent_config=cfg)
            plan = router.route("What is the capital of France?")

            assert plan.intent == Intent.GENERAL
