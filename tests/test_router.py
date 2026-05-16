"""Unit tests for QueryRouter — dual-graph routing and entity indexing."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from paramem.server.router import (
    QueryRouter,
    _interim_sort_key,
    _is_interrogative,
)
from paramem.training.memory_store import MemoryStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pair(
    key: str,
    subject: str,
    obj: str,
    *,
    speaker_id: str = "",
    question: str = "Q?",
    answer: str = "A.",
    predicate: str = "related_to",
    first_seen_cycle: int = 1,
    adapter_id: str = "episodic",
) -> dict:
    """Return a canonical cache-entry dict for use in router tests.

    Uses the canonical field names (``subject``, ``predicate``, ``object``).
    ``adapter_id`` controls which tier the router assigns this key to when
    building the loop registry.
    """
    return {
        "key": key,
        "question": question,
        "answer": answer,
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "speaker_id": speaker_id,
        "first_seen_cycle": first_seen_cycle,
        "_adapter_id": adapter_id,  # internal: used by _make_router_from_pairs
    }


def _make_router_from_pairs(
    pairs: list[dict],
    *,
    adapter_dir: "Path | None" = None,
    ha_graph: "MagicMock | None" = None,
    simulate_dir: "Path | None" = None,
) -> "QueryRouter":
    """Create a QueryRouter backed by a mock ConsolidationLoop.

    Builds an ``indexed_key_cache`` from *pairs* and a matching ``KeyRegistry``
    so ``reload()`` populates entity/speaker indexes without touching the file
    system.  The ``_adapter_id`` field in each pair controls which tier the
    key is registered under.

    Parameters
    ----------
    pairs:
        List of dicts as returned by ``_make_pair``.
    adapter_dir:
        Optional path for the router's ``adapter_dir``.  Ignored for entity
        indexing (which now comes from the loop cache) but kept so callers
        that also write graph files can specify where to look.
    ha_graph:
        Optional mock HA entity graph.
    simulate_dir:
        Optional simulate-mode directory (for graph-based entity loading).
    """
    from paramem.training.memory_store import MemoryStore

    store = MemoryStore(replay_enabled=True)
    for pair in pairs:
        key = pair["key"]
        adapter_id = pair.get("_adapter_id", "episodic")
        entry = {k: v for k, v in pair.items() if k != "_adapter_id"}
        store.put(adapter_id, key, entry)

    kwargs: dict = {
        "adapter_dir": adapter_dir or Path("/nonexistent"),
        "memory_store": store,
    }
    if ha_graph is not None:
        kwargs["ha_graph"] = ha_graph
    if simulate_dir is not None:
        kwargs["simulate_dir"] = simulate_dir
    return QueryRouter(**kwargs)


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
    def test_empty_cache_loads_cleanly(self):
        router = _make_router_from_pairs([])
        assert router.entity_count == 0
        assert router.adapter_count == 0

    def test_nonexistent_directory_loads_cleanly(self):
        router = QueryRouter(
            adapter_dir=Path("/nonexistent/path/xyz"),
            memory_store=MemoryStore(replay_enabled=False),
        )
        assert router.entity_count == 0

    def test_episodic_adapter_id_appears_in_entity_key_index(self):
        """Key registered under episodic appears in that tier's entity index."""
        router = _make_router_from_pairs(
            [_make_pair("graph1", "Alice", "Berlin", speaker_id="alice", adapter_id="episodic")]
        )
        assert "episodic" in router._entity_key_index

    def test_semantic_adapter_id_appears_in_entity_key_index(self):
        """Key registered under semantic appears under the semantic tier."""
        router = _make_router_from_pairs(
            [_make_pair("graph1", "Alice", "Berlin", speaker_id="alice", adapter_id="semantic")]
        )
        assert "semantic" in router._entity_key_index

    def test_subject_and_object_both_indexed(self):
        router = _make_router_from_pairs(
            [_make_pair("graph1", "Alice", "Berlin", speaker_id="alice")]
        )
        assert "alice" in router._all_entities
        assert "berlin" in router._all_entities

    def test_speaker_id_indexed(self):
        router = _make_router_from_pairs(
            [_make_pair("graph1", "Alice", "Berlin", speaker_id="alice")]
        )
        assert "alice" in router._speaker_key_index
        assert "graph1" in router._speaker_key_index["alice"]

    def test_pair_without_speaker_id_not_in_speaker_index(self):
        router = _make_router_from_pairs([_make_pair("graph1", "Alice", "Berlin")])
        assert len(router._speaker_key_index) == 0

    def test_no_loop_provider_gives_empty_indexes(self):
        """Without a loop_provider, indexes stay empty (files no longer read)."""
        with tempfile.TemporaryDirectory() as tmp:
            router = QueryRouter(
                adapter_dir=Path(tmp), memory_store=MemoryStore(replay_enabled=False)
            )
            assert router.entity_count == 0

    def test_multiple_adapters_indexed(self):
        router = _make_router_from_pairs(
            [
                _make_pair("ep1", "Alice", "Berlin", speaker_id="alice", adapter_id="episodic"),
                _make_pair("se1", "Alice", "Python", speaker_id="alice", adapter_id="semantic"),
            ]
        )
        assert router.adapter_count == 2
        assert "episodic" in router._entity_key_index
        assert "semantic" in router._entity_key_index

    def test_reload_replaces_index(self):
        """reload() observes mutations to the injected store."""
        store = MemoryStore(replay_enabled=True)

        router = QueryRouter(
            adapter_dir=Path("/nonexistent"),
            memory_store=store,
        )
        assert router.entity_count == 0

        # Seed the same store, reload — index populates
        store.put(
            "episodic",
            "graph1",
            {
                "key": "graph1",
                "subject": "Alice",
                "predicate": "lives_in",
                "object": "Berlin",
                "speaker_id": "alice",
                "first_seen_cycle": 1,
            },
        )
        router.reload()
        assert router.entity_count > 0

        # Clear the store, reload — index must clear
        store.delete("graph1")
        router.reload()
        assert router.entity_count == 0

    def test_entity_shorter_than_two_chars_skipped(self):
        router = _make_router_from_pairs([_make_pair("graph1", "A", "Berlin", speaker_id="alice")])
        assert "a" not in router._all_entities
        assert "berlin" in router._all_entities


# ---------------------------------------------------------------------------
# QueryRouter — route(): PA path
# ---------------------------------------------------------------------------


class TestQueryRouterRoutePA:
    @staticmethod
    def _make_router() -> QueryRouter:
        return _make_router_from_pairs(
            [
                _make_pair("graph1", "Alice", "Berlin", speaker_id="alice"),
                _make_pair("graph2", "Alice", "Python", speaker_id="alice"),
            ]
        )

    def test_no_speaker_id_returns_no_steps(self):
        from paramem.server.router import Intent

        router = self._make_router()
        plan = router.route("Where does Alice live?", speaker_id=None)
        assert plan.steps == []
        assert plan.intent == Intent.UNKNOWN

    def test_wrong_speaker_id_returns_no_steps(self):
        router = self._make_router()
        plan = router.route("Where does Alice live?", speaker_id="bob")
        assert plan.steps == []

    def test_correct_speaker_id_pa_match(self):
        from paramem.server.router import Intent

        router = self._make_router()
        plan = router.route("Where does Alice live?", speaker_id="alice")
        assert plan.steps  # PA match → steps populated
        assert plan.intent == Intent.PERSONAL

    def test_pa_match_steps_populated(self):
        router = self._make_router()
        plan = router.route("Tell me about Alice", speaker_id="alice")
        assert plan.steps
        keys = [k for step in plan.steps for k in step.keys_to_probe]
        assert "graph1" in keys or "graph2" in keys

    def test_strategy_targeted_probe_when_steps(self):
        router = self._make_router()
        plan = router.route("Alice lives in Berlin", speaker_id="alice")
        assert plan.strategy == "targeted_probe"

    def test_strategy_direct_when_no_steps(self):
        router = self._make_router()
        # No entity in query → no steps
        plan = router.route("What is the capital of France?", speaker_id="alice")
        # "france" not in entity list → no steps
        assert plan.strategy == "direct"

    def test_speaker_name_injected_as_implicit_entity(self):
        from paramem.server.router import Intent

        # Only "alice" in the index
        router = _make_router_from_pairs(
            [_make_pair("graph1", "Alice", "Berlin", speaker_id="alice")]
        )
        # Query doesn't mention alice, but speaker="Alice" injects it
        plan = router.route("Where do I live?", speaker="Alice", speaker_id="alice")
        # alice is now an implicit entity — should trigger PA match
        assert plan.intent == Intent.PERSONAL
        assert plan.steps

    def test_matched_entities_in_plan(self):
        router = self._make_router()
        plan = router.route("Alice lives in Berlin", speaker_id="alice")
        assert "alice" in plan.matched_entities or "berlin" in plan.matched_entities

    def test_fuzzy_entity_match(self):
        from paramem.server.router import Intent

        router = self._make_router()
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
        router = _make_router_from_pairs(
            [
                _make_pair("ep1", "Alice", "Berlin", speaker_id="alice", adapter_id="episodic"),
                _make_pair("se1", "Alice", "Python", speaker_id="alice", adapter_id="semantic"),
            ]
        )
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
            router = QueryRouter(
                adapter_dir=Path(tmp), memory_store=MemoryStore(replay_enabled=False), ha_graph=None
            )
            plan = router.route("Turn on the kitchen light")
            assert plan.ha_domains == []
            assert plan.steps == []

    def test_ha_only_match(self):
        from paramem.server.router import Intent

        with tempfile.TemporaryDirectory() as tmp:
            ha = _make_ha_graph(_make_ha_match(entities=["kitchen light"], domains=["light"]))
            router = QueryRouter(
                adapter_dir=Path(tmp), memory_store=MemoryStore(replay_enabled=False), ha_graph=ha
            )
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
            router = QueryRouter(
                adapter_dir=Path(tmp), memory_store=MemoryStore(replay_enabled=False), ha_graph=ha
            )
            plan = router.route("Is turn on working?")
            assert plan.steps == []
            assert plan.intent != Intent.PERSONAL

    def test_ha_domains_populated(self):
        with tempfile.TemporaryDirectory() as tmp:
            ha = _make_ha_graph(
                _make_ha_match(entities=["bedroom thermostat"], domains=["climate"])
            )
            router = QueryRouter(
                adapter_dir=Path(tmp), memory_store=MemoryStore(replay_enabled=False), ha_graph=ha
            )
            plan = router.route("bedroom thermostat temperature")
            assert "climate" in plan.ha_domains

    def test_ha_domains_empty_when_no_ha_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            ha = _make_ha_graph(None)
            router = QueryRouter(
                adapter_dir=Path(tmp), memory_store=MemoryStore(replay_enabled=False), ha_graph=ha
            )
            plan = router.route("Tell me something random")
            assert plan.ha_domains == []

    def test_pa_and_ha_overlap_classifies_personal(self):
        """When PA and HA both match the same query, intent=PERSONAL (PA wins
        in classify_intent's state-first dispatch).  PA steps and HA domains
        both populated so downstream consumers can still see both signals.
        """
        from paramem.server.router import Intent

        ha = _make_ha_graph(_make_ha_match(entities=["kitchen light"], domains=["light"]))
        router = _make_router_from_pairs(
            [_make_pair("graph1", "Alice", "kitchen light", speaker_id="alice")],
            ha_graph=ha,
        )
        plan = router.route("Alice kitchen light", speaker_id="alice")
        assert plan.intent == Intent.PERSONAL
        assert plan.steps
        assert "light" in plan.ha_domains


# ---------------------------------------------------------------------------
# QueryRouter — no Top-K cap on the authenticated path
# ---------------------------------------------------------------------------


class TestQueryRouterNoTopKCap:
    """Authenticated speakers receive every entity-narrowed key they own
    in plan.steps — no slice.  The legacy MAX_KEYS_PER_QUERY=10 cap
    caused narrow attribute keys (``has_email``) to lose a non-
    deterministic coin-flip when the speaker had >10 keys touching the
    matched entity, breaking attribute queries.  Closed by removing the
    slice; inference latency is mitigated by ``inference.preload_cache``
    (default true) which serves probes from the RAM cache instead of
    one ``model.generate`` per key.
    """

    def test_no_cap_for_authenticated_speaker(self):
        """All 15 keys touching the matched entity reach plan.steps."""
        pairs = [
            _make_pair(f"graph{i}", "Alice", f"Entity{i}", speaker_id="alice") for i in range(15)
        ]
        router = _make_router_from_pairs(pairs)
        plan = router.route("Alice", speaker_id="alice")
        assert plan.steps, "Authenticated speaker with entity match must emit steps"
        total_keys = sum(len(step.keys_to_probe) for step in plan.steps)
        assert total_keys == 15, (
            f"Expected all 15 entity-narrowed keys to reach steps, got {total_keys}"
        )

    def test_keys_to_probe_are_deterministic_sorted(self):
        """plan.steps[*].keys_to_probe is sorted (no hash-randomization flake)."""
        pairs = [
            _make_pair(f"graph{i:03d}", "Alice", f"Entity{i}", speaker_id="alice")
            for i in range(15)
        ]
        router = _make_router_from_pairs(pairs)
        plan = router.route("Alice", speaker_id="alice")
        for step in plan.steps:
            assert step.keys_to_probe == sorted(step.keys_to_probe), (
                f"keys_to_probe must be sorted: {step.keys_to_probe}"
            )


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
    """Tests for multi-adapter interim routing via cache-based indexing.

    Interim adapters keep their real name (``episodic_interim_<stamp>``) in
    the routing plan so probe-time ``switch_adapter(model, step.adapter_name)``
    activates the actual trained slot in ``model.peft_config`` — the same
    name the consolidation pipeline created the adapter under.  Probe order
    (``route()`` bottom block) places interim names newest-first, ahead of
    the main ``"episodic"`` slot.  Folding interim into ``"episodic"`` was a
    regression caught 2026-05-14: it pointed inference at an untrained main
    slot while the actual trained weights stayed inert under the unstripped
    interim name (training-time recall 134/134, inference-time recall 0/10).
    """

    def test_interim_keys_routed_under_their_real_adapter_name(self):
        """Interim-registered keys appear in a step named after the interim adapter."""
        router = _make_router_from_pairs(
            [
                _make_pair("ep1", "Alice", "Berlin", speaker_id="alice", adapter_id="episodic"),
                _make_pair(
                    "int1",
                    "Alice",
                    "Prague",
                    speaker_id="alice",
                    adapter_id="episodic_interim_20260417T0000",
                ),
            ]
        )
        plan = router.route("Alice", speaker_id="alice")
        names = [s.adapter_name for s in plan.steps]
        assert "episodic" in names, "main episodic step missing"
        assert "episodic_interim_20260417T0000" in names, "interim step missing"
        # Each key lives in its own tier's step.
        ep_keys = next(
            (set(s.keys_to_probe) for s in plan.steps if s.adapter_name == "episodic"), set()
        )
        int_keys = next(
            (
                set(s.keys_to_probe)
                for s in plan.steps
                if s.adapter_name == "episodic_interim_20260417T0000"
            ),
            set(),
        )
        assert "ep1" in ep_keys and "ep1" not in int_keys
        assert "int1" in int_keys and "int1" not in ep_keys

    def test_interim_keys_multiple_registered(self):
        """Multiple interim adapter IDs each get their own step, newest-stamp first."""
        pairs = [
            _make_pair("ep1", "Alice", "Berlin", speaker_id="alice", adapter_id="episodic"),
        ]
        for stamp in ("20260415T0000", "20260416T0000", "20260417T0000"):
            pairs.append(
                _make_pair(
                    f"int_{stamp}",
                    "Alice",
                    f"City_{stamp}",
                    speaker_id="alice",
                    adapter_id=f"episodic_interim_{stamp}",
                )
            )
        router = _make_router_from_pairs(pairs)
        plan = router.route("Alice", speaker_id="alice")

        names = [s.adapter_name for s in plan.steps]
        # Each interim adapter must appear under its real name (no folding).
        assert "episodic_interim_20260415T0000" in names
        assert "episodic_interim_20260416T0000" in names
        assert "episodic_interim_20260417T0000" in names
        assert "episodic" in names
        # Probe order: interim newest-first, then main episodic (per route() block).
        interim_in_order = [n for n in names if n.startswith("episodic_interim_")]
        assert interim_in_order == [
            "episodic_interim_20260417T0000",
            "episodic_interim_20260416T0000",
            "episodic_interim_20260415T0000",
        ], f"Interim adapters must be newest-first: {interim_in_order}"
        # Main episodic comes after all interims.
        assert names.index("episodic") > max(names.index(n) for n in interim_in_order)

    def test_route_speaker_scoping_applies_to_interim_keys(self):
        """Interim key tagged with 'alice' is invisible to speaker_id='bob'."""
        router = _make_router_from_pairs(
            [
                _make_pair(
                    "int1",
                    "Alice",
                    "Prague",
                    speaker_id="alice",
                    adapter_id="episodic_interim_20260417T0000",
                )
            ]
        )
        # Bob queries — alice's keys must be invisible
        plan = router.route("Alice", speaker_id="bob")
        assert plan.steps == [], (
            "No step should appear when speaker scoping filters all interim keys"
        )

    def test_reload_with_cleared_cache_removes_interim_keys(self):
        """After reload() with the store emptied, formerly interim-registered keys vanish."""
        store = MemoryStore(replay_enabled=True)
        store.put(
            "episodic_interim_20260417T0000",
            "int1",
            {
                "key": "int1",
                "subject": "Alice",
                "predicate": "lives_in",
                "object": "Prague",
                "speaker_id": "alice",
                "first_seen_cycle": 1,
            },
        )

        router = QueryRouter(
            adapter_dir=Path("/nonexistent"),
            memory_store=store,
        )
        assert router.entity_count > 0

        # Empty the same store in place, reload — all keys gone
        store.delete("int1")
        router.reload()
        plan = router.route("Alice", speaker_id="alice")
        assert plan.steps == [], "Steps must be empty after store cleared"

    def test_route_interim_only_match_emits_interim_step(self):
        """Entity matched ONLY in an interim-registered key → step named after the interim adapter.

        The plan step must carry the same name PEFT knows the adapter by
        (``episodic_interim_<stamp>``) so probe-time ``switch_adapter`` lands
        on the trained slot.  No main-episodic step is emitted because no
        keys are registered under the bare ``"episodic"`` name.
        """
        from paramem.server.router import Intent

        router = _make_router_from_pairs(
            [
                _make_pair(
                    "int1",
                    "Alice",
                    "Vienna",
                    speaker_id="alice",
                    adapter_id="episodic_interim_20260417T0000",
                )
            ]
        )
        plan = router.route("Alice", speaker_id="alice")

        assert len(plan.steps) == 1, f"Expected exactly one step, got {plan.steps}"
        assert plan.steps[0].adapter_name == "episodic_interim_20260417T0000"
        assert plan.intent == Intent.PERSONAL

    def test_route_three_mains_correct_tier_order(self):
        """All three main tiers: order must be procedural → episodic → semantic.

        Probe order: procedural (preferences shape style; load-bearing per
        feedback_router_procedural_first.md) → episodic → semantic.
        """
        router = _make_router_from_pairs(
            [
                _make_pair(
                    "proc1", "Alice", "DarkMode", speaker_id="alice", adapter_id="procedural"
                ),
                _make_pair("ep1", "Alice", "Berlin", speaker_id="alice", adapter_id="episodic"),
                _make_pair("se1", "Alice", "Python", speaker_id="alice", adapter_id="semantic"),
                _make_pair(
                    "int1",
                    "Alice",
                    "InterimFact",
                    speaker_id="alice",
                    adapter_id="episodic_interim_20260418T0000",
                ),
            ]
        )
        plan = router.route("Alice", speaker_id="alice")

        names = [s.adapter_name for s in plan.steps]
        assert names[0] == "procedural", f"First step must be procedural, got {names}"
        assert "episodic" in names[1:], f"Episodic must follow procedural, got {names}"
        if "semantic" in names:
            ep_idx = names.index("episodic")
            sem_idx = names.index("semantic")
            assert ep_idx < sem_idx, f"Episodic must precede semantic, got {names}"


class TestRouteIntentField:
    """G4 wiring: route() populates RoutingPlan.intent via classify_intent.

    These tests pin the contract between router state and the intent
    classifier without exercising the encoder — they cover the state-first
    tiers (PA hit → PERSONAL, HA hit → COMMAND) and the no-state degraded
    path (returns UNKNOWN when no IntentConfig is supplied)."""

    def test_pa_match_yields_personal_intent(self):
        from paramem.server.router import Intent

        router = _make_router_from_pairs(
            [_make_pair("alex_loc", "Alex", "Berlin", speaker_id="alex")],
        )
        plan = router.route("Where does Alex live?", speaker_id="alex")

        assert plan.intent == Intent.PERSONAL
        assert plan.steps  # PA produced probe steps

    def test_ha_only_match_yields_command_intent(self):
        from paramem.server.router import Intent

        ha = _make_ha_graph(_make_ha_match(entities=["light.kitchen"], domains=["light"]))
        with tempfile.TemporaryDirectory() as tmp:
            router = QueryRouter(
                adapter_dir=Path(tmp), memory_store=MemoryStore(replay_enabled=False), ha_graph=ha
            )
            plan = router.route("Turn on the kitchen light")

            assert plan.intent == Intent.COMMAND
            assert plan.ha_domains == ["light"]
            assert plan.steps == []

    def test_both_pa_and_ha_match_personal_wins(self):
        # Privacy-first: PA + HA overlap routes to PERSONAL so cloud
        # escalation stays gated on graph match.
        from paramem.server.router import Intent

        ha = _make_ha_graph(_make_ha_match(entities=["light.bedroom"], domains=["light"]))
        router = _make_router_from_pairs(
            [_make_pair("alex_room", "Alex", "bedroom", speaker_id="alex")],
            ha_graph=ha,
        )
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
            router = QueryRouter(
                adapter_dir=Path(tmp), memory_store=MemoryStore(replay_enabled=False), ha_graph=ha
            )
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
            router = QueryRouter(
                adapter_dir=Path(tmp),
                memory_store=MemoryStore(replay_enabled=False),
                ha_graph=ha,
                intent_config=cfg,
            )
            plan = router.route("What is the capital of France?")

            assert plan.intent == Intent.PERSONAL


# ---------------------------------------------------------------------------
# QueryRouter — entry-format cache entries (B1 regression guard)
# ---------------------------------------------------------------------------


class TestQueryRouterEntryFormat:
    """B1 regression guard: router correctly indexes entry-schema cache entries.

    The canonical source of truth for entity indexing is
    ``ConsolidationLoop.store._entries_flat_view()``, which stores six-field
    entries (``key``, ``subject``, ``predicate``, ``object``,
    ``speaker_id``, ``first_seen_cycle``).  These tests verify that the
    ``_index_pairs`` path reads ``subject``/``object`` (not legacy
    ``source_subject``/``source_object``) and that the entity and speaker
    indexes are correctly built.
    """

    def test_entry_loaded_without_error(self):
        """Router builds a non-empty index from a six-field entry cache."""
        router = _make_router_from_pairs(
            [_make_pair("graph1", "Alice", "Berlin", speaker_id="alice")],
        )
        assert router.entity_count > 0

    def test_subject_and_object_indexed(self):
        """Both ``subject`` and ``object`` are indexed as entities from an entry."""
        router = _make_router_from_pairs(
            [_make_pair("graph1", "Alice", "Berlin", speaker_id="alice")],
        )
        assert "alice" in router._all_entities
        assert "berlin" in router._all_entities

    def test_speaker_id_indexed(self):
        """``speaker_id`` from a cache entry is indexed in ``_speaker_key_index``."""
        router = _make_router_from_pairs(
            [_make_pair("graph1", "Alice", "Berlin", speaker_id="alice")],
        )
        assert "alice" in router._speaker_key_index
        assert "graph1" in router._speaker_key_index["alice"]

    def test_pa_match_produces_steps(self):
        """A query mentioning the entity from a cache entry resolves to a routing step."""
        from paramem.server.router import Intent

        router = _make_router_from_pairs(
            [_make_pair("graph1", "Alice", "Berlin", speaker_id="alice")],
        )
        plan = router.route("Where does Alice live?", speaker_id="alice")
        assert plan.steps, "Expected PA routing steps from cache entry"
        assert plan.intent == Intent.PERSONAL
        keys = [k for step in plan.steps for k in step.keys_to_probe]
        assert "graph1" in keys

    def test_reload_with_updated_cache_repopulates_index(self):
        """reload() after updating the loop store repopulates the entity index."""
        # Start with empty store
        store = MemoryStore(replay_enabled=True)
        router = QueryRouter(adapter_dir=Path("/nonexistent"), memory_store=store)
        assert router.entity_count == 0

        # Add an entry to the store and reload
        store.put(
            "episodic",
            "graph1",
            {
                "key": "graph1",
                "subject": "Alice",
                "predicate": "related_to",
                "object": "Berlin",
                "speaker_id": "alice",
                "first_seen_cycle": 1,
            },
        )
        router.reload()
        assert router.entity_count > 0
        assert "alice" in router._all_entities

    def test_qa_format_entry_still_indexes_via_subject_object(self):
        """QA-format cache entries (with question/answer) still index subject/object correctly."""
        router = _make_router_from_pairs(
            [_make_pair("graph1", "Alice", "Berlin", speaker_id="alice")],
        )
        assert "alice" in router._all_entities
        assert "berlin" in router._all_entities
        assert "alice" in router._speaker_key_index

    def test_no_state_with_general_fail_closed(self):
        from paramem.server.config import IntentConfig
        from paramem.server.router import Intent

        ha = _make_ha_graph(None)
        cfg = IntentConfig(fail_closed_intent="general")
        router = QueryRouter(
            adapter_dir=Path("/nonexistent"),
            memory_store=MemoryStore(replay_enabled=False),
            ha_graph=ha,
            intent_config=cfg,
        )
        plan = router.route("What is the capital of France?")

        assert plan.intent == Intent.GENERAL


# ---------------------------------------------------------------------------
# QueryRouter — attribute-predicate indexing (Bug 3 regression guard)
# ---------------------------------------------------------------------------


class TestAttributePredicateIndexing:
    """Bug 3 regression guard: ``has_<attr>`` predicates are indexed under
    their bare attribute name so attribute queries ("what is my email
    address?") resolve to the correct key.  Pairs with the no-Top-K
    change above: the indexing fix makes the attribute key reachable
    via a second route, and the no-Top-K change ensures it survives
    routing when the speaker entity also matches many keys.
    """

    def _make_has_email_pair(
        self,
        key: str = "graph2",
        speaker_id: str = "alice",
    ) -> dict:
        """Return a canonical cache entry whose predicate is has_email."""
        return {
            "key": key,
            "question": "What is Alice's email?",
            "answer": "alice@example.com",
            "subject": "Alice",
            "predicate": "has_email",
            "object": "alice@example.com",
            "speaker_id": speaker_id,
            "first_seen_cycle": 1,
            "_adapter_id": "episodic",
        }

    def test_has_email_predicate_indexed_as_email(self):
        """has_email cache entry is indexed under 'email' entity."""
        router = _make_router_from_pairs([self._make_has_email_pair()])
        # "email" must appear in the entity list.
        assert "email" in router._all_entities

    def test_has_email_key_reachable_via_email_attribute_index(self):
        """The has_email key maps to the 'email' attribute-index entry — deterministic.

        With the Top-K cap removed on the authenticated path, the
        has_email key also reaches ``plan.steps`` even when the speaker
        entity drags dozens of unrelated keys into the merge — see
        :class:`TestQueryRouterNoTopKCap`.  This test guards only the
        indexing fix: the bare attribute name resolves to the attribute
        key in ``_entity_key_index``.
        """
        # A flood of Alice's other keys + the has_email key.
        pairs = [
            _make_pair(f"graph{i}", "Alice", f"Entity{i}", speaker_id="alice") for i in range(15)
        ]
        pairs.append(self._make_has_email_pair(key="graph_email", speaker_id="alice"))
        router = _make_router_from_pairs(pairs)

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
        """has_phone cache entry is indexed under 'phone'."""
        pair = {
            "key": "graph3",
            "question": "What is Alice's phone?",
            "answer": "+1-555-0100",
            "subject": "Alice",
            "predicate": "has_phone",
            "object": "+1-555-0100",
            "speaker_id": "alice",
            "first_seen_cycle": 1,
            "_adapter_id": "episodic",
        }
        router = _make_router_from_pairs([pair])
        assert "phone" in router._all_entities

    def test_non_has_predicate_not_indexed_as_attribute(self):
        """A plain predicate like 'related_to' does NOT produce an 'elated_to' entry."""
        router = _make_router_from_pairs([_make_pair("graph1", "Alice", "Berlin")])
        # "related_to" starts with 'r', not "has_"; de-prefixed form must not appear.
        assert "elated_to" not in router._all_entities
        assert "related_to" not in router._all_entities

    def test_has_prefix_only_predicate_not_indexed(self):
        """A predicate that is exactly 'has_' (empty attr name) must not add a blank entry."""
        pair = {
            "key": "graph5",
            "question": "Q?",
            "answer": "A.",
            "subject": "Alice",
            "predicate": "has_",
            "object": "something",
            "speaker_id": "alice",
            "first_seen_cycle": 1,
            "_adapter_id": "episodic",
        }
        router = _make_router_from_pairs([pair])
        # Empty de-prefixed name must not be added to entities.
        assert "" not in router._all_entities


# ---------------------------------------------------------------------------
# QueryRouter — simulate-mode routing via loop cache
# ---------------------------------------------------------------------------


class TestQueryRouterSimulateDir:
    """Simulate-mode routing: router indexes entities from the loop's memory store.

    In simulate mode, ``ConsolidationLoop`` populates ``store`` from
    the simulate store (graph.json) just as train mode populates it from adapter
    weights.  The router's ``reload()`` uses only ``loop.store`` as the
    canonical source — ``simulate_dir`` is stored for back-compat but is not
    read during reload.

    These tests verify entity indexing and routing from simulate-mode cache entries.
    """

    def test_empty_loop_cache_loads_cleanly(self):
        """Router with an empty store returns entity_count == 0."""
        router = QueryRouter(
            adapter_dir=Path("/nonexistent"),
            memory_store=MemoryStore(replay_enabled=False),
        )
        assert router.entity_count == 0

    def test_no_loop_provider_loads_cleanly(self):
        """Router with a freshly-constructed empty store returns entity_count == 0."""
        router = QueryRouter(
            adapter_dir=Path("/nonexistent"), memory_store=MemoryStore(replay_enabled=False)
        )
        assert router.entity_count == 0

    def test_episodic_cache_entities_indexed(self):
        """Entities from episodic cache entries appear in _all_entities."""
        router = _make_router_from_pairs(
            [_make_pair("graph1", "Alice", "Berlin", speaker_id="alice")],
        )
        assert "alice" in router._all_entities
        assert "berlin" in router._all_entities

    def test_semantic_and_procedural_cache_entries_indexed(self):
        """All three tiers' cache entries contribute to the index."""
        router = _make_router_from_pairs(
            [
                _make_pair("sem1", "Bob", "Python", speaker_id="bob", adapter_id="semantic"),
                _make_pair("proc1", "Carol", "Tea", speaker_id="carol", adapter_id="procedural"),
            ]
        )
        assert "bob" in router._all_entities
        assert "carol" in router._all_entities

    def test_speaker_id_from_cache_indexed(self):
        """speaker_id from cache entries appears in _speaker_key_index."""
        router = _make_router_from_pairs(
            [_make_pair("graph1", "Alice", "Berlin", speaker_id="alice")],
        )
        assert "alice" in router._speaker_key_index
        assert "graph1" in router._speaker_key_index["alice"]

    def test_cache_entry_produces_pa_routing_steps(self):
        """A query mentioning a cached entity resolves to a PA step."""
        from paramem.server.router import Intent

        router = _make_router_from_pairs(
            [_make_pair("graph1", "Alice", "Berlin", speaker_id="alice")],
        )
        plan = router.route("Where does Alice live?", speaker_id="alice")
        assert plan.steps, "Expected PA routing steps from cache entry"
        assert plan.intent == Intent.PERSONAL
        keys = [k for step in plan.steps for k in step.keys_to_probe]
        assert "graph1" in keys

    def test_multiple_tiers_in_cache_indexed_without_conflict(self):
        """Cache entries from different tiers are all indexed without conflict."""
        router = _make_router_from_pairs(
            [
                _make_pair("ep1", "TrainEntity", "Berlin", speaker_id="alice"),
                _make_pair(
                    "sem1", "SimEntity", "Munich", speaker_id="alice", adapter_id="semantic"
                ),
            ]
        )
        assert "trainentity" in router._all_entities
        assert "simentity" in router._all_entities

    def test_reload_picks_up_updated_cache(self):
        """reload() after mutating the injected store repopulates the entity index."""
        store = MemoryStore(replay_enabled=True)
        router = QueryRouter(adapter_dir=Path("/nonexistent"), memory_store=store)
        assert router.entity_count == 0

        store.put(
            "episodic",
            "graph1",
            {
                "key": "graph1",
                "subject": "Alice",
                "predicate": "related_to",
                "object": "Berlin",
                "speaker_id": "alice",
                "first_seen_cycle": 1,
            },
        )
        router.reload()
        assert router.entity_count > 0
        assert "alice" in router._all_entities

    def test_empty_cache_does_not_crash(self):
        """Empty store returns entity_count == 0 without raising."""
        store = MemoryStore(replay_enabled=True)
        router = QueryRouter(adapter_dir=Path("/nonexistent"), memory_store=store)
        assert router.entity_count == 0

    def test_has_email_predicate_indexed_from_cache(self):
        """has_email predicate in a cache entry is indexed under 'email' entity."""
        router = _make_router_from_pairs(
            [
                _make_pair(
                    "graph2",
                    "Alice",
                    "alice@example.com",
                    speaker_id="alice",
                    predicate="has_email",
                )
            ]
        )
        assert "email" in router._all_entities
