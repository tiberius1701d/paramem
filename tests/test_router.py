"""Unit tests for QueryRouter — speaker-only routing + intent-driven tier selection."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from paramem.memory.store import MemoryStore
from paramem.server.router import (
    Intent,
    QueryRouter,
    RoutingPlan,
    RoutingStep,
    _interim_sort_key,
    _is_interrogative,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(
    key: str,
    subject: str,
    obj: str,
    *,
    speaker_id: str = "",
    question: str = "Q?",
    answer: str = "A.",
    predicate: str = "related_to",
    adapter_id: str = "episodic",
) -> dict:
    """Return a canonical :class:`MemoryStore` entry dict for use in router tests."""
    return {
        "key": key,
        "question": question,
        "answer": answer,
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "speaker_id": speaker_id,
        "_adapter_id": adapter_id,  # internal: consumed by _make_router_from_entries
    }


def _make_router_from_entries(
    entries: list[dict],
    *,
    adapter_dir: "Path | None" = None,
    ha_graph: "MagicMock | None" = None,
    intent_config=None,
) -> "QueryRouter":
    """Create a QueryRouter backed by a seeded :class:`MemoryStore`.

    Each entry is written via ``MemoryStore.put`` (which registers the key
    in the per-tier ``KeyRegistry`` — the source ``_tier_keys`` reads) AND
    via ``store.set_bookkeeping`` when the entry carries ``speaker_id`` (the
    source ``reload()`` reads to build the speaker index).  ``_adapter_id``
    controls the tier.
    """
    store = MemoryStore(replay_enabled=True)
    for entry in entries:
        key = entry["key"]
        adapter_id = entry.get("_adapter_id", "episodic")
        payload = {k: v for k, v in entry.items() if k != "_adapter_id"}
        store.put(adapter_id, key, payload)
        # Mirror the production write: new-key store.put is always paired with
        # set_bookkeeping so the router's iter_bookkeeping() finds the speaker.
        spk = payload.get("speaker_id", "")
        rtype = payload.get("relation_type", "factual")
        if spk:
            store.set_bookkeeping(
                key,
                speaker_id=spk,
                relation_type=rtype,
                allow_empty_speaker=(spk == ""),
            )

    kwargs: dict = {
        "adapter_dir": adapter_dir or Path("/nonexistent"),
        "memory_store": store,
    }
    if ha_graph is not None:
        kwargs["ha_graph"] = ha_graph
    if intent_config is not None:
        kwargs["intent_config"] = intent_config
    return QueryRouter(**kwargs)


def _make_ha_match(
    *,
    entities: list[str] | None = None,
    areas: list[str] | None = None,
    verbs: list[str] | None = None,
    domains: list[str] | None = None,
) -> MagicMock:
    """Return a mock HAMatchResult.  ``has_entity_match`` mirrors
    ``ha_graph.HAMatchResult.has_entity_match`` — entities OR areas;
    verb-only matches do NOT satisfy ``has_entity_match``."""
    m = MagicMock()
    m.matched_entities = entities or []
    m.matched_areas = areas or []
    m.matched_verbs = verbs or []
    m.domains = domains or []
    m.has_entity_match = bool(entities or areas)
    m.has_verb_match = bool(verbs)
    return m


def _make_ha_graph(match_result: MagicMock | None) -> MagicMock:
    """Return a mock HAEntityGraph whose ``.match()`` returns *match_result*."""
    g = MagicMock()
    g.match.return_value = match_result
    return g


def _stub_intent(monkeypatch, verdict: Intent) -> MagicMock:
    """Stub ``paramem.server.intent.classify_intent`` to always return *verdict*.

    The router imports ``classify_intent`` lazily inside ``route()``, so
    patching the attribute on the intent module is sufficient.  Returns
    the mock so tests can assert on call args.
    """
    stub = MagicMock(return_value=verdict)
    monkeypatch.setattr("paramem.server.intent.classify_intent", stub)
    return stub


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
        assert _is_interrogative(text) is True

    def test_greek_semicolon_is_not_interrogative(self):
        """Greek uses ``;`` (U+003B) as a question mark, but the glyph is
        identical to the ASCII semicolon.  Treating it as interrogative
        would mis-classify any declarative sentence ending in ``;``."""
        assert _is_interrogative("Turn off the light;") is False

    def test_declarative_with_trailing_period_is_not_interrogative(self):
        assert _is_interrogative("Mache das Licht aus.") is False
        assert _is_interrogative("Schalte die Lampe an!") is False

    def test_encoder_tier_overrides_punctuation_when_decisive(self):
        from paramem.server.config import SentenceTypeConfig
        from paramem.server.sentence_type import SentenceType

        cfg = SentenceTypeConfig()
        with patch(
            "paramem.server.sentence_type.classify_sentence_type",
            return_value=SentenceType.INTERROGATIVE,
        ) as mock_classify:
            assert _is_interrogative("the door locked", config=cfg) is True
        mock_classify.assert_called_once()

    def test_encoder_tier_returns_false_when_classifier_says_non(self):
        from paramem.server.config import SentenceTypeConfig
        from paramem.server.sentence_type import SentenceType

        cfg = SentenceTypeConfig()
        with patch(
            "paramem.server.sentence_type.classify_sentence_type",
            return_value=SentenceType.NON_INTERROGATIVE,
        ):
            assert _is_interrogative("Berlin is in Germany?", config=cfg) is False

    def test_encoder_tier_falls_through_when_classifier_returns_none(self):
        from paramem.server.config import SentenceTypeConfig

        cfg = SentenceTypeConfig()
        with patch(
            "paramem.server.sentence_type.classify_sentence_type",
            return_value=None,
        ):
            assert _is_interrogative("Wo wohne ich?", config=cfg) is True
            assert _is_interrogative("What is my name", config=cfg) is True
            assert _is_interrogative("Mache das Licht an", config=cfg) is False


# ---------------------------------------------------------------------------
# _interim_sort_key
# ---------------------------------------------------------------------------


class TestInterimSortKey:
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
        assert _interim_sort_key("episodic_interim_20260417T0000_partial") is None


# ---------------------------------------------------------------------------
# QueryRouter — reload()
# ---------------------------------------------------------------------------


class TestRouterLoad:
    def test_empty_store_loads_cleanly(self):
        router = _make_router_from_entries([])
        assert router._speaker_key_index == {}

    def test_anonymous_entry_not_indexed(self):
        """Entries with empty speaker_id contribute no speaker mapping."""
        router = _make_router_from_entries(
            [_make_entry("graph1", "Alice", "Berlin", speaker_id="")]
        )
        assert router._speaker_key_index == {}

    def test_speaker_keys_indexed_flat_across_tiers(self):
        """One speaker's keys collapse across tiers in _speaker_key_index."""
        router = _make_router_from_entries(
            [
                _make_entry("ep1", "Alice", "Berlin", speaker_id="alice", adapter_id="episodic"),
                _make_entry("se1", "Alice", "Python", speaker_id="alice", adapter_id="semantic"),
                _make_entry("pr1", "Alice", "Cycling", speaker_id="alice", adapter_id="procedural"),
            ]
        )
        assert router._speaker_key_index == {"alice": {"ep1", "se1", "pr1"}}

    def test_multiple_speakers_get_separate_key_sets(self):
        router = _make_router_from_entries(
            [
                _make_entry("k1", "Alice", "Berlin", speaker_id="alice"),
                _make_entry("k2", "Bob", "Munich", speaker_id="bob"),
            ]
        )
        assert router._speaker_key_index["alice"] == {"k1"}
        assert router._speaker_key_index["bob"] == {"k2"}

    def test_reload_clears_stale_state(self):
        router = _make_router_from_entries(
            [_make_entry("k1", "Alice", "Berlin", speaker_id="alice")]
        )
        assert "alice" in router._speaker_key_index

        # Swap the store for an empty one; reload must clear stale state.
        router._memory_store = MemoryStore(replay_enabled=True)
        router.reload()

        assert router._speaker_key_index == {}


# ---------------------------------------------------------------------------
# QueryRouter — speaker scoping (privacy boundary)
# ---------------------------------------------------------------------------


class TestSpeakerScoping:
    """allowed_keys is the privacy boundary.  No speaker → no steps; an
    unknown speaker_id → no steps; an enrolled speaker can only reach
    their own keys."""

    def test_anonymous_speaker_produces_empty_steps(self, monkeypatch):
        # Even when intent says PERSONAL, no speaker_id → empty steps.
        _stub_intent(monkeypatch, Intent.PERSONAL)
        router = _make_router_from_entries(
            [_make_entry("k1", "Alice", "Berlin", speaker_id="alice")]
        )
        plan = router.route("Where do I live?", speaker_id=None)
        assert plan.steps == []
        assert plan.strategy == "direct"

    def test_unknown_speaker_id_produces_empty_steps(self, monkeypatch):
        _stub_intent(monkeypatch, Intent.PERSONAL)
        router = _make_router_from_entries(
            [_make_entry("k1", "Alice", "Berlin", speaker_id="alice")]
        )
        plan = router.route("Where do I live?", speaker_id="bob")
        assert plan.steps == []

    def test_enrolled_speaker_only_reaches_own_keys(self, monkeypatch):
        _stub_intent(monkeypatch, Intent.PERSONAL)
        router = _make_router_from_entries(
            [
                _make_entry("k_alice", "Alice", "Berlin", speaker_id="alice"),
                _make_entry("k_bob", "Bob", "Munich", speaker_id="bob"),
            ]
        )
        plan = router.route("anything", speaker_id="alice")
        all_keys = [k for step in plan.steps for k in step.keys_to_probe]
        assert all_keys == ["k_alice"]


# ---------------------------------------------------------------------------
# QueryRouter — intent-driven tier selection
# ---------------------------------------------------------------------------


class TestIntentTierSelection:
    """Tier table:

    * PERSONAL → procedural → newest interim first → episodic → semantic
    * COMMAND  → procedural only (preferences for HA injection)
    * GENERAL  → no steps (route to SOTA without personal injection)
    * UNKNOWN  → resolved via IntentConfig.fail_closed_intent (default PERSONAL)
    """

    def _three_tier_router(self):
        return _make_router_from_entries(
            [
                _make_entry("ep1", "Alice", "Berlin", speaker_id="alice", adapter_id="episodic"),
                _make_entry("se1", "Alice", "Python", speaker_id="alice", adapter_id="semantic"),
                _make_entry("pr1", "Alice", "Cycling", speaker_id="alice", adapter_id="procedural"),
            ]
        )

    def test_personal_selects_all_three_main_tiers_in_order(self, monkeypatch):
        _stub_intent(monkeypatch, Intent.PERSONAL)
        router = self._three_tier_router()
        plan = router.route("Where do I live?", speaker_id="alice")
        assert [s.adapter_name for s in plan.steps] == ["procedural", "episodic", "semantic"]
        assert plan.intent is Intent.PERSONAL
        assert plan.strategy == "targeted_probe"

    def test_command_selects_procedural_only(self, monkeypatch):
        _stub_intent(monkeypatch, Intent.COMMAND)
        router = self._three_tier_router()
        plan = router.route("Play HR3 Radio.", speaker_id="alice")
        assert [s.adapter_name for s in plan.steps] == ["procedural"]
        assert plan.intent is Intent.COMMAND

    def test_command_never_exposes_episodic_or_semantic(self, monkeypatch):
        """Privacy invariant: COMMAND must NOT include identity facts.

        Episodic / semantic tiers carry the speaker's biographical context;
        COMMAND queries go to HA payloads which leave the local trust zone.
        Procedural (preferences) is the only tier safe to inject.
        """
        _stub_intent(monkeypatch, Intent.COMMAND)
        router = self._three_tier_router()
        plan = router.route("Play music", speaker_id="alice")
        tiers = {s.adapter_name for s in plan.steps}
        assert "episodic" not in tiers
        assert "semantic" not in tiers

    def test_general_produces_empty_steps(self, monkeypatch):
        _stub_intent(monkeypatch, Intent.GENERAL)
        router = self._three_tier_router()
        plan = router.route("What's the capital of France?", speaker_id="alice")
        assert plan.steps == []
        assert plan.intent is Intent.GENERAL
        assert plan.strategy == "direct"

    def test_unknown_without_config_defaults_to_personal(self, monkeypatch):
        _stub_intent(monkeypatch, Intent.UNKNOWN)
        router = self._three_tier_router()
        plan = router.route("ambiguous", speaker_id="alice")
        # No intent_config → PERSONAL fallback.
        assert [s.adapter_name for s in plan.steps] == ["procedural", "episodic", "semantic"]

    def test_unknown_resolves_via_intent_config_fail_closed(self, monkeypatch):
        from paramem.server.config import IntentConfig

        _stub_intent(monkeypatch, Intent.UNKNOWN)
        router = _make_router_from_entries(
            [
                _make_entry("ep1", "Alice", "Berlin", speaker_id="alice", adapter_id="episodic"),
                _make_entry("pr1", "Alice", "Cycling", speaker_id="alice", adapter_id="procedural"),
            ],
            intent_config=IntentConfig(fail_closed_intent="general"),
        )
        plan = router.route("ambiguous", speaker_id="alice")
        # general → no steps
        assert plan.steps == []

    def test_tier_missing_from_store_is_skipped(self, monkeypatch):
        """Speaker has only procedural keys → PERSONAL plan skips the
        empty episodic/semantic tiers without raising."""
        _stub_intent(monkeypatch, Intent.PERSONAL)
        router = _make_router_from_entries(
            [_make_entry("pr1", "Alice", "Cycling", speaker_id="alice", adapter_id="procedural")]
        )
        plan = router.route("hi", speaker_id="alice")
        assert [s.adapter_name for s in plan.steps] == ["procedural"]


# ---------------------------------------------------------------------------
# QueryRouter — COMMAND path privacy filter
# ---------------------------------------------------------------------------


class TestCommandInterimPrivacy:
    """Regression: COMMAND path emits preference keys from interim slots
    but NEVER emits factual/temporal/social interim keys.

    The router reaches interim slots via ``_command_interim_tiers()`` which
    reuses the same ``tiers_with_registry()`` enumeration as the PERSONAL
    path.  Per-key filtering is applied by ``_tier_keys(preference_only=True)``:
    only keys whose name starts with "proc" OR whose bookkeeping
    ``relation_type`` is "preference" are included.  All other relation types
    are excluded by construction (DEFAULT-DENY).
    """

    def _command_router_with_interim(
        self, *, factual_key: bool = True, preference_key: bool = True, proc_main_key: bool = True
    ):
        """Build a router with procedural MAIN + one interim slot.

        The interim slot contains:
        - graph1 (factual, relation_type="factual") when factual_key=True
        - proc1  (preference, relation_type="preference") when preference_key=True
        The procedural MAIN contains:
        - pref0  (preference, proc-prefix key) when proc_main_key=True
        """
        entries = []
        if proc_main_key:
            entries.append(
                {
                    **_make_entry(
                        "pref0",
                        "Alice",
                        "Jazz",
                        speaker_id="alice",
                        adapter_id="procedural",
                        predicate="likes",
                    ),
                    "relation_type": "preference",
                }
            )
        if factual_key:
            entries.append(
                {
                    **_make_entry(
                        "graph1",
                        "Alice",
                        "London",
                        speaker_id="alice",
                        adapter_id="episodic_interim_20260618T0800",
                        predicate="lives_in",
                    ),
                    "relation_type": "factual",
                }
            )
        if preference_key:
            entries.append(
                {
                    **_make_entry(
                        "proc2",
                        "Alice",
                        "Tea",
                        speaker_id="alice",
                        adapter_id="episodic_interim_20260618T0800",
                        predicate="prefers",
                    ),
                    "relation_type": "preference",
                }
            )
        return _make_router_from_entries(entries)

    def test_factual_interim_key_never_emitted_on_command(self, monkeypatch):
        """Privacy invariant: COMMAND path must NOT emit factual interim keys.

        graph1 (relation_type="factual") in an interim slot carries biographical
        context (where Alice lives) that must not reach HA payloads.
        """
        _stub_intent(monkeypatch, Intent.COMMAND)
        router = self._command_router_with_interim(
            factual_key=True, preference_key=False, proc_main_key=False
        )
        plan = router.route("Turn off the lights.", speaker_id="alice")
        all_keys = [k for step in plan.steps for k in step.keys_to_probe]
        assert "graph1" not in all_keys, (
            "Factual interim key 'graph1' must never appear in COMMAND steps; "
            f"got steps: {[(s.adapter_name, s.keys_to_probe) for s in plan.steps]}"
        )

    def test_preference_interim_key_emitted_on_command(self, monkeypatch):
        """COMMAND path emits preference-typed interim keys.

        Keys with proc-prefix or relation_type="preference" in an interim slot
        carry style context (Alice prefers Tea) that is safe for HA payloads.
        """
        _stub_intent(monkeypatch, Intent.COMMAND)
        router = self._command_router_with_interim(
            factual_key=False, preference_key=True, proc_main_key=False
        )
        plan = router.route("Set the mood.", speaker_id="alice")
        all_keys = [k for step in plan.steps for k in step.keys_to_probe]
        assert "proc2" in all_keys, (
            "Preference-typed interim key 'proc2' must appear in COMMAND steps; "
            f"got steps: {[(s.adapter_name, s.keys_to_probe) for s in plan.steps]}"
        )

    def test_proc_main_keys_unfiltered_on_command(self, monkeypatch):
        """Procedural MAIN keys are always unfiltered on the COMMAND path.

        pref0 (in procedural MAIN) must appear regardless of relation_type
        — the preference_only filter applies ONLY to interim slots.
        """
        _stub_intent(monkeypatch, Intent.COMMAND)
        router = self._command_router_with_interim(
            factual_key=False, preference_key=False, proc_main_key=True
        )
        plan = router.route("Adjust lights.", speaker_id="alice")
        all_keys = [k for step in plan.steps for k in step.keys_to_probe]
        assert "pref0" in all_keys, (
            "Procedural MAIN key 'pref0' must appear in COMMAND steps (unfiltered); "
            f"got steps: {[(s.adapter_name, s.keys_to_probe) for s in plan.steps]}"
        )

    def test_command_interim_step_is_for_interim_adapter_not_procedural(self, monkeypatch):
        """Interim preference keys appear under the interim adapter name, not 'procedural'."""
        _stub_intent(monkeypatch, Intent.COMMAND)
        router = self._command_router_with_interim(
            factual_key=False, preference_key=True, proc_main_key=False
        )
        plan = router.route("What music?", speaker_id="alice")
        interim_steps = [s for s in plan.steps if s.adapter_name.startswith("episodic_interim_")]
        assert interim_steps, (
            "Interim preference key must appear in an episodic_interim_* step; "
            f"got steps: {[(s.adapter_name, s.keys_to_probe) for s in plan.steps]}"
        )
        assert "proc2" in interim_steps[0].keys_to_probe, (
            f"proc2 must be in the interim step's keys; got {interim_steps[0].keys_to_probe}"
        )

    def test_command_factual_and_preference_keys_together(self, monkeypatch):
        """When interim has both factual and preference keys, only preference is emitted."""
        _stub_intent(monkeypatch, Intent.COMMAND)
        router = self._command_router_with_interim(
            factual_key=True, preference_key=True, proc_main_key=False
        )
        plan = router.route("Turn on lights.", speaker_id="alice")
        all_keys = [k for step in plan.steps for k in step.keys_to_probe]
        assert "graph1" not in all_keys, (
            "Factual key 'graph1' must be excluded from COMMAND interim step"
        )
        assert "proc2" in all_keys, (
            "Preference key 'proc2' must be included in COMMAND interim step"
        )

    def test_command_preference_or_branch_graph_prefix_with_preference_relation_type(
        self, monkeypatch
    ):
        """OR-branch: a graph-prefixed key (NO proc prefix) with relation_type='preference'
        IS admitted on the COMMAND interim path.

        The filter is: key.startswith("proc") OR relation_type=="preference".
        Both branches must be exercised; this test locks the standalone
        relation_type=="preference" branch that proc-prefix keys never exercise.
        """
        _stub_intent(monkeypatch, Intent.COMMAND)
        # graph-prefixed key, stored in an interim slot, relation_type="preference"
        pref_graph_entry = {
            **_make_entry(
                "graph42",
                "Alice",
                "Jazz",
                speaker_id="alice",
                adapter_id="episodic_interim_20260618T0800",
                predicate="enjoys",
            ),
            "relation_type": "preference",
        }
        router = _make_router_from_entries([pref_graph_entry])
        plan = router.route("Play something.", speaker_id="alice")
        all_keys = [k for step in plan.steps for k in step.keys_to_probe]
        assert "graph42" in all_keys, (
            "graph-prefixed key with relation_type='preference' must be admitted on "
            f"COMMAND interim path (OR-branch); got steps: "
            f"{[(s.adapter_name, s.keys_to_probe) for s in plan.steps]}"
        )

    def test_command_or_branch_graph_prefix_factual_excluded(self, monkeypatch):
        """OR-branch negative: a graph-prefixed key with relation_type='factual' is NOT admitted.

        Confirms DEFAULT-DENY: only the preference OR-branch grants access;
        any other relation_type is excluded even without the proc prefix.
        """
        _stub_intent(monkeypatch, Intent.COMMAND)
        factual_graph_entry = {
            **_make_entry(
                "graph99",
                "Alice",
                "London",
                speaker_id="alice",
                adapter_id="episodic_interim_20260618T0800",
                predicate="lives_in",
            ),
            "relation_type": "factual",
        }
        router = _make_router_from_entries([factual_graph_entry])
        plan = router.route("Turn off all lights.", speaker_id="alice")
        all_keys = [k for step in plan.steps for k in step.keys_to_probe]
        assert "graph99" not in all_keys, (
            "graph-prefixed key with relation_type='factual' must be excluded from "
            f"COMMAND interim path; got steps: "
            f"{[(s.adapter_name, s.keys_to_probe) for s in plan.steps]}"
        )


# ---------------------------------------------------------------------------
# QueryRouter — PERSONAL probe order with interim adapters
# ---------------------------------------------------------------------------


class TestPersonalTierOrder:
    def test_interim_adapters_sort_newest_first(self, monkeypatch):
        _stub_intent(monkeypatch, Intent.PERSONAL)
        router = _make_router_from_entries(
            [
                _make_entry("ep1", "Alice", "Berlin", speaker_id="alice", adapter_id="episodic"),
                _make_entry("pr1", "Alice", "Cycling", speaker_id="alice", adapter_id="procedural"),
                _make_entry(
                    "i0",
                    "Alice",
                    "X",
                    speaker_id="alice",
                    adapter_id="episodic_interim_20260415T0000",
                ),
                _make_entry(
                    "i1",
                    "Alice",
                    "Y",
                    speaker_id="alice",
                    adapter_id="episodic_interim_20260417T0000",
                ),
                _make_entry(
                    "i2",
                    "Alice",
                    "Z",
                    speaker_id="alice",
                    adapter_id="episodic_interim_20260416T0000",
                ),
            ]
        )
        plan = router.route("anything", speaker_id="alice")
        names = [s.adapter_name for s in plan.steps]
        assert names == [
            "procedural",
            "episodic_interim_20260417T0000",  # newest first
            "episodic_interim_20260416T0000",
            "episodic_interim_20260415T0000",
            "episodic",
        ]

    def test_interim_names_are_not_normalised(self, monkeypatch):
        """Probe-time switch_adapter() must find the trained slot under its
        full canonical name including the stamp.  Stripping the stamp would
        miss the slot in model.peft_config."""
        _stub_intent(monkeypatch, Intent.PERSONAL)
        router = _make_router_from_entries(
            [
                _make_entry(
                    "i0",
                    "Alice",
                    "X",
                    speaker_id="alice",
                    adapter_id="episodic_interim_20260415T0000",
                )
            ]
        )
        plan = router.route("anything", speaker_id="alice")
        assert any(s.adapter_name == "episodic_interim_20260415T0000" for s in plan.steps)

    def test_keys_to_probe_are_sorted(self, monkeypatch):
        """Deterministic key ordering — no hash-randomisation flake."""
        _stub_intent(monkeypatch, Intent.PERSONAL)
        router = _make_router_from_entries(
            [
                _make_entry(
                    f"k{i:03d}", "Alice", f"x{i}", speaker_id="alice", adapter_id="episodic"
                )
                for i in range(10)
            ]
        )
        plan = router.route("anything", speaker_id="alice")
        for step in plan.steps:
            assert step.keys_to_probe == sorted(step.keys_to_probe)


# ---------------------------------------------------------------------------
# QueryRouter — HA fast path
# ---------------------------------------------------------------------------


class TestRouteHA:
    def test_ha_entity_match_drives_command(self):
        """has_ha_match=True → classify_intent returns COMMAND.

        Verified end-to-end through the real classify_intent (no encoder
        loaded so the residual is bypassed; HA fast-path fires).
        """
        ha_match = _make_ha_match(entities=["kitchen light"], domains=["light"])
        router = _make_router_from_entries(
            [_make_entry("pr1", "Alice", "Cycling", speaker_id="alice", adapter_id="procedural")],
            ha_graph=_make_ha_graph(ha_match),
        )
        plan = router.route("Turn on the kitchen light", speaker_id="alice")
        assert plan.intent is Intent.COMMAND
        assert plan.ha_domains == ["light"]
        # COMMAND → procedural only
        assert [s.adapter_name for s in plan.steps] == ["procedural"]

    def test_ha_verb_only_does_not_satisfy_fast_path(self, monkeypatch):
        """has_entity_match requires entity OR area; verb-only is not enough.

        ``Play music`` against an HA install with no media_player entity
        named ``music`` matches only the ``play media`` verb.  The HA fast
        path does not fire; intent falls to the encoder residual (stubbed
        here as GENERAL for the test).
        """
        _stub_intent(monkeypatch, Intent.GENERAL)
        ha_match = _make_ha_match(verbs=["play media"], domains=["media_player"])
        router = _make_router_from_entries(
            [_make_entry("pr1", "Alice", "Cycling", speaker_id="alice", adapter_id="procedural")],
            ha_graph=_make_ha_graph(ha_match),
        )
        plan = router.route("Play music", speaker_id="alice")
        # ha_match exists but has_entity_match is False → encoder takes over.
        # Stubbed encoder verdict GENERAL → no steps.
        assert plan.intent is Intent.GENERAL
        assert plan.steps == []
        # ha_domains is still surfaced for observability.
        assert plan.ha_domains == ["media_player"]

    def test_no_ha_graph_short_path(self, monkeypatch):
        """Router constructed without ha_graph: has_ha is always False."""
        _stub_intent(monkeypatch, Intent.PERSONAL)
        router = _make_router_from_entries(
            [_make_entry("ep1", "Alice", "Berlin", speaker_id="alice", adapter_id="episodic")]
        )
        plan = router.route("Where do I live?", speaker_id="alice")
        assert plan.ha_domains == []


# ---------------------------------------------------------------------------
# QueryRouter — RoutingPlan shape
# ---------------------------------------------------------------------------


class TestRoutingPlanShape:
    def test_defaults(self):
        plan = RoutingPlan()
        assert plan.steps == []
        assert plan.strategy == "direct"
        assert plan.ha_domains == []
        assert plan.intent is Intent.UNKNOWN

    def test_strategy_reflects_steps(self, monkeypatch):
        _stub_intent(monkeypatch, Intent.PERSONAL)
        router = _make_router_from_entries(
            [_make_entry("ep1", "Alice", "Berlin", speaker_id="alice", adapter_id="episodic")]
        )
        plan = router.route("anything", speaker_id="alice")
        assert plan.strategy == "targeted_probe"
        assert plan.steps != []

    def test_routing_step_dataclass(self):
        step = RoutingStep(adapter_name="episodic", keys_to_probe=["a", "b"])
        assert step.adapter_name == "episodic"
        assert step.keys_to_probe == ["a", "b"]


# ---------------------------------------------------------------------------
# QueryRouter — preload independence
# ---------------------------------------------------------------------------


class TestPreloadIndependence:
    """``_speaker_key_index`` must populate even when ``_entries`` (content cache)
    is empty — the preload-off boot scenario.

    The router's :meth:`reload` iterates :meth:`MemoryStore.iter_bookkeeping`,
    NOT ``iter_entries``.  :meth:`MemoryStore.load_bookkeeping_from_disk`
    populates ``_bookkeeping`` at boot unconditionally, regardless of
    ``inference.preload_cache``.

    These tests seed ``_bookkeeping`` directly via ``set_bookkeeping`` (no
    ``put``, so ``_entries`` stays empty) and assert the router indexes from it.
    """

    def test_metadata_only_entries_populate_speaker_index(self):
        store = MemoryStore(replay_enabled=True)
        # Simulate load_bookkeeping_from_disk: no SPO in _entries, just bookkeeping.
        store.set_bookkeeping("k_meta_only", speaker_id="alice", relation_type="factual")
        # Register the key so _tier_keys resolves it.
        from paramem.training.key_registry import KeyRegistry

        reg = KeyRegistry()
        reg.add("k_meta_only")
        store.load_registry("episodic", reg)
        assert store.get("k_meta_only") is None  # _entries is empty
        router = QueryRouter(adapter_dir=Path("/nonexistent"), memory_store=store)
        assert router._speaker_key_index["alice"] == {"k_meta_only"}

    def test_metadata_only_routes_produce_steps(self, monkeypatch):
        _stub_intent(monkeypatch, Intent.PERSONAL)
        store = MemoryStore(replay_enabled=True)
        # Seed bookkeeping only — no content entry.
        store.set_bookkeeping("k_pref", speaker_id="alice", relation_type="preference")
        from paramem.training.key_registry import KeyRegistry

        reg = KeyRegistry()
        reg.add("k_pref")
        store.load_registry("procedural", reg)
        assert store.get("k_pref") is None  # _entries is empty
        router = QueryRouter(adapter_dir=Path("/nonexistent"), memory_store=store)
        plan = router.route("anything", speaker_id="alice")
        assert [s.adapter_name for s in plan.steps] == ["procedural"]
        assert plan.steps[0].keys_to_probe == ["k_pref"]

    def test_router_index_built_under_empty_entries(self):
        """Even with no content entries, router must index speakers from bookkeeping."""
        store = MemoryStore(replay_enabled=True)
        for i in range(3):
            key = f"graph{i}"
            store.set_bookkeeping(key, speaker_id="charlie", relation_type="factual")
        assert len(store) == 0  # _entries empty
        router = QueryRouter(adapter_dir=Path("/nonexistent"), memory_store=store)
        assert "charlie" in router._speaker_key_index
        assert len(router._speaker_key_index["charlie"]) == 3


# ---------------------------------------------------------------------------
# QueryRouter — HR3 regression
# ---------------------------------------------------------------------------


class TestHR3Regression:
    """Speaker enrollment must NOT classify the speaker's queries as PERSONAL.

    Prior code injected the speaker as an implicit entity and did one-hop
    graph expansion, both of which made ``has_pa=True`` for essentially
    every query from an enrolled speaker.  Combined with the now-removed
    ``has_graph_match → PERSONAL`` short-circuit, imperatives like
    ``"Play HR3 Radio."`` routed to the personal-adapter PA path instead
    of HA.

    These tests confirm: the new code passes only ``has_ha_match`` to
    ``classify_intent``; enrollment plays no role in intent.
    """

    def test_classify_intent_call_does_not_receive_graph_match_kwarg(self, monkeypatch):
        stub = _stub_intent(monkeypatch, Intent.COMMAND)
        router = _make_router_from_entries(
            [_make_entry("ep1", "Alice", "Berlin", speaker_id="alice", adapter_id="episodic")]
        )
        router.route("Play HR3 Radio.", speaker_id="alice")
        # The router must pass has_ha_match only; has_graph_match is gone.
        call = stub.call_args
        assert "has_graph_match" not in call.kwargs
        assert "has_ha_match" in call.kwargs

    def test_imperative_from_enrolled_speaker_with_ha_match_is_command(self):
        """End-to-end: HA-match imperative from enrolled speaker → COMMAND.

        Uses the real ``classify_intent`` (no encoder loaded, no graph-match
        signal).  Speaker is enrolled with many keys; intent must still be
        COMMAND, never PERSONAL.
        """
        ha_match = _make_ha_match(entities=["kitchen light"], domains=["light"])
        router = _make_router_from_entries(
            [
                _make_entry(f"k{i}", "Alice", f"fact{i}", speaker_id="alice", adapter_id="episodic")
                for i in range(50)
            ],
            ha_graph=_make_ha_graph(ha_match),
        )
        plan = router.route("Turn on the kitchen light", speaker_id="alice")
        assert plan.intent is Intent.COMMAND


# ---------------------------------------------------------------------------
# Contract: _speaker_key_index key casing matches route() lookup key casing
# ---------------------------------------------------------------------------


class TestSpeakerKeyIndexCasingContract:
    """``_speaker_key_index`` keys and the ``route()`` lookup key must share casing.

    Both are lowercase under the speaker-identity refactor.  A cased lookup
    (``route(speaker_id="Speaker0")``) would miss a lowercase-indexed key
    (``_speaker_key_index["speaker0"]``), producing an empty routing plan and
    silently losing all that speaker's keys.

    This test locks the invariant: the index keys match what ``route()`` receives.
    """

    def test_index_keys_are_lowercase(self):
        """After reload, all _speaker_key_index keys are lowercase speaker{N} ids."""
        router = _make_router_from_entries(
            [
                _make_entry("ep1", "Alex", "Berlin", speaker_id="speaker0", adapter_id="episodic"),
                _make_entry("se1", "Alex", "Python", speaker_id="speaker0", adapter_id="semantic"),
            ]
        )
        for key in router._speaker_key_index:
            assert key == key.lower(), (
                f"_speaker_key_index key {key!r} is not lowercase — "
                "route(speaker_id=lowercase) would miss this speaker's keys."
            )
        assert "speaker0" in router._speaker_key_index, (
            "_speaker_key_index must contain 'speaker0' after seeding lowercase speaker_id."
        )

    def test_lowercase_route_lookup_finds_indexed_keys(self):
        """A lowercase route() speaker_id resolves the speaker's keys from the index."""
        router = _make_router_from_entries(
            [
                _make_entry("ep1", "Alex", "Berlin", speaker_id="speaker0", adapter_id="episodic"),
                _make_entry("se1", "Alex", "Python", speaker_id="speaker0", adapter_id="semantic"),
            ]
        )
        # The index is keyed by lowercase 'speaker0'; the route() lookup must
        # use the same casing so keys are not silently dropped.
        assert router._speaker_key_index.get("speaker0") == {"ep1", "se1"}, (
            "route(speaker_id='speaker0') must find both keys; "
            "a cased 'Speaker0' lookup would return None and drop the facts."
        )
