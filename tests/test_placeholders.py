"""Unit tests for paramem.graph.placeholders — the anonymize <-> deanonymize
placeholder contract module.

Most of this module's functions (``_apply_bindings``, ``_resolution_map``,
``_build_anonymization_mapping``, ...) already have extensive coverage in
``tests/test_extraction_pipeline.py`` (moved there unchanged when this
module was carved out of ``paramem.graph.extractor``). This file covers
the NEW unified primitives introduced by that carve-out: ``mint_placeholder``,
``braced``, ``entity_type_to_prefix``, ``prefix_to_entity_type``, and the
generalized (``placeholder_side``) table normalize/validate pair.
"""

from __future__ import annotations

from paramem.graph.name_match import is_speaker_id
from paramem.graph.placeholders import (
    PLACEHOLDER_SHAPE_RE,
    _build_anonymization_mapping,
    _mapping_is_canonical,
    _normalize_anonymization_mapping,
    _substitute_whole_words,
    braced,
    deanonymize_text,
    entity_type_to_prefix,
    mint_placeholder,
    placeholder_entity_type,
    prefix_to_entity_type,
)
from paramem.graph.schema import Entity, SessionGraph


class TestMintPlaceholder:
    def test_mints_first_index_on_empty_table(self):
        assert mint_placeholder([], "Person") == "Person_1"

    def test_scans_existing_values_for_next_free_index(self):
        assert mint_placeholder(["Person_1", "Person_2", "Org_1"], "Person") == "Person_3"

    def test_ignores_non_string_and_other_prefix_values(self):
        assert mint_placeholder(["Org_1", None, 42, "Person_5"], "Person") == "Person_6"

    def test_blind_to_llm_hint_is_avoided_by_scanning_the_caller_supplied_table(self):
        """The whole point of collapsing onto ONE scan-the-map mint (vs a
        counter blind to LLM-emitted hints already in the table): a caller
        who merges LLM hints into the values it passes gets a
        non-colliding mint back."""
        merged = {"Alex": "Person_1", "Riley": "Person_2"}  # e.g. an LLM hint
        assert mint_placeholder(merged.values(), "Person") == "Person_3"


class TestBraced:
    def test_wraps_bare_token(self):
        assert braced("Person_1") == "{Person_1}"

    def test_does_not_double_wrap(self):
        # Not a claimed invariant, but pins the literal behaviour: braced()
        # is a pure wrap, not idempotent — callers never feed it a
        # pre-braced token.
        assert braced("{Person_1}") == "{{Person_1}}"


class TestEntityTypeToPrefix:
    def test_closed_vocabulary_matches_schema_config(self):
        assert entity_type_to_prefix("person") == "Person"
        assert entity_type_to_prefix("place") == "City"
        assert entity_type_to_prefix("organization") == "Org"
        assert entity_type_to_prefix("concept") == "Thing"

    def test_open_vocabulary_pascal_cases_multi_word_labels(self):
        """Collapses the old two-implementation drift: the PascalCase
        rule wins over the ``.capitalize()`` rule for any open type."""
        assert entity_type_to_prefix("event") == "Event"
        assert entity_type_to_prefix("work_of_art") == "WorkOfArt"
        assert entity_type_to_prefix("self-driving") == "SelfDriving"
        assert entity_type_to_prefix("law enforcement") == "LawEnforcement"

    def test_empty_or_blank_falls_back_to_entity(self):
        assert entity_type_to_prefix("") == "Entity"
        assert entity_type_to_prefix("   ") == "Entity"
        assert entity_type_to_prefix(None) == "Entity"


class TestPrefixToEntityType:
    def test_closed_vocabulary_matches_schema_config(self):
        assert prefix_to_entity_type("City") == "place"
        assert prefix_to_entity_type("Org") == "organization"
        assert prefix_to_entity_type("Person") == "person"
        assert prefix_to_entity_type("Thing") == "concept"

    def test_open_vocabulary_derives_type_from_prefix_itself(self):
        """The open policy (SOTA's brace-binding protocol: the prefix IS
        the type name for a novel entity) — matches the pre-refactor
        behaviour of the entity-rebuild loop in ``_sota_pipeline``, now
        also applied by ``entity_correction.correct_entity_surfaces``."""
        assert prefix_to_entity_type("Project") == "project"
        assert prefix_to_entity_type("Language") == "language"

    def test_case_insensitive_lookup(self):
        assert prefix_to_entity_type("ORG") == "organization"
        assert prefix_to_entity_type("org") == "organization"

    def test_empty_prefix_falls_back_to_concept(self):
        assert prefix_to_entity_type("") == "concept"
        assert prefix_to_entity_type(None) == "concept"


class TestPlaceholderEntityType:
    """F5 — the ONE site deriving an entity type from a placeholder TOKEN
    (brace-tolerant), collapsing the three previously-duplicated inline
    ``prefix_to_entity_type(placeholder.split("_")[0])`` derivations in
    ``consolidation.py``/``extractor.py``/``entity_correction.py``.
    """

    def test_bare_token_matches_prefix_to_entity_type(self):
        assert placeholder_entity_type("Person_1") == "person"
        assert placeholder_entity_type("Org_3") == "organization"
        assert placeholder_entity_type("City_2") == "place"

    def test_open_vocabulary_bare_token(self):
        assert placeholder_entity_type("Project_1") == "project"

    def test_braced_token_still_derives_correct_type(self):
        """The braced format flip (bare ``Person_1`` -> braced
        ``{Person_1}``) must not silently unmask a person.

        Mutation: revert to ``prefix_to_entity_type(token.split("_")[0])``
        without stripping braces first -> ``"{Person_1}".split("_")[0]`` is
        ``"{Person"``, which is not in the closed vocabulary and passes
        through open-vocabulary as its own (wrong) type ``"{person"`` —
        this test fails (``"{person" != "person"``).
        """
        assert placeholder_entity_type("{Person_1}") == "person"
        assert placeholder_entity_type("{Org_3}") == "organization"
        assert placeholder_entity_type("{Project_1}") == "project"

    def test_empty_or_none_falls_back_to_concept(self):
        assert placeholder_entity_type("") == "concept"
        assert placeholder_entity_type(None) == "concept"


class TestNormalizeAndValidateTableBothDirections:
    """The single normalize/validate primitive, generalized by
    ``placeholder_side`` to serve both the CORE anonymizer table
    (``{real_name: placeholder}``, ``placeholder_side="value"``, the
    default) and the SOTA ``bindings`` table (``{placeholder: real_text}``,
    ``placeholder_side="key"`` — the OPPOSITE direction).
    """

    def test_core_table_default_direction_unchanged(self):
        mapping, stats = _normalize_anonymization_mapping({"Alex": "Person_1"})
        assert mapping == {"Alex": "Person_1"}
        assert stats == {"inverted": 0, "dropped": 0}

    def test_core_table_inverted_pair_is_corrected(self):
        mapping, stats = _normalize_anonymization_mapping({"Person_1": "Alex"})
        assert mapping == {"Alex": "Person_1"}
        assert stats["inverted"] == 1

    def test_bindings_table_correct_direction_kept_as_is(self):
        mapping, stats = _normalize_anonymization_mapping(
            {"Event_1": "the agile transformation initiative"}, placeholder_side="key"
        )
        assert mapping == {"Event_1": "the agile transformation initiative"}
        assert stats == {"inverted": 0, "dropped": 0}

    def test_bindings_table_inverted_pair_is_corrected(self):
        """The exact bug this generalization closes: an inverted binding
        (real text as key, placeholder as value) is corrected to
        canonical ``{placeholder: real_text}`` direction rather than
        passed straight through."""
        mapping, stats = _normalize_anonymization_mapping({"Acme": "Org_9"}, placeholder_side="key")
        assert mapping == {"Org_9": "Acme"}
        assert stats["inverted"] == 1

    def test_bindings_table_neither_side_shaped_is_dropped(self):
        mapping, stats = _normalize_anonymization_mapping(
            {"my company": "Acme Corp"}, placeholder_side="key"
        )
        assert mapping == {}
        assert stats["dropped"] == 1

    def test_bindings_table_both_sides_shaped_ties_to_declared_key_side(self):
        """FIX 3: a binding where both sides happen to be placeholder-
        shaped (e.g. `GPT_4`, a real model name) is not ambiguous — the
        declared `placeholder_side` breaks the tie rather than the
        entry being dropped and the whole delta rejected."""
        mapping, stats = _normalize_anonymization_mapping(
            {"Model_1": "GPT_4"}, placeholder_side="key"
        )
        assert mapping == {"Model_1": "GPT_4"}
        assert stats == {"inverted": 0, "dropped": 0}

    def test_core_table_both_sides_shaped_ties_to_declared_value_side(self):
        """Same tie-break, CORE table direction (`placeholder_side="value"`,
        the default)."""
        mapping, stats = _normalize_anonymization_mapping({"Person_2": "Windows_11"})
        assert mapping == {"Person_2": "Windows_11"}
        assert stats == {"inverted": 0, "dropped": 0}

    def test_core_table_is_canonical_default_direction(self):
        assert _mapping_is_canonical({"Alex": "Person_1"}) is True
        assert _mapping_is_canonical({"Person_1": "Alex"}) is False

    def test_bindings_table_is_canonical_key_direction(self):
        assert _mapping_is_canonical({"Event_1": "the trip"}, placeholder_side="key") is True
        assert _mapping_is_canonical({"the trip": "Event_1"}, placeholder_side="key") is False

    def test_empty_table_is_canonical_either_direction(self):
        assert _mapping_is_canonical({}) is True
        assert _mapping_is_canonical({}, placeholder_side="key") is True


class TestSubstituteWholeWordsLongestFirst:
    """FIX 7.1 — the longest-first hazard.  ``Person_10`` and ``Person_1``
    share a prefix; without length-descending ordering at each position,
    a naive scan matching ``Person_1`` first would leave the ``0`` of
    ``Person_10`` dangling in the output.  Pinned in BOTH dict insertion
    orders since Python dicts preserve insertion order and the previous
    implementation's bug was order-dependent.
    """

    def test_longer_key_wins_short_key_first_insertion_order(self):
        mapping = {"Person_1": "Alex", "Person_10": "Riley"}
        out = _substitute_whole_words("Person_10 met Person_1", mapping)
        assert out == "Riley met Alex"

    def test_longer_key_wins_long_key_first_insertion_order(self):
        mapping = {"Person_10": "Riley", "Person_1": "Alex"}
        out = _substitute_whole_words("Person_10 met Person_1", mapping)
        assert out == "Riley met Alex"

    def test_glued_forms_are_not_substituted(self):
        """A placeholder glued onto a longer identifier is not a whole-word
        match and must survive untouched — this is the word-boundary half
        of the same invariant (:data:`_is_word_char` transition), not the
        length-sort half, but the two only cooperate correctly together.

        Mutation: remove the length-descending sort
        (``paramem/graph/placeholders.py`` sort in ``_substitute_whole_words``)
        OR the trailing word-boundary check -> this test (or its siblings
        above) fails.
        """
        mapping = {"Person_1": "Alex", "Person_10": "Riley"}
        out = _substitute_whole_words("xPerson_1 Person_1x", mapping)
        assert out == "xPerson_1 Person_1x"

    def test_longer_multi_word_key_preempts_shorter_prefix_key(self):
        """The genuinely sort-dependent case: ``Person_1``/``Person_10``
        (above) are always disambiguated by the trailing word-boundary
        check alone (a digit is a word character, so ``Person_1``
        glued onto ``Person_10`` never boundary-matches) — that pair
        does not actually mutation-kill a sort removal on its own.  A
        multi-word key that is NOT a numeric-suffix prefix of the
        other (real PII attribute values: ``"New York"`` vs. ``"New
        York City"``) DOES need the sort: ``"New York"`` legitimately
        ends at a word boundary (a following space), so without
        trying the longer key first, the shorter key wins and leaves
        ``"City"`` dangling.

        Mutation: remove the length-descending sort -> this test fails
        (independently of the word-boundary check, which cannot catch
        this case since the shorter key's match IS a valid whole word).
        """
        mapping = {"New York": "City_2", "New York City": "City_1"}
        out = _substitute_whole_words("I visited New York City yesterday.", mapping)
        assert out == "I visited City_1 yesterday."


class TestSubstituteWholeWordsExactMatchRegression:
    """F1 — matching in :func:`_substitute_whole_words` is exact (raw
    ``==``), never routed through
    :func:`~paramem.graph.name_match.canonical`. Canonical (case-/
    separator-/diacritic-folded) matching would let a mapped person name
    (e.g. ``"Bill"``) silently consume its lowercase common-noun homograph
    (``"bill"``) in free transcript text, and would defeat the fail-closed
    residual-token drop on the deanonymize side. The graph-tier local
    anonymizer's mapping keys (which may differ in case/separators from
    the fold graph's own canonical node text, e.g. ``"Yang Ming"`` vs.
    ``"yang ming"``) are instead reconciled at their own call site — F1b,
    pinned in
    ``tests/test_graph_enrichment.py::TestGraphTierMappingReconciliation``
    — not by loosening this shared primitive's matching.

    Mutation: reintroduce ``canonical()`` matching in
    ``_substitute_whole_words`` -> both tests below fail.
    """

    def test_anonymize_direction_does_not_eat_common_noun_homograph(self):
        """ANONYMIZE direction (``_anonymize_transcript`` and friends): a
        mapped person name (``Bill``) must not consume its lowercase
        common-noun homograph (``the electricity bill``) — canonical
        (case-insensitive) matching would fold ``"Bill"`` and ``"bill"``
        onto the same identity and corrupt free-flowing transcript text
        the cloud model reasons over.
        """
        mapping = {"Bill": "Person_1"}
        text = "Bill said the electricity bill was late."
        out = _substitute_whole_words(text, mapping)
        assert out == "Person_1 said the electricity bill was late."

    def test_deanon_direction_does_not_substitute_literal_lowercase_text(self):
        """DEANON direction (:func:`deanonymize_text` / ``_apply_bindings``):
        literal text a human wrote (``person 1``, ``Person 1``) must NOT
        be substituted to the real name a DIFFERENT, exact-case, machine
        -minted token (``Person_1``) stands for — canonical matching would
        fold the human-written phrase onto the token's identity and, in
        the full pipeline, consume it before the fail-closed residual-
        token drop (b14a880) ever saw it.
        """
        reverse = {"Person_1": "Yang Ming"}
        assert deanonymize_text("person 1 in the queue", reverse) == "person 1 in the queue"
        assert deanonymize_text("Person 1 of 3 slides", reverse) == "Person 1 of 3 slides"


class TestPlaceholderShapeRegex:
    """FIX 7.2 — re-homed from ``tests/test_schema_config.py`` (deleted
    there when ``anonymizer_placeholder_pattern()`` was retired in favour
    of the single module-level :data:`PLACEHOLDER_SHAPE_RE`). This is the
    ONE regex a future bare -> braced format flip must change — direct
    contract coverage on it must not be lost in that carve-out.
    """

    def test_matches_common_placeholders(self):
        for token in ("Person_1", "City_42", "Country_3", "Org_10", "Thing_999"):
            assert PLACEHOLDER_SHAPE_RE.match(token), f"Pattern should match {token!r}"

    def test_matches_invented_prefixes(self):
        """The prefix vocabulary is open — type-appropriate PascalCase
        prefixes outside the common set must match."""
        for token in (
            "University_1",
            "Project_3",
            "Paper_1",
            "Language_2",
            "Currency_1",
            "Event_5",
            "Role_1",
            "Tool_99",
        ):
            assert PLACEHOLDER_SHAPE_RE.match(token), (
                f"Pattern should match invented prefix {token!r}"
            )

    def test_requires_uppercase_first_letter(self):
        """Lowercase-start prefix must NOT match — the most common LLM
        error mode, signalling the model ignored the shape contract."""
        assert not PLACEHOLDER_SHAPE_RE.match("person_1")
        assert not PLACEHOLDER_SHAPE_RE.match("city_42")

    def test_does_not_match_real_names(self):
        for token in ("Alex", "Berlin", "Apple", ""):
            assert not PLACEHOLDER_SHAPE_RE.match(token), f"Pattern should NOT match {token!r}"

    def test_does_not_match_prefix_without_suffix(self):
        assert not PLACEHOLDER_SHAPE_RE.match("Person")
        assert not PLACEHOLDER_SHAPE_RE.match("Person_")
        assert not PLACEHOLDER_SHAPE_RE.match("Person_abc")


class TestSpeakerAnchorReverseSkip:
    """FIX 7.3 — pins the most dangerous half of invariant 5 (currently
    asserted only implicitly by the module docstring): ``reverse`` must
    NEVER gain a ``speaker{N}``-keyed entry, even from a hostile LLM hint
    that scrubs a real name onto the anchor.

    Mutation: drop the ``if is_speaker_id(v): continue`` guard in the
    LLM-hint merge loop (``_build_anonymization_mapping``,
    ``paramem/graph/placeholders.py``) -> ``reverse["speaker0"] =
    "RealName"`` and a real display name is restored onto every
    speaker-subject fact at deanon time.
    """

    def test_hostile_llm_hint_never_creates_speaker_keyed_reverse_entry(self):
        graph = SessionGraph(
            session_id="s",
            timestamp="2026-05-06T00:00:00Z",
            entities=[Entity(name="speaker0", entity_type="person", speaker_id="speaker0")],
            relations=[],
        )
        forward, reverse = _build_anonymization_mapping(
            graph.entities,
            {"RealName": "speaker0"},  # hostile/hallucinated LLM hint
            pii_scope={"person"},
            speaker_name=None,
        )
        assert "speaker0" not in reverse
        assert is_speaker_id("speaker0")
        # The forward scrub is harmless and still useful (keeps "RealName"
        # out of anon_transcript) — only the reverse write is dangerous.
        assert forward.get("RealName") == "speaker0"
