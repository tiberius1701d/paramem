"""Unit tests for paramem.cloud.placeholders — the anonymize <-> deanonymize
placeholder primitive kit.

Most of this module's functions (``_apply_bindings``, ``_resolution_map``,
``build_forward_table``, ...) already have extensive coverage in
``tests/test_extraction_pipeline.py``. This file covers the primitives
that live only in ``paramem.cloud.placeholders``: ``mint_placeholder``,
``braced``, ``entity_type_to_prefix``, ``prefix_to_entity_type``, and the
generalized (``placeholder_side``) table normalize/validate pair.

``entity_type_to_prefix``/``prefix_to_entity_type``/``placeholder_entity_type``
live in :mod:`paramem.config.taxonomy`: they derive from the graph
entity-type taxonomy (``configs/schema.yaml``).
"""

from __future__ import annotations

import logging

import pytest

from paramem.cloud.anonymize_steps import ScanResult
from paramem.cloud.placeholders import (
    _MAX_MAPPING_TEXT_CHARS,
    PLACEHOLDER_SHAPE_RE,
    PLACEHOLDER_TOKEN_RE,
    ForwardTable,
    _binding_collisions,
    _decompose_token,
    _fact_orphans,
    _fact_tokens,
    _normalize_anonymization_mapping,
    _placeholder_tokens,
    _rendering_fold,
    _resolution_map,
    _substitute_whole_words,
    _substitute_whole_words_and_applied,
    applied_whole_word_keys,
    braced,
    build_forward_table,
    insert_placeholders,
    invert_forward_mapping,
    mint_placeholder,
    substitute_declared_renderings,
    unbraced,
)
from paramem.config.taxonomy import (
    ScrubCategory,
    entity_type_to_prefix,
    placeholder_entity_type,
    prefix_to_entity_type,
)


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
    """Closed vocabulary only (:func:`~paramem.config.taxonomy.
    anonymizer_type_to_prefix` — the schema's ``primary_for_type`` rows):
    an entity type with no primary row raises rather than composing an
    open-vocabulary PascalCase prefix."""

    def test_closed_vocabulary_matches_taxonomy(self):
        assert entity_type_to_prefix("person") == "Person"
        assert entity_type_to_prefix("place") == "City"
        assert entity_type_to_prefix("organization") == "Org"
        assert entity_type_to_prefix("concept") == "Thing"

    def test_a_type_with_no_primary_row_raises(self):
        """``event`` is a declared entity_type but no shipped row sets
        ``primary_for_type`` for it (only person/place/organization/
        concept do) — this must raise, not compose ``"Event"``."""
        with pytest.raises(ValueError, match="no primary_for_type row"):
            entity_type_to_prefix("event")

    def test_empty_or_blank_raises(self):
        with pytest.raises(ValueError):
            entity_type_to_prefix("")
        with pytest.raises(ValueError):
            entity_type_to_prefix(None)


class TestPrefixToEntityType:
    def test_closed_vocabulary_matches_taxonomy(self):
        assert prefix_to_entity_type("City") == "place"
        assert prefix_to_entity_type("Org") == "organization"
        assert prefix_to_entity_type("Person") == "person"
        assert prefix_to_entity_type("Thing") == "concept"

    def test_open_vocabulary_derives_type_from_prefix_itself(self):
        """The open policy (cloud's brace-binding protocol: the prefix IS
        the type name for a novel entity) — matches the pre-refactor
        behaviour of the entity-rebuild loop in ``_cloud_pipeline``, now
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
    """The ONE site deriving an entity type from a placeholder TOKEN
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
    default) and the cloud ``bindings`` table (``{placeholder: real_text}``,
    ``placeholder_side="key"`` — the OPPOSITE direction).
    """

    def test_core_table_default_direction_unchanged(self):
        mapping, stats = _normalize_anonymization_mapping({"Alex": "Person_1"})
        assert mapping == {"Alex": "Person_1"}
        assert stats == {"inverted": 0, "dropped": 0, "dropped_entries": []}

    def test_core_table_inverted_pair_is_corrected(self):
        mapping, stats = _normalize_anonymization_mapping({"Person_1": "Alex"})
        assert mapping == {"Alex": "Person_1"}
        assert stats["inverted"] == 1

    def test_bindings_table_correct_direction_kept_as_is(self):
        mapping, stats = _normalize_anonymization_mapping(
            {"Event_1": "the agile transformation initiative"}, placeholder_side="key"
        )
        assert mapping == {"Event_1": "the agile transformation initiative"}
        assert stats == {"inverted": 0, "dropped": 0, "dropped_entries": []}

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
        """A binding where both sides happen to be placeholder-shaped
        (e.g. `GPT_4`, a real model name) is not ambiguous — the declared
        `placeholder_side` breaks the tie rather than the entry being
        dropped and the whole delta rejected."""
        mapping, stats = _normalize_anonymization_mapping(
            {"Model_1": "GPT_4"}, placeholder_side="key"
        )
        assert mapping == {"Model_1": "GPT_4"}
        assert stats == {"inverted": 0, "dropped": 0, "dropped_entries": []}

    def test_core_table_both_sides_shaped_ties_to_declared_value_side(self):
        """Same tie-break, CORE table direction (`placeholder_side="value"`,
        the default)."""
        mapping, stats = _normalize_anonymization_mapping({"Person_2": "Windows_11"})
        assert mapping == {"Person_2": "Windows_11"}
        assert stats == {"inverted": 0, "dropped": 0, "dropped_entries": []}


class TestSpeakerIdValueCarveIn:
    """Fold-onto-token anonymization's normalizer carve-in: a
    speaker-id-shaped VALUE is accepted as placeholder-shaped on the CORE
    table (``placeholder_side="value"``) even though it never matches
    :data:`PLACEHOLDER_SHAPE_RE` — but the carve-in is deliberately NOT
    extended to the ``bindings`` table direction (``placeholder_side="key"``),
    where a speaker-id-shaped KEY stays genuinely ambiguous (a cloud model
    has no authority to bind new content onto the identity anchor).
    """

    def test_value_direction_keeps_speaker_id_shaped_value(self):
        mapping, stats = _normalize_anonymization_mapping(
            {"Real": "speaker0"}, placeholder_side="value"
        )
        assert mapping == {"Real": "speaker0"}
        assert stats["dropped"] == 0

    def test_key_direction_still_drops_the_same_pair(self):
        """The SAME mapping, ``placeholder_side="key"`` (the deanonymize
        ``bindings`` direction) — the carve-in must NEVER extend there:
        neither side is genuinely placeholder-shaped under that
        direction's own test ("speaker0" is not KEY-shaped either), so
        the entry stays dropped."""
        mapping, stats = _normalize_anonymization_mapping(
            {"Real": "speaker0"}, placeholder_side="key"
        )
        assert mapping == {}
        assert stats["dropped"] == 1


class TestSubstituteWholeWordsLongestFirst:
    """The longest-first hazard.  ``Person_10`` and ``Person_1``
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
        (``paramem/cloud/placeholders.py`` sort in ``_substitute_whole_words``)
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


class TestPossessiveIsASubstitutionProperty:
    """Possessive coverage is a property of ``_substitute_whole_words``
    itself, never a forward-table entry: ``'`` is not a word character
    (:data:`_is_word_char`), so a key matches inside its own possessive
    without needing a dedicated ``"Alex's"`` mapping.  This is also why a
    caller relying on this (e.g. the speaker fold in
    ``paramem.cloud.placeholders.build_forward_table``, which enters both
    a short and a longer surface as separate keys onto the same value)
    must enter the LONGER surface (``"Alex Miller"``) as its own key, not
    rely on the shorter ``"Alex"`` key alone.
    """

    def test_a_key_matches_inside_its_own_possessive_without_a_table_entry(self):
        mapping = {"Alex": "speaker0"}
        out = _substitute_whole_words("Alex's book is on the table.", mapping)
        assert out == "speaker0's book is on the table."

    def test_an_unrelated_word_sharing_a_prefix_is_left_alone(self):
        mapping = {"Alex": "speaker0"}
        out = _substitute_whole_words("Billing was Alex's job.", mapping)
        assert out == "Billing was speaker0's job."
        assert "Billing" in out

    def test_a_shorter_key_alone_leaves_the_longer_surfaces_surname_dangling(self):
        """The failure mode a fold covering only "Alex" would produce:
        without a separate "Alex Miller" entry, the surname egresses
        verbatim — exactly why the fold must re-point every matching key,
        not just the shorter one.
        """
        mapping = {"Alex": "speaker0"}
        out = _substitute_whole_words("Alex Miller called.", mapping)
        assert out == "speaker0 Miller called."

    def test_both_keys_present_fully_covers_the_longer_surface(self):
        mapping = {"Alex": "speaker0", "Alex Miller": "speaker0"}
        out = _substitute_whole_words("Alex Miller called.", mapping)
        assert out == "speaker0 called."


class TestAppliedWholeWordKeys:
    """The reporting form of :func:`_substitute_whole_words` —
    :func:`applied_whole_word_keys` — the ONE substitution walk
    (:func:`_substitute_whole_words_and_applied`) shared by both. Used by
    :func:`build_forward_table`'s prune pass to keep only the keys that
    are actually live over one payload.
    """

    def test_returns_only_the_keys_that_actually_matched(self) -> None:
        mapping = {"Alex": "Person_1", "Riley": "Person_2"}
        applied = applied_whole_word_keys("Alex went to the store.", mapping.keys())
        assert applied == {"Alex"}

    def test_a_key_present_nowhere_in_text_is_not_applied(self) -> None:
        mapping = {"Alex": "Person_1"}
        assert applied_whole_word_keys("Nothing here matches.", mapping.keys()) == set()

    def test_overlapping_non_nesting_spans_only_the_first_applied_key_survives(self) -> None:
        # Longest-first substitution consumes the first key's match; the
        # second key's own text is gone from the walk by the time its
        # turn to match would come, so it applies nowhere — the exact
        # shape behind the "inert forward key" defect.
        text = "Schillerpromenade 63, 12049 Berlin, Abteilung 3."
        mapping = {
            "Schillerpromenade 63, 12049 Berlin": "Address_1",
            "12049 Berlin, Abteilung 3": "Address_2",
        }
        applied = applied_whole_word_keys(text, mapping.keys())
        assert applied == {"Schillerpromenade 63, 12049 Berlin"}

    def test_applied_keys_agree_with_the_str_only_forms_own_substitutions(self) -> None:
        mapping = {"Person_10": "Riley", "Person_1": "Alex"}
        text = "Person_10 met Person_1"
        substituted, applied = _substitute_whole_words_and_applied(text, mapping)
        assert substituted == _substitute_whole_words(text, mapping)
        assert applied == {"Person_10", "Person_1"}

    def test_empty_text_or_mapping_yields_an_empty_applied_set(self) -> None:
        assert applied_whole_word_keys("", {"Alex": "Person_1"}.keys()) == set()
        assert applied_whole_word_keys("Alex", {}.keys()) == set()


class TestInsertPlaceholders:
    """``insert_placeholders`` — the one consumer that substitutes
    ``subject``/``object`` through a forward mapping and copies every
    other field verbatim, never touching the predicate.
    """

    def test_subject_and_object_are_substituted_predicate_is_untouched(self) -> None:
        facts = [{"subject": "Alex", "predicate": "lives at", "object": "Riley's place"}]
        mapping = {"Alex": "Person_1", "Riley's place": "Address_1"}
        out = insert_placeholders(facts, mapping)
        assert out == [{"subject": "Person_1", "predicate": "lives at", "object": "Address_1"}]

    def test_fields_other_than_subject_and_object_pass_through_unchanged(self) -> None:
        facts = [
            {
                "subject": "Alex",
                "predicate": "lives at",
                "object": "somewhere",
                "relation_type": "attribute",
                "confidence": 0.8,
                "speaker_id": "speaker1",
            }
        ]
        out = insert_placeholders(facts, {"Alex": "Person_1"})
        assert out[0]["relation_type"] == "attribute"
        assert out[0]["confidence"] == 0.8
        assert out[0]["speaker_id"] == "speaker1"

    def test_a_quote_backslash_and_non_ascii_value_substitutes_correctly(self) -> None:
        raw_value = 'Lindenstraße 44, "Hinterhof"\\Path'
        facts = [{"subject": "speaker1", "predicate": "lives at", "object": raw_value}]
        out = insert_placeholders(facts, {raw_value: "Address_1"})
        assert out[0]["object"] == "Address_1"

    def test_a_mapping_key_absent_from_the_fact_leaves_the_fact_unchanged(self) -> None:
        facts = [{"subject": "Alex", "predicate": "lives at", "object": "somewhere"}]
        out = insert_placeholders(facts, {"Riley": "Person_2"})
        assert out == facts


class TestSubstituteWholeWordsEdgeAwareBoundaries:
    """The edge-aware boundary rewrite (the confirmed live-PII-leak
    fix). The walk requires a boundary on a side only if the KEY's OWN
    edge char on that side is a word char (:func:`_is_word_char`), not
    "only attempt a match at a word-char position" (the previous
    implementation, which never even tried a key starting with a non-word
    char like ``"+"``).

    Mutation: revert to entering the match-search loop only when
    ``_is_word_char(text[pos])`` is true -> the leading-``+`` phone case
    below fails (the match is never attempted at all).
    """

    def test_non_word_leading_key_is_scrubbed(self):
        """A phone number key starting with ``"+"`` (a non-word char) must
        still be matched and scrubbed, even though the walk only starts a
        match search at a word-char position."""
        mapping = {"+49 151 2345": "Phone_1"}
        out = _substitute_whole_words("Call me at +49 151 2345 tomorrow.", mapping)
        assert out == "Call me at Phone_1 tomorrow."

    def test_non_word_leading_key_requires_right_boundary_when_key_edge_is_word_char(self):
        """The key's trailing char (``"5"``) IS a word char, so a right
        boundary is still required — the key must not match when glued
        onto more digits."""
        mapping = {"+49 151 2345": "Phone_1"}
        out = _substitute_whole_words("+49 151 23456 is not the number.", mapping)
        assert out == "+49 151 23456 is not the number."

    def test_case_sensitive_bill_does_not_match_lowercase_bill(self):
        mapping = {"Bill": "Person_1"}
        out = _substitute_whole_words("Bill paid the bill.", mapping)
        assert out == "Person_1 paid the bill."

    def test_bill_not_matched_inside_billing(self):
        """``"Bill"`` is word-char-bounded on both edges, so it requires a
        boundary on both sides — it must not match the ``"Bill"`` prefix
        of ``"Billing"`` (a word-char continues past the key's end)."""
        mapping = {"Bill": "Person_1"}
        out = _substitute_whole_words("The Billing department called Bill.", mapping)
        assert out == "The Billing department called Person_1."

    def test_longest_first_person_2_before_person(self):
        """``"Person_2"`` must preempt the shorter ``"Person"`` key at the
        same starting position (longest-key-first ordering, not merely
        the numeric-suffix case already covered by
        ``TestSubstituteWholeWordsLongestFirst``)."""
        mapping = {"Person": "Human_1", "Person_2": "Riley"}
        out = _substitute_whole_words("Person_2 arrived before Person.", mapping)
        assert out == "Riley arrived before Human_1."


class TestSubstituteWholeWordsExactMatchRegression:
    """Matching in :func:`_substitute_whole_words` is exact (raw
    ``==``), never routed through
    :func:`~paramem.utils.identity.canonical`. Canonical (case-/
    separator-/diacritic-folded) matching would let a mapped person name
    (e.g. ``"Bill"``) silently consume its lowercase common-noun homograph
    (``"bill"``) in free transcript text, and would defeat the fail-closed
    residual-token drop on the deanonymize side. The graph-tier local
    anonymizer's mapping keys (which may differ in case/separators from
    the fold graph's own canonical node text, e.g. ``"Yang Ming"`` vs.
    ``"yang ming"``) are instead reconciled at their own call site —
    pinned in
    ``tests/test_graph_enrichment.py::TestGraphTierMappingReconciliation``
    — not by loosening this shared primitive's matching.

    Mutation: reintroduce ``canonical()`` matching in
    ``_substitute_whole_words`` -> the tests below fail.
    """

    def test_anonymize_direction_does_not_eat_common_noun_homograph(self):
        """ANONYMIZE direction (mechanical forward substitution over facts,
        via ``_substitute_whole_words``): a mapped person name (``Bill``)
        must not consume its lowercase common-noun homograph (``the
        electricity bill``) — canonical (case-insensitive) matching would
        fold ``"Bill"`` and ``"bill"`` onto the same identity and corrupt
        free-flowing text. (Prose/transcript anonymization is now
        model-authored, not this mechanical primitive's job — but the
        case-sensitive invariant it enforces must still hold wherever this
        function runs.)
        """
        mapping = {"Bill": "Person_1"}
        text = "Bill said the electricity bill was late."
        out = _substitute_whole_words(text, mapping)
        assert out == "Person_1 said the electricity bill was late."

    def test_outbound_walk_does_not_match_a_reply_side_rendering(self):
        """The tolerance a reply may lean on (``person_1`` for
        ``Person_1``) is exclusive to :func:`substitute_declared_renderings`,
        the reply-restore walk — the outbound direction never gains a
        rendering mode. A lowercased rendering of a placeholder key is not
        a byte-exact match and is left untouched."""
        mapping = {"Person_1": "Alex"}
        text = "Hi person_1, welcome."
        out = _substitute_whole_words(text, mapping)
        assert out == text


class TestPlaceholderShapeRegex:
    """Re-homed from ``tests/test_schema_config.py`` (deleted
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


class TestMultiSegmentPlaceholderShape:
    """The real-leak fix.  A model emits multi-segment PascalCase
    prefixes for in-scope categories whose natural label is itself
    multi-word (``Home_Address_1``, ``Car_Plate_1``).  The pre-fix shape
    (single PascalCase segment + ``_N``) rejected these; an entry keyed on
    one would match neither side of :func:`_normalize_anonymization_mapping`
    and be silently DROPPED — the real value it stood for was never
    substituted into the script-built facts and egressed to the cloud
    verbatim.

    Mutation: revert ``_BARE_PLACEHOLDER_SHAPE`` to
    ``r"[A-Z][A-Za-z]*_\\d+"`` (single segment only) -> every test below
    fails.
    """

    def test_matches_multi_segment_prefixes(self):
        for token in ("Home_Address_1", "Car_Plate_1", "GPT_4", "COVID_19"):
            assert PLACEHOLDER_SHAPE_RE.match(token), f"Pattern should match {token!r}"

    def test_does_not_match_multi_segment_without_trailing_digits(self):
        """``Foo_Bar`` (no trailing ``_N``) must still NOT match — only the
        FINAL underscore-digit suffix is the mandatory numeric tail."""
        assert not PLACEHOLDER_SHAPE_RE.match("Foo_Bar")

    def test_multi_segment_placeholder_survives_normalization(self):
        """The exact leak this fix closes: a real value mapped to a
        multi-segment placeholder must be KEPT (placeholder on the value
        side, canonical CORE-table direction), not dropped as ambiguous."""
        mapping, stats = _normalize_anonymization_mapping(
            {"123 Main Street": "Home_Address_1", "AB-123-CD": "Car_Plate_1"}
        )
        assert mapping == {"123 Main Street": "Home_Address_1", "AB-123-CD": "Car_Plate_1"}
        assert stats == {"inverted": 0, "dropped": 0, "dropped_entries": []}

    def test_multi_segment_placeholder_survives_normalization_inverted(self):
        """Same entries handed in inverted (placeholder-as-key) direction
        are corrected, not dropped."""
        mapping, stats = _normalize_anonymization_mapping({"Home_Address_1": "123 Main Street"})
        assert mapping == {"123 Main Street": "Home_Address_1"}
        assert stats["inverted"] == 1
        assert stats["dropped"] == 0

    def test_in_text_detector_matches_multi_segment_placeholder(self):
        """The unanchored in-text detector (:data:`PLACEHOLDER_TOKEN_RE`)
        must find a multi-segment placeholder embedded in surrounding
        text — the same net :func:`_placeholder_tokens` and the deanon
        residual sweep rely on."""
        text = "Please confirm Home_Address_1 is correct before shipping Car_Plate_1."
        found = [t[0] or t[1] for t in PLACEHOLDER_TOKEN_RE.findall(text)]
        assert found == ["Home_Address_1", "Car_Plate_1"]

    def test_in_text_detector_does_not_over_match_across_word_boundary(self):
        """Broadening the shape must not let two adjacent bare tokens
        (space-separated, never glued) merge into one over-match."""
        text = "Person_1 Home_Address_1"
        found = [t[0] or t[1] for t in PLACEHOLDER_TOKEN_RE.findall(text)]
        assert found == ["Person_1", "Home_Address_1"]


class TestPlaceholderTokens:
    """``_placeholder_tokens`` — THE ``PLACEHOLDER_TOKEN_RE.findall`` +
    braced/bare name-extraction site. Every other primitive that needs
    "which placeholder tokens appear in this string" (``_fact_tokens``,
    ``CloudScope.response``'s binding-value pruning) routes through this
    function — these tests pin the primitive directly.
    """

    def test_bare_token_found(self):
        assert _placeholder_tokens("Person_1 lives in City_1.") == {"Person_1", "City_1"}

    def test_braced_token_found(self):
        assert _placeholder_tokens("Person_1 works at {Org_9}.") == {"Person_1", "Org_9"}

    def test_no_tokens_is_empty_set(self):
        assert _placeholder_tokens("Alex lives in Berlin.") == set()

    def test_empty_string_is_empty_set(self):
        assert _placeholder_tokens("") == set()

    def test_duplicate_occurrences_deduplicated(self):
        """A token mentioned twice contributes ONE set member — this is a
        SET, not a list of occurrences."""
        assert _placeholder_tokens("Person_1 met Person_1 again.") == {"Person_1"}

    def test_token_embedded_in_compound_string_found(self):
        """A token embedded in a larger compound string, but still
        word-boundary separated (e.g. a possessive), still surfaces."""
        found = _placeholder_tokens("software for Product_1's Legend")
        assert found == {"Product_1"}

    def test_token_glued_onto_a_longer_identifier_is_not_found(self):
        """Deliberately NOT caught: the ``\\b`` word-boundary anchor
        misses a token GLUED onto a longer identifier with no separating
        non-word character (``_`` is itself a word character) — e.g. a
        placeholder glued into a predicate
        (``language_proficiency_Language_3``). This is why
        ``_declared_placeholder_tokens`` exists as a SEPARATE,
        substring-based check for that class of bug — not a defect in
        this function."""
        assert _placeholder_tokens("language_proficiency_Language_3") == set()

    def test_multi_segment_prefix_token_found(self):
        assert _placeholder_tokens("See Home_Address_1 for details.") == {"Home_Address_1"}


class TestFactTokens:
    """``_fact_tokens`` — union of :func:`_placeholder_tokens` over a
    fact dict's ``subject``/``object`` fields ONLY. ``predicate`` is a
    SEPARATE invariant (owned by ``_apply_bindings`` step 1), never
    scanned here.
    """

    def test_subject_and_object_both_scanned(self):
        fact = {"subject": "Person_1", "predicate": "lives_in", "object": "City_1"}
        assert _fact_tokens(fact) == {"Person_1", "City_1"}

    def test_predicate_is_never_scanned(self):
        """A placeholder glued into the predicate (the motivating bug,
        ``language_proficiency_Language_3``) is invisible to this
        function — it is a real name and a real place, no token at all
        in subject/object."""
        fact = {
            "subject": "Alex",
            "predicate": "language_proficiency_Language_3",
            "object": "Advanced",
        }
        assert _fact_tokens(fact) == set()

    def test_missing_fields_treated_as_empty(self):
        """A fact dict missing ``subject``/``object`` entirely does not
        crash — treated as an empty string, contributing no tokens."""
        assert _fact_tokens({}) == set()

    def test_no_placeholder_present_is_empty(self):
        fact = {"subject": "Alex", "predicate": "lives_in", "object": "Berlin"}
        assert _fact_tokens(fact) == set()

    def test_braced_and_bare_both_contribute(self):
        fact = {"subject": "Person_1", "predicate": "child_of", "object": "{Person_2}"}
        assert _fact_tokens(fact) == {"Person_1", "Person_2"}


class TestFactOrphans:
    """``_fact_orphans`` — :func:`_fact_tokens` minus ``resolvable``. THE
    per-fact orphan predicate consumed by
    ``paramem.graph.extractor._apply_enrichment_delta``'s per-``add``/
    ``modify`` accept/reject test.
    """

    def test_all_tokens_resolvable_is_empty(self):
        fact = {"subject": "Person_1", "predicate": "lives_in", "object": "City_1"}
        assert _fact_orphans(fact, {"Person_1", "City_1"}) == set()

    def test_unresolvable_token_returned(self):
        fact = {"subject": "Person_1", "predicate": "married_to", "object": "Person_9"}
        assert _fact_orphans(fact, {"Person_1"}) == {"Person_9"}

    def test_empty_resolvable_set_makes_every_token_an_orphan(self):
        fact = {"subject": "Person_1", "predicate": "lives_in", "object": "City_1"}
        assert _fact_orphans(fact, set()) == {"Person_1", "City_1"}

    def test_no_tokens_at_all_is_empty_regardless_of_resolvable(self):
        fact = {"subject": "Alex", "predicate": "lives_in", "object": "Berlin"}
        assert _fact_orphans(fact, set()) == set()

    def test_partial_resolution_returns_only_the_unresolved_subset(self):
        fact = {"subject": "Person_1", "predicate": "knows", "object": "Person_9"}
        assert _fact_orphans(fact, {"Person_1", "Person_2"}) == {"Person_9"}


class TestReverseMapInversionAgreement:
    """Both the session tier and the graph tier reach their reverse
    ``{placeholder: real_name}`` table through the SAME
    :func:`build_forward_table` -> :func:`invert_forward_mapping` chain —
    :func:`invert_forward_mapping`'s first-wins tie-break on a many-to-one
    forward map is exercised directly here (the map-construction side is
    covered by :class:`TestBuildForwardTable` below).
    """

    _MANY_TO_ONE = {"Alice": "Person_1", "Bob": "Person_1"}

    def test_invert_forward_mapping_is_first_wins(self):
        assert invert_forward_mapping(self._MANY_TO_ONE) == {"Person_1": "Alice"}


class TestUnbraced:
    """``unbraced`` — the inverse of :func:`braced`, one surrounding
    ``{...}`` pair removed."""

    def test_removes_one_surrounding_pair(self):
        assert unbraced("{Person_1}") == "Person_1"

    def test_no_pair_returned_verbatim(self):
        assert unbraced("Person_1") == "Person_1"

    def test_unbalanced_brace_returned_verbatim(self):
        assert unbraced("{Person_1") == "{Person_1"
        assert unbraced("Person_1}") == "Person_1}"

    def test_empty_string(self):
        assert unbraced("") == ""

    def test_only_outer_pair_removed(self):
        """A doubly-braced token loses only the OUTER pair per call."""
        assert unbraced("{{Person_1}}") == "{Person_1}"

    def test_no_whitespace_trimming(self):
        assert unbraced("{ Person_1 }") == " Person_1 "


class TestNormalizeAnonymizationMappingBracedCandidates:
    """One token shape per surface, canonicalized at the
    normalizer boundary. A braced candidate is stripped via
    :func:`unbraced` before shape validation and stored bare on the
    placeholder side; the real side is stored verbatim (braces and all)."""

    def test_braced_binding_key_honored(self):
        """The cloud enrichment wire shape: a braced binding key."""
        mapping, stats = _normalize_anonymization_mapping(
            {"{Event_1}": "the agile transformation initiative"}, placeholder_side="key"
        )
        assert mapping == {"Event_1": "the agile transformation initiative"}
        assert stats["dropped"] == 0

    def test_braced_core_value_honored(self):
        """A braced CORE anonymizer-table value."""
        mapping, stats = _normalize_anonymization_mapping({"Jane Doe": "{Person_1}"})
        assert mapping == {"Jane Doe": "Person_1"}
        assert stats["dropped"] == 0

    def test_idempotent_on_the_double_normalize_production_path(self):
        """``_parse_enrichment_delta`` (extractor.py) then
        ``CloudScope.response`` (deanonymize.py) both normalize the same
        cloud ``bindings`` table — normalizing an already-normalized
        table a second time must be a no-op."""
        first, _ = _normalize_anonymization_mapping(
            {"{Event_1}": "the agile transformation initiative"}, placeholder_side="key"
        )
        second, stats2 = _normalize_anonymization_mapping(first, placeholder_side="key")
        assert second == first
        assert stats2["dropped"] == 0

    def test_real_side_keeps_braces_and_whitespace_verbatim(self):
        mapping, _stats = _normalize_anonymization_mapping({"{Jane}": "Person_1"})
        assert mapping == {"{Jane}": "Person_1"}

    def test_double_braced_placeholder_strips_once_and_still_drops(self):
        mapping, stats = _normalize_anonymization_mapping(
            {"{{Person_1}}": "x"}, placeholder_side="key"
        )
        assert mapping == {}
        assert stats["dropped"] == 1

    def test_dropped_entries_payload_shape_key_side(self):
        mapping, stats = _normalize_anonymization_mapping(
            {"{some phrase}": "a value"}, placeholder_side="key"
        )
        assert mapping == {}
        [entry] = stats["dropped_entries"]
        assert entry["side"] == "key"
        assert entry["text"] == "{some phrase}"
        assert entry["counterpart_len"] == len("a value")

    def test_dropped_entries_payload_shape_value_side(self):
        mapping, stats = _normalize_anonymization_mapping({"Jane": "{some phrase}"})
        assert mapping == {}
        [entry] = stats["dropped_entries"]
        assert entry["side"] == "value"
        assert entry["text"] == "{some phrase}"
        assert entry["counterpart_len"] == len("Jane")

    def test_dropped_entries_text_truncated(self):
        long_junk = "x" * 200
        mapping, stats = _normalize_anonymization_mapping(
            {long_junk: "also junk"}, placeholder_side="key"
        )
        assert mapping == {}
        [entry] = stats["dropped_entries"]
        assert len(entry["text"]) == _MAX_MAPPING_TEXT_CHARS
        assert entry["text"] == long_junk[:_MAX_MAPPING_TEXT_CHARS]

    def test_warning_log_line_carries_counts_only(self, caplog):
        """The privacy-relevant half: the WARNING log for a dropped entry
        must never carry the offending text as a substring."""
        caplog.set_level(logging.WARNING, logger="paramem.cloud.placeholders")
        secret = "a-genuinely-unshaped-junk-string-Q7z"
        _normalize_anonymization_mapping({secret: "also junk"}, placeholder_side="key")
        log_text = caplog.text
        assert secret not in log_text
        assert "dropped" in log_text.lower()


class TestDecomposeToken:
    """``_decompose_token`` — the mint's own ``f"{prefix}_{n}"`` format
    (:func:`mint_placeholder`) read backwards, not an independent spelling
    of the shape. Composed by :func:`substitute_declared_renderings` and
    by :func:`_rendering_fold`.
    """

    def test_accepts_a_single_segment_token(self):
        assert _decompose_token("Person_1") == ("Person", "1")

    def test_accepts_a_multi_segment_prefix(self):
        assert _decompose_token("Home_Address_1") == ("Home_Address", "1")

    def test_rejects_a_non_numeric_tail(self):
        assert _decompose_token("Foo_Bar") is None

    def test_rejects_a_zero_padded_tail(self):
        assert _decompose_token("Person_01") is None

    def test_rejects_a_separator_free_speaker_token(self):
        """A speaker token never decomposes — the speaker family is
        governed separately and is never a rendering-matcher candidate."""
        assert _decompose_token("speaker1") is None

    def test_rejects_a_non_decimal_unicode_digit_without_raising(self):
        """A non-decimal Unicode digit (e.g. a superscript) passes
        ``str.isdigit()`` but ``int()`` rejects it — ``isdecimal()`` is
        what keeps this function total instead of raising."""
        assert _decompose_token("Person_²") is None


class TestSubstituteDeclaredRenderings:
    """``substitute_declared_renderings`` — the reply-side tolerant walk
    beside the exact ``_substitute_whole_words`` walk. The rendering
    domain is exactly the minted form, any casing of it, and the final
    ``_`` written as one space; the domain stops there deliberately.
    """

    def test_lowercase_rendering_restores(self):
        out = substitute_declared_renderings("Hi person_1, welcome.", {"Person_1": "Alex"})
        assert out == "Hi Alex, welcome."

    def test_uppercase_rendering_restores(self):
        out = substitute_declared_renderings("Hi PERSON_1, welcome.", {"Person_1": "Alex"})
        assert out == "Hi Alex, welcome."

    def test_space_separated_rendering_restores(self):
        out = substitute_declared_renderings("Hi Person 1, welcome.", {"Person_1": "Alex"})
        assert out == "Hi Alex, welcome."

    def test_possessive_rendering_restores(self):
        """A following non-word char (the possessive apostrophe) does not
        block a match — the same edge-aware boundary rule the exact walk
        uses."""
        out = substitute_declared_renderings("Person_1's book is here.", {"Person_1": "Alex"})
        assert out == "Alex's book is here."

    def test_longest_token_wins_at_the_same_start(self):
        """Regex alternation is leftmost-alternative; the longest-first
        sort is what makes ``Person_10`` beat ``Person_1`` rather than
        leaving a dangling ``0``."""
        mapping = {"Person_1": "Alex", "Person_10": "Riley"}
        out = substitute_declared_renderings("Person_10 called Person_1.", mapping)
        assert out == "Riley called Alex."

    def test_hyphenated_surface_is_not_a_rendering(self):
        """``Person-1`` is outside the rendering domain, even though
        ``Person_1`` is declared — a declared token mangled this way is
        left in the text as written."""
        out = substitute_declared_renderings("Hi Person-1, welcome.", {"Person_1": "Alex"})
        assert out == "Hi Person-1, welcome."

    def test_zero_padded_surface_is_not_a_rendering(self):
        out = substitute_declared_renderings("Hi Person_01, welcome.", {"Person_1": "Alex"})
        assert out == "Hi Person_01, welcome."

    def test_unminted_token_is_not_a_rendering(self):
        out = substitute_declared_renderings("Hi Person_9, welcome.", {"Person_1": "Alex"})
        assert out == "Hi Person_9, welcome."

    def test_a_speaker_shaped_mapping_key_contributes_no_alternative(self):
        """No ``speaker{N}`` is ever a declared token, and the matcher
        confirms it: a speaker-shaped key in the handed mapping never
        decomposes, so it never becomes a rendering alternative."""
        out = substitute_declared_renderings("SPEAKER0 said hi.", {"speaker0": "Alex"})
        assert out == "SPEAKER0 said hi."


class TestRenderingFoldDistinctness:
    """The declared vocabulary is distinct under the rendering equivalence
    :func:`_rendering_fold` defines: a cloud binding whose key is
    rendering-equal to ANY core token — shown or not — or to another
    cloud binding, is inert in :func:`_resolution_map` (the CORE value
    wins, CORE-LAST, and a sibling collision class is inert on every
    member) and is named by :func:`_binding_collisions` — the same
    equivalence at both membership sites.
    """

    def test_case_variant_binding_is_inert_core_wins_in_resolution(self):
        reverse = {"Person_1": "Alex"}
        cloud_bindings = {"PERSON_1": "someone else"}
        observed = {"Person_1"}
        resolution = _resolution_map(reverse, cloud_bindings, observed)
        assert resolution["Person_1"] == "Alex"
        assert "PERSON_1" not in resolution

    def test_case_variant_binding_is_named_by_binding_collisions(self):
        collisions = _binding_collisions(
            {"Person_1": "Alex"},
            cloud_bindings={"PERSON_1": "someone else"},
        )
        assert collisions == ["PERSON_1"]

    def test_exact_match_and_case_variant_fold_to_the_same_key(self):
        assert _rendering_fold("Person_1") == _rendering_fold("PERSON_1")

    def test_a_token_with_no_rendering_folds_to_itself(self):
        """A token that does not decompose (``Foo_Bar``) has no
        equivalence class beyond itself — its comparisons stay
        exact-string."""
        assert _rendering_fold("Foo_Bar") == "Foo_Bar"

    def test_binding_rendering_an_unshown_core_token_is_inert_and_named(self):
        """A cloud binding whose key renders an UNSHOWN core token is
        inert in resolution — not merely a shown one — because a rendering
        the external service writes back cannot itself carry the
        shown/unshown distinction; the same binding is named by
        :func:`_binding_collisions`."""
        reverse = {"Person_1": "Alex"}
        cloud_bindings = {"person_1": "the neighbour"}
        observed: set[str] = set()
        resolution = _resolution_map(reverse, cloud_bindings, observed)
        assert "person_1" not in resolution
        assert "Person_1" not in resolution
        collisions = _binding_collisions(reverse, cloud_bindings=cloud_bindings)
        assert collisions == ["person_1"]

    def test_sibling_bindings_rendering_equal_are_both_inert_and_named(self):
        """Two cloud bindings rendering-equal to EACH OTHER, with no core
        token involved, are both inert in resolution and both named —
        every member of a binding-side collision class is inert."""
        cloud_bindings = {"Org_1": "first", "ORG_1": "second"}
        observed = {"Person_9"}
        resolution = _resolution_map({}, cloud_bindings, observed)
        assert "Org_1" not in resolution
        assert "ORG_1" not in resolution
        collisions = _binding_collisions({}, cloud_bindings=cloud_bindings)
        assert collisions == ["ORG_1", "Org_1"]

    def test_glued_span_is_not_a_rendering_match(self):
        """The rendering walk itself refuses a glued span on either side —
        pinned at the walk level, independent of the composed refusal
        gate in :func:`~paramem.cloud.deanonymize.deanonymize_text`."""
        out = substitute_declared_renderings("xPerson_1 and Person_1x", {"Person_1": "Alex"})
        assert out == "xPerson_1 and Person_1x"

    def test_empty_mapping_returns_text_unchanged(self):
        out = substitute_declared_renderings("Person_1 said hi.", {})
        assert out == "Person_1 said hi."

    def test_substituted_value_is_never_rescanned(self):
        """The result is assembled in a single left-to-right pass over the
        ORIGINAL text's match spans — a substituted real value that
        happens to look like another declared token's rendering is never
        fed back through the matcher."""
        out = substitute_declared_renderings(
            "Person_1", {"Person_1": "Person_2", "Person_2": "Riley"}
        )
        assert out == "Person_2"


def _build(
    scans,
    tag_text,
    *,
    anchor_names=frozenset(),
    speaker_id=None,
    speaker_name=None,
    identity_domain=None,
    person_prefix="Person",
) -> ForwardTable:
    return build_forward_table(
        scans,
        tag_text=tag_text,
        anchor_names=anchor_names,
        speaker_id=speaker_id,
        speaker_name=speaker_name,
        identity_domain=identity_domain,
        person_prefix=person_prefix,
    )


class TestBuildForwardTable:
    """``build_forward_table`` — the CORE table's one constructor: canonical
    sharing, per-prefix uniqueness, first-category-wins, containment
    pooling, the speaker fold, reconciliation before pruning, the prune of
    inert entries, and the :class:`ForwardTable` return invariants.
    """

    def test_canonically_equal_surfaces_share_one_placeholder(self):
        scans = (ScanResult(category=ScrubCategory("Person"), values=("Alex", "ALEX")),)
        table = _build(scans, "Alex met ALEX.")
        assert table.forward == {"Alex": "Person_1", "ALEX": "Person_1"}

    def test_distinct_surfaces_mint_distinct_numbers_in_first_occurrence_order(self):
        scans = (ScanResult(category=ScrubCategory("Person"), values=("Alex", "Riley")),)
        table = _build(scans, "Alex met Riley.")
        assert table.forward["Alex"] == "Person_1"
        assert table.forward["Riley"] == "Person_2"

    def test_a_value_scanned_under_two_categories_keeps_the_first_categorys_prefix(self):
        scans = (
            ScanResult(category=ScrubCategory("Person"), values=("Alex",)),
            ScanResult(category=ScrubCategory("Artist"), values=("Alex",)),
        )
        table = _build(scans, "Alex is here.")
        assert table.forward["Alex"] == "Person_1"

    def test_a_bare_surface_beside_its_longer_container_shares_the_containers_placeholder(self):
        scans = (
            ScanResult(category=ScrubCategory("Person"), values=("Bettina Schuster", "Bettina")),
        )
        table = _build(scans, "Bettina Schuster and Bettina came.")
        assert table.forward["Bettina"] == table.forward["Bettina Schuster"]
        # Exactly one mint — the pooled group takes one placeholder.
        assert len(set(table.forward.values())) == 1

    def test_speaker_fold_on_attestation(self):
        scans = (ScanResult(category=ScrubCategory("Person"), values=("Priya",)),)
        table = _build(
            scans,
            "Hi, I'm Priya.",
            anchor_names=frozenset({"Priya"}),
            speaker_id="speaker1",
        )
        assert table.forward["Priya"] == "speaker1"

    def test_speaker_fold_on_exact_enrolled_name_needs_no_attestation(self):
        scans = (ScanResult(category=ScrubCategory("Person"), values=("Alex",)),)
        table = _build(
            scans,
            "Alex went home.",
            speaker_id="speaker1",
            speaker_name="Alex",
        )
        assert table.forward["Alex"] == "speaker1"

    def test_an_attested_namesake_inconsistent_with_the_enrolled_name_does_not_fold(self):
        """Attested but NOT the enrolled speaker (a different person the
        speaker introduces) — mints its own placeholder, never the
        speaker's token."""
        scans = (ScanResult(category=ScrubCategory("Person"), values=("Mira",)),)
        table = _build(
            scans,
            "This is Mira.",
            anchor_names=frozenset({"Mira"}),
            speaker_id="speaker1",
            speaker_name="Alex",
        )
        assert table.forward["Mira"] == "Person_1"

    def test_speaker_group_is_never_a_containment_merge_target(self):
        """A shorter surface (``Rivera``) whose only textual container is
        the enrolled speaker's own longer surface (``Alex Rivera``, folded
        onto the speaker token by exact match) keeps its own identity — the
        speaker group absorbs nothing via containment."""
        scans = (ScanResult(category=ScrubCategory("Person"), values=("Alex Rivera", "Rivera")),)
        table = _build(
            scans,
            "Alex Rivera introduced Rivera to everyone.",
            speaker_id="speaker1",
            speaker_name="Alex Rivera",
        )
        assert table.forward["Alex Rivera"] == "speaker1"
        assert table.forward["Rivera"] not in ("speaker1",)
        assert table.forward["Rivera"].startswith("Person_")

    def test_reconciliation_rekeys_onto_the_domain_surface_before_the_prune(self):
        """A scanned surface that differs from the fold graph's own
        (canonical) node key is re-keyed onto that node key — and it is
        the RE-KEYED surface the prune tests against ``tag_text``, not the
        original scanned one."""
        scans = (ScanResult(category=ScrubCategory("Person"), values=("Yang Ming",)),)
        table = _build(
            scans,
            "yang ming is a colleague.",
            identity_domain=["yang ming"],
        )
        assert table.forward == {"yang ming": "Person_1"}
        assert table.rekey_dropped == 0

    def test_reconciliation_drops_a_surface_with_no_domain_match(self):
        scans = (ScanResult(category=ScrubCategory("Person"), values=("Alex",)),)
        table = _build(
            scans,
            "Alex is here.",
            identity_domain=["someone else"],
        )
        assert table.forward == {}
        assert table.rekey_dropped == 1

    def test_a_surface_absent_from_tag_text_is_pruned_as_inert(self):
        scans = (ScanResult(category=ScrubCategory("Person"), values=("Alex",)),)
        table = _build(scans, "Nothing about anyone here.")
        assert table.forward == {}
        [entry] = table.inert_entries
        assert entry == {"category": "Person", "side": "table", "text": "Alex", "reason": "inert"}

    def test_forward_table_return_shape(self):
        scans = (ScanResult(category=ScrubCategory("Person"), values=("Alex",)),)
        table = _build(scans, "Alex is here.")
        assert isinstance(table, ForwardTable)
        assert table.forward == {"Alex": "Person_1"}
        assert table.rekey_dropped == 0
        assert table.inert_entries == ()

    def test_prefix_numbering_stays_contiguous_after_a_share_and_an_inert_prune(self):
        """A canonical share consumes no mint number of its own (pass 1),
        and a group pruned as inert (pass 4) leaves no gap in the
        surviving mint order (pass 5's "no holes" invariant): "Alex"/"ALEX"
        share one number, the inert "Ghost" group between them in creation
        order is dropped before minting, and "Riley" still mints 2, not 3."""
        scans = (
            ScanResult(category=ScrubCategory("Person"), values=("Alex", "ALEX", "Ghost", "Riley")),
        )
        table = _build(scans, "Alex met ALEX and Riley.")
        assert table.forward == {"Alex": "Person_1", "ALEX": "Person_1", "Riley": "Person_2"}
        [entry] = table.inert_entries
        assert entry["text"] == "Ghost"

    def test_a_three_way_containment_chain_collapses_onto_one_group(self):
        """Longest-first judging (pass 2) settles a container's own merge
        before a shorter member is judged, so a transitive chain collapses
        onto one group in a single pass: "Ann" inside "Ann Marie" inside
        "Ann Marie Bell"."""
        scans = (
            ScanResult(
                category=ScrubCategory("Person"), values=("Ann Marie", "Ann Marie Bell", "Ann")
            ),
        )
        table = _build(scans, "Ann Marie Bell introduced Ann Marie and Ann.")
        assert table.forward == {
            "Ann": "Person_1",
            "Ann Marie": "Person_1",
            "Ann Marie Bell": "Person_1",
        }

    def test_the_transitive_partition_is_invariant_under_scan_order(self):
        """The containment judging order is sorted by surface length
        (pass 2's ``judged_groups.sort``), never by scan/creation order —
        so the same three surfaces collapse onto the identical one-group
        partition whichever order they arrive in."""
        tag_text = "Ann Marie Bell introduced Ann Marie and Ann."
        ascending = _build(
            (
                ScanResult(
                    category=ScrubCategory("Person"), values=("Ann", "Ann Marie", "Ann Marie Bell")
                ),
            ),
            tag_text,
        ).forward
        descending = _build(
            (
                ScanResult(
                    category=ScrubCategory("Person"), values=("Ann Marie Bell", "Ann Marie", "Ann")
                ),
            ),
            tag_text,
        ).forward
        assert (
            ascending
            == descending
            == {"Ann": "Person_1", "Ann Marie": "Person_1", "Ann Marie Bell": "Person_1"}
        )

    def test_a_canonical_equality_group_moves_together_on_a_containment_repoint(self):
        """Two surfaces "Bettina" and "BETTINA" share one group by
        canonical equality (pass 1) before containment runs; when that
        group is later found contained in "Bettina Schuster" (pass 2),
        EVERY member of the shared group repoints to the container — not
        only the member whose literal surface matched it."""
        scans = (
            ScanResult(
                category=ScrubCategory("Person"), values=("Bettina", "BETTINA", "Bettina Schuster")
            ),
        )
        table = _build(scans, "Bettina Schuster met Bettina and BETTINA.")
        assert table.forward == {
            "Bettina Schuster": "Person_1",
            "Bettina": "Person_1",
            "BETTINA": "Person_1",
        }

    def test_speaker_fold_on_an_attested_extension_of_the_enrolled_name(self):
        """``_is_speaker_surface``: an attested surface that EXTENDS the
        enrolled name ("Alex Rivera" attested, enrolled "Alex") is
        consistent (``low.startswith(name + " ")``) and folds onto the
        speaker token."""
        scans = (ScanResult(category=ScrubCategory("Person"), values=("Alex Rivera",)),)
        table = _build(
            scans,
            "This is Alex Rivera.",
            anchor_names=frozenset({"Alex Rivera"}),
            speaker_id="speaker1",
            speaker_name="Alex",
        )
        assert table.forward == {"Alex Rivera": "speaker1"}

    def test_speaker_fold_on_an_attested_short_form_of_the_enrolled_name(self):
        """``_is_speaker_surface``: an attested surface that is a SHORT
        FORM of the enrolled name ("Alex" attested, enrolled "Alex
        Rivera") is consistent (``name.startswith(low + " ")``) and folds
        onto the speaker token."""
        scans = (ScanResult(category=ScrubCategory("Person"), values=("Alex",)),)
        table = _build(
            scans,
            "This is Alex.",
            anchor_names=frozenset({"Alex"}),
            speaker_id="speaker1",
            speaker_name="Alex Rivera",
        )
        assert table.forward == {"Alex": "speaker1"}

    def test_no_person_category_active_means_no_speaker_entry_at_all(self):
        """``person_idx`` is ``-1`` when no scan category's prefix equals
        ``person_prefix`` (the operator narrowed ``scrub`` away from
        person names): both speaker-fold gates (``target``, ``enrolled``)
        resolve to ``None`` regardless of ``speaker_id``/``speaker_name``/
        ``anchor_names``, so nothing ever reaches the speaker token."""
        scans = (ScanResult(category=ScrubCategory("City"), values=("Ghent",)),)
        table = _build(
            scans,
            "Alex went to Ghent.",
            anchor_names=frozenset({"Alex"}),
            speaker_id="speaker1",
            speaker_name="Alex",
        )
        assert "speaker1" not in table.forward.values()
        assert table.forward == {"Ghent": "City_1"}

    def test_speaker_name_without_a_speaker_id_is_never_consumed(self):
        """``target`` gates on ``speaker_id`` alone (truthy AND well-shaped
        AND a person category active); every read of ``enrolled`` sits
        inside a ``target is not None`` branch, so a ``speaker_name`` with
        no ``speaker_id`` is inert and the value mints an ordinary
        placeholder instead of folding."""
        scans = (ScanResult(category=ScrubCategory("Person"), values=("Alex",)),)
        table = _build(
            scans,
            "Alex is here.",
            anchor_names=frozenset({"Alex"}),
            speaker_name="Alex",
        )
        assert table.forward == {"Alex": "Person_1"}

    def test_the_enrolled_name_is_pruned_like_any_other_key_when_absent_from_the_text(self):
        """The enrolled name is entered as an additional forward key on
        the speaker group (pass 1b) whenever it is not already a member;
        it is then an ordinary member for pass 4's prune, same as any
        other key — absent from ``tag_text``, it is dropped and recorded
        with ``category=""`` (the speaker group mints nothing, so it
        carries no mint prefix), while the surface that actually folded
        survives."""
        scans = (ScanResult(category=ScrubCategory("Person"), values=("ALEX",)),)
        table = _build(
            scans,
            "ALEX said hi.",
            speaker_id="speaker1",
            speaker_name="Alex",
        )
        assert table.forward == {"ALEX": "speaker1"}
        [entry] = table.inert_entries
        assert entry == {"category": "", "side": "table", "text": "Alex", "reason": "inert"}

    def test_containment_matching_is_case_sensitive(self):
        """``_whole_word_contains`` folds only through ``canonical(...,
        mode="spaces")`` — blanks only, case and diacritics preserved —
        unlike pass 1's canonical-equality SHARE, which folds full
        casefold. Two surfaces differing only in case ("MIRA" vs "Mira
        Santos") are therefore never pooled by containment, even though
        "MIRA" would be a substring of "Mira Santos" case-insensitively;
        each mints its own placeholder."""
        scans = (ScanResult(category=ScrubCategory("Person"), values=("MIRA", "Mira Santos")),)
        table = _build(scans, "MIRA and Mira Santos both replied.")
        assert table.forward == {"MIRA": "Person_1", "Mira Santos": "Person_2"}

    def test_containment_pooling_never_crosses_a_category_boundary(self):
        """Pass 2's containment loop is scoped per category ``idx``: the
        candidate containers considered for a judged group are only that
        SAME category's own ``surfaces`` list, and only that category's
        OWN groups (``owner == idx``) are ever judged. A surface scanned
        under one category is never absorbed by a same-text container
        scanned under a DIFFERENT category, even when one textually
        contains the other."""
        scans = (
            ScanResult(category=ScrubCategory("Person"), values=("Bettina",)),
            ScanResult(category=ScrubCategory("Artist"), values=("Bettina Schuster",)),
        )
        table = _build(scans, "Bettina Schuster and Bettina performed.")
        assert table.forward == {"Bettina": "Person_1", "Bettina Schuster": "Artist_1"}
