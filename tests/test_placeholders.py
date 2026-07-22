"""Unit tests for paramem.cloud.placeholders — the anonymize <-> deanonymize
placeholder primitive kit.

Most of this module's functions (``_apply_bindings``, ``_resolution_map``,
``_build_anonymization_mapping``, ...) already have extensive coverage in
``tests/test_extraction_pipeline.py`` (moved there unchanged when this
module was carved out of ``paramem.graph.extractor``). This file covers
the NEW unified primitives introduced by that carve-out: ``mint_placeholder``,
``braced``, ``entity_type_to_prefix``, ``prefix_to_entity_type``, and the
generalized (``placeholder_side``) table normalize/validate pair.

``entity_type_to_prefix``/``prefix_to_entity_type``/``placeholder_entity_type``
moved to :mod:`paramem.config.taxonomy` (originally
``paramem.graph.schema_config``, in the ``paramem/cloud/`` package carve of
2026-07-21; re-homed to ``paramem.config`` when the taxonomy loader itself
moved out of ``paramem.graph``): they derive from the graph entity-type
taxonomy (``configs/schema.yaml``).
"""

from __future__ import annotations

import inspect

from paramem.cloud.placeholders import (
    PLACEHOLDER_SHAPE_RE,
    PLACEHOLDER_TOKEN_RE,
    _build_anonymization_mapping,
    _fact_orphans,
    _fact_tokens,
    _normalize_anonymization_mapping,
    _placeholder_tokens,
    _substitute_whole_words,
    braced,
    invert_forward_mapping,
    mint_placeholder,
)
from paramem.config.taxonomy import (
    entity_type_to_prefix,
    placeholder_entity_type,
    prefix_to_entity_type,
)
from paramem.utils.identity import is_speaker_id


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
    def test_closed_vocabulary_matches_taxonomy(self):
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
        """The confirmed live leak: a phone number key starting with
        ``"+"`` (a non-word char) was never attempted under the old
        "only match at a word-char position" walk."""
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
    ``_substitute_whole_words`` -> both tests below fail.
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

    def test_deanon_direction_does_not_substitute_literal_lowercase_text(self):
        """DEANON direction (:func:`~paramem.cloud.deanonymize.deanonymize_text`
        / ``_apply_bindings``): literal text a human wrote (``person 1``,
        ``Person 1``) must NOT be substituted to the real name a
        DIFFERENT, exact-case, machine-minted token (``Person_1``) stands
        for — canonical matching would fold the human-written phrase onto
        the token's identity and, in the full pipeline, consume it before
        the fail-closed residual-token drop (b14a880) ever saw it.

        Exercised directly against ``_substitute_whole_words`` — the
        composed ``paramem.cloud.deanonymize.deanonymize_text`` is this
        exact call plus a declared-token fail-closed check (a standalone
        ``deanonymize_text(text, resolution)`` primitive used to wrap it,
        but it was a pure null-guard pass-through already subsumed by
        ``_substitute_whole_words``'s own guard, so it was retired rather
        than kept as a second name for the same one-line body).
        """
        reverse = {"Person_1": "Yang Ming"}
        assert _substitute_whole_words("person 1 in the queue", reverse) == (
            "person 1 in the queue"
        )
        assert _substitute_whole_words("Person 1 of 3 slides", reverse) == "Person 1 of 3 slides"


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
        assert stats == {"inverted": 0, "dropped": 0}

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
    braced/bare name-extraction site (2026-07-22 cloud-admission
    redesign de-duplication follow-up). Every other primitive that needs
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
        forward, reverse = _build_anonymization_mapping(
            {"RealName": "speaker0"},  # hostile/hallucinated LLM hint
            speaker_name=None,
        )
        assert "speaker0" not in reverse
        assert is_speaker_id("speaker0")
        # The forward scrub is harmless and still useful (keeps "RealName"
        # out of anon_transcript) — only the reverse write is dangerous.
        assert forward.get("RealName") == "speaker0"


class TestLlmHintMergeUnconditional:
    """Post cloud-egress-PII redesign: ``_build_anonymization_mapping``
    holds NO entity walk and NO scope gate of its own — the model's
    anonymizer prompt is the SOLE scope authority. Every LLM hint not
    keyed/valued on the speaker anchor is
    merged in as-is; this builder does not re-judge, re-walk, or float a
    graph-derived completeness floor under the model's scope decision.

    Supersedes the deleted ``TestLlmHintMergeScopedToPiiScope`` (pinned
    the now-removed ``pii_scope`` gate via ``placeholder_entity_type``).
    """

    def test_llm_hint_of_any_entity_type_is_merged(self):
        """Mutation: reintroduce a type-based gate -> 'Acme Corp' is
        dropped instead of merged -> this test fails.
        """
        forward, reverse = _build_anonymization_mapping(
            {"Acme Corp": "Org_1"},
            speaker_name=None,
        )
        assert forward.get("Acme Corp") == "Org_1"
        assert reverse.get("Org_1") == "Acme Corp"
        text = "I work at Acme Corp in the automotive division."
        out = _substitute_whole_words(text, forward)
        assert out == "I work at Org_1 in the automotive division."

    def test_person_llm_hint_is_kept(self):
        forward, reverse = _build_anonymization_mapping(
            {"Kim": "Person_1"},
            speaker_name=None,
        )
        assert forward.get("Kim") == "Person_1"
        assert reverse.get("Person_1") == "Kim"
        assert _substitute_whole_words("Kim called yesterday.", forward) == (
            "Person_1 called yesterday."
        )

    def test_speaker_id_hint_key_still_dropped(self):
        """The speaker-id key guard is unconditional (never gated on
        scope) — a hallucinated ``{"speaker0": "Person_1"}`` hint
        (observed in real logs) is dropped forward AND reverse.
        """
        forward, reverse = _build_anonymization_mapping(
            {"speaker0": "Person_1"},
            speaker_name=None,
        )
        assert "speaker0" not in forward
        assert "Person_1" not in reverse


class TestNoEntityWalk:
    """The deterministic entity walk is DELETED
    outright, not scoped or gated. ``_build_anonymization_mapping`` no
    longer accepts an ``entities`` argument at all: the model's
    ``llm_mapping`` is the SOLE source of the built table.
    """

    def test_signature_has_no_entities_parameter(self):
        params = inspect.signature(_build_anonymization_mapping).parameters
        assert "entities" not in params
        # No ``person_prefix`` parameter either: the speaker-name-seeding
        # mint resolves its placeholder prefix directly via
        # ``paramem.config.taxonomy.entity_type_to_prefix`` — a leaf
        # package ``paramem.cloud`` may import without crossing the
        # cloud/graph boundary — rather than taking it as a caller-supplied
        # value.
        assert list(params) == ["llm_mapping", "speaker_name"]

    def test_empty_llm_mapping_and_no_speaker_name_yields_empty_tables(self):
        """With no entity walk, an empty model mapping and no runtime
        speaker name produces nothing to scrub — there is no
        graph-derived floor to fall back on."""
        forward, reverse = _build_anonymization_mapping({}, speaker_name=None)
        assert forward == {}
        assert reverse == {}

    def test_name_absent_from_llm_mapping_is_not_scrubbed(self):
        """A real name the model did not classify in scope is NOT
        independently discovered or minted by this builder — confirms
        there is no completeness floor over any entity inventory."""
        forward, _reverse = _build_anonymization_mapping(
            {"Alex": "Person_1"},
            speaker_name=None,
        )
        assert "Riley" not in forward


class TestSpeakerNameSeeding:
    """The one remaining seeding mechanism (the deterministic entity walk
    stays deleted): a
    runtime-known speaker display name the model never sees as an
    explicit prompt field is guaranteed coverage — reusing the model's
    own hint when it happened to name the speaker, or minting a fresh
    placeholder otherwise.
    """

    def test_reuses_exact_llm_hint_for_speaker_name(self):
        forward, reverse = _build_anonymization_mapping(
            {"Alex": "Person_1"},
            speaker_name="Alex",
        )
        assert forward["Alex"] == "Person_1"
        assert reverse["Person_1"] == "Alex"

    def test_reuses_full_name_llm_hint_for_first_name_speaker_name(self):
        forward, reverse = _build_anonymization_mapping(
            {"Alex Rivera": "Person_1"},
            speaker_name="Alex",
        )
        assert forward["Alex"] == "Person_1"
        assert reverse["Person_1"] == "Alex Rivera"

    def test_mints_fresh_placeholder_when_no_llm_hint_names_speaker(self):
        forward, reverse = _build_anonymization_mapping(
            {"Riley": "Person_1"},
            speaker_name="Alex",
        )
        assert forward["Alex"] == "Person_2"
        assert reverse["Person_2"] == "Alex"

    def test_speaker_name_already_in_mapping_is_left_untouched(self):
        forward, reverse = _build_anonymization_mapping(
            {"Alex": "Person_5"},
            speaker_name="Alex",
        )
        assert forward["Alex"] == "Person_5"
        assert reverse["Person_5"] == "Alex"


class TestReverseMapInversionAgreement:
    """The session tier and the graph tier
    (:func:`~paramem.graph.extractor.request_graph_enrichment`) both reach
    their reverse ``{placeholder: real_name}`` table through the SAME
    :func:`_build_anonymization_mapping` — the graph tier no longer
    inverts a forward map itself at all; it receives
    ``AnonymizedContract.reverse`` (built by
    :func:`~paramem.cloud.anonymize.anonymize`'s call to
    :func:`_build_anonymization_mapping`) directly as a caller-supplied
    argument. Before the fix the two sites disagreed on the tie-break for
    a many-to-one forward map (two real names scrubbed onto the SAME
    placeholder): the session tier was first-wins
    (``reverse.setdefault(v, k)``), the graph tier was last-wins (a plain
    dict comprehension) — so de-anonymizing the identical placeholder
    restored a DIFFERENT real name depending on which tier ran the deanon.

    Mutation: reintroduce a graph-tier-local inversion (raw
    ``invert_forward_mapping(mapping)`` or a hand-rolled dict comprehension
    inside ``request_graph_enrichment``) instead of taking
    ``payload.reverse`` as-is -> these tests fail.
    """

    _MANY_TO_ONE = {"Alice": "Person_1", "Bob": "Person_1"}

    def test_invert_forward_mapping_is_first_wins(self):
        assert invert_forward_mapping(self._MANY_TO_ONE) == {"Person_1": "Alice"}

    def test_session_tier_reverse_agrees_with_shared_helper(self):
        _forward, reverse = _build_anonymization_mapping(dict(self._MANY_TO_ONE), speaker_name=None)
        assert reverse == invert_forward_mapping(self._MANY_TO_ONE) == {"Person_1": "Alice"}

    def test_graph_tier_uses_the_same_shared_helper(self):
        """The graph tier's cloud round trip uses EXACTLY the reverse table
        :func:`_build_anonymization_mapping` produced — not a
        re-derivation — proven by round-tripping a placeholder through
        ``request_graph_enrichment`` and confirming it resolves to the
        SAME first-wins real name the shared helper picked.
        """
        from unittest.mock import patch

        from paramem.cloud.anonymize import AnonymizedContract
        from paramem.graph.extractor import request_graph_enrichment
        from paramem.graph.schema import SessionGraph

        forward, reverse = _build_anonymization_mapping(dict(self._MANY_TO_ONE), speaker_name=None)
        assert reverse == {"Person_1": "Alice"}
        # Realistic shape: Cloud can only propose a relation naming a
        # placeholder it was actually SHOWN — ``request_graph_enrichment``
        # derives the anonymized triples directly from ``triples`` +
        # ``payload.forward`` via ``insert_placeholders`` (no separate
        # ``anon_facts`` field on the contract).
        triples = [
            {
                "subject": "Alice",
                "predicate": "knows",
                "object": "acme",
                "relation_type": "factual",
                "speaker_id": "",
            }
        ]
        payload = AnonymizedContract(
            status="ok",
            forward=forward,
            reverse=reverse,
            anon_transcript="",
            declared=frozenset(reverse.keys()),
            norm_stats={"inverted": 0, "dropped": 0},
            rekey_dropped=0,
            raw="",
        )
        graph = SessionGraph(session_id="t", timestamp="", entities=[], relations=[])
        canned_raw = (
            '{"relations": [{"subject": "Person_1", "predicate": "knows", '
            '"object": "Person_1", "relation_type": "social", "confidence": 0.9}], '
            '"same_as": []}'
        )
        with patch("paramem.graph.extractor._cloud_call", return_value=canned_raw):
            result = request_graph_enrichment(
                triples=triples,
                payload=payload,
                graph=graph,
                api_key="test-key",
                provider="anthropic",
                filter_model="claude-sonnet-4-6",
            )

        assert result is not None
        new_rels, _same_as, _raw, _verdict = result
        assert len(new_rels) == 1
        assert new_rels[0]["subject"] == "Alice"
        assert new_rels[0]["object"] == "Alice"

    def test_graph_tier_hostile_speaker_value_hint_never_reaches_reverse(self):
        """The graph-tier twin of :class:`TestSpeakerAnchorReverseSkip` (above) —
        a hostile LLM hint scrubbing a real name onto the speaker anchor
        must never produce a ``speaker0``-keyed reverse entry reachable
        from the graph tier.  Since the graph tier's ONLY route to a
        reverse map is the caller-supplied ``payload.reverse`` (built by
        :func:`_build_anonymization_mapping`, whose speaker-value guard
        already dropped ``speaker0`` — see :class:`TestSpeakerAnchorReverseSkip`),
        there is no code path left in ``request_graph_enrichment`` that
        could reintroduce it.
        """
        forward, reverse = _build_anonymization_mapping({"RealName": "speaker0"}, speaker_name=None)
        assert "speaker0" not in reverse
        assert forward.get("RealName") == "speaker0"
