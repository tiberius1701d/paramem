"""``build_forward_table`` — canonical-equal sharing (never deletion),
per-prefix uniqueness, first-category-wins, the anchor fold, speaker-id
exclusion, and the closed speaker group.
"""

from __future__ import annotations

import itertools

from paramem.cloud.anonymize_steps import ScanResult
from paramem.cloud.placeholders import build_forward_table, invert_forward_mapping
from paramem.config.taxonomy import ScrubCategory
from paramem.utils.identity import is_speaker_id

PERSON = ScrubCategory(
    name="Person", prefix="Person", hints=("person name",), tagger_labels=("person",)
)
PROFILE = ScrubCategory(
    name="Profile", prefix="Profile", hints=("social media handle",), tagger_labels=("username",)
)
ADDRESS = ScrubCategory(
    name="Address", prefix="Address", hints=("street address",), tagger_labels=("address",)
)
EMAIL = ScrubCategory(
    name="Email", prefix="Email", hints=("email address",), tagger_labels=("email",)
)


def _scan(category: ScrubCategory, values: tuple[str, ...]):
    return ScanResult(category=category, values=values, dropped=())


def _tag_text(scans) -> str:
    """Every scanned surface, whole-word, space-joined — so
    ``build_forward_table``'s prune pass finds each surface this module's
    tests expect to survive live over the payload. ``tag_text`` is a
    required keyword (the prune pass reads it); this is the one helper
    every call in this module shares."""
    return " ".join(v for scan in scans for v in scan.values if isinstance(v, str))


def _forward(scans, **kwargs) -> dict[str, str]:
    """Call :func:`build_forward_table` with this module's own
    ``tag_text``/``identity_domain`` defaults and return ``.forward`` —
    the shape every test in this module asserts against.
    """
    kwargs.setdefault("tag_text", _tag_text(scans))
    kwargs.setdefault("identity_domain", None)
    return build_forward_table(scans, **kwargs).forward


def _derived_reverse(forward: dict[str, str]) -> dict[str, str]:
    """The exact reverse-derivation expression production uses after
    ``build_forward_table`` — copied from
    ``paramem.cloud.anonymize.anonymize``.
    """
    return invert_forward_mapping({k: v for k, v in forward.items() if not is_speaker_id(v)})


class TestCanonicallyEqualSurfacesShareOnePlaceholder:
    def test_lena_and_lena_lowercase_map_to_the_same_placeholder_as_two_keys(self) -> None:
        scans = (_scan(PERSON, ("Lena", "lena")),)
        forward = _forward(
            scans,
            anchor_names=frozenset(),
            speaker_id=None,
            speaker_name=None,
        )
        assert forward["Lena"] == forward["lena"]
        assert len(forward) == 2  # never deleted — each surface is its own key

    def test_substitute_whole_words_would_replace_both_surfaces(self) -> None:
        from paramem.cloud.placeholders import _substitute_whole_words

        scans = (_scan(PERSON, ("Lena", "lena")),)
        forward = _forward(
            scans,
            anchor_names=frozenset(),
            speaker_id=None,
            speaker_name=None,
        )
        text = "Lena and lena both attended."
        result = _substitute_whole_words(text, forward)
        assert forward["Lena"] not in ("Lena", "lena")
        placeholder = forward["Lena"]
        assert result == f"{placeholder} and {placeholder} both attended."


class TestPerPrefixUniqueness:
    def test_two_distinct_person_names_get_distinct_person_prefixed_placeholders(self) -> None:
        scans = (_scan(PERSON, ("Alex", "Sam")),)
        forward = _forward(
            scans,
            anchor_names=frozenset(),
            speaker_id=None,
            speaker_name=None,
        )
        assert forward["Alex"] != forward["Sam"]
        assert forward["Alex"].startswith("Person_")
        assert forward["Sam"].startswith("Person_")
        # No collision — mint_placeholder scans existing values.
        assert len({forward["Alex"], forward["Sam"]}) == 2


class TestFirstCategoryWinsIndependentOfPayloadPosition:
    def test_cross_category_duplicate_resolves_to_the_earlier_categorys_prefix(self) -> None:
        # "Alex" scanned under BOTH Person (earlier, category order) and
        # Profile (later) — Person's prefix wins regardless of payload
        # position, because the tie is decided by category order alone.
        scans = (
            _scan(PERSON, ("Alex",)),
            _scan(PROFILE, ("Alex",)),
        )
        forward = _forward(
            scans,
            anchor_names=frozenset(),
            speaker_id=None,
            speaker_name=None,
        )
        assert forward["Alex"].startswith("Person_")

    def test_reversing_category_order_still_wins_on_declaration_order_not_position(self) -> None:
        # Profile is now first in `scans` — first-category-wins tracks
        # declaration order of the `scans` sequence itself, so the winner
        # flips when the caller's own category order flips; payload
        # position never enters this decision (build_forward_table is
        # never given payload offsets at all).
        scans = (
            _scan(PROFILE, ("Alex",)),
            _scan(PERSON, ("Alex",)),
        )
        forward = _forward(
            scans,
            anchor_names=frozenset(),
            speaker_id=None,
            speaker_name=None,
        )
        assert forward["Alex"].startswith("Profile_")


class TestAnchorFoldWritesForwardAndSuppressesReverse:
    def test_anchor_named_value_folds_onto_speaker_id(self) -> None:
        scans = (_scan(PERSON, ("Alex",)),)
        forward = _forward(
            scans,
            anchor_names=frozenset({"Alex"}),
            speaker_id="speaker1",
            speaker_name=None,
        )
        reverse = _derived_reverse(forward)
        assert forward["Alex"] == "speaker1"
        assert (
            "speaker1" not in reverse
        )  # never restores a real name onto every speaker-subject fact


class TestSpeakerIdShapedKeyNeverEntersForward:
    def test_a_speaker_id_shaped_scanned_value_is_dropped(self) -> None:
        scans = (_scan(PERSON, ("speaker1", "Alex")),)
        forward = _forward(
            scans,
            anchor_names=frozenset(),
            speaker_id=None,
            speaker_name=None,
        )
        assert "speaker1" not in forward
        assert "Alex" in forward


class TestSurfaceContainmentSharing:
    def test_full_name_and_both_name_fragments_share_one_placeholder(self) -> None:
        # "Elena Varga" scanned first (the docstring's own first-occurrence
        # convention) — its shorter fragments each occur as a whole-word
        # sub-string of exactly one longer surface, so both fold onto it.
        scans = (_scan(PERSON, ("Elena Varga", "Varga", "Elena")),)
        forward = _forward(
            scans,
            anchor_names=frozenset(),
            speaker_id=None,
            speaker_name=None,
        )
        reverse = _derived_reverse(forward)

        placeholder = forward["Elena Varga"]
        assert forward["Varga"] == placeholder
        assert forward["Elena"] == placeholder
        assert len(forward) == 3  # each surface remains its own forward key

        # reverse keeps its existing first-forward-key-wins rule
        # (invert_forward_mapping) — "Elena Varga" was inserted first, so
        # it is what the shared placeholder resolves back to.
        assert reverse[placeholder] == "Elena Varga"
        assert len(reverse) == 1

    def test_substitute_whole_words_replaces_every_shared_surface(self) -> None:
        from paramem.cloud.placeholders import _substitute_whole_words

        scans = (_scan(PERSON, ("Elena Varga", "Varga", "Elena")),)
        forward = _forward(
            scans,
            anchor_names=frozenset(),
            speaker_id=None,
            speaker_name=None,
        )
        placeholder = forward["Elena Varga"]
        text = "Elena Varga called. Varga will visit. Elena said hi."
        result = _substitute_whole_words(text, forward)
        assert result == f"{placeholder} called. {placeholder} will visit. {placeholder} said hi."

    def test_a_fragment_contained_in_two_longer_surfaces_mints_its_own_placeholder(self) -> None:
        # "Elena" occurs as a whole-word sub-string of BOTH "Elena Varga"
        # and "Elena Fischer" — genuinely ambiguous, so it keeps whatever
        # placeholder it was originally minted, never guessing.
        scans = (_scan(PERSON, ("Elena", "Elena Varga", "Elena Fischer")),)
        forward = _forward(
            scans,
            anchor_names=frozenset(),
            speaker_id=None,
            speaker_name=None,
        )
        assert forward["Elena"] != forward["Elena Varga"]
        assert forward["Elena"] != forward["Elena Fischer"]
        assert forward["Elena Varga"] != forward["Elena Fischer"]
        assert len({forward["Elena"], forward["Elena Varga"], forward["Elena Fischer"]}) == 3

    def test_self_introduced_short_form_stays_on_the_speaker_anchor(self) -> None:
        # The speaker group is excluded from the judged groups entirely
        # (it is never judged and never a merge target) — "Priya" keeps
        # speaker0 even though "Priya Sharma" (a longer, ordinary surface)
        # is present.
        scans = (_scan(PERSON, ("Priya", "Priya Sharma")),)
        forward = _forward(
            scans,
            anchor_names=frozenset({"Priya"}),
            speaker_id="speaker0",
            speaker_name=None,
        )
        reverse = _derived_reverse(forward)
        assert forward["Priya"] == "speaker0"
        assert forward["Priya Sharma"] != "speaker0"
        assert forward["Priya Sharma"].startswith("Person_")
        assert "speaker0" not in reverse

    def test_containment_case_sensitivity_lowercase_bill_does_not_fold_onto_capitalized_surface(
        self,
    ) -> None:
        # "bill" and "Bill Murray" are NOT canonically equal (default
        # canonical() casefolds, but "bill" != "bill murray"), so this
        # exercises the CONTAINMENT rule specifically, on its
        # case/diacritic-preserving `mode="spaces"` canonical form: a
        # case-sensitive whole-word scan never matches lowercase "bill"
        # inside "Bill Murray" (capital B), so the two stay separate —
        # never a guessed fold across a case difference.
        scans = (_scan(PERSON, ("bill", "Bill Murray")),)
        forward = _forward(
            scans,
            anchor_names=frozenset(),
            speaker_id=None,
            speaker_name=None,
        )
        assert forward["bill"] != forward["Bill Murray"]

    def test_non_whole_word_substring_does_not_share(self) -> None:
        # "Ann" occurs inside "Annika Berg" only as a substring, not a
        # whole word (immediately followed by "ika") — no sharing.
        scans = (_scan(PERSON, ("Ann", "Annika Berg")),)
        forward = _forward(
            scans,
            anchor_names=frozenset(),
            speaker_id=None,
            speaker_name=None,
        )
        assert forward["Ann"] != forward["Annika Berg"]

    def test_sharing_never_crosses_category_boundaries(self) -> None:
        # "Varga" as a Person surface never shares with an Address surface
        # that also contains the word "Varga" — containment sharing is
        # evaluated per category (per scan), never across scans.
        scans = (
            _scan(PERSON, ("Varga",)),
            _scan(ADDRESS, ("Varga Street 5",)),
        )
        forward = _forward(
            scans,
            anchor_names=frozenset(),
            speaker_id=None,
            speaker_name=None,
        )
        assert forward["Varga"] != forward["Varga Street 5"]
        assert forward["Varga"].startswith("Person_")
        assert forward["Varga Street 5"].startswith(ADDRESS.prefix + "_")


class TestDerivedReverseNamesTheContainer:
    """Order invariance on the longest-canonical-form-first containment
    pass: a transitive containment chain (``Ann`` is contained in ``Ann
    Marie``, which is contained in ``Ann Marie Bell``) is ONE entity
    regardless of scan order -- every fragment folds onto the outermost
    container, and the derived ``reverse`` map (production's own
    ``invert_forward_mapping({... not is_speaker_id ...})`` expression)
    always names that outermost container, never an intermediate or
    shortest fragment.
    """

    def test_short_form_scanned_before_its_container_still_resolves_to_the_full_name(
        self,
    ) -> None:
        # "Elena" is scanned FIRST, but containers are settled
        # longest-canonical-form-first regardless of scan order -- so the
        # derived reverse resolves back to the full name, never the
        # fragment that happened to appear earlier in the payload.
        scans = (_scan(PERSON, ("Elena", "Elena Varga")),)
        forward = _forward(
            scans,
            anchor_names=frozenset(),
            speaker_id=None,
            speaker_name=None,
        )
        reverse = _derived_reverse(forward)
        placeholder = forward["Elena"]
        assert forward["Elena Varga"] == placeholder
        assert reverse == {placeholder: "Elena Varga"}

    def test_three_way_fragment_ascending_order_forms_one_transitive_chain(self) -> None:
        # "Ann" is contained in "Ann Marie", which is contained in "Ann
        # Marie Bell" -- a transitive chain, not a two-way ambiguity:
        # containers are settled longest-first, so by the time "Ann" is
        # judged, "Ann Marie" already shares "Ann Marie Bell"'s
        # placeholder and both of "Ann"'s containers resolve to that one
        # placeholder. All three surfaces share one placeholder; the
        # derived reverse names the outermost container alone.
        scans = (_scan(PERSON, ("Ann", "Ann Marie", "Ann Marie Bell")),)
        forward = _forward(
            scans,
            anchor_names=frozenset(),
            speaker_id=None,
            speaker_name=None,
        )
        reverse = _derived_reverse(forward)
        placeholder = forward["Ann Marie Bell"]
        assert forward["Ann"] == placeholder
        assert forward["Ann Marie"] == placeholder
        assert reverse == {placeholder: "Ann Marie Bell"}

    def test_three_way_fragment_descending_order_forms_the_same_chain(self) -> None:
        # Same three surfaces, longest-to-shortest scan order -- the
        # partition and the derived reverse are identical to the
        # ascending-order case: scan order must not change the verdict.
        scans = (_scan(PERSON, ("Ann Marie Bell", "Ann Marie", "Ann")),)
        forward = _forward(
            scans,
            anchor_names=frozenset(),
            speaker_id=None,
            speaker_name=None,
        )
        reverse = _derived_reverse(forward)
        placeholder = forward["Ann Marie Bell"]
        assert forward["Ann"] == placeholder
        assert forward["Ann Marie"] == placeholder
        assert reverse == {placeholder: "Ann Marie Bell"}

    def test_every_permutation_of_the_transitive_chain_yields_the_same_partition(self) -> None:
        # All six scan-order permutations of the same three surfaces must
        # produce the identical partition (all three surfaces sharing one
        # placeholder) and the identical derived-reverse surface -- the
        # containment pass's longest-canonical-form-first ordering is
        # independent of the caller's payload order.
        surfaces = ("Ann", "Ann Marie", "Ann Marie Bell")
        for perm in itertools.permutations(surfaces):
            scans = (_scan(PERSON, perm),)
            forward = _forward(
                scans,
                anchor_names=frozenset(),
                speaker_id=None,
                speaker_name=None,
            )
            reverse = _derived_reverse(forward)
            placeholders = {forward[s] for s in surfaces}
            assert len(placeholders) == 1, f"order {perm} split the chain: {forward}"
            (placeholder,) = placeholders
            assert reverse == {placeholder: "Ann Marie Bell"}, f"order {perm}: {reverse}"


class TestSharingRespectsFirstCategoryOwnership:
    def test_a_value_scanned_in_two_categories_keeps_the_minting_categorys_placeholder(
        self,
    ) -> None:
        # "alex.stone@example.de" is scanned under BOTH Person (first,
        # category order) and Email (second) -- it is minted once, in
        # Person, and the Email category's own containment pass must not
        # re-point it onto anything from Email's own surface set, even
        # though the value is present in that category's scan too.
        person = _scan(PERSON, ("alex.stone@example.de",))
        email = EMAIL
        email_scan = _scan(email, ("alex.stone@example.de", "long.alex.stone@example.de"))
        forward = _forward(
            (person, email_scan),
            anchor_names=frozenset(),
            speaker_id=None,
            speaker_name=None,
        )
        assert forward["alex.stone@example.de"].startswith("Person_")
        # The longer email has no whole-word container of its own (it is
        # not a substring of the shared surface -- it's the other way
        # around) and mints its own Email-prefixed placeholder.
        assert forward["long.alex.stone@example.de"].startswith("Email_")
        assert forward["long.alex.stone@example.de"] != forward["alex.stone@example.de"]


class TestCaseAndDiacriticContainersDoNotCreateAmbiguity:
    def test_case_and_diacritic_variants_of_one_container_share_a_single_placeholder(
        self,
    ) -> None:
        # "Elena Varga" and "Elena VARGA" already share one placeholder
        # from the canonical-equality mint loop (case-folded identical) --
        # so when "Elena" is checked for containment, both count as ONE
        # container, not two, and folds onto it too.
        scans = (_scan(PERSON, ("Elena Varga", "Elena VARGA", "Elena")),)
        forward = _forward(
            scans,
            anchor_names=frozenset(),
            speaker_id=None,
            speaker_name=None,
        )
        placeholder = forward["Elena Varga"]
        assert forward["Elena VARGA"] == placeholder
        assert forward["Elena"] == placeholder

    def test_genuine_ambiguity_across_two_distinct_containers_still_isolates_the_fragment(
        self,
    ) -> None:
        # "Elena Varga" and "Elena Fischer" are genuinely distinct
        # entities (not case/diacritic variants of one another) -- "Elena"
        # is contained in both and stays on its own placeholder.
        scans = (_scan(PERSON, ("Elena Varga", "Elena Fischer", "Elena")),)
        forward = _forward(
            scans,
            anchor_names=frozenset(),
            speaker_id=None,
            speaker_name=None,
        )
        assert forward["Elena"] != forward["Elena Varga"]
        assert forward["Elena"] != forward["Elena Fischer"]
        assert forward["Elena Varga"] != forward["Elena Fischer"]


class TestAnchoredContainerDoesNotCarryItsFragment:
    def test_an_unattested_fragment_of_the_anchored_surface_keeps_its_own_placeholder(
        self,
    ) -> None:
        # The anchor call names "Elena Varga" itself (not the shorter
        # fragment) -- the anchor fold writes it directly onto speaker_id.
        # The speaker group is CLOSED: containment never merges anything
        # into it, so "Elena" -- unattested on its own -- mints its own
        # ordinary placeholder instead of riding the fold.
        scans = (_scan(PERSON, ("Elena Varga", "Elena")),)
        forward = _forward(
            scans,
            anchor_names=frozenset({"Elena Varga"}),
            speaker_id="speaker0",
            speaker_name=None,
        )
        reverse = _derived_reverse(forward)
        assert forward["Elena Varga"] == "speaker0"
        assert forward["Elena"].startswith("Person_")
        assert forward["Elena"] != "speaker0"
        assert reverse == {"Person_1": "Elena"}


class TestCanonicalEqualityGroupMovesTogetherOnContainmentRepoint:
    """A canonical-equality group (surfaces that already share one
    placeholder from the mint loop, e.g. ``lena``/``Lena``) is ONE surface
    for every containment-pass decision: when one member re-points onto a
    container's placeholder, every other member of the same group moves
    with it. The group is never split -- one member keeping its
    pre-containment placeholder while a canonically-equal sibling
    re-points would be exactly that split.
    """

    def test_case_variant_pair_and_their_container_share_one_placeholder_every_order(
        self,
    ) -> None:
        # "lena" and "Lena" are canonically equal (share one placeholder
        # from the mint loop); "Lena Marie Fischer" is their sole
        # container. All three must end up on the SAME placeholder --
        # never "lena"/"Lena" splitting so that only one of the pair
        # re-points onto the container while the other stays behind on
        # its own pre-containment placeholder -- and the result must not
        # depend on scan order.
        values = ("lena", "Marta", "Lena Marie Fischer", "Lena")
        for perm in itertools.permutations(values):
            scans = (_scan(PERSON, perm),)
            forward = _forward(
                scans,
                anchor_names=frozenset(),
                speaker_id=None,
                speaker_name=None,
            )
            reverse = _derived_reverse(forward)

            group_placeholders = {forward["lena"], forward["Lena"], forward["Lena Marie Fischer"]}
            assert len(group_placeholders) == 1, (
                f"order {perm} split the canonical-equality group: {forward}"
            )
            (placeholder,) = group_placeholders
            assert forward["Marta"] != placeholder, f"order {perm}: Marta wrongly shared: {forward}"
            assert reverse[placeholder] == "Lena Marie Fischer", f"order {perm}: {reverse}"

    def test_case_variant_group_of_four_all_share_one_placeholder(self) -> None:
        # "Elena Varga" / "Elena VARGA" are canonically equal to each
        # other (their own group), and "Elena" / "elena" are canonically
        # equal to each other (a second group) that is contained in the
        # first -- containment must re-point BOTH members of the
        # "Elena"/"elena" group onto the SAME (shared) container
        # placeholder, collapsing all four surfaces onto one placeholder.
        scans = (_scan(PERSON, ("Elena Varga", "Elena VARGA", "Elena", "elena")),)
        forward = _forward(
            scans,
            anchor_names=frozenset(),
            speaker_id=None,
            speaker_name=None,
        )
        reverse = _derived_reverse(forward)

        placeholder = forward["Elena Varga"]
        assert forward["Elena VARGA"] == placeholder
        assert forward["Elena"] == placeholder
        assert forward["elena"] == placeholder
        assert len({forward[v] for v in ("Elena Varga", "Elena VARGA", "Elena", "elena")}) == 1
        assert reverse == {placeholder: "Elena Varga"}

    def test_genuine_ambiguity_isolates_the_whole_fragment_group_not_just_one_member(
        self,
    ) -> None:
        # "Elena" and "elena" (one canonical-equality group) are each
        # contained in BOTH "Elena Varga" and "Elena Fischer" -- two
        # distinct containers, so the group is genuinely ambiguous and
        # must keep its OWN (shared) placeholder, distinct from both
        # containers -- and the two group members must still agree with
        # each other, never one splitting off onto a container while the
        # other stays isolated.
        scans = (_scan(PERSON, ("Elena Varga", "Elena Fischer", "Elena", "elena")),)
        forward = _forward(
            scans,
            anchor_names=frozenset(),
            speaker_id=None,
            speaker_name=None,
        )

        assert forward["Elena"] == forward["elena"]
        assert forward["Elena"] != forward["Elena Varga"]
        assert forward["Elena"] != forward["Elena Fischer"]
        assert forward["elena"] != forward["Elena Varga"]
        assert forward["elena"] != forward["Elena Fischer"]
        assert forward["Elena Varga"] != forward["Elena Fischer"]

    def test_anchored_containers_canonical_equality_group_keeps_its_own_shared_placeholder(
        self,
    ) -> None:
        # The anchor call names "Lena Marie Fischer" itself -- the anchor
        # fold writes it directly onto speaker_id. The speaker group is
        # CLOSED: containment never merges "Lena"/"lena" into it. The
        # canonical-twin rule still binds "Lena" and "lena" to EACH OTHER
        # (never split apart), so the two share one ordinary placeholder,
        # distinct from speaker_id.
        scans = (_scan(PERSON, ("Lena Marie Fischer", "Lena", "lena")),)
        forward = _forward(
            scans,
            anchor_names=frozenset({"Lena Marie Fischer"}),
            speaker_id="speaker0",
            speaker_name=None,
        )
        reverse = _derived_reverse(forward)
        assert forward["Lena Marie Fischer"] == "speaker0"
        assert forward["Lena"] == forward["lena"]
        assert forward["Lena"] != "speaker0"
        assert reverse == {forward["Lena"]: "Lena"}


class TestNumbersAreContiguousPerPrefix:
    def test_numbers_run_from_one_with_no_holes_after_a_containment_share(self) -> None:
        """A containment merge leaves one surviving group per prefix, so
        the one placeholder it mints is ``Person_1`` — no ``Person_2``
        anywhere in the table.
        """
        scans = (_scan(PERSON, ("Elena", "Elena Varga")),)
        forward = _forward(
            scans,
            anchor_names=frozenset(),
            speaker_id=None,
            speaker_name=None,
        )
        assert forward["Elena"] == "Person_1"
        assert forward["Elena Varga"] == "Person_1"
        assert "Person_2" not in forward.values()

    def test_numbers_run_from_one_with_no_holes_after_an_inert_prune(self) -> None:
        """A scanned surface absent from ``tag_text`` is pruned before
        minting, so the next surviving surface still takes ``Person_1`` —
        no hole left where the pruned surface would have minted.
        """
        scans = (_scan(PERSON, ("Ghost", "Alex")),)
        forward = _forward(
            scans,
            tag_text="Alex only here",
            anchor_names=frozenset(),
            speaker_id=None,
            speaker_name=None,
        )
        assert "Ghost" not in forward
        assert forward["Alex"] == "Person_1"

    def test_first_occurrence_order_decides_the_number_within_a_prefix(self) -> None:
        """Two independent surfaces number in first-occurrence (scan)
        order; a group formed by a containment merge takes its earliest
        member's ordinal, so it still numbers at the position its
        founding member occupied, not the position its container
        occupied.
        """
        scans = (_scan(PERSON, ("Sam", "Alex")),)
        forward = _forward(
            scans,
            anchor_names=frozenset(),
            speaker_id=None,
            speaker_name=None,
        )
        assert forward["Sam"] == "Person_1"
        assert forward["Alex"] == "Person_2"

        # "Elena" (order 1) is absorbed into "Elena Varga" (order 2) on
        # containment; the merged group keeps "Elena"'s earlier ordinal,
        # so it numbers second overall (after "Bob", order 0), not third.
        scans = (_scan(PERSON, ("Bob", "Elena", "Elena Varga")),)
        forward = _forward(
            scans,
            anchor_names=frozenset(),
            speaker_id=None,
            speaker_name=None,
        )
        assert forward["Bob"] == "Person_1"
        assert forward["Elena"] == "Person_2"
        assert forward["Elena Varga"] == "Person_2"


class TestNamesakeKeepsItsOwnPlaceholder:
    def test_a_longer_unattested_surface_sharing_the_enrolled_first_name_does_not_fold(
        self,
    ) -> None:
        """A scanned surface merely CONSISTENT with the enrolled name
        (containing it as a leading word) but never attested does not
        fold onto the speaker token — only the enrolled name itself does.
        """
        scans = (_scan(PERSON, ("Alex", "Alex Rivera")),)
        forward = _forward(
            scans,
            anchor_names=frozenset(),
            speaker_id="speaker1",
            speaker_name="Alex",
        )
        assert forward["Alex"] == "speaker1"
        assert forward["Alex Rivera"].startswith("Person_")


class TestAttestationIsFilteredByEnrolledNameConsistency:
    def test_an_attested_extension_of_the_enrolled_name_folds(self) -> None:
        """An attested surface that extends the enrolled name (the
        enrolled name plus a trailing word) is consistent with it and
        folds onto the speaker token."""
        scans = (_scan(PERSON, ("Alex Rivera",)),)
        forward = _forward(
            scans,
            anchor_names=frozenset({"Alex Rivera"}),
            speaker_id="speaker1",
            speaker_name="Alex",
        )
        reverse = _derived_reverse(forward)
        assert forward["Alex Rivera"] == "speaker1"
        assert reverse == {}

    def test_an_attested_short_form_of_a_full_enrolled_name_folds(self) -> None:
        """Consistency is symmetric: an attested surface that is a
        LEADING fragment of a full enrolled name also folds."""
        scans = (_scan(PERSON, ("Alex",)),)
        forward = _forward(
            scans,
            anchor_names=frozenset({"Alex"}),
            speaker_id="speaker1",
            speaker_name="Alex Rivera",
        )
        assert forward["Alex"] == "speaker1"

    def test_an_attested_name_that_is_not_the_enrolled_name_keeps_a_placeholder(self) -> None:
        """An attested surface that is neither equal to nor an
        extension/short-form of the enrolled name (a namesake) is
        refused by the fold and mints an ordinary placeholder instead."""
        scans = (_scan(PERSON, ("Mira",)),)
        forward = _forward(
            scans,
            anchor_names=frozenset({"Mira"}),
            speaker_id="speaker1",
            speaker_name="Alex",
        )
        assert forward["Mira"].startswith("Person_")
        assert "speaker1" not in forward.values()


class TestAnonymousVoiceSpeakerFoldsOnAttestationAlone:
    def test_no_enrolled_name_folds_the_attested_surface(self) -> None:
        """With no display name to compare against (``speaker_name`` is
        ``None``, or token-shaped and therefore treated as absent),
        attestation alone decides the fold."""
        for speaker_name in (None, "speaker1"):
            scans = (_scan(PERSON, ("Mira",)),)
            forward = _forward(
                scans,
                anchor_names=frozenset({"Mira"}),
                speaker_id="speaker1",
                speaker_name=speaker_name,
            )
            assert forward["Mira"] == "speaker1", f"speaker_name={speaker_name!r}: {forward}"


class TestDeferralFoldsOnEqualityAlone:
    def test_an_empty_anchor_set_still_folds_the_enrolled_name(self) -> None:
        """With the ANCHOR call never issued (``anchor_names`` empty), a
        surface equal to the enrolled name still folds — equality with
        the enrolled name needs no attestation."""
        scans = (_scan(PERSON, ("Alex",)),)
        forward = _forward(
            scans,
            anchor_names=frozenset(),
            speaker_id="speaker1",
            speaker_name="Alex",
        )
        assert forward["Alex"] == "speaker1"


class TestPersonScrubbingOffMakesNoSpeakerEntry:
    def test_no_person_category_means_no_fold_and_no_enrolled_name_key(self) -> None:
        """With no person-name category among the active scan categories,
        the speaker's own name is never entered and never folds,
        whatever the caller passed."""
        scans = (_scan(ADDRESS, ()),)
        forward = _forward(
            scans,
            tag_text="Alex is here",
            anchor_names=frozenset({"Alex"}),
            speaker_id="speaker1",
            speaker_name="Alex",
        )
        assert "Alex" not in forward
        assert "speaker1" not in forward.values()


class TestSpeakerNameWithoutASpeakerIdIsNotConsumed:
    def test_no_enrolled_name_key_and_no_speaker_value(self) -> None:
        """A ``speaker_name`` without a well-shaped ``speaker_id`` is not
        consumed at all: no enrolled-name key is entered, and no value in
        the table is ever speaker-id-shaped."""
        for speaker_id in (None, "not-a-token"):
            scans = (_scan(PERSON, ()),)
            forward = _forward(
                scans,
                tag_text="Alex is here",
                anchor_names=frozenset({"Alex"}),
                speaker_id=speaker_id,
                speaker_name="Alex",
            )
            assert "Alex" not in forward, f"speaker_id={speaker_id!r}: {forward}"
            assert not any(is_speaker_id(v) for v in forward.values()), (
                f"speaker_id={speaker_id!r}: {forward}"
            )


class TestSpeakerGroupIsClosed:
    def test_a_scanned_fragment_of_the_enrolled_name_keeps_its_own_placeholder(self) -> None:
        """A scanned fragment of the enrolled name, itself unattested, is
        an ordinary surface — the speaker group is closed on the
        containment side, so nothing joins it by being a piece of the
        enrolled name."""
        scans = (_scan(PERSON, ("Rivera",)),)
        forward = _forward(
            scans,
            anchor_names=frozenset(),
            speaker_id="speaker1",
            speaker_name="Alex Rivera",
        )
        assert forward["Rivera"].startswith("Person_")

    def test_an_unattested_fragment_of_an_attested_surface_keeps_its_own_placeholder(self) -> None:
        """An unattested fragment of an attested (folded) surface is an
        ordinary surface — the fold decides on evidence about the
        surface itself, never by containment inside a surface that had
        the evidence."""
        scans = (_scan(PERSON, ("Elena Varga", "Elena")),)
        forward = _forward(
            scans,
            anchor_names=frozenset({"Elena Varga"}),
            speaker_id="speaker1",
            speaker_name=None,
        )
        reverse = _derived_reverse(forward)
        assert forward["Elena Varga"] == "speaker1"
        assert forward["Elena"].startswith("Person_")
        assert reverse == {"Person_1": "Elena"}

    def test_a_speaker_group_surface_still_counts_toward_containment_ambiguity(self) -> None:
        """A surface folded onto the speaker group still counts as a
        container for the ambiguity test on an ordinary group's
        containment pass, exactly like any other surface."""
        scans = (_scan(PERSON, ("Elena Varga", "Elena Fischer", "Elena")),)
        forward = _forward(
            scans,
            anchor_names=frozenset({"Elena Varga"}),
            speaker_id="speaker1",
            speaker_name=None,
        )
        # "Elena" sits inside both "Elena Varga" (folded onto the speaker
        # group) and "Elena Fischer" (an ordinary group) — two distinct
        # containers, so it keeps its own placeholder, unchanged from the
        # unattested/no-speaker case.
        assert forward["Elena"].startswith("Person_")
        assert forward["Elena"] != forward["Elena Varga"]
        assert forward["Elena"] != forward["Elena Fischer"]


class TestEnrolledNameIsPrunedLikeAnyKey:
    def test_an_enrolled_name_absent_from_the_payload_is_dropped(self) -> None:
        """An enrolled name the payload never contains is pruned like any
        other forward-table key, and its inert record carries the
        speaker group's own (empty) category."""
        scans = (_scan(PERSON, ()),)
        table = build_forward_table(
            scans,
            tag_text="nothing relevant here",
            anchor_names=frozenset(),
            speaker_id="speaker1",
            speaker_name="Alex Morgan",
            identity_domain=None,
        )
        assert "Alex Morgan" not in table.forward
        assert len(table.inert_entries) == 1
        assert table.inert_entries[0]["category"] == ""


class TestFoldedCanonicalTwinFolds:
    def test_a_case_variant_of_a_folded_surface_folds_too(self) -> None:
        """A case/diacritic twin of a surface that folds onto the speaker
        group is registered onto that same group by default-canonical
        form, so it folds too."""
        scans = (_scan(PERSON, ("Priya", "priya")),)
        forward = _forward(
            scans,
            anchor_names=frozenset({"Priya"}),
            speaker_id="speaker0",
            speaker_name=None,
        )
        assert forward["Priya"] == "speaker0"
        assert forward["priya"] == "speaker0"


class TestReturnValueInvariant:
    def test_every_minted_number_is_carried_by_a_key(self) -> None:
        """For every prefix that mints at least one placeholder, the set
        of numbers actually used is exactly ``1..N`` with no holes —
        every minted number is carried by at least one surviving key."""
        scans = (_scan(PERSON, ("Sam", "Alex", "Ghost")),)
        forward = _forward(
            scans,
            tag_text="Sam Alex",
            anchor_names=frozenset(),
            speaker_id=None,
            speaker_name=None,
        )
        by_prefix: dict[str, set[int]] = {}
        for value in forward.values():
            prefix, _, tail = value.rpartition("_")
            by_prefix.setdefault(prefix, set()).add(int(tail))
        for prefix, numbers in by_prefix.items():
            assert numbers == set(range(1, len(numbers) + 1)), f"{prefix}: {numbers}"


class TestIdentityDomainReconciliationRunsBeforePrune:
    def test_a_domain_match_is_rekeyed_before_the_prune_and_a_miss_is_dropped_and_counted(
        self,
    ) -> None:
        """Reconciliation (pass 3) runs BEFORE pruning (pass 4): "alex" is
        re-keyed onto the identity domain's "Alex" surface first, so the
        prune then tests THAT surface against ``tag_text`` — which is what
        actually appears there — rather than the original scanned surface,
        which never appears verbatim in this payload. "riley" has no
        domain match at all, so it is dropped by reconciliation itself
        (counted into ``rekey_dropped``) and never reaches the prune.
        """
        scans = (_scan(PERSON, ("alex", "riley")),)
        table = build_forward_table(
            scans,
            tag_text="Alex mentioned something.",
            anchor_names=frozenset(),
            speaker_id=None,
            speaker_name=None,
            identity_domain=["Alex"],
        )
        assert table.forward == {"Alex": "Person_1"}
        assert "alex" not in table.forward
        assert "riley" not in table.forward
        assert table.rekey_dropped == 1
