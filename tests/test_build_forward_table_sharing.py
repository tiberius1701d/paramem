"""``build_forward_table`` — canonical-equal sharing (never deletion),
per-prefix uniqueness, first-category-wins, the anchor fold, speaker-id
exclusion, and speaker-name seeding.
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


def _derived_reverse(forward: dict[str, str]) -> dict[str, str]:
    """The exact reverse-derivation expression production uses after
    ``build_forward_table`` — copied from
    ``paramem.cloud.anonymize.anonymize`` (~line 672).
    """
    return invert_forward_mapping({k: v for k, v in forward.items() if not is_speaker_id(v)})


class TestCanonicallyEqualSurfacesShareOnePlaceholder:
    def test_lena_and_lena_lowercase_map_to_the_same_placeholder_as_two_keys(self) -> None:
        scans = (_scan(PERSON, ("Lena", "lena")),)
        forward = build_forward_table(
            scans, anchor_names=frozenset(), speaker_id=None, speaker_name=None
        )
        assert forward["Lena"] == forward["lena"]
        assert len(forward) == 2  # never deleted — each surface is its own key

    def test_substitute_whole_words_would_replace_both_surfaces(self) -> None:
        from paramem.cloud.placeholders import _substitute_whole_words

        scans = (_scan(PERSON, ("Lena", "lena")),)
        forward = build_forward_table(
            scans, anchor_names=frozenset(), speaker_id=None, speaker_name=None
        )
        text = "Lena and lena both attended."
        result = _substitute_whole_words(text, forward)
        assert forward["Lena"] not in ("Lena", "lena")
        placeholder = forward["Lena"]
        assert result == f"{placeholder} and {placeholder} both attended."


class TestPerPrefixUniqueness:
    def test_two_distinct_person_names_get_distinct_person_prefixed_placeholders(self) -> None:
        scans = (_scan(PERSON, ("Alex", "Sam")),)
        forward = build_forward_table(
            scans, anchor_names=frozenset(), speaker_id=None, speaker_name=None
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
        forward = build_forward_table(
            scans, anchor_names=frozenset(), speaker_id=None, speaker_name=None
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
        forward = build_forward_table(
            scans, anchor_names=frozenset(), speaker_id=None, speaker_name=None
        )
        assert forward["Alex"].startswith("Profile_")


class TestAnchorFoldWritesForwardAndSuppressesReverse:
    def test_anchor_named_value_folds_onto_speaker_id(self) -> None:
        scans = (_scan(PERSON, ("Alex",)),)
        forward = build_forward_table(
            scans, anchor_names=frozenset({"Alex"}), speaker_id="speaker1", speaker_name=None
        )
        reverse = _derived_reverse(forward)
        assert forward["Alex"] == "speaker1"
        assert (
            "speaker1" not in reverse
        )  # never restores a real name onto every speaker-subject fact


class TestSpeakerIdShapedKeyNeverEntersForward:
    def test_a_speaker_id_shaped_scanned_value_is_dropped(self) -> None:
        scans = (_scan(PERSON, ("speaker1", "Alex")),)
        forward = build_forward_table(
            scans, anchor_names=frozenset(), speaker_id=None, speaker_name=None
        )
        assert "speaker1" not in forward
        assert "Alex" in forward


class TestSpeakerNameSeeding:
    def test_exact_match_reuses_the_already_minted_placeholder(self) -> None:
        scans = (_scan(PERSON, ("Alex",)),)
        forward = build_forward_table(
            scans, anchor_names=frozenset(), speaker_id=None, speaker_name="Alex"
        )
        # "Alex" is already a key -- seeding must not overwrite or duplicate it.
        assert forward["Alex"].startswith("Person_")
        assert len(forward) == 1

    def test_full_name_match_reuses_the_full_names_placeholder(self) -> None:
        # A scanned surface that is a FULL name ("Alex Rivera") reuses its
        # placeholder for the speaker's own (shorter) display name ("Alex")
        # — the docstring's own example pairing.
        scans = (_scan(PERSON, ("Alex Rivera",)),)
        forward = build_forward_table(
            scans, anchor_names=frozenset(), speaker_id=None, speaker_name="Alex"
        )
        assert forward["Alex"] == forward["Alex Rivera"]

    def test_no_match_mints_a_fresh_placeholder_for_the_speaker_name(self) -> None:
        scans = (_scan(PERSON, ("Sam",)),)
        forward = build_forward_table(
            scans, anchor_names=frozenset(), speaker_id=None, speaker_name="Alex"
        )
        assert "Alex" in forward
        assert forward["Alex"] != forward["Sam"]

    def test_speaker_id_shaped_speaker_name_is_never_seeded(self) -> None:
        scans = (_scan(PERSON, ("Sam",)),)
        forward = build_forward_table(
            scans, anchor_names=frozenset(), speaker_id=None, speaker_name="speaker1"
        )
        assert "speaker1" not in forward


class TestSurfaceContainmentSharing:
    def test_full_name_and_both_name_fragments_share_one_placeholder(self) -> None:
        # "Elena Varga" scanned first (the docstring's own first-occurrence
        # convention) — its shorter fragments each occur as a whole-word
        # sub-string of exactly one longer surface, so both fold onto it.
        scans = (_scan(PERSON, ("Elena Varga", "Varga", "Elena")),)
        forward = build_forward_table(
            scans, anchor_names=frozenset(), speaker_id=None, speaker_name=None
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
        forward = build_forward_table(
            scans, anchor_names=frozenset(), speaker_id=None, speaker_name=None
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
        forward = build_forward_table(
            scans, anchor_names=frozenset(), speaker_id=None, speaker_name=None
        )
        assert forward["Elena"] != forward["Elena Varga"]
        assert forward["Elena"] != forward["Elena Fischer"]
        assert forward["Elena Varga"] != forward["Elena Fischer"]
        assert len({forward["Elena"], forward["Elena Varga"], forward["Elena Fischer"]}) == 3

    def test_self_introduced_short_form_stays_on_the_speaker_anchor(self) -> None:
        # The rule runs AFTER the anchor fold and skips a value already
        # resolved to the speaker anchor — "Priya" keeps speaker0 even
        # though "Priya Sharma" (a longer, ordinary surface) is present.
        scans = (_scan(PERSON, ("Priya", "Priya Sharma")),)
        forward = build_forward_table(
            scans, anchor_names=frozenset({"Priya"}), speaker_id="speaker0", speaker_name=None
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
        forward = build_forward_table(
            scans, anchor_names=frozenset(), speaker_id=None, speaker_name=None
        )
        assert forward["bill"] != forward["Bill Murray"]

    def test_non_whole_word_substring_does_not_share(self) -> None:
        # "Ann" occurs inside "Annika Berg" only as a substring, not a
        # whole word (immediately followed by "ika") — no sharing.
        scans = (_scan(PERSON, ("Ann", "Annika Berg")),)
        forward = build_forward_table(
            scans, anchor_names=frozenset(), speaker_id=None, speaker_name=None
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
        forward = build_forward_table(
            scans, anchor_names=frozenset(), speaker_id=None, speaker_name=None
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
        forward = build_forward_table(
            scans, anchor_names=frozenset(), speaker_id=None, speaker_name=None
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
        forward = build_forward_table(
            scans, anchor_names=frozenset(), speaker_id=None, speaker_name=None
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
        forward = build_forward_table(
            scans, anchor_names=frozenset(), speaker_id=None, speaker_name=None
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
            forward = build_forward_table(
                scans, anchor_names=frozenset(), speaker_id=None, speaker_name=None
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
        forward = build_forward_table(
            (person, email_scan), anchor_names=frozenset(), speaker_id=None, speaker_name=None
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
        forward = build_forward_table(
            scans, anchor_names=frozenset(), speaker_id=None, speaker_name=None
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
        forward = build_forward_table(
            scans, anchor_names=frozenset(), speaker_id=None, speaker_name=None
        )
        assert forward["Elena"] != forward["Elena Varga"]
        assert forward["Elena"] != forward["Elena Fischer"]
        assert forward["Elena Varga"] != forward["Elena Fischer"]


class TestAnchoredContainerFoldIsTotal:
    def test_anchor_and_its_fragment_both_fold_onto_speaker_id_with_empty_reverse(
        self,
    ) -> None:
        # The anchor call names "Elena Varga" itself (not the shorter
        # fragment) -- the anchor fold writes it directly onto speaker_id,
        # and the containment pass then folds "Elena" onto the anchored
        # placeholder too, since "Elena Varga" is its sole container.
        # Neither entry survives into the derived reverse -- an anchored
        # fold never restores a real name onto every speaker-subject fact.
        scans = (_scan(PERSON, ("Elena Varga", "Elena")),)
        forward = build_forward_table(
            scans,
            anchor_names=frozenset({"Elena Varga"}),
            speaker_id="speaker0",
            speaker_name=None,
        )
        reverse = _derived_reverse(forward)
        assert forward["Elena Varga"] == "speaker0"
        assert forward["Elena"] == "speaker0"
        assert reverse == {}


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
            forward = build_forward_table(
                scans, anchor_names=frozenset(), speaker_id=None, speaker_name=None
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
        forward = build_forward_table(
            scans, anchor_names=frozenset(), speaker_id=None, speaker_name=None
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
        forward = build_forward_table(
            scans, anchor_names=frozenset(), speaker_id=None, speaker_name=None
        )

        assert forward["Elena"] == forward["elena"]
        assert forward["Elena"] != forward["Elena Varga"]
        assert forward["Elena"] != forward["Elena Fischer"]
        assert forward["elena"] != forward["Elena Varga"]
        assert forward["elena"] != forward["Elena Fischer"]
        assert forward["Elena Varga"] != forward["Elena Fischer"]

    def test_anchored_canonical_equality_group_folds_entirely_onto_speaker_id(self) -> None:
        # The anchor call names "Lena Marie Fischer" itself -- the anchor
        # fold writes it directly onto speaker_id, and the containment
        # pass must then fold BOTH "Lena" and its canonical-equality
        # sibling "lena" onto that same anchored placeholder, since
        # "Lena Marie Fischer" is their sole container. Neither entry
        # survives into the derived reverse.
        scans = (_scan(PERSON, ("Lena Marie Fischer", "Lena", "lena")),)
        forward = build_forward_table(
            scans,
            anchor_names=frozenset({"Lena Marie Fischer"}),
            speaker_id="speaker0",
            speaker_name=None,
        )
        reverse = _derived_reverse(forward)
        assert forward["Lena Marie Fischer"] == "speaker0"
        assert forward["Lena"] == "speaker0"
        assert forward["lena"] == "speaker0"
        assert reverse == {}
