"""Unit tests for :func:`paramem.utils.identity.canonical` and the
speaker-identity primitive :func:`~paramem.utils.identity.is_speaker_id`.

One surface form, project-wide: lower-case, blank runs → single ``_``,
``-`` and ``_`` preserved verbatim.

Verifies the GUARANTEED-identical-only contract: only Unicode canonical
form, case (incl. ligatures covered by str.casefold), diacritics, and
whitespace runs are collapsed.  ``-`` is NOT folded into ``_`` (nor either
into a space).  NFKC compatibility forms (superscript digits, full-width),
typos, and substrings are NOT folded.

Speaker-identity: ONE canonical lowercase form ``speaker{N}`` everywhere.
:func:`is_speaker_id` accepts both wire casings (``Speaker0`` from legacy
models, ``speaker0`` from the canonical store) so the ingest safety-net can
detect and coerce either form.  The coercion OUTPUT is always lowercase.
``canonical_speaker`` and ``speaker_ref_matches`` were deleted in the
lowercase-uniform refactor — speaker equality is plain ``==``.
"""

from paramem.utils.identity import (
    canonical,
    is_speaker_id,
    prose_fold,
)


class TestCaseFolding:
    def test_lowercase(self):
        assert canonical("Alex") == "alex"

    def test_already_lowercase(self):
        assert canonical("alex") == "alex"

    def test_mixed_case(self):
        assert canonical("HELLO World") == "hello_world"

    def test_sharp_s(self):
        """German ß casefolds to ss (Unicode full-case folding, not lower())."""
        assert canonical("Straße") == "strasse"

    def test_ligature_fi(self):
        """str.casefold() folds ﬁ → fi on CPython 3.11+.

        casefold() DOES fold the fi ligature (U+FB01), so assert equality,
        not inequality.
        """
        assert canonical("ﬁle") == "file"


class TestDiacriticFolding:
    def test_acute_accent(self):
        assert canonical("José") == "jose"

    def test_cedilla(self):
        assert canonical("François") == "francois"

    def test_umlaut(self):
        assert canonical("Müller") == "muller"

    def test_tilde(self):
        assert canonical("Señor") == "senor"


class TestSeparatorFolding:
    """Blanks fold to ``_``; ``-`` and ``_`` survive verbatim."""

    def test_underscore_preserved(self):
        assert canonical("works_at") == "works_at"

    def test_space_becomes_underscore(self):
        assert canonical("works at") == "works_at"

    def test_space_and_underscore_are_one_value(self):
        """The multiplicity fix: extraction's ``has_hobby`` and prose's
        ``has hobby`` are a single stored/rendered surface."""
        assert canonical("has hobby") == canonical("has_hobby") == "has_hobby"

    def test_hyphen_preserved(self):
        assert canonical("sister-in-law") == "sister-in-law"

    def test_hyphen_not_folded_into_underscore(self):
        """``-`` is a distinguishing character, not a separator variant."""
        assert canonical("sister-in-law") != canonical("sister_in_law")

    def test_hyphen_distinct_from_space(self):
        assert canonical("Anna-Maria") != canonical("Anna Maria")

    def test_whitespace_run_collapse(self):
        assert canonical("  hello   world  ") == "hello_world"

    def test_mixed_separators(self):
        assert canonical("phone_number-ext") == "phone_number-ext"


class TestEdgeCases:
    def test_empty_string(self):
        assert canonical("") == ""

    def test_whitespace_only(self):
        assert canonical("   ") == ""

    def test_none_like_empty(self):
        """Empty string returns empty string (guards None-like callers)."""
        assert canonical("") == ""


class TestIdempotence:
    def test_idempotent_basic(self):
        """f(f(x)) == f(x) for basic inputs."""
        for s in (
            "Alex",
            "José",
            "works_at",
            "has hobby",
            "Hello World",
            "sister-in-law",
            "phone_number-ext",
            "Straße",
            "ﬁle",
        ):
            once = canonical(s)
            twice = canonical(once)
            assert once == twice, f"Not idempotent for {s!r}: {once!r} → {twice!r}"

    def test_idempotent_diacritic(self):
        once = canonical("Müller")
        assert canonical(once) == once

    def test_idempotent_mixed(self):
        once = canonical("  José_Müller  ")
        assert canonical(once) == once


class TestNFCNotNFKC:
    """Canonical NFC is used, NOT NFKC.

    NFKC would collapse compatibility forms like superscript digits and
    full-width characters — canonical() must NOT fold these.
    """

    def test_superscript_not_folded(self):
        """E² must NOT collapse to E2.  NFKC would fold U+00B2 → 2; NFC keeps it."""
        assert canonical("E²") != canonical("E2")

    def test_full_width_not_folded(self):
        """Full-width A (U+FF21) must NOT collapse to ASCII A under NFC."""
        assert canonical("Ａ") != canonical("A")


class TestNegatives:
    """Inputs that must NOT be considered identical after canonicalization."""

    def test_typos_not_folded(self):
        """HR3 and HS3 differ by one character — not folded."""
        assert canonical("HR3") != canonical("HS3")

    def test_substrings_not_folded(self):
        """autonomous systems ≠ autonomous systems research."""
        assert canonical("autonomous systems") != canonical("autonomous systems research")

    def test_distinct_words_not_folded(self):
        """Berlin and Munich are different cities."""
        assert canonical("Berlin") != canonical("Munich")

    def test_numbers_not_folded(self):
        assert canonical("1") != canonical("2")


# ---------------------------------------------------------------------------
# Speaker-identity primitive tests (§0 invariant)
# ---------------------------------------------------------------------------


class TestProseFold:
    """``prose_fold`` is the case-insensitive fold for matching a needle
    against free prose — it must be exactly ``str.lower()``, deliberately
    weaker than ``canonical`` (no whitespace→``_`` fold, no diacritic fold
    beyond what ``.lower()`` does)."""

    def test_equals_lower(self):
        for s in (
            "Hello World",
            "ALREADY UPPER",
            "MiXeD CaSe",
            "",
            "  spaced out  ",
            "Straße",
            "José",
        ):
            assert prose_fold(s) == s.lower()

    def test_does_not_fold_whitespace_to_underscore(self):
        """Unlike canonical(), prose_fold must preserve spaces — folding them
        to ``_`` would break substring matching against free prose."""
        assert prose_fold("Living Room") == "living room"
        assert prose_fold("Living Room") != canonical("Living Room")

    def test_does_not_diacritic_fold(self):
        """Unlike canonical(), prose_fold must NOT strip diacritics — only
        str.lower() is applied."""
        assert prose_fold("José") == "josé"
        assert prose_fold("José") != canonical("José")

    def test_substring_match_against_prose_survives(self):
        """The exact failure mode canonical() would introduce: a multi-word
        entity name must still be found as a substring of a sentence after
        folding both sides."""
        query = "please turn on the Living Room light"
        entity = "Living Room"
        assert prose_fold(entity) in prose_fold(query)


class TestIsSpeakerId:
    """``is_speaker_id`` structural test for the ``speaker{N}`` format.

    The canonical stored form is lowercase (``"speaker0"``).  The regex also
    accepts residual cased forms (``"Speaker0"``) so the ingest safety-net can
    detect and coerce them.  Coercion output is always lowercase — this class
    only tests the structural predicate, not the coercion.
    """

    def test_lowercase_single_digit(self):
        """Canonical lowercase form used everywhere in storage and training."""
        assert is_speaker_id("speaker0") is True

    def test_lowercase_multi_digit(self):
        assert is_speaker_id("speaker12") is True

    def test_cased_single_digit_accepted(self):
        """Wire casing accepted so the safety-net can coerce it."""
        assert is_speaker_id("Speaker0") is True

    def test_cased_multi_digit_accepted(self):
        assert is_speaker_id("Speaker12") is True

    def test_plain_name_false(self):
        assert is_speaker_id("alex") is False

    def test_lowercase_word_speaker_false(self):
        """'speaker' without a digit suffix is not a speaker id."""
        assert is_speaker_id("speaker") is False

    def test_cased_word_only_false(self):
        """'Speaker' without a digit suffix is not a speaker id."""
        assert is_speaker_id("Speaker") is False

    def test_letter_suffix_false(self):
        """'SpeakerX' uses a letter suffix, not digits — must be False."""
        assert is_speaker_id("SpeakerX") is False

    def test_empty_string_false(self):
        assert is_speaker_id("") is False

    def test_prefix_only_false(self):
        """String must ONLY contain the speaker{N} pattern, no trailing chars."""
        assert is_speaker_id("speaker0Extra") is False
