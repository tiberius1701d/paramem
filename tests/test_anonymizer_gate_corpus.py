"""The unified anonymizer-gate corpus checker
(``tests/fixtures/anonymizer_gate.json``).

Mechanically enforces the fictional-data guarantee and the corpus's own
schema over the fixture: id uniqueness, gold-offset correctness (delegated
to ``scripts/dev/anonymizer_gate.py::validate_corpus``, the one
implementation), non-overlapping gold spans per turn, no speaker token in
gold, the forbidden-real-name check, the honorific rule over person spans,
the word-count bounds by kind (the dense, planted, facts and transcript
kinds exempt — the "transcript" kind holds realistic multi-turn reply
turns, whose current-turn text is a natural short reply such as a single
recipient name, not a standalone utterance designed to meet a word-count
floor), the fictional phone and email patterns, and decoys occurring in
the text they are attached to.

The forbidden-real-name check never carries a real name in this file: the
real names it must reject are salted (:data:`_NAME_SALT`) and SHA-256
hashed once, offline, into :data:`_FORBIDDEN_REAL_NAME_DIGESTS`; the check
re-hashes every lowercase word of every entry's own model-facing surfaces
and fails on a digest match, so the rule stays mechanical and enforced
without the repo ever holding the plaintext list.
"""

from __future__ import annotations

import hashlib
import re
import sys
from collections import Counter
from pathlib import Path

import pytest

# Make the tool importable without installing it as a package — the same
# shim tests/test_calibrate_prompts_harness.py uses for scripts/dev.
_SCRIPTS_DEV = Path(__file__).resolve().parents[1] / "scripts" / "dev"
if str(_SCRIPTS_DEV) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DEV))

import anonymizer_gate  # noqa: E402 (scripts/dev is not a package)

_FIXTURE_PATH = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "anonymizer_gate.json"
_ANONYMIZATION_PROMPT_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "prompts" / "anonymization.txt"
)

_LANGS = {"en", "de", "fr", "es"}
_CASINGS = {"cased", "lower"}
# Word-count-bound-exempt kinds: dense contact lists, planted-value
# entries, facts entries (no linear "text" field to count words over at
# all), and transcript entries (see the module docstring).
_WORD_BOUND_EXEMPT_KINDS = {"dense", "planted", "facts", "transcript"}
_WORD_MIN, _WORD_MAX = 4, 40

# Fixed salt for the forbidden-real-name digests below — never the real
# names themselves, so the repo carries no name. A digest is
# ``sha256(_NAME_SALT + name.lower())``; the check applies the identical
# formula to every lowercase word it finds and rejects a match. Rotating
# this salt invalidates every digest below and requires regenerating them
# from the real name list, which is never committed to the repository.
_NAME_SALT = "paramem-anonymizer-gate-forbidden-name-salt-v1"

# SHA-256 digests of ``_NAME_SALT + name.lower()`` for every real name the
# corpus must never contain, generated once offline (see this module's
# docstring). Sorted, so the list itself carries no naming order to infer
# from.
_FORBIDDEN_REAL_NAME_DIGESTS = (
    "1125e268374f19fcfc074da844de3d6e485cc6ce173ce6dcafb4d80c666159cf",
    "113a7c28050a41596bb12126e55469cfdd91f40652a286ea376a8d67e7ae36db",
    "29a6791745b0586b188362f2acac9b6a119596e5747d7e1d1ffb8b1fcc2155c3",
    "46ed73cbe9c894149bab94334b05b83880e17c1a7713cf12981416cca4779123",
    "489b9cc02af77885823996b8ffa750bd734f8541531532ec7f2ddcebeffafd9c",
    "4ef72a498fd00b10320e9bf12b26bb9125f9728fdb45c2d21c17ef1af300e728",
    "757ac6dddb0352ef11e31f3c391bcd2b7370855496acdf947786253ffbc7f6ec",
    "8463963c3c58294f2676b592b8f182cc4a2ee61a5ad3698550ea7d79c0c34d6e",
    "876ad9164a6e7a2200557b4a3f0c05c6c25b8a6642886dd4894ca68421acd6f8",
    "94201901f0984c76914baa8ce17193416024bd8c1527ec9454f2db16cbc26fb7",
    "9679cbce9e85b5fa6de18b04d096af428331baf03aa8e1bd8a628ac2b48c4634",
    "a613b3325954ddf0761de07229160f51754d03d779d48793b2b84601adb509d1",
    "b5af3e2fe092bae2bd4960f3add6be5a051adb2435df2524079177777766c7c0",
    "bb2946da818e9126b86df0b48c31c86c5296268a9d0ffaa7bcddcf1b227332ca",
    "c63df641792d58320cc833d0a49f23e6b7df51c3844be6849a134a5ca07fb9a9",
    "c834c687f0da8a7c0397e86ccc2e9d19afbbd070d71141b7283d021706ed69a5",
    "d9c00c564c3dbdc110a832da6302a74fa27964b6f8e8abb25d5ad1bfbc3665a2",
    "e13f5ab5321e69bafc00706873734426a39b5a73fcd1ea578590d0abfc215c5d",
    "ea9005d547e72a9c5d4c08e7c1b5ca4efd24d0a4bbe7cf6dc27b798df49ebc05",
    "eb74dec5157be260bbb77538305a866a0a9d070a954982de2e5c50b8cf8e8ac8",
    "fa5044dd625234d1496cc258e06480036efaa57f3553e8833d43a9485b8cb074",
)

# Whole-word tokenizer for the digest check: Unicode word characters only
# (no underscore, matching ``\W`` semantics), so punctuation and
# whitespace never glue two words into one token and a name is hashed at
# the same granularity it would appear in a corpus entry.
_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


def _word_digest(word: str) -> str:
    """The forbidden-name check's own hash: ``sha256(_NAME_SALT +
    word.lower())``, hex-encoded — applied identically to a candidate
    household name (when the digest tuple above was generated) and to
    every word found in the corpus (see :func:`_entry_words`).
    """
    return hashlib.sha256((_NAME_SALT + word.lower()).encode("utf-8")).hexdigest()


def _entry_words(entry: dict) -> list[str]:
    """Every lowercase word from *entry*'s own model-facing surfaces:
    ``text``, each ``history`` turn's ``text``, each fact's ``subject``/
    ``predicate``/``object``, and ``speaker_name`` when set — never
    ``description``, which is authoring metadata a model never reads.
    """
    surfaces: list[str] = []
    if "text" in entry:
        surfaces.append(entry["text"])
    for turn in entry.get("history", ()):
        surfaces.append(turn.get("text", ""))
    for fact in entry.get("facts", ()):
        surfaces.append(str(fact.get("subject", "")))
        surfaces.append(str(fact.get("predicate", "")))
        surfaces.append(str(fact.get("object", "")))
    speaker_name = entry.get("speaker_name")
    if speaker_name:
        surfaces.append(speaker_name)
    words: list[str] = []
    for surface in surfaces:
        words.extend(_WORD_RE.findall(surface.lower()))
    return words


# Every fictional phone shape actually used in the corpus, matched in full
# rather than as a prefix, so a real-looking number cannot enter behind a
# familiar-looking country code. Two families carry an invented marker
# that places the whole number outside any real numbering plan (a
# reserved German mobile block, "152 0", and its Spanish counterpart,
# "6 00"); the rest borrow an ordinary-looking prefix but carry a drama
# infix — the "555" convention used for a number that must not ring a
# real line — inside the subscriber part, or, for the French landline
# family, the same infix folded into its own digit grouping ("55 50").
# The French mobile family instead repeats a fixed marker sequence,
# "12 34 56", ahead of two free digits.
_FICTIONAL_PHONE_PATTERNS = (
    re.compile(r"^\+49 152 0 \d{3} \d{4}$"),  # German invented marker, international
    re.compile(r"^0152 0 \d{3} \d{4}$"),  # German invented marker, local
    re.compile(r"^\+34 6 00 \d{3} \d{3}$"),  # Spanish invented marker
    re.compile(
        r"^\+49 (?:151|160|162|170|171|173|175|176|178|179) 5550\d{3}$"
    ),  # German drama infix, international
    re.compile(
        r"^0(?:151|160|162|170|171|173|175|176|178|179) 5550\d{3}$"
    ),  # German drama infix, local
    re.compile(r"^\+353 91 555 \d{4}$"),  # Irish drama infix
    re.compile(r"^0800 555 \d{4}$"),  # shared support line, drama infix
    re.compile(r"^\+33 1 55 50 \d{2} \d{2}$"),  # French landline, drama infix
    re.compile(r"^\+33 6 12 34 56 \d{2}$"),  # French mobile marker, international
    re.compile(r"^06 12 34 56 \d{2}$"),  # French mobile marker, local
)


def _is_fictional_phone(value: str) -> bool:
    """Whether *value* matches one of the corpus's invented phone shapes
    in full — never a prefix match, so a real-looking number cannot enter
    behind a familiar-looking start.
    """
    return any(pattern.match(value) for pattern in _FICTIONAL_PHONE_PATTERNS)


_FICTIONAL_EMAIL_DOMAINS = (
    "@example.org",
    "@example.de",
    "@example.net",
    "@example.com",
    ".example.de",
)

# Free-text scan patterns for the phone/email checks below: broad enough
# to catch a phone- or email-shaped substring anywhere in an entry's own
# surfaces, whether or not it is gold-tagged (a decoy or incidental prose
# value could otherwise carry a non-fictional number or domain
# undetected). Not a production detector — a corpus-authoring guard only.
_PHONE_LIKE_RE = re.compile(r"[+0]\d[\d ]{5,}\d")
_EMAIL_LIKE_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

# Honorifics that must not appear as the first token of a Person gold span,
# compared case-insensitively with a trailing period stripped so "Mr" and
# "Mr." both match.
_HONORIFICS = {
    "mr",
    "mrs",
    "ms",
    "dr",
    "herr",
    "frau",
    "m",
    "mme",
    "mlle",
    "sr",
    "sra",
    "don",
    "doña",
}


def _first_token_is_honorific(value: str) -> bool:
    tokens = value.split()
    if not tokens:
        return False
    return tokens[0].rstrip(".").lower() in _HONORIFICS


@pytest.fixture(scope="module")
def entries() -> list[dict]:
    return anonymizer_gate.load_corpus(_FIXTURE_PATH)


def test_corpus_file_parses_and_is_non_empty(entries):
    assert len(entries) > 0


def test_validate_corpus_passes(entries):
    """Schema shape and gold-offset correctness — the tool's own mechanical
    check, run here as a test so a corpus edit is caught by CI.
    """
    anonymizer_gate.validate_corpus(entries)


def test_ids_are_unique(entries):
    ids = [e["id"] for e in entries]
    dupes = [i for i in set(ids) if ids.count(i) > 1]
    assert dupes == []


def test_lang_and_casing_are_recognised(entries):
    for e in entries:
        assert e["lang"] in _LANGS, e["id"]
        assert e["casing"] in _CASINGS, e["id"]


def test_lower_casing_entries_are_actually_lowercase(entries):
    for e in entries:
        if e["casing"] != "lower":
            continue
        text = e.get("text")
        if text is None:
            continue
        assert text == text.lower(), f"{e['id']}: marked lower but has uppercase"


def test_no_forbidden_real_name(entries):
    """No real household name appears in any model-facing surface
    (``text``/``history``/``facts``/``speaker_name``) — checked whole-word
    by salted digest (see :func:`_entry_words`, :func:`_word_digest`) so
    the check stays mechanical without the repo carrying a plaintext name
    list, and tightened to a whole word so ordinary prose containing a
    name as a substring (e.g. "the number still the same?") never
    false-positives.
    """
    errors = []
    for e in entries:
        for word in _entry_words(e):
            if _word_digest(word) in _FORBIDDEN_REAL_NAME_DIGESTS:
                errors.append(f"{e['id']}: forbidden word found (digest match)")
    assert errors == [], "\n".join(errors)


def test_anonymization_prompt_carries_no_forbidden_name():
    """``configs/prompts/anonymization.txt`` is model-facing text too: its
    worked examples are tokenised the same way an entry's surfaces are
    (:data:`_WORD_RE`, the tokenizer :func:`_entry_words` applies), and no
    word's salted digest may appear in :data:`_FORBIDDEN_REAL_NAME_DIGESTS`.
    """
    text = _ANONYMIZATION_PROMPT_PATH.read_text(encoding="utf-8")
    words = _WORD_RE.findall(text.lower())
    matches = [w for w in words if _word_digest(w) in _FORBIDDEN_REAL_NAME_DIGESTS]
    assert matches == [], f"forbidden word found in {_ANONYMIZATION_PROMPT_PATH.name}: {matches}"


def test_gold_spans_do_not_overlap_within_a_turn(entries):
    errors = []
    for e in entries:
        by_turn: dict[object, list[dict]] = {}
        for g in e["gold"]:
            by_turn.setdefault(g["turn"], []).append(g)
        for turn, golds in by_turn.items():
            golds_sorted = sorted(golds, key=lambda g: g["start"])
            prev_end = -1
            for g in golds_sorted:
                if g["start"] < prev_end:
                    errors.append(f"{e['id']} turn {turn!r}: overlapping gold at {g['start']}")
                prev_end = max(prev_end, g["end"])
    assert errors == [], "\n".join(errors)


def test_no_speaker_token_in_gold(entries):
    errors = [
        f"{e['id']}: speaker token in gold {g['value']!r}"
        for e in entries
        for g in e["gold"]
        if g["value"].startswith("speaker")
    ]
    assert errors == [], "\n".join(errors)


def test_no_honorific_inside_person_span(entries):
    errors = [
        f"{e['id']}: honorific inside person span {g['value']!r}"
        for e in entries
        for g in e["gold"]
        if g["category"] == "Person" and _first_token_is_honorific(g["value"])
    ]
    assert errors == [], "\n".join(errors)


def test_fictional_phone_range(entries):
    """Every gold ``Phone`` value, and every phone-shaped substring found
    anywhere in an entry's own text/history by :data:`_PHONE_LIKE_RE`
    (gold or not — a decoy or incidental prose value is checked exactly
    like a gold one), matches one of the corpus's invented phone shapes in
    full, never merely a prefix of one.
    """
    errors = []
    for e in entries:
        for g in e["gold"]:
            if g["category"] == "Phone" and not _is_fictional_phone(g["value"]):
                errors.append(f"{e['id']}: gold phone outside fictional range {g['value']!r}")
        for _turn, text in anonymizer_gate.entry_surfaces(e):
            for match in _PHONE_LIKE_RE.finditer(text):
                if not _is_fictional_phone(match.group()):
                    errors.append(
                        f"{e['id']}: phone-shaped text outside fictional range {match.group()!r}"
                    )
    assert errors == [], "\n".join(errors)


def test_fictional_email_domains(entries):
    """Every gold ``Email`` value, and every email-shaped substring found
    anywhere in an entry's own text/history by :data:`_EMAIL_LIKE_RE`
    (gold or not), ends with one of the fictional domains.
    """
    errors = []
    for e in entries:
        for g in e["gold"]:
            if g["category"] == "Email" and not g["value"].endswith(_FICTIONAL_EMAIL_DOMAINS):
                errors.append(f"{e['id']}: gold email domain {g['value']!r}")
        for _turn, text in anonymizer_gate.entry_surfaces(e):
            for match in _EMAIL_LIKE_RE.finditer(text):
                if not match.group().endswith(_FICTIONAL_EMAIL_DOMAINS):
                    errors.append(
                        f"{e['id']}: email-shaped text with a non-fictional "
                        f"domain {match.group()!r}"
                    )
    assert errors == [], "\n".join(errors)


def test_word_count_bounds_by_kind(entries):
    errors = []
    for e in entries:
        if e["kind"] in _WORD_BOUND_EXEMPT_KINDS:
            continue
        text = e.get("text", "")
        words = [w for w in text.split() if any(ch.isalnum() for ch in w)]
        if not (_WORD_MIN <= len(words) <= _WORD_MAX):
            errors.append(f"{e['id']}: {len(words)} words")
    assert errors == [], "\n".join(errors)


def test_decoys_occur_in_the_entry_text(entries):
    """Every decoy is findable somewhere in the entry's own surfaces —
    otherwise it exercises nothing.
    """
    errors = []
    for e in entries:
        if not e["decoys"]:
            continue
        surfaces = anonymizer_gate.entry_surfaces(e)
        for decoy in e["decoys"]:
            found = any(anonymizer_gate.find_occurrences(text, [decoy]) for _turn, text in surfaces)
            if not found:
                errors.append(f"{e['id']}: decoy {decoy!r} occurs nowhere")
    assert errors == [], "\n".join(errors)


def test_gold_categories_map_to_a_schema_prefix_row(entries):
    """Every gold category is one of ``configs/schema.yaml``'s
    ``anonymizer.prefixes`` row names — the corpus never invents its own
    category vocabulary.
    """
    from paramem.config.taxonomy import prefix_descriptions

    known = {prefix for prefix, _description in prefix_descriptions()}
    bad = {g["category"] for e in entries for g in e["gold"]} - known
    assert bad == set(), f"gold categories with no schema.yaml row: {sorted(bad)}"


def test_census_by_kind_and_lang(entries):
    """Composition sanity: every entry lands in a recognised kind and
    language, and the corpus is not accidentally empty in any dimension.
    """
    kinds = Counter(e["kind"] for e in entries)
    langs = Counter(e["lang"] for e in entries)
    assert sum(kinds.values()) == len(entries)
    assert sum(langs.values()) == len(entries)
    assert set(langs) <= _LANGS
