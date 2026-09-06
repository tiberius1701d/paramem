"""The anonymizer gate's scorer (``scripts/dev/anonymizer_gate.py``), on a
hand-built corpus and hand-built :class:`AnonymizedContract` objects — no
model, no GPU.

Cases built by hand against the forward-table contract this gate actually
scores: a partial catch, a value inside a longer word (now an ``inert``
drop, never a forward key — production's own table build already prunes
an unsubstitutable candidate before the contract is returned), a reversed
longer value containing a scrubbed shorter one, a phone number starting
with ``+`` glued to a letter, an inert value, a junk scrub, two values
covering one gold entity jointly, an unknown word on a Person value
counted in the census under ``Person``, and a decoy scrubbed.

Every expected count is derived by hand from the gate's own matching
rules, not by running the scorer first.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the tool importable without installing it as a package — the same
# shim tests/test_calibrate_prompts_harness.py uses for scripts/dev.
_SCRIPTS_DEV = Path(__file__).resolve().parents[1] / "scripts" / "dev"
if str(_SCRIPTS_DEV) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DEV))

import anonymizer_gate  # noqa: E402 (scripts/dev is not a package)

from paramem.cloud.anonymize import AnonymizedContract  # noqa: E402

CONFIGURED = {"Person", "Phone", "Email"}


def _gold(value: str, category: str, turn, start: int, end: int) -> dict:
    return {"value": value, "category": category, "turn": turn, "start": start, "end": end}


def _entry(entry_id: str, text: str, gold: list[dict], *, decoys: list[str] = ()) -> dict:
    return {
        "id": entry_id,
        "lang": "en",
        "casing": "cased",
        "kind": "name_mention",
        "description": "hand-built scorer test entry",
        "speaker_id": None,
        "speaker_name": None,
        "history": [],
        "text": text,
        "gold": gold,
        "decoys": list(decoys),
    }


def _contract(
    forward: dict[str, str],
    *,
    dropped: list[dict] = (),
    inert: int = 0,
    raw: str = "{}",
) -> AnonymizedContract:
    return AnonymizedContract(
        status="ok",
        forward=forward,
        reverse={},
        anon_transcript="",
        declared=frozenset(),
        rekey_dropped=0,
        raw=raw,
        failure=None,
        facts=[],
        model_calls=1,
        call_tokens=(),
        scan_dropped=len(dropped),
        scan_dropped_entries=list(dropped),
        inert_dropped=inert,
    )


# --- Entry A (syn-001): partial catch; an occurrence LARGER than gold
#     still catches it; a value inside a longer word, pre-empted at its
#     only real position, is inert (never a forward key). ---

ENTRY_A = _entry(
    "syn-001",
    "Ada Lovelace paid the Billing office; Bill Gates was there.",
    [
        _gold("Ada Lovelace", "Person", -1, 0, 12),
        _gold("Bill", "Person", -1, 38, 42),
    ],
)
CONTRACT_A = _contract(
    {"Ada": "Person_1", "Bill Gates": "Person_2"},
    dropped=[{"category": "Person", "side": "table", "text": "Bill", "reason": "inert"}],
    inert=1,
)

# --- Entry B (syn-002): a reversed LONGER value containing a scrubbed
#     shorter one; the shorter one's own separate occurrence overlaps an
#     out-of-scope gold (a wrong-type scrub). ---

ENTRY_B = _entry(
    "syn-002",
    "Lena asked the Sonos Office Speaker about Sonos.",
    [
        _gold("Lena", "Person", -1, 0, 4),
        _gold("Sonos Office Speaker", "Product", -1, 15, 35),
        _gold("Sonos", "Org", -1, 42, 47),
    ],
)
CONTRACT_B = _contract(
    {"Lena": "Person_2", "Sonos": "Person_1"},
    dropped=[
        {
            "category": "Product",
            "side": "scan",
            "text": "Sonos Office Speaker",
            "reason": "reverted",
            "word": None,
        }
    ],
)

# --- Entry C (syn-003): a phone number starting with '+' glued to a
#     preceding letter; an email; an inert value; a junk scrub. ---

ENTRY_C = _entry(
    "syn-003",
    "Call x+49 151 2345 or write to nora@example.org.",
    [
        _gold("+49 151 2345", "Phone", -1, 6, 18),
        _gold("nora@example.org", "Email", -1, 31, 47),
    ],
)
CONTRACT_C = _contract(
    {"+49 151 2345": "Phone_1", "nora@example.org": "Email_1", "Call": "Person_2"},
    dropped=[{"category": "Person", "side": "table", "text": "Herr Doktor", "reason": "inert"}],
    inert=1,
)

# --- Entry D (syn-004): two scrubbed values that TOGETHER cover the gold
#     span, neither alone -> partial, not caught (a catch requires one
#     occurrence to cover the gold span in full). ---

ENTRY_D = _entry(
    "syn-004",
    "meet Ada Lovelace tomorrow.",
    [_gold("Ada Lovelace", "Person", -1, 5, 17)],
)
CONTRACT_D = _contract({"Ada": "Person_1", "Lovelace": "Person_2"})

# --- Entry E (syn-005): an unknown word on a Person value — the drop
#     record carries the real value on "text" and the model's own
#     unrecognised keyword on "word" — counted in the census under
#     "Person" (by value's gold category) and under "Persn" (by the
#     model's own word); a decoy scrubbed. ---

ENTRY_E = _entry(
    "syn-005",
    "Priya Raman said hello to Nils.",
    [_gold("Priya Raman", "Person", -1, 0, 11)],
    decoys=["Nils"],
)
CONTRACT_E = _contract(
    {"Nils": "Person_3"},
    dropped=[
        {
            "category": "",
            "side": "scan",
            "text": "Priya Raman",
            "reason": "unknown_word",
            "word": "Persn",
        }
    ],
)

CORPUS = [ENTRY_A, ENTRY_B, ENTRY_C, ENTRY_D, ENTRY_E]
CONTRACTS = {
    "syn-001": CONTRACT_A,
    "syn-002": CONTRACT_B,
    "syn-003": CONTRACT_C,
    "syn-004": CONTRACT_D,
    "syn-005": CONTRACT_E,
}


def _score() -> anonymizer_gate.Result:
    return anonymizer_gate.score_corpus(CORPUS, CONTRACTS, CONFIGURED)


def test_offsets_are_correct():
    """Guard the fixture itself: every gold span slices back to its stated value."""
    for entry in CORPUS:
        surfaces = dict(anonymizer_gate.entry_surfaces(entry))
        for g in entry["gold"]:
            assert surfaces[g["turn"]][g["start"] : g["end"]] == g["value"], entry["id"]


def test_partial_catch_first_name_out_of_full_name():
    r = _score()
    assert ("syn-001", -1, "Person", "Ada Lovelace") in r.partial_list
    assert ("syn-001", -1, "Person", "Ada Lovelace") not in r.miss_list


def test_occurrence_larger_than_gold_still_catches():
    r = _score()
    assert ("syn-001", -1, "Person", "Bill") not in r.partial_list
    assert ("syn-001", -1, "Person", "Bill") not in r.miss_list


def test_value_inside_longer_word_is_inert_not_forward():
    """'Bill' never matches inside 'Billing' (word-boundary fails) and its
    only other position is pre-empted by 'Bill Gates' — production's own
    table build prunes it before the contract is returned, so it never
    reaches ``forward`` at all and is counted only via ``inert_dropped``.
    """
    r = _score()
    assert ("syn-001", "Bill") in r.unsub_list
    assert not any(v == "Bill" for _id, v, _ph in r.junk_list)
    assert r.tot["unsubstitutable"] >= 1


def test_reversed_longer_value_contains_scrubbed_shorter_one():
    r = _score()
    assert ("syn-002", "Sonos", "Person_1", "Org") in r.wrong_type_list
    assert r.tot["out_scope_scrubbed"] == 1
    assert r.tot["out_scope_reversed"] == 1  # "Sonos Office Speaker" itself


def test_phone_starting_with_plus_glued_to_letter():
    r = _score()
    assert ("syn-003", -1, "Phone", "+49 151 2345") not in r.miss_list
    assert ("syn-003", -1, "Phone", "+49 151 2345") not in r.partial_list


def test_inert_value_counted_not_junk():
    r = _score()
    assert ("syn-003", "Herr Doktor") in r.unsub_list
    assert not any(v == "Herr Doktor" for _id, v, _ph in r.junk_list)


def test_junk_scrub():
    r = _score()
    assert any(v == "Call" for _id, v, _ph in r.junk_list)
    assert not any(v == "Call" for _id, v, _ph, _cat in r.wrong_type_list)


def test_two_values_covering_one_gold_jointly_is_partial_not_caught():
    r = _score()
    assert ("syn-004", -1, "Person", "Ada Lovelace") in r.partial_list
    assert ("syn-004", -1, "Person", "Ada Lovelace") not in r.miss_list


def test_unknown_word_on_person_value_counted_under_person():
    r = _score()
    assert ("syn-005", -1, "Person", "Priya Raman") in r.miss_list
    assert r.unknown_word_census["Person"] == 1
    assert r.unknown_word_by_word["Persn"] == 1


def test_decoy_scrubbed():
    r = _score()
    assert ("syn-005", "Nils") in r.decoys_scrubbed


def test_aggregate_counts_match_the_hand_derivation():
    r = _score()
    assert r.tot["gold_in_scope"] == 7  # 2 + 1 + 2 + 1 + 1
    assert r.tot["caught_in_scope"] == 4  # Bill, Lena, phone, email
    assert r.tot["partial_in_scope"] == 2  # Ada Lovelace x2 (syn-001, syn-004)
    assert r.tot["missed_in_scope"] == 1  # Priya Raman (syn-005)
    assert r.tot["scrubbed_correct"] == 7  # Ada,BillGates,Lena,Phone,Email,Ada,Lovelace
    assert r.tot["scrubbed_wrong_type"] == 1  # Sonos
    assert r.tot["scrubbed_junk"] == 2  # Call, Nils
    assert r.tot["unsubstitutable"] == 2  # Bill, Herr Doktor


def test_self_check_runs_clean_over_the_hand_built_corpus():
    """``score_entry`` runs its own self-check, after building its own
    occurrence list, against production's substitution walk on every
    entry — both arms (the scorer's walk agreeing with production's own
    ``applied_whole_word_keys``, and every forward key being
    substitutable) — on every entry; reaching this point without an
    ``AssertionError`` IS the assertion.
    """
    _score()


def test_self_check_raises_when_it_disagrees_with_production(monkeypatch):
    """A forward table naming a key production's own walk would never
    place must be caught, not silently under-reported.
    """
    monkeypatch.setattr(anonymizer_gate, "applied_whole_word_keys", lambda text, keys: set())
    with pytest.raises(AssertionError, match="self-check disagreement"):
        anonymizer_gate.score_entry(ENTRY_A, CONTRACT_A, CONFIGURED, anonymizer_gate.Result())


def test_self_check_raises_when_the_scorers_own_walk_diverges(monkeypatch):
    """The scorer's own occurrence walk (:func:`anonymizer_gate.
    find_occurrences`) is the OTHER half of the self-check: a walk that
    places nothing, while production's own ``applied_whole_word_keys``
    finds the forward keys in the reconstructed payload, is caught by the
    walk-agreement arm — distinct from (and checked before) the prune arm
    the test above exercises.
    """
    monkeypatch.setattr(anonymizer_gate, "find_occurrences", lambda text, values: [])
    with pytest.raises(AssertionError, match="scorer's own occurrence walk"):
        anonymizer_gate.score_entry(ENTRY_A, CONTRACT_A, CONFIGURED, anonymizer_gate.Result())


def test_gold_category_with_no_configured_membership_is_out_of_scope():
    """A category absent from *configured* is scored as out-of-scope —
    exercised above by "Product"/"Org"; this pins that the membership test
    is a plain set lookup, never a literal list in the scorer.
    """
    r = anonymizer_gate.score_corpus([ENTRY_B], {"syn-002": CONTRACT_B}, set())
    # With nothing configured, "Lena" itself is now out-of-scope too.
    assert r.tot["gold_in_scope"] == 0
    assert r.tot["gold_out_scope"] == 3


def test_unknown_word_values_reads_the_drop_records_own_text():
    """The drop record's own ``text`` field is the real value directly —
    no parse of ``contract.raw`` is involved.
    """
    values = anonymizer_gate._unknown_word_values(CONTRACT_E)
    assert values == ["Priya Raman"]


def test_unknown_word_values_empty_without_a_drop_record():
    values = anonymizer_gate._unknown_word_values(CONTRACT_A)
    assert values == []


def test_scorecard_dict_and_regression_columns():
    r = _score()
    scorecard = anonymizer_gate.scorecard_dict(r)
    # A baseline identical to the current run regresses on nothing.
    assert anonymizer_gate.regression_columns(scorecard, scorecard) == []
    # A baseline with a higher precision than achieved here names the column.
    worse_precision = {**scorecard, "precision": 100.0}
    assert "precision" in anonymizer_gate.regression_columns(scorecard, worse_precision)
    # A baseline with fewer junk scrubs than achieved here names the column.
    better_junk_baseline = {**scorecard, "junk": 0}
    assert "junk" in anonymizer_gate.regression_columns(scorecard, better_junk_baseline)
