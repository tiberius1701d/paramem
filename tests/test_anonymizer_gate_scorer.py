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
from paramem.utils.tokens import (  # noqa: E402
    ANONYMIZE_SCAN_PROMPT_SKELETON_TOKENS,
    ANONYMIZE_SCAN_REPLY_RATIO,
)

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
    call_tokens: tuple[dict, ...] = (),
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
        call_tokens=call_tokens,
        scan_dropped=len(dropped),
        scan_dropped_entries=list(dropped),
        inert_dropped=inert,
    )


def _scan_call(prompt_tokens: int, output_tokens: int) -> dict:
    """One synthetic ``anonymize.scan`` ``call_tokens`` record."""
    return {
        "label": "anonymize.scan",
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
    }


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

# --- Entry B (syn-002): a reverted LONGER value containing a scrubbed
#     shorter one; the shorter one substitutes inside that reverted
#     surface and at its own standalone position, and each of the two
#     lands on an out-of-scope gold (a wrong-type scrub). ---

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
    """The reverted surface never reaches the forward table, so nothing
    shields the kept shorter value inside it: "Sonos" substitutes both
    inside "Sonos Office Speaker" and at its own standalone position.
    """
    r = _score()
    # The first of the two positions sits inside the Product gold.
    assert ("syn-002", "Sonos", "Person_1", "Product") in r.wrong_type_list
    assert r.tot["out_scope_scrubbed"] == 2  # Product in part, Org in full
    assert r.tot["out_scope_reversed"] == 0


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


# ---------------------------------------------------------------------------
# `failed` — the scorecard's own column for the count of skipped
# (non-``"ok"``) entries; the skipped entries' own gold, tallied so the
# printed detail can state how much gold sits outside the recall
# percentages.
# ---------------------------------------------------------------------------


def test_score_corpus_tallies_a_missing_contracts_gold_into_names_and_contact():
    """An entry with no contract at all (never run, or its artifact absent)
    is skipped with status ``"missing"`` — never silently dropped — and its
    own gold is tallied into ``skipped_gold_names``/``skipped_gold_contact``,
    never into the recall counters ``score_entry`` would have produced."""
    text = "Nora called from +49 152 0 445 3311."
    phone = "+49 152 0 445 3311"
    p_start = text.index(phone)
    entry = _entry(
        "syn-006",
        text,
        [
            _gold("Nora", "Person", -1, 0, 4),
            _gold(phone, "Phone", -1, p_start, p_start + len(phone)),
        ],
    )

    result = anonymizer_gate.score_corpus([entry], {}, CONFIGURED)

    assert result.skipped == [("syn-006", "missing")]
    assert result.skipped_gold_names == 1
    assert result.skipped_gold_contact == 1
    assert result.tot["entries"] == 0
    assert result.tot["gold_in_scope"] == 0


def test_score_corpus_tallies_a_failed_status_contracts_gold_too():
    """A contract that completed but did not reach ``"ok"`` (``"failed"``)
    is skipped identically to a missing one — its own status is recorded,
    and its gold still counts toward the skipped-gold tallies."""
    entry = _entry("syn-007", "Nora said hi.", [_gold("Nora", "Person", -1, 0, 4)])
    failed_contract = AnonymizedContract(
        status="failed",
        forward={},
        reverse={},
        anon_transcript="",
        declared=frozenset(),
        rekey_dropped=0,
        raw="",
        failure="scan_failed",
        facts=[],
        model_calls=1,
        call_tokens=(),
        scan_dropped=0,
        scan_dropped_entries=[],
        inert_dropped=0,
    )

    result = anonymizer_gate.score_corpus([entry], {"syn-007": failed_contract}, CONFIGURED)

    assert result.skipped == [("syn-007", "failed")]
    assert result.skipped_gold_names == 1
    assert result.skipped_gold_contact == 0


def test_scorecard_dict_carries_failed_as_the_count_of_skipped_entries():
    entry = _entry("syn-008", "Nora said hi.", [_gold("Nora", "Person", -1, 0, 4)])
    result = anonymizer_gate.score_corpus([ENTRY_A, entry], {"syn-001": CONTRACT_A}, CONFIGURED)

    scorecard = anonymizer_gate.scorecard_dict(result)

    assert scorecard["failed"] == 1
    assert "failed" in anonymizer_gate._LOWER_IS_BETTER


def test_regression_columns_names_failed_when_current_exceeds_the_baseline():
    scorecard = anonymizer_gate.scorecard_dict(_score())
    current = {**scorecard, "failed": 3}
    baseline = {**scorecard, "failed": 1}

    assert "failed" in anonymizer_gate.regression_columns(current, baseline)


def test_regression_columns_is_silent_on_failed_when_the_baseline_lacks_the_key():
    """A baseline file carrying no ``failed`` key compares as ``None`` on
    that key — never a false regression."""
    scorecard = anonymizer_gate.scorecard_dict(_score())
    current = {**scorecard, "failed": 3}
    baseline = {k: v for k, v in scorecard.items() if k != "failed"}

    assert "failed" not in anonymizer_gate.regression_columns(current, baseline)


def test_print_detail_states_the_skipped_entry_gold_outside_the_percentages(capsys):
    entry = _entry("syn-006", "Nora called.", [_gold("Nora", "Person", -1, 0, 4)])
    result = anonymizer_gate.score_corpus([entry], {}, CONFIGURED)

    anonymizer_gate.print_detail(result)

    out = capsys.readouterr().out
    assert "skipped-entry gold outside the percentages: names 1, contact 0" in out


# ---------------------------------------------------------------------------
# `invented` — the scorecard's own column for the count of `unknown_word`
# drop records over the run.
# ---------------------------------------------------------------------------


def test_scorecard_dict_carries_invented_as_the_sum_of_unknown_word_by_word():
    """ENTRY_E's one ``unknown_word`` drop record is the run's only one."""
    r = _score()
    scorecard = anonymizer_gate.scorecard_dict(r)

    assert scorecard["invented"] == sum(r.unknown_word_by_word.values())
    assert scorecard["invented"] == 1
    assert "invented" in anonymizer_gate._LOWER_IS_BETTER


def test_scorecard_dict_invented_sums_multiple_drop_records():
    """Two ``unknown_word`` drops on one entry — same word twice, a
    different word once — all count, since ``invented`` counts records,
    not distinct words."""
    entry = _entry("syn-invented", "Ana met Timo and Vera.", [])
    unknown = {"category": "", "side": "scan", "reason": "unknown_word"}
    contract = _contract(
        {},
        dropped=[
            {**unknown, "text": "Ana", "word": "Nom"},
            {**unknown, "text": "Timo", "word": "Nom"},
            {**unknown, "text": "Vera", "word": "Tag"},
        ],
    )

    result = anonymizer_gate.Result()
    anonymizer_gate.score_entry(entry, contract, CONFIGURED, result)
    scorecard = anonymizer_gate.scorecard_dict(result)

    assert scorecard["invented"] == 3


def test_regression_columns_names_invented_when_current_exceeds_the_baseline():
    scorecard = anonymizer_gate.scorecard_dict(_score())
    current = {**scorecard, "invented": 5}
    baseline = {**scorecard, "invented": 1}

    assert "invented" in anonymizer_gate.regression_columns(current, baseline)


def test_regression_columns_is_silent_on_invented_when_the_baseline_lacks_the_key():
    """A baseline file lacking the ``invented`` key compares as ``None``
    on that key — never a false regression, and never a crash."""
    scorecard = anonymizer_gate.scorecard_dict(_score())
    current = {**scorecard, "invented": 5}
    baseline = {k: v for k, v in scorecard.items() if k != "invented"}

    assert "invented" not in anonymizer_gate.regression_columns(current, baseline)


# ---------------------------------------------------------------------------
# The scan reply ratio readout — two distinct trackings updated inside
# score_entry from each contract's own `anonymize.scan` call_tokens
# record: the run totals (Result.scan_reply_run_output_tokens /
# Result.scan_reply_run_payload_tokens, whose ratio is the re-pin source
# for ANONYMIZE_SCAN_REPLY_RATIO) and the per-entry maximum
# (Result.largest_scan_reply_ratio / Result.largest_scan_reply_ratio_entry,
# a reading of the ANONYMIZE_SCAN_MAX_OUTPUT_TOKENS plateau).
# ---------------------------------------------------------------------------


def test_largest_scan_reply_ratio_tracks_the_maximum_and_its_entry():
    entry_low = _entry("ratio-low", "Nora said hi.", [])
    entry_high = _entry("ratio-high", "Omar said hi.", [])
    contract_low = _contract(
        {}, call_tokens=(_scan_call(ANONYMIZE_SCAN_PROMPT_SKELETON_TOKENS + 100, 50),)
    )
    contract_high = _contract(
        {}, call_tokens=(_scan_call(ANONYMIZE_SCAN_PROMPT_SKELETON_TOKENS + 100, 250),)
    )

    result = anonymizer_gate.Result()
    anonymizer_gate.score_entry(entry_low, contract_low, CONFIGURED, result)
    anonymizer_gate.score_entry(entry_high, contract_high, CONFIGURED, result)

    assert result.largest_scan_reply_ratio == pytest.approx(2.5)
    assert result.largest_scan_reply_ratio_entry == "ratio-high"


def test_scan_reply_run_totals_sum_across_entries():
    """The run totals sum every scored call's output and payload tokens —
    distinct from the per-entry maximum tracked alongside them."""
    entry_low = _entry("ratio-low", "Nora said hi.", [])
    entry_high = _entry("ratio-high", "Omar said hi.", [])
    contract_low = _contract(
        {}, call_tokens=(_scan_call(ANONYMIZE_SCAN_PROMPT_SKELETON_TOKENS + 100, 50),)
    )
    contract_high = _contract(
        {}, call_tokens=(_scan_call(ANONYMIZE_SCAN_PROMPT_SKELETON_TOKENS + 100, 250),)
    )

    result = anonymizer_gate.Result()
    anonymizer_gate.score_entry(entry_low, contract_low, CONFIGURED, result)
    anonymizer_gate.score_entry(entry_high, contract_high, CONFIGURED, result)

    assert result.scan_reply_run_output_tokens == 300
    assert result.scan_reply_run_payload_tokens == 200


@pytest.mark.parametrize("payload_delta", [0, -5])
def test_largest_scan_reply_ratio_skips_a_call_with_non_positive_payload_tokens(payload_delta):
    """A call whose payload tokens (prompt tokens less the pinned skeleton)
    are zero or below takes no part in the maximum, nor in the run totals."""
    entry = _entry("ratio-empty", "hi.", [])
    contract = _contract(
        {}, call_tokens=(_scan_call(ANONYMIZE_SCAN_PROMPT_SKELETON_TOKENS + payload_delta, 5),)
    )

    result = anonymizer_gate.Result()
    anonymizer_gate.score_entry(entry, contract, CONFIGURED, result)

    assert result.largest_scan_reply_ratio is None
    assert result.largest_scan_reply_ratio_entry is None
    assert result.scan_reply_run_output_tokens == 0
    assert result.scan_reply_run_payload_tokens == 0


def test_largest_scan_reply_ratio_ignores_non_scan_call_labels():
    entry = _entry("ratio-anchor-only", "hi.", [])
    contract = _contract(
        {},
        call_tokens=(
            {
                "label": "anonymize.anchor",
                "prompt_tokens": ANONYMIZE_SCAN_PROMPT_SKELETON_TOKENS + 500,
                "output_tokens": 400,
            },
        ),
    )

    result = anonymizer_gate.Result()
    anonymizer_gate.score_entry(entry, contract, CONFIGURED, result)

    assert result.largest_scan_reply_ratio is None
    assert result.scan_reply_run_payload_tokens == 0


def test_print_detail_prints_the_scan_reply_ratio_line(capsys):
    result = anonymizer_gate.Result(
        largest_scan_reply_ratio=3.10,
        largest_scan_reply_ratio_entry="de-dev-057",
        scan_reply_run_output_tokens=142,
        scan_reply_run_payload_tokens=100,
    )

    anonymizer_gate.print_detail(result)

    out = capsys.readouterr().out
    assert (
        f"scan reply ratio: run 1.42 "
        f"(pinned ANONYMIZE_SCAN_REPLY_RATIO={ANONYMIZE_SCAN_REPLY_RATIO}); "
        "largest single entry 3.10 (de-dev-057)" in out
    )


def test_print_detail_prints_na_when_no_scan_reply_ratio_observed(capsys):
    result = anonymizer_gate.Result()

    anonymizer_gate.print_detail(result)

    out = capsys.readouterr().out
    assert "scan reply ratio: n/a" in out


# ---------------------------------------------------------------------------
# Three populations, three walks. Production substitutes the forward table
# by itself (``paramem.cloud.placeholders._substitute_whole_words``), so
# the scorer's view of what substitutes is a walk over the forward keys
# alone; the reverted surfaces and the unknown-word values — which
# production substitutes nowhere — are located by walks of their own,
# since the gate scores their positions against gold spans.
# ---------------------------------------------------------------------------


def test_kept_value_inside_a_reverted_surface_is_scrubbed_where_it_sits():
    """A kept value whose only occurrence lies inside a longer reverted
    surface still substitutes there: the reverted surface never reaches
    the forward table, so nothing shields the name inside it.
    """
    text = "My boss Mara Feldmann wants the report by Monday."
    name_start = text.index("Mara Feldmann")
    entry = _entry(
        "syn-009",
        text,
        [_gold("Mara Feldmann", "Person", -1, name_start, name_start + len("Mara Feldmann"))],
    )
    contract = _contract(
        {"Mara Feldmann": "Person_1"},
        dropped=[
            {
                "category": "Org",
                "side": "scan",
                "text": "My boss Mara Feldmann",
                "reason": "reverted",
                "word": None,
            }
        ],
    )

    result = anonymizer_gate.score_corpus([entry], {"syn-009": contract}, CONFIGURED)

    assert result.tot["caught_in_scope"] == 1
    assert result.tot["caught_names"] == 1
    assert result.miss_list == []
    assert result.partial_list == []
    assert result.tot["scrubbed_correct"] == 1
    assert result.junk_list == []
    assert result.wrong_type_list == []


def test_reverted_surface_keeps_its_own_position_under_a_longer_forward_key():
    """A reverted surface is scored by position against out-of-scope gold,
    so it is located by its own walk: a longer forward key covering one of
    its occurrences leaves its other occurrences intact.
    """
    text = "The landlord asked; write to landlord@example.de."
    email = "landlord@example.de"
    email_start = text.index(email)
    role_start = text.index("landlord")  # the standalone one, before the address
    entry = _entry(
        "syn-010",
        text,
        [
            _gold(email, "Email", -1, email_start, email_start + len(email)),
            _gold("landlord", "Profession", -1, role_start, role_start + len("landlord")),
        ],
    )
    contract = _contract(
        {email: "Email_1"},
        dropped=[
            {
                "category": "Profession",
                "side": "scan",
                "text": "landlord",
                "reason": "reverted",
                "word": None,
            }
        ],
    )

    result = anonymizer_gate.score_corpus([entry], {"syn-010": contract}, CONFIGURED)

    assert result.tot["caught_in_scope"] == 1
    assert result.tot["out_scope_reversed"] == 1
    assert result.tot["out_scope_scrubbed"] == 0
    assert result.tot["out_scope_untagged"] == 0
