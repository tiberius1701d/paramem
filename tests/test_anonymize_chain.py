"""``anonymize()`` — the opt-out door's constructor parity, the empty-
transcript precondition, the anchor candidate domain, and the structural
invariant that no caller declares a per-caller speaker-name linking policy.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from paramem.cloud import anonymize as anonymize_module
from paramem.cloud import anonymize_steps as anonymize_steps_module
from paramem.cloud.anonymize import (
    AnonymizerPrompts,
    _anchor_candidates,
    anonymize,
    failed_contract,
    opted_out_contract,
)
from paramem.cloud.anonymize_steps import ScanResult
from paramem.config.taxonomy import ScrubCategory
from paramem.graph.flows import anonymize_turn
from tests._guard_utils import tracked_python_files
from tests.anonymizer_doubles import ScriptedGenerate, ScriptedTokenizer, basic_prompts

_PROMPTS = AnonymizerPrompts(scan_system="s", scan="s", anchor_system="a", anchor="a")


def _run_chain(
    monkeypatch,
    replies: list[str],
    *,
    categories: tuple[ScrubCategory, ...],
    facts: list[dict] | None = None,
    transcript: str = "",
    history: tuple[str, ...] = (),
    token_envelope: int = 8192,
    speaker_id: str | None = None,
    speaker_name: str | None = None,
) -> tuple:
    """Drive the full ``anonymize()`` chain against a scripted SCAN reply
    per slice — no model, no GPU.

    ``effective_token_envelope`` is patched to a strict passthrough so the
    scripted budget-precondition tests are deterministic regardless of live
    free VRAM (the same behaviour the real function already guarantees when
    no CUDA device is available — see its own docstring).

    Returns ``(contract, fake_generate)`` — *fake_generate* records every
    rendered prompt it was called with (``fake_generate.calls``).
    """
    monkeypatch.setattr(anonymize_module, "effective_token_envelope", lambda te: (te, None))
    fake_generate = ScriptedGenerate(list(replies))
    monkeypatch.setattr(anonymize_steps_module, "generate_answer", fake_generate)
    contract = anonymize(
        list(facts) if facts is not None else [],
        object(),
        ScriptedTokenizer(),
        transcript=transcript,
        history=history,
        categories=categories,
        speaker_id=speaker_id,
        speaker_name=speaker_name,
        token_envelope=token_envelope,
        prompts=basic_prompts(),
    )
    return contract, fake_generate


def _person_scan(values: tuple[str, ...]) -> tuple[ScanResult, ...]:
    return (ScanResult(category=ScrubCategory(prefix="Person"), values=values),)


# ---------------------------------------------------------------------------
# Contract: constructors
# ---------------------------------------------------------------------------


class TestContractConstructors:
    def test_opted_out_contract_matches_the_chains_own_opt_out_shape(self) -> None:
        via_constructor = opted_out_contract("hello", facts=[{"a": 1}])
        via_chain = anonymize(
            [{"a": 1}],
            None,
            None,
            transcript="hello",
            categories=[],
            prompts=basic_prompts(),
        )
        assert via_constructor == via_chain

    def test_opted_out_issues_zero_model_calls(self) -> None:
        contract = anonymize(
            [], None, None, transcript="hello", categories=[], prompts=basic_prompts()
        )
        assert contract.model_calls == 0


class TestFailedContractRequiresFailure:
    def test_a_failed_contract_without_a_cause_cannot_be_built(self) -> None:
        with pytest.raises(TypeError):
            failed_contract()  # type: ignore[call-arg]


class TestAnonymizeTurnEmptyTranscriptPrecondition:
    def test_empty_transcript_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            anonymize_turn("", None, None, categories=(), token_envelope=8192)

    def test_whitespace_only_transcript_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            anonymize_turn("   \n\t", None, None, categories=(), token_envelope=8192)


# ---------------------------------------------------------------------------
# The anchor candidate domain.
# ---------------------------------------------------------------------------


class TestAnchorCandidatesDomain:
    def test_a_value_only_in_history_or_facts_is_never_offered(self) -> None:
        # tag_text = history + transcript + facts; anchor_range covers only
        # the transcript slice — "Riley" lives in the fact line, outside it.
        tag_text = "Alex said hi.\nRiley works at Acme"
        scans = _person_scan(("Alex", "Riley"))
        candidates = _anchor_candidates(scans, tag_text, (0, 14), "Person")
        assert candidates == ("Alex",)

    def test_a_bare_surface_inside_a_longer_kept_surface_is_not_offered(self) -> None:
        tag_text = "Samantha Lee said hi."
        scans = _person_scan(("Sam", "Samantha Lee"))
        candidates = _anchor_candidates(scans, tag_text, (0, len(tag_text)), "Person")
        assert candidates == ("Samantha Lee",)

    def test_candidates_are_ordered_by_first_position_in_the_region(self) -> None:
        tag_text = "Riley called. Later, Alex called too."
        scans = _person_scan(("Alex", "Riley"))  # reply order: Alex, Riley
        candidates = _anchor_candidates(scans, tag_text, (0, len(tag_text)), "Person")
        assert candidates == ("Riley", "Alex")

    def test_no_person_category_among_scans_yields_no_candidates(self) -> None:
        scans = (ScanResult(category=ScrubCategory(prefix="City"), values=("Berlin",)),)
        candidates = _anchor_candidates(scans, "Berlin is nice.", (0, 15), "Person")
        assert candidates == ()


# ---------------------------------------------------------------------------
# Structural — no caller declares a speaker-name policy.
# ---------------------------------------------------------------------------


class TestNoCallerDeclaresASpeakerNamePolicy:
    def test_no_tracked_python_file_mentions_the_retired_keyword(self) -> None:
        """Every caller of the anonymize chain gets the same fold rule —
        there is no per-caller keyword declaring a speaker-name linking
        policy anywhere in the tracked source tree."""
        this_file = Path(__file__).resolve()
        repo_root = this_file.parent.parent
        retired_keyword = "link_speaker_name"
        offenders = []
        for py_file in tracked_python_files(repo_root):
            if py_file.resolve() == this_file:
                # This guard's own source names the retired keyword (as a
                # plain string, to search for it) -- not a re-declaration
                # of the caller-policy the keyword used to select.
                continue
            rel = py_file.relative_to(repo_root).as_posix()
            if not (rel.startswith("paramem/") or rel.startswith("tests/")):
                continue
            if retired_keyword in py_file.read_text(encoding="utf-8"):
                offenders.append(rel)
        assert offenders == []


# ---------------------------------------------------------------------------
# The merge — resolving one value named under different keywords across
# several scanned slices into one keyword before ``keep_or_revert`` runs.
# ---------------------------------------------------------------------------


class TestSliceMappingMerge:
    def test_an_activated_row_in_a_later_slice_overrides_an_earlier_unactivated_naming(
        self, monkeypatch
    ) -> None:
        """ "Sam" is named "City" (an unactivated allow row) by the history
        slice, then "Person" (the operator's one activated row) by the
        transcript slice — the later, activated naming wins."""
        contract, fake = _run_chain(
            monkeypatch,
            ['{"mapping": {"Sam": "City"}}', '{"mapping": {"Sam": "Person"}}'],
            categories=(ScrubCategory(prefix="Person"),),
            history=["hi Sam"],
            transcript="[user] regarding Sam second turn",
        )
        assert len(fake.calls) == 2
        assert contract.status == "ok"
        assert contract.forward["Sam"].startswith("Person_")
        assert not any(e["text"] == "Sam" for e in contract.scan_dropped_entries)

    def test_when_no_slice_names_it_active_the_earliest_slices_keyword_stands(
        self, monkeypatch
    ) -> None:
        """Neither slice names "Sam" under the one activated row — the
        earliest slice's own (unactivated) keyword ("City") is what the
        merged mapping carries into ``keep_or_revert``, not the later
        slice's ("Org")."""
        contract, fake = _run_chain(
            monkeypatch,
            ['{"mapping": {"Sam": "City"}}', '{"mapping": {"Sam": "Org"}}'],
            categories=(ScrubCategory(prefix="Person"),),
            history=["hi Sam"],
            transcript="[user] regarding Sam second turn",
        )
        assert len(fake.calls) == 2
        assert contract.status == "ok"
        assert "Sam" not in contract.forward
        entry = next(e for e in contract.scan_dropped_entries if e["text"] == "Sam")
        assert entry["category"] == "City"
        assert entry["reason"] == "reverted"

    def test_two_activated_namings_of_the_same_value_resolve_to_the_earlier_one(
        self, monkeypatch
    ) -> None:
        """ "007" is named "Phone" by the history slice and "Person" by the
        transcript slice — both activated rows, so the earlier ("Phone")
        settles the value and the later naming is never applied."""
        contract, fake = _run_chain(
            monkeypatch,
            ['{"mapping": {"007": "Phone"}}', '{"mapping": {"007": "Person"}}'],
            categories=(ScrubCategory(prefix="Person"), ScrubCategory(prefix="Phone")),
            history=["call 007"],
            transcript="[user] mention 007 again",
        )
        assert len(fake.calls) == 2
        assert contract.status == "ok"
        assert contract.forward["007"].startswith("Phone_")

    def test_a_keyword_differing_only_in_case_or_spacing_resolves_to_the_row(
        self, monkeypatch
    ) -> None:
        """The row's own name is "Person"; the scripted reply names it
        "  PERSON  " (case-folded and padded with extra whitespace) — the
        row lookup folds both before comparing."""
        contract, fake = _run_chain(
            monkeypatch,
            ['{"mapping": {"Alex": "  PERSON  "}}'],
            categories=(ScrubCategory(prefix="Person"),),
            transcript="[user] Alex is here",
        )
        assert len(fake.calls) == 1
        assert contract.status == "ok"
        assert contract.forward["Alex"].startswith("Person_")


# ---------------------------------------------------------------------------
# The counts and the raw record.
# ---------------------------------------------------------------------------


class TestModelCallsMatchCallTokens:
    def test_on_a_completed_call_call_tokens_length_equals_model_calls(self, monkeypatch) -> None:
        contract, fake = _run_chain(
            monkeypatch,
            ['{"mapping": {}}', '{"mapping": {}}'],
            categories=(ScrubCategory(prefix="Person"),),
            history=["hi"],
            transcript="[user] second turn",
        )
        assert len(fake.calls) == 2
        assert contract.status == "ok"
        assert contract.model_calls == 2
        assert len(contract.call_tokens) == contract.model_calls

    def test_on_a_failed_call_call_tokens_length_still_equals_model_calls(
        self, monkeypatch
    ) -> None:
        contract, fake = _run_chain(
            monkeypatch,
            ['{"mapping": {}}', "not json at all"],
            categories=(ScrubCategory(prefix="Person"),),
            history=["hi"],
            transcript="[user] second turn",
        )
        assert len(fake.calls) == 2
        assert contract.status == "failed"
        assert contract.failure == "scan_failed"
        assert contract.model_calls == 2
        assert len(contract.call_tokens) == contract.model_calls


class TestRawRecordCarriesOneReplyPerScannedSlice:
    def test_raw_scan_list_has_one_entry_per_slice_in_payload_order(self, monkeypatch) -> None:
        replies = [
            '{"mapping": {"hi": "City"}}',
            '{"mapping": {"second": "City"}}',
            '{"mapping": {"X": "City"}}',
        ]
        contract, fake = _run_chain(
            monkeypatch,
            replies,
            categories=(ScrubCategory(prefix="Person"),),
            history=["hi"],
            transcript="[user] second turn",
            facts=[{"subject": "X", "predicate": "p", "object": "Y"}],
        )
        assert len(fake.calls) == 3
        assert contract.status == "ok"
        raw = json.loads(contract.raw)
        assert raw["scan"] == replies
        assert raw["anchor"] == ""


# ---------------------------------------------------------------------------
# A marked value the slice does not contain as written substitutes nowhere, so
# it is dropped before a placeholder is minted.
# ---------------------------------------------------------------------------


class TestAValueTheSliceDoesNotContain:
    def test_a_value_that_substitutes_nowhere_is_never_minted(self, monkeypatch) -> None:
        contract, fake = _run_chain(
            monkeypatch,
            ['{"mapping": {"Lena": "Person"}}'],
            categories=(ScrubCategory(prefix="Person"),),
            transcript="[user] yes, lena from the choir",
        )
        assert contract.status == "ok"
        assert len(fake.calls) == 1
        assert contract.forward == {}
        assert contract.inert_dropped == 1

    def test_a_value_the_slice_contains_is_minted(self, monkeypatch) -> None:
        contract, fake = _run_chain(
            monkeypatch,
            ['{"mapping": {"Marta": "Person"}}'],
            categories=(ScrubCategory(prefix="Person"),),
            transcript="[user] her sister Marta will drive her",
        )
        assert contract.status == "ok"
        assert len(fake.calls) == 1
        assert list(contract.forward) == ["Marta"]


# ---------------------------------------------------------------------------
# Failure — a slice that fails to parse, and a slice the budget precondition
# refuses before a call is ever issued, both fail the whole call closed.
# ---------------------------------------------------------------------------


class TestScanFailureClosesTheWholeCall:
    def test_a_slice_whose_reply_does_not_parse_fails_closed(self, monkeypatch) -> None:
        contract, fake = _run_chain(
            monkeypatch,
            ['{"mapping": {}}', "not json at all"],
            categories=(ScrubCategory(prefix="Person"),),
            history=["hi"],
            transcript="[user] second turn",
            facts=[{"subject": "X", "predicate": "p", "object": "Y"}],
        )
        # The third slice (the fact block) is never reached.
        assert len(fake.calls) == 2
        assert contract.status == "failed"
        assert contract.failure == "scan_failed"
        assert contract.forward == {}
        assert contract.reverse == {}
        assert contract.anon_transcript == ""
        raw = json.loads(contract.raw)
        assert raw["scan"] == ['{"mapping": {}}', "not json at all"]
        assert raw["anchor"] == ""

    def test_a_slice_the_budget_precondition_refuses_fails_closed(self, monkeypatch) -> None:
        """The transcript slice (300 words) plus the SCAN keyword table
        cannot fit an 800-token envelope, but the one-word history slice
        can — the refusal fires on the second slice, never reaching a
        ``generate()`` call for it."""
        long_transcript = "[user] " + ("word " * 300).strip()
        contract, fake = _run_chain(
            monkeypatch,
            ['{"mapping": {}}'],
            categories=(ScrubCategory(prefix="Person"),),
            history=["hi"],
            transcript=long_transcript,
            token_envelope=800,
        )
        # Only the first (fitting) slice ever reaches generate().
        assert len(fake.calls) == 1
        assert contract.status == "failed"
        assert contract.failure == "scan_failed"
        assert contract.model_calls == 1
        assert len(contract.call_tokens) == 1
        raw = json.loads(contract.raw)
        # The contract still carries the earlier issued call's reply plus
        # the refused slice's own refusal message.
        assert raw["scan"] == ['{"mapping": {}}', "anonymize.scan budget refused: anonymize.scan"]
        assert raw["anchor"] == ""
