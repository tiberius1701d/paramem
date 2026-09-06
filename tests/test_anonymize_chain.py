"""``anonymize()`` — the opt-out door's constructor parity, the chain's
own step order and failure mapping, the anchor candidate domain, and the
structural invariant that no caller declares a per-caller speaker-name
linking policy.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from paramem.cloud.anonymize import (
    AnonymizerPrompts,
    _anchor_candidates,
    anonymize,
    failed_contract,
    opted_out_contract,
)
from paramem.cloud.anonymize_steps import AnonymizeBudgetRefused, ScanFailed, ScanResult
from paramem.config.taxonomy import ScrubCategory
from paramem.graph.flows import anonymize_turn
from tests._guard_utils import tracked_python_files
from tests.anonymizer_doubles import basic_prompts

_PROMPTS = AnonymizerPrompts(scan_system="s", scan="s", anchor_system="a", anchor="a")


def _person_scan(values: tuple[str, ...]) -> tuple[ScanResult, ...]:
    return (ScanResult(category=ScrubCategory(prefix="Person"), values=values),)


# ---------------------------------------------------------------------------
# Contract: constructors and raw
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
# model_unavailable — the SCAN call cannot be issued at all.
# ---------------------------------------------------------------------------


class TestModelUnavailableShortCircuitsBeforeAnyStep:
    def test_model_none_fails_closed_with_zero_model_calls(self, monkeypatch) -> None:
        def _explode(*args, **kwargs):
            raise AssertionError("no step function may run when the model is unavailable")

        monkeypatch.setattr("paramem.cloud.anonymize.scan_values", _explode)
        monkeypatch.setattr("paramem.cloud.anonymize.ask_speaker_anchor", _explode)

        contract = anonymize(
            [],
            None,
            None,
            transcript="Alex said hi.",
            categories=(ScrubCategory(prefix="Person"),),
            prompts=_PROMPTS,
        )

        assert contract.status == "failed"
        assert contract.failure == "model_unavailable"
        assert contract.model_calls == 0


# ---------------------------------------------------------------------------
# scan_failed — both causes (parse failure, budget refusal).
# ---------------------------------------------------------------------------


class TestScanFailed:
    def test_a_parse_failure_carries_the_raw_reply_and_one_issued_call(self, monkeypatch) -> None:
        def _raise_scan_failed(*args, **kwargs):
            raise ScanFailed("garbage", "did not parse", ({"label": "anonymize.scan"},))

        monkeypatch.setattr("paramem.cloud.anonymize.scan_values", _raise_scan_failed)

        contract = anonymize(
            [],
            object(),
            object(),
            transcript="Alex said hi.",
            categories=(ScrubCategory(prefix="Person"),),
            prompts=_PROMPTS,
        )

        assert contract.status == "failed"
        assert contract.failure == "scan_failed"
        assert contract.raw == "garbage"
        assert contract.model_calls == 1

    def test_a_budget_refusal_carries_the_refusal_message_and_zero_issued_calls(
        self, monkeypatch
    ) -> None:
        def _raise_budget_refused(*args, **kwargs):
            raise AnonymizeBudgetRefused("anonymize.scan")

        monkeypatch.setattr("paramem.cloud.anonymize.scan_values", _raise_budget_refused)

        contract = anonymize(
            [],
            object(),
            object(),
            transcript="Alex said hi.",
            categories=(ScrubCategory(prefix="Person"),),
            prompts=_PROMPTS,
        )

        assert contract.status == "failed"
        assert contract.failure == "scan_failed"
        assert "anonymize.scan" in contract.raw
        assert contract.model_calls == 0


# ---------------------------------------------------------------------------
# guard — the domain-scoped fail-closed guard.
# ---------------------------------------------------------------------------


class TestDomainScopedGuard:
    def test_named_entities_surviving_no_reconciliation_with_real_endpoints_fails_closed(
        self, monkeypatch
    ) -> None:
        scans = _person_scan(("Alex",))
        monkeypatch.setattr(
            "paramem.cloud.anonymize.scan_values",
            lambda *a, **k: (scans, (), "scan-raw", ({"label": "anonymize.scan"},)),
        )

        contract = anonymize(
            [{"subject": "Alex", "predicate": "knows", "object": "someone"}],
            object(),
            object(),
            transcript="",
            categories=(ScrubCategory(prefix="Person"),),
            identity_domain=["nomatch"],
            prompts=_PROMPTS,
        )

        assert contract.status == "failed"
        assert contract.failure == "guard"
        assert contract.rekey_dropped == 1
        assert contract.model_calls == 1


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
# model_calls: 0 / 1 / 2, and the person prefix resolved exactly once.
# ---------------------------------------------------------------------------


class TestModelCallsCount:
    def test_scan_only_no_person_category_is_one_call(self, monkeypatch) -> None:
        scans = (ScanResult(category=ScrubCategory(prefix="City"), values=("Berlin",)),)
        monkeypatch.setattr(
            "paramem.cloud.anonymize.scan_values",
            lambda *a, **k: (scans, (), "scan-raw", ({"label": "anonymize.scan"},)),
        )

        contract = anonymize(
            [],
            object(),
            object(),
            transcript="Berlin is nice.",
            categories=(ScrubCategory(prefix="City"),),
            prompts=_PROMPTS,
        )

        assert contract.model_calls == 1

    def test_scan_plus_anchor_when_a_person_value_is_a_candidate_is_two_calls(
        self, monkeypatch
    ) -> None:
        scans = _person_scan(("Alex",))
        monkeypatch.setattr(
            "paramem.cloud.anonymize.scan_values",
            lambda *a, **k: (scans, (), "scan-raw", ({"label": "anonymize.scan"},)),
        )
        monkeypatch.setattr(
            "paramem.cloud.anonymize.ask_speaker_anchor",
            lambda *a, **k: (frozenset(), "anchor-raw", ({"label": "anonymize.anchor"},)),
        )

        contract = anonymize(
            [],
            object(),
            object(),
            transcript="Alex said hi.",
            speaker_id="speaker1",
            categories=(ScrubCategory(prefix="Person"),),
            prompts=_PROMPTS,
        )

        assert contract.model_calls == 2


class TestPersonPrefixResolvedOnce:
    def test_entity_type_to_prefix_is_called_exactly_once_per_anonymize_call(
        self, monkeypatch
    ) -> None:
        from paramem.config import taxonomy as taxonomy_module

        calls: list[str] = []
        real = taxonomy_module.entity_type_to_prefix

        def _counting(entity_type: str) -> str:
            calls.append(entity_type)
            return real(entity_type)

        monkeypatch.setattr("paramem.cloud.anonymize.entity_type_to_prefix", _counting)
        scans = (ScanResult(category=ScrubCategory(prefix="City"), values=("Berlin",)),)
        monkeypatch.setattr(
            "paramem.cloud.anonymize.scan_values",
            lambda *a, **k: (scans, (), "scan-raw", ({"label": "anonymize.scan"},)),
        )

        anonymize(
            [],
            object(),
            object(),
            transcript="Berlin is nice.",
            categories=(ScrubCategory(prefix="City"),),
            prompts=_PROMPTS,
        )

        assert calls == ["person"]


# ---------------------------------------------------------------------------
# raw is one JSON document holding both calls.
# ---------------------------------------------------------------------------


class TestRawIsOneJsonDocument:
    def test_raw_parses_to_one_object_with_scan_and_anchor_keys(self, monkeypatch) -> None:
        scans = _person_scan(("Alex",))
        monkeypatch.setattr(
            "paramem.cloud.anonymize.scan_values",
            lambda *a, **k: (scans, (), "scan-raw-text", ({"label": "anonymize.scan"},)),
        )
        monkeypatch.setattr(
            "paramem.cloud.anonymize.ask_speaker_anchor",
            lambda *a, **k: (frozenset(), "anchor-raw-text", ({"label": "anonymize.anchor"},)),
        )

        contract = anonymize(
            [],
            object(),
            object(),
            transcript="Alex said hi.",
            speaker_id="speaker1",
            categories=(ScrubCategory(prefix="Person"),),
            prompts=_PROMPTS,
        )

        parsed = json.loads(contract.raw)
        assert parsed == {"scan": "scan-raw-text", "anchor": "anchor-raw-text"}


# ---------------------------------------------------------------------------
# The forward table substitutes into the transcript; inert entries counted.
# ---------------------------------------------------------------------------


class TestForwardTableSubstitutesTheTranscript:
    def test_a_kept_value_is_placeholdered_in_anon_transcript(self, monkeypatch) -> None:
        scans = _person_scan(("Alex",))
        monkeypatch.setattr(
            "paramem.cloud.anonymize.scan_values",
            lambda *a, **k: (scans, (), "scan-raw", ({"label": "anonymize.scan"},)),
        )

        contract = anonymize(
            [],
            object(),
            object(),
            transcript="Alex said hi.",
            categories=(ScrubCategory(prefix="Person"),),
            prompts=_PROMPTS,
        )

        assert contract.status == "ok"
        assert contract.anon_transcript == "Person_1 said hi."


class TestInertEntriesCounted:
    def test_a_value_absent_from_the_payload_is_pruned_and_counted(self, monkeypatch) -> None:
        scans = _person_scan(("Riley",))  # never occurs in the transcript below
        monkeypatch.setattr(
            "paramem.cloud.anonymize.scan_values",
            lambda *a, **k: (scans, (), "scan-raw", ({"label": "anonymize.scan"},)),
        )

        contract = anonymize(
            [],
            object(),
            object(),
            transcript="Alex said hi.",
            categories=(ScrubCategory(prefix="Person"),),
            prompts=_PROMPTS,
        )

        assert contract.status == "ok"
        assert contract.inert_dropped == 1
        assert contract.forward == {}


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
