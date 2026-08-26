"""``anonymize()`` — the chain: opt-out, the "ok" empty-table verdict, the
domain-scoped guard, identity-domain reconciliation, the effective
envelope measured once, the tagger failure vocabulary, and the chat-path
composition (``anonymize_turn``).

``model=None, tokenizer=None`` on most cases: on a deferral the ANCHOR
gate in :func:`~paramem.cloud.anonymize.anonymize` is closed, so it never
fires and the VRAM clamp it guards is never measured. Exercising the
anchor arm itself needs a resident-shaped ``model``/``tokenizer``
double — this module constructs one, ``_peft_model_mock``, for exactly
those cases.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from peft import PeftModel

from paramem.cloud import span_tagger
from paramem.cloud.anonymize import (
    AnonymizedContract,
    _assemble_payload,
    anonymize,
    failed_contract,
    opted_out_contract,
)
from paramem.cloud.placeholders import _substitute_whole_words, insert_placeholders
from paramem.cloud.span_tagger import TaggedSpan, TaggerUnavailable, TagResult
from paramem.config.taxonomy import ScrubCategory
from paramem.graph.flows import anonymize_turn
from paramem.utils.turn_markers import format_turn
from tests._guard_utils import tracked_python_files
from tests.anonymizer_doubles import basic_prompts


def _peft_model_mock():
    model = MagicMock()
    model.__class__ = PeftModel
    return model


PERSON = ScrubCategory(
    name="Person", prefix="Person", hints=("person name",), tagger_labels=("person",)
)
ADDRESS = ScrubCategory(
    name="Address", prefix="Address", hints=("street address",), tagger_labels=("address",)
)


def _install_tag(monkeypatch, spans: tuple[TaggedSpan, ...] = (), *, windows: int = 1):
    calls: list[str] = []

    def _fake_tag(text, labels):
        calls.append(text)
        return TagResult(spans=spans, windows=windows)

    monkeypatch.setattr(span_tagger, "tag", _fake_tag)
    return calls


def _spans_for(tag_text: str, terms: list[tuple[str, str, float]]) -> tuple[TaggedSpan, ...]:
    spans = []
    for value, label, score in terms:
        idx = tag_text.index(value)
        spans.append(
            TaggedSpan(start=idx, end=idx + len(value), text=value, label=label, score=score)
        )
    return tuple(spans)


def _forbid_tag(monkeypatch) -> None:
    def _boom(text, labels):
        raise AssertionError("span_tagger.tag must not be called on the opt-out path")

    monkeypatch.setattr(span_tagger, "tag", _boom)


def _run_anonymize(
    monkeypatch,
    *,
    transcript,
    history=(),
    facts=None,
    categories,
    model=None,
    tokenizer=None,
    **kwargs,
):
    """Runs the real anonymize() with the tagger stubbed against the
    ACTUAL tag_text the chain assembles (via _assemble_payload), so
    span offsets are always correct without hand computation.

    ``model``/``tokenizer`` default to ``None`` (a deferral) — the shape
    every existing call site in this module relies on; a caller exercising
    the ANCHOR call's positive path passes a resident-shaped double
    explicitly.
    """
    facts = facts if facts is not None else []
    payload = _assemble_payload(list(history), transcript, facts)
    terms = kwargs.pop("terms", [])
    spans = _spans_for(payload.tag_text, terms)
    _install_tag(monkeypatch, spans, windows=kwargs.pop("windows", 1))
    return anonymize(
        facts,
        model,
        tokenizer,
        transcript=transcript,
        history=history,
        categories=categories,
        prompts=basic_prompts(),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Opt-out
# ---------------------------------------------------------------------------


class TestOptOut:
    def test_empty_categories_reaches_opted_out_with_zero_tagger_and_model_calls(
        self, monkeypatch
    ) -> None:
        _forbid_tag(monkeypatch)
        contract = anonymize(
            [],
            None,
            None,
            transcript="[user] hi",
            categories=[],
            prompts=basic_prompts(),
        )
        assert contract.status == "opted_out"
        assert contract.model_calls == 0
        assert contract.tagger_windows == 0

    def test_anonymize_turn_reaches_the_same_opt_out_door(self, monkeypatch) -> None:
        _forbid_tag(monkeypatch)
        contract = anonymize_turn(
            "hi there",
            _peft_model_mock(),
            MagicMock(),
            categories=(),
            token_envelope=8192,
        )
        assert contract.status == "opted_out"
        assert contract.model_calls == 0


# ---------------------------------------------------------------------------
# status="ok" with an empty table is not a failure
# ---------------------------------------------------------------------------


class TestOkWithEmptyTableIsNotAFailure:
    def test_nothing_in_scope_is_a_legitimate_ok_verdict(self, monkeypatch) -> None:
        contract = _run_anonymize(
            monkeypatch,
            transcript="[user] the weather is nice today",
            categories=[PERSON],
        )
        assert contract.status == "ok"
        assert contract.failure is None
        assert contract.forward == {}


# ---------------------------------------------------------------------------
# The domain-scoped fail-closed guard
# ---------------------------------------------------------------------------


class TestDomainGuard:
    def test_guard_fires_when_reconciliation_drops_everything_and_facts_have_real_endpoints(
        self, monkeypatch
    ) -> None:
        facts = [{"subject": "speaker1", "predicate": "knows", "object": "Someone Real"}]
        contract = _run_anonymize(
            monkeypatch,
            transcript="[user] I know Alex",
            facts=facts,
            categories=[PERSON],
            identity_domain=["a name unrelated to alex"],
            terms=[("Alex", "person", 0.9)],
        )
        assert contract.status == "failed"
        assert contract.failure == "guard"

    def test_guard_does_not_fire_when_facts_endpoints_are_speaker_only(self, monkeypatch) -> None:
        # identity_domain carries a real non-speaker name, but the SLICE's
        # own facts endpoints are speaker-only — the guard reads facts,
        # never identity_domain, so it must not fire here.
        facts = [{"subject": "speaker1", "predicate": "greeted", "object": "speaker1"}]
        contract = _run_anonymize(
            monkeypatch,
            transcript="[user] I know Alex",
            facts=facts,
            categories=[PERSON],
            identity_domain=["some unrelated real person"],
            terms=[("Alex", "person", 0.9)],
        )
        assert contract.status == "ok"
        assert contract.failure is None

    def test_a_failed_guard_contract_keeps_its_real_counters(self, monkeypatch) -> None:
        facts = [{"subject": "speaker1", "predicate": "knows", "object": "Someone Real"}]
        contract = _run_anonymize(
            monkeypatch,
            transcript="[user] I know Alex, said speaker1",
            facts=facts,
            categories=[PERSON],
            identity_domain=["a name unrelated to alex"],
            terms=[("Alex", "person", 0.9), ("speaker1", "person", 0.9)],
        )
        assert contract.status == "failed"
        assert contract.failure == "guard"
        # "speaker1" was scanned but dropped (speaker-id-shaped) before
        # reconciliation ever ran — the real, non-zero diagnostic survives
        # on the failed contract.
        assert contract.scan_dropped == 1
        assert contract.scan_dropped_entries[0]["reason"] == "speaker_id"


class TestIdentityDomainNoneSkipsReconciliation:
    def test_forward_keeps_the_raw_scanned_surface_unreconciled(self, monkeypatch) -> None:
        contract = _run_anonymize(
            monkeypatch,
            transcript="[user] My name is Alex",
            categories=[PERSON],
            terms=[("Alex", "person", 0.9)],
        )
        assert contract.status == "ok"
        assert contract.rekey_dropped == 0
        assert "Alex" in contract.forward


# ---------------------------------------------------------------------------
# tagger_windows, anon_transcript, envelope measurement
# ---------------------------------------------------------------------------


class TestTaggerWindowsEqualsTagResultWindows:
    def test_windows_field_is_carried_through_verbatim(self, monkeypatch) -> None:
        _install_tag(monkeypatch, spans=(), windows=5)
        contract = anonymize(
            [],
            None,
            None,
            transcript="",
            categories=[PERSON],
            prompts=basic_prompts(),
        )
        assert contract.tagger_windows == 5


class TestAnonTranscriptSubstitution:
    def test_anon_transcript_is_substitute_whole_words_over_transcript_and_forward(
        self, monkeypatch
    ) -> None:
        transcript = "[user] My name is Alex"
        contract = _run_anonymize(
            monkeypatch, transcript=transcript, categories=[PERSON], terms=[("Alex", "person", 0.9)]
        )
        assert contract.anon_transcript == _substitute_whole_words(transcript, contract.forward)
        assert contract.anon_transcript.startswith("[user] ")  # marker-bearing
        assert "Alex" not in contract.anon_transcript

    def test_anon_transcript_is_empty_only_when_transcript_is_empty(self, monkeypatch) -> None:
        contract = _run_anonymize(monkeypatch, transcript="", categories=[PERSON])
        assert contract.anon_transcript == ""


class TestInertKeyPruning:
    """A forward-table key that substitutes nowhere in the payload is not
    a key: two same-category spans that overlap without nesting both
    survive the scan, but longest-first substitution means the second
    writes zero placeholders. It is pruned before
    ``reverse``/``declared``/``anon_transcript`` are derived, counted into
    ``inert_dropped``, and recorded in ``scan_dropped_entries`` with
    ``reason="inert"``/``side="table"``.
    """

    def test_overlapping_non_nesting_address_spans_prune_the_inert_one(self, monkeypatch) -> None:
        transcript = "[user] Lives at Schillerpromenade 63, 12049 Berlin, Abteilung 3."
        contract = _run_anonymize(
            monkeypatch,
            transcript=transcript,
            categories=[ADDRESS],
            terms=[
                ("Schillerpromenade 63, 12049 Berlin", "address", 0.9),
                ("12049 Berlin, Abteilung 3", "address", 0.9),
            ],
        )
        assert contract.status == "ok"
        assert contract.forward == {"Schillerpromenade 63, 12049 Berlin": "Address_1"}
        assert contract.inert_dropped == 1
        inert_entries = [e for e in contract.scan_dropped_entries if e["reason"] == "inert"]
        assert len(inert_entries) == 1
        assert inert_entries[0]["side"] == "table"
        assert inert_entries[0]["category"] == "Address"
        # No partial-address reverse entry survives to be pulled back
        # through by a cloud reply that mints its placeholder.
        assert "12049 Berlin, Abteilung 3" not in contract.reverse.values()
        assert contract.reverse == {"Address_1": "Schillerpromenade 63, 12049 Berlin"}
        assert contract.declared == frozenset({"Address_1"})

    def test_a_single_live_span_is_not_pruned(self, monkeypatch) -> None:
        transcript = "[user] My name is Alex"
        contract = _run_anonymize(
            monkeypatch, transcript=transcript, categories=[PERSON], terms=[("Alex", "person", 0.9)]
        )
        assert contract.inert_dropped == 0
        assert not any(e["reason"] == "inert" for e in contract.scan_dropped_entries)
        assert "Alex" in contract.forward


class TestShadowedKeyLiveAtAStandaloneOccurrence:
    """A forward-table key that a longer key shadows at ONE position (the
    longest-first walk consumes the longer match there, so the shorter key
    never gets to try that position) is still live overall when it also
    occurs standalone at another position — pruning is decided over the
    WHOLE payload, not per-occurrence.
    """

    def test_a_shorter_key_shadowed_once_but_standalone_elsewhere_is_not_pruned(
        self, monkeypatch
    ) -> None:
        transcript = "[user] Alex called. Then Alex Smith also texted."
        contract = _run_anonymize(
            monkeypatch,
            transcript=transcript,
            categories=[PERSON],
            terms=[("Alex", "person", 0.9), ("Alex Smith", "person", 0.9)],
        )
        assert contract.status == "ok"
        assert contract.inert_dropped == 0
        assert not any(e["reason"] == "inert" for e in contract.scan_dropped_entries)
        assert "Alex" in contract.forward
        assert "Alex Smith" in contract.forward


class TestDomainGuardIsDecidedOnThePrunedTable:
    """The domain-scoped fail-closed guard reads the forward table AFTER
    inert-key pruning, not before: a reconciliation match that re-keys
    onto the identity domain's own (differently-cased) surface can
    produce a non-empty table that is nonetheless entirely inert against
    the actual payload text (case-sensitive substitution). Once pruned,
    that table is empty, and the guard must fire on it — never on the
    pre-prune snapshot, which would silently let an unreconciled fact
    through as an "ok" verdict with a quietly empty forward map.
    """

    def test_a_reconciled_but_case_mismatched_key_still_trips_the_guard(self, monkeypatch) -> None:
        facts = [{"subject": "speaker1", "predicate": "knows", "object": "Someone Real"}]
        contract = _run_anonymize(
            monkeypatch,
            transcript="[user] I know Alex",
            facts=facts,
            categories=[PERSON],
            identity_domain=["ALEX"],
            terms=[("Alex", "person", 0.9)],
        )
        assert contract.status == "failed"
        assert contract.failure == "guard"


class TestInertEntriesReachTheContract:
    """Every entry the table build prunes as inert lands in
    ``scan_dropped_entries``, and ``inert_dropped`` agrees exactly with
    the count of ``reason == "inert"`` entries in that list — a count
    that can disagree with the list it summarises is not a field.
    """

    def test_the_builds_inert_entries_land_in_scan_dropped_entries_and_the_count(
        self, monkeypatch
    ) -> None:
        transcript = "[user] Lives at Schillerpromenade 63, 12049 Berlin, Abteilung 3."
        contract = _run_anonymize(
            monkeypatch,
            transcript=transcript,
            categories=[ADDRESS],
            terms=[
                ("Schillerpromenade 63, 12049 Berlin", "address", 0.9),
                ("12049 Berlin, Abteilung 3", "address", 0.9),
            ],
        )
        inert_entries = [e for e in contract.scan_dropped_entries if e["reason"] == "inert"]
        assert contract.inert_dropped == len(inert_entries)
        assert inert_entries
        for entry in inert_entries:
            assert entry["side"] == "table"
            assert entry["category"] == ADDRESS.prefix


class TestFoldedKeyIsPrunedLikeAnyOther:
    """The enrolled name, entered as a forward key on the speaker group,
    is pruned exactly like any other key when the payload never contains
    it — its inert record carries the speaker group's own empty
    category, not the person prefix.
    """

    def test_a_folded_key_absent_from_the_payload_is_counted_inert_with_an_empty_category(
        self, monkeypatch
    ) -> None:
        contract = _run_anonymize(
            monkeypatch,
            transcript="[user] I went for a walk.",
            categories=[PERSON],
            speaker_id="speaker1",
            speaker_name="Alex Morgan",
        )
        assert contract.status == "ok"
        assert contract.forward == {}
        assert contract.inert_dropped == 1
        inert_entries = [e for e in contract.scan_dropped_entries if e["reason"] == "inert"]
        assert len(inert_entries) == 1
        assert inert_entries[0]["category"] == ""
        assert inert_entries[0]["side"] == "table"


class TestScanDroppedCountExcludesTableSideDrops:
    """``scan_dropped`` (the int) counts only ``side="scan"`` entries — the
    scan step's own drops. Once an inert (``side="table"``) entry lands in
    ``scan_dropped_entries``, the int is strictly smaller than the list's
    length.
    """

    def test_scan_dropped_is_smaller_than_len_scan_dropped_entries_once_inert_lands(
        self, monkeypatch
    ) -> None:
        transcript = "[user] speaker1 lives at Schillerpromenade 63, 12049 Berlin, Abteilung 3."
        contract = _run_anonymize(
            monkeypatch,
            transcript=transcript,
            categories=[ADDRESS, PERSON],
            terms=[
                ("Schillerpromenade 63, 12049 Berlin", "address", 0.9),
                ("12049 Berlin, Abteilung 3", "address", 0.9),
                ("speaker1", "person", 0.9),
            ],
        )
        assert contract.scan_dropped == 1
        assert len(contract.scan_dropped_entries) == 2
        assert contract.scan_dropped < len(contract.scan_dropped_entries)


class TestNonAsciiAndQuoteSurviveEndToEnd:
    """The tagger reads exactly the strings the consumer substitutes:
    non-ASCII, ``"`` and ``\\`` survive verbatim, so a fact value carrying
    them is placeholdered end-to-end — ``anonymize()`` then
    :func:`~paramem.cloud.placeholders.insert_placeholders`.
    """

    def test_a_quoted_non_ascii_fact_value_is_placeholdered_end_to_end(self, monkeypatch) -> None:
        raw_value = 'Lindenstraße 44, "Hinterhof"'
        facts = [{"subject": "speaker1", "predicate": "lives at", "object": raw_value}]
        payload = _assemble_payload([], "", facts)

        # The tagger must see the raw value verbatim in tag_text — not a
        # rendering that has escaped the quote or the non-ASCII char away
        # from what insert_placeholders will later substitute.
        assert raw_value in payload.tag_text

        idx = payload.tag_text.index(raw_value)
        span = TaggedSpan(
            start=idx, end=idx + len(raw_value), text=raw_value, label="address", score=0.9
        )
        _install_tag(monkeypatch, (span,), windows=1)

        contract = anonymize(
            facts,
            None,
            None,
            transcript="",
            categories=[ADDRESS],
            prompts=basic_prompts(),
        )

        assert contract.status == "ok"
        anonymized = insert_placeholders(facts, contract.forward)
        assert anonymized[0]["object"] == "Address_1"
        assert "Hinterhof" not in anonymized[0]["object"]


class TestHistoryOnlyValueIsInForward:
    def test_a_name_appearing_only_in_history_is_placeholdered(self, monkeypatch) -> None:
        history = ["[user] My name is Alex"]
        transcript = "[assistant] Nice to meet you"
        contract = _run_anonymize(
            monkeypatch,
            transcript=transcript,
            history=history,
            categories=[PERSON],
            terms=[("Alex", "person", 0.9)],
        )
        assert "Alex" in contract.forward
        assert contract.status == "ok"


# ---------------------------------------------------------------------------
# Failure vocabulary — the tagger cause
# ---------------------------------------------------------------------------


class TestTaggerFailureVocabulary:
    def test_a_raising_tagger_yields_failed_tagger_with_empty_forward_and_facts(
        self, monkeypatch
    ) -> None:
        def _raising_tag(text, labels):
            raise TaggerUnavailable("no handle loaded")

        monkeypatch.setattr(span_tagger, "tag", _raising_tag)

        contract = anonymize(
            [{"subject": "speaker1", "predicate": "knows", "object": "Alex"}],
            None,
            None,
            transcript="[user] hi",
            categories=[PERSON],
            prompts=basic_prompts(),
        )

        assert contract.status == "failed"
        assert contract.failure == "tagger"
        assert contract.forward == {}
        assert contract.facts == []
        assert "no handle loaded" in contract.raw
        assert contract.tagger_windows == 0
        assert contract.model_calls == 0

    def test_no_exception_escapes_the_chain(self, monkeypatch) -> None:
        def _raising_tag(text, labels):
            raise TaggerUnavailable("boom")

        monkeypatch.setattr(span_tagger, "tag", _raising_tag)

        # Must not raise.
        contract = anonymize(
            [],
            None,
            None,
            transcript="[user] hi",
            categories=[PERSON],
            prompts=basic_prompts(),
        )
        assert contract.status == "failed"


class TestRuntimeErrorPropagatesOutOfAnonymizeTurn:
    def test_a_bare_runtime_error_from_the_tagger_propagates_unchanged(self, monkeypatch) -> None:
        def _raising_tag(text, labels):
            raise RuntimeError("device not ready")

        monkeypatch.setattr(span_tagger, "tag", _raising_tag)

        with pytest.raises(RuntimeError, match="device not ready"):
            anonymize_turn(
                "hello",
                _peft_model_mock(),
                MagicMock(),
                categories=[PERSON],
                token_envelope=8192,
            )


# ---------------------------------------------------------------------------
# Contract: constructors and raw
# ---------------------------------------------------------------------------


class TestContractConstructors:
    def test_failure_is_guard_tagger_or_none(self, monkeypatch) -> None:
        ok_contract = _run_anonymize(monkeypatch, transcript="[user] hi", categories=[PERSON])
        assert ok_contract.failure is None

        def _raising_tag(text, labels):
            raise TaggerUnavailable("boom")

        monkeypatch.setattr(span_tagger, "tag", _raising_tag)
        tagger_failed = anonymize(
            [],
            None,
            None,
            transcript="[user] hi",
            categories=[PERSON],
            prompts=basic_prompts(),
        )
        assert tagger_failed.failure == "tagger"

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

    def test_failed_contract_matches_the_chains_own_tagger_failure_shape(self, monkeypatch) -> None:
        def _raising_tag(text, labels):
            raise TaggerUnavailable("boom message")

        monkeypatch.setattr(span_tagger, "tag", _raising_tag)
        via_chain = anonymize(
            [],
            None,
            None,
            transcript="[user] hi",
            categories=[PERSON],
            prompts=basic_prompts(),
        )
        via_constructor = failed_contract(failure="tagger", raw="boom message")
        assert via_constructor == via_chain

    def test_render_scan_raw_is_the_only_producer_of_raw_on_a_successful_terminal(
        self, monkeypatch
    ) -> None:
        contract = _run_anonymize(
            monkeypatch,
            transcript="[user] My name is Alex",
            categories=[PERSON],
            terms=[("Alex", "person", 0.9)],
        )
        import json

        parsed = json.loads(contract.raw)
        assert "spans" in parsed
        assert "anchor" in parsed

    def test_a_tagger_terminal_carries_the_refusal_message_as_raw(self, monkeypatch) -> None:
        def _raising_tag(text, labels):
            raise TaggerUnavailable("refusal text here")

        monkeypatch.setattr(span_tagger, "tag", _raising_tag)
        contract = anonymize(
            [],
            None,
            None,
            transcript="[user] hi",
            categories=[PERSON],
            prompts=basic_prompts(),
        )
        assert "refusal text here" in contract.raw


# ---------------------------------------------------------------------------
# The ANCHOR call fires only when its three-way precondition holds.
# ---------------------------------------------------------------------------


class TestAnchorFires:
    def test_anchor_call_does_not_fire_without_a_well_shaped_speaker_id(self, monkeypatch) -> None:
        import paramem.cloud.anonymize as anonymize_module

        def _fake_anchor(*a, **k):  # pragma: no cover - must not be reached
            raise AssertionError("ask_speaker_anchor must not be called without speaker_id")

        monkeypatch.setattr(anonymize_module, "ask_speaker_anchor", _fake_anchor)

        contract = _run_anonymize(
            monkeypatch,
            transcript="[user] My name is Alex",
            categories=[PERSON],
            speaker_id=None,
            terms=[("Alex", "person", 0.9)],
        )
        assert contract.model_calls == 0

    def test_anchor_call_does_not_fire_on_a_deferral_even_with_candidates_and_speaker_id(
        self, monkeypatch
    ) -> None:
        """``model=None, tokenizer=None`` closes the ANCHOR gate even when
        the other two preconditions (well-shaped ``speaker_id``, non-empty
        candidates) hold — the residency term added at the gate.  The VRAM
        clamp (:func:`~paramem.utils.vram_guard.effective_token_envelope`)
        has exactly one consumer, the anchor call, so it must not be
        measured either: on a cloud-only deferral this must open no CUDA
        context.
        """
        import paramem.cloud.anonymize as anonymize_module

        def _fake_anchor(*a, **k):  # pragma: no cover - must not be reached
            raise AssertionError("ask_speaker_anchor must not be called on a deferral")

        def _fake_envelope(*a, **k):  # pragma: no cover - must not be reached
            raise AssertionError("effective_token_envelope must not be measured on a deferral")

        monkeypatch.setattr(anonymize_module, "ask_speaker_anchor", _fake_anchor)
        monkeypatch.setattr(anonymize_module, "effective_token_envelope", _fake_envelope)

        contract = _run_anonymize(
            monkeypatch,
            transcript="[user] My name is Alex",
            categories=[PERSON],
            speaker_id="speaker1",
            terms=[("Alex", "person", 0.9)],
        )
        assert contract.status == "ok"
        assert contract.model_calls == 0
        # The rest of the chain still ran to completion — a deferral is a
        # designed input, not a failure: the tagger's own scan still mints
        # a forward-table entry for the candidate.
        assert "Alex" in contract.forward

    def test_anchor_call_fires_on_a_resident_double_and_measures_the_envelope_once(
        self, monkeypatch
    ) -> None:
        """The positive-path mirror of the two tests above: a
        resident-shaped ``model``/``tokenizer`` double, a well-shaped
        ``speaker_id`` and a non-empty candidate set together open the
        gate — ``ask_speaker_anchor`` is called exactly once and the VRAM
        clamp is measured exactly once.
        """
        import paramem.cloud.anonymize as anonymize_module

        anchor_calls: list[tuple] = []
        envelope_calls: list[int] = []

        def _fake_anchor(*a, **k):
            anchor_calls.append((a, k))
            return (frozenset({"Alex"}), "anchor raw text", ())

        def _fake_envelope(configured_envelope):
            envelope_calls.append(configured_envelope)
            return configured_envelope, None

        monkeypatch.setattr(anonymize_module, "ask_speaker_anchor", _fake_anchor)
        monkeypatch.setattr(anonymize_module, "effective_token_envelope", _fake_envelope)

        contract = _run_anonymize(
            monkeypatch,
            transcript="[user] My name is Alex",
            categories=[PERSON],
            speaker_id="speaker1",
            terms=[("Alex", "person", 0.9)],
            model=_peft_model_mock(),
            tokenizer=MagicMock(),
        )

        assert len(anchor_calls) == 1
        assert len(envelope_calls) == 1
        assert contract.status == "ok"
        # The anchor fold, not the mint loop, wrote this entry — "Alex"
        # folds directly onto speaker_id.
        assert contract.forward["Alex"] == "speaker1"


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
# Chat path — anonymize_turn
# ---------------------------------------------------------------------------


class TestChatPath:
    def test_anonymize_turn_returns_the_contract_object_unmodified(self, monkeypatch) -> None:
        sentinel = AnonymizedContract(
            status="ok",
            forward={"Alex": "Person_1"},
            reverse={"Person_1": "Alex"},
            anon_transcript="[user] hi Person_1",
            declared=frozenset({"Person_1"}),
            rekey_dropped=0,
            raw="{}",
        )

        import paramem.graph.flows as flows_module

        monkeypatch.setattr(flows_module, "anonymize", lambda *a, **k: sentinel)

        result = anonymize_turn(
            "hi there", _peft_model_mock(), MagicMock(), categories=[PERSON], token_envelope=8192
        )

        assert result is sentinel  # identity, not a dataclasses.replace copy

    def test_current_turn_substitution_matches_the_production_call_shape(self, monkeypatch) -> None:
        text = "My name is Alex"
        payload = _run_anonymize_turn(monkeypatch, text=text, terms=[("Alex", "person", 0.9)])

        anon_text = _substitute_whole_words(text, payload.forward)

        assert anon_text == "My name is Person_1"
        assert "[user]" not in anon_text

    def test_history_only_name_is_placeholdered_in_both_current_turn_and_history(
        self, monkeypatch
    ) -> None:
        history = [{"role": "user", "text": "My name is Alex"}]
        text = "Hello there"
        payload = _run_anonymize_turn(
            monkeypatch, text=text, history=history, terms=[("Alex", "person", 0.9)]
        )

        anon_text = _substitute_whole_words(text, payload.forward)
        sanitized_history = [
            {**turn, "text": _substitute_whole_words(turn["text"], payload.forward)}
            for turn in history
        ]

        assert anon_text == "Hello there"  # no Alex in the current turn itself
        assert "Alex" not in sanitized_history[0]["text"]
        assert "Person_1" in sanitized_history[0]["text"]


class TestChatPathDeferral:
    """``anonymize_turn(model=None, tokenizer=None)`` — a cloud-only
    deferral, a designed input of the chain (see ``anonymize_turn``'s
    docstring), never a configuration.  The tagger scan and the rest of
    the chain run identically to the resident case; only the ANCHOR gate
    inside :func:`~paramem.cloud.anonymize.anonymize` closes.
    """

    def test_a_deferral_never_enters_base_model_inference(self, monkeypatch) -> None:
        import paramem.graph.flows as flows_module

        def _boom(model):  # pragma: no cover - must not be reached
            raise AssertionError("base_model_inference must not be entered when model is None")

        monkeypatch.setattr(flows_module, "base_model_inference", _boom)

        text = "My name is Alex"
        history_lines = []
        model_facing_transcript = format_turn("user", text)
        payload = _assemble_payload(history_lines, model_facing_transcript, [])
        spans = _spans_for(payload.tag_text, [("Alex", "person", 0.9)])
        _install_tag(monkeypatch, spans, windows=1)

        result = anonymize_turn(
            text,
            None,
            None,
            categories=[PERSON],
            token_envelope=8192,
        )

        assert result.status == "ok"

    def test_a_deferral_still_reaches_anonymize_and_its_forward_table_comes_from_the_tagger(
        self, monkeypatch
    ) -> None:
        text = "My name is Alex"
        history_lines = []
        model_facing_transcript = format_turn("user", text)
        payload = _assemble_payload(history_lines, model_facing_transcript, [])
        spans = _spans_for(payload.tag_text, [("Alex", "person", 0.9)])
        calls = _install_tag(monkeypatch, spans, windows=1)

        result = anonymize_turn(
            text,
            None,
            None,
            categories=[PERSON],
            token_envelope=8192,
        )

        assert calls, "the tagger must still be invoked on a deferral"
        assert result.status == "ok"
        assert "Alex" in result.forward
        assert result.forward["Alex"].startswith("Person_")


def _run_anonymize_turn(monkeypatch, *, text, history=(), terms=()):
    """Mirrors anonymize_turn's own marker rendering so the tagger stub's
    planted spans land at the right tag_text offsets.
    """
    history_lines = [format_turn(t["role"], t["text"]) for t in history]
    model_facing_transcript = format_turn("user", text)
    payload = _assemble_payload(history_lines, model_facing_transcript, [])
    spans = _spans_for(payload.tag_text, list(terms))
    _install_tag(monkeypatch, spans, windows=1)
    return anonymize_turn(
        text,
        _peft_model_mock(),
        MagicMock(),
        history=history,
        categories=[PERSON],
        token_envelope=8192,
    )
