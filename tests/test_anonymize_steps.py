"""``scan_values`` and ``_anchor_candidates`` — the tagger-backed scan and
the closed candidate domain the ANCHOR call is shown.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from paramem.cloud import anonymize_steps as anonymize_steps_module
from paramem.cloud import span_tagger
from paramem.cloud.anonymize import _anchor_candidates
from paramem.cloud.anonymize_steps import ask_speaker_anchor, scan_values
from paramem.cloud.span_tagger import TaggedSpan, TaggerUnavailable, TagResult
from paramem.config.taxonomy import ScrubCategory

PERSON = ScrubCategory(
    name="Person", prefix="Person", hints=("person name",), tagger_labels=("person",)
)
PHONE = ScrubCategory(
    name="Phone", prefix="Phone", hints=("phone number",), tagger_labels=("phone number",)
)
EMAIL = ScrubCategory(
    name="Email", prefix="Email", hints=("email address",), tagger_labels=("email",)
)
ADDRESS = ScrubCategory(
    name="Address", prefix="Address", hints=("street address",), tagger_labels=("address",)
)


def _install_tag(monkeypatch, spans: tuple[TaggedSpan, ...], *, windows: int = 1):
    calls: list[list[str]] = []

    def _fake_tag(text, labels):
        calls.append(list(labels))
        return TagResult(spans=spans, windows=windows)

    monkeypatch.setattr(span_tagger, "tag", _fake_tag)
    return calls


class TestScanValuesLabelUnion:
    def test_label_list_is_the_ordered_union_of_category_tagger_labels(self, monkeypatch) -> None:
        calls = _install_tag(monkeypatch, ())
        scan_values("some payload", categories=[PERSON, PHONE, EMAIL])
        assert calls == [["person", "phone number", "email"]]

    def test_exactly_one_tag_call_regardless_of_category_count(self, monkeypatch) -> None:
        calls = _install_tag(monkeypatch, ())
        scan_values("some payload", categories=[PERSON, PHONE, EMAIL])
        assert len(calls) == 1


class TestScanValuesPartitionsByOwningCategory:
    def test_spans_are_grouped_under_their_owning_category(self, monkeypatch) -> None:
        payload = "Alex called 555-1234 today"
        spans = (
            TaggedSpan(start=0, end=4, text="Alex", label="person", score=0.9),
            TaggedSpan(start=12, end=20, text="555-1234", label="phone number", score=0.9),
        )
        _install_tag(monkeypatch, spans)

        results, _tag_result = scan_values(payload, categories=[PERSON, PHONE])

        by_prefix = {r.category.prefix: r.values for r in results}
        assert by_prefix["Person"] == ("Alex",)
        assert by_prefix["Phone"] == ("555-1234",)


class TestScanValuesOneResultPerCategory:
    def test_every_configured_category_gets_a_result_including_empty_ones(
        self, monkeypatch
    ) -> None:
        payload = "Alex called someone"
        spans = (TaggedSpan(start=0, end=4, text="Alex", label="person", score=0.9),)
        _install_tag(monkeypatch, spans)

        results, _tag_result = scan_values(payload, categories=[PERSON, PHONE, EMAIL])

        assert [r.category.prefix for r in results] == ["Person", "Phone", "Email"]
        assert results[0].values == ("Alex",)
        assert results[1].values == ()
        assert results[2].values == ()


class TestScanValuesDropReasons:
    def test_speaker_id_shaped_surface_drops_with_reason(self, monkeypatch) -> None:
        payload = "speaker1 said hello"
        spans = (TaggedSpan(start=0, end=8, text="speaker1", label="person", score=0.9),)
        _install_tag(monkeypatch, spans)

        results, _tag_result = scan_values(payload, categories=[PERSON])

        assert results[0].values == ()
        assert len(results[0].dropped) == 1
        assert results[0].dropped[0]["reason"] == "speaker_id"
        assert results[0].dropped[0]["text"] == "speaker1"
        assert results[0].dropped[0]["category"] == "Person"
        assert results[0].dropped[0]["side"] == "scan"

    def test_non_whole_word_span_drops_with_reason(self, monkeypatch) -> None:
        payload = "Billing dept"
        # "Bill" at [0:4) is immediately followed by 'i' — not a whole word.
        spans = (TaggedSpan(start=0, end=4, text="Bill", label="person", score=0.9),)
        _install_tag(monkeypatch, spans)

        results, _tag_result = scan_values(payload, categories=[PERSON])

        assert results[0].values == ()
        assert results[0].dropped[0]["reason"] == "not_whole_word"


class TestScanValuesExactSurfaceDedupAndOrdering:
    def test_lena_and_lena_lowercase_both_survive_as_distinct_values(self, monkeypatch) -> None:
        payload = "Lena met lena later"
        spans = (
            TaggedSpan(start=0, end=4, text="Lena", label="person", score=0.9),
            TaggedSpan(start=9, end=13, text="lena", label="person", score=0.9),
        )
        _install_tag(monkeypatch, spans)

        results, _tag_result = scan_values(payload, categories=[PERSON])

        assert results[0].values == ("Lena", "lena")

    def test_values_are_ordered_by_first_occurrence_offset(self, monkeypatch) -> None:
        payload = "Zoe then Amy"
        # Deliberately out of offset order in the raw span list.
        spans = (
            TaggedSpan(start=9, end=12, text="Amy", label="person", score=0.9),
            TaggedSpan(start=0, end=3, text="Zoe", label="person", score=0.9),
        )
        _install_tag(monkeypatch, spans)

        results, _tag_result = scan_values(payload, categories=[PERSON])

        assert results[0].values == ("Zoe", "Amy")


class TestScanValuesContainmentDrop:
    def test_a_fragment_strictly_inside_a_longer_same_category_span_is_dropped_contained(
        self, monkeypatch
    ) -> None:
        full = "friedrich.ammerschlaeger@example.de"
        fragment = "ammerschlaeger@example.de"
        payload = f"Email: {full} please"
        start_full = payload.index(full)
        end_full = start_full + len(full)
        start_frag = payload.index(fragment)
        end_frag = start_frag + len(fragment)
        assert start_full < start_frag and end_frag <= end_full  # fragment strictly inside

        spans = (
            TaggedSpan(start=start_full, end=end_full, text=full, label="email", score=0.9),
            TaggedSpan(start=start_frag, end=end_frag, text=fragment, label="email", score=0.9),
        )
        _install_tag(monkeypatch, spans)

        results, _tag_result = scan_values(payload, categories=[EMAIL])

        assert results[0].values == (full,)
        assert len(results[0].dropped) == 1
        assert results[0].dropped[0]["reason"] == "contained"
        assert results[0].dropped[0]["text"] == fragment
        assert results[0].dropped[0]["category"] == "Email"

    def test_a_span_contained_in_a_different_category_span_is_never_dropped_for_containment(
        self, monkeypatch
    ) -> None:
        # The whole payload IS the address span, so it strictly covers the
        # phone number's offsets by construction — but it is a DIFFERENT
        # category, and containment is same-category-only, so both survive.
        payload = "12 Main St 555-1234 Springfield"
        phone = "555-1234"
        start_phone = payload.index(phone)
        end_phone = start_phone + len(phone)

        spans = (
            TaggedSpan(start=0, end=len(payload), text=payload, label="address", score=0.9),
            TaggedSpan(
                start=start_phone, end=end_phone, text=phone, label="phone number", score=0.9
            ),
        )
        _install_tag(monkeypatch, spans)

        results, _tag_result = scan_values(payload, categories=[ADDRESS, PHONE])

        by_prefix = {r.category.prefix: r.values for r in results}
        assert by_prefix["Address"] == (payload,)
        assert by_prefix["Phone"] == (phone,)
        assert not any(r.dropped for r in results)

    def test_equal_length_identical_offset_duplicate_is_not_reported_as_contained(
        self, monkeypatch
    ) -> None:
        # Two spans at the IDENTICAL offset/length — containment requires
        # the covering span to be STRICTLY longer, so this never reaches
        # the "contained" branch; it instead hits pass 2's existing
        # exact-surface dedup (silently, no dropped-entry record — see
        # TestScanValuesExactSurfaceDedupAndOrdering for that mechanism's
        # own distinct-surface case).
        spans = (
            TaggedSpan(start=0, end=4, text="Lena", label="person", score=0.9),
            TaggedSpan(start=0, end=4, text="Lena", label="person", score=0.9),
        )
        _install_tag(monkeypatch, spans)

        results, _tag_result = scan_values("Lena called", categories=[PERSON])

        assert results[0].values == ("Lena",)
        assert results[0].dropped == ()

    def test_single_span_verification_reports_not_whole_word_before_containment_is_evaluated(
        self, monkeypatch
    ) -> None:
        # "Bill" at [0:4) is immediately followed by "inger" — fails the
        # whole-word check — AND it is also strictly contained inside the
        # "Billinger" span. Pass 1 (speaker id / whole word) runs over
        # every span BEFORE pass 2 (containment) ever sees the survivors,
        # so a span failing pass 1 is dropped "not_whole_word" and never
        # reaches the containment check at all.
        payload = "Billinger sent an invoice"
        spans = (
            TaggedSpan(start=0, end=9, text="Billinger", label="person", score=0.9),
            TaggedSpan(start=0, end=4, text="Bill", label="person", score=0.9),
        )
        _install_tag(monkeypatch, spans)

        results, _tag_result = scan_values(payload, categories=[PERSON])

        assert results[0].values == ("Billinger",)
        assert len(results[0].dropped) == 1
        assert results[0].dropped[0]["reason"] == "not_whole_word"
        assert results[0].dropped[0]["text"] == "Bill"

    def test_tag_result_span_count_is_unaffected_by_the_containment_drop(self, monkeypatch) -> None:
        full = "friedrich.ammerschlaeger@example.de"
        fragment = "ammerschlaeger@example.de"
        payload = f"Email: {full} please"
        start_full = payload.index(full)
        end_full = start_full + len(full)
        start_frag = payload.index(fragment)
        end_frag = start_frag + len(fragment)

        spans = (
            TaggedSpan(start=start_full, end=end_full, text=full, label="email", score=0.9),
            TaggedSpan(start=start_frag, end=end_frag, text=fragment, label="email", score=0.9),
        )
        _install_tag(monkeypatch, spans, windows=3)

        _results, tag_result = scan_values(payload, categories=[EMAIL])

        # The containment drop only shapes the returned ScanResult; the
        # raw TagResult the tagger produced is returned unchanged.
        assert tag_result.spans == spans
        assert len(tag_result.spans) == 2
        assert tag_result.windows == 3


class TestScanValuesPropagatesTaggerUnavailable:
    def test_raise_from_tag_propagates_unchanged(self, monkeypatch) -> None:
        def _raising_tag(text, labels):
            raise TaggerUnavailable("no handle loaded")

        monkeypatch.setattr(span_tagger, "tag", _raising_tag)

        with pytest.raises(TaggerUnavailable):
            scan_values("payload", categories=[PERSON])


# ---------------------------------------------------------------------------
# _anchor_candidates
# ---------------------------------------------------------------------------


def _scan_result(category: ScrubCategory, values: tuple[str, ...]):
    from paramem.cloud.anonymize_steps import ScanResult

    return ScanResult(category=category, values=values, dropped=())


class TestAnchorCandidates:
    def test_returns_only_person_prefix_surfaces(self) -> None:
        scans = (
            _scan_result(PERSON, ("Alex",)),
            _scan_result(PHONE, ("555-1234",)),
        )
        spans = (
            TaggedSpan(start=15, end=19, text="Alex", label="person", score=0.9),
            TaggedSpan(start=25, end=33, text="555-1234", label="phone number", score=0.9),
        )
        candidates = _anchor_candidates(scans, spans, anchor_range=(10, 50), person_prefix="Person")
        assert candidates == ("Alex",)

    def test_only_spans_inside_anchor_range_are_candidates(self) -> None:
        scans = (_scan_result(PERSON, ("Sam", "Alex")),)
        spans = (
            TaggedSpan(start=0, end=3, text="Sam", label="person", score=0.9),  # before range
            TaggedSpan(start=15, end=19, text="Alex", label="person", score=0.9),  # inside range
        )
        candidates = _anchor_candidates(scans, spans, anchor_range=(10, 50), person_prefix="Person")
        assert candidates == ("Alex",)

    def test_first_appearance_order_is_preserved(self) -> None:
        scans = (_scan_result(PERSON, ("Blair", "Alex")),)
        spans = (
            TaggedSpan(start=30, end=35, text="Blair", label="person", score=0.9),
            TaggedSpan(start=15, end=19, text="Alex", label="person", score=0.9),
        )
        candidates = _anchor_candidates(scans, spans, anchor_range=(10, 50), person_prefix="Person")
        assert candidates == ("Blair", "Alex")

    def test_empty_when_the_only_person_spans_are_in_history(self) -> None:
        scans = (_scan_result(PERSON, ("Sam",)),)
        spans = (TaggedSpan(start=0, end=3, text="Sam", label="person", score=0.9),)
        candidates = _anchor_candidates(scans, spans, anchor_range=(10, 50), person_prefix="Person")
        assert candidates == ()


class TestAskSpeakerAnchorRendersNonAsciiValuesVerbatim:
    """The ANCHOR user prompt embeds *values* via
    ``json.dumps(list(values), ensure_ascii=False)`` — a non-ASCII
    candidate surface (e.g. a German name) must reach the model as the
    real character, never a ``\\uXXXX`` escape the model would have to
    echo back byte-for-byte to survive the answer's own
    ``v in values_set`` membership check.
    """

    def test_a_non_ascii_candidate_is_not_escaped_in_the_rendered_prompt(self, monkeypatch) -> None:
        captured: dict = {}

        def _fake_render_chat_prompt(messages, tokenizer, add_generation_prompt=True):
            captured["messages"] = messages
            return "irrelevant rendered text"

        monkeypatch.setattr(anonymize_steps_module, "render_chat_prompt", _fake_render_chat_prompt)

        ask_speaker_anchor(
            "text",
            MagicMock(),
            MagicMock(),
            values=["Straße"],
            speaker_id="speaker1",
            section="speaker={speaker_id} values={values} text={text}",
            system_prompt="system",
            # Too small to fit any real budget -> AnonymizeBudgetRefused
            # after render, so no generate() call is issued — the render
            # step (and the assertion below) still runs unconditionally
            # before the budget check.
            token_envelope=1,
        )

        user_content = captured["messages"][1]["content"]
        assert "Straße" in user_content
        assert "\\u00df" not in user_content
