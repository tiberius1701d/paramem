"""``assemble_payload`` — the SCAN payload's turn slicing and the four
derivations (``tag_text``, ``slice_ranges``, ``anchor_range``,
``anchor_evidence``) it produces from one set of inputs.
"""

from __future__ import annotations

from paramem.cloud.anonymize import assemble_payload

# ---------------------------------------------------------------------------
# The slices.
# ---------------------------------------------------------------------------


class TestHistorySlicing:
    def test_each_history_turn_is_its_own_slice(self) -> None:
        payload = assemble_payload(["[user] first turn", "[assistant] second turn"], "", [])
        assert len(payload.slice_ranges) == 2
        texts = [payload.tag_text[s:e] for s, e in payload.slice_ranges]
        assert texts == ["first turn", "second turn"]


class TestTranscriptSlicing:
    def test_a_multi_line_turn_is_one_slice(self) -> None:
        transcript = "[user] line one\nline two\nline three"
        payload = assemble_payload([], transcript, [])
        assert len(payload.slice_ranges) == 1
        (start, end) = payload.slice_ranges[0]
        assert payload.tag_text[start:end] == "line one\nline two\nline three"

    def test_each_marker_line_opens_a_new_slice(self) -> None:
        transcript = "[user] hi there\n[assistant] hello back"
        payload = assemble_payload([], transcript, [])
        assert len(payload.slice_ranges) == 2
        texts = [payload.tag_text[s:e] for s, e in payload.slice_ranges]
        assert texts == ["hi there", "hello back"]

    def test_a_continuation_line_shaped_like_a_marker_opens_its_own_slice(self) -> None:
        """A continuation line that itself begins with a bracketed word
        followed by a space is indistinguishable, by the marker regex, from
        a real turn marker — it opens its own slice. This is the accepted
        behaviour, not a defect: the marker grammar has no way to tell
        "[bracketed]" prose apart from a role marker.
        """
        transcript = "[user] hello\n[bracketed] like a marker\nplain continuation"
        payload = assemble_payload([], transcript, [])
        assert len(payload.slice_ranges) == 2
        texts = [payload.tag_text[s:e] for s, e in payload.slice_ranges]
        assert texts == ["hello", "like a marker\nplain continuation"]


class TestFactBlockSlicing:
    def test_the_fact_block_is_one_slice_for_any_number_of_facts(self) -> None:
        facts = [
            {"subject": "Alex", "predicate": "works_at", "object": "Acme"},
            {"subject": "Alex", "predicate": "lives_in", "object": "Berlin"},
            {"subject": "Riley", "predicate": "knows", "object": "Alex"},
        ]
        payload = assemble_payload([], "", facts)
        assert len(payload.slice_ranges) == 1
        (start, end) = payload.slice_ranges[0]
        assert payload.tag_text[start:end] == "Alex Acme\nAlex Berlin\nRiley Alex"

    def test_no_facts_contributes_no_slice(self) -> None:
        payload = assemble_payload(["[user] hi"], "", [])
        assert len(payload.slice_ranges) == 1  # only the history turn
        texts = [payload.tag_text[s:e] for s, e in payload.slice_ranges]
        assert texts == ["hi"]


class TestPayloadShapesPerCaller:
    def test_chat_egress_payload_is_one_turn_no_history_no_facts(self) -> None:
        """No history, one transcript turn, no facts -> exactly one slice,
        costing exactly one SCAN call."""
        payload = assemble_payload([], "[user] hello there", [])
        assert len(payload.slice_ranges) == 1
        (start, end) = payload.slice_ranges[0]
        assert payload.tag_text[start:end] == "hello there"

    def test_graph_tier_payload_is_facts_only_no_transcript(self) -> None:
        """No transcript, facts only -> exactly one slice."""
        facts = [{"subject": "Alex", "predicate": "works_at", "object": "Acme"}]
        payload = assemble_payload([], "", facts)
        assert len(payload.slice_ranges) == 1
        (start, end) = payload.slice_ranges[0]
        assert payload.tag_text[start:end] == "Alex Acme"


# ---------------------------------------------------------------------------
# What the payload keeps.
# ---------------------------------------------------------------------------


class TestTagTextIsMarkerFreeAndOrdered:
    def test_history_then_transcript_then_facts_marker_free(self) -> None:
        history = ["[user] HISTORY_SENTINEL turn"]
        transcript = "[assistant] TRANSCRIPT_SENTINEL turn"
        facts = [{"subject": "FACT_SENTINEL_SUBJ", "predicate": "p", "object": "FACT_SENTINEL_OBJ"}]

        payload = assemble_payload(history, transcript, facts)

        assert "[user]" not in payload.tag_text
        assert "[assistant]" not in payload.tag_text
        assert (
            payload.tag_text.index("HISTORY_SENTINEL")
            < payload.tag_text.index("TRANSCRIPT_SENTINEL")
            < payload.tag_text.index("FACT_SENTINEL_SUBJ")
        )


class TestAnchorRangeSelectsExactlyTheTranscript:
    def test_anchor_range_covers_only_the_stripped_transcript(self) -> None:
        history = ["[user] history turn"]
        transcript = "[user] transcript line one\ntranscript line two"
        facts = [{"subject": "S", "predicate": "p", "object": "O"}]

        payload = assemble_payload(history, transcript, facts)

        start, end = payload.anchor_range
        assert payload.tag_text[start:end] == "transcript line one\ntranscript line two"
        # Neither the history nor the fact block leaks into the region.
        assert "history turn" not in payload.tag_text[start:end]
        assert "S O" not in payload.tag_text[start:end]


class TestAnchorEvidenceIsMarkerBearingWithNoFactBlock:
    def test_anchor_evidence_joins_history_and_transcript_lines_verbatim(self) -> None:
        history = ["[user] history turn"]
        transcript = "[user] transcript line one\n[assistant] transcript line two"
        facts = [{"subject": "FACT_SUBJ", "predicate": "p", "object": "FACT_OBJ"}]

        payload = assemble_payload(history, transcript, facts)

        assert payload.anchor_evidence == (
            "[user] history turn\n[user] transcript line one\n[assistant] transcript line two"
        )
        assert "FACT_SUBJ" not in payload.anchor_evidence
        assert "FACT_OBJ" not in payload.anchor_evidence

    def test_anchor_evidence_is_empty_facts_agnostic_when_no_facts(self) -> None:
        history = ["[user] hi"]
        transcript = "[assistant] hello"
        payload = assemble_payload(history, transcript, [])
        assert payload.anchor_evidence == "[user] hi\n[assistant] hello"


class TestFactValuesReachThePayloadVerbatim:
    def test_subject_and_object_reach_tag_text_verbatim_including_special_characters(self) -> None:
        subject = 'Café "Chéz" \\René\\'
        obj = "Müller & Söhne"
        facts = [{"subject": subject, "predicate": "works_at", "object": obj}]

        payload = assemble_payload([], "", facts)

        assert subject in payload.tag_text
        assert obj in payload.tag_text
        # No JSON quoting/escaping of the special characters.
        assert '\\"' not in payload.tag_text
        assert "\\\\" not in payload.tag_text

    def test_predicate_never_appears_in_the_payload(self) -> None:
        facts = [{"subject": "A", "predicate": "UNIQUE_PREDICATE_TOKEN", "object": "B"}]
        payload = assemble_payload([], "", facts)
        assert "UNIQUE_PREDICATE_TOKEN" not in payload.tag_text
