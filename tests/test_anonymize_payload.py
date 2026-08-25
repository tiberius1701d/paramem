"""``_assemble_payload`` — the one assembler producing both ``TagPayload``
derivations (``tag_text`` for the tagger, ``anchor_evidence`` for the
ANCHOR call).
"""

from __future__ import annotations

from paramem.cloud.anonymize import _assemble_payload


class TestTagTextIsMarkerFreeAndOrdered:
    def test_tag_text_orders_history_then_transcript_then_fact_lines(self) -> None:
        history = ["[user] earlier turn", "[assistant] earlier reply"]
        transcript = "[user] current turn"
        facts = [{"subject": "alex", "predicate": "lives in", "object": "berlin"}]

        payload = _assemble_payload(history, transcript, facts)

        lines = payload.tag_text.split("\n")
        assert lines[0] == "earlier turn"
        assert lines[1] == "earlier reply"
        assert lines[2] == "current turn"
        # One line per fact: subject and object verbatim, space-joined —
        # the same two fields insert_placeholders later substitutes.
        # predicate is deliberately excluded (never a substitution target).
        assert lines[3] == "alex berlin"

    def test_every_marker_is_stripped_from_tag_text(self) -> None:
        payload = _assemble_payload(["[user] hi"], "[assistant] hello there", facts=[])
        assert "[user]" not in payload.tag_text
        assert "[assistant]" not in payload.tag_text

    def test_the_marker_stripped_word_user_never_survives_into_tag_text(self) -> None:
        # The historical false-positive: "[user" tagged as a person span
        # when the marker is left in. Stripping it means the bare word
        # "user" is never produced by the marker itself.
        payload = _assemble_payload([], "[user] hello", facts=[])
        assert "user" not in payload.tag_text


class TestAnchorRangeSelectsExactlyTheTranscriptRegion:
    def test_anchor_range_slice_equals_the_stripped_transcript_text(self) -> None:
        history = ["[user] earlier"]
        transcript = "[user] current turn text"
        payload = _assemble_payload(history, transcript, facts=[])

        start, end = payload.anchor_range
        assert payload.tag_text[start:end] == "current turn text"

    def test_empty_transcript_gives_an_empty_anchor_range(self) -> None:
        payload = _assemble_payload(["[user] hi"], "", facts=[])
        start, end = payload.anchor_range
        assert start == end


class TestAnchorEvidenceIsMarkerBearingAndExcludesFacts:
    def test_anchor_evidence_is_the_exact_join_of_history_and_transcript_lines(self) -> None:
        history = ["[user] earlier turn", "[assistant] earlier reply"]
        transcript = "[user] current turn"
        facts = [{"subject": "alex", "predicate": "lives in", "object": "berlin"}]

        payload = _assemble_payload(history, transcript, facts)

        assert payload.anchor_evidence == "\n".join([*history, transcript])

    def test_anchor_evidence_carries_markers_verbatim(self) -> None:
        payload = _assemble_payload(["[user] hi"], "[assistant] hello", facts=[])
        assert "[user] hi" in payload.anchor_evidence
        assert "[assistant] hello" in payload.anchor_evidence

    def test_anchor_evidence_excludes_the_fact_block(self) -> None:
        facts = [{"subject": "alex", "predicate": "lives in", "object": "berlin"}]
        payload = _assemble_payload([], "[user] hi", facts)
        assert "berlin" not in payload.anchor_evidence
        assert "subject" not in payload.anchor_evidence

    def test_empty_history_makes_anchor_evidence_equal_the_transcript(self) -> None:
        transcript = "[user] just this turn"
        payload = _assemble_payload([], transcript, facts=[])
        assert payload.anchor_evidence == transcript


class TestFactBlockPreservesNonAsciiCharactersVerbatim:
    """The tagger reads exactly the strings the consumer substitutes: a
    non-ASCII fact value (e.g. ``ß``) must appear verbatim in ``tag_text``
    — a rendering that escapes it to ``\\uXXXX`` would make the tagger tag
    a surface that never equals the real value at substitution time.
    """

    def test_a_non_ascii_fact_value_appears_verbatim_in_tag_text(self) -> None:
        facts = [
            {
                "subject": "speaker0",
                "predicate": "lives at",
                "object": "Lindenstraße 44, 10115 Berlin",
            }
        ]
        payload = _assemble_payload([], "", facts)
        assert "Lindenstraße 44, 10115 Berlin" in payload.tag_text
        assert "\\u00df" not in payload.tag_text


class TestFactBlockPreservesQuotesAndBackslashesVerbatim:
    """The same "tagger reads exactly the consumer's substitution string"
    invariant applies to a literal ``"`` or ``\\`` inside a fact value: a
    rendering that escapes either (a JSON string literal escapes both)
    produces a tagged surface that never equals the raw value
    :func:`~paramem.cloud.placeholders.insert_placeholders` later
    substitutes against.
    """

    def test_a_double_quote_in_a_fact_value_survives_verbatim_in_tag_text(self) -> None:
        facts = [{"subject": "speaker0", "predicate": "said", "object": 'the "best" cafe'}]
        payload = _assemble_payload([], "", facts)
        assert 'the "best" cafe' in payload.tag_text

    def test_a_backslash_in_a_fact_value_survives_verbatim_in_tag_text(self) -> None:
        facts = [{"subject": "speaker0", "predicate": "saved path", "object": "C:\\Users\\alex"}]
        payload = _assemble_payload([], "", facts)
        assert "C:\\Users\\alex" in payload.tag_text


class TestFactBlockIsBuiltFromRawSubjectAndObjectStrings:
    """The fact block carries each fact's own raw ``subject``/``object``
    strings — asserted by presence in ``tag_text``, never by a specific
    rendering format (JSON, plain lines, or otherwise), since the
    rendering choice is an internal representation detail the tagger and
    the consumer never need to agree on beyond "the raw string appears".

    ``predicate`` is deliberately never rendered into the fact block
    (:func:`~paramem.cloud.anonymize._render_fact_lines`'s own docstring):
    it is never a substitution target for
    :func:`~paramem.cloud.placeholders.insert_placeholders`, so tagging it
    would only inflate ``scan_dropped``/``inert_dropped`` for no
    substitution benefit — this class asserts that omission explicitly
    rather than assuming subject/object presence implies predicate
    presence too.
    """

    def test_every_facts_subject_and_object_string_is_present_in_tag_text(self) -> None:
        facts = [{"subject": "Alex Rivera", "predicate": "lives at", "object": "Lindenstraße 44"}]
        payload = _assemble_payload([], "", facts)
        assert "Alex Rivera" in payload.tag_text
        assert "Lindenstraße 44" in payload.tag_text

    def test_multiple_facts_subject_and_object_strings_are_all_present_in_tag_text(self) -> None:
        facts = [
            {"subject": "Alex Rivera", "predicate": "knows", "object": "Jamie Lee"},
            {"subject": "Jamie Lee", "predicate": "works at", "object": "Northwind Traders"},
        ]
        payload = _assemble_payload([], "", facts)
        for fact in facts:
            assert fact["subject"] in payload.tag_text
            assert fact["object"] in payload.tag_text

    def test_the_predicate_is_never_rendered_into_the_fact_block(self) -> None:
        facts = [
            {"subject": "Alex Rivera", "predicate": "has_unique_marker_predicate", "object": "x"}
        ]
        payload = _assemble_payload([], "", facts)
        assert "has_unique_marker_predicate" not in payload.tag_text
