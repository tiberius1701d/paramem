"""Direct tests for the text-parsing helpers in ``tests/_prompt_text_parsing.py``.

The prompt-contract and prompt-render suites use these helpers to assert
the ABSENCE of something (no stray placeholder, no off-schema value). A
helper that silently finds nothing would let those suites pass for the
wrong reason. Each test here proves the corresponding helper finds what it
must on a small, hand-built inline text, independent of any real prompt
file.
"""

from __future__ import annotations

from tests._prompt_text_parsing import (
    double_quoted_spans,
    example_json_objects,
    has_first_person_pronoun,
    has_glued_possessive_object,
    has_non_speaker_alpha_subject,
    is_ascii_letter_led,
    is_capitalized_word,
    is_proper_name,
    json_objects,
    leading_rule_id,
    split_blocks,
    stray_placeholders,
    subject_values,
)


class TestSplitBlocks:
    def test_splits_on_one_or_more_blank_lines(self):
        text = "first line\nsecond line\n\n\nthird block\n"
        assert split_blocks(text) == ["first line\nsecond line", "third block"]

    def test_no_blank_line_is_a_single_block(self):
        assert split_blocks("only one block\nstill one block") == [
            "only one block\nstill one block"
        ]


class TestJsonObjects:
    def test_finds_a_well_formed_object(self):
        text = 'Example: {"subject": "Ada", "object": "Bill"}'
        assert json_objects(text) == [{"subject": "Ada", "object": "Bill"}]

    def test_doubled_brace_literal_yields_only_the_inner_object(self):
        text = 'BAD output: {{"subject": "Ada", ...}}'
        # The outer doubled-brace span fails to parse and is discarded; the
        # inner span is itself incomplete (trailing `...`) so nothing at
        # all should be recovered from a bare elision with no real object.
        assert json_objects(text) == []

    def test_elided_example_still_yields_a_parseable_inner_object(self):
        text = 'BAD output: {{"subject":"Person_1", "object":"x"}, ...}}'
        # The inner single-brace object is complete and well-formed even
        # though the doubled outer span around it is not.
        assert json_objects(text) == [{"subject": "Person_1", "object": "x"}]


class TestSubjectValues:
    def test_finds_a_plain_subject_value(self):
        text = '{"subject": "Ada Lovelace", "object": "Bill"}'
        assert subject_values(text) == {"Ada Lovelace"}

    def test_elided_example_still_yields_its_subject(self):
        # The enclosing object as a whole is not valid JSON (trailing
        # elision), but the "subject" value sitting before it is.
        text = '{{"subject":"Person_1", ...}}'
        assert subject_values(text) == {"Person_1"}


class TestIsAsciiLetterLed:
    def test_true_for_a_letter_led_value(self):
        assert is_ascii_letter_led("Ada")

    def test_false_for_empty_or_non_letter_led(self):
        assert not is_ascii_letter_led("")
        assert not is_ascii_letter_led("1Ada")


class TestHasNonSpeakerAlphaSubject:
    def test_true_for_a_letter_led_subject_other_than_speaker0(self):
        text = '{"subject": "Ada", "object": "x"}'
        assert has_non_speaker_alpha_subject(text)

    def test_false_when_only_subject_is_speaker0(self):
        text = '{"subject": "speaker0", "object": "x"}'
        assert not has_non_speaker_alpha_subject(text)


class TestHasGluedPossessiveObject:
    def test_true_for_a_glued_possessive(self):
        text = '{"subject": "Theo", "object": "Theo\'s orchids"}'
        assert has_glued_possessive_object(text)

    def test_false_with_no_possessive(self):
        text = '{"subject": "Theo", "object": "orchids"}'
        assert not has_glued_possessive_object(text)


class TestIsCapitalizedWord:
    def test_true_for_a_capitalized_ascii_word(self):
        assert is_capitalized_word("Lovelace")

    def test_false_for_lowercase_or_non_letters(self):
        assert not is_capitalized_word("lovelace")
        assert not is_capitalized_word("Ada2")


class TestIsProperName:
    def test_true_for_space_separated_capitalized_words(self):
        assert is_proper_name("Ada Lovelace")

    def test_false_when_any_word_is_not_capitalized(self):
        assert not is_proper_name("Ada lovelace")


class TestDoubleQuotedSpans:
    def test_pairs_quoted_spans_left_to_right(self):
        text = '"Ada" said "hi" to "Bill"'
        assert double_quoted_spans(text) == ["Ada", "hi", "Bill"]

    def test_no_quotes_yields_no_spans(self):
        assert double_quoted_spans("no quotes here") == []


class TestHasFirstPersonPronoun:
    def test_true_for_a_whole_word_pronoun_case_insensitive(self):
        assert has_first_person_pronoun("My favorite color is blue")

    def test_false_when_pronoun_is_only_a_substring(self):
        # "my" must match as a whole word, not as a substring of "mystery".
        assert not has_first_person_pronoun("It's a total mystery, not solved")


class TestExampleJsonObjects:
    def test_parses_the_object_after_an_example_line(self):
        text = 'Example: {"subject": "Ada", "object": "Bill"}\nOther text'
        assert example_json_objects(text) == [{"subject": "Ada", "object": "Bill"}]

    def test_ignores_lines_not_starting_with_example(self):
        text = 'Not an example: {"subject": "Ada"}'
        assert example_json_objects(text) == []


class TestLeadingRuleId:
    def test_finds_a_rule_id_prefix(self):
        assert leading_rule_id("R4. Do not invent facts.") == "R4"

    def test_none_when_no_rule_id_prefix(self):
        assert leading_rule_id("Rule 4: do not invent facts.") is None


class TestStrayPlaceholders:
    def test_finds_every_stray_span_sorted(self):
        # Occurrence order in the text is {stray_two} then {stray_one};
        # the helper returns them sorted, not in occurrence order.
        text = "first {stray_two} then second {stray_one}"
        assert stray_placeholders(text) == ["{stray_one}", "{stray_two}"]

    def test_placeholder_nested_inside_a_json_example_is_reported_as_stray(self):
        text = '{"subject": "{speaker_name}", "object": "x"}'
        assert stray_placeholders(text) == ["{speaker_name}"]

    def test_doubled_json_braces_are_not_reported(self):
        text = 'Empty result: {{"mapping": {{}}}}'
        assert stray_placeholders(text) == []

    def test_declared_intentional_literal_is_excluded(self):
        text = "The slot renders as {SPEAKER_NAME} in the example."
        assert stray_placeholders(text, intentional_literals=frozenset({"{SPEAKER_NAME}"})) == []
