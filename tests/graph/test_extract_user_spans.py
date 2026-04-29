"""Unit tests for _extract_user_spans — role-aware transcript parser."""

from paramem.graph.extractor import _extract_user_spans


class TestDefaultFormatUserBracket:
    def test_default_format_user_bracket(self):
        transcript = "[user] I visited Boulder last summer\n[assistant] That sounds great"
        result = _extract_user_spans(transcript)
        assert "I visited Boulder last summer" in result
        assert "That sounds great" not in result


class TestDefaultFormatUserColon:
    def test_default_format_user_colon_lowercase(self):
        transcript = "user: I work at CERN\nassistant: Interesting!"
        result = _extract_user_spans(transcript)
        assert "I work at CERN" in result
        assert "Interesting!" not in result

    def test_default_format_user_colon_mixed_case(self):
        transcript = "User: Hello there\nAssistant: Hi"
        result = _extract_user_spans(transcript)
        assert "Hello there" in result
        assert "Hi" not in result


class TestSpeakerPrefixMatchesWhenProvided:
    def test_speaker_prefix_matches_when_provided(self):
        transcript = "Alex: I visited Boulder\nAssistant: Sounds wonderful"
        result = _extract_user_spans(transcript, speaker_name="Alex")
        assert "I visited Boulder" in result
        assert "Sounds wonderful" not in result

    def test_speaker_prefix_strips_colon_and_leading_space(self):
        transcript = "Alex:   content with leading spaces"
        result = _extract_user_spans(transcript, speaker_name="Alex")
        assert result.strip() == "content with leading spaces"


class TestSpeakerPrefixCaseInsensitive:
    def test_speaker_prefix_case_insensitive_lowercase_line(self):
        transcript = "alex: my dog is named Rex\nAssistant: Nice name"
        result = _extract_user_spans(transcript, speaker_name="Alex")
        assert "my dog is named Rex" in result
        assert "Nice name" not in result

    def test_speaker_prefix_case_insensitive_uppercase_line(self):
        transcript = "ALEX: I love hiking\nAssistant: Great hobby"
        result = _extract_user_spans(transcript, speaker_name="Alex")
        assert "I love hiking" in result


class TestSpeakerPrefixIgnoredWhenNoSpeakerName:
    def test_speaker_prefix_ignored_when_no_speaker_name(self):
        transcript = "Alex: I visited Boulder\nAssistant: Sounds great"
        result = _extract_user_spans(transcript)
        assert result == ""

    def test_speaker_prefix_ignored_when_speaker_name_none(self):
        transcript = "Alex: My birthday is in March"
        result = _extract_user_spans(transcript, speaker_name=None)
        assert result == ""


class TestSpeakerPrefixDoesNotMatchOtherNames:
    def test_speaker_prefix_does_not_match_other_names(self):
        transcript = "Bob: I visited Phoenix\nAlex: I visited Boulder"
        result = _extract_user_spans(transcript, speaker_name="Alex")
        assert "I visited Phoenix" not in result
        assert "I visited Boulder" in result

    def test_partial_name_match_does_not_fire(self):
        transcript = "AlexX: this should not match\nAlex: this should match"
        result = _extract_user_spans(transcript, speaker_name="Alex")
        assert "this should not match" not in result
        assert "this should match" in result


class TestAssistantLinesNeverMatch:
    def test_assistant_bracket_prefix_never_matches(self):
        transcript = "[assistant] I know about Boulder"
        result = _extract_user_spans(transcript, speaker_name="Assistant")
        assert result == ""

    def test_assistant_colon_prefix_never_matches(self):
        # speaker_name="Assistant" must not collapse user/assistant role
        # separation — the `assistant:` prefix is exclusively a model marker.
        transcript = "assistant: The capital of France is Paris"
        result = _extract_user_spans(transcript, speaker_name="Assistant")
        assert result == ""

    def test_assistant_capitalized_colon_never_matches(self):
        transcript = "Assistant: Your appointment is tomorrow"
        result = _extract_user_spans(transcript, speaker_name="Assistant")
        assert result == ""


class TestMixedFormatSession:
    def test_mixed_format_session_only_speaker_lines_extracted(self):
        transcript = (
            "Alex: I live in Portland\n"
            "Assistant: Portland is a lovely city in Oregon\n"
            "this is a continuation line without a prefix\n"
            "Alex: My cat is named Whiskers\n"
            "Assistant: Whiskers is a great name"
        )
        result = _extract_user_spans(transcript, speaker_name="Alex")
        assert "I live in Portland" in result
        assert "My cat is named Whiskers" in result
        assert "Portland is a lovely city" not in result
        assert "continuation line" not in result
        assert "Whiskers is a great name" not in result

    def test_mixed_format_session_multi_line_output_order(self):
        transcript = "Alex: first\nAssistant: ignored\nAlex: second"
        result = _extract_user_spans(transcript, speaker_name="Alex")
        lines = result.splitlines()
        assert lines == ["first", "second"]


class TestLegacyAndSpeakerPrefixCoexist:
    def test_legacy_and_speaker_prefix_in_same_transcript(self):
        # Legacy [user] / user: must continue to win, and the speaker_prefix
        # branch contributes additionally — order in output mirrors line order.
        transcript = (
            "[user] from bracket form\n"
            "user: from colon form\n"
            "Alex: from speaker-prefix form\n"
            "Assistant: ignored"
        )
        result = _extract_user_spans(transcript, speaker_name="Alex")
        lines = result.splitlines()
        assert lines == [
            "from bracket form",
            "from colon form",
            "from speaker-prefix form",
        ]

    def test_speaker_named_user_does_not_double_extract(self):
        # If the operator's speaker_name happens to be "user", the legacy
        # `user:` branch fires first; the speaker_prefix branch never sees
        # the line because it's already consumed.  No double-emit.
        transcript = "user: only once please"
        result = _extract_user_spans(transcript, speaker_name="user")
        assert result.splitlines() == ["only once please"]
