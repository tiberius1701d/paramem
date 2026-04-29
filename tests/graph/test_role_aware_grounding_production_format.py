"""Contract tests for _apply_grounding_gate with production-format transcripts.

SessionBuffer._format_turns renders user turns as "{speaker_name}: {text}"
and model turns as "{Role}: {text}" (e.g. "Assistant: {text}").  These tests
verify that the role-aware grounding gate correctly distinguishes user-supplied
facts from assistant-supplied facts when given that format.

No model loading; CPU-only.
"""

from paramem.graph.extractor import _apply_grounding_gate

SPEAKER = "Alex"


def _make_transcript(pairs: list[tuple[str, str]]) -> str:
    """Build a transcript in production format from (role, text) pairs."""
    lines = []
    for role, text in pairs:
        if role == "user":
            lines.append(f"{SPEAKER}: {text}")
        else:
            lines.append(f"Assistant: {text}")
    return "\n".join(lines)


class TestProductionFormatUserFacts:
    def test_user_fact_object_in_user_turn_is_kept(self):
        """A fact about the speaker whose object appears only in user turns is kept."""
        transcript = _make_transcript(
            [
                ("user", "I work at Acme in Portland"),
                ("assistant", "That is interesting"),
            ]
        )
        facts = [{"subject": SPEAKER, "predicate": "works at", "object": "Acme"}]
        known_names: set[str] = {SPEAKER}

        kept, dropped, _ = _apply_grounding_gate(
            facts,
            transcript,
            known_names,
            speaker_name=SPEAKER,
            mode="active",
        )

        assert len(kept) == 1
        assert kept[0]["object"] == "Acme"
        assert dropped == []

    def test_user_fact_object_only_in_assistant_turn_is_dropped(self):
        """A fact whose object appears ONLY in assistant output is dropped in active mode."""
        transcript = _make_transcript(
            [
                ("user", "Tell me about Barcelona"),
                ("assistant", "Barcelona is a city in Catalonia Spain"),
            ]
        )
        facts = [{"subject": SPEAKER, "predicate": "lives in", "object": "Catalonia"}]
        known_names: set[str] = {SPEAKER}

        kept, dropped, _ = _apply_grounding_gate(
            facts,
            transcript,
            known_names,
            speaker_name=SPEAKER,
            mode="active",
        )

        assert kept == []
        assert len(dropped) == 1
        assert dropped[0]["object"] == "Catalonia"

    def test_multiple_user_turns_all_objects_kept(self):
        """Objects from multiple user turns are all grounded correctly."""
        transcript = _make_transcript(
            [
                ("user", "I have a dog named Rex"),
                ("assistant", "Rex sounds fun"),
                ("user", "I drive a Subaru"),
                ("assistant", "Nice car"),
            ]
        )
        facts = [
            {"subject": SPEAKER, "predicate": "has pet", "object": "Rex"},
            {"subject": SPEAKER, "predicate": "drives", "object": "Subaru"},
        ]
        known_names: set[str] = {SPEAKER}

        kept, dropped, _ = _apply_grounding_gate(
            facts,
            transcript,
            known_names,
            speaker_name=SPEAKER,
            mode="active",
        )

        assert len(kept) == 2
        assert dropped == []

    def test_assistant_introduced_object_dropped_speaker_subject(self):
        """An object mentioned only by the assistant is dropped when speaker is subject."""
        transcript = _make_transcript(
            [
                ("user", "I visited Boulder"),
                ("assistant", "Boulder is near the mountains in Colorado"),
            ]
        )
        # "Colorado" was introduced by assistant, not by the speaker.
        facts = [{"subject": SPEAKER, "predicate": "visited", "object": "Colorado"}]
        known_names: set[str] = {SPEAKER}

        kept, dropped, _ = _apply_grounding_gate(
            facts,
            transcript,
            known_names,
            speaker_name=SPEAKER,
            mode="active",
        )

        assert kept == []
        assert len(dropped) == 1

    def test_non_speaker_subject_uses_full_transcript(self):
        """Facts whose subject is NOT the speaker use full-transcript grounding (role-blind)."""
        transcript = _make_transcript(
            [
                ("user", "Tell me about Globex Corporation"),
                ("assistant", "Globex Corporation is a major company founded in Springfield"),
            ]
        )
        # Subject is not the speaker; object grounding uses full transcript.
        facts = [
            {"subject": "Globex Corporation", "predicate": "founded in", "object": "Springfield"}
        ]
        known_names: set[str] = {SPEAKER}

        kept, dropped, _ = _apply_grounding_gate(
            facts,
            transcript,
            known_names,
            speaker_name=SPEAKER,
            mode="active",
        )

        assert len(kept) == 1
        assert dropped == []

    def test_known_name_in_object_bypasses_role_check(self):
        """Objects that are known real-names bypass the role-aware grounding check."""
        transcript = _make_transcript(
            [
                ("user", "I went with someone"),
                ("assistant", "Sounds good"),
            ]
        )
        # "Bob" is a known name from the anonymisation mapping.
        facts = [{"subject": SPEAKER, "predicate": "traveled with", "object": "Bob"}]
        known_names: set[str] = {SPEAKER, "Bob"}

        kept, dropped, _ = _apply_grounding_gate(
            facts,
            transcript,
            known_names,
            speaker_name=SPEAKER,
            mode="active",
        )

        assert len(kept) == 1
        assert dropped == []


class TestDiagnosticModeRegression:
    """Regression tests for the parser fix.

    Before the fix, _extract_user_spans returned empty for {speaker_name}:
    transcripts.  The role-aware gate then dropped every fact whose object
    appeared only in user turns — populating role_aware_would_drop with
    spurious entries.  These tests lock the post-fix behaviour: production
    facts that appear in user turns are NOT dropped and NOT flagged.
    """

    def test_diagnostic_mode_keeps_user_grounded_fact(self):
        transcript = _make_transcript(
            [
                ("user", "I like hiking in the mountains"),
                ("assistant", "The mountains are beautiful"),
            ]
        )
        # "hiking" appears only in user turn.
        facts = [{"subject": SPEAKER, "predicate": "likes", "object": "hiking"}]
        known_names: set[str] = {SPEAKER}

        kept, dropped, would_drop = _apply_grounding_gate(
            facts,
            transcript,
            known_names,
            speaker_name=SPEAKER,
            mode="diagnostic",
        )

        # Pre-fix bug: would_drop would have contained the fact (user_norm
        # was empty because the parser couldn't see "Alex:" prefixes).
        # Post-fix: user_norm contains "hiking", role-aware agrees with
        # role-blind, would_drop is empty.
        assert len(kept) == 1
        assert dropped == []
        assert would_drop == []

    def test_diagnostic_mode_flags_only_assistant_only_objects(self):
        # In diagnostic mode the gate keeps the role-blind result but populates
        # would_drop with facts the role-aware view would have dropped.  An
        # object that's in the full transcript (so role-blind keeps it) but
        # only in assistant lines (so role-aware would drop it) is the precise
        # signal we want.
        transcript = _make_transcript(
            [
                ("user", "Tell me about Barcelona"),
                ("assistant", "Barcelona is the capital of Catalonia in Spain"),
            ]
        )
        facts = [{"subject": SPEAKER, "predicate": "lives in", "object": "Catalonia"}]
        known_names: set[str] = {SPEAKER}

        kept, dropped, would_drop = _apply_grounding_gate(
            facts,
            transcript,
            known_names,
            speaker_name=SPEAKER,
            mode="diagnostic",
        )

        # role-blind: object grounded in full transcript → kept.
        # role-aware: object NOT in user spans → would-be-dropped.
        # diagnostic preserves role-blind kept and surfaces the would-drop.
        assert len(kept) == 1
        assert dropped == []
        assert len(would_drop) == 1
        assert would_drop[0]["object"] == "Catalonia"
