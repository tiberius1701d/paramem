"""Unit tests for the graph-anchored sanitizer.

The sanitizer detects personal content using ground truth that already
exists in the running system:

* ``known_entities`` — caller-supplied set of entity / speaker names
  (typically ``router._all_entities | speaker_store.speaker_names()``).
  Reuses the extraction pipeline's ``_anonymize_transcript`` primitive to
  decide whether the query references any of them.
* ``speaker_id`` + first-person pronouns from a fixed token set — covers
  cold-start before the graph has facts.

There are no static keyword lists, no regex patterns.  A query like
"What time does my dentist's office open?" is not personal unless the
speaker actually has a "dentist" entity.  A query like "Did Pat call?"
is personal once Pat is enrolled or graphed.
"""

from paramem.server.sanitizer import check_personal_content, sanitize_for_cloud

# ---------------------------------------------------------------------------
# Graph-anchored personal-entity detection
# ---------------------------------------------------------------------------


class TestPersonalEntityDetection:
    """``personal_entity`` fires when the query mentions a known entity."""

    def test_named_entity_in_known_set_flags_personal(self):
        findings = check_personal_content(
            "Did Pat call?",
            known_entities={"pat"},
        )
        assert "personal_entity" in findings

    def test_named_entity_not_in_known_set_is_clean(self):
        # Without graph state, a generic name is not personal.
        findings = check_personal_content(
            "Did Pat call?",
            known_entities=set(),
        )
        assert findings == []

    def test_relationship_noun_alone_is_not_personal(self):
        # The old regex blocked anything mentioning "wife" / "dentist" /
        # "house" / "favourite". Graph-anchored: only blocked if the
        # speaker actually has such an entity in their graph.
        findings = check_personal_content(
            "What time does my dentist's office open?",
            known_entities={"dr_smith"},  # the dentist isn't a known entity
            speaker_id="Speaker0",
        )
        assert "personal_entity" not in findings
        # First-person + speaker_id + interrogative — the self_referential
        # arm fires (which is correct: it's about the speaker).
        assert "self_referential" in findings

    def test_word_boundary_prevents_substring_false_positives(self):
        # "Pat" must not match inside "patron" — _anonymize_transcript
        # uses \b...\b boundaries.
        findings = check_personal_content(
            "Where is the nearest patron saint?",
            known_entities={"pat"},
        )
        assert findings == []

    def test_no_known_entities_supplied_returns_clean(self):
        # Back-compat: callers that don't yet pass known_entities and
        # also don't pass speaker_id get an empty findings list.  The
        # primary protection is then entity-based routing upstream.
        findings = check_personal_content("Did Pat call?")
        assert findings == []


# ---------------------------------------------------------------------------
# First-person + speaker_id resolution
# ---------------------------------------------------------------------------


class TestSelfReference:
    """First-person pronouns resolve against the identified speaker."""

    def test_self_referential_question_with_speaker(self):
        findings = check_personal_content(
            "Where do I live?",
            speaker_id="Speaker0",
        )
        assert "self_referential" in findings

    def test_personal_claim_statement_with_speaker(self):
        findings = check_personal_content(
            "I live in Kelkham.",
            speaker_id="Speaker0",
        )
        assert "personal_claim" in findings
        assert "self_referential" not in findings

    def test_first_person_without_speaker_is_clean(self):
        # No identified speaker → no resolution target for "I" → clean.
        findings = check_personal_content("Where do I live?")
        assert findings == []

    def test_no_first_person_no_finding(self):
        findings = check_personal_content(
            "What's the capital of France?",
            speaker_id="Speaker0",
        )
        assert findings == []

    def test_first_person_anywhere_in_text_matches(self):
        # "my" appears mid-sentence, not first word.
        findings = check_personal_content(
            "Tell me what's on my schedule today.",
            speaker_id="Speaker0",
        )
        assert "self_referential" in findings or "personal_claim" in findings


# ---------------------------------------------------------------------------
# sanitize_for_cloud — mode behaviour and contract preservation
# ---------------------------------------------------------------------------


class TestSanitizeForCloud:
    def test_mode_off_passes_everything(self):
        query, findings = sanitize_for_cloud(
            "Where do I live?",
            mode="off",
            speaker_id="Speaker0",
        )
        assert query == "Where do I live?"
        assert findings == []

    def test_mode_warn_passes_with_findings(self):
        query, findings = sanitize_for_cloud(
            "Where do I live?",
            mode="warn",
            speaker_id="Speaker0",
        )
        assert query == "Where do I live?"
        assert "self_referential" in findings

    def test_mode_block_returns_none_on_self_reference(self):
        query, findings = sanitize_for_cloud(
            "Where do I live?",
            mode="block",
            speaker_id="Speaker0",
        )
        assert query is None
        assert "self_referential" in findings

    def test_mode_block_returns_none_on_known_entity(self):
        query, findings = sanitize_for_cloud(
            "Did Pat call?",
            mode="block",
            known_entities={"pat"},
        )
        assert query is None
        assert "personal_entity" in findings

    def test_mode_block_passes_clean_query(self):
        query, findings = sanitize_for_cloud(
            "What's the weather today?",
            mode="block",
            speaker_id="Speaker0",
            known_entities={"pat"},
        )
        assert query == "What's the weather today?"
        assert findings == []

    def test_clean_query_passes_all_modes(self):
        for mode in ("off", "warn", "block"):
            query, findings = sanitize_for_cloud(
                "Turn on the kitchen light",
                mode=mode,
                speaker_id="Speaker0",
                known_entities={"pat"},
            )
            assert query == "Turn on the kitchen light"
            assert findings == []
