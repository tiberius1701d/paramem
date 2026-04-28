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
        # First-person + speaker_id — first_person_personal fires
        # (correct: the query is about the speaker).
        assert "first_person_personal" in findings

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


class TestFirstPersonResolution:
    """First-person pronouns resolve against the identified speaker.

    The interrogative-vs-declarative split that used to live here was
    removed once ``Intent`` + ``_is_interrogative`` in inference.py
    took over as the routing signals; the sanitizer now emits a single
    ``first_person_personal`` finding for both shapes.
    """

    def test_question_with_speaker_flags_first_person_personal(self):
        findings = check_personal_content(
            "Where do I live?",
            speaker_id="Speaker0",
        )
        assert "first_person_personal" in findings

    def test_statement_with_speaker_flags_first_person_personal(self):
        findings = check_personal_content(
            "I live in Kelkham.",
            speaker_id="Speaker0",
        )
        assert "first_person_personal" in findings

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
        assert "first_person_personal" in findings


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
        assert "first_person_personal" in findings

    def test_mode_block_returns_none_on_first_person(self):
        query, findings = sanitize_for_cloud(
            "Where do I live?",
            mode="block",
            speaker_id="Speaker0",
        )
        assert query is None
        assert "first_person_personal" in findings

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


# ---------------------------------------------------------------------------
# SanitizationConfig — cloud_mode validator + YAML loader wiring
# ---------------------------------------------------------------------------


class TestSanitizationConfigCloudMode:
    """The cloud_mode field is the egress-policy knob added in Architecture #3.

    These tests pin the dataclass surface (defaults, validator) and the
    load_server_config wiring; the actual behavior change (anonymize-and-send,
    block-PERSONAL, etc.) is tested in tests of inference.py once the
    cloud_anonymizer module is wired.
    """

    def test_default_is_block(self):
        from paramem.server.config import SanitizationConfig

        cfg = SanitizationConfig()
        assert cfg.cloud_mode == "block"

    def test_anonymize_value_accepted(self):
        from paramem.server.config import SanitizationConfig

        cfg = SanitizationConfig(cloud_mode="anonymize")
        assert cfg.cloud_mode == "anonymize"

    def test_both_value_accepted(self):
        from paramem.server.config import SanitizationConfig

        cfg = SanitizationConfig(cloud_mode="both")
        assert cfg.cloud_mode == "both"

    def test_invalid_value_rejected(self):
        import pytest

        from paramem.server.config import SanitizationConfig

        with pytest.raises(ValueError, match="cloud_mode"):
            SanitizationConfig(cloud_mode="not_a_real_mode")

    def test_mode_validator_still_fires(self):
        # Regression guard: adding the cloud_mode validator must not break
        # the existing mode validator.
        import pytest

        from paramem.server.config import SanitizationConfig

        with pytest.raises(ValueError, match="sanitization mode"):
            SanitizationConfig(mode="bogus")

    def test_loaded_from_yaml(self, tmp_path):
        """load_server_config wires sanitization.cloud_mode through SanitizationConfig(**raw)."""
        from paramem.server.config import load_server_config

        yaml_file = tmp_path / "server.yaml"
        yaml_file.write_text(
            "sanitization:\n  mode: block\n  cloud_mode: anonymize\n", encoding="utf-8"
        )
        config = load_server_config(yaml_file)
        assert config.sanitization.cloud_mode == "anonymize"

    def test_yaml_omits_cloud_mode_falls_back_to_default(self, tmp_path):
        """Existing server.yaml files that don't carry cloud_mode get the safe default."""
        from paramem.server.config import load_server_config

        yaml_file = tmp_path / "server.yaml"
        yaml_file.write_text("sanitization:\n  mode: warn\n", encoding="utf-8")
        config = load_server_config(yaml_file)
        assert config.sanitization.mode == "warn"
        assert config.sanitization.cloud_mode == "block"  # dataclass default

    def test_project_server_yaml_loads_cleanly(self):
        """The shipped configs/server.yaml parses without validator errors."""
        from paramem.server.config import load_server_config

        config = load_server_config("configs/server.yaml")
        assert config.sanitization.cloud_mode in {"block", "anonymize", "both"}
