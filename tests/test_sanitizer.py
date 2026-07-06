"""Unit tests for the graph-anchored sanitizer.

The sanitizer detects personal content using ground truth that already
exists in the running system:

* ``known_entities`` — caller-supplied set of lowercased entity / speaker
  names.  Assembled by ``handle_chat`` from ``memory_store.iter_entries()``
  subject/object fields plus the resolved speaker display name (M3 — plumbed
  directly from the ``speaker`` argument so the real name is covered even when
  it is no longer a registry subject under id-as-subject extraction).
  Reuses the extraction pipeline's ``_anonymize_transcript`` primitive to
  decide whether the query references any of them.
* ``speaker_id`` + first-person pronouns from a fixed token set — covers
  cold-start before the graph has facts.

There are no static keyword lists, no regex patterns.  A query like
"What time does my dentist's office open?" is not personal unless the
speaker actually has a "dentist" entity.  A query like "Did Pat call?"
is personal once Pat is enrolled or graphed.
"""

from pathlib import Path

import pytest

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
            speaker_id="speaker0",
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
# Paraphrase pass: catches plural / re-ordered / partial references to
# multi-word entities that the strict word-boundary scrub misses.
# ---------------------------------------------------------------------------


class TestPersonalEntityParaphraseDetection:
    """``personal_entity`` also fires when ≥2 content tokens of a multi-word
    entity name appear as substrings in the query — closes the gap that
    let 2026-05 paraphrased ADAS-platform queries reach the cloud."""

    def test_paraphrase_two_token_overlap_flags_personal(self):
        # Surface-form scrub bug: personal_entity check
        # misses because the indexed name reorders + adds modifier
        # ("multi-OEM") that the user query omits.
        findings = check_personal_content(
            "Tell me about the ADAS compute platforms project",
            known_entities={"critical multi-OEM ADAS platform turnaround"},
        )
        assert "personal_entity" in findings

    def test_pluralisation_single_token_does_not_fire(self):
        # Only ONE entity content token in the query ("platform") — the
        # 2-token floor must hold to avoid generic-word false positives.
        findings = check_personal_content(
            "What is a platform-as-a-service?",
            known_entities={"platform engineering team"},
        )
        assert findings == []

    def test_two_token_overlap_via_different_words_flags(self):
        # Reordered, no modifier overlap — but two content tokens match.
        findings = check_personal_content(
            "When is the team building event?",
            known_entities={"annual team building Q3"},  # Q3 < 3 chars → dropped
        )
        assert "personal_entity" in findings

    def test_single_token_entity_relies_on_surface_form(self):
        # "alice" alone — only one content token, so paraphrase pass is
        # skipped.  Surface-form word-boundary still catches direct
        # mentions of "Alice".
        findings_direct = check_personal_content(
            "Did Alice call?",
            known_entities={"alice"},
        )
        assert "personal_entity" in findings_direct
        # And substring false positive still rejected: "patron" ≠ "pat"
        # (handled by surface-form word-boundary).
        findings_fp = check_personal_content(
            "I want a patron saint.",
            known_entities={"pat"},
        )
        assert "personal_entity" not in findings_fp

    def test_generic_stopwords_do_not_count_as_content_tokens(self):
        # Entity name "the system and the platform" — after stopword +
        # length filter, content tokens are {system, platform}.  Query
        # "the and for" hits no content tokens → no fire.
        findings = check_personal_content(
            "the and for with",
            known_entities={"the system and the platform"},
        )
        assert findings == []

    def test_paraphrase_does_not_double_fire(self):
        # Surface-form already caught the entity by exact match — the
        # paraphrase pass must not add a second copy of the finding.
        findings = check_personal_content(
            "tell me about ADAS platform",
            known_entities={"adas platform"},
        )
        assert findings.count("personal_entity") == 1

    def test_empty_known_entities_paraphrase_pass_noop(self):
        findings = check_personal_content(
            "Tell me about the ADAS compute platforms project",
            known_entities=set(),
        )
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
            speaker_id="speaker0",
        )
        assert "first_person_personal" in findings

    def test_statement_with_speaker_flags_first_person_personal(self):
        findings = check_personal_content(
            "I live in Kelkham.",
            speaker_id="speaker0",
        )
        assert "first_person_personal" in findings

    def test_first_person_without_speaker_is_clean(self):
        # No identified speaker → no resolution target for "I" → clean.
        findings = check_personal_content("Where do I live?")
        assert findings == []

    def test_no_first_person_no_finding(self):
        findings = check_personal_content(
            "What's the capital of France?",
            speaker_id="speaker0",
        )
        assert findings == []

    def test_encoder_path_overrides_token_set_for_german(self):
        """Encoder-based classification fires when ``personal_referent_config``
        is provided; classifies German first-person queries that the
        legacy English token-set would miss.

        Closes the multilingual sanitizer gap demonstrated by the
        live probe: ``"Wo wohne ich?"`` was passing through unsanitized
        because ``_contains_first_person`` is English-only.
        """
        from unittest.mock import patch

        from paramem.server.config import PersonalReferentConfig
        from paramem.server.personal_referent import PersonalReferent

        cfg = PersonalReferentConfig()
        with patch(
            "paramem.server.personal_referent.classify_personal_referent",
            return_value=PersonalReferent.ABOUT_SPEAKER,
        ):
            findings = check_personal_content(
                "Wo wohne ich?",
                speaker_id="speaker0",
                personal_referent_config=cfg,
            )
        assert "first_person_personal" in findings

    def test_encoder_returning_not_about_speaker_clears_finding(self):
        """Encoder verdict NOT_ABOUT_SPEAKER suppresses the finding even
        when the English token-set heuristic would fire.  (The encoder
        recognises that the surface "I" doesn't refer to the speaker.)
        """
        from unittest.mock import patch

        from paramem.server.config import PersonalReferentConfig
        from paramem.server.personal_referent import PersonalReferent

        cfg = PersonalReferentConfig()
        with patch(
            "paramem.server.personal_referent.classify_personal_referent",
            return_value=PersonalReferent.NOT_ABOUT_SPEAKER,
        ):
            findings = check_personal_content(
                "I read that the Eiffel Tower was built in 1889.",
                speaker_id="speaker0",
                personal_referent_config=cfg,
            )
        assert "first_person_personal" not in findings

    def test_encoder_uncertain_falls_back_to_token_set(self):
        """Encoder returning ``None`` (margin not met / not loaded) falls
        through to the English token-set check.  Confirms the
        encoderless fallback path works as designed.
        """
        from unittest.mock import patch

        from paramem.server.config import PersonalReferentConfig

        cfg = PersonalReferentConfig()
        with patch(
            "paramem.server.personal_referent.classify_personal_referent",
            return_value=None,
        ):
            findings = check_personal_content(
                "Where do I live?",
                speaker_id="speaker0",
                personal_referent_config=cfg,
            )
        assert "first_person_personal" in findings

    def test_first_person_anywhere_in_text_matches(self):
        # "my" appears mid-sentence, not first word.
        findings = check_personal_content(
            "Tell me what's on my schedule today.",
            speaker_id="speaker0",
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
            speaker_id="speaker0",
        )
        assert query == "Where do I live?"
        assert findings == []

    def test_mode_warn_passes_with_findings(self):
        query, findings = sanitize_for_cloud(
            "Where do I live?",
            mode="warn",
            speaker_id="speaker0",
        )
        assert query == "Where do I live?"
        assert "first_person_personal" in findings

    def test_mode_block_returns_none_on_first_person(self):
        query, findings = sanitize_for_cloud(
            "Where do I live?",
            mode="block",
            speaker_id="speaker0",
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
            speaker_id="speaker0",
            known_entities={"pat"},
        )
        assert query == "What's the weather today?"
        assert findings == []

    def test_clean_query_passes_all_modes(self):
        for mode in ("off", "warn", "block"):
            query, findings = sanitize_for_cloud(
                "Turn on the kitchen light",
                mode=mode,
                speaker_id="speaker0",
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

    @pytest.mark.skipif(
        not Path("configs/server.yaml").exists(),
        reason="operator-local configs/server.yaml absent (CI / fresh clone)",
    )
    def test_project_server_yaml_loads_cleanly(self):
        """The shipped configs/server.yaml parses without validator errors."""
        from paramem.server.config import load_server_config

        config = load_server_config("configs/server.yaml")
        assert config.sanitization.cloud_mode in {"block", "anonymize", "both"}


# ---------------------------------------------------------------------------
# M3 — speaker display-name coverage in known_entities (handle_chat plumbing)
# ---------------------------------------------------------------------------


class TestM3SpeakerDisplayNameCoverage:
    """M3 coverage: the speaker's display name must be flagged as a personal
    referent even when it is not a registry subject.

    Under the id-as-subject refactor (step 8+) the registry holds ``speaker0``
    as the subject rather than ``"Tobias"``.  inference.py::handle_chat now
    explicitly adds the resolved ``speaker`` display name to ``known_entities``
    so the sanitizer still catches queries that mention the real name.

    These tests verify the sanitizer side of the contract: passing the display
    name in ``known_entities`` flags it as personal.  The inference.py plumbing
    (which adds ``speaker`` to ``known_entities``) is covered by the integration
    test below.
    """

    def test_display_name_in_known_entities_flags_personal(self):
        """Querying the speaker by their display name is caught as personal."""
        findings = check_personal_content(
            "What does Tobias do for work?",
            known_entities={"tobias"},
        )
        assert "personal_entity" in findings

    def test_display_name_absent_from_known_entities_is_clean(self):
        """Without the display name in known_entities, a name query is not personal.

        This is the regression case for M3: if the display name falls out of
        known_entities the sanitizer would let personal queries through.
        """
        findings = check_personal_content(
            "What does Tobias do for work?",
            known_entities=set(),
        )
        assert "personal_entity" not in findings

    def test_handle_chat_adds_speaker_to_known_entities(self):
        """handle_chat plumbs the speaker display name into known_entities.

        This unit test verifies the inference.py M3 fix without invoking the
        GPU model.  We patch the downstream sanitize_for_cloud call to capture
        the known_entities argument and confirm it contains the speaker name.
        """
        from unittest.mock import MagicMock, patch

        from paramem.server.inference import handle_chat

        captured: dict = {}

        def _fake_sanitize(text, mode="warn", *, speaker_id=None, known_entities=None, **kw):
            captured["known_entities"] = known_entities
            return text, []

        mock_config = MagicMock()
        mock_config.sanitization.mode = "warn"
        mock_config.personal_referent = None
        mock_config.debug = False
        mock_config.voice.load_prompt.return_value = "base"
        mock_config.sentence_type = None
        mock_config.abstention = MagicMock()
        mock_config.abstention.enabled = False

        mock_router = MagicMock()
        mock_plan = MagicMock()
        mock_plan.intent.value = "GENERAL"
        from paramem.server.router import Intent

        mock_plan.intent = Intent.GENERAL
        mock_plan.steps = []
        mock_router.route.return_value = mock_plan

        mock_model = MagicMock()
        mock_model.gradient_checkpointing_disable = MagicMock()
        mock_tokenizer = MagicMock()

        with (
            patch("paramem.server.inference.sanitize_for_cloud", side_effect=_fake_sanitize),
            patch(
                "paramem.server.inference._base_model_answer",
                return_value=MagicMock(text="ok", escalated=False, probed_keys=[]),
            ),
        ):
            handle_chat(
                text="What does Tobias do for work?",
                conversation_id="conv1",
                speaker="Tobias",
                speaker_id="speaker0",
                history=None,
                model=mock_model,
                tokenizer=mock_tokenizer,
                config=mock_config,
                router=mock_router,
            )

        assert "known_entities" in captured
        assert captured["known_entities"] is not None
        assert "tobias" in captured["known_entities"]

    def test_handle_chat_anonymous_speaker_not_added_to_known_entities(self):
        """Anonymous display-name suppression: None speaker must not pollute known_entities.

        When resolve_speaker_name returns None (anonymous profile), handle_chat
        receives speaker=None and must not add anything to known_entities.
        """
        from unittest.mock import MagicMock, patch

        from paramem.server.inference import handle_chat

        captured: dict = {}

        def _fake_sanitize(text, mode="warn", *, speaker_id=None, known_entities=None, **kw):
            captured["known_entities"] = known_entities
            return text, []

        mock_config = MagicMock()
        mock_config.sanitization.mode = "warn"
        mock_config.personal_referent = None
        mock_config.debug = False
        mock_config.voice.load_prompt.return_value = "base"
        mock_config.sentence_type = None
        mock_config.abstention = MagicMock()
        mock_config.abstention.enabled = False

        mock_router = MagicMock()
        mock_plan = MagicMock()
        from paramem.server.router import Intent

        mock_plan.intent = Intent.GENERAL
        mock_plan.steps = []
        mock_router.route.return_value = mock_plan

        mock_model = MagicMock()
        mock_model.gradient_checkpointing_disable = MagicMock()
        mock_tokenizer = MagicMock()

        with (
            patch("paramem.server.inference.sanitize_for_cloud", side_effect=_fake_sanitize),
            patch(
                "paramem.server.inference._base_model_answer",
                return_value=MagicMock(text="ok", escalated=False, probed_keys=[]),
            ),
        ):
            handle_chat(
                text="Hello there.",
                conversation_id="conv2",
                speaker=None,  # anonymous / not resolved
                speaker_id="speaker0",
                history=None,
                model=mock_model,
                tokenizer=mock_tokenizer,
                config=mock_config,
                router=mock_router,
            )

        # known_entities may be None (no memory_store) or an empty set —
        # but must NOT contain a non-None speaker name since speaker is None.
        ke = captured.get("known_entities")
        if ke is not None:
            assert not any(v for v in ke if v)  # no truthy entity from anonymous
