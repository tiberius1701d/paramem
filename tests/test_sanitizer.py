"""Unit tests for the self-reference sanitizer gate.

``is_self_referential`` decides whether a piece of text refers to / asks
about the speaker themselves.  Detection is two-tier: an encoder-based
classifier when ``personal_referent_config`` is supplied, falling back to
an explicit first-person token set.  There is no static keyword list for
personal *content* — that arm was removed; self-reference is the only
signal this module contributes to the ``is_personal`` verdict.

Re-spec (speakerless-serving boundary decision): the predicate is
content-only now — it takes no ``speaker_id`` and has no null-target gate.
The former ``bool(speaker_id) and ...`` gate is deleted; a caller decides
separately whether a resolved speaker exists to apply the verdict to (see
``paramem.server.inference.handle_chat``'s speaker-present contract).
"""

from pathlib import Path

import pytest

from paramem.server.sanitizer import is_self_referential

# ---------------------------------------------------------------------------
# First-person resolution — content-only, no speaker_id parameter
# ---------------------------------------------------------------------------


class TestFirstPersonResolution:
    """First-person pronouns are classified from content alone.

    The interrogative-vs-declarative split that used to live here was
    removed once ``Intent`` + ``_is_interrogative`` in inference.py
    took over as the routing signals; the sanitizer now emits a single
    boolean verdict for both shapes.
    """

    def test_question_is_self_referential(self):
        assert is_self_referential("Where do I live?") is True

    def test_statement_is_self_referential(self):
        assert is_self_referential("I live in Kelkham.") is True

    def test_first_person_is_self_referential_with_no_speaker_known(self):
        # Content-only: classification does not depend on — and cannot be
        # passed — a speaker_id.  A speakerless caller's personal-shaped
        # text still classifies as self-referential in content; whether
        # anything downstream ACTS on that verdict is a separate decision
        # made by the caller (e.g. the relay path's identity_absent gate).
        assert is_self_referential("Where do I live?") is True

    def test_no_first_person_is_false(self):
        assert is_self_referential("What's the capital of France?") is False

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
            result = is_self_referential("Wo wohne ich?", personal_referent_config=cfg)
        assert result is True

    def test_encoder_returning_not_about_speaker_clears_verdict(self):
        """Encoder verdict NOT_ABOUT_SPEAKER suppresses the verdict even
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
            result = is_self_referential(
                "I read that the Eiffel Tower was built in 1889.",
                personal_referent_config=cfg,
            )
        assert result is False

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
            result = is_self_referential("Where do I live?", personal_referent_config=cfg)
        assert result is True

    def test_first_person_anywhere_in_text_matches(self):
        # "my" appears mid-sentence, not first word.
        assert is_self_referential("Tell me what's on my schedule today.") is True


# ---------------------------------------------------------------------------
# is_self_referential is the predicate — there is no policy knob
# ---------------------------------------------------------------------------


class TestSelfReferentialPredicateIsUnconditional:
    """``sanitization.mode`` (off/warn/block) is deleted.

    Detection always runs and always reports; the caller owns what to do
    about a personal verdict.
    """

    def test_first_person_is_personal(self):
        assert is_self_referential("Where do I live?") is True

    def test_clean_query_is_not_personal(self):
        assert is_self_referential("What's the weather today?") is False

    def test_imperative_without_first_person_is_not_personal(self):
        assert is_self_referential("Turn on the kitchen light") is False

    def test_no_mode_parameter_survives(self):
        """Regression guard: the deleted knob must not come back as a kwarg."""
        import inspect

        assert "mode" not in inspect.signature(is_self_referential).parameters

    def test_no_known_entities_parameter_survives(self):
        """Regression guard: the deleted known-entity arm must not come back."""
        import inspect

        assert "known_entities" not in inspect.signature(is_self_referential).parameters

    def test_no_speaker_id_parameter_survives(self):
        """Regression guard: the deleted null-target gate must not come
        back as a kwarg — classification is content-only now."""
        import inspect

        assert "speaker_id" not in inspect.signature(is_self_referential).parameters


# ---------------------------------------------------------------------------
# SanitizationConfig — cloud_mode validator + YAML loader wiring
# ---------------------------------------------------------------------------


class TestSanitizationConfigCloudMode:
    """``cloud_mode`` is the surviving egress-policy knob.

    These tests pin the dataclass surface (defaults, validator) and the
    load_server_config wiring; the behaviour it selects (anonymize-and-send,
    block-PERSONAL) is tested against ``answer_via_cloud``.
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

    def test_deleted_mode_field_is_rejected_by_name(self):
        """``sanitization.mode`` is gone — a pinned value must fail loudly.

        Removed keys raise by name at config load; that is the desired
        failure mode, not a silent ignore.
        """
        import pytest

        from paramem.server.config import SanitizationConfig

        with pytest.raises(TypeError, match="mode"):
            SanitizationConfig(mode="block")

    def test_loaded_from_yaml(self, tmp_path):
        """load_server_config wires sanitization.cloud_mode through SanitizationConfig(**raw)."""
        from paramem.server.config import load_server_config

        yaml_file = tmp_path / "server.yaml"
        yaml_file.write_text("sanitization:\n  cloud_mode: anonymize\n", encoding="utf-8")
        config = load_server_config(yaml_file)
        assert config.sanitization.cloud_mode == "anonymize"

    def test_yaml_omits_cloud_mode_falls_back_to_default(self, tmp_path):
        """A server.yaml that doesn't carry cloud_mode gets the safe default."""
        from paramem.server.config import load_server_config

        yaml_file = tmp_path / "server.yaml"
        yaml_file.write_text("sanitization:\n  scrub: [person name]\n", encoding="utf-8")
        config = load_server_config(yaml_file)
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
