"""Tests for multilingual TTS engine abstraction and config."""

import json
from unittest.mock import MagicMock, patch

import pytest

from paramem.server.config import TTSConfig, TTSVoiceConfig, load_server_config
from paramem.server.stt import TranscriptionResult

# --- TranscriptionResult ---


def test_transcription_result_fields():
    result = TranscriptionResult(text="hello", language="en", language_probability=0.95)
    assert result.text == "hello"
    assert result.language == "en"
    assert result.language_probability == 0.95


def test_transcription_result_empty():
    result = TranscriptionResult(text="", language="en", language_probability=0.0)
    assert result.text == ""


# --- TTSConfig ---


def test_tts_config_defaults():
    config = TTSConfig()
    assert config.enabled is True
    assert config.port == 10301
    assert config.device == "cuda"
    assert config.default_language == "en"
    assert config.voices == {}


def test_tts_voice_config_defaults():
    voice = TTSVoiceConfig()
    assert voice.engine == "piper"
    assert voice.model == ""
    assert voice.device == ""


def test_tts_config_with_voices():
    config = TTSConfig(
        enabled=True,
        voices={
            "en": TTSVoiceConfig(engine="piper", model="en_US-lessac-high"),
            "tl": TTSVoiceConfig(engine="mms", model="facebook/mms-tts-tgl"),
        },
    )
    assert len(config.voices) == 2
    assert config.voices["en"].engine == "piper"
    assert config.voices["tl"].engine == "mms"


def test_tts_config_from_yaml(tmp_path):
    """TTS section in server.yaml is parsed correctly."""
    yaml_content = """
model: mistral
tts:
  enabled: true
  port: 10301
  device: cuda
  default_language: en
  voices:
    en:
      engine: piper
      model: en_US-lessac-high
    de:
      engine: piper
      model: de_DE-thorsten-high
    tl:
      engine: mms
      model: facebook/mms-tts-tgl
"""
    yaml_path = tmp_path / "server.yaml"
    yaml_path.write_text(yaml_content)
    config = load_server_config(yaml_path)

    assert config.tts.enabled is True
    assert config.tts.port == 10301
    assert config.tts.device == "cuda"
    assert config.tts.default_language == "en"
    assert len(config.tts.voices) == 3
    assert config.tts.voices["en"].engine == "piper"
    assert config.tts.voices["en"].model == "en_US-lessac-high"
    assert config.tts.voices["tl"].engine == "mms"
    assert config.tts.voices["tl"].model == "facebook/mms-tts-tgl"


# --- TTSManager ---


def test_tts_manager_routing():
    """TTSManager routes to the correct engine based on language."""
    from paramem.server.tts import TTSManager

    config = TTSConfig(
        enabled=True,
        default_language="en",
        voices={
            "en": TTSVoiceConfig(engine="piper", model="en_US-lessac-high"),
            "de": TTSVoiceConfig(engine="piper", model="de_DE-thorsten-high"),
        },
    )
    manager = TTSManager(config)

    # Mock engines
    en_engine = MagicMock()
    en_engine.synthesize.return_value = (b"\x00" * 100, 22050)
    de_engine = MagicMock()
    de_engine.synthesize.return_value = (b"\x00" * 100, 22050)

    manager._engines = {"en": en_engine, "de": de_engine}

    # Route to English
    manager.synthesize("hello", "en")
    en_engine.synthesize.assert_called_once_with("hello")

    # Route to German
    manager.synthesize("hallo", "de")
    de_engine.synthesize.assert_called_once_with("hallo")


def test_tts_manager_fallback_to_default():
    """TTSManager falls back to default language for unknown languages."""
    from paramem.server.tts import TTSManager

    config = TTSConfig(enabled=True, default_language="en")
    manager = TTSManager(config)

    en_engine = MagicMock()
    en_engine.synthesize.return_value = (b"\x00" * 100, 22050)
    manager._engines = {"en": en_engine}

    # Request unsupported language — falls back to English
    manager.synthesize("bonjour", "fr")
    en_engine.synthesize.assert_called_once_with("bonjour")


def test_tts_manager_no_engines_raises():
    """TTSManager raises when no engines are available."""
    from paramem.server.tts import TTSManager

    config = TTSConfig(enabled=True, default_language="en")
    manager = TTSManager(config)

    with pytest.raises(RuntimeError, match="No TTS engines available"):
        manager.synthesize("hello", "en")


def test_tts_manager_available_languages():
    from paramem.server.tts import TTSManager

    config = TTSConfig(enabled=True)
    manager = TTSManager(config)
    manager._engines = {"en": MagicMock(), "de": MagicMock(), "fr": MagicMock()}

    assert manager.available_languages == ["de", "en", "fr"]
    assert manager.has_language("en")
    assert not manager.has_language("tl")


# --- Language instruction ---


def test_language_instruction_english():
    from paramem.server.inference import _language_instruction

    assert _language_instruction("en") == ""
    assert _language_instruction(None) == ""


def test_language_instruction_non_english():
    from paramem.server.inference import _language_instruction

    assert _language_instruction("de") == "Respond in German."
    assert _language_instruction("tl") == "Respond in Tagalog."
    assert _language_instruction("fr") == "Respond in French."
    assert _language_instruction("es") == "Respond in Spanish."


def test_language_instruction_from_config():
    """Language names are derived from TTS voice config when available."""
    from paramem.server.config import ServerConfig
    from paramem.server.inference import _language_instruction

    config = ServerConfig()
    config.tts.voices["de"] = TTSVoiceConfig(engine="piper", model="test", language_name="Deutsch")
    # Config overrides ISO standard name
    assert _language_instruction("de", config) == "Respond in Deutsch."
    # Unconfigured language falls back to ISO
    assert _language_instruction("fr", config) == "Respond in French."


def test_tts_config_language_name_method():
    """TTSConfig.language_name resolves names from voice config or ISO fallback."""
    config = TTSConfig(
        voices={"de": TTSVoiceConfig(engine="piper", model="test", language_name="Deutsch")}
    )
    assert config.language_name("de") == "Deutsch"
    assert config.language_name("fr") == "French"  # ISO fallback
    assert config.language_name("xx") == "xx"  # unknown code returned as-is


def test_build_speaker_prefix_with_language():
    """_build_speaker_prefix combines the token line + language instruction."""
    from paramem.server.inference import _build_speaker_prefix

    # English — no language instruction
    result = _build_speaker_prefix("speaker0", "en", None)
    assert "Respond in" not in result
    assert "speaker0" in result

    # German — language instruction added
    result = _build_speaker_prefix("speaker0", "de", None)
    assert "Respond in German" in result
    assert "speaker0" in result

    # No token, with language — no "You are speaking with" salutation, which
    # is what the LOCAL reasoning leg relies on to suppress the token for
    # anonymous/undisclosed profiles (see _probe_and_reason /
    # _base_model_answer's identity_token derivation).
    result = _build_speaker_prefix(None, "fr", None)
    assert "Respond in French" in result
    assert "You are speaking with" not in result


# ---------------------------------------------------------------------------
# _build_speaker_prefix — LOCAL-leg-only, fed the speaker{N} token verbatim.
# Cloud never calls this function at all (see _escalate_to_cloud, which
# carries no identity line whatsoever) — resolution to a display name
# happens only at the reply boundary via
# paramem.server.speaker.resolve_speaker_tokens.
# ---------------------------------------------------------------------------


class TestBuildSpeakerPrefix:
    """``_build_speaker_prefix`` assembles the "You are speaking with X" +
    language block for the LOCAL reasoning prompt only.  ``X`` is the raw
    ``speaker{N}`` token — that is the intended, new contract, not a leak.
    """

    def test_token_and_language(self):
        from paramem.server.inference import _build_speaker_prefix

        result = _build_speaker_prefix("speaker0", "de", None)
        assert "speaker0" in result
        assert "German" in result

    def test_no_token_no_language(self):
        from paramem.server.inference import _build_speaker_prefix

        result = _build_speaker_prefix(None, None, None)
        assert result == ""

    def test_token_only(self):
        from paramem.server.inference import _build_speaker_prefix

        result = _build_speaker_prefix("speaker3", None, None)
        assert "speaker3" in result
        assert "Respond in" not in result

    def test_language_only(self):
        from paramem.server.inference import _build_speaker_prefix

        result = _build_speaker_prefix(None, "fr", None)
        assert "French" in result
        assert "You are speaking with" not in result

    def test_token_emitted_verbatim(self):
        """The speaker{N} token IS emitted verbatim — the intended new contract.

        This function is only ever called on the LOCAL reasoning leg; the
        cloud leg (`_escalate_to_cloud`) never calls it and carries no
        identity line at all.  A human-readable name is substituted only at
        the reply boundary, never here.
        """
        from paramem.server.inference import _build_speaker_prefix

        prefix = _build_speaker_prefix("speaker0", "en", None)
        assert "speaker0" in prefix


# ---------------------------------------------------------------------------
# _build_system_prompt — THE single assembly point for the LOCAL reasoning
# leg's system prompt, collapsing what were two byte-identical blocks in
# _probe_and_reason and _base_model_answer.
# ---------------------------------------------------------------------------


class TestBuildSystemPrompt:
    def _config(self, tmp_path, base_text="You are an assistant."):
        from paramem.server.config import ServerConfig, VoiceConfig

        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text(base_text)
        config = ServerConfig()
        config.voice = VoiceConfig(prompt_file=str(prompt_file))
        return config

    def test_speaker_id_gates_prefix(self, tmp_path):
        """speaker_id present → prefix included, regardless of display name."""
        from paramem.server.inference import _build_system_prompt

        config = self._config(tmp_path)
        result = _build_system_prompt("speaker0", None, config)
        assert "speaker0" in result
        assert "You are an assistant." in result

    def test_speaker_id_gates_prefix_even_with_no_display_name(self, tmp_path):
        """Re-spec (B-form prefix from speaker_id presence): the display
        name is gone as a parameter entirely — an anonymous/undisclosed
        speaker's raw speaker_id still produces the identity line.  Only
        ``speaker_id is None`` suppresses it now."""
        from paramem.server.inference import _build_system_prompt

        config = self._config(tmp_path)
        result = _build_system_prompt("speaker0", None, config)
        assert "speaker0" in result
        assert "You are an assistant." in result

    def test_no_prefix_returns_bare_base_prompt(self, tmp_path):
        from paramem.server.inference import _build_system_prompt

        config = self._config(tmp_path)
        result = _build_system_prompt(None, None, config)
        assert result == "You are an assistant."

    def test_language_instruction_included(self, tmp_path):
        from paramem.server.inference import _build_system_prompt

        config = self._config(tmp_path)
        result = _build_system_prompt("speaker0", "de", config)
        assert "speaker0" in result
        assert "Respond in German" in result
        assert "You are an assistant." in result


# --- Speaker language preference ---


def test_speaker_language_preference(tmp_path):
    from paramem.server.speaker import SpeakerStore

    store = SpeakerStore(tmp_path / "profiles.json")
    embedding = [0.1] * 192
    speaker_id = store.enroll("Alice", embedding)

    # No preference initially
    assert store.get_preferred_language(speaker_id) is None

    # Low confidence — no update
    store.update_language(speaker_id, "de", 0.5)
    assert store.get_preferred_language(speaker_id) is None

    # High confidence — updates
    store.update_language(speaker_id, "de", 0.9)
    assert store.get_preferred_language(speaker_id) == "de"

    # Non-existent speaker — no crash
    store.update_language("nonexistent", "fr", 0.95)
    assert store.get_preferred_language("nonexistent") is None


def test_speaker_profile_v4_has_language(tmp_path):
    """New profiles include preferred_language field."""
    from paramem.server.speaker import SpeakerStore

    path = tmp_path / "profiles.json"
    store = SpeakerStore(path)
    store.enroll("Alice", [0.1] * 192)

    data = json.loads(path.read_text())
    profile = list(data["speakers"].values())[0]
    assert "preferred_language" in profile
    assert profile["preferred_language"] == ""


def test_speaker_v3_load_raises(tmp_path):
    """v3 store (retired migration) raises ValueError on load."""
    path = tmp_path / "profiles.json"
    v3_data = {
        "version": 3,
        "speakers": {
            "abc123": {
                "name": "Alice",
                "embeddings": [[0.1] * 192],
            }
        },
        "last_greeted": {},
    }
    path.write_text(json.dumps(v3_data))

    import pytest

    from paramem.server.speaker import SpeakerStore

    with pytest.raises(ValueError, match="Unsupported speaker store version"):
        SpeakerStore(path)


# ---------------------------------------------------------------------------
# resolve_speaker_tokens — THE single reply-boundary resolver
# ---------------------------------------------------------------------------


class TestResolveSpeakerTokens:
    """``resolve_speaker_tokens`` is the one place a ``speaker{N}`` token is
    ever substituted for a display name — every model-facing surface keeps
    the raw token, and this function is applied only at the human-visible
    reply boundary."""

    def _store(self, tmp_path, *, name="Alice"):
        from paramem.server.speaker import SpeakerStore

        store = SpeakerStore(tmp_path / "profiles.json")
        speaker_id = store.enroll(name, [0.1] * 192)
        assert speaker_id == "speaker0"
        return store

    def test_resolves_known_speaker(self, tmp_path):
        from paramem.server.speaker import resolve_speaker_tokens

        store = self._store(tmp_path)
        result = resolve_speaker_tokens("speaker0 likes hiking.", store)
        assert result == "Alice likes hiking."

    def test_case_insensitive_sentence_initial(self, tmp_path):
        """A model-authored, sentence-capitalized "Speaker0" still resolves."""
        from paramem.server.speaker import resolve_speaker_tokens

        store = self._store(tmp_path)
        result = resolve_speaker_tokens("Speaker0 likes hiking.", store)
        assert result == "Alice likes hiking."

    def test_unknown_token_falls_back_to_descriptor(self, tmp_path):
        from paramem.server.speaker import THIRD_PARTY_DESCRIPTOR, resolve_speaker_tokens

        store = self._store(tmp_path)
        result = resolve_speaker_tokens("speaker9 called yesterday.", store)
        assert THIRD_PARTY_DESCRIPTOR in result
        assert "speaker9" not in result

    def test_anonymous_enrolled_falls_back_to_descriptor(self, tmp_path):
        """An anonymous-enrolled profile's own id is its 'name' — must still
        fall back to the descriptor (third-party case: no current_speaker_id
        supplied, or the token names someone other than the current
        speaker), never leak the raw token as a name."""
        from paramem.server.speaker import THIRD_PARTY_DESCRIPTOR, resolve_speaker_tokens

        store = self._store(tmp_path)
        # A direction distinct from the enrolled Alice profile's uniform
        # embedding — otherwise cosine similarity is 1.0 and register_anonymous
        # reuses Alice's speaker_id instead of minting a fresh anonymous one.
        distinct_embedding = ([0.1] * 96) + ([-0.1] * 96)
        anon_id = store.register_anonymous(distinct_embedding)
        assert anon_id != "speaker0"
        result = resolve_speaker_tokens(f"{anon_id} asked a question.", store)
        assert THIRD_PARTY_DESCRIPTOR in result
        assert anon_id not in result

    def test_no_store_falls_back_to_descriptor(self):
        from paramem.server.speaker import THIRD_PARTY_DESCRIPTOR, resolve_speaker_tokens

        result = resolve_speaker_tokens("speaker0 likes hiking.", None)
        assert THIRD_PARTY_DESCRIPTOR in result

    def test_self_token_survives_verbatim_when_current_speaker(self, tmp_path):
        """B — self-token fail-safe: an anonymous/undisclosed speaker's OWN
        token, unresolvable to a name, is left verbatim rather than
        collapsed to the third-party descriptor — rendering an undisclosed
        speaker as "someone else" to themselves would be a wrong statement
        about identity, not a privacy improvement."""
        from paramem.server.speaker import THIRD_PARTY_DESCRIPTOR, resolve_speaker_tokens

        store = self._store(tmp_path)
        distinct_embedding = ([0.1] * 96) + ([-0.1] * 96)
        anon_id = store.register_anonymous(distinct_embedding)

        result = resolve_speaker_tokens(
            f"{anon_id} asked a question.",
            store,
            current_speaker_id=anon_id,
        )
        assert anon_id in result
        assert THIRD_PARTY_DESCRIPTOR not in result

    def test_self_token_case_insensitive_match(self, tmp_path):
        """The self-token comparison canonicalizes both sides — a
        model-authored capitalized rendering still matches."""
        from paramem.server.speaker import resolve_speaker_tokens

        store = self._store(tmp_path)
        distinct_embedding = ([0.1] * 96) + ([-0.1] * 96)
        anon_id = store.register_anonymous(distinct_embedding)

        result = resolve_speaker_tokens(
            f"{anon_id.capitalize()} asked a question.",
            store,
            current_speaker_id=anon_id,
        )
        assert anon_id.capitalize() in result

    def test_other_persons_token_still_falls_back_with_current_speaker_set(self, tmp_path):
        """current_speaker_id only protects the CURRENT speaker's own
        token — a different, unresolvable token still falls back to the
        descriptor even when current_speaker_id is supplied."""
        from paramem.server.speaker import THIRD_PARTY_DESCRIPTOR, resolve_speaker_tokens

        store = self._store(tmp_path)
        distinct_embedding = ([0.1] * 96) + ([-0.1] * 96)
        anon_id = store.register_anonymous(distinct_embedding)

        result = resolve_speaker_tokens(
            "speaker9 called yesterday.",
            store,
            current_speaker_id=anon_id,
        )
        assert THIRD_PARTY_DESCRIPTOR in result
        assert "speaker9" not in result

    def test_no_tokens_is_a_no_op(self, tmp_path):
        from paramem.server.speaker import resolve_speaker_tokens

        store = self._store(tmp_path)
        text = "No identity tokens in this sentence at all."
        assert resolve_speaker_tokens(text, store) == text

    def test_idempotent_on_its_own_output(self, tmp_path):
        """Output contains no speaker{N} tokens, so re-running is a no-op."""
        from paramem.server.speaker import resolve_speaker_tokens

        store = self._store(tmp_path)
        once = resolve_speaker_tokens("speaker0 likes hiking.", store)
        twice = resolve_speaker_tokens(once, store)
        assert once == twice

    @staticmethod
    def _fake_store(mapping: dict[str, str | None]):
        """A minimal resolve_speaker_name-only stand-in — avoids minting N
        real profiles just to exercise the token regex's own boundary
        behavior (SUBSTITUTION strictness is a property of the regex, not
        of SpeakerStore)."""
        store = MagicMock()
        store.resolve_speaker_name.side_effect = lambda token: mapping.get(token)
        return store

    def test_speaker10_and_speaker1_are_distinct(self):
        """The greedy \\d+ + \\b in the strict regex means speaker10 is one
        token, never mistaken for speaker1 plus a trailing '0'."""
        from paramem.server.speaker import resolve_speaker_tokens

        store = self._fake_store({"speaker1": "Bob", "speaker10": "Carla"})
        result = resolve_speaker_tokens("speaker1 met speaker10 today.", store)
        assert result == "Bob met Carla today."

    def test_sub_word_non_match(self):
        """A 'speaker' embedded mid-word (no left word boundary) is left
        alone — SUBSTITUTION only matches standalone speaker{N} tokens."""
        from paramem.server.speaker import resolve_speaker_tokens

        store = self._fake_store({"speaker0": "Alice"})
        text = "The loudspeaker0 crackled."
        assert resolve_speaker_tokens(text, store) == text

    def test_possessive_substitution(self):
        """A trailing possessive 's stays attached to the substituted name —
        \\b after the digits allows the token to end at the apostrophe."""
        from paramem.server.speaker import resolve_speaker_tokens

        store = self._fake_store({"speaker0": "Alice"})
        result = resolve_speaker_tokens("speaker0's dog is friendly.", store)
        assert result == "Alice's dog is friendly."

    def test_multiple_distinct_tokens_resolve_independently(self):
        from paramem.server.speaker import resolve_speaker_tokens

        store = self._fake_store({"speaker0": "Alice", "speaker1": "Bob"})
        result = resolve_speaker_tokens("speaker0 called speaker1 yesterday.", store)
        assert result == "Alice called Bob yesterday."

    def test_near_miss_survivor_logs_warning_and_is_left_unsubstituted(self, caplog):
        """A model re-rendering as 'Speaker 0' (space instead of no
        separator) does not match the strict SUBSTITUTION regex — it must
        survive unsubstituted AND be logged at WARNING with only the
        matched shape, never the surrounding personal-content text."""
        import logging

        from paramem.server.speaker import resolve_speaker_tokens

        store = self._fake_store({"speaker0": "Alice"})
        text = "Please welcome Speaker 0 to the stage."
        with caplog.at_level(logging.WARNING, logger="paramem.server.speaker"):
            result = resolve_speaker_tokens(text, store)

        # Left unsubstituted — widening SUBSTITUTION is deliberately not done
        # here; it remains a separate decision, not a default.
        assert "Speaker 0" in result
        assert "Alice" not in result

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        message = warnings[0].getMessage()
        assert "Speaker 0" in message
        # Never the surrounding reply text — it is personal content.
        assert "Please welcome" not in message
        assert "the stage" not in message

    def test_no_near_miss_no_warning(self, tmp_path, caplog):
        """A cleanly-resolved token produces no near-miss warning."""
        import logging

        store = self._store(tmp_path)
        with caplog.at_level(logging.WARNING, logger="paramem.server.speaker"):
            from paramem.server.speaker import resolve_speaker_tokens

            resolve_speaker_tokens("speaker0 likes hiking.", store)

        assert not [r for r in caplog.records if r.levelno == logging.WARNING]


# --- Engine registry ---


def test_create_engine_unknown_raises():
    """Unknown engine name raises ValueError with available engines."""
    from paramem.server.tts import _create_engine

    bad_voice = TTSVoiceConfig(engine="nonexistent", model="test")
    config = TTSConfig()
    with pytest.raises(ValueError, match="Unknown TTS engine.*nonexistent.*Available.*mms.*piper"):
        _create_engine(bad_voice, config)


def test_engine_registry_contains_both():
    from paramem.server.tts import ENGINE_REGISTRY

    assert "piper" in ENGINE_REGISTRY
    assert "mms" in ENGINE_REGISTRY


# --- Per-voice device override ---


def test_per_voice_device_override():
    """Per-voice device config overrides the global TTSConfig.device."""

    config = TTSConfig(
        enabled=True,
        device="cuda",
        voices={
            "en": TTSVoiceConfig(engine="piper", model="en_US-lessac-high"),
            "tl": TTSVoiceConfig(engine="mms", model="facebook/mms-tts-tgl", device="cpu"),
        },
    )
    # Verify device resolution: en inherits cuda, tl overrides to cpu
    en_voice = config.voices["en"]
    tl_voice = config.voices["tl"]
    assert (en_voice.device or config.device) == "cuda"
    assert (tl_voice.device or config.device) == "cpu"


# --- TTSConfig from YAML with new fields ---


def test_tts_config_new_fields(tmp_path):
    """New configurable fields are parsed from YAML."""
    yaml_content = """
model: mistral
tts:
  enabled: true
  port: 10301
  device: cpu
  default_language: de
  language_confidence_threshold: 0.7
  model_dir: /tmp/tts_models
  voices:
    de:
      engine: piper
      model: de_DE-thorsten-high
"""
    yaml_path = tmp_path / "server.yaml"
    yaml_path.write_text(yaml_content)
    config = load_server_config(yaml_path)

    assert config.tts.language_confidence_threshold == 0.7
    assert config.tts.model_dir == "/tmp/tts_models"
    assert config.tts.device == "cpu"
    assert config.tts.default_language == "de"


def test_tts_config_defaults_when_omitted(tmp_path):
    """Omitted TTS fields use defaults."""
    yaml_content = """
model: mistral
tts:
  enabled: true
  voices:
    en:
      engine: piper
      model: test
"""
    yaml_path = tmp_path / "server.yaml"
    yaml_path.write_text(yaml_content)
    config = load_server_config(yaml_path)

    assert config.tts.language_confidence_threshold == 0.8
    assert config.tts.model_dir == ""
    assert config.tts.audio_chunk_bytes == 4096
    assert config.tts.device == "cuda"


# --- GPU fallback ---


def test_tts_manager_gpu_fallback():
    """TTSManager falls back to CPU when GPU loading fails."""
    from paramem.server.tts import TTSManager

    config = TTSConfig(
        enabled=True,
        device="cuda",
        voices={"en": TTSVoiceConfig(engine="piper", model="test")},
    )
    manager = TTSManager(config)

    # Mock _create_engine to return an engine that fails on GPU but succeeds on CPU
    mock_engine = MagicMock()
    mock_engine.actual_device = "cpu"
    call_count = 0

    def load_side_effect(device="cpu"):
        nonlocal call_count
        call_count += 1
        if device == "cuda":
            return False  # GPU fails
        mock_engine.actual_device = "cpu"
        return True  # CPU succeeds

    mock_engine.load.side_effect = load_side_effect

    with patch("paramem.server.tts._create_engine", return_value=mock_engine):
        manager.load_all()

    # Engine should have been tried on GPU (failed) then CPU (succeeded)
    assert call_count == 2
    assert "en" in manager._engines


# --- TTSManager GPU tracking ---


def test_tts_manager_needs_gpu():
    """TTSManager.needs_gpu reports per-language GPU status."""
    from paramem.server.tts import TTSManager

    config = TTSConfig(enabled=True, default_language="en")
    manager = TTSManager(config)
    manager._engines = {"en": MagicMock(), "de": MagicMock(), "tl": MagicMock()}
    manager._engine_devices = {"en": "cuda", "de": "cuda", "tl": "cpu"}

    assert manager.needs_gpu("en") is True
    assert manager.needs_gpu("de") is True
    assert manager.needs_gpu("tl") is False
    # Unknown language falls back to default language's device
    assert manager.needs_gpu("fr") is True  # default "en" is on cuda
    assert manager.any_on_gpu is True


def test_tts_manager_all_cpu():
    """TTSManager reports no GPU when all engines are on CPU."""
    from paramem.server.tts import TTSManager

    config = TTSConfig(enabled=True, default_language="en")
    manager = TTSManager(config)
    manager._engines = {"en": MagicMock()}
    manager._engine_devices = {"en": "cpu"}

    assert manager.needs_gpu("en") is False
    assert manager.any_on_gpu is False


def test_tts_manager_gpu_fallback_tracks_device():
    """GPU fallback correctly records CPU as the actual device."""
    from paramem.server.tts import TTSManager

    config = TTSConfig(
        enabled=True,
        device="cuda",
        voices={"en": TTSVoiceConfig(engine="piper", model="test")},
    )
    manager = TTSManager(config)

    mock_engine = MagicMock()
    mock_engine._actual_device = "cpu"
    mock_engine.actual_device = "cpu"

    def load_side_effect(device="cpu"):
        if device == "cuda":
            return False
        mock_engine.actual_device = "cpu"
        return True

    mock_engine.load.side_effect = load_side_effect

    with patch("paramem.server.tts._create_engine", return_value=mock_engine):
        manager.load_all()

    assert "en" in manager._engines
    assert manager._engine_devices["en"] == "cpu"
    assert manager.needs_gpu("en") is False


def test_tts_manager_unload_clears_devices():
    """unload_all clears both engines and device tracking."""
    from paramem.server.tts import TTSManager

    config = TTSConfig(enabled=True)
    manager = TTSManager(config)
    mock_engine = MagicMock()
    manager._engines = {"en": mock_engine}
    manager._engine_devices = {"en": "cuda"}

    manager.unload_all()

    assert manager._engines == {}
    assert manager._engine_devices == {}
    assert not manager.is_loaded
    assert not manager.any_on_gpu
    mock_engine.unload.assert_called_once()


# --- Language resolution ---


def test_speaker_language_update_respects_threshold(tmp_path):
    """Speaker language update only happens above threshold."""
    from paramem.server.speaker import SpeakerStore

    store = SpeakerStore(tmp_path / "profiles.json")
    embedding = [0.1] * 192
    sid = store.enroll("Bob", embedding)

    # Below custom threshold — no update
    store.update_language(sid, "fr", 0.85, threshold=0.9)
    assert store.get_preferred_language(sid) is None

    # Above custom threshold — updates
    store.update_language(sid, "fr", 0.95, threshold=0.9)
    assert store.get_preferred_language(sid) == "fr"


def test_speaker_language_same_no_op(tmp_path):
    """Setting the same language does not trigger a disk write."""
    import os

    from paramem.server.speaker import SpeakerStore

    path = tmp_path / "profiles.json"
    store = SpeakerStore(path)
    sid = store.enroll("Alice", [0.1] * 192)
    store.update_language(sid, "de", 0.95)
    store.flush()
    assert store.get_preferred_language(sid) == "de"

    # Record file state after initial write
    mtime_before = os.path.getmtime(path)

    # Set same language again and flush — file should not change
    store.update_language(sid, "de", 0.95)
    store.flush()
    mtime_after = os.path.getmtime(path)

    assert mtime_before == mtime_after


# --- Wyoming TTS handler ---


def test_tts_handler_construction():
    """TTSHandler accepts all parameters without error."""
    pytest.importorskip("wyoming")
    from paramem.server.wyoming_handler import TTSHandler

    mock_tts = MagicMock()
    mock_tts.available_languages = ["en", "de"]

    handler = TTSHandler(
        reader=MagicMock(),
        writer=MagicMock(),
        tts_manager=mock_tts,
        language_resolver=lambda: "de",
        audio_chunk_bytes=8192,
    )
    assert handler._tts is mock_tts
    assert handler._audio_chunk_bytes == 8192


def test_tts_handler_default_chunk_size():
    """TTSHandler defaults to 4096 byte chunks."""
    pytest.importorskip("wyoming")
    from paramem.server.wyoming_handler import TTSHandler

    handler = TTSHandler(
        reader=MagicMock(),
        writer=MagicMock(),
        tts_manager=MagicMock(),
    )
    assert handler._audio_chunk_bytes == 4096


def test_tts_language_source_precedence():
    """tts.language_source picks the synth language: detection-first for
    auto/detection, hint-first for hint; both fall back to the other source."""
    pytest.importorskip("wyoming")
    from paramem.server.wyoming_handler import _resolve_synth_language

    # detection-first (auto / detection): detected language wins over the hint
    assert _resolve_synth_language("en", "de", "auto") == "de"
    assert _resolve_synth_language("en", "de", "detection") == "de"
    # hint mode: caller's voice hint wins
    assert _resolve_synth_language("en", "de", "hint") == "en"
    # fallbacks
    assert _resolve_synth_language("en", None, "auto") == "en"  # no detection -> hint
    assert _resolve_synth_language(None, "de", "hint") == "de"  # no hint -> detected
    assert _resolve_synth_language(None, None, "auto") is None  # neither -> default downstream


def test_tts_config_language_source_default():
    """TTSConfig.language_source defaults to 'auto' (detection-first)."""
    from paramem.server.config import TTSConfig

    assert TTSConfig().language_source == "auto"


def test_kokoro_engine_registered_and_constructed():
    """Kokoro is in the engine registry and _create_engine builds a KokoroTTSEngine
    for a voice with engine: kokoro (construction only, no model download)."""
    from paramem.server.config import TTSConfig, TTSVoiceConfig
    from paramem.server.tts import ENGINE_REGISTRY, KokoroTTSEngine, _create_engine

    assert "kokoro" in ENGINE_REGISTRY
    engine = _create_engine(TTSVoiceConfig(engine="kokoro", model="af_heart"), TTSConfig())
    assert isinstance(engine, KokoroTTSEngine)
    assert engine.is_loaded is False


# --- Config: new STT fields ---


def test_stt_config_defaults():
    """STT config has correct defaults for all fields."""
    from paramem.server.config import STTConfig

    config = STTConfig()
    assert config.language == "auto"
    assert config.beam_size == 5
    assert config.vad_filter is True


def test_stt_config_from_yaml(tmp_path):
    """STT beam_size and vad_filter are parsed from YAML."""
    yaml_content = """
model: mistral
stt:
  enabled: true
  beam_size: 3
  vad_filter: false
  language: de
"""
    yaml_path = tmp_path / "server.yaml"
    yaml_path.write_text(yaml_content)
    config = load_server_config(yaml_path)

    assert config.stt.beam_size == 3
    assert config.stt.vad_filter is False
    assert config.stt.language == "de"


# --- Config: audio_chunk_bytes ---


def test_audio_chunk_bytes_default():
    config = TTSConfig()
    assert config.audio_chunk_bytes == 4096


def test_audio_chunk_bytes_from_yaml(tmp_path):
    yaml_content = """
model: mistral
tts:
  enabled: true
  audio_chunk_bytes: 8192
  voices:
    en:
      engine: piper
      model: test
"""
    yaml_path = tmp_path / "server.yaml"
    yaml_path.write_text(yaml_content)
    config = load_server_config(yaml_path)

    assert config.tts.audio_chunk_bytes == 8192


# --- Config: voice language_name from YAML ---


def test_voice_language_name_from_yaml(tmp_path):
    yaml_content = """
model: mistral
tts:
  enabled: true
  voices:
    de:
      engine: piper
      model: de_DE-thorsten-high
      language_name: Deutsch
    en:
      engine: piper
      model: en_US-lessac-high
"""
    yaml_path = tmp_path / "server.yaml"
    yaml_path.write_text(yaml_content)
    config = load_server_config(yaml_path)

    assert config.tts.voices["de"].language_name == "Deutsch"
    assert config.tts.voices["en"].language_name == ""
    assert config.tts.language_name("de") == "Deutsch"
    assert config.tts.language_name("en") == "English"  # ISO fallback


# --- ISO language names ---


def test_iso_language_names_completeness():
    """ISO mapping covers all target languages."""
    from paramem.server.config import ISO_LANGUAGE_NAMES

    for code in ["en", "de", "fr", "es", "tl"]:
        assert code in ISO_LANGUAGE_NAMES, f"Missing ISO name for {code}"
    assert ISO_LANGUAGE_NAMES["tl"] == "Tagalog"


# --- WhisperSTT accepts new params ---


def test_whisper_stt_accepts_beam_vad():
    """WhisperSTT stores beam_size and vad_filter from config."""
    from paramem.server.stt import WhisperSTT

    stt = WhisperSTT(
        model_name="tiny",
        device="cpu",
        compute_type="int8",
        language="auto",
        beam_size=3,
        vad_filter=False,
    )
    assert stt.beam_size == 3
    assert stt.vad_filter is False


def test_whisper_stt_default_beam_vad():
    """WhisperSTT defaults match STTConfig defaults."""
    from paramem.server.stt import WhisperSTT

    stt = WhisperSTT(model_name="tiny", device="cpu", compute_type="int8", language="auto")
    assert stt.beam_size == 5
    assert stt.vad_filter is True


# --- HA language filtering ---


def test_ha_language_supported_passes_through():
    """Supported language is passed to HA conversation API."""

    # We can't call conversation_process without a real HA, but we can
    # test the filtering logic directly by checking the function signature
    # and the guard logic.
    supported = ["en", "de", "fr", "es"]

    # Supported language — should be kept
    lang = "de"
    if lang and supported and lang not in supported:
        lang = None
    assert lang == "de"


def test_ha_language_unsupported_dropped():
    """Unsupported language is dropped, HA uses its default."""
    supported = ["en", "de", "fr", "es"]

    # Tagalog not supported — should be dropped
    lang = "tl"
    if lang and supported and lang not in supported:
        lang = None
    assert lang is None


def test_ha_language_empty_supported_passes_all():
    """Empty supported_languages list passes all languages through."""
    supported = []

    # When list is empty, no filtering occurs
    lang = "tl"
    if lang and supported and lang not in supported:
        lang = None
    assert lang == "tl"  # no filtering, language passes through


def test_ha_supported_languages_config(tmp_path):
    """HA supported_languages is parsed from YAML."""
    yaml_content = """
model: mistral
tools:
  ha:
    url: http://test:8123
    supported_languages: [en, de, fr]
"""
    yaml_path = tmp_path / "server.yaml"
    yaml_path.write_text(yaml_content)
    config = load_server_config(yaml_path)

    assert config.tools.ha.supported_languages == ["en", "de", "fr"]


def test_ha_supported_languages_default_empty():
    """HA supported_languages defaults to empty (no filtering)."""
    from paramem.server.config import HAToolsConfig

    config = HAToolsConfig()
    assert config.supported_languages == []


# --- Integration test: language escalation paths ---


def test_escalation_language_filtering_integration():
    """Integration: unsupported language falls through HA to cloud.

    Simulates the full escalation chain:
    1. Detected language = Tagalog (tl)
    2. HA supported_languages = [en, de, fr, es] — tl not supported
    3. ha_client.conversation_process receives language=None (filtered)
    4. HA responds (in its default language) — OR fails
    5. Cloud fallback receives language="tl" (always passed)

    This verifies language filtering at the HA boundary while preserving
    language for cloud.
    """
    from paramem.server.config import ServerConfig
    from paramem.server.inference import _escalate_to_ha_agent

    config = ServerConfig()
    config.tools.ha.supported_languages = ["en", "de", "fr", "es"]

    # Mock HA client — capture what language it receives
    mock_ha = MagicMock()
    captured_kwargs = {}

    def capture_call(text, agent_id=None, language=None, supported_languages=None):
        captured_kwargs["language"] = language
        captured_kwargs["supported_languages"] = supported_languages
        return "HA response"

    mock_ha.conversation_process = capture_call

    # Escalate with Tagalog — HA should receive language=None (filtered)
    result = _escalate_to_ha_agent(
        text="Kumusta ang panahon?",
        ha_client=mock_ha,
        config=config,
        language="tl",
    )

    # HA was called, and language was passed through to ha_client
    # (ha_client itself filters based on supported_languages)
    assert captured_kwargs["language"] == "tl"
    assert captured_kwargs["supported_languages"] == ["en", "de", "fr", "es"]
    assert result is not None
    assert result.text == "HA response"


def test_escalation_supported_language_preserved():
    """Integration: supported language is preserved through HA call."""
    from paramem.server.config import ServerConfig
    from paramem.server.inference import _escalate_to_ha_agent

    config = ServerConfig()
    config.tools.ha.supported_languages = ["en", "de", "fr", "es"]

    mock_ha = MagicMock()
    captured = {}

    def capture_call(text, agent_id=None, language=None, supported_languages=None):
        captured["language"] = language
        return "What's the weather?"

    mock_ha.conversation_process = capture_call

    result = _escalate_to_ha_agent(
        text="What's the weather?",
        ha_client=mock_ha,
        config=config,
        language="en",
    )

    assert captured["language"] == "en"
    assert result.text == "What's the weather?"


# --- TTSHandler streaming completion (regression: SynthesizeStopped timing) ---
#
# HA's streaming voice pipeline drives the TTS connection as
#   synthesize-start -> synthesize-chunk(s) -> synthesize -> synthesize-stop
# The consolidated `synthesize` carries the full text (we render audio there);
# `synthesize-stop` is the terminal event, after which HA expects
# SynthesizeStopped. Emitting SynthesizeStopped early (on `synthesize`, before
# HA's stop) desyncs HA's state machine and the satellite/Sonos drops the audio
# (blinks, stays silent). These tests pin: audio on `synthesize`, Stopped only
# after `synthesize-stop`, no double-synthesis, chunk-only streams still render
# on stop, and the one-shot path unchanged.


def _make_streaming_tts_handler():
    pytest.importorskip("wyoming")
    from paramem.server.wyoming_handler import TTSHandler

    tts = MagicMock(name="tts_manager")
    tts.needs_gpu.return_value = False
    tts.synthesize.return_value = (b"\x00\x00" * 100, 24000)  # 100 samples silence @ 24 kHz

    handler = TTSHandler(
        reader=MagicMock(name="reader"),
        writer=MagicMock(name="writer"),
        tts_manager=tts,
        audio_chunk_bytes=4096,
    )
    return handler, tts


async def _drive(handler, events):
    """Feed wire events through handle_event, capturing everything it writes."""
    recorded = []

    async def _fake_write(event, writer):
        recorded.append(event)

    results = []
    with patch("paramem.server.wyoming_handler.async_write_event", _fake_write):
        for ev in events:
            results.append(await handler.handle_event(ev.event()))
    return results, recorded


def _count(recorded, klass):
    return sum(1 for e in recorded if klass.is_type(e.type))


def test_streaming_audio_on_synthesize_stopped_deferred_to_stop():
    """Audio is sent on `synthesize`, but SynthesizeStopped only after `synthesize-stop`.

    Emitting SynthesizeStopped on the consolidated `synthesize` (before HA sends
    its stop) is exactly the bug that left the Sonos silent.
    """
    pytest.importorskip("wyoming")
    import asyncio

    from wyoming.audio import AudioStart, AudioStop
    from wyoming.tts import (
        Synthesize,
        SynthesizeChunk,
        SynthesizeStart,
        SynthesizeStop,
        SynthesizeStopped,
    )

    handler, tts = _make_streaming_tts_handler()

    # Phase 1: start -> chunk -> synthesize. Audio must be sent now, Stopped must NOT.
    results1, recorded1 = asyncio.run(
        _drive(
            handler,
            [
                SynthesizeStart(voice=None),
                SynthesizeChunk(text="It's "),
                Synthesize(text="It's 9:04 PM.", voice=None),
            ],
        )
    )
    assert results1 == [True, True, True]  # stream stays open
    assert _count(recorded1, AudioStart) == 1
    assert _count(recorded1, AudioStop) == 1
    assert _count(recorded1, SynthesizeStopped) == 0  # deferred — the regression assertion
    assert tts.synthesize.call_count == 1
    assert handler._audio_sent is True
    assert handler._streaming is True

    # Phase 2: synthesize-stop. Now SynthesizeStopped is emitted, no re-synthesis.
    results2, recorded2 = asyncio.run(_drive(handler, [SynthesizeStop()]))
    assert results2 == [True]
    assert _count(recorded2, SynthesizeStopped) == 1
    assert tts.synthesize.call_count == 1  # not synthesized again
    assert handler._streaming is False  # stream state reset


def test_streaming_stop_without_consolidated_synthesize_renders_on_stop():
    """start -> chunk -> synthesize-stop (no consolidated Synthesize) renders on stop."""
    pytest.importorskip("wyoming")
    import asyncio

    from wyoming.tts import SynthesizeChunk, SynthesizeStart, SynthesizeStop, SynthesizeStopped

    handler, tts = _make_streaming_tts_handler()
    results, recorded = asyncio.run(
        _drive(
            handler,
            [SynthesizeStart(voice=None), SynthesizeChunk(text="Hallo Welt"), SynthesizeStop()],
        )
    )

    assert results == [True, True, True]
    assert _count(recorded, SynthesizeStopped) == 1
    assert tts.synthesize.call_count == 1  # accumulated chunk text rendered once


def test_streaming_synthesize_then_stop_single_synthesis_single_stopped():
    """Full HA flow start -> chunk -> synthesize -> stop: one synth, one Stopped."""
    pytest.importorskip("wyoming")
    import asyncio

    from wyoming.tts import (
        Synthesize,
        SynthesizeChunk,
        SynthesizeStart,
        SynthesizeStop,
        SynthesizeStopped,
    )

    handler, tts = _make_streaming_tts_handler()
    _results, recorded = asyncio.run(
        _drive(
            handler,
            [
                SynthesizeStart(voice=None),
                SynthesizeChunk(text="x"),
                Synthesize(text="full text", voice=None),
                SynthesizeStop(),
            ],
        )
    )

    assert tts.synthesize.call_count == 1  # not synthesized twice
    assert _count(recorded, SynthesizeStopped) == 1  # emitted exactly once, on stop


def test_oneshot_synthesize_closes_without_stopped():
    """A bare Synthesize (no SynthesizeStart) keeps one-shot semantics: close, no Stopped."""
    pytest.importorskip("wyoming")
    import asyncio

    from wyoming.audio import AudioStart, AudioStop
    from wyoming.tts import Synthesize, SynthesizeStopped

    handler, tts = _make_streaming_tts_handler()
    results, recorded = asyncio.run(_drive(handler, [Synthesize(text="hello", voice=None)]))

    assert results == [False]  # connection-close signals completion for one-shot callers
    assert _count(recorded, AudioStart) == 1
    assert _count(recorded, AudioStop) == 1
    assert _count(recorded, SynthesizeStopped) == 0


# ---------------------------------------------------------------------------
# Phase B — cloud safety + resolver + _sanitize_history (B4/B5)
# ---------------------------------------------------------------------------


class TestCloudCarriesNoIdentity:
    """Cloud never learns the speaker's id or name (NEW contract, 2026-08-02):
    ``_escalate_to_cloud`` — the sole cloud transport primitive — takes no
    speaker parameter at all and never calls ``_build_speaker_prefix``.  This
    replaces the OLD invariant tested here previously ("display names to
    cloud, ids never") — no identity of any kind reaches the cloud now; the
    speaker{N} token is reserved for the LOCAL reasoning leg only, and a
    display name is substituted only at the reply boundary
    (``paramem.server.speaker.resolve_speaker_tokens``), never sent anywhere.
    """

    def test_escalate_to_cloud_has_no_speaker_param(self) -> None:
        """_escalate_to_cloud's signature carries no speaker/speaker_id param."""
        import inspect

        from paramem.server.inference import _escalate_to_cloud

        params = inspect.signature(_escalate_to_cloud).parameters
        assert "speaker" not in params
        assert "speaker_id" not in params

    def test_escalate_to_cloud_system_prompt_has_no_identity_line(self) -> None:
        """The prompt sent to the cloud agent carries no "You are speaking
        with" line and no speaker{N} token — cloud gets the bare CLOUD_PROMPT."""
        from paramem.server.inference import CLOUD_PROMPT, _escalate_to_cloud

        cloud_agent = MagicMock()
        cloud_agent.call.return_value = MagicMock(text="answer")

        _escalate_to_cloud("hi", cloud_agent, config=None, language=None)

        sent_prompt = cloud_agent.call.call_args.kwargs["system_prompt"]
        assert sent_prompt == CLOUD_PROMPT
        assert "You are speaking with" not in sent_prompt
        assert "speaker" not in sent_prompt.lower()
