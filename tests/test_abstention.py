"""Tests for the last-resort abstention short-circuit.

When the sanitizer blocks cloud escalation (self-referential / personal
query) AND no local parametric-memory match was found, ``handle_chat``
must return the configured canned response rather than invoking the bare
base model, which would otherwise confabulate plausible-sounding personal
data (observed: untrained adapter + "Where do I live?" → "New York City").
"""

from unittest.mock import MagicMock, patch

from paramem.server.config import (
    AbstentionConfig,
    ServerConfig,
    load_server_config,
)


class TestAbstentionConfig:
    def test_defaults(self):
        cfg = AbstentionConfig()
        assert cfg.enabled is True
        assert cfg.response == "I don't have that information stored yet."

    def test_server_config_includes_abstention(self):
        config = ServerConfig()
        assert isinstance(config.abstention, AbstentionConfig)
        assert config.abstention.enabled is True

    def test_yaml_override(self, tmp_path):
        config_file = tmp_path / "server.yaml"
        config_file.write_text(
            'abstention:\n  enabled: false\n  response: "Custom abstention message."\n'
        )
        config = load_server_config(config_file)
        assert config.abstention.enabled is False
        assert config.abstention.response == "Custom abstention message."

    def test_yaml_partial_override_keeps_defaults(self, tmp_path):
        config_file = tmp_path / "server.yaml"
        config_file.write_text("abstention:\n  enabled: false\n")
        config = load_server_config(config_file)
        assert config.abstention.enabled is False
        assert config.abstention.response == "I don't have that information stored yet."

    def test_project_server_yaml_has_abstention_enabled(self):
        config = load_server_config("configs/server.yaml")
        assert config.abstention.enabled is True
        assert config.abstention.response


class TestAbstentionShortCircuit:
    """Verify the short-circuit fires at the correct decision point in
    ``handle_chat`` and does not perturb paths that shouldn't be affected.
    """

    def _make_none_match_router(self):
        from paramem.server.router import RoutingPlan

        router = MagicMock()
        router.route = lambda text, speaker=None, speaker_id=None: RoutingPlan(
            strategy="direct", match_source="none"
        )
        router._speaker_key_index = {}
        return router

    def _minimal_mock_model(self):
        model = MagicMock()
        model.gradient_checkpointing_disable = MagicMock()
        return model

    def test_fires_when_sanitizer_blocks_and_no_match(self):
        """Self-referential query + untrained adapter → canned response,
        never invokes ``_base_model_answer``."""
        from paramem.server.inference import handle_chat

        config = ServerConfig()
        # Explicit defaults for defensive clarity
        assert config.abstention.enabled is True

        with (
            patch(
                "paramem.server.inference.sanitize_for_cloud",
                return_value=(None, ["self_referential"]),
            ),
            patch("paramem.server.inference._base_model_answer") as mock_base_model,
            patch(
                "paramem.server.inference.detect_temporal_query",
                return_value=None,
            ),
        ):
            result = handle_chat(
                text="Where do I live?",
                conversation_id="test",
                speaker="Alex",
                history=None,
                model=self._minimal_mock_model(),
                tokenizer=MagicMock(),
                config=config,
                router=self._make_none_match_router(),
                speaker_id="spk-abc123",
            )

        assert result.text == config.abstention.response
        mock_base_model.assert_not_called()

    def test_fires_for_anonymous_speaker_with_id(self):
        """speaker_id present but real name absent — still fires (per
        deferred-identity-binding design: id is sufficient for attribution)."""
        from paramem.server.inference import handle_chat

        config = ServerConfig()

        with (
            patch(
                "paramem.server.inference.sanitize_for_cloud",
                return_value=(None, ["self_referential"]),
            ),
            patch("paramem.server.inference._base_model_answer") as mock_base_model,
            patch(
                "paramem.server.inference.detect_temporal_query",
                return_value=None,
            ),
        ):
            result = handle_chat(
                text="What's my birthday?",
                conversation_id="test",
                speaker=None,  # name not yet disclosed
                history=None,
                model=self._minimal_mock_model(),
                tokenizer=MagicMock(),
                config=config,
                router=self._make_none_match_router(),
                speaker_id="spk-anon-42",
            )

        assert result.text == config.abstention.response
        mock_base_model.assert_not_called()

    def test_self_introduction_falls_through_to_base_model(self):
        """Statement-form personal content (self-introduction, fact-sharing)
        is not a confabulation risk — the user is the source of the facts in
        the same turn — so the abstention must NOT fire even though the
        sanitizer blocks the cloud path. Without this, the user gets
        ``"I don't have that information stored yet."`` in response to
        ``"I'm Alex. I live in Kelkham."`` instead of a conversational
        acknowledgement.
        """
        from paramem.server.inference import ChatResult, handle_chat

        config = ServerConfig()

        with (
            patch(
                "paramem.server.inference.sanitize_for_cloud",
                return_value=(None, ["personal_claim", "possessive_personal"]),
            ),
            patch(
                "paramem.server.inference._base_model_answer",
                return_value=ChatResult(text="Nice to meet you, Alex."),
            ) as mock_base_model,
            patch(
                "paramem.server.inference.detect_temporal_query",
                return_value=None,
            ),
        ):
            result = handle_chat(
                text="I'm Alex. I live in Kelkham with my wife Pat.",
                conversation_id="test",
                speaker=None,
                history=None,
                model=self._minimal_mock_model(),
                tokenizer=MagicMock(),
                config=config,
                router=self._make_none_match_router(),
                speaker_id="spk-anon-1",
            )

        mock_base_model.assert_called_once()
        assert result.text != config.abstention.response

    def test_disabled_falls_through_to_base_model(self):
        """With abstention.enabled=False, behavior matches pre-change:
        last-resort ``_base_model_answer`` still runs."""
        from paramem.server.inference import ChatResult, handle_chat

        config = ServerConfig()
        config.abstention.enabled = False

        with (
            patch(
                "paramem.server.inference.sanitize_for_cloud",
                return_value=(None, ["self_referential"]),
            ),
            patch(
                "paramem.server.inference._base_model_answer",
                return_value=ChatResult(text="base model answer"),
            ) as mock_base_model,
            patch(
                "paramem.server.inference.detect_temporal_query",
                return_value=None,
            ),
        ):
            result = handle_chat(
                text="Where do I live?",
                conversation_id="test",
                speaker="Alex",
                history=None,
                model=self._minimal_mock_model(),
                tokenizer=MagicMock(),
                config=config,
                router=self._make_none_match_router(),
                speaker_id="spk-abc123",
            )

        mock_base_model.assert_called_once()
        assert result.text == "base model answer"

    def test_skipped_when_cloud_available(self):
        """Non-personal query (sanitizer allowed) + no local match →
        cloud escalation path runs; abstention does not fire."""
        from paramem.server.inference import ChatResult, handle_chat

        config = ServerConfig()

        with (
            patch(
                "paramem.server.inference.sanitize_for_cloud",
                return_value=("What's the weather?", []),
            ),
            patch(
                "paramem.server.inference._escalate_to_ha_agent",
                return_value=ChatResult(text="cloud handled it", escalated=True),
            ) as mock_ha,
            patch("paramem.server.inference._base_model_answer") as mock_base_model,
            patch(
                "paramem.server.inference.detect_temporal_query",
                return_value=None,
            ),
        ):
            result = handle_chat(
                text="What's the weather?",
                conversation_id="test",
                speaker="Alex",
                history=None,
                model=self._minimal_mock_model(),
                tokenizer=MagicMock(),
                config=config,
                router=self._make_none_match_router(),
                speaker_id="spk-abc123",
            )

        mock_ha.assert_called_once()
        mock_base_model.assert_not_called()
        assert result.text == "cloud handled it"

    def test_skipped_when_cloud_fails_on_non_personal_query(self):
        """Sanitizer allowed the query (non-personal) but cloud is
        unavailable → base model fallback, NOT abstention. The short-circuit
        is scoped to the sanitizer-blocked case; cloud-outage on general
        queries still uses base-model general knowledge."""
        from paramem.server.inference import ChatResult, handle_chat

        config = ServerConfig()

        with (
            patch(
                "paramem.server.inference.sanitize_for_cloud",
                return_value=("What's the weather?", []),
            ),
            patch(
                "paramem.server.inference._escalate_to_ha_agent",
                return_value=None,  # HA unavailable
            ),
            patch(
                "paramem.server.inference._base_model_answer",
                return_value=ChatResult(text="base fallback"),
            ) as mock_base_model,
            patch(
                "paramem.server.inference.detect_temporal_query",
                return_value=None,
            ),
        ):
            result = handle_chat(
                text="What's the weather?",
                conversation_id="test",
                speaker="Alex",
                history=None,
                model=self._minimal_mock_model(),
                tokenizer=MagicMock(),
                config=config,
                router=self._make_none_match_router(),
                sota_agent=None,  # no SOTA available either
                speaker_id="spk-abc123",
            )

        mock_base_model.assert_called_once()
        assert result.text == "base fallback"
