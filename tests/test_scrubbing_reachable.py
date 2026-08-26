"""``scrubbing_reachable``: the shipped example config is unreachable
through every term (cloud disabled, no HA agent), the fixture is
reachable through term (b) unconditionally, term (a)'s env-key variant,
``scrub_enabled=False`` is never reachable, and ``cloud_only`` does not
appear in the signature.
"""

from __future__ import annotations

import inspect

from paramem.cloud.admission import scrubbing_reachable
from paramem.server.config import load_server_config

_PROVIDER_KEY_ENV_VARS = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "GROQ_API_KEY",
    "MISTRAL_API_KEY",
)


def _terms(config):
    return dict(
        cloud_enabled=config.cloud.enabled,
        cloud_mode=config.sanitization.cloud_mode,
        provider=config.consolidation.extraction_enrichment_provider,
        model=config.consolidation.extraction_enrichment_provider_model,
        endpoint=config.consolidation.extraction_enrichment_provider_endpoint or None,
        ha_agent_id="",
        ha_tools_configured=False,
    )


class TestExampleConfigIsNotEgressCapable:
    """The shipped ``configs/server.yaml.example`` ships
    ``cloud.enabled: false`` and no HA agent configured — a non-empty
    ``sanitization.scrub`` must not, by itself, make the default
    deployment egress-capable through any of the three terms."""

    def test_example_config_is_unreachable(self, monkeypatch) -> None:
        for env_var in _PROVIDER_KEY_ENV_VARS:
            monkeypatch.delenv(env_var, raising=False)
        monkeypatch.delenv("HA_URL", raising=False)
        monkeypatch.delenv("HA_TOKEN", raising=False)

        config = load_server_config("configs/server.yaml.example")

        assert config.cloud.enabled is False
        assert config.ha_agent_id == ""
        assert config.tools.ha.configured is False
        # scrub is ON by default -- the unreachable verdict below must
        # come from the egress terms, not from an off-by-default scrub.
        assert bool(config.sanitization.scrub_categories) is True

        assert (
            scrubbing_reachable(
                scrub_enabled=bool(config.sanitization.scrub_categories),
                cloud_enabled=config.cloud.enabled,
                cloud_mode=config.sanitization.cloud_mode,
                provider=config.consolidation.extraction_enrichment_provider,
                model=config.consolidation.extraction_enrichment_provider_model,
                endpoint=config.consolidation.extraction_enrichment_provider_endpoint or None,
                ha_agent_id=config.ha_agent_id,
                ha_tools_configured=config.tools.ha.configured,
            )
            is False
        )


class TestFixtureReachableThroughTermBUnconditionally:
    def test_fixture_cloud_enabled_anonymize_is_reachable_regardless_of_provider(
        self, monkeypatch
    ) -> None:
        config = load_server_config("tests/fixtures/server.yaml")
        assert config.cloud.enabled is True
        assert config.sanitization.cloud_mode == "anonymize"
        # No provider key exported at all — term (a) cannot hold; term (b)
        # must carry this alone.
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert (
            scrubbing_reachable(
                scrub_enabled=True,
                cloud_enabled=config.cloud.enabled,
                cloud_mode=config.sanitization.cloud_mode,
                provider="",
                model="",
                endpoint=None,
                ha_agent_id="",
                ha_tools_configured=False,
            )
            is True
        )


class TestTermAEnvKeyVariant:
    def test_reachable_with_the_providers_key_exported(self, monkeypatch) -> None:
        config = load_server_config("tests/fixtures/server.yaml")
        config.sanitization.cloud_mode = "block"  # term (b) can never fire here
        assert config.consolidation.extraction_enrichment_provider == "anthropic"
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-value")
        assert scrubbing_reachable(scrub_enabled=True, **_terms(config)) is True

    def test_not_reachable_without_the_providers_key(self, monkeypatch) -> None:
        config = load_server_config("tests/fixtures/server.yaml")
        config.sanitization.cloud_mode = "block"
        assert config.consolidation.extraction_enrichment_provider == "anthropic"
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert scrubbing_reachable(scrub_enabled=True, **_terms(config)) is False


class TestScrubEnabledFalseIsNeverReachable:
    def test_false_scrub_enabled_beats_every_other_term(self, monkeypatch) -> None:
        config = load_server_config("tests/fixtures/server.yaml")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-value")
        assert config.cloud.enabled is True
        assert config.sanitization.cloud_mode == "anonymize"
        assert scrubbing_reachable(scrub_enabled=False, **_terms(config)) is False


class TestCloudOnlyNotInSignature:
    def test_signature_has_no_cloud_only_parameter(self) -> None:
        params = inspect.signature(scrubbing_reachable).parameters
        assert "cloud_only" not in params
