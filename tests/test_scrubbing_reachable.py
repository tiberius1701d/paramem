"""``scrubbing_reachable``: the example config is unreachable, the fixture
is reachable through term (b) unconditionally, term (a)'s env-key
variant, ``scrub_enabled=False`` is never reachable, and ``cloud_only``
does not appear in the signature.
"""

from __future__ import annotations

import inspect

from paramem.cloud.admission import scrubbing_reachable
from paramem.server.config import load_server_config


def _terms(config):
    return dict(
        cloud_enabled=config.cloud.enabled,
        cloud_mode=config.sanitization.cloud_mode,
        provider=config.consolidation.extraction_enrichment_provider,
        model=config.consolidation.extraction_enrichment_provider_model,
        endpoint=config.consolidation.extraction_enrichment_provider_endpoint or None,
    )


class TestExampleConfigNotReachable:
    def test_shipped_example_is_not_egress_capable(self) -> None:
        config = load_server_config("configs/server.yaml.example")
        scrub_enabled = bool(config.sanitization.scrub_categories)
        assert scrubbing_reachable(scrub_enabled=scrub_enabled, **_terms(config)) is False


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
