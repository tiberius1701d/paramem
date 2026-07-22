"""Cloud agent registry — maps provider names to adapter classes."""

import logging
from dataclasses import replace

from paramem.cloud.admission import evaluate_cloud_egress
from paramem.cloud.providers.base import CloudAgent, CloudAgentConfig
from paramem.cloud.providers.openai_compat import COMPATIBLE_PROVIDERS, OpenAICompatAgent

logger = logging.getLogger(__name__)

ALL_PROVIDERS = sorted(COMPATIBLE_PROVIDERS | {"anthropic", "google"})


def get_cloud_agent(config: CloudAgentConfig, *, cloud_enabled: bool) -> CloudAgent | None:
    """Create a cloud agent for the configured provider, or ``None`` if inadmissible.

    Admission is delegated wholesale to
    :func:`~paramem.cloud.admission.evaluate_cloud_egress` — the single
    egress predicate shared with the extraction chain, the graph-tier
    enrichment pass and ``/calibrate/enrich``.  The agent is built only
    when the verdict permits, so a provider with an endpoint but no
    resolvable API key is refused here instead of sending
    ``Authorization: Bearer `` (empty) at call time.

    The returned agent's ``api_key`` is the verdict's env-resolved key, not
    whatever the YAML happened to carry: one call, one key source.

    Optional providers (anthropic, google) additionally require their SDK,
    installed via ``pip install paramem[anthropic]`` / ``paramem[google]``.

    Args:
        config: Provider/model/endpoint/credentials for one cloud agent.
            Carries no on-off switch of its own.
        cloud_enabled: The deployment-wide master switch,
            ``ServerConfig.cloud.enabled``.  ``False`` refuses every
            provider regardless of how well it is configured.

    Returns:
        A ready :class:`~paramem.cloud.providers.base.CloudAgent`, or
        ``None`` when egress is refused, the provider is unknown, or an
        optional provider's SDK is missing.  Every ``None`` is logged with
        the reason.
    """
    provider = config.provider.lower()
    verdict = evaluate_cloud_egress(
        cloud_enabled=cloud_enabled,
        provider=provider,
        model=config.model,
        endpoint=config.endpoint or None,
    )
    if not verdict.permitted:
        logger.info("Cloud agent %r not admitted: %s", provider, "; ".join(verdict.gaps))
        return None

    # One key source: the verdict's env-resolved key replaces the YAML
    # surface so the credential that authenticates the call is the same one
    # admission checked.
    admitted = replace(config, api_key=verdict.api_key)

    if provider in COMPATIBLE_PROVIDERS:
        logger.info("Cloud agent: %s (%s)", provider, config.model)
        return OpenAICompatAgent(admitted)

    if provider == "anthropic":
        try:
            from paramem.cloud.providers.anthropic_adapter import AnthropicAgent
        except ImportError:
            logger.error(
                "Anthropic provider requires the anthropic SDK. "
                "Install with: pip install paramem[anthropic]"
            )
            return None

        logger.info("Cloud agent: anthropic (%s)", config.model)
        return AnthropicAgent(admitted)

    if provider == "google":
        try:
            from paramem.cloud.providers.google_adapter import GoogleAgent
        except ImportError:
            logger.error(
                "Google provider requires the google-genai SDK. "
                "Install with: pip install paramem[google]"
            )
            return None

        logger.info("Cloud agent: google (%s)", config.model)
        return GoogleAgent(admitted)

    # Unreachable via evaluate_cloud_egress (an unknown provider is refused
    # above), but ALL_PROVIDERS and PROVIDER_KEY_ENV are separate tables:
    # a provider registered for key lookup with no adapter lands here.
    logger.warning(
        "No adapter for cloud provider: '%s'. Available: %s",
        provider,
        ALL_PROVIDERS,
    )
    return None
