"""Cloud agent registry — maps provider names to adapter classes."""

import logging

from paramem.server.cloud.base import CloudAgent
from paramem.server.cloud.openai_compat import COMPATIBLE_PROVIDERS, OpenAICompatAgent
from paramem.server.config import CloudAgentConfig

logger = logging.getLogger(__name__)

ALL_PROVIDERS = sorted(COMPATIBLE_PROVIDERS | {"anthropic", "google"})


def get_cloud_agent(config: CloudAgentConfig) -> CloudAgent | None:
    """Create a cloud agent for the configured provider.

    Returns None if cloud is not enabled or not configured.
    Optional providers (anthropic, google) require their SDK to be installed
    via pip install paramem[anthropic] or paramem[google].
    """
    if not config.enabled:
        return None

    provider = config.provider.lower()

    if provider in COMPATIBLE_PROVIDERS:
        agent = OpenAICompatAgent(config)
        if agent.is_available():
            logger.info("Cloud agent: %s (%s)", provider, config.model)
            return agent
        logger.warning("Cloud agent %s configured but API key missing", provider)
        return None

    if provider == "anthropic":
        try:
            from paramem.server.cloud.anthropic_adapter import AnthropicAgent
        except ImportError:
            logger.error(
                "Anthropic provider requires the anthropic SDK. "
                "Install with: pip install paramem[anthropic]"
            )
            return None

        agent = AnthropicAgent(config)
        if agent.is_available():
            logger.info("Cloud agent: anthropic (%s)", config.model)
            return agent
        logger.warning("Anthropic agent configured but API key missing")
        return None

    if provider == "google":
        try:
            from paramem.server.cloud.google_adapter import GoogleAgent
        except ImportError:
            logger.error(
                "Google provider requires the google-genai SDK. "
                "Install with: pip install paramem[google]"
            )
            return None

        agent = GoogleAgent(config)
        if agent.is_available():
            logger.info("Cloud agent: google (%s)", config.model)
            return agent
        logger.warning("Google agent configured but API key missing")
        return None

    logger.warning(
        "Unknown cloud provider: '%s'. Available: %s",
        provider,
        ALL_PROVIDERS,
    )
    return None
