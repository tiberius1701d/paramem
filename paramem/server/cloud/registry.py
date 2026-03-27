"""Cloud agent registry — maps provider names to adapter classes."""

import logging

from paramem.server.cloud.base import CloudAgent
from paramem.server.cloud.openai_compat import COMPATIBLE_PROVIDERS, OpenAICompatAgent
from paramem.server.config import GeneralAgentConfig

logger = logging.getLogger(__name__)


def get_cloud_agent(config: GeneralAgentConfig) -> CloudAgent | None:
    """Create a cloud agent for the configured provider.

    Returns None if cloud is not enabled or not configured.
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
        logger.warning("Anthropic adapter not yet implemented")
        return None

    if provider == "google":
        logger.warning("Google adapter not yet implemented")
        return None

    logger.warning(
        "Unknown cloud provider: '%s'. Available: %s",
        provider,
        COMPATIBLE_PROVIDERS,
    )
    return None
