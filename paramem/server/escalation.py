"""Cloud escalation — forwards queries to a SOTA model when the local model can't answer."""

import logging

import httpx

from paramem.server.config import GeneralAgentConfig

logger = logging.getLogger(__name__)

ESCALATE_TAG = "[ESCALATE]"


def detect_escalation(response: str) -> tuple[bool, str]:
    """Check if the model response contains an escalation signal.

    Returns (should_escalate, forwarded_query).
    The forwarded query is the text after the [ESCALATE] tag.
    The tag may appear mid-response (e.g. "I don't know [ESCALATE] ...").
    """
    idx = response.find(ESCALATE_TAG)
    if idx >= 0:
        query = response[idx + len(ESCALATE_TAG) :].strip().lstrip(":").strip()
        return True, query
    return False, ""


def escalate_to_cloud(query: str, cloud_config: GeneralAgentConfig) -> str:
    """Forward a query to the cloud SOTA model.

    Only the query is sent — no conversation history, no personal context.
    The cloud response is returned verbatim.

    This is the legacy OpenAI-compatible escalation path. Will be replaced
    by the provider adapter pattern in F5.2a (cloud/base.py).
    """
    if not cloud_config.enabled or not cloud_config.endpoint:
        logger.warning("Cloud escalation requested but cloud is not configured")
        return ""

    payload = {
        "model": cloud_config.model,
        "messages": [{"role": "user", "content": query}],
        "temperature": 0.7,
        "max_tokens": 512,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {cloud_config.api_key}",
    }

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(cloud_config.endpoint, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
    except httpx.TimeoutException:
        logger.error("Cloud escalation timed out for query: %s", query[:100])
        return "I'm sorry, I couldn't reach the cloud service in time."
    except (httpx.HTTPStatusError, KeyError, IndexError) as e:
        logger.error("Cloud escalation failed: %s", e)
        return "I'm sorry, I couldn't get an answer from the cloud service."
