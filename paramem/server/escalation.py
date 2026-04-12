"""Escalation signal detection for local model responses."""

import logging

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
