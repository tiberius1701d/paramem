"""Query sanitizer — detects personal context before cloud escalation.

Catches queries that bypass entity matching but still contain implicit
personal information: possessive pronouns + relationship terms, self-
referential questions, location/workplace references.

Three modes:
  - "off": no sanitization (cloud sees everything)
  - "warn": log a warning but send to cloud anyway
  - "block": fall back to local model instead of sending to cloud

This is defense-in-depth. The primary protection is entity-based routing
(queries with known entities never reach cloud). The sanitizer catches
what entity matching misses.
"""

import logging
import re

logger = logging.getLogger(__name__)

# Possessive + relationship: "my wife", "our house", "my sister's"
_POSSESSIVE_PERSONAL = re.compile(
    r"\b(?:my|our|mine)\s+"
    r"(?:wife|husband|partner|spouse|girlfriend|boyfriend|"
    r"son|daughter|child|children|kids|baby|"
    r"mother|father|mom|dad|mum|parent|parents|"
    r"brother|sister|sibling|"
    r"family|uncle|aunt|cousin|grandma|grandmother|grandpa|grandfather|"
    r"friend|best friend|neighbor|colleague|boss|"
    r"dog|cat|pet|"
    r"house|home|apartment|flat|place|"
    r"car|job|work|office|company|school|doctor|dentist|"
    r"birthday|anniversary|wedding|"
    r"favorite|favourite)\b",
    re.IGNORECASE,
)

# Self-referential: "where do I live", "what's my name", "when is my"
_SELF_REFERENTIAL = re.compile(
    r"\b(?:where\s+(?:do|did|was)\s+I|what(?:'s| is)\s+my|when\s+(?:is|was)\s+my|"
    r"who\s+(?:is|are)\s+my|how\s+old\s+am\s+I|"
    r"what\s+do\s+I\s+(?:do|like|prefer|work)|"
    r"tell\s+me\s+about\s+(?:my|me)|"
    r"do\s+I\s+(?:have|own|like|know))\b",
    re.IGNORECASE,
)

# Direct personal claims: "I live in", "I work at", "I'm married"
_PERSONAL_CLAIMS = re.compile(
    r"\bI\s+(?:live|work|study|go to|attend|moved|grew up)\s+(?:in|at|to)\b|"
    r"\bI(?:'m| am)\s+(?:married|engaged|divorced|single|pregnant)\b|"
    r"\bI\s+(?:have|own)\s+(?:a |an |two |three )?"
    r"(?:dog|cat|pet|car|house|kid|child|daughter|son)\b",
    re.IGNORECASE,
)


def check_personal_content(text: str) -> list[str]:
    """Check if text contains personal context patterns.

    Returns a list of matched pattern descriptions. Empty list means clean.
    """
    findings = []

    if _POSSESSIVE_PERSONAL.search(text):
        findings.append("possessive_personal")

    if _SELF_REFERENTIAL.search(text):
        findings.append("self_referential")

    if _PERSONAL_CLAIMS.search(text):
        findings.append("personal_claim")

    return findings


def sanitize_for_cloud(
    text: str,
    mode: str = "warn",
) -> tuple[str | None, list[str]]:
    """Check query before sending to cloud. Returns (query, findings).

    Args:
        text: The query to check.
        mode: "off" (skip), "warn" (log + pass through), "block" (return None).

    Returns:
        (sanitized_query, findings). If mode="block" and personal content
        found, sanitized_query is None (caller should fall back to local).
    """
    if mode == "off":
        return text, []

    findings = check_personal_content(text)

    if not findings:
        return text, []

    if mode == "warn":
        logger.warning(
            "Personal content detected in cloud-bound query: %s — %s",
            findings,
            text[:80],
        )
        return text, findings

    if mode == "block":
        logger.info(
            "Blocked cloud escalation due to personal content: %s — %s",
            findings,
            text[:80],
        )
        return None, findings

    # Unknown mode — fail closed
    logger.warning("Unknown sanitization mode '%s', treating as 'block'", mode)
    return None, findings
