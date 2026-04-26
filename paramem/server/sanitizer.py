"""Query sanitizer — graph-anchored personal-content gate before cloud escalation.

Anchored on runtime ground truth, not lexical patterns:

* The speaker's **known entities** (the router's entity index plus the speaker
  store's enrolled names) are the source of truth for what counts as personal.
  Detection reuses ``_anonymize_transcript`` from the extraction pipeline —
  the same primitive that produces SOTA-safe transcripts.  Anonymization
  replacing anything means the text contained a personal reference.
* **First-person pronouns** plus an identified speaker are treated as a
  self-reference even when no known entity is named (covers cold-start before
  the graph has facts to anchor on).  The token set is explicit; there is no
  pattern matching.

The contract ``(sanitized_text_or_None, findings)`` is unchanged so callers
in ``inference.py`` and ``test_abstention.py`` keep working.  Findings:

* ``personal_entity`` — query mentions an entity in the speaker's graph or
  an enrolled name.
* ``self_referential`` — first-person pronoun + identified speaker +
  interrogative shape (consumed by the abstention short-circuit).
* ``personal_claim`` — first-person pronoun + identified speaker +
  declarative shape.

Modes: ``off`` / ``warn`` / ``block``.  Same semantics as before.
"""

import logging

from paramem.graph.extractor import _anonymize_transcript
from paramem.server.router import _is_interrogative

logger = logging.getLogger(__name__)


# Token-set lookup, not a pattern.  Explicit list of first-person openings
# the chat handler resolves to the identified speaker.  Includes
# contractions because chat input is unmodified text.
_FIRST_PERSON_TOKENS = frozenset(
    {
        "i",
        "i'm",
        "i'd",
        "i've",
        "i'll",
        "me",
        "my",
        "mine",
        "myself",
        "we",
        "we're",
        "we'd",
        "we've",
        "we'll",
        "us",
        "our",
        "ours",
        "ourselves",
    }
)

# Punctuation stripped before token comparison so "I'm." or "me," still match.
_PUNCT = ".,!?;:'\"()[]{}"


def _contains_first_person(text: str) -> bool:
    """Token-level scan for first-person pronouns.  No regex."""
    for raw in text.split():
        token = raw.strip(_PUNCT).lower()
        if token in _FIRST_PERSON_TOKENS:
            return True
    return False


def _build_known_entity_mapping(known_entities: set[str] | None) -> dict[str, str]:
    """Build a name → opaque-placeholder mapping for ``_anonymize_transcript``.

    Placeholders are unique per known entity so the comparison
    ``anonymized != original`` reliably detects matches.  The actual
    placeholder strings are not surfaced — callers only see the
    ``personal_entity`` finding.

    Empty input yields an empty dict; ``_anonymize_transcript`` is a no-op
    on an empty mapping, which preserves back-compat for callers that
    don't yet supply ``known_entities``.
    """
    if not known_entities:
        return {}
    return {name: f"__PERSONAL_{i}__" for i, name in enumerate(known_entities) if name}


def check_personal_content(
    text: str,
    *,
    speaker_id: str | None = None,
    known_entities: set[str] | None = None,
) -> list[str]:
    """Return findings explaining why the text is personal.  Empty = clean.

    Back-compat: if neither ``speaker_id`` nor ``known_entities`` is supplied
    the function can still detect first-person without speaker context only as
    informational; without an identified speaker it returns ``[]`` because
    there is no resolution target for "I" / "my".  This matches the project's
    "personal data is graph-truth" principle: a pronoun in a vacuum is not
    personal until it resolves to someone we know.
    """
    findings: list[str] = []

    mapping = _build_known_entity_mapping(known_entities)
    if mapping:
        # Case-insensitive comparison: production stores lowercased entity
        # names (router._all_entities, speaker_names lowercased by the
        # caller), but the user's query is mixed-case.  Lowercase both
        # sides and run _anonymize_transcript with the same word-boundary
        # substitution the extraction path uses; the original text is
        # preserved for the return value.
        text_lower = text.lower()
        anonymized = _anonymize_transcript(text_lower, mapping)
        if anonymized != text_lower:
            findings.append("personal_entity")

    if speaker_id and _contains_first_person(text):
        if _is_interrogative(text):
            findings.append("self_referential")
        else:
            findings.append("personal_claim")

    return findings


def sanitize_for_cloud(
    text: str,
    mode: str = "warn",
    *,
    speaker_id: str | None = None,
    known_entities: set[str] | None = None,
) -> tuple[str | None, list[str]]:
    """Check query before sending to cloud.  Returns ``(query, findings)``.

    Args:
        text: query to check.
        mode: ``off`` (skip), ``warn`` (log + pass through),
            ``block`` (return ``None`` instead of the text when personal
            content is found).
        speaker_id: identifier of the resolved speaker, or ``None`` if the
            speaker has not been resolved.  Used to gate first-person
            interpretation — without an identified speaker there is no one
            for "I" / "my" to refer to.
        known_entities: lowercased set of entity / speaker names that count
            as personal references.  Typically assembled by the chat handler
            from ``router._all_entities`` plus enrolled
            ``SpeakerStore.speaker_names()``.

    Returns:
        ``(sanitized_query, findings)``.  When ``mode="block"`` and personal
        content is found, ``sanitized_query`` is ``None`` and the caller
        should fall back to the local model.
    """
    if mode == "off":
        return text, []

    findings = check_personal_content(text, speaker_id=speaker_id, known_entities=known_entities)
    if not findings:
        return text, []

    if mode == "warn":
        logger.warning("Personal content detected (mode=warn): %s — %s", findings, text[:100])
        return text, findings

    # mode == "block"
    logger.info("Blocked cloud escalation due to personal content: %s — %s", findings, text[:100])
    return None, findings
