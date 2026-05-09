"""Query sanitizer — graph-anchored personal-content gate before cloud escalation.

Anchored on runtime ground truth, not lexical patterns:

* The speaker's **known entities** (the router's entity index plus the speaker
  store's enrolled names) are the source of truth for what counts as personal.
  Detection reuses ``_anonymize_transcript`` from the extraction pipeline —
  the same primitive that produces SOTA-safe transcripts.  Anonymization
  replacing anything means the text contained a personal reference.
* **First-person pronouns** plus an identified speaker count as personal
  even when no known entity is named (covers cold-start before the graph
  has facts to anchor on).  The token set is explicit; there is no
  pattern matching.

Findings emitted (purely for diagnostics / mode=warn logging — production
routing reads :class:`paramem.server.router.Intent` instead):

* ``personal_entity`` — query mentions an entity in the speaker's graph or
  an enrolled name.
* ``first_person_personal`` — query contains a first-person pronoun and an
  identified ``speaker_id``.

Modes: ``off`` / ``warn`` / ``block``.
"""

import logging

from paramem.graph.extractor import _anonymize_transcript

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


def _is_about_speaker(text: str, personal_referent_config) -> bool:
    """Decide whether ``text`` refers to / asks about the speaker themselves.

    Two-tier detection, in order:

    1. **Encoder-based classifier** (when ``personal_referent_config`` is
       provided and the encoder + exemplar bank are loaded — production
       path).  Cosine vs multilingual exemplars + margin gate.  Returns
       the encoder's verdict directly when confidence is sufficient.
       Below the margin or on any classifier failure: the classifier
       returns ``None`` and we fall through to tier 2.
    2. **English token-set lookup** (legacy fallback) — frozenset
       membership against :data:`_FIRST_PERSON_TOKENS`.  Catches English
       first-person pronouns; misses non-English entirely.  Used only
       when tier 1 produced ``None``.

    The cost asymmetry is the same as the abstention path: a
    false-positive (sanitizer blocks a non-personal query) is mildly
    annoying but privacy-safe; a false-negative (sanitizer passes a
    personal query to the cloud) is the privacy hole the encoder layer
    exists to close.  Tier 1 generalises across languages via the
    multilingual encoder; tier 2 catches English-without-encoder.
    """
    if personal_referent_config is not None:
        from paramem.server.personal_referent import (
            PersonalReferent,
            classify_personal_referent,
        )

        verdict = classify_personal_referent(text, config=personal_referent_config)
        if verdict is PersonalReferent.ABOUT_SPEAKER:
            return True
        if verdict is PersonalReferent.NOT_ABOUT_SPEAKER:
            return False
        # verdict is None — encoder unavailable / margin not met.
        # Fall through to the English token-set fallback below.
    return _contains_first_person(text)


def check_personal_content(
    text: str,
    *,
    speaker_id: str | None = None,
    known_entities: set[str] | None = None,
    personal_referent_config=None,
) -> list[str]:
    """Return findings explaining why the text is personal.  Empty = clean.

    Two detection arms:

    * **Known-entity scrub** — the speaker's known entities (the router's
      entity index plus enrolled speaker names) are substituted via
      :func:`paramem.graph.extractor._anonymize_transcript`.  Substitution
      anywhere → personal.  Already language-agnostic (entity names are
      surface forms).
    * **Self-reference gate** — :func:`_is_about_speaker` decides whether
      the query refers to the speaker.  Production path uses the
      encoder-based classifier (``personal_referent_config`` provided +
      encoder loaded), legacy path uses the English token-set in
      :func:`_contains_first_person`.  The gate is bound by ``speaker_id``
      because a self-referential query with no resolved speaker has no
      target to be personal about.

    Back-compat: if neither ``speaker_id`` nor ``known_entities`` is supplied
    the function returns ``[]`` because there is nothing to anchor against.
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

    if speaker_id and _is_about_speaker(text, personal_referent_config):
        findings.append("first_person_personal")

    return findings


def sanitize_for_cloud(
    text: str,
    mode: str = "warn",
    *,
    speaker_id: str | None = None,
    known_entities: set[str] | None = None,
    personal_referent_config=None,
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
        personal_referent_config: optional
            :class:`paramem.server.config.PersonalReferentConfig`.  When
            supplied (production), the encoder-based classifier is used
            for the self-reference gate; otherwise the English token-set
            fallback (:func:`_contains_first_person`) is used.

    Returns:
        ``(sanitized_query, findings)``.  When ``mode="block"`` and personal
        content is found, ``sanitized_query`` is ``None`` and the caller
        should fall back to the local model.
    """
    if mode == "off":
        return text, []

    findings = check_personal_content(
        text,
        speaker_id=speaker_id,
        known_entities=known_entities,
        personal_referent_config=personal_referent_config,
    )
    if not findings:
        return text, []

    if mode == "warn":
        logger.warning("Personal content detected (mode=warn): %s — %s", findings, text[:100])
        return text, findings

    # mode == "block"
    logger.info("Blocked cloud escalation due to personal content: %s — %s", findings, text[:100])
    return None, findings
