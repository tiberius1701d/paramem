"""Self-reference gate before cloud escalation.

:func:`is_self_referential` decides whether a piece of text refers to /
asks about the identified speaker.  It is one of two arms that feed the
``is_personal`` verdict computed once in ``handle_chat``
(:mod:`paramem.server.inference`) — the other arm is the intent
classifier.  There is no policy knob here — what a caller DOES about a
personal verdict (drop the turn, suppress the cloud call, abstain) is
the caller's decision.

Detection is two-tier, in order:

1. **Encoder-based classifier** (:func:`paramem.server.personal_referent.
   classify_personal_referent`) when a config is supplied and the encoder
   + exemplar bank are loaded — the production path.  Multilingual.
2. **First-person token-set lookup** (:func:`_contains_first_person`) —
   English-only fallback used when tier 1 is unavailable or uncertain.

Content-only: the predicate classifies purely from ``text`` — it does not
accept or require a ``speaker_id``.  Whether there is a resolved speaker to
apply the verdict to is a separate concern the caller owns (e.g.
:func:`paramem.server.inference.handle_chat` asserts a resolved
``speaker_id`` before it ever reaches a call site that reads this verdict —
speakerless requests are served by the relay path, which never consults
parametric memory or this predicate's verdict as a personal-content gate).
"""

from paramem.utils.identity import prose_fold

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
        token = prose_fold(raw.strip(_PUNCT))
        if token in _FIRST_PERSON_TOKENS:
            return True
    return False


def is_self_referential(
    text: str,
    *,
    personal_referent_config=None,
) -> bool:
    """Return True when ``text`` refers to / asks about the speaker themselves.

    Content-only classification — there is no ``speaker_id`` parameter and
    no null-target gate.  A caller that needs to know "personal content
    about *whom*" resolves the speaker separately; this predicate only
    answers "is this text self-referential in content".

    THE one implementation of the personal-referent verdict — every caller
    (``handle_chat``'s ``is_personal`` union, the relay path's no-identity
    short-circuit in ``paramem.server.app._relay_route``, the forwarded-query
    verdict in ``_maybe_escalate``, the history drop-gate in
    ``_sanitize_history``) imports this name directly rather than
    re-deriving the conjunction below.  There used to be a second, private
    ``_is_about_speaker`` wrapper with the identical body; it is retired —
    a caller that imported it privately was re-deriving logic already
    public here.

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

    Args:
        text: The text to classify — a chat turn, verbatim and unmodified.
        personal_referent_config: Optional
            :class:`paramem.server.config.PersonalReferentConfig`.  When
            supplied (production), the encoder-based classifier drives the
            decision; otherwise the English token-set fallback
            (:func:`_contains_first_person`) does.

    Returns:
        ``True`` when the text is self-referential in content; ``False``
        otherwise.
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
