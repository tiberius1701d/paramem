"""Serving-path prompt accessors — the only module naming a serving prompt file.

``inference.py``, ``intent.py``, and ``temporal_selection.py`` import named
functions from here rather than filename/section-name literals of their own,
so a serving prompt's basename or section name has exactly one occurrence in
the codebase. This is not a second loader: every function body below is a
call to :func:`paramem.graph.prompts._load_prompt` or
:func:`paramem.graph.prompts._load_prompt_section` — the single chokepoint
every model-facing prompt in the codebase resolves through.

Serving prompts do NOT thread ``prompts_dir``: they always resolve through
the loader's guaranteed final search directory (``configs/prompts/`` in the
checkout). Operators tune them by editing the file in place — the same
posture every extraction prompt already has, and the only self-consistent
one across the three serving call paths (only one of the three has a
``ServerConfig`` in hand at the call site). Because loads happen at call
time, an edit to any file here takes effect on the next request — no server
restart required.
"""

from paramem.graph.prompts import _load_prompt, _load_prompt_section

_SERVING_SYSTEM_FILE = "serving_system.txt"
_SERVING_DIRECTIVES_FILE = "serving_directives.txt"
_CLOUD_SERVING_SYSTEM_FILE = "cloud_serving_system.txt"
_INTENT_CLASSIFIER_FILE = "intent_classifier.txt"
_RECALL_SELECTION_FILE = "recall_selection.txt"


def serving_system_prompt() -> str:
    """Load the local reasoning leg's base system prompt, verbatim.

    Returns:
        The contents of ``configs/prompts/serving_system.txt``.
    """
    return _load_prompt(_SERVING_SYSTEM_FILE)


def cloud_serving_system_prompt() -> str:
    """Load the cloud reasoning leg's base system prompt, verbatim.

    Carries no identity line and no speaker token — the cloud system
    prompt is deliberately identity-free (see
    :func:`paramem.server.inference._escalate_to_cloud`).

    Returns:
        The contents of ``configs/prompts/cloud_serving_system.txt``.
    """
    return _load_prompt(_CLOUD_SERVING_SYSTEM_FILE)


def intent_classifier_prompt() -> str:
    """Load the intent classifier's system prompt, verbatim.

    Returns:
        The contents of ``configs/prompts/intent_classifier.txt``.
    """
    return _load_prompt(_INTENT_CLASSIFIER_FILE)


def recall_selection_prompt() -> str:
    """Load the date-group selection stage's system prompt, verbatim.

    Contains literal JSON example braces (``{"all": true}`` and friends) —
    this content is passed to the model as-is and must NEVER be run
    through ``str.format()``; a stray literal brace elsewhere in the file
    would otherwise raise ``KeyError`` at request time.

    Returns:
        The contents of ``configs/prompts/recall_selection.txt``.
    """
    return _load_prompt(_RECALL_SELECTION_FILE)


def identity_line(speaker_id: str) -> str:
    """Render the "You are speaking with X." identity line.

    Args:
        speaker_id: The resolved ``speaker{N}`` token from the routing
            layer — never a display name; identity stays in token space
            in every model-facing surface.

    Returns:
        The rendered identity line, e.g. ``"You are speaking with speaker0."``.
    """
    return _load_prompt_section(_SERVING_DIRECTIVES_FILE, "IDENTITY-LINE").format(
        speaker_id=speaker_id
    )


def language_line(language_name: str) -> str:
    """Render the "Respond in X." language instruction line.

    Args:
        language_name: The display name of the target language, e.g.
            ``config.tts.language_name(language)`` or
            ``ISO_LANGUAGE_NAMES[language]``.

    Returns:
        The rendered language line, e.g. ``"Respond in German."``.
    """
    return _load_prompt_section(_SERVING_DIRECTIVES_FILE, "LANGUAGE-LINE").format(
        language_name=language_name
    )


def reasoning_turn(*, today_prefix: str, context: str, question: str) -> str:
    """Render the full reasoning-turn user prompt.

    Args:
        today_prefix: ``""`` or ``f"Today is {weekday_name(today)}, "
            f"{today.isoformat()}. "`` — built by
            :func:`paramem.server.inference._render_augmented_text`.
        context: The assembled tier/note body (may be empty).
        question: The user's current turn.

    Returns:
        The rendered reasoning-turn prompt.
    """
    return _load_prompt_section(_SERVING_DIRECTIVES_FILE, "REASONING-TURN").format(
        today_prefix=today_prefix, context=context, question=question
    )


def recorded_dates_suffix(dates: str) -> str:
    """Render the trailing "Recorded dates: ..." clause of the empty-period note.

    Args:
        dates: Comma-joined ISO date strings, e.g.
            ``"2026-01-01, 2026-01-02"``.

    Returns:
        The suffix text WITH the leading separator space required to
        concatenate directly onto :func:`empty_period_note`'s return value
        — e.g. ``" Recorded dates: 2026-01-01, 2026-01-02."``. The section
        text in ``serving_directives.txt`` cannot carry that leading space
        itself (every section is stripped on load), so it is prepended
        here to preserve byte parity with the pre-externalization string.
    """
    return " " + _load_prompt_section(_SERVING_DIRECTIVES_FILE, "RECORDED-DATES-SUFFIX").format(
        dates=dates
    )


def empty_period_note() -> str:
    """Load the "nothing was recorded" note, verbatim.

    Returns:
        The contents of the ``EMPTY-PERIOD-NOTE`` section of
        ``configs/prompts/serving_directives.txt``.
    """
    return _load_prompt_section(_SERVING_DIRECTIVES_FILE, "EMPTY-PERIOD-NOTE")
