"""Non-regex text-parsing helpers shared by the prompt-contract and
prompt-render test suites.

Both suites inspect prompt template text the same handful of ways:
splitting a template into its blank-line-delimited few-shot blocks,
extracting the JSON objects embedded in a block (or in an
already-rendered prompt) without assuming the surrounding text is
itself valid JSON, and recognising a first-person pronoun or a
``{word}``-shaped span left in already-rendered text. Each is written
as an explicit character/line scan rather than a regular expression.
"""

from __future__ import annotations

import json
import string

_ASCII_LETTERS = set(string.ascii_letters)
_ASCII_UPPER = set(string.ascii_uppercase)
_PLACEHOLDER_WORD_CHARS = _ASCII_LETTERS | {"_"}
_FIRST_PERSON_PRONOUNS = {"i", "me", "my", "we", "our"}


def split_blocks(text: str) -> list[str]:
    """Split *text* into its blank-line-delimited paragraphs.

    A paragraph break is one or more consecutive blank (whitespace-only)
    lines; runs of blank lines collapse to a single break, matching how
    the prompt files separate one few-shot example from the next.
    """
    blocks: list[str] = []
    current: list[str] = []
    for line in text.split("\n"):
        if line.strip() == "":
            if current:
                blocks.append("\n".join(current))
                current = []
            continue
        current.append(line)
    if current:
        blocks.append("\n".join(current))
    return blocks


def json_objects(text: str) -> list[dict]:
    """Every syntactically complete brace-delimited JSON *object* in *text*.

    Scans character by character, tracking JSON string state (so a brace
    written inside a quoted string never affects nesting) and a stack of
    open-brace positions. Whenever a ``}`` closes a matching ``{``, the
    resulting span is attempted as ``json.loads``. A doubled brace
    (``{{...}}``, the prompt files' ``str.format`` escape for a literal
    brace) always yields both an inner, well-formed single-brace span and
    an outer, unparsable doubled-brace span — the outer attempt fails to
    parse and is discarded, leaving the inner object intact. A non-JSON
    elision inside a worked "BAD output" example (e.g. the literal text
    ``, ...}}``) likewise fails to parse and is discarded, without
    preventing any other, complete object elsewhere in *text* from being
    recovered. Only dict results are returned — a bare JSON array or
    scalar span is not.
    """
    objects: list[dict] = []
    stack: list[int] = []
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            stack.append(i)
        elif ch == "}":
            if stack:
                start = stack.pop()
                candidate = text[start : i + 1]
                try:
                    parsed = json.loads(candidate)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    objects.append(parsed)
    return objects


def subject_values(text: str) -> set[str]:
    """Every distinct ``"subject"`` value literally present in *text*.

    Locates each occurrence of the literal key text ``"subject"`` by
    substring search, then decodes only the JSON value that immediately
    follows its ``:`` with ``json.JSONDecoder().raw_decode`` — independent
    of whether the surrounding object as a whole is valid JSON. This is
    deliberately not built on :func:`json_objects`: several worked
    examples in the prompt files write a fact literal with a trailing
    ``, ...}}`` elision (e.g. ``{{"subject":"Person_1", ...}}``) to show
    "more fields go here" without spelling every field out, which makes
    the enclosing object as a whole fail ``json.loads`` — but the
    ``"subject"`` value itself, sitting before the elision, is still
    well-formed JSON on its own and is exactly the value this function
    must not miss.
    """
    decoder = json.JSONDecoder()
    subjects: set[str] = set()
    key = '"subject"'
    search_from = 0
    while True:
        key_start = text.find(key, search_from)
        if key_start == -1:
            break
        pos = key_start + len(key)
        while pos < len(text) and text[pos] in " \t\n\r":
            pos += 1
        if pos < len(text) and text[pos] == ":":
            pos += 1
            while pos < len(text) and text[pos] in " \t\n\r":
                pos += 1
        search_from = key_start + len(key)
        try:
            value, _end = decoder.raw_decode(text, pos)
        except json.JSONDecodeError:
            continue
        if isinstance(value, str):
            subjects.add(value)
    return subjects


def is_ascii_letter_led(value: str) -> bool:
    """True if *value* is non-empty and its first character is an ASCII letter."""
    return bool(value) and value[0] in _ASCII_LETTERS


def has_non_speaker_alpha_subject(text: str) -> bool:
    """True if some JSON object embedded in *text* has a letter-led subject
    other than ``"speaker0"``."""
    return any(
        subject != "speaker0" and is_ascii_letter_led(subject) for subject in subject_values(text)
    )


def has_glued_possessive_object(text: str) -> bool:
    """True if some JSON object's ``"object"`` string value contains a
    glued possessive — the substring ``"'s "`` (e.g. "Theo's orchids")."""
    return any(
        isinstance(obj.get("object"), str) and "'s " in obj["object"] for obj in json_objects(text)
    )


def is_capitalized_word(word: str) -> bool:
    """True if *word* is non-empty, ASCII-letters-only, and starts uppercase."""
    return bool(word) and word[0] in _ASCII_UPPER and all(c in _ASCII_LETTERS for c in word)


def is_proper_name(value: str) -> bool:
    """True if *value* is one or more single-space-separated Capitalized words."""
    words = value.split(" ")
    return all(is_capitalized_word(w) for w in words)


def double_quoted_spans(text: str) -> list[str]:
    """Every double-quoted span in *text*, paired left to right."""
    spans: list[str] = []
    start: int | None = None
    for i, ch in enumerate(text):
        if ch != '"':
            continue
        if start is None:
            start = i + 1
        else:
            spans.append(text[start:i])
            start = None
    return spans


def has_first_person_pronoun(span: str) -> bool:
    """True if *span* contains a whole-word first-person pronoun
    (``I``/``me``/``my``/``we``/``our``, case-insensitive)."""
    words: list[str] = []
    current: list[str] = []
    for ch in span:
        if ch.isalpha():
            current.append(ch)
        elif current:
            words.append("".join(current))
            current = []
    if current:
        words.append("".join(current))
    return any(word.lower() in _FIRST_PERSON_PRONOUNS for word in words)


def example_json_objects(text: str) -> list[dict]:
    """Parse the JSON object trailing each line beginning with "Example"."""
    objects: list[dict] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("Example"):
            continue
        _prefix, sep, remainder = stripped.partition(":")
        if not sep:
            continue
        remainder = remainder.strip()
        if remainder.startswith("{"):
            objects.append(json.loads(remainder))
    return objects


def leading_rule_id(line: str) -> str | None:
    """The rule id (e.g. ``"R4"``) if *line* starts with ``R<digits>.``, else ``None``."""
    if not line.startswith("R"):
        return None
    i = 1
    while i < len(line) and line[i].isdigit():
        i += 1
    if i == 1 or i >= len(line) or line[i] != ".":
        return None
    return line[:i]


def stray_placeholders(
    text: str, *, intentional_literals: frozenset[str] = frozenset()
) -> list[str]:
    """Every ``{word}``-shaped span literally present in already-rendered
    *text*, excluding a doubled ``{{`` escape and any declared literal.

    A span is a ``{`` not immediately preceded by another ``{``, followed
    by one or more ASCII letters/underscores, followed by ``}`` — the
    shape of a ``str.format`` slot that survived rendering (an
    un-threaded field, or a doubled-brace escape such as
    ``{{speaker_name}}`` that was meant to stay literal but reads as an
    unrendered slot), as opposed to a literal JSON brace (which is always
    followed by a quote, bracket, digit, or another brace, never bare
    word text).

    This scans *text*'s own characters directly rather than through any
    format-string parser, so a stray span nested inside a JSON worked
    example's outer brace pair (e.g. ``{"subject": "{speaker_name}", ...}``)
    is found on its own — a format-string parser would fold it into the
    surrounding field's format spec and miss it.

    *intentional_literals* names braced tokens (e.g. ``"{SPEAKER_NAME}"``)
    that are deliberate literal output, not a stray placeholder.
    """
    matches: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] == "{" and (i == 0 or text[i - 1] != "{"):
            j = i + 1
            while j < n and text[j] in _PLACEHOLDER_WORD_CHARS:
                j += 1
            if j > i + 1 and j < n and text[j] == "}":
                matches.append(text[i : j + 1])
                i = j + 1
                continue
        i += 1
    return sorted(m for m in matches if m not in intentional_literals)
