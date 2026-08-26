"""The ``[user]`` / ``[assistant]`` transcript turn-marker vocabulary.

One owner for the marker surface every extraction/anonymization few-shot
is calibrated on. Production mints the marker for the session transcript
at :meth:`~paramem.server.session_buffer.SessionBuffer._format_turns`,
which calls :func:`format_turn`; a second consumer (the cloud-egress
payload assembler, which cannot import ``paramem.server``) reaches the
same vocabulary through this leaf module. Stateless, deterministic, no
I/O.
"""

import re

_MARKER_RE = re.compile(r"^\[[^\[\]]+\] ")


def format_turn(role: str, text: str) -> str:
    """Render one conversation turn as a marker-prefixed line.

    Produces ``"[<role>] <text>"`` — e.g. ``format_turn("user", "hello")``
    returns ``"[user] hello"``. ``role`` is written verbatim inside the
    brackets (no vocabulary restriction to ``user``/``assistant``); this
    matches, byte-for-byte, the marker
    :meth:`~paramem.server.session_buffer.SessionBuffer._format_turns`
    produces for every role.
    """
    return f"[{role}] {text}"


def split_marker(line: str) -> tuple[str, str]:
    """Split a marker-prefixed line into its marker and text.

    Returns ``(marker, text)`` where ``marker`` is the leading
    ``"[<role>] "`` (brackets and the one trailing space included) and
    ``text`` is everything after it, so ``marker + text == line`` always
    holds — this is the exact inverse of :func:`format_turn`. A line with
    no leading ``[<role>] `` marker (e.g. no brackets, or brackets not at
    the start of the line) is not a marker line: this returns
    ``("", line)``, so the round-trip identity still holds trivially.
    """
    match = _MARKER_RE.match(line)
    if match is None:
        return "", line
    marker = match.group(0)
    return marker, line[len(marker) :]
