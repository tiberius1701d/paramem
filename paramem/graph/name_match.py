"""String-identity canonicalization for entity names, objects, and predicates.

Used by :class:`paramem.graph.merger.GraphMerger` to produce deterministic node
keys.  Stateless, deterministic, no I/O.  Thread-safe by construction.

Over-collapse boundary (GUARANTEED-identical-only):
  Folded: Unicode canonical form, case (incl. ligatures `str.casefold` covers,
  e.g. ``ﬁ``→``fi``), diacritics, ``_``/``-``/whitespace runs.
  NOT folded: typos (HR3 ≠ HS3), substrings, NFKC compatibility forms
  (superscript ``²``, full-width), honorifics, token subsets.  Those are
  layer-2 LLM SAME_AS coreference, out of scope here.

Speaker-identity
----------------
Speaker ids are ONE canonical lowercase form: ``speaker{N}``
(e.g. ``"speaker0"``).  This is the form stored as the graph node key, the
profile key in :class:`~paramem.server.speaker.SpeakerStore`, the
``speaker_id`` attribute on graph nodes, and the ``speaker_id`` field on
:class:`~paramem.graph.schema.Relation` and :class:`~paramem.memory.entry`
objects.  Speaker equality is plain ``==`` — no bridging function is needed.
:func:`is_speaker_id` is the structural gate for the ``speaker{N}`` format.
The ingest safety-net in :func:`~paramem.graph.extractor._normalize_extraction`
lowercases any matching token at the extraction boundary (a deliberate, scoped
exception to the "extraction only `.strip()`s" rule for display entities).
"""

import re
import unicodedata

# Built once at module load.  ``_``/``-`` → space; space-run collapse happens
# in the ``str.split()`` call so no special entry for multiple spaces is needed.
_SEP = str.maketrans({"_": " ", "-": " "})

# Compiled once: matches the system speaker-id format "speaker{N}" (the
# canonical lowercase form) where N is one or more decimal digits.
# Case-insensitive so a residual cased form like "Speaker0" also passes the
# structural test — the regex tolerates the cased form as input; it is never
# the canonical form.
_SPEAKER_ID_RE = re.compile(r"^[Ss]peaker\d+$")


def canonical(s: str) -> str:
    """Deterministic identity key for entity names, objects, and predicates.

    Unicode-standard canonical caseless matching (Unicode §3.13), diacritic-
    insensitive, separator- and whitespace-normalized.  Drives matching/dedup
    ONLY; never overwrites the stored/displayed surface form.

    GUARANTEED-identical-only.  Folds: Unicode canonical form, case (incl. the
    few ligatures ``str.casefold`` covers, e.g. ``ﬁ``→``fi``), diacritics,
    ``_``/``-``/whitespace runs.  Does NOT fold: typos (HR3 ≠ HS3), substrings
    (autonomous_systems ≠ autonomous_systems_research), NFKC compatibility forms
    (superscript ``²``, full-width), honorifics, token subsets — those are
    layer-2 LLM SAME_AS coreference, out of scope.

    Args:
        s: Raw string to canonicalize.  May be a name, predicate, or object.

    Returns:
        Canonical form as a lower-case, NFC-normalized string with separators
        and diacritics folded.  Returns ``""`` for empty or whitespace-only input.

    Order rationale (Unicode §3.13): NFC before casefold handles the U+0345
    family that casefolds correctly only after decomposition; re-NFC after
    casefold + combining-strip restores a stable form so ``f(f(x)) == f(x)``.
    Separator/whitespace folding is last — it only touches ASCII
    space/``_``/``-``, which no normalization step reintroduces, so it does not
    disturb the Unicode invariants and re-running is a no-op.  Idempotent and
    stable across runs for a fixed CPython build.
    """
    if not s:
        return ""
    s = unicodedata.normalize("NFC", s)
    s = s.casefold()
    s = "".join(ch for ch in unicodedata.normalize("NFD", s) if not unicodedata.combining(ch))
    s = unicodedata.normalize("NFC", s)
    s = s.translate(_SEP)
    return " ".join(s.split())


# ---------------------------------------------------------------------------
# Speaker-identity primitives (P1 / P2) — §0 invariant
# ---------------------------------------------------------------------------


def is_speaker_id(s: str) -> bool:
    """Return ``True`` when *s* is a speaker-id of the form ``speaker{N}``.

    The canonical stored form is lowercase (``"speaker0"``, ``"speaker12"``).
    The regex keeps ``[Ss]`` so the ingest safety-net can also detect and
    coerce any residual cased form (``"Speaker0"``) that a model emits — the
    coercion output is ALWAYS lowercase.  The structural test is purely
    syntactic — it does NOT check whether the id corresponds to a registered
    speaker.

    Args:
        s: String to test.

    Returns:
        ``True`` iff *s* matches ``[Ss]peaker`` followed by one or more decimal
        digits and nothing else.  ``False`` for partial matches (``"Speaker"``,
        ``"speaker"``, ``"tobias"``, ``"SpeakerX"``) and empty strings.

    Examples::

        >>> is_speaker_id("speaker0")
        True
        >>> is_speaker_id("speaker12")
        True
        >>> is_speaker_id("Speaker0")
        True
        >>> is_speaker_id("Speaker")
        False
        >>> is_speaker_id("tobias")
        False
    """
    if not s:
        return False
    return bool(_SPEAKER_ID_RE.match(s))
