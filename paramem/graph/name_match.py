"""String-identity canonicalization for entity names, objects, and predicates.

Used by :class:`paramem.graph.merger.GraphMerger` to produce deterministic node
keys.  Stateless, deterministic, no I/O.  Thread-safe by construction.

Over-collapse boundary (GUARANTEED-identical-only):
  Folded: Unicode canonical form, case (incl. ligatures `str.casefold` covers,
  e.g. ``ﬁ``→``fi``), diacritics, ``_``/``-``/whitespace runs.
  NOT folded: typos (HR3 ≠ HS3), substrings, NFKC compatibility forms
  (superscript ``²``, full-width), honorifics, token subsets.  Those are
  layer-2 LLM SAME_AS coreference, out of scope here.

Speaker-identity primitives (P1 / P2)
--------------------------------------
Speaker nodes are keyed by the **casefolded** form of their system id
(``canonical_speaker("Speaker0") == "speaker0"``).  All speaker-identity
equality tests must route through :func:`speaker_ref_matches` — the single
canonical comparison — so that a cased id (``"Speaker0"``) and a casefolded
node key (``"speaker0"``) are always treated as the same identity.

§0 invariant: every "is this the speaker?" or "do two speaker references
match?" test is ``canonical(a) == canonical(b)`` via :func:`speaker_ref_matches`.
There must be exactly ONE place in the codebase allowed to compare a
``Relation.subject`` against a ``Relation.speaker_id``, and it MUST call this
function.
"""

import re
import unicodedata

# Built once at module load.  ``_``/``-`` → space; space-run collapse happens
# in the ``str.split()`` call so no special entry for multiple spaces is needed.
_SEP = str.maketrans({"_": " ", "-": " "})

# Compiled once: matches the system speaker-id format "Speaker{N}" where N is
# one or more decimal digits.  Case-insensitive so "speaker0" (casefolded node
# key) also passes the structural test.
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
    """Return ``True`` when *s* is a system speaker-id of the form ``Speaker{N}``.

    Accepts both the cased form emitted by the speaker store (``"Speaker0"``,
    ``"Speaker12"``) and the casefolded node-key form (``"speaker0"``).  The
    structural test is purely syntactic — it does NOT check whether the id
    corresponds to a registered speaker.

    Args:
        s: String to test.

    Returns:
        ``True`` iff *s* matches ``[Ss]peaker`` followed by one or more decimal
        digits and nothing else.  ``False`` for partial matches (``"Speaker"``,
        ``"speaker"``, ``"tobias"``, ``"SpeakerX"``) and empty strings.

    Examples::

        >>> is_speaker_id("Speaker0")
        True
        >>> is_speaker_id("speaker12")
        True
        >>> is_speaker_id("Speaker")
        False
        >>> is_speaker_id("tobias")
        False
    """
    if not s:
        return False
    return bool(_SPEAKER_ID_RE.match(s))


def speaker_ref_matches(a: str, b: str) -> bool:
    """Return ``True`` when *a* and *b* refer to the same speaker identity.

    This is the **single canonical comparison** for all speaker-identity
    equality tests (§0 invariant).  It applies :func:`canonical` to both
    sides so that a cased id (``"Speaker0"``) and the casefolded node key
    (``"speaker0"``) are always treated as the same identity.

    Every "is this the speaker?" test that compares two speaker-related
    strings (e.g. ``Relation.subject`` vs ``Relation.speaker_id``) MUST
    route through this function instead of raw ``==``.

    Args:
        a: First speaker reference (may be cased or casefolded).
        b: Second speaker reference (may be cased or casefolded).

    Returns:
        ``True`` iff ``canonical(a) == canonical(b)``.

    Examples::

        >>> speaker_ref_matches("Speaker0", "speaker0")
        True
        >>> speaker_ref_matches("Speaker0", "Speaker1")
        False
        >>> speaker_ref_matches("Speaker12", "SPEAKER12")
        True
    """
    return canonical(a) == canonical(b)


def canonical_speaker(speaker_id: str) -> str:
    """Map a speaker id to its graph node key (casefolded canonical form).

    This is the **single function** that converts a speaker id to the node key
    used in the graph.  It is idempotent: applying it to an already-casefolded
    key returns the same key unchanged.

    Node keys for speakers are always the casefolded form so that a
    ``Speaker0`` Entity (from the entity path) and a ``Speaker0`` relation
    endpoint (from the fallback path) always resolve to the same node key
    (``"speaker0"``), preventing the casing-collision dup.

    Args:
        speaker_id: A speaker system id, either cased (``"Speaker0"``) or
            already casefolded (``"speaker0"``).

    Returns:
        The casefolded node key (e.g. ``"speaker0"``).  Returns ``""`` for
        empty input.

    Examples::

        >>> canonical_speaker("Speaker0")
        'speaker0'
        >>> canonical_speaker("speaker0")
        'speaker0'
        >>> canonical_speaker("Speaker12")
        'speaker12'
    """
    return canonical(speaker_id)
