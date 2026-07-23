"""String-normalization primitives — two separate families, one module.

This module owns BOTH normalization families used project-wide:

* :func:`canonical` — the **identity** fold, for comparing/hashing whole
  strings as keys (node keys, dedup, fuzzy match, the SimHash fingerprint).
* :func:`prose_fold` — the **prose-matching** fold, for case-insensitive
  substring/token matching against free text (entity-name matching in HA
  routing, interrogative-prefix detection, first-person pronoun detection).
  It is deliberately weaker than :func:`canonical`: it must NOT fold
  whitespace to ``_`` (that would break matching a multi-word name embedded
  in space-separated prose), and it must NOT go beyond simple
  case-lowering (one consumer is the sanitizer's first-person PII gate,
  where casefold/diacritic-folding would change privacy-relevant matching
  behavior — out of scope for that fold).

:class:`paramem.graph.merger.GraphMerger` applies :func:`canonical` at the
node-identity boundary to produce deterministic node keys; the server,
training, memory, evaluation and cli packages route every identity/dedup
comparison through the same two ``canonical``-family functions.  Name
*matching* (fuzzy/surface tiers) is a separate concern and lives in the
merger, not here.  Stateless, deterministic, no I/O.  Thread-safe by
construction.

One surface form, project-wide: lower-case, blank runs collapsed to a single
``_``, ``-`` and ``_`` preserved verbatim.  This is the identity form — the
node key, the stored predicate, the SimHash and dedup key.  It is NOT the
display surface: rendering to prose happens at
:func:`paramem.memory.entry.entry_fact_text`, which turns ``_`` back into a
space (``has_hobby`` → ``"has hobby"``).  Matching names against free prose is
a separate contract that does not call this function at all.

Over-collapse boundary (GUARANTEED-identical-only):
  Folded: Unicode canonical form, case (incl. ligatures `str.casefold` covers,
  e.g. ``ﬁ``→``fi``), diacritics, whitespace runs (→ ``_``).
  NOT folded: ``-`` vs ``_`` (``sister-in-law`` ≠ ``sister_in_law``), typos
  (HR3 ≠ HS3), substrings, NFKC compatibility forms (superscript ``²``,
  full-width), honorifics, token subsets.  Those are layer-2 LLM SAME_AS
  coreference, out of scope here.

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

# Compiled once: matches the system speaker-id format "speaker{N}" (the
# canonical lowercase form) where N is one or more decimal digits.
# Case-insensitive so a residual cased form like "Speaker0" also passes the
# structural test — the regex tolerates the cased form as input; it is never
# the canonical form.
_SPEAKER_ID_RE = re.compile(r"^[Ss]peaker\d+$")


def canonical(s: str) -> str:
    """The single surface form for entity names, objects, and predicates.

    Lower-cases (Unicode full case folding, §3.13), folds diacritics, and
    collapses whitespace runs to a single ``_``.  ``-`` and ``_`` are preserved
    verbatim — they are NOT interchangeable.  ``"has hobby"`` and
    ``"has_hobby"`` are one value, ``has_hobby``.

    The output is the **identity key** — what is stored, compared, keyed and
    trained.  It is NOT the display surface: rendering back to prose happens at
    :func:`~paramem.memory.entry.entry_fact_text`, which emits ``_`` as a space
    (``"Alex has sister-in-law Mia"``).  Matching entity names against free
    prose is a separate contract that does not use this function at all.

    GUARANTEED-identical-only.  Folds: Unicode canonical form, case (incl. the
    few ligatures ``str.casefold`` covers, e.g. ``ﬁ``→``fi``), diacritics,
    whitespace runs.  Does NOT fold: ``-`` against ``_`` (``sister-in-law`` ≠
    ``sister_in_law``), typos (HR3 ≠ HS3), substrings (autonomous_systems ≠
    autonomous_systems_research), NFKC compatibility forms (superscript ``²``,
    full-width), honorifics, token subsets — those are layer-2 LLM SAME_AS
    coreference, out of scope.

    Args:
        s: Raw string to canonicalize.  May be a name, predicate, or object.

    Returns:
        Lower-case, NFC-normalized string with diacritics folded and blank runs
        replaced by single underscores.  ``""`` for empty or whitespace-only
        input.

    Examples::

        >>> canonical("has hobby") == canonical("has_hobby")
        True
        >>> canonical("sister-in-law")
        'sister-in-law'
        >>> canonical("HELLO World")
        'hello_world'

    Order rationale (Unicode §3.13): NFC before casefold handles the U+0345
    family that casefolds correctly only after decomposition; re-NFC after
    casefold + combining-strip restores a stable form so ``f(f(x)) == f(x)``.
    Whitespace folding is last — it only consumes ASCII/Unicode blanks and
    emits ``_``, which ``str.split()`` does not treat as a separator, so
    re-running is a no-op.  Idempotent and stable across runs for a fixed
    CPython build.
    """
    if not s:
        return ""
    s = unicodedata.normalize("NFC", s)
    s = s.casefold()
    s = "".join(ch for ch in unicodedata.normalize("NFD", s) if not unicodedata.combining(ch))
    s = unicodedata.normalize("NFC", s)
    return "_".join(s.split())


def prose_fold(text: str) -> str:
    """Case-insensitive fold for matching a needle against free prose.

    This is the **prose-matching** counterpart to :func:`canonical` — the
    two families are not interchangeable.  ``canonical`` folds whitespace to
    ``_``, which breaks substring/token matching against space-separated
    prose (a folded multi-word entity name would never be found embedded in
    a sentence).  ``prose_fold`` deliberately does neither: it does NOT fold
    whitespace, and it does NOT go beyond ``str.lower()`` (no casefold, no
    diacritic-folding).  One consumer is the sanitizer's first-person PII
    gate (:mod:`paramem.server.sanitizer`), where changing case/diacritic
    matching behavior would alter privacy-relevant matching — out of scope
    for this fold.

    Args:
        text: Free-text string (or a token/needle extracted from one) to
            fold for case-insensitive comparison.

    Returns:
        ``text.lower()`` — behavior-preserving; identical to what every
        prose-matching call site already did before routing through this
        function.
    """
    return text.lower()


# ---------------------------------------------------------------------------
# Speaker-identity primitives — §0 invariant
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
        ``"speaker"``, ``"alex"``, ``"SpeakerX"``) and empty strings.

    Examples::

        >>> is_speaker_id("speaker0")
        True
        >>> is_speaker_id("speaker12")
        True
        >>> is_speaker_id("Speaker0")
        True
        >>> is_speaker_id("Speaker")
        False
        >>> is_speaker_id("alex")
        False
    """
    if not s:
        return False
    return bool(_SPEAKER_ID_RE.match(s))
