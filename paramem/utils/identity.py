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

One surface form, project-wide: lower-case, blank runs (including ``_``)
collapsed to a single space, ``-`` preserved verbatim.  This is the identity
form — the node key, the stored predicate, the SimHash and dedup key.  It IS
also the display surface for the predicate: :func:`paramem.memory.entry.entry_fact_text`
renders the predicate as-is (no ``_``→space substitution happens there any
more, since the identity form already uses spaces).  Matching names against
free prose is a separate contract that does not call this function at all.

:func:`canonical` takes a ``mode`` argument selecting between two folds:
``"full"`` (default) — case-fold + diacritic-fold + space-fold, the identity
key used everywhere above — and ``"spaces"`` — space-fold only (``_``/blank
runs → single space), case and exact surface otherwise preserved.  ``mode="spaces"``
exists for the SimHash register/recall symmetry fold in
:func:`paramem.memory.entry.entry_simhash` / :func:`~paramem.memory.entry.verify_confidence`,
where diacritic/NFC distinctions must stay discriminating (``"café"`` ≠
``"cafe"`` in that comparison) but a ``_``↔space recall drift on the same
fact must not desync the fingerprint.  (The SimHash tokenizer lowercases, so
case never reaches the fingerprint; ``mode="full"`` would additionally fold
diacritics and collapse ``café``/``cafe`` — hence ``"spaces"`` there.)

Over-collapse boundary (GUARANTEED-identical-only, ``mode="full"``):
  Folded: Unicode canonical form, case (incl. ligatures `str.casefold` covers,
  e.g. ``ﬁ``→``fi``), diacritics, whitespace runs including ``_`` (→ single
  space).
  NOT folded: ``-`` (``sister-in-law`` ≠ ``sister in law``), typos
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

import unicodedata

# THE single declaration of the speaker-id format: the canonical lowercase
# prefix followed by a decimal index.  Both the mint
# (``server.speaker.SpeakerStore``) and the structural gate
# (:func:`is_speaker_id`) compose from this constant, so the shape is never
# rendered twice.
SPEAKER_ID_PREFIX = "speaker"


def canonical(s: str, mode: str = "full") -> str:
    """The single surface form for entity names, objects, and predicates.

    Two modes, selecting how much folding is applied:

    * ``mode="full"`` (default) — lower-cases (Unicode full case folding,
      §3.13), folds diacritics, and collapses ``_``/whitespace runs to a
      single space.  ``-`` is preserved verbatim — it is NOT a blank.
      ``"has hobby"`` and ``"has_hobby"`` are one value, ``"has hobby"``.
      This is the **identity key** — what is stored, compared, keyed and
      trained (node keys, predicates, the SimHash/dedup key).  It is also the
      display surface for the predicate: :func:`~paramem.memory.entry.entry_fact_text`
      renders it as-is, no further substitution.  Matching entity names
      against free prose is a separate contract (:func:`prose_fold`) that
      does not use this function at all.
    * ``mode="spaces"`` — folds ONLY ``_``/whitespace runs to a single space;
      no casefold, no diacritic-fold, exact surface otherwise preserved.
      Used where a ``_``↔space recall drift on the same fact must not desync a
      fingerprint, but diacritic/NFC distinctions must stay discriminating
      (e.g. the SimHash register/recall fold in
      :func:`~paramem.memory.entry.entry_simhash` /
      :func:`~paramem.memory.entry.verify_confidence`, where ``"café"`` must
      still differ from ``"cafe"``; the tokenizer lowercases, so ``"full"``'s
      extra diacritic-fold would wrongly collapse them).

    GUARANTEED-identical-only (``mode="full"``).  Folds: Unicode canonical
    form, case (incl. the few ligatures ``str.casefold`` covers, e.g.
    ``ﬁ``→``fi``), diacritics, whitespace runs including ``_``.  Does NOT
    fold: ``-`` (``sister-in-law`` ≠ ``sister in law``), typos (HR3 ≠ HS3),
    substrings (autonomous systems ≠ autonomous systems research), NFKC
    compatibility forms (superscript ``²``, full-width), honorifics, token
    subsets — those are layer-2 LLM SAME_AS coreference, out of scope.

    Args:
        s: Raw string to canonicalize.  May be a name, predicate, or object.
        mode: ``"full"`` (default) for the identity fold, or ``"spaces"`` for
            the space-only fold (case/diacritics preserved).

    Returns:
        Normalized string with blank runs (``_``/whitespace) collapsed to a
        single space; under ``mode="full"`` also NFC-normalized, lower-cased
        and diacritic-folded.  ``""`` for empty or whitespace-only input.

    Examples::

        >>> canonical("has hobby") == canonical("has_hobby")
        True
        >>> canonical("has hobby")
        'has hobby'
        >>> canonical("sister-in-law")
        'sister-in-law'
        >>> canonical("HELLO World")
        'hello world'
        >>> canonical("New_York", mode="spaces")
        'New York'

    Order rationale (Unicode §3.13, ``mode="full"``): NFC before casefold
    handles the U+0345 family that casefolds correctly only after
    decomposition; re-NFC after casefold + combining-strip restores a stable
    form so ``f(f(x)) == f(x)``.  Whitespace/underscore folding is last — it
    only consumes ``_`` and Unicode blanks and emits a single space, which
    ``str.split()`` treats as a separator, so re-running is a no-op.
    Idempotent and stable across runs for a fixed CPython build, in both
    modes.
    """
    if not s:
        return ""
    if mode == "spaces":
        return " ".join(s.replace("_", " ").split())
    s = unicodedata.normalize("NFC", s)
    s = s.casefold()
    s = "".join(ch for ch in unicodedata.normalize("NFD", s) if not unicodedata.combining(ch))
    s = unicodedata.normalize("NFC", s)
    return " ".join(s.replace("_", " ").split())


def prose_fold(text: str) -> str:
    """Case-insensitive fold for matching a needle against free prose.

    This is the **prose-matching** counterpart to :func:`canonical` — the
    two families are not interchangeable.  ``canonical`` casefolds and
    diacritic-folds (and collapses blank runs), which breaks case- and
    accent-sensitive substring/token matching against free prose.
    ``prose_fold`` deliberately does neither: it does NOT go beyond
    ``str.lower()`` (no casefold, no diacritic-folding, no blank collapse).
    One consumer is the sanitizer's first-person PII
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
    The prefix test is case-insensitive so the ingest safety-net can also
    detect and coerce any residual cased form (``"Speaker0"``) that a model
    emits — the coercion output is ALWAYS lowercase.  The structural test is
    purely syntactic — it does NOT check whether the id corresponds to a
    registered speaker.

    The index test is ``str.isdecimal``, which accepts exactly Unicode
    category Nd — the same set the previous ``\\d`` pattern matched.
    ``str.isdigit`` would additionally accept superscripts and is NOT
    equivalent.

    Args:
        s: String to test.

    Returns:
        ``True`` iff *s* is :data:`SPEAKER_ID_PREFIX` (in any case) followed by
        one or more decimal digits and nothing else.  ``False`` for partial
        matches (``"Speaker"``, ``"speaker"``, ``"alex"``, ``"SpeakerX"``) and
        empty strings.

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
    # Only the leading character is case-tolerant, matching the previous
    # ``[Ss]peaker`` pattern exactly: "Speaker0" passes, "SPEAKER0" does not.
    if s[:1].lower() + s[1 : len(SPEAKER_ID_PREFIX)] != SPEAKER_ID_PREFIX:
        return False
    return s[len(SPEAKER_ID_PREFIX) :].isdecimal()
