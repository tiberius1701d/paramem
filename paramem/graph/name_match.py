"""String-identity canonicalization for entity names, objects, and predicates.

Used by :class:`paramem.graph.merger.GraphMerger` to produce deterministic node
keys.  Stateless, deterministic, no I/O.  Thread-safe by construction.

Over-collapse boundary (GUARANTEED-identical-only):
  Folded: Unicode canonical form, case (incl. ligatures `str.casefold` covers,
  e.g. ``ﬁ``→``fi``), diacritics, ``_``/``-``/whitespace runs.
  NOT folded: typos (HR3 ≠ HS3), substrings, NFKC compatibility forms
  (superscript ``²``, full-width), honorifics, token subsets.  Those are
  layer-2 LLM SAME_AS coreference, out of scope here.
"""

import unicodedata

# Built once at module load.  ``_``/``-`` → space; space-run collapse happens
# in the ``str.split()`` call so no special entry for multiple spaces is needed.
_SEP = str.maketrans({"_": " ", "-": " "})


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
