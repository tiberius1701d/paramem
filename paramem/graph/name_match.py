"""Name normalization helper used by :class:`paramem.graph.merger.GraphMerger`.

Stateless, deterministic, no I/O.  Thread-safe by construction.
"""

import unicodedata


def _nfkd(name: str) -> str:
    """Apply NFKD Unicode normalization and strip combining characters."""
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", name) if not unicodedata.combining(ch)
    )


def normalize_name(name: str) -> str:
    """Normalize *name* for coreference matching.

    Steps:
    1. NFKD Unicode fold (strips diacritics).
    2. Lowercase, strip leading/trailing whitespace.
    3. Collapse internal whitespace runs to a single space.
    """
    if not name:
        return ""
    folded = _nfkd(name)
    return " ".join(folded.lower().split())
