"""Sentence-type classifier — encoder-based, multilingual.

Mirrors :mod:`paramem.server.intent` in shape: cosine vs an exemplar
bank, margin gate, fail-closed default.  Reuses the same multilingual
sentence encoder that the intent classifier loads
(``intfloat/multilingual-e5-small`` by default), so the runtime cost
of adding this axis on the routing path is one extra forward pass +
one matrix-vector dot product — no new model.

Used by :func:`paramem.server.router._is_interrogative` to decide
whether a query is asking a question (``INTERROGATIVE``) or making a
statement / giving a command (``NON_INTERROGATIVE``).  The decision
gates the abstention helper at the routing path: only interrogatives
are eligible for the canned response.

Design tier with the rest of the routing path:

* **Pattern axis 1 — intent** (``paramem.server.intent``):
  PERSONAL / COMMAND / GENERAL / UNKNOWN routing decision.
* **Pattern axis 2 — sentence type** (this module):
  INTERROGATIVE / NON_INTERROGATIVE shape detection.
* **Pattern axis 3 — personal referent**
  (``paramem.server.personal_referent``, separate commit):
  ABOUT_SPEAKER / NOT_ABOUT_SPEAKER, replaces the English regex in
  :func:`paramem.server.sanitizer._contains_first_person`.

All three axes share the same encoder + exemplar pattern; multilingual
coverage scales by adding exemplar files, not code.

The exemplar bank is loaded once at server lifespan startup
alongside the intent classifier.  Test environments that never call
:func:`classify_sentence_type` pay no model-load cost.

Fail-closed semantics: when the encoder isn't loaded, the exemplar
bank is missing, the config disables the classifier, or the margin
gate isn't met, the classifier returns ``None`` so the caller can
fall back to a deterministic heuristic (today: punctuation + English
first-word lexicon at :func:`paramem.server.router._is_interrogative`).
This is asymmetric with the intent classifier on purpose — the
caller has a meaningful fallback at hand and benefits from the
distinction "encoder couldn't decide" vs "encoder said NON".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from paramem.server.config import SentenceTypeConfig
from paramem.server.intent import _read_exemplar_file, get_encoder

logger = logging.getLogger(__name__)


class SentenceType(str, Enum):
    """Binary classification for the abstention gate.

    The boundary is "is the speaker asking a question?" — interrogatives
    in any natural form (wh-questions, yes/no questions, indirect
    questions like "Tell me where I live", declaratives ending in ``?``)
    classify as ``INTERROGATIVE``; imperatives (commands) and
    declaratives (statements / fact-sharing) classify as
    ``NON_INTERROGATIVE``.
    """

    INTERROGATIVE = "interrogative"
    NON_INTERROGATIVE = "non_interrogative"


@dataclass
class _ExemplarBank:
    """Loaded exemplars + their precomputed L2-normalised embeddings.

    Two parallel arrays: ``types[i]`` is the class label for
    ``embeddings[i, :]``.  Embeddings are computed once at server
    lifespan startup; cosine similarity at query time is a single
    matrix-vector dot product.
    """

    types: list[SentenceType]
    embeddings: object  # numpy.ndarray of shape (N, D); typed loosely
    source_files: tuple[str, ...]


# Module-level singleton populated by :func:`load_exemplars` at lifespan
# startup.  Tests reset it via ``sentence_type._exemplars_singleton = None``.
_exemplars_singleton: _ExemplarBank | None = None


def load_exemplars(config: SentenceTypeConfig) -> _ExemplarBank | None:
    """Load all exemplar files from ``config.exemplars_dir`` and embed them.

    File layout: ``<class>.<lang>.txt`` (e.g. ``interrogative.en.txt``,
    ``non_interrogative.de.txt``).  The class portion of the filename
    must be a valid :class:`SentenceType` value; files with unrecognised
    class names are skipped with a warning so typos surface in the log.

    Returns ``None`` if the encoder isn't loaded, the exemplars dir is
    missing, the config disables the classifier, or no exemplars were
    collected.  Callers fall back to the deterministic heuristic in that
    case.

    The encoder applies ``encoder.query_prefix`` (e.g. the literal
    ``"query: "`` for the E5 family) to every exemplar before embedding,
    matching the prefix that :func:`classify_sentence_type` uses at
    query time.  No prefix-mismatch class is possible because the
    encoder handle is shared with the intent classifier.
    """
    global _exemplars_singleton

    if not config.enabled:
        logger.info("Sentence-type classifier disabled; skipping exemplar load")
        _exemplars_singleton = None
        return None

    encoder = get_encoder()
    if encoder is None:
        logger.info("Intent encoder not loaded; sentence-type classifier residual disabled")
        _exemplars_singleton = None
        return None

    dir_path = Path(config.exemplars_dir)
    if not dir_path.exists() or not dir_path.is_dir():
        logger.warning("Sentence-type exemplars directory not found: %s", dir_path)
        _exemplars_singleton = None
        return None

    rows: list[tuple[SentenceType, str]] = []
    sources: list[str] = []
    for txt_path in sorted(dir_path.glob("*.txt")):
        class_label = txt_path.stem.split(".", 1)[0]
        try:
            stype = SentenceType(class_label)
        except ValueError:
            logger.warning(
                "Skipping exemplar file with unrecognised class %r: %s",
                class_label,
                txt_path.name,
            )
            continue
        try:
            lines = _read_exemplar_file(txt_path)
        except OSError:
            logger.exception("Failed to read exemplar file %s", txt_path)
            continue
        for line in lines:
            rows.append((stype, line))
        sources.append(txt_path.name)

    if not rows:
        logger.warning("No sentence-type exemplars found under %s", dir_path)
        _exemplars_singleton = None
        return None

    types_list = [r[0] for r in rows]
    texts = [encoder.query_prefix + r[1] for r in rows]
    try:
        embeddings = encoder.model.encode(texts, normalize_embeddings=True)
    except Exception:
        logger.exception("Exemplar embedding failed; sentence-type residual disabled")
        _exemplars_singleton = None
        return None

    bank = _ExemplarBank(
        types=types_list,
        embeddings=embeddings,
        source_files=tuple(sources),
    )
    _exemplars_singleton = bank
    counts: dict[SentenceType, int] = {}
    for stype in types_list:
        counts[stype] = counts.get(stype, 0) + 1
    logger.info(
        "Loaded %d sentence-type exemplars across %d classes from %s (%s)",
        len(rows),
        len(counts),
        dir_path,
        ", ".join(f"{s.value}={n}" for s, n in counts.items()),
    )
    return bank


def get_exemplars() -> _ExemplarBank | None:
    """Return the cached exemplar bank, or ``None`` if not loaded."""
    return _exemplars_singleton


def _classify_via_encoder(
    text: str,
    encoder,
    bank: _ExemplarBank,
    config: SentenceTypeConfig,
) -> SentenceType | None:
    """Cosine-similarity vs per-class exemplar bank, with margin gate.

    Encodes the query (with the configured prefix), computes cosine
    against every exemplar embedding, takes the per-class top score,
    and returns the top class only when the margin between top-1 and
    top-2 classes meets ``config.confidence_margin``.  Below the
    margin returns ``None`` so the caller falls back to its
    deterministic heuristic.

    Embeddings are L2-normalised at load time, so cosine reduces to a
    plain dot product.  Per-class top score is the max similarity
    among that class's exemplars (one strong match within a class
    beats many weak ones).
    """
    try:
        query_emb = encoder.model.encode([encoder.query_prefix + text], normalize_embeddings=True)[
            0
        ]
    except Exception:
        logger.exception("Sentence-type query embedding failed; falling back")
        return None

    sims = bank.embeddings @ query_emb

    by_class: dict[SentenceType, float] = {}
    for i, stype in enumerate(bank.types):
        score = float(sims[i])
        if stype not in by_class or score > by_class[stype]:
            by_class[stype] = score

    if not by_class:
        return None

    sorted_classes = sorted(by_class.items(), key=lambda kv: -kv[1])
    top_type, top_score = sorted_classes[0]
    margin = (top_score - sorted_classes[1][1]) if len(sorted_classes) >= 2 else top_score

    if margin < config.confidence_margin:
        logger.debug(
            "Sentence-type margin below threshold (top=%s score=%.3f margin=%.3f); fall back",
            top_type.value,
            top_score,
            margin,
        )
        return None

    return top_type


def classify_sentence_type(
    text: str,
    *,
    config: SentenceTypeConfig | None = None,
) -> SentenceType | None:
    """Classify a query into ``INTERROGATIVE`` or ``NON_INTERROGATIVE``.

    Returns ``None`` when the encoder isn't loaded, exemplars aren't
    loaded, the config is missing or disables the classifier, or the
    margin gate isn't met.  Callers should treat ``None`` as "I don't
    know" and apply their own deterministic fallback (the abstention
    path falls back to the punctuation + lexicon heuristic in
    :func:`paramem.server.router._is_interrogative`).

    The function never raises — encoder/embedding errors or
    misconfiguration produce ``None`` rather than blocking the chat
    handler.
    """
    if config is None or not config.enabled:
        return None
    encoder = get_encoder()
    bank = get_exemplars()
    if encoder is None or bank is None:
        return None
    return _classify_via_encoder(text, encoder, bank, config)
