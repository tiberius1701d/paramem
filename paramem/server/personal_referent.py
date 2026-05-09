"""Personal-referent classifier — encoder-based, multilingual.

Mirrors :mod:`paramem.server.sentence_type` in shape: cosine vs an
exemplar bank, margin gate, returns ``None`` on uncertainty so the
caller falls back to a deterministic heuristic.  Reuses the same
multilingual sentence encoder loaded by the intent classifier — no
new model.

Used by :func:`paramem.server.sanitizer.check_personal_content` to
decide whether a query is asking about / referring to the speaker
themselves (``ABOUT_SPEAKER``) or about the world, third parties, or
device state (``NOT_ABOUT_SPEAKER``).  Replaces the
English-only token-set lookup in
:func:`paramem.server.sanitizer._contains_first_person`, which
silently let German / Mandarin / etc. self-referential queries past
the cloud-egress gate.

Design tier on the routing path:

* Pattern axis 1 — intent (``paramem.server.intent``).
* Pattern axis 2 — sentence type (``paramem.server.sentence_type``).
* Pattern axis 3 — personal referent (this module).

All three share the same encoder + exemplar pattern; multilingual
coverage scales by adding exemplar files, not code.

The exemplar bank is loaded once at server lifespan startup
alongside the intent and sentence-type classifiers.  Test
environments that never call :func:`classify_personal_referent` pay
no model-load cost.

Fail-soft semantics: when the encoder isn't loaded, the exemplar
bank is missing, the config disables the classifier, or the margin
gate isn't met, the classifier returns ``None`` so the sanitizer
can fall back to its existing
:func:`paramem.server.sanitizer._contains_first_person` token check.
This keeps the encoderless boot path (tests, degraded mode) working
on the same code path the production pre-encoder build used.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from paramem.server.config import PersonalReferentConfig
from paramem.server.intent import _read_exemplar_file, get_encoder

logger = logging.getLogger(__name__)


class PersonalReferent(str, Enum):
    """Binary classification for the cloud-egress sanitizer's personal
    check.

    The boundary is "is the speaker asking about themselves?" —
    self-referential queries (first-person interrogatives,
    self-introductions, possessive references like ``"my address"``,
    indirect first-person like ``"tell me where I live"``) classify
    as ``ABOUT_SPEAKER``; questions about the world / devices /
    third parties classify as ``NOT_ABOUT_SPEAKER``.
    """

    ABOUT_SPEAKER = "about_speaker"
    NOT_ABOUT_SPEAKER = "not_about_speaker"


@dataclass
class _ExemplarBank:
    """Loaded exemplars + their precomputed L2-normalised embeddings.

    Two parallel arrays: ``referents[i]`` is the class label for
    ``embeddings[i, :]``.  Embeddings are computed once at server
    lifespan startup; cosine similarity at query time is a single
    matrix-vector dot product.
    """

    referents: list[PersonalReferent]
    embeddings: object  # numpy.ndarray of shape (N, D); typed loosely
    source_files: tuple[str, ...]


# Module-level singleton populated by :func:`load_exemplars` at lifespan
# startup.  Tests reset it via ``personal_referent._exemplars_singleton = None``.
_exemplars_singleton: _ExemplarBank | None = None


def load_exemplars(config: PersonalReferentConfig) -> _ExemplarBank | None:
    """Load all exemplar files from ``config.exemplars_dir`` and embed them.

    File layout: ``<class>.<lang>.txt`` (e.g. ``about_speaker.en.txt``,
    ``not_about_speaker.de.txt``).  The class portion of the filename
    must be a valid :class:`PersonalReferent` value; files with
    unrecognised class names are skipped with a warning so typos
    surface in the log.

    Returns ``None`` if the encoder isn't loaded, the exemplars dir is
    missing, the config disables the classifier, or no exemplars were
    collected.  Callers fall back to the deterministic heuristic in
    that case.
    """
    global _exemplars_singleton

    if not config.enabled:
        logger.info("Personal-referent classifier disabled; skipping exemplar load")
        _exemplars_singleton = None
        return None

    encoder = get_encoder()
    if encoder is None:
        logger.info("Intent encoder not loaded; personal-referent classifier residual disabled")
        _exemplars_singleton = None
        return None

    dir_path = Path(config.exemplars_dir)
    if not dir_path.exists() or not dir_path.is_dir():
        logger.warning("Personal-referent exemplars directory not found: %s", dir_path)
        _exemplars_singleton = None
        return None

    rows: list[tuple[PersonalReferent, str]] = []
    sources: list[str] = []
    for txt_path in sorted(dir_path.glob("*.txt")):
        class_label = txt_path.stem.split(".", 1)[0]
        try:
            referent = PersonalReferent(class_label)
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
            rows.append((referent, line))
        sources.append(txt_path.name)

    if not rows:
        logger.warning("No personal-referent exemplars found under %s", dir_path)
        _exemplars_singleton = None
        return None

    referents_list = [r[0] for r in rows]
    texts = [encoder.query_prefix + r[1] for r in rows]
    try:
        embeddings = encoder.model.encode(texts, normalize_embeddings=True)
    except Exception:
        logger.exception("Exemplar embedding failed; personal-referent residual disabled")
        _exemplars_singleton = None
        return None

    bank = _ExemplarBank(
        referents=referents_list,
        embeddings=embeddings,
        source_files=tuple(sources),
    )
    _exemplars_singleton = bank
    counts: dict[PersonalReferent, int] = {}
    for referent in referents_list:
        counts[referent] = counts.get(referent, 0) + 1
    logger.info(
        "Loaded %d personal-referent exemplars across %d classes from %s (%s)",
        len(rows),
        len(counts),
        dir_path,
        ", ".join(f"{r.value}={n}" for r, n in counts.items()),
    )
    return bank


def get_exemplars() -> _ExemplarBank | None:
    """Return the cached exemplar bank, or ``None`` if not loaded."""
    return _exemplars_singleton


def _classify_via_encoder(
    text: str,
    encoder,
    bank: _ExemplarBank,
    config: PersonalReferentConfig,
) -> PersonalReferent | None:
    """Cosine-similarity vs per-class exemplar bank, with margin gate.

    Returns ``None`` when the margin between top-1 and top-2 classes
    is below ``config.confidence_margin``, so the caller can fall back
    to its deterministic heuristic.
    """
    try:
        query_emb = encoder.model.encode([encoder.query_prefix + text], normalize_embeddings=True)[
            0
        ]
    except Exception:
        logger.exception("Personal-referent query embedding failed; falling back")
        return None

    sims = bank.embeddings @ query_emb

    by_class: dict[PersonalReferent, float] = {}
    for i, referent in enumerate(bank.referents):
        score = float(sims[i])
        if referent not in by_class or score > by_class[referent]:
            by_class[referent] = score

    if not by_class:
        return None

    sorted_classes = sorted(by_class.items(), key=lambda kv: -kv[1])
    top_referent, top_score = sorted_classes[0]
    margin = (top_score - sorted_classes[1][1]) if len(sorted_classes) >= 2 else top_score

    if margin < config.confidence_margin:
        logger.debug(
            "Personal-referent margin below threshold (top=%s score=%.3f margin=%.3f); fall back",
            top_referent.value,
            top_score,
            margin,
        )
        return None

    return top_referent


def classify_personal_referent(
    text: str,
    *,
    config: PersonalReferentConfig | None = None,
) -> PersonalReferent | None:
    """Classify a query into ``ABOUT_SPEAKER`` or ``NOT_ABOUT_SPEAKER``.

    Returns ``None`` when the encoder isn't loaded, exemplars aren't
    loaded, the config is missing or disables the classifier, or the
    margin gate isn't met.  The sanitizer treats ``None`` as
    "I don't know" and falls back to the English token-set
    :func:`paramem.server.sanitizer._contains_first_person` check.

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
