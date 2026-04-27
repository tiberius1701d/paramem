"""Residual intent classifier — state-first with encoder fallback.

Routing decisions follow a tiered model:

1. **State (deterministic, cheap).**  Speaker's knowledge graph hits or
   HA entity-graph hits resolve cleanly:

   * graph match → :attr:`Intent.PERSONAL`
   * HA match    → :attr:`Intent.COMMAND`

2. **Encoder fallback (residual).**  When neither state signal fires,
   a sentence-encoder + cosine-match against per-class exemplars gives
   the call.  Per-class top similarity (max over the class's
   exemplars) is gated by a margin between the top-1 and top-2
   classes; below the margin the classifier hands off to the
   fail-closed default rather than guessing.  Embeddings are
   L2-normalised at load time so cosine reduces to a plain dot
   product at query time.

3. **Fail-closed.**  When the residual classifier is unavailable
   (encoder didn't load, exemplars missing, confidence below margin),
   the classifier returns the configured ``fail_closed_intent`` —
   defaulting to ``personal`` so the cloud-escalation gate stays
   privacy-preserving under uncertainty.

The encoder and exemplar bank are loaded once at server lifespan
startup and live in :data:`_encoder_singleton` and
:data:`_exemplars_singleton`.  Test environments that never call
:func:`classify_intent` pay no model-load cost.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from paramem.server.config import IntentConfig
from paramem.server.router import Intent

logger = logging.getLogger(__name__)


@dataclass
class _EncoderHandle:
    """Light wrapper around the loaded sentence-encoder.

    The wrapper exists so call sites depend on ``_EncoderHandle`` rather
    than ``sentence_transformers.SentenceTransformer`` directly — keeps
    the classifier surface mockable in unit tests without importing the
    heavy library.
    """

    model: object  # SentenceTransformer instance; typed loosely to avoid forcing the import
    query_prefix: str
    device: str  # resolved device ("cuda" / "cpu")
    dtype: str  # resolved dtype ("float16" / "float32")


# Module-level singleton populated by :func:`load_encoder` at lifespan
# startup.  Tests reset it via ``intent._encoder_singleton = None``.
_encoder_singleton: _EncoderHandle | None = None


@dataclass
class _ExemplarBank:
    """Loaded exemplars + their precomputed L2-normalised embeddings.

    Two parallel arrays: ``intents[i]`` is the class label for
    ``embeddings[i, :]``.  Embeddings are computed once at server
    lifespan startup; cosine similarity at query time is a single
    matrix-vector dot product.
    """

    intents: list[Intent]
    embeddings: object  # numpy.ndarray of shape (N, D); typed loosely
    source_files: tuple[str, ...]


# Module-level singleton populated by :func:`load_exemplars` at lifespan
# startup.  Tests reset it via ``intent._exemplars_singleton = None``.
_exemplars_singleton: _ExemplarBank | None = None


def _resolve_device(requested: str) -> str:
    """``auto`` → cuda when available, else cpu.  Other values pass through."""
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_encoder(config: IntentConfig) -> _EncoderHandle | None:
    """Load the sentence-encoder per ``config`` and store the singleton.

    Returns the loaded handle on success, or ``None`` on any failure
    (missing dep, model download error, OOM, invalid config).  Failure
    is non-fatal: callers fall back to ``Intent.UNKNOWN`` from the
    residual path, and :func:`classify_intent` honours the
    fail-closed default.

    Idempotent — calling with the same config returns the cached
    handle without reloading.  A different ``encoder_model`` /
    ``encoder_device`` / ``encoder_dtype`` triggers a fresh load.
    """
    global _encoder_singleton

    if not config.enabled:
        logger.info("Intent classifier disabled; skipping encoder load")
        _encoder_singleton = None
        return None

    if _encoder_singleton is not None:
        prior = _encoder_singleton
        if (
            prior.model is not None
            and prior.device == _resolve_device(config.encoder_device)
            and prior.dtype == config.encoder_dtype
            # Model identity is not directly comparable (no name on
            # SentenceTransformer); we rely on the singleton + config-
            # driven invalidation flow at startup.  If the operator
            # changes encoder_model the server must restart anyway
            # (PIPELINE_ALTERING tier) so this branch is correct in
            # the only path that exercises it.
        ):
            return prior

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.warning("sentence-transformers not available; intent classifier residual disabled")
        _encoder_singleton = None
        return None

    device = _resolve_device(config.encoder_device)
    dtype = config.encoder_dtype

    try:
        model = SentenceTransformer(config.encoder_model, device=device)
        if device == "cuda" and dtype == "float16":
            model.half()
    except Exception:
        logger.exception(
            "Encoder load failed (model=%s device=%s); intent classifier residual disabled",
            config.encoder_model,
            device,
        )
        _encoder_singleton = None
        return None

    handle = _EncoderHandle(
        model=model,
        query_prefix=config.encoder_query_prefix,
        device=device,
        dtype=dtype,
    )
    _encoder_singleton = handle
    logger.info(
        "Intent encoder loaded: model=%s device=%s dtype=%s prefix=%r",
        config.encoder_model,
        device,
        dtype,
        config.encoder_query_prefix,
    )
    return handle


def get_encoder() -> _EncoderHandle | None:
    """Return the cached encoder handle, or ``None`` if not loaded."""
    return _encoder_singleton


def _read_exemplar_file(path: Path) -> list[str]:
    """One query per line; ``#``-prefix and blank lines ignored."""
    lines: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines


def load_exemplars(config: IntentConfig, encoder: _EncoderHandle | None) -> _ExemplarBank | None:
    """Load all exemplar files from ``config.exemplars_dir`` and embed them.

    File layout: ``<class>.<lang>.txt`` (e.g. ``personal.en.txt``,
    ``command.de.txt``).  The class portion of the filename must be a
    valid :class:`Intent` value (``personal`` / ``command`` / ``general``);
    files with unrecognised class names are skipped with a warning so
    typos surface in the log.

    Returns ``None`` if the encoder isn't loaded, the exemplars dir is
    missing, or no exemplars were collected.  Callers fall back to the
    fail-closed default in that case.

    The encoder applies ``config.encoder_query_prefix`` (e.g. the literal
    ``"query: "`` for the E5 family) to every exemplar before embedding,
    matching the prefix that ``classify_intent`` will use at query time.
    """
    global _exemplars_singleton

    if encoder is None:
        _exemplars_singleton = None
        return None

    dir_path = Path(config.exemplars_dir)
    if not dir_path.exists() or not dir_path.is_dir():
        logger.warning("Intent exemplars directory not found: %s", dir_path)
        _exemplars_singleton = None
        return None

    rows: list[tuple[Intent, str]] = []
    sources: list[str] = []
    for txt_path in sorted(dir_path.glob("*.txt")):
        # Filename: <class>.<lang>.txt — class is the first dotted segment.
        class_label = txt_path.stem.split(".", 1)[0]
        try:
            intent = Intent(class_label)
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
            rows.append((intent, line))
        sources.append(txt_path.name)

    if not rows:
        logger.warning("No intent exemplars found under %s", dir_path)
        _exemplars_singleton = None
        return None

    intents_list = [r[0] for r in rows]
    texts = [encoder.query_prefix + r[1] for r in rows]
    try:
        embeddings = encoder.model.encode(texts, normalize_embeddings=True)
    except Exception:
        logger.exception("Exemplar embedding failed; intent residual disabled")
        _exemplars_singleton = None
        return None

    bank = _ExemplarBank(
        intents=intents_list,
        embeddings=embeddings,
        source_files=tuple(sources),
    )
    _exemplars_singleton = bank
    counts: dict[Intent, int] = {}
    for intent in intents_list:
        counts[intent] = counts.get(intent, 0) + 1
    logger.info(
        "Loaded %d intent exemplars across %d classes from %s (%s)",
        len(rows),
        len(counts),
        dir_path,
        ", ".join(f"{i.value}={n}" for i, n in counts.items()),
    )
    return bank


def get_exemplars() -> _ExemplarBank | None:
    """Return the cached exemplar bank, or ``None`` if not loaded."""
    return _exemplars_singleton


def _fail_closed_intent(config: IntentConfig) -> Intent:
    """Resolve ``config.fail_closed_intent`` to an :class:`Intent` value.

    Honours arbitrary operator-supplied strings by falling back to
    ``Intent.PERSONAL`` (the privacy-preserving default) when the
    string is not a recognised intent value.
    """
    try:
        return Intent(config.fail_closed_intent)
    except ValueError:
        logger.warning(
            "Invalid fail_closed_intent=%r; defaulting to PERSONAL",
            config.fail_closed_intent,
        )
        return Intent.PERSONAL


def _classify_via_encoder(
    text: str,
    encoder: _EncoderHandle,
    bank: _ExemplarBank,
    config: IntentConfig,
) -> Intent:
    """Cosine-similarity vs per-class exemplar bank, with margin gate.

    Encodes the query (with the configured prefix), computes cosine
    against every exemplar embedding, takes the per-class top score,
    and returns the top class only when the margin between top-1 and
    top-2 classes meets ``config.confidence_margin``.  Below the
    margin the classifier returns the configured fail-closed intent.

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
        logger.exception("Query embedding failed; falling back")
        return _fail_closed_intent(config)

    # Cosine since both are L2-normalised: simple dot product.
    sims = bank.embeddings @ query_emb

    by_class: dict[Intent, float] = {}
    for i, intent in enumerate(bank.intents):
        score = float(sims[i])
        if intent not in by_class or score > by_class[intent]:
            by_class[intent] = score

    if not by_class:
        return _fail_closed_intent(config)

    sorted_classes = sorted(by_class.items(), key=lambda kv: -kv[1])
    top_intent, top_score = sorted_classes[0]
    margin = (top_score - sorted_classes[1][1]) if len(sorted_classes) >= 2 else top_score

    if margin < config.confidence_margin:
        logger.debug(
            "Intent classifier margin below threshold (top=%s score=%.3f margin=%.3f); fail-closed",
            top_intent.value,
            top_score,
            margin,
        )
        return _fail_closed_intent(config)

    return top_intent


def classify_intent(
    text: str,
    *,
    has_graph_match: bool,
    has_ha_match: bool,
    config: IntentConfig | None = None,
) -> Intent:
    """Classify a query into PERSONAL / COMMAND / GENERAL / UNKNOWN.

    Tier 1 — state signals (cheap, deterministic):

    * ``has_graph_match`` → :attr:`Intent.PERSONAL`
    * ``has_ha_match``    → :attr:`Intent.COMMAND`

    Tier 2 — encoder residual (when encoder + exemplars are loaded):

    Cosine similarity against per-class exemplar embeddings, with a
    confidence-margin gate (top-1 vs top-2 class scores).  Below the
    margin the classifier returns the fail-closed default.

    Tier 3 — degraded mode:

    Encoder or exemplars unavailable → fall back to the fail-closed
    default with a config; ``Intent.UNKNOWN`` without one.

    The function never raises — encoder/embedding errors or
    misconfiguration produce a fail-safe result rather than blocking
    the chat handler.
    """
    if has_graph_match:
        return Intent.PERSONAL
    if has_ha_match:
        return Intent.COMMAND

    encoder = get_encoder()
    bank = get_exemplars()
    if encoder is None or bank is None:
        if config is None:
            return Intent.UNKNOWN
        return _fail_closed_intent(config)

    if config is None:
        # Without config we have no margin threshold or fail-closed
        # default; degrade to UNKNOWN so callers treat conservatively.
        return Intent.UNKNOWN

    return _classify_via_encoder(text, encoder, bank, config)
