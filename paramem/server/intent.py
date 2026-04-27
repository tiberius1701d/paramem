"""Residual intent classifier — state-first with encoder fallback.

Routing decisions follow a tiered model:

1. **State (deterministic, cheap).**  Speaker's knowledge graph hits or
   HA entity-graph hits resolve cleanly:

   * graph match → :attr:`Intent.PERSONAL`
   * HA match    → :attr:`Intent.COMMAND`

2. **Encoder fallback (residual).**  When neither state signal fires,
   a sentence-encoder + cosine-match against per-class exemplars gives
   the call.  This commit ships the loader scaffolding only; the
   exemplar-loading and actual cosine-routing logic arrive in a
   follow-up commit, at which point :func:`classify_intent` will
   return PERSONAL / COMMAND / GENERAL from the encoder.  Until then
   the residual returns :attr:`Intent.UNKNOWN`.

3. **Fail-closed.**  When the residual classifier is unavailable
   (encoder didn't load, exemplars missing, confidence below margin),
   the classifier returns the configured ``fail_closed_intent`` —
   defaulting to ``personal`` so the cloud-escalation gate stays
   privacy-preserving under uncertainty.

The encoder is loaded once at server lifespan startup and lives in
the module-level :data:`_encoder_singleton`.  Loading is lazy
(:func:`get_encoder` triggers it on first call) so test environments
that never call :func:`classify_intent` pay no model-load cost.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

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

    Tier 2 — encoder residual:

    The actual cosine-vs-exemplars logic ships in a subsequent commit.
    Until then the residual returns the fail-closed default
    (``Intent.PERSONAL`` by default) so the routing decision stays
    privacy-preserving while the classifier is being wired up.

    The function never raises — encoder unavailability or
    misconfiguration produces ``Intent.UNKNOWN`` or the fail-closed
    default rather than blocking the chat handler.
    """
    if has_graph_match:
        return Intent.PERSONAL
    if has_ha_match:
        return Intent.COMMAND

    # Residual: encoder fallback not yet wired (no exemplars in this commit).
    # When the encoder + exemplars land, this branch becomes:
    #   scores = _cosine_against_exemplars(text, encoder, exemplars)
    #   if margin(scores) < config.confidence_margin: return _fail_closed_intent(config)
    #   return Intent(top_class(scores))
    if config is None:
        return Intent.UNKNOWN
    return _fail_closed_intent(config)
