"""Intent classifier — HA fast-path + content-driven residual + fail-closed.

Routing decisions follow a tiered model:

1. **HA fast path (deterministic, cheap).**  HA entity-graph hits resolve
   immediately to :attr:`Intent.COMMAND`.  HA's namespace is closed
   (operator's installation) so the lexical match is reliable when it
   fires.

2. **Content-driven residual.**  When the HA fast path does not fire,
   ``IntentConfig.mode`` selects the residual classifier:

   * ``"embeddings"`` (default) — sentence-encoder cosine vs per-class
     exemplar bank, gated by top-1/top-2 margin.  ~1 ms.  Brittle on
     paraphrase / named entities outside the bank.
   * ``"llm"`` — single-token generation from the loaded local model
     using the intent-classifier section of ``voice.prompt_file``.
     ~50–200 ms.  No exemplar curation needed; handles paraphrase and
     named entities directly.  Automatically falls back to the encoder
     path when no classifier model is registered (cloud-only mode,
     model load failed).

3. **Fail-closed.**  When the residual classifier is unavailable or
   below the confidence margin, the classifier returns the configured
   ``fail_closed_intent`` — defaulting to ``personal`` so the
   cloud-escalation gate stays privacy-preserving under uncertainty.

PA graph match is **not** a state signal here.  Speaker enrollment must
not classify the speaker's queries as PERSONAL — that signal is the
residual classifier's job, derived from query content.  The router
scopes keys by speaker but lets the classifier decide intent.

The encoder, exemplar bank, and (when ``mode=llm``) the local model
handle are loaded once at server lifespan startup and live in
:data:`_encoder_singleton`, :data:`_exemplars_singleton`, and
:data:`_classifier_model_singleton` respectively.  Test environments
that never call :func:`classify_intent` pay no model-load cost.
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


@dataclass
class _ClassifierModelHandle:
    """Loose wrapper around the local LLM used by ``mode=llm`` classification.

    Populated by :func:`set_classifier_model` from the lifespan after the
    main model has loaded.  Cloud-only mode leaves it as ``None`` so the
    LLM-classify path automatically falls back to the encoder residual.
    """

    model: object  # transformers PreTrainedModel; typed loosely to avoid the heavy import
    tokenizer: object  # transformers PreTrainedTokenizer


# Module-level singleton populated by :func:`set_classifier_model` from
# the lifespan when the local model is loaded.  Stays ``None`` in cloud-
# only mode and when LLM classification is disabled.
_classifier_model_singleton: _ClassifierModelHandle | None = None


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


def set_classifier_model(model, tokenizer) -> None:
    """Register the local LLM used by ``mode=llm`` classification.

    Called from the lifespan after the main model has loaded.  Pass
    ``None`` for both arguments to clear the registration (e.g. before
    a model unload / cloud-only switch).  Idempotent.
    """
    global _classifier_model_singleton
    if model is None or tokenizer is None:
        _classifier_model_singleton = None
        return
    # BASE-MODEL HOLDER (intent classifier, mode=llm): module-global handle —
    # _release_base_model_in_process clears it via set_classifier_model(None, None).
    _classifier_model_singleton = _ClassifierModelHandle(model=model, tokenizer=tokenizer)


def get_classifier_model() -> _ClassifierModelHandle | None:
    """Return the cached classifier handle, or ``None`` if not registered."""
    return _classifier_model_singleton


_VALID_LABEL_TOKENS = {"PERSONAL", "COMMAND", "GENERAL", "UNKNOWN"}


def _parse_intent_label(text: str) -> Intent | None:
    """Extract the first valid intent label from a model response.

    The model is instructed to output a bare label, but in practice it
    occasionally prepends quotes, brackets, or a leading sentence
    fragment.  We accept the first occurrence of any valid label token,
    case-insensitive, as a substring of the response.  Returns ``None``
    when no recognised label is present so callers can fail-close.
    """
    upper = text.upper()
    best_pos: int | None = None
    best_label: Intent | None = None
    for token in _VALID_LABEL_TOKENS:
        pos = upper.find(token)
        if pos < 0:
            continue
        if best_pos is None or pos < best_pos:
            best_pos = pos
            best_label = Intent(token.lower())
    return best_label


def _build_voice_config(config: IntentConfig):
    """Lazy import of VoiceConfig to read the classifier-prompt section.

    The dependency on ``server.config`` is module-level circular; lazy
    import keeps the load order clean.
    """
    from paramem.server.config import VoiceConfig

    return VoiceConfig()


def _classify_via_llm(
    text: str,
    handle: _ClassifierModelHandle,
    config: IntentConfig,
) -> Intent:
    """Classify *text* by generating one short label from the local LLM.

    Uses the intent-classifier section of ``configs/prompts/pa_voice.txt``
    (loaded via :meth:`VoiceConfig.load_intent_classifier_prompt`) as the
    system prompt.  Generation is deterministic (``temperature=0``,
    ``do_sample=False``) and bounded to
    :attr:`IntentConfig.llm_max_new_tokens` tokens — enough for one
    label plus possible whitespace.

    Failure modes (prompt missing, generation error, unparseable
    output) all return the configured fail-closed intent; the function
    never raises.
    """
    voice = _build_voice_config(config)
    classifier_prompt = voice.load_intent_classifier_prompt()
    if classifier_prompt is None:
        logger.warning(
            "LLM intent classification requested but classifier prompt section "
            "is missing from voice.prompt_file; falling back"
        )
        return _fail_closed_intent(config)

    try:
        tokenizer = handle.tokenizer
        model = handle.model

        messages = [
            {"role": "system", "content": classifier_prompt},
            {"role": "user", "content": text},
        ]
        # Two-step tokenization mirrors the production inference path
        # (paramem/server/inference.py:900-908):
        #   1. apply_chat_template with tokenize=False → string prompt;
        #   2. tokenizer(prompt, return_tensors="pt") → BatchEncoding;
        #   3. model.generate(**inputs).
        # The earlier shortcut of passing apply_chat_template's tensor result
        # directly to generate() crashes on transformers >= 5 because the
        # call returns a BatchEncoding (no .shape attribute).
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Gradient checkpointing must be off during generate() (HF silently
        # disables KV cache when checkpointing is active).  Pair every
        # disable with a matching restore in `finally`.
        was_checkpointing = bool(getattr(model, "is_gradient_checkpointing", False))
        if was_checkpointing:
            model.gradient_checkpointing_disable()

        # Intent classification runs on the bare base model.  The PA
        # adapter is trained on personal-key recall and would bias the
        # classifier toward PERSONAL on content-free imperatives.
        # ``disable_adapter`` is a no-op for unwrapped base models.
        from peft import PeftModel as _PeftModel

        try:
            if isinstance(model, _PeftModel):
                with model.disable_adapter():
                    import torch as _torch

                    with _torch.no_grad():
                        output_ids = model.generate(
                            **inputs,
                            max_new_tokens=config.llm_max_new_tokens,
                            do_sample=False,
                            temperature=config.llm_temperature,
                            pad_token_id=getattr(tokenizer, "pad_token_id", None)
                            or getattr(tokenizer, "eos_token_id", None),
                        )
            else:
                import torch as _torch

                with _torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=config.llm_max_new_tokens,
                        do_sample=False,
                        temperature=config.llm_temperature,
                        pad_token_id=getattr(tokenizer, "pad_token_id", None)
                        or getattr(tokenizer, "eos_token_id", None),
                    )
        finally:
            if was_checkpointing:
                model.gradient_checkpointing_enable()

        generated = output_ids[0][inputs["input_ids"].shape[-1] :]
        response = tokenizer.decode(generated, skip_special_tokens=True)
    except Exception:
        logger.exception("LLM intent classification failed; falling back")
        return _fail_closed_intent(config)

    label = _parse_intent_label(response)
    if label is None:
        logger.warning(
            "LLM intent classifier produced no recognised label (response=%r); fail-closed",
            response[:200],
        )
        return _fail_closed_intent(config)
    logger.debug("LLM intent classifier verdict=%s (raw=%r)", label.value, response[:80])
    return label


def classify_intent(
    text: str,
    *,
    has_ha_match: bool,
    config: IntentConfig | None = None,
) -> Intent:
    """Classify a query into PERSONAL / COMMAND / GENERAL / UNKNOWN.

    Tier 1 — HA fast path:

    * ``has_ha_match`` → :attr:`Intent.COMMAND`

    Tier 2 — encoder residual (when encoder + exemplars are loaded):

    Cosine similarity against per-class exemplar embeddings, with a
    confidence-margin gate (top-1 vs top-2 class scores).  Below the
    margin the classifier returns the fail-closed default.

    Tier 3 — degraded mode:

    Encoder or exemplars unavailable → fall back to the fail-closed
    default when a ``config`` is given; ``Intent.UNKNOWN`` without one.

    PA graph match is intentionally **not** a state signal.  Speaker
    enrollment scopes keys at the router layer; intent comes from query
    content via the encoder.  This removes the previous "speaker-in-graph
    → PERSONAL" short-circuit that misrouted imperatives from enrolled
    speakers.

    The function never raises — encoder/embedding errors or
    misconfiguration produce a fail-safe result rather than blocking
    the chat handler.
    """
    if has_ha_match:
        return Intent.COMMAND

    # ``mode=llm`` dispatch (cheap fallback when the local model is not
    # registered — cloud-only mode, model load failed).  When the LLM
    # backend is unavailable we slide to the encoder path automatically
    # so a misconfigured ``mode=llm`` operator can't disable routing.
    if config is not None and config.mode == "llm":
        handle = get_classifier_model()
        if handle is not None:
            return _classify_via_llm(text, handle, config)
        logger.info(
            "intent.mode=llm but no classifier model registered; falling back to encoder residual"
        )

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
