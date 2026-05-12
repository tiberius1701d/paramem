"""Quadruple-encoded indexed-key memory — production primitives.

Parallel to :mod:`paramem.training.indexed_memory` but uses the
``(key, subject, predicate, object)`` quadruple encoding instead of the
``(key, question, answer)`` QA-pair encoding.

Key differences vs. the QA format:

* **One training example per fact, not two.** The QA helper emits both a
  keyed-recall example and a natural-language QA example. Quadruples have
  no natural question, so only the keyed-recall example is emitted. This
  halves per-cycle training time.
* **JSON envelope is** ``{"key", "subject", "predicate", "object"}`` —
  same proven keyed-retrieval shape, round-trip-clean reconstruction.
* **Recall template is** ``"Recall the fact stored under key '{key}'."``
  — same imperative shape as the QA template, "fact" instead of "QA pair".
* **No natural-language recall path** — by design. The keyed prompt is the
  only guaranteed interface; natural-language questions are not trained.

SimHash, registry management, and probe functions mirror the QA equivalents
in :mod:`paramem.training.indexed_memory` so the two formats are interchangeable
at the call-site level.
"""

import hashlib
import json
import logging

from paramem.training.dataset import SYSTEM_PROMPT
from paramem.training.indexed_memory import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    SIMHASH_BITS,
    _clean_generation_artifacts,
    _tokenize_features,
    _tokenize_with_prompt_masking,
    get_simhash,
    simhash_confidence,
)

logger = logging.getLogger(__name__)

QUAD_RECALL_TEMPLATE = "Recall the fact stored under key '{key}'."


# --- Key assignment ---


def assign_quad_keys(
    triples: list[tuple[str, str, str]],
    start_index: int = 1,
    prefix: str = "graph",
) -> list[dict]:
    """Assign sequential ``<prefix><N>`` keys to a list of (subject, predicate, object) triples.

    Returns dicts with the four canonical fields used everywhere downstream.
    The ``start_index`` parameter mirrors :func:`paramem.training.indexed_memory.assign_keys`
    so callers can concatenate key ranges (e.g. episodic keys 1–N, procedural
    keys N+1–M) without collisions.

    Args:
        triples: List of ``(subject, predicate, object)`` 3-tuples.
        start_index: First key index; the i-th triple gets key
            ``f"{prefix}{start_index + i}"``.
        prefix: Key prefix string.  Episodic/semantic keys use ``"graph"``
            (default); procedural keys use ``"proc"`` to match the QA-format
            convention.

    Returns:
        List of ``{"key", "subject", "predicate", "object"}`` dicts in the
        same order as the input.
    """
    return [
        {
            "key": f"{prefix}{start_index + i}",
            "subject": s,
            "predicate": p,
            "object": o,
        }
        for i, (s, p, o) in enumerate(triples)
    ]


# --- Training format ---


def _build_quad_response(quad: dict) -> str:
    """Build the JSON response string for a single quadruple.

    Args:
        quad: Dict containing ``key``, ``subject``, ``predicate``, and ``object``.

    Returns:
        JSON string with exactly those four fields.
    """
    return json.dumps(
        {
            "key": quad["key"],
            "subject": quad["subject"],
            "predicate": quad["predicate"],
            "object": quad["object"],
        }
    )


def format_quadruple_training(
    quads: list[dict],
    tokenizer,
    max_length: int = 1024,
) -> list[dict]:
    """Build training examples — one keyed-recall example per quadruple.

    No natural-language second example (that would require a question, and the
    quadruple form has none by design). The single-example-per-quad cost halves
    the per-cycle training time vs. the QA format.

    Args:
        quads: List of quad dicts (each with ``key``, ``subject``,
            ``predicate``, ``object``).
        tokenizer: HuggingFace tokenizer compatible with
            :func:`paramem.training.indexed_memory._tokenize_with_prompt_masking`.
        max_length: Maximum token length per example (passed to the tokenizer).

    Returns:
        List of pre-tokenized training example dicts with ``input_ids``,
        ``attention_mask``, and ``labels`` (prompt tokens masked to -100).
    """
    from paramem.models.loader import adapt_messages

    examples = []
    for quad in quads:
        recall_prompt = QUAD_RECALL_TEMPLATE.format(key=quad["key"])
        recall_response = _build_quad_response(quad)
        messages = adapt_messages(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": recall_prompt},
                {"role": "assistant", "content": recall_response},
            ],
            tokenizer,
        )
        examples.append(_tokenize_with_prompt_masking(messages, tokenizer, max_length))
    return examples


# --- Recall parsing ---


def parse_recalled_quad(text: str) -> dict | None:
    """Parse a single recalled quadruple JSON object from model output.

    Tries the raw text first, then — only if cleaning changed anything — retries
    on :func:`paramem.training.indexed_memory._clean_generation_artifacts`'d text
    (strips markdown bold markers and excess newlines that some instruct models
    inject). This mirrors :func:`paramem.training.indexed_memory.parse_recalled_pair`.

    Uses progressive first-object extraction: scans forward for ``{``, tries each
    ``}`` (via ``raw_decode``, never ``rfind``) to avoid swallowing chained objects.

    A list-valued ``object`` field is coerced to a comma-joined string so
    multi-value objects survive the round-trip.

    Args:
        text: Raw model output string.

    Returns:
        ``{"key", "subject", "predicate", "object"}`` dict, or ``None`` if the
        output is not parseable or does not contain the required fields.
    """
    text = text.strip()
    required = {"key", "subject", "predicate", "object"}
    decoder = json.JSONDecoder()

    def _coerce(v) -> str:
        if isinstance(v, str):
            return v
        if isinstance(v, list):
            return ", ".join(str(x) for x in v)
        return str(v)

    def _try_parse(t: str) -> dict | None:
        for i, ch in enumerate(t):
            if ch != "{":
                continue
            try:
                obj, _end = decoder.raw_decode(t[i:])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and required.issubset(obj.keys()):
                return {k: _coerce(obj[k]) for k in ("key", "subject", "predicate", "object")}
        return None

    # Try raw text first, then cleaned (only if cleaning changed anything).
    result = _try_parse(text)
    if result is not None:
        return result

    cleaned = _clean_generation_artifacts(text)
    if cleaned != text:
        return _try_parse(cleaned)

    return None


# --- SimHash fingerprinting ---


def compute_simhash(
    key: str,
    subject: str,
    predicate: str,
    obj: str,
    num_bits: int = SIMHASH_BITS,
) -> int:
    """Compute a SimHash fingerprint from key + subject + predicate + object.

    Mirrors :func:`paramem.training.indexed_memory.compute_simhash` using the
    same unigram+bigram feature tokenization and bit-vote algorithm. The key is
    included so that identical triple content under different keys produces
    different fingerprints — catches hallucinations where the model echoes the
    queried key but returns another key's content.

    Note: this fingerprint is NOT compatible with the QA-format SimHash — they
    hash different content strings. Existing ``simhash_registry_*.json`` files
    built with the QA format must be regenerated after switching to quads.

    Args:
        key: The ``graphN`` key string.
        subject: Triple subject.
        predicate: Triple predicate.
        obj: Triple object.
        num_bits: Number of bits in the fingerprint (default 64).

    Returns:
        A ``num_bits``-bit integer fingerprint.
    """
    text = f"{key} {subject} {predicate} {obj}"
    features = _tokenize_features(text)

    if not features:
        return 0

    sums = [0] * num_bits
    for feature in features:
        h = int(hashlib.md5(feature.encode()).hexdigest(), 16)
        for i in range(num_bits):
            if h & (1 << i):
                sums[i] += 1
            else:
                sums[i] -= 1

    fingerprint = 0
    for i in range(num_bits):
        if sums[i] > 0:
            fingerprint |= 1 << i

    return fingerprint


def verify_confidence(
    recalled: dict,
    registry: dict[str, int] | dict[str, dict] | None = None,
) -> float:
    """Verify a recalled quadruple against a SimHash registry.

    Mirrors the contract of :func:`paramem.training.indexed_memory.verify_confidence`
    exactly:

    - Returns ``1.0`` when no registry is provided (no verification).
    - Returns ``0.0`` when the key is absent from the registry (untrained key).
    - Returns a ``0.0–1.0`` SimHash similarity score otherwise (higher = more
      likely genuine recall).

    Supports both the simple ``{key: int}`` registry shape and the enriched
    ``{key: {"simhash": int, ...}}`` shape via
    :func:`paramem.training.indexed_memory.get_simhash`.

    Args:
        recalled: Dict containing at minimum ``key``, ``subject``,
            ``predicate``, and ``object``.
        registry: Optional SimHash registry mapping key → fingerprint.

    Returns:
        Confidence score in ``[0.0, 1.0]``.
    """
    if registry is None:
        return 1.0

    key = recalled.get("key", "")
    expected = get_simhash(registry, key)
    if expected is None:
        return 0.0

    actual = compute_simhash(
        key,
        recalled.get("subject", ""),
        recalled.get("predicate", ""),
        recalled.get("object", ""),
    )
    return simhash_confidence(actual, expected)


# --- Registry management ---


def build_registry(quad_pairs: list[dict]) -> dict[str, int]:
    """Build a SimHash registry from quad pairs.

    Returns a mapping of ``key → 64-bit SimHash fingerprint``.
    This is the simple format used by the training pipeline.
    For enriched metadata, see :func:`build_enriched_registry`.

    Args:
        quad_pairs: List of quad dicts, each containing ``key``,
            ``subject``, ``predicate``, and ``object``.

    Returns:
        Dict mapping each key to its SimHash fingerprint.
    """
    return {
        p["key"]: compute_simhash(p["key"], p["subject"], p["predicate"], p["object"])
        for p in quad_pairs
    }


def build_enriched_registry(
    quad_pairs: list[dict],
    session_id: str | None = None,
    existing: dict | None = None,
) -> dict:
    """Build an enriched registry with temporal metadata.

    Mirrors :func:`paramem.training.indexed_memory.build_enriched_registry`
    exactly — same entry shape, same merge/stale semantics — but uses the
    quad SimHash instead of the QA SimHash.

    Format per key::

        {
            "simhash": int,
            "created_at": ISO timestamp,
            "last_seen_at": ISO timestamp,
            "session_id": str,
            "status": "active" | "stale",
            "stale_since": ISO timestamp | null,
            "stale_cycles": 0
        }

    If an existing registry is provided:

    - Active keys present in ``quad_pairs`` get their ``last_seen_at`` updated.
    - Active keys NOT in ``quad_pairs`` are unchanged (not auto-staled).
    - Stale keys NOT in ``quad_pairs`` get ``stale_cycles`` incremented.
    - Stale keys present in ``quad_pairs`` are reactivated.

    Args:
        quad_pairs: List of quad dicts for the current training set.
        session_id: Optional session identifier to tag new entries.
        existing: Optional existing registry dict to merge into.

    Returns:
        Updated enriched registry dict.
    """
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()
    current_keys = {p["key"] for p in quad_pairs}

    if existing is None:
        existing = {}

    registry = dict(existing)

    for p in quad_pairs:
        key = p["key"]
        simhash = compute_simhash(key, p["subject"], p["predicate"], p["object"])

        if key in registry:
            registry[key]["simhash"] = simhash
            registry[key]["last_seen_at"] = now
            registry[key]["status"] = "active"
            registry[key]["stale_since"] = None
            registry[key]["stale_cycles"] = 0
            if session_id:
                registry[key]["session_id"] = session_id
        else:
            registry[key] = {
                "simhash": simhash,
                "created_at": now,
                "last_seen_at": now,
                "session_id": session_id or "",
                "status": "active",
                "stale_since": None,
                "stale_cycles": 0,
            }

    for key, entry in registry.items():
        if key not in current_keys and entry.get("status") == "stale":
            entry["stale_cycles"] = entry.get("stale_cycles", 0) + 1

    return registry


# --- Probe ---


def probe_quad(
    model,
    tokenizer,
    key: str,
    max_new_tokens: int = 128,
    registry: dict | None = None,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> dict | None:
    """Prompt the model to recall a single quadruple by key.

    Mirrors :func:`paramem.training.indexed_memory.probe_key`:

    - Returns a dict with ``key``, ``subject``, ``predicate``, ``object``,
      ``confidence``, ``raw_output``, ``format`` (``"quad"``), and
      ``fact_text`` (human-readable via :func:`quad_fact_text`) on success
      (``confidence`` is ``1.0`` when no registry is supplied — same as
      ``probe_key``).
    - Returns ``{"raw_output": ..., "failure_reason": "parse_failure"}``
      when the output cannot be parsed.
    - Returns ``{"raw_output": ..., "failure_reason": "key_mismatch:<got>"}``
      when the recalled key does not match the queried key.
    - Returns ``{"raw_output": ..., "failure_reason": "low_confidence:<val>"}``
      when the confidence falls below the threshold.

    The new trailing parameters (``registry``, ``confidence_threshold``) are
    optional and default-valued, so existing positional callers in
    ``experiments/`` keep working unchanged.

    Args:
        model: Loaded HuggingFace / PEFT model.
        tokenizer: Tokenizer matching the model.
        key: Key to recall (e.g. ``"graph3"``).
        max_new_tokens: Maximum tokens to generate.
        registry: Optional SimHash registry for confidence verification.
        confidence_threshold: Minimum confidence to accept a recalled quad.

    Returns:
        Result dict on success or failure — never ``None`` (shape contract
        matches :func:`paramem.training.indexed_memory.probe_key`).
    """
    from paramem.evaluation.recall import generate_answer
    from paramem.training.dataset import _format_inference_prompt

    prompt = QUAD_RECALL_TEMPLATE.format(key=key)
    formatted = _format_inference_prompt(prompt, tokenizer)
    raw = generate_answer(
        model,
        tokenizer,
        formatted,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
    )
    parsed = parse_recalled_quad(raw)
    if parsed is None:
        logger.debug("Parse failure for key '%s': %s", key, raw[:200])
        return {"raw_output": raw, "failure_reason": "parse_failure"}
    if parsed["key"] != key:
        logger.debug("Key mismatch: queried '%s', got '%s'", key, parsed["key"])
        return {
            "raw_output": raw,
            "failure_reason": f"key_mismatch:{parsed['key']}",
        }

    # Check confidence against registry (verify_confidence returns 1.0 when
    # registry is None, so confidence is always present on the returned dict —
    # same contract as paramem.training.indexed_memory.probe_key).
    confidence = verify_confidence(parsed, registry)
    if confidence < confidence_threshold:
        logger.debug(
            "Low confidence for key '%s': %.3f < %.3f threshold",
            key,
            confidence,
            confidence_threshold,
        )
        return {"raw_output": raw, "failure_reason": f"low_confidence:{confidence:.3f}"}

    parsed["confidence"] = confidence
    parsed["raw_output"] = raw
    # Symmetric with indexed_memory.probe_key which sets format="qa" and
    # fact_text=answer on its own success dict. Direct callers (gates, tests)
    # should not need to know the encoding to read fact_text.
    parsed["format"] = "quad"
    parsed["fact_text"] = quad_fact_text(parsed)
    return parsed


# --- Fact-text helper ---


def quad_fact_text(quad: dict) -> str:
    """Render a recalled quadruple as a human-readable fact string.

    Converts the predicate from snake_case to space-separated words (predicates
    are normalized to lowercase-underscore at extraction time per project
    convention) and joins the three triple components into a natural sentence.

    Used by inference consumers so format-specific string construction stays in
    the probe layer — callers read ``result["fact_text"]`` without needing to
    know the format.

    Args:
        quad: Dict containing at minimum ``subject``, ``predicate``, and
            ``object`` keys.

    Returns:
        Human-readable fact string, e.g. ``"Alex lives in Heilbronn"``.
    """
    predicate_spaced = " ".join(quad["predicate"].replace("_", " ").split())
    return f"{quad['subject']} {predicate_spaced} {quad['object']}"


# --- Registry persistence (re-exported for convenience) ---

# These are identical to the QA-format registry persistence functions since
# the file format is the same (JSON dict). Import and re-export so callers
# don't need to mix imports from both modules.

from paramem.training.indexed_memory import (  # noqa: E402, F401
    get_active_keys,
    get_reclaimable_keys,
    load_registry,
    mark_stale,
    save_registry,
)
