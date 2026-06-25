"""Indexed-key entry memory — production primitives.

Each entry has the canonical shape ``(key, subject, predicate, object)``.

Key properties:

* **One training example per fact.** Only the keyed-recall example is
  emitted — no natural-language second example, which halves per-cycle
  training time.
* **JSON envelope is** ``{"key", "subject", "predicate", "object"}`` —
  round-trip-clean, deterministic reconstruction.
* **Recall template is** ``"Recall the fact stored under key '{key}'."``
* **No natural-language recall path** — by design. The keyed prompt is the
  only guaranteed interface; natural-language questions are not trained.

SimHash fingerprinting constants and helpers are defined directly in this
module.  Registry-lifecycle helpers (``save_registry``, ``load_registry``,
``get_active_keys``) are defined in :mod:`paramem.memory.persistence` and
re-exported here so callers can import them from either module.
"""

import hashlib
import json
import logging
from collections.abc import Callable

from paramem.training.dataset import SYSTEM_PROMPT, _tokenize_with_prompt_masking

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shape-agnostic SimHash constants and helpers
# (relocated from paramem.training.indexed_memory on 2026-05-20)
# ---------------------------------------------------------------------------

SIMHASH_BITS = 64
DEFAULT_CONFIDENCE_THRESHOLD = 0.75


def _tokenize_features(text: str) -> list[str]:
    """Tokenize text into word unigrams and bigrams for SimHash."""
    words = text.lower().split()
    features = list(words)
    for i in range(len(words) - 1):
        features.append(f"{words[i]} {words[i + 1]}")
    return features


def simhash_confidence(hash_a: int, hash_b: int, num_bits: int = SIMHASH_BITS) -> float:
    """Compute similarity confidence from two SimHash fingerprints.

    Returns ``1.0`` for identical fingerprints, ~``0.5`` for unrelated content.
    Based on normalized Hamming distance.

    Args:
        hash_a: First SimHash fingerprint integer.
        hash_b: Second SimHash fingerprint integer.
        num_bits: Number of bits in each fingerprint.

    Returns:
        Similarity score in ``[0.0, 1.0]``.
    """
    distance = bin(hash_a ^ hash_b).count("1")
    return 1.0 - (distance / num_bits)


def get_simhash(registry: dict, key: str) -> int | None:
    """Extract simhash from either simple or enriched registry format.

    Args:
        registry: Either ``{key: int}`` (simple) or ``{key: {"simhash": int, ...}}``
            (enriched).
        key: Registry key to look up.

    Returns:
        Integer SimHash fingerprint, or ``None`` when the key is absent.
    """
    entry = registry.get(key)
    if entry is None:
        return None
    if isinstance(entry, int):
        return entry
    if isinstance(entry, dict):
        return entry.get("simhash")
    return None


def _clean_generation_artifacts(text: str) -> str:
    """Strip markdown formatting artifacts from model output.

    Some instruct models (notably Gemma 2) inject ``**`` bold markers and
    excess newlines into generated JSON, breaking structure.  This cleans
    the text before JSON parsing.

    NOTE: Artifact patterns may evolve across reinforcement/consolidation
    cycles as the model retrains on its own output.  Monitor parse failure
    rates across cycles and extend patterns here if new artifacts emerge.

    Args:
        text: Raw model output string.

    Returns:
        Cleaned text with bold markers removed and excess newlines collapsed.
    """
    import re

    text = text.replace("**", "")
    text = re.sub(r"\n{3,}", "\n", text)
    return text


RECALL_TEMPLATE = "Recall the fact stored under key '{key}'."


# --- Key assignment ---


def assign_keys(
    triples: list[tuple[str, str, str]],
    start_index: int = 1,
    prefix: str = "graph",
) -> list[dict]:
    """Assign sequential ``<prefix><N>`` keys to a list of (subject, predicate, object) triples.

    Returns dicts with the four canonical fields used everywhere downstream.
    The ``start_index`` parameter lets callers concatenate key ranges
    (e.g. episodic keys 1–N, procedural keys N+1–M) without collisions.

    Args:
        triples: List of ``(subject, predicate, object)`` 3-tuples.
        start_index: First key index; the i-th triple gets key
            ``f"{prefix}{start_index + i}"``.
        prefix: Key prefix string.  Episodic/semantic keys use ``"graph"``
            (default); procedural keys use ``"proc"`` by convention.

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


def _build_response(entry: dict) -> str:
    """Build the JSON response string for a single entry.

    Args:
        entry: Dict containing ``key``, ``subject``, ``predicate``, and ``object``.

    Returns:
        JSON string with exactly those four fields.
    """
    return json.dumps(
        {
            "key": entry["key"],
            "subject": entry["subject"],
            "predicate": entry["predicate"],
            "object": entry["object"],
        }
    )


def format_entry_training(
    entries: list[dict],
    tokenizer,
    max_length: int = 1024,
) -> list[dict]:
    """Build training examples — one keyed-recall example per entry.

    No natural-language second example. One example per entry halves
    the per-cycle training time.

    Args:
        entries: List of entry dicts (each with ``key``, ``subject``,
            ``predicate``, ``object``).
        tokenizer: HuggingFace tokenizer compatible with
            :func:`paramem.training.dataset._tokenize_with_prompt_masking`.
        max_length: Maximum token length per example (passed to the tokenizer).

    Returns:
        List of pre-tokenized training example dicts with ``input_ids``,
        ``attention_mask``, and ``labels`` (prompt tokens masked to -100).
    """
    from paramem.models.loader import adapt_messages

    examples = []
    for entry in entries:
        recall_prompt = RECALL_TEMPLATE.format(key=entry["key"])
        recall_response = _build_response(entry)
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


def parse_recalled_entry(text: str) -> dict | None:
    """Parse a single recalled entry JSON object from model output.

    Tries the raw text first, then — only if cleaning changed anything — retries
    on :func:`paramem.memory.entry._clean_generation_artifacts`'d text
    (strips markdown bold markers and excess newlines that some instruct models
    inject).

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

    Uses unigram+bigram feature tokenization and a bit-vote algorithm. The key is
    included so that identical triple content under different keys produces
    different fingerprints — catches hallucinations where the model echoes the
    queried key but returns another key's content.

    Note: this fingerprint is NOT compatible with legacy SimHash registry files
    built with an earlier content string format.
    Existing ``simhash_registry_*.json`` files built with the legacy format must
    be regenerated.

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
    """Verify a recalled entry against a SimHash registry.

    Contract:

    - Returns ``1.0`` when no registry is provided (no verification).
    - Returns ``0.0`` when the key is absent from the registry (untrained key).
    - Returns a ``0.0–1.0`` SimHash similarity score otherwise (higher = more
      likely genuine recall).

    Supports both the simple ``{key: int}`` registry shape and the enriched
    ``{key: {"simhash": int, ...}}`` shape via
    :func:`paramem.memory.entry.get_simhash`.

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


def build_registry(entries: list[dict]) -> dict[str, int]:
    """Build a SimHash registry from entries.

    Returns a mapping of ``key → 64-bit SimHash fingerprint``.
    This is the simple format used by the training pipeline.

    Args:
        entries: List of entry dicts, each containing ``key``,
            ``subject``, ``predicate``, and ``object``.

    Returns:
        Dict mapping each key to its SimHash fingerprint.
    """
    return {
        p["key"]: compute_simhash(p["key"], p["subject"], p["predicate"], p["object"])
        for p in entries
    }


# --- Probe ---


def finalize_recalled(
    raw: str,
    key: str,
    registry: dict | None,
    confidence_threshold: float,
) -> dict:
    """Turn a raw model output into the recalled-entry contract dict.

    Used by the batched recall path in recall_eval.py. Returns either the
    parsed-entry dict augmented with confidence/raw_output/fact_text, or a
    failure-reason dict.
    """
    parsed = parse_recalled_entry(raw)
    if parsed is None:
        logger.debug("Parse failure for key '%s': %s", key, raw[:200])
        return {"raw_output": raw, "failure_reason": "parse_failure"}
    if parsed["key"] != key:
        logger.debug("Key mismatch: queried '%s', got '%s'", key, parsed["key"])
        return {"raw_output": raw, "failure_reason": f"key_mismatch:{parsed['key']}"}
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
    parsed["fact_text"] = entry_fact_text(parsed)
    return parsed


# --- Fact-text helper ---


def entry_fact_text(
    entry: dict,
    resolve: Callable[[str], str] | None = None,
) -> str:
    """Render a recalled entry as a human-readable fact string.

    Converts the predicate from snake_case to space-separated words (predicates
    are normalized to lowercase-underscore at extraction time per project
    convention) and joins the three triple components into a natural sentence.

    Used by inference consumers so string construction stays in the probe
    layer — callers read ``result["fact_text"]``.

    Args:
        entry: Dict containing at minimum ``subject``, ``predicate``, and
            ``object`` keys.
        resolve: Optional callable that maps a raw subject/object token to its
            display form.  When provided, ``entry["subject"]`` and
            ``entry["object"]`` are passed through it before assembly.  ``None``
            (the default) preserves byte-identical behaviour — the raw SPO
            tokens are used verbatim.  The resolver must be a pure function
            (no side-effects) and must never raise; returning the token
            unchanged for unrecognised inputs is the expected contract.
            Typical use: resolve ``speaker{N}`` ids to display names at the
            fact-render boundary (household-wide name injection).

    Returns:
        Human-readable fact string, e.g. ``"Alex lives in Heilbronn"``.
    """
    subject = resolve(entry["subject"]) if resolve is not None else entry["subject"]
    obj = resolve(entry["object"]) if resolve is not None else entry["object"]
    predicate_spaced = " ".join(entry["predicate"].replace("_", " ").split())
    return f"{subject} {predicate_spaced} {obj}"


# --- Registry persistence (re-exported for convenience) ---

# These helpers are defined in paramem.memory.persistence (the registry I/O
# module) and re-exported here so callers can import from either location.
# Triple-hop re-export chain:
#   paramem.memory.persistence  ←defines←  save_registry / load_registry / …
#   paramem.memory.entry        ←re-exports← (this block)
#   paramem.memory.__init__     ←re-exports← (package surface)
# persistence.py does NOT import entry.py → no import-time cycle.

from paramem.memory.persistence import (  # noqa: E402, F401
    get_active_keys,
    load_registry,
    save_registry,
)
