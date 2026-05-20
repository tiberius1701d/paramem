"""Legacy QA-format indexed memory functions — retired 2026-05-14.

Production retired the ``{question, answer}`` training shape on 2026-05-14.
These functions are preserved for archived experiments (Tests 1-16) calibrated
on the legacy format.  Do NOT import from production code or live tests.

Shape-agnostic helpers (``SIMHASH_BITS``, ``DEFAULT_CONFIDENCE_THRESHOLD``,
``simhash_confidence``, ``get_simhash``, etc.) now live in
:mod:`paramem.memory.entry` and :mod:`paramem.memory.persistence`.  The
functions in this module import them from their new homes.

The entry-format equivalents for each retired function live in
:mod:`paramem.memory.entry` (``compute_simhash``, ``verify_confidence``,
``build_registry``, ``assign_keys``, ``format_entry_training``, etc.).
"""

import hashlib
import json
import logging

from paramem.memory.entry import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    SIMHASH_BITS,
    _clean_generation_artifacts,
    _tokenize_features,
    get_simhash,
    simhash_confidence,
)
from paramem.training.dataset import SYSTEM_PROMPT, _tokenize_with_prompt_masking

logger = logging.getLogger(__name__)

RECALL_TEMPLATE = "Recall the QA pair stored under key '{key}'."


def compute_simhash(key: str, question: str, answer: str, num_bits: int = SIMHASH_BITS) -> int:
    """Compute a SimHash fingerprint from key + question + answer (QA format).

    The key is included so that identical content under different keys
    produces different fingerprints — catches hallucinations where the
    model echoes the queried key but returns another key's content.

    NOTE: NOT compatible with the entry-format SimHash in
    :mod:`paramem.memory.entry` — they hash different content strings.

    Returns a num_bits-bit integer fingerprint.
    """
    text = f"{key} {question} {answer}"
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
    registry: dict[str, int] | None = None,
) -> float:
    """Verify recalled QA content against a SimHash registry (QA format).

    Returns a confidence score:
    - 1.0: no registry provided (no verification)
    - 0.0: key not in registry (untrained key)
    - 0.0-1.0: SimHash similarity (higher = more likely genuine recall)
    """
    if registry is None:
        return 1.0

    key = recalled.get("key", "")
    expected = get_simhash(registry, key)
    if expected is None:
        return 0.0

    actual = compute_simhash(
        key,
        recalled.get("question", ""),
        recalled.get("answer", ""),
    )
    return simhash_confidence(actual, expected)


def build_registry(pairs: list[dict]) -> dict[str, int]:
    """Build a SimHash registry from keyed QA pairs (legacy format).

    Returns a mapping of key → 64-bit SimHash fingerprint.
    """
    return {kp["key"]: compute_simhash(kp["key"], kp["question"], kp["answer"]) for kp in pairs}


def build_enriched_registry(
    pairs: list[dict],
    session_id: str | None = None,
    existing: dict | None = None,
) -> dict:
    """Build an enriched registry with temporal metadata (legacy QA format).

    Format per key:
        {
            "simhash": int,
            "created_at": ISO timestamp,
            "last_seen_at": ISO timestamp,
            "session_id": str,
            "status": "active" | "stale",
            "stale_since": ISO timestamp | null,
            "stale_cycles": 0
        }
    """
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()
    current_keys = {kp["key"] for kp in pairs}

    if existing is None:
        existing = {}

    registry = dict(existing)

    for kp in pairs:
        key = kp["key"]
        simhash = compute_simhash(key, kp["question"], kp["answer"])

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


def assign_keys(qa_pairs: list[dict], start_index: int = 1) -> list[dict]:
    """Assign sequential keys to QA pairs (legacy format).

    Returns list of ``{"key": "graph1", "question": ..., "answer": ...}``
    plus any extra canonical fields present in the input.
    """
    keyed = []
    for i, qa in enumerate(qa_pairs, start=start_index):
        entry = {
            "key": f"graph{i}",
            "question": qa["question"],
            "answer": qa["answer"],
        }
        for extra_key in (
            "subject",
            "predicate",
            "object",
            "first_seen_cycle",
            "speaker_id",
        ):
            if extra_key in qa:
                entry[extra_key] = qa[extra_key]
        keyed.append(entry)
    return keyed


def _build_recall_response(keyed_pair: dict) -> str:
    """Build the JSON response string for a single keyed pair."""
    return json.dumps(
        {
            "key": keyed_pair["key"],
            "question": keyed_pair["question"],
            "answer": keyed_pair["answer"],
        }
    )


def format_indexed_training(pairs: list[dict], tokenizer, max_length: int = 1024) -> list[dict]:
    """Build training examples for indexed recall (legacy QA format).

    For each keyed pair, creates two training examples:
    1. Indexed recall: "Recall the QA pair stored under key 'graphN'." -> JSON
    2. Individual QA: question -> answer

    Returns pre-tokenized examples with label masking.
    """
    from paramem.models.loader import adapt_messages

    examples = []

    for kp in pairs:
        recall_prompt = RECALL_TEMPLATE.format(key=kp["key"])
        recall_response = _build_recall_response(kp)

        recall_messages = adapt_messages(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": recall_prompt},
                {"role": "assistant", "content": recall_response},
            ],
            tokenizer,
        )
        examples.append(_tokenize_with_prompt_masking(recall_messages, tokenizer, max_length))

        qa_messages = adapt_messages(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": kp["question"]},
                {"role": "assistant", "content": kp["answer"]},
            ],
            tokenizer,
        )
        examples.append(_tokenize_with_prompt_masking(qa_messages, tokenizer, max_length))

    return examples


def parse_recalled_pair(text: str) -> dict | None:
    """Parse a single recalled JSON pair from model output (QA format).

    Returns {"key", "question", "answer"} or None if unparseable.
    """
    text = text.strip()

    def _has_required_fields(data):
        return isinstance(data, dict) and "question" in data and "answer" in data

    def _try_parse(t):
        try:
            data = json.loads(t)
            if _has_required_fields(data):
                return data
        except json.JSONDecodeError:
            pass

        start = t.find("{")
        if start >= 0:
            for end in range(start + 1, len(t)):
                if t[end] == "}":
                    try:
                        data = json.loads(t[start : end + 1])
                        if _has_required_fields(data):
                            return data
                    except json.JSONDecodeError:
                        continue
        return None

    result = _try_parse(text)
    if result is not None:
        return result

    cleaned = _clean_generation_artifacts(text)
    if cleaned != text:
        return _try_parse(cleaned)

    return None


def probe_key(
    model,
    tokenizer,
    key: str,
    max_new_tokens: int = 200,
    registry: dict[str, int] | None = None,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> dict | None:
    """Prompt the model to recall a single key (legacy QA format).

    Returns a dict on success or failure — never ``None``.

    On success, returns ``{"key", "question", "answer", "confidence",
    "raw_output", "format", "fact_text"}`` where:

    * ``format`` is always ``"qa"`` for this function.
    * ``fact_text`` is ``answer``.

    On failure, returns ``{"raw_output", "failure_reason"}``.
    """
    from paramem.evaluation.recall import generate_answer
    from paramem.training.dataset import _format_inference_prompt

    prompt_text = RECALL_TEMPLATE.format(key=key)
    formatted = _format_inference_prompt(prompt_text, tokenizer)

    raw = generate_answer(
        model,
        tokenizer,
        formatted,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
    )

    recalled = parse_recalled_pair(raw)
    if recalled is None:
        logger.debug("Parse failure for key '%s': %s", key, raw[:200])
        return {"raw_output": raw, "failure_reason": "parse_failure"}

    if recalled.get("key") != key:
        logger.debug("Key mismatch: queried '%s', got '%s'", key, recalled.get("key"))
        return {"raw_output": raw, "failure_reason": f"key_mismatch:{recalled.get('key')}"}

    confidence = verify_confidence(recalled, registry)
    if confidence < confidence_threshold:
        logger.debug(
            "Low confidence for key '%s': %.3f < %.3f threshold",
            key,
            confidence,
            confidence_threshold,
        )
        return {"raw_output": raw, "failure_reason": f"low_confidence:{confidence:.3f}"}

    recalled["confidence"] = confidence
    recalled["raw_output"] = raw
    recalled["fact_text"] = recalled.get("answer", "")
    return recalled


def probe_all_keys(
    model,
    tokenizer,
    keys: list[str],
    max_new_tokens: int = 200,
    registry: dict[str, int] | None = None,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> dict[str, dict | None]:
    """Probe all keys, return mapping of key -> recalled pair (legacy QA format)."""
    return {
        key: probe_key(model, tokenizer, key, max_new_tokens, registry, confidence_threshold)
        for key in keys
    }


def validate_recall(
    recalled: dict | None,
    original: dict,
    registry: dict[str, int] | None = None,
) -> dict:
    """Compare recalled QA pair against original (legacy QA format).

    Returns dict with exact_match, question_match, answer_match,
    key_match, confidence flags and the recalled content.
    """
    if recalled is None or "failure_reason" in recalled:
        return {
            "exact_match": False,
            "question_match": False,
            "answer_match": False,
            "key_match": False,
            "confidence": 0.0,
            "recalled": recalled,
            "expected_word_count": len(original["answer"].split()),
            "recalled_word_count": 0,
        }

    key_match = recalled.get("key", "") == original.get("key", "")
    q_match = recalled.get("question", "").strip() == original["question"].strip()
    a_match = recalled.get("answer", "").strip() == original["answer"].strip()
    confidence = verify_confidence(recalled, registry)

    expected_words = len(original["answer"].split())
    recalled_words = len(recalled.get("answer", "").split())

    return {
        "exact_match": (
            q_match and a_match and key_match and confidence >= DEFAULT_CONFIDENCE_THRESHOLD
        ),
        "question_match": q_match,
        "answer_match": a_match,
        "key_match": key_match,
        "confidence": confidence,
        "recalled": recalled,
        "expected_word_count": expected_words,
        "recalled_word_count": recalled_words,
    }
