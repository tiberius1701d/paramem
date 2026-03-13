"""Indexed key memory — per-fact addressable recall from adapter weights.

Assigns each QA pair a unique key (graph1, graph2, ...) and trains the model
to recall individual facts by key. Enables per-fact readout, capacity detection,
and targeted reinforcement without external storage.

Training format per key:
  User: "Recall the QA pair stored under key 'graph3'."
  Assistant: {"key": "graph3", "question": "...", "answer": "..."}

Hallucination detection uses an external SimHash registry — a lightweight
key→fingerprint mapping saved alongside the adapter. SimHash is a locality-
sensitive hash: similar content produces similar fingerprints, so verification
returns a continuous confidence score (0.0-1.0) rather than a binary pass/fail.
This tolerates minor recall variations while rejecting hallucinated content.
"""

import hashlib
import json
import logging
from pathlib import Path

from paramem.training.dataset import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

RECALL_TEMPLATE = "Recall the QA pair stored under key '{key}'."
SIMHASH_BITS = 64
DEFAULT_CONFIDENCE_THRESHOLD = 0.75


# --- SimHash fingerprinting ---


def _tokenize_features(text: str) -> list[str]:
    """Tokenize text into word unigrams and bigrams for SimHash."""
    words = text.lower().split()
    features = list(words)
    for i in range(len(words) - 1):
        features.append(f"{words[i]} {words[i + 1]}")
    return features


def compute_simhash(key: str, question: str, answer: str, num_bits: int = SIMHASH_BITS) -> int:
    """Compute a SimHash fingerprint from key + question + answer.

    The key is included so that identical content under different keys
    produces different fingerprints — catches hallucinations where the
    model echoes the queried key but returns another key's content.

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


def simhash_confidence(hash_a: int, hash_b: int, num_bits: int = SIMHASH_BITS) -> float:
    """Compute similarity confidence from two SimHash fingerprints.

    Returns 1.0 for identical fingerprints, ~0.5 for unrelated content.
    Based on normalized Hamming distance.
    """
    distance = bin(hash_a ^ hash_b).count("1")
    return 1.0 - (distance / num_bits)


def verify_confidence(
    recalled: dict,
    registry: dict[str, int] | None = None,
) -> float:
    """Verify recalled content against a SimHash registry.

    Returns a confidence score:
    - 1.0: no registry provided (no verification)
    - 0.0: key not in registry (untrained key)
    - 0.0-1.0: SimHash similarity (higher = more likely genuine recall)
    """
    if registry is None:
        return 1.0

    key = recalled.get("key", "")
    expected = registry.get(key)
    if expected is None:
        return 0.0

    actual = compute_simhash(
        key,
        recalled.get("question", ""),
        recalled.get("answer", ""),
    )
    return simhash_confidence(actual, expected)


# --- Registry management ---


def build_registry(keyed_pairs: list[dict]) -> dict[str, int]:
    """Build a SimHash registry from keyed QA pairs.

    Returns a mapping of key → 64-bit SimHash fingerprint.
    """
    return {
        kp["key"]: compute_simhash(kp["key"], kp["question"], kp["answer"]) for kp in keyed_pairs
    }


def save_registry(registry: dict[str, int], path: str | Path) -> None:
    """Save SimHash registry to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(registry, f, indent=2)


def load_registry(path: str | Path) -> dict[str, int]:
    """Load SimHash registry from a JSON file."""
    with open(path) as f:
        return json.load(f)


# --- Key assignment and training format ---


def assign_keys(qa_pairs: list[dict], start_index: int = 1) -> list[dict]:
    """Assign sequential keys to QA pairs.

    Returns list of {"key": "graph1", "question": ..., "answer": ...}.
    Extra fields (e.g. source_subject) are preserved for metadata tracking.
    """
    keyed = []
    for i, qa in enumerate(qa_pairs, start=start_index):
        entry = {
            "key": f"graph{i}",
            "question": qa["question"],
            "answer": qa["answer"],
        }
        # Preserve source metadata for promotion tracking
        for extra_key in ("source_subject", "source_predicate", "source_object"):
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


def _tokenize_with_prompt_masking(messages: list[dict], tokenizer, max_length: int) -> dict:
    """Tokenize a chat message list with prompt token masking.

    Returns {"input_ids", "attention_mask", "labels"} with prompt tokens
    masked to -100 in labels.
    """
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_text = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )

    full_enc = tokenizer(full_text, truncation=True, max_length=max_length, return_tensors="pt")
    prompt_enc = tokenizer(prompt_text, truncation=True, max_length=max_length, return_tensors="pt")

    input_ids = full_enc["input_ids"].squeeze()
    attention_mask = full_enc["attention_mask"].squeeze()
    prompt_length = prompt_enc["input_ids"].shape[1]

    labels = input_ids.clone()
    labels[:prompt_length] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def format_indexed_training(
    keyed_pairs: list[dict], tokenizer, max_length: int = 1024
) -> list[dict]:
    """Build training examples for indexed recall.

    For each keyed pair, creates two training examples:
    1. Indexed recall: "Recall the QA pair stored under key 'graphN'." -> JSON
    2. Individual QA: question -> answer (standard format for backward compat)

    Returns pre-tokenized examples with label masking.
    """
    examples = []

    for kp in keyed_pairs:
        # Indexed recall example
        recall_prompt = RECALL_TEMPLATE.format(key=kp["key"])
        recall_response = _build_recall_response(kp)

        recall_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": recall_prompt},
            {"role": "assistant", "content": recall_response},
        ]
        examples.append(_tokenize_with_prompt_masking(recall_messages, tokenizer, max_length))

        # Individual QA example (proven format)
        qa_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": kp["question"]},
            {"role": "assistant", "content": kp["answer"]},
        ]
        examples.append(_tokenize_with_prompt_masking(qa_messages, tokenizer, max_length))

    return examples


# --- Recall parsing and probing ---


def parse_recalled_pair(text: str) -> dict | None:
    """Parse a single recalled JSON pair from model output.

    Returns {"key", "question", "answer"} or None if unparseable.
    """
    text = text.strip()

    def _has_required_fields(data):
        return isinstance(data, dict) and "question" in data and "answer" in data

    # Try direct parse
    try:
        data = json.loads(text)
        if _has_required_fields(data):
            return data
    except json.JSONDecodeError:
        pass

    # Try extracting the first JSON object from surrounding text.
    # Use find (not rfind) for "}" to avoid spanning multiple objects
    # when the model generates garbage after the first valid JSON.
    start = text.find("{")
    if start >= 0:
        for end in range(start + 1, len(text)):
            if text[end] == "}":
                try:
                    data = json.loads(text[start : end + 1])
                    if _has_required_fields(data):
                        return data
                except json.JSONDecodeError:
                    continue

    return None


def probe_key(
    model,
    tokenizer,
    key: str,
    max_new_tokens: int = 256,
    registry: dict[str, int] | None = None,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> dict | None:
    """Prompt the model to recall a single key.

    Returns parsed {"key", "question", "answer"} with an added "confidence"
    field, or None if the output is unparseable, the key mismatches, or
    confidence falls below the threshold.

    If registry is provided, verifies the recalled content against it.
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
        temperature=0.1,
        repetition_penalty=1.3,
    )

    recalled = parse_recalled_pair(raw)
    if recalled is None:
        return None

    # Reject if returned key doesn't match queried key
    if recalled.get("key") != key:
        logger.debug("Key mismatch: queried '%s', got '%s'", key, recalled.get("key"))
        return None

    # Check confidence against registry
    confidence = verify_confidence(recalled, registry)
    if confidence < confidence_threshold:
        logger.debug(
            "Low confidence for key '%s': %.3f < %.3f threshold",
            key,
            confidence,
            confidence_threshold,
        )
        return None

    recalled["confidence"] = confidence
    return recalled


def probe_all_keys(
    model,
    tokenizer,
    keys: list[str],
    max_new_tokens: int = 256,
    registry: dict[str, int] | None = None,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> dict[str, dict | None]:
    """Probe all keys, return mapping of key -> recalled pair or None."""
    return {
        key: probe_key(model, tokenizer, key, max_new_tokens, registry, confidence_threshold)
        for key in keys
    }


def validate_recall(
    recalled: dict | None,
    original: dict,
    registry: dict[str, int] | None = None,
) -> dict:
    """Compare recalled pair against original.

    Returns dict with exact_match, question_match, answer_match,
    key_match, confidence (float) flags and the recalled content.
    """
    if recalled is None:
        return {
            "exact_match": False,
            "question_match": False,
            "answer_match": False,
            "key_match": False,
            "confidence": 0.0,
            "recalled": None,
        }

    key_match = recalled.get("key", "") == original.get("key", "")
    q_match = recalled.get("question", "").strip() == original["question"].strip()
    a_match = recalled.get("answer", "").strip() == original["answer"].strip()
    confidence = verify_confidence(recalled, registry)

    return {
        "exact_match": (
            q_match and a_match and key_match and confidence >= DEFAULT_CONFIDENCE_THRESHOLD
        ),
        "question_match": q_match,
        "answer_match": a_match,
        "key_match": key_match,
        "confidence": confidence,
        "recalled": recalled,
    }
