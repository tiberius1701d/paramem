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
    expected = get_simhash(registry, key)
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
    This is the simple format used by the training pipeline.
    For enriched metadata, see build_enriched_registry().
    """
    return {
        kp["key"]: compute_simhash(kp["key"], kp["question"], kp["answer"]) for kp in keyed_pairs
    }


def build_enriched_registry(
    keyed_pairs: list[dict],
    session_id: str | None = None,
    existing: dict | None = None,
) -> dict:
    """Build an enriched registry with temporal metadata.

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

    If existing registry is provided, updates it:
    - Active keys present in keyed_pairs get their last_seen_at updated
    - Active keys NOT in keyed_pairs are unchanged (not auto-staled)
    - Stale keys NOT in keyed_pairs get stale_cycles incremented
    - Stale keys present in keyed_pairs are reactivated
    """
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()
    current_keys = {kp["key"] for kp in keyed_pairs}

    if existing is None:
        existing = {}

    registry = dict(existing)

    # Update or add keys from current training set
    for kp in keyed_pairs:
        key = kp["key"]
        simhash = compute_simhash(key, kp["question"], kp["answer"])

        if key in registry:
            # Reactivate if stale, update last_seen
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

    # Increment stale_cycles for keys already marked stale
    for key, entry in registry.items():
        if key not in current_keys and entry.get("status") == "stale":
            entry["stale_cycles"] = entry.get("stale_cycles", 0) + 1

    return registry


def mark_stale(registry: dict, keys: list[str]) -> None:
    """Mark keys as stale in an enriched registry.

    Stale keys are excluded from enumeration but remain in the registry
    until their key IDs are reclaimed. The adapter retains the old
    weights — they decay naturally through subsequent training cycles.
    """
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()
    for key in keys:
        if key in registry and isinstance(registry[key], dict):
            registry[key]["status"] = "stale"
            if registry[key].get("stale_since") is None:
                registry[key]["stale_since"] = now


def get_reclaimable_keys(registry: dict, min_stale_cycles: int = 5) -> list[str]:
    """Return stale key IDs that have been inactive long enough to reclaim.

    After min_stale_cycles consolidation cycles, the adapter's gradient
    residue for these keys has likely decayed enough to reassign them.
    """
    reclaimable = []
    for key, entry in registry.items():
        if (
            isinstance(entry, dict)
            and entry.get("status") == "stale"
            and entry.get("stale_cycles", 0) >= min_stale_cycles
        ):
            reclaimable.append(key)
    return sorted(reclaimable)


def get_active_keys(registry: dict) -> list[str]:
    """Return only active (non-stale) keys from an enriched registry.

    Also handles the simple format (key → int) for backward compatibility.
    """
    active = []
    for key, entry in registry.items():
        if isinstance(entry, int):
            active.append(key)
        elif isinstance(entry, dict) and entry.get("status", "active") == "active":
            active.append(key)
    return sorted(active)


def get_simhash(registry: dict, key: str) -> int | None:
    """Extract simhash from either simple or enriched registry format."""
    entry = registry.get(key)
    if entry is None:
        return None
    if isinstance(entry, int):
        return entry
    if isinstance(entry, dict):
        return entry.get("simhash")
    return None


def save_registry(registry: dict, path: str | Path) -> None:
    """Save registry to a JSON file. Handles both simple and enriched formats."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(registry, f, indent=2)


def load_registry(path: str | Path) -> dict:
    """Load registry from a JSON file. Handles both simple and enriched formats."""
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
        # Preserve source and cohort metadata
        for extra_key in (
            "source_subject",
            "source_predicate",
            "source_object",
            "first_seen_cycle",
            "source_character",
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
    from paramem.models.loader import adapt_messages

    examples = []

    for kp in keyed_pairs:
        # Indexed recall example
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

        # Individual QA example (proven format)
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


# --- Recall parsing and probing ---


def _clean_generation_artifacts(text: str) -> str:
    """Strip markdown formatting artifacts from model output.

    Some instruct models (notably Gemma 2) inject ** bold markers and
    excess newlines into generated JSON, breaking structure. This cleans
    the text before JSON parsing.

    NOTE: Artifact patterns may evolve across reinforcement/consolidation
    cycles as the model retrains on its own output. Monitor parse failure
    rates across cycles and extend patterns here if new artifacts emerge.
    """
    import re

    text = text.replace("**", "")
    text = re.sub(r"\n{3,}", "\n", text)
    return text


def parse_recalled_pair(text: str) -> dict | None:
    """Parse a single recalled JSON pair from model output.

    Returns {"key", "question", "answer"} or None if unparseable.
    """
    text = text.strip()

    def _has_required_fields(data):
        return isinstance(data, dict) and "question" in data and "answer" in data

    def _try_parse(t):
        # Try direct parse
        try:
            data = json.loads(t)
            if _has_required_fields(data):
                return data
        except json.JSONDecodeError:
            pass

        # Try extracting the first JSON object from surrounding text.
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

    # Try raw text first, then cleaned
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
    """Prompt the model to recall a single key.

    Returns parsed {"key", "question", "answer", "confidence", "raw_output"}
    dict, or None if the output is unparseable, the key mismatches, or
    confidence falls below the threshold. The raw_output field is always
    present when a dict is returned. For failures, returns a dict with
    "raw_output" and "failure_reason" so diagnostic data is never lost.

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
        temperature=0.0,
    )

    recalled = parse_recalled_pair(raw)
    if recalled is None:
        logger.debug("Parse failure for key '%s': %s", key, raw[:200])
        return {"raw_output": raw, "failure_reason": "parse_failure"}

    # Reject if returned key doesn't match queried key
    if recalled.get("key") != key:
        logger.debug("Key mismatch: queried '%s', got '%s'", key, recalled.get("key"))
        return {"raw_output": raw, "failure_reason": f"key_mismatch:{recalled.get('key')}"}

    # Check confidence against registry
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
    return recalled


def probe_all_keys(
    model,
    tokenizer,
    keys: list[str],
    max_new_tokens: int = 200,
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
    Failure dicts (with "failure_reason") are treated as non-matches
    but preserve raw_output for diagnostics.
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
