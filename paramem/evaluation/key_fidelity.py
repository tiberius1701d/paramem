"""Per-key and per-entity reconstruction fidelity metrics.

Measures how accurately the adapter reconstructs stored graph triples
by comparing reconstructed output against the original triples.
Supports both key-based (structured triple) and entity-based (NL profile)
reconstruction.
"""

import logging
import re

from paramem.graph.merger import _normalize_predicate

logger = logging.getLogger(__name__)


def _normalize_triple(triple: dict) -> tuple[str, str, str]:
    """Normalize a triple for comparison."""
    return (
        triple["subject"].strip().lower(),
        _normalize_predicate(triple["predicate"]),
        triple["object"].strip().lower(),
    )


def measure_fidelity(
    original_triples: list[dict],
    reconstructed_triples: list[dict],
) -> dict:
    """Measure reconstruction fidelity as precision and recall on triples.

    Args:
        original_triples: Ground-truth triples for a key.
        reconstructed_triples: Triples reconstructed from the adapter.

    Returns:
        Dict with precision, recall, f1, and counts.
    """
    original_set = {_normalize_triple(t) for t in original_triples}
    reconstructed_set = {_normalize_triple(t) for t in reconstructed_triples}

    if not original_set and not reconstructed_set:
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "original_count": 0,
            "reconstructed_count": 0,
            "matched_count": 0,
        }

    matched = original_set & reconstructed_set
    matched_count = len(matched)

    precision = matched_count / len(reconstructed_set) if reconstructed_set else 0.0
    recall = matched_count / len(original_set) if original_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "original_count": len(original_set),
        "reconstructed_count": len(reconstructed_set),
        "matched_count": matched_count,
    }


def measure_all_fidelity(
    original_triples_by_key: dict[str, list[dict]],
    reconstructed_triples_by_key: dict[str, list[dict]],
) -> dict:
    """Measure fidelity across all keys.

    Args:
        original_triples_by_key: Key -> original triples.
        reconstructed_triples_by_key: Key -> reconstructed triples.

    Returns:
        Dict with per-key metrics and aggregate summary.
    """
    per_key = {}
    for key in original_triples_by_key:
        original = original_triples_by_key[key]
        reconstructed = reconstructed_triples_by_key.get(key, [])
        per_key[key] = measure_fidelity(original, reconstructed)

    if not per_key:
        return {
            "per_key": {},
            "mean_precision": 0.0,
            "mean_recall": 0.0,
            "mean_f1": 0.0,
            "num_keys": 0,
        }

    mean_precision = sum(m["precision"] for m in per_key.values()) / len(per_key)
    mean_recall = sum(m["recall"] for m in per_key.values()) / len(per_key)
    mean_f1 = sum(m["f1"] for m in per_key.values()) / len(per_key)

    logger.info(
        "Fidelity across %d keys: precision=%.3f, recall=%.3f, f1=%.3f",
        len(per_key),
        mean_precision,
        mean_recall,
        mean_f1,
    )

    return {
        "per_key": per_key,
        "mean_precision": mean_precision,
        "mean_recall": mean_recall,
        "mean_f1": mean_f1,
        "num_keys": len(per_key),
    }


# Verb→predicate mappings for profile parsing (reverse of sentence templates)
_VERB_PATTERNS = [
    (r"lives in (.+)", "lives_in"),
    (r"works at (.+)", "works_at"),
    (r"works as (.+)", "works_as"),
    (r"has a pet: (.+)", "has_pet"),
    (r"prefers (.+)", "prefers"),
    (r"studied at (.+)", "studies_at"),
    (r"speaks (.+)", "speaks"),
    (r"knows (.+)", "knows"),
    (r"enjoys (.+)", "has_hobby"),
    (r"manages (.+)", "manages"),
    (r"is (\d+.+)", "has_age"),
    (r"uses (.+)", "uses"),
    (r"likes (.+)", "likes"),
    (r"visited (.+)", "visited"),
    (r"read (.+)", "read"),
    (r"fixed (.+)", "fixed"),
    (r"attended (.+)", "attended"),
    (r"bought (.+)", "bought"),
    (r"watched (.+)", "watched"),
    (r"cooked (.+)", "cooked"),
    (r"debugged (.+)", "debugged"),
    (r"presented (.+)", "presented"),
    (r"started (.+)", "started"),
    (r"collaborates with (.+)", "collaborates_with"),
]


def parse_profile_to_triples(
    entity_name: str,
    profile_text: str,
) -> list[dict]:
    """Parse a natural language entity profile back into triples.

    Uses pattern matching to extract (subject, predicate, object) triples
    from profile sentences. This is a lightweight parser for reconstruction
    fidelity measurement — not a full NER/RE pipeline.

    Handles common sentence patterns:
    - "{Subject} lives in {Object}."
    - "{Subject} works at {Object}."
    - "{Subject} {predicate} {Object}."

    Args:
        entity_name: The entity whose profile is being parsed.
        profile_text: Natural language profile text.

    Returns:
        List of triple dicts with subject, predicate, object.
    """
    if not profile_text.strip():
        return []

    triples = []
    entity_lower = entity_name.lower()

    # Split profile into sentences
    sentences = re.split(r"(?<=[.!?])\s+", profile_text.strip())

    for sentence in sentences:
        sentence = sentence.strip().rstrip(".")
        if not sentence:
            continue

        # Strip temporal prefixes
        sentence = re.sub(r"^(As of |In |On )[^,]+,\s*", "", sentence)

        # Check if sentence starts with the entity name
        sentence_lower = sentence.lower()
        if not sentence_lower.startswith(entity_lower):
            continue

        remainder = sentence[len(entity_name) :].strip()

        # Try to match against known verb patterns
        matched = False
        for pattern, predicate in _VERB_PATTERNS:
            match = re.match(pattern, remainder, re.IGNORECASE)
            if match:
                obj = match.group(1).strip().rstrip(".")
                if obj:
                    triples.append(
                        {
                            "subject": entity_name,
                            "predicate": predicate,
                            "object": obj,
                        }
                    )
                    matched = True
                    break

        # Fallback: extract generic "verb object" pattern
        if not matched and remainder:
            parts = remainder.split(None, 1)
            if len(parts) == 2:
                verb, obj = parts
                obj = obj.strip().rstrip(".")
                if obj:
                    triples.append(
                        {
                            "subject": entity_name,
                            "predicate": _normalize_predicate(verb),
                            "object": obj,
                        }
                    )

    return triples


def measure_entity_fidelity(
    entity_name: str,
    reconstructed_text: str,
    ground_truth_relations: list[dict],
) -> dict:
    """Measure entity reconstruction fidelity.

    Parses the NL reconstruction back to triples and compares against
    the entity's known ground-truth relations.

    Args:
        entity_name: The entity being evaluated.
        reconstructed_text: NL profile from the adapter.
        ground_truth_relations: Known relations for this entity.

    Returns:
        Dict with precision, recall, f1, and counts.
    """
    reconstructed_triples = parse_profile_to_triples(entity_name, reconstructed_text)
    return measure_fidelity(ground_truth_relations, reconstructed_triples)
