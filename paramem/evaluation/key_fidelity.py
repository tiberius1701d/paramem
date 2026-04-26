"""Per-key and per-entity reconstruction fidelity metrics.

Measures how accurately the adapter reconstructs stored graph triples
by comparing reconstructed output against the original triples.
Supports both key-based (structured triple) and entity-based (NL profile)
reconstruction.
"""

import logging

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


# Verb prefix → predicate mapping for profile parsing (reverse of sentence
# templates).  Each key is a literal lowercase prefix that the sentence
# remainder (after the entity name) must start with, followed by a space;
# the rest of the remainder becomes the object.  Order is irrelevant —
# match dispatch sorts by length so longer prefixes ("works as" before
# "works at" before just "works") are tried first.
_VERB_PREFIX_TO_PREDICATE: dict[str, str] = {
    "lives in": "lives_in",
    "works at": "works_at",
    "works as": "works_as",
    "has a pet:": "has_pet",
    "prefers": "prefers",
    "studied at": "studies_at",
    "speaks": "speaks",
    "knows": "knows",
    "enjoys": "has_hobby",
    "manages": "manages",
    "uses": "uses",
    "likes": "likes",
    "visited": "visited",
    "read": "read",
    "fixed": "fixed",
    "attended": "attended",
    "bought": "bought",
    "watched": "watched",
    "cooked": "cooked",
    "debugged": "debugged",
    "presented": "presented",
    "started": "started",
    "collaborates with": "collaborates_with",
}

# "is N…" → has_age is handled separately because the original regex
# pinned the object to start with a digit; replicated below in
# _parse_age_remainder so a literal-prefix table stays homogeneous.

# Sentence-terminating punctuation used by ``_split_sentences``.
_SENTENCE_TERMINATORS = ".!?"

# Temporal prefixes stripped from the start of a sentence (each is followed
# by some date/period text + a comma + optional whitespace).  Replicates
# the previous ``re.sub(r"^(As of |In |On )[^,]+,\s*", "", sentence)``.
_TEMPORAL_PREFIXES = ("As of ", "In ", "On ")


def _split_sentences(text: str) -> list[str]:
    """Split on sentence-terminating punctuation followed by whitespace.

    Mirrors the previous ``re.split(r"(?<=[.!?])\\s+", …)`` behaviour:
    the terminator stays attached to the preceding sentence.
    """
    if not text:
        return []
    sentences: list[str] = []
    current: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        current.append(ch)
        if ch in _SENTENCE_TERMINATORS and i + 1 < n and text[i + 1].isspace():
            sentences.append("".join(current))
            current = []
            i += 1
            while i < n and text[i].isspace():
                i += 1
            continue
        i += 1
    if current:
        sentences.append("".join(current))
    return [s.strip() for s in sentences if s.strip()]


def _strip_temporal_prefix(sentence: str) -> str:
    """Strip ``As of <…>,`` / ``In <…>,`` / ``On <…>,`` from sentence start."""
    for prefix in _TEMPORAL_PREFIXES:
        if sentence.startswith(prefix):
            comma_idx = sentence.find(",", len(prefix))
            if comma_idx >= 0:
                return sentence[comma_idx + 1 :].lstrip()
    return sentence


def _parse_age_remainder(remainder: str) -> str | None:
    """If remainder is ``is N…``, return the object ``N…``; else None.

    Replicates the previous ``r"is (\\d+.+)"`` pattern.
    """
    if not remainder.lower().startswith("is "):
        return None
    after_is = remainder[3:]
    if not after_is or not after_is[0].isdigit():
        return None
    return after_is.strip().rstrip(".") or None


def _match_verb_prefix(remainder: str) -> tuple[str, str] | None:
    """Find the longest verb prefix that matches the start of remainder.

    Returns ``(predicate, object)`` on success or ``None``.  Case-
    insensitive match on the prefix; the object preserves the original
    casing.  Replicates the previous ``re.match(pattern, remainder,
    re.IGNORECASE)`` loop with a deterministic longest-prefix-first
    dispatch.
    """
    remainder_lower = remainder.lower()
    for prefix in sorted(_VERB_PREFIX_TO_PREDICATE, key=len, reverse=True):
        prefix_with_space = prefix + " "
        if remainder_lower.startswith(prefix_with_space):
            obj = remainder[len(prefix_with_space) :].strip().rstrip(".")
            if obj:
                return _VERB_PREFIX_TO_PREDICATE[prefix], obj
    return None


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

    sentences = _split_sentences(profile_text.strip())

    for raw_sentence in sentences:
        sentence = raw_sentence.strip().rstrip(".")
        if not sentence:
            continue

        sentence = _strip_temporal_prefix(sentence)

        sentence_lower = sentence.lower()
        if not sentence_lower.startswith(entity_lower):
            continue

        remainder = sentence[len(entity_name) :].strip()

        matched = False
        verb_match = _match_verb_prefix(remainder)
        if verb_match is not None:
            predicate, obj = verb_match
            triples.append(
                {
                    "subject": entity_name,
                    "predicate": predicate,
                    "object": obj,
                }
            )
            matched = True
        elif (age_obj := _parse_age_remainder(remainder)) is not None:
            triples.append(
                {
                    "subject": entity_name,
                    "predicate": "has_age",
                    "object": age_obj,
                }
            )
            matched = True

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
