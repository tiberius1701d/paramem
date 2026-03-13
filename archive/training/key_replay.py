"""Key-addressable replay for consolidation.

Stores knowledge graphs in adapter weights with unique retrieval keys.
During replay, the model reconstructs graphs from its own weights rather
than reading from external files. Adapter weights are the single source
of truth — no external QA pair storage.
"""

import logging
import re

from paramem.evaluation.recall import generate_answer
from paramem.graph.merger import _normalize_predicate
from paramem.training.key_registry import KeyRegistry

logger = logging.getLogger(__name__)

# Training format: keyed memory block with graph triples
_MEMORY_TEMPLATE = '<memory key="{key}">\n{triples}\n</memory>'
_TRIPLE_FORMAT = "({subject}, {predicate}, {object})"

# Reconstruction prompt
_RECALL_SYSTEM = (
    "You are a memory system. When given a memory key, recall all knowledge "
    "stored under that key. Output as structured graph triples in the format: "
    "(subject, predicate, object), one per line. Output only the triples."
)
_RECALL_PROMPT = 'Recall all knowledge stored under key "{key}".'


def format_triples(relations: list[dict]) -> str:
    """Format relations as triple strings, one per line."""
    return "\n".join(
        _TRIPLE_FORMAT.format(
            subject=rel["subject"],
            predicate=rel["predicate"],
            object=rel["object"],
        )
        for rel in relations
    )


def format_keyed_training(key: str, relations: list[dict]) -> list[dict]:
    """Format keyed triples as training examples.

    Each key maps to a single training example where the user asks for
    the memory key and the assistant produces the structured triples.

    Args:
        key: Unique memory key (e.g. session ID).
        relations: List of relation dicts with subject, predicate, object.

    Returns:
        List of training dicts with 'question' and 'answer' keys,
        compatible with the consolidation training pipeline.
    """
    if not relations:
        return []

    triples_text = format_triples(relations)
    memory_block = _MEMORY_TEMPLATE.format(key=key, triples=triples_text)

    return [
        {
            "question": _RECALL_PROMPT.format(key=key),
            "answer": memory_block,
        }
    ]


def parse_triples(text: str) -> list[dict]:
    """Parse reconstructed triples from model output.

    Handles both clean (subject, predicate, object) format and
    noisy model output with extra whitespace or formatting.

    Returns:
        List of dicts with subject, predicate, object keys.
    """
    triples = []
    # Match (subject, predicate, object) patterns
    pattern = re.compile(r"\(\s*([^,]+?)\s*,\s*([^,]+?)\s*,\s*([^)]+?)\s*\)")
    for match in pattern.finditer(text):
        subject = match.group(1).strip()
        predicate = _normalize_predicate(match.group(2).strip())
        obj = match.group(3).strip()
        if subject and predicate and obj:
            triples.append(
                {
                    "subject": subject,
                    "predicate": predicate,
                    "object": obj,
                }
            )
    return triples


def reconstruct_key(
    model,
    tokenizer,
    key: str,
    temperature: float = 0.1,
    max_new_tokens: int = 512,
) -> list[dict]:
    """Prompt the model to reconstruct triples for a given key.

    Args:
        model: The adapted model (gradient checkpointing must be disabled).
        tokenizer: Model tokenizer.
        key: The memory key to reconstruct.
        temperature: Generation temperature (low for faithful reconstruction).
        max_new_tokens: Max tokens for reconstruction output.

    Returns:
        List of parsed triple dicts.
    """
    messages = [
        {"role": "system", "content": _RECALL_SYSTEM},
        {"role": "user", "content": _RECALL_PROMPT.format(key=key)},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    output = generate_answer(
        model,
        tokenizer,
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    triples = parse_triples(output)
    logger.debug(
        "Reconstructed key '%s': %d triples from %d chars output",
        key,
        len(triples),
        len(output),
    )
    return triples


def reconstruct_all(
    model,
    tokenizer,
    registry: KeyRegistry,
    max_keys: int = 50,
    temperature: float = 0.1,
) -> dict[str, list[dict]]:
    """Reconstruct all active keys from the adapter.

    Args:
        model: The adapted model.
        tokenizer: Model tokenizer.
        registry: Active key registry.
        max_keys: Maximum keys to reconstruct (caps latency).
        temperature: Generation temperature.

    Returns:
        Dict mapping key -> list of triple dicts.
    """
    active_keys = registry.list_active()
    if len(active_keys) > max_keys:
        logger.warning(
            "Capping reconstruction from %d to %d keys",
            len(active_keys),
            max_keys,
        )
        active_keys = active_keys[:max_keys]

    reconstructions = {}
    for key in active_keys:
        triples = reconstruct_key(model, tokenizer, key, temperature=temperature)
        reconstructions[key] = triples

    total_triples = sum(len(t) for t in reconstructions.values())
    logger.info(
        "Reconstructed %d keys, %d total triples",
        len(reconstructions),
        total_triples,
    )
    return reconstructions


def merge_reconstructions_with_session(
    reconstructions: dict[str, list[dict]],
    session_relations: list[dict],
    session_key: str,
    similarity_threshold: float = 85.0,
) -> list[dict]:
    """Merge all reconstructed triples with the new session's relations.

    Uses normalized comparison to deduplicate across reconstructions
    and the new session. Returns a flat list of deduplicated relations.

    Args:
        reconstructions: Key -> triples from reconstruct_all.
        session_relations: Relations from the current session extraction.
        session_key: Key for the current session.
        similarity_threshold: Entity resolution threshold (unused, kept for compatibility).

    Returns:
        Deduplicated list of relation dicts.
    """
    # Deduplicate by normalized key: (subject, predicate, object)
    seen = set()
    deduplicated = []

    # Process reconstructions first
    for triples in reconstructions.values():
        for rel in triples:
            norm_key = (
                rel["subject"].strip().lower(),
                _normalize_predicate(rel["predicate"]),
                rel["object"].strip().lower(),
            )
            if norm_key not in seen:
                seen.add(norm_key)
                deduplicated.append(rel)

    # Then process session relations
    for rel in session_relations:
        norm_key = (
            rel["subject"].strip().lower(),
            _normalize_predicate(rel["predicate"]),
            rel["object"].strip().lower(),
        )
        if norm_key not in seen:
            seen.add(norm_key)
            deduplicated.append(rel)

    logger.info(
        "Merged reconstructions + session: %d deduplicated relations",
        len(deduplicated),
    )
    return deduplicated


def retire_stale_keys(
    registry: KeyRegistry,
    threshold: float = 0.1,
    consecutive_cycles: int = 3,
) -> list[str]:
    """Retire keys with sustained low reconstruction fidelity.

    Returns list of retired key names.
    """
    retired = []
    for key in registry.list_active():
        if registry.should_retire(key, threshold, consecutive_cycles):
            registry.remove(key)
            retired.append(key)
            logger.info(
                "Retired key '%s' (fidelity below %.2f for %d cycles)",
                key,
                threshold,
                consecutive_cycles,
            )
    return retired
