"""Format-neutral relation/entity preparation helpers.

Partitions preference relations to the procedural adapter, and projects
entity scalar attributes into relation triples.  No LLM.  Used by both
``qa_generator.py`` (for human-readable fact text) and the indexed-key path.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from paramem.graph.schema import Entity

# Supplementary predicate set for procedural filtering.
# Primary gate is relation_type == "preference"; this catches cases where
# the extractor used a preference predicate but tagged the relation as factual.
_PROCEDURAL_PREDICATES = frozenset(
    {
        "prefers",
        "likes",
        "dislikes",
        "has_hobby",
        "drinks",
        "eats",
        "watches",
        "listens_to",
        "avoids",
        "favorite",
    }
)


def filter_procedural_relations(relations: list[dict]) -> list[dict]:
    """Filter relations that represent behavioral preferences or habits.

    Primary gate: relation_type == "preference" (catches model-coined predicates).
    Secondary: predicate in supplementary set (catches mis-tagged preferences).
    """
    result = []
    for rel in relations:
        if rel.get("relation_type") == "preference":
            result.append(rel)
        elif rel.get("predicate", "").lower() in _PROCEDURAL_PREDICATES:
            result.append(rel)
    return result


def partition_relations(
    relations: list[dict], procedural_enabled: bool
) -> tuple[list[dict], list[dict]]:
    """Split session relations into (episodic, procedural) sets.

    When procedural_enabled=True, preference relations route to the procedural
    adapter and are removed from the episodic set to avoid duplicate encoding.
    When procedural_enabled=False, everything stays in episodic so preferences
    are never lost.

    Called per-extraction so config changes are picked up automatically.
    """
    if not procedural_enabled:
        return list(relations), []
    procedural = filter_procedural_relations(relations)
    proc_ids = {id(r) for r in procedural}
    episodic = [r for r in relations if id(r) not in proc_ids]
    return episodic, procedural


def _flatten_entity_attributes(
    entities: "list[Entity]",
    *,
    exclude_pairs: "set[tuple[str, str]] | None" = None,
) -> list[dict]:
    """Project ``Entity.attributes`` into the canonical relation-dict shape.

    Internal projection used by :func:`generate_qa_from_graph`.  The graph's
    knowledge lives in two surfaces — relations and entity attributes — and
    both must reach the QA generator.  This helper converts the attribute
    surface into the relation-dict shape, so the QA generator's input is the
    union of "real" relations and "projected" attributes.

    One projected relation is emitted per (entity, attribute_key) pair:

        {
            "subject": entity.name,
            "predicate": "has_<normalised_key>",
            "object": str(attr_val),
            "relation_type": "attribute",
        }

    Predicate normalisation: attribute keys are lower-cased and
    spaces/dashes replaced with underscores (``"phone number"`` →
    ``"has_phone_number"``).

    Pairs whose ``(subject, predicate)`` already appears in ``exclude_pairs``
    are skipped — prevents duplicate keying when an explicit ``has_<key>``
    relation was already extracted.  Pairs with ``None`` or whitespace-only
    values are skipped.  Input entities are not mutated.
    """
    _exclude = exclude_pairs if exclude_pairs is not None else set()
    result: list[dict] = []
    for entity in entities:
        if not entity.attributes:
            continue
        for raw_key, attr_val in entity.attributes.items():
            # Skip empty values
            if attr_val is None:
                continue
            val_str = str(attr_val).strip()
            if not val_str:
                continue
            # Normalise predicate: lowercase + spaces/dashes → underscores
            norm_key = raw_key.lower().replace(" ", "_").replace("-", "_")
            predicate = f"has_{norm_key}"
            pair = (entity.name, predicate)
            if pair in _exclude:
                continue
            result.append(
                {
                    "subject": entity.name,
                    "predicate": predicate,
                    "object": val_str,
                    "relation_type": "attribute",
                }
            )
    return result
