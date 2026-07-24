"""Format-neutral relation/entity preparation helpers.

Partitions preference relations to the procedural adapter, and projects
entity scalar attributes into relation triples.  No LLM.  Used by the
indexed-key distillation path (``ConsolidationLoop._entries_from_graph``).
"""

from typing import TYPE_CHECKING

from paramem.utils.identity import canonical

if TYPE_CHECKING:
    from paramem.graph.schema import Entity

# Supplementary predicate set for procedural filtering.
# Primary gate is relation_type == "preference"; this catches cases where
# the extractor used a preference predicate but tagged the relation as factual.
# Both this set and the incoming predicate go through canonical(), so the set
# and the comparison share one surface-form contract — a predicate the model
# emits as "has hobby" and one it emits as "has_hobby" both canonicalize to
# the same member, "has hobby".
_PROCEDURAL_PREDICATES = frozenset(
    canonical(p)
    for p in (
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
    )
)


def filter_procedural_relations(relations: list[dict]) -> list[dict]:
    """Filter relations that represent behavioral preferences or habits.

    Primary gate: relation_type == "preference" (catches model-coined predicates).
    Secondary: predicate in supplementary set (catches mis-tagged preferences).

    The secondary gate compares canonical surface forms on both sides, so
    ``"has_hobby"``, ``"has hobby"`` and ``"Has Hobby"`` all match the same
    member.
    """
    result = []
    for rel in relations:
        if rel.get("relation_type") == "preference":
            result.append(rel)
        elif canonical(rel.get("predicate", "")) in _PROCEDURAL_PREDICATES:
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


def attr_predicate(key: str) -> str:
    """The one predicate surface for a projected attribute fact: ``f"has {canonical(key)}"``.

    Shared by every surface that turns an attribute (subject, key, value)
    pair into a trainable relation-dict predicate: :func:`_flatten_entity_attributes`
    (the interim path, projecting ``Entity.attributes``) and
    :meth:`~paramem.training.consolidation.ConsolidationLoop._build_all_edge_entries_into`'s
    node-attribute walk (the full-cycle path, projecting ``GraphMerger``
    node ``attributes``). Both surfaces MUST derive the predicate through
    this one function — a second inline copy of the formula is how the two
    paths silently diverge (e.g. a node-attribute key copied verbatim from
    ``Entity.attributes`` by ``GraphMerger._upsert_entity``, still carrying
    an underscore, must be re-canonicalized here rather than glued as-is).
    ``canonical()`` is idempotent, so calling this on an already-canonical
    key (e.g. one the merger's attribute gate wrote) is a no-op.

    Args:
        key: Raw or already-canonical attribute key.

    Returns:
        ``"has "`` followed by the canonical (space-folded, case-folded)
        form of *key* — the one project-wide identity surface, never a
        mixed-separator glue.
    """
    return f"has {canonical(key)}"


def _flatten_entity_attributes(
    entities: "list[Entity]",
    *,
    exclude_pairs: "set[tuple[str, str]] | None" = None,
) -> list[dict]:
    """Project ``Entity.attributes`` into the canonical relation-dict shape.

    Internal projection used by
    :meth:`paramem.training.consolidation.ConsolidationLoop._entries_from_graph`.
    The graph's knowledge lives in two surfaces — relations and entity
    attributes — and both must reach the indexed-key distillation stage.
    This helper converts the attribute surface into the relation-dict shape,
    so the distillation input is the union of "real" relations and
    "projected" attributes.

    One projected relation is emitted per (entity, attribute_key) pair:

        {
            "subject": entity.name,
            "predicate": "has <normalised_key>",
            "object": str(attr_val),
            "relation_type": "attribute",
        }

    Predicate normalisation goes through :func:`attr_predicate` — the ONE
    formula (``f"has {canonical(key)}"``) shared with
    :meth:`~paramem.training.consolidation.ConsolidationLoop._build_all_edge_entries_into`'s
    node-attribute walk (the full-cycle path), so the two surfaces an
    attribute fact can be trained under never diverge and share one
    SimHash fingerprint.

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
            predicate = attr_predicate(raw_key)
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
