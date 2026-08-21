"""Format-neutral relation/entity preparation helpers.

Partitions preference relations to the procedural adapter, and projects a
session graph's entity scalar attributes into attribute-typed relations. No
LLM. Also hosts the two small helpers the attribute-projection surfaces
share: :func:`attribute_value_is_empty` (what counts as no information) and
:func:`strip_has_prefix` (the raw-predicate -> attribute-key reduction),
both consumed here and by the merger's node-attribute gate.
"""

from typing import TYPE_CHECKING

from paramem.graph.schema import Relation
from paramem.utils.identity import canonical

if TYPE_CHECKING:
    from paramem.graph.schema import SessionGraph

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


def attribute_value_is_empty(value) -> bool:
    """True iff an entity-attribute value carries no information.

    Treats the empty string, the literal placeholder strings ``"N/A"`` /
    ``"n/a"`` / ``"None"`` / ``"null"`` / ``"unknown"`` (case-insensitive),
    and ``None`` as empty. Shared by :func:`attribute_relations` and the
    merger's node-attribute gate, so a non-empty value captured in one
    chunk is never overwritten by an LLM-emitted placeholder from another
    chunk that happened to lack the data.

    Args:
        value: The raw attribute value (any type; only ``str``/``None`` are
            treated specially).

    Returns:
        ``True`` when *value* is ``None``, whitespace-only, or a recognised
        placeholder string; ``False`` otherwise.
    """
    if value is None:
        return True
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return True
        if stripped.lower() in ("n/a", "none", "null", "unknown"):
            return True
    return False


def strip_has_prefix(pred: str) -> str:
    """Strip leading ``has_``/``has `` tokens from *pred* to a FIXED POINT.

    Used to derive an attribute key from a raw relation predicate (e.g.
    ``has_certification`` -> ``certification``) before that key is passed
    through :func:`~paramem.utils.identity.canonical`. Recognises both the
    underscore form (``"has_certification"``) and the space-separated form
    (``"has certification"``), since this function runs on the RAW
    predicate, before any canonicalization.

    Strips repeatedly rather than once: the merger's attribute gate strips
    one ``has`` to derive the node's attribute key, :func:`attr_predicate`
    adds one back when that key is later read out as a predicate, and the
    next fold's gate strips one again — gate, walk, gate. A single-strip
    rule leaves a key that entered doubled (``"has_has_email"``) doubled
    forever, because each round trip removes only one of the two prefixes
    it carries. Stripping to a fixed point instead means every key shape
    converges to its bare form in one pass, however many ``has`` tokens it
    accumulated.

    Args:
        pred: Raw (uncanonicalized) relation predicate.

    Returns:
        *pred* with every leading ``has_``/``has `` token removed — e.g.
        ``"has_has_email"`` -> ``"email"`` — or *pred* unchanged when it
        carries neither prefix.
    """
    while True:
        if pred.startswith("has_"):
            pred = pred[len("has_") :]
        elif pred.startswith("has "):
            pred = pred[len("has ") :]
        else:
            return pred


def attr_predicate(key: str) -> str:
    """The one predicate surface for a projected attribute fact.

    ``f"has {canonical(strip_has_prefix(key))}"`` — shared by every surface
    that turns an attribute (subject, key, value) pair into a trainable
    relation predicate: :func:`attribute_relations` (projecting
    ``Entity.attributes`` at extraction time) and
    :meth:`~paramem.training.consolidation.ConsolidationLoop._build_working_keyed_walk`'s
    node-attribute walk (projecting ``GraphMerger`` node ``attributes``).
    Both surfaces MUST derive the predicate through this one function — a
    second inline copy of the formula is how the two paths silently
    diverge.

    Idempotent under repeated ``has`` prefixes: ``"has_last_name"``,
    ``"last name"``, ``"Last Name"`` and ``"has has last name"`` all yield
    ``"has last name"``, because :func:`strip_has_prefix` reduces to a
    fixed point before ``canonical()`` and ``"has "`` is prepended exactly
    once. This closes the gate -> walk -> gate round trip described on
    :func:`strip_has_prefix`: whatever shape a key arrives in, one pass
    through this function produces the one canonical predicate surface.

    Args:
        key: Raw or already-canonical attribute key.

    Returns:
        ``"has "`` followed by the canonical (space-folded, case-folded)
        form of *key* with every leading ``has`` token removed first — the
        one project-wide identity surface, never a mixed-separator glue.
    """
    return f"has {canonical(strip_has_prefix(key))}"


def attribute_relations(graph: "SessionGraph", *, speaker_id: str) -> "list[Relation]":
    """Project ``graph.entities[*].attributes`` into attribute-typed relations.

    One :class:`~paramem.graph.schema.Relation` per (entity, attribute key)
    pair with a non-empty value: ``subject=entity.name``,
    ``predicate=attr_predicate(key)``, ``object=str(value).strip()``,
    ``relation_type="attribute"``, ``confidence=1.0``,
    ``speaker_id=speaker_id``.

    Skipped:

    - values failing :func:`attribute_value_is_empty`;
    - pairs whose ``(canonical(subject), canonical(predicate))`` already
      appears among ``graph.relations`` — an explicit ``has_<key>``
      relation the extractor emitted directly takes precedence over the
      projection of the same fact;
    - pairs where ``canonical(entity.name) == canonical(value)`` — a
      self-loop, mirroring the extractor's own subject/object self-loop
      guard.

    Args:
        graph: The session graph whose entities carry the attributes to
            project. Not mutated.
        speaker_id: The speaker to attribute every projected relation to —
            the session's contributing speaker.

    Returns:
        A fresh list of :class:`~paramem.graph.schema.Relation` objects.
        May be empty when no entity carries a projectable attribute.
    """
    existing_pairs = {(canonical(r.subject), canonical(r.predicate)) for r in graph.relations}
    result: list[Relation] = []
    for entity in graph.entities:
        if not entity.attributes:
            continue
        for raw_key, attr_val in entity.attributes.items():
            if attribute_value_is_empty(attr_val):
                continue
            val_str = str(attr_val).strip()
            predicate = attr_predicate(raw_key)
            if (canonical(entity.name), canonical(predicate)) in existing_pairs:
                continue
            if canonical(entity.name) == canonical(val_str):
                continue
            result.append(
                Relation(
                    subject=entity.name,
                    predicate=predicate,
                    object=val_str,
                    relation_type="attribute",
                    confidence=1.0,
                    speaker_id=speaker_id,
                )
            )
    return result
