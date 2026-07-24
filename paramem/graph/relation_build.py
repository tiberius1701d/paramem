"""Pure post-processing from a de-anonymized fact list to a ``SessionGraph``.

Boundary / why this module exists
---------------------------------

Everything here is a total function of its arguments: facts in, relations
and entities out. There is no cloud call, no model, no tokenizer, no
config and no prompt — the most testable unit of the extraction flow, and
(before the split) the least tested one, because it lived buried in the
middle of ``_cloud_pipeline`` where reaching it required standing up the
whole anonymize → enrich → judge chain first.

It owns three things:

* ``build_relations`` — schema validation of the surviving fact dicts,
  with the drops recorded rather than swallowed.
* ``apply_rebuild`` — the entity surface that must accompany a relation
  set (pruning, placeholder-derived typing of names the cloud minted).
  Literal-value facts (phone/email/date/certification/job title, ...) are
  no longer routed here by object shape: the model tags them
  ``relation_type="attribute"`` at extraction time, and
  :class:`~paramem.graph.merger.GraphMerger` (the one merge boundary every
  path crosses) folds them onto the subject node's ``attributes`` dict —
  see that module's ``relation_type == "attribute"`` branch in ``merge()``.
* ``recovery_gate`` — the all-dropped safety net's DECISION and its cause
  bookkeeping. The recovery ACTION (re-judging the pre-enrichment facts
  on the local model) stays with its caller: it needs the model and the
  tokenizer, which this module deliberately never sees.

The ``CAUSE_*`` vocabulary (and its ``cause_kind`` classifier) used to be
defined here on the reasoning that the gate was its only consumer; it now
lives in :mod:`paramem.graph.empty_cause` because it describes FLOW STATE
(the extraction flow's ``StageState.empty_cause``), not relation
rebuilding, and has three consumers: :func:`recovery_gate` below, the flow
tail stages (``paramem.graph.flows``), and the ``enrich`` stage
(``paramem.graph.stage_enrich``). This module imports the pieces
:func:`recovery_gate` needs back from there.
"""

from __future__ import annotations

import logging

from paramem.config.taxonomy import placeholder_entity_type
from paramem.graph.empty_cause import CAUSE_UNATTRIBUTED, cause_kind
from paramem.graph.schema import Entity, Relation, SessionGraph

logger = logging.getLogger(__name__)


def build_relations(
    graph: SessionGraph,
    facts: list[dict],
    *,
    speaker_id: str,
) -> list[Relation]:
    """Validate ``facts`` into :class:`Relation` objects, recording drops.

    A fact that fails ``Relation`` construction (commonly a
    ``relation_type`` outside the schema's ``Literal`` set) is dropped and
    recorded on ``graph.diagnostics["pydantic_validation_dropped"]`` as a
    ``{subject, predicate, object, relation_type, reason}`` dict — the
    drop is never silent.

    Args:
        graph: The session graph whose ``diagnostics`` receive the drop
            record. ``graph.relations`` is NOT modified here — installing
            the result is :func:`apply_rebuild`'s job.
        facts: De-anonymized fact dicts surviving every upstream filter.
        speaker_id: Speaker store ID stamped onto every built relation as
            provenance.

    Returns:
        The relations that validated, in input order.
    """
    kept: list[Relation] = []
    validation_dropped: list[dict] = []
    for fact in facts:
        try:
            kept.append(
                Relation(
                    subject=fact.get("subject", ""),
                    predicate=fact.get("predicate", ""),
                    object=fact.get("object", ""),
                    relation_type=fact.get("relation_type", "factual"),
                    confidence=float(fact.get("confidence", 1.0)),
                    speaker_id=speaker_id,
                    symmetric=bool(fact.get("symmetric")),
                )
            )
        except Exception as exc:
            validation_dropped.append(
                {
                    "subject": fact.get("subject", ""),
                    "predicate": fact.get("predicate", ""),
                    "object": fact.get("object", ""),
                    "relation_type": fact.get("relation_type", ""),
                    "reason": f"{type(exc).__name__}: {exc}"[:200],
                }
            )
            continue
    if validation_dropped:
        graph.diagnostics["pydantic_validation_dropped"] = validation_dropped
        logger.warning(
            "Dropped %d fact(s) at Relation schema validation "
            "(commonly: relation_type outside Literal set)",
            len(validation_dropped),
        )
    return kept


def recovery_gate(
    graph: SessionGraph,
    kept_relations: list[Relation],
    original_count: int,
    cause: str | None,
) -> bool:
    """Decide whether the all-dropped recovery net must fire, and record why.

    Fires when every relation was dropped somewhere in the pipeline AND the
    local extraction had produced facts to begin with. Failing that, there
    was no loss to recover from and nothing is recorded.

    The triggering ``cause`` (and its judgment/breakage classification) is
    written to ``graph.diagnostics["all_dropped_cause"]`` whenever the gate
    fires, so an operator can tell a judge's verdict from a mechanical
    breakage without re-deriving it from the surrounding drop counters.

    Args:
        graph: Session graph whose diagnostics receive the trigger record.
        kept_relations: Relations surviving schema validation.
        original_count: Relation count before the cloud pipeline ran.
        cause: The ``CAUSE_*`` site that emptied the working fact set, or
            ``None`` if no site claimed it.

    Returns:
        ``True`` when the caller must run the recovery path.
    """
    if kept_relations or original_count <= 0:
        return False
    kind = cause_kind(cause)
    graph.diagnostics["all_dropped_cause"] = {
        "cause": cause or CAUSE_UNATTRIBUTED,
        "kind": kind,
    }
    logger.warning(
        "All %d relation(s) dropped by pipeline (cause=%s, %s) — triggering all_dropped fallback",
        original_count,
        cause or CAUSE_UNATTRIBUTED,
        cause_kind(cause),
    )
    return True


def apply_rebuild(
    graph: SessionGraph,
    kept_relations: list[Relation],
    resolution: dict[str, str],
) -> None:
    """Install ``kept_relations`` on ``graph`` with a matching entity surface.

    Every relation endpoint must have a corresponding ``Entity`` record.
    Entity-type inference goes through
    :func:`~paramem.config.taxonomy.placeholder_entity_type` (open
    vocabulary): known prefixes (``Person``, ``Org``, ``City``, ...) use
    the configured closed mapping; novel prefixes the cloud introduces
    (``Project_1``, ``Paper_1``, ...) derive the type from the prefix
    itself — the prefix name IS the type name in the brace-binding
    protocol. ``entity_type`` is open (no ``Literal``), so the derived
    type passes through.

    ``name`` here is a de-anonymized real name (a ``kept_relations``
    subject/object), so it is looked up in the INVERTED resolution map
    (real name -> placeholder); ``resolution`` itself is keyed by
    placeholder and would never match. Inverting the SAME map the
    substitution used preserves CORE-LAST precedence with no new
    tie-break rule.

    Args:
        graph: Session graph mutated in place (``entities``, ``relations``).
        kept_relations: The relations to install.
        resolution: Placeholder -> real-name map used for the substitution
            that produced ``kept_relations``.
    """
    kept_names = {r.subject for r in kept_relations} | {r.object for r in kept_relations}
    existing_names = {e.name for e in graph.entities}
    graph.entities = [e for e in graph.entities if e.name in kept_names]
    name_to_placeholder = {real: token for token, real in resolution.items()}
    for name in kept_names - existing_names:
        entity_type = "concept"
        placeholder = name_to_placeholder.get(name)
        if placeholder:
            entity_type = placeholder_entity_type(placeholder)
        graph.entities.append(Entity(name=name, entity_type=entity_type))

    graph.relations = kept_relations
