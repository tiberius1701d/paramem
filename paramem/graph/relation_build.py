"""Pure post-processing from a de-anonymized fact list to a ``SessionGraph``.

Boundary / why this module exists
---------------------------------

Everything here is a total function of its arguments: facts in, relations
and entities out. There is no cloud call, no model, no tokenizer, no
config and no prompt — the most testable unit of the extraction flow, and
(before the split) the least tested one, because it lived buried in the
middle of ``_sota_pipeline`` where reaching it required standing up the
whole anonymize → enrich → judge chain first.

It owns four things:

* ``build_relations`` — schema validation of the surviving fact dicts,
  with the drops recorded rather than swallowed.
* ``apply_rebuild`` — the entity surface that must accompany a relation
  set (pruning, placeholder-derived typing of names the cloud minted)
  plus the scalar-attribute projection.
* ``partition_scalar_facts`` / ``project_scalar_facts_to_attributes``
  (and their ``is_scalar_value`` predicate) — the scalar routing. The
  PARTITION is invoked from inside the ``deanonymize`` stage's span (it
  must run BEFORE the deanon plausibility judge, whose drop rule targets
  URI-shaped tokens); the PROJECTION runs here, after the entity rebuild,
  so the subject entity survives pruning. Ownership and position are
  independent.
* ``recovery_gate`` — the all-dropped safety net's DECISION and its cause
  bookkeeping. The recovery ACTION (re-judging the pre-enrichment facts
  on the local model) stays with its caller: it needs the model and the
  tokenizer, which this module deliberately never sees.

The ``CAUSE_*`` vocabulary is defined here because the gate is its only
consumer; the flow stages that detect an emptied fact set import the
constants from here and record them on ``StageState.empty_cause``.
"""

from __future__ import annotations

import logging

from paramem.graph.placeholders import placeholder_entity_type
from paramem.graph.schema import Entity, Relation, SessionGraph

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Why the working fact set emptied.
#
# An empty relation set is a SYMPTOM that structurally different
# situations reach. These constants name the SITE at which the working
# fact set actually went empty, and ``EMPTY_CAUSE_KIND`` classifies each
# site into one of THREE kinds:
#
# * ``judgment``  — some judge decided the facts should not survive
#                   (the cloud enricher returned nothing, a plausibility
#                   judge dropped everything).
# * ``breakage``  — a mechanical step could not carry the facts through
#                   (placeholder substitution, ``Relation`` schema
#                   validation).  Facts were LOST.
# * ``routing``   — nothing was lost and nobody judged: the facts left
#                   the relation surface for another surface of the same
#                   graph.  Today the only site is the scalar partition,
#                   which moves verbatim-identifier facts onto
#                   ``Entity.attributes``; the attribute surface reaches
#                   indexed-key distillation just as relations do (see
#                   :func:`recovery_gate`).
#
# The kind is consulted in exactly ONE place: ``routing`` suppresses the
# all-dropped recovery net in :func:`recovery_gate`, because there is
# nothing to recover.  ``judgment`` and ``breakage`` are recorded only —
# neither the gate nor the empty-result terminal nor the fallback
# distinguishes them.
# ---------------------------------------------------------------------------

#: The cloud enricher returned an empty surviving fact set.
CAUSE_CLOUD_EMPTY = "cloud_returned_empty"
#: The anon-stage (cloud) plausibility judge dropped every fact.
CAUSE_ANON_JUDGE = "anon_judge_dropped_all"
#: Placeholder substitution dropped every fact (predicate-invariant
#: violations plus the residual-placeholder sweep).
CAUSE_DEANON_SUBSTITUTION = "deanon_substitution_dropped_all"
#: The deanon-stage (local) plausibility judge dropped every fact.
CAUSE_DEANON_JUDGE = "deanon_judge_dropped_all"
#: Every surviving fact failed ``Relation`` schema validation.
CAUSE_SCHEMA_VALIDATION = "schema_validation_dropped_all"
#: :func:`partition_scalar_facts` routed every surviving fact onto the
#: entity-attribute surface — a scalar-only session (all objects are
#: URLs, emails, phone numbers, DOIs, version-tagged tool names).
CAUSE_SCALAR_PARTITION = "scalar_partition_absorbed_all"

#: judgment = a judge decided; breakage = a mechanical step failed;
#: routing = the facts moved surface, nothing was lost.
EMPTY_CAUSE_KIND: dict[str, str] = {
    CAUSE_CLOUD_EMPTY: "judgment",
    CAUSE_ANON_JUDGE: "judgment",
    CAUSE_DEANON_SUBSTITUTION: "breakage",
    CAUSE_DEANON_JUDGE: "judgment",
    CAUSE_SCHEMA_VALIDATION: "breakage",
    CAUSE_SCALAR_PARTITION: "routing",
}

#: Recorded when the gate fires with no site having claimed the emptying.
CAUSE_UNATTRIBUTED = "unattributed"


def cause_kind(cause: str | None) -> str:
    """Classify an ``EMPTY_CAUSE`` value as judgment / breakage / routing.

    See the vocabulary comment above for the three-way distinction.

    Args:
        cause: One of the ``CAUSE_*`` constants, or ``None`` when no site
            claimed responsibility for the emptied fact set.

    Returns:
        ``"judgment"``, ``"breakage"``, ``"routing"``, or
        ``"unattributed"`` for ``None`` / any value outside the
        vocabulary.
    """
    if cause is None:
        return CAUSE_UNATTRIBUTED
    return EMPTY_CAUSE_KIND.get(cause, CAUSE_UNATTRIBUTED)


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

    Two conditions, in order:

    1. Every relation was dropped somewhere in the pipeline AND the local
       extraction had produced facts to begin with. Failing this, there
       was no loss to recover from and nothing is recorded.
    2. The emptying was not a pure ``routing`` cause. A scalar-only
       session (every surviving fact's object is a URL, email, phone
       number, DOI or version-tagged tool name) leaves the relation set
       empty ON PURPOSE: :func:`partition_scalar_facts` moved those facts
       onto ``Entity.attributes``, where
       :func:`~paramem.graph.relation_prep._flatten_entity_attributes`
       — called from
       ``ConsolidationLoop._entries_from_graph`` — projects them back
       into the relation-dict shape, so the distillation input is the
       union of real relations and projected attributes. Nothing was
       lost, so firing the net would be actively wrong: the recovery
       path returns before :func:`apply_rebuild`, which is the only
       caller of :func:`project_scalar_facts_to_attributes`, so the
       projection would never land AND the fallback would
       re-materialize the same facts as ``Relation`` objects — silently
       reversing the partition.

    The triggering ``cause`` (and its judgment/breakage/routing
    classification) is written to ``graph.diagnostics["all_dropped_cause"]``
    in BOTH outcomes, so an operator can tell a judge's verdict from a
    mechanical breakage from a deliberate surface change without
    re-deriving it from the surrounding drop counters.

    Args:
        graph: Session graph whose diagnostics receive the trigger record.
        kept_relations: Relations surviving schema validation.
        original_count: Relation count before the SOTA pipeline ran.
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
    if kind == "routing":
        logger.info(
            "All %d relation(s) left the relation surface (cause=%s, routing) — "
            "legitimate empty relation set, no fallback",
            original_count,
            cause,
        )
        return False
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
    scalar_facts: list[dict],
    resolution: dict[str, str],
) -> None:
    """Install ``kept_relations`` on ``graph`` with a matching entity surface.

    Every relation endpoint must have a corresponding ``Entity`` record.
    Entity-type inference goes through
    :func:`~paramem.graph.placeholders.placeholder_entity_type` (open
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

    Scalar-fact subjects are added to the survivor set before pruning so
    the projection below can attach to them.

    Args:
        graph: Session graph mutated in place (``entities``, ``relations``).
        kept_relations: The relations to install.
        scalar_facts: Facts routed off the relation surface by
            :func:`partition_scalar_facts`, projected onto subject
            attributes after the entity rebuild.
        resolution: Placeholder -> real-name map used for the substitution
            that produced ``kept_relations``.
    """
    kept_names = (
        {r.subject for r in kept_relations}
        | {r.object for r in kept_relations}
        | {str(f.get("subject", "")) for f in scalar_facts if str(f.get("subject", "")).strip()}
    )
    existing_names = {e.name for e in graph.entities}
    graph.entities = [e for e in graph.entities if e.name in kept_names]
    name_to_placeholder = {real: token for token, real in resolution.items()}
    for name in kept_names - existing_names:
        entity_type = "concept"
        placeholder = name_to_placeholder.get(name)
        if placeholder:
            entity_type = placeholder_entity_type(placeholder)
        graph.entities.append(Entity(name=name, entity_type=entity_type))

    project_scalar_facts_to_attributes(graph, scalar_facts)

    graph.relations = kept_relations


def is_scalar_value(value: str) -> bool:
    """True iff ``value`` is a verbatim identifier rather than a content phrase.

    Scalars belong on ``Entity.attributes`` (alongside email/phone/linkedin
    extracted by the local extractor), where
    :func:`~paramem.graph.relation_prep._flatten_entity_attributes` mints
    keyed pairs for indexed-key distillation without additional filtering.
    Routing them through plausibility as relations is wrong: a
    plausibility judge may reject values whose alpha portion lives
    concatenated with digits in the source transcript
    (``username1234``, ``ROS2``, ``H100``), even though the scalar is
    verbatim in the text.

    Detection is structural — no regex.  Recognised shapes:

    * **Phone**: ≥7 digits, characters drawn only from
      ``digits + " +()-."``.  Spaces are permitted only inside this class
      (phone numbers are the one multi-token scalar).
    * **Email**: contains ``@`` and a ``.`` after the ``@``, no whitespace.
    * **URL with scheme**: starts with ``http://`` / ``https://`` /
      ``ftp://``.
    * **URL without scheme**: contains ``/`` and the part before the
      first ``/`` looks like a domain (alphanumeric + dots/dashes,
      contains at least one ``.``).
    * **DOI**: starts with ``10.`` and contains ``/``.
    * **Version-tagged identifier**: contains both letters and digits,
      no spaces, characters drawn only from ``alnum + "/-_+."``
      (e.g. ``ROS2``, ``H100``, ``IPv6``, ``ROS2/Gazebo``, ``x86_64``).
    """
    s = (value or "").strip()
    if not s:
        return False
    digit_count = sum(c.isdigit() for c in s)
    # Phone: ≥7 digits, only phone-shaped chars (the only multi-token scalar).
    if digit_count >= 7 and all(c.isdigit() or c in "+()-. " for c in s):
        return True
    # After the phone exemption, multi-word means content phrase.
    if any(c.isspace() for c in s):
        return False
    # Email: "@" with a domain ending in a TLD-like dot suffix.
    if "@" in s:
        domain = s.rsplit("@", 1)[-1]
        if "." in domain and len(domain) > 2:
            return True
    lower = s.lower()
    # URL with scheme.
    if lower.startswith(("http://", "https://", "ftp://")):
        return True
    # URL without scheme: domain.tld[/path].
    if "/" in s:
        head = s.split("/", 1)[0]
        if "." in head and head.replace(".", "").replace("-", "").isalnum():
            return True
    # DOI.
    if s.startswith("10.") and "/" in s and digit_count > 0:
        return True
    # Version-tagged identifier.
    if digit_count > 0 and any(c.isalpha() for c in s):
        if all(c.isalnum() or c in "/-_+." for c in s):
            return True
    return False


def partition_scalar_facts(
    facts: list[dict], entities: list[Entity] | None = None
) -> tuple[list[dict], list[dict]]:
    """Split facts by object-shape into ``(scalar, non_scalar)``.

    Scalars are projected onto ``Entity.attributes`` of the subject and
    flow through to plausibility and downstream filters without modification;
    non-scalars are treated as concept-level claims.  Facts already marked
    ``synthetic=True`` are passed through untouched.

    ``entities`` is the graph's current entity list (local-extraction state,
    read before this pipeline's own entity rebuild).  A fact's object is
    forced to ``non_scalar`` whenever it already names an existing entity
    node, regardless of what :func:`is_scalar_value` would say — a
    digit-bearing entity name (``speaker0``, ``speaker1``, a droid-style
    person name like ``R2D2``) otherwise mis-triggers the version-tagged-
    identifier heuristic and gets flattened into a string attribute instead
    of staying a first-class edge. There is no dedicated "literal" entity
    type in the schema (entity_type is open-vocabulary: person, place,
    organization, event, preference, concept, or model-invented types), so
    any existing node — not just ``person`` — qualifies: object-is-a-known-
    entity-node is the gate; object shape is only consulted when no node
    exists.
    """
    entity_names = {e.name for e in (entities or [])}
    scalar: list[dict] = []
    non_scalar: list[dict] = []
    for f in facts:
        if not isinstance(f, dict):
            non_scalar.append(f)
            continue
        if f.get("synthetic") is True:
            non_scalar.append(f)
            continue
        obj = str(f.get("object", ""))
        if obj in entity_names:
            non_scalar.append(f)
            continue
        if is_scalar_value(obj):
            scalar.append(f)
        else:
            non_scalar.append(f)
    return scalar, non_scalar


def project_scalar_facts_to_attributes(graph: SessionGraph, scalar_facts: list[dict]) -> None:
    """Fold scalar-object facts onto ``Entity.attributes`` of the subject.

    Mutates ``graph.entities`` in place: ensures every scalar-fact subject
    has an ``Entity`` record (creating one with type ``"concept"`` when
    missing), then sets ``entity.attributes[<key>] = object``.  ``<key>``
    is the relation predicate with a leading ``has_`` stripped so
    :func:`~paramem.graph.relation_prep._flatten_entity_attributes` (which
    re-prefixes attribute keys with ``has_`` when minting projected
    relations for indexed-key distillation) does not produce
    ``has_has_<key>``.
    """
    if not scalar_facts:
        return
    name_to_entity = {e.name: e for e in graph.entities}
    for f in scalar_facts:
        subj = str(f.get("subject", "")).strip()
        pred = str(f.get("predicate", "")).strip()
        obj = str(f.get("object", "")).strip()
        if not (subj and pred and obj):
            continue
        ent = name_to_entity.get(subj)
        if ent is None:
            ent = Entity(name=subj, entity_type="concept")
            graph.entities.append(ent)
            name_to_entity[subj] = ent
        attr_key = pred.removeprefix("has_") if pred.startswith("has_") else pred
        ent.attributes[attr_key] = obj
