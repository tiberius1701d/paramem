"""Vocabulary for "why did the working fact set go empty?" — FLOW STATE,
not relation-rebuilding.

This module used to live inside ``paramem.graph.relation_build`` (the
gate's own home), on the reasoning that the gate was its only consumer.
That stopped being true once the extraction flow was carved into
declarative stages: the vocabulary now has THREE consumers —
:func:`~paramem.graph.relation_build.recovery_gate` (which still owns the
DECISION of whether the all-dropped recovery net must fire),
the flow tail stages (``deanonymize``/``rebuild`` in ``paramem.graph.flows``,
which detect and record which site emptied ``StageState.facts``), and the
``enrich`` stage (``paramem.graph.stage_enrich``, where the cloud enricher or the
anon-stage judge can empty the fact set). A vocabulary three call sites
share is no longer "the gate's own" — it is flow state, and belongs in its
own module.

It must NOT live in ``paramem.graph.flow`` — that module is generic
topology (``StageContext``/``StageState``/``StageSpec``/``run_flow``) and
must not learn extraction semantics; the ``CAUSE_*`` vocabulary IS
extraction semantics (it names actual pipeline sites: the cloud enricher,
the anon-stage judge, placeholder substitution, ...).

``paramem.graph.relation_build`` imports the constants and :func:`cause_kind`
back from here for :func:`~paramem.graph.relation_build.recovery_gate`'s own
use — that module keeps the DECISION, this module keeps the VOCABULARY.
"""

from __future__ import annotations

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
#                   :func:`~paramem.graph.relation_build.recovery_gate`).
#
# The kind is consulted in exactly ONE place:
# :func:`~paramem.graph.relation_build.recovery_gate` — ``routing``
# suppresses the all-dropped recovery net, because there is nothing to
# recover.  ``judgment`` and ``breakage`` are recorded only — neither the
# gate nor the empty-result terminal nor the fallback distinguishes them.
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
#: :func:`~paramem.graph.relation_build.partition_scalar_facts` routed
#: every surviving fact onto the entity-attribute surface — a
#: scalar-only session (all objects are URLs, emails, phone numbers,
#: DOIs, version-tagged tool names).
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
