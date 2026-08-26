"""Consolidation loop orchestrator.

Runs the full consolidation pipeline: extract graph from session,
merge into cumulative graph, score for promotion, train
episodic and semantic adapters.
"""

import hashlib
import logging
import secrets
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal, Optional, Sequence

from torch.utils.data import Dataset

from paramem.cloud.admission import evaluate_cloud_egress
from paramem.config.taxonomy import ScrubCategory, fallback_relation_type, relation_types
from paramem.graph.extraction_pipeline import ExtractionConfig, ExtractionPipeline
from paramem.graph.extractor import ExtractionFailed, local_parse_failure
from paramem.graph.merger import GraphMerger, attribute_fact, min_nonempty, node_display
from paramem.graph.phase_trace import ExtractionTrace, extraction_trace, phase_trace
from paramem.graph.relation_prep import partition_relations
from paramem.graph.schema import Relation, SessionGraph
from paramem.memory.bookkeeping import bookkeeping_row, credit_reinforcement
from paramem.memory.entry import (
    assign_keys,
    content_only_entry,
    entry_simhash,
    format_entry_training,
)
from paramem.models.loader import has_prior_trained_weights
from paramem.training import graph_tier
from paramem.training.donor import DONOR_BUILD_ADAPTER_NAME, DONOR_KEY_FLOOR
from paramem.training.key_registry import KeyRegistry
from paramem.training.recall_eval import RecallProbe
from paramem.training.thermal_throttle import ThermalPolicy
from paramem.training.trainer import (
    STAGING_ADAPTER,
    TrainingHooks,
    staged_weights,
)
from paramem.utils.artifacts import (
    debug_run,
    on_cycle_end,
    on_extraction_end,
    on_fold_assignments,
    on_fold_graph,
    on_recall_probe,
    on_removal_ledger,
    on_session_extracted,
)
from paramem.utils.config import (
    AdapterConfig,
    ConsolidationConfig,
    GraphConfig,
    TrainingConfig,
    WandbConfig,
    budget_for,
)
from paramem.utils.identity import canonical
from paramem.utils.vram_guard import safe_empty_cache

if TYPE_CHECKING:
    from typing import Final

    from paramem.adapters.registry_binding import TierBinding
    from paramem.memory.increment import TierIncrement, TierWriteContext
    from paramem.training.early_stop import _EarlyStopState
    from paramem.training.stage_ledger import StageLedger

logger = logging.getLogger(__name__)

# Frozen set of valid relation types drawn from the single source of truth in
# paramem.config.taxonomy so that the stage-2 clamp stays in sync with the
# Pydantic Relation schema (_RelationType = Literal[relation_types()]).
_VALID_RTYPES: frozenset[str] = frozenset(relation_types())
_FALLBACK_RTYPE: str = fallback_relation_type()


def _relation_to_entry_dict(r: "Relation") -> dict:
    """Project a single ``Relation`` into the ``{subject, predicate, object,
    relation_type}`` shape used to seed interim-tier entry dicts.

    ``predicate`` is canonicalized via :func:`~paramem.utils.identity.canonical`
    so interim-tier entries match the identity form the merger stamps onto the
    cumulative edge (``merger.py:710``) — without this, an interim entry built
    straight from ``session_graph.relations``/``proc_graph.relations`` carries
    the raw extraction surface (e.g. ``"Works_At"``) while the full-cycle edge
    entry (:meth:`ConsolidationLoop._build_working_keyed_walk`) carries the
    canonical form (``"works at"``), desyncing the SimHash fingerprint below
    :data:`~paramem.memory.entry.DEFAULT_CONFIDENCE_THRESHOLD`.  ``subject`` and
    ``object`` are left as-is — display surfaces, not identity keys.  ``r``
    itself (in particular ``r.predicate``) is never mutated: the raw surface
    remains load-bearing provenance for the merger's ``removal_ledger``
    (``merger.py:815``).

    Args:
        r: A single extracted or reconstructed ``Relation``.

    Returns:
        Dict with ``subject``, ``predicate`` (canonical), ``object``, and
        ``relation_type``.
    """
    return {
        "subject": r.subject,
        "predicate": canonical(r.predicate),
        "object": r.object,
        "relation_type": r.relation_type,
    }


def _recall_bind_telemetry(
    recall_state: "_EarlyStopState | None", n_keys: int, accum: int
) -> "tuple[int | None, int | None, bool | None]":
    """Derive ``(epochs_to_bind, steps_to_bind, hit_cap)`` for fold telemetry.

    ``epochs_to_bind`` is the epoch at which the recall early-stop signal
    fired — ``recall_state.stop_epoch``, set at the SAME epoch as
    ``stable_perfect_epoch`` for the window-based stop path (the only path
    the production policy uses; see
    ``paramem.training.early_stop.RecallEarlyStopCallback.on_epoch_end``,
    the block setting ``state.stable_perfect_epoch`` immediately precedes
    the block setting ``state.stop_epoch`` in the same call). ``steps_to_bind``
    converts that epoch count to optimizer steps at the project's fixed
    ``batch=1``: ``ceil(n_keys / accum) * epochs_to_bind``. ``hit_cap`` is
    True when training ran to the full derived epoch budget WITHOUT the
    early-stop signal ever firing (``stop_epoch is None``) — i.e. the
    recall gate never bound within the budget.

    **Left-censored by the probe schedule — the field's consumer (the
    budget-bucket re-fit) MUST account for this.** The probe cadence is
    ``probe_from_epoch=signal_from_epoch=recall_signal_from_epoch`` (default
    20, ``paramem.server.config.py:1110``) every
    ``recall_probe_every_n_epochs`` epochs, and the window-based stop needs
    ``recall_window`` consecutive perfect probes
    (``paramem.training.early_stop.RecallEarlyStopCallback.on_epoch_end``).
    The earliest attainable ``stop_epoch`` is therefore
    ``floor + probe_every_n_epochs * (window - 1)``, NOT epoch 1 — under the
    shipped config (``configs/server.yaml``: floor=20, every_n_epochs=3,
    window=2) that floor is epoch 23. A fold whose weights would have bound
    at, say, epoch 4 still records ``epochs_to_bind=23`` because no probe
    ran before then; ``steps_to_bind`` inherits the same left-censoring
    linearly. Treat recorded values near this floor as "at least this fast",
    not as the true convergence point.

    **Does not see abort state.** This function has no visibility into the
    trainer's own abort signal (``_tier_metrics``/``epi_metrics``'s
    ``"aborted"`` key) — an abort signalled by an inference request needing
    the GPU during background training, via
    ``_abort_background_training_for_inference``
    (``BackgroundTrainer.abort_for_inference``), or by server shutdown
    (``shutdown_requested``), also leaves ``stop_epoch=None`` and is
    indistinguishable from a genuine "ran to the full budget without
    binding" from this function's inputs alone. Callers MUST additionally
    check the trainer's abort flag and suppress the ``hit_cap`` field on the
    abort path — see ``ConsolidationLoop._train_gate_write``'s own check of
    ``metrics.get("aborted", False)`` for exactly this reason.

    **Crash-resume warm-start vs. measured init="cold".** On an interim
    crash-resume, a missing interim slot is recreated LoRA-zero
    (``consolidation.py:3142``) even though the resumed training call then
    reloads the checkpoint's actual (possibly far-from-zero) weights via HF
    Trainer's ``resume_from_checkpoint`` — so the ``init`` field measured at
    fold entry (before that reload) can read ``"cold"`` on a path whose
    ACTUAL training start is warm. This function doesn't touch ``init``, but
    the caveat lives here because both fields are read together at the same
    call sites during the bucket re-fit.

    Args:
        recall_state: The ``_EarlyStopState`` returned by
            ``ConsolidationLoop._train_tier_adapter``, or ``None`` when
            early stopping is disabled or the entries list was empty.
        n_keys: Number of keyed entries trained (``len(entries)``).
        accum: Derived gradient-accumulation steps for this fold
            (``paramem.utils.config.budget_for``'s second return value).

    Returns:
        ``(epochs_to_bind, steps_to_bind, hit_cap)``, each ``None`` when
        ``recall_state`` is ``None`` — these fields are only meaningful on
        the training success path; the ring omits absent fields.
    """
    if recall_state is None:
        return None, None, None
    stop_epoch = recall_state.stop_epoch
    if stop_epoch is None:
        return None, None, True
    steps_per_epoch = -(-n_keys // accum)  # ceil division at batch=1
    return stop_epoch, steps_per_epoch * stop_epoch, False


class RecallGateRejected(RuntimeError):
    """Raised when a recall verdict falls short of the required bar.

    Two raise sites: :meth:`ConsolidationLoop._assert_tier_recall` (a tier's
    own training-completeness verdict below 100% over its full key set,
    probed on the staged weights immediately before the write — used by both
    the full event and the interim event), and
    :func:`~paramem.server.active_store_migration._migrate_tier_simulate_to_train`
    (the migration path's own probe of its staged weights).

    A deterministic quality verdict, NOT a crash: the adapter trained
    successfully, and the probe simply did not reach the threshold.  Raised
    INSIDE the tier's ``staged_weights`` scope, before the staging slot is
    ever saved into the tier's live-facing slot — the tier's on-disk state is
    therefore untouched by the refusal, and ``tier_backup_scope``'s
    ``except BaseException`` restore arm covers only the VRAM state of the
    one tier currently training (never a whole event's worth of tiers — no
    tier's weights are activated live until the whole bundle writes).
    :meth:`ConsolidationLoop._write_built_tier` catches this at the write site,
    disposes the event's ledger and extraction tree, and re-raises unchanged
    onto the same loud-failure path: an incident is recorded and every
    contributing session stays pending; the next attempt re-extracts from
    scratch.  Holding sessions pending for a non-encoding outcome is
    structural, not a mechanism — a session is retired only at successful
    disposal, so a raise here simply never reaches the retirement call.

    Subclasses ``RuntimeError`` so existing broad handlers keep their
    current behaviour.
    """

    def __init__(
        self,
        message: str,
        *,
        adapter_name: str,
        recall_rate: float,
        threshold: float,
        failed_keys: tuple[str, ...] = (),
    ):
        super().__init__(message)
        self.adapter_name = adapter_name
        self.recall_rate = recall_rate
        self.threshold = threshold
        self.failed_keys = failed_keys


class ActiveKeyHydrationFailure(RuntimeError):
    """Raised by :meth:`ConsolidationLoop._hydrate_store_for_fold` when a
    registered active key has no content in the store cache and none in the
    fold's venue (adapter weights or ``graph.json``).

    The registry is the durable record of a key's existence; a fold must
    never retire a registered active key because a READ of its content
    failed.  A failed source read is evidence about the read path, not
    about the key, so it is never treated as proof the key's fact no
    longer exists.  Deliberate retirement paths — dedup staling
    (``store.discard_keys()`` on a registry-true duplicate),
    the removal ledger's acting-site fate decisions
    (:meth:`_apply_working_fate_decisions`), and
    the explicit operator doors ``/debug/erase-keys`` and
    ``/speaker/forget`` — are unaffected; this exception guards only the
    unreadable case.

    Fired BEFORE any registry mutation or durable write for this fold, so
    the prior (pre-fold) weights, registries, and interim slots stay
    intact and the next cycle simply retries the hydration.

    Carries the dropped key list and the venue that failed to produce them
    so the caller's incident record names exactly what could not be
    hydrated.
    """

    def __init__(self, *, dropped_keys: "list[str]", venue: str):
        dropped_keys = sorted(dropped_keys)
        message = (
            f"{len(dropped_keys)} active key(s) could not be hydrated from the "
            f"{venue} source; aborting fold before any registry mutation or "
            f"durable write: {dropped_keys[:10]}"
        )
        super().__init__(message)
        self.dropped_keys = dropped_keys
        self.venue = venue


class ConsolidationResumeBlocked(RuntimeError):
    """Raised by the build/write driver's resume routine when a not-yet-done
    tier's live registry belongs to neither this event's pre-state nor its
    own shadow payload (:data:`FOREIGN`, see :func:`classify_partial_build`).

    Something outside this event wrote the tier while its ledger was
    pending: the resume refuses to rebuild over a stranger's write and
    holds. The ledger and every shadow artifact are left untouched — no
    phase runs — so every later dispatch re-enters this same event and
    meets the same refusal until the pending record is disposed. Recording
    an incident and clearing the record (``POST /reconsolidate``) is the
    caller's job, not this exception's.
    """

    def __init__(self, *, tier: str, reason: str):
        super().__init__(
            f"consolidation resume blocked: tier {tier!r} classified {reason!r} — "
            "something outside this event wrote its live registry; refusing to "
            "rebuild over it"
        )
        self.tier = tier
        self.reason = reason


class ConsolidationArtifactsMissing(RuntimeError):
    """Raised by :meth:`ConsolidationLoop.run_build_and_publish` when the
    ledger's extraction entry no longer verifies against on-disk bytes and
    at least one tier this event names is not already live.

    The event's staged content — the shadow tree phase 1 wrote — is gone
    or corrupted, so nothing in it can be trusted to build an increment
    from. The pending record (ledger, extraction tree, every scratch dir)
    is disposed BEFORE this raises: the contributing transcripts were
    never retired, so the next dispatch re-extracts them fresh rather than
    this call silently reading the gap as a rows-only tier (an empty
    ``keyed.json`` would read identically to a legitimate rows-only
    member — the whole point of verifying first).
    """

    def __init__(self, *, event: str, missing: "list[str]"):
        missing = sorted(missing)
        super().__init__(
            f"consolidation artifacts missing for the pending {event!r} event — "
            f"the extraction entry no longer verifies; disposed the pending "
            f"record so the next dispatch re-extracts. Unverified path(s): {missing[:10]}"
        )
        self.event = event
        self.missing = missing


# ============================================================================
# Event staging — extract, merge, enrich, assign, assert.
# Built beside the existing fold spine above: the live MemoryStore is read
# only at the recall boundary (_recall_working_tiers), which seeds an
# independent WorkingTier copy per tier in the event's working universe.
# Every read and mutation the staging pass performs after that goes to
# those copies, never back to the live store.  Staging an event persists
# the result as a per-event shadow artifact tree plus a stage-ledger
# extraction entry, consumed by the build/write/go-live driver further below
# (ConsolidationLoop.run_build_and_publish).
# ============================================================================

#: The one reason string that changes the resume action for a not-yet-done
#: tier (see classify_partial_build): every other outcome means the tier is
#: ours to rebuild.
FOREIGN: "Final[str]" = "foreign"


@dataclass
class WorkingTier:
    """One tier's recalled working copy — the seed state the staging pass mutates.

    The registry, the bookkeeping rows and the entry content are independent
    clones of that tier's persisted live state at the moment of recall.
    Every read and every mutation the staging pass performs afterwards goes to
    these copies; the live ``MemoryStore`` is never touched.  ``dirty`` is
    set by every mutating step (reinforcement credit, promotion, a fate
    decision) that touches this tier and is what lets a dedup-only/candidate
    tier be told apart from one this event never actually changed.

    ``rebuilt`` is set once at recall (:meth:`ConsolidationLoop._recall_working_tiers`)
    to whether THIS event re-derives the tier's content — true for a primary
    tier, false for a dedup-only/candidate tier.  It is what seeds the
    registry active-only for a rebuilt tier (a marker ends at its own tier's
    rebuild) and what :meth:`ConsolidationLoop._apply_working_fate_decisions`
    reads to decide whether a retired key here is dropped outright or
    withheld behind a marker.

    ``scratch_dir`` is this tier's resolved HF ``TrainingArguments`` working
    directory (:meth:`ConsolidationLoop._training_output_dir`), fixed once
    at recall time and carried into the ledger's ``tiers[tier]["scratch"]``
    field by :meth:`~ConsolidationLoop.stage_event` -- the trainer call and
    every disposer read it back from there rather than recomputing it, so a
    resumed event in a fresh process (whose live ``cycle_count`` may have
    drifted from the crashed pass) trains into and disposes the SAME
    directory.
    """

    tier: str
    adapter_name: str
    pre_sha: str
    scratch_dir: Path
    registry: "KeyRegistry"
    rows: "dict[str, dict]"
    entries: "dict[str, dict]"
    rebuilt: bool
    dirty: bool = False

    def adopt_key_from(self, source: "WorkingTier", key: str) -> "dict":
        """Move *key* out of *source*'s working copy into this one.

        The staging-layer half of the one carry rule for a key changing
        tier (:meth:`KeyRegistry.adopt_key_from` is the registry-layer
        half): delegates the registry move (active standing and
        fingerprint), moves the entry content and the bookkeeping row, and
        marks both tiers dirty.  Active-onlyness is a property of THIS
        layer, not the registry layer below it: a rebuilt tier's working
        copy is seeded ``active_only=True`` (see
        :meth:`KeyRegistry.working_copy`), so ``source`` never holds a
        withheld key to begin with — the registry-layer move has nothing
        to refuse. ``_promote_working_keys`` and ``_route_absorbed_keyed_fact``
        are the two callers — a promotion (episodic -> semantic) and an
        absorbed candidate tier's key routed into a primary tier before the
        candidate is reaped whole.

        Every known key carries a bookkeeping row, so *source* having none
        for *key* is a violation of that invariant, checked and raised
        BEFORE either registry is mutated (leaving both working copies
        untouched on failure) rather than routed around.  Fold-local
        hydration likewise guarantees a working entry for every active
        key, so *source* having no entry for *key* is the same class of
        violation, checked and raised BEFORE either registry is mutated as
        well.

        Returns the moved row.

        Raises:
            ~paramem.memory.store.BookkeepingInvariantViolation: *source*
                has no bookkeeping row for *key*, or *source* has no
                working entry for *key*.
        """
        if key not in source.rows:
            from paramem.memory.store import raise_bookkeeping_invariant_violation

            raise_bookkeeping_invariant_violation(source.tier, [key], "working tier key adoption")
        if key not in source.entries:
            from paramem.memory.store import raise_bookkeeping_invariant_violation

            raise_bookkeeping_invariant_violation(
                source.tier, [key], "working tier key adoption: active key has no working entry"
            )
        self.registry.adopt_key_from(source.registry, key)
        self.entries[key] = source.entries.pop(key)
        row = source.rows.pop(key)
        self.rows[key] = row
        self.dirty = True
        source.dirty = True
        return row


@dataclass(frozen=True)
class StagedEvent:
    """What staging one event produced — the handle a driver consumes.

    ``ledger`` is the in-memory mirror of what :func:`stage_event` just
    wrote to ``stage_ledger.json`` (one ``"extraction"`` stage entry).  A
    caller resolves each built tier's :class:`~paramem.memory.increment.TierIncrement`
    via ``build_tier_increment(tier=tier, adapter_name=ledger.tiers[tier]["adapter"],
    pre_sha=ledger.tiers[tier]["pre_sha"], shadow_dir=extraction_dir(state_dir,
    event) / "shadow" / tier)``.
    """

    event: str
    venue: str
    state_dir: Path
    built_tiers: "tuple[str, ...]"
    ledger: "StageLedger"


def classify_partial_build(*, increment: "TierIncrement", output_dir: Path) -> str:
    """Say why a NOT-DONE tier's on-disk state is what it is.  Resume repair only.

    Two comparisons over one subject — the tier's live registry's plaintext
    bytes — with no venue branch and no binding call: the live digest against
    the digest the staging pass recorded (``increment.pre_sha``), then against the
    digest of the payload the increment itself carries
    (``sha256(increment.registry_bytes)``).  A match against either means the
    tier is ours to rebuild (``"not_built"`` / ``"torn_own_write"`` — the
    caller's remedy is identical either way); a match against neither means
    something outside this event wrote the tier (:data:`FOREIGN`), and the
    caller refuses and holds rather than rebuilding over a stranger's write.

    Never answers "done" — doneness is the stage ledger's verified
    ``tier_live`` entry, and this function is consulted only for a tier that
    is not done.

    Args:
        increment: The tier's assembled increment (``build_tier_increment``'s
            output) — supplies ``pre_sha`` and ``registry_bytes``.
        output_dir: The adapter tree root the tier's live slot resolves
            under (``loop.output_dir`` / ``ctx.output_dir``).

    Returns:
        ``"not_built"``, ``"torn_own_write"``, or :data:`FOREIGN`.
    """
    from paramem.adapters.manifest import tier_registry_sha256
    from paramem.memory.interim_adapter import adapter_slot_root_for_name

    tier_root = adapter_slot_root_for_name(output_dir, increment.adapter_name)
    live_digest = tier_registry_sha256(tier_root)
    if live_digest == increment.pre_sha:
        return "not_built"
    if live_digest == hashlib.sha256(increment.registry_bytes).hexdigest():
        return "torn_own_write"
    return FOREIGN


def interim_outcome_label(build_summary: dict, *, venue: str) -> str:
    """Derive an interim event's outcome label from its build summary.

    The one derivation, shared by the fresh interim cycle
    (:meth:`ConsolidationLoop.run_consolidation_cycle`) and the resumed one
    (:func:`~paramem.server.app._finish_resumed_event`) — both read the same
    :meth:`~ConsolidationLoop.run_build_and_publish` summary shape and must
    agree on what it means.

    Args:
        build_summary: The dict :meth:`ConsolidationLoop.run_build_and_publish`
            returns — ``{"published_tiers", "skipped_live_tiers", "all_live",
            "aborted"}``.
        venue: This event's venue (``"weights"`` or ``"disk"``) — the same
            axis :func:`stage_event`'s own ``venue`` parameter and
            :attr:`StagedEvent.venue` carry.

    Returns:
        ``"aborted"`` when the trainer yielded mid-bundle (checked first, so
        an aborted event is never mistaken for a genuine no-op); otherwise
        ``"noop"`` when nothing went live; otherwise ``"trained"`` (venue
        ``"weights"``) or ``"simulated"`` (venue ``"disk"``).
    """
    if build_summary["aborted"]:
        return "aborted"
    if not build_summary["all_live"]:
        return "noop"
    return "trained" if venue == "weights" else "simulated"


@dataclass(frozen=True)
class PendingRelations:
    """One batch's merged extraction product, captured at the extraction
    boundary and consumed by a fold.

    ``episodic`` / ``procedural`` are the split :class:`~paramem.graph.schema.Relation`
    lists ``stage_event`` receives through its own pending-relations
    channel.  Empty lists are a valid, meaningful value: an extraction that
    merged nothing.
    """

    episodic: "list[Relation]"
    procedural: "list[Relation]"

    def is_empty(self) -> bool:
        """``True`` when neither list carries a relation."""
        return not self.episodic and not self.procedural


@dataclass(frozen=True)
class FoldScope:
    """Immutable descriptor that parameterizes one consolidation event's
    stage-then-build-and-publish pass (:meth:`ConsolidationLoop.stage_event`
    / :meth:`ConsolidationLoop.run_build_and_publish`).

    A frozen dataclass (not a mode string) so dispatch is structural — no
    ``mode == "simulate"`` / ``mode == "train"`` literals inside the fold's
    own methods (the mode-fork-guard enforces this).

    Attributes:
        source: **The venue discriminator.**  ``"weights"`` is the train venue:
            adapter weights exist, so the fold reconstructs from them, trains,
            and saves them.  ``"disk"`` is the simulate venue: no weights exist,
            so the weight-only blocks are skipped and the persist medium is
            per-tier ``graph.json``.  Both venues read the same
            :class:`~paramem.memory.store.MemoryStore` for their fold input —
            ``source`` selects the weight *probe*, never the input medium.
        persist: Persist venue, dispatched at the end of the spine — every
            built tier's increment writes via
            :func:`~paramem.memory.persistence.write_tier_slot` and goes live
            via :func:`~paramem.memory.persistence.publish_tier_registry`
            (:func:`~paramem.training.go_live.publish_bundle`'s one publish
            act), never :func:`~paramem.memory.persistence.commit_tier_slot`
            (that helper backs the unrelated ``commit_main_tiers`` /
            active-store-migration paths).

            - ``"interim_slot"`` — the interim cycle's own target slot.
            - ``"main_tiers"`` — full fold.  Writes adapter weights when
              ``source == "weights"``, a per-tier ``graph.json`` otherwise.
        normalize: When ``True``, run the whole-graph normalization pass —
            forwarded to :meth:`~ConsolidationLoop.stage_event` as its own
            ``normalize`` argument, which passes it straight to
            :meth:`~paramem.training.graph_tier.GraphTierRefiner.refine`.
            Pinned ``False`` for the interim scope, structurally, like
            ``enrich``.
        enrich: When ``True``, run cloud graph enrichment — forwarded to
            :meth:`~ConsolidationLoop.stage_event` the same way as
            ``normalize``.  Pinned ``False`` for the interim scope
            regardless of ``refinement_enrichment`` / ``cloud_enabled`` —
            graph-tier enrichment is a full-fold-only pass.  Session-tier
            cloud enrichment already runs at extraction time over the
            anonymized transcript (:mod:`paramem.graph.stage_enrich`), which
            has strictly better context than a graph-only pass would at
            interim scope; cross-session inference over the cumulative
            graph remains the full fold's job.

    Whether an event folds pending-session relations is not a field of
    this class at all: it is the presence of a :class:`PendingRelations`
    argument at the call site (:meth:`~ConsolidationLoop.run_consolidation_cycle` /
    :meth:`~ConsolidationLoop.consolidate`), so a flag and a value can
    never disagree.
    """

    # --- identity / dispatch ---
    source: "Literal['weights', 'disk']"
    persist: "Literal['interim_slot', 'main_tiers']"

    # --- refine gate ---
    normalize: bool = False
    enrich: bool = False


def enrichment_signal(loop: "ConsolidationLoop", session_id: str) -> dict:
    """One session's enrichment-health record, read off ``loop.last_session_graph``
    right after :meth:`ConsolidationLoop.extract_session` returns for it.

    The single producer of this record's shape (``{session_id, anonymize,
    cloud_enrichment_degraded}``): every staging caller of ``extract_session``
    (``_run_extraction_phase``, ``_extract_pending_sessions`` — both in
    ``paramem.server.app``) calls this once per session instead of each
    re-reading ``loop.last_session_graph.diagnostics`` inline.  Not folded
    into ``extract_session``'s own return shape — that 2-tuple
    (``episodic_rels``, ``procedural_rels``) is destructured by six
    non-server callers (``experiments/dataset_probe.py``,
    ``experiments/lme_graph_builder.py``, four
    ``scripts/dev/probe_*_live.py`` sites) that would all need updating for
    a shape no calibration or non-staging caller needs — see
    ``extract_session``'s own docstring for why writing the incident here
    would be wrong (a non-staging run mutating production incident state);
    this function only reads, it arbitrates nothing.  A batch's collected
    records are passed to :meth:`ConsolidationLoop.arbitrate_enrichment_incidents`
    by the staging caller once extraction finishes.
    """
    session_graph = loop.last_session_graph
    return {
        "session_id": session_id,
        "anonymize": (
            session_graph.diagnostics.get("anonymize") if session_graph is not None else None
        ),
        "cloud_enrichment_degraded": (
            session_graph.diagnostics.get("cloud_enrichment_degraded")
            if session_graph is not None
            else None
        ),
    }


class ConsolidationLoop:
    """Manages the full consolidation pipeline across sessions.

    Each cycle:
    1. Extract knowledge graph from session transcript
    2. Merge into cumulative graph
    3. Score nodes for promotion
    4. Generate QA training pairs from graph
    5. Train episodic adapter (new + replay)
    6. Train semantic adapter (promoted + replay)
    """

    # Class-level default so instances built via ``object.__new__`` (test
    # harnesses that skip ``__init__`` to avoid loading a model) still
    # resolve this attribute.
    _telemetry_dir: "Path | None" = None

    # Same reason: ``cloud_enabled`` must resolve on instances built via
    # ``object.__new__``.  Default OFF — a harness that skips ``__init__``
    # gets no cloud egress unless it says so.
    cloud_enabled: bool = False

    # Same reason.  ``None`` means donor stores live beside this loop's own
    # tier stores; ``borrow_donor_cache`` sets an instance attribute pointing
    # at another root's cache, read-only (see that method).
    _borrowed_donor_root: "Path | None" = None

    def __init__(
        self,
        model,
        tokenizer,
        consolidation_config: ConsolidationConfig,
        training_config: TrainingConfig,
        *,
        memory_store,
        tier_adapters: Mapping[str, AdapterConfig],
        wandb_config: Optional[WandbConfig] = None,
        output_dir: str | Path = "outputs/phase3",
        extraction_temperature: float = 0.0,
        extraction_max_tokens: int,
        extraction_plausibility_max_tokens: int,
        extraction_anonymize_token_envelope: int,
        save_cycle_snapshots: bool = True,
        snapshot_dir: str | Path | None = None,
        run_id: str | None = None,
        prompts_dir: str | Path | None = None,
        model_name: str | None = None,
        extraction_enrichment_provider: str = "",
        extraction_enrichment_provider_model: str = "claude-sonnet-4-6",
        extraction_enrichment_provider_endpoint: str | None = None,
        extraction_plausibility_judge: str = "auto",
        extraction_plausibility_stage: str = "deanon",
        extraction_plausibility_model: str = "claude-sonnet-4-6",
        extraction_plausibility_endpoint: str | None = None,
        extraction_scrub_categories: tuple[ScrubCategory, ...],
        extraction_correction_entity_types: set[str] | frozenset[str] | None = None,
        graph_config: Optional[GraphConfig] = None,
        cloud_enabled: bool = False,
        graph_enrichment_neighborhood_hops: int = 2,
        graph_enrichment_max_entities_per_pass: int = 50,
        thermal_policy: ThermalPolicy | None = None,
        keep_prior_slots: int = 3,
        telemetry_dir: str | Path | None = None,
        incidents_state_dir: str | Path | None = None,
    ):
        # Bounded fold VRAM/adapter telemetry ring (paths.telemetry). ``None``
        # (the default for every experiment/test construction site) skips all
        # telemetry writes — only the production site
        # (paramem/server/consolidation.py) passes ``config.telemetry_dir``,
        # which is always a Path (ServerConfig.telemetry_dir ->
        # self.paths.telemetry, a dataclass field always Path-wrapped by the
        # yaml loader) — never an empty string in practice, but ``is not
        # None`` (not truthiness) is the correct check regardless.
        # Always-on when set; NOT gated on ``debug``.
        self._telemetry_dir: Path | None = (
            Path(telemetry_dir) if telemetry_dir is not None else None
        )
        # Operator-visible incident store (``data/state``) for non-fatal
        # degradations — e.g. a cloud-enrichment hiccup that fell back to
        # pre-enrichment facts.  Threaded from the server bootstrap exactly like
        # ``telemetry_dir``; ``None`` for experiments/tests, which record no
        # incidents.  Recorded through the same ``record_incident`` primitive
        # the outage path uses — one incident surface, called directly the way
        # ``commit_tier_slot`` calls ``save_adapter``.
        self._incidents_state_dir: Path | None = (
            Path(incidents_state_dir) if incidents_state_dir is not None else None
        )
        self._keep_prior_slots = keep_prior_slots
        # ``ServerConfig.cloud.enabled``, passed in at the bootstrap call site
        # (paramem/server/consolidation.py).  Necessary but not sufficient —
        # every call is admitted by ``evaluate_cloud_egress``.
        self.cloud_enabled = cloud_enabled
        # BASE-MODEL HOLDER (ConsolidationLoop): released via
        # _state["consolidation_loop"]=None in _release_base_model_in_process.
        self.model = model
        self.tokenizer = tokenizer
        self.config = consolidation_config
        self.training_config = training_config
        self.shutdown_requested = False  # set by the server's lifespan shutdown to stop training
        # Thermal policy is supplied by the caller (None when
        # consolidation.training_temp_limit <= 0, the default).  Live-server
        # only by construction: experiments and tests that don't override the
        # default get None and the throttle is never installed at the
        # train_adapter call site.  The schedule config (which actually
        # carries the thermal fields) lives in server.config and is not
        # reachable from this module — the loop accepts the precomputed
        # ThermalPolicy instead of re-deriving it.
        self._thermal_policy = thermal_policy
        # THE tier-existence fact for this loop: {tier_name: AdapterConfig},
        # in MAIN_TIERS order for the tiers this deployment has — a tier
        # exists iff adapters.<tier>.enabled. PRODUCTION SOURCE:
        # config.tier_config_map(), threaded by
        # paramem.server.consolidation.create_consolidation_loop.
        self.tier_adapters: dict[str, AdapterConfig] = dict(tier_adapters)
        self.wandb_config = wandb_config
        self.save_cycle_snapshots = save_cycle_snapshots
        # Run ID identifies a single ConsolidationLoop construction so successive
        # /consolidate calls (and parallel test workers) don't clobber each
        # other's debug artifacts. Format: YYYYmmddTHHMMSSZ_<hex6> — sortable
        # lexicographically, human-readable, sub-second-unique. Stdlib only.
        if run_id is None:
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            run_id = f"{ts}_{secrets.token_hex(3)}"
        self.run_id = run_id
        # Debug-snapshot root (paths.debug); tier/interim/cycle/run nesting
        # is added per-write via :meth:`snapshot_dir_for`
        # (paths.debug/episodic/[interim_<stamp>/]cycle_<N>/run_<run_id>/).
        # ``self.snapshot_dir`` is preserved for the HF-Trainer checkpoint
        # dir builder in :meth:`_training_output_dir`.
        self._debug_base: Path | None = Path(snapshot_dir) if snapshot_dir else None
        self.snapshot_dir = self._debug_base / f"run_{self.run_id}" if self._debug_base else None
        self.prompts_dir = prompts_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Extraction pipeline — the single chokepoint for ``extract_graph`` /
        # ``extract_procedural_graph``.  Owns the 12 cloud-pipeline tunables
        # (temperature, max_tokens, anonymizer / enrichment_provider / plausibility /
        # scrub categories, etc.) sourced from the ``extraction_*``
        # ConsolidationLoop kwargs.  Every consolidation call site reaches the
        # extractors through ``self.extraction.run`` / ``run_procedural`` —
        # no direct ``extract_graph(...)`` calls in this module.
        #
        # Cloud egress PII anonymization scope (``extraction_scrub_categories``)
        # is sourced at the bootstrap call site from
        # ``ServerConfig.sanitization.scrub_categories`` — the ``scrub`` hint
        # list already resolved once at config construction — so consolidation
        # honours the same operator policy as inference-time cloud egress.
        # Required — no implicit default anywhere below the config layer (the
        # tagger's configured labels are the sole scope authority; a
        # graph-layer fallback constant would be a duplicated, out-of-layer
        # privacy policy — see ``paramem/cloud/placeholders.py``'s
        # ``build_forward_table`` docstring).
        # BASE-MODEL HOLDER (loop.extraction.model): ExtractionPipeline stores
        # model on self.extraction.model; released via loop.release() →
        # self.extraction.model = None.
        self.extraction = ExtractionPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            config=ExtractionConfig(
                temperature=extraction_temperature,
                max_tokens=extraction_max_tokens,
                plausibility_max_tokens=extraction_plausibility_max_tokens,
                anonymize_token_envelope=extraction_anonymize_token_envelope,
                enrichment_provider=extraction_enrichment_provider,
                enrichment_provider_model=extraction_enrichment_provider_model,
                enrichment_provider_endpoint=extraction_enrichment_provider_endpoint,
                plausibility_judge=extraction_plausibility_judge,
                plausibility_stage=extraction_plausibility_stage,
                plausibility_model=extraction_plausibility_model,
                plausibility_endpoint=extraction_plausibility_endpoint,
                scrub_categories=extraction_scrub_categories,
                correction_entity_types=extraction_correction_entity_types,
                cloud_enabled=cloud_enabled,
            ),
            prompts_dir=prompts_dir,
            model_name=model_name,
        )

        # Graph-level cloud enrichment knobs.
        self.graph_enrichment_neighborhood_hops = graph_enrichment_neighborhood_hops
        self.graph_enrichment_max_entities_per_pass = graph_enrichment_max_entities_per_pass

        gc = graph_config or GraphConfig()
        self.merger = GraphMerger(
            similarity_threshold=gc.entity_similarity_threshold,
            prompts_dir=self.prompts_dir,
        )
        self.last_session_graph = None

        # The cumulative graph is NOT loaded at construction.  The fold
        # (consolidate) calls merger.reset_graph() before
        # re-merging registry-true relations so the keying surface is always
        # fresh; any prior graph state would be discarded at fold entry and
        # loading it here would only populate ingest-time data that nobody reads.

        # Every tier this loop trains is already resident on ``self.model`` —
        # created cold by load_base_model / ensure_resident_tiers at boot,
        # before this loop is ever constructed. This is what lets
        # ConsolidationLoop(model=None) (cloud-only) construct without
        # crashing rather than needing a model to create adapters on.
        self._clean_stale_staging_dir()

        # Attach the live model to the merger so model-only contradiction
        # resolution is always-on during merge calls.
        # BASE-MODEL HOLDER (GraphMerger): model-only contradiction at merge.
        self.merger.model = self.model
        self.merger.tokenizer = self.tokenizer

        # Per-tier indexed-key memory store — injected by the caller.
        # Lifespan-owned in production; experiments construct + hydrate
        # their own and pass it in.  The store is the single source of
        # truth for {entry, simhash, registry} of every indexed key.
        self.store = memory_store
        # Real (non-donor) minting floor: paramem.training.donor reserves
        # graph1-graph{DONOR_KEY_BAND_WIDTH} and proc1-proc{DONOR_KEY_BAND_WIDTH}
        # for the donor's synthetic training population, so counters seed at
        # DONOR_KEY_FLOOR (= width + 1) rather than 1.  The caller is
        # responsible for having hydrated registries before this point (via
        # ``MemoryStore.load_registries_from_disk`` or by injecting the
        # lifespan-loaded store) — see :meth:`_derive_key_counters` for the
        # scan that raises these off the floor.
        self._indexed_next_index: int = DONOR_KEY_FLOOR
        self._procedural_next_index: int = DONOR_KEY_FLOOR
        self._derive_key_counters()

        self.cycle_count = 0

        # Keys already promoted (prevent re-promotion after restart)
        self.promoted_keys: set[str] = set()

        # Transient per-staging-pass set of keys `_promote_working_keys`
        # decided to promote, adopted into `promoted_keys` only once the
        # event's `all_live` verdict is confirmed at the end of
        # `run_build_and_publish` -- see `_promote_working_keys`'s own
        # docstring for why the merge cannot happen while the event is
        # still staging.
        self._pending_promoted_keys: set[str] = set()

        # BackgroundTrainer reference — wired after construction by the server
        # lifespan or create_consolidation_loop caller.  When set,
        # _build_training_hooks routes through bt.training_hooks_for_job so
        # the abort event is included in the shutdown predicate.
        self._bg_trainer = None

    def _build_training_hooks(
        self,
        *,
        on_epoch_persist: "Optional[Callable[[int, str], None]]" = None,
        on_save_persist: "Optional[Callable[[int, str], None]]" = None,
    ) -> TrainingHooks:
        """Construct TrainingHooks honouring consolidation shutdown + BG abort.

        Routes through ``self._bg_trainer.training_hooks_for_job`` when a
        BackgroundTrainer is wired, so the abort event (set by
        ``abort_for_inference()``) is ORed into the shutdown predicate
        alongside the consolidation ``shutdown_requested`` flag.

        When no BackgroundTrainer is wired (experiment paths), returns a plain
        ``TrainingHooks`` with just the consolidation shutdown_requested check.

        Args:
            on_epoch_persist: Passed through to ``TrainingHooks`` unchanged.
            on_save_persist: Passed through to ``TrainingHooks`` unchanged.

        Returns:
            A ``TrainingHooks`` instance ready to pass to ``train_adapter``.
        """

        def base() -> bool:
            return self.shutdown_requested

        bt = getattr(self, "_bg_trainer", None)
        if bt is not None:
            return bt.training_hooks_for_job(
                base_shutdown_predicate=base,
                on_epoch_persist=on_epoch_persist,
                on_save_persist=on_save_persist,
            )
        return TrainingHooks(
            on_shutdown_check=base,
            on_epoch_persist=on_epoch_persist,
            on_save_persist=on_save_persist,
        )

    def release(self) -> None:
        """Drop all base-model references this loop holds so the model can be freed.

        Called by :func:`paramem.server.app._release_base_model_in_process`.
        Nulls ``model``/``tokenizer``/``_bg_trainer``/``extraction`` (and
        ``extraction.model``), and delegates to :meth:`GraphMerger.release` to
        null ``merger.model``/``merger.tokenizer``.  After this call no live
        attribute on this object (or on any sub-object it owns) retains a
        reference to the base model.

        ``ExtractionPipeline`` stores only ``self.model`` and
        ``self.tokenizer`` at the top level; there are no deeper sub-object
        holders, so nulling ``extraction.model`` is sufficient.

        Idempotent: safe to call multiple times or on a partially-constructed
        loop.
        """
        self.model = None
        self.tokenizer = None
        self._bg_trainer = None
        if getattr(self, "extraction", None) is not None:
            self.extraction.model = None  # BASE-MODEL HOLDER: ExtractionPipeline.model
            self.extraction = None
        if getattr(self, "merger", None) is not None:
            self.merger.release()  # BASE-MODEL HOLDER (GraphMerger)

    def seed_key_metadata(self, cycle_count: int) -> None:
        """Restore loop-level state derived from the per-tier ``key_metadata.json`` files.

        Args:
            cycle_count: The maximum ``tier_cycle`` across every tier's own
                ``key_metadata.json``, as computed by
                :func:`~paramem.server.consolidation.load_max_tier_cycle`.  A
                derivation, not a durably-flushed global counter: a cycle
                that writes no tier never advances what a later boot
                recomputes here.

        Rebuilds ``promoted_keys`` from the per-key ``promoted`` flag on
        :attr:`store`'s already-loaded bookkeeping — the ordinary boot
        sequence loads registries, then bookkeeping
        (:meth:`~paramem.memory.store.MemoryStore.load_bookkeeping_from_disk`),
        then calls this — rather than from a global ``promoted_keys`` list.
        Per-key rows are the bookkeeping loader's business, not this
        method's.

        Per the wipe invariant: a tier's ``key_metadata.json`` is
        bookkeeping, not a recovery source.
        """
        self.cycle_count = cycle_count
        # A promoted-then-staled key is still legitimately known via
        # is_known(); the flag itself already only lives on rows the
        # bookkeeping loader accepted (tier_for_known_key resolved), so this
        # check is a defensive re-verification, not the primary filter.
        self.promoted_keys = {
            key
            for key, bk in self.store.iter_bookkeeping()
            if bk.get("promoted") and self.store.is_known(key)
        }
        logger.info(
            "Seeded key metadata: cycle=%d, %d promoted",
            self.cycle_count,
            len(self.promoted_keys),
        )

    def _derive_key_counters(self) -> None:
        """(Re)derive the mint-index counters from every known key in the store.

        Raises ``_indexed_next_index``/``_procedural_next_index`` to one past
        the highest numeric suffix found among ``self.store.all_known_keys()``
        (active AND withheld — a withheld id still has to keep its number
        reserved: it stays known until its own tier's rebuild retires it,
        and a reissued number before then would mint a fresh key over an id
        an enumerator may still report known). Never lowers either counter —
        both start at :data:`DONOR_KEY_FLOOR` and only ``max()`` upward from
        there.

        Called from ``__init__``, scanning whatever the injected store was
        hydrated with — the id-reservation floor for every subsequent mint
        on this loop instance. Also re-called by
        :func:`~paramem.server.app._finish_resumed_event` after it rebinds
        ``self.store`` to the freshly published post-heal store (a
        quarantine lift completing mid-resume) — the same re-derivation
        against the new store's known keys, on the same loop instance.
        """
        self._indexed_next_index = DONOR_KEY_FLOOR
        self._procedural_next_index = DONOR_KEY_FLOOR
        for key in self.store.all_known_keys():
            if key.startswith("graph"):
                try:
                    idx = int(key.removeprefix("graph"))
                    self._indexed_next_index = max(self._indexed_next_index, idx + 1)
                except ValueError:
                    pass
            elif key.startswith("proc"):
                try:
                    idx = int(key.removeprefix("proc"))
                    self._procedural_next_index = max(self._procedural_next_index, idx + 1)
                except ValueError:
                    pass

    def commit_main_tiers(self, tiers: "list[str]", *, output_dir: Path) -> "set[str]":
        """Commit each tier in *tiers* through the per-tier primitive, train mode.

        The one caller-facing collapse of "derive the full-window stamp, then
        loop :func:`~paramem.memory.persistence.commit_tier_slot` once per
        main tier" — the composition every main-tier weights commit shares.
        Derives the stamp internally via
        :func:`~paramem.memory.interim_adapter.current_full_consolidation_stamp`
        (``self.full_consolidation_period_string``), then calls
        ``commit_tier_slot(loop=self, tier=t, adapter_name=t, stamp=<derived>,
        mode="train", all_keyed=[], output_dir=output_dir)`` once per tier in
        *tiers*, in order.  ``all_keyed=[]`` triggers ``commit_tier_slot``'s
        canonical-store projection fallback — this call carries no
        keyed-entry list of its own.

        Used by the trial-migration path in ``paramem.server.app`` (*tiers* =
        the resident main adapters, unchanged by the trial event, copied
        into the trial tree).  Commits whatever list it is given,
        unconditionally and without a residency check.

        Args:
            tiers: Tier/adapter names to commit, e.g.
                ``["episodic", "semantic"]``.  An empty list is a no-op.
            output_dir: Adapter store root forwarded to
                :func:`~paramem.memory.persistence.commit_tier_slot` as its
                own ``output_dir`` — the live store root for a main-tiers
                fold, or an isolated trial adapter tree for the
                trial-migration path.

        Returns:
            The set of tier names committed — always ``set(tiers)``; every
            tier in *tiers* is committed, none filtered.
        """
        from paramem.memory.interim_adapter import current_full_consolidation_stamp
        from paramem.memory.persistence import commit_tier_slot

        _period = getattr(self, "full_consolidation_period_string", "")
        _stamp = current_full_consolidation_stamp(_period)
        committed: "set[str]" = set()
        for tier in tiers:
            commit_tier_slot(
                loop=self,
                tier=tier,
                adapter_name=tier,
                stamp=_stamp,
                mode="train",
                all_keyed=[],
                output_dir=output_dir,
            )
            committed.add(tier)
        return committed

    @staticmethod
    def dedup_episodic(qa_list: list[dict]) -> list[dict]:
        """Deduplicate episodic QA/relation dicts by triple identity.

        Identity key is ``(canonical(subject), canonical(predicate), canonical(object))``.
        First occurrence wins.  Entries missing any of the three identity
        fields are DROPPED — an incomplete triple cannot be keyed and must
        not produce a ghost ``__unkeyed__`` entry.
        """
        seen: set[tuple] = set()
        out: list[dict] = []
        for qa in qa_list:
            subj = canonical(qa.get("subject") or "")
            pred = canonical(qa.get("predicate") or "")
            obj = canonical(qa.get("object") or "")
            if not (subj and pred and obj):
                continue
            key = (subj, pred, obj)
            if key in seen:
                continue
            seen.add(key)
            out.append(qa)
        return out

    @staticmethod
    def dedup_procedural(rels: list[dict]) -> list[dict]:
        """Deduplicate procedural relations by (subject, predicate, object).

        Identity key is ``(canonical(subject), canonical(predicate), canonical(object))``.
        Entries missing any of the three identity fields are DROPPED — an
        incomplete triple cannot be keyed and must not produce a ghost entry.
        """
        seen: set[tuple] = set()
        out: list[dict] = []
        for rel in rels:
            subj = canonical(rel.get("subject") or "")
            pred = canonical(rel.get("predicate") or "")
            obj = canonical(rel.get("object") or "")
            if not (subj and pred and obj):
                continue
            key = (subj, pred, obj)
            if key in seen:
                continue
            seen.add(key)
            out.append(rel)
        return out

    def _cache_entry(
        self,
        *,
        key: str,
        subject: str,
        predicate: str,
        object: str,
        speaker_id: str,
        relation_type: str = "factual",
    ) -> dict:
        """Build a uniform ``indexed_key_cache`` cache entry.

        Carries ``subject``/``predicate``/``object`` as the canonical triple
        fields.

        Using this helper for every cache-write site ensures the uniform shape
        is maintained by construction — every downstream reader (promotion-match,
        full-fold triple-lookup) reads the canonical field
        names.

        Args:
            key: The ``graphN`` / ``procN`` key string.
            subject: Triple subject.
            predicate: Triple predicate.
            object: Triple object.
            speaker_id: Speaker scope.
            relation_type: Model-assigned relation type from extraction
                (e.g. ``"factual"``, ``"preference"``, ``"temporal"``,
                ``"social"``).  Defaults to ``"factual"`` when unspecified;
                pass explicitly at every call site that has one available.

        Returns:
            Dict with the canonical cache shape.
        """
        return {
            "key": key,
            "subject": subject,
            "predicate": predicate,
            "object": object,
            "speaker_id": speaker_id,
            "relation_type": relation_type,
        }

    def _probe_recall(self, adapter_name: str, entries: "list[dict]") -> RecallProbe:
        """Run an uncapped per-key recall probe of *adapter_name* over *entries*.

        The ONE probe primitive every staged-weights verdict is built on:
        the all-or-nothing gate (:meth:`_assert_tier_recall`, shared by
        both the main-tiers fold and the interim-slot fold) routes through
        this method.  Always probes the FULL entries list — no sampling cap.

        Exceptions propagate — a probe that cannot run is not a verdict.

        Gradient checkpointing is disabled by ``evaluate_indexed_recall``
        itself for the duration of the probe and re-enabled here afterward
        (when configured on) — this runs mid-fold, before the promote and
        before the next tier trains.

        Args:
            adapter_name: Active adapter name for the probe — the staging
                slot for a just-trained tier.
            entries: Full per-tier entry list to probe (no truncation).

        Returns:
            :class:`~paramem.training.recall_eval.RecallProbe` carrying the
            per-key verdict.
        """
        from paramem.memory.entry import build_registry as _build_registry_inner
        from paramem.training.recall_eval import evaluate_indexed_recall

        try:
            result = evaluate_indexed_recall(
                self.model,
                self.tokenizer,
                entries,
                _build_registry_inner(entries),
                adapter_name=adapter_name,
                batch_size=self.training_config.recall_probe_batch_size,
            )
        finally:
            self._enable_gradient_checkpointing()
        return RecallProbe(per_key=tuple(result["per_key"]))

    def _assert_tier_recall(self, adapter_name: str, probe: RecallProbe) -> None:
        """The ONE training-completeness verdict, shared by every fold.

        Called against the staged weights (:data:`~paramem.training.trainer.STAGING_ADAPTER`)
        immediately before the write, inside :meth:`ConsolidationLoop._train_gate_write` —
        for a main tier and an interim slot alike.  A refusal therefore
        never touches any tier's live, serving state: the write that would
        write the tier's slot has not run yet, and no tier is mounted live
        until the whole bundle has written.  It says nothing about the
        tier's VRAM state during training: only a LoRA-config-mismatch
        recreate (:func:`~paramem.models.loader.ensure_adapter_matching`,
        run before :func:`~paramem.models.loader.tier_backup_scope` is even
        entered) may already have rewritten the resident adapter before
        this tier's own training started — donor seeding and a cold-init
        reconcile touch only the transient staging slot, never the resident
        adapter — so ``tier_backup_scope``'s restore covers exactly that
        one tier's training.

        Verdict over the tier's FULL key set, never a sample: *probe* already
        ran against every entry this fold assembled for the tier.

        Duplicate-tolerant: the denominator is ``probe.distinct_total``, the
        DISTINCT key count — a caller that probed a list with a repeated key
        must never refuse at genuine 100% recall.

        Deliberately not threshold-fed: unlike the migration path's own
        probe (:func:`~paramem.server.active_store_migration._migrate_tier_simulate_to_train`,
        ``recall_sanity_threshold``), this gate never admits a partial
        pass — the comparison is ``passing == total``.

        Args:
            adapter_name: The slot's adapter name — one of the three main
                tiers (``"episodic"`` / ``"semantic"`` / ``"procedural"``)
                or an interim slot name (e.g.
                ``"episodic_interim_<stamp>"``) — named in the refusal
                message, not the probe target (the probe already ran against
                the staged weights before this is called).
            probe: The :class:`~paramem.training.recall_eval.RecallProbe`
                already run against the tier's staged weights.

        Raises:
            RecallGateRejected: when any key fell short of exact-match
                recall on the staged weights.  Names the tier, the
                passing/total count, the failing key names, and the two
                operator levers: the tier's LoRA ``rank``/``alpha`` under
                ``adapters:`` in ``server.yaml``, and the training-budget
                table in ``paramem/utils/config.py``.
        """
        total = probe.distinct_total
        n_passing = len(probe.passing_keys)
        rate = n_passing / total if total else 1.0
        if n_passing < total:
            failed_keys = tuple(sorted({r["key"] for r in probe.failed}))
            raise RecallGateRejected(
                f"_assert_tier_recall: tier '{adapter_name}' reached {n_passing}/{total} "
                f"keys ({rate:.3f}) on its own trained weights — short of the required "
                "100%.  This is a capacity/epoch-budget limit, not a transient failure; "
                "it will recur every cycle until the tier gets more capacity.  Raise "
                "this tier's LoRA rank (and alpha) under adapters: in server.yaml, or "
                "review the training-budget table in paramem/utils/config.py.  The "
                "fold has been refused before any durable write — former "
                "adapters/registries stay live, pending conversations stay pending.",
                adapter_name=adapter_name,
                recall_rate=rate,
                threshold=1.0,
                failed_keys=failed_keys,
            )

    def _entries_from_graph(
        self,
        session_graph,
        *,
        procedural_enabled: bool,
    ) -> tuple[list[dict], list[dict]]:
        """Build entry relation dicts from a session graph — no model call.

        Partitions the session graph's relations into episodic/procedural.
        Each relation is projected via :func:`_relation_to_entry_dict`, which
        canonicalizes the predicate so interim-tier entries match the identity
        form the merger stamps onto the cumulative edge.  Entity scalar
        attributes are already present in ``session_graph.relations`` as
        attribute-typed relations by the time this method runs —
        :meth:`~paramem.graph.extraction_pipeline.ExtractionPipeline._run_extractor`
        projects them via ``relation_prep.attribute_relations`` before the
        graph is returned — so no separate projection step is needed here.

        Returns:
            ``(episodic_relations, procedural_relations)`` — both are lists of
            relation dicts suitable for ``assign_keys``.

        Note:
            This method has no ``model.generate`` calls, so no vram_scope
            wrapping is needed here, though a trailing
            ``torch.cuda.empty_cache()`` at the call site is still recommended
            for allocator hygiene on multi-session cycles.
        """
        from paramem.graph import relation_prep

        relation_dicts = [_relation_to_entry_dict(r) for r in session_graph.relations]
        return relation_prep.partition_relations(
            relation_dicts, procedural_enabled=procedural_enabled
        )

    def snapshot_dir_for(self, *, interim_stamp: str | None = None) -> Path | None:
        """Return this loop's per-cycle/per-run debug-snapshot directory.

        Layout:

            paths.debug/episodic/[interim_<stamp>/]cycle_<N>/run_<run_id>/

        Tier prefix is fixed to ``episodic`` since every cycle's
        graph/relation/sessions debug artifacts are anchored on the
        episodic-primary extraction; a procedural / semantic-only writer
        can introduce its own tier root when needed.

        Returns ``None`` when debug snapshots are disabled (no
        ``snapshot_dir`` was wired into the loop).
        """
        if not self.save_cycle_snapshots or self._debug_base is None:
            return None
        parts: list[str] = ["episodic"]
        if interim_stamp:
            parts.append(f"interim_{interim_stamp}")
        parts.append(f"cycle_{self.cycle_count}")
        parts.append(f"run_{self.run_id}")
        return self._debug_base.joinpath(*parts)

    def _artifact_scope(self, *, interim_stamp: str | None = None):
        """Open the debug artifact root for the work in this block.

        Every artifact hook fired inside — at any depth, including from the
        extraction pipeline and the graph-tier refiner — lands under
        :meth:`snapshot_dir_for`.  ``None`` from that method (debug off) is
        passed through: :func:`~paramem.utils.artifacts.debug_run` treats it
        as "this producer is inactive", which is how the gate is expressed
        without any caller testing the flag.
        """
        return debug_run(self.snapshot_dir_for(interim_stamp=interim_stamp))

    def extract_session(
        self,
        session_transcript: str,
        session_id: str,
        speaker_id: str,
        speaker_name: str | None = None,
        enrichment_provider: str | None = None,
        enrichment_provider_model: str | None = None,
        enrichment_provider_endpoint: str | None = None,
        plausibility_judge: str | None = None,
        plausibility_stage: str | None = None,
        source_type: str = "transcript",
        event_time: str | None = None,
    ) -> tuple[list[dict], list[dict]]:
        """Extract and generate relations from a session without training.

        Returns ``(episodic_rels, procedural_relations)`` for deferred training.
        Merges the session graph into the cumulative graph.

        Args:
            session_transcript: Raw session text (conversation transcript or
                document chunk).
            session_id: Unique identifier for this session.
            speaker_id: Speaker identifier for preference scoping. Required —
                callers must always supply a real speaker ID.
            speaker_name: Real speaker name injected via ``{speaker_context}``
                in the user template for narrator binding.
            source_type: ``"transcript"`` (default) for voice/chat sessions;
                ``"document"`` for written documents fed through the ingest
                pipeline.  Selects the ``{document_context}`` rendering
                (:func:`~paramem.graph.extractor.build_document_context`)
                and gates the document-only exact-full-name speaker
                rewrite.  The system prompt and user template are the same
                for every source type; narrator binding for document
                sources uses the same ``build_speaker_context`` mechanism
                as transcripts — no separate ``doc_title`` or context
                string is needed.
            event_time: Session-start assertion time (ISO 8601), typically
                the session's ``started_at``. Forwarded to the extraction
                chokepoint as ``timestamp`` so a NEW fact's edge
                ``last_seen`` reflects when it was asserted, not when
                extraction ran. ``None`` (default) falls back to ``now()``
                at the extractor layer — for callers that have no real
                session-start time to supply.

        Raises:
            ExtractionFailed: A local-extraction pass (``local_extract``,
                ``second_order_extract``, or ``procedural_extract``) failed
                to parse its output, or the cloud ``cloud_enrich`` stage
                failed. Raised before this session's merge, so no partial
                content reaches ``self.merger.graph`` for the failing pass;
                the merger graph is also reset before the exception leaves
                this method, invalidating the fold's accumulated
                extraction state (including any earlier session already
                merged this fold).
        """
        logger.info("=== Extraction (session=%s) ===", session_id)

        # Outer extraction_trace scope wraps the whole session body so the
        # orchestrator phases (merge_into_cumulative, procedural_extract,
        # dedup_*) record into the same trace as the inner extract_graph /
        # extract_procedural_graph calls — those traces nest-no-op into this
        # one.  The final attach_to(...) calls below capture the complete
        # phase history on each session graph before it is dumped.
        #
        # ONE invalidation site for BOTH ExtractionFailed raise origins — the
        # local-parse abort detected by _abort_on_local_parse_failure below,
        # and the existing cloud_enrich raise from inside self.extraction.run.
        # extract_session is the single common ancestor of every caller (both
        # consolidation callers, the trial path, experiment callers), so the
        # merger-graph reset lives here, never duplicated per caller.
        # VramExhausted is deliberately NOT caught — per-chunk isolation
        # keeps the batch's earlier merges intact.
        try:
            with extraction_trace() as trace:
                # --- EXTRACT ---
                _mark = len(trace.records)
                session_graph = self.extraction.run(
                    session_transcript,
                    session_id,
                    source_type=source_type,
                    enrichment_provider=enrichment_provider,
                    enrichment_provider_model=enrichment_provider_model,
                    enrichment_provider_endpoint=enrichment_provider_endpoint,
                    speaker_name=speaker_name,
                    speaker_id=speaker_id,
                    plausibility_judge=plausibility_judge,
                    plausibility_stage=plausibility_stage,
                    timestamp=event_time,
                )
                self._abort_on_local_parse_failure(trace, _mark)

                logger.info(
                    "Extracted %d entities, %d relations",
                    len(session_graph.entities),
                    len(session_graph.relations),
                )

                # --- MERGE ---
                with phase_trace("merge_into_cumulative") as t:
                    # Always merge into the cumulative graph.  resolve_contradictions
                    # is driven by refinement_contradiction config: when "off", Case-2
                    # cardinality resolution is skipped (no model call, no edge removal).
                    # When "on", the model may supersede older edges via the recency rule.
                    # Disable gradient checkpointing: merger.merge may call
                    # model.generate() when a model is present and
                    # resolve_contradictions=True.  HF silently disables the KV cache
                    # when checkpointing is active (CLAUDE.md rule).
                    self._disable_gradient_checkpointing()
                    try:
                        self.merger.merge(
                            session_graph,
                            resolve_contradictions=(self.config.refinement_contradiction == "on"),
                        )
                    finally:
                        self._enable_gradient_checkpointing()
                    t.add("triples_added", len(session_graph.relations))

                # --- BUILD ENTRY RELATION DICTS ---
                # Single entry point for graph → entries.  Builds relation dicts
                # directly from session_graph with no model.generate calls.
                episodic_rels, procedural_rels = self._entries_from_graph(
                    session_graph,
                    procedural_enabled="procedural" in self.tier_adapters,
                )

                # --- PROCEDURAL: separate extraction pass ---
                # extract_procedural_graph self-traces the "procedural_extract"
                # phase (nest-no-ops onto this outer extraction_trace scope) via
                # the shared _run_local_extraction primitive, so no wrapper is
                # needed here.
                proc_graph: SessionGraph | None = None
                if "procedural" in self.tier_adapters:
                    _mark = len(trace.records)
                    proc_graph = self.extraction.run_procedural(
                        session_transcript,
                        session_id,
                        speaker_name=speaker_name,
                        source_type=source_type,
                        speaker_id=speaker_id,
                        timestamp=event_time,
                    )
                    self._abort_on_local_parse_failure(trace, _mark)
                    # Route the procedural extractor's own relations through the
                    # same partition rule every other tier-set entry point uses
                    # (_entries_from_graph above, the keyed walk below) — a
                    # projected entity attribute the procedural extractor's
                    # graph carries (e.g. an email address) has
                    # relation_type="attribute" and routes procedural only when
                    # its predicate matches _PROCEDURAL_PREDICATES; scalar PII
                    # predicates route episodic, agreeing with the keyed walk's
                    # decision for the same node record.
                    _proc_dicts = [_relation_to_entry_dict(r) for r in proc_graph.relations]
                    _pg_episodic, _pg_procedural = partition_relations(
                        _proc_dicts, procedural_enabled=True
                    )
                    episodic_rels.extend(_pg_episodic)
                    procedural_rels.extend(_pg_procedural)
                    # Merge proc_graph into the cumulative graph so its relations
                    # reach the unified keying surface (stage_event's working-copy
                    # keyed walk, _build_working_keyed_walk) at the next
                    # run_consolidation_cycle call.  Same
                    # resolve_contradictions flag and gradient-checkpointing discipline
                    # as the session_graph merge above — merger.merge may call
                    # model.generate() when a model is present (CLAUDE.md rule).
                    self._disable_gradient_checkpointing()
                    try:
                        self.merger.merge(
                            proc_graph,
                            resolve_contradictions=(self.config.refinement_contradiction == "on"),
                        )
                    finally:
                        self._enable_gradient_checkpointing()

                # Unified dedup (identical policy across every consolidation caller).
                with phase_trace("dedup_episodic") as t:
                    episodic_rels = self.dedup_episodic(episodic_rels)
                    t.add("count", len(episodic_rels))
                with phase_trace("dedup_procedural") as t:
                    procedural_rels = self.dedup_procedural(procedural_rels)
                    t.add("count", len(procedural_rels))

                # Attach the complete trace (extraction + orchestrator phases) to
                # each session graph before dumping so diagnostics["phases"] holds
                # everything that fired this session.
                trace.attach_to(session_graph)
                with self._artifact_scope():
                    on_session_extracted(session_graph, session_id, "graph")
                    if proc_graph is not None:
                        trace.attach_to(proc_graph)
                        on_session_extracted(proc_graph, session_id, "procedural_graph")
        except ExtractionFailed:
            self.merger.reset_graph()
            raise

        self.last_session_graph = session_graph

        # Session-tier enrichment health is NOT arbitrated into an incident
        # here: writing one from this method would mean a non-staging run
        # (a calibration probe running the same extraction chain) mutates
        # production incident state.  ``self.last_session_graph.diagnostics``
        # carries the two signals (``"anonymize"`` / ``"cloud_enrichment_degraded"``)
        # a caller needs to build its own record; a staging caller passes the
        # batch's records to :meth:`arbitrate_enrichment_incidents`.

        # Release reclaimable device memory back to the WSL2 dxg layer at every
        # session boundary.  PyTorch's caching allocator retains freed blocks
        # (``reserved`` − ``allocated``); on this 8 GiB laptop, after a session's
        # plausibility-filter peak, that retained pool can hold ~700-1500 MiB
        # which dxg counts as in-use.  Without this, multi-session cycles
        # accumulate host-side residency until ``dxgkio_make_resident`` fails
        # with ENOMEM on the next session's first growth.  Uses
        # ``safe_empty_cache`` (not a bare
        # ``torch.cuda.empty_cache``) so the cuBLAS workspaces the extraction
        # chain's ~4 generate calls allocate outside the PyTorch allocator
        # (~280 MiB/cycle, untouched by ``empty_cache``) are released too.  In
        # the server path ``vram_scope`` already runs ``safe_empty_cache`` in
        # its ``finally`` after this call; this matters for experiment callers
        # of ``extract_session`` (e.g. ``dataset_probe.py``, ``lme_graph_builder.py``)
        # that are not wrapped.
        try:
            safe_empty_cache()
        except Exception:  # noqa: BLE001
            pass

        return episodic_rels, procedural_rels

    def _abort_on_local_parse_failure(self, trace: "ExtractionTrace", mark: int) -> None:
        """Raise :class:`~paramem.graph.extractor.ExtractionFailed` when a
        local-extraction phase recorded since *mark* failed to parse.

        Reads ``trace.records[mark:]`` — the phase records appended by the
        single extraction call the caller just made — via
        :func:`~paramem.graph.extractor.local_parse_failure`, the one
        implementation of this read. A slice, not the whole trace, is
        required: ``extraction_trace`` nests as a no-op and
        ``tests/conftest.py::_extraction_trace_scope`` wraps every test, so
        ``trace.records`` also carries records from earlier in the same
        scope.

        A no-op when nothing in the slice failed — a legitimately-empty
        extraction (no failed record) is not mistaken for a failure here.

        Args:
            trace: The active :class:`~paramem.graph.phase_trace.ExtractionTrace`
                (``extract_session``'s own ``with extraction_trace() as
                trace:`` scope).
            mark: ``len(trace.records)`` captured immediately before the
                extraction call this guard follows — both parameters are
                call-local, produced two lines above each call site.

        Raises:
            ExtractionFailed: With ``phase`` set to the failing
                local-extraction phase name and ``reason`` from the phase
                record (falling back to a one-line description when the
                record carries none).
        """
        record = local_parse_failure(trace.records[mark:])
        if record is None:
            return
        raise ExtractionFailed(record.name, record.reason or f"{record.name} failed to parse")

    def arbitrate_enrichment_incidents(self, signals: "list[dict]") -> None:
        """Reconcile the ``enrichment_degraded`` incident state for a batch
        of sessions.

        THE public door for the arbitration — ``extract_session`` itself
        never performs it (writing an incident from inside extraction would
        let a non-staging run — a calibration probe running the same chain —
        mutate production incident state).  Called once per batch, after
        extraction, by a STAGING caller only — the server layer's own
        extraction pre-stage collects one signal per session from
        ``session_graph.diagnostics`` right after each ``extract_session``
        call and passes the whole batch here.

        Args:
            signals: One record per session —
                ``{"session_id", "anonymize", "cloud_enrichment_degraded"}``,
                the same two diagnostic fields ``stage_anonymize`` wrote onto
                each session's graph.

        A safe no-op when ``self._incidents_state_dir is None``.
        """
        if self._incidents_state_dir is None:
            return
        for signal in signals:
            self._arbitrate_one_enrichment_signal(signal)

    def _arbitrate_one_enrichment_signal(self, signal: dict) -> None:
        """Reconcile the ``enrichment_degraded`` incident state for one
        session against its own enrichment signal (see
        :meth:`arbitrate_enrichment_incidents`).
        ``cloud_enrichment_degraded`` alone can't tell "ran cleanly" apart
        from "never ran" (empty relations, a calibration stop, cloud egress
        refused), so reading it alone would silently resolve a standing
        incident with no evidence of recovery; ``anonymize`` gates ``enrich``
        with the same admission check and separates the two cases.

        - ``"ok"`` / ``"opted_out"`` (opted-out still reaches ``enrich``):
          resolve ``anonymize``, then arbitrate ``cloud_enrich`` by
          ``cloud_enrichment_degraded``.
        - ``"failed"``: record a distinct incident keyed ``anonymize``
          (same-type-different-key, like ``graph_enrich_vram``);
          ``cloud_enrich`` is left untouched.
        - absent (``None``): no signal, EXCEPT when cloud egress is refused —
          neither sub-incident can ever self-heal there, so resolve every
          open ``enrichment_degraded`` incident with a persisted reason.
        - any other value: not a real writer output — log a warning and
          touch nothing, rather than conflating it with "absent".

        Callers already guard ``self._incidents_state_dir is None`` (see
        :meth:`arbitrate_enrichment_incidents`).
        """
        from paramem.server.incidents import (
            record_incident,
            resolve_incident,
            resolve_incidents_by_type,
        )

        session_id = signal["session_id"]
        anonymize_outcome = signal.get("anonymize")

        if anonymize_outcome in ("ok", "opted_out"):
            resolve_incident(self._incidents_state_dir, "enrichment_degraded", "anonymize")

            degraded = signal.get("cloud_enrichment_degraded")
            if degraded is None:
                resolve_incident(self._incidents_state_dir, "enrichment_degraded", "cloud_enrich")
            else:
                record_incident(
                    self._incidents_state_dir,
                    type="enrichment_degraded",
                    key="cloud_enrich",
                    severity="warning",
                    summary=(
                        "Session-tier cloud enrichment degraded (per transcript, at "
                        "extraction) — unparseable response; kept pre-enrichment facts"
                    ),
                    detail={
                        "type": "enrichment_degraded",
                        "session_id": session_id,
                        **degraded,
                        "at": datetime.now(timezone.utc).isoformat(),
                    },
                )
        elif anonymize_outcome == "failed":
            record_incident(
                self._incidents_state_dir,
                type="enrichment_degraded",
                key="anonymize",
                severity="warning",
                summary=(
                    "Session-tier anonymization failed — cloud enrichment skipped this session"
                ),
                detail={
                    "type": "enrichment_degraded",
                    "session_id": session_id,
                    "at": datetime.now(timezone.utc).isoformat(),
                },
            )
        elif anonymize_outcome is None:
            # anonymize never ran this session — only actionable when cloud
            # egress is off, the same admission check that gated
            # anonymize/enrich in the first place (graph_tier.py's
            # normalization gate uses the identical no-override shape).
            cfg = self.extraction.config
            verdict = evaluate_cloud_egress(
                cloud_enabled=self.cloud_enabled,
                provider=cfg.enrichment_provider,
                model=cfg.enrichment_provider_model,
                endpoint=cfg.enrichment_provider_endpoint,
            )
            if not verdict.permitted:
                reason = "cloud egress disabled — enrichment cannot run"
                resolved = resolve_incidents_by_type(
                    self._incidents_state_dir, "enrichment_degraded", reason=reason
                )
                logger.info("Resolved %d enrichment_degraded incident(s) — %s", resolved, reason)
        else:
            # A value neither of the three writers ("ok"/"opted_out"/"failed")
            # ever produces — do not conflate it with "the op never ran"
            # (the case above): touch nothing, surface it so the drift gets
            # noticed rather than silently resolving on a guess.
            logger.warning(
                "Unrecognized session_graph.diagnostics['anonymize'] value %r — "
                "enrichment_degraded incidents left untouched",
                anonymize_outcome,
            )

    def train_adapters(
        self,
        all_episodic_rels: list[dict],
        all_procedural_relations: list[dict],
        speaker_id: str,
    ) -> dict:
        """Train all adapters once on accumulated relations (blocking).

        Called after all sessions have been extracted.  Returns the
        consolidation cycle's result dict verbatim.

        Delegates to :meth:`run_consolidation_cycle` (unified episodic +
        procedural pipeline) so experiment scripts exercise the same code path
        as the scheduled interim training path.  After the cycle, calls
        :meth:`consolidate` in train mode to fold the freshly-trained interim
        slot into the main ``"episodic"`` adapter so callers that probe
        ``model.set_adapter("episodic")`` read the trained weights, not the
        stale main slot.  This mirrors production's full fold, compressed for the
        one-shot experiment use case.

        The method is retained as the stable public API used by experiment
        scripts; its body is a single-call delegation — not a parallel
        implementation.

        Args:
            all_episodic_rels: Deduplicated episodic relations for this cycle.
            all_procedural_relations: Deduplicated procedural relations.
            speaker_id: Fallback speaker scope for procedural contradiction
                detection. Required — callers must always supply a real ID.

        Note: this method trains AND saves.  Experiment scripts use this
        combined method directly.
        """
        # This method's own callers merge session content directly via
        # extract_session (never through the server's own
        # _extract_pending_sessions), so this is where that batch's
        # extraction lifetime ends — the one take, ahead of the fold.
        pending = self.take_pending_relations()

        # cycle_count advances inside run_build_and_publish, only once this
        # call's event reaches all_live -- a pre-call snapshot, not a
        # lookahead to a number this call is guaranteed to reach.
        cycle_result = self.run_consolidation_cycle(
            all_episodic_rels,
            all_procedural_relations,
            speaker_id=speaker_id,
            mode="train",
            pending=pending,
            run_label=f"train-adapters-cycle{self.cycle_count}",
        )

        # --- Roll interim slot into main ---
        # run_consolidation_cycle trains into episodic_interim_<stamp>.  Callers
        # that probe model.set_adapter("episodic") need the trained weights in the
        # main slot.  Submit the train fold via an ephemeral BackgroundTrainer so
        # the GPU lock is held for the full per-tier rebuild (consolidate requires
        # this in train mode — see its entry guard).
        # submit_and_wait blocks until the worker finishes and re-raises on error.
        from paramem.memory.interim_adapter import INTERIM_NAME_PREFIX

        if "episodic" in self.model.peft_config or any(
            k.startswith(INTERIM_NAME_PREFIX) for k in self.model.peft_config
        ):
            from paramem.server.background_trainer import BackgroundTrainer

            _bt = BackgroundTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                training_config=self.training_config,
                output_dir=self.output_dir,
                thermal_policy=getattr(self, "_thermal_policy", None),
            )

            def _consolidate() -> None:
                # The roll-into-main step: the product self.take_pending_relations()
                # captured above was already consumed by run_consolidation_cycle,
                # so this event folds no pending-session content of its own.
                self.consolidate(mode="train", pending=None, trainer=_bt)

            try:
                _bt.submit_and_wait(_consolidate)
            finally:
                _bt.close()

        logger.info("Training complete: %s", cycle_result)
        return cycle_result

    def _tag_speaker_id_defaults(self, rels: list[dict], speaker_id: str) -> None:
        """Tag relations missing a ``speaker_id`` with the caller-supplied default.

        Mutates *rels* in place — every entry that does not already carry a
        ``speaker_id`` key receives the caller-supplied *speaker_id*.  Entries
        that already carry one (even an empty string) are left unchanged so
        per-relation speaker scoping is not overwritten.

        Args:
            rels: Relation dicts to tag.  Modified in place.
            speaker_id: Default speaker identifier to inject.
        """
        for r in rels:
            if "speaker_id" not in r:
                r["speaker_id"] = speaker_id

    def _resolve_target_slot(
        self,
        stamp: str,
    ) -> str:
        """Compute the target interim adapter name for this sub-interval.

        Pure name-minting helper: returns ``"episodic_interim_<stamp>"``.
        Ring-full detection and cap-pending routing live in
        ``run_consolidation_cycle``, which inspects PEFT config before deciding
        whether to stage and build this event.

        Args:
            stamp: The sub-interval stamp (``YYYYMMDDTHHMM``).

        Returns:
            Adapter name string ``"episodic_interim_<stamp>"``.
        """
        from paramem.memory.interim_adapter import INTERIM_NAME_PREFIX

        return f"{INTERIM_NAME_PREFIX}{stamp}"

    def _mint_keyed_entries(
        self,
        rels: list[dict],
        *,
        prefix: str,
        start_index: int,
        speaker_id: str,
    ) -> list[dict]:
        """Mint a fresh keyed-entry list from *rels* without mutating any shared state.

        This is a pure minting helper: it calls ``assign_keys`` and wraps each
        result in a :meth:`_cache_entry` dict.  It does **NOT** advance
        ``_indexed_next_index`` or ``_procedural_next_index``, does NOT write the
        :class:`~paramem.memory.store.MemoryStore`, and does NOT update the
        simhash registries.  All those mutations remain the caller's
        responsibility so the deferred-mutation contract of the surrounding
        training paths is preserved.

        Threads ``relation_type`` from each source relation dict through to the
        minted entry — both the episodic and procedural inline loops pass this
        field so the tier routing and the stored bytes are correct for
        ``"preference"``, ``"temporal"``, ``"social"`` entries.  Without it the
        store would silently re-tag them as ``"factual"`` and corrupt
        procedural routing.

        Args:
            rels: Relation dicts carrying at minimum ``subject``, ``predicate``,
                ``object``.  ``speaker_id`` and ``relation_type`` are read as
                optional fields with per-entry fallbacks.
            prefix: Key prefix (``"graph"`` for episodic/semantic;
                ``"proc"`` for procedural).
            start_index: First numeric index for the minted key sequence.
                The i-th entry gets key ``f"{prefix}{start_index + i}"``.
            speaker_id: Fallback speaker tag used when the relation dict does
                not carry a ``speaker_id`` field.

        Returns:
            List of cache-entry dicts in the same order as *rels*.
        """
        raw_keyed = assign_keys(
            [(r["subject"], r["predicate"], r["object"]) for r in rels],
            start_index=start_index,
            prefix=prefix,
        )
        minted: list[dict] = []
        for i, kp in enumerate(raw_keyed):
            rel = rels[i] if i < len(rels) else {}
            # assign_keys output never carries speaker_id (only key/s/p/o);
            # resolve from the source relation with the caller's id as fallback.
            sid = rel.get("speaker_id", speaker_id)
            entry = self._cache_entry(
                key=kp["key"],
                subject=kp["subject"],
                predicate=kp["predicate"],
                object=kp["object"],
                speaker_id=sid,
                relation_type=rel.get("relation_type", "factual"),
            )
            minted.append(entry)
        return minted

    @staticmethod
    def _indexed_dataset(examples: list[dict]) -> Dataset:
        """Wrap pre-tokenized indexed memory examples as a Dataset."""

        class _IndexedDataset(Dataset):
            def __init__(self, items):
                self.items = items

            def __len__(self):
                return len(self.items)

            def __getitem__(self, idx):
                return self.items[idx]

        return _IndexedDataset(examples)

    @property
    def _fold_state_dir(self) -> Path:
        """Parent state directory for ``stage_ledger.json``.

        Derived via :func:`~paramem.training.stage_ledger.data_state_dir`
        from ``output_dir.parent`` to match the production layout
        (``config.paths.data / "state"``).  For experiment callers with
        ``output_dir="outputs/phase3"`` this yields ``outputs/state``, which
        is self-contained and harmless. Ensures the directory exists — the
        one call site in this property that needs the mkdir side effect,
        since :func:`data_state_dir` itself has none.
        """
        from paramem.training.stage_ledger import data_state_dir

        d = data_state_dir(self.output_dir.parent)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _training_output_dir(self, adapter_name: str) -> Path:
        """Path passed to HuggingFace ``TrainingArguments(output_dir=...)``.

        This is **HF Trainer's required working directory**, not a ParaMem
        concept.  HF writes its ``checkpoint-<step>/`` subdirs there at every
        epoch (live config: ``save_strategy="epoch"``, ``save_total_limit=2``);
        :class:`EncryptCheckpointCallback` wraps each one in the age envelope
        in-place.  The :class:`BackgroundTrainer` resume path
        (``trainer.train(resume_from_checkpoint=...)``) reads the latest
        ``checkpoint-<step>/`` from the same directory after a graceful
        shutdown / restart, so this is NOT throwaway scratch — it is the
        substrate the resume mechanism depends on.

        Distinct from:

        - ``paths.debug/...`` — inspection artifacts (graph snapshots,
          relation dumps, retained session JSONL).  Plaintext, gated on
          ``debug=true``.  Produced by :meth:`snapshot_dir_for`.
        - ``paths.adapters/<tier>/[interim_<stamp>/]<slot_date>/`` — committed
          v3 adapter slots.  Written by :func:`atomic_save_adapter` at end of
          training.

        This function returns a path under ``paths.adapters/`` that lives
        alongside the slots in the same tier hierarchy but is always a
        scope-named ``cycle_<N>`` sub-dir NESTED one level below the tier
        (or interim-tier) root, so it is disjoint from every published
        surface at that root — the timestamped slot dir, and (interim only)
        ``indexed_key_registry.json`` / ``key_metadata.json`` — and HF's
        step-numbered ``checkpoint-<step>/`` subdirs never collide with any
        of them. This directory is disposable scratch: a caller may
        ``shutil.rmtree`` it without touching anything published. The same
        ``cycle_<N>`` shape nested under ``interim_<stamp>/`` is also the
        debug-snapshot layout's convention (:meth:`snapshot_dir_for`).

        Paths:

        - ``adapter_name == "episodic_interim_<stamp>"``:
          ``<output_dir>/episodic/interim_<stamp>/cycle_<N>/``
        - Full cycle, ``adapter_name in {episodic, semantic, procedural}``:
          ``<output_dir>/<adapter_name>/cycle_<N>/``

        Args:
            adapter_name: The PEFT adapter being trained.  One of
                ``"episodic"``, ``"semantic"``, ``"procedural"``, or
                ``"episodic_interim_<stamp>"``.

        Returns:
            Absolute :class:`~pathlib.Path` to give HF Trainer.

        Raises:
            ValueError: when ``adapter_name`` doesn't match any known tier
                or the interim-adapter naming convention.
        """
        from paramem.memory.interim_adapter import INTERIM_NAME_PREFIX, interim_stamp_from_name
        from paramem.utils.tiers import MAIN_TIERS

        # Episodic interim slot: scratch nested one level under the interim
        # tier root (sibling of <slot_date>/, indexed_key_registry.json and
        # key_metadata.json), scoped by cycle_<N> like the full-cycle branch
        # below — never the interim tier root itself, which also holds those
        # published files and must survive scratch disposal intact.
        stamp = interim_stamp_from_name(adapter_name)
        if stamp is not None:
            return self.output_dir / "episodic" / f"interim_{stamp}" / f"cycle_{self.cycle_count}"
        if adapter_name.startswith(INTERIM_NAME_PREFIX):
            raise ValueError(f"Malformed interim adapter name: {adapter_name!r}")

        if adapter_name not in MAIN_TIERS:
            raise ValueError(f"Unknown adapter name for training output dir: {adapter_name!r}")

        # Tier-level scratch under <tier>/, scoped to the current full cycle.
        return self.output_dir / adapter_name / f"cycle_{self.cycle_count}"

    def run_consolidation_cycle(
        self,
        episodic_rels: list[dict],
        procedural_rels: list[dict],
        *,
        speaker_id: str,
        mode: "Literal['simulate', 'train']",
        run_label: str,
        pending: "PendingRelations",
        schedule: str = "",
        max_interim_count: int = 7,
        interim_overflow_slack: int = 0,
        stamp: str | None = None,
        session_ids: "list[str] | None" = None,
    ) -> dict:
        """Unified interim-cycle entry: key prep + optional training + atomic persistence.

        Train and simulate execute the same stage-then-build-and-publish
        pipeline (:meth:`stage_event` /
        :meth:`run_build_and_publish`) — the ONLY mode-conditional code is
        inside :func:`~paramem.memory.persistence.write_tier_slot`'s own
        ``mode`` fork: train writes adapter weights, simulate writes a
        ``graph.json`` payload — both venues write through the same
        timestamped-slot envelope (:func:`~paramem.adapters.slot.write_slot`),
        never a tier-root sidecar file.

        Everything else — cycle counter, guards, speaker tagging, enrichment,
        procedural key prep, simhash update, end-of-cycle adapter switch — is
        mode-agnostic.

        Internal flow:

        1. Guard: no relations → early return ``{"mode": "noop", ...}``.
        2. Tag relations with caller's ``speaker_id`` as default.
        3. Compute stamp (when not provided) and call ``_resolve_target_slot``
           to obtain ``adapter_name``.
        4. Ring-full detection (train mode only, target slot new only): the
           3-way gate — ``cap_pending`` (ring + overflow both exhausted,
           returns immediately, sessions stay pending), overflow mint
           (tagged for the caller's own incident), or a normal mint (falls
           through).
        5. Mint/refresh the interim PEFT slot (weights venue only) via
           :func:`~paramem.memory.interim_adapter.create_interim_adapter` /
           :func:`~paramem.models.loader.ensure_adapter_matching`, then
           :meth:`_hydrate_store_for_fold`.  The batch's own merged
           relations arrive as the caller-supplied ``pending`` argument
           (the caller's own :meth:`take_pending_relations` take) —
           nothing here re-captures ``merger.graph``'s edges.
        6. Stage this event's shadow tree via :meth:`stage_event` — the
           interim scope always pins ``normalize``/``enrich`` ``False`` (see
           :class:`FoldScope`'s own docstring for the rationale); the interim
           slot holds BOTH factual (episodic) and preference (procedural)
           keys, trained with the attention-only episodic adapter config by
           design — procedural keys fold to the ``procedural`` main adapter
           only at the full fold.
        7. Build, gate, write and take the event live via
           :meth:`run_build_and_publish` (train mode) or its simulate-venue
           write (writes the ``graph.json`` payload instead of adapter
           weights; no training, no gate).  ``self.cycle_count`` advances
           exactly once here, only when the event's whole bundle reaches
           ``all_live`` — the one increment site, shared by every event kind
           and venue (see :meth:`run_build_and_publish`'s own docstring).
        8. Assemble the result dict from ``run_build_and_publish``'s summary
           and the ledger's own recorded extraction stage (session ids,
           relation counts).

        Args:
            episodic_rels: Pre-extracted episodic relations.  May already carry
                ``speaker_id``; missing entries are tagged with *speaker_id*.
            procedural_rels: Pre-extracted procedural relations.  Used for the
                no-relations guard (step 2) and debug output; procedural facts
                reach the training set via merger.graph (merged by
                extract_session), not via this argument directly.
            speaker_id: Default speaker tag for relations missing one.
                Required — callers must always supply a real speaker ID.
            mode: ``"train"`` writes adapter weights; ``"simulate"`` writes
                a ``graph.json`` payload into the same written-slot envelope,
                without touching PEFT.
            run_label: Tag woven into the wandb ``run_name`` for traceability.
                Pass ``session_id`` for per-session calls, or
                ``"tick-<stamp>"`` for batch calls from the scheduled tick.
            pending: The batch's merged extraction product — the caller's own
                :meth:`take_pending_relations` take, captured at the
                extraction boundary before this call.  Required: the
                pending session IS every interim cycle's content, so there
                is no caller of this method with nothing to pass.
            schedule: Consolidation refresh-cadence string used to compute the
                sub-interval stamp when *stamp* is not provided.
            max_interim_count: Cap on concurrent interim adapters.  When the
                ring is at or beyond capacity (train mode only), the 3-way gate
                below determines the outcome.  ``max_interim_count < 1`` is
                rejected by the config validator.
            interim_overflow_slack: Number of extra overflow slots allowed
                beyond ``max_interim_count`` before keep-pending kicks in.
                At 0 (default), cap_pending fires immediately when ``c >= N``.
                At slack > 0, the gate is:
                    c < N           → normal mint
                    N <= c < N+slack → overflow mint; result["overflow_slot"]=True
                    c >= N+slack    → cap_pending (keep sessions pending)
                Counted against PEFT-resident adapters; the slack is proven
                to fit VRAM at boot via ``required_working_set_bytes``.
            stamp: Override the computed sub-interval stamp (test injection).
            session_ids: The app layer's own authoritative list of
                successfully-extracted session ids for this batch
                (``extraction.completed_session_ids(session_buffer)``) —
                recorded verbatim in the ledger's extraction stage.  Session
                provenance can NOT be derived from the merged relations
                alone: a session yielding zero relations (or only
                attribute-typed facts, which never become edge
                ``Relation`` objects) would otherwise never appear in any
                relation's ``session_ids`` and would never retire.
                ``None`` (the default, kept for callers with no app-layer
                tracker — experiments, direct unit tests) falls back to the
                relation-derived set.

        Returns:
            Result dict with keys ``{"triples_extracted", "new_keys",
            "adapter_name", "mode", "venue", "error"}``.  ``mode`` is the
            outcome (``"trained"``, ``"simulated"``, ``"cap_pending"``,
            ``"aborted"``, or ``"noop"``) — ``"aborted"`` means training
            yielded (to inference, or a graceful shutdown) mid-bundle,
            distinct from a genuine ``"noop"`` (nothing new to encode);
            ``venue`` is the training medium (``"train"`` or ``"simulate"``).
        """
        triples_extracted = len(episodic_rels)

        # --- 1. Guard: no relations ---
        if not episodic_rels and not procedural_rels:
            return {
                "triples_extracted": 0,
                "new_keys": [],
                "adapter_name": None,
                "mode": "noop",
                "venue": mode,
                "error": None,
            }

        # --- 2. Tag speaker_id defaults ---
        self._tag_speaker_id_defaults(episodic_rels, speaker_id)
        self._tag_speaker_id_defaults(procedural_rels, speaker_id)

        # --- 3. Resolve stamp and target slot ---
        from paramem.memory.interim_adapter import (
            INTERIM_NAME_PREFIX,
        )
        from paramem.memory.interim_adapter import (
            current_interim_stamp as _cis,
        )

        if stamp is None:
            stamp = _cis(schedule)

        adapter_name = self._resolve_target_slot(stamp)

        # --- 4. 3-way gate (train mode only) ---
        # Count source: PEFT-resident adapters (what the VRAM ceiling constrains;
        # on-disk count and PEFT count measure different things and converge
        # only at tick boundaries).
        # Gate terms apply only when: train mode AND target slot is new.
        # Simulate has no PEFT slots so the count is meaningless; simulate
        # always falls through to stage/build below.
        existing_interim_count = len(
            [a for a in self.model.peft_config if a.startswith(INTERIM_NAME_PREFIX)]
        )
        _gate_active = mode != "simulate" and adapter_name not in self.model.peft_config
        is_overflow = False
        if _gate_active:
            c = existing_interim_count
            N = max_interim_count
            slack = interim_overflow_slack
            if c >= N + slack:
                # cap_pending: ring + overflow both exhausted — keep sessions
                # pending until the full fold drains the ring (lossless).
                logger.warning(
                    "run_consolidation_cycle: interim ring full (%d/%d+%d slots) — "
                    "keeping %d triples pending until next full fold",
                    c,
                    N,
                    slack,
                    len(episodic_rels),
                )
                cap_pending_summary = {
                    "triples_extracted": triples_extracted,
                    "new_keys": [],
                    "adapter_name": None,
                    "mode": "cap_pending",
                    "venue": mode,
                    "error": None,
                }
                with self._artifact_scope(interim_stamp=stamp):
                    on_cycle_end(cap_pending_summary)
                return cap_pending_summary
            elif c >= N:
                # overflow mint: ring is full but slack allows a later-stamped
                # overflow slot.  Fall through to the single stage/build call
                # below; tag the result so the caller can fire the
                # interim_cap_reached incident (only on a real "trained" mint).
                logger.warning(
                    "run_consolidation_cycle: interim ring full (%d/%d slots), "
                    "minting overflow slot %d/%d+%d — full fold is overdue",
                    c,
                    N,
                    c - N + 1,
                    N,
                    slack,
                )
                is_overflow = True
            # else: c < N — normal mint, fall through to stage/build below.

        # --- 5. Stage this event's shadow tree, then build/gate/write/publish it ---
        # source is picked from mode: "weights" for train, "disk" for simulate.
        # Map the caller's mode Literal to the FoldScope source axis without a mode== fork.
        _interim_source: "Literal['weights', 'disk']" = {"train": "weights", "simulate": "disk"}[
            mode
        ]
        _interim_scope = FoldScope(
            source=_interim_source,
            persist="interim_slot",
            normalize=False,  # normalization is full-fold only
            enrich=False,  # graph enrichment is full-fold only
        )
        # Every artifact the fold and its nested passes emit lands in this
        # cycle's debug root; a calibration run, when one is open, adds its own
        # root independently.
        with self._artifact_scope(interim_stamp=stamp):
            on_extraction_end(episodic_rels or [], procedural_rels or [])

            # Mint/refresh the interim PEFT slot BEFORE hydration (weights venue
            # only) so the venue's weight source below reads the current model.
            if _interim_scope.source == "weights":
                from paramem.memory.interim_adapter import create_interim_adapter
                from paramem.models.loader import ensure_adapter_matching

                if adapter_name not in self.model.peft_config:
                    create_interim_adapter(self.model, self.tier_adapters["episodic"], stamp)
                    logger.info("run_consolidation_cycle: created interim adapter %s", adapter_name)
                else:
                    ensure_adapter_matching(
                        self.model, self.tier_adapters["episodic"], adapter_name
                    )

            _recalled_entries = self._hydrate_store_for_fold(_interim_scope)

            # Working universe: this tick's own new slot is the sole primary
            # tier; the three main tiers and every sibling interim slot are
            # candidates (dedup-only, never absorbed — the interim topology
            # never absorbs its candidate tiers, see stage_event's own
            # full-topology-absorption contract).
            from paramem.memory.interim_adapter import interim_tiers_newest_first

            _sibling_interim_tiers = [
                t for t in interim_tiers_newest_first(self.store) if t != adapter_name
            ]
            _candidate_tiers = {t: t for t in [*self.tier_adapters, *_sibling_interim_tiers]}

            _session_ids = (
                sorted(session_ids)
                if session_ids is not None
                else sorted(
                    {
                        sid
                        for rel in (*pending.episodic, *pending.procedural)
                        for sid in (rel.session_ids or [])
                    }
                )
            )
            _pre_active = set(self.store.active_keys_in_tier(adapter_name))

            staged_event = self.stage_event(
                event="interim",
                venue=_interim_source,
                stamp=stamp,
                primary_tiers={adapter_name: adapter_name},
                recalled_entries=_recalled_entries,
                candidate_tiers=_candidate_tiers,
                episodic_rels=pending.episodic,
                procedural_rels=pending.procedural,
                session_ids=_session_ids,
                promote=False,
                normalize=False,
                enrich=False,
                resolve_contradictions=(self.config.refinement_contradiction == "on"),
            )

            if staged_event is None:
                # No ledger was ever written -- stage_event's own no-material
                # exit runs before anything lands on disk -- so there is no
                # event record for any later terminal to retire from.  The
                # app layer's own shared helper (_retire_extracted_sessions)
                # is what retires this pre-stage's completed sessions on this
                # outcome; "completed" is explicit False here rather than
                # implied by the key's absence, so a caller reading it never
                # mistakes this for a genuinely completed event.
                result = {
                    "triples_extracted": triples_extracted,
                    "new_keys": [],
                    "adapter_name": None,
                    "mode": "noop",
                    "venue": mode,
                    "error": None,
                    "completed": False,
                }
                on_cycle_end(result)
                return result

            # The extraction stage's own recorded session list and relation
            # counts — captured before run_build_and_publish, which disposes
            # the ledger once the tier goes live.  The honest source for what
            # the app-layer finalizer retires and reports: completed
            # extractions only, exactly what stage_event wrote before any
            # artifact landed.
            from paramem.training import stage_ledger as _sl

            _extraction_stage = _sl.extraction_entry(staged_event.ledger) or {}
            consumed_session_ids = list(_extraction_stage.get("sessions", []))
            consumed_episodic_rels = _extraction_stage.get("episodic_rels", 0)
            consumed_procedural_rels = _extraction_stage.get("procedural_rels", 0)

            build_summary = self.run_build_and_publish(staged_event)

            new_keys = sorted(set(self.store.active_keys_in_tier(adapter_name)) - _pre_active)
            _interim_mode_label = interim_outcome_label(build_summary, venue=_interim_scope.source)

            result = {
                "triples_extracted": triples_extracted,
                "new_keys": new_keys,
                "adapter_name": adapter_name,
                "mode": _interim_mode_label,
                "venue": mode,
                "error": None,
                "tiers_rebuilt": [adapter_name] if _interim_scope.source == "weights" else [],
                "consumed_session_ids": consumed_session_ids,
                "consumed_episodic_rels": consumed_episodic_rels,
                "consumed_procedural_rels": consumed_procedural_rels,
                "completed": build_summary["all_live"],
                "tier_bindings": build_summary["tier_bindings"],
            }
            on_cycle_end(result)

        # Only tag a real mint: an aborted overflow fold must not trigger
        # the interim_cap_reached incident on the app.py consumer side.
        if is_overflow and result.get("mode") == "trained":
            result["overflow_slot"] = True
        return result

    def _current_extraction_config(self) -> "ExtractionConfig":
        """Resolve the live :class:`ExtractionConfig` off the extraction pipeline.

        Handed to the graph tier (:class:`~paramem.training.graph_tier.GraphTierRefiner`
        and :func:`~paramem.training.graph_enrich.enrich_graph`) as a
        bound method rather than a resolved value, so the read happens only on
        the paths that actually consume the config — all of which sit past
        those passes' ``no_model``/``floor`` skips.

        This method touches ``self.extraction``, which :meth:`release` nulls
        alongside ``self.model``.  It is therefore only safe to CALL when the
        base model is live, and the tier's guards are what guarantee that: a
        released loop skips on ``model is None`` and never gets here.  Passing
        ``self.extraction.config`` eagerly instead would evaluate the read
        before any of those guards could run.
        """
        return self.extraction.config

    def _capture_pending_relations(self) -> "list[Relation]":
        """Snapshot current merger.graph edges AND node attributes into a list[Relation].

        The one implementation :meth:`take_pending_relations` (its ONE
        caller) wraps: called by the extraction boundary — the server
        layer's ``_extract_pending_sessions`` at its single return, or
        :meth:`train_adapters` for a caller that merged directly — before
        that boundary resets the graph, so the pending-session content
        survives the reset and re-enters the merge through *episodic_rels*
        / *procedural_rels*, ``stage_event``'s own pending-relations
        channel, via a :class:`PendingRelations` value the caller threads
        explicitly rather than a flag read off ``merger.graph`` state.

        An attribute-typed fact (``relation_type == "attribute"``) never
        becomes an edge — :meth:`~paramem.graph.merger.GraphMerger.merge`
        folds it onto the subject node's ``attributes`` dict instead, as a
        provenance-bearing record — so an edge-only walk would silently drop
        it at the reset this method exists to protect against.  A second
        pass over ``merger.graph.nodes`` recovers it, reading each record
        through :func:`~paramem.graph.merger.attribute_fact`, the one
        node-record -> fact projection also used by
        :meth:`_build_working_keyed_walk`, so the predicate/value/provenance
        shape the two surfaces produce never diverges.

        Returns an empty list when the graph is absent, or has no edges and
        no node attributes; both ``None`` and ``[]`` are valid no-ops for
        :meth:`stage_event`'s own ``episodic_rels or []`` handling.

        Returns:
            list[Relation]: Relation objects built from the current merger
                graph.  Each edge contributes exactly one :class:`Relation`
                with:

                - ``subject``/``object`` the endpoints' DISPLAY surfaces via
                  :func:`~paramem.graph.merger.node_display` — re-merging this
                  relation through :meth:`~paramem.graph.merger.GraphMerger.merge`
                  folds the subject through ``canonical()`` for node identity
                  (so a node-key subject would resolve to the same node
                  anyway), but only the display surface is safe to feed back
                  into the merger's first-seen-wins ``display_name`` write —
                  the folded node key would otherwise become the node's
                  display surface for the rest of its life;
                - ``predicate`` taken from the edge ``"predicate"`` attribute
                  (edges with an empty predicate are skipped);
                - ``relation_type`` validated against :data:`_VALID_RTYPES`,
                  falling back to :data:`_FALLBACK_RTYPE`;
                - ``speaker_id`` the edge's own ``"speaker_id"`` attribute —
                  every edge carries a non-empty one once it has crossed
                  :meth:`~paramem.graph.merger.GraphMerger.merge`'s
                  provenance rule, so no node fallback is read;
                - ``session_ids`` from the edge ``"sessions"`` attribute;
                - ``last_seen`` from the edge ``"last_seen"`` attribute (empty
                  string when absent).  Propagating the real ingest-time stamp
                  ensures pending relations carry genuine recency through the
                  ``merger.merge_relations`` call so a newer pending fact can
                  supersede a strictly-older dated registry-true rival.  Without
                  this field the captured relation would have ``last_seen=""``
                  (undated) and lose outright to the dated registry-true rival —
                  a dated candidate always outranks an undated one — suppressing
                  the intended supersession (COEXIST only applies when every
                  candidate is undated, which would not be the case here).
                - ``first_seen`` from the edge ``"first_seen"`` attribute (empty
                  string when absent) — symmetric carry so the re-merge's
                  ``min_nonempty`` window-start logic sees the real earliest
                  assertion instead of losing it to a synthetic fold sentinel.

                Each node attribute record contributes one further
                :class:`Relation`, via :func:`~paramem.graph.merger.attribute_fact`:

                - ``subject`` the node's DISPLAY surface, ``predicate`` via
                  :func:`~paramem.graph.relation_prep.attr_predicate`,
                  ``object`` the record's ``value``, ``relation_type="attribute"``;
                - ``speaker_id``/``first_seen``/``last_seen`` from the record
                  itself — a node attribute now carries its own provenance,
                  written by the merger's attribute gate;
                - ``session_ids`` from the node's ``"sessions"`` attribute.

                A pair already emitted by the edge walk is excluded from the
                attribute pass under canonical comparison (``canonical(subject)``,
                ``canonical(predicate)``) — the edge arm supplies display
                surfaces and the attribute arm now does too, but comparing
                canonically keeps the exclusion correct regardless of which
                surface either arm happens to carry.
        """
        import networkx as _nx

        _g = getattr(self.merger, "graph", None)
        if not isinstance(_g, _nx.MultiDiGraph):
            return []
        _result: list[Relation] = []
        for _er_subj, _er_obj, _er_data in _g.edges(data=True):
            _er_pred = _er_data.get("predicate", "")
            if not _er_pred:
                continue
            _er_rt_raw = _er_data.get("relation_type", _FALLBACK_RTYPE)
            _er_rt: str = _er_rt_raw if _er_rt_raw in _VALID_RTYPES else _FALLBACK_RTYPE
            _er_subj_display = node_display(_g.nodes.get(_er_subj, {}), _er_subj)
            _er_obj_display = node_display(_g.nodes.get(_er_obj, {}), _er_obj)
            _result.append(
                Relation(
                    subject=_er_subj_display,
                    predicate=_er_pred,
                    object=_er_obj_display,
                    relation_type=_er_rt,  # type: ignore[arg-type]
                    confidence=_er_data.get("confidence", 1.0),
                    speaker_id=_er_data["speaker_id"],
                    session_ids=list(_er_data.get("sessions", [])),
                    last_seen=_er_data.get("last_seen", ""),
                    first_seen=_er_data.get("first_seen", ""),
                )
            )

        # Attribute-typed facts: never edges, so a second pass over node
        # attributes is the only way to recover them before the reset.
        _exclude_pairs = {(canonical(r.subject), canonical(r.predicate)) for r in _result}
        for _node_id, _node_data in _g.nodes(data=True):
            _node_attrs = _node_data.get("attributes") or {}
            if not _node_attrs:
                continue
            for _attr_key, _record in _node_attrs.items():
                _fact = attribute_fact(_node_data, _node_id, _attr_key, _record)
                if (canonical(_fact["subject"]), canonical(_fact["predicate"])) in _exclude_pairs:
                    continue
                _result.append(
                    Relation(
                        subject=_fact["subject"],
                        predicate=_fact["predicate"],
                        object=_fact["object"],
                        relation_type="attribute",
                        confidence=1.0,
                        speaker_id=_fact["speaker_id"],
                        session_ids=list(_node_data.get("sessions", [])),
                        last_seen=_fact["last_seen"],
                        first_seen=_fact["first_seen"],
                    )
                )
        return _result

    def _split_pending_relations(
        self, pending: "list[Relation]"
    ) -> "tuple[list[Relation], list[Relation]]":
        """Partition captured pending relations into ``(episodic, procedural)``.

        Delegates the classification to
        :func:`~paramem.graph.relation_prep.filter_procedural_relations` — the
        same primary (``relation_type == "preference"``) plus secondary
        (predicate set) gate :meth:`_entries_from_graph` uses for freshly
        extracted relations — over a positional dict view of *pending*, so the
        predicate set is read from exactly one place. Gated on whether a
        procedural adapter is configured (mirrors
        :func:`~paramem.graph.relation_prep.partition_relations`'s own
        ``procedural_enabled`` gate): with none configured, everything stays
        episodic so preferences are never lost.

        Args:
            pending: The combined list :meth:`_capture_pending_relations`
                returned.

        Returns:
            ``(episodic, procedural)`` — the same :class:`Relation` objects
            from *pending*, partitioned with relative order preserved.
        """
        if "procedural" not in self.tier_adapters or not pending:
            return list(pending), []
        from paramem.graph.relation_prep import filter_procedural_relations

        views = [
            {"relation_type": r.relation_type, "predicate": r.predicate, "_idx": i}
            for i, r in enumerate(pending)
        ]
        procedural_idx = {v["_idx"] for v in filter_procedural_relations(views)}
        episodic = [r for i, r in enumerate(pending) if i not in procedural_idx]
        procedural = [r for i, r in enumerate(pending) if i in procedural_idx]
        return episodic, procedural

    def take_pending_relations(self) -> "PendingRelations":
        """End the extraction graph's lifetime: capture the merged product
        and reset the keying surface.

        The ONE door out of the extraction accumulation, called exactly
        once per batch by the boundary that opened it — the server layer's
        ``_extract_pending_sessions`` at its single return, for a caller
        whose sessions merged through the server's own extraction stage;
        :meth:`train_adapters` for a caller that merged directly via
        :meth:`extract_session`.  Nothing downstream reads
        ``self.merger.graph`` for extraction content again after this
        returns — ``stage_event`` opens its own, separate fold lifetime
        with its own ``reset_graph()``, which finds this surface already
        empty and is a no-op there.

        Returns:
            PendingRelations: The captured product, already split into
                ``episodic`` / ``procedural`` (see :meth:`_split_pending_relations`).
                Empty lists on a batch that merged nothing.
        """
        captured = self._capture_pending_relations()
        episodic, procedural = self._split_pending_relations(captured)
        self.merger.reset_graph()
        return PendingRelations(episodic=episodic, procedural=procedural)

    # ------------------------------------------------------------------
    # Unified persist dispatch — one venue fork for graph-json simulate,
    # interim-slot, and main-tiers full-fold persistence.
    # ------------------------------------------------------------------

    @staticmethod
    def _venue_from_scope(scope: "FoldScope") -> "Literal['train', 'simulate']":
        """Derive the venue string every mode-keyed collaborator expects from *scope*.

        Two consumers:
        :func:`~paramem.memory.persistence.commit_tier_slot` (``mode=``) and
        :func:`~paramem.memory.source.build_memory_source` (``mode=``).  Both
        take the same ``"train"`` / ``"simulate"`` vocabulary, so the fold
        translates its structural venue exactly once, here.

        This is a derivation, not a mode fork: the result flows from the
        structural ``scope.source`` enum — no ``mode == "train"`` comparison
        is introduced here, so the mode-fork guard is not triggered.

        Args:
            scope: The immutable :class:`FoldScope` for the current fold.

        Returns:
            ``"train"`` when weights are being written (``scope.source ==
            "weights"``); ``"simulate"`` otherwise.
        """
        return "train" if scope.source == "weights" else "simulate"

    def _hydrate_store_for_fold(self, scope: "FoldScope") -> "dict[str, dict[str, dict]]":
        """Reconstruct every live key's entry from the venue, into fold-local state.

        The fold is decoupled from serving: this reconstructs the tier ->
        key -> entry map :meth:`_recall_working_tiers` seeds
        ``WorkingTier.entries`` from directly — never through the store's
        entry mirror.  The fold neither reads nor writes
        :attr:`~paramem.memory.store.MemoryStore._entries` in either
        direction; only the tier registries are read (:meth:`tiers_with_registry`,
        :meth:`active_keys_in_tier`), which are a separate structure from
        the mirror and stay live-store reads by design (the registry is
        durable truth this event's own working copy will diverge from).

        Every active key of every registered tier is probed against the
        venue's own :class:`~paramem.memory.source.MemorySource` — adapter
        weights in the train venue, the per-tier ``graph.json`` in the
        simulate venue — in one grouped call per venue.  Both source
        implementations gate their own results against the tier's stored
        SimHash fingerprint before returning (no second gate runs here — a
        second invocation would be a redundant transformation on
        already-admitted content); a key whose result is absent, a gate
        failure (``failure_reason``), or missing one of the four content
        fields is dropped; when this reconstruction is done, the fold's
        gap scan raises :class:`ActiveKeyHydrationFailure` naming every
        dropped key and the venue, so the fold aborts before anything is
        staged or trained rather than silently losing an active key from
        its working universe — see that class's docstring for which
        retirement paths this does and does not affect.  The per-fold
        source probe is the reconstruction step; its cost is noise next to
        the fold's own multi-epoch training.

        **BASE-MODEL HOLDER** — the :class:`WeightMemorySource` built here
        captures the base model.  It is a frame-local, built from ``self.model``
        at call time (which the fold rebinds around adapter creation, so it must
        never be cached on ``self``) and dropped when this method returns, the
        same no-frame-retention pattern ``app._preload_memory_store`` uses.

        Args:
            scope: The immutable :class:`FoldScope` for the current fold.  Its
                ``source`` selects the venue via :meth:`_venue_from_scope`.

        Returns:
            ``{tier: {key: content_only_entry}}`` for every tier with at
            least one active key — the fold-local reconstruction result,
            threaded into :meth:`stage_event` and from there into
            :meth:`_recall_working_tiers`.  Empty when no tier has an
            active key (nothing to reconstruct).

        A source probe that raises is NOT swallowed: proceeding into the fold
        with an unknown-partial reconstruction is the data loss this method
        exists to prevent, so the exception aborts the fold before anything
        is staged or trained.
        """
        from paramem.memory.entry import content_only_entry, is_admissible_probe_result
        from paramem.memory.source import build_memory_source

        venue = self._venue_from_scope(scope)
        keys_by_tier = {
            tier: keys
            for tier in self.store.tiers_with_registry()
            if (keys := self.store.active_keys_in_tier(tier))
        }
        if not keys_by_tier:
            return {}

        source = build_memory_source(
            mode=venue,
            adapter_dir=self.output_dir,
            batch_size=self.training_config.recall_probe_batch_size,
            model=self.model,
            tokenizer=self.tokenizer,
        )
        source_results = source.probe(keys_by_tier)
        source = None  # BASE-MODEL HOLDER frame-local — drop before returning

        recalled: "dict[str, dict[str, dict]]" = {}
        dropped: list[str] = []
        for tier, keys in keys_by_tier.items():
            for key in keys:
                result = source_results.get(key)
                if not is_admissible_probe_result(result):
                    dropped.append(key)
                    continue
                recalled.setdefault(tier, {})[key] = content_only_entry(result)

        if dropped:
            raise ActiveKeyHydrationFailure(dropped_keys=dropped, venue=venue)

        return recalled

    def consolidate(
        self,
        *,
        mode: str,
        event: "Literal['full', 'reconcile']" = "full",
        pending: "PendingRelations | None" = None,
        trainer=None,
        router=None,
        session_ids: "list[str] | None" = None,
    ) -> dict:
        """Run the full consolidation fold — the single public fold entry.

        The fold does what it is told.  Whether there is anything to consolidate at
        all is decided by the caller (the server's dispatch layer): this method has
        no content gate, no notion of who asked for the fold, and no way to bypass
        the recall gate or the caller-side content gate.

        Both venues run the SAME stage spine over the SAME input — the
        :class:`~paramem.memory.store.MemoryStore`, whose main-tier and
        interim-slot registries are hydrated at boot and after every cycle.
        Stage this event's shadow tree (:meth:`stage_event` — recall,
        refine, promote, build entries, write the extraction stage) then
        build/gate/write each tier and take the bundle live
        (:meth:`run_build_and_publish`) is one code path for both venues.
        *mode* selects only:

        - **train** (``source="weights"``): backs the main tiers up, retrains
          ``episodic`` / ``semantic`` / ``procedural``, probes each tier's
          staged weights for recall misses before promotion, and persists
          the weights.  Requires the caller to already hold
          ``_gpu_thread_lock`` (submit via ``BackgroundTrainer.submit()``);
          the entry guard below raises when it does not.  On a failed
          per-tier recall verdict, ``tier_backup_scope`` restores only the
          in-VRAM state of the one tier that was training — nothing on disk
          or live-serving has changed — and the fold aborts (see
          :class:`RecallGateRejected`).
        - **simulate** (``source="disk"``): skips those weight-only blocks and
          persists each main tier's ``graph.json`` into a fresh written slot
          under ``<adapter_dir>/<tier>/`` (never a tier-root path — nothing
          writes one), the BOUND slot :class:`~paramem.memory.source.DiskMemorySource`
          reads back.  No model, no GPU.

        Both venues route through :meth:`_stage_and_publish_full_event`; the
        ``mode`` string is translated into a :class:`FoldScope` here and never
        travels further (the mode-fork guard requires downstream dispatch on
        ``scope.source`` / ``scope.persist``).

        A reconcile (``/reconsolidate``) IS a full consolidation whose input
        excludes pending sessions: one fold topology throughout — the interim
        ring is always recalled, always absorbed into the main tiers, and
        always reaped, exactly as any full fold; warm start is uniform, with
        no cold-start arm.  *event* exists only to name the door in the
        ledger and in reporting; it changes no fold behaviour here beyond the
        recorded label — the caller is what keeps sessions pending for a
        reconcile, via *pending*.

        Args:
            mode: ``"train"`` or ``"simulate"``.  Required — ``ConsolidationConfig``
                carries no ``mode`` field; the server passes
                ``config.consolidation.mode``.
            event: ``"full"`` (the default) or ``"reconcile"`` — the door
                name recorded in the ledger head and read back verbatim by
                reporting (``paramem.server.app``'s pending-action name).
                Both run the identical fold: every interim slot is always a
                read-only candidate this event absorbs whole (see
                :meth:`_stage_and_publish_full_event`).
            pending: The caller's own :meth:`take_pending_relations` take
                (train only), when the pending-session content already
                extracted by the caller's own extraction pre-stage should
                train into the main tiers.  The caller derives whether to
                pass one from its schedule config (``max_interim_count == 0
                and mode != "simulate"``); a reconcile event never passes
                one — pending sessions stay pending.  ``None`` (default)
                means this event folds no pending-session content.
            trainer: :class:`~paramem.server.background_trainer.BackgroundTrainer`
                holding the GPU lock (train only).  Its ``_set_is_training`` flag is
                narrowed to ``False`` around the staging pass's CPU-only phase so a
                concurrent inference turn's ``abort_for_inference`` returns fast
                instead of waiting out a quiesce timeout for a training step that
                is not running; ``None`` skips the narrowing.
            router: Router instance whose ``reload()`` is called as a
                ``publish_bundle`` go-live step (both venues).  ``None`` is
                safe — skipped.
            session_ids: The app layer's own authoritative list of
                successfully-extracted session ids for this *pending* batch
                (``extraction.completed_session_ids(session_buffer)``)
                — recorded verbatim in the ledger's extraction stage.
                Ignored when *pending* is ``None`` (nothing this call
                retires).  Required when *pending* is not ``None`` — a
                missing list raises ``TypeError``, never a silent
                relation-derived guess.

        Returns:
            The full-fold result dict (see :meth:`_stage_and_publish_full_event`)
            — one schema for both venues and every terminal return.

        Raises:
            ValueError: When *pending* is supplied on the simulate venue.
                The simulate fold has no weight venue to train pending sessions into,
                so it would discard the content; callers derive whether to pass
                *pending* from ``max_interim_count == 0 and mode != "simulate"``,
                which cannot produce that pairing.  The guard exists so a
                future caller that gets the derivation wrong fails loudly instead
                of silently ingesting nothing.
            RuntimeError: When ``mode="train"`` is called without the GPU lock held.
        """
        if mode == "simulate" and pending is not None:
            raise ValueError(
                "consolidate(mode='simulate') cannot consume pending sessions: the "
                "simulate venue writes graph.json and trains nothing. Pass "
                "pending=None, or run the train venue."
            )

        if mode == "simulate":
            # Every artifact the fold and its nested passes emit lands in this
            # cycle's debug root; a calibration run, when one is open, adds its
            # own root independently.
            # promote is ON: it is a pure store operation, so it belongs to
            # this venue exactly as much as to the weights venue.
            with self._artifact_scope():
                return self._stage_and_publish_full_event(
                    source="disk",
                    event=event,
                    pending=None,
                    router=router,
                )

        from paramem.server.gpu_lock import gpu_lock_is_held

        # --- Entry guard: the caller must hold the GPU lock (leak-safe) ---
        if not gpu_lock_is_held():
            raise RuntimeError(
                "consolidate(mode='train') requires the caller to hold "
                "the GPU lock (submit via BackgroundTrainer.submit())"
            )

        # Every artifact the fold and its nested passes emit lands in this
        # cycle's debug root; a calibration run, when one is open, adds its own
        # root independently.
        with self._artifact_scope():
            return self._stage_and_publish_full_event(
                source="weights",
                event=event,
                pending=pending,
                router=router,
                trainer=trainer,
                session_ids=session_ids,
            )

    def _stage_and_publish_full_event(
        self,
        *,
        source: "Literal['weights', 'disk']",
        event: "Literal['full', 'reconcile']",
        pending: "PendingRelations | None",
        router,
        trainer=None,
        session_ids: "list[str] | None" = None,
    ) -> dict:
        """Stage one full-consolidation event and take it live — the shared
        body of :meth:`consolidate`'s two venue branches.

        One fold topology for both doors: the three main tiers are always
        primary, and every interim slot is always a read-only candidate this
        event absorbs whole and reaps — a reconcile (``/reconsolidate``) runs
        the identical spine, differing from an ordinary full fold only in
        *event*'s recorded label and in the caller leaving sessions pending
        (``pending=None``).  No main tier's adapter is
        deleted or recreated here except when the operator changed its LoRA
        shape: :func:`~paramem.models.loader.ensure_adapter_matching`
        recreates the resident adapter cold in that one case, before
        :func:`~paramem.models.loader.tier_backup_scope`, because
        shape-mismatched weights cannot be kept — otherwise a live tier's
        weights change only at the go-live mount.  Every tier's transient
        staging slot warm-starts uniformly (see
        :func:`~paramem.training.trainer.train_adapter`'s ``warm_start``
        table) — there is no cold-start arm for either door.

        Args:
            source: ``"weights"`` (train) or ``"disk"`` (simulate) — this
                event's venue.
            event: ``"full"`` or ``"reconcile"`` — the door name recorded in
                the ledger head; changes no fold behaviour here.
            pending: The caller's own :meth:`take_pending_relations` take to
                fold into this event, or ``None`` when this event folds no
                pending-session content.
            router: The live ``QueryRouter`` to reload once per bundle;
                ``None`` skips the reload.
            session_ids: The app layer's own authoritative completed-session
                list, forwarded verbatim from :meth:`consolidate`.  Required
                when *pending* is not ``None`` — a missing list raises
                ``TypeError`` at the sort call, never a silent
                relation-derived guess.  Unused (may be ``None``) otherwise.

        Returns:
            A result dict carrying the fields meaningful under this design
            (``tiers_rebuilt``) plus fields kept for callers that read them
            positionally.
        """
        scope = FoldScope(
            source=source,
            persist="main_tiers",
            normalize=(self.config.refinement_normalization == "on"),
            enrich=(self.config.refinement_enrichment == "on" and self.cloud_enabled),
        )

        _recalled_entries = self._hydrate_store_for_fold(scope)

        primary_tiers = {t: t for t in self.tier_adapters}

        # Every full-topology event absorbs the interim ring whole.
        from paramem.memory.interim_adapter import interim_tiers_newest_first

        _ring = interim_tiers_newest_first(self.store)
        candidate_tiers: "dict[str, str] | None" = {t: t for t in _ring}

        _staged_session_ids: "list[str] | None" = None
        if pending is not None:
            _staged_session_ids = sorted(session_ids)

        from paramem.memory.interim_adapter import current_full_consolidation_stamp

        _period = getattr(self, "full_consolidation_period_string", "")
        stamp = current_full_consolidation_stamp(_period)

        # Mark this CPU-only staging phase as "not training" so a /chat
        # arriving mid-fold gets abort_for_inference's fast no-op instead of
        # waiting out its full quiesce timeout for a training step that is
        # not running (BackgroundTrainer._set_is_training's own docstring).
        # Restored before the per-tier train/gate/write loop inside
        # run_build_and_publish, which does touch the GPU.
        if trainer is not None:
            trainer._set_is_training(False)
        _pending_episodic = pending.episodic if pending is not None else []
        _pending_procedural = pending.procedural if pending is not None else []
        try:
            staged_event = self.stage_event(
                event=event,
                venue=source,
                stamp=stamp,
                primary_tiers=primary_tiers,
                recalled_entries=_recalled_entries,
                candidate_tiers=candidate_tiers,
                episodic_rels=_pending_episodic,
                procedural_rels=_pending_procedural,
                session_ids=_staged_session_ids,
                promote=True,
                normalize=scope.normalize,
                enrich=scope.enrich,
                resolve_contradictions=(self.config.refinement_contradiction == "on"),
            )
        finally:
            if trainer is not None:
                trainer._set_is_training(True)

        if staged_event is None:
            return {
                "tiers_rebuilt": [],
                "consumed_session_ids": [],
                "completed": False,
                "aborted": False,
                "tier_bindings": {},
            }

        # The extraction stage's own recorded session list — the honest
        # source for what the app-layer terminal retires: completed
        # extractions only, exactly what stage_event wrote before any
        # artifact landed.  Disposal itself is the caller's own act, at its
        # terminal, once "completed" below confirms every tier went live —
        # never performed inside run_build_and_publish (retire-then-dispose
        # ordering; see that method's own docstring).
        from paramem.training import stage_ledger as _sl

        _extraction_stage = _sl.extraction_entry(staged_event.ledger) or {}
        consumed_session_ids = list(_extraction_stage.get("sessions", []))

        build_summary = self.run_build_and_publish(staged_event, router=router)

        return {
            "tiers_rebuilt": build_summary["published_tiers"],
            "consumed_session_ids": consumed_session_ids,
            "completed": build_summary["all_live"],
            "aborted": build_summary["aborted"],
            "tier_bindings": build_summary["tier_bindings"],
        }

    def build_tier_refiner(self, merger) -> "graph_tier.GraphTierRefiner":
        """Construct a graph-tier refiner over *merger* with this loop's config.

        THE construction site. The consolidation cycle passes its own
        ``self.merger``; a calibration run passes a throwaway merger holding
        the relations the operator injected, so the pass it exercises is the
        production pass — same engine selection, same survivor rule — rather
        than a second implementation of it.

        Args:
            merger: The single mutation target for both refinement passes.

        Returns:
            A refiner bound to *merger* and this loop's model handle.
        """
        return graph_tier.GraphTierRefiner(
            merger,
            model=self.model,
            tokenizer=self.tokenizer,
            extraction_config_provider=self._current_extraction_config,
            cloud_enabled=self.cloud_enabled,
            neighborhood_hops=self.graph_enrichment_neighborhood_hops,
            max_entities_per_pass=self.graph_enrichment_max_entities_per_pass,
            prompts_dir=self.prompts_dir,
            gc_disable=self._disable_gradient_checkpointing,
            gc_enable=self._enable_gradient_checkpointing,
        )

    #: Per-``aborted_reason`` incident summaries for
    #: :meth:`_record_enrichment_incident`. The key derives the incident's
    #: own ``key`` (``f"graph_enrich_{aborted_reason}"``) — no rename of
    #: the existing ``graph_enrich_vram`` key, so no persisted-incident
    #: migration.
    _ENRICHMENT_ABORT_SUMMARIES = {
        "vram": (
            "Graph-tier cloud enrichment degraded (merged graph, full fold "
            "only) — VRAM exhausted; kept already-merged chunks"
        ),
        "tagger": (
            "Graph-tier cloud enrichment degraded (merged graph, full fold "
            "only) — span tagger unavailable; kept already-merged chunks"
        ),
    }

    def _record_enrichment_incident(self, result: "graph_tier.RefineResult") -> None:
        """Surface a graph-tier enrichment degrade as an operator-visible
        incident — the SAME ``record_incident`` surface ``extract_session``'s
        ``cloud_enrichment_degraded`` path uses.

        ``result.enrichment`` is the raw diagnostics dict
        :func:`~paramem.training.graph_enrich.enrich_graph` returns.
        ``aborted_reason`` names which degrade stopped the chunk loop
        early — ``"vram"`` (:class:`~paramem.utils.vram_guard.VramExhausted`)
        or ``"tagger"`` (the span tagger unavailable, or its model call
        raised) — while keeping whatever the pass already merged rather
        than aborting the fold. Each reason records under its own key
        (``graph_enrich_vram`` / ``graph_enrich_tagger``) so the two
        degrades are distinguishable in the incident store. Severity
        ``"warning"`` (the fold succeeds regardless): enrichment
        self-heals at the next FULL fold, since the pass runs over the
        cumulative graph every full fold (never at an intervening interim
        cycle — full-fold only), so there is nothing to retry here.

        A pass that ran to completion (``aborted_reason is None``)
        resolves the WHOLE ``enrichment_degraded`` family by type — the
        same by-type sweep :meth:`_arbitrate_one_enrichment_signal`'s
        cloud-disabled arm already uses for this key, since a completed
        pass can never happen while cloud egress is refused.
        ``result.enrichment is None`` means the pass never ran (enrichment
        off, or an interim scope), which is not evidence of recovery and
        must not clear a standing incident — a no-op in that case.
        """
        if result.enrichment is None or self._incidents_state_dir is None:
            return

        from paramem.server.incidents import record_incident, resolve_incidents_by_type

        aborted_reason = result.enrichment.get("aborted_reason")
        if aborted_reason is not None:
            record_incident(
                self._incidents_state_dir,
                type="enrichment_degraded",
                key=f"graph_enrich_{aborted_reason}",
                severity="warning",
                summary=self._ENRICHMENT_ABORT_SUMMARIES[aborted_reason],
                detail={
                    "type": "enrichment_degraded",
                    "chunks": result.enrichment.get("chunks", 0),
                    "at": datetime.now(timezone.utc).isoformat(),
                },
            )
        else:
            resolve_incidents_by_type(self._incidents_state_dir, "enrichment_degraded")

    #: Ledger reasons whose collapse is an independent sighting of the fact and
    #: may therefore EARN a reinforcement (subject to the store's temporal-order
    #: check).  A predicate-synonym collapse is absent deliberately: it rewrites
    #: how a fact is spelled, it does not observe the fact again.
    _REOBSERVED_REASONS = frozenset({"dedup"})

    def _clean_stale_staging_dir(self) -> None:
        """Remove stale on-disk staging checkpoints left by a prior crash.

        ``output_dir/in_training`` is HF-Trainer scratch, unrelated to the
        PEFT slot lifecycle — filesystem-level debris from a crash-resume
        attempt that never completed. Called once at construction. Tier
        creation happens once, before this loop is ever constructed, via
        ``load_base_model`` / ``paramem.models.loader.ensure_resident_tiers``.
        """
        import shutil

        stale_dir = Path(self.output_dir) / "in_training"
        if stale_dir.exists():
            logger.info("Cleaning stale in_training checkpoints at %s", stale_dir)
            shutil.rmtree(stale_dir)

    def _disable_gradient_checkpointing(self) -> None:
        """Disable gradient checkpointing for generation."""
        self.model.gradient_checkpointing_disable()

    def _enable_gradient_checkpointing(self) -> None:
        """Re-enable gradient checkpointing if configured."""
        if self.training_config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

    def _maybe_make_recall_callback(
        self,
        entries: list[dict],
        *,
        adapter_name: str,
        output_dir,
        phase_name: str,
        num_epochs: int,
    ):
        """Construct a RecallEarlyStopCallback when configured.

        Returns ``(None, None)`` when ``training_config.recall_early_stopping``
        is False or when the entries list is empty (probing an empty set is
        a no-op).  Returns ``(callback, state)`` otherwise, where ``state``
        is the ``_EarlyStopState`` shared with the callback; the callback's
        only responsibility is the stop signal (``state.stop_epoch`` and the
        first/stable-perfect epoch markers) — the per-key recall verdict on
        the FINAL trained weights is the caller's own staged-weights probe
        (:meth:`_probe_recall`) after training returns, never read from this
        state.

        The probe target is the unmodified entries list — the same per-tier
        full-replay set that ``format_entry_training`` consumes.  This
        is the convergence gate — only safe if the caller passes the FULL
        active-key set for ``adapter_name``, not an incremental delta.

        Production-reachable callers (must pass full per-tier active set):
          - ConsolidationLoop._train_tier_adapter — the single funnel for
            every production training path (episodic/interim, the full
            fold, and
            active_store_migration._migrate_tier_simulate_to_train, which
            is routed through this funnel rather than wiring its own
            callback). None of those three call this helper directly —
            they all reach it transitively via _train_tier_adapter.

        A new production-reachable caller MUST call this helper; the
        AST structural test in tests/test_consolidation_recall_early_stop.py
        enforces the contract and will fail at PR-CI if violated.

        Args:
            entries: The full per-tier active entries (the training target).
            adapter_name: The adapter slot being trained (matches the
                ``adapter_name`` arg passed to ``train_adapter``).
            output_dir: HF Trainer ``output_dir`` for this call.
                ``progress.json`` and ``epoch_log.json`` are written
                alongside (parent of HF's ``checkpoint-N/`` tree).
            phase_name: Label for ``progress.json`` ("phase4-episodic",
                "interim-episodic-tickXY", "consolidate-episodic",
                "migrate-episodic", etc.).
            num_epochs: The ACTUAL epoch count the trainer will run for this
                call — the callback's forced final-epoch probe fires at this
                epoch.  Required: the sole caller (_train_tier_adapter)
                always has this value on hand (the derived budget from
                ``paramem.utils.config.budget_for``), so there is no
                well-defined fallback to resolve to.

        Returns:
            ``(RecallEarlyStopCallback, _EarlyStopState)`` when configured and
            entries is non-empty; ``(None, None)`` otherwise.
        """
        if not self.training_config.recall_early_stopping:
            return None, None
        if not entries:
            return None, None
        from pathlib import Path

        from paramem.training.early_stop import (
            EarlyStopPolicy,
            RecallEarlyStopCallback,
            _EarlyStopState,
        )

        output_dir = Path(output_dir)
        # probe_from_epoch is pinned to the signal floor: a single probe runs
        # 137-ish ``generate(max_new_tokens=128)`` calls (paramem/training/
        # recall_eval.py::probe_entries), which is ~12-40× the per-epoch
        # training cost.  Probes below the floor cannot influence
        # ``control.should_training_stop`` (see early_stop.py:494-499 — the
        # signal-trigger ANDs ``epoch >= signal_from_epoch`` with the window
        # check) and the only artifacts they produce (epoch_log.json,
        # stable_perfect_epoch) have no production consumer.  Aligning the
        # probe start with the signal floor eliminates that wasted compute;
        # the operator-tunable knob is ``recall_signal_from_epoch`` in
        # server.yaml.
        floor = self.training_config.early_stopping_floor
        policy = EarlyStopPolicy(
            probe_from_epoch=floor,
            signal_from_epoch=floor,
            window=self.training_config.recall_window,
            probe_every_n_epochs=self.training_config.recall_probe_every_n_epochs,
        )

        from paramem.memory.entry import build_registry as _build_registry
        from paramem.training.recall_eval import evaluate_indexed_recall

        _batch = self.training_config.recall_probe_batch_size
        if _batch > 1:
            import functools

            _eval_fn = functools.partial(evaluate_indexed_recall, batch_size=_batch)
        else:
            _eval_fn = evaluate_indexed_recall  # bare reference; preserves patchability

        state = _EarlyStopState()
        callback = RecallEarlyStopCallback(
            model=self.model,
            tokenizer=self.tokenizer,
            target_keyed=entries,
            target_registry=_build_registry(entries),
            adapter_name=adapter_name,
            policy=policy,
            state_out=state,
            progress_path=output_dir / "progress.json",
            epoch_log_path=output_dir / "epoch_log.json",
            first_perfect_log_path=None,  # production has no per-key log
            phase_name=phase_name,
            num_epochs=num_epochs,
            pause_file=None,  # production pause via gpu_lock_sync, not file
            eval_fn=_eval_fn,
        )
        return callback, state

    def _train_tier_adapter(
        self,
        entries: "list[dict]",
        *,
        adapter_name: str,
        adapter_config,
        training_config,
        output_dir,
        run_name: str,
        phase_name: str,
        retain_scratch_until_external_commit: bool = False,
    ):
        """Format → derive budget → dataset → recall callback → train_adapter for one tier.

        Returns ``(metrics, recall_state)``.  Returns ``(None, None)`` when
        there are no training examples (empty entries list).

        This is the ONLY shared training-invocation site.  Abort handling,
        recall-verdict application, and persistence stay at the call sites
        (scope-specific).

        The per-fold training budget (epoch count, gradient-accumulation
        steps, LR-decay steps) is derived here from ``len(entries)`` via
        ``paramem.utils.config.budget_for`` and applied to the incoming
        ``training_config`` via ``dataclasses.replace`` — every production
        caller (interim, the full fold, and
        ``active_store_migration._migrate_tier_simulate_to_train``) inherits
        the SAME derivation with no special case; there is no off switch
        (the derivation is the unconditional standard mechanism, validated
        via Test 20 -- see ``benchmarking.md``).

        The ``from paramem.training.trainer import train_adapter`` import is
        kept INSIDE this method so tests can patch
        ``paramem.training.trainer.train_adapter`` and intercept calls
        at this site.

        Args:
            entries: The full per-tier active entries (key/subject/predicate/
                object dicts).
            adapter_name: The adapter slot being trained.
            adapter_config: PEFT ``AdapterConfig`` for this tier.
            training_config: ``TrainingConfig`` for this call. The derived
                epoch/accum/lr-decay values REPLACE this config's fields
                before training (see above); the caller's own copy is not
                mutated (``dataclasses.replace`` returns a new instance).
            output_dir: HF Trainer ``output_dir``; also used by the recall
                callback for ``progress.json`` / ``epoch_log.json``.
            run_name: W&B / HF Trainer run name.
            phase_name: Label for the recall callback's ``progress.json``
                (e.g. ``"interim-episodic-tick42"``, ``"consolidate-semantic"``).
            retain_scratch_until_external_commit: Forwarded verbatim to
                :func:`paramem.training.trainer.train_adapter`.  When ``True``,
                the success path skips ``_clean_scratch`` / ``staging_resume.json``
                deletion so the durable ``checkpoint-N`` directory survives until
                the fold's own external ``commit_tier_slot`` call.  Default
                ``False`` preserves clean-on-success for all other callers (BG
                trainer, replay, migration, interim).

        Returns:
            ``(metrics_dict, recall_state)`` on success; ``(None, None)`` if
            ``entries`` yields no training examples.  ``metrics_dict``
            carries ``"init"`` (``"warm" | "donor" | "cold"``, the staging
            slot's starting-weights outcome — see
            :func:`~paramem.training.trainer.train_adapter`), ``"accum"``
            and ``"epochs"`` (this call's derived training budget) for the
            caller's telemetry record.

        Donor resolution: unconditional (no feature flag; validated via Test
        20 -- see ``benchmarking.md``). This method is reachable ONLY from
        the weights venue (every call site sits inside its enclosing ``if
        scope.source == "weights":`` branch — see ``consolidation.py``'s own
        ``_train_tier_adapter`` call site and
        ``active_store_migration._migrate_tier_simulate_to_train``, plus
        :func:`~paramem.training.donor.build_donor`'s own funnel call
        (training the donor's transient build slot itself, gated out of
        recursive resolution below) — all routed through this one funnel),
        so the disk/simulate venue never resolves a donor. When
        *adapter_name* is not the donor's own transient build slot
        (``DONOR_BUILD_ADAPTER_NAME`` — excluding it here is what stops
        :func:`~paramem.training.donor.build_donor`'s own funnel call from
        recursively re-triggering donor resolution on the adapter it is
        training), the donor checkpoint is resolved via
        :meth:`_resolve_donor_checkpoint` and handed to
        :func:`~paramem.training.trainer.train_adapter` as
        ``donor_checkpoint_dir`` — that call, not this one, decides whether
        the donor actually applies (only when the tier has no prior trained
        weights) and performs the copy into the transient staging slot. This
        method never writes any adapter's weights itself.
        """
        from paramem.training.trainer import train_adapter

        examples = format_entry_training(
            entries, self.tokenizer, max_length=training_config.max_seq_length
        )
        if not examples:
            return None, None

        donor_checkpoint_dir = None
        if adapter_name != DONOR_BUILD_ADAPTER_NAME:
            donor_checkpoint_dir = self._resolve_donor_checkpoint(adapter_name, adapter_config)

        derived_epochs, derived_accum, derived_lr_decay_steps = budget_for(len(entries))
        training_config = replace(
            training_config,
            num_epochs=derived_epochs,
            gradient_accumulation_steps=derived_accum,
            lr_decay_steps=derived_lr_decay_steps,
        )
        dataset = self._indexed_dataset(examples)
        self._enable_gradient_checkpointing()
        recall_cb, recall_state = self._maybe_make_recall_callback(
            entries=entries,
            adapter_name=adapter_name,
            output_dir=output_dir,
            phase_name=phase_name,
            num_epochs=derived_epochs,
        )
        metrics = train_adapter(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            adapter_name=adapter_name,
            training_config=training_config,
            adapter_config=adapter_config,
            wandb_config=self.wandb_config,
            output_dir=output_dir,
            run_name=run_name,
            thermal_policy=self._thermal_policy,
            hooks=self._build_training_hooks(),
            callbacks_extra=[recall_cb] if recall_cb is not None else None,
            retain_scratch_until_external_commit=retain_scratch_until_external_commit,
            donor_checkpoint_dir=donor_checkpoint_dir,
        )
        metrics["accum"] = derived_accum
        metrics["epochs"] = derived_epochs
        return metrics, recall_state

    @property
    def donor_adapter_root(self) -> Path:
        """Adapter root whose donor stores this loop resolves against.

        ``self.output_dir`` unless :meth:`borrow_donor_cache` pointed this
        loop at another root's cache.
        """
        return self._borrowed_donor_root or self.output_dir

    @property
    def _borrowed_donor_cache(self) -> bool:
        """True when this loop reads a donor cache it does not own."""
        return self._borrowed_donor_root is not None

    def borrow_donor_cache(self, adapter_root: "Path | str") -> None:
        """Resolve donors against *adapter_root*'s stores, read-only.

        For a loop whose ``output_dir`` is a scratch tree but whose base
        model matches a real deployment's — a migration trial. Without this
        the trial resolves donors under its own empty output dir, misses,
        and pays a full inline donor build (see :mod:`paramem.training.donor`)
        for an artifact the deployment already holds.

        Borrowing is read-only BY CONSTRUCTION, not by a second flag: a
        borrowed cache is never built into and never pruned, so a trial
        running a candidate config can neither add a store to nor remove one
        from the live tree. When the borrowed cache holds nothing valid for
        this loop's base model and topology — the base-swap trial case — the
        target simply trains cold.

        Args:
            adapter_root: The adapter root to borrow from — in production
                ``config.adapter_dir``, the live deployment's own root.
        """
        self._borrowed_donor_root = Path(adapter_root)

    def _resolve_donor_checkpoint(self, adapter_name: str, adapter_config) -> "Path | None":
        """Resolve the donor checkpoint *adapter_name*'s staging slot should
        start from, or ``None``.

        Helper for :meth:`_train_tier_adapter`'s donor resolution — see that
        method's docstring. Validates, builds one when missing and this loop
        owns its donor cache, and returns the validated store directory.
        Copies nothing: applying the returned path (only when the tier has
        no prior trained weights) is :func:`~paramem.training.trainer.
        train_adapter`'s job, not this one's — no donor write ever lands in
        a live tier.

        Returns ``None`` on every degradation branch (the tier already has
        prior trained weights, an unresolvable base id, a borrowed cache
        with no valid checkpoint, a build that could not complete this fold
        — :class:`~paramem.training.donor.DonorBuildIncomplete`, caught here
        specifically — or a checkpoint that still fails to validate after a
        build attempt), without raising: donor seeding is an optimisation
        over LoRA-zero and never costs the fold.

        Args:
            adapter_name: The adapter slot about to train.
            adapter_config: *adapter_name*'s own ``AdapterConfig`` — the
                same object ``_train_tier_adapter`` was called with. Its
                shape (rank, alpha, target_modules) determines which
                topology's donor checkpoint is resolved and, when a build
                is needed, which topology :func:`~paramem.training.donor.build_donor`
                writes into.
        """
        from paramem.models.loader import lora_shape_fields
        from paramem.training.donor import (
            DonorBuildIncomplete,
            build_donor,
            donor_checkpoint_valid,
            donor_store_dir,
        )

        if has_prior_trained_weights(self.model, adapter_name):
            return None

        base_model_id = getattr(self.model.get_base_model().config, "_name_or_path", None)
        if base_model_id is None:
            logger.warning(
                "_resolve_donor_checkpoint: skipping donor resolution for %s -- "
                "base model id unresolved",
                adapter_name,
            )
            return None

        # The donor is built/validated at the TARGET tier's own topology --
        # comparing the CURRENT shape against the checkpoint's recorded
        # shape catches an operator rank/target-modules edit BEFORE
        # copy_adapter_weights (inside train_adapter) would hit a
        # tensor-shape mismatch and abort the fold.
        lora_shape = lora_shape_fields(adapter_config)
        store_dir = donor_store_dir(self.donor_adapter_root, base_model_id, lora_shape)
        if not donor_checkpoint_valid(store_dir, base_model_id, lora_shape):
            if self._borrowed_donor_cache:
                logger.info(
                    "_resolve_donor_checkpoint: borrowed donor cache at %s holds no "
                    "valid checkpoint for this base/topology -- %s trains cold "
                    "(a borrowing loop never builds into a cache it does not own)",
                    self.donor_adapter_root,
                    adapter_name,
                )
                return None
            logger.info(
                "_resolve_donor_checkpoint: donor checkpoint missing/mismatched "
                "for base %s -- building before this fold's training",
                base_model_id,
            )
            try:
                build_donor(self, adapter_config=adapter_config)
            except DonorBuildIncomplete as exc:
                logger.warning(
                    "_resolve_donor_checkpoint: donor build did not complete this "
                    "fold (%s) -- %s trains cold this fold; the next "
                    "measured-cold fold will retry the build",
                    exc,
                    adapter_name,
                )
                return None

            if not donor_checkpoint_valid(store_dir, base_model_id, lora_shape):
                logger.warning(
                    "_resolve_donor_checkpoint: no valid checkpoint for %s after "
                    "build attempt -- trains cold this fold",
                    adapter_name,
                )
                return None

        return store_dir

    # ------------------------------------------------------------------
    # Event staging — see the "Event staging" module header near the top
    # of this file for the boundary this section keeps: ``self.store`` is
    # read only inside ``_recall_working_tiers`` below (the recall
    # boundary); every other method in this section reads and mutates
    # WorkingTier copies only.
    # ------------------------------------------------------------------

    def _recall_working_tiers(
        self,
        primary_tiers: "dict[str, str]",
        candidate_tiers: "dict[str, str]",
        recalled_entries: "dict[str, dict[str, dict]]",
    ) -> "dict[str, WorkingTier]":
        """Seed working state for every tier this event recalls (the recall).

        Merges *primary_tiers* and *candidate_tiers* internally — the whole
        working universe, primary members first — and reads each tier's
        persisted live registry and bookkeeping rows from ``self.store``, and
        its entry content from *recalled_entries* — the fold-local
        reconstruction :meth:`_hydrate_store_for_fold` produced directly from
        the venue — into an independent :class:`WorkingTier` copy.  Every
        read and mutation the staging pass performs afterwards goes to these
        copies; the store's entry mirror
        (:attr:`~paramem.memory.store.MemoryStore._entries`) is never
        touched by this method or anywhere else in this section — only the
        registry and bookkeeping reads below reach ``self.store``, and
        those are a separate structure from the mirror.  Also resolves each
        tier's :meth:`_training_output_dir` scratch path once, here, and
        carries it on the returned :class:`WorkingTier` (``scratch_dir``) —
        the value :meth:`stage_event` records into the ledger for the
        trainer call and every disposer to read back unchanged.

        Each tier's registry is seeded via :meth:`KeyRegistry.working_copy`,
        ``active_only=True`` for a tier named in *primary_tiers* (this event
        rebuilds it, so its markers end here) and ``active_only=False`` for a
        tier named only in *candidate_tiers* (this event only dedups against
        it, so its markers wait for that tier's own rebuild).  The returned
        :class:`WorkingTier`'s ``rebuilt`` field records which case applied.
        Bookkeeping rows follow the SEEDED registry, not the live one — one
        rule, so a rebuilt tier's working rows are active-only from the first
        staging mutation, matching its working entries.

        This is a validation boundary: it meets persisted data, so every key
        the seeded registry knows (active-only for a rebuilt tier, active ∪
        withheld otherwise) must have a bookkeeping row in the live store.  A
        gap raises via
        :func:`~paramem.memory.store.raise_bookkeeping_invariant_violation`,
        naming the tier and every offending key, before this tier's
        :class:`WorkingTier` is built — the every-known-key-has-a-row
        invariant does not tolerate a silent skip here.

        Args:
            primary_tiers: Logical tier name -> adapter/slot name, for every
                tier this event unconditionally rebuilds.
            candidate_tiers: Logical tier name -> adapter/slot name, for
                every dedup-only/candidate member of this event's working
                universe — never a primary tier of the same event (the
                caller's own membership split is a partition).
            recalled_entries: ``{tier: {key: content_only_entry}}`` from
                :meth:`_hydrate_store_for_fold` — this event's own
                reconstruction of every active key's content, already gap-scanned
                (:class:`ActiveKeyHydrationFailure` aborted the fold before
                this method ever runs when the venue could not produce one).
                A tier absent from this map (no active keys) seeds an empty
                entries dict.

        Returns:
            tier name -> :class:`WorkingTier`.

        Raises:
            ~paramem.memory.store.BookkeepingInvariantViolation: One or more
                keys *tier*'s seeded registry reports known have no
                bookkeeping row in the live store.
        """
        from paramem.adapters.manifest import tier_registry_sha256
        from paramem.memory.interim_adapter import adapter_slot_root_for_name
        from paramem.memory.store import raise_bookkeeping_invariant_violation

        tier_adapter_map: dict[str, str] = {**primary_tiers, **candidate_tiers}
        working: dict[str, WorkingTier] = {}
        for tier, adapter_name in tier_adapter_map.items():
            live_registry = (
                self.store.registry(tier) if self.store.has_registry(tier) else KeyRegistry()
            )
            rebuilt = tier in primary_tiers
            seed = live_registry.working_copy(active_only=rebuilt)
            rows = {}
            missing_rows: list[str] = []
            for key in seed.list_known():
                bk = self.store.bookkeeping_for_key(key)
                if bk is None:
                    missing_rows.append(key)
                    continue
                rows[key] = dict(bk)
            if missing_rows:
                raise_bookkeeping_invariant_violation(tier, missing_rows, "working tier recall")
            entries = dict(recalled_entries.get(tier, {}))
            tier_root = adapter_slot_root_for_name(self.output_dir, adapter_name)
            pre_sha = tier_registry_sha256(tier_root)
            working[tier] = WorkingTier(
                tier=tier,
                adapter_name=adapter_name,
                pre_sha=pre_sha,
                rebuilt=rebuilt,
                scratch_dir=self._training_output_dir(adapter_name),
                registry=seed,
                rows=rows,
                entries=entries,
            )
        return working

    def _working_registry_true_relations(self, working_tier: "WorkingTier") -> "list[Relation]":
        """Build registry-true :class:`Relation` objects from one WORKING tier.

        Reads *working_tier*'s recalled active keys against its own working
        entries/rows rather than the live store, so the staging pass's merge
        input is grounded in registry-true (subject, predicate, object)
        content.  Fold-local hydration guarantees a working entry for every
        active key, so a key with no working entry is a designed-impossible
        state, raised rather than skipped; a key whose entry carries content
        but no predicate is unkeyable and is ledgered via
        ``merger.record_removal`` so :meth:`_apply_working_fate_decisions`
        treats it as an explicit removal.

        Raises:
            ~paramem.memory.store.BookkeepingInvariantViolation: *working_tier*
                reports a key active but carries no working entry for it.
        """
        relations: list[Relation] = []
        for key in working_tier.registry.list_active():
            entry = working_tier.entries.get(key)
            bk = working_tier.rows[key]
            if entry is None:
                from paramem.memory.store import raise_bookkeeping_invariant_violation

                raise_bookkeeping_invariant_violation(
                    working_tier.tier,
                    [key],
                    "working merge input: active key has no working entry",
                )
            subj = entry.get("subject", "")
            pred = entry.get("predicate", "")
            obj = entry.get("object", "")
            if not pred:
                if subj or obj:
                    self.merger.record_removal(key, reason="unkeyable_no_predicate")
                logger.debug(
                    "_working_registry_true_relations: key=%s has no predicate -- skipping",
                    key,
                )
                continue
            rt_raw = bk.get("relation_type", _FALLBACK_RTYPE)
            rt: str = rt_raw if rt_raw in _VALID_RTYPES else _FALLBACK_RTYPE
            spk: str = bk.get("speaker_id") or ""
            relations.append(
                Relation(
                    subject=subj,
                    predicate=pred,
                    object=obj,
                    relation_type=rt,  # type: ignore[arg-type]
                    confidence=1.0,
                    speaker_id=spk,
                    indexed_key=key,
                    last_seen=bk.get("last_seen", ""),
                    first_seen=bk.get("first_seen", ""),
                )
            )
        return relations

    def _working_tier_owning(
        self, working: "dict[str, WorkingTier]", key: str
    ) -> "WorkingTier | None":
        """Return whichever WORKING tier currently calls *key* active or withheld.

        Working-copy analogue of ``MemoryStore.tier_for_known_key`` /
        ``tier_of``, scoped to the tiers this staging pass recalled.
        """
        for working_tier in working.values():
            if working_tier.registry.knows(key):
                return working_tier
        return None

    def _apply_working_reinforcement_credit(
        self,
        working: "dict[str, WorkingTier]",
        adopt_reinforcements: "dict[str, tuple[str, str]]",
    ) -> None:
        """Transfer reinforcement credit to the survivors of this event's merges.

        Two channels — every ``merger.removal_ledger`` entry carrying a
        ``survivor_key`` (the survivor inherits the retired keys' counts) and
        *adopt_reinforcements* (a re-sighting of an already-keyed fact that
        displaced nothing) — routed through
        :func:`paramem.memory.bookkeeping.credit_reinforcement` against
        whichever working tier's rows currently own the key, instead of
        ``self.store.reinforce``.  A survivor or an adopted key outside this
        event's recalled universe is silently skipped: nothing recalled it,
        so there is no working row to credit.

        A retired key's owning tier is resolved independently of the
        survivor's — the two may differ (a cross-tier absorption) — so each
        retired key's durable count is read from ITS OWN owning tier's rows,
        never from the survivor's tier's rows: the callee only ever sees
        *this* tier's rows and cannot resolve a cross-tier count itself.

        Idempotent by construction over ``removal_ledger`` entries it has
        already processed: re-running against an unchanged ledger earns zero
        for every already-credited survivor (the earn check requires
        *timestamp* strictly newer than the row's now-stored ``last_seen``,
        which the prior run already advanced to at least that value), so a
        second call with ``adopt_reinforcements={}`` is how the caller picks
        up survivor-bearing entries the keyed walk adds to the ledger after
        the first call already ran.
        """
        ledger: dict[str, dict] = self.merger.removal_ledger
        absorbed: dict[str, list[str]] = {}
        reobserved: dict[str, bool] = {}
        for retired_key, record in ledger.items():
            survivor = record.get("survivor_key")
            if not survivor:
                continue
            absorbed.setdefault(survivor, []).append(retired_key)
            if record.get("reason") in self._REOBSERVED_REASONS:
                reobserved[survivor] = True

        for survivor, retired_keys in absorbed.items():
            owner = self._working_tier_owning(working, survivor)
            if owner is None:
                continue
            owning_tiers = [self._working_tier_owning(working, k) for k in retired_keys]
            bk_rows = [
                (wt.rows[k] if wt is not None else {}) for wt, k in zip(owning_tiers, retired_keys)
            ]
            last_seen = max((bk.get("last_seen", "") for bk in bk_rows), default="")
            first_seen = ""
            for bk in bk_rows:
                first_seen = min_nonempty(first_seen, bk.get("first_seen", ""))
            absorbed_counts = [
                wt.rows[k].get("reinforcement_count", 1)
                for wt, k in zip(owning_tiers, retired_keys)
                if wt is not None
            ]
            credit_reinforcement(
                owner.rows,
                survivor,
                cycle=self.cycle_count,
                timestamp=last_seen,
                first_seen=first_seen,
                absorbed_counts=absorbed_counts,
                reobserved=reobserved.get(survivor, False),
            )
            owner.dirty = True

        for adopted_key, (adopt_ls, adopt_fs) in adopt_reinforcements.items():
            if not adopted_key:
                continue
            owner = self._working_tier_owning(working, adopted_key)
            if owner is None:
                continue
            credit_reinforcement(
                owner.rows,
                adopted_key,
                cycle=self.cycle_count,
                timestamp=adopt_ls,
                first_seen=adopt_fs,
                reobserved=True,
            )
            owner.dirty = True

    def _promote_working_keys(self, working: "dict[str, WorkingTier]") -> "list[str]":
        """Promote matured keys from the working episodic tier to the
        working semantic tier.

        Only meaningful when both ``"episodic"`` and ``"semantic"`` are in
        *working* (a full fold — an interim event never promotes); the
        caller gates the call on its own ``promote`` flag.
        Moves a matured key's registry entry, entry content and bookkeeping
        row from the working episodic tier to the working semantic tier —
        via :meth:`WorkingTier.adopt_key_from` — and flags a key already
        resident in semantic as promoted without moving it.  Every active
        key's bookkeeping row is read directly off its owning working
        tier — the every-known-key-has-a-row invariant, established at
        recall (:meth:`_recall_working_tiers`), means there is no rowless
        case left to skip here; a regression surfaces as ``KeyError`` naming
        the key.  Never deletes a key — an unreinforced key is simply never
        evicted.

        Does NOT mutate ``self.promoted_keys`` directly — this runs during
        STAGING, before the event is known to go live.  A key decided
        promoted here is recorded on the transient
        ``self._pending_promoted_keys`` instead; :meth:`run_build_and_publish`
        merges it into ``self.promoted_keys`` only once the event's
        ``all_live`` verdict is ``True``.  Mutating ``self.promoted_keys``
        here directly would poison it on an abort or a gate rejection: the
        working-copy mutation is discarded (never adopted into the live
        store), but the in-process set is not, so the key's real ``continue``
        guard above would skip it forever — no later event in this process
        would ever attempt to promote it again, even though it was never
        actually promoted.

        Returns:
            List of key ids newly promoted (moved) into semantic this call.
        """
        if "episodic" not in working or "semantic" not in working:
            return []
        threshold = self.config.promotion_threshold
        episodic = working["episodic"]
        semantic = working["semantic"]
        newly_promoted: list[str] = []
        pending_promoted: set[str] = set()

        for key in list(episodic.registry.list_active()):
            if key in self.promoted_keys:
                continue
            bk = episodic.rows[key]
            rec = bk.get("reinforcement_count", 1)
            if rec >= threshold:
                row = semantic.adopt_key_from(episodic, key)
                row["promoted"] = True
                pending_promoted.add(key)
                newly_promoted.append(key)
                logger.info(
                    "_promote_working_keys: key=%s promoted to semantic "
                    "(reinforcement_count=%d >= threshold=%d)",
                    key,
                    rec,
                    threshold,
                )

        for key in list(semantic.registry.list_active()):
            if key in self.promoted_keys:
                continue
            bk = semantic.rows[key]
            if bk.get("reinforcement_count", 1) >= threshold:
                bk["promoted"] = True
                semantic.rows[key] = bk
                semantic.dirty = True
                pending_promoted.add(key)

        self._pending_promoted_keys = pending_promoted
        return newly_promoted

    def _route_absorbed_keyed_fact(
        self,
        working: "dict[str, WorkingTier]",
        tier_keyed: "dict[str, list[dict]]",
        *,
        key: str,
        owner: "WorkingTier",
    ) -> None:
        """Route one keyed fact owned by an about-to-be-reaped candidate tier
        into whichever primary tier its own stored ``relation_type`` selects.

        Used only when this staging pass absorbs its candidate tiers whole (a
        full fold consuming the interim ring, :func:`stage_event`'s
        ``absorb_candidates``): an absorbed tier is never built, published or
        restamped, so a key it alone owns — no duplicate survives elsewhere —
        must migrate into a primary tier's increment now or vanish silently
        when the ring is reaped at go-live.  Mirrors the keyless-mint
        routing in this same walk (``partition_relations`` on the owning
        tier's stored ``relation_type``), except nothing is minted: the key,
        its SimHash fingerprint, its entry and its bookkeeping row move to
        the destination tier's working copy via :meth:`WorkingTier.adopt_key_from`.
        A duplicate that a survivor elsewhere in the working universe already
        absorbed never reaches here — the merger's dedup collapse removes it
        from the graph before this walk runs, and
        :meth:`_apply_working_reinforcement_credit` is what carries its count
        onto the survivor.

        *owner* IS mutated (the key is relinquished, via the same primitive
        every carry uses) but this is invisible: *owner* is a candidate tier
        this staging pass is about to discard whole, unwritten, at the go-live
        regardless of what its own working copy still shows active — the key
        survives only under its destination tier.

        Routing is total: every active key has a working entry (the entry-
        cache completeness invariant asserted pre-mutation in
        :meth:`~paramem.memory.store.MemoryStore.adopt_increments`, and the
        source recall builds working entries from that same store), so the
        entry lookup indexes ``owner.entries`` directly. This helper only
        runs when ``absorb_candidates`` is true, which the caller sets for
        every full-topology event absorbing the whole interim ring
        (``stage_ledger.full_topology(event)``, true for a full fold and a
        reconcile alike) — episodic is always among that event's primary
        tiers (``_tier_config_map`` includes it
        unconditionally, and ``partition_relations`` is called with the same
        ``procedural_enabled`` predicate that decides whether procedural is
        in the primary set), so ``dest_tier`` is always a real destination
        and ``tier_keyed``/``working`` are indexed directly. A violated
        topology in either guarantee raises ``KeyError`` loudly before any
        artifact is written, rather than silently dropping the key.

        Args:
            working: Every tier's working copy, keyed by tier name.
            tier_keyed: This walk's per-primary-tier keyed-row accumulator,
                mutated in place on route.
            key: The indexed key to route.
            owner: The candidate tier's :class:`WorkingTier` currently
                holding *key*.
        """
        entry = owner.entries[key]
        bk = owner.rows[key]
        rt_raw = bk.get("relation_type", _FALLBACK_RTYPE)
        rt: str = rt_raw if rt_raw in _VALID_RTYPES else _FALLBACK_RTYPE
        spk: str = bk.get("speaker_id") or ""
        dummy = [
            {
                "subject": entry["subject"],
                "predicate": entry["predicate"],
                "object": entry["object"],
                "relation_type": rt,
            }
        ]
        _ep_rels, _proc_rels = partition_relations(
            dummy, procedural_enabled="procedural" in self.tier_adapters
        )
        dest_tier = "procedural" if _proc_rels else "episodic"
        dest = working[dest_tier]
        dest.adopt_key_from(owner, key)
        tier_keyed[dest_tier].append(
            {
                "key": key,
                "subject": entry["subject"],
                "predicate": entry["predicate"],
                "object": entry["object"],
                "speaker_id": spk,
                "relation_type": rt,
            }
        )

    def _build_working_keyed_walk(
        self,
        working: "dict[str, WorkingTier]",
        *,
        exclude_keys: "set[str]",
        absorb_candidates: bool = False,
    ) -> "dict[str, list[dict]]":
        """Walk every merged-graph edge AND node attribute; assemble this
        event's per-tier keyed list.

        Walks every edge of ``self.merger.graph``.  The rebuilt set — every
        tier in *working* whose :attr:`WorkingTier.rebuilt` is ``True`` —
        is derived once, up front, preserving *working*'s own iteration
        order; a keyless edge mints a new key — via
        :meth:`_mint_keyed_entries`, which touches no shared state — and
        registers it on the emitting rebuilt tier's working registry, rows
        and entries.  A keyed edge is an anti-forgetting replay of an
        existing key: its owning tier is resolved via
        :meth:`_working_tier_owning`.  When that tier is in the rebuilt set
        the edge replays there unchanged.  When it is not — a dedup-only/
        candidate tier's key — the edge is *routed* via
        :meth:`_route_absorbed_keyed_fact` when *absorb_candidates* is
        ``True`` (this staging pass's candidate tiers are read-only members it
        is about to reap whole, so their unique content must migrate into a
        rebuilt tier or be lost), and otherwise dropped, exactly as it is
        when the key is named in *exclude_keys* (a dedup-only candidate
        tier's key is never keyed-replayed into a fresh keyed list — it
        stays resident where it already lives).  A keyless fact whose
        computed tier is not in the rebuilt set (e.g. a procedural fact
        surfacing while this event rebuilds only an episodic-shaped tier)
        is likewise dropped — this event has no shadow tree to place it in.

        A second pass after the edge walk covers node ``attributes``
        (``Entity.attributes`` — phone/email/date/job-title style facts,
        invisible to an edge iteration).  Each entry under
        ``node["attributes"]`` is a provenance-bearing record
        (``{value, speaker_id, first_seen, last_seen, edge_source?,
        ik_key?}``, written by :meth:`~paramem.graph.merger.GraphMerger.merge`'s
        attribute gate); this pass reads it through
        :func:`~paramem.graph.merger.attribute_fact`, the one node-record ->
        fact projection also used by :meth:`_capture_pending_relations`. A
        keyed attribute (the record carries an ``ik_key``) is an
        anti-forgetting replay sourced from its owning working tier, handled
        by the same rebuilt/route/drop rule as a keyed edge; a keyless
        attribute mints a new key — carrying the record's own ``speaker_id``
        and window, via :func:`~paramem.memory.bookkeeping.bookkeeping_row` —
        on the emitting node's working tier, exactly like a keyless edge.

        Cross-representation dedup: subject and predicate together are an
        attribute's full identity (one value per predicate per node), so
        the edge walk's every emitted ``(subject, predicate)`` pair is
        recorded; the attribute pass skips any pair it already saw rather
        than emitting a second key for the same fact.  A skipped pair that
        carries a registered attribute key retires that key via
        :meth:`~paramem.graph.merger.GraphMerger.record_removal` with
        ``reason="duplicate_projection"`` and the edge's key as survivor —
        the caller's post-walk reinforcement-credit pass carries its count
        onto the surviving edge key, and :meth:`_apply_working_fate_decisions`
        retires it outright or withholds it behind a marker, per its owning
        tier's rebuilt state. A skipped pair with no
        registered attribute key (never keyed) is dropped with nothing to
        ledger.

        Args:
            working: Every tier's working copy, keyed by tier name.  The
                rebuilt set this walk builds against is derived from each
                member's own :attr:`WorkingTier.rebuilt` flag — no separate
                tier-name list is threaded in.
            exclude_keys: Keys to drop unconditionally, before either the
                rebuilt-tier or the routing check (a dedup-only candidate
                tier's own active keys, when *absorb_candidates* is
                ``False``).
            absorb_candidates: ``True`` for a full fold consuming its
                candidate tiers whole (the interim ring) — routes a
                candidate-owned keyed fact into its relation_type's rebuilt
                tier instead of leaving it stranded on a tier that is never
                built, published or restamped.  ``False`` (the default, and
                every other caller) drops it instead.

        Returns:
            tier -> list of keyed rows, one list per rebuilt tier in
            *working* — each row ``{key, subject, predicate, object,
            speaker_id, relation_type}``, the shape ``keyed.json`` persists.

        Raises:
            ~paramem.memory.store.BookkeepingInvariantViolation: A keyed
                edge's owning working tier reports the key active but
                carries no working entry for it — fold-local hydration
                guarantees a working entry for every active key, so this is
                a designed-impossible state, not a skip.
        """
        from paramem.memory.interim_adapter import interim_stamp_from_name
        from paramem.memory.persistence import _IK_KEY_ATTR as _IK_ATTR

        rebuilt_tiers = [t for t, wt in working.items() if wt.rebuilt]
        tier_keyed: dict[str, list[dict]] = {t: [] for t in rebuilt_tiers}

        # A keyless mint's destination tier is chosen by KIND (episodic- or
        # procedural-shaped), never by the literal main-tier names
        # "episodic"/"procedural": an interim event's sole rebuilt tier is
        # named "episodic_interim_<stamp>" and receives BOTH kinds -- the
        # interim commit window mints procedural-typed keys beside episodic
        # ones, differentiated only by the stored relation_type until an
        # absorbing full fold's keyed walk later routes them into
        # "procedural" by that stored type.  A literal "episodic"/
        # "procedural" destination never matches an interim rebuilt tier's
        # name, so a genuinely new fact minted during an interim tick would
        # land nowhere.
        mint_destination: dict[str, str] = {}
        for _rebuilt_name in rebuilt_tiers:
            if interim_stamp_from_name(_rebuilt_name) is not None:
                mint_destination.setdefault("episodic", _rebuilt_name)
                mint_destination.setdefault("procedural", _rebuilt_name)
            elif _rebuilt_name in ("episodic", "procedural"):
                mint_destination[_rebuilt_name] = _rebuilt_name

        local_indexed: "int | None" = None
        local_procedural: "int | None" = None

        # Cross-representation dedup: every (subject node, predicate) pair
        # this walk actually emits into tier_keyed, mapped to the key it
        # was emitted under.  The node-attribute pass below consults this
        # to recognize the same fact projected a second way -- subject and
        # predicate together are the full identity of an attribute
        # (Entity.attributes holds one value per predicate per node), so a
        # pair already emitted by the edge walk can only be the same fact.
        emitted_pairs: "dict[tuple[str, str], str]" = {}

        for subj_node, obj_node, data in self.merger.graph.edges(data=True):
            key = data.get(_IK_ATTR)
            pred = data.get("predicate", "")
            if not pred:
                continue
            if key and key in exclude_keys:
                continue

            if not key:
                rt_raw = data.get("relation_type", _FALLBACK_RTYPE)
                rt: str = rt_raw if rt_raw in _VALID_RTYPES else _FALLBACK_RTYPE
                subj_display = node_display(self.merger.graph.nodes[subj_node], subj_node)
                obj_display = node_display(self.merger.graph.nodes[obj_node], obj_node)
                subj_sid = data["speaker_id"]

                dummy = [
                    {
                        "subject": subj_display,
                        "predicate": pred,
                        "object": obj_display,
                        "relation_type": rt,
                    }
                ]
                _ep_rels, _proc_rels = partition_relations(
                    dummy, procedural_enabled="procedural" in self.tier_adapters
                )
                kind = "procedural" if _proc_rels else "episodic"
                tier = mint_destination.get(kind)
                if tier is None:
                    continue

                prefix = "proc" if kind == "procedural" else "graph"
                if kind == "procedural":
                    if local_procedural is None:
                        local_procedural = self._procedural_next_index
                    start_index = local_procedural
                else:
                    if local_indexed is None:
                        local_indexed = self._indexed_next_index
                    start_index = local_indexed

                minted = self._mint_keyed_entries(
                    [
                        {
                            "subject": subj_display,
                            "predicate": pred,
                            "object": obj_display,
                            "relation_type": rt,
                            "speaker_id": subj_sid,
                        }
                    ],
                    prefix=prefix,
                    start_index=start_index,
                    speaker_id=subj_sid,
                )[0]

                if kind == "procedural":
                    local_procedural += 1
                    self._procedural_next_index += 1
                else:
                    local_indexed += 1
                    self._indexed_next_index += 1

                minted_key = minted["key"]
                wt = working[tier]
                wt.registry.add(minted_key)
                wt.registry.set_simhash(minted_key, entry_simhash(minted))
                wt.entries[minted_key] = content_only_entry(minted)
                wt.rows[minted_key] = bookkeeping_row(
                    minted_key,
                    speaker_id=subj_sid,
                    relation_type=rt,
                    first_seen=data.get("first_seen", ""),
                    promoted=False,
                    last_reinforced_cycle=self.cycle_count,
                    last_seen=data.get("last_seen", ""),
                )
                wt.dirty = True
                tier_keyed[tier].append(
                    {
                        "key": minted_key,
                        "subject": minted["subject"],
                        "predicate": pred,
                        "object": minted["object"],
                        "speaker_id": subj_sid,
                        "relation_type": rt,
                    }
                )
                emitted_pairs[(subj_node, pred)] = minted_key
            else:
                owner = self._working_tier_owning(working, key)
                if owner is None:
                    continue
                if owner.tier not in tier_keyed:
                    if absorb_candidates:
                        self._route_absorbed_keyed_fact(working, tier_keyed, key=key, owner=owner)
                    continue
                entry = owner.entries.get(key)
                if entry is None:
                    from paramem.memory.store import raise_bookkeeping_invariant_violation

                    raise_bookkeeping_invariant_violation(
                        owner.tier, [key], "keyed replay walk: active key has no working entry"
                    )
                bk = owner.rows[key]
                rt_raw = bk.get("relation_type", _FALLBACK_RTYPE)
                rt = rt_raw if rt_raw in _VALID_RTYPES else _FALLBACK_RTYPE
                spk = bk.get("speaker_id") or ""
                tier_keyed[owner.tier].append(
                    {
                        "key": key,
                        "subject": entry["subject"],
                        "predicate": entry["predicate"],
                        "object": entry["object"],
                        "speaker_id": spk,
                        "relation_type": rt,
                    }
                )
                emitted_pairs[(subj_node, pred)] = key

        # ---- Node-attribute walk: attribute-typed relations never become
        # edges (GraphMerger.merge diverts relation_type == "attribute" onto
        # the subject node's "attributes" dict), so they are invisible to
        # the edge walk above.  Mirrors both edge-walk branches: a keyed
        # attribute is an anti-forgetting replay from its owning working
        # tier, a keyless one mints a new key on the emitting node's tier.
        for node, node_data in self.merger.graph.nodes(data=True):
            node_attrs = node_data.get("attributes", {}) or {}
            if not node_attrs:
                continue
            for attr_key, record in node_attrs.items():
                fact = attribute_fact(node_data, node, attr_key, record)
                attr_pred = fact["predicate"]
                attr_key_id = fact["ik_key"]

                survivor_key = emitted_pairs.get((node, attr_pred))
                if survivor_key is not None:
                    # Same fact already emitted this event as an edge --
                    # cross-representation duplicate.  A registered
                    # duplicate is retired onto the surviving edge key so
                    # reinforcement credit carries its count forward and
                    # _apply_working_fate_decisions decides its fate (retired
                    # outright or withheld behind a marker); an unregistered
                    # one has nothing to ledger and is simply dropped.
                    if attr_key_id:
                        self.merger.record_removal(
                            attr_key_id,
                            reason="duplicate_projection",
                            survivor_key=survivor_key,
                        )
                    continue

                if attr_key_id and attr_key_id in exclude_keys:
                    continue

                if attr_key_id:
                    owner = self._working_tier_owning(working, attr_key_id)
                    if owner is None:
                        continue
                    if owner.tier not in tier_keyed:
                        if absorb_candidates:
                            self._route_absorbed_keyed_fact(
                                working, tier_keyed, key=attr_key_id, owner=owner
                            )
                        continue
                    entry = owner.entries.get(attr_key_id)
                    if entry is None:
                        logger.debug(
                            "_build_working_keyed_walk: attribute key=%s has no"
                            " working entry -- skipping",
                            attr_key_id,
                        )
                        continue
                    bk = owner.rows[attr_key_id]
                    rt_raw = bk.get("relation_type", _FALLBACK_RTYPE)
                    rt = rt_raw if rt_raw in _VALID_RTYPES else _FALLBACK_RTYPE
                    spk = bk.get("speaker_id") or ""
                    tier_keyed[owner.tier].append(
                        {
                            "key": attr_key_id,
                            "subject": entry["subject"],
                            "predicate": entry["predicate"],
                            "object": entry["object"],
                            "speaker_id": spk,
                            "relation_type": rt,
                        }
                    )
                else:
                    rt = "attribute"
                    subj_sid = fact["speaker_id"]
                    dummy = [
                        {
                            "subject": fact["subject"],
                            "predicate": attr_pred,
                            "object": fact["object"],
                            "relation_type": rt,
                        }
                    ]
                    _ep_rels, _proc_rels = partition_relations(
                        dummy, procedural_enabled="procedural" in self.tier_adapters
                    )
                    kind = "procedural" if _proc_rels else "episodic"
                    tier = mint_destination.get(kind)
                    if tier is None:
                        continue

                    prefix = "proc" if kind == "procedural" else "graph"
                    if kind == "procedural":
                        if local_procedural is None:
                            local_procedural = self._procedural_next_index
                        start_index = local_procedural
                    else:
                        if local_indexed is None:
                            local_indexed = self._indexed_next_index
                        start_index = local_indexed

                    minted = self._mint_keyed_entries(
                        [
                            {
                                "subject": fact["subject"],
                                "predicate": attr_pred,
                                "object": fact["object"],
                                "relation_type": rt,
                                "speaker_id": subj_sid,
                            }
                        ],
                        prefix=prefix,
                        start_index=start_index,
                        speaker_id=subj_sid,
                    )[0]

                    if kind == "procedural":
                        local_procedural += 1
                        self._procedural_next_index += 1
                    else:
                        local_indexed += 1
                        self._indexed_next_index += 1

                    minted_key = minted["key"]
                    wt = working[tier]
                    wt.registry.add(minted_key)
                    wt.registry.set_simhash(minted_key, entry_simhash(minted))
                    wt.entries[minted_key] = content_only_entry(minted)
                    wt.rows[minted_key] = bookkeeping_row(
                        minted_key,
                        speaker_id=subj_sid,
                        relation_type=rt,
                        first_seen=fact["first_seen"],
                        promoted=False,
                        last_reinforced_cycle=self.cycle_count,
                        last_seen=fact["last_seen"],
                    )
                    wt.dirty = True
                    tier_keyed[tier].append(
                        {
                            "key": minted_key,
                            "subject": minted["subject"],
                            "predicate": attr_pred,
                            "object": minted["object"],
                            "speaker_id": subj_sid,
                            "relation_type": rt,
                        }
                    )

        return tier_keyed

    def _apply_working_fate_decisions(self, working: "dict[str, WorkingTier]") -> None:
        """Working-copy port of the acting-site fate rule over ``merger.removal_ledger``.

        The fate of a retired key follows one axis, read from the owning
        working tier's own ``rebuilt`` field — never ``survivor_key``: a
        tier this event REBUILDS is already re-deriving its content from a
        key set the retired id is not in, so the key is retired outright —
        removed from the registry, the entries and the rows, and after this
        event's go-live nothing enumerates the id.  A tier this event does
        NOT rebuild publishes rows-only and leaves its adapter answering
        whatever it already answers, so its registry is the only thing
        standing between the id and an enumerator: the key is withheld
        behind a marker instead, reserving the id until that tier's own
        rebuild retires it.  Applies to whichever recalled tier currently
        owns the retired key, main or dedup-only/candidate alike
        (:meth:`_working_tier_owning`).  ``survivor_key`` keeps its one real
        consumer, :meth:`_apply_working_reinforcement_credit`, which runs
        before this method on every call path.
        """
        for retired_key in self.merger.removal_ledger:
            owner = self._working_tier_owning(working, retired_key)
            if owner is None:
                continue
            if owner.rebuilt:
                owner.registry.remove(retired_key)
                owner.entries.pop(retired_key, None)
                owner.rows.pop(retired_key, None)
            else:
                owner.registry.stale(retired_key)
            owner.dirty = True

    def _write_shadow_tier(
        self,
        *,
        shadow_root: Path,
        tier: str,
        working_tier: "WorkingTier",
        keyed: "list[dict] | None",
    ) -> "list[Path]":
        """Write one tier's shadow artifacts under ``shadow_root / tier``.

        ``keyed=None`` writes a rows-only member (registry + rows, no
        ``keyed.json`` — the file-presence signal
        :func:`~paramem.memory.increment.build_tier_increment` reads as
        ``rebuilt=False``) AND unlinks any ``keyed.json`` already present in
        this tier's shadow directory, so a rows-only write can never leave a
        stale keyed list behind for :func:`~paramem.memory.increment.build_tier_increment`
        to misread as ``rebuilt=True``; a list (possibly empty) writes a full
        member. The two signals must agree: *keyed* is a list exactly when
        *working_tier* was recalled with ``rebuilt=True``
        (:meth:`_recall_working_tiers`), so ``working_tier.registry`` here
        already carries no withheld markers for a full member — it was
        seeded active-only at recall — while a rows-only member's registry
        still carries whatever markers this event left on it, including any
        this event's own fate decisions just minted. ``stage_event``'s two
        write loops produce that agreement positionally (a list for every
        *primary_tiers* member, ``None`` for every *candidate_tiers* member,
        matching how ``rebuilt`` was seeded from that same membership at
        recall) — this method checks it structurally instead of trusting
        the positions, since it is the boundary that owns the write. The
        registry is written last via :meth:`KeyRegistry.save` — the same
        primitive the live path uses — so a shadow tree observed mid-write
        never shows a registry with no matching rows file.

        Returns:
            Every path written, registry last.

        Raises:
            RuntimeError: *keyed* and *working_tier.rebuilt* disagree —
                one caller passed a list where ``rebuilt`` is ``False``, or
                ``None`` where ``rebuilt`` is ``True``.
        """
        from paramem.backup.encryption import write_infra_json

        if (keyed is not None) != working_tier.rebuilt:
            raise RuntimeError(
                f"_write_shadow_tier: keyed/rebuilt mismatch for tier {tier!r} -- "
                f"keyed is {'a list' if keyed is not None else 'None'} but "
                f"working_tier.rebuilt={working_tier.rebuilt!r}"
            )

        tier_dir = shadow_root / tier
        tier_dir.mkdir(parents=True, exist_ok=True)

        rows_path = tier_dir / "key_metadata.json"
        write_infra_json(
            rows_path,
            {"tier_cycle": self.cycle_count, "keys": dict(working_tier.rows)},
        )
        written = [rows_path]

        keyed_path = tier_dir / "keyed.json"
        if keyed is not None:
            write_infra_json(keyed_path, keyed)
            written.append(keyed_path)
        else:
            keyed_path.unlink(missing_ok=True)

        registry_path = tier_dir / "indexed_key_registry.json"
        working_tier.registry.save(registry_path)
        written.append(registry_path)
        return written

    def stage_event(
        self,
        *,
        event: "Literal['interim', 'full', 'reconcile']",
        venue: "Literal['weights', 'disk']",
        stamp: str,
        primary_tiers: "dict[str, str]",
        recalled_entries: "dict[str, dict[str, dict]]",
        candidate_tiers: "dict[str, str] | None" = None,
        episodic_rels: "list[Relation] | None" = None,
        procedural_rels: "list[Relation] | None" = None,
        session_ids: "list[str] | None" = None,
        promote: bool = False,
        normalize: bool = False,
        enrich: bool = False,
        resolve_contradictions: bool = False,
    ) -> "StagedEvent | None":
        """Stage one training event's extraction product into its shadow tree.

        One implementation, used by every training event (interim, full, and
        reconcile) with their differing working universes; a simulate event
        of either full-topology kind runs this same staging pass (venue
        recorded as ``"disk"``).

        0. Clears ``<state_dir>/extraction/`` wholesale, before anything
           else runs. This method is the ONLY writer under that tree, and
           the tree's own event-scoped shadow content only becomes a
           pending event once the ledger's ``"extraction"`` entry is
           written at step 4 — so any content already there at entry is
           debris from a staging pass that crashed before that write ever
           landed, never a currently-pending event's own material.
           Clearing the whole tree rather than only ``extraction/<event>/``
           also reclaims a crashed OTHER-kind staging pass's debris, which
           nothing else ever revisits.
        1. Consumes the event's already-extracted material (*episodic_rels*
           / *procedural_rels* — the same :class:`Relation` objects the
           extraction pipeline produces) through the same
           :class:`~paramem.graph.merger.GraphMerger` and the same
           :meth:`build_tier_refiner` enrichment topology.  Extraction
           itself (session -> relations) runs wherever the caller invokes
           it; this method never calls it.
        2. Recalls every tier in *primary_tiers* / *candidate_tiers* into a
           working copy (:meth:`_recall_working_tiers`), then, in this
           order: credits reinforcement
           (:meth:`_apply_working_reinforcement_credit`, against the merge
           outcome) — promotes matured keys when *promote*
           (:meth:`_promote_working_keys`) — builds the keyed walk
           (:meth:`_build_working_keyed_walk`, which can surface further
           removal-ledger entries of its own) — credits reinforcement again
           (a no-op for everything already credited, first credit for
           anything the walk just surfaced) — applies fate decisions at
           their acting sites on the working registries
           (:meth:`_apply_working_fate_decisions` — retired outright on a
           tier this event rebuilds, withheld behind a marker otherwise).
           Fate runs LAST and credit runs before it (twice) so that a key
           this step retires has already earned any reinforcement credit
           it is due — dropping a row in the fate step can never starve
           credit, because credit already ran against it.

           Then determines which tiers this event actually built.
        3. Writes the per-event shadow tree under this loop's own
           ``<state_dir>/extraction/<event>/`` (`stage_ledger.extraction_dir`):
           per built tier the shadow registry, shadow rows and (for a
           primary tier) the keyed list; ``event``, venue, stamp, and,
           per built tier, its adapter name, recalled ``pre_sha`` and
           resolved training scratch dir, recorded once in the ledger's
           ``tiers`` map (see :class:`WorkingTier`'s own ``scratch_dir``
           docstring for why it is recorded rather than recomputed). No
           event-root graph snapshot is written — resume runs its later
           phase on a fresh empty merger, and the simulate payload projects
           from ``increment.keyed``, so a snapshot has no reader; the
           ``on_fold_graph`` debug hook remains the sanctioned diagnostics
           channel for the merged graph.
        4. Writes the ledger's ``"extraction"`` entry via
           :func:`~paramem.training.stage_ledger.write_stages` — this
           method's completion signal.
        5. Owns the empty-relations early exit: an event that stages no
           relations to merge terminates here, returning ``None``
           — no shadow tree, no ledger, nothing to dispose.

        Args:
            event: ``"interim"``, ``"full"``, or ``"reconcile"`` — recorded
                in the ledger head, AND the signal this method uses to
                decide whether its *candidate_tiers* are absorbed whole
                (:func:`~paramem.training.stage_ledger.full_topology`,
                ``event != "interim"`` — a full fold and a reconcile absorb
                every candidate tier; an interim tick's candidate tiers are
                merely dedup-only/read-only).
            venue: ``"weights"`` (train) or ``"disk"`` (simulate) — recorded
                only; this method makes no venue branch of its own (no
                weight probe runs here — see the module header).
            stamp: The window/interim stamp — recorded only.
            primary_tiers: Logical tier -> adapter name, for every tier this
                event unconditionally builds (an interim tick's own new
                slot; a full-topology event's three main tiers).  Always in
                the returned ``built_tiers``, each with a shadow
                ``keyed.json``.
            candidate_tiers: Logical tier -> adapter name, for every
                dedup-only/read-only member of this event's working universe
                (a full-topology event's absorbed interim slots; an interim
                tick's three main tiers and sibling slots).  Recalled and
                folded into the merge like any other tier.  Two shapes,
                chosen by whether this event absorbs its candidate tiers
                whole (see *event*): an absorbing full-topology event's
                candidate tiers are never keyed-replayed under their own
                name and never written to the shadow tree at all — never
                built, published or restamped, because they are reaped
                whole at the go-live — and a keyed fact one of them alone
                owns is instead routed into whichever primary tier its own
                stored ``relation_type`` selects
                (:meth:`_route_absorbed_keyed_fact`), so it survives the
                reap under a new tier.  A non-absorbing event's (an interim
                tick's) candidate tiers are never keyed-replayed either, but
                stay resident where they already live — written as a
                rows-only shadow (no ``keyed.json``) only when this event
                actually changed that tier's registry or rows, and left
                untouched — not built at all — otherwise.
            recalled_entries: ``{tier: {key: content_only_entry}}`` — this
                event's own venue reconstruction, from the caller's own
                :meth:`_hydrate_store_for_fold` call (a gap-scanned,
                already-raised-on-failure result covering every tier this
                call's *primary_tiers* / *candidate_tiers* name).  Threaded
                straight into :meth:`_recall_working_tiers`; the store's
                entry mirror is never read here.
            episodic_rels: This event's newly extracted episodic-shaped
                relations (the pending-session content).
            procedural_rels: This event's newly extracted procedural-shaped
                relations.
            session_ids: Completed session ids this event consumed — recorded
                verbatim in the ledger's extraction entry.
            promote: Run :meth:`_promote_working_keys` after reinforcement
                credit.  ``True`` only for a full fold in either venue; an
                interim event never promotes.
            normalize: Run the whole-graph predicate/entity normalization
                pass (local-model — touches the GPU when a real model is
                mounted).
            enrich: Run cloud graph-tier enrichment.
            resolve_contradictions: Forwarded to the merger's own and
                pending-relation merges (never to the candidate-tier dedup
                merge, which is always ``resolve_contradictions=False``).

        Returns:
            The :class:`StagedEvent`, or ``None`` on the no-facts
            outcome.
        """
        import shutil

        from paramem.training.stage_ledger import (
            StageLedger,
            build_artifact_list,
            extraction_dir,
            extraction_stage,
            full_topology,
            write_stages,
        )

        # --- step 0: reclaim debris from a staging pass that crashed before
        # its ledger was ever written (see this method's own docstring) ---
        state_dir = self._fold_state_dir
        extraction_tree = extraction_dir(state_dir, event).parent
        if extraction_tree.exists():
            shutil.rmtree(extraction_tree, ignore_errors=True)

        candidate_tiers = candidate_tiers or {}
        tier_map: dict[str, str] = {**primary_tiers, **candidate_tiers}
        # Every full-topology event (a full fold or a reconcile) absorbs
        # every candidate tier whole -- see stage_event's own *event* doc.
        # An interim tick's candidate tiers are dedup-only/read-only and
        # stay resident where they live.
        absorb_candidates = full_topology(event)

        # --- empty-relations early exit (cheap, pre-recall) ---
        has_new_material = bool(episodic_rels) or bool(procedural_rels)
        has_existing_content = any(self.store.active_keys_in_tier(t) for t in tier_map)
        if not has_new_material and not has_existing_content:
            return None

        working = self._recall_working_tiers(primary_tiers, candidate_tiers, recalled_entries)

        self.merger.reset_graph()

        needs_guard = self.model is not None and resolve_contradictions

        # 1. This event's own (primary-tier) recalled content.
        recon_relations: list[Relation] = []
        for tier in primary_tiers:
            recon_relations.extend(self._working_registry_true_relations(working[tier]))
        if needs_guard:
            self._disable_gradient_checkpointing()
        try:
            self.merger.merge_relations(
                recon_relations,
                session_id="__stage_recon__",
                log_label="reconstructed triples",
                resolve_contradictions=resolve_contradictions,
            )
        finally:
            if needs_guard:
                self._enable_gradient_checkpointing()

        # 2. This event's newly extracted material.
        extra_relations = list(episodic_rels or []) + list(procedural_rels or [])
        if needs_guard:
            self._disable_gradient_checkpointing()
        try:
            self.merger.merge_relations(
                extra_relations,
                session_id="__stage_pending__",
                log_label="pending relations",
                resolve_contradictions=resolve_contradictions,
                credit_adopt_reinforcement=True,
            )
        finally:
            if needs_guard:
                self._enable_gradient_checkpointing()

        # 3. Dedup-only/candidate-tier content, merged LAST, never
        #    contradiction-resolved.
        dedup_relations: list[Relation] = []
        for tier in candidate_tiers:
            dedup_relations.extend(self._working_registry_true_relations(working[tier]))
        if dedup_relations:
            self.merger.merge_relations(
                dedup_relations,
                session_id="__stage_dedup_targets__",
                log_label="dedup-target relations",
                resolve_contradictions=False,
                credit_adopt_reinforcement=True,
            )

        # Debug: snapshot the merged graph (after the three merges above,
        # before refinement).  `graph_merged_snapshot.json` has a real
        # consumer — paramem.server.calibrate reads it via an
        # operator-supplied snapshot_path.  Self-gated; no-op when
        # save_cycle_snapshots=False.
        on_fold_graph(self.merger.graph, label="merged")

        refiner = self.build_tier_refiner(self.merger)
        result = refiner.refine(normalize=normalize, enrich=enrich)
        self._record_enrichment_incident(result)
        self._apply_working_reinforcement_credit(working, result.adopt_reinforcements)

        if promote:
            self._promote_working_keys(working)

        exclude_keys: set[str] = set()
        if not absorb_candidates:
            for tier in candidate_tiers:
                exclude_keys |= set(working[tier].registry.list_active())

        tier_keyed = self._build_working_keyed_walk(
            working,
            exclude_keys=exclude_keys,
            absorb_candidates=absorb_candidates,
        )
        on_fold_assignments(tier_keyed)

        # The walk above can add its own removal_ledger entries (a
        # cross-representation duplicate found only while walking node
        # attributes) after the merge-stage credit call above already ran.
        # Re-running the same credit pass is safe: every entry it already
        # processed recomputes to the same last_seen/count (the earn check
        # compares against the row's now-stored last_seen) and is a no-op,
        # while any new survivor-bearing entry from the walk is credited
        # here for the first time.  No second adopt_reinforcements pass --
        # those were fully applied above.
        self._apply_working_reinforcement_credit(working, {})

        self._apply_working_fate_decisions(working)
        on_removal_ledger(dict(self.merger.removal_ledger))

        # --- write the shadow tree ---
        event_dir = extraction_dir(state_dir, event)
        shadow_root = event_dir / "shadow"

        artifact_paths: list[Path] = []
        tiers_map: dict[str, dict] = {}
        built_tiers: list[str] = []

        for tier in primary_tiers:
            wt = working[tier]
            written = self._write_shadow_tier(
                shadow_root=shadow_root,
                tier=tier,
                working_tier=wt,
                keyed=tier_keyed.get(tier, []),
            )
            artifact_paths.extend(written)
            tiers_map[tier] = {
                "adapter": wt.adapter_name,
                "pre_sha": wt.pre_sha,
                "scratch": str(wt.scratch_dir),
            }
            built_tiers.append(tier)

        if not absorb_candidates:
            for tier in candidate_tiers:
                wt = working[tier]
                if not wt.dirty:
                    continue
                written = self._write_shadow_tier(
                    shadow_root=shadow_root, tier=tier, working_tier=wt, keyed=None
                )
                artifact_paths.extend(written)
                tiers_map[tier] = {
                    "adapter": wt.adapter_name,
                    "pre_sha": wt.pre_sha,
                    "scratch": str(wt.scratch_dir),
                }
                built_tiers.append(tier)
        # else: an absorbed candidate tier is never built, published or
        # restamped -- it is reaped whole at the go-live (stage_event's
        # own *event* doc).  Any key it alone owned has already been routed
        # into a primary tier's increment by _route_absorbed_keyed_fact; a
        # duplicate a primary-tier survivor absorbed was credited by
        # _apply_working_reinforcement_credit.  The tier's own working copy
        # is discarded here, unwritten.

        ledger = StageLedger(
            event=event,
            venue=venue,
            stamp=stamp,
            tiers=tiers_map,
            absorbed_interim_tiers=tuple(candidate_tiers) if absorb_candidates else (),
        )
        extraction_entry = extraction_stage(
            completed_at=datetime.now(timezone.utc).isoformat(),
            sessions=list(session_ids or []),
            episodic_rels=len(episodic_rels or []),
            procedural_rels=len(procedural_rels or []),
            artifacts=build_artifact_list(artifact_paths),
        )
        write_stages(state_dir, ledger, [extraction_entry])
        ledger = replace(ledger, stages=(extraction_entry,))

        return StagedEvent(
            event=event,
            venue=venue,
            state_dir=state_dir,
            built_tiers=tuple(built_tiers),
            ledger=ledger,
        )

    # ------------------------------------------------------------------
    # Build / write / go-live driver — the second phase of every
    # consolidation event, consuming a StagedEvent from the
    # staging section above (or a resumed event's own pending ledger).
    # ------------------------------------------------------------------

    def _build_write_context(self, *, extra_tiers: "Sequence[str]" = ()) -> "TierWriteContext":
        """Assemble this event's :class:`~paramem.memory.increment.TierWriteContext`
        from the loop's own state.

        Rebuilt fresh at every call site that needs one rather than cached,
        since ``TierWriteContext`` is frozen and a caller building one
        earlier in the event may not reflect a config change mid-process —
        the loop's own state (``self.tier_adapters``, ``self.model``) is the
        single source, re-read each time. ``self.model``'s object identity
        never changes (fixed at load time), so this is about staying
        current with ``self.tier_adapters``, not with a reassigned model.

        Args:
            extra_tiers: Tier names beyond this loop's main tiers that the
                caller's bundle may name (an interim slot such as
                ``"episodic_interim_<stamp>"``). Each is resolved through
                :meth:`_tier_adapter_config` — the one rule home for the
                interim-is-episodic-shaped fallback — and folded into the
                returned context's ``tier_configs``, so
                :func:`~paramem.training.go_live.publish_bundle` can do a
                plain total lookup over every tier its bundle names.
        """
        from paramem.memory.increment import TierWriteContext

        tier_configs = dict(self.tier_adapters)
        for tier in extra_tiers:
            tier_configs.setdefault(tier, self._tier_adapter_config(tier))

        return TierWriteContext(
            model=self.model,
            tokenizer=self.tokenizer,
            fingerprint_cache=self.fingerprint_cache,
            output_dir=self.output_dir,
            tier_configs=tier_configs,
            store=self.store,
            keep_prior_slots=self._keep_prior_slots,
        )

    def _tier_adapter_config(self, tier: str) -> "AdapterConfig":
        """One tier's ``AdapterConfig`` — ``self.tier_adapters`` plus the
        interim-is-episodic-shaped fallback.

        Any tier name not itself a key of ``self.tier_adapters`` (an interim
        slot, e.g. ``"episodic_interim_<stamp>"``) is always episodic-shaped
        in this codebase, so it maps to ``self.tier_adapters["episodic"]`` —
        the same assumption :meth:`_training_output_dir` and every existing
        interim-mint call site make.

        Raises:
            KeyError: *tier* is neither a key of ``self.tier_adapters`` nor
                an interim adapter name (e.g. a main tier the operator has
                disabled).
        """
        if tier in self.tier_adapters:
            return self.tier_adapters[tier]

        from paramem.memory.interim_adapter import INTERIM_NAME_PREFIX

        if tier.startswith(INTERIM_NAME_PREFIX):
            return self.tier_adapters["episodic"]
        raise KeyError(tier)

    @staticmethod
    def _latest_stage(ledger: "StageLedger", tier: str, stage: str) -> "dict | None":
        """The most recent *stage* entry for *tier* in *ledger*, or ``None``.

        ``stages`` is append-only, so the latest entry of a kind is always
        the last matching one.
        """
        for entry in reversed(ledger.stages):
            if entry.get("stage") == stage and entry.get("tier") == tier:
                return entry
        return None

    def _classify_or_raise(self, *, increment: "TierIncrement") -> str:
        """Classify *increment*'s live-registry state against this event's
        own pre-image and payload; raise when it belongs to neither.

        The one call site for :func:`classify_partial_build` plus its
        :data:`FOREIGN` verdict's raise — both :meth:`_classify_ledger_tier`
        arms that need to know whether the current live registry is ours
        (a not-yet-built tier, or a torn retry of this event's own write)
        share this rather than hand-rolling the check-then-raise twice.

        Returns:
            ``"not_built"`` or ``"torn_own_write"`` — the live registry is
            ours; the caller decides what to do next.

        Raises:
            ConsolidationResumeBlocked: the live registry matches neither
                this event's pre-image nor its own payload (:data:`FOREIGN`)
                — something outside this event wrote it; refuse and hold.
        """
        reason = classify_partial_build(increment=increment, output_dir=self.output_dir)
        if reason == FOREIGN:
            raise ConsolidationResumeBlocked(tier=increment.tier, reason=reason)
        return reason

    def _classify_ledger_tier(
        self, *, ledger: "StageLedger", increment: "TierIncrement"
    ) -> "tuple[str, dict | None]":
        """The resume routine's per-tier classification.

        Returns ``("live", None)`` when this tier's ``tier_live`` entry
        verifies — already done, excluded from every bundle.  Otherwise,
        when a ``tier_written`` entry verifies, the CURRENT live registry is
        classified (:meth:`_classify_or_raise`) before the written skip is
        honored: a verifying written entry only proves the written slot copy
        is intact, never that the live registry beside it is still this
        event's own — a stranger could have rewritten the live registry
        after go-live while this event's ledger sat pending, and the written
        entry alone cannot see that.  An "ours" answer (``"not_built"`` — a
        revert to the pre-event state, or ``"torn_own_write"`` — a retried
        write of this event's own payload) proceeds into
        ``("written", <tier_written entry>)`` — the member keeps its gated
        payload and is carried into the joint go-live untouched, never
        retrained.  A foreign answer raises, same as the build arm below.
        With no verifying ``tier_written`` entry either, classifies the
        partial build the same way and returns ``("build", None)``.

        Args:
            ledger: This event's ledger, read fresh from disk by the caller.
            increment: The tier's assembled increment.

        Raises:
            ConsolidationResumeBlocked: the tier is neither done, written
                over an unmolested live registry, nor ours to rebuild
                (:data:`FOREIGN`) — refuse and hold.
        """
        from paramem.training import stage_ledger as _sl

        live_entry = self._latest_stage(ledger, increment.tier, "tier_live")
        if live_entry is not None and _sl.verify(live_entry):
            return "live", None

        written_entry = self._latest_stage(ledger, increment.tier, "tier_written")
        if written_entry is not None and _sl.verify(written_entry):
            self._classify_or_raise(increment=increment)
            return "written", written_entry

        self._classify_or_raise(increment=increment)
        return "build", None

    def _record_fold_telemetry(self, *, ledger: "StageLedger", kind: str, record: dict) -> None:
        """Write one record into the fold telemetry ring.

        A safe no-op when this loop has no telemetry dir (every experiment/
        test construction site) — always-on in production, never gated on
        ``debug``.  Binds the ring's ``cycle_stamp`` to run identity: the
        event's own extraction-entry ``completed_at``
        (:func:`~paramem.training.stage_ledger.extraction_entry`), written
        once when the event was staged and read back unchanged here, so a resumed event
        appends to the cycle its pre-crash phases opened rather than
        starting a second one, and two folds over an unchanged dataset never
        collapse into one growing entry.  Also a no-op when *ledger* carries
        no extraction entry — nothing to bind the record's run identity to.

        An ``OSError`` from the ring write itself is caught and logged, not
        raised — diagnostics must never block the fold that writes them
        (:mod:`paramem.server.fold_telemetry`'s own stated contract); this
        call site is inside ``tier_backup_scope``'s critical path, so a
        ring-write failure (e.g. a full disk) must not fail the training
        event.

        Args:
            ledger: The event's ``StageLedger`` — supplies the extraction
                entry's ``completed_at`` as ``cycle_stamp``.
            kind: Record-kind discriminator (``"backup_creation"`` |
                ``"tier_train"``).
            record: Caller-supplied integer/boolean/short-enum-string
                fields — never transcripts, facts, keys or speaker ids (the
                ring's own contract, :mod:`paramem.server.fold_telemetry`).
        """
        if self._telemetry_dir is None:
            return

        from paramem.server.fold_telemetry import record_fold_telemetry
        from paramem.training import stage_ledger as _sl

        entry = _sl.extraction_entry(ledger)
        if entry is None:
            return
        try:
            record_fold_telemetry(
                self._telemetry_dir,
                cycle_stamp=entry["completed_at"],
                kind=kind,
                record=record,
            )
        except OSError:
            logger.warning(
                "_record_fold_telemetry: failed to write kind=%s record to %s -- "
                "diagnostics only, the fold continues",
                kind,
                self._telemetry_dir,
                exc_info=True,
            )

    def _train_gate_write(
        self, *, increment: "TierIncrement", ledger: "StageLedger"
    ) -> "Path | None":
        """Train, gate and write one payload-bearing tier in the weights venue.

        ``ensure_adapter_matching`` runs FIRST, before
        :func:`~paramem.models.loader.tier_backup_scope` is even entered —
        an operator rank/config change is recreated here so the scope's own
        snapshot copy can never raise mid-scope on a shape mismatch.
        Training then runs inside ``tier_backup_scope`` — one tier's
        training covered by one backup, restored on any exception; nothing
        this event trains is ever activated live before the whole bundle
        writes, so the scope's only job is the unwind of a tier trained warm
        in place.  The shape-mismatch recreate above is this method's only
        adapter recreate; every event's transient staging slot inside
        ``train_adapter`` warm-starts uniformly (no cold-start arm for a
        RECONCILE event), never touching *tier* itself.  The gate
        (:meth:`_probe_recall` -> :meth:`_assert_tier_recall`) runs once, on
        the staged weights,
        immediately before the write — the design's one-probe rule; a
        rejection (:class:`RecallGateRejected`) propagates to the caller
        unchanged.

        Records fold telemetry (:meth:`_record_fold_telemetry`) once per
        tier for each of two kinds — ``"backup_creation"`` right after the
        scope's own snapshot, ``"tier_train"`` right after training returns
        and BEFORE the abort early-return below, so an aborted tier is still
        recorded.

        Returns:
            The written slot directory, or ``None`` when training produced no
            examples or the run aborted (yield-to-inference, graceful
            shutdown) — identical to a crash for this tier: nothing durable
            changed, and a resume rebuilds it from the same shadow
            artifacts.
        """
        from paramem.memory.persistence import write_tier_slot
        from paramem.models.loader import ensure_adapter_matching, tier_backup_scope

        tier = increment.tier
        adapter_config = self._tier_adapter_config(tier)

        # The tier's scratch dir comes from the ledger's own record
        # (fixed when the event was staged, see WorkingTier.scratch_dir),
        # never recomputed from self._training_output_dir here -- a resumed
        # event's live cycle_count may have drifted from the crashed
        # pass's, and training into a different scratch dir than the
        # crashed pass used would lose the epoch-level checkpoint resume
        # that directory exists for.
        output_dir = Path(ledger.tiers[tier]["scratch"])

        ensure_adapter_matching(self.model, adapter_config, tier)

        with tier_backup_scope(self.model, adapter_config, tier) as scope:
            self._record_fold_telemetry(
                ledger=ledger,
                kind="backup_creation",
                record={"tier": tier, **scope.vram},
            )

            metrics, recall_state = self._train_tier_adapter(
                increment.keyed,
                adapter_name=tier,
                adapter_config=adapter_config,
                training_config=self.training_config,
                output_dir=output_dir,
                run_name=f"consolidate-{tier}",
                phase_name=f"consolidate-{tier}",
                retain_scratch_until_external_commit=True,
            )
            aborted = bool(metrics is not None and metrics.get("aborted", False))

            _metrics = metrics or {}
            epochs_to_bind, steps_to_bind, hit_cap = _recall_bind_telemetry(
                recall_state, len(increment.keyed), _metrics.get("accum")
            )
            if aborted:
                # _recall_bind_telemetry cannot see the trainer's own abort
                # signal -- suppress hit_cap here, per that helper's stated
                # contract, rather than report a meaningless "ran to budget".
                hit_cap = None
            _telemetry_record: dict = {
                "tier": tier,
                "n_keys": len(increment.keyed),
                "aborted": aborted,
            }
            for _field in ("accum", "epochs", "init"):
                if _field in _metrics:
                    _telemetry_record[_field] = _metrics[_field]
            for _field, _value in (
                ("epochs_to_bind", epochs_to_bind),
                ("steps_to_bind", steps_to_bind),
                ("hit_cap", hit_cap),
            ):
                if _value is not None:
                    _telemetry_record[_field] = _value
            self._record_fold_telemetry(ledger=ledger, kind="tier_train", record=_telemetry_record)

        if metrics is None or aborted:
            return None

        ctx = self._build_write_context()
        with staged_weights(self.model, fallback_adapter=tier):
            probe = self._probe_recall(STAGING_ADAPTER, increment.keyed)
            on_recall_probe(list(probe.per_key), phase="staged", adapter_name=tier)
            self._assert_tier_recall(tier, probe)
            return write_tier_slot(ctx=ctx, increment=increment, stamp=ledger.stamp, mode="train")

    def _assert_increment_registry_bookkeeping_parity(self, increment: "TierIncrement") -> None:
        """Fail closed when one increment's known keys have no shadow entry
        and/or no shadow bookkeeping record.

        Reads only *increment*'s own ``registry`` / ``entries`` /
        ``bookkeeping`` fields — never ``self.store`` — so it is exercised
        identically whether *increment* came from a fresh staging pass or a
        resumed ledger's shadow tree.  Runs per increment, immediately
        before its write (:meth:`_write_built_tier`'s first act) — so a
        divergent increment is refused before any training, gating, or
        write for that tier.

        Predicate per key in ``increment.registry.list_known()`` (active ∪
        withheld — every key this member's shadow claims to know, not only
        the rows this event freshly minted or replayed):

        * bookkeeping is REQUIRED for every known key, active or withheld,
          and on every member including a rows-only one — a non-empty record
          must exist in ``increment.bookkeeping`` (an empty ``{}`` counts
          the same as absent).
        * a materialized entry is REQUIRED for every known key on a REBUILT
          member (``increment.rebuilt``) — mirrors
          :meth:`~paramem.memory.store.MemoryStore.adopt_increments`'s own
          entry-cache completeness check. A rebuilt member's registry never
          carries a withheld id (a marker ends at its own tier's rebuild, so
          ``list_known()`` on a rebuilt member is exactly ``list_active()``);
          a rows-only member's known keys — active or withheld — are never
          expected to have a materialized entry
          (``increment.entries`` stays empty for a rows-only member by
          construction — see :class:`~paramem.memory.increment.TierIncrement`).

        Checking ``list_known()`` rather than ``increment.keyed`` covers both
        a rows-only member (``increment.keyed == []``) and a withheld id
        (which ``keyed`` never lists).

        Args:
            increment: The tier's assembled increment
                (``build_tier_increment``'s output) — supplies ``registry``,
                ``entries`` and ``bookkeeping``.

        Raises:
            ~paramem.memory.store.BookkeepingInvariantViolation: One or more
                of *increment*'s known keys have no shadow bookkeeping
                record, or (on a rebuilt member) no shadow entry — raised
                via
                :func:`~paramem.memory.store.raise_bookkeeping_invariant_violation`
                with the context ``"pre-write parity"``, the same
                every-known-key-has-a-row invariant enforced everywhere else
                this exception is raised, checked here against the
                increment's shadow artifacts rather than the live store.
        """
        divergent: list[str] = []
        for key in increment.registry.list_known():
            bk = increment.bookkeeping.get(key)
            entry_ok = not increment.rebuilt or increment.entries.get(key) is not None
            if bk and entry_ok:
                continue
            divergent.append(key)

        if divergent:
            from paramem.memory.store import raise_bookkeeping_invariant_violation

            raise_bookkeeping_invariant_violation(
                increment.tier, sorted(divergent), "pre-write parity"
            )

    def _write_built_tier(
        self,
        *,
        increment: "TierIncrement",
        ledger: "StageLedger",
        state_dir: Path,
        mode: "Literal['train', 'simulate']",
    ) -> "tuple[Path | None, StageLedger]":
        """Write one tier classified ``"build"`` and record its ``tier_written``
        entry.  Returns ``(written_slot_path, updated_ledger)``.

        Runs :meth:`_assert_increment_registry_bookkeeping_parity` first,
        against *increment* itself — before any training, gating or
        writing for this tier.  A member with a payload
        (``increment.has_payload``) then trains, gates and writes via
        :meth:`_train_gate_write` in the weights venue, or writes its
        projected knowledge graph the same way (no training, no gate) in
        the disk venue — both venues write into a timestamped slot directory
        under the tier root (:func:`~paramem.memory.persistence.write_tier_slot`),
        so the written path this method returns is a directory in either
        venue.  A member with no payload — a rows-only member, or a tier
        rebuilt to zero keys — writes nothing; its ``tier_written`` entry
        hashes the shadow artifact set alone and records ``slot=None``.

        A gate rejection (:class:`RecallGateRejected`) disposes this
        event's ledger, extraction tree and training scratch and
        re-raises unchanged — live artifacts are never touched, and every
        contributing transcript stays pending.  A parity divergence
        (:class:`~paramem.memory.store.BookkeepingInvariantViolation`)
        propagates unchanged and leaves the ledger pending, exactly like a
        resume blocked on a FOREIGN tier: this is a defect in the staging
        pass itself, not a transient training outcome, and disposing the
        record would only invite an identical rebuild to fail the same way.
        """
        self._assert_increment_registry_bookkeeping_parity(increment)

        from paramem.memory.persistence import write_tier_slot
        from paramem.training import stage_ledger as _sl
        from paramem.training.stage_ledger import build_artifact_list, tier_written_stage

        tier = increment.tier
        written_slot: "Path | None" = None

        try:
            if increment.has_payload:
                if mode == "train":
                    written_slot = self._train_gate_write(increment=increment, ledger=ledger)
                else:
                    ctx = self._build_write_context()
                    written_slot = write_tier_slot(
                        ctx=ctx, increment=increment, stamp=ledger.stamp, mode="simulate"
                    )
        except RecallGateRejected:
            _sl.dispose(self._fold_state_dir)
            raise

        if increment.has_payload and written_slot is None:
            # Training aborted (yield-to-inference, graceful shutdown) rather
            # than completing — distinct from a member that legitimately
            # carries no payload.  No tier_written entry is recorded: writing
            # one here would let a later resume believe this tier written
            # (skip-written, never retrained) when it never actually trained.
            # The trainer's own checkpoint scratch (staging_resume.json,
            # checkpoint-N/) is untouched and is exactly what that resume
            # reads to continue training in place.  The caller stops the
            # whole call on this signal (written_slot is None while
            # increment.has_payload is True) rather than publishing a bundle
            # with a member that never trained.
            return None, ledger

        artifact_paths: list[Path] = []
        if written_slot is not None:
            artifact_paths.extend(sorted(p for p in written_slot.iterdir() if p.is_file()))

        shadow_dir = _sl.extraction_dir(state_dir, ledger.event) / "shadow" / tier
        for name in ("indexed_key_registry.json", "key_metadata.json", "keyed.json"):
            candidate = shadow_dir / name
            if candidate.exists():
                artifact_paths.append(candidate)

        entry = tier_written_stage(
            tier=tier,
            completed_at=datetime.now(timezone.utc).isoformat(),
            slot=written_slot,
            artifacts=build_artifact_list(artifact_paths),
        )
        _sl.write_stages(state_dir, ledger, [entry])
        ledger = replace(ledger, stages=tuple(ledger.stages) + (entry,))
        return written_slot, ledger

    def _ordered_publish_bundle(
        self, to_publish: "dict[str, TierIncrement]"
    ) -> "list[TierIncrement]":
        """This event's single go-live bundle, ordered by tier role.

        Every increment this event built goes live together, in ONE
        :func:`~paramem.training.go_live.publish_bundle` call — no live-
        registry read, no per-event inspection of what actually moved. The
        bundle's members go live together; no observer ever sees one member
        live while another is not, since the whole disk-write sequence
        precedes the single go-live ledger record and a crash before that
        record makes the resume republish the entire bundle from scratch.
        Order is fixed by role alone: ``semantic``, then ``episodic``, then
        every other tier by name.  A key moving from episodic to semantic
        is correct under this fixed order regardless of whether THIS event
        actually promoted anything: semantic's write landing before
        episodic's is the deliberate in-window severity choice — during the
        unobservable interval between the two on-disk writes, a
        transiently doubly-reachable key is the less severe intermediate
        than a transiently unreachable one; if nothing moved, the order
        costs nothing.

        Args:
            to_publish: Every increment this event still needs to take
                live, keyed by tier name.

        Returns:
            The ordered member list — ``semantic`` and ``episodic`` first
            (whichever of the two are present), then the rest sorted by
            name.
        """
        role_order = ("semantic", "episodic")
        remaining = dict(to_publish)
        ordered: "list[TierIncrement]" = []
        for tier in role_order:
            if tier in remaining:
                ordered.append(remaining.pop(tier))
        for tier in sorted(remaining):
            ordered.append(remaining[tier])
        return ordered

    def _ledger_all_tiers_live(self, ledger: "StageLedger") -> bool:
        """True when every tier *ledger* names has a verifying ``tier_live`` entry.

        Read-only against the ledger's own stages — never recomputed from
        the live store. Shared by :meth:`run_build_and_publish`'s pre-build
        verification pre-pass ("does this call even need to trust the
        shadow tree") and its post-publish completion check ("did this
        call finish the whole event"), so the two never compute the
        doneness question two different ways.
        """
        from paramem.training import stage_ledger as _sl

        return all(
            (entry := self._latest_stage(ledger, tier, "tier_live")) is not None
            and _sl.verify(entry)
            for tier in ledger.tiers
        )

    def run_build_and_publish(
        self,
        staged_event: "StagedEvent",
        *,
        router=None,
    ) -> dict:
        """The per-tier train -> gate -> write driver, plus the joint go-live.

        Consumes one :class:`StagedEvent` from :meth:`stage_event` (or
        a resumed event's own pending ledger, read fresh here) and takes
        every built tier live.  The second phase of every consolidation
        event — called by :meth:`run_consolidation_cycle` (interim) and
        :meth:`_stage_and_publish_full_event` (full, both venues).

        The ledger is always re-read from disk at entry, never trusted from
        *staged_event* across a process boundary: a fresh process re-entering
        this event has no phase-1 locals, only the ledger file.

        Before building any increment, verifies the ledger's ``"extraction"``
        entry (:func:`~paramem.training.stage_ledger.extraction_entry`)
        against on-disk bytes — unless every tier this ledger names already
        carries a verifying ``tier_live`` entry, in which case nothing needs
        the shadow tree at all and the check is skipped.  A missing or
        mismatched artifact means this event's staged content no longer
        exists to build from: the pending record is disposed and
        :class:`ConsolidationArtifactsMissing` raises naming the failed
        paths.  The contributing transcripts were never retired, so the
        next dispatch re-extracts them fresh — never silently reading the
        gap as a rows-only tier (an absent ``keyed.json`` reads identically
        to a legitimate rows-only member; verifying first is what tells
        them apart).

        Per built tier (``ledger.tiers``), the resume routine
        (:meth:`_classify_ledger_tier`) decides: a verified ``tier_live``
        entry means already done, excluded from the bundle; a verified
        ``tier_written`` entry (no ``tier_live``) means reused as-is, never
        retrained; neither means :func:`classify_partial_build` decides —
        :data:`FOREIGN` raises :class:`ConsolidationResumeBlocked` (refuse
        and hold, no phase runs, the ledger and every shadow artifact are
        left untouched), otherwise this tier trains, gates and writes now
        (:meth:`_write_built_tier`).

        Every not-yet-live tier is then ordered into this event's ONE go-live
        bundle (:meth:`_ordered_publish_bundle`) and taken live through
        exactly one :func:`~paramem.training.go_live.publish_bundle` call,
        always passing ``ledger.absorbed_interim_tiers`` (recorded when the
        event was staged, phase 1 — never recomputed from the live store,
        which could disagree with what the original staging pass actually
        absorbed) for a full-topology event's ring reap.  When every tier this call
        names was ALREADY live before this call ran (a resume finding
        nothing left to publish), no bundle publishes at all — the reap
        already ran, inside whichever call's bundle publish actually landed
        it (:func:`~paramem.training.go_live.publish_bundle` reaps before it
        records, so a durable ``tier_live`` entry proves the reap already
        happened; see that function's own docstring).

        Disposal is NOT performed here.  Retire-then-dispose ordering is
        the caller's own act, at its terminal — reading this call's
        ``"all_live"`` verdict and then, on ``True``, retiring the sessions
        the ledger recorded (read fresh from the ledger, still on disk at
        that point) before calling
        :func:`~paramem.training.stage_ledger.dispose`.  Disposing inside
        this method would let the ledger vanish before that retirement ever
        ran: a crash in the gap would strand the sessions pending with no
        record left for any resume to complete.

        Args:
            staged_event: The handoff from :meth:`stage_event` (or, on a
                resumed dispatch, a caller-reconstructed equivalent naming
                the same ``state_dir``/``event``).
            router: The live ``QueryRouter`` to reload once per bundle for a
                full-topology event (a full fold or a reconcile); ``None``
                for an interim event, whose own finalizer owns the reload.

        Returns:
            A summary dict: ``{"published_tiers": [...],
            "skipped_live_tiers": [...], "all_live": bool, "aborted": bool,
            "tier_bindings": dict[str, TierBinding]}`` — ``all_live`` is
            ``True`` only when every tier the ledger names verifies
            ``tier_live``; the caller's own terminal is what turns that
            into disposal.  ``tier_bindings`` is the publish verdict: one
            :func:`~paramem.adapters.registry_binding.verify_tier_binding`
            read per tier ``ledger.tiers`` names, populated only when
            ``all_live`` (``{}`` otherwise) — the post-event tier-health
            source a caller's unverified-tier incident sweep consumes
            instead of re-walking the whole adapter tree.  ``aborted`` is
            ``True`` exactly when a
            payload-bearing tier's own training yielded (to inference, or a
            graceful shutdown) mid-call — the same "nothing published"
            outcome as an event that staged no facts on ``published_tiers`` alone,
            but a caller deciding whether to retire contributing sessions
            MUST NOT conflate the two: an abort means those sessions'
            content was never actually learned, so it must stay pending for
            retry, never retired.

        Side effects:
            ``self.cycle_count`` advances by exactly one when *this call*
            observes ``all_live``, and is unchanged otherwise (an abort or a
            raise leaves it exactly where a resumed re-entry into the same
            event needs it, so a crash-resumed tier's scratch directory --
            named from ``self.cycle_count`` at train time -- stays the same
            directory across the crash).  This is the ONE increment site
            for the whole fold: every event kind and venue, fresh or
            resumed, converges here.

        Raises:
            ConsolidationArtifactsMissing: The ledger's extraction entry no
                longer verifies and at least one tier is not already live
                (raised after this event's ledger and extraction tree are
                disposed).
            ConsolidationResumeBlocked: A not-yet-done tier classified
                FOREIGN.
            RecallGateRejected: A tier's staged weights failed the recall
                gate (raised after this event's ledger and extraction tree
                are disposed).
        """
        from paramem.memory.increment import build_tier_increment as _build_tier_increment
        from paramem.training import stage_ledger as _sl
        from paramem.training.go_live import publish_bundle

        state_dir = staged_event.state_dir
        ledger = _sl.read_ledger(state_dir)
        if ledger is None:
            raise RuntimeError(
                f"run_build_and_publish: no stage ledger at {state_dir} for event "
                f"{staged_event.event!r} -- stage_event must run (or resume) before this call"
            )

        if not self._ledger_all_tiers_live(ledger):
            extraction_entry = _sl.extraction_entry(ledger)
            if extraction_entry is None or not _sl.verify(extraction_entry):
                missing = _sl.missing_artifacts(extraction_entry) if extraction_entry else []
                _sl.dispose(state_dir)
                raise ConsolidationArtifactsMissing(event=ledger.event, missing=missing)

        mode: "Literal['train', 'simulate']" = "train" if ledger.venue == "weights" else "simulate"
        shadow_root = _sl.extraction_dir(state_dir, ledger.event) / "shadow"

        published_tiers: list[str] = []
        skipped_live_tiers: list[str] = []
        to_publish: "dict[str, TierIncrement]" = {}
        written_slots: "dict[str, Path | None]" = {}
        aborted = False

        for tier, tier_meta in ledger.tiers.items():
            # A tier already verified live needs no increment at all -- its
            # shadow tree is never read (the invariant the verification
            # pre-pass above trades on: an all-live event never touches the
            # shadow tree).
            live_entry = self._latest_stage(ledger, tier, "tier_live")
            if live_entry is not None and _sl.verify(live_entry):
                skipped_live_tiers.append(tier)
                continue

            increment = _build_tier_increment(
                tier=tier,
                adapter_name=tier_meta["adapter"],
                pre_sha=tier_meta["pre_sha"],
                shadow_dir=shadow_root / tier,
            )
            status, written_entry = self._classify_ledger_tier(ledger=ledger, increment=increment)

            if status == "written":
                written_slots[tier] = _sl.written_slot_path(written_entry)
                to_publish[tier] = increment
                continue

            written_slot, ledger = self._write_built_tier(
                increment=increment, ledger=ledger, state_dir=state_dir, mode=mode
            )
            if increment.has_payload and written_slot is None:
                # Training aborted for this tier -- _write_built_tier recorded
                # no tier_written entry for it (see its own docstring).  Stop
                # the whole call here: go-live is one joint act over the
                # bundle, so nothing already written in THIS call (or before
                # it) may be published without this tier.  A tier this call
                # already written keeps its durable tier_written entry and its
                # inert slot; a later dispatch resumes from exactly that
                # state and retrains only what never written.
                to_publish = {}
                aborted = True
                break
            written_slots[tier] = written_slot
            to_publish[tier] = increment

        if to_publish:
            bundle = self._ordered_publish_bundle(to_publish)
            ctx = self._build_write_context(extra_tiers=[inc.tier for inc in bundle])
            publish_bundle(
                bundle,
                ctx=ctx,
                ledger=ledger,
                written_slots={inc.tier: written_slots.get(inc.tier) for inc in bundle},
                router=router,
                absorbed_interim_tiers=ledger.absorbed_interim_tiers,
            )
            published_tiers.extend(inc.tier for inc in bundle)
            ledger = _sl.read_ledger(state_dir)

        all_live = self._ledger_all_tiers_live(ledger)

        # The one cycle_count increment site, on the event spine: every
        # event kind (interim, full) and every venue (train, simulate) --
        # fresh dispatch or resumed -- converges here, the single point
        # where an event's completion is known.  Advances exactly once per
        # COMPLETED event, never on an abort or a gate rejection (those
        # resume into the SAME cycle number, which is what keeps a
        # crash-resumed tier's scratch directory -- named from
        # ``self.cycle_count`` at the time it trains -- identical across the
        # crash).  A call that finds every tier already live (nothing left
        # to publish) still advances it: that IS this event's completion,
        # observed for the first time in this process.
        if all_live:
            self.cycle_count += 1

        # No end-of-event ring sweep: publish_bundle reaps the absorbed ring
        # BEFORE it records the bundle's tier_live entries (see that
        # function's own docstring), so a durable tier_live entry already
        # proves the reap ran.  A resume that finds every tier live is a
        # resume that finds the ring already gone -- there is no reachable
        # state where the ring survives an all-live ledger.

        # Adopt this event's promotion decisions into the in-process
        # promoted_keys set ONLY now that the whole bundle is confirmed
        # live -- see _promote_working_keys' own docstring for why an
        # earlier merge (while the event was still staging) would poison the set on an
        # abort or a gate rejection.  Discarded either way (never leaked
        # into a later, unrelated event) whether or not this event went
        # live.
        if self._pending_promoted_keys:
            if all_live:
                self.promoted_keys.update(self._pending_promoted_keys)
            self._pending_promoted_keys = set()

        # The publish verdict: one verify_tier_binding read per tier THIS
        # event's ledger names -- never a whole-tree walk -- only when
        # all_live, since that verdict is what proves every one of them
        # (published this call or already live going in) is genuinely
        # live.  This is the post-event tier-health source: the caller's
        # unverified-tier incident sweep consumes it directly instead of
        # re-walking the adapter tree fresh.  Sound under the single-writer
        # architecture -- tier state changes only at publish, and this
        # event's own all_live verdict just proved every member of its own
        # bundle (a full event's bundle covers every main tier, zero-key
        # fixed point included; an interim event's covers its one slot).
        # An untouched tier outside this ledger is simply not in the map --
        # its own incident, if any, stays until that tier's own next
        # publish or the next boot check.
        tier_bindings: "dict[str, TierBinding]" = {}
        if all_live:
            from paramem.adapters.registry_binding import verify_tier_binding
            from paramem.memory.interim_adapter import adapter_slot_root_for_name

            tier_bindings = {
                tier: verify_tier_binding(tier, adapter_slot_root_for_name(self.output_dir, tier))
                for tier in ledger.tiers
            }

        return {
            "published_tiers": published_tiers,
            "skipped_live_tiers": skipped_live_tiers,
            "all_live": all_live,
            "aborted": aborted,
            "tier_bindings": tier_bindings,
        }
