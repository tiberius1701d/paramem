"""Consolidation loop orchestrator.

Runs the full consolidation pipeline: extract graph from session,
merge into cumulative graph, score for promotion/decay, train
episodic and semantic adapters.
"""

import hashlib
import logging
import random
import secrets
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal, Optional

import torch
from torch.utils.data import Dataset

from paramem.config.taxonomy import fallback_relation_type, relation_types
from paramem.graph.extraction_pipeline import ExtractionConfig, ExtractionPipeline
from paramem.graph.merger import GraphMerger
from paramem.graph.phase_trace import extraction_trace, phase_trace
from paramem.graph.reconstruct import reconstruct_graph
from paramem.graph.relation_prep import (
    attr_predicate,
    partition_relations,
)
from paramem.graph.schema import Relation, SessionGraph
from paramem.memory.entry import (
    assign_keys,
    build_registry,
    entry_simhash,
    format_entry_training,
)
from paramem.models.loader import atomic_save_adapter, measured_adapter_init_state, switch_adapter
from paramem.server.fold_telemetry import record_fold_telemetry
from paramem.training import graph_tier
from paramem.training.donor import DONOR_BUILD_ADAPTER_NAME, DONOR_KEY_FLOOR
from paramem.training.key_registry import KeyRegistry
from paramem.training.thermal_throttle import ThermalPolicy
from paramem.training.trainer import TrainingHooks
from paramem.utils.artifacts import (
    debug_run,
    on_cycle_end,
    on_extraction_end,
    on_fold_assignments,
    on_fold_graph,
    on_main_adapters_saved,
    on_recall_probe,
    on_removal_ledger,
    on_session_extracted,
    on_tier_delta,
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
    from paramem.training.early_stop import _EarlyStopState

logger = logging.getLogger(__name__)

# Frozen set of valid relation types drawn from the single source of truth in
# paramem.config.taxonomy so that the stage-2 clamp stays in sync with the
# Pydantic Relation schema (_RelationType = Literal[relation_types()]).
_VALID_RTYPES: frozenset[str] = frozenset(relation_types())
_FALLBACK_RTYPE: str = fallback_relation_type()

# Synthetic session-id sentinels used by the three fold/re-merge paths that
# call GraphMerger.merge_relations or GraphMerger.merge with a pseudo-id rather
# than a real session identifier.  The harvest filter in
# _build_all_edge_entries_into subtracts these from edge["sessions"] so that
# the deferred-write record carries ONLY real contributing session ids — synthetic
# sentinels are subtracted in _build_all_edge_entries_into.
_SYNTHETIC_SESSION_IDS: frozenset[str] = frozenset(
    {
        "__full_consolidation_recon__",
        "__interim_pending_sessions__",
        "__simulate_consolidation_merge__",
        "__graph_enrichment__",
    }
)

# Ownership cue prepended to the local extraction slot for document sources
# with a known speaker name — a soft, in-prompt signal raising first-pass
# model compliance with the exact-full-name speaker rewrite that
# _stamp_speaker_entity applies deterministically regardless (extractor.py).
# Not a prompt file: this is a caller-layer prepend onto the slot content,
# per the single-topology one-prompt-pair design (extraction_pipeline.py
# kwargs() docstring).
OWNERSHIP_CUE = (
    "[Document provided by {name} ({sid}). Statements describing {name} are facts about {sid}.]\n\n"
)


def _fingerprint_entries(entries: "list[dict]") -> str:
    """SHA-256 fingerprint over the sorted ``(key, subject, predicate, object)`` tuples.

    Content-only identity signal for a keyed-entry list, stored in
    ``fold_resume.json`` by :meth:`ConsolidationLoop._persist_fold_assignment`
    (both the ``main_tiers`` and ``interim_slot`` fold branches call this same
    helper — no duplicated fingerprint loop).  Sorting makes the result
    independent of entry order, so it detects a genuine content change (not
    mere reordering) between the pre-crash and post-crash ``train_assignment``.

    Distinct from ``paramem.training.trainer._fingerprint_dataset``, which
    fingerprints the *tokenized* training examples (the signal that actually
    gates HF Trainer's ``resume_from_checkpoint`` inside ``trainer.py``) — this
    is the coarser, pre-tokenization SPO fingerprint.

    Args:
        entries: Keyed-entry dicts, each carrying at least ``key``,
            ``subject``, ``predicate``, ``object``.

    Returns:
        Hex-encoded SHA-256 digest string.
    """
    fp = hashlib.sha256()
    for spo in sorted(
        (e.get("key", ""), e.get("subject", ""), e.get("predicate", ""), e.get("object", ""))
        for e in entries
    ):
        fp.update(repr(spo).encode("utf-8"))
    return fp.hexdigest()


def _relation_to_entry_dict(r: "Relation") -> dict:
    """Project a single ``Relation`` into the ``{subject, predicate, object,
    relation_type}`` shape used to seed interim-tier entry dicts.

    ``predicate`` is canonicalized via :func:`~paramem.utils.identity.canonical`
    so interim-tier entries match the identity form the merger stamps onto the
    cumulative edge (``merger.py:710``) — without this, an interim entry built
    straight from ``session_graph.relations``/``proc_graph.relations`` carries
    the raw extraction surface (e.g. ``"Works_At"``) while the full-cycle edge
    entry (:meth:`ConsolidationLoop._build_all_edge_entries_into`) carries the
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


def _persisted_from_entry_and_rec(entry: "dict", tier: str, rec: "dict | None") -> "dict":
    """Build one ``fold_resume.json`` train_assignment entry for the interim fold.

    Serialize half of the round-trip pair with :func:`_rec_from_persisted`
    (the deserialize half) — together they are the single place the
    interim-slot ``deferred_writes``/``new_keyed_interim`` "rec" shape (built
    by the graph-walk in
    :meth:`ConsolidationLoop._build_all_edge_entries_into` — see the
    cross-link comment at that ``rec = {...}`` construction) is projected to
    and reconstructed from the persisted marker.  *entry* is the uniform
    ``tier_keyed`` shape (``{key, subject, predicate, object, speaker_id}``);
    *rec* is the matching deferred-write record — present only for keys
    newly minted THIS cycle (``None`` for anti-forgetting-replay entries,
    which carry no rec).

    Args:
        entry: One uniform ``tier_keyed`` entry.
        tier: ``"episodic"`` or ``"procedural"`` — which ``_tier_keyed`` list
            *entry* came from.
        rec: The matching ``deferred_writes`` record, or ``None`` when
            *entry* is an existing (already-keyed) key.

    Returns:
        *entry* combined with ``"tier"`` and, when *rec* is not ``None``,
        ``relation_type``/``session_ids``/``last_seen``/``first_seen``.
    """
    persisted = dict(entry)
    persisted["tier"] = tier
    if rec is not None:
        persisted["relation_type"] = rec["relation_type"]
        persisted["session_ids"] = rec["session_ids"]
        persisted["last_seen"] = rec["last_seen"]
        persisted["first_seen"] = rec["first_seen"]
    return persisted


def _rec_from_persisted(pe: "dict") -> "dict":
    """Rebuild one ``new_keyed_interim``/``deferred_writes`` "rec" record from
    a persisted (``fold_resume.json``) train_assignment entry.

    Deserialize half of the round-trip pair with
    :func:`_persisted_from_entry_and_rec` (the serialize half) — see that
    function's docstring for the shared contract.  Callers must first check
    that *pe* actually carries ``"relation_type"`` (only entries newly minted
    pre-crash do — see :func:`_persisted_from_entry_and_rec`'s *rec* param);
    this function does not gate on that itself and will ``KeyError`` if called
    on a replay entry.

    Args:
        pe: One enriched entry from the persisted ``train_assignment`` list.

    Returns:
        A rec dict shaped like one element of ``deferred_writes`` — no
        ``canon_subj``/``canon_obj``: the interim commit window never reads
        those two graph-walk-only fields, so they are not part of the
        round-trip contract.
    """
    entry = {
        "key": pe["key"],
        "subject": pe["subject"],
        "predicate": pe["predicate"],
        "object": pe["object"],
        "speaker_id": pe["speaker_id"],
    }
    return {
        "entry": entry,
        "tier": pe.get("tier", "episodic"),
        "predicate": pe["predicate"],
        "relation_type": pe["relation_type"],
        "speaker_id": pe["speaker_id"],
        "session_ids": pe.get("session_ids", []),
        "last_seen": pe.get("last_seen", ""),
        "first_seen": pe.get("first_seen", ""),
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
    ``"aborted"`` key) — a thermal-throttle or operator-pause abort also
    leaves ``stop_epoch=None`` and is indistinguishable from a genuine
    "ran to the full budget without binding" from this function's inputs
    alone. Callers MUST additionally check the trainer's abort flag and
    suppress the ``hit_cap`` field on the abort path — see the two call
    sites in ``ConsolidationLoop._run_fold``, which add an ``aborted``
    field to the record for exactly this reason.

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


class TrialActiveError(RuntimeError):
    """Raised by ConsolidationLoop.guard_trial_state when a migration TRIAL is active.

    Bubbles up to /scheduled-tick and /consolidate handlers, which return
    409 trial_active.  Experiment scripts that do not carry server _state
    never trigger this error (guard is a no-op when state is None).
    """


class RecallGateRejected(RuntimeError):
    """Raised by :meth:`ConsolidationLoop._verify_saved_adapter_from_disk` when the
    post-save recall probe lands below ``recall_sanity_threshold``.

    A deterministic quality verdict, NOT a crash: the adapter trained and saved
    successfully, and the probe simply did not reach the threshold.  The interim
    fold catches this specific type, rolls back the cycle's store mutations, and
    returns ``mode="recall_failed"`` with the contributing session ids — the
    normal-return contract that ``app.py``'s retry bookkeeping is written
    against ("crash != recall failure", ``app.py:14154-14156``).  Letting it
    propagate as a bare exception skips that bookkeeping entirely, which leaves
    the durable retry counter at zero and the release valve unreachable.

    Subclasses ``RuntimeError`` so existing broad handlers on the main-tier
    path keep their current behaviour.
    """

    def __init__(self, message: str, *, recall_rate: float, threshold: float):
        super().__init__(message)
        self.recall_rate = recall_rate
        self.threshold = threshold


class AbortedDuringConsolidation(Exception):
    """Raised by the train fold (:meth:`ConsolidationLoop.consolidate`) when training
    is aborted mid-tier.

    The caller (app.py ``_run_full_cycle``) catches this, restores all three
    production tiers from their ``<tier>_backup`` slots via
    ``copy_adapter_weights``, skips the atomic finalize step (registry rewrite
    → persist → interim purge → router reload), and logs the cycle as
    ``mode="aborted"``.  Partial progress is lost but VRAM state is consistent
    with the pre-cycle baseline.
    """


@dataclass(frozen=True)
class FoldScope:
    """Immutable descriptor that parameterizes one invocation of
    :meth:`ConsolidationLoop._run_fold`.

    A frozen dataclass (not a mode string) so dispatch is structural — no
    ``mode == "simulate"`` / ``mode == "train"`` literals inside
    :meth:`_run_fold` or its wrappers (the mode-fork-guard enforces this).

    Attributes:
        name: Human-readable label (``"interim"`` | ``"full"``).  Used in log
            messages and debug artifacts only; has no dispatch semantics.
        source: **The venue discriminator.**  ``"weights"`` is the train venue:
            adapter weights exist, so the fold reconstructs from them, trains,
            and saves them.  ``"disk"`` is the simulate venue: no weights exist,
            so the weight-only blocks are skipped and the persist medium is
            per-tier ``graph.json``.  Both venues read the same
            :class:`~paramem.memory.store.MemoryStore` for their fold input —
            ``source`` selects the weight *probe*, never the input medium.
            Also forwarded to
            :meth:`~ConsolidationLoop._materialize_consolidation_graph`.
        persist: Persist venue, dispatched at the end of the spine.

            - ``"interim_slot"`` — call
              :func:`~paramem.memory.persistence.commit_tier_slot` (interim cycle).
            - ``"main_tiers"`` — full fold.  Writes adapter weights via
              :meth:`~ConsolidationLoop._save_adapters` when ``source ==
              "weights"``, per-tier ``graph.json`` projected from the store
              otherwise.
        tier: Target adapter name for the interim scope (e.g.
            ``"episodic_interim_YYYYMMDDTHHMM"``).  ``None`` for the full fold
            (all tiers are rebuilt).
        defer: Forwarded as the ``defer`` flag to
            :meth:`~ConsolidationLoop._build_all_edge_entries_into`.  ``True``
            for the interim slot (atomicity: registry entry deferred until after
            training succeeds); ``False`` for the full fold.
        tag_new: Forwarded as the ``tag_new`` flag to
            :meth:`~ConsolidationLoop._build_all_edge_entries_into`.  ``True``
            for the interim slot (new-entry tracking); ``False`` for the full fold.
        normalize: When ``True``, run the whole-graph normalization pass via
            :meth:`~ConsolidationLoop._refine_consolidation_graph`.  Pinned
            ``False`` for the interim scope, structurally, like ``enrich``.
        enrich: When ``True``, run cloud graph enrichment via
            :meth:`~ConsolidationLoop._refine_consolidation_graph`.  Pinned
            ``False`` for the interim scope regardless of
            ``refinement_enrichment`` / ``cloud_enabled`` — graph-tier
            enrichment is a full-fold-only pass.  Session-tier cloud
            enrichment already runs at extraction time over the anonymized
            transcript (:mod:`paramem.graph.stage_enrich`), which has
            strictly better context than a graph-only pass would at interim
            scope; the graph-only pass's measured interim output was 11/15
            predicate paraphrases (2026-07-28); cross-session inference over
            the cumulative graph remains the full fold's job.
        promote: When ``True``, call
            :meth:`~ConsolidationLoop._promote_mature_keys_inline` after the
            Refine stage.  ``True`` for the full fold in BOTH venues —
            promotion is a pure store operation with no weight dependency.
        subtractive_scope: Forwarded to
            :meth:`~ConsolidationLoop._apply_subtractive_removals_to_store`
            (``"interim"`` | ``"fold"``).
        consume_pending: When ``True``, the fold snapshots the pending-session
            relations sitting in ``merger.graph`` via
            :meth:`~ConsolidationLoop._capture_pending_relations` and feeds them
            to :meth:`~ConsolidationLoop._materialize_consolidation_graph`
            through its ``extra_relations`` channel, so they survive the graph
            reset.  ``True`` for every interim cycle (the pending session IS the
            cycle's content) and for the full fold in the
            ``max_interim_count == 0`` consume-pending mode the server selects.
            ``False`` means no supplemental relations enter the merge.
        keys_from: **The key source of a ``main_tiers`` fold** — which of the
            store's registered tiers this fold owns, resolved live by
            :meth:`~ConsolidationLoop._fold_active_keys`.

            - ``"all_tiers"`` — every active key, interim slots included.  The
              slots are folded into main and reaped afterwards.
            - ``"main_tiers"`` — ``episodic`` / ``semantic`` / ``procedural``
              only.  Interim keys never enter the merge, so they are neither
              retrained nor drift-partitioned, and the slots are left on disk:
              a fold that did not absorb them must not reap them.  Interim
              disposal follows from this field and is NOT a second flag.

            Ignored by the ``interim_slot`` scope, which is scoped to its own
            slot by ``tier`` instead.
    """

    # --- identity / dispatch ---
    name: str  # "interim" | "full"  (log/debug label only)
    source: "Literal['weights', 'disk']"
    persist: "Literal['interim_slot', 'main_tiers']"

    # --- materialize scoping ---
    tier: "str | None" = None
    defer: bool = False
    tag_new: bool = False

    # --- refine gate ---
    normalize: bool = False
    enrich: bool = False

    # --- spine stage gates ---
    promote: bool = False
    subtractive_scope: "Literal['interim', 'fold']" = "fold"

    # --- pending capture ---
    consume_pending: bool = False  # merge pending-session relations in-fold

    # --- key source (main_tiers scope) ---
    keys_from: "Literal['all_tiers', 'main_tiers']" = "all_tiers"

    @property
    def cold_init(self) -> bool:
        """Whether the main-tier fold preamble must delete and recreate each tier.

        Derived, not a field: ``persist == "main_tiers" and keys_from ==
        "main_tiers"`` is the structural identity of RECONCILE
        (``/reconsolidate``) — the one door that narrows a
        main-tier fold to the main tiers' own keys, bound at the single
        arbitrator site (``paramem.server.app._dispatch_consolidation``,
        where ``_keys_from = "main_tiers" if action is
        ConsolidationAction.RECONCILE else "all_tiers"``). Every other
        caller — the scheduled FULL fold and every interim cycle — leaves
        ``keys_from`` at its ``"all_tiers"`` default (or the field does not
        apply to the ``interim_slot`` persist venue at all), so
        ``cold_init`` is ``False`` there: the main-tier preamble keeps a
        resident, config-matching tier's weights (warm init, the default),
        and :func:`~paramem.models.loader.ensure_adapter_matching` recreates
        cold only on a genuine config mismatch or a first-boot absence —
        never as a blanket policy. When ``True``, the preamble reproduces
        today's unconditional delete+recreate exactly (RECONCILE's cold
        rebuild semantics; no new behaviour is invented).
        """
        return self.persist == "main_tiers" and self.keys_from == "main_tiers"


class ConsolidationLoop:
    """Manages the full consolidation pipeline across sessions.

    Each cycle:
    1. Extract knowledge graph from session transcript
    2. Merge into cumulative graph
    3. Score nodes for promotion/decay
    4. Generate QA training pairs from graph
    5. Train episodic adapter (new + replay)
    6. Train semantic adapter (promoted + replay)
    7. Decay unreinforced episodic memories
    """

    # Class-level default so instances built via ``object.__new__`` (test
    # harnesses that skip ``__init__`` to avoid loading a model — see
    # tests/test_procedural.py, tests/test_run_consolidation_cycle.py,
    # tests/test_adapter_verification.py) still resolve this attribute.
    _telemetry_dir: "Path | None" = None

    # Same reason: ``cloud_enabled`` must resolve on instances built via
    # ``object.__new__``.  Default OFF — a harness that skips ``__init__``
    # gets no cloud egress unless it says so.
    cloud_enabled: bool = False

    def __init__(
        self,
        model,
        tokenizer,
        consolidation_config: ConsolidationConfig,
        training_config: TrainingConfig,
        episodic_adapter_config: AdapterConfig,
        semantic_adapter_config: AdapterConfig,
        *,
        memory_store,
        procedural_adapter_config: Optional[AdapterConfig] = None,
        wandb_config: Optional[WandbConfig] = None,
        output_dir: str | Path = "outputs/phase3",
        extraction_temperature: float = 0.0,
        extraction_max_tokens: int = 8192,
        extraction_plausibility_max_tokens: int = 8192,
        extraction_anonymize_token_envelope: int = 8192,
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
        extraction_scrub: set[str] | frozenset[str],
        extraction_correction_entity_types: set[str] | frozenset[str] | None = None,
        graph_config: Optional[GraphConfig] = None,
        cloud_enabled: bool = False,
        graph_enrichment_neighborhood_hops: int = 2,
        graph_enrichment_max_entities_per_pass: int = 50,
        state_provider=None,
        thermal_policy: ThermalPolicy | None = None,
        keep_prior_slots: int = 3,
        telemetry_dir: str | Path | None = None,
        incidents_state_dir: str | Path | None = None,
    ):
        # Optional callable that returns the server ``_state`` dict.  When
        # provided, ``run_cycle`` calls ``self.guard_trial_state(state_provider())``
        # at entry to block new consolidation cycles during a migration TRIAL.
        # Experiment scripts pass nothing (default ``None``) so the guard is a
        # no-op and experiment paths are unaffected.
        self.state_provider = state_provider
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
        # ``_save_adapters`` calls ``save_adapter``.
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
        self.shutdown_requested = False  # set by signal handler to stop training
        # Thermal policy is supplied by the caller (None when
        # consolidation.training_temp_limit <= 0, the default).  Live-server
        # only by construction: experiments and tests that don't override the
        # default get None and the throttle is never installed at the
        # train_adapter call site.  The schedule config (which actually
        # carries the thermal fields) lives in server.config and is not
        # reachable from this module — the loop accepts the precomputed
        # ThermalPolicy instead of re-deriving it.
        self._thermal_policy = thermal_policy
        self.episodic_config = episodic_adapter_config
        self.semantic_config = semantic_adapter_config
        self.procedural_config = procedural_adapter_config
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
        # is added per-write via :meth:`snapshot_dir_for` (2026-05-15
        # hierarchy spec — paths.debug/episodic/[interim_<stamp>/]cycle_<N>/
        # run_<run_id>/).  ``self.snapshot_dir`` (legacy attribute) preserved
        # for the HF-Trainer checkpoint dir builder in
        # :meth:`_training_output_dir`, which is out of scope for the
        # debug-layout cleanup.
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
        # Cloud egress PII anonymization scope (``extraction_scrub``) is
        # sourced at the bootstrap call site from
        # ``ServerConfig.sanitization.scrub`` so consolidation honours the
        # same operator policy as inference-time cloud egress.  Required —
        # no implicit default anywhere below the config layer (the model's
        # anonymizer prompt is the sole scope authority; a graph-layer
        # fallback constant would be a duplicated, out-of-layer privacy
        # policy — see ``paramem/graph/placeholders.py``'s
        # ``_build_anonymization_mapping`` docstring).
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
                scrub=extraction_scrub,
                correction_entity_types=extraction_correction_entity_types,
                cloud_enabled=cloud_enabled,
            ),
            prompts_dir=prompts_dir,
            model_name=model_name,
        )

        # Graph-level cloud enrichment knobs (Task #10).
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

        # Ensure both adapters exist on the model
        self.model = self._ensure_adapters()

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
        # DONOR_KEY_FLOOR (= width + 1) rather than 1, and the max() below
        # (against the store's own high-water key) can only raise it further —
        # never below the floor.
        self._indexed_next_index: int = DONOR_KEY_FLOOR
        self._procedural_next_index: int = DONOR_KEY_FLOOR
        # Derive next-index counters from every KNOWN key in the store — active
        # AND stale (paramem.training.donor's DONOR_KEY_FLOOR is a starting
        # value, not a replacement for this scan). Root fix: scanning
        # all_active_keys() alone excludes the stale partition, so a
        # soft-staled highest key plus a restart would re-mint its id and
        # set_simhash would then route the new fingerprint into the stale
        # record (paramem.training.key_registry — the stale slot has no live
        # simhash reader, so the collision surfaces as a silent recall miss on
        # the NEW key, not an error). all_known_keys() is active ∪ stale, so a
        # stale highest key still bumps the floor and the id is never reissued.
        # The caller is responsible for having hydrated registries before
        # this point (via ``MemoryStore.load_registries_from_disk`` or by
        # injecting the lifespan-loaded store).
        if self.store.replay_enabled:
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

        self.cycle_count = 0

        # Keys already promoted (prevent re-promotion after restart)
        self.promoted_keys: set[str] = set()

        # BackgroundTrainer reference — wired after construction by the server
        # lifespan or create_consolidation_loop caller.  When set,
        # _build_training_hooks routes through bt.training_hooks_for_job so
        # the abort event is included in the shutdown predicate.
        self._bg_trainer = None

    def _build_training_hooks(
        self,
        *,
        on_step_yield: "Optional[Callable[[int], None]]" = None,
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
            on_step_yield: Passed through to ``TrainingHooks`` unchanged.
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
                on_step_yield=on_step_yield,
                on_epoch_persist=on_epoch_persist,
                on_save_persist=on_save_persist,
            )
        return TrainingHooks(
            on_shutdown_check=base,
            on_step_yield=on_step_yield,
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

    def guard_trial_state(self, state: dict | None) -> None:
        """Raise TrialActiveError when a migration TRIAL is in progress.

        Called at the top of run_cycle and from /scheduled-tick and
        /consolidate handlers to block new consolidation cycles while the
        operator reviews trial results.

        Parameters
        ----------
        state:
            The server ``_state`` dict, or ``None`` for experiment scripts
            that do not carry server state.  When ``None``, this method is
            a no-op so experiment paths are unaffected.

        Raises
        ------
        TrialActiveError
            When ``state["migration"]["state"] == "TRIAL"``.
        """
        if state is None:
            return
        migration = state.get("migration")
        if migration is None:
            return
        if migration.get("state") == "TRIAL":
            raise TrialActiveError(
                "consolidation blocked: a migration TRIAL is active. "
                "Use POST /migration/accept or POST /migration/rollback to proceed."
            )

    def seed_key_metadata(self, metadata: dict) -> None:
        """Restore loop-level state from persisted key_metadata.json.

        Restores ``cycle_count`` and ``promoted_keys``.  Keys in the metadata
        file whose tier registry is not on disk are treated as orphans and
        dropped — the slot is the source of truth for active keys.  The orphan
        count is logged so callers can distinguish a clean restore from one
        where stale metadata entries were pruned.

        Per the wipe invariant (2026-05-14): ``key_metadata.json`` is
        bookkeeping, not a recovery source.

        Per-key bookkeeping (``speaker_id``, ``relation_type``,
        ``reinforcement_count``, ``last_reinforced_cycle``, ``last_seen``) is
        owned by :attr:`MemoryStore._bookkeeping` and loaded by
        :meth:`MemoryStore.load_bookkeeping_from_disk` at lifespan boot.
        This method does NOT touch the store's bookkeeping — that was the
        ``setdefault_entry`` parasitic write that created payload-less stubs
        and caused cache-off hallucination (``preload_cache=false`` bug).
        """
        self.cycle_count = metadata.get("cycle_count", 0)
        orphan_count = 0
        for key, key_meta in metadata.get("keys", {}).items():
            tier = self.store.tier_for_known_key(key)
            if tier is None:
                # No tier knows this key (not active, not stale) — slot was
                # wiped or never existed.  Drop the metadata entry; the next
                # _save_key_metadata write will not re-emit it.
                orphan_count += 1
                continue
        # promoted_keys is similarly slot-owned — drop entries whose tier is
        # gone so the next save doesn't re-emit them.  A promoted-then-staled
        # key is still legitimately known; retain its promotion record.
        raw_promoted = set(metadata.get("promoted_keys", []))
        self.promoted_keys = {k for k in raw_promoted if self.store.is_known(k)}
        if orphan_count:
            logger.info(
                "seed_key_metadata: dropped %d orphan key(s) (metadata present, no tier registry)",
                orphan_count,
            )
        logger.info(
            "Seeded key metadata: cycle=%d, %d promoted",
            self.cycle_count,
            len(self.promoted_keys),
        )

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
                ``"social"``).  Defaults to ``"factual"`` for legacy callers
                that pre-date this field; pass explicitly at every new site.

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

    def _ensure_store(self) -> None:
        """Auto-create a :class:`MemoryStore` when the loop has no store yet.

        Called in production by the fold ledger-attribution method
        (``_attribute_ledger_removals_to_tiers``) to guarantee a store exists
        before attributing removal entries.  Also guards bare-loop construction
        paths (e.g. ``object.__new__`` in tests) that set store-dependent
        attributes before ``__init__`` runs."""
        if not hasattr(self, "store") or self.store is None:
            from paramem.memory.store import MemoryStore

            self.store = MemoryStore(replay_enabled=True)

    def _all_active_keys(self) -> list[str]:
        """Every active key across every registered tier — order is tier-then-insertion."""
        return self.store.all_active_keys()

    def _fold_active_keys(self, scope: "FoldScope") -> list[str]:
        """The active keys a ``main_tiers`` fold owns, per ``scope.keys_from``.

        The one place the fold's key source is turned into keys.  Read LIVE
        from the store on every call (the spine calls it twice: once for the
        merge input, once as the drift-partition universe, and the store is
        mutated in between by minting, promotion and soft-staling — a cached
        snapshot would misclassify all three).

        ``"all_tiers"`` returns :meth:`MemoryStore.all_active_keys` verbatim,
        so the absorbing fold sees exactly what it always saw; ``"main_tiers"``
        returns the three main tiers' keys via
        :meth:`MemoryStore.active_keys_in_tier`, leaving every interim slot's
        keys out of the fold entirely.

        Args:
            scope: The immutable :class:`FoldScope` for the current fold.

        Returns:
            Active key strings in tier-then-insertion order.
        """
        if scope.keys_from == "all_tiers":
            return self.store.all_active_keys()
        return [
            key
            for tier in ("episodic", "semantic", "procedural")
            for key in self.store.active_keys_in_tier(tier)
        ]

    def _recall_passing_keys(
        self,
        state: "object | None",
        entries: "list[dict]",
    ) -> "set[str] | None":
        """Return the set of keys whose ``exact_match`` verdict is True.

        Reads ``state.last_per_key`` — the per-key verdict from the final
        fill probe written by ``RecallEarlyStopCallback.on_epoch_end``.

        Returns a set of passing key strings when the verdict is available,
        or ``None`` when ``state`` is ``None`` (early-stop disabled) or
        ``state.last_per_key`` is ``None`` (no probe has run yet).  A ``None``
        return is the explicit "no verdict" signal; callers MUST route it to
        ``_probe_passing_keys`` — never treat it as an empty passing-set.

        Args:
            state: The ``_EarlyStopState`` returned alongside the callback by
                ``_maybe_make_recall_callback``, or ``None`` when the callback
                was not constructed.
            entries: The key-entry list (used only for logging; not filtered
                here).

        Returns:
            ``set[str]`` of passing key names, or ``None`` if no verdict.
        """
        if state is None:
            return None
        last_per_key = getattr(state, "last_per_key", None)
        if last_per_key is None:
            return None
        return {r["key"] for r in last_per_key if r["exact_match"]}

    def _probe_passing_keys(
        self,
        adapter_name: str,
        entries: "list[dict]",
    ) -> "set[str]":
        """Run a dedicated per-key recall probe and return the passing set.

        Called when ``_recall_passing_keys`` returns ``None`` — i.e. when the
        early-stop callback was not active (``recall_early_stopping=False``) or
        had not yet run a probe.  This ensures recall-gated registration always
        has a verdict on the FINAL trained weights, regardless of whether the
        callback fired.

        Uses the full entries list without any sampling cap (unlike
        ``_run_recall_sanity_probe`` which caps at max_probe=100).  Probe
        failures propagate as raised exceptions — do NOT swallow errors.

        Gradient checkpointing is disabled before the probe (required for
        ``model.generate()`` to use the KV cache) and NOT re-enabled
        afterward, because this is called after training has completed.

        Args:
            adapter_name: Active adapter name for the probe.
            entries: Full per-tier entry list to probe (no truncation).

        Returns:
            Set of key strings whose ``exact_match`` verdict is True.
        """
        from paramem.memory.entry import build_registry as _build_registry_inner
        from paramem.training.recall_eval import evaluate_indexed_recall

        self._disable_gradient_checkpointing()
        result = evaluate_indexed_recall(
            self.model,
            self.tokenizer,
            entries,
            _build_registry_inner(entries),
            adapter_name=adapter_name,
            batch_size=self.training_config.recall_probe_batch_size,
        )
        return {r["key"] for r in result["per_key"] if r["exact_match"]}

    def _reset_main_tier_registries_and_simhashes(
        self,
        tier_keyed: dict[str, list[dict]],
        passing_sets_by_tier: "dict[str, set[str] | None] | None" = None,
        *,
        soft_stale_by_tier: "dict[str, dict[str, dict]] | None" = None,
    ) -> None:
        """Reset each main tier's KeyRegistry AND SimHash registry from ``tier_keyed``.

        The registry and the SimHash registry MUST be rebuilt together: rewriting
        the registry alone leaves a fold-rebuilt tier (e.g. episodic consolidated
        from interim) with an EMPTY SimHash registry, so SimHash-confidence recall —
        the primary recall metric — returns 0.000 for every key, breaking
        ``reconstruct_graph`` / train→simulate and the hallucination/recall
        verification.  Co-locating both updates here makes that pairing the only
        callable form, so the registry can never be reset without its SimHashes.
        Sets both registry keys and simhash fingerprints together — the active
        simhash is written directly onto the fresh :class:`KeyRegistry` before
        it is loaded into the store.

        Recall-gated registration (stage 9): only keys whose ``exact_match``
        verdict is True on the FINAL trained weights are admitted.  The verdict
        is supplied via ``passing_sets_by_tier``.  For any tier whose entry is
        ``None`` (verdict absent — early-stop disabled or tier not trained),
        a dedicated per-key probe is run on the trained weights as the fail-safe
        (``_probe_passing_keys``).  A ``None`` verdict NEVER admits all keys
        blindly — that would constitute silent total knowledge loss if the model
        had not actually learned them.

        Soft-stale preservation: the fresh ``KeyRegistry()`` that replaces the
        live registry would wipe any stale flip applied during the drift-partition
        step.  Pass ``soft_stale_by_tier`` so the rebuilt registry seeds the stale
        partition BEFORE adding passing (active) keys.  Stale simhashes are also
        merged back into the rebuilt simhash dict so they survive on disk for the
        stale-echo seam.

        Args:
            tier_keyed: Per-tier keyed-entry lists (full post-consolidation set).
            passing_sets_by_tier: Per-tier sets of keys that passed the recall
                gate.  A ``None`` entry for a tier triggers the probe fallback.
                Pass ``None`` for the entire dict to skip recall gating: every
                key in ``tier_keyed`` is admitted without a verdict.  That is the
                DISK VENUE's production contract — recall gating is a verdict on
                adapter weights, and the disk venue has none to probe, so there
                is nothing a probe could add and nothing a missing verdict could
                hide.  Distinct from a per-tier ``None`` (weights venue, verdict
                absent for that tier), which triggers ``_probe_passing_keys``.
            soft_stale_by_tier: Per-tier dict of soft-staled keys captured at the
                drift-partition step.  Keys map to
                ``{"stale_cycles": int, "simhash": int | None}``.  When ``None``
                (the default, for callers that do not have a stale partition), no
                stale seeding occurs.
        """
        _stale_partition = soft_stale_by_tier or {}
        for _main_tier in ("episodic", "semantic", "procedural"):
            keyed = tier_keyed.get(_main_tier, [])
            _stale_recs = _stale_partition.get(_main_tier, {})
            if not keyed:
                # No active keys for this tier — clear the registry but
                # STILL seed any stale records (an empty tier with stale keys
                # must retain them).  Stale simhashes live in _stale[key]["simhash"]
                # so they are carried automatically into the new registry.
                new_reg = KeyRegistry()
                new_reg._stale = dict(_stale_recs)  # seed stale partition
                self.store.load_registry(_main_tier, new_reg)
                continue

            # Determine the recall-passing set for this tier.
            if passing_sets_by_tier is not None:
                passing = passing_sets_by_tier.get(_main_tier)
                if passing is None:
                    # FAIL-SAFE: None verdict → dedicated per-key probe.
                    # Never treat None as "admit all" or "drop all".
                    passing = self._probe_passing_keys(_main_tier, keyed)
                    logger.info(
                        "_reset_main_tier_registries_and_simhashes: tier %s — "
                        "no verdict from callback, ran dedicated probe (%d/%d passed)",
                        _main_tier,
                        len(passing),
                        len(keyed),
                    )
                else:
                    logger.info(
                        "_reset_main_tier_registries_and_simhashes: tier %s — "
                        "%d/%d keys passed recall gate",
                        _main_tier,
                        len(passing),
                        len(keyed),
                    )
                keyed = [kp for kp in keyed if kp["key"] in passing]

            # Build the fresh registry:
            # (a) seed stale records FIRST — they must survive the rebuild;
            # (b) then add passing active keys with their simhashes.
            # Simhashes are set directly on the registry; stale simhashes live
            # in _stale[key]["simhash"] already (carried by the stale records).
            new_reg = KeyRegistry()
            new_reg._stale = dict(_stale_recs)  # seed stale partition before active keys
            active_simhashes = dict(build_registry(keyed))
            for kp in keyed:
                new_reg.add(kp["key"])
                fp = active_simhashes.get(kp["key"])
                if fp is not None:
                    new_reg.set_simhash(kp["key"], fp)
            self.store.load_registry(_main_tier, new_reg)

    def _drop_interim_tier_registries(self) -> int:
        """Drop every interim tier registry from the store.

        Returns the count of tiers dropped.  Called at the end of a full
        consolidation cycle when interim adapters are unloaded and their
        per-tier registries are no longer needed.
        """
        interim_tiers = [t for t in self.store.tiers_with_registry() if "_interim_" in t]
        for t in interim_tiers:
            self.store.drop_registry(t)
        return len(interim_tiers)

    def _entries_from_graph(
        self,
        session_graph,
        *,
        procedural_enabled: bool,
    ) -> tuple[list[dict], list[dict]]:
        """Build entry relation dicts from a session graph — no model call.

        Projects relations and entity attributes into a unified relation-dict
        set, then partitions it into episodic/procedural.  Each relation is
        projected via :func:`_relation_to_entry_dict`, which canonicalizes
        the predicate so interim-tier entries match the identity form the
        merger stamps onto the cumulative edge.  The attribute surface
        (``Entity.attributes``) is projected via
        ``relation_prep._flatten_entity_attributes`` so scalar-PII keying
        (email/phone/linkedin) is not silently dropped.

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
        exclude = {(r["subject"], r["predicate"]) for r in relation_dicts}
        projected = relation_prep._flatten_entity_attributes(
            session_graph.entities, exclude_pairs=exclude
        )
        if projected:
            logger.info(
                "Entry distillation: projected %d entity attribute(s) into relation set",
                len(projected),
            )
        relation_dicts.extend(projected)
        return relation_prep.partition_relations(
            relation_dicts, procedural_enabled=procedural_enabled
        )

    def snapshot_dir_for(self, *, interim_stamp: str | None = None) -> Path | None:
        """Return this loop's per-cycle/per-run debug-snapshot directory.

        Layout (2026-05-14 locked spec):

            paths.debug/episodic/[interim_<stamp>/]cycle_<N>/run_<run_id>/

        Tier prefix is fixed to ``episodic`` since every cycle's
        graph/relation/sessions debug artifacts are anchored on the
        episodic-primary extraction; procedural / semantic-only writers
        (none today) can introduce their own tier roots when needed.

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
                pipeline.  Selects both the system prompt and the user
                template.  Narrator binding for document sources uses the
                same ``build_speaker_context`` mechanism as transcripts — no
                separate ``doc_title`` or context string is needed.
            event_time: Session-start assertion time (ISO 8601), typically
                the session's ``started_at``. Forwarded to the extraction
                chokepoint as ``timestamp`` so a NEW fact's edge
                ``last_seen`` reflects when it was asserted, not when
                extraction ran. ``None`` (default) falls back to ``now()``
                at the extractor layer — preserves behaviour for callers
                that don't yet have a real session-start time.
        """
        logger.info("=== Extraction (session=%s) ===", session_id)

        # Ownership cue: a soft, in-prompt compliance aid for document
        # sources with a known speaker name — see OWNERSHIP_CUE. Built as a
        # LOCAL extraction-only variable; session_transcript itself is left
        # unmodified because it is reused below for STT anchoring / traces.
        # The deterministic exact-full-name rewrite in _stamp_speaker_entity
        # does not depend on this cue; it only raises first-pass compliance.
        extraction_input = session_transcript
        if source_type == "document" and speaker_name:
            extraction_input = (
                OWNERSHIP_CUE.format(name=speaker_name, sid=speaker_id) + session_transcript
            )

        # Outer extraction_trace scope wraps the whole session body so the
        # orchestrator phases (merge_into_cumulative, procedural_extract,
        # dedup_*) record into the same trace as the inner extract_graph /
        # extract_procedural_graph calls — those traces nest-no-op into this
        # one.  The final attach_to(...) calls below capture the complete
        # phase history on each session graph before it is dumped.
        with extraction_trace() as trace:
            # --- EXTRACT ---
            session_graph = self.extraction.run(
                extraction_input,
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
                procedural_enabled=self.procedural_config is not None,
            )

            # --- PROCEDURAL: separate extraction pass ---
            # extract_procedural_graph self-traces the "procedural_extract"
            # phase (nest-no-ops onto this outer extraction_trace scope) via
            # the shared _run_local_extraction primitive, so no wrapper is
            # needed here.
            proc_graph: SessionGraph | None = None
            if self.procedural_config is not None:
                proc_graph = self.extraction.run_procedural(
                    extraction_input,
                    session_id,
                    speaker_name=speaker_name,
                    source_type=source_type,
                    speaker_id=speaker_id,
                    timestamp=event_time,
                )
                procedural_rels.extend(_relation_to_entry_dict(r) for r in proc_graph.relations)
                # Merge proc_graph into the cumulative graph so its relations
                # reach the unified keying surface (_build_all_edge_entries_into)
                # at the next run_consolidation_cycle call.  Same
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

            # Unified dedup (identical policy as run_cycle + server path).
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

        self.last_session_graph = session_graph

        # Surface a non-fatal cloud-enrichment degradation (the hiccup fail-open
        # in ``stage_enrich``) as an operator-visible incident — the SAME
        # ``record_incident`` surface the outage path uses, called directly here
        # the way this method already calls ``on_session_extracted`` and
        # ``_save_adapters`` calls ``save_adapter``.  ``session_graph`` is this
        # method's own local, not a side-channel read.  Severity ``"warning"``
        # (the run succeeded); keyed by phase so repeated hiccups bump one
        # incident rather than flooding the store.
        degraded = session_graph.diagnostics.get("cloud_enrichment_degraded")
        if degraded is not None and self._incidents_state_dir is not None:
            from paramem.server.incidents import record_incident

            record_incident(
                self._incidents_state_dir,
                type="enrichment_degraded",
                key="cloud_enrich",
                severity="warning",
                summary=(
                    "Consolidation: cloud enrichment degraded — kept "
                    "pre-enrichment facts (unparseable response)"
                ),
                detail={
                    "type": "enrichment_degraded",
                    "session_id": session_id,
                    **degraded,
                    "at": datetime.now(timezone.utc).isoformat(),
                },
            )

        # Release reclaimable device memory back to the WSL2 dxg layer at every
        # session boundary.  PyTorch's caching allocator retains freed blocks
        # (``reserved`` − ``allocated``); on this 8 GiB laptop, after a session's
        # plausibility-filter peak, that retained pool can hold ~700-1500 MiB
        # which dxg counts as in-use.  Without this, multi-session cycles
        # accumulate host-side residency until ``dxgkio_make_resident`` fails
        # with ENOMEM on the next session's first growth — the dxg crash class
        # we measured on 2026-05-04.  Uses ``safe_empty_cache`` (not a bare
        # ``torch.cuda.empty_cache``) so the cuBLAS workspaces the extraction
        # chain's ~4 generate calls allocate outside the PyTorch allocator
        # (~280 MiB/cycle, untouched by ``empty_cache``) are released too.  In
        # the server path ``vram_scope`` already runs ``safe_empty_cache`` in
        # its ``finally`` after this call; this matters for experiment callers
        # of ``extract_session`` (e.g. ``run_cycle``) that are not wrapped.
        try:
            safe_empty_cache()
        except Exception:  # noqa: BLE001
            pass

        return episodic_rels, procedural_rels

    def train_adapters(
        self,
        all_episodic_rels: list[dict],
        all_procedural_relations: list[dict],
        speaker_id: str,
    ) -> dict:
        """Train all adapters once on accumulated relations (blocking).

        Called after all sessions have been extracted.  Returns dict with
        train losses per adapter.

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
        if not self.store.replay_enabled:
            logger.warning("No indexed key registry — skipping training")
            return {}

        # cycle_count is incremented inside run_consolidation_cycle.
        cycle_result = self.run_consolidation_cycle(
            all_episodic_rels,
            all_procedural_relations,
            speaker_id=speaker_id,
            mode="train",
            run_label=f"train-adapters-cycle{self.cycle_count + 1}",
        )

        # --- Roll interim slot into main ---
        # run_consolidation_cycle trains into episodic_interim_<stamp>.  Callers
        # that probe model.set_adapter("episodic") need the trained weights in the
        # main slot.  Submit the train fold via an ephemeral BackgroundTrainer so
        # the GPU lock is held for the full per-tier rebuild (consolidate requires
        # this in train mode — see its entry guard).
        # submit_and_wait blocks until the worker finishes and re-raises on error.
        _folded = False
        if "episodic" in self.model.peft_config or any(
            k.startswith("episodic_interim_") for k in self.model.peft_config
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
                self.consolidate(mode="train", trainer=_bt)

            try:
                _bt.submit_and_wait(_consolidate)
                _folded = True
            finally:
                _bt.close()

        # --- SAVE main slots ---
        # The train fold persists+verifies the merged main weights itself
        # (between its registry rewrite and interim purge), so a
        # successful fold already wrote durable main slots.  Re-saving here would
        # just re-run the same atomic save + disk-integrity verify.  Only save
        # when the fold branch did NOT run (no interim/episodic adapter to roll),
        # in which case this is the sole main persist.
        if not _folded:
            self._save_adapters()

        # Propagate per-tier train losses from the cycle result so callers
        # (experiment scripts) can inspect convergence without re-parsing logs.
        result = {
            "episodic_train_loss": cycle_result.get("episodic_train_loss"),
        }
        logger.info("Training complete: %s", cycle_result)
        return result

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
        whether to delegate to ``_run_fold``.

        Args:
            stamp: The sub-interval stamp (``YYYYMMDDTHHMM``).

        Returns:
            Adapter name string ``f"episodic_interim_{stamp}"``.
        """
        return f"episodic_interim_{stamp}"

    def _mint_keyed_entries(
        self,
        rels: list[dict],
        *,
        prefix: str,
        start_index: int,
        speaker_id: str,
        tag_new: bool = True,
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
            tag_new: When ``True`` (default), each minted entry receives
                ``entry["_new"] = True`` so the caller can identify newly-minted
                entries for deferred ``store.put`` / counter advancement.  Set
                ``False`` when the caller does not need the sentinel (e.g. the
                fold pre-pass or the procedural TRAIN path).

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
            if tag_new:
                entry["_new"] = True
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

    def _save_adapters(self) -> None:
        """Save adapters and registries to disk using the atomic registry-last ordering.

        Saves to two locations:
        - ``output_dir/<tier>/`` — canonical latest state (server use)
        - ``paths.debug/.../training/tiers/<tier>/adapter_weights/`` —
          per-cycle plaintext shadow for inspection (only when
          ``save_cycle_snapshots`` is on; written by
          :func:`~paramem.utils.artifacts.on_main_adapters_saved`).

        Atomic save ordering — registry written last as the commit signal:
          1. ``save_bytes`` → in-memory registry bytes (no disk write).
          2. ``sha256`` the bytes so the manifest can stamp them pre-write.
          3. Build manifest with ``registry_sha256_override=hash`` for each adapter.
          4. Save adapter weights + manifest into the new slot.
          4a. Post-save disk-integrity verify: reload the slot into an isolated
              verify adapter and probe recall.  Raises ``RuntimeError`` when the
              on-disk artifact is corrupted (partial write / dirty-page flush
              race).  The exception propagates to the caller's try/except, which
              then skips ``mark_consolidated`` so sessions remain pending.
          5. Per-cycle snapshots (no manifest).
          6. Per-tier ``indexed_key_registry.json`` written to
             ``<adapter_dir>/<tier>/indexed_key_registry.json``.
             The registry now carries the unified simhash map (active∪stale)
             in its ``"simhash"`` key — a separate ``simhash_registry.json``
             is no longer written.
          7. ``save_from_bytes`` — flush the identical registry bytes; this
             is the commit signal for ``find_live_slot``.

        Crash semantics: a kill after step 4 but before step 8 leaves the
        new slot present with a manifest stamping the new registry hash,
        while the on-disk registry still carries the old hash.
        ``find_live_slot`` won't match → slot is latent, harmless.

        Every saved main slot is stamped with the cadence-window floor that is
        current at save time.  ``window_stamp`` is provenance only — no code
        compares stamps to decide whether a fold is due.

        The recall gate threshold is read from ``self.config.recall_sanity_threshold``
        (set once at construction from the YAML field of the same name).
        """
        import hashlib as _hashlib

        from paramem.adapters.manifest import build_manifest_for
        from paramem.memory.interim_adapter import current_full_consolidation_stamp

        fingerprint_cache = getattr(self, "fingerprint_cache", None)
        full_period = getattr(self, "full_consolidation_period_string", "")
        full_window_stamp = current_full_consolidation_stamp(full_period)

        # Serialise each tier's registry to bytes and hash them — no disk I/O at this point.
        # Per-tier: tier_name → (payload_bytes, sha256_hex)
        tier_payloads: dict[str, tuple[bytes, str]] = {}
        if self.store.replay_enabled:
            for _tier_name in self.store.tiers_with_registry():
                _tier_reg = self.store.registry(_tier_name)
                _payload = _tier_reg.save_bytes()
                tier_payloads[_tier_name] = (_payload, _hashlib.sha256(_payload).hexdigest())
        total_key_count = len(self._all_active_keys()) if self.store.replay_enabled else None

        def _build(name: str) -> "object":
            # Use the tier's own registry hash when available.
            # A manifest failure is a load-bearing bug — the slot becomes
            # unmountable because find_live_slot cannot match the registry
            # hash.  Let the exception propagate to the caller so sessions
            # stay pending and are retried rather than silently losing the
            # manifest.
            _sha = tier_payloads.get(name, (None, None))[1]
            return build_manifest_for(
                self.model,
                self.tokenizer,
                name,
                registry_path=None,
                key_count=total_key_count,
                base_model_hash_cache=fingerprint_cache,
                registry_sha256_override=_sha,
                window_stamp=full_window_stamp,
                adapter_root=self.output_dir,
            )

        def _entries_for_tier(simhash_registry: dict) -> list[dict]:
            """Return the entries list for one adapter tier.

            Builds the in-memory entries that were encoded into the saved weights,
            for use by ``_verify_saved_adapter_from_disk``.

            Returns ``{key, subject, predicate, object}`` entries.
            """
            pairs: list[dict] = []
            for key in simhash_registry:
                qa = self.store.get(key)
                if qa is None:
                    continue
                pairs.append(
                    {
                        "key": key,
                        "subject": qa["subject"],
                        "predicate": qa["predicate"],
                        "object": qa["object"],
                    }
                )
            return pairs

        def _save_and_verify(
            adapter_name: str,
            simhash: dict,
        ) -> Path:
            """Save adapter, probe disk artifact, clean up slot on probe failure.

            Wraps ``atomic_save_adapter`` + ``_verify_saved_adapter_from_disk``
            so that a failed disk-integrity probe deletes the bad slot before
            re-raising.  This prevents a latent corrupted slot from surviving
            until the next rotation or operator inspection.

            Args:
                adapter_name: PEFT adapter name (e.g. ``"episodic"``).
                simhash: Per-tier SimHash registry dict used to filter pairs.

            Returns:
                Path to the slot directory written by ``atomic_save_adapter``.
            """
            import shutil as _shutil

            from paramem.memory.interim_adapter import adapter_slot_root_for_name

            slot = atomic_save_adapter(
                self.model,
                adapter_slot_root_for_name(self.output_dir, adapter_name),
                adapter_name,
                manifest=_build(adapter_name),
            )
            try:
                self._verify_saved_adapter_from_disk(
                    adapter_name,
                    slot,
                    _entries_for_tier(simhash),
                    threshold=self.config.recall_sanity_threshold,
                )
            except Exception:
                # Delete the bad slot so a latent corrupted artifact is not
                # left on disk; re-raise so the caller skips mark_consolidated.
                try:
                    _shutil.rmtree(slot, ignore_errors=True)
                    logger.warning(
                        "_save_adapters: deleted bad slot %s after failed disk-verify",
                        slot,
                    )
                except Exception as _cleanup_exc:  # noqa: BLE001
                    logger.warning(
                        "_save_adapters: could not remove bad slot %s: %s",
                        slot,
                        _cleanup_exc,
                    )
                raise
            return slot

        # Save each adapter to a slot, then immediately reload from disk and probe recall
        # to catch silent partial writes before ``mark_consolidated`` fires.
        # On probe failure the bad slot is deleted and RuntimeError propagates.
        # Collect slot paths for post-registry-commit pruning.
        _saved_slots: dict[str, Path] = {}
        _saved_slots["episodic"] = _save_and_verify(
            "episodic", self.store.tier_simhashes("episodic", include_stale=False)
        )
        if "semantic" in self.model.peft_config:
            _saved_slots["semantic"] = _save_and_verify(
                "semantic", self.store.tier_simhashes("semantic", include_stale=False)
            )
        if "procedural" in self.model.peft_config:
            _saved_slots["procedural"] = _save_and_verify(
                "procedural", self.store.tier_simhashes("procedural", include_stale=False)
            )

        # Per-cycle adapter-weight shadows (debug/analysis only — no
        # manifest).  Layout owned by paramem.utils.artifacts:
        #   paths.debug/.../training/tiers/<tier>/adapter_weights/
        # The artifact scope opened below resolves to no root when
        # save_cycle_snapshots is off, so the hook no-ops without a flag check.
        tier_shadow = ["episodic"]
        if "semantic" in self.model.peft_config:
            tier_shadow.append("semantic")
        if "procedural" in self.model.peft_config:
            tier_shadow.append("procedural")
        with self._artifact_scope():
            on_main_adapters_saved(self.model, tier_shadow)

        # Flush the indexed_key_registry per tier (the unified file now carries
        # active∪stale simhashes in its "simhash" key), then the registry commit signal.
        # The separate simhash_registry.json is no longer written.
        if self.store.replay_enabled and tier_payloads:
            # LAST: flush the exact bytes that were hashed in step 2, so
            # ``find_live_slot`` on restart can match meta.registry_sha256
            # against hashlib.sha256(registry_path.read_bytes()).
            # Registry is written per-tier so each tier has its own signal.
            for _tier in ("episodic", "semantic", "procedural"):
                _tier_payload, _ = tier_payloads.get(_tier, (None, None))
                if _tier_payload is None:
                    continue
                _tier_dir = self.output_dir / _tier
                _tier_dir.mkdir(parents=True, exist_ok=True)
                _tier_registry_path = _tier_dir / "indexed_key_registry.json"
                self.store.registry(_tier).save_from_bytes(
                    _tier_payload, _tier_registry_path, consolidating=True
                )

        # Post-registry-commit slot pruning: runs AFTER the commit signal so
        # find_live_slot always sees a consistent (slot, registry) pair even
        # during a brief prune.  Prune only tiers that were saved this cycle.
        from paramem.memory.interim_adapter import adapter_slot_root_for_name as _asr

        for _tier, _live_slot in _saved_slots.items():
            self._prune_old_slots(
                tier_root=_asr(self.output_dir, _tier),
                live_slot=_live_slot,
                keep=self._keep_prior_slots,
            )

    # ------------------------------------------------------------------
    # Fold-resume durable marker helpers
    # ------------------------------------------------------------------
    # ``fold_resume.json`` lives under ``output_dir.parent / "state"``
    # (same dir as ``consolidation_retry.json``).  It is age-wrapped via
    # ``write_infra_bytes`` when a daily identity is loaded — the marker
    # carries train_assignment SPO fact content.  Single-writer (only the
    # consolidation loop thread inside ``_run_fold`` writes it), so no
    # flock is needed.
    # Schema version: 1.
    # NOT in infra_paths() — control-plane only, never served.

    _FOLD_RESUME_VERSION: int = 1
    _FOLD_RESUME_FILENAME: str = "fold_resume.json"

    @property
    def _fold_state_dir(self) -> Path:
        """Parent state directory for ``fold_resume.json``.

        Derived as ``output_dir.parent / "state"`` to match the production
        layout (``config.paths.data / "state"``).  For experiment callers
        with ``output_dir="outputs/phase3"`` this yields ``outputs/state``,
        which is self-contained and harmless.
        """
        d = self.output_dir.parent / "state"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @staticmethod
    def _new_telemetry_run_stamp() -> str:
        """Mint a per-run telemetry ring key, unique per fold *run*.

        Microsecond-precision UTC timestamp (``%f``) — distinct from
        ``_compute_fold_stamp``, which hashes the SPO keyset and is
        therefore IDENTICAL across two runs over an unchanged keyset (the
        common steady-state case). Reusing that content fingerprint as the
        ring key would upsert unrelated runs into one growing cycle entry;
        this stamp instead identifies the run itself, minted once at fold
        entry and threaded to every telemetry write of that run.
        """
        return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")

    def _compute_fold_stamp(self, *, tier: "str | None" = None) -> str:
        """SHA-256 over the active registry-true SPO keyset at ``_run_fold`` entry.

        Stable across process restarts because (1) the on-disk registries (key
        set + simhash) are not rewritten until the fold finalizes
        (``_reset_main_tier_registries_and_simhashes`` at ``:4392``), and
        (2) ``preload_cache`` deterministically reconstructs identical SPO from
        the unchanged adapter weights — the weights are not retrained on a
        crash-resume.  The registries carry keys + simhash only, not SPO
        (``store.py:1170``); SPO comes from the weight probe.  If reconstruction
        yields different SPO than pre-crash the stamp diverges and the fold
        safely re-runs fresh rather than resuming on a stale stamp.

        Args:
            tier: When set, scope the stamp to
                ``store.active_keys_in_tier(tier)`` (interim-slot fold).
                When ``None``, use all active keys across all tiers (full fold).

        Returns:
            Hex-encoded SHA-256 digest of the sorted ``(key, subject, predicate,
            object)`` tuples for the active keyset.
        """
        import hashlib
        import json as _json

        h = hashlib.sha256()
        if tier is not None:
            keys = list(self.store.active_keys_in_tier(tier))
        else:
            keys = list(self.store.all_active_keys())

        tuples = []
        for k in keys:
            entry = self.store.get(k)
            if entry is None:
                tuples.append((k, "", "", ""))
            else:
                tuples.append(
                    (
                        k,
                        entry.get("subject", ""),
                        entry.get("predicate", ""),
                        entry.get("object", ""),
                    )
                )
        tuples.sort()
        for t in tuples:
            h.update(_json.dumps(t, sort_keys=True).encode("utf-8"))
        return h.hexdigest()

    def _write_fold_resume(self, state: dict) -> None:
        """Atomically write *state* to ``fold_resume.json`` via ``write_infra_bytes``.

        The file is age-encrypted when a daily identity is loaded; plaintext
        otherwise.  On ``OSError`` (e.g. ENOSPC), logs a loud warning and
        continues — crash-resume degrades to fresh-restart for that fold,
        which is the behaviour from before crash-resume markers existed.
        Non-IO exceptions propagate.

        Args:
            state: JSON-serialisable dict representing the full marker state.
        """
        import json as _json

        from paramem.backup.encryption import write_infra_bytes

        path = self._fold_state_dir / self._FOLD_RESUME_FILENAME
        payload = _json.dumps(state, indent=2).encode("utf-8")
        try:
            write_infra_bytes(path, payload)
        except OSError:  # boundary: ENOSPC / filesystem error
            logger.warning(
                "_write_fold_resume: failed to write %s — crash-resume degraded "
                "to fresh-restart for this fold",
                path,
                exc_info=True,
            )

    def _read_fold_resume(self) -> "dict | None":
        """Read and parse ``fold_resume.json``, returning ``None`` on absence or error.

        Boundary read: any ``OSError`` or parse error returns ``None`` so
        callers always fall through to the fresh-fold path.

        Returns:
            Parsed dict on success, ``None`` when the file is absent,
            unreadable, or malformed.
        """
        import json as _json

        from paramem.backup.encryption import read_maybe_encrypted

        path = self._fold_state_dir / self._FOLD_RESUME_FILENAME
        if not path.exists():
            return None
        try:
            raw = read_maybe_encrypted(path)
            return _json.loads(raw.decode("utf-8"))
        except Exception:  # noqa: BLE001  # boundary: external-file read
            logger.debug(
                "_read_fold_resume: %s unreadable — treating as absent", path, exc_info=True
            )
            return None

    def _persist_fold_assignment(
        self,
        scope_name: str,
        fold_stamp: str,
        train_assignment: "dict[str, list[dict]]",
        dataset_fingerprints: "dict[str, str]",
        *,
        pending_session_ids: "list[str] | None" = None,
    ) -> None:
        """Write the initial ``fold_resume.json`` marker once the assignment is finalized.

        Called on the TRAINING path once the per-tier assignment is final —
        there is no early return between assignment and this call, so the
        marker always reflects what the fold is about to train.

        ``completed_tiers`` starts empty; the first ``in_flight_tier`` is the
        first tier that has training entries.

        Args:
            scope_name: ``"main_tiers"`` (full fold) or ``"interim_slot"``
                (interim-slot fold).
            fold_stamp: SHA-256 from ``_compute_fold_stamp`` (pre-mutation).
            train_assignment: Per-tier lists of entry dicts
                (``key/subject/predicate/object/speaker_id``).
            dataset_fingerprints: Per-tier ``_fingerprint_entries`` hexdigest.
            pending_session_ids: Sorted list of session ids the current
                pending-session batch was extracted from (``scope_name ==
                "interim_slot"`` only — ``fold_stamp`` alone is degenerate for
                a brand-new interim tier since ``active_keys_in_tier`` is empty
                pre-training, so it cannot detect a changed pending-session
                set on its own).  ``None`` (the ``main_tiers`` default) stores
                an empty list — ``main_tiers`` has no equivalent per-cycle
                pending-session identity to track.
        """
        non_empty_tiers = [t for t in train_assignment if train_assignment[t]]
        in_flight = non_empty_tiers[0] if non_empty_tiers else None
        state: dict = {
            "version": self._FOLD_RESUME_VERSION,
            "scope": scope_name,
            "fold_stamp": fold_stamp,
            "completed_tiers": [],
            "tier_checkpoints": {},
            "in_flight_tier": in_flight,
            "train_assignment": train_assignment,
            "dataset_fingerprint": dataset_fingerprints,
            "pending_session_ids": sorted(pending_session_ids)
            if pending_session_ids is not None
            else [],
        }
        self._write_fold_resume(state)
        logger.debug(
            "_persist_fold_assignment: wrote fold_resume.json scope=%s in_flight=%s",
            scope_name,
            in_flight,
        )

    def _mark_tier_complete(self, tier: str, checkpoint_path: "str | None") -> None:
        """Append *tier* to ``completed_tiers`` in ``fold_resume.json``.

        Also updates ``tier_checkpoints`` and advances ``in_flight_tier`` to the
        next tier with training entries (or ``None``).

        Safe when the file is absent (logs a warning and no-ops).  On write
        failure the error is absorbed — the marker is advisory; a
        corrupt/missing marker degrades to fresh-restart, which is safe.

        Args:
            tier: The tier name that just completed training (``"episodic"``,
                ``"semantic"``, or ``"procedural"``).
            checkpoint_path: Path to the retained ``checkpoint-N`` directory for
                this tier, or ``None`` when :meth:`_latest_checkpoint_in_dir`
                found no checkpoint under the tier's training-scratch directory
                (reload then falls back to the production slot).
        """
        state = self._read_fold_resume()
        if state is None:
            logger.warning(
                "_mark_tier_complete: fold_resume.json absent when marking %s complete — skipping",
                tier,
            )
            return
        completed: list = state.get("completed_tiers", [])
        if tier not in completed:
            completed.append(tier)
        state["completed_tiers"] = completed
        if checkpoint_path is not None:
            checkpoints: dict = state.get("tier_checkpoints", {})
            checkpoints[tier] = checkpoint_path
            state["tier_checkpoints"] = checkpoints
        # Advance in_flight_tier to the next non-empty, non-completed tier.
        _ta: dict = state.get("train_assignment", {})
        _completed_set = set(completed)
        next_in_flight = None
        for _t in ("episodic", "semantic", "procedural"):
            if _t not in _completed_set and _ta.get(_t):
                next_in_flight = _t
                break
        state["in_flight_tier"] = next_in_flight
        self._write_fold_resume(state)
        logger.debug("_mark_tier_complete: tier=%s next_in_flight=%s", tier, next_in_flight)

    def _clear_fold_resume(self) -> None:
        """Remove ``fold_resume.json`` on clean fold completion.

        Idempotent: a no-op when the file is absent.
        """
        path = self._fold_state_dir / self._FOLD_RESUME_FILENAME
        path.unlink(missing_ok=True)
        logger.debug("_clear_fold_resume: removed %s", path)

    @staticmethod
    def _latest_checkpoint_in_dir(directory: Path) -> "str | None":
        """Return the path of the highest-numbered ``checkpoint-*`` dir under *directory*.

        Globs recursively so a RAM-mode training run — whose ``checkpoint-N/``
        dirs are HF Trainer's own writes into ``/dev/shm``, mirrored by
        ``_RamEpochCopyCallback``/``_StagingResumeCallback`` to
        ``<directory>/bg_checkpoint_epoch/checkpoint-N/`` — is found exactly
        like the disk-mode default, where HF Trainer writes ``checkpoint-N/``
        directly under *directory*.  Without this, RAM mode's checkpoint is
        invisible here and ``_mark_tier_complete`` records ``None``, which
        forces the next crash-resume to reload the tier's stale production
        slot instead of its actually-trained checkpoint.

        Returns ``None`` when no matching directory is found.  Used to locate
        the durable epoch checkpoint for ``_mark_tier_complete``.
        """
        checkpoints = sorted(
            directory.glob("**/checkpoint-*"),
            key=lambda p: int(p.name.split("-")[1]) if p.name.split("-")[1].isdigit() else -1,
        )
        for ckpt in reversed(checkpoints):
            if ckpt.is_dir():
                return str(ckpt)
        return None

    def _training_output_dir(self, adapter_name: str, *, interim_stamp: str | None = None) -> Path:
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
        alongside the slots in the same tier hierarchy but uses scope-named
        sub-dirs (``cycle_<N>`` or ``interim_<stamp>``) so HF's step-numbered
        ``checkpoint-<step>/`` subdirs are isolated per training run and
        never collide with the slot dir's date-named ``<slot_date>/``.

        Paths:

        - ``adapter_name == "episodic_interim_<stamp>"``:
          ``<output_dir>/episodic/interim_<stamp>/``
        - Full cycle, ``adapter_name in {episodic, semantic, procedural}``:
          ``<output_dir>/<adapter_name>/cycle_<N>/``
        - Interim cycle (stamp explicit via ``interim_stamp`` kwarg), tier-level
          adapter: ``<output_dir>/<adapter_name>/interim_<stamp>/``

        Args:
            adapter_name: The PEFT adapter being trained.  One of
                ``"episodic"``, ``"semantic"``, ``"procedural"``, or
                ``"episodic_interim_<stamp>"``.
            interim_stamp: Optional YYYYMMDDTHHMM stamp passed directly by
                ``run_consolidation_cycle``.  Falls back to the
                ``_current_interim_stamp`` instance attribute when set.

        Returns:
            Absolute :class:`~pathlib.Path` to give HF Trainer.

        Raises:
            ValueError: when ``adapter_name`` doesn't match any known tier
                or the interim-adapter naming convention.
        """
        from paramem.memory.interim_adapter import INTERIM_NAME_PREFIX, interim_stamp_from_name

        resolved_stamp = interim_stamp or getattr(self, "_current_interim_stamp", None)

        # Episodic interim slot: scratch nested under the interim sub-dir of
        # the episodic tier (sibling of <slot_date>/).
        if adapter_name.startswith(INTERIM_NAME_PREFIX):
            stamp = interim_stamp_from_name(adapter_name)
            return self.output_dir / "episodic" / f"interim_{stamp}"

        if adapter_name not in ("episodic", "semantic", "procedural"):
            raise ValueError(f"Unknown adapter name for training output dir: {adapter_name!r}")

        # Tier-level scratch under <tier>/.  Interim cycles use the interim
        # stamp as the scope; full cycles use cycle_<N>.
        scope = f"interim_{resolved_stamp}" if resolved_stamp else f"cycle_{self.cycle_count}"
        return self.output_dir / adapter_name / scope

    def run_consolidation_cycle(
        self,
        episodic_rels: list[dict],
        procedural_rels: list[dict],
        *,
        speaker_id: str,
        mode: "Literal['simulate', 'train']",
        run_label: str,
        schedule: str = "",
        max_interim_count: int = 7,
        interim_overflow_slack: int = 0,
        stamp: str | None = None,
    ) -> dict:
        """Unified interim-cycle entry: key prep + optional training + atomic persistence.

        Replaces the former ``_train_extracted_into_interim`` (train) and
        ``simulated_training`` (simulate) methods.  Both modes execute the same
        pipeline — the ONLY mode-conditional code is:

        * :func:`paramem.memory.persistence.commit_tier_slot` — venue
          write (train: save adapter weights; simulate: write sidecar JSON).

        Everything else — cycle counter, guards, speaker tagging, enrichment,
        procedural key prep, simhash update, end-of-cycle adapter switch — is
        mode-agnostic.

        Internal flow:

        1. ``self.cycle_count += 1``.
        2. Guard: no registry → early return ``{"status": "skipped", ...}``.
        3. Guard: no relations → early return ``{"status": "noop", ...}``.
        4. Tag relations with caller's ``speaker_id`` as default.
        5. Compute stamp (when not provided) and call ``_resolve_target_slot``
           to obtain ``adapter_name``.
        6. Ring-full detection (train mode only): when the interim ring is at
           ``max_interim_count`` and the target slot is new, return
           ``mode="cap_pending"`` immediately — sessions stay in the session
           buffer and re-extract on the next tick.
        7. Mint PEFT slot (train only).
        8. Materialize: call :meth:`_materialize_consolidation_graph` scoped
           to the current slot for the recall-miss diagnostic and to rebuild the
           keying surface (pending-session relations from ``merger.graph`` are
           passed as ``extra_relations`` so they survive the graph reset).
        8c. Refine: call :meth:`_refine_consolidation_graph` with both
           ``normalize`` and ``enrich`` pinned ``False`` — the interim scope
           runs neither graph-tier pass (see :attr:`FoldScope.enrich` for the
           rationale).  The recurrence-bump still runs at every level.
        9. Build interim key list via graph-walk (episodic + procedural entries).
           The interim slot holds BOTH factual (episodic) and preference
           (procedural) keys, trained with the attention-only episodic adapter
           config by design; procedural keys fold to the ``procedural`` main
           adapter only at the full fold.
        10. Train (train mode) or skip training (simulate mode).
        11. Apply deferred store mutations; advance counters for both episodic
            and procedural minted keys.
        12. Persist interim slot via ``commit_tier_slot``.
        13. Restore ``"episodic"`` as the active adapter (mode-agnostic).
        14. Return result dict.

        Args:
            episodic_rels: Pre-extracted episodic relations.  May already carry
                ``speaker_id``; missing entries are tagged with *speaker_id*.
            procedural_rels: Pre-extracted procedural relations.  Used for the
                no-relations guard (step 3) and debug output; procedural facts
                reach the training set via merger.graph (merged by
                extract_session / run_cycle), not via this argument directly.
            speaker_id: Default speaker tag for relations missing one.
                Required — callers must always supply a real speaker ID.
            mode: ``"train"`` writes adapter weights; ``"simulate"`` writes
                sidecar JSON registry without touching PEFT.
            run_label: Tag woven into the wandb ``run_name`` for traceability.
                Pass ``session_id`` for per-session calls, or
                ``"tick-<stamp>"`` for batch calls from the scheduled tick.
            schedule: Consolidation refresh-cadence string used to compute the
                sub-interval stamp when *stamp* is not provided.
            max_interim_count: Cap on concurrent interim adapters.  When the
                ring is at or beyond capacity (train mode only), the 3-way gate
                below determines the outcome.  ``max_interim_count < 1`` is
                rejected by the config validator.
            interim_overflow_slack: Number of extra overflow slots allowed
                beyond ``max_interim_count`` before keep-pending kicks in.
                At 0 (default), cap_pending fires immediately when ``c >= N``
                (identical to the original no-slack behavior).  At slack > 0,
                the gate is:
                    c < N           → normal mint (unchanged)
                    N <= c < N+slack → overflow mint; result["overflow_slot"]=True
                    c >= N+slack    → cap_pending (keep sessions pending)
                Counted against PEFT-resident adapters; the slack is proven
                to fit VRAM at boot via ``required_working_set_bytes``.
            stamp: Override the computed sub-interval stamp (test injection).

        Returns:
            Result dict with keys ``{"triples_extracted", "new_keys",
            "adapter_name", "mode", "venue", "error"}``.  ``mode`` is the
            outcome (``"trained"``, ``"simulated"``, ``"cap_pending"``,
            or ``"noop"``); ``venue`` is the training medium (``"train"`` or
            ``"simulate"``).
        """
        # --- 1. Cycle counter ---
        self.cycle_count += 1

        # --- 2. Guard: no registry ---
        if not self.store.replay_enabled:
            logger.warning("run_consolidation_cycle: no indexed key registry — skipping")
            return {
                "triples_extracted": 0,
                "new_keys": [],
                "adapter_name": None,
                "mode": "noop",
                "venue": mode,
                "error": "no_registry",
            }

        triples_extracted = len(episodic_rels)

        # --- 3. Guard: no relations ---
        if not episodic_rels and not procedural_rels:
            return {
                "triples_extracted": 0,
                "new_keys": [],
                "adapter_name": None,
                "mode": "noop",
                "venue": mode,
                "error": None,
            }

        # --- 4. Tag speaker_id defaults ---
        self._tag_speaker_id_defaults(episodic_rels, speaker_id)
        self._tag_speaker_id_defaults(procedural_rels, speaker_id)

        # --- 5. Resolve stamp and target slot ---
        if stamp is None:
            from paramem.memory.interim_adapter import current_interim_stamp as _cis

            stamp = _cis(schedule)

        adapter_name = self._resolve_target_slot(stamp)

        # --- 6. 3-way gate (train mode only) ---
        # Count source: PEFT-resident adapters (what the VRAM ceiling constrains;
        # see SF-9: on-disk count and PEFT count measure different things and
        # converge only at tick boundaries).
        # Gate terms apply only when: train mode AND target slot is new AND
        # registry is live.  Simulate has no PEFT slots so the count is
        # meaningless; simulate always falls through to _run_fold.
        existing_interim_count = len(
            [a for a in self.model.peft_config if a.startswith("episodic_interim_")]
        )
        _gate_active = (
            mode != "simulate"
            and adapter_name not in self.model.peft_config
            and self.store.replay_enabled
        )
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
                # overflow slot.  Fall through to the single _run_fold delegation
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
            # else: c < N — normal mint, fall through to _run_fold below.

        # --- 7. Delegate pipeline to _run_fold (interim_slot scope) ---
        # source is derived from mode: "weights" for train, "disk" for simulate.
        # All pipeline stages (materialize, refine, build-keyed, train/skip, commit)
        # execute inside _run_fold; this wrapper only owns pre-resolution + early-exits.
        # Map the caller's mode Literal to the FoldScope source axis without a mode== fork.
        _interim_source = {"train": "weights", "simulate": "disk"}[mode]
        # Every artifact the fold and its nested passes emit lands in this
        # cycle's debug root; a calibration run, when one is open, adds its own
        # root independently.
        with self._artifact_scope():
            result = self._run_fold(
                FoldScope(
                    name="interim",
                    source=_interim_source,
                    persist="interim_slot",
                    tier=adapter_name,
                    consume_pending=True,
                    defer=True,
                    tag_new=True,
                    normalize=False,  # normalization is full-fold only
                    enrich=False,  # graph enrichment is full-fold only
                    promote=False,
                    subtractive_scope="interim",
                ),
                adapter_name=adapter_name,
                stamp=stamp,
                run_label=run_label,
                triples_extracted=triples_extracted,
                episodic_rels=episodic_rels,
                procedural_rels=procedural_rels,
            )
        # Only tag a real mint: an aborted overflow fold must not trigger
        # the interim_cap_reached incident on the app.py consumer side.
        if is_overflow and result.get("mode") == "trained":
            result["overflow_slot"] = True
        return result

    def _apply_subtractive_removals_to_store(
        self,
        *,
        scope: str,
    ) -> "dict[str, dict[str, dict]]":
        """Consume ``merger.removal_ledger`` entries and soft-stale their keys.

        This is the shared soft-stale stage called by BOTH
        ``run_consolidation_cycle`` (interim) and ``consolidate`` (full fold)
        after every merge that can produce subtractive removals.  The
        shared body is identical for both scopes; the persist/registry-seed step
        that follows is scope-specific and stays in the caller.

        **Always-stale reasons (both scopes):**
        - ``"predicate_synonym_collapse"`` — synonym-predicate collapse from the
          whole-graph normalization pass
          (:meth:`~paramem.training.graph_tier.GraphTierRefiner.run_normalization`).
        - ``"semantic_dedup"`` — near-duplicate triple collapse from the normalization
          pass.
        - ``"entity_merge"`` — edge incident to a same_as variant node (normalization
          pass stale+add).
        - ``"contradiction_same_pred"`` — recency-backed contradiction removal:
          the merger only emits this ledger entry when timestamps pick a unique
          winner; empty/tied → no entry → no stale.  Safe to stale at fold scope
          because a timestamp-less key that tied would never appear here.

        ``"enrichment_same_as"`` and other retain-only reasons stay in the fold's
        ``drift_intended_removal`` bucket (handled inline in the fold spine);
        this helper does NOT soft-stale those.

        Args:
            scope: ``"interim"`` or ``"fold"``.  The stale logic is identical at
                both scopes; the parameter is kept for logging context and
                compatibility with existing callers.

        Returns:
            ``soft_stale_by_tier`` — a per-tier dict mapping staled key strings
            to ``{"stale_cycles": int, "simhash": int|None}`` records.  Passed
            by the fold caller to
            :meth:`_reset_main_tier_registries_and_simhashes` so the rebuilt
            registry seeds the stale partition.  The interim caller (in
            :meth:`_run_fold`) also captures it: on a failed commit those
            keys are re-activated via :meth:`MemoryStore.reactivate` before
            re-raising (``store.discard_keys`` already mutated the in-memory
            registry; ``commit_tier_slot`` persists it on success).
        """
        _ledger: dict[str, dict] = getattr(self.merger, "removal_ledger", {})
        # Reasons that become soft-stale at ALL scopes (ingest, interim, fold).
        # predicate_synonym_collapse: synonym-predicate collapse (normalization pass).
        # semantic_dedup: near-duplicate triple collapse (normalization pass).
        # entity_merge: edge incident to a same_as variant node (normalization pass).
        # contradiction_same_pred: recency-backed contradiction (freshest last_seen wins).
        #   The merger only writes this entry when timestamps pick a unique winner;
        #   empty/tied → coexist (no entry) → safe to stale at fold scope too.
        _always_stale_reasons = {
            "predicate_synonym_collapse",
            "semantic_dedup",
            "entity_merge",
            "contradiction_same_pred",
        }

        soft_stale_by_tier: dict[str, dict[str, dict]] = {}

        for _ik, _entry in list(_ledger.items()):
            _reason = _entry.get("reason", "")
            _should_stale = _reason in _always_stale_reasons
            if not _should_stale:
                continue

            # LOAD-BEARING ORDERING: resolve tier BEFORE flipping the key stale.
            # KeyRegistry.stale() removes the key from _active_keys, so
            # tier_for_active_key() called AFTER the flip returns None.
            _dk_tier = self.store.tier_for_active_key(_ik)
            _dk_simhash: int | None = None
            if _dk_tier is not None:
                _dk_simhash = self.store.simhash(_dk_tier, _ik)

            # Soft-stale in-memory: registry entry retained, simhash retained.
            self.store.discard_keys([_ik], mode="stale")

            if _dk_tier is not None:
                _stale_rec: dict = {"stale_cycles": 0}
                if _dk_simhash is not None:
                    _stale_rec["simhash"] = _dk_simhash
                soft_stale_by_tier.setdefault(_dk_tier, {})[_ik] = _stale_rec
            logger.info(
                "subtractive_removal soft-staled key=%s reason=%s scope=%s",
                _ik,
                _reason,
                scope,
            )

        return soft_stale_by_tier

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
        """Snapshot current merger.graph edges into a list[Relation].

        Called BEFORE :meth:`_materialize_consolidation_graph` resets the graph,
        so the pending-session content survives the reset and re-enters the merge
        via the ``extra_relations`` channel.

        Both fold scopes call this on ``scope.consume_pending`` — the one gate.
        The interim fold always sets it (the pending session IS that cycle's
        content); the full fold sets it in the ``max_interim_count == 0``
        consume-pending mode, where app.py has pre-populated ``merger.graph``
        with the pending-session relations before entering the fold.

        Returns an empty list when the graph is absent or has no edges; both
        ``None`` and ``[]`` are valid no-ops for the ``if extra_relations`` check
        inside :meth:`_materialize_consolidation_graph`.

        Returns:
            list[Relation]: Relation objects built from the current merger graph
                edges.  Each edge contributes exactly one :class:`Relation` with:

                - ``predicate`` taken from the edge ``"predicate"`` attribute
                  (edges with an empty predicate are skipped);
                - ``relation_type`` validated against :data:`_VALID_RTYPES`,
                  falling back to :data:`_FALLBACK_RTYPE`;
                - ``speaker_id`` inherited from the subject node's
                  ``"speaker_id"`` attribute (empty string when absent);
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
        """
        import networkx as _nx

        _g = getattr(self.merger, "graph", None)
        if not isinstance(_g, _nx.MultiDiGraph) or _g.number_of_edges() == 0:
            return []
        _result: list[Relation] = []
        for _er_subj, _er_obj, _er_data in _g.edges(data=True):
            _er_pred = _er_data.get("predicate", "")
            if not _er_pred:
                continue
            _er_rt_raw = _er_data.get("relation_type", _FALLBACK_RTYPE)
            _er_rt: str = _er_rt_raw if _er_rt_raw in _VALID_RTYPES else _FALLBACK_RTYPE
            _er_subj_node = _g.nodes.get(_er_subj, {})
            # C-2: prefer edge-carried speaker_id (stamped by merger A-1/A-2),
            # fall back to subject node's speaker_id when the edge carries none.
            _er_spk = _er_data.get("speaker_id") or _er_subj_node.get("speaker_id", "")
            _result.append(
                Relation(
                    subject=_er_subj,
                    predicate=_er_pred,
                    object=_er_obj,
                    relation_type=_er_rt,  # type: ignore[arg-type]
                    confidence=_er_data.get("confidence", 1.0),
                    speaker_id=_er_spk,
                    session_ids=list(_er_data.get("sessions", [])),
                    last_seen=_er_data.get("last_seen", ""),
                    first_seen=_er_data.get("first_seen", ""),
                )
            )
        return _result

    # ------------------------------------------------------------------
    # Unified persist dispatch — replaces the three independent persist tails
    # (graph-json simulate, interim-slot, main-tiers full fold) that previously
    # lived inline in _run_fold.
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

    def _hydrate_store_for_fold(self, scope: "FoldScope") -> None:
        """Materialise every live key's entry into the store before the fold reads it.

        The fold reads entry content through ``store.get`` at three places —
        :meth:`_build_registry_true_relations`, the keyed branch of
        :meth:`_build_all_edge_entries_into`, and the interim recital-dedup
        scope test.  ``store.get`` is cache-only: on a miss it returns ``None``
        and each of those places drops the key.  A dropped key does not reach
        ``tier_keyed``, and the finalize step rewrites every main-tier
        registry from ``tier_keyed`` — so a key the cache happened not to
        hold is *deregistered and flushed to disk*, and the drift partition
        buckets it as an orphan.  That is silent data loss, and the store can
        legitimately be partially hydrated: ``app._build_store_contents``
        reports exactly this as ``boot_degraded={"reason": "preload_partial"}``
        when the boot probe materialises only some of the active keys.

        So the fold hydrates first.  Every active key of every registered tier
        is resolved through :meth:`~paramem.memory.store.MemoryStore.probe`
        against the venue's :class:`~paramem.memory.source.MemorySource`: cache
        hits cost nothing, misses are materialised from the source of truth
        (adapter weights or ``graph.json``) in one batched pass, and only a key
        that no venue can produce is left for the three sites to drop — which
        is then a true orphan rather than a cache artifact.

        ``memoize=True`` is not conditional on ``inference.preload_cache``.
        That toggle governs the *read* path (boot preload + per-query on-miss
        caching); the fold is a *write* path that already puts every minted
        entry into the store unconditionally, and its own persist tail reads
        those entries back — ``_persist_fold`` projects the store through
        :func:`~paramem.memory.persistence.build_tier_graph_from_store`, which
        raises ``KeyError`` on an active key with no entry.  Hydrating without
        memoizing would therefore break the disk venue's persist, and re-probe
        the same keys once per read site on the weights venue.

        **BASE-MODEL HOLDER** — the :class:`WeightMemorySource` built here
        captures the base model.  It is a frame-local, built from ``self.model``
        at call time (which the fold rebinds around adapter creation, so it must
        never be cached on ``self``) and dropped when this method returns, the
        same no-frame-retention pattern ``app._preload_memory_store`` uses.

        Args:
            scope: The immutable :class:`FoldScope` for the current fold.  Its
                ``source`` selects the venue via :meth:`_venue_from_scope`.

        A source probe that raises is NOT swallowed: proceeding into the fold
        with an unknown-partial store is the data loss this method exists to
        prevent, so the exception aborts the fold before anything is rewritten.
        """
        from paramem.memory.source import build_memory_source

        venue = self._venue_from_scope(scope)
        keys_by_tier = {
            tier: keys
            for tier in self.store.tiers_with_registry()
            if (keys := self.store.active_keys_in_tier(tier))
        }
        if not keys_by_tier:
            return

        source = build_memory_source(
            mode=venue,
            adapter_dir=self.output_dir,
            batch_size=self.training_config.recall_probe_batch_size,
            model=self.model,
            tokenizer=self.tokenizer,
        )
        if source is not None:
            self.store.probe(keys_by_tier, source=source, memoize=True)
            source = None

        dropped = [k for keys in keys_by_tier.values() for k in keys if self.store.get(k) is None]
        if dropped:
            logger.warning(
                "_hydrate_store_for_fold: %d live key(s) have no content in the store "
                "and none in the %s source — this fold will drop them: %s",
                len(dropped),
                venue,
                sorted(dropped)[:10],
            )

    def _verify_committed_slot(
        self,
        adapter_name: str,
        all_keyed: "list[dict]",
        slot: Path,
    ) -> None:
        """Reload an interim slot from disk and probe recall integrity.

        Bridges :meth:`_persist_fold` (interim-train path) to the shared
        :meth:`_verify_saved_adapter_from_disk` method that ``_save_adapters``
        uses for main tiers — closing the disk-verify gap for interim slots
        without mirroring the verify implementation.

        Entry shape passed to the probe contains ONLY the four canonical
        SPO fields (``key``, ``subject``, ``predicate``, ``object``).
        Sentinel fields such as ``_new`` carried by ``all_interim_keyed``
        are intentionally stripped so the probe entry shape matches what
        ``_entries_for_tier`` produces and what ``_run_recall_sanity_probe``
        expects.

        Called as the *verify* callback in :func:`~paramem.memory.persistence.commit_tier_slot`
        (train branch only, before the registry flush).  A raise propagates
        unchanged; the ``finally`` orphan-cleanup in ``commit_tier_slot``
        removes the half-committed slot because ``_registry_flushed`` is still
        ``False`` at the point of the raise.

        Reuses :meth:`_verify_saved_adapter_from_disk` and the same
        ``recall_sanity_threshold`` as main-tier verification.  No second
        cleanup path is added here — slot removal is delegated entirely to
        ``commit_tier_slot``'s existing ``finally`` block.

        Args:
            adapter_name: PEFT adapter name of the interim slot just written
                (e.g. ``"episodic_interim_YYYYMMDDTHHMM"``).
            all_keyed: Full keyed-pair list from the fold, as passed to
                ``commit_tier_slot``.  May carry extra sentinel fields
                (``_new``, etc.) — these are stripped before the probe.
            slot: Path to the timestamped slot directory returned by
                :func:`~paramem.models.loader.save_adapter`.
        """
        entries = [
            {
                "key": kp["key"],
                "subject": kp["subject"],
                "predicate": kp["predicate"],
                "object": kp["object"],
            }
            for kp in all_keyed
        ]
        self._verify_saved_adapter_from_disk(
            adapter_name,
            slot,
            entries,
            threshold=self.config.recall_sanity_threshold,
        )

    def _persist_fold(
        self,
        scope: "FoldScope",
        *,
        # interim_slot inputs
        adapter_name: "str | None" = None,
        stamp: "str | None" = None,
        all_keyed: "list[dict] | None" = None,
    ) -> None:
        """Single persist tail for both fold scopes, in both venues.

        Dispatches on ``scope.persist`` for the scope and on ``scope.source``
        for the venue — never on a ``mode == "train"`` / ``mode == "simulate"``
        literal (the mode-fork guard is satisfied).  Each branch writes its
        venue artifact and runs disk-integrity verification where adapter
        weights were written:

        - ``interim_slot`` (weights): passes a
          :meth:`_verify_committed_slot` callback into
          :func:`~paramem.memory.persistence.commit_tier_slot` so the slot
          is reloaded and probed before the registry flush (commit signal).
          A failed probe propagates; ``commit_tier_slot``'s ``finally``
          orphan-cleanup removes the half-committed slot.
        - ``interim_slot`` (disk): ``verify=None`` — no weights, no probe;
          ``commit_tier_slot`` writes the slot ``graph.json`` instead.
        - ``main_tiers`` (weights): :meth:`_save_adapters` rebuilds the main
          adapter slots; its disk verify is already inside that method.
        - ``main_tiers`` (disk): each main tier's slice of the store is
          projected with
          :func:`~paramem.memory.persistence.build_tier_graph_from_store` and
          written to ``<output_dir>/<tier>/graph.json`` — the exact path
          :class:`~paramem.memory.source.DiskMemorySource` reads back, so the
          round trip is symmetric.  All three main tiers are written
          unconditionally, mirroring the unconditional per-tier registry
          rewrite that immediately precedes this call in the spine.

        Called by :meth:`_run_fold` in place of the independent persist tails
        that previously closed each fold branch.  The surrounding grooming
        (refine, build-entries, train, result-dict assembly) stays inline in
        :meth:`_run_fold`; only the **save action** is unified here.

        Args:
            scope: Immutable :class:`FoldScope` describing this fold.
                ``persist`` selects the scope branch, ``source`` the venue.
            adapter_name: Interim adapter name (``interim_slot`` path only).
            stamp: Sub-interval stamp forwarded to
                :func:`~paramem.memory.persistence.commit_tier_slot`
                (``interim_slot`` path only).
            all_keyed: Full keyed-pair list for the interim slot
                (``interim_slot`` path only).
        """
        from paramem.memory.persistence import (
            build_tier_graph_from_store,
            commit_tier_slot,
            save_memory_to_disk,
        )

        if scope.persist == "interim_slot":
            # interim slot: commit adapter weights (train) or graph.json (simulate).
            # The mode-fork lives inside commit_tier_slot, which is allowlisted.
            # For train interim, pass a verify callback so the slot is probed
            # before the registry flush — closing the disk-verify gap.
            # For simulate interim (no weights), pass verify=None.
            _keyed = all_keyed or []
            _verify: "Callable[[Path], None] | None" = (
                (lambda slot: self._verify_committed_slot(adapter_name, _keyed, slot))  # type: ignore[arg-type]
                if scope.source == "weights"
                else None
            )
            commit_tier_slot(
                loop=self,
                tier="episodic",
                adapter_name=adapter_name,  # type: ignore[arg-type]
                stamp=stamp,  # type: ignore[arg-type]
                mode=self._venue_from_scope(scope),
                all_keyed=_keyed,
                output_dir=self.output_dir,
                verify=_verify,
            )
        elif scope.persist == "main_tiers":
            if scope.source == "weights":
                # Rebuild main adapter weights.  Disk verify is inside
                # _save_adapters (already had it pre-unification).
                self._save_adapters()
            else:
                # No weights: project the store's per-tier slice to graph.json.
                #
                # No empty-projection guard here, and that is deliberate — the
                # two call sites of build_tier_graph_from_store are consistent,
                # not divergent.  commit_tier_slot's `if all_keyed:` branch is an
                # INPUT fallback: it takes a caller-supplied keyed list and, when
                # that list is empty, falls back to this same canonical store
                # projection rather than writing the caller's emptiness.  This
                # branch has no second input — it starts at the authority the
                # fallback reaches for, so there is nothing to fall back FROM.
                # Neither site suppresses an empty projection of a genuinely
                # empty store slice, and neither should: after the fold the store
                # is the post-fold truth, and a tier that ends with no keys must
                # end with no graph.json content.  Keeping the previous file
                # would resurrect retired keys on the next boot, since
                # DiskMemorySource hydrates entries from exactly these files.
                from paramem.memory.interim_adapter import adapter_slot_root_for_name

                for _pf_tier in ("episodic", "semantic", "procedural"):
                    _pf_root = adapter_slot_root_for_name(self.output_dir, _pf_tier)
                    _pf_root.mkdir(parents=True, exist_ok=True)
                    save_memory_to_disk(
                        build_tier_graph_from_store(self.store, _pf_tier),
                        _pf_root / "graph.json",
                    )
                    logger.info("_persist_fold: tier graph written to %s", _pf_root / "graph.json")

    def _run_fold(
        self,
        scope: "FoldScope",
        *,
        trainer=None,
        router=None,
        # interim-scope extras (only consumed when scope.persist == "interim_slot")
        adapter_name: "str | None" = None,
        stamp: "str | None" = None,
        run_label: str = "",
        triples_extracted: int = 0,
        episodic_rels: "list[dict] | None" = None,
        procedural_rels: "list[dict] | None" = None,
    ) -> dict:
        """Scope-parameterized consolidation fold spine — the single shared pipeline.

        Two scopes route through this method, each in either venue:

        - ``scope.persist == "interim_slot"`` (interim mini-fold): single-tier
          fold + :func:`~paramem.memory.persistence.commit_tier_slot`.
          Replaces the pipeline body of ``run_consolidation_cycle``.
        - ``scope.persist == "main_tiers"`` (full fold): multi-tier rebuild
          + :meth:`_persist_fold`.  Reached via :meth:`consolidate`.

        The fold does what it is told.  Whether there is anything to do at all is
        decided by the caller — the spine carries no content gate and no notion of
        who asked for the fold.

        **Both venues read the same store and run the same stage spine.**  The
        fold input is :class:`~paramem.memory.store.MemoryStore` in both cases —
        registry-true relations from ``_build_registry_true_relations``, hydrated
        from the per-tier and per-interim-slot registries at boot and after every
        cycle.  The fold does not trust that hydration to be complete: each
        fresh-derivation path opens with :meth:`_hydrate_store_for_fold`, which
        materialises every live key still missing from the entry cache out of the
        venue's source of truth.  All grooming stages
        (:meth:`_materialize_consolidation_graph`,
        :meth:`_refine_consolidation_graph`, :meth:`_promote_mature_keys_inline`,
        :meth:`_build_all_edge_entries_into`, the drift partition,
        :meth:`_apply_subtractive_removals_to_store`, the registry rewrite,
        :meth:`_build_tier_delta`) are shared and venue-agnostic.

        The two venues fork on ``scope.source`` at exactly two kinds of site:

        1. **Weight-only blocks with no simulate meaning** — the recall-miss
           reconstruction probe, ``main_tier_backup_scope``, the per-tier
           training loop, the PEFT interim unload, and the closing
           ``switch_adapter``.  In the disk venue ``self.model``
           is a bare base model with no ``peft_config``, so these are skipped.
        2. **The persist medium** — adapter weights vs per-tier ``graph.json``,
           dispatched inside :meth:`_persist_fold`.

        **Return schema:** always the same schema, in both venues and from every
        terminal return.

        **Mode-fork-guard invariant:** this method and all callers dispatch on
        ``scope.source`` / ``scope.persist`` structural enum attributes — never on
        a ``mode == "simulate"`` / ``mode == "train"`` string literal.  The
        ``mode`` string is computed internally, via :meth:`_venue_from_scope`,
        only where required by lower-level collaborators (``commit_tier_slot``,
        ``build_memory_source``) that are themselves in the allowlist.

        Args:
            scope: Immutable :class:`FoldScope` descriptor.  Selects pipeline stages
                and the persist venue.  Constructed by the thin public-method wrappers;
                never by app-layer callers.
            trainer: :class:`~paramem.server.background_trainer.BackgroundTrainer`
                instance.  Required for the per-tier re-arm pattern in the
                ``main_tiers`` weights venue; ``None`` for the disk venue and for
                ``interim_slot`` paths that do not need the abort-for-inference
                machinery.
            router: Router instance whose ``reload()`` is called at fold completion
                (``main_tiers`` path, both venues).  ``None`` is safe — skipped.
            adapter_name: Interim adapter name (``interim_slot`` path only).  Matches
                ``scope.tier``.
            stamp: Sub-interval stamp for :func:`~paramem.memory.persistence.commit_tier_slot`
                (``interim_slot`` path only).
            run_label: Tag woven into the wandb ``run_name`` for traceability
                (``interim_slot`` path only).
            triples_extracted: Number of episodic relations extracted this cycle
                (``interim_slot`` path only; carried through to the result dict).
            episodic_rels: Pre-extracted episodic relations, used only for the
                end-of-extraction debug snapshot (``interim_slot`` path only).
            procedural_rels: Pre-extracted procedural relations, used only for the
                end-of-extraction debug snapshot (``interim_slot`` path only).

        Returns:
            Result dict using the full train schema.  Fields present in all paths::

                {
                    "tiers_rebuilt": list[str],
                    "graph_drift_count": int,
                    "drift_deduplicated": int,
                    "drift_orphan": int,
                    "drift_genuine_loss": int,
                    "drift_intended_removal": int,
                    "drift_intended_removal_by_reason": dict,
                    "recall_miss_keys": list[str],
                    "keys_per_tier": dict[str, int],
                    "tier_keyed": dict,
                    "rolled_back": bool,
                    "rollback_tier": str | None,
                    "tier_delta": dict,
                }

            The ``interim_slot`` path additionally carries::

                {
                    "triples_extracted": int,
                    "new_keys": list[str],
                    "adapter_name": str | None,
                    "mode": "trained" | "simulated",
                    "venue": "train" | "simulate",
                    "error": str | None,
                    "episodic_train_loss": float | None,
                    "recall_failed_session_ids": list[str],
                }
        """
        # ------------------------------------------------------------------
        # interim mini-fold (scope.persist == "interim_slot")
        # ------------------------------------------------------------------
        # Source: weights (train) or disk (simulate) — reconstruct scoped to
        # tier for both.  Persist: commit_tier_slot (writes adapter weights
        # for train, graph.json sidecar for simulate).  Single-tier training
        # (weights only); promote=False.
        # Extracted from the training body of run_consolidation_cycle.
        # Transactional commit window: a raise from _persist_fold onward (or
        # from the simhash/store-write steps just before it) is compensated —
        # soft-stales re-activated, promotions reversed, the fresh interim
        # tier dropped wholesale — before the exception is re-raised, so a
        # failed commit leaves the store byte-identical to its pre-cycle state.
        # Unconditional across BOTH venues: the simulate path's store writes
        # live inside this same commit window (no separate pre-window put
        # loop), so a simulate-mode _persist_fold failure is compensated
        # identically to the weights path.
        # ------------------------------------------------------------------
        if scope.persist == "interim_slot":
            # --- interim-slot fold-stamp (minted before any store mutation) ---
            # scope.tier gives the logical tier; adapter_name is the PEFT slot name.
            _fold_stamp_b = self._compute_fold_stamp(tier=adapter_name or scope.tier)
            # Per-run telemetry ring key (see _new_telemetry_run_stamp) — NOT
            # _fold_stamp_b, which is a content fingerprint shared by every
            # run over an unchanged keyset.
            _telemetry_run_stamp_b = self._new_telemetry_run_stamp()

            # --- Crash-resume marker detection (interim fold) ---
            # Mirrors the main_tiers fold_stamp + fold_resume.json check below
            # (main-tiers branch): a persisted marker resumes ONLY when the
            # scope is unchanged.  fold_stamp alone is insufficient here — for
            # a brand-new interim tier active_keys_in_tier(adapter_name) is
            # empty pre-training, so fold_stamp is a constant (empty-keyset)
            # hash across ANY fresh slot's first cycle.  Pending-session
            # identity is sourced from episodic_rels/procedural_rels'
            # session_id field (stamped by _extract_and_start_training) since
            # the persisted train_assignment entries carry no session_id of
            # their own.  adapter_name doubling as the marker's single
            # train_assignment key gives stamp/cadence-window matching for
            # free — a marker minted for a different stamp never has this key.
            _resume_marker_b = self._read_fold_resume()
            _marker_ta_b: "dict[str, list[dict]]" = (
                _resume_marker_b.get("train_assignment", {}) if _resume_marker_b is not None else {}
            )
            _pending_session_ids_b = sorted(
                {
                    _rel.get("session_id", "")
                    for _rel in list(episodic_rels or []) + list(procedural_rels or [])
                    if _rel.get("session_id")
                }
            )
            _resume_b = (
                _resume_marker_b is not None
                and _resume_marker_b.get("scope") == "interim_slot"
                and _resume_marker_b.get("fold_stamp") == _fold_stamp_b
                and adapter_name in _marker_ta_b
                and sorted(_resume_marker_b.get("pending_session_ids", []))
                == _pending_session_ids_b
            )
            if _resume_marker_b is not None and not _resume_b:
                # Stale marker (different scope, fold_stamp, tier, or
                # pending-session set): clear it — the fresh-derivation path
                # below re-extracts and re-persists.  No scratch-tree removal
                # needed here (unlike main_tiers): each interim cycle's
                # output_dir is per-stamp (_training_output_dir(adapter_name,
                # interim_stamp=stamp)), so a stale marker minted for a
                # DIFFERENT stamp never points at the current scratch dir; a
                # same-stamp content mismatch is caught by trainer.py's own
                # dataset-fingerprint guard (_resolve_resume_checkpoint),
                # which purges stale checkpoints itself.
                self._clear_fold_resume()
                logger.info(
                    "_run_fold[interim]: cleared stale fold_resume.json (scope/"
                    "fold_stamp/tier/pending-session mismatch) — proceeding as"
                    " fresh cycle"
                )

            # _run_fold always controls the merger.graph lifecycle for the interim path.
            try:
                # --- End-of-extraction debug dump (interim only) ---
                # Interim-stamped root, like this path's cycle summary: these
                # are the interim cycle's own inputs, not the fold's.
                with self._artifact_scope(interim_stamp=stamp):
                    on_extraction_end(episodic_rels or [], procedural_rels or [])

                # --- Mint PEFT slot (weights source only) ---
                if scope.source == "weights":
                    from paramem.memory.interim_adapter import create_interim_adapter
                    from paramem.models.loader import ensure_adapter_matching

                    if adapter_name not in self.model.peft_config:
                        self.model = create_interim_adapter(self.model, self.episodic_config, stamp)
                        logger.info("_run_fold[interim]: created interim adapter %s", adapter_name)
                    else:
                        # Resident slot (re-fold within the cadence window):
                        # keep its weights (warm) unless the config no longer
                        # matches, in which case recreate cold (the same
                        # config-mismatch guard shared with the main-tier
                        # preamble below).  A retry after a recall-gate
                        # rejection never reaches this branch — the rejection
                        # handler below deletes the slot, so the `if` above
                        # re-mints it fresh/cold.
                        self.model = ensure_adapter_matching(
                            self.model, self.episodic_config, adapter_name
                        )

                if _resume_b:
                    # -------------------------------------------------------------
                    # RESUME FAST-PATH: skip re-extraction, rebuild the training
                    # set from the persisted marker so the dataset fingerprints
                    # identically and trainer.py's own staging_resume.json
                    # resumes the checkpoint (mirrors the main_tiers resume
                    # fast-path below).
                    # -------------------------------------------------------------
                    logger.info(
                        "_run_fold[interim]: CRASH-RESUME — fold_stamp + pending-session"
                        " scope match marker; rebuilding train set from persisted data"
                    )
                    if scope.source == "weights":
                        switch_adapter(self.model, adapter_name)
                    recall_miss_keys: "list[str]" = []
                    _tier_keyed: dict[str, list[dict]] = {
                        "episodic": [],
                        "procedural": [],
                        "semantic": [],
                    }
                    all_interim_keyed: "list[dict]" = []
                    new_keyed_interim: "list[dict]" = []
                    for _pe in _marker_ta_b.get(adapter_name, []):
                        _pt = _pe.get("tier", "episodic")
                        if "relation_type" in _pe:
                            # Present only for keys newly minted pre-crash — see
                            # _persisted_from_entry_and_rec, the enrichment
                            # applied at persist time below.
                            _rec = _rec_from_persisted(_pe)
                            _entry = _rec["entry"]
                            new_keyed_interim.append(_rec)
                        else:
                            _entry = {
                                "key": _pe["key"],
                                "subject": _pe["subject"],
                                "predicate": _pe["predicate"],
                                "object": _pe["object"],
                                "speaker_id": _pe["speaker_id"],
                            }
                        all_interim_keyed.append(_entry)
                        _tier_keyed[_pt].append(_entry)
                    new_key_ids = [r["entry"]["key"] for r in new_keyed_interim]
                    # Recall-miss diagnostics are NOT re-derived on resume —
                    # non-training-critical, and safely re-evaluated on the
                    # NEXT (non-resumed) cycle.  Accepted resume-path
                    # divergence — mirrors main_tiers' drift-counter zeroing
                    # on resume below.
                else:
                    # -------------------------------------------------------------
                    # FRESH-DERIVATION PATH: hydrate -> materialize -> refine -> build set.
                    # -------------------------------------------------------------
                    # --- Hydrate: every live key must have content before the
                    # store is read, or the finalize step deregisters whatever
                    # the cache happened to be missing.  Runs AFTER the interim
                    # slot is minted so the weight source sees the current model.
                    self._hydrate_store_for_fold(scope)

                    # --- Materialize: recall-miss diagnostic + rebuild keying surface ---
                    # Scoped to the current slot: reconstruct only the slot's registered keys
                    # (tier=adapter_name) for the recall-miss diagnostic, then reset and re-merge:
                    #   (a) registry-true relations for this slot
                    #   (b) the pending-session relations captured from merger.graph before the
                    #       reset (extra_relations), so they survive the graph reset.
                    _extra: "list[Relation] | None" = (
                        self._capture_pending_relations() if scope.consume_pending else None
                    )

                    _slot_keys: "list[str]" = list(
                        self.store.active_keys_in_tier(adapter_name or scope.tier)
                    )

                    # --- Interim recital dedup (unconditional) ---
                    # Scope the dedup targets to main-tier keys whose SPO touches
                    # an entity present in THIS cycle's pending-session relations
                    # (_extra) — no entity-to-key index exists, so this reads
                    # registry SPO directly rather than rebuilding a graph.  A
                    # recited fact IS in _extra, so its entities are in
                    # _session_entities, so its main-tier twin is always in scope
                    # (Case-1 can never miss a legitimate target).  A recital's
                    # reinforcement is credited to the surviving main key via the
                    # merger's adopt_reinforcements accumulator, consumed by
                    # _refine_consolidation_graph's bump loop.
                    _session_entities = {r.subject for r in (_extra or [])} | {
                        r.object for r in (_extra or [])
                    }
                    _slot_keys_set = set(_slot_keys)

                    def _dedup_touches_session(_dk: str) -> bool:
                        # _hydrate_store_for_fold ran above, so a miss here means
                        # the venue's source of truth holds no content for this
                        # live key — it cannot be a dedup target for anything.
                        _dk_entry = self.store.get(_dk)
                        if _dk_entry is None:
                            return False
                        return (
                            canonical(_dk_entry.get("subject", "")) in _session_entities
                            or canonical(_dk_entry.get("object", "")) in _session_entities
                        )

                    _dedup_keys: "list[str]" = [
                        _dk
                        for _dk_tier in ("episodic", "semantic", "procedural")
                        for _dk in self.store.active_keys_in_tier(_dk_tier)
                        if _dk not in _slot_keys_set and _dedup_touches_session(_dk)
                    ]

                    recall_miss_keys, recon_relations = self._materialize_consolidation_graph(
                        tier=scope.tier,
                        keys=_slot_keys,
                        extra_relations=_extra,
                        dedup_target_keys=_dedup_keys,
                        resolve_contradictions_recon=(self.config.refinement_contradiction == "on"),
                        resolve_contradictions_extra=(self.config.refinement_contradiction == "on"),
                    )
                    if recall_miss_keys:
                        logger.info(
                            "_run_fold[interim]: %d recall-miss key(s) in slot %s "
                            "(kept in training set with registry-true content)",
                            len(recall_miss_keys),
                            adapter_name,
                        )

                    # --- Refine ---
                    self._refine_consolidation_graph(
                        recon_relations,
                        normalize=scope.normalize,
                        enrich=scope.enrich,
                    )

                    # --- Build keyed training set ---
                    if scope.source == "weights":
                        switch_adapter(self.model, adapter_name)
                    _tier_keyed = {
                        "episodic": [],
                        "procedural": [],
                        "semantic": [],
                    }
                    _, _deferred_writes = self._build_all_edge_entries_into(
                        _tier_keyed,
                        defer=scope.defer,
                        tag_new=scope.tag_new,
                        exclude_keys=set(_dedup_keys),
                    )

                    all_interim_keyed = _tier_keyed["episodic"] + _tier_keyed["procedural"]
                    new_keyed_episodic = [r for r in _deferred_writes if r["tier"] == "episodic"]
                    new_keyed_proc = [r for r in _deferred_writes if r["tier"] == "procedural"]
                    new_keyed_interim = new_keyed_episodic + new_keyed_proc
                    new_key_ids = [r["entry"]["key"] for r in new_keyed_interim]

                    # NOTE: the simulate path (scope.source != "weights") no longer
                    # applies its store mutations here.  There is no training step
                    # to gate on for simulate, but the writes are otherwise
                    # identical to the weights path's deferred-mutation loop below
                    # (same new_keyed_interim list) — both now go through the SAME
                    # commit-window try so a simulate-mode _persist_fold failure is
                    # compensated identically to the weights path (see the commit
                    # window below).

                    # --- interim slot: write single-entry fold_resume.json marker ---
                    # Written AFTER all_interim_keyed is fully finalized — not
                    # at fold entry.  Persists the REAL train_assignment via
                    # _persisted_from_entry_and_rec: entry
                    # dicts enriched with "tier" and, for newly-minted keys, the
                    # deferred-write metadata the commit window below needs
                    # (relation_type/session_ids/last_seen/first_seen —
                    # new_keyed_interim's rec shape carries fields the uniform
                    # tier_keyed entry shape does not) — so a crash-resume rebuilds
                    # an IDENTICAL training dataset AND an identical commit-window
                    # write set.  On crash, the marker enables epoch-resume via
                    # _resolve_resume_checkpoint (the epoch checkpoint path is
                    # already wired) — see the RESUME FAST-PATH branch above, which
                    # reverses this enrichment via _rec_from_persisted.
                    # Interim does NOT pass retain_scratch_until_external_commit:
                    # commit_tier_slot is an inline durable write right after training,
                    # so there is no multi-tier window where a completed-but-uncommitted
                    # tier can be lost.
                    if scope.source == "weights":
                        _new_meta_by_key = {r["entry"]["key"]: r for r in new_keyed_interim}
                        _b_persist_entries: "list[dict]" = []
                        for _pt2, _pt2_entries in (
                            ("episodic", _tier_keyed["episodic"]),
                            ("procedural", _tier_keyed["procedural"]),
                        ):
                            for _pe2 in _pt2_entries:
                                _b_persist_entries.append(
                                    _persisted_from_entry_and_rec(
                                        _pe2, _pt2, _new_meta_by_key.get(_pe2["key"])
                                    )
                                )
                        _b_assignment = {adapter_name: _b_persist_entries}

                        _b_dataset_fingerprints: dict[str, str] = {}
                        if all_interim_keyed:
                            _b_dataset_fingerprints[adapter_name] = _fingerprint_entries(
                                all_interim_keyed
                            )
                        self._persist_fold_assignment(
                            "interim_slot",
                            _fold_stamp_b,
                            _b_assignment,
                            _b_dataset_fingerprints,
                            pending_session_ids=_pending_session_ids_b,
                        )

                # --- Train (weights source) or skip (disk source) ---
                epi_train_loss: "float | None" = None
                if scope.source == "weights" and all_interim_keyed:
                    # --- Per-tier device-saturation telemetry ---
                    # Same pattern as the main-tiers train call (see comment
                    # there): bare snapshots only, never vram_measure; the
                    # peak is process-wide, not adapter-attributable. This
                    # try/finally exists only to run the measurement — it has
                    # no ``except``, so it never alters the exception type
                    # reaching the caller.
                    from paramem.memory.interim_adapter import INTERIM_NAME_PREFIX

                    _telemetry_int_free_before: int | None = None
                    _telemetry_int_total: int | None = None
                    _telemetry_int_n_keys = len(all_interim_keyed)
                    # Derived here (not read off self.training_config.num_epochs)
                    # so the finally-path telemetry below records the TRUE budget
                    # even when training raises. _train_tier_adapter derives the
                    # identical value from the same n_keys input -- budget_for is
                    # pure, so the two calls agree.
                    _telemetry_int_epochs, _telemetry_int_accum, _ = budget_for(
                        _telemetry_int_n_keys
                    )
                    # Measured BEFORE training starts -- same enclosing-scope
                    # hoist as the budget derivation above, so the finally-path
                    # record below carries the true pre-training weight state
                    # even when training raises.
                    _telemetry_int_init = measured_adapter_init_state(self.model, adapter_name)
                    _telemetry_int_stale = len(self.store.stale_keys_in_tier(adapter_name))
                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()
                        _telemetry_int_free_before, _telemetry_int_total = torch.cuda.mem_get_info()
                    recall_state = None
                    epi_metrics = None
                    try:
                        epi_metrics, recall_state = self._train_tier_adapter(
                            all_interim_keyed,
                            adapter_name=adapter_name,
                            adapter_config=self.episodic_config,
                            training_config=self.training_config,
                            output_dir=self._training_output_dir(adapter_name, interim_stamp=stamp),
                            run_name=f"interim-{adapter_name}-{run_label}",
                            phase_name=f"interim-{adapter_name}-{run_label}",
                        )
                    finally:
                        # The ENTIRE record build (not just the write below) is
                        # inside this try/except: constructing the dict reads
                        # self.model.peft_config and calls _recall_bind_telemetry,
                        # either of which could raise on a sufficiently broken
                        # state, and a raise here in a bare finally (unguarded)
                        # would REPLACE an in-flight exception from the try above
                        # (e.g. AbortedDuringConsolidation) with whatever this
                        # construction raised -- silently misrouting an abort to
                        # the crash-incident path. Losing a telemetry record is
                        # strictly preferable to that.
                        try:
                            # aborted is a NORMAL RETURN VALUE from
                            # _train_tier_adapter (the trainer's own
                            # thermal-throttle/operator-pause signal), not a
                            # raised exception -- epi_metrics is bound whenever
                            # training returns at all (aborted or not) and stays
                            # at its pre-declared None only when the call above
                            # actually raised.
                            _int_aborted = bool(
                                epi_metrics.get("aborted") if epi_metrics is not None else False
                            )
                            # Budget/bind/init/stale fields do not depend on CUDA
                            # introspection -- the record is always written; only
                            # the VRAM fields below are conditional on it.
                            _telemetry_int_record: dict = {
                                "tier": adapter_name,
                                "fold_stamp": _fold_stamp_b,
                                "adapter_count": len(self.model.peft_config),
                                "interim_count": len(
                                    [
                                        a
                                        for a in self.model.peft_config
                                        if a.startswith(INTERIM_NAME_PREFIX)
                                    ]
                                ),
                                "epochs": _telemetry_int_epochs,
                                "n_keys": _telemetry_int_n_keys,
                                "accum": _telemetry_int_accum,
                                # Always 0 at the interim site by construction:
                                # a fresh/resumed interim slot's registry only
                                # gains stale entries in the commit window AFTER
                                # this measurement, never before it. The signal
                                # this field carries lives at the main-tier site
                                # (measured after prior cycles' commit windows);
                                # kept here too for one schema across both kinds.
                                "stale_keys": _telemetry_int_stale,
                                "aborted": _int_aborted,
                            }
                            if _telemetry_int_init is not None:
                                _telemetry_int_record["init"] = _telemetry_int_init
                            # _train_tier_adapter tags its returned metrics dict
                            # when donor seeding actually copied weights this
                            # fold -- override the PRE-training "cold" measurement
                            # above with "donor" rather than re-measuring (one
                            # measurement, per the funnel's own docstring). Only
                            # reachable when epi_metrics is bound (the success/
                            # abort return path); the exception path leaves the
                            # measured value untouched, as intended.
                            if epi_metrics is not None and epi_metrics.get("donor_seeded"):
                                _telemetry_int_record["init"] = "donor"
                            _int_epochs_to_bind, _int_steps_to_bind, _int_hit_cap = (
                                _recall_bind_telemetry(
                                    recall_state, _telemetry_int_n_keys, _telemetry_int_accum
                                )
                            )
                            if _int_epochs_to_bind is not None:
                                _telemetry_int_record["epochs_to_bind"] = _int_epochs_to_bind
                                _telemetry_int_record["steps_to_bind"] = _int_steps_to_bind
                            # hit_cap is suppressed on the abort path: stop_epoch
                            # is None whenever the trainer never reached (or
                            # never signalled) recall convergence, and an abort
                            # is exactly such a case -- emitting hit_cap=True
                            # there would be indistinguishable from a genuine
                            # "budget too small" outcome in the bucket re-fit.
                            if not _int_aborted and _int_hit_cap is not None:
                                _telemetry_int_record["hit_cap"] = _int_hit_cap
                            if torch.cuda.is_available() and _telemetry_int_free_before is not None:
                                _telemetry_int_peak = torch.cuda.max_memory_allocated()
                                # peak_reserved is the OOM-relevant quantity: the
                                # caching allocator raises when it cannot reserve,
                                # not when driver-free (mem_get_info) drops —
                                # driver-free counts cached-but-unused allocator
                                # segments as used, which peak_reserved does not.
                                _telemetry_int_peak_reserved = torch.cuda.max_memory_reserved()
                                _telemetry_int_free_after = torch.cuda.mem_get_info()[0]
                                logger.info(
                                    "_run_fold[interim]: telemetry interim-train[%s] "
                                    "(device-saturation indicator, not adapter cost) — "
                                    "free_before=%d free_after=%d peak_alloc=%d peak_reserved=%d",
                                    adapter_name,
                                    _telemetry_int_free_before,
                                    _telemetry_int_free_after,
                                    _telemetry_int_peak,
                                    _telemetry_int_peak_reserved,
                                )
                                _telemetry_int_record.update(
                                    {
                                        "free_before": _telemetry_int_free_before,
                                        "free_after": _telemetry_int_free_after,
                                        "peak_alloc": _telemetry_int_peak,
                                        "peak_reserved": _telemetry_int_peak_reserved,
                                        "total": _telemetry_int_total,
                                    }
                                )
                            if self._telemetry_dir is not None:
                                record_fold_telemetry(
                                    self._telemetry_dir,
                                    cycle_stamp=_telemetry_run_stamp_b,
                                    kind="interim_tier_train",
                                    record=_telemetry_int_record,
                                )
                        except Exception:  # noqa: BLE001  # boundary: telemetry
                            # runs in a finally on the exception path too — a
                            # failure anywhere in record construction OR the
                            # write (disk full, permissions, corrupt store, a
                            # broken model/store attribute) must never replace
                            # the in-flight exception (e.g. AbortedDuringConsolidation
                            # would get swapped for the construction/write error
                            # and misrouted). Losing a telemetry record is
                            # strictly preferable.
                            logger.warning(
                                "_run_fold[interim]: telemetry write failed for %s",
                                adapter_name,
                                exc_info=True,
                            )
                    epi_train_loss = (
                        epi_metrics.get("train_loss") if epi_metrics is not None else None
                    )
                    if epi_metrics is not None and epi_metrics.get("aborted"):
                        logger.info("_run_fold[interim]: training aborted — skipping commit")
                        return {"mode": "aborted", "adapter_name": adapter_name}

                    _epi_passing = self._recall_passing_keys(recall_state, all_interim_keyed)
                    if _epi_passing is None:
                        _epi_passing = self._probe_passing_keys(adapter_name, all_interim_keyed)
                else:
                    _epi_passing = None

                # --- Commit window: simhash registration, deferred store writes,
                # subtractive soft-stales, and the durable persist — all-or-nothing.
                # A raise anywhere in this window (including inside _persist_fold)
                # means the interim commit failed; the in-memory mutations already
                # applied this cycle are compensated: soft-staled pre-existing
                # keys are re-activated, then the fresh interim tier is dropped
                # wholesale — so the store is left byte-identical to its
                # pre-cycle state before the exception is re-raised unchanged
                # (caller retry/pinning semantics untouched).
                # Counters are deliberately NOT part of the compensated state —
                # they are only incremented after this block succeeds, so a
                # failure leaves them unadvanced with no capture/restore needed.
                _passing_interim = (
                    [kp for kp in all_interim_keyed if kp["key"] in _epi_passing]
                    if _epi_passing is not None
                    else all_interim_keyed
                )
                _recall_failed_session_ids: set[str] = set()
                _recall_gate_rejected = False
                _ep_flushed = 0
                _proc_flushed = 0
                _soft_stale_by_tier: dict[str, dict] = {}
                try:
                    # Update interim simhash registry.
                    self.store.replace_simhashes_in_tier(
                        adapter_name, build_registry(_passing_interim)
                    )

                    # --- Apply deferred interim store mutations (both venues) ---
                    # Weights source gates on the post-training recall verdict
                    # (_epi_passing); simulate has no training step, so
                    # _epi_passing is None (set in the else branch above) and
                    # every key passes through unconditionally — the same
                    # "no verdict admits all" rule already used for a
                    # weights-source cycle with early-stop disabled.
                    for rec in new_keyed_interim:
                        _entry = rec["entry"]
                        _key = _entry["key"]
                        if _epi_passing is not None and _key not in _epi_passing:
                            logger.debug(
                                "_run_fold[interim]: key %s failed recall gate"
                                " — skipping registration",
                                _key,
                            )
                            _recall_failed_session_ids.update(rec.get("session_ids", []))
                            continue
                        self.store.put(
                            adapter_name,
                            _key,
                            _entry,
                            simhash=entry_simhash(_entry),
                        )
                        self.store.set_bookkeeping(
                            _key,
                            speaker_id=rec["speaker_id"],
                            relation_type=rec["relation_type"],
                            reinforcement_count=1,
                            last_reinforced_cycle=self.cycle_count,
                            last_seen=rec.get("last_seen", ""),
                            first_seen=rec.get("first_seen", ""),
                            allow_empty_speaker=(rec["speaker_id"] == ""),
                        )
                        if rec["tier"] == "procedural":
                            _proc_flushed += 1
                        else:
                            _ep_flushed += 1

                    # --- Shared soft-stale stage ---
                    _soft_stale_by_tier = self._apply_subtractive_removals_to_store(
                        scope=scope.subtractive_scope
                    )

                    # --- Persist interim slot ---
                    self._persist_fold(
                        scope,
                        adapter_name=adapter_name,
                        stamp=stamp,
                        all_keyed=all_interim_keyed,
                    )
                except RecallGateRejected as _gate:
                    # Deterministic quality verdict, not a crash.  Roll back
                    # identically to the generic handler below, then RETURN
                    # normally with the contributing session ids so app.py's
                    # retry bookkeeping runs — it is written against a normal
                    # cycle return and is skipped entirely when this
                    # propagates.  commit_tier_slot has already removed the
                    # un-flushed slot in its own finally.
                    for _stale_tier, _stale_keys in _soft_stale_by_tier.items():
                        for _stale_key in _stale_keys:
                            self.store.reactivate(_stale_tier, _stale_key)
                    self.store.drop_tier(adapter_name)
                    # Drop the rejected slot from VRAM too — commit_tier_slot's
                    # finally only removes the disk artifact, and
                    # _verify_saved_adapter_from_disk restores adapter_name as
                    # the active adapter before raising, so without this the
                    # trained-but-rejected weights stay resident and a
                    # same-window retry would silently warm-start from a state
                    # that exists nowhere on disk.  Rejection now leaves
                    # neither a disk slot nor a VRAM slot — deterministic cold
                    # re-entry, matching disk truth and surviving restarts
                    # identically.  The mint guard above (``if adapter_name
                    # not in self.model.peft_config``) recreates the slot
                    # fresh on the next fold attempt, so re-entry's init state
                    # is whatever the standard mechanism provides (cold today,
                    # donor-seeded when that mechanism is enabled) rather than
                    # the rejected checkpoint.  Switch off the slot before
                    # deleting it — PEFT's delete_adapter silently reassigns
                    # the active adapter to whichever resident adapter it
                    # encounters first when the deleted one was active, which
                    # would leave the post-rejection active adapter
                    # non-deterministic; switching to episodic first (the same
                    # pattern already used by _verify_saved_adapter_from_disk's
                    # own verify-slot teardown) keeps it deterministic.  The
                    # "Restore episodic as active adapter" step below is
                    # idempotent on an already-active episodic and never
                    # touches a slot this block already removed.
                    if adapter_name in self.model.peft_config:
                        if "episodic" in self.model.peft_config:
                            switch_adapter(self.model, "episodic")
                        self.model.delete_adapter(adapter_name)
                        logger.info(
                            "_run_fold[interim]: deleted rejected interim adapter"
                            " %s from VRAM — retry starts cold",
                            adapter_name,
                        )
                    _recall_gate_rejected = True
                    _recall_failed_session_ids.update(_pending_session_ids_b)
                    logger.warning(
                        "_run_fold[interim]: recall gate rejected %s "
                        "(recall %.3f < threshold %.2f) — %d session(s) stay pending",
                        adapter_name,
                        _gate.recall_rate,
                        _gate.threshold,
                        len(_pending_session_ids_b),
                    )
                except Exception:
                    # Undo this cycle's mutations before re-raising: reactivate
                    # any key the shared soft-stale stage staled this cycle
                    # (reversing _apply_subtractive_removals_to_store), then
                    # drop the freshly-minted interim tier wholesale — restoring
                    # the store to its pre-cycle state.
                    for _stale_tier, _stale_keys in _soft_stale_by_tier.items():
                        for _stale_key in _stale_keys:
                            self.store.reactivate(_stale_tier, _stale_key)
                    self.store.drop_tier(adapter_name)
                    raise

                # Counters advance only after the commit above succeeded —
                # for both venues; simulate's counters are no longer bumped
                # eagerly outside the commit window (see the NOTE in the
                # fresh-derivation path above).
                # A recall-gate rejection rolled the mutations back above, so the
                # keys were never committed and the counters must not advance —
                # otherwise the next cycle mints from a gap and the rejected key
                # numbers are burned.
                if not _recall_gate_rejected:
                    self._indexed_next_index += _ep_flushed
                    self._procedural_next_index += _proc_flushed

                # Clear the interim-slot fold_resume.json marker.  On rejection
                # the slot is already gone (commit_tier_slot's finally), so a
                # surviving marker would point at a deleted slot.
                self._clear_fold_resume()

                # --- Restore episodic as active adapter ---
                if "episodic" in self.model.peft_config:
                    switch_adapter(self.model, "episodic")

                if _recall_gate_rejected:
                    _interim_mode_label = "recall_failed"
                else:
                    _interim_mode_label = "trained" if scope.source == "weights" else "simulated"
                _interim_venue = self._venue_from_scope(scope)
                logger.info(
                    "_run_fold[interim]: %s %s — %d new keys, %d total interim keys",
                    _interim_mode_label,
                    adapter_name,
                    len(new_key_ids),
                    len(all_interim_keyed),
                )

                cycle_summary = {
                    "triples_extracted": triples_extracted,
                    "new_keys": new_key_ids,
                    "adapter_name": adapter_name,
                    "mode": _interim_mode_label,
                    "venue": _interim_venue,
                    "error": None,
                    "episodic_train_loss": epi_train_loss,
                    "recall_failed_session_ids": sorted(_recall_failed_session_ids),
                    # Full schema fields (zeros/empties for interim path callers that
                    # don't use them — ensures the dict is always a superset of the
                    # train schema so generic callers never KeyError).
                    "tiers_rebuilt": [adapter_name] if scope.source == "weights" else [],
                    "graph_drift_count": 0,
                    "drift_deduplicated": 0,
                    "drift_orphan": 0,
                    "drift_genuine_loss": 0,
                    "drift_intended_removal": 0,
                    "drift_intended_removal_by_reason": {},
                    "recall_miss_keys": sorted(recall_miss_keys),
                    "keys_per_tier": {
                        "episodic": len(_tier_keyed["episodic"]),
                        "procedural": len(_tier_keyed["procedural"]),
                    },
                    "tier_keyed": _tier_keyed,
                    "rolled_back": False,
                    "rollback_tier": None,
                    "tier_delta": {},
                }
                with self._artifact_scope(interim_stamp=stamp):
                    on_cycle_end(cycle_summary)
                return cycle_summary
            finally:
                self.merger.reset_graph()

        # ------------------------------------------------------------------
        # main-tiers full fold (scope.persist == "main_tiers")
        # ------------------------------------------------------------------
        # Store-sourced in both venues.  The weights venue additionally probes
        # the adapters for recall misses, backs the main tiers up, retrains
        # them, and saves the weights; the disk venue skips those blocks and
        # persists per-tier graph.json instead (see _persist_fold).  Every
        # other stage — promote, drift partition, registry
        # rewrite, tier delta — runs identically in both.
        # ------------------------------------------------------------------
        from paramem.memory.interim_adapter import (
            INTERIM_NAME_PREFIX,
            unload_interim_adapters,
        )
        from paramem.models.loader import (
            create_adapter,
            ensure_adapter_matching,
            main_tier_backup_scope,
        )

        # --- Fold-stamp + crash-resume marker (full fold) ---
        # Mint fold_stamp BEFORE any store mutation (promote /
        # _build_all_edge_entries_into both mutate the store; the stamp must
        # reflect the pristine on-disk registry so it is byte-identical on
        # re-entry after a crash).
        _fold_stamp_c = self._compute_fold_stamp(tier=None)
        # Per-run telemetry ring key (see _new_telemetry_run_stamp) — NOT
        # _fold_stamp_c, which is a content fingerprint shared by every run
        # over an unchanged keyset (the common steady-state refresh case).
        _telemetry_run_stamp_c = self._new_telemetry_run_stamp()
        _resume_marker = self._read_fold_resume()
        _resume_c = (
            _resume_marker is not None
            and _resume_marker.get("fold_stamp") == _fold_stamp_c
            and _resume_marker.get("scope") == "main_tiers"
        )
        if _resume_marker is not None and not _resume_c:
            # Stale marker (different fold inputs or scope): clear it and
            # delete any retained checkpoint scratch from that stale fold.
            import shutil as _shutil

            _stale_refresh = self.output_dir / "consolidation_refresh"
            if _stale_refresh.exists():
                _shutil.rmtree(_stale_refresh, ignore_errors=True)
                logger.info(
                    "_run_fold[main_tiers]: removed stale consolidation_refresh tree"
                    " from prior mismatched fold"
                )
            self._clear_fold_resume()
            logger.info(
                "_run_fold[main_tiers]: cleared stale fold_resume.json"
                " (fold_stamp or scope mismatch) — proceeding as fresh fold"
            )

        try:
            # -----------------------------------------------------------------
            # RESUME FAST-PATH: skip derivation, rebuild from persisted marker.
            # -----------------------------------------------------------------
            if _resume_c:
                logger.info(
                    "_run_fold[main_tiers]: CRASH-RESUME — fold_stamp matches marker;"
                    " rebuilding train_assignment from persisted data"
                )
                _marker_ta: "dict[str, list[dict]]" = _resume_marker.get(  # type: ignore[union-attr]
                    "train_assignment", {}
                )
                tier_keyed = {
                    t: list(_marker_ta.get(t, [])) for t in ("episodic", "semantic", "procedural")
                }
                recall_miss_keys: list[str] = []
                minted_by_tier: dict = {}
                _train_active_before: dict[str, int] = {
                    t: len(tier_keyed[t]) for t in ("episodic", "semantic", "procedural")
                }
                # Drift counters zero on resume. Finalize never ran pre-crash, so drift
                # soft-stale flips were NOT durably applied — they are intentionally skipped
                # here (accepted divergence, affects only non-assigned duplicate/contradiction
                # keys, never primary facts). Accepted as an intentional resume-path divergence.
                graph_drift_count = 0
                drift_deduplicated_count = 0
                drift_orphan_count = 0
                drift_genuine_loss_count = 0
                drift_intended_removal_count = 0
                drift_intended_removal_by_reason: dict[str, int] = {}
                soft_stale_by_tier: dict[str, dict] = {}
                _soft_stale_keys: set[str] = set()
                # Fingerprints come from the marker (already computed pre-crash).
                _resume_fingerprints: "dict[str, str]" = _resume_marker.get(  # type: ignore[union-attr]
                    "dataset_fingerprint", {}
                )
                _dataset_fingerprints = _resume_fingerprints
            else:
                # -----------------------------------------------------------------
                # FRESH-DERIVATION PATH: hydrate → reconstruct → promote → assign.
                # -----------------------------------------------------------------
                # --- Hydrate: every live key must have content before the store
                # is read, or the finalize step below rewrites each main-tier
                # registry without whatever the cache happened to be missing.
                self._hydrate_store_for_fold(scope)

                # Capture pending-session relations from merger.graph BEFORE
                # _materialize_consolidation_graph resets the graph (ordering:
                # capture-before-reset, re-merge-after-reset via extra_relations).
                # Only active when scope.consume_pending is True (the consume-pending
                # full fold, where app.py has pre-populated merger.graph).
                # The fast-path resume branch above intentionally has NO capture —
                # the persisted fold_resume.json marker already carries the folded
                # pending facts in its train_assignment.
                _pending_extra: "list[Relation] | None" = None
                if scope.consume_pending:
                    _pending_extra = self._capture_pending_relations()
                    logger.info(
                        "_run_fold[main_tiers]: consume-pending — captured %d pending relation(s)",
                        len(_pending_extra),
                    )
                recall_miss_keys, recon_relations = self._materialize_consolidation_graph(
                    source=scope.source,
                    keys=self._fold_active_keys(scope),
                    resolve_contradictions_recon=(self.config.refinement_contradiction == "on"),
                    resolve_contradictions_extra=(self.config.refinement_contradiction == "on"),
                    extra_relations=_pending_extra,
                )
                self._refine_consolidation_graph(
                    recon_relations,
                    normalize=scope.normalize,
                    enrich=scope.enrich,
                )

                # --- Inline promotion (scope-gated) ---
                if scope.promote:
                    _inline_promoted = self._promote_mature_keys_inline()
                    if _inline_promoted:
                        logger.info(
                            "_run_fold[main_tiers]: %d key(s) promoted to semantic "
                            "before tier assignment",
                            len(_inline_promoted),
                        )

                tier_keyed: dict[str, list[dict]] = {
                    "episodic": [],
                    "semantic": [],
                    "procedural": [],
                }

                minted_by_tier, _ = self._build_all_edge_entries_into(
                    tier_keyed,
                    defer=scope.defer,
                    tag_new=scope.tag_new,
                )

                if recall_miss_keys:
                    logger.info(
                        "_run_fold[main_tiers]: %d key(s) in recall-miss set "
                        "(retrained with registry-true content — not dropped): %s",
                        len(recall_miss_keys),
                        sorted(recall_miss_keys),
                    )

                on_fold_graph(self.merger.graph, label="keyed")

                on_fold_assignments(tier_keyed)

                _train_active_before: dict[str, int] = {
                    t: len(tier_keyed[t]) for t in ("episodic", "semantic", "procedural")
                }

                # end of fresh-derivation path.
                # Compute dataset fingerprints and persist the fold assignment marker
                # now that the assignment is final (there is no further mutation
                # between here and the per-tier training loop below).
                # Fingerprint is over sorted SPO tuples, NOT tokenized examples.
                # Calling format_entry_training here (before the per-tier loop) would
                # interfere with per-tier format spy patterns in existing tests and is
                # unnecessary — SPO identity is the only change-detection signal needed.
                _dataset_fingerprints: dict[str, str] = {}
                for _t in ("episodic", "semantic", "procedural"):
                    _ta_entries = tier_keyed[_t]
                    if _ta_entries:
                        _dataset_fingerprints[_t] = _fingerprint_entries(_ta_entries)
                self._persist_fold_assignment(
                    "main_tiers", _fold_stamp_c, tier_keyed, _dataset_fingerprints
                )

            # --- Drift partition (fresh-fold only; skipped on crash-resume) ---
            # On crash-resume, drift was already applied pre-crash and registries
            # are pristine.  Re-running subtractive removals would double-apply.
            # Counters are pre-zeroed in the resume fast-path above.
            if not _resume_c:
                _all_keyed = {e["key"] for tier_list in tier_keyed.values() for e in tier_list}

                for _surviving_key in _all_keyed:
                    _sbk = self.store.bookkeeping_for_key(_surviving_key)
                    if _sbk is not None:
                        _sbk["last_reinforced_cycle"] = self.cycle_count

                _subtractive_stale_by_tier = self._apply_subtractive_removals_to_store(
                    scope=scope.subtractive_scope
                )

                # Drift is measured against the keys THIS fold owns: a key the
                # fold never read cannot have drifted out of its merged graph.
                _drift_keys = [k for k in self._fold_active_keys(scope) if k not in _all_keyed]

                _collapsed_set: set[str] = set(getattr(self.merger, "collapsed", []))
                _ledger: dict[str, dict] = getattr(self.merger, "removal_ledger", {})

                drift_deduplicated: list[str] = []
                drift_orphan: list[str] = []
                drift_genuine_loss: list[str] = []
                drift_intended_removal: list[str] = []
                drift_intended_removal_by_reason = {}

                soft_stale_by_tier = {
                    tier: dict(entries) for tier, entries in _subtractive_stale_by_tier.items()
                }

                for _dk in _drift_keys:
                    if _dk in _collapsed_set:
                        drift_deduplicated.append(_dk)
                        _dk_tier = self.store.tier_for_active_key(_dk)
                        _dk_simhash: "int | None" = None
                        if _dk_tier is not None:
                            _dk_simhash = self.store.simhash(_dk_tier, _dk)
                        self.store.discard_keys([_dk], mode="stale")
                        if _dk_tier is not None:
                            _stale_rec = {"stale_cycles": 0}
                            if _dk_simhash is not None:
                                _stale_rec["simhash"] = _dk_simhash
                            soft_stale_by_tier.setdefault(_dk_tier, {})[_dk] = _stale_rec
                    elif _dk in _ledger:
                        drift_intended_removal.append(_dk)
                        _r = _ledger[_dk]["reason"]
                        drift_intended_removal_by_reason[_r] = (
                            drift_intended_removal_by_reason.get(_r, 0) + 1
                        )
                    else:
                        _dk_bk = self.store.bookkeeping_for_key(_dk)
                        _dk_entry = self.store.get(_dk)
                        _entry_subj = (_dk_entry or {}).get("subject", "")
                        _entry_pred = (_dk_entry or {}).get("predicate", "")
                        _entry_obj = (_dk_entry or {}).get("object", "")
                        _bk_subj = (_dk_bk or {}).get("subject", "")
                        _bk_pred = (_dk_bk or {}).get("predicate", "")
                        _bk_obj = (_dk_bk or {}).get("object", "")
                        if not _entry_subj and not _entry_pred and not _entry_obj:
                            if _bk_subj or _bk_pred or _bk_obj:
                                drift_genuine_loss.append(_dk)
                            else:
                                drift_orphan.append(_dk)
                        else:
                            drift_genuine_loss.append(_dk)

                graph_drift_count = len(_drift_keys)
                drift_deduplicated_count = len(drift_deduplicated)
                drift_orphan_count = len(drift_orphan)
                drift_genuine_loss_count = len(drift_genuine_loss)
                drift_intended_removal_count = len(drift_intended_removal)

                _soft_stale_keys = {
                    k for tier_stale in soft_stale_by_tier.values() for k in tier_stale
                }
                _stale_in_active = _soft_stale_keys & _all_keyed
                if _stale_in_active:
                    logger.warning(
                        "_run_fold[main_tiers]: invariant violation — %d key(s) appear"
                        " in both soft_stale_by_tier and _all_keyed (trained as active AND"
                        " stale); this indicates tier_keyed was mutated after _all_keyed"
                        " was built: %s",
                        len(_stale_in_active),
                        sorted(_stale_in_active),
                    )

                for _dk in drift_deduplicated:
                    _dk_entry = self.store.get(_dk)
                    logger.info(
                        "graph_drift_key key=%s bucket=deduplicated"
                        " subject=%r predicate=%r object=%r"
                        " (registry-true duplicate — soft-staled; record retained"
                        " for stale-echo seam)",
                        _dk,
                        (_dk_entry or {}).get("subject", ""),
                        (_dk_entry or {}).get("predicate", ""),
                        (_dk_entry or {}).get("object", ""),
                    )
                for _dk in drift_orphan:
                    logger.info(
                        "graph_drift_key key=%s bucket=orphan"
                        " (no subject/predicate/object content; correctly dropped)",
                        _dk,
                    )
                for _dk in drift_genuine_loss:
                    _dk_entry = self.store.get(_dk)
                    logger.info(
                        "graph_drift_key key=%s bucket=genuine_loss"
                        " subject=%r predicate=%r object=%r"
                        " (reconstruction failure — retrained with"
                        " registry-true content; not a data loss)",
                        _dk,
                        (_dk_entry or {}).get("subject", ""),
                        (_dk_entry or {}).get("predicate", ""),
                        (_dk_entry or {}).get("object", ""),
                    )
                for _dk in drift_intended_removal:
                    logger.info(
                        "graph_drift_key key=%s bucket=intended_removal reason=%s"
                        " (merger-recorded intentional removal — key retained, not staled)",
                        _dk,
                        (_ledger.get(_dk) or {}).get("reason", ""),
                    )

                if drift_deduplicated_count:
                    logger.info(
                        "_run_fold[main_tiers]: %d key(s) deduplicated (registry-true"
                        " duplicate; soft-staled — record retained, excluded from training)",
                        drift_deduplicated_count,
                    )
                if drift_orphan_count:
                    logger.info(
                        "_run_fold[main_tiers]: %d orphan key(s) dropped (no SPO content)",
                        drift_orphan_count,
                    )
                if drift_intended_removal_count:
                    logger.info(
                        "_run_fold[main_tiers]: %d key(s) in intended_removal"
                        " (merger-recorded removal: by_reason=%s)",
                        drift_intended_removal_count,
                        drift_intended_removal_by_reason,
                    )

                if drift_genuine_loss_count > 0:
                    logger.warning(
                        "_run_fold[main_tiers]: %d genuine reconstruction loss(es) — "
                        "these keys had content but produced no merged edge (reconstruction"
                        " failure or hydration-miss); they were retrained with registry-true"
                        " content (should trend to ~0): %s",
                        drift_genuine_loss_count,
                        drift_genuine_loss,
                    )

                logger.info(
                    "_run_fold[main_tiers]: key distribution — episodic=%d semantic=%d "
                    "procedural=%d drift=%d (deduplicated=%d orphan=%d genuine_loss=%d"
                    " intended_removal=%d)",
                    len(tier_keyed["episodic"]),
                    len(tier_keyed["semantic"]),
                    len(tier_keyed["procedural"]),
                    graph_drift_count,
                    drift_deduplicated_count,
                    drift_orphan_count,
                    drift_genuine_loss_count,
                    drift_intended_removal_count,
                )

                on_removal_ledger(getattr(self.merger, "removal_ledger", {}))

            tiers_rebuilt: list[str] = []
            last_per_key_by_tier: dict[str, "list | None"] = {}

            if scope.source != "weights":
                # Disk venue: no adapter weights exist, so there is nothing to
                # back up or retrain. A tier counts as rebuilt
                # when it carries keys to project — the same predicate the
                # weights venue applies before it trains a tier.
                tiers_rebuilt = [t for t in ("episodic", "semantic", "procedural") if tier_keyed[t]]
            else:
                # --- Build per-tier TrainingJob objects ---
                from paramem.server.background_trainer import TrainingJob

                refresh_training_config = self.training_config

                jobs_by_tier = {
                    "episodic": TrainingJob(
                        entries=tier_keyed["episodic"],
                        adapter_name="episodic",
                        adapter_config=self.episodic_config,
                        inference_fallback_adapter="episodic_backup",
                    ),
                    "semantic": TrainingJob(
                        entries=tier_keyed["semantic"],
                        adapter_name="semantic",
                        adapter_config=self.semantic_config,
                        inference_fallback_adapter="semantic_backup",
                    ),
                    "procedural": TrainingJob(
                        entries=tier_keyed["procedural"],
                        adapter_name="procedural",
                        adapter_config=self.procedural_config or self.episodic_config,
                        inference_fallback_adapter="procedural_backup",
                    ),
                }

                # --- Per-tier warm-default / RECONCILE-cold rebuild ---
                tier_config_for_backup = {
                    "episodic": self.episodic_config,
                    "semantic": self.semantic_config,
                    "procedural": self.procedural_config or self.episodic_config,
                }

                # --- Pre-backup config reconciliation (resident tiers only) ---
                # main_tier_backup_scope (entered below) snapshots each resident
                # tier via copy_adapter_weights(src=tier, dst=backup), which checks
                # PARAMETER KEY SETS, not tensor shapes (loader.py's
                # copy_adapter_weights) -- a rank change keeps the same key names
                # but different tensor shapes, so it passes that check and then
                # raises a shape-mismatch RuntimeError inside the per-tensor
                # ``.data.copy_()``, before the per-tier loop's own cold_init /
                # ensure_adapter_matching branch is ever reached. Reconciling here,
                # ahead of the backup scope, means a resident tier is already
                # config-matching by the time it is snapshotted -- restricted to
                # RESIDENT tiers (``if _t in self.model.peft_config``) so a
                # disabled/not-yet-created tier is not born early.
                for _t, _cfg in tier_config_for_backup.items():
                    if _t in self.model.peft_config:
                        self.model = ensure_adapter_matching(self.model, _cfg, _t)

                # --- Backup-creation window telemetry (adapter-attributable) ---
                # The only measurement in this module that is attributable to
                # adapter VRAM cost — main_tier_backup_scope creates up to three
                # transient <tier>_backup adapters. free_before is sampled
                # immediately before the CM opens; free_after is the first
                # statement inside its body, after backup creation.
                _telemetry_free_before = (
                    torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else None
                )
                with main_tier_backup_scope(self.model, tier_config_for_backup) as _bscope:
                    self.model = _bscope.model
                    if torch.cuda.is_available() and _telemetry_free_before is not None:
                        _telemetry_free_after, _telemetry_total = torch.cuda.mem_get_info()
                        _telemetry_adapter_count = len(self.model.peft_config)
                        _telemetry_interim_count = len(
                            [a for a in self.model.peft_config if a.startswith(INTERIM_NAME_PREFIX)]
                        )
                        logger.info(
                            "_run_fold[main_tiers]: telemetry backup_creation — "
                            "free_before=%d free_after=%d adapter_count=%d interim_count=%d",
                            _telemetry_free_before,
                            _telemetry_free_after,
                            _telemetry_adapter_count,
                            _telemetry_interim_count,
                        )
                        if self._telemetry_dir is not None:
                            try:
                                record_fold_telemetry(
                                    self._telemetry_dir,
                                    cycle_stamp=_telemetry_run_stamp_c,
                                    kind="backup_creation",
                                    record={
                                        "fold_stamp": _fold_stamp_c,
                                        "free_before": _telemetry_free_before,
                                        "free_after": _telemetry_free_after,
                                        "total": _telemetry_total,
                                        "adapter_count": _telemetry_adapter_count,
                                        "interim_count": _telemetry_interim_count,
                                    },
                                )
                            except Exception:  # noqa: BLE001  # boundary: telemetry runs
                                # inside the CM's entered body — a write failure must
                                # never replace an in-flight exception (e.g. an
                                # AbortedDuringConsolidation raised later in the tier
                                # loop). Losing a telemetry record is strictly
                                # preferable to a misrouted abort.
                                logger.warning(
                                    "_run_fold[main_tiers]: telemetry write failed"
                                    " for backup_creation",
                                    exc_info=True,
                                )

                    # Completed-tier set from resume marker (empty on fresh fold).
                    _completed_in_marker: set[str] = (
                        set(_resume_marker.get("completed_tiers", []))  # type: ignore[union-attr]
                        if _resume_c
                        else set()
                    )
                    _marker_checkpoints: dict[str, str] = (
                        _resume_marker.get("tier_checkpoints", {})  # type: ignore[union-attr]
                        if _resume_c
                        else {}
                    )

                    for tier in ("episodic", "semantic", "procedural"):
                        backup_name = f"{tier}_backup"
                        job = jobs_by_tier[tier]

                        if not job.entries:
                            logger.info(
                                "_run_fold[main_tiers]: no keys for tier %s — skipping rebuild",
                                tier,
                            )
                            continue

                        # --- Crash-resume: reload completed tiers from durable checkpoint ---
                        if _resume_c and tier in _completed_in_marker:
                            # The checkpoint path stored in the marker (may be absent
                            # when _latest_checkpoint_in_dir found no checkpoint-N dir
                            # for this tier).
                            _ckpt_path = _marker_checkpoints.get(tier)
                            logger.info(
                                "_run_fold[main_tiers]: CRASH-RESUME tier=%s — reloading from"
                                " durable checkpoint (no retrain); checkpoint=%s",
                                tier,
                                _ckpt_path or "production-slot",
                            )
                            # Delete the stale production slot (pre-crash _save_adapters never
                            # ran — weights are stale) and reload from the checkpoint dir or the
                            # existing production slot when no checkpoint was recorded.
                            # The per-tier backups created above mean the deleted
                            # slot is never the last adapter on the PeftModel
                            # (no base-unwrap needed).
                            if tier in self.model.peft_config:
                                if backup_name in self.model.peft_config:
                                    from paramem.models.loader import switch_adapter as _sw_pre

                                    _sw_pre(self.model, backup_name)
                                self.model.delete_adapter(tier)
                                logger.debug(
                                    "_run_fold[main_tiers]: crash-resume deleted stale slot %s",
                                    tier,
                                )
                            if _ckpt_path and Path(_ckpt_path).is_dir():
                                # checkpoint-N dir present — load the staged adapter
                                # from it.  HF Trainer saves all PEFT adapters under
                                # checkpoint-N/<adapter_name>/ (one subdir per adapter).
                                # The training adapter staging slot is "in_training"
                                # (trainer._STAGING_ADAPTER), so the weights live at
                                # checkpoint-N/in_training/adapter_model.safetensors.
                                # Decrypt into /dev/shm when security is ON (mirrors
                                # trainer.py:962-976).
                                from paramem.backup import key_store as _ks
                                from paramem.training.trainer import (
                                    _STAGING_ADAPTER as _STAGING_SLOT,
                                )

                                # Resolve to the staging-adapter subdir within the checkpoint.
                                _ckpt_staging_path = Path(_ckpt_path) / _STAGING_SLOT
                                _ckpt_effective = (
                                    str(_ckpt_staging_path)
                                    if _ckpt_staging_path.is_dir()
                                    else _ckpt_path
                                )
                                _ckpt_shm_dir = None
                                if _ks.daily_identity_loadable(_ks.DAILY_KEY_PATH_DEFAULT):
                                    from paramem.backup.checkpoint_shard import (
                                        materialize_checkpoint_to_shm,
                                    )

                                    _ckpt_shm_dir = materialize_checkpoint_to_shm(
                                        Path(_ckpt_effective)
                                    )
                                    _ckpt_load_path = str(_ckpt_shm_dir)
                                else:
                                    _ckpt_load_path = _ckpt_effective
                                try:
                                    self.model.load_adapter(_ckpt_load_path, adapter_name=tier)
                                    logger.info(
                                        "_run_fold[main_tiers]: crash-resume loaded %s from"
                                        " checkpoint %s (staging slot=%s)",
                                        tier,
                                        _ckpt_path,
                                        _STAGING_SLOT,
                                    )
                                finally:
                                    if (
                                        _ckpt_shm_dir is not None
                                        and Path(str(_ckpt_shm_dir)).exists()
                                    ):
                                        import shutil as _s

                                        _s.rmtree(_ckpt_shm_dir, ignore_errors=True)
                            else:
                                # no checkpoint-N dir recorded for this tier (see
                                # _latest_checkpoint_in_dir). Reload from the EXISTING
                                # production slot on disk — it was not overwritten
                                # (final _save_adapters never ran on crash).
                                from paramem.memory.interim_adapter import (
                                    adapter_slot_root_for_name as _asr_fn,
                                )
                                from paramem.models.loader import load_adapter as _la

                                _prod_root = _asr_fn(self.output_dir, tier)
                                _la(self.model, _prod_root.parent, tier)
                                logger.info(
                                    "_run_fold[main_tiers]: crash-resume (no recorded checkpoint)"
                                    " loaded %s from production slot %s",
                                    tier,
                                    _prod_root.parent,
                                )
                            from paramem.models.loader import switch_adapter as _sw_resume

                            _sw_resume(self.model, tier)
                            last_per_key_by_tier[tier] = None
                            tiers_rebuilt.append(tier)
                            continue

                        tier_cfg = (
                            self.episodic_config
                            if tier == "episodic"
                            else (
                                self.semantic_config
                                if tier == "semantic"
                                else (self.procedural_config or self.episodic_config)
                            )
                        )

                        if backup_name in self.model.peft_config:
                            from paramem.models.loader import switch_adapter as _sw_backup

                            _sw_backup(self.model, backup_name)

                        if scope.cold_init:
                            # RECONCILE only (FoldScope.cold_init) — reproduce
                            # today's unconditional cold rebuild exactly.
                            if tier in self.model.peft_config:
                                self.model.delete_adapter(tier)
                                logger.debug(
                                    "_run_fold[main_tiers]: deleted adapter %s"
                                    " (cold_init: RECONCILE)",
                                    tier,
                                )
                            self.model = create_adapter(self.model, tier_cfg, tier)
                            logger.debug(
                                "_run_fold[main_tiers]: created fresh adapter %s"
                                " (cold_init: RECONCILE)",
                                tier,
                            )
                        else:
                            # Warm default: keep the resident tier's weights —
                            # the funnel's staging copy (trainer.py:944-948)
                            # warm-starts training from them. Recreates cold
                            # only on first-boot absence or a genuine LoRA
                            # config mismatch (never as blanket policy).
                            self.model = ensure_adapter_matching(self.model, tier_cfg, tier)

                        from paramem.models.loader import switch_adapter as _sw

                        _sw(self.model, tier)

                        prior_job = None
                        recall_state = None
                        _tier_metrics = None
                        if trainer is not None:
                            prior_job = trainer._current_job
                            trainer._current_job = job
                            trainer._set_is_training(True)
                        # --- Per-tier device-saturation telemetry ---
                        # max_memory_allocated() is a process-wide PyTorch-allocator
                        # counter, polluted by the per-epoch recall probe's
                        # model.generate() and by inference served during
                        # BackgroundTrainer step-yields — NOT an adapter cost.
                        # Bare snapshots only (never vram_measure: that captures
                        # endpoint free-deltas, not the intra-training peak this
                        # needs, and its OOM->VramExhausted transform is beside the
                        # point here since abort/rollback is gated on
                        # _tier_metrics.get("aborted"), a normal return value).
                        _telemetry_tier_free_before: int | None = None
                        _telemetry_tier_total: int | None = None
                        _telemetry_tier_n_keys = len(job.entries)
                        # Derived here (not read off refresh_training_config.num_epochs)
                        # so the finally-path telemetry below records the TRUE
                        # budget even when training raises. _train_tier_adapter
                        # derives the identical value from the same n_keys input --
                        # budget_for is pure.
                        _telemetry_tier_epochs, _telemetry_tier_accum, _ = budget_for(
                            _telemetry_tier_n_keys
                        )
                        # Measured BEFORE training starts -- same enclosing-scope
                        # hoist as the budget derivation above, so the
                        # finally-path record below carries the true
                        # pre-training weight state even when training raises.
                        _telemetry_tier_init = measured_adapter_init_state(self.model, tier)
                        _telemetry_tier_stale = len(self.store.stale_keys_in_tier(tier))
                        if torch.cuda.is_available():
                            torch.cuda.reset_peak_memory_stats()
                            _telemetry_tier_free_before, _telemetry_tier_total = (
                                torch.cuda.mem_get_info()
                            )
                        try:
                            _tier_metrics, recall_state = self._train_tier_adapter(
                                job.entries,
                                adapter_name=tier,
                                adapter_config=tier_cfg,
                                training_config=refresh_training_config,
                                output_dir=self.output_dir / "consolidation_refresh" / tier,
                                run_name=f"consolidate-{tier}",
                                phase_name=f"consolidate-{tier}",
                                retain_scratch_until_external_commit=True,
                            )
                            if _tier_metrics is not None:
                                if _tier_metrics.get("aborted"):
                                    logger.info(
                                        "_run_fold[main_tiers]: training aborted on tier %s "
                                        "— restoring all tiers from backups",
                                        tier,
                                    )
                                    raise AbortedDuringConsolidation(
                                        f"training aborted on tier {tier!r}"
                                    )
                                else:
                                    logger.info(
                                        "_run_fold[main_tiers]: trained %s on %d keys",
                                        tier,
                                        len(job.entries),
                                    )
                        finally:
                            if trainer is not None:
                                trainer._set_is_training(False)
                                trainer._current_job = prior_job
                            # The ENTIRE record build (not just the write below)
                            # is inside this try/except: constructing the dict
                            # reads self.model.peft_config and calls
                            # _recall_bind_telemetry, either of which could raise
                            # on a sufficiently broken state, and a raise here in
                            # a bare finally (unguarded) would REPLACE an
                            # in-flight exception from the try above (e.g.
                            # AbortedDuringConsolidation) with whatever this
                            # construction raised -- silently misrouting an abort
                            # to the crash-incident path via main_tier_backup_scope's
                            # except. Losing a telemetry record is strictly
                            # preferable to that.
                            try:
                                # aborted is a NORMAL RETURN VALUE from
                                # _train_tier_adapter (the trainer's own
                                # thermal-throttle/operator-pause signal) that
                                # this branch converts to a raised
                                # AbortedDuringConsolidation AFTER the
                                # assignment above succeeds -- so _tier_metrics
                                # is bound (with aborted=True) on that path, and
                                # stays at its pre-declared None only when
                                # _train_tier_adapter itself raised before
                                # returning.
                                _tier_aborted = bool(
                                    _tier_metrics.get("aborted")
                                    if _tier_metrics is not None
                                    else False
                                )
                                # Budget/bind/init/stale fields do not depend on
                                # CUDA introspection -- the record is always
                                # written; only the VRAM fields below are
                                # conditional on it.
                                _telemetry_tier_record: dict = {
                                    "tier": tier,
                                    "fold_stamp": _fold_stamp_c,
                                    "adapter_count": len(self.model.peft_config),
                                    "interim_count": len(
                                        [
                                            a
                                            for a in self.model.peft_config
                                            if a.startswith(INTERIM_NAME_PREFIX)
                                        ]
                                    ),
                                    "epochs": _telemetry_tier_epochs,
                                    "n_keys": _telemetry_tier_n_keys,
                                    "accum": _telemetry_tier_accum,
                                    "stale_keys": _telemetry_tier_stale,
                                    "aborted": _tier_aborted,
                                }
                                if _telemetry_tier_init is not None:
                                    _telemetry_tier_record["init"] = _telemetry_tier_init
                                # See the interim call site's identical comment:
                                # _train_tier_adapter tags its returned metrics
                                # dict on an actual donor-seeded copy; the
                                # exception path (where _tier_metrics stays None)
                                # leaves the pre-training measured value alone.
                                if _tier_metrics is not None and _tier_metrics.get("donor_seeded"):
                                    _telemetry_tier_record["init"] = "donor"
                                _tier_epochs_to_bind, _tier_steps_to_bind, _tier_hit_cap = (
                                    _recall_bind_telemetry(
                                        recall_state, _telemetry_tier_n_keys, _telemetry_tier_accum
                                    )
                                )
                                if _tier_epochs_to_bind is not None:
                                    _telemetry_tier_record["epochs_to_bind"] = _tier_epochs_to_bind
                                    _telemetry_tier_record["steps_to_bind"] = _tier_steps_to_bind
                                # hit_cap is suppressed on the abort path: stop_epoch
                                # is None whenever the trainer never reached (or
                                # never signalled) recall convergence, and an abort
                                # is exactly such a case -- emitting hit_cap=True
                                # there would be indistinguishable from a genuine
                                # "budget too small" outcome in the bucket re-fit.
                                if not _tier_aborted and _tier_hit_cap is not None:
                                    _telemetry_tier_record["hit_cap"] = _tier_hit_cap
                                if (
                                    torch.cuda.is_available()
                                    and _telemetry_tier_free_before is not None
                                ):
                                    _telemetry_tier_peak = torch.cuda.max_memory_allocated()
                                    # peak_reserved is the OOM-relevant quantity: the
                                    # caching allocator raises when it cannot reserve,
                                    # not when driver-free (mem_get_info) drops —
                                    # driver-free counts cached-but-unused allocator
                                    # segments as used, which peak_reserved does not.
                                    _telemetry_tier_peak_reserved = torch.cuda.max_memory_reserved()
                                    _telemetry_tier_free_after = torch.cuda.mem_get_info()[0]
                                    logger.info(
                                        "_run_fold[main_tiers]: telemetry tier_train[%s] "
                                        "(device-saturation indicator, not adapter cost) — "
                                        "free_before=%d free_after=%d peak_alloc=%d "
                                        "peak_reserved=%d",
                                        tier,
                                        _telemetry_tier_free_before,
                                        _telemetry_tier_free_after,
                                        _telemetry_tier_peak,
                                        _telemetry_tier_peak_reserved,
                                    )
                                    _telemetry_tier_record.update(
                                        {
                                            "free_before": _telemetry_tier_free_before,
                                            "free_after": _telemetry_tier_free_after,
                                            "peak_alloc": _telemetry_tier_peak,
                                            "peak_reserved": _telemetry_tier_peak_reserved,
                                            "total": _telemetry_tier_total,
                                        }
                                    )
                                if self._telemetry_dir is not None:
                                    record_fold_telemetry(
                                        self._telemetry_dir,
                                        cycle_stamp=_telemetry_run_stamp_c,
                                        kind="tier_train",
                                        record=_telemetry_tier_record,
                                    )
                            except Exception:  # noqa: BLE001  # boundary: this
                                # finally runs on the abort path too — e.g.
                                # AbortedDuringConsolidation is raised in the try
                                # above and would reach main_tier_backup_scope's
                                # except only if this finally does not itself
                                # raise. A failure anywhere in record
                                # construction OR the write (disk full,
                                # permissions, corrupt store, a broken
                                # model/store attribute) must never replace the
                                # in-flight exception and misroute an abort to
                                # the crash-incident path. Losing a telemetry
                                # record is strictly preferable.
                                logger.warning(
                                    "_run_fold[main_tiers]: telemetry write failed for tier %s",
                                    tier,
                                    exc_info=True,
                                )

                        last_per_key_by_tier[tier] = (
                            recall_state.last_per_key if recall_state is not None else None
                        )
                        if recall_state is not None and recall_state.last_per_key is not None:
                            on_recall_probe(
                                recall_state.last_per_key,
                                phase="train_fill",
                                adapter_name=tier,
                            )
                        tiers_rebuilt.append(tier)
                        # Mark this tier complete in the fold_resume.json marker so that a
                        # crash AFTER training but BEFORE _save_adapters can reload it without
                        # retraining on the next re-entry.  Locate the retained checkpoint-N dir
                        # (retain_scratch_until_external_commit=True keeps it alive until
                        # _save_adapters below).
                        _tier_ckpt_path = self._latest_checkpoint_in_dir(
                            self.output_dir / "consolidation_refresh" / tier
                        )
                        self._mark_tier_complete(tier, _tier_ckpt_path)

                    if trainer is not None:
                        trainer._set_is_training(False)

            # --- Atomic finalize ---
            # Interim disposal follows from the key source, not from a flag of
            # its own: a fold whose keys came from the main tiers alone never
            # read the interim slots, so it must leave both their registries and
            # their on-disk payload exactly where they are.
            _absorbed_interims = scope.keys_from == "all_tiers"

            if self.store.replay_enabled:
                # Recall gating is a weight verdict.  The disk venue has no
                # weights to verify against, so it passes None for the whole
                # dict — the documented "skip recall gating" signal, distinct
                # from a per-tier None (which triggers the weight probe).
                passing_sets_by_tier: "dict[str, set[str] | None] | None" = None
                if scope.source == "weights":
                    passing_sets_by_tier = {}
                    for _tier in ("episodic", "semantic", "procedural"):
                        _lpk = last_per_key_by_tier.get(_tier)
                        if _lpk is not None:
                            _serve_keys = {e["key"] for e in tier_keyed[_tier]}
                            passing_sets_by_tier[_tier] = {
                                r["key"] for r in _lpk if r["exact_match"]
                            } & _serve_keys
                        else:
                            passing_sets_by_tier[_tier] = None

                self._reset_main_tier_registries_and_simhashes(
                    tier_keyed,
                    passing_sets_by_tier,
                    soft_stale_by_tier=soft_stale_by_tier,
                )
                if _absorbed_interims:
                    self._drop_interim_tier_registries()
                for _reg_tier in ("episodic", "semantic", "procedural"):
                    _reg_tier_dir = self.output_dir / _reg_tier
                    _reg_tier_dir.mkdir(parents=True, exist_ok=True)
                    _reg_path = _reg_tier_dir / "indexed_key_registry.json"
                    self.store.registry(_reg_tier).save(_reg_path)
                    logger.info(
                        "_run_fold[main_tiers]: registry rewritten to %s",
                        _reg_path,
                    )

            # ONE predicate for persist AND reap.  A fold that wrote nothing must
            # not destroy what it read: the interim slots are the only copy of
            # their content until the merged main tiers are on disk (in the disk
            # venue the slot's graph.json IS the payload; in the weights venue it
            # is the slot adapter).  Reaping them after a no-persist fold is data
            # loss by construction, so the two guards are the same expression,
            # bound once so they cannot drift apart.
            _fold_persisted = self.store.replay_enabled and bool(tiers_rebuilt)

            if _fold_persisted:
                self._persist_fold(scope)
                logger.info("_run_fold[main_tiers]: merged main tiers persisted")
                # Clean fold-resume marker + retained scratch checkpoints after
                # the persist succeeds.  On persist FAILURE (the except
                # above re-raises) the marker is intentionally LEFT so a retry can
                # resume completed tiers without retraining.
                self._clear_fold_resume()
                import shutil as _sh_fold

                _refresh_root = self.output_dir / "consolidation_refresh"
                if _refresh_root.exists():
                    _sh_fold.rmtree(_refresh_root, ignore_errors=True)
                    logger.debug(
                        "_run_fold[main_tiers]: cleaned consolidation_refresh scratch"
                        " after _save_adapters"
                    )

            if self.store.replay_enabled and soft_stale_by_tier:
                for _st_tier in ("episodic", "semantic", "procedural"):
                    self.store.registry(_st_tier).increment_stale_cycles()
                logger.debug(
                    "_run_fold[main_tiers]: stale_cycles advanced for %d soft-staled key(s)",
                    sum(len(v) for v in soft_stale_by_tier.values()),
                )

            if not _absorbed_interims:
                logger.info(
                    "_run_fold[main_tiers]: rebuilt from the main tiers' own keys"
                    " — interim slots untouched (not folded in, so not reaped)"
                )
            elif _fold_persisted:
                unload_interim_adapters(self.model, self.output_dir)
                logger.info("_run_fold[main_tiers]: interim slots reaped")
            else:
                logger.info(
                    "_run_fold[main_tiers]: nothing persisted — interim slots kept"
                    " (their content is still the only copy)"
                )

            if router is not None:
                try:
                    router.reload()
                    logger.info("_run_fold[main_tiers]: router reloaded")
                except Exception:
                    logger.exception("_run_fold[main_tiers]: router reload failed")

            if scope.source == "weights" and "episodic" in self.model.peft_config:
                from paramem.models.loader import switch_adapter as _sw2

                _sw2(self.model, "episodic")

            _train_tiers = ("episodic", "semantic", "procedural")
            _train_tier_delta = self._build_tier_delta(
                active_before=_train_active_before,
                active_after={t: len(tier_keyed.get(t, [])) for t in _train_tiers},
                minted_by_tier=minted_by_tier,
            )
            on_tier_delta(_train_tier_delta)

            logger.info(
                "_run_fold[main_tiers]: complete — rebuilt %s, drift=%d"
                " (deduplicated=%d orphan=%d genuine_loss=%d intended_removal=%d)",
                tiers_rebuilt,
                graph_drift_count,
                drift_deduplicated_count,
                drift_orphan_count,
                drift_genuine_loss_count,
                drift_intended_removal_count,
            )

            return {
                "tiers_rebuilt": tiers_rebuilt,
                "graph_drift_count": graph_drift_count,
                "drift_deduplicated": drift_deduplicated_count,
                "drift_orphan": drift_orphan_count,
                "drift_genuine_loss": drift_genuine_loss_count,
                "drift_intended_removal": drift_intended_removal_count,
                "drift_intended_removal_by_reason": drift_intended_removal_by_reason,
                "recall_miss_keys": sorted(recall_miss_keys),
                "keys_per_tier": {t: len(v) for t, v in tier_keyed.items()},
                "tier_keyed": tier_keyed,
                "rolled_back": False,
                "rollback_tier": None,
                "tier_delta": _train_tier_delta,
            }
        finally:
            self._current_interim_stamp = None  # type: ignore[assignment]
            self.merger.reset_graph()

    def consolidate(
        self,
        *,
        mode: str,
        keys_from: "Literal['all_tiers', 'main_tiers']" = "all_tiers",
        consume_pending: bool = False,
        trainer=None,
        router=None,
    ) -> dict:
        """Run the full consolidation fold — the single public fold entry.

        The fold does what it is told.  Whether there is anything to consolidate at
        all is decided by the caller (the server's dispatch layer): this method has
        no content gate, no notion of who asked for the fold, and no way to bypass
        the recall gate or the caller-side content gate.

        Both venues run the SAME stage spine over the SAME input — the
        :class:`~paramem.memory.store.MemoryStore`, whose main-tier and
        interim-slot registries are hydrated at boot and after every cycle.
        Materialize → refine → promote → build entries → drift
        partition → registry rewrite → persist → interim unload → router reload
        → tier delta is one code path.  *mode* selects only:

        - **train** (``source="weights"``): additionally probes the adapters for
          recall misses, backs the main tiers up, retrains
          ``episodic`` / ``semantic`` / ``procedural``, and persists + verifies
          the weights.  Requires the caller to already hold ``_gpu_thread_lock``
          (submit via ``BackgroundTrainer.submit()``); the entry guard below
          raises when it does not.  On a failed per-tier recall-sanity check the
          tier is restored from its backup slot and the fold aborts.
        - **simulate** (``source="disk"``): skips those weight-only blocks and
          persists each main tier as ``<adapter_dir>/<tier>/graph.json``, the
          path :class:`~paramem.memory.source.DiskMemorySource` reads back.  No
          model, no GPU.

        Both venues route through :meth:`_run_fold`; the ``mode`` string is translated
        into a :class:`FoldScope` here and never travels further (the mode-fork guard
        requires downstream dispatch on ``scope.source`` / ``scope.persist``).

        Args:
            mode: ``"train"`` or ``"simulate"``.  Required — ``ConsolidationConfig``
                carries no ``mode`` field; the server passes
                ``config.consolidation.mode``.
            keys_from: The fold's key source (see :class:`FoldScope`).
                ``"all_tiers"`` (the default) folds the interim slots into main
                and reaps them; ``"main_tiers"`` rebuilds main memory from its
                own keys and leaves every interim slot on disk.  It is a filter
                on one fold, not a second ingest path: both values run the same
                spine over the same store.
            consume_pending: When ``True`` (train only), the fold snapshots the
                pending-session relations already deposited in ``merger.graph`` by the
                caller's extraction pre-stage and trains them into the main tiers.  The
                caller derives this from its schedule config
                (``max_interim_count == 0 and mode != "simulate"``).
            trainer: :class:`~paramem.server.background_trainer.BackgroundTrainer`
                holding the GPU lock (train only).  Required for the per-tier re-arm
                pattern.
            router: Router instance whose ``reload()`` is called at the end of the
                atomic finalize sequence (both venues).  ``None`` is safe — skipped.

        Returns:
            The full-fold result dict (see :meth:`_run_fold`) — one schema for both
            venues and every terminal return.

        Raises:
            ValueError: When ``consume_pending`` is requested on the simulate venue.
                The simulate fold has no weight venue to train pending sessions into,
                so it would discard the flag; callers derive ``consume_pending`` from
                ``max_interim_count == 0 and mode != "simulate"``, which cannot produce
                that pairing today.  The guard exists so a future caller that gets the
                derivation wrong fails loudly instead of silently ingesting nothing.
            RuntimeError: When ``mode="train"`` is called without the GPU lock held.
        """
        self._current_interim_stamp = None  # type: ignore[assignment]

        if mode == "simulate" and consume_pending:
            raise ValueError(
                "consolidate(mode='simulate') cannot consume pending sessions: the "
                "simulate venue writes graph.json and trains nothing. Pass "
                "consume_pending=False, or run the train venue."
            )

        if mode == "simulate":
            # Every artifact the fold and its nested passes emit lands in this
            # cycle's debug root; a calibration run, when one is open, adds its
            # own root independently.
            # promote is ON: it is a pure store operation, so it belongs to
            # this venue exactly as much as to the weights venue.
            with self._artifact_scope():
                return self._run_fold(
                    FoldScope(
                        name="full",
                        source="disk",
                        persist="main_tiers",
                        tier=None,
                        defer=False,
                        tag_new=False,
                        normalize=(self.config.refinement_normalization == "on"),
                        enrich=(self.config.refinement_enrichment == "on" and self.cloud_enabled),
                        promote=True,
                        subtractive_scope="fold",
                        keys_from=keys_from,
                    ),
                    router=router,
                )

        from paramem.server.gpu_lock import _gpu_thread_lock

        # --- Entry guard: verify the GPU lock is held by the caller (leak-safe) ---
        acquired = _gpu_thread_lock.acquire(blocking=False)
        if acquired:
            # The lock was NOT held — we just accidentally acquired it ourselves.
            # Release immediately before raising so the process is recoverable.
            _gpu_thread_lock.release()
            raise RuntimeError(
                "consolidate(mode='train') requires the caller to hold "
                "_gpu_thread_lock (submit via BackgroundTrainer.submit())"
            )

        # Every artifact the fold and its nested passes emit lands in this
        # cycle's debug root; a calibration run, when one is open, adds its own
        # root independently.
        with self._artifact_scope():
            return self._run_fold(
                FoldScope(
                    name="full",
                    source="weights",
                    persist="main_tiers",
                    tier=None,
                    defer=False,
                    tag_new=False,
                    normalize=(self.config.refinement_normalization == "on"),
                    enrich=(self.config.refinement_enrichment == "on" and self.cloud_enabled),
                    promote=True,
                    subtractive_scope="fold",
                    consume_pending=consume_pending,
                    keys_from=keys_from,
                ),
                trainer=trainer,
                router=router,
            )

    def _promote_mature_keys_inline(self) -> list[str]:
        """Promote episodic keys whose reinforcement_count has reached the promotion threshold.

        Mirrors the logic of the removed ``server.consolidation._promote_mature_keys``
        helper but runs INSIDE the fold spine, AFTER the
        recurrence-bump step and BEFORE ``tier_keyed`` is built.  This ordering
        guarantees that reconstruction probes each key against the adapter tier
        where its weights actually live (episodic) rather than against the
        semantic adapter that has not yet learned the key — the root cause of
        silent post-promotion fact loss.

        Reads thresholds from ``self.config`` (``ConsolidationConfig``), which
        is set at construction time.  Does NOT import ``ServerConfig`` — this
        module must remain server-independent.

        Steps:
        1. Iterate ``self.store.all_active_keys()``.
        2. Skip keys already in ``self.promoted_keys`` (already promoted or
           already in the ``has_simhash("semantic")`` branch from a prior fold).
        3. Promote keys whose ``reinforcement_count`` >= ``self.config.promotion_threshold``
           by calling ``self.store.move(key, "semantic")`` then
           ``self.promoted_keys.add(key)``.
        4. Log decay candidates (keys whose ``last_reinforced_cycle`` is more than
           ``self.config.decay_window`` cycles old) without deleting them
           (passive-fade policy — no fact loss).

        Returns:
            List of newly promoted key IDs (keys moved from episodic to semantic
            in this call; does NOT include previously promoted keys).
        """
        threshold = self.config.promotion_threshold
        decay_window = self.config.decay_window
        current_cycle = self.cycle_count
        newly_promoted: list[str] = []

        for key in self.store.all_active_keys():
            bk = self.store.bookkeeping_for_key(key) or {}
            rec = bk.get("reinforcement_count", 1)
            last = bk.get("last_reinforced_cycle", 0)

            if key in self.promoted_keys:
                continue

            if rec >= threshold:
                if self.store.has_simhash("episodic", key):
                    # Move entry + simhash + registry entry atomically to semantic.
                    self.store.move(key, "semantic")
                    newly_promoted.append(key)
                    logger.info(
                        "_promote_mature_keys_inline: key=%s promoted to semantic "
                        "(reinforcement_count=%d >= threshold=%d)",
                        key,
                        rec,
                        threshold,
                    )
                elif self.store.has_simhash("semantic", key):
                    logger.debug(
                        "_promote_mature_keys_inline: key=%s already in semantic, marking promoted",
                        key,
                    )
                self.promoted_keys.add(key)
            elif decay_window > 0 and (current_cycle - last) >= decay_window:
                # Decay candidate: key has not been re-seen for decay_window cycles.
                # Passive fade — log only; no deletion (consistent with
                # no-active-delete policy).
                logger.info(
                    "_promote_mature_keys_inline: key=%s decay candidate "
                    "(last_reinforced_cycle=%d, current_cycle=%d, window=%d)",
                    key,
                    last,
                    current_cycle,
                    decay_window,
                )

        if newly_promoted:
            logger.info(
                "_promote_mature_keys_inline: promoted %d key(s) to semantic",
                len(newly_promoted),
            )

        return newly_promoted

    def _build_all_edge_entries_into(
        self,
        tier_keyed: "dict[str, list[dict]]",
        *,
        defer: bool = False,
        tag_new: bool = False,
        exclude_keys: "set[str] | None" = None,
    ) -> "tuple[dict[str, int], list[dict]]":
        """Walk ALL merged-graph edges AND node attributes; populate *tier_keyed*.

        Single unified edge→entry builder that subsumes the former three-step
        sequence of ``_harvest_keyless_edge_entries`` →
        ``_apply_keyless_edge_entries`` → ``_collect_keyed_edges_into``.
        A second pass after the edge walk covers node ``attributes``:
        ``GraphMerger.merge`` diverts ``relation_type == "attribute"``
        relations (phone/email/date/certification/job title, ...) onto the
        SUBJECT node's ``attributes`` dict instead of an edge, so they are
        invisible to the edge walk and need their own projection back into
        ``tier_keyed`` — see the node-attribute walk below, which mirrors
        both edge-walk branches (tier derivation, store commit discipline,
        entry shape) exactly.

        **One pass, two branches per edge:**

        Keyless edges (no ``ik_key`` on the edge attribute, i.e. newly-extracted or
        Cloud-enrichment facts):
            - A key is minted via :meth:`_mint_keyed_entries` using a local running
              counter seeded from ``_indexed_next_index`` / ``_procedural_next_index``
              (the real counters are never touched until the write is committed).
            - ``speaker_id`` is resolved from the edge's ``speaker_id`` attribute
              first (stamped by the merger from ``Relation.speaker_id``), then falls
              back to the subject node's top-level ``speaker_id`` attribute.
              When neither is set the value is ``""`` (concept-rooted edge with no
              speaker attribution — allowed via ``allow_empty_speaker=True`` at the
              mint site).
            - When ``defer=False`` (fold discipline): ``store.put``,
              ``store.set_bookkeeping``, and counter advances are applied immediately.
            - When ``defer=True`` (interim atomicity): all store writes and counter
              advances are SKIPPED; the harvest record is added to ``deferred_writes``
              so the caller can flush after recall-confirmed training.
            - ``tag_new=True`` attaches ``entry["_new"] = True`` for callers that
              need to identify newly-minted entries in the result.

        Keyed edges (``ik_key`` present):
            - The training entry is sourced from ``store.get(key)`` (registry-true
              content); edges with no content entry are silently skipped.
            - ``speaker_id`` is sourced from bookkeeping (``bookkeeping_for_key``),
              which carries the original attribution — not from the edge attribute
              (which may reflect merge-time provenance rather than extraction-time
              provenance).
            - No ``store.put`` / ``store.set_bookkeeping`` / counter advances (key
              already registered; these are anti-forgetting replay entries).
            - ``_new`` is never set on existing keyed entries.

        Both branches append to ``tier_keyed`` with the **identical shape**:
        ``{key, subject, predicate, object, speaker_id}``.  The deferred-write
        record additionally carries ``session_ids`` (real contributing session ids,
        synthetic fold sentinels excluded).

        The ``ik_key`` attribute is intentionally NOT stamped onto keyless edges
        (direct-append variant) to avoid the MultiDiGraph parallel-edge integer-key
        hazard.  Both the keyless and keyed branch guard their pass via
        ``if not key`` / ``if key`` rather than edge mutation, so the same edge
        object is safe to iterate once.

        Args:
            tier_keyed: Mutable mapping of tier name → list of training-entry dicts.
                Both branches append in-place.
            defer: When ``True`` (interim path), all store writes and counter
                advances for NEW (keyless/minted) entries are deferred and returned
                in ``deferred_writes``.  Existing keyed entries are never written
                regardless of this flag.  Default ``False`` (fold discipline).
            tag_new: When ``True``, each minted entry receives ``entry["_new"] =
                True`` so the caller can identify newly-minted entries in
                ``tier_keyed``.  Default ``False``.
            exclude_keys: Optional set of ``ik_key`` strings to skip entirely
                during the edge walk — neither minted (N/A; these edges always
                already carry a key) nor keyed-replayed into ``tier_keyed``.
                Used by the (unconditional) interim recital-dedup feature to
                exclude main-tier facts that :meth:`_materialize_consolidation_graph`
                merged in as
                ``dedup_target_keys``: those facts participate in the merge's
                Case-1 identity (so a recited pending fact collapses onto
                them) but must never acquire interim-adapter weight residence
                (the main-tier/interim separation invariant) or be retrained
                wholesale into every interim slot.  Default ``None`` — today's behaviour,
                unaffected for every other caller.

        Returns:
            A 2-tuple ``(minted_by_tier, deferred_writes)`` where:

            - ``minted_by_tier`` — per-tier count of newly minted keys,
              e.g. ``{"episodic": 2, "procedural": 1}``.  Existing keyed entries
              do NOT contribute to this count.
            - ``deferred_writes`` — harvest records for new entries whose store
              writes have not yet been applied.  When ``defer=False`` this is
              always ``[]``; when ``defer=True`` this is one record per minted key.
              Each record has: ``"entry"``, ``"tier"``, ``"canon_subj"``,
              ``"canon_obj"``, ``"predicate"``, ``"relation_type"``, ``"speaker_id"``,
              ``"session_ids"`` (sorted list of real contributing session ids,
              synthetic fold sentinels excluded), ``"last_seen"`` (ISO 8601
              wall-clock from the merged edge; ``""`` when unavailable).

            Mutates *tier_keyed* in-place.  When ``defer=False``, also mutates
            the :class:`~paramem.memory.store.MemoryStore` and advances
            ``_indexed_next_index`` / ``_procedural_next_index`` for each minted key.
        """
        from paramem.memory.persistence import _EDGE_SOURCE_ATTR
        from paramem.memory.persistence import _IK_KEY_ATTR as _IK_ATTR

        minted_by_tier: dict[str, int] = {"episodic": 0, "procedural": 0}
        deferred_writes: list[dict] = []

        # Local running counters for key minting — never mutate the real self.*
        # counters inside the walk; they are advanced only at the commit site
        # (immediately for defer=False; by the caller for defer=True).
        # Seeded lazily on first use per tier so the real counters are not read
        # when no keyless edges of that tier are present.
        _local_indexed: int | None = None
        _local_procedural: int | None = None

        # (subject, predicate) pairs emitted by the edge walk below — read by
        # the node-attribute walk after it to skip a node attribute whose
        # pair was already emitted as an edge (defensive dedup for a mixed
        # graph, e.g. a pre-Unit-4 fold artifact still carrying an edge for
        # what a fresh extraction would now route to node["attributes"]).
        _emitted_pairs: set[tuple[str, str]] = set()

        def _commit_keyless_mint(
            *,
            subject_display: str,
            predicate: str,
            object_value: str,
            relation_type: str,
            speaker_id: str,
            canon_subj: str,
            canon_obj: str,
            session_ids: list[str],
            last_seen: str,
            first_seen: str,
        ) -> dict:
            """Mint one keyless fact, commit-or-defer it, and append to *tier_keyed*.

            The shared commit sequence behind BOTH keyless branches (edge
            walk and node-attribute walk): derive tier, mint via
            :meth:`_mint_keyed_entries` against the shared local running
            counter, persist immediately (``defer=False``) or record a
            deferred-write ``rec`` (``defer=True``), then append the
            uniform ``tier_keyed`` shape. Closes over this call's
            ``tier_keyed``/``minted_by_tier``/``deferred_writes``/``defer``/
            ``tag_new`` and the ``_local_indexed``/``_local_procedural``
            running counters (mutated via ``nonlocal`` — the two branches
            share ONE counter sequence, so it cannot be a plain parameter).

            The only differences between the two call sites are how they
            derive these arguments: an edge has two endpoints and an
            edge-carried speaker_id/session/timestamp trail; a node
            attribute has one endpoint (the subject) and none of that
            edge-carried provenance (``session_ids=[]``,
            ``last_seen=first_seen=""``, ``canon_obj=""``).

            Returns:
                The dict appended to ``tier_keyed[tier]``
                (``{key, subject, predicate, object, speaker_id}``).
            """
            nonlocal _local_indexed, _local_procedural

            _dummy = [
                {
                    "subject": subject_display,
                    "predicate": predicate,
                    "object": object_value,
                    "relation_type": relation_type,
                }
            ]
            _ep_rels, _proc_rels = partition_relations(
                _dummy, procedural_enabled=self.procedural_config is not None
            )
            tier = "procedural" if _proc_rels else "episodic"

            # Mint via the shared helper (single-element list).
            # Use LOCAL running counter as start_index; advance after each mint.
            prefix = "proc" if tier == "procedural" else "graph"
            if tier == "procedural":
                if _local_procedural is None:
                    _local_procedural = self._procedural_next_index
                start_index = _local_procedural
            else:
                if _local_indexed is None:
                    _local_indexed = self._indexed_next_index
                start_index = _local_indexed

            minted = self._mint_keyed_entries(
                [
                    {
                        "subject": subject_display,
                        "predicate": predicate,
                        "object": object_value,
                        "relation_type": relation_type,
                        "speaker_id": speaker_id,
                    }
                ],
                prefix=prefix,
                start_index=start_index,
                speaker_id=speaker_id,
                tag_new=tag_new,
            )

            # Advance the local counter for the chosen tier.
            if tier == "procedural":
                _local_procedural += 1
            else:
                _local_indexed += 1

            entry = minted[0]
            minted_key = entry["key"]
            # This "rec" shape (minus canon_subj/canon_obj, which the interim
            # commit window never reads) is the round-trip contract with
            # module-level _persisted_from_entry_and_rec (serialize into
            # fold_resume.json) / _rec_from_persisted (deserialize on
            # crash-resume) — see those functions' docstrings.
            rec = {
                "entry": entry,
                "tier": tier,
                "canon_subj": canon_subj,
                "canon_obj": canon_obj,
                "predicate": predicate,
                "relation_type": relation_type,
                "speaker_id": speaker_id,
                "session_ids": session_ids,
                "last_seen": last_seen,
                "first_seen": first_seen,
            }

            if not defer:
                # Fold discipline: persist immediately.
                self.store.put(
                    tier,
                    minted_key,
                    entry,
                    simhash=entry_simhash(entry),
                )
                self.store.set_bookkeeping(
                    minted_key,
                    speaker_id=speaker_id,
                    relation_type=relation_type,
                    reinforcement_count=1,
                    last_reinforced_cycle=self.cycle_count,
                    last_seen=last_seen,
                    first_seen=first_seen,
                    allow_empty_speaker=(speaker_id == ""),
                )
                # Advance the committed counter for the chosen tier.
                if tier == "procedural":
                    self._procedural_next_index += 1
                else:
                    self._indexed_next_index += 1
            else:
                # Interim atomicity: defer all store writes + counter advances.
                deferred_writes.append(rec)

            # Append to tier_keyed (uniform shape, same as the keyed branch).
            result_entry = {
                "key": minted_key,
                "subject": entry["subject"],
                "predicate": predicate,
                "object": entry["object"],
                "speaker_id": speaker_id,
            }
            tier_keyed[tier].append(result_entry)
            minted_by_tier[tier] += 1
            return result_entry

        for _t_subj, _t_obj, _t_data in self.merger.graph.edges(data=True):
            key = _t_data.get(_IK_ATTR)
            pred = _t_data.get("predicate", "")
            if not pred:
                # Edges with no predicate are not keyable — skip unconditionally.
                continue

            if key and exclude_keys and key in exclude_keys:
                # Interim recital-dedup target (main-tier fact merged in by
                # _materialize_consolidation_graph's dedup_target_keys
                # channel) — skip unconditionally.  Neither minted (already
                # keyed) nor keyed-replayed into tier_keyed: excluding it here
                # is what keeps main-tier facts out of the interim adapter's
                # training set (the main-tier/interim separation invariant).
                continue

            if not key:
                # ---- Keyless branch: mint a new key ----
                # Read relation_type from the edge; clamp to valid schema values.
                _rt_raw = _t_data.get("relation_type", _FALLBACK_RTYPE)
                _rt: str = _rt_raw if _rt_raw in _VALID_RTYPES else _FALLBACK_RTYPE

                # Resolve endpoint surface from node attributes["name"].
                # For speaker subjects: _endpoint_str returns the node key (lowercase
                # speaker{N}); paramem.graph.merger._synth_speaker_entities emits
                # Entity(name=speaker_id) which refreshes attributes["name"] to the
                # lowercase speaker_id during GraphMerger.merge_relations.  So
                # _subj_display yields the lowercase speaker_id for speaker subjects.
                # For non-speaker subjects this yields the stored display name.
                _subj_display = (
                    self.merger.graph.nodes[_t_subj].get("attributes", {}).get("name") or _t_subj
                )
                _obj_display = (
                    self.merger.graph.nodes[_t_obj].get("attributes", {}).get("name") or _t_obj
                )
                # C-1: resolve speaker_id from the edge first (stamped by the merger
                # A-1 from Relation.speaker_id), then fall back to the subject node's
                # top-level speaker_id attribute.  When both are empty, try the
                # unique-speaker-predecessor fallback (concept-rooted enrichment edges
                # whose subject is a role/project/org concept with exactly one speaker
                # pointing in).  Terminal fallback is "" (allow_empty path).
                _edge_sid = _t_data.get("speaker_id", None)
                if _edge_sid:
                    _subj_sid = _edge_sid
                else:
                    _node_attrs = self.merger.graph.nodes.get(_t_subj, {}) or {}
                    _subj_sid = _node_attrs.get("speaker_id", "") or ""
                    if not _subj_sid and _t_data.get(_EDGE_SOURCE_ATTR) == "graph_enrichment":
                        # FALLBACK-ONLY (enrichment edges only): subject node carries no
                        # speaker_id and this edge is a cloud-enrichment edge.  Inherit
                        # from the subject's UNIQUE non-empty speaker predecessor (1-hop,
                        # direct in-edges).  Exactly one distinct speaker → use it;
                        # zero or ≥2 → keep "".
                        # Extraction concept-edges (no edge_source / different value)
                        # keep the existing "" terminal — deliberate unattributed facts
                        # (e.g. company-location) must NOT be attributed to a speaker.
                        _subj_sid = self._unique_speaker_predecessor(_t_subj)

                # Source the contributing session ids from the merged edge,
                # excluding synthetic fold sentinels.  The result is a sorted
                # list of real session ids that contributed this fact.
                # This field is TRANSIENT — it rides the in-RAM record only;
                # it is never written to the persisted entry dict (store.put)
                # or to bookkeeping (store.set_bookkeeping).  The drop site
                # (step 11b) reads rec["session_ids"] to identify which
                # sessions contributed a recall-failed key.
                _rec_session_ids: list[str] = sorted(
                    set(_t_data.get("sessions", [])) - _SYNTHETIC_SESSION_IDS
                )
                # The ik_key attribute is intentionally NOT stamped onto the edge so
                # the MultiDiGraph parallel-edge integer key field is not disturbed —
                # _commit_keyless_mint never mutates the edge/node it was called for.
                _commit_keyless_mint(
                    subject_display=_subj_display,
                    predicate=pred,
                    object_value=_obj_display,
                    relation_type=_rt,
                    speaker_id=_subj_sid,
                    canon_subj=_t_subj,
                    canon_obj=_t_obj,
                    session_ids=_rec_session_ids,
                    # Real session wall-clock carried from the edge; sourced from
                    # session_graph.timestamp at ingest via merger._upsert_relation.
                    # Never fabricate now() here.
                    last_seen=_t_data.get("last_seen", ""),
                    first_seen=_t_data.get("first_seen", ""),
                )
                _emitted_pairs.add((_t_subj, pred))

            else:
                # ---- Keyed branch: existing key, anti-forgetting replay ----
                entry = self.store.get(key)
                if entry is None:
                    # Registered but content-free EVERYWHERE: the fold hydrated
                    # the store from the venue's source of truth before reaching
                    # here (_hydrate_store_for_fold), so this is not a cache
                    # artifact and the key genuinely has nothing to replay.
                    logger.debug(
                        "_build_all_edge_entries_into: key %s has no content entry — skipping",
                        key,
                    )
                    continue

                # Tier from per-key bookkeeping relation_type (not from the edge,
                # which may carry the merge-time value rather than the original type).
                _bk = self.store.bookkeeping_for_key(key) or {}
                _rt_raw = _bk.get("relation_type", _FALLBACK_RTYPE)
                _rt = _rt_raw if _rt_raw in _VALID_RTYPES else _FALLBACK_RTYPE
                # Speaker_id from bookkeeping — original extraction-time attribution.
                _subj_sid = _bk.get("speaker_id") or ""
                current_adapter_id = self.store.tier_for_active_key(key) or "episodic"
                _dummy = [
                    {
                        "subject": _t_subj,
                        "predicate": pred,
                        "object": _t_obj,
                        "relation_type": _rt,
                    }
                ]
                _ep_rels, _proc_rels = partition_relations(
                    _dummy, procedural_enabled=self.procedural_config is not None
                )
                if _proc_rels:
                    tier = "procedural"
                elif _ep_rels:
                    # Semantic keys remain semantic; all others map to episodic.
                    tier = "semantic" if current_adapter_id == "semantic" else "episodic"
                else:
                    tier = "episodic"

                # Uniform entry shape — identical to the keyless branch.
                tier_keyed[tier].append(
                    {
                        "key": key,
                        "subject": entry["subject"],
                        "predicate": entry["predicate"],
                        "object": entry["object"],
                        "speaker_id": _subj_sid,
                    }
                )
                # Existing keyed entries are never counted as minted and never
                # deferred — they are already in the store.
                _emitted_pairs.add((_t_subj, pred))

        # ---- Node-attribute walk: attribute-typed relations never become
        # edges (GraphMerger.merge diverts them onto the SUBJECT node's
        # "attributes" dict — see merger.py's relation_type == "attribute"
        # branch), so they are invisible to the edge walk above.  Mirrors
        # the two edge-walk branches above byte-for-byte: same tier
        # derivation via partition_relations, same store.put /
        # set_bookkeeping / deferred_writes commit discipline, same
        # tier_keyed entry shape.  The predicate rendered here goes through
        # relation_prep.attr_predicate — the ONE formula shared with
        # relation_prep._flatten_entity_attributes's projected predicate —
        # both are the sole surfaces an attribute fact can be trained under,
        # and they must share one SimHash fingerprint.  Routing through
        # canonical() here also corrects a node ``attributes`` key that
        # entered verbatim via GraphMerger._upsert_entity (the Entity.attributes
        # merge path, which does not canonicalize) rather than this module's
        # own attribute gate (which always writes canonical keys).
        for _n, _n_data in self.merger.graph.nodes(data=True):
            _n_attrs = _n_data.get("attributes", {}) or {}
            if not _n_attrs:
                continue
            _n_attr_keys = _n_data.get("attribute_keys", {}) or {}
            _n_subj_display = _n_attrs.get("name") or _n
            for attr_key, attr_value in _n_attrs.items():
                if attr_key == "name":
                    # Display surface, not a projected attribute fact.
                    continue
                attr_pred = attr_predicate(attr_key)
                if (_n, attr_pred) in _emitted_pairs:
                    # Defensive dedup: a mixed graph (e.g. a pre-Unit-4 fold
                    # artifact) already emitted this (subject, predicate)
                    # pair as an edge — never emit it twice.
                    continue

                attr_key_id = _n_attr_keys.get(attr_key)
                if attr_key_id:
                    # ---- Keyed branch: existing key, anti-forgetting replay ----
                    entry = self.store.get(attr_key_id)
                    if entry is None:
                        logger.debug(
                            "_build_all_edge_entries_into: attribute key %s "
                            "has no content entry — skipping",
                            attr_key_id,
                        )
                        continue
                    _bk = self.store.bookkeeping_for_key(attr_key_id) or {}
                    _rt_raw = _bk.get("relation_type", _FALLBACK_RTYPE)
                    _rt = _rt_raw if _rt_raw in _VALID_RTYPES else _FALLBACK_RTYPE
                    _subj_sid = _bk.get("speaker_id") or ""
                    current_adapter_id = self.store.tier_for_active_key(attr_key_id) or "episodic"
                    _dummy = [
                        {
                            "subject": _n,
                            "predicate": attr_pred,
                            "object": attr_value,
                            "relation_type": _rt,
                        }
                    ]
                    _ep_rels, _proc_rels = partition_relations(
                        _dummy, procedural_enabled=self.procedural_config is not None
                    )
                    if _proc_rels:
                        tier = "procedural"
                    elif _ep_rels:
                        tier = "semantic" if current_adapter_id == "semantic" else "episodic"
                    else:
                        tier = "episodic"

                    tier_keyed[tier].append(
                        {
                            "key": attr_key_id,
                            "subject": entry["subject"],
                            "predicate": entry["predicate"],
                            "object": entry["object"],
                            "speaker_id": _subj_sid,
                        }
                    )
                    # Existing keyed entries are never counted as minted and
                    # never deferred — they are already in the store.
                else:
                    # ---- Keyless branch: mint a new key ----
                    # attribute_keys is intentionally NOT stamped onto the
                    # node here — mirrors the edge branch's "no ik_key
                    # stamped on mint" discipline; the key is registered via
                    # GraphMerger's own gate the next time this fact reaches
                    # the merger with relation.indexed_key set (the fold's
                    # registry-true re-merge pass).
                    _rt = "attribute"
                    _subj_sid = _n_data.get("speaker_id", "") or ""
                    _commit_keyless_mint(
                        subject_display=_n_subj_display,
                        predicate=attr_pred,
                        object_value=attr_value,
                        relation_type=_rt,
                        speaker_id=_subj_sid,
                        canon_subj=_n,
                        canon_obj="",
                        # Attribute facts have no edge to source contributing
                        # session ids from — the node carries no per-fact
                        # session list.  Empty, not fabricated.
                        session_ids=[],
                        last_seen="",
                        first_seen="",
                    )

        total_minted = sum(minted_by_tier.values())
        if total_minted:
            logger.info(
                "_build_all_edge_entries_into: minted %d new key(s) (episodic=%d procedural=%d)%s",
                total_minted,
                minted_by_tier["episodic"],
                minted_by_tier["procedural"],
                " [deferred]" if defer else "",
            )
        return minted_by_tier, deferred_writes

    def _unique_speaker_predecessor(self, node: str) -> str:
        """Return the single non-empty ``speaker_id`` among *node*'s direct
        (1-hop) graph predecessors, or ``""`` when there is not exactly one.

        Reads ``self.merger.graph``.  Only DIRECT predecessors (in-edge source
        nodes) are considered; the walk is 1-hop, never transitive — a
        predecessor that itself carries no ``speaker_id`` contributes nothing
        and does NOT propagate a chain.  Authoritative-graph-state signal: a
        predecessor whose own node attribute ``speaker_id`` is non-empty.  No
        static predicate map is consulted.

        Used ONLY by the keyless-branch terminal fallback in
        :meth:`_build_all_edge_entries_into`, and ONLY when the subject node
        has no ``speaker_id`` of its own — fills gaps, never overwrites.

        Exactly one distinct non-empty speaker predecessor → return that
        ``speaker_id``.  Zero predecessors, all predecessors have empty
        ``speaker_id``, or ≥2 distinct non-empty speaker predecessors →
        return ``""`` (ambiguous or unattributed — never mis-attribute across
        speakers).

        Args:
            node: Canonical node key in ``self.merger.graph``.

        Returns:
            A non-empty ``speaker_id`` string when exactly one distinct speaker
            predecessor exists; ``""`` otherwise.
        """
        g = self.merger.graph
        if node not in g:
            return ""
        speakers = {
            sid
            for pred in g.predecessors(node)
            if (sid := (g.nodes[pred].get("speaker_id", "") or ""))
        }
        return next(iter(speakers)) if len(speakers) == 1 else ""

    def _build_tier_delta(
        self,
        *,
        active_before: dict[str, int],
        active_after: dict[str, int],
        minted_by_tier: dict[str, int],
    ) -> dict[str, dict]:
        """Build the per-tier delta record from shared grooming output.

        Unifies the ``staled_by_reason`` and ``minted`` fields that were
        previously computed in two divergent forked blocks (one per
        consolidation mode).  ``active_before`` and ``active_after`` remain
        mode-supplied inputs because they legitimately measure different
        substrates per mode (graph edges for simulate, served-key lengths for
        train).  Only ``staled_by_reason`` and ``minted`` are pure functions
        of shared grooming output and must converge.

        ``staled_by_reason`` is built by iterating ``self.merger.removal_ledger``
        and attributing each removed key to a tier via ``self.store.tier_of``.
        This includes ALL merger removal reasons (dedup, enrichment_same_as,
        contradiction_*, etc.) — more complete than the former train-only
        dedup-only approach.  Keys whose store entry is absent (``tier_of``
        returns ``None``) are genuinely unattributable and are skipped — this
        is a boundary skip, not error suppression.

        Args:
            active_before: Per-tier key count before the fold, e.g.
                ``{"episodic": 5, "semantic": 0, "procedural": 2}``.
            active_after: Per-tier key count after the fold.  Same shape as
                *active_before* but reflecting the post-fold state.
            minted_by_tier: Per-tier count of newly minted keys, e.g.
                ``{"episodic": 1, "procedural": 0}``.  For simulate mode,
                pass a single-tier dict derived from enrichment ``new_edges``.

        Returns:
            A mapping from tier name to
            ``{active_before, active_after, staled_by_reason, minted}``.
            Only tiers that appear in at least one of the three input dicts
            are included (generic — no hardcoded tier list).
        """
        self._ensure_store()

        # Attribute each ledger removal to a tier via the entry store.
        ledger = getattr(self.merger, "removal_ledger", {})
        staled: dict[str, dict[str, int]] = {}
        for removed_key, rec in ledger.items():
            tier = self.store.tier_of(removed_key)
            if tier is None:
                # Key not owned by the store — genuinely unattributable (e.g.
                # simulate mode has no store entries; enrichment_same_as keys
                # that were removed before store registration).  Boundary skip.
                continue
            reason = rec.get("reason", "dedup")
            tier_bucket = staled.setdefault(tier, {})
            tier_bucket[reason] = tier_bucket.get(reason, 0) + 1

        all_tiers = set(active_before) | set(active_after) | set(minted_by_tier)
        result: dict[str, dict] = {}
        for t in all_tiers:
            result[t] = {
                "active_before": active_before.get(t, 0),
                "active_after": active_after.get(t, 0),
                "staled_by_reason": staled.get(t, {}),
                "minted": minted_by_tier.get(t, 0),
            }
        return result

    def _build_registry_true_relations(self, keys: "list[str] | None" = None) -> "list[Relation]":
        """Build registry-true :class:`Relation` objects for a set of active keys.

        Used as the fold's re-merge input so the merge surface is grounded in
        registry-true (subject, predicate, object) content rather than the
        lossy reconstruction result.

        For each key the content is sourced from the store entry
        (``store.get(key)``).  The fold hydrates the store from the venue's
        :class:`~paramem.memory.source.MemorySource` before this runs
        (:meth:`_hydrate_store_for_fold`), so an absent entry means no venue
        holds content for that live key; it is logged as an orphan and skipped.
        Bookkeeping never carries SPO.

        ``relation_type``, ``speaker_id``, ``last_seen``, and ``first_seen``
        always come from bookkeeping (never from the entry payload which
        carries the merge-time value), each via ``bk.get(...)`` with a
        tolerant default — ``bk`` is legitimately ``{}`` for an active key
        that has content but no bookkeeping record at all (e.g. a key
        migrated by ``active_store_migration._migrate_tier_simulate_to_train``,
        which writes ``store.put`` without ``set_bookkeeping``).  This is a
        distinct case from a bookkeeping record that predates a field: the
        mandatory-``first_seen`` guarantee is enforced at the write side
        (``set_bookkeeping`` requires it as a keyword); this reconstruction
        read tolerates a missing record entirely, exactly like its sibling
        fields.

        Args:
            keys: Optional explicit list of active-key strings to process.
                When ``None`` (the default), iterates ``store.all_active_keys()``
                so behavior is identical to the pre-parameter baseline.  When
                provided, only those keys are processed; the caller is responsible
                for supplying a subset of active keys.

        Returns:
            A list of :class:`Relation` with ``indexed_key`` set so the key
            travels through :class:`GraphMerger` onto the merged edge.
        """
        relations: list[Relation] = []
        key_iter = keys if keys is not None else self.store.all_active_keys()
        for key in key_iter:
            entry = self.store.get(key)
            bk = self.store.bookkeeping_for_key(key) or {}

            if entry is not None:
                subj = entry.get("subject", "")
                pred = entry.get("predicate", "")
                obj = entry.get("object", "")
            else:
                # No content in the store and none in the source of truth.
                # Bookkeeping never carries SPO; log and skip (orphan).
                logger.debug(
                    "_build_registry_true_relations: key=%s has no entry — skipping (orphan)",
                    key,
                )
                continue

            if not pred:
                # No predicate: not keyable — skip.
                logger.debug(
                    "_build_registry_true_relations: key=%s has no predicate — skipping",
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

    def _materialize_consolidation_graph(
        self,
        *,
        source: "Literal['weights', 'disk']" = "weights",
        tier: "str | None" = None,
        keys: "list[str] | None" = None,
        extra_relations: "list[Relation] | None" = None,
        dedup_target_keys: "list[str] | None" = None,
        resolve_contradictions_recon: bool = False,
        resolve_contradictions_extra: bool = False,
    ) -> "tuple[set[str], list[Relation]]":
        """Reconstruct active keys from adapter weights and re-merge registry-true relations.

        This is the *Materialize* stage of the fold pipeline:

        1. Probe every active key from adapter weights via :func:`reconstruct_graph`
           (``strict=False``).  Skipped when ``source="disk"`` — that venue has no
           adapter weights to probe.
        2. Compute ``recall_miss_keys`` — keys whose reconstructed SPO disagrees with
           registry-true SPO, or whose reconstruction failed outright.  The set is
           computed over the CALLER'S key set (*keys*, defaulting to
           ``store.all_active_keys()``) BEFORE the graph reset, so only registered
           keys the caller actually folds can appear in the miss set.  Skipped with
           step 1; ``set()`` for ``source="disk"``.
        3. Reset the merger's keying graph to empty (``merger.reset_graph()``).
        4. Build registry-true :class:`Relation` objects via
           :meth:`_build_registry_true_relations` and re-merge them into the fresh
           keying graph inside a gradient-checkpointing guard.
        5. If ``extra_relations`` is supplied and non-empty, re-merge those relations
           into the fresh keying graph (see *resolve_contradictions_extra*).  This
           allows the interim mini-fold to inject the current cycle's pending-session
           relations alongside the slot's recalled registry-true keys.  At interim,
           merge order (slot first, pending second) encodes recency: the NEW pending
           supersedes the OLD slot when ``resolve_contradictions_extra=True``.
        6. If ``dedup_target_keys`` is not ``None`` (interim recital dedup, always
           computed by the interim caller), build registry-true relations for
           that key subset and re-merge them LAST — AFTER the ``extra_relations``
           merge in step 5 — with ``resolve_contradictions=False`` unconditionally.
           See the INVARIANT below.
        7. Emit debug snapshots ("reconstructed" before re-merge, "merged" after).

        **INVARIANT — extra_relations and the recall-miss set:**
        ``extra_relations`` participate in the MERGE / Case-1-adopt step ONLY.
        They MUST NOT enter the ``recall_miss_keys`` set.  That set is computed
        over the resolved *keys* in step 2, BEFORE the reset — pending
        unregistered relations (not yet in the registry) therefore cannot distort it.
        Both ``extra_relations=None`` and ``extra_relations=[]`` are valid no-ops for
        the fold caller (fold passes ``None``; the check is ``if extra_relations``).

        **INVARIANT — dedup_target_keys, ordering, and exclusion contract:**
        ``dedup_target_keys`` relations participate in the MERGE / Case-1 step
        ONLY, exactly like ``extra_relations`` — they are excluded from keying
        by the CALLER, which must pass the same key set as ``exclude_keys`` to
        :meth:`_build_all_edge_entries_into` so the dedup-target (main-tier)
        keyed edges are neither minted nor keyed-replayed into the training
        set.  The merge fires ONLY when ``dedup_target_keys is not None`` —
        ``None`` is a true no-op (the full-fold callers never
        pass this param, so their behavior is byte-identical to before this
        change).  Never pass ``None`` to :meth:`_build_registry_true_relations`
        as the resolved ``keys=`` argument here — that means "all active
        keys" and would silently pull the entire store into the merge; an
        empty *dedup_target_keys* list (feature enabled but no dedup targets
        found) is the correct "nothing to dedup" signal and resolves to ``[]``.
        The merge is placed LAST (after ``extra_relations``, not before) so
        that when ``refinement_contradiction == "on"``, the contradiction-
        enabled recon/extra merges complete before the dedup-target edges
        exist — a session fact contradicting a main-tier dedup target cannot
        retire the main-tier edge via Case-2 REPLACE, because there is no
        main-tier edge present yet at that point.  ``resolve_contradictions``
        is hardcoded ``False`` for this merge (not driven by config) — it must
        never run cardinality resolution over main-tier facts.

        **Speaker-ID note (unified path):** Both the recon path and the
        ``extra_relations`` path call
        :meth:`~paramem.graph.merger.GraphMerger.merge_relations`, which
        invokes the module-level ``_synth_speaker_entities`` helper in
        :mod:`paramem.graph.merger` to produce a synthetic
        :class:`~paramem.graph.schema.Entity` (``entity_type="person"``) for each
        speaker-attributed subject.
        :meth:`~paramem.graph.merger.GraphMerger._upsert_entity` stamps
        ``speaker_id`` onto the subject node so that
        :meth:`_build_all_edge_entries_into` reads the correct ``speaker_id``
        (dcf4189 invariant: minted interim keys must inherit their subject node's
        ``speaker_id``, not fall back to ``""``).
        Non-speaker subjects (``speaker_id == ""``) require no entity — their nodes
        remain attribute-free for ``speaker_id``, which resolves to ``""`` in the
        walk (correct default).

        Args:
            source: **Weight-probe gate only.**  It does NOT select the merge
                input — :meth:`_build_registry_true_relations` reads the store
                in both venues, and the store is populated in both venues
                (``MemoryStore.load_registries_from_disk`` hydrates every main
                and interim tier regardless of venue; the per-entry payload
                comes from the venue's
                :class:`~paramem.memory.source.MemorySource`).

                - ``"weights"``: run steps 1-2 — probe adapter weights via
                  :func:`reconstruct_graph` and compute ``recall_miss_keys``.
                - ``"disk"``: skip steps 1-2 (no adapter weights exist).
                  ``recall_miss_keys`` is ``set()`` — a retrain signal is
                  meaningless for a venue that does not retrain.  Every other
                  step runs identically.
            tier: Forwarded to :func:`reconstruct_graph` as ``tier``.  When
                ``None`` (the default), all tiers are probed.  Ignored when
                ``source="disk"`` (no reconstruction runs).
            keys: **The caller's key set for this materialize.**  Scopes both
                the registry-true merge input
                (:meth:`_build_registry_true_relations`) and the recall-miss
                comparison, so a key the caller did not fold can neither enter
                the merge nor be reported as a recall miss.  When ``None`` (the
                default) it resolves to ``store.all_active_keys()``.  Honoured
                in BOTH venues.
            extra_relations: Optional list of :class:`Relation` objects to merge
                into the fresh keying graph after the registry-true re-merge.
                Intended for the interim mini-fold: the caller captures the
                pending-session relations from ``self.merger.graph`` BEFORE calling
                this method (since the reset inside will wipe them) and passes them
                here so they survive the reset and co-reside with the slot's
                recalled facts.  The non-consume-pending fold caller passes
                ``None`` (no-op).
            dedup_target_keys: Optional list of active-key strings identifying
                main-tier facts to merge as dedup targets — see the INVARIANT
                above.  ``None`` (the default) is a true no-op: no dedup merge
                runs.  The full-fold callers never pass this param.  The interim
                fresh-derivation caller always passes the caller-scoped subset
                (session-touched main-tier keys; possibly empty).
            resolve_contradictions_recon: Forwarded to
                :meth:`~paramem.graph.merger.GraphMerger.merge_relations` for
                the registry-true recon merge.  Driven by
                ``config.refinement_contradiction == "on"``.
                At fold, ``timestamp=""`` is passed to the merger so legacy
                relations (``last_seen=""``) never fabricate a NOW recency value.
                A legacy relation coexists with its rivals only when every rival
                is also undated; a genuinely dated rival always outranks it
                (dated wins over undated) and the legacy relation is retired.
            resolve_contradictions_extra: Forwarded to
                :meth:`~paramem.graph.merger.GraphMerger.merge_relations` for
                the ``extra_relations`` (pending-session) merge.  Driven by
                ``config.refinement_contradiction == "on"``.
                Ignored when ``extra_relations`` is empty.

        Returns:
            A 2-tuple ``(recall_miss_keys, recon_relations)`` where:

            - ``recall_miss_keys`` — :class:`set` of key strings that failed
              reconstruction or whose SPO diverged from the registry.  Always
              ``set()`` for ``source="disk"`` (no weight reconstruction).
            - ``recon_relations`` — the :class:`list` of :class:`Relation` objects
              fed into the registry-true re-merge (registry-true SPO, with
              ``indexed_key`` set).  ``extra_relations`` are NOT included here —
              they travel through a separate merge call inside this method.
        """
        # --- Reconstruct all active keys from adapter weights (weights venue) ---
        # Probes every active key across all tiers; recovers (subject, predicate, object)
        # from the trained weights.  Reconstruction yields SPO ONLY — no relation_type.
        # strict=False: failures are logged and recorded in recon_result.failures; the
        # cycle continues with whatever SPO triples can be recovered.
        # Reconstruction is used ONLY to identify recall-miss keys (keys whose
        # reconstructed SPO disagrees with registry-true SPO, or whose reconstruction
        # failed outright).  A recall miss is a retry signal; the key stays in the
        # training set with its registry-true content.  It does NOT drop the key.
        # The disk venue has no adapter weights, so it skips the probe outright:
        # recall_miss_keys stays empty and _recon_graph stays None.
        # The caller's key set, resolved once: it scopes the recall-miss
        # comparison AND the merge input, so the two can never describe
        # different key sets.
        scoped_keys: list[str] = (
            list(keys) if keys is not None else list(self.store.all_active_keys())
        )

        recall_miss_keys: set[str] = set()
        _recon_graph = None
        if source == "weights":
            recon_result = reconstruct_graph(self, tier=tier, strict=False)
            if recon_result.failures:
                logger.warning(
                    "_materialize_consolidation_graph: %d key(s) failed reconstruction "
                    "(retry signal — keys kept in training set with registry-true content)",
                    len(recon_result.failures),
                )

            # --- Compute recall-health/retry set BEFORE reset_graph() ---
            # This MUST run after reconstruct_graph (which produces recon_result.graph
            # as a SEPARATE nx.MultiDiGraph, distinct from self.merger.graph) and BEFORE
            # reset_graph() (which clears self.merger.graph).  The ordering is safe because
            # recon_result.graph is a freshly constructed MultiDiGraph
            # (reconstruct.py:142) unaffected by the subsequent reset.
            #
            # Build a lookup of reconstructed SPO per key from the recon graph.
            from paramem.memory.persistence import _IK_KEY_ATTR as _IK_ATTR

            _recon_spo_by_key: dict[str, tuple[str, str, str]] = {}
            for _rh_subj, _rh_obj, _rh_data in recon_result.graph.edges(data=True):
                _rh_key = _rh_data.get(_IK_ATTR, "")
                _rh_pred = _rh_data.get("predicate", "")
                if _rh_key and _rh_pred:
                    _recon_spo_by_key[_rh_key] = (_rh_subj, _rh_pred, _rh_obj)

            # recall_miss_keys: keys whose reconstruction failed OR whose reconstructed
            # SPO disagrees with registry-true SPO.  These are flagged for retrain but
            # their registry-true triple still enters the merge input (never dropped).
            _scoped_key_set = set(scoped_keys)
            recall_miss_keys: set[str] = {
                f["key"] for f in recon_result.failures if f["key"] in _scoped_key_set
            }
            for _rh_key in scoped_keys:
                _rt_entry = self.store.get(_rh_key)
                _rt_subj = (_rt_entry or {}).get("subject", "") if _rt_entry else ""
                _rt_pred = (_rt_entry or {}).get("predicate", "") if _rt_entry else ""
                _rt_obj = (_rt_entry or {}).get("object", "") if _rt_entry else ""
                _recon_spo = _recon_spo_by_key.get(_rh_key)
                if _recon_spo is None:
                    # No recon edge: counts as a failure (already in
                    # recon_result.failures, or missing outright).
                    recall_miss_keys.add(_rh_key)
                else:
                    _r_subj, _r_pred, _r_obj = _recon_spo
                    if (
                        _r_subj != _rt_subj
                        or canonical(_r_pred) != canonical(_rt_pred)
                        or _r_obj != _rt_obj
                    ):
                        recall_miss_keys.add(_rh_key)

            if recall_miss_keys:
                logger.info(
                    "_materialize_consolidation_graph: %d key(s) in recall-miss set "
                    "(kept in training with registry-true content): %s",
                    len(recall_miss_keys),
                    sorted(recall_miss_keys),
                )
            _recon_graph = recon_result.graph

        # --- Reset keying graph and re-merge registry-true relations ---
        # Reset the merger's keying surface to EMPTY before re-merging so
        # provenance keying is unconditional.  Without the reset, pre-existing
        # edges from ingest-time merges or a loaded graph would share the keying
        # surface and the Case-1-adopt collision path could degrade provenance
        # keying.
        # Recurrence is now durable in bookkeeping — discarding the prior graph
        # loses nothing; the transient graph edge counts were the broken store.
        self.merger.reset_graph()
        logger.info(
            "_materialize_consolidation_graph: keying graph reset to empty for the"
            " reconstruct→re-merge pass"
        )

        # --- Build merge input from registry-true SPO (NOT reconstruction) ---
        # Each relation carries its indexed_key so the key travels through
        # GraphMerger.merge() onto the merged edge (provenance keying).
        #
        # resolve_contradictions_recon is driven by config.refinement_contradiction.
        # When "on": the merger may retire strictly-older registry-true edges; since
        # timestamp="" is passed (default), a legacy relation (last_seen="") never
        # fabricates a NOW recency value.  An empty last_seen sorts as the oldest
        # possible timestamp, so a legacy key coexists with its rivals only when
        # every rival is ALSO undated; a genuinely dated rival always outranks it
        # (dated wins over undated) and the legacy key is retired.
        # Two registry keys sharing identical (s,p,o) STILL fire Case-1 (the merger
        # identity is correct given correct inputs), and the collapsed key is recorded
        # in merger.collapsed.  The drift-partition step below soft-stales that key.
        # Debug: snapshot the reconstructed graph (before re-merge mutates the
        # keying surface).  Self-gated; no-op when save_cycle_snapshots=False.
        # The disk venue ran no reconstruction, so it snapshots the just-reset
        # (empty) keying graph — the artifact chain (reconstructed → merged →
        # enriched) is emitted in both venues.
        on_fold_graph(
            self.merger.graph if _recon_graph is None else _recon_graph,
            label="reconstructed",
        )

        recon_relations: list[Relation] = self._build_registry_true_relations(keys=scoped_keys)

        # Merge registry-true reconstructed relations.  merger.merge_relations
        # synthesises speaker entities from the relation list (same logic as the
        # extra-relations path below) so reconstructed person nodes receive
        # entity_type="person" + speaker_id from bookkeeping.  Before unification
        # this block used entities=[] → concept nodes with no speaker_id.
        # resolve_contradictions_recon is driven by config.refinement_contradiction.
        # timestamp="" (default) ensures legacy keys (last_seen="") never fabricate a
        # NOW recency value; a legacy key coexists only when every rival is also
        # undated, and is retired when a genuinely dated rival outranks it (dated
        # wins over undated).
        # The gradient-checkpointing guard fires when resolve_contradictions is
        # True and a model is present — the contradiction path calls model.generate().
        _recon_needs_guard = (
            getattr(self, "model", None) is not None and resolve_contradictions_recon
        )
        if _recon_needs_guard:
            self._disable_gradient_checkpointing()
        try:
            self.merger.merge_relations(
                recon_relations,
                session_id="__full_consolidation_recon__",
                log_label="reconstructed triples",
                resolve_contradictions=resolve_contradictions_recon,
            )
        finally:
            if _recon_needs_guard:
                self._enable_gradient_checkpointing()

        # --- Re-merge extra_relations (interim mini-fold pending-session content) ---
        # INVARIANT: extra_relations participate in MERGE / Case-1-adopt ONLY.
        # They are NOT included in recall_miss_keys (computed above, before the reset).
        # extra_relations=None and extra_relations=[] are both valid no-ops (fold caller
        # passes None; interim passes the pending-session relations from merger.graph).
        # resolve_contradictions_extra: driven by config.refinement_contradiction.
        # At fold extra_relations=None so this merge is a no-op.
        _extra_needs_guard = (
            getattr(self, "model", None) is not None and resolve_contradictions_extra
        )
        if _extra_needs_guard:
            self._disable_gradient_checkpointing()
        try:
            self.merger.merge_relations(
                extra_relations or [],
                session_id="__interim_pending_sessions__",
                log_label="extra (pending-session) relations",
                resolve_contradictions=resolve_contradictions_extra,
            )
        finally:
            if _extra_needs_guard:
                self._enable_gradient_checkpointing()

        # --- Re-merge dedup_target_keys (interim recital dedup) LAST ---
        # Unconditional feature: the interim fresh-derivation caller always
        # computes and passes dedup_target_keys (possibly an empty list).
        # GUARD: only merge when dedup_target_keys is not None — None is the
        # byte-identical no-op for every caller that doesn't pass this param
        # (full-fold, simulate).  Never pass None straight through to
        # _build_registry_true_relations(keys=...) — that sentinel means "all
        # active keys" there, not "no keys".
        # Placed AFTER the extra_relations merge above (not before): with
        # refinement_contradiction="on", the contradiction-enabled recon/extra
        # merges complete before any dedup-target (main-tier) edge exists, so
        # a session fact contradicting a main-tier dedup target cannot retire
        # that main-tier edge via Case-2 REPLACE.  resolve_contradictions is
        # hardcoded False (not config-driven): this merge must never run
        # cardinality resolution over main-tier facts.  No gradient-checkpointing
        # guard is needed here — resolve_contradictions=False never fires
        # model.generate().
        if dedup_target_keys is not None:
            dedup_relations = self._build_registry_true_relations(keys=dedup_target_keys)
            self.merger.merge_relations(
                dedup_relations,
                session_id="__interim_maintier_dedup__",
                log_label="main-tier recital dedup",
                resolve_contradictions=False,
                credit_adopt_reinforcement=True,
            )

        # Debug: snapshot the merged graph (after re-merge, before enrichment).
        # Emits even when recon_relations is empty so the fold always produces a
        # merged snapshot.  Self-gated; no-op when save_cycle_snapshots=False.
        on_fold_graph(self.merger.graph, label="merged")

        return recall_miss_keys, recon_relations

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
            gc_disable=self._disable_gradient_checkpointing,
            gc_enable=self._enable_gradient_checkpointing,
        )

    def _refine_consolidation_graph(
        self,
        recon_relations: "list[Relation]",
        *,
        normalize: bool = False,
        enrich: bool = False,
    ) -> None:
        """Run graph normalization, enrichment, and recurrence bumps after the Materialize stage.

        This is the *Refine* stage of the fold pipeline:

        1. Construct a per-call :class:`~paramem.training.graph_tier.GraphTierRefiner`
           bound to this loop's ``self.merger`` and the current ``self.model`` /
           ``self.tokenizer``, and call
           :meth:`~paramem.training.graph_tier.GraphTierRefiner.refine` with the
           ``normalize`` / ``enrich`` flags.  ``refine()`` runs Cloud graph
           enrichment (additive second-order discovery) when ``enrich`` is
           ``True``, THEN the whole-graph local-model normalization pass
           (predicate alignment + entity merge + predicate-synonym
           normalization) when ``normalize`` is ``True`` — enrichment first,
           so normalization sees any predicate synonym enrichment just
           minted and collapses it before the fold's key assembly reads the
           graph. Constructed fresh on every call, never cached: ``self.model``
           is re-wrapped by adapter operations elsewhere in the fold, so a
           cached refiner would risk pinning a stale handle.
        2. When ``result.enrichment`` carries ``aborted_reason == "vram"``
           (the enrichment pass stopped early on
           :class:`~paramem.utils.vram_guard.VramExhausted` but kept
           whatever it already merged — see
           :func:`~paramem.training.graph_enrich.enrich_graph`'s
           docstring), record an ``enrichment_degraded`` incident (severity
           ``"warning"``) via the same :func:`~paramem.server.incidents.
           record_incident` surface used elsewhere in this class, when
           ``self._incidents_state_dir`` is configured.  Never raises: the
           fold always proceeds past this step, training on the
           merged-but-unenriched graph — enrichment self-heals at the next
           FULL fold (the pass is full-fold only; an intervening interim
           cycle never runs it, so recovery does not happen there).
        3. Emit a debug snapshot ("enriched") after the refine step (or
           immediately when both stages are skipped). Emitted from the loop
           rather than the refiner, which calls :func:`on_normalization`
           directly for its own pass.
        4. Two INDEPENDENTLY-guarded recurrence-bump blocks, reading the
           reinforcement maps off the returned
           :class:`~paramem.training.graph_tier.RefineResult` (never unioned
           under one relaxed guard — each dict is consumed by its own loop with
           its own guard, so the recital-dedup credit adds a bump without
           changing the pre-existing ``reinforcements`` bump contract):

           - If ``recon_relations`` is non-empty, scan ``result.reinforcements``
             for Case-1 duplicate-SPO collapses and call
             :meth:`~paramem.memory.store.MemoryStore.bump_recurrence` for each
             surviving key.  This guard is byte-identical to the original
             contract: a re-merge must actually have run for this bump to fire.
           - If ``result.adopt_reinforcements`` is non-empty, call
             :meth:`~paramem.memory.store.MemoryStore.bump_recurrence` for each
             main-tier key credited by the interim recital-dedup merge's
             Case-1-adopt.  This guard is independent of ``recon_relations`` —
             a recital-only interim cycle has empty ``recon_relations`` (no slot
             keys reconstructed) but a non-empty ``adopt_reinforcements``, and
             must still bump.  The two dicts are disjoint by construction (see
             the block below), so no key is ever double-bumped across the two.

        Args:
            recon_relations: The list of registry-true :class:`Relation` objects
                produced by :meth:`_materialize_consolidation_graph`.  Used as
                the guard for the ``result.reinforcements`` bump block only (see
                point 3) — when empty, that block is skipped (no re-merge was
                performed so ``result.reinforcements`` will be empty too).  Does
                NOT gate the independent ``adopt_reinforcements`` bump block.
            normalize: When ``True``, run the local-model predicate-synonym
                normalization pass.
                Callers pass ``normalize=scope.normalize``.
                Default ``False``.
            enrich: When ``True``, run cloud-cloud graph enrichment (additive
                discovery).  Callers pass ``enrich=scope.enrich`` — at the
                full fold ``scope.enrich`` is set at construction to
                ``refinement_enrichment=="on" and cloud_enabled``; at the
                interim scope it is pinned ``False`` unconditionally
                (graph-tier enrichment is a full-fold-only pass — see
                :attr:`FoldScope.enrich`).
                Default ``False``.
        """
        refiner = self.build_tier_refiner(self.merger)
        result = refiner.refine(normalize=normalize, enrich=enrich)

        # Surface a VRAM-driven enrichment degrade (U5) as an operator-visible
        # incident — the SAME record_incident surface extract_session's
        # cloud_enrichment_degraded path uses above.  result.enrichment is the
        # raw diagnostics dict enrich_graph returns; aborted_reason == "vram"
        # means the chunk loop stopped early on VramExhausted but kept
        # whatever it already merged (see enrich_graph's docstring) rather
        # than aborting the fold.  Severity "warning" (the fold succeeds
        # regardless): enrichment self-heals at the next FULL fold, since the
        # pass runs over the cumulative graph every full fold (never at an
        # intervening interim cycle — full-fold only), so there is nothing to
        # retry here.
        if (
            result.enrichment is not None
            and result.enrichment.get("aborted_reason") == "vram"
            and self._incidents_state_dir is not None
        ):
            from paramem.server.incidents import record_incident

            record_incident(
                self._incidents_state_dir,
                type="enrichment_degraded",
                key="graph_enrich_vram",
                severity="warning",
                summary=(
                    "Consolidation: graph-tier enrichment stopped early on VRAM "
                    "exhaustion — kept already-merged chunks"
                ),
                detail={
                    "type": "enrichment_degraded",
                    "chunks": result.enrichment.get("chunks", 0),
                    "at": datetime.now(timezone.utc).isoformat(),
                },
            )

        # Debug: snapshot the refined graph (after normalization + enrichment, or
        # immediately when both are skipped at level off).
        # Self-gated; no-op when save_cycle_snapshots=False.
        on_fold_graph(self.merger.graph, label="enriched")

        if recon_relations:
            # --- Reinforcement bump: Case-1 duplicate-SPO collapses ---
            # result.reinforcements contains the surviving ik_key for every Case-1
            # collision fired during the re-merge.  A collision means two active keys
            # shared the same (s,p,o) — the incoming key drifts and the existing
            # edge's key is the survivor.  The survivor's reinforcement_count
            # represents how many times this fact was independently extracted
            # (and re-keyed) across sessions before this fold collapsed the duplicates.
            # result.reinforcements is dict[ik_key, (last_seen, first_seen)] —
            # the freshest last_seen and earliest first_seen are carried directly
            # from the edge so bump_recurrence can advance bookkeeping without
            # fabricating now().  This guard (recon_relations non-empty) is the
            # original, unchanged contract — the recital-dedup credit does not
            # relax it; it only adds the independent adopt-credit block below.
            for _rein_key, (_rein_ls, _rein_fs) in result.reinforcements.items():
                if _rein_key:
                    self.store.bump_recurrence(
                        _rein_key,
                        cycle=self.cycle_count,
                        timestamp=_rein_ls,
                        first_seen=_rein_fs,
                    )
                    logger.debug(
                        "_refine_consolidation_graph: bumped recurrence for key=%s "
                        "(intra-fold duplicate-SPO collapse)",
                        _rein_key,
                    )

        # --- Reinforcement bump: interim recital-dedup Case-1-adopt credits ---
        # result.adopt_reinforcements contains the main-tier ik_key credited by the
        # interim recital-dedup merge's Case-1-adopt (a recited pending fact adopts
        # a main key onto its keyless edge).  Independently guarded from the block
        # above — NOT unioned under one relaxed guard — so this credit is additive
        # and never changes when the ``reinforcements`` block fires.  A recital-only
        # interim cycle has empty ``recon_relations`` (no slot keys reconstructed)
        # but a non-empty ``adopt_reinforcements``, and must still bump here.  The
        # two dicts are disjoint by construction (reinforcements records the
        # SLOT/surviving key from the both-keyed collision elif — both edges
        # keyed; adopt_reinforcements records the MAIN key from the Case-1-adopt
        # branch — existing edge keyless), so no key is ever double-bumped across
        # the two blocks.
        if result.adopt_reinforcements:
            for _rein_key, (_rein_ls, _rein_fs) in result.adopt_reinforcements.items():
                if _rein_key:
                    self.store.bump_recurrence(
                        _rein_key,
                        cycle=self.cycle_count,
                        timestamp=_rein_ls,
                        first_seen=_rein_fs,
                    )
                    logger.debug(
                        "_refine_consolidation_graph: bumped recurrence for key=%s "
                        "(recital-dedup adopt)",
                        _rein_key,
                    )

    def _run_recall_sanity_probe(
        self,
        adapter_name: str,
        entries: list[dict],
        *,
        max_probe: int = 100,
        debug_phase: str | None = None,
    ) -> float:
        """Probe up to *max_probe* entries against *adapter_name* and return the recall rate.

        Used by :meth:`_verify_saved_adapter_from_disk` to check the recall
        of an adapter reloaded from disk.  Keeping the logic in one place
        makes the sanity contract identical everywhere: same sample size,
        same probe harness, same failure semantics (probe exception → ``0.0``
        so callers treat it as a rollback trigger rather than a mysterious
        skip).

        The caller is responsible for deciding what to do with the
        returned rate (threshold compare, rollback, health update).

        Args:
            adapter_name: Adapter to probe.  Must be loaded and switchable
                (caller holds the GPU lock).  The default ``"episodic"``
                in :func:`evaluate_indexed_recall` is deliberately NOT
                relied on — silently probing the wrong tier would mask
                tier-specific regressions.
            entries: Candidate entries to probe.  Sampled uniformly
                down to *max_probe* when longer.  An empty list returns
                ``1.0`` (nothing to prove → healthy by default).
            max_probe: Cap on probe size.  100 is chosen to keep the
                probe cheap enough to run inline even inside the
                interim training path.
            debug_phase: When not ``None``, the per-key verdict (including
                ``raw_output``) is persisted to the debug snapshot via
                :func:`~paramem.utils.artifacts.on_recall_probe`
                under ``<debug_base>/recall_probes/<debug_phase>_<adapter_name>.json``.
                Only written on the success path (where ``recall_result`` is
                available); probe exceptions still return ``0.0`` without writing.

        Returns:
            Recall rate in ``[0.0, 1.0]``.  On probe-harness exception,
            returns ``0.0`` so the caller trips its sanity threshold.
        """
        if not entries:
            return 1.0

        probe_pairs = entries
        if len(probe_pairs) > max_probe:
            probe_pairs = random.sample(probe_pairs, max_probe)

        try:
            from paramem.memory.entry import build_registry
            from paramem.training.recall_eval import evaluate_indexed_recall

            probe_registry = build_registry(probe_pairs)
            self._disable_gradient_checkpointing()
            recall_result = evaluate_indexed_recall(
                self.model,
                self.tokenizer,
                probe_pairs,
                probe_registry,
                adapter_name=adapter_name,
                batch_size=self.training_config.recall_probe_batch_size,
            )
            if debug_phase is not None:
                with self._artifact_scope():
                    on_recall_probe(
                        recall_result["per_key"],
                        phase=debug_phase,
                        adapter_name=adapter_name,
                    )
            return float(recall_result["rate"])
        except Exception:
            logger.exception(
                "_run_recall_sanity_probe: recall probe failed for adapter %s — "
                "returning 0.0 so caller trips the sanity gate",
                adapter_name,
            )
            return 0.0

    def _prune_old_slots(self, tier_root: Path, live_slot: Path, keep: int) -> None:
        """Remove post-promotion adapter slots beyond the retention budget.

        Scans *tier_root* (e.g. data/ha/adapters/episodic/) for slot-shaped
        subdirectories. The slot just promoted (*live_slot*) is always retained
        — pass it explicitly because the registry commit at the end of
        _save_adapters writes its hash to disk AFTER this call, and reading
        the registry here would race. Remaining slots are ordered by st_mtime
        descending; the *keep* most-recent are retained, older ones are
        rmtree'd.

        Filters via paramem.adapters.manifest.is_slot_name so non-slot
        siblings (interim_<stamp>/, indexed_key_registry.json, .pending/)
        are untouched.

        Args:
            tier_root: <adapter_dir>/<tier>/ scoped to one adapter kind.
            live_slot: Path to the slot just promoted; immune to pruning.
            keep: Max number of non-live prior slots to retain (>=0).
        """
        import shutil as _shutil

        from paramem.adapters.manifest import is_slot_name

        if not tier_root.is_dir() or keep < 0:
            return
        candidates: list[Path] = []
        for entry in tier_root.iterdir():
            if entry.name.startswith("."):
                continue
            if not entry.is_dir():
                continue
            if entry == live_slot:
                continue
            if not is_slot_name(entry.name):
                continue
            candidates.append(entry)
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for stale in candidates[keep:]:
            _shutil.rmtree(stale, ignore_errors=False)
            logger.info("_prune_old_slots: removed %s (retention=%d)", stale, keep)

    def _verify_saved_adapter_from_disk(
        self,
        adapter_name: str,
        slot_path: Path,
        entries: list[dict],
        *,
        threshold: "float | None" = None,
        max_probe: int = 100,
    ) -> float:
        """Reload an adapter from its on-disk slot and probe recall integrity.

        Closes the silent-partial-write gap: the in-RAM recall probe in
        :meth:`_run_recall_sanity_probe` runs on the trained weights still in
        memory.  This method loads the *saved* artifact back from disk into an
        isolated verify slot, probes it with the same harness, then drops the
        slot.  A corrupt or truncated ``adapter_model.safetensors`` (e.g. dirty
        pages not flushed before a kernel crash) will either fail to parse —
        triggering a ``recall=0.0`` → gate trip — or produce degraded recall
        that falls below *threshold*.

        The verify slot is named ``f"{adapter_name}_verify"`` so it cannot
        collide with any production adapter name (``episodic``, ``semantic``,
        ``procedural``, or any ``episodic_interim_*`` slot).  The original
        adapter remains active throughout; after the probe the verify slot is
        dropped and the original adapter is re-activated so the model is left
        in the same state as on entry.

        PEFT pitfall avoidance:
        - Uses ``model.load_adapter(slot_path, adapter_name=verify_name)``
          (same as ``_mount_adapters_from_slots``) rather than
          ``PeftModel.from_pretrained`` to avoid nested tensor name prefixes.
        - Patches ``peft_config[verify_name].base_model_name_or_path`` when
          PEFT sets it to ``None`` (happens for second-and-later adapters).
        - Uses ``try/finally`` so the verify slot is always dropped even if
          the probe raises.  Does NOT call ``add_adapter`` or ``get_peft_model``
          after ``delete_adapter`` (CLAUDE.md PEFT rule).

        Args:
            adapter_name: Production adapter that was just saved.  Used as the
                active adapter to restore after the probe.
            slot_path: Absolute path to the slot directory written by
                :func:`~paramem.models.loader.atomic_save_adapter`.  The
                adapter files (``adapter_model.safetensors``,
                ``adapter_config.json``) sit directly inside this directory
                (post-flatten step of ``atomic_save_adapter``).
            entries: Entries encoded into the adapter.  Sampled down to
                *max_probe* if longer.  An empty list returns ``1.0`` (no keys
                to verify → healthy by default).
            threshold: Minimum recall the disk artifact must achieve.  When
                ``None`` (default), the value is read from
                ``self.config.recall_sanity_threshold``.
            max_probe: Maximum number of entries to probe.  Passed through to
                :meth:`_run_recall_sanity_probe`.

        Returns:
            Recall rate from the disk-loaded adapter in ``[0.0, 1.0]``.

        Raises:
            RuntimeError: When ``recall < threshold``, signalling that the
                on-disk artifact is corrupt or degraded.  The caller's
                try/except in ``_run_extraction_phase`` (app.py) will then skip
                ``mark_consolidated``, leaving sessions pending for the next
                cycle to retry.
        """
        from peft import PeftModel

        # Resolve threshold from config when the caller did not supply an override.
        if threshold is None:
            threshold = self.config.recall_sanity_threshold

        if not entries:
            logger.debug(
                "_verify_saved_adapter_from_disk: no entries for %s — skipping",
                adapter_name,
            )
            return 1.0

        verify_name = f"{adapter_name}_verify"
        logger.info(
            "_verify_saved_adapter_from_disk: loading slot %s as '%s' for integrity check",
            slot_path,
            verify_name,
        )

        from paramem.models.loader import _adapter_slot_for_load

        recall_rate: float = 0.0
        try:
            # Load the saved slot into an isolated verify adapter.
            # Use the same pattern as _mount_adapters_from_slots (app.py L955):
            # model.load_adapter(str(slot), adapter_name=name) for PeftModel.
            # _adapter_slot_for_load transparently decrypts the safetensors into
            # an anonymous in-memory file (memfd) so the encrypted disk artifact
            # exercises the real round-trip: save → encrypt → decrypt → verify.
            if isinstance(self.model, PeftModel):
                with _adapter_slot_for_load(slot_path) as load_path:
                    self.model.load_adapter(str(load_path), adapter_name=verify_name)
            else:
                # Base model — cannot load a second adapter without wrapping.
                # This branch should not occur in production (the model is always
                # a PeftModel by the time _save_adapters is called), but guard
                # defensively to avoid a silent skip.
                logger.warning(
                    "_verify_saved_adapter_from_disk: model is not a PeftModel "
                    "— skipping disk verify for %s",
                    adapter_name,
                )
                return 1.0

            # Patch base_model_name_or_path when PEFT sets it to None for
            # second-and-later adapters (same pattern as create_adapter in loader.py).
            if self.model.peft_config[verify_name].base_model_name_or_path is None:
                base_name = getattr(self.model.get_base_model().config, "_name_or_path", None)
                if base_name:
                    self.model.peft_config[verify_name].base_model_name_or_path = base_name

            # Activate verify slot, probe, then restore original.
            switch_adapter(self.model, verify_name)
            recall_rate = self._run_recall_sanity_probe(
                verify_name,
                entries,
                max_probe=max_probe,
                debug_phase="disk_verify",
            )
            switch_adapter(self.model, adapter_name)

            logger.info(
                "_verify_saved_adapter_from_disk: %s slot=%s recall=%.3f threshold=%.3f",
                adapter_name,
                slot_path.name,
                recall_rate,
                threshold,
            )
        finally:
            # Always drop the verify slot — even if the probe raised.
            # Re-activate the original adapter so the model is left in the
            # same state as on entry regardless of which branch was taken.
            if verify_name in self.model.peft_config:
                try:
                    switch_adapter(self.model, adapter_name)
                except Exception:  # noqa: BLE001
                    pass
                self.model.delete_adapter(verify_name)
                logger.debug(
                    "_verify_saved_adapter_from_disk: verify slot '%s' dropped",
                    verify_name,
                )

        if recall_rate < threshold:
            raise RecallGateRejected(
                f"Post-save disk-integrity probe failed for adapter '{adapter_name}': "
                f"recall {recall_rate:.3f} < threshold {threshold:.2f} "
                f"(slot: {slot_path}). "
                "The on-disk artifact may be corrupt. "
                "Sessions will remain pending for retry on the next cycle.",
                recall_rate=recall_rate,
                threshold=threshold,
            )

        return recall_rate

    def _ensure_adapters(self):
        """Create production adapters that don't exist yet.

        Production adapters (episodic, semantic, procedural) are created based
        on configuration.  The staging slot (``in_training``) is NOT created
        here: per the staging+promote contract, the slot is transient and is
        created/destroyed per training event by
        ``trainer._ensure_staging_slot`` and the post-save cleanup at each
        save site.  Pre-creating it at startup would violate the
        "transient — exists only while a training event is in flight"
        invariant.
        """
        import shutil

        from peft import PeftModel

        from paramem.models.loader import create_adapter

        has_peft = isinstance(self.model, PeftModel)
        if not has_peft or "episodic" not in self.model.peft_config:
            logger.info("Creating episodic adapter")
            self.model = create_adapter(self.model, self.episodic_config, "episodic")
        if self.config.promotion_threshold > 0 and "semantic" not in self.model.peft_config:
            logger.info("Creating semantic adapter")
            self.model = create_adapter(self.model, self.semantic_config, "semantic")
        if self.procedural_config is not None and "procedural" not in self.model.peft_config:
            logger.info("Creating procedural adapter")
            self.model = create_adapter(self.model, self.procedural_config, "procedural")

        # Clean stale on-disk staging checkpoints (HF Trainer output_dir/in_training).
        # These are filesystem-level debris from a prior crash-resume attempt,
        # unrelated to the PEFT slot lifecycle.
        stale_dir = Path(self.output_dir) / "in_training"
        if stale_dir.exists():
            logger.info("Cleaning stale in_training checkpoints at %s", stale_dir)
            shutil.rmtree(stale_dir)

        return self.model

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
        is the ``_EarlyStopState`` shared with the callback; callers read
        ``state.last_per_key`` after ``_train_adapter`` returns to obtain the
        per-key recall verdict from the FINAL trained weights.

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
            callback). None of those three call this helper directly any
            more — they all reach it transitively via _train_tier_adapter.

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
                the fold's own ``_save_adapters`` external commit.  Default
                ``False`` preserves clean-on-success for all other callers (BG
                trainer, replay, migration, interim).

        Returns:
            ``(metrics_dict, recall_state)`` on success; ``(None, None)`` if
            ``entries`` yields no training examples. When donor seeding
            actually copied weights into *adapter_name* this fold,
            ``metrics_dict["donor_seeded"]`` is ``True`` — call sites use this
            (not a second measurement) to tag their telemetry ``init`` field
            ``"donor"`` instead of the pre-training ``"cold"`` measurement.

        Donor seeding: unconditional (no feature flag; validated via Test 20
        -- see ``benchmarking.md``). This method is reachable ONLY from the
        weights venue (every call site sits inside its enclosing
        ``if scope.source == "weights":`` branch — see ``consolidation.py``'s
        two ``_train_tier_adapter`` call sites and
        ``active_store_migration._migrate_tier_simulate_to_train``, plus
        :func:`~paramem.training.donor.build_donor`'s own funnel call
        (training the donor's transient build slot itself, gated out of
        recursive seeding below) -- FOUR call sites total, all routed
        through this one funnel), so the disk/simulate venue never seeds.
        When *adapter_name* is not the donor's own transient build
        slot (``DONOR_BUILD_ADAPTER_NAME`` — excluding it here is what stops
        :func:`~paramem.training.donor.build_donor`'s own funnel call from
        recursively re-triggering donor seeding on the adapter it is training):
        measure the target's LoRA-B Frobenius norm
        (:func:`~paramem.models.loader.measured_adapter_init_state`); on
        ``"cold"``, resolve the current base model id and *adapter_config*'s
        LoRA shape and check :func:`~paramem.training.donor.donor_checkpoint_valid`
        against that shape's own topology directory. A missing OR mismatched
        (base model OR topology) checkpoint is built fresh at *adapter_config*'s
        topology (for the CURRENT base/shape — never seeds cross-base or
        cross-topology) via :func:`~paramem.training.donor.build_donor`
        before this fold's own training — a cold interim is the failure this
        mechanism exists to fix. If the checkpoint is (now) valid, the donor
        is loaded into a transient slot
        (:func:`~paramem.training.donor.load_donor_into_transient_slot`) and
        copied into *adapter_name* via
        :func:`~paramem.models.loader.copy_adapter_weights` (the strict full
        copy — the donor and target now always share the SAME topology, so
        the parameter sets are equal by construction), and the transient
        slot is always deleted in a ``finally``. An unresolvable base id, a
        checkpoint that still fails to validate after the build attempt, or
        a build that could not complete this fold
        (:class:`~paramem.training.donor.DonorBuildIncomplete`), skips
        seeding and logs — never raises.
        """
        from paramem.training.trainer import train_adapter

        examples = format_entry_training(entries, self.tokenizer, max_length=1024)
        if not examples:
            return None, None

        donor_seeded = False
        if adapter_name != DONOR_BUILD_ADAPTER_NAME:
            donor_seeded = self._maybe_seed_from_donor(adapter_name, adapter_config)

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
        )
        metrics["donor_seeded"] = donor_seeded
        return metrics, recall_state

    def _maybe_seed_from_donor(self, adapter_name: str, adapter_config) -> bool:
        """Seed *adapter_name* from its topology's donor checkpoint when it
        measures cold.

        Helper for :meth:`_train_tier_adapter`'s donor-seeding gate — see that
        method's docstring for the full decision tree. Returns ``True`` only
        when weights were actually copied into *adapter_name* this call (so
        the caller can tag its telemetry ``init`` field ``"donor"``); returns
        ``False`` on every other branch (warm target, unresolvable base id,
        a checkpoint that still fails to validate after a build attempt, or a
        build attempt that could not complete this fold —
        :class:`~paramem.training.donor.DonorBuildIncomplete`, caught here
        specifically), without raising.

        Args:
            adapter_name: The adapter slot being seeded.
            adapter_config: *adapter_name*'s own ``AdapterConfig`` — the
                same object ``_train_tier_adapter`` was called with. Its
                shape (rank, alpha, target_modules) determines which
                topology's donor checkpoint is resolved and, when a build
                is needed, which topology :func:`~paramem.training.donor.build_donor`
                writes into.
        """
        from paramem.models.loader import _lora_shape_fields
        from paramem.training.donor import (
            DONOR_LOAD_ADAPTER_NAME,
            DonorBuildIncomplete,
            build_donor,
            donor_checkpoint_dir,
            donor_checkpoint_valid,
            load_donor_into_transient_slot,
        )

        init_state = measured_adapter_init_state(self.model, adapter_name)
        if init_state != "cold":
            return False

        base_model_id = getattr(self.model.get_base_model().config, "_name_or_path", None)
        if base_model_id is None:
            logger.warning(
                "_maybe_seed_from_donor: skipping donor seeding for %s -- base model id unresolved",
                adapter_name,
            )
            return False

        # The donor is built/validated at the TARGET tier's own topology --
        # comparing the CURRENT shape against the checkpoint's recorded
        # shape catches an operator rank/target-modules edit BEFORE
        # copy_adapter_weights would hit a tensor-shape mismatch and abort
        # the fold.
        lora_shape = _lora_shape_fields(adapter_config)
        checkpoint_dir = donor_checkpoint_dir(self.output_dir, lora_shape)
        if not donor_checkpoint_valid(checkpoint_dir, base_model_id, lora_shape):
            logger.info(
                "_maybe_seed_from_donor: donor checkpoint missing/mismatched "
                "for base %s -- building before this fold's training",
                base_model_id,
            )
            try:
                build_donor(self, adapter_config=adapter_config)
            except DonorBuildIncomplete as exc:
                logger.warning(
                    "_maybe_seed_from_donor: donor build did not complete this "
                    "fold (%s) -- skipping seeding for %s; the next "
                    "measured-cold fold will retry the build",
                    exc,
                    adapter_name,
                )
                return False

        if not donor_checkpoint_valid(checkpoint_dir, base_model_id, lora_shape):
            logger.warning(
                "_maybe_seed_from_donor: skipping donor seeding for %s -- "
                "no valid checkpoint after build attempt",
                adapter_name,
            )
            return False

        from paramem.models.loader import active_adapter_name, copy_adapter_weights

        try:
            load_donor_into_transient_slot(self.model, checkpoint_dir, DONOR_LOAD_ADAPTER_NAME)
            copy_adapter_weights(self.model, src=DONOR_LOAD_ADAPTER_NAME, dst=adapter_name)
            logger.info(
                "_maybe_seed_from_donor: seeded %s from donor checkpoint (base=%s)",
                adapter_name,
                base_model_id,
            )
            return True
        finally:
            if DONOR_LOAD_ADAPTER_NAME in self.model.peft_config:
                if active_adapter_name(self.model) == DONOR_LOAD_ADAPTER_NAME:
                    switch_adapter(self.model, adapter_name)
                self.model.delete_adapter(DONOR_LOAD_ADAPTER_NAME)
