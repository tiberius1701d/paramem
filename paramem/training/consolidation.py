"""Consolidation loop orchestrator.

Runs the full consolidation pipeline: extract graph from session,
merge into cumulative graph, score for promotion/decay, train
episodic and semantic adapters.
"""

import logging
import os
import random
import secrets
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import cached_property
from pathlib import Path
from typing import Callable, Literal, Optional

from torch.utils.data import Dataset

from paramem.graph.extraction_pipeline import ExtractionConfig, ExtractionPipeline
from paramem.graph.extractor import (
    PROVIDER_KEY_ENV,
    _graph_enrich_with_sota,
)
from paramem.graph.merger import GraphMerger
from paramem.graph.name_match import canonical
from paramem.graph.phase_trace import extraction_trace, phase_trace
from paramem.graph.qa_generator import (
    partition_relations,
)
from paramem.graph.reconstruct import reconstruct_graph
from paramem.graph.schema import Relation, SessionGraph
from paramem.graph.schema_config import fallback_relation_type, relation_types
from paramem.memory.entry import (
    assign_keys,
    build_registry,
    compute_simhash,
    format_entry_training,
)
from paramem.models.loader import atomic_save_adapter, switch_adapter
from paramem.server.vram_guard import safe_empty_cache
from paramem.training.curriculum import CurriculumSampler
from paramem.training.key_registry import KeyRegistry
from paramem.training.recall_eval import probe_entries
from paramem.training.replay import MixedReplayDataset, SyntheticQADataset
from paramem.training.thermal_throttle import ThermalPolicy
from paramem.training.trainer import TrainingHooks, train_adapter
from paramem.utils.config import (
    AdapterConfig,
    ConsolidationConfig,
    GraphConfig,
    TrainingConfig,
    WandbConfig,
)

logger = logging.getLogger(__name__)

# Frozen set of valid relation types drawn from the single source of truth in
# schema_config so that the stage-2 clamp stays in sync with the Pydantic
# Relation schema (_RelationType = Literal[relation_types()]).
_VALID_RTYPES: frozenset[str] = frozenset(relation_types())
_FALLBACK_RTYPE: str = fallback_relation_type()

# Synthetic session-id sentinels used by the three fold/re-merge paths that
# call _merge_registry_relations or GraphMerger.merge with a pseudo-id rather
# than a real session identifier.  The harvest filter in
# _build_all_edge_entries_into subtracts these from edge["sessions"] so that
# the deferred-write record carries ONLY real contributing session ids (B7-A).
_SYNTHETIC_SESSION_IDS: frozenset[str] = frozenset(
    {
        "__full_consolidation_recon__",
        "__interim_pending_sessions__",
        "__simulate_consolidation_merge__",
    }
)


@dataclass
class CycleResult:
    """Results from a single consolidation cycle."""

    cycle_index: int
    session_id: str
    entities_extracted: int = 0
    relations_extracted: int = 0
    nodes_retained: int = 0
    episodic_train_loss: Optional[float] = None
    semantic_train_loss: Optional[float] = None
    procedural_train_loss: Optional[float] = None
    episodic_recall: Optional[float] = None
    semantic_recall: Optional[float] = None
    wall_clock_seconds: float = 0.0


class TrialActiveError(RuntimeError):
    """Raised by ConsolidationLoop.guard_trial_state when a migration TRIAL is active.

    Bubbles up to /scheduled-tick and /consolidate handlers, which return
    409 trial_active.  Experiment scripts that do not carry server _state
    never trigger this error (guard is a no-op when state is None).
    """


class AbortedDuringConsolidation(Exception):
    """Raised by ``consolidate_interim_adapters`` when training is aborted mid-tier.

    The caller (app.py ``_run_full_cycle``) catches this, restores all three
    production tiers from their ``<tier>_backup`` slots via
    ``copy_adapter_weights``, skips the atomic finalize step (registry rewrite
    → persist → interim purge → router reload), and logs the cycle as
    ``mode="aborted"``.  Partial progress is lost but VRAM state is consistent
    with the pre-cycle baseline.
    """


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
        save_cycle_snapshots: bool = True,
        snapshot_dir: str | Path | None = None,
        run_id: str | None = None,
        prompts_dir: str | Path | None = None,
        model_name: str | None = None,
        extraction_stt_correction: bool = True,
        extraction_ha_validation: bool = True,
        extraction_noise_filter: str = "",
        extraction_noise_filter_model: str = "claude-sonnet-4-6",
        extraction_noise_filter_endpoint: str | None = None,
        extraction_ner_check: bool = False,
        extraction_ner_model: str = "en_core_web_sm",
        extraction_plausibility_judge: str = "auto",
        extraction_plausibility_stage: str = "deanon",
        extraction_verify_anonymization: bool = True,
        extraction_pii_scope: set[str] | frozenset[str] | None = None,
        graph_config: Optional[GraphConfig] = None,
        graph_enrichment_enabled: bool = True,
        graph_enrichment_neighborhood_hops: int = 2,
        graph_enrichment_max_entities_per_pass: int = 50,
        graph_enrichment_interim_enabled: bool = True,
        graph_enrichment_min_triples_floor: int = 20,
        state_provider=None,
        thermal_policy: ThermalPolicy | None = None,
        keep_prior_slots: int = 3,
    ):
        # Optional callable that returns the server ``_state`` dict.  When
        # provided, ``run_cycle`` calls ``self.guard_trial_state(state_provider())``
        # at entry to block new consolidation cycles during a migration TRIAL.
        # Experiment scripts pass nothing (default ``None``) so the guard is a
        # no-op and experiment paths are unaffected.
        self.state_provider = state_provider
        self._keep_prior_slots = keep_prior_slots
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
        # ``extract_procedural_graph``.  Owns the 15 SOTA-pipeline tunables
        # (temperature, max_tokens, anonymizer / noise_filter / plausibility /
        # NER flags, PII scope, etc.) sourced from the ``extraction_*``
        # ConsolidationLoop kwargs.  Every consolidation call site reaches the
        # extractors through ``self.extraction.run`` / ``run_procedural`` —
        # no direct ``extract_graph(...)`` calls in this module.
        #
        # Cloud egress PII anonymization scope (``extraction_pii_scope``) is
        # sourced at the bootstrap call site from
        # ``ServerConfig.sanitization.cloud_scope`` so consolidation honours
        # the same operator policy as inference-time cloud egress.  ``None``
        # falls back to the primitive default in ``extractor.py``
        # (``{person, place}``) for back-compat with experiment scripts.
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
                stt_correction=extraction_stt_correction,
                ha_validation=extraction_ha_validation,
                noise_filter=extraction_noise_filter,
                noise_filter_model=extraction_noise_filter_model,
                noise_filter_endpoint=extraction_noise_filter_endpoint,
                ner_check=extraction_ner_check,
                ner_model=extraction_ner_model,
                plausibility_judge=extraction_plausibility_judge,
                plausibility_stage=extraction_plausibility_stage,
                verify_anonymization=extraction_verify_anonymization,
                pii_scope=extraction_pii_scope,
            ),
            prompts_dir=prompts_dir,
            model_name=model_name,
        )

        # Graph-level SOTA enrichment knobs (Task #10).
        self.graph_enrichment_enabled = graph_enrichment_enabled
        self.graph_enrichment_neighborhood_hops = graph_enrichment_neighborhood_hops
        self.graph_enrichment_max_entities_per_pass = graph_enrichment_max_entities_per_pass
        # Interim mini-enrichment: fires at sub-interval rollover when enough
        # new triples have accumulated. RAM-only counter, reset after each
        # successful enrichment pass (full or interim).
        self.graph_enrichment_interim_enabled = graph_enrichment_interim_enabled
        self.graph_enrichment_min_triples_floor = graph_enrichment_min_triples_floor
        self._triples_since_last_enrichment = 0

        gc = graph_config or GraphConfig()
        self.merger = GraphMerger(
            similarity_threshold=gc.entity_similarity_threshold,
            cross_predicate_contradiction=gc.cross_predicate_contradiction,
            prompts_dir=self.prompts_dir,
        )
        self.last_session_graph = None

        # The cumulative graph is NOT loaded at construction.  The fold
        # (consolidate_interim_adapters) calls merger.reset_graph() before
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

        # Replay pools: track relations available for replay per adapter
        # Each entry: {"question": str, "answer": str}
        self.episodic_replay_pool: list[dict] = []
        self.semantic_replay_pool: list[dict] = []

        # Curriculum-aware replay sampling
        self.curriculum_sampler: Optional[CurriculumSampler] = None
        if self.config.curriculum_enabled:
            self.curriculum_sampler = CurriculumSampler(
                min_exposure_cycles=self.config.min_exposure_cycles,
            )

        # Per-tier indexed-key memory store — injected by the caller.
        # Lifespan-owned in production; experiments construct + hydrate
        # their own and pass it in.  The store is the single source of
        # truth for {entry, simhash, registry} of every indexed key.
        self.store = memory_store
        self._indexed_next_index: int = 1
        self._procedural_next_index: int = 1
        # Derive next-index counters from all active keys in the store.
        # The caller is responsible for having hydrated registries before
        # this point (via ``MemoryStore.load_registries_from_disk`` or by
        # injecting the lifespan-loaded store).
        if self.store.replay_enabled:
            for key in self.store.all_active_keys():
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

        # RAM-only queue for max_interim_count==0 (queue-until-consolidation).
        # Facts extracted when no interim adapter is configured accumulate here
        # until the next full consolidation fold (consolidate_interim_adapters)
        # folds them into the mains.
        # Privacy invariant: this list is never written to disk — what isn't
        # trained doesn't exist.  Snapshot persistence is deferred to the fold.
        # TODO: consume pending_interim_triples at the start of
        # consolidate_interim_adapters() before training the full key set.
        self.pending_interim_triples: list[dict] = []
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

        Per-key ``speaker_id`` / ``first_seen_cycle`` are owned by
        :attr:`MemoryStore._bookkeeping` and loaded by
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
        first_seen_cycle: int,
        relation_type: str = "factual",
        question: Optional[str] = None,
        answer: Optional[str] = None,
    ) -> dict:
        """Build a uniform ``indexed_key_cache`` cache entry.

        Carries ``subject``/``predicate``/``object`` as the canonical triple
        fields.  Optional ``question`` and ``answer`` are kept for callers that
        carry them (e.g. boot-time seeders reading legacy disk shapes).

        Using this helper for every cache-write site ensures the uniform shape
        is maintained by construction — every downstream reader (promotion-match,
        ``consolidate_interim_adapters`` triple-lookup) reads the canonical field
        names.

        Args:
            key: The ``graphN`` / ``procN`` key string.
            subject: Triple subject.
            predicate: Triple predicate.
            object: Triple object.
            speaker_id: Speaker scope.
            first_seen_cycle: Cycle count at first insertion.
            relation_type: Model-assigned relation type from extraction
                (e.g. ``"factual"``, ``"preference"``, ``"temporal"``,
                ``"social"``).  Defaults to ``"factual"`` for legacy callers
                that pre-date this field; pass explicitly at every new site.
            question: Optional legacy question text; omit for entry-format keys.
            answer: Optional legacy answer text; omit for entry-format keys.

        Returns:
            Dict with the canonical cache shape.
        """
        entry: dict = {
            "key": key,
            "subject": subject,
            "predicate": predicate,
            "object": object,
            "speaker_id": speaker_id,
            "first_seen_cycle": first_seen_cycle,
            "relation_type": relation_type,
        }
        if question is not None:
            entry["question"] = question
        if answer is not None:
            entry["answer"] = answer
        return entry

    # ------------------------------------------------------------------
    # DEPRECATED — transitional accessors for the legacy attribute names
    # ------------------------------------------------------------------
    # The 2026-05-14 MemoryStore cleanup moved {entries, simhashes, registry}
    # off the loop and into ``self.store``.  Production code is fully
    # migrated.  These accessors exist ONLY to keep the test surface green
    # while we migrate the ~250 test sites that still reach in by name —
    # they are NOT part of the supported API and MUST be deleted once
    # tests are migrated.
    #
    # Read-write coupling: indexed_key_cache returns the *live* internal dict
    # via the store, so existing tests that mutate it
    # (``loop.indexed_key_cache[k] = v``) propagate to the store.
    # ``indexed_key_registry`` returns the store's internal dict for the same
    # reason.
    #
    # TODO(arch-cleanup): delete indexed_key_cache + indexed_key_registry
    # (+ the two _*-aliases) once all remaining test sites are migrated.
    # The episodic_simhash/semantic_simhash/procedural_simhash trio was deleted
    # 2026-06-13; call sites now use store.replace_simhashes_in_tier directly.

    def _ensure_store(self) -> None:
        """Auto-create a :class:`MemoryStore` for bare-loop test instances.

        Some tests construct ``ConsolidationLoop`` via ``object.__new__``
        without invoking ``__init__``, then assign attributes one by one.
        The deprecated setters tolerate that pattern by lazily attaching
        a store with replay enabled.  Production code always goes through
        ``__init__`` and never reaches this path."""
        if not hasattr(self, "store") or self.store is None:
            from paramem.memory.store import MemoryStore

            self.store = MemoryStore(replay_enabled=True)

    @property
    def indexed_key_cache(self) -> dict[str, dict]:
        """DEPRECATED: flat-across-tier view of the store's entries.

        Mutations through ``loop.indexed_key_cache[key] = entry`` route to
        the most-recently-touched tier or default to ``"episodic"`` — they
        do NOT preserve tier semantics.  New code uses
        :meth:`MemoryStore.put` / :meth:`MemoryStore.get` directly.
        """
        from paramem.memory.store import _LegacyFlatCacheView as _LV

        self._ensure_store()
        return _LV(self.store)

    @indexed_key_cache.setter
    def indexed_key_cache(self, value: dict[str, dict]) -> None:
        """DEPRECATED: replace the entire cache (test-only reset path)."""
        self._ensure_store()
        self.store._entries = {"_legacy": dict(value)} if value else {}

    @property
    def indexed_key_registry(self) -> "dict[str, KeyRegistry] | None":
        """DEPRECATED: returns the store's internal registry dict, or ``None``
        when replay is disabled — preserves the legacy None-gate semantics."""
        self._ensure_store()
        return self.store._registry

    @indexed_key_registry.setter
    def indexed_key_registry(self, value: "dict[str, KeyRegistry] | None") -> None:
        self._ensure_store()
        self.store._registry = value
        # Reflect the replay flag so MemoryStore.replay_enabled stays honest.
        self.store._replay_enabled = value is not None

    # ------------------------------------------------------------------
    # Per-tier registry helpers
    # ------------------------------------------------------------------

    def _tier_registry(self, tier: str) -> KeyRegistry:
        """Return the per-tier registry, creating it lazily.

        Raises ``RuntimeError`` when replay is disabled.  Thin passthrough to
        :meth:`MemoryStore.registry` — kept on the loop because many internal
        sites call it with a stable tier name.
        """
        if not self.store.replay_enabled:
            raise RuntimeError("indexed_key_replay not enabled")
        return self.store.registry(tier)

    def _all_active_keys(self) -> list[str]:
        """Every active key across every registered tier — order is tier-then-insertion."""
        return self.store.all_active_keys()

    def _safe_kp_from_cache(self, key: str) -> dict | None:
        """Return a training-ready keyed-pair from the in-RAM cache, or ``None``.

        Used by the existing-key reconstruction loops as a last-resort
        fallback when ``probe_entries`` / ``probe_key`` fails (typically when
        the adapter isn't mountable due to a hash mismatch).

        The boot-time ``seed_key_metadata`` populates the store with
        partial entries (``speaker_id`` + ``first_seen_cycle`` only — no
        subject/predicate/object).  Treating those entries as usable
        crashes the cycle with ``KeyError: 'subject'``.  This helper
        returns ``None`` for partial entries so the caller can skip them
        cleanly.
        """
        qa = self.store.get(key)
        if qa is None:
            return None
        if not all(f in qa for f in ("subject", "predicate", "object")):
            return None
        return {
            "key": key,
            "subject": qa["subject"],
            "predicate": qa["predicate"],
            "object": qa["object"],
        }
        if not all(f in qa for f in ("question", "answer")):
            return None
        return {"key": key, "question": qa["question"], "answer": qa["answer"]}

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

        B1 (soft-stale preservation): the fresh ``KeyRegistry()`` that replaces the
        live registry would wipe any stale flip applied during the drift-partition
        step.  Pass ``soft_stale_by_tier`` so the rebuilt registry seeds the stale
        partition BEFORE adding passing (active) keys.  Stale simhashes are also
        merged back into the rebuilt simhash dict so they survive on disk for the
        stale-echo seam.

        Args:
            tier_keyed: Per-tier keyed-entry lists (full post-consolidation set).
            passing_sets_by_tier: Per-tier sets of keys that passed the recall
                gate.  A ``None`` entry for a tier triggers the probe fallback.
                Pass ``None`` for the entire dict to skip recall gating (legacy /
                non-production caller).
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
            # (a) seed stale records FIRST (B1 — must survive the rebuild);
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

        Mirrors the partition-only half of ``generate_qa_from_graph`` but
        produces relation dicts instead of relations.  The attribute surface
        (``Entity.attributes``) is projected via
        ``relation_prep._flatten_entity_attributes`` so scalar-PII keying
        (email/phone/linkedin) is not silently dropped.

        Returns:
            ``(episodic_relations, procedural_relations)`` — both are lists of
            relation dicts suitable for ``assign_keys``.

        Note:
            This method has no ``model.generate`` calls — the vram_scope
            wrapping present on the QA-gen path can be omitted here, though a
            trailing ``torch.cuda.empty_cache()`` at the call site is still
            recommended for allocator hygiene on multi-session cycles.
        """
        from paramem.graph import relation_prep

        relation_dicts = [
            {
                "subject": r.subject,
                "predicate": r.predicate,
                "object": r.object,
                "relation_type": r.relation_type,
            }
            for r in session_graph.relations
        ]
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
        if self._debug_base is None or not self.save_cycle_snapshots:
            return None
        parts: list[str] = ["episodic"]
        if interim_stamp:
            parts.append(f"interim_{interim_stamp}")
        parts.append(f"cycle_{self.cycle_count}")
        parts.append(f"run_{self.run_id}")
        return self._debug_base.joinpath(*parts)

    def _current_interim_stamp_or_none(self) -> str | None:
        """Convenience accessor for the active interim stamp, if any."""
        return getattr(self, "_current_interim_stamp", None)

    @cached_property
    def _debug_writer(self):
        """Single owner for every plaintext write under ``paths.debug``.

        Self-gates on ``save_cycle_snapshots`` and ``_debug_base`` so callers
        never check.  Constructed lazily so test fixtures that bypass
        ``__init__`` (``object.__new__(ConsolidationLoop)``) still have a
        functioning writer without explicit wiring.
        """
        from paramem.training.debug_snapshot import DebugSnapshotWriter

        return DebugSnapshotWriter(self)

    def extract_session(
        self,
        session_transcript: str,
        session_id: str,
        speaker_id: str,
        speaker_name: str | None = None,
        ha_context: dict | None = None,
        stt_correction: bool | None = None,
        ha_validation: bool | None = None,
        noise_filter: str | None = None,
        noise_filter_model: str | None = None,
        noise_filter_endpoint: str | None = None,
        ner_check: bool | None = None,
        ner_model: str | None = None,
        plausibility_judge: str | None = None,
        plausibility_stage: str | None = None,
        verify_anonymization: bool | None = None,
        source_type: str = "transcript",
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
        """
        logger.info("=== Extraction (session=%s) ===", session_id)

        # Outer extraction_trace scope wraps the whole session body so the
        # orchestrator phases (merge_into_cumulative, procedural_extract,
        # dedup_*) record into the same trace as the inner extract_graph /
        # extract_procedural_graph calls — those traces nest-no-op into this
        # one.  The final attach_to(...) calls below capture the complete
        # phase history on each session graph before it is dumped.
        with extraction_trace() as trace:
            # --- EXTRACT ---
            session_graph = self.extraction.run(
                session_transcript,
                session_id,
                source_type=source_type,
                ha_context=ha_context,
                stt_correction=stt_correction,
                ha_validation=ha_validation,
                noise_filter=noise_filter,
                noise_filter_model=noise_filter_model,
                noise_filter_endpoint=noise_filter_endpoint,
                speaker_name=speaker_name,
                speaker_id=speaker_id,
                ner_check=ner_check,
                ner_model=ner_model,
                plausibility_judge=plausibility_judge,
                plausibility_stage=plausibility_stage,
                verify_anonymization=verify_anonymization,
            )

            logger.info(
                "Extracted %d entities, %d relations",
                len(session_graph.entities),
                len(session_graph.relations),
            )

            # --- MERGE ---
            with phase_trace("merge_into_cumulative") as t:
                if self.config.interim_refinement != "off":
                    # Path A: merge into cumulative graph each interim cycle.
                    # Disable gradient checkpointing: merger.merge may call model.generate()
                    # for contradiction resolution when a model is present.  HF silently
                    # disables the KV cache when checkpointing is active (CLAUDE.md rule).
                    self._disable_gradient_checkpointing()
                    try:
                        self.merger.merge(session_graph)
                    finally:
                        self._enable_gradient_checkpointing()
                    self._triples_since_last_enrichment += len(session_graph.relations)
                    t.add("triples_added", len(session_graph.relations))
                else:
                    # Path B: merge deferred to full consolidation (interim_refinement="off").
                    # The counter is NOT incremented — interim enrichment consumes
                    # _triples_since_last_enrichment over merger.graph, which was not
                    # updated; enrichment stays Path-B-only by design.
                    t.add("triples_added", 0)

            # --- BUILD ENTRY RELATION DICTS ---
            # Single entry point for graph → entries.  Builds relation dicts
            # directly from session_graph with no model.generate calls.
            episodic_rels, procedural_rels = self._entries_from_graph(
                session_graph,
                procedural_enabled=self.procedural_config is not None,
            )

            # --- PROCEDURAL: separate extraction pass ---
            proc_graph: SessionGraph | None = None
            if self.procedural_config is not None:
                with phase_trace("procedural_extract") as t:
                    proc_graph = self.extraction.run_procedural(
                        session_transcript,
                        session_id,
                        speaker_name=speaker_name,
                        stt_correction=stt_correction,
                        source_type=source_type,
                        speaker_id=speaker_id,
                    )
                    t.add("relation_count", len(proc_graph.relations))
                procedural_rels.extend(
                    {
                        "subject": r.subject,
                        "predicate": r.predicate,
                        "object": r.object,
                        "relation_type": r.relation_type,
                    }
                    for r in proc_graph.relations
                )

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
            self._debug_writer.on_session_extracted(session_graph, session_id, "graph")
            if proc_graph is not None:
                trace.attach_to(proc_graph)
                self._debug_writer.on_session_extracted(proc_graph, session_id, "procedural_graph")

        self.last_session_graph = session_graph

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
        as the production post-session training hook.  After the cycle, calls
        :meth:`consolidate_interim_adapters` to fold the freshly-trained interim
        slot into the main ``"episodic"`` adapter so callers that probe
        ``model.set_adapter("episodic")`` read the trained weights, not the
        stale main slot.  This mirrors production's full-cycle
        consolidate-interim-adapters step, compressed for the one-shot
        experiment use case.

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
            new_promotions=None,
        )

        # --- Roll interim slot into main ---
        # run_consolidation_cycle trains into episodic_interim_<stamp>.  Callers
        # that probe model.set_adapter("episodic") need the trained weights in the
        # main slot.  Submit consolidate_interim_adapters via an ephemeral
        # BackgroundTrainer so the GPU lock is held for the full per-tier rebuild
        # (consolidate_interim_adapters requires this — see its entry guard).
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
                self.consolidate_interim_adapters(trainer=_bt)

            try:
                _bt.submit_and_wait(_consolidate)
                _folded = True
            finally:
                _bt.close()

        # --- SAVE main slots ---
        # consolidate_interim_adapters now persists+verifies the merged main
        # weights itself (between its registry rewrite and interim purge), so a
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
            "procedural_train_loss": cycle_result.get("procedural_train_loss"),
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
        max_interim_count: int,
        *,
        mode: "Literal['simulate', 'train']",
    ) -> "tuple[str, bool, bool, bool]":
        """Compute the target interim adapter name and control-flow flags.

        Returns a 4-tuple ``(adapter_name, cap_reached_absorb, queue_only,
        degenerated_skip)``:

        - ``adapter_name``: The PEFT adapter name for this sub-interval
          (``episodic_interim_<stamp>``).
        - ``cap_reached_absorb``: When ``True``, the cap was reached and the
          new facts should be absorbed into the newest existing interim adapter
          via a full retrain.  **Train mode only** — always ``False`` in
          simulate mode (no PEFT slots to count).
        - ``queue_only``: When ``True``, the caller passed
          ``max_interim_count == 0`` and facts should be queued without
          creating an adapter.  Mode-agnostic.
        - ``degenerated_skip``: When ``True``, the newest interim adapter is
          marked degenerated and new facts should be queued.  **Train mode
          only** — always ``False`` in simulate mode.

        In simulate mode the slot is always freshly minted (no PEFT config to
        consult) so ``cap_reached_absorb`` and ``degenerated_skip`` are both
        ``False``.

        Args:
            stamp: The sub-interval stamp (``YYYYMMDDTHHMM``).
            max_interim_count: Cap on concurrent interim adapters.
                ``0`` → queue-only branch.
            mode: ``"train"`` or ``"simulate"``.

        Returns:
            ``(adapter_name, cap_reached_absorb, queue_only, degenerated_skip)``
        """
        adapter_name = f"episodic_interim_{stamp}"

        if max_interim_count == 0:
            return adapter_name, False, True, False

        if mode == "simulate":
            # Simulate never touches PEFT config — fresh slot by definition.
            return adapter_name, False, False, False

        # Train mode: check PEFT config for cap-reached / degenerated conditions.
        if adapter_name not in self.model.peft_config:
            existing_interim = sorted(
                a for a in self.model.peft_config if a.startswith("episodic_interim_")
            )
            if self.store.replay_enabled and len(existing_interim) >= max_interim_count:
                newest = existing_interim[-1]
                _newest_reg = (
                    self.store.registry(newest) if self.store.has_registry(newest) else None
                )
                if _newest_reg is None or not _newest_reg.is_healthy():
                    return adapter_name, False, False, True
                # Cap reached and newest is healthy — absorb into newest.
                return newest, True, False, False

        return adapter_name, False, False, False

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
                first_seen_cycle=self.cycle_count,
                relation_type=rel.get("relation_type", "factual"),
            )
            if tag_new:
                entry["_new"] = True
            minted.append(entry)
        return minted

    def _prepare_episodic_keys_for_tier(
        self,
        adapter_name: str,
        rels: list[dict],
        speaker_id: str,
        *,
        mode: "Literal['simulate', 'train']",
    ) -> list[dict]:
        """Assign new keys and gather existing keys for the target interim tier.

        Unified replacement for ``_simulate_indexed_key_episodic``,
        ``_run_indexed_key_episodic`` (key-prep body), and the
        ``_train_extracted_into_interim`` key-prep block (lines 2864-2901 of
        the original file).

        Tier scope: ``self.store.active_keys_in_tier(adapter_name)`` — per-tier
        only.  Cross-tier scoping (``_all_active_keys()``) is NOT used here;
        each interim slot manages its own membership.

        First-seen preservation: existing entries retain their original
        ``first_seen_cycle`` rather than being stamped with the current cycle.

        Speaker-ID default: each new keyed pair receives
        ``kp.get("speaker_id", speaker_id)`` — the caller's id, never ``""``.

        Reconstruction policy (TRAIN mode only):
            Calls ``probe_entries`` on each existing key and uses the adapter
            weights as the authoritative source.  Falls back to
            ``_safe_kp_from_cache`` when probe fails.

        Simulate policy:
            Reads existing entries from ``self.store.get(key)`` — no
            ``model.generate`` calls.

        DOES NOT train.  DOES NOT update simhash registries.  DOES NOT call
        ``store.put`` or advance ``self._indexed_next_index`` — callers own that
        mutation and must apply it at the correct point in their failure-safety
        boundary (after ``train_adapter`` for the interim path, immediately for
        the full-train paths where failure rolls back the whole cycle).

        New entries are returned with ``"_new": True`` so callers can identify
        them for deferred store.put / index-counter advancement.  Existing
        entries never carry ``"_new"``.

        Args:
            adapter_name: Target tier name (e.g. ``"episodic_interim_YYYYMMDDTHHMM"``
                or ``"episodic"``).  Determines which per-tier key set is used for
                existing-key reconstruction.
            rels: Pre-extracted episodic relation dicts, already tagged with
                ``speaker_id`` by ``_tag_speaker_id_defaults``.
            speaker_id: Fallback speaker tag for new keys missing one.
            mode: ``"train"`` probes adapter weights; ``"simulate"`` reads cache.

        Returns:
            Full keyed-pair list (new + existing) ready for training (train) or
            registry rebuild (simulate).  Each entry has
            ``{key, subject, predicate, object, speaker_id, first_seen_cycle}``.
            New entries additionally carry ``"_new": True``.
        """
        # Assign new keys from the incoming relations via the shared minting helper.
        # store.put and _indexed_next_index advancement are intentionally deferred to
        # the caller — see _mint_keyed_entries docstring and this method's docstring.
        new_keyed = self._mint_keyed_entries(
            rels,
            prefix="graph",
            start_index=self._indexed_next_index,
            speaker_id=speaker_id,
            tag_new=True,
        )

        # Gather existing keys for the target tier (full-replay anti-forgetting).
        new_key_set = {kp["key"] for kp in new_keyed}
        existing_tier_keys = [
            k for k in self.store.active_keys_in_tier(adapter_name) if k not in new_key_set
        ]

        existing_keyed: list[dict] = []
        if mode == "train":
            # TRAIN: reconstruct from adapter weights.
            # Keys with no simhash fingerprint cannot be verified by probe_entries
            # (no reference hash to compute confidence against) — fall back to
            # cache immediately for those keys and only probe the rest.
            self._disable_gradient_checkpointing()
            tier_simhash = self.store.tier_simhashes(adapter_name, include_stale=False)
            keys_with_hash = [k for k in existing_tier_keys if k in tier_simhash]
            keys_without_hash = [k for k in existing_tier_keys if k not in tier_simhash]
            reconstructed: dict[str, dict] = {}
            entries = [{"key": k} for k in keys_with_hash]
            for entry, recalled in probe_entries(
                self.model,
                self.tokenizer,
                entries,
                registry=tier_simhash,
                batch_size=self.training_config.recall_probe_batch_size,
                confidence_threshold=0.5,
            ):
                key = entry["key"]
                if recalled is not None and "failure_reason" not in recalled:
                    bk = self.store.bookkeeping_for_key(key) or {}
                    first_seen = bk.get("first_seen_cycle", self.cycle_count)
                    reconstructed[key] = {
                        "key": key,
                        "subject": recalled["subject"],
                        "predicate": recalled["predicate"],
                        "object": recalled["object"],
                        "speaker_id": bk.get("speaker_id", speaker_id),
                        "first_seen_cycle": first_seen,
                    }
            logger.info(
                "_prepare_episodic_keys_for_tier: reconstruction %d/%d (adapter=%s)",
                len(reconstructed),
                len(keys_with_hash),
                adapter_name,
            )
            for key in keys_with_hash:
                if key in reconstructed:
                    existing_keyed.append(reconstructed[key])
                    continue
                kp = self._safe_kp_from_cache(key)
                if kp is not None:
                    existing_keyed.append(kp)
            for key in keys_without_hash:
                # No simhash — cannot probe; use cache as ground truth.
                kp = self._safe_kp_from_cache(key)
                if kp is not None:
                    existing_keyed.append(kp)
        else:
            # SIMULATE: read SPO from cache; bookkeeping from _bookkeeping.
            for key in existing_tier_keys:
                qa = self.store.get(key)
                if qa is not None:
                    bk = self.store.bookkeeping_for_key(key) or {}
                    first_seen = bk.get("first_seen_cycle", self.cycle_count)
                    existing_keyed.append(
                        {
                            "key": key,
                            "subject": qa["subject"],
                            "predicate": qa["predicate"],
                            "object": qa["object"],
                            "speaker_id": bk.get("speaker_id", speaker_id),
                            "first_seen_cycle": first_seen,
                        }
                    )

        return existing_keyed + new_keyed

    def _prepare_procedural_keys_for_tier(
        self,
        rels: list[dict],
        speaker_id: str,
        *,
        mode: "Literal['simulate', 'train']",
    ) -> tuple[list[dict], list[dict]]:
        """Assign new procedural keys and gather existing keys for retraining.

        Symmetric counterpart of :meth:`_prepare_episodic_keys_for_tier` for
        the procedural tier.  Always targets the stable ``"procedural"`` main
        adapter — there is no interim tier for procedural.

        Per-session contradiction retirement via ``sp_index`` has been removed.
        Procedural contradiction is now resolved at full consolidation by the
        model-bearing :class:`~paramem.graph.merger.GraphMerger`.  Duplicate
        procedural keys from within a session are tolerated in the interim and
        collapsed at the next full-consolidation cycle.

        Deferred-mutation discipline (TRAIN mode):
            Store mutations are applied tentatively before the train call in
            :meth:`run_consolidation_cycle`.  The index counter
            (``_procedural_next_index``) is advanced only after this helper
            returns successfully.

        Simulate mode:
            Writes the store immediately (no train call can fail so no
            deferral is needed).

        Args:
            rels: Pre-extracted procedural relation dicts, already tagged with
                ``speaker_id`` by ``_tag_speaker_id_defaults``.
            speaker_id: Fallback speaker tag for new keys missing one.
            mode: ``"train"`` or ``"simulate"``.

        Returns:
            ``(new_keyed, existing_keyed)`` as a 2-tuple.

            - ``new_keyed``: cache-entry dicts for the new procedural keys.
            - ``existing_keyed``: reconstructed / cache-read dicts for existing
              procedural keys.
        """
        if not rels:
            return [], []

        logger.info("_prepare_procedural_keys_for_tier: %d relations (mode=%s)", len(rels), mode)

        tentative_next = self._procedural_next_index
        # Mint new keys via the shared helper; tag_new=False because the procedural
        # TRAIN path does not use the _new sentinel (deferred mutations are tracked
        # via _procedural_tentative_next_index on self instead).
        new_keyed = self._mint_keyed_entries(
            rels,
            prefix="proc",
            start_index=tentative_next,
            speaker_id=speaker_id,
            tag_new=False,
        )
        tentative_next += len(new_keyed)

        new_key_set = {kp["key"] for kp in new_keyed}

        if mode == "simulate":
            # Apply mutations immediately — no train call that can fail.
            for kp in new_keyed:
                fingerprint = compute_simhash(
                    kp["key"], kp["subject"], kp["predicate"], kp["object"]
                )
                self.store.put("procedural", kp["key"], kp, simhash=fingerprint)
                self.store.set_bookkeeping(
                    kp["key"],
                    speaker_id=kp["speaker_id"],
                    first_seen_cycle=kp["first_seen_cycle"],
                    relation_type=kp.get("relation_type", "factual"),
                    recurrence_count=1,
                    last_seen_cycle=self.cycle_count,
                )
            self._procedural_next_index = tentative_next
            return new_keyed, []

        # TRAIN mode: gather existing keys for reconstruction (deferred mutations).
        # Use active-only fingerprints — stale keys must not be fed into reconstruction
        # (they are superseded facts). The include_stale=False call makes the
        # active-vs-known distinction explicit and prevents the enumeration-set bug.
        existing_proc_keys = [
            k
            for k in self.store.tier_simhashes("procedural", include_stale=False)
            if k not in new_key_set
        ]

        self._disable_gradient_checkpointing()
        proc_simhash = self.store.tier_simhashes("procedural", include_stale=False)
        reconstructed: dict[str, dict] = {}
        entries = [{"key": k} for k in existing_proc_keys]
        for entry, recalled in probe_entries(
            self.model,
            self.tokenizer,
            entries,
            registry=proc_simhash,
            batch_size=self.training_config.recall_probe_batch_size,
            confidence_threshold=0.5,
        ):
            key = entry["key"]
            if recalled is not None and "failure_reason" not in recalled:
                bk = self.store.bookkeeping_for_key(key) or {}
                reconstructed[key] = {
                    "key": key,
                    "subject": recalled["subject"],
                    "predicate": recalled["predicate"],
                    "object": recalled["object"],
                    "speaker_id": bk.get("speaker_id", speaker_id),
                    "first_seen_cycle": bk.get("first_seen_cycle", self.cycle_count),
                }

        logger.info(
            "_prepare_procedural_keys_for_tier: reconstruction %d/%d",
            len(reconstructed),
            len(existing_proc_keys),
        )

        existing_keyed: list[dict] = []
        for key in existing_proc_keys:
            if key in reconstructed:
                existing_keyed.append(reconstructed[key])
                continue
            kp = self._safe_kp_from_cache(key)
            if kp is not None:
                existing_keyed.append(kp)

        # Commit the counter ONLY if we return successfully (train path
        # does not advance the counter here — it is advanced in
        # run_consolidation_cycle after train_adapter returns).
        # Store tentative_next on self so the caller can commit it.
        self._procedural_tentative_next_index = tentative_next

        return new_keyed, existing_keyed

    def finalize_training(self) -> None:
        """Save adapters and registries after background training completes."""
        self._save_adapters()
        logger.info("Training finalized — adapters and registries saved")

    def run_cycle(
        self,
        session_transcript: str,
        session_id: str,
        speaker_id: str,
        speaker_name: str | None = None,
        source_type: str = "transcript",
    ) -> CycleResult:
        """Run one consolidation cycle for a new session.

        Legacy method — extracts and trains in one pass.
        Used by experiment scripts. Server uses extract_session + train_adapters.

        Raises ``TrialActiveError`` before any extraction when a migration TRIAL
        is active (gated via the injected ``state_provider``; experiment scripts
        that do not set ``state_provider`` are unaffected).

        Args:
            session_transcript: The raw session transcript text.
            session_id: Unique identifier for this session.
            speaker_id: Speaker identifier for preference scoping. Required —
                callers must always supply a real speaker ID.
            source_type: ``"transcript"`` (default) or ``"document"``. Passed
                through to the extractor to select the appropriate system prompt.

        Returns:
            CycleResult with metrics and timing.
        """
        # Guard: block new cycles during a migration TRIAL.
        if self.state_provider is not None:
            self.guard_trial_state(self.state_provider())

        start_time = time.time()
        self.cycle_count += 1
        result = CycleResult(cycle_index=self.cycle_count, session_id=session_id)

        logger.info(
            "=== Consolidation cycle %d (session=%s) ===",
            self.cycle_count,
            session_id,
        )

        # --- 1. EXTRACT ---
        # Unified extraction path: every consolidation site reaches the
        # extractors through ``self.extraction`` so every SOTA pipeline flag
        # configured on the loop is applied identically.
        session_graph = self.extraction.run(
            session_transcript,
            session_id,
            source_type=source_type,
            speaker_name=speaker_name,
            speaker_id=speaker_id,
        )
        self._debug_writer.on_session_extracted(session_graph, session_id, "graph")

        result.entities_extracted = len(session_graph.entities)
        result.relations_extracted = len(session_graph.relations)

        # --- 2. MERGE ---
        if self.config.interim_refinement != "off":
            # Path A: merge into cumulative graph each run_cycle call.
            # Disable gradient checkpointing: merger.merge may call model.generate()
            # for contradiction resolution when a model is present.  HF silently
            # disables the KV cache when checkpointing is active (CLAUDE.md rule).
            self._disable_gradient_checkpointing()
            try:
                self.merger.merge(session_graph)
            finally:
                self._enable_gradient_checkpointing()

        # --- 4. BUILD ENTRY RELATION DICTS ---
        # Single entry point: graph → (episodic_rels, procedural_rels).
        # Builds relation dicts directly via _entries_from_graph — no
        # model.generate calls.
        episodic_rels, procedural_rels = self._entries_from_graph(
            session_graph,
            procedural_enabled=self.procedural_config is not None,
        )

        # Mirror extract_session(): run the dedicated procedural prompt so
        # experiments exercise the same pipeline as production.
        if self.procedural_config is not None:
            proc_graph = self.extraction.run_procedural(
                session_transcript,
                session_id,
                speaker_name=speaker_name,
                source_type=source_type,
                speaker_id=speaker_id,
            )
            self._debug_writer.on_session_extracted(proc_graph, session_id, "procedural_graph")
            procedural_rels.extend(
                {
                    "subject": r.subject,
                    "predicate": r.predicate,
                    "object": r.object,
                    "relation_type": r.relation_type,
                }
                for r in proc_graph.relations
            )

        # Apply same dedup as server path (identical policy across all paths).
        episodic_rels = self.dedup_episodic(episodic_rels)
        procedural_rels = self.dedup_procedural(procedural_rels)

        # --- 4b. INDEXED KEY REPLAY (F4.9c validated) ---
        # Delegate to run_consolidation_cycle (unified episodic + procedural
        # pipeline) so experiments exercise the same code path as production.
        # cycle_count was already incremented at the top of this method; pass
        # stamp=None so run_consolidation_cycle computes a fresh sub-interval
        # stamp and does NOT double-increment (it increments internally).
        # Caller undoes the pre-increment so run_consolidation_cycle owns the
        # authoritative count for this cycle.
        if self.store.replay_enabled:
            # Undo the cycle_count increment done at the top of run_cycle so
            # run_consolidation_cycle's own increment lands on the correct value.
            self.cycle_count -= 1
            cycle_result = self.run_consolidation_cycle(
                episodic_rels,
                procedural_rels,
                speaker_id=speaker_id,
                mode="train",
                run_label=session_id,
                new_promotions=None,
            )
            # cycle_count is now managed by run_consolidation_cycle.
            # Propagate per-tier train losses from the cycle result dict.
            _epi_loss = cycle_result.get("episodic_train_loss")
            _proc_loss = cycle_result.get("procedural_train_loss")
            if _epi_loss is not None:
                result.episodic_train_loss = _epi_loss
            if _proc_loss is not None:
                result.procedural_train_loss = _proc_loss

        else:
            # --- 4b-alt. CURRICULUM PROBE (optional) ---
            episodic_recall_scores: dict[str, float] = {}
            if self.curriculum_sampler and self.episodic_replay_pool:
                self._disable_gradient_checkpointing()
                switch_adapter(self.model, "episodic")
                episodic_recall_scores = self.curriculum_sampler.probe_recall(
                    self.model,
                    self.tokenizer,
                    self.episodic_replay_pool,
                )
                self.curriculum_sampler.update_history(episodic_recall_scores)

            # --- 5. TRAIN EPISODIC ---
            if episodic_rels:
                episodic_loss = self._train_adapter_with_replay(
                    "episodic",
                    episodic_rels,
                    self.episodic_replay_pool,
                    self.config.episodic_new_weight,
                    f"phase3-episodic-cycle{self.cycle_count}",
                    recall_scores=episodic_recall_scores,
                )
                result.episodic_train_loss = episodic_loss

                # Add new relations to episodic replay pool (cap at 100)
                self.episodic_replay_pool.extend(
                    {"question": qa["question"], "answer": qa["answer"]} for qa in episodic_rels
                )
                if len(self.episodic_replay_pool) > 100:
                    self.episodic_replay_pool = self.episodic_replay_pool[-100:]

        # --- 6. SAVE ---
        self._save_adapters()

        result.wall_clock_seconds = time.time() - start_time
        logger.info(
            "Cycle %d complete: %d extracted (%.1fs)",
            self.cycle_count,
            result.entities_extracted,
            result.wall_clock_seconds,
        )

        return result

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

    def _train_adapter_with_replay(
        self,
        adapter_name: str,
        new_qa_pairs: list[dict],
        replay_pool: list[dict],
        replay_weight: float,
        run_name: str,
        recall_scores: Optional[dict[str, float]] = None,
    ) -> Optional[float]:
        """Train adapter on new relations with optional replay.

        Args:
            adapter_name: "episodic" or "semantic"
            new_qa_pairs: List of new QA dicts
            replay_pool: Existing replay pool for this adapter
            replay_weight: Weight for replay (1 - weight is new material)
            run_name: wandb run name
            recall_scores: Optional curriculum recall scores for weighted sampling.

        Returns:
            Training loss or None if dataset is empty.
        """
        switch_adapter(self.model, adapter_name)
        new_dataset = self._qa_to_dataset(new_qa_pairs)

        train_dataset = new_dataset
        if replay_pool:
            self._disable_gradient_checkpointing()
            replay_examples = self._generate_replay_from_pool(
                replay_pool, recall_scores=recall_scores
            )
            self._enable_gradient_checkpointing()

            if replay_examples:
                replay_dataset = self._qa_to_dataset(replay_examples)
                train_dataset = MixedReplayDataset(
                    new_dataset=new_dataset,
                    replay_dataset=replay_dataset,
                    replay_ratio=replay_weight,
                )

        if len(train_dataset) == 0:
            return None

        training_config = self._make_training_config(num_epochs=self.training_config.num_epochs)

        self._enable_gradient_checkpointing()
        metrics = train_adapter(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            adapter_name=adapter_name,
            training_config=training_config,
            adapter_config=(
                self.episodic_config if adapter_name == "episodic" else self.semantic_config
            ),
            wandb_config=self.wandb_config,
            output_dir=self._training_output_dir(adapter_name),
            run_name=run_name,
            thermal_policy=self._thermal_policy,
            hooks=self._build_training_hooks(),
        )

        if metrics is not None and metrics.get("aborted"):
            logger.info(
                "_train_adapter_with_replay: training aborted for %s — skipping registry update",
                adapter_name,
            )
            return None

        return metrics.get("train_loss") if metrics is not None else None

    def _generate_replay_from_pool(
        self,
        pool: list[dict],
        max_replay: int = 8,
        recall_scores: Optional[dict[str, float]] = None,
    ) -> list[dict]:
        """Generate replay examples by querying the model on pool questions.

        Uses the current adapter's knowledge to regenerate answers,
        implementing generative replay without storing raw training data.
        Samples up to max_replay items from the pool to keep cycle time bounded.

        When curriculum sampling is enabled and recall_scores are provided,
        sampling is weighted toward items with lower recall (harder facts).
        """
        from paramem.evaluation.recall import generate_answer
        from paramem.training.dataset import format_inference_prompt

        # Sample from pool — curriculum-weighted or uniform
        if self.curriculum_sampler and recall_scores:
            sampled = self.curriculum_sampler.weighted_sample(
                pool, recall_scores, n_samples=max_replay
            )
        elif len(pool) > max_replay:
            sampled = random.sample(pool, max_replay)
        else:
            sampled = pool

        replay_examples = []
        for item in sampled:
            prompt = format_inference_prompt(item["question"], self.tokenizer)
            generated = generate_answer(
                self.model,
                self.tokenizer,
                prompt,
                temperature=0.0,
            )

            # Quality gate
            if len(generated.split()) < 3:
                continue
            if generated.strip().lower() == item["question"].strip().lower():
                continue

            replay_examples.append(
                {
                    "question": item["question"],
                    "answer": generated,
                }
            )

        return replay_examples

    def _qa_to_dataset(self, qa_pairs: list[dict]) -> Dataset:
        """Convert relation dicts to a SyntheticQADataset."""
        return SyntheticQADataset(
            examples=[{"question": qa["question"], "answer": qa["answer"]} for qa in qa_pairs],
            tokenizer=self.tokenizer,
            max_length=self.training_config.max_seq_length,
        )

    def _make_training_config(self, num_epochs: int) -> TrainingConfig:
        """Build a TrainingConfig for consolidation training.

        Propagates all yaml-configurable hyperparameters from
        ``self.training_config``, including the three fields that were
        previously dropped (``warmup_steps``, ``lr_scheduler_type``,
        ``lr_decay_steps``), so yaml overrides reach ``train_adapter``.
        """
        return TrainingConfig(
            batch_size=self.training_config.batch_size,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            max_seq_length=self.training_config.max_seq_length,
            num_epochs=num_epochs,
            warmup_steps=self.training_config.warmup_steps,
            warmup_ratio=self.training_config.warmup_ratio,
            lr_scheduler_type=self.training_config.lr_scheduler_type,
            lr_decay_steps=self.training_config.lr_decay_steps,
            weight_decay=self.training_config.weight_decay,
            gradient_checkpointing=self.training_config.gradient_checkpointing,
            max_grad_norm=self.training_config.max_grad_norm,
            seed=self.training_config.seed,
        )

    def _save_adapters(
        self,
        *,
        window_stamp_override: "str | None" = None,
    ) -> None:
        """Save adapters and registries to disk using the atomic registry-last ordering.

        Saves to two locations:
        - ``output_dir/<tier>/`` — canonical latest state (server use)
        - ``paths.debug/.../training/tiers/<tier>/adapter_weights/`` —
          per-cycle plaintext shadow for inspection (only when
          ``save_cycle_snapshots`` is on; written by
          :meth:`DebugSnapshotWriter.on_main_adapters_saved`).

        Atomic save ordering — registry written last as the commit signal
        (mirrors ``post_session_train`` step 7):
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

        Args:
            window_stamp_override: When not ``None``, write this value as the
                ``window_stamp`` on all saved main slots instead of computing a
                fresh floor.  Used by :meth:`run_housekeeping` to preserve the
                existing window stamp so :func:`_is_full_cycle_due` is not
                perturbed by the re-groom (a housekeeping fold must not advance
                the cadence window).

        The recall gate threshold is read from ``self.config.recall_sanity_threshold``
        (set once at construction from the YAML field of the same name).
        """
        import hashlib as _hashlib

        from paramem.adapters.manifest import build_manifest_for
        from paramem.memory.interim_adapter import current_full_consolidation_stamp

        fingerprint_cache = getattr(self, "fingerprint_cache", None)
        full_period = getattr(self, "full_consolidation_period_string", "")
        # Use the override when supplied (housekeeping path — preserve the existing
        # window so scheduling is not perturbed); otherwise floor to the current period.
        if window_stamp_override is not None:
            full_window_stamp = window_stamp_override
        else:
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
        # manifest).  Layout owned by DebugSnapshotWriter:
        #   paths.debug/.../training/tiers/<tier>/adapter_weights/
        # Writer self-gates on save_cycle_snapshots; callers do not check.
        tier_shadow = ["episodic"]
        if "semantic" in self.model.peft_config:
            tier_shadow.append("semantic")
        if "procedural" in self.model.peft_config:
            tier_shadow.append("procedural")
        self._debug_writer.on_main_adapters_saved(tier_shadow)

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
                callers (e.g. ``run_consolidation_cycle`` → ``_run_indexed_key_procedural``).
                The ``_current_interim_stamp`` instance-attribute fallback is
                preserved for backward-compat but is always ``None`` now that
                the attribute is never set.

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

    def post_session_train(
        self,
        session_transcript: str,
        session_id: str,
        *,
        speaker_id: str,
        speaker_name: str | None = None,
        ha_context: dict | None = None,
        schedule: str = "",
        max_interim_count: int = 7,
        stamp: str | None = None,
        recall_sanity_threshold: "float | None" = None,
    ) -> dict:
        """Extract one conversation, train onto the current interim adapter, register on success.

        This is the post-conversation training hook for multi-adapter interim routing.
        "Session" here means one conversation (the existing meaning in the codebase),
        not the adapter tier.  The adapter it trains into is the interim adapter
        (``episodic_interim_YYYYMMDDTHHMM``).

        Failure-safe ordering: register keys **only after** training returns
        successfully so that extraction or training failures never leave
        orphaned keys in the registry.

        If ``max_interim_count == 0``, new facts are appended to
        ``self.pending_interim_triples`` (RAM-only) without training.  The pending
        queue is consumed by the next full consolidation fold
        (``consolidate_interim_adapters``).  What is not trained does not exist on
        disk — this upholds the privacy invariant.

        Procedural relations extracted from the same transcript are trained onto
        the stable ``procedural`` main adapter (no interim tier for procedural —
        preferences are small-volume and slow-changing).  This pass runs inline,
        immediately after the episodic training pass and before any registry
        writes, so a failure in either pass leaves the registry clean.

        Registry write ordering (atomic: registry written last as commit signal):
        1. ``save_bytes()`` — serialise registry to bytes without writing.
        2. ``sha256(payload)`` — hash for manifest pre-stamp.
        3. Build manifest with ``registry_sha256_override``.
        4. ``save_adapter(..., manifest=manifest)`` — adapter + manifest on disk.
        5. ``save_from_bytes(payload, path)`` — flush registry bytes to
           ``<adapter_dir>/<tier>/indexed_key_registry.json`` (LAST, with
           unified simhash map embedded under the ``"simhash"`` key).

        The registry is the commit signal: its presence on disk means every
        preceding file is complete.  On restart, the lifespan consistency check
        scans the registry and drops any entry whose adapter slot is missing,
        recovering from a crash between steps 4 and 6.

        Args:
            session_transcript: Raw transcript text for this conversation.
            session_id: Unique conversation identifier (used by the extraction pipeline).
            speaker_id: Speaker identifier for key ownership and preference scoping.
                Required — callers must always supply a real speaker ID.
            speaker_name: Human-readable speaker name for extraction personalisation.
            ha_context: Optional Home Assistant context dict for location validation.
            schedule: Consolidation schedule string (e.g. ``"every 2h"``, ``"03:00"``).
                Used together with *max_interim_count* to compute the sub-interval stamp.
            max_interim_count: Number of sub-intervals per consolidation period.
                ``0`` → queue-until-consolidation branch (no interim adapter created).
            stamp: Override the computed sub-interval stamp.  Injected by tests so the
                flooring logic can be exercised without mocking ``datetime.now()``.

        Returns:
            Result dict with at minimum::

                {
                    "triples_extracted": int,
                    "new_keys": list[str],
                    "adapter_name": str | None,   # None if queued or noop
                    "mode": "trained" | "queued" | "noop",
                    "error": str | None,
                }
        """
        # --- 1. Extract ---
        episodic_rels, procedural_rels = self.extract_session(
            session_transcript,
            session_id,
            speaker_id=speaker_id,
            speaker_name=speaker_name,
            ha_context=ha_context,
        )

        logger.info(
            "post_session_train: session=%s extracted %d episodic relations",
            session_id,
            len(episodic_rels),
        )

        return self.run_consolidation_cycle(
            episodic_rels,
            procedural_rels,
            speaker_id=speaker_id,
            mode="train",
            run_label=session_id,
            schedule=schedule,
            max_interim_count=max_interim_count,
            stamp=stamp,
            recall_sanity_threshold=recall_sanity_threshold,
        )

    def _run_indexed_key_procedural(
        self,
        rels: list[dict],
        speaker_id: str,
        *,
        mode: "Literal['simulate', 'train']" = "train",
        stamp: "str | None" = None,
        run_label: str = "interim",
        failed_session_ids: "set[str] | None" = None,
    ) -> Optional[float]:
        """Train (or simulate) the procedural adapter on *rels*.

        Called by :meth:`run_consolidation_cycle` (primary) and the legacy
        :meth:`run_cycle` path.  The method is extracted (rather than inlined)
        so tests can patch it at the instance level to intercept the call.

        All context that formerly leaked through instance attributes
        (``_current_run_mode``, ``_current_run_label``, ``_current_interim_stamp``)
        is now passed as explicit keyword arguments, eliminating the
        set-before-call / clear-in-finally smuggling pattern.

        Args:
            rels: Deduplicated procedural relations for this cycle, already
                tagged with ``speaker_id`` by ``_tag_speaker_id_defaults``.
            speaker_id: Fallback speaker scope for contradiction detection.
            mode: ``"train"`` writes adapter weights; ``"simulate"`` writes
                sidecar JSON registry without touching PEFT.  Default ``"train"``.
            stamp: Sub-interval stamp forwarded to ``_training_output_dir`` via
                the ``interim_stamp`` parameter.  ``None`` when not called from
                an interim cycle.
            run_label: Tag woven into the wandb ``run_name`` for traceability.
                Default ``"interim"``.
            failed_session_ids: Mutable set shared with the caller
                (:meth:`run_consolidation_cycle`).  When a new procedural key
                fails the recall gate, its contributing session ids (from the
                position-aligned ``_proc_sid_by_key`` dict built from the source
                rels' T9 ``session_id`` fields) are added here so the caller can
                keep those sessions pending (B7-B / T7).  ``None`` disables
                collection (simulate mode or callers that do not need provenance).

        Returns:
            Train loss as ``float`` if training ran, otherwise ``None``.
        """
        if not rels:
            return None

        # Local import alias so tests can patch paramem.training.trainer.train_adapter.
        from paramem.training.trainer import train_adapter as _train_adapter

        if mode == "train":
            switch_adapter(self.model, "procedural")
        new_proc_keyed, existing_proc_keyed = self._prepare_procedural_keys_for_tier(
            rels, speaker_id, mode=mode
        )
        # B7-B (T7): build key → contributing session ids map from the source rels
        # (position-aligned with new_proc_keyed — _mint_keyed_entries is 1-to-1 with rels).
        # Kept separate from the entry dict so session_ids are never persisted (D4 invariant).
        _proc_sid_by_key: dict[str, list[str]] = {}
        for _i, _kp in enumerate(new_proc_keyed):
            _src_sid = rels[_i].get("session_id", "") if _i < len(rels) else ""
            _proc_sid_by_key[_kp["key"]] = [_src_sid] if _src_sid else []
        all_procedural = existing_proc_keyed + new_proc_keyed
        if not (mode == "train" and all_procedural):
            return None
        examples = format_entry_training(all_procedural, self.tokenizer, max_length=1024)
        dataset = self._indexed_dataset(examples)
        training_config = self._make_training_config(num_epochs=self.training_config.num_epochs)
        self._enable_gradient_checkpointing()
        recall_cb, recall_state = self._maybe_make_recall_callback(
            entries=all_procedural,
            adapter_name="procedural",
            output_dir=self._training_output_dir("procedural", interim_stamp=stamp),
            phase_name=f"interim-procedural-{run_label}",
        )
        proc_metrics = _train_adapter(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            adapter_name="procedural",
            training_config=training_config,
            adapter_config=self.procedural_config,
            wandb_config=self.wandb_config,
            output_dir=self._training_output_dir("procedural", interim_stamp=stamp),
            run_name=f"interim-procedural-{run_label}",
            thermal_policy=self._thermal_policy,
            hooks=self._build_training_hooks(),
            callbacks_extra=[recall_cb] if recall_cb is not None else None,
        )
        if proc_metrics is not None and proc_metrics.get("aborted"):
            logger.info(
                "_run_indexed_key_procedural: training aborted — skipping deferred mutations"
            )
            return None
        # Deferred mutations — apply only after _train_adapter succeeds without abort.
        # Compute the recall-passing set: keys whose exact_match verdict is True
        # on the FINAL trained weights.  None recall_state means early-stop was
        # disabled; fall back to a dedicated per-key probe.
        passing = self._recall_passing_keys(recall_state, all_procedural)
        if passing is None:
            passing = self._probe_passing_keys("procedural", all_procedural)
        self._procedural_next_index = self._procedural_tentative_next_index
        for kp in new_proc_keyed:
            if kp["key"] not in passing:
                logger.debug(
                    "_run_indexed_key_procedural: key %s failed recall gate"
                    " — skipping registration",
                    kp["key"],
                )
                # B7-B (T7): accumulate the contributing session ids for this
                # recall-failed key into the shared cycle-local set so the
                # caller (_run_consolidation_cycle) can keep those sessions
                # pending.  _proc_sid_by_key is built above from the source
                # rels' T9 session_id fields; session_ids are NOT on the entry
                # dict to preserve the D4 no-persistence invariant.
                if failed_session_ids is not None:
                    failed_session_ids.update(_proc_sid_by_key.get(kp["key"], []))
                continue
            fingerprint = compute_simhash(kp["key"], kp["subject"], kp["predicate"], kp["object"])
            self.store.put("procedural", kp["key"], kp, simhash=fingerprint)
            self.store.set_bookkeeping(
                kp["key"],
                speaker_id=kp["speaker_id"],
                first_seen_cycle=kp["first_seen_cycle"],
                relation_type=kp.get("relation_type", "factual"),
                recurrence_count=1,
                last_seen_cycle=self.cycle_count,
            )
        return proc_metrics.get("train_loss") if proc_metrics is not None else None

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
        stamp: str | None = None,
        recall_sanity_threshold: "float | None" = None,
        new_promotions: "list[str] | None" = None,
    ) -> dict:
        """Unified interim-cycle entry: key prep + optional training + atomic persistence.

        Replaces the former ``_train_extracted_into_interim`` (train) and
        ``simulated_training`` (simulate) methods.  Both modes execute the same
        pipeline — the ONLY mode-conditional code is:

        * :meth:`_resolve_target_slot` — slot-selection / cap logic (train-mode
          only; simulate always mints fresh).
        * :meth:`_prepare_episodic_keys_for_tier` — reconstruction source (probe
          weights in train; read cache in simulate).
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
           to obtain ``(adapter_name, cap_reached_absorb, queue_only,
           degenerated_skip)``.
        6. Handle control-flow branches: queue-only and degenerated-skip.
        7. Mint PEFT slot (train only).
        8. Materialize (B1): call :meth:`_materialize_consolidation_graph` scoped
           to the current slot for the recall-miss diagnostic and to rebuild the
           keying surface (pending-session relations from ``merger.graph`` are
           passed as ``extra_relations`` so they survive the graph reset).
        8c. Refine (B2): call :meth:`_refine_consolidation_graph` with
           ``enrich=(interim_refinement=="full")`` so SOTA enrichment is
           level-gated.  The recurrence-bump runs at every level.
        9. Prepare episodic key list via ``_prepare_episodic_keys_for_tier``.
           When *new_promotions* is non-empty, move matching keys from episodic
           to semantic before training (mirrors the former
           ``_run_indexed_key_episodic`` promotion-transfer step).
        10. Train (train mode) or skip training (simulate mode).
        11. Procedural: ``_prepare_procedural_keys_for_tier`` + train (train) or
            skip (simulate).
        12. Persist both tiers via ``commit_tier_slot``.
        13. Restore ``"episodic"`` as the active adapter (mode-agnostic).
        14. Return result dict.

        Args:
            episodic_rels: Pre-extracted episodic relations.  May already carry
                ``speaker_id``; missing entries are tagged with *speaker_id*.
            procedural_rels: Pre-extracted procedural relations.  Always trained
                onto the stable ``"procedural"`` main adapter (no interim tier).
            speaker_id: Default speaker tag for relations missing one; also
                used for procedural contradiction scoping.  Required — callers
                must always supply a real speaker ID.
            mode: ``"train"`` writes adapter weights; ``"simulate"`` writes
                sidecar JSON registry without touching PEFT.
            run_label: Tag woven into the wandb ``run_name`` for traceability.
                Pass ``session_id`` for per-session calls, or
                ``"tick-<stamp>"`` for batch calls from the scheduled tick.
            schedule: Consolidation refresh-cadence string used to compute the
                sub-interval stamp when *stamp* is not provided.
            max_interim_count: Cap on concurrent interim adapters.
                ``0`` → queue-until-consolidation branch (RAM-only, no training).
            stamp: Override the computed sub-interval stamp (test injection).
            recall_sanity_threshold: Accepted for caller override compatibility;
                the active threshold is read from ``self.config.recall_sanity_threshold``
                when this is ``None`` (the default).  Not currently consumed inside
                this method — interim-cycle saves go through ``commit_tier_slot``,
                which reads the threshold from the loop config.
            new_promotions: Optional list of entity names promoted to semantic
                this cycle.  When non-empty, matching episodic keys are moved to
                the semantic tier via ``store.move`` before training, so they are
                excluded from the episodic training set.  Only meaningful in
                train mode; ignored in simulate mode.

        Returns:
            Result dict with keys ``{"triples_extracted", "new_keys",
            "adapter_name", "mode", "venue", "error"}``.  ``mode`` is the
            outcome (``"trained"``, ``"simulated"``, ``"queued"``,
            ``"degenerated"``, or ``"noop"``); ``venue`` is the training
            medium (``"train"`` or ``"simulate"``).
        """
        from paramem.memory.persistence import commit_tier_slot

        # Resolve threshold from config when the caller did not supply an override.
        # Not currently consumed inside this method; resolved here so the override
        # contract is honoured and future callers can rely on the config path.
        if recall_sanity_threshold is None:
            recall_sanity_threshold = self.config.recall_sanity_threshold

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

        # --- 6. Resolve stamp and target slot ---
        if stamp is None:
            from paramem.memory.interim_adapter import current_interim_stamp as _cis

            stamp = _cis(schedule)

        adapter_name, cap_reached_absorb, queue_only, degenerated_skip = self._resolve_target_slot(
            stamp, max_interim_count, mode=mode
        )

        # --- 7a. Queue-only branch ---
        if queue_only:
            self.pending_interim_triples.extend(episodic_rels)
            logger.info(
                "run_consolidation_cycle: max_interim_count=0 — queued %d triples "
                "(total pending: %d)",
                len(episodic_rels),
                len(self.pending_interim_triples),
            )
            queued_summary = {
                "triples_extracted": triples_extracted,
                "new_keys": [],
                "adapter_name": None,
                "mode": "queued",
                "venue": mode,
                "error": None,
            }
            self._debug_writer.on_cycle_end(queued_summary, interim_stamp=stamp)
            return queued_summary

        # --- 7b. Degenerated-skip branch (train mode only) ---
        if degenerated_skip:
            self.pending_interim_triples.extend(episodic_rels)
            logger.warning(
                "run_consolidation_cycle: target adapter degenerated — "
                "queued %d triples (pending: %d)",
                len(episodic_rels),
                len(self.pending_interim_triples),
            )
            degenerated_summary = {
                "triples_extracted": triples_extracted,
                "new_keys": [],
                "adapter_name": adapter_name,
                "mode": "degenerated",
                "venue": mode,
                "error": "target_adapter_degenerated",
            }
            self._debug_writer.on_cycle_end(degenerated_summary, interim_stamp=stamp)
            return degenerated_summary

        # --- End-of-extraction debug dump (per-tier relation lists) ---
        # Self-gated; no-op when save_cycle_snapshots=False.
        self._debug_writer.on_extraction_end(episodic_rels, procedural_rels, interim_stamp=stamp)

        # --- 8. Mint PEFT slot (train only) ---
        if mode == "train":
            from paramem.memory.interim_adapter import create_interim_adapter
            from paramem.models.loader import create_adapter as _create_adapter

            if cap_reached_absorb:
                # Reset absorb target to LoRA zeros for a clean retrain.
                self.model.delete_adapter(adapter_name)
                self.model = _create_adapter(self.model, self.episodic_config, adapter_name)
                logger.info(
                    "run_consolidation_cycle: cap-reached — reset %s to LoRA zeros",
                    adapter_name,
                )
            elif adapter_name not in self.model.peft_config:
                # Fresh slot for this sub-interval.
                self.model = create_interim_adapter(self.model, self.episodic_config, stamp)
                logger.info("run_consolidation_cycle: created interim adapter %s", adapter_name)

        # --- 8b. Materialize: recall-miss diagnostic + rebuild keying surface ---
        # Scoped to the current slot: reconstruct only the slot's registered keys
        # (tier=adapter_name, keys=active_keys_in_tier(adapter_name)) for the
        # recall-miss diagnostic, then reset the keying graph and re-merge:
        #   (a) registry-true relations for this slot
        #   (b) the pending-session relations captured from merger.graph before the
        #       reset (extra_relations), so they survive and co-reside with the
        #       slot's recalled facts in the fresh keying surface.
        #
        # INVARIANT: extra_relations NEVER enter the recall_miss_keys set.
        # recall_miss_keys is computed against store.all_active_keys() BEFORE the
        # graph reset inside _materialize_consolidation_graph — pending unregistered
        # relations therefore cannot distort it.
        #
        # Cap-absorb scope (risk #9): _resolve_target_slot returns the NEWEST slot
        # as adapter_name when cap_reached_absorb=True; active_keys_in_tier
        # correctly scopes to that slot's keys — no special-casing needed here.
        #
        # extra_relations are populated from merger.graph edges only when
        # interim_refinement != "off" (under "off", extract_session skips the merge
        # so merger.graph does not contain pending-session content; B6 will change
        # this to always merge, but until then the "off" path passes None).
        _pending_relations: "list[Relation] | None" = None
        if self.config.interim_refinement != "off":
            import networkx as _nx

            _g = getattr(self.merger, "graph", None)
            if isinstance(_g, _nx.MultiDiGraph) and _g.number_of_edges() > 0:
                _pending_relations = []
                for _er_subj, _er_obj, _er_data in _g.edges(data=True):
                    _er_pred = _er_data.get("predicate", "")
                    if not _er_pred:
                        continue
                    _er_rt_raw = _er_data.get("relation_type", _FALLBACK_RTYPE)
                    _er_rt: str = _er_rt_raw if _er_rt_raw in _VALID_RTYPES else _FALLBACK_RTYPE
                    # speaker_id: prefer the subject node's speaker_id attribute
                    # (set by _upsert_entity for speaker entities, where the node
                    # key IS the speaker_id).  Falls back to "" for non-speaker
                    # subjects (B3 will improve this via entity re-merge).
                    _er_subj_node = _g.nodes.get(_er_subj, {})
                    _er_spk = _er_subj_node.get("speaker_id", "")
                    _pending_relations.append(
                        Relation(
                            subject=_er_subj,
                            predicate=_er_pred,
                            object=_er_obj,
                            relation_type=_er_rt,  # type: ignore[arg-type]
                            confidence=_er_data.get("confidence", 1.0),
                            speaker_id=_er_spk,
                            # B7-A: recover the real contributing session ids
                            # from the merger edge so they survive the re-merge
                            # through _merge_registry_relations (T3).  The
                            # merger already accumulated the UNION in
                            # edge["sessions"] via T2; we carry them forward so
                            # _build_all_edge_entries_into can expose them on
                            # the deferred-write record (T5).
                            session_ids=list(_er_data.get("sessions", [])),
                        )
                    )

        recall_miss_keys, recon_relations = self._materialize_consolidation_graph(
            tier=adapter_name,
            keys=list(self.store.active_keys_in_tier(adapter_name)),
            extra_relations=_pending_relations,
        )
        if recall_miss_keys:
            logger.info(
                "run_consolidation_cycle: %d recall-miss key(s) in slot %s "
                "(kept in training set with registry-true content)",
                len(recall_miss_keys),
                adapter_name,
            )

        # --- 8c. Refine: SOTA enrichment (level-gated) + recurrence bumps ---
        # enrich=True only when interim_refinement=="full"; the recurrence-bump
        # loop inside _refine_consolidation_graph runs regardless of enrich so
        # Case-1 duplicate-SPO collapses are captured at every merge level.
        # Note on atomicity: bump_recurrence targets ALREADY-REGISTERED keys
        # (Case-1 survivors from the merge); it does NOT register new keys.
        # Bumping a registered key's counter before training is benign — the
        # key remains valid whether training aborts or not, and the increment
        # is re-applied identically on the next cycle (idempotent semantics).
        self._refine_consolidation_graph(
            recon_relations,
            enrich=(self.config.interim_refinement == "full"),
        )

        # --- 9. Build keyed training set via graph-walk (B3) ---
        # Replaces the flat _prepare_episodic_keys_for_tier probe path.  After
        # _materialize_consolidation_graph (B1) + _refine_consolidation_graph (B2),
        # merger.graph holds: (a) registry-true re-merged keys (as keyed edges with
        # ik_key stamped) plus (b) the pending-session relations (as keyless edges).
        # _build_all_edge_entries_into handles both sets in one pass:
        #   - Keyless edges (new): minted keys with defer=True so a training abort
        #     cannot leave orphan registry entries (interim atomicity).
        #   - Keyed edges (existing): anti-forgetting replay entries sourced from
        #     the store — not registered as new.
        #
        # Procedural handling (B3): tier_keyed["procedural"] may be populated by the
        # walk (procedural-typed keyless edges from the pending sessions co-merged
        # into merger.graph) but is NOT trained in B3 — _run_indexed_key_procedural
        # (step 11) continues to handle procedural via the flat path.  Deferred
        # writes for procedural entries are NOT flushed in B3 (no counter advance,
        # no store write); _run_indexed_key_procedural owns the procedural indices
        # from _procedural_next_index independently.
        if mode == "train":
            switch_adapter(self.model, adapter_name)
        _tier_keyed: dict[str, list[dict]] = {"episodic": [], "procedural": [], "semantic": []}
        _, _deferred_writes = self._build_all_edge_entries_into(
            _tier_keyed,
            default_speaker_id=speaker_id,
            defer=True,
            tag_new=True,
        )

        # all_interim_keyed: episodic entries only (new minted + existing keyed).
        # Procedural bucket (_tier_keyed["procedural"]) is collected but unused in B3.
        all_interim_keyed = _tier_keyed["episodic"]

        # new_keyed_episodic: deferred writes for episodic minted entries only.
        # Procedural deferred writes are excluded (handled by _run_indexed_key_procedural).
        new_keyed_episodic = [rec for rec in _deferred_writes if rec["tier"] == "episodic"]
        new_key_ids = [rec["entry"]["key"] for rec in new_keyed_episodic]

        if mode == "simulate":
            # No training step — apply episodic store mutations immediately.
            # Mirrors the former new_keyed_episodic simulate path; uses the
            # deferred-write payload from _build_all_edge_entries_into(defer=True).
            for rec in new_keyed_episodic:
                _entry = rec["entry"]
                _key = _entry["key"]
                self.store.put(
                    adapter_name,
                    _key,
                    _entry,
                    simhash=compute_simhash(
                        _key, _entry["subject"], rec["predicate"], _entry["object"]
                    ),
                )
                self.store.set_bookkeeping(
                    _key,
                    speaker_id=rec["speaker_id"],
                    first_seen_cycle=self.cycle_count,
                    relation_type=rec["relation_type"],
                    recurrence_count=1,
                    last_seen_cycle=self.cycle_count,
                )
            self._indexed_next_index += len(new_keyed_episodic)

        # --- 9b. Semantic promotion transfer (train mode, when promotions supplied) ---
        # Move keys for promoted entities from episodic to the semantic tier so
        # they are excluded from the episodic training set.  This mirrors the
        # logic that formerly lived inside _run_indexed_key_episodic.
        promoted_key_set: set[str] = set()
        if mode == "train" and new_promotions:
            promoted_set = {n.lower() for n in new_promotions}
            for _tier, key, entry in list(self.store.iter_entries()):
                if key.startswith("proc"):
                    continue
                subject = entry.get("subject", "").lower()
                obj = entry.get("object", "").lower()
                mentions = (subject and subject in promoted_set) or (obj and obj in promoted_set)
                if mentions and not self.store.has_simhash("semantic", key):
                    promoted_key_set.add(key)
            for key in promoted_key_set:
                self.store.move(key, "semantic")
            # Exclude promoted keys from the episodic training set.
            all_interim_keyed = [
                kp for kp in all_interim_keyed if kp["key"] not in promoted_key_set
            ]

        # --- 10. Train (train mode) or skip (simulate) ---
        epi_train_loss: float | None = None
        if mode == "train" and all_interim_keyed:
            epi_metrics, recall_state = self._train_tier_adapter(
                all_interim_keyed,
                adapter_name=adapter_name,
                adapter_config=self.episodic_config,
                training_config=self._make_training_config(
                    num_epochs=self.training_config.num_epochs
                ),
                output_dir=self._training_output_dir(adapter_name, interim_stamp=stamp),
                run_name=f"interim-{adapter_name}-{run_label}",
                phase_name=f"interim-{adapter_name}-{run_label}",
            )
            epi_train_loss = epi_metrics.get("train_loss") if epi_metrics is not None else None
            if epi_metrics is not None and epi_metrics.get("aborted"):
                # Training was aborted for inference.  Skip simhash update,
                # procedural training, deferred episodic mutations, and the
                # commit_tier_slot call below.  The production adapter on disk
                # is untouched; the next cycle will retrain from scratch.
                logger.info("run_consolidation_cycle: episodic training aborted — skipping commit")
                return {"mode": "aborted", "adapter_name": adapter_name}
            # Episodic store mutations are deferred until AFTER the procedural
            # step completes — see the mutation block below the procedural gate.
            # This guarantees full-cycle atomicity: if any training step fails,
            # the registry stays clean.

            # Compute the recall-passing set for the episodic interim tier.
            # None state means early-stop disabled; fall back to dedicated probe.
            # FAIL-SAFE: None MUST route to the probe — never treated as empty.
            _epi_passing = self._recall_passing_keys(recall_state, all_interim_keyed)
            if _epi_passing is None:
                _epi_passing = self._probe_passing_keys(adapter_name, all_interim_keyed)
        else:
            # Simulate mode or no keyed pairs: admit all without a probe.
            _epi_passing = None

        # Update episodic-interim simhash registry from ground-truth pairs,
        # filtered to recall-passing keys only.
        if _epi_passing is not None:
            _passing_interim = [kp for kp in all_interim_keyed if kp["key"] in _epi_passing]
        else:
            _passing_interim = all_interim_keyed
        self.store.replace_simhashes_in_tier(adapter_name, build_registry(_passing_interim))

        # --- 11. Procedural (always onto stable "procedural" main adapter) ---
        # B7-B (T6/T7): cycle-local set accumulates session ids whose contributed
        # fact failed the recall gate (episodic at 11b, procedural inside
        # _run_indexed_key_procedural).  Guarded to train mode: in simulate mode
        # _epi_passing is None so the episodic drop site is never reached, and
        # the procedural drop site only receives the set when mode=="train".
        # Simulate callsite (app.py:12499 B2) is explicitly NOT plumbed.
        _recall_failed_session_ids: set[str] = set()
        proc_train_loss: float | None = None
        if self.procedural_config is not None and procedural_rels:
            proc_train_loss = self._run_indexed_key_procedural(
                procedural_rels,
                speaker_id,
                mode=mode,
                stamp=stamp,
                run_label=run_label,
                # Pass None in simulate mode so the procedural gate is not reached
                # and the set stays empty (B2: simulate has no recall gate).
                failed_session_ids=_recall_failed_session_ids if mode == "train" else None,
            )

        # --- 11b. Apply deferred episodic store mutations ---
        # Reached only when all training steps above completed without raising.
        # In simulate mode the episodic writes were already applied before training
        # (in the simulate branch of step 9 above); the counter was already advanced.
        #
        # new_keyed_episodic carries the harvest records (from _deferred_writes)
        # filtered to the episodic tier.  Each record has all fields needed for
        # store.put (entry["key"], entry, canon_subj/pred/canon_obj for simhash)
        # and store.set_bookkeeping (speaker_id, relation_type).
        if mode == "train":
            _flushed_count = 0
            for rec in new_keyed_episodic:
                _entry = rec["entry"]
                _key = _entry["key"]
                if _epi_passing is not None and _key not in _epi_passing:
                    logger.debug(
                        "run_consolidation_cycle: key %s failed recall gate"
                        " — skipping registration",
                        _key,
                    )
                    # B7-B (T6): accumulate contributing session ids for this
                    # recall-failed episodic key so the caller can keep those
                    # sessions pending.  rec["session_ids"] is populated by
                    # _build_all_edge_entries_into (T5) from edge["sessions"]
                    # with synthetic sentinels already excluded.
                    _recall_failed_session_ids.update(rec.get("session_ids", []))
                    continue
                self.store.put(
                    adapter_name,
                    _key,
                    _entry,
                    simhash=compute_simhash(
                        _key, _entry["subject"], rec["predicate"], _entry["object"]
                    ),
                )
                self.store.set_bookkeeping(
                    _key,
                    speaker_id=rec["speaker_id"],
                    first_seen_cycle=self.cycle_count,
                    relation_type=rec["relation_type"],
                    recurrence_count=1,
                    last_seen_cycle=self.cycle_count,
                )
                _flushed_count += 1
            self._indexed_next_index += _flushed_count

        # --- 12. Persist both tiers ---
        # Each commit_tier_slot is internally crash-safe (registry-last write is
        # the commit signal; a torn slot is skipped by the boot validator).  The
        # PAIR is NOT transactional: a crash between the episodic and procedural
        # commits leaves the episodic interim slot committed and the procedural
        # main slot at its prior state.  That window is RECOVERABLE and must stay
        # so: the source session is NOT marked consolidated until this method
        # returns successfully (the production caller — app.py's post-session
        # tick / full-cycle path — calls session_buffer.mark_consolidated only
        # after run_consolidation_cycle returns).  A crash here therefore raises,
        # the caller never marks consolidated, the session stays pending, and the
        # procedural facts are re-extracted next cycle.  Do NOT mark sessions
        # consolidated from inside this method, and do NOT wrap these commits in a
        # heavy pseudo-transaction — the residual "re-extract procedural next
        # cycle" cost is acceptable and the recovery is provably safe.
        # Regression guard: tests/test_post_session_train.py
        # ::TestInterTierCommitRecoverable.
        commit_tier_slot(
            loop=self,
            tier="episodic",
            adapter_name=adapter_name,
            stamp=stamp,
            mode=mode,
            all_keyed=all_interim_keyed,
            output_dir=self.output_dir,
        )
        if self.procedural_config is not None and "procedural" in (
            self.model.peft_config if mode == "train" else {"procedural"}
        ):
            commit_tier_slot(
                loop=self,
                tier="procedural",
                adapter_name="procedural",
                stamp=stamp,
                mode=mode,
                all_keyed=[],
                output_dir=self.output_dir,
            )

        # --- 13. End-of-cycle: restore episodic as active adapter ---
        # Mode-agnostic: the peft_config check already short-circuits in simulate
        # (no adapters were created), so the guard is sufficient without the
        # mode == "train" clause.
        if "episodic" in self.model.peft_config:
            switch_adapter(self.model, "episodic")

        logger.info(
            "run_consolidation_cycle: %s %s — %d new keys, %d total episodic keys",
            mode,
            adapter_name,
            len(new_key_ids),
            len(all_interim_keyed),
        )

        cycle_summary = {
            "triples_extracted": triples_extracted,
            "new_keys": new_key_ids,
            "adapter_name": adapter_name,
            "mode": "trained" if mode == "train" else "simulated",
            "venue": mode,
            "error": None,
            "episodic_train_loss": epi_train_loss,
            "procedural_train_loss": proc_train_loss,
            # B7-B (T8): contributing session ids whose new key failed the recall
            # gate this cycle.  Callers use .get("recall_failed_session_ids", [])
            # so early-return paths (aborted, no-relations, queued) that do not
            # run the recall gate are safe.  Simulate mode always produces [].
            "recall_failed_session_ids": sorted(_recall_failed_session_ids),
        }
        self._debug_writer.on_cycle_end(cycle_summary, interim_stamp=stamp)
        return cycle_summary

    def _run_graph_enrichment(self) -> dict:
        """Post-merge graph-level SOTA enrichment pass (Task #10).

        Runs at full consolidation over the cumulative ``merger.graph`` to
        capture cross-transcript second-order relations that per-transcript
        enrichment cannot see.  Folds in coreference resolution via
        ``same_as`` pairs emitted by the SOTA response.

        The method mutates ``self.merger.graph`` in place: first applying
        ``same_as`` node contractions, then inserting new edges tagged with
        the provenance attribute ``edge_source="graph_enrichment"`` (stored
        under :data:`paramem.memory.persistence._EDGE_SOURCE_ATTR`, not the
        NetworkX-reserved ``"source"`` field, so the tag survives persist).

        Early-return conditions (all return ``skipped=True``):
        - ``graph_enrichment_enabled`` is ``False``.
        - Graph has fewer than 10 nodes (floor — too little signal).
        - ``extraction_noise_filter`` is empty (no SOTA provider configured).
        - Provider env-var is absent (API key not set).

        Chunking strategy:
        Entities are ranked by ``recurrence_count`` descending.  For each
        focal entity an N-hop ego-graph is built (``radius=neighborhood_hops``).
        Chunks are deduplicated by node frozenset so overlapping ego-graphs do
        not re-send the same payload.  The number of chunks is capped at
        ``ceil(total_nodes / max_entities_per_pass)`` to prevent O(N) SOTA
        calls on large graphs.

        Returns:
            Diagnostics dict with keys:
                - ``chunks`` (int): number of SOTA calls made.
                - ``new_edges`` (int): edges added to the graph.
                - ``same_as_merges`` (int): node contractions applied.
                - ``skipped`` (bool): ``True`` when enrichment was bypassed.
                - ``skip_reason`` (str | None): reason token when skipped.
        """
        import math

        import networkx as nx

        from paramem.memory.persistence import _EDGE_SOURCE_ATTR, _IK_KEY_ATTR

        _empty = {"chunks": 0, "new_edges": 0, "same_as_merges": 0}

        if not self.graph_enrichment_enabled:
            logger.info("graph_enrichment: disabled — skipping")
            return {**_empty, "skipped": True, "skip_reason": "disabled"}

        graph = self.merger.graph
        node_count = graph.number_of_nodes()

        if node_count < 10:
            logger.info(
                "graph_enrichment: graph too small (%d nodes < 10 floor) — skipping",
                node_count,
            )
            return {**_empty, "skipped": True, "skip_reason": "floor"}

        # Graph-tier SOTA enrichment shares the operator-configured provider
        # with session-tier extraction (anonymize → noise-filter → plausibility
        # chain).  Reading from ``self.extraction.config`` keeps both tiers
        # pointing at the same model + endpoint without an extra knob.
        ext_cfg = self.extraction.config
        provider = ext_cfg.noise_filter
        if not provider:
            logger.info("graph_enrichment: no SOTA provider configured — skipping")
            return {**_empty, "skipped": True, "skip_reason": "no_provider"}

        key_env = PROVIDER_KEY_ENV.get(provider, "")
        api_key = os.environ.get(key_env, "") if key_env else ""
        if not api_key:
            logger.warning(
                "graph_enrichment: provider=%r env_var=%r not set — skipping",
                provider,
                key_env,
            )
            return {**_empty, "skipped": True, "skip_reason": "no_api_key"}

        filter_model = ext_cfg.noise_filter_model
        endpoint = ext_cfg.noise_filter_endpoint or None
        max_entities = max(1, self.graph_enrichment_max_entities_per_pass)
        hops = max(1, self.graph_enrichment_neighborhood_hops)

        # Rank nodes by recurrence descending.
        nodes_by_recurrence = sorted(
            graph.nodes(data=True),
            key=lambda nd: nd[1].get("recurrence_count", 0),
            reverse=True,
        )

        # Build deduplicated chunks from N-hop ego-graphs.
        undirected = graph.to_undirected(as_view=True)
        seen_chunks: set[frozenset] = set()
        chunks: list[list[str]] = []
        chunk_cap = max(1, math.ceil(node_count / max_entities))

        for focal, _ in nodes_by_recurrence:
            if len(chunks) >= chunk_cap:
                break
            if focal not in undirected:
                continue
            ego = nx.ego_graph(undirected, focal, radius=hops)
            nodes = list(ego.nodes)
            if len(nodes) > max_entities:
                # Trim: keep focal + top-(cap-1) neighbours by degree.
                neighbours = sorted(
                    (n for n in nodes if n != focal),
                    key=lambda n: undirected.degree(n),
                    reverse=True,
                )
                nodes = [focal] + neighbours[: max_entities - 1]
            key = frozenset(nodes)
            if key in seen_chunks:
                continue
            seen_chunks.add(key)
            chunks.append(nodes)

        total_new = 0
        total_merges = 0
        calls_made = 0
        seen_merge_keys: set[frozenset] = set()
        # Accumulates ik_keys from edges dropped by successful same_as contractions.
        # Keys are written to self.merger.removal_ledger after the loop completes
        # so the classifier can distinguish intended enrichment-driven removals from
        # genuine reconstruction failures.
        _collapsed_ik: dict[str, str] = {}  # ik_key → keep node

        for chunk_nodes in chunks:
            try:
                chunk_subgraph = graph.subgraph(chunk_nodes)
                triples = serialize_subgraph_triples(chunk_subgraph)
                result = _graph_enrich_with_sota(
                    triples,
                    api_key,
                    provider,
                    filter_model,
                    endpoint,
                )
                calls_made += 1
                if result is None:
                    logger.warning("graph_enrichment: SOTA call returned None for chunk")
                    continue
                new_rels, same_as_pairs, _raw = result
            except Exception as exc:
                logger.warning(
                    "graph_enrichment: exception during SOTA call — %s: %s",
                    type(exc).__name__,
                    exc,
                )
                continue

            # Apply same_as contractions FIRST so subsequent edge inserts
            # reference canonical nodes. Gate on:
            #   1. Both endpoints exist in the live graph.
            #   2. Unordered-pair dedup across the whole enrichment pass.
            #   3. Surface-form safety gate (token-subset + Jaro-Winkler).
            coref_map: dict[str, str] = {}
            for pair in same_as_pairs:
                keep, drop = pair[0], pair[1]
                if keep == drop:
                    continue
                # Canonicalize for graph lookup; SOTA returns surface names, nodes
                # are canonical post-model-A.  BL-1: keep _safe_to_merge_surface on
                # the original SURFACE strings (fuzzy layer-2 check).
                keep_canon = canonical(keep)
                drop_canon = canonical(drop)
                if keep_canon == drop_canon:
                    continue
                if keep_canon not in graph or drop_canon not in graph:
                    logger.debug(
                        "graph_enrichment: same_as skip — keep=%r drop=%r not both in graph",
                        keep_canon,
                        drop_canon,
                    )
                    continue
                merge_key = frozenset({keep_canon, drop_canon})
                if merge_key in seen_merge_keys:
                    logger.debug(
                        "graph_enrichment: same_as dedup — %r / %r already seen",
                        keep_canon,
                        drop_canon,
                    )
                    continue
                seen_merge_keys.add(merge_key)
                if not _safe_to_merge_surface(keep, drop):
                    logger.info(
                        "graph_enrichment: same_as rejected by surface gate — %r / %r",
                        keep,
                        drop,
                    )
                    continue
                # Collect ik_keys from edges in both directions that will become
                # self-loops (and be dropped by self_loops=False) on success.
                # Use inner-dict .values() iteration — MultiDiGraph get_edge_data
                # returns {edge_id: data_dict}; do NOT treat the outer dict as data.
                _pending: dict[str, str] = {}
                for _u, _v in [(keep_canon, drop_canon), (drop_canon, keep_canon)]:
                    for _edata in (graph.get_edge_data(_u, _v) or {}).values():
                        _ik = _edata.get(_IK_KEY_ATTR)
                        if _ik:
                            _pending[_ik] = keep_canon
                try:
                    nx.contracted_nodes(graph, keep_canon, drop_canon, self_loops=False, copy=False)
                    total_merges += 1
                    # Success — absorb pending keys into the accumulator.
                    _collapsed_ik.update(_pending)
                    coref_map[drop_canon] = keep_canon
                    logger.debug("graph_enrichment: contracted %r → %r", drop_canon, keep_canon)
                except Exception as exc:
                    logger.warning(
                        "graph_enrichment: same_as contraction failed %r → %r: %s",
                        drop_canon,
                        keep_canon,
                        exc,
                    )

            def _resolve_name(name: str) -> str:
                # Canonicalize surface name then follow drop→keep chains.
                name = canonical(name)
                seen: set[str] = set()
                while name in coref_map and name not in seen:
                    seen.add(name)
                    name = coref_map[name]
                return name

            # Apply new edges.
            fallback_rtype = "factual"
            valid_rtypes = {"factual", "temporal", "preference", "social"}
            for rel in new_rels:
                if not isinstance(rel, dict):
                    continue
                # Remap endpoints through this chunk's coref map so edges
                # referencing a to-be-dropped node still land on the canonical.
                subj = _resolve_name(rel.get("subject", ""))
                raw_pred = rel.get("predicate", "")
                obj = _resolve_name(rel.get("object", ""))
                rtype = rel.get("relation_type", fallback_rtype)
                if rtype not in valid_rtypes:
                    rtype = fallback_rtype
                pred = canonical(raw_pred)
                if not (subj and pred and obj and subj != obj):
                    continue

                # Canonicalize symmetric predicates so (A,P,B) and (B,P,A)
                # collapse to a single direction (subj < obj lexicographically).
                if pred in _SYMMETRIC_ENRICHMENT_PREDICATES and subj > obj:
                    subj, obj = obj, subj

                # Ensure both endpoint nodes exist.
                for node_name in (subj, obj):
                    if node_name not in graph:
                        graph.add_node(
                            node_name,
                            entity_type="concept",
                            attributes={},
                            recurrence_count=1,
                            sessions=[],
                            first_seen="graph_enrichment",
                            last_seen="graph_enrichment",
                        )

                # Skip exact-triple duplicates.
                duplicate = any(
                    d.get("predicate") == pred and tgt == obj
                    for _, tgt, d in graph.out_edges(subj, data=True)
                )
                if duplicate:
                    continue

                try:
                    confidence = float(rel.get("confidence", 0.8))
                except (TypeError, ValueError):
                    confidence = 0.8
                # Safety net for the prompt-level 0.7 rule: discard low-confidence
                # enriched edges even if the model ignored its own instruction.
                if confidence < 0.7:
                    continue

                graph.add_edge(
                    subj,
                    obj,
                    predicate=pred,
                    relation_type=rtype,
                    confidence=confidence,
                    # Stored under _EDGE_SOURCE_ATTR ("edge_source"), not "source":
                    # NetworkX's node_link_data reserves "source" for the edge's
                    # source-NODE name and would clobber this provenance tag on
                    # persist (same collision class as "key" → "ik_key").
                    **{_EDGE_SOURCE_ATTR: "graph_enrichment"},
                    sessions=[],
                )
                total_new += 1

        logger.info(
            "graph_enrichment: provider=%s chunks=%d new_edges=%d same_as_merges=%d",
            provider,
            calls_made,
            total_new,
            total_merges,
        )
        # Write enrichment-collapsed ik_keys to the merger's removal ledger so the
        # drift classifier can route them to drift_intended_removal rather than
        # drift_genuine_loss.  Only keys from SUCCESSFUL contractions are written
        # (failures were discarded from _pending before _collapsed_ik was updated).
        for _ik, _keep in _collapsed_ik.items():
            self.merger.removal_ledger[_ik] = {
                "reason": "enrichment_same_as",
                "merged_into": _keep,
            }
        # Reset the accumulator — any subsequent interim-rollover pass must
        # re-cross the floor before firing again.
        self._triples_since_last_enrichment = 0
        return {
            "chunks": calls_made,
            "new_edges": total_new,
            "same_as_merges": total_merges,
            "skipped": False,
            "skip_reason": None,
        }

    def consolidate_interim_graphs(self, *, housekeeping: bool = False) -> dict:
        """Full-cycle consolidation for simulate mode: merge interim graph.json sidecars.

        Called by ``_run_full_consolidation_sync`` when
        ``config.consolidation.mode == "simulate"``.  Simulate mode writes per-cycle
        graph.json files under ``<adapter_dir>/episodic/interim_<stamp>/graph.json``
        rather than PEFT adapter weights.  This method merges those interim graphs into
        the canonical main-tier graph at ``<adapter_dir>/episodic/graph.json`` and
        removes the interim slot directories.

        Architectural symmetry with the train fold: grooming is LITERALLY identical
        (``canonical()`` node identity + Case-1/Case-2 dedup via ``self.merger``,
        then ``_run_graph_enrichment``).  The ONLY permitted divergence is the
        persistence/retrain tail:

        - **source**: disk-load (``load_memory_from_disk``) vs reconstruct-from-weights.
        - **sink**: write ``graph.json`` (``save_memory_to_disk``) vs retrain adapters.

        The cross-slot merge routes through
        ``self.merger.merge(_synthetic_session, additive=True)`` so
        ``canonical()`` + Case-1/Case-2 dedup apply identically to the train fold.
        ``additive=True`` skips the model-gated Case-2 branch, so no model is
        required (the simulate merger typically holds ``model=None``).

        Args:
            housekeeping: When ``True``, bypass the empty-interim-slot noop so the
                enriched graph is persisted even with no interims present.  Used by
                :meth:`run_housekeeping` for on-demand re-grooming.  Sessions are
                NOT marked consolidated; window stamp is not advanced.

        Steps:

        1. Reset the merger and build ``Relation`` objects from the disk-loaded main
           graph AND every interim ``graph.json`` slot, then merge them via
           ``self.merger.merge(additive=True)`` — the same topology the train fold uses.
        2. Optional SOTA enrichment (same guard as ``consolidate_interim_adapters``).
           Runs AFTER the merge on the freshly-merged ``self.merger.graph``, mirroring
           the train fold's re-merge → enrichment ordering.
        3. Write the enriched, merged (canonicalized, deduped) graph to
           ``<adapter_dir>/episodic/graph.json`` via
           :func:`paramem.memory.persistence.save_memory_to_disk`.
        4. Remove the interim slot directories that were merged.
        5. Return a result dict shaped like ``consolidate_interim_adapters``'s return
           (``tiers_rebuilt``, ``graph_drift_count``, etc.) for compatibility with the
           post-cycle bookkeeping in ``_run_full_consolidation_sync``.

        Returns:
            Result dict with keys matching ``consolidate_interim_adapters``'s schema
            so the caller (``_run_full_consolidation_sync``) can handle both paths
            with the same bookkeeping logic.  Simulate mode never rolls back, so
            ``rolled_back`` is always ``False``.
        """
        import shutil as _shutil

        from paramem.memory.interim_adapter import iter_interim_dirs
        from paramem.memory.persistence import (
            iter_entries,
            load_memory_from_disk,
            save_memory_to_disk,
        )

        adapter_dir = self.output_dir

        # Housekeeping debug-dir labeling: stamp the interim slot so the
        # housekeeping run's debug artifacts nest under a housekeeping-labelled
        # dir distinct from a scheduled fold's.  Reuses the existing nesting
        # mechanism (_current_interim_stamp → snapshot_dir_for nesting).
        if housekeeping:
            _hk_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            self._current_interim_stamp = f"housekeeping_{_hk_ts}"
        else:
            # Ensure no stale housekeeping stamp leaks into a scheduled run.
            self._current_interim_stamp = None  # type: ignore[assignment]

        # --- Seed merger from disk-loaded main graph + interim slots ---
        # Reset the merger's keying surface so provenance keying is unconditional
        # (mirrors the train fold's reset_graph() call before re-merging registry
        # relations).
        self.merger.reset_graph()

        episodic_main_path = adapter_dir / "episodic" / "graph.json"
        # Count edges BEFORE the merge to compute active_before for tier_delta.
        _main_graph_before = load_memory_from_disk(episodic_main_path)
        _active_before_count = _main_graph_before.number_of_edges()

        # Collect all Relation objects from the main graph + all interim slots.
        # For each graph entry, build a Relation with indexed_key so provenance
        # is carried through the merger onto the merged edge (same pattern as
        # _build_registry_true_relations used by the train fold).
        _all_relations: list[Relation] = []
        interim_dirs_merged: list[Path] = []

        for _entry in iter_entries(_main_graph_before):
            _pred = _entry.get("predicate", "")
            if not _pred:
                continue
            _all_relations.append(
                Relation(
                    subject=_entry["subject"],
                    predicate=_pred,
                    object=_entry["object"],
                    relation_type=_FALLBACK_RTYPE,  # graph.json edges carry no relation_type
                    confidence=1.0,
                    speaker_id=_entry.get("speaker_id", ""),
                    indexed_key=_entry["key"],
                )
            )

        for _interim_name, interim_dir in iter_interim_dirs(adapter_dir):
            interim_graph_path = interim_dir / "graph.json"
            if not interim_graph_path.exists():
                # Skip train-mode interim slots (no graph.json).
                continue
            slot_graph = load_memory_from_disk(interim_graph_path)
            for _entry in iter_entries(slot_graph):
                _pred = _entry.get("predicate", "")
                if not _pred:
                    continue
                _all_relations.append(
                    Relation(
                        subject=_entry["subject"],
                        predicate=_pred,
                        object=_entry["object"],
                        relation_type=_FALLBACK_RTYPE,  # type: ignore[arg-type]
                        confidence=1.0,
                        speaker_id=_entry.get("speaker_id", ""),
                        indexed_key=_entry["key"],
                    )
                )
            interim_dirs_merged.append(interim_dir)
            logger.debug(
                "consolidate_interim_graphs: queued %d entries from %s",
                slot_graph.number_of_edges(),
                interim_dir,
            )

        # Merge through the unified builder (model-free — additive=True skips
        # the only model-gated Case-2 branch; canonical() + Case-1 dedup are
        # deterministic).  _merge_registry_relations synthesises speaker
        # entities via _synth_speaker_entities so person nodes carry
        # speaker_id — the same path the train fold uses.
        if _all_relations:
            self._merge_registry_relations(
                _all_relations,
                session_id="__simulate_consolidation_merge__",
                log_label="relations via GraphMerger",
            )
            logger.info(
                "consolidate_interim_graphs: merged %d relations (%d interim slot(s))",
                len(_all_relations),
                len(interim_dirs_merged),
            )
        else:
            logger.debug("consolidate_interim_graphs: no relations to merge")

        # Debug: snapshot the merged graph.  Self-gated.
        self._debug_writer.on_fold_graph(self.merger, label="merged")

        # --- Optional SOTA graph enrichment ---
        # Runs AFTER the merge on the freshly-merged self.merger.graph, mirroring
        # the train fold's re-merge → enrichment ordering so both modes apply the
        # same grooming topology.  The external/SOTA boundary is handled inside
        # _run_graph_enrichment (per-chunk except catches network errors), so a
        # network failure degrades gracefully there.  A programming error here
        # propagates and aborts the fold — correct behaviour, same as train.
        _enrichment_result = self._run_graph_enrichment()
        if not _enrichment_result.get("skipped"):
            logger.info(
                "consolidate_interim_graphs: graph_enrichment complete:"
                " chunks=%d new_edges=%d same_as_merges=%d",
                _enrichment_result.get("chunks", 0),
                _enrichment_result.get("new_edges", 0),
                _enrichment_result.get("same_as_merges", 0),
            )

        # Debug: snapshot the enriched graph (after SOTA enrichment).  Self-gated.
        self._debug_writer.on_fold_graph(self.merger, label="enriched")

        # When housekeeping=True, ALWAYS persist the enriched graph back to disk —
        # even when no interim slots were present.  The scheduled path returns a noop
        # dict here (nothing changed), but the housekeeping path re-grooms the
        # persisted knowledge so it MUST write the merger's canonicalized graph back
        # to disk.
        if not housekeeping and not interim_dirs_merged:
            logger.info("consolidate_interim_graphs: no simulate-mode interim slots found — noop")
            # Reset the housekeeping stamp before returning.
            self._current_interim_stamp = None  # type: ignore[assignment]
            return {
                "tiers_rebuilt": [],
                "graph_drift_count": 0,
                "drift_deduplicated": 0,
                "drift_orphan": 0,
                "drift_genuine_loss": 0,
                "keys_per_tier": {},
                "recall_per_tier": {},
                "rolled_back": False,
                "rollback_tier": None,
            }

        # --- Write merged graph to main episodic slot ---
        episodic_main_path.parent.mkdir(parents=True, exist_ok=True)
        save_memory_to_disk(self.merger.graph, episodic_main_path)
        _after_count = self.merger.graph.number_of_edges()
        _collapsed_count = sum(
            1
            for _rl_entry in self.merger.removal_ledger.values()
            if _rl_entry.get("reason") == "dedup"
        )
        logger.info(
            "consolidate_interim_graphs: wrote merged graph to %s"
            " (%d edges, %d interim slot(s), %d dedup collapse(s))",
            episodic_main_path,
            _after_count,
            len(interim_dirs_merged),
            _collapsed_count,
        )

        # Per-tier delta for simulate path.
        # staled_by_reason is derived from merger.removal_ledger via _build_tier_delta;
        # for simulate mode the entry store is empty (no KeyRegistry mutation) so
        # tier_of returns None for all removed keys and staled_by_reason == {} in
        # practice — a persistence-tail divergence, NOT a skipped grooming step.
        # The merger's Case-1 collapse STILL fires and is recorded in removal_ledger.
        _sim_tier_delta = self._build_tier_delta(
            active_before={"episodic": _active_before_count},
            active_after={"episodic": _after_count},
            minted_by_tier={"episodic": _enrichment_result.get("new_edges", 0)},
        )
        self._debug_writer.on_tier_delta(_sim_tier_delta)
        # Also emit the removal ledger so the pre_surfaces evidence is
        # visible in the simulate housekeeping artifacts.
        self._debug_writer.on_removal_ledger(getattr(self.merger, "removal_ledger", {}))

        # --- Remove merged interim slot directories ---
        for interim_dir in interim_dirs_merged:
            try:
                _shutil.rmtree(interim_dir, ignore_errors=True)
                logger.debug("consolidate_interim_graphs: removed interim slot %s", interim_dir)
            except Exception as _rm_exc:
                logger.warning(
                    "consolidate_interim_graphs: failed to remove %s: %s",
                    interim_dir,
                    _rm_exc,
                )

        # Reset housekeeping stamp.
        self._current_interim_stamp = None  # type: ignore[assignment]

        return {
            "tiers_rebuilt": ["episodic"],
            "graph_drift_count": _collapsed_count,
            "drift_deduplicated": _collapsed_count,
            "drift_orphan": 0,
            "drift_genuine_loss": 0,
            "keys_per_tier": {"episodic": _after_count},
            "recall_per_tier": {"episodic": 1.0},  # graph merge is lossless
            "rolled_back": False,
            "rollback_tier": None,
            "tier_delta": _sim_tier_delta,
        }

    def run_housekeeping(
        self,
        trainer=None,
        router=None,
        recall_sanity_threshold: "float | None" = None,
        refresh_epochs: int = 30,
        mode: str = "simulate",
    ) -> dict:
        """On-demand fold that re-grooms the canonical knowledge graph without advancing sessions.

        This is a thin dispatcher for the ``POST /consolidate/housekeeping`` endpoint.
        It routes to the correct underlying fold method depending on ``mode``:

        - **simulate**: calls :meth:`consolidate_interim_graphs` with ``housekeeping=True``.
          Re-runs the GraphMerger topology over the persisted main graph (with no interim
          slots required).  Writes the canonicalized/deduped graph back to disk.  Clears
          stale surface variants; emits ``tier_delta`` + ``removal_ledger`` (with
          ``pre_surfaces`` per discarded key) in the debug artifacts.  No model is required.
        - **train**: calls :meth:`consolidate_interim_adapters` with ``housekeeping=True``.
          Bypasses gate (d) (the per-tier floor accumulate guard) so the fold runs even
          when the active key count is below the floor.  Retrains adapter weights from
          registry-true content; emits ``tier_delta`` in the result and debug artifacts.
          The GPU lock must be held by the caller (same contract as the normal train fold).

        In both modes:
        - Sessions are NOT marked consolidated (window stamp not advanced).
        - ``_current_interim_stamp`` is set to ``"housekeeping_<ts>"`` during the fold
          so debug artifacts land under a housekeeping-labelled dir, distinct from a
          scheduled fold's artifacts.  Cleared on return.
        - The ``tier_delta`` key is present in the result dict and in the debug snapshot.

        Args:
            trainer: BackgroundTrainer (train mode only).  Must hold the GPU lock.
            router: Router instance for reload at fold completion (train mode only).
            recall_sanity_threshold: Override for the recall gate (train only).  When
                ``None`` (default), the value is read from
                ``self.config.recall_sanity_threshold``.
            refresh_epochs: Forwarded to the underlying adapter fold (train only).
            mode: Consolidation mode — ``"simulate"`` (default, no GPU/model required) or
                ``"train"`` (retrains adapter weights).  Passed by the app layer from
                ``config.consolidation.mode``; ``self.config`` (``ConsolidationConfig``)
                does not carry a ``mode`` attribute.

        Returns:
            Result dict from the underlying fold, with ``tier_delta`` guaranteed present.
        """
        if mode == "simulate":
            return self.consolidate_interim_graphs(housekeeping=True)
        else:
            # Read the window stamp that was written by the last scheduled fold and
            # pass it as window_stamp_override so _save_adapters re-writes the SAME
            # stamp onto every re-saved main slot.  This keeps _is_full_cycle_due's
            # identity check stable — a housekeeping run must not advance the cadence
            # window, only a scheduled fold may.
            from paramem.server.app import _last_full_consolidation_window as _lfw

            _preserved_stamp = _lfw(self.output_dir)
            return self.consolidate_interim_adapters(
                trainer=trainer,
                router=router,
                recall_sanity_threshold=recall_sanity_threshold,
                refresh_epochs=refresh_epochs,
                housekeeping=True,
                window_stamp_override=_preserved_stamp,
            )

    def _promote_mature_keys_inline(self) -> list[str]:
        """Promote episodic keys whose recurrence_count has reached the promotion threshold.

        Mirrors the logic of the removed ``server.consolidation._promote_mature_keys``
        helper but runs INSIDE the fold (``consolidate_interim_adapters``), AFTER the
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
        3. Promote keys whose ``recurrence_count`` >= ``self.config.promotion_threshold``
           by calling ``self.store.move(key, "semantic")`` then
           ``self.promoted_keys.add(key)``.
        4. Log decay candidates (keys whose ``last_seen_cycle`` is more than
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
            rec = bk.get("recurrence_count", 1)
            last = bk.get("last_seen_cycle", bk.get("first_seen_cycle", 0))

            if key in self.promoted_keys:
                continue

            if rec >= threshold:
                if self.store.has_simhash("episodic", key):
                    # Move entry + simhash + registry entry atomically to semantic.
                    self.store.move(key, "semantic")
                    newly_promoted.append(key)
                    logger.info(
                        "_promote_mature_keys_inline: key=%s promoted to semantic "
                        "(recurrence_count=%d >= threshold=%d)",
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
                    "(last_seen_cycle=%d, current_cycle=%d, window=%d)",
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
        default_speaker_id: str,
        defer: bool = False,
        tag_new: bool = False,
    ) -> "tuple[dict[str, int], list[dict]]":
        """Walk ALL merged-graph edges and populate *tier_keyed* with uniform entry dicts.

        Single unified edge→entry builder that subsumes the former three-step
        sequence of ``_harvest_keyless_edge_entries`` →
        ``_apply_keyless_edge_entries`` → ``_collect_keyed_edges_into``.

        **One pass, two branches per edge:**

        Keyless edges (no ``ik_key`` on the edge attribute, i.e. newly-extracted or
        SOTA-enrichment facts):
            - A key is minted via :meth:`_mint_keyed_entries` using a local running
              counter seeded from ``_indexed_next_index`` / ``_procedural_next_index``
              (the real counters are never touched until the write is committed).
            - ``speaker_id`` is resolved from the subject node's top-level
              ``speaker_id`` attribute; when the attribute key is ABSENT,
              ``default_speaker_id`` is used as the fallback (an explicit empty
              value is kept as ``""`` — this matches :meth:`_tag_speaker_id_defaults`
              semantics).
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
        synthetic fold sentinels excluded — B7-A provenance plumbing).

        The ``ik_key`` attribute is intentionally NOT stamped onto keyless edges
        (direct-append variant) to avoid the MultiDiGraph parallel-edge integer-key
        hazard.  Both the keyless and keyed branch guard their pass via
        ``if not key`` / ``if key`` rather than edge mutation, so the same edge
        object is safe to iterate once.

        Args:
            tier_keyed: Mutable mapping of tier name → list of training-entry dicts.
                Both branches append in-place.
            default_speaker_id: Fallback ``speaker_id`` for keyless edges whose
                subject node has no ``speaker_id`` attribute.  The fold passes ``""``
                (no cycle attribution); the interim path passes the cycle's
                ``speaker_id`` so unattributed nodes receive the cycle's attribution.
            defer: When ``True`` (interim path), all store writes and counter
                advances for NEW (keyless/minted) entries are deferred and returned
                in ``deferred_writes``.  Existing keyed entries are never written
                regardless of this flag.  Default ``False`` (fold discipline).
            tag_new: When ``True``, each minted entry receives ``entry["_new"] =
                True`` so the caller can identify newly-minted entries in
                ``tier_keyed``.  Default ``False``.

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
              synthetic fold sentinels excluded — B7-A provenance plumbing).

            Mutates *tier_keyed* in-place.  When ``defer=False``, also mutates
            the :class:`~paramem.memory.store.MemoryStore` and advances
            ``_indexed_next_index`` / ``_procedural_next_index`` for each minted key.
        """
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

        for _t_subj, _t_obj, _t_data in self.merger.graph.edges(data=True):
            key = _t_data.get(_IK_ATTR)
            pred = _t_data.get("predicate", "")
            if not pred:
                # Edges with no predicate are not keyable — skip unconditionally.
                continue

            if not key:
                # ---- Keyless branch: mint a new key ----
                # Read relation_type from the edge; clamp to valid schema values.
                _rt_raw = _t_data.get("relation_type", _FALLBACK_RTYPE)
                _rt: str = _rt_raw if _rt_raw in _VALID_RTYPES else _FALLBACK_RTYPE

                # Resolve display names from node attributes["name"].
                _subj_display = (
                    self.merger.graph.nodes[_t_subj].get("attributes", {}).get("name") or _t_subj
                )
                _obj_display = (
                    self.merger.graph.nodes[_t_obj].get("attributes", {}).get("name") or _t_obj
                )
                # Resolve speaker_id from the subject node's top-level attribute.
                # When the attribute key is ABSENT, use default_speaker_id.
                # When the attribute is PRESENT but empty (""), keep "" — the node
                # explicitly carries no speaker attribution.
                _node_attrs = self.merger.graph.nodes.get(_t_subj, {}) or {}
                _subj_sid_raw = _node_attrs.get("speaker_id", None)  # None = key absent
                _subj_sid = default_speaker_id if _subj_sid_raw is None else _subj_sid_raw

                # Derive tier via partition_relations (mirrors the keyed branch).
                _dummy = [
                    {
                        "subject": _subj_display,
                        "predicate": pred,
                        "object": _obj_display,
                        "relation_type": _rt,
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
                            "subject": _subj_display,
                            "predicate": pred,
                            "object": _obj_display,
                            "relation_type": _rt,
                            "speaker_id": _subj_sid,
                        }
                    ],
                    prefix=prefix,
                    start_index=start_index,
                    speaker_id=_subj_sid,
                    tag_new=tag_new,
                )

                # Advance the local counter for the chosen tier.
                if tier == "procedural":
                    _local_procedural += 1
                else:
                    _local_indexed += 1

                entry = minted[0]
                minted_key = entry["key"]
                # B7-A: source the contributing session ids from the merged
                # edge, excluding synthetic fold sentinels.  The result is a
                # sorted list of REAL session ids that contributed this fact.
                # This field is TRANSIENT — it rides the in-RAM record only;
                # it is never written to the persisted entry dict (store.put)
                # or to bookkeeping (store.set_bookkeeping).  The drop site
                # (step 11b) reads rec["session_ids"] to identify which
                # sessions contributed a recall-failed key.
                _rec_session_ids: list[str] = sorted(
                    set(_t_data.get("sessions", [])) - _SYNTHETIC_SESSION_IDS
                )
                rec = {
                    "entry": entry,
                    "tier": tier,
                    "canon_subj": _t_subj,
                    "canon_obj": _t_obj,
                    "predicate": pred,
                    "relation_type": _rt,
                    "speaker_id": _subj_sid,
                    "session_ids": _rec_session_ids,
                }

                if not defer:
                    # Fold discipline: persist immediately.
                    self.store.put(
                        tier,
                        minted_key,
                        entry,
                        simhash=compute_simhash(minted_key, _t_subj, pred, _t_obj),
                    )
                    self.store.set_bookkeeping(
                        minted_key,
                        speaker_id=_subj_sid,
                        first_seen_cycle=self.cycle_count,
                        relation_type=_rt,
                        recurrence_count=1,
                        last_seen_cycle=self.cycle_count,
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
                # The ik_key attribute is intentionally NOT stamped onto the edge so
                # the MultiDiGraph parallel-edge integer key field is not disturbed.
                tier_keyed[tier].append(
                    {
                        "key": minted_key,
                        "subject": entry["subject"],
                        "predicate": pred,
                        "object": entry["object"],
                        "speaker_id": _subj_sid,
                    }
                )
                minted_by_tier[tier] += 1

            else:
                # ---- Keyed branch: existing key, anti-forgetting replay ----
                entry = self.store.get(key)
                if entry is None:
                    # Key is registered but has no content entry — skip.
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

        For each key the content is sourced in priority order:

        1. ``store.get(key)`` — the content cache entry (subject/predicate/object).
        2. ``store.bookkeeping_for_key(key)`` — hydration-miss fallback: when
           ``store.get`` returns ``None`` for a LIVE active key (e.g. under
           ``boot_degraded``), but bookkeeping carries non-empty SPO, the key is
           still present and must NOT be silently dropped.  Log a warning; source
           SPO from bookkeeping.
        3. If both are absent or empty-SPO: skip and log (true orphan; the
           drift-orphan classification at the drift-partition site will bucket it).

        ``relation_type`` and ``speaker_id`` always come from bookkeeping (never
        from the entry payload which carries the merge-time value).

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
                # Hydration-miss — entry cache is empty for this live key.
                # Attempt bookkeeping SPO fallback before treating as orphan.
                subj = bk.get("subject", "")
                pred = bk.get("predicate", "")
                obj = bk.get("object", "")
                if subj or pred or obj:
                    logger.warning(
                        "_build_registry_true_relations: key=%s entry=None but "
                        "bookkeeping has SPO — sourcing from bookkeeping (boot_degraded?)",
                        key,
                    )
                else:
                    logger.debug(
                        "_build_registry_true_relations: key=%s has no entry and "
                        "no bookkeeping SPO — skipping (orphan)",
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
                )
            )
        return relations

    def _synth_speaker_entities(self, relations: "list[Relation]") -> "list":
        """Synthesise :class:`~paramem.graph.schema.Entity` objects for speaker-attributed subjects.

        For each :class:`Relation` in *relations* whose ``speaker_id`` is non-empty
        and whose ``subject`` equals ``speaker_id`` (the node key IS the speaker's
        system ID per ``_resolve_entity``), emit one
        :class:`~paramem.graph.schema.Entity` with ``entity_type="person"`` and the
        matching ``speaker_id``.  Deduplicates by ``speaker_id`` so exactly one
        entity is produced per unique speaker.

        Non-speaker subjects (``speaker_id == ""`` OR ``subject != speaker_id``) are
        skipped; their nodes retain no ``speaker_id`` attribute, which resolves to
        ``""`` in the keyed-walk (the correct default for non-person nodes).

        Used by :meth:`_merge_registry_relations` so that
        :meth:`~paramem.graph.merger.GraphMerger._upsert_entity` stamps
        ``speaker_id`` onto the subject node before the edge walk in
        :meth:`_build_all_edge_entries_into` reads it.  Without the entity the
        node would lack ``speaker_id``, causing minted keys to fall back to
        ``speaker_id=""`` (dcf4189 regression).

        Args:
            relations: The list of :class:`Relation` objects from which speaker
                entities are derived.  Both the recon path and the extra-relations
                (pending-session) path pass their respective relation lists here.

        Returns:
            A :class:`list` of :class:`~paramem.graph.schema.Entity` objects, one
            per unique speaker subject found in *relations*.  May be empty when no
            relation carries a non-empty ``speaker_id`` whose subject equals the
            speaker ID.
        """
        from paramem.graph.schema import Entity as _Entity

        _seen_speaker_ids: set[str] = set()
        entities: list[_Entity] = []
        for _r in relations:
            if _r.speaker_id and _r.speaker_id not in _seen_speaker_ids:
                # Only create an entity when the subject is the speaker node
                # (node key == speaker_id, set by _resolve_entity for speakers).
                # Non-speaker subjects are skipped; their nodes retain no speaker_id.
                if _r.subject == _r.speaker_id:
                    entities.append(
                        _Entity(
                            name=_r.subject,
                            entity_type="person",
                            speaker_id=_r.speaker_id,
                        )
                    )
                    _seen_speaker_ids.add(_r.speaker_id)
        return entities

    def _merge_registry_relations(
        self,
        relations: "list[Relation]",
        *,
        session_id: str,
        log_label: str,
    ) -> None:
        """Build a synthetic :class:`SessionGraph` from *relations* and merge it.

        This is the single shared builder for turning a ``list[Relation]`` into an
        entitied, merged :class:`SessionGraph`.  All three merge paths route through
        here: the recon path (``session_id="__full_consolidation_recon__"``), the
        extra-relations (pending-session) path
        (``session_id="__interim_pending_sessions__"``), and the simulate full-fold
        path (``session_id="__simulate_consolidation_merge__"``).  There is exactly
        ONE place that constructs a ``SessionGraph``.

        The entity list is synthesised from *relations* via
        :meth:`_synth_speaker_entities`, which stamps ``speaker_id`` onto speaker
        subject nodes.  This ensures the edge walk in
        :meth:`_build_all_edge_entries_into` reads the correct ``speaker_id`` from
        each node (dcf4189 invariant).  Before this unification the recon path used
        ``entities=[]``, causing reconstructed person nodes to be stored as
        ``entity_type="concept"`` with no ``speaker_id``; graph-enrichment then
        rooted facts at unattributed nodes.  The simulate path had the same latent
        bug: relations with ``speaker_id`` set produced person nodes that became
        concepts with no ``speaker_id`` because ``entities=[]`` was passed directly.

        The gradient-checkpointing guard is conditional on ``self.model`` being
        present.  Simulate callers may omit the model entirely (model-free paths);
        ``additive=True`` skips the model-gated branch regardless, so the guard is
        defensive-only for those callers.

        Returns early without side effects when *relations* is empty.

        Args:
            relations: The :class:`Relation` objects to merge.  May be the
                registry-true recon set, the pending extra-relations set, or the
                simulate fold's collected interim-slot relations.
            session_id: Synthetic session identifier passed to
                :class:`~paramem.graph.schema.SessionGraph`.  Used only for
                logging/debugging.
            log_label: Human-readable label for the count log line, e.g.
                ``"reconstructed triples"`` or ``"extra (pending-session) relations"``.
        """
        if not relations:
            return
        entities = self._synth_speaker_entities(relations)
        _session = SessionGraph(
            session_id=session_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            entities=entities,
            relations=relations,
        )
        # Disable gradient checkpointing: merger.merge may call model.generate()
        # for contradiction resolution when a model is present (CLAUDE.md rule
        # applies to ANY model.generate() site).  Guard is conditional so that
        # simulate callers where self.model is absent or None (model-free paths)
        # do not crash — additive=True already skips the model-gated branch, so
        # the guard is defensive-only for those callers.
        _has_model = getattr(self, "model", None) is not None
        if _has_model:
            self._disable_gradient_checkpointing()
        try:
            self.merger.merge(_session, additive=True)
        finally:
            if _has_model:
                self._enable_gradient_checkpointing()
        logger.info(
            "_materialize_consolidation_graph: re-merged %d %s into keying graph",
            len(relations),
            log_label,
        )

    def _materialize_consolidation_graph(
        self,
        *,
        tier: "str | None" = None,
        keys: "list[str] | None" = None,
        extra_relations: "list[Relation] | None" = None,
    ) -> "tuple[set[str], list[Relation]]":
        """Reconstruct active keys from adapter weights and re-merge registry-true relations.

        This is the *Materialize* stage of the fold pipeline:

        1. Probe every active key from adapter weights via :func:`reconstruct_graph`
           (``strict=False``).
        2. Compute ``recall_miss_keys`` — keys whose reconstructed SPO disagrees with
           registry-true SPO, or whose reconstruction failed outright.  The set is
           computed against ``store.all_active_keys()`` BEFORE the graph reset, so
           only registered keys can appear in the miss set.
        3. Reset the merger's keying graph to empty (``merger.reset_graph()``).
        4. Build registry-true :class:`Relation` objects via
           :meth:`_build_registry_true_relations` and re-merge them into the fresh
           keying graph inside a gradient-checkpointing guard.
        5. If ``extra_relations`` is supplied and non-empty, re-merge those relations
           into the fresh keying graph (additive, inside the same gradient-checkpointing
           guard).  This allows the interim mini-fold to inject the current cycle's
           pending-session relations alongside the slot's recalled registry-true keys.
        6. Emit debug snapshots ("reconstructed" before re-merge, "merged" after).

        **INVARIANT — extra_relations and the recall-miss set:**
        ``extra_relations`` participate in the MERGE / Case-1-adopt step ONLY.
        They MUST NOT enter the ``recall_miss_keys`` set.  That set is computed
        against ``store.all_active_keys()`` in step 2, BEFORE the reset — pending
        unregistered relations (not yet in the registry) therefore cannot distort it.
        Both ``extra_relations=None`` and ``extra_relations=[]`` are valid no-ops for
        the fold caller (fold passes ``None``; the check is ``if extra_relations``).

        **Speaker-ID note (unified path):** Both the recon path and the
        ``extra_relations`` path call :meth:`_merge_registry_relations`, which
        invokes :meth:`_synth_speaker_entities` to produce a synthetic
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
            tier: Forwarded to :func:`reconstruct_graph` as ``tier``.  When
                ``None`` (the default), all tiers are probed — byte-identical to
                the original inline fold behaviour.
            keys: Forwarded to :meth:`_build_registry_true_relations` as ``keys``.
                When ``None`` (the default), all active keys are processed —
                byte-identical to the original inline fold behaviour.
            extra_relations: Optional list of :class:`Relation` objects to merge
                into the fresh keying graph after the registry-true re-merge.
                Intended for the interim mini-fold: the caller captures the
                pending-session relations from ``self.merger.graph`` BEFORE calling
                this method (since the reset inside will wipe them) and passes them
                here so they survive the reset and co-reside with the slot's
                recalled facts.  The fold caller passes ``None`` (no-op).

        Returns:
            A 2-tuple ``(recall_miss_keys, recon_relations)`` where:

            - ``recall_miss_keys`` — :class:`set` of key strings that failed
              reconstruction or whose SPO diverged from the registry.
            - ``recon_relations`` — the :class:`list` of :class:`Relation` objects
              fed into the registry-true re-merge (registry-true SPO, with
              ``indexed_key`` set).  ``extra_relations`` are NOT included here —
              they travel through a separate merge call inside this method.
        """
        # --- Reconstruct all active keys from adapter weights ---
        # Probes every active key across all tiers; recovers (subject, predicate, object)
        # from the trained weights.  Reconstruction yields SPO ONLY — no relation_type.
        # strict=False: failures are logged and recorded in recon_result.failures; the
        # cycle continues with whatever SPO triples can be recovered.
        # Reconstruction is used ONLY to identify recall-miss keys (keys whose
        # reconstructed SPO disagrees with registry-true SPO, or whose reconstruction
        # failed outright).  A recall miss is a retry signal; the key stays in the
        # training set with its registry-true content.  It does NOT drop the key.
        recon_result = reconstruct_graph(self, tier=tier, strict=False)
        if recon_result.failures:
            logger.warning(
                "consolidate_interim_adapters: %d key(s) failed reconstruction "
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
        recall_miss_keys: set[str] = set(recon_result.failures)
        for _rh_key in self.store.all_active_keys():
            _rt_entry = self.store.get(_rh_key)
            _rt_subj = (_rt_entry or {}).get("subject", "") if _rt_entry else ""
            _rt_pred = (_rt_entry or {}).get("predicate", "") if _rt_entry else ""
            _rt_obj = (_rt_entry or {}).get("object", "") if _rt_entry else ""
            _recon_spo = _recon_spo_by_key.get(_rh_key)
            if _recon_spo is None:
                # No recon edge: counts as failure (already in recon_result.failures or missing).
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
                "consolidate_interim_adapters: %d key(s) in recall-miss set "
                "(kept in training with registry-true content): %s",
                len(recall_miss_keys),
                sorted(recall_miss_keys),
            )

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
            "consolidate_interim_adapters: keying graph reset to empty for the"
            " reconstruct→re-merge pass"
        )

        # --- Build merge input from registry-true SPO (NOT reconstruction) ---
        # Each relation carries its indexed_key so the key travels through
        # GraphMerger.merge() onto the merged edge (provenance keying).
        #
        # additive=True (fold-only): the merger never removes a registered edge during
        # the re-merge.  All registry-true keys coexist after the fold regardless
        # of same-(s,p)/different-o conflicts; REPLACE cardinality is skipped.
        # Two registry keys sharing identical (s,p,o) STILL fire Case-1 (the merger
        # identity is correct given correct inputs), and the collapsed key is recorded
        # in merger.collapsed.  The drift-partition step below soft-stales that key.
        # Debug: snapshot the reconstructed graph (before re-merge mutates the
        # keying surface).  Self-gated; no-op when save_cycle_snapshots=False.
        self._debug_writer.on_fold_graph(recon_result.graph, label="reconstructed")

        recon_relations: list[Relation] = self._build_registry_true_relations(keys=keys)

        # Merge registry-true reconstructed relations.  _merge_registry_relations
        # synthesises speaker entities from the relation list (same logic as the
        # extra-relations path below) so reconstructed person nodes receive
        # entity_type="person" + speaker_id from bookkeeping.  Before unification
        # this block used entities=[] → concept nodes with no speaker_id.
        self._merge_registry_relations(
            recon_relations,
            session_id="__full_consolidation_recon__",
            log_label="reconstructed triples",
        )

        # --- Re-merge extra_relations (interim mini-fold pending-session content) ---
        # INVARIANT: extra_relations participate in MERGE / Case-1-adopt ONLY.
        # They are NOT included in recall_miss_keys (computed above, before the reset).
        # extra_relations=None and extra_relations=[] are both valid no-ops (fold caller
        # passes None; interim passes the pending-session relations from merger.graph).
        self._merge_registry_relations(
            extra_relations or [],
            session_id="__interim_pending_sessions__",
            log_label="extra (pending-session) relations",
        )

        # Debug: snapshot the merged graph (after re-merge, before enrichment).
        # Emits even when recon_relations is empty so the fold always produces a
        # merged snapshot.  Self-gated; no-op when save_cycle_snapshots=False.
        self._debug_writer.on_fold_graph(self.merger, label="merged")

        return recall_miss_keys, recon_relations

    def _refine_consolidation_graph(
        self, recon_relations: "list[Relation]", *, enrich: bool = True
    ) -> None:
        """Run graph enrichment and recurrence bumps after the Materialize stage.

        This is the *Refine* stage of the fold pipeline:

        1. Optionally run SOTA graph enrichment via :meth:`_run_graph_enrichment`,
           which mutates ``self.merger.graph`` in place.  Enrichment is skipped
           when *enrich* is ``False`` (interim path under ``light`` or ``off``
           refinement level).
        2. Emit a debug snapshot ("enriched") after enrichment (or immediately
           when *enrich* is ``False``).
        3. If ``recon_relations`` is non-empty, scan ``merger.reinforcements`` for
           Case-1 duplicate-SPO collapses and call
           :meth:`~paramem.memory.store.MemoryStore.bump_recurrence` for each
           surviving key.  The recurrence-bump runs regardless of *enrich* — it
           reflects duplicate-SPO collapses from the merge, which happen at every
           level that merges.

        Args:
            recon_relations: The list of registry-true :class:`Relation` objects
                produced by :meth:`_materialize_consolidation_graph`.  Used only
                as a boolean guard — when empty, the recurrence-bump loop is
                skipped (no re-merge was performed so ``merger.reinforcements``
                will be empty too).
            enrich: When ``True`` (the default), run ``_run_graph_enrichment``
                before the recurrence-bump.  The full fold always passes the
                default.  The interim path passes ``enrich=(interim_refinement
                == "full")`` so SOTA enrichment is level-gated.
        """
        # --- Graph-level SOTA enrichment ---
        # Runs AFTER the re-merge so enrichment operates on the populated
        # reconstructed graph (not an empty one).  Under production defaults the
        # pre-fold graph was always empty, so enrichment silently skipped every cycle
        # at the node_count<10 floor; now it fires on the reconstructed knowledge.
        # Mutates self.merger.graph in place.  The external/SOTA boundary is handled
        # inside _run_graph_enrichment (per-chunk except catches network errors and
        # continues), so a network failure degrades gracefully there.  A programming
        # error here propagates and aborts the fold (sessions stay pending/retriable),
        # which is correct — better than silently dropping all enrichment.
        # Enrichment-added edges are keyless and are picked up by the pre-pass that
        # runs after inline-promotion.
        if enrich:
            enrichment_result = self._run_graph_enrichment()
            if not enrichment_result.get("skipped"):
                logger.info(
                    "graph_enrichment complete: chunks=%d new_edges=%d same_as_merges=%d",
                    enrichment_result.get("chunks", 0),
                    enrichment_result.get("new_edges", 0),
                    enrichment_result.get("same_as_merges", 0),
                )

        # Debug: snapshot the enriched graph (after SOTA enrichment, or immediately
        # when enrichment is skipped at this level).
        # Self-gated; no-op when save_cycle_snapshots=False.
        self._debug_writer.on_fold_graph(self.merger, label="enriched")

        if recon_relations:
            # --- Recurrence bump: Case-1 duplicate-SPO collapses ---
            # merger.reinforcements contains the surviving ik_key for every Case-1
            # collision fired during the re-merge.  A collision means two active keys
            # shared the same (s,p,o) — the incoming key drifts and the existing
            # edge's key is the survivor.  The survivor's recurrence_count represents
            # how many times this fact was independently extracted (and re-keyed)
            # across sessions before this fold collapsed the duplicates.
            _reinforced: set[str] = set()
            for _rein_key in getattr(self.merger, "reinforcements", []):
                if _rein_key and _rein_key not in _reinforced:
                    self.store.bump_recurrence(_rein_key, cycle=self.cycle_count)
                    _reinforced.add(_rein_key)
                    logger.debug(
                        "consolidate_interim_adapters: bumped recurrence for key=%s "
                        "(intra-fold duplicate-SPO collapse)",
                        _rein_key,
                    )

    def consolidate_interim_adapters(
        self,
        trainer=None,
        router=None,
        recall_sanity_threshold: "float | None" = None,
        refresh_epochs: int = 30,
        *,
        housekeeping: bool = False,
        window_stamp_override: "str | None" = None,
    ) -> dict:
        """Weekly refresh: collapse all episodic_interim_* adapters into the main tiers.

        This method MUST be invoked via BackgroundTrainer.submit() so that the
        GPU lock is held for the entire call duration.  It MUST NOT be called
        directly without the lock, and MUST NOT call gpu_lock_sync() internally
        (non-reentrant threading.Lock would deadlock).

        Invariants (enforced by the entry guard):
        - _gpu_thread_lock is held by the caller on entry.
        - The finalize block (registry rewrite → interim purge → router reload)
          runs as plain sequential statements — no training calls, no HF Trainer-
          driven routines.  Do not smuggle a training call into the finalize
          section; the abort-via-shutdown-predicate hook installed by
          ``train_adapter`` only fires at step boundaries inside HF Trainer's
          loop and would not protect non-training code paths.

        Steps:
        1. Verify the GPU lock is held (entry guard — leak-safe pattern).
        2. Reconstruct: probe every active key from adapter weights via
           ``reconstruct_graph``; recover (subject, predicate, object).
        3. Re-merge registry-true relations: for each active key source the
           authoritative (subject, predicate, object) from the registry (not from
           reconstruction), inject ``relation_type`` from persisted bookkeeping,
           then feed through ``merger.merge()`` so the cumulative graph edges are
           re-stamped.
        4. Optional SOTA graph enrichment (runs after re-merge on the populated
           graph so second-order relations are captured).
        5. Dedup + tier assignment: walk merged graph edges; for each edge, read
           the ``ik_key`` attribute stamped by the merger, derive tier from
           ``relation_type`` via ``partition_relations``; build ``tier_keyed``
           lists.  The merged graph is the dedup authority (duplicate triples
           collapse to one key).
        6. Load backup adapters (if available) and all interim adapters into PEFT.
        7. For each tier (episodic → semantic → procedural):
           a. Set active adapter to <tier>_backup before deleting the main.
           b. delete_adapter(<tier>) + create_adapter(<tier>).
           c. Set <tier>_is_training=True, call _train_adapter, set False.
           d. Recall-sanity check; on failure roll back and abort.
        8. Atomic finalize: registry rewrite → durable main-weight persist+verify
           (_save_adapters) → unload_interim_adapters → router reload.  The
           weight persist runs BEFORE the interim purge so the merged knowledge
           always has a verified on-disk copy before any interim slot is deleted
           (closes the full-cycle data-loss crash window).  _save_adapters runs
           a recall PROBE on the reloaded slot — it is not an HF Trainer routine,
           so the finalize invariant (no training calls) holds.
        9. On success: unload backup adapters.

        Args:
            trainer: BackgroundTrainer instance (must be the one holding the GPU
                lock via submit()).  Required for per-tier B2 re-arm pattern.
            router: Router instance whose reload() is called at the end of the
                atomic finalize sequence.  Optional — skipped when None.
            recall_sanity_threshold: Override for the minimum recall rate for the
                post-save disk-integrity probe in ``_save_adapters``.  When
                ``None`` (default), the value is read from
                ``self.config.recall_sanity_threshold``.
            refresh_epochs: Number of training epochs for the full per-tier
                rebuild (default 30; matches the per-key baseline from Tests 1-7b).
            housekeeping: When ``True``, bypass gate (d) (the whole-fold
                ``_total_trainable < _floor`` accumulating early-return) and
                proceed to retrain even when the total key count is below the
                per-tier floor.  Used by :meth:`run_housekeeping` so an
                on-demand grooming fold runs regardless of accumulation state.
                Sessions are NOT marked consolidated; window stamp is not advanced.
            window_stamp_override: When not ``None``, forwarded verbatim to
                :meth:`_save_adapters` so the saved main slots carry this value
                as ``window_stamp`` instead of the fresh floor computed at save
                time.  Used by :meth:`run_housekeeping` to preserve the window
                stamp that was already written by the last scheduled fold, so
                :func:`_is_full_cycle_due` is not perturbed by a housekeeping
                run (a housekeeping fold must not advance the cadence window).
                ``None`` is the correct value for every non-housekeeping call.

        Returns:
            Result dict with keys:
                {
                    "tiers_rebuilt": list[str],
                    # total absent keys (backward-compat)
                    "graph_drift_count": int,
                    # exact-SPO collapse; fact preserved
                    "drift_deduplicated": int,
                    # no SPO content; correctly dropped
                    "drift_orphan": int,
                    # reconstruction failure / hydration-miss
                    "drift_genuine_loss": int,
                    # merger-recorded intentional removal
                    "drift_intended_removal": int,
                    # reason → count breakdown
                    "drift_intended_removal_by_reason": dict,
                    "keys_per_tier": dict[str, int],
                    "recall_per_tier": dict[str, float],
                    "rolled_back": bool,
                    "rollback_tier": str | None,
                }
        """
        from paramem.memory.interim_adapter import unload_interim_adapters
        from paramem.models.loader import create_adapter
        from paramem.server.gpu_lock import _gpu_thread_lock

        # Resolve threshold from config when the caller did not supply an override.
        if recall_sanity_threshold is None:
            recall_sanity_threshold = self.config.recall_sanity_threshold

        # --- Entry guard: verify the GPU lock is held by the caller (leak-safe) ---
        acquired = _gpu_thread_lock.acquire(blocking=False)
        if acquired:
            # The lock was NOT held — we just accidentally acquired it ourselves.
            # Release immediately before raising so the process is recoverable.
            _gpu_thread_lock.release()
            raise RuntimeError(
                "consolidate_interim_adapters requires the caller to hold "
                "_gpu_thread_lock (submit via BackgroundTrainer.submit())"
            )

        # --- Housekeeping debug-dir labeling ---
        # Set a distinct interim stamp so debug artifacts for a housekeeping fold
        # land under ``housekeeping_<ts>/`` rather than a scheduled fold name.
        # Cleared in the finally-equivalent tail (after _save_adapters / rollback).
        if housekeeping:
            _hk_ts_adp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            self._current_interim_stamp = f"housekeeping_{_hk_ts_adp}"
        else:
            self._current_interim_stamp = None  # type: ignore[assignment]

        recall_miss_keys, recon_relations = self._materialize_consolidation_graph()
        self._refine_consolidation_graph(recon_relations)

        # --- Inline promotion: move matured episodic keys to semantic ---
        # Runs AFTER the recurrence-bump step (so bump-triggered threshold crossings
        # are captured) and BEFORE tier_keyed is built (so the promoted key lands
        # in tier_keyed["semantic"] and is trained into the semantic adapter this
        # fold).  Ordering invariant: reconstruction ran while the key was still in
        # episodic, so its weights were found there; only NOW is it
        # moved to semantic.
        _inline_promoted = self._promote_mature_keys_inline()
        if _inline_promoted:
            logger.info(
                "consolidate_interim_adapters: %d key(s) promoted to semantic "
                "before tier assignment",
                len(_inline_promoted),
            )

        # tier_keyed is initialised here (before the pre-pass) so that both
        # _build_all_edge_entries_into populates tier_keyed in one pass over all edges.
        tier_keyed: dict[str, list[dict]] = {
            "episodic": [],
            "semantic": [],
            "procedural": [],
        }

        # --- Edge-to-entry pass: mint new keys + collect existing keys ---
        # _build_all_edge_entries_into walks ALL merged-graph edges in one pass:
        #   - Keyless edges (SOTA-enrichment or pending facts): minted immediately
        #     (defer=False); the committed counter is advanced per minted key; the
        #     ik_key attribute is intentionally NOT stamped onto the edge so the
        #     MultiDiGraph parallel-edge integer key field is not disturbed.
        #   - Keyed edges (registry-true reconstructed facts): anti-forgetting replay
        #     entries sourced from the store; NOT counted as minted, NOT re-registered.
        # Both branches produce uniform {key, subject, predicate, object, speaker_id}
        # entries in tier_keyed.
        # Dedup note: when two active keys share identical (s,p,o), the re-merge fires
        # Case-1 — the surviving edge keeps its ik_key; the incoming key is recorded
        # in merger.collapsed (soft-staled later).  Only ONE edge survives per (s,p,o),
        # so only ONE key appears in tier_keyed — this is intended dedup, NOT data loss.
        minted_by_tier, _ = self._build_all_edge_entries_into(
            tier_keyed,
            default_speaker_id="",
            defer=False,
            tag_new=False,
        )

        if recall_miss_keys:
            logger.info(
                "consolidate_interim_adapters: %d key(s) in recall-miss set "
                "(retrained with registry-true content — not dropped): %s",
                len(recall_miss_keys),
                sorted(recall_miss_keys),
            )

        # Debug: snapshot the graph after the keyed-edge walk (before the floor-gate pass
        # mutates tier_keyed).  Self-gated; no-op when save_cycle_snapshots=False.
        self._debug_writer.on_fold_graph(self.merger, label="keyed")

        # --- Per-tier floor gate (must run BEFORE _all_keyed is computed) ---
        # This pass runs before _all_keyed is built so the post-park, post-graduation
        # serve_assignment is the source for all downstream consumers: _all_keyed,
        # last_seen refresh, and the soft-stale/drift partition.
        #
        # Two-map decoupling:
        #   serve_assignment — which tier SERVES/RECALLS each key (registry + simhash
        #                      owner).  This IS the returned "tier_keyed" layout.
        #   train_assignment — what each adapter TRAINS on.  For fast-start graduating
        #                      tiers, episodic trains on the graduating keys too (as
        #                      universal donor); the graduating tier trains on nothing
        #                      (empty list, copied from the informed episodic instead).
        # The two maps are IDENTICAL for all steady-state and parked keys; they diverge
        # ONLY for fast-start graduating tiers.  All downstream registry/drift/stale
        # consumers read serve_assignment; only jobs_by_tier reads train_assignment.
        #
        # Liveness signal — disk-slot presence (NOT store-tier label, NOT peft_config):
        #   _promote_mature_keys_inline store.move()s keys to "semantic" WITHOUT training
        #   the semantic adapter.  After promotion, store.tier_for_active_key(key) ==
        #   "semantic" even though the WEIGHTS are still in episodic.  Using the store
        #   tier for liveness would wrongly exclude semantic from fast-start (the
        #   weights-vs-store-tier conflation bug).  The correct signal is whether the
        #   tier's output_dir has any saved adapter slot written by a prior
        #   _save_adapters call.  A tier with NO disk slot was NEVER trained →
        #   first-cross graduation applies.  A tier WITH any disk slot was trained
        #   before → steady-state train normally.
        #
        # Episodic is the parking lot and is exempt from parking-into-episodic.
        # Episodic's own floor (whole-fold accumulate) is handled by the
        # accumulating early-return below.
        _floor = self.config.min_tier_key_floor
        # _fast_start_graduating: tiers graduating via copy-from-episodic this fold.
        # The per-tier loop skips train_adapter for these and copies instead.
        _fast_start_graduating: set[str] = set()

        # Helper: True when tier has at least one saved adapter slot on disk.
        # A slot name is a YYYYMMDD-HHMMSS timestamp (is_slot_name).  Using
        # disk-slot presence (not store-tier, not peft_config) avoids the
        # store-tier-vs-weights conflation for promotion-moved semantic keys.
        def _tier_has_disk_slot(tier_name: str) -> bool:
            from paramem.adapters.manifest import is_slot_name as _isn

            tier_dir = self.output_dir / tier_name
            if not tier_dir.is_dir():
                return False
            return any(entry.is_dir() and _isn(entry.name) for entry in tier_dir.iterdir())

        # serve_assignment: repurpose tier_keyed as the served layout after Pass-2.
        # The variable is renamed at the end of the pass; upstream keyed-edge-walk code keeps
        # tier_keyed as its local name until this rename point so no callers change.
        for _pt in ("semantic", "procedural"):
            _pt_entries = tier_keyed[_pt]
            if not _pt_entries:
                continue
            if len(_pt_entries) < _floor:
                # --- PARK: under floor — move to episodic ---
                logger.info(
                    "consolidate_interim_adapters: tier %s has %d key(s) < floor %d"
                    " — parking in episodic until floor reached",
                    _pt,
                    len(_pt_entries),
                    _floor,
                )
                for _pe in _pt_entries:
                    tier_keyed["episodic"].append(_pe)
                    # If the key is currently in a non-episodic store tier, move it back.
                    _current_store_tier = self.store.tier_for_active_key(_pe["key"])
                    if _current_store_tier is not None and _current_store_tier != "episodic":
                        self.store.move(_pe["key"], "episodic")
                    # Discard from promoted_keys so the key can be re-promoted on a
                    # future fold once the tier's population crosses the floor.
                    self.promoted_keys.discard(_pe["key"])
                tier_keyed[_pt] = []
            else:
                # At or above floor.  Check whether this tier is graduating for the
                # FIRST TIME — i.e. it has NO saved adapter slot on disk.  Liveness
                # uses disk-slot presence to avoid the store-tier-vs-weights conflation:
                # a promoted semantic key has store-tier "semantic" but its WEIGHTS are
                # still in episodic (promotion does NOT train the semantic adapter).
                _tier_is_live = _tier_has_disk_slot(_pt)
                if not _tier_is_live:
                    # First-time graduation: tier crossed floor and has no trained
                    # adapter on disk — episodic's weights are the valid donor.
                    if self.config.tier_fast_start:
                        # Strategy (b): copy episodic weights + rebook.  The actual
                        # copy runs in the per-tier loop; mark for fast-start here.
                        # Keys STAY in tier_keyed[_pt] (serve_assignment[_pt]) so the
                        # registry serves them from _pt.  NO store.move needed here —
                        # semantic keys are already store-tier semantic (promotion);
                        # procedural keys' deferred store.move runs at accept (:4563).
                        logger.info(
                            "consolidate_interim_adapters: tier %s graduating (fast-start)"
                            " — %d key(s) >= floor %d; will copy episodic weights",
                            _pt,
                            len(_pt_entries),
                            _floor,
                        )
                        _fast_start_graduating.add(_pt)
                    else:
                        # Strategy (a): move keys to their own store tier so they
                        # train from scratch under the tier's own adapter config.
                        # For semantic keys that are already store-tier "semantic" via
                        # promotion, store.move is a no-op (move to same tier).
                        logger.info(
                            "consolidate_interim_adapters: tier %s graduating"
                            " (train-from-scratch) — %d key(s) >= floor %d",
                            _pt,
                            len(_pt_entries),
                            _floor,
                        )
                        for _pe in _pt_entries:
                            self.store.move(_pe["key"], _pt)
                # (No action needed for already-live tiers: they are >= floor
                # and already have a disk slot — flow through normally.)

        # Rename tier_keyed to serve_assignment at the end of Pass-2 so all
        # downstream consumers read the served layout (not the training layout).
        # The local name tier_keyed is reused in the keyed-edge walk above to avoid
        # upstream churn; from this point on only serve_assignment is read by registry,
        # drift, and finalize consumers.
        serve_assignment = tier_keyed

        # Build train_assignment from serve_assignment: identical for all tiers
        # except fast-start graduating ones.  Episodic is the UNIVERSAL DONOR:
        # it trains on its own served keys PLUS every graduating tier's served keys,
        # so the copy that follows reads an episodic adapter that already encodes
        # all graduating keys.  A fast-start graduating tier trains on NOTHING (the
        # copy replaces training); the dedup is a construction property of the union
        # (graduating keys are in serve_assignment[T], NOT in serve_assignment["episodic"],
        # so the two sets are disjoint — no per-key guard required).
        train_assignment: dict[str, list[dict]] = {
            t: list(serve_assignment[t]) for t in ("episodic", "semantic", "procedural")
        }
        for _fst in _fast_start_graduating:
            # Build episodic's augmented training set as a dict keyed on entry["key"]
            # to guarantee dedup (in case an upstream double-insert ever occurs).
            _ep_union: dict[str, dict] = {e["key"]: e for e in train_assignment["episodic"]}
            for _fse in serve_assignment[_fst]:
                _ep_union.setdefault(_fse["key"], _fse)
            train_assignment["episodic"] = list(_ep_union.values())
            train_assignment[_fst] = []

        # Debug: persist serve/train tier assignment maps.  Self-gated.
        self._debug_writer.on_fold_assignments(serve_assignment, train_assignment)

        # --- Capture per-tier active_before for the tier_delta record ---
        # Must run BEFORE the whole-fold accumulate guard (which may early-return
        # without building tier_delta) and BEFORE training mutates the store so
        # the "before" snapshot is accurate.  The matching "active_after" snapshot
        # runs after _save_adapters in the finalize section.
        _train_active_before: dict[str, int] = {
            t: len(serve_assignment[t]) for t in ("episodic", "semantic", "procedural")
        }

        # --- Whole-fold accumulate guard ---
        # After Pass-2, any under-floor non-episodic tier is empty (parked into
        # episodic).  If episodic itself is also below the floor, the total
        # served set is too small — return accumulating without training.
        # Read serve_assignment, NOT train_assignment: graduating keys inflate the
        # episodic training set but the floor is about the SERVED key population.
        _total_trainable = sum(len(v) for v in serve_assignment.values())
        if not housekeeping and _total_trainable < _floor:
            logger.info(
                "consolidate_interim_adapters: total trainable keys %d < floor %d"
                " — returning accumulating (sessions stay pending)",
                _total_trainable,
                _floor,
            )
            _acc_parked = {
                t: len(serve_assignment[t])
                for t in ("semantic", "procedural")
                if len(serve_assignment[t]) > 0
            }
            # Defensive cleanup: remove any backup adapter left by a prior aborted
            # fold.  The accumulating early-return fires BEFORE this fold's per-tier
            # backup-creation loop, so there are no new backups to clean up here —
            # only pre-existing leftovers from an earlier run that crashed between
            # backup-creation and the success-commit cleanup.
            for _bname in ("episodic_backup", "semantic_backup", "procedural_backup"):
                if _bname in self.model.peft_config:
                    self.model.delete_adapter(_bname)
                    logger.debug(
                        "consolidate_interim_adapters: cleaned up stale backup %s"
                        " (left by prior aborted fold) on accumulating return",
                        _bname,
                    )
            # Reset housekeeping stamp (not cleared on success path at this point).
            self._current_interim_stamp = None  # type: ignore[assignment]
            return {
                "status": "accumulating",
                "accumulating_reason": {
                    "floor": _floor,
                    "parked": _acc_parked,
                    "episodic": len(serve_assignment["episodic"]),
                },
                "tiers_rebuilt": [],
                "graph_drift_count": 0,
                "drift_deduplicated": 0,
                "drift_orphan": 0,
                "drift_genuine_loss": 0,
                "drift_intended_removal": 0,
                "drift_intended_removal_by_reason": {},
                "recall_miss_keys": [],
                "keys_per_tier": {t: len(v) for t, v in serve_assignment.items()},
                "tier_keyed": serve_assignment,
                "recall_per_tier": {},
                "rolled_back": False,
                "rollback_tier": None,
            }

        # Compute drift: active keys with no surviving merged edge after the fold.
        # Under additive fold, a key drifts iff its recon edge was absent (recon
        # failure) — REPLACE cardinality is skipped at fold time so it can never
        # remove a registered edge.  No triple-identity matching — provenance from
        # the merged edge is the sole authority.
        # serve_assignment is the authoritative served layout; train_assignment may
        # contain additional graduating keys in episodic's training set but those
        # must NOT influence drift/stale accounting (they are served from tier T).
        _all_keyed = {e["key"] for tier_list in serve_assignment.values() for e in tier_list}

        # Refresh last_seen_cycle for every key that survived into tier_keyed.
        # A key whose adapter successfully recalled its fact this fold was
        # "seen" this cycle — refresh so decay never fires for stable keys.
        # This is separate from the recurrence bump (which only fires on
        # duplicate-SPO collapses); every surviving key gets a last_seen refresh.
        for _surviving_key in _all_keyed:
            _sbk = self.store.bookkeeping_for_key(_surviving_key)
            if _sbk is not None:
                _sbk["last_seen_cycle"] = self.cycle_count

        active_keys = self.store.all_active_keys()
        _drift_keys = [k for k in active_keys if k not in _all_keyed]

        # --- 3-way drift partition + soft-stale write-back ---
        # merger.collapsed records INCOMING keys that were deduplicated away in a
        # Case-1 duplicate-SPO collapse.  Because the merge input is registry-true,
        # a Case-1 collapse means TWO registry keys genuinely share the same (s,p,o).
        # The discarded key is SOFT-STALED: registry entry retained, simhash
        # retained, excluded from training.
        #
        # LOAD-BEARING ORDERING: resolve tier BEFORE flipping the key stale.
        # KeyRegistry.stale() removes the key from _active_keys, so
        # tier_for_active_key() called AFTER the flip returns None.
        _collapsed_set: set[str] = set(getattr(self.merger, "collapsed", []))
        # Removal ledger: records every edge removed by the merger (dedup,
        # contradiction, enrichment same_as) with a stable reason code.
        # Keyed by the removed edge's ik_key; values carry "reason" + per-reason detail.
        _ledger: dict[str, dict] = getattr(self.merger, "removal_ledger", {})

        drift_deduplicated: list[str] = []
        drift_orphan: list[str] = []
        # drift_genuine_loss: reconstruction-failed non-duplicate keys now in the
        # recall-miss/retry set.  Because the merge input is registry-true, these
        # keys stay in the training set so they are retrained; they should trend to
        # ~0.  Kept as a counter for monitoring but they are NOT dropped.
        drift_genuine_loss: list[str] = []
        # drift_intended_removal: keys removed by the merger for a known, intentional
        # reason (enrichment same_as contraction, contradiction resolution).  These
        # are RETAIN-ONLY — no staling, no tier-resolution.  Separate from dedup
        # which has its own soft-stale semantics.
        drift_intended_removal: list[str] = []
        drift_intended_removal_by_reason: dict[str, int] = {}

        # Per-tier dict capturing soft-staled keys for _reset_main_tier_registries_and_simhashes.
        # Maps tier -> {key: {"stale_since": ISO, "stale_cycles": int, "simhash": int | None}}.
        soft_stale_by_tier: dict[str, dict[str, dict]] = {}

        for _dk in _drift_keys:
            if _dk in _collapsed_set:
                drift_deduplicated.append(_dk)
                # LOAD-BEARING: resolve tier BEFORE staling (staling removes from _active_keys).
                _dk_tier = self.store.tier_for_active_key(_dk)
                # Capture current simhash BEFORE any registry mutation so the finalize
                # step can re-insert it into the rebuilt simhash dict.
                _dk_simhash: int | None = None
                if _dk_tier is not None:
                    _dk_simhash = self.store.simhash(_dk_tier, _dk)
                # Flip to stale in the live in-memory registry.
                self.store.discard_keys([_dk], mode="stale")
                # Record for the finalize step that rebuilds registries with stale-seeding.
                if _dk_tier is not None:
                    _stale_rec = {"stale_cycles": 0}
                    if _dk_simhash is not None:
                        _stale_rec["simhash"] = _dk_simhash
                    soft_stale_by_tier.setdefault(_dk_tier, {})[_dk] = _stale_rec
            elif _dk in _ledger:
                # Intended removal: the merger recorded a known-reason removal for
                # this key (enrichment same_as contraction or contradiction
                # resolution).  RETAIN-ONLY — no staling, no tier-resolution.
                # Note: dedup keys are ALSO in _ledger, but the _collapsed_set
                # branch above fires first for them (preserving soft-stale + R4
                # semantics).  Only non-collapsed ledger entries reach here.
                drift_intended_removal.append(_dk)
                _r = _ledger[_dk]["reason"]
                drift_intended_removal_by_reason[_r] = (
                    drift_intended_removal_by_reason.get(_r, 0) + 1
                )
            else:
                _dk_bk = self.store.bookkeeping_for_key(_dk)
                _dk_entry = self.store.get(_dk)
                # Distinguish hydration-miss from true orphan.
                # Orphan: entry is present but carries empty SPO (all-empty test).
                # Hydration-miss: entry is None but bookkeeping carries SPO — the
                # key is live; the cache merely failed to hydrate (boot_degraded).
                # In both cases we do NOT drop the key from training (it is already
                # in recon_relations via _build_registry_true_relations).
                _entry_subj = (_dk_entry or {}).get("subject", "")
                _entry_pred = (_dk_entry or {}).get("predicate", "")
                _entry_obj = (_dk_entry or {}).get("object", "")
                _bk_subj = (_dk_bk or {}).get("subject", "")
                _bk_pred = (_dk_bk or {}).get("predicate", "")
                _bk_obj = (_dk_bk or {}).get("object", "")
                if not _entry_subj and not _entry_pred and not _entry_obj:
                    if _bk_subj or _bk_pred or _bk_obj:
                        # Hydration-miss: no content entry but bookkeeping has SPO.
                        # Bucket as genuine_loss (recall miss, retry), not orphan.
                        drift_genuine_loss.append(_dk)
                    else:
                        drift_orphan.append(_dk)
                else:
                    # Registry-true content present but no merged edge: recall miss / retry.
                    drift_genuine_loss.append(_dk)

        graph_drift_count = len(_drift_keys)
        drift_deduplicated_count = len(drift_deduplicated)
        drift_orphan_count = len(drift_orphan)
        drift_genuine_loss_count = len(drift_genuine_loss)
        drift_intended_removal_count = len(drift_intended_removal)

        # R4 invariant: soft-stale keys must be disjoint from _all_keyed.
        # A key in both sets would be trained as active this fold AND written to
        # the stale partition — contradictory states that indicate tier_keyed[T]
        # was cleared after _all_keyed was computed, which would be a regression.
        _soft_stale_keys: set[str] = {
            k for tier_stale in soft_stale_by_tier.values() for k in tier_stale
        }
        _stale_in_active = _soft_stale_keys & _all_keyed
        if _stale_in_active:
            logger.warning(
                "consolidate_interim_adapters: R4 invariant violation — %d key(s) appear"
                " in both soft_stale_by_tier and _all_keyed (trained as active AND stale);"
                " this indicates tier_keyed was mutated after _all_keyed was built: %s",
                len(_stale_in_active),
                sorted(_stale_in_active),
            )

        # Per-key log lines name the bucket so the journal is self-explanatory.
        for _dk in drift_deduplicated:
            _dk_entry = self.store.get(_dk)
            logger.info(
                "graph_drift_key key=%s bucket=deduplicated subject=%r predicate=%r object=%r"
                " (registry-true duplicate — soft-staled; record retained for stale-echo seam)",
                _dk,
                (_dk_entry or {}).get("subject", ""),
                (_dk_entry or {}).get("predicate", ""),
                (_dk_entry or {}).get("object", ""),
            )
        for _dk in drift_orphan:
            logger.info(
                "graph_drift_key key=%s bucket=orphan (no subject/predicate/object content;"
                " correctly dropped)",
                _dk,
            )
        for _dk in drift_genuine_loss:
            _dk_entry = self.store.get(_dk)
            logger.info(
                "graph_drift_key key=%s bucket=genuine_loss subject=%r predicate=%r object=%r"
                " (reconstruction failure or hydration-miss — retrained with registry-true"
                " content; not a data loss)",
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
                "consolidate_interim_adapters: %d key(s) deduplicated (registry-true"
                " duplicate; soft-staled — record retained, excluded from training)",
                drift_deduplicated_count,
            )
        if drift_orphan_count:
            logger.info(
                "consolidate_interim_adapters: %d orphan key(s) dropped (no SPO content)",
                drift_orphan_count,
            )
        if drift_intended_removal_count:
            logger.info(
                "consolidate_interim_adapters: %d key(s) in intended_removal"
                " (merger-recorded removal: by_reason=%s)",
                drift_intended_removal_count,
                drift_intended_removal_by_reason,
            )

        # Gate the WARNING on genuine_loss only — dedup, orphan, and intended_removal
        # are expected/normal.  genuine_loss should trend to ~0 because the registry-
        # true merge input means a recall miss keeps the key in training rather than
        # dropping it.  genuine_loss strictly covers reconstruction failure or
        # hydration-miss — enrichment/contradiction removals are captured in
        # drift_intended_removal.
        if drift_genuine_loss_count > 0:
            logger.warning(
                "consolidate_interim_adapters: %d genuine reconstruction loss(es) — "
                "these keys had content but produced no merged edge (reconstruction"
                " failure or hydration-miss); they were retrained with registry-true"
                " content (should trend to ~0): %s",
                drift_genuine_loss_count,
                drift_genuine_loss,
            )

        logger.info(
            "consolidate_interim_adapters: key distribution — episodic=%d semantic=%d "
            "procedural=%d drift=%d (deduplicated=%d orphan=%d genuine_loss=%d"
            " intended_removal=%d)",
            len(serve_assignment["episodic"]),
            len(serve_assignment["semantic"]),
            len(serve_assignment["procedural"]),
            graph_drift_count,
            drift_deduplicated_count,
            drift_orphan_count,
            drift_genuine_loss_count,
            drift_intended_removal_count,
        )

        # Debug: persist the merger removal ledger — reset_graph cleared it before
        # the fold; re-merge + enrichment populated it.  Emitted here (always
        # reached) so the artifact exists even on a rolled-back or aborted fold.
        # Self-gated.
        self._debug_writer.on_removal_ledger(getattr(self.merger, "removal_ledger", {}))

        # --- Build per-tier TrainingJob objects ---
        # Must be constructed before any tier rebuild begins so that
        # inference_fallback_adapter is set to the backup adapter name.
        # Because the public train_adapter (not BackgroundTrainer._train_adapter)
        # is called here, _current_job is NOT updated automatically.  The
        # per-tier loop manually swaps trainer._current_job to the tier-specific
        # job before calling train_adapter so that pause() reads the correct
        # inference_fallback_adapter for each tier rebuild.
        from paramem.server.background_trainer import TrainingJob

        refresh_training_config = self._make_training_config(num_epochs=refresh_epochs)

        # train_assignment drives each adapter's training set.  For fast-start
        # graduating tiers, train_assignment[T] == [] (the tier is copied from the
        # informed episodic, not trained); episodic's set is augmented with all
        # graduating keys so the copy donor is informed before the copy reads it.
        # All other tiers: train_assignment[T] == serve_assignment[T].
        jobs_by_tier = {
            "episodic": TrainingJob(
                entries=train_assignment["episodic"],
                adapter_name="episodic",
                adapter_config=self.episodic_config,
                inference_fallback_adapter="episodic_backup",
            ),
            "semantic": TrainingJob(
                entries=train_assignment["semantic"],
                adapter_name="semantic",
                adapter_config=self.semantic_config,
                inference_fallback_adapter="semantic_backup",
            ),
            "procedural": TrainingJob(
                entries=train_assignment["procedural"],
                adapter_name="procedural",
                adapter_config=self.procedural_config or self.episodic_config,
                inference_fallback_adapter="procedural_backup",
            ),
        }

        # --- Per-tier fresh-adapter rebuild ---
        # Load backup adapters (PEFT load-or-skip idempotency).
        # In the absence of a real encrypted backup dir we skip the load
        # and fall back to the live main adapters as the de-facto backup.
        # Production systems should implement an encrypted backup path for
        # durable out-of-process backup.
        #
        # Each backup MUST be created from its source tier's config — the tiers
        # have heterogeneous LoRA shapes (procedural targets attn+mlp,
        # episodic/semantic target attn-only) so a single template can't serve
        # all three. ``copy_adapter_weights`` checks parameter-set equality and
        # would otherwise raise on procedural → procedural_backup.
        tier_config_for_backup = {
            "episodic": self.episodic_config,
            "semantic": self.semantic_config,
            "procedural": self.procedural_config or self.episodic_config,
        }
        for backup_name in ("episodic_backup", "semantic_backup", "procedural_backup"):
            if backup_name not in self.model.peft_config:
                # No backup available — copy main weights into backup slot.
                # This keeps PEFT adapter count ≥ 5 throughout the rebuild.
                base_tier = backup_name.replace("_backup", "")
                if base_tier in self.model.peft_config:
                    from paramem.models.loader import copy_adapter_weights

                    backup_config = tier_config_for_backup[base_tier]
                    self.model = create_adapter(self.model, backup_config, backup_name)
                    copy_adapter_weights(self.model, src=base_tier, dst=backup_name)
                    logger.info(
                        "consolidate_interim_adapters: created in-memory backup %s from %s",
                        backup_name,
                        base_tier,
                    )

        tiers_rebuilt: list[str] = []
        recall_per_tier: dict[str, float] = {}
        # Per-tier per-key verdict from the forced final-epoch fill probe.
        # None for a tier that was skipped (no entries) or trained without
        # a recall callback (early-stop disabled).  Passed into
        # _reset_main_tier_registries_and_simhashes to filter registration.
        last_per_key_by_tier: dict[str, list | None] = {}

        for tier in ("episodic", "semantic", "procedural"):
            backup_name = f"{tier}_backup"
            job = jobs_by_tier[tier]

            # A fast-start graduating tier has train_assignment[T] == [] (it is
            # copied from the informed episodic, not trained) but MUST reach the
            # copy block below; skip only non-graduating empty-train tiers.
            # Invariant: a fast-start graduating tier always has
            # serve_assignment[T] >= floor, so the probe denominator is never 0.
            if not job.entries and tier not in _fast_start_graduating:
                logger.info(
                    "consolidate_interim_adapters: no keys for tier %s — skipping rebuild", tier
                )
                continue

            # a. Switch active adapter away from the tier being rebuilt so
            #    delete_adapter does not fire a PEFT UserWarning and auto-pick.
            if backup_name in self.model.peft_config:
                from paramem.models.loader import switch_adapter

                switch_adapter(self.model, backup_name)

            # b. delete_adapter(<tier>) + create_adapter(<tier>) — safe because
            #    ≥5 adapters remain at this point ({2 other mains + 3 backups +
            #    N interims}).
            if tier in self.model.peft_config:
                self.model.delete_adapter(tier)
                logger.debug("consolidate_interim_adapters: deleted adapter %s", tier)

            tier_cfg = (
                self.episodic_config
                if tier == "episodic"
                else (
                    self.semantic_config
                    if tier == "semantic"
                    else (self.procedural_config or self.episodic_config)
                )
            )
            self.model = create_adapter(self.model, tier_cfg, tier)
            logger.debug("consolidate_interim_adapters: created fresh adapter %s", tier)

            # c. Set <tier> active for training.
            from paramem.models.loader import switch_adapter as _sw

            _sw(self.model, tier)

            # --- Fast-start graduation branch (strategy (b), R2 pre-save probe) ---
            # When this tier graduated for the first time this fold via copy-from-
            # episodic, copy the episodic adapter's LoRA weights into the fresh tier
            # adapter, run a pre-save recall probe (R2), and — if the probe passes —
            # rebook the keys into the tier's registry and add to tiers_rebuilt
            # WITHOUT calling train_adapter.  If the probe fails, fall through to
            # the normal train-from-scratch path (strategy (a) fall-back).
            if tier in _fast_start_graduating:
                from paramem.models.loader import copy_adapter_weights as _copy_aw
                from paramem.models.loader import (
                    copy_adapter_weights_subset as _copy_aw_subset,
                )

                # Semantic and episodic share the same 4 attn modules (equal param
                # sets); procedural has 7 modules (attn+mlp) — use subset copy.
                if tier == "procedural":
                    _copy_aw_subset(self.model, src="episodic", dst=tier)
                    logger.info(
                        "consolidate_interim_adapters: fast-start graduation — "
                        "copied episodic attn weights into procedural (mlp stays zero-init)"
                    )
                else:
                    _copy_aw(self.model, src="episodic", dst=tier)
                    logger.info(
                        "consolidate_interim_adapters: fast-start graduation — "
                        "copied episodic weights into %s (full param set)",
                        tier,
                    )

                # R2: pre-save recall probe on the copied adapter (tier is now active).
                # Use serve_assignment[tier] as the probe target — NOT job.entries, which
                # is train_assignment[tier] == [] for a fast-start graduating tier.
                # serve_assignment[tier] holds the keys T must recall after the copy.
                _serve_entries = serve_assignment[tier]
                _probe_passing = self._probe_passing_keys(tier, _serve_entries)
                _probe_rate = len(_probe_passing) / len(_serve_entries) if _serve_entries else 1.0
                logger.info(
                    "consolidate_interim_adapters: fast-start graduation probe %s"
                    " — %d/%d passed (%.3f), threshold %.3f",
                    tier,
                    len(_probe_passing),
                    len(_serve_entries),
                    _probe_rate,
                    recall_sanity_threshold,
                )

                if _probe_rate >= recall_sanity_threshold:
                    # Copy recalled at or above threshold: rebook keys and accept.
                    # Iterate serve_assignment[tier], NOT job.entries (empty for fast-start).
                    # For procedural: keys are still store-tier "episodic" (parked) →
                    #   store.move genuinely rebooks them to "procedural".
                    # For semantic: keys are already store-tier "semantic" (promotion) →
                    #   store.move is a no-op (store.py:372-376), harmlessly elided.
                    for _fse in _serve_entries:
                        self.store.move(_fse["key"], tier)
                    # Record None verdict so _reset_main_tier_registries_and_simhashes
                    # falls into its _probe_passing_keys fail-safe, which will re-probe
                    # the copied weights and admit only the passing keys.
                    last_per_key_by_tier[tier] = None
                    tiers_rebuilt.append(tier)
                    logger.info(
                        "consolidate_interim_adapters: fast-start graduation accepted"
                        " for %s (%d keys rebooked)",
                        tier,
                        len(_serve_entries),
                    )
                    continue  # skip train_adapter for this tier

                # Probe failed: fall back to strategy (a) — delete the copied adapter,
                # recreate it fresh, and let the normal train path below handle it.
                logger.warning(
                    "consolidate_interim_adapters: fast-start probe FAILED for %s"
                    " (%.3f < %.3f) — falling back to train-from-scratch",
                    tier,
                    _probe_rate,
                    recall_sanity_threshold,
                )
                _fast_start_graduating.discard(tier)
                # Restore train entries from serve set so the fallback train path
                # trains T on its full ≥N serve set (train_assignment[T] was [] for
                # fast-start; job.entries must be updated before format_entry_training).
                job.entries = list(_serve_entries)
                # Move keys to their own store tier so train-from-scratch uses the
                # correct config and the registry reflects the new tier owner.
                # For semantic keys already store-tier "semantic", store.move is no-op.
                for _fse in _serve_entries:
                    self.store.move(_fse["key"], tier)
                # Recreate the adapter fresh (the copy corrupted its init).
                if tier in self.model.peft_config:
                    if backup_name in self.model.peft_config:
                        _sw(self.model, backup_name)
                    self.model.delete_adapter(tier)
                self.model = create_adapter(self.model, tier_cfg, tier)
                _sw(self.model, tier)
                logger.debug(
                    "consolidate_interim_adapters: recreated fresh adapter %s"
                    " for fallback training",
                    tier,
                )

            # d. B2 re-arm: flip _is_training True BEFORE _train_adapter,
            #    False in finally so it fires on success AND failure.
            #    Also swap _current_job to the per-tier job so that pause()
            #    reads the correct inference_fallback_adapter (e.g.
            #    "episodic_backup") during this tier's rebuild.  The prior
            #    sentinel (installed by _run_callable_queue) is restored in
            #    the finally block to preserve the outer invariant.
            prior_job = None
            recall_state = None  # initialise so the capture below is always defined
            if trainer is not None:
                prior_job = trainer._current_job
                trainer._current_job = job
                trainer._set_is_training(True)
            try:
                # Format, build recall callback, and train — shared seam.
                # Fresh recall callback per tier — _signaled_stop and
                # epoch_log do not leak across tiers.
                _tier_metrics, recall_state = self._train_tier_adapter(
                    job.entries,
                    adapter_name=tier,
                    adapter_config=tier_cfg,
                    training_config=refresh_training_config,
                    output_dir=self.output_dir / "consolidation_refresh" / tier,
                    run_name=f"consolidate-{tier}",
                    phase_name=f"consolidate-{tier}",
                    num_epochs=refresh_epochs,
                )
                if _tier_metrics is not None:
                    if _tier_metrics.get("aborted"):
                        # Training was aborted for inference mid-tier.  Restore
                        # all three production tiers from their backup slots so
                        # VRAM and weights are consistent with the pre-cycle
                        # baseline, then raise AbortedDuringConsolidation so the
                        # outer caller (app.py _run_full_cycle) can log and skip
                        # the atomic finalize step.
                        logger.info(
                            "consolidate_interim_adapters: training aborted on tier %s "
                            "— restoring all tiers from backups",
                            tier,
                        )
                        from paramem.models.loader import copy_adapter_weights as _copy_w

                        for _t in ("episodic", "semantic", "procedural"):
                            _backup = f"{_t}_backup"
                            if _backup in self.model.peft_config and _t in self.model.peft_config:
                                _copy_w(self.model, src=_backup, dst=_t)
                        # Clear the housekeeping stamp before propagating so the next
                        # call does not inherit a stale label (self-heals on re-entry
                        # anyway, but explicit clear is more robust).
                        self._current_interim_stamp = None  # type: ignore[assignment]
                        raise AbortedDuringConsolidation(f"training aborted on tier {tier!r}")
                    else:
                        logger.info(
                            "consolidate_interim_adapters: trained %s on %d keys",
                            tier,
                            len(job.entries),
                        )
            finally:
                if trainer is not None:
                    trainer._set_is_training(False)
                    trainer._current_job = prior_job

            # Disk-integrity for this tier is gated post-save by _save_adapters
            # (via _verify_saved_adapter_from_disk on each saved slot).  The
            # pre-save in-RAM recall probe has been removed; a post-save failure
            # propagates as RuntimeError from _save_adapters so the caller
            # (app.py _run_full_cycle) skips mark_consolidated and sessions stay
            # pending for the next cycle.
            # Capture the per-key verdict from the forced final-epoch fill probe
            # so _reset_main_tier_registries_and_simhashes can filter by recall.
            last_per_key_by_tier[tier] = (
                recall_state.last_per_key if recall_state is not None else None
            )
            if recall_state is not None and recall_state.last_per_key is not None:
                self._debug_writer.on_recall_probe(
                    recall_state.last_per_key,
                    phase="train_fill",
                    adapter_name=tier,
                )
            tiers_rebuilt.append(tier)

        # flip off _is_training at finalize entry (belt-and-braces).
        if trainer is not None:
            trainer._set_is_training(False)

        # --- Atomic finalize ---
        # Invariant: registry rewrite FIRST, Router.reload() LAST.
        # The finalize block runs purely as registry / disk / PEFT / router ops.
        # No _train_adapter() call, no HF Trainer-driven routine may appear here.

        # Registry rewrite (MUST be first).
        # Rebuild per-tier registries from the authoritative tier_keyed layout:
        # each tier gets a fresh KeyRegistry containing only the keys that belong
        # to it after re-derivation from the cumulative graph.  Interim-tier
        # registries are then dropped (they are superseded by the main tiers).
        if self.store.replay_enabled:
            # Build per-tier passing-sets from the captured final-epoch verdict.
            # _recall_passing_keys returns None when the verdict is absent (disabled
            # early-stop or skipped tier); _reset_main_tier_registries_and_simhashes
            # will call _probe_passing_keys for those tiers as the fail-safe.
            passing_sets_by_tier: dict[str, set[str] | None] = {}
            for _tier in ("episodic", "semantic", "procedural"):
                _lpk = last_per_key_by_tier.get(_tier)
                if _lpk is not None:
                    # Intersect with serve_assignment[_tier] so graduating keys whose
                    # verdicts appear in episodic's last_per_key (because episodic
                    # trained on the augmented set) cannot enter episodic's registry
                    # or inflate the recall-gate log.  For semantic/procedural the
                    # intersection is a no-op (they trained on their own serve set);
                    # applying it uniformly is the clean invariant.
                    _serve_keys = {e["key"] for e in serve_assignment[_tier]}
                    passing_sets_by_tier[_tier] = {
                        r["key"] for r in _lpk if r["exact_match"]
                    } & _serve_keys
                else:
                    passing_sets_by_tier[_tier] = None

            # Reset each main tier's registry AND simhash to the post-consolidation
            # membership (the pairing is load-bearing — see the helper's docstring).
            # Thread soft_stale_by_tier so the rebuilt registry seeds the stale
            # partition (B1 fix — a bare new KeyRegistry() would wipe it).
            self._reset_main_tier_registries_and_simhashes(
                serve_assignment,
                passing_sets_by_tier,
                soft_stale_by_tier=soft_stale_by_tier,
            )
            # Drop interim-tier registries — they are no longer valid.
            self._drop_interim_tier_registries()
            # Save each main tier's registry unconditionally (create dir if needed).
            for _reg_tier in ("episodic", "semantic", "procedural"):
                _reg_tier_dir = self.output_dir / _reg_tier
                _reg_tier_dir.mkdir(parents=True, exist_ok=True)
                _reg_path = _reg_tier_dir / "indexed_key_registry.json"
                self.store.registry(_reg_tier).save(_reg_path)
                logger.info(
                    "consolidate_interim_adapters: registry rewritten to %s",
                    _reg_path,
                )

        # Persist + verify the merged main-tier weights BEFORE purging the interim slots
        # (crash-window guard — MUST precede the interim purge below).
        # The interim slot dirs hold the ONLY durable copy of the folded
        # knowledge until the rebuilt main weights are written+verified to disk.
        # _save_adapters() writes the main adapter weights + manifest + registry
        # and runs a post-save disk-integrity verify that RAISES on a corrupt /
        # partial artifact.  Persisting here — BEFORE the interim purge below —
        # closes the data-loss window: if the save (or its verify) fails, the
        # exception propagates, the purge does NOT run, the interim slots
        # survive, and the cycle stays retriable.  On-disk result of an
        # interrupted fold is therefore always "interims present OR main fully
        # persisted", never neither.  Mirrors the simulate-mode ordering in
        # consolidate_interim_graphs (merged graph written before interim rmtree).
        #
        # Gated on tiers_rebuilt: a no-op fold (nothing trained) has no fresh
        # weights to stamp — re-saving zero-init slots would corrupt the live
        # main manifests (the same precondition the caller honored at the
        # orchestration layer).
        if self.store.replay_enabled and tiers_rebuilt:
            try:
                self._save_adapters(
                    window_stamp_override=window_stamp_override,
                )
            except Exception:
                # Clear the housekeeping stamp before propagating so the next call
                # does not inherit a stale label.  try/finally is not used here
                # because the success path does its own explicit clear below and
                # the exception must propagate unchanged (crash-window guard:
                # callers skip mark_consolidated so sessions stay pending).
                self._current_interim_stamp = None  # type: ignore[assignment]
                raise
            logger.info("consolidate_interim_adapters: merged main weights persisted+verified")

        # Increment stale_cycles AFTER the durable write.
        # A fold that aborts before the durable write does NOT advance decay on
        # un-persisted stale sets.  A key staled in this fold has stale_cycles=0
        # at the durable write; stale_cycles=1 after this call (unobservable until
        # the NEXT fold reads it from disk).
        if self.store.replay_enabled and soft_stale_by_tier:
            for _st_tier in ("episodic", "semantic", "procedural"):
                # registry() always returns a real KeyRegistry when replay is
                # enabled (registry-always-exists invariant; the None guard
                # was dead and is removed per plan F6).
                self.store.registry(_st_tier).increment_stale_cycles()
            logger.debug(
                "consolidate_interim_adapters: stale_cycles advanced for %d soft-staled key(s)",
                sum(len(v) for v in soft_stale_by_tier.values()),
            )

        # Interim purge — single call covers both PEFT delete AND on-disk rmtree.
        unload_interim_adapters(self.model, self.output_dir)
        logger.info("consolidate_interim_adapters: interim adapters unloaded")

        # Router reload (MUST be last).
        if router is not None:
            try:
                router.reload()
                logger.info("consolidate_interim_adapters: router reloaded")
            except Exception:
                logger.exception("consolidate_interim_adapters: router reload failed")

        # --- Success commit: unload backup adapters ---
        # The three main adapters remain loaded throughout; only the backups are cleaned up.
        for backup_name in ("episodic_backup", "semantic_backup", "procedural_backup"):
            if backup_name in self.model.peft_config:
                self.model.delete_adapter(backup_name)
                logger.debug(
                    "consolidate_interim_adapters: unloaded backup adapter %s",
                    backup_name,
                )

        # Restore episodic as the active adapter for subsequent inference.
        if "episodic" in self.model.peft_config:
            from paramem.models.loader import switch_adapter as _sw2

            _sw2(self.model, "episodic")

        # --- Per-tier delta for train mode ---
        # Build the tier_delta record from the before-snapshot + serve_assignment
        # (active after) + minted_by_tier from _build_all_edge_entries_into.
        # staled_by_reason is derived from merger.removal_ledger via _build_tier_delta
        # (includes all removal reasons: dedup, enrichment_same_as, contradiction_*).
        _train_tiers = ("episodic", "semantic", "procedural")
        _train_tier_delta = self._build_tier_delta(
            active_before=_train_active_before,
            active_after={t: len(serve_assignment.get(t, [])) for t in _train_tiers},
            minted_by_tier=minted_by_tier,
        )
        self._debug_writer.on_tier_delta(_train_tier_delta)

        # Reset housekeeping interim stamp.
        self._current_interim_stamp = None  # type: ignore[assignment]

        logger.info(
            "consolidate_interim_adapters: complete — rebuilt %s, drift=%d"
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
            # graph_drift_count: total keys absent from tier_keyed (backward-compat).
            "graph_drift_count": graph_drift_count,
            # 4-way breakdown — callers that want accurate data-loss signal use these.
            "drift_deduplicated": drift_deduplicated_count,
            "drift_orphan": drift_orphan_count,
            # genuine_loss = reconstruction-failure / hydration-miss keys (not dropped;
            # retrained); should trend to ~0 as the store stabilises.
            "drift_genuine_loss": drift_genuine_loss_count,
            # Merger-recorded intentional removals (enrichment, contradiction).
            # RETAIN-ONLY: keys are not staled, not dropped, not trained (absorbed).
            "drift_intended_removal": drift_intended_removal_count,
            "drift_intended_removal_by_reason": drift_intended_removal_by_reason,
            # recall_miss_keys: keys whose reconstructed SPO disagreed with registry-true
            # SPO, or whose reconstruction failed.  Retry signal; keys were retrained.
            "recall_miss_keys": sorted(recall_miss_keys),
            "keys_per_tier": {t: len(v) for t, v in serve_assignment.items()},
            "tier_keyed": serve_assignment,
            "recall_per_tier": recall_per_tier,
            "rolled_back": False,
            "rollback_tier": None,
            "tier_delta": _train_tier_delta,
        }

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
                post-session training path.
            debug_phase: When not ``None``, the per-key verdict (including
                ``raw_output``) is persisted to the debug snapshot via
                :meth:`~paramem.training.debug_snapshot.DebugSnapshotWriter.on_recall_probe`
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
                self._debug_writer.on_recall_probe(
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
            raise RuntimeError(
                f"Post-save disk-integrity probe failed for adapter '{adapter_name}': "
                f"recall {recall_rate:.3f} < threshold {threshold:.2f} "
                f"(slot: {slot_path}). "
                "The on-disk artifact may be corrupt. "
                "Sessions will remain pending for retry on the next cycle."
            )

        return recall_rate

    def _append_capacity_ceiling_log(
        self,
        *,
        tier: str,
        n_keys_pre: int,
        n_keys_post: int,
        recall_pre: float,
        recall_post: float,
        log_path: str | Path | None = None,
    ) -> None:
        """Append one JSONL row to the capacity-ceiling log.

        Creates the parent directory if absent (idempotent).  Each row records
        the tier, key counts, and recall rates at the time of a ceiling event so
        operators can diagnose capacity degradation without re-running evals.

        Args:
            tier: The adapter tier that tripped the ceiling (``"episodic"`` etc.).
            n_keys_pre: Number of keys in the tier before the rebuild attempt.
            n_keys_post: Number of keys in the tier after the rebuild attempt.
            recall_pre: Estimated recall rate before the rebuild (1.0 by default
                when no pre-rebuild probe is available).
            recall_post: Recall rate measured immediately after the rebuild.
            log_path: Override path for the JSONL file.  Defaults to
                ``outputs/capacity_ceiling.jsonl``.
        """
        import datetime
        import json

        if log_path is None:
            log_path = Path("outputs") / "capacity_ceiling.jsonl"
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        row = {
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
            "tier": tier,
            "n_keys_pre": n_keys_pre,
            "n_keys_post": n_keys_post,
            "recall_pre": recall_pre,
            "recall_post": recall_post,
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")

        logger.warning(
            "capacity_ceiling_event tier=%s recall_pre=%.3f recall_post=%.3f keys=%d log=%s",
            tier,
            recall_pre,
            recall_post,
            n_keys_post,
            log_path,
        )

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
        num_epochs: int | None = None,
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
          - ConsolidationLoop._train_tier_adapter — single funnel for the
            episodic / interim / fold training path; run_consolidation_cycle
            and consolidate_interim_adapters reach this callback transitively
            via _train_tier_adapter and do not call it directly.
          - ConsolidationLoop._run_indexed_key_procedural (direct — procedural
            inline path bypasses _train_tier_adapter)
          - active_store_migration._migrate_tier_simulate_to_train (direct)
            (paramem/server/active_store_migration.py)

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
                call.  When provided, the callback's forced final-epoch probe
                fires at this epoch rather than at
                ``self.training_config.num_epochs``.  Callers that train with
                a different epoch budget (e.g. ``consolidate_interim_adapters``
                using ``refresh_epochs``) MUST pass this so
                ``RecallEarlyStopCallback._num_epochs`` matches the trainer's
                epoch count.  Defaults to ``None`` which resolves to
                ``self.training_config.num_epochs`` — the correct value for
                callers that use the default training budget.

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
        effective_num_epochs = (
            num_epochs if num_epochs is not None else self.training_config.num_epochs
        )
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
            num_epochs=effective_num_epochs,
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
        num_epochs: "int | None" = None,
    ):
        """Format → dataset → recall callback → train_adapter for one tier.

        Returns ``(metrics, recall_state)``.  Returns ``(None, None)`` when
        there are no training examples (empty entries list).

        This is the ONLY shared training-invocation site.  Abort handling,
        recall-verdict application, and persistence stay at the call sites
        (scope-specific).

        The ``from paramem.training.trainer import train_adapter`` import is
        kept INSIDE this method so tests can patch
        ``paramem.training.trainer.train_adapter`` and intercept calls
        at this site.

        Args:
            entries: The full per-tier active entries (key/subject/predicate/
                object dicts).
            adapter_name: The adapter slot being trained.
            adapter_config: PEFT ``AdapterConfig`` for this tier.
            training_config: ``TrainingConfig`` for this call (epoch count,
                LR schedule, etc.).
            output_dir: HF Trainer ``output_dir``; also used by the recall
                callback for ``progress.json`` / ``epoch_log.json``.
            run_name: W&B / HF Trainer run name.
            phase_name: Label for the recall callback's ``progress.json``
                (e.g. ``"interim-episodic-tick42"``, ``"consolidate-semantic"``).
            num_epochs: Passed through to ``_maybe_make_recall_callback`` so
                the callback's forced final-epoch probe fires at the right
                epoch.  ``None`` resolves to
                ``self.training_config.num_epochs`` inside the callback helper.

        Returns:
            ``(metrics_dict, recall_state)`` on success; ``(None, None)`` if
            ``entries`` yields no training examples.
        """
        from paramem.training.trainer import train_adapter

        examples = format_entry_training(entries, self.tokenizer, max_length=1024)
        if not examples:
            return None, None
        dataset = self._indexed_dataset(examples)
        self._enable_gradient_checkpointing()
        recall_cb, recall_state = self._maybe_make_recall_callback(
            entries=entries,
            adapter_name=adapter_name,
            output_dir=output_dir,
            phase_name=phase_name,
            num_epochs=num_epochs,
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
        )
        return metrics, recall_state


def run_multi_session(
    model,
    tokenizer,
    sessions: list[dict],
    consolidation_config: ConsolidationConfig,
    training_config: TrainingConfig,
    episodic_adapter_config: AdapterConfig,
    semantic_adapter_config: AdapterConfig,
    wandb_config: Optional[WandbConfig] = None,
    output_dir: str | Path = "outputs/phase3",
    extraction_temperature: float = 0.0,
) -> list[CycleResult]:
    """Run consolidation loop across multiple sessions.

    Args:
        sessions: List of dicts with 'session_id' and 'transcript' keys.

    Returns:
        List of CycleResult, one per session.
    """
    from paramem.memory.store import MemoryStore

    store = MemoryStore(
        replay_enabled=consolidation_config.indexed_key_replay_enabled,
    )
    try:
        store.load_registries_from_disk(Path(output_dir))
    except Exception:
        logger.exception("run_multi_session: registry load failed; starting with empty store")

    loop = ConsolidationLoop(
        model=model,
        tokenizer=tokenizer,
        consolidation_config=consolidation_config,
        training_config=training_config,
        episodic_adapter_config=episodic_adapter_config,
        semantic_adapter_config=semantic_adapter_config,
        memory_store=store,
        wandb_config=wandb_config,
        output_dir=output_dir,
        extraction_temperature=extraction_temperature,
    )

    results = []
    for session in sessions:
        result = loop.run_cycle(
            session_transcript=session["transcript"],
            session_id=session["session_id"],
        )
        results.append(result)

    return results


def _mentions_any(text: str, terms: set[str]) -> bool:
    """Check if text mentions any of the given terms."""
    text_lower = text.lower()
    return any(term in text_lower for term in terms)


_SYMMETRIC_ENRICHMENT_PREDICATES = frozenset(
    canonical(p)
    for p in {
        "colleague_of",
        "friend_of",
        "neighbor_of",
        "sibling_of",
        "married_to",
        "teammate_of",
        "classmate_of",
        "shares_interest_with",
        "attended_with",
        "knows",
        "family_of",
    }
)


_SAME_AS_HONORIFICS = {
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "professor",
    "sir",
    "madam",
    "mister",
}


def _strip_honorifics(name: str) -> list[str]:
    """Return lowercased tokens of ``name`` with trailing-dot honorifics removed."""
    toks = []
    for raw in name.lower().split():
        t = raw.rstrip(".,")
        if t and t not in _SAME_AS_HONORIFICS:
            toks.append(t)
    return toks


def _safe_to_merge_surface(a: str, b: str) -> bool:
    """Heuristic gate: is it safe to merge two surface forms as the same entity?

    Two-stage check:

    1. Token-subset after honorific strip. "Mr. Yang" → {"yang"} is a
       subset of "Yang Ming" → {"yang", "ming"}. Safe.
    2. Single-token diff + Jaro-Winkler on the distinct tokens only.
       "Catherine Holmes" / "Katherine Holmes" share "holmes"; JW on
       "catherine" vs "katherine" ≈ 0.95 → accept. "Zhang Min" /
       "Wang Min" share "min"; JW on "zhang" vs "wang" ≈ 0.50 →
       reject. Multi-token symmetric difference always rejects.

    Returns ``False`` on empty or all-honorific inputs.
    """
    a_toks = _strip_honorifics(a)
    b_toks = _strip_honorifics(b)
    if not a_toks or not b_toks:
        return False
    a_set = set(a_toks)
    b_set = set(b_toks)
    if a_set <= b_set or b_set <= a_set:
        return True
    only_a = a_set - b_set
    only_b = b_set - a_set
    if len(only_a) != 1 or len(only_b) != 1:
        return False
    from rapidfuzz.distance import JaroWinkler

    jw = JaroWinkler.normalized_similarity(next(iter(only_a)), next(iter(only_b)))
    return jw >= 0.85


def serialize_subgraph_triples(subgraph) -> list[dict]:
    """Serialize a NetworkX subgraph into a list of triple dicts.

    Iterates ``subgraph.edges(data=True)`` and produces one dict per edge with
    keys ``subject``, ``predicate``, ``object``, and ``relation_type``.  The
    ``predicate`` field is taken directly from the edge ``"predicate"``
    attribute; ``relation_type`` defaults to ``"factual"`` when absent.

    Args:
        subgraph: A NetworkX (Multi)DiGraph subgraph view or instance.

    Returns:
        List of ``{"subject": str, "predicate": str, "object": str,
        "relation_type": str}`` dicts, one per directed edge.
    """
    triples = []
    for src, tgt, data in subgraph.edges(data=True):
        triples.append(
            {
                "subject": str(src),
                "predicate": str(data.get("predicate", "")),
                "object": str(tgt),
                "relation_type": str(data.get("relation_type", "factual")),
            }
        )
    return triples
