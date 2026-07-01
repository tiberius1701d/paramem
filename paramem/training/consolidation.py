"""Consolidation loop orchestrator.

Runs the full consolidation pipeline: extract graph from session,
merge into cumulative graph, score for promotion/decay, train
episodic and semantic adapters.
"""

import logging
import os
import random
import secrets
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
    dedup_synonym_predicates,
)
from paramem.graph.merger import GraphMerger, min_nonempty
from paramem.graph.name_match import (
    canonical,
    is_speaker_id,
)
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
from paramem.training.key_registry import KeyRegistry
from paramem.training.thermal_throttle import ThermalPolicy
from paramem.training.trainer import TrainingHooks
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


def resolve_to_node_key(
    name: str,
    in_graph: "Callable[[str], bool]",
    coref_map: "dict[str, str] | None" = None,
) -> str:
    """Resolve a surface name to the actual node key used in the graph (P5).

    Collapses the two formerly-duplicated nested resolvers
    (``_resolve_node_key`` / ``_resolve_name``) into one module-level
    function so the resolution logic lives in exactly one place.

    Resolution order:

    1. **Membership shortcut** — if ``in_graph(name)`` is ``True``, the name
       IS already a valid node key; return it unchanged.  This handles ordinary
       non-speaker node keys (node-key model A: the key IS the canonical form)
       without an extra ``canonical()`` call.  Note: with casefolded speaker
       keys (§0 invariant), verbatim speaker ids (``"Speaker0"``) are NOT in
       the graph (the key is ``"speaker0"``), so they fall through to step 2.
    2. **Canonical fallback** — ``canonical(name)`` (casefolds, diacritic-folds,
       separator-normalizes).  This resolves speaker ids to their casefolded node
       keys (``"Speaker0"`` → ``"speaker0"``) and ordinary display-surface names
       to their canonical keys.
    3. **Coref-chain follow** (optional) — if ``coref_map`` is provided, follow
       the drop→keep chain on the resolved key.  Cycle-guarded via a ``seen``
       set so a malformed coref loop does not block.

    The stale rationale "verbatim-first because speaker nodes are keyed
    VERBATIM" no longer applies: after Step 2 of the speaker-identity
    unification, speaker node keys ARE casefolded, so the membership shortcut
    is only useful for ordinary node keys.

    Args:
        name: Surface name or node key to resolve.
        in_graph: Callable that returns ``True`` when its argument is a live
            node key in the graph (typically ``graph.__contains__`` or
            ``lambda n: n in graph``).
        coref_map: Optional mapping from dropped node key to kept node key,
            built during the same_as contraction pass.  When provided, the
            resolved key is followed through the chain (cycle-guarded).

    Returns:
        The resolved node key as a string.  May not be present in the graph
        if neither the input nor its canonical form is a live node.
    """
    # Step 1: membership shortcut (node already keyed canonically).
    if in_graph(name):
        return name
    # Step 2: canonical fallback (casefolds speaker ids, normalizes surfaces).
    resolved = canonical(name)
    if coref_map is None:
        return resolved
    # Step 3: follow the drop→keep coref chain (cycle-guarded).
    seen: set[str] = set()
    while resolved in coref_map and resolved not in seen:
        seen.add(resolved)
        resolved = coref_map[resolved]
    return resolved


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
        source: Materialize axis forwarded to
            :meth:`~ConsolidationLoop._materialize_consolidation_graph`.
            ``"weights"`` reconstructs from adapter weights (train path);
            ``"disk"`` loads relations from ``graph.json`` files (simulate path).
        persist: Persist venue, dispatched at the end of the spine.

            - ``"interim_slot"`` — call
              :func:`~paramem.memory.persistence.commit_tier_slot` (interim cycle).
            - ``"main_tiers"`` — call :meth:`~ConsolidationLoop._save_adapters`
              (full-fold train path).
            - ``"graph_json"`` — write
              :func:`~paramem.memory.persistence.save_memory_to_disk` **directly**
              on ``self.merger.graph`` (simulate full fold).  MUST NOT derive the
              graph from ``tier_keyed``: for ``source="disk"`` disk relations carry
              ``ik_key``, the keyed branch of ``_build_all_edge_entries_into`` does
              ``store.get(key)`` against the empty simulate store and silently skips
              unmatched edges → ``tier_keyed`` is empty.
        tier: Target adapter name for the interim scope (e.g.
            ``"episodic_interim_YYYYMMDDTHHMM"``).  ``None`` for the full fold
            (all tiers are rebuilt).
        extra_relations_source: Origin of supplemental relations injected via the
            ``extra_relations`` channel of
            :meth:`~ConsolidationLoop._materialize_consolidation_graph`.

            - ``"pending"`` — relations captured from ``merger.graph`` (interim
              cycle pending-session content).
            - ``"disk"`` — relations from ``_collect_disk_fold_relations``
              (simulate full fold).
            - ``"none"`` — no supplemental relations (full-fold train path).
        defer: Forwarded as the ``defer`` flag to
            :meth:`~ConsolidationLoop._build_all_edge_entries_into`.  ``True``
            for the interim slot (atomicity: registry entry deferred until after
            training succeeds); ``False`` for the full fold.
        tag_new: Forwarded as the ``tag_new`` flag to
            :meth:`~ConsolidationLoop._build_all_edge_entries_into`.  ``True``
            for the interim slot (new-entry tracking); ``False`` for the full fold.
        normalize: When ``True``, run the whole-graph normalization pass via
            :meth:`~ConsolidationLoop._refine_consolidation_graph`.
        enrich: When ``True``, run SOTA graph enrichment via
            :meth:`~ConsolidationLoop._refine_consolidation_graph`.
        promote: When ``True``, call
            :meth:`~ConsolidationLoop._promote_mature_keys_inline` after the
            Refine stage.  ``True`` for the full-fold train path only — the
            simulate path skips it because there are no weight-resident keys to
            promote.
        tier_floor: When ``True``, run the per-tier floor park/graduate pass
            (Pass-2) after the build-entries stage.  ``True`` for the full-fold
            train path only.
        subtractive_scope: Forwarded to
            :meth:`~ConsolidationLoop._apply_subtractive_removals_to_store`
            (``"interim"`` | ``"fold"``).
        consume_pending: When ``True``, the full fold extracts pending sessions
            in-fold and merges them before training (the ``max_interim_count == 0``
            consume-pending mode, set by the server).
        cross_tier_resume: When ``True``, the fold would checkpoint completed
            tiers for cross-tier crash resume.  Declared but not yet wired —
            no code path currently sets or reads it.
    """

    # --- identity / dispatch ---
    name: str  # "interim" | "full"  (log/debug label only)
    source: "Literal['weights', 'disk']"
    persist: "Literal['interim_slot', 'main_tiers', 'graph_json']"

    # --- materialize scoping ---
    tier: "str | None" = None
    extra_relations_source: "Literal['pending', 'disk', 'none']" = "none"
    defer: bool = False
    tag_new: bool = False

    # --- refine gate ---
    normalize: bool = False
    enrich: bool = False

    # --- spine stage gates ---
    promote: bool = False
    tier_floor: bool = False
    subtractive_scope: "Literal['interim', 'fold']" = "fold"

    # --- pending capture / resume ---
    consume_pending: bool = False  # extract pending sessions in-fold
    cross_tier_resume: bool = False  # checkpoint completed tiers (not yet wired)


@dataclass(frozen=True)
class DiskFoldInput:
    """Collected input for a disk-source (simulate-mode) full fold.

    Produced by :meth:`ConsolidationLoop._collect_disk_fold_relations` and
    consumed by :meth:`ConsolidationLoop._run_fold` (via the ``"disk"``
    persist venue) so that ``active_before_count`` is captured BEFORE
    :meth:`~ConsolidationLoop._materialize_consolidation_graph` resets the
    merger graph (after the reset the pre-merge edge count is unrecoverable).

    Attributes:
        relations: All :class:`~paramem.graph.schema.Relation` objects
            collected from the canonical ``episodic/graph.json`` and every
            simulate-mode interim slot ``graph.json``.  Fed into
            ``_materialize_consolidation_graph`` via ``extra_relations``.
        interim_dirs: Directories that were merged, in iteration order.
            The caller uses this list for the post-persist ``shutil.rmtree``
            cleanup step.
        active_before_count: Number of edges in the canonical graph BEFORE
            the reset/merge, captured from the just-loaded canonical graph.
            Required by :meth:`~ConsolidationLoop._build_tier_delta` as the
            ``active_before`` input for the ``episodic`` tier.
    """

    relations: list  # list[Relation] — avoids forward-ref in frozen dataclass
    interim_dirs: list  # list[Path]
    active_before_count: int


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
        sota_enabled: bool = False,
        graph_enrichment_neighborhood_hops: int = 2,
        graph_enrichment_max_entities_per_pass: int = 50,
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
                sota_enabled=sota_enabled,
            ),
            prompts_dir=prompts_dir,
            model_name=model_name,
        )

        # Graph-level SOTA enrichment knobs (Task #10).
        self.graph_enrichment_neighborhood_hops = graph_enrichment_neighborhood_hops
        self.graph_enrichment_max_entities_per_pass = graph_enrichment_max_entities_per_pass

        gc = graph_config or GraphConfig()
        self.merger = GraphMerger(
            similarity_threshold=gc.entity_similarity_threshold,
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
        ``consolidate_interim_adapters`` triple-lookup) reads the canonical field
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
                        timestamp=event_time,
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
        as the scheduled interim training path.  After the cycle, calls
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
    ) -> None:
        """Write the initial ``fold_resume.json`` marker once the assignment is finalized.

        Called on the TRAINING path, AFTER the whole-fold accumulate guard,
        so an accumulating early-return never leaves a stale marker.

        ``completed_tiers`` starts empty; the first ``in_flight_tier`` is the
        first tier that has training entries.

        Args:
            scope_name: ``"main_tiers"`` (full fold) or ``"interim_slot"``
                (interim-slot fold).
            fold_stamp: SHA-256 from ``_compute_fold_stamp`` (pre-mutation).
            train_assignment: Per-tier lists of entry dicts
                (``key/subject/predicate/object/speaker_id``).
            dataset_fingerprints: Per-tier ``_fingerprint_dataset`` hexdigest.
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
                this tier, or ``None`` when the tier was a fast-start graduation
                (no checkpoint dir exists; reload comes from the production slot).
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

        Returns ``None`` when no matching directory is found.  Used to locate
        the durable epoch checkpoint for ``_mark_tier_complete``.
        """
        checkpoints = sorted(
            directory.glob("checkpoint-*"),
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
                ``_current_interim_stamp`` instance attribute when set (e.g.
                during housekeeping folds).

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
        recall_sanity_threshold: "float | None" = None,
        new_promotions: "list[str] | None" = None,
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
        8c. Refine: call :meth:`_refine_consolidation_graph` with
           ``enrich=(refinement_enrichment=="on" and sota_enabled)`` so SOTA
           enrichment is gated.  The recurrence-bump runs at every level.
        9. Build interim key list via graph-walk (episodic + procedural entries).
           The interim slot holds BOTH factual (episodic) and preference
           (procedural) keys, trained with the attention-only episodic adapter
           config by design; procedural keys fold to the ``procedural`` main
           adapter only at the full fold.
           When *new_promotions* is non-empty, move matching episodic keys to
           semantic before training.
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
                (identical to S4 behavior).  At slack > 0, the gate is:
                    c < N           → normal mint (unchanged)
                    N <= c < N+slack → overflow mint; result["overflow_slot"]=True
                    c >= N+slack    → cap_pending (keep sessions pending)
                Counted against PEFT-resident adapters; the slack is proven
                to fit VRAM at boot via ``required_working_set_bytes``.
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
            outcome (``"trained"``, ``"simulated"``, ``"cap_pending"``,
            or ``"noop"``); ``venue`` is the training medium (``"train"`` or
            ``"simulate"``).
        """
        # Resolve threshold from config when the caller did not supply an override.
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
                self._debug_writer.on_cycle_end(cap_pending_summary, interim_stamp=stamp)
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
        # new_promotions forwarded as a parameter — _run_fold handles
        # the semantic-transfer block scope-gated on source=="weights" && new_promotions.
        # Map the caller's mode Literal to the FoldScope source axis without a mode== fork.
        _interim_source = {"train": "weights", "simulate": "disk"}[mode]
        result = self._run_fold(
            FoldScope(
                name="interim",
                source=_interim_source,
                persist="interim_slot",
                tier=adapter_name,
                extra_relations_source="pending",
                defer=True,
                tag_new=True,
                normalize=False,  # normalization is full-fold only
                enrich=(self.config.refinement_enrichment == "on" and self.config.sota_enabled),
                promote=False,
                tier_floor=False,
                subtractive_scope="interim",
            ),
            adapter_name=adapter_name,
            stamp=stamp,
            new_promotions=new_promotions,
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

        This is the shared soft-stale stage (M5) called by BOTH
        ``run_consolidation_cycle`` (interim) and ``consolidate_interim_adapters``
        (fold) after every merge that can produce subtractive removals.  The
        shared body is identical for both scopes; the persist/registry-seed step
        that follows is scope-specific and stays in the caller.

        **Always-stale reasons (both scopes):**
        - ``"predicate_synonym_collapse"`` — synonym-predicate collapse from the
          whole-graph normalization pass (:meth:`_run_graph_normalization`).
        - ``"semantic_dedup"`` — near-duplicate triple collapse from the normalization
          pass.
        - ``"entity_merge"`` — edge incident to a same_as variant node (normalization
          pass stale+add).
        - ``"contradiction_same_pred"`` — recency-backed contradiction removal:
          the merger only emits this ledger entry when timestamps pick a unique
          winner; empty/tied → no entry → no stale.  Safe to stale at fold scope
          because a timestamp-less key that tied would never appear here.

        ``"enrichment_same_as"`` and other retain-only reasons stay in the fold's
        ``drift_intended_removal`` bucket (handled inline in
        ``consolidate_interim_adapters``); this helper does NOT soft-stale those.

        Args:
            scope: ``"interim"`` or ``"fold"``.  The stale logic is identical at
                both scopes; the parameter is kept for logging context and
                compatibility with existing callers.

        Returns:
            ``soft_stale_by_tier`` — a per-tier dict mapping staled key strings
            to ``{"stale_cycles": int, "simhash": int|None}`` records.  Passed
            by the fold caller to
            :meth:`_reset_main_tier_registries_and_simhashes` so the rebuilt
            registry seeds the stale partition.  Interim callers can ignore the
            return value (``store.discard_keys`` already mutated the in-memory
            registry; ``commit_tier_slot`` persists it).
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

        Each enrichment edge is a second-order fact derived from its chunk's
        source edges, so it inherits their assertion window: ``last_seen`` is
        the max (most recent) and ``first_seen`` the earliest non-empty
        (:func:`~paramem.graph.merger.min_nonempty`) across the chunk
        subgraph's edges, computed before same_as contraction mutates the
        graph.  This mirrors :meth:`_build_registry_true_relations` stamping
        ``last_seen``/``first_seen`` from bookkeeping.

        Early-return conditions (all return ``skipped=True``):
        - Graph has fewer than 10 nodes (floor — too little signal).
        - ``extraction_noise_filter`` is empty (no SOTA provider configured).
        - Provider env-var is absent (API key not set).

        Chunking strategy:
        Entities are ranked by ``reinforcement_count`` descending.  For each
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

        from paramem.memory.persistence import _IK_KEY_ATTR

        _empty = {"chunks": 0, "new_edges": 0, "same_as_merges": 0}

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

        # Rank nodes by reinforcement descending.
        nodes_by_recurrence = sorted(
            graph.nodes(data=True),
            key=lambda nd: nd[1].get("reinforcement_count", 0),
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

        total_merges = 0
        calls_made = 0
        seen_merge_keys: set[frozenset] = set()
        # Accumulates ik_keys from edges dropped by successful same_as contractions.
        # Keys are written to self.merger.removal_ledger after the loop completes
        # so the classifier can distinguish intended enrichment-driven removals from
        # genuine reconstruction failures.
        _collapsed_ik: dict[str, str] = {}  # ik_key → keep node
        # SF-2: accumulate Relation objects across all chunks; edge-count delta
        # computed after _merge_registry_relations so merger deduplication is counted.
        enrichment_relations: list[Relation] = []
        _edges_before = graph.number_of_edges()

        for chunk_nodes in chunks:
            try:
                chunk_subgraph = graph.subgraph(chunk_nodes)
                # Enrichment edges are second-order facts derived from this
                # chunk's source edges, so they inherit the chunk's assertion
                # window rather than landing untimed.  Computed from the
                # subgraph view BEFORE any same_as contraction below mutates
                # ``graph`` (contraction would change what the view sees).
                _chunk_last_seen = ""
                _chunk_first_seen = ""
                for _u, _v, _edata in chunk_subgraph.edges(data=True):
                    _chunk_last_seen = max(_chunk_last_seen, _edata.get("last_seen") or "")
                    _chunk_first_seen = min_nonempty(
                        _chunk_first_seen, _edata.get("first_seen") or ""
                    )
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

            # _in_graph closure passed to resolve_to_node_key (P5).
            _in_graph = graph.__contains__

            for pair in same_as_pairs:
                keep, drop = pair[0], pair[1]
                if keep == drop:
                    continue
                # W1 guard: skip same_as pairs where BOTH surface strings are
                # speaker ids.  Speaker identity is authoritative (voice/enrollment);
                # it must never be coalesced by a surface-similarity heuristic.
                # Two speaker-id surfaces are either the SAME speaker (already
                # unified by canonical node-keying, so no merge needed) or
                # DIFFERENT speakers (must never merge — Jaro-Winkler treats the
                # distinguishing digit as a typo and would incorrectly merge
                # Speaker0/Speaker1).  Skip unconditionally: the pair is always
                # either redundant or catastrophically wrong.
                # Note: the ``keep_canon == drop_canon`` post-resolution check
                # handles the casing-only case (Speaker0/speaker0), but does NOT
                # catch distinct speaker ids (speaker0 ≠ speaker1) — this guard
                # is load-bearing for the distinct-speaker scenario.
                if is_speaker_id(keep) and is_speaker_id(drop):
                    logger.debug(
                        "graph_enrichment: same_as skip — both surfaces are speaker ids %r / %r",
                        keep,
                        drop,
                    )
                    continue
                # Resolve to actual node keys via P5 (membership shortcut then
                # canonical).  BL-1: keep _safe_to_merge_surface on the ORIGINAL
                # SURFACE strings (fuzzy layer-2 check; done before resolution).
                keep_canon = resolve_to_node_key(keep, _in_graph)
                drop_canon = resolve_to_node_key(drop, _in_graph)
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

            # Build Relation objects from SOTA-emitted new_rels for this chunk.
            # BLOCKER-1 endpoint surface rule: speaker endpoints pass their canonical
            # key (the speaker_id), non-speaker endpoints pass the display surface.
            fallback_rtype = "factual"
            valid_rtypes = {"factual", "temporal", "preference", "social"}
            for rel in new_rels:
                if not isinstance(rel, dict):
                    continue
                # Remap endpoints through this chunk's coref map so edges
                # referencing a to-be-dropped node still land on the canonical.
                # P5: resolve_to_node_key(membership shortcut → canonical → coref chain).
                subj_canon = resolve_to_node_key(rel.get("subject", ""), _in_graph, coref_map)
                raw_pred = rel.get("predicate", "")
                obj_canon = resolve_to_node_key(rel.get("object", ""), _in_graph, coref_map)
                rtype = rel.get("relation_type", fallback_rtype)
                if rtype not in valid_rtypes:
                    rtype = fallback_rtype
                if not (subj_canon and raw_pred and obj_canon):
                    continue

                # Choose endpoint surface string per endpoint.
                # Speaker endpoint (node carries speaker_id attribute): pass the node
                # key (lowercase canonical speaker{N} id) so _synth_speaker_entities
                # can emit the correct Entity from the canonical key.
                # Non-speaker endpoint: pass the display surface from node attributes.
                def _endpoint_str(canon: str) -> str:
                    _n = graph.nodes.get(canon, {})
                    if _n.get("speaker_id"):
                        return canon
                    return _n.get("attributes", {}).get("name", canon)

                subj_endpoint = _endpoint_str(subj_canon)
                obj_endpoint = _endpoint_str(obj_canon)

                if not (subj_endpoint and obj_endpoint and subj_endpoint != obj_endpoint):
                    continue

                try:
                    confidence = float(rel.get("confidence", 0.8))
                except (TypeError, ValueError):
                    confidence = 0.8
                # Safety net for the prompt-level 0.7 rule: discard low-confidence
                # enriched edges even if the model ignored its own instruction.
                if confidence < 0.7:
                    continue

                # B-3: derive speaker_id from the subject node's speaker_id attribute.
                _subj_sid = graph.nodes.get(subj_canon, {}).get("speaker_id", "")

                enrichment_relations.append(
                    Relation(
                        subject=subj_endpoint,
                        predicate=raw_pred,
                        object=obj_endpoint,
                        relation_type=rtype,  # type: ignore[arg-type]
                        confidence=confidence,
                        speaker_id=_subj_sid,
                        symmetric=bool(rel.get("symmetric")),
                        edge_source="graph_enrichment",
                        last_seen=_chunk_last_seen,
                        first_seen=_chunk_first_seen,
                    )
                )

        # Route all accumulated enrichment relations through the merger so they
        # receive full Case-1/Case-3 treatment (dedup, edge-source stamp, speaker_id).
        if enrichment_relations:
            self._merge_registry_relations(
                enrichment_relations,
                session_id="__graph_enrichment__",
                log_label="enrichment relations",
                resolve_contradictions=False,
            )

        # SF-2: edge-count delta (merger may absorb some via Case-1).
        total_new = max(0, graph.number_of_edges() - _edges_before)

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
        return {
            "chunks": calls_made,
            "new_edges": total_new,
            "same_as_merges": total_merges,
            "skipped": False,
            "skip_reason": None,
        }

    def _run_graph_normalization(self) -> dict:
        """Whole-graph normalization pass using :func:`dedup_synonym_predicates`.

        Runs after the Materialize stage at refinement level ``light`` or higher,
        over the cumulative ``merger.graph``.  Collapses synonym predicates on the
        same ``(subject, object)`` pair.  One model call per candidate group.

        Engine selection (resolved once before the primitive call):

        * **Local** (default): ``model`` + ``tokenizer`` from ``self``.
        * **SOTA**: when ``config.sota_enabled`` is ``True`` AND
          ``extraction.config.noise_filter`` is set AND the corresponding
          ``PROVIDER_KEY_ENV`` variable is non-empty.  Falls back to local
          silently when credentials are absent; normalization still runs.

        The model receives the predicate list for each candidate group via the
        ``{predicates_json}`` slot in ``graph_dedup_filter.txt`` and returns the
        cluster schema ``{"clusters": [["predA", "predB"], ...]}``.  GC
        disable/enable is handled inside :func:`dedup_synonym_predicates`.

        Survivor selection: within each cluster, the predicate with the highest
        ``reinforcement_count`` (summed across all edges for that predicate)
        survives.  Tie broken by max ``last_seen``; remaining ties by cluster
        iteration order.

        Overlapping clusters (same predicate in two clusters — a model
        hallucination) are handled by ``_retired_in_so_group`` tracking per
        ``(s, o)`` group: a predicate retired by cluster N is excluded from
        cluster N+1's candidate list, and ``graph.remove_edge`` is called
        fail-loud (no try/except) since the tracking guarantees each edge is
        retired at most once.

        Applied changes — same-(subject, object) collapse only (subtractive,
        grounded in input):

        1. Build ``_flat_relations`` and ``_so_groups`` from all predicate-bearing
           edges in ``merger.graph``.
        2. Call :func:`dedup_synonym_predicates` with the resolved engine kwargs.
        3. For each returned ``(can_s, can_o)`` cluster group: pick the MAX
           ``reinforcement_count`` edge as survivor; retire the rest.

           a. Union ``sessions``, sum ``reinforcement_count``, max ``confidence``,
              max ``last_seen`` of all retired edges onto the survivor BEFORE removal.
           b. Write ``removal_ledger[ik_key] = {"reason": "predicate_synonym_collapse",
              "merged_into": <survivor_pred>}`` for each retired keyed edge.
           c. Remove retired edges from the graph.

        4. Single-predicate ``(s, o)`` facts are NEVER retired.
        5. Hallucinated predicates in model output are grounded inside
           :func:`dedup_synonym_predicates` — no new edges are minted.

        Early-return conditions (all return ``skipped=True``):

        - No local model available (``self.model is None``).
        - Graph has fewer than 10 nodes (too little signal).
        - Prompt file missing: raises ``FileNotFoundError`` (fail-loud).

        Returns:
            Diagnostics dict with keys:
                - ``groups_collapsed`` (int): ``(s, o)`` groups with >=1 retirement.
                - ``edges_retired`` (int): total edges removed across all groups.
                - ``chunks`` (int): total model calls made.
                - ``skipped`` (bool): ``True`` when the pass was bypassed.
                - ``skip_reason`` (str | None): reason token when skipped.
        """
        from paramem.graph.prompts import _load_prompt
        from paramem.memory.persistence import _IK_KEY_ATTR

        _empty = {"groups_collapsed": 0, "edges_retired": 0, "chunks": 0}

        if self.model is None:
            logger.info("graph_normalization: no local model — skipping")
            return {**_empty, "skipped": True, "skip_reason": "no_model"}

        graph = self.merger.graph
        node_count = graph.number_of_nodes()

        if node_count < 10:
            logger.info(
                "graph_normalization: graph too small (%d nodes < 10 floor) — skipping",
                node_count,
            )
            return {**_empty, "skipped": True, "skip_reason": "floor"}

        # Load the filter prompt — fail loud if missing (misconfiguration, not transient).
        filter_prompt = _load_prompt("graph_dedup_filter.txt", required=True)

        # Build flat relation list and per-(s,o) group index.
        # Each _info entry: {u, v, eid, edata, ik_key, can_s, can_o, can_pred}
        _flat_relations: list[dict] = []
        _so_groups: dict[tuple, dict[str, list[dict]]] = {}

        for _u, _v, _eid, _edata in graph.edges(keys=True, data=True):
            _pred = _edata.get("predicate", "")
            if not _pred:
                continue
            _ik = _edata.get(_IK_KEY_ATTR) or None
            _can_s = canonical(_u)
            _can_o = canonical(_v)
            _can_pred = canonical(_pred)
            _info = {
                "u": _u,
                "v": _v,
                "eid": _eid,
                "edata": _edata,
                "ik_key": _ik,
                "can_s": _can_s,
                "can_o": _can_o,
                "can_pred": _can_pred,
            }
            _flat_relations.append({"subject": _u, "predicate": _pred, "object": _v})
            _so_groups.setdefault((_can_s, _can_o), {}).setdefault(_can_pred, []).append(_info)

        # Resolve which backend to use: SOTA when sota_enabled + provider + api_key
        # are all present; fall back to local silently when credentials are absent
        # (local model is always present; normalization still runs on the knob).
        engine_kwargs: dict = {"model": self.model, "tokenizer": self.tokenizer}
        if self.config.sota_enabled:
            ext_cfg = self.extraction.config
            provider = ext_cfg.noise_filter
            api_key = os.environ.get(PROVIDER_KEY_ENV.get(provider, ""), "") if provider else ""
            if provider and api_key:
                engine_kwargs = {
                    "sota": {
                        "api_key": api_key,
                        "provider": provider,
                        "filter_model": ext_cfg.noise_filter_model,
                        "endpoint": ext_cfg.noise_filter_endpoint or None,
                        "system_prompt": (
                            "You identify synonym predicate clusters. Output valid JSON only."
                        ),
                    }
                }
                logger.info("graph_normalization: engine=SOTA provider=%s", provider)
            else:
                logger.info(
                    "graph_normalization: sota_enabled but no provider/api_key -- engine=local"
                )

        # Delegate model calls to dedup_synonym_predicates.
        # GC disable/enable is handled inside dedup_synonym_predicates.
        clusters_by_so, _dedup_diag = dedup_synonym_predicates(
            _flat_relations, filter_prompt=filter_prompt, **engine_kwargs
        )

        total_edges_retired = 0
        total_groups_collapsed = 0

        for so_key, group_clusters in clusters_by_so.items():
            pred_map = _so_groups.get(so_key)
            if not pred_map:
                continue

            # Track predicates already retired in this (s,o) group so that
            # overlapping clusters (model hallucination: same predicate in two
            # clusters) do not attempt a second remove_edge on an already-absent
            # edge, which would be an undetectable bug if silenced by try/except.
            _retired_in_so_group: set[str] = set()

            for cluster in group_clusters:
                # cluster is a list of canonical predicates confirmed as synonyms.
                # At least 2 members guaranteed by dedup_synonym_predicates.
                # Exclude predicates already retired by a prior cluster in this group
                # (can arise when model returns overlapping clusters).
                cluster_preds = [
                    cp for cp in cluster if cp in pred_map and cp not in _retired_in_so_group
                ]
                if len(cluster_preds) < 2:
                    continue

                # Survivor selection: MAX reinforcement_count (summed across all edges
                # for that predicate); tie-broken by MAX last_seen (ISO string max).
                def _pred_sort_key(cp: str) -> tuple:
                    edges = pred_map[cp]
                    rec = sum(e["edata"].get("reinforcement_count", 1) for e in edges)
                    ls = max((e["edata"].get("last_seen", "") for e in edges), default="")
                    return (rec, ls)

                _survivor_pred = max(cluster_preds, key=_pred_sort_key)
                _survivor_edges = pred_map[_survivor_pred]
                _survivor_edata = _survivor_edges[0]["edata"]

                # Union provenance from retired edges onto survivor BEFORE removal.
                _surv_sessions: list[str] = list(_survivor_edata.get("sessions", []))
                _surv_recurrence: int = _survivor_edata.get("reinforcement_count", 1)
                _surv_confidence: float = _survivor_edata.get("confidence", 0.0)
                _surv_last_seen: str = _survivor_edata.get("last_seen", "")
                _surv_first_seen: str = _survivor_edata.get("first_seen", "")

                _retired_preds = [cp for cp in cluster_preds if cp != _survivor_pred]
                for _ret_pred in _retired_preds:
                    for _ret_info in pred_map[_ret_pred]:
                        for _sid in _ret_info["edata"].get("sessions", []):
                            if _sid not in _surv_sessions:
                                _surv_sessions.append(_sid)
                        _surv_recurrence += _ret_info["edata"].get("reinforcement_count", 1)
                        _surv_confidence = max(
                            _surv_confidence, _ret_info["edata"].get("confidence", 0.0)
                        )
                        _surv_last_seen = max(
                            _surv_last_seen, _ret_info["edata"].get("last_seen", "")
                        )
                        _surv_first_seen = min_nonempty(
                            _surv_first_seen, _ret_info["edata"].get("first_seen", "")
                        )

                _survivor_edata["sessions"] = _surv_sessions
                _survivor_edata["reinforcement_count"] = _surv_recurrence
                _survivor_edata["confidence"] = _surv_confidence
                _survivor_edata["last_seen"] = _surv_last_seen
                _survivor_edata["first_seen"] = _surv_first_seen

                # Retire each non-survivor edge.
                group_retired = 0
                for _ret_pred in _retired_preds:
                    for _ret_info in pred_map[_ret_pred]:
                        _ret_ik = _ret_info["ik_key"]
                        if _ret_ik:
                            self.merger.removal_ledger[_ret_ik] = {
                                "reason": "predicate_synonym_collapse",
                                "merged_into": _survivor_pred,
                            }
                        # Fail-loud: the (u, v, eid) triple was built from
                        # graph.edges(keys=True) and each predicate is retired at most
                        # once (guaranteed by _retired_in_so_group tracking above),
                        # so remove_edge must always succeed.
                        graph.remove_edge(_ret_info["u"], _ret_info["v"], _ret_info["eid"])
                        logger.info(
                            "graph_normalization: retired (%r, %r, %r) -> survivor=%r",
                            _ret_info["u"],
                            _ret_info["v"],
                            _ret_pred,
                            _survivor_pred,
                        )
                        group_retired += 1

                _retired_in_so_group.update(_retired_preds)
                if group_retired:
                    total_groups_collapsed += 1
                    total_edges_retired += group_retired

        logger.info(
            "graph_normalization: groups_collapsed=%d edges_retired=%d model_calls=%d",
            total_groups_collapsed,
            total_edges_retired,
            _dedup_diag.get("model_calls", 0),
        )

        _applied = {
            "groups_collapsed": total_groups_collapsed,
            "edges_retired": total_edges_retired,
        }
        _decisions = [
            {"subject": s, "object": o, "clusters": cl} for (s, o), cl in clusters_by_so.items()
        ]
        self._debug_writer.on_normalization(
            _dedup_diag.get("raw_outputs", []), _decisions, _applied
        )

        return {
            **_applied,
            "chunks": _dedup_diag.get("model_calls", 0),
            "skipped": False,
            "skip_reason": None,
        }

    def _collect_disk_fold_relations(self, adapter_dir: "Path") -> "DiskFoldInput":
        """Load all :class:`Relation` objects from the canonical graph and interim slots.

        Extracts the disk-relation BUILD for the simulate-mode full fold.  Returns a
        :class:`DiskFoldInput` containing the collected relations, the list of interim
        directories merged (for post-persist ``shutil.rmtree``), and the pre-merge
        edge count from the canonical graph (captured BEFORE
        :meth:`_materialize_consolidation_graph` resets the merger — after the reset
        the count is unrecoverable).

        Steps:

        1. Load the canonical ``episodic/graph.json`` via
           :func:`~paramem.memory.persistence.load_memory_from_disk`.  Capture its
           edge count as ``active_before_count`` (must be captured here, before
           the merger reset in :meth:`_materialize_consolidation_graph`).
        2. Build :class:`Relation` objects from every edge in the canonical graph.
        3. For each ``episodic/interim_<stamp>`` directory via
           :func:`~paramem.memory.interim_adapter.iter_interim_dirs`: skip slots
           without a ``graph.json`` (train-mode interim slots carry only PEFT
           weights); build :class:`Relation` objects from the slot graph.
        4. Return a :class:`DiskFoldInput` with the accumulated relations, the interim
           directory list, and the pre-merge edge count.

        Args:
            adapter_dir: The loop's ``output_dir``, used as the adapter root.

        Returns:
            A :class:`DiskFoldInput` with ``relations``, ``interim_dirs``, and
            ``active_before_count``.
        """
        from paramem.memory.interim_adapter import iter_interim_dirs
        from paramem.memory.persistence import iter_entries, load_memory_from_disk

        canonical_graph_path = adapter_dir / "episodic" / "graph.json"
        # Capture the pre-merge edge count BEFORE _materialize resets the merger.
        # load_memory_from_disk returns an empty graph when the file does not exist
        # (first fold, no prior canonical graph).
        _canonical_graph_before = load_memory_from_disk(canonical_graph_path)
        active_before_count = _canonical_graph_before.number_of_edges()

        _all_relations: list[Relation] = []
        interim_dirs: list[Path] = []

        # Build Relations from the canonical graph.
        for _entry in iter_entries(_canonical_graph_before):
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

        # Build Relations from each simulate-mode interim slot.
        for _interim_name, interim_dir in iter_interim_dirs(adapter_dir):
            interim_graph_path = interim_dir / "graph.json"
            if not interim_graph_path.exists():
                # Skip train-mode interim slots (no graph.json — only PEFT weights).
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
            interim_dirs.append(interim_dir)
            logger.debug(
                "_collect_disk_fold_relations: queued %d entries from %s",
                slot_graph.number_of_edges(),
                interim_dir,
            )

        return DiskFoldInput(
            relations=_all_relations,
            interim_dirs=interim_dirs,
            active_before_count=active_before_count,
        )

    def _capture_pending_relations(self) -> "list[Relation]":
        """Snapshot current merger.graph edges into a list[Relation].

        Called BEFORE :meth:`_materialize_consolidation_graph` resets the graph,
        so the pending-session content survives the reset and re-enters the merge
        via the ``extra_relations`` channel.

        Shared by the interim fold and the consume-pending full fold.
        The interim fold calls this when ``scope.extra_relations_source == "pending"``.
        The full fold calls this when ``scope.consume_pending`` is ``True`` (the
        consume-pending full fold, where app.py has pre-populated
        ``merger.graph`` with pending-session relations).

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
                  ``_merge_registry_relations`` call so a newer pending fact can
                  supersede a strictly-older dated registry-true rival.  Without
                  this field the captured relation would have ``last_seen=""``
                  and the any-empty COEXIST rule would suppress all supersession.
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
    def _interim_venue_from_scope(scope: "FoldScope") -> "Literal['train', 'simulate']":
        """Derive the ``commit_tier_slot`` venue string from *scope*.

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
        are intentionally stripped (M2) so the probe entry shape matches what
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
        # main_tiers input
        window_stamp_override: "str | None" = None,
        # graph_json input
        graph_path: "Path | None" = None,
    ) -> None:
        """Single persist tail for all three fold venues.

        Dispatches on ``scope.persist`` only — never on a ``mode == "train"``
        / ``mode == "simulate"`` literal (the mode-fork guard is satisfied).
        Each branch writes its venue artifact and runs disk-integrity
        verification where adapter weights were written:

        - ``graph_json``: no weights, no verify.
        - ``interim_slot`` (train): passes a
          :meth:`_verify_committed_slot` callback into
          :func:`~paramem.memory.persistence.commit_tier_slot` so the slot
          is reloaded and probed before the registry flush (commit signal).
          A failed probe propagates; ``commit_tier_slot``'s ``finally``
          orphan-cleanup removes the half-committed slot.
        - ``interim_slot`` (simulate): ``verify=None`` — no weights, no probe.
        - ``main_tiers``: disk verify is already inside :meth:`_save_adapters`.

        Called once by :meth:`_run_fold` in place of the three independent
        persist tails that previously closed each of the graph-json,
        interim-slot, and main-tiers early-return blocks.  The surrounding
        venue-specific grooming (refine,
        build-entries, train, result-dict assembly) stays inline in
        :meth:`_run_fold`; only the **save action** is unified here.

        Args:
            scope: Immutable :class:`FoldScope` describing this fold.  The
                ``persist`` field selects the dispatch branch.
            adapter_name: Interim adapter name (``interim_slot`` path only).
            stamp: Sub-interval stamp forwarded to
                :func:`~paramem.memory.persistence.commit_tier_slot`
                (``interim_slot`` path only).
            all_keyed: Full keyed-pair list for the interim slot
                (``interim_slot`` path only).
            window_stamp_override: Forwarded to :meth:`_save_adapters`
                (``main_tiers`` path only).
            graph_path: Destination path for the merged graph
                (``graph_json`` path only).
        """
        from paramem.memory.persistence import commit_tier_slot, save_memory_to_disk

        if scope.persist == "graph_json":
            # graph-json simulate: write merger.graph directly (not tier_keyed).  No weights.
            save_memory_to_disk(self.merger.graph, graph_path)  # type: ignore[arg-type]
        elif scope.persist == "interim_slot":
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
                mode=self._interim_venue_from_scope(scope),
                all_keyed=_keyed,
                output_dir=self.output_dir,
                verify=_verify,
            )
        elif scope.persist == "main_tiers":
            # main tiers full fold: rebuild main adapter weights.  Disk verify is inside
            # _save_adapters (already had it pre-unification).
            self._save_adapters(window_stamp_override=window_stamp_override)

    def _run_fold(
        self,
        scope: "FoldScope",
        *,
        trainer=None,
        router=None,
        recall_sanity_threshold: "float | None" = None,
        refresh_epochs: int = 30,
        housekeeping: bool = False,
        window_stamp_override: "str | None" = None,
        # interim-scope extras (only consumed when scope.persist == "interim_slot")
        adapter_name: "str | None" = None,
        stamp: "str | None" = None,
        new_promotions: "list[str] | None" = None,
        run_label: str = "",
        triples_extracted: int = 0,
        episodic_rels: "list[dict] | None" = None,
        procedural_rels: "list[dict] | None" = None,
    ) -> dict:
        """Scope-parameterized consolidation fold spine — the single shared pipeline.

        All three consolidation paths route through this method:

        - ``scope.persist == "interim_slot"`` (interim mini-fold): single-tier
          training + :func:`~paramem.memory.persistence.commit_tier_slot`.
          Replaces the pipeline body of ``run_consolidation_cycle``.
        - ``scope.persist == "main_tiers"`` (full-fold train): multi-tier rebuild
          + :meth:`_save_adapters`.  Extracted from ``consolidate_interim_adapters``.
        - ``scope.persist == "graph_json"`` (full-fold simulate): disk-source merge
          + :func:`~paramem.memory.persistence.save_memory_to_disk`.
          Replaces the deleted ``consolidate_interim_to_canonical_graph``.

        The three paths differ ONLY in ``scope.source`` (weights vs disk) and the
        persist tail (``scope.persist``).  All grooming stages
        (:meth:`_materialize_consolidation_graph`, :meth:`_refine_consolidation_graph`,
        :meth:`_build_all_edge_entries_into`, :meth:`_apply_subtractive_removals_to_store`)
        are shared and unconditional.

        Promotion (:meth:`_promote_mature_keys_inline`) and the tier-floor pass are
        scope-gated via ``scope.promote`` and ``scope.tier_floor`` respectively — both
        are weight-venue stages that the simulate path correctly skips by scope design.

        **Return schema:** always returns the FULL train schema.
        The ``graph_json`` persist path returns zero/empty equivalents for fields that
        have no meaning for the disk venue (``drift_intended_removal``,
        ``recall_miss_keys``, ``tier_keyed``, etc.) so callers never ``KeyError``
        after the collapse.

        **Mode-fork-guard invariant:** this method and all callers dispatch on
        ``scope.source`` / ``scope.persist`` structural enum attributes — never on
        a ``mode == "simulate"`` / ``mode == "train"`` string literal.  The
        ``mode`` string is computed internally only where required by lower-level
        helpers (``commit_tier_slot``) that are themselves in the allowlist.

        Args:
            scope: Immutable :class:`FoldScope` descriptor.  Selects pipeline stages
                and the persist venue.  Constructed by the thin public-method wrappers;
                never by app-layer callers.
            trainer: :class:`~paramem.server.background_trainer.BackgroundTrainer`
                instance.  Required for the per-tier re-arm pattern in the
                ``main_tiers`` path; ``None`` for ``graph_json`` and for ``interim_slot``
                paths that do not need the abort-for-inference machinery.
            router: Router instance whose ``reload()`` is called at fold completion
                (``main_tiers`` path only).  ``None`` is safe — skipped.
            recall_sanity_threshold: Override for the recall gate (``main_tiers`` path).
                Reads ``self.config.recall_sanity_threshold`` when ``None``.
            refresh_epochs: Training epochs for the full per-tier rebuild
                (``main_tiers`` path only; ``interim_slot`` uses
                ``self.training_config.num_epochs``).
            housekeeping: When ``True`` (``graph_json`` path), bypass the noop gate so
                the re-groomed graph is persisted even with no interim slots.
            window_stamp_override: Forwarded to :meth:`_save_adapters` (``main_tiers``
                path).  Preserves the prior fold's window stamp for housekeeping runs.
            adapter_name: Interim adapter name (``interim_slot`` path only).  Matches
                ``scope.tier``.
            stamp: Sub-interval stamp for :func:`~paramem.memory.persistence.commit_tier_slot`
                (``interim_slot`` path only).
            new_promotions: Optional list of entity names to transfer to semantic
                before training (``interim_slot`` / ``source="weights"`` path only).
                Retained explicitly so the semantic-transfer block is
                not lost when ``run_consolidation_cycle`` becomes a thin wrapper.
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
                    "recall_per_tier": dict[str, float],
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
        import shutil as _shutil

        if recall_sanity_threshold is None:
            recall_sanity_threshold = self.config.recall_sanity_threshold

        # ------------------------------------------------------------------
        # graph-json simulate full fold (scope.persist == "graph_json")
        # ------------------------------------------------------------------
        # Source: disk (graph.json files); no weight reconstruction.
        # Persist: save_memory_to_disk(merger.graph, canonical_graph_path).
        # No training, no registry mutation, no tier-floor.
        # Replaces the deleted consolidate_interim_to_canonical_graph body.
        # ------------------------------------------------------------------
        if scope.persist == "graph_json":
            adapter_dir = self.output_dir
            canonical_graph_path = adapter_dir / "episodic" / "graph.json"

            # Collect disk relations + pre-merge edge count BEFORE _materialize
            # resets the merger graph (after the reset the count is unrecoverable).
            inp = self._collect_disk_fold_relations(adapter_dir)

            logger.info(
                "_run_fold[graph_json]: collected %d relations from canonical graph"
                " + %d interim slot(s)",
                len(inp.relations),
                len(inp.interim_dirs),
            )

            # Noop gate: when housekeeping=False and no interim slots are present,
            # there is nothing to merge — return the full schema with zero values.
            if not housekeeping and not inp.interim_dirs:
                logger.info("_run_fold[graph_json]: no simulate-mode slots — noop")
                self._current_interim_stamp = None  # type: ignore[assignment]
                return {
                    "tiers_rebuilt": [],
                    "graph_drift_count": 0,
                    "drift_deduplicated": 0,
                    "drift_orphan": 0,
                    "drift_genuine_loss": 0,
                    "drift_intended_removal": 0,
                    "drift_intended_removal_by_reason": {},
                    "recall_miss_keys": [],
                    "keys_per_tier": {},
                    "recall_per_tier": {},
                    "tier_keyed": {},
                    "rolled_back": False,
                    "rollback_tier": None,
                    "tier_delta": {},
                }

            try:
                # Materialize: merge disk relations via the shared helper.
                # source="disk" skips weight reconstruction; feeds inp.relations as
                # extra_relations; merger.reset_graph() runs inside;
                # recall_miss_keys=set() (no recall probe on the simulate path).
                # The persist tail writes merger.graph DIRECTLY via save_memory_to_disk
                # — NOT via tier_keyed.  Disk edges carry ik_key; the keyed branch of
                # _build_all_edge_entries_into does store.get against the empty simulate
                # store and silently skips unmatched edges → tier_keyed would be empty.
                self._materialize_consolidation_graph(
                    source="disk",
                    extra_relations=inp.relations,
                )

                # Capture edge count BEFORE refine so we can compute minted edges.
                # minted_count = enrichment new_edges (refine-added beyond the post-merge
                # count).  NOTE: also nets normalization removals that reduce the
                # post-merge edge count before enrichment adds edges; this is a known
                # approximation (same as the previous inline body).
                _after_merge_count = self.merger.graph.number_of_edges()
                self._refine_consolidation_graph(
                    [],
                    normalize=scope.normalize,
                    enrich=scope.enrich,
                )

                # Write merger.graph directly (not tier_keyed).
                canonical_graph_path.parent.mkdir(parents=True, exist_ok=True)
                self._persist_fold(scope, graph_path=canonical_graph_path)
                _after_count = self.merger.graph.number_of_edges()
                _minted_count = max(0, _after_count - _after_merge_count)
                _collapsed_count = sum(
                    1
                    for _rl_e in self.merger.removal_ledger.values()
                    if _rl_e.get("reason") == "dedup"
                )
                logger.info(
                    "_run_fold[graph_json]: wrote merged graph to %s"
                    " (%d edges, %d interim slot(s), %d dedup collapse(s))",
                    canonical_graph_path,
                    _after_count,
                    len(inp.interim_dirs),
                    _collapsed_count,
                )

                _sim_tier_delta = self._build_tier_delta(
                    active_before={"episodic": inp.active_before_count},
                    active_after={"episodic": _after_count},
                    minted_by_tier={"episodic": _minted_count},
                )
                self._debug_writer.on_tier_delta(_sim_tier_delta)
                self._debug_writer.on_removal_ledger(getattr(self.merger, "removal_ledger", {}))

                for _idir in inp.interim_dirs:
                    try:
                        _shutil.rmtree(_idir, ignore_errors=True)
                        logger.debug("_run_fold[graph_json]: removed interim slot %s", _idir)
                    except Exception as _rm_exc:
                        logger.warning(
                            "_run_fold[graph_json]: failed to remove %s: %s",
                            _idir,
                            _rm_exc,
                        )

                self._current_interim_stamp = None  # type: ignore[assignment]

                return {
                    "tiers_rebuilt": ["episodic"],
                    "graph_drift_count": _collapsed_count,
                    "drift_deduplicated": _collapsed_count,
                    "drift_orphan": 0,
                    "drift_genuine_loss": 0,
                    "drift_intended_removal": 0,
                    "drift_intended_removal_by_reason": {},
                    "recall_miss_keys": [],
                    "keys_per_tier": {"episodic": _after_count},
                    "recall_per_tier": {"episodic": 1.0},  # graph merge is lossless
                    "tier_keyed": {},
                    "rolled_back": False,
                    "rollback_tier": None,
                    "tier_delta": _sim_tier_delta,
                }
            finally:
                self.merger.reset_graph()

        # ------------------------------------------------------------------
        # interim mini-fold (scope.persist == "interim_slot")
        # ------------------------------------------------------------------
        # Source: weights (reconstruct from adapter weights, scoped to tier).
        # Persist: commit_tier_slot (writes adapter weights + sidecar JSON).
        # Single-tier training; promote=False, tier_floor=False.
        # Extracted from the training body of run_consolidation_cycle.
        # ------------------------------------------------------------------
        if scope.persist == "interim_slot":
            # --- interim-slot fold-stamp (minted before any store mutation) ---
            # scope.tier gives the logical tier; adapter_name is the PEFT slot name.
            _fold_stamp_b = self._compute_fold_stamp(tier=adapter_name or scope.tier)

            # _run_fold always controls the merger.graph lifecycle for the interim path.
            try:
                # --- End-of-extraction debug dump (interim only) ---
                self._debug_writer.on_extraction_end(
                    episodic_rels or [],
                    procedural_rels or [],
                    interim_stamp=stamp,
                )

                # --- Mint PEFT slot (weights source only) ---
                if scope.source == "weights":
                    from paramem.memory.interim_adapter import create_interim_adapter

                    if adapter_name not in self.model.peft_config:
                        self.model = create_interim_adapter(self.model, self.episodic_config, stamp)
                        logger.info("_run_fold[interim]: created interim adapter %s", adapter_name)

                # --- Materialize: recall-miss diagnostic + rebuild keying surface ---
                # Scoped to the current slot: reconstruct only the slot's registered keys
                # (tier=adapter_name) for the recall-miss diagnostic, then reset and re-merge:
                #   (a) registry-true relations for this slot
                #   (b) the pending-session relations captured from merger.graph before the
                #       reset (extra_relations), so they survive the graph reset.
                _extra: "list[Relation] | None" = (
                    self._capture_pending_relations()
                    if scope.extra_relations_source == "pending"
                    else None
                )

                recall_miss_keys, recon_relations = self._materialize_consolidation_graph(
                    tier=scope.tier,
                    keys=list(self.store.active_keys_in_tier(adapter_name or scope.tier)),
                    extra_relations=_extra,
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
                _tier_keyed: dict[str, list[dict]] = {
                    "episodic": [],
                    "procedural": [],
                    "semantic": [],
                }
                _, _deferred_writes = self._build_all_edge_entries_into(
                    _tier_keyed,
                    defer=scope.defer,
                    tag_new=scope.tag_new,
                )

                all_interim_keyed = _tier_keyed["episodic"] + _tier_keyed["procedural"]
                new_keyed_episodic = [r for r in _deferred_writes if r["tier"] == "episodic"]
                new_keyed_proc = [r for r in _deferred_writes if r["tier"] == "procedural"]
                new_keyed_interim = new_keyed_episodic + new_keyed_proc
                new_key_ids = [r["entry"]["key"] for r in new_keyed_interim]

                # Simulate path: apply store mutations immediately (no training step).
                if scope.source != "weights":
                    for rec in new_keyed_interim:
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
                            relation_type=rec["relation_type"],
                            reinforcement_count=1,
                            last_reinforced_cycle=self.cycle_count,
                            last_seen=rec.get("last_seen", ""),
                            first_seen=rec.get("first_seen", ""),
                            allow_empty_speaker=(rec["speaker_id"] == ""),
                        )
                    self._indexed_next_index += len(new_keyed_episodic)
                    self._procedural_next_index += len(new_keyed_proc)

                # --- Semantic promotion transfer (weights path, when
                # promotions supplied).  Separate from _promote_mature_keys_inline;
                # operates on the interim keyed set before training.
                promoted_key_set: set[str] = set()
                if scope.source == "weights" and new_promotions:
                    _promo_set = {n.lower() for n in new_promotions}
                    for _t, _k, _e in list(self.store.iter_entries()):
                        if _k.startswith("proc"):
                            continue
                        _subj = _e.get("subject", "").lower()
                        _obj = _e.get("object", "").lower()
                        _mentions = (_subj and _subj in _promo_set) or (_obj and _obj in _promo_set)
                        if _mentions and not self.store.has_simhash("semantic", _k):
                            promoted_key_set.add(_k)
                    for _pk in promoted_key_set:
                        self.store.move(_pk, "semantic")
                    all_interim_keyed = [
                        kp for kp in all_interim_keyed if kp["key"] not in promoted_key_set
                    ]

                # --- interim slot: write single-entry fold_resume.json marker ---
                # Written AFTER all_interim_keyed is fully finalized (promotion
                # filter above may shrink it) — not at fold entry.
                # On crash, the marker enables epoch-resume via _resolve_resume_checkpoint
                # (the epoch checkpoint path is already wired).
                # Interim does NOT pass retain_scratch_until_external_commit:
                # commit_tier_slot is an inline durable write right after training,
                # so there is no multi-tier window where a completed-but-uncommitted
                # tier can be lost.
                if scope.source == "weights":
                    _b_assignment = {adapter_name: list(all_interim_keyed)}
                    self._persist_fold_assignment("interim_slot", _fold_stamp_b, _b_assignment, {})

                # --- Train (weights source) or skip (disk source) ---
                epi_train_loss: "float | None" = None
                if scope.source == "weights" and all_interim_keyed:
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

                # Update interim simhash registry.
                _passing_interim = (
                    [kp for kp in all_interim_keyed if kp["key"] in _epi_passing]
                    if _epi_passing is not None
                    else all_interim_keyed
                )
                self.store.replace_simhashes_in_tier(adapter_name, build_registry(_passing_interim))

                # --- Apply deferred interim store mutations (weights source) ---
                _recall_failed_session_ids: set[str] = set()
                if scope.source == "weights":
                    _ep_flushed = 0
                    _proc_flushed = 0
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
                            simhash=compute_simhash(
                                _key,
                                _entry["subject"],
                                rec["predicate"],
                                _entry["object"],
                            ),
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
                    self._indexed_next_index += _ep_flushed
                    self._procedural_next_index += _proc_flushed

                # --- Shared soft-stale stage (M5) ---
                self._apply_subtractive_removals_to_store(scope=scope.subtractive_scope)

                # --- Persist interim slot ---
                self._persist_fold(
                    scope,
                    adapter_name=adapter_name,
                    stamp=stamp,
                    all_keyed=all_interim_keyed,
                )
                # Clear the interim-slot fold_resume.json marker on clean commit.
                self._clear_fold_resume()

                # --- Restore episodic as active adapter ---
                if "episodic" in self.model.peft_config:
                    switch_adapter(self.model, "episodic")

                _interim_mode_label = "trained" if scope.source == "weights" else "simulated"
                _interim_venue = self._interim_venue_from_scope(scope)
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
                    "recall_per_tier": {},
                    "tier_keyed": _tier_keyed,
                    "rolled_back": False,
                    "rollback_tier": None,
                    "tier_delta": {},
                }
                self._debug_writer.on_cycle_end(cycle_summary, interim_stamp=stamp)
                return cycle_summary
            finally:
                self.merger.reset_graph()

        # ------------------------------------------------------------------
        # main-tiers full-fold train (scope.persist == "main_tiers")
        # ------------------------------------------------------------------
        # Source: weights (reconstruct all tiers from adapter weights).
        # Persist: _save_adapters (rebuild main adapter weights + manifest).
        # Multi-tier training loop; promote=scope.promote, tier_floor=scope.tier_floor.
        # Extracted from consolidate_interim_adapters.
        # ------------------------------------------------------------------
        from paramem.memory.interim_adapter import unload_interim_adapters
        from paramem.models.loader import create_adapter

        # --- Fold-stamp + crash-resume marker (full fold) ---
        # Mint fold_stamp BEFORE any store mutation (promote/tier-floor/
        # _build_all_edge_entries_into all mutate the store; the stamp must
        # reflect the pristine on-disk registry so it is byte-identical on
        # re-entry after a crash).
        _fold_stamp_c = self._compute_fold_stamp(tier=None)
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
                train_assignment = {
                    t: list(_marker_ta.get(t, [])) for t in ("episodic", "semantic", "procedural")
                }
                serve_assignment = train_assignment  # drift not re-derived on resume
                recall_miss_keys: list[str] = []
                minted_by_tier: dict = {}
                _fast_start_graduating: set[str] = set()
                _floor = self.config.min_tier_key_floor
                _train_active_before: dict[str, int] = {
                    t: len(serve_assignment[t]) for t in ("episodic", "semantic", "procedural")
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
                # FRESH-DERIVATION PATH: reconstruct → promote → tier-floor → assign.
                # -----------------------------------------------------------------
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

                self._debug_writer.on_fold_graph(self.merger, label="keyed")

                # --- Per-tier floor gate (scope-gated) ---
                _floor = self.config.min_tier_key_floor
                _fast_start_graduating: set[str] = set()

                def _tier_has_disk_slot(tier_name: str) -> bool:
                    from paramem.adapters.manifest import is_slot_name as _isn

                    tier_dir = self.output_dir / tier_name
                    if not tier_dir.is_dir():
                        return False
                    return any(entry.is_dir() and _isn(entry.name) for entry in tier_dir.iterdir())

                if scope.tier_floor:
                    for _pt in ("semantic", "procedural"):
                        _pt_entries = tier_keyed[_pt]
                        if not _pt_entries:
                            continue
                        if len(_pt_entries) < _floor:
                            logger.info(
                                "_run_fold[main_tiers]: tier %s has %d key(s) < floor %d"
                                " — parking in episodic until floor reached",
                                _pt,
                                len(_pt_entries),
                                _floor,
                            )
                            for _pe in _pt_entries:
                                tier_keyed["episodic"].append(_pe)
                                _cur = self.store.tier_for_active_key(_pe["key"])
                                if _cur is not None and _cur != "episodic":
                                    self.store.move(_pe["key"], "episodic")
                                self.promoted_keys.discard(_pe["key"])
                            tier_keyed[_pt] = []
                        else:
                            _tier_is_live = _tier_has_disk_slot(_pt)
                            if not _tier_is_live:
                                if self.config.tier_fast_start:
                                    logger.info(
                                        "_run_fold[main_tiers]: tier %s graduating (fast-start)"
                                        " — %d key(s) >= floor %d; will copy episodic weights",
                                        _pt,
                                        len(_pt_entries),
                                        _floor,
                                    )
                                    _fast_start_graduating.add(_pt)
                                else:
                                    logger.info(
                                        "_run_fold[main_tiers]: tier %s graduating"
                                        " (train-from-scratch) — %d key(s) >= floor %d",
                                        _pt,
                                        len(_pt_entries),
                                        _floor,
                                    )
                                    for _pe in _pt_entries:
                                        self.store.move(_pe["key"], _pt)

                serve_assignment = tier_keyed

                train_assignment: dict[str, list[dict]] = {
                    t: list(serve_assignment[t]) for t in ("episodic", "semantic", "procedural")
                }
                for _fst in _fast_start_graduating:
                    _ep_union: dict[str, dict] = {e["key"]: e for e in train_assignment["episodic"]}
                    for _fse in serve_assignment[_fst]:
                        _ep_union.setdefault(_fse["key"], _fse)
                    train_assignment["episodic"] = list(_ep_union.values())
                    train_assignment[_fst] = []

                self._debug_writer.on_fold_assignments(serve_assignment, train_assignment)

                _train_active_before: dict[str, int] = {
                    t: len(serve_assignment[t]) for t in ("episodic", "semantic", "procedural")
                }

                # --- Whole-fold accumulate guard ---
                _total_trainable = sum(len(v) for v in serve_assignment.values())
                if not housekeeping and _total_trainable < _floor:
                    logger.info(
                        "_run_fold[main_tiers]: total trainable keys %d < floor %d"
                        " — returning accumulating (sessions stay pending)",
                        _total_trainable,
                        _floor,
                    )
                    _acc_parked = {
                        t: len(serve_assignment[t])
                        for t in ("semantic", "procedural")
                        if len(serve_assignment[t]) > 0
                    }
                    for _bname in (
                        "episodic_backup",
                        "semantic_backup",
                        "procedural_backup",
                    ):
                        if _bname in self.model.peft_config:
                            self.model.delete_adapter(_bname)
                            logger.debug(
                                "_run_fold[main_tiers]: cleaned up stale backup %s"
                                " (left by prior aborted fold) on accumulating return",
                                _bname,
                            )
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
                # end of fresh-derivation path.
                # Compute dataset fingerprints and persist the fold assignment marker
                # AFTER the accumulate guard (so accumulating returns never leave a
                # stale marker).
                # Fingerprint is over sorted SPO tuples, NOT tokenized examples.
                # Calling format_entry_training here (before the per-tier loop) would
                # interfere with per-tier format spy patterns in existing tests and is
                # unnecessary — SPO identity is the only change-detection signal needed.
                import hashlib as _hashlib

                _dataset_fingerprints: dict[str, str] = {}
                for _t in ("episodic", "semantic", "procedural"):
                    _ta_entries = train_assignment[_t]
                    if _ta_entries:
                        _fp_h = _hashlib.sha256()
                        for _spo in sorted(
                            (
                                e.get("key", ""),
                                e.get("subject", ""),
                                e.get("predicate", ""),
                                e.get("object", ""),
                            )
                            for e in _ta_entries
                        ):
                            _fp_h.update(repr(_spo).encode("utf-8"))
                        _dataset_fingerprints[_t] = _fp_h.hexdigest()
                self._persist_fold_assignment(
                    "main_tiers", _fold_stamp_c, train_assignment, _dataset_fingerprints
                )

            # --- Drift partition (fresh-fold only; skipped on crash-resume) ---
            # On crash-resume, drift was already applied pre-crash and registries
            # are pristine.  Re-running subtractive removals would double-apply.
            # Counters are pre-zeroed in the resume fast-path above.
            if not _resume_c:
                _all_keyed = {
                    e["key"] for tier_list in serve_assignment.values() for e in tier_list
                }

                for _surviving_key in _all_keyed:
                    _sbk = self.store.bookkeeping_for_key(_surviving_key)
                    if _sbk is not None:
                        _sbk["last_reinforced_cycle"] = self.cycle_count

                _subtractive_stale_by_tier = self._apply_subtractive_removals_to_store(
                    scope=scope.subtractive_scope
                )

                active_keys = self.store.all_active_keys()
                _drift_keys = [k for k in active_keys if k not in _all_keyed]

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
                        "_run_fold[main_tiers]: R4 invariant violation — %d key(s) appear"
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
                        " (reconstruction failure or hydration-miss — retrained with"
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
                    len(serve_assignment["episodic"]),
                    len(serve_assignment["semantic"]),
                    len(serve_assignment["procedural"]),
                    graph_drift_count,
                    drift_deduplicated_count,
                    drift_orphan_count,
                    drift_genuine_loss_count,
                    drift_intended_removal_count,
                )

                self._debug_writer.on_removal_ledger(getattr(self.merger, "removal_ledger", {}))

            # --- Build per-tier TrainingJob objects ---
            from paramem.server.background_trainer import TrainingJob

            refresh_training_config = self._make_training_config(num_epochs=refresh_epochs)

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
            tier_config_for_backup = {
                "episodic": self.episodic_config,
                "semantic": self.semantic_config,
                "procedural": self.procedural_config or self.episodic_config,
            }
            for backup_name in ("episodic_backup", "semantic_backup", "procedural_backup"):
                if backup_name not in self.model.peft_config:
                    base_tier = backup_name.replace("_backup", "")
                    if base_tier in self.model.peft_config:
                        from paramem.models.loader import copy_adapter_weights

                        backup_config = tier_config_for_backup[base_tier]
                        self.model = create_adapter(self.model, backup_config, backup_name)
                        copy_adapter_weights(self.model, src=base_tier, dst=backup_name)
                        logger.info(
                            "_run_fold[main_tiers]: created in-memory backup %s from %s",
                            backup_name,
                            base_tier,
                        )

            tiers_rebuilt: list[str] = []
            recall_per_tier: dict[str, float] = {}
            last_per_key_by_tier: dict[str, "list | None"] = {}

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

                if not job.entries and tier not in _fast_start_graduating:
                    logger.info(
                        "_run_fold[main_tiers]: no keys for tier %s — skipping rebuild", tier
                    )
                    continue

                # --- Crash-resume: reload completed tiers from durable checkpoint ---
                if _resume_c and tier in _completed_in_marker:
                    # The checkpoint path stored in the marker (may be absent for
                    # fast-start tiers, which have no checkpoint-N dir).
                    _ckpt_path = _marker_checkpoints.get(tier)
                    logger.info(
                        "_run_fold[main_tiers]: CRASH-RESUME tier=%s — reloading from"
                        " durable checkpoint (no retrain); checkpoint=%s",
                        tier,
                        _ckpt_path or "production-slot",
                    )
                    # Delete the stale production slot (pre-crash _save_adapters never
                    # ran — weights are stale) and reload from the checkpoint dir or the
                    # existing production slot for fast-start tiers.
                    # Per-tier backups created above (lines 4533-4551) mean the deleted
                    # slot is never the last adapter on the PeftModel (no base-unwrap needed).
                    if tier in self.model.peft_config:
                        if backup_name in self.model.peft_config:
                            from paramem.models.loader import switch_adapter as _sw_pre

                            _sw_pre(self.model, backup_name)
                        self.model.delete_adapter(tier)
                        logger.debug(
                            "_run_fold[main_tiers]: crash-resume deleted stale slot %s", tier
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
                        from paramem.training.trainer import _STAGING_ADAPTER as _STAGING_SLOT

                        # Resolve to the staging-adapter subdir within the checkpoint.
                        _ckpt_staging_path = Path(_ckpt_path) / _STAGING_SLOT
                        _ckpt_effective = (
                            str(_ckpt_staging_path) if _ckpt_staging_path.is_dir() else _ckpt_path
                        )
                        _ckpt_shm_dir = None
                        if _ks.daily_identity_loadable(_ks.DAILY_KEY_PATH_DEFAULT):
                            from paramem.backup.checkpoint_shard import (
                                materialize_checkpoint_to_shm,
                            )

                            _ckpt_shm_dir = materialize_checkpoint_to_shm(Path(_ckpt_effective))
                            _ckpt_load_path = str(_ckpt_shm_dir)
                        else:
                            _ckpt_load_path = _ckpt_effective
                        try:
                            self.model.load_adapter(_ckpt_load_path, adapter_name=tier)
                            logger.info(
                                "_run_fold[main_tiers]: crash-resume loaded %s from checkpoint %s"
                                " (staging slot=%s)",
                                tier,
                                _ckpt_path,
                                _STAGING_SLOT,
                            )
                        finally:
                            if _ckpt_shm_dir is not None and Path(str(_ckpt_shm_dir)).exists():
                                import shutil as _s

                                _s.rmtree(_ckpt_shm_dir, ignore_errors=True)
                    else:
                        # no checkpoint dir (fast-start tier, or checkpoint missing).
                        # Reload from the EXISTING production slot on disk — it was not
                        # overwritten (final _save_adapters never ran on crash).
                        from paramem.memory.interim_adapter import (
                            adapter_slot_root_for_name as _asr_fn,
                        )
                        from paramem.models.loader import load_adapter as _la

                        _prod_root = _asr_fn(self.output_dir, tier)
                        _la(self.model, _prod_root.parent, tier)
                        logger.info(
                            "_run_fold[main_tiers]: crash-resume (fast-start/no-ckpt)"
                            " loaded %s from production slot %s",
                            tier,
                            _prod_root.parent,
                        )
                    from paramem.models.loader import switch_adapter as _sw_resume

                    _sw_resume(self.model, tier)
                    last_per_key_by_tier[tier] = None
                    tiers_rebuilt.append(tier)
                    continue

                if backup_name in self.model.peft_config:
                    from paramem.models.loader import switch_adapter as _sw_backup

                    _sw_backup(self.model, backup_name)

                if tier in self.model.peft_config:
                    self.model.delete_adapter(tier)
                    logger.debug("_run_fold[main_tiers]: deleted adapter %s", tier)

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
                logger.debug("_run_fold[main_tiers]: created fresh adapter %s", tier)

                from paramem.models.loader import switch_adapter as _sw

                _sw(self.model, tier)

                # --- Fast-start graduation branch ---
                if tier in _fast_start_graduating:
                    from paramem.models.loader import copy_adapter_weights as _copy_aw
                    from paramem.models.loader import (
                        copy_adapter_weights_subset as _copy_aw_subset,
                    )

                    if tier == "procedural":
                        _copy_aw_subset(self.model, src="episodic", dst=tier)
                        logger.info(
                            "_run_fold[main_tiers]: fast-start graduation — "
                            "copied episodic attn weights into procedural (mlp stays zero-init)"
                        )
                    else:
                        _copy_aw(self.model, src="episodic", dst=tier)
                        logger.info(
                            "_run_fold[main_tiers]: fast-start graduation — "
                            "copied episodic weights into %s (full param set)",
                            tier,
                        )

                    _serve_entries = serve_assignment[tier]
                    _probe_passing = self._probe_passing_keys(tier, _serve_entries)
                    _probe_rate = (
                        len(_probe_passing) / len(_serve_entries) if _serve_entries else 1.0
                    )
                    logger.info(
                        "_run_fold[main_tiers]: fast-start graduation probe %s"
                        " — %d/%d passed (%.3f), threshold %.3f",
                        tier,
                        len(_probe_passing),
                        len(_serve_entries),
                        _probe_rate,
                        recall_sanity_threshold,
                    )

                    if _probe_rate >= recall_sanity_threshold:
                        for _fse in _serve_entries:
                            self.store.move(_fse["key"], tier)
                        last_per_key_by_tier[tier] = None
                        tiers_rebuilt.append(tier)
                        # mark fast-start tier complete (no checkpoint-N dir
                        # exists; _mark_tier_complete stores None for reload-from-
                        # production-slot on a subsequent crash-resume).
                        self._mark_tier_complete(tier, None)
                        logger.info(
                            "_run_fold[main_tiers]: fast-start graduation accepted"
                            " for %s (%d keys rebooked)",
                            tier,
                            len(_serve_entries),
                        )
                        continue

                    logger.warning(
                        "_run_fold[main_tiers]: fast-start probe FAILED for %s"
                        " (%.3f < %.3f) — falling back to train-from-scratch",
                        tier,
                        _probe_rate,
                        recall_sanity_threshold,
                    )
                    _fast_start_graduating.discard(tier)
                    job.entries = list(_serve_entries)
                    for _fse in _serve_entries:
                        self.store.move(_fse["key"], tier)
                    if tier in self.model.peft_config:
                        if backup_name in self.model.peft_config:
                            _sw(self.model, backup_name)
                        self.model.delete_adapter(tier)
                    self.model = create_adapter(self.model, tier_cfg, tier)
                    _sw(self.model, tier)
                    logger.debug(
                        "_run_fold[main_tiers]: recreated fresh adapter %s for fallback training",
                        tier,
                    )

                prior_job = None
                recall_state = None
                if trainer is not None:
                    prior_job = trainer._current_job
                    trainer._current_job = job
                    trainer._set_is_training(True)
                try:
                    _tier_metrics, recall_state = self._train_tier_adapter(
                        job.entries,
                        adapter_name=tier,
                        adapter_config=tier_cfg,
                        training_config=refresh_training_config,
                        output_dir=self.output_dir / "consolidation_refresh" / tier,
                        run_name=f"consolidate-{tier}",
                        phase_name=f"consolidate-{tier}",
                        num_epochs=refresh_epochs,
                        retain_scratch_until_external_commit=True,
                    )
                    if _tier_metrics is not None:
                        if _tier_metrics.get("aborted"):
                            logger.info(
                                "_run_fold[main_tiers]: training aborted on tier %s "
                                "— restoring all tiers from backups",
                                tier,
                            )
                            from paramem.models.loader import copy_adapter_weights as _copy_w

                            for _t in ("episodic", "semantic", "procedural"):
                                _backup = f"{_t}_backup"
                                if (
                                    _backup in self.model.peft_config
                                    and _t in self.model.peft_config
                                ):
                                    _copy_w(self.model, src=_backup, dst=_t)
                            self._current_interim_stamp = None  # type: ignore[assignment]
                            raise AbortedDuringConsolidation(f"training aborted on tier {tier!r}")
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
            if self.store.replay_enabled:
                passing_sets_by_tier: dict[str, "set[str] | None"] = {}
                for _tier in ("episodic", "semantic", "procedural"):
                    _lpk = last_per_key_by_tier.get(_tier)
                    if _lpk is not None:
                        _serve_keys = {e["key"] for e in serve_assignment[_tier]}
                        passing_sets_by_tier[_tier] = {
                            r["key"] for r in _lpk if r["exact_match"]
                        } & _serve_keys
                    else:
                        passing_sets_by_tier[_tier] = None

                self._reset_main_tier_registries_and_simhashes(
                    serve_assignment,
                    passing_sets_by_tier,
                    soft_stale_by_tier=soft_stale_by_tier,
                )
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

            if self.store.replay_enabled and tiers_rebuilt:
                try:
                    self._persist_fold(scope, window_stamp_override=window_stamp_override)
                except Exception:
                    self._current_interim_stamp = None  # type: ignore[assignment]
                    raise
                logger.info("_run_fold[main_tiers]: merged main weights persisted+verified")
                # Clean fold-resume marker + retained scratch checkpoints after
                # _save_adapters succeeds.  On _save_adapters FAILURE (the except
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

            unload_interim_adapters(self.model, self.output_dir)
            logger.info("_run_fold[main_tiers]: interim adapters unloaded")

            if router is not None:
                try:
                    router.reload()
                    logger.info("_run_fold[main_tiers]: router reloaded")
                except Exception:
                    logger.exception("_run_fold[main_tiers]: router reload failed")

            for backup_name in ("episodic_backup", "semantic_backup", "procedural_backup"):
                if backup_name in self.model.peft_config:
                    self.model.delete_adapter(backup_name)
                    logger.debug("_run_fold[main_tiers]: unloaded backup adapter %s", backup_name)

            if "episodic" in self.model.peft_config:
                from paramem.models.loader import switch_adapter as _sw2

                _sw2(self.model, "episodic")

            _train_tiers = ("episodic", "semantic", "procedural")
            _train_tier_delta = self._build_tier_delta(
                active_before=_train_active_before,
                active_after={t: len(serve_assignment.get(t, [])) for t in _train_tiers},
                minted_by_tier=minted_by_tier,
            )
            self._debug_writer.on_tier_delta(_train_tier_delta)

            self._current_interim_stamp = None  # type: ignore[assignment]

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
                "keys_per_tier": {t: len(v) for t, v in serve_assignment.items()},
                "tier_keyed": serve_assignment,
                "recall_per_tier": recall_per_tier,
                "rolled_back": False,
                "rollback_tier": None,
                "tier_delta": _train_tier_delta,
            }
        finally:
            self.merger.reset_graph()

    def consolidate_simulate_fold(self, *, housekeeping: bool = False) -> dict:
        """Thin public entry for the simulate full fold — routes to :meth:`_run_fold`.

        Replaces the deleted ``consolidate_interim_to_canonical_graph``.  Builds the
        :class:`FoldScope` for the ``graph_json`` persist venue and delegates all logic
        to :meth:`_run_fold`.  Callers (``run_housekeeping``, ``app.py``) never
        construct :class:`FoldScope` directly (layering rule).

        Args:
            housekeeping: Forwarded to :meth:`_run_fold`; bypasses the noop gate when
                ``True`` so the re-groomed graph is persisted even with no interim slots.
        """
        if housekeeping:
            _hk_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            self._current_interim_stamp = f"housekeeping_{_hk_ts}"
        else:
            self._current_interim_stamp = None  # type: ignore[assignment]

        return self._run_fold(
            FoldScope(
                name="full",
                source="disk",
                persist="graph_json",
                tier=None,
                extra_relations_source="disk",
                defer=False,
                tag_new=False,
                normalize=(self.config.refinement_normalization == "on"),
                enrich=(self.config.refinement_enrichment == "on" and self.config.sota_enabled),
                promote=False,
                tier_floor=False,
                subtractive_scope="fold",
            ),
            housekeeping=housekeeping,
        )

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

        - **simulate**: calls :meth:`consolidate_simulate_fold`
          with ``housekeeping=True``.
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
            return self.consolidate_simulate_fold(housekeeping=True)
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
        """Promote episodic keys whose reinforcement_count has reached the promotion threshold.

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

                # Resolve endpoint surface from node attributes["name"].
                # For speaker subjects: _endpoint_str returns the node key (lowercase
                # speaker{N}); _synth_speaker_entities emits Entity(name=speaker_id)
                # which refreshes attributes["name"] to the lowercase speaker_id during
                # _merge_registry_relations.  So _subj_display yields the lowercase
                # speaker_id for speaker subjects.
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
                        # speaker_id and this edge is a SOTA-enrichment edge.  Inherit
                        # from the subject's UNIQUE non-empty speaker predecessor (1-hop,
                        # direct in-edges).  Exactly one distinct speaker → use it;
                        # zero or ≥2 → keep "".
                        # Extraction concept-edges (no edge_source / different value)
                        # keep the existing "" terminal — deliberate unattributed facts
                        # (e.g. company-location) must NOT be attributed to a speaker.
                        _subj_sid = self._unique_speaker_predecessor(_t_subj)

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
                rec = {
                    "entry": entry,
                    "tier": tier,
                    "canon_subj": _t_subj,
                    "canon_obj": _t_obj,
                    "predicate": pred,
                    "relation_type": _rt,
                    "speaker_id": _subj_sid,
                    "session_ids": _rec_session_ids,
                    # Real session wall-clock carried from the edge; sourced from
                    # session_graph.timestamp at ingest via merger._upsert_relation.
                    # Never fabricate now() here.
                    "last_seen": _t_data.get("last_seen", ""),
                    "first_seen": _t_data.get("first_seen", ""),
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
                        relation_type=_rt,
                        reinforcement_count=1,
                        last_reinforced_cycle=self.cycle_count,
                        last_seen=_t_data.get("last_seen", ""),
                        first_seen=_t_data.get("first_seen", ""),
                        allow_empty_speaker=(_subj_sid == ""),
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
        (``store.get(key)``).  When the entry is absent (hydration miss on a
        LIVE active key, e.g. under ``boot_degraded``), the key is logged as
        an orphan and skipped.  Bookkeeping never carries SPO.

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
                # Hydration-miss — entry cache is empty for this live key.
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

    def _synth_speaker_entities(self, relations: "list[Relation]") -> "list":
        """Synthesise :class:`~paramem.graph.schema.Entity` objects for speaker-attributed subjects.

        For each :class:`Relation` in *relations* whose ``speaker_id`` is non-empty
        and whose ``subject == speaker_id`` (plain equality — both are lowercase
        ``speaker{N}`` under the lowercase-uniform design), emit one
        :class:`~paramem.graph.schema.Entity` with ``entity_type="person"`` and
        the matching ``speaker_id``.  Deduplicates by ``speaker_id`` so exactly
        one entity is produced per unique speaker.

        Non-speaker subjects (``speaker_id == ""`` OR ``subject != speaker_id``)
        are skipped; their nodes retain no ``speaker_id`` attribute, which resolves
        to ``""`` in the keyed-walk (the correct default for non-person nodes).

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
                # Both subject and speaker_id are lowercase speaker{N} — plain ==
                # is sufficient.  Non-speaker subjects are skipped.
                if _r.subject == _r.speaker_id:
                    entities.append(
                        _Entity(
                            # Use _r.subject (== _r.speaker_id) as the entity name.
                            # This refreshes attributes["name"] to the lowercase
                            # speaker_id on the existing speaker node.
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
        timestamp: str = "",
        resolve_contradictions: bool = False,
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

        The gradient-checkpointing guard fires when ``resolve_contradictions`` is
        True and a model is present — the contradiction path calls
        ``model.generate()``.  Simulate callers may omit the model entirely.

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
            timestamp: The session timestamp passed to :class:`SessionGraph`.
                Default ``""`` (empty string) for all HISTORICAL callers
                (recon/registry-true/simulate-disk/enrichment), which ensures the
                merger's ``relation.last_seen or timestamp`` fallback yields ``""``
                for legacy relations — triggering the any-empty COEXIST rule instead
                of fabricating a NOW recency value.  Pass ``datetime.now()`` only for
                genuinely FRESH sessions where an empty ``last_seen`` should resolve
                to the current wall-clock time.  Currently all callers are historical
                and omit this parameter (receive the ``""`` default).
            resolve_contradictions: Forwarded to
                :meth:`~paramem.graph.merger.GraphMerger.merge`.  Default
                ``False`` (no cardinality resolution).  Set ``True`` when the config
                ``refinement_contradiction == "on"``.
        """
        if not relations:
            return
        entities = self._synth_speaker_entities(relations)
        _session = SessionGraph(
            session_id=session_id,
            timestamp=timestamp,
            entities=entities,
            relations=relations,
        )
        # The gradient-checkpointing guard fires when resolve_contradictions is
        # True and a model is present — the contradiction path calls model.generate().
        _has_model = getattr(self, "model", None) is not None
        _needs_guard = _has_model and resolve_contradictions
        if _needs_guard:
            self._disable_gradient_checkpointing()
        try:
            self.merger.merge(
                _session,
                resolve_contradictions=resolve_contradictions,
            )
        finally:
            if _needs_guard:
                self._enable_gradient_checkpointing()
        logger.info(
            "_materialize_consolidation_graph: re-merged %d %s into keying graph",
            len(relations),
            log_label,
        )

    def _materialize_consolidation_graph(
        self,
        *,
        source: "Literal['weights', 'disk']" = "weights",
        tier: "str | None" = None,
        keys: "list[str] | None" = None,
        extra_relations: "list[Relation] | None" = None,
        resolve_contradictions_recon: bool = False,
        resolve_contradictions_extra: bool = False,
    ) -> "tuple[set[str], list[Relation]]":
        """Reconstruct active keys from adapter weights and re-merge registry-true relations.

        This is the *Materialize* stage of the fold pipeline:

        1. Probe every active key from adapter weights via :func:`reconstruct_graph`
           (``strict=False``).  Skipped when ``source="disk"`` (no weight reconstruction
           for the simulate venue — disk relations are supplied via ``extra_relations``).
        2. Compute ``recall_miss_keys`` — keys whose reconstructed SPO disagrees with
           registry-true SPO, or whose reconstruction failed outright.  The set is
           computed against ``store.all_active_keys()`` BEFORE the graph reset, so
           only registered keys can appear in the miss set.
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
            source: Materialize axis — ``"weights"`` (default) or ``"disk"``.

                - ``"weights"``: today's behaviour byte-for-byte.  Calls
                  :func:`reconstruct_graph`, computes ``recall_miss_keys``, resets
                  the merger, builds registry-true :class:`Relation` objects via
                  :meth:`_build_registry_true_relations` and re-merges them.
                - ``"disk"``: skips weight reconstruction entirely (no adapter weights
                  available in the simulate venue).  ``recall_miss_keys`` is the empty
                  ``set()`` — a retrain signal is meaningless when no retraining
                  occurs.  The disk :class:`Relation` objects are supplied by the
                  caller via ``extra_relations`` and enter the merge via the same
                  ``extra_relations`` channel.  :meth:`_build_registry_true_relations`
                  is NOT called (returns ``[]`` from the empty simulate store — calling
                  it would silently zero the merge input).  ``recon_relations`` returns
                  ``[]``, making :meth:`_refine_consolidation_graph`'s recurrence-bump
                  guard a no-op (correct: simulate store has no registered keys to bump).

                **Persist contract:** for ``source="disk"`` the persist tail
                MUST write ``save_memory_to_disk(self.merger.graph, path)`` DIRECTLY —
                NOT derive the graph from ``tier_keyed``.  Disk relations carry ``ik_key``;
                the keyed branch of ``_build_all_edge_entries_into`` does ``store.get(key)``
                against the empty simulate store and silently skips unmatched edges, making
                ``tier_keyed`` empty.  The direct graph write is the only correct path.
            tier: Forwarded to :func:`reconstruct_graph` as ``tier``.  When
                ``None`` (the default), all tiers are probed — byte-identical to
                the original inline fold behaviour.  Ignored when ``source="disk"``.
            keys: Forwarded to :meth:`_build_registry_true_relations` as ``keys``.
                When ``None`` (the default), all active keys are processed —
                byte-identical to the original inline fold behaviour.  Ignored when
                ``source="disk"``.
            extra_relations: Optional list of :class:`Relation` objects to merge
                into the fresh keying graph after the registry-true re-merge.
                Intended for the interim mini-fold: the caller captures the
                pending-session relations from ``self.merger.graph`` BEFORE calling
                this method (since the reset inside will wipe them) and passes them
                here so they survive the reset and co-reside with the slot's
                recalled facts.  The fold caller passes ``None`` (no-op).
                For ``source="disk"`` callers, pass the disk-loaded :class:`Relation`
                objects here — they are the ONLY merge input for the disk path.
            resolve_contradictions_recon: Forwarded to
                :meth:`_merge_registry_relations` for the registry-true recon
                merge.  Driven by ``config.refinement_contradiction == "on"``.
                At fold, ``timestamp=""`` is passed to the merger so legacy
                relations (``last_seen=""``) trigger the any-empty COEXIST rule
                rather than fabricating a NOW value — a safe no-op for legacy keys.
                Ignored when ``source="disk"`` (no recon merge performed).
            resolve_contradictions_extra: Forwarded to
                :meth:`_merge_registry_relations` for the ``extra_relations``
                (pending-session) merge.  Driven by
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
              Always ``[]`` for ``source="disk"``.
        """
        # --- Disk-source path: skip weight reconstruction entirely ---
        # For the simulate venue (source="disk") there are no adapter weights to probe.
        # The disk-loaded Relations are supplied via extra_relations and enter the merge
        # through the existing extra_relations channel below.
        # recall_miss_keys is empty: a retrain signal is meaningless for a venue that
        # does not retrain.
        # _build_registry_true_relations is NOT called: it reads self.store (empty in
        # simulate) and would return [], silently zeroing the merge input.
        if source == "disk":
            self.merger.reset_graph()
            logger.info(
                "_materialize_consolidation_graph(source=disk): keying graph reset;"
                " merging %d disk relations via extra_relations channel",
                len(extra_relations) if extra_relations else 0,
            )
            # Emit debug snapshot of the empty (pre-merge) graph so the artifact
            # chain matches the weights path (reconstructed → merged → enriched).
            self._debug_writer.on_fold_graph(self.merger, label="reconstructed")
            # Merge the disk-loaded relations through the extra_relations channel.
            # resolve_contradictions mirrors the train path: driven by
            # config.refinement_contradiction.  With timestamp="" (default), legacy
            # relations (last_seen="") trigger the any-empty COEXIST rule rather
            # than fabricating a NOW recency value — simulate==train invariant holds.
            self._merge_registry_relations(
                extra_relations or [],
                session_id="__simulate_consolidation_merge__",
                log_label="disk relations (simulate full fold)",
                resolve_contradictions=(self.config.refinement_contradiction == "on"),
            )
            self._debug_writer.on_fold_graph(self.merger, label="merged")
            return set(), []

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
        recall_miss_keys: set[str] = {f["key"] for f in recon_result.failures}
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
        # resolve_contradictions_recon is driven by config.refinement_contradiction.
        # When "on": the merger may retire strictly-older registry-true edges; but since
        # timestamp="" is passed (default), legacy relations (last_seen="") always trigger
        # the any-empty COEXIST rule — no removal for legacy keys.  Only fully-dated
        # rival sets participate in recency selection.
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
        # resolve_contradictions_recon is driven by config.refinement_contradiction.
        # timestamp="" (default) ensures legacy keys (last_seen="") hit the any-empty
        # COEXIST rule rather than fabricating a NOW recency value.
        self._merge_registry_relations(
            recon_relations,
            session_id="__full_consolidation_recon__",
            log_label="reconstructed triples",
            resolve_contradictions=resolve_contradictions_recon,
        )

        # --- Re-merge extra_relations (interim mini-fold pending-session content) ---
        # INVARIANT: extra_relations participate in MERGE / Case-1-adopt ONLY.
        # They are NOT included in recall_miss_keys (computed above, before the reset).
        # extra_relations=None and extra_relations=[] are both valid no-ops (fold caller
        # passes None; interim passes the pending-session relations from merger.graph).
        # resolve_contradictions_extra: driven by config.refinement_contradiction.
        # At fold extra_relations=None so this merge is a no-op.
        self._merge_registry_relations(
            extra_relations or [],
            session_id="__interim_pending_sessions__",
            log_label="extra (pending-session) relations",
            resolve_contradictions=resolve_contradictions_extra,
        )

        # Debug: snapshot the merged graph (after re-merge, before enrichment).
        # Emits even when recon_relations is empty so the fold always produces a
        # merged snapshot.  Self-gated; no-op when save_cycle_snapshots=False.
        self._debug_writer.on_fold_graph(self.merger, label="merged")

        return recall_miss_keys, recon_relations

    def _refine_consolidation_graph(
        self,
        recon_relations: "list[Relation]",
        *,
        normalize: bool = False,
        enrich: bool = False,
    ) -> None:
        """Run graph normalization, enrichment, and recurrence bumps after the Materialize stage.

        This is the *Refine* stage of the fold pipeline:

        1. Optionally run the whole-graph local-model normalization pass via
           :meth:`_run_graph_normalization` (predicate alignment + entity merge +
           semantic dedup).  Runs when ``normalize`` is ``True``.  Local model;
           no cloud dependency.
        2. Optionally run SOTA graph enrichment via :meth:`_run_graph_enrichment`
           (additive second-order discovery).  Runs when ``enrich`` is ``True``.
           Cloud dependency.
        3. Emit a debug snapshot ("enriched") after the refine step (or immediately
           when both stages are skipped).
        4. If ``recon_relations`` is non-empty, scan ``merger.reinforcements`` for
           Case-1 duplicate-SPO collapses and call
           :meth:`~paramem.memory.store.MemoryStore.bump_recurrence` for each
           surviving key.  The recurrence-bump runs regardless of *normalize* or
           *enrich* — it reflects duplicate-SPO collapses from the merge, which
           happen at every level that merges.

        Args:
            recon_relations: The list of registry-true :class:`Relation` objects
                produced by :meth:`_materialize_consolidation_graph`.  Used only
                as a boolean guard — when empty, the recurrence-bump loop is
                skipped (no re-merge was performed so ``merger.reinforcements``
                will be empty too).
            normalize: When ``True``, run :meth:`_run_graph_normalization`
                (local-model predicate alignment + entity merge + dedup).
                Callers pass ``normalize=scope.normalize``.
                Default ``False``.
            enrich: When ``True``, run :meth:`_run_graph_enrichment` (cloud-SOTA
                additive discovery).  Callers pass ``enrich=scope.enrich`` (the
                scope field is set at construction to
                ``refinement_enrichment=="on" and sota_enabled``).
                Default ``False``.
        """
        # --- Whole-graph local-model normalization (predicate synonym collapse) ---
        # Runs at level light+ for full-fold only (interim scopes set normalize=False).
        # The local model sees the full merged graph and collapses synonym predicates.
        # Removals flow into merger.removal_ledger and are consumed by
        # _apply_subtractive_removals_to_store after this method returns.
        if normalize:
            norm_result = self._run_graph_normalization()
            if not norm_result.get("skipped"):
                logger.info(
                    "graph_normalization complete: chunks=%d groups_collapsed=%d edges_retired=%d",
                    norm_result.get("chunks", 0),
                    norm_result.get("groups_collapsed", 0),
                    norm_result.get("edges_retired", 0),
                )

        # --- Graph-level SOTA enrichment (additive, cloud, HELD) ---
        # Runs AFTER normalization so enrichment operates on the already-normalized
        # graph.  The external/SOTA boundary is handled inside _run_graph_enrichment
        # (per-chunk except catches network errors and continues), so a network
        # failure degrades gracefully there.  A programming error here propagates
        # and aborts the fold (sessions stay pending/retriable) — correct behaviour.
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

        # Debug: snapshot the refined graph (after normalization + enrichment, or
        # immediately when both are skipped at level off).
        # Self-gated; no-op when save_cycle_snapshots=False.
        self._debug_writer.on_fold_graph(self.merger, label="enriched")

        if recon_relations:
            # --- Reinforcement bump: Case-1 duplicate-SPO collapses ---
            # merger.reinforcements contains the surviving ik_key for every Case-1
            # collision fired during the re-merge.  A collision means two active keys
            # shared the same (s,p,o) — the incoming key drifts and the existing
            # edge's key is the survivor.  The survivor's reinforcement_count
            # represents how many times this fact was independently extracted
            # (and re-keyed) across sessions before this fold collapsed the duplicates.
            # merger.reinforcements is now dict[ik_key, (last_seen, first_seen)] —
            # the freshest last_seen and earliest first_seen are carried directly
            # from the edge so bump_recurrence can advance bookkeeping without
            # fabricating now().
            _reinforcements: dict[str, tuple[str, str]] = getattr(self.merger, "reinforcements", {})
            for _rein_key, (_rein_ls, _rein_fs) in _reinforcements.items():
                if _rein_key:
                    self.store.bump_recurrence(
                        _rein_key,
                        cycle=self.cycle_count,
                        timestamp=_rein_ls,
                        first_seen=_rein_fs,
                    )
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
        consume_pending: bool = False,
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
                lock via submit()).  Required for the per-tier re-arm pattern.
            router: Router instance whose reload() is called at the end of the
                atomic finalize sequence.  Optional — skipped when None.
            recall_sanity_threshold: Override for the minimum recall rate for the
                post-save disk-integrity probe in ``_save_adapters``.  When
                ``None`` (default), the value is read from
                ``self.config.recall_sanity_threshold``.
            refresh_epochs: Number of training epochs for the full per-tier
                rebuild (default 30; matches the per-key baseline from Tests 1-7b).
            consume_pending: When ``True``, the fold captures pending-session
                relations from ``merger.graph`` and injects them into the fold via
                the ``extra_relations`` channel.  The full-fold :class:`FoldScope`
                sets ``consume_pending=True`` and
                ``extra_relations_source="pending"``.  When ``False`` (default),
                the fold ignores pending-session relations (normal non-consume
                behaviour).  The caller is responsible for translating
                server-schedule config (e.g. ``config.consolidation.max_interim_count
                == 0``) into this boolean decision before calling.
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
        from paramem.server.gpu_lock import _gpu_thread_lock

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
        # Cleared inside _run_fold (main_tiers path) at success / abort.
        if housekeeping:
            _hk_ts_adp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            self._current_interim_stamp = f"housekeeping_{_hk_ts_adp}"
        else:
            self._current_interim_stamp = None  # type: ignore[assignment]

        return self._run_fold(
            FoldScope(
                name="full",
                source="weights",
                persist="main_tiers",
                tier=None,
                extra_relations_source="pending" if consume_pending else "none",
                defer=False,
                tag_new=False,
                normalize=(self.config.refinement_normalization == "on"),
                enrich=(self.config.refinement_enrichment == "on" and self.config.sota_enabled),
                promote=True,
                tier_floor=True,
                subtractive_scope="fold",
                consume_pending=consume_pending,
            ),
            trainer=trainer,
            router=router,
            recall_sanity_threshold=recall_sanity_threshold,
            refresh_epochs=refresh_epochs,
            housekeeping=housekeeping,
            window_stamp_override=window_stamp_override,
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
        retain_scratch_until_external_commit: bool = False,
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
            retain_scratch_until_external_commit: Forwarded verbatim to
                :func:`paramem.training.trainer.train_adapter`.  When ``True``,
                the success path skips ``_clean_scratch`` / ``staging_resume.json``
                deletion so the durable ``checkpoint-N`` directory survives until
                the fold's own ``_save_adapters`` external commit.  Default
                ``False`` preserves clean-on-success for all other callers (BG
                trainer, replay, migration, interim).

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
            retain_scratch_until_external_commit=retain_scratch_until_external_commit,
        )
        return metrics, recall_state


def _mentions_any(text: str, terms: set[str]) -> bool:
    """Check if text mentions any of the given terms."""
    text_lower = text.lower()
    return any(term in text_lower for term in terms)


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
    keys ``subject``, ``predicate``, ``object``, ``relation_type``, and
    ``speaker_id``.  The ``predicate`` field is taken directly from the edge
    ``"predicate"`` attribute; ``relation_type`` defaults to ``"factual"`` when
    absent; ``speaker_id`` defaults to ``""`` when absent.

    The ``speaker_id`` field allows the SOTA enrichment prompt to identify speaker
    endpoints and apply the speaker↔speaker exception (emit BOTH directions of a
    symmetric relation when both endpoints are speakers).

    Args:
        subgraph: A NetworkX (Multi)DiGraph subgraph view or instance.

    Returns:
        List of ``{"subject": str, "predicate": str, "object": str,
        "relation_type": str, "speaker_id": str}`` dicts, one per directed edge.
    """
    triples = []
    for src, tgt, data in subgraph.edges(data=True):
        triples.append(
            {
                "subject": str(src),
                "predicate": str(data.get("predicate", "")),
                "object": str(tgt),
                "relation_type": str(data.get("relation_type", "factual")),
                "speaker_id": str(data.get("speaker_id", "")),
            }
        )
    return triples
