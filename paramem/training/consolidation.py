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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import cached_property
from pathlib import Path
from typing import Literal, Optional

from torch.utils.data import Dataset

from paramem.graph.extraction_pipeline import ExtractionConfig, ExtractionPipeline
from paramem.graph.extractor import (
    PROVIDER_KEY_ENV,
    _graph_enrich_with_sota,
)
from paramem.graph.merger import GraphMerger, _normalize_predicate
from paramem.graph.phase_trace import extraction_trace, phase_trace
from paramem.graph.qa_generator import (
    partition_relations,
)
from paramem.graph.schema import SessionGraph
from paramem.graph.scoring import (
    PromotionScorer,
)
from paramem.memory.entry import (
    assign_keys,
    build_registry,
    compute_simhash,
    format_entry_training,
    probe_entry,
)
from paramem.memory.persistence import save_registry
from paramem.models.loader import atomic_save_adapter, switch_adapter
from paramem.server.vram_guard import safe_empty_cache
from paramem.training.curriculum import CurriculumSampler
from paramem.training.key_registry import KeyRegistry
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


@dataclass
class CycleResult:
    """Results from a single consolidation cycle."""

    cycle_index: int
    session_id: str
    entities_extracted: int = 0
    relations_extracted: int = 0
    nodes_promoted: int = 0
    nodes_decayed: int = 0
    nodes_retained: int = 0
    episodic_train_loss: Optional[float] = None
    semantic_train_loss: Optional[float] = None
    procedural_train_loss: Optional[float] = None
    episodic_recall: Optional[float] = None
    semantic_recall: Optional[float] = None
    wall_clock_seconds: float = 0.0
    promoted_nodes: list[str] = field(default_factory=list)
    decayed_nodes: list[str] = field(default_factory=list)


class TrialActiveError(RuntimeError):
    """Raised by ConsolidationLoop.guard_trial_state when a migration TRIAL is active.

    Bubbles up to /scheduled-tick and /consolidate handlers, which return
    409 trial_active.  Experiment scripts that do not carry server _state
    never trigger this error (guard is a no-op when state is None).
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
        graph_path: Optional[str | Path] = None,
        extraction_temperature: float = 0.0,
        extraction_max_tokens: int = 8192,
        extraction_plausibility_max_tokens: int = 8192,
        save_cycle_snapshots: bool = True,
        snapshot_dir: str | Path | None = None,
        run_id: str | None = None,
        persist_graph: bool = True,
        prompts_dir: str | Path | None = None,
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
    ):
        # Optional callable that returns the server ``_state`` dict.  When
        # provided, ``run_cycle`` calls ``self.guard_trial_state(state_provider())``
        # at entry to block new consolidation cycles during a migration TRIAL.
        # Experiment scripts pass nothing (default ``None``) so the guard is a
        # no-op and experiment paths are unaffected.
        self.state_provider = state_provider
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
        self.persist_graph = persist_graph
        self.prompts_dir = prompts_dir
        # Entity-level promotion requires a persistent graph for cross-restart
        # recurrence tracking. When the graph is transient (persist_graph=False),
        # the caller must handle promotion externally (e.g. key-level promotion).
        self.enable_entity_promotion = persist_graph
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
        )
        self.scorer = PromotionScorer()
        self.last_session_graph = None
        if graph_path:
            self.graph_path = Path(graph_path)
        elif self.persist_graph:
            self.graph_path = self.output_dir / "cumulative_graph.json"
        else:
            self.graph_path = None

        # Load existing graph if persistence is enabled
        if self.persist_graph and self.graph_path.exists():
            self.merger.load_graph(self.graph_path)

        # Ensure both adapters exist on the model
        self.model = self._ensure_adapters()

        # Replay pools: track relations available for replay per adapter
        # Each entry: {"question": str, "answer": str}
        self.episodic_replay_pool: list[dict] = []
        self.semantic_replay_pool: list[dict] = []

        # Track which nodes have been promoted (avoid re-promoting)
        self.promoted_nodes: set[str] = set()

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
        # (speaker_id, subject, predicate) -> key for contradiction detection.
        # Domain-specific index for procedural extraction; lives on the loop
        # (not the store) because its scope is procedural-only and its key
        # shape is a tuple, not a tier name.
        self.procedural_sp_index: dict[tuple[str, str, str], str] = {}
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

        # Per-key session counts (for key-level promotion in server mode)
        self.key_sessions: dict[str, int] = {}
        # Keys already promoted (prevent re-promotion after restart)
        self.promoted_keys: set[str] = set()

        # RAM-only queue for max_interim_count==0 (queue-until-consolidation).
        # Facts extracted when no interim adapter is configured accumulate here
        # until the next consolidation cycle (Step 7) folds them into the mains.
        # Privacy invariant: this list is never written to disk — what isn't
        # trained doesn't exist.  Snapshot persistence is deferred (Step 7).
        # TODO(Step 7): consume pending_interim_triples at the start of
        # consolidate_interim_adapters() before training the full key set.
        self.pending_interim_triples: list[dict] = []

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
        """Restore key-level metadata from persisted key_metadata.json.

        Used when persist_graph=False to restore cycle count, promoted
        keys, and per-key session counts across server restarts.

        Reads ``sessions_seen``, ``speaker_id``, and ``first_seen_cycle``
        from each key entry.  Per the wipe invariant (2026-05-14):
        ``key_metadata.json`` is bookkeeping, not a recovery source.  Keys
        in the metadata file whose tier registry is not on disk anymore
        are treated as orphans and dropped — the slot is the source of
        truth for active keys.  Pre-refactor files that lack
        ``speaker_id`` and ``first_seen_cycle`` are upgraded in-memory by
        filling ``""`` / ``0`` respectively; the next
        :func:`_save_key_metadata` call writes the complete shape.
        """
        self.cycle_count = metadata.get("cycle_count", 0)
        legacy_upgrade_count = 0
        orphan_count = 0
        for key, key_meta in metadata.get("keys", {}).items():
            tier = self.store.tier_for_active_key(key)
            if tier is None:
                # No tier owns this key — slot was wiped or never existed.
                # Drop the metadata entry; the next _save_key_metadata
                # write will not re-emit it.
                orphan_count += 1
                continue
            self.key_sessions[key] = key_meta.get("sessions_seen", 0)
            # Populate the store's entry slot with partial metadata so the
            # boot rebuild (Critical #5) can join against it without
            # missing fields.
            cache_entry = self.store.setdefault_entry(tier, key, {})
            if "speaker_id" not in key_meta or "first_seen_cycle" not in key_meta:
                legacy_upgrade_count += 1
            cache_entry["speaker_id"] = key_meta.get("speaker_id", "")
            cache_entry["first_seen_cycle"] = key_meta.get("first_seen_cycle", 0)
        # promoted_keys is similarly slot-owned — drop entries whose tier
        # is gone so the next save doesn't re-emit them.
        raw_promoted = set(metadata.get("promoted_keys", []))
        self.promoted_keys = {
            k for k in raw_promoted if self.store.tier_for_active_key(k) is not None
        }
        if orphan_count:
            logger.info(
                "seed_key_metadata: dropped %d orphan key(s) (metadata present, no tier registry)",
                orphan_count,
            )
        if legacy_upgrade_count:
            logger.info(
                "key_metadata.json legacy upgrade: %d keys missing speaker_id/first_seen_cycle; "
                "filled with '' / 0; run /consolidate to refresh",
                legacy_upgrade_count,
            )
        logger.info(
            "Seeded key metadata: cycle=%d, %d promoted, %d keys tracked",
            self.cycle_count,
            len(self.promoted_keys),
            len(self.key_sessions),
        )

    @staticmethod
    def dedup_episodic(qa_list: list[dict]) -> list[dict]:
        """Deduplicate episodic QA/relation dicts by triple identity.

        Identity key is ``(subject, predicate, object)``.  Case-insensitive;
        first occurrence wins.  Entries missing all identity fields fall back
        to per-object identity so they survive rather than collide.
        """
        seen: set[tuple] = set()
        out: list[dict] = []
        for qa in qa_list:
            subj = (qa.get("subject") or "").strip().lower()
            pred = (qa.get("predicate") or "").strip().lower()
            obj = (qa.get("object") or "").strip().lower()
            key = (subj, pred, obj) if (subj and pred and obj) else ("__unkeyed__", id(qa))
            if key in seen:
                continue
            seen.add(key)
            out.append(qa)
        return out

    @staticmethod
    def dedup_procedural(rels: list[dict]) -> list[dict]:
        """Deduplicate procedural relations by (subject, predicate, object)."""
        seen: set[tuple] = set()
        out: list[dict] = []
        for rel in rels:
            subj = (rel.get("subject") or "").strip().lower()
            pred = (rel.get("predicate") or "").strip().lower()
            obj = (rel.get("object") or "").strip().lower()
            key = (subj, pred, obj) if (subj and pred and obj) else ("__unkeyed__", id(rel))
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
        question: Optional[str] = None,
        answer: Optional[str] = None,
    ) -> dict:
        """Build a uniform ``indexed_key_cache`` cache entry.

        Carries ``subject``/``predicate``/``object`` as the canonical triple
        fields.  Optional ``question`` and ``answer`` are kept for callers that
        carry them (e.g. boot-time seeders reading legacy disk shapes).

        Using this helper for every cache-write site ensures the uniform shape
        is maintained by construction — every downstream reader (promotion-match,
        sp_index, ``_increment_key_sessions``, ``consolidate_interim_adapters``
        triple-lookup) reads the canonical field names.

        Args:
            key: The ``graphN`` / ``procN`` key string.
            subject: Triple subject.
            predicate: Triple predicate.
            object: Triple object.
            speaker_id: Speaker scope.
            first_seen_cycle: Cycle count at first insertion.
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
    # Read-write coupling: the simhash trio and indexed_key_cache return
    # the *live* internal dicts via the store, so existing tests that
    # mutate them (``loop.X_simhash[key] = v``, ``loop.indexed_key_cache[k] = v``)
    # propagate to the store.  ``indexed_key_registry`` returns the store's
    # internal dict for the same reason.
    #
    # TODO(arch-cleanup): delete these six properties + the two `_*` aliases
    # once `tests/` no longer references the legacy names.  Tracked under
    # the architecture-cleanup task.

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
    def episodic_simhash(self) -> dict[str, int]:
        """DEPRECATED: ``self.store.simhashes_in_tier("episodic")``."""
        self._ensure_store()
        return self.store.simhashes_in_tier("episodic")

    @episodic_simhash.setter
    def episodic_simhash(self, value: dict[str, int]) -> None:
        self._ensure_store()
        self.store.replace_simhashes_in_tier("episodic", value)

    @property
    def semantic_simhash(self) -> dict[str, int]:
        """DEPRECATED: ``self.store.simhashes_in_tier("semantic")``."""
        self._ensure_store()
        return self.store.simhashes_in_tier("semantic")

    @semantic_simhash.setter
    def semantic_simhash(self, value: dict[str, int]) -> None:
        self._ensure_store()
        self.store.replace_simhashes_in_tier("semantic", value)

    @property
    def procedural_simhash(self) -> dict[str, int]:
        """DEPRECATED: ``self.store.simhashes_in_tier("procedural")``."""
        self._ensure_store()
        return self.store.simhashes_in_tier("procedural")

    @procedural_simhash.setter
    def procedural_simhash(self, value: dict[str, int]) -> None:
        self._ensure_store()
        self.store.replace_simhashes_in_tier("procedural", value)

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

    def _tier_for_key(self, key: str) -> "str | None":
        """DEPRECATED: ``self.store.tier_for_active_key(key)``."""
        return self.store.tier_for_active_key(key)

    def _reassign_key(self, key: str, new_tier: str) -> None:
        """DEPRECATED: ``self.store.move(key, new_tier)``."""
        self.store.move(key, new_tier)

    # ------------------------------------------------------------------
    # Per-tier registry helpers
    # ------------------------------------------------------------------

    def _tier_registry(self, tier: str) -> KeyRegistry:
        """Return the per-tier registry, creating it lazily.

        Raises ``RuntimeError`` when replay is disabled.  Thin passthrough to
        :meth:`MemoryStore.registry` — kept on the loop because many internal
        sites call it with a stable tier name.
        """
        reg = self.store.registry(tier)
        if reg is None:
            raise RuntimeError("indexed_key_replay not enabled")
        return reg

    def _all_active_keys(self) -> list[str]:
        """Every active key across every registered tier — order is tier-then-insertion."""
        return self.store.all_active_keys()

    def _safe_kp_from_cache(self, key: str) -> dict | None:
        """Return a training-ready keyed-pair from the in-RAM cache, or ``None``.

        Used by the existing-key reconstruction loops as a last-resort
        fallback when ``probe_entry`` / ``probe_key`` fails (typically when
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
                self.merger.merge(session_graph)
                self._triples_since_last_enrichment += len(session_graph.relations)
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
            finally:
                _bt.close()

        # --- SAVE main slots ---
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

    def _maybe_run_interim_enrichment(self) -> None:
        """Fire a mini graph-enrichment pass at sub-interval rollover if the floor is crossed.

        Checks ``graph_enrichment_interim_enabled`` and
        ``_triples_since_last_enrichment >= graph_enrichment_min_triples_floor``
        before calling ``_run_graph_enrichment``.  Any SOTA failure is logged
        but never blocks interim adapter creation — the try/except is boundary
        error handling (external Anthropic API call), not control-flow suppression.

        Called from :meth:`run_consolidation_cycle` at sub-interval rollover
        (when the new stamp minted a fresh interim adapter slot).  Lifted verbatim
        from the former ``_train_extracted_into_interim`` rollover branch so the
        same guards and semantics are preserved exactly.
        """
        if not (
            self.graph_enrichment_interim_enabled
            and self._triples_since_last_enrichment >= self.graph_enrichment_min_triples_floor
        ):
            return

        logger.info(
            "_maybe_run_interim_enrichment: running mini graph-enrichment "
            "(triples_since_last=%d >= floor=%d)",
            self._triples_since_last_enrichment,
            self.graph_enrichment_min_triples_floor,
        )
        try:
            self._run_graph_enrichment()
        except Exception as _mini_exc:
            logger.warning(
                "_maybe_run_interim_enrichment: mini graph-enrichment raised "
                "— interim creation continues: %s",
                _mini_exc,
            )

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
            Calls ``probe_entry`` on each existing key and uses the adapter
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
        # Assign new keys from the incoming relations.
        new_keyed_raw = assign_keys(
            [(r["subject"], r["predicate"], r["object"]) for r in rels],
            start_index=self._indexed_next_index,
            prefix="graph",
        )
        new_keyed: list[dict] = []
        for i, kp in enumerate(new_keyed_raw):
            rel = rels[i] if i < len(rels) else {}
            sid = kp.get("speaker_id") or rel.get("speaker_id", speaker_id)
            entry = self._cache_entry(
                key=kp["key"],
                subject=kp["subject"],
                predicate=kp["predicate"],
                object=kp["object"],
                speaker_id=sid,
                first_seen_cycle=self.cycle_count,
            )
            entry["_new"] = True
            new_keyed.append(entry)
        # NOTE: store.put and _indexed_next_index advancement are intentionally
        # deferred to the caller — see docstring.

        # Gather existing keys for the target tier (full-replay anti-forgetting).
        new_key_set = {kp["key"] for kp in new_keyed}
        existing_tier_keys = [
            k for k in self.store.active_keys_in_tier(adapter_name) if k not in new_key_set
        ]

        existing_keyed: list[dict] = []
        if mode == "train":
            # TRAIN: reconstruct from adapter weights.
            # Keys with no simhash fingerprint cannot be verified by probe_entry
            # (no reference hash to compute confidence against) — fall back to
            # cache immediately for those keys and only probe the rest.
            self._disable_gradient_checkpointing()
            tier_simhash = self.store.simhashes_in_tier(adapter_name)
            keys_with_hash = [k for k in existing_tier_keys if k in tier_simhash]
            keys_without_hash = [k for k in existing_tier_keys if k not in tier_simhash]
            reconstructed: dict[str, dict] = {}
            for key in keys_with_hash:
                recalled = probe_entry(
                    self.model,
                    self.tokenizer,
                    key,
                    registry=tier_simhash,
                    confidence_threshold=0.5,
                )
                if recalled is not None and "failure_reason" not in recalled:
                    cached = self.store.get(key) or {}
                    first_seen = cached.get("first_seen_cycle", self.cycle_count)
                    reconstructed[key] = {
                        "key": key,
                        "subject": recalled["subject"],
                        "predicate": recalled["predicate"],
                        "object": recalled["object"],
                        "speaker_id": cached.get("speaker_id", speaker_id),
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
            # SIMULATE: read from cache — no model.generate calls.
            for key in existing_tier_keys:
                qa = self.store.get(key)
                if qa is not None:
                    first_seen = qa.get("first_seen_cycle", self.cycle_count)
                    existing_keyed.append(
                        {
                            "key": key,
                            "subject": qa["subject"],
                            "predicate": qa["predicate"],
                            "object": qa["object"],
                            "speaker_id": qa.get("speaker_id", speaker_id),
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
    ) -> list[dict]:
        """Assign new procedural keys and gather existing keys for retraining.

        Symmetric counterpart of :meth:`_prepare_episodic_keys_for_tier` for
        the procedural tier.  Always targets the stable ``"procedural"`` main
        adapter — there is no interim tier for procedural.

        Deferred-mutation discipline (TRAIN mode):
            Contradiction detection and sp_index / store mutations are applied
            tentatively before the train call in
            :meth:`run_consolidation_cycle`.  The index counter
            (``_procedural_next_index``) is advanced only after this helper
            returns successfully.

        Simulate mode:
            Applies contradiction retirement and writes the store immediately
            (no train call can fail so no deferral is needed).

        Args:
            rels: Pre-extracted procedural relation dicts, already tagged with
                ``speaker_id`` by ``_tag_speaker_id_defaults``.
            speaker_id: Fallback speaker tag for new keys missing one.
            mode: ``"train"`` or ``"simulate"``.

        Returns:
            ``(new_keyed, existing_keyed, keys_to_retire, new_sp_mappings)``
            as a 4-tuple so ``run_consolidation_cycle`` can apply post-train
            deferred mutations atomically.

            - ``new_keyed``: cache-entry dicts for the new procedural keys.
            - ``existing_keyed``: reconstructed / cache-read dicts for existing
              procedural keys that survive retirement.
            - ``keys_to_retire``: old keys to delete after training succeeds
              (TRAIN mode only — simulate deletes immediately).
            - ``new_sp_mappings``: ``sp_key -> new_key`` entries for
              ``procedural_sp_index`` (TRAIN mode only — simulate writes immediately).
        """
        if not rels:
            return [], [], [], {}

        logger.info("_prepare_procedural_keys_for_tier: %d relations (mode=%s)", len(rels), mode)

        tentative_next = self._procedural_next_index
        entries_keyed = assign_keys(
            [(r["subject"], r["predicate"], r["object"]) for r in rels],
            start_index=tentative_next,
            prefix="proc",
        )
        tentative_next += len(entries_keyed)

        new_keyed: list[dict] = []
        for i, kp in enumerate(entries_keyed):
            rel = rels[i] if i < len(rels) else {}
            rel_speaker = rel.get("speaker_id", speaker_id)
            new_keyed.append(
                self._cache_entry(
                    key=kp["key"],
                    subject=kp["subject"],
                    predicate=kp["predicate"],
                    object=kp["object"],
                    speaker_id=rel_speaker,
                    first_seen_cycle=self.cycle_count,
                )
            )

        # Compute intended mutations (sp_key contradiction check).
        keys_to_retire: list[str] = []
        new_sp_mappings: dict[tuple, str] = {}
        new_key_set = {kp["key"] for kp in new_keyed}
        for kp in new_keyed:
            sp_key = (
                kp["speaker_id"],
                kp["subject"].lower(),
                kp["predicate"].lower(),
            )
            old_key = self.procedural_sp_index.get(sp_key)
            if old_key and self.store.has_simhash("procedural", old_key):
                keys_to_retire.append(old_key)
            new_sp_mappings[sp_key] = kp["key"]

        if mode == "simulate":
            # Apply mutations immediately — no train call that can fail.
            for old_key in keys_to_retire:
                logger.info("_prepare_procedural_keys_for_tier: retiring %s (simulate)", old_key)
                self.store.delete(old_key)
            for kp in new_keyed:
                fingerprint = compute_simhash(
                    kp["key"], kp["subject"], kp["predicate"], kp["object"]
                )
                self.store.put("procedural", kp["key"], kp, simhash=fingerprint)
            for sp_key, new_key in new_sp_mappings.items():
                self.procedural_sp_index[sp_key] = new_key
            self._procedural_next_index = tentative_next
            return new_keyed, [], [], {}

        # TRAIN mode: gather existing keys for reconstruction (deferred mutations).
        retired_set = set(keys_to_retire)
        existing_proc_keys = [
            k
            for k in self.store.simhashes_in_tier("procedural")
            if k not in new_key_set and k not in retired_set
        ]

        self._disable_gradient_checkpointing()
        proc_simhash = self.store.simhashes_in_tier("procedural")
        reconstructed: dict[str, dict] = {}
        for key in existing_proc_keys:
            recalled = probe_entry(
                self.model,
                self.tokenizer,
                key,
                registry=proc_simhash,
                confidence_threshold=0.5,
            )
            if recalled is not None and "failure_reason" not in recalled:
                cached = self.store.get(key) or {}
                reconstructed[key] = {
                    "key": key,
                    "subject": recalled["subject"],
                    "predicate": recalled["predicate"],
                    "object": recalled["object"],
                    "speaker_id": cached.get("speaker_id", speaker_id),
                    "first_seen_cycle": cached.get("first_seen_cycle", self.cycle_count),
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

        return new_keyed, existing_keyed, keys_to_retire, new_sp_mappings

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
        self.merger.merge(session_graph)
        graph = self.merger.graph

        # --- 3. SCORE & CLASSIFY ---
        if self.enable_entity_promotion:
            classification = self.scorer.classify_nodes(
                graph,
                promotion_threshold=self.config.promotion_threshold,
                decay_window=self.config.decay_window,
                current_cycle=self.cycle_count,
            )
            new_promotions = [n for n in classification.promote if n not in self.promoted_nodes]
            result.nodes_promoted = len(new_promotions)
            result.nodes_decayed = len(classification.decay)
            result.nodes_retained = len(classification.retain)
            result.promoted_nodes = new_promotions
            result.decayed_nodes = classification.decay
        else:
            # Entity-level promotion disabled (server uses key-level promotion)
            new_promotions = []

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

        # Promotion is handled at the indexed-key replay layer — no
        # model.generate calls for promoted keys.  ``promote_qa`` is kept
        # as an empty list so downstream call sites that still iterate it
        # are no-ops; the variable will be removed once those sites are
        # audited.
        promote_qa: list[dict] = []

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
                new_promotions=new_promotions if new_promotions else None,
            )
            # cycle_count is now managed by run_consolidation_cycle.
            # Propagate per-tier train losses from the cycle result dict.
            _epi_loss = cycle_result.get("episodic_train_loss")
            _proc_loss = cycle_result.get("procedural_train_loss")
            if _epi_loss is not None:
                result.episodic_train_loss = _epi_loss
            if _proc_loss is not None:
                result.procedural_train_loss = _proc_loss

            if new_promotions:
                semantic_loss = self._run_indexed_key_semantic(new_promotions)
                if semantic_loss is not None:
                    result.semantic_train_loss = semantic_loss
                self.promoted_nodes.update(new_promotions)
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

        # --- 6. TRAIN SEMANTIC (pool-based, only when indexed key replay is not active) ---
        if not self.config.indexed_key_replay_enabled and promote_qa:
            semantic_loss = self._train_adapter_with_replay(
                "semantic",
                promote_qa,
                self.semantic_replay_pool,
                self.config.semantic_replay_weight,
                f"phase3-semantic-cycle{self.cycle_count}",
            )
            result.semantic_train_loss = semantic_loss
            self.promoted_nodes.update(new_promotions)

            # Add promoted relations to semantic replay pool (cap at 100)
            self.semantic_replay_pool.extend(
                {"question": qa["question"], "answer": qa["answer"]} for qa in promote_qa
            )
            if len(self.semantic_replay_pool) > 100:
                self.semantic_replay_pool = self.semantic_replay_pool[-100:]

        # --- 7. DECAY ---
        if self.enable_entity_promotion:
            self._apply_decay(classification.decay)

        # --- 8. SAVE ---
        # persist_graph=True → authoritative production store → encrypted.
        # Cumulative-graph plaintext debug dump (when save_cycle_snapshots is
        # on) is owned by run_consolidation_cycle's on_extraction_end; the
        # legacy run_cycle path does not duplicate it.
        if self.persist_graph:
            self.merger.save_graph(self.graph_path)
        self._save_adapters()

        result.wall_clock_seconds = time.time() - start_time
        logger.info(
            "Cycle %d complete: %d extracted, %d promoted, %d decayed (%.1fs)",
            self.cycle_count,
            result.entities_extracted,
            result.nodes_promoted,
            result.nodes_decayed,
            result.wall_clock_seconds,
        )

        return result

    def _run_indexed_key_semantic(
        self,
        new_promotions: list[str],
    ) -> Optional[float]:
        """Train semantic adapter on promoted indexed keys.

        Collects all keys that belong to promoted entities and trains
        the semantic adapter on them with the indexed format.
        """
        # Collect promoted keys (already transferred in the run_cycle episodic block)
        promoted_set = {n.lower() for n in new_promotions}
        semantic_keyed = []
        # Loop A — newly-promoted keys
        for _tier, key, qa_info in self.store.iter_entries():
            subject = qa_info.get("subject", "").lower()
            obj = qa_info.get("object", "").lower()
            mentions_promoted = (subject and subject in promoted_set) or (
                obj and obj in promoted_set
            )
            if mentions_promoted and self.store.has_simhash("semantic", key):
                semantic_keyed.append(
                    {
                        "key": key,
                        "subject": qa_info["subject"],
                        "predicate": qa_info["predicate"],
                        "object": qa_info["object"],
                    }
                )

        # Loop B — previously promoted keys still in semantic
        seen_semantic_keys = {kp["key"] for kp in semantic_keyed}
        for key in list(self.store.simhashes_in_tier("semantic").keys()):
            if key in seen_semantic_keys:
                continue
            qa = self.store.get(key)
            if qa is not None:
                semantic_keyed.append(
                    {
                        "key": key,
                        "subject": qa["subject"],
                        "predicate": qa["predicate"],
                        "object": qa["object"],
                    }
                )

        if not semantic_keyed:
            logger.info("No promoted keys for semantic training")
            return None

        logger.info("Training semantic adapter on %d promoted keys", len(semantic_keyed))

        switch_adapter(self.model, "semantic")
        examples = format_entry_training(semantic_keyed, self.tokenizer, max_length=1024)
        dataset = self._indexed_dataset(examples)
        training_config = self._make_training_config(num_epochs=self.training_config.num_epochs)
        self._enable_gradient_checkpointing()

        metrics = train_adapter(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            adapter_name="semantic",
            training_config=training_config,
            adapter_config=self.semantic_config,
            wandb_config=self.wandb_config,
            output_dir=self._training_output_dir("semantic"),
            run_name=f"phase4-indexed-semantic-cycle{self.cycle_count}",
            thermal_policy=self._thermal_policy,
            hooks=TrainingHooks(on_shutdown_check=lambda: self.shutdown_requested),
        )

        # Update semantic SimHash registry
        self.store.replace_simhashes_in_tier("semantic", build_registry(semantic_keyed))

        return metrics.get("train_loss")

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
            hooks=TrainingHooks(on_shutdown_check=lambda: self.shutdown_requested),
        )

        return metrics.get("train_loss")

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
        from paramem.training.dataset import _format_inference_prompt

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
            prompt = _format_inference_prompt(item["question"], self.tokenizer)
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

    def _apply_decay(self, decayed_nodes: list[str]) -> None:
        """Remove decayed nodes' relations from the episodic replay pool.

        Decayed memories are not actively deleted from the adapter —
        they fade naturally as the adapter is trained on new content
        without reinforcement.

        When curriculum sampling is enabled, decay is blocked for facts
        that haven't met the minimum exposure guarantee.
        """
        if not decayed_nodes:
            return

        decayed_set = {n.lower() for n in decayed_nodes}
        before = len(self.episodic_replay_pool)

        kept = []
        protected = 0
        for qa in self.episodic_replay_pool:
            if not _mentions_any(qa["question"] + " " + qa["answer"], decayed_set):
                kept.append(qa)
            elif self.curriculum_sampler and not self.curriculum_sampler.can_decay(qa["question"]):
                kept.append(qa)
                protected += 1
            # else: decayed — not added to kept

        self.episodic_replay_pool = kept
        removed = before - len(self.episodic_replay_pool)
        if removed > 0:
            logger.info("Decayed %d relations from episodic replay pool", removed)
        if protected > 0:
            logger.info("Protected %d relations from decay (min exposure not met)", protected)

    def _qa_to_dataset(self, qa_pairs: list[dict]) -> Dataset:
        """Convert relation dicts to a SyntheticQADataset."""
        return SyntheticQADataset(
            examples=[{"question": qa["question"], "answer": qa["answer"]} for qa in qa_pairs],
            tokenizer=self.tokenizer,
            max_length=self.training_config.max_seq_length,
        )

    def _make_training_config(self, num_epochs: int) -> TrainingConfig:
        """Build a TrainingConfig for consolidation training."""
        return TrainingConfig(
            batch_size=self.training_config.batch_size,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            max_seq_length=self.training_config.max_seq_length,
            num_epochs=num_epochs,
            warmup_ratio=self.training_config.warmup_ratio,
            weight_decay=self.training_config.weight_decay,
            gradient_checkpointing=self.training_config.gradient_checkpointing,
            max_grad_norm=self.training_config.max_grad_norm,
            seed=self.training_config.seed,
        )

    def _save_adapters(
        self,
        *,
        recall_sanity_threshold: float = 0.95,
    ) -> None:
        """Save adapters and registries to disk using the I5 reorder (§2.5).

        Saves to two locations:
        - ``output_dir/<tier>/`` — canonical latest state (server use)
        - ``paths.debug/.../training/tiers/<tier>/adapter_weights/`` —
          per-cycle plaintext shadow for inspection (only when
          ``save_cycle_snapshots`` is on; written by
          :meth:`DebugSnapshotWriter.on_main_adapters_saved`).

        I5 ordering (mirrors ``post_session_train`` step 7):
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
          6. SimHash registries written to per-tier paths:
             ``<adapter_dir>/<tier>/simhash_registry.json``.
          7. Per-tier ``indexed_key_registry.json`` written to
             ``<adapter_dir>/<tier>/indexed_key_registry.json``.
          8. ``save_from_bytes`` — flush the identical registry bytes; this
             is the commit signal for ``find_live_slot``.

        Crash semantics: a kill after step 4 but before step 8 leaves the
        new slot present with a manifest stamping the new registry hash,
        while the on-disk registry still carries the old hash.
        ``find_live_slot`` won't match → slot is latent, harmless.

        Args:
            recall_sanity_threshold: Minimum recall rate that the post-save
                disk-integrity probe must achieve.  Forwarded verbatim to
                :meth:`_verify_saved_adapter_from_disk`.  Defaults to 0.95 to
                match the in-RAM recall gates in
                :meth:`consolidate_interim_adapters` and
                :meth:`post_session_train`.
        """
        import hashlib as _hashlib

        from paramem.adapters.manifest import build_manifest_for
        from paramem.memory.interim_adapter import current_full_consolidation_stamp

        fingerprint_cache = getattr(self, "fingerprint_cache", None)
        full_period = getattr(self, "full_consolidation_period_string", "")
        full_window_stamp = current_full_consolidation_stamp(full_period)

        # I5 Step 1+2: Serialise each tier's registry to bytes and hash them — no disk I/O.
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
        ) -> None:
            """Save adapter, probe disk artifact, clean up slot on probe failure.

            Wraps ``atomic_save_adapter`` + ``_verify_saved_adapter_from_disk``
            so that a failed disk-integrity probe deletes the bad slot before
            re-raising.  This prevents a latent corrupted slot from surviving
            until the next rotation or operator inspection.

            Args:
                adapter_name: PEFT adapter name (e.g. ``"episodic"``).
                simhash: Per-tier SimHash registry dict used to filter pairs.
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
                    threshold=recall_sanity_threshold,
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

        # I5 Steps 3–5 per adapter: manifest → slot save.
        # Step 5a follows each slot save: reload from disk and probe recall to
        # catch silent partial writes before ``mark_consolidated`` fires.
        # On probe failure the bad slot is deleted and RuntimeError propagates.
        _save_and_verify("episodic", self.store.simhashes_in_tier("episodic"))
        if "semantic" in self.model.peft_config:
            _save_and_verify("semantic", self.store.simhashes_in_tier("semantic"))
        if "procedural" in self.model.peft_config:
            _save_and_verify("procedural", self.store.simhashes_in_tier("procedural"))

        # I5 Step 6: Per-cycle adapter-weight shadows (debug/analysis only — no
        # manifest).  Layout owned by DebugSnapshotWriter:
        #   paths.debug/.../training/tiers/<tier>/adapter_weights/
        # Writer self-gates on save_cycle_snapshots; callers do not check.
        tier_shadow = ["episodic"]
        if "semantic" in self.model.peft_config:
            tier_shadow.append("semantic")
        if "procedural" in self.model.peft_config:
            tier_shadow.append("procedural")
        self._debug_writer.on_main_adapters_saved(tier_shadow)

        # I5 Steps 6–8: SimHash registries (per-tier), indexed_key_registry
        # (per-tier), then the registry commit signal.
        if self.store.replay_enabled and tier_payloads:
            for _tier in ("episodic", "semantic", "procedural"):
                save_registry(
                    self.store.simhashes_in_tier(_tier),
                    self.output_dir / _tier / "simhash_registry.json",
                )
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
        recall_sanity_threshold: float = 0.95,
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
        queue is consumed by the next consolidation cycle (Step 7).  What is not
        trained does not exist on disk — this upholds the privacy invariant.

        Procedural relations extracted from the same transcript are trained onto
        the stable ``procedural`` main adapter (no interim tier for procedural —
        preferences are small-volume and slow-changing).  This pass runs inline,
        immediately after the episodic training pass and before any registry
        writes, so a failure in either pass leaves the registry clean.

        Registry write ordering (I5 atomicity §2.5):
        1. ``save_bytes()`` — serialise registry to bytes without writing.
        2. ``sha256(payload)`` — hash for manifest pre-stamp.
        3. Build manifest with ``registry_sha256_override``.
        4. ``save_adapter(..., manifest=manifest)`` — adapter + manifest on disk.
        5. Save SimHash registries to per-tier paths:
           ``<adapter_dir>/<tier>/simhash_registry.json``.
        6. ``save_from_bytes(payload, path)`` — flush registry bytes to
           ``<adapter_dir>/<tier>/indexed_key_registry.json`` (LAST).

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

        Returns:
            Train loss as ``float`` if training ran, otherwise ``None``.
        """
        if not rels:
            return None

        # Local import alias so tests can patch paramem.training.trainer.train_adapter.
        from paramem.training.trainer import train_adapter as _train_adapter

        if mode == "train":
            switch_adapter(self.model, "procedural")
        new_proc_keyed, existing_proc_keyed, keys_to_retire, new_sp_mappings = (
            self._prepare_procedural_keys_for_tier(rels, speaker_id, mode=mode)
        )
        all_procedural = existing_proc_keyed + new_proc_keyed
        if not (mode == "train" and all_procedural):
            return None
        examples = format_entry_training(all_procedural, self.tokenizer, max_length=1024)
        dataset = self._indexed_dataset(examples)
        training_config = self._make_training_config(num_epochs=self.training_config.num_epochs)
        self._enable_gradient_checkpointing()
        recall_cb = self._maybe_make_recall_callback(
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
            hooks=TrainingHooks(on_shutdown_check=lambda: self.shutdown_requested),
            callbacks_extra=[recall_cb] if recall_cb is not None else None,
        )
        # Deferred mutations — apply only after _train_adapter succeeds.
        self._procedural_next_index = self._procedural_tentative_next_index
        for old_key in keys_to_retire:
            self.store.delete(old_key)
        for kp in new_proc_keyed:
            fingerprint = compute_simhash(kp["key"], kp["subject"], kp["predicate"], kp["object"])
            self.store.put("procedural", kp["key"], kp, simhash=fingerprint)
        for sp_key, new_key in new_sp_mappings.items():
            self.procedural_sp_index[sp_key] = new_key
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
        recall_sanity_threshold: float = 0.95,
        new_promotions: "list[str] | None" = None,
    ) -> dict:
        """Unified interim-cycle entry: key prep + optional training + I5 persistence.

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
        5. ``_maybe_run_interim_enrichment()`` at sub-interval rollover.
        6. Compute stamp (when not provided) and call ``_resolve_target_slot``
           to obtain ``(adapter_name, cap_reached_absorb, queue_only,
           degenerated_skip)``.
        7. Handle control-flow branches: queue-only and degenerated-skip.
        8. Mint PEFT slot (train only); run ``_maybe_run_interim_enrichment``
           (simulate skips enrichment, train runs it at rollover).
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
            recall_sanity_threshold: Forwarded to ``commit_tier_slot`` for the
                post-save disk-integrity probe threshold (train mode only).
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

        # --- 5. Interim enrichment at sub-interval rollover ---
        # Runs AFTER queue/degenerate checks so it doesn't fire for queued
        # sessions.  Only meaningful when a fresh slot is being minted.
        # Enrichment is mode-agnostic — it mutates the cumulative graph used by
        # both train (key assignment from graph triples) and simulate (registry
        # rebuild from the same graph).  The "is_fresh_slot" guard preserves the
        # original rollover-only cadence; mode is irrelevant.
        is_fresh_slot = (
            mode == "simulate"  # simulate has no peft_config; every cycle is "fresh"
            or (adapter_name not in self.model.peft_config and not cap_reached_absorb)
        )
        if is_fresh_slot:
            self._maybe_run_interim_enrichment()

        # --- End-of-extraction debug dump (cumulative graph + relations) ---
        # Fires after enrichment mutates the cumulative graph; the writer
        # self-gates on save_cycle_snapshots so the call is unconditional.
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

        # --- 9. Prepare episodic keys ---
        if mode == "train":
            switch_adapter(self.model, adapter_name)
        all_interim_keyed = self._prepare_episodic_keys_for_tier(
            adapter_name, episodic_rels, speaker_id, mode=mode
        )
        # Identify the new entries returned by _prepare_episodic_keys_for_tier
        # (marked with "_new": True).  store.put and _indexed_next_index advancement
        # are deferred: applied AFTER _train_adapter returns (train mode) or
        # immediately (simulate mode — no training step can fail).
        new_keyed_episodic = [kp for kp in all_interim_keyed if kp.get("_new")]
        new_key_ids = [kp["key"] for kp in new_keyed_episodic]

        if mode == "simulate":
            # No training step — apply store mutations immediately.
            for kp in new_keyed_episodic:
                self.store.put(adapter_name, kp["key"], kp)
            self._indexed_next_index += len(new_keyed_episodic)

        # --- 9b. Semantic promotion transfer (train mode, when promotions supplied) ---
        # Move keys for promoted entities from episodic to the semantic tier so
        # they are excluded from the episodic training set and picked up by
        # _run_indexed_key_semantic in the caller (run_cycle).  This mirrors the
        # logic that formerly lived inside _run_indexed_key_episodic.
        promoted_key_set: set[str] = set()
        if mode == "train" and new_promotions:
            promoted_set = {n.lower() for n in new_promotions}
            for _tier, key, qa_info in list(self.store.iter_entries()):
                if key.startswith("proc"):
                    continue
                subject = qa_info.get("subject", "").lower()
                obj = qa_info.get("object", "").lower()
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
        # Local import alias so tests can patch paramem.training.trainer.train_adapter
        # (matching the pattern of the former _train_extracted_into_interim).
        from paramem.training.trainer import train_adapter as _train_adapter

        epi_train_loss: float | None = None
        if mode == "train" and all_interim_keyed:
            examples = format_entry_training(all_interim_keyed, self.tokenizer, max_length=1024)
            dataset = self._indexed_dataset(examples)
            training_config = self._make_training_config(num_epochs=self.training_config.num_epochs)
            self._enable_gradient_checkpointing()
            recall_cb = self._maybe_make_recall_callback(
                entries=all_interim_keyed,
                adapter_name=adapter_name,
                output_dir=self._training_output_dir(adapter_name, interim_stamp=stamp),
                phase_name=f"interim-{adapter_name}-{run_label}",
            )
            epi_metrics = _train_adapter(
                model=self.model,
                tokenizer=self.tokenizer,
                train_dataset=dataset,
                adapter_name=adapter_name,
                training_config=training_config,
                adapter_config=self.episodic_config,
                wandb_config=self.wandb_config,
                output_dir=self._training_output_dir(adapter_name, interim_stamp=stamp),
                run_name=f"interim-{adapter_name}-{run_label}",
                thermal_policy=self._thermal_policy,
                hooks=TrainingHooks(on_shutdown_check=lambda: self.shutdown_requested),
                callbacks_extra=[recall_cb] if recall_cb is not None else None,
            )
            epi_train_loss = epi_metrics.get("train_loss") if epi_metrics is not None else None
            # Episodic store mutations are deferred until AFTER the procedural
            # step completes — see the mutation block below the procedural gate.
            # This guarantees full-cycle atomicity: if any training step fails,
            # the registry stays clean.

        # Update episodic-interim simhash registry from ground-truth pairs.
        self.store.replace_simhashes_in_tier(adapter_name, build_registry(all_interim_keyed))

        # --- 11. Procedural (always onto stable "procedural" main adapter) ---
        proc_train_loss: float | None = None
        if self.procedural_config is not None and procedural_rels:
            proc_train_loss = self._run_indexed_key_procedural(
                procedural_rels,
                speaker_id,
                mode=mode,
                stamp=stamp,
                run_label=run_label,
            )

        # --- 11b. Apply deferred episodic store mutations ---
        # Reached only when all training steps above completed without raising.
        # In simulate mode this block was already handled before training.
        if mode == "train":
            for kp in new_keyed_episodic:
                self.store.put(adapter_name, kp["key"], kp)
            self._indexed_next_index += len(new_keyed_episodic)

        # --- 12. Persist both tiers ---
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
        ``source="graph_enrichment"``.

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

        for chunk_nodes in chunks:
            try:
                chunk_subgraph = graph.subgraph(chunk_nodes)
                triples = _serialize_subgraph_triples(chunk_subgraph)
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
                if keep not in graph or drop not in graph:
                    logger.debug(
                        "graph_enrichment: same_as skip — keep=%r drop=%r not both in graph",
                        keep,
                        drop,
                    )
                    continue
                merge_key = frozenset({keep.lower(), drop.lower()})
                if merge_key in seen_merge_keys:
                    logger.debug(
                        "graph_enrichment: same_as dedup — %r / %r already seen",
                        keep,
                        drop,
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
                try:
                    nx.contracted_nodes(graph, keep, drop, self_loops=False, copy=False)
                    total_merges += 1
                    coref_map[drop] = keep
                    logger.debug("graph_enrichment: contracted %r → %r", drop, keep)
                except Exception as exc:
                    logger.warning(
                        "graph_enrichment: same_as contraction failed %r → %r: %s",
                        drop,
                        keep,
                        exc,
                    )

            def _resolve_name(name: str) -> str:
                # Follow drop→keep chains with a visited guard.
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
                pred = _normalize_predicate(raw_pred)
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
                    source="graph_enrichment",
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

    def consolidate_interim_graphs(self) -> dict:
        """Full-cycle consolidation for simulate mode: merge interim graph.json sidecars.

        Called by ``_run_full_consolidation_sync`` when
        ``config.consolidation.mode == "simulate"``.  Simulate mode writes per-cycle
        graph.json files under ``<adapter_dir>/episodic/interim_<stamp>/graph.json``
        rather than PEFT adapter weights.  This method merges those interim graphs into
        the canonical main-tier graph at ``<adapter_dir>/episodic/graph.json`` and
        removes the interim slot directories.

        Steps:

        1. Optional SOTA enrichment (same guard as ``consolidate_interim_adapters`` step 0).
        2. Walk all ``episodic/interim_<stamp>/graph.json`` files; load and merge each
           into a combined graph, tagging edges from each slot with the slot's keys.
        3. Write the merged graph to ``<adapter_dir>/episodic/graph.json`` via
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

        # --- Step 0: optional SOTA enrichment ---
        try:
            self._run_graph_enrichment()
        except Exception as _enr_exc:
            logger.warning(
                "consolidate_interim_graphs: SOTA enrichment raised — continuing: %s",
                _enr_exc,
            )

        # --- Step 2: walk interim simulate-mode graph.json sidecars ---
        merged_keys: set[str] = set()
        interim_dirs_merged: list[Path] = []

        episodic_main_path = adapter_dir / "episodic" / "graph.json"
        # Start from the existing main graph if present.
        merged_graph = load_memory_from_disk(episodic_main_path)

        for _interim_name, interim_dir in iter_interim_dirs(adapter_dir):
            interim_graph_path = interim_dir / "graph.json"
            if not interim_graph_path.exists():
                # Skip train-mode interim slots (no graph.json).
                continue
            slot_graph = load_memory_from_disk(interim_graph_path)
            for entry in iter_entries(slot_graph):
                merged_keys.add(entry["key"])
                # Add edge to merged graph (add_edge is idempotent on
                # duplicate ik_key via the MultiDiGraph semantics — we
                # re-add with the same attributes so the latest write wins).
                from paramem.memory.persistence import _add_keyed_edge

                _add_keyed_edge(
                    merged_graph,
                    entry["subject"],
                    entry["object"],
                    indexed_key=entry["key"],
                    predicate=entry.get("predicate", ""),
                    speaker_id=entry.get("speaker_id", ""),
                    first_seen_cycle=entry.get("first_seen_cycle", 0),
                )
            interim_dirs_merged.append(interim_dir)
            logger.debug(
                "consolidate_interim_graphs: merged %d edges from %s",
                slot_graph.number_of_edges(),
                interim_dir,
            )

        if not interim_dirs_merged:
            logger.info("consolidate_interim_graphs: no simulate-mode interim slots found — noop")
            return {
                "tiers_rebuilt": [],
                "graph_drift_count": 0,
                "keys_per_tier": {},
                "recall_per_tier": {},
                "rolled_back": False,
                "rollback_tier": None,
            }

        # --- Step 3: write merged graph to main episodic slot ---
        episodic_main_path.parent.mkdir(parents=True, exist_ok=True)
        save_memory_to_disk(merged_graph, episodic_main_path)
        logger.info(
            "consolidate_interim_graphs: wrote merged graph to %s (%d edges, %d new keys)",
            episodic_main_path,
            merged_graph.number_of_edges(),
            len(merged_keys),
        )

        # --- Step 4: remove merged interim slot directories ---
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

        return {
            "tiers_rebuilt": ["episodic"],
            "graph_drift_count": 0,
            "keys_per_tier": {"episodic": merged_graph.number_of_edges()},
            "recall_per_tier": {"episodic": 1.0},  # graph merge is lossless
            "rolled_back": False,
            "rollback_tier": None,
        }

    def consolidate_interim_adapters(
        self,
        trainer=None,
        router=None,
        recall_sanity_threshold: float = 0.95,
        refresh_epochs: int = 30,
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
          section; the inference-yield hook installed by ``train_adapter`` only
          fires at epoch/step boundaries and would not protect non-training
          code paths.

        Steps:
        1. Verify the GPU lock is held (entry guard — leak-safe pattern).
        2. Walk all active keys, re-derive their tier from the cumulative graph,
           handle graph-drift keys, and build per-tier keyed-pair lists.
        3. Load backup adapters (if available) and all interim adapters into PEFT.
        4. For each tier (episodic → semantic → procedural):
           a. Set active adapter to <tier>_backup before deleting the main.
           b. delete_adapter(<tier>) + create_adapter(<tier>).
           c. Set <tier>_is_training=True, call _train_adapter, set False.
           d. Recall-sanity check; on failure roll back and abort.
        5. Atomic finalize: registry rewrite → unload_interim_adapters → router reload.
        6. On success: unload backup adapters.

        Args:
            trainer: BackgroundTrainer instance (must be the one holding the GPU
                lock via submit()).  Required for per-tier B2 re-arm pattern.
            router: Router instance whose reload() is called at the end of the
                atomic finalize sequence.  Optional — skipped when None.
            recall_sanity_threshold: Minimum recall rate for the post-save
                disk-integrity probe in ``_save_adapters`` (forwarded verbatim).
                In-RAM pre-save probes have been removed; the gate fires
                post-save via ``_verify_saved_adapter_from_disk``.
            refresh_epochs: Number of training epochs for the full per-tier
                rebuild (default 30; matches the per-key baseline from Tests 1-7b).

        Returns:
            Result dict with keys:
                {
                    "tiers_rebuilt": list[str],
                    "graph_drift_count": int,
                    "keys_per_tier": dict[str, int],
                    "recall_per_tier": dict[str, float],
                    "rolled_back": bool,
                    "rollback_tier": str | None,
                }
        """
        from paramem.memory.interim_adapter import unload_interim_adapters
        from paramem.models.loader import create_adapter
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

        # --- Graph-level SOTA enrichment (Task #10) ---
        # Runs after all per-session merges are complete and before
        # partition_relations so enriched edges flow into all tiers identically.
        # Mutates self.merger.graph in place.  Wrapped in try/except so a SOTA
        # failure cannot abort consolidation.
        try:
            enrichment_result = self._run_graph_enrichment()
            if not enrichment_result.get("skipped"):
                logger.info(
                    "graph_enrichment complete: chunks=%d new_edges=%d same_as_merges=%d",
                    enrichment_result.get("chunks", 0),
                    enrichment_result.get("new_edges", 0),
                    enrichment_result.get("same_as_merges", 0),
                )
        except Exception as _enrich_exc:
            logger.warning(
                "graph_enrichment raised unexpected exception — consolidation continues: %s",
                _enrich_exc,
            )

        # --- Step 2: Snapshot cumulative graph ---
        graph = self.merger.graph

        # --- Step 3: Re-derive per-tier keyed-pair lists ---
        # Build a lookup from (subject, predicate, object) → relation_type in
        # the cumulative graph so we can re-derive the tier for each key.
        graph_triple_to_type: dict[tuple[str, str, str], str] = {}
        for rel in getattr(graph, "relations", []) if hasattr(graph, "relations") else []:
            triple = (
                getattr(rel, "subject", ""),
                getattr(rel, "predicate", ""),
                getattr(rel, "object", ""),
            )
            graph_triple_to_type[triple] = getattr(rel, "relation_type", "factual")

        active_keys = self._all_active_keys()

        tier_keyed: dict[str, list[dict]] = {
            "episodic": [],
            "semantic": [],
            "procedural": [],
        }
        graph_drift_count = 0

        for key in active_keys:
            qa_info = self.store.get(key)
            if qa_info is None:
                # No QA metadata — skip (should not happen with intact registry)
                logger.warning("consolidate_interim_adapters: no QA metadata for key %s", key)
                continue

            src_subj = qa_info.get("subject", "")
            src_pred = qa_info.get("predicate", "")
            src_obj = qa_info.get("object", "")
            triple_key = (src_subj, src_pred, src_obj)

            # current_adapter_id = the tier dict key that owns this key
            current_adapter_id = self.store.tier_for_active_key(key) or "episodic"

            # Try to re-derive tier from the cumulative graph.
            relation_type = graph_triple_to_type.get(triple_key)
            if relation_type is not None:
                # Re-derive tier using the same partition_relations policy.
                dummy_rel = [
                    {
                        "subject": src_subj,
                        "predicate": src_pred,
                        "object": src_obj,
                        "relation_type": relation_type,
                    }
                ]
                ep_rels, proc_rels = partition_relations(
                    dummy_rel, procedural_enabled=self.procedural_config is not None
                )
                if proc_rels:
                    tier = "procedural"
                elif ep_rels:
                    # Determine if this is episodic or semantic based on the
                    # current adapter_id (semantic keys stay semantic).
                    if current_adapter_id == "semantic":
                        tier = "semantic"
                    else:
                        tier = "episodic"
                else:
                    tier = "episodic"
            else:
                # Graph drift: triple not found in cumulative graph.
                graph_drift_count += 1
                logger.info(
                    "graph_drift_key key=%s subject=%r predicate=%r "
                    "object=%r current_adapter_id=%r",
                    key,
                    src_subj,
                    src_pred,
                    src_obj,
                    current_adapter_id,
                )
                # Apply the interim-adapter bucketing rule:
                #   episodic_interim_* → episodic
                #   non-interim adapter_id → must be one of the three main tiers
                if current_adapter_id.startswith(
                    ("episodic_interim_", "semantic_interim_", "procedural_interim_")
                ):
                    tier = current_adapter_id.split("_interim_")[0]
                else:
                    assert current_adapter_id in {
                        "episodic",
                        "semantic",
                        "procedural",
                    }, (
                        f"Registry invariant violation: key {key!r} has "
                        f"unexpected tier {current_adapter_id!r}"
                    )
                    tier = (
                        current_adapter_id
                        if current_adapter_id in {"episodic", "semantic", "procedural"}
                        else "episodic"
                    )

            tier_keyed[tier].append(
                {
                    "key": key,
                    "subject": qa_info["subject"],
                    "predicate": qa_info["predicate"],
                    "object": qa_info["object"],
                }
            )

        # Warn if graph drift exceeds 10 % of active keys.
        if active_keys and graph_drift_count > len(active_keys) * 0.10:
            logger.warning(
                "consolidate_interim_adapters: high graph drift (%d / %d keys = %.0f%%) — "
                "indicates merge regression or drifted corpus; review recommended",
                graph_drift_count,
                len(active_keys),
                100.0 * graph_drift_count / len(active_keys),
            )

        logger.info(
            "consolidate_interim_adapters: key distribution — episodic=%d semantic=%d "
            "procedural=%d drift=%d",
            len(tier_keyed["episodic"]),
            len(tier_keyed["semantic"]),
            len(tier_keyed["procedural"]),
            graph_drift_count,
        )

        # --- Step 4: Build per-tier TrainingJob objects ---
        # Must be constructed before any tier rebuild begins so that
        # inference_fallback_adapter is set to the backup adapter name.
        # Because the public train_adapter (not BackgroundTrainer._train_adapter)
        # is called here, _current_job is NOT updated automatically.  The
        # per-tier loop manually swaps trainer._current_job to the tier-specific
        # job before calling train_adapter so that pause() reads the correct
        # inference_fallback_adapter for each tier rebuild.
        from paramem.server.background_trainer import TrainingJob

        refresh_training_config = self._make_training_config(num_epochs=refresh_epochs)

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

        # --- Step 5: Per-tier fresh-adapter rebuild ---
        # Load backup adapters (PEFT load-or-skip idempotency: NC-2).
        # In the absence of a real encrypted backup dir we skip the load
        # and fall back to the live main adapters as the de-facto backup.
        # Production systems should implement the encrypted backup path (Step 7d).
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

        for tier in ("episodic", "semantic", "procedural"):
            backup_name = f"{tier}_backup"
            job = jobs_by_tier[tier]

            if not job.entries:
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

            # d. B2 re-arm: flip _is_training True BEFORE _train_adapter,
            #    False in finally so it fires on success AND failure.
            #    Also swap _current_job to the per-tier job so that pause()
            #    reads the correct inference_fallback_adapter (e.g.
            #    "episodic_backup") during this tier's rebuild.  The prior
            #    sentinel (installed by _run_callable_queue) is restored in
            #    the finally block to preserve the outer invariant.
            prior_job = None
            if trainer is not None:
                prior_job = trainer._current_job
                trainer._current_job = job
                trainer._set_is_training(True)
            try:
                # Format training data and update job config for refresh epochs.
                # Tier_keyed entries are {key,subject,predicate,object}.
                from paramem.training.trainer import train_adapter as _train_adapter_fn

                examples = format_entry_training(job.entries, self.tokenizer, max_length=1024)
                if examples:
                    dataset = self._indexed_dataset(examples)
                    self._enable_gradient_checkpointing()
                    # Fresh recall callback per tier — _signaled_stop and
                    # epoch_log do not leak across tiers.
                    recall_cb = self._maybe_make_recall_callback(
                        entries=job.entries,
                        adapter_name=tier,
                        output_dir=self.output_dir / "consolidation_refresh" / tier,
                        phase_name=f"consolidate-{tier}",
                    )
                    _train_adapter_fn(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        train_dataset=dataset,
                        adapter_name=tier,
                        training_config=refresh_training_config,
                        adapter_config=tier_cfg,
                        wandb_config=self.wandb_config,
                        output_dir=self.output_dir / "consolidation_refresh" / tier,
                        run_name=f"consolidate-{tier}",
                        thermal_policy=self._thermal_policy,
                        hooks=TrainingHooks(on_shutdown_check=lambda: self.shutdown_requested),
                        callbacks_extra=[recall_cb] if recall_cb is not None else None,
                    )
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
            tiers_rebuilt.append(tier)

        # flip off _is_training at finalize entry (belt-and-braces).
        if trainer is not None:
            trainer._set_is_training(False)

        # --- Step 6: Atomic finalize ---
        # Invariant: registry rewrite FIRST, Router.reload() LAST.
        # The finalize block runs purely as registry / disk / PEFT / router ops.
        # No _train_adapter() call, no HF Trainer-driven routine may appear here.

        # 6a. Registry rewrite (MUST be first).
        # Rebuild per-tier registries from the authoritative tier_keyed layout:
        # each tier gets a fresh KeyRegistry containing only the keys that belong
        # to it after re-derivation from the cumulative graph.  Interim-tier
        # registries are then dropped (they are superseded by the main tiers).
        if self.store.replay_enabled:
            # Reset each main tier's registry to the post-consolidation membership.
            for _main_tier in ("episodic", "semantic", "procedural"):
                new_reg = KeyRegistry()
                for kp in tier_keyed.get(_main_tier, []):
                    new_reg.add(kp["key"])
                self.store.load_registry(_main_tier, new_reg)
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

        # 6b. Interim purge — single call covers both PEFT delete AND on-disk rmtree.
        unload_interim_adapters(self.model, self.output_dir)
        logger.info("consolidate_interim_adapters: interim adapters unloaded")

        # 6c. Router reload (MUST be last).
        if router is not None:
            try:
                router.reload()
                logger.info("consolidate_interim_adapters: router reloaded")
            except Exception:
                logger.exception("consolidate_interim_adapters: router reload failed")

        # --- Step 7: Success commit ---
        # Unload backup adapters (the three mains remain loaded throughout).
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

        logger.info(
            "consolidate_interim_adapters: complete — rebuilt %s, drift=%d",
            tiers_rebuilt,
            graph_drift_count,
        )

        return {
            "tiers_rebuilt": tiers_rebuilt,
            "graph_drift_count": graph_drift_count,
            "keys_per_tier": {t: len(v) for t, v in tier_keyed.items()},
            "recall_per_tier": recall_per_tier,
            "rolled_back": False,
            "rollback_tier": None,
        }

    def _run_recall_sanity_probe(
        self,
        adapter_name: str,
        entries: list[dict],
        *,
        max_probe: int = 100,
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
            return float(recall_result["rate"])
        except Exception:
            logger.exception(
                "_run_recall_sanity_probe: recall probe failed for adapter %s — "
                "returning 0.0 so caller trips the sanity gate",
                adapter_name,
            )
            return 0.0

    def _verify_saved_adapter_from_disk(
        self,
        adapter_name: str,
        slot_path: Path,
        entries: list[dict],
        *,
        threshold: float = 0.95,
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
            threshold: Minimum recall the disk artifact must achieve.  Defaults
                to ``0.95`` matching the production recall gates.
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
        """Create adapters that don't exist yet.

        Production adapters (episodic, semantic, procedural) are created
        based on configuration. A reusable `in_training` staging adapter is
        also created — background training uses it so that inference during
        pauses falls back to the last-known-good production adapter weights.
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

        # Staging adapter for on-the-fly training. Templated on episodic_config
        # at startup; the BackgroundTrainer rebuilds the slot with the current
        # job's tier config when shapes diverge (e.g. switching between an
        # attn-only tier like episodic/semantic and procedural's attn+mlp).
        # No uniform-shape constraint across production tiers — per-tier shape
        # differences (procedural's MLP targeting) are the design.
        if "in_training" not in self.model.peft_config:
            logger.info("Creating in_training staging adapter")
            self.model = create_adapter(self.model, self.episodic_config, "in_training")

        # Clean stale staging checkpoints on disk unless a valid resume state
        # is present — in that case the directory holds a live checkpoint that
        # the BackgroundTrainer will use on the next job submission.
        stale_dir = Path(self.output_dir) / "in_training"
        if stale_dir.exists():
            from paramem.server.background_trainer import _RESUME_STATE_FILE, _read_resume_state

            resume_state_path = stale_dir / _RESUME_STATE_FILE
            resume_state = _read_resume_state(resume_state_path)
            has_live_checkpoint = (
                resume_state is not None
                and bool(resume_state.get("checkpoint_path", ""))
                and Path(resume_state["checkpoint_path"]).is_dir()
            )
            if has_live_checkpoint:
                logger.info(
                    "Preserving in_training state for resume: adapter=%s epoch=%d/%d",
                    resume_state.get("adapter_name", "?"),
                    resume_state.get("last_completed_epoch", 0),
                    resume_state.get("total_epochs", 0),
                )
            else:
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
    ):
        """Construct a RecallEarlyStopCallback when configured.

        Returns None when ``training_config.recall_early_stopping`` is
        False or when the entries list is empty (probing an empty set is
        a no-op).

        The probe target is the unmodified entries list — the same per-tier
        full-replay set that ``format_entry_training`` consumes.  This
        is the convergence gate — only safe if the caller passes the FULL
        active-key set for ``adapter_name``, not an incremental delta.

        Production-reachable callers (must pass full per-tier active set):
          - ConsolidationLoop.run_cycle (episodic/procedural inline blocks)
          - ConsolidationLoop.train_adapters (episodic/procedural inline blocks)
          - ConsolidationLoop.run_consolidation_cycle (episodic/procedural blocks)
          - ConsolidationLoop.consolidate_interim_adapters
          - active_store_migration._migrate_tier_simulate_to_train
            (paramem/server/active_store_migration.py:420)

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
        """
        if not self.training_config.recall_early_stopping:
            return None
        if not entries:
            return None
        from pathlib import Path

        from paramem.training.early_stop import (
            EarlyStopPolicy,
            RecallEarlyStopCallback,
            _EarlyStopState,
        )

        output_dir = Path(output_dir)
        # probe_from_epoch is pinned to the signal floor: a single probe runs
        # 137-ish ``generate(max_new_tokens=128)`` calls (paramem/training/
        # entry_memory.py::probe_entry), which is ~12-40× the per-epoch
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

        return RecallEarlyStopCallback(
            model=self.model,
            tokenizer=self.tokenizer,
            target_keyed=entries,
            target_registry=_build_registry(entries),
            adapter_name=adapter_name,
            policy=policy,
            state_out=_EarlyStopState(),
            progress_path=output_dir / "progress.json",
            epoch_log_path=output_dir / "epoch_log.json",
            first_perfect_log_path=None,  # production has no per-key log
            phase_name=phase_name,
            num_epochs=self.training_config.num_epochs,
            pause_file=None,  # production pause via gpu_lock_sync, not file
            eval_fn=_eval_fn,
        )


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
    {
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


def _serialize_subgraph_triples(subgraph) -> list[dict]:
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
