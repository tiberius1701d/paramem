"""Consolidation loop orchestrator.

Runs the full consolidation pipeline: extract graph from session,
merge into cumulative graph, score for promotion/decay, train
episodic and semantic adapters.
"""

import logging
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from torch.utils.data import Dataset

from paramem.graph.extractor import (
    PROVIDER_KEY_ENV,
    _graph_enrich_with_sota,
    extract_graph,
    extract_procedural_graph,
)
from paramem.graph.merger import GraphMerger, _normalize_predicate
from paramem.graph.qa_generator import (
    generate_qa_from_relations,
    partition_relations,
)
from paramem.graph.scoring import (
    PromotionScorer,
    get_relations_for_nodes,
)
from paramem.models.loader import save_adapter, switch_adapter
from paramem.training.curriculum import CurriculumSampler
from paramem.training.indexed_memory import (
    assign_keys,
    build_registry,
    compute_simhash,
    format_indexed_training,
    load_registry,
    probe_key,
    save_registry,
)
from paramem.training.key_registry import ADAPTER_HEALTH_DEGENERATED, KeyRegistry
from paramem.training.replay import MixedReplayDataset, SyntheticQADataset
from paramem.training.trainer import GracefulShutdownCallback, train_adapter
from paramem.utils.config import (
    AdapterConfig,
    ConsolidationConfig,
    GraphConfig,
    TrainingConfig,
    WandbConfig,
)

logger = logging.getLogger(__name__)


def _validate_staging_compat(*adapter_configs) -> None:
    """Ensure all production adapter configs are weight-shape compatible with staging.

    The `in_training` staging slot uses episodic_config as template. The
    fingerprint captures ONLY the fields that determine LoRA tensor shapes:
    rank (A/B matrix inner dim), target_modules (which layers get adapted),
    and bias setting (whether bias tensors exist). alpha and dropout affect
    training behavior but not tensor shapes, so they may differ across
    adapters. A fingerprint mismatch means copy_adapter_weights would fail
    at the parameter-set-equality check — we fail faster here.
    """
    reference = None
    for cfg in adapter_configs:
        if cfg is None:
            continue
        fingerprint = (
            getattr(cfg, "rank", None),
            tuple(sorted(getattr(cfg, "target_modules", []) or [])),
            getattr(cfg, "bias", None),
        )
        if reference is None:
            reference = fingerprint
        elif fingerprint != reference:
            raise ValueError(
                f"Adapter configs incompatible for staging: {reference} vs {fingerprint}. "
                "All production adapters must share rank, target_modules, and bias."
            )


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
        procedural_adapter_config: Optional[AdapterConfig] = None,
        wandb_config: Optional[WandbConfig] = None,
        output_dir: str | Path = "outputs/phase3",
        graph_path: Optional[str | Path] = None,
        extraction_temperature: float = 0.0,
        extraction_max_tokens: int = 2048,
        save_cycle_snapshots: bool = True,
        snapshot_dir: str | Path | None = None,
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
        graph_config: Optional[GraphConfig] = None,
        graph_enrichment_enabled: bool = True,
        graph_enrichment_neighborhood_hops: int = 2,
        graph_enrichment_max_entities_per_pass: int = 50,
        graph_enrichment_interim_enabled: bool = True,
        graph_enrichment_min_triples_floor: int = 20,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = consolidation_config
        self.training_config = training_config
        self.shutdown_requested = False  # set by signal handler to stop training
        self._shutdown_callbacks = [GracefulShutdownCallback(lambda: self.shutdown_requested)]
        self.episodic_config = episodic_adapter_config
        self.semantic_config = semantic_adapter_config
        self.procedural_config = procedural_adapter_config
        self.wandb_config = wandb_config
        self.save_cycle_snapshots = save_cycle_snapshots
        self.snapshot_dir = Path(snapshot_dir) if snapshot_dir else None
        self.persist_graph = persist_graph
        self.prompts_dir = prompts_dir
        # Entity-level promotion requires a persistent graph for cross-restart
        # recurrence tracking. When the graph is transient (persist_graph=False),
        # the caller must handle promotion externally (e.g. key-level promotion).
        self.enable_entity_promotion = persist_graph
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.extraction_temperature = extraction_temperature
        self.extraction_max_tokens = extraction_max_tokens

        # Extraction pipeline flags — the SOTA enrichment chain (anonymize →
        # noise-filter → plausibility → de-anonymize → NER). Loop-level defaults
        # propagate to every extraction; callers may override per call.
        self.extraction_stt_correction = extraction_stt_correction
        self.extraction_ha_validation = extraction_ha_validation
        self.extraction_noise_filter = extraction_noise_filter
        self.extraction_noise_filter_model = extraction_noise_filter_model
        self.extraction_noise_filter_endpoint = extraction_noise_filter_endpoint
        self.extraction_ner_check = extraction_ner_check
        self.extraction_ner_model = extraction_ner_model
        self.extraction_plausibility_judge = extraction_plausibility_judge
        self.extraction_plausibility_stage = extraction_plausibility_stage
        self.extraction_verify_anonymization = extraction_verify_anonymization

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

        # Replay pools: track QA pairs available for replay per adapter
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

        # Indexed key replay (F4.9c validated)
        self.indexed_key_registry: Optional[KeyRegistry] = None
        # key -> {"key": str, "question": str, "answer": str, "source_subject": str}
        self.indexed_key_qa: dict[str, dict] = {}
        # Per-adapter SimHash registries: key -> 64-bit fingerprint
        self.episodic_simhash: dict[str, int] = {}
        self.semantic_simhash: dict[str, int] = {}
        self.procedural_simhash: dict[str, int] = {}
        # (speaker_id, subject, predicate) -> key for contradiction detection
        self.procedural_sp_index: dict[tuple[str, str, str], str] = {}
        self._indexed_next_index: int = 1
        self._procedural_next_index: int = 1
        if self.config.indexed_key_replay_enabled:
            registry_path = self.output_dir / "indexed_key_registry.json"
            self.indexed_key_registry = KeyRegistry.load(registry_path)
            # Derive next index from existing keys
            for key in self.indexed_key_registry.list_active():
                if key.startswith("graph"):
                    try:
                        idx = int(key[5:])
                        self._indexed_next_index = max(self._indexed_next_index, idx + 1)
                    except ValueError:
                        pass
                elif key.startswith("proc"):
                    try:
                        idx = int(key[4:])
                        self._procedural_next_index = max(self._procedural_next_index, idx + 1)
                    except ValueError:
                        pass
            # Load persisted SimHash registries
            ep_simhash_path = self.output_dir / "simhash_registry_episodic.json"
            sem_simhash_path = self.output_dir / "simhash_registry_semantic.json"
            proc_simhash_path = self.output_dir / "simhash_registry_procedural.json"
            if ep_simhash_path.exists():
                self.episodic_simhash = load_registry(ep_simhash_path)
            if sem_simhash_path.exists():
                self.semantic_simhash = load_registry(sem_simhash_path)
            if proc_simhash_path.exists():
                self.procedural_simhash = load_registry(proc_simhash_path)

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

        # --- Multi-adapter interim routing state (Step 7) ---
        # Set of triples already encoded into an interim adapter.  Reset at the
        # start of every consolidate_interim_adapters() call — the full rebuild
        # makes prior "seen" state irrelevant (Step 7e).  Not persisted to disk;
        # restart cost is at most one window of duplicate QA generation.
        self.seen_triples: set[tuple[str, str, str]] = set()

        # Counter of triples encoded since the last successful full
        # consolidation.  Incremented after each successful interim training
        # pass (post_session_train "trained" path).  Reset only on successful
        # consolidate_interim_adapters() completion.  Decoupled from
        # seen_triples — it is NOT reset when seen_triples resets.
        self.triples_since_last_full: int = 0

    def seed_key_metadata(self, metadata: dict) -> None:
        """Restore key-level metadata from persisted key_metadata.json.

        Used when persist_graph=False to restore cycle count, promoted
        keys, and per-key session counts across server restarts.
        """
        self.cycle_count = metadata.get("cycle_count", 0)
        self.promoted_keys = set(metadata.get("promoted_keys", []))
        for key, key_meta in metadata.get("keys", {}).items():
            self.key_sessions[key] = key_meta.get("sessions_seen", 0)
        logger.info(
            "Seeded key metadata: cycle=%d, %d promoted, %d keys tracked",
            self.cycle_count,
            len(self.promoted_keys),
            len(self.key_sessions),
        )

    def seed_procedural_qa(self, keyed_pairs: list[dict]) -> None:
        """Rebuild the contradiction index from persisted procedural keyed_pairs.

        Called at startup. QA pairs go into indexed_key_qa (shared store).
        The sp_index is rebuilt for contradiction detection.
        """
        for kp in keyed_pairs:
            key = kp["key"]
            self.indexed_key_qa[key] = kp
            if self.indexed_key_registry and key not in self.indexed_key_registry.list_active():
                self.indexed_key_registry.add(key)
            subject = kp.get("source_subject", "").lower()
            predicate = kp.get("source_predicate", "").lower()
            speaker = kp.get("speaker_id", "")
            if subject and predicate:
                self.procedural_sp_index[(speaker, subject, predicate)] = key
        logger.info(
            "Seeded procedural sp_index: %d entries from %d keys",
            len(self.procedural_sp_index),
            len(keyed_pairs),
        )

    @staticmethod
    def dedup_episodic(qa_list: list[dict]) -> list[dict]:
        """Deduplicate episodic QA by (source_subject, source_predicate, source_object).

        Case-insensitive; first occurrence wins. Entries missing identity fields
        fall back to per-object identity so they survive rather than collide.
        """
        seen: set[tuple] = set()
        out: list[dict] = []
        for qa in qa_list:
            subj = (qa.get("source_subject") or "").strip().lower()
            pred = (qa.get("source_predicate") or "").strip().lower()
            obj = (qa.get("source_object") or "").strip().lower()
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

    def _extraction_kwargs(self, **overrides) -> dict:
        """Build the full kwarg set passed to extract_graph / extract_procedural_graph.

        Resolves each flag to the per-call override if given (not None), else
        the loop-level default. Single source of truth for every extraction path.
        """

        def pick(name: str, fallback):
            val = overrides.get(name, None)
            return fallback if val is None else val

        return dict(
            temperature=self.extraction_temperature,
            max_tokens=self.extraction_max_tokens,
            prompts_dir=self.prompts_dir,
            ha_context=overrides.get("ha_context"),
            stt_correction=pick("stt_correction", self.extraction_stt_correction),
            ha_validation=pick("ha_validation", self.extraction_ha_validation),
            noise_filter=pick("noise_filter", self.extraction_noise_filter),
            noise_filter_model=pick("noise_filter_model", self.extraction_noise_filter_model),
            noise_filter_endpoint=pick(
                "noise_filter_endpoint", self.extraction_noise_filter_endpoint
            ),
            speaker_name=overrides.get("speaker_name"),
            ner_check=pick("ner_check", self.extraction_ner_check),
            ner_model=pick("ner_model", self.extraction_ner_model),
            plausibility_judge=pick("plausibility_judge", self.extraction_plausibility_judge),
            plausibility_stage=pick("plausibility_stage", self.extraction_plausibility_stage),
            verify_anonymization=pick("verify_anonymization", self.extraction_verify_anonymization),
        )

    def _run_extract_graph(
        self,
        session_transcript: str,
        session_id: str,
        **overrides,
    ):
        """Single entry-point to extract_graph with unified flags + adapter guard.

        Every orchestrator (extract_session, run_cycle, future callers) goes
        through this helper so the pipeline cannot diverge by accident.
        """
        from peft import PeftModel as _PeftModel

        self._disable_gradient_checkpointing()
        kwargs = self._extraction_kwargs(**overrides)
        if isinstance(self.model, _PeftModel):
            with self.model.disable_adapter():
                return extract_graph(
                    self.model, self.tokenizer, session_transcript, session_id, **kwargs
                )
        return extract_graph(self.model, self.tokenizer, session_transcript, session_id, **kwargs)

    def _run_extract_procedural_graph(
        self,
        session_transcript: str,
        session_id: str,
        speaker_name: str | None = None,
        stt_correction: bool | None = None,
    ):
        """Single entry-point to extract_procedural_graph with adapter guard."""
        from peft import PeftModel as _PeftModel

        self._disable_gradient_checkpointing()
        stt = self.extraction_stt_correction if stt_correction is None else stt_correction
        call_kwargs = dict(
            max_tokens=self.extraction_max_tokens,
            prompts_dir=self.prompts_dir,
            stt_correction=stt,
            speaker_name=speaker_name,
        )
        if isinstance(self.model, _PeftModel):
            with self.model.disable_adapter():
                return extract_procedural_graph(
                    self.model, self.tokenizer, session_transcript, session_id, **call_kwargs
                )
        return extract_procedural_graph(
            self.model, self.tokenizer, session_transcript, session_id, **call_kwargs
        )

    def extract_session(
        self,
        session_transcript: str,
        session_id: str,
        speaker_id: str = "",
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
    ) -> tuple[list[dict], list[dict]]:
        """Extract and generate QA pairs from a session without training.

        Returns (episodic_qa, procedural_relations) for deferred training.
        Merges the session graph into the cumulative graph.
        """
        logger.info("=== Extraction (session=%s) ===", session_id)

        # --- EXTRACT ---
        session_graph = self._run_extract_graph(
            session_transcript,
            session_id,
            ha_context=ha_context,
            stt_correction=stt_correction,
            ha_validation=ha_validation,
            noise_filter=noise_filter,
            noise_filter_model=noise_filter_model,
            noise_filter_endpoint=noise_filter_endpoint,
            speaker_name=speaker_name,
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
        self.merger.merge(session_graph)
        self._triples_since_last_enrichment += len(session_graph.relations)

        # Save graph snapshot if debug mode
        if self.save_cycle_snapshots and self.snapshot_dir:
            snapshot_graph = self.snapshot_dir / f"cycle_{self.cycle_count}" / "graph.json"
            self.merger.save_graph(snapshot_graph)

        # --- GENERATE EPISODIC QA ---
        session_relations = [
            {
                "subject": r.subject,
                "predicate": r.predicate,
                "object": r.object,
                "relation_type": r.relation_type,
            }
            for r in session_graph.relations
        ]

        episodic_relations, procedural_rels = partition_relations(
            session_relations, procedural_enabled=self.procedural_config is not None
        )
        episodic_qa = generate_qa_from_relations(
            episodic_relations, model=self.model, tokenizer=self.tokenizer
        )

        # --- PROCEDURAL: separate extraction pass ---
        if self.procedural_config is not None:
            proc_graph = self._run_extract_procedural_graph(
                session_transcript,
                session_id,
                speaker_name=speaker_name,
                stt_correction=stt_correction,
            )
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
        episodic_qa = self.dedup_episodic(episodic_qa)
        procedural_rels = self.dedup_procedural(procedural_rels)

        self.last_session_graph = session_graph
        return episodic_qa, procedural_rels

    def train_adapters(
        self,
        all_episodic_qa: list[dict],
        all_procedural_relations: list[dict],
        speaker_id: str = "",
    ) -> dict:
        """Train all adapters once on accumulated QA pairs (blocking).

        Called after all sessions have been extracted.
        Returns dict with train losses per adapter.
        """
        self.cycle_count += 1
        result = {}

        if self.indexed_key_registry is None:
            logger.warning("No indexed key registry — skipping training")
            return result

        # --- EPISODIC ---
        if all_episodic_qa:
            episodic_loss = self._run_indexed_key_episodic(all_episodic_qa, new_promotions=[])
            if episodic_loss is not None:
                result["episodic_train_loss"] = episodic_loss

        # --- PROCEDURAL ---
        if self.procedural_config is not None and all_procedural_relations:
            procedural_loss = self._run_indexed_key_procedural(
                all_procedural_relations, speaker_id=speaker_id
            )
            if procedural_loss is not None:
                result["procedural_train_loss"] = procedural_loss

        # --- SAVE ---
        self._save_adapters()

        logger.info("Training complete: %s", result)
        return result

    def prepare_training_data(
        self,
        all_episodic_qa: list[dict],
        all_procedural_relations: list[dict],
        speaker_id: str = "",
    ) -> list[tuple[str, list[dict]]]:
        """Prepare keyed pairs for background training without training.

        Does key assignment, reconstruction, contradiction detection.
        Returns list of (adapter_name, keyed_pairs) tuples ready for training.

        State is snapshotted before preparation. Call rollback_preparation()
        if training fails to restore pre-preparation state.
        """
        # Snapshot mutable state for rollback on training failure
        import copy

        self._prep_snapshot = {
            "cycle_count": self.cycle_count,
            "indexed_next_index": self._indexed_next_index,
            "procedural_next_index": self._procedural_next_index,
            "indexed_key_qa": copy.deepcopy(self.indexed_key_qa),
            "procedural_sp_index": dict(self.procedural_sp_index),
            "episodic_simhash": dict(self.episodic_simhash),
            "semantic_simhash": dict(self.semantic_simhash),
            "procedural_simhash": dict(self.procedural_simhash),
        }

        self.cycle_count += 1
        jobs = []

        if self.indexed_key_registry is None:
            logger.warning("No indexed key registry — skipping preparation")
            return jobs

        # --- EPISODIC: assign keys + reconstruct ---
        if all_episodic_qa:
            episodic_keyed = self._prepare_episodic_keys(all_episodic_qa)
            if episodic_keyed:
                jobs.append(("episodic", episodic_keyed))

        # --- SEMANTIC: collect promoted keys ---
        semantic_keyed = self._collect_semantic_keys()
        if semantic_keyed:
            jobs.append(("semantic", semantic_keyed))

        # --- PROCEDURAL: assign + reconstruct ---
        if self.procedural_config is not None and all_procedural_relations:
            procedural_keyed = self._prepare_procedural_keys(all_procedural_relations, speaker_id)
            if procedural_keyed:
                jobs.append(("procedural", procedural_keyed))

        return jobs

    def rollback_preparation(self) -> None:
        """Restore state to before prepare_training_data was called.

        Called when background training fails to ensure no stale keys
        remain in the registry or QA store.
        """
        snap = getattr(self, "_prep_snapshot", None)
        if snap is None:
            logger.warning("No preparation snapshot to roll back")
            return

        self.cycle_count = snap["cycle_count"]
        self._indexed_next_index = snap["indexed_next_index"]
        self._procedural_next_index = snap["procedural_next_index"]
        self.indexed_key_qa = snap["indexed_key_qa"]
        self.procedural_sp_index = snap["procedural_sp_index"]
        self.episodic_simhash = snap["episodic_simhash"]
        self.semantic_simhash = snap["semantic_simhash"]
        self.procedural_simhash = snap["procedural_simhash"]
        self._prep_snapshot = None
        logger.info("Preparation state rolled back")

    def _prepare_episodic_keys(self, session_qa: list[dict]) -> list[dict]:
        """Assign keys and reconstruct existing episodic keys.

        Returns the full keyed pair list ready for training.
        Does NOT train — caller handles that.
        """
        switch_adapter(self.model, "episodic")
        self._disable_gradient_checkpointing()

        # Assign keys to new QA pairs
        new_keyed = assign_keys(session_qa, start_index=self._indexed_next_index)
        for kp in new_keyed:
            self.indexed_key_registry.add(kp["key"])
            self.indexed_key_qa[kp["key"]] = {
                "key": kp["key"],
                "question": kp["question"],
                "answer": kp["answer"],
                "source_subject": kp.get("source_subject", ""),
                "source_object": kp.get("source_object", ""),
                "speaker_id": kp.get("speaker_id", ""),
            }
        self._indexed_next_index += len(new_keyed)

        # Reconstruct existing episodic keys from adapter weights
        active_keys = self.indexed_key_registry.list_active()
        new_key_set = {kp["key"] for kp in new_keyed}
        existing_keys = [k for k in active_keys if k.startswith("graph") and k not in new_key_set]

        reconstructed = {}
        for key in existing_keys:
            recalled = probe_key(
                self.model,
                self.tokenizer,
                key,
                registry=self.episodic_simhash,
                confidence_threshold=0.5,
            )
            if recalled is not None and "failure_reason" not in recalled:
                reconstructed[key] = {
                    "key": key,
                    "question": recalled["question"],
                    "answer": recalled["answer"],
                }

        logger.info(
            "Episodic key reconstruction: %d/%d recovered",
            len(reconstructed),
            len(existing_keys),
        )

        # Build full keyed pair list
        episodic_keyed = []
        for key in existing_keys:
            if key in reconstructed:
                episodic_keyed.append(reconstructed[key])
            elif key in self.indexed_key_qa:
                qa = self.indexed_key_qa[key]
                episodic_keyed.append(
                    {"key": key, "question": qa["question"], "answer": qa["answer"]}
                )
        episodic_keyed.extend(new_keyed)

        return episodic_keyed

    def _collect_semantic_keys(self) -> list[dict]:
        """Collect all semantic keys for training."""
        if not self.semantic_simhash:
            return []

        semantic_keyed = []
        for key in self.semantic_simhash:
            if key in self.indexed_key_qa:
                qa = self.indexed_key_qa[key]
                semantic_keyed.append(
                    {"key": key, "question": qa["question"], "answer": qa["answer"]}
                )
        return semantic_keyed

    def _prepare_procedural_keys(
        self, procedural_relations: list[dict], speaker_id: str
    ) -> list[dict]:
        """Assign and reconstruct procedural keys.

        Expects pre-filtered preference relations from the dedicated
        procedural extraction prompt.
        Returns the full keyed pair list ready for training.
        """
        if not procedural_relations:
            return []

        logger.info("Found %d procedural relations", len(procedural_relations))

        # Generate QA pairs
        new_qa = generate_qa_from_relations(
            procedural_relations, model=self.model, tokenizer=self.tokenizer
        )
        if not new_qa:
            return []

        # Assign keys with proc prefix, per-relation speaker
        new_keyed = []
        for i, qa in enumerate(new_qa):
            key = f"proc{self._procedural_next_index}"
            self._procedural_next_index += 1
            rel_speaker = (
                procedural_relations[i].get("speaker_id", speaker_id)
                if i < len(procedural_relations)
                else speaker_id
            )
            keyed = {
                "key": key,
                "question": qa["question"],
                "answer": qa["answer"],
                "source_subject": qa.get("source_subject", ""),
                "source_object": qa.get("source_object", ""),
                "source_predicate": qa.get("source_predicate", ""),
                "speaker_id": rel_speaker,
            }
            new_keyed.append(keyed)

        # Contradiction check
        for kp in new_keyed:
            sp_key = (
                kp["speaker_id"],
                kp["source_subject"].lower(),
                kp["source_predicate"].lower(),
            )
            old_key = self.procedural_sp_index.get(sp_key)
            if old_key and old_key in self.procedural_simhash:
                logger.info("Procedural contradiction: retiring %s", old_key)
                self.procedural_simhash.pop(old_key, None)
                self.indexed_key_qa.pop(old_key, None)
                if self.indexed_key_registry is not None:
                    self.indexed_key_registry.remove(old_key)

        # Store new keys
        for kp in new_keyed:
            self.indexed_key_qa[kp["key"]] = kp
            if self.indexed_key_registry is not None:
                self.indexed_key_registry.add(kp["key"])
            sp_key = (
                kp["speaker_id"],
                kp["source_subject"].lower(),
                kp["source_predicate"].lower(),
            )
            self.procedural_sp_index[sp_key] = kp["key"]

        # Reconstruct existing procedural keys
        switch_adapter(self.model, "procedural")
        self._disable_gradient_checkpointing()

        existing_keys = [
            k for k in self.procedural_simhash if k not in {kp["key"] for kp in new_keyed}
        ]
        reconstructed = {}
        for key in existing_keys:
            recalled = probe_key(
                self.model,
                self.tokenizer,
                key,
                registry=self.procedural_simhash,
                confidence_threshold=0.5,
            )
            if recalled is not None and "failure_reason" not in recalled:
                reconstructed[key] = {
                    "key": key,
                    "question": recalled["question"],
                    "answer": recalled["answer"],
                }

        logger.info(
            "Procedural key reconstruction: %d/%d recovered",
            len(reconstructed),
            len(existing_keys),
        )

        all_procedural = []
        for key in existing_keys:
            if key in reconstructed:
                all_procedural.append(reconstructed[key])
            elif key in self.indexed_key_qa:
                qa = self.indexed_key_qa[key]
                all_procedural.append(
                    {"key": key, "question": qa["question"], "answer": qa["answer"]}
                )
        all_procedural.extend(new_keyed)

        return all_procedural

    def finalize_training(self) -> None:
        """Save adapters and registries after background training completes."""
        self._save_adapters()
        logger.info("Training finalized — adapters and registries saved")

    def run_cycle(
        self,
        session_transcript: str,
        session_id: str,
        speaker_id: str = "",
        speaker_name: str | None = None,
    ) -> CycleResult:
        """Run one consolidation cycle for a new session.

        Legacy method — extracts and trains in one pass.
        Used by experiment scripts. Server uses extract_session + train_adapters.

        Args:
            session_transcript: The raw session transcript text.
            session_id: Unique identifier for this session.
            speaker_id: Speaker identifier for preference scoping.

        Returns:
            CycleResult with metrics and timing.
        """
        start_time = time.time()
        self.cycle_count += 1
        result = CycleResult(cycle_index=self.cycle_count, session_id=session_id)

        logger.info(
            "=== Consolidation cycle %d (session=%s) ===",
            self.cycle_count,
            session_id,
        )

        # --- 1. EXTRACT ---
        # Unified extraction path: same helper as extract_session(), so every
        # SOTA pipeline flag configured on the loop is applied identically.
        session_graph = self._run_extract_graph(
            session_transcript,
            session_id,
            speaker_name=speaker_name,
        )

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

        # --- 4. GENERATE QA PAIRS ---
        # Episodic: only relations from the *current session's* extraction
        # (replay pool provides continuity with past sessions)
        session_relations = [
            {
                "subject": r.subject,
                "predicate": r.predicate,
                "object": r.object,
                "relation_type": r.relation_type,
            }
            for r in session_graph.relations
        ]
        episodic_relations, procedural_rels = partition_relations(
            session_relations, procedural_enabled=self.procedural_config is not None
        )

        # Mirror extract_session(): run the dedicated procedural prompt so
        # experiments exercise the same pipeline as production.
        if self.procedural_config is not None:
            proc_graph = self._run_extract_procedural_graph(
                session_transcript,
                session_id,
                speaker_name=speaker_name,
            )
            procedural_rels.extend(
                {
                    "subject": r.subject,
                    "predicate": r.predicate,
                    "object": r.object,
                    "relation_type": r.relation_type,
                }
                for r in proc_graph.relations
            )

        episodic_qa = generate_qa_from_relations(
            episodic_relations,
            model=self.model,
            tokenizer=self.tokenizer,
        )

        # Apply same dedup as server path (identical policy across all paths).
        episodic_qa = self.dedup_episodic(episodic_qa)
        procedural_rels = self.dedup_procedural(procedural_rels)

        # Promotion: relations for promoted entities (cap to keep training bounded)
        if new_promotions:
            promote_relations = get_relations_for_nodes(graph, new_promotions)
            if len(promote_relations) > 20:
                promote_relations = random.sample(promote_relations, 20)
            promote_qa = generate_qa_from_relations(
                promote_relations,
                model=self.model,
                tokenizer=self.tokenizer,
            )
        else:
            promote_qa = []

        # --- 4b. INDEXED KEY REPLAY (F4.9c validated) ---
        if self.indexed_key_registry is not None:
            episodic_loss = self._run_indexed_key_episodic(episodic_qa, new_promotions)
            if episodic_loss is not None:
                result.episodic_train_loss = episodic_loss

            if new_promotions:
                semantic_loss = self._run_indexed_key_semantic(new_promotions)
                if semantic_loss is not None:
                    result.semantic_train_loss = semantic_loss
                self.promoted_nodes.update(new_promotions)

            # --- 4c. PROCEDURAL (behavioral preferences) ---
            if self.procedural_config is not None and procedural_rels:
                procedural_loss = self._run_indexed_key_procedural(
                    procedural_rels, speaker_id=speaker_id
                )
                if procedural_loss is not None:
                    result.procedural_train_loss = procedural_loss
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
            if episodic_qa:
                episodic_loss = self._train_adapter_with_replay(
                    "episodic",
                    episodic_qa,
                    self.episodic_replay_pool,
                    self.config.episodic_new_weight,
                    f"phase3-episodic-cycle{self.cycle_count}",
                    recall_scores=episodic_recall_scores,
                )
                result.episodic_train_loss = episodic_loss

                # Add new QA pairs to episodic replay pool (cap at 100)
                self.episodic_replay_pool.extend(
                    {"question": qa["question"], "answer": qa["answer"]} for qa in episodic_qa
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

            # Add promoted QA pairs to semantic replay pool (cap at 100)
            self.semantic_replay_pool.extend(
                {"question": qa["question"], "answer": qa["answer"]} for qa in promote_qa
            )
            if len(self.semantic_replay_pool) > 100:
                self.semantic_replay_pool = self.semantic_replay_pool[-100:]

        # --- 7. DECAY ---
        if self.enable_entity_promotion:
            self._apply_decay(classification.decay)

        # --- 8. SAVE ---
        if self.persist_graph:
            self.merger.save_graph(self.graph_path)
        elif self.save_cycle_snapshots and self.snapshot_dir:
            snapshot_graph = self.snapshot_dir / f"cycle_{self.cycle_count}" / "graph.json"
            self.merger.save_graph(snapshot_graph)
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

    def _run_indexed_key_episodic(
        self,
        session_qa: list[dict],
        new_promotions: list[str],
    ) -> Optional[float]:
        """Run indexed key replay for the episodic adapter.

        1. Assign graphN keys to new session QA pairs
        2. Probe existing episodic keys → reconstruct from adapter weights
        3. Transfer promoted keys to semantic (remove from episodic)
        4. Build training data from all active episodic keys
        5. Train episodic adapter (indexed format)
        6. Update SimHash registry
        7. Retire stale keys on reconstruction cycles
        """
        switch_adapter(self.model, "episodic")
        self._disable_gradient_checkpointing()

        # Assign keys to new session QA pairs
        new_keyed = assign_keys(session_qa, start_index=self._indexed_next_index)
        for kp in new_keyed:
            self.indexed_key_registry.add(kp["key"])
            self.indexed_key_qa[kp["key"]] = {
                "key": kp["key"],
                "question": kp["question"],
                "answer": kp["answer"],
                "source_subject": kp.get("source_subject", ""),
                "source_object": kp.get("source_object", ""),
                "speaker_id": kp.get("speaker_id", ""),
            }
        self._indexed_next_index += len(new_keyed)

        # Promote keys for promoted entities (move to semantic before episodic training)
        # Check both source_subject and source_object — reverse QA templates
        # swap subject/object, so a promoted entity may appear in either field.
        promoted_keys = []
        if new_promotions:
            promoted_set = {n.lower() for n in new_promotions}
            for key, qa_info in list(self.indexed_key_qa.items()):
                if key.startswith("proc"):
                    continue  # Procedural keys never promote
                subject = qa_info.get("source_subject", "").lower()
                obj = qa_info.get("source_object", "").lower()
                mentions_promoted = (subject and subject in promoted_set) or (
                    obj and obj in promoted_set
                )
                if mentions_promoted and key not in self.semantic_simhash:
                    promoted_keys.append(key)

        # Reconstruct existing episodic keys from adapter weights
        active_keys = self.indexed_key_registry.list_active()
        # Only episodic keys (graph prefix), exclude new and promoted
        new_key_set = {kp["key"] for kp in new_keyed}
        promoted_key_set = set(promoted_keys)
        existing_keys = [
            k
            for k in active_keys
            if k.startswith("graph") and k not in new_key_set and k not in promoted_key_set
        ]

        reconstructed = {}
        for key in existing_keys:
            recalled = probe_key(
                self.model,
                self.tokenizer,
                key,
                registry=self.episodic_simhash,
                confidence_threshold=0.5,  # Lower threshold for reconstruction
            )
            if recalled is not None and "failure_reason" not in recalled:
                reconstructed[key] = {
                    "key": key,
                    "question": recalled["question"],
                    "answer": recalled["answer"],
                }

        logger.info(
            "Indexed key reconstruction: %d/%d existing keys recovered",
            len(reconstructed),
            len(existing_keys),
        )

        # Build full episodic keyed pair set
        episodic_keyed = []

        # Add reconstructed existing keys
        for key in existing_keys:
            if key in reconstructed:
                episodic_keyed.append(reconstructed[key])
            elif key in self.indexed_key_qa:
                # Fallback to stored QA if reconstruction failed
                qa = self.indexed_key_qa[key]
                episodic_keyed.append(
                    {
                        "key": key,
                        "question": qa["question"],
                        "answer": qa["answer"],
                    }
                )

        # Add new session keys
        episodic_keyed.extend(new_keyed)

        # Remove promoted keys from episodic
        for key in promoted_keys:
            self.indexed_key_registry.remove(key)
            # Move SimHash entry to semantic
            if key in self.episodic_simhash:
                self.semantic_simhash[key] = self.episodic_simhash.pop(key)

        if not episodic_keyed:
            return None

        # Format and train
        examples = format_indexed_training(episodic_keyed, self.tokenizer, max_length=1024)
        dataset = self._indexed_dataset(examples)
        training_config = self._make_training_config(num_epochs=self.training_config.num_epochs)
        self._enable_gradient_checkpointing()

        metrics = train_adapter(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            adapter_name="episodic",
            training_config=training_config,
            adapter_config=self.episodic_config,
            wandb_config=self.wandb_config,
            output_dir=self._training_output_dir("episodic"),
            run_name=f"phase4-indexed-episodic-cycle{self.cycle_count}",
            callbacks_extra=self._shutdown_callbacks,
        )

        # Update episodic SimHash registry from ground-truth QA
        self.episodic_simhash = build_registry(episodic_keyed)

        return metrics.get("train_loss")

    def _run_indexed_key_semantic(
        self,
        new_promotions: list[str],
    ) -> Optional[float]:
        """Train semantic adapter on promoted indexed keys.

        Collects all keys that belong to promoted entities and trains
        the semantic adapter on them with the indexed format.
        """
        # Collect promoted keys (already transferred in _run_indexed_key_episodic)
        promoted_set = {n.lower() for n in new_promotions}
        semantic_keyed = []
        for key, qa_info in self.indexed_key_qa.items():
            subject = qa_info.get("source_subject", "").lower()
            obj = qa_info.get("source_object", "").lower()
            mentions_promoted = (subject and subject in promoted_set) or (
                obj and obj in promoted_set
            )
            if mentions_promoted and key in self.semantic_simhash:
                semantic_keyed.append(
                    {
                        "key": key,
                        "question": qa_info["question"],
                        "answer": qa_info["answer"],
                    }
                )

        # Also include previously promoted keys still in semantic
        for key in list(self.semantic_simhash.keys()):
            if key not in {kp["key"] for kp in semantic_keyed} and key in self.indexed_key_qa:
                qa = self.indexed_key_qa[key]
                semantic_keyed.append(
                    {
                        "key": key,
                        "question": qa["question"],
                        "answer": qa["answer"],
                    }
                )

        if not semantic_keyed:
            logger.info("No promoted keys for semantic training")
            return None

        logger.info("Training semantic adapter on %d promoted keys", len(semantic_keyed))

        switch_adapter(self.model, "semantic")
        examples = format_indexed_training(semantic_keyed, self.tokenizer, max_length=1024)
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
            callbacks_extra=self._shutdown_callbacks,
        )

        # Update semantic SimHash registry
        self.semantic_simhash = build_registry(semantic_keyed)

        return metrics.get("train_loss")

    def _run_indexed_key_procedural(
        self,
        procedural_relations: list[dict],
        speaker_id: str = "",
    ) -> Optional[float]:
        """Train procedural adapter on preference/habit relations.

        Expects pre-filtered preference relations (caller runs
        partition_relations before passing).

        Same mechanism as episodic: reconstruct existing keys from adapter
        weights, add new keys, retrain the full set. Knowledge lives in
        the weights.

        Contradiction handling: when a new preference shares the same
        (speaker_id, subject, predicate) as an existing key, the old key
        is retired and replaced.

        Registry/index invariant: all mutations to ``procedural_simhash``,
        ``indexed_key_qa``, ``indexed_key_registry``, and
        ``procedural_sp_index`` are deferred until **after**
        ``train_adapter`` returns successfully.  If training raises, shared
        state is left unchanged so the caller can safely retry or skip.
        """
        if not procedural_relations:
            return None

        logger.info(
            "Found %d procedural relations",
            len(procedural_relations),
        )

        # Generate QA pairs from preference relations
        new_qa = generate_qa_from_relations(
            procedural_relations, model=self.model, tokenizer=self.tokenizer
        )
        if not new_qa:
            return None

        # Assign keys with proc prefix — use per-relation speaker_id.
        # Use a local tentative counter so self._procedural_next_index is not
        # advanced until after train_adapter returns successfully.  If training
        # raises, the index slots are not burned and a retry will reuse them.
        new_keyed = []
        tentative_next_index = self._procedural_next_index
        for i, qa in enumerate(new_qa):
            key = f"proc{tentative_next_index}"
            tentative_next_index += 1
            # Get speaker from the original relation if available
            rel_speaker = (
                procedural_relations[i].get("speaker_id", speaker_id)
                if i < len(procedural_relations)
                else speaker_id
            )
            keyed = {
                "key": key,
                "question": qa["question"],
                "answer": qa["answer"],
                "source_subject": qa.get("source_subject", ""),
                "source_object": qa.get("source_object", ""),
                "source_predicate": qa.get("source_predicate", ""),
                "speaker_id": rel_speaker,
            }
            new_keyed.append(keyed)

        # Compute intended mutations without touching shared state yet.
        # keys_to_retire: old contradicted keys that will be removed on success.
        # new_sp_mappings: (sp_key -> new_key) entries for procedural_sp_index.
        keys_to_retire: list[str] = []
        new_sp_mappings: dict[tuple, str] = {}
        new_key_set = {kp["key"] for kp in new_keyed}
        for kp in new_keyed:
            sp_key = (
                kp["speaker_id"],
                kp["source_subject"].lower(),
                kp["source_predicate"].lower(),
            )
            old_key = self.procedural_sp_index.get(sp_key)
            if old_key and old_key in self.procedural_simhash:
                logger.info(
                    "Procedural contradiction: retiring %s for %s",
                    old_key,
                    sp_key,
                )
                keys_to_retire.append(old_key)
            new_sp_mappings[sp_key] = kp["key"]

        # Reconstruct existing procedural keys from adapter weights.
        # existing_keys excludes both new keys and keys scheduled for retirement
        # so the training set is identical to what a post-commit read would see.
        retired_set = set(keys_to_retire)
        switch_adapter(self.model, "procedural")
        self._disable_gradient_checkpointing()

        existing_keys = [
            k for k in self.procedural_simhash if k not in new_key_set and k not in retired_set
        ]
        reconstructed = {}
        for key in existing_keys:
            recalled = probe_key(
                self.model,
                self.tokenizer,
                key,
                registry=self.procedural_simhash,
                confidence_threshold=0.5,
            )
            if recalled is not None and "failure_reason" not in recalled:
                reconstructed[key] = {
                    "key": key,
                    "question": recalled["question"],
                    "answer": recalled["answer"],
                }

        logger.info(
            "Procedural key reconstruction: %d/%d existing keys recovered",
            len(reconstructed),
            len(existing_keys),
        )

        # Build full procedural training set from reconstructed + fallback QA + new.
        # Read existing QA from self.indexed_key_qa directly — shared state is still
        # unmodified at this point.
        all_procedural = []
        for key in existing_keys:
            if key in reconstructed:
                all_procedural.append(reconstructed[key])
            elif key in self.indexed_key_qa:
                qa = self.indexed_key_qa[key]
                all_procedural.append(
                    {"key": key, "question": qa["question"], "answer": qa["answer"]}
                )
        all_procedural.extend(new_keyed)

        if not all_procedural:
            return None

        logger.info(
            "Training procedural adapter on %d keys (%d new)",
            len(all_procedural),
            len(new_keyed),
        )

        examples = format_indexed_training(all_procedural, self.tokenizer, max_length=1024)
        dataset = self._indexed_dataset(examples)
        training_config = self._make_training_config(num_epochs=self.training_config.num_epochs)
        self._enable_gradient_checkpointing()

        # train_adapter may raise — shared state must not have been mutated yet.
        metrics = train_adapter(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            adapter_name="procedural",
            training_config=training_config,
            adapter_config=self.procedural_config,
            wandb_config=self.wandb_config,
            output_dir=self._training_output_dir("procedural"),
            run_name=f"phase4-indexed-procedural-cycle{self.cycle_count}",
            callbacks_extra=self._shutdown_callbacks,
        )

        # Training succeeded — apply all deferred mutations atomically.
        # 1. Commit the next-index counter now that key slots are confirmed used.
        self._procedural_next_index = tentative_next_index

        # 2. Retire contradicted keys from all indexes.
        for old_key in keys_to_retire:
            self.procedural_simhash.pop(old_key, None)
            self.indexed_key_qa.pop(old_key, None)
            if self.indexed_key_registry is not None:
                self.indexed_key_registry.remove(old_key)

        # 3. Register new QA pairs, update sp_index, and add simhash entries.
        #    Explicit incremental mutations keep all four indexes consistent and
        #    symmetric — no full reconstruction needed, since existing-key hashes
        #    are unchanged (compute_simhash is deterministic from key/question/answer).
        for kp in new_keyed:
            self.indexed_key_qa[kp["key"]] = kp
            if self.indexed_key_registry is not None:
                self.indexed_key_registry.add(kp["key"])
            self.procedural_simhash[kp["key"]] = compute_simhash(
                kp["key"], kp["question"], kp["answer"]
            )
        for sp_key, new_key in new_sp_mappings.items():
            self.procedural_sp_index[sp_key] = new_key

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
        """Train adapter on new QA pairs with optional replay.

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
            callbacks_extra=self._shutdown_callbacks,
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
        """Remove decayed nodes' QA pairs from the episodic replay pool.

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
            logger.info("Decayed %d QA pairs from episodic replay pool", removed)
        if protected > 0:
            logger.info("Protected %d QA pairs from decay (min exposure not met)", protected)

    def _qa_to_dataset(self, qa_pairs: list[dict]) -> Dataset:
        """Convert QA pair dicts to a SyntheticQADataset."""
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

    def _save_adapters(self) -> None:
        """Save adapters and registries to disk.

        Saves to two locations:
        - output_dir/episodic/, output_dir/semantic/ — latest state (server use)
        - output_dir/cycle_N/episodic/, cycle_N/semantic/ — per-cycle snapshots (analysis)
        """
        # Latest state at flat paths
        save_adapter(self.model, self.output_dir, "episodic")
        if "semantic" in self.model.peft_config:
            save_adapter(self.model, self.output_dir, "semantic")
        if "procedural" in self.model.peft_config:
            save_adapter(self.model, self.output_dir, "procedural")

        # Per-cycle snapshots (debug/analysis only)
        if self.save_cycle_snapshots:
            base = self.snapshot_dir if self.snapshot_dir else self.output_dir
            cycle_dir = base / f"cycle_{self.cycle_count}"
            save_adapter(self.model, cycle_dir / "episodic", "episodic")
            if "semantic" in self.model.peft_config:
                save_adapter(self.model, cycle_dir / "semantic", "semantic")
            if "procedural" in self.model.peft_config:
                save_adapter(self.model, cycle_dir / "procedural", "procedural")

        if self.indexed_key_registry is not None:
            self.indexed_key_registry.save(self.output_dir / "indexed_key_registry.json")
            save_registry(self.episodic_simhash, self.output_dir / "simhash_registry_episodic.json")
            save_registry(self.semantic_simhash, self.output_dir / "simhash_registry_semantic.json")
            save_registry(
                self.procedural_simhash,
                self.output_dir / "simhash_registry_procedural.json",
            )

    def _training_output_dir(self, adapter_name: str, *, interim_stamp: str | None = None) -> Path:
        """Training checkpoint directory for the current cycle or interim pass.

        When *interim_stamp* is given (either directly or via the instance
        attribute ``_current_interim_stamp`` set by ``post_session_train``),
        the directory is derived from the stamp rather than ``cycle_count`` so
        that consecutive post-session calls within the same sub-interval do not
        share a directory, and calls across different stamps always land in
        distinct directories.

        For ordinary run_cycle / train_adapters calls (no stamp) the behaviour is
        unchanged: snapshot_dir or output_dir with ``cycle_N`` infix.

        Args:
            adapter_name: The adapter being trained (e.g. ``"episodic"``,
                ``"procedural"``).
            interim_stamp: Optional YYYYMMDDTHHMM stamp used by
                ``post_session_train`` to keep each sub-interval in its own
                output directory.  When set, cycle_count is NOT used.  May also
                be provided implicitly via ``self._current_interim_stamp``.

        Returns:
            Absolute :class:`~pathlib.Path` for training checkpoints.
        """
        # Resolve stamp: explicit kwarg wins, then instance attribute (set by
        # post_session_train to thread the stamp into helper methods like
        # _run_indexed_key_procedural that don't accept it directly).
        resolved_stamp = interim_stamp or getattr(self, "_current_interim_stamp", None)
        if resolved_stamp is not None:
            # Interim-pass layout: output_dir/interim_<stamp>/<adapter_name>/
            return self.output_dir / f"interim_{resolved_stamp}" / adapter_name
        if self.save_cycle_snapshots and self.snapshot_dir:
            return self.snapshot_dir / f"cycle_{self.cycle_count}" / adapter_name
        return self.output_dir / f"cycle_{self.cycle_count}" / adapter_name

    def post_session_train(
        self,
        session_transcript: str,
        session_id: str,
        *,
        speaker_id: str = "",
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

        Implementation ordering follows the CLAUDE.md ``seen_triples`` discipline:
        register keys **only after** training returns successfully so that extraction
        or training failures never leave orphaned keys in the registry.

        If ``max_interim_count == 0``, new facts are appended to
        ``self.pending_interim_triples`` (RAM-only) without training.  The pending
        queue is consumed by the next consolidation cycle (Step 7).  What is not
        trained does not exist on disk — this upholds the privacy invariant.

        Procedural relations extracted from the same transcript are trained onto
        the stable ``procedural`` main adapter (no interim tier for procedural —
        preferences are small-volume and slow-changing).  This pass runs inline,
        immediately after the episodic training pass and before any registry
        writes, so a failure in either pass leaves the registry clean.

        Registry write ordering (I5 atomicity):
        1. Save adapter weights (episodic interim + procedural main).
        2. Save SimHash registries.
        3. Save keyed_pairs.json.
        4. Save key registry (LAST — its presence signals a clean commit).

        On restart, the lifespan consistency check scans the registry and drops
        any entry whose adapter file is missing, recovering from a crash between
        steps 1-3 and 4.

        Args:
            session_transcript: Raw transcript text for this conversation.
            session_id: Unique conversation identifier (used by the extraction pipeline).
            speaker_id: Speaker identifier for key ownership and preference scoping.
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
        from paramem.server.interim_adapter import create_interim_adapter
        from paramem.training.indexed_memory import (
            assign_keys,
            build_registry,
            format_indexed_training,
        )

        # --- 1. Extract ---
        episodic_qa, procedural_rels = self.extract_session(
            session_transcript,
            session_id,
            speaker_id=speaker_id,
            speaker_name=speaker_name,
            ha_context=ha_context,
        )

        triples_extracted = len(episodic_qa)
        logger.info(
            "post_session_train: session=%s extracted %d episodic QA pairs",
            session_id,
            triples_extracted,
        )

        # --- 2. Noop: no facts extracted ---
        if triples_extracted == 0:
            return {
                "triples_extracted": 0,
                "new_keys": [],
                "adapter_name": None,
                "mode": "noop",
                "error": None,
            }

        # --- 3. Queue branch: max_interim_count == 0 ---
        if max_interim_count == 0:
            # Append to RAM queue for the next consolidation cycle (Step 7).
            # The pending queue is picked up by consolidate() and folded into the
            # cumulative graph before the main rebuild.  Queue lives in RAM;
            # snapshot persistence is deferred — matches the privacy invariant
            # "what isn't trained doesn't exist".
            # TODO(Step 7): consume self.pending_interim_triples at the start of
            # consolidate_interim_adapters() before training the full key set.
            if not hasattr(self, "pending_interim_triples"):
                self.pending_interim_triples: list[dict] = []
            self.pending_interim_triples.extend(episodic_qa)
            logger.info(
                "post_session_train: max_interim_count=0 — queued %d triples (total pending: %d)",
                len(episodic_qa),
                len(self.pending_interim_triples),
            )
            return {
                "triples_extracted": triples_extracted,
                "new_keys": [],
                "adapter_name": None,
                "mode": "queued",
                "error": None,
            }

        # --- 4. Normal branch: compute stamp and adapter name ---
        # NOTE: ``schedule`` is the refresh_cadence here (historical parameter
        # name kept on the method signature for call-site stability; semantics
        # changed to match the new config model). The stamp IS the cadence
        # boundary — no division by max_interim_count.
        if stamp is None:
            from paramem.server.interim_adapter import current_interim_stamp as _cis

            stamp = _cis(schedule)
        adapter_name = f"episodic_interim_{stamp}"

        # Cap-reached detection.  When every VRAM slot is occupied by an
        # interim adapter for a previous sub-interval and the current
        # sub-interval has not yet minted one, redirect the new keys into
        # the *newest* existing interim and retrain it from scratch on the
        # union of its old keys + the new ones.  This keeps the PEFT
        # adapter count at or below ``max_interim_count`` without rolling
        # eviction (which would silently drop knowledge).  The retrain is
        # followed by a recall sanity probe; a failed probe rolls back to
        # the snapshot and marks the adapter ``degenerated`` for the rest
        # of the cycle.
        cap_reached_absorb = False
        if adapter_name not in self.model.peft_config:
            existing_interim = sorted(
                a for a in self.model.peft_config if a.startswith("episodic_interim_")
            )
            if self.indexed_key_registry is not None and len(existing_interim) >= max_interim_count:
                newest = existing_interim[-1]  # lex-max stamp = most recent
                # Health gate: refuse to retrain an adapter already marked
                # degenerated this cycle.  Queue the new facts in RAM so
                # nothing is silently dropped; the next full consolidation
                # clears the degenerated flag and folds the queue in.
                if not self.indexed_key_registry.is_adapter_healthy(newest):
                    if not hasattr(self, "pending_interim_triples"):
                        self.pending_interim_triples: list[dict] = []
                    self.pending_interim_triples.extend(episodic_qa)
                    logger.warning(
                        "post_session_train: newest interim %s is degenerated — "
                        "queued %d triples for next full consolidation (pending: %d)",
                        newest,
                        len(episodic_qa),
                        len(self.pending_interim_triples),
                    )
                    return {
                        "triples_extracted": triples_extracted,
                        "new_keys": [],
                        "adapter_name": newest,
                        "mode": "degenerated",
                        "error": "target_adapter_degenerated",
                    }
                logger.info(
                    "post_session_train: interim cap (%d >= %d) reached — "
                    "absorbing %d new triples into newest interim %s via full retrain",
                    len(existing_interim),
                    max_interim_count,
                    triples_extracted,
                    newest,
                )
                adapter_name = newest
                cap_reached_absorb = True
            else:
                # Sub-interval rollover: first session of a new stamp.  If the
                # floor is crossed and interim enrichment is enabled, run a
                # mini graph-enrichment pass on the cumulative graph now so
                # the 84h full-consolidation amortises its SOTA cost across
                # the interim windows.  Wrapped so any SOTA failure is
                # logged but never blocks interim adapter creation.
                if (
                    self.graph_enrichment_interim_enabled
                    and self._triples_since_last_enrichment
                    >= self.graph_enrichment_min_triples_floor
                ):
                    logger.info(
                        "post_session_train: interim rollover — running mini "
                        "graph-enrichment (triples_since_last=%d ≥ floor=%d)",
                        self._triples_since_last_enrichment,
                        self.graph_enrichment_min_triples_floor,
                    )
                    try:
                        self._run_graph_enrichment()
                    except Exception as _mini_exc:
                        logger.warning(
                            "post_session_train: mini graph-enrichment raised "
                            "— interim creation continues: %s",
                            _mini_exc,
                        )

                # Normal: create a fresh interim adapter for this sub-interval.
                # create_interim_adapter is idempotent — no-op when the adapter
                # already exists.  The return value MUST be assigned back to
                # self.model; PEFT's add_adapter may rebind the PeftModel wrapper.
                self.model = create_interim_adapter(self.model, self.episodic_config, stamp)
                logger.info("post_session_train: created interim adapter %s", adapter_name)

        # --- 5. Assign keys to the new QA pairs ---
        if self.indexed_key_registry is None:
            logger.warning("post_session_train: no indexed key registry — aborting")
            return {
                "triples_extracted": triples_extracted,
                "new_keys": [],
                "adapter_name": adapter_name,
                "mode": "noop",
                "error": "no_registry",
            }

        # Tag QA pairs with speaker_id before key assignment.
        for qa in episodic_qa:
            if "speaker_id" not in qa:
                qa["speaker_id"] = speaker_id

        new_keyed = assign_keys(episodic_qa, start_index=self._indexed_next_index)

        # Collect all existing keys for this interim adapter so we can rebuild
        # the full training set (avoids catastrophic forgetting — same pattern
        # as _run_indexed_key_episodic's full-replay approach).
        existing_interim_keyed: list[dict] = []
        for k in self.indexed_key_registry.keys_for_adapter(adapter_name):
            qa_info = self.indexed_key_qa.get(k)
            if qa_info is not None:
                existing_interim_keyed.append(
                    {"key": k, "question": qa_info["question"], "answer": qa_info["answer"]}
                )

        all_interim_keyed = existing_interim_keyed + new_keyed

        # --- 6. Train episodic interim adapter ---
        # On failure, let the exception propagate; caller logs and returns error.
        # The registry is written AFTER both training passes succeed (I5 atomicity).
        from paramem.models.loader import copy_adapter_weights, create_adapter, switch_adapter

        # Cap-reached retrain: snapshot the target adapter, then fresh-init its
        # weights so the retrain starts from LoRA zeros rather than warm-starting
        # from accumulated interference.  Snapshot is kept in a side slot and
        # either discarded (success) or copied back (sanity fail).
        snapshot_name: str | None = None
        if cap_reached_absorb:
            snapshot_name = f"{adapter_name}_retrain_snapshot"
            if snapshot_name in self.model.peft_config:
                self.model.delete_adapter(snapshot_name)
            self.model = create_adapter(self.model, self.episodic_config, snapshot_name)
            copy_adapter_weights(self.model, src=adapter_name, dst=snapshot_name)
            self.model.delete_adapter(adapter_name)
            self.model = create_adapter(self.model, self.episodic_config, adapter_name)

        switch_adapter(self.model, adapter_name)
        self._disable_gradient_checkpointing()
        examples = format_indexed_training(all_interim_keyed, self.tokenizer, max_length=1024)
        dataset = self._indexed_dataset(examples)
        training_config = self._make_training_config(num_epochs=self.training_config.num_epochs)
        self._enable_gradient_checkpointing()

        from paramem.training.trainer import train_adapter as _train_adapter

        _train_adapter(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            adapter_name=adapter_name,
            training_config=training_config,
            adapter_config=self.episodic_config,
            wandb_config=self.wandb_config,
            # I3: use interim_stamp so consecutive post-session calls within the
            # same sub-interval do NOT share a directory, and different stamps
            # always produce distinct directories.
            output_dir=self._training_output_dir(adapter_name, interim_stamp=stamp),
            run_name=f"interim-{adapter_name}-{session_id}",
            callbacks_extra=self._shutdown_callbacks,
        )

        # --- 6a. Recall sanity probe (cap-reached retrain only) ---
        # The normal append-to-existing path warm-starts from weights that
        # already encode the earlier keys, so incremental degradation is
        # rare.  The cap-reached retrain reinitialises and trains on every
        # key the adapter has ever seen plus the new ones; if capacity is
        # exhausted the probe is the only thing that catches it before the
        # registry is mutated.
        if cap_reached_absorb:
            recall_rate = self._run_recall_sanity_probe(adapter_name, all_interim_keyed)
            if recall_rate < recall_sanity_threshold:
                # Rollback: restore snapshot weights, mark the adapter
                # degenerated, skip procedural + registration, persist the
                # registry so the flag survives a restart.
                copy_adapter_weights(self.model, src=snapshot_name, dst=adapter_name)
                self.model.delete_adapter(snapshot_name)
                self.indexed_key_registry.set_adapter_health(
                    adapter_name,
                    ADAPTER_HEALTH_DEGENERATED,
                    reason=(
                        f"recall {recall_rate:.3f} < {recall_sanity_threshold:.2f} "
                        f"after cap-reached retrain on {len(all_interim_keyed)} keys"
                    ),
                )
                self.indexed_key_registry.save(self.output_dir / "indexed_key_registry.json")
                logger.warning(
                    "post_session_train: cap-reached retrain failed sanity "
                    "(recall=%.3f < %.2f) on %s — rolled back, marked degenerated",
                    recall_rate,
                    recall_sanity_threshold,
                    adapter_name,
                )
                if "episodic" in self.model.peft_config:
                    switch_adapter(self.model, "episodic")
                return {
                    "triples_extracted": triples_extracted,
                    "new_keys": [],
                    "adapter_name": adapter_name,
                    "mode": "degenerated",
                    "error": f"recall_sanity_failed:{recall_rate:.3f}",
                }
            # Success — drop the snapshot slot and continue.
            self.model.delete_adapter(snapshot_name)
            snapshot_name = None
            logger.info(
                "post_session_train: cap-reached retrain passed sanity (recall=%.3f) on %s",
                recall_rate,
                adapter_name,
            )

        # --- 6b. Train procedural main adapter (I1) ---
        # Procedural relations always go to the stable 'procedural' main adapter —
        # no interim tier for procedural (preferences are small-volume,
        # slow-changing).  This pass runs inline so a failure here also leaves the
        # registry clean (registry writes happen in step 7 below).
        #
        # _run_indexed_key_procedural calls self._training_output_dir("procedural")
        # internally.  We set _current_interim_stamp so _training_output_dir routes
        # checkpoints under the same interim_<stamp> directory rather than
        # cycle_<N>, keeping each post-session pass in its own directory (I3).
        # The attribute is cleared in a finally block regardless of outcome.
        if self.procedural_config is not None and procedural_rels:
            self._current_interim_stamp = stamp
            try:
                self._run_indexed_key_procedural(
                    procedural_rels,
                    speaker_id=speaker_id,
                )
            finally:
                self._current_interim_stamp = None

        # --- 7. Register episodic keys ONLY after both training passes succeed ---
        # Procedural keys are registered inside _run_indexed_key_procedural (same
        # invariant — keys added only after train_adapter returns).
        new_key_ids: list[str] = []
        for kp in new_keyed:
            k = kp["key"]
            self.indexed_key_registry.add(k, adapter_id=adapter_name)
            self.indexed_key_qa[k] = {
                "key": k,
                "question": kp["question"],
                "answer": kp["answer"],
                "source_subject": kp.get("source_subject", ""),
                "source_object": kp.get("source_object", ""),
                "speaker_id": kp.get("speaker_id", speaker_id),
            }
            new_key_ids.append(k)
        self._indexed_next_index += len(new_keyed)

        # I5 — Registry-last write order.  Adapter weights must hit disk first;
        # then SimHash; then keyed_pairs.json; then the registry.  The registry
        # is the "commit" signal: its presence means every preceding file is
        # complete.  If the process is killed between adapter-save and
        # registry-save, the lifespan consistency check (app.py) drops the
        # orphan registry entries on the next startup.

        # Step 1: Save adapter weights so the router and startup loader can pick them up.
        from paramem.models.loader import save_adapter as _save_adapter

        _save_adapter(self.model, self.output_dir, adapter_name)
        if self.procedural_config is not None and "procedural" in self.model.peft_config:
            _save_adapter(self.model, self.output_dir, "procedural")

        # Step 2: Save SimHash registries.
        interim_simhash = build_registry(all_interim_keyed)
        save_registry(
            interim_simhash,
            self.output_dir / f"simhash_registry_{adapter_name}.json",
        )
        # Procedural simhash is updated inside _run_indexed_key_procedural; persist
        # the updated state here so it survives a restart even if the process is
        # killed before the full _save_adapters() call at end of run_cycle.
        if self.procedural_config is not None and procedural_rels:
            save_registry(
                self.procedural_simhash,
                self.output_dir / "simhash_registry_procedural.json",
            )

        # Step 3: Save keyed_pairs.json inside the interim adapter subdir so the
        # router can index entity→key mappings without loading every adapter.
        interim_pairs = [
            {"key": kp["key"], "question": kp["question"], "answer": kp["answer"]}
            for kp in all_interim_keyed
        ]
        import json as _json
        import os as _os

        interim_dir = self.output_dir / adapter_name
        interim_dir.mkdir(parents=True, exist_ok=True)
        _tmp = interim_dir / "keyed_pairs.json.tmp"
        with open(_tmp, "w") as _f:
            _json.dump(interim_pairs, _f, indent=2)
        _os.replace(_tmp, interim_dir / "keyed_pairs.json")

        # Step 4 (LAST): Persist key registry — its presence is the commit signal.
        self.indexed_key_registry.save(self.output_dir / "indexed_key_registry.json")

        # Restore the main episodic adapter as the active adapter so subsequent
        # inference probes default to the consolidated tier.
        if "episodic" in self.model.peft_config:
            switch_adapter(self.model, "episodic")

        logger.info(
            "post_session_train: trained %s — %d new keys, %d total keys",
            adapter_name,
            len(new_key_ids),
            len(all_interim_keyed),
        )

        return {
            "triples_extracted": triples_extracted,
            "new_keys": new_key_ids,
            "adapter_name": adapter_name,
            "mode": "trained",
            "error": None,
        }

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

        provider = self.extraction_noise_filter
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

        filter_model = self.extraction_noise_filter_model
        endpoint = self.extraction_noise_filter_endpoint or None
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
          section; the _PauseForInferenceCallback only fires at epoch/step
          boundaries and would not protect non-training code paths.

        Steps:
        1. Verify the GPU lock is held (entry guard — leak-safe pattern).
        2. Reset seen_triples (fresh rebuild makes prior state irrelevant).
        3. Walk all active keys, re-derive their tier from the cumulative graph,
           handle graph-drift keys, and build per-tier keyed-pair lists.
        4. Load backup adapters (if available) and all interim adapters into PEFT.
        5. For each tier (episodic → semantic → procedural):
           a. Set active adapter to <tier>_backup before deleting the main.
           b. delete_adapter(<tier>) + create_adapter(<tier>).
           c. Set <tier>_is_training=True, call _train_adapter, set False.
           d. Recall-sanity check; on failure roll back and abort.
        6. Atomic finalize: registry rewrite → unload_interim_adapters → router reload.
        7. On success: reset triples_since_last_full; unload backup adapters.

        Args:
            trainer: BackgroundTrainer instance (must be the one holding the GPU
                lock via submit()).  Required for per-tier B2 re-arm pattern.
            router: Router instance whose reload() is called at the end of the
                atomic finalize sequence.  Optional — skipped when None.
            recall_sanity_threshold: Minimum recall rate to accept a rebuilt tier
                (default 0.95).  Tiers below this trigger rollback and abort.
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
        from paramem.models.loader import create_adapter
        from paramem.server.gpu_lock import _gpu_thread_lock
        from paramem.server.interim_adapter import unload_interim_adapters

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

        # --- Step 1: Reset seen_triples ---
        self.seen_triples = set()

        # --- Step 1b: Graph-level SOTA enrichment (Task #10) ---
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

        active_keys = self.indexed_key_registry.list_active() if self.indexed_key_registry else []

        tier_keyed: dict[str, list[dict]] = {
            "episodic": [],
            "semantic": [],
            "procedural": [],
        }
        graph_drift_count = 0

        for key in active_keys:
            qa_info = self.indexed_key_qa.get(key)
            if qa_info is None:
                # No QA metadata — skip (should not happen with intact registry)
                logger.warning("consolidate_interim_adapters: no QA metadata for key %s", key)
                continue

            src_subj = qa_info.get("source_subject", "")
            src_pred = qa_info.get("source_predicate", "")
            src_obj = qa_info.get("source_object", "")
            triple_key = (src_subj, src_pred, src_obj)

            current_adapter_id = self.indexed_key_registry.get_adapter_id(key)

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
                    "graph_drift_key key=%s source_subject=%r source_predicate=%r "
                    "source_object=%r current_adapter_id=%r",
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
                        "main",  # legacy default from KeyRegistry
                    }, (
                        f"Registry invariant violation: key {key!r} has "
                        f"unexpected adapter_id {current_adapter_id!r}"
                    )
                    tier = (
                        current_adapter_id
                        if current_adapter_id in {"episodic", "semantic", "procedural"}
                        else "episodic"
                    )

            tier_keyed[tier].append(
                {
                    "key": key,
                    "question": qa_info["question"],
                    "answer": qa_info["answer"],
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
                keyed_pairs=tier_keyed["episodic"],
                adapter_name="episodic",
                adapter_config=self.episodic_config,
                inference_fallback_adapter="episodic_backup",
            ),
            "semantic": TrainingJob(
                keyed_pairs=tier_keyed["semantic"],
                adapter_name="semantic",
                adapter_config=self.semantic_config,
                inference_fallback_adapter="semantic_backup",
            ),
            "procedural": TrainingJob(
                keyed_pairs=tier_keyed["procedural"],
                adapter_name="procedural",
                adapter_config=self.procedural_config or self.episodic_config,
                inference_fallback_adapter="procedural_backup",
            ),
        }

        # --- Step 5: Per-tier fresh-adapter rebuild ---
        # Load backup adapters (PEFT load-or-skip idempotency: NC-2).
        # In the absence of a real encrypted backup dir we skip the load
        # and fall back to the live main adapters as the de-facto backup.
        # Production systems should implement the Fernet backup path (Step 7d).
        for backup_name in ("episodic_backup", "semantic_backup", "procedural_backup"):
            if backup_name not in self.model.peft_config:
                # No backup available — copy main weights into backup slot.
                # This keeps PEFT adapter count ≥ 5 throughout the rebuild.
                base_tier = backup_name.replace("_backup", "")
                if base_tier in self.model.peft_config:
                    from paramem.models.loader import copy_adapter_weights

                    self.model = create_adapter(self.model, self.episodic_config, backup_name)
                    copy_adapter_weights(self.model, src=base_tier, dst=backup_name)
                    logger.info(
                        "consolidate_interim_adapters: created in-memory backup %s from %s",
                        backup_name,
                        base_tier,
                    )

        tiers_rebuilt: list[str] = []
        recall_per_tier: dict[str, float] = {}
        rollback_tier: str | None = None

        for tier in ("episodic", "semantic", "procedural"):
            backup_name = f"{tier}_backup"
            job = jobs_by_tier[tier]

            if not job.keyed_pairs:
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
                from paramem.training.indexed_memory import format_indexed_training
                from paramem.training.trainer import train_adapter as _train_adapter_fn

                examples = format_indexed_training(job.keyed_pairs, self.tokenizer, max_length=1024)
                if examples:
                    dataset = self._indexed_dataset(examples)
                    self._enable_gradient_checkpointing()
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
                        callbacks_extra=self._shutdown_callbacks,
                    )
                    logger.info(
                        "consolidate_interim_adapters: trained %s on %d keys",
                        tier,
                        len(job.keyed_pairs),
                    )
            finally:
                if trainer is not None:
                    trainer._set_is_training(False)
                    trainer._current_job = prior_job

            # e. Recall-sanity check (Step 7d) — shared helper with cap-reached retrain.
            recall_rate = self._run_recall_sanity_probe(tier, job.keyed_pairs)

            recall_per_tier[tier] = recall_rate
            logger.info(
                "consolidate_interim_adapters: recall check tier=%s rate=%.3f threshold=%.3f",
                tier,
                recall_rate,
                recall_sanity_threshold,
            )

            if recall_rate < recall_sanity_threshold:
                # Capacity ceiling hit — rollback and abort.
                rollback_tier = tier
                logger.error(
                    "consolidate_interim_adapters: recall %.3f < threshold %.3f for tier %s "
                    "— rolling back and aborting finalize",
                    recall_rate,
                    recall_sanity_threshold,
                    tier,
                )
                self._append_capacity_ceiling_log(
                    tier=tier,
                    n_keys_pre=len(job.keyed_pairs),
                    n_keys_post=len(job.keyed_pairs),
                    recall_pre=1.0,  # assume pre-refresh was at threshold
                    recall_post=recall_rate,
                )
                # Restore from backup: copy backup weights back into the main tier.
                if backup_name in self.model.peft_config:
                    from paramem.models.loader import copy_adapter_weights

                    _sw(self.model, backup_name)
                    if tier not in self.model.peft_config:
                        self.model = create_adapter(self.model, self.episodic_config, tier)
                    copy_adapter_weights(self.model, src=backup_name, dst=tier)
                    logger.info(
                        "consolidate_interim_adapters: restored %s from %s after ceiling trip",
                        tier,
                        backup_name,
                    )
                return {
                    "tiers_rebuilt": tiers_rebuilt,
                    "graph_drift_count": graph_drift_count,
                    "keys_per_tier": {t: len(v) for t, v in tier_keyed.items()},
                    "recall_per_tier": recall_per_tier,
                    "rolled_back": True,
                    "rollback_tier": rollback_tier,
                }

            tiers_rebuilt.append(tier)

        # flip off _is_training at finalize entry (belt-and-braces).
        if trainer is not None:
            trainer._set_is_training(False)

        # --- Step 6: Atomic finalize ---
        # Invariant: registry rewrite FIRST, Router.reload() LAST.
        # The finalize block runs purely as registry / disk / PEFT / router ops.
        # No _train_adapter() call, no HF Trainer-driven routine may appear here.

        # 6a. Registry rewrite (MUST be first).
        if self.indexed_key_registry is not None:
            for tier, pairs in tier_keyed.items():
                for kp in pairs:
                    self.indexed_key_registry.set_adapter_id(kp["key"], tier)
            registry_path = self.output_dir / "indexed_key_registry.json"
            self.indexed_key_registry.save(registry_path)
            logger.info("consolidate_interim_adapters: registry rewritten to %s", registry_path)

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
        self.triples_since_last_full = 0

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
        keyed_pairs: list[dict],
        *,
        max_probe: int = 100,
    ) -> float:
        """Probe up to *max_probe* keyed pairs against *adapter_name* and return the recall rate.

        Shared by :meth:`consolidate_interim_adapters` (Step 7) and the
        cap-reached retrain path in :meth:`post_session_train`.  Keeping
        the logic in one place makes the sanity contract identical
        everywhere: same sample size, same probe harness, same failure
        semantics (probe exception → ``0.0`` so callers treat it as a
        rollback trigger rather than a mysterious skip).

        The caller is responsible for deciding what to do with the
        returned rate (threshold compare, rollback, health update).

        Args:
            adapter_name: Adapter to probe.  Must be loaded and switchable
                (caller holds the GPU lock).  The default ``"episodic"``
                in :func:`evaluate_indexed_recall` is deliberately NOT
                relied on — silently probing the wrong tier would mask
                tier-specific regressions.
            keyed_pairs: Candidate pairs to probe.  Sampled uniformly
                down to *max_probe* when longer.  An empty list returns
                ``1.0`` (nothing to prove → healthy by default).
            max_probe: Cap on probe size.  100 is chosen to keep the
                probe cheap enough to run inline even inside the
                post-session training path.

        Returns:
            Recall rate in ``[0.0, 1.0]``.  On probe-harness exception,
            returns ``0.0`` so the caller trips its sanity threshold.
        """
        if not keyed_pairs:
            return 1.0

        probe_pairs = keyed_pairs
        if len(probe_pairs) > max_probe:
            probe_pairs = random.sample(probe_pairs, max_probe)

        from paramem.training.indexed_memory import build_registry as _build_reg

        probe_registry = _build_reg(probe_pairs)

        try:
            from experiments.utils.test_harness import evaluate_indexed_recall

            self._disable_gradient_checkpointing()
            recall_result = evaluate_indexed_recall(
                self.model,
                self.tokenizer,
                probe_pairs,
                probe_registry,
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

        from paramem.models.loader import create_adapter

        has_peft = hasattr(self.model, "peft_config")
        if not has_peft or "episodic" not in self.model.peft_config:
            logger.info("Creating episodic adapter")
            self.model = create_adapter(self.model, self.episodic_config, "episodic")
        if self.config.promotion_threshold > 0 and "semantic" not in self.model.peft_config:
            logger.info("Creating semantic adapter")
            self.model = create_adapter(self.model, self.semantic_config, "semantic")
        if self.procedural_config is not None and "procedural" not in self.model.peft_config:
            logger.info("Creating procedural adapter")
            self.model = create_adapter(self.model, self.procedural_config, "procedural")

        # Staging adapter for on-the-fly training. Uses episodic config as
        # the template. All production adapters MUST share the same rank,
        # target_modules, and bias settings (the fields that determine LoRA
        # tensor shapes). alpha and dropout may differ per adapter since
        # they affect training behavior, not tensor shapes.
        _validate_staging_compat(self.episodic_config, self.semantic_config, self.procedural_config)
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
    loop = ConsolidationLoop(
        model=model,
        tokenizer=tokenizer,
        consolidation_config=consolidation_config,
        training_config=training_config,
        episodic_adapter_config=episodic_adapter_config,
        semantic_adapter_config=semantic_adapter_config,
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
