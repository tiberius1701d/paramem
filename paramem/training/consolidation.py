"""Consolidation loop orchestrator.

Runs the full consolidation pipeline: extract graph from session,
merge into cumulative graph, score for promotion/decay, train
episodic and semantic adapters.
"""

import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from torch.utils.data import Dataset

from paramem.graph.extractor import extract_graph, extract_procedural_graph
from paramem.graph.merger import GraphMerger
from paramem.graph.qa_generator import (
    filter_procedural_relations,
    generate_qa_from_relations,
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
    format_indexed_training,
    load_registry,
    probe_key,
    save_registry,
)
from paramem.training.key_registry import KeyRegistry
from paramem.training.replay import MixedReplayDataset, SyntheticQADataset
from paramem.training.trainer import GracefulShutdownCallback, train_adapter
from paramem.utils.config import (
    AdapterConfig,
    ConsolidationConfig,
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
        distillation_config=None,
        save_cycle_snapshots: bool = True,
        snapshot_dir: str | Path | None = None,
        persist_graph: bool = True,
        prompts_dir: str | Path | None = None,
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

        # Distillation pipeline (optional, for instruct-class model)
        self.distillation_pipeline = None
        if distillation_config and distillation_config.enabled:
            from paramem.graph.distiller import DistillationPipeline

            self.distillation_pipeline = DistillationPipeline(distillation_config)

        # Graph state
        self.merger = GraphMerger()
        self.scorer = PromotionScorer()
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

    def extract_session(
        self,
        session_transcript: str,
        session_id: str,
        speaker_id: str = "",
        speaker_name: str | None = None,
        ha_context: dict | None = None,
        stt_correction: bool = True,
        ha_validation: bool = True,
        noise_filter: str = "",
        noise_filter_model: str = "claude-sonnet-4-6",
        noise_filter_endpoint: str | None = None,
        ner_check: bool = False,
        ner_model: str = "en_core_web_sm",
        plausibility_judge: str = "auto",
        plausibility_stage: str = "deanon",
        verify_anonymization: bool = True,
    ) -> tuple[list[dict], list[dict]]:
        """Extract and generate QA pairs from a session without training.

        Returns (episodic_qa, procedural_relations) for deferred training.
        Merges the session graph into the cumulative graph.
        """
        logger.info("=== Extraction (session=%s) ===", session_id)

        # --- EXTRACT ---
        if self.distillation_pipeline:
            self.distillation_pipeline.load()
            session_graph = self.distillation_pipeline.extract_graph(
                session_transcript, session_id, speaker_name=speaker_name
            )
        else:
            self._disable_gradient_checkpointing()
            from peft import PeftModel as _PeftModel

            extraction_kwargs = dict(
                temperature=self.extraction_temperature,
                max_tokens=self.extraction_max_tokens,
                prompts_dir=self.prompts_dir,
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
            if isinstance(self.model, _PeftModel):
                with self.model.disable_adapter():
                    session_graph = extract_graph(
                        self.model,
                        self.tokenizer,
                        session_transcript,
                        session_id,
                        **extraction_kwargs,
                    )
            else:
                session_graph = extract_graph(
                    self.model,
                    self.tokenizer,
                    session_transcript,
                    session_id,
                    **extraction_kwargs,
                )

        logger.info(
            "Extracted %d entities, %d relations",
            len(session_graph.entities),
            len(session_graph.relations),
        )

        # --- MERGE ---
        self.merger.merge(session_graph)

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

        qa_model = self.model
        qa_tokenizer = self.tokenizer
        if self.distillation_pipeline and self.distillation_pipeline.is_loaded():
            qa_model = self.distillation_pipeline.model
            qa_tokenizer = self.distillation_pipeline.tokenizer

        episodic_qa = generate_qa_from_relations(
            session_relations, model=qa_model, tokenizer=qa_tokenizer
        )

        # Unload distillation model to free VRAM
        if self.distillation_pipeline and self.distillation_pipeline.is_loaded():
            self.distillation_pipeline.unload()

        # --- PROCEDURAL: separate extraction pass ---
        procedural_rels = []
        if self.procedural_config is not None:
            self._disable_gradient_checkpointing()
            from peft import PeftModel as _PM

            if isinstance(self.model, _PM):
                with self.model.disable_adapter():
                    proc_graph = extract_procedural_graph(
                        self.model,
                        self.tokenizer,
                        session_transcript,
                        session_id,
                        max_tokens=self.extraction_max_tokens,
                        prompts_dir=self.prompts_dir,
                        stt_correction=stt_correction,
                    )
            else:
                proc_graph = extract_procedural_graph(
                    self.model,
                    self.tokenizer,
                    session_transcript,
                    session_id,
                    max_tokens=self.extraction_max_tokens,
                    prompts_dir=self.prompts_dir,
                    stt_correction=stt_correction,
                )
            procedural_rels = [
                {
                    "subject": r.subject,
                    "predicate": r.predicate,
                    "object": r.object,
                    "relation_type": r.relation_type,
                }
                for r in proc_graph.relations
            ]

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
        qa_model = self.model
        qa_tokenizer = self.tokenizer
        if self.distillation_pipeline and self.distillation_pipeline.is_loaded():
            qa_model = self.distillation_pipeline.model
            qa_tokenizer = self.distillation_pipeline.tokenizer

        new_qa = generate_qa_from_relations(
            procedural_relations, model=qa_model, tokenizer=qa_tokenizer
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
        if self.distillation_pipeline:
            self.distillation_pipeline.load()
            session_graph = self.distillation_pipeline.extract_graph(
                session_transcript,
                session_id,
                speaker_name=speaker_name,
            )
        else:
            # Disable adapters for extraction (use base model reasoning)
            self._disable_gradient_checkpointing()
            from peft import PeftModel as _PeftModel

            if isinstance(self.model, _PeftModel):
                with self.model.disable_adapter():
                    session_graph = extract_graph(
                        self.model,
                        self.tokenizer,
                        session_transcript,
                        session_id,
                        temperature=self.extraction_temperature,
                        max_tokens=self.extraction_max_tokens,
                        prompts_dir=self.prompts_dir,
                        speaker_name=speaker_name,
                    )
            else:
                session_graph = extract_graph(
                    self.model,
                    self.tokenizer,
                    session_transcript,
                    session_id,
                    temperature=self.extraction_temperature,
                    max_tokens=self.extraction_max_tokens,
                    prompts_dir=self.prompts_dir,
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
        qa_model = self.model
        qa_tokenizer = self.tokenizer
        if self.distillation_pipeline and self.distillation_pipeline.is_loaded():
            qa_model = self.distillation_pipeline.model
            qa_tokenizer = self.distillation_pipeline.tokenizer

        # When procedural adapter is enabled, exclude preference relations
        # from episodic — they route to procedural instead
        procedural_rels = []
        if self.procedural_config is not None:
            procedural_rels = filter_procedural_relations(session_relations)
            procedural_set = set(id(r) for r in procedural_rels)
            episodic_relations = [r for r in session_relations if id(r) not in procedural_set]
        else:
            episodic_relations = session_relations

        episodic_qa = generate_qa_from_relations(
            episodic_relations,
            model=qa_model,
            tokenizer=qa_tokenizer,
        )

        # Promotion: relations for promoted entities (cap to keep training bounded)
        if new_promotions:
            promote_relations = get_relations_for_nodes(graph, new_promotions)
            if len(promote_relations) > 20:
                promote_relations = random.sample(promote_relations, 20)
            promote_qa = generate_qa_from_relations(
                promote_relations,
                model=qa_model,
                tokenizer=qa_tokenizer,
            )
        else:
            promote_qa = []

        # Unload distillation model before training to free VRAM
        if self.distillation_pipeline and self.distillation_pipeline.is_loaded():
            self.distillation_pipeline.unload()

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

        # Periodic fidelity check and retirement
        should_check = (
            self.config.reconstruction_interval > 0
            and self.cycle_count % self.config.reconstruction_interval == 0
        )
        if should_check:
            self._check_indexed_key_fidelity("episodic")

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
        filter_procedural_relations before passing).

        Same mechanism as episodic: reconstruct existing keys from adapter
        weights, add new keys, retrain the full set. Knowledge lives in
        the weights.

        Contradiction handling: when a new preference shares the same
        (speaker_id, subject, predicate) as an existing key, the old key
        is retired and replaced.
        """
        if not procedural_relations:
            return None

        logger.info(
            "Found %d procedural relations",
            len(procedural_relations),
        )

        # Generate QA pairs from preference relations
        qa_model = self.model
        qa_tokenizer = self.tokenizer
        if self.distillation_pipeline and self.distillation_pipeline.is_loaded():
            qa_model = self.distillation_pipeline.model
            qa_tokenizer = self.distillation_pipeline.tokenizer

        new_qa = generate_qa_from_relations(
            procedural_relations, model=qa_model, tokenizer=qa_tokenizer
        )
        if not new_qa:
            return None

        # Assign keys with proc prefix — use per-relation speaker_id
        new_keyed = []
        for i, qa in enumerate(new_qa):
            key = f"proc{self._procedural_next_index}"
            self._procedural_next_index += 1
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

        # Contradiction check: retire old keys with same (speaker, subject, predicate)
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
                self.procedural_simhash.pop(old_key, None)
                self.indexed_key_qa.pop(old_key, None)
                if self.indexed_key_registry is not None:
                    self.indexed_key_registry.remove(old_key)

        # Store new QA pairs in shared indexed_key_qa and update indexes
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

        # Reconstruct existing procedural keys from adapter weights
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
            "Procedural key reconstruction: %d/%d existing keys recovered",
            len(reconstructed),
            len(existing_keys),
        )

        # Build full procedural training set
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

        # Update procedural SimHash registry
        self.procedural_simhash = build_registry(all_procedural)

        return metrics.get("train_loss")

    def _check_indexed_key_fidelity(self, adapter_name: str) -> None:
        """Probe indexed keys and retire those with sustained low confidence."""
        switch_adapter(self.model, adapter_name)
        self._disable_gradient_checkpointing()

        registry = self.indexed_key_registry
        simhash_map = {
            "episodic": self.episodic_simhash,
            "semantic": self.semantic_simhash,
            "procedural": self.procedural_simhash,
        }
        simhash = simhash_map.get(adapter_name, self.episodic_simhash)

        all_active = registry.list_active()
        # Only check keys that belong to this adapter
        prefix = "proc" if adapter_name == "procedural" else "graph"
        active_keys = [k for k in all_active if k.startswith(prefix)]
        retired = []
        for key in active_keys:
            recalled = probe_key(
                self.model,
                self.tokenizer,
                key,
                registry=simhash,
                confidence_threshold=0.0,
            )
            confidence = recalled["confidence"] if recalled else 0.0
            registry.update_fidelity(key, confidence)

            if registry.should_retire(
                key,
                threshold=self.config.key_retirement_threshold,
                consecutive_cycles=self.config.key_retirement_cycles,
            ):
                registry.remove(key)
                self.indexed_key_qa.pop(key, None)
                simhash.pop(key, None)
                retired.append(key)
                logger.info("Retired indexed key '%s' (sustained low fidelity)", key)

        if retired:
            logger.info("Retired %d indexed keys from %s adapter", len(retired), adapter_name)

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

    def _training_output_dir(self, adapter_name: str) -> Path:
        """Training checkpoint directory for the current cycle.

        When cycle snapshots are enabled, checkpoints go under output_dir.
        When disabled, they go under snapshot_dir (debug) or a temp subdir
        to keep the production adapter directory clean.
        """
        if self.save_cycle_snapshots and self.snapshot_dir:
            return self.snapshot_dir / f"cycle_{self.cycle_count}" / adapter_name
        return self.output_dir / f"cycle_{self.cycle_count}" / adapter_name

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

        # Clean stale staging checkpoints on disk — in_training is never
        # authoritative on disk; only production adapters are.
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
    extraction_temperature: float = 0.3,
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
