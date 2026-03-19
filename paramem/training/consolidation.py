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

from paramem.graph.extractor import extract_graph
from paramem.graph.merger import GraphMerger
from paramem.graph.qa_generator import generate_qa_from_relations
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
from paramem.training.trainer import train_adapter
from paramem.utils.config import (
    AdapterConfig,
    ConsolidationConfig,
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
        distillation_config=None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = consolidation_config
        self.training_config = training_config
        self.episodic_config = episodic_adapter_config
        self.semantic_config = semantic_adapter_config
        self.procedural_config = procedural_adapter_config
        self.wandb_config = wandb_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.extraction_temperature = extraction_temperature

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
        else:
            self.graph_path = self.output_dir / "cumulative_graph.json"

        # Load existing graph if present
        if self.graph_path.exists():
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
        self._indexed_next_index: int = 1
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
            # Load persisted SimHash registries
            ep_simhash_path = self.output_dir / "simhash_registry_episodic.json"
            sem_simhash_path = self.output_dir / "simhash_registry_semantic.json"
            if ep_simhash_path.exists():
                self.episodic_simhash = load_registry(ep_simhash_path)
            if sem_simhash_path.exists():
                self.semantic_simhash = load_registry(sem_simhash_path)

        self.cycle_count = 0

    def run_cycle(
        self,
        session_transcript: str,
        session_id: str,
    ) -> CycleResult:
        """Run one consolidation cycle for a new session.

        Args:
            session_transcript: The raw session transcript text.
            session_id: Unique identifier for this session.

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
                    )
            else:
                session_graph = extract_graph(
                    self.model,
                    self.tokenizer,
                    session_transcript,
                    session_id,
                    temperature=self.extraction_temperature,
                )

        result.entities_extracted = len(session_graph.entities)
        result.relations_extracted = len(session_graph.relations)

        # --- 2. MERGE ---
        self.merger.merge(session_graph)
        graph = self.merger.graph

        # --- 3. SCORE & CLASSIFY ---
        classification = self.scorer.classify_nodes(
            graph,
            promotion_threshold=self.config.promotion_threshold,
            decay_window=self.config.decay_window,
            current_cycle=self.cycle_count,
        )

        # Filter out already-promoted nodes
        new_promotions = [n for n in classification.promote if n not in self.promoted_nodes]
        result.nodes_promoted = len(new_promotions)
        result.nodes_decayed = len(classification.decay)
        result.nodes_retained = len(classification.retain)
        result.promoted_nodes = new_promotions
        result.decayed_nodes = classification.decay

        # --- 4. GENERATE QA PAIRS ---
        # Episodic: only relations from the *current session's* extraction
        # (replay pool provides continuity with past sessions)
        session_relations = [
            {
                "subject": r.subject,
                "predicate": r.predicate,
                "object": r.object,
            }
            for r in session_graph.relations
        ]
        qa_model = self.model
        qa_tokenizer = self.tokenizer
        if self.distillation_pipeline and self.distillation_pipeline.is_loaded():
            qa_model = self.distillation_pipeline.model
            qa_tokenizer = self.distillation_pipeline.tokenizer

        episodic_qa = generate_qa_from_relations(
            session_relations,
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
        self._apply_decay(classification.decay)

        # --- 8. SAVE ---
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
            }
        self._indexed_next_index += len(new_keyed)

        # Promote keys for promoted entities (move to semantic before episodic training)
        # Check both source_subject and source_object — reverse QA templates
        # swap subject/object, so a promoted entity may appear in either field.
        promoted_keys = []
        if new_promotions:
            promoted_set = {n.lower() for n in new_promotions}
            for key, qa_info in list(self.indexed_key_qa.items()):
                subject = qa_info.get("source_subject", "").lower()
                obj = qa_info.get("source_object", "").lower()
                mentions_promoted = (subject and subject in promoted_set) or (
                    obj and obj in promoted_set
                )
                if mentions_promoted and key not in self.semantic_simhash:
                    promoted_keys.append(key)

        # Reconstruct existing episodic keys from adapter weights
        active_keys = self.indexed_key_registry.list_active()
        # Exclude just-added keys (they haven't been trained yet) and promoted keys
        new_key_set = {kp["key"] for kp in new_keyed}
        promoted_key_set = set(promoted_keys)
        existing_keys = [
            k for k in active_keys if k not in new_key_set and k not in promoted_key_set
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
            if recalled is not None:
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

        # Cap at max_active_keys (evict oldest if over)
        if len(episodic_keyed) > self.config.max_active_keys:
            evicted = episodic_keyed[: len(episodic_keyed) - self.config.max_active_keys]
            episodic_keyed = episodic_keyed[len(evicted) :]
            for kp in evicted:
                self.indexed_key_registry.remove(kp["key"])
                self.indexed_key_qa.pop(kp["key"], None)
                self.episodic_simhash.pop(kp["key"], None)
            logger.info(
                "Evicted %d oldest keys (capacity cap %d)",
                len(evicted),
                self.config.max_active_keys,
            )

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
            output_dir=self.output_dir / f"cycle_{self.cycle_count}" / "episodic",
            run_name=f"phase4-indexed-episodic-cycle{self.cycle_count}",
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
            output_dir=self.output_dir / f"cycle_{self.cycle_count}" / "semantic",
            run_name=f"phase4-indexed-semantic-cycle{self.cycle_count}",
        )

        # Update semantic SimHash registry
        self.semantic_simhash = build_registry(semantic_keyed)

        return metrics.get("train_loss")

    def _check_indexed_key_fidelity(self, adapter_name: str) -> None:
        """Probe indexed keys and retire those with sustained low confidence."""
        switch_adapter(self.model, adapter_name)
        self._disable_gradient_checkpointing()

        registry = self.indexed_key_registry
        simhash = self.episodic_simhash if adapter_name == "episodic" else self.semantic_simhash

        active_keys = registry.list_active()
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
            output_dir=self.output_dir / f"cycle_{self.cycle_count}" / adapter_name,
            run_name=run_name,
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
        """Save both adapters and registries to disk."""
        cycle_dir = self.output_dir / f"cycle_{self.cycle_count}"
        save_adapter(self.model, cycle_dir / "episodic", "episodic")
        save_adapter(self.model, cycle_dir / "semantic", "semantic")
        if self.indexed_key_registry is not None:
            self.indexed_key_registry.save(self.output_dir / "indexed_key_registry.json")
            save_registry(self.episodic_simhash, self.output_dir / "simhash_registry_episodic.json")
            save_registry(self.semantic_simhash, self.output_dir / "simhash_registry_semantic.json")

    def _ensure_adapters(self):
        """Create adapters that don't exist yet.

        Episodic is always created. Semantic is created if promotion
        is enabled. Procedural is created if its config was provided.
        """
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
