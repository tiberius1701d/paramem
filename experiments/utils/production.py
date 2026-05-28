"""Curated re-export façade for production symbols used by experiments.

This module is the **only** place experiments should import the load-bearing
``paramem.*`` building blocks from.  It is a *flat re-export manifest* — no
wrapping, no behavior change, no defaulting.  The point is that when a
production module moves, gets renamed, or has its signature tightened, the
break surfaces here once instead of in every experiment script.

Rules
-----
* Re-exports must be public (no leading underscore).  Private symbols are
  off-limits to experiments by design — see
  :mod:`tests.test_experiment_boundary` for the enforcement.
* No defaulting, no parameter rewriting.  ``from experiments.utils.production
  import X`` must be byte-equivalent to ``from <real-path> import X``.
* When the production API changes shape, fix it *here* (and let the
  experiments break loudly).  Do not silently translate.

What this is NOT
----------------
* Not a wrapper.  No sidecar logic, no compatibility shim.
* Not a replacement for ``experiments/utils/test_harness.py`` — that file
  owns the experiment-only conveniences (``BENCHMARK_MODELS``,
  ``model_output_dir``, ``add_model_args``).  This file owns the boundary.

Migration is voluntary
----------------------
Existing experiments that import directly from ``paramem.*`` keep working.
New experiments should prefer this façade.  The structural guard only blocks
*private* (underscored) cross-boundary imports.
"""

from __future__ import annotations

# --- adapters -------------------------------------------------------------
from paramem.adapters import resolve_adapter_slot
from paramem.adapters.manifest import build_manifest_for

# --- backup ---------------------------------------------------------------
from paramem.backup.age_envelope import is_age_envelope
from paramem.backup.checkpoint_shard import materialize_checkpoint_to_shm

# --- evaluation -----------------------------------------------------------
from paramem.evaluation.embedding_scorer import compute_similarity
from paramem.evaluation.recall import generate_answer

# --- graph ----------------------------------------------------------------
from paramem.graph.extraction_pipeline import ExtractionConfig, ExtractionPipeline
from paramem.graph.merger import GraphMerger
from paramem.graph.qa_generator import generate_qa_from_relations
from paramem.graph.schema import Entity, Relation, SessionGraph

# --- memory ---------------------------------------------------------------
from paramem.memory.entry import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    RECALL_TEMPLATE,
    assign_keys,
    build_registry,
    format_entry_training,
)
from paramem.memory.persistence import load_registry, save_registry

# --- models ---------------------------------------------------------------
from paramem.models.loader import (
    adapt_messages,
    create_adapter,
    load_adapter,
    load_base_model,
    save_adapter,
    switch_adapter,
    unload_model,
)

# --- server / orchestration ----------------------------------------------
from paramem.server.config import ServerConfig, load_server_config
from paramem.server.consolidation import create_consolidation_loop
from paramem.server.gpu_lock import gpu_lock_sync

# --- training -------------------------------------------------------------
from paramem.training.consolidation import ConsolidationLoop
from paramem.training.early_stop import EarlyStopPolicy
from paramem.training.recall_eval import evaluate_indexed_recall, probe_entries
from paramem.training.thermal_throttle import ThermalPolicy, ThermalThrottleCallback
from paramem.training.trainer import TrainingHooks, train_adapter

# --- config ---------------------------------------------------------------
from paramem.utils.config import AdapterConfig, ModelConfig, TrainingConfig

__all__ = [
    # adapters
    "resolve_adapter_slot",
    "build_manifest_for",
    # backup
    "is_age_envelope",
    "materialize_checkpoint_to_shm",
    # evaluation
    "compute_similarity",
    "generate_answer",
    # graph
    "ExtractionConfig",
    "ExtractionPipeline",
    "GraphMerger",
    "generate_qa_from_relations",
    "Entity",
    "Relation",
    "SessionGraph",
    # memory
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "RECALL_TEMPLATE",
    "assign_keys",
    "build_registry",
    "format_entry_training",
    "load_registry",
    "save_registry",
    # models
    "adapt_messages",
    "create_adapter",
    "load_adapter",
    "load_base_model",
    "save_adapter",
    "switch_adapter",
    "unload_model",
    # server
    "ServerConfig",
    "load_server_config",
    "create_consolidation_loop",
    "gpu_lock_sync",
    # training
    "ConsolidationLoop",
    "EarlyStopPolicy",
    "evaluate_indexed_recall",
    "probe_entries",
    "ThermalPolicy",
    "ThermalThrottleCallback",
    "TrainingHooks",
    "train_adapter",
    # config
    "AdapterConfig",
    "ModelConfig",
    "TrainingConfig",
]
