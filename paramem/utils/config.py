"""Configuration loading and validation — archived training-pipeline loader.

This module is the YAML loader for the archived Phase 1-4 research scripts
under ``archive/experiments/`` and their reference YAML at
``archive/configs/default.yaml``. All active runtime, server, test, and
example code was migrated off ``load_config`` during the default.yaml
retirement arc (2026-04-28); the lint guard
``tests/test_test_config_loader_usage.py`` enforces this.

The dataclasses defined below (``ParaMemConfig``, ``AdapterConfig``,
``TrainingConfig``, ``ConsolidationConfig``, etc.) remain importable by
active code as standalone types — many are reused directly by
``paramem.server.config``. Only ``load_config`` and the YAML population
path are archive-only.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ModelConfig:
    model_id: str = "Qwen/Qwen2.5-3B"
    quantization: str = "nf4"
    compute_dtype: str = "bfloat16"
    trust_remote_code: bool = True
    cpu_offload: bool = False
    max_memory_gpu: str = "7GiB"
    max_memory_cpu: str = "20GiB"


@dataclass
class AdapterConfig:
    rank: int = 8
    alpha: int = 16
    learning_rate: float = 1e-4
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    dropout: float = 0.05


@dataclass
class TrainingConfig:
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_seq_length: int = 512
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    warmup_steps: int = 0  # If > 0, overrides warmup_ratio
    lr_scheduler_type: str = "linear"  # HF default; use "constant" for grokking
    # When set, the LR scheduler decays over this many steps regardless of
    # num_train_epochs. Steps past it sit at the scheduler's tail value
    # (LR=0 for "linear"). Decouples decay shape from training budget so
    # early-stop on a recall metric dominates the run length without the
    # budget bleeding into the per-step LR trajectory.
    lr_decay_steps: int | None = None
    weight_decay: float = 0.01
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0
    seed: int = 42
    save_strategy: str = "epoch"
    save_steps: int = 0  # Steps between saves when save_strategy="steps"; 0 → HF default (500)
    save_total_limit: int = 2
    # HF Trainer logging cadence. Default 1 matches train_adapter's prior
    # hardcode. The BG-trainer call site overrides via dataclasses.replace
    # to keep its historical log volume (10 steps) when delegating to
    # train_adapter; other callers inherit the verbose default.
    logging_steps: int = 1
    # RAM-mode checkpointing: when > 0, train_adapter writes checkpoints to
    # /dev/shm instead of the caller's output_dir (avoids encrypted-disk IO
    # overhead on every save), then copies the latest checkpoint to
    # <output_dir>/bg_checkpoint_epoch/checkpoint-N/ at each epoch boundary.
    # Trade-off: /dev/shm is not durable across restarts; a crash loses the
    # in-flight checkpoint.  Set to 0 (default) to disable.
    save_steps_ram: int = 0
    early_stopping: bool = False
    early_stopping_threshold: float = 0.01
    early_stopping_floor: int = 10
    early_stopping_patience: int = 2
    # Recall-based early stopping (separate from loss-based above).
    # When True, ConsolidationLoop wires RecallEarlyStopCallback at every
    # production train_adapter call site (via _maybe_make_recall_callback).
    # Probes the staged adapter at epoch boundaries and fires
    # control.should_training_stop after `recall_window` consecutive
    # 100%-recall probes past `early_stopping_floor` (reused as
    # signal_from_epoch).  Validated at multi-seed for N=100 by Test
    # 14; untested at N=500+; ship default-OFF.
    recall_early_stopping: bool = False
    recall_window: int = 3
    # Probe cadence — system-wide.  3× cheaper than =1 at production
    # scale.  Smaller cycles still probe every `recall_probe_every_n_epochs`
    # epochs.  ``signal_from_epoch`` (=``early_stopping_floor`` above)
    # gates when the stop signal can fire AND when the first probe runs:
    # production wiring (consolidation.py::_maybe_make_recall_callback)
    # pins ``probe_from_epoch`` to the same floor so pre-floor probes —
    # which can never trigger a stop and whose log artifacts have no
    # production consumer — are not paid for.  Adjust the floor to lower
    # the earliest possible stop, not to "probe earlier".
    recall_probe_every_n_epochs: int = 3
    # Recall probe batch size — generate this many prompts per model.generate
    # call when probing the staged adapter at epoch boundaries.  Default 16
    # = validated production setting: ~4.75× faster than serial at ~346 MiB
    # peak VRAM delta on RTX 5070 8 GB, 137/137 recall parity vs serial,
    # multi-cycle retention parity confirmed in production conditions.
    # Empirical curve at adapter idle:
    #   b=1  baseline (serial),   ~125 MiB peak delta
    #   b=8   2.91× faster,        257 MiB peak delta
    #   b=16  4.75× faster,        346 MiB peak delta   ← production default
    #   b=32  6.02× faster,        574 MiB peak delta
    #   b=64  8.25× faster,       1032 MiB peak delta
    #   b=128 10.64× faster,      2073 MiB peak delta
    # In-training VRAM residual eats into headroom — drop to 1 only if a
    # specific deployment's training-residual pressure forces a downgrade.
    recall_probe_batch_size: int = 16


@dataclass
class TopicConfig:
    name: str = ""
    fact_ids: list[str] = field(default_factory=list)


@dataclass
class ReplayConfig:
    epochs_per_topic: int = 60
    gradient_accumulation_steps: int = 2
    naive_replay_ratio: float = 0.5
    generative_replay_ratio: float = 0.5
    generative_temperature: float = 0.3
    ewc_lambda: float = 400.0
    ewc_fisher_samples: Optional[int] = None
    topics: list[TopicConfig] = field(default_factory=list)


@dataclass
class GraphConfig:
    extraction_temperature: float = 0.3
    max_extraction_tokens: int = 1024
    entity_similarity_threshold: float = 85.0


@dataclass
class ConsolidationConfig:
    promotion_threshold: int = 3
    decay_window: int = 10
    procedural_detection_window: int = 5
    episodic_new_weight: float = 0.7
    semantic_replay_weight: float = 0.9
    curriculum_enabled: bool = False
    min_exposure_cycles: int = 5
    max_active_keys: int = 100000  # no practical limit
    key_retirement_threshold: float = 0.1
    key_retirement_cycles: int = 3
    indexed_key_replay_enabled: bool = False
    merge_at_interim: bool = False


@dataclass
class WandbConfig:
    project: str = "paramem"
    entity: str = ""
    enabled: bool = True


@dataclass
class PathsConfig:
    data_dir: str = "data"
    output_dir: str = "outputs"
    adapter_dir: str = "outputs/adapters"
    graph_dir: str = "outputs/graphs"


@dataclass
class ParaMemConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    adapters: dict[str, AdapterConfig] = field(default_factory=dict)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    consolidation: ConsolidationConfig = field(default_factory=ConsolidationConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)


def _build_dataclass(cls, data: dict):
    """Build a dataclass from a dict, ignoring unknown keys."""
    valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in valid_keys}
    return cls(**filtered)


def load_config(
    config_path: Optional[str | Path] = None,
) -> ParaMemConfig:
    """Load configuration from a YAML file, falling back to defaults.

    Archived loader: ``configs/default.yaml`` was retired during the
    default.yaml retirement arc (2026-04-28). The yaml lives at
    ``archive/configs/default.yaml`` alongside the archived Phase 1-4
    research scripts that depend on it. Active code does not call this
    function -- enforced by ``tests/test_test_config_loader_usage.py``.
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "archive" / "configs" / "default.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        return ParaMemConfig()

    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}

    model = _build_dataclass(ModelConfig, raw.get("model", {}))
    training = _build_dataclass(TrainingConfig, raw.get("training", {}))
    graph = _build_dataclass(GraphConfig, raw.get("graph", {}))
    consolidation = _build_dataclass(ConsolidationConfig, raw.get("consolidation", {}))
    wandb_cfg = _build_dataclass(WandbConfig, raw.get("wandb", {}))
    paths = _build_dataclass(PathsConfig, raw.get("paths", {}))

    adapters = {}
    for name, adapter_data in raw.get("adapters", {}).items():
        adapters[name] = _build_dataclass(AdapterConfig, adapter_data)

    replay_raw = raw.get("replay", {})
    topics_raw = replay_raw.pop("topics", [])
    replay = _build_dataclass(ReplayConfig, replay_raw)
    replay.topics = [_build_dataclass(TopicConfig, t) for t in topics_raw]

    return ParaMemConfig(
        model=model,
        adapters=adapters,
        training=training,
        replay=replay,
        graph=graph,
        consolidation=consolidation,
        wandb=wandb_cfg,
        paths=paths,
    )
