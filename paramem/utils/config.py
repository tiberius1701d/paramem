"""Configuration loading and validation."""

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
    weight_decay: float = 0.01
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0
    seed: int = 42


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
    max_active_keys: int = 50
    key_retirement_threshold: float = 0.1
    key_retirement_cycles: int = 3
    indexed_key_replay_enabled: bool = False
    reconstruction_interval: int = 5


@dataclass
class DistillationConfig:
    enabled: bool = False
    model_id: str = "google/gemma-2-9b-it"
    quantization: str = "nf4"
    compute_dtype: str = "bfloat16"
    trust_remote_code: bool = True
    max_memory_gpu: str = "7GiB"
    max_memory_cpu: str = "20GiB"
    cpu_offload: bool = True
    default_subject_name: str = "the user"
    temperature: float = 0.2
    max_new_tokens: int = 2048
    repetition_penalty: float = 1.3


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
class NeuroMemConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    adapters: dict[str, AdapterConfig] = field(default_factory=dict)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    consolidation: ConsolidationConfig = field(default_factory=ConsolidationConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)


def _build_dataclass(cls, data: dict):
    """Build a dataclass from a dict, ignoring unknown keys."""
    valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in valid_keys}
    return cls(**filtered)


def load_config(
    config_path: Optional[str | Path] = None,
) -> NeuroMemConfig:
    """Load configuration from a YAML file, falling back to defaults."""
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "configs" / "default.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        return NeuroMemConfig()

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

    distillation = _build_dataclass(
        DistillationConfig, raw.get("distillation", {}),
    )

    return NeuroMemConfig(
        model=model,
        adapters=adapters,
        training=training,
        replay=replay,
        graph=graph,
        consolidation=consolidation,
        distillation=distillation,
        wandb=wandb_cfg,
        paths=paths,
    )
