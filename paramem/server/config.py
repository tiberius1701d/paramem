"""Server configuration — loads server.yaml into typed dataclasses."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from paramem.utils.config import AdapterConfig, ModelConfig, TrainingConfig

# Duplicated from experiments/utils/test_harness.py to avoid modifying that file
# while benchmarks are running. TODO: refactor into shared paramem.models.registry
MODEL_REGISTRY = {
    "mistral": ModelConfig(
        model_id="mistralai/Mistral-7B-Instruct-v0.3",
        quantization="nf4",
        compute_dtype="bfloat16",
        trust_remote_code=True,
        cpu_offload=False,
    ),
    "gemma": ModelConfig(
        model_id="google/gemma-2-9b-it",
        quantization="nf4",
        compute_dtype="bfloat16",
        trust_remote_code=True,
        cpu_offload=True,
        max_memory_gpu="7GiB",
        max_memory_cpu="20GiB",
    ),
    "qwen3b": ModelConfig(
        model_id="Qwen/Qwen2.5-3B-Instruct",
        quantization="nf4",
        compute_dtype="bfloat16",
        trust_remote_code=True,
        cpu_offload=False,
    ),
}


@dataclass
class ServerNetConfig:
    host: str = "0.0.0.0"
    port: int = 8420


@dataclass
class CloudConfig:
    enabled: bool = False
    endpoint: str = ""
    model: str = ""
    api_key: str = ""


@dataclass
class VoiceConfig:
    system_prompt: str = (
        "You are a personal memory assistant. Answer concisely in 1-2 spoken sentences. "
        "Do not use markdown, lists, or structured formatting. "
        "Speak naturally as if you simply remember. "
        "If you cannot answer a question from your knowledge, respond ONLY with "
        "[ESCALATE] followed by the question to forward."
    )


@dataclass
class ConsolidationScheduleConfig:
    schedule: str = "02:00"


@dataclass
class ServerTrainingConfig:
    epochs: int = 30
    rank: int = 8
    alpha: int = 16
    learning_rate: float = 1e-4


@dataclass
class ServerConfig:
    server: ServerNetConfig = field(default_factory=ServerNetConfig)
    model_name: str = "mistral"
    adapter_dir: Path = Path("data/ha/adapters")
    registry_path: Path = Path("data/ha/registry.json")
    graph_path: Path = Path("data/ha/graph.json")
    session_dir: Path = Path("data/ha/sessions")
    training: ServerTrainingConfig = field(default_factory=ServerTrainingConfig)
    consolidation: ConsolidationScheduleConfig = field(default_factory=ConsolidationScheduleConfig)
    cloud: CloudConfig = field(default_factory=CloudConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)

    @property
    def model_config(self) -> ModelConfig:
        if self.model_name not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model '{self.model_name}'. Available: {list(MODEL_REGISTRY.keys())}"
            )
        return MODEL_REGISTRY[self.model_name]

    @property
    def adapter_config(self) -> AdapterConfig:
        return AdapterConfig(
            rank=self.training.rank,
            alpha=self.training.alpha,
            learning_rate=self.training.learning_rate,
        )

    @property
    def training_config(self) -> TrainingConfig:
        return TrainingConfig(num_epochs=self.training.epochs)


def load_server_config(path: str | Path = "configs/server.yaml") -> ServerConfig:
    """Load server configuration from YAML file."""
    path = Path(path)
    if not path.exists():
        return ServerConfig()

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    config = ServerConfig()
    config.server = ServerNetConfig(**raw.get("server", {}))
    config.model_name = raw.get("model", config.model_name)
    config.adapter_dir = Path(raw.get("adapter_dir", config.adapter_dir))
    config.registry_path = Path(raw.get("registry_path", config.registry_path))
    config.graph_path = Path(raw.get("graph_path", config.graph_path))
    config.session_dir = Path(raw.get("session_dir", config.session_dir))
    config.training = ServerTrainingConfig(**raw.get("training", {}))
    config.consolidation = ConsolidationScheduleConfig(**raw.get("consolidation", {}))
    config.cloud = CloudConfig(**raw.get("cloud", {}))

    voice_raw = raw.get("voice", {})
    if voice_raw:
        config.voice = VoiceConfig(**voice_raw)

    return config
