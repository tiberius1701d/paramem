"""Server configuration — loads server.yaml into typed dataclasses."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from paramem.utils.config import (
    AdapterConfig,
    ConsolidationConfig,
    ModelConfig,
    TrainingConfig,
)

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

# Validated training parameters from test campaign (Tests 1-8).
# Single source of truth — do not override individually.
VALIDATED_TRAINING_CONFIG = TrainingConfig(
    batch_size=1,
    gradient_accumulation_steps=2,
    max_seq_length=1024,
    num_epochs=30,
    warmup_ratio=0.1,
    weight_decay=0.01,
    gradient_checkpointing=True,
    max_grad_norm=1.0,
    seed=42,
)


@dataclass
class ServerNetConfig:
    host: str = "0.0.0.0"
    port: int = 8420


@dataclass
class PathsConfig:
    """Mount points for persistent data, sessions, and debug output."""

    data: Path = Path("data/ha")
    sessions: Path = Path("data/ha/sessions")
    debug: Path = Path("data/ha/debug")

    @property
    def adapters(self) -> Path:
        return self.data / "adapters"

    @property
    def registry(self) -> Path:
        return self.data / "registry.json"

    @property
    def key_metadata(self) -> Path:
        return self.data / "key_metadata.json"


@dataclass
class CloudConfig:
    enabled: bool = False
    endpoint: str = ""
    model: str = ""
    api_key: str = field(default="", repr=False)


@dataclass
class VoiceConfig:
    prompt_file: str = "configs/prompts/ha_voice.txt"
    system_prompt: str = ""

    def load_prompt(self) -> str:
        """Load system prompt from file, falling back to inline default."""
        if self.prompt_file:
            path = Path(self.prompt_file)
            if path.exists():
                return path.read_text().strip()
        if self.system_prompt:
            return self.system_prompt
        return (
            "You are a personal memory assistant. Answer concisely in 1-2 spoken sentences. "
            "Speak naturally as if you simply remember."
        )


@dataclass
class ConsolidationScheduleConfig:
    schedule: str = "02:00"
    promotion_threshold: int = 3
    max_active_keys: int = 50
    retain_sessions: bool = True


@dataclass
class ServerAdapterConfig:
    enabled: bool = True
    rank: int = 8
    alpha: int = 16
    learning_rate: float = 1e-4


@dataclass
class ServerAdaptersConfig:
    episodic: ServerAdapterConfig = field(default_factory=ServerAdapterConfig)
    semantic: ServerAdapterConfig = field(
        default_factory=lambda: ServerAdapterConfig(
            enabled=True, rank=8, alpha=16, learning_rate=1e-5
        )
    )
    procedural: ServerAdapterConfig = field(
        default_factory=lambda: ServerAdapterConfig(
            enabled=False, rank=8, alpha=16, learning_rate=5e-5
        )
    )


@dataclass
class ServerConfig:
    server: ServerNetConfig = field(default_factory=ServerNetConfig)
    model_name: str = "mistral"
    debug: bool = True
    paths: PathsConfig = field(default_factory=PathsConfig)
    adapters: ServerAdaptersConfig = field(default_factory=ServerAdaptersConfig)
    consolidation: ConsolidationScheduleConfig = field(default_factory=ConsolidationScheduleConfig)
    cloud: CloudConfig = field(default_factory=CloudConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)

    # Derived path accessors for backward compatibility
    @property
    def adapter_dir(self) -> Path:
        return self.paths.adapters

    @property
    def registry_path(self) -> Path:
        return self.paths.registry

    @property
    def key_metadata_path(self) -> Path:
        return self.paths.key_metadata

    @property
    def session_dir(self) -> Path:
        return self.paths.sessions

    @property
    def debug_dir(self) -> Path:
        return self.paths.debug

    @property
    def model_config(self) -> ModelConfig:
        if self.model_name not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model '{self.model_name}'. Available: {list(MODEL_REGISTRY.keys())}"
            )
        return MODEL_REGISTRY[self.model_name]

    def _make_adapter_config(self, sac: ServerAdapterConfig) -> AdapterConfig:
        """Build an AdapterConfig with validated defaults (dropout=0.0)."""
        return AdapterConfig(
            rank=sac.rank,
            alpha=sac.alpha,
            learning_rate=sac.learning_rate,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            dropout=0.0,
        )

    @property
    def episodic_adapter_config(self) -> AdapterConfig:
        return self._make_adapter_config(self.adapters.episodic)

    @property
    def semantic_adapter_config(self) -> AdapterConfig:
        return self._make_adapter_config(self.adapters.semantic)

    @property
    def procedural_adapter_config(self) -> AdapterConfig:
        return self._make_adapter_config(self.adapters.procedural)

    @property
    def training_config(self) -> TrainingConfig:
        """Validated training config from test campaign."""
        return TrainingConfig(
            batch_size=VALIDATED_TRAINING_CONFIG.batch_size,
            gradient_accumulation_steps=VALIDATED_TRAINING_CONFIG.gradient_accumulation_steps,
            max_seq_length=VALIDATED_TRAINING_CONFIG.max_seq_length,
            num_epochs=VALIDATED_TRAINING_CONFIG.num_epochs,
            warmup_ratio=VALIDATED_TRAINING_CONFIG.warmup_ratio,
            weight_decay=VALIDATED_TRAINING_CONFIG.weight_decay,
            gradient_checkpointing=VALIDATED_TRAINING_CONFIG.gradient_checkpointing,
            max_grad_norm=VALIDATED_TRAINING_CONFIG.max_grad_norm,
            seed=VALIDATED_TRAINING_CONFIG.seed,
        )

    @property
    def consolidation_config(self) -> ConsolidationConfig:
        """Build ConsolidationConfig for ConsolidationLoop."""
        return ConsolidationConfig(
            promotion_threshold=self.consolidation.promotion_threshold,
            max_active_keys=self.consolidation.max_active_keys,
            indexed_key_replay_enabled=True,
            reconstruction_interval=5,
            decay_window=10,
        )


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
    config.debug = raw.get("debug", config.debug)

    # Paths
    paths_raw = raw.get("paths", {})
    if paths_raw:
        config.paths = PathsConfig(
            data=Path(paths_raw.get("data", config.paths.data)),
            sessions=Path(paths_raw.get("sessions", config.paths.sessions)),
            debug=Path(paths_raw.get("debug", config.paths.debug)),
        )

    # Adapters
    adapters_raw = raw.get("adapters", {})
    if adapters_raw:
        ep = adapters_raw.get("episodic", {})
        sem = adapters_raw.get("semantic", {})
        proc = adapters_raw.get("procedural", {})
        config.adapters = ServerAdaptersConfig(
            episodic=ServerAdapterConfig(**ep) if ep else ServerAdapterConfig(),
            semantic=ServerAdapterConfig(**sem)
            if sem
            else ServerAdapterConfig(rank=8, alpha=16, learning_rate=1e-5),
            procedural=ServerAdapterConfig(**proc)
            if proc
            else ServerAdapterConfig(rank=8, alpha=16, learning_rate=5e-5),
        )

    # Consolidation
    consolidation_raw = raw.get("consolidation", {})
    if consolidation_raw:
        config.consolidation = ConsolidationScheduleConfig(**consolidation_raw)

    config.cloud = CloudConfig(**raw.get("cloud", {}))

    voice_raw = raw.get("voice", {})
    if voice_raw:
        config.voice = VoiceConfig(**voice_raw)

    return config
