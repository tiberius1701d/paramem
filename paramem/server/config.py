"""Server configuration — loads server.yaml into typed dataclasses."""

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from paramem.utils.config import (
    AdapterConfig,
    ConsolidationConfig,
    GraphConfig,
    ModelConfig,
    TrainingConfig,
)

logger = logging.getLogger(__name__)

# Pattern for ${VAR_NAME} env var references in YAML values
_ENV_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _interpolate_env_vars(value):
    """Recursively replace ${VAR_NAME} with os.environ values.

    Warns if a secret-like field contains a literal value that looks like
    an API key (long alphanumeric string) instead of an env var reference.
    """
    if isinstance(value, str):

        def _replace(match):
            var_name = match.group(1)
            env_val = os.environ.get(var_name)
            if env_val is None:
                logger.warning("Env var %s not set, using empty string", var_name)
                return ""
            return env_val

        return _ENV_VAR_PATTERN.sub(_replace, value)
    if isinstance(value, dict):
        return {k: _interpolate_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate_env_vars(item) for item in value]
    return value


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
    "qwen": ModelConfig(
        model_id="Qwen/Qwen2.5-7B-Instruct",
        quantization="nf4",
        compute_dtype="bfloat16",
        trust_remote_code=True,
        cpu_offload=False,
    ),
    "ministral": ModelConfig(
        model_id="mistralai/Ministral-8B-Instruct-2410",
        quantization="nf4",
        compute_dtype="bfloat16",
        trust_remote_code=True,
        cpu_offload=False,
    ),
    "llama": ModelConfig(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        quantization="nf4",
        compute_dtype="bfloat16",
        trust_remote_code=True,
        cpu_offload=False,
    ),
    "gemma4": ModelConfig(
        model_id="principled-intelligence/gemma-4-E4B-it-text-only",
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
    reclaim_interval_minutes: int = 10  # auto-reclaim GPU check interval
    vram_safety_margin_mb: int = 200  # free VRAM to keep after loading all GPU components


@dataclass
class PathsConfig:
    """Mount points for persistent data, sessions, and debug output."""

    data: Path = Path("data/ha")
    sessions: Path = Path("data/ha/sessions")
    debug: Path = Path("data/ha/debug")
    prompts: Path = Path("configs/prompts")

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
class CloudAgentConfig:
    """Configuration for a cloud (SOTA) agent."""

    enabled: bool = False
    provider: str = "openai"  # openai, anthropic, google, groq
    model: str = ""
    api_key: str = field(default="", repr=False)
    endpoint: str = ""  # optional custom endpoint (for Groq, ollama, etc.)


@dataclass
class HAToolsConfig:
    """Configuration for HA-proxied tool execution."""

    url: str = ""
    token: str = field(default="", repr=False)
    auto_discover: bool = False
    allowlist: list[str] = field(default_factory=list)
    sensitive_override: bool = False
    supported_languages: list[str] = field(default_factory=list)  # HA conversation agent langs


@dataclass
class ToolsConfig:
    """Tool use configuration."""

    ha: HAToolsConfig = field(default_factory=HAToolsConfig)
    tool_timeout_seconds: float = 3.0


@dataclass
class SanitizationConfig:
    """PII sanitization for cloud-bound queries."""

    mode: str = "block"  # off, warn, block

    def __post_init__(self):
        valid = {"off", "warn", "block"}
        if self.mode not in valid:
            raise ValueError(f"Invalid sanitization mode '{self.mode}'. Must be one of: {valid}")


@dataclass
class VoiceConfig:
    prompt_file: str = "configs/prompts/ha_voice.txt"
    system_prompt: str = ""
    greeting_interval_hours: int = 24  # hours between greetings per speaker (0 = disabled)

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
    # The interim refresh cadence — the only scheduling knob the operator sets.
    # Every refresh_cadence a new episodic_interim_<stamp> adapter is minted
    # (subject to the activity gate). Full consolidation fires every
    # refresh_cadence × max_interim_count (derived; see
    # consolidation_period_seconds / consolidation_period_string properties).
    # Grammar: "every Nh" / "every Nm" / "HH:MM" (daily) / "daily" / "" (manual).
    refresh_cadence: str = (
        "12h"  # default: one new interim every 12h → 84h full consolidation at count=7
    )
    mode: str = "train"  # "train" = full pipeline, "simulate" = extract only
    promotion_threshold: int = 3
    retain_sessions: bool = True
    indexed_key_replay: bool = True  # indexed key training mechanism
    decay_window: int = 10  # cycles before unreinforced keys decay
    extraction_max_tokens: int = 2048  # max output tokens for graph extraction
    extraction_stt_correction: bool = True  # correct STT errors from assistant responses
    extraction_ha_validation: bool = True  # validate locations against HA home context
    extraction_noise_filter: str = "anthropic"  # SOTA provider for noise filtering ("" = disabled)
    extraction_noise_filter_model: str = "claude-sonnet-4-6"  # model for noise filtering
    extraction_noise_filter_endpoint: str = ""  # custom endpoint for OpenAI-compatible providers
    training_temp_limit: int = 0  # GPU temp ceiling for background training (0 = disabled)
    training_temp_check_interval: int = 5  # check temp every N training steps
    # --- Extraction-pipeline stages (all configurable) ---
    #
    # Plausibility filter: final quality gate on extracted facts.
    #   "auto" → local judge at deanon stage (zero cloud cost, privacy-safe)
    #   "off"  → disable plausibility entirely (not recommended)
    #   SOTA provider name (e.g. "claude") → cloud judge at anon stage (paid,
    #     runs on anonymized data only). IMPORTANT: use stage="anon" with any
    #     cloud provider — stage="deanon" with a cloud provider sends real names
    #     to the cloud and is rejected at server start.
    extraction_plausibility_judge: str = "auto"
    extraction_plausibility_stage: str = "deanon"  # "anon" | "deanon"
    extraction_verify_anonymization: bool = True  # forward-path privacy guard
    extraction_ner_check: bool = False  # optional spaCy PII cross-check
    extraction_ner_model: str = "en_core_web_sm"  # spaCy model when ner_check=True
    # Maximum number of concurrent interim adapters. Caps the rolling
    # episodic_interim_YYYYMMDDTHHMM adapters resident in VRAM at any one time.
    # Must be proven to fit by the startup VRAM validator before the server starts.
    max_interim_count: int = 7
    # Enable post-session (inline) training after each conversation turn.
    # When True, a background job extracts facts from the session transcript and
    # trains them onto the current interim adapter immediately after the
    # assistant response is returned.  When False (default) post-session training
    # is suppressed; facts accumulate only via the scheduled full-consolidation
    # cycle.  Setting this True does NOT require changing ``mode`` — it is an
    # independent gate so operators can enable inline training while keeping the
    # full cycle in any mode.
    post_session_train_enabled: bool = False
    # entity_similarity_threshold mirrors GraphConfig (paramem/utils/config.py);
    # bridged into GraphMerger construction via ServerConfig.graph_config.
    entity_similarity_threshold: float = 85.0
    # --- Graph-level SOTA enrichment (Task #10) ---
    # Runs at full consolidation over the cumulative graph to capture
    # cross-transcript second-order relations that per-transcript enrichment
    # cannot see. Reuses the noise-filter SOTA credentials.
    graph_enrichment_enabled: bool = True
    graph_enrichment_neighborhood_hops: int = 2
    graph_enrichment_max_entities_per_pass: int = 50

    def __post_init__(self) -> None:
        """Validate privacy-critical config combinations at construction time.

        Raises ValueError if a cloud SOTA provider is configured as the
        plausibility judge AND the stage is set to "deanon". That combination
        would send real (de-anonymized) names to a cloud API, violating the
        privacy guarantee. Yaml comments alone are insufficient enforcement.

        Safe combinations:
        - judge="auto"  + any stage  → local model, no cloud exposure
        - judge="off"   + any stage  → plausibility disabled, no cloud exposure
        - judge=<cloud> + stage="anon" → cloud judge on anonymized data only
        """
        judge = self.extraction_plausibility_judge
        stage = self.extraction_plausibility_stage
        # "auto" and "off" are always safe (local or disabled). Any other value
        # is treated as a cloud provider name.
        if judge not in ("auto", "off") and stage == "deanon":
            raise ValueError(
                f"Privacy violation: extraction_plausibility_judge={judge!r} + "
                f"extraction_plausibility_stage='deanon' would send REAL NAMES to a "
                f"cloud API. Use stage='anon' for cloud judges, or judge='auto' for "
                f"local judging."
            )

    @property
    def consolidation_period_seconds(self) -> int | None:
        """Full consolidation period in seconds — derived, not configured.

        Returns ``refresh_cadence × max_interim_count`` seconds, or ``None``
        when ``refresh_cadence`` is disabled (``""``/``"off"``/etc.). Used by
        the systemd timer to schedule the full-consolidation cycle and by
        ``pstatus`` to display the effective cadence.
        """
        from paramem.server.interim_adapter import compute_schedule_period_seconds

        refresh_seconds = compute_schedule_period_seconds(self.refresh_cadence)
        if refresh_seconds is None:
            return None
        return refresh_seconds * self.max_interim_count

    @property
    def consolidation_period_string(self) -> str:
        """Human-readable form of :attr:`consolidation_period_seconds`.

        Produces an ``"every Nh"`` / ``"every Nm"`` string compatible with
        :func:`paramem.server.systemd_timer.parse_schedule`. Returns ``""``
        (manual-only) when ``refresh_cadence`` is disabled.
        """
        total = self.consolidation_period_seconds
        if total is None:
            return ""
        if total % 3600 == 0:
            return f"every {total // 3600}h"
        return f"every {total // 60}m"


@dataclass
class SpeakerConfig:
    """Voice-based speaker identification settings."""

    enabled: bool = False
    high_confidence_threshold: float = 0.60
    low_confidence_threshold: float = 0.45
    store_path: str = ""  # empty = default (data/ha/speaker_profiles.json)
    enrollment_prompt: str = "By the way, I don't think we've met yet. Please introduce yourself."
    enrollment_idle_timeout: int = 120  # seconds of /chat silence before LLM name extraction
    enrollment_reprompt_interval: int = 600  # seconds between re-prompting unknown speakers
    enrollment_check_interval: int = 15  # seconds between idle-loop checks
    # Discard embeddings from utterances shorter than this; pyannote needs sustained voice.
    min_embedding_duration_seconds: float = 1.0
    max_embeddings_per_profile: int = 50  # cap on stored embeddings per speaker
    redundancy_threshold: float = 0.95  # skip add_embedding if similarity to centroid exceeds this
    grouping_threshold_factor: float = 0.6  # unknown grouping = low_threshold * this factor


@dataclass
class STTConfig:
    """Local speech-to-text via Faster Whisper."""

    enabled: bool = False
    model: str = "small"  # tiny, base, small, medium, large-v3, distil-large-v3
    cpu_fallback_model: str = "distil-small.en"  # smaller model for CPU when GPU unavailable
    device: str = "cuda"  # cuda, cpu, auto
    compute_type: str = "int8"  # int8, float16, float32
    port: int = 10300  # Wyoming STT listener port
    language: str = "auto"  # "auto" for multilingual detection, or fixed code
    beam_size: int = 5  # Whisper beam search width (higher = better quality, slower)
    vad_filter: bool = True  # voice activity detection (trims silence, may clip short commands)


@dataclass
class TTSVoiceConfig:
    """Configuration for a single TTS voice."""

    engine: str = "piper"  # "piper" or "mms"
    model: str = ""  # Piper model name or HuggingFace model ID
    device: str = ""  # "" = inherit from TTSConfig.device
    language_name: str = ""  # display name for LLM prompt (e.g. "German"); "" = auto from ISO


# Standard ISO 639-1 language names — fallback when language_name is not set in config.
ISO_LANGUAGE_NAMES: dict[str, str] = {
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "nl": "Dutch",
    "pt": "Portuguese",
    "tl": "Tagalog",
    "zh": "Chinese",
}


@dataclass
class TTSConfig:
    """Local text-to-speech via Wyoming protocol."""

    enabled: bool = True
    port: int = 10301  # Wyoming TTS listener port
    device: str = "cuda"  # default device for all voices; per-voice override possible
    default_language: str = "en"
    language_confidence_threshold: float = 0.8  # minimum Whisper probability to trust detection
    model_dir: str = ""  # directory for TTS model files; "" = paths.data / "tts"
    audio_chunk_bytes: int = 4096  # bytes per Wyoming audio chunk (tradeoff: latency vs overhead)
    voices: dict[str, TTSVoiceConfig] = field(default_factory=dict)

    def language_name(self, code: str) -> str:
        """Resolve display name for a language code.

        Priority: voice config language_name > ISO standard > raw code.
        """
        voice = self.voices.get(code)
        if voice and voice.language_name:
            return voice.language_name
        return ISO_LANGUAGE_NAMES.get(code, code)


_ATTENTION_TARGETS = ["q_proj", "v_proj", "k_proj", "o_proj"]
_ATTENTION_PLUS_MLP_TARGETS = [
    "q_proj",
    "v_proj",
    "k_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


@dataclass
class ServerAdapterConfig:
    enabled: bool = True
    rank: int = 8
    alpha: int = 16
    learning_rate: float = 1e-4
    # Default: attention-only. Episodic + semantic adapters route via attention
    # (indexed-key retrieval is routing, not representation). Procedural
    # overrides to include MLP layers for representational imprinting of
    # persistent preferences/habits — see `ServerAdaptersConfig.procedural`.
    target_modules: list[str] = field(default_factory=lambda: list(_ATTENTION_TARGETS))


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
            enabled=False,
            rank=8,
            alpha=16,
            learning_rate=5e-5,
            # MLP targeting: behavioral patterns (style, formatting, prefs) live
            # in the model's feed-forward layers; attention-only would limit
            # procedural learning to routing, not representational change.
            target_modules=list(_ATTENTION_PLUS_MLP_TARGETS),
        )
    )


@dataclass
class ServerConfig:
    server: ServerNetConfig = field(default_factory=ServerNetConfig)
    model_name: str = "mistral"
    debug: bool = True
    # cloud_only: route every query to the configured SOTA agent instead of the
    # local parametric-memory model.  Routing only — extraction and consolidation
    # continue to run.  Combined with the --cloud-only CLI flag via OR: either
    # source being True enables cloud-only mode.  See configs/server.yaml for the
    # full privacy and quality implications.
    cloud_only: bool = False
    # headless_boot: ensure the server auto-starts before any interactive login.
    # True  → systemd user linger enabled for the current user AND (WSL only) a
    #         Windows scheduled task at system startup that launches the WSL VM.
    # False → linger disabled, Windows task removed (server starts on first login).
    # Reconciled on every startup by scripts/setup/headless-boot.sh (non-fatal,
    # logs WARN if elevation is unavailable and drift exists).
    headless_boot: bool = False
    snapshot_key: str = ""  # Fernet key for encrypted session snapshots; auto-generated if empty
    paths: PathsConfig = field(default_factory=PathsConfig)
    adapters: ServerAdaptersConfig = field(default_factory=ServerAdaptersConfig)
    consolidation: ConsolidationScheduleConfig = field(default_factory=ConsolidationScheduleConfig)
    sota_agent: CloudAgentConfig = field(default_factory=CloudAgentConfig)
    sota_providers: dict[str, CloudAgentConfig] = field(default_factory=dict)
    ha_agent_id: str = "conversation.groq"  # HA conversation agent for escalation
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    sanitization: SanitizationConfig = field(default_factory=SanitizationConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    speaker: SpeakerConfig = field(default_factory=SpeakerConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)

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
    def prompts_dir(self) -> Path:
        return self.paths.prompts

    @property
    def model_config(self) -> ModelConfig:
        if self.model_name not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model '{self.model_name}'. Available: {list(MODEL_REGISTRY.keys())}"
            )
        return MODEL_REGISTRY[self.model_name]

    def _make_adapter_config(self, sac: ServerAdapterConfig) -> AdapterConfig:
        """Build an AdapterConfig with validated defaults (dropout=0.0).

        Per-adapter `target_modules` are honoured — procedural targets MLP in
        addition to attention, episodic/semantic target attention only.
        """
        return AdapterConfig(
            rank=sac.rank,
            alpha=sac.alpha,
            learning_rate=sac.learning_rate,
            target_modules=list(sac.target_modules),
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
            indexed_key_replay_enabled=self.consolidation.indexed_key_replay,
            decay_window=self.consolidation.decay_window,
        )

    @property
    def graph_config(self) -> GraphConfig:
        """Build GraphConfig for GraphMerger construction.

        Mirrors ``entity_similarity_threshold`` from ``self.consolidation``.
        """
        return GraphConfig(
            entity_similarity_threshold=self.consolidation.entity_similarity_threshold,
        )


def load_server_config(path: str | Path = "configs/server.yaml") -> ServerConfig:
    """Load server configuration from YAML file.

    Supports ${VAR_NAME} env var interpolation in all string values.
    Accepts both new 'agents:' key and deprecated 'cloud:' key.
    """
    path = Path(path)
    if not path.exists():
        return ServerConfig()

    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    # Interpolate env vars in all string values
    raw = _interpolate_env_vars(raw)

    config = ServerConfig()
    config.server = ServerNetConfig(**raw.get("server", {}))
    config.model_name = raw.get("model", config.model_name)
    config.debug = raw.get("debug", config.debug)
    config.cloud_only = bool(raw.get("cloud_only", config.cloud_only))
    config.headless_boot = bool(raw.get("headless_boot", config.headless_boot))
    config.snapshot_key = raw.get("snapshot_key", config.snapshot_key)

    # Paths — resolve relative paths against config file directory so they
    # work regardless of the process's working directory at runtime.
    config_dir = Path(path).resolve().parent.parent  # configs/ → project root
    paths_raw = raw.get("paths", {})
    if paths_raw:
        config.paths = PathsConfig(
            data=Path(paths_raw.get("data", config.paths.data)),
            sessions=Path(paths_raw.get("sessions", config.paths.sessions)),
            debug=Path(paths_raw.get("debug", config.paths.debug)),
            prompts=Path(paths_raw.get("prompts", config.paths.prompts)),
        )
    # Make relative paths absolute (anchored to project root)
    for path_field in ("data", "sessions", "debug", "prompts"):
        p = getattr(config.paths, path_field)
        if not p.is_absolute():
            setattr(config.paths, path_field, config_dir / p)

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

    # Consolidation — refresh_cadence is the single user-facing scheduling knob.
    # Legacy configs carry `schedule` (the old full-period setting); translate
    # with a WARNING so the operator can update their yaml.
    consolidation_raw = raw.get("consolidation", {})
    legacy_schedule = consolidation_raw.pop("schedule", None)
    if legacy_schedule is not None:
        if "refresh_cadence" in consolidation_raw:
            logger.warning(
                "consolidation.schedule=%r and consolidation.refresh_cadence=%r both "
                "present in yaml — using refresh_cadence, ignoring legacy schedule. "
                "Remove `schedule` from your config.",
                legacy_schedule,
                consolidation_raw["refresh_cadence"],
            )
        else:
            consolidation_raw["refresh_cadence"] = legacy_schedule
            logger.warning(
                "consolidation.schedule=%r is deprecated — rename to "
                "consolidation.refresh_cadence. Effective full consolidation period "
                "is now refresh_cadence × max_interim_count.",
                legacy_schedule,
            )
    if consolidation_raw:
        config.consolidation = ConsolidationScheduleConfig(**consolidation_raw)

    agents_raw = raw.get("agents", {})
    config.ha_agent_id = agents_raw.get("ha_agent_id", "conversation.groq")

    # SOTA agent — high-capability model for reasoning queries
    sota_raw = agents_raw.get("sota", {})
    if sota_raw:
        config.sota_agent = CloudAgentConfig(**sota_raw)

    # Additional SOTA providers for direct routing (sota:anthropic, sota:openai, etc.)
    sota_providers_raw = agents_raw.get("sota_providers", {})
    for name, provider_raw in sota_providers_raw.items():
        if isinstance(provider_raw, dict):
            config.sota_providers[name] = CloudAgentConfig(**provider_raw)

    # Tools
    tools_raw = raw.get("tools", {})
    if tools_raw:
        ha_raw = tools_raw.get("ha", {})
        ha_config = HAToolsConfig(
            url=ha_raw.get("url", ""),
            token=ha_raw.get("token", ""),
            auto_discover=ha_raw.get("auto_discover", False),
            allowlist=ha_raw.get("allowlist", []),
            sensitive_override=ha_raw.get("sensitive_override", False),
            supported_languages=ha_raw.get("supported_languages", []),
        )
        config.tools = ToolsConfig(
            ha=ha_config,
            tool_timeout_seconds=tools_raw.get("tool_timeout_seconds", 3.0),
        )

    # Sanitization
    sanitization_raw = raw.get("sanitization", {})
    if sanitization_raw:
        config.sanitization = SanitizationConfig(**sanitization_raw)

    voice_raw = raw.get("voice", {})
    if voice_raw:
        config.voice = VoiceConfig(**voice_raw)

    speaker_raw = raw.get("speaker", {})
    if speaker_raw:
        config.speaker = SpeakerConfig(**speaker_raw)

    stt_raw = raw.get("stt", {})
    if stt_raw:
        config.stt = STTConfig(**stt_raw)

    tts_raw = raw.get("tts", {})
    if tts_raw:
        voices_raw = tts_raw.pop("voices", {})
        config.tts = TTSConfig(**tts_raw)
        for lang_code, voice_data in voices_raw.items():
            if isinstance(voice_data, dict):
                config.tts.voices[lang_code] = TTSVoiceConfig(**voice_data)

    return config
