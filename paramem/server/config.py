"""Server configuration — loads server.yaml into typed dataclasses."""

import logging
import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

import yaml

from paramem.backup.types import FatalConfigError
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
    "qwen3-4b": ModelConfig(
        model_id="Qwen/Qwen3-4B-Instruct-2507",
        quantization="nf4",
        compute_dtype="bfloat16",
        trust_remote_code=True,
        cpu_offload=False,
    ),
}


@dataclass
class OrphanSweepConfig:
    """Config knob for migration crash-recovery orphan sweep.

    ``max_age_hours`` is the look-back window for the orphan-sweep case in
    ``recover_migration_state``.  Pre-migration backup slots created within
    this window and whose ``pre_trial_hash`` matches the live config are
    considered orphaned step-2 artifacts and deleted on startup.  Slots
    outside the window are left in place (operator visibility).

    Absence of the key in ``server.yaml`` yields the 24h default.
    """

    max_age_hours: int = 24


@dataclass
class RetentionTierConfig:
    """Per-tier retention knobs.

    Attributes
    ----------
    keep:
        Maximum slots to retain in the tier (oldest pruned first).  Integer,
        or the literal string ``"unlimited"`` for tiers like ``manual`` that
        should never be time-pruned.  ``0`` disables emission for the tier
        (runner skips writing; existing slots are treated as obsolete and
        pruned by ``prune()``).
    max_disk_gb:
        Optional per-tier disk cap (``None`` = no per-tier cap).  When set
        and exceeded, oldest-first slots within the tier are pruned regardless
        of ``keep``.  Rule 2 of the retention precedence: per-tier disk cap,
        applied before keep-count pruning.
    """

    keep: int | str = 7  # int OR Literal["unlimited"]; YAML loader coerces
    max_disk_gb: float | None = None


@dataclass
class RetentionConfig:
    """Per-tier retention configuration (keep count + optional per-tier disk cap)."""

    daily: RetentionTierConfig = field(default_factory=lambda: RetentionTierConfig(keep=7))
    weekly: RetentionTierConfig = field(default_factory=lambda: RetentionTierConfig(keep=4))
    monthly: RetentionTierConfig = field(default_factory=lambda: RetentionTierConfig(keep=12))
    yearly: RetentionTierConfig = field(default_factory=lambda: RetentionTierConfig(keep=3))
    pre_migration: RetentionTierConfig = field(default_factory=lambda: RetentionTierConfig(keep=10))
    pre_base_swap: RetentionTierConfig = field(default_factory=lambda: RetentionTierConfig(keep=10))
    trial_adapter: RetentionTierConfig = field(default_factory=lambda: RetentionTierConfig(keep=5))
    manual: RetentionTierConfig = field(
        default_factory=lambda: RetentionTierConfig(keep="unlimited", max_disk_gb=5.0)
    )


@dataclass
class ServerBackupsConfig:
    """Sub-config for security.backups (``orphan_sweep``, ``retention``,
    ``schedule``, ``artifacts``, ``max_total_disk_gb``, ``adapter_scope``).

    Merged into ``SecurityConfig`` chain as ``security.backups``.
    """

    orphan_sweep: OrphanSweepConfig = field(default_factory=OrphanSweepConfig)
    retention: RetentionConfig = field(default_factory=RetentionConfig)
    schedule: str = "daily 04:00"  # "off" disables scheduled backups
    # Deprecated: the per-artifact list ["config", "graph", "registry"] is superseded
    # by the self-contained recovery bundle ("snapshot_bundle").  The bundle path ignores
    # this field; it is kept to avoid breaking existing server.yaml configs.  New
    # installations should set artifacts: ["snapshot_bundle"] (or rely on the runner
    # default when kinds are not explicitly configured in server.yaml).
    artifacts: list[str] = field(default_factory=lambda: ["config", "graph", "registry"])
    max_total_disk_gb: float = 20.0  # global cap; writes refused when usage reaches this
    adapter_scope: str = "live"
    """Controls which adapter slots are captured by ``write_bundle()``.

    ``"live"`` (default): capture the live-serving slot for each enabled
    adapter.  This includes an interim slot when no finalized main slot has
    been written yet — it ensures episodic adapters (which may only ever
    live as interim slots between full consolidation cycles) are always
    captured.

    ``"main"``: capture only finalized main slots; the bundle writer raises
    ``BackupError`` when an enabled adapter has no finalized main slot.  Use
    this mode when you need to guarantee that only post-consolidation weights
    enter the bundle.

    Valid values: ``"live"`` | ``"main"``.  Any other value raises
    ``ValueError`` at validation time.
    """

    def __post_init__(self) -> None:
        valid_scopes = {"live", "main"}
        if self.adapter_scope not in valid_scopes:
            raise ValueError(
                f"security.backups.adapter_scope must be one of {sorted(valid_scopes)!r}; "
                f"got {self.adapter_scope!r}"
            )


@dataclass
class SecurityConfig:
    """Top-level security configuration block.

    ``require_encryption`` is the single uniform knob operators use to opt into
    a fail-loud posture: when ``True`` and the daily age identity is not
    loadable at startup, the server refuses to start. The default ``False``
    matches AUTO semantics everywhere — encrypt when a key is loaded,
    plaintext otherwise.
    """

    backups: ServerBackupsConfig = field(default_factory=ServerBackupsConfig)
    require_encryption: bool = False


@dataclass
class RestartConfig:
    """systemd restart policy for the paramem-server service.

    Values flow through scripts/setup/server-restart-reconcile.sh into
    a drop-in at ~/.config/systemd/user/paramem-server.service.d/restart.conf
    on next server start. Changing values here without re-running the
    reconciler (or restarting the service, which calls it) has no effect.

    Fields map directly to systemd unit options:

    - on_failure         → Restart=on-failure (True) | no (False)
    - interval_seconds   → RestartSec=
    - max_attempts       → StartLimitBurst=
    - window_seconds     → StartLimitIntervalSec=
    - permanent_failure_exit_codes → RestartPreventExitStatus= (space-joined).
      Must be non-empty: an empty list would render an unset value to
      systemd, silently disabling the permanent-failure short-circuit and
      re-introducing the retry-storm risk this knob exists to prevent.

    Defaults are conservative: 3 retries / 60 s, with FatalConfigError
    (exit 3) treated as permanent so a config refusal doesn't burn cycles.
    """

    on_failure: bool = True
    interval_seconds: int = 30
    max_attempts: int = 3
    window_seconds: int = 60
    permanent_failure_exit_codes: list[int] = field(default_factory=lambda: [3])

    def __post_init__(self) -> None:
        if self.interval_seconds < 1:
            raise ValueError(
                f"process.restart.interval_seconds must be >= 1; got {self.interval_seconds!r}"
            )
        if self.max_attempts < 1:
            raise ValueError(
                f"process.restart.max_attempts must be >= 1; got {self.max_attempts!r}"
            )
        if self.window_seconds < 1:
            raise ValueError(
                f"process.restart.window_seconds must be >= 1; got {self.window_seconds!r}"
            )
        if not self.permanent_failure_exit_codes:
            raise ValueError(
                "process.restart.permanent_failure_exit_codes must not be empty; "
                "include at least exit code 3 (FatalConfigError) to prevent retry storms"
            )
        for code in self.permanent_failure_exit_codes:
            if not (0 <= code <= 255):
                raise ValueError(
                    "process.restart.permanent_failure_exit_codes entries must be in [0, 255];"
                    f" got {code!r}"
                )


@dataclass
class ProcessConfig:
    """Process-level lifecycle policy. Currently just systemd restart behaviour."""

    restart: RestartConfig = field(default_factory=RestartConfig)


@dataclass
class VramConfig:
    """Process-side VRAM safety net configuration.

    ``process_cap_fraction`` is the share of device VRAM that
    ``torch.cuda.set_per_process_memory_fraction`` allows the paramem
    process to allocate. The reserved ``(1 - fraction)`` is a bulkhead
    between PyTorch's allocator and the host GPU driver: an over-
    allocation past the cap surfaces as ``torch.cuda.OutOfMemoryError``
    in Python rather than as a host driver fault (which on WSL2 takes
    the VM down with it).

    Default 0.85 leaves ~15 % of the device for OS / other consumers
    on an 8 GiB device that is the project's primary target. Lower it
    if other GPU workloads share the device; raise it cautiously if
    you've measured spare headroom and need every byte.
    """

    process_cap_fraction: float = 0.85
    # KV-cache + activation headroom passed to assess_topology and the post-load
    # gate. Code default 1.0 GiB is the conservative minimum for callers that
    # don't load a yaml; the shipped yamls (configs/server.yaml,
    # tests/fixtures/server.yaml, configs/server.yaml.example) ship 1.5 GiB for
    # the per-phase peak — single-sequence 8K-token Mistral KV cache is ~1 GiB
    # on bf16 KV; 0.5 GiB margin for activations. vram_scope's inter-phase
    # empty_cache releases the allocator pool between phases, so the reservation
    # only needs to cover one phase's max KV growth.
    vram_cache_headroom_gib: float = 1.0
    # Nominal target GPU VRAM in GiB. assess_topology compares the worst-case
    # working set against this baseline and surfaces a /status.attention warning
    # when the configured topology exceeds it. ParaMem targets 8 GiB GPUs
    # (RTX 5070 Laptop, RTX 3060) as the published minimum; raise on bigger
    # hardware so the "fits" verdict reflects what you actually run on.
    baseline_vram_gib: int = 8
    # ── Calibration knobs (predictor + adapter math) ────────────────────────
    # Empirical factors that depend on the loaded model / library version.
    # Surfaced as config so a model or quant-scheme swap can be tuned without
    # a code edit; defaults are the values measured on the project's validated
    # models (Mistral 7B / Gemma 2 9B / Qwen 2.5 7B NF4, distil-large-v3 int8,
    # Piper + ONNX Runtime).  The live post-load gate catches mis-calibration,
    # so a wrong number here is a noisy warning, never a silent OOM.
    #
    # Runtime bytes per byte of bf16/fp16 safetensors on disk under NF4 + BNB
    # double-quant at block_size=64.  Verified 0.275 ± 1% across the three
    # validated 7-9B models on RTX 5070.  Re-measure on bnb major-bump.
    nf4_disk_to_runtime_factor: float = 0.275
    # CT2 / faster-whisper activation + workspace overhead multiplier on top of
    # the compute_type-quantized weight size.  1.1 ≈ 10% overhead, calibrated
    # against distil-large-v3 int8 (770 MiB predicted vs 960 MiB measured).
    stt_workspace_factor: float = 1.1
    # ONNX Runtime CUDA context size in MiB (shared across all Piper voices in
    # one process — counted once regardless of voice count).
    tts_piper_ort_context_mib: int = 300
    # PEFT per-adapter residual overhead in MiB, beyond the pure LoRA A+B
    # tensors.  Measured ~10 MiB for [q,v,k,o] rank=8 on Mistral 7B (PEFT
    # ModulesToSave wrappers + adapter-config metadata + allocator alignment).
    # Larger target_modules sets shift this; re-measure if you extend to MLP.
    peft_overhead_per_adapter_mib: int = 10

    def __post_init__(self) -> None:
        if not (0.0 < self.process_cap_fraction <= 1.0):
            raise ValueError(
                f"vram.process_cap_fraction must be in (0, 1]; got {self.process_cap_fraction!r}"
            )
        if self.vram_cache_headroom_gib <= 0:
            raise ValueError(
                f"vram.vram_cache_headroom_gib must be > 0; got {self.vram_cache_headroom_gib!r}"
            )
        if self.baseline_vram_gib <= 0:
            raise ValueError(f"vram.baseline_vram_gib must be > 0; got {self.baseline_vram_gib!r}")
        if self.nf4_disk_to_runtime_factor <= 0:
            raise ValueError(
                f"vram.nf4_disk_to_runtime_factor must be > 0; "
                f"got {self.nf4_disk_to_runtime_factor!r}"
            )
        if self.stt_workspace_factor <= 0:
            raise ValueError(
                f"vram.stt_workspace_factor must be > 0; got {self.stt_workspace_factor!r}"
            )
        if self.tts_piper_ort_context_mib < 0:
            raise ValueError(
                f"vram.tts_piper_ort_context_mib must be >= 0; "
                f"got {self.tts_piper_ort_context_mib!r}"
            )
        if self.peft_overhead_per_adapter_mib < 0:
            raise ValueError(
                f"vram.peft_overhead_per_adapter_mib must be >= 0; "
                f"got {self.peft_overhead_per_adapter_mib!r}"
            )


@dataclass
class ServerNetConfig:
    host: str = "0.0.0.0"
    port: int = 8420
    reclaim_interval_minutes: int = 10  # auto-reclaim GPU check interval


@dataclass
class PathsConfig:
    """Mount points for persistent data, sessions, debug output, and prompts."""

    data: Path = Path("data/ha")
    sessions: Path = Path("data/ha/sessions")
    debug: Path = Path("data/ha/debug")
    prompts: Path = Path("configs/prompts")

    @property
    def adapters(self) -> Path:
        """Path to the adapters directory.

        Raises
        ------
        ValueError
            If ``data`` is ``None``.
        """
        if self.data is None:
            raise ValueError("paths.data must be set to derive adapters path")
        return self.data / "adapters"

    @property
    def registry_dir(self) -> Path:
        """Directory that holds key_metadata.json and the combined SimHash registry.

        Raises
        ------
        ValueError
            If ``data`` is ``None`` (Fix 7, 2026-04-23).
        """
        if self.data is None:
            raise ValueError("paths.data must be set to derive registry_dir path")
        return self.data / "registry"

    @property
    def key_metadata(self) -> Path:
        """Path to the key-level persistence file written by the consolidation loop.

        Canonical on-disk location: ``<data>/registry/key_metadata.json``.
        This matches the path that the consolidation writer uses and that the
        production read sites (attention, app) previously hardcoded as a
        workaround.  Use this property instead of constructing the path inline.

        Raises
        ------
        ValueError
            If ``data`` is ``None`` (Fix 7, 2026-04-23).
        """
        if self.data is None:
            raise ValueError("paths.data must be set to derive key_metadata path")
        return self.data / "registry" / "key_metadata.json"

    @property
    def registry(self) -> Path:
        """Path to the combined SimHash registry written by the consolidation
        loop's ``_save_registry`` and read by ``inference.py`` for hallucination
        detection. **Distinct from ``key_metadata``** — different file, different
        schema. Do NOT alias the two: ``key_metadata`` carries per-key metadata;
        ``registry`` carries the combined SimHash dict.

        Raises
        ------
        ValueError
            If ``data`` is ``None`` (Fix 7, 2026-04-23).
        """
        if self.data is None:
            raise ValueError("paths.data must be set to derive registry path")
        return self.data / "registry.json"


@dataclass
class CloudAgentConfig:
    """Configuration for a cloud (SOTA) agent."""

    enabled: bool = False
    provider: str = "openai"  # openai, anthropic, google, groq
    model: str = ""
    api_key: str = field(default="", repr=False)
    endpoint: str = ""  # optional custom endpoint (for Groq, ollama, etc.)
    timeout_seconds: float = 90.0  # request timeout per call to this provider's API


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
    """PII sanitization + cloud egress policy.

    Three layered knobs:

    * ``mode`` — sanitizer detection layer (off / warn / block).  Controls
      whether ``sanitize_for_cloud`` flags personal-marker queries and
      whether it null-returns them so callers suppress the cloud call.

    * ``cloud_mode`` — egress policy for direct SOTA escalation
      (block / anonymize / both):

        - ``block``     — PERSONAL queries dropped before SOTA;
          non-PERSONAL sent verbatim.  Smallest cloud surface, zero
          anonymizer cost.  Default.
        - ``anonymize`` — cloud-bound text is anonymized via the local
          LLM (scope set by ``cloud_scope``), sent to SOTA as
          placeholders, and de-anonymized on return.  PERSONAL queries
          reach cloud as redacted text.
        - ``both``      — strictest privacy posture.  PERSONAL blocked
          AND non-PERSONAL anonymized per ``cloud_scope``.

    * ``cloud_scope`` — list of NER categories anonymized when
      ``cloud_mode`` selects an anonymizing mode.  Defaults to
      ``["person"]``: only person names are placeholdered; place,
      organization, etc. pass through verbatim so the cloud can answer
      questions like "What's a good restaurant in Berlin?" sensibly.
      An empty list (``[]``) disables the anonymization branch
      entirely under ``cloud_mode=anonymize|both`` — cloud sees the
      original text unchanged.  Operator owns the privacy/utility
      tradeoff; values are passed through to the NER filter without an
      allowlist check, so any spaCy-supported category may be used
      (see ``configs/server.yaml.example`` for the documented set).

    HA path is independent of ``cloud_mode`` and ``cloud_scope``: HA
    receives cleartext gated by intent classification.  Hardening the
    HA hop is a planned follow-up.
    """

    mode: str = "block"  # off, warn, block
    cloud_mode: str = "block"  # block, anonymize, both
    cloud_scope: list[str] = field(default_factory=lambda: ["person"])

    def __post_init__(self):
        valid_mode = {"off", "warn", "block"}
        if self.mode not in valid_mode:
            raise ValueError(
                f"Invalid sanitization mode '{self.mode}'. Must be one of: {valid_mode}"
            )
        valid_cloud_mode = {"block", "anonymize", "both"}
        if self.cloud_mode not in valid_cloud_mode:
            raise ValueError(
                f"Invalid sanitization cloud_mode '{self.cloud_mode}'. "
                f"Must be one of: {valid_cloud_mode}"
            )
        if not isinstance(self.cloud_scope, list) or not all(
            isinstance(v, str) and v for v in self.cloud_scope
        ):
            raise ValueError(
                f"Invalid sanitization cloud_scope '{self.cloud_scope}'. "
                f"Must be a list of non-empty strings (may be empty list to disable)."
            )


_ABSTENTION_RESPONSE_FALLBACK = "I don't have that information stored yet."
_ABSTENTION_COLD_START_FALLBACK = (
    "I'm still getting to know you, but I don't have that information yet."
)


@dataclass
class AbstentionConfig:
    """Last-resort confabulation guard.

    When no local parametric-memory match exists AND the sanitizer blocked
    cloud escalation (personal / self-referential query), ``handle_chat``
    would otherwise fall through to the bare base model with no context and
    confabulate. When ``enabled``, the appropriate canned message is
    returned verbatim instead.

    Two messages distinguish the two states a self-referential query can hit:

    * ``response`` — fired when the speaker has parametric facts but this
      particular query missed (a known limitation in coverage).
    * ``cold_start_response`` — fired when the speaker is identified but
      has no parametric facts yet (cold start before consolidation has
      absorbed their introduction). The canned ``response`` reads as
      confused in this state because the system *can't* know anything
      about a freshly enrolled speaker.

    Both messages are externalised to files under ``configs/prompts/`` so
    they can be tuned without code changes — same pattern as
    :class:`VoiceConfig`. ``*_override`` fields let an operator pin a
    specific string in the YAML when the file path is not desired; an empty
    override falls back to the file, and a missing file falls back to the
    module-level constant.
    """

    enabled: bool = True
    response_file: str = "configs/prompts/abstention_response.txt"
    cold_start_response_file: str = "configs/prompts/abstention_cold_start.txt"
    response_override: str = ""
    cold_start_response_override: str = ""

    def load_response(self) -> str:
        """Resolve the standard abstention message: override → file → fallback."""
        if self.response_override:
            return self.response_override
        if self.response_file:
            path = Path(self.response_file)
            if path.exists():
                return path.read_text().strip()
        return _ABSTENTION_RESPONSE_FALLBACK

    def load_cold_start_response(self) -> str:
        """Resolve the cold-start message: override → file → fallback."""
        if self.cold_start_response_override:
            return self.cold_start_response_override
        if self.cold_start_response_file:
            path = Path(self.cold_start_response_file)
            if path.exists():
                return path.read_text().strip()
        return _ABSTENTION_COLD_START_FALLBACK


@dataclass
class IntentConfig:
    """Intent classifier — HA fast-path + content-driven residual.

    The dispatcher tries the deterministic HA fast-path first; on miss,
    it falls to the residual classifier configured by :attr:`mode`.
    Below-margin / unavailable cases use :attr:`fail_closed_intent`.

    Two residual backends:

    * ``mode="llm"`` (production default).  Single-token generation
      from the loaded local model using the intent-classifier section
      of ``voice.prompt_file``.  Handles paraphrase, named entities,
      compound noisy transcripts, and synonyms without exemplar
      maintenance.  ~2–4 forward passes through the local LLM per
      query.  Automatically falls back to the encoder path when no
      local model is registered (cloud-only mode, model load failed).
    * ``mode="embeddings"``.  Sentence-encoder cosine vs per-class
      exemplar bank, gated by a top-1/top-2 margin.  ~1 forward pass
      through ``intfloat/multilingual-e5-small`` (118M params,
      384-dim, multilingual).  Cheap per query but the bank is static
      — every dynamic phrasing the operator hasn't anticipated either
      stays below margin or misclassifies.  Used in cloud-only mode
      and by operators who prefer no per-query LLM cost.

    Why LLM is the default: routing is an open-vocabulary problem.
    Each user-facing phrase the static bank doesn't cover surfaces as
    a missed classification (observed in production: named-station
    play, stop-shaped imperatives, compound radio-bleed-through
    transcripts).  The LLM is already loaded for the PA path and
    handles these natively; the per-query cost (~50–200 ms) is buried
    under the chat handler's other overhead.  ``embeddings`` remains
    for cloud-only deployments and as the automatic fallback when no
    local model is available.

    Defaults summary:

    * Encoder model: ``intfloat/multilingual-e5-small`` (MIT, 118M
      params, 384-dim multilingual sentence encoder).
    * Device: ``auto`` — prefer cuda, fall back to cpu.
    * Dtype: ``float16`` on cuda, ``float32`` on cpu.
    * E5 family requires the literal ``"query: "`` prefix on inputs.
    * Exemplar files live under ``configs/intents/`` as
      ``<class>.<lang>.txt`` (one query per line) — used by the
      ``embeddings`` backend and by the encoder-reusing sentence-type
      / personal-referent classifiers.
    * Confidence is the margin between the top-1 and top-2 cosine
      similarity scores against the exemplar set; below threshold
      classification falls back to ``fail_closed_intent``.
    * ``fail_closed_intent="personal"`` keeps privacy-preserving
      defaults under uncertainty — a misclassified personal query
      never escalates.

    All fields can be overridden in ``configs/server.yaml`` so
    operators can swap mode, encoders, exemplar sets, or thresholds
    without code changes.
    """

    enabled: bool = True
    # Classifier backend.  "embeddings" uses the sentence-encoder + per-class
    # exemplar bank at ``exemplars_dir`` (cheap, ~1 ms; needs curated exemplars
    # per class per language).  "llm" generates one token from the loaded local
    # model using the intent-classifier section of ``voice.prompt_file``
    # (~50–200 ms; no exemplar curation, handles paraphrase / named entities
    # the encoder bank misses).  When "llm" is selected but no local model has
    # been registered with the intent module (cloud-only mode, model load
    # failed), classification falls back to the encoder path automatically.
    mode: str = "embeddings"  # "embeddings" | "llm"
    encoder_model: str = "intfloat/multilingual-e5-small"
    encoder_device: str = "auto"  # auto | cuda | cpu
    encoder_dtype: str = "float16"  # float16 | float32
    encoder_query_prefix: str = "query: "  # E5 family requires this; empty for others
    exemplars_dir: str = "configs/intents"
    confidence_margin: float = 0.05
    fail_closed_intent: str = "personal"  # personal | command | general | unknown
    # LLM-mode generation knobs.  Kept tight: one label, deterministic.
    llm_max_new_tokens: int = 8
    llm_temperature: float = 0.0


@dataclass
class PersonalReferentConfig:
    """Encoder-based personal-referent classifier — ABOUT_SPEAKER /
    NOT_ABOUT_SPEAKER.

    Reuses the multilingual sentence encoder loaded by
    :class:`IntentConfig` (singleton at
    :func:`paramem.server.intent.get_encoder`); only the exemplar bank
    and the threshold are independent here.  Used by the cloud-egress
    sanitizer to gate first-person detection multilingually — replaces
    the English-only token-set lookup in
    :func:`paramem.server.sanitizer._contains_first_person`.

    * Exemplar files live under ``configs/personal_referent/`` as
      ``<class>.<lang>.txt`` (e.g. ``about_speaker.de.txt``).
      ``<class>`` must be a valid
      :class:`paramem.server.personal_referent.PersonalReferent` value
      (``about_speaker`` / ``not_about_speaker``).  Adding a new
      language is just a new file pair; no code change.
    * ``confidence_margin`` is the gap between top-1 and top-2 cosine
      similarity scores.  Below the margin the classifier returns
      ``None`` and the caller falls back to the English token-set
      heuristic.
    """

    enabled: bool = True
    exemplars_dir: str = "configs/personal_referent"
    confidence_margin: float = 0.05


@dataclass
class SentenceTypeConfig:
    """Encoder-based sentence-type classifier — INTERROGATIVE / NON_INTERROGATIVE.

    Reuses the multilingual sentence encoder loaded by
    :class:`IntentConfig` (singleton at
    :func:`paramem.server.intent.get_encoder`); only the exemplar bank
    and the threshold are independent here.  Used by the abstention
    helper to gate canned-response short-circuits multilingually
    without per-language lexicons.

    * Exemplar files live under ``configs/sentence_types/`` as
      ``<class>.<lang>.txt`` (e.g. ``interrogative.de.txt``).  ``<class>``
      must be a valid :class:`paramem.server.sentence_type.SentenceType`
      value (``interrogative`` / ``non_interrogative``).  Adding a new
      language is just a new file pair; no code change.
    * ``confidence_margin`` is the gap between top-1 and top-2 cosine
      similarity scores.  Below the margin the classifier returns
      ``None`` and the caller falls back to its deterministic heuristic
      (punctuation + English first-word lexicon) — that's *more*
      informative than fail-closing to a fixed default at the gate
      level, where the heuristic is meaningful.
    """

    enabled: bool = True
    exemplars_dir: str = "configs/sentence_types"
    confidence_margin: float = 0.05


_INTENT_CLASSIFIER_MARKER = "##---INTENT-CLASSIFIER-SECTION---"


@dataclass
class VoiceConfig:
    prompt_file: str = "configs/prompts/pa_voice.txt"
    system_prompt: str = ""
    greeting_interval_hours: int = 24  # hours between greetings per speaker (0 = disabled)
    # Per-language, per-period greetings (period = morning|afternoon|evening),
    # prepended app-side in the detected language. Operator-editable in server.yaml;
    # unknown languages fall back to "en".
    greetings: dict[str, dict[str, str]] = field(
        default_factory=lambda: {
            "en": {
                "morning": "Good morning",
                "afternoon": "Good afternoon",
                "evening": "Good evening",
            },
            "de": {"morning": "Guten Morgen", "afternoon": "Guten Tag", "evening": "Guten Abend"},
            "fr": {"morning": "Bonjour", "afternoon": "Bon après-midi", "evening": "Bonsoir"},
            "es": {
                "morning": "Buenos días",
                "afternoon": "Buenas tardes",
                "evening": "Buenas noches",
            },
        }
    )

    def _read_prompt_file(self) -> str | None:
        """Read the raw prompt file, or ``None`` if absent / unconfigured."""
        if not self.prompt_file:
            return None
        path = Path(self.prompt_file)
        if not path.exists():
            return None
        return path.read_text()

    def load_prompt(self) -> str:
        """Load the PA-path system prompt.

        If the prompt file contains an ``##---INTENT-CLASSIFIER-SECTION---``
        marker, only the content **before** that marker is returned —
        downstream callers (PA-path reasoning) must not see the
        classifier section's instructions.  Files without the marker are
        returned whole (back-compat with prompts authored before LLM
        intent classification existed).
        """
        raw = self._read_prompt_file()
        if raw is not None:
            head = raw.split(_INTENT_CLASSIFIER_MARKER, 1)[0]
            return head.strip()
        if self.system_prompt:
            return self.system_prompt
        return (
            "You are a personal memory assistant. Answer concisely in 1-2 spoken sentences. "
            "Use only facts that appear in the context above. Never invent personal details."
        )

    def load_intent_classifier_prompt(self) -> str | None:
        """Return the intent-classifier section of the prompt file, or
        ``None`` if the file is absent or the marker is missing.

        Used by :func:`paramem.server.intent._classify_via_llm` when
        ``IntentConfig.mode == "llm"``.  Operators tune the classifier
        prompt by editing the section after the marker in
        ``configs/prompts/pa_voice.txt``; no code change required.
        """
        raw = self._read_prompt_file()
        if raw is None:
            return None
        parts = raw.split(_INTENT_CLASSIFIER_MARKER, 1)
        if len(parts) != 2:
            return None
        tail = parts[1].strip()
        return tail or None


@dataclass
class ConsolidationScheduleConfig(ConsolidationConfig):
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
    retain_sessions: bool = True
    # Maximum number of interim cycles a session can be held pending because
    # of a recall-gate failure or DEGENERATE outcome.  When the counter
    # reaches this cap the session is released (no longer pinned), a
    # consolidation_retry_exhausted incident is recorded, and a WARNING is
    # logged.  ABORT cycles (yield-to-inference) do NOT increment — only
    # genuine encoding failures count.  Must be a positive integer (> 0).
    # Read at SessionBuffer construction (app.py).
    consolidation_retry_cap: int = 3
    # Maximum LoRA training epochs per consolidation cycle. None = use the
    # validated 30 default (Test 17 floor) for 100% indexed-key recall on the
    # validated models.
    #
    # Override semantics: ceiling, not target. Once the recall-driven
    # early-stop callback ships, training will terminate at the recall
    # plateau; this field caps the upper bound. Until then, training runs
    # the full configured count.
    #
    # Operator guidance: do NOT lower below 30 in production without an
    # empirical recall-vs-epochs validation on your specific base model.
    # The 30-epoch value was anchored on Mistral 7B; future base models
    # may converge sooner. Lower this only with evidence.
    #
    # Development harnesses (e.g. scripts/dev/e2e_artifact_smoke.py) may
    # set this to a small value to skip the full validated budget when
    # testing layout/encryption invariants rather than recall quality.
    max_epochs: int | None = None
    # LoRA training hyperparameters (Test 17 recipe).
    # num_epochs is the ceiling max_epochs above (None → 30); not repeated here.
    training_batch_size: int = 1
    training_gradient_accumulation_steps: int = 2
    training_max_seq_length: int = 1024
    training_warmup_steps: int = 30  # fixed steps; overrides warmup_ratio when > 0
    training_warmup_ratio: float = 0.0  # explicit 0.0 so TrainingConfig default (0.1) cannot leak
    training_lr_scheduler_type: str = "linear"
    training_weight_decay: float = 0.1
    training_gradient_checkpointing: bool = True
    training_max_grad_norm: float = 1.0
    training_seed: int = 42
    training_lr_decay_steps: int | None = None
    # Recall-based early stopping (default OFF).
    # When True, ConsolidationLoop wires RecallEarlyStopCallback at every
    # production train_adapter call site (via _maybe_make_recall_callback).
    # Probes the staged adapter at epoch boundaries and fires
    # control.should_training_stop after `recall_window` consecutive
    # 100%-recall probes past `early_stopping_floor` (TrainingConfig field;
    # currently 10 by default, 20 in the production fixture).
    # Validated at multi-seed for N=100 (Test 14: 9 cells stopped cleanly
    # e18-e26); untested at N=500+.  Flip after the first clean cycle.
    recall_early_stopping: bool = False
    # Consecutive 100%-recall probes required to fire the stop signal.
    # Test 14's validated default.  Lower (e.g. 2) for noisier
    # convergence curves; higher for tighter stop semantics.
    recall_window: int = 3
    # Probe cadence — system-wide.  Default 3 → probe every 3 epochs;
    # 3× cheaper than =1 at production scale (~N=500+).  Smaller cycles
    # still probe every 3 epochs from epoch 1.
    # If episodic and procedural cycles have very different epoch
    # budgets, this default is conservatively tuned for episodic;
    # procedural may probe at slightly worse signal-to-cost.
    recall_probe_every_n_epochs: int = 3
    # Signal floor — earliest epoch the recall callback can fire the
    # stop signal.  Maps to TrainingConfig.early_stopping_floor (which
    # the experiment-side loss-based callback also uses; recall path
    # reuses the same field as ``signal_from_epoch``).  Default 20 is
    # the conservative production posture — Test 14's multi-seed cells
    # all stopped between e18 and e26, so 20 sits inside the empirical
    # band without firing too early on outlier seeds.  Test 14's
    # research default was 10; that value remains in
    # ``TrainingConfig.early_stopping_floor`` for direct-construction
    # contexts that don't go through the YAML.
    recall_signal_from_epoch: int = 20
    # Recall probe batch size — see TrainingConfig.recall_probe_batch_size
    # for the empirical curve.  Default 16: ~4.75× probe wall-clock at
    # ~346 MiB peak VRAM delta, 137/137 recall parity vs serial, multi-cycle
    # retention parity confirmed in production conditions.
    recall_probe_batch_size: int = 16
    # Output token budget — drives every LLM call in the extraction pipeline
    # (local extract, anonymize, SOTA enrich, deanon) and every direct
    # response across modes.
    extraction_max_tokens: int = 8192
    # Plausibility-stage cap, separately configurable so an operator can
    # tighten it under VRAM pressure without dragging SOTA enrichment down
    # too. Default tracks the unified ``extraction_max_tokens`` budget —
    # observed PII-attribute capture (email/phone/linkedin) is more reliable
    # at the larger cap. Tighten (e.g. to 4096) only when the WSL2 host
    # headroom forces a trade-off; KV-cache savings:
    # (extraction_max_tokens − plausibility) × ~131 KB/token of peak ceiling
    # reduction on the worst-case generation.
    extraction_plausibility_max_tokens: int = 8192
    # Calibration endpoint — POST /calibrate/{extract,anonymize,plausibility}
    # exposes per-stage prompt iteration against the same Mistral instance the
    # production cycle uses. Default OFF — opt-in dev tool, never live in
    # production. Server short-circuits with 503 when a real consolidation
    # cycle is running so calibration calls cannot race against the model.
    calibrate_endpoint_enabled: bool = False
    extraction_stt_correction: bool = True  # correct STT errors from assistant responses
    extraction_ha_validation: bool = True  # validate locations against HA home context
    extraction_noise_filter: str = "anthropic"  # SOTA provider for noise filtering ("" = disabled)
    extraction_noise_filter_model: str = "claude-sonnet-4-6"  # model for noise filtering
    extraction_noise_filter_endpoint: str = ""  # custom endpoint for OpenAI-compatible providers
    training_temp_limit: int = 0  # GPU temp ceiling for background training (0 = disabled)
    training_temp_check_interval: int = 5  # check temp every N training steps
    # Quiet-hours policy: schedules when the thermal throttle is active.
    # Smartphone-style sleep mode for the thermal gate.
    #   "always_on"  — throttle 24/7 (shared living space; dataclass fallback)
    #   "always_off" — throttle never (server rack / cellar PC; ignore fan noise)
    #   "auto"       — throttle only inside [start, end) window (daytime-at-work etc.)
    # Window uses naive local time, is half-open (end exclusive), and wraps past
    # midnight when end < start (e.g. start=22:00 end=07:00 silences 22:00–07:00).
    # Shipped YAMLs both use "auto": default.yaml 22:00–07:00 (home/night),
    # server.yaml 07:00–22:00 (work-laptop daytime).
    quiet_hours_mode: str = "always_on"
    quiet_hours_start: str = "22:00"
    quiet_hours_end: str = "07:00"
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
    # Bounded temporary extension of the interim ring beyond max_interim_count.
    # When the ring is full (c >= max_interim_count) but c < max_interim_count +
    # interim_overflow_slack, a new later-stamped overflow slot is minted instead
    # of keeping sessions pending.  Each overflow slot gets its own adapter,
    # preserving temporal order.  The slack is included in the boot-time VRAM
    # budget (vram_validator.py) so the extension is proven to fit before the
    # server starts.  At 0 (default) overflow slots are never minted — cap_pending
    # fires immediately when the ring is full (identical to S4 behavior).
    interim_overflow_slack: int = 0
    # entity_similarity_threshold mirrors GraphConfig (paramem/utils/config.py);
    # bridged into GraphMerger construction via ServerConfig.graph_config.
    entity_similarity_threshold: float = 85.0
    # Cross-predicate contradiction detection; off by default —
    # over-removes multi-valued/independent facts (observed in live use).
    # Mirrors GraphConfig.cross_predicate_contradiction; bridged via graph_config.
    cross_predicate_contradiction: bool = False
    # --- Graph-level SOTA enrichment (Task #10) ---
    # Runs at full consolidation over the cumulative graph to capture
    # cross-transcript second-order relations that per-transcript enrichment
    # cannot see. Reuses the noise-filter SOTA credentials.
    graph_enrichment_enabled: bool = True
    graph_enrichment_neighborhood_hops: int = 2
    graph_enrichment_max_entities_per_pass: int = 50
    # RAM-mode checkpointing: when > 0, train_adapter writes checkpoints to
    # /dev/shm instead of the caller's output_dir, then copies the latest
    # checkpoint to <output_dir>/bg_checkpoint_epoch/ at each epoch boundary.
    # Trade-off: /dev/shm is not durable across restarts; a crash loses the
    # in-flight checkpoint.  Set to 0 (default) to disable.
    training_save_steps_ram: int = 0
    # Slot retention — see ConsolidationLoop._prune_old_slots.
    #
    # After each promotion atomic_save_adapter writes a NEW timestamped slot
    # under <adapter_dir>/<tier>/<ts>/; find_live_slot picks whichever slot's
    # meta.registry_sha256 matches the live registry. Older slots stay on
    # disk indefinitely. This knob caps post-promotion prior slots per tier:
    # the live (registry-matched) slot is always kept; up to N additional
    # most-recent prior slots are retained for rollback; older slots are
    # rmtree'd inside _save_adapters AFTER the registry-commit step so a
    # brief commit-time race cannot expose unmatched state to readers.
    # Set to 0 to keep only the live slot. Set high (e.g. 50) when validating
    # slot lineage and disk is cheap.
    training_keep_prior_slots: int = 3
    # Skip new training submissions while /chat has fired within this window.
    # A scheduled-tick arriving sooner records a "deferred_idle" status.
    # Measured against _state["last_chat_monotonic"] via time.monotonic() so
    # a wall-clock NTP step does not break the predicate. Set to 0 to disable.
    training_idle_debounce_s: int = 30
    # Time /chat waits for the BG trainer to abort at the next step boundary
    # before falling back to setting _shutdown_requested directly. At step
    # times >> this, /chat falls through to the shutdown-flag path which is
    # heavier-handed. 30s covers typical BG training step durations.
    abort_quiesce_timeout_s: float = 30.0
    # Holdable-session retirement TTL.  Sessions that carry a voice embedding
    # (retro-claimable) or are anonymous-voice are held pending rather than
    # dropped immediately.  When a holdable session exceeds this age it is
    # retired to the discard sink (debug=True) or unlinked (debug=False).
    # Grammar: same as refresh_cadence — "every Nh" / "every Nm" / "HH:MM" /
    #   "daily" / "off" / "" (off = never retire holdable sessions).
    # Default "off" matches the pre-change behaviour: holdable sessions were
    # simply never extracted; now they are explicitly held indefinitely.
    orphan_retirement: str = "off"

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
        super().__post_init__()
        if self.training_save_steps_ram < 0:
            raise ValueError(
                f"consolidation.training_save_steps_ram must be >= 0; "
                f"got {self.training_save_steps_ram!r}"
            )
        if self.training_keep_prior_slots < 0:
            raise ValueError(
                f"consolidation.training_keep_prior_slots must be >= 0; "
                f"got {self.training_keep_prior_slots}"
            )
        if self.training_idle_debounce_s < 0:
            raise ValueError(
                f"consolidation.training_idle_debounce_s must be >= 0; "
                f"got {self.training_idle_debounce_s}"
            )
        if self.abort_quiesce_timeout_s <= 0:
            raise ValueError(
                f"consolidation.abort_quiesce_timeout_s must be > 0; "
                f"got {self.abort_quiesce_timeout_s}"
            )

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

        if self.consolidation_retry_cap < 1:
            raise ValueError(
                f"consolidation.consolidation_retry_cap must be > 0; "
                f"got {self.consolidation_retry_cap!r}"
            )

        if self.max_interim_count < 0:
            raise ValueError(
                f"consolidation.max_interim_count must be >= 0; "
                f"got {self.max_interim_count!r}. "
                f"Use 0 for full-fold-only consume-pending mode (no interim adapters "
                f"minted; pending sessions consumed by the scheduled full fold on "
                f"refresh_cadence). Use >= 1 for the standard interim-adapter mode."
            )

        if self.max_interim_count == 0:
            try:
                from paramem.memory.interim_adapter import compute_schedule_period_seconds

                _period = compute_schedule_period_seconds(self.refresh_cadence)
            except ValueError as exc:
                raise ValueError(
                    f"consolidation.max_interim_count=0 requires a valid refresh_cadence; "
                    f"got {self.refresh_cadence!r}: {exc}"
                ) from exc
            if _period is None:
                raise ValueError(
                    f"consolidation.max_interim_count=0 requires a non-empty "
                    f"refresh_cadence; got {self.refresh_cadence!r}. "
                    f"At count=0 the full fold is the only training venue and runs "
                    f"every refresh_cadence — with no cadence configured the full fold "
                    f"never runs and pending sessions accumulate unboundedly."
                )

        if self.interim_overflow_slack < 0:
            raise ValueError(
                f"consolidation.interim_overflow_slack must be >= 0; "
                f"got {self.interim_overflow_slack!r}. "
                f"Use 0 (default) to disable overflow slots entirely, "
                f"or a positive integer to allow that many extra adapters "
                f"beyond max_interim_count before keep-pending kicks in."
            )

        # orphan_retirement: validate the schedule string early so the operator
        # sees a clear error at startup, not at the first tick.
        try:
            from paramem.memory.interim_adapter import compute_schedule_period_seconds

            compute_schedule_period_seconds(self.orphan_retirement)
        except ValueError as exc:
            raise ValueError(
                f"consolidation.orphan_retirement={self.orphan_retirement!r} is not a valid "
                f"schedule string. Use 'off', 'every Nh', 'every Nm', 'HH:MM', or 'daily'. "
                f"Original error: {exc}"
            ) from exc

        # Quiet-hours: reject unknown modes and malformed windows early.
        mode = self.quiet_hours_mode
        if mode not in ("always_on", "always_off", "auto"):
            raise ValueError(
                f"quiet_hours_mode={mode!r} must be one of 'always_on', 'always_off', 'auto'."
            )
        if mode == "auto":
            for fld, val in (
                ("quiet_hours_start", self.quiet_hours_start),
                ("quiet_hours_end", self.quiet_hours_end),
            ):
                try:
                    h, m = val.split(":")
                    hh, mm = int(h), int(m)
                    if not (0 <= hh < 24 and 0 <= mm < 60):
                        raise ValueError
                except Exception:
                    raise ValueError(
                        f"{fld}={val!r} must be HH:MM (24h); got invalid value."
                    ) from None

    @property
    def consolidation_period_seconds(self) -> int | None:
        """Full consolidation period in seconds — derived, not configured.

        Returns ``refresh_cadence × max_interim_count`` seconds, or ``None``
        when ``refresh_cadence`` is disabled (``""``/``"off"``/etc.). Used by
        the systemd timer to schedule the full-consolidation cycle and by
        ``pstatus`` to display the effective cadence.

        Special case: when ``max_interim_count == 0`` (full-fold-only
        consume-pending mode) there are no interim adapters; the full fold
        runs every ``refresh_cadence`` directly, so this returns
        ``refresh_seconds`` rather than ``refresh_seconds * 0``.
        """
        from paramem.memory.interim_adapter import compute_schedule_period_seconds

        refresh_seconds = compute_schedule_period_seconds(self.refresh_cadence)
        if refresh_seconds is None:
            return None
        if self.max_interim_count == 0:
            return refresh_seconds
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

    @property
    def orphan_retirement_seconds(self) -> int | None:
        """Holdable-session TTL in seconds, or ``None`` when retirement is off.

        ``None`` means holdable sessions (anonymous-voice or embedding-only)
        are held pending indefinitely — they are never retired by age.

        Uses :func:`~paramem.memory.interim_adapter.compute_schedule_period_seconds`
        with the same grammar as :attr:`refresh_cadence`.
        """
        from paramem.memory.interim_adapter import compute_schedule_period_seconds

        return compute_schedule_period_seconds(self.orphan_retirement)


@dataclass
class SpeakerConfig:
    """Voice-based speaker identification settings."""

    enabled: bool = False
    high_confidence_threshold: float = 0.60
    low_confidence_threshold: float = 0.45
    store_path: str = ""  # empty = default (data/ha/speaker_profiles.json)
    enrollment_prompt: str = "By the way, I don't think we've met yet. Please introduce yourself."
    enrollment_reprompt_interval: int = 600  # seconds between re-prompting unknown speakers
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
    # How the synth language is chosen when both a caller voice hint and a detected
    # language are available: "auto"/"detection" prefer ParaMem's detected language
    # (latest_language_detection) over the caller's voice.language hint; "hint" makes
    # the caller's hint win. All fall back to the other source, then default_language.
    language_source: str = "auto"
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
class TextLangDetectionConfig:
    """fastText lid.176-based language detection on the text /chat path.

    STT carries Whisper's language signal. Pure-text /chat requests have no
    equivalent; when both ``enabled`` is true and the request lacks a
    speaker embedding, ``paramem.server.lang_id`` is invoked on the request
    text and its verdict feeds the same resolver chain Whisper detection
    flows through.

    Disabled by default so existing deployments and the CI fixture do not
    require the 126 MB model file. Enable in production after running
    ``scripts/setup/download-langid-model.sh``.
    """

    enabled: bool = False
    # Conservative — text-side detection is the only language signal on the
    # text /chat path (no STT, no speaker preference fallback in practice).
    # Below this threshold, leave ``effective_language`` unset rather than
    # commit to a wrong language and harden the response in the wrong tongue.
    confidence_threshold: float = 0.65
    # Empty → falls back to ``~/.cache/paramem/lang_id/lid.176.bin``.
    model_path: str = ""


@dataclass
class MobilePwaConfig:
    """Progressive Web App (PWA) configuration for the mobile client.

    ``enabled``: serve the static PWA bundle and activate cookie-based
    authentication for the mobile web client.  False by default so existing
    API-key deployments are unaffected until the mobile slice is wired in.

    ``static_dir``: filesystem path to the compiled static bundle.  Empty
    string defers resolution to the mount point (``paramem/web/static``)
    configured in a later slice.

    ``cookie_name``: name of the cookie the middleware will accept if the
    client presents one.  The server does not issue this cookie; tokens are
    carried via the ``Authorization: Bearer`` header in practice.

    ``push_enabled``: enable Web Push notifications.  When true, the server
    auto-generates a VAPID EC P-256 keypair on first startup (persisted as
    ``vapid_keys.json`` in the data directory, age-encrypted when a daily key
    is loaded) and activates the ``/push/vapid-public-key`` and
    ``/push/subscribe`` endpoints.  Requires per-user bearer tokens (the
    subscribe endpoint returns 403 for shared-token or unauthenticated
    requests).  False by default (opt-in).

    ``vapid_contact``: the ``sub`` claim in the VAPID JWT.  Must be a
    ``mailto:`` URI identifying the server operator; sent to push relays for
    abuse-contact purposes.  A sane default is provided; operators should
    replace it with their own address.
    """

    enabled: bool = False
    # Empty → paramem/web/static (resolved by the static-mount slice).
    static_dir: str = ""
    cookie_name: str = "paramem_token"
    push_enabled: bool = False
    vapid_contact: str = "mailto:admin@localhost"


@dataclass
class InferenceConfig:
    """Inference-time options that govern the per-query probe path.

    ``preload_cache``: at boot, populate the lifespan-owned
    :class:`paramem.memory.store.MemoryStore` by probing every
    active key through the mode-appropriate
    :class:`paramem.memory.source.MemorySource`
    (:class:`~paramem.memory.source.WeightMemorySource` in
    train mode, :class:`~paramem.memory.source.DiskMemorySource`
    in simulate mode).  Inference then serves cache hits in O(1) and
    falls through to the source only on cache miss.

    The default is ``True``: with hundreds of keys per speaker, paying
    the per-key source latency on every conversational turn does not
    survive the latency budget (weight probe is ~0.3 s/key on Mistral
    7B Q4, unverified; disk read is faster but still adds up).

    Switching this to ``False`` is supported: the store stays empty for
    entries (registries + simhashes still load) and every cache miss
    delegates to the source.  Slower per query but correct — operators
    who want to validate parametric recall against the weights live
    on this path.  This is a temporary toggle until per-key probing is
    fast enough to drop the cache entirely.
    """

    preload_cache: bool = True


@dataclass
class ServerConfig:
    security: SecurityConfig = field(default_factory=SecurityConfig)
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
    paths: PathsConfig = field(default_factory=PathsConfig)
    adapters: ServerAdaptersConfig = field(default_factory=ServerAdaptersConfig)
    consolidation: ConsolidationScheduleConfig = field(default_factory=ConsolidationScheduleConfig)
    sota_agent: CloudAgentConfig = field(default_factory=CloudAgentConfig)
    sota_providers: dict[str, CloudAgentConfig] = field(default_factory=dict)
    ha_agent_id: str = ""  # HA conversation agent for escalation; empty disables HA escalation
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    sanitization: SanitizationConfig = field(default_factory=SanitizationConfig)
    abstention: AbstentionConfig = field(default_factory=AbstentionConfig)
    intent: IntentConfig = field(default_factory=IntentConfig)
    sentence_type: SentenceTypeConfig = field(default_factory=SentenceTypeConfig)
    personal_referent: PersonalReferentConfig = field(default_factory=PersonalReferentConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    speaker: SpeakerConfig = field(default_factory=SpeakerConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    text_lang_detection: TextLangDetectionConfig = field(default_factory=TextLangDetectionConfig)
    mobile_pwa: MobilePwaConfig = field(default_factory=MobilePwaConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    vram: VramConfig = field(default_factory=VramConfig)
    process: ProcessConfig = field(default_factory=ProcessConfig)

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
        """Training recipe from yaml (Test 17 defaults via consolidation.training_* fields).

        ``num_epochs`` honours ``consolidation.max_epochs`` when set,
        otherwise falls back to 30 (the validated floor). The override is
        a ceiling — once recall-driven early-stop ships, training will
        terminate at the recall plateau bounded by this value.

        All LoRA hyperparameters come from the yaml-configurable
        ``consolidation.training_*`` fields; see ConsolidationScheduleConfig.
        """
        return TrainingConfig(
            batch_size=self.consolidation.training_batch_size,
            gradient_accumulation_steps=self.consolidation.training_gradient_accumulation_steps,
            max_seq_length=self.consolidation.training_max_seq_length,
            num_epochs=(
                self.consolidation.max_epochs if self.consolidation.max_epochs is not None else 30
            ),
            warmup_steps=self.consolidation.training_warmup_steps,
            warmup_ratio=self.consolidation.training_warmup_ratio,
            lr_scheduler_type=self.consolidation.training_lr_scheduler_type,
            lr_decay_steps=self.consolidation.training_lr_decay_steps,
            weight_decay=self.consolidation.training_weight_decay,
            gradient_checkpointing=self.consolidation.training_gradient_checkpointing,
            max_grad_norm=self.consolidation.training_max_grad_norm,
            seed=self.consolidation.training_seed,
            # logging_steps lives on TrainingConfig (default 1, matches the
            # historical train_adapter hardcode).  The BG-trainer call site
            # uses dataclasses.replace to override to 10 when delegating to
            # train_adapter, preserving its prior log volume; the
            # consolidation/server path inherits the default.
            early_stopping_floor=self.consolidation.recall_signal_from_epoch,
            recall_early_stopping=self.consolidation.recall_early_stopping,
            recall_window=self.consolidation.recall_window,
            recall_probe_every_n_epochs=self.consolidation.recall_probe_every_n_epochs,
            recall_probe_batch_size=self.consolidation.recall_probe_batch_size,
            save_steps_ram=self.consolidation.training_save_steps_ram,
        )

    @property
    def consolidation_config(self) -> ConsolidationConfig:
        """Build ConsolidationConfig for ConsolidationLoop.

        All three-axis knobs (interim_refinement, fold_refinement,
        contradiction_detection) are threaded through here so ConsolidationLoop
        reads them from config rather than from direct attribute access.
        """
        return ConsolidationConfig(
            promotion_threshold=self.consolidation.promotion_threshold,
            indexed_key_replay=self.consolidation.indexed_key_replay,
            decay_window=self.consolidation.decay_window,
            interim_refinement=self.consolidation.interim_refinement,
            fold_refinement=self.consolidation.fold_refinement,
            contradiction_detection=self.consolidation.contradiction_detection,
            recall_sanity_threshold=self.consolidation.recall_sanity_threshold,
            min_tier_key_floor=self.consolidation.min_tier_key_floor,
            tier_fast_start=self.consolidation.tier_fast_start,
        )

    @property
    def graph_config(self) -> GraphConfig:
        """Build GraphConfig for GraphMerger construction.

        Mirrors ``entity_similarity_threshold`` and
        ``cross_predicate_contradiction`` from ``self.consolidation``.
        """
        return GraphConfig(
            entity_similarity_threshold=self.consolidation.entity_similarity_threshold,
            cross_predicate_contradiction=self.consolidation.cross_predicate_contradiction,
        )


DEFAULT_SERVER_CONFIG_PATH = Path("configs/server.yaml")


def load_server_config(path: str | Path = "configs/server.yaml") -> ServerConfig:
    """Load server configuration from YAML file.

    Supports ${VAR_NAME} env var interpolation in all string values.
    Accepts both new 'agents:' key and deprecated 'cloud:' key.

    Fresh-clone fallback: when ``path`` is the default operator-local
    location and the file does not exist (gitignored, never created), fall
    back to the shipped ``configs/server.yaml.example`` so CI runs and
    fresh checkouts boot without a manual copy step.
    """
    path = Path(path)
    if not path.exists():
        if path == DEFAULT_SERVER_CONFIG_PATH:
            template = path.parent / (path.name + ".example")
            if template.exists():
                logger.info(
                    "configs/server.yaml not found; loading shipped template "
                    "configs/server.yaml.example. Copy it to configs/server.yaml "
                    "to add operator-local overrides."
                )
                path = template
            else:
                return ServerConfig()
        else:
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

    # Paths — resolve relative paths against the project root so they work
    # regardless of (a) the process's working directory at runtime and
    # (b) where in the tree the yaml file lives. The project root is the
    # nearest ancestor of the yaml file containing ``pyproject.toml``;
    # this works for both the production case (``configs/server.yaml``)
    # and the test-fixture case (``tests/fixtures/server.yaml``) without
    # the loader having to know which sub-directory the yaml lives in.
    # Falls back to ``yaml_dir.parent.parent`` (the historical assumption
    # that yamls live one level deep under the project root) when no
    # ``pyproject.toml`` is found above the file — preserves legacy
    # behaviour for yamls supplied from outside the repo tree.
    yaml_dir = Path(path).resolve().parent
    config_dir = yaml_dir
    walker = yaml_dir
    while walker != walker.parent:
        if (walker / "pyproject.toml").exists():
            config_dir = walker
            break
        walker = walker.parent
    else:
        config_dir = yaml_dir.parent  # legacy fallback
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
    #
    # Merge YAML keys onto the factory-default ServerAdaptersConfig (not the
    # ServerAdapterConfig dataclass defaults). The factories carry slot-
    # specific defaults that the dataclass alone does not — most importantly
    # procedural's MLP target_modules. Constructing fresh
    # ServerAdapterConfig(**proc) instead would silently fall back to
    # attention-only target_modules whenever YAML omits target_modules but
    # specifies any other procedural key.
    adapters_raw = raw.get("adapters", {})
    if adapters_raw:
        factory = ServerAdaptersConfig()
        config.adapters = ServerAdaptersConfig(
            episodic=replace(factory.episodic, **adapters_raw.get("episodic", {})),
            semantic=replace(factory.semantic, **adapters_raw.get("semantic", {})),
            procedural=replace(factory.procedural, **adapters_raw.get("procedural", {})),
        )

    # Loader guard — explicit-yaml posture for load-bearing adapter fields.
    #
    # When an operator partially specifies a tier in yaml (any field present)
    # AND that tier ends up enabled, ``target_modules`` must be spelled out.
    # The loader otherwise silently merges from the factory default, which
    # hides architectural choices like procedural's attn+mlp targeting from
    # the operator's config. Refuse-loud here so future drift surfaces at
    # startup rather than as a confusing runtime mismatch later.
    #
    # Operators who omit the tier block entirely (no fields specified) are
    # signalling "use defaults for everything" — that's a legitimate posture
    # and is not gated.
    for _tier in ("episodic", "semantic", "procedural"):
        _raw_tier = adapters_raw.get(_tier)
        if not _raw_tier:
            continue  # operator did not mention this tier; defaults apply silently
        if "target_modules" in _raw_tier:
            continue  # operator was explicit
        _merged = getattr(config.adapters, _tier)
        if not _merged.enabled:
            continue  # tier disabled by operator; the field is moot
        raise FatalConfigError(
            f"adapters.{_tier}.enabled=true but target_modules is missing in "
            f"{path}.\n"
            f"\n"
            f"Likely cause:\n"
            f"  The yaml partially specifies adapters.{_tier} (e.g. enabled,\n"
            f"  rank, alpha, learning_rate) but omits target_modules. The loader\n"
            f"  would silently fall back to the factory default — hiding the\n"
            f"  architectural choice ({_tier}'s LoRA shape) from the operator's\n"
            f"  config. The yaml is the contract for load-bearing fields.\n"
            f"\n"
            f"Remediation:\n"
            f"  - Open {path} and add under adapters.{_tier}:\n"
            f"      target_modules: [...]\n"
            f"  - See configs/server.yaml.example for canonical values\n"
            f"    (procedural targets attn+mlp; episodic and semantic target\n"
            f"    attn-only).\n"
            f"  - OR remove the entire adapters.{_tier} block to fall back to\n"
            f"    the factory defaults intentionally."
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
    # Warn on legacy `simulate:` key rather than passing it to the dataclass
    # constructor (which would raise TypeError).  The current API uses
    # ``consolidation.mode: "simulate"`` as the authoritative knob.
    _legacy_simulate = consolidation_raw.pop("simulate", None)
    if _legacy_simulate is not None:
        logger.warning(
            "consolidation.simulate=%r is a legacy key — use "
            "consolidation.mode='simulate' instead. The legacy key is ignored.",
            _legacy_simulate,
        )
    # Retired keys: training_save_strategy_bg and training_save_steps_bg were
    # removed when the override layer was unified into TrainingConfig directly.
    # Detect them and raise loud rather than silently dropping (dataclass
    # **kwargs would raise TypeError anyway, but a targeted message is faster
    # to diagnose).
    for _retired_key in ("training_save_strategy_bg", "training_save_steps_bg"):
        if _retired_key in consolidation_raw:
            consolidation_raw.pop(_retired_key)
            raise ValueError(
                f"config error: `consolidation.{_retired_key}` was removed. "
                f"Use `training.save_strategy` and `training.save_steps` directly, "
                f"or set `consolidation.training_save_steps_ram` for RAM-mode "
                f"checkpointing. Remove `{_retired_key}` from your config file."
            )
    if consolidation_raw:
        config.consolidation = ConsolidationScheduleConfig(**consolidation_raw)

    # VRAM safety net (process-side cap; see VramConfig).
    vram_raw = raw.get("vram", {})
    if vram_raw:
        config.vram = VramConfig(**vram_raw)

    # Process lifecycle (systemd restart policy; see ProcessConfig / RestartConfig).
    process_raw = raw.get("process")
    if process_raw is not None:
        restart_raw = process_raw.get("restart")
        if restart_raw is not None:
            config.process = ProcessConfig(restart=RestartConfig(**restart_raw))
        else:
            config.process = ProcessConfig()

    agents_raw = raw.get("agents", {})
    config.ha_agent_id = agents_raw.get("ha_agent_id", "")

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

    # Abstention
    abstention_raw = raw.get("abstention", {})
    if abstention_raw:
        config.abstention = AbstentionConfig(**abstention_raw)

    # Intent classifier (residual classifier for routing decisions)
    intent_raw = raw.get("intent", {})
    if intent_raw:
        config.intent = IntentConfig(**intent_raw)

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

    text_lang_raw = raw.get("text_lang_detection", {})
    if text_lang_raw:
        config.text_lang_detection = TextLangDetectionConfig(**text_lang_raw)

    mobile_pwa_raw = raw.get("mobile_pwa", {})
    if mobile_pwa_raw:
        config.mobile_pwa = MobilePwaConfig(**mobile_pwa_raw)

    inference_raw = raw.get("inference", {})
    if inference_raw:
        config.inference = InferenceConfig(**inference_raw)

    # Security — nested: security.backups.{orphan_sweep, retention, schedule,
    # artifacts, max_total_disk_gb}
    security_raw = raw.get("security") or {}
    backups_raw = security_raw.get("backups") or {}

    # orphan_sweep
    orphan_raw = backups_raw.get("orphan_sweep") or {}
    orphan_cfg = OrphanSweepConfig(**orphan_raw) if orphan_raw else OrphanSweepConfig()

    # retention — per-tier dict; coerce keep="unlimited" string verbatim,
    # coerce numeric keep to int, coerce max_disk_gb to float when set.
    retention_raw = backups_raw.get("retention") or {}
    retention_cfg = RetentionConfig()
    for _tier_name in (
        "daily",
        "weekly",
        "monthly",
        "yearly",
        "pre_migration",
        "pre_base_swap",
        "trial_adapter",
        "manual",
    ):
        _tier_raw = retention_raw.get(_tier_name)
        if _tier_raw is None:
            continue
        _keep = _tier_raw.get("keep", getattr(retention_cfg, _tier_name).keep)
        _max_disk_gb = _tier_raw.get("max_disk_gb")
        if isinstance(_keep, str) and _keep != "unlimited":
            try:
                _keep = int(_keep)
            except ValueError:
                raise ValueError(
                    f"security.backups.retention.{_tier_name}.keep must be int "
                    f'or "unlimited"; got {_keep!r}'
                ) from None
        elif isinstance(_keep, float):
            _keep = int(_keep)
        setattr(
            retention_cfg,
            _tier_name,
            RetentionTierConfig(
                keep=_keep,
                max_disk_gb=float(_max_disk_gb) if _max_disk_gb is not None else None,
            ),
        )

    schedule = backups_raw.get("schedule", "daily 04:00")
    artifacts = list(backups_raw.get("artifacts", ["config", "graph", "registry"]))
    max_total_disk_gb = float(backups_raw.get("max_total_disk_gb", 20.0))

    # Validate artifacts — must be non-empty subset of {config, graph, registry}.
    _valid_artifacts = {"config", "graph", "registry"}
    for _art in artifacts:
        if _art not in _valid_artifacts:
            raise ValueError(
                f"security.backups.artifacts: invalid entry {_art!r}; "
                f"must be a subset of {sorted(_valid_artifacts)}"
            )
    if not artifacts:
        raise ValueError(f"security.backups.artifacts must be non-empty; got {artifacts!r}")

    config.security = SecurityConfig(
        backups=ServerBackupsConfig(
            orphan_sweep=orphan_cfg,
            retention=retention_cfg,
            schedule=schedule,
            artifacts=artifacts,
            max_total_disk_gb=max_total_disk_gb,
        )
    )

    return config
