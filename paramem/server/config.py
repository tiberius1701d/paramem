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
        of ``keep``.  Spec §L572 (rule 2).
    """

    keep: int | str = 7  # int OR Literal["unlimited"]; YAML loader coerces
    max_disk_gb: float | None = None


@dataclass
class RetentionConfig:
    """Per-tier retention configuration.  Spec §L552–566."""

    daily: RetentionTierConfig = field(default_factory=lambda: RetentionTierConfig(keep=7))
    weekly: RetentionTierConfig = field(default_factory=lambda: RetentionTierConfig(keep=4))
    monthly: RetentionTierConfig = field(default_factory=lambda: RetentionTierConfig(keep=12))
    yearly: RetentionTierConfig = field(default_factory=lambda: RetentionTierConfig(keep=3))
    pre_migration: RetentionTierConfig = field(default_factory=lambda: RetentionTierConfig(keep=10))
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
    max_total_disk_gb: float = 20.0  # global cap across all tiers (spec §L566, L571)
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
    # KV-cache + activation headroom passed to assess_topology and the lifespan
    # VRAM gate. Code default 1.0 GiB is the conservative minimum for callers
    # that don't load a yaml; production yaml ships 2.0 GiB (configs/server.yaml
    # and tests/fixtures/server.yaml) for the per-phase peak. vram_scope's
    # inter-phase empty_cache releases the allocator pool between phases, so
    # the reservation only needs to cover one phase's max KV growth.
    vram_cache_headroom_gib: float = 1.0

    def __post_init__(self) -> None:
        if not (0.0 < self.process_cap_fraction <= 1.0):
            raise ValueError(
                f"vram.process_cap_fraction must be in (0, 1]; got {self.process_cap_fraction!r}"
            )
        if self.vram_cache_headroom_gib <= 0:
            raise ValueError(
                f"vram.vram_cache_headroom_gib must be > 0; got {self.vram_cache_headroom_gib!r}"
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
    # Maximum LoRA training epochs per consolidation cycle. None = use the
    # validated 30 from VALIDATED_TRAINING_CONFIG (the Test 1-8 campaign
    # floor for 100% indexed-key recall on the validated models).
    #
    # Override semantics: ceiling, not target. Once the recall-driven
    # early-stop callback (project_recall_early_stop_design.md) ships,
    # training will terminate at the recall plateau; this field caps the
    # upper bound. Until then, training runs the full configured count.
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
    # Recall-based early stopping (default OFF — see plan-recall-early-stop-online-v3.md).
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
    # Mini-enrichment at interim-adapter rollover (per sub-interval, default 12h).
    # Amortises enrichment cost across the 84h full-consolidation cycle. Gated
    # on a minimum triple-floor so low-traffic sub-intervals skip the pass.
    graph_enrichment_interim_enabled: bool = True
    graph_enrichment_min_triples_floor: int = 20

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
        """
        from paramem.memory.interim_adapter import compute_schedule_period_seconds

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
        """Validated training config from test campaign.

        ``num_epochs`` honours ``consolidation.max_epochs`` when set,
        otherwise falls back to the validated default. The override is
        a ceiling — once recall-driven early-stop ships, training will
        terminate at the recall plateau bounded by this value.
        """
        return TrainingConfig(
            batch_size=VALIDATED_TRAINING_CONFIG.batch_size,
            gradient_accumulation_steps=VALIDATED_TRAINING_CONFIG.gradient_accumulation_steps,
            max_seq_length=VALIDATED_TRAINING_CONFIG.max_seq_length,
            num_epochs=(
                self.consolidation.max_epochs
                if self.consolidation.max_epochs is not None
                else VALIDATED_TRAINING_CONFIG.num_epochs
            ),
            warmup_ratio=VALIDATED_TRAINING_CONFIG.warmup_ratio,
            weight_decay=VALIDATED_TRAINING_CONFIG.weight_decay,
            gradient_checkpointing=VALIDATED_TRAINING_CONFIG.gradient_checkpointing,
            max_grad_norm=VALIDATED_TRAINING_CONFIG.max_grad_norm,
            seed=VALIDATED_TRAINING_CONFIG.seed,
            # logging_steps lives on TrainingConfig (default 1, matches the
            # historical train_adapter hardcode).  The BG-trainer call site
            # uses dataclasses.replace to override to 10 when delegating to
            # train_adapter, preserving its prior log volume; the
            # consolidation/server path inherits the default.
            logging_steps=VALIDATED_TRAINING_CONFIG.logging_steps,
            early_stopping_floor=self.consolidation.recall_signal_from_epoch,
            recall_early_stopping=self.consolidation.recall_early_stopping,
            recall_window=self.consolidation.recall_window,
            recall_probe_every_n_epochs=self.consolidation.recall_probe_every_n_epochs,
            recall_probe_batch_size=self.consolidation.recall_probe_batch_size,
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
