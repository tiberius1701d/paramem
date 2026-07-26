"""Configuration loading and validation — archived training-pipeline loader,
plus a live per-fold training-budget derivation.

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

``budget_for`` (and the ``_BUDGET_TABLE`` it reads) is NOT archived: it is a
live production function, called from the consolidation training funnel
(``paramem.training.consolidation.ConsolidationLoop._train_tier_adapter``)
on every fold to derive the per-fold epoch / gradient-accumulation /
LR-decay budget from the training set's key-triple count.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from paramem.utils.paths import find_project_root

# Valid values for refinement_enrichment and refinement_normalization.
# Shared between utils.config and server.config (server.config imports this).
_REFINEMENT_TOGGLES: frozenset[str] = frozenset({"off", "on"})


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
    # Absolute-step warmup count, passed straight to HF TrainingArguments as
    # the sole warmup knob. There is deliberately no ratio-based sibling
    # field — that HF field is deprecated, banned by project rule, and was
    # the mechanism that silently zeroed warmup in every fold ever trained
    # under transformers 5.5 (see paramem.training.trainer's
    # TrainingArguments call). 0 (default) means no warmup.
    warmup_steps: int = 0
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


# Per-fold training budget table. Ordered by descending key-count floor; the
# first row whose floor `n_keys` meets is the bucket. `batch` is not part of
# the table — it stays 1 everywhere (TrainingConfig.batch_size /
# ConsolidationScheduleConfig.training_batch_size default). Epoch counts are
# CAPS, not targets: when TrainingConfig.recall_early_stopping is on, the
# recall gate terminates training before the cap in the common case.
# `lr_decay_steps` defaults to None for every bucket (today's behaviour:
# create_scheduler's no-op passthrough; see TrainingConfig.lr_decay_steps).
#
# See configs/server.yaml.example for the per-bucket epoch/accum values and
# their evidence status. The largest bucket is anchored on production fold
# telemetry. The 16-127 bucket (50 epochs) is now anchored at N=21 (4 seeds,
# 21/21 exact-match at 50 epochs / 550 decay-pinned steps) — see
# benchmarking.md's "Test 20: Small-N Cold-Init Recall Gate" section; the
# rest of the 16-127 band and the <16 bucket remain extrapolated.
_BUDGET_TABLE: tuple[tuple[int, int, int, "int | None"], ...] = (
    # (n_keys floor, epochs, accum, lr_decay_steps)
    (128, 30, 2, None),
    (16, 50, 2, None),
    (0, 80, 1, None),
)


def budget_for(n_keys: int) -> "tuple[int, int, int | None]":
    """Derive the per-fold training budget from the key-triple count.

    The unconditional standard mechanism (validated via Test 20; the prior
    ``budget_derivation_enabled`` feature flag was retired once the
    validation arms passed -- see ``benchmarking.md``). Pure, module-level,
    and callable independently of whether training has run (see the call
    sites in ``paramem.training.consolidation``, which invoke this in the
    enclosing scope BEFORE the training ``try`` block so the
    ``finally``-path telemetry records the true budget even when training
    raises).

    The bucket table governs unclamped -- there is no operator ceiling. A
    prior ``TrainingConfig.budget_max_epochs`` / ``ConsolidationScheduleConfig
    .max_epochs`` clamp was retired 2026-07-26; wall-time feedback now comes
    from ``hit_cap`` telemetry and recall-based early stopping, not a config
    clamp. The clamp's removal also retired the ``training_config`` parameter
    itself (2026-07-26) -- the derivation reads nothing off it anymore.

    Args:
        n_keys: number of key-triples (entries) in the training set for this
            fold -- ``len(entries)`` at the call site.

    Returns:
        ``(epochs, gradient_accumulation_steps, lr_decay_steps)`` from the
        bucket in ``_BUDGET_TABLE`` matching ``n_keys``, applied as-is.
    """
    # Single lookup: the first row whose floor n_keys meets or exceeds. The
    # table's last row has floor=0, so this always matches for n_keys >= 0
    # (len() of a list is never negative) -- no default fallback needed.
    epochs, accum, lr_decay_steps = next(
        (row_epochs, row_accum, row_lr_decay)
        for row_floor, row_epochs, row_accum, row_lr_decay in _BUDGET_TABLE
        if n_keys >= row_floor
    )
    return epochs, accum, lr_decay_steps


@dataclass
class GraphConfig:
    extraction_temperature: float = 0.3
    max_extraction_tokens: int = 1024
    entity_similarity_threshold: float = 85.0


@dataclass
class ConsolidationConfig:
    promotion_threshold: int = 3
    decay_window: int = 10
    indexed_key_replay: bool = True
    # Ship-safe posture: base defaults OFF (no refinement). Operator YAMLs
    # (fixture/local) opt in explicitly.
    # The cloud master switch is NOT here: it is `cloud.enabled`
    # (paramem.server.config.CloudConfig), the single switch for all cloud
    # egress, and reaches ConsolidationLoop as an explicit constructor
    # argument.  A second copy on this dataclass is what let the pipeline and
    # the conversation agent disagree about whether cloud was on.
    # graph-stage GraphTierRefiner.run_enrichment (cloud-only). off|on
    refinement_enrichment: str = "off"
    refinement_normalization: str = "on"  # full-fold predicate-synonym collapse. off|on
    # Whether the merger resolves same-predicate/different-object cardinality
    # conflicts (Case-2 COEXIST/REPLACE) at ingest, interim, and fold.
    # Unified recency rule (REPLACE cardinality): an empty last_seen ("") sorts as
    # the oldest possible timestamp, so a dated candidate always outranks an
    # undated one.  Coexist fires ONLY when EVERY candidate (incoming + all
    # rivals) is undated (covers fully-legacy registries with no recency signal
    # anywhere).  Otherwise at least one candidate is dated: strictly-older
    # rivals (including undated ones) are retired; ties at the max timestamp
    # coexist; incoming loses if strictly older than the winner.  The model
    # determines the cardinality axis (COEXIST vs REPLACE); recency fires only
    # when REPLACE is returned.  Applied uniformly at ingest, interim, and fold;
    # no positional fork.  off|on.
    refinement_contradiction: str = "off"
    # Minimum recall fraction (0, 1] that every recall gate must reach before the
    # adapter fold is accepted.  Applied to: post-save disk-integrity probes
    # (_verify_saved_adapter_from_disk), interim-cycle training
    # (run_consolidation_cycle), the full fold (ConsolidationLoop.consolidate),
    # and simulate→train migration (_migrate_tier_simulate_to_train).
    # 1.0 = sharp recall (all keys must be recalled, no tolerance); lower only with
    # empirical evidence and explicit operator acknowledgment.
    recall_sanity_threshold: float = 1.0

    def __post_init__(self) -> None:
        """Validate field values at construction time."""
        for _toggle_field in (
            "refinement_enrichment",
            "refinement_normalization",
            "refinement_contradiction",
        ):
            _val = getattr(self, _toggle_field)
            if _val not in _REFINEMENT_TOGGLES:
                raise ValueError(
                    f"ConsolidationConfig.{_toggle_field} must be one of "
                    f"{sorted(_REFINEMENT_TOGGLES)!r}; got {_val!r}"
                )
        if not (0.0 < self.recall_sanity_threshold <= 1.0):
            raise ValueError(
                f"ConsolidationConfig.recall_sanity_threshold must be in (0.0, 1.0]; "
                f"got {self.recall_sanity_threshold!r}"
            )

    # Floor below which a tier's keys park in episodic until the tier's own
    # population reaches this count.  30 is an UNVALIDATED conservative default —
    # no experiment on the production model establishes it (calibration-pending).
    # The floor is general: no consolidation trigger bypasses it.
    min_tier_key_floor: int = 30
    # Graduation strategy when a parked tier first crosses min_tier_key_floor.
    # True = copy episodic's LoRA weights into the graduating tier and rebook
    # the registry (training-free fast-start; default).  False = train the tier
    # from scratch on its own now-large key set (principled baseline, opt-out).
    # Governs semantic/procedural only; episodic always trains from scratch.
    tier_fast_start: bool = True


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
        config_path = (
            (find_project_root(Path(__file__)) or Path(__file__).resolve().parents[2])
            / "archive"
            / "configs"
            / "default.yaml"
        )
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

    return ParaMemConfig(
        model=model,
        adapters=adapters,
        training=training,
        graph=graph,
        consolidation=consolidation,
        wandb=wandb_cfg,
        paths=paths,
    )
