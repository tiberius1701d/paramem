"""Authoritative dotted-path → migration tier map for ``configs/server.yaml``.

Every migration tool routes through :func:`classify` as the single
source of truth for tier assignment.  The lookup table is a module-level dict
literal — no file I/O, no YAML dependency — easy to audit by diff.

Scope: ``configs/server.yaml`` only (Resolved Decision 16).  Fields in
``configs/default.yaml`` (``training:``, ``replay:``, ``graph:``, ``wandb:``)
are experiment-script config, not server runtime, and are excluded.

Fallback rule (Resolved Decision 8): any path that is not present in
:data:`CLASSIFICATION` and not reachable via wildcard substitution is treated
as :attr:`Tier.DESTRUCTIVE`.  Fail-safe over fail-open.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Final, Iterator


class Tier(Enum):
    """Impact tier for a ``server.yaml`` field change.

    Glyph values are reused by the preview renderer so callers do not
    need a second lookup table.
    """

    DESTRUCTIVE = "destructive"
    PIPELINE_ALTERING = "pipeline_altering"
    OPERATIONAL = "operational"

    @property
    def glyph(self) -> str:
        """Single-character warning glyph for CLI output."""
        return _GLYPHS[self]


_GLYPHS: dict[Tier, str] = {
    Tier.DESTRUCTIVE: "⚠",
    Tier.PIPELINE_ALTERING: "~",
    Tier.OPERATIONAL: "·",
}

# ---------------------------------------------------------------------------
# Extension fields — not present in the shipped server.yaml but documented in
# docs/config_impact.md:180-184.  Referenced by test_no_orphan_classifications
# to avoid false positives when verifying that every CLASSIFICATION entry maps
# to a real yaml key or a declared extension.
# ---------------------------------------------------------------------------
EXTENSION_FIELDS: frozenset[str] = frozenset(
    {
        "adapters.*.target_modules",
        "adapters.*.dropout",
        # Planned pipeline flag — not in the shipped server.yaml yet but
        # documented in the consolidation block for future SOTA provider routing.
        "consolidation.extraction_noise_filter_endpoint",
    }
)

# ---------------------------------------------------------------------------
# Classification table — one entry per dotted path in shipped server.yaml,
# plus the extension fields above.  Wildcards use ``*`` in the dynamic segment
# position.
# ---------------------------------------------------------------------------
CLASSIFICATION: Final[dict[str, Tier]] = {
    # --- Top-level scalars ---
    "model": Tier.DESTRUCTIVE,
    "debug": Tier.PIPELINE_ALTERING,
    "cloud_only": Tier.DESTRUCTIVE,
    "headless_boot": Tier.OPERATIONAL,
    # --- paths ---
    "paths.data": Tier.DESTRUCTIVE,
    "paths.sessions": Tier.DESTRUCTIVE,
    "paths.prompts": Tier.DESTRUCTIVE,
    "paths.debug": Tier.OPERATIONAL,
    # --- server ---
    "server.host": Tier.OPERATIONAL,
    "server.port": Tier.OPERATIONAL,
    "server.reclaim_interval_minutes": Tier.OPERATIONAL,
    "server.vram_safety_margin_mb": Tier.OPERATIONAL,
    # --- adapters (wildcard for any adapter name) ---
    "adapters.*.enabled": Tier.DESTRUCTIVE,
    "adapters.*.rank": Tier.DESTRUCTIVE,
    "adapters.*.alpha": Tier.DESTRUCTIVE,
    "adapters.*.learning_rate": Tier.PIPELINE_ALTERING,
    # Extension fields (not in shipped yaml — documented in config_impact.md)
    "adapters.*.target_modules": Tier.DESTRUCTIVE,
    "adapters.*.dropout": Tier.DESTRUCTIVE,
    # --- consolidation ---
    "consolidation.refresh_cadence": Tier.PIPELINE_ALTERING,
    "consolidation.mode": Tier.PIPELINE_ALTERING,
    "consolidation.promotion_threshold": Tier.PIPELINE_ALTERING,
    "consolidation.retain_sessions": Tier.DESTRUCTIVE,
    "consolidation.indexed_key_replay": Tier.PIPELINE_ALTERING,
    "consolidation.decay_window": Tier.PIPELINE_ALTERING,
    "consolidation.max_interim_count": Tier.PIPELINE_ALTERING,
    "consolidation.post_session_train_enabled": Tier.PIPELINE_ALTERING,
    "consolidation.extraction_max_tokens": Tier.PIPELINE_ALTERING,
    "consolidation.extraction_stt_correction": Tier.PIPELINE_ALTERING,
    "consolidation.extraction_ha_validation": Tier.PIPELINE_ALTERING,
    "consolidation.extraction_noise_filter": Tier.PIPELINE_ALTERING,
    "consolidation.extraction_noise_filter_model": Tier.PIPELINE_ALTERING,
    "consolidation.extraction_noise_filter_endpoint": Tier.PIPELINE_ALTERING,
    "consolidation.extraction_plausibility_judge": Tier.PIPELINE_ALTERING,
    "consolidation.extraction_plausibility_stage": Tier.PIPELINE_ALTERING,
    "consolidation.extraction_verify_anonymization": Tier.PIPELINE_ALTERING,
    "consolidation.extraction_ner_check": Tier.PIPELINE_ALTERING,
    "consolidation.extraction_ner_model": Tier.PIPELINE_ALTERING,
    "consolidation.training_temp_limit": Tier.PIPELINE_ALTERING,
    "consolidation.training_temp_check_interval": Tier.PIPELINE_ALTERING,
    "consolidation.quiet_hours_mode": Tier.PIPELINE_ALTERING,
    "consolidation.quiet_hours_start": Tier.PIPELINE_ALTERING,
    "consolidation.quiet_hours_end": Tier.PIPELINE_ALTERING,
    "consolidation.entity_similarity_threshold": Tier.PIPELINE_ALTERING,
    "consolidation.graph_enrichment_enabled": Tier.PIPELINE_ALTERING,
    "consolidation.graph_enrichment_neighborhood_hops": Tier.PIPELINE_ALTERING,
    "consolidation.graph_enrichment_max_entities_per_pass": Tier.PIPELINE_ALTERING,
    "consolidation.graph_enrichment_interim_enabled": Tier.PIPELINE_ALTERING,
    "consolidation.graph_enrichment_min_triples_floor": Tier.PIPELINE_ALTERING,
    # --- agents.sota ---
    "agents.sota.enabled": Tier.PIPELINE_ALTERING,
    "agents.sota.provider": Tier.PIPELINE_ALTERING,
    "agents.sota.model": Tier.PIPELINE_ALTERING,
    "agents.sota.api_key": Tier.PIPELINE_ALTERING,
    "agents.sota.endpoint": Tier.PIPELINE_ALTERING,
    # --- agents.sota_providers (wildcard for any provider name) ---
    "agents.sota_providers.*.enabled": Tier.PIPELINE_ALTERING,
    "agents.sota_providers.*.provider": Tier.PIPELINE_ALTERING,
    "agents.sota_providers.*.model": Tier.PIPELINE_ALTERING,
    "agents.sota_providers.*.api_key": Tier.PIPELINE_ALTERING,
    "agents.sota_providers.*.endpoint": Tier.PIPELINE_ALTERING,
    # --- agents.ha_agent_id ---
    "agents.ha_agent_id": Tier.PIPELINE_ALTERING,
    # --- tools ---
    "tools.ha.url": Tier.OPERATIONAL,
    "tools.ha.token": Tier.OPERATIONAL,
    "tools.ha.auto_discover": Tier.OPERATIONAL,
    "tools.ha.supported_languages": Tier.OPERATIONAL,
    "tools.ha.allowlist": Tier.OPERATIONAL,
    "tools.tool_timeout_seconds": Tier.OPERATIONAL,
    # --- sanitization ---
    "sanitization.mode": Tier.OPERATIONAL,
    # --- abstention ---
    "abstention.enabled": Tier.PIPELINE_ALTERING,
    "abstention.response": Tier.PIPELINE_ALTERING,
    # --- voice ---
    "voice.prompt_file": Tier.OPERATIONAL,
    "voice.greeting_interval_hours": Tier.OPERATIONAL,
    # --- speaker ---
    "speaker.enabled": Tier.OPERATIONAL,
    "speaker.high_confidence_threshold": Tier.OPERATIONAL,
    "speaker.low_confidence_threshold": Tier.OPERATIONAL,
    "speaker.enrollment_prompt": Tier.OPERATIONAL,
    "speaker.enrollment_idle_timeout": Tier.OPERATIONAL,
    "speaker.enrollment_reprompt_interval": Tier.OPERATIONAL,
    "speaker.enrollment_check_interval": Tier.OPERATIONAL,
    "speaker.min_embedding_duration_seconds": Tier.OPERATIONAL,
    "speaker.max_embeddings_per_profile": Tier.OPERATIONAL,
    "speaker.redundancy_threshold": Tier.OPERATIONAL,
    "speaker.grouping_threshold_factor": Tier.OPERATIONAL,
    # --- stt ---
    "stt.enabled": Tier.OPERATIONAL,
    "stt.model": Tier.OPERATIONAL,
    "stt.cpu_fallback_model": Tier.OPERATIONAL,
    "stt.device": Tier.OPERATIONAL,
    "stt.compute_type": Tier.OPERATIONAL,
    "stt.port": Tier.OPERATIONAL,
    "stt.language": Tier.OPERATIONAL,
    "stt.beam_size": Tier.OPERATIONAL,
    "stt.vad_filter": Tier.OPERATIONAL,
    # --- tts ---
    "tts.enabled": Tier.OPERATIONAL,
    "tts.port": Tier.OPERATIONAL,
    "tts.device": Tier.OPERATIONAL,
    "tts.default_language": Tier.OPERATIONAL,
    "tts.language_confidence_threshold": Tier.OPERATIONAL,
    # --- tts.voices (wildcard for any language code) ---
    "tts.voices.*.engine": Tier.OPERATIONAL,
    "tts.voices.*.model": Tier.OPERATIONAL,
    "tts.voices.*.device": Tier.OPERATIONAL,
}


_DEFAULT_DYNAMIC_CONTAINERS: frozenset[str] = frozenset({"adapters", "sota_providers", "voices"})


def walk_dict_leaves(
    node: object,
    *,
    prefix: str = "",
    dynamic_containers: frozenset[str] = _DEFAULT_DYNAMIC_CONTAINERS,
) -> Iterator[tuple[str, Any]]:
    """Yield ``(dotted_path, value)`` for every leaf in a nested dict.

    This is the shared traversal helper used by both the tier-diff engine
    (``paramem.server.migration._walk_leaves``) and the shipped-path enumerator
    (``iter_shipped_paths``).  Extracting the logic here means a single change
    keeps both consumers consistent.

    Dynamic containers (``adapters``, ``sota_providers``, ``voices``) are
    traversed using their concrete child names — e.g.
    ``adapters.episodic.rank`` — not a wildcard form.  The :func:`classify`
    function handles wildcard expansion internally.

    Parameters
    ----------
    node:
        The current YAML node (dict, list, or scalar).
    prefix:
        Accumulated dotted-path prefix.  Pass ``""`` at the root call.
    dynamic_containers:
        Names of container keys whose children are concrete dynamic names
        (not sub-keys of the parent schema).  Defaults to
        ``{"adapters", "sota_providers", "voices"}``.

    Yields
    ------
    tuple[str, Any]
        ``(dotted_path, value)`` for each leaf node in dict-insertion order.
        Non-dict nodes at the root yield ``(prefix, node)`` directly.
    """
    if not isinstance(node, dict):
        yield prefix, node
        return
    for key, value in node.items():
        child = f"{prefix}.{key}" if prefix else key
        parent_key = prefix.split(".")[-1] if prefix else ""
        if parent_key in dynamic_containers or isinstance(value, dict):
            yield from walk_dict_leaves(value, prefix=child, dynamic_containers=dynamic_containers)
        else:
            yield child, value


def classify(dotted_path: str) -> Tier:
    """Return the migration tier for a dotted ``server.yaml`` path.

    Lookup order:

    1. Exact match in :data:`CLASSIFICATION`.
    2. Wildcard match — rewrite the **middle** segment to ``*`` and retry.
       Middle-first because every current dynamic yaml shape
       (``adapters.<name>.<subkey>``, ``agents.sota_providers.<name>.<subkey>``,
       ``tts.voices.<name>.<subkey>``) places the dynamic segment in a middle
       position.  Last-first would always miss on the first attempt.
    3. Wildcard match — rewrite the **last** segment to ``*`` and retry.
       Defensive; no current yaml shape requires it, but keeps the lookup
       complete if a future flat dynamic key is added.
    4. Fall through to :attr:`Tier.DESTRUCTIVE` (Resolved Decision 8).

    Parameters
    ----------
    dotted_path:
        A dotted key path as it appears in ``server.yaml``
        (e.g. ``"adapters.episodic.rank"``).

    Returns
    -------
    Tier
        The impact tier for the given path.  Never raises — unknown paths
        return :attr:`Tier.DESTRUCTIVE` by design.
    """
    if dotted_path in CLASSIFICATION:
        return CLASSIFICATION[dotted_path]

    parts = dotted_path.split(".")
    if len(parts) >= 3:
        # middle-first: replace second-to-last named segment with *
        mid_idx = len(parts) - 2
        wildcard_parts = parts[:mid_idx] + ["*"] + parts[mid_idx + 1 :]
        wildcard_path = ".".join(wildcard_parts)
        if wildcard_path in CLASSIFICATION:
            return CLASSIFICATION[wildcard_path]

    if len(parts) >= 2:
        # last segment wildcard: defensive path for future flat dynamic keys
        wildcard_parts = parts[:-1] + ["*"]
        wildcard_path = ".".join(wildcard_parts)
        if wildcard_path in CLASSIFICATION:
            return CLASSIFICATION[wildcard_path]

    return Tier.DESTRUCTIVE


def iter_shipped_paths(server_yaml_path: Path) -> Iterator[str]:
    """Yield every top-level dotted path that appears in the given
    ``server.yaml``.

    Used by the classification unit test to drive coverage assertions without
    hard-coding a second path list.

    Recursion rules (match yaml structure, not classification rules):

    - Leaf scalars and lists yield the containing dotted path.
    - Dicts recurse.
    - ``adapters.*``, ``agents.sota_providers.*``, and ``tts.voices.*`` are
      dynamic-key containers — the walker yields the concrete
      ``adapters.<name>.<subkey>`` form.  The test-side assertion verifies that
      the wildcard form classifies each concrete name, not that the concrete
      form itself appears in :data:`CLASSIFICATION`.

    Implemented as a thin wrapper around :func:`walk_dict_leaves`.

    Parameters
    ----------
    server_yaml_path:
        Filesystem path to the ``server.yaml`` to walk.

    Yields
    ------
    str
        Dotted path string for each leaf in the yaml tree.
    """
    import yaml

    with open(server_yaml_path) as fh:
        data = yaml.safe_load(fh)

    for path, _ in walk_dict_leaves(data):
        yield path
