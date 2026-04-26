"""ParaMem server — REST API wrapping the parametric memory pipeline.

Usage:
    python -m paramem.server.app --config configs/server.yaml

GPU lifecycle (service-level):
    Stop service to free GPU, restart to reclaim.
    --defer-model: start without GPU model, auto-reclaim when GPU is free.
    --cloud-only: permanent cloud-only mode, no auto-reclaim.
"""

import argparse
import asyncio
import hashlib as _hashlib
import json
import logging
import os
import secrets
import shutil
import signal
import subprocess
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import torch
from dotenv import load_dotenv
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Migration / backup imports at module level so tests can patch them.
from paramem.backup.backup import write as backup_write
from paramem.backup.types import ArtifactKind
from paramem.models.loader import load_base_model, switch_adapter, unload_model
from paramem.server.cloud import get_cloud_agent
from paramem.server.config import TTSConfig, load_server_config
from paramem.server.consolidation import run_consolidation
from paramem.server.ha_graph import HAEntityGraph
from paramem.server.inference import (
    ChatResult,
    _escalate_to_sota,
    enqueue_post_session_train,
    handle_chat,
)
from paramem.server.post_session_queue import PostSessionQueue
from paramem.server.router import QueryRouter
from paramem.server.session_buffer import SessionBuffer  # "session" here = conversation
from paramem.server.tools.ha_client import HAClient
from paramem.server.trial_state import (
    TrialMarker,
    clear_trial_marker,
    write_trial_marker,
)
from paramem.server.vram_validator import (
    ConfigurationError,
    assess_topology,
    enforce_live_budget,
    estimate_stt_bytes,
    estimate_tts_bytes,
    format_tier_table,
    measure_external_vram,
)
from paramem.utils.notify import SERVER_CLOUD_ONLY, notify_server

logger = logging.getLogger(__name__)


def _rename_config(src: "str | Path", dst: "str | Path") -> None:
    """Rename *src* to *dst* for the atomic config swap in step 4 of confirm.

    Extracted to a module-level function so integration tests can patch it
    independently from os.rename (which is also used by backup.atomic and
    trial_state for their own atomic renames).

    Parameters
    ----------
    src:
        Source path (candidate config file).
    dst:
        Destination path (live config file).
    """
    os.rename(src, dst)


# Resolve nvidia-smi at import time. On WSL2 it lives in /usr/lib/wsl/lib
# which systemd may not have on PATH; on native Linux it's typically in
# /usr/bin.  Falls back to bare name (will fail with FileNotFoundError,
# caught by the caller).
_NVIDIA_SMI = (
    shutil.which("nvidia-smi")
    or shutil.which("nvidia-smi", path="/usr/lib/wsl/lib:/usr/bin:/usr/local/bin")
    or "nvidia-smi"
)

# Global state — single model, single adapter, single server
_state = {
    "model": None,
    "tokenizer": None,
    "config": None,
    "session_buffer": None,
    "router": None,
    "sota_agent": None,
    "sota_providers": {},
    "ha_client": None,
    "consolidation_loop": None,
    "consolidating": False,
    "last_consolidation": None,
    "background_trainer": None,
    "reclaim_task": None,
    "config_path": None,
    "config_drift_task": None,
    "post_session_queue": None,  # PostSessionQueue instance (local mode only)
    "mode": "local",  # "local" or "cloud-only"
    "cloud_only_reason": None,  # "explicit", "training", "gpu_conflict", or None
    "cloud_only_startup": False,  # set by --cloud-only CLI flag before app start
    "defer_model": False,  # set by --defer-model CLI flag before app start
    "ha_graph": None,  # HAEntityGraph built from HA states/services at startup
    "event_loop": None,  # asyncio event loop reference for cross-thread scheduling
    "speaker_store": None,
    "stt": None,
    "wyoming_server": None,
    "tts_manager": None,
    "wyoming_tts_server": None,
    "latest_embedding": None,
    "latest_language_detection": None,  # {language: str, probability: float}
    "last_chat_time": None,
    "pending_enrollments": set(),
    # Unknown speaker groups: temp_id → {embeddings, conversations, first_seen}.
    # Mutations happen on the asyncio event loop (cooperative scheduling).
    # Safe without locks.
    "unknown_speakers": {},
    "migration": None,  # MigrationStashState — populated in lifespan
    "server_started_at": "",  # ISO-8601 UTC timestamp set in lifespan
}


# --- Request/Response schemas ---


class ChatRequest(BaseModel):
    text: str
    conversation_id: str = "default"
    speaker: str | None = None
    speaker_embedding: list[float] | None = None  # Voice embedding from STT
    history: list[dict] | None = None
    route: str | None = None  # Force routing: "ha", "sota", or None (auto)


class ChatResponse(BaseModel):
    text: str
    escalated: bool = False
    speaker: str | None = None
    follow_up: str | None = None  # Server-initiated follow-up (e.g. introduction)


class BackupBlock(BaseModel):
    """Backup subsystem state for /status.  Spec §L485–500.

    All fields default to ``None`` / ``0`` / ``False`` so a never-run server
    (no ``state/backup.json``) still serialises a valid block.

    Attributes
    ----------
    schedule:
        The configured backup schedule string (e.g. ``"daily 04:00"``).  Added
        beyond the spec to let pstatus choose the rendering branch without
        re-fetching the config.
    last_success_at:
        ISO-8601 UTC timestamp of the most recent successful backup run.
    last_failure_at:
        ISO-8601 UTC timestamp of the most recent failed backup run.
    last_failure_reason:
        Short error string from the most recent failure.
    next_scheduled_at:
        ISO-8601 UTC timestamp of the next scheduled run (read from the live
        systemd timer state).
    stale:
        ``True`` when ``last_success_at`` is older than 2× the configured
        cadence interval.  Always ``False`` when ``schedule="off"`` or
        ``last_success_at is None``.
    disk_used_bytes:
        Total bytes used across all backup slots.
    disk_cap_bytes:
        Global disk cap in bytes (``max_total_disk_gb * 1024**3``).
    """

    schedule: str = ""
    last_success_at: str | None = None
    last_failure_at: str | None = None
    last_failure_reason: str | None = None
    next_scheduled_at: str | None = None
    stale: bool = False
    disk_used_bytes: int = 0
    disk_cap_bytes: int = 0


class StatusResponse(BaseModel):
    model: str
    # Full HF model identifier (e.g. "mistralai/Mistral-7B-Instruct-v0.3").
    # Lets pstatus surface the variant alongside the short name.
    model_id: str | None = None
    model_device: str | None = None  # cuda / cpu / None (cloud-only)
    mode: str  # "local" or "cloud-only"
    cloud_only_reason: str | None  # "explicit", "training", "gpu_conflict", or None
    adapter_loaded: bool  # legacy: True when episodic main adapter is loaded
    # Rank of the episodic LoRA adapter (load-bearing for indexed-key recall).
    # None when no adapter is configured (cloud-only or all kinds disabled).
    episodic_rank: int | None = None
    # Per-kind adapter spec. Episodic / semantic / procedural can diverge in
    # learning rate and target modules (procedural adds MLP targets for
    # representational imprint), so each kind gets its own row. Shape:
    #   {kind: {"rank", "alpha", "learning_rate", "target_kind"}}
    # "target_kind" is "attn" when target_modules are attention-only, or
    # "attn+mlp" when MLP layers are included.
    adapter_specs: dict = {}
    # Adapter inventory: kind → configured count (kinds absent with count==0).
    # Main kinds (episodic/semantic/procedural) contribute 1 when enabled in
    # yaml; "interim" contributes max_interim_count.
    adapter_config: dict[str, int] = {}
    # Name of the currently active adapter on the live PeftModel, or None
    # when no adapters are loaded (fresh install / cloud-only).
    active_adapter: str | None = None
    keys_count: int
    pending_sessions: int
    consolidating: bool
    last_consolidation: str | None
    speaker_profiles: int = 0
    # Speaker-embedding backend (pyannote) + HF model id + device. None in
    # three cases: speaker id disabled in yaml, pyannote not installed, or
    # the model failed to load at startup.
    speaker_embedding_backend: str | None = None
    speaker_embedding_model: str | None = None
    speaker_embedding_device: str | None = None
    stt_loaded: bool = False
    stt_model: str | None = None
    stt_device: str | None = None  # cuda / cpu / None (unloaded)
    stt_engine: str | None = None  # "whisper" — backend family, currently fixed
    tts_loaded: bool = False
    tts_languages: list[str] = []  # loaded TTS voices by language code
    tts_device: str | None = None  # cuda / cpu / mixed / None (unloaded)
    # Backend family across loaded voices: "piper", "mms_tts", or "piper+mms"
    # when voices span both. None when no TTS is loaded.
    tts_engine: str | None = None
    # Interim cadence knob + derived full-cycle period.
    refresh_cadence: str = ""
    consolidation_period: str = ""  # refresh_cadence × max_interim_count
    max_interim_count: int = 0
    mode_config: str = ""  # "train" or "simulate"
    next_run_seconds: int | None = None  # seconds until next FULL consolidation
    # Seconds until the next interim cadence boundary (post_session_train
    # rolls over to a new stamp at every such boundary). None when refresh
    # cadence is disabled.
    next_interim_seconds: int | None = None
    orphaned_pending: int = 0  # pending sessions without speaker_id
    oldest_pending_seconds: int | None = None
    speakers: list[dict] = []  # [{id, name, embeddings, pending, enroll_method}]
    bg_trainer_active: bool = False
    bg_trainer_adapter: str | None = None
    last_consolidation_result: dict | None = None  # last completed run summary
    pending_enrollments: int = 0  # unknown speakers awaiting name extraction
    scheduler_started: bool = False  # True once scheduler first ticked
    # adapter_id -> {status, reason, updated_at, keys_at_mark}
    adapter_health: dict = {}
    # Per-adapter manifest-provenance rows (distinct from adapter_health).
    # Keyed by adapter name; empty when all manifests are healthy.
    # Schema per row: {status, reason, field, severity, slot_path, checked_at}.
    # Populated by _mount_adapters_from_slots at startup; surfaced to /status.
    adapter_manifest: dict = {}
    # Thermal-throttle / quiet-hours policy snapshot.
    # mode: "always_on" | "always_off" | "auto"
    # start/end: "HH:MM" local (populated for all modes, consumed only when mode=auto)
    # currently_throttling: true iff the thermal throttle is active right now
    thermal_policy: dict = {}
    # Config drift state: {detected, loaded_hash, disk_hash, last_checked_at}.
    # Populated after startup; empty dict when server started without a config path
    # (cloud-only test mode).
    config_drift: dict = {}
    # Deferred-mode GPU hold (PARAMEM_EXTRA_ARGS=--defer-model in systemd --user
    # env).  Set by gpu_guard / tresume when an ML workload wants the GPU; the
    # server stays cloud-only until the holder clears it.  Surfaces the owner
    # PID + liveness so an operator can spot orphaned holds (SIGKILLed test
    # processes) and clear with ``pstatus --force-local``.  Schema:
    #   {hold_active, owner_pid, owner_alive, age_seconds}
    hold: dict = {}
    # Operator-attention block. Always present; ``items`` is empty when no
    # alert is active. Each item is the dict form of an ``AttentionItem``
    # dataclass: {kind, level, summary, action_hint, age_seconds}.
    # See ``paramem.server.attention``.
    attention: dict = {}
    # Migration summary block. Always present; values reflect the
    # current migration state. Sub-fields:
    #   state          : "live" | "staging" | "trial" | "failed"
    #   config_rev     : 8-char prefix of sha256(server.yaml at load time)
    #   trial_started_at : ISO-8601 UTC, or None when not in TRIAL
    #   gates          : copy of _state["migration"]["trial"]["gates"], or None
    #   comparison     : {"rendered": bool, "flags": list[str]} or None
    migration: dict = {}
    # Backup subsystem state. Always present; fields default to
    # None/0/False when no scheduled backup has run yet.
    backup: BackupBlock = BackupBlock()
    # Startup security posture — "on" when the daily age identity loaded at
    # lifespan entry, "off" otherwise. Mirrors the SECURITY: ON/OFF startup
    # log line selected by security_posture.security_posture_log_line.
    encryption: str = "off"
    # ISO-8601 UTC timestamp of when the server process started.
    # Required so pstatus can render the "applied <YYYY-MM-DD>" part of the
    # Migrate footer (spec L458).
    server_started_at: str = ""
    # Document-ingest state (Phase 2+).
    # preview: true while a POST /consolidate?preview=true call is mid-flight.
    # Phase 2 always returns False; Phase 3 will toggle it.
    preview: bool = False
    # Pending session counts split by source type. Populated from
    # session_buffer.get_summary()["per_source_type"].
    pending_documents: int = 0
    pending_transcripts: int = 0


class ConsolidateResponse(BaseModel):
    status: str


# --- Document ingest schemas ---


class IngestChunk(BaseModel):
    """One pre-chunked document segment posted by the ingest CLI.

    Attributes
    ----------
    source:
        Original file path — display-only; the server never re-reads it.
    chunk:
        The text content of this chunk.
    chunk_index:
        Zero-based position of this chunk within the source file.
    source_type:
        Fixed to ``"document"`` for all ingest-CLI payloads.
    doc_title:
        Human-readable document title (filename stem); used for registry
        traceability and ``GET /status`` attribution.
    """

    source: str
    chunk: str
    chunk_index: int
    source_type: Literal["document"]
    doc_title: str


class IngestSessionsRequest(BaseModel):
    """Request body for ``POST /ingest-sessions``.

    Attributes
    ----------
    speaker_id:
        Known speaker identifier from ``SpeakerStore``.  Must be non-empty
        and must match an enrolled profile; the endpoint returns 400 / 404
        otherwise.
    sessions:
        List of pre-chunked document segments to enqueue.
    """

    speaker_id: str
    sessions: list[IngestChunk]


class IngestSessionsResponse(BaseModel):
    """Response body for ``POST /ingest-sessions``.

    Attributes
    ----------
    queued:
        Session IDs appended to the ``SessionBuffer`` (form ``doc-<hex8>``).
    total_chunks:
        Always equals ``len(request.sessions)`` — the raw count before
        idempotency filtering.
    registry_skipped:
        Chunks whose hash was already recorded in the registry; they were
        not re-queued.
    rejected_unknown_speaker:
        ``True`` when the speaker_id is not in ``SpeakerStore``.
    rejected_no_speaker_id:
        ``True`` when ``speaker_id`` is an empty string.
    """

    queued: list[str]
    total_chunks: int
    registry_skipped: int
    rejected_unknown_speaker: bool = False
    rejected_no_speaker_id: bool = False


class IngestCancelRequest(BaseModel):
    """Request body for ``POST /ingest-sessions/cancel``.

    Attributes
    ----------
    session_ids:
        Session IDs to remove from the ``SessionBuffer``.
    """

    session_ids: list[str]


class IngestCancelResponse(BaseModel):
    """Response body for ``POST /ingest-sessions/cancel``.

    Attributes
    ----------
    cancelled:
        Session IDs that were present and successfully discarded.
    not_found:
        Session IDs that were not found in the buffer (no-op).
    """

    cancelled: list[str]
    not_found: list[str]


# --- Migration schemas ---


class TierDiffRow(BaseModel):
    """One row in the tier-classified field-change list.

    Attributes
    ----------
    dotted_path:
        Dotted yaml key path (e.g. ``"adapters.episodic.rank"``).
    old_value:
        Value in the live config, or ``None`` when the field is new.
    new_value:
        Value in the candidate config, or ``None`` when the field is removed.
    tier:
        Impact tier string: ``"destructive"`` / ``"pipeline_altering"``
        / ``"operational"``.
    """

    dotted_path: str
    old_value: object = None
    new_value: object = None
    tier: str


class ShapeChange(BaseModel):
    """One field-level LoRA shape delta for a single adapter.

    Attributes
    ----------
    adapter:
        Adapter name (e.g. ``"episodic"``).
    field:
        LoRA field name: ``"rank"``, ``"alpha"``, ``"target_modules"``,
        or ``"dropout"``.
    old_value:
        Value in the on-disk ``meta.json``, or ``None`` when unavailable.
    new_value:
        Value requested by the candidate config.
    consequence:
        Human-readable consequence string (spec §L257–271).
    """

    adapter: str
    field: str
    old_value: object = None
    new_value: object = None
    consequence: str


class PreviewRequest(BaseModel):
    """Request body for ``POST /migration/preview``.

    Attributes
    ----------
    candidate_path:
        Absolute local filesystem path to the candidate ``server.yaml``.
    """

    candidate_path: str


class PreviewResponse(BaseModel):
    """Response body for ``POST /migration/preview`` and ``GET /migration/diff``.

    Attributes
    ----------
    state:
        ``"STAGING"`` after a successful preview.
    candidate_path:
        Echo of the validated candidate path.
    candidate_hash:
        Full hex SHA-256 of the candidate file bytes.
    staged_at:
        ISO-8601 UTC timestamp when STAGING was entered.
    simulate_mode_override:
        ``True`` when the candidate sets ``consolidation.mode: simulate``.
    unified_diff:
        Unified diff of live vs candidate YAML text.
    tier_diff:
        Tier-classified change rows (destructive first).
    shape_changes:
        Shape-change rows for enabled adapters with on-disk meta.json.
    pre_flight_fail:
        ``None`` when no pre-flight check fires; ``"disk_pressure"`` (or
        similar) when a pre-flight check rejects the preview.  Always present
        in the response so callers can check the field unconditionally
        (Condition 3).
    """

    state: str
    candidate_path: str
    candidate_hash: str
    staged_at: str
    simulate_mode_override: bool
    unified_diff: str
    tier_diff: list[TierDiffRow]
    shape_changes: list[ShapeChange]
    pre_flight_fail: str | None = None
    pre_flight_disk_used_gb: float | None = None
    pre_flight_disk_cap_gb: float | None = None


# MigrationDiffResponse is an alias — same shape as PreviewResponse.
MigrationDiffResponse = PreviewResponse


class MigrationStatusResponse(BaseModel):
    """Response body for ``GET /migration/status``.

    Attributes
    ----------
    state:
        ``"LIVE"``, ``"STAGING"``, or ``"TRIAL"``.
    candidate_path:
        Path of the staged candidate, or ``None`` when LIVE.
    candidate_hash:
        SHA-256 of the staged candidate, or ``None`` when LIVE.
    staged_at:
        ISO-8601 UTC timestamp when STAGING was entered, or ``None``.
    simulate_mode_override:
        ``True`` when the staged candidate sets ``consolidation.mode: simulate``.
    consolidating:
        ``True`` when a consolidation run is currently in progress.
    server_started_at:
        ISO-8601 UTC timestamp when the server lifespan started (Condition 6).
    trial_started_at:
        ISO-8601 UTC timestamp when TRIAL was entered, or ``None``.
    pre_trial_config_sha256:
        SHA-256 of the live config before the atomic rename, or ``None``.
    candidate_config_sha256:
        SHA-256 of the candidate config, or ``None``.
    backup_paths:
        Dict of absolute backup slot paths, or ``None``.
    trial_adapter_dir:
        Absolute path to the trial adapter directory, or ``None``.
    trial_graph_dir:
        Absolute path to the trial graph directory, or ``None``.
    gates:
        Trial gate status dict (``{"status": "pending"|"no_new_sessions"|
        "trial_exception", ...}``), or ``None``.
    recovery_required:
        Human-readable rows populated when AMBIGUOUS recovery was detected.
        Empty list otherwise.
    """

    state: str
    candidate_path: str | None = None
    candidate_hash: str | None = None
    staged_at: str | None = None
    simulate_mode_override: bool = False
    consolidating: bool = False
    server_started_at: str = ""
    # Forward-compat fields for 3b.3 long-poll and operator visibility.
    trial_started_at: str | None = None
    pre_trial_config_sha256: str | None = None
    candidate_config_sha256: str | None = None
    backup_paths: dict | None = None
    trial_adapter_dir: str | None = None
    trial_graph_dir: str | None = None
    gates: dict | None = None
    recovery_required: list[str] = []
    # Comparison report populated when TRIAL + gates eligible + completed.
    # None in LIVE/STAGING or when gates are still pending/failed/running.
    comparison_report: dict | None = None


class ConfirmRequest(BaseModel):
    """Request body for ``POST /migration/confirm``.

    No parameters — the server uses the in-memory STAGING stash.
    """


class ConfirmResponse(BaseModel):
    """Response body for ``POST /migration/confirm``.

    Attributes
    ----------
    state:
        Always ``"TRIAL"`` on success.
    trial_started_at:
        ISO-8601 UTC timestamp when TRIAL was entered.
    pre_trial_config_sha256:
        SHA-256 of the live config before the atomic rename.
    candidate_config_sha256:
        SHA-256 of the candidate bytes.
    backup_paths:
        Absolute paths to the three pre-migration backup slot directories.
    trial_adapter_dir:
        Absolute path to the trial adapter directory.
    trial_graph_dir:
        Absolute path to the trial graph directory.
    """

    state: str
    trial_started_at: str
    pre_trial_config_sha256: str
    candidate_config_sha256: str
    backup_paths: dict[str, str]
    trial_adapter_dir: str
    trial_graph_dir: str


class MigrationCancelResponse(BaseModel):
    """Response body for ``POST /migration/cancel``.

    Attributes
    ----------
    state:
        Always ``"LIVE"`` — the server has returned to LIVE state.
    cleared_path:
        The candidate path that was discarded.
    """

    state: str
    cleared_path: str


class AcceptResponse(BaseModel):
    """Response body for ``POST /migration/accept``.

    Attributes
    ----------
    state:
        Always ``"LIVE"`` on success (B config is now live).
    trial_adapter_archive_path:
        Absolute path to the trial adapter archive slot directory.
    restart_required:
        Always ``True`` — new configuration takes effect on server restart.
    restart_hint:
        Human-readable restart command string.
    pre_migration_backup_retained:
        Always ``True`` — the A-config backup is retained post-accept.
    """

    state: str
    trial_adapter_archive_path: str
    restart_required: bool
    restart_hint: str
    pre_migration_backup_retained: bool


class RollbackResponse(BaseModel):
    """Response body for ``POST /migration/rollback``.

    Attributes
    ----------
    state:
        Always ``"LIVE"`` on success (A config is restored).
    trial_adapter_archive_path:
        Absolute path to the trial adapter archive slot directory (or the
        still-in-place state/trial_adapter/ when rotation failed — 207).
    rollback_pre_mortem_backup_path:
        Absolute path to the rollback pre-mortem B-config snapshot slot.
    restart_required:
        Always ``True`` — rollback renames server.yaml; restart to apply.
    restart_hint:
        Human-readable restart command string.
    """

    state: str
    trial_adapter_archive_path: str
    rollback_pre_mortem_backup_path: str
    restart_required: bool
    restart_hint: str


# --- Adapter mount helper ---


def _mount_adapters_from_slots(model, tokenizer, config, state: dict):
    """Load enabled adapters from slot-dir layout with manifest verification.

    Implements the startup validator specified in §2.1 of the migration plan.
    For each enabled adapter kind:

    1. Sweep orphan ``.pending`` dirs.
    2. Resolve the live registry SHA-256.
    3. Call ``find_live_slot`` to locate the matching slot.
    4. Read the manifest; compare base model / tokenizer / LoRA fingerprints.
    5. Mount matching slots; record mismatch / missing rows in
       ``state["adapter_manifest_status"]``.

    Interim adapters (``episodic_interim_*``) are handled with the same logic.
    The registry consistency check (I5 / orphan-key cleanup) runs here after
    all mounts complete.

    Args:
        model: Base model (or existing PeftModel) to load adapters onto.
        tokenizer: Loaded tokenizer (for fingerprint comparison).
        config: Loaded ``ServerConfig``.
        state: The global ``_state`` dict (mutated in-place for manifest status).

    Returns:
        The updated model (PeftModel when any adapter was loaded, otherwise
        the original base model).
    """
    from datetime import datetime, timezone

    from peft import PeftModel

    from paramem.adapters.manifest import (
        ManifestNotFoundError,
        ManifestSchemaError,
        find_live_slot,
        read_manifest,
    )
    from paramem.backup.backup import sweep_orphan_pending

    manifest_status: dict = state.setdefault("adapter_manifest_status", {})

    # Compute live registry hash once (used for all adapter kinds).
    # Hash the plaintext content — see manifest.py::build_manifest_for for
    # why ciphertext-based hashing breaks drift detection under Security ON.
    _registry_path = config.adapter_dir / "indexed_key_registry.json"
    live_registry_sha256 = ""
    if _registry_path.exists():
        try:
            import hashlib as _rhash

            from paramem.backup.encryption import read_maybe_encrypted as _rme

            live_registry_sha256 = _rhash.sha256(_rme(_registry_path)).hexdigest()
        except Exception:  # noqa: BLE001
            live_registry_sha256 = ""

    def _checked_at() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def _is_primary(name: str) -> bool:
        """Episodic is primary (red on mismatch); others are secondary (yellow)."""
        return name == "episodic"

    def _record_row(name: str, status: str, reason: str, severity: str, slot_path=None, field=None):
        manifest_status[name] = {
            "status": status,
            "reason": reason,
            "field": field,
            "severity": severity,
            "slot_path": str(slot_path.name) if slot_path else None,
            "checked_at": _checked_at(),
        }

    def _load_one(name: str, slot: Path):
        """Mount a single adapter from *slot* onto *model* (mutates nonlocal model).

        Does not overwrite an existing manifest status row — caller may have already
        recorded a manifest_missing or migrated_unverified row before calling this.
        """
        nonlocal model
        try:
            if isinstance(model, PeftModel):
                model.load_adapter(str(slot), adapter_name=name)
            else:
                model = PeftModel.from_pretrained(model, str(slot), adapter_name=name)
            logger.info("Mounted adapter %s from slot %s", name, slot.name)
        except Exception as exc:
            logger.error("Failed to load adapter %s from %s: %s", name, slot, exc)
            # Only record load_failed if there is no prior manifest row
            if name not in manifest_status:
                _record_row(
                    name,
                    "manifest_missing",
                    "load_failed",
                    "red" if _is_primary(name) else "yellow",
                    slot,
                )

    # ---- Main adapter kinds ----
    main_adapters = [
        ("episodic", config.adapters.episodic),
        ("semantic", config.adapters.semantic),
        ("procedural", config.adapters.procedural),
    ]

    for name, adapter_cfg in main_adapters:
        if not adapter_cfg.enabled:
            continue

        kind_dir = config.adapter_dir / name
        # Sweep stale pending slots before scanning
        if kind_dir.exists():
            sweep_orphan_pending(kind_dir)

        slot = find_live_slot(kind_dir, live_registry_sha256)

        if slot is None:
            # Check whether any slots exist at all (distinguishes fresh install from mismatch)
            has_slots = (
                kind_dir.exists()
                and any(e for e in kind_dir.iterdir() if not e.name.startswith(".") and e.is_dir())
                if kind_dir.exists()
                else False
            )
            if has_slots:
                severity = "red" if _is_primary(name) else "yellow"
                _record_row(name, "no_matching_slot", "no_matching_slot", severity)
                logger.warning("Adapter %s: no slot matching registry hash — skipping mount", name)
            else:
                logger.info("Adapter %s: no slots found — fresh install", name)
            continue

        # Read and verify manifest
        try:
            manifest = read_manifest(slot)
        except ManifestNotFoundError:
            severity = "red" if _is_primary(name) else "yellow"
            _record_row(name, "manifest_missing", "manifest_missing", severity, slot)
            # Still mount — weights are present even without manifest
            _load_one(name, slot)
            continue
        except ManifestSchemaError as exc:
            severity = "red" if _is_primary(name) else "yellow"
            _record_row(name, "mismatch", "manifest_unreadable", severity, slot)
            logger.warning("Adapter %s: corrupt meta.json (%s) — skipping mount", name, exc)
            continue

        # Fingerprint comparison
        mismatch_field = _check_manifest_fingerprints(manifest, model, tokenizer, adapter_cfg)
        if mismatch_field is not None:
            severity = "red" if _is_primary(name) else "yellow"
            _record_row(name, "mismatch", "fingerprint_mismatch", severity, slot, mismatch_field)
            logger.warning(
                "Adapter %s: fingerprint mismatch on field '%s' — skipping mount",
                name,
                mismatch_field,
            )
            continue

        # Check for UNKNOWN fields
        unknown_field = _first_unknown_field(manifest)
        if unknown_field is not None:
            if manifest.synthesized:
                # synthesized + UNKNOWN → yellow (acceptable transient state)
                _record_row(
                    name,
                    "migrated_unverified",
                    "unknown_fields_in_manifest",
                    "yellow",
                    slot,
                    unknown_field,
                )
            else:
                # fresh-built + UNKNOWN → red (build_manifest_for failed silently)
                _record_row(
                    name,
                    "migrated_unverified",
                    "unknown_fields_in_manifest",
                    "red",
                    slot,
                    unknown_field,
                )
            logger.info(
                "Adapter %s: UNKNOWN field '%s' in manifest (synthesized=%s)"
                " — mounting with warning",
                name,
                unknown_field,
                manifest.synthesized,
            )

        _load_one(name, slot)

    # ---- Interim adapters ----
    for _interim_path in sorted(config.adapter_dir.glob("episodic_interim_*")):
        if not _interim_path.is_dir():
            continue
        sweep_orphan_pending(_interim_path)
        _interim_name = _interim_path.name

        slot = find_live_slot(_interim_path, live_registry_sha256)
        if slot is None:
            # Fallback: old flat layout (no slot-dir yet) — look for adapter files directly
            if (_interim_path / "adapter_config.json").exists() and (
                _interim_path / "adapter_model.safetensors"
            ).exists():
                logger.info("Loading interim adapter (flat layout): %s", _interim_name)
                try:
                    if isinstance(model, PeftModel):
                        model.load_adapter(str(_interim_path), adapter_name=_interim_name)
                    else:
                        model = PeftModel.from_pretrained(
                            model, str(_interim_path), adapter_name=_interim_name
                        )
                except Exception as exc:
                    logger.error("Failed to load interim adapter %s: %s", _interim_name, exc)
            else:
                logger.warning("Interim adapter %s: no matching slot — skipping", _interim_name)
            continue

        _load_one(_interim_name, slot)

    # ---- I5 — Registry consistency check ----
    # Drop orphan registry entries whose adapter slot is missing.
    if _registry_path.exists():
        from paramem.training.key_registry import KeyRegistry as _KeyRegistry

        _reg = _KeyRegistry.load(_registry_path)
        _orphaned: list[str] = []
        for _key in list(_reg.list_active()):
            _aid = _reg.get_adapter_id(_key)
            if _aid.startswith("episodic_interim_"):
                # Slot-dir: check via find_live_slot
                _aid_dir = config.adapter_dir / _aid
                _slot = find_live_slot(_aid_dir, live_registry_sha256)
                if _slot is None:
                    # Also check flat layout fallback
                    _flat_weights = _aid_dir / "adapter_model.safetensors"
                    if not _flat_weights.exists():
                        _reg.remove(_key)
                        _orphaned.append(_key)
        if _orphaned:
            logger.warning(
                "Startup registry check: dropped %d orphan key(s) whose adapter "
                "weights are missing (adapter save was interrupted): %s",
                len(_orphaned),
                _orphaned,
            )
            _reg.save(_registry_path)

    if hasattr(model, "peft_config") and model.peft_config:
        logger.info("Adapters loaded: %s", list(model.peft_config.keys()))
    else:
        logger.info("No adapters found — starting fresh")

    return model


def _check_manifest_fingerprints(manifest, model, tokenizer, adapter_cfg) -> "str | None":
    """Compare manifest fingerprints against live runtime state.

    Skips UNKNOWN values (cannot verify).  Returns the name of the first
    mismatching field, or ``None`` when all non-UNKNOWN fields match.

    Args:
        manifest: :class:`~paramem.adapters.manifest.AdapterManifest` to check.
        model: Live base model (or PeftModel) with ``config`` attribute.
        tokenizer: Live tokenizer.
        adapter_cfg: Per-adapter config from server.yaml (rank, alpha, etc.).

    Returns:
        Field name string on mismatch, ``None`` on match.
    """
    from paramem.adapters.manifest import UNKNOWN

    # base_model.sha — most specific identifier
    live_sha = getattr(getattr(model, "config", None), "_commit_hash", None) or UNKNOWN
    if manifest.base_model.sha != UNKNOWN and live_sha != UNKNOWN:
        if manifest.base_model.sha != live_sha:
            return "base_model.sha"

    # base_model.repo
    live_repo = getattr(getattr(model, "config", None), "_name_or_path", None) or UNKNOWN
    if manifest.base_model.repo != UNKNOWN and live_repo != UNKNOWN:
        if manifest.base_model.repo != live_repo:
            return "base_model.repo"

    # LoRA shape
    if isinstance(manifest.lora.rank, int) and manifest.lora.rank != 0:
        if manifest.lora.rank != adapter_cfg.rank:
            return "lora.rank"
    if isinstance(manifest.lora.alpha, int) and manifest.lora.alpha != 0:
        if manifest.lora.alpha != adapter_cfg.alpha:
            return "lora.alpha"
    if manifest.lora.target_modules:
        live_targets = tuple(sorted(adapter_cfg.target_modules or []))
        if manifest.lora.target_modules != live_targets:
            return "lora.target_modules"

    return None


def _first_unknown_field(manifest) -> "str | None":
    """Return the name of the first UNKNOWN-valued field, or None.

    Only checks fields that are material for adapter verification.
    """
    from paramem.adapters.manifest import UNKNOWN

    checks = [
        ("base_model.repo", manifest.base_model.repo),
        ("base_model.sha", manifest.base_model.sha),
        ("base_model.hash", manifest.base_model.hash),
        ("tokenizer.name_or_path", manifest.tokenizer.name_or_path),
        ("registry_sha256", manifest.registry_sha256),
    ]
    for field, value in checks:
        if value == UNKNOWN:
            return field
    if manifest.key_count == UNKNOWN:
        return "key_count"
    return None


# --- Lifespan ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, clean up on shutdown."""
    config = _state["config"]

    from paramem.backup.encryption import (
        assert_mode_consistency as _assert_mode,
    )
    from paramem.server.drift import drift_poll_loop, initial_drift_state
    from paramem.server.migration import initial_migration_state
    from paramem.server.migration_recovery import (
        MigrationRecoveryResult,
        RecoveryAction,
        recover_migration_state,
    )

    # Record server start time for /migration/status (Condition 6).
    _state["server_started_at"] = datetime.now(timezone.utc).isoformat()
    # Seed the migration stash to LIVE so the endpoint is available immediately.
    _state["migration"] = initial_migration_state()

    # Create the migration lock (must be inside a running event loop).
    _state["migration_lock"] = asyncio.Lock()

    # Security startup gate (SECURITY.md §4):
    # 1) Refuse startup when security.require_encryption=true and the daily
    #    age identity is not loadable — uniform fail-loud gate covering
    #    every feature (snapshots, shards, backups, infra).
    # 2) Refuse startup on any mode-mismatch case (plaintext alongside age
    #    envelopes, or age files without the daily identity loaded).
    # 3) Emit the canonical SECURITY: ON/OFF line so operators see posture.
    from paramem.backup.key_store import (
        daily_identity_loadable as _daily_loadable,
    )
    from paramem.backup.key_store import (
        recovery_pub_available as _recovery_available,
    )
    from paramem.server.security_posture import (
        assert_startup_posture,
        security_posture_log_line,
    )

    _daily_ok = _daily_loadable()
    _recovery_ok = _recovery_available()
    assert_startup_posture(
        require_encryption=config.security.require_encryption,
        daily_loadable=_daily_ok,
    )
    _assert_mode(
        config.paths.data,
        daily_identity_loadable=_daily_ok,
    )
    _line, _is_on = security_posture_log_line(
        daily_loadable=_daily_ok,
        recovery_available=_recovery_ok,
    )
    if _is_on:
        logger.info(_line)
    else:
        logger.warning(_line)
    _state["encryption"] = "on" if _is_on else "off"

    # --- Crash recovery: inspect disk state BEFORE drift init ---
    # This ensures _state["migration"] reflects any partially-completed
    # /migration/confirm before any request handler can observe stale state.
    try:
        live_config_path = (
            Path(_state["config_path"])
            if _state.get("config_path")
            else Path("configs/server.yaml")
        )
        state_dir = config.paths.data / "state"
        backups_root = config.paths.data / "backups"
        max_age_hours = getattr(
            getattr(getattr(config, "security", None), "backups", None),
            "orphan_sweep",
            None,
        )
        max_age_hours = max_age_hours.max_age_hours if max_age_hours is not None else 24

        recovery_result: MigrationRecoveryResult = recover_migration_state(
            state_dir=state_dir,
            live_config_path=live_config_path,
            backups_root=backups_root,
            max_age_hours=max_age_hours,
        )

        # Emit the recovery log lines.
        for level, msg in recovery_result.log_lines:
            getattr(logger, level.lower(), logger.info)(msg)

        # Seed _state["migration"] from recovery result.
        if (
            recovery_result.action == RecoveryAction.RESUME_TRIAL
            and recovery_result.trial_marker is not None
        ):
            m = recovery_result.trial_marker
            from paramem.server.migration import TrialStash

            trial_stash: TrialStash = TrialStash(
                started_at=m.started_at,
                pre_trial_config_sha256=m.pre_trial_config_sha256,
                candidate_config_sha256=m.candidate_config_sha256,
                backup_paths={
                    "config": m.backup_paths.get("config", ""),
                    "graph": m.backup_paths.get("graph", ""),
                    "registry": m.backup_paths.get("registry", ""),
                },
                trial_adapter_dir=m.trial_adapter_dir,
                trial_graph_dir=m.trial_graph_dir,
                gates={"status": "pending"},
            )
            _state["migration"]["state"] = "TRIAL"
            _state["migration"]["trial"] = trial_stash
            _state["migration"]["recovery_required"] = []
        elif recovery_result.recovery_required:
            _state["migration"]["recovery_required"] = list(recovery_result.recovery_required)

    except Exception as _recovery_exc:  # noqa: BLE001
        logger.error("migration recovery failed unexpectedly: %s", _recovery_exc, exc_info=True)

    if _state.get("config_path"):
        _state["config_drift"] = initial_drift_state(Path(_state["config_path"]))

    # cloud_only is enabled if ANY of the following is true:
    #   1. --cloud-only CLI flag was passed at startup.
    #   2. cloud_only: true is set in server.yaml (YAML cannot be silently overridden).
    #   3. --defer-model flag was passed (start cloud-only then auto-reclaim GPU).
    # OR is the correct combiner: both are opt-in signals for cloud-only mode.
    cloud_only = (
        _state.get("cloud_only_startup", False)
        or config.cloud_only
        or _state.get("defer_model", False)
    )

    # Track why we're cloud-only
    if _state.get("cloud_only_startup", False):
        _state["cloud_only_reason"] = "explicit"
    elif _state.get("defer_model", False):
        _state["cloud_only_reason"] = "training"
    else:
        _state["cloud_only_reason"] = None

    # Auto-detect GPU conflict: if another process holds the GPU, start cloud-only
    if not cloud_only and _gpu_occupied():
        logger.warning(
            "GPU is occupied by another process — starting in cloud-only mode. "
            "Will auto-reclaim when GPU is free."
        )
        notify_server(SERVER_CLOUD_ONLY)
        cloud_only = True
        _state["cloud_only_reason"] = "gpu_conflict"

    if cloud_only:
        logger.info("Starting in cloud-only mode — skipping model load")
        _state["model"] = None
        _state["tokenizer"] = None
    else:
        # Startup VRAM validation runs BEFORE loading the base model.
        # On (re)start ParaMem's own footprint is wiped — the CUDA context
        # begins empty for this process. ``torch.cuda.memory_allocated`` at
        # this point therefore reports bytes held by OTHER GPU consumers
        # (Windows desktop, another model server, etc.). That external
        # occupancy is subtracted from the hardware cap to yield the live
        # budget the configured topology must fit into. The topology math
        # itself is hardware-agnostic — logged at INFO with a per-tier fit
        # table so operators see what a bigger card would buy.
        main_adapter_count = sum(
            1
            for adapter_cfg in (
                config.adapters.episodic,
                config.adapters.semantic,
                config.adapters.procedural,
            )
            if adapter_cfg.enabled
        )
        stt_pre_bytes = estimate_stt_bytes(config.stt, cloud_only=cloud_only)
        tts_pre_bytes = estimate_tts_bytes(config.tts, cloud_only=cloud_only)

        if not torch.cuda.is_available():
            logger.error(
                "Local model mode requires a CUDA-capable GPU but none was detected. "
                "Either provide a GPU, or start in cloud-only mode."
            )
            sys.exit(1)

        assessment = assess_topology(
            config.episodic_adapter_config,
            max_interim_count=config.consolidation.max_interim_count,
            model_name=config.model_name,
            model_id=config.model_config.model_id,
            main_adapter_count=main_adapter_count,
            stt_bytes=stt_pre_bytes,
            tts_bytes=tts_pre_bytes,
        )
        logger.info("VRAM topology assessment:\n%s", assessment.breakdown)
        logger.info("%s", format_tier_table(assessment))

        total_memory_bytes, external_bytes = measure_external_vram()
        try:
            enforce_live_budget(assessment, total_memory_bytes, external_bytes)
        except ConfigurationError as exc:
            logger.error("VRAM configuration error:\n%s", exc)
            sys.exit(1)

        logger.info("Loading model: %s (%s)", config.model_name, config.model_config.model_id)
        model, tokenizer = load_base_model(config.model_config)

        # Initialize manifest-related state (new Slice-3a surface).
        _state["adapter_manifest_status"] = {}
        _state["base_model_hash_cache"] = {}

        # Load adapters using the slot-dir resolver with manifest verification.
        model = _mount_adapters_from_slots(model, tokenizer, config, _state)

        # Restore the main episodic adapter as the active adapter so that all
        # inference probes default to the consolidated tier.  Only needed when at
        # least one adapter is present; on a fresh install peft_config is absent.
        if hasattr(model, "peft_config") and "episodic" in model.peft_config:
            switch_adapter(model, "episodic")

        _state["model"] = model
        _state["tokenizer"] = tokenizer
    _state["session_buffer"] = SessionBuffer(
        config.session_dir,
        retain_sessions=config.consolidation.retain_sessions,
        debug=config.debug,
    )
    _state["session_buffer"].load_snapshot()

    # Initialize speaker store (voice-based identification)
    if config.speaker.enabled:
        from paramem.server.speaker import SpeakerStore

        speaker_path = (
            Path(config.speaker.store_path)
            if config.speaker.store_path
            else config.paths.data / "speaker_profiles.json"
        )
        _state["speaker_store"] = SpeakerStore(
            speaker_path,
            high_threshold=config.speaker.high_confidence_threshold,
            low_threshold=config.speaker.low_confidence_threshold,
            max_embeddings=config.speaker.max_embeddings_per_profile,
            redundancy_threshold=config.speaker.redundancy_threshold,
        )
        logger.info("Speaker store: %d profiles", _state["speaker_store"].profile_count)

        # Preload speaker embedding model (CPU, ~17 MB)
        from paramem.server.speaker_embedding import load_embedding_model

        if load_embedding_model():
            logger.info("Speaker embedding model ready")
        else:
            logger.warning("Speaker embedding unavailable — install paramem[speaker]")
    else:
        _state["speaker_store"] = None
        logger.info("Speaker identification disabled")

    # Initialize local STT (Faster Whisper) if configured
    # In cloud-only mode, force CPU to avoid GPU contention with training
    _state["stt"] = None
    _state["wyoming_server"] = None
    if config.stt.enabled:
        from paramem.server.stt import WhisperSTT

        if cloud_only and config.stt.device != "cpu":
            stt_device = "cpu"
            stt_model = config.stt.cpu_fallback_model
            stt_compute = "int8"
            logger.info(
                "Cloud-only mode — Whisper on CPU (%s instead of %s)",
                stt_model,
                config.stt.model,
            )
        else:
            stt_device = config.stt.device
            stt_model = config.stt.model
            stt_compute = config.stt.compute_type

        stt = WhisperSTT(
            model_name=stt_model,
            device=stt_device,
            compute_type=stt_compute,
            language=config.stt.language,
            beam_size=config.stt.beam_size,
            vad_filter=config.stt.vad_filter,
        )
        if stt.load():
            _state["stt"] = stt
            logger.info("Local STT: Whisper %s on %s", stt_model, stt_device)

            # Start Wyoming STT server
            from paramem.server.wyoming_handler import start_wyoming_server

            def _on_stt_embedding(embedding):
                """Store latest speaker embedding from Wyoming STT."""
                if embedding:
                    _state["latest_embedding"] = embedding

            def _on_stt_language(language: str, probability: float):
                """Store latest detected language from Wyoming STT.

                Written as single dict to avoid race between language and probability.
                Read by both /chat endpoint and TTS language resolver.
                """
                _state["latest_language_detection"] = {
                    "language": language,
                    "probability": probability,
                }

            _state["wyoming_server"] = await start_wyoming_server(
                host=config.server.host,
                port=config.stt.port,
                stt=stt,
                speaker_store=_state.get("speaker_store"),
                embedding_callback=_on_stt_embedding,
                language_callback=_on_stt_language,
                min_embedding_duration_seconds=config.speaker.min_embedding_duration_seconds,
            )
            logger.info("Wyoming STT server listening on port %d", config.stt.port)
        else:
            logger.warning("Local STT disabled — model failed to load")
    else:
        logger.info("Local STT: disabled")

    # Initialize local TTS if configured
    if config.tts.enabled:
        from paramem.server.tts import TTSManager

        tts_config = config.tts
        if cloud_only and tts_config.device != "cpu":
            logger.info("Cloud-only mode — forcing TTS to CPU")
            # Override device to CPU so engines don't attempt GPU loading
            tts_config = TTSConfig(
                enabled=tts_config.enabled,
                port=tts_config.port,
                device="cpu",
                default_language=tts_config.default_language,
                language_confidence_threshold=tts_config.language_confidence_threshold,
                model_dir=tts_config.model_dir,
                audio_chunk_bytes=tts_config.audio_chunk_bytes,
                voices=tts_config.voices,
            )

        tts_manager = TTSManager(
            tts_config,
            vram_safety_margin_mb=config.server.vram_safety_margin_mb,
        )
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, tts_manager.load_all)

        if tts_manager.is_loaded:
            _state["tts_manager"] = tts_manager
            logger.info("Local TTS: %s", ", ".join(tts_manager.available_languages))

            from paramem.server.wyoming_handler import start_wyoming_tts_server

            lang_conf_threshold = config.tts.language_confidence_threshold

            def _resolve_language():
                """Return the most recently detected language if confidence is sufficient.

                Consumes (pops) the detection so stale values don't persist
                across requests.
                """
                detection = _state.pop("latest_language_detection", None)
                if not detection:
                    return None
                if detection["probability"] >= lang_conf_threshold:
                    return detection["language"]
                return None

            _state["wyoming_tts_server"] = await start_wyoming_tts_server(
                host=config.server.host,
                port=config.tts.port,
                tts_manager=tts_manager,
                language_resolver=_resolve_language,
                audio_chunk_bytes=config.tts.audio_chunk_bytes,
            )
            logger.info("Wyoming TTS server listening on port %d", config.tts.port)
        else:
            logger.warning("Local TTS disabled — no voices loaded")
    else:
        logger.info("Local TTS: disabled")

    # Post-load sanity: compare actual ParaMem allocation against the prediction.
    # Not a gate — the pre-load check already enforced the budget against
    # (total_memory − external_bytes). This surfaces any drift between the
    # static VRAM tables (base model, STT, TTS) and reality, so calibration
    # errors show up in logs rather than silently over-committing.
    if _state.get("model") is not None and torch.cuda.is_available():
        actual_bytes = torch.cuda.memory_allocated(0)
        predicted_bytes = assessment.required_bytes
        delta_mib = (actual_bytes - predicted_bytes) / (1024 * 1024)
        logger.info(
            "VRAM post-load sanity: predicted %.2f GiB, measured %.2f GiB (delta %+.0f MiB)",
            predicted_bytes / 2**30,
            actual_bytes / 2**30,
            delta_mib,
        )
        if actual_bytes > predicted_bytes * 1.10:
            logger.warning(
                "Actual VRAM usage exceeds prediction by >10%% — "
                "re-calibrate _MODEL_VRAM_BYTES / STT / TTS tables."
            )

    # Initialize SOTA agent if configured
    _state["sota_agent"] = get_cloud_agent(config.sota_agent)
    if _state["sota_agent"]:
        logger.info(
            "SOTA agent: %s (%s)",
            config.sota_agent.provider,
            config.sota_agent.model,
        )
    else:
        logger.info("SOTA agent: not configured")

    # Register additional SOTA providers for direct routing (sota:anthropic, etc.)
    _state["sota_providers"] = {}
    for name, provider_config in config.sota_providers.items():
        agent = get_cloud_agent(provider_config)
        if agent:
            _state["sota_providers"][name] = agent
            logger.info("SOTA provider registered: %s (%s)", name, provider_config.model)
    logger.info("SOTA providers available: %s", list(_state["sota_providers"].keys()))

    # Initialize HA client, tool registry, and HA entity graph
    ha_graph = None
    tools_config = config.tools
    if tools_config.ha.url and tools_config.ha.token:
        ha_client = HAClient(
            url=tools_config.ha.url,
            token=tools_config.ha.token,
            timeout=tools_config.tool_timeout_seconds,
        )
        health = ha_client.health_check()
        if health:
            logger.info("HA client: connected to %s", tools_config.ha.url)
            entity_count = ha_client.load_entity_map()
            logger.info("HA entity map: %d entities", entity_count)

            # Build HA entity graph for dual-graph routing
            ha_services = ha_client.get_services()
            ha_graph = HAEntityGraph.build(ha_client._raw_states, ha_services)
        else:
            logger.warning("HA client: configured but unreachable at %s", tools_config.ha.url)
        _state["ha_client"] = ha_client

    else:
        logger.info("HA tools: not configured")

    _state["ha_graph"] = ha_graph
    _state["router"] = QueryRouter(
        adapter_dir=config.adapter_dir,
        ha_graph=ha_graph,
    )

    # Global observed-language tracker — records STT-detected languages with
    # high confidence, publishes to HA as input_text.voice_observed_languages
    # so the conversation agent can use it as context when interpreting
    # (potentially mangled) transcripts from CPU fallback STT.
    from paramem.server.language_tracker import LanguageTracker

    _state["language_tracker"] = LanguageTracker(
        store_path=config.paths.data / "observed_languages.json",
        ha_client=_state.get("ha_client"),
    )

    # Reconcile the systemd user timer with the DERIVED full consolidation
    # period (= refresh_cadence × max_interim_count). The yaml exposes only
    # refresh_cadence; the timer sees the full cycle via the derived
    # ``consolidation_period_string`` property. Timer fires POST /scheduled-tick
    # on that cadence regardless of mode; the tick handler defers when GPU is
    # busy rather than skipping silently. Timer survives server restart.
    from paramem.server import systemd_timer

    derived_period = config.consolidation.consolidation_period_string
    logger.info(
        "Consolidation cadence — refresh every %s, max_interim_count=%d, "
        "derived full-consolidation period=%s",
        config.consolidation.refresh_cadence or "<disabled>",
        config.consolidation.max_interim_count,
        derived_period or "<manual only>",
    )
    try:
        msg = systemd_timer.reconcile(derived_period)
        logger.info("%s", msg)
    except Exception:
        logger.exception("Failed to reconcile consolidation timer — continuing without schedule")

    # Reconcile the scheduled-backup timer.
    from paramem.backup import timer as backup_timer

    backup_schedule = config.security.backups.schedule or ""
    try:
        backup_msg = backup_timer.reconcile(
            backup_schedule,
            python_path=sys.executable,
            project_root=str(Path(__file__).resolve().parent.parent.parent),
        )
        logger.info("%s", backup_msg)
    except Exception:
        logger.exception("Failed to reconcile backup timer — continuing without scheduled backups")

    # Graceful shutdown: save encrypted session snapshot before exit.
    # SIGUSR1 (GPU release for training) and SIGTERM (systemd stop) both
    # trigger a snapshot so unconsolidated conversations survive restarts.
    def _graceful_exit(signum, _frame):
        sig_name = signal.Signals(signum).name
        logger.info("Received %s — saving session snapshot before exit", sig_name)
        # Signal training to stop at the next epoch boundary
        consolidation_loop = _state.get("consolidation_loop")
        if consolidation_loop is not None:
            consolidation_loop.shutdown_requested = True
            logger.info("Shutdown flag set — training will stop after current epoch")
        bg_trainer = _state.get("background_trainer")
        if bg_trainer is not None and bg_trainer.is_training:
            bg_trainer.stop(timeout=30)
            logger.info("Background trainer stopped")
        buffer = _state.get("session_buffer")
        if buffer:
            buffer.save_snapshot()
        if signum == signal.SIGUSR1:
            # GPU release: exit non-zero so systemd restarts (Restart=on-failure)
            os._exit(1)
        else:
            # SIGTERM: intentional stop, don't restart
            raise SystemExit(0)

    signal.signal(signal.SIGUSR1, _graceful_exit)
    signal.signal(signal.SIGTERM, _graceful_exit)

    _state["mode"] = "cloud-only" if cloud_only else "local"
    _state["event_loop"] = asyncio.get_running_loop()

    # Auto-reclaim: only when started without the model (--defer-model).
    # In local mode we already have the GPU — nothing to reclaim.
    if cloud_only and not _state.get("cloud_only_startup", False):
        reclaim_interval = config.server.reclaim_interval_minutes
        _state["reclaim_task"] = asyncio.create_task(_auto_reclaim_loop(reclaim_interval))
    # Speaker enrollment is utterance-driven: the chat handler invokes
    # _run_enrollment_for_group synchronously when a self-introduction
    # marker fires. There is no idle-driven background loop.
    if _state.get("config_path"):
        _state["config_drift_task"] = asyncio.create_task(
            drift_poll_loop(Path(_state["config_path"]), _state)
        )

    # Startup diagnostic: surface any in-flight training state left by a
    # prior interrupted run.  The actual resume happens when the next
    # training job for that adapter is submitted — this is informational only.
    if _state.get("model") is not None:
        from paramem.server.background_trainer import _RESUME_STATE_FILE, _read_resume_state

        _resume_path = config.adapter_dir / "in_training" / _RESUME_STATE_FILE
        _resume_state = _read_resume_state(_resume_path)
        if _resume_state is not None and _resume_state.get("checkpoint_path"):
            logger.info(
                "Detected in-flight training state: adapter=%s epoch=%d/%d"
                " — will resume when the job is next submitted",
                _resume_state.get("adapter_name", "?"),
                _resume_state.get("last_completed_epoch", 0),
                _resume_state.get("total_epochs", 0),
            )

    # --- Post-session queue: persistent queue for missed training triggers ---
    # Instantiate the queue in local mode so the chat handler can use it.
    # In cloud-only mode there is no model to train — skip.
    if not cloud_only:
        _queue_path = config.adapter_dir / "post_session_queue.json"
        # Ensure the adapter directory exists (may not exist on fresh install).
        config.adapter_dir.mkdir(parents=True, exist_ok=True)
        _state["post_session_queue"] = PostSessionQueue(_queue_path)

        # Replay any entries that were enqueued before a previous restart.
        # We need the ConsolidationLoop + BackgroundTrainer to replay — create
        # them lazily here only when there are entries to process.
        if config.consolidation.post_session_train_enabled:
            _pending = _state["post_session_queue"].peek()
            if _pending:
                logger.info(
                    "post_session_queue: %d pending entry(s) from previous run — replaying",
                    len(_pending),
                )
                from paramem.server.background_trainer import BackgroundTrainer
                from paramem.server.consolidation import create_consolidation_loop

                if _state.get("consolidation_loop") is None:
                    _replay_loop = create_consolidation_loop(
                        _state["model"],
                        _state["tokenizer"],
                        config,
                        state_provider=lambda: _state,
                    )
                    _state["consolidation_loop"] = _replay_loop
                    _state["model"] = _replay_loop.model

                if _state.get("background_trainer") is None:
                    _replay_bt = BackgroundTrainer(
                        model=_state["model"],
                        tokenizer=_state["tokenizer"],
                        training_config=config.training_config,
                        output_dir=config.adapter_dir,
                        temp_limit=config.consolidation.training_temp_limit,
                        temp_check_interval=config.consolidation.training_temp_check_interval,
                        quiet_hours_mode=config.consolidation.quiet_hours_mode,
                        quiet_hours_start=config.consolidation.quiet_hours_start,
                        quiet_hours_end=config.consolidation.quiet_hours_end,
                    )
                    _state["background_trainer"] = _replay_bt

                # Drain and replay — entries are removed individually upon
                # successful completion inside enqueue_post_session_train's
                # wrapper (same pattern as the chat handler path).
                _replay_entries = _state["post_session_queue"].drain()
                for _entry in _replay_entries:
                    # Re-enqueue via the same path as the chat handler uses.
                    # The queue.remove() call on success is handled inside the
                    # _run_post_session_train wrapper added below for the chat
                    # handler path.  For replay we enqueue again so the
                    # enqueue→train→remove pattern is preserved.
                    _state["post_session_queue"].enqueue(_entry)
                    enqueue_post_session_train(
                        conversation_id=_entry["session_id"],
                        transcript=_entry["transcript"],
                        speaker_id=_entry["speaker_id"],
                        speaker_name=_entry.get("speaker_name"),
                        loop=_state["consolidation_loop"],
                        background_trainer=_state["background_trainer"],
                        config=config,
                        state=_state,
                    )
    else:
        _state["post_session_queue"] = None

    logger.info("ParaMem server ready — mode: %s, model: %s", _state["mode"], config.model_name)

    yield

    # Shutdown — flush deferred speaker profile writes
    store = _state.get("speaker_store")
    if store:
        store.flush()

    if _state.get("reclaim_task"):
        _state["reclaim_task"].cancel()
    if _state.get("config_drift_task"):
        _state["config_drift_task"].cancel()
    wyoming_server = _state.get("wyoming_server")
    if wyoming_server is not None:
        wyoming_server.stop()
    wyoming_tts = _state.get("wyoming_tts_server")
    if wyoming_tts is not None:
        wyoming_tts.stop()
    stt = _state.get("stt")
    if stt is not None:
        stt.unload()
    tts_manager = _state.get("tts_manager")
    if tts_manager is not None:
        tts_manager.unload_all()
    if _state.get("ha_client"):
        _state["ha_client"].close()
    if _state["model"]:
        unload_model(_state["model"], _state["tokenizer"])
        logger.info("Model unloaded")


app = FastAPI(title="ParaMem", version="0.1.0", lifespan=lifespan)

# Bearer-token auth on all REST endpoints when PARAMEM_API_TOKEN is set.
# No-op when unset (loud WARN at startup).
from paramem.server.auth import (  # noqa: E402
    BearerTokenMiddleware,
    load_token_from_env,
    log_startup_posture,
)

_api_token = load_token_from_env()
app.add_middleware(BearerTokenMiddleware, token=_api_token)
log_startup_posture(_api_token)


# --- Endpoints ---


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Handle a conversation turn with speaker identification."""
    _state["last_chat_time"] = datetime.now(timezone.utc)
    buffer = _state["session_buffer"]

    # Forced routing — bypass normal routing for direct provider testing.
    # Supports: "ha", "sota", "sota:anthropic", "sota:openai", "sota:google"
    if request.route and request.route.startswith(("ha", "sota")):
        _speaker_id, speaker = _resolve_speaker(request, buffer, _state.get("speaker_store"))
        loop = asyncio.get_running_loop()

        result = None
        if request.route == "ha" and _state.get("ha_client") is not None:
            response_text = await loop.run_in_executor(
                None,
                lambda: _state["ha_client"].conversation_process(
                    request.text, agent_id=_state["config"].ha_agent_id
                ),
            )
            if response_text is not None:
                result = ChatResult(text=response_text, escalated=True)
        elif request.route.startswith("sota"):
            parts = request.route.split(":", 1)
            agent = (
                _state.get("sota_providers", {}).get(parts[1])
                if len(parts) == 2
                else _state.get("sota_agent")
            )
            if agent is not None:
                result = await loop.run_in_executor(
                    None,
                    lambda: _escalate_to_sota(
                        request.text,
                        agent,
                        _state["config"],
                        speaker=speaker,
                        history=request.history,
                    ),
                )
        if result and result.text:
            return ChatResponse(text=result.text, escalated=True, speaker=speaker)
        return ChatResponse(
            text=f"Route '{request.route}' unavailable.",
            escalated=False,
            speaker=speaker,
        )

    # Pick up speaker embedding from latest STT if not in request
    latest_embedding = _state.get("latest_embedding")
    if not request.speaker_embedding and latest_embedding is not None:
        request.speaker_embedding = latest_embedding
        _state["latest_embedding"] = None
        logger.info("Picked up STT embedding (%d dims)", len(request.speaker_embedding))
    elif not request.speaker_embedding:
        logger.info("No speaker embedding available")

    # Pick up detected language from latest STT.
    # Not cleared here — TTS resolver reads it independently during synthesis
    # (which runs after /chat returns). Cleared after TTS consumes it via
    # the Wyoming handler's on_synthesize callback.
    lang_detection = _state.get("latest_language_detection")
    detected_language = lang_detection["language"] if lang_detection else None
    detected_language_prob = lang_detection["probability"] if lang_detection else 0.0
    if detected_language:
        logger.info("Detected language: %s (prob=%.2f)", detected_language, detected_language_prob)
        tracker = _state.get("language_tracker")
        if tracker is not None:
            tracker.record(detected_language, detected_language_prob)

    # Speaker resolution: embedding → session history → anonymous.
    # Never let speaker ID failure kill the request — proceed as anonymous.
    try:
        speaker_id, speaker = _resolve_speaker(request, buffer, _state.get("speaker_store"))
    except Exception:
        logger.exception("Speaker resolution failed — proceeding as anonymous")
        speaker_id, speaker = None, None
    follow_up = None
    store = _state.get("speaker_store")

    # Deferred enrollment: unknown voice → group by embedding, prompt on first
    # encounter for each group, then re-prompt after per-group cooldown.
    # Enrollment failure must never block the query.
    try:
        if speaker_id is None and request.speaker_embedding and store:
            conv_id = request.conversation_id
            now = datetime.now(timezone.utc)
            unknown_group_id = _match_unknown_speaker(request.speaker_embedding)

            if unknown_group_id:
                group = _state["unknown_speakers"][unknown_group_id]
                group["conversations"].add(conv_id)
                group["embeddings"].append(request.speaker_embedding)
                _state["pending_enrollments"].add(conv_id)
                logger.info("Unknown speaker — grouped into %s", unknown_group_id)
            else:
                unknown_group_id = uuid.uuid4().hex[:8]
                group = {
                    "embeddings": [request.speaker_embedding],
                    "conversations": {conv_id},
                    "first_seen": now,
                    "last_prompted": None,
                    "last_extract_turn_count": 0,
                }
                _state["unknown_speakers"][unknown_group_id] = group
                _state["pending_enrollments"].add(conv_id)
                logger.info("Unknown speaker — new group %s", unknown_group_id)

            # Per-group enrollment prompt: fire on first encounter, re-prompt
            # after reprompt_interval seconds of the same unresolved group.
            reprompt_interval = _state["config"].speaker.enrollment_reprompt_interval
            last_prompted = group.get("last_prompted")
            if last_prompted is None or (now - last_prompted).total_seconds() >= reprompt_interval:
                follow_up = _state["config"].speaker.enrollment_prompt
                group["last_prompted"] = now
                logger.info(
                    "Enrollment prompt sent for group %s (interval %ds)",
                    unknown_group_id,
                    reprompt_interval,
                )

            # Promote to a canonical Speaker{N} ID so facts flow through
            # extraction and adapter training. Orthogonal to the enrollment
            # prompt above — that coordinates "what's your name?" prompts;
            # this ensures sessions are not silently discarded at consolidation.
            try:
                anon_id = store.register_anonymous(request.speaker_embedding)
                speaker_id = anon_id
                # Anonymous speakers use their canonical ID as the display name until disclosure
                buffer.set_speaker(conv_id, anon_id, anon_id)
                logger.info("Anonymous speaker promoted to canonical ID: %s", anon_id)
            except Exception:
                logger.exception(
                    "register_anonymous failed — session will proceed without speaker attribution"
                )
    except Exception:
        logger.exception("Speaker enrollment failed — continuing without enrollment")

    # Run the LLM enrollment helper on every anonymous turn that carries
    # a voice embedding. The LLM extractor is the sole filter — it
    # returns NONE on non-introductions, so non-intro turns have no
    # side effect, only the extraction latency. This matches the
    # original mechanism's "LLM as filter" principle and avoids the
    # fragility of pattern-matching introduction phrasings. Operates on
    # speaker_id directly so it works for both freshly-promoted voices
    # and returning anonymous speakers (whose unknown_speakers group is
    # gone after the server restart that allocated their Speaker{N}).
    try:
        if speaker_id and store and store.is_anonymous(speaker_id) and request.speaker_embedding:
            extracted = await _run_enrollment_for_speaker(
                speaker_id,
                request.conversation_id,
                request.speaker_embedding,
                extra_turns=[{"role": "user", "text": request.text}],
            )
            if extracted:
                speaker = extracted
                follow_up = None  # already enrolled; no need to ask again
    except Exception:
        logger.exception("Enrollment trigger failed — re-prompt will fire on next anonymous turn")

    # Update speaker language preference from STT detection
    tts_config = _state["config"].tts
    if speaker_id and store and detected_language and detected_language_prob > 0:
        store.update_language(
            speaker_id,
            detected_language,
            detected_language_prob,
            threshold=tts_config.language_confidence_threshold,
        )

    # Resolve effective language for this request:
    # 1. High-confidence Whisper detection
    # 2. Speaker's stored preference
    # 3. Config default (English)
    lang_threshold = tts_config.language_confidence_threshold
    if detected_language and detected_language_prob >= lang_threshold:
        effective_language = detected_language
    elif speaker_id and store:
        effective_language = store.get_preferred_language(speaker_id)
    else:
        effective_language = None
    # Replace detected_language with the resolved effective language
    detected_language = effective_language

    # User-facing salutation: suppress the canonical "Speaker{N}" token for
    # voice-promoted but undisclosed profiles. Internal speaker_id is kept
    # for attribution; only the display name is dropped so the greeting
    # prefix and the system-prompt "You are speaking with X" string skip the
    # robotic label until the user introduces themselves.
    display_speaker: str | None = speaker
    if speaker_id and store and store.is_anonymous(speaker_id):
        display_speaker = None

    # Check greeting before routing (applies to all paths)
    greeting_prefix = None
    greeting_interval = _state["config"].voice.greeting_interval_hours
    if speaker_id and store and greeting_interval > 0:
        greeting = store.should_greet(speaker_id, greeting_interval)
        if greeting:
            if display_speaker:
                greeting_prefix = f"{greeting}, {display_speaker}. "
            else:
                greeting_prefix = f"{greeting}. "
            store.confirm_greeting(speaker_id)

    # Cloud-only mode — route via HA graph + SOTA, no local model
    if _state["mode"] == "cloud-only":
        result = _cloud_only_route(
            text=request.text,
            speaker=display_speaker,
            history=request.history,
            config=_state["config"],
            router=_state.get("router"),
            ha_client=_state.get("ha_client"),
            sota_agent=_state.get("sota_agent"),
            language=detected_language,
        )
        buffer.append(
            request.conversation_id,
            "user",
            request.text,
            embedding=request.speaker_embedding,
        )
        cloud_text = result.text
        buffer.append(request.conversation_id, "assistant", cloud_text)
        spoken_text = f"{greeting_prefix}{cloud_text}" if greeting_prefix else cloud_text
        return ChatResponse(text=spoken_text, escalated=True, speaker=speaker, follow_up=follow_up)

    # Local mode — normal inference with entity routing
    # Pause background training if active, then acquire GPU lock
    bg_trainer = _state.get("background_trainer")
    training_paused = False
    training_stopped = False
    if bg_trainer is not None and bg_trainer.is_training:
        if bg_trainer.pause():
            training_paused = True
        else:
            logger.warning("Could not pause training — stopping trainer before inference")
            bg_trainer.stop(timeout=30)
            training_stopped = True

    from paramem.server.gpu_lock import gpu_lock

    async with gpu_lock():
        loop = asyncio.get_running_loop()
        result: ChatResult = await loop.run_in_executor(
            None,
            lambda: handle_chat(
                text=request.text,
                conversation_id=request.conversation_id,
                speaker=display_speaker,
                speaker_id=speaker_id,
                history=request.history,
                model=_state["model"],
                tokenizer=_state["tokenizer"],
                config=_state["config"],
                router=_state["router"],
                sota_agent=_state.get("sota_agent"),
                ha_client=_state.get("ha_client"),
                language=detected_language,
            ),
        )

    buffer.append(
        request.conversation_id,
        "user",
        request.text,
        embedding=request.speaker_embedding,
    )
    response_text = result.text
    buffer.append(request.conversation_id, "assistant", response_text)

    # Post-conversation training: after each assistant response, enqueue a
    # background job to extract and train onto the current interim adapter.
    # Only fires when:
    #   - a speaker is identified (speaker_id is required for key ownership)
    #   - post_session_train_enabled is True in the consolidation config
    #   - the consolidation loop exists (initialised at startup or by first
    #     consolidation run)
    # The job runs in a daemon thread — it never blocks the response path.
    # model handle is updated inside enqueue_post_session_train after training.
    if (
        speaker_id is not None
        and _state.get("consolidation_loop") is not None
        and _state["config"].consolidation.post_session_train_enabled
    ):
        transcript_turns = buffer.get_session_turns(request.conversation_id)
        if transcript_turns:
            transcript_text = "\n".join(
                f"{t.get('role', 'user')}: {t.get('text', '')}" for t in transcript_turns
            )
            # Persist the job to the queue BEFORE submitting to the trainer.
            # If the server crashes between here and the training completing,
            # the entry will be replayed on the next startup.
            _psq = _state.get("post_session_queue")
            if _psq is not None:
                _psq.enqueue(
                    {
                        "session_id": request.conversation_id,
                        "transcript": transcript_text,
                        "speaker_id": speaker_id,
                        "speaker_name": speaker,
                    }
                )
            enqueue_post_session_train(
                conversation_id=request.conversation_id,
                transcript=transcript_text,
                speaker_id=speaker_id,
                speaker_name=speaker,
                loop=_state["consolidation_loop"],
                background_trainer=_state.get("background_trainer"),
                config=_state["config"],
                state=_state,
                post_session_queue=_psq,
            )

    # Resume training after inference — only if we paused (not stopped)
    if training_paused and bg_trainer is not None:
        bg_trainer.resume()
    elif training_stopped:
        logger.warning(
            "Training was stopped for inference — will restart on next scheduled training interval"
        )

    spoken_text = f"{greeting_prefix}{response_text}" if greeting_prefix else response_text
    return ChatResponse(
        text=spoken_text,
        escalated=result.escalated,
        speaker=speaker,
        follow_up=follow_up,
    )


def _match_unknown_speaker(embedding: list[float]) -> str | None:
    """Match an embedding against unknown speaker groups.

    Uses a lenient threshold (low_confidence * 0.6 ≈ 0.27) because we're
    grouping noisy unknowns, not confirming identity. Compares against each
    group's centroid (improves as more embeddings accumulate).
    """
    from paramem.server.speaker import compute_centroid, cosine_similarity

    cfg = _state["config"].speaker
    threshold = cfg.low_confidence_threshold * cfg.grouping_threshold_factor
    best_id = None
    best_score = 0.0
    for group_id, group in _state["unknown_speakers"].items():
        centroid = compute_centroid(group["embeddings"])
        if not centroid:
            continue
        score = cosine_similarity(embedding, centroid)
        if score >= threshold and score > best_score:
            best_score = score
            best_id = group_id
    return best_id


def _resolve_speaker(request: ChatRequest, buffer, speaker_store) -> tuple[str | None, str | None]:
    """Resolve speaker identity from multiple sources.

    Returns (speaker_id, speaker_name) tuple.

    Priority:
    1. Voice embedding match (via SpeakerStore, high confidence only)
    2. Session history (previously identified in this conversation)
    3. Anonymous (None, None)
    """
    # 1. Voice embedding match
    if request.speaker_embedding and speaker_store:
        match = speaker_store.match(request.speaker_embedding)
        if match.speaker_id and not match.tentative:
            buffer.set_speaker(request.conversation_id, match.speaker_id, match.name)
            # Enrich profile with this embedding (strengthens cross-device centroid)
            speaker_store.add_embedding(match.speaker_id, request.speaker_embedding)
            return match.speaker_id, match.name

    # 2. Previously identified in session
    existing_id = buffer.get_speaker_id(request.conversation_id)
    existing_name = buffer.get_speaker(request.conversation_id)
    if existing_id:
        return existing_id, existing_name

    return None, None


@app.get("/status", response_model=StatusResponse)
async def status():
    """Server health and state."""
    config = _state["config"]
    model = _state["model"]

    adapter_loaded = (
        hasattr(model, "peft_config") and "episodic" in model.peft_config if model else False
    )

    # Adapter inventory: enumerate configured kinds + interim capacity. Main
    # adapters contribute 1 each when enabled in yaml; interim contributes
    # max_interim_count (the capacity ceiling enforced by the VRAM validator).
    adapter_config_counts: dict[str, int] = {}
    for kind, cfg in (
        ("episodic", config.adapters.episodic),
        ("semantic", config.adapters.semantic),
        ("procedural", config.adapters.procedural),
    ):
        if cfg.enabled:
            adapter_config_counts[kind] = 1
    if config.consolidation.max_interim_count > 0:
        adapter_config_counts["interim"] = config.consolidation.max_interim_count

    # Currently active adapter on the live PeftModel. PEFT only keeps one
    # adapter active at a time (set_adapter / switch_adapter). None when the
    # model hasn't loaded (cloud-only) or no adapters exist yet.
    active_adapter: str | None = None
    if model is not None and hasattr(model, "active_adapter"):
        raw_active = model.active_adapter
        if isinstance(raw_active, list):
            active_adapter = raw_active[0] if raw_active else None
        else:
            active_adapter = raw_active

    keys_count = 0
    if config.registry_path.exists():
        from paramem.backup.encryption import read_maybe_encrypted as _rme

        registry = json.loads(_rme(config.registry_path).decode("utf-8"))
        keys_count = len(registry)

    # Session buffer summary (pending counts, orphan attribution, age)
    buf = _state.get("session_buffer")
    summary = (
        buf.get_summary()
        if buf
        else {
            "total": 0,
            "orphaned": 0,
            "oldest_age_seconds": None,
            "per_speaker": {},
            "per_source_type": {},
        }
    )
    _per_source_type: dict = summary.get("per_source_type") or {}
    pending_documents: int = _per_source_type.get("document", 0)
    pending_transcripts: int = _per_source_type.get("transcript", 0)

    # Per-speaker profile snapshot enriched with pending-session counts
    store = _state.get("speaker_store")
    speaker_rows: list[dict] = []
    if store is not None:
        per_speaker = summary["per_speaker"]
        for prof in store.list_profiles():
            prof["pending"] = per_speaker.get(prof["id"], 0)
            speaker_rows.append(prof)

    # Next scheduled run — sourced from the systemd user timer (wall-clock,
    # survives server restart). See paramem/server/systemd_timer.py.
    # Cached for 5s because /status is polled frequently (HA, pstatus) and
    # `systemctl show` forks a subprocess on every call.
    from paramem.server import systemd_timer

    timer_state = systemd_timer.cached_timer_state(max_age_seconds=5)
    next_run_seconds: int | None = None
    scheduler_active = bool(timer_state.get("active", False))
    next_us = timer_state.get("next_elapse_us") or ""
    # systemd uses UINT64_MAX as the "no next elapse" sentinel. Treat any
    # timestamp > 100 years from now (1e11 seconds) as "not scheduled".
    if next_us.isdigit():
        next_epoch = int(next_us) / 1_000_000
        if next_epoch - time.time() < 3.15e9:  # < ~100 years ahead
            next_run_seconds = max(0, int(next_epoch - time.time()))

    # Next interim bucket boundary: post_session_train floors its stamp to
    # the refresh_cadence boundary measured from midnight, so the next
    # boundary is fully deterministic from the clock. None when cadence is
    # disabled (manual-only mode).
    from paramem.server.interim_adapter import compute_schedule_period_seconds

    next_interim_seconds: int | None = None
    _refresh_seconds = compute_schedule_period_seconds(config.consolidation.refresh_cadence)
    if _refresh_seconds and _refresh_seconds > 0:
        _now = datetime.now()
        _midnight = _now.replace(hour=0, minute=0, second=0, microsecond=0)
        _since_mid = int((_now - _midnight).total_seconds())
        _next_boundary = ((_since_mid // _refresh_seconds) + 1) * _refresh_seconds
        next_interim_seconds = max(0, _next_boundary - _since_mid)

    # Background trainer
    bt = _state.get("background_trainer")
    bg_active = bool(bt and getattr(bt, "is_training", False))
    bg_adapter = getattr(bt, "current_adapter_name", None) if bg_active else None

    # Thermal-throttle / quiet-hours snapshot. Read from the loaded config so the
    # block is present even before the BackgroundTrainer has been constructed.
    # ``currently_throttling`` reflects the policy gate only; the actual throttle
    # additionally requires ``training_temp_limit > 0`` and temp above limit —
    # surfaced here is "would the policy allow throttling right now".
    from paramem.server.background_trainer import is_thermal_policy_active

    thermal_policy = {
        "mode": config.consolidation.quiet_hours_mode,
        "start": config.consolidation.quiet_hours_start,
        "end": config.consolidation.quiet_hours_end,
        "temp_limit": config.consolidation.training_temp_limit,
        "currently_throttling": is_thermal_policy_active(
            config.consolidation.quiet_hours_mode,
            config.consolidation.quiet_hours_start,
            config.consolidation.quiet_hours_end,
        ),
    }

    # Adapter health — stored in the KeyRegistry JSON (distinct from the
    # SimHash registry at config.registry_path).  Surfaced to pstatus so a
    # degenerated adapter is visible without grepping logs.
    adapter_health: dict = {}
    _key_reg_path = config.adapter_dir / "indexed_key_registry.json"
    if _key_reg_path.exists():
        try:
            from paramem.backup.encryption import read_maybe_encrypted

            _key_reg_json = json.loads(read_maybe_encrypted(_key_reg_path).decode("utf-8"))
            adapter_health = _key_reg_json.get("adapter_health", {}) or {}
        except Exception:
            adapter_health = {}

    # TTS inventory: which languages are loaded and on which device. When
    # voices span devices (one on CUDA, one on CPU) we report "mixed" so the
    # fallback path is visible in pstatus without dumping per-voice rows.
    tts_manager = _state.get("tts_manager")
    tts_loaded = bool(tts_manager and tts_manager.is_loaded)
    tts_languages: list[str] = tts_manager.available_languages if tts_loaded else []
    tts_device: str | None = None
    if tts_loaded:
        _tts_devices = set(tts_manager.engine_devices.values())
        if len(_tts_devices) == 1:
            tts_device = next(iter(_tts_devices))
        elif _tts_devices:
            tts_device = "mixed"

    stt = _state.get("stt")
    stt_loaded = stt is not None and stt.is_loaded
    # WhisperSTT keeps `self.device` as the RESOLVED device string (cuda/cpu)
    # by the time load() returns True — "auto" is reassigned before load.
    stt_device = stt.device if stt_loaded else None
    # Only one STT backend family is supported today (faster-whisper).
    stt_engine = "whisper" if stt_loaded else None

    # TTS engine family: derive from the class name of each loaded engine
    # (piper / mms_tts). "piper+mms" when voices span both backends.
    tts_engine: str | None = None
    if tts_loaded:
        _kinds: set[str] = set()
        for _eng in tts_manager._engines.values():
            _cls = type(_eng).__name__.lower()
            if "piper" in _cls:
                _kinds.add("piper")
            elif "mms" in _cls:
                _kinds.add("mms_tts")
        if len(_kinds) == 1:
            tts_engine = next(iter(_kinds))
        elif _kinds:
            tts_engine = "piper+mms"

    # Live device of the loaded LLM. Resolved from the first parameter's
    # device so we reflect actual placement, not the config intent (which
    # can diverge in cloud-only or CPU-fallback cases).
    model_device: str | None = None
    if model is not None:
        try:
            model_device = next(model.parameters()).device.type
        except (StopIteration, AttributeError):
            model_device = None

    # HF model identifier from the registry. Safe even in cloud-only mode —
    # model_config resolves off the registry and has no GPU dependency.
    model_id: str | None = None
    try:
        model_id = config.model_config.model_id
    except (KeyError, ValueError):
        model_id = None

    # Episodic adapter rank surfaces the primary knob for indexed-key recall.
    episodic_rank = config.adapters.episodic.rank if config.adapters.episodic.enabled else None

    # Per-kind adapter spec. Only include enabled kinds so pstatus doesn't
    # display disabled rows. target_kind compresses target_modules into a
    # category label — "attn+mlp" means MLP layers are in the set, else
    # "attn". A caller interested in the exact list can hit the yaml.
    def _target_kind(target_modules: list[str]) -> str:
        for t in target_modules or []:
            tl = t.lower()
            if "mlp" in tl or "gate" in tl or "up_proj" in tl or "down_proj" in tl:
                return "attn+mlp"
        return "attn"

    adapter_specs: dict[str, dict] = {}
    for _kind, _cfg in (
        ("episodic", config.adapters.episodic),
        ("semantic", config.adapters.semantic),
        ("procedural", config.adapters.procedural),
    ):
        if not _cfg.enabled:
            continue
        adapter_specs[_kind] = {
            "rank": _cfg.rank,
            "alpha": _cfg.alpha,
            "learning_rate": _cfg.learning_rate,
            "target_kind": _target_kind(_cfg.target_modules),
        }

    # Speaker-embedding backend. Only populate when the pyannote model is
    # actually loaded — disabled / failed-load paths leave the fields None
    # so pstatus can skip the row entirely.
    speaker_embedding_backend: str | None = None
    speaker_embedding_model: str | None = None
    speaker_embedding_device: str | None = None
    try:
        from paramem.server import speaker_embedding as _spk_emb

        if _spk_emb.is_loaded():
            speaker_embedding_backend = _spk_emb.EMBEDDING_BACKEND
            speaker_embedding_model = _spk_emb.EMBEDDING_MODEL_NAME
            speaker_embedding_device = _spk_emb.EMBEDDING_DEVICE
    except ImportError:
        pass

    # Attention block.
    from paramem.server.attention import collect_attention_items

    _attention_items = collect_attention_items(_state, config)
    attention_block = {"items": [it.to_dict() for it in _attention_items]}

    # Migration summary block.
    _mig = _state.get("migration") or {}
    _mig_state = (_mig.get("state") or "LIVE").lower()
    _trial = _mig.get("trial") or {}
    _gates = _trial.get("gates") or None
    # config_rev: first 8 hex chars of sha256(server.yaml at load time).
    _loaded_hash = (_state.get("config_drift") or {}).get("loaded_hash", "")
    _config_rev = _loaded_hash[:8] if _loaded_hash else ""
    # Mirror accept-eligibility gate so /status agrees with /migration/status.
    _comparison_block: dict | None = None
    _ACCEPT_ELIGIBLE_MIG = frozenset({"pass", "no_new_sessions"})
    if (
        _mig_state == "trial"
        and _gates is not None
        and _gates.get("status") in _ACCEPT_ELIGIBLE_MIG
        and _gates.get("completed_at")
    ):
        _comparison_block = {"rendered": True, "flags": []}
    migration_block = {
        "state": _mig_state,
        "config_rev": _config_rev,
        "trial_started_at": _trial.get("started_at") or None,
        "gates": _gates,
        "comparison": _comparison_block,
    }

    hold_block = _get_hold_state()

    # Backup block.  Reads state/backup.json (written by the runner),
    # computes current disk usage, derives next-scheduled-at and stale flag.
    # The entire block is guarded: a MagicMock config (used in unit tests) or
    # any transient I/O error must not crash /status — fall back to an empty
    # default BackupBlock instead.
    backup_block = BackupBlock()
    try:
        from paramem.backup import retention as _backup_retention
        from paramem.backup import state as _backup_state
        from paramem.backup import timer as _backup_timer
        from paramem.backup.timer import _backup_timer_interval_seconds

        _backups_root = (config.paths.data / "backups").resolve()
        _state_dir = (config.paths.data / "state").resolve()

        # Read persisted runner state — None when no run has ever happened.
        _backup_record = None
        try:
            _backup_record = _backup_state.read_backup_state(_state_dir)
        except Exception:
            logger.exception("Failed to read backup state — surfacing empty block")

        # Disk usage — always fresh (TTL-cached in retention module).
        _disk_used_bytes: int = 0
        _disk_cap_bytes: int = 0
        try:
            _disk_usage = _backup_retention.compute_disk_usage(
                _backups_root, config.security.backups
            )
            _disk_used_bytes = _disk_usage.total_bytes
            _disk_cap_bytes = _disk_usage.cap_bytes
        except Exception:
            logger.exception("Failed to compute backup disk usage — defaulting to 0")
            try:
                _disk_cap_bytes = int(config.security.backups.max_total_disk_gb * 1024**3)
            except Exception:
                _disk_cap_bytes = 0

        # Next scheduled — read from the live backup timer state when installed.
        _backup_timer_state = _backup_timer.cached_timer_state(max_age_seconds=5)
        _next_scheduled_at: str | None = None
        _next_us = _backup_timer_state.get("next_elapse_us") or ""
        if str(_next_us).isdigit():
            _next_epoch = int(_next_us) / 1_000_000
            if _next_epoch - time.time() < 3.15e9:  # sanity: within ~100 years
                _next_scheduled_at = datetime.fromtimestamp(
                    _next_epoch, tz=timezone.utc
                ).isoformat()

        # Stale — last success older than 2× cadence interval.  False when
        # schedule=off or last_success_at is None.
        _raw_schedule = config.security.backups.schedule
        _schedule_str = (str(_raw_schedule) if _raw_schedule else "").strip().lower()
        _stale = False
        if (
            _backup_record
            and _backup_record.last_success_at
            and _schedule_str
            not in (
                "",
                "off",
                "disabled",
                "none",
            )
        ):
            _interval_s = _backup_timer_interval_seconds(_schedule_str)
            if _interval_s and _interval_s > 0:
                try:
                    _last_ok = datetime.fromisoformat(_backup_record.last_success_at)
                    if _last_ok.tzinfo is None:
                        _last_ok = _last_ok.replace(tzinfo=timezone.utc)
                    _age = (datetime.now(timezone.utc) - _last_ok).total_seconds()
                    _stale = _age > 2 * _interval_s
                except Exception:
                    pass  # malformed timestamp — leave stale=False

        backup_block = BackupBlock(
            schedule=str(_raw_schedule) if _raw_schedule else "",
            last_success_at=_backup_record.last_success_at if _backup_record else None,
            last_failure_at=_backup_record.last_failure_at if _backup_record else None,
            last_failure_reason=(_backup_record.last_failure_reason if _backup_record else None),
            next_scheduled_at=_next_scheduled_at,
            stale=_stale,
            disk_used_bytes=_disk_used_bytes,
            disk_cap_bytes=_disk_cap_bytes,
        )
    except Exception:
        logger.exception("Failed to build backup block — returning empty default")

    return StatusResponse(
        model=config.model_name,
        model_id=model_id,
        model_device=model_device,
        episodic_rank=episodic_rank,
        adapter_specs=adapter_specs,
        speaker_embedding_backend=speaker_embedding_backend,
        speaker_embedding_model=speaker_embedding_model,
        speaker_embedding_device=speaker_embedding_device,
        stt_engine=stt_engine,
        tts_engine=tts_engine,
        mode=_state["mode"],
        cloud_only_reason=_state.get("cloud_only_reason"),
        adapter_loaded=adapter_loaded,
        adapter_config=adapter_config_counts,
        active_adapter=active_adapter,
        keys_count=keys_count,
        pending_sessions=summary["total"],
        consolidating=_state["consolidating"],
        last_consolidation=_state["last_consolidation"],
        speaker_profiles=store.profile_count if store else 0,
        stt_loaded=stt_loaded,
        stt_model=stt.model_name if stt_loaded else None,
        stt_device=stt_device,
        tts_loaded=tts_loaded,
        tts_languages=tts_languages,
        tts_device=tts_device,
        refresh_cadence=config.consolidation.refresh_cadence,
        consolidation_period=config.consolidation.consolidation_period_string,
        max_interim_count=config.consolidation.max_interim_count,
        mode_config=config.consolidation.mode,
        next_run_seconds=next_run_seconds,
        next_interim_seconds=next_interim_seconds,
        orphaned_pending=summary["orphaned"],
        oldest_pending_seconds=summary["oldest_age_seconds"],
        speakers=speaker_rows,
        bg_trainer_active=bg_active,
        bg_trainer_adapter=bg_adapter,
        thermal_policy=thermal_policy,
        last_consolidation_result=_state.get("last_consolidation_result"),
        pending_enrollments=len(_state.get("pending_enrollments") or []),
        scheduler_started=scheduler_active,
        adapter_health=adapter_health,
        adapter_manifest=_state.get("adapter_manifest_status", {}),
        config_drift=_state.get("config_drift", {}),
        attention=attention_block,
        migration=migration_block,
        backup=backup_block,
        hold=hold_block,
        encryption=_state.get("encryption", "off"),
        server_started_at=_state.get("server_started_at", ""),
        preview=bool(_state.get("preview", False)),
        pending_documents=pending_documents,
        pending_transcripts=pending_transcripts,
    )


@app.post("/gpu/force-local")
async def gpu_force_local():
    """Operator override: drop any PARAMEM_EXTRA_ARGS=--defer-model hold and,
    if the current process is in defer mode, restart so the next boot loads
    the model locally.

    Called by ``pstatus --force-local``.  Use when /status or pstatus reports
    an orphaned hold (``hold.owner_alive == false`` or ``owner_pid == null``
    after auto-reclaim has stopped).  Idempotent: safe when no hold is set.
    """
    hold_before = _get_hold_state()
    cleared = _clear_hold_env()
    will_restart = bool(_state.get("defer_model", False))
    if cleared and will_restart:
        _restart_service()
    return {
        "cleared": cleared,
        "was_active": hold_before["hold_active"],
        "owner_pid": hold_before["owner_pid"],
        "owner_alive": hold_before["owner_alive"],
        "will_restart": cleared and will_restart,
    }


@app.post("/refresh-ha")
async def refresh_ha():
    """Rebuild the HA entity graph from the HA API.

    Call after adding/removing devices, renaming entities, or
    reorganizing areas in Home Assistant.
    """
    ha_client = _state.get("ha_client")
    if ha_client is None:
        return {"status": "not_configured"}

    ha_client.load_entity_map()
    ha_services = ha_client.get_services()
    ha_graph = _state.get("ha_graph")
    if ha_graph is not None:
        ha_graph.refresh(ha_client._raw_states, ha_services)
    else:
        ha_graph = HAEntityGraph.build(ha_client._raw_states, ha_services)
        _state["ha_graph"] = ha_graph
        # Rebuild router with new graph
        config = _state["config"]
        _state["router"] = QueryRouter(
            adapter_dir=config.adapter_dir,
            ha_graph=ha_graph,
        )

    return {
        "status": "refreshed",
        "entities": ha_graph.entity_count,
        "areas": ha_graph.area_count,
        "verbs": ha_graph.verb_count,
    }


@app.post("/debug/assign-orphans")
async def debug_assign_orphans(speaker_id: str | None = None):
    """Test-only: attribute all orphan sessions to a single speaker.

    Requires debug mode. Uses the given speaker_id, or the first
    enrolled profile if omitted. Mutates in-memory turns and, in
    debug persistence mode, rewrites the jsonl files on disk.
    """
    config = _state["config"]
    if not getattr(config, "debug", False):
        return {"status": "forbidden_not_debug"}
    store = _state.get("speaker_store")
    buffer = _state.get("session_buffer")
    if store is None or buffer is None:
        return {"status": "not_ready"}
    profiles = store.list_profiles()
    if not profiles:
        return {"status": "no_speakers_enrolled"}
    target = next((p for p in profiles if p["id"] == speaker_id), profiles[0])
    sid, sname = target["id"], target["name"]
    claimed = 0
    for conv_id, turns in buffer._turns.items():
        if any(t.get("speaker_id") for t in turns):
            continue
        for turn in turns:
            turn["speaker"] = sname
            turn["speaker_id"] = sid
        claimed += 1
        if buffer.debug:
            path = buffer.session_dir / f"{conv_id}.jsonl"
            if path.exists():
                with open(path, "w") as f:
                    for turn in turns:
                        f.write(json.dumps(turn) + "\n")
    logger.info("debug/assign-orphans: %d sessions → speaker %s (%s)", claimed, sname, sid)
    return {"status": "ok", "claimed": claimed, "speaker": sname, "speaker_id": sid}


@app.post("/scheduled-tick", response_model=ConsolidateResponse)
async def scheduled_tick():
    """Systemd user-timer entrypoint (paramem-consolidate.timer).

    Runs the cooperative extract + background-train path. If the GPU is
    unavailable (cloud-only or bg training active), returns a 'deferred'
    status — the timer will fire again on its next wall-clock tick.

    Returns 409 ``trial_active`` when a migration TRIAL is in progress
    (spec §L302–303 — "refuses new cycles" while TRIAL is active).
    """
    from fastapi import HTTPException

    # Guard: block new cycles while a migration TRIAL is active.
    migration = _state.get("migration", {})
    if migration.get("state") == "TRIAL":
        raise HTTPException(
            status_code=409,
            detail={
                "error": "trial_active",
                "message": (
                    "A migration TRIAL is active — consolidation is blocked. "
                    "Use POST /migration/accept or POST /migration/rollback to proceed."
                ),
            },
        )

    _state["scheduler_last_tick_epoch"] = time.time()
    status = _maybe_trigger_scheduled_consolidation()
    _state["scheduler_last_tick_status"] = status
    return ConsolidateResponse(status=status)


@app.post("/consolidate", response_model=ConsolidateResponse)
async def consolidate():
    """Trigger consolidation manually.

    Returns 409 ``trial_active`` when a migration TRIAL is in progress.
    """
    from fastapi import HTTPException

    # Guard: block consolidation while a migration TRIAL is active.
    migration = _state.get("migration", {})
    if migration.get("state") == "TRIAL":
        raise HTTPException(
            status_code=409,
            detail={
                "error": "trial_active",
                "message": (
                    "A migration TRIAL is active — consolidation is blocked. "
                    "Use POST /migration/accept or POST /migration/rollback to proceed."
                ),
            },
        )

    if _state["mode"] == "cloud-only":
        return ConsolidateResponse(status="rejected_cloud_only")
    if _state["consolidating"]:
        return ConsolidateResponse(status="already_running")

    bg_trainer = _state.get("background_trainer")
    if bg_trainer is not None and bg_trainer.is_training:
        return ConsolidateResponse(status="training_active")

    _state["consolidating"] = True
    loop = asyncio.get_running_loop()
    future = loop.run_in_executor(None, _run_consolidation_sync)
    future.add_done_callback(_consolidation_done_callback)

    return ConsolidateResponse(status="started")


# --- Document ingest endpoints ---


@app.post("/ingest-sessions", response_model=IngestSessionsResponse)
async def ingest_sessions(request: IngestSessionsRequest):
    """Queue pre-chunked document segments for consolidation.

    Each chunk becomes a separate session in the ``SessionBuffer`` with
    ``source_type="document"``.  The endpoint is idempotent: chunks whose
    SHA-256 fingerprint is already in the ingest registry are silently
    skipped (``registry_skipped`` counter).

    Errors
    ------
    400
        ``speaker_id`` is an empty string.
    404
        ``speaker_id`` not found in ``SpeakerStore``.
    409
        A migration TRIAL is in progress.

    Args:
        request: Payload containing ``speaker_id`` and a list of
            :class:`IngestChunk` items.

    Returns:
        :class:`IngestSessionsResponse` with session IDs enqueued,
        skip counts, and rejection flags.
    """
    from paramem.server.document_ingest import IngestRegistry, _now_iso8601, normalize_chunk_text

    total_chunks = len(request.sessions)

    # Gate 1: empty speaker_id
    if not request.speaker_id:
        return JSONResponse(
            status_code=400,
            content=IngestSessionsResponse(
                queued=[],
                total_chunks=total_chunks,
                registry_skipped=0,
                rejected_no_speaker_id=True,
            ).model_dump(),
        )

    # Gate 2: unknown speaker
    store = _state.get("speaker_store")
    speaker_name: str | None = None
    if store is not None:
        profiles = store.list_profiles()
        matched = next((p for p in profiles if p["id"] == request.speaker_id), None)
        if matched is None:
            return JSONResponse(
                status_code=404,
                content=IngestSessionsResponse(
                    queued=[],
                    total_chunks=total_chunks,
                    registry_skipped=0,
                    rejected_unknown_speaker=True,
                ).model_dump(),
            )
        speaker_name = matched.get("name", "")
    else:
        # No speaker store — treat as unknown
        return JSONResponse(
            status_code=404,
            content=IngestSessionsResponse(
                queued=[],
                total_chunks=total_chunks,
                registry_skipped=0,
                rejected_unknown_speaker=True,
            ).model_dump(),
        )

    # Gate 3: migration TRIAL in progress
    migration = _state.get("migration") or {}
    if migration.get("state") == "TRIAL":
        from fastapi import HTTPException

        raise HTTPException(
            status_code=409,
            detail={
                "error": "trial_active",
                "message": (
                    "A migration TRIAL is active — ingest is blocked. "
                    "Use POST /migration/accept or POST /migration/rollback to proceed."
                ),
            },
        )

    config = _state["config"]
    buffer: SessionBuffer = _state["session_buffer"]

    registry_path = Path(config.paths.sessions) / ".ingest_registry.json"
    registry = IngestRegistry(registry_path)

    queued: list[str] = []
    registry_skipped = 0

    for chunk in request.sessions:
        normalized = normalize_chunk_text(chunk.chunk)
        chunk_hash = registry.chunk_hash(
            speaker_id=request.speaker_id,
            source_path=chunk.source,
            chunk_index=chunk.chunk_index,
            normalized_text=normalized,
            source_type=chunk.source_type,
        )
        if registry.is_known(chunk_hash):
            registry_skipped += 1
            continue

        session_id = "doc-" + secrets.token_hex(4)

        # set_speaker must precede append so get_pending finds speaker_id
        buffer.set_speaker(session_id, request.speaker_id, speaker_name or "")
        buffer.append(
            session_id,
            "user",
            chunk.chunk,
            embedding=None,
            metadata={
                "source_type": "document",
                "doc_title": chunk.doc_title,
                "chunk_index": chunk.chunk_index,
                "source_path": chunk.source,
            },
        )

        registry.record(
            chunk_hash,
            session_id=session_id,
            speaker_id=request.speaker_id,
            source_path=chunk.source,
            chunk_index=chunk.chunk_index,
            source_type="document",
            doc_title=chunk.doc_title,
            ingested_at=_now_iso8601(),
        )
        queued.append(session_id)

    registry.flush()

    return IngestSessionsResponse(
        queued=queued,
        total_chunks=total_chunks,
        registry_skipped=registry_skipped,
    )


@app.post("/ingest-sessions/cancel", response_model=IngestCancelResponse)
async def ingest_sessions_cancel(request: IngestCancelRequest):
    """Discard queued ingest sessions without running consolidation.

    Calls :meth:`SessionBuffer.discard_sessions` (not ``mark_consolidated``)
    so the operator can cleanly remove document chunks they no longer want
    to ingest, without any implication that consolidation occurred.

    Args:
        request: Payload with a list of session IDs to cancel.

    Returns:
        :class:`IngestCancelResponse` splitting the requested IDs into
        ``cancelled`` (found and removed) and ``not_found`` (unknown).
    """
    buffer: SessionBuffer = _state["session_buffer"]

    # Snapshot before so we can classify each id as found or not-found
    before: set[str] = set(buffer._turns.keys())

    buffer.discard_sessions(request.session_ids)

    cancelled = [sid for sid in request.session_ids if sid in before]
    not_found = [sid for sid in request.session_ids if sid not in before]

    return IngestCancelResponse(cancelled=cancelled, not_found=not_found)


# --- Migration endpoints ---


@app.post("/migration/preview", response_model=PreviewResponse)
async def migration_preview(request: PreviewRequest):
    """Stage a candidate ``server.yaml`` and return a preview diff.

    Validates the candidate path, parses the YAML, computes the unified diff,
    tier-classified change list, and shape-change block, then stores the stash
    in ``_state["migration"]`` with ``state="STAGING"``.  **No files are
    written** — disk writes, atomic swap, trial markers, and TRIAL state are
    handled by ``/migration/confirm``.

    Concurrency note: ``_state["consolidating"]`` is read once at the top of
    this handler.  The flag is mutated from a mix of event-loop callbacks and
    worker threads (see ``paramem/server/app.py`` for the ~10 mutation sites —
    several are inside executor/worker-thread callbacks such as
    ``_extract_and_start_training`` no-data branches and ``_on_training_error``).
    A single ``bool`` read under CPython's GIL is atomic, so the worst case is
    observing a stale value across the read-to-action gap — at most a few
    microseconds of mutex slack between consolidation and migration.  This is
    acceptable for the STAGING-only preview gate.  ``/migration/confirm``
    tightens the mutex with a CAS-style transition on
    ``_state["migration"]["state"]``.

    Errors
    ------
    400 ``candidate_path_invalid``
        Relative, missing, unreadable, cross-filesystem, or not a regular file.
    400 ``candidate_unparseable``
        ``yaml.safe_load`` raised on the candidate bytes (Condition 7).
    409 ``consolidating``
        A consolidation run is currently in progress.
    409 ``already_staging``
        The migration stash is already in ``STAGING`` state.
    409 ``trial_active``
        A trial is in progress.
    """
    from fastapi import HTTPException

    from paramem.server.migration import (
        _parse_candidate,
        _sha256_bytes,
        compute_shape_changes,
        compute_tier_diff,
        compute_unified_diff,
        detect_simulate_mode,
        initial_migration_state,
        render_preview_response,
        validate_candidate_path,
    )

    # --- Concurrency gate: read once, race-free under cooperative scheduling ---
    if _state.get("consolidating", False):
        raise HTTPException(
            status_code=409,
            detail={
                "error": "consolidating",
                "message": "Consolidation is currently running. Retry after it completes.",
            },
        )

    migration = _state.get("migration") or initial_migration_state()
    current_state = migration.get("state", "LIVE")

    if current_state == "STAGING":
        raise HTTPException(
            status_code=409,
            detail={
                "error": "already_staging",
                "message": "A candidate is already staged. POST /migration/cancel first.",
            },
        )

    if current_state == "TRIAL":
        raise HTTPException(
            status_code=409,
            detail={
                "error": "trial_active",
                "message": "A trial is active. Accept or rollback before staging a new candidate.",
            },
        )

    # --- Validate path ---
    config = _state.get("config")
    live_config_path = (
        Path(_state["config_path"]) if _state.get("config_path") else Path("configs/server.yaml")
    )

    try:
        candidate_path = validate_candidate_path(request.candidate_path, live_config_path)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={"error": "candidate_path_invalid", "message": str(exc)},
        ) from exc

    # --- Read candidate ---
    candidate_bytes = candidate_path.read_bytes()
    candidate_hash = _sha256_bytes(candidate_bytes)
    try:
        candidate_text = candidate_bytes.decode("utf-8")
    except UnicodeDecodeError:
        candidate_text = candidate_bytes.decode("latin-1")

    # --- Parse candidate (yaml.safe_load, NOT load_server_config) ---
    try:
        parsed_candidate = _parse_candidate(candidate_bytes)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={"error": "candidate_unparseable", "message": str(exc)},
        ) from exc

    # --- Read live yaml for diff ---
    live_text = live_config_path.read_text(encoding="utf-8") if live_config_path.exists() else ""
    import yaml as _yaml

    live_yaml = _yaml.safe_load(live_text.encode("utf-8")) if live_text else {}
    if not isinstance(live_yaml, dict):
        live_yaml = {}

    # --- Compute diffs ---
    unified_diff = compute_unified_diff(live_text, candidate_text)
    tier_diff = compute_tier_diff(live_yaml, parsed_candidate)

    # --- Shape-change detection ---
    adapter_dir = config.adapter_dir if config is not None else Path("data/ha/adapters")
    live_registry_sha256 = ""
    if config is not None:
        registry_path = None
        try:
            if hasattr(config, "paths") and config.paths.data is not None:
                registry_path = config.paths.key_metadata
        except (AttributeError, TypeError):
            registry_path = None
        if registry_path is None:
            registry_path = Path(str(adapter_dir)).parent / "registry" / "key_metadata.json"
        if registry_path.exists():
            # Hash plaintext content — see manifest.py::build_manifest_for.
            import hashlib as _rhash

            from paramem.backup.encryption import read_maybe_encrypted as _rme

            try:
                live_registry_sha256 = _rhash.sha256(_rme(registry_path)).hexdigest()
            except Exception:  # noqa: BLE001
                live_registry_sha256 = ""

    shape_changes = compute_shape_changes(parsed_candidate, adapter_dir, live_registry_sha256)

    # --- Detect simulate-mode ---
    simulate_mode_override = detect_simulate_mode(parsed_candidate)

    # --- Pre-flight: disk-pressure gate on backup store ---
    # compute_pre_flight_check guards itself against MagicMock / non-real configs
    # (returns no-pressure result when max_total_disk_gb is not a real numeric).
    # No call-site guard needed here.
    from paramem.backup.preflight import compute_pre_flight_check as _compute_pre_flight
    from paramem.server.migration import MigrationStashState

    _registry_path_for_pf = None
    try:
        if config is not None and hasattr(config, "paths") and config.paths.data is not None:
            _registry_path_for_pf = config.paths.key_metadata
    except (AttributeError, TypeError):
        _registry_path_for_pf = None

    try:
        _backups_root_for_pf = (config.paths.data / "backups").resolve()
    except (AttributeError, TypeError):
        _backups_root_for_pf = Path("data/ha/backups").resolve()

    pre_flight = None
    try:
        pre_flight = _compute_pre_flight(
            server_config=config,
            loop=_state.get("consolidation_loop"),
            backups_root=_backups_root_for_pf,
            live_config_path=live_config_path,
            registry_path=_registry_path_for_pf,
        )
    except Exception:
        # Defensive: pre-flight failure must not block the preview entirely.
        pre_flight = None

    if pre_flight is not None and pre_flight.fail_code is not None:
        # State stays LIVE (Decision A) — do NOT store the stash.
        # Build a preview-only (non-stored) stash for render_preview_response.
        now_iso = datetime.now(timezone.utc).isoformat()
        preview_stash = MigrationStashState(
            state="LIVE",
            candidate_path=str(candidate_path),
            candidate_hash=candidate_hash,
            candidate_bytes=candidate_bytes,
            candidate_text=candidate_text,
            parsed_candidate=parsed_candidate,
            staged_at=now_iso,
            simulate_mode_override=simulate_mode_override,
            shape_changes=shape_changes,
            tier_diff=tier_diff,
            unified_diff=unified_diff,
            trial=None,
            recovery_required=list(_state.get("migration", {}).get("recovery_required", [])),
        )
        payload = render_preview_response(preview_stash, pre_flight_fail=pre_flight.fail_code)
        payload["pre_flight_disk_used_gb"] = pre_flight.disk_used_bytes / (1024**3)
        payload["pre_flight_disk_cap_gb"] = pre_flight.disk_cap_bytes / (1024**3)
        return PreviewResponse(**payload)

    # --- Build stash (pre-flight passed) ---
    now_iso = datetime.now(timezone.utc).isoformat()

    stash = MigrationStashState(
        state="STAGING",
        candidate_path=str(candidate_path),
        candidate_hash=candidate_hash,
        candidate_bytes=candidate_bytes,
        candidate_text=candidate_text,
        parsed_candidate=parsed_candidate,
        staged_at=now_iso,
        simulate_mode_override=simulate_mode_override,
        shape_changes=shape_changes,
        tier_diff=tier_diff,
        unified_diff=unified_diff,
        trial=None,
        recovery_required=list(_state.get("migration", {}).get("recovery_required", [])),
    )
    _state["migration"] = stash

    payload = render_preview_response(stash, pre_flight_fail=None)
    return PreviewResponse(**payload)


@app.post("/migration/cancel", response_model=MigrationCancelResponse)
async def migration_cancel():
    """Clear the staged candidate and return to LIVE state.

    Returns the candidate path that was discarded so the caller can confirm
    which staging session was cancelled.

    Errors
    ------
    409 ``not_staging``
        The server is not currently in STAGING state (nothing to cancel).
    """
    from fastapi import HTTPException

    from paramem.server.migration import initial_migration_state

    migration = _state.get("migration") or initial_migration_state()
    current_state = migration.get("state", "LIVE")

    if current_state != "STAGING":
        raise HTTPException(
            status_code=409,
            detail={
                "error": "not_staging",
                "message": "No candidate is staged; nothing to cancel.",
            },
        )

    cleared_path = migration.get("candidate_path", "")
    _state["migration"] = initial_migration_state()
    return MigrationCancelResponse(state="LIVE", cleared_path=cleared_path)


@app.post("/migration/confirm", response_model=ConfirmResponse)
async def migration_confirm(request: ConfirmRequest):
    """Atomically transition from STAGING to TRIAL.

    Implements the 5-step atomic ordering (spec §L277–283):

    1. Acquire migration lock + verify STAGING + verify not consolidating.
    2. Write 3 pre-migration backup slots (config, graph, registry).
    3. Write ``state/trial.json`` marker.
    4. ``os.rename(candidate → configs/server.yaml)`` — atomic config swap.
    5. Set ``_state["migration"]["state"] = "TRIAL"``; kick off trial
       consolidation via ``asyncio.create_task``.

    Each step's failure rolls back all previously-completed steps and returns
    an appropriate 5xx error.  The migration lock is unconditionally released
    in a ``finally`` block (Correction 1).

    Errors
    ------
    409 ``consolidating``
        A consolidation run is in progress.
    409 ``not_staging``
        The server is not in STAGING state.
    409 ``migration_in_progress``
        A concurrent confirm is already holding the lock.
    409 ``trial_active``
        The server is already in TRIAL (post-recovery edge case).
    500 ``backup_write_failed``
        Step 2 failed; no state change.
    500 ``marker_write_failed``
        Step 3 failed; step-2 backups deleted.
    500 ``config_swap_failed``
        Step 4 failed; marker and backups deleted.
    """
    from fastapi import HTTPException

    from paramem.server.migration import TrialStash, initial_migration_state

    # --- Step 1: Pre-checks (outside the lock for fast fail) ---
    if _state.get("consolidating", False):
        raise HTTPException(
            status_code=409,
            detail={
                "error": "consolidating",
                "message": "Consolidation is currently running. Retry after it completes.",
            },
        )

    migration = _state.get("migration") or initial_migration_state()
    current_state = migration.get("state", "LIVE")

    if current_state == "TRIAL":
        raise HTTPException(
            status_code=409,
            detail={
                "error": "trial_active",
                "message": (
                    "A trial is already active. "
                    "Accept or rollback before confirming a new candidate."
                ),
            },
        )

    if current_state != "STAGING":
        raise HTTPException(
            status_code=409,
            detail={
                "error": "not_staging",
                "message": "No candidate is staged. POST /migration/preview first.",
            },
        )

    # --- Acquire the migration lock (non-blocking try) ---
    lock: asyncio.Lock = _state.get("migration_lock") or asyncio.Lock()
    if lock.locked():
        raise HTTPException(
            status_code=409,
            detail={
                "error": "migration_in_progress",
                "message": "A migration confirm is already in progress.",
            },
        )

    # Prepare paths from config.
    config = _state.get("config")
    live_config_path = (
        Path(_state["config_path"]) if _state.get("config_path") else Path("configs/server.yaml")
    )
    if config is not None:
        state_dir = (config.paths.data / "state").resolve()
        backups_root = (config.paths.data / "backups").resolve()
    else:
        state_dir = (Path("data/ha/state")).resolve()
        backups_root = (Path("data/ha/backups")).resolve()

    trial_adapter_dir = str((state_dir / "trial_adapter").resolve())
    trial_graph_dir = str((state_dir / "trial_graph").resolve())

    async with lock:
        # Re-check inside the lock (state may have changed while waiting).
        migration = _state.get("migration") or initial_migration_state()
        current_state = migration.get("state", "LIVE")
        if current_state == "TRIAL":
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "trial_active",
                    "message": "State changed to TRIAL while acquiring lock.",
                },
            )
        if current_state != "STAGING":
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "not_staging",
                    "message": "Staging state was lost while acquiring lock.",
                },
            )
        if _state.get("consolidating", False):
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "consolidating",
                    "message": "Consolidation started while acquiring lock.",
                },
            )

        # Snapshot the STAGING stash fields we need.
        candidate_path_str = migration.get("candidate_path", "")
        candidate_hash = migration.get("candidate_hash", "")

        # Steps 2–4 are wrapped in try/finally so the lock is always released
        # even on partial failure (Correction 1).
        now_iso = datetime.now(timezone.utc).isoformat()

        # --- Step 2: snapshot pre-trial hashes, write 3 backups ---
        pre_trial_hash = ""
        if live_config_path.exists():
            pre_trial_hash = _hashlib.sha256(live_config_path.read_bytes()).hexdigest()

        written_slots: list[Path] = []
        try:
            # 2a: Config backup — read live config bytes.
            config_bytes = live_config_path.read_bytes() if live_config_path.exists() else b""
            config_slot = backup_write(
                ArtifactKind.CONFIG,
                config_bytes,
                meta_fields={"tier": "pre_migration", "pre_trial_hash": pre_trial_hash},
                base_dir=backups_root / "config",
            )
            written_slots.append(config_slot)

            # 2b: Graph backup — use merger.save_bytes() when loop is available.
            loop_obj = _state.get("consolidation_loop")
            if loop_obj is not None and hasattr(loop_obj, "merger"):
                graph_bytes = loop_obj.merger.save_bytes()
            else:
                graph_bytes = b"{}"
            graph_slot = backup_write(
                ArtifactKind.GRAPH,
                graph_bytes,
                meta_fields={"tier": "pre_migration", "pre_trial_hash": pre_trial_hash},
                base_dir=backups_root / "graph",
            )
            written_slots.append(graph_slot)

            # 2c: Registry backup.
            if config is not None:
                registry_bytes = (
                    config.key_metadata_path.read_bytes()
                    if config.key_metadata_path.exists()
                    else b"{}"
                )
            else:
                registry_bytes = b"{}"
            registry_slot = backup_write(
                ArtifactKind.REGISTRY,
                registry_bytes,
                meta_fields={"tier": "pre_migration", "pre_trial_hash": pre_trial_hash},
                base_dir=backups_root / "registry",
            )
            written_slots.append(registry_slot)

        except Exception as exc:
            # Step 2 failure: clean up any written slots.
            for slot in written_slots:
                try:
                    shutil.rmtree(slot)
                except OSError:
                    pass
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "backup_write_failed",
                    "message": f"Failed to write pre-migration backups: {exc}",
                },
            ) from exc

        # --- Step 3: Write trial marker ---
        # REQUIRED FIX 1: capture config artifact filename so the rollback
        # handler can resolve the exact A-config file without directory listing.
        # Filter uses endswith(".meta.json") to exclude all sidecar variants
        # (e.g. "config-<ts>.meta.json") regardless of prefix — the old
        # exact-match "meta.json" filter missed prefixed sidecars and caused
        # rollback to restore the sidecar JSON instead of the config artifact
        # (B1 bug, 2026-04-22 E2E baseline).
        config_artifact_filename = ""
        for _entry in Path(config_slot).iterdir():
            if not _entry.name.endswith(".meta.json") and not _entry.name.startswith("."):
                config_artifact_filename = _entry.name
                break

        marker = TrialMarker(
            schema_version=1,
            started_at=now_iso,
            pre_trial_config_sha256=pre_trial_hash,
            candidate_config_sha256=candidate_hash,
            backup_paths={
                "config": str(config_slot.resolve()),
                "graph": str(graph_slot.resolve()),
                "registry": str(registry_slot.resolve()),
            },
            trial_adapter_dir=trial_adapter_dir,
            trial_graph_dir=trial_graph_dir,
            config_artifact_filename=config_artifact_filename,
        )
        try:
            write_trial_marker(state_dir, marker)
        except Exception as exc:
            # Step 3 failure: delete step-2 backups.
            for slot in written_slots:
                try:
                    shutil.rmtree(slot)
                except OSError:
                    pass
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "marker_write_failed",
                    "message": f"Failed to write trial marker: {exc}",
                },
            ) from exc

        # --- Step 4: Atomic rename candidate → live config ---
        # Uses _rename_config (module-level) so tests can patch it independently
        # from os.rename (which backup.atomic also uses for its own renames).
        candidate_path = Path(candidate_path_str)
        try:
            _rename_config(candidate_path, live_config_path)
            # fsync parent for rename durability.
            dir_fd = os.open(str(live_config_path.parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            except OSError:
                pass
            finally:
                os.close(dir_fd)
        except Exception as exc:
            # Step 4 failure: delete marker and all backups.
            try:
                clear_trial_marker(state_dir)
            except OSError:
                pass
            for slot in written_slots:
                try:
                    shutil.rmtree(slot)
                except OSError:
                    pass
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "config_swap_failed",
                    "message": f"Atomic config rename failed: {exc}",
                },
            ) from exc

        # --- Step 5: Update _state and kick off trial consolidation ---
        trial_stash = TrialStash(
            started_at=now_iso,
            pre_trial_config_sha256=pre_trial_hash,
            candidate_config_sha256=candidate_hash,
            backup_paths={
                "config": str(config_slot.resolve()),
                "graph": str(graph_slot.resolve()),
                "registry": str(registry_slot.resolve()),
            },
            trial_adapter_dir=trial_adapter_dir,
            trial_graph_dir=trial_graph_dir,
            gates={"status": "pending"},
        )
        _state["migration"]["state"] = "TRIAL"
        _state["migration"]["trial"] = trial_stash
        _state["migration"]["recovery_required"] = []

        # Kick off trial consolidation as a background task.
        asyncio.create_task(_run_trial_consolidation())

        return ConfirmResponse(
            state="TRIAL",
            trial_started_at=now_iso,
            pre_trial_config_sha256=pre_trial_hash,
            candidate_config_sha256=candidate_hash,
            backup_paths={
                "config": str(config_slot.resolve()),
                "graph": str(graph_slot.resolve()),
                "registry": str(registry_slot.resolve()),
            },
            trial_adapter_dir=trial_adapter_dir,
            trial_graph_dir=trial_graph_dir,
        )


async def _run_trial_consolidation() -> None:
    """Run a trial consolidation cycle in the background.

    Acquires the GPU lock, reloads config from the newly-active server.yaml,
    builds a trial ConsolidationLoop with overrides (mode=train, paths →
    state/trial_adapter/, persist_graph=True on state/trial_graph/), and
    calls ``run_consolidation`` with ``mark_consolidated_callback=lambda _: None``.

    The ``mark_consolidated_callback`` no-op ensures that
    ``session_buffer.mark_consolidated`` is **never** called from the trial
    loop (spec L364 — "Transcript sweeper blocks archive+delete").  Pending
    sessions remain in the buffer after the trial cycle completes so that
    ``/migration/rollback`` (3b.3) can restore the full queue, and
    ``/migration/accept`` (3b.3) can call ``mark_consolidated`` itself.

    On completion, sets ``_state["migration"]["trial"]["gates"]`` to a dict
    with ``status`` drawn from
    ``{"pass", "no_new_sessions", "fail", "trial_exception"}`` and a
    ``details`` list of four :class:`~paramem.server.gates.GateResult` dicts.
    """
    from paramem.server.config import load_server_config

    try:
        from paramem.server.gpu_lock import gpu_lock_sync

        live_config_path = (
            Path(_state["config_path"])
            if _state.get("config_path")
            else Path("configs/server.yaml")
        )

        # Reload config from the newly-active candidate.
        trial_config = load_server_config(live_config_path)
        # Override: trial mode is always "train" regardless of what the candidate
        # config specifies (Resolved Decision 27, spec L239).
        trial_config.consolidation.mode = "train"

        # Determine trial adapter and graph paths from the marker.
        migration = _state.get("migration", {})
        trial_data = migration.get("trial") or {}
        # REQUIRED FIX 1 — read trial_adapter_dir directly from state; do NOT
        # resolve via find_live_slot.
        trial_adapter_dir_str = trial_data.get("trial_adapter_dir", "")
        trial_graph_dir_str = trial_data.get("trial_graph_dir", "")

        # CRITICAL Fix 1 (2026-04-23): do NOT override trial_config.paths.data here.
        # Previously this was set to trial_adapter_dir.parent.parent (= data/ha), causing
        # _save_registry / _save_key_metadata to resolve to the LIVE registry paths.
        # Registry path isolation is now handled entirely inside _build_trial_loop via
        # loop.trial_registry_path / loop.trial_key_metadata_path overrides.

        model = _state.get("model")
        tokenizer = _state.get("tokenizer")

        if model is None or tokenizer is None:
            logger.warning("trial consolidation: model not loaded (cloud-only mode?), skipping")
            await _update_trial_gates(
                {"status": "trial_exception", "exception": "model not loaded"}
            )
            return

        # --- Session buffer check ---
        session_buffer = _state.get("session_buffer")
        session_buffer_empty = session_buffer is None or session_buffer.pending_count == 0

        summary: dict | None = None
        exc_captured: Exception | None = None

        from paramem.server.gates import TrialLogCapture, evaluate_gates

        # Open TrialLogCapture BEFORE the consolidation executor so
        # WARNING/ERROR/CRITICAL records from extraction, training, adapter
        # reload, AND gate evaluation are all captured as a whole-run signal
        # (spec L398 — "New ERROR lines in trial log").  The `with` closes
        # after gates_payload["trial_log"] is populated but before
        # _update_trial_gates so the snapshot is frozen when the status
        # becomes observable.
        with TrialLogCapture() as _trial_log_capture:
            if not session_buffer_empty:
                loop = await asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda: _build_trial_loop(
                        model,
                        tokenizer,
                        trial_config,
                        Path(trial_adapter_dir_str) if trial_adapter_dir_str else None,
                        Path(trial_graph_dir_str) if trial_graph_dir_str else None,
                    ),
                )

                ha_context = _state.get("ha_context")
                speaker_store = _state.get("speaker_store")

                def _run():
                    with gpu_lock_sync():
                        from paramem.server.consolidation import run_consolidation as _run_consol

                        return _run_consol(
                            model,
                            tokenizer,
                            trial_config,
                            session_buffer,
                            loop=loop,
                            ha_context=ha_context,
                            speaker_store=speaker_store,
                            # Trial path: do NOT call mark_consolidated — pending sessions
                            # stay in the buffer so /migration/rollback (3b.3) can restore
                            # the full queue (spec L364).
                            mark_consolidated_callback=lambda _: None,
                        )

                try:
                    summary = await asyncio.get_running_loop().run_in_executor(None, _run)
                except Exception as _exc:  # noqa: BLE001
                    exc_captured = _exc

            # --- Gate evaluation ---
            # live_registry_path comes from the PRE-TRIAL config (REQUIRED FIX 1).
            live_config = _state.get("config")
            if live_config is None:
                raise RuntimeError(
                    "trial consolidation: _state['config'] is missing — "
                    "cannot resolve live registry path for gate 4"
                )
            # Use the canonical property — config.paths.key_metadata resolves to
            # config.paths.data / "registry" / "key_metadata.json", matching the
            # path that the consolidation writer uses (Cleanup 1, 2026-04-22).
            live_registry_path: Path = live_config.paths.key_metadata
            trial_adapter_dir = (
                Path(trial_adapter_dir_str)
                if trial_adapter_dir_str
                else Path("state/trial_adapter")
            )

            # Use loop.model (the PeftModel wrapper) instead of the raw
            # _state["model"] (base MistralForCausalLM). The trial loop's
            # create_adapter calls rebind self.model = PeftModel(...) — that
            # wrapper holds the trial adapters in a form that PEFT's
            # set_adapter / probe_key can use. The raw base model's
            # peft_config is populated in-place but `_hf_peft_config_loaded`
            # is False, which causes set_adapter to raise "No adapter loaded".
            gate_model = (
                loop.model
                if not session_buffer_empty and "loop" in locals() and loop is not None
                else model
            )
            results = evaluate_gates(
                model=gate_model,
                tokenizer=tokenizer,
                trial_adapter_dir=trial_adapter_dir,
                live_registry_path=live_registry_path,
                session_buffer_empty=session_buffer_empty,
                consolidation_summary=summary,
                consolidation_exception=exc_captured,
            )

            overall_status = _rollup_gate_status(results, session_buffer_empty)
            completed_at = datetime.now(timezone.utc).isoformat()

            gates_payload: dict = {
                "status": overall_status,
                "completed_at": completed_at,
                "summary": ({k: v for k, v in summary.items() if k != "loop"} if summary else {}),
                "details": [r.to_dict() for r in results],
                # trial_log captured across the entire consolidation + gate run
                # (top-level, not nested in per-gate metrics).
                "trial_log": _trial_log_capture.metrics,
            }
            if exc_captured is not None:
                gates_payload["exception"] = str(exc_captured)  # backward-compat with 3b.3

        await _update_trial_gates(gates_payload)
        logger.info("trial consolidation complete: status=%s", overall_status)

    except Exception as exc:  # noqa: BLE001
        completed_at = datetime.now(timezone.utc).isoformat()
        await _update_trial_gates(
            {"status": "trial_exception", "exception": str(exc), "completed_at": completed_at}
        )
        logger.exception("trial consolidation failed: %s", exc)


def _rollup_gate_status(results: list, session_buffer_empty: bool) -> str:
    """Compute the overall trial status from a list of GateResult objects.

    Decision table (spec §L368–412 — overall status rollup):

    - Any ``"fail"`` → ``"fail"``
    - All 4 ``"skipped"`` → ``"no_new_sessions"``
    - Gates 1/2/3 ``"skipped"`` + gate 4 ``"pass"`` → ``"no_new_sessions"``
    - Gates 1/2/3 ``"skipped"`` + gate 4 ``"fail"`` → ``"fail"``
    - Any subset with gate 4 ``"skipped"`` (< 20 keys) and no fails:
      ``"pass"`` when any of gates 1–3 passed, else ``"no_new_sessions"``
    - All 4 ``"pass"`` → ``"pass"``

    Parameters
    ----------
    results:
        List of four :class:`~paramem.server.gates.GateResult` objects in
        gate order (1, 2, 3, 4).
    session_buffer_empty:
        Passed through for logging context; not used in the rollup logic
        (gate statuses already encode the buffer-empty information).

    Returns
    -------
    str
        One of ``"pass"``, ``"no_new_sessions"``, or ``"fail"``.
    """
    statuses = [r.status for r in results]

    if "fail" in statuses:
        return "fail"

    # All skipped → no new sessions.
    if all(s == "skipped" for s in statuses):
        return "no_new_sessions"

    # Gates 1/2/3 all skipped but gate 4 is not → derive from gate 4.
    early_statuses = statuses[:3]
    gate4_status = statuses[3] if len(statuses) == 4 else "skipped"

    if all(s == "skipped" for s in early_statuses):
        if gate4_status == "pass":
            return "no_new_sessions"
        if gate4_status == "fail":
            return "fail"
        # gate 4 also skipped (<20 keys) — all 4 skipped covered above.
        return "no_new_sessions"

    # Gate 4 skipped (registry < 20 keys) — use gate 1–3 results.
    if gate4_status == "skipped":
        if "pass" in early_statuses:
            return "pass"
        # Only skipped among 1–3 (no pass, no fail).
        return "no_new_sessions"

    # Mix of pass and skipped in gates 1–3 with gate 4 pass.
    return "pass"


def _build_trial_loop(model, tokenizer, trial_config, trial_adapter_dir, trial_graph_dir):
    """Build a ConsolidationLoop for the trial, overriding output paths.

    CRITICAL Fix 1 (2026-04-23) — registry isolation:
    ``loop.trial_registry_path`` and ``loop.trial_key_metadata_path`` are set
    to paths inside a ``trial_registry/`` sibling of ``trial_adapter/`` so that
    ``_save_registry`` / ``_save_key_metadata`` in consolidation.py write to
    the trial-isolated directory instead of the live ``data/ha/registry.json``
    / ``data/ha/registry/key_metadata.json``.

    The previous pattern (``trial_config.paths.data = trial_adapter_dir.parent.parent``)
    is removed: it pointed ``paths.data`` back to ``data/ha`` and caused both
    registry writers to resolve to the LIVE paths.  The adapter output path is
    now set via ``loop.output_dir`` only, leaving ``trial_config.paths.data``
    alone so config-derived paths (sessions, debug, prompts) remain valid.
    """
    from paramem.server.consolidation import create_consolidation_loop

    loop = create_consolidation_loop(model, tokenizer, trial_config)

    if trial_adapter_dir is not None:
        loop.output_dir = trial_adapter_dir
        trial_adapter_dir.mkdir(parents=True, exist_ok=True)

        # CRITICAL Fix 1: redirect registry writes to a trial-isolated directory
        # (sibling of trial_adapter/, e.g. data/ha/state/trial_registry/).
        # _save_registry / _save_key_metadata check these attributes and use them
        # instead of config.registry_path / config.key_metadata_path.
        trial_registry_dir = trial_adapter_dir.parent / "trial_registry"
        loop.trial_registry_path = trial_registry_dir / "registry.json"
        loop.trial_key_metadata_path = trial_registry_dir / "registry" / "key_metadata.json"

    if trial_graph_dir is not None:
        loop.persist_graph = True
        loop.graph_path = trial_graph_dir / "cumulative_graph.json"
        trial_graph_dir.mkdir(parents=True, exist_ok=True)

    return loop


async def _update_trial_gates(gates: dict) -> None:
    """Update ``_state["migration"]["trial"]["gates"]`` under ``migration_lock``.

    Fix 5 (2026-04-23): the trial coroutine ``_run_trial_consolidation`` has
    ``await`` points (run_in_executor for both build_trial_loop and the
    consolidation executor) before this function runs, so a concurrent
    ``/migration/cancel`` may execute and clear ``_state["migration"]["trial"]``
    between the trial coroutine starting and this update. ``/migration/cancel``
    holds ``migration_lock`` while clearing trial state, so this writer must
    hold the same lock to observe a consistent view.

    Without the lock, the coroutine could observe a half-cleared state — e.g.
    ``state == "LIVE"`` but ``trial`` still set transiently — and write gates
    into a trial dict that is about to be (or just was) detached from
    ``_state["migration"]``.
    """
    lock: asyncio.Lock = _state.get("migration_lock") or asyncio.Lock()
    async with lock:
        migration = _state.get("migration")
        if migration is None:
            return
        trial = migration.get("trial")
        if trial is None:
            return
        trial["gates"] = gates


# Accept-eligible gate statuses (set membership for forward-compat — Decision 24).
# Accept-eligible values are "pass" and "no_new_sessions".  Gate evaluation
# also emits "fail" and "trial_exception".  Cluster-variance warnings from
# gate 4 live in `gates["details"][3]["metrics"]["warnings"]`, not as a new
# top-level status.
_ACCEPT_ELIGIBLE_STATUSES: frozenset[str] = frozenset({"pass", "no_new_sessions"})


@app.get("/migration/status", response_model=MigrationStatusResponse)
async def migration_status():
    """Return the current migration state and server metadata.

    Never raises — returns LIVE defaults when no preview has been requested.

    Populates ``comparison_report`` when the server is in TRIAL state, gates
    have completed with an accept-eligible status (``pass`` or
    ``no_new_sessions``), and ``completed_at`` is set.  ``None`` otherwise.
    """
    from paramem.server.migration import initial_migration_state
    from paramem.server.migration_report import build_comparison_report

    migration = _state.get("migration") or initial_migration_state()
    ms = migration.get("state", "LIVE")

    trial = migration.get("trial") or {}
    gates = trial.get("gates") or {}

    # Populate comparison_report when TRIAL + accept-eligible + completed.
    # _ACCEPT_ELIGIBLE_STATUSES contains {"pass", "no_new_sessions"}.
    # Cluster-variance warnings live in gate details, not as a separate
    # top-level status (Decision 24).
    comparison_report: dict | None = None
    if (
        ms == "TRIAL"
        and gates.get("status") in _ACCEPT_ELIGIBLE_STATUSES
        and gates.get("completed_at")
    ):
        # Resolve graph paths from state.
        # Pre-trial graph: prefer in-memory loop's merger graph (production
        # runs with persist_graph=False, so no file exists); fall back to a
        # config-derived path for completeness.
        pre_trial_graph = None
        pre_trial_graph_path: Path | None = None
        _loop_obj = _state.get("consolidation_loop")
        if _loop_obj is not None:
            _merger = getattr(_loop_obj, "merger", None)
            if _merger is not None:
                pre_trial_graph = getattr(_merger, "graph", None)
            _gpath = getattr(_loop_obj, "graph_path", None)
            if _gpath is not None:
                pre_trial_graph_path = Path(_gpath)
        if pre_trial_graph_path is None:
            _live_cfg = _state.get("config")
            if _live_cfg is not None:
                _adata = getattr(getattr(_live_cfg, "paths", None), "data", None)
                if _adata is not None:
                    pre_trial_graph_path = Path(_adata) / "ha" / "state" / "cumulative_graph.json"

        # Trial graph: from the trial marker (always set during TRIAL).
        trial_graph_dir_str = trial.get("trial_graph_dir") or ""
        trial_graph_path: Path | None = (
            Path(trial_graph_dir_str) / "cumulative_graph.json" if trial_graph_dir_str else None
        )

        comparison_report = build_comparison_report(
            gates=gates,
            pre_trial_graph_path=pre_trial_graph_path,
            trial_graph_path=trial_graph_path,
            pre_trial_graph=pre_trial_graph,
        )

    return MigrationStatusResponse(
        state=ms,
        candidate_path=migration.get("candidate_path") or None,
        candidate_hash=migration.get("candidate_hash") or None,
        staged_at=migration.get("staged_at") or None,
        simulate_mode_override=bool(migration.get("simulate_mode_override", False)),
        consolidating=bool(_state.get("consolidating", False)),
        server_started_at=_state.get("server_started_at", ""),
        # Forward-compat TRIAL fields (3b.3 long-poll).
        trial_started_at=trial.get("started_at") or None,
        pre_trial_config_sha256=trial.get("pre_trial_config_sha256") or None,
        candidate_config_sha256=trial.get("candidate_config_sha256") or None,
        backup_paths=trial.get("backup_paths") or None,
        trial_adapter_dir=trial.get("trial_adapter_dir") or None,
        trial_graph_dir=trial.get("trial_graph_dir") or None,
        gates=gates or None,
        recovery_required=list(migration.get("recovery_required") or []),
        comparison_report=comparison_report,
    )


@app.get("/migration/diff", response_model=MigrationDiffResponse)
async def migration_diff():
    """Return the diff for the currently-staged candidate.

    Same payload shape as ``/migration/preview``.  Valid only when STAGING.

    Errors
    ------
    409 ``not_staging``
        The server is not currently in STAGING state.
    """
    from fastapi import HTTPException

    from paramem.server.migration import initial_migration_state, render_preview_response

    migration = _state.get("migration") or initial_migration_state()
    current_state = migration.get("state", "LIVE")

    if current_state != "STAGING":
        raise HTTPException(
            status_code=409,
            detail={
                "error": "not_staging",
                "message": "No candidate is staged; POST /migration/preview first.",
            },
        )

    payload = render_preview_response(migration, pre_flight_fail=None)
    return MigrationDiffResponse(**payload)


_RESTART_HINT: str = "systemctl --user restart paramem-server"


@app.post("/migration/accept", response_model=AcceptResponse)
async def migration_accept():
    """Promote trial config B to live, archive the trial adapter, and clear trial state.

    Only valid when the server is in TRIAL state and gates have finished with an
    accept-eligible status (``pass`` or ``no_new_sessions``).

    5-step atomic ordering (spec §L353–359, IMPROVEMENT 7 — marker cleared
    before adapter/graph move):

    1. Re-verify preconditions inside lock (state, gates).
    2. Build rotation slot for trial adapter archive.
    3. **Clear trial marker** (BEFORE adapter/graph move — IMPROVEMENT 7).
    4. Move trial adapter + graph into the rotation slot.
    5. Refresh drift state (REQUIRED FIX 3) and set restart banner.

    Errors
    ------
    404 ``not_found``
        No trial is active (``migration.state == "LIVE"``).
    409 ``not_trial``
        Server is in STAGING, not TRIAL.
    409 ``gates_not_finished``
        Trial gates have not finished (status pending/running or no completed_at).
    409 ``gates_failed``
        Trial gates failed — only rollback is valid.
    409 ``migration_in_progress``
        Lock already held by a concurrent operation.
    500 ``trial_archive_failed``
        Could not create the rotation slot for the trial adapter.
    """
    from fastapi import HTTPException

    from paramem.server.drift import ConfigDriftState, compute_config_hash
    from paramem.server.migration import initial_migration_state

    # --- Pre-checks outside the lock (fast 4xx path) ---
    migration = _state.get("migration") or initial_migration_state()
    current_state = migration.get("state", "LIVE")

    if current_state == "LIVE":
        raise HTTPException(
            status_code=404,
            detail={"error": "not_found", "message": "No trial is active."},
        )

    if current_state == "STAGING":
        raise HTTPException(
            status_code=409,
            detail={
                "error": "not_trial",
                "message": (
                    "Cannot accept — server is in STAGING, not TRIAL. "
                    "Run POST /migration/confirm first."
                ),
            },
        )

    trial = migration.get("trial") or {}
    gates = trial.get("gates") or {}
    gates_status = gates.get("status", "")

    if not gates_status or gates_status in {"pending", "running"} or not gates.get("completed_at"):
        raise HTTPException(
            status_code=409,
            detail={
                "error": "gates_not_finished",
                "message": "Trial gates have not finished.",
            },
        )

    if gates_status not in _ACCEPT_ELIGIBLE_STATUSES:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "gates_failed",
                "message": "Trial gates failed — only POST /migration/rollback is valid.",
            },
        )

    lock: asyncio.Lock = _state.get("migration_lock") or asyncio.Lock()
    if lock.locked():
        raise HTTPException(
            status_code=409,
            detail={
                "error": "migration_in_progress",
                "message": "Another migration operation in progress.",
            },
        )

    config = _state.get("config")
    live_config_path = (
        Path(_state["config_path"]) if _state.get("config_path") else Path("configs/server.yaml")
    )
    if config is not None:
        state_dir = (config.paths.data / "state").resolve()
        backups_root = (config.paths.data / "backups").resolve()
    else:
        state_dir = Path("data/ha/state").resolve()
        backups_root = Path("data/ha/backups").resolve()

    trial_adapters_dir = backups_root / "trial_adapters"

    async with lock:
        # --- Re-verify inside lock (state may have changed) ---
        migration = _state.get("migration") or initial_migration_state()
        current_state = migration.get("state", "LIVE")

        if current_state != "TRIAL":
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "not_trial",
                    "message": "Server left TRIAL state while acquiring lock.",
                },
            )

        trial = migration.get("trial") or {}
        gates = trial.get("gates") or {}
        gates_status = gates.get("status", "")

        if gates_status not in _ACCEPT_ELIGIBLE_STATUSES or not gates.get("completed_at"):
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "gates_not_finished",
                    "message": "Trial gates status changed while acquiring lock.",
                },
            )

        trial_adapter_dir_str = trial.get("trial_adapter_dir", "")
        trial_graph_dir_str = trial.get("trial_graph_dir", "")
        pre_trial_config_sha256 = trial.get("pre_trial_config_sha256", "")
        candidate_config_sha256 = trial.get("candidate_config_sha256", "")
        trial_started_at = trial.get("started_at", "")

        # --- Step 2: Build rotation slot ---
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
        slot_dir = trial_adapters_dir / ts
        pending_dir = trial_adapters_dir / ".pending" / ts
        archive_path = str(slot_dir.resolve())

        try:
            pending_dir.mkdir(parents=True, exist_ok=False)
            meta = {
                "schema_version": 1,
                "rotated_at": datetime.now(timezone.utc).isoformat(),
                "source": "accept",
                "pre_trial_config_sha256": pre_trial_config_sha256,
                "candidate_config_sha256": candidate_config_sha256,
                "gates_status": gates_status,
                "trial_started_at": trial_started_at,
            }
            (pending_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            os.rename(str(pending_dir), str(slot_dir))
            # fsync parent for rename durability
            _dir_fd = os.open(str(trial_adapters_dir), os.O_RDONLY)
            try:
                os.fsync(_dir_fd)
            except OSError:
                pass
            finally:
                os.close(_dir_fd)
        except Exception as exc:
            # Clean up pending dir on failure
            try:
                shutil.rmtree(pending_dir, ignore_errors=True)
            except OSError:
                pass
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "trial_archive_failed",
                    "message": f"Failed to create trial adapter archive slot: {exc}",
                },
            ) from exc

        # --- Step 3: Clear trial marker BEFORE adapter/graph move (IMPROVEMENT 7) ---
        # Rationale: if marker-clear fails, nothing else has mutated yet.
        # If rotation below fails after marker-clear, state/trial_adapter/ is still
        # intact and state/trial.json is gone → startup recovery sees no marker +
        # B live → clean LIVE, no stale marker pointing at an already-rotated slot.
        try:
            clear_trial_marker(state_dir)
        except OSError as exc:
            logger.error("accept: failed to clear trial marker: %s", exc)
            # Clean up the slot we created
            try:
                shutil.rmtree(slot_dir, ignore_errors=True)
            except OSError:
                pass
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "marker_clear_failed",
                    "message": f"Failed to clear trial marker: {exc}",
                },
            ) from exc

        # --- Step 4: Move trial adapter + graph into the rotation slot ---
        # Non-fatal: config + marker are already coherent. Rotation is cosmetic.
        rotation_incomplete = False
        for src_str, dest_name in [
            (trial_adapter_dir_str, "adapter"),
            (trial_graph_dir_str, "graph"),
        ]:
            if not src_str:
                continue
            src = Path(src_str)
            if src.exists():
                dest = slot_dir / dest_name
                try:
                    shutil.move(str(src), str(dest))
                except Exception as mv_exc:
                    logger.error(
                        "accept: failed to move %s to archive slot: %s — "
                        "ARCHIVE INCOMPLETE — trial artifact remains at %s, archive manually",
                        src,
                        mv_exc,
                        src,
                    )
                    rotation_incomplete = True
                    archive_path = src_str  # degraded: point at still-in-place location

        # --- Step 5: Refresh drift state + set restart banner (REQUIRED FIX 3) ---
        # Replace the full ConfigDriftState dict, not just loaded_hash.
        if _state.get("config_path") and live_config_path.exists():
            try:
                new_hash = compute_config_hash(live_config_path)
                _state["config_drift"] = ConfigDriftState(
                    detected=False,
                    loaded_hash=new_hash,
                    disk_hash=new_hash,
                    last_checked_at=datetime.now(timezone.utc).isoformat(),
                )
            except OSError as exc:
                logger.warning("accept: could not compute new config hash: %s", exc)

        restart_banner = (
            "Migration: RESTART REQUIRED — new configuration takes effect on server restart"
        )
        if rotation_incomplete:
            restart_banner = (
                "Migration: RESTART REQUIRED — new configuration takes effect on server restart; "
                "ARCHIVE INCOMPLETE — trial adapter not fully rotated, archive manually"
            )

        # Reset migration state to LIVE, preserving recovery_required.
        prior_recovery = list(migration.get("recovery_required") or [])
        _state["migration"] = initial_migration_state()
        _state["migration"]["recovery_required"] = prior_recovery + [restart_banner]

        return AcceptResponse(
            state="LIVE",
            trial_adapter_archive_path=archive_path,
            restart_required=True,
            restart_hint=_RESTART_HINT,
            pre_migration_backup_retained=True,
        )


@app.post("/migration/rollback")
async def migration_rollback():
    """Restore config A from backup, archive trial adapter, clear trial state.

    Valid from TRIAL at any time (no gate-status check — spec §L208).

    8-step atomic ordering (IMPROVEMENT 8 — marker cleared before rotation):

    1. Re-verify inside lock (state=TRIAL).
    2. Snapshot B into rollback_pre_mortem backup.
    3. Resolve A config artifact from marker.
    4. Atomic rename A artifact → live config path.
    5. **Clear trial marker** (BEFORE rotation — IMPROVEMENT 8).
    6. Rotate trial adapter + graph (non-fatal; triggers 207 on failure).
    7. Append restart banner.
    8. Reset migration state to LIVE.

    Returns HTTP 200 on full success or HTTP 207 when rotation fails (config
    restored, marker cleared, but trial adapter archive incomplete).

    Errors
    ------
    404 ``not_found``
        No trial is active.
    409 ``not_trial``
        Server is in STAGING.
    409 ``migration_in_progress``
        Lock already held.
    500 ``rollback_backup_failed``
        Step 2 snapshot failed; state=TRIAL preserved.
    500 ``rollback_precondition_failed``
        A config artifact missing; pre-mortem backup deleted.
    500 ``config_restore_failed``
        Step 4 rename failed; pre-mortem backup deleted.
    500 ``marker_clear_failed``
        Step 5 marker clear failed; internal inconsistency.
    """
    from fastapi import HTTPException

    from paramem.backup.types import ArtifactKind
    from paramem.server.migration import initial_migration_state
    from paramem.server.trial_state import read_trial_marker

    # --- Pre-checks outside lock ---
    migration = _state.get("migration") or initial_migration_state()
    current_state = migration.get("state", "LIVE")

    if current_state == "LIVE":
        raise HTTPException(
            status_code=404,
            detail={"error": "not_found", "message": "No trial is active."},
        )

    if current_state == "STAGING":
        raise HTTPException(
            status_code=409,
            detail={
                "error": "not_trial",
                "message": (
                    "Cannot rollback — server is in STAGING. Run POST /migration/cancel instead."
                ),
            },
        )

    lock: asyncio.Lock = _state.get("migration_lock") or asyncio.Lock()
    if lock.locked():
        raise HTTPException(
            status_code=409,
            detail={
                "error": "migration_in_progress",
                "message": "Another migration operation in progress.",
            },
        )

    config = _state.get("config")
    live_config_path = (
        Path(_state["config_path"]) if _state.get("config_path") else Path("configs/server.yaml")
    )
    if config is not None:
        state_dir = (config.paths.data / "state").resolve()
        backups_root = (config.paths.data / "backups").resolve()
    else:
        state_dir = Path("data/ha/state").resolve()
        backups_root = Path("data/ha/backups").resolve()

    trial_adapters_dir = backups_root / "trial_adapters"

    async with lock:
        # --- Step 1: Re-verify inside lock ---
        migration = _state.get("migration") or initial_migration_state()
        current_state = migration.get("state", "LIVE")

        if current_state != "TRIAL":
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "not_trial",
                    "message": "Server left TRIAL state while acquiring lock.",
                },
            )

        trial = migration.get("trial") or {}
        trial_adapter_dir_str = trial.get("trial_adapter_dir", "")
        trial_graph_dir_str = trial.get("trial_graph_dir", "")
        pre_trial_config_sha256 = trial.get("pre_trial_config_sha256", "")
        trial_started_at = trial.get("started_at", "")

        # --- Step 2: Snapshot B into rollback_pre_mortem backup ---
        pre_mortem_slot: Path | None = None
        try:
            b_bytes = live_config_path.read_bytes() if live_config_path.exists() else b""
            pre_mortem_slot = backup_write(
                ArtifactKind.CONFIG,
                b_bytes,
                meta_fields={
                    "tier": "rollback_pre_mortem",
                    "pre_trial_hash": pre_trial_config_sha256,
                },
                base_dir=backups_root / "config",
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "rollback_backup_failed",
                    "message": f"Failed to snapshot B config before rollback: {exc}",
                },
            ) from exc

        rollback_pre_mortem_path = str(pre_mortem_slot.resolve())

        # --- Step 3: Resolve A backup artifact ---
        # Read marker from disk to get config_artifact_filename (REQUIRED FIX 1).
        marker = read_trial_marker(state_dir)
        config_artifact_filename = ""
        config_backup_slot_str = trial.get("backup_paths", {}).get("config", "") if trial else ""

        if marker is not None:
            config_artifact_filename = marker.config_artifact_filename
            if marker.backup_paths.get("config"):
                config_backup_slot_str = marker.backup_paths["config"]

        if not config_artifact_filename:
            # Defensive: marker missing or written before 3b.3.
            try:
                shutil.rmtree(pre_mortem_slot, ignore_errors=True)
            except OSError:
                pass
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "rollback_precondition_failed",
                    "message": (
                        "config_artifact_filename is empty in trial marker — "
                        "cannot locate A config backup file. "
                        "Marker may have been written by an older server version."
                    ),
                },
            )

        if not config_backup_slot_str:
            try:
                shutil.rmtree(pre_mortem_slot, ignore_errors=True)
            except OSError:
                pass
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "rollback_precondition_failed",
                    "message": "config backup slot path missing from trial marker.",
                },
            )

        a_yaml_file = Path(config_backup_slot_str) / config_artifact_filename
        if not a_yaml_file.exists():
            try:
                shutil.rmtree(pre_mortem_slot, ignore_errors=True)
            except OSError:
                pass
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "rollback_precondition_failed",
                    "message": (
                        f"A config artifact not found at {a_yaml_file}. "
                        "Backup may have been manually deleted."
                    ),
                },
            )

        # --- Step 4: Decrypt A artifact and write to live config path ---
        # Mirror backup_restore's decrypt-first ordering: read the slot via
        # backup.read() which dispatches on envelope magic (age or plaintext)
        # and returns plaintext — then write plaintext to a .pending temp,
        # fsync, and atomic rename.  A raw os.rename of the artifact would
        # write ciphertext bytes into configs/server.yaml, causing
        # yaml.safe_load to fail on the next server start (B6 — 2026-04-22).
        # RuntimeError = daily identity not loaded; other exceptions surface
        # as a generic decrypt failure with the exception message included.
        try:
            from paramem.backup.backup import read as backup_read

            a_slot_dir = Path(config_backup_slot_str)
            plaintext_bytes, _a_meta = backup_read(a_slot_dir)
        except RuntimeError as exc:
            try:
                shutil.rmtree(pre_mortem_slot, ignore_errors=True)
            except OSError:
                pass
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "decrypt_no_key",
                    "message": (
                        "A-config backup is age-encrypted but the daily identity "
                        f"is not loaded: {exc}"
                    ),
                },
            ) from exc
        except Exception as exc:  # noqa: BLE001
            try:
                shutil.rmtree(pre_mortem_slot, ignore_errors=True)
            except OSError:
                pass
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "config_restore_failed",
                    "message": f"Failed to read/decrypt A config artifact: {exc}",
                },
            ) from exc

        # Fix 3 (2026-04-23): initialize before the try block so the except
        # cleanup guard never raises UnboundLocalError if the assignment itself
        # (e.g. datetime.now) raises before pending_restore is set.
        pending_restore: Path | None = None
        try:
            _ts_suffix = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
            pending_restore = live_config_path.parent / f".pending-rollback-{_ts_suffix}.yaml"
            # Fix 2 (2026-04-23): write the temp file at 0o600 to prevent a
            # plaintext-exposure window under the default umask (0644).
            # os.O_EXCL ensures the file is created atomically; write via fd.
            _fd = os.open(str(pending_restore), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                os.write(_fd, plaintext_bytes)
                os.fsync(_fd)
            finally:
                os.close(_fd)
            os.rename(str(pending_restore), str(live_config_path))
            _dir_fd = os.open(str(live_config_path.parent), os.O_RDONLY)
            try:
                os.fsync(_dir_fd)
            except OSError:
                pass
            finally:
                os.close(_dir_fd)
        except Exception as exc:
            # Clean up pending temp if it was created (Fix 3 guard).
            if pending_restore is not None:
                try:
                    pending_restore.unlink(missing_ok=True)
                except OSError:
                    pass
            try:
                shutil.rmtree(pre_mortem_slot, ignore_errors=True)
            except OSError:
                pass
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "config_restore_failed",
                    "message": f"Failed to restore A config: {exc}",
                },
            ) from exc

        # --- Step 5: Clear trial marker BEFORE rotation (IMPROVEMENT 8) ---
        # Rationale: if rotation fails, no stale marker misdirects recovery.
        # Config is already restored; marker-clear failure is a consistency issue.
        try:
            clear_trial_marker(state_dir)
        except OSError as exc:
            logger.error("rollback: failed to clear trial marker: %s", exc)
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "marker_clear_failed",
                    "message": f"Config restored but trial marker could not be cleared: {exc}",
                },
            ) from exc

        # --- Step 6: Rotate trial adapter + graph (non-fatal → 207) ---
        rotation_failed = False
        archive_path = ""
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
        slot_dir = trial_adapters_dir / ts
        pending_dir = trial_adapters_dir / ".pending" / ts

        try:
            pending_dir.mkdir(parents=True, exist_ok=False)
            meta = {
                "schema_version": 1,
                "rotated_at": datetime.now(timezone.utc).isoformat(),
                "source": "rollback",
                "pre_trial_config_sha256": pre_trial_config_sha256,
                "trial_started_at": trial_started_at,
            }
            (pending_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            os.rename(str(pending_dir), str(slot_dir))
            _dir_fd = os.open(str(trial_adapters_dir), os.O_RDONLY)
            try:
                os.fsync(_dir_fd)
            except OSError:
                pass
            finally:
                os.close(_dir_fd)
            archive_path = str(slot_dir.resolve())

            # Move adapter + graph into the slot.
            for src_str, dest_name in [
                (trial_adapter_dir_str, "adapter"),
                (trial_graph_dir_str, "graph"),
            ]:
                if not src_str:
                    continue
                src = Path(src_str)
                if src.exists():
                    shutil.move(str(src), str(slot_dir / dest_name))

        except Exception as rot_exc:
            logger.error(
                "rollback: trial adapter rotation failed (non-fatal): %s — "
                "config restored, marker cleared, adapter remains at %s",
                rot_exc,
                trial_adapter_dir_str,
            )
            rotation_failed = True
            archive_path = trial_adapter_dir_str
            # Clean up partial slot
            try:
                shutil.rmtree(pending_dir, ignore_errors=True)
                shutil.rmtree(slot_dir, ignore_errors=True)
            except OSError:
                pass

        # --- Step 7: Append restart banner ---
        restart_banner = (
            "Migration: RESTART REQUIRED — rollback renamed configs/server.yaml; "
            "restart to clear recovery banner"
        )
        # NOTE: do NOT refresh config_drift — in-memory config still matches A;
        # the drift loop stays coherent (spec: rollback handler does not refresh drift).

        # --- Step 8: Reset migration state to LIVE ---
        prior_recovery = list(migration.get("recovery_required") or [])
        _state["migration"] = initial_migration_state()
        _state["migration"]["recovery_required"] = prior_recovery + [restart_banner]

        if rotation_failed:
            # HTTP 207 Multi-Status: primary action succeeded, rotation failed.
            body = {
                "state": "LIVE",
                "trial_adapter_archive_path": archive_path,
                "rollback_pre_mortem_backup_path": rollback_pre_mortem_path,
                "restart_required": True,
                "restart_hint": _RESTART_HINT,
                "archive_warning": {
                    "path": archive_path,
                    "message": (
                        "Trial adapter rotation failed — adapter remains at the "
                        "path above. Archive manually or accept the data loss."
                    ),
                },
            }
            return Response(
                content=json.dumps(body),
                status_code=207,
                media_type="application/json",
            )

        return RollbackResponse(
            state="LIVE",
            trial_adapter_archive_path=archive_path,
            rollback_pre_mortem_backup_path=rollback_pre_mortem_path,
            restart_required=True,
            restart_hint=_RESTART_HINT,
        )


# ---------------------------------------------------------------------------
# Backup REST endpoints
# ---------------------------------------------------------------------------


class BackupListItem(BaseModel):
    """One row in the ``/backup/list`` response.

    Attributes
    ----------
    backup_id:
        Slot directory name (e.g. ``"20260421-04000012"``).
    kind:
        Artifact kind string (``"config"`` | ``"graph"`` | ``"registry"`` |
        ``"snapshot"`` | ``"resume"``).
    tier:
        Backup tier (``"daily"`` | ``"manual"`` | ``"pre_migration"`` | …).
    timestamp:
        ISO-8601 UTC timestamp derived from the slot directory name.
    size_bytes:
        Total file size of the slot on disk.
    label:
        Optional operator-supplied annotation; ``None`` when absent.
    path:
        Absolute path to the slot directory.
    """

    backup_id: str
    kind: str
    tier: str
    timestamp: str
    size_bytes: int
    label: str | None
    path: str


class BackupListResponse(BaseModel):
    """Response body for ``GET /backup/list``.

    Attributes
    ----------
    items:
        Backup records, newest-first.
    disk_used_bytes:
        Current total backup-store usage.
    disk_cap_bytes:
        Global cap (``max_total_disk_gb * 1024**3``).
    """

    items: list[BackupListItem]
    disk_used_bytes: int
    disk_cap_bytes: int


class BackupCreateRequest(BaseModel):
    """Request body for ``POST /backup/create``.

    Attributes
    ----------
    kinds:
        Artifact kinds to back up.  ``None`` or ``[]`` → default
        ``["config", "graph", "registry"]``.
    label:
        Optional annotation written into each slot sidecar.
    """

    kinds: list[str] | None = None
    label: str | None = None


class SkippedArtifact(BaseModel):
    """One skipped artifact entry in ``BackupCreateResponse``.

    Attributes
    ----------
    kind:
        Artifact kind that was skipped.
    reason:
        Human-readable reason (e.g. ``"registry empty (no keys yet)"``).
    """

    kind: str
    reason: str


class BackupCreateResponse(BaseModel):
    """Response body for ``POST /backup/create``.

    Attributes
    ----------
    success:
        ``True`` when at least one artifact was written; ``False`` on
        disk-pressure refusal or write error.
    tier:
        Always ``"manual"`` for operator-initiated backups.
    written_slots:
        Mapping of artifact name → absolute slot directory path.
    skipped_artifacts:
        Artifacts that were not written, with reasons.
    error:
        Short error description when ``success=False``; ``None`` otherwise.
    """

    success: bool
    tier: str
    written_slots: dict[str, str]
    skipped_artifacts: list[SkippedArtifact] = []
    error: str | None


class BackupRestoreRequest(BaseModel):
    """Request body for ``POST /backup/restore``.

    Attributes
    ----------
    backup_id:
        Slot directory name to restore (e.g. ``"20260421-04000012"``).
    """

    backup_id: str


class BackupRestoreResponse(BaseModel):
    """Response body for ``POST /backup/restore``.

    Attributes
    ----------
    restored:
        Mapping of artifact kind → live path that was overwritten.
    backed_up_pre_restore:
        Mapping of kind → safety backup slot path taken before restore.
    restart_required:
        Always ``True`` — the server must be restarted to load the restored config.
    restart_hint:
        Human-readable restart command.
    """

    restored: dict[str, str]
    backed_up_pre_restore: dict[str, str]
    restart_required: bool = True
    restart_hint: str


class BackupPruneRequest(BaseModel):
    """Request body for ``POST /backup/prune``.

    Attributes
    ----------
    dry_run:
        When ``True``, populate ``would_delete_next`` but do not delete.
    """

    dry_run: bool = False


class BackupPruneResponse(BaseModel):
    """Response body for ``POST /backup/prune``.

    Attributes
    ----------
    deleted:
        Slot directories removed (stringified paths).
    preserved_immune:
        Slots saved by live-TRIAL immunity.
    preserved_pre_migration_window:
        Slots saved by the 30-day pre-migration window immunity (rule 4).
    would_delete_next:
        Dry-run preview: slots that would be deleted on the next non-dry-run call.
    disk_usage_before:
        Disk usage snapshot before any deletions.
    disk_usage_after:
        Disk usage snapshot after deletions (equals before in dry-run).
    invalid_slots:
        ``[[path, reason]]`` pairs for slots with unreadable sidecars.
    dry_run:
        Echoes the request flag.
    """

    deleted: list[str]
    preserved_immune: list[str]
    preserved_pre_migration_window: list[str]
    would_delete_next: list[str]
    disk_usage_before: dict
    disk_usage_after: dict
    invalid_slots: list[list[str]]
    dry_run: bool


@app.get("/backup/list", response_model=BackupListResponse)
async def backup_list(kind: str | None = None):
    """Enumerate backups across all kinds, newest-first.

    Query parameter ``kind`` filters by artifact kind (``"config"`` |
    ``"graph"`` | ``"registry"`` | ``"snapshot"`` | ``"resume"``).
    Unknown values return 400 ``kind_invalid``.

    Reads via ``enumerate_backups(backups_root, kind=...)``.  Size is taken
    from the sum of file sizes in each slot directory.

    Errors
    ------
    400 ``kind_invalid``
        When ``kind`` is not in ``ArtifactKind`` enum values.
    """
    from fastapi import HTTPException

    from paramem.backup.enumerate import enumerate_backups
    from paramem.backup.retention import compute_disk_usage

    config = _state.get("config")

    # Resolve backups_root.
    if config is not None:
        try:
            backups_root = (config.paths.data / "backups").resolve()
        except (AttributeError, TypeError):
            backups_root = Path("data/ha/backups").resolve()
    else:
        backups_root = Path("data/ha/backups").resolve()

    # Validate and coerce kind query parameter.
    kind_enum = None
    if kind is not None:
        try:
            kind_enum = ArtifactKind(kind)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "kind_invalid",
                    "message": (
                        f"kind must be one of {[k.value for k in ArtifactKind]}; got {kind!r}"
                    ),
                },
            )

    records = enumerate_backups(backups_root, kind=kind_enum)

    # Build BackupListItem list.
    from paramem.backup.retention import _slot_size_bytes

    items: list[BackupListItem] = []
    for record in records:
        # Compute slot size.
        size = _slot_size_bytes(record.slot_dir)
        # Convert timestamp to ISO-8601 (YYYYMMDD-HHMMSSff → datetime str).
        ts = record.created_at.isoformat()
        items.append(
            BackupListItem(
                backup_id=record.slot_dir.name,
                kind=record.kind.value,
                tier=record.meta.tier,
                timestamp=ts,
                size_bytes=size,
                label=record.label,
                path=str(record.slot_dir),
            )
        )

    # Disk usage (TTL-cached).
    disk_used_bytes = 0
    disk_cap_bytes = 0
    if config is not None:
        try:
            backups_cfg = config.security.backups
            usage = compute_disk_usage(backups_root, backups_cfg)
            disk_used_bytes = usage.total_bytes
            disk_cap_bytes = usage.cap_bytes
        except Exception:
            pass

    return BackupListResponse(
        items=items,
        disk_used_bytes=disk_used_bytes,
        disk_cap_bytes=disk_cap_bytes,
    )


@app.post("/backup/create", response_model=BackupCreateResponse)
async def backup_create(req: BackupCreateRequest):
    """Take an immediate manual backup of the requested artifacts.

    Delegates to ``run_scheduled_backup`` with ``tier="manual"`` and a
    per-call shallow-cloned config with ``.artifacts`` replaced by the
    request's ``kinds`` list (defaults to ``["config", "graph", "registry"]``).

    Persists the result via ``update_backup_state`` so the next ``/status``
    reflects the freshly-updated ``last_success_at``.

    Errors
    ------
    400 ``kind_invalid``
        When any entry in ``kinds`` is not ``"config"``, ``"graph"``, or
        ``"registry"``.
    """
    import dataclasses

    from fastapi import HTTPException

    from paramem.backup.runner import run_scheduled_backup
    from paramem.backup.state import update_backup_state

    _VALID_KINDS = {"config", "graph", "registry"}

    # Validate kinds.
    kinds = req.kinds
    if not kinds:
        kinds = ["config", "graph", "registry"]

    for k in kinds:
        if k not in _VALID_KINDS:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "kind_invalid",
                    "message": (f"kind must be one of {sorted(_VALID_KINDS)}; got {k!r}"),
                },
            )

    config = _state.get("config")
    if config is None:
        # Cloud-only or uninitialized — still attempt with a default config fallback.
        from paramem.server.config import PathsConfig, SecurityConfig, ServerConfig

        _fallback_config = ServerConfig.__new__(ServerConfig)
        _fallback_config.paths = PathsConfig()
        _fallback_config.security = SecurityConfig()
        config = _fallback_config

    # Build a per-call copy of the backups config with the requested artifacts.
    # dataclasses.replace performs a shallow copy; retention is shared by reference
    # (safe because run_scheduled_backup only reads it).
    per_call_backups_cfg = dataclasses.replace(config.security.backups, artifacts=kinds)

    # Build a temporary server_config with the replaced backups config.
    import dataclasses as _dc

    per_call_security = _dc.replace(config.security, backups=per_call_backups_cfg)

    # We cannot use dataclasses.replace on ServerConfig directly because it may
    # not be a plain dataclass in all test shims. Build a shallow wrapper instead.
    class _ConfigProxy:
        """Thin proxy that swaps in the per-call security config."""

        def __init__(self, base, security):
            self._base = base
            self.security = security

        def __getattr__(self, name):
            return getattr(self._base, name)

    proxy_config = _ConfigProxy(config, per_call_security)

    state_dir = (config.paths.data / "state").resolve()
    backups_root = (config.paths.data / "backups").resolve()
    live_config_path = (
        Path(_state["config_path"]) if _state.get("config_path") else Path("configs/server.yaml")
    )
    loop = _state.get("consolidation_loop")

    result = run_scheduled_backup(
        server_config=proxy_config,
        loop=loop,
        state_dir=state_dir,
        backups_root=backups_root,
        live_config_path=live_config_path,
        tier="manual",
        label=req.label,
    )

    # Persist state so /status shows updated last_success_at.
    try:
        update_backup_state(state_dir, result)
    except Exception as exc:
        logger.warning("backup_create: update_backup_state failed: %s", exc)

    skipped = [SkippedArtifact(kind=k, reason=r) for k, r in result.skipped_artifacts]

    return BackupCreateResponse(
        success=result.success,
        tier=result.tier,
        written_slots=result.written_slots,
        skipped_artifacts=skipped,
        error=result.error,
    )


@app.post("/backup/restore", response_model=BackupRestoreResponse)
async def backup_restore(req: BackupRestoreRequest):
    """Restore a **config** backup atop the live server.yaml.

    Only ``kind="config"`` is currently supported (restore-kind-restriction).
    Graph / registry restore requires an offline-coordinator path and is not
    yet implemented.

    Atomic restore sequence (decrypt-first, then safety backup, then rename):

    1. Verify preconditions (no TRIAL/STAGING, no active consolidation).
    2. Locate the slot by ``backup_id``.
    3. Assert ``kind == config`` (400 otherwise).
    4. Decrypt backup via ``backup.read(slot_dir)`` → plaintext bytes.
    5. Take a manual safety backup of the current live config.
    6. Write plaintext to ``live_config_path + ".restore-pending"``, fsync,
       ``os.rename`` to live path, fsync parent.
    7. Append recovery banner to ``_state["migration"]["recovery_required"]``.
    8. Return 200.

    Errors
    ------
    409 ``trial_active``
        Migration state is TRIAL.
    409 ``staging_active``
        Migration state is STAGING.
    409 ``consolidating``
        A consolidation run is in progress.
    404 ``not_found``
        No slot with the given ``backup_id`` exists.
    400 ``restore_kind_not_supported``
        The slot is not kind=config.
    500 ``decrypt_no_key``
        The backup is age-encrypted but the daily identity is not loaded.
    500 ``decrypt_invalid_token``
        Decryption failed (stale daily identity or corrupted backup).
    500 ``config_restore_failed``
        Atomic rename of the restore temp file failed.
    """
    from fastapi import HTTPException

    from paramem.backup.backup import read as backup_read
    from paramem.backup.backup import write as backup_write_fn
    from paramem.backup.enumerate import enumerate_backups
    from paramem.server.migration import initial_migration_state

    # --- Step 1: Precondition checks ---
    migration = _state.get("migration") or initial_migration_state()
    mig_state = migration.get("state", "LIVE")

    if mig_state == "TRIAL":
        raise HTTPException(
            status_code=409,
            detail={
                "error": "trial_active",
                "state": "TRIAL",
                "message": ("Cannot restore during TRIAL. Accept or rollback the migration first."),
            },
        )
    if mig_state == "STAGING":
        raise HTTPException(
            status_code=409,
            detail={
                "error": "staging_active",
                "state": "STAGING",
                "message": ("Cannot restore during STAGING. Cancel the migration first."),
            },
        )
    if _state.get("consolidating"):
        raise HTTPException(
            status_code=409,
            detail={
                "error": "consolidating",
                "message": "Consolidation is running; wait for completion before restoring.",
            },
        )

    config = _state.get("config")
    if config is not None:
        try:
            backups_root = (config.paths.data / "backups").resolve()
        except (AttributeError, TypeError):
            backups_root = Path("data/ha/backups").resolve()
    else:
        backups_root = Path("data/ha/backups").resolve()

    # --- Step 2: Locate slot ---
    all_records = enumerate_backups(backups_root, kind=None)
    target_record = None
    for record in all_records:
        if record.slot_dir.name == req.backup_id:
            target_record = record
            break

    if target_record is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "not_found",
                "message": f"No backup slot found with backup_id={req.backup_id!r}.",
            },
        )

    # --- Step 3: Kind gate (only config in 6b) ---
    if target_record.kind != ArtifactKind.CONFIG:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "restore_kind_not_supported",
                "message": (
                    f"Only kind='config' restore is supported in this release. "
                    f"Found kind={target_record.kind.value!r} at {req.backup_id}."
                ),
            },
        )

    # --- Step 4: Decrypt backup (BEFORE safety backup — order matters) ---
    # Age-encrypted backups have no stored fingerprint; a stale daily identity
    # surfaces here as a decrypt error (RuntimeError for "identity not loaded",
    # other exceptions for corruption / wrong recipient list).
    try:
        plaintext_bytes, _meta = backup_read(target_record.slot_dir)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "decrypt_no_key",
                "message": (f"backup is age-encrypted but the daily identity is not loaded: {exc}"),
            },
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail={
                "error": "decrypt_invalid_token",
                "message": (
                    "backup decryption failed — ciphertext corrupt or encrypted "
                    f"to a different recipient list (stale daily identity?): {exc}"
                ),
            },
        ) from exc

    live_config_path = (
        Path(_state["config_path"]) if _state.get("config_path") else Path("configs/server.yaml")
    )

    # --- Step 5: Safety backup of current live config ---
    safety_slot_path: str = ""
    safety_label = f"pre_restore_safety_{req.backup_id}"
    try:
        safety_slot = backup_write_fn(
            ArtifactKind.CONFIG,
            live_config_path.read_bytes() if live_config_path.exists() else b"",
            meta_fields={"tier": "manual", "label": safety_label},
            base_dir=backups_root / "config",
        )
        safety_slot_path = str(safety_slot)
    except Exception as exc:
        logger.error("backup_restore: safety backup failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "config_restore_failed",
                "message": f"Safety backup failed; restore aborted: {exc}",
            },
        ) from exc

    # --- Step 6: Atomic restore ---
    restore_pending = live_config_path.with_suffix(live_config_path.suffix + ".restore-pending")
    try:
        # Fix 2 (2026-04-23): write at 0o600 to prevent plaintext exposure under
        # the default umask (0644) during the rename window.
        _fd = os.open(str(restore_pending), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(_fd, plaintext_bytes)
            os.fsync(_fd)
        finally:
            os.close(_fd)
        os.rename(str(restore_pending), str(live_config_path))
        # fsync parent directory for rename durability.
        _dir_fd = os.open(str(live_config_path.parent), os.O_RDONLY)
        try:
            os.fsync(_dir_fd)
        except OSError as _fsync_exc:
            logger.warning("backup_restore: parent fsync failed: %s", _fsync_exc)
        finally:
            os.close(_dir_fd)
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "config_restore_failed",
                "message": (
                    f"Failed to atomically rename restore temp file: {exc}. "
                    f"Safety backup is at {safety_slot_path}."
                ),
            },
        ) from exc

    # --- Step 7: Append recovery banner ---
    if "migration" not in _state:
        _state["migration"] = initial_migration_state()
    if "recovery_required" not in _state["migration"]:
        _state["migration"]["recovery_required"] = []
    _state["migration"]["recovery_required"].append(
        f"Restored config from backup {req.backup_id} — restart to clear recovery banner."
    )

    return BackupRestoreResponse(
        restored={"config": str(live_config_path)},
        backed_up_pre_restore={"config": safety_slot_path},
        restart_required=True,
        restart_hint="systemctl --user restart paramem-server",
    )


@app.post("/backup/prune", response_model=BackupPruneResponse)
async def backup_prune(req: BackupPruneRequest):
    """Apply the 5-rule retention policy.

    Thin wrapper over ``retention.prune()``.  Serialises ``PruneResult``
    into the response — ``DiskUsage`` dicts include ``total_bytes``,
    ``by_tier``, ``cap_bytes``, and ``pct_of_cap``.

    Errors
    ------
    500 ``prune_failed``
        Unexpected error during pruning (e.g. permission denied).
    """
    from fastapi import HTTPException

    from paramem.backup.retention import prune

    config = _state.get("config")
    if config is not None:
        try:
            backups_root = (config.paths.data / "backups").resolve()
            state_dir = (config.paths.data / "state").resolve()
            backups_cfg = config.security.backups
        except (AttributeError, TypeError) as exc:
            raise HTTPException(
                status_code=500,
                detail={"error": "prune_failed", "message": f"Config error: {exc}"},
            ) from exc
    else:
        from paramem.server.config import ServerBackupsConfig

        backups_root = Path("data/ha/backups").resolve()
        state_dir = Path("data/ha/state").resolve()
        backups_cfg = ServerBackupsConfig()

    try:
        pr = prune(
            backups_root=backups_root,
            state_dir=state_dir,
            config=backups_cfg,
            dry_run=req.dry_run,
        )
    except Exception as exc:
        logger.error("backup_prune: prune() raised: %s", exc)
        raise HTTPException(
            status_code=500,
            detail={"error": "prune_failed", "message": str(exc)},
        ) from exc

    def _du_to_dict(du) -> dict:
        return {
            "total_bytes": du.total_bytes,
            "by_tier": du.by_tier,
            "cap_bytes": du.cap_bytes,
            "pct_of_cap": du.pct_of_cap,
        }

    return BackupPruneResponse(
        deleted=[str(p) for p in pr.deleted],
        preserved_immune=[str(p) for p in pr.preserved_immune],
        preserved_pre_migration_window=[str(p) for p in pr.preserved_pre_migration_window],
        would_delete_next=[str(p) for p in pr.would_delete_next],
        disk_usage_before=_du_to_dict(pr.disk_usage_before),
        disk_usage_after=_du_to_dict(pr.disk_usage_after),
        invalid_slots=[[str(p), r] for p, r in pr.invalid_slots],
        dry_run=req.dry_run,
    )


def _cloud_only_route(
    text: str,
    speaker: str | None,
    history: list[dict] | None,
    config,
    router=None,
    ha_client=None,
    sota_agent=None,
    language: str | None = None,
) -> ChatResult:
    """Route queries when the local model is unavailable (cloud-only mode).

    HA first (has tools for weather, time, devices), SOTA as fallback
    for reasoning. Mutual fallback: if one fails, try the other.
    """
    from paramem.server.sanitizer import sanitize_for_cloud

    # Cloud-only: no local model for PA probing.
    # HA first (has tools for weather, time, devices), SOTA as fallback.

    # Try HA conversation agent — it has tools and real-time data
    # Language passed via HA's native conversation API parameter
    if ha_client is not None:
        logger.debug("Cloud-only route: trying HA agent for: %s", text[:100])
        ha_languages = config.tools.ha.supported_languages
        response_text = ha_client.conversation_process(
            text,
            agent_id=config.ha_agent_id,
            language=language,
            supported_languages=ha_languages,
        )
        if response_text is not None:
            logger.info("Cloud-only route: HA agent responded")
            return ChatResult(text=response_text, escalated=True)
        logger.info("Cloud-only route: HA agent failed, trying SOTA")

    # HA failed or unavailable → try SOTA for reasoning
    if sota_agent is not None:
        sanitized, _ = sanitize_for_cloud(text, mode=config.sanitization.mode)
        if sanitized is not None:
            logger.info("Cloud-only route: escalating to SOTA")
            result = _escalate_to_sota(
                sanitized,
                sota_agent,
                config,
                speaker=speaker,
                history=history,
                language=language,
            )
            if result.text:
                logger.info("Cloud-only route: SOTA responded")
                return result

    logger.warning("Cloud-only route: all services failed")
    return ChatResult(
        text="I'm running in limited mode right now. Please try again later.",
        escalated=True,
    )


# --- Internal ---


def _run_consolidation_sync():
    """Run consolidation in a background thread.

    The consolidating flag is set by the caller before submitting to
    the executor, and cleared by _consolidation_done_callback after
    completion. Holds the GPU lock for the entire consolidation to
    prevent concurrent CUDA access from STT/TTS/inference.
    """
    from paramem.server.gpu_lock import gpu_lock_sync

    with gpu_lock_sync():
        _run_consolidation_inner()


def _run_consolidation_inner():
    """Consolidation logic — must be called with GPU lock held."""
    # Read HA home context for location validation
    ha_context = None
    ha_client = _state.get("ha_client")
    if ha_client is not None:
        ha_context = ha_client.get_home_context()

    result = run_consolidation(
        model=_state["model"],
        tokenizer=_state["tokenizer"],
        config=_state["config"],
        session_buffer=_state["session_buffer"],
        loop=_state["consolidation_loop"],
        ha_context=ha_context,
    )
    # Store loop for reuse across runs (preserves graph, registries, cycle count)
    if "loop" in result:
        loop = result.pop("loop")
        _state["consolidation_loop"] = loop
        # Update model reference — ConsolidationLoop wraps the base model
        # in PeftModel with adapters during training
        _state["model"] = loop.model
    _state["last_consolidation"] = datetime.now(timezone.utc).isoformat()
    _state["last_consolidation_result"] = {k: v for k, v in result.items() if k != "loop"}
    _state["router"].reload()
    # Refresh HA entity graph if available
    ha_graph = _state.get("ha_graph")
    ha_client = _state.get("ha_client")
    if ha_graph is not None and ha_client is not None:
        ha_client.load_entity_map()
        ha_services = ha_client.get_services()
        ha_graph.refresh(ha_client._raw_states, ha_services)
    logger.info("Consolidation result: %s", result)


def _consolidation_done_callback(future):
    """Called when consolidation completes or fails."""
    _state["consolidating"] = False
    exc = future.exception()
    if exc:
        logger.exception("Consolidation failed: %s", exc)


def _maybe_trigger_scheduled_consolidation() -> str:
    """Gate + dispatch the scheduled extract + BG-train run.

    Returns a short status string. GPU-busy states return 'deferred_*' so the
    caller can distinguish a missed-but-rescheduled tick from a true no-op.
    The systemd timer will fire again on its next wall-clock tick and retry.
    """
    if _state["consolidating"]:
        logger.info("Scheduler tick: consolidation already running — deferred")
        return "deferred_already_running"
    if _state["mode"] != "local":
        logger.info(
            "Scheduler tick: mode=%s (reason=%s) — deferred, will retry on next tick",
            _state["mode"],
            _state.get("cloud_only_reason"),
        )
        return "deferred_cloud_only"
    bg = _state.get("background_trainer")
    if bg is not None and bg.is_training:
        logger.info("Scheduler tick: background training active — deferred")
        return "deferred_bg_training"

    buffer = _state["session_buffer"]

    # Retroactive voice-match claim: scan orphan sessions against every
    # enrolled speaker. Attributes sessions whose embeddings match an
    # existing profile at high confidence. Cheap — centroids are cached.
    _retro_claim_orphan_sessions()

    pending = buffer.get_pending()
    if not pending:
        logger.info("Scheduler tick: no pending sessions — noop")
        return "noop_no_pending"
    if not any(s.get("speaker_id") for s in pending):
        logger.info("Scheduler tick: no sessions with speaker_id — noop")
        return "noop_no_speaker"

    logger.info("Scheduler tick: %d pending sessions, starting extract + train", len(pending))
    _state["consolidating"] = True

    event_loop = asyncio.get_running_loop()
    future = event_loop.run_in_executor(None, _extract_and_start_training)
    future.add_done_callback(_scheduled_extract_done_callback)
    return "started"


def _retro_claim_orphan_sessions() -> int:
    """Attribute orphaned pending sessions to existing speaker profiles via voice match.

    Runs before each scheduled consolidation tick. For each enrolled speaker,
    invokes `SessionBuffer.claim_sessions_for_speaker` which scans orphan
    sessions for user turns whose stored embeddings match the speaker at
    high confidence. Idempotent — already-claimed sessions are skipped.

    Runs on the asyncio event loop thread, so there is no lock contention
    over `SessionBuffer._turns` with the chat handler's enrollment trigger
    (also event-loop-bound). `_extract_and_start_training` runs in an
    executor but is dispatched only after this function returns.

    A corrupt profile or session must not take down the entire tick —
    each speaker is wrapped independently.

    Returns the total number of sessions claimed across all speakers.
    """
    buffer = _state.get("session_buffer")
    store = _state.get("speaker_store")
    if buffer is None or store is None:
        return 0

    total = 0
    for profile in store.list_profiles():
        try:
            claimed = buffer.claim_sessions_for_speaker(profile["id"], profile["name"], store)
        except Exception:
            logger.exception(
                "Retro-claim failed for speaker %s (%s) — skipping",
                profile.get("name", "?"),
                profile.get("id", "?"),
            )
            continue
        total += claimed
    if total > 0:
        logger.info("Retro-claim: attributed %d orphan sessions to known speakers", total)
    else:
        logger.debug("Retro-claim: no new orphan sessions matched known speakers")
    return total


def _scheduled_extract_done_callback(future):
    """Clear the consolidating flag only if the extraction phase failed.

    On success, _extract_and_start_training has handed off to the background
    trainer and the flag will be cleared by _finalize_background_training.
    """
    exc = future.exception()
    if exc:
        logger.exception("Scheduled extraction failed: %s", exc)
        _state["consolidating"] = False


def _extract_and_start_training():
    """Extract sessions, prepare training data, start background trainer.

    Runs in executor thread. Extraction holds the GPU lock (LLM inference).
    Training runs in a background thread and acquires/releases the GPU lock
    per training step via the BackgroundTrainer pause/resume mechanism.
    """
    from paramem.server.background_trainer import BackgroundTrainer, TrainingJob
    from paramem.server.consolidation import (
        _increment_key_sessions,
        _promote_mature_keys,
        _save_key_metadata,
        _save_keyed_pairs_for_router,
        _save_registry,
        _save_simulation_results,
        create_consolidation_loop,
    )

    config = _state["config"]
    session_buffer = _state["session_buffer"]

    # Create or reuse consolidation loop
    loop = _state.get("consolidation_loop")
    if loop is None:
        loop = create_consolidation_loop(
            _state["model"],
            _state["tokenizer"],
            config,
            state_provider=lambda: _state,
        )
        _state["consolidation_loop"] = loop
        _state["model"] = loop.model

    # Read HA home context for location validation
    ha_context = None
    ha_client = _state.get("ha_client")
    if ha_client is not None:
        ha_context = ha_client.get_home_context()

    # --- Phase 1: Extract all sessions (holds GPU lock) ---
    from paramem.server.gpu_lock import gpu_lock_sync

    pending = session_buffer.get_pending()
    all_episodic_qa = []
    all_procedural_rels = []
    session_ids = []
    speaker_ids = []

    with gpu_lock_sync():
        for session in pending:
            session_id = session["session_id"]
            transcript = session["transcript"]
            session_speaker_id = session.get("speaker_id")
            session_ids.append(session_id)

            if not session_speaker_id:
                continue

            if loop.shutdown_requested:
                logger.info("Shutdown — stopping extraction early")
                break

            speaker_name = None
            if _state.get("speaker_store") is not None:
                try:
                    speaker_name = _state["speaker_store"].get_name(session_speaker_id)
                except Exception as e:
                    logger.warning("speaker_store.get_name(%s) failed: %s", session_speaker_id, e)

            episodic_qa, procedural_rels = loop.extract_session(
                transcript,
                session_id,
                speaker_id=session_speaker_id,
                speaker_name=speaker_name,
                ha_context=ha_context,
                stt_correction=config.consolidation.extraction_stt_correction,
                ha_validation=config.consolidation.extraction_ha_validation,
                noise_filter=config.consolidation.extraction_noise_filter,
                noise_filter_model=config.consolidation.extraction_noise_filter_model,
                noise_filter_endpoint=config.consolidation.extraction_noise_filter_endpoint or None,
                ner_check=config.consolidation.extraction_ner_check,
                ner_model=config.consolidation.extraction_ner_model,
                plausibility_judge=config.consolidation.extraction_plausibility_judge,
                plausibility_stage=config.consolidation.extraction_plausibility_stage,
                verify_anonymization=config.consolidation.extraction_verify_anonymization,
                source_type=session.get("source_type", "transcript"),
            )
            _increment_key_sessions(loop, session_id)

            for qa in episodic_qa:
                qa["speaker_id"] = session_speaker_id
            for rel in procedural_rels:
                rel["speaker_id"] = session_speaker_id

            all_episodic_qa.extend(episodic_qa)
            all_procedural_rels.extend(procedural_rels)
            speaker_ids.append(session_speaker_id)

    if not all_episodic_qa and not all_procedural_rels:
        logger.info("No QA pairs extracted — skipping")
        session_buffer.mark_consolidated(session_ids)
        _state["last_consolidation"] = datetime.now(timezone.utc).isoformat()
        _state["last_consolidation_result"] = {
            "status": "no_facts",
            "sessions": len(session_ids),
            "episodic_qa": 0,
            "procedural_rels": 0,
        }
        _state["consolidating"] = False
        return

    # --- Simulate mode: full pipeline minus weight update, minus archival ---
    # Blackbox-equivalent to train: same key assignment, contradiction handling,
    # SimHash registry, on-disk keyed_pairs + registry. Intentional deltas:
    #   * no LoRA weight update  → inference recalls from disk
    #   * no mark_consolidated   → pending sessions keep feeding extraction
    #                              (merger is idempotent on s/p/o)
    if config.consolidation.mode == "simulate":
        primary_speaker_sim = speaker_ids[-1] if speaker_ids else ""
        with gpu_lock_sync():
            sim_result = loop.simulated_training(
                all_episodic_qa, all_procedural_rels, speaker_id=primary_speaker_sim
            )
            newly_promoted = _promote_mature_keys(loop, config)
            _save_keyed_pairs_for_router(loop, config)
            _save_registry(loop, config)
            _save_key_metadata(loop, config)
            if config.debug:
                _save_simulation_results(all_episodic_qa, all_procedural_rels, loop, config)

        _state["last_consolidation"] = datetime.now(timezone.utc).isoformat()
        _state["last_consolidation_result"] = {
            "status": "simulated",
            "sessions": len(session_ids),
            "episodic_qa": len(all_episodic_qa),
            "procedural_rels": len(all_procedural_rels),
            "newly_promoted": len(newly_promoted),
            "simulated": sim_result.get("simulated", True),
        }
        _state["consolidating"] = False
        logger.info(
            "Simulation complete: %d episodic QA, %d procedural rels, %d promoted",
            len(all_episodic_qa),
            len(all_procedural_rels),
            len(newly_promoted),
        )
        return

    # --- Phase 2: Prepare training data (key assignment, reconstruction) ---
    if not loop.config.indexed_key_replay_enabled:
        logger.warning("Indexed key replay disabled — skipping training")
        session_buffer.mark_consolidated(session_ids)
        _state["consolidating"] = False
        return

    primary_speaker = speaker_ids[-1] if speaker_ids else ""
    jobs_data = loop.prepare_training_data(
        all_episodic_qa, all_procedural_rels, speaker_id=primary_speaker
    )

    if not jobs_data:
        logger.info("No training jobs prepared — skipping")
        session_buffer.mark_consolidated(session_ids)
        _state["consolidating"] = False
        return

    # Build TrainingJob objects
    adapter_configs = {
        "episodic": config.episodic_adapter_config,
        "semantic": config.semantic_adapter_config,
        "procedural": config.procedural_adapter_config,
    }
    training_jobs = [
        TrainingJob(
            keyed_pairs=keyed_pairs,
            adapter_name=adapter_name,
            adapter_config=adapter_configs[adapter_name],
        )
        for adapter_name, keyed_pairs in jobs_data
    ]

    # --- Phase 3: Start background training ---
    def _on_training_complete():
        """Called from training thread when all jobs finish.

        Disk I/O is safe from any thread. SimHash updates and state
        mutations are posted to the event loop via call_soon_threadsafe
        to avoid race conditions with inference reads.
        """
        # Disk I/O — safe from any thread
        loop.finalize_training()
        _promote_mature_keys(loop, config)
        _save_keyed_pairs_for_router(loop, config)
        _save_registry(loop, config)
        _save_key_metadata(loop, config)
        session_buffer.mark_consolidated(session_ids)

        # SimHash updates + state mutations + router reload — all on event loop
        aio_loop = _state.get("event_loop")
        if aio_loop is not None and aio_loop.is_running():
            aio_loop.call_soon_threadsafe(_finalize_background_training, loop, jobs_data)
        else:
            _finalize_background_training(loop, jobs_data)

    # Build TrainingJob objects and start background trainer
    bt = BackgroundTrainer(
        model=_state["model"],
        tokenizer=_state["tokenizer"],
        training_config=config.training_config,
        output_dir=config.adapter_dir,
        temp_limit=config.consolidation.training_temp_limit,
        temp_check_interval=config.consolidation.training_temp_check_interval,
        quiet_hours_mode=config.consolidation.quiet_hours_mode,
        quiet_hours_start=config.consolidation.quiet_hours_start,
        quiet_hours_end=config.consolidation.quiet_hours_end,
    )

    def _on_training_error():
        loop.rollback_preparation()
        _state["consolidating"] = False
        logger.error("Background training failed — state rolled back")

    _state["background_trainer"] = bt
    bt.start_jobs(
        training_jobs,
        on_complete=_on_training_complete,
        on_error=_on_training_error,
    )
    logger.info(
        "Extraction done — %d training jobs started in background",
        len(training_jobs),
    )


def _finalize_background_training(loop, jobs_data=None):
    """Update shared server state after background training.

    Must run on the event loop thread to avoid racing with /chat.
    SimHash updates happen here (not in the training thread) for thread safety.
    """
    from paramem.training.indexed_memory import build_registry

    # Update SimHash registries on the event loop thread
    if jobs_data is not None:
        for adapter_name, keyed_pairs in jobs_data:
            new_registry = build_registry(keyed_pairs)
            if adapter_name == "episodic":
                loop.episodic_simhash.update(new_registry)
            elif adapter_name == "semantic":
                loop.semantic_simhash.update(new_registry)
            elif adapter_name == "procedural":
                loop.procedural_simhash.update(new_registry)

    loop.model.eval()
    _state["last_consolidation"] = datetime.now(timezone.utc).isoformat()
    _state["router"].reload()
    total_keys = (
        len(loop.episodic_simhash) + len(loop.semantic_simhash) + len(loop.procedural_simhash)
    )
    _state["last_consolidation_result"] = {
        "status": "trained",
        "total_keys": total_keys,
        "jobs": [job[0] for job in (jobs_data or [])],
    }
    _state["consolidating"] = False
    logger.info("Background training complete — %d keys saved", total_keys)


# --- Speaker enrollment (utterance-driven) ---
#
# When an unknown speaker talks, their voice embedding is stored alongside
# each transcript turn and a canonical Speaker{N} ID is allocated. The
# chat handler then invokes _run_enrollment_for_group synchronously on
# every anonymous turn so the LLM extractor and the follow-on
# store.enroll / claim_sessions / cleanup run before the response is
# rendered. The LLM is the sole filter — non-introduction turns return
# NONE and have no side effect. There is no background idle loop and
# no regex pre-filter; the user's own utterance is always the trigger.


def _extract_name_via_llm(
    turns: list[dict],
    model,
    tokenizer,
) -> str | None:
    """Use the local LLM to extract a speaker's self-introduced name from transcript.

    The LLM reasons about context to distinguish the speaker's own name
    from other names mentioned in conversation. Returns the name or None.
    """
    from paramem.evaluation.recall import generate_answer

    lines = []
    for turn in turns:
        role = turn.get("role", "unknown")
        text = turn.get("text", "")
        lines.append(f"{role}: {text}")
    transcript_text = "\n".join(lines)

    system_msg = (
        "You extract speaker names from conversation transcripts. "
        "Return ONLY the first name the speaker claims as their own identity, "
        "or NONE if they did not introduce themselves. "
        "Do NOT extract names of other people mentioned in conversation."
    )
    user_msg = (
        "Extract the speaker's self-introduced name from this transcript.\n\n"
        "Examples:\n"
        'Transcript: "user: My name is Alex. What time is it?"\n'
        "Answer: Alex\n\n"
        'Transcript: "user: Tell me about John Smith\'s schedule"\n'
        "Answer: NONE\n\n"
        'Transcript: "user: Stop playing music"\n'
        "Answer: NONE\n\n"
        f"Transcript:\n{transcript_text}\n\n"
        "Answer:"
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    result = generate_answer(model, tokenizer, prompt, max_new_tokens=64, temperature=0.0)
    result = result.strip().strip('"').strip("'").strip(".")

    if not result or result.upper() == "NONE" or len(result) > 30:
        return None

    words = result.split()
    if len(words) > 3 or len(words) == 0:
        return None

    return result


async def _run_enrollment_for_speaker(
    speaker_id: str,
    conv_id: str,
    embedding: list[float],
    *,
    extra_turns: list[dict] | None = None,
) -> str | None:
    """Extract speaker name for an anonymous profile and apply enrollment.

    Invoked synchronously by the chat handler on every anonymous turn.
    The live turn is passed through ``extra_turns`` so the LLM sees it
    without waiting for the post-handler buffer append. The LLM is the
    sole filter — non-introduction turns return None and have no side
    effect (no enroll, no buffer mutation, no cleanup).

    Works for both freshly-promoted and returning anonymous speakers:
    operates on the speaker_id + voice embedding directly, without
    relying on a transient ``unknown_speakers`` group (which exists only
    for new voices in the current server lifetime).

    Returns the enrolled name on success, or ``None`` (extractor returned
    NONE / voice already enrolled under a different profile / no turns
    / model not loaded). On success: store profile is upgraded in place
    (speaker_id preserved), this conversation is attributed, cross-
    session orphan sessions are retro-claimed via voice matching, and
    any in-memory ``unknown_speakers`` group containing this conv is
    dropped along with its ``pending_enrollments`` entries.
    """
    buffer = _state["session_buffer"]
    store = _state.get("speaker_store")
    model = _state.get("model")
    tokenizer = _state.get("tokenizer")

    if not store or not model or not tokenizer:
        return None

    all_turns: list[dict] = list(extra_turns) if extra_turns else []
    all_turns.extend(buffer.get_session_turns(conv_id))

    if not all_turns:
        return None

    from paramem.server.gpu_lock import gpu_lock

    async with gpu_lock():
        loop = asyncio.get_running_loop()
        extracted = await loop.run_in_executor(
            None,
            lambda t=all_turns: _extract_name_via_llm(t, model, tokenizer),
        )

    if not extracted:
        return None

    new_id = store.enroll(extracted, embedding)
    if not new_id:
        logger.info("Enrollment skipped for %s (voice already enrolled)", speaker_id)
        return None
    if new_id != speaker_id:
        # Voice matched a different (named) profile — the chat handler will
        # see the corrected speaker on the next turn via _resolve_speaker.
        logger.info("Enrollment redirected: voice resolved to %s (was %s)", new_id, speaker_id)

    buffer.set_speaker(conv_id, new_id, extracted)
    claimed = buffer.claim_sessions_for_speaker(new_id, extracted, store)
    _state["pending_enrollments"].discard(conv_id)
    for _gid, _g in list(_state["unknown_speakers"].items()):
        if conv_id in _g["conversations"]:
            for _cid in _g["conversations"]:
                _state["pending_enrollments"].discard(_cid)
            del _state["unknown_speakers"][_gid]
            break
    logger.info(
        "Enrollment for conv %s: %s (id=%s, claimed %d sessions)",
        conv_id,
        extracted,
        new_id,
        claimed,
    )
    return extracted


# --- GPU lifecycle ---
#
# GPU management is service-level: stop the service to free GPU,
# restart to reclaim. No in-process GPU release/reclaim — process
# exit is the only way to fully free the CUDA context.
#
# Flow:
#   tresume: stop service → start with --defer-model → launch training
#   Training finishes: auto-reclaim detects GPU free → restart service
#   Fresh lifespan loads both LLM + STT on startup


def _gpu_occupied() -> bool:
    """Check if another process is using the GPU at startup time.

    Used to prevent loading the model when an ML workload is running.
    """
    return _gpu_has_compute_processes()


_HOLD_ENV_VARS = (
    "PARAMEM_EXTRA_ARGS",
    "PARAMEM_HOLD_PID",
    "PARAMEM_HOLD_STARTED_AT",
    "PARAMEM_HOLD_CMD",
)


def _unquote_systemd_value(v: str) -> str:
    """Reverse systemd's ANSI-C quoting on ``show-environment`` output.

    systemd emits values containing shell-special characters (spaces,
    slashes, quotes, …) as ``$'...'`` with backslash escapes.  Simple
    values are emitted verbatim.  Double-quoted form is also accepted
    for forward compatibility.
    """
    if len(v) >= 3 and v.startswith("$'") and v.endswith("'"):
        inner = v[2:-1]
        try:
            return inner.encode("latin-1", "backslashreplace").decode("unicode_escape")
        except UnicodeDecodeError:
            return inner
    if len(v) >= 2 and v.startswith('"') and v.endswith('"'):
        return v[1:-1]
    return v


def _read_systemd_user_env() -> dict[str, str]:
    """Read the current systemd --user environment block.

    Needed because PARAMEM_EXTRA_ARGS / PARAMEM_HOLD_* are set via
    ``systemctl --user set-environment`` and are inherited by services at
    start time.  An already-running process keeps its original os.environ
    snapshot, so we re-read systemd's block to get the live value.
    """
    try:
        result = subprocess.run(
            ["systemctl", "--user", "show-environment"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return {}
    env: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            env[k] = _unquote_systemd_value(v)
    return env


def _pid_alive(pid: int) -> bool:
    """Return True if a process with the given PID exists.

    Uses signal 0 (no-op) which only checks existence + permissions.
    PermissionError means the PID exists but is owned by another user —
    still counts as alive for our purposes.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _get_hold_state() -> dict:
    """Inspect the PARAMEM_EXTRA_ARGS=--defer-model hold in systemd --user env.

    Returned dict shape:
        {
            "hold_active":   bool,         # env var set with --defer-model
            "owner_pid":     int | None,   # from PARAMEM_HOLD_PID, if stamped
            "owner_alive":   bool | None,  # PID liveness; None when unstamped
            "age_seconds":   int | None,   # now - PARAMEM_HOLD_STARTED_AT
            "owner_hint":    str | None,   # PARAMEM_HOLD_CMD, e.g. "python / paramem.server.app"
        }

    Used by /status for operator visibility and by _auto_reclaim_loop to
    distinguish a legitimate mid-training hold from an orphaned env var.
    """
    env = _read_systemd_user_env()
    extra_args = env.get("PARAMEM_EXTRA_ARGS", "")
    hold_active = "--defer-model" in extra_args
    if not hold_active:
        return {
            "hold_active": False,
            "owner_pid": None,
            "owner_alive": None,
            "age_seconds": None,
            "owner_hint": None,
        }
    pid_str = env.get("PARAMEM_HOLD_PID", "").strip()
    owner_pid: int | None = None
    if pid_str.isdigit():
        owner_pid = int(pid_str)
    owner_alive: bool | None = None
    if owner_pid is not None:
        owner_alive = _pid_alive(owner_pid)
    started_str = env.get("PARAMEM_HOLD_STARTED_AT", "").strip()
    age_seconds: int | None = None
    if started_str.isdigit():
        age_seconds = max(0, int(time.time()) - int(started_str))
    owner_hint = env.get("PARAMEM_HOLD_CMD") or None
    return {
        "hold_active": True,
        "owner_pid": owner_pid,
        "owner_alive": owner_alive,
        "age_seconds": age_seconds,
        "owner_hint": owner_hint,
    }


def _clear_hold_env() -> bool:
    """Unset PARAMEM_EXTRA_ARGS / PARAMEM_HOLD_PID / PARAMEM_HOLD_STARTED_AT.

    Returns True on success.  Idempotent — safe to call when variables are
    already unset.
    """
    try:
        subprocess.run(
            ["systemctl", "--user", "unset-environment", *_HOLD_ENV_VARS],
            check=False,
            capture_output=True,
            timeout=5,
        )
        return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        logger.exception("Failed to unset PARAMEM_EXTRA_ARGS / PARAMEM_HOLD_*")
        return False


def _gpu_has_compute_processes() -> bool:
    """Check if any non-server process is using the GPU."""
    try:
        result = subprocess.run(
            [_NVIDIA_SMI, "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        pids = [int(p.strip()) for p in result.stdout.strip().split("\n") if p.strip()]
        # Filter out our own process
        own_pid = os.getpid()
        external_pids = [p for p in pids if p != own_pid]
        return len(external_pids) > 0
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError) as e:
        # Fail safe: if we can't check, assume GPU is occupied
        logger.warning(
            "nvidia-smi unavailable at %s (%s) — assuming GPU occupied",
            _NVIDIA_SMI,
            type(e).__name__,
        )
        return True


async def _auto_reclaim_loop(interval_minutes: int = 10):
    """Periodically check if GPU is free and reclaim it.

    Only started when the server is in cloud-only/defer mode (no model loaded).
    Each tick:

    1. If any non-server GPU compute process is running → wait.
    2. Else inspect the hold state (PARAMEM_EXTRA_ARGS in systemd --user env):
       - Hold cleared → reclaim: restart for a clean model load.
       - Hold set, holder PID alive → legitimate mid-training window
         (cooldown, model swap) → keep polling, do not restart.
       - Hold set, holder PID dead or unregistered → orphan suspected →
         emit one WARN and exit the loop.  Operator clears via
         ``pstatus --force-local`` (POST /gpu/force-local).

    Exiting on orphan stops the infinite systemctl-restart loop that was
    happening when a SIGKILLed test left PARAMEM_EXTRA_ARGS=--defer-model
    behind: visibility over silent auto-heal, per design.
    """
    interval_seconds = interval_minutes * 60
    while True:
        await asyncio.sleep(interval_seconds)
        if _gpu_has_compute_processes():
            logger.debug("Auto-reclaim: GPU still occupied, waiting")
            continue
        hold = _get_hold_state()
        if not hold["hold_active"]:
            logger.info("Auto-reclaim: GPU free — restarting for clean model load")
            _restart_service()
            return
        owner_pid = hold["owner_pid"]
        owner_alive = hold["owner_alive"]
        if owner_alive is True:
            # Holder alive (mid-cycle model swap or similar) — respect the hold.
            logger.debug(
                "Auto-reclaim: compute-free but holder PID %s alive — waiting",
                owner_pid,
            )
            continue
        if owner_alive is False:
            logger.warning(
                "Auto-reclaim: PARAMEM_EXTRA_ARGS=--defer-model still set but "
                "holder PID %s is dead — orphaned hold. Clear and reclaim with "
                "`pstatus --force-local` (POST /gpu/force-local).",
                owner_pid,
            )
        else:
            logger.warning(
                "Auto-reclaim: PARAMEM_EXTRA_ARGS=--defer-model still set but no "
                "holder PID registered — suspected orphan. Clear and reclaim with "
                "`pstatus --force-local` (POST /gpu/force-local)."
            )
        return


def _restart_service():
    """Restart the systemd service for a clean process.

    Used by auto-reclaim: model loading requires CUDA initialization on the
    main thread at process startup. In-process reclaim from a thread pool
    creates meta-device tensors that aren't resident in VRAM.
    """
    logger.info("Restarting paramem-server service...")
    try:
        subprocess.Popen(
            ["systemctl", "--user", "restart", "paramem-server"],
            start_new_session=True,
        )
    except Exception:
        logger.exception("Failed to restart service")


# --- Entry point ---


def main():
    parser = argparse.ArgumentParser(description="ParaMem Server")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/server.yaml",
        help="Path to server config YAML",
    )
    parser.add_argument(
        "--cloud-only",
        action="store_true",
        help="Start in cloud-only mode permanently (skip model loading, no auto-reclaim)",
    )
    parser.add_argument(
        "--defer-model",
        action="store_true",
        help="Start without model (cloud-only) but auto-reclaim GPU when free",
    )
    args = parser.parse_args()

    # Setup
    project_root = Path(__file__).parent.parent.parent
    load_dotenv(project_root / ".env")

    # Per-secret file layout under ~/.config/paramem/secrets/ with strict
    # permissions. Loaded after .env so shell env + .env take precedence.
    # Missing directory = back-compat no-op.
    from paramem.server.secret_store import (  # noqa: E402
        load_secrets_from_dir,
    )
    from paramem.server.secret_store import (
        log_startup_posture as log_secrets_posture,
    )

    _loaded_secrets = load_secrets_from_dir()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    log_secrets_posture(_loaded_secrets)

    config = load_server_config(args.config)
    _state["config"] = config
    _state["config_path"] = args.config
    _state["cloud_only_startup"] = args.cloud_only
    _state["defer_model"] = args.defer_model

    # Clear stale systemd env var so it can't resurface on next restart.
    # tresume sets PARAMEM_EXTRA_ARGS=--defer-model; tpause normally clears
    # it, but if a training run was paused without tpause (e.g. killed),
    # the var persists and every restart silently starts cloud-only. When
    # this process starts WITHOUT --defer-model, treat it as authoritative
    # and clear the var so the intent is consistent with future restarts.
    if not args.defer_model and not args.cloud_only:
        import subprocess

        try:
            subprocess.run(
                ["systemctl", "--user", "unset-environment", "PARAMEM_EXTRA_ARGS"],
                check=False,
                capture_output=True,
                timeout=5,
            )
        except Exception:
            pass

    import uvicorn

    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        log_level="info",
        log_config=None,
    )


if __name__ == "__main__":
    main()
