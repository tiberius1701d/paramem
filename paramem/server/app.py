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
from contextlib import asynccontextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from paramem.server.user_tokens import UserTokenStore

import torch
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Migration / backup imports at module level so tests can patch them.
from paramem.backup.backup import write as backup_write
from paramem.backup.backup import write_bundle
from paramem.backup.types import ArtifactKind
from paramem.graph.extractor import ExtractionFailed
from paramem.models.loader import load_base_model, switch_adapter, unload_model
from paramem.server import calibrate as calibrate_module
from paramem.server.active_store_migration import migrate
from paramem.server.background_trainer import BackgroundTrainer
from paramem.server.cloud import get_cloud_agent
from paramem.server.config import TTSConfig, TTSVoiceConfig, load_server_config
from paramem.server.consolidation import create_consolidation_loop
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
    read_trial_marker,
    write_trial_marker,
)
from paramem.server.voice_pipeline import process_utterance
from paramem.server.vram_guard import (
    VramExhausted,
    apply_process_cap,
    check_vram_headroom,
    safe_empty_cache,
    vram_measure,
    vram_scope,
)
from paramem.server.vram_predict import predict_base_bytes
from paramem.server.vram_validator import (
    assess_topology,
    check_post_load_budget,
    estimate_stt_bytes,
    estimate_tts_bytes,
    format_baseline_fit,
)
from paramem.training.consolidation import AbortedDuringConsolidation
from paramem.training.thermal_throttle import ThermalPolicy
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
    "user_token_store": None,
    "session_buffer": None,
    "router": None,
    "sota_agent": None,
    "sota_providers": {},
    "ha_client": None,
    "consolidation_loop": None,
    "memory_store": None,
    # Set to a BootDegraded dict when ``inference.preload_cache=True`` and the
    # lifespan preload could not materialise every active key.  Recall is
    # unaffected — the inference path probes the weights on a cache miss
    # (``MemoryStore.probe`` on-miss source delegation); only first-recall
    # latency is paid until the cache re-warms.  Surfaced in the /status
    # attention block; cleared on full hydration, on preload_cache=False
    # (intentional opt-out), and on a successful live apply.
    "boot_degraded": None,
    "consolidating": False,
    "last_consolidation": None,
    # Last structured error from a consolidation attempt. Populated by the
    # done-callback when the cycle raised an exception that the operator
    # should be able to see via /status without scraping logs. Currently
    # populated only for VramExhausted; other failures still log loudly.
    # Shape: {"type": "vram_exhausted", "phase": str, "at": iso8601} | None
    "last_consolidation_error": None,
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
    "stt_gpu": None,
    "stt_cpu": None,
    "tts_manager": None,
    "tts_gpu": None,
    "tts_cpu": None,
    "voice_box": None,  # {"stt": <active>, "tts_manager": <active>} or None
    "voice_profile": None,  # "gpu" | "cpu" | None (pre-init)
    # Boot-time VRAM topology assessment (TopologyAssessment | None). Computed
    # once at startup and reused by the GPU reclaim path's live-budget
    # pre-flight — config is static for the process, so there is no second
    # estimator. None when boot couldn't assess (cloud-only / HF cache miss).
    "topology_assessment": None,
    "last_reclaim_error": None,  # {"at", "error", "attempt_count"} or None
    "wyoming_server": None,
    "wyoming_tts_server": None,
    "latest_embedding": None,
    "latest_language_detection": None,  # {language: str, probability: float}
    "last_chat_time": None,
    "last_chat_monotonic": None,  # time.monotonic() stamp of the most recent /chat turn
    "pending_enrollments": set(),
    # Unknown speaker groups: temp_id → {embeddings, conversations, first_seen}.
    # Mutations happen on the asyncio event loop (cooperative scheduling).
    # Safe without locks.
    "unknown_speakers": {},
    "migration": None,  # MigrationStashState — populated in lifespan
    "server_started_at": "",  # ISO-8601 UTC timestamp set in lifespan
    # Set to True when boot-time load_registries_from_disk raises.  The store
    # may hold zero or partial tiers; downstream migration must not run against
    # a degraded store (it would vacuously complete with no-op relocations).
    # Cleared to False on successful load.
    "store_load_degraded": False,
    # Whether the daily age identity was loadable at boot.  Set in lifespan
    # alongside ``encryption``; used by ``GET /integrity`` and the boot
    # integrity gate to distinguish no-key from corruption failures.
    "daily_loadable": False,
}


# --- Request/Response schemas ---


class ChatRequest(BaseModel):
    text: str
    conversation_id: str = "default"
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
    # Active-store migration state. True when the operator flipped
    # consolidation.mode (simulate↔train) and the per-tier migration is
    # in progress or interrupted. Inference falls back to ``effective_mode``
    # while this is True (the source store stays authoritative until the
    # 1.0 recall gate has cleared every tier).
    pending_rehydration: bool = False
    # When ``pending_rehydration`` is True, the mode the inference path
    # is actually using (== source_mode of the in-flight migration).
    # Equals ``mode_config`` otherwise.
    effective_mode: str | None = None
    last_consolidation: str | None
    # Structured error from the most recent failed consolidation, surfaced
    # so operators can see VRAM exhaustion without scraping journald.
    # None means the last cycle finished cleanly (or none has run yet).
    # Shape: {"type": "vram_exhausted", "phase": str, "at": iso8601}
    last_consolidation_error: dict | None = None
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
    # True when TTS is up but not every CONFIGURED voice loaded (e.g. Piper voices
    # missing while MMS loaded). is_loaded stays "any engine" for internal gates;
    # this is the honest health signal so /status doesn't report a dead default voice
    # as fully healthy.
    tts_degraded: bool = False
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
    # processes) and clear with ``pstatus --acquire``.  Schema:
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
    # Auto-reclaim error tracking (D3). Populated when the in-process reclaim
    # loop fails a tick; cleared on next successful reclaim. None means the last
    # reclaim completed cleanly (or none has run yet).
    # Shape: {"at": <iso8601>, "error": <str>, "attempt_count": <int>}
    last_reclaim_error: dict | None = None


class IntegrityCheckItem(BaseModel):
    """One file-level check result from the integrity verifier."""

    path: str
    category: str
    tier: str
    status: str
    detail: str


class IntegrityResponse(BaseModel):
    """Response schema for ``GET /integrity``."""

    ok: bool
    checks: list[IntegrityCheckItem]
    failures: list[IntegrityCheckItem]


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


# --- Speaker forget schemas ---


class SpeakerForgetRequest(BaseModel):
    """Request body for ``POST /speaker/forget``.

    Attributes
    ----------
    speaker_id:
        The speaker ID to forget (e.g. ``"Speaker0"``).  Exact match.
    strategy:
        Erasure strategy.  Only ``"mark_stale"`` is implemented; the field is
        a future extension point for a ``"discard_interim"`` strategy
        (discard the whole interim slot).

    Note
    ----
    ``discard_interim`` strategy (discard the whole interim slot) is out of
    scope for this revision.  Extend this field and add a handler branch when
    that strategy is needed.
    """

    speaker_id: str
    strategy: str = "mark_stale"


class SpeakerForgetResponse(BaseModel):
    """Response body for ``POST /speaker/forget``.

    Attributes
    ----------
    removed_speaker:
        ``True`` when the speaker profile was found and removed from
        :class:`~paramem.server.speaker.SpeakerStore`.  ``False`` when the
        speaker ID was unknown to the store (no profile to delete).
    stale_keys:
        Indexed-memory keys that were removed from every per-tier
        :class:`~paramem.training.key_registry.KeyRegistry` and their
        corresponding SimHash entries — the keys' weights decay naturally
        through future training cycles.
    discarded_sessions:
        Pending conversation IDs that were found in the
        :class:`~paramem.server.session_buffer.SessionBuffer` attributed to
        the speaker and discarded (JSONL deleted, turns dropped).
    """

    removed_speaker: bool
    stale_keys: list[str]
    discarded_sessions: list[str]


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
    mode_switch: dict | None = None
    base_change: dict | None = None


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
        Dict ``{"config": "<abs_path>"}`` of the pre-migration config backup
        slot, or ``None``.
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
        ``"TRIAL"`` on the normal trial path; ``"LIVE"`` on a pure
        ``consolidation.mode`` change (applied directly, no trial).
    trial_started_at:
        ISO-8601 UTC timestamp when TRIAL was entered (or when the mode-switch
        confirm completed).
    pre_trial_config_sha256:
        SHA-256 of the live config before the atomic rename.
    candidate_config_sha256:
        SHA-256 of the candidate bytes.
    backup_paths:
        Dict ``{"config": "<abs_path>"}`` of the pre-migration config backup
        slot directory.  Empty dict ``{}`` on a pure mode-switch (no backup
        written — reverting = flip the mode back).
    trial_adapter_dir:
        Absolute path to the trial adapter directory.  ``""`` on a pure
        mode-switch (no trial runs).
    trial_graph_dir:
        Absolute path to the trial graph directory.  ``""`` on a pure
        mode-switch (no trial runs).
    mode_switch:
        Present (non-None) only on a pure ``consolidation.mode`` change.
        Describes the direction, mechanism, and semantics of the rebuild so
        CLI and API consumers can explain the outcome without polling for
        gate results.
    base_swap:
        ``True`` when the confirm launched a base-model-swap background task.
        The server is in ``"TRIAL"`` state; Phase A runs asynchronously.
        Phase A captures all keyed facts from the current base model, then
        reloads the new base model in-process (the server is briefly
        cloud-only during the reload).  Phase B retrains all tiers on the new
        base and gates on 100% recall.  No server restart is required.
        Poll ``/migration/status`` for progress; use ``POST /migration/rollback``
        to restore the prior base model from the pre-migration bundle.
    """

    state: str
    trial_started_at: str
    pre_trial_config_sha256: str
    candidate_config_sha256: str
    backup_paths: dict[str, str]
    trial_adapter_dir: str
    trial_graph_dir: str
    mode_switch: dict | None = None
    base_swap: bool = False


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
        ``True`` when a restart is still needed (live apply was declined or
        failed, or a named R-PORT/R-PATHS carve fired).  ``False`` when the
        config was applied fully in-process.
    restart_hint:
        Human-readable restart command string.
    pre_migration_backup_retained:
        Always ``True`` — the A-config backup is retained post-accept.
    applied_live:
        ``True`` when the new config was applied in-process without a restart.
        ``False`` when a restart is required (apply failed, or a named carve).
    restart_required_reason:
        Named reason for ``restart_required=True``.  One of
        ``"stt_port_change"``, ``"tts_port_change"``, ``"paths_change"``,
        ``"apply_failed"``, ``"lock_timeout"``, ``"consolidating"``, or
        ``None`` when no restart is needed.
    auto_restart_scheduled:
        Always ``False`` — the server never self-fires a restart.  Retained
        for backward compatibility; use ``restart_eligible`` instead.
    restart_eligible:
        ``True`` when an R-PORT carve pre-flighted successfully and the CLI
        may trigger a prompted restart via the ``restart_hint`` command.
        ``False`` for R-PATHS (data-not-migrated warning; operator-driven)
        and for failures.  The server does NOT fire the restart — the CLI
        prompts the operator and runs ``restart_hint`` via subprocess on
        consent.
    """

    state: str
    trial_adapter_archive_path: str
    restart_required: bool
    restart_hint: str
    pre_migration_backup_retained: bool
    applied_live: bool = False
    restart_required_reason: str | None = None
    auto_restart_scheduled: bool = False
    restart_eligible: bool = False


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
        ``True`` when a restart is still needed.  For rollback, the no-op
        skip (disk hash == memory hash) returns ``applied_live=True`` and
        ``restart_required=False`` because config A is already in memory.
    restart_hint:
        Human-readable restart command string.
    applied_live:
        ``True`` when the config was applied in-process (or the no-op skip
        confirmed it was already applied).  ``False`` on apply failure.
    restart_required_reason:
        Named reason for ``restart_required=True``, or ``None``.
    auto_restart_scheduled:
        Always ``False`` — the server never self-fires a restart.  Retained
        for backward compatibility; use ``restart_eligible`` instead.
    restart_eligible:
        ``True`` when an R-PORT carve pre-flighted successfully and the CLI
        may trigger a prompted restart via the ``restart_hint`` command.
        ``False`` for R-PATHS and for failures.
    """

    state: str
    trial_adapter_archive_path: str
    rollback_pre_mortem_backup_path: str
    restart_required: bool
    restart_hint: str
    applied_live: bool = False
    restart_required_reason: str | None = None
    auto_restart_scheduled: bool = False
    restart_eligible: bool = False


# --- Adapter manifest validation + mount helpers ---
#
# The boot-time validator (_mount_adapters_from_slots) and the post-full-cycle
# revalidator (_revalidate_main_adapter_manifests) share the same per-tier
# decision logic — extracted into _validate_main_adapter_slot below so there
# is a single source of truth for "what does this slot's manifest say about
# its health, and should it be mounted?"


def _compute_tier_registry_sha256(config, tier: str) -> str:
    """SHA-256 of ``<adapter_dir>/<tier>/indexed_key_registry.json`` plaintext bytes.

    Each main-tier slot's manifest is stamped with that tier's OWN registry
    hash (see ``commit_tier_slot`` step 2 / I5 ordering for the procedural
    case).  Slot matching must therefore use the per-tier hash too — passing a
    single episodic hash across all tiers makes procedural and semantic slots
    unmountable on boot.

    Hashes plaintext (after ``read_maybe_encrypted`` decrypt) — see
    ``manifest.py::build_manifest_for`` for why ciphertext-based hashing breaks
    drift detection under Security ON.  Empty string when the tier's registry
    does not exist or cannot be read.
    """
    registry_path = config.adapter_dir / tier / "indexed_key_registry.json"
    if not registry_path.exists():
        return ""
    try:
        import hashlib as _rhash

        from paramem.backup.encryption import read_maybe_encrypted as _rme

        return _rhash.sha256(_rme(registry_path)).hexdigest()
    except Exception:  # noqa: BLE001
        return ""


def _is_primary_adapter(name: str) -> bool:
    """Episodic is primary (red on mismatch); semantic / procedural are
    secondary (yellow).  Drives severity in adapter_manifest_status rows."""
    return name == "episodic"


def _record_manifest_row(
    manifest_status: dict,
    name: str,
    status: str,
    reason: str,
    severity: str,
    slot_path: "Path | None" = None,
    field: "str | None" = None,
) -> None:
    """Write one validation row into ``state['adapter_manifest_status']``."""
    manifest_status[name] = {
        "status": status,
        "reason": reason,
        "field": field,
        "severity": severity,
        "slot_path": str(slot_path.name) if slot_path else None,
        "checked_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def _validate_main_adapter_slot(
    name: str,
    adapter_cfg,
    model,
    tokenizer,
    config,
    live_registry_sha256: str,
    manifest_status: dict,
) -> "tuple[Path | None, object | None, bool]":
    """Validate one main adapter's live slot.

    Single source of truth for the per-tier validation decision.  Updates
    ``manifest_status[name]`` with a row for unhealthy outcomes; pops any
    prior row for healthy outcomes (so post-cycle revalidation clears the
    stale boot-time snapshot).

    Returns ``(slot, manifest, should_mount)``:
      * ``slot``: resolved live slot Path, or ``None`` when no matching slot.
      * ``manifest``: parsed AdapterManifest when read succeeded, else ``None``.
      * ``should_mount``: True when the boot caller should mount this slot.
        False for "no slot," "unreadable manifest," or "fingerprint mismatch."

    Used by:
      * :func:`_mount_adapters_from_slots` (boot path) — uses the return
        triple to decide what to mount.
      * :func:`_revalidate_main_adapter_manifests` (post-full-cycle) — only
        the side-effect on ``manifest_status`` matters; return value is
        discarded.
    """
    from paramem.adapters.manifest import (
        ManifestNotFoundError,
        ManifestSchemaError,
        find_live_slot,
        read_manifest,
    )
    from paramem.backup.backup import sweep_orphan_pending

    severity = "red" if _is_primary_adapter(name) else "yellow"

    kind_dir = config.adapter_dir / name
    if kind_dir.exists():
        sweep_orphan_pending(kind_dir)

    slot = find_live_slot(kind_dir, live_registry_sha256) if kind_dir.exists() else None

    if slot is None:
        has_slots = kind_dir.exists() and any(
            e for e in kind_dir.iterdir() if not e.name.startswith(".") and e.is_dir()
        )
        if has_slots:
            _record_manifest_row(
                manifest_status, name, "no_matching_slot", "no_matching_slot", severity
            )
            logger.warning("Adapter %s: no slot matching registry hash — skipping mount", name)
        else:
            manifest_status.pop(name, None)
            logger.info("Adapter %s: no slots found — fresh install", name)
        return None, None, False

    try:
        manifest = read_manifest(slot)
    except ManifestNotFoundError:
        _record_manifest_row(
            manifest_status, name, "manifest_missing", "manifest_missing", severity, slot
        )
        # Still mount — weights are present even without manifest.
        return slot, None, True
    except ManifestSchemaError as exc:
        _record_manifest_row(
            manifest_status, name, "mismatch", "manifest_unreadable", severity, slot
        )
        logger.warning("Adapter %s: corrupt meta.json (%s) — skipping mount", name, exc)
        return slot, None, False

    mismatch_field = _check_manifest_fingerprints(manifest, model, tokenizer, adapter_cfg)
    if mismatch_field is not None:
        _record_manifest_row(
            manifest_status,
            name,
            "mismatch",
            "fingerprint_mismatch",
            severity,
            slot,
            mismatch_field,
        )
        logger.warning(
            "Adapter %s: fingerprint mismatch on field '%s' — skipping mount",
            name,
            mismatch_field,
        )
        return slot, manifest, False

    unknown_field = _first_unknown_field(manifest)
    if unknown_field is not None:
        unknown_severity = "yellow" if manifest.synthesized else "red"
        _record_manifest_row(
            manifest_status,
            name,
            "migrated_unverified",
            "unknown_fields_in_manifest",
            unknown_severity,
            slot,
            unknown_field,
        )
        logger.info(
            "Adapter %s: UNKNOWN field '%s' in manifest (synthesized=%s) — mounting with warning",
            name,
            unknown_field,
            manifest.synthesized,
        )
        return slot, manifest, True

    # Healthy — clear any prior row so post-cycle revalidation removes
    # stale boot-time entries.
    manifest_status.pop(name, None)
    return slot, manifest, True


def _revalidate_main_adapter_manifests(state: dict) -> None:
    """Re-run the main-adapter manifest validator and refresh
    ``state['adapter_manifest_status']``.

    Boot's :func:`_mount_adapters_from_slots` snapshots adapter health from
    the on-disk state at startup.  After a full cycle re-saves main slots
    with a fresh registry hash, that snapshot is stale — operators see
    ``FINGERPRINT MISMATCH … PA routing DISABLED`` on /status / pstatus
    even though main is healthy.  Calling this from ``_finalize_full``
    clears those stale rows.

    Pure validation: model + tokenizer come from state but are never
    mutated.  Healthy main adapters have their row removed; unhealthy
    ones get a fresh row stamped with the current ``checked_at``.
    """
    config = state.get("config")
    model = state.get("model")
    tokenizer = state.get("tokenizer")
    if config is None or model is None or tokenizer is None:
        return

    manifest_status = state.setdefault("adapter_manifest_status", {})

    for name, adapter_cfg in (
        ("episodic", config.adapters.episodic),
        ("semantic", config.adapters.semantic),
        ("procedural", config.adapters.procedural),
    ):
        if not adapter_cfg.enabled:
            manifest_status.pop(name, None)
            continue
        _validate_main_adapter_slot(
            name,
            adapter_cfg,
            model,
            tokenizer,
            config,
            _compute_tier_registry_sha256(config, name),
            manifest_status,
        )


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
    from peft import PeftModel

    from paramem.adapters.manifest import find_live_slot
    from paramem.backup.backup import sweep_orphan_pending

    manifest_status: dict = state.setdefault("adapter_manifest_status", {})
    # Per-tier paths live at <adapter_dir>/<tier>/indexed_key_registry.json; each
    # tier's slot manifest is stamped with that tier's own registry hash, so
    # slot matching is per-tier (see _compute_tier_registry_sha256).

    def _load_one(name: str, slot: Path):
        """Mount a single adapter from *slot* onto *model* (mutates nonlocal model).

        Does not overwrite an existing manifest status row — the validator may
        have already recorded a manifest_missing or migrated_unverified row.
        Transparently decrypts age-encrypted ``adapter_model.safetensors`` via
        :func:`~paramem.models.loader._adapter_slot_for_load`.
        """
        from paramem.models.loader import _adapter_slot_for_load

        nonlocal model
        try:
            with _adapter_slot_for_load(slot) as load_path:
                if isinstance(model, PeftModel):
                    model.load_adapter(str(load_path), adapter_name=name)
                else:
                    model = PeftModel.from_pretrained(model, str(load_path), adapter_name=name)
            logger.info("Mounted adapter %s from slot %s", name, slot.name)
        except Exception as exc:
            logger.error("Failed to load adapter %s from %s: %s", name, slot, exc)
            if name not in manifest_status:
                _record_manifest_row(
                    manifest_status,
                    name,
                    "manifest_missing",
                    "load_failed",
                    "red" if _is_primary_adapter(name) else "yellow",
                    slot,
                )

    # ---- Main adapter kinds ----
    # Per-tier validation is delegated to _validate_main_adapter_slot so the
    # boot path and post-full-cycle revalidation share one decision tree.
    for name, adapter_cfg in (
        ("episodic", config.adapters.episodic),
        ("semantic", config.adapters.semantic),
        ("procedural", config.adapters.procedural),
    ):
        if not adapter_cfg.enabled:
            continue
        slot, _manifest, should_mount = _validate_main_adapter_slot(
            name,
            adapter_cfg,
            model,
            tokenizer,
            config,
            _compute_tier_registry_sha256(config, name),
            manifest_status,
        )
        if should_mount and slot is not None:
            _load_one(name, slot)

    # ---- Interim adapters ----
    from paramem.memory.interim_adapter import iter_interim_dirs

    for _interim_name, _interim_path in iter_interim_dirs(config.adapter_dir):
        sweep_orphan_pending(_interim_path)

        # Interim slots are matched against their OWN per-interim registry,
        # not the main-episodic ``live_registry_sha256``.  Each interim's
        # ``indexed_key_registry.json`` is the authoritative ledger for that
        # interim's slot; comparing to the main hash always misses when a
        # full cycle hasn't run yet (the main registry is empty).
        _interim_registry_path = _interim_path / "indexed_key_registry.json"
        _interim_hash = ""
        if _interim_registry_path.exists():
            import hashlib as _ihash

            from paramem.backup.encryption import read_maybe_encrypted as _irme

            try:
                _interim_hash = _ihash.sha256(_irme(_interim_registry_path)).hexdigest()
            except Exception:  # noqa: BLE001
                _interim_hash = ""

        slot = find_live_slot(_interim_path, _interim_hash)
        if slot is None:
            # Fallback: old flat layout (no slot-dir yet) — look for adapter files directly
            if (_interim_path / "adapter_config.json").exists() and (
                _interim_path / "adapter_model.safetensors"
            ).exists():
                logger.info("Loading interim adapter (flat layout): %s", _interim_name)
                try:
                    from paramem.models.loader import _adapter_slot_for_load

                    with _adapter_slot_for_load(_interim_path) as _load_path:
                        if isinstance(model, PeftModel):
                            model.load_adapter(str(_load_path), adapter_name=_interim_name)
                        else:
                            model = PeftModel.from_pretrained(
                                model, str(_load_path), adapter_name=_interim_name
                            )
                except Exception as exc:
                    logger.error("Failed to load interim adapter %s: %s", _interim_name, exc)
                # Flat layout has no meta.json → default "qa"
            else:
                logger.warning("Interim adapter %s: no matching slot — skipping", _interim_name)
            continue

        _load_one(_interim_name, slot)

    # ---- I5 — Registry consistency check ----
    # Drop orphan interim-tier registries that are genuinely torn: registry
    # written but neither adapter weights NOR a simulate-mode graph.json
    # landed on disk.
    #
    # Slot classification:
    #   has_weights (adapter_model.safetensors anywhere under slot dir) →
    #       train-mode slot — keep regardless of graph.json.
    #   has_graph (graph.json at slot dir root) + no weights →
    #       simulate-mode slot — keep; wiping it would destroy the prior
    #       simulate cycle's persisted state on every restart.
    #   neither →
    #       genuinely torn write (crash between registry and payload writes)
    #       — wipe.
    #
    # IMPORTANT: this is torn-save cleanup, NOT hash-mismatch cleanup.
    # ``find_live_slot is None`` is not proof that weights are missing —
    # it also returns None when slot dirs exist but their manifest's
    # ``registry_sha256`` does not match the live hash (registry drift
    # after a partial cycle).  Treating that case as "weights missing"
    # would ``rmtree`` a fully-trained adapter.  The only legitimate
    # trigger is "neither weights NOR graph.json present anywhere under
    # the interim dir".  Hash mismatch is a separate failure mode and is
    # surfaced via I4 ``manifest_status``.
    for _interim_name, _interim_reg_dir in iter_interim_dirs(config.adapter_dir):
        _interim_reg_path = _interim_reg_dir / "indexed_key_registry.json"
        if not _interim_reg_path.exists():
            continue
        _has_weights = any(_interim_reg_dir.rglob("adapter_model.safetensors"))
        if _has_weights:
            # Train-mode slot — already handled above.
            continue
        _has_graph = (_interim_reg_dir / "graph.json").exists()
        if _has_graph:
            # Simulate-mode slot — keep.
            logger.debug(
                "Startup registry check: interim adapter %s is a simulate-mode "
                "slot (has graph.json, no safetensors) — preserving",
                _interim_name,
            )
            continue
        # Neither weights nor graph.  This is only a genuinely torn write when the
        # registry carries NO active keys (the save was interrupted before any key
        # was committed).  If the registry still lists active keys, those facts were
        # never folded into a persisted main-tier slot — rmtree would permanently
        # delete them and arm prune_key_metadata_orphans to drop their bookkeeping
        # too (data-loss incident 2026-06-02).  Fold-or-refuse: preserve the dir and
        # surface it loudly instead of silently deleting unfolded facts.
        from paramem.training.key_registry import KeyRegistry as _KeyRegistry

        _active = _KeyRegistry.load(_interim_reg_path).list_active()
        if _active:
            logger.error(
                "Startup registry check: interim adapter %s has a registry with %d "
                "active key(s) but no weights and no graph.json — PRESERVING the "
                "slot (these keys were never folded into a main tier; refusing to "
                "delete unfolded facts). Re-run consolidation to fold or repair.",
                _interim_name,
                len(_active),
            )
            continue
        # Empty registry, no weights, no graph — genuinely torn write.
        logger.warning(
            "Startup registry check: interim adapter %s has an empty registry, no "
            "weights and no graph.json — removing its registry (adapter save was "
            "interrupted before any key was committed)",
            _interim_name,
        )
        import shutil as _shutil

        _shutil.rmtree(_interim_reg_dir, ignore_errors=True)

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


# --- Boot GPU drain helper ---

# Timeout and polling parameters for the boot-time GPU drain wait.
# Made module constants (not magic literals) so tests can patch them and
# operators can inspect the values without digging into call sites.
_BOOT_GPU_DRAIN_TIMEOUT_S: float = 55.0  # seconds before giving up and degrading to cloud-only
_BOOT_GPU_DRAIN_POLL_INTERVAL_S: float = 1.5  # seconds between mem_get_info polls
_BOOT_GPU_DRAIN_STABLE_READS: int = 3  # consecutive reads ≥ needed before declaring "drained"

# Upper bound on this process's CUDA primary-context baseline (kernel images +
# cuBLAS/cuDNN handles): warmed lazily during model load, process-exit-only
# (safe_empty_cache can't reclaim it; cudaDeviceReset would crash warm bnb state).
# Measured ~0.3 GiB; 0.5 GiB allows variance.  The live-reload gate credits this
# back (capped at the pristine ceiling) so a reload's free reading — taken after
# the context warms — isn't penalised vs boot's pre-warm reading.
_CUDA_CONTEXT_ALLOWANCE_BYTES: int = 512 * 2**20  # 0.5 GiB


def _effective_free_bytes() -> int:
    """Free VRAM credited with this process's reclaimable CUDA context — the one
    "is there room for a model" measure used identically by boot and reload.

    Reads ``mem_get_info()[0]`` (CUDA runtime free, NOT nvidia-smi which false-frees
    under WSL2) + ``_CUDA_CONTEXT_ALLOWANCE_BYTES``, capped at the cached ceiling.
    The credit is a no-op at boot (cold context: free ≈ ceiling → min == ceiling)
    and corrects a warm reload's ~0.3 GiB-lower reading (the reloaded model reuses
    the resident context); the cap means a genuine external consumer still fails.
    """
    free_bytes = torch.cuda.mem_get_info(0)[0]
    effective = free_bytes + _CUDA_CONTEXT_ALLOWANCE_BYTES
    ceiling = _state.get("usable_ceiling_bytes")
    if ceiling is not None:
        effective = min(effective, ceiling)
    return effective


def _wait_for_gpu_drain(
    needed_bytes: int,
    *,
    timeout_s: float = _BOOT_GPU_DRAIN_TIMEOUT_S,
    stable_reads: int = _BOOT_GPU_DRAIN_STABLE_READS,
    poll_interval_s: float = _BOOT_GPU_DRAIN_POLL_INTERVAL_S,
) -> bool:
    """Poll until there is room for the model, or degrade.  ONE gate for all paths.

    Uses :func:`_effective_free_bytes` (CUDA-runtime free + reclaimable-context
    credit, capped at the ceiling — NOT nvidia-smi).  Blocks until effective free
    ≥ ``needed_bytes`` for ``stable_reads`` CONSECUTIVE reads, or until
    ``timeout_s`` elapses.

    The consecutive-read requirement filters out transient under/over-reports
    that occur during the host driver's lazy-reclaim window after a predecessor
    process exits: a single passing read may be followed immediately by a
    failing read as reclaim continues.  Three stable passing reads provides
    sufficient signal that reclaim is complete without adding meaningful latency
    on a fully empty device (three reads take ~4.5 s in the worst case).  On a
    live reload the device is already free after the upfront release, so the
    first read passes and the poll returns immediately.

    Args:
        needed_bytes: Minimum effective-free bytes required to begin loading.
            Caller derives this from ``assessment.required_bytes`` (which already
            includes the safety margin from ``assess_topology``).  When the
            assessment was skipped (HF cache miss / hidden_size unavailable),
            caller should pass ``base_pred + headroom`` as a conservative estimate.
        timeout_s: Maximum wall-clock seconds to wait before returning False.
        stable_reads: Number of consecutive reads ≥ ``needed_bytes`` required
            before declaring the GPU ready and returning True.
        poll_interval_s: Seconds to sleep between polls.

    Returns:
        True if there is room (``stable_reads`` consecutive reads satisfied
        ``needed_bytes``), or if CUDA is unavailable (no-op case, always returns
        True to avoid blocking CPU-only environments).
        False if ``timeout_s`` elapsed before the room condition was met.
    """
    if not torch.cuda.is_available():
        return True

    deadline = time.monotonic() + timeout_s
    consecutive = 0
    while True:
        free_bytes = _effective_free_bytes()
        if free_bytes >= needed_bytes:
            consecutive += 1
            if consecutive >= stable_reads:
                logger.info(
                    "GPU room: effective free %.2f GiB ≥ needed %.2f GiB "
                    "(%d consecutive reads) — proceeding with model load",
                    free_bytes / 2**30,
                    needed_bytes / 2**30,
                    consecutive,
                )
                return True
        else:
            consecutive = 0
            logger.debug(
                "GPU room: effective free %.2f GiB < needed %.2f GiB (consecutive=%d) — waiting",
                free_bytes / 2**30,
                needed_bytes / 2**30,
                consecutive,
            )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            logger.warning(
                "GPU room: timed out after %.0f s — effective free %.2f GiB, needed %.2f GiB",
                timeout_s,
                free_bytes / 2**30,
                needed_bytes / 2**30,
            )
            return False
        time.sleep(min(poll_interval_s, remaining))


def _compute_topology_assessment(config, base_pred: int | None):
    """Estimate the VRAM working-set topology for *config*'s model.

    Returns a ``TopologyAssessment``, or ``None`` when it cannot be computed
    (base model not cached, or AutoConfig unreadable) — callers fall back to the
    live load gate.  Shared by lifespan boot and :func:`_live_reload_base_model`
    so the estimate always reflects the model actually being loaded, not a stale
    boot-time value after a base-model swap.

    Args:
        config: ServerConfig.
        base_pred: Output of :func:`predict_base_bytes` for ``config.model_config``,
            read once by the caller (lifespan keeps it as a frame local for the
            drain-wait fallback). ``None`` → no estimate.

    ``local_files_only=True`` on the AutoConfig read: ``base_pred`` being non-None
    means the cache is populated, so the AutoConfig read must hit cache — refusing
    the network matches the offline-first posture and avoids a boot stall on an
    unhealthy network.
    """
    if base_pred is None:
        logger.warning(
            "Base model %s not cached; topology estimate skipped — "
            "live load gate is authoritative.",
            config.model_config.model_id,
        )
        return None
    try:
        import transformers

        hf_cfg = transformers.AutoConfig.from_pretrained(
            config.model_config.model_id,
            trust_remote_code=config.model_config.trust_remote_code,
            local_files_only=True,
        )
        hidden_size = getattr(hf_cfg, "hidden_size", None)
        num_layers = getattr(hf_cfg, "num_hidden_layers", getattr(hf_cfg, "n_layers", None))
    except Exception as _cfg_exc:  # noqa: BLE001 — boundary read; fall back to live gate
        logger.warning(
            "AutoConfig read failed for %s (%s); topology estimate skipped — "
            "live load gate is authoritative.",
            config.model_config.model_id,
            _cfg_exc,
        )
        return None
    if hidden_size is None or num_layers is None:
        logger.warning(
            "AutoConfig for %s did not expose hidden_size/num_hidden_layers; "
            "topology estimate skipped — live load gate is authoritative.",
            config.model_config.model_id,
        )
        return None
    # LoRA tensors inherit the base model's compute_dtype (PEFT + bitsandbytes
    # contract — see paramem/models/loader.py where bnb_4bit_compute_dtype is set
    # from model_config.compute_dtype). torch.element_size() derives the bytes
    # so an fp32 build doesn't silently halve the adapter estimate.
    lora_dtype_bytes = torch.tensor(
        [], dtype=getattr(torch, config.model_config.compute_dtype)
    ).element_size()
    peft_overhead_bytes = config.vram.peft_overhead_per_adapter_mib * 1024 * 1024
    piper_ort_context_bytes = config.vram.tts_piper_ort_context_mib * 1024 * 1024
    assessment = assess_topology(
        config.episodic_adapter_config,
        max_interim_count=config.consolidation.max_interim_count,
        base_bytes=base_pred,
        hidden_size=hidden_size,
        num_layers=num_layers,
        lora_dtype_bytes=lora_dtype_bytes,
        peft_overhead_bytes=peft_overhead_bytes,
        baseline_vram_gib=config.vram.baseline_vram_gib,
        model_id=config.model_config.model_id,
        quant_label=config.model_config.quantization,
        main_adapter_count=sum(
            1
            for adapter_cfg in (
                config.adapters.episodic,
                config.adapters.semantic,
                config.adapters.procedural,
            )
            if adapter_cfg.enabled
        ),
        headroom_gib=config.vram.vram_cache_headroom_gib,
        stt_bytes=estimate_stt_bytes(
            config.stt,
            workspace_factor=config.vram.stt_workspace_factor,
            permanent_cloud_only=False,
        ),
        tts_bytes=estimate_tts_bytes(
            config.tts,
            piper_ort_context_bytes=piper_ort_context_bytes,
            permanent_cloud_only=False,
        ),
    )
    logger.info("VRAM topology assessment:\n%s", assessment.breakdown)
    logger.info("%s", format_baseline_fit(assessment))
    return assessment


# --- Lifespan ---


def _build_user_token_store(config) -> "UserTokenStore | None":
    """Return a :class:`~paramem.server.user_tokens.UserTokenStore` when per-user
    auth is opted in, or ``None`` when it is not.

    The store is constructed only when ``config.mobile_pwa.enabled`` is
    ``True``.  Leaving it ``None`` keeps the middleware in OFF mode for
    default deployments that have neither a shared token nor the PWA enabled —
    restoring the original open-by-default behavior.

    The decision logic is extracted here so it can be unit-tested without
    starting the full app lifespan.

    Parameters
    ----------
    config:
        A :class:`~paramem.server.config.ServerConfig` instance.

    Returns
    -------
    UserTokenStore | None
        A wired store (potentially empty) when ``mobile_pwa.enabled``, else
        ``None``.
    """
    if not config.mobile_pwa.enabled:
        return None
    from paramem.server.user_tokens import UserTokenStore as _UserTokenStore

    return _UserTokenStore(config.paths.data / "user_tokens.json")


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

    # Security startup gate:
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
    _state["daily_loadable"] = _daily_ok

    # Per-user token store — opt-in via mobile_pwa.enabled.  Only constructed
    # when the PWA slice is active so a default deployment (no shared token,
    # mobile_pwa.enabled=false) leaves the store None and the middleware OFF.
    _state["user_token_store"] = _build_user_token_store(config)
    if _state["user_token_store"] is not None:
        logger.info(
            "User token store ready — %d entries",
            len(_state["user_token_store"].list()),
        )

    # VAPID keypair — opt-in via mobile_pwa.push_enabled.  Generated once and
    # persisted as vapid_keys.json (age-encrypted when a daily key is loaded).
    # Loaded after assert_mode_consistency so the write lands in the validated
    # encryption mode.  Skipped when push_enabled=false so default deployments
    # hold zero VAPID state.
    _state["vapid"] = None
    _state["push_store"] = None
    if config.mobile_pwa.enabled and config.mobile_pwa.push_enabled:
        from paramem.server.push import PushSubscriptionStore
        from paramem.server.vapid import application_server_key, ensure_vapid_keypair

        _data_dir = Path(config.paths.data)
        _vapid_handle = ensure_vapid_keypair(_data_dir)
        _state["vapid"] = _vapid_handle
        _state["push_store"] = PushSubscriptionStore(_data_dir / "push_subscriptions.json")
        logger.info(
            "Web Push ready — public key: %s",
            application_server_key(_vapid_handle),
        )

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

        # Sweep .pending/ residue from the snapshot bundle backup directory.
        # This mirrors the per-kind sweep in _validate_main_adapter_slot for
        # adapter dirs; the snapshot/ dir is the new home for bundle slots and
        # must be swept at startup so a crash mid-write doesn't leave residue.
        from paramem.backup.backup import sweep_orphan_pending as _sweep_backup

        _snapshot_backup_dir = backups_root / "snapshot"
        if _snapshot_backup_dir.exists():
            _n_removed = _sweep_backup(_snapshot_backup_dir)
            if _n_removed:
                logger.info(
                    "Swept %d orphaned pending bundle slot(s) from %s",
                    _n_removed,
                    _snapshot_backup_dir,
                )

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
                backup_paths={"config": m.backup_paths.get("config", "")},
                trial_adapter_dir=m.trial_adapter_dir,
                trial_graph_dir=m.trial_graph_dir,
                gates={"status": "pending"},
            )
            _state["migration"]["state"] = "TRIAL"
            _state["migration"]["trial"] = trial_stash
            _state["migration"]["recovery_required"] = []
        elif (
            recovery_result.action == RecoveryAction.RESUME_BASE_SWAP
            and recovery_result.trial_marker is not None
        ):
            # Base-swap resumed: re-enter TRIAL state so the migration lock and
            # rollback endpoint remain functional.  The orchestration coroutine is
            # re-launched below (after the model loads) to resume from the marker's
            # base_swap_phase.  backup_paths holds the bundle slot, not a config slot.
            m_bs = recovery_result.trial_marker
            from paramem.server.migration import TrialStash

            trial_stash_bs: TrialStash = TrialStash(
                started_at=m_bs.started_at,
                pre_trial_config_sha256=m_bs.pre_trial_config_sha256,
                candidate_config_sha256=m_bs.candidate_config_sha256,
                backup_paths=m_bs.backup_paths,
                trial_adapter_dir=m_bs.trial_adapter_dir,
                trial_graph_dir=m_bs.trial_graph_dir,
                gates={"status": "pending"},
            )
            _state["migration"]["state"] = "TRIAL"
            _state["migration"]["trial"] = trial_stash_bs
            _state["migration"]["recovery_required"] = []
            # Stash the marker so the post-startup resume launcher can read it.
            _state["_base_swap_resume_marker"] = m_bs
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

    # Permanent cloud-only: the GPU pair will NEVER be loaded this process
    # lifetime. explicit/gpu_conflict → no auto-reclaim. training (--defer-model)
    # → auto-reclaim will load the GPU pair; reserve the bytes ahead of time.
    permanent_cloud_only = _state.get("cloud_only_reason") in ("explicit", "gpu_conflict")

    # Startup VRAM validation. Hoisted out of the ``if not cloud_only:`` branch
    # so --defer-model startups (cloud_only=True but GPU pair loaded later by
    # auto-reclaim) still reserve the correct budget. The CUDA-availability check
    # and model load below stay inside the branch (eager load is skipped).
    # Pre-load topology estimate. base_pred is None on a cache miss → skip the
    # estimate, rely on the live load gate. Read once here as a lifespan-frame
    # local so the assessment, the drain-wait fallback, and the post-load
    # calibration log share a single HF-cache read.
    base_pred = predict_base_bytes(
        config.model_config,
        nf4_disk_to_runtime_factor=config.vram.nf4_disk_to_runtime_factor,
    )

    if not permanent_cloud_only:
        if not torch.cuda.is_available():
            logger.error(
                "Local model mode requires a CUDA-capable GPU but none was detected. "
                "Either provide a GPU, or start in cloud-only mode."
            )
            sys.exit(1)

        assessment = _compute_topology_assessment(config, base_pred)
        if assessment is not None:
            # Snapshot the device-wide usable ceiling at a quiet moment before any
            # model load — accounts for the WDDM/WSL2 reservation that
            # get_device_properties().total_memory does not. Cached on _state so
            # the live-reload gate (_CUDA_CONTEXT_ALLOWANCE_BYTES) and the
            # _collect_vram_overflow_items attention populator can compare against
            # required_bytes on every /status poll without re-reading the device.
            _state["usable_ceiling_bytes"] = torch.cuda.mem_get_info(0)[0]
            _state["device_total_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
            if assessment.required_bytes > _state["usable_ceiling_bytes"]:
                # Log once at boot; the attention populator surfaces the
                # persistent /status warning from the cached assessment.
                logger.warning(
                    "VRAM CONFIG OVERFLOW — local model config requires "
                    "%.2f GiB working set but only ~%.2f GiB is usable on "
                    "this %.0f GiB GPU (WDDM/WSL2 reserves ~%.2f GiB). "
                    "The server will run cloud-only/degraded. Reduce model "
                    "size, adapter rank/count, or "
                    "consolidation.max_interim_count.",
                    assessment.required_bytes / 2**30,
                    _state["usable_ceiling_bytes"] / 2**30,
                    _state["device_total_memory_bytes"] / 2**30,
                    (_state["device_total_memory_bytes"] - _state["usable_ceiling_bytes"]) / 2**30,
                )

        # Cache the assessment for the GPU reclaim path's live-budget pre-flight;
        # None when skipped above. _live_reload_base_model recomputes it on every
        # live reload, so a base-model swap re-estimates for the new model.
        _state["topology_assessment"] = assessment

    if cloud_only:
        logger.info("Starting in cloud-only mode — skipping model load")
        _state["model"] = None
        _state["tokenizer"] = None
    else:
        # Boot-time drain wait: poll device-wide free VRAM (via CUDA runtime,
        # no nvidia-smi) until there is room for the model, or degrade to
        # cloud-only on timeout. The poll-and-degrade behavior tolerates the
        # host driver's lazy-reclaim window on fast restarts (VRAM may not
        # have been returned yet when the new process boots). The post-load
        # gate (check_post_load_budget) is the authoritative reject — it
        # also degrades to cloud-only (no sys.exit) so a boot-time overflow
        # produces a running cloud-only server rather than a crash.
        #
        # needed_bytes derivation:
        #   • assessment available → use assessment.required_bytes (which
        #     already includes the safety margin from assess_topology).
        #   • assessment was skipped (HF cache miss / AutoConfig failure) →
        #     fall back to base_pred + headroom as a conservative lower bound.
        #     When base_pred is also None (model not cached) → skip the wait
        #     entirely and proceed; the post-load gate is authoritative.
        _needed_bytes: int | None = None
        if assessment is not None:
            _needed_bytes = assessment.required_bytes
        elif base_pred is not None:
            headroom_bytes = int(config.vram.vram_cache_headroom_gib * 2**30)
            _needed_bytes = base_pred + headroom_bytes

        if _needed_bytes is not None:
            drained = _wait_for_gpu_drain(_needed_bytes)
            if not drained:
                logger.warning(
                    "Boot GPU drain: GPU did not free %.2f GiB within %.0f s — "
                    "starting cloud-only; will auto-reclaim when the GPU frees.",
                    _needed_bytes / 2**30,
                    _BOOT_GPU_DRAIN_TIMEOUT_S,
                )
                cloud_only = True
                _state["cloud_only_reason"] = "insufficient_vram"
                _state["model"] = None
                _state["tokenizer"] = None

        if not cloud_only:
            # Model load + adapter mount factored into ``_load_model_into_state``
            # so the model never enters the lifespan async-generator's frame.
            # See that function's docstring for why this is load-bearing.  The
            # per-process VRAM cap is applied as that function's first step
            # (before any tensor allocation) — no separate cap call here.
            _load_model_into_state(config)

    # Config-derived component construction — single shared routine called by
    # BOTH the lifespan (here) and the live-apply path.  At boot the session
    # buffer is always rebuilt (rebuild_session_buffer=True, the default).
    # Note: _apply_config_in_progress is not set here (boot path); the D6 gate
    # inside the routine treats a None/absent store as cold and runs the probe.
    _build_config_derived_state(config, cloud_only=cloud_only)

    # Post-load authoritative gate. Runs AFTER _build_config_derived_state so
    # the measured allocation includes the STT/TTS GPU footprint. On failure,
    # release the partially-loaded GPU pair and continue in cloud-only mode —
    # symmetric with _live_reload_base_model and consistent with the boot
    # drain-wait degrade path. A persistent /status.attention item
    # (vram_post_load_budget) tells the operator exactly what overflowed.
    if _state.get("model") is not None and torch.cuda.is_available():
        actual_bytes = torch.cuda.memory_allocated(0)
        headroom_bytes = int(config.vram.vram_cache_headroom_gib * 2**30)
        total_bytes = torch.cuda.get_device_properties(0).total_memory
        overflow_reason = check_post_load_budget(actual_bytes, total_bytes, headroom_bytes)
        if overflow_reason is not None:
            logger.error(
                "VRAM post-load gate failed — degrading to cloud-only:\n%s",
                overflow_reason,
            )
            _state["post_load_budget_warning"] = {
                "measured_gib": actual_bytes / 2**30,
                "total_gib": total_bytes / 2**30,
                "headroom_gib": headroom_bytes / 2**30,
                "reason": overflow_reason,
            }
            _release_base_model_in_process()
            _state["cloud_only_reason"] = "insufficient_vram"
            _state["model"] = None
            _state["tokenizer"] = None
            cloud_only = True
            notify_server(SERVER_CLOUD_ONLY)
        elif base_pred is not None:
            delta_mib = (actual_bytes - base_pred) / (1024 * 1024)
            logger.info(
                "VRAM calibration drift: predicted %.2f GiB, measured %.2f GiB (delta %+.0f MiB)",
                base_pred / 2**30,
                actual_bytes / 2**30,
                delta_mib,
            )

    # Wyoming listener sockets — bound ONCE here in the lifespan with provider
    # lambdas so profile swaps (cpu⟷gpu) re-point the active pair without
    # re-binding the sockets.  The live-apply path MUST NOT call these again.
    if _state.get("voice_box") is not None or (config.stt.enabled or config.tts.enabled):
        from paramem.server.wyoming_handler import start_wyoming_server, start_wyoming_tts_server

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

        if config.stt.enabled:
            _state["wyoming_server"] = await start_wyoming_server(
                host=config.server.host,
                port=config.stt.port,
                speaker_store=_state.get("speaker_store"),
                embedding_callback=_on_stt_embedding,
                language_callback=_on_stt_language,
                min_embedding_duration_seconds=config.speaker.min_embedding_duration_seconds,
                stt_provider=lambda: _state["voice_box"]["stt"],
            )
            logger.info("Wyoming STT server listening on port %d", config.stt.port)

        if config.tts.enabled:
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
                language_resolver=_resolve_language,
                audio_chunk_bytes=config.tts.audio_chunk_bytes,
                tts_manager_provider=lambda: _state["voice_box"]["tts_manager"],
                language_source=config.tts.language_source,
            )
            logger.info("Wyoming TTS server listening on port %d", config.tts.port)

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
        msg = systemd_timer.reconcile(
            derived_period,
            project_root=str(Path(__file__).resolve().parent.parent.parent),
        )
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
            bg_trainer._shutdown_requested = True
            bg_trainer._is_training = False
            logger.info("Background trainer stopped")
        buffer = _state.get("session_buffer")
        if buffer:
            buffer.save_snapshot()
        # Free all base-model holders before handing control back to the OS so
        # the dying process does not pin the ~5 GiB base model on the device while
        # systemd launches the replacement process.  (STT/TTS, ~1.5-2 GiB, remain
        # until process exit reaps them — the successor's boot drain-wait keys on
        # assessment.required_bytes, which includes the STT/TTS footprint, so it
        # waits for that residual to reclaim too.)  Using _release_base_model_in_process
        # (not bare unload_model) because consolidation_loop / bg_trainer /
        # intent handles keep the model alive through their own references —
        # plain unload_model would not drive the refcount to zero.
        # The try/except is boundary handling (the process is exiting; a teardown
        # error must not hang the exit) — it is NOT error-suppression of a logic bug.
        try:
            _release_base_model_in_process()
            safe_empty_cache()
            logger.info("graceful exit: GPU released")
        except Exception:
            logger.exception("graceful exit: error during GPU release — continuing with exit")
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

    # Active-store migration detection. Triggered when the operator flips
    # consolidation.mode in server.yaml: the on-disk state for the new mode's
    # active store is empty/stale and needs to be rebuilt from the previous
    # mode's store. The check is read-only at startup; the actual migration
    # runs via the consolidation dispatcher (next /consolidate call) under
    # the GPU lock. Inference falls back to ``source_mode`` while a
    # migration is pending so the system stays consistent until ALL tiers
    # have cleared the 1.0 recall gate.  Shared with the live config-reload
    # path (_live_reload_base_model) via _arm_active_store_migration.
    _arm_active_store_migration(config)

    # --- Base-swap resume: re-launch orchestration on crash recovery ---
    # When recovery_result.action == RESUME_BASE_SWAP, the lifespan seeded
    # _state["_base_swap_resume_marker"] above.  Now that the model is loaded
    # and all config-derived state is initialised, launch the orchestration
    # coroutine to resume from wherever the marker left off.
    #
    # Resume semantics by base_swap_phase:
    #   "phaseA":      Phase A was in progress — re-run from the start of
    #                  Phase A (the active-store state file is still on disk).
    #   "phaseA_done": Phase A complete, config already Qwen3 on disk, model
    #                  loaded as Qwen3 at boot.  The orchestration will skip
    #                  Phase A (state file absent / all tiers done) and proceed
    #                  directly to Phase B.  If the reload deferred, the gates
    #                  are set to reload_deferred and the operator re-triggers.
    #   "phaseB":      Resume at Phase B (simulate→train on Qwen3). The
    #                  active-store state file is present on disk; migrate()
    #                  is idempotent on completed tiers.
    # In all cases we pass the original orchestration parameters extracted from
    # the marker.  The config was already renamed in Phase A, so candidate_path_str
    # can be any sentinel (unused when candidate file is gone); live_config_path
    # is the current config path.
    _bs_resume = _state.pop("_base_swap_resume_marker", None)
    if _bs_resume is not None and not cloud_only:
        _bs_live_cfg = (
            Path(_state["config_path"])
            if _state.get("config_path")
            else Path("configs/server.yaml")
        )
        _bs_state_dir = (config.paths.data / "state").resolve()
        _bs_backups_root = (config.paths.data / "backups").resolve()
        asyncio.create_task(
            _run_base_swap_orchestration(
                candidate_path_str=str(_bs_live_cfg),  # config already renamed in Phase A
                live_config_path=_bs_live_cfg,
                state_dir=_bs_state_dir,
                backups_root=_bs_backups_root,
                old_model=_bs_resume.old_model,
                new_model=_bs_resume.new_model,
                started_at=_bs_resume.started_at,
                candidate_hash=_bs_resume.candidate_config_sha256,
                resume_phase=_bs_resume.base_swap_phase,
            )
        )
        logger.info(
            "base-swap resume launched: base_swap_phase=%s old=%s new=%s",
            _bs_resume.base_swap_phase,
            _bs_resume.old_model,
            _bs_resume.new_model,
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
                        _state["memory_store"],
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
                        thermal_policy=ThermalPolicy.from_consolidation_config(
                            config.consolidation
                        ),
                        preload_cache=config.inference.preload_cache,
                    )
                    _state["background_trainer"] = _replay_bt

                # Wire the BG trainer into the consolidation loop so its abort
                # event is included in the training hooks shutdown predicate.
                _state["consolidation_loop"]._bg_trainer = _state["background_trainer"]

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

    # Trim the ~390 MiB of allocator-pool slack accumulated during
    # startup (model + adapter mount + STT/TTS load) — but ONLY when
    # we have a base model loaded. In cloud-only mode no model is
    # loaded, no GPU allocator activity has happened, and there is no
    # CUDA context to interact with. Calling safe_empty_cache here
    # would unconditionally create a CUDA context (via
    # torch.cuda.synchronize / torch._C._cuda_clearCublasWorkspaces),
    # making paramem show up in nvidia-smi compute-apps even when
    # --defer-model'd — which breaks training-control.sh's strict
    # zero-compute-apps cleanup check (scripts/dev/training-control.sh:509)
    # used by tresume's "defer to cloud-only" path.
    if _state.get("model") is not None:
        safe_empty_cache()

    logger.info("ParaMem server ready — mode: %s, model: %s", _state["mode"], config.model_name)

    # PWA static mount — opt-in via config.mobile_pwa.enabled.  Mounted at a
    # sub-path (/app) so it cannot shadow API routes.  Deferred to lifespan so
    # config is available and the mount is skipped in headless/API-only mode.
    if config.mobile_pwa.enabled:
        _pwa_dir = (
            Path(config.mobile_pwa.static_dir)
            if config.mobile_pwa.static_dir
            else Path(__file__).parent.parent / "web" / "static"
        )
        app.mount("/app", StaticFiles(directory=str(_pwa_dir), html=True), name="pwa")
        logger.info("PWA static mount active — serving %s at /app", _pwa_dir)

    # Auth startup posture — logged once here after the store is wired so the
    # message accurately reflects runtime state.  Supersedes the import-time
    # call (which was always AUTH: OFF because the store was not yet assigned).
    # per_user_active is keyed on store presence (matching the middleware
    # enablement rule), so a wired-but-empty store logs ON-per-user (fail-closed)
    # rather than OFF.
    _posture_store = _state.get("user_token_store")
    _n_user_tokens = len(_posture_store.list()) if _posture_store is not None else 0
    log_startup_posture(
        _api_token,
        n_user_tokens=_n_user_tokens,
        per_user_active=_posture_store is not None,
    )

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

    # Final mop-up — covers cuBLAS workspaces / allocator slack the per-component
    # unloads couldn't reach while their frame-locals were still live. Without
    # this, the next paramem boot's _gpu_has_compute_processes() sees a [Not Found]
    # ghost PID and routes into permanent gpu_conflict (no auto-reclaim).
    safe_empty_cache()


app = FastAPI(title="ParaMem", version="0.1.0", lifespan=lifespan)

# Bearer-token auth on all REST endpoints when PARAMEM_API_TOKEN is set.
# No-op when unset (loud WARN emitted from lifespan after store is wired).
from paramem.server.auth import (  # noqa: E402
    BearerTokenMiddleware,
    load_token_from_env,
    log_startup_posture,
)

_api_token = load_token_from_env()
app.add_middleware(
    BearerTokenMiddleware,
    token=_api_token,
    user_token_getter=lambda: _state.get("user_token_store"),
    cookie_name_getter=lambda: (
        _state["config"].mobile_pwa.cookie_name if _state.get("config") else None
    ),
    # "/app" added so a bare /app request reaches the StaticFiles 307→/app/
    # redirect instead of being 401'd before the mount can handle it.
    # "/health" added for unauthenticated liveness polling (e.g. HA binary_sensor).
    exempt_paths=("/", "/app", "/health"),
    exempt_prefixes=("/app/",),
)


# --- Admin-scope gate ---


def require_admin(request: Request) -> None:
    """Admin-scope gate; 403 unless ``request.state.scope == 'admin'``.

    Used as a FastAPI dependency (``dependencies=[Depends(require_admin)]``) on
    every privileged/operational endpoint.  Generalises the former one-off
    ``PARAMEM_API_TOKEN`` check at ``/admin/assign-orphans`` into a single,
    fail-closed guard.

    Accept condition: ``request.state.scope == "admin"`` — default-deny.
    Fail-closed: a chat-scope per-user token has ``scope == "chat"`` and is
    denied.  In auth-OFF mode (no shared token AND no user-token store) the
    server is open by design — ``BearerTokenMiddleware`` stamps
    ``scope == "admin"`` on every pass-through request (see
    ``auth.py`` OFF branch), so admin endpoints stay reachable, preserving the
    prior open-in-OFF behaviour.  The gate therefore enforces only in ON mode
    (shared token set OR user-token store wired), where a chat-scope token 403s.

    Raising ``HTTPException`` is FastAPI-idiomatic boundary rejection — NOT a
    suppressing try/except.
    """
    if getattr(request.state, "scope", None) != "admin":
        raise HTTPException(
            status_code=403,
            detail={
                "status": "admin_scope_required",
                "detail": "This endpoint requires an admin-scope token.",
            },
        )


# --- Endpoints ---


# Default PWA static directory — matches the lifespan resolution logic so the
# route handler uses a consistent path when ``config.mobile_pwa.static_dir`` is
# unset (the common case).
_PWA_STATIC_DEFAULT = Path(__file__).parent.parent / "web" / "static"


@app.get("/app/sw.js")
async def serve_sw_js():
    """Serve the PWA service-worker script with ``Cache-Control: no-cache``.

    Standard ``StaticFiles`` sets ``ETag`` / ``Last-Modified`` but omits
    ``Cache-Control``, so browsers apply heuristic caching to ``sw.js`` and
    skip the service-worker update check after a ``CACHE_VERSION`` bump.

    This dedicated route overrides the ``/app`` ``StaticFiles`` mount (which is
    registered later in the lifespan) by being added to the router first, so
    Starlette resolves ``/app/sw.js`` here rather than via the mount.  The
    ``Cache-Control: no-cache`` header forces the browser to revalidate on
    every navigation, ensuring a ``CACHE_VERSION`` bump propagates promptly.

    The path is under the ``/app/`` exempt prefix (see
    ``BearerTokenMiddleware`` configuration) and requires no bearer token —
    the service-worker script must be fetchable before the user completes
    onboarding.
    """
    config = _state.get("config")
    if config is not None and config.mobile_pwa.static_dir:
        pwa_dir = Path(config.mobile_pwa.static_dir)
    else:
        pwa_dir = _PWA_STATIC_DEFAULT

    sw_path = pwa_dir / "sw.js"
    if not sw_path.is_file():
        return JSONResponse(status_code=404, content={"error": "not_found"})

    return FileResponse(
        path=str(sw_path),
        media_type="application/javascript",
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/health")
async def health():
    """Unauthenticated liveness probe.

    Returns ``{"status": "ok"}`` with HTTP 200.  Exempt from bearer-token
    auth (see ``exempt_paths`` in the middleware wiring) so external pollers
    (e.g. a Home Assistant ``binary_sensor`` platform: rest) can reach it
    without a token.  Does not touch ``_state`` and has no dependency on
    model or GPU availability.
    """
    return {"status": "ok"}


@app.get("/")
async def root_redirect():
    """Redirect the bare root to the PWA shell.

    The ``/`` path is exempt from bearer-token auth so the browser can
    follow the redirect before a token is presented.  The ``/app/``
    prefix is also exempt, so the shell and its static assets load freely.
    """
    return RedirectResponse("/app/")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, http_request: Request):
    """Handle a conversation turn with speaker identification.

    *http_request* is injected by FastAPI for access to per-request state
    (e.g. ``speaker_id`` set by BearerTokenMiddleware on authenticated
    per-user requests).  It is NOT parsed as a JSON body.
    """
    _state["last_chat_time"] = datetime.now(timezone.utc)
    # Monotonic stamp for debounce — immune to NTP wall-clock steps.
    # Both writes happen on the asyncio event loop thread (cooperative
    # scheduling), so no lock is needed.
    _state["last_chat_monotonic"] = time.monotonic()
    buffer = _state["session_buffer"]

    # Authenticated speaker from per-user bearer token (set by
    # BearerTokenMiddleware on ON-per-user requests).  None for legacy shared
    # token calls (no speaker attribution) and unauthenticated mode.
    auth_speaker_id: str | None = getattr(http_request.state, "speaker_id", None)

    # Forced routing — bypass normal routing for direct provider testing.
    # Supports: "ha", "sota", "sota:anthropic", "sota:openai", "sota:google"
    if request.route and request.route.startswith(("ha", "sota")):
        _speaker_id, speaker = _resolve_speaker(
            request, buffer, _state.get("speaker_store"), auth_speaker_id=auth_speaker_id
        )
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

    # Text-only path fallback: STT didn't fire and the request carries no
    # voice embedding, so Whisper produced nothing. Run the offline fastText
    # detector against the request text so cloud routing can speak the
    # language the user wrote in instead of defaulting to English.
    text_lang_cfg = _state["config"].text_lang_detection
    if detected_language is None and not request.speaker_embedding:
        from paramem.server import lang_id

        text_lang, text_prob = lang_id.resolve_text_language(request.text, text_lang_cfg)
        if text_lang:
            detected_language = text_lang
            detected_language_prob = text_prob
            logger.info("Text-side lang_id: %s (prob=%.2f)", text_lang, text_prob)

    _resolved = await _resolve_and_enroll_speaker(
        request=request,
        auth_speaker_id=auth_speaker_id,
        buffer=buffer,
        store=_state.get("speaker_store"),
        detected_language=detected_language,
        detected_language_prob=detected_language_prob,
    )
    speaker_id, speaker, display_speaker = (
        _resolved.speaker_id,
        _resolved.speaker,
        _resolved.display_speaker,
    )
    follow_up, greeting_prefix, detected_language = (
        _resolved.follow_up,
        _resolved.greeting_prefix,
        _resolved.effective_language,
    )

    result, spoken_text = await _run_chat_turn(
        text=request.text,
        conversation_id=request.conversation_id,
        speaker_id=speaker_id,
        speaker=speaker,
        display_speaker=display_speaker,
        history=request.history,
        speaker_embedding=request.speaker_embedding,
        language=detected_language,
        greeting_prefix=greeting_prefix,
    )
    return ChatResponse(
        text=spoken_text,
        escalated=result.escalated,
        speaker=speaker,
        follow_up=follow_up,
    )


@dataclass(frozen=True)
class ResolvedSpeaker:
    """Speaker resolution result shared by POST /chat and POST /voice.

    Attributes
    ----------
    speaker_id:
        Canonical speaker identifier (e.g. ``"Speaker0"``), or ``None``
        when resolution failed or the caller is fully anonymous.
    speaker:
        Display name returned by the speaker store (used in ChatResponse).
    display_speaker:
        Anonymization-safe name for system-prompt injection and greeting.
        ``None`` when the speaker is anonymous (``is_anonymous`` is true)
        so the robotic ``Speaker{N}`` label is suppressed until disclosure.
    follow_up:
        Server-initiated follow-up prompt (e.g. "What's your name?") when
        the voice is unknown.  ``None`` after successful disclosure or for
        the token-authoritative path.
    greeting_prefix:
        Time-gated greeting string (e.g. ``"Good morning, Alice. "``), or
        ``None`` when no greeting is due.
    effective_language:
        Resolved language code for this request (Whisper → stored preference
        → ``None``).  Rebinds ``detected_language`` in the caller.
    """

    speaker_id: str | None
    speaker: str | None
    display_speaker: str | None
    follow_up: str | None
    greeting_prefix: str | None
    effective_language: str | None


async def _resolve_and_enroll_speaker(
    *,
    request: "ChatRequest",
    auth_speaker_id: str | None,
    buffer,
    store,
    detected_language: str | None,
    detected_language_prob: float,
) -> ResolvedSpeaker:
    """Resolve speaker identity and run deferred enrollment.

    Extracted verbatim from the POST /chat inline block so POST /voice can
    share the same enrollment/greeting/language-resolution path when the
    device carries a shared token (``auth_speaker_id is None``).

    Parameters
    ----------
    request:
        Incoming :class:`ChatRequest`.  ``request.speaker_embedding`` is
        used for embedding-based resolution and enrollment when present.
    auth_speaker_id:
        Speaker ID attached by :class:`~paramem.server.auth.BearerTokenMiddleware`
        for per-user tokens.  ``None`` on the shared-token path.
    buffer:
        Active :class:`~paramem.server.session_buffer.SessionBuffer`.
    store:
        :class:`~paramem.server.speaker.SpeakerStore` instance, or ``None``.
    detected_language:
        Language code from STT or text-side detector (may be ``None``).
    detected_language_prob:
        Confidence score for *detected_language* (0.0 when unknown).

    Returns
    -------
    ResolvedSpeaker
        All six resolution outputs; callers destructure as needed.
    """
    # Speaker resolution: auth token → embedding → session history → anonymous.
    # Never let speaker ID failure kill the request — proceed as anonymous.
    try:
        speaker_id, speaker = _resolve_speaker(
            request, buffer, store, auth_speaker_id=auth_speaker_id
        )
    except Exception:
        logger.exception("Speaker resolution failed — proceeding as anonymous")
        speaker_id, speaker = None, None
    follow_up = None

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
        greeting = store.should_greet(
            speaker_id,
            greeting_interval,
            _state["config"].voice.greetings,
            language=effective_language or "en",
        )
        if greeting:
            if display_speaker:
                greeting_prefix = f"{greeting}, {display_speaker}. "
            else:
                greeting_prefix = f"{greeting}. "
            store.confirm_greeting(speaker_id)

    return ResolvedSpeaker(
        speaker_id=speaker_id,
        speaker=speaker,
        display_speaker=display_speaker,
        follow_up=follow_up,
        greeting_prefix=greeting_prefix,
        effective_language=effective_language,
    )


async def _run_chat_turn(
    *,
    text: str,
    conversation_id: str,
    speaker_id: str | None,
    speaker: str | None,
    display_speaker: str | None,
    history: list[dict] | None,
    speaker_embedding: list[float] | None,
    language: str | None,
    greeting_prefix: str | None,
) -> tuple[ChatResult, str]:
    """Execute a single conversation turn (shared by POST /chat and POST /voice).

    Encapsulates the post-speaker-resolution orchestration that is identical
    for text and voice turns: the scheduler debounce stamps, training-abort,
    cloud-only vs local routing, session buffer appends, post-session training
    enqueue, and greeting prefix application.

    Both ``/chat`` and ``/voice`` callers are responsible for resolving
    *display_speaker* (anonymization check) and *greeting_prefix* before
    calling this function.

    Parameters
    ----------
    text:
        The user's message text (already transcribed for voice turns).
    conversation_id:
        Conversation / session identifier.
    speaker_id:
        Resolved canonical speaker ID, or ``None`` for anonymous turns.
    speaker:
        Display name of the speaker (resolved by ``_resolve_speaker``), or
        ``None``.
    display_speaker:
        User-facing salutation: ``None`` when the speaker is anonymous (suppress
        the canonical ``Speaker{N}`` token); otherwise same as *speaker*.
    history:
        Prior conversation turns to pass to ``handle_chat``, or ``None``.
    speaker_embedding:
        Float embedding to attach to the user buffer entry, or ``None``.
    language:
        Resolved BCP-47 language code for this turn, or ``None``.
    greeting_prefix:
        Greeting string to prepend to the assistant reply (e.g. ``"Good
        morning, Alice. "``), or ``None``.

    Returns
    -------
    tuple[ChatResult, str]
        ``(result, spoken_text)`` where *result* is the raw
        :class:`~paramem.server.inference.ChatResult` and *spoken_text* is
        ``result.text`` with *greeting_prefix* prepended when applicable.
    """
    buffer = _state["session_buffer"]

    # Debounce stamps — monotonic for scheduler, wall-clock for /status display.
    # Both writes run on the asyncio event-loop thread (cooperative scheduling),
    # so no lock is needed.
    _state["last_chat_time"] = datetime.now(timezone.utc)
    _state["last_chat_monotonic"] = time.monotonic()

    # Cloud-only mode — route via HA graph + SOTA, no local model.
    if _state["mode"] == "cloud-only":
        result = _cloud_only_route(
            text=text,
            speaker=display_speaker,
            history=history,
            config=_state["config"],
            router=_state.get("router"),
            ha_client=_state.get("ha_client"),
            sota_agent=_state.get("sota_agent"),
            language=language,
        )
        buffer.append(
            conversation_id,
            "user",
            text,
            embedding=speaker_embedding,
            speaker_id=speaker_id,
            speaker=speaker,
        )
        cloud_text = result.text
        buffer.append(
            conversation_id,
            "assistant",
            cloud_text,
            speaker_id=speaker_id,
            speaker=speaker,
        )
        spoken_text = f"{greeting_prefix}{cloud_text}" if greeting_prefix else cloud_text
        return result, spoken_text

    # Local mode — normal inference with entity routing.
    # Abort background training if active, then acquire the GPU lock.
    # abort_for_inference() sets the per-job abort flag and waits up to 30 s
    # for training to stop at the next step boundary and release the GPU lock.
    # _active_quiesced is set OUTSIDE gpu_lock_sync so the caller's
    # async with gpu_lock() below succeeds without lock contention.
    bg_trainer = _state.get("background_trainer")
    if bg_trainer is not None and bg_trainer.is_training:
        _abort_timeout = _state["config"].consolidation.abort_quiesce_timeout_s
        aborted = bg_trainer.abort_for_inference(timeout=_abort_timeout)
        if not aborted:
            logger.warning(
                "Training did not abort within %.1f s — forcing trainer stop before inference",
                _abort_timeout,
            )
            bg_trainer._shutdown_requested = True
            bg_trainer._is_training = False

    from paramem.server.gpu_lock import gpu_lock

    async with gpu_lock():
        loop = asyncio.get_running_loop()
        result: ChatResult = await loop.run_in_executor(
            None,
            lambda: handle_chat(
                text=text,
                conversation_id=conversation_id,
                speaker=display_speaker,
                speaker_id=speaker_id,
                history=history,
                model=_state["model"],
                tokenizer=_state["tokenizer"],
                config=_state["config"],
                router=_state["router"],
                sota_agent=_state.get("sota_agent"),
                ha_client=_state.get("ha_client"),
                language=language,
                # Active-store migration override: when a mode-switch is in
                # progress or interrupted, the inference path falls back to
                # the source mode's store. None == use config.consolidation.mode.
                effective_mode=_state.get("effective_mode"),
                memory_store=_state["memory_store"],
            ),
        )

    buffer.append(
        conversation_id,
        "user",
        text,
        embedding=speaker_embedding,
        speaker_id=speaker_id,
        speaker=speaker,
    )
    response_text = result.text
    buffer.append(
        conversation_id,
        "assistant",
        response_text,
        speaker_id=speaker_id,
        speaker=speaker,
    )

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
        transcript_turns = buffer.get_session_turns(conversation_id)
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
                        "session_id": conversation_id,
                        "transcript": transcript_text,
                        "speaker_id": speaker_id,
                        "speaker_name": speaker,
                    }
                )
            enqueue_post_session_train(
                conversation_id=conversation_id,
                transcript=transcript_text,
                speaker_id=speaker_id,
                speaker_name=speaker,
                loop=_state["consolidation_loop"],
                background_trainer=_state.get("background_trainer"),
                config=_state["config"],
                state=_state,
                post_session_queue=_psq,
            )

    # Training was aborted (not paused) — the next job submission will resume
    # from the checkpoint automatically via the resume_state.json path.

    spoken_text = f"{greeting_prefix}{response_text}" if greeting_prefix else response_text
    return result, spoken_text


# ---------------------------------------------------------------------------
# POST /voice  — mobile-PWA voice endpoint
# ---------------------------------------------------------------------------


class VoiceResponse(BaseModel):
    """Response body for POST /voice.

    Attributes
    ----------
    transcript:
        The transcribed text.  Empty string when the audio was silent or
        STT produced no output.
    reply:
        The assistant's reply text.  Empty string when the transcript was
        empty (nothing to reply to).
    audio:
        Base64-encoded WAV (RIFF/PCM int16 mono) of the synthesized reply.
        Empty string when TTS is unavailable or synthesis failed — callers
        must treat this as optional and fall back to text-only rendering.
    audio_format:
        Container format of ``audio``; always ``"wav"`` when ``audio`` is
        non-empty, empty string otherwise.
    follow_up:
        Server-initiated follow-up prompt sent after the reply (e.g. an
        enrollment "What's your name?" message on the shared-token path).
        ``None`` when no follow-up is needed.
    """

    transcript: str
    reply: str
    audio: str = ""
    audio_format: str = ""
    follow_up: str | None = None


def _build_wav_bytes(pcm: bytes, sample_rate: int) -> bytes:
    """Wrap raw int16 mono PCM in a minimal RIFF/WAV container.

    Prepends the standard 44-byte WAV header so browsers and audio players
    can decode the result directly without an external decoder.

    Parameters
    ----------
    pcm:
        Raw 16-bit signed integer PCM samples, mono, little-endian.
    sample_rate:
        Sample rate in Hz (e.g. 22050, 24000).

    Returns
    -------
    bytes
        44-byte RIFF header + ``pcm``.
    """
    import struct

    num_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    data_size = len(pcm)
    chunk_size = 36 + data_size  # RIFF chunk body = header tail (36) + data

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        chunk_size,
        b"WAVE",
        b"fmt ",
        16,  # PCM sub-chunk size
        1,  # PCM format tag
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )
    return header + pcm


def _decode_audio_to_pcm(audio_bytes: bytes, content_type: str) -> bytes:
    """Decode a container audio blob to 16 kHz int16 mono PCM via ffmpeg.

    Supports ``audio/mp4``, ``audio/webm``, ``audio/webm;codecs=opus``,
    ``audio/ogg``, and any format ffmpeg can sniff from the byte stream.
    Raw PCM (``audio/L16``) is returned as-is.

    Parameters
    ----------
    audio_bytes:
        Raw bytes from the HTTP request body.
    content_type:
        ``Content-Type`` header value.  Used to select a fast passthrough
        path for already-canonical PCM and to write the temporary file with
        the right extension so ffmpeg can apply a demuxer hint.

    Returns
    -------
    bytes
        16 kHz int16 mono PCM.

    Raises
    ------
    subprocess.CalledProcessError
        When ffmpeg exits non-zero.
    RuntimeError
        When ffmpeg produces empty output (silent or unreadable input), or
        when ffmpeg does not complete within 30 seconds.
    """
    import os
    import subprocess
    import tempfile

    # Fast path: raw 16 kHz int16 PCM — no decoding needed.
    base_ct = content_type.split(";")[0].strip().lower()
    if base_ct in ("audio/l16", "audio/pcm", "audio/x-raw"):
        return audio_bytes

    # Determine file extension for the temporary file so ffmpeg gets a
    # demuxer hint for formats that require it (e.g. WebM, MP4).
    _CT_EXT = {
        "audio/mp4": ".mp4",
        "audio/m4a": ".m4a",
        "audio/webm": ".webm",
        "audio/ogg": ".ogg",
        "audio/mpeg": ".mp3",
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
    }
    ext = _CT_EXT.get(base_ct, ".audio")

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        cmd = [
            "/usr/bin/ffmpeg",
            "-y",
            "-i",
            tmp_path,
            "-ar",
            "16000",
            "-ac",
            "1",
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-",
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, check=True, timeout=30)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"ffmpeg timed out after 30 s for content-type={content_type!r}"
            ) from exc
        pcm = result.stdout
        if not pcm:
            raise RuntimeError(
                f"ffmpeg produced empty output for content-type={content_type!r}. "
                f"stderr: {result.stderr.decode(errors='replace')}"
            )
        return pcm
    finally:
        os.unlink(tmp_path)


_VOICE_BODY_MAX_BYTES = 25 * 1024 * 1024  # 25 MB — generous for push-to-talk clips


@app.post("/voice", response_model=VoiceResponse)
async def voice(http_request: Request):
    """Handle a voice utterance from the mobile PWA.

    Accepts a raw audio blob (``audio/mp4``, ``audio/webm``, ``audio/L16``…),
    decodes it to 16 kHz int16 mono PCM, runs Whisper STT, then routes the
    transcript through :func:`_run_chat_turn` — the same shared orchestration
    as ``POST /chat``.

    **Token-type selector:**

    - **Per-user token** (``request.state.speaker_id`` set by
      :class:`~paramem.server.auth.BearerTokenMiddleware`): identity is
      authoritative from the token.  Voice embedding is NOT computed (cheap
      path).  A stable per-conversation-id from the ``x-conversation-id``
      header (or ``"voice-default"``) is used.

    - **Shared token / no attributed identity** (``auth_speaker_id is None``):
      the voice embedding IS computed and passed through
      :func:`_resolve_and_enroll_speaker` — the same enrollment/greeting/
      name-disclosure path as ``POST /chat``.  A fresh per-utterance
      ``conversation_id`` is generated on each request (one POST = one
      push-to-talk press).  The session-buffer retro-claim propagates identity
      across conversation_ids by embedding so two-turn enrollment still works.

    Returns ``{"transcript": "", "reply": ""}`` when the audio is silent or
    the STT returned no text.

    Returns HTTP 404 when ``mobile_pwa.enabled`` is ``False``.
    Returns HTTP 413 when the request body exceeds :data:`_VOICE_BODY_MAX_BYTES`.
    Returns HTTP 503 when the STT model is not loaded (cloud-only mode).
    """
    config = _state.get("config")
    if config is None or not config.mobile_pwa.enabled:
        return JSONResponse(status_code=404, content={"error": "not_found"})

    stt = _state.get("stt")
    if stt is None or not stt.is_loaded:
        return JSONResponse(status_code=503, content={"error": "stt_unavailable"})

    # Reject oversized uploads before reading the body — push-to-talk clips are
    # tiny; 25 MB is generous.  Check Content-Length first (O(1)); fall back to
    # a bounded read when the header is absent.
    content_length_str = http_request.headers.get("content-length")
    if content_length_str is not None:
        try:
            if int(content_length_str) > _VOICE_BODY_MAX_BYTES:
                return JSONResponse(status_code=413, content={"error": "audio_too_large"})
        except ValueError:
            pass  # malformed header — let the bounded read handle it

    audio_bytes = await http_request.body()
    if len(audio_bytes) > _VOICE_BODY_MAX_BYTES:
        return JSONResponse(status_code=413, content={"error": "audio_too_large"})

    if not audio_bytes:
        return VoiceResponse(transcript="", reply="")

    content_type = http_request.headers.get("content-type", "audio/webm")

    # Decode the container audio to raw PCM (boundary — bad audio = clear error).
    try:
        pcm_bytes = await asyncio.get_running_loop().run_in_executor(
            None, _decode_audio_to_pcm, audio_bytes, content_type
        )
    except Exception:
        logger.warning("POST /voice: audio decode failed", exc_info=True)
        return JSONResponse(
            status_code=400,
            content={"error": "audio_decode_failed"},
        )

    # Resolve speaker from bearer token before transcription: when the device
    # carries a per-user token, identity is authoritative and no embedding is
    # needed (cheap path).  On the shared-token path (auth_speaker_id is None)
    # the voice embedding is required for identification and enrollment.
    auth_speaker_id: str | None = getattr(http_request.state, "speaker_id", None)

    # Compute embedding only on the shared-token path; skip on per-user token
    # (saves CPU cost and avoids unnecessary WeSpeaker inference).
    compute_embedding = auth_speaker_id is None

    # Transcribe and optionally compute voice embedding.
    utterance = await process_utterance(
        pcm_bytes,
        16000,
        2,
        1,
        stt=stt,
        compute_embedding=compute_embedding,
    )

    text = utterance.text
    if not text:
        return VoiceResponse(transcript="", reply="")

    # Per-utterance conversation_id on the shared-token path: each push-to-talk
    # press is an independent POST /voice call, so a fresh id ensures the session
    # buffer does not conflate turns from different speakers on the same device.
    # The retro-claim in session_buffer.claim_sessions_for_speaker works across
    # conversation_ids by embedding, so two-turn enrollment still works.
    # On the per-user path, a stable id allows multi-turn context.
    if auth_speaker_id is None:
        conversation_id = f"voice-{uuid.uuid4().hex[:12]}"
    else:
        conversation_id = http_request.headers.get("x-conversation-id", "voice-default")

    buffer = _state["session_buffer"]
    chat_req = ChatRequest(
        text=text,
        conversation_id=conversation_id,
        speaker_embedding=utterance.embedding,
    )

    # Shared resolution + enrollment seam: runs the same enrollment/greeting/
    # language-resolution logic as POST /chat (verbatim body).
    _resolved = await _resolve_and_enroll_speaker(
        request=chat_req,
        auth_speaker_id=auth_speaker_id,
        buffer=buffer,
        store=_state.get("speaker_store"),
        detected_language=utterance.language,
        detected_language_prob=utterance.language_probability,
    )

    # Route through the shared turn orchestrator.  Use the resolved effective
    # language (Whisper → stored preference → None) for routing and TTS.
    result, spoken_text = await _run_chat_turn(
        text=text,
        conversation_id=conversation_id,
        speaker_id=_resolved.speaker_id,
        speaker=_resolved.speaker,
        display_speaker=_resolved.display_speaker,
        history=None,
        speaker_embedding=utterance.embedding,
        language=_resolved.effective_language,
        greeting_prefix=_resolved.greeting_prefix,
    )

    # Synthesize the reply to speech if TTS is available.
    # Use the resolved effective language for consistent voice selection.
    # Boundary: any synthesis failure is non-fatal — the response falls back to
    # text-only with audio="" so the PWA still renders the reply.
    audio_b64 = ""
    audio_fmt = ""
    tts_manager = _state.get("tts_manager")
    if tts_manager is not None and tts_manager.is_loaded and spoken_text:
        import base64

        try:
            loop = asyncio.get_running_loop()
            pcm_bytes, sample_rate = await loop.run_in_executor(
                None, tts_manager.synthesize, spoken_text, _resolved.effective_language
            )
            if pcm_bytes:
                wav_bytes = _build_wav_bytes(pcm_bytes, sample_rate)
                audio_b64 = base64.b64encode(wav_bytes).decode("ascii")
                audio_fmt = "wav"
        except Exception:
            logger.warning("POST /voice: TTS synthesis failed — returning text-only", exc_info=True)

    return VoiceResponse(
        transcript=text,
        reply=spoken_text,
        audio=audio_b64,
        audio_format=audio_fmt,
        follow_up=_resolved.follow_up,
    )


# ---------------------------------------------------------------------------
# Web Push endpoints  (mobile_pwa.push_enabled=true)
# ---------------------------------------------------------------------------


class PushSubscribeRequest(BaseModel):
    """Request body for POST /push/subscribe.

    Fields mirror the browser ``PushSubscription.toJSON()`` shape:
    ``endpoint`` is the push relay URL; ``keys`` holds the ECDH public key
    (``p256dh``) and the authentication secret (``auth``) in unpadded
    base64url encoding.

    No ``speaker_id`` field is accepted — identity is taken exclusively from
    the per-user bearer token (``request.state.speaker_id``).  Any
    ``speaker_id`` key in the request body is silently discarded by Pydantic's
    ``extra="ignore"`` model default.
    """

    model_config = {"extra": "ignore"}

    endpoint: str
    keys: dict


@app.get("/push/vapid-public-key")
async def push_vapid_public_key():
    """Return the VAPID application server public key.

    Used by the PWA to call ``PushManager.subscribe({applicationServerKey})``.
    The key is the unpadded base64url-encoded uncompressed EC P-256 point
    (65 bytes, ``0x04`` prefix).

    Returns
    -------
    JSON
        ``{"key": "<base64url>"}`` on success.
    HTTP 503
        When ``push_enabled`` is false or the VAPID handle is not initialised.
    """
    config = _state.get("config")
    vapid = _state.get("vapid")

    if config is None or not config.mobile_pwa.push_enabled or vapid is None:
        return JSONResponse(
            status_code=503,
            content={"error": "push_not_enabled"},
        )

    from paramem.server.vapid import application_server_key

    return {"key": application_server_key(vapid)}


@app.post("/push/subscribe")
async def push_subscribe(body: PushSubscribeRequest, http_request: Request):
    """Register a push subscription for the authenticated speaker.

    The subscription is persisted under the speaker_id bound to the
    per-user bearer token (set by
    :class:`~paramem.server.auth.BearerTokenMiddleware`).  Shared-token or
    unauthenticated requests are rejected with HTTP 403.  The endpoint is
    deduplicated per speaker — re-subscribing the same endpoint is a no-op.

    Request body (``application/json``) must be the browser
    ``PushSubscription.toJSON()`` shape::

        {
            "endpoint": "https://web.push.apple.com/...",
            "keys": {"p256dh": "...", "auth": "..."}
        }

    Returns
    -------
    JSON
        ``{"status": "subscribed"}`` on success (new or duplicate).
    HTTP 403
        When no per-user speaker_id is attached to the request (shared token
        or unauthenticated).
    HTTP 503
        When ``push_enabled`` is false or the push store is not initialised.
    """
    config = _state.get("config")
    push_store = _state.get("push_store")

    if config is None or not config.mobile_pwa.push_enabled or push_store is None:
        return JSONResponse(
            status_code=503,
            content={"error": "push_not_enabled"},
        )

    # Require a per-user token — shared tokens do not bind to a speaker_id.
    auth_speaker_id: str | None = getattr(http_request.state, "speaker_id", None)
    if auth_speaker_id is None:
        return JSONResponse(
            status_code=403,
            content={"error": "per_user_token_required"},
        )

    subscription = {"endpoint": body.endpoint, "keys": body.keys}
    try:
        push_store.add(auth_speaker_id, subscription)
    except ValueError as exc:
        return JSONResponse(
            status_code=400,
            content={"status": "invalid_subscription", "detail": str(exc)},
        )
    return {"status": "subscribed"}


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


def _resolve_speaker(
    request: ChatRequest,
    buffer,
    speaker_store,
    auth_speaker_id: str | None = None,
) -> tuple[str | None, str | None]:
    """Resolve speaker identity from multiple sources.

    Returns (speaker_id, speaker_name) tuple.

    Priority:
    0. Authenticated token identity — authoritative (cryptographic).  When
       *auth_speaker_id* is set and is known to the speaker store (or when the
       store is absent/unavailable), this identity is returned immediately
       without consulting voice embeddings.  A per-user bearer token is a
       stronger signal than a probabilistic voice match; overriding it would
       allow a voice impersonation to bypass token-based auth.
    1. Voice embedding match (via SpeakerStore, high confidence only)
    2. Session history (previously identified in this conversation)
    3. Anonymous (None, None)

    Parameters
    ----------
    request:
        The incoming chat request (provides speaker_embedding, conversation_id).
    buffer:
        Active SessionBuffer for session-level speaker tracking.
    speaker_store:
        Optional SpeakerStore for voice-embedding based identification.
    auth_speaker_id:
        Speaker ID from the bearer token, attached by BearerTokenMiddleware.
        When set, this identity is authoritative and returned before the
        voice/session resolution path.
    """
    # 0. Authenticated token identity — authoritative.
    if auth_speaker_id is not None:
        speaker_name: str | None = None
        if speaker_store is not None:
            speaker_name = speaker_store.get_name(auth_speaker_id)
        # Record on session state for multi-turn continuity (priority 2 below)
        # and any session-state reader, mirroring the voice branch. Turn
        # attribution itself no longer depends on this — append() now takes the
        # resolved speaker_id explicitly — but keeping the two representations
        # consistent avoids a stale session-state read on later turns.
        buffer.set_speaker(request.conversation_id, auth_speaker_id, speaker_name or "")
        return auth_speaker_id, speaker_name

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

    # Active key count comes from the one authoritative MemoryStore — the same
    # store /debug/dump and the recall path read.  It is loaded from disk (main
    # + interim tiers) at boot via load_registries_from_disk.  When no live
    # store is constructed (e.g. cloud-only before preload), count from disk
    # using the store's own loader rather than re-inlining a registry scan.
    store = _state.get("memory_store")
    if store is not None and store.replay_enabled:
        keys_count = len(store.all_active_keys())
    else:
        from paramem.memory.store import MemoryStore as _MemoryStore

        _cold = _MemoryStore(replay_enabled=True)
        _cold.load_registries_from_disk(config.adapter_dir)
        keys_count = len(_cold.all_active_keys())

    # _live_loop is used by the adapter_health block below.
    _live_loop = _state.get("consolidation_loop")

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
    from paramem.memory.interim_adapter import compute_schedule_period_seconds

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
    bg_adapter = None  # surfaced in /status; resolved by the active training caller now

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

    # Adapter health — stored in per-tier KeyRegistry JSON files:
    # <adapter_dir>/<tier>/indexed_key_registry.json → field "health".
    # Prefer reading from the in-memory ConsolidationLoop when available.
    # Surfaced to pstatus so a degenerated adapter is visible without grepping logs.
    adapter_health: dict = {}
    if _live_loop is not None and _live_loop.store.replay_enabled:
        for _tier_name in _live_loop.store.tiers_with_registry():
            _h = _live_loop.store.registry(_tier_name).get_health()
            if _h is not None:
                adapter_health[_tier_name] = _h
    else:
        from paramem.training.key_registry import KeyRegistry as _KeyRegHealth

        try:
            for _tier in ("episodic", "semantic", "procedural"):
                _reg_path = config.adapter_dir / _tier / "indexed_key_registry.json"
                if _reg_path.exists():
                    _reg = _KeyRegHealth.load(_reg_path)
                    _h = _reg.get_health()
                    if _h is not None:
                        adapter_health[_tier] = _h
            from paramem.memory.interim_adapter import iter_interim_dirs as _iter_int_h

            for _interim_name, _interim_d in _iter_int_h(config.adapter_dir):
                _reg_path = _interim_d / "indexed_key_registry.json"
                if _reg_path.exists():
                    _reg = _KeyRegHealth.load(_reg_path)
                    _h = _reg.get_health()
                    if _h is not None:
                        adapter_health[_interim_name] = _h
        except Exception:
            adapter_health = {}

    # TTS inventory: which languages are loaded and on which device. When
    # voices span devices (one on CUDA, one on CPU) we report "mixed" so the
    # fallback path is visible in pstatus without dumping per-voice rows.
    tts_manager = _state.get("tts_manager")
    tts_loaded = bool(tts_manager and tts_manager.is_loaded)
    tts_languages: list[str] = tts_manager.available_languages if tts_loaded else []
    tts_degraded = bool(tts_loaded and set(tts_languages) != set(tts_manager.configured_languages))
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
    # base_swap_phase: name the in-flight base-swap phase (phaseA/phaseA_done/phaseB)
    # so operators can interpret a TRIAL as a base-model swap and see how far it got.
    # Read from the on-disk marker only while a trial is active; a corrupt marker
    # must not 500 /status (boundary read of a display-only field).
    _base_swap_phase = None
    if _mig_state == "trial":
        try:
            _sm = read_trial_marker((config.paths.data / "state").resolve())
            if _sm is not None and _sm.migration_kind == "base_swap":
                _base_swap_phase = _sm.base_swap_phase or "phaseA"
        except Exception:  # noqa: BLE001 — display-only; never fail /status on it
            _base_swap_phase = None
    migration_block = {
        "state": _mig_state,
        "config_rev": _config_rev,
        "trial_started_at": _trial.get("started_at") or None,
        "gates": _gates,
        "comparison": _comparison_block,
        "base_swap_phase": _base_swap_phase,
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
        last_consolidation_error=_state.get("last_consolidation_error"),
        speaker_profiles=store.profile_count if store else 0,
        stt_loaded=stt_loaded,
        stt_model=stt.model_name if stt_loaded else None,
        stt_device=stt_device,
        tts_loaded=tts_loaded,
        tts_languages=tts_languages,
        tts_device=tts_device,
        tts_degraded=tts_degraded,
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
        pending_rehydration=bool(_state.get("pending_rehydration", False)),
        effective_mode=_state.get("effective_mode"),
        last_reclaim_error=_state.get("last_reclaim_error"),
    )


@app.get("/integrity", response_model=IntegrityResponse, dependencies=[Depends(require_admin)])
async def integrity_check():
    """Run the infrastructure integrity check and return the report.

    Cloud-only-safe — no GPU or model dependency.  Verifies every tier's
    ``indexed_key_registry.json``, ``simhash_registry.json``, ``meta.json``
    (live weight slot), and ``graph.json`` (simulate mode only), plus common
    files (``key_metadata.json``, ``speaker_profiles.json``,
    ``observed_languages.json``, ``state/backup.json``).

    Returns a JSON report with ``ok``, ``checks``, and ``failures`` fields.
    """
    from paramem.backup.integrity import verify_infrastructure_integrity

    config = _state["config"]
    daily_loadable = _state.get("daily_loadable", False)
    memory_store = _state.get("memory_store")

    report = verify_infrastructure_integrity(
        config,
        store=memory_store,
        daily_loadable=daily_loadable,
    )

    return IntegrityResponse(
        ok=report.ok,
        checks=[IntegrityCheckItem(**c.to_dict()) for c in report.checks],
        failures=[IntegrityCheckItem(**c.to_dict()) for c in report.failures],
    )


@app.post("/gpu/acquire", dependencies=[Depends(require_admin)])
async def gpu_acquire():
    """Reclaim the GPU in-process and switch to local mode.

    Standard reclaim primitive — symmetric counterpart of ``/gpu/release``.
    Triggers an in-process base-model reload + voice profile switch to gpu
    so existing FastAPI listener, Wyoming sockets, and HA satellites stay
    connected. Also clears any stale ``PARAMEM_EXTRA_ARGS=--defer-model``
    hold from systemd user env.

    Acts whenever mode is cloud-only EXCEPT when ``cloud_only_reason=="explicit"``
    (yaml ``cloud_only: true``) — that flag represents persistent operator
    intent and requires a config edit + restart to leave. Idempotent in
    local mode (returns 200 with ``reloaded_live: false``).

    On reload failure, falls back to ``_restart_service`` so the operator's
    intent is honoured even when the live path fails — EXCEPT when the
    reload was declined for insufficient free VRAM (an external GPU
    consumer holds the device), where a restart would only crash-loop on
    the lifespan VRAM budget gate. In that case the server stays cloud-only
    and the response carries ``deferred_insufficient_vram: true``.
    """
    hold_before = _get_hold_state()
    cleared = _clear_hold_env()
    # Reload whenever ParaMem is in cloud-only mode UNLESS the operator
    # opted in via the yaml ``cloud_only: true`` setting (reason="explicit").
    # The action is operator-driven and explicit; covers the post-/gpu/release
    # reclaim, the --defer-model orphan recovery, and the gpu-conflict autoswitch.
    # ``cloud_only: true`` in yaml represents persistent operator intent and is
    # respected — operators must edit yaml and restart to leave that mode.
    needs_reload = (
        _state.get("mode") == "cloud-only" and _state.get("cloud_only_reason") != "explicit"
    )
    reloaded_live = False
    deferred_insufficient_vram = False
    if needs_reload:
        try:
            await asyncio.get_running_loop().run_in_executor(None, _live_reload_base_model)
            reloaded_live = _state.get("mode") == "local"
        except Exception:  # noqa: BLE001
            logger.exception(
                "In-process reload failed during /gpu/acquire; falling back to restart"
            )
        if reloaded_live:
            # Voice drain+restore is now owned by _live_reload_base_model
            # (partial-path success restore runs inside the primitive).
            # ── Base-swap deferred-resume hook ────────────────────────────────
            # When a phaseA_done base-swap marker exists and the orchestration
            # is not actively running (base_swap_active=False), the reload that
            # just succeeded means Phase B can now run.  Re-launch the
            # orchestration in resume mode so Phase B proceeds automatically
            # without operator intervention.
            _bs_mig = _state.get("migration") or {}
            if not _bs_mig.get("base_swap_active", False):
                _config_for_resume = _state.get("config")
                if _config_for_resume is not None:
                    _sd_resume = (_config_for_resume.paths.data / "state").resolve()
                    _br_resume = (_config_for_resume.paths.data / "backups").resolve()
                    _deferred_marker = read_trial_marker(_sd_resume)
                    if (
                        _deferred_marker is not None
                        and _deferred_marker.migration_kind == "base_swap"
                        and _deferred_marker.base_swap_phase == "phaseA_done"
                    ):
                        _live_cfg_resume = (
                            Path(_state["config_path"])
                            if _state.get("config_path")
                            else Path("configs/server.yaml")
                        )
                        asyncio.create_task(
                            _run_base_swap_orchestration(
                                candidate_path_str=str(_live_cfg_resume),
                                live_config_path=_live_cfg_resume,
                                state_dir=_sd_resume,
                                backups_root=_br_resume,
                                old_model=_deferred_marker.old_model,
                                new_model=_deferred_marker.new_model,
                                started_at=_deferred_marker.started_at,
                                candidate_hash=_deferred_marker.candidate_config_sha256,
                                resume_phase="phaseA_done",
                            )
                        )
                        logger.info(
                            "/gpu/acquire: re-launching deferred base-swap Phase B (old=%s new=%s)",
                            _deferred_marker.old_model,
                            _deferred_marker.new_model,
                        )
        elif _state.get("cloud_only_reason") == "insufficient_vram":
            # Free device memory cannot hold the model (an external GPU
            # consumer holds it). A restart would only re-hit the lifespan
            # VRAM budget gate and crash-loop, so stay cloud-only and tell
            # the operator to free the GPU first. Voice back to CPU so the
            # server holds no GPU memory.
            deferred_insufficient_vram = True
            await asyncio.get_running_loop().run_in_executor(
                None, _set_voice_pipeline_profile, "cpu"
            )
            logger.warning(
                "/gpu/acquire: insufficient free VRAM to reload the model — "
                "staying cloud-only. Free the GPU and retry `pstatus --acquire`."
            )
        else:
            _restart_service()
    return {
        "cleared": cleared,
        "was_active": hold_before["hold_active"],
        "owner_pid": hold_before["owner_pid"],
        "owner_alive": hold_before["owner_alive"],
        "will_restart": needs_reload and not reloaded_live and not deferred_insufficient_vram,
        "reloaded_live": reloaded_live,
        "deferred_insufficient_vram": deferred_insufficient_vram,
    }


def _preload_memory_store(config, *, model, tokenizer):
    """Build the MemoryStore, load registries, and hydrate the active-key cache.

    Called by :func:`_build_config_derived_state`.  Returned store is assigned
    to ``_state["memory_store"]`` by the caller.

    Source selection uses ``config.consolidation.mode`` (NOT
    ``_state["mode"]``).  This prevents conflating the consolidation
    persistence mode (train/simulate) with the runtime mode (local/cloud-only).

    boot_degraded lifecycle:
    - Cleared when full hydration succeeds.
    - Cleared when ``config.inference.preload_cache=False`` (intentional opt-out).
    - Cleared (with an empty store, early return) while a base-model swap is in
      flight — the on-disk registry describes the PREVIOUS model and is invalid
      for the loaded one (see the base-swap gate below).
    - Set when partial hydration occurs (some active keys not materialised),
      including the registry-sha256 binding-bug (slot present but unmounted).

    The ``WeightMemorySource`` is kept as a frame-local and dropped on return —
    mirrors the no-frame-retention pattern of ``_load_model_into_state`` so the
    base model is not pinned past the preload.

    Parameters
    ----------
    config:
        Live server config object.
    model:
        Base model handle (``_state["model"]``) passed directly as a kwarg
        expression at the call site — do NOT bind to a caller local.
    tokenizer:
        Tokenizer handle (``_state["tokenizer"]``) passed the same way.

    Returns
    -------
    MemoryStore
        The fully-constructed store (registries loaded; entries hydrated when
        ``preload_cache=True`` and the source probe succeeded).
    """
    from paramem.memory.source import (
        DiskMemorySource as _DiskMemorySource,
    )
    from paramem.memory.source import (
        WeightMemorySource as _WeightMemorySource,
    )
    from paramem.memory.store import MemoryStore as _MemoryStore

    memory_store = _MemoryStore(
        replay_enabled=config.consolidation.indexed_key_replay,
    )

    # Base-swap invalidity gate.  While a base-model swap is in flight (Phase A has
    # deleted the old model's weight slots; Phase B has not yet retrained the new
    # model), the on-disk per-tier registries describe the OLD model and have no
    # relation to the loaded NEW one.  Do NOT load them into the live store — the
    # new model knows nothing until Phase B completes.  The registry files stay on
    # disk untouched (Phase B retrains from each tier's graph.json; a rollback
    # restores them from the swap bundle).  This keeps the live store consistent
    # with the loaded weights instead of carrying phantom previous-model keys.  The
    # marker is written before the reload and cleared on Phase B success, so it
    # covers both the in-process reload and a boot-resume.
    from paramem.server.trial_state import read_trial_marker as _read_trial_marker

    _swap_marker = _read_trial_marker((config.paths.data / "state").resolve())
    if _swap_marker is not None and _swap_marker.migration_kind == "base_swap":
        _state["store_load_degraded"] = False
        _state["boot_degraded"] = None
        logger.info(
            "preload_cache: base-swap in flight (phase=%s) — on-disk registry "
            "describes the previous model; live store starts empty until Phase B "
            "retrains the new model.",
            _swap_marker.base_swap_phase or "?",
        )
        return memory_store

    try:
        memory_store.load_registries_from_disk(config.adapter_dir)
        _state["store_load_degraded"] = False
    except Exception:
        logger.error(
            "Boot-time registry load failed; memory store will start empty — "
            "active-store migration will be refused until this is resolved",
            exc_info=True,
        )
        _state["store_load_degraded"] = True

    if not config.inference.preload_cache:
        # Intentional opt-out: inference pays per-key latency on misses.
        # Clear boot_degraded so the cold-cache attention item is not raised on
        # a preload-off deployment after an apply (correction #5 boot_degraded
        # lifecycle).
        _state["boot_degraded"] = None
        logger.info(
            "preload_cache: disabled — store stays entry-empty; inference pays source latency"
        )
    else:
        _preload_keys_by_tier: dict[str, list[str]] = {}
        for _tier in memory_store.tiers_with_registry():
            _active = memory_store.active_keys_in_tier(_tier)
            if _active:
                _preload_keys_by_tier[_tier] = _active

        if not _preload_keys_by_tier:
            # No active keys — nothing to preload; store is correctly empty.
            _state["boot_degraded"] = None
        else:
            # Mode-aware source: simulate persists graph.json (DiskMemorySource);
            # train persists only adapter weights (WeightMemorySource).
            # Select from config.consolidation.mode — NOT from _state["mode"]
            # (that conflates consolidation persistence mode with runtime mode).
            _mode = config.consolidation.mode
            # BASE-MODEL HOLDER (function-local _source — dropped on return)
            _source = None
            if _mode == "simulate":
                _source = _DiskMemorySource(config.adapter_dir)
            elif model is not None:
                # Pass the on-disk SimHash registry to the WeightMemorySource so
                # it gates BEFORE caching (belt-and-suspenders early-out).
                # The store-boundary gate in MemoryStore.probe is the hermetic
                # authority; this prevents below-threshold entries from entering
                # the cache in the first place, matching the per-query path
                # (inference.py:782 passes registry= to WeightMemorySource).
                from paramem.server.inference import (
                    _load_simhash_registry as _boot_load_simhash_registry,
                )

                _boot_registry = _boot_load_simhash_registry(config.adapter_dir)
                _source = _WeightMemorySource(
                    model,
                    tokenizer,
                    registry=_boot_registry,
                    batch_size=config.consolidation.recall_probe_batch_size,
                )
            else:
                logger.info(
                    "preload_cache: skipping entry preload — no model loaded "
                    "(cloud-only mode or model load failed); store will stay "
                    "empty for entries and inference will pay source latency "
                    "on each query"
                )

            if _source is not None:
                _total = sum(len(v) for v in _preload_keys_by_tier.values())
                logger.info(
                    "preload_cache: probing %d active key(s) across %d tier(s) via %s",
                    _total,
                    len(_preload_keys_by_tier),
                    type(_source).__name__,
                )
                try:
                    _results = _source.probe(_preload_keys_by_tier)
                except Exception:
                    logger.exception(
                        "preload_cache: source probe failed; store remains "
                        "empty for entries (queries will retry per-key on demand)"
                    )
                    _results = {}

                _hits = 0
                _missed_by_tier: dict[str, list[str]] = {}
                for _tier, _keys in _preload_keys_by_tier.items():
                    for _key in _keys:
                        _entry = _results.get(_key)
                        if _entry is None or "failure_reason" in _entry:
                            _missed_by_tier.setdefault(_tier, []).append(_key)
                            continue
                        # Don't re-register; the registry is already loaded.
                        memory_store.put(_tier, _key, _entry, register=False)
                        _hits += 1
                logger.info("preload_cache: cached %d / %d active key(s)", _hits, _total)
                if _hits < _total:
                    _state["boot_degraded"] = {
                        "reason": "preload_partial",
                        "hits": _hits,
                        "total": _total,
                        "missed_by_tier": {
                            tier: keys[:10] for tier, keys in _missed_by_tier.items()
                        },
                        "source": type(_source).__name__,
                    }
                    logger.warning(
                        "boot_degraded: preload_cache could not materialise %d / %d "
                        "active keys via %s — recall self-heals via on-miss weight "
                        "probing; the cache re-warms on the next apply or /gpu/acquire",
                        _total - _hits,
                        _total,
                        type(_source).__name__,
                    )
                else:
                    # Full hydration — clear any prior degraded flag.
                    _state["boot_degraded"] = None

            # Drop the WeightMemorySource frame-local — the preload probe is
            # complete; the source must not outlive this function's frame.
            _source = None

    # Load per-key bookkeeping (speaker_id, first_seen_cycle) into
    # MemoryStore._bookkeeping from key_metadata.json.  Entry-independent —
    # runs regardless of preload setting or boot_degraded.  Under
    # inference.preload_cache=False this is the sole provenance write at boot,
    # and is sufficient for the router's speaker index (iter_bookkeeping).
    try:
        _meta_stats = memory_store.load_bookkeeping_from_disk(config.key_metadata_path)
        logger.info(
            "load_bookkeeping_from_disk: loaded=%d orphaned=%d legacy_upgraded=%d",
            _meta_stats["loaded"],
            _meta_stats["orphaned"],
            _meta_stats["legacy_upgraded"],
        )
    except Exception:
        logger.exception(
            "Boot-time key_metadata bookkeeping load failed; _bookkeeping will be empty "
            "until next consolidation cycle (speaker scoping will cold-start)"
        )

    # Infrastructure integrity check — runs after all loaders so the store
    # is fully populated for cross-consistency checks.  A corrupt registry
    # blocks migrations and flags store_load_degraded (a corrupt registry is a
    # different, more severe condition than boot_degraded's cold cache).
    #
    # Boot housekeeping runs FIRST: cleanup_partial_slots deletes any
    # subdirectory under <adapter_dir>/<tier>/ that is missing one of the
    # canonical three slot files.  These are scratch left by interrupted
    # training (a class that Commit 5′'s staging+promote contract should
    # have eliminated going forward, but historical state may still carry
    # them).  Removals are recorded on _state["integrity_cleanup"] so the
    # attention populator can surface them to the operator.
    try:
        from paramem.backup import key_store as _key_store_mod
        from paramem.backup.integrity import (
            cleanup_partial_slots,
            verify_infrastructure_integrity,
        )

        _removed = cleanup_partial_slots(config.adapter_dir)
        if _removed:
            _state["integrity_cleanup"] = _removed
            logger.warning(
                "integrity-cleanup removed %d partial slot(s) before integrity check",
                len(_removed),
            )

        _daily_ok_local = _key_store_mod.daily_identity_loadable(
            _key_store_mod.DAILY_KEY_PATH_DEFAULT
        )
        _integrity_report = verify_infrastructure_integrity(
            config,
            store=memory_store,
            daily_loadable=_daily_ok_local,
        )
        if not _integrity_report.ok:
            for _fc in _integrity_report.failures:
                logger.error(
                    "Integrity failure [%s/%s] %s: %s",
                    _fc.category,
                    _fc.tier,
                    _fc.path,
                    _fc.detail,
                )
            _state["store_load_degraded"] = True
            logger.error(
                "Boot-time integrity check found %d failure(s); "
                "active-store migration will be refused until this is resolved",
                len(_integrity_report.failures),
            )
        else:
            logger.info(
                "Boot-time integrity check passed (%d checks)",
                len(_integrity_report.checks),
            )
    except Exception:
        logger.exception(
            "Boot-time integrity check raised unexpectedly; store_load_degraded left unchanged"
        )

    return memory_store


def _build_config_derived_state(
    config,
    *,
    cloud_only: bool,
    rebuild_session_buffer: bool = True,
    full_rebuild: bool = True,
) -> None:
    """Construct every config-derived component into ``_state``.

    Single idempotent routine called by BOTH the lifespan startup and the
    live-apply path.  Replaces the lifespan's inline construction blocks at
    ``app.py:1546-1732`` + ``1753-2017`` (excluding the build-once post-load
    VRAM gate at ``1733-1751`` — see B-1 / §3.x of the plan).

    **EXCLUDES** strictly-once lifespan concerns (signal handlers, asyncio
    tasks, timer reconciliation, mode init) and the build-once post-load VRAM
    gate.  The gate runs inline in the lifespan AFTER this routine returns so
    the measured allocation includes the STT/TTS GPU footprint (correction S2).
    The gate is incompatible with this routine (it ``sys.exit(1)``s and reads
    a lifespan-frame local ``base_pred``).  The apply path's VRAM safety is
    the ``mem_get_info``-based fit-check inside ``_live_reload_base_model``.

    **Wyoming listener sockets are NOT re-bound here.** The lifespan binds
    them once with provider lambdas so profile swaps re-point automatically.
    The apply path must not call ``start_wyoming_server`` /
    ``start_wyoming_tts_server`` again.

    Per §3.11 ordering:

    1. session_buffer (snapshot old → construct → rehydrate → load_snapshot)
       — only when ``rebuild_session_buffer=True`` AND ``full_rebuild=True``.
    2. speaker_store (+ embedding model)  — ``full_rebuild=True`` only.
    3. STT/TTS managers: construct fresh from ``config`` → flip voice_box
       — ``full_rebuild=True`` only.
    4. sota_agent + sota_providers  — ``full_rebuild=True`` only.
    5. ha_client (close old → construct new) + ha_graph
       — ``full_rebuild=True`` only.
    6. memory store preload (``_preload_memory_store``) → assign
       ``_state["memory_store"]`` — always (D6 gate may skip probe on warm).
    7. router (captures memory_store + ha_graph)  — always.
    8. exemplar banks + ``set_classifier_model``
       — ``full_rebuild=True`` only for exemplar banks; ``set_classifier_model``
       runs on both paths so the freshly loaded model is registered.
    9. language_tracker + lang_id  — ``full_rebuild=True`` only.

    Parameters
    ----------
    config:
        Live server config object (already updated in ``_state["config"]``
        before this is called from the apply path).
    cloud_only:
        Whether the server is starting/applying in cloud-only mode.  Controls
        GPU pair construction for STT/TTS and intent exemplar loading.
    rebuild_session_buffer:
        When ``True`` (default, used at lifespan boot and the apply path when
        ``retain_sessions`` or ``debug`` changed), always construct a fresh
        ``SessionBuffer``.  When ``False``, leave the live buffer intact to
        avoid losing in-flight state.  Only meaningful when ``full_rebuild=True``
        (the session buffer is never rebuilt on a plain reclaim).
    full_rebuild:
        When ``True`` (default, boot + apply path): rebuild ALL config-derived
        components (steps 1–9 above).
        When ``False`` (plain ``/gpu/acquire`` + auto-reclaim same config):
        skip the expensive, potentially network-touching steps (speaker_store,
        STT/TTS construction, sota_agent, ha_client reconnect, exemplar banks,
        language_tracker).  Only steps 6 (memory-store probe, D6-gated) and 7
        (Router re-point) are run, plus ``set_classifier_model`` to register
        the freshly reloaded model handle (step 8 partial).  This avoids
        spurious STT/TTS ``load()`` calls and HA ``health_check()`` /
        ``load_entity_map()`` / ``get_services()`` network calls on every
        plain warm reclaim (correction S1).
    """
    # ── 1. session_buffer ────────────────────────────────────────────────────
    # Only on full_rebuild paths (boot + apply); plain reclaim keeps the live
    # buffer to avoid losing in-flight state.
    if full_rebuild and rebuild_session_buffer:
        # Save snapshot on the OLD buffer FIRST (mirrors SIGTERM/SIGUSR1 path)
        # so mid-turn _sessions state round-trips when an encryption key is
        # loaded (correction S-1).  No-op on a no-key deployment.
        old_buffer = _state.get("session_buffer")
        if old_buffer is not None:
            old_buffer.save_snapshot()

        _state["session_buffer"] = SessionBuffer(
            config.session_dir,
            retain_sessions=config.consolidation.retain_sessions,
            debug=config.debug,
        )
        # Cold-start: rehydrate pending JSONL into memory before loading the
        # encrypted snapshot (snapshot carries mid-turn _sessions state only).
        _state["session_buffer"].rehydrate_from_disk()
        _state["session_buffer"].load_snapshot()

    if full_rebuild:
        # ── 2. speaker_store ─────────────────────────────────────────────────
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

        # ── 3. STT / TTS managers (construct-then-flip, §3.6) ───────────────
        # Construct fresh managers from config, then atomically flip voice_box.
        # DO NOT call start_wyoming_server / start_wyoming_tts_server here —
        # those are lifespan-only.  The Wyoming provider lambdas re-point
        # automatically once voice_box is updated.
        #
        # Correction §3.6: _set_voice_pipeline_profile only lazy-constructs when
        # _state["stt_gpu"] is None; it does NOT reconstruct when the config
        # changes.  The apply must construct fresh instances from config B and
        # install them before flipping.
        #
        # Correction N3: flush the allocator-pool before the GPU STT load so
        # vram_measure reads accurate free-memory (mirrors
        # _set_voice_pipeline_profile which calls safe_empty_cache before load).
        _state["stt"] = None
        _state["wyoming_server"] = _state.get("wyoming_server")  # preserve existing listener ref

        if config.stt.enabled:
            from paramem.server.stt import WhisperSTT

            # CPU pair — always loaded, permanently resident.
            stt_cpu = WhisperSTT(
                model_name=config.stt.cpu_fallback_model,
                device="cpu",
                compute_type="int8",
                language=config.stt.language,
                beam_size=config.stt.beam_size,
                vad_filter=config.stt.vad_filter,
            )
            # Synchronous load (function is called from a thread on the apply
            # path; _set_voice_pipeline_profile loads synchronously for the same
            # reason).
            if stt_cpu.load():
                _state["stt_cpu"] = stt_cpu
                logger.info("Local STT CPU: %s on cpu", config.stt.cpu_fallback_model)
            else:
                logger.warning(
                    "Local STT CPU pair failed to load — voice path unavailable in cloud-only mode"
                )

            if not cloud_only:
                # N3: flush allocator-pool before GPU STT load so vram_measure
                # reads accurate free-memory (mirrors _set_voice_pipeline_profile).
                safe_empty_cache()
                stt_gpu = WhisperSTT(
                    model_name=config.stt.model,
                    device=config.stt.device,
                    compute_type=config.stt.compute_type,
                    language=config.stt.language,
                    beam_size=config.stt.beam_size,
                    vad_filter=config.stt.vad_filter,
                )
                if stt_gpu.load():
                    _state["stt_gpu"] = stt_gpu
                    logger.info(
                        "Local STT GPU: Whisper %s on %s", config.stt.model, config.stt.device
                    )
                else:
                    logger.warning("Local STT GPU pair failed to load")
        else:
            logger.info("Local STT: disabled")

        if config.tts.enabled:
            from paramem.server.tts import TTSManager

            # CPU pair — always loaded.
            tts_cpu = TTSManager(_build_cpu_tts_config(config.tts))
            tts_cpu.load_all()
            if tts_cpu.is_loaded:
                _state["tts_cpu"] = tts_cpu
                logger.info("Local TTS CPU: %s", ", ".join(tts_cpu.available_languages))
            else:
                logger.warning(
                    "Local TTS CPU pair failed to load — voice path unavailable in cloud-only mode"
                )

            if not cloud_only:
                tts_gpu = TTSManager(config.tts)
                tts_gpu.load_all()
                if tts_gpu.is_loaded:
                    _state["tts_gpu"] = tts_gpu
                    logger.info("Local TTS GPU: %s", ", ".join(tts_gpu.available_languages))
                else:
                    logger.warning("Local TTS GPU pair failed to load")
        else:
            logger.info("Local TTS: disabled")

        # Seed voice_box and voice_profile based on startup/apply mode.
        _active_stt = _state.get("stt_gpu") if not cloud_only else _state.get("stt_cpu")
        _active_tts = _state.get("tts_gpu") if not cloud_only else _state.get("tts_cpu")
        if _active_stt is not None or _active_tts is not None:
            _state["voice_box"] = {"stt": _active_stt, "tts_manager": _active_tts}
            _state["stt"] = _active_stt
            _state["tts_manager"] = _active_tts

        _state["voice_profile"] = "cpu" if cloud_only else "gpu"
        logger.info("Voice pipeline profile: %r", _state["voice_profile"])

        # ── 4. SOTA agent + providers ─────────────────────────────────────────
        _state["sota_agent"] = get_cloud_agent(config.sota_agent)
        if _state["sota_agent"]:
            logger.info(
                "SOTA agent: %s (%s)",
                config.sota_agent.provider,
                config.sota_agent.model,
            )
        else:
            logger.info("SOTA agent: not configured")

        _state["sota_providers"] = {}
        for name, provider_config in config.sota_providers.items():
            agent = get_cloud_agent(provider_config)
            if agent:
                _state["sota_providers"][name] = agent
                logger.info("SOTA provider registered: %s (%s)", name, provider_config.model)
        logger.info("SOTA providers available: %s", list(_state["sota_providers"].keys()))

        # ── 5. HA client + ha_graph ──────────────────────────────────────────
        # Mandatory teardown (correction S-2): HAClient holds an httpx.Client pool.
        # Close the OLD client before reassigning — else the pool leaks on every apply.
        old_ha_client = _state.get("ha_client")
        if old_ha_client is not None:
            try:
                old_ha_client.close()
            except Exception:
                logger.exception("HA client close failed during rebuild; pool may leak")

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
                ha_services = ha_client.get_services()
                ha_graph = HAEntityGraph.build(ha_client._raw_states, ha_services)
            else:
                logger.warning("HA client: configured but unreachable at %s", tools_config.ha.url)
            _state["ha_client"] = ha_client
        else:
            _state["ha_client"] = None
            logger.info("HA tools: not configured")

        _state["ha_graph"] = ha_graph
    else:
        # Plain reclaim path (full_rebuild=False): keep existing STT/TTS managers
        # and ha_graph — config did not change, no network reconnect needed
        # (correction S1).
        ha_graph = _state.get("ha_graph")
        logger.info(
            "_build_config_derived_state: plain reclaim — skipping STT/TTS/HA/SOTA/exemplar "
            "rebuild (full_rebuild=False, same config)"
        )

    # ── 6. Memory store preload ───────────────────────────────────────────────
    # D6 gate: re-probe only when (a) the cache is cold (boot_degraded set or
    # store is None/empty), or (b) called from the apply path (caller sets
    # _state["_apply_config_in_progress"] = True before calling this routine
    # and clears it after).  On a plain reclaim with a warm store, still
    # rebuild the Router cheaply but skip the probe.
    _do_probe = (
        _state.get("boot_degraded") is not None
        or _state.get("memory_store") is None
        or _state.get("_apply_config_in_progress", False)
    )

    if _do_probe:
        memory_store = _preload_memory_store(
            config,
            model=_state.get("model"),
            tokenizer=_state.get("tokenizer"),
        )
        _state["memory_store"] = memory_store
    else:
        # Warm store survives — keep it, just rebuild the Router below.
        logger.info(
            "_build_config_derived_state: warm store present (boot_degraded=None) — "
            "skipping re-probe (D6 gate); rebuilding router only"
        )
        memory_store = _state["memory_store"]

    # ── 7. Router ────────────────────────────────────────────────────────────
    # Router must be rebuilt AFTER ha_graph (step 5 / or preserved from _state)
    # because it captures ha_graph.
    _state["router"] = QueryRouter(
        adapter_dir=config.adapter_dir,
        memory_store=memory_store,
        ha_graph=ha_graph,
        intent_config=config.intent,
    )

    # ── 8. Exemplar banks + classifier model ──────────────────────────────────
    # set_classifier_model runs on ALL paths so the freshly reloaded model handle
    # is registered after every model load.  Exemplar banks (load_encoder +
    # load_exemplars) run only on full_rebuild (config may have changed).
    if not cloud_only:
        from paramem.server.intent import set_classifier_model

        if full_rebuild:
            from paramem.server.intent import (
                load_encoder,
                load_exemplars,
            )

            encoder_handle = load_encoder(config.intent)
            if encoder_handle is not None:
                load_exemplars(config.intent, encoder_handle)

        # Register the local LLM for intent.mode=llm classification.
        # BASE-MODEL HOLDER (function-local _classifier_model /
        # _classifier_tokenizer): drop them immediately — the routine's frame
        # collapses on return, but being explicit mirrors the lifespan pattern
        # (lifespan is a suspended generator; this function is not, but
        # clarity is load-bearing for holder audits).
        _classifier_model = _state.get("model")
        _classifier_tokenizer = _state.get("tokenizer")
        if _classifier_model is not None and _classifier_tokenizer is not None:
            set_classifier_model(_classifier_model, _classifier_tokenizer)
        _classifier_model = None
        _classifier_tokenizer = None

        if full_rebuild:
            from paramem.server.sentence_type import (
                load_exemplars as load_sentence_type_exemplars,
            )

            load_sentence_type_exemplars(config.sentence_type)

            from paramem.server.personal_referent import (
                load_exemplars as load_personal_referent_exemplars,
            )

            load_personal_referent_exemplars(config.personal_referent)

    # ── 9. language_tracker + lang_id ────────────────────────────────────────
    # Only on full_rebuild: plain reclaim keeps the existing tracker (same config).
    if full_rebuild:
        from paramem.server.language_tracker import LanguageTracker

        _state["language_tracker"] = LanguageTracker(
            store_path=config.paths.data / "observed_languages.json",
            ha_client=_state.get("ha_client"),
        )

        if config.text_lang_detection.enabled:
            from paramem.server import lang_id

            lang_id.load_at_startup(config.text_lang_detection.model_path)


def _load_model_into_state(config) -> None:
    """Load base model + adapters into ``_state`` without retaining the
    handles in the caller's frame.

    Called from lifespan startup AND from :func:`_live_reload_base_model`.
    Architecturally this function returns nothing — keeping the model
    out of the caller's frame is load-bearing. The original lifespan
    inlined this block; the resulting ``model``/``tokenizer`` locals
    were retained by the lifespan async generator's frame for the app's
    lifetime, pinning the entire ``MistralForCausalLM`` (~4 GiB of
    bitsandbytes ``Params4bit`` tensors) and making the device
    unrecoverable via any in-process release path. Factoring the load
    into a function whose locals go out of scope on return solves this
    by construction.

    Manifest caches (``adapter_manifest_status``,
    ``base_model_hash_cache``) are reset on every load — they describe
    the model that's currently in ``_state["model"]`` and a swap
    invalidates them.

    The per-process VRAM cap is applied here (not just at lifespan
    startup) so a transition cloud-only → local via /gpu/acquire
    — where the lifespan's own ``apply_process_cap`` branch was
    skipped — still gets the safety bulkhead before any tensor
    allocation. ``apply_process_cap`` is idempotent.
    """
    apply_process_cap(fraction=config.vram.process_cap_fraction)
    logger.info("Loading model: %s (%s)", config.model_name, config.model_config.model_id)
    model, tokenizer = load_base_model(config.model_config)

    # Manifest caches are model-specific; re-init on every load.
    _state["adapter_manifest_status"] = {}
    _state["base_model_hash_cache"] = {}

    # Refuse to start on a legacy adapter layout (pre-2026-05-14 hierarchy
    # refactor). Operator must run scripts/migrate/restructure_adapter_dir.py
    # before restart. The new code paths use iter_interim_dirs() which scans
    # adapter_dir/episodic/interim_*, so legacy adapter_dir/episodic_interim_*
    # dirs would be invisible and produce a silently degraded server.
    from paramem.memory.interim_adapter import detect_legacy_adapter_layout

    _legacy = detect_legacy_adapter_layout(config.adapter_dir)
    if _legacy:
        names = ", ".join(p.name for p in _legacy)
        raise RuntimeError(
            f"Legacy adapter layout detected at {config.adapter_dir} ({names}). "
            "Run scripts/migrate/restructure_adapter_dir.py to relocate interim "
            "adapters under adapter_dir/episodic/, then restart."
        )

    # Mount adapters from slots — wraps the model in PeftModel.
    model = _mount_adapters_from_slots(model, tokenizer, config, _state)

    # Drop orphan entries from key_metadata.json — keys whose tier registry was
    # wiped (e.g. by the interim-cleanup pass above) must not linger as
    # bookkeeping. Wipe invariant: key_metadata is for active keys, not
    # recovery. Runs every boot; no-op when nothing to prune.
    from paramem.server.consolidation import prune_key_metadata_orphans

    try:
        prune_key_metadata_orphans(config)
    except Exception:
        logger.exception("Boot-time key_metadata orphan prune failed; continuing")

    # Restore the main episodic adapter as the active adapter.
    if hasattr(model, "peft_config") and "episodic" in model.peft_config:
        switch_adapter(model, "episodic")

    _state["model"] = model
    _state["tokenizer"] = tokenizer
    # Drop the locals — see the docstring rationale. Without this, this
    # function's own frame holds the model and a caller doing
    # release+reload would never reclaim memory.
    del model, tokenizer


def _refresh_config_from_disk_into_state():
    """Load the live config from disk, commit it to ``_state``, and arm the active-store rebuild.

    Shared by the lifespan-mirror mode-switch confirm path (``migration_confirm``) and
    the model-reload config-apply path (``_live_reload_base_model``).  Calling this
    without a subsequent model reload is correct for a pure ``consolidation.mode`` change
    because the mode affects consolidation persistence only — not the base model, adapters,
    router, or inference.  The rebuild runs later at ``/consolidate`` (pre-empted by
    ``pending_rehydration`` in ``_maybe_trigger_scheduled_consolidation``) with its own 1.0
    gate + source-mode fallback.

    Returns
    -------
    ServerConfig or None
        The freshly-loaded config on success.  ``None`` when ``config_path`` is unset
        (preserves the existing warning from ``_live_reload_base_model``).
    """
    config_path = _state.get("config_path")
    if config_path:
        new_config = load_server_config(Path(config_path))
        _state["config"] = new_config
        logger.info(
            "Live config refresh: loaded config from %s",
            config_path,
        )
        # Arm the active-store rebuild if this refresh flipped
        # consolidation.mode. The lifespan path arms at startup; without
        # this call a LIVE-applied mode change (migration accept / config
        # apply) would leave the on-disk store stale until the next
        # restart. Reuses the same single arming helper. No-op when the
        # mode is unchanged.
        _arm_active_store_migration(new_config)
        return new_config
    logger.warning(
        "refresh_config_from_disk requested but config_path is not set — "
        "proceeding with current in-memory config"
    )
    return None


def _live_reload_base_model(
    refresh_config_from_disk: bool = False,
    rebuild_session_buffer: bool = False,
    lock_held: bool = False,
) -> None:
    """Release+reload the base model in-process to recover device memory.

    Used as the recovery path when STT cannot reload post-cycle because
    the cycle's allocator-pool growth is reserved-but-unused inside
    PyTorch and cannot collapse while the base model is alive (the
    model's tensors keep the pool's segments active). Genuinely
    destroying and re-creating the model is the only path to actually
    return that headroom to the device — verified empirically: a
    post-cycle device at 4666 MiB used drops to 772 MiB used after this
    function runs, sufficient for STT to reload cleanly.

    Preserves the ``ConsolidationLoop`` and ``BackgroundTrainer`` object
    identities (only swaps their ``self.model``). Their in-memory state
    (cycle_count, key_sessions counters, simhash sets) is not always
    persisted per-cycle and is load-bearing for promotion + debug
    snapshots.

    Voice-pipeline drain/restore (invariant owned by this primitive):
    This function drains the voice pipeline to CPU before its VRAM gate
    (when ``voice_profile=="gpu"``) and restores it after a successful
    PARTIAL reload (``refresh_config_from_disk=False``).  The FULL-rebuild
    path (``refresh_config_from_disk=True``) restores voice via
    ``_build_config_derived_state`` — adding a restore here for that branch
    would double-load.  Failure branches leave voice on CPU so the
    cloud-only server holds ~0 GiB.

    The drain is idempotent for cloud-only callers (``voice_profile=="cpu"``
    → ``_set_voice_pipeline_profile`` early-returns on a matching profile).
    It also closes the double-voice leak in ``_build_config_derived_state``
    full-rebuild: the rebuild overwrites ``_state["stt_gpu"]``/``["tts_gpu"]``
    without unloading the old GPU instances; the drain unloads and nulls them
    first so the rebuild starts from a clean slate.

    Caller responsibilities:
    - ``_apply_config_live`` holds ``gpu_lock_sync()`` around this call and
      MUST pass ``lock_held=True`` so the internal
      ``_set_voice_pipeline_profile`` calls do not re-acquire the non-reentrant
      lock (deadlock).  ``_auto_reclaim_loop`` holds ``gpu_lock()`` across this
      call and likewise passes ``lock_held=True``.  Plain ``/gpu/acquire``
      dispatches this WITHOUT holding ``gpu_lock`` (correction N-1, verified at
      ``app.py:3196``) and keeps the default ``lock_held=False``.
    - Must accept ~25-30 s of model-load latency. Mode is flipped to
      cloud-only for the duration so any concurrent /chat handler
      routes to SOTA rather than crashing on a None model.

    Parameters
    ----------
    refresh_config_from_disk:
        When ``True`` (the config-apply path, called from
        ``_apply_config_live``):

        - Re-read ``_state["config"]`` from disk via ``load_server_config``
          **before** ``_release_base_model_in_process`` — so the new config is
          committed before the release (recoverable if the reload then fails).
        - After a successful ``_load_model_into_state``, call
          ``_build_config_derived_state(config, cloud_only=False,
          full_rebuild=True)`` to rebuild ALL config-derived components
          (memory store preload, router, exemplar banks, STT/TTS managers,
          ha_client/ha_graph, etc.).
        - Flip ``_state["mode"]="local"`` after a clean full rebuild.  A
          partial preload (``boot_degraded`` set) is NOT a failure — recall
          self-heals via on-miss weight probing, so the server stays local and
          surfaces ``boot_degraded`` as a /status attention signal.
        - On rebuild failure: stay cloud-only, set ``cloud_only_reason`` to
          ``"apply_failed"``.

        When ``False`` (plain ``/gpu/acquire`` + auto-reclaim, same config):

        - Model reload + ``set_classifier_model`` (to register the new handle).
        - ``_build_config_derived_state`` is called with ``full_rebuild=False``
          (rebuilds memory-store probe [D6-gated] + Router re-point only).
          STT/TTS, HA reconnect, exemplar banks, and language_tracker are
          skipped — same config, no delta (correction S1).
    rebuild_session_buffer:
        Threaded from ``_apply_config_live`` (correction S3): ``True`` when
        ``retain_sessions`` or ``debug`` changed between config A and config B,
        so the ``SessionBuffer`` is rebuilt.  Always ``False`` on the plain
        reclaim path (``refresh_config_from_disk=False``) — the session config
        did not change.
    lock_held:
        When ``True``, the caller already holds ``gpu_lock_sync()`` (the shared
        non-reentrant threading.Lock from ``paramem/server/gpu_lock.py``).
        The internal ``_set_voice_pipeline_profile`` calls forward this flag so
        they skip re-acquisition.  Must be ``True`` for ``_apply_config_live``
        and ``_auto_reclaim_loop`` callers (both hold the lock); must be
        ``False`` (default) for ``/gpu/acquire`` and base-swap step-6 callers
        (neither holds the lock).

    Note on the synchronous maintenance guard (correction S-4, §3.12):
    When ``refresh_config_from_disk=True`` the CALLER (``_apply_config_live``)
    sets ``_state["mode"]="cloud-only"`` + ``cloud_only_reason="live_reload"``
    on the event loop BEFORE dispatching this function via ``run_in_executor``,
    so the scheduler's ``mode != "local"`` defer at ``app.py:6831`` fires.
    This function is NOT responsible for setting that guard.
    """
    # When refreshing config from disk, load the new config BEFORE releasing
    # the model — so the new config is committed to _state even if the reload
    # then fails (partial-rebuild recovery per §3.15: _state["config"] is
    # already B, so the next /gpu/acquire or restart rebuilds coherently).
    if refresh_config_from_disk:
        _refresh_config_from_disk_into_state()

    config = _state["config"]
    logger.info(
        "Live model reload (refresh_config=%s) — releasing and reloading base model in-process",
        refresh_config_from_disk,
    )
    _state["mode"] = "cloud-only"
    _state["cloud_only_reason"] = "live_reload"

    # Entry drain: move the voice pipeline to CPU before releasing the base
    # model and before the VRAM gate.  Without this drain the gate sees
    # ~4.3 GiB still occupied by STT large-v3-turbo + TTS on 8 GiB hardware
    # and defers to cloud-only (observed 2026-05-29: effective free 3.28 GiB,
    # needed 5.00 GiB).  Idempotent: _set_voice_pipeline_profile early-returns
    # when voice_profile already matches ("cpu" for cloud-only callers).
    # Also closes the double-voice leak in _build_config_derived_state
    # full-rebuild: the rebuild overwrites _state["stt_gpu"]/_state["tts_gpu"]
    # without unloading the old GPU instances; draining here unloads and nulls
    # them first so the rebuild constructs on a clean slate.
    if _state.get("voice_profile") == "gpu":
        _set_voice_pipeline_profile("cpu", lock_held=lock_held)

    # No local refs to ConsolidationLoop / BackgroundTrainer — saving
    # them in this function's frame would re-pin the model graph
    # transitively (verified empirically: an attempt to preserve and
    # re-attach object identity OOM'd at the load step because the
    # local frame kept the prior model alive). Instead we drop both
    # entirely; the next consolidation tick lazily re-creates them via
    # ``create_consolidation_loop``, which seeds from disk
    # (``key_metadata.json``, ``indexed_key_registry.json``,
    # ``simhash_registry_*.json``). The only state that resets is
    # ``cycle_count`` (used for debug snapshot dir naming) — acceptable
    # tradeoff for getting the device back to a fresh memory profile.
    # Release our own model first so the occupancy snapshot below reflects
    # only EXTERNAL consumers.
    _release_base_model_in_process()

    # Re-estimate the VRAM topology ONLY when the config may have changed the
    # model — a base-model swap (refresh_config_from_disk=True).  The boot
    # estimate is cached for the boot model; a swap changes it, so a stale
    # estimate would gate the new model's free against the OLD model's footprint.
    # A plain reclaim reloads the SAME model, so the existing assessment is still
    # valid — recomputing it would be redundant and, on an HF cache miss, would
    # discard the valid boot estimate (returning None → skipping the preflight).
    # Shared with the lifespan boot path via the one estimator; None when
    # uncomputable (cache miss / AutoConfig failure) → live load gate.
    if refresh_config_from_disk:
        _state["topology_assessment"] = _compute_topology_assessment(
            config,
            predict_base_bytes(
                config.model_config,
                nf4_disk_to_runtime_factor=config.vram.nf4_disk_to_runtime_factor,
            ),
        )

    # Look before you leap: refuse the load when the GPU cannot fit the topology.
    # Identical gate to boot — _wait_for_gpu_drain over _effective_free_bytes (the
    # device is already free after the upfront release, so the poll's first read
    # passes immediately; a genuine external consumer still fails it after the
    # wait).  ``topology_assessment`` is None when the estimate could not be
    # computed (HF cache miss / AutoConfig failure) → defer to the live load gate.
    assessment = _state.get("topology_assessment")
    if (
        assessment is not None
        and torch.cuda.is_available()
        and not _wait_for_gpu_drain(assessment.required_bytes)
    ):
        _state["cloud_only_reason"] = "insufficient_vram"
        logger.warning(
            "Live model reload skipped — insufficient GPU room for required "
            "%.2f GiB for %s; staying cloud-only, will retry when VRAM frees.",
            assessment.required_bytes / 2**30,
            config.model_name,
        )
        return

    load_failed = False
    try:
        _load_model_into_state(config)
    except Exception:
        # Log here (the traceback is still live), but do NOT free here:
        # the partially-loaded model is pinned by this active traceback,
        # so ``safe_empty_cache`` would not return its bytes. The cleanup
        # runs below, after the except block drops the traceback.
        logger.exception("Live model reload failed during base-model load")
        load_failed = True

    if load_failed:
        # (b) Fail clean. The traceback is gone now, so the partial model
        # is unreferenced — ``_release_base_model_in_process`` ->
        # ``safe_empty_cache`` (gc.collect + clearCublasWorkspaces +
        # empty_cache) actually returns its device memory, leaving the
        # cloud-only server at ~0 GiB. STT/TTS GPU teardown is the
        # caller's job: it must run outside the ``gpu_lock`` this function
        # may be holding (the auto-reclaim path calls us under that lock).
        _release_base_model_in_process()
        _state["cloud_only_reason"] = "reload_failed"
        logger.error(
            "Live model reload failed — released partial allocation, "
            "server stays cloud-only until the GPU frees or a restart."
        )
        return

    if refresh_config_from_disk:
        # Config-apply path: full component rebuild via the shared routine.
        # Signal D6 gate that this is an apply (probe even when store is warm).
        _state["_apply_config_in_progress"] = True
        rebuild_failed = False
        try:
            # S3: rebuild_session_buffer is threaded from _apply_config_live
            # (True when retain_sessions or debug changed between A and B).
            _build_config_derived_state(
                config,
                cloud_only=False,
                rebuild_session_buffer=rebuild_session_buffer,
                full_rebuild=True,
            )
        except Exception:
            logger.exception("Live config apply: _build_config_derived_state failed")
            rebuild_failed = True
        finally:
            _state.pop("_apply_config_in_progress", None)

        if rebuild_failed:
            _release_base_model_in_process()
            _state["cloud_only_reason"] = "apply_failed"
            logger.error(
                "Live config apply: component rebuild failed — staying cloud-only. "
                "Config is already on disk; restart or /gpu/acquire to retry."
            )
            return

        # Full rebuild succeeded.  A partial preload (boot_degraded set by
        # _preload_memory_store inside _build_config_derived_state) is NOT a
        # failure: recall self-heals via on-miss weight probing and the cache
        # re-warms on demand, so the server stays local.  boot_degraded stays
        # set as a signal — surfaced in the /status attention block — and is
        # cleared when a later preload fully hydrates.  The set_classifier_model
        # call is inside _build_config_derived_state (step 8 exemplar banks).
        _state["mode"] = "local"
        _state["cloud_only_reason"] = None
        logger.info("Live config apply — complete; mode=local")
    else:
        # Plain reclaim path (same config): rebuild Router + classifier handle.
        # D6 gate: _build_config_derived_state skips the expensive weight-probe
        # when the store is warm (boot_degraded=None, memory_store non-None).
        # _apply_config_in_progress is NOT set here.
        rebuild_failed = False
        try:
            # S1: full_rebuild=False — plain reclaim rebuilds only the memory
            # store (D6-gated), Router (re-point at warm store), and
            # set_classifier_model (re-register the new model handle).
            # STT/TTS construction, HA reconnect, sota_agent, exemplar banks,
            # and language_tracker are skipped (same config, no delta).
            _build_config_derived_state(
                config,
                cloud_only=False,
                rebuild_session_buffer=False,
                full_rebuild=False,
            )
        except Exception:
            logger.exception(
                "Live model reload: _build_config_derived_state failed; "
                "intent/router may be stale until next reload or restart"
            )
            rebuild_failed = True

        if not rebuild_failed:
            _state["mode"] = "local"
            _state["cloud_only_reason"] = None
            # Partial-path success restore: the entry-drain moved voice to CPU;
            # put it back now that the base model is live.  The full-rebuild path
            # (refresh_config_from_disk=True) skips this — _build_config_derived_state
            # already reconstructed voice on GPU and set voice_profile="gpu".
            _set_voice_pipeline_profile("gpu", lock_held=lock_held)
            logger.info("Live model reload — complete; mode=local")
        else:
            _release_base_model_in_process()
            _state["cloud_only_reason"] = "reload_failed"
            logger.error(
                "Live model reload: component rebuild failed after successful model load — "
                "released allocation, staying cloud-only."
            )


# ════════════════════════════════════════════════════════════════════════════
#  INVARIANT — BASE-MODEL HOLDER REGISTRY  (cloud-only VRAM-leak guard)
#  Every reference to the base model (``_state["model"]``) must be dropped on
#  release so a cloud-only server holds ~0 GiB. Holders accumulate as new
#  components capture the model; a teardown that silently goes stale leaks the
#  whole base model (~4 GiB) — exactly what happened pre-2026-05-21.
#    • Find every holder:   grep -rn "BASE-MODEL HOLDER" paramem/
#    • Object / module-global holders  → drop them in this function.
#    • Lifespan-frame locals           → drop them IN THE LIFESPAN. This
#      function runs OUTSIDE the suspended ``@asynccontextmanager`` frame and
#      CANNOT reach a frame local (same reason _load_model_into_state keeps the
#      model out of its caller's frame).
#    • ALWAYS verify a change with a LIVE  POST /gpu/release → nvidia-smi ~0.
#      Unit tests mock this function and will NOT catch a leak.
# ════════════════════════════════════════════════════════════════════════════
def _release_base_model_in_process() -> None:
    """Drop every reference to the base model and free its device memory.

    The base model is reachable through five holders:

    1. ``_state["model"]`` — primary handle.
    2. ``_state["consolidation_loop"].model`` (and ``.extraction.model``,
       ``.merger.model``) — captured at ``ConsolidationLoop.__init__``;
       released via ``loop.release()``.  The model-bearing
       ``GraphMerger`` (``loop.merger``) is a sub-object holder reached
       via ``loop.merger.model``; it is released transitively through
       ``loop.release()`` → ``merger.release()`` (the release is
       encapsulated — this function must NOT reach into ``loop.merger.model``
       directly, honouring the INVARIANT).
    3. ``_state["background_trainer"].model`` — captured at
       ``BackgroundTrainer.__init__``; released via ``bt.release()``.
    4. **The bg-trainer worker thread's frame.** After a train-mode cycle,
       ``bt._worker_thread`` is parked on ``self._job_queue.get()`` with
       stale locals from the prior job (closure-captured ``loop``).  Until a
       NEW job arrives those locals pin ``loop.model``.  ``bt.release()``
       calls ``_stop_callable_worker()``, which sends ``_WORKER_STOP``,
       joins the thread, then nulls ``_worker_thread`` — breaking the
       ``bt ↔ Thread._target (bound method)`` cycle that ``join`` alone
       does not sever.  Verified by a live ``gc.get_referrers`` walk
       (2026-05-29): 2.796 GiB still allocated after join-only; 0 GiB
       after the explicit null.
    5. ``intent._classifier_model_singleton`` — the ``_ClassifierModelHandle``
       set by ``set_classifier_model`` for ``intent.mode=llm``. Cleared
       here via ``set_classifier_model(None, None)``.

    Holders 2–4 are encapsulated in ``bt.release()`` and ``loop.release()``.
    Holder 5 is cleared below.  See ``BackgroundTrainer.release()``,
    ``ConsolidationLoop.release()``, and ``GraphMerger.release()`` for the
    ownership contract.

    NOTE: this function CANNOT reach references held in the **lifespan
    async-generator frame** (it stays suspended at ``yield`` for the app's
    lifetime). Those — the ``WeightMemorySource`` boot-preload local and the
    ``_classifier_model`` local — are dropped in the lifespan itself
    (``_source = None`` / ``_classifier_model = None`` after use), mirroring
    why :func:`_load_model_into_state` keeps the model out of its caller's
    frame.

    Idempotent: callable when the model is already absent.
    """
    from paramem.server.intent import set_classifier_model

    bt = _state.get("background_trainer")
    loop = _state.get("consolidation_loop")
    if bt is not None:
        try:
            bt.release()  # stops worker, breaks cycle, drops model/tokenizer
        except Exception:
            logger.exception("Error releasing bg-trainer during model release")
    if loop is not None:
        try:
            loop.release()  # drops model + extraction.model + _bg_trainer
        except Exception:
            logger.exception("Error releasing consolidation loop during model release")
    # Now null the dict-entry holders. With the worker dead and the model
    # refs severed on bt/loop, dropping the _state entries drives refcounts
    # to zero even if gc hasn't yet collected the bt/_worker_thread cycle.
    _state["consolidation_loop"] = None
    _state["background_trainer"] = None
    if _state.get("model") is not None:
        try:
            unload_model(_state["model"], _state.get("tokenizer"))
        except Exception:
            logger.exception("Error unloading model during in-process release")
        _state["model"] = None
        _state["tokenizer"] = None
    # Holder 5: the intent-classifier handle (intent.mode=llm). Clearing it
    # is the documented "before a model unload / cloud-only switch" path.
    set_classifier_model(None, None)
    # Belt-and-braces: rerun the cache flush after every holder was
    # cleared. ``unload_model`` already calls gc.collect+empty_cache,
    # but at that moment the loop/trainer/worker may have been live;
    # running it again now that they're gone collapses the allocator
    # pool slack the cycle accumulated while the model was alive.
    safe_empty_cache()
    # === RELPROBE — permanent debug-gated holder diagnostic ===
    # The cheap tripwire (allocated/reserved/mem_get_info_free) runs on every
    # release call so stale holders surface in logs without GPU overhead.
    # The heavy leak branch (module census + memory_summary + referrer walk)
    # runs only when config.debug=True so it is never active in production.
    try:
        import torch as _torch

        if _torch.cuda.is_available():
            _st = _torch.cuda.memory_stats()
            _alloc = _torch.cuda.memory_allocated()
            logger.info(
                "RELPROBE post-release: allocated=%.3f reserved=%.3f inactive_split=%.3f "
                "mem_get_info_free=%.3f GiB alloc_retries=%d",
                _alloc / 2**30,
                _torch.cuda.memory_reserved() / 2**30,
                _st.get("inactive_split_bytes.all.current", 0) / 2**30,
                _torch.cuda.mem_get_info(0)[0] / 2**30,
                _st.get("num_alloc_retries", 0),
            )
            # Tripwire: always-on WARNING when base model was not fully freed.
            # >1 GiB still allocated after a full release means a live object
            # somewhere holds a reference to the base model (see INVARIANT
            # header above for the canonical holder registry).  This fires
            # regardless of config.debug so a future regression is an
            # immediate greppable alarm; the heavy census walk below is still
            # gated on debug to avoid production overhead.
            if _alloc > 2**30:
                logger.warning(
                    "RELPROBE: possible base-model holder leak — %.3f GiB still "
                    "allocated after release (expected ~0; set debug=true for a "
                    "gc.get_referrers holder walk).",
                    _alloc / 2**30,
                )
            # Heavy holder census: gated on config.debug to avoid production overhead.
            # >1 GiB still ALLOCATED after a full release => a live object still
            # holds GPU tensors. Census live modules, print memory_summary, and
            # walk referrers to name the holder.
            _config = _state.get("config")
            _debug = bool(getattr(_config, "debug", False))
            if _alloc > 2**30 and _debug:
                import gc as _gc
                from collections import Counter as _Counter

                _mods = [_o for _o in _gc.get_objects() if isinstance(_o, _torch.nn.Module)]
                logger.info(
                    "RELPROBE LEAK: %.3f GiB allocated after release; "
                    "live nn.Module count=%d top=%s",
                    _alloc / 2**30,
                    len(_mods),
                    _Counter(type(_m).__name__ for _m in _mods).most_common(15),
                )
                logger.info(
                    "RELPROBE memory_summary:\n%s",
                    _torch.cuda.memory_summary(abbreviated=True),
                )
                # --- referrer walk: NAME the live holder of the base model ---
                # Bounded: depth ≤5, total-visit budget ≤300, per-object
                # referrer cap ≤40. Frames are terminal (not recursed into).
                # Must not raise — wrapped by the outer try/except.
                import types as _types

                _roots = [
                    _o
                    for _o in _gc.get_objects()
                    if isinstance(_o, _torch.nn.Module)
                    and (
                        type(_o).__name__.endswith("ForCausalLM")
                        or "PeftModel" in type(_o).__name__
                    )
                ]
                logger.info("RELPROBE roots: %s", [type(_o).__name__ for _o in _roots[:8]])
                _ignore = {id(_roots)}
                _seen: set[int] = set()
                _frontier = [(_r, 0) for _r in _roots[:3]]
                _ignore.add(id(_frontier))
                _ignore.add(id(_seen))
                _budget = 300
                while _frontier and _budget > 0:
                    _obj, _depth = _frontier.pop()
                    _budget -= 1
                    if id(_obj) in _seen or _depth > 5:
                        continue
                    _seen.add(id(_obj))
                    try:
                        _refs = _gc.get_referrers(_obj)
                    except Exception:
                        continue
                    for _ref in _refs[:40]:
                        if id(_ref) in _ignore or id(_ref) in _seen or _ref is _refs:
                            continue
                        if isinstance(_ref, _types.FrameType):
                            logger.info(
                                "RELPROBE HOLDER frame: %s @ %s:%d (depth %d)",
                                _ref.f_code.co_name,
                                _ref.f_code.co_filename,
                                _ref.f_lineno,
                                _depth,
                            )
                            # frames are terminal — do not recurse (entire interpreter)
                        elif _ref is _state:
                            logger.info("RELPROBE HOLDER: _state dict (depth %d)", _depth)
                        elif isinstance(_ref, _types.CellType):
                            logger.info("RELPROBE HOLDER cell (depth %d) — walking up", _depth)
                            _frontier.append((_ref, _depth + 1))
                        elif isinstance(_ref, dict):
                            _keys = [k for k in list(_ref.keys())[:10] if isinstance(k, str)]
                            logger.info(
                                "RELPROBE HOLDER dict (depth %d) keys=%s",
                                _depth,
                                _keys,
                            )
                            _frontier.append((_ref, _depth + 1))
                        elif isinstance(_ref, _torch.nn.Module):
                            _frontier.append((_ref, _depth + 1))
                        else:
                            logger.info(
                                "RELPROBE HOLDER obj (depth %d) type=%s",
                                _depth,
                                type(_ref).__name__,
                            )
                            _frontier.append((_ref, _depth + 1))
                # --- end referrer walk ---
    except Exception:
        logger.exception("RELPROBE failed")
    # === END RELPROBE ===


# GPU-lock timeout for _apply_config_live: long enough for a post-cycle lock
# to release, short enough to avoid wedging the migration handler indefinitely.
_APPLY_CONFIG_LOCK_TIMEOUT_S: float = 60.0


def _apply_config_live() -> dict:
    """Apply the on-disk ``configs/server.yaml`` to the running server in-process.

    Acquires ``gpu_lock_sync`` with a bounded timeout, then:

    1. Re-checks ``_state["consolidating"]`` under the lock (guards against a
       pre-TRIAL cycle still running at accept/rollback time).
    2. Performs a no-op skip when the on-disk config hash equals
       ``_state["config_drift"]["loaded_hash"]`` (the hash of the config that
       was active in memory when the server last booted or accepted a migration).
       This is the rollback case — disk is back to A, memory is A.  Returns
       immediately without GPU churn (correction S-6).

       **WP2 precondition:** the accept handler MUST dispatch
       ``_apply_config_live`` BEFORE refreshing ``config_drift.loaded_hash``
       to config B.  If the refresh happens first, ``disk_hash == loaded_hash``
       fires on the accept path and the apply is incorrectly skipped.
       ``ServerConfig`` has no ``source_path`` attribute — the prior
       implementation that computed ``mem_hash`` via that attribute was always
       falling back to the live path, causing ``disk_hash == mem_hash`` on every
       call (correction B1).
    3. Detects R-PORT / R-PATHS carve deltas (§3.14):

       - ``stt.port`` / ``tts.port`` change → R-PORT carve:
         ``restart_required_reason in {"stt_port_change", "tts_port_change"}``.
         Performs a transient ``socket.bind`` pre-flight on the new port(s); on
         bind failure returns ``restart_eligible=False`` + a "port in use"
         reason.  On bind success returns ``restart_eligible=True`` so the
         CLI can prompt the operator and fire ``restart_hint`` via subprocess
         on consent.  ``_apply_config_live`` itself never calls
         ``_restart_service``.  ``auto_restart_scheduled`` is always ``False``
         (kept for backward compatibility; use ``restart_eligible`` instead).
       - ``paths.sessions`` / ``paths.data`` change → R-PATHS carve:
         short-circuits BEFORE any live reload.
         ``restart_required_reason="paths_change"``, ``restart_eligible=False``.
         Data is NOT migrated automatically; operator must move adapters,
         registry, and sessions to the new root before restarting.
       - Mixed deltas (carve + non-carve fields): apply non-carve fields live
         first, then signal the carve.  A ``paths.*`` mix is always
         manual-restart regardless of other fields.

    4. Calls ``_live_reload_base_model(refresh_config_from_disk=True)`` for
       non-carve fields.
    5. On ``mode==local`` after the rebuild, calls
       ``_set_voice_pipeline_profile("gpu")`` (no-op if already gpu).

    **Caller contract (correction S-4 / §3.12):**
    The CALLER must set the synchronous maintenance guard on the event loop
    BEFORE dispatching this function via ``run_in_executor``:

    .. code-block:: python

        _state["mode"] = "cloud-only"
        _state["cloud_only_reason"] = "live_reload"
        await loop.run_in_executor(None, _apply_config_live)

    This ensures the scheduler's ``mode != "local"`` defer at ``app.py:6831``
    fires before the executor runs.  This function does NOT set the guard
    internally (it runs in an executor thread, not on the event loop).

    Note: ``_live_reload_base_model`` also sets ``mode="cloud-only"`` directly
    as part of the drain sequence; the caller-set guard above is still required
    for the event-loop / scheduler visibility window before the executor thread
    even starts.

    **Lock contract (correction N-1):**
    This function acquires ``gpu_lock_sync`` internally (bounded timeout).
    ``_live_reload_base_model`` must NOT acquire it again — double-acquire on
    the non-reentrant ``threading.Lock`` deadlocks.

    Returns
    -------
    dict
        ``{
            "applied_live": bool,
            "cloud_only_reason": str | None,
            "restart_required_reason": str | None,
            "auto_restart_scheduled": bool,
            "restart_eligible": bool,
            "skipped": str | None,
        }``

        ``restart_eligible`` is ``True`` when an R-PORT carve pre-flighted
        successfully and the CLI may trigger a prompted restart via
        ``restart_hint``.  The server never fires the restart itself.
        ``auto_restart_scheduled`` is always ``False`` (kept for backward
        compatibility).
    """
    import socket as _socket

    from paramem.server.drift import compute_config_hash
    from paramem.server.gpu_lock import gpu_lock_sync

    # ── acquire GPU lock with bounded timeout (N1) ───────────────────────────
    # Guard just the `__enter__` with try/except TimeoutError so the lock
    # cannot leak if entry raises.  The `with` form handles `__exit__` on
    # ALL exits from the body (normal, exception, return) without a
    # separate `finally` that might fire when `__enter__` never succeeded.
    lock_ctx = gpu_lock_sync(timeout=_APPLY_CONFIG_LOCK_TIMEOUT_S)
    try:
        lock_ctx.__enter__()
    except TimeoutError:
        logger.error(
            "_apply_config_live: could not acquire GPU lock within %ss — apply aborted; "
            "config is on disk, restart to apply",
            _APPLY_CONFIG_LOCK_TIMEOUT_S,
        )
        return {
            "applied_live": False,
            "cloud_only_reason": _state.get("cloud_only_reason"),
            "restart_required_reason": "lock_timeout",
            "auto_restart_scheduled": False,
            "restart_eligible": False,
            "skipped": None,
        }

    try:
        # ── re-check consolidating under the lock (correction S-5) ────────────
        if _state.get("consolidating", False):
            logger.warning(
                "_apply_config_live: a consolidation cycle is still running — "
                "apply aborted; config is on disk, restart or retry to apply"
            )
            return {
                "applied_live": False,
                "cloud_only_reason": _state.get("cloud_only_reason"),
                "restart_required_reason": "consolidating",
                "auto_restart_scheduled": False,
                "restart_eligible": False,
                "skipped": None,
            }

        # ── no-op skip (correction S-6 / §3.14) ─────────────────────────────
        config_path_str = _state.get("config_path")
        config_a = _state.get("config")
        live_config_path = Path(config_path_str) if config_path_str else Path("configs/server.yaml")
        if live_config_path.exists():
            disk_hash = compute_config_hash(live_config_path)
            # Compare the on-disk hash against the hash of config A that was
            # captured at boot (or at the last accept).  ``ServerConfig`` has
            # NO ``source_path`` attribute; the in-memory hash is tracked in
            # ``_state["config_drift"]["loaded_hash"]`` (set by
            # ``initial_drift_state`` at boot, refreshed by the accept handler,
            # intentionally NOT refreshed by rollback).  Using the on-disk file
            # as a proxy for the in-memory hash is wrong whenever the on-disk
            # file was swapped by a TRIAL write (disk = B, memory = A) — both
            # paths would hash B and the skip would fire on every accept call.
            #
            # WP2 precondition (document, do not implement here): the accept
            # handler MUST read and capture ``loaded_hash`` BEFORE refreshing it
            # to config B, then dispatch ``_apply_config_live`` with the
            # pre-refresh hash still in ``_state["config_drift"]["loaded_hash"]``.
            # If WP2 refreshes ``loaded_hash`` BEFORE dispatching the apply,
            # disk_hash == loaded_hash would fire on the accept path and the
            # no-op skip would wrongly suppress the rebuild.
            loaded_hash = (_state.get("config_drift") or {}).get("loaded_hash")
            if disk_hash and loaded_hash and disk_hash == loaded_hash:
                logger.info(
                    "_apply_config_live: disk hash == memory hash — no-op skip "
                    "(rollback restored prior config; no GPU churn)"
                )
                return {
                    "applied_live": True,
                    "cloud_only_reason": None,
                    "restart_required_reason": None,
                    "auto_restart_scheduled": False,
                    "restart_eligible": False,
                    "skipped": "no_change",
                }

        # ── config-A-vs-B carve classification (§3.14) ─────────────────────
        # Load config B from disk without committing it yet.
        config_b = None
        if live_config_path.exists():
            try:
                config_b = load_server_config(live_config_path)
            except Exception:
                logger.exception(
                    "_apply_config_live: failed to load config B from disk for carve diff"
                )

        restart_required_reason: str | None = None
        restart_eligible: bool = False

        if config_a is not None and config_b is not None:
            # R-PORT check: stt.port / tts.port delta
            stt_port_changed = getattr(getattr(config_b, "stt", None), "port", None) != getattr(
                getattr(config_a, "stt", None), "port", None
            )
            tts_port_changed = getattr(getattr(config_b, "tts", None), "port", None) != getattr(
                getattr(config_a, "tts", None), "port", None
            )

            # R-PATHS check: paths.sessions / paths.data delta
            paths_a = getattr(config_a, "paths", None)
            paths_b = getattr(config_b, "paths", None)
            sessions_changed = str(getattr(paths_b, "sessions", "")) != str(
                getattr(paths_a, "sessions", "")
            )
            data_changed = str(getattr(paths_b, "data", "")) != str(getattr(paths_a, "data", ""))
            paths_changed = sessions_changed or data_changed

            if paths_changed:
                # R-PATHS carve: short-circuit BEFORE any live reload.
                # A paths.* change cannot be applied live — the session buffer,
                # memory store, and speaker store are all rooted at paths.data.
                # Re-pointing them live while the session buffer stays at the old
                # path creates a split-brain.  Leave config B on disk; the
                # operator must move data to the new root and restart manually.
                logger.info(
                    "_apply_config_live: R-PATHS carve detected (paths.sessions or paths.data "
                    "changed) — short-circuiting BEFORE live reload; manual restart required; "
                    "data not migrated automatically"
                )
                return {
                    "applied_live": False,
                    "cloud_only_reason": _state.get("cloud_only_reason"),
                    "restart_required_reason": "paths_change",
                    "auto_restart_scheduled": False,
                    "restart_eligible": False,
                    "skipped": None,
                }

            elif stt_port_changed or tts_port_changed:
                # R-PORT carve: pre-flight bind check on new port(s).
                reason_parts = []
                if stt_port_changed:
                    reason_parts.append("stt_port_change")
                if tts_port_changed:
                    reason_parts.append("tts_port_change")
                carve_reason = reason_parts[0] if len(reason_parts) == 1 else ",".join(reason_parts)

                # Pre-flight: attempt transient bind on each new port.
                port_in_use_reason: str | None = None
                for _field, _changed, _cfg in [
                    ("stt.port", stt_port_changed, config_b.stt),
                    ("tts.port", tts_port_changed, config_b.tts),
                ]:
                    if not _changed:
                        continue
                    new_port = getattr(_cfg, "port", None)
                    if new_port is None:
                        continue
                    host = getattr(getattr(config_b, "server", None), "host", "0.0.0.0")
                    try:
                        _s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
                        _s.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
                        _s.bind((host, new_port))
                        _s.close()
                    except OSError as _e:
                        port_in_use_reason = f"{_field}={new_port} is not bindable: {_e}"
                        break

                if port_in_use_reason:
                    logger.warning(
                        "_apply_config_live: R-PORT pre-flight failed — %s; "
                        "restart not eligible; free the port and restart manually",
                        port_in_use_reason,
                    )
                    return {
                        "applied_live": False,
                        "cloud_only_reason": _state.get("cloud_only_reason"),
                        "restart_required_reason": carve_reason,
                        "auto_restart_scheduled": False,
                        "restart_eligible": False,
                        "skipped": None,
                        "port_in_use_reason": port_in_use_reason,
                    }

                # Pre-flight passed.  Detect whether this is a pure-port-only delta
                # (no non-carve fields differ) or a mixed delta.
                # Pure port delta: skip the model reload entirely — reloading the model
                # does NOT apply a port change (the listener socket is already bound).
                # Mixed delta: apply non-carve fields live first (fall through to reload),
                # then signal the carve to the caller.
                import dataclasses as _dc  # noqa: PLC0415

                try:
                    # Build port-normalised copies: reset both port fields to a
                    # canonical sentinel (0) on both sides, then compare.  If the
                    # normalised configs are equal, only the port(s) differ.
                    _stt_a = _dc.replace(config_a.stt, port=0)
                    _stt_b = _dc.replace(config_b.stt, port=0)
                    _tts_a = _dc.replace(config_a.tts, port=0)
                    _tts_b = _dc.replace(config_b.tts, port=0)
                    _a_norm = _dc.replace(config_a, stt=_stt_a, tts=_tts_a)
                    _b_norm = _dc.replace(config_b, stt=_stt_b, tts=_tts_b)
                    pure_port_delta = _a_norm == _b_norm
                except Exception:
                    # dataclasses.replace or __eq__ not available (e.g. mock in tests) —
                    # conservatively assume mixed so we don't skip a needed reload.
                    pure_port_delta = False

                logger.info(
                    "_apply_config_live: R-PORT carve (%s) — pre-flight passed; "
                    "restart_eligible=True; pure_port_delta=%s "
                    "(server does NOT self-fire restart — CLI prompts operator)",
                    carve_reason,
                    pure_port_delta,
                )
                restart_required_reason = carve_reason
                restart_eligible = True

                if pure_port_delta:
                    # Pure port: the model reload would not apply the port change
                    # (the listener socket is already bound).  Short-circuit.
                    return {
                        "applied_live": False,
                        "cloud_only_reason": _state.get("cloud_only_reason"),
                        "restart_required_reason": restart_required_reason,
                        "auto_restart_scheduled": False,
                        "restart_eligible": True,
                        "skipped": None,
                    }
                # Mixed delta: fall through to reload; carve signalled in return dict.

        # ── S3: compute retain_sessions / debug delta → rebuild_session_buffer ─
        # Compare config A (in-memory, pre-apply) against config B (on-disk,
        # already loaded above).  If either field changed, the SessionBuffer must
        # be rebuilt so the new retention / debug semantics take effect.
        _rebuild_session_buf = False
        if config_a is not None and config_b is not None:
            retain_a = getattr(getattr(config_a, "consolidation", None), "retain_sessions", None)
            retain_b = getattr(getattr(config_b, "consolidation", None), "retain_sessions", None)
            debug_a = getattr(config_a, "debug", None)
            debug_b = getattr(config_b, "debug", None)
            _rebuild_session_buf = (retain_a != retain_b) or (debug_a != debug_b)
            if _rebuild_session_buf:
                logger.info(
                    "_apply_config_live: retain_sessions or debug changed "
                    "(retain: %r→%r, debug: %r→%r) — session buffer will be rebuilt",
                    retain_a,
                    retain_b,
                    debug_a,
                    debug_b,
                )

        # ── full live apply: reload model + rebuild all components ────────────
        # lock_held=True: _apply_config_live holds gpu_lock_sync() (acquired
        # above at ~4508); the primitive's internal _set_voice_pipeline_profile
        # calls must not re-acquire the non-reentrant threading.Lock.
        _live_reload_base_model(
            refresh_config_from_disk=True,
            rebuild_session_buffer=_rebuild_session_buf,
            lock_held=True,
        )

        if _state.get("mode") == "local":
            # Voice pipeline was set by _build_config_derived_state inside
            # _live_reload_base_model; final no-op profile flip to confirm gpu.
            _set_voice_pipeline_profile("gpu", lock_held=True)
            applied_live = True
        else:
            # Model reload or component rebuild failed; _live_reload_base_model
            # set cloud_only_reason appropriately.
            applied_live = False

        return {
            "applied_live": applied_live,
            "cloud_only_reason": _state.get("cloud_only_reason"),
            "restart_required_reason": restart_required_reason,
            "auto_restart_scheduled": False,  # server never self-fires restart
            "restart_eligible": restart_eligible if applied_live else False,
            "skipped": None,
        }

    finally:
        lock_ctx.__exit__(None, None, None)


async def _apply_config_live_guarded() -> dict:
    """Dispatch ``_apply_config_live`` under the synchronous maintenance guard.

    Sole owner of the guard+dispatch+restore pattern shared by the migration
    accept, base-swap-rollback, and migration_rollback handlers (correction
    S-4).  Sets the cloud-only guard (``mode="cloud-only"``,
    ``cloud_only_reason="live_reload"``) BEFORE dispatching so the scheduler's
    ``mode != "local"`` defer fires during the ~25-30 s GPU reload, runs
    ``_apply_config_live`` in an executor (it blocks), then restores the
    pre-guard mode IFF the apply did NOT transition it.

    A successful reload already set ``mode="local"``; a genuine reload failure
    set a specific ``cloud_only_reason`` (apply_failed / insufficient_vram /
    reload_failed).  Only the untouched guard state
    (cloud-only + ``"live_reload"``) means no transition happened — the running
    server is still serving in its prior mode (carve changes take effect on the
    operator's restart) and must not be left degraded to cloud-only.

    Returns the ``_apply_config_live`` result dict (``applied_live``,
    ``restart_required_reason``, ``restart_eligible``, ...).  Callers that only
    need the mode-restore side effect may ignore the return.
    """
    _prior_mode = _state.get("mode")
    _prior_cloud_only_reason = _state.get("cloud_only_reason")
    _state["mode"] = "cloud-only"
    _state["cloud_only_reason"] = "live_reload"

    loop = asyncio.get_running_loop()
    apply_result = await loop.run_in_executor(None, _apply_config_live)

    if _state.get("mode") == "cloud-only" and _state.get("cloud_only_reason") == "live_reload":
        _state["mode"] = _prior_mode
        _state["cloud_only_reason"] = _prior_cloud_only_reason
    return apply_result


@app.post("/gpu/release", dependencies=[Depends(require_admin)])
async def gpu_release():
    """Release the GPU model in-process; switch to cloud-only mode.

    External GPU consumers (gpu_guard ConfigConsumer, lerobot, etc.) call
    this endpoint to ask paramem to step aside without exiting. Idempotent:
    a server already in cloud-only returns 200 immediately.

    During an in-flight consolidation cycle the server returns 503; the
    caller may retry. Mid-cycle release would corrupt extraction state:
    the cycle re-extracts on next start, but losing partial-cycle work is
    wasteful (the older SIGUSR1-exit release path had the same flaw).

    On success the response is synchronous — by the time the POST returns,
    the model is unloaded and ``_state["mode"]`` is ``"cloud-only"``. The
    auto-reclaim loop is started so paramem will reclaim the GPU once the
    external consumer goes away (cloud-only → local via service restart,
    same code path as ``--defer-model``).

    Returns:
        200 ``{"mode": "cloud-only", "released": bool, "reason": str}``.
            ``released=False`` when the server was already cloud-only.
        503 ``{"error": "consolidating", ...}`` when a cycle is in flight.
    """
    if _state["mode"] == "cloud-only":
        return {
            "mode": "cloud-only",
            "released": False,
            "reason": _state.get("cloud_only_reason"),
        }

    if _state.get("consolidating", False):
        return JSONResponse(
            status_code=503,
            content={
                "error": "consolidating",
                "detail": (
                    "GPU release refused: a consolidation cycle is in flight. "
                    "Retry once /status reports consolidating=false."
                ),
            },
        )

    logger.info("Release requested via /gpu/release — unloading model and switching to cloud-only.")

    _release_base_model_in_process()
    await asyncio.get_running_loop().run_in_executor(None, _set_voice_pipeline_profile, "cpu")

    _state["mode"] = "cloud-only"
    _state["cloud_only_reason"] = "released"

    reclaim_task = _state.get("reclaim_task")
    if reclaim_task is None or reclaim_task.done():
        reclaim_interval = _state["config"].server.reclaim_interval_minutes
        _state["reclaim_task"] = asyncio.create_task(_auto_reclaim_loop(reclaim_interval))

    return {"mode": "cloud-only", "released": True, "reason": "released"}


@app.post("/refresh-ha", dependencies=[Depends(require_admin)])
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
            memory_store=_state["memory_store"],
            ha_graph=ha_graph,
            intent_config=config.intent,
        )

    return {
        "status": "refreshed",
        "entities": ha_graph.entity_count,
        "areas": ha_graph.area_count,
        "verbs": ha_graph.verb_count,
    }


@app.post("/admin/assign-orphans", dependencies=[Depends(require_admin)])
async def admin_assign_orphans(speaker_id: str | None = None):
    """Operator-only: permanently attribute orphan sessions to a single speaker.

    This is an admin/corrective operation, not a debug helper: it claims
    orphan session turns under a single enrolled speaker and persists the
    binding through the next consolidation cycle.  For ephemeral,
    non-polluting probing of a speaker↔transcript combination, use
    ``/debug/probe`` instead.

    **Auth:** gated by the ``require_admin`` dependency — requires an
    admin-scope token (either the shared ``PARAMEM_API_TOKEN`` or a per-user
    token minted with ``--scope admin``).  The endpoint is unreachable in
    auth-OFF mode (no token configured) and with chat-scope tokens, so admin
    actions are never reachable anonymously regardless of ``config.debug``.
    A per-user admin token is accepted, allowing it to be reached without the
    shared env token when ``mobile_pwa.enabled: true``.

    Body: optional ``speaker_id`` query parameter — defaults to the first
    enrolled profile when omitted.  When ``buffer.debug=True`` the on-disk
    session jsonls are rewritten in place; in production-mode (no on-disk
    transcripts) the binding lives only in memory but still flows through
    the next consolidation into adapter weights — the
    durable medium changes with the deployment posture, the operation is
    the same.
    """
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
        # Rewrite the on-disk jsonl when one exists.  In production
        # (debug=false) no jsonls are written, so ``path.exists()`` is
        # False and the rewrite skips — the in-memory mutation above is
        # the only durable medium until consolidation runs.  Mode-agnostic.
        path = buffer.session_dir / f"{conv_id}.jsonl"
        if path.exists():
            with open(path, "w") as f:
                for turn in turns:
                    f.write(json.dumps(turn) + "\n")
    logger.info("admin/assign-orphans: %d sessions → speaker %s (%s)", claimed, sname, sid)
    return {"status": "ok", "claimed": claimed, "speaker": sname, "speaker_id": sid}


@app.post(
    "/speaker/forget",
    response_model=SpeakerForgetResponse,
    dependencies=[Depends(require_admin)],
)
async def speaker_forget(request: SpeakerForgetRequest):
    """Forget a speaker: remove their profile, mark their keys stale, discard pending sessions.

    This is the ``"mark_stale"`` strategy — registry-level erasure that is safe
    on a live store.  The speaker's adapter weights decay naturally through future
    training cycles; this endpoint does not trigger retraining.

    Steps
    -----
    1. **Locate the speaker's indexed-memory keys** via
       :func:`~paramem.memory.persistence.keys_for_speaker` on the live merged
       graph held by the :class:`~paramem.training.consolidation.ConsolidationLoop`.

    2. **Remove keys from every per-tier KeyRegistry** — both in-memory (so
       keyed recall no longer serves them) and on disk (so the next restart
       does not resurrect them).  Simhash entries for the same keys are dropped
       from the in-memory store and saved to disk so the SimHash gate cannot
       verify them at inference time.

    3. **Remove the speaker profile** from
       :class:`~paramem.server.speaker.SpeakerStore` (persisted immediately).

    4. **Discard any pending sessions** for the speaker from
       :class:`~paramem.server.session_buffer.SessionBuffer`.

    Args:
        request: :class:`SpeakerForgetRequest` with ``speaker_id`` and
            optional ``strategy`` (only ``"mark_stale"`` is implemented).

    Returns:
        :class:`SpeakerForgetResponse` reporting what was removed.

    Raises:
        HTTPException 503: When the consolidation loop is not initialised
            (server not yet ready or in cloud-only mode without a loaded model).
        HTTPException 400: When ``strategy`` is not ``"mark_stale"``.

    Note
    ----
    ``discard_interim`` strategy (discard the whole interim slot) is out of
    scope for this revision.  Extend ``SpeakerForgetRequest.strategy`` and add
    a handler branch here when that strategy is needed.
    """
    from paramem.memory.persistence import keys_for_speaker as _keys_for_speaker
    from paramem.memory.persistence import save_registry as _save_simhash_registry

    if request.strategy != "mark_stale":
        raise HTTPException(
            status_code=400,
            detail={
                "status": "unsupported_strategy",
                "detail": f"Strategy {request.strategy!r} is not implemented. "
                "Only 'mark_stale' is supported.",
            },
        )

    loop = _state.get("consolidation_loop")
    if loop is None:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "detail": "Consolidation loop is not initialised. "
                "The server may be in cloud-only mode or not yet started.",
            },
        )

    config = _state["config"]
    speaker_id = request.speaker_id

    # Step 1: locate keys for this speaker via the live merged graph.
    graph = loop.merger.graph
    keys: set[str] = _keys_for_speaker(graph, speaker_id)
    stale_keys: list[str] = sorted(keys)

    # Step 2: remove keys from every per-tier KeyRegistry (in-memory + disk).
    # Also drop their simhash entries so the SimHash gate cannot verify them.
    if stale_keys:
        for tier_name in loop.store.tiers_with_registry():
            registry = loop.store.registry(tier_name)
            if registry is None:
                continue
            tier_keys_to_remove = [k for k in stale_keys if k in registry]
            if not tier_keys_to_remove:
                continue
            for key in tier_keys_to_remove:
                registry.remove(key)
            # Persist the updated KeyRegistry to disk.
            # Determine the disk path: main tiers live at
            # <adapter_dir>/<tier>/indexed_key_registry.json; interim tiers
            # follow the same pattern (tier_name encodes the full slot name).
            tier_reg_path = config.adapter_dir / tier_name / "indexed_key_registry.json"
            registry.save(tier_reg_path)
            logger.info(
                "speaker/forget: removed %d key(s) from KeyRegistry tier %s for speaker %s",
                len(tier_keys_to_remove),
                tier_name,
                speaker_id,
            )

        # Drop simhash entries for the speaker's keys and persist simhash registries.
        for tier_name in ("episodic", "semantic", "procedural"):
            simhashes = loop.store.simhashes_in_tier(tier_name)
            tier_keys_in_simhash = [k for k in stale_keys if k in simhashes]
            if not tier_keys_in_simhash:
                continue
            for key in tier_keys_in_simhash:
                simhashes.pop(key, None)
            _save_simhash_registry(
                simhashes,
                config.adapter_dir / tier_name / "simhash_registry.json",
            )
            logger.info(
                "speaker/forget: removed %d simhash entry(ies) from tier %s for speaker %s",
                len(tier_keys_in_simhash),
                tier_name,
                speaker_id,
            )

    # Step 3: remove the speaker profile.
    speaker_store = _state.get("speaker_store")
    removed_speaker = False
    if speaker_store is not None:
        removed_speaker = speaker_store.remove(speaker_id)

    # Step 4: discard pending sessions attributed to this speaker.
    buffer = _state["session_buffer"]
    speaker_conv_ids = [
        conv_id
        for conv_id, session_meta in buffer._sessions.items()
        if session_meta.get("speaker_id") == speaker_id
    ]
    if speaker_conv_ids:
        buffer.discard_sessions(speaker_conv_ids)
    discarded_sessions = speaker_conv_ids

    logger.info(
        "speaker/forget: speaker=%s keys=%d profile_removed=%s sessions=%d",
        speaker_id,
        len(stale_keys),
        removed_speaker,
        len(discarded_sessions),
    )
    return SpeakerForgetResponse(
        removed_speaker=removed_speaker,
        stale_keys=stale_keys,
        discarded_sessions=discarded_sessions,
    )


# --------------------------------------------------------------------------
# Debug probe endpoint — non-polluting speaker↔transcript probe.
# Gated by config.debug.  Bypasses _resolve_speaker so the chat handler
# can be exercised against an enrolled speaker without binding the
# speaker to any session_buffer entry: no mutation of buffer._turns or
# buffer._sessions, no jsonl rewrite on disk, no buffer.append on the
# conversation_id, no consolidation impact.  Pure single-call probe in
# RAM only.
#
# Generic by design — additional probe modes (live PA voice probe,
# document ingest probe by an admin) may share this endpoint or extend
# it later.  Today only the chat-style invocation is wired.
# --------------------------------------------------------------------------


class DebugProbeRequest(BaseModel):
    """Probe the chat handler with explicit speaker_id injection."""

    text: str
    speaker_id: str  # explicit; bypasses _resolve_speaker
    conversation_id: str = "debug-probe"
    history: list[dict] | None = None


@app.post("/debug/probe", response_model=ChatResponse, dependencies=[Depends(require_admin)])
async def debug_probe(request: DebugProbeRequest):
    """Single-call /chat-equivalent probe with operator-supplied speaker_id.

    Returns ``forbidden_not_debug`` when ``config.debug=false``.  An
    unknown ``speaker_id`` returns 404.  Mirrors the dispatch shape of
    ``/chat`` (cloud-only branch + local branch with gpu_lock and
    background-trainer pause), minus the buffer.append, greeting flow,
    STT language detection, and ``tracker.record`` side-effects.

    Text-side language detection (``lang_id.resolve_text_language``) is run
    here because it is side-effect-free: it reads only the request text and
    the server config, mutates no state, and does not call
    ``tracker.record``.  The resolved language is forwarded to both the
    cloud-only and local dispatch branches so non-English probe texts are
    handled correctly.
    """
    # Count as a /chat-equivalent turn for the idle-debounce gate so operator
    # probe calls do not trigger consolidation mid-session.
    _state["last_chat_monotonic"] = time.monotonic()
    config = _state["config"]
    if not getattr(config, "debug", False):
        return JSONResponse({"status": "forbidden_not_debug"}, status_code=403)

    store = _state.get("speaker_store")
    if store is None:
        return JSONResponse({"status": "not_ready"}, status_code=503)

    speaker_name = store.get_name(request.speaker_id)
    if speaker_name is None:
        return JSONResponse(
            {"status": "speaker_not_found", "speaker_id": request.speaker_id},
            status_code=404,
        )

    # Side-effect-free text-side language detection.  STT detection and
    # tracker.record are /chat-only side-effects and are intentionally omitted.
    from paramem.server import lang_id as _lang_id

    detected_language, _ = _lang_id.resolve_text_language(request.text, config.text_lang_detection)

    # Cloud-only mode mirrors /chat dispatch — no GPU lock, no model.
    if _state["mode"] == "cloud-only":
        cloud_result = _cloud_only_route(
            text=request.text,
            speaker=speaker_name,
            history=request.history,
            config=config,
            router=_state.get("router"),
            ha_client=_state.get("ha_client"),
            sota_agent=_state.get("sota_agent"),
            language=detected_language,
        )
        return ChatResponse(text=cloud_result.text, escalated=True, speaker=speaker_name)

    # Local mode — abort BG trainer + acquire gpu_lock, mirroring /chat.
    bg_trainer = _state.get("background_trainer")
    if bg_trainer is not None and bg_trainer.is_training:
        _abort_timeout = config.consolidation.abort_quiesce_timeout_s
        aborted = bg_trainer.abort_for_inference(timeout=_abort_timeout)
        if not aborted:
            bg_trainer._shutdown_requested = True
            bg_trainer._is_training = False

    from paramem.server.gpu_lock import gpu_lock

    async with gpu_lock():
        loop = asyncio.get_running_loop()
        result: ChatResult = await loop.run_in_executor(
            None,
            lambda: handle_chat(
                text=request.text,
                conversation_id=request.conversation_id,
                speaker=speaker_name,
                speaker_id=request.speaker_id,
                history=request.history,
                model=_state["model"],
                tokenizer=_state["tokenizer"],
                config=config,
                router=_state["router"],
                sota_agent=_state.get("sota_agent"),
                ha_client=_state.get("ha_client"),
                language=detected_language,
                effective_mode=_state.get("effective_mode"),
                memory_store=_state["memory_store"],
            ),
        )

    return ChatResponse(
        text=result.text,
        escalated=result.escalated,
        speaker=speaker_name,
    )


# --------------------------------------------------------------------------
# Debug recall endpoint — direct adapter probe.
# Bypasses QueryRouter and _probe_and_reason entirely: activates the
# requested adapter (or disables all adapters when adapter="none"),
# sends the caller's prompt verbatim through the model, and returns the
# raw output.  No speaker scoping, no per-key enumeration, no bullet-
# context reasoning step.  Mirrors /debug/probe's side-effect contract:
# no buffer mutation, no jsonl write, no consolidation impact.
#
# Use case: testing whether a triple-format-trained adapter responds to
# natural-language questions without the targeted-probe pipeline doing
# the work — i.e. measuring direct recall from adapter weights as a
# distinct capability from cache-driven enumerate-then-reason.
# --------------------------------------------------------------------------


class DebugRecallRequest(BaseModel):
    """Direct adapter recall probe with caller-supplied prompt."""

    text: str
    adapter: str  # adapter name in model.peft_config, or "none" to disable all
    system_prompt: str | None = None  # None → paramem.training.dataset.SYSTEM_PROMPT
    max_new_tokens: int = 256
    temperature: float = 0.0


class DebugRecallResponse(BaseModel):
    """Raw model output from a direct adapter probe."""

    text: str
    adapter_active: str  # echoes adapter; "disabled" when adapter="none"
    parsed_entry: dict | None
    latency_ms: int
    adapter_available: list[str]


@app.post(
    "/debug/recall",
    response_model=DebugRecallResponse,
    dependencies=[Depends(require_admin)],
)
async def debug_recall(request: DebugRecallRequest):
    """Run *request.text* through the model with *request.adapter* active.

    Bypasses the chat handler, the router, and the reason-over-bullets
    step.  Returns the raw model output, an attempted JSON parse via
    :func:`paramem.memory.entry.parse_recalled_entry`, and the active
    adapter name for the call.

    Returns ``forbidden_not_debug`` (403) when ``config.debug=false``.
    Returns ``not_ready`` (503) when the local model isn't loaded or the
    server is in ``cloud-only`` mode.  Returns ``unknown_adapter`` (400)
    with the available list when *adapter* is not in ``model.peft_config``
    and is not the literal ``"none"``.
    """
    config = _state["config"]
    if not getattr(config, "debug", False):
        return JSONResponse({"status": "forbidden_not_debug"}, status_code=403)

    if _state.get("mode") == "cloud-only":
        return JSONResponse(
            {"status": "not_ready", "detail": "cloud-only mode has no local model to probe"},
            status_code=503,
        )

    model = _state.get("model")
    tokenizer = _state.get("tokenizer")
    if model is None or tokenizer is None:
        return JSONResponse({"status": "not_ready"}, status_code=503)

    from peft import PeftModel

    available = sorted(model.peft_config.keys()) if isinstance(model, PeftModel) else []
    if request.adapter != "none" and request.adapter not in available:
        return JSONResponse(
            {
                "status": "unknown_adapter",
                "requested": request.adapter,
                "available": available + ["none"],
            },
            status_code=400,
        )

    bg_trainer = _state.get("background_trainer")
    if bg_trainer is not None and bg_trainer.is_training:
        _abort_timeout = config.consolidation.abort_quiesce_timeout_s
        aborted = bg_trainer.abort_for_inference(timeout=_abort_timeout)
        if not aborted:
            bg_trainer._shutdown_requested = True
            bg_trainer._is_training = False

    from paramem.evaluation.recall import generate_answer
    from paramem.memory.entry import parse_recalled_entry
    from paramem.models.loader import adapt_messages, switch_adapter
    from paramem.server.gpu_lock import gpu_lock
    from paramem.training.dataset import SYSTEM_PROMPT

    system_prompt = request.system_prompt if request.system_prompt is not None else SYSTEM_PROMPT
    messages = adapt_messages(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.text},
        ],
        tokenizer,
    )
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _run() -> tuple[str, str, int]:
        # Capture prior active adapter so we can restore.  PEFT exposes both
        # `active_adapter` (legacy single name) and `active_adapters` (list).
        # Falling back through both keeps us robust to model not-yet-PEFT-wrapped
        # cases — handled above by the unknown_adapter gate.
        prior: list[str] = []
        raw_active = getattr(model, "active_adapter", None)
        if isinstance(raw_active, list):
            prior = list(raw_active)
        elif isinstance(raw_active, str):
            prior = [raw_active]

        # Defensive: handle_chat (inference.py:305) disables gradient
        # checkpointing at the top of every chat turn.  Replicate so this
        # endpoint also avoids the silent KV-cache disable.
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()

        t0 = time.monotonic()
        try:
            if request.adapter == "none":
                if isinstance(model, PeftModel):
                    with model.disable_adapter():
                        raw = generate_answer(
                            model,
                            tokenizer,
                            prompt,
                            max_new_tokens=request.max_new_tokens,
                            temperature=request.temperature,
                        )
                else:
                    raw = generate_answer(
                        model,
                        tokenizer,
                        prompt,
                        max_new_tokens=request.max_new_tokens,
                        temperature=request.temperature,
                    )
                adapter_active_label = "disabled"
            else:
                switch_adapter(model, request.adapter)
                raw = generate_answer(
                    model,
                    tokenizer,
                    prompt,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                )
                adapter_active_label = request.adapter
        finally:
            # Restore prior adapter so the next /chat starts predictable.
            if prior and isinstance(model, PeftModel):
                switch_adapter(model, prior[0] if len(prior) == 1 else prior)

        latency_ms = int((time.monotonic() - t0) * 1000)
        return raw, adapter_active_label, latency_ms

    async with gpu_lock():
        loop = asyncio.get_running_loop()
        raw_text, adapter_label, latency_ms = await loop.run_in_executor(None, _run)

    return DebugRecallResponse(
        text=raw_text,
        adapter_active=adapter_label,
        parsed_entry=parse_recalled_entry(raw_text),
        latency_ms=latency_ms,
        adapter_available=available + ["none"],
    )


# --------------------------------------------------------------------------
# Debug dump endpoint — zero-GPU read of the in-memory MemoryStore.
# Walks ``_state["memory_store"].iter_entries()`` and returns the canonical
# entry payload per (tier, key).  No model invocation, no adapter switch,
# no per-key generate — pure cache read.  ~5 ms for 250 entries vs ~8 min
# for the equivalent per-key /debug/recall sweep on this hardware.
#
# Use when the goal is "what does this adapter hold" (registry inventory
# for scoring, cross-model A/B setup, content audit).  Use /debug/recall
# when the goal is "what does the model say given a custom prompt"
# (weight-recall behavior under natural-language probes).
# --------------------------------------------------------------------------


class DebugDumpResponse(BaseModel):
    """Flat list of every content entry in the live ``MemoryStore``.

    ``entries``/``total`` reflect the inference CONTENT cache (``_entries``)
    only — empty under ``inference.preload_cache=False``, which is correct.
    ``bookkeeping_total`` reflects ``_bookkeeping`` (speaker provenance for
    every registered key) — populated regardless of preload setting.
    """

    entries: list[dict]
    total: int
    tiers: dict[str, int]  # tier name → entry count
    bookkeeping_total: int  # keys with speaker/cycle provenance in _bookkeeping


@app.get("/debug/dump", response_model=DebugDumpResponse, dependencies=[Depends(require_admin)])
async def debug_dump():
    """Dump every (tier, key, entry) the live ``MemoryStore`` holds.

    Returns ``forbidden_not_debug`` (403) when ``config.debug=false``.
    Returns ``not_ready`` (503) when the memory store isn't constructed
    yet (early-boot, cloud-only with no preload).  When
    ``inference.preload_cache=false`` the store is empty by design and
    this endpoint returns an empty list — that's a correct read, not
    an error.  ``bookkeeping_total`` will still be non-zero when the
    router has speaker provenance loaded from ``key_metadata.json``.

    Each entry dict is the entry payload as stored, with ``tier`` and
    ``key`` fields added inline for flat consumption.
    """
    config = _state["config"]
    if not getattr(config, "debug", False):
        return JSONResponse({"status": "forbidden_not_debug"}, status_code=403)

    store = _state.get("memory_store")
    if store is None:
        return JSONResponse({"status": "not_ready"}, status_code=503)

    entries: list[dict] = []
    tiers: dict[str, int] = {}
    for tier, key, entry in store.iter_entries():
        row = {"tier": tier, "key": key, **entry}
        entries.append(row)
        tiers[tier] = tiers.get(tier, 0) + 1

    return DebugDumpResponse(
        entries=entries,
        total=len(entries),
        tiers=tiers,
        bookkeeping_total=store.bookkeeping_count(),
    )


# --------------------------------------------------------------------------
# Calibration endpoints — opt-in dev tool for live prompt iteration.
# Gated by consolidation.calibrate_endpoint_enabled (default False).
# Each endpoint is a thin wrapper around the existing pipeline helper for
# that stage — injection of prompts/params, capture of output.  Stages
# are stop points: the calibration client chains them, and "skip stage X"
# means don't call X's endpoint.  No call modifies weights or writes
# production data on disk.  See paramem/server/calibrate.py.
# --------------------------------------------------------------------------


@app.post("/calibrate/extract", dependencies=[Depends(require_admin)])
async def calibrate_extract_route(req: calibrate_module.CalibrateExtractRequest):
    return calibrate_module.calibrate_extract(_state, req)


@app.post("/calibrate/anonymize", dependencies=[Depends(require_admin)])
async def calibrate_anonymize_route(req: calibrate_module.CalibrateAnonymizeRequest):
    return calibrate_module.calibrate_anonymize(_state, req)


@app.post("/calibrate/plausibility", dependencies=[Depends(require_admin)])
async def calibrate_plausibility_route(req: calibrate_module.CalibratePlausibilityRequest):
    return calibrate_module.calibrate_plausibility(_state, req)


@app.post(
    "/scheduled-tick",
    response_model=ConsolidateResponse,
    dependencies=[Depends(require_admin)],
)
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


@app.post("/consolidate", response_model=ConsolidateResponse, dependencies=[Depends(require_admin)])
async def consolidate():
    """Trigger consolidation manually — alias for the scheduled tick.

    Manual ``/consolidate`` and the systemd-driven ``/scheduled-tick`` share
    the same dispatcher: the gate decides between full-cycle (collapse
    interim slots into main) and interim-cycle (extract pending sessions
    into a new ``episodic_interim_<stamp>`` slot) based on the
    ``window_stamp`` recorded on the lex-max main episodic slot. There is
    no separate "force-full" semantic — operators that genuinely need to
    re-trigger a full cycle can clear the latest main slot's window_stamp
    and call this endpoint, but in normal operation the gate already
    decides correctly.

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

    status = _maybe_trigger_scheduled_consolidation()
    return ConsolidateResponse(status=status)


# --- Document ingest endpoints ---


@app.post(
    "/ingest-sessions",
    response_model=IngestSessionsResponse,
    dependencies=[Depends(require_admin)],
)
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
            # doc_title is persisted here (and in the registry record below)
            # for observability and dedup only — extraction binds the narrator
            # via build_speaker_context, so extract_session has no doc_title
            # parameter by design.
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


@app.post(
    "/ingest-sessions/cancel",
    response_model=IngestCancelResponse,
    dependencies=[Depends(require_admin)],
)
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


@app.post(
    "/migration/preview",
    response_model=PreviewResponse,
    dependencies=[Depends(require_admin)],
)
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
            parsed_live=live_yaml,
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
        parsed_live=live_yaml,
    )
    _state["migration"] = stash

    payload = render_preview_response(stash, pre_flight_fail=None)
    return PreviewResponse(**payload)


@app.post(
    "/migration/cancel",
    response_model=MigrationCancelResponse,
    dependencies=[Depends(require_admin)],
)
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


@app.post(
    "/migration/confirm",
    response_model=ConfirmResponse,
    dependencies=[Depends(require_admin)],
)
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

    from paramem.server.migration import (
        TrialStash,
        _build_mode_switch_block,
        initial_migration_state,
    )

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
        # R2: Reject confirm if a base-swap orchestration is actively running.
        # In practice the state=TRIAL check above fires first, but this guard
        # is explicit in case the state flag lags the base_swap_active flag.
        if migration.get("base_swap_active", False):
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "base_swap_active",
                    "message": (
                        "A base-swap migration is actively running. "
                        "Wait for it to complete (or fail) before confirming."
                    ),
                },
            )

        # Snapshot the STAGING stash fields we need.
        candidate_path_str = migration.get("candidate_path", "")
        candidate_hash = migration.get("candidate_hash", "")

        # Steps 2–4 are wrapped in try/finally so the lock is always released
        # even on partial failure (Correction 1).
        now_iso = datetime.now(timezone.utc).isoformat()

        # --- Pure mode-switch fast path: skip the trial entirely ---
        # A migration whose ONLY change is consolidation.mode (simulate↔train)
        # is a persistence-venue switch owned by the active-store rebuild
        # (active_store_migration), NOT the generic trial.  The rebuild runs
        # per-tier at the next /consolidate with a 1.0 recall gate and
        # source-mode fallback.  Running a trial here would force-train and
        # (for simulate→train) train twice.  So: swap the config live, arm the
        # rebuild (mirroring the lifespan path — NO model reload, because the
        # mode affects consolidation persistence only, not the base model /
        # adapters / router / inference), and drop straight back to LIVE.
        tier_diff = migration.get("tier_diff", [])
        is_pure_mode_switch = (
            len(tier_diff) == 1 and tier_diff[0]["dotted_path"] == "consolidation.mode"
        )
        is_base_swap = any(r["dotted_path"] == "model" for r in tier_diff)
        if is_pure_mode_switch:
            # Integrity gate UP-FRONT — before any mutation — mirroring the
            # base-swap branch.  On a corrupt store the arm would refuse anyway
            # (see _arm_active_store_migration), but by then the config is
            # already renamed → config/store divergence.  Raise 409 here so
            # no mutation occurs on the failure path.
            from paramem.backup.integrity import (
                verify_infrastructure_integrity as _verify_integrity,
            )

            _daily_ok_ms = _state.get("daily_loadable", False)
            _ms_integrity = _verify_integrity(
                _state["config"],
                store=_state.get("consolidation_loop", None)
                and getattr(_state.get("consolidation_loop"), "store", None),
                daily_loadable=_daily_ok_ms,
            )
            if not _ms_integrity.ok:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "error": "integrity_failure",
                        "failing_files": [c.to_dict() for c in _ms_integrity.failures],
                    },
                )

            pre_trial_hash = ""
            if live_config_path.exists():
                pre_trial_hash = _hashlib.sha256(live_config_path.read_bytes()).hexdigest()
            try:
                _rename_config(Path(candidate_path_str), live_config_path)
                dir_fd = os.open(str(live_config_path.parent), os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                except OSError:
                    pass
                finally:
                    os.close(dir_fd)
            except Exception as exc:
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": "config_swap_failed",
                        "message": f"Atomic config rename failed: {exc}",
                    },
                ) from exc

            # Refresh _state["config"] to the new mode and arm the per-tier
            # rebuild (lifespan-mirror; no model reload), then return to LIVE.
            _refresh_config_from_disk_into_state()
            _state["migration"] = initial_migration_state()

            return ConfirmResponse(
                state="LIVE",
                trial_started_at=now_iso,
                pre_trial_config_sha256=pre_trial_hash,
                candidate_config_sha256=candidate_hash,
                backup_paths={},
                trial_adapter_dir="",
                trial_graph_dir="",
                mode_switch=_build_mode_switch_block(
                    tier_diff[0]["old_value"], tier_diff[0]["new_value"]
                ),
            )

        # --- Base-swap path: Phase A background task ---
        # When the candidate changes the base model, we arm a background task
        # that (1) writes a full bundle backup, (2) runs Phase A (train→simulate
        # active-store migration to reconstruct keyed facts into per-tier
        # graph.json and delete the adapter weight slots), (3) atomically renames
        # the candidate config over the live config, and (4) updates the marker
        # to phaseA_done + sets migration status to restart_required.
        if is_base_swap:
            # Integrity gate: refuse base-swap when the store is corrupt.
            from paramem.backup.integrity import (
                verify_infrastructure_integrity as _verify_integrity,
            )

            _daily_ok_for_swap = _state.get("daily_loadable", False)
            _swap_integrity = _verify_integrity(
                _state["config"],
                store=_state.get("consolidation_loop", None)
                and getattr(_state.get("consolidation_loop"), "store", None),
                daily_loadable=_daily_ok_for_swap,
            )
            if not _swap_integrity.ok:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "error": "integrity_failure",
                        "failing_files": [c.to_dict() for c in _swap_integrity.failures],
                    },
                )

            from paramem.server.config import MODEL_REGISTRY

            # Read the candidate's model alias and resolve it via MODEL_REGISTRY.
            parsed_candidate = migration.get("parsed_candidate", {})
            candidate_model_alias = parsed_candidate.get("model", "")
            if candidate_model_alias not in MODEL_REGISTRY:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "error": "unknown_model",
                        "message": (
                            f"Candidate model alias {candidate_model_alias!r} is not in "
                            f"MODEL_REGISTRY. Available: {list(MODEL_REGISTRY.keys())}"
                        ),
                    },
                )
            candidate_model_config = MODEL_REGISTRY[candidate_model_alias]
            predicted = predict_base_bytes(
                candidate_model_config,
                nf4_disk_to_runtime_factor=_state["config"].vram.nf4_disk_to_runtime_factor,
            )
            if predicted is None:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "error": "model_not_cached",
                        "message": (
                            f"Model '{candidate_model_alias}' "
                            f"({candidate_model_config.model_id}) is not in the "
                            "HuggingFace cache. Download it first so the pre-load "
                            "VRAM assessment can run. Example: "
                            f"huggingface-cli download {candidate_model_config.model_id}"
                        ),
                    },
                )

            # Resolve old model alias from the live config.
            live_config = _state.get("config")
            old_model_alias = getattr(live_config, "model_name", "") if live_config else ""

            # Set state to TRIAL immediately (async task updates it further).
            _state["migration"]["state"] = "TRIAL"
            _state["migration"]["trial"] = TrialStash(
                started_at=now_iso,
                pre_trial_config_sha256="",
                candidate_config_sha256=candidate_hash,
                backup_paths={},
                trial_adapter_dir="",
                trial_graph_dir="",
                gates={"status": "pending"},
            )
            _state["migration"]["recovery_required"] = []

            # Kick off Phase A as a background task.
            asyncio.create_task(
                _run_base_swap_orchestration(
                    candidate_path_str=candidate_path_str,
                    live_config_path=live_config_path,
                    state_dir=state_dir,
                    backups_root=backups_root,
                    old_model=old_model_alias,
                    new_model=candidate_model_alias,
                    started_at=now_iso,
                    candidate_hash=candidate_hash,
                )
            )

            return ConfirmResponse(
                state="TRIAL",
                trial_started_at=now_iso,
                pre_trial_config_sha256="",
                candidate_config_sha256=candidate_hash,
                backup_paths={},
                trial_adapter_dir="",
                trial_graph_dir="",
                base_swap=True,
            )

        # --- Step 2: snapshot pre-trial hash, write the config backup ---
        # Config is the ONLY required pre-migration artifact.  The migration's
        # sole live mutation is the atomic config swap in step 4: the trial
        # consolidation writes its adapters / registry / graph into isolated
        # dirs (_build_trial_loop) and never marks sessions consolidated, so
        # rollback (and crash recovery) only ever restore the config.  This
        # holds in BOTH persistence modes — train (weights) and simulate
        # (graph.json) both write only to the trial-isolated output paths.
        # Backing up graph / registry here would be dead writes; nothing reads
        # backup_paths["graph"] / ["registry"].
        pre_trial_hash = ""
        if live_config_path.exists():
            pre_trial_hash = _hashlib.sha256(live_config_path.read_bytes()).hexdigest()

        written_slots: list[Path] = []
        try:
            config_bytes = live_config_path.read_bytes() if live_config_path.exists() else b""
            config_slot = backup_write(
                ArtifactKind.CONFIG,
                config_bytes,
                meta_fields={"tier": "pre_migration", "pre_trial_hash": pre_trial_hash},
                base_dir=backups_root / "config",
            )
            written_slots.append(config_slot)

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
            backup_paths={"config": str(config_slot.resolve())},
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
            backup_paths={"config": str(config_slot.resolve())},
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
            backup_paths={"config": str(config_slot.resolve())},
            trial_adapter_dir=trial_adapter_dir,
            trial_graph_dir=trial_graph_dir,
        )


async def _run_trial_consolidation() -> None:
    """Run a trial consolidation cycle in the background.

    Acquires the GPU lock, reloads config from the newly-active server.yaml,
    builds a trial ConsolidationLoop with overrides (mode=train, paths →
    state/trial_adapter/, persist_graph=True on state/trial_graph/), and
    calls ``_run_extraction_phase`` with ``mark_callback=lambda _: None``.

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

        # Reload config from the newly-active candidate.  The trial runs in the
        # candidate's CONFIGURED consolidation.mode — no force-train override.
        # Pure mode-switch migrations never reach this coroutine (they are
        # applied directly by migration_confirm and rebuilt by the active-store
        # migration); only non-mode changes run a trial here, where the live
        # mode is unchanged so the trial faithfully reflects it.
        trial_config = load_server_config(live_config_path)

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
                    # Temporarily override _state for the trial context so that
                    # _run_extraction_phase reads the trial config and the live
                    # session_buffer, then restore _state after the call.
                    # Trial keeps its own loop (separate adapter/graph dirs).
                    prior_config = _state.get("config")
                    prior_ha_client = _state.get("ha_client")
                    prior_speaker_store = _state.get("speaker_store")
                    _state["config"] = trial_config
                    # ha_context and speaker_store are passed implicitly through _state
                    # ha_client.get_home_context() is called inside _run_extraction_phase;
                    # inject the pre-fetched context via a shim that returns it directly.
                    if ha_context is not None:

                        class _HAContextShim:
                            """Stub used by trial-migration when no live HA client is available.

                            Mirrors only the surface that ``_run_extraction_phase`` reads via
                            ``ha_client.get_home_context()``.  The pre-fetched ``ha_context``
                            is returned directly so no real HA API call is issued during the
                            trial run.
                            """

                            def get_home_context(self):
                                return ha_context

                        _state["ha_client"] = _HAContextShim()
                    else:
                        _state["ha_client"] = None
                    _state["speaker_store"] = speaker_store
                    try:
                        with gpu_lock_sync():
                            # Trial path: mark_callback=lambda _: None so sessions stay
                            # pending (spec L364) and /migration/rollback can restore queue.
                            return _run_extraction_phase(
                                loop,
                                mark_callback=lambda _: None,
                            )
                    finally:
                        _state["config"] = prior_config
                        _state["ha_client"] = prior_ha_client
                        _state["speaker_store"] = prior_speaker_store

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
                recall_probe_batch_size=trial_config.consolidation.recall_probe_batch_size,
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


async def _run_base_swap_orchestration(
    *,
    candidate_path_str: str,
    live_config_path: Path,
    state_dir: Path,
    backups_root: Path,
    old_model: str,
    new_model: str,
    started_at: str,
    candidate_hash: str,
    resume_phase: str = "",
) -> None:
    """Run the full base-swap orchestration: Phase A → reload → Phase B → done.

    This single coroutine owns the entire base-model-swap lifecycle.  It is
    launched as a background ``asyncio.Task`` from ``migration_confirm`` and
    runs to completion (or a retryable-deferred checkpoint) without requiring
    a server restart.

    Sequence
    --------
    1. **Bundle backup** — full snapshot of Mistral weights + per-tier adapter
       dirs + registry + speaker_profiles, written before any mutations.
       Rollback anchor.  **Written exactly once** at fresh start; resume paths
       read ``bundle_slot`` from the existing marker and NEVER call
       ``write_bundle`` again.
    2. **Phase A** — arm the active-store ``train→simulate`` migration state
       file and submit it to a fresh ``BackgroundTrainer`` worker (holds GPU
       lock during execution).  Reconstructs keyed facts from live Mistral
       weights into per-tier ``graph.json`` files, then deletes the adapter
       weight slots.  Marker → ``phaseA_done``.  Config atomically renamed to
       the candidate (Qwen3) variant.
    3. **In-process reload** — two sub-cases based on ``resume_phase``:

       - **Fresh start** (``resume_phase == ""``): drain and reload the base via
         the existing ``/gpu/release`` + ``/gpu/acquire`` primitives.
         ``gpu_release`` drains BOTH the base model AND the voice pipeline to
         cloud-only; ``gpu_acquire`` then loads the renamed-config base on the
         clean GPU, so the VRAM gate sees accurate free bytes and the reload
         fits in-process without a restart.
         If the reload was **deferred** (insufficient VRAM), set gates to
         ``reload_deferred`` and return.  ``/gpu/acquire`` re-launches
         (``resume_phase="phaseA_done"``) when VRAM frees.

       - **Resume** (``resume_phase == "phaseA_done"``): the server was restarted
         after writing the phaseA_done marker; boot already loaded the new
         config from the renamed server.yaml.  Skip ``_apply_config_live``.
         Verify the new model is resident (``mode=="local"`` AND
         ``config.model_name == new_model``); if not, set ``reload_deferred``
         and return for retry.

    4. **Phase B** — re-create ``ConsolidationLoop`` + ``BackgroundTrainer`` for
       Qwen3 (seeded from disk: registries + ``graph.json`` present, no weights),
       arm the ``simulate→train`` migration state file, and submit to the new
       worker.  Runs the full ``migrate()`` call under GPU lock.  Uses the same
       ``migrate()`` entry point as Phase A so there is no hand-rolled per-tier
       loop here.  Marker → ``phaseB`` before the job is submitted; ``done``
       after success.
    5. **Recall gate** — Phase B's ``migrate()`` call uses the uncapped probe
       (``max_probe=len(entries)``).  If any tier fails the 1.0 gate, the
       ``MigrationState.all_tiers_done`` check inside ``migrate()`` catches it
       and the state file stays on disk.
    6. **Post-Phase-B in-process reload** — call
       ``_live_reload_base_model(refresh_config_from_disk=False)`` to
       align the in-RAM ``model.peft_config`` with disk.  Phase B's
       per-tier ``migrate()`` loop leaves the PeftModel mounted in the
       last tier's transient shape; without this reload, the published
       ``adapter_available`` topology (and recall behaviour) stays stale
       until the next systemctl restart.  Voice drain/restore is owned by
       the primitive: it drains STT/TTS to CPU before its VRAM gate
       (preventing the ~4.3 GiB voice footprint from blocking the gate on
       8 GiB hardware) and restores voice to GPU after a successful
       partial reload.  Best-effort: on internal reload failure the server
       lands in cloud-only with ``cloud_only_reason`` set, but the swap
       is complete on disk so step 7 still fires.
    7. **Success** — clear the active-store state file (already done by
       ``migrate()`` on all_tiers_done), clear the trial marker, clear the
       base-swap marker in the in-memory trial stash, set
       ``_state["mode"] = "local"`` (should already be set by the reload, but
       explicit for clarity after Phase B), set migration status to ``pass``.

    **Resume semantics** (controlled by ``resume_phase``)

    - ``""`` or ``"phaseA"`` (fresh start): run all steps 1–7.
    - ``"phaseA_done"``: Phase A and the bundle backup are already complete.
      Read ``bundle_slot`` from the on-disk marker; skip steps 1–2 and skip
      the ``_apply_config_live`` reload (boot already loaded the new config).
      Verify the new model is resident; if not, defer.  ``write_bundle`` is NOT
      called.
    - ``"phaseB"``: Phase A and reload already done.  Read ``bundle_slot`` from
      the on-disk marker; skip steps 1–3; resume at step 4 (Phase B setup).
      ``write_bundle`` is NOT called.

    **In-flight guard** (R2)

    ``_state["migration"]["base_swap_active"]`` is ``True`` while this
    coroutine is actively executing phases.  It is cleared in ``finally`` so it
    is ``False`` whenever the coroutine has exited — whether by success,
    failure, or deferred return.  ``POST /migration/confirm`` and
    ``POST /migration/rollback`` reject with 409 while the flag is ``True``.
    Rollback remains available when the flag is ``False`` and a ``phaseA_done``
    or ``phaseB`` marker exists (stranded deferred swap).

    **Failure semantics**

    - Phase A failure: gates → ``phase_a_failed``; bundle + marker preserved
      for ``POST /migration/rollback``.
    - Reload deferred: gates → ``reload_deferred``; marker stays at
      ``phaseA_done`` (resume entry point after VRAM frees).
    - Phase B failure (recall gate miss or tier exception): gates →
      ``phase_b_failed``; marker stays at ``phaseB`` for a subsequent retry.
      The operator can also call ``POST /migration/rollback`` to restore from
      the bundle.

    **No hand-rolled threading / GPU-lock acquisition here** — Phase A and
    Phase B both delegate to ``BackgroundTrainer.submit`` which holds the GPU
    lock for the duration of each job.  The reload between them runs via
    ``_apply_config_live`` (which acquires its own bounded lock) dispatched
    from the event-loop thread via ``run_in_executor`` — exactly as
    ``migration_accept`` does.

    Parameters
    ----------
    candidate_path_str:
        Absolute path to the candidate server.yaml (with ``model: <new>``).
        Unused when ``resume_phase`` is ``"phaseA_done"`` or ``"phaseB"``
        (config rename already happened in Phase A).
    live_config_path:
        Path to the live server.yaml to be overwritten by the atomic rename.
    state_dir:
        Directory for the trial marker (``<data>/state/``).
    backups_root:
        Root directory for backups (``<data>/backups/``).
    old_model:
        MODEL_REGISTRY alias of the live (Mistral) model.
    new_model:
        MODEL_REGISTRY alias of the replacement model (e.g. ``"qwen3-4b"``).
    started_at:
        ISO-8601 UTC timestamp when the confirm handler ran.
    candidate_hash:
        SHA-256 hex of the candidate config bytes.
    resume_phase:
        Phase to resume from.  ``""`` or ``"phaseA"`` for a fresh start (run
        all steps).  ``"phaseA_done"`` to skip bundle backup and Phase A and
        resume at the reload-check.  ``"phaseB"`` to skip straight to Phase B.
        Default ``""`` (fresh start).
    """
    from paramem.server.active_store_migration import MigrationState, save_state

    # ── In-flight guard: set base_swap_active so confirm/rollback can reject ──
    migration = _state.get("migration")
    if isinstance(migration, dict):
        migration["base_swap_active"] = True

    try:
        config = _state.get("config")
        if config is None:
            raise RuntimeError("base-swap orchestration: server config is None")

        # ── Resolve bundle_slot_str and pre_trial_hash ────────────────────────
        # On fresh start: derive from the live config.
        # On resume: read from the on-disk marker (bundle_slot was written
        # exactly once at fresh start and must not be re-derived).
        if resume_phase in ("phaseA_done", "phaseB"):
            # Resume path — read existing marker for the bundle slot and hashes.
            # write_bundle MUST NOT be called; the bundle already exists.
            _resume_marker = read_trial_marker(state_dir)
            if _resume_marker is None:
                raise RuntimeError(
                    f"base-swap resume (phase={resume_phase!r}): "
                    "trial marker not found — cannot determine bundle_slot"
                )
            bundle_slot_str = _resume_marker.bundle_slot
            if not bundle_slot_str:
                raise RuntimeError(
                    f"base-swap resume (phase={resume_phase!r}): "
                    "marker.bundle_slot is empty — rollback anchor lost"
                )
            pre_trial_hash = _resume_marker.pre_trial_config_sha256
        else:
            # Fresh start — compute from disk.
            live_registry_sha256 = ""
            try:
                if hasattr(config, "paths") and config.paths.data is not None:
                    reg_path = config.paths.key_metadata
                    if reg_path.exists():
                        import hashlib as _hlib

                        from paramem.backup.encryption import read_maybe_encrypted as _rme

                        live_registry_sha256 = _hlib.sha256(_rme(reg_path)).hexdigest()
            except Exception:  # noqa: BLE001
                live_registry_sha256 = ""

            # Build per-tier adapter_dirs dict from the live config.
            adapter_dirs: dict[str, Path] = {}
            adapters_cfg = getattr(config, "adapters", None)
            for _tier_name in ("episodic", "semantic", "procedural"):
                _tier_cfg = getattr(adapters_cfg, _tier_name, None) if adapters_cfg else None
                if _tier_cfg is not None and getattr(_tier_cfg, "enabled", False):
                    adapter_dirs[_tier_name] = Path(config.adapter_dir) / _tier_name

            registry_path = Path(config.paths.key_metadata)
            data_dir = Path(config.paths.data)
            speaker_profiles_path = data_dir / "speaker_profiles.json"

            # ── Step 1: Bundle backup (rollback anchor) ───────────────────────
            # Written exactly once — before any mutation.  Resume paths never
            # reach this block (guarded by the resume_phase check above).
            bundle_slot = write_bundle(
                config_path=live_config_path,
                registry_path=registry_path,
                adapter_dirs=adapter_dirs,
                base_dir=backups_root / "snapshot",
                meta_fields={"tier": "pre_base_swap", "label": f"pre_base_swap_{new_model}"},
                adapter_scope="live",
                live_registry_sha256=live_registry_sha256,
                speaker_profiles_path=(
                    speaker_profiles_path if speaker_profiles_path.exists() else None
                ),
                candidate_config_path=Path(candidate_path_str),
            )
            bundle_slot_str = str(bundle_slot.resolve())

            # Update the in-memory trial stash with the bundle slot.
            migration_stash = _state.get("migration", {})
            trial_data = migration_stash.get("trial") or {}
            trial_data["backup_paths"] = {"bundle": bundle_slot_str}
            migration_stash["trial"] = trial_data

            pre_trial_hash = ""
            if live_config_path.exists():
                import hashlib as _hlib2

                pre_trial_hash = _hlib2.sha256(live_config_path.read_bytes()).hexdigest()

        if resume_phase not in ("phaseA_done", "phaseB"):
            # ── Step 2: Phase A — train→simulate on Mistral ──────────────────
            # Skipped on resume at phaseA_done or phaseB.

            # Write marker at phaseA before any mutations.
            marker = TrialMarker(
                schema_version=1,
                started_at=started_at,
                pre_trial_config_sha256=pre_trial_hash,
                candidate_config_sha256=candidate_hash,
                backup_paths={"bundle": bundle_slot_str},
                trial_adapter_dir="",
                trial_graph_dir="",
                config_artifact_filename="",
                migration_kind="base_swap",
                base_swap_phase="phaseA",
                old_model=old_model,
                new_model=new_model,
                bundle_slot=bundle_slot_str,
            )
            write_trial_marker(state_dir, marker)

            # Arm the train→simulate active-store migration state file.
            migration_state = MigrationState.for_mode_switch(
                source_mode="train", target_mode="simulate"
            )
            save_state(Path(config.adapter_dir), migration_state)

            loop = _state.get("consolidation_loop")
            if loop is None:
                loop = create_consolidation_loop(
                    _state["model"],
                    _state["tokenizer"],
                    config,
                    _state["memory_store"],
                    state_provider=lambda: _state,
                )
                _state["consolidation_loop"] = loop
                _state["model"] = loop.model

            bt = BackgroundTrainer(
                model=_state["model"],
                tokenizer=_state["tokenizer"],
                training_config=config.training_config,
                output_dir=config.adapter_dir,
                thermal_policy=ThermalPolicy.from_consolidation_config(config.consolidation),
                preload_cache=config.inference.preload_cache,
            )
            _state["background_trainer"] = bt
            _state["consolidation_loop"]._bg_trainer = bt

            # asyncio.Event lets us await the BG-worker result from this coroutine.
            # Capture the running event loop at creation time so the worker thread
            # can signal completion via call_soon_threadsafe even when
            # _state["event_loop"] is not yet populated (e.g. during unit tests).
            _phase_a_aio_loop = asyncio.get_event_loop()
            done_event = asyncio.Event()
            phase_a_error: list[Exception] = []

            def _run_phase_a_on_worker() -> None:
                """Run on the BG-trainer worker thread under the GPU lock."""
                from paramem.server.active_store_migration import load_state as _phase_a_load_state

                _fresh_state = _phase_a_load_state(Path(config.adapter_dir))
                if _fresh_state is None:
                    phase_a_error.append(RuntimeError("Phase A: migration state file vanished"))
                    _phase_a_aio_loop.call_soon_threadsafe(done_event.set)
                    return
                updated = migrate(loop, config, _fresh_state)
                _state["model"] = loop.model
                if not updated.all_tiers_done(loop.store.tiers_with_registry()):
                    first_fail = next(iter(updated.failed_tiers.values()), "unknown")
                    phase_a_error.append(RuntimeError(f"Phase A incomplete: {first_fail}"))
                _phase_a_aio_loop.call_soon_threadsafe(done_event.set)

            bt.submit(_run_phase_a_on_worker, inference_fallback_adapter="episodic")
            await done_event.wait()

            if phase_a_error:
                raise phase_a_error[0]

            # Phase A succeeded: atomic config rename, then advance marker to
            # phaseA_done.  The worker job for Phase A has completed; the worker
            # is now idle.  The reload (Step 3 below) runs from THIS coroutine,
            # not from a worker job — so _release_base_model_in_process →
            # bt._stop_callable_worker() stops only an idle worker.
            # No worker-kill hazard.
            _rename_config(Path(candidate_path_str), live_config_path)
            dir_fd = os.open(str(live_config_path.parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            except OSError:
                pass
            finally:
                os.close(dir_fd)

            phase_a_done_marker = TrialMarker(
                schema_version=1,
                started_at=started_at,
                pre_trial_config_sha256=pre_trial_hash,
                candidate_config_sha256=candidate_hash,
                backup_paths={"bundle": bundle_slot_str},
                trial_adapter_dir="",
                trial_graph_dir="",
                config_artifact_filename="",
                migration_kind="base_swap",
                base_swap_phase="phaseA_done",
                old_model=old_model,
                new_model=new_model,
                bundle_slot=bundle_slot_str,
            )
            write_trial_marker(state_dir, phase_a_done_marker)
            logger.info(
                "base-swap Phase A complete: old=%s new=%s bundle=%s",
                old_model,
                new_model,
                bundle_slot_str,
            )

        if resume_phase != "phaseB":
            # ── Step 3: In-process reload — release Mistral, load Qwen3 ─────
            # Skipped on resume at phaseB (reload already succeeded).
            #
            # Two sub-cases:
            #
            # a) Fresh start (resume_phase == ""): the process is still running
            #    with the old model.  gpu_release drains voice + releases base to
            #    cloud-only; gpu_acquire then loads the new base on the clean
            #    GPU.  The VRAM gate inside _live_reload_base_model (called by
            #    gpu_acquire) sees accurate free bytes and the reload completes
            #    in-process without requiring a restart/resume detour.
            #
            # b) Resume (resume_phase == "phaseA_done"): the old process crashed
            #    after writing the phaseA_done marker and the server was
            #    restarted.  Boot already loaded the new config (Qwen3) from the
            #    renamed server.yaml, so the new model is already resident.
            #    There is nothing to reload.  Go straight to the Phase-B
            #    identity guard below.  If the model turns out NOT to be loaded
            #    (e.g. boot came up cloud-only due to VRAM pressure), set
            #    reload_deferred so /gpu/acquire re-triggers Phase B.
            #
            if resume_phase == "":
                # Fresh start: reload the NEW base in-process.  Three pieces, each
                # load-bearing (proven empirically):
                #   1. Drop THIS coroutine's references to Phase A's
                #      ConsolidationLoop / BackgroundTrainer — they pin the OLD
                #      base model.  Phase B re-creates its own loop_b / bt_b.
                #   2. gpu_release: the only path that actually reclaims the old
                #      base + voice here (the bare reload-release left it resident
                #      at 1.34 GiB free; gpu_release drains to ~6.5 GiB free).
                #   3. _apply_config_live: re-reads the renamed-on-disk config and
                #      reloads the NEW base with a full rebuild (gpu_acquire would
                #      keep the stale in-memory config).  _live_reload_base_model
                #      recomputes the VRAM topology for the new model so the gate
                #      uses its footprint, not the old base's.
                # base_swap_active is True, so neither re-enters this orchestration.
                loop = None
                bt = None
                await gpu_release()
                await asyncio.get_running_loop().run_in_executor(None, _apply_config_live)

                # After the executor returns, check whether the reload succeeded
                # or was deferred due to insufficient VRAM.
                if _state.get("mode") != "local":
                    deferred_reason = _state.get("cloud_only_reason", "reload_deferred")
                    completed_at = datetime.now(timezone.utc).isoformat()
                    await _update_trial_gates(
                        {
                            "status": "reload_deferred",
                            "completed_at": completed_at,
                            "cloud_only_reason": deferred_reason,
                            "message": (
                                f"Phase A complete but base-model reload deferred "
                                f"(cloud_only_reason={deferred_reason!r}). "
                                "Phase B will run automatically once the new model "
                                "is loaded (POST /gpu/acquire triggers this)."
                            ),
                        }
                    )
                    logger.warning(
                        "base-swap reload deferred: cloud_only_reason=%s; Phase B not started",
                        deferred_reason,
                    )
                    return

            else:
                # resume_phase == "phaseA_done": boot already loaded the new
                # config; confirm the new model is actually resident before
                # proceeding to Phase B.
                _resume_mode = _state.get("mode")
                _resume_model_name = getattr(_state.get("config"), "model_name", None)
                _new_model_loaded = _resume_mode == "local" and _resume_model_name == new_model
                if not _new_model_loaded:
                    # Boot came up cloud-only (e.g. VRAM pressure) or with the
                    # wrong model.  Phase B cannot run; defer until /gpu/acquire
                    # re-launches with the new model loaded.
                    deferred_reason = _state.get("cloud_only_reason") or "reload_deferred"
                    completed_at = datetime.now(timezone.utc).isoformat()
                    await _update_trial_gates(
                        {
                            "status": "reload_deferred",
                            "completed_at": completed_at,
                            "cloud_only_reason": deferred_reason,
                            "message": (
                                f"Phase A complete (resume) but new model not loaded "
                                f"(mode={_resume_mode!r}, "
                                f"config.model_name={_resume_model_name!r}, "
                                f"expected {new_model!r}). "
                                "Phase B will run automatically once the new model "
                                "is loaded (POST /gpu/acquire triggers this)."
                            ),
                        }
                    )
                    logger.warning(
                        "base-swap resume: new model not loaded "
                        "(mode=%s, model=%s, expected=%s); Phase B deferred",
                        _resume_mode,
                        _resume_model_name,
                        new_model,
                    )
                    return

        # ── Step 4: Phase B — simulate→train on Qwen3 ────────────────────────
        # The reload dropped the old ConsolidationLoop and BackgroundTrainer
        # (_live_reload_base_model → _release_base_model_in_process →
        # bt._stop_callable_worker).  Re-create them from disk (registries +
        # graph.json present, no weights) using the new config that was loaded
        # into _state by _apply_config_live → _refresh_config_from_disk_into_state.
        # This mirrors the pattern in _run_active_store_migration_sync.
        marker_phase_b = TrialMarker(
            schema_version=1,
            started_at=started_at,
            pre_trial_config_sha256=pre_trial_hash,
            candidate_config_sha256=candidate_hash,
            backup_paths={"bundle": bundle_slot_str},
            trial_adapter_dir="",
            trial_graph_dir="",
            config_artifact_filename="",
            migration_kind="base_swap",
            base_swap_phase="phaseB",
            old_model=old_model,
            new_model=new_model,
            bundle_slot=bundle_slot_str,
        )
        write_trial_marker(state_dir, marker_phase_b)

        # Re-read config from _state (may have changed during reload).
        config_b = _state.get("config")
        if config_b is None:
            raise RuntimeError("base-swap Phase B: server config is None after reload")

        # ── Phase B model-identity guard ─────────────────────────────────────
        # Fail loud if the loaded base model is not the expected new model.
        # This prevents Phase B from retraining adapters on the wrong base —
        # a silent wrong outcome where the recall gate would pass on the old
        # model.  Two conditions must hold:
        #   1. mode must be "local" (model is loaded and serving).
        #   2. The live config's model_name must match new_model (set by
        #      _apply_config_live → _refresh_config_from_disk_into_state).
        # On mismatch: record phase_b_model_mismatch, leave marker+bundle
        # intact for rollback, and return without calling migrate().
        _mode_after_reload = _state.get("mode")
        _config_model_name = getattr(config_b, "model_name", None)
        _model_identity_ok = _mode_after_reload == "local" and _config_model_name == new_model
        if not _model_identity_ok:
            _mismatch_reason = (
                f"mode={_mode_after_reload!r} (expected 'local'), "
                f"config.model_name={_config_model_name!r} (expected {new_model!r})"
            )
            _mismatch_at = datetime.now(timezone.utc).isoformat()
            await _update_trial_gates(
                {
                    "status": "phase_b_model_mismatch",
                    "completed_at": _mismatch_at,
                    "mismatch_reason": _mismatch_reason,
                    "message": (
                        f"Phase B aborted: loaded model does not match new_model={new_model!r}. "
                        f"Detail: {_mismatch_reason}. "
                        "Bundle and marker preserved — run `paramem migrate --rollback` to restore."
                    ),
                }
            )
            logger.error("base-swap Phase B aborted (model mismatch): %s", _mismatch_reason)
            return
        # ── end guard ────────────────────────────────────────────────────────

        # Arm the simulate→train migration state file for Phase B.
        migration_state_b = MigrationState.for_mode_switch(
            source_mode="simulate", target_mode="train"
        )
        save_state(Path(config_b.adapter_dir), migration_state_b)

        loop_b = _state.get("consolidation_loop")
        if loop_b is None:
            loop_b = create_consolidation_loop(
                _state["model"],
                _state["tokenizer"],
                config_b,
                _state["memory_store"],
                state_provider=lambda: _state,
            )
            _state["consolidation_loop"] = loop_b
            _state["model"] = loop_b.model

        bt_b = BackgroundTrainer(
            model=_state["model"],
            tokenizer=_state["tokenizer"],
            training_config=config_b.training_config,
            output_dir=config_b.adapter_dir,
            thermal_policy=ThermalPolicy.from_consolidation_config(config_b.consolidation),
            preload_cache=config_b.inference.preload_cache,
        )
        _state["background_trainer"] = bt_b
        _state["consolidation_loop"]._bg_trainer = bt_b

        _phase_b_aio_loop = asyncio.get_event_loop()
        done_event_b = asyncio.Event()
        phase_b_error: list[Exception] = []

        def _run_phase_b_on_worker() -> None:
            """Run Phase B on the BG-trainer worker thread under the GPU lock."""
            from paramem.server.active_store_migration import load_state as _phase_b_load_state

            _fresh_state_b = _phase_b_load_state(Path(config_b.adapter_dir))
            if _fresh_state_b is None:
                phase_b_error.append(
                    RuntimeError("Phase B: migration state file vanished before Phase B ran")
                )
                _phase_b_aio_loop.call_soon_threadsafe(done_event_b.set)
                return
            # The base-swap preload gate left the live store empty — the on-disk
            # registries belong to the OLD (Mistral) model and are NOT model B's
            # inference state.  They ARE, however, Phase B's training INPUT:
            # migrate() iterates loop.store.tiers_with_registry() to know which
            # tiers to retrain.  Load them into loop_b's store now (worker thread,
            # GPU lock held → inference is cloud-routed) so migrate has the tier
            # list; it rebuilds each tier from graph.json into model B's fresh
            # registry.  Without this the store is empty → migrate refuses with
            # "0 tiers but on-disk content exists".
            loop_b.store.load_registries_from_disk(config_b.adapter_dir)
            updated_b = migrate(loop_b, config_b, _fresh_state_b)
            _state["model"] = loop_b.model
            if not updated_b.all_tiers_done(loop_b.store.tiers_with_registry()):
                first_fail = next(iter(updated_b.failed_tiers.values()), "unknown")
                phase_b_error.append(RuntimeError(f"Phase B incomplete: {first_fail}"))
            _phase_b_aio_loop.call_soon_threadsafe(done_event_b.set)

        bt_b.submit(_run_phase_b_on_worker, inference_fallback_adapter="episodic")
        await done_event_b.wait()

        if phase_b_error:
            raise phase_b_error[0]

        # ── Promotion carry-over ─────────────────────────────────────────────
        # The migration never writes key_metadata.json, so it still holds the
        # PREVIOUS model's promotion state (per-key sessions_seen + promoted_keys).
        # loop_b was created against the empty live store (the base-swap preload
        # gate skips loading the old registry), so its construction-time seed
        # orphan-dropped every key.  Now that Phase B has retrained the SAME keys
        # (stable via graph.json ``ik_key``) and repopulated the store, re-seed
        # from the preserved key_metadata.json so promotion momentum carries
        # across the swap — a key at sessions_seen=N does not reset to 0, and the
        # already-promoted set is restored.  Without this, the next consolidation's
        # _save_key_metadata would overwrite the on-disk counts with loop_b's empty
        # in-memory state.  seed_key_metadata SETs (not increments) so it is
        # idempotent; the keys match by construction so there is no orphan-drop now.
        from paramem.server.consolidation import _load_key_metadata as _carry_load_meta

        _carry_meta = _carry_load_meta(config_b.key_metadata_path)
        if _carry_meta is not None:
            loop_b.seed_key_metadata(_carry_meta)
            logger.info(
                "base-swap: carried over promotion state — %d key(s) tracked, %d promoted",
                len(loop_b.key_sessions),
                len(loop_b.promoted_keys),
            )

        # ── Step 6: Post-Phase-B in-process reload — align in-RAM peft_config
        # with disk.  Phase B's migrate() promoted weights for every tier and
        # called wrap_lora()/create_adapter() per tier as it iterated; the last
        # tier through migrate leaves the in-RAM PeftModel mounted in its
        # transient mid-iteration shape (the symptom traced 2026-05-28:
        # semantic mounted as a Qwen-shape LoRA-zero, hiding the just-promoted
        # weights from /debug/recall until a manual systemctl restart).  A
        # plain reclaim-style reload (refresh_config_from_disk=False) tears
        # down the PeftModel and rebuilds from disk, picking up each tier's
        # promoted adapter cleanly.
        #
        # Same-config reload — no config delta to apply — so it routes through
        # _live_reload_base_model directly rather than _apply_config_live.
        # Drop our locals so they do not pin the old base graph (same pattern
        # as the Phase A → Phase B reload at Step 3).  base_swap_active is
        # still True throughout, so /gpu/release and /gpu/acquire cannot race
        # us.
        #
        # Voice drain/restore is owned by _live_reload_base_model: the
        # primitive drains STT/TTS to CPU before its VRAM gate (preventing
        # the ~4.3 GiB voice footprint from blocking the gate) and restores
        # to GPU after a successful partial reload.  This caller does not hold
        # the GPU lock, so lock_held=False (default) is correct.
        #
        # If the reload fails internally it leaves the server cloud-only with
        # cloud_only_reason set; the swap is already complete on disk, so we
        # still mark status=pass and let the next /gpu/acquire recover.
        loop_b = None
        bt_b = None
        try:
            _loop = asyncio.get_running_loop()
            await _loop.run_in_executor(None, _live_reload_base_model)
        except Exception:  # noqa: BLE001
            logger.exception(
                "base-swap: post-Phase-B live reload raised; weights are on disk "
                "but in-RAM peft_config may be stale until /gpu/acquire or restart"
            )

        # ── Step 7: Success — clear state, clear marker, status=pass ─────────
        # active_store_migration.migrate() already cleared the state file on
        # all_tiers_done.  Clear the trial marker and reset migration state.
        from paramem.server.active_store_migration import clear_state as _clear_migrate_state

        _clear_migrate_state(Path(config_b.adapter_dir))
        clear_trial_marker(state_dir)

        prior_recovery = list((_state.get("migration") or {}).get("recovery_required") or [])
        from paramem.server.migration import initial_migration_state

        _state["migration"] = initial_migration_state()
        _state["migration"]["recovery_required"] = prior_recovery

        completed_at = datetime.now(timezone.utc).isoformat()
        await _update_trial_gates(
            {
                "status": "pass",
                "completed_at": completed_at,
                "message": (
                    f"Base-swap migration complete. "
                    f"Model: {old_model} → {new_model}. "
                    "All tiers trained on new base model."
                ),
            }
        )
        logger.info(
            "base-swap orchestration complete: old=%s new=%s",
            old_model,
            new_model,
        )

    except Exception as exc:  # noqa: BLE001
        completed_at = datetime.now(timezone.utc).isoformat()
        # Determine which phase failed from the current marker on disk.
        _failed_marker = None
        try:
            _failed_marker = read_trial_marker(state_dir)
        except Exception:  # noqa: BLE001
            pass
        _phase = (
            getattr(_failed_marker, "base_swap_phase", "unknown")
            if _failed_marker is not None
            else "unknown"
        )
        if _phase in ("phaseA", ""):
            _status = "phase_a_failed"
        else:
            _status = "phase_b_failed"

        await _update_trial_gates(
            {
                "status": _status,
                "exception": str(exc),
                "completed_at": completed_at,
            }
        )
        logger.exception("base-swap orchestration failed (phase=%s): %s", _phase, exc)
        # Bundle and marker are preserved for rollback.

    finally:
        # ── Clear the in-flight guard ────────────────────────────────────────
        # base_swap_active is True while this coroutine actively executes phases.
        # Clearing it here (in finally) ensures it is False whether the
        # coroutine exits by success, failure, or deferred return — so
        # rollback (the escape hatch from a stranded deferred swap) is
        # unblocked as soon as the coroutine is no longer running.
        _mig = _state.get("migration")
        if isinstance(_mig, dict):
            _mig["base_swap_active"] = False


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
    from paramem.memory.store import MemoryStore as _MemoryStore
    from paramem.server.consolidation import create_consolidation_loop

    # Trial path: construct a fresh, isolated store that mirrors the trial
    # adapter dir's registries.  Do NOT reuse the live ``_state["memory_store"]``
    # — the trial must not pollute the production store.
    trial_store = _MemoryStore(
        replay_enabled=trial_config.consolidation.indexed_key_replay,
    )
    if trial_adapter_dir is not None:
        try:
            trial_store.load_registries_from_disk(trial_adapter_dir)
        except Exception:
            logger.exception("Trial memory_store registry load failed; starting empty")
    loop = create_consolidation_loop(model, tokenizer, trial_config, trial_store)

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


@app.get(
    "/migration/status",
    response_model=MigrationStatusResponse,
    dependencies=[Depends(require_admin)],
)
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


@app.get(
    "/migration/diff",
    response_model=MigrationDiffResponse,
    dependencies=[Depends(require_admin)],
)
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


@app.post("/migration/accept", response_model=AcceptResponse, dependencies=[Depends(require_admin)])
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

        # --- Step 4: Move trial adapter into the rotation slot; delete trial graph ---
        # Non-fatal: config + marker are already coherent. Rotation is cosmetic.
        # The trial graph is transient by design (graph is RAM-only in production;
        # persist_graph=True is only used during the trial window to render the
        # before/after comparison report).  Once the operator accepts, the trial
        # graph is dead weight and contradicts the "graph is transient" invariant.
        # Delete it unconditionally; production re-builds the graph in RAM on the
        # next live cycle.
        rotation_incomplete = False
        for src_str, dest_name in [
            (trial_adapter_dir_str, "adapter"),
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

        # Delete the trial graph (transient by design — no value in archiving).
        if trial_graph_dir_str:
            _tg = Path(trial_graph_dir_str)
            if _tg.exists():
                shutil.rmtree(_tg, ignore_errors=True)
                logger.debug("accept: trial graph deleted post-accept (transient by design)")

        # --- Step 5: Apply config live + refresh drift state + set banner ---
        # WP2 / B-1 precondition: _apply_config_live compares disk_hash against
        # config_drift["loaded_hash"] (hash of config A, captured at boot).
        # The drift refresh below must happen AFTER _apply_config_live so the
        # no-op skip (disk_hash == loaded_hash) does not fire on the accept path.
        #
        # Apply ordering:
        #  a) Set synchronous maintenance guard BEFORE dispatching executor (S-4):
        #     mode="cloud-only" so the scheduler's mode != "local" defer fires
        #     during the brief window between migration-state reset and apply.
        #  b) Dispatch _apply_config_live via run_in_executor (runs synchronously
        #     from the accept handler's perspective — we await the result).
        #  c) Refresh config_drift AFTER a successful apply (loaded_hash now B).
        #  d) Build the banner and response.  The server does NOT fire _restart_service
        #     for R-PORT; the CLI prompts the operator and runs the restart_hint command.

        # Synchronous maintenance guard + dispatch + restore-if-untouched
        # (correction S-4) — see _apply_config_live_guarded for the full rationale.
        apply_result = await _apply_config_live_guarded()

        applied_live: bool = apply_result.get("applied_live", False)
        apply_reason: str | None = apply_result.get("restart_required_reason")
        auto_restart_scheduled: bool = False  # server never self-fires restart
        restart_eligible: bool = apply_result.get("restart_eligible", False)

        # c) Refresh config_drift AFTER the apply.
        # When applied_live=True the loaded_hash is now B (the apply updated it
        # via _live_reload_base_model → load_server_config).  When applied_live=False
        # the config is on disk (B) but still needs a restart — refresh drift so the
        # drift detector does not re-alarm (drift detected=False; loaded_hash still A
        # is the honest state, but the file IS on disk so disk_hash=B is correct).
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

        # d) Build banner — replace "RESTART REQUIRED" with "applied live" on success.
        if applied_live and apply_reason is None:
            # Full live apply succeeded; no restart needed.
            if rotation_incomplete:
                banner = (
                    "Migration: applied live (no restart required); "
                    "ARCHIVE INCOMPLETE — trial adapter not fully rotated, archive manually"
                )
            else:
                banner = "Migration: applied live — new configuration is active"
        elif applied_live and apply_reason == "paths_change":
            # Mixed delta: non-path fields applied live; paths carve needs manual restart.
            banner = (
                "Migration: partial live apply — non-path fields active; "
                "path change on disk, effective on NEXT restart; "
                "DATA IS NOT MIGRATED — move adapters/registry/sessions "
                "to the new path before restarting"
            )
        elif apply_reason in ("stt_port_change", "tts_port_change"):
            if restart_eligible:
                # Port pre-flighted successfully; CLI will prompt operator for consent.
                banner = (
                    f"Migration: RESTART REQUIRED — port change ({apply_reason}); "
                    "restart via the CLI or run the restart_hint command manually"
                )
            else:
                # Port was in use — pre-flight declined.
                port_in_use = apply_result.get("port_in_use_reason", "port in use")
                banner = (
                    f"Migration: {apply_reason} — port not bindable ({port_in_use}); "
                    "free the port and restart manually"
                )
        elif apply_reason == "paths_change":
            # Pure paths carve (short-circuited before live reload).
            banner = (
                "Migration: RESTART REQUIRED — path change on disk, effective on NEXT restart; "
                "DATA IS NOT MIGRATED — move adapters/registry/sessions "
                "to the new path before restarting"
            )
            if rotation_incomplete:
                banner += "; ARCHIVE INCOMPLETE — trial adapter not fully rotated, archive manually"
        else:
            # Apply failed or other reason — keep RESTART REQUIRED banner.
            banner = (
                "Migration: RESTART REQUIRED — new configuration takes effect on server restart"
            )
            if rotation_incomplete:
                banner += "; ARCHIVE INCOMPLETE — trial adapter not fully rotated, archive manually"

        # Reset migration state to LIVE, preserving recovery_required.
        prior_recovery = list(migration.get("recovery_required") or [])
        _state["migration"] = initial_migration_state()
        _state["migration"]["recovery_required"] = prior_recovery + [banner]

        restart_required = not applied_live or (apply_reason is not None)

        return AcceptResponse(
            state="LIVE",
            trial_adapter_archive_path=archive_path,
            restart_required=restart_required,
            restart_hint=_RESTART_HINT,
            pre_migration_backup_retained=True,
            applied_live=applied_live,
            restart_required_reason=apply_reason,
            auto_restart_scheduled=auto_restart_scheduled,
            restart_eligible=restart_eligible,
        )


@app.post("/migration/rollback", dependencies=[Depends(require_admin)])
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

        # --- R2: In-flight guard — reject rollback while orchestration runs ---
        # base_swap_active is True while _run_base_swap_orchestration is
        # actively executing.  It is False when the coroutine is not running
        # (success, failure, or deferred/stranded state).  Rollback while
        # actively running risks concurrent writes to the same adapter dirs.
        # When the coroutine is NOT running (deferred/stranded), rollback is
        # the escape hatch and must remain available.
        if migration.get("base_swap_active", False):
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "base_swap_active",
                    "message": (
                        "A base-swap migration is actively running. "
                        "Wait for Phase A or Phase B to complete (or fail) "
                        "before rolling back."
                    ),
                },
            )

        trial = migration.get("trial") or {}
        trial_adapter_dir_str = trial.get("trial_adapter_dir", "")
        trial_graph_dir_str = trial.get("trial_graph_dir", "")
        pre_trial_config_sha256 = trial.get("pre_trial_config_sha256", "")
        trial_started_at = trial.get("started_at", "")

        # --- Base-swap rollback branch ---
        # When migration_kind == "base_swap" (written into the marker by
        # _run_base_swap_orchestration), the A-config is inside a full bundle
        # slot, not a standalone config backup.  Restore via restore_bundle
        # (config + registry + per-tier adapters + speaker_profiles) so
        # Mistral weights come back alongside the Mistral config.
        #
        # Split-brain fix: after restore_bundle writes the Mistral config to
        # disk, the in-memory model may be Qwen3 (if the reload in Step 3 of
        # the orchestration succeeded before the rollback was triggered).  The
        # no-op skip inside _apply_config_live compares disk_hash against
        # _state["config_drift"]["loaded_hash"]. After a config-swap the loaded
        # hash equals the Qwen3 config, NOT the Mistral config now on disk, so
        # the skip fires on the WRONG branch and leaves Qwen3 resident.
        # Fix: invalidate loaded_hash BEFORE dispatching _apply_config_live so
        # the skip cannot fire and the full reload runs.
        _bs_marker = read_trial_marker(state_dir)
        if _bs_marker is not None and _bs_marker.migration_kind == "base_swap":
            bundle_slot_path_str = _bs_marker.bundle_slot
            if not bundle_slot_path_str:
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": "rollback_precondition_failed",
                        "message": (
                            "Base-swap rollback: bundle_slot is empty in trial marker. "
                            "Cannot restore without the bundle backup."
                        ),
                    },
                )

            bundle_slot_path = Path(bundle_slot_path_str)
            if not bundle_slot_path.exists():
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": "rollback_precondition_failed",
                        "message": (
                            f"Base-swap rollback: bundle slot not found at "
                            f"{bundle_slot_path}. "
                            "Backup may have been manually deleted."
                        ),
                    },
                )

            if config is None:
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": "rollback_precondition_failed",
                        "message": (
                            "Base-swap rollback: server config is None; cannot derive data_dir."
                        ),
                    },
                )

            from paramem.backup.backup import restore_bundle as _restore_bundle_fn
            from paramem.backup.types import BundleManifestError, FingerprintMismatchError

            data_dir_rb = Path(config.paths.data).resolve()

            try:
                _restore_bundle_fn(
                    bundle_slot_path,
                    data_dir=data_dir_rb,
                    config_path=live_config_path,
                    restore_config=True,
                )
            except (BundleManifestError, FingerprintMismatchError) as exc:
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": "bundle_corrupt",
                        "message": f"Base-swap rollback bundle is corrupt: {exc}",
                    },
                ) from exc
            except RuntimeError as exc:
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": "decrypt_no_key",
                        "message": (
                            "Base-swap rollback bundle is age-encrypted but the daily "
                            f"identity is not loaded: {exc}"
                        ),
                    },
                ) from exc

            # Mistral config is now on disk.  Invalidate config_drift.loaded_hash
            # so _apply_config_live's no-op skip (disk_hash == loaded_hash) cannot
            # fire — we need a forced reload regardless of what was previously in
            # memory.  Using a sentinel string that can never equal a real SHA-256
            # (which is 64 lower-hex characters) is safe.
            _cd = _state.get("config_drift")
            if isinstance(_cd, dict):
                _cd["loaded_hash"] = "__rollback_invalidated__"
            else:
                _state["config_drift"] = {"loaded_hash": "__rollback_invalidated__"}

            # Clear the active-store migration state file and the base-swap marker
            # BEFORE the reload.  The swap is being abandoned, so the preload
            # base-swap gate must NOT fire on this reload — otherwise it would
            # leave the live store empty even though the bundle restore just put
            # Mistral's registries back on disk.  With the marker cleared, the
            # reload loads the restored Mistral registries normally.
            from paramem.server.active_store_migration import clear_state as _clear_as_state

            _clear_as_state(Path(config.adapter_dir))
            clear_trial_marker(state_dir)

            # Dispatch the in-process reload to bring Mistral back, under the
            # shared synchronous maintenance guard (restore-on-no-op handled
            # inside).  This path does not consume the apply result.
            await _apply_config_live_guarded()

            # Reset migration state to LIVE.
            prior_recovery_rb = list(migration.get("recovery_required") or [])
            from paramem.server.migration import initial_migration_state as _init_ms

            _state["migration"] = _init_ms()
            restart_banner_rb = (
                "Migration: base-swap rolled back — Mistral weights and config restored from bundle"
            )
            _state["migration"]["recovery_required"] = prior_recovery_rb + [restart_banner_rb]

            return RollbackResponse(
                state="LIVE",
                trial_adapter_archive_path="",
                rollback_pre_mortem_backup_path="",
                restart_required=False,
                restart_hint="",
                applied_live=True,
                restart_required_reason=None,
                auto_restart_scheduled=False,
                restart_eligible=False,
            )

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

            # Move only the trial adapter into the slot (graph excluded — transient
            # by design; deleted below, not archived).
            for src_str, dest_name in [
                (trial_adapter_dir_str, "adapter"),
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

        # Delete the trial graph (transient by design — no value in archiving after rollback).
        if trial_graph_dir_str:
            _tg = Path(trial_graph_dir_str)
            if _tg.exists():
                shutil.rmtree(_tg, ignore_errors=True)
                logger.debug("rollback: trial graph deleted post-rollback (transient by design)")

        # --- Step 7: Apply config live (no-op skip for rollback) ---
        # WP2 / S-6: rollback restored disk to A and config_drift["loaded_hash"]
        # is still A (rollback does NOT refresh drift — see comment at step 6240).
        # So _apply_config_live will take the no-op skip (disk_hash == loaded_hash)
        # and return applied_live=True, skipped="no_change" without GPU churn.
        #
        # Synchronous maintenance guard + dispatch + restore-if-untouched
        # (correction S-4) — rollback's apply normally takes the no-op skip
        # (disk==memory==A), returning with the guard untouched, so the restore
        # inside _apply_config_live_guarded keeps the server from sticking
        # cloud-only.
        apply_result = await _apply_config_live_guarded()

        applied_live: bool = apply_result.get("applied_live", False)
        apply_reason: str | None = apply_result.get("restart_required_reason")
        restart_eligible: bool = apply_result.get("restart_eligible", False)

        # NOTE: do NOT refresh config_drift for rollback — in-memory config already
        # matches A; the drift loop stays coherent (config_drift.loaded_hash is A).
        # The no-op skip confirmed the apply; no hash update needed.

        # --- Step 8: Build banner + reset migration state to LIVE ---
        if applied_live and apply_reason is None:
            # No-op skip: config A already applied, no restart needed.
            restart_banner = (
                "Migration: rolled back — config A is already active; no restart required"
            )
        elif applied_live and apply_reason == "paths_change":
            restart_banner = (
                "Migration: partial rollback live apply — non-path fields active; "
                "path change on disk, effective on NEXT restart; "
                "DATA IS NOT MIGRATED — move adapters/registry/sessions before restarting"
            )
        elif apply_reason in ("stt_port_change", "tts_port_change"):
            if restart_eligible:
                # Port pre-flighted; CLI will prompt operator.
                restart_banner = (
                    f"Migration: RESTART REQUIRED — port change ({apply_reason}); "
                    "restart via the CLI or run the restart_hint command manually"
                )
            else:
                port_in_use = apply_result.get("port_in_use_reason", "port in use")
                restart_banner = (
                    f"Migration: RESTART REQUIRED — {apply_reason}: {port_in_use}; "
                    "free the port and restart manually"
                )
        elif apply_reason == "paths_change":
            restart_banner = (
                "Migration: RESTART REQUIRED — rollback renamed configs/server.yaml; "
                "path change on disk, effective on NEXT restart; "
                "DATA IS NOT MIGRATED — move adapters/registry/sessions before restarting"
            )
        else:
            restart_banner = (
                "Migration: RESTART REQUIRED — rollback renamed configs/server.yaml; "
                "restart to clear recovery banner"
            )

        prior_recovery = list(migration.get("recovery_required") or [])
        _state["migration"] = initial_migration_state()
        _state["migration"]["recovery_required"] = prior_recovery + [restart_banner]

        restart_required = not applied_live or (apply_reason is not None)

        if rotation_failed:
            # HTTP 207 Multi-Status: primary action succeeded, rotation failed.
            body = {
                "state": "LIVE",
                "trial_adapter_archive_path": archive_path,
                "rollback_pre_mortem_backup_path": rollback_pre_mortem_path,
                "restart_required": restart_required,
                "restart_hint": _RESTART_HINT,
                "applied_live": applied_live,
                "restart_required_reason": apply_reason,
                "auto_restart_scheduled": False,
                "restart_eligible": restart_eligible,
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
            restart_required=restart_required,
            restart_hint=_RESTART_HINT,
            applied_live=applied_live,
            restart_required_reason=apply_reason,
            auto_restart_scheduled=False,
            restart_eligible=restart_eligible,
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
        ``["snapshot_bundle"]`` (self-contained recovery bundle).  The
        deprecated per-artifact kinds ``"config"``, ``"graph"``, and
        ``"registry"`` are still accepted for backward compatibility.
    label:
        Optional annotation written into each slot sidecar.
    tier:
        Retention tier the slot is filed under.  Defaults to ``"manual"``
        (operator-initiated, time-immune).  The scheduled systemd timer
        delegates here with ``tier="daily"`` so the bundle is captured under
        the daily retention policy.
    """

    kinds: list[str] | None = None
    tier: str = "manual"
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
        Retention tier the slot was filed under — ``"manual"`` for
        operator-initiated backups, ``"daily"`` for the scheduled timer.
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
    restore_config:
        When ``True``, atomically restore the bundle's ``server.yaml`` to the
        live config path.  Default ``False`` — leave the live config untouched.
        Only applicable to ``snapshot_bundle`` restores; ignored for
        ``config``-kind restores (those always replace the config).
    """

    backup_id: str
    restore_config: bool = False


class BackupRestoreResponse(BaseModel):
    """Response body for ``POST /backup/restore``.

    Attributes
    ----------
    restored:
        Mapping of artifact kind / name → live path that was overwritten.
        For ``snapshot_bundle`` restores this maps adapter names to their new
        slot directories and includes ``"registry"`` and (optionally)
        ``"speaker_profiles"`` and ``"config"``.
    backed_up_pre_restore:
        Mapping of kind → safety backup slot path taken before restore.
        For ``snapshot_bundle`` restores the key is ``"bundle"`` and the value
        is the pre-restore safety bundle slot path (or ``""`` when skipped
        because the live store was empty).
    restart_required:
        Always ``True`` — the server must be restarted to load the restored
        config and re-mount adapters from the restored slots.
    restart_hint:
        Human-readable restart command.
    restored_adapters:
        List of adapter names restored from the bundle.  Empty for
        ``config``-kind restores.
    """

    restored: dict[str, str]
    backed_up_pre_restore: dict[str, str]
    restart_required: bool = True
    restart_hint: str
    restored_adapters: list[str] = []


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
    preserved_migration_window:
        Slots preserved by the 30-day window-immunity rule for migration tiers
        (``pre_migration`` and ``pre_base_swap``).  Rule 4.
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
    preserved_migration_window: list[str]
    would_delete_next: list[str]
    disk_usage_before: dict
    disk_usage_after: dict
    invalid_slots: list[list[str]]
    dry_run: bool


@app.get("/backup/list", response_model=BackupListResponse, dependencies=[Depends(require_admin)])
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


@app.post(
    "/backup/create",
    response_model=BackupCreateResponse,
    dependencies=[Depends(require_admin)],
)
async def backup_create(req: BackupCreateRequest):
    """Take an immediate backup of the requested artifacts.

    Delegates to ``run_scheduled_backup`` with the request's ``tier``
    (default ``"manual"``) and a per-call shallow-cloned config with
    ``.artifacts`` replaced by the request's ``kinds`` list (defaults to
    ``["snapshot_bundle"]``).  The scheduled systemd timer posts here with
    ``tier="daily"`` so the self-contained recovery bundle is captured under
    daily retention.

    ``snapshot_bundle`` produces a single self-contained bundle slot under
    ``backups_root/snapshot/`` containing the full recovery set (config,
    registry, adapter weights, speaker profiles).  The server holds the
    ``PARAMEM_DAILY_PASSPHRASE`` needed to decrypt registries for per-tier
    hash resolution, which is why scheduled backups are server-mediated.

    The deprecated per-artifact kinds ``"config"``, ``"graph"``, and
    ``"registry"`` are still accepted for backward compatibility.

    Persists the result via ``update_backup_state`` so the next ``/status``
    reflects the freshly-updated ``last_success_at``.

    Errors
    ------
    400 ``kind_invalid``
        When any entry in ``kinds`` is not a recognised artifact kind.
    """
    import dataclasses

    from fastapi import HTTPException

    from paramem.backup.runner import run_scheduled_backup
    from paramem.backup.state import update_backup_state

    _VALID_KINDS = {"config", "graph", "registry", "snapshot_bundle"}

    # Validate kinds.
    kinds = req.kinds
    if not kinds:
        kinds = ["snapshot_bundle"]

    for k in kinds:
        if k not in _VALID_KINDS:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "kind_invalid",
                    "message": (f"kind must be one of {sorted(_VALID_KINDS)}; got {k!r}"),
                },
            )

    _VALID_TIERS = {
        "daily",
        "weekly",
        "monthly",
        "yearly",
        "manual",
        "pre_migration",
        "trial_adapter",
    }
    if req.tier not in _VALID_TIERS:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "tier_invalid",
                "message": f"tier must be one of {sorted(_VALID_TIERS)}; got {req.tier!r}",
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
        tier=req.tier,
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


@app.post(
    "/backup/restore",
    response_model=BackupRestoreResponse,
    dependencies=[Depends(require_admin)],
)
async def backup_restore(req: BackupRestoreRequest):
    """Restore a backup atop the live store.

    Supports two restore kinds:

    - ``kind="config"`` — restore a single ``server.yaml`` from a per-artifact
      config slot.  The existing ``config``-kind branch is unchanged.
    - ``kind="snapshot_bundle"`` — restore a complete self-contained recovery
      set (adapter weights, per-tier registries, speaker profiles, and
      optionally config) from a bundle slot.

    Atomic restore sequence for ``snapshot_bundle`` (decrypt-probe, then
    safety bundle, then atomic per-file swaps, registry written LAST):

    1. Verify preconditions (no TRIAL/STAGING, no active consolidation, no
       background training).
    2. Locate the slot by ``backup_id``.
    3. Dispatch to the appropriate kind handler.
    4. For ``snapshot_bundle``:
       a. Verify all file hashes in ``bundle.meta.json`` against on-disk bytes.
       b. Decrypt-probe encrypted metadata files (BEFORE any mutation).
       c. Take a manual safety bundle of the current live state.
       d. Atomic restore via ``restore_bundle()``, registry written LAST.
       e. Append recovery banner + return ``restart_required=True``.

    Errors
    ------
    409 ``trial_active``
        Migration state is TRIAL.
    409 ``staging_active``
        Migration state is STAGING.
    409 ``consolidating``
        A consolidation run is in progress.
    409 ``training_active``
        A background training run is in progress.  Between registry-swap and
        restart a running background trainer could ``find_live_slot`` on the
        restored slot and stomp it with a checkpoint.  Refuse until training
        is idle.
    404 ``not_found``
        No slot with the given ``backup_id`` exists.
    400 ``restore_kind_not_supported``
        The slot kind is not ``config`` or ``snapshot_bundle``.
    500 ``decrypt_no_key``
        A backup file is age-encrypted but the daily identity is not loaded.
    500 ``decrypt_invalid_token``
        Decryption failed (stale daily identity or corrupted backup).
    500 ``bundle_corrupt``
        A file hash in ``bundle.meta.json`` does not match on-disk bytes.
    500 ``config_restore_failed``
        Atomic rename of the restore temp file failed.
    500 ``bundle_restore_failed``
        Unexpected error during bundle restore.
    """
    from fastapi import HTTPException

    from paramem.backup.backup import read as backup_read
    from paramem.backup.backup import restore_bundle as _restore_bundle
    from paramem.backup.backup import write as backup_write_fn
    from paramem.backup.enumerate import enumerate_backups
    from paramem.backup.types import (
        BundleManifestError,
        FingerprintMismatchError,
        RestoreAbortedError,
    )
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

    # Background trainer check (S3 from plan-review): a live BG trainer could
    # call find_live_slot on the restored slot between the registry swap and the
    # server restart, stomping the restored weights with a checkpoint.
    _bg_trainer = _state.get("background_trainer")
    if _bg_trainer is not None and getattr(_bg_trainer, "is_training", False):
        raise HTTPException(
            status_code=409,
            detail={
                "error": "training_active",
                "message": (
                    "Background training is active. Stop or wait for training to complete "
                    "before restoring a bundle — a running trainer can overwrite the "
                    "restored adapter slot with a checkpoint."
                ),
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

    live_config_path = (
        Path(_state["config_path"]) if _state.get("config_path") else Path("configs/server.yaml")
    )

    # --- Step 3: Dispatch by kind ---

    # -------------------------------------------------------------------------
    # SNAPSHOT_BUNDLE branch
    # -------------------------------------------------------------------------
    if target_record.kind == ArtifactKind.SNAPSHOT_BUNDLE:
        # Derive data_dir from config.paths.data when available.
        if config is not None:
            try:
                data_dir = config.paths.data.resolve()
            except (AttributeError, TypeError):
                data_dir = Path("data/ha").resolve()
        else:
            data_dir = Path("data/ha").resolve()

        try:
            result = _restore_bundle(
                target_record.slot_dir,
                data_dir=data_dir,
                config_path=live_config_path,
                restore_config=req.restore_config,
            )
        except BundleManifestError as exc:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "bundle_corrupt",
                    "message": f"Bundle manifest invalid or schema-mismatched: {exc}",
                },
            ) from exc
        except FingerprintMismatchError as exc:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "bundle_corrupt",
                    "message": (
                        f"Bundle file hash mismatch — bundle is corrupt: {exc}. "
                        "No live mutation has occurred."
                    ),
                },
            ) from exc
        except RuntimeError as exc:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "decrypt_no_key",
                    "message": (
                        f"Bundle contains age-encrypted metadata but the daily identity "
                        f"is not loaded: {exc}"
                    ),
                },
            ) from exc
        except RestoreAbortedError as exc:
            # S3: restore_bundle raises RestoreAbortedError when the atomic
            # write phase (step 5) fails after the safety bundle was already
            # captured (step 4).  Surface the safety_slot path so the operator
            # can recover without searching server logs.
            safety_path = str(exc.safety_slot) if exc.safety_slot is not None else ""
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "bundle_restore_failed",
                    "message": str(exc),
                    "safety_slot": safety_path,
                },
            ) from exc
        except Exception as exc:  # noqa: BLE001
            # Catch pyrage.DecryptError (not a RuntimeError subclass) and any
            # unexpected restore-phase OSError that was not wrapped into
            # RestoreAbortedError (e.g. errors before step 5 starts).
            error_code = "bundle_restore_failed"
            exc_str = str(exc)
            # Distinguish decrypt failures (wrong recipient) from other errors.
            if "decrypt" in type(exc).__name__.lower() or "decrypt" in exc_str.lower():
                error_code = "decrypt_invalid_token"
            raise HTTPException(
                status_code=500,
                detail={
                    "error": error_code,
                    "message": (
                        f"Bundle restore failed: {exc}. "
                        "If a safety bundle was captured before the error, "
                        "check the server log for its path."
                    ),
                },
            ) from exc

        # Build restored dict: adapter name → new slot path, plus registry/profiles/config.
        restored_map: dict[str, str] = {}
        for adapter_name in result.restored_adapters:
            from paramem.memory.interim_adapter import adapter_slot_root_for_name

            tier_root = adapter_slot_root_for_name(data_dir / "adapters", adapter_name)
            restored_map[adapter_name] = str(tier_root)
        if (data_dir / "registry" / "key_metadata.json").exists():
            restored_map["registry"] = str(data_dir / "registry" / "key_metadata.json")
        if (data_dir / "speaker_profiles.json").exists():
            restored_map["speaker_profiles"] = str(data_dir / "speaker_profiles.json")
        if result.restored_config:
            restored_map["config"] = str(live_config_path)

        safety_slot_str = str(result.safety_slot) if result.safety_slot is not None else ""

        # Append recovery banner.
        if "migration" not in _state:
            _state["migration"] = initial_migration_state()
        if "recovery_required" not in _state["migration"]:
            _state["migration"]["recovery_required"] = []
        _state["migration"]["recovery_required"].append(
            f"Restored snapshot_bundle from backup {req.backup_id} — "
            "restart server to re-mount adapters from restored slots."
        )

        return BackupRestoreResponse(
            restored=restored_map,
            backed_up_pre_restore={"bundle": safety_slot_str},
            restart_required=True,
            restart_hint="systemctl --user restart paramem-server",
            restored_adapters=result.restored_adapters,
        )

    # -------------------------------------------------------------------------
    # CONFIG branch (unchanged)
    # -------------------------------------------------------------------------
    if target_record.kind != ArtifactKind.CONFIG:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "restore_kind_not_supported",
                "message": (
                    f"Only kind='config' and kind='snapshot_bundle' restores are supported. "
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
        restored_adapters=[],
    )


@app.post(
    "/backup/prune",
    response_model=BackupPruneResponse,
    dependencies=[Depends(require_admin)],
)
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
        preserved_migration_window=[str(p) for p in pr.preserved_migration_window],
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


def _last_full_consolidation_window(adapter_dir: Path) -> "str | None":
    """Return the window_stamp recorded on the most recent main episodic slot.

    The canonical main ``episodic`` adapter is only written by full
    consolidation paths (``_save_adapters`` and
    ``consolidate_interim_adapters``); both stamp ``window_stamp`` with the
    full-consolidation cadence boundary that was active at save time. So the
    lex-max slot's ``meta.json.window_stamp`` is the canonical record of
    "which window did the last full cycle consolidate?"

    Returns ``None`` when:
      * There is no main episodic slot yet (fresh install).
      * ``meta.json`` is missing or unreadable.
      * The recorded ``window_stamp`` is empty (legacy v1 manifest, or a
        synthesized fallback that did not know its window).

    Callers treat ``None`` as "no main slot exists yet — defer to the
    interim path." Fresh installs and post-migration states bootstrap by
    extracting pending sessions into an interim adapter first; only after
    at least one interim has been written does the full cycle have anything
    to consolidate (see ``_is_full_cycle_due``).
    """
    from paramem.memory.interim_adapter import INTERIM_DIR_PREFIX

    episodic_dir = adapter_dir / "episodic"
    if not episodic_dir.is_dir():
        return None
    slots = sorted(
        d
        for d in episodic_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".") and not d.name.startswith(INTERIM_DIR_PREFIX)
    )
    if not slots:
        return None
    meta_path = slots[-1] / "meta.json"
    if not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text())
        ws = meta.get("window_stamp", "")
        return ws if ws else None
    except Exception:
        return None


def _is_full_cycle_due(config) -> bool:
    """Window-stamp gate: has the current full-consolidation window already
    been consolidated?

    Compares two stamps produced by the same flooring primitive
    (``current_full_consolidation_stamp``):
      * ``current``: ``floor(now, full_cycle_period)``.
      * ``last``: the ``window_stamp`` recorded on the lex-max main
        ``episodic/<ts>/meta.json`` at save time.

    Identity check — no wall-clock-elapsed math, no derived-at-read flooring.
    Two ticks inside the same window agree on ``current`` so re-firing is
    idempotent. The first tick of a new window returns ``current != last``
    and dispatches the full cycle.

    Returns ``False`` when ``refresh_cadence`` is disabled (period string is
    empty → manual-only operation).

    ``last is None`` (no main slot on disk) has two sub-cases:

    * **Interim-only / no-main** (the production catch-up state): at least one
      ``episodic/interim_<stamp>/`` directory exists. The fold
      (``consolidate_interim_adapters``) has real content to work with, so
      return ``True`` to trigger it. Once the fold completes it writes the
      main slot, making ``last`` non-``None`` on the next tick — no re-fire
      risk.

    * **Fresh install / no interim**: the episodic tree is empty or absent.
      The full cycle would be a no-op with no main slot written, so the gate
      would re-fire forever. Return ``False`` to stay on the interim path
      (which extracts pending sessions first).
    """
    from paramem.memory.interim_adapter import (
        current_full_consolidation_stamp,
        iter_interim_dirs,
    )

    period = config.consolidation.consolidation_period_string
    if not period:
        return False
    current = current_full_consolidation_stamp(period)
    if not current:
        return False
    last = _last_full_consolidation_window(config.adapter_dir)
    if last is None:
        return any(iter_interim_dirs(config.adapter_dir))
    return current != last


def _maybe_trigger_scheduled_consolidation() -> str:
    """Gate + dispatch the scheduled tick.

    Two production routes share this entry:

    1. **Full cycle** (every ``refresh_cadence × max_interim_count``): collapse
       all ``episodic_interim_*`` slots into the main episodic / semantic /
       procedural adapters via ``loop.consolidate_interim_adapters``. Re-saves
       main slot manifests in lockstep with the latest registry, restoring
       boot-time mount-ability that interim cycles otherwise drift away from.
    2. **Interim cycle** (every other tick): extract any pending sessions and
       train them into ``episodic_interim_<stamp>`` via
       ``_extract_and_start_training``. Main adapters are not touched.

    Returns a short status string. GPU-busy states return ``deferred_*`` so
    the caller can distinguish a missed-but-rescheduled tick from a true
    no-op. The systemd timer fires again on its next wall-clock tick and
    retries.
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

    config = _state["config"]

    debounce_s = config.consolidation.training_idle_debounce_s
    last_chat = _state.get("last_chat_monotonic")
    # Check last_chat first so MagicMock configs (tests that patch _state with
    # a minimal mock config) short-circuit before the int comparison fires.
    if last_chat is not None and debounce_s > 0 and (time.monotonic() - last_chat) < debounce_s:
        elapsed = time.monotonic() - last_chat
        logger.info(
            "Scheduler tick: chat %.1fs ago < debounce %ds — deferred",
            elapsed,
            debounce_s,
        )
        return "deferred_idle"

    buffer = _state["session_buffer"]

    # Retroactive voice-match claim: scan orphan sessions against every
    # enrolled speaker. Attributes sessions whose embeddings match an
    # existing profile at high confidence. Cheap — centroids are cached.
    _retro_claim_orphan_sessions()

    # Active-store migration gate: when a mode-switch was detected at startup
    # (or an in-progress migration was interrupted), every consolidation tick
    # routes to the migration sync until all tiers have cleared the 1.0
    # recall gate. This pre-empts the full/interim cycle gates because the
    # active store is not yet coherent with the operator's yaml mode.
    if _state.get("pending_rehydration", False):
        if _state.get("store_load_degraded", False):
            logger.warning(
                "Scheduler tick: active-store migration pending but store_load_degraded=True "
                "— refusing to dispatch (boot-time registry load failed; resolve the corrupt "
                "registry file and restart the server to retry)"
            )
            return "migration_skipped_degraded"
        logger.info("Scheduler tick: active-store migration pending — running migration")
        _state["last_consolidation_error"] = None
        _state["consolidating"] = True
        event_loop = asyncio.get_running_loop()
        future = event_loop.run_in_executor(None, _run_active_store_migration_sync)
        future.add_done_callback(_scheduled_extract_done_callback)
        return "started_migration"

    # Full-cycle gate: when the period has elapsed, route to full
    # consolidation regardless of whether sessions are pending. Pending
    # sessions wait one tick (~refresh_cadence) for their interim write —
    # acceptable given the period is much longer than the cadence.
    if _is_full_cycle_due(config):
        logger.info("Scheduler tick: full cycle due — running consolidate_interim_adapters")
        # Clear stale error from a prior failed cycle — a new attempt is
        # starting; the done-callback will re-populate on failure.
        _state["last_consolidation_error"] = None
        _state["consolidating"] = True
        event_loop = asyncio.get_running_loop()
        future = event_loop.run_in_executor(None, _run_full_consolidation_sync)
        future.add_done_callback(_scheduled_extract_done_callback)
        return "started_full"

    pending = buffer.get_pending()
    if not pending:
        logger.info("Scheduler tick: no pending sessions — noop")
        return "noop_no_pending"
    if not any(s.get("speaker_id") for s in pending):
        logger.info("Scheduler tick: no sessions with speaker_id — noop")
        return "noop_no_speaker"

    logger.info("Scheduler tick: %d pending sessions, starting extract + train", len(pending))
    _state["last_consolidation_error"] = None
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


# _evict_voice_pipeline and _load_voice_pipeline were deleted here.
# All call sites have been migrated to _set_voice_pipeline_profile.


# ---------------------------------------------------------------------------
# Voice-pipeline profile helpers
# ---------------------------------------------------------------------------


def _build_cpu_tts_config(tts_config: TTSConfig) -> TTSConfig:
    """Return a CPU-only TTSConfig derived from ``tts_config``.

    Sets top-level ``device="cpu"`` and resets every voice's ``device`` to
    ``""`` (inherit) so per-voice cuda overrides do not survive the cpu
    profile (G1).  All other fields are preserved verbatim.
    """
    cpu_voices = {
        lang: TTSVoiceConfig(
            engine=voice_cfg.engine,
            model=voice_cfg.model,
            language_name=voice_cfg.language_name,
            device="",  # reset: inherit top-level "cpu"
        )
        for lang, voice_cfg in (tts_config.voices or {}).items()
    }
    return TTSConfig(
        enabled=tts_config.enabled,
        port=tts_config.port,
        device="cpu",
        default_language=tts_config.default_language,
        language_confidence_threshold=tts_config.language_confidence_threshold,
        model_dir=tts_config.model_dir,
        audio_chunk_bytes=tts_config.audio_chunk_bytes,
        voices=cpu_voices,
    )


def _target_profile() -> Literal["gpu", "cpu"]:
    """Return the profile the voice pipeline should restore to after a cycle.

    Cloud-only mode permanently targets cpu; local mode targets gpu.
    """
    return "cpu" if _state.get("mode") == "cloud-only" else "gpu"


def _set_voice_pipeline_profile(
    profile: Literal["gpu", "cpu"],
    *,
    lock_held: bool = False,
) -> None:
    """Atomically swap the active STT+TTS pair under the GPU lock.

    Idempotent: returns immediately (DEBUG) when the current profile already
    matches ``profile``.

    Atomic ordering (B2): construct/ensure NEW pair loaded → update
    ``_state["voice_box"]`` and ``_state["stt"]``/``_state["tts_manager"]``
    mirrors → unload OLD GPU pair (only on gpu→cpu; CPU pair is never torn
    down).

    Lock contract: if ``lock_held=True``, the caller already holds
    ``gpu_lock_sync()`` and no acquisition is attempted. Default
    (``False``) acquires the lock before entering the critical section.

    Best-effort: on STT/TTS load failure ``_state["voice_profile"]`` is
    updated to the target anyway, a WARN is emitted, and the GPU instance
    is left present-but-unloaded (existing degradation contract,
    wyoming_handler.py).
    """
    if profile == _state.get("voice_profile"):
        logger.debug("_set_voice_pipeline_profile: already %r — no-op", profile)
        return

    from paramem.server.gpu_lock import gpu_lock_sync
    from paramem.server.stt import WhisperSTT
    from paramem.server.tts import TTSManager

    config = _state["config"]

    ctx = gpu_lock_sync() if not lock_held else nullcontext()
    with ctx:
        if profile == "gpu":
            # Flush allocator-pool slack from prior cycle work before
            # WhisperSTT.load() — vram_measure reads mem_get_info before the
            # load, and uncollapsed pool inflates "used", skewing the delta.
            safe_empty_cache()
            # Lazy-construct GPU pair when absent or after a prior gpu→cpu flip.
            if config.stt.enabled and _state["stt_gpu"] is None:
                _state["stt_gpu"] = WhisperSTT(
                    model_name=config.stt.model,
                    device=config.stt.device,  # "auto" passes through (G2)
                    compute_type=config.stt.compute_type,
                    language=config.stt.language,
                    beam_size=config.stt.beam_size,
                    vad_filter=config.stt.vad_filter,
                )
            if config.tts.enabled and _state["tts_gpu"] is None:
                _state["tts_gpu"] = TTSManager(config.tts)  # per-voice overrides respected (G1)

            # Ensure loaded.
            stt_gpu = _state.get("stt_gpu")
            if config.stt.enabled and stt_gpu is not None and not stt_gpu.is_loaded:
                try:
                    with vram_measure("stt-gpu-profile"):
                        loaded = stt_gpu.load()
                except Exception:  # noqa: BLE001
                    loaded = False
                if not loaded:
                    logger.warning(
                        "_set_voice_pipeline_profile('gpu'): STT load failed; voice path degraded"
                    )
            tts_gpu = _state.get("tts_gpu")
            if config.tts.enabled and tts_gpu is not None and not tts_gpu.is_loaded:
                try:
                    tts_gpu.load_all()
                except Exception:  # noqa: BLE001
                    logger.warning(
                        "_set_voice_pipeline_profile('gpu'): TTS load failed; voice path degraded"
                    )

            # Atomic flip: update box and mirrors.
            _state["voice_box"] = {
                "stt": _state.get("stt_gpu"),
                "tts_manager": _state.get("tts_gpu"),
            }
            _state["stt"] = _state["voice_box"]["stt"]
            _state["tts_manager"] = _state["voice_box"]["tts_manager"]
            # Transition from cpu: nothing to unload (CPU pair stays resident).

        else:  # profile == "cpu"
            # CPU pair is always resident (loaded at startup); no construction.
            old_stt_gpu = _state.get("stt_gpu")
            old_tts_gpu = _state.get("tts_gpu")

            # Atomic flip: box and mirrors point at CPU pair BEFORE unloading GPU pair (B2).
            _state["voice_box"] = {
                "stt": _state.get("stt_cpu"),
                "tts_manager": _state.get("tts_cpu"),
            }
            _state["stt"] = _state["voice_box"]["stt"]
            _state["tts_manager"] = _state["voice_box"]["tts_manager"]

            # Unload old GPU pair and reclaim VRAM.
            if old_stt_gpu is not None:
                try:
                    old_stt_gpu.unload()
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "_set_voice_pipeline_profile('cpu'): stt_gpu.unload() failed; continuing"
                    )
                _state["stt_gpu"] = None

            if old_tts_gpu is not None:
                try:
                    old_tts_gpu.unload_all()
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "_set_voice_pipeline_profile('cpu'): tts_gpu.unload_all() failed; "
                        "continuing"
                    )
                _state["tts_gpu"] = None

            safe_empty_cache()

    _state["voice_profile"] = profile
    logger.info("Voice pipeline profile: %r", profile)


def _await_bg_cycle(
    *,
    loop,
    config,
    episodic_rels: list,
    procedural_rels: list,
    speaker_id: str,
    mode: "Literal['simulate', 'train']",
    run_label: str,
    schedule: str = "",
    max_interim_count: int = 7,
    inference_fallback_adapter: str = "episodic",
) -> dict:
    """Submit ``run_consolidation_cycle`` to the BG trainer and block until done.

    Constructs a fresh :class:`~paramem.server.background_trainer.BackgroundTrainer`,
    submits the cycle as a callable, and blocks on a ``threading.Event`` until
    the worker finishes.  The BG worker thread acquires ``gpu_lock_sync()``
    internally, so the caller must NOT already hold the lock (that would
    deadlock — the non-reentrant ``threading.Lock`` in ``gpu_lock.py`` cannot
    be acquired twice on the same thread).

    Use this helper from sites that do NOT hold the GPU lock (e.g. the
    simulate and train branches of ``_extract_and_start_training``).  Sites
    that already hold the GPU lock (e.g. the trial-path caller of
    ``_run_extraction_phase``) must call ``loop.run_consolidation_cycle``
    directly.

    Args:
        loop: Active :class:`~paramem.training.consolidation.ConsolidationLoop`.
        config: Live :class:`~paramem.server.config.ServerConfig` from ``_state``.
        episodic_rels: Extracted episodic relations for this cycle.
        procedural_rels: Extracted procedural relations for this cycle.
        speaker_id: Default speaker tag for relations without one.
        mode: ``"train"`` writes adapter weights; ``"simulate"`` writes sidecar JSON.
        run_label: Traceability tag passed to ``run_consolidation_cycle``.
        schedule: Consolidation refresh-cadence string for stamp computation.
        max_interim_count: Cap on concurrent interim adapters.
        inference_fallback_adapter: Adapter the BG pause mechanism switches to
            when inference is requested mid-cycle.  Defaults to ``"episodic"``.

    Returns:
        Result dict from ``run_consolidation_cycle``:
        ``{"triples_extracted", "new_keys", "adapter_name", "mode", "venue",
        "error"}``.

    Raises:
        Exception: Re-raises any exception thrown inside the BG worker.
    """
    from paramem.server.background_trainer import BackgroundTrainer

    bt = BackgroundTrainer(
        model=_state["model"],
        tokenizer=_state["tokenizer"],
        training_config=config.training_config,
        output_dir=config.adapter_dir,
        thermal_policy=ThermalPolicy.from_consolidation_config(config.consolidation),
        preload_cache=config.inference.preload_cache,
    )
    # Wire the throwaway trainer into both _state and the consolidation loop
    # so /chat's abort_for_inference targets the same BT instance whose
    # training_hooks_for_job is installed inside run_consolidation_cycle.
    # Without the loop wiring, the loop's _build_training_hooks would close
    # over a stale (or None) _bg_trainer and silently drop the abort signal.
    _state["background_trainer"] = bt
    loop._bg_trainer = bt

    _result_holder: dict = {}

    def _run() -> None:
        _result_holder["result"] = loop.run_consolidation_cycle(
            episodic_rels,
            procedural_rels,
            speaker_id=speaker_id,
            mode=mode,
            run_label=run_label,
            schedule=schedule,
            max_interim_count=max_interim_count,
        )
        # Propagate any PeftModel handle rebinding from create_interim_adapter.
        _state["model"] = loop.model

    bt.submit_and_wait(_run, inference_fallback_adapter=inference_fallback_adapter)
    return _result_holder["result"]


def _run_extraction_phase(
    loop,
    mark_callback=None,
) -> dict:
    """Extract all pending sessions and train once (full-cycle path).

    Direct port of the deleted ``paramem.server.consolidation.run_consolidation``.
    Reads ``session_buffer`` and ``config`` from ``_state``; isolated from
    ``_state`` mutations so the trial path can run it with a separate loop
    without touching live state.

    Parameters
    ----------
    loop:
        :class:`~paramem.server.consolidation.ConsolidationLoop` instance.
        Must be pre-constructed by the caller (trial or production).
    mark_callback:
        Optional callable accepting a list of session IDs.  When ``None``,
        ``session_buffer.mark_consolidated`` is called (production behaviour).
        Pass ``lambda _: None`` for the trial path so pending sessions remain
        in the buffer (spec L364 — "transcript sweeper blocks archive+delete").

    Returns a result dict including the loop instance for reuse.

    D2 classification: all class-1 mock sites (prevent real execution only;
    ``_run_extraction_phase`` may close over ``_state`` directly).
    """
    import time

    from paramem.server.consolidation import (
        _dedup_episodic,
        _dedup_procedural,
        _do_mark_consolidated,
        _increment_key_sessions,
        _promote_mature_keys,
        _save_key_metadata,
        session_retention_dir,
    )

    config = _state["config"]
    session_buffer = _state["session_buffer"]

    if not config.adapters.episodic.enabled:
        logger.info("Episodic adapter is disabled in config, skipping consolidation")
        return {"status": "disabled", "sessions": 0, "loop": loop}

    start_time = time.time()

    pending = session_buffer.get_pending()
    if not pending:
        logger.info("No pending sessions to consolidate")
        return {"status": "no_pending", "sessions": 0, "loop": loop}

    logger.info("Consolidating %d pending sessions", len(pending))

    # --- Phase 1: Extract all sessions ---
    all_episodic_rels = []
    all_procedural_rels = []
    session_ids = []
    speaker_ids = []
    total_relations = 0

    ha_context = None
    ha_client = _state.get("ha_client")
    if ha_client is not None:
        ha_context = ha_client.get_home_context()
    speaker_store = _state.get("speaker_store")

    for session in pending:
        session_id = session["session_id"]
        transcript = session["transcript"]
        session_speaker_id = session.get("speaker_id")

        session_ids.append(session_id)

        if not session_speaker_id:
            logger.debug(
                "Session %s has no speaker_id — skipping (text-only, no voice attribution)",
                session_id,
            )
            continue

        speaker_name = None
        if speaker_store is not None:
            try:
                speaker_name = speaker_store.get_name(session_speaker_id)
            except Exception as e:
                logger.warning("speaker_store.get_name(%s) failed: %s", session_speaker_id, e)
        with vram_scope(session_id):
            episodic_rels, procedural_rels = loop.extract_session(
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

        for qa in episodic_rels:
            qa["speaker_id"] = session_speaker_id
        for rel in procedural_rels:
            rel["speaker_id"] = session_speaker_id

        all_episodic_rels.extend(episodic_rels)
        all_procedural_rels.extend(procedural_rels)
        total_relations += len(episodic_rels) + len(procedural_rels)
        speaker_ids.append(session_speaker_id)

        logger.info(
            "Extracted session %s: %d episodic, %d procedural relations",
            session_id,
            len(episodic_rels),
            len(procedural_rels),
        )

        if loop.shutdown_requested:
            logger.info("Shutdown requested — stopping extraction after %s", session_id)
            break

    if not all_episodic_rels and not all_procedural_rels:
        logger.info("No relations extracted — skipping training")
        _do_mark_consolidated(
            session_buffer,
            session_ids,
            mark_callback,
            retention_dir=session_retention_dir(loop, config),
        )
        return {
            "status": "no_facts",
            "sessions": len(session_ids),
            "loop": loop,
        }

    # --- Cross-session dedup on (subject, predicate, object) identity ---
    pre_ep, pre_pr = len(all_episodic_rels), len(all_procedural_rels)
    all_episodic_rels = _dedup_episodic(all_episodic_rels)
    all_procedural_rels = _dedup_procedural(all_procedural_rels)
    if pre_ep != len(all_episodic_rels) or pre_pr != len(all_procedural_rels):
        logger.info(
            "Dedup: episodic %d→%d, procedural %d→%d",
            pre_ep,
            len(all_episodic_rels),
            pre_pr,
            len(all_procedural_rels),
        )

    simulate = config.consolidation.mode == "simulate"
    if simulate:
        primary_speaker_sim = speaker_ids[-1] if speaker_ids else ""
        try:
            # Callsite 1: simulate branch.  The GPU lock is already held by the
            # trial-path caller (``with gpu_lock_sync()`` at the migration
            # endpoint), so ``_await_bg_cycle`` cannot be used here — its BG
            # worker would deadlock trying to re-acquire the non-reentrant lock.
            # Call ``run_consolidation_cycle`` directly: the GPU lock is held,
            # the cycle is CPU-bound key-preparation + JSON persistence only,
            # and the pause-for-inference contract is satisfied by the outer lock.
            sim_result = loop.run_consolidation_cycle(
                all_episodic_rels,
                all_procedural_rels,
                speaker_id=primary_speaker_sim,
                mode="simulate",
                run_label=f"full-{primary_speaker_sim or 'anon'}",
                schedule=config.consolidation.refresh_cadence,
                max_interim_count=config.consolidation.max_interim_count,
            )
            newly_promoted = _promote_mature_keys(loop, config)
            _save_key_metadata(loop, config)
        except Exception:
            logger.exception(
                "Simulated consolidation failed — leaving %d sessions pending",
                len(session_ids),
            )
            raise

        _do_mark_consolidated(
            session_buffer,
            session_ids,
            mark_callback,
            retention_dir=session_retention_dir(loop, config),
        )
        elapsed = time.time() - start_time
        summary = {
            "status": "simulated",
            "sessions": len(session_ids),
            "total_relations": total_relations,
            "newly_promoted": len(newly_promoted),
            "episodic_rels": len(all_episodic_rels),
            "procedural_rels": len(all_procedural_rels),
            "episodic_keys": loop.store.simhash_count_in_tier("episodic"),
            "semantic_keys": loop.store.simhash_count_in_tier("semantic"),
            "procedural_keys": loop.store.simhash_count_in_tier("procedural"),
            "elapsed_seconds": round(elapsed, 1),
            "simulated": sim_result.get("mode") == "simulated",
            "loop": loop,
        }
        logger.info("Simulation complete: %s", {k: v for k, v in summary.items() if k != "loop"})
        return summary

    # --- Phase 2: Train once ---
    logger.info(
        "Training on %d episodic + %d procedural relations",
        len(all_episodic_rels),
        len(all_procedural_rels),
    )
    primary_speaker = speaker_ids[-1] if speaker_ids else ""
    try:
        # Callsite 2: train branch.  Same lock-held constraint as callsite 1 —
        # the trial-path caller holds ``gpu_lock_sync()``, so ``_await_bg_cycle``
        # would deadlock.  Call ``run_consolidation_cycle`` directly.
        # The ``vram_scope("training")`` wrapper surfaces OutOfMemoryError as
        # VramExhausted("training") so the /status operator endpoint shows the
        # phase label.  The trial-path outer lock already serialises GPU access
        # for this synchronous call, so vram_scope adds only the phase label and
        # empty_cache discipline — not a new lock.
        from paramem.server.vram_guard import vram_scope as _vram_scope

        with _vram_scope("training"):
            train_result = loop.run_consolidation_cycle(
                all_episodic_rels,
                all_procedural_rels,
                speaker_id=primary_speaker,
                mode="train",
                run_label=f"full-{primary_speaker or 'anon'}",
                schedule=config.consolidation.refresh_cadence,
                max_interim_count=config.consolidation.max_interim_count,
            )
        newly_promoted = _promote_mature_keys(loop, config)
        loop._save_adapters()
        _save_key_metadata(loop, config)
    except Exception:
        logger.exception(
            "Consolidation failed during train/save — leaving %d sessions pending",
            len(session_ids),
        )
        raise

    _do_mark_consolidated(
        session_buffer,
        session_ids,
        mark_callback,
        retention_dir=session_retention_dir(loop, config),
    )

    elapsed = time.time() - start_time
    summary = {
        "status": "complete",
        "sessions": len(session_ids),
        "total_relations": total_relations,
        "newly_promoted": len(newly_promoted),
        "episodic_keys": loop.store.simhash_count_in_tier("episodic"),
        "semantic_keys": loop.store.simhash_count_in_tier("semantic"),
        "procedural_keys": loop.store.simhash_count_in_tier("procedural"),
        "train_loss": train_result.get("train_loss"),
        "elapsed_seconds": round(elapsed, 1),
        "loop": loop,
    }
    logger.info(
        "Consolidation complete: %s",
        {k: v for k, v in summary.items() if k != "loop"},
    )
    return summary


def _scheduled_extract_done_callback(future):
    """Clear the consolidating flag only if the extraction phase failed.

    On success, _extract_and_start_training has submitted the interim
    training job to the BG trainer and the flag will be cleared by the
    job's _finalize_interim closure when the worker thread completes.

    On VramExhausted, populate ``_state["last_consolidation_error"]`` so
    the failure is visible via ``/status`` without scraping logs. Other
    exceptions still surface only through the loud ``logger.exception``
    line (operators tail the journal).

    Self-healing voice restore: this callback is attached to three executor
    entry points — ``_extract_and_start_training``, ``_run_full_consolidation_sync``,
    and ``_run_active_store_migration_sync``.  On any failure path it calls
    ``_set_voice_pipeline_profile`` unconditionally; idempotency inside that
    function makes this safe on paths that did not evict voice.  Lock state at
    this site is ``lock_held=False``: the callback runs on the event-loop thread
    after the executor future has already returned.
    """
    exc = future.exception()
    if exc:
        logger.error("Scheduled extraction failed: %s", exc, exc_info=exc)
        if isinstance(exc, VramExhausted):
            phase = exc.args[0] if exc.args else "unknown"
            _state["last_consolidation_error"] = {
                "type": "vram_exhausted",
                "phase": phase,
                "at": datetime.now(timezone.utc).isoformat(),
            }
        # Voice was evicted at the cycle's start (doc-only path). On
        # exception the cycle's normal end-of-cycle restore was not
        # reached; restore here so STT/TTS aren't stuck unloaded until
        # the next /chat triggers a wyoming reconnect.
        try:
            _set_voice_pipeline_profile(_target_profile(), lock_held=False)
        except Exception:
            logger.exception("Voice restore on extraction failure path raised; ignoring")
        _state["consolidating"] = False


def _extract_and_start_training():
    """Extract pending sessions and submit a single interim-training job.

    Runs in executor thread.  Phase 1 (extraction) holds the GPU lock and
    produces the per-batch (episodic_rels, procedural_rels) tuple.  Phase 2
    submits one callable to ``BackgroundTrainer`` that mints a fresh
    ``episodic_interim_<stamp>`` slot (or absorbs into the newest existing
    when the cap is hit) and trains the batch into it.  Main adapters are
    only updated by the full-cycle path that calls
    ``consolidate_interim_adapters``.
    """
    from paramem.server.background_trainer import BackgroundTrainer
    from paramem.server.consolidation import (
        _increment_key_sessions,
        _promote_mature_keys,
        _save_key_metadata,
        create_consolidation_loop,
        session_retention_dir,
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
            _state["memory_store"],
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
    all_episodic_rels = []
    all_procedural_rels = []
    session_ids = []
    speaker_ids = []
    # Sessions whose extract_session call exhausted VRAM. Per-chunk
    # isolation: a single bad chunk (e.g. a pathologically dense
    # document) must not poison the whole cycle. Failed chunks are NOT
    # marked_consolidated so they remain pending for the next cycle,
    # which starts on a fresh cache (lazy STT/TTS reload only happens
    # on voice activity, so the bg cycle still sees the headroom).
    failed_session_ids: set[str] = set()

    # Voice eviction triggers when the pending batch contains ANY document
    # session. Document chunks have no density bound — a dense ~1500-word
    # resume chunk is the regime that exhausts VRAM on this 8 GiB host once
    # the ~1.5 GiB STT(Whisper)+TTS(Piper) GPU pair is added on top of the
    # 4-bit Mistral 7B base and the extraction chain's ~6 GiB working-set
    # peak. A *mixed* batch (one transcript probe + several dense docs) is
    # the case that bit us: under the prior all()-predicate one transcript
    # session kept voice resident through the dense doc extraction and the
    # plausibility filter's KV-cache growth OOM'd mid-generate. Eviction is
    # cheap — the CPU STT/TTS pair stays resident (loaded at startup), so
    # voice still works during the cycle, just on CPU; the GPU pair is
    # lazily reconstructed on the next voice activity or restored to
    # _target_profile() at cycle end. A pure-transcript batch keeps the GPU
    # voice pair resident: those sessions are produced by turn-by-turn
    # dialog, are not in the dense-document regime, and likely imply recent
    # voice activity where the lazy GPU reload would add latency.
    evict_voice_for_cycle = bool(pending) and any(
        s.get("source_type") == "document" for s in pending
    )

    with gpu_lock_sync():
        if evict_voice_for_cycle:
            _set_voice_pipeline_profile("cpu", lock_held=True)
        for session in pending:
            session_id = session["session_id"]
            transcript = session["transcript"]
            session_speaker_id = session.get("speaker_id")
            session_ids.append(session_id)

            if not session_speaker_id:
                logger.warning(
                    "Skipping session %s: no speaker_id — cannot attribute extraction",
                    session_id,
                )
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

            # vram_scope mirrors the full-cycle path in paramem/server/
            # consolidation.py:335 — empty_cache between sessions and OOM
            # → VramExhausted so the done-callback populates
            # last_consolidation_error visibly via /status.
            try:
                # Pre-chunk headroom check: warn (no abort) when free VRAM has
                # dropped below the configured KV-cache buffer. vram_scope catches
                # an actual OOM mid-generate; this is the early operator signal
                # that the booked headroom is being consumed.
                check_vram_headroom(
                    session_id,
                    int(config.vram.vram_cache_headroom_gib * 2**30),
                    _state,
                )
                with vram_scope(session_id):
                    episodic_rels, procedural_rels = loop.extract_session(
                        transcript,
                        session_id,
                        speaker_id=session_speaker_id,
                        speaker_name=speaker_name,
                        ha_context=ha_context,
                        stt_correction=config.consolidation.extraction_stt_correction,
                        ha_validation=config.consolidation.extraction_ha_validation,
                        noise_filter=config.consolidation.extraction_noise_filter,
                        noise_filter_model=config.consolidation.extraction_noise_filter_model,
                        noise_filter_endpoint=config.consolidation.extraction_noise_filter_endpoint
                        or None,
                        ner_check=config.consolidation.extraction_ner_check,
                        ner_model=config.consolidation.extraction_ner_model,
                        plausibility_judge=config.consolidation.extraction_plausibility_judge,
                        plausibility_stage=config.consolidation.extraction_plausibility_stage,
                        verify_anonymization=config.consolidation.extraction_verify_anonymization,
                        source_type=session.get("source_type", "transcript"),
                    )
            except VramExhausted as exc:
                # Per-chunk isolation: log, record, skip — continue to next
                # chunk under a fresh cache (vram_scope's finally clause
                # already ran empty_cache before re-raising). Do NOT mark
                # this chunk consolidated; it stays pending for next cycle.
                phase = exc.args[0] if exc.args else "unknown"
                logger.warning(
                    "Chunk %s OOM at phase=%s — skipping; remains pending for retry",
                    session_id,
                    phase,
                )
                failed_session_ids.add(session_id)
                _state.setdefault("chunk_failures", []).append(
                    {
                        "session_id": session_id,
                        "phase": phase,
                        "at": datetime.now(timezone.utc).isoformat(),
                    }
                )
                continue
            except ExtractionFailed as exc:
                # Whole-cycle abort: unlike VramExhausted (resource constraint, per-
                # chunk isolation OK), ExtractionFailed means the extractor
                # actively refused to bake a degraded snapshot — proceeding
                # with the OTHER chunks would silently commit a partial CV /
                # document set, which is exactly the "keep pre-enrichment
                # facts and proceed" anti-pattern the design rejects.  Drop
                # everything extracted so far, leave ALL sessions pending
                # (including the ones already processed in this loop), and
                # let the next tick retry the whole batch.
                logger.error(
                    "Cycle aborted: chunk %s extraction failed at phase=%s — %s. "
                    "All %d session(s) in this batch remain pending; next cycle will retry.",
                    session_id,
                    exc.phase,
                    exc.reason,
                    len(session_ids),
                )
                _state.setdefault("chunk_failures", []).append(
                    {
                        "session_id": session_id,
                        "phase": exc.phase,
                        "reason": exc.reason,
                        "at": datetime.now(timezone.utc).isoformat(),
                    }
                )
                _state["last_consolidation_error"] = {
                    "type": "extraction_failed",
                    "phase": exc.phase,
                    "reason": exc.reason,
                    "session_id": session_id,
                    "at": datetime.now(timezone.utc).isoformat(),
                }
                # Reclaim voice pipeline if we evicted it earlier — the cycle
                # is dropping out without going through the normal finalize.
                # This abort handler runs inside the `with gpu_lock_sync():` at
                # L7190, so pass ``lock_held=True`` (the lock is non-reentrant;
                # ``lock_held=False`` would deadlock the executor thread and
                # leave ``_state["consolidating"]`` stuck at True, blocking the
                # retry path mandated by project_extraction_failure_fails_cycle).
                if evict_voice_for_cycle:
                    _set_voice_pipeline_profile(_target_profile(), lock_held=True)
                _state["consolidating"] = False
                return
            _increment_key_sessions(loop, session_id)

            for qa in episodic_rels:
                qa["speaker_id"] = session_speaker_id
            for rel in procedural_rels:
                rel["speaker_id"] = session_speaker_id

            all_episodic_rels.extend(episodic_rels)
            all_procedural_rels.extend(procedural_rels)
            speaker_ids.append(session_speaker_id)

    # Helper: mark only sessions whose extract_session call succeeded.
    # Failed chunks remain pending so the next cycle retries them.
    def _completed_session_ids() -> list[str]:
        return [sid for sid in session_ids if sid not in failed_session_ids]

    if not all_episodic_rels and not all_procedural_rels:
        logger.info("No relations extracted — skipping")
        session_buffer.mark_consolidated(
            _completed_session_ids(),
            retention_dir=session_retention_dir(loop, config),
        )

        if evict_voice_for_cycle:
            _set_voice_pipeline_profile(_target_profile(), lock_held=False)

        # State mutations + router reload — post to the event loop so the
        # router cache and inference path see post-cycle state atomically with
        # the consolidating flag clear.  Mode-agnostic AND yield-agnostic:
        # every consolidation cycle ends with router.reload() so /chat
        # handlers never see stale state for any reason.  When no new keyed
        # pairs were written, the reload is a no-op against unchanged disk
        # state — the routing-point invariant still holds, which is what
        # callers rely on when they treat document ingest and transcript
        # ingest as equivalent inputs (project_document_transcript_equivalence).
        def _finalize_no_facts() -> None:
            _state["last_consolidation"] = datetime.now(timezone.utc).isoformat()
            _state["router"].reload()
            _state["last_consolidation_result"] = {
                "status": "no_facts",
                "sessions": len(session_ids),
                "skipped_oom": len(failed_session_ids),
                "episodic_rels": 0,
                "procedural_rels": 0,
            }
            _state["consolidating"] = False

        aio_loop = _state.get("event_loop")
        if aio_loop is not None and aio_loop.is_running():
            aio_loop.call_soon_threadsafe(_finalize_no_facts)
        else:
            _finalize_no_facts()
        return

    # --- Simulate mode: peer storage backend ---
    # Same upstream pipeline as train (extraction → dedup → key assignment →
    # contradiction handling → SimHash registry); the persistence venue is
    # graph.json under adapter_dir/<tier>/ instead of LoRA weight updates.
    # mark_consolidated runs in both branches — sessions retire when their
    # work has been persisted, regardless of medium. Inference reads from
    # the graph.json via DiskMemorySource at retrieval time.

    if config.consolidation.mode == "simulate":
        primary_speaker_sim = speaker_ids[-1] if speaker_ids else ""
        # Callsite 3: post-session simulate.  The caller does NOT hold the GPU
        # lock at this point (the old ``with gpu_lock_sync()`` wrapper is
        # dropped — the BG worker thread acquires the lock internally via
        # ``_run_callable_queue``).  ``_await_bg_cycle`` submits the cycle and
        # blocks until the worker finishes, preserving the pause-for-inference
        # contract without a redundant outer lock acquisition.
        sim_result = _await_bg_cycle(
            loop=loop,
            config=config,
            episodic_rels=all_episodic_rels,
            procedural_rels=all_procedural_rels,
            speaker_id=primary_speaker_sim,
            mode="simulate",
            run_label=f"tick-{primary_speaker_sim or 'anon'}",
            schedule=config.consolidation.refresh_cadence,
            max_interim_count=config.consolidation.max_interim_count,
        )
        newly_promoted = _promote_mature_keys(loop, config)
        # _save_registry retired (Plan A, landed in commits 47df093 + e2217c1):
        # the combined SimHash registry at config.registry_path was not
        # maintained by interim or full-cycle production paths post-Phase-3+5,
        # and the temporal-query reader (filter_registry_by_date) — itself
        # retired in Plan A.3 — needed fields the writer never emitted.
        # Per-adapter simhash_registry_<adapter>.json files are now the
        # canonical source of truth; inference._load_simhash_registry combines
        # them at read time.
        _save_key_metadata(loop, config)
        # Per-tier graph.json is written by commit_tier_slot inside
        # run_consolidation_cycle; cycle_<N>/ snapshots are dropped.

        # Simulate is peer storage — retire successfully-extracted sessions
        # like train. OOM-skipped chunks stay pending for retry.
        session_buffer.mark_consolidated(
            _completed_session_ids(),
            retention_dir=session_retention_dir(loop, config),
        )

        if evict_voice_for_cycle:
            _set_voice_pipeline_profile(_target_profile(), lock_held=False)

        # State mutations + router reload — post to the event loop so the
        # router cache and inference path see the freshly-written state
        # atomically with the consolidating flag clear.  Mode-agnostic at the
        # routing point: mirrors `_finalize_interim` / `_finalize_full` so a
        # simulate-mode cycle is queryable without a server restart, identical
        # to a trained cycle.
        def _finalize_simulate() -> None:
            _state["last_consolidation"] = datetime.now(timezone.utc).isoformat()
            _state["router"].reload()
            # A later flip to train mode — or any code path that re-reads —
            # sees the stale snapshot from the last train cycle.
            _state["last_consolidation_result"] = {
                "status": "simulated",
                "sessions": len(session_ids),
                "skipped_oom": len(failed_session_ids),
                "episodic_rels": len(all_episodic_rels),
                "procedural_rels": len(all_procedural_rels),
                "newly_promoted": len(newly_promoted),
                "simulated": sim_result.get("mode") == "simulated",
            }
            _state["consolidating"] = False
            logger.info(
                "Simulation complete: %d episodic, %d procedural relations, %d promoted",
                len(all_episodic_rels),
                len(all_procedural_rels),
                len(newly_promoted),
            )

        aio_loop = _state.get("event_loop")
        if aio_loop is not None and aio_loop.is_running():
            aio_loop.call_soon_threadsafe(_finalize_simulate)
        else:
            _finalize_simulate()
        return

    # --- Phase 2 + 3: Train into a fresh interim slot via the BG trainer ---
    # The scheduled tick mints a new ``episodic_interim_<stamp>`` adapter and
    # trains the batch into it (or absorbs into the newest existing slot when
    # the cap is reached).  Main adapters are NOT touched here — they only
    # change at the full-cycle boundary, where consolidate_interim_adapters()
    # collapses the accumulated interim slots into episodic / semantic /
    # procedural.  Freshness-wins router order: probing the newest interim
    # before main means recently
    # learned facts surface ahead of the stale main snapshot.
    if not loop.config.indexed_key_replay_enabled:
        logger.warning("Indexed key replay disabled — skipping training")
        session_buffer.mark_consolidated(
            _completed_session_ids(),
            retention_dir=session_retention_dir(loop, config),
        )
        if evict_voice_for_cycle:
            _set_voice_pipeline_profile(_target_profile(), lock_held=False)
        _state["consolidating"] = False
        return

    primary_speaker = speaker_ids[-1] if speaker_ids else ""
    schedule = config.consolidation.refresh_cadence
    max_interim_count = config.consolidation.max_interim_count

    # The BG-trainer worker thread holds the GPU lock for the duration of the
    # callable so concurrent STT/TTS inference abort requests reach it.  A new
    # BackgroundTrainer is constructed per cycle to match the historical
    # pattern; the previous instance (if any) is replaced on _state.
    bt = BackgroundTrainer(
        model=_state["model"],
        tokenizer=_state["tokenizer"],
        training_config=config.training_config,
        output_dir=config.adapter_dir,
        thermal_policy=ThermalPolicy.from_consolidation_config(config.consolidation),
        preload_cache=config.inference.preload_cache,
    )
    _state["background_trainer"] = bt
    if _state.get("consolidation_loop") is not None:
        _state["consolidation_loop"]._bg_trainer = bt

    def _run_interim_training() -> None:
        """Execute the interim training pass + post-train bookkeeping.

        Runs on the BG trainer worker thread under the GPU lock.  The helper
        does its own atomic I5 save of the interim slot + registry; we then
        run the cross-cycle bookkeeping (promotion check, key-metadata save,
        session marking, router reload, state updates) after each cycle.
        """
        try:
            # Callsite 4: post-session async-train.  Runs inside the BG worker
            # thread under ``gpu_lock_sync()`` (acquired by
            # ``_run_callable_queue``).  The surrounding BG-trainer construction
            # and closure pattern are unchanged; only the deleted
            # ``_train_extracted_into_interim`` is replaced with the unified
            # ``run_consolidation_cycle``.  Fire-and-forget semantics are
            # preserved — this is NOT converted to ``_await_bg_cycle``.
            result = loop.run_consolidation_cycle(
                all_episodic_rels,
                all_procedural_rels,
                speaker_id=primary_speaker,
                mode="train",
                run_label=f"tick-{primary_speaker or 'anon'}",
                schedule=schedule,
                max_interim_count=max_interim_count,
            )
        except Exception:
            logger.exception(
                "Scheduled-tick interim training failed — leaving %d sessions pending",
                len(session_ids),
            )
            _state["last_consolidation_result"] = {
                "status": "error",
                "sessions": len(session_ids),
            }
            # Restore voice pipeline even on training failure — we evicted
            # at cycle start, the device should return to its post-startup
            # baseline regardless of whether training succeeded.
            if evict_voice_for_cycle:
                _set_voice_pipeline_profile(_target_profile(), lock_held=True)
            return

        # Pick up any PeftModel handle rebinding from create_interim_adapter.
        _state["model"] = loop.model

        logger.info(
            "Scheduled-tick interim training: mode=%s, adapter=%s, new_keys=%d",
            result.get("mode"),
            result.get("adapter_name"),
            len(result.get("new_keys", [])),
        )

        # Disk I/O — safe from any thread.  Promotion + key-metadata persistence
        # mirror the previous main-write callback; the interim helper already
        # handled the registry / simhash writes atomically.
        try:
            _promote_mature_keys(loop, config)
            _save_key_metadata(loop, config)
            session_buffer.mark_consolidated(
                _completed_session_ids(),
                retention_dir=session_retention_dir(loop, config),
            )
        except Exception:
            logger.exception("Post-interim bookkeeping failed (non-fatal)")

        # Restore voice pipeline on the BG worker thread, before posting
        # _finalize_interim to the event loop. Voice load is multi-second
        # GPU work; running it on the event loop would block asyncio.
        # Symmetric with the eviction at cycle start: paired with the
        # _state["consolidating"] = False signal inside _finalize_interim.
        if evict_voice_for_cycle:
            _set_voice_pipeline_profile(_target_profile(), lock_held=True)

        # State mutations + router reload — post to the event loop so the
        # router cache and inference path see the new slot atomically.
        def _finalize_interim() -> None:
            loop.model.eval()
            _state["last_consolidation"] = datetime.now(timezone.utc).isoformat()
            _state["router"].reload()
            # inference path sees the just-written interim slot (and any
            # tier whose format drifted from the loop's current setting).
            # Count via the indexed_key_registry — it tracks every active key
            # regardless of which adapter (main or interim) currently holds it.
            # The previous main-tier-simhash sum under-reported by the count of
            # keys living in episodic_interim_<stamp> slots between full cycles.
            total_keys = len(loop.store.all_active_keys()) if loop.store.replay_enabled else 0
            _state["last_consolidation_result"] = {
                "status": result.get("mode", "trained"),
                "sessions": len(session_ids),
                "total_keys": total_keys,
                "adapter": result.get("adapter_name"),
            }
            _state["consolidating"] = False
            logger.info(
                "Scheduled-tick complete — adapter=%s, %d total keys",
                result.get("adapter_name"),
                total_keys,
            )

        aio_loop = _state.get("event_loop")
        if aio_loop is not None and aio_loop.is_running():
            aio_loop.call_soon_threadsafe(_finalize_interim)
        else:
            _finalize_interim()

    bt.submit(_run_interim_training, inference_fallback_adapter="episodic")
    logger.info(
        "Extraction done — interim-training job submitted to BG trainer (%d sessions)",
        len(session_ids),
    )


def _run_full_consolidation_sync() -> None:
    """Submit a full-cycle consolidation: collapse interims into main.

    Runs ``loop.consolidate_interim_adapters`` on the BG trainer so the GPU
    lock is held for the entire per-tier rebuild (the helper's docstring
    requires this — calling without the lock raises). The function deletes
    each main adapter, recreates it, trains on the cumulative keyed-pair
    set derived from every interim slot's keys, and on success unloads the
    interim adapters and reloads the router. On a failed recall-sanity
    check it rolls back to the snapshot and aborts that tier.

    After a successful merge the main adapter manifests are stamped with
    the current registry hash, restoring boot-time mount-ability that
    interim cycles otherwise drift away from.
    """
    from paramem.server.background_trainer import BackgroundTrainer
    from paramem.server.consolidation import (
        _save_key_metadata,
        create_consolidation_loop,
    )

    config = _state["config"]
    session_buffer = _state.get("session_buffer")

    loop = _state.get("consolidation_loop")
    if loop is None:
        loop = create_consolidation_loop(
            _state["model"],
            _state["tokenizer"],
            config,
            _state["memory_store"],
            state_provider=lambda: _state,
        )
        _state["consolidation_loop"] = loop
        _state["model"] = loop.model

    bt = BackgroundTrainer(
        model=_state["model"],
        tokenizer=_state["tokenizer"],
        training_config=config.training_config,
        output_dir=config.adapter_dir,
        thermal_policy=ThermalPolicy.from_consolidation_config(config.consolidation),
        preload_cache=config.inference.preload_cache,
    )
    _state["background_trainer"] = bt
    _state["consolidation_loop"]._bg_trainer = bt

    def _run_full_cycle() -> None:
        """Run on the BG trainer worker thread under the GPU lock."""
        # No voice eviction here. The full cycle collapses existing
        # interim slots (already-trained keys) into main; it does not
        # re-run the extraction chain that drives the KV-cache spikes
        # the eviction was meant to mitigate. Voice eviction is scoped
        # to the extraction path in _extract_and_start_training, and
        # only when the pending batch is document-only.
        #
        # Mode dispatch: simulate mode has no PEFT interim adapters —
        # calling consolidate_interim_adapters would trigger
        # delete_adapter / create_adapter on a non-existent slot.
        # Simulate mode uses consolidate_interim_graphs instead, which
        # merges the per-cycle graph.json sidecars into the canonical
        # main-tier graph without touching PEFT weights.
        _mode = config.consolidation.mode
        try:
            if _mode == "simulate":
                result = loop.consolidate_interim_graphs()
            else:
                result = loop.consolidate_interim_adapters(
                    trainer=bt,
                    router=_state.get("router"),
                )
        except AbortedDuringConsolidation as exc:
            logger.info("Full consolidation aborted for inference: %s", exc)
            _state["last_consolidation_result"] = {"status": "aborted"}
            _state["consolidating"] = False
            return
        except Exception:
            logger.exception("Full consolidation failed")
            _state["last_consolidation_result"] = {"status": "error"}
            _state["consolidating"] = False
            return

        # Pick up any PeftModel wrapper rebinding done by the per-tier
        # delete_adapter / create_adapter cycles inside consolidate_interim_adapters.
        _state["model"] = loop.model

        logger.info(
            "Full cycle complete — tiers_rebuilt=%s, drift=%d, rolled_back=%s",
            result.get("tiers_rebuilt"),
            result.get("graph_drift_count", 0),
            result.get("rolled_back"),
        )

        # Layering boundary. ``consolidate_interim_adapters`` already
        # finished its internal finalize on its way out (registry rewrite,
        # WEIGHT-LEVEL persist+verify of the rebuilt main slots, interim
        # purge, router reload — see its Step 6 block) regardless of whether
        # anything was rebuilt. The merged main weights are now durably saved
        # BEFORE the interim slots are deleted, inside the method, so there is
        # no crash window where the folded knowledge has no on-disk copy.
        # The post-cycle work below (key-metadata persistence, session marking,
        # ``_finalize_full``) is bookkeeping whose precondition is
        # ``tiers_rebuilt != []``. Calling it on the no-op outcome violates that
        # precondition: key-metadata persistence has nothing new and
        # ``_finalize_full`` would double the router reload that the inner
        # finalize already did.
        #
        # Honor the precondition at the orchestration layer (here) rather
        # than retrofitting each downstream helper with no-op tolerance.
        # Clear the consolidating flag, record the no-op as a successful
        # cycle outcome, and return.
        if not result.get("tiers_rebuilt"):
            _state["last_consolidation"] = datetime.now(timezone.utc).isoformat()
            _state["last_consolidation_result"] = {
                "status": "noop",
                "tiers_rebuilt": [],
                "graph_drift_count": result.get("graph_drift_count", 0),
            }
            _state["consolidating"] = False
            logger.info(
                "Full cycle no-op — nothing to rebuild, inner finalize already "
                "ran inside consolidate_interim_adapters; consolidating flag cleared"
            )
            return

        # The rebuilt main slots were already persisted+verified to disk inside
        # consolidate_interim_adapters' finalize (between the registry rewrite
        # and the interim purge), so the on-disk main meta.json carries a fresh
        # window_stamp + registry_sha256 and the slots remount on restart. If
        # that persist/verify had failed, the method would have raised and we
        # would have landed in the ``except Exception`` handler above with
        # sessions left pending — so reaching here means main is durable.

        # Persist key metadata and mark any pending sessions consolidated —
        # the merge folded their facts into main, so they should not be
        # re-extracted on the next tick.
        try:
            _save_key_metadata(loop, config)
            if session_buffer is not None:
                pending_ids = [s["session_id"] for s in session_buffer.get_pending()]
                if pending_ids:
                    from paramem.server.consolidation import (
                        session_retention_dir as _retdir,
                    )

                    session_buffer.mark_consolidated(
                        pending_ids,
                        retention_dir=_retdir(loop, config),
                    )
        except Exception:
            logger.exception("Post-full-cycle bookkeeping failed (non-fatal)")

        def _finalize_full() -> None:
            loop.model.eval()
            _state["last_consolidation"] = datetime.now(timezone.utc).isoformat()
            _state["router"].reload()
            # Re-run the per-tier manifest validator now that main slots have
            # been re-saved with a fresh registry hash + window_stamp. Without
            # this, /status and pstatus keep showing the boot-time snapshot —
            # operators see "FINGERPRINT MISMATCH … PA routing DISABLED" red
            # rows even though main is healthy.
            _revalidate_main_adapter_manifests(_state)
            # were just re-saved with a fresh registry hash + format; the
            # manifest is authoritative.  Also drops entries for retired
            # interim slots so the inference path stops probing them.
            total_keys = len(loop.store.all_active_keys()) if loop.store.replay_enabled else 0
            _state["last_consolidation_result"] = {
                "status": "rolled_back" if result.get("rolled_back") else "full_trained",
                "tiers_rebuilt": result.get("tiers_rebuilt", []),
                "rollback_tier": result.get("rollback_tier"),
                "graph_drift_count": result.get("graph_drift_count", 0),
                "total_keys": total_keys,
            }
            _state["consolidating"] = False
            logger.info("Full cycle bookkeeping complete — %d total keys", total_keys)

        aio_loop = _state.get("event_loop")
        if aio_loop is not None and aio_loop.is_running():
            aio_loop.call_soon_threadsafe(_finalize_full)
        else:
            _finalize_full()

    bt.submit(_run_full_cycle, inference_fallback_adapter="episodic")
    logger.info("Full consolidation submitted to BG trainer")


def _arm_active_store_migration(config) -> bool:
    """Arm a pending active-store rebuild when the live ``consolidation.mode``
    diverges from the on-disk active store.

    Single source for the mode-switch arming used by BOTH the lifespan startup
    path AND the live config-reload path (``_live_reload_base_model``).  When a
    divergence is detected it sets ``pending_rehydration`` + ``effective_mode``
    (the source mode, so inference falls back to it) and persists the migration
    state so the next ``/consolidate`` runs the tier-by-tier rebuild under the
    1.0 recall gate.  The rebuild itself is NOT run here — this is GPU-free and
    read-only w.r.t. the model.

    ``detect_mode_switch`` returns ``None`` when there is no divergence, so this
    is a safe no-op for every non-mode config change.

    Returns ``True`` when a migration was armed, ``False`` otherwise.
    """
    from paramem.server.active_store_migration import detect_mode_switch, save_state

    migration_state = detect_mode_switch(config)
    if migration_state is not None:
        # Integrity gate: refuse arming when the store is corrupt.
        from paramem.backup.integrity import verify_infrastructure_integrity as _verify_integrity

        _daily_ok_arm = _state.get("daily_loadable", False)
        _arm_integrity = _verify_integrity(
            config,
            store=_state.get("memory_store"),
            daily_loadable=_daily_ok_arm,
        )
        if not _arm_integrity.ok:
            logger.error(
                "_arm_active_store_migration: integrity check failed (%d failures) — "
                "not arming migration; resolve integrity failures first.",
                len(_arm_integrity.failures),
            )
            for _fc in _arm_integrity.failures:
                logger.error("  [%s/%s] %s: %s", _fc.category, _fc.tier, _fc.path, _fc.detail)
            _state["pending_rehydration"] = False
            _state["effective_mode"] = config.consolidation.mode
            return False

        _state["pending_rehydration"] = True
        _state["effective_mode"] = migration_state.source_mode
        # Persist so the next /consolidate's load_state finds it and runs the
        # rebuild. Idempotent when an in-flight state file already existed.
        save_state(config.adapter_dir, migration_state)
        logger.warning(
            "Active-store migration pending: %s; falling back to %s mode for "
            "inference until tier-by-tier rebuild completes "
            "(completed_tiers=%s, failed_tiers=%s). Trigger via /consolidate.",
            migration_state.direction,
            migration_state.source_mode,
            migration_state.completed_tiers or "[]",
            list(migration_state.failed_tiers.keys()) or "[]",
        )
        return True
    _state["pending_rehydration"] = False
    _state["effective_mode"] = config.consolidation.mode
    return False


def _run_active_store_migration_sync() -> None:
    """Execute the pending active-store migration on a worker thread.

    Triggered by ``_maybe_trigger_scheduled_consolidation`` when
    ``_state["pending_rehydration"]`` is True — meaning startup detection
    (or an interrupted prior migration) saw a divergence between the
    operator's yaml ``consolidation.mode`` and the on-disk active store.

    Per-tier execution lives in ``active_store_migration.migrate``. The
    train direction holds the GPU lock (training + recall probe drive the
    model). The simulate direction is disk-only but we hold the lock anyway
    for symmetry with the full-cycle code path. On all-tiers-done the
    state file is removed and ``_state["pending_rehydration"]`` is cleared
    via ``_finalize_migration``; otherwise the operator can re-trigger to
    retry remaining tiers.
    """
    from paramem.server.active_store_migration import (
        load_state,
        migrate,
    )
    from paramem.server.background_trainer import BackgroundTrainer
    from paramem.server.consolidation import create_consolidation_loop

    config = _state["config"]
    state = load_state(config.adapter_dir)
    if state is None:
        # Nothing to do — flag was set but state file is gone (already
        # completed by another caller). Clear pending and return.
        _state["pending_rehydration"] = False
        _state["effective_mode"] = config.consolidation.mode
        _state["consolidating"] = False
        logger.info("Active-store migration: state file absent — clearing pending flag")
        return

    loop = _state.get("consolidation_loop")
    if loop is None:
        loop = create_consolidation_loop(
            _state["model"],
            _state["tokenizer"],
            config,
            _state["memory_store"],
            state_provider=lambda: _state,
        )
        _state["consolidation_loop"] = loop
        _state["model"] = loop.model

    bt = BackgroundTrainer(
        model=_state["model"],
        tokenizer=_state["tokenizer"],
        training_config=config.training_config,
        output_dir=config.adapter_dir,
        thermal_policy=ThermalPolicy.from_consolidation_config(config.consolidation),
        preload_cache=config.inference.preload_cache,
    )
    _state["background_trainer"] = bt
    _state["consolidation_loop"]._bg_trainer = bt

    def _run_migration_on_worker() -> None:
        """Run on the BG-trainer worker thread under the GPU lock."""
        try:
            updated = migrate(loop, config, state)
        except Exception:
            logger.exception("Active-store migration raised — leaving state file on disk")
            _state["last_consolidation_result"] = {"status": "migration_error"}
            _state["consolidating"] = False
            return

        # Pick up any PEFT rebinding inside migrate's create_adapter calls.
        _state["model"] = loop.model

        logger.info(
            "Active-store migration done: direction=%s completed=%s failed=%s",
            updated.direction,
            updated.completed_tiers,
            list(updated.failed_tiers.keys()),
        )

        def _finalize_migration() -> None:
            loop.model.eval()
            _state["last_consolidation"] = datetime.now(timezone.utc).isoformat()
            _state["last_consolidation_result"] = {
                "status": (
                    "migration_complete"
                    if updated.all_tiers_done(loop.store.tiers_with_registry())
                    else "migration_partial"
                ),
                "direction": updated.direction,
                "completed_tiers": list(updated.completed_tiers),
                "failed_tiers": dict(updated.failed_tiers),
            }
            if updated.all_tiers_done(loop.store.tiers_with_registry()):
                _state["pending_rehydration"] = False
                _state["effective_mode"] = config.consolidation.mode
                logger.info(
                    "Active-store migration complete; effective_mode=%s",
                    config.consolidation.mode,
                )
            # Partial completion: pending_rehydration stays True so a re-trigger
            # picks up remaining tiers. effective_mode stays at source_mode.
            # peft_config alone is not a safe key set: a ``train → simulate``
            # migration unloads main adapters, so iterating it misses them and
            # the inference path sees a stale snapshot from boot.  Reading
            # from disk also drops entries for slots the migration deleted.
            _state["consolidating"] = False

        aio_loop = _state.get("event_loop")
        if aio_loop is not None and aio_loop.is_running():
            aio_loop.call_soon_threadsafe(_finalize_migration)
        else:
            _finalize_migration()

    bt.submit(_run_migration_on_worker, inference_fallback_adapter="episodic")
    logger.info("Active-store migration submitted to BG trainer")


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

    1. If any non-server GPU compute process is running → wait. (This is a
       cheap, context-free early-out; the authoritative who-agnostic budget
       check is the live-budget pre-flight inside
       :func:`_live_reload_base_model`, which sees consumers in other WSL
       distros / the Windows host that the compute-app list cannot.)
    2. Else inspect the hold state (PARAMEM_EXTRA_ARGS in systemd --user env):
       - Hold cleared → reclaim: reload model in-process (no restart).
         On transient failure: log WARN, record ``last_reclaim_error``,
         continue loop; retry on next tick.
       - Hold set, holder PID alive → legitimate mid-training window
         (cooldown, model swap) → keep polling, do not reclaim.
       - Hold set, holder PID dead or unregistered → orphan suspected →
         emit one WARN and exit the loop.  Operator clears via
         ``pstatus --acquire`` (POST /gpu/acquire).

    Exiting on orphan stops the infinite restart loop that was happening
    when a SIGKILLed test left PARAMEM_EXTRA_ARGS=--defer-model behind:
    visibility over silent auto-heal, per design.

    On transient reclaim failures the loop retries every ``interval_minutes``
    minutes without restarting; ``last_reclaim_error`` on ``/status``
    tracks attempt count and last error message.
    """
    loop = asyncio.get_event_loop()
    interval_seconds = interval_minutes * 60
    while True:
        await asyncio.sleep(interval_seconds)
        # The GPU may have been reclaimed externally during the sleep (operator
        # /gpu/acquire, a config apply, or a base-swap reload).  The loop exists
        # only to reclaim a cloud-only server — if we are already local, our job
        # is done; reclaiming again would release+reload an already-loaded model
        # (a needless ~10 s cloud-only churn window).  Exit cleanly.
        if _state.get("mode") == "local":
            logger.info("Auto-reclaim: already local (reclaimed externally) — stopping loop")
            return
        if _gpu_has_compute_processes():
            logger.debug("Auto-reclaim: GPU still occupied, waiting")
            continue
        hold = _get_hold_state()
        if not hold["hold_active"]:
            # Hold cleared — reload in-process; no service restart.
            try:
                from paramem.server.gpu_lock import gpu_lock

                # lock_held=True: gpu_lock() holds the non-reentrant threading.Lock
                # across run_in_executor; the primitive's internal
                # _set_voice_pipeline_profile calls must not re-acquire it.
                # Use a lambda to bind lock_held so run_in_executor (which takes
                # positional args only) delivers the keyword argument correctly.
                async with gpu_lock():
                    await loop.run_in_executor(
                        None,
                        lambda: _live_reload_base_model(lock_held=True),
                    )
                if _state.get("mode") != "local":
                    # Reload was declined (insufficient free VRAM) or failed
                    # and self-cleaned — the base model is NOT loaded. Do not
                    # load the STT/TTS GPU pair: that is exactly how a
                    # cloud-only server ends up squatting VRAM. Force voice
                    # back to CPU (outside the gpu_lock) and keep polling; the
                    # GPU may free on a later tick.
                    await loop.run_in_executor(None, _set_voice_pipeline_profile, "cpu")
                    logger.info(
                        "Auto-reclaim: reload deferred (mode=%s, reason=%s) — retrying next tick",
                        _state.get("mode"),
                        _state.get("cloud_only_reason"),
                    )
                    continue
                # Voice drain+restore is now owned by _live_reload_base_model
                # (partial-path success restore runs inside the primitive).
                _state["last_reclaim_error"] = None
                logger.info("Auto-reclaim: GPU reclaimed in-process")
                return
            except Exception as exc:  # noqa: BLE001
                attempt_count = (_state.get("last_reclaim_error") or {}).get("attempt_count", 0) + 1
                _state["last_reclaim_error"] = {
                    "at": datetime.now(timezone.utc).isoformat(),
                    "error": str(exc),
                    "attempt_count": attempt_count,
                }
                logger.warning(
                    "Auto-reclaim: in-process reclaim failed (attempt %d): %s"
                    " — will retry on next tick",
                    attempt_count,
                    exc,
                    exc_info=True,
                )
                continue
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
                "`pstatus --acquire` (POST /gpu/acquire).",
                owner_pid,
            )
        else:
            logger.warning(
                "Auto-reclaim: PARAMEM_EXTRA_ARGS=--defer-model still set but no "
                "holder PID registered — suspected orphan. Clear and reclaim with "
                "`pstatus --acquire` (POST /gpu/acquire)."
            )
        return


def _restart_service():
    """Restart the systemd service for a clean process.

    No longer called from auto-reclaim (D3: auto-reclaim now uses in-process
    reload via ``_live_reload_base_model`` + ``_set_voice_pipeline_profile``).
    Still used by :func:`gpu_acquire` as a fallback when in-process reload
    fails, and is kept defined for future emergency use. Process-level death
    is handled orthogonally by ``Restart=on-failure`` + ``RestartSec=30`` in
    the systemd unit.
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
